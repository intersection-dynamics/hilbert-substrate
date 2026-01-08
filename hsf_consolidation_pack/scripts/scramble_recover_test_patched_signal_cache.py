"""
scramble_recover_test.py (v3)
============================

Two layers, cleanly separated:

1) RECOVERY (can be geometry-blind)
   - Scramble a known local Hamiltonian (as a generator of test instances)
   - Recover a local-ish basis using:
       STROBE (discrete 2-qubit basis moves; Metropolis)
       FLOW   (isospectral descent using truncated Pauli-weight locality)

   Strobe objective:
     --strobe-objective sparse : geometry-blind sparsification of pair couplings (no distances)
     --strobe-objective range  : probe-geometry range leakage (uses ring distances)

2) MEASUREMENT (optional probe + geometry-blind emergent graph)
   - Pair coupling strengths matrix S_ij (geometry-blind)
   - Infer interaction graph from S_ij (topK / threshold)
   - Optional ring-probe V(d) ruler (diagnostic only)
   - Fermion audit (Part III):
       A) Sector additivity test (free-fermion fingerprint; best on XX)
       B) Jordan–Wigner anticommutator test on ground state
       C) “Pauli pressure” curvature from sector ground energies

CPU parallel sweeps:
  Use processes across seeds:
    --jobs 8 --blas-threads 1

Windows: one-liner commands.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh, expm


# =============================================================================
# Thread control (critical when using --jobs > 1)
# =============================================================================

def set_thread_env(threads: int = 1) -> None:
    n = str(int(threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n


# =============================================================================
# Pauli utilities
# =============================================================================

def dense_pauli() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    # basis: |0> = up, |1> = down
    sig_minus = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0><1|
    sig_plus  = np.array([[0, 0], [1, 0]], dtype=np.complex128)  # |1><0|
    return I, X, Y, Z, sig_minus, sig_plus


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def hermitianize(H: np.ndarray) -> np.ndarray:
    return 0.5 * (H + H.conj().T)


# =============================================================================
# Generator Hamiltonians (test instances; not “fundamental geometry”)
# =============================================================================

def spin_ring_dense(N: int, model: str = "xxx", J: float = 1.0, Delta: float = 1.0) -> np.ndarray:
    """
    model:
      - "xx"  : X_i X_{i+1} + Y_i Y_{i+1}   (free fermions under Jordan–Wigner)
      - "xxz" : XX + YY + Delta * ZZ
      - "xxx" : XX + YY + ZZ  (Delta=1)
    """
    model = model.lower()
    I, X, Y, Z, _, _ = dense_pauli()
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)

    if model == "xxx":
        Delta = 1.0
    elif model == "xx":
        Delta = 0.0
    elif model == "xxz":
        pass
    else:
        raise ValueError(f"Unknown model: {model}")

    for i in range(N):
        j = (i + 1) % N
        # XX + YY
        for P in (X, Y):
            ops = [I] * N
            ops[i] = P
            ops[j] = P
            H += J * kron_n(ops)
        # Delta * ZZ
        if abs(Delta) > 0:
            ops = [I] * N
            ops[i] = Z
            ops[j] = Z
            H += (J * Delta) * kron_n(ops)

    return hermitianize(H)


# =============================================================================
# Probe graphs / distances (measurement layer)
# =============================================================================

def ring_edges(N: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % N) for i in range(N)]


def all_edges(N: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            out.append((i, j))
    return out


def ring_distances(N: int) -> np.ndarray:
    D = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            D[i, j] = min(d, N - d)
    return D


# =============================================================================
# Scrambling (local/global)
# =============================================================================

def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    Z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.where(np.abs(d) > 0, np.abs(d), 1.0)
    Q = Q * ph
    return Q


def random_su2(rng: np.random.Generator) -> np.ndarray:
    Z = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.where(np.abs(d) > 0, np.abs(d), 1.0)
    U = Q * ph
    det = np.linalg.det(U)
    U = U / det ** 0.5
    return U


def build_local_scrambler(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = random_su2(rng)
    for _ in range(N - 1):
        U = np.kron(U, random_su2(rng))
    return U


def build_global_scrambler(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return random_unitary(2 ** N, rng)


def scramble_hamiltonian(H: np.ndarray, U: np.ndarray) -> np.ndarray:
    return hermitianize(U @ H @ U.conj().T)


# =============================================================================
# Geometry-blind pair coupling extraction
#   n_ij = || Tr_rest(H) ||_F for 2-qubit reduced operator on (i,j)
# =============================================================================

def two_qubit_reduced_operator(H: np.ndarray, N: int, q1: int, q2: int) -> np.ndarray:
    if q1 == q2:
        raise ValueError("q1==q2")
    a, b = (q1, q2) if q1 < q2 else (q2, q1)

    Ht = H.reshape([2] * N + [2] * N)

    keep = [a, b]
    trace_sites = [i for i in range(N) if i not in keep]

    row_order = keep + trace_sites
    col_order = keep + trace_sites
    perm = row_order + [N + i for i in col_order]
    Hp = np.transpose(Ht, axes=perm)

    rest = 2 ** (N - 2)
    Hp = Hp.reshape(4, rest, 4, rest)

    H_red = np.einsum("arbr->ab", Hp)
    return H_red


def pair_strengths_matrix(H: np.ndarray, N: int) -> np.ndarray:
    S = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            Hred = two_qubit_reduced_operator(H, N, i, j)
            n = float(np.linalg.norm(Hred, ord="fro"))
            S[i, j] = n
            S[j, i] = n
    return S


# =============================================================================
# STROBE objectives
# =============================================================================

def objective_range_leakage(H: np.ndarray, N: int, D: np.ndarray, power: int = 2) -> float:
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = int(D[i, j])
            if d <= 1:
                continue
            Hred = two_qubit_reduced_operator(H, N, i, j)
            w = float(d ** power)
            total += w * float(np.linalg.norm(Hred, ord="fro") ** 2)
    return total


def objective_sparse_ratio(H: np.ndarray, N: int, eps: float = 1e-12) -> float:
    S = pair_strengths_matrix(H, N)
    triu = S[np.triu_indices(N, k=1)]
    num = float(np.sum(triu))
    den = float(np.sqrt(np.sum(triu * triu) + eps))
    return num / den



def objective_signal_entropy(H: np.ndarray, N: int, eps: float = 1e-12) -> float:
    """Geometry-blind signaling-dispersion proxy.

    Interpret S_ij = ||Tr_rest(H)||_F on (i,j) as a crude upper bound on
    instantaneous influence between sites i and j. For each site i we form:

        p_j ∝ S_ij   (j != i)
        H_i = - Σ_j p_j log p_j

    and return mean_i( H_i / log(N-1) ) in [0,1].

    Lower means influence is concentrated to a few partners (more local-like).
    Higher means influence is dispersed broadly (more delocalized).

    This uses no geometric distances or target adjacency.
    """
    if N <= 2:
        return 0.0
    S = pair_strengths_matrix(H, N)
    denom = math.log(float(N - 1))
    ent_sum = 0.0
    for i in range(N):
        row = np.array([S[i, j] for j in range(N) if j != i], dtype=np.float64) + float(eps)
        Z = float(np.sum(row))
        if Z <= 0:
            continue
        p = row / Z
        ent = -float(np.sum(p * np.log(p)))
        ent_sum += ent / denom
    return ent_sum / float(N)


def topk_strength_share(S: np.ndarray, topk: int) -> float:
    N = S.shape[0]
    vals = []
    for i in range(N):
        for j in range(i + 1, N):
            vals.append(float(S[i, j]))
    vals = np.array(vals, dtype=np.float64)
    total = float(np.sum(vals)) + 1e-18
    if topk <= 0:
        return 0.0
    top = float(np.sum(np.sort(vals)[::-1][: min(topk, vals.size)]))
    return top / total


# =============================================================================
# Two-qubit conjugation of H without constructing full 2^N gate
# =============================================================================

def apply_two_qubit_conjugation_to_operator(H: np.ndarray, N: int, q1: int, q2: int, U2: np.ndarray) -> np.ndarray:
    if q1 == q2:
        return H
    a, b = (q1, q2) if q1 < q2 else (q2, q1)
    dim = 2 ** N
    Ht = H.reshape([2] * N + [2] * N)

    row_order = [a, b] + [i for i in range(N) if i not in (a, b)]
    col_order = [a, b] + [i for i in range(N) if i not in (a, b)]
    perm = row_order + [N + i for i in range(N) if i not in (a, b)]
    perm = row_order + [N + i for i in col_order]
    Hperm = np.transpose(Ht, axes=perm)

    rest_dim = 2 ** (N - 2)
    Hperm = Hperm.reshape(4, rest_dim, 4, rest_dim)

    U = U2
    Ud = U2.conj().T

    tmp = np.tensordot(U, Hperm, axes=([1], [0]))
    out = np.tensordot(tmp, Ud, axes=([2], [0]))

    out = out.reshape([2, 2] + [2] * (N - 2) + [2, 2] + [2] * (N - 2))
    inv = np.argsort(perm)
    out = np.transpose(out, axes=inv).reshape(dim, dim)
    return hermitianize(out)


def random_small_two_qubit_gate(rng: np.random.Generator, eps: float) -> np.ndarray:
    X = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2.0)
    A = hermitianize(X)
    return expm(1j * eps * A)


# =============================================================================
# FLOW objective: Pauli-weight locality (continuous)
# =============================================================================

@dataclass(frozen=True)
class PauliWord:
    indices: Tuple[int, ...]
    weight: int


def iter_pauli_words(N: int, max_weight: Optional[int]) -> Iterable[PauliWord]:
    idx = [0] * N

    def rec(pos: int, w: int):
        if pos == N:
            yield PauliWord(tuple(idx), w)
            return
        idx[pos] = 0
        yield from rec(pos + 1, w)
        if max_weight is None or w < max_weight:
            for v in (1, 2, 3):
                idx[pos] = v
                yield from rec(pos + 1, w + 1)

    yield from rec(0, 0)


class PauliKronWorkspace:
    """Reusable buffers to build N-qubit Pauli Kronecker products with low allocation.

    Why this exists:
      The naive implementation `kron_n([mats[i] ...])` allocates many intermediate
      arrays (one per Kronecker step) *for every Pauli word*, which can lead to
      large transient RSS spikes and allocator churn during FLOW.

    This workspace builds the full (2**N x 2**N) matrix using two reusable
    buffers (`buf`, `tmp`) and in-place slice filling, so per-word allocation
    is avoided inside tight loops.
    """

    def __init__(self, N: int):
        self.N = int(N)
        I, X, Y, Z, _, _ = dense_pauli()
        self.mats = [I, X, Y, Z]
        dim = 2 ** self.N
        self.buf = np.empty((dim, dim), dtype=np.complex128)
        self.tmp = np.empty((dim, dim), dtype=np.complex128)

    def build(self, indices: Tuple[int, ...]) -> np.ndarray:
        """Return a view of the internal buffer containing ⊗_k mats[indices[k]]."""
        if len(indices) != self.N:
            raise ValueError(f"Pauli word length {len(indices)} != N {self.N}")

        # Start as 1x1 identity in the top-left corner of buf.
        self.buf.fill(0.0)
        self.buf[0, 0] = 1.0 + 0.0j
        cur_dim = 1

        for idx in indices:
            M = self.mats[int(idx)]
            next_dim = cur_dim * 2

            # Build kron(current, M) into tmp using in-place slice ops:
            # tmp[2a,2b] = current[a,b] * M[0,0], etc.
            # Use np.multiply(..., out=...) to avoid temporaries.
            self.tmp[:next_dim, :next_dim] = 0.0

            cur = self.buf[:cur_dim, :cur_dim]
            np.multiply(cur, M[0, 0], out=self.tmp[0:next_dim:2, 0:next_dim:2])
            np.multiply(cur, M[0, 1], out=self.tmp[0:next_dim:2, 1:next_dim:2])
            np.multiply(cur, M[1, 0], out=self.tmp[1:next_dim:2, 0:next_dim:2])
            np.multiply(cur, M[1, 1], out=self.tmp[1:next_dim:2, 1:next_dim:2])

            # Swap buffers: tmp -> buf
            self.buf[:next_dim, :next_dim] = self.tmp[:next_dim, :next_dim]
            cur_dim = next_dim

        return self.buf


def pauli_matrix_from_word(word: PauliWord, ws: Optional[PauliKronWorkspace] = None) -> np.ndarray:
    """Build the dense Pauli matrix for a word.

    For performance-sensitive loops, pass a PauliKronWorkspace to reuse buffers.
    If ws is None, this allocates a workspace internally (slower, higher churn).
    """
    if ws is None:
        ws = PauliKronWorkspace(len(word.indices))
    return ws.build(word.indices)


def locality_cost_and_total_sq(H: np.ndarray, N: int, p: int, max_weight: Optional[int], eps: float = 1e-18) -> Tuple[float, float]:
    dim = 2 ** N
    num = 0.0
    den = 0.0

    # Reuse a workspace to avoid allocating Pauli matrices repeatedly.
    ws = PauliKronWorkspace(N)

    for w in iter_pauli_words(N, max_weight=max_weight):
        P = pauli_matrix_from_word(w, ws=ws)
        tr = float(np.real(np.einsum("ij,ji->", H, P)))
        c = tr / dim
        c2 = c * c
        den += c2
        num += (w.weight ** p) * c2

    den += eps
    return float(num / den), float(den)




def locality_gradient(H: np.ndarray, N: int, p: int, max_weight: Optional[int], denom_eps: float = 1e-18) -> np.ndarray:
    dim = 2 ** N
    C, denom = locality_cost_and_total_sq(H, N, p=p, max_weight=max_weight, eps=denom_eps)
    inv = 1.0 / denom

    M = np.zeros_like(H, dtype=np.complex128)

    # Reuse a workspace and do in-place scaling to reduce temporaries.
    ws = PauliKronWorkspace(N)
    for w in iter_pauli_words(N, max_weight=max_weight):
        P = pauli_matrix_from_word(w, ws=ws)
        tr = float(np.real(np.einsum("ij,ji->", H, P)))
        c = tr / dim
        deriv = 2.0 * c * ((w.weight ** p) - C) * inv

        # M += (deriv / dim) * P  without allocating (deriv*P) temp
        alpha = (deriv / dim)
        # scale P buffer in-place then add, then restore not needed (buffer reused)
        np.multiply(P, alpha, out=P)
        M += P

    return hermitianize(M)



    M = np.zeros_like(H, dtype=np.complex128)
    for w in iter_pauli_words(N, max_weight=max_weight):
        P = pauli_matrix_from_word(w)
        tr = float(np.real(np.einsum("ij,ji->", H, P)))
        c = tr / dim
        deriv = 2.0 * c * ((w.weight ** p) - C) * inv
        M += (deriv / dim) * P

    return hermitianize(M)


def isospectral_flow_step(H: np.ndarray, N: int, dt: float, p: int, max_weight: Optional[int]) -> np.ndarray:
    M = locality_gradient(H, N, p=p, max_weight=max_weight)
    G = H @ M - M @ H
    G = 0.5 * (G - G.conj().T)  # skew-Hermitian
    U = expm(+dt * G)
    return hermitianize(U @ H @ U.conj().T)


# =============================================================================
# Probe measurement: V(d) on ring (diagnostic ruler only)
# =============================================================================

def measure_V_vs_d_ring(H: np.ndarray, N: int) -> Dict[int, float]:
    I, X, _, _, _, _ = dense_pauli()

    evals, evecs = eigh(H)
    ground = evecs[:, 0]
    E0 = float(np.real(evals[0]))

    X_ops: List[np.ndarray] = []
    single_E: List[float] = []

    for site in range(N):
        ops = [I] * N
        ops[site] = X
        Xi = kron_n(ops)
        X_ops.append(Xi)
        psi = Xi @ ground
        psi /= np.linalg.norm(psi)
        Ei = float(np.real(psi.conj() @ (H @ psi)))
        single_E.append(Ei)

    D = ring_distances(N)
    buckets: Dict[int, List[float]] = {}

    for i in range(N):
        for j in range(i + 1, N):
            d = int(D[i, j])
            psi = X_ops[j] @ (X_ops[i] @ ground)
            psi /= np.linalg.norm(psi)
            Eij = float(np.real(psi.conj() @ (H @ psi)))
            V = Eij - single_E[i] - single_E[j] + E0
            buckets.setdefault(d, []).append(V)

    return {d: float(np.mean(vs)) for d, vs in buckets.items()}


# =============================================================================
# Emergent graph inference + diagnostics
# =============================================================================

def infer_graph_from_strengths(S: np.ndarray, mode: str) -> List[Tuple[int, int, float]]:
    N = S.shape[0]
    pairs: List[Tuple[int, int, float]] = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j, float(S[i, j])))

    if mode == "none":
        return []

    if mode.startswith("topk:"):
        K = int(mode.split(":", 1)[1])
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[: max(0, min(K, len(pairs)))]

    if mode.startswith("threshold:"):
        T = float(mode.split(":", 1)[1])
        return [(i, j, w) for (i, j, w) in pairs if w >= T]

    raise ValueError(f"Unknown --infer-graph mode: {mode}")


def graph_diagnostics(N: int, edges: List[Tuple[int, int]]) -> Dict[str, Any]:
    deg = [0] * N
    adj = [[] for _ in range(N)]
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
        adj[i].append(j)
        adj[j].append(i)

    seen = [False] * N
    comps = 0
    sizes: List[int] = []
    for s in range(N):
        if seen[s]:
            continue
        comps += 1
        q = [s]
        seen[s] = True
        cnt = 0
        while q:
            u = q.pop()
            cnt += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        sizes.append(cnt)

    return {
        "num_edges": int(len(edges)),
        "degree_min": int(min(deg)) if deg else 0,
        "degree_max": int(max(deg)) if deg else 0,
        "degree_mean": float(np.mean(deg)) if deg else 0.0,
        "components": int(comps),
        "component_sizes": sizes,
    }


# =============================================================================
# Fermion audit (Part III)
# =============================================================================

def basis_indices_fixed_n(N: int, n_down: int) -> np.ndarray:
    """Computational basis indices with popcount == n_down (|1>=down)."""
    dim = 2 ** N
    idx = []
    for s in range(dim):
        if int(s).bit_count() == n_down:
            idx.append(s)
    return np.array(idx, dtype=np.int64)


def sector_submatrix(H: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return H[np.ix_(idx, idx)]


def fermion_sector_additivity_audit(H: np.ndarray, N: int, max_n: int = 3) -> Dict[str, Any]:
    """
    Free-fermion fingerprint:
      - Diagonalize n=1 sector -> eps_k
      - Check that n=2 energies are ~ sums eps_a + eps_b (a<b)
      - Optionally check n=3 similarly
    """
    out: Dict[str, Any] = {"max_n": int(max_n)}

    # n=0 ground energy (reference)
    idx0 = basis_indices_fixed_n(N, 0)
    H0 = sector_submatrix(H, idx0)
    e0 = float(np.real(eigh(H0, eigvals_only=True)[0]))
    out["E0_n0"] = e0

    # n=1
    idx1 = basis_indices_fixed_n(N, 1)
    H1 = sector_submatrix(H, idx1)
    e1 = np.real(eigh(H1, eigvals_only=True))
    eps = e1 - e0  # shift by n=0 ground
    out["eps_n1"] = [float(x) for x in eps.tolist()]

    def best_rms_for_n(n: int) -> Tuple[float, float]:
        idxn = basis_indices_fixed_n(N, n)
        Hn = sector_submatrix(H, idxn)
        en = np.real(eigh(Hn, eigvals_only=True))
        # shift by n=0 ground
        En = en - e0

        # generate all distinct sums of n one-particle energies (no repeated modes)
        # For N=8 and n<=3 this is cheap.
        eps_list = eps.tolist()
        sums: List[float] = []

        if n == 2:
            for a in range(len(eps_list)):
                for b in range(a + 1, len(eps_list)):
                    sums.append(eps_list[a] + eps_list[b])
        elif n == 3:
            for a in range(len(eps_list)):
                for b in range(a + 1, len(eps_list)):
                    for c in range(b + 1, len(eps_list)):
                        sums.append(eps_list[a] + eps_list[b] + eps_list[c])
        else:
            raise ValueError("n must be 2 or 3 here")

        sums = np.array(sorted(sums), dtype=np.float64)

        # match each En level to nearest sum
        diffs = []
        for val in En:
            j = int(np.searchsorted(sums, val))
            candidates = []
            if 0 <= j < len(sums):
                candidates.append(abs(val - sums[j]))
            if 0 <= j - 1 < len(sums):
                candidates.append(abs(val - sums[j - 1]))
            diffs.append(min(candidates) if candidates else float("nan"))

        diffs = np.array(diffs, dtype=np.float64)
        rms = float(np.sqrt(np.nanmean(diffs * diffs)))
        max_abs = float(np.nanmax(np.abs(diffs)))
        return rms, max_abs

    if max_n >= 2:
        rms2, max2 = best_rms_for_n(2)
        out["additivity_n2_rms"] = rms2
        out["additivity_n2_max_abs"] = max2

    if max_n >= 3:
        rms3, max3 = best_rms_for_n(3)
        out["additivity_n3_rms"] = rms3
        out["additivity_n3_max_abs"] = max3

    return out


def jordan_wigner_ops(N: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build full matrices c_j and c_j^\\dagger using:
      c_j = (prod_{m<j} Z_m) sigma^-_j
    """
    I, _, _, Z, sm, sp = dense_pauli()
    c = []
    cd = []
    for j in range(N):
        ops_c = []
        ops_cd = []
        for m in range(N):
            if m < j:
                ops_c.append(Z)
                ops_cd.append(Z)
            elif m == j:
                ops_c.append(sm)
                ops_cd.append(sp)
            else:
                ops_c.append(I)
                ops_cd.append(I)
        c.append(kron_n(ops_c))
        cd.append(kron_n(ops_cd))
    return c, cd


def jw_anticommutator_audit(H: np.ndarray, N: int) -> Dict[str, Any]:
    """
    Check anticommutation relations on the ground state:
      <{c_i, c_j}> ~ 0
      <{c_i, c_j^dagger}> ~ delta_ij
    Report max absolute deviations.
    """
    evals, evecs = eigh(H)
    psi0 = evecs[:, 0]
    c, cd = jordan_wigner_ops(N)

    max_cc = 0.0
    max_ccd_off = 0.0
    max_ccd_diag = 0.0

    for i in range(N):
        for j in range(N):
            anti_cc = c[i] @ c[j] + c[j] @ c[i]
            val_cc = complex(np.vdot(psi0, anti_cc @ psi0))
            max_cc = max(max_cc, abs(val_cc))

            anti_ccd = c[i] @ cd[j] + cd[j] @ c[i]
            val_ccd = complex(np.vdot(psi0, anti_ccd @ psi0))
            if i == j:
                # should be ~1
                max_ccd_diag = max(max_ccd_diag, abs(val_ccd - 1.0))
            else:
                # should be ~0
                max_ccd_off = max(max_ccd_off, abs(val_ccd))

    return {
        "jw_max_abs_anticomm_cc": float(max_cc),
        "jw_max_abs_anticomm_ccd_offdiag": float(max_ccd_off),
        "jw_max_abs_anticomm_ccd_diag_minus1": float(max_ccd_diag),
    }


def pauli_pressure_curvature(H: np.ndarray, N: int, n_max: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute ground energies E0(n) in fixed n_down sectors, then curvature:
      kappa^{-1}(n) = E0(n+1) - 2 E0(n) + E0(n-1)
    """
    if n_max is None:
        n_max = N
    n_max = min(n_max, N)

    E0n = []
    for n in range(n_max + 1):
        idx = basis_indices_fixed_n(N, n)
        Hn = sector_submatrix(H, idx)
        e = float(np.real(eigh(Hn, eigvals_only=True)[0]))
        E0n.append(e)

    curv = {}
    for n in range(1, n_max):
        curv[n] = float(E0n[n + 1] - 2.0 * E0n[n] + E0n[n - 1])

    return {"E0_by_n": [float(x) for x in E0n], "curvature": {str(k): float(v) for k, v in curv.items()}}


# =============================================================================
# STROBE optimizer (stores best moves; can load+apply moves)
# =============================================================================

@dataclass
class StrobeConfig:
    cycles: int = 8000
    temp: float = 0.05
    temp_decay: float = 0.9995
    gate_eps: float = 0.05
    edges: str = "all"            # move-neighborhood: "ring" or "all"
    objective: str = "sparse"     # "sparse" or "range"
    leak_power: int = 2           # only for "range"
    early_stop: float = 1e-10
    early_patience: int = 1500


@dataclass
class StrobeMove:
    step: int
    q1: int
    q2: int
    eps: float
    U2: np.ndarray


def strobe_objective_value(H: np.ndarray, N: int, cfg: StrobeConfig, D_probe: Optional[np.ndarray]) -> float:
    if cfg.objective == "sparse":
        return objective_sparse_ratio(H, N)
    if cfg.objective == "signal":
        return objective_signal_entropy(H, N)
    if cfg.objective == "range":
        if D_probe is None:
            raise ValueError("range objective requested but no probe distances provided")
        return objective_range_leakage(H, N, D_probe, power=cfg.leak_power)
    raise ValueError(f"Unknown strobe objective: {cfg.objective}")


def strobe_optimize(H0: np.ndarray, N: int, cfg: StrobeConfig, seed: int, D_probe: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any], List[StrobeMove]]:
    rng = np.random.default_rng(seed)

    move_edges = ring_edges(N) if cfg.edges == "ring" else all_edges(N)
    if not move_edges:
        raise RuntimeError("No move edges available")

    H = H0.copy()
    cost = float(strobe_objective_value(H, N, cfg, D_probe))

    best_H = H.copy()
    best_cost = float(cost)
    accepted_moves: List[StrobeMove] = []
    best_moves: List[StrobeMove] = []

    T = float(cfg.temp)
    accepted = 0
    improved = 0
    t0 = time.time()
    last_best = 0

    for step in range(1, cfg.cycles + 1):
        if best_cost <= cfg.early_stop:
            break
        if (step - last_best) >= cfg.early_patience:
            break

        q1, q2 = move_edges[int(rng.integers(0, len(move_edges)))]
        U2 = random_small_two_qubit_gate(rng, eps=float(cfg.gate_eps))

        Hcand = apply_two_qubit_conjugation_to_operator(H, N, q1, q2, U2)
        cand_cost = float(strobe_objective_value(Hcand, N, cfg, D_probe))

        dE = cand_cost - cost
        if dE <= 0 or (T > 1e-12 and rng.random() < math.exp(-dE / T)):
            H = Hcand
            cost = cand_cost
            accepted += 1
            mv = StrobeMove(step=step, q1=int(q1), q2=int(q2), eps=float(cfg.gate_eps), U2=U2)
            accepted_moves.append(mv)
            if dE < 0:
                improved += 1

            if cost < best_cost:
                best_cost = float(cost)
                best_H = H.copy()
                best_moves = list(accepted_moves)
                last_best = step

        T *= cfg.temp_decay

    meta = {
        "objective": cfg.objective,
        "edges": cfg.edges,
        "cycles": int(cfg.cycles),
        "seed": int(seed),
        "temp": float(cfg.temp),
        "temp_decay": float(cfg.temp_decay),
        "gate_eps": float(cfg.gate_eps),
        "leak_power": int(cfg.leak_power),
        "initial_cost": float(strobe_objective_value(H0, N, cfg, D_probe)),
        "best_cost": float(best_cost),
        "accepted": int(accepted),
        "improved": int(improved),
        "best_move_count": int(len(best_moves)),
        "steps_executed": int(step),
        "stopped_early": bool(step < cfg.cycles),
        "wall_time_s": float(time.time() - t0),
    }
    return best_H, meta, best_moves


def save_moves_json(path: str, N: int, seed: int, moves: List[StrobeMove], cfg: StrobeConfig) -> None:
    payload = {
        "N": int(N),
        "seed": int(seed),
        "strobe": {"objective": cfg.objective, "edges": cfg.edges, "gate_eps": float(cfg.gate_eps)},
        "moves": [
            {
                "step": int(m.step),
                "q1": int(m.q1),
                "q2": int(m.q2),
                "eps": float(m.eps),
                "U2_re": m.U2.real.tolist(),
                "U2_im": m.U2.imag.tolist(),
            }
            for m in moves
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_moves_json(path: str) -> List[StrobeMove]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    moves: List[StrobeMove] = []
    for m in payload.get("moves", []):
        Ure = np.array(m["U2_re"], dtype=np.float64)
        Uim = np.array(m["U2_im"], dtype=np.float64)
        U2 = (Ure + 1j * Uim).astype(np.complex128)
        moves.append(StrobeMove(step=int(m["step"]), q1=int(m["q1"]), q2=int(m["q2"]), eps=float(m["eps"]), U2=U2))
    return moves


def apply_moves(H: np.ndarray, N: int, moves: List[StrobeMove]) -> np.ndarray:
    out = H
    for m in moves:
        out = apply_two_qubit_conjugation_to_operator(out, N, m.q1, m.q2, m.U2)
    return out


# =============================================================================
# Metrics bundle
# =============================================================================

def compute_metrics(H: np.ndarray, N: int, p: int, max_weight_opt: Optional[int], audit: bool) -> Dict[str, Any]:
    C_opt, _ = locality_cost_and_total_sq(H, N, p=p, max_weight=max_weight_opt)
    S = pair_strengths_matrix(H, N)
    sparse_cost = float(objective_sparse_ratio(H, N))
    signal_entropy = float(objective_signal_entropy(H, N))
    top_share = float(topk_strength_share(S, topk=N))  # N edges ~ ring density

    out: Dict[str, Any] = {
        "locality_cost_opt": float(C_opt),
        "sparse_cost": float(sparse_cost),
        "signal_entropy": float(signal_entropy),
        "topN_share": float(top_share),
        "pair_strengths": S.tolist(),
    }

    # diagnostic ring ruler (not used for geometry-blind success)
    V_ring = measure_V_vs_d_ring(H, N)
    out["V_ring"] = {int(k): float(v) for k, v in V_ring.items()}
    out["V2_ring_abs"] = float(abs(out["V_ring"].get(2, 0.0)))

    if audit:
        C_full, _ = locality_cost_and_total_sq(H, N, p=p, max_weight=None)
        out["locality_cost_full"] = float(C_full)

    return out


# =============================================================================
# One run (seed)
# =============================================================================

def run_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    set_thread_env(int(payload.get("blas_threads", 1)))

    N = int(payload["N"])
    seed = int(payload["seed"])
    scramble = payload["scramble"]
    recover = payload["recover"]
    model = payload["model"]
    Delta = float(payload.get("Delta", 1.0))

    # flow
    flow_steps = int(payload["flow_steps"])
    dt = float(payload["dt"])
    p = int(payload["p"])
    max_weight_opt = payload["max_weight_opt"]

    # strobe
    strobe_cfg = StrobeConfig(**payload["strobe_cfg"])
    save_moves_path = payload.get("save_moves", None)
    load_moves_path = payload.get("load_moves", None)

    # measurement
    infer_mode = payload.get("infer_graph", "topk:8")
    audit = bool(payload.get("audit", False))
    fermion_audit = bool(payload.get("fermion_audit", False))

    out: Dict[str, Any] = {
        "N": N,
        "dim": int(2 ** N),
        "seed": seed,
        "model": model,
        "Delta": Delta,
        "scramble": scramble,
        "recover": recover,
        "flow": {"steps": flow_steps, "dt": dt, "p": p, "max_weight_opt": max_weight_opt},
        "strobe": strobe_cfg.__dict__,
        "infer_graph": infer_mode,
        "audit": audit,
        "fermion_audit": fermion_audit,
    }

    # generator instance
    H_spatial = spin_ring_dense(N, model=model, J=1.0, Delta=Delta)

    # scramble
    if scramble == "local":
        U = build_local_scrambler(N, seed)
    else:
        U = build_global_scrambler(N, seed)
    H = scramble_hamiltonian(H_spatial, U)

    out["metrics_scrambled"] = compute_metrics(H, N, p=p, max_weight_opt=max_weight_opt, audit=audit)

    D_probe = ring_distances(N) if strobe_cfg.objective == "range" else None

    # RECOVERY: strobe
    if recover in ("strobe", "both"):
        if load_moves_path:
            moves_used = load_moves_json(load_moves_path)
            H = apply_moves(H, N, moves_used)
            out["strobe_loaded_moves"] = {"path": load_moves_path, "count": len(moves_used)}
            out["strobe_meta"] = {"objective": "loaded", "best_move_count": len(moves_used)}
        else:
            H_best, meta, best_moves = strobe_optimize(H, N, strobe_cfg, seed=seed + 1, D_probe=D_probe)
            H = H_best
            out["strobe_meta"] = meta
            if save_moves_path:
                save_moves_json(save_moves_path, N, seed, best_moves, strobe_cfg)
                out["strobe_saved_moves"] = {"path": save_moves_path, "count": len(best_moves)}

        out["metrics_after_strobe"] = compute_metrics(H, N, p=p, max_weight_opt=max_weight_opt, audit=audit)

    # RECOVERY: flow
    if recover in ("flow", "both") and flow_steps > 0:
        t0 = time.time()
        for _ in range(flow_steps):
            H = isospectral_flow_step(H, N, dt=dt, p=p, max_weight=max_weight_opt)
        out["flow_wall_time_s"] = float(time.time() - t0)
        out["metrics_final"] = compute_metrics(H, N, p=p, max_weight_opt=max_weight_opt, audit=audit)
    else:
        out["metrics_final"] = out.get("metrics_after_strobe", out["metrics_scrambled"])

    # Emergent graph inference
    S = np.array(out["metrics_final"]["pair_strengths"], dtype=np.float64)
    edges_w = infer_graph_from_strengths(S, infer_mode)
    edges = [(i, j) for (i, j, _) in edges_w]
    out["emergent_graph"] = {
        "edges": [{"i": int(i), "j": int(j), "w": float(w)} for (i, j, w) in edges_w],
        "diagnostics": graph_diagnostics(N, edges),
    }

    # Geometry-blind success metrics (these matter for sparse objective)
    sc_i = float(out["metrics_scrambled"]["sparse_cost"])
    sc_f = float(out["metrics_final"]["sparse_cost"])
    out["sparse_cost_initial"] = sc_i
    out["sparse_cost_final"] = sc_f
    out["sparse_reduction"] = float((sc_i / sc_f) if sc_f > 1e-12 else float("inf"))
    out["topN_share_final"] = float(out["metrics_final"]["topN_share"])
    out["locality_recovered_sparse"] = bool((out["sparse_reduction"] >= 4.0) or (out["topN_share_final"] >= 0.70))

    # Signal-dispersion objective diagnostics (geometry-blind)
    se_i = float(out["metrics_scrambled"].get("signal_entropy", float("nan")))
    se_f = float(out["metrics_final"].get("signal_entropy", float("nan")))
    out["signal_entropy_initial"] = se_i
    out["signal_entropy_final"] = se_f
    out["signal_entropy_reduction"] = float((se_i / se_f) if (se_f > 1e-12) else float("inf"))
    # Soft success flag for signaling (tunable)
    out["locality_recovered_signal"] = bool(
        (not math.isnan(se_i)) and (not math.isnan(se_f)) and ((out["signal_entropy_reduction"] >= 1.5) or (se_f <= 0.55))
    )


    # Diagnostic ring ruler (keep, but don't use as main success in sparse mode)
    V2_i = float(out["metrics_scrambled"]["V2_ring_abs"])
    V2_f = float(out["metrics_final"]["V2_ring_abs"])
    out["V2_ring_initial"] = V2_i
    out["V2_ring_final"] = V2_f
    out["V2_ring_reduction"] = float((V2_i / V2_f) if V2_f > 1e-12 else float("inf"))

    # Fermion audit (Part III)
    if fermion_audit:
        out["fermion_audit_results"] = {
            "sector_additivity": fermion_sector_additivity_audit(H, N, max_n=3),
            "jw_anticommutators": jw_anticommutator_audit(H, N),
            "pauli_pressure": pauli_pressure_curvature(H, N, n_max=min(N, 8)),
        }

    return out


# =============================================================================
# Sweep summary
# =============================================================================

def summarize_sweep(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if isinstance(r, dict) and "error" not in r]
    succ_sparse = sum(1 for r in ok if r.get("locality_recovered_sparse", False))
    succ_signal = sum(1 for r in ok if r.get("locality_recovered_signal", False))
    reds = [r.get("sparse_reduction", float("nan")) for r in ok]
    sreds = [r.get("signal_entropy_reduction", float("nan")) for r in ok]
    reds = [x for x in reds if isinstance(x, (int, float)) and np.isfinite(x)]
    sreds = [x for x in sreds if isinstance(x, (int, float)) and np.isfinite(x)]
    return {
        "runs": int(len(results)),
        "successes_sparse": int(succ_sparse),
        "success_rate_sparse": float(succ_sparse / len(ok)) if ok else 0.0,
        "successes_signal": int(succ_signal),
        "success_rate_signal": float(succ_signal / len(ok)) if ok else 0.0,
        "median_sparse_reduction": float(np.median(reds)) if reds else float("nan"),
        "mean_sparse_reduction": float(np.mean(reds)) if reds else float("nan"),
        "min_sparse_reduction": float(np.min(reds)) if reds else float("nan"),
        "max_sparse_reduction": float(np.max(reds)) if reds else float("nan"),
        "median_signal_entropy_reduction": float(np.median(sreds)) if sreds else float("nan"),
        "mean_signal_entropy_reduction": float(np.mean(sreds)) if sreds else float("nan"),
    }


# =============================================================================
# CLI parsing helpers
# =============================================================================

def parse_infer_graph(s: str) -> str:
    s = s.strip().lower()
    if s == "none":
        return "none"
    if s.startswith("topk:"):
        _ = int(s.split(":", 1)[1])
        return s
    if s.startswith("threshold:"):
        _ = float(s.split(":", 1)[1])
        return s
    raise argparse.ArgumentTypeError("infer-graph must be: none | topk:K | threshold:T")


def parse_max_weight(s: str) -> Optional[int]:
    s = str(s).strip().lower()
    if s in ("none", "null", "inf", "full"):
        return None
    return int(s)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Scramble → Recover Locality (v3)")

    # core
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--model", choices=["xx", "xxz", "xxx"], default="xxx")
    parser.add_argument("--Delta", type=float, default=1.0, help="Only used if model=xxz (ZZ coefficient).")
    parser.add_argument("--scramble", choices=["global", "local"], default="global")
    parser.add_argument("--recover", choices=["none", "flow", "strobe", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=1)

    # flow
    parser.add_argument("--flow-steps", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--max-weight", type=parse_max_weight, default=4, help="Int or 'none' for full basis (slow).")

    # strobe
    parser.add_argument("--cycles", type=int, default=8000)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--temp-decay", type=float, default=0.9995)
    parser.add_argument("--gate-eps", type=float, default=0.05)
    parser.add_argument("--strobe-edges", choices=["ring", "all"], default="all")
    parser.add_argument("--strobe-objective", choices=["sparse", "signal", "range"], default="sparse")
    parser.add_argument("--leak-power", type=int, default=2)

    # moves
    parser.add_argument("--save-moves", type=str, default=None)
    parser.add_argument("--load-moves", type=str, default=None)

    # measurement/inference
    parser.add_argument("--infer-graph", type=parse_infer_graph, default="topk:8")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--fermion-audit", action="store_true")

    # parallel sweep
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--partial-output", type=str, default=None, help="Write JSONL as results finish (good for long sweeps).")

    args = parser.parse_args()

    max_weight_opt: Optional[int] = args.max_weight

    strobe_cfg = StrobeConfig(
        cycles=int(args.cycles),
        temp=float(args.temp),
        temp_decay=float(args.temp_decay),
        gate_eps=float(args.gate_eps),
        edges=str(args.strobe_edges),
        objective=str(args.strobe_objective),
        leak_power=int(args.leak_power),
    )

    # Sweep?
    if args.seed_count > 1:
        start = int(args.seed_start if args.seed_start is not None else args.seed)
        seeds = [start + k for k in range(int(args.seed_count))]
        jobs = max(1, int(args.jobs))

        set_thread_env(int(args.blas_threads))

        payloads: List[Dict[str, Any]] = []
        for s in seeds:
            payloads.append({
                "N": int(args.N),
                "seed": int(s),
                "model": str(args.model),
                "Delta": float(args.Delta),
                "scramble": str(args.scramble),
                "recover": str(args.recover),
                "flow_steps": int(args.flow_steps),
                "dt": float(args.dt),
                "p": int(args.p),
                "max_weight_opt": max_weight_opt,
                "strobe_cfg": strobe_cfg.__dict__,
                "save_moves": None,
                "load_moves": None,
                "infer_graph": str(args.infer_graph),
                "audit": bool(args.audit),
                "fermion_audit": bool(args.fermion_audit),
                "blas_threads": int(args.blas_threads),
            })

        # prepare partial output
        if args.partial_output:
            # truncate existing
            with open(args.partial_output, "w", encoding="utf-8") as f:
                f.write("")

        t0 = time.time()
        results: List[Dict[str, Any]] = []

        if jobs == 1:
            for k, pl in enumerate(payloads, start=1):
                r = run_one(pl)
                results.append(r)
                if args.partial_output:
                    with open(args.partial_output, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                if args.progress and not args.quiet:
                    print(f"[{k}/{len(payloads)}] seed={r['seed']} sparse_red={r['sparse_reduction']:.3g} sparse_ok={r['locality_recovered_sparse']} V2_red={r['V2_ring_reduction']:.3g}", flush=True)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            if not args.quiet:
                print(f"\nLaunching sweep: N={args.N} model={args.model} seeds={len(payloads)} jobs={jobs} blas_threads={args.blas_threads} strobe_obj={strobe_cfg.objective}", flush=True)

            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(run_one, pl): pl["seed"] for pl in payloads}
                if not args.quiet:
                    print(f"Submitted {len(futs)} jobs.", flush=True)

                done = 0
                for fut in as_completed(futs):
                    seed_done = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        r = {"seed": int(seed_done), "error": repr(e)}
                    results.append(r)
                    done += 1

                    if args.partial_output:
                        with open(args.partial_output, "a", encoding="utf-8") as f:
                            f.write(json.dumps(r) + "\n")
                            f.flush()

                    if args.progress and not args.quiet:
                        if "error" in r:
                            print(f"[{done}/{len(payloads)}] seed={seed_done} ERROR {r['error']}", flush=True)
                        else:
                            print(f"[{done}/{len(payloads)}] seed={seed_done} sparse_red={r['sparse_reduction']:.3g} sparse_ok={r['locality_recovered_sparse']} V2_red={r['V2_ring_reduction']:.3g}", flush=True)

        results.sort(key=lambda x: int(x.get("seed", 0)))
        summary = summarize_sweep([r for r in results if isinstance(r, dict)])

        payload_out = {
            "sweep": {
                "N": int(args.N),
                "model": str(args.model),
                "Delta": float(args.Delta),
                "scramble": str(args.scramble),
                "recover": str(args.recover),
                "seed_start": int(seeds[0]),
                "seed_count": int(len(seeds)),
                "flow": {"steps": int(args.flow_steps), "dt": float(args.dt), "p": int(args.p), "max_weight_opt": max_weight_opt},
                "strobe": strobe_cfg.__dict__,
                "infer_graph": str(args.infer_graph),
                "audit": bool(args.audit),
                "fermion_audit": bool(args.fermion_audit),
                "jobs": int(jobs),
                "blas_threads": int(args.blas_threads),
                "wall_time_s": float(time.time() - t0),
            },
            "summary": summary,
            "runs": results,
        }

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload_out, f, indent=2)
            if not args.quiet:
                print(f"\nSaved sweep: {args.output}", flush=True)
                if args.partial_output:
                    print(f"Partial (JSONL): {args.partial_output}", flush=True)
        else:
            print(json.dumps(summary, indent=2))

        return

    # Single run
    set_thread_env(int(args.blas_threads))
    pl = {
        "N": int(args.N),
        "seed": int(args.seed),
        "model": str(args.model),
        "Delta": float(args.Delta),
        "scramble": str(args.scramble),
        "recover": str(args.recover),
        "flow_steps": int(args.flow_steps),
        "dt": float(args.dt),
        "p": int(args.p),
        "max_weight_opt": max_weight_opt,
        "strobe_cfg": strobe_cfg.__dict__,
        "save_moves": args.save_moves,
        "load_moves": args.load_moves,
        "infer_graph": str(args.infer_graph),
        "audit": bool(args.audit),
        "fermion_audit": bool(args.fermion_audit),
        "blas_threads": int(args.blas_threads),
    }

    t0 = time.time()
    r = run_one(pl)
    r["wall_time_s"] = float(time.time() - t0)

    if not args.quiet:
        print("\n================================================================================")
        print("SCRAMBLE → RECOVER (v3)")
        print("================================================================================")
        print(f"System: N={r['N']} dim={r['dim']} model={r['model']} Delta={r['Delta']}")
        print(f"Scramble={r['scramble']} | Recover={r['recover']}")
        print(f"Strobe: objective={strobe_cfg.objective} edges={strobe_cfg.edges} cycles={strobe_cfg.cycles} eps={strobe_cfg.gate_eps}")
        print(f"Flow: steps={args.flow_steps} dt={args.dt} p={args.p} max_weight_opt={max_weight_opt}")
        print(f"Sparse: cost {r['sparse_cost_initial']:.6g} → {r['sparse_cost_final']:.6g}  (reduction {r['sparse_reduction']:.3g}x) | topN_share={r['topN_share_final']:.3f}")
        print(f"Recovered (sparse)? {'YES' if r['locality_recovered_sparse'] else 'NO'}")
        print(f"Ring probe V2 reduction (diagnostic): {r['V2_ring_reduction']:.3g}x")
        if args.fermion_audit:
            fa = r.get("fermion_audit_results", {})
            sa = fa.get("sector_additivity", {})
            jw = fa.get("jw_anticommutators", {})
            print("---- Fermion audit ----")
            if "additivity_n2_rms" in sa:
                print(f"Sector additivity n=2 RMS: {sa['additivity_n2_rms']:.3e} | max: {sa['additivity_n2_max_abs']:.3e}")
            if "additivity_n3_rms" in sa:
                print(f"Sector additivity n=3 RMS: {sa['additivity_n3_rms']:.3e} | max: {sa['additivity_n3_max_abs']:.3e}")
            print(f"JW max |<{{c,c}}>|: {jw.get('jw_max_abs_anticomm_cc', float('nan')):.3e}")
            print(f"JW max offdiag |<{{c,c†}}>|: {jw.get('jw_max_abs_anticomm_ccd_offdiag', float('nan')):.3e}")
            print(f"JW max diag |<{{c,c†}}>-1|: {jw.get('jw_max_abs_anticomm_ccd_diag_minus1', float('nan')):.3e}")
        print(f"Wall time: {r['wall_time_s']:.2f}s")
        print("================================================================================\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        if not args.quiet:
            print(f"Saved run: {args.output}", flush=True)
    else:
        print(json.dumps({
            "seed": r["seed"],
            "sparse_reduction": r["sparse_reduction"],
            "locality_recovered_sparse": r["locality_recovered_sparse"],
            "V2_ring_reduction": r["V2_ring_reduction"],
        }, indent=2))


if __name__ == "__main__":
    main()
