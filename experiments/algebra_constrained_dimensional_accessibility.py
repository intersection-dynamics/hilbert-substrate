#!/usr/bin/env python3
"""
Algebra-Constrained Dimensional Accessibility (Option A: Source-local scrambling)

This script respects the Paper-2 "accessibility barrier":
  - We DO NOT globally scramble into an unreachable basin.
  - We scramble only using local moves on a chosen SOURCE geometry graph.
  - Then we compare how accessible different TARGET geometries are, using their
    own local move sets, given the same reachable starting Hamiltonian.

Core question operationalized:
  Does conditioning on an algebraic (JW/CAR-compatible) subsystem ordering
  bias the accessibility landscape of emergent geometry (e.g., 3D) compared
  to controls?

Important baseline limitation:
  "Factorizations" are modeled as qubit permutations (relabelings). This is
  conservative and avoids claiming more than we compute. Extend later if needed.

Outputs:
  JSON report with conditioned vs control rankings over target geometries.

Boring and falsifiable. No particle claims.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Random unitaries
# -----------------------------

def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-ish random unitary via QR of complex Gaussian matrix."""
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex128)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.where(np.abs(d) > 0, np.abs(d), 1.0)
    q = q * np.conj(ph)
    return q


def random_two_qubit_unitary(rng: np.random.Generator) -> np.ndarray:
    return random_unitary(4, rng)


# -----------------------------
# Pauli operators / sampling
# -----------------------------

PAULI_SINGLE = {
    'I': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

PAULI_LABELS = np.array(['I', 'X', 'Y', 'Z'])


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def sample_pauli_strings(
    N: int,
    m: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample m Pauli strings uniformly from {I,X,Y,Z}^N.

    Returns:
      labels: (m, N) int8 in {0,1,2,3} for I,X,Y,Z
      weights: (m,) int32 = number of non-identity factors (k-body size)
    """
    labels = rng.integers(low=0, high=4, size=(m, N), dtype=np.int8)
    weights = np.sum(labels != 0, axis=1).astype(np.int32)
    return labels, weights


def pauli_labels_to_matrix(labels_row: np.ndarray) -> np.ndarray:
    ops = [PAULI_SINGLE[PAULI_LABELS[int(v)]] for v in labels_row]
    return kron_n(ops)


def estimate_klocal_leak(
    H: np.ndarray,
    pauli_samples: Tuple[np.ndarray, np.ndarray],
    k_max: int = 2,
) -> float:
    """
    Monte Carlo estimate of leakage fraction into terms with body-size > k_max.

    For sampled Pauli strings P:
      a_P = Tr(P H) / 2^N
      weight contribution ~ |a_P|^2
    We estimate fraction of sampled weight with k>k_max.
    """
    labels, weights = pauli_samples
    N = labels.shape[1]
    dim = 2 ** N

    num = 0.0
    den = 0.0
    for row, k in zip(labels, weights):
        P = pauli_labels_to_matrix(row)
        a = np.trace(P @ H) / dim
        w = (np.abs(a) ** 2).real
        den += w
        if k > k_max:
            num += w

    if den <= 0:
        return 1.0
    return float(num / den)


# -----------------------------
# Graph builders (geometries)
# -----------------------------

def edges_1d_ring(N: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % N) for i in range(N)]


def edges_1d_chain(N: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(N - 1)]


def edges_2d_lattice(N: int) -> Optional[List[Tuple[int, int]]]:
    L = int(round(math.sqrt(N)))
    if L * L != N:
        return None
    edges = []
    def idx(r, c): return r * L + c
    for r in range(L):
        for c in range(L):
            if c + 1 < L:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < L:
                edges.append((idx(r, c), idx(r + 1, c)))
    return edges


def edges_3d_lattice(N: int) -> Optional[List[Tuple[int, int]]]:
    L = int(round(N ** (1/3)))
    if L * L * L != N:
        return None
    edges = []
    def idx(x, y, z): return (x * L + y) * L + z
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if z + 1 < L:
                    edges.append((idx(x, y, z), idx(x, y, z + 1)))
                if y + 1 < L:
                    edges.append((idx(x, y, z), idx(x, y + 1, z)))
                if x + 1 < L:
                    edges.append((idx(x, y, z), idx(x + 1, y, z)))
    return edges


def make_random_regular_edges(N: int, d: int, rng: np.random.Generator, max_tries: int = 20000) -> Optional[List[Tuple[int, int]]]:
    """
    Stub-matching random regular graph generator. Returns None if it fails.
    Increased retry budget for small-N stability.
    """
    if d >= N or (N * d) % 2 != 0:
        return None

    for _ in range(max_tries):
        stubs = np.repeat(np.arange(N, dtype=np.int32), d)
        rng.shuffle(stubs)
        edges = []
        ok = True
        for i in range(0, len(stubs), 2):
            a = int(stubs[i]); b = int(stubs[i + 1])
            if a == b:
                ok = False
                break
            e = (a, b) if a < b else (b, a)
            edges.append(e)
        if not ok:
            continue
        if len(set(edges)) != len(edges):
            continue
        return edges

    return None


def parse_list(arg: str) -> List[str]:
    return [s.strip() for s in arg.split(",") if s.strip()]


# -----------------------------
# Hamiltonian construction
# -----------------------------

def embed_two_qubit_op(N: int, i: int, j: int, op2: np.ndarray) -> np.ndarray:
    """Embed a 4x4 operator acting on qubits (i,j) into 2^N space using Pauli expansion."""
    if i == j:
        raise ValueError("i == j")
    if i > j:
        i, j = j, i

    paulis = ['I', 'X', 'Y', 'Z']
    basis = []
    for a in paulis:
        for b in paulis:
            basis.append(np.kron(PAULI_SINGLE[a], PAULI_SINGLE[b]))
    B = np.stack([b.reshape(-1) for b in basis], axis=1)  # 16 x 16
    coeffs = np.linalg.solve(B, op2.reshape(-1))

    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)

    idx = 0
    for a in paulis:
        for b in paulis:
            c = coeffs[idx]
            idx += 1
            if np.abs(c) < 1e-14:
                continue
            ops_full = []
            for q in range(N):
                if q == i:
                    ops_full.append(PAULI_SINGLE[a])
                elif q == j:
                    ops_full.append(PAULI_SINGLE[b])
                else:
                    ops_full.append(PAULI_SINGLE['I'])
            H += c * kron_n(ops_full)
    return H


def build_xx_hamiltonian(N: int, edges: List[Tuple[int, int]], J: float = 1.0) -> np.ndarray:
    """H = sum_{(i,j)} J*(X_i X_j + Y_i Y_j)/2 (XX model)."""
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)

    XX = np.kron(PAULI_SINGLE['X'], PAULI_SINGLE['X'])
    YY = np.kron(PAULI_SINGLE['Y'], PAULI_SINGLE['Y'])
    op2 = 0.5 * (XX + YY)

    for (i, j) in edges:
        H += J * embed_two_qubit_op(N, i, j, op2)

    H = 0.5 * (H + H.conj().T)
    return H


def apply_two_qubit_conjugation(H: np.ndarray, N: int, i: int, j: int, U2: np.ndarray) -> np.ndarray:
    """
    Conjugate H by a 2-qubit unitary acting on qubits i and j:
      H' = U H U†, where U embeds U2 into N-qubit space.

    Correct tensor-index implementation on 2N axes.
    """
    if i == j:
        raise ValueError("i == j")
    if i > j:
        i, j = j, i

    dim = 2 ** N
    shp = (2,) * N
    Ht = H.reshape(shp + shp)

    qubits = list(range(N))
    rest = [q for q in qubits if q not in (i, j)]

    ket_order = [i, j] + rest
    bra_order = [i + N, j + N] + [q + N for q in rest]
    perm = ket_order + bra_order  # length 2N

    Hperm = np.transpose(Ht, axes=perm)
    Hperm = Hperm.reshape(4, 2 ** (N - 2), 4, 2 ** (N - 2))

    U = U2
    Udag = U2.conj().T
    Hnew = np.einsum("am,mxny,bn->axby", U, Hperm, Udag, optimize=True)

    Hnew = Hnew.reshape((2, 2) + (2,) * (N - 2) + (2, 2) + (2,) * (N - 2))
    inv = np.argsort(np.array(perm))
    Hback = np.transpose(Hnew, axes=inv).reshape(dim, dim)

    Hback = 0.5 * (Hback + Hback.conj().T)
    return Hback


# -----------------------------
# Local (source) scrambling
# -----------------------------

def local_scramble_on_edges(
    H: np.ndarray,
    N: int,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
    steps: int,
) -> np.ndarray:
    """
    Scramble by applying random 2-qubit conjugations ONLY along given edges.
    This is "reachable" under the same locality constraints (Paper-2 compatible).
    """
    Hs = H.copy()
    for _ in range(steps):
        (i, j) = edges[int(rng.integers(0, len(edges)))]
        U2 = random_two_qubit_unitary(rng)
        Hs = apply_two_qubit_conjugation(Hs, N, i, j, U2)
    return Hs


# -----------------------------
# JW operators and quadratic projection score
# -----------------------------

def jw_creation_annihilation_ops(N: int, ordering: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build JW annihilation operators c_k for a given qubit ordering (permutation).
    c_k = (Z on all previous sites) ⊗ sigma^- on site k.
    """
    dim = 2 ** N
    sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    Z = PAULI_SINGLE['Z']
    I = PAULI_SINGLE['I']

    pos_of = {q: idx for idx, q in enumerate(ordering)}

    cs = []
    cds = []
    for k in range(N):
        qk = ordering[k]
        ops = []
        for q in range(N):
            if q == qk:
                ops.append(sm)
            else:
                ops.append(Z if pos_of[q] < k else I)
        c = kron_n(ops)
        cs.append(c)
        cds.append(c.conj().T)
    return cs, cds


def quadratic_subspace_basis(N: int, ordering: List[int]) -> List[np.ndarray]:
    """
    Basis for quadratic fermionic operators:
      I
      c_i^† c_j (all i,j)
      c_i c_j and c_i^† c_j^† (i<j)
    """
    cs, cds = jw_creation_annihilation_ops(N, ordering)
    dim = 2 ** N
    basis = [np.eye(dim, dtype=np.complex128)]

    for i in range(N):
        for j in range(N):
            basis.append(cds[i] @ cs[j])

    for i in range(N):
        for j in range(i + 1, N):
            basis.append(cs[i] @ cs[j])
            basis.append(cds[i] @ cds[j])

    return basis


def quadratic_projection_score(H: np.ndarray, basis: List[np.ndarray]) -> float:
    """
    Least-squares projection of H onto span(basis) in Frobenius inner product.
    Q = ||Proj(H)||_F^2 / ||H||_F^2
    """
    vecs = [B.reshape(-1) for B in basis]
    A = np.stack(vecs, axis=1)
    h = H.reshape(-1)

    x, *_ = np.linalg.lstsq(A, h, rcond=None)
    proj = A @ x

    num = float(np.vdot(proj, proj).real)
    den = float(np.vdot(h, h).real)
    if den <= 0:
        return 0.0
    return num / den


# -----------------------------
# Accessibility recovery
# -----------------------------

@dataclass
class RecoveryResult:
    leak_initial: float
    leak_final: float
    leak_reduction: float
    accepted_moves: int
    steps: int


def annealed_recovery(
    H_start: np.ndarray,
    N: int,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
    steps: int,
    temp0: float,
    temp_decay: float,
    pauli_samples: Tuple[np.ndarray, np.ndarray],
    k_max: int = 2,
) -> RecoveryResult:
    """
    Annealed local-unitary recovery constrained to edges.
    Objective: minimize k-local leak fraction.
    """
    H = H_start.copy()
    leak0 = estimate_klocal_leak(H, pauli_samples, k_max=k_max)
    best = leak0
    accepted = 0
    T = temp0

    for _ in range(steps):
        (i, j) = edges[int(rng.integers(0, len(edges)))]
        U2 = random_two_qubit_unitary(rng)
        Hcand = apply_two_qubit_conjugation(H, N, i, j, U2)
        leak = estimate_klocal_leak(Hcand, pauli_samples, k_max=k_max)

        d = leak - best
        if d <= 0:
            accept = True
        else:
            accept = (rng.random() < math.exp(-d / max(T, 1e-12)))

        if accept:
            H = Hcand
            if leak < best:
                best = leak
            accepted += 1

        T *= temp_decay

    leakF = best
    return RecoveryResult(
        leak_initial=float(leak0),
        leak_final=float(leakF),
        leak_reduction=float(leak0 - leakF),
        accepted_moves=int(accepted),
        steps=int(steps),
    )


# -----------------------------
# Permuting operators (factorizations)
# -----------------------------

def permute_operator_qubits(H: np.ndarray, N: int, p: List[int]) -> np.ndarray:
    """
    Apply qubit permutation p to operator tensor.
    p is a list length N where new axis k corresponds to old axis p[k].
    """
    shp = (2,) * N
    T = H.reshape(shp + shp)
    ket_axes = p
    bra_axes = [q + N for q in p]
    perm_axes = ket_axes + bra_axes
    Tp = np.transpose(T, axes=perm_axes)
    return Tp.reshape(2**N, 2**N)


# -----------------------------
# Main experiment
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Algebra-Constrained Dimensional Accessibility (Option A: source-local scrambling)"
    )

    ap.add_argument("--out", required=True, help="Output JSON path (or directory).")
    ap.add_argument("--N", type=int, default=8, help="Number of qubits (dense matrices; keep small).")
    ap.add_argument("--seed", type=int, default=0, help="Master RNG seed.")

    ap.add_argument("--source-topology", default="1d_ring",
                    choices=["1d_ring", "1d_chain", "2d", "3d", "rr4", "rr6"],
                    help="Source topology for the initial local Hamiltonian and the scramble move set.")

    ap.add_argument("--targets", default="1d_ring,3d,rr4",
                    help="Comma-separated target geometries to test accessibility (e.g., 1d_ring,2d,3d,rr4,rr6).")

    ap.add_argument("--scramble-steps", type=int, default=40,
                    help="Number of LOCAL scramble moves on source edges (reachable scrambling).")

    ap.add_argument("--factorizations", type=int, default=60, help="Number of candidate factorizations (permutations).")

    ap.add_argument("--q-thresh", type=float, default=None,
                    help="JW-quadratic score threshold. If omitted, use --q-quantile.")
    ap.add_argument("--q-quantile", type=float, default=0.85,
                    help="If --q-thresh is omitted, set threshold at this quantile of Q distribution (0..1).")

    ap.add_argument("--runs-per-set", type=int, default=12,
                    help="Max runs per target per set (conditioned/control).")

    ap.add_argument("--steps", type=int, default=600, help="Recovery steps per run.")
    ap.add_argument("--temp0", type=float, default=0.02, help="Initial annealing temperature.")
    ap.add_argument("--temp-decay", type=float, default=0.9995, help="Temperature decay per step.")

    ap.add_argument("--mc-samples", type=int, default=1024, help="Monte Carlo Pauli samples for leak estimator.")
    ap.add_argument("--kmax", type=int, default=2, help="k-local cutoff; leak counts support size > kmax.")

    ap.add_argument("--progress", action="store_true", help="Print progress.")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    N = args.N
    dim = 2 ** N
    if N > 12:
        raise SystemExit("N>12 is too slow for dense-matrix approach in this script. Use smaller N or a tensor backend.")

    def build_edges(name: str) -> List[Tuple[int, int]]:
        if name == "1d_ring":
            return edges_1d_ring(N)
        if name == "1d_chain":
            return edges_1d_chain(N)
        if name == "2d":
            e = edges_2d_lattice(N)
            if e is None:
                raise ValueError(f"2D lattice requires N=L^2; got N={N}")
            return e
        if name == "3d":
            e = edges_3d_lattice(N)
            if e is None:
                raise ValueError(f"3D lattice requires N=L^3; got N={N}")
            return e
        if name == "rr4":
            e = make_random_regular_edges(N, 4, rng)
            if e is None:
                raise ValueError(f"Failed to generate rr4 for N={N}")
            return e
        if name == "rr6":
            e = make_random_regular_edges(N, 6, rng)
            if e is None:
                raise ValueError(f"Failed to generate rr6 for N={N}")
            return e
        raise ValueError(f"Unknown topology: {name}")

    targets = parse_list(args.targets)

    # Build source edges and initial local Hamiltonian
    src_edges = build_edges(args.source_topology)
    H0 = build_xx_hamiltonian(N, src_edges, J=1.0)

    # Reachable scramble: only local moves on source edges
    Hscr = local_scramble_on_edges(H0, N, src_edges, rng, steps=args.scramble_steps)

    # Shared MC samples for leak estimator
    pauli_samples = sample_pauli_strings(N, args.mc_samples, rng)

    # Sample candidate factorizations (permutations)
    perms: List[List[int]] = []
    for _ in range(args.factorizations):
        p = list(range(N))
        rng.shuffle(p)
        perms.append(p)

    if args.progress:
        print(f"N={N} dim={dim} source={args.source_topology} scramble_steps={args.scramble_steps}")
        print(f"factorizations={args.factorizations}  runs_per_set={args.runs_per_set}")
        print("Scoring factorizations (JW-quadratic projection on reachable-scrambled H)...")

    fact_scores: List[Tuple[List[int], float]] = []
    for idx, p in enumerate(perms):
        basis = quadratic_subspace_basis(N, p)
        Q = quadratic_projection_score(Hscr, basis)
        fact_scores.append((p, float(Q)))
        if args.progress and (idx + 1) % max(1, args.factorizations // 5) == 0:
            print(f"  scored {idx+1}/{args.factorizations}")

    qs = np.array([Q for (_, Q) in fact_scores], dtype=np.float64)

    # Determine threshold
    if args.q_thresh is None:
        q = float(np.clip(args.q_quantile, 0.0, 1.0))
        thresh = float(np.quantile(qs, q)) if len(qs) else 0.0
        thresh_source = f"quantile({q})"
    else:
        thresh = float(args.q_thresh)
        thresh_source = "explicit"

    conditioned = [(p, Q) for (p, Q) in fact_scores if Q >= thresh]
    control = fact_scores[:]

    if args.progress:
        print(f"Q stats: mean={float(np.mean(qs)):.6f} std={float(np.std(qs)):.6f} min={float(np.min(qs)):.6f} max={float(np.max(qs)):.6f}")
        print(f"Threshold ({thresh_source}) = {thresh:.6f}")
        print(f"Conditioned count: {len(conditioned)}/{len(control)} (Q >= threshold)")

    def run_suite(label: str, pool: List[Tuple[List[int], float]]) -> Dict[str, dict]:
        if label == "conditioned":
            pool_sorted = sorted(pool, key=lambda x: x[1], reverse=True)
        else:
            pool_sorted = pool[:]
            rng.shuffle(pool_sorted)

        K = min(args.runs_per_set, len(pool_sorted))
        chosen = pool_sorted[:K]

        out: Dict[str, dict] = {}
        for tgt in targets:
            edges = build_edges(tgt)

            results = []
            for run_i, (p, Q) in enumerate(chosen):
                Hf = permute_operator_qubits(Hscr, N, p)

                rr = annealed_recovery(
                    H_start=Hf,
                    N=N,
                    edges=edges,
                    rng=rng,
                    steps=args.steps,
                    temp0=args.temp0,
                    temp_decay=args.temp_decay,
                    pauli_samples=pauli_samples,
                    k_max=args.kmax,
                )

                results.append({
                    "perm": p,
                    "Q": float(Q),
                    "leak_initial": rr.leak_initial,
                    "leak_final": rr.leak_final,
                    "leak_reduction": rr.leak_reduction,
                    "accepted_moves": rr.accepted_moves,
                    "steps": rr.steps,
                })

                if args.progress:
                    print(f"[{label}] target={tgt:8s} run {run_i+1:2d}/{K}  Q={Q:.6f}  leak_red={rr.leak_reduction:.6f}")

            leak_reds = np.array([r["leak_reduction"] for r in results], dtype=np.float64)
            out[tgt] = {
                "runs": int(len(results)),
                "mean_leak_reduction": float(np.mean(leak_reds)) if len(leak_reds) else 0.0,
                "std_leak_reduction": float(np.std(leak_reds)) if len(leak_reds) else 0.0,
                "min_leak_reduction": float(np.min(leak_reds)) if len(leak_reds) else 0.0,
                "max_leak_reduction": float(np.max(leak_reds)) if len(leak_reds) else 0.0,
                "details": results,
            }

        ranking = sorted([(k, v["mean_leak_reduction"]) for k, v in out.items()],
                         key=lambda x: x[1], reverse=True)
        out["_ranking"] = [{"target": k, "mean_leak_reduction": float(m)} for (k, m) in ranking]
        return out

    suite_conditioned = run_suite("conditioned", conditioned) if len(conditioned) else {"_ranking": [], "_empty": True}
    suite_control = run_suite("control", control)

    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(args.seed),
        "N": int(N),
        "source_topology": args.source_topology,
        "targets": targets,
        "scramble": {
            "type": "local_on_source_edges",
            "scramble_steps": int(args.scramble_steps),
        },
        "factorizations": int(args.factorizations),
        "Q_threshold": {
            "source": thresh_source,
            "value": float(thresh),
            "quantile": float(args.q_quantile) if args.q_thresh is None else None,
        },
        "factorization_Q_stats": {
            "mean": float(np.mean(qs)) if len(qs) else 0.0,
            "std": float(np.std(qs)) if len(qs) else 0.0,
            "min": float(np.min(qs)) if len(qs) else 0.0,
            "max": float(np.max(qs)) if len(qs) else 0.0,
            "conditioned_count": int(len(conditioned)),
            "total": int(len(control)),
        },
        "recovery": {
            "steps": int(args.steps),
            "temp0": float(args.temp0),
            "temp_decay": float(args.temp_decay),
            "mc_samples": int(args.mc_samples),
            "kmax": int(args.kmax),
            "runs_per_set": int(args.runs_per_set),
        },
        "conditioned": suite_conditioned,
        "control": suite_control,
        "runtime_sec": float(time.time() - t0),
        "note": (
            "Option A: scramble is restricted to local 2-qubit conjugations on SOURCE edges "
            "(Paper-2 accessibility-respecting). Factorizations are qubit permutations. "
            "Algebra filter uses JW-quadratic projection score Q computed on the reachable-scrambled Hamiltonian."
        ),
    }

    out_path = args.out
    if os.path.isdir(out_path) or out_path.endswith(os.sep):
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, "algebra_constrained_dimensional_accessibility.json")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_file = out_path

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.progress:
        print(f"\nWrote: {out_file}")
        print("Conditioned ranking:", report["conditioned"].get("_ranking", []))
        print("Control ranking:    ", report["control"].get("_ranking", []))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
