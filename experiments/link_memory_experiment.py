#!/usr/bin/env python3
"""
Link-Memory Experiment: "History lives in the wake"
===================================================

Goal
----
Test the idea that *interaction links* carry memory: when subsystem i influences
subsystem j, the i-j link is modified, and those modifications accumulate into a
stable relational structure.

We model:
- N qubits (nodes)
- A symmetric coupling matrix J_ij (links)
- Unitary evolution under a 2-local Hamiltonian built from J_ij
- After each step, we estimate directional influence i->j by perturbing i and
  measuring distinguishability at j
- Then we update J_ij using a "link memory" rule (Hebbian-ish)

Key outputs
-----------
- Whether J becomes sparse / structured vs staying all-to-all
- Whether influence correlates with graph distance (emergent locality)
- Whether strong links stabilize over time (wake persistence)

This does NOT assume a geometric lattice. It begins with a random dense J.

Notes on scale
--------------
Statevector sim is O(2^N). N=10 is usually fine; N=12 may be slow on CPU.
Influence estimation is expensive, so we sample K directed pairs per step.

Windows run example
-------------------
python link_memory_experiment.py --N 10 --steps 200 --dt 0.05 --pairs-per-step 40 --seed 0 --out results_link_memory.json --progress
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------
# Basic quantum helpers
# ----------------------------

def normalize_state(psi: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(psi)
    if n == 0:
        raise ValueError("State norm is zero.")
    return psi / n


def kron_n(mats: List[np.ndarray]) -> np.ndarray:
    out = np.array([1.0 + 0j])
    for M in mats:
        out = np.kron(out, M)
    return out


def single_qubit_rho(psi: np.ndarray, N: int, q: int) -> np.ndarray:
    """
    Reduced density matrix for qubit q from a pure state |psi>.
    Uses reshape trick: psi -> (2, 2^(N-1)) then rho = A A†.
    """
    # Bring target qubit to front by reshaping/permuting axes
    psi_t = psi.reshape([2] * N)
    axes = [q] + [i for i in range(N) if i != q]
    psi_perm = np.transpose(psi_t, axes).reshape(2, -1)
    rho = psi_perm @ psi_perm.conj().T
    # Ensure Hermitian numerical stability
    rho = 0.5 * (rho + rho.conj().T)
    return rho


def trace_distance_2x2(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Trace distance for 2x2 density matrices:
    0.5 * ||rho - sigma||_1 via eigenvalues of Hermitian delta.
    """
    delta = rho - sigma
    delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))


def apply_two_qubit_gate_statevector(psi: np.ndarray, N: int, a: int, b: int, U4: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 gate U4 to qubits (a,b) in an N-qubit statevector psi.
    Uses tensor reshape and tensordot. Assumes qubits are in [0..N-1].
    """
    if a == b:
        return psi
    if a > b:
        a, b = b, a

    psi_t = psi.reshape([2] * N)

    # Move qubits a,b to last two axes
    axes = [i for i in range(N) if i not in (a, b)] + [a, b]
    inv_axes = np.argsort(axes)
    psi_perm = np.transpose(psi_t, axes)
    rest_dim = 2 ** (N - 2)
    psi_mat = psi_perm.reshape(rest_dim, 4)

    # Apply gate on the 4-dim subspace
    psi_mat2 = psi_mat @ U4.T  # (rest,4)

    psi_perm2 = psi_mat2.reshape([2] * (N - 2) + [2, 2])
    psi_t2 = np.transpose(psi_perm2, inv_axes).reshape(-1)

    return psi_t2


# ----------------------------
# Gates / Hamiltonian pieces
# ----------------------------

def paulis() -> Dict[str, np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def two_qubit_unitary_xx_yy_zz(dt: float, J: float, Delta: float) -> np.ndarray:
    """
    Two-qubit gate U = exp(-i dt * J * (XX + YY + Delta ZZ)).
    This is a standard interaction that supports nontrivial transport when J varies.
    """
    P = paulis()
    XX = np.kron(P["X"], P["X"])
    YY = np.kron(P["Y"], P["Y"])
    ZZ = np.kron(P["Z"], P["Z"])
    H = J * (XX + YY + Delta * ZZ)

    # Diagonalize 4x4 for fast expm
    w, V = np.linalg.eigh(H)
    U = V @ np.diag(np.exp(-1j * dt * w)) @ V.conj().T
    return U


def random_product_state(N: int, rng: np.random.Generator) -> np.ndarray:
    """Random product state over N qubits."""
    psi = np.array([1.0 + 0j])
    for _ in range(N):
        v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        v = v / np.linalg.norm(v)
        psi = np.kron(psi, v)
    return normalize_state(psi)


def random_single_qubit_unitary(rng: np.random.Generator) -> np.ndarray:
    """Random 2x2 unitary via QR."""
    A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    Q, R = np.linalg.qr(A)
    Q = Q * np.exp(-1j * np.angle(np.diag(R)))
    return Q


def apply_single_qubit_unitary(psi: np.ndarray, N: int, q: int, U2: np.ndarray) -> np.ndarray:
    """Apply 2x2 unitary to qubit q by decomposing into two-qubit ops is overkill; do reshape method."""
    psi_t = psi.reshape([2] * N)
    axes = [q] + [i for i in range(N) if i != q]
    inv_axes = np.argsort(axes)
    psi_perm = np.transpose(psi_t, axes).reshape(2, -1)
    psi_perm2 = (U2 @ psi_perm).reshape([2] + [2] * (N - 1))
    psi_t2 = np.transpose(psi_perm2, inv_axes).reshape(-1)
    return psi_t2


# ----------------------------
# Experiment core
# ----------------------------

@dataclass
class Params:
    N: int
    steps: int
    dt: float
    Delta: float
    pairs_per_step: int
    eta: float
    decay: float
    J_init_scale: float
    J_clip: float
    lock_threshold: float
    lock_strength: float
    influence_eps: float
    seed: int


def initialize_J(N: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Symmetric J with zeros on diagonal."""
    A = rng.standard_normal((N, N))
    J = 0.5 * (A + A.T)
    np.fill_diagonal(J, 0.0)
    # Normalize
    J = scale * J / (np.std(J) + 1e-12)
    return J


def evolve_one_step(psi: np.ndarray, J: np.ndarray, params: Params, rng: np.random.Generator) -> np.ndarray:
    """
    One Trotter step: apply two-qubit gates for a sampled set of edges.
    For simplicity, we apply *all* pairs i<j each step if N is small; otherwise sample.
    """
    N = params.N
    dt = params.dt
    Delta = params.Delta

    # For N<=10, applying all pairs is okay; for bigger, sample
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if N <= 10:
        edges = all_edges
    else:
        # sample ~N*(N-1)/4 edges per step
        m = min(len(all_edges), max(10, (N * (N - 1)) // 4))
        idx = rng.choice(len(all_edges), size=m, replace=False)
        edges = [all_edges[k] for k in idx]

    # Randomize application order to avoid bias
    rng.shuffle(edges)

    psi2 = psi
    for (i, j) in edges:
        Jij = float(J[i, j])
        if abs(Jij) < 1e-12:
            continue
        U4 = two_qubit_unitary_xx_yy_zz(dt, Jij, Delta)
        psi2 = apply_two_qubit_gate_statevector(psi2, N, i, j, U4)

    return normalize_state(psi2)


def estimate_influence_pair(psi: np.ndarray, J: np.ndarray, params: Params, rng: np.random.Generator, src: int, dst: int) -> float:
    """
    Estimate directional influence src -> dst:
    - Apply a small random unitary to src (perturb)
    - Evolve one step with current J
    - Compare reduced state at dst between perturbed/unperturbed
    """
    N = params.N

    # Small perturbation: random U, mixed with identity by influence_eps
    U_rand = random_single_qubit_unitary(rng)
    eps = params.influence_eps
    U2 = normalize_state((1 - eps) * np.eye(2, dtype=np.complex128) + eps * U_rand)

    psi_pert = apply_single_qubit_unitary(psi, N, src, U2)

    psi_a = evolve_one_step(psi, J, params, rng)
    psi_b = evolve_one_step(psi_pert, J, params, rng)

    rho_a = single_qubit_rho(psi_a, N, dst)
    rho_b = single_qubit_rho(psi_b, N, dst)

    return trace_distance_2x2(rho_a, rho_b)


def update_links(J: np.ndarray, influences: List[Tuple[int, int, float]], params: Params) -> np.ndarray:
    """
    Link-memory update:
    J_ij <- (1-decay)*J_ij + eta * influence(i->j) (symmetrized)
    plus a no-refolding-ish lock: strong links resist weakening below lock_threshold.
    """
    N = params.N
    eta = params.eta
    decay = params.decay

    J2 = (1.0 - decay) * J

    # Accumulate directional influences into symmetric increments
    inc = np.zeros_like(J2)
    for (i, j, val) in influences:
        inc[i, j] += val
        # we also push the symmetric edge; you can remove this if you want directed links
        inc[j, i] += val

    # Normalize increment scale so eta is meaningful across different samples
    if influences:
        inc = inc / (np.max(np.abs(inc)) + 1e-12)

    J2 = J2 + eta * inc
    np.fill_diagonal(J2, 0.0)

    # Clip
    J2 = np.clip(J2, -params.J_clip, params.J_clip)

    # No-refolding-ish: if an edge was strong, resist dropping it
    # This is a *soft anchor* rather than an absolute freeze.
    lock = np.abs(J) >= params.lock_threshold
    if np.any(lock):
        # Pull locked edges back toward previous magnitude
        J2[lock] = (1.0 - params.lock_strength) * J2[lock] + params.lock_strength * J[lock]

    return J2


def summarize_structure(J: np.ndarray, thr: float = 0.5) -> Dict:
    """
    Cheap graph-like summaries from J:
    - sparsity above threshold
    - degree distribution (count edges with |J|>=thr)
    - Gini-ish inequality of |J|
    """
    N = J.shape[0]
    A = (np.abs(J) >= thr).astype(np.int32)
    np.fill_diagonal(A, 0)

    degrees = A.sum(axis=1).tolist()
    m = int(A.sum() // 2)

    vals = np.abs(J[np.triu_indices(N, 1)])
    vals_sorted = np.sort(vals)
    # Gini coefficient (inequality): 0 uniform, 1 very concentrated
    if np.all(vals_sorted < 1e-12):
        gini = 0.0
    else:
        n = len(vals_sorted)
        cum = np.cumsum(vals_sorted)
        gini = float((n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-12)) / n)

    return {
        "thr": float(thr),
        "edges_ge_thr": m,
        "deg_min": int(np.min(degrees)),
        "deg_max": int(np.max(degrees)),
        "deg_mean": float(np.mean(degrees)),
        "deg_list": degrees,
        "gini_absJ": gini,
        "absJ_mean": float(np.mean(vals)),
        "absJ_std": float(np.std(vals)),
    }


def influence_vs_distance(J: np.ndarray, infl_samples: List[Tuple[int, int, float]], thr: float = 0.5) -> Dict:
    """
    Compare influence to graph distance on thresholded graph |J|>=thr.
    We compute unweighted shortest path distances; if disconnected, dist=inf.
    """
    N = J.shape[0]
    A = (np.abs(J) >= thr).astype(np.int32)
    np.fill_diagonal(A, 0)

    # BFS distances from each node as needed
    def bfs(src: int) -> List[float]:
        dist = [math.inf] * N
        dist[src] = 0
        q = [src]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in np.where(A[u] > 0)[0]:
                if dist[v] == math.inf:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # Precompute distances for sources that appear
    srcs = sorted(set(i for (i, _, _) in infl_samples))
    dist_map = {s: bfs(s) for s in srcs}

    pairs = []
    for (i, j, val) in infl_samples:
        d = dist_map[i][j]
        pairs.append((d, val))

    finite = [(d, v) for (d, v) in pairs if math.isfinite(d)]
    if not finite:
        return {"thr": float(thr), "n": len(pairs), "n_finite": 0}

    ds = np.array([d for (d, _) in finite], dtype=np.float64)
    vs = np.array([v for (_, v) in finite], dtype=np.float64)

    # Correlation between distance and influence (expect negative if locality emerges)
    if np.std(ds) < 1e-12 or np.std(vs) < 1e-12:
        corr = None
    else:
        corr = float(np.corrcoef(ds, vs)[0, 1])

    # Bin means by distance
    out_bins = {}
    for d in sorted(set(ds.tolist())):
        mask = ds == d
        out_bins[int(d)] = float(np.mean(vs[mask]))

    return {
        "thr": float(thr),
        "n": len(pairs),
        "n_finite": int(len(finite)),
        "corr_dist_influence": corr,
        "mean_influence_by_dist": out_bins,
    }


def run(params: Params) -> Dict:
    rng = np.random.default_rng(params.seed)

    psi = random_product_state(params.N, rng)
    J = initialize_J(params.N, rng, params.J_init_scale)

    history = []
    infl_buffer_last = []

    for step in range(params.steps):
        # Sample directed pairs (src != dst)
        infl_samples = []
        for _ in range(params.pairs_per_step):
            i = int(rng.integers(0, params.N))
            j = int(rng.integers(0, params.N - 1))
            if j >= i:
                j += 1
            val = estimate_influence_pair(psi, J, params, rng, i, j)
            infl_samples.append((i, j, float(val)))

        infl_buffer_last = infl_samples

        # Update links from influence (this is the "wake memory")
        J = update_links(J, infl_samples, params)

        # Evolve actual state under the new links
        psi = evolve_one_step(psi, J, params, rng)

        if (step % max(1, params.steps // 10) == 0) or (step == params.steps - 1):
            struct = summarize_structure(J, thr=0.5)
            dist_cmp = influence_vs_distance(J, infl_samples, thr=0.5)
            history.append({
                "step": step,
                "structure": struct,
                "dist_cmp": dist_cmp,
                "mean_influence": float(np.mean([v for (_, _, v) in infl_samples])),
                "max_influence": float(np.max([v for (_, _, v) in infl_samples])),
            })

    final_struct = summarize_structure(J, thr=0.5)
    final_dist = influence_vs_distance(J, infl_buffer_last, thr=0.5)

    return {
        "meta": vars(params),
        "final": {
            "structure": final_struct,
            "dist_cmp": final_dist,
        },
        "history": history,
        "J_final": J.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--Delta", type=float, default=0.0, help="Interaction anisotropy in XX+YY+Delta ZZ")
    ap.add_argument("--pairs-per-step", type=int, default=40)
    ap.add_argument("--eta", type=float, default=0.15, help="Link update learning rate")
    ap.add_argument("--decay", type=float, default=0.01, help="Link decay per step")
    ap.add_argument("--J-init-scale", type=float, default=1.0)
    ap.add_argument("--J-clip", type=float, default=2.5)
    ap.add_argument("--lock-threshold", type=float, default=1.25, help="Edges with |J|>=threshold resist weakening")
    ap.add_argument("--lock-strength", type=float, default=0.25, help="0..1 anchor strength for locked edges")
    ap.add_argument("--influence-eps", type=float, default=0.08, help="Perturbation strength for influence probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results_link_memory.json")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    params = Params(
        N=args.N,
        steps=args.steps,
        dt=args.dt,
        Delta=args.Delta,
        pairs_per_step=args.pairs_per_step,
        eta=args.eta,
        decay=args.decay,
        J_init_scale=args.J_init_scale,
        J_clip=args.J_clip,
        lock_threshold=args.lock_threshold,
        lock_strength=args.lock_strength,
        influence_eps=args.influence_eps,
        seed=args.seed,
    )

    result = run(params)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if args.progress:
        fin = result["final"]
        print("FINAL STRUCTURE:", fin["structure"])
        print("FINAL DIST/INFL:", fin["dist_cmp"])
        print("Wrote:", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
