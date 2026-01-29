#!/usr/bin/env python3
"""
Scale-Up Simulation: Emergent Geometry Hunt
===========================================
Refactored for N=16 (or higher) to find geometric correlations.

Targeting the critical transition region: Λ ∈ [2.5, 4.5]
Scale: N=16 (State vector size: 2^16 = 65,536 amplitudes)

Usage:
------
python scale_up_simulation.py --N 16 --steps 150 --seeds 4 --progress
"""

import argparse
import json
import math
import os
import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# Core Quantum / Math Utils
# =============================================================================

def normalize_state(psi: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(psi)
    if n <= 0:
        raise ValueError("State norm is zero.")
    return psi / n

def nearest_unitary(U: np.ndarray) -> np.ndarray:
    X, _, Yh = np.linalg.svd(U)
    return X @ Yh

def paulis() -> Dict[str, np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return {"I": I, "X": X, "Y": Y, "Z": Z}

def single_qubit_rho(psi: np.ndarray, N: int, q: int) -> np.ndarray:
    # Optimized for larger N: reduce reshape overhead if possible
    psi_t = psi.reshape([2] * N)
    axes = [q] + [i for i in range(N) if i != q]
    psi_perm = np.transpose(psi_t, axes).reshape(2, -1)
    rho = psi_perm @ psi_perm.conj().T
    # Trace out the rest
    return rho

def trace_distance_2x2(rho: np.ndarray, sigma: np.ndarray) -> float:
    delta = rho - sigma
    delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))

def apply_two_qubit_gate_statevector(psi: np.ndarray, N: int, a: int, b: int, U4: np.ndarray) -> np.ndarray:
    if a == b: return psi
    if a > b: a, b = b, a
    psi_t = psi.reshape([2] * N)
    axes = [i for i in range(N) if i not in (a, b)] + [a, b]
    inv_axes = np.argsort(axes)
    psi_perm = np.transpose(psi_t, axes)
    
    # Batch matrix multiplication: (Rest x 4) @ (4 x 4).T
    rest_dim = 2 ** (N - 2)
    psi_mat = psi_perm.reshape(rest_dim, 4)
    psi_mat2 = psi_mat @ U4.T
    
    psi_perm2 = psi_mat2.reshape([2] * (N - 2) + [2, 2])
    psi_t2 = np.transpose(psi_perm2, inv_axes).reshape(-1)
    return psi_t2

def apply_single_qubit_unitary(psi: np.ndarray, N: int, q: int, U2: np.ndarray) -> np.ndarray:
    psi_t = psi.reshape([2] * N)
    axes = [q] + [i for i in range(N) if i != q]
    inv_axes = np.argsort(axes)
    psi_perm = np.transpose(psi_t, axes).reshape(2, -1)
    psi_perm2 = (U2 @ psi_perm).reshape([2] + [2] * (N - 1))
    psi_t2 = np.transpose(psi_perm2, inv_axes).reshape(-1)
    return psi_t2

def two_qubit_unitary_xx_yy_zz(dt: float, J: float, Delta: float) -> np.ndarray:
    P = paulis()
    XX = np.kron(P["X"], P["X"])
    YY = np.kron(P["Y"], P["Y"])
    ZZ = np.kron(P["Z"], P["Z"])
    H = J * (XX + YY + Delta * ZZ)
    w, V = np.linalg.eigh(H)
    U = V @ np.diag(np.exp(-1j * dt * w)) @ V.conj().T
    return U

def random_product_state(N: int, rng: np.random.Generator) -> np.ndarray:
    psi = np.array([1.0 + 0j])
    for _ in range(N):
        v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        v = v / (np.linalg.norm(v) + 1e-12)
        psi = np.kron(psi, v)
    return normalize_state(psi)

def random_single_qubit_unitary(rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    Q, R = np.linalg.qr(A)
    Q = Q * np.exp(-1j * np.angle(np.diag(R)))
    return Q

# =============================================================================
# Graph & Budget Logic
# =============================================================================

def initialize_J(N: int, rng: np.random.Generator, scale: float, clip: float) -> np.ndarray:
    A = rng.standard_normal((N, N))
    J = 0.5 * (A + A.T)
    np.fill_diagonal(J, 0.0)
    J = scale * J / (np.std(J) + 1e-12)
    J = np.clip(J, -clip, clip)
    return J

def row_l1_offdiag(J: np.ndarray) -> np.ndarray:
    A = np.abs(J)
    np.fill_diagonal(A, 0.0)
    return A.sum(axis=1)

def apply_soft_budget_symmetric(J: np.ndarray, budget: float, iters: int = 6) -> np.ndarray:
    if budget <= 0:
        return J
    N = J.shape[0]
    a = np.ones(N, dtype=np.float64)
    J_work = J.copy()
    np.fill_diagonal(J_work, 0.0)
    eps = 1e-12
    for _ in range(max(1, iters)):
        S = row_l1_offdiag(J_work) + eps
        f = budget / S
        a *= f
        g = np.sqrt(np.outer(a, a))
        J_work = J * g
        np.fill_diagonal(J_work, 0.0)
        J_work = 0.5 * (J_work + J_work.T)
    return J_work

# =============================================================================
# Diagnostics (Distance & Correlation)
# =============================================================================

def weighted_influence_vs_distance(
    J: np.ndarray,
    infl_samples: List[Tuple[int, int, float]],
    eps: float = 1e-6,
) -> Dict:
    N = J.shape[0]
    absJ = np.abs(J).astype(np.float64)
    np.fill_diagonal(absJ, 0.0)

    # Build adjacency list: cost = 1.0 / weight
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j: continue
            w = absJ[i, j]
            if w <= 0.01: continue  # Filter weak links to reduce noise in "distance"
            length = 1.0 / (w + float(eps))
            adj[i].append((j, length))

    def dijkstra(src: int) -> List[float]:
        dist = [math.inf] * N
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]: continue
            for v, wlen in adj[u]:
                nd = d + wlen
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # Compute distances only for sampled pairs
    # Optimize: only run Dijkstra for unique sources in the sample
    unique_srcs = list(set(s for s, _, _ in infl_samples))
    dist_map = {s: dijkstra(s) for s in unique_srcs}

    ds: List[float] = []
    vs: List[float] = []
    
    for (i, j, val) in infl_samples:
        if i not in dist_map: continue
        d = dist_map[i][j]
        if math.isfinite(d):
            ds.append(float(d))
            vs.append(float(val))

    if len(ds) < 5:
        return {"corr_wd": None, "slope_wd": None, "n_finite": len(ds)}

    ds_arr = np.array(ds, dtype=np.float64)
    vs_arr = np.array(vs, dtype=np.float64)

    corr = float(np.corrcoef(ds_arr, vs_arr)[0, 1])
    
    # Linear regression slope
    A = np.vstack([np.ones_like(ds_arr), ds_arr]).T
    coeff, *_ = np.linalg.lstsq(A, vs_arr, rcond=None)
    slope = float(coeff[1])

    return {"corr_wd": corr, "slope_wd": slope, "n_finite": len(ds)}

def summarize_structure(J: np.ndarray) -> Dict:
    N = J.shape[0]
    vals = np.abs(J[np.triu_indices(N, 1)])
    
    # Count strong edges
    edges_03 = int(np.sum(vals > 0.3))
    
    # Gini
    vals_sorted = np.sort(vals)
    if np.sum(vals_sorted) < 1e-12:
        gini = 0.0
    else:
        n = len(vals_sorted)
        cum = np.cumsum(vals_sorted)
        gini = float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)
        
    return {
        "absJ_mean": float(np.mean(vals)),
        "absJ_max": float(np.max(vals)),
        "edges_thr0.3": edges_03,
        "gini": gini
    }

# =============================================================================
# Simulation
# =============================================================================

@dataclass
class Params:
    N: int
    steps: int
    dt: float
    pairs_per_step: int
    eta: float
    decay: float
    budget: float
    seed: int
    k_steps: int

def sample_edges_for_step(N: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    # Scale edges with N to keep evolution speed roughly constant per qubit
    # N=10 -> ~45 pairs. N=16 -> ~120 pairs.
    # We sample a subset to keep simulation fast.
    n_sample = max(10, int(N * 3)) 
    all_possible = N * (N - 1) // 2
    
    # If graph is small, take all. If large, sample.
    if all_possible <= n_sample * 1.5:
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(edges)
        return edges
        
    # Rejection sampling or index sampling
    edges = set()
    while len(edges) < n_sample:
        u = rng.integers(0, N)
        v = rng.integers(0, N)
        if u != v:
            if u > v: u, v = v, u
            edges.add((u, v))
    return list(edges)

def evolve_k_steps(psi, J, params, rng, k):
    psi2 = psi
    # Masking threshold for probe evolution (speed up)
    mask_thr = 0.05
    Delta = 0.0
    
    for _ in range(max(1, int(k))):
        edges = sample_edges_for_step(params.N, rng)
        for (i, j) in edges:
            Jij = float(J[i, j])
            if abs(Jij) < mask_thr: continue
            # Use fixed strength for probes? Or real J? 
            # Original script used "masked_uniform" for probes. Let's stick to real J for accuracy here.
            U4 = two_qubit_unitary_xx_yy_zz(params.dt, Jij, Delta)
            psi2 = apply_two_qubit_gate_statevector(psi2, params.N, i, j, U4)
    return normalize_state(psi2)

def estimate_influence_pair(psi, J, params, rng, src, dst):
    N = params.N
    eps = 0.08
    U_rand = random_single_qubit_unitary(rng)
    U_mix = (1.0 - eps) * np.eye(2, dtype=np.complex128) + eps * U_rand
    U2 = nearest_unitary(U_mix)
    
    psi_pert = apply_single_qubit_unitary(psi, N, src, U2)
    
    # Fork RNG
    bitgen_state = rng.bit_generator.state
    
    rng_base = np.random.default_rng()
    rng_base.bit_generator.state = bitgen_state
    psi_a = evolve_k_steps(psi, J, params, rng_base, params.k_steps)
    
    rng_pert = np.random.default_rng()
    rng_pert.bit_generator.state = bitgen_state
    psi_b = evolve_k_steps(psi_pert, J, params, rng_pert, params.k_steps)
    
    # Restore RNG
    rng.bit_generator.state = rng_base.bit_generator.state
    
    rho_a = single_qubit_rho(psi_a, N, dst)
    rho_b = single_qubit_rho(psi_b, N, dst)
    return trace_distance_2x2(rho_a, rho_b)

def run_simulation(params: Params):
    rng = np.random.default_rng(params.seed)
    
    # Init State
    psi = random_product_state(params.N, rng)
    
    # Init Graph
    J = initialize_J(params.N, rng, scale=1.0, clip=2.5)
    J = apply_soft_budget_symmetric(J, params.budget)
    np.fill_diagonal(J, 0.0)
    
    # Loop
    for step in range(params.steps):
        # 1. Measure Influence
        influences = []
        # Sample random pairs for learning
        for _ in range(params.pairs_per_step):
            u = rng.integers(0, params.N)
            v = rng.integers(0, params.N - 1)
            if v >= u: v += 1
            val = estimate_influence_pair(psi, J, params, rng, u, v)
            influences.append((u, v, val))
            
        # 2. Update Graph
        inc = np.zeros_like(J)
        for (u, v, val) in influences:
            inc[u, v] += val
            inc[v, u] += val
            
        max_inc = np.max(np.abs(inc))
        if max_inc > 1e-9:
            inc /= max_inc
            
        J = (1.0 - params.decay) * J + params.eta * inc
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        J = np.clip(J, -2.5, 2.5)
        J = apply_soft_budget_symmetric(J, params.budget)
        np.fill_diagonal(J, 0.0)
        
        # 3. Evolve World
        edges = sample_edges_for_step(params.N, rng)
        for (u, v) in edges:
            Jij = float(J[u, v])
            if abs(Jij) < 1e-9: continue
            U4 = two_qubit_unitary_xx_yy_zz(params.dt, Jij, 0.0)
            psi = apply_two_qubit_gate_statevector(psi, params.N, u, v, U4)
            
    # Final Analysis
    struct = summarize_structure(J)
    
    # Dense sampling for final correlation check
    # Sample 200 pairs to get good stats
    final_influences = []
    n_check = 200
    for _ in range(n_check):
        u = rng.integers(0, params.N)
        v = rng.integers(0, params.N - 1)
        if v >= u: v += 1
        val = estimate_influence_pair(psi, J, params, rng, u, v)
        final_influences.append((u, v, val))
        
    dist_stats = weighted_influence_vs_distance(J, final_influences)
    
    return {
        "budget": params.budget,
        "seed": params.seed,
        **struct,
        **dist_stats
    }

# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16, help="System size")
    ap.add_argument("--steps", type=int, default=150, help="Simulation steps")
    ap.add_argument("--seeds", type=int, default=4, help="Seeds per budget")
    ap.add_argument("--out", type=str, default="scale_up_results.json")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    
    # Critical Window Scan
    budgets = [2.5, 3.0, 3.5, 4.0, 4.5]
    
    results = []
    total = len(budgets) * args.seeds
    count = 0
    start = time.time()
    
    if args.progress:
        print(f"Starting Scale-Up Run (N={args.N})")
        print(f"Budgets: {budgets}")
        print("-" * 60)
        
    for b in budgets:
        for s in range(args.seeds):
            p = Params(
                N=args.N,
                steps=args.steps,
                dt=0.05,
                pairs_per_step=80, # higher for larger N
                eta=0.2,
                decay=0.01,
                budget=b,
                seed=s,
                k_steps=3
            )
            res = run_simulation(p)
            results.append(res)
            count += 1
            
            if args.progress:
                elapsed = time.time() - start
                rate = elapsed / count
                eta = (total - count) * rate
                corr = res.get('corr_wd')
                c_str = f"{corr:.3f}" if corr is not None else "nan"
                print(f"[{count}/{total}] B={b:.1f} S={s} | Edge>0.3: {res['edges_thr0.3']:3d} | Corr: {c_str:>6} | ETA: {eta/60:.1f}m")
                
    with open(args.out, "w") as f:
        json.dump({"meta": vars(args), "results": results}, f, indent=2)
        
    if args.progress:
        print("\nDone.")

if __name__ == "__main__":
    main()