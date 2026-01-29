#!/usr/bin/env python3
"""
Focused Budget Sweep: Critical Transition Region
=================================================

The initial sweep revealed the transition happens in Λ ∈ [2, 6].
This script zooms in on that region with:
- Finer budget resolution
- More seeds for statistical power
- Lower edge threshold to see structure at very low budgets

Usage:
------
python budget_sweep_focused.py --seeds 6 --progress

Expected runtime: ~20-30 minutes
"""

from __future__ import annotations

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
# Core simulation code (same as before)
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
    psi_t = psi.reshape([2] * N)
    axes = [q] + [i for i in range(N) if i != q]
    psi_perm = np.transpose(psi_t, axes).reshape(2, -1)
    rho = psi_perm @ psi_perm.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    return rho


def trace_distance_2x2(rho: np.ndarray, sigma: np.ndarray) -> float:
    delta = rho - sigma
    delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))


def apply_two_qubit_gate_statevector(psi: np.ndarray, N: int, a: int, b: int, U4: np.ndarray) -> np.ndarray:
    if a == b:
        return psi
    if a > b:
        a, b = b, a
    psi_t = psi.reshape([2] * N)
    axes = [i for i in range(N) if i not in (a, b)] + [a, b]
    inv_axes = np.argsort(axes)
    psi_perm = np.transpose(psi_t, axes)
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
# Diagnostics - with MULTIPLE thresholds
# =============================================================================

def summarize_structure_multi_thr(J: np.ndarray, thresholds: List[float] = [0.1, 0.3, 0.5]) -> Dict:
    """Count edges at multiple thresholds to see structure at low budgets."""
    N = J.shape[0]
    vals = np.abs(J[np.triu_indices(N, 1)])
    
    result = {
        "absJ_mean": float(np.mean(vals)),
        "absJ_max": float(np.max(vals)),
        "absJ_std": float(np.std(vals)),
    }
    
    for thr in thresholds:
        A = (np.abs(J) >= thr).astype(np.int32)
        np.fill_diagonal(A, 0)
        edges = int(A.sum() // 2)
        result[f"edges_thr{thr}"] = edges
    
    # Gini on all values
    vals_sorted = np.sort(vals)
    if np.all(vals_sorted < 1e-12):
        gini = 0.0
    else:
        n = len(vals_sorted)
        cum = np.cumsum(vals_sorted)
        gini = float((n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-12)) / n)
    result["gini"] = gini
    
    return result


def weighted_influence_vs_distance(
    J: np.ndarray,
    infl_samples: List[Tuple[int, int, float]],
    eps: float = 1e-6,
) -> Dict:
    N = J.shape[0]
    absJ = np.abs(J).astype(np.float64)
    np.fill_diagonal(absJ, 0.0)

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = absJ[i, j]
            if w <= 0.0:
                continue
            length = 1.0 / (w + float(eps))
            adj[i].append((j, length))

    def dijkstra(src: int) -> List[float]:
        dist = [math.inf] * N
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, wlen in adj[u]:
                nd = d + wlen
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    srcs = sorted(set(i for (i, _, _) in infl_samples))
    dist_map = {s: dijkstra(s) for s in srcs}

    ds: List[float] = []
    vs: List[float] = []
    for (i, j, val) in infl_samples:
        d = dist_map[i][j]
        if math.isfinite(d):
            ds.append(float(d))
            vs.append(float(val))

    if len(ds) == 0:
        return {"corr_wd": None, "slope_wd": None, "n_finite": 0}

    ds_arr = np.array(ds, dtype=np.float64)
    vs_arr = np.array(vs, dtype=np.float64)

    corr = None
    if np.std(ds_arr) > 1e-12 and np.std(vs_arr) > 1e-12:
        corr = float(np.corrcoef(ds_arr, vs_arr)[0, 1])

    slope = None
    if np.std(ds_arr) > 1e-12:
        A = np.vstack([np.ones_like(ds_arr), ds_arr]).T
        coeff, *_ = np.linalg.lstsq(A, vs_arr, rcond=None)
        slope = float(coeff[1])

    return {"corr_wd": corr, "slope_wd": slope, "n_finite": len(ds)}


# =============================================================================
# Core simulation
# =============================================================================

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
    influence_eps: float
    budget: float
    budget_iters: int
    seed: int
    k_steps: int
    mask_thr: float
    uniform_strength: float


def sample_edges_for_step(N: int, rng: np.random.Generator, sample: bool) -> List[Tuple[int, int]]:
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if not sample:
        edges = all_edges
    else:
        m = min(len(all_edges), max(10, (N * (N - 1)) // 4))
        idx = rng.choice(len(all_edges), size=m, replace=False)
        edges = [all_edges[k] for k in idx]
    rng.shuffle(edges)
    return edges


def evolve_one_step_with_edges(psi, J, params, edges, mode="directJ"):
    """
    mode:
      - directJ: use actual Jij for gate strength (for world evolution)
      - masked_uniform: use uniform_strength on edges where |J| > mask_thr (for probes)
    """
    N = params.N
    dt = params.dt
    Delta = params.Delta

    psi2 = psi
    
    if mode == "directJ":
        for (i, j) in edges:
            Jij = float(J[i, j])
            if abs(Jij) < 1e-12:
                continue
            U4 = two_qubit_unitary_xx_yy_zz(dt, Jij, Delta)
            psi2 = apply_two_qubit_gate_statevector(psi2, N, i, j, U4)
    elif mode == "masked_uniform":
        thr = float(params.mask_thr)
        mag = float(params.uniform_strength)
        for (i, j) in edges:
            Jij = float(J[i, j])
            if abs(Jij) <= thr:
                continue
            J_eff = float(np.sign(Jij) * mag)
            if abs(J_eff) < 1e-12:
                continue
            U4 = two_qubit_unitary_xx_yy_zz(dt, J_eff, Delta)
            psi2 = apply_two_qubit_gate_statevector(psi2, N, i, j, U4)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return normalize_state(psi2)


def evolve_k_steps(psi, J, params, rng, k, mode="masked_uniform"):
    psi2 = psi
    for _ in range(max(1, int(k))):
        edges = sample_edges_for_step(params.N, rng, sample=(params.N > 10))
        psi2 = evolve_one_step_with_edges(psi2, J, params, edges, mode=mode)
    return psi2


def estimate_influence_pair(psi, J, params, rng, src, dst):
    N = params.N
    U_rand = random_single_qubit_unitary(rng)
    eps = float(params.influence_eps)
    U_mix = (1.0 - eps) * np.eye(2, dtype=np.complex128) + eps * U_rand
    U2 = nearest_unitary(U_mix)
    psi_pert = apply_single_qubit_unitary(psi, N, src, U2)

    bitgen_state = rng.bit_generator.state

    rng_base = np.random.default_rng()
    rng_base.bit_generator.state = bitgen_state
    psi_a = evolve_k_steps(psi, J, params, rng_base, params.k_steps, mode="masked_uniform")

    rng_pert = np.random.default_rng()
    rng_pert.bit_generator.state = bitgen_state
    psi_b = evolve_k_steps(psi_pert, J, params, rng_pert, params.k_steps, mode="masked_uniform")

    rng.bit_generator.state = rng_base.bit_generator.state

    rho_a = single_qubit_rho(psi_a, N, dst)
    rho_b = single_qubit_rho(psi_b, N, dst)
    return trace_distance_2x2(rho_a, rho_b)


def update_links_learn(J, influences, params):
    eta = float(params.eta)
    decay = float(params.decay)

    J2 = (1.0 - decay) * J
    inc = np.zeros_like(J2)

    for (i, j, val) in influences:
        inc[i, j] += val
        inc[j, i] += val

    max_abs = float(np.max(np.abs(inc)))
    if max_abs > 1e-12:
        inc = inc / max_abs

    J2 = J2 + eta * inc
    J2 = 0.5 * (J2 + J2.T)
    np.fill_diagonal(J2, 0.0)
    J2 = np.clip(J2, -params.J_clip, params.J_clip)

    J2 = apply_soft_budget_symmetric(J2, params.budget, params.budget_iters)
    J2 = np.clip(J2, -params.J_clip, params.J_clip)
    np.fill_diagonal(J2, 0.0)
    return J2


def run_single(params: Params) -> Dict:
    rng = np.random.default_rng(params.seed)

    psi = random_product_state(params.N, rng)
    J = initialize_J(params.N, rng, params.J_init_scale, params.J_clip)
    J = apply_soft_budget_symmetric(J, params.budget, params.budget_iters)
    J = np.clip(J, -params.J_clip, params.J_clip)
    np.fill_diagonal(J, 0.0)

    N = params.N

    for step in range(params.steps):
        # Sample influence pairs the SAME WAY as original
        influences = []
        for _ in range(params.pairs_per_step):
            i = int(rng.integers(0, N))
            j = int(rng.integers(0, N - 1))
            if j >= i:
                j += 1
            val = estimate_influence_pair(psi, J, params, rng, i, j)
            influences.append((i, j, float(val)))

        J = update_links_learn(J, influences, params)

        # World evolution uses directJ (actual coupling strengths)
        edges = sample_edges_for_step(N, rng, sample=(N > 10))
        psi = evolve_one_step_with_edges(psi, J, params, edges, mode="directJ")

    # Final diagnostics
    all_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    n_final = min(60, len(all_pairs))
    idx = rng.choice(len(all_pairs), size=n_final, replace=False)
    final_pairs = [all_pairs[k] for k in idx]

    final_influences = []
    for (src, dst) in final_pairs:
        val = estimate_influence_pair(psi, J, params, rng, src, dst)
        final_influences.append((src, dst, val))

    struct = summarize_structure_multi_thr(J, thresholds=[0.05, 0.1, 0.2, 0.3, 0.5])
    wdist = weighted_influence_vs_distance(J, final_influences)

    return {
        "budget": params.budget,
        "seed": params.seed,
        **struct,
        **wdist,
    }


# =============================================================================
# Main driver
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Focused budget sweep in transition region")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="budget_sweep_focused")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # FOCUSED budget values in the transition region
    # Plus boundary values for reference
    budgets = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 8.0]
    
    base_params = {
        "N": args.N,
        "steps": args.steps,
        "dt": 0.05,
        "Delta": 0.0,
        "pairs_per_step": 40,
        "eta": 0.20,
        "decay": 0.01,
        "J_init_scale": 1.0,
        "J_clip": 2.5,
        "influence_eps": 0.08,
        "budget_iters": 6,
        "k_steps": 3,
        "mask_thr": 0.05,
        "uniform_strength": 1.0,
    }

    all_results = []
    total_runs = len(budgets) * args.seeds
    run_count = 0
    start_time = time.time()

    if args.progress:
        print(f"Focused Budget Sweep: {len(budgets)} budgets × {args.seeds} seeds = {total_runs} runs")
        print(f"Budget values: {budgets}")
        print("=" * 70)

    for budget in budgets:
        budget_results = []
        
        for seed_offset in range(args.seeds):
            seed = args.seed_start + seed_offset
            
            params = Params(budget=float(budget), seed=seed, **base_params)
            result = run_single(params)
            budget_results.append(result)
            all_results.append(result)
            
            run_count += 1
            
            out_path = os.path.join(args.outdir, f"budget_{budget:.2f}_seed{seed:03d}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            
            if args.progress:
                elapsed = time.time() - start_time
                eta_sec = (elapsed / run_count) * (total_runs - run_count)
                corr_str = f"{result['corr_wd']:.3f}" if result['corr_wd'] else "None"
                e05 = result.get('edges_thr0.05', '?')
                e5 = result.get('edges_thr0.5', '?')
                print(f"[{run_count:3d}/{total_runs}] Λ={budget:.1f} seed={seed} "
                      f"edges(0.05)={e05:2} edges(0.5)={e5:2} corr={corr_str:>7} "
                      f"ETA: {eta_sec/60:.1f}m")

    # Aggregate by budget
    budget_stats = []
    for budget in budgets:
        runs = [r for r in all_results if abs(r["budget"] - budget) < 0.01]
        
        def safe_mean(key):
            vals = [r[key] for r in runs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        
        def safe_std(key):
            vals = [r[key] for r in runs if r.get(key) is not None]
            return float(np.std(vals)) if vals else None
        
        budget_stats.append({
            "budget": float(budget),
            "n_runs": len(runs),
            "edges_thr0.05_mean": safe_mean("edges_thr0.05"),
            "edges_thr0.1_mean": safe_mean("edges_thr0.1"),
            "edges_thr0.2_mean": safe_mean("edges_thr0.2"),
            "edges_thr0.3_mean": safe_mean("edges_thr0.3"),
            "edges_thr0.5_mean": safe_mean("edges_thr0.5"),
            "edges_thr0.5_std": safe_std("edges_thr0.5"),
            "absJ_mean_mean": safe_mean("absJ_mean"),
            "absJ_max_mean": safe_mean("absJ_max"),
            "gini_mean": safe_mean("gini"),
            "gini_std": safe_std("gini"),
            "corr_wd_mean": safe_mean("corr_wd"),
            "corr_wd_std": safe_std("corr_wd"),
            "slope_wd_mean": safe_mean("slope_wd"),
            "slope_wd_std": safe_std("slope_wd"),
        })

    summary = {
        "meta": {
            "N": args.N,
            "steps": args.steps,
            "seeds_per_budget": args.seeds,
            "budgets": budgets,
            "total_runs": total_runs,
            "base_params": base_params,
        },
        "budget_stats": budget_stats,
        "all_results": all_results,
    }
    
    summary_path = os.path.join(args.outdir, "focused_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.progress:
        print("=" * 70)
        print(f"Complete! Total time: {(time.time() - start_time)/60:.1f} min")
        print(f"\nSaved to: {args.outdir}/")
        
        # Summary table
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Budget':>6} {'E(0.05)':>8} {'E(0.5)':>8} {'|J|_max':>8} {'Gini':>8} {'corr_wd':>10}")
        print("-" * 55)
        for s in budget_stats:
            e05 = f"{s['edges_thr0.05_mean']:.1f}" if s['edges_thr0.05_mean'] else "N/A"
            e5 = f"{s['edges_thr0.5_mean']:.1f}" if s['edges_thr0.5_mean'] else "N/A"
            jmax = f"{s['absJ_max_mean']:.3f}" if s['absJ_max_mean'] else "N/A"
            gini = f"{s['gini_mean']:.3f}" if s['gini_mean'] else "N/A"
            corr = f"{s['corr_wd_mean']:.3f}" if s['corr_wd_mean'] else "N/A"
            print(f"{s['budget']:>6.1f} {e05:>8} {e5:>8} {jmax:>8} {gini:>8} {corr:>10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())