#!/usr/bin/env python3
"""
Budget Sweep Experiment
=======================

Sweeps the per-node capacity budget to map the locality phase transition.

Hypothesis: Lower budget → stronger locality (more pruning forces local solutions)
           Higher budget → weaker locality (approaches unconstrained case)

Usage:
------
python budget_sweep.py --seeds 8 --progress

Output:
-------
- budget_sweep_results/sweep_SUMMARY.json  (aggregate statistics)
- budget_sweep_results/budget_X.XX_seedYYY.json  (individual runs)
- budget_sweep_results/sweep_plot_data.json  (for easy plotting)

Expected runtime: ~15-30 minutes depending on hardware
"""

from __future__ import annotations

import argparse
import json
import math
import os
import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# =============================================================================
# Helpers (copied from original experiment)
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


# =============================================================================
# State operations
# =============================================================================

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


# =============================================================================
# Gates / interactions
# =============================================================================

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
# Budget operations
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
# Diagnostics
# =============================================================================

def summarize_structure(J: np.ndarray, thr: float = 0.5) -> Dict:
    N = J.shape[0]
    A = (np.abs(J) >= thr).astype(np.int32)
    np.fill_diagonal(A, 0)
    degrees = A.sum(axis=1).tolist()
    m = int(A.sum() // 2)

    vals = np.abs(J[np.triu_indices(N, 1)])
    vals_sorted = np.sort(vals)

    if np.all(vals_sorted < 1e-12):
        gini = 0.0
    else:
        n = len(vals_sorted)
        cum = np.cumsum(vals_sorted)
        gini = float((n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-12)) / n)

    rl1 = row_l1_offdiag(J)
    return {
        "edges_ge_thr": m,
        "deg_mean": float(np.mean(degrees)),
        "gini_absJ": float(gini),
        "rowL1_mean": float(np.mean(rl1)),
    }


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
        return {"corr_wdist_influence": None, "slope_influence_vs_wdist": None}

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

    return {"corr_wdist_influence": corr, "slope_influence_vs_wdist": slope}


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
    probe_mode: str
    k_steps: int
    mask_thr: float
    uniform_strength: float


def sample_edges_for_step(N: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    m = min(len(all_edges), max(10, (N * (N - 1)) // 4))
    idx = rng.choice(len(all_edges), size=m, replace=False)
    edges = [all_edges[k] for k in idx]
    rng.shuffle(edges)
    return edges


def evolve_one_step_with_edges(
    psi: np.ndarray,
    J: np.ndarray,
    params: Params,
    edges: List[Tuple[int, int]],
) -> np.ndarray:
    N = params.N
    dt = params.dt
    Delta = params.Delta
    thr = float(params.mask_thr)
    mag = float(params.uniform_strength)

    psi2 = psi
    for (i, j) in edges:
        Jij = float(J[i, j])
        if abs(Jij) <= thr:
            continue
        J_eff = float(np.sign(Jij) * mag)
        if abs(J_eff) < 1e-12:
            continue
        U4 = two_qubit_unitary_xx_yy_zz(dt, J_eff, Delta)
        psi2 = apply_two_qubit_gate_statevector(psi2, N, i, j, U4)

    return normalize_state(psi2)


def evolve_k_steps(
    psi: np.ndarray,
    J: np.ndarray,
    params: Params,
    rng: np.random.Generator,
    k: int,
) -> np.ndarray:
    psi2 = psi
    for _ in range(max(1, int(k))):
        edges = sample_edges_for_step(params.N, rng)
        psi2 = evolve_one_step_with_edges(psi2, J, params, edges)
    return psi2


def estimate_influence_pair(
    psi: np.ndarray,
    J: np.ndarray,
    params: Params,
    rng: np.random.Generator,
    src: int,
    dst: int,
) -> float:
    N = params.N
    U_rand = random_single_qubit_unitary(rng)
    eps = float(params.influence_eps)
    U_mix = (1.0 - eps) * np.eye(2, dtype=np.complex128) + eps * U_rand
    U2 = nearest_unitary(U_mix)
    psi_pert = apply_single_qubit_unitary(psi, N, src, U2)

    bitgen_state = rng.bit_generator.state

    rng_base = np.random.default_rng()
    rng_base.bit_generator.state = bitgen_state
    psi_a = evolve_k_steps(psi, J, params, rng_base, params.k_steps)

    rng_pert = np.random.default_rng()
    rng_pert.bit_generator.state = bitgen_state
    psi_b = evolve_k_steps(psi_pert, J, params, rng_pert, params.k_steps)

    rng.bit_generator.state = rng_base.bit_generator.state

    rho_a = single_qubit_rho(psi_a, N, dst)
    rho_b = single_qubit_rho(psi_b, N, dst)
    return trace_distance_2x2(rho_a, rho_b)


def update_links_learn(
    J: np.ndarray,
    influences: List[Tuple[int, int, float]],
    params: Params,
) -> np.ndarray:
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
    """Run a single budget/seed combination."""
    rng = np.random.default_rng(params.seed)

    psi = random_product_state(params.N, rng)
    J = initialize_J(params.N, rng, params.J_init_scale, params.J_clip)
    J = apply_soft_budget_symmetric(J, params.budget, params.budget_iters)

    N = params.N

    for step in range(params.steps):
        # Sample pairs for influence measurement
        all_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        n_pairs = min(params.pairs_per_step, len(all_pairs))
        idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]

        # Measure influence
        influences = []
        for (src, dst) in pairs:
            val = estimate_influence_pair(psi, J, params, rng, src, dst)
            influences.append((src, dst, val))

        # Update links (learn mode)
        J = update_links_learn(J, influences, params)

        # Evolve state
        edges = sample_edges_for_step(N, rng)
        psi = evolve_one_step_with_edges(psi, J, params, edges)

    # Final diagnostics
    all_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    n_final = min(60, len(all_pairs))
    idx = rng.choice(len(all_pairs), size=n_final, replace=False)
    final_pairs = [all_pairs[k] for k in idx]

    final_influences = []
    for (src, dst) in final_pairs:
        val = estimate_influence_pair(psi, J, params, rng, src, dst)
        final_influences.append((src, dst, val))

    struct = summarize_structure(J, thr=0.5)
    wdist = weighted_influence_vs_distance(J, final_influences)

    return {
        "budget": params.budget,
        "seed": params.seed,
        "edges": struct["edges_ge_thr"],
        "deg_mean": struct["deg_mean"],
        "gini": struct["gini_absJ"],
        "corr_wd": wdist["corr_wdist_influence"],
        "slope_wd": wdist["slope_influence_vs_wdist"],
    }


# =============================================================================
# Sweep driver
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Budget sweep to map locality phase transition")
    ap.add_argument("--N", type=int, default=10, help="Number of qubits")
    ap.add_argument("--steps", type=int, default=200, help="Evolution steps per run")
    ap.add_argument("--seeds", type=int, default=8, help="Seeds per budget value")
    ap.add_argument("--seed-start", type=int, default=0, help="Starting seed")
    ap.add_argument("--budget-min", type=float, default=1.0, help="Minimum budget")
    ap.add_argument("--budget-max", type=float, default=12.0, help="Maximum budget")
    ap.add_argument("--budget-steps", type=int, default=12, help="Number of budget values")
    ap.add_argument("--outdir", type=str, default="budget_sweep_results", help="Output directory")
    ap.add_argument("--progress", action="store_true", help="Print progress")
    args = ap.parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Budget values to sweep
    budgets = np.linspace(args.budget_min, args.budget_max, args.budget_steps)
    
    # Fixed parameters
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
        "probe_mode": "masked_uniform",
        "k_steps": 3,
        "mask_thr": 0.05,
        "uniform_strength": 1.0,
    }

    all_results = []
    total_runs = len(budgets) * args.seeds
    run_count = 0
    start_time = time.time()

    if args.progress:
        print(f"Budget Sweep: {len(budgets)} budget values × {args.seeds} seeds = {total_runs} runs")
        print(f"Budget range: [{args.budget_min:.1f}, {args.budget_max:.1f}]")
        print(f"N = {args.N}, steps = {args.steps}")
        print("=" * 70)

    for budget in budgets:
        budget_results = []
        
        for seed_offset in range(args.seeds):
            seed = args.seed_start + seed_offset
            
            params = Params(
                budget=float(budget),
                seed=seed,
                **base_params
            )
            
            result = run_single(params)
            budget_results.append(result)
            all_results.append(result)
            
            run_count += 1
            
            # Save individual result
            out_path = os.path.join(args.outdir, f"budget_{budget:.2f}_seed{seed:03d}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            
            if args.progress:
                elapsed = time.time() - start_time
                eta_sec = (elapsed / run_count) * (total_runs - run_count)
                corr_str = f"{result['corr_wd']:.3f}" if result['corr_wd'] else "None"
                print(f"[{run_count:3d}/{total_runs}] budget={budget:.2f} seed={seed:2d} "
                      f"edges={result['edges']:2d} corr_wd={corr_str:>7s} "
                      f"ETA: {eta_sec/60:.1f}min")

        # Compute budget-level statistics
        edges_vals = [r["edges"] for r in budget_results]
        gini_vals = [r["gini"] for r in budget_results]
        corr_vals = [r["corr_wd"] for r in budget_results if r["corr_wd"] is not None]
        
        if args.progress:
            corr_mean = np.mean(corr_vals) if corr_vals else float('nan')
            print(f"  → Budget {budget:.2f}: edges={np.mean(edges_vals):.1f}±{np.std(edges_vals):.1f}, "
                  f"corr_wd={corr_mean:.3f}±{np.std(corr_vals):.3f}")

    # Aggregate statistics by budget
    budget_stats = []
    for budget in budgets:
        budget_runs = [r for r in all_results if abs(r["budget"] - budget) < 0.01]
        
        edges_vals = [r["edges"] for r in budget_runs]
        gini_vals = [r["gini"] for r in budget_runs]
        corr_vals = [r["corr_wd"] for r in budget_runs if r["corr_wd"] is not None]
        slope_vals = [r["slope_wd"] for r in budget_runs if r["slope_wd"] is not None]
        
        budget_stats.append({
            "budget": float(budget),
            "n_runs": len(budget_runs),
            "edges_mean": float(np.mean(edges_vals)),
            "edges_std": float(np.std(edges_vals)),
            "gini_mean": float(np.mean(gini_vals)),
            "gini_std": float(np.std(gini_vals)),
            "corr_wd_mean": float(np.mean(corr_vals)) if corr_vals else None,
            "corr_wd_std": float(np.std(corr_vals)) if corr_vals else None,
            "slope_wd_mean": float(np.mean(slope_vals)) if slope_vals else None,
            "slope_wd_std": float(np.std(slope_vals)) if slope_vals else None,
        })

    # Save summary
    summary = {
        "meta": {
            "N": args.N,
            "steps": args.steps,
            "seeds_per_budget": args.seeds,
            "budget_min": args.budget_min,
            "budget_max": args.budget_max,
            "budget_steps": args.budget_steps,
            "total_runs": total_runs,
            "base_params": base_params,
        },
        "budget_stats": budget_stats,
        "all_results": all_results,
    }
    
    summary_path = os.path.join(args.outdir, "sweep_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save plot-ready data
    plot_data = {
        "budgets": [s["budget"] for s in budget_stats],
        "edges_mean": [s["edges_mean"] for s in budget_stats],
        "edges_std": [s["edges_std"] for s in budget_stats],
        "gini_mean": [s["gini_mean"] for s in budget_stats],
        "gini_std": [s["gini_std"] for s in budget_stats],
        "corr_wd_mean": [s["corr_wd_mean"] for s in budget_stats],
        "corr_wd_std": [s["corr_wd_std"] for s in budget_stats],
    }
    
    plot_path = os.path.join(args.outdir, "sweep_plot_data.json")
    with open(plot_path, "w") as f:
        json.dump(plot_data, f, indent=2)

    if args.progress:
        print("=" * 70)
        print(f"Sweep complete! Total time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Results saved to: {args.outdir}/")
        print(f"  - sweep_SUMMARY.json (full statistics)")
        print(f"  - sweep_plot_data.json (for plotting)")
        print(f"  - budget_X.XX_seedYYY.json (individual runs)")
        
        # Quick summary table
        print("\n" + "=" * 70)
        print("QUICK SUMMARY")
        print("=" * 70)
        print(f"{'Budget':>8} {'Edges':>12} {'Gini':>12} {'corr_wd':>14}")
        print("-" * 50)
        for s in budget_stats:
            corr_str = f"{s['corr_wd_mean']:.3f}±{s['corr_wd_std']:.2f}" if s['corr_wd_mean'] else "N/A"
            print(f"{s['budget']:>8.2f} "
                  f"{s['edges_mean']:>6.1f}±{s['edges_std']:<4.1f} "
                  f"{s['gini_mean']:>6.3f}±{s['gini_std']:<5.3f} "
                  f"{corr_str:>14}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())