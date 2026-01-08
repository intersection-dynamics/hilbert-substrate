#!/usr/bin/env python3
"""
Emergent Geometry Analysis for HIP Framework

Computes distance matrices from information propagation data and performs:
  - MDS embedding stress vs dimension (representation test)
  - Ball-growth (Hausdorff-like) dimension estimate (intrinsic test)

Two modes:
1) Quick proxy analysis from results.npz (intensity correlations)
2) Full analysis computing pairwise permeabilities T_{i->j}(t) and derived distances

NEW:
  - Ball-growth dimension D_H from distance matrices (log|B(r)| vs log r)

Usage:
    python emergent_geometry_analysis.py --results path\to\results.npz
    python emergent_geometry_analysis.py --full --n_nodes 10
    python emergent_geometry_analysis.py --analyze-perm path\to\permeability_matrices.npz

Author: Ben Bray (+ additions)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

# Optional: for full simulation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# DISTANCE DEFINITIONS
# ============================================================

def distance_from_inverse_permeability(T_matrix, eps=1e-12):
    """
    d(i,j) = 1 / max(T_{i->j}, T_{j->i})  (symmetrized)
    """
    n = T_matrix.shape[0]
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            Tij = max(T_matrix[i, j], T_matrix[j, i])
            if Tij > eps:
                d = 1.0 / Tij
                D[i, j] = d
                D[j, i] = d
    return D


def distance_from_propagation_time(T_time_series, times, epsilon=0.1):
    """
    d(i,j) = inf{ t : max(T_{i->j}(t), T_{j->i}(t)) > epsilon }
    """
    n_times, n, _ = T_time_series.shape
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            d_ij = np.inf
            for t_idx, t in enumerate(times):
                Tij = max(T_time_series[t_idx, i, j], T_time_series[t_idx, j, i])
                if Tij > epsilon:
                    d_ij = float(t)
                    break
            D[i, j] = d_ij
            D[j, i] = d_ij
    return D


def distance_from_intensity_correlation(intensities):
    """
    Proxy: d(i,j) = 1 - corr(I_i(t), I_j(t))
    """
    corr = np.corrcoef(intensities.T)
    D = 1.0 - corr
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def graph_distance(G):
    """Shortest path distance on the unweighted graph."""
    n = G.number_of_nodes()
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    for i in range(n):
        for j, d in lengths[i].items():
            D[i, j] = float(d)
    return D


# ============================================================
# MDS ANALYSIS
# ============================================================

def mds_analysis(D, max_dim=6, n_init=10):
    """
    MDS on a precomputed distance matrix.
    Returns embeddings and stress per dimension.
    """
    # Replace inf with a large finite value
    D_f = D.copy()
    finite = np.isfinite(D_f)
    if not finite.any():
        raise ValueError("All distances are infinite; cannot run MDS.")
    max_finite = D_f[finite].max()
    D_f[~finite] = max_finite * 10.0

    embeddings = {}
    stress = {}

    for dim in range(1, max_dim + 1):
        mds = MDS(
            n_components=dim,
            dissimilarity="precomputed",
            normalized_stress="auto",
            n_init=n_init,
            random_state=42,
        )
        coords = mds.fit_transform(D_f)
        embeddings[dim] = coords
        stress[dim] = float(mds.stress_)
        print(f"  Dimension {dim}: stress = {mds.stress_:.6f}")

    return embeddings, stress


# ============================================================
# BALL-GROWTH DIMENSION (INTRINSIC)
# ============================================================

def ball_growth_counts(D, centers=None, radii=None):
    """
    Compute mean ball volume |B(r)| averaged over chosen centers.
    B_i(r) = { j : D(i,j) <= r }.
    """
    n = D.shape[0]
    if centers is None:
        centers = list(range(n))

    finite_vals = D[np.isfinite(D) & (D > 0)]
    if finite_vals.size == 0:
        return np.array([]), np.array([])

    if radii is None:
        # Use a compressed set of radii from percentiles to avoid huge unique sets.
        qs = np.linspace(5, 95, 19)
        radii = np.unique(np.quantile(finite_vals, qs))
        radii = radii[radii > 0]

    vols = []
    for r in radii:
        v = []
        for i in centers:
            v.append(np.sum(D[i, :] <= r))
        vols.append(np.mean(v))
    return np.array(radii, dtype=np.float64), np.array(vols, dtype=np.float64)


def fit_ball_growth_dimension(radii, vols, n_nodes, min_vol=2, max_vol=None):
    """
    Fit log(vol) = a + D_H * log(r) over a stable middle region.
    Returns D_H, intercept, mask_used, local_slopes.
    """
    if max_vol is None:
        max_vol = n_nodes - 1

    # Valid region: positive r, volume between [min_vol, max_vol]
    mask = (radii > 0) & (vols >= min_vol) & (vols <= max_vol)
    if np.sum(mask) < 3:
        return np.nan, np.nan, mask, np.array([])

    x = np.log(radii[mask])
    y = np.log(vols[mask])

    # Linear fit
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # Local slopes for inspection (finite diff on masked region)
    local_slopes = np.gradient(y, x)

    return float(slope), float(intercept), mask, local_slopes


def plot_ball_growth(radii, vols, slope, intercept, mask, output_path, title):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(radii, vols, "o-", linewidth=2)
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Mean ball volume |B(r)|")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add log-log fit overlay (in log space, but we can plot in log-log axes)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(radii, vols, "o-", linewidth=2, label="data")

    if np.isfinite(slope):
        r_fit = radii[mask]
        v_fit = np.exp(intercept) * (r_fit ** slope)
        ax2.plot(r_fit, v_fit, "--", linewidth=2, label=f"fit: D_H={slope:.3f}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Radius r (log)")
    ax2.set_ylabel("|B(r)| (log)")
    ax2.set_title(title + " (log-log)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(output_path.replace(".png", "_linear.png"), dpi=160)
    fig2.savefig(output_path, dpi=160)
    plt.close(fig)
    plt.close(fig2)


def plot_ball_growth_slopes(radii, mask, local_slopes, output_path, title):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if local_slopes.size == 0:
        return

    r_m = radii[mask]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r_m, local_slopes, "o-", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Radius r (log)")
    ax.set_ylabel("Local slope d log|B| / d log r")
    ax.set_title(title + " (local slopes)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ============================================================
# FULL PERMEABILITY COMPUTATION (requires PyTorch)
# ============================================================

def compute_full_permeability_matrix(n_nodes, edge_prob, times, h, J,
                                     n_state_samples, n_operator_samples,
                                     seed, device_str="cuda"):
    """
    Compute full T_{i->j}(t) matrices for all node pairs and times.
    Returns (T_matrices, G).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for full permeability computation")

    from hip_cuda_simulation import (
        get_device, build_heisenberg_hamiltonian, time_evolution_operator,
        typical_permeability
    )

    device = get_device(force_cpu=(device_str == "cpu"))
    print(f"Using device: {device}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed + 1)

    print(f"Graph: {n_nodes} nodes, {G.number_of_edges()} edges")

    H = build_heisenberg_hamiltonian(G, device, h, J)

    T_matrices = np.zeros((len(times), n_nodes, n_nodes), dtype=np.float64)

    for t_idx, t in enumerate(times):
        print(f"\nTime t = {t:.3f} ({t_idx+1}/{len(times)})")
        U = time_evolution_operator(H, float(t))

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                T_matrices[t_idx, i, j] = typical_permeability(
                    H, U, G, i, j, n_state_samples, n_operator_samples, device
                )
            print(f"  Source {i} done")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return T_matrices, G


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_ball_growth_suite(name, D, output_dir):
    n = D.shape[0]
    radii, vols = ball_growth_counts(D)

    if radii.size == 0:
        print(f"[ball-growth] {name}: insufficient finite distances.")
        return {"D_H": np.nan}

    D_H, intercept, mask, local_slopes = fit_ball_growth_dimension(radii, vols, n_nodes=n)

    print(f"[ball-growth] {name}: D_H ≈ {D_H:.4f} (fit points={np.sum(mask)})")

    plot_ball_growth(
        radii, vols, D_H, intercept, mask,
        output_path=os.path.join(output_dir, f"ball_growth_{name}.png"),
        title=f"Ball growth: {name} (D_H ≈ {D_H:.3f})"
    )
    plot_ball_growth_slopes(
        radii, mask, local_slopes,
        output_path=os.path.join(output_dir, f"ball_growth_slopes_{name}.png"),
        title=f"Ball growth slopes: {name}"
    )

    return {"D_H": D_H, "radii": radii, "vols": vols}


def analyze_from_results_file(results_path, output_dir):
    """
    Quick proxy analysis using intensity correlation distance.
    """
    print("=" * 60)
    print("EMERGENT GEOMETRY ANALYSIS (proxy: intensity correlations)")
    print("=" * 60)

    data = np.load(results_path, allow_pickle=True)
    intensities = data["intensities"]
    n_times, n_nodes = intensities.shape
    print(f"\nLoaded intensities: {n_nodes} nodes, {n_times} time points")

    D_corr = distance_from_intensity_correlation(intensities)

    print("\nMDS analysis (proxy distance)...")
    embeddings, stress = mds_analysis(D_corr, max_dim=min(6, n_nodes - 1))

    print("\nBall-growth dimension (proxy distance)...")
    bg = run_ball_growth_suite("corr_proxy", D_corr, output_dir)

    print("\nOutput saved to:", output_dir)
    return {"D_corr": D_corr, "stress": stress, "ball_growth": bg}


def analyze_from_permeability_npz(perm_path, output_dir, epsilon=0.1):
    """
    Analyze a saved permeability_matrices.npz (no recompute).
    """
    print("=" * 60)
    print("EMERGENT GEOMETRY ANALYSIS (from permeability_matrices.npz)")
    print("=" * 60)

    data = np.load(perm_path, allow_pickle=True)
    T_matrices = data["T_matrices"]
    times = data["times"]
    n_nodes = int(data["n_nodes"])
    print(f"\nLoaded T_matrices: times={len(times)}, nodes={n_nodes}")

    # Distances
    T_final = T_matrices[-1]
    D_inv = distance_from_inverse_permeability(T_final)
    D_prop = distance_from_propagation_time(T_matrices, times, epsilon=epsilon)

    # If graph was not saved, we can't compute D_graph here.
    # Still fine; the intrinsic tests don't require it.

    # MDS
    for name, D in [("inverse_perm", D_inv), ("propagation", D_prop)]:
        print(f"\n--- MDS: {name} ---")
        _emb, _stress = mds_analysis(D, max_dim=min(6, n_nodes - 1))

        print(f"\n--- Ball-growth: {name} ---")
        run_ball_growth_suite(name, D, output_dir)

    print("\nOutput saved to:", output_dir)


def full_analysis(n_nodes, edge_prob, times, h, J,
                  n_state_samples, n_operator_samples, seed,
                  output_dir, device="cuda", epsilon=0.1):
    """
    Full analysis computing T_{i->j} and then geometry diagnostics.
    """
    print("=" * 60)
    print("EMERGENT GEOMETRY ANALYSIS (FULL)")
    print("=" * 60)

    # Compute permeabilities
    T_matrices, G = compute_full_permeability_matrix(
        n_nodes, edge_prob, times, h, J,
        n_state_samples, n_operator_samples, seed, device
    )

    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "permeability_matrices.npz"),
             T_matrices=T_matrices, times=times, n_nodes=n_nodes)

    # Distance metrics
    print("\nComputing distance matrices...")
    T_final = T_matrices[-1]
    D_inv = distance_from_inverse_permeability(T_final)
    D_prop = distance_from_propagation_time(T_matrices, times, epsilon=epsilon)
    D_graph = graph_distance(G)

    # Summaries
    def _rng(D):
        fin = D[np.isfinite(D)]
        return float(np.min(fin)), float(np.max(fin))

    print(f"D_inverse_perm: range [{_rng(D_inv)[0]:.4f}, {_rng(D_inv)[1]:.4f}]")
    print(f"D_propagation:  range [{_rng(D_prop)[0]:.4f}, {_rng(D_prop)[1]:.4f}]")
    print(f"D_graph:        range [{_rng(D_graph)[0]:.4f}, {_rng(D_graph)[1]:.4f}]")

    # Correlations (pairwise finite)
    mask = np.isfinite(D_inv) & np.isfinite(D_graph)
    if np.sum(mask) > 0:
        corr_inv_graph = np.corrcoef(D_inv[mask].ravel(), D_graph[mask].ravel())[0, 1]
        print(f"Correlation (inverse_perm, graph): {corr_inv_graph:.4f}")

    mask = np.isfinite(D_prop) & np.isfinite(D_graph)
    if np.sum(mask) > 0:
        corr_prop_graph = np.corrcoef(D_prop[mask].ravel(), D_graph[mask].ravel())[0, 1]
        print(f"Correlation (propagation, graph): {corr_prop_graph:.4f}")

    # MDS + Ball-growth for each metric
    for name, D in [("inverse_perm", D_inv), ("propagation", D_prop), ("graph", D_graph)]:
        print(f"\n--- MDS Analysis for {name} distance ---")
        mds_analysis(D, max_dim=min(6, n_nodes - 1))

        print(f"\n--- Ball-growth Analysis for {name} distance ---")
        run_ball_growth_suite(name, D, output_dir)

    print("\nOutput saved to:", output_dir)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Emergent geometry analysis for HIP framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--results", type=str, default=None,
                        help="Path to results.npz from HIP simulation (proxy mode)")
    parser.add_argument("--analyze-perm", type=str, default=None,
                        help="Path to permeability_matrices.npz to analyze without recompute")
    parser.add_argument("--full", action="store_true",
                        help="Run full analysis computing T_{i->j} for all pairs")

    parser.add_argument("--n_nodes", type=int, default=10)
    parser.add_argument("--edge_prob", type=float, default=0.4)
    parser.add_argument("--times", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--n_state_samples", type=int, default=10)
    parser.add_argument("--n_operator_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="geometry_analysis")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Threshold for propagation-time distance")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.full:
        full_analysis(
            n_nodes=args.n_nodes,
            edge_prob=args.edge_prob,
            times=np.array(args.times, dtype=np.float64),
            h=args.h,
            J=args.J,
            n_state_samples=args.n_state_samples,
            n_operator_samples=args.n_operator_samples,
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device,
            epsilon=args.epsilon
        )
    elif args.analyze_perm:
        analyze_from_permeability_npz(args.analyze_perm, args.output_dir, epsilon=args.epsilon)
    elif args.results:
        analyze_from_results_file(args.results, args.output_dir)
    else:
        print("Please specify one of:")
        print("  --full")
        print("  --analyze-perm <permeability_matrices.npz>")
        print("  --results <results.npz>")


if __name__ == "__main__":
    main()
