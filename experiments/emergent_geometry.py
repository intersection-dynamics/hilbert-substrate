#!/usr/bin/env python3
"""
Emergent Geometry from HIP
===========================

Given HIP permeability data, derive an information-geometric distance
and test whether it embeds into low-dimensional space.

The core question: does the dynamics induce an effective geometry?

Distance candidates:
1. Propagation time: d(i,j) = min{t : T_{i->j}(t) > threshold}
2. Inverse typicality: d(i,j) = 1 / T_{i,j} (low accessibility = far)
3. Gap-based: d(i,j) = G_{i,j} = C - T (hard to reach = far)
4. Correlation-based: d(i,j) = -log(T_{i,j})

Then we:
1. Build a distance matrix for all node pairs
2. Use MDS to embed into R^n for various n
3. Measure reconstruction error (stress) vs dimension
4. Look for a "knee" indicating natural dimensionality
"""

import argparse
import json
import os
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DISTANCE METRICS FROM HIP DATA
# =============================================================================

def load_hip_results(filepath: str) -> dict:
    """Load HIP comparison results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def build_graph_from_edges(edge_keys: List[str]) -> nx.Graph:
    """Reconstruct graph from edge keys like '0-1', '2-3'."""
    G = nx.Graph()
    for key in edge_keys:
        u, v = map(int, key.split('-'))
        G.add_edge(u, v)
    return G


def floyd_warshall_with_weights(G: nx.Graph, edge_weights: Dict[str, float]) -> np.ndarray:
    """
    Compute all-pairs shortest paths using edge weights.
    Returns distance matrix indexed by node.
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize with infinity
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)
    
    # Set edge weights
    for (u, v) in G.edges():
        key1 = f"{u}-{v}"
        key2 = f"{v}-{u}"
        w = edge_weights.get(key1, edge_weights.get(key2, 1.0))
        i, j = node_idx[u], node_idx[v]
        dist[i, j] = w
        dist[j, i] = w
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    return dist, nodes


def distance_from_inverse_typicality(results: dict, time_idx: int = -1) -> Tuple[np.ndarray, list]:
    """
    Distance = 1 / T_{i,j}
    Low typicality = hard to reach = far
    """
    typ_data = results['typicality']
    edges = list(typ_data.keys())
    G = build_graph_from_edges(edges)
    
    # Edge weights: inverse typicality (add small epsilon to avoid division by zero)
    edge_weights = {}
    for edge_key, typ_values in typ_data.items():
        t_val = typ_values[time_idx]
        edge_weights[edge_key] = 1.0 / (t_val + 1e-6)
    
    return floyd_warshall_with_weights(G, edge_weights)


def distance_from_gap(results: dict, time_idx: int = -1) -> Tuple[np.ndarray, list]:
    """
    Distance = G_{i,j} = C - T
    Large gap = needs fine-tuning = far (in accessibility sense)
    """
    gap_data = results['gap']
    edges = list(gap_data.keys())
    G = build_graph_from_edges(edges)
    
    edge_weights = {}
    for edge_key, gap_values in gap_data.items():
        edge_weights[edge_key] = gap_values[time_idx]
    
    return floyd_warshall_with_weights(G, edge_weights)


def distance_from_log_typicality(results: dict, time_idx: int = -1) -> Tuple[np.ndarray, list]:
    """
    Distance = -log(T_{i,j})
    Analogous to correlation decay: d ~ -log(correlation)
    """
    typ_data = results['typicality']
    edges = list(typ_data.keys())
    G = build_graph_from_edges(edges)
    
    edge_weights = {}
    for edge_key, typ_values in typ_data.items():
        t_val = typ_values[time_idx]
        edge_weights[edge_key] = -np.log(t_val + 1e-6)
    
    return floyd_warshall_with_weights(G, edge_weights)


def distance_from_capacity_gap_ratio(results: dict, time_idx: int = -1) -> Tuple[np.ndarray, list]:
    """
    Distance = G / C = (C - T) / C = 1 - T/C
    Normalized gap: what fraction of capacity is inaccessible?
    """
    cap_data = results['capacity']
    typ_data = results['typicality']
    edges = list(cap_data.keys())
    G = build_graph_from_edges(edges)
    
    edge_weights = {}
    for edge_key in edges:
        c_val = cap_data[edge_key][time_idx]
        t_val = typ_data[edge_key][time_idx]
        edge_weights[edge_key] = 1.0 - (t_val / (c_val + 1e-6))
    
    return floyd_warshall_with_weights(G, edge_weights)


# =============================================================================
# MULTIDIMENSIONAL SCALING
# =============================================================================

def classical_mds(D: np.ndarray, n_components: int) -> np.ndarray:
    """
    Classical (metric) MDS embedding.
    Given distance matrix D, find coordinates X such that ||x_i - x_j|| ≈ D_ij
    """
    n = D.shape[0]
    
    # Center the squared distance matrix
    D_sq = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    B = -0.5 * H @ D_sq @ H  # Double-centered matrix
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    
    # Handle negative eigenvalues (can happen with non-Euclidean distances)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Coordinates
    X = eigenvectors * np.sqrt(eigenvalues)
    
    return X, eigenvalues


def compute_stress(D_original: np.ndarray, X: np.ndarray) -> float:
    """
    Compute normalized stress (Kruskal's stress-1):
    stress = sqrt(sum((d_ij - d'_ij)^2) / sum(d_ij^2))
    where d'_ij is the embedded distance.
    """
    n = X.shape[0]
    
    # Compute embedded distances
    D_embedded = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D_embedded[i, j] = np.linalg.norm(X[i] - X[j])
    
    # Stress
    numerator = np.sum((D_original - D_embedded) ** 2)
    denominator = np.sum(D_original ** 2)
    
    if denominator < 1e-12:
        return 0.0
    
    return np.sqrt(numerator / denominator)


def compute_eigenvalue_spectrum(D: np.ndarray, max_dim: int = 10) -> np.ndarray:
    """
    Compute eigenvalues of the MDS kernel matrix.
    The spectrum reveals the "natural" dimensionality.
    """
    n = D.shape[0]
    D_sq = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H
    
    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    
    return eigenvalues[:max_dim]


def analyze_dimensionality(D: np.ndarray, max_dim: int = 8) -> dict:
    """
    Full dimensionality analysis:
    - Embed into dimensions 1 through max_dim
    - Compute stress for each
    - Compute eigenvalue spectrum
    - Find the "knee" (elbow) in the stress curve
    """
    n = D.shape[0]
    max_dim = min(max_dim, n - 1)
    
    dimensions = list(range(1, max_dim + 1))
    stresses = []
    embeddings = []
    
    for dim in dimensions:
        X, eigenvalues = classical_mds(D, dim)
        stress = compute_stress(D, X)
        stresses.append(stress)
        embeddings.append(X)
    
    # Eigenvalue spectrum
    eigenvalues = compute_eigenvalue_spectrum(D, max_dim)
    
    # Find knee point (maximum curvature in stress curve)
    # Simple heuristic: largest second derivative
    if len(stresses) >= 3:
        second_deriv = np.diff(stresses, 2)
        knee_idx = np.argmax(second_deriv) + 1  # +1 because diff reduces length
        knee_dim = dimensions[knee_idx]
    else:
        knee_dim = 1
    
    # Variance explained by each dimension
    total_var = np.sum(np.maximum(eigenvalues, 0))
    if total_var > 1e-12:
        var_explained = np.cumsum(np.maximum(eigenvalues, 0)) / total_var
    else:
        var_explained = np.ones(len(eigenvalues))
    
    return {
        'dimensions': dimensions,
        'stresses': stresses,
        'eigenvalues': eigenvalues.tolist(),
        'variance_explained': var_explained.tolist(),
        'knee_dimension': knee_dim,
        'embeddings': embeddings
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_analysis(results: dict, analysis: dict, distance_name: str, 
                  nodes: list, outdir: str):
    """Generate all visualization plots."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    os.makedirs(outdir, exist_ok=True)
    
    # 1. Stress vs Dimension
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(analysis['dimensions'], analysis['stresses'], 'o-', lw=2, ms=8)
    axes[0].axvline(analysis['knee_dimension'], color='r', linestyle='--', 
                    label=f"Knee at d={analysis['knee_dimension']}")
    axes[0].set_xlabel("Embedding Dimension", fontsize=12)
    axes[0].set_ylabel("Stress (reconstruction error)", fontsize=12)
    axes[0].set_title("MDS Stress vs Dimension", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Variance explained
    axes[1].bar(analysis['dimensions'], analysis['variance_explained'][:len(analysis['dimensions'])],
                color='steelblue', alpha=0.7)
    axes[1].axhline(0.9, color='r', linestyle='--', label='90% threshold')
    axes[1].set_xlabel("Embedding Dimension", fontsize=12)
    axes[1].set_ylabel("Cumulative Variance Explained", fontsize=12)
    axes[1].set_title("Eigenvalue Spectrum", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    fig.suptitle(f"Emergent Dimensionality Analysis\nDistance: {distance_name}", 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dimensionality_analysis.png"), dpi=150)
    plt.close(fig)
    
    # 3. Eigenvalue scree plot
    fig, ax = plt.subplots(figsize=(8, 5))
    eigenvalues = analysis['eigenvalues']
    ax.bar(range(1, len(eigenvalues) + 1), eigenvalues, color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel("Eigenvalue Index", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_title("MDS Eigenvalue Spectrum\n(Negative values indicate non-Euclidean structure)", 
                 fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "eigenvalue_spectrum.png"), dpi=150)
    plt.close(fig)
    
    # 4. 2D embedding
    if len(analysis['embeddings']) >= 2:
        X_2d = analysis['embeddings'][1]  # dim=2
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(X_2d[:, 0], X_2d[:, 1], s=200, c='steelblue', alpha=0.7, edgecolors='black')
        for i, node in enumerate(nodes):
            ax.annotate(str(node), (X_2d[i, 0], X_2d[i, 1]), fontsize=12, ha='center', va='center')
        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.set_title(f"2D Embedding (stress={analysis['stresses'][1]:.4f})", fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "embedding_2d.png"), dpi=150)
        plt.close(fig)
    
    # 5. 3D embedding
    if len(analysis['embeddings']) >= 3:
        X_3d = analysis['embeddings'][2]  # dim=3
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=200, c='steelblue', alpha=0.7, edgecolors='black')
        for i, node in enumerate(nodes):
            ax.text(X_3d[i, 0], X_3d[i, 1], X_3d[i, 2], str(node), fontsize=10)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.set_title(f"3D Embedding (stress={analysis['stresses'][2]:.4f})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "embedding_3d.png"), dpi=150)
        plt.close(fig)
    
    # 6. Compare all distance metrics (if we have the data)
    print(f"Plots saved to {outdir}/")


def plot_distance_matrix(D: np.ndarray, nodes: list, title: str, outdir: str):
    """Visualize the distance matrix as a heatmap."""
    import matplotlib.pyplot as plt
    
    os.makedirs(outdir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D, cmap='viridis')
    ax.set_xticks(range(len(nodes)))
    ax.set_yticks(range(len(nodes)))
    ax.set_xticklabels(nodes)
    ax.set_yticklabels(nodes)
    ax.set_xlabel("Node j")
    ax.set_ylabel("Node i")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Distance d(i,j)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "distance_matrix.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Derive emergent geometry from HIP data")
    parser.add_argument("--input", type=str, required=True, help="Path to HIP results JSON")
    parser.add_argument("--metric", type=str, default="log_typicality",
                        choices=["inverse_typicality", "gap", "log_typicality", "gap_ratio"],
                        help="Distance metric to use")
    parser.add_argument("--time-idx", type=int, default=-1, 
                        help="Time index to use (-1 = last)")
    parser.add_argument("--max-dim", type=int, default=8,
                        help="Maximum embedding dimension to test")
    parser.add_argument("--out", type=str, default="outputs_geometry",
                        help="Output directory")
    parser.add_argument("--no-plots", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  EMERGENT GEOMETRY FROM HIP")
    print("=" * 60)
    
    # Load data
    print(f"Loading HIP data from {args.input}...")
    results = load_hip_results(args.input)
    
    times = results['times']
    time_idx = args.time_idx if args.time_idx >= 0 else len(times) + args.time_idx
    print(f"Using time t = {times[time_idx]}")
    print(f"Distance metric: {args.metric}")
    
    # Compute distance matrix
    print("\nComputing distance matrix...")
    
    if args.metric == "inverse_typicality":
        D, nodes = distance_from_inverse_typicality(results, time_idx)
        metric_name = "Inverse Typicality: d = 1/T"
    elif args.metric == "gap":
        D, nodes = distance_from_gap(results, time_idx)
        metric_name = "Gap: d = C - T"
    elif args.metric == "log_typicality":
        D, nodes = distance_from_log_typicality(results, time_idx)
        metric_name = "Log Typicality: d = -log(T)"
    elif args.metric == "gap_ratio":
        D, nodes = distance_from_capacity_gap_ratio(results, time_idx)
        metric_name = "Gap Ratio: d = 1 - T/C"
    
    print(f"Distance matrix: {D.shape[0]} x {D.shape[1]}")
    print(f"Distance range: [{D[D < np.inf].min():.4f}, {D[D < np.inf].max():.4f}]")
    
    # Check for infinite distances (disconnected components)
    if np.any(np.isinf(D)):
        print("WARNING: Graph has disconnected components (infinite distances)")
        D = np.where(np.isinf(D), D[D < np.inf].max() * 2, D)
    
    # Analyze dimensionality
    print("\nAnalyzing emergent dimensionality...")
    analysis = analyze_dimensionality(D, max_dim=args.max_dim)
    
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)
    print(f"{'Dim':<6} {'Stress':<12} {'Var Explained':<15}")
    for i, dim in enumerate(analysis['dimensions']):
        var_exp = analysis['variance_explained'][i] if i < len(analysis['variance_explained']) else 0
        print(f"{dim:<6} {analysis['stresses'][i]:<12.4f} {var_exp:<15.4f}")
    
    print("-" * 40)
    print(f"Suggested dimension (knee): {analysis['knee_dimension']}")
    
    # Eigenvalue interpretation
    eigenvalues = np.array(analysis['eigenvalues'])
    n_positive = np.sum(eigenvalues > 1e-6)
    n_negative = np.sum(eigenvalues < -1e-6)
    print(f"Positive eigenvalues: {n_positive}")
    print(f"Negative eigenvalues: {n_negative}")
    
    if n_negative > 0:
        print("\n⚠ Negative eigenvalues detected!")
        print("  This indicates the distance is NON-EUCLIDEAN.")
        print("  The information geometry may be hyperbolic or have curvature.")
    
    # Save results
    os.makedirs(args.out, exist_ok=True)
    
    output_data = {
        'metric': args.metric,
        'time': times[time_idx],
        'nodes': nodes,
        'distance_matrix': D.tolist(),
        'dimensions': analysis['dimensions'],
        'stresses': analysis['stresses'],
        'eigenvalues': analysis['eigenvalues'],
        'variance_explained': analysis['variance_explained'],
        'knee_dimension': analysis['knee_dimension'],
        'n_positive_eigenvalues': int(n_positive),
        'n_negative_eigenvalues': int(n_negative)
    }
    
    with open(os.path.join(args.out, "geometry_results.json"), 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.out}/geometry_results.json")
    
    # Plots
    if not args.no_plots:
        plot_distance_matrix(D, nodes, f"Information Distance ({metric_name})", args.out)
        plot_analysis(results, analysis, metric_name, nodes, args.out)
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    knee = analysis['knee_dimension']
    stress_at_knee = analysis['stresses'][knee - 1]
    
    if stress_at_knee < 0.1:
        print(f"✓ Low stress ({stress_at_knee:.4f}) at dimension {knee}")
        print(f"  The information geometry embeds well into {knee}D Euclidean space.")
        if knee == 3:
            print("  → Emergent 3D spatial structure!")
        elif knee == 4:
            print("  → Emergent 4D structure (3+1 spacetime?)")
    elif stress_at_knee < 0.2:
        print(f"◐ Moderate stress ({stress_at_knee:.4f}) at dimension {knee}")
        print(f"  Approximate embedding into {knee}D, but with distortion.")
    else:
        print(f"✗ High stress ({stress_at_knee:.4f}) even at dimension {knee}")
        print("  The information geometry may be intrinsically non-Euclidean.")
        print("  Consider: hyperbolic embedding, or the system lacks geometric structure.")
    
    if n_negative > n_positive * 0.3:
        print("\n⚠ Strong non-Euclidean signature in the distance structure.")
        print("  This could indicate:")
        print("  - Hyperbolic geometry (negative curvature)")
        print("  - Lorentzian signature (timelike vs spacelike)")
        print("  - Fundamental non-metric structure")


if __name__ == "__main__":
    main()