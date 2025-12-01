#!/usr/bin/env python3
"""
Cosmic inflation experiment using *continuous-weight* geometry.

This script:

  - builds a Substrate from substrate.Config,
  - evolves it with unitary + information-based defrag,
  - at intervals, builds an information-geometric distance from the full
    |J_ij| matrix without any hard threshold,
  - measures:
      * center-based radius_50 and radius_90 (from node 0),
      * global pairwise radius_50 and radius_90,
  - records everything to inflation_history_continuous.csv.

Geometry definition
-------------------
Given the coupling matrix J:

  w_ij = |J_ij|               (magnitude of coupling)
  w_max = max_{i!=j} w_ij

For edges with w_ij > 0, define a "cost" (inverse closeness):

  c_ij = -log( w_ij / w_max )

Then:

  - strongest edges (w_ij ≈ w_max) have c_ij ≈ 0,
  - weaker edges have larger c_ij,
  - non-edges (w_ij = 0) are treated as no direct path (infinite cost).

Information distance between nodes is the minimal path cost (sum of c_ij
along a path). Distances are computed with Dijkstra's algorithm.

No thresholds are used in the geometry itself: all nonzero couplings
contribute, with strength encoded continuously in the distance.
"""

import argparse
import csv
import math
from typing import List, Tuple

import numpy as np

from substrate import Config, Substrate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_numpy(x) -> np.ndarray:
    """Convert numpy/cupy array to NumPy ndarray."""
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def build_cost_matrix(sub: Substrate) -> np.ndarray:
    """
    Build a continuous information-geometric cost matrix from substrate couplings.

    Parameters
    ----------
    sub : Substrate

    Returns
    -------
    cost : np.ndarray, shape (N, N)
        cost[i, j] >= 0 is the "distance" cost along edge i->j.
        cost[i, j] = +inf if there is no direct edge (|J_ij| == 0).
        Diagonal entries are 0.
    """
    J = sub.couplings
    J_np = to_numpy(J)
    w = np.abs(J_np)

    n = w.shape[0]

    # No self-edges
    np.fill_diagonal(w, 0.0)

    # Find maximum nonzero weight
    nonzero_mask = w > 0.0
    if not np.any(nonzero_mask):
        # No edges at all; cost is inf off-diagonal, 0 on diagonal
        cost = np.full((n, n), np.inf, dtype=float)
        np.fill_diagonal(cost, 0.0)
        return cost

    w_max = float(w[nonzero_mask].max())

    # Cost for existing edges: c_ij = -log(w_ij / w_max)
    # Strongest edges -> cost ~ 0; weaker edges -> larger cost
    cost = np.full((n, n), np.inf, dtype=float)
    with np.errstate(divide="ignore"):
        ratio = np.zeros_like(w, dtype=float)
        ratio[nonzero_mask] = w[nonzero_mask] / w_max
        # ratio in (0, 1]; log(ratio) <= 0; negative of that >= 0
        cost[nonzero_mask] = -np.log(ratio[nonzero_mask])

    # Zero cost to stay at the same node
    np.fill_diagonal(cost, 0.0)

    return cost


def dijkstra_from(cost: np.ndarray, src: int) -> np.ndarray:
    """
    Dijkstra's algorithm for a single source on a dense cost matrix.

    Parameters
    ----------
    cost : np.ndarray, shape (N, N)
        cost[i, j] >= 0, cost[i, j] = +inf if no direct edge.
    src : int
        Source node index.

    Returns
    -------
    dist : np.ndarray, shape (N,)
        Minimal path cost from src to each node.
        dist[j] = +inf if j unreachable from src.
    """
    import heapq

    n = cost.shape[0]
    dist = np.full(n, np.inf, dtype=float)
    visited = np.zeros(n, dtype=bool)

    dist[src] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        if d > dist[u]:
            continue

        # Relax neighbors
        row = cost[u]
        # neighbors where cost is finite and > 0 or 0
        for v in range(n):
            c_uv = row[v]
            if not math.isfinite(c_uv):
                continue
            alt = d + c_uv
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(heap, (alt, v))

    return dist


def pairwise_distance_stats(cost: np.ndarray) -> Tuple[float, float]:
    """
    Compute 50th and 90th percentile of minimal path distances over all
    unordered node pairs (i < j) that are mutually reachable.

    Parameters
    ----------
    cost : np.ndarray, shape (N, N)

    Returns
    -------
    pair_r50 : float
    pair_r90 : float
    """
    n = cost.shape[0]
    if n <= 1:
        return 0.0, 0.0

    # Compute all-pairs shortest paths by running Dijkstra from each node.
    # For 128 nodes this is trivial.
    all_dists = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        all_dists[i] = dijkstra_from(cost, i)

    # Use only i < j and finite distances
    upper = []
    for i in range(n):
        for j in range(i + 1, n):
            d = all_dists[i, j]
            if math.isfinite(d):
                upper.append(d)

    if not upper:
        return 0.0, 0.0

    vals = np.array(upper, dtype=float)
    pair_r50 = float(np.quantile(vals, 0.5))
    pair_r90 = float(np.quantile(vals, 0.9))
    return pair_r50, pair_r90


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmic inflation probe with continuous-weight geometry.")
    parser.add_argument("--n_nodes", type=int, default=128, help="Number of substrate nodes.")
    parser.add_argument("--internal_dim", type=int, default=3, help="Local Hilbert-space dimension per node.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of evolution steps.")
    parser.add_argument("--record_stride", type=int, default=10, help="Record diagnostics every this many steps.")
    parser.add_argument("--defrag_rate", type=float, default=0.5, help="Defrag step size in substrate.")
    parser.add_argument("--connectivity", type=float, default=1.0, help="Initial edge probability between node pairs.")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step for Hamiltonian evolution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    print("[inflation-continuous] Building substrate...")

    cfg = Config(
        n_nodes=args.n_nodes,
        internal_dim=args.internal_dim,
        monogamy_budget=1.0,
        defrag_rate=args.defrag_rate,
        dt=args.dt,
        seed=args.seed,
        connectivity=args.connectivity,
    )

    sub = Substrate(cfg)

    print("[inflation-continuous] Starting evolution...")

    center_idx = 0
    n = args.n_nodes

    history_rows: List[Tuple] = []

    # We'll also store the initial center_r90 and pair_r90 so you can
    # reconstruct a "scale factor" later if you want, outside this script.
    center_r90_initial = None
    pair_r90_initial = None

    for step in range(0, args.steps + 1):
        if step % args.record_stride == 0:
            t = step * args.dt

            cost = build_cost_matrix(sub)

            # Center-based distances
            dist_center = dijkstra_from(cost, center_idx)
            finite_center = dist_center[np.isfinite(dist_center)]
            if finite_center.size == 0:
                center_r50 = 0.0
                center_r90 = 0.0
            else:
                center_r50 = float(np.quantile(finite_center, 0.5))
                center_r90 = float(np.quantile(finite_center, 0.9))

            # Global pairwise distance stats
            pair_r50, pair_r90 = pairwise_distance_stats(cost)

            # Initialize reference scales if needed
            if center_r90_initial is None:
                center_r90_initial = center_r90 if center_r90 > 0 else 1.0
            if pair_r90_initial is None:
                pair_r90_initial = pair_r90 if pair_r90 > 0 else 1.0

            # Relative "scale factors" (dimensionless) using initial values
            a_center_90 = center_r90 / center_r90_initial if center_r90_initial > 0 else 1.0
            a_pair_90 = pair_r90 / pair_r90_initial if pair_r90_initial > 0 else 1.0

            # Simple console summary
            print(
                f"[t={t:7.3f}] step={step:6d} "
                f"center(r50,r90)=({center_r50:6.3f},{center_r90:6.3f}) "
                f"pair(r50,r90)=({pair_r50:6.3f},{pair_r90:6.3f}) "
                f"a_center_90={a_center_90:6.3f} a_pair_90={a_pair_90:6.3f}"
            )

            # Record row for CSV
            history_rows.append(
                (
                    t,
                    step,
                    center_r50,
                    center_r90,
                    pair_r50,
                    pair_r90,
                    a_center_90,
                    a_pair_90,
                )
            )

        # Advance dynamics
        if step < args.steps:
            sub.evolve(n_steps=1)

    # Write CSV
    out_path = "inflation_history_continuous.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t_step",
                "step",
                "center_radius_50",
                "center_radius_90",
                "pair_radius_50",
                "pair_radius_90",
                "a_center_90",
                "a_pair_90",
            ]
        )
        for row in history_rows:
            writer.writerow(row)

    print(f"[inflation-continuous] Wrote history to {out_path}")


if __name__ == "__main__":
    main()
