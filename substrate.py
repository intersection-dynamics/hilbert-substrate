#!/usr/bin/env python3
"""
Substrate: Hilbert Graph with Lieb–Robinson Emergent Geometry
=============================================================

This file defines a foundational substrate engine:

  - A finite set of abstract sites i = 0,...,N-1, with no coordinates.
  - A local Hilbert space of dimension d per site (default d=2).
  - A global Hilbert space H = (C^d)^{⊗ N}.
  - A graph-based Hamiltonian built from local hopping terms.
  - Exact unitary time evolution via matrix exponentials (small N).
  - True Lieb–Robinson commutators:
        C_ij(t) = [A_i(t), B_j], with A_i, B_j local.
    and norms ||C_ij(t)|| as a function of time.
  - Emergent operational distance from LR arrival times.
  - Optional classical embedding of the LR metric into R^2 or R^3
    via multidimensional scaling (MDS).

There is no built-in notion of space here. Geometry is whatever
distance matrix you read off from LR commutators.

This is intended as the geometric core for higher-level models
(pointer hydrogen, emergent fields, etc.) to sit on top of.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Literal

import numpy as np
from scipy.linalg import expm, eigh


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SubstrateConfig:
    """
    Configuration for the Hilbert graph substrate.

    n_sites         : number of abstract sites (N)
    d_local         : local Hilbert dimension per site (d)
                      (d=2 => qubit / truncated 0-1 boson)
    coupling        : overall hopping strength in the Hamiltonian
    connectivity    : how to wire the underlying graph:
                        'chain' (1D nearest-neighbor),
                        'ring'  (periodic chain),
                        'complete' (all-to-all),
                        'random'  (Erdos-Renyi)
    random_p        : connection probability if connectivity='random'
    seed            : RNG seed
    lr_threshold    : norm threshold for LR "arrival"
    lr_t_max        : maximum time for LR evolution
    lr_n_steps      : number of time slices for LR evolution
    lr_norm         : operator norm for LR:
                        'fro' or 'spectral'
    """

    n_sites: int = 8
    d_local: int = 2
    coupling: float = 1.0

    connectivity: Literal["chain", "ring", "complete", "random"] = "chain"
    random_p: float = 0.3
    seed: int = 12345

    lr_threshold: float = 1e-3
    lr_t_max: float = 8.0
    lr_n_steps: int = 120
    lr_norm: Literal["fro", "spectral"] = "fro"


# =============================================================================
# Substrate: global Hilbert space and local operators
# =============================================================================


class HilbertSubstrate:
    """
    Global Hilbert substrate on an abstract graph.

    - Sites i = 0,...,N-1, no coordinates.
    - Local dimension d (same at each site).
    - Global dimension D = d^N.

    This class provides:
      - global basis indexing
      - local creation/annihilation (truncated boson style for d>1)
      - local number operators n_i
    """

    def __init__(self, n_sites: int, d_local: int):
        self.n_sites = n_sites
        self.d_local = d_local
        self.dim_total = d_local ** n_sites

    # ---- basis encoding/decoding -------------------------------------------------

    def index_to_config(self, idx: int) -> Tuple[int, ...]:
        """
        Convert a basis index (0..D-1) to a tuple of local occupations
        (n_0, ..., n_{N-1}) in base-d representation.
        """
        conf = [0] * self.n_sites
        d = self.d_local
        for k in range(self.n_sites):
            power = self.n_sites - 1 - k
            base = d ** power
            conf[k] = (idx // base) % d
        return tuple(conf)

    def config_to_index(self, conf: Tuple[int, ...]) -> int:
        """
        Convert a tuple of local occupations (n_0,...,n_{N-1}) to
        a basis index 0..D-1.
        """
        idx = 0
        d = self.d_local
        for k, n in enumerate(conf):
            power = self.n_sites - 1 - k
            idx += int(n) * (d ** power)
        return idx

    # ---- states ------------------------------------------------------------------

    def vacuum(self) -> np.ndarray:
        """Return |0,0,...,0> as a global state vector."""
        psi = np.zeros(self.dim_total, dtype=complex)
        psi[0] = 1.0
        return psi

    def normalize(self, psi: np.ndarray) -> np.ndarray:
        """Normalize a state vector."""
        norm = np.linalg.norm(psi)
        if norm == 0:
            return psi
        return psi / norm

    def excite_site(self, psi: np.ndarray, site: int) -> np.ndarray:
        """
        Apply a truncated bosonic creation operator a_site^\dagger
        to the global state, assuming local occupations 0..d-1.
        """
        d = self.d_local
        D = self.dim_total
        out = np.zeros_like(psi)

        for idx in range(D):
            conf = list(self.index_to_config(idx))
            n = conf[site]
            if n < d - 1:
                conf[site] = n + 1
                new_idx = self.config_to_index(tuple(conf))
                out[new_idx] += psi[idx] * np.sqrt(n + 1)
        return out

    def local_number_operator(self, site: int) -> np.ndarray:
        """
        Build the full many-body operator matrix n_site.

          n_site |n_0,...,n_i,...> = n_i |n_0,...,n_i,...>.
        """
        D = self.dim_total
        n_op = np.zeros((D, D), dtype=complex)
        for idx in range(D):
            conf = self.index_to_config(idx)
            n_op[idx, idx] = conf[site]
        return n_op

    # ---- local expectation -------------------------------------------------------

    def local_occupation(self, psi: np.ndarray, site: int) -> float:
        """
        Compute expectation <n_site> for a pure state psi.
        """
        D = self.dim_total
        exp_val = 0.0
        for idx in range(D):
            conf = self.index_to_config(idx)
            exp_val += (abs(psi[idx]) ** 2) * conf[site]
        return float(exp_val)


# =============================================================================
# Graph + Hamiltonian
# =============================================================================


def build_connectivity(cfg: SubstrateConfig) -> Dict[int, List[int]]:
    """
    Build an adjacency list describing the bare connectivity.

    Note: this is not yet spacetime. It's just a bare graph
    for local couplings; the emergent metric comes from LR.
    """
    N = cfg.n_sites
    rng = np.random.default_rng(cfg.seed)

    if cfg.connectivity == "chain":
        return {i: [j for j in (i - 1, i + 1) if 0 <= j < N] for i in range(N)}

    if cfg.connectivity == "ring":
        return {i: [((i - 1) % N), ((i + 1) % N)] for i in range(N)}

    if cfg.connectivity == "complete":
        return {i: [j for j in range(N) if j != i] for i in range(N)}

    if cfg.connectivity == "random":
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < cfg.random_p:
                    adj[i].append(j)
                    adj[j].append(i)
        return adj

    raise ValueError(f"Unknown connectivity type: {cfg.connectivity}")


def build_hamiltonian(substrate: HilbertSubstrate,
                      connectivity: Dict[int, List[int]],
                      coupling: float) -> np.ndarray:
    """
    Build a hopping Hamiltonian on the global Hilbert space:

      H = coupling * sum_{i<j, (i,j) in edges}
              (a_i^\dagger a_j + a_j^\dagger a_i)

    where a_i, a_i^\dagger are truncated bosonic ops in 0..d-1
    on site i. This is the simplest local, symmetric interaction
    one can use; it's not tied to any classical geometry.

    Complexity scales as O(D^2 * N * degree) with D = d^N.
    This is strictly a small-N, few-site lab engine.
    """
    D = substrate.dim_total
    H = np.zeros((D, D), dtype=complex)
    d = substrate.d_local

    for i, neighbors in connectivity.items():
        for j in neighbors:
            if j <= i:
                # only treat each edge once
                continue

            for idx in range(D):
                conf = list(substrate.index_to_config(idx))
                n_i = conf[i]
                n_j = conf[j]

                # a_i^\dagger a_j
                if n_j > 0 and n_i < d - 1:
                    coeff = np.sqrt(n_j) * np.sqrt(n_i + 1)
                    new_conf = conf.copy()
                    new_conf[j] -= 1
                    new_conf[i] += 1
                    new_idx = substrate.config_to_index(tuple(new_conf))
                    H[new_idx, idx] += coupling * coeff

                # a_j^\dagger a_i
                if n_i > 0 and n_j < d - 1:
                    coeff = np.sqrt(n_i) * np.sqrt(n_j + 1)
                    new_conf = conf.copy()
                    new_conf[i] -= 1
                    new_conf[j] += 1
                    new_idx = substrate.config_to_index(tuple(new_conf))
                    H[new_idx, idx] += coupling * coeff

    return H


# =============================================================================
# Time evolution and propagation-based signals
# =============================================================================


def evolve_state(H: np.ndarray, psi0: np.ndarray, t_max: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve psi under H for times t in [0, t_max], with n_steps points.

    Returns:
      times: shape (n_steps,)
      states: shape (n_steps, D)
    """
    times = np.linspace(0.0, t_max, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    U = expm(-1j * H * dt)
    psi = psi0.copy()

    D = H.shape[0]
    states = np.zeros((n_steps, D), dtype=complex)

    for k in range(n_steps):
        states[k] = psi
        psi = U @ psi

    return times, states


def excitation_propagation(substrate: HilbertSubstrate,
                           H: np.ndarray,
                           source_site: int,
                           t_max: float,
                           n_steps: int,
                           threshold: float = 0.01) -> Dict[str, object]:
    """
    Original propagation-based operational signal:

      - Start with vacuum |Ω>.
      - Excite one site: psi0 = a_source^\dagger |Ω>.
      - Evolve under H.
      - Track ⟨n_j(t)⟩ for all j.
      - Define arrival time t_ij^prop where ⟨n_j(t)⟩ crosses threshold.

    Returns:
      times        : (T,)
      occupations  : (N, T)
      arrival      : dict j -> t_arrival (or inf)
    """
    psi0 = substrate.excite_site(substrate.vacuum(), source_site)
    psi0 = substrate.normalize(psi0)

    times, states = evolve_state(H, psi0, t_max, n_steps)
    N = substrate.n_sites
    T = len(times)

    occ = np.zeros((N, T), dtype=float)

    for t_idx in range(T):
        psi = states[t_idx]
        for j in range(N):
            occ[j, t_idx] = substrate.local_occupation(psi, j)

    arrival: Dict[int, float] = {}
    for j in range(N):
        if j == source_site:
            arrival[j] = 0.0
            continue
        above = np.where(occ[j] > threshold)[0]
        arrival[j] = float(times[above[0]]) if len(above) > 0 else float("inf")

    return {
        "times": times,
        "occupations": occ,
        "arrival": arrival,
        "source": source_site,
        "threshold": threshold,
    }


def excitation_metric_from_arrival(substrate: HilbertSubstrate,
                                   H: np.ndarray,
                                   t_max: float,
                                   n_steps: int,
                                   threshold: float = 0.01) -> np.ndarray:
    """
    Build a symmetric 'distance' matrix from excitation arrival times:

      d_ij^prop = arrival time of ⟨n_j(t)⟩ after exciting i.

    This is not Lieb–Robinson, but it's a useful comparison.
    """
    N = substrate.n_sites
    Dmat = np.zeros((N, N), dtype=float)

    for source in range(N):
        res = excitation_propagation(substrate, H, source, t_max, n_steps, threshold)
        for j in range(N):
            Dmat[source, j] = res["arrival"][j]  # type: ignore[index]

    Dmat = 0.5 * (Dmat + Dmat.T)
    return Dmat


# =============================================================================
# Lieb–Robinson commutators and metric
# =============================================================================


def lieb_robinson_commutators(substrate: HilbertSubstrate,
                              H: np.ndarray,
                              source_site: int,
                              t_max: float,
                              n_steps: int,
                              threshold: float,
                              norm_type: Literal["fro", "spectral"]) -> Dict[str, object]:
    """
    Compute the true LR commutator:

      C_ij(t) = [A_i(t), B_j] = A_i(t) B_j - B_j A_i(t)
      with A_i(0) = n_i, B_j = n_j.

    Evolution in Heisenberg picture:
      A_i(t+dt) = U† A_i(t) U, with U = exp(-i H dt).

    We record ||C_ij(t)|| for all j and t, using:

      norm_type = 'fro'      -> Frobenius norm
                   'spectral' -> operator 2-norm (largest singular value)

    Arrival time t_ij^LR is defined as the first t where ||C_ij(t)|| >= threshold.
    """

    N = substrate.n_sites

    times = np.linspace(0.0, t_max, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    U = expm(-1j * H * dt)
    U_dag = U.conj().T

    # Precompute local number operators
    n_ops = [substrate.local_number_operator(j) for j in range(N)]

    # A_i(0) = n_i
    A_t = n_ops[source_site].copy()

    comm_norms = np.zeros((N, n_steps), dtype=float)

    for t_idx in range(n_steps):
        for j in range(N):
            C = A_t @ n_ops[j] - n_ops[j] @ A_t
            if norm_type == "spectral":
                comm_norms[j, t_idx] = float(np.linalg.norm(C, 2))
            else:
                comm_norms[j, t_idx] = float(np.linalg.norm(C, "fro"))

        # Heisenberg update
        A_t = U_dag @ A_t @ U

    # Arrival times
    arrival: Dict[int, float] = {}
    for j in range(N):
        if j == source_site:
            arrival[j] = 0.0
            continue
        above = np.where(comm_norms[j] >= threshold)[0]
        arrival[j] = float(times[above[0]]) if len(above) > 0 else float("inf")

    return {
        "times": times,
        "comm_norms": comm_norms,
        "arrival": arrival,
        "source": source_site,
        "threshold": threshold,
        "norm_type": norm_type,
    }


def lieb_robinson_metric(substrate: HilbertSubstrate,
                         H: np.ndarray,
                         t_max: float,
                         n_steps: int,
                         threshold: float,
                         norm_type: Literal["fro", "spectral"]) -> np.ndarray:
    """
    Build an operational distance matrix from LR commutator arrival times:

      d_ij^LR = first t where ||[A_i(t), B_j]|| crosses threshold.

    This is the actual LR-based emergent metric (up to thresholds
    and finite-size/time effects), not a proxy based on ⟨n⟩.
    """
    N = substrate.n_sites
    Dmat = np.zeros((N, N), dtype=float)

    for source in range(N):
        lr = lieb_robinson_commutators(
            substrate,
            H,
            source_site=source,
            t_max=t_max,
            n_steps=n_steps,
            threshold=threshold,
            norm_type=norm_type,
        )
        for j in range(N):
            Dmat[source, j] = lr["arrival"][j]  # type: ignore[index]

    Dmat = 0.5 * (Dmat + Dmat.T)
    return Dmat


# =============================================================================
# Graph distances and "inflation" analysis
# =============================================================================


def compute_graph_distances(connectivity: Dict[int, List[int]]) -> np.ndarray:
    """
    Compute all-pairs graph distances (shortest path lengths) on the
    bare connectivity graph using BFS.

    Returns:
      dist[i,j] = minimal number of edges in any path from i to j
                  (or np.inf if disconnected).
    """
    nodes = sorted(connectivity.keys())
    N = len(nodes)
    dist = np.full((N, N), np.inf, dtype=float)

    for i in nodes:
        # BFS from i
        d = {i: 0}
        frontier = [i]
        while frontier:
            new_frontier: List[int] = []
            for u in frontier:
                for v in connectivity[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        new_frontier.append(v)
            frontier = new_frontier

        for j, dj in d.items():
            dist[i, j] = float(dj)
            dist[j, i] = float(dj)

    return dist


def analyze_lr_inflation(D_lr: np.ndarray,
                         graph_dist: np.ndarray,
                         source: int) -> None:
    """
    Given:
      - D_lr[i,j]   : LR "distance" (arrival time) between i and j,
      - graph_dist[i,j] : bare graph distance between i and j,
      - source      : index of the source site,

    build a shell profile:

      t_shell[d]   = average LR distance to sites at graph distance d,
      dt_shell[d]  = t_shell[d] - t_shell[d-1]  (extra LR time per hop),
      v_eff[d]     = 1 / dt_shell[d]            (effective speed per shell),

    and print a little table so we can see whether "inflation" slows
    or keeps ramping up.
    """
    N = D_lr.shape[0]
    d_max = int(np.nanmax(graph_dist[source, np.isfinite(graph_dist[source])]))

    # collect arrival times by graph shell
    shell_times: Dict[int, List[float]] = {d: [] for d in range(d_max + 1)}

    for j in range(N):
        d_g = graph_dist[source, j]
        if not np.isfinite(d_g):
            continue
        d_int = int(d_g)
        t_lr = D_lr[source, j]
        if np.isfinite(t_lr):
            shell_times[d_int].append(t_lr)

    t_shell = np.zeros(d_max + 1, dtype=float)
    for d in range(d_max + 1):
        if shell_times[d]:
            t_shell[d] = float(np.mean(shell_times[d]))
        else:
            t_shell[d] = np.inf

    dt_shell = np.full(d_max + 1, np.nan, dtype=float)
    v_eff = np.full(d_max + 1, np.nan, dtype=float)

    for d in range(1, d_max + 1):
        if np.isfinite(t_shell[d]) and np.isfinite(t_shell[d - 1]):
            dt = t_shell[d] - t_shell[d - 1]
            dt_shell[d] = dt
            if dt > 0:
                v_eff[d] = 1.0 / dt

    print("Lieb–Robinson 'inflation' profile (source = %d):" % source)
    print("  d_graph |  t_LR_shell  |  Δt(d) = t(d)-t(d-1)  |  v_eff(d) = 1/Δt")
    print("  ---------------------------------------------------------------")
    print(f"      0   |  {t_shell[0]:.6f}    |        ---              |     ---")
    for d in range(1, d_max + 1):
        t = t_shell[d]
        dt = dt_shell[d]
        v = v_eff[d]
        if np.isfinite(t):
            t_str = f"{t:.6f}"
        else:
            t_str = "  inf   "
        if np.isfinite(dt):
            dt_str = f"{dt:.6f}"
        else:
            dt_str = "   nan  "
        if np.isfinite(v):
            v_str = f"{v:.3f}"
        else:
            v_str = "  nan "
        print(f"     {d:2d}   |  {t_str}    |   {dt_str}         |   {v_str}")
    print("  ---------------------------------------------------------------")
    print("  Note: decreasing v_eff(d) with d means 'inflation' slowing.")
    print()


# =============================================================================
# Metric → coordinate embedding (MDS)
# =============================================================================


def classical_mds_from_dist(Dmat: np.ndarray, dim: int = 3) -> np.ndarray:
    """
    Classical multidimensional scaling (MDS) from a distance matrix D:

      - Input: Dmat[i,j] = distance between sites i and j.
      - Output: X[i,:] = coordinate in R^dim.

    Steps:

      1. Compute squared distances: D^2.
      2. Double-center: B = -0.5 * J D^2 J, where J = I - 1_N/N.
      3. Eigen-decompose B = V diag(λ) V^T.
      4. Take the top 'dim' eigenvalues λ > 0 and eigenvectors v.
      5. Coordinates: X = V_dim * sqrt(Λ_dim).

    This is a classical embedding of your operational metric
    into Euclidean space. The axes and orientation are arbitrary,
    but relative geometry is meaningful.
    """
    D2 = Dmat ** 2
    N = Dmat.shape[0]
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J

    # Eigen-decomposition (symmetric)
    vals, vecs = eigh(B)

    # Sort by descending eigenvalue
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Take top 'dim' non-negative eigenvalues
    pos_mask = vals > 1e-12
    vals_pos = vals[pos_mask][:dim]
    vecs_pos = vecs[:, pos_mask][:, :dim]

    # Coordinates
    L_sqrt = np.sqrt(vals_pos)
    X = vecs_pos * L_sqrt[np.newaxis, :]
    return X  # shape (N, dim)


# =============================================================================
# Demo / CLI
# =============================================================================


def run_demo(cfg: SubstrateConfig, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Substrate config:")
    print(asdict(cfg))
    print()

    # Build substrate + Hamiltonian
    substrate = HilbertSubstrate(cfg.n_sites, cfg.d_local)
    connectivity = build_connectivity(cfg)
    H = build_hamiltonian(substrate, connectivity, cfg.coupling)

    print(f"Global dimension: {substrate.dim_total}")
    print("Connectivity:")
    for i in range(cfg.n_sites):
        print(f"  {i}: {connectivity.get(i, [])}")
    print()

    # Graph distances (combinatorial)
    graph_dist = compute_graph_distances(connectivity)

    source = cfg.n_sites // 2
    print(f"Using site {source} as LR source\n")

    # 1. excitation-based metric (for comparison)
    prop_res = excitation_propagation(
        substrate,
        H,
        source_site=source,
        t_max=cfg.lr_t_max,
        n_steps=cfg.lr_n_steps,
        threshold=0.01,
    )
    D_prop = excitation_metric_from_arrival(
        substrate,
        H,
        t_max=cfg.lr_t_max,
        n_steps=cfg.lr_n_steps,
        threshold=0.01,
    )

    print("Propagation-based arrival times from source:")
    for j in range(cfg.n_sites):
        print(f"  {j}: {prop_res['arrival'][j]:.3f}")  # type: ignore[index]
    print()

    # 2. Lieb–Robinson metric
    lr_res = lieb_robinson_commutators(
        substrate,
        H,
        source_site=source,
        t_max=cfg.lr_t_max,
        n_steps=cfg.lr_n_steps,
        threshold=cfg.lr_threshold,
        norm_type=cfg.lr_norm,
    )
    D_lr = lieb_robinson_metric(
        substrate,
        H,
        t_max=cfg.lr_t_max,
        n_steps=cfg.lr_n_steps,
        threshold=cfg.lr_threshold,
        norm_type=cfg.lr_norm,
    )

    print("LR commutator arrival times from source:")
    for j in range(cfg.n_sites):
        print(f"  {j}: {lr_res['arrival'][j]:.3f}")  # type: ignore[index]
    print()

    # 3. Quantify "inflation" of LR distance vs graph distance
    analyze_lr_inflation(D_lr, graph_dist, source)

    # 4. Embed LR metric into R^2
    try:
        X2 = classical_mds_from_dist(D_lr, dim=2)
        # Save coordinates
        np.savez(os.path.join(out_dir, "lr_embedding_2d.npz"),
                 X2=X2, D_lr=D_lr, D_prop=D_prop, graph_dist=graph_dist)
        print("Saved LR 2D embedding to lr_embedding_2d.npz")
    except Exception as exc:
        print("Could not embed LR metric:", exc)

    # Also dump metrics for later analysis
    np.savez(os.path.join(out_dir, "lr_metrics.npz"), D_lr=D_lr, D_prop=D_prop, graph_dist=graph_dist)

    print("\nDone.")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hilbert graph substrate with Lieb–Robinson emergent geometry."
    )
    parser.add_argument("--n-sites", type=int, default=8, help="Number of abstract sites.")
    parser.add_argument("--d-local", type=int, default=2, help="Local Hilbert dimension per site.")
    parser.add_argument(
        "--connectivity",
        type=str,
        default="chain",
        choices=["chain", "ring", "complete", "random"],
        help="Bare graph connectivity type.",
    )
    parser.add_argument("--random-p", type=float, default=0.3, help="Connection prob for random graphs.")
    parser.add_argument("--coupling", type=float, default=1.0, help="Hopping coupling strength.")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    parser.add_argument("--lr-threshold", type=float, default=1e-3, help="LR commutator norm threshold.")
    parser.add_argument("--lr-t-max", type=float, default=8.0, help="Max time for LR evolution.")
    parser.add_argument("--lr-n-steps", type=int, default=120, help="Number of time steps for LR evolution.")
    parser.add_argument(
        "--lr-norm",
        type=str,
        default="fro",
        choices=["fro", "spectral"],
        help="Operator norm for LR commutators.",
    )
    parser.add_argument("--output-dir", type=str, default="substrate_outputs", help="Output directory.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    cfg = SubstrateConfig(
        n_sites=args.n_sites,
        d_local=args.d_local,
        coupling=args.coupling,
        connectivity=args.connectivity,  # type: ignore[arg-type]
        random_p=args.random_p,
        seed=args.seed,
        lr_threshold=args.lr_threshold,
        lr_t_max=args.lr_t_max,
        lr_n_steps=args.lr_n_steps,
        lr_norm=args.lr_norm,  # type: ignore[arg-type]
    )

    run_demo(cfg, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
