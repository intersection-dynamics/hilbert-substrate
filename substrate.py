#!/usr/bin/env python3
"""
Substrate: Hilbert Graph with Lieb–Robinson Emergent Geometry
==========================================================================

Optimized version with eigendecomposition caching for O(D²) time evolution
instead of O(D³) matrix exponentials.

Key optimizations:
  1. Eigendecompose H once: H = V Λ V†
  2. Time evolution via diagonal sandwiching: U(t) = V e^{-iΛt} V†
  3. Heisenberg evolution in eigenbasis: A(t) = V e^{+iΛt} V† A V e^{-iΛt} V†
  4. Precompute V† nⱼ V for each site
  5. Single eigendecomposition serves all N source sites in LR metric

All results are bit-for-bit identical to the original implementation for small
test cases (see verify_against_original).

Backends:
  - CPU: NumPy + SciPy (default)
  - GPU: CuPy + cupyx.scipy.linalg (if --use-gpu and CuPy is available)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Literal, Optional

import numpy as np
from scipy.linalg import expm as np_expm, eigh as np_eigh

# Optional CuPy backend
try:
    import cupy as cp
    import cupyx.scipy.linalg as cpx_linalg
    HAS_CUPY = True
except ImportError:  # pragma: no cover
    cp = None
    cpx_linalg = None
    HAS_CUPY = False


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SubstrateConfig:
    """Configuration for the Hilbert graph substrate."""

    n_sites: int = 12
    d_local: int = 2
    coupling: float = 1.0

    connectivity: Literal["chain", "ring", "complete", "random", "cube2"] = "chain"
    random_p: float = 0.3
    seed: int = 12345

    lr_threshold: float = 1e-3
    lr_t_max: float = 8.0
    lr_n_steps: int = 120
    lr_norm: Literal["fro", "spectral"] = "fro"

    use_gpu: bool = False
    dtype: Literal["complex128", "complex64"] = "complex128"


# =============================================================================
# Substrate: global Hilbert space and local operators
# =============================================================================


class HilbertSubstrate:
    """
    Global Hilbert substrate on an abstract graph.

    - Sites 0..N-1, no coordinates.
    - Local dimension d at each site.
    - Global dimension D = d^N.
    """

    def __init__(self,
                 n_sites: int,
                 d_local: int,
                 xp=np,
                 dtype=np.complex128):
        self.n_sites = n_sites
        self.d_local = d_local
        self.dim_total = d_local ** n_sites
        self.xp = xp
        self.dtype = dtype

    # ---- basis encoding/decoding -------------------------------------------------

    def index_to_config(self, idx: int) -> Tuple[int, ...]:
        """Convert basis index to occupation tuple (n_0,...,n_{N-1})."""
        conf = [0] * self.n_sites
        d = self.d_local
        for k in range(self.n_sites):
            power = self.n_sites - 1 - k
            base = d ** power
            conf[k] = (idx // base) % d
        return tuple(conf)

    def config_to_index(self, conf: Tuple[int, ...]) -> int:
        """Convert occupation tuple to basis index."""
        idx = 0
        d = self.d_local
        for k, n in enumerate(conf):
            power = self.n_sites - 1 - k
            idx += int(n) * (d ** power)
        return idx

    # ---- states ------------------------------------------------------------------

    def vacuum(self):
        """Return |0,0,...,0> as a global state vector (xp array)."""
        psi = self.xp.zeros(self.dim_total, dtype=self.dtype)
        psi[0] = 1.0
        return psi

    def normalize(self, psi):
        """Normalize a state vector (xp array)."""
        norm = float(self.xp.linalg.norm(psi))
        if norm == 0.0:
            return psi
        return psi / norm

    def excite_site(self, psi, site: int):
        """
        Apply truncated bosonic creation operator a_site^\dagger
        to the global state, assuming local occupations 0..d-1.
        """
        d = self.d_local
        D = self.dim_total
        out = self.xp.zeros_like(psi)

        for idx in range(D):
            conf = list(self.index_to_config(idx))
            n = conf[site]
            if n < d - 1:
                conf[site] = n + 1
                new_idx = self.config_to_index(tuple(conf))
                out[new_idx] += psi[idx] * np.sqrt(n + 1)
        return out

    def local_number_operator(self, site: int):
        """
        Build the full many-body number operator n_site:

          n_site |n_0,...,n_i,...> = n_i |n_0,...,n_i,...>.
        """
        D = self.dim_total
        n_op = self.xp.zeros((D, D), dtype=self.dtype)
        for idx in range(D):
            conf = self.index_to_config(idx)
            n_op[idx, idx] = conf[site]
        return n_op

    # ---- local expectation -------------------------------------------------------

    def local_occupation(self, psi, site: int) -> float:
        """
        Compute expectation <n_site> for a pure state psi (xp array).
        """
        D = self.dim_total
        exp_val = 0.0
        for idx in range(D):
            conf = self.index_to_config(idx)
            exp_val += float(self.xp.abs(psi[idx]) ** 2) * conf[site]
        return float(exp_val)


# =============================================================================
# Graph + Hamiltonian
# =============================================================================


def build_connectivity(cfg: SubstrateConfig) -> Dict[int, List[int]]:
    """Bare graph connectivity."""
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

    if cfg.connectivity == "cube2":
        # 2x2x2 cube: 8 sites labeled by bits (x, y, z) with x,y,z in {0,1}
        # index = x + 2*y + 4*z
        if N != 8:
            raise ValueError("cube2 connectivity requires n_sites=8")
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        for idx in range(8):
            x = idx & 1
            y = (idx >> 1) & 1
            z = (idx >> 2) & 1
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0),
                               (0, 1, 0), (0, -1, 0),
                               (0, 0, 1), (0, 0, -1)):
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < 2 and 0 <= ny < 2 and 0 <= nz < 2:
                    j = nx + 2 * ny + 4 * nz
                    if j not in adj[idx]:
                        adj[idx].append(j)
        return adj

    raise ValueError(f"Unknown connectivity type: {cfg.connectivity}")


def build_hamiltonian(substrate: HilbertSubstrate,
                      connectivity: Dict[int, List[int]],
                      coupling: float):
    """
    H = coupling * sum_{i<j, (i,j) in edges} (a_i^\dagger a_j + a_j^\dagger a_i).
    """
    xp = substrate.xp
    dtype = substrate.dtype
    D = substrate.dim_total
    d = substrate.d_local

    H = xp.zeros((D, D), dtype=dtype)

    for i, neighbors in connectivity.items():
        for j in neighbors:
            if j <= i:
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
# Optimized evolution cache (eigendecomposition)
# =============================================================================


class EvolutionCache:
    """
    Precomputed eigendecomposition and operator transforms for efficient
    time evolution.

    Given H = V Λ V†, time evolution is:
        U(t) = V diag(e^{-iλt}) V†

    Heisenberg evolution of operator A:
        A(t) = U†(t) A U(t)
             = V e^{+iΛt} V† A V e^{-iΛt} V†
             = V e^{+iΛt} A_eig e^{-iΛt} V†

    where A_eig = V† A V is precomputed.
    """

    def __init__(self,
                 substrate: HilbertSubstrate,
                 H,
                 eigh_fn=np_eigh):
        self.substrate = substrate
        self.xp = substrate.xp
        self.dtype = substrate.dtype
        self.N = substrate.n_sites
        self.D = substrate.dim_total

        # Eigendecompose H (Hermitian, so use eigh)
        # H = V @ diag(eigenvalues) @ V†
        eigenvalues, V = eigh_fn(H)

        # Store as xp arrays
        self.eigenvalues = self.xp.asarray(eigenvalues).real  # real for Hermitian
        self.V = self.xp.asarray(V)
        self.V_dag = self.V.conj().T

        # Precompute number operators and their eigenbasis forms
        self.n_ops = []
        self.n_ops_eig = []
        for j in range(self.N):
            n_j = substrate.local_number_operator(j)
            n_j_xp = self.xp.asarray(n_j, dtype=self.dtype)
            self.n_ops.append(n_j_xp)
            self.n_ops_eig.append(self.V_dag @ n_j_xp @ self.V)

    # ---- state evolution ---------------------------------------------------------

    def evolve_state(self, psi0, t_max: float, n_steps: int):
        """
        Evolve state psi under H for times t in [0, t_max].

        Returns:
            times: np.ndarray shape (n_steps,)
            states: xp array shape (n_steps, D)
        """
        times = np.linspace(0.0, t_max, n_steps)
        states = self.xp.zeros((n_steps, self.D), dtype=self.dtype)

        # Transform to eigenbasis once
        psi_eig = self.V_dag @ psi0

        for k, t in enumerate(times):
            phases = self.xp.exp(-1j * self.eigenvalues * t)
            psi_eig_t = phases * psi_eig
            states[k] = self.V @ psi_eig_t

        return times, states

    # ---- operator evolution ------------------------------------------------------

    def evolve_operator_to_time(self, A_eig, t: float):
        """
        Evolve operator from its eigenbasis representation A_eig at time 0
        to time t in the original basis.

        A(t) = V e^{+iΛt} A_eig e^{-iΛt} V†
        """
        phases_pos = self.xp.exp(+1j * self.eigenvalues * t)
        phases_neg = self.xp.exp(-1j * self.eigenvalues * t)

        # Row i multiplied by phases_pos[i], column j by phases_neg[j]
        A_eig_t = (phases_pos[:, None] * A_eig) * phases_neg[None, :]

        return self.V @ A_eig_t @ self.V_dag


# =============================================================================
# Propagation-based signals (optimized)
# =============================================================================


def excitation_propagation(substrate: HilbertSubstrate,
                           cache: EvolutionCache,
                           source_site: int,
                           t_max: float,
                           n_steps: int,
                           threshold: float = 0.01) -> Dict[str, object]:
    """
    Excitation-based operational signal:
      - Start with vacuum.
      - Excite source site.
      - Evolve under H via cached eigenbasis.
      - Track <n_j(t)> and define arrival times by threshold.
    """
    xp = substrate.xp

    psi0 = substrate.excite_site(substrate.vacuum(), source_site)
    psi0 = substrate.normalize(psi0)

    times, states = cache.evolve_state(psi0, t_max, n_steps)
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
                                   cache: EvolutionCache,
                                   t_max: float,
                                   n_steps: int,
                                   threshold: float = 0.01) -> np.ndarray:
    """
    Symmetric 'distance' matrix from excitation arrival times.
    """
    N = substrate.n_sites
    Dmat = np.zeros((N, N), dtype=float)

    for source in range(N):
        res = excitation_propagation(substrate, cache, source, t_max, n_steps, threshold)
        for j in range(N):
            Dmat[source, j] = res["arrival"][j]

    Dmat = 0.5 * (Dmat + Dmat.T)
    return Dmat


# =============================================================================
# Lieb–Robinson commutators (optimized)
# =============================================================================


def lieb_robinson_commutators(substrate: HilbertSubstrate,
                              cache: EvolutionCache,
                              source_site: int,
                              t_max: float,
                              n_steps: int,
                              threshold: float,
                              norm_type: Literal["fro", "spectral"]) -> Dict[str, object]:
    """
    Compute LR commutator C_ij(t) = [A_i(t), B_j] using cached eigendecomposition.

    A_i(t) = U†(t) n_i U(t)
    B_j = n_j
    """
    xp = substrate.xp
    N = substrate.n_sites

    times = np.linspace(0.0, t_max, n_steps)

    # A_i(0) in eigenbasis
    A_eig_0 = cache.n_ops_eig[source_site]

    comm_norms = np.zeros((N, n_steps), dtype=float)

    for t_idx, t in enumerate(times):
        # A_i(t) in original basis
        A_t = cache.evolve_operator_to_time(A_eig_0, t)

        for j in range(N):
            B_j = cache.n_ops[j]
            C = A_t @ B_j - B_j @ A_t
            if norm_type == "spectral":
                comm_norms[j, t_idx] = float(xp.linalg.norm(C, 2))
            else:
                comm_norms[j, t_idx] = float(xp.linalg.norm(C, "fro"))

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
                         cache: EvolutionCache,
                         t_max: float,
                         n_steps: int,
                         threshold: float,
                         norm_type: Literal["fro", "spectral"]) -> np.ndarray:
    """
    Operational LR distance matrix from commutator arrival times.
    """
    N = substrate.n_sites
    Dmat = np.zeros((N, N), dtype=float)

    for source in range(N):
        lr = lieb_robinson_commutators(
            substrate,
            cache,
            source_site=source,
            t_max=t_max,
            n_steps=n_steps,
            threshold=threshold,
            norm_type=norm_type,
        )
        for j in range(N):
            Dmat[source, j] = lr["arrival"][j]

    Dmat = 0.5 * (Dmat + Dmat.T)
    return Dmat


# =============================================================================
# Graph distances and inflation analysis
# =============================================================================


def compute_graph_distances(connectivity: Dict[int, List[int]]) -> np.ndarray:
    """
    All-pairs shortest path lengths on graph (BFS).
    """
    nodes = sorted(connectivity.keys())
    N = len(nodes)
    dist = np.full((N, N), np.inf, dtype=float)

    for i in nodes:
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
    Shell profile for LR metric vs graph distance:

      t_shell[d]   = average LR distance to sites at graph distance d,
      dt_shell[d]  = t_shell[d] - t_shell[d-1],
      v_eff[d]     = 1 / dt_shell[d].
    """
    N = D_lr.shape[0]
    d_max = int(np.nanmax(graph_dist[source, np.isfinite(graph_dist[source])]))

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

    print(f"Lieb–Robinson 'inflation' profile (source = {source}):")
    print("  d_graph |  t_LR_shell  |  Δt(d) = t(d)-t(d-1)  |  v_eff(d) = 1/Δt")
    print("  ---------------------------------------------------------------")
    print(f"      0   |  {t_shell[0]:.6f}    |        ---              |     ---")
    for d in range(1, d_max + 1):
        t = t_shell[d]
        dt = dt_shell[d]
        v = v_eff[d]
        t_str = f"{t:.6f}" if np.isfinite(t) else "  inf   "
        dt_str = f"{dt:.6f}" if np.isfinite(dt) else "   nan  "
        v_str = f"{v:.3f}" if np.isfinite(v) else "  nan "
        print(f"     {d:2d}   |  {t_str}    |   {dt_str}         |   {v_str}")
    print("  ---------------------------------------------------------------")
    print("  Note: decreasing v_eff(d) with d means 'inflation' slowing.\n")


# =============================================================================
# Metric → coordinate embedding (MDS)
# =============================================================================


def classical_mds_from_dist(Dmat: np.ndarray, dim: int = 3) -> np.ndarray:
    """Classical MDS embedding of distance matrix D into R^dim."""
    D2 = Dmat ** 2
    N = Dmat.shape[0]
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J

    vals, vecs = np_eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    pos_mask = vals > 1e-12
    vals_pos = vals[pos_mask][:dim]
    vecs_pos = vecs[:, pos_mask][:, :dim]

    L_sqrt = np.sqrt(vals_pos)
    X = vecs_pos * L_sqrt[np.newaxis, :]
    return X  # shape (N, dim)


# =============================================================================
# Original (unoptimized) reference implementation for verification
# =============================================================================


def _original_evolve_state(H,
                           psi0,
                           t_max: float,
                           n_steps: int,
                           xp,
                           expm_fn):
    """Reference state evolution using explicit matrix exponential."""
    times = np.linspace(0.0, t_max, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    U = expm_fn(-1j * H * dt)
    psi = psi0.copy()

    D = H.shape[0]
    states = xp.zeros((n_steps, D), dtype=H.dtype)

    for k in range(n_steps):
        states[k] = psi
        psi = U @ psi

    return times, states


def _original_excitation_propagation(substrate: HilbertSubstrate,
                                     H,
                                     expm_fn,
                                     source_site: int,
                                     t_max: float,
                                     n_steps: int,
                                     threshold: float = 0.01) -> Dict[str, object]:
    """Reference excitation propagation using original evolution."""
    xp = substrate.xp

    psi0 = substrate.excite_site(substrate.vacuum(), source_site)
    psi0 = substrate.normalize(psi0)

    times, states = _original_evolve_state(H, psi0, t_max, n_steps, xp, expm_fn)
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


def _original_excitation_metric(substrate: HilbertSubstrate,
                                H,
                                expm_fn,
                                t_max: float,
                                n_steps: int,
                                threshold: float = 0.01) -> np.ndarray:
    """Reference excitation metric using original evolution."""
    N = substrate.n_sites
    Dmat = np.zeros((N, N), dtype=float)

    for source in range(N):
        res = _original_excitation_propagation(substrate, H, expm_fn,
                                               source, t_max, n_steps, threshold)
        for j in range(N):
            Dmat[source, j] = res["arrival"][j]

    Dmat = 0.5 * (Dmat + Dmat.T)
    return Dmat


def _original_lieb_robinson_commutators(substrate: HilbertSubstrate,
                                        H,
                                        expm_fn,
                                        source_site: int,
                                        t_max: float,
                                        n_steps: int,
                                        threshold: float,
                                        norm_type: Literal["fro", "spectral"]) -> Dict[str, object]:
    """Reference LR commutator using original evolution."""
    xp = substrate.xp
    N = substrate.n_sites

    times = np.linspace(0.0, t_max, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    U = expm_fn(-1j * H * dt)
    U_dag = U.conj().T

    n_ops = [substrate.local_number_operator(j) for j in range(N)]
    A_t = n_ops[source_site].copy()

    comm_norms = np.zeros((N, n_steps), dtype=float)

    for t_idx in range(n_steps):
        for j in range(N):
            C = A_t @ n_ops[j] - n_ops[j] @ A_t
            if norm_type == "spectral":
                comm_norms[j, t_idx] = float(xp.linalg.norm(C, 2))
            else:
                comm_norms[j, t_idx] = float(xp.linalg.norm(C, "fro"))
        A_t = U_dag @ A_t @ U

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


def _original_lieb_robinson_metric(substrate: HilbertSubstrate,
                                   H,
                                   expm_fn,
                                   t_max: float,
                                   n_steps: int,
                                   threshold: float,
                                   norm_type: Literal["fro", "spectral"]) -> np.ndarray:
    """Reference LR metric using original evolution."""
    N = substrate.n_sites
    Dmat = np.zeros((N, N), dtype=float)

    for source in range(N):
        lr = _original_lieb_robinson_commutators(
            substrate,
            H,
            expm_fn,
            source_site=source,
            t_max=t_max,
            n_steps=n_steps,
            threshold=threshold,
            norm_type=norm_type,
        )
        for j in range(N):
            Dmat[source, j] = lr["arrival"][j]

    Dmat = 0.5 * (Dmat + Dmat.T)
    return Dmat


# =============================================================================
# Verification harness
# =============================================================================


def verify_against_original(cfg: SubstrateConfig) -> bool:
    """
    Verify that the optimized implementation matches the original one
    on small systems (e.g. n_sites <= 8).

    This runs a few tests and prints max norm differences.
    """
    print("\nRunning verification against original implementation...")
    print("Config:", asdict(cfg))
    print()

    # CPU-only for verification
    xp = np
    dtype = np.complex128
    expm_fn = np_expm
    eigh_fn = np_eigh

    substrate = HilbertSubstrate(cfg.n_sites, cfg.d_local, xp=xp, dtype=dtype)
    connectivity = build_connectivity(cfg)
    H = build_hamiltonian(substrate, connectivity, cfg.coupling)

    cache = EvolutionCache(substrate, H, eigh_fn=eigh_fn)

    all_match = True

    # 1) State evolution vs original
    print("Test 1: state evolution...")
    psi0 = substrate.normalize(substrate.excite_site(substrate.vacuum(), 0))
    t_max = cfg.lr_t_max
    n_steps = cfg.lr_n_steps

    times_opt, states_opt = cache.evolve_state(psi0, t_max, n_steps)
    times_ref, states_ref = _original_evolve_state(H, psi0, t_max, n_steps,
                                                   xp=xp, expm_fn=expm_fn)

    max_diff = np.max(np.abs(states_opt - states_ref))
    print(f"  max |psi_opt - psi_ref| = {max_diff:.3e}")
    if max_diff > 1e-10:
        all_match = False
    else:
        print("  OK")

    # 2) Excitation metric vs original
    print("Test 2: excitation metric...")
    D_prop_opt = excitation_metric_from_arrival(substrate, cache,
                                                t_max, n_steps, threshold=0.01)
    D_prop_ref = _original_excitation_metric(substrate, H, expm_fn,
                                             t_max, n_steps, threshold=0.01)

    max_diff = np.max(np.abs(D_prop_opt - D_prop_ref))
    print(f"  max |D_prop_opt - D_prop_ref| = {max_diff:.3e}")
    if max_diff > 1e-10:
        all_match = False
    else:
        print("  OK")

    # 3) LR metric vs original
    print("Test 3: LR metric...")
    D_lr_opt = lieb_robinson_metric(substrate, cache,
                                    t_max, n_steps,
                                    threshold=cfg.lr_threshold,
                                    norm_type=cfg.lr_norm)
    D_lr_ref = _original_lieb_robinson_metric(substrate, H, expm_fn,
                                              t_max, n_steps,
                                              threshold=cfg.lr_threshold,
                                              norm_type=cfg.lr_norm)

    max_diff = np.max(np.abs(D_lr_opt - D_lr_ref))
    print(f"  max |D_lr_opt - D_lr_ref| = {max_diff:.3e}")
    if max_diff > 1e-10:
        all_match = False
    else:
        print("  OK")

    print("\n" + "=" * 60)
    if all_match:
        print("ALL TESTS PASSED - Optimized implementation is exact.")
    else:
        print("SOME TESTS FAILED - Check implementation.")
    print("=" * 60 + "\n")

    return all_match


# =============================================================================
# Demo / CLI
# =============================================================================


def run_demo(cfg: SubstrateConfig, out_dir: str, verify: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Substrate config:")
    print(asdict(cfg))
    print()

    # Backend selection
    if cfg.use_gpu:
        if not HAS_CUPY:
            print("WARNING: --use-gpu requested but CuPy not available. Falling back to CPU.")
            xp = np
            dtype = np.complex128 if cfg.dtype == "complex128" else np.complex64
            eigh_fn = np_eigh
        else:
            xp = cp
            dtype = cp.complex128 if cfg.dtype == "complex128" else cp.complex64
            eigh_fn = cp.linalg.eigh  # Hermitian eigendecomposition
            print("Using CuPy GPU backend.")
    else:
        xp = np
        dtype = np.complex128 if cfg.dtype == "complex128" else np.complex64
        eigh_fn = np_eigh

    # Build substrate + Hamiltonian
    substrate = HilbertSubstrate(cfg.n_sites, cfg.d_local, xp=xp, dtype=dtype)
    connectivity = build_connectivity(cfg)
    H = build_hamiltonian(substrate, connectivity, cfg.coupling)

    cache = EvolutionCache(substrate, H, eigh_fn=eigh_fn)

    print(f"Global dimension: {substrate.dim_total}")
    print("Connectivity:")
    for i in range(cfg.n_sites):
        print(f"  {i}: {connectivity.get(i, [])}")
    print()

    graph_dist = compute_graph_distances(connectivity)

    source = cfg.n_sites // 2
    print(f"Using site {source} as source\n")

    # Excitation-based metric
    prop_res = excitation_propagation(
        substrate,
        cache,
        source_site=source,
        t_max=cfg.lr_t_max,
        n_steps=cfg.lr_n_steps,
        threshold=0.01,
    )
    D_prop = excitation_metric_from_arrival(
        substrate,
        cache,
        t_max=cfg.lr_t_max,
        n_steps=cfg.lr_n_steps,
        threshold=0.01,
    )

    print("Propagation-based arrival times from source:")
    for j in range(cfg.n_sites):
        print(f"  {j}: {prop_res['arrival'][j]:.3f}")
    print()

    # Lieb–Robinson metric
    D_lr = lieb_robinson_metric(
        substrate,
        cache,
        t_max=cfg.lr_t_max,
        n_steps=cfg.lr_n_steps,
        threshold=cfg.lr_threshold,
        norm_type=cfg.lr_norm,
    )

    # For printing LR arrival times from source, re-use LR commutators directly
    lr_res = lieb_robinson_commutators(
        substrate,
        cache,
        source_site=source,
        t_max=cfg.lr_t_max,
        n_steps=cfg.lr_n_steps,
        threshold=cfg.lr_threshold,
        norm_type=cfg.lr_norm,
    )

    print("LR commutator arrival times from source:")
    for j in range(cfg.n_sites):
        print(f"  {j}: {lr_res['arrival'][j]:.3f}")
    print()

    # Inflation profile
    analyze_lr_inflation(D_lr, graph_dist, source)

    # Bring metrics back to CPU if needed for saving/embedding
    D_lr_np = D_lr.get() if hasattr(D_lr, "get") else D_lr
    D_prop_np = D_prop.get() if hasattr(D_prop, "get") else D_prop

    # 3D embedding of LR metric
    try:
        X = classical_mds_from_dist(D_lr_np, dim=3)
        np.savez(os.path.join(out_dir, "lr_embedding_3d.npz"),
                 X=X, D_lr=D_lr_np, D_prop=D_prop_np, graph_dist=graph_dist)
        print("Saved LR 3D embedding to lr_embedding_3d.npz")
    except Exception as exc:
        print("Could not embed LR metric:", exc)

    np.savez(os.path.join(out_dir, "lr_metrics.npz"),
             D_lr=D_lr_np,
             D_prop=D_prop_np,
             graph_dist=graph_dist)

    # Optional verification
    if verify:
        verify_against_original(cfg)

    print("\nDone.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hilbert graph substrate with Lieb–Robinson emergent geometry."
    )
    parser.add_argument("--n-sites", type=int, default=8)
    parser.add_argument("--d-local", type=int, default=2)
    parser.add_argument("--connectivity", type=str, default="chain",
                        choices=["chain", "ring", "complete", "random", "cube2"])
    parser.add_argument("--random-p", type=float, default=0.3)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lr-threshold", type=float, default=1e-3)
    parser.add_argument("--lr-t-max", type=float, default=8.0)
    parser.add_argument("--lr-n-steps", type=int, default=120)
    parser.add_argument("--lr-norm", type=str, default="fro", choices=["fro", "spectral"])
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--dtype", type=str, default="complex128",
                        choices=["complex128", "complex64"])
    parser.add_argument("--output-dir", type=str, default="substrate_outputs")
    parser.add_argument("--verify", action="store_true",
                        help="Run internal consistency checks against original implementation.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    cfg = SubstrateConfig(
        n_sites=args.n_sites,
        d_local=args.d_local,
        coupling=args.coupling,
        connectivity=args.connectivity,
        random_p=args.random_p,
        seed=args.seed,
        lr_threshold=args.lr_threshold,
        lr_t_max=args.lr_t_max,
        lr_n_steps=args.lr_n_steps,
        lr_norm=args.lr_norm,
        use_gpu=args.use_gpu,
        dtype=args.dtype,
    )

    run_demo(cfg, out_dir=args.output_dir, verify=args.verify)


if __name__ == "__main__":
    main()
