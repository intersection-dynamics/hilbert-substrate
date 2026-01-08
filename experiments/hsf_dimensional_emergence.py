#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
HSF Dimensional Emergence Study
===============================

Tests the hypothesis that 3D geometry emerges uniquely from the Hilbert
Substrate under the No-Refolding constraint.

Approach:
---------
We construct Hamiltonians on interaction graphs with different intrinsic
dimensions (1D chain, 2D lattice, 3D cubic, 4D hypercubic, random regular).
After global scrambling, we apply the locality recovery flow and measure:

1. Recovery Rate: Can the flow find the local basin?
2. Basin Depth: How much does locality improve?
3. Spectral Dimension: What effective dimension does the recovered H have?
4. Stability: How robust is the recovered structure to perturbations?

If 3D is special, we expect to see a signature in these metrics.

Usage:
------
python hsf_dimensional_emergence.py --out ./results --N 8 --seeds 10 --progress

Requirements: numpy (scipy optional for larger systems)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

# =============================================================================
# UTILITIES
# =============================================================================

def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()

def zip_folder(folder: Path) -> Path:
    zip_path = Path(str(folder) + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(folder)
                z.write(full, arcname=str(rel))
    return zip_path

# =============================================================================
# GRAPH TOPOLOGIES
# =============================================================================

@dataclass
class GraphTopology:
    """Represents an interaction graph topology."""
    name: str
    dimension: Optional[int]  # None for non-geometric graphs
    edges: List[Tuple[int, int]]
    N: int
    metadata: Dict = field(default_factory=dict)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)
    
    @property 
    def degree(self) -> float:
        """Average degree of the graph."""
        return 2 * self.num_edges / self.N
    
    def adjacency_dict(self) -> Dict[int, Set[int]]:
        """Return adjacency list representation."""
        adj = {i: set() for i in range(self.N)}
        for (i, j) in self.edges:
            adj[i].add(j)
            adj[j].add(i)
        return adj


def make_1d_ring(N: int) -> GraphTopology:
    """1D periodic chain (ring)."""
    edges = [(i, (i + 1) % N) for i in range(N)]
    return GraphTopology(
        name="1D_ring",
        dimension=1,
        edges=edges,
        N=N,
        metadata={"boundary": "periodic", "L": N}
    )


def make_1d_chain(N: int) -> GraphTopology:
    """1D open chain."""
    edges = [(i, i + 1) for i in range(N - 1)]
    return GraphTopology(
        name="1D_chain",
        dimension=1,
        edges=edges,
        N=N,
        metadata={"boundary": "open", "L": N}
    )


def make_2d_lattice(Lx: int, Ly: int, periodic: bool = True) -> GraphTopology:
    """2D square lattice."""
    N = Lx * Ly
    edges = []
    
    def idx(x, y):
        return x * Ly + y
    
    for x in range(Lx):
        for y in range(Ly):
            # Right neighbor
            if periodic or x < Lx - 1:
                nx = (x + 1) % Lx
                edges.append((idx(x, y), idx(nx, y)))
            # Up neighbor
            if periodic or y < Ly - 1:
                ny = (y + 1) % Ly
                edges.append((idx(x, y), idx(x, ny)))
    
    return GraphTopology(
        name="2D_lattice",
        dimension=2,
        edges=edges,
        N=N,
        metadata={"boundary": "periodic" if periodic else "open", "Lx": Lx, "Ly": Ly}
    )


def make_3d_lattice(Lx: int, Ly: int, Lz: int, periodic: bool = True) -> GraphTopology:
    """3D cubic lattice."""
    N = Lx * Ly * Lz
    edges = []
    
    def idx(x, y, z):
        return x * Ly * Lz + y * Lz + z
    
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                # X neighbor
                if periodic or x < Lx - 1:
                    nx = (x + 1) % Lx
                    edges.append((idx(x, y, z), idx(nx, y, z)))
                # Y neighbor
                if periodic or y < Ly - 1:
                    ny = (y + 1) % Ly
                    edges.append((idx(x, y, z), idx(x, ny, z)))
                # Z neighbor
                if periodic or z < Lz - 1:
                    nz = (z + 1) % Lz
                    edges.append((idx(x, y, z), idx(x, y, nz)))
    
    return GraphTopology(
        name="3D_lattice",
        dimension=3,
        edges=edges,
        N=N,
        metadata={"boundary": "periodic" if periodic else "open", "Lx": Lx, "Ly": Ly, "Lz": Lz}
    )


def make_4d_lattice(L: int, periodic: bool = True) -> GraphTopology:
    """4D hypercubic lattice with side length L."""
    N = L ** 4
    edges = []
    
    def idx(x, y, z, w):
        return x * L**3 + y * L**2 + z * L + w
    
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for w in range(L):
                    # Each of 4 directions
                    if periodic or x < L - 1:
                        edges.append((idx(x, y, z, w), idx((x+1)%L, y, z, w)))
                    if periodic or y < L - 1:
                        edges.append((idx(x, y, z, w), idx(x, (y+1)%L, z, w)))
                    if periodic or z < L - 1:
                        edges.append((idx(x, y, z, w), idx(x, y, (z+1)%L, w)))
                    if periodic or w < L - 1:
                        edges.append((idx(x, y, z, w), idx(x, y, z, (w+1)%L)))
    
    return GraphTopology(
        name="4D_lattice",
        dimension=4,
        edges=edges,
        N=N,
        metadata={"boundary": "periodic" if periodic else "open", "L": L}
    )


def make_random_regular(N: int, degree: int, rng: np.random.Generator) -> GraphTopology:
    """
    Random regular graph with specified degree.
    Uses configuration model with rejection for simplicity.
    """
    if N * degree % 2 != 0:
        raise ValueError("N * degree must be even")
    
    max_attempts = 1000
    for attempt in range(max_attempts):
        # Configuration model: create degree stubs per node
        stubs = []
        for node in range(N):
            stubs.extend([node] * degree)
        
        rng.shuffle(stubs)
        
        edges = set()
        valid = True
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:  # Self-loop
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:  # Multi-edge
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return GraphTopology(
                name=f"random_regular_d{degree}",
                dimension=None,  # No intrinsic geometry
                edges=list(edges),
                N=N,
                metadata={"degree": degree, "seed_attempt": attempt}
            )
    
    raise RuntimeError(f"Failed to generate random regular graph after {max_attempts} attempts")


def make_complete_graph(N: int) -> GraphTopology:
    """Complete graph K_N - maximally connected, no locality."""
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    return GraphTopology(
        name="complete",
        dimension=None,  # No meaningful dimension
        edges=edges,
        N=N,
        metadata={"type": "complete"}
    )

# =============================================================================
# QUANTUM OPERATORS
# =============================================================================

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = {"I": I, "X": X, "Y": Y, "Z": Z}


def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of matrices."""
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def op_on_site(op: np.ndarray, N: int, site: int) -> np.ndarray:
    """Embed single-site operator at specified site."""
    mats = [I if i != site else op for i in range(N)]
    return kron_all(mats)


def two_site_op(opA: np.ndarray, opB: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    """Two-site operator with opA at site i, opB at site j."""
    mats = []
    for k in range(N):
        if k == i:
            mats.append(opA)
        elif k == j:
            mats.append(opB)
        else:
            mats.append(I)
    return kron_all(mats)


def hermitian_rand(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random Hermitian matrix."""
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (a + a.conj().T) / 2.0


def unitary_from_hermitian(h: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Unitary from matrix exponential of Hermitian."""
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * t * w)) @ v.conj().T


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary matrix."""
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.maximum(1e-12, np.abs(d))
    return q * ph

# =============================================================================
# HAMILTONIAN CONSTRUCTION
# =============================================================================

def build_heisenberg_hamiltonian(graph: GraphTopology, J: float = 1.0) -> np.ndarray:
    """
    Build Heisenberg XXX Hamiltonian on the given graph:
    H = J * sum_{<i,j>} (X_i X_j + Y_i Y_j + Z_i Z_j)
    """
    N = graph.N
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    
    for (i, j) in graph.edges:
        for P in [X, Y, Z]:
            H += J * two_site_op(P, P, N, i, j)
    
    return H


def build_xx_hamiltonian(graph: GraphTopology, J: float = 1.0) -> np.ndarray:
    """
    Build XX Hamiltonian on the given graph:
    H = (J/2) * sum_{<i,j>} (X_i X_j + Y_i Y_j)
    """
    N = graph.N
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    
    for (i, j) in graph.edges:
        H += (J / 2.0) * (two_site_op(X, X, N, i, j) + two_site_op(Y, Y, N, i, j))
    
    return H

# =============================================================================
# LOCAL BASIS AND LOCALITY METRIC
# =============================================================================

@dataclass
class LocalBasis:
    """Local operator basis for a given graph topology."""
    ops: List[np.ndarray]
    weights: List[int]  # Locality weight (1 = single-site, 2 = two-site, etc.)
    tags: List[str]


def build_local_basis(graph: GraphTopology) -> LocalBasis:
    """
    Build basis of local operators: single-site Paulis + two-site Paulis on edges.
    """
    N = graph.N
    ops = []
    weights = []
    tags = []
    
    # Single-site terms (weight 1)
    for i in range(N):
        for name, P in [("X", X), ("Y", Y), ("Z", Z)]:
            ops.append(op_on_site(P, N, i))
            weights.append(1)
            tags.append(f"{name}_{i}")
    
    # Two-site terms on edges (weight 2)
    for (i, j) in graph.edges:
        for na, Pa in [("X", X), ("Y", Y), ("Z", Z)]:
            for nb, Pb in [("X", X), ("Y", Y), ("Z", Z)]:
                ops.append(two_site_op(Pa, Pb, N, i, j))
                weights.append(2)
                tags.append(f"{na}{nb}_{i}-{j}")
    
    return LocalBasis(ops=ops, weights=weights, tags=tags)


def frob_norm_sq(H: np.ndarray) -> float:
    """Squared Frobenius norm."""
    return float(np.vdot(H, H).real)


def project_to_local_basis(H: np.ndarray, basis: LocalBasis, N: int) -> float:
    """
    Compute ||P_local(H)||^2 where P_local projects onto local subspace.
    """
    d = 2 ** N
    acc = 0.0
    for op in basis.ops:
        # Coefficient c_k = Tr(P_k^dag H) / d
        coef = np.trace(op.conj().T @ H) / d
        acc += float((coef.conjugate() * coef).real) * d  # Contribution to norm^2
    return acc


def locality_metric(H: np.ndarray, basis: LocalBasis, N: int) -> Dict[str, float]:
    """
    Compute locality metrics:
    - local_frac: fraction of H in local subspace
    - leak_frac: 1 - local_frac (fraction non-local)
    """
    total = frob_norm_sq(H)
    local = project_to_local_basis(H, basis, N)
    local_frac = local / (total + 1e-18)
    leak_frac = 1.0 - local_frac
    return {
        "total_norm_sq": total,
        "local_norm_sq": local,
        "local_frac": local_frac,
        "leak_frac": leak_frac
    }

# =============================================================================
# SCRAMBLING
# =============================================================================

def scramble_local(H: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """Apply independent random single-site unitaries."""
    mats = [unitary_from_hermitian(hermitian_rand(2, rng)) for _ in range(N)]
    U = kron_all(mats)
    return U @ H @ U.conj().T


def scramble_global(H: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """Apply Haar-random global unitary."""
    U = haar_unitary(2 ** N, rng)
    return U @ H @ U.conj().T


def embed_two_qubit_gate_general(U2: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    """
    Embed a 2-qubit gate acting on qubits i and j (not necessarily adjacent).
    """
    if i > j:
        i, j = j, i
    
    dim = 2 ** N
    U_full = np.zeros((dim, dim), dtype=complex)
    
    # For each computational basis state, apply U2 to qubits i, j
    for basis_idx in range(dim):
        # Extract bits
        bits = [(basis_idx >> k) & 1 for k in range(N)]
        bi, bj = bits[i], bits[j]
        
        # Input state for U2: |bi, bj>
        input_2q = bi * 2 + bj
        
        # Apply U2
        for output_2q in range(4):
            coef = U2[output_2q, input_2q]
            if abs(coef) < 1e-15:
                continue
            
            # New bits
            new_bi = (output_2q >> 1) & 1
            new_bj = output_2q & 1
            
            new_bits = bits.copy()
            new_bits[i] = new_bi
            new_bits[j] = new_bj
            
            new_idx = sum(b << k for k, b in enumerate(new_bits))
            U_full[new_idx, basis_idx] += coef
    
    return U_full


def scramble_layer(H: np.ndarray, graph: GraphTopology, rng: np.random.Generator, depth: int = 1) -> np.ndarray:
    """
    Apply random two-site unitaries on graph edges (circuit-style scrambling).
    This scrambles respecting the graph structure.
    """
    N = graph.N
    H_out = H.copy()
    
    for _ in range(depth):
        for (i, j) in graph.edges:
            # Random 2-qubit unitary
            U2 = unitary_from_hermitian(hermitian_rand(4, rng))
            U_full = embed_two_qubit_gate_general(U2, N, i, j)
            H_out = U_full @ H_out @ U_full.conj().T
    
    return H_out

# =============================================================================
# LOCALITY RECOVERY FLOW
# =============================================================================

@dataclass
class FlowParams:
    steps: int = 2000
    eps: float = 0.06
    temp0: float = 0.02
    temp_decay: float = 0.9995
    cost_every: int = 1


def flow_recover(H: np.ndarray, graph: GraphTopology, basis: LocalBasis,
                 rng: np.random.Generator, params: FlowParams,
                 use_graph_gates: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Stochastic locality recovery flow.
    
    If use_graph_gates=True, applies 2-qubit gates only on graph edges.
    Otherwise, uses adjacent qubit gates (original behavior).
    """
    N = graph.N
    H_curr = H.copy()
    H_best = H.copy()
    
    metrics = locality_metric(H_curr, basis, N)
    cost = metrics["leak_frac"]
    best_cost = cost
    temp = params.temp0
    
    accepted = 0
    evaluated = 0
    cost_history = [cost]
    
    # Prepare gate options
    if use_graph_gates:
        gate_edges = graph.edges
    else:
        gate_edges = [(i, i + 1) for i in range(N - 1)]
    
    for step in range(params.steps):
        # Pick random edge to apply gate
        edge_idx = int(rng.integers(0, len(gate_edges)))
        i, j = gate_edges[edge_idx]
        
        # Random 2-qubit unitary (small rotation)
        U2 = unitary_from_hermitian(hermitian_rand(4, rng), t=params.eps)
        U_full = embed_two_qubit_gate_general(U2, N, i, j)
        H_new = U_full @ H_curr @ U_full.conj().T
        
        # Evaluate cost
        if step % params.cost_every == 0:
            evaluated += 1
            metrics_new = locality_metric(H_new, basis, N)
            cost_new = metrics_new["leak_frac"]
        else:
            cost_new = cost
        
        # Metropolis acceptance
        accept = False
        if cost_new <= cost:
            accept = True
        elif temp > 0:
            p = math.exp(-(cost_new - cost) / max(1e-12, temp))
            if rng.random() < p:
                accept = True
        
        if accept:
            accepted += 1
            H_curr = H_new
            cost = cost_new
            if cost < best_cost:
                best_cost = cost
                H_best = H_curr.copy()
        
        temp *= params.temp_decay
        
        if step % 100 == 0:
            cost_history.append(cost)
    
    # Final metrics
    final_metrics = locality_metric(H_curr, basis, N)
    best_metrics = locality_metric(H_best, basis, N)
    
    diagnostics = {
        "accepted": accepted,
        "evaluated": evaluated,
        "final_leak": final_metrics["leak_frac"],
        "final_local": final_metrics["local_frac"],
        "best_leak": best_metrics["leak_frac"],
        "best_local": best_metrics["local_frac"],
        "cost_history": cost_history
    }
    
    return H_best, diagnostics

# =============================================================================
# SPECTRAL DIMENSION MEASUREMENT
# =============================================================================

def measure_spectral_dimension(graph: GraphTopology, t_values: Optional[List[float]] = None) -> Dict:
    """
    Measure spectral dimension via heat kernel on the graph.
    
    d_s = -2 * d(log P(t)) / d(log t)
    
    where P(t) = (1/N) * Tr(exp(-t * L)) is the average return probability
    and L is the graph Laplacian.
    """
    N = graph.N
    adj = graph.adjacency_dict()
    
    # Build graph Laplacian
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = len(adj[i])  # Degree
        for j in adj[i]:
            L[i, j] = -1
    
    # Eigendecompose
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Heat kernel trace: P(t) = (1/N) * sum_k exp(-t * lambda_k)
    if t_values is None:
        t_values = np.logspace(-2, 2, 50)
    
    P_t = []
    for t in t_values:
        P = np.mean(np.exp(-t * eigvals))
        P_t.append(P)
    
    P_t = np.array(P_t)
    
    # Estimate spectral dimension from slope in log-log
    log_t = np.log(t_values)
    log_P = np.log(P_t + 1e-30)
    
    # Use middle region for fit (avoid boundary effects)
    mid_start = len(t_values) // 4
    mid_end = 3 * len(t_values) // 4
    
    # Linear fit: log P ≈ -(d_s/2) * log t + const
    coeffs = np.polyfit(log_t[mid_start:mid_end], log_P[mid_start:mid_end], 1)
    d_spectral = -2 * coeffs[0]
    
    return {
        "spectral_dimension": float(d_spectral),
        "laplacian_eigenvalues": eigvals.tolist(),
        "fit_slope": float(coeffs[0]),
        "fit_intercept": float(coeffs[1])
    }

# =============================================================================
# EFFECTIVE DIMENSION FROM RECOVERED HAMILTONIAN
# =============================================================================

def measure_effective_dimension(H: np.ndarray, graph: GraphTopology, basis: LocalBasis) -> Dict:
    """
    Estimate the effective dimension of interactions in the recovered Hamiltonian.
    
    Method: Decompose H in Pauli basis, look at which terms have significant weight,
    and analyze the "effective interaction graph" implied by those terms.
    """
    N = graph.N
    d = 2 ** N
    
    # Project onto local basis and get coefficients
    coefficients = {}
    total_weight = 0.0
    
    for op, tag in zip(basis.ops, basis.tags):
        coef = np.trace(op.conj().T @ H) / d
        weight = float((coef.conjugate() * coef).real)
        if weight > 1e-10:
            coefficients[tag] = {"coef_real": float(coef.real), 
                                 "coef_imag": float(coef.imag),
                                 "weight": weight}
            total_weight += weight
    
    # Normalize
    for tag in coefficients:
        coefficients[tag]["weight_frac"] = coefficients[tag]["weight"] / (total_weight + 1e-18)
    
    # Count significant terms by type
    single_site_weight = 0.0
    two_site_weight = 0.0
    
    for tag, data in coefficients.items():
        if "-" in tag:  # Two-site term
            two_site_weight += data["weight"]
        else:
            single_site_weight += data["weight"]
    
    return {
        "num_significant_terms": len(coefficients),
        "single_site_weight": single_site_weight / (total_weight + 1e-18),
        "two_site_weight": two_site_weight / (total_weight + 1e-18),
        "top_terms": sorted(coefficients.items(), key=lambda x: -x[1]["weight"])[:10]
    }

# =============================================================================
# GRAPH TOPOLOGY GENERATORS FOR FIXED N
# =============================================================================

def generate_topologies_for_N(N: int, rng: np.random.Generator) -> List[GraphTopology]:
    """
    Generate a suite of graph topologies for a given number of qubits N.
    Returns topologies that are valid for the given N.
    """
    topologies = []
    
    # 1D always works
    topologies.append(make_1d_ring(N))
    topologies.append(make_1d_chain(N))
    
    # 2D: need N = Lx * Ly
    for Lx in range(2, N + 1):
        if N % Lx == 0:
            Ly = N // Lx
            if Ly >= 2 and Lx >= 2:
                topologies.append(make_2d_lattice(Lx, Ly, periodic=True))
                break  # Just take one 2D factorization
    
    # 3D: need N = Lx * Ly * Lz
    found_3d = False
    for Lx in range(2, N + 1):
        if found_3d:
            break
        if N % Lx != 0:
            continue
        remainder = N // Lx
        for Ly in range(2, remainder + 1):
            if remainder % Ly == 0:
                Lz = remainder // Ly
                if Lz >= 2:
                    topologies.append(make_3d_lattice(Lx, Ly, Lz, periodic=True))
                    found_3d = True
                    break
    
    # 4D: need N = L^4
    L4 = round(N ** 0.25)
    if L4 ** 4 == N and L4 >= 2:
        topologies.append(make_4d_lattice(L4, periodic=True))
    
    # Random regular graph (degree 4 to match 2D lattice)
    if N >= 4 and (N * 4) % 2 == 0:
        try:
            topologies.append(make_random_regular(N, degree=4, rng=rng))
        except RuntimeError:
            pass
    
    # Random regular with degree 6 (to match 3D lattice)
    if N >= 6 and (N * 6) % 2 == 0:
        try:
            topologies.append(make_random_regular(N, degree=6, rng=rng))
        except RuntimeError:
            pass
    
    return topologies

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(graph: GraphTopology, seed: int, scramble_type: str,
                   hamiltonian_type: str, flow_params: FlowParams,
                   progress: bool = False) -> Dict:
    """Run a single experiment on a given graph topology."""
    
    rng = np.random.default_rng(seed)
    N = graph.N
    
    # Build Hamiltonian
    if hamiltonian_type == "heisenberg":
        H0 = build_heisenberg_hamiltonian(graph)
    else:
        H0 = build_xx_hamiltonian(graph)
    
    # Build local basis
    basis = build_local_basis(graph)
    
    # Measure baseline
    baseline_metrics = locality_metric(H0, basis, N)
    
    # Scramble
    if scramble_type == "local":
        Hs = scramble_local(H0, N, rng)
    elif scramble_type == "global":
        Hs = scramble_global(H0, N, rng)
    elif scramble_type == "layer":
        Hs = scramble_layer(H0, graph, rng, depth=N)
    else:
        raise ValueError(f"Unknown scramble type: {scramble_type}")
    
    scrambled_metrics = locality_metric(Hs, basis, N)
    
    # Run recovery flow
    Hr, flow_diag = flow_recover(Hs, graph, basis, rng, flow_params, use_graph_gates=True)
    
    recovered_metrics = locality_metric(Hr, basis, N)
    
    # Measure spectral dimension of the graph
    spectral_info = measure_spectral_dimension(graph)
    
    # Measure effective dimension of recovered H
    effective_info = measure_effective_dimension(Hr, graph, basis)
    
    # Compile results
    result = {
        "meta": {
            "created_utc": now_utc_iso(),
            "seed": seed,
            "N": N,
            "graph_name": graph.name,
            "graph_dimension": graph.dimension,
            "graph_edges": len(graph.edges),
            "graph_degree": graph.degree,
            "graph_metadata": graph.metadata,
            "scramble_type": scramble_type,
            "hamiltonian_type": hamiltonian_type,
            "flow_steps": flow_params.steps,
        },
        "baseline": {
            "local_frac": baseline_metrics["local_frac"],
            "leak_frac": baseline_metrics["leak_frac"],
        },
        "scrambled": {
            "local_frac": scrambled_metrics["local_frac"],
            "leak_frac": scrambled_metrics["leak_frac"],
        },
        "recovered": {
            "local_frac": recovered_metrics["local_frac"],
            "leak_frac": recovered_metrics["leak_frac"],
            "best_local": flow_diag["best_local"],
            "best_leak": flow_diag["best_leak"],
        },
        "flow": {
            "accepted": flow_diag["accepted"],
            "evaluated": flow_diag["evaluated"],
            "acceptance_rate": flow_diag["accepted"] / max(1, flow_params.steps),
        },
        "recovery": {
            "leak_reduction": scrambled_metrics["leak_frac"] - recovered_metrics["leak_frac"],
            "local_improvement": recovered_metrics["local_frac"] - scrambled_metrics["local_frac"],
            "recovered": bool(recovered_metrics["local_frac"] > scrambled_metrics["local_frac"]),
        },
        "spectral": spectral_info,
        "effective": {
            "num_terms": effective_info["num_significant_terms"],
            "single_site_frac": effective_info["single_site_weight"],
            "two_site_frac": effective_info["two_site_weight"],
        }
    }
    
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="HSF Dimensional Emergence Study")
    
    # Output
    ap.add_argument("--out", required=True, help="Output directory")
    
    # System parameters
    ap.add_argument("--N", type=int, default=8, help="Number of qubits")
    ap.add_argument("--hamiltonian", choices=["xx", "heisenberg"], default="xx")
    
    # Experiment parameters  
    ap.add_argument("--seeds", type=int, default=5, help="Number of random seeds per configuration")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--scrambles", default="global", help="Comma-separated: local,global,layer")
    
    # Flow parameters
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--temp0", type=float, default=0.02)
    ap.add_argument("--temp-decay", type=float, default=0.9995)
    ap.add_argument("--cost-every", type=int, default=1)
    
    # Topology selection
    ap.add_argument("--topologies", default="all", 
                    help="Comma-separated topology names, or 'all' for auto-generation")
    
    # Options
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--zip", action="store_true")
    
    args = ap.parse_args()
    
    # Setup output
    outdir = Path(args.out).resolve()
    ensure_dir(outdir)
    ensure_dir(outdir / "runs")
    
    # Write manifest
    write_text(outdir / "manifest.json", json.dumps({
        "created_utc": now_utc_iso(),
        "tool": "hsf_dimensional_emergence.py",
        "args": vars(args),
    }, indent=2))
    
    # Parse scramble types
    scrambles = [s.strip().lower() for s in args.scrambles.split(",") if s.strip()]
    
    # Generate topologies
    topo_rng = np.random.default_rng(999)  # Fixed seed for reproducible topology generation
    
    if args.topologies == "all":
        topologies = generate_topologies_for_N(args.N, topo_rng)
    else:
        # Manual topology specification
        topo_names = [t.strip() for t in args.topologies.split(",")]
        topologies = []
        for name in topo_names:
            if name == "1d_ring":
                topologies.append(make_1d_ring(args.N))
            elif name == "1d_chain":
                topologies.append(make_1d_chain(args.N))
            # Add more as needed
    
    # Print topology summary
    print("=" * 60)
    print("HSF Dimensional Emergence Study")
    print("=" * 60)
    print(f"N = {args.N} qubits (Hilbert space dim = {2**args.N})")
    print(f"Seeds: {args.seeds}")
    print(f"Scrambles: {scrambles}")
    print(f"\nTopologies to test:")
    for topo in topologies:
        dim_str = f"d={topo.dimension}" if topo.dimension else "non-geometric"
        print(f"  - {topo.name}: {topo.num_edges} edges, {dim_str}")
    print("=" * 60)
    
    # Flow parameters
    flow_params = FlowParams(
        steps=args.steps,
        eps=args.eps,
        temp0=args.temp0,
        temp_decay=args.temp_decay,
        cost_every=args.cost_every
    )
    
    # Run experiments
    runs_path = outdir / "runs" / "runs.jsonl"
    if runs_path.exists():
        runs_path.unlink()
    
    all_results = []
    total_runs = len(topologies) * len(scrambles) * args.seeds
    run_idx = 0
    t0 = time.time()
    
    for topo in topologies:
        for scramble in scrambles:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                run_idx += 1
                
                if args.progress:
                    print(f"[{run_idx}/{total_runs}] {topo.name} | {scramble} | seed={seed}")
                
                try:
                    result = run_experiment(
                        graph=topo,
                        seed=seed,
                        scramble_type=scramble,
                        hamiltonian_type=args.hamiltonian,
                        flow_params=flow_params,
                        progress=args.progress
                    )
                    
                    all_results.append(result)
                    append_jsonl(runs_path, result)
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue
    
    runtime = time.time() - t0
    
    # Write summary
    summary = {
        "created_utc": now_utc_iso(),
        "total_runs": len(all_results),
        "runtime_sec": runtime,
        "by_topology": {},
        "by_dimension": {}
    }
    
    # Aggregate by topology
    for topo in topologies:
        topo_results = [r for r in all_results if r["meta"]["graph_name"] == topo.name]
        if topo_results:
            leak_reductions = [r["recovery"]["leak_reduction"] for r in topo_results]
            local_improvements = [r["recovery"]["local_improvement"] for r in topo_results]
            recovered_count = sum(1 for r in topo_results if r["recovery"]["recovered"])
            
            summary["by_topology"][topo.name] = {
                "dimension": topo.dimension,
                "runs": len(topo_results),
                "recovery_rate": recovered_count / len(topo_results),
                "mean_leak_reduction": float(np.mean(leak_reductions)),
                "mean_local_improvement": float(np.mean(local_improvements)),
                "spectral_dimension": topo_results[0]["spectral"]["spectral_dimension"]
            }
    
    # Aggregate by intrinsic dimension
    for dim in [1, 2, 3, 4, None]:
        dim_results = [r for r in all_results if r["meta"]["graph_dimension"] == dim]
        if dim_results:
            leak_reductions = [r["recovery"]["leak_reduction"] for r in dim_results]
            dim_key = str(dim) if dim else "non-geometric"
            summary["by_dimension"][dim_key] = {
                "runs": len(dim_results),
                "mean_leak_reduction": float(np.mean(leak_reductions)),
                "std_leak_reduction": float(np.std(leak_reductions))
            }
    
    write_text(outdir / "summary.json", json.dumps(summary, indent=2))
    
    # Write report
    report_lines = [
        "# HSF Dimensional Emergence Study Report",
        f"- Created: `{now_utc_iso()}`",
        f"- N = {args.N} qubits",
        f"- Total runs: {len(all_results)}",
        f"- Runtime: {runtime:.1f}s",
        "",
        "## Results by Topology",
        ""
    ]
    
    for topo_name, data in summary["by_topology"].items():
        report_lines.append(f"### {topo_name} (d={data['dimension']})")
        report_lines.append(f"- Recovery rate: {data['recovery_rate']:.1%}")
        report_lines.append(f"- Mean leak reduction: {data['mean_leak_reduction']:.4f}")
        report_lines.append(f"- Spectral dimension: {data['spectral_dimension']:.2f}")
        report_lines.append("")
    
    report_lines.extend([
        "## Results by Intrinsic Dimension",
        ""
    ])
    
    for dim_key, data in summary["by_dimension"].items():
        report_lines.append(f"### Dimension {dim_key}")
        report_lines.append(f"- Runs: {data['runs']}")
        report_lines.append(f"- Mean leak reduction: {data['mean_leak_reduction']:.4f} ± {data['std_leak_reduction']:.4f}")
        report_lines.append("")
    
    write_text(outdir / "REPORT.md", "\n".join(report_lines))
    
    if args.zip:
        z = zip_folder(outdir)
        print(f"Wrote ZIP: {z}")
    
    print(f"\nDONE. Results in: {outdir}")
    print(f"Open: {outdir / 'REPORT.md'}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())