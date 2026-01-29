# constraint_emergence_test_v3_scalable.py
# ------------------------------------------------------------
# HSF Constraint Test V3 — SCALABLE VERSION for N=20-30 qubits
#
# Key optimizations:
#   1. Sparse Hamiltonian: O(N·2^N) memory instead of O(4^N)
#   2. Krylov evolution: exp(-iHt)|ψ⟩ via Lanczos, no eigendecomp
#   3. Efficient partial trace: reshape/transpose, no explicit sum
#   4. Optional Trotter decomposition for very large N
#
# The four constraints tested:
#   (1) No-signaling: finite-speed propagation
#   (2) No-forgetting: persistent recoverability
#   (3) No-refolding: maintain structural complexity
#   (4) Finite bandwidth: limited information capacity per block
#
# Usage:
#   python constraint_emergence_test_v3_scalable.py --N 20 --bandwidth 1.5 --progress
#   python constraint_emergence_test_v3_scalable.py --N 24 --bandwidth-sweep "1.0,1.5,2.0" --progress
#
# Memory estimates:
#   N=20: ~16 MB per state vector, ~500 MB sparse H
#   N=24: ~256 MB per state vector, ~8 GB sparse H  
#   N=28: ~4 GB per state vector (pushing limits)
#
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import os
import time
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, kron as sparse_kron, eye as sparse_eye
from scipy.sparse.linalg import expm_multiply, LinearOperator

# ---------------------------
# Sparse Pauli matrices
# ---------------------------

def sparse_pauli(p: str) -> csr_matrix:
    """Return sparse 2x2 Pauli matrix."""
    if p == "I":
        return sparse_eye(2, dtype=np.complex128, format='csr')
    elif p == "X":
        return csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    elif p == "Y":
        return csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
    elif p == "Z":
        return csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    else:
        raise ValueError(f"Unknown Pauli: {p}")

# Cache for efficiency
_PAULI_CACHE = {p: sparse_pauli(p) for p in ["I", "X", "Y", "Z"]}


def sparse_kron_n(ops: List[csr_matrix]) -> csr_matrix:
    """Kronecker product of list of sparse matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = sparse_kron(result, op, format='csr')
    return result


def sparse_pauli_string(N: int, paulis: Dict[int, str]) -> csr_matrix:
    """
    Build sparse operator for a Pauli string on N qubits.
    paulis: dict mapping qubit index -> Pauli label ('X', 'Y', 'Z')
    Qubits not in dict get identity.
    """
    ops = []
    for i in range(N):
        p = paulis.get(i, "I")
        ops.append(_PAULI_CACHE[p])
    return sparse_kron_n(ops)


# ---------------------------
# Graph constructors
# ---------------------------

def build_edges(N: int, topology: str, rng: np.random.Generator, rr_deg: int = 3) -> List[Tuple[int, int]]:
    """Build edge list for given topology."""
    topology = topology.lower()
    edges: set[Tuple[int, int]] = set()

    if topology == "ring":
        for i in range(N):
            j = (i + 1) % N
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    elif topology == "line":
        for i in range(N - 1):
            edges.add((i, i + 1))
    elif topology == "complete":
        for i in range(N):
            for j in range(i + 1, N):
                edges.add((i, j))
    elif topology.startswith("rr"):
        # Random regular graph
        deg = rr_deg
        if topology != "rr":
            try:
                deg = int(topology[2:])
            except:
                pass
        if deg >= N or (N * deg) % 2 != 0:
            raise ValueError(f"Invalid random regular: N={N}, deg={deg}")
        
        for _attempt in range(5000):
            stubs = []
            for i in range(N):
                stubs.extend([i] * deg)
            rng.shuffle(stubs)
            edges.clear()
            ok = True
            for k in range(0, len(stubs), 2):
                a, b = stubs[k], stubs[k + 1]
                if a == b:
                    ok = False
                    break
                e = (min(a, b), max(a, b))
                if e in edges:
                    ok = False
                    break
                edges.add(e)
            if ok:
                break
        else:
            raise RuntimeError("Failed to build random regular graph")
    elif topology == "ladder":
        # 2×(N/2) ladder - good for testing 2D-like locality
        if N % 2 != 0:
            raise ValueError("Ladder requires even N")
        L = N // 2
        for i in range(L - 1):
            edges.add((i, i + 1))  # top rail
            edges.add((L + i, L + i + 1))  # bottom rail
        for i in range(L):
            edges.add((i, L + i))  # rungs
    elif topology == "grid2d":
        # Approximate 2D grid (sqrt(N) × sqrt(N))
        side = int(np.sqrt(N))
        if side * side != N:
            raise ValueError(f"grid2d requires perfect square N, got {N}")
        for r in range(side):
            for c in range(side):
                i = r * side + c
                if c < side - 1:
                    edges.add((i, i + 1))
                if r < side - 1:
                    edges.add((i, i + side))
    else:
        raise ValueError(f"Unknown topology: {topology}")

    return sorted(edges)


def neighbors_from_edges(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    nbrs = [[] for _ in range(n)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def graph_distance_matrix(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """BFS-based shortest path distances."""
    nbrs = neighbors_from_edges(n, edges)
    dist = np.full((n, n), fill_value=10**9, dtype=np.int32)
    for s in range(n):
        dist[s, s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in nbrs[u]:
                if dist[s, v] > dist[s, u] + 1:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
    return dist


# ---------------------------
# Sparse Hamiltonian builder
# ---------------------------

def build_sparse_2local_hamiltonian(
    N: int,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
    onsite_scale: float = 0.25,
    edge_scale: float = 0.8,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> csr_matrix:
    """
    Build sparse 2-local Hamiltonian.
    
    H = Σ_i (hx_i X_i + hy_i Y_i + hz_i Z_i) + Σ_{ij} Σ_{PQ} J_{ij}^{PQ} P_i Q_j
    
    Memory: O(N·2^N + E·2^N) ≈ O((N+9E)·2^N) non-zeros
    """
    dim = 2 ** N
    
    # Accumulate terms
    H = csr_matrix((dim, dim), dtype=np.complex128)
    
    # Onsite terms: N × 3 = 3N terms
    if progress_callback:
        progress_callback(f"Building onsite terms (3×{N})...")
    
    for i in range(N):
        hx, hy, hz = rng.normal(scale=onsite_scale, size=3)
        if abs(hx) > 1e-12:
            H = H + hx * sparse_pauli_string(N, {i: "X"})
        if abs(hy) > 1e-12:
            H = H + hy * sparse_pauli_string(N, {i: "Y"})
        if abs(hz) > 1e-12:
            H = H + hz * sparse_pauli_string(N, {i: "Z"})
    
    # Edge terms: |E| × 9 terms
    if progress_callback:
        progress_callback(f"Building edge terms (9×{len(edges)})...")
    
    paulis = ["X", "Y", "Z"]
    for idx, (i, j) in enumerate(edges):
        if progress_callback and idx % max(1, len(edges) // 10) == 0:
            progress_callback(f"  Edge {idx+1}/{len(edges)}...")
        for Pi in paulis:
            for Pj in paulis:
                J = rng.normal(scale=edge_scale)
                if abs(J) > 1e-12:
                    H = H + J * sparse_pauli_string(N, {i: Pi, j: Pj})
    
    # Hermitian symmetrize
    H = 0.5 * (H + H.conj().T)
    H.eliminate_zeros()
    
    if progress_callback:
        nnz = H.nnz
        density = nnz / (dim * dim)
        progress_callback(f"H built: {nnz:,} non-zeros ({density:.2e} density)")
    
    return H


# ---------------------------
# Krylov time evolution (for small-medium N)
# ---------------------------

@dataclass
class SparseEvolver:
    """Time evolution using Krylov subspace methods."""
    N: int
    H: csr_matrix
    
    def evolve_state(self, psi0: np.ndarray, t: float) -> np.ndarray:
        """
        Compute exp(-i H t) |psi0⟩ using scipy's expm_multiply.
        """
        if abs(t) < 1e-14:
            return psi0.copy()
        
        result = expm_multiply(-1j * self.H, psi0, start=0, stop=t, num=2, endpoint=True)
        return result[-1]
    
    def evolve_state_times(self, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Evolve state to multiple times efficiently.
        Returns array of shape (len(times), dim).
        """
        if len(times) == 0:
            return np.zeros((0, len(psi0)), dtype=np.complex128)
        
        t_max = float(times[-1])
        if t_max < 1e-14:
            return np.tile(psi0, (len(times), 1))
        
        result = expm_multiply(-1j * self.H, psi0, start=0, stop=t_max, 
                               num=len(times), endpoint=True)
        return result


# ---------------------------
# Trotter evolution (for large N - memory efficient)
# ---------------------------

class TrotterEvolver:
    """
    Time evolution using Trotter decomposition.
    
    Memory efficient for large N because we never form the full
    matrix exponential - just apply 2-qubit gates sequentially.
    
    H = Σ_i h_i + Σ_{ij} H_{ij}
    exp(-iHt) ≈ Π_i exp(-i h_i dt) Π_{ij} exp(-i H_{ij} dt) + O(dt²)
    """
    
    def __init__(self, N: int, edges: List[Tuple[int, int]], 
                 onsite_coeffs: Dict[int, Tuple[float, float, float]],
                 edge_coeffs: Dict[Tuple[int,int], np.ndarray],
                 dt: float = 0.05):
        """
        Initialize Trotter evolver.
        
        onsite_coeffs: {qubit: (hx, hy, hz)}
        edge_coeffs: {(i,j): 3x3 array of J[Pi,Pj]}
        """
        self.N = N
        self.edges = edges
        self.onsite_coeffs = onsite_coeffs
        self.edge_coeffs = edge_coeffs
        self.dt = dt
        
        # Precompute 2-qubit gate exponentials for each edge
        self._precompute_gates()
    
    def _precompute_gates(self):
        """Precompute the 4x4 unitary for each edge interaction."""
        paulis_2x2 = {
            'I': np.eye(2, dtype=np.complex128),
            'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        pauli_labels = ['X', 'Y', 'Z']
        
        # Onsite gates: exp(-i (hx X + hy Y + hz Z) dt)
        self.onsite_gates = {}
        for q, (hx, hy, hz) in self.onsite_coeffs.items():
            H_local = hx * paulis_2x2['X'] + hy * paulis_2x2['Y'] + hz * paulis_2x2['Z']
            self.onsite_gates[q] = np.linalg.matrix_power(
                np.eye(2) - 1j * H_local * self.dt / 20, 20
            ).astype(np.complex128)
            # Better: use exact formula for single-qubit rotation
            norm = np.sqrt(hx**2 + hy**2 + hz**2)
            if norm > 1e-12:
                n = np.array([hx, hy, hz]) / norm
                angle = norm * self.dt
                # exp(-i θ n·σ) = cos(θ)I - i sin(θ) n·σ
                self.onsite_gates[q] = (
                    np.cos(angle) * paulis_2x2['I'] 
                    - 1j * np.sin(angle) * (n[0]*paulis_2x2['X'] + n[1]*paulis_2x2['Y'] + n[2]*paulis_2x2['Z'])
                )
        
        # Edge gates: exp(-i Σ_{PQ} J_{PQ} P⊗Q dt)
        self.edge_gates = {}
        for (i, j), J_matrix in self.edge_coeffs.items():
            # Build 4x4 Hamiltonian for this edge
            H_edge = np.zeros((4, 4), dtype=np.complex128)
            for pi, Pi in enumerate(pauli_labels):
                for pj, Pj in enumerate(pauli_labels):
                    H_edge += J_matrix[pi, pj] * np.kron(paulis_2x2[Pi], paulis_2x2[Pj])
            
            # Compute matrix exponential
            evals, evecs = np.linalg.eigh(H_edge)
            self.edge_gates[(i, j)] = evecs @ np.diag(np.exp(-1j * evals * self.dt)) @ evecs.conj().T
    
    def _apply_single_qubit_gate(self, psi: np.ndarray, qubit: int, gate: np.ndarray) -> np.ndarray:
        """Apply 2x2 gate to specified qubit."""
        psi_tensor = psi.reshape([2] * self.N)
        
        # Move target qubit to last axis
        psi_tensor = np.moveaxis(psi_tensor, qubit, -1)
        shape = psi_tensor.shape
        
        # Reshape to (everything_else, 2)
        psi_flat = psi_tensor.reshape(-1, 2)
        
        # Apply gate
        psi_flat = psi_flat @ gate.T
        
        # Reshape back
        psi_tensor = psi_flat.reshape(shape)
        psi_tensor = np.moveaxis(psi_tensor, -1, qubit)
        
        return psi_tensor.reshape(-1)
    
    def _apply_two_qubit_gate(self, psi: np.ndarray, q1: int, q2: int, gate: np.ndarray) -> np.ndarray:
        """Apply 4x4 gate to specified qubit pair."""
        psi_tensor = psi.reshape([2] * self.N)
        
        # Move target qubits to last two axes
        axes = list(range(self.N))
        axes.remove(q1)
        axes.remove(q2)
        axes.extend([q1, q2])
        
        psi_tensor = np.transpose(psi_tensor, axes)
        shape = psi_tensor.shape
        
        # Reshape to (everything_else, 4)
        psi_flat = psi_tensor.reshape(-1, 4)
        
        # Apply gate
        psi_flat = psi_flat @ gate.T
        
        # Reshape and transpose back
        psi_tensor = psi_flat.reshape(shape)
        
        # Inverse permutation
        inv_axes = [0] * self.N
        for new_pos, old_pos in enumerate(axes):
            inv_axes[old_pos] = new_pos
        
        psi_tensor = np.transpose(psi_tensor, inv_axes)
        
        return psi_tensor.reshape(-1)
    
    def trotter_step(self, psi: np.ndarray) -> np.ndarray:
        """Apply one Trotter step."""
        # Apply onsite terms
        for q, gate in self.onsite_gates.items():
            psi = self._apply_single_qubit_gate(psi, q, gate)
        
        # Apply edge terms
        for (i, j), gate in self.edge_gates.items():
            psi = self._apply_two_qubit_gate(psi, i, j, gate)
        
        return psi
    
    def evolve_state(self, psi0: np.ndarray, t: float) -> np.ndarray:
        """Evolve state to time t using Trotter decomposition."""
        if abs(t) < 1e-14:
            return psi0.copy()
        
        n_steps = max(1, int(np.ceil(t / self.dt)))
        actual_dt = t / n_steps
        
        # Rescale gates for actual_dt if different from self.dt
        # (For simplicity, we just use more steps with original dt)
        
        psi = psi0.copy()
        for _ in range(n_steps):
            psi = self.trotter_step(psi)
        
        # Renormalize to prevent drift
        psi = psi / np.linalg.norm(psi)
        return psi
    
    def evolve_state_times(self, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Evolve to multiple times, returning all intermediate states."""
        results = [psi0.copy()]
        psi = psi0.copy()
        
        for i in range(1, len(times)):
            dt_needed = times[i] - times[i-1]
            n_steps = max(1, int(np.ceil(dt_needed / self.dt)))
            
            for _ in range(n_steps):
                psi = self.trotter_step(psi)
            
            psi = psi / np.linalg.norm(psi)
            results.append(psi.copy())
        
        return np.array(results)


def build_trotter_evolver(
    N: int,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
    onsite_scale: float = 0.25,
    edge_scale: float = 0.8,
    dt: float = 0.05,
) -> TrotterEvolver:
    """Build a Trotter evolver with random coefficients."""
    
    # Generate onsite coefficients
    onsite_coeffs = {}
    for i in range(N):
        hx, hy, hz = rng.normal(scale=onsite_scale, size=3)
        onsite_coeffs[i] = (hx, hy, hz)
    
    # Generate edge coefficients
    edge_coeffs = {}
    for (i, j) in edges:
        J_matrix = rng.normal(scale=edge_scale, size=(3, 3))
        edge_coeffs[(i, j)] = J_matrix
    
    return TrotterEvolver(N, edges, onsite_coeffs, edge_coeffs, dt=dt)


# ---------------------------
# Efficient partial trace
# ---------------------------

def partial_trace_pure_state(psi: np.ndarray, keep: List[int], N: int) -> np.ndarray:
    """
    Compute reduced density matrix by tracing out qubits not in 'keep'.
    
    For pure state |ψ⟩, ρ_A = Tr_B(|ψ⟩⟨ψ|)
    
    Efficient implementation using reshape and einsum.
    """
    keep = sorted(keep)
    n_keep = len(keep)
    n_trace = N - n_keep
    
    if n_keep == 0:
        # Trace out everything -> scalar (should not happen)
        return np.array([[np.abs(np.vdot(psi, psi))]], dtype=np.complex128)
    
    if n_keep == N:
        # Keep everything - DON'T compute full density matrix (too large)
        # For trace distance purposes, we only need to know that two pure states
        # have trace distance = sqrt(1 - |<ψ|φ>|²)
        # Return a sentinel that signals "full system" case
        # We'll handle this in trace_distance_from_rhos
        return None  # Sentinel for full system
    
    # Reshape psi into tensor
    psi_tensor = psi.reshape([2] * N)
    
    # Permute to put 'keep' indices first
    trace_out = [i for i in range(N) if i not in keep]
    perm = keep + trace_out
    psi_perm = np.transpose(psi_tensor, perm)
    
    # Reshape: (keep_dims, trace_dims)
    dim_keep = 2 ** n_keep
    dim_trace = 2 ** n_trace
    psi_mat = psi_perm.reshape(dim_keep, dim_trace)
    
    # ρ = |ψ⟩⟨ψ| traced = psi_mat @ psi_mat†
    rho = psi_mat @ psi_mat.conj().T
    return rho


def trace_distance_from_rhos(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance D(ρ,σ) = (1/2)||ρ-σ||_1 = (1/2)Σ|λ_i|."""
    # Handle sentinel case (full system, can't compute density matrices)
    if rho is None or sigma is None:
        return 0.0  # No meaningful trace distance for trivial blocking
    diff = rho - sigma
    eigs = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigs)))


# ---------------------------
# Initial states
# ---------------------------

def random_product_state(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random product state |ψ⟩ = ⊗_i |θ_i, φ_i⟩."""
    psi = np.array([1.0], dtype=np.complex128)
    for _ in range(N):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        qubit = np.array([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2)
        ], dtype=np.complex128)
        psi = np.kron(psi, qubit)
    return psi


def apply_local_pauli(psi: np.ndarray, qubit: int, pauli: str, N: int) -> np.ndarray:
    """Apply single-qubit Pauli to state vector efficiently."""
    psi_tensor = psi.reshape([2] * N)
    
    if pauli == "X":
        # Swap |0⟩ ↔ |1⟩ on qubit
        psi_tensor = np.swapaxes(psi_tensor, qubit, -1)
        result = psi_tensor.copy()
        result[..., 0], result[..., 1] = psi_tensor[..., 1].copy(), psi_tensor[..., 0].copy()
        result = np.swapaxes(result, qubit, -1)
    elif pauli == "Y":
        psi_tensor = np.swapaxes(psi_tensor, qubit, -1)
        result = np.zeros_like(psi_tensor)
        result[..., 0] = -1j * psi_tensor[..., 1]
        result[..., 1] = 1j * psi_tensor[..., 0]
        result = np.swapaxes(result, qubit, -1)
    elif pauli == "Z":
        psi_tensor = np.swapaxes(psi_tensor, qubit, -1)
        result = psi_tensor.copy()
        result[..., 1] *= -1
        result = np.swapaxes(result, qubit, -1)
    elif pauli == "I":
        return psi.copy()
    else:
        raise ValueError(f"Unknown Pauli: {pauli}")
    
    return result.reshape(-1)


# ---------------------------
# Bandwidth utilities
# ---------------------------

def bandwidth_efficiency(block_size: int, bandwidth_capacity: float) -> float:
    """Fraction of information a block can effectively process."""
    if block_size <= 0 or bandwidth_capacity <= 0:
        return 0.0
    return min(float(block_size), bandwidth_capacity) / float(block_size)


def bandwidth_penalty(block_size: int, bandwidth_capacity: float, scale: float = 1.0) -> float:
    """Penalty for exceeding bandwidth capacity."""
    if block_size <= bandwidth_capacity:
        return 0.0
    return scale * (block_size - bandwidth_capacity) / bandwidth_capacity


# ---------------------------
# Blocking utilities
# ---------------------------

def make_contiguous_blocks(N: int, block_size: int) -> List[List[int]]:
    if N % block_size != 0:
        raise ValueError(f"block_size {block_size} does not divide N={N}")
    nB = N // block_size
    return [list(range(b * block_size, (b + 1) * block_size)) for b in range(nB)]


def block_graph_edges(qubit_edges: List[Tuple[int, int]], blocks: List[List[int]]) -> List[Tuple[int, int]]:
    qubit_to_block = {}
    for b_idx, blk in enumerate(blocks):
        for q in blk:
            qubit_to_block[q] = b_idx
    
    block_edges_set: set[Tuple[int, int]] = set()
    for (i, j) in qubit_edges:
        bi = qubit_to_block.get(i)
        bj = qubit_to_block.get(j)
        if bi is not None and bj is not None and bi != bj:
            block_edges_set.add((min(bi, bj), max(bi, bj)))
    return sorted(block_edges_set)


# ---------------------------
# Main evaluation function
# ---------------------------

def evaluate_blocking_v3_scalable(
    N: int,
    qubit_edges: List[Tuple[int, int]],
    evolver: SparseEvolver,
    psi_base: np.ndarray,
    source_qubit: int,
    times: np.ndarray,
    block_size: int,
    speed_threshold: float = 0.02,
    recover_threshold: float = 0.03,
    d_remote_min: int = 2,
    t_late_frac: float = 0.4,
    min_blocks: int = 4,
    bandwidth_capacity: float = 1.5,
    bandwidth_penalty_scale: float = 1.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Evaluate a blocking under all four HSF constraints.
    Scalable version using sparse evolution.
    """
    blocks = make_contiguous_blocks(N, block_size)
    nB = len(blocks)
    
    bedges = block_graph_edges(qubit_edges, blocks)
    bdist = graph_distance_matrix(nB, bedges) if nB > 1 else np.zeros((1, 1), dtype=np.int32)
    
    source_block = source_qubit // block_size
    dist_from_source = [int(bdist[source_block, b]) for b in range(nB)]
    
    bw_eff = bandwidth_efficiency(block_size, bandwidth_capacity)
    bw_pen = bandwidth_penalty(block_size, bandwidth_capacity, bandwidth_penalty_scale)
    
    # Apply perturbation
    rng_pert = np.random.default_rng(12345)
    pauli_choice = ["X", "Y", "Z"][rng_pert.integers(0, 3)]
    psi_pert = apply_local_pauli(psi_base, source_qubit, pauli_choice, N)
    
    # Evolve both states to all times
    if progress_callback:
        progress_callback(f"    Evolving base state to {len(times)} times...")
    
    psi_base_t = evolver.evolve_state_times(psi_base, times)
    
    if progress_callback:
        progress_callback(f"    Evolving perturbed state...")
    
    psi_pert_t = evolver.evolve_state_times(psi_pert, times)
    
    # Compute block-level influence
    if progress_callback:
        progress_callback(f"    Computing {len(times)}×{nB} partial traces...")
    
    nt = len(times)
    infl = np.zeros((nt, nB), dtype=np.float64)
    
    for ti in range(nt):
        for b in range(nB):
            rho_a = partial_trace_pure_state(psi_base_t[ti], blocks[b], N)
            rho_b = partial_trace_pure_state(psi_pert_t[ti], blocks[b], N)
            infl[ti, b] = trace_distance_from_rhos(rho_a, rho_b)
    
    # Free memory
    del psi_base_t, psi_pert_t
    gc.collect()
    
    # No-signaling: crossing times
    t_cross = [None] * nB
    for b in range(nB):
        for ti, t in enumerate(times):
            if infl[ti, b] >= speed_threshold:
                t_cross[b] = float(t)
                break
    
    v_candidates = []
    instant_reach = 0
    reached = 0
    for b in range(nB):
        d = dist_from_source[b]
        if d <= 0:
            continue
        tc = t_cross[b]
        if tc is None:
            continue
        reached += 1
        if tc <= 1e-12:
            instant_reach += 1
        else:
            v_candidates.append(d / tc)
    
    v_eff = float(max(v_candidates)) if v_candidates else 0.0
    frac_reached = float(reached / max(1, nB - 1))
    
    # No-forgetting: remote recoverability
    remote_blocks = [b for b in range(nB) if b != source_block and dist_from_source[b] >= d_remote_min]
    
    if not remote_blocks:
        remote_best_raw = np.zeros(nt, dtype=np.float64)
    else:
        remote_best_raw = infl[:, remote_blocks].max(axis=1)
    
    remote_best_eff = remote_best_raw * bw_eff
    
    t_late_start = float(times[0] + t_late_frac * (times[-1] - times[0]))
    late_mask = times >= t_late_start
    
    if np.any(late_mask):
        frac_recover_raw = float(np.mean(remote_best_raw[late_mask] >= recover_threshold))
        mean_recover_raw = float(np.mean(remote_best_raw[late_mask]))
        frac_recover_eff = float(np.mean(remote_best_eff[late_mask] >= recover_threshold))
        mean_recover_eff = float(np.mean(remote_best_eff[late_mask]))
    else:
        frac_recover_raw = mean_recover_raw = 0.0
        frac_recover_eff = mean_recover_eff = 0.0
    
    # Penalties
    hard_penalty = 0.0
    if nB < min_blocks:
        hard_penalty += 2.0
    if nB > 1 and len(bedges) == 0:
        hard_penalty += 2.0
    
    soft_penalty = 0.0
    soft_penalty += 0.75 * float(instant_reach)
    soft_penalty += 0.50 * max(0.0, (v_eff - 20.0) / 20.0)
    soft_penalty += bw_pen
    
    # Score
    score = (
        2.0 * frac_recover_eff
        + 0.8 * math.tanh(5.0 * mean_recover_eff)
        + 0.5 * frac_reached
        - 1.0 * hard_penalty
        - 1.0 * soft_penalty
    )
    
    return {
        "block_size": int(block_size),
        "n_blocks": int(nB),
        "block_edges": int(len(bedges)),
        "bandwidth_capacity": float(bandwidth_capacity),
        "bandwidth_efficiency": float(bw_eff),
        "bandwidth_penalty": float(bw_pen),
        "frac_remote_recover_late_RAW": float(frac_recover_raw),
        "remote_recover_mean_late_RAW": float(mean_recover_raw),
        "frac_remote_recover_late_EFF": float(frac_recover_eff),
        "remote_recover_mean_late_EFF": float(mean_recover_eff),
        "frac_blocks_reached": float(frac_reached),
        "v_eff": float(v_eff),
        "instant_reach_blocks": int(instant_reach),
        "hard_penalty": float(hard_penalty),
        "soft_penalty": float(soft_penalty),
        "score": float(score),
    }


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="HSF V3 Scalable (N=20-30)")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--topology", type=str, default="ring",
                    help="ring|line|ladder|grid2d|complete|rr|rr4...")
    ap.add_argument("--rr-deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    
    ap.add_argument("--tmax", type=float, default=6.0)
    ap.add_argument("--nt", type=int, default=61)
    
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--onsite-scale", type=float, default=0.25)
    ap.add_argument("--edge-scale", type=float, default=0.8)
    
    ap.add_argument("--blockings", type=str, default=None,
                    help="Comma-separated block sizes, e.g. '1,2,4,8'. Default: factors of N")
    ap.add_argument("--speed-threshold", type=float, default=0.02)
    ap.add_argument("--recover-threshold", type=float, default=0.03)
    ap.add_argument("--d-remote-min", type=int, default=2)
    ap.add_argument("--t-late-frac", type=float, default=0.4)
    ap.add_argument("--min-blocks", type=int, default=4)
    
    ap.add_argument("--bandwidth", type=float, default=1.5)
    ap.add_argument("--bandwidth-penalty-scale", type=float, default=1.0)
    ap.add_argument("--bandwidth-sweep", type=str, default=None)
    
    # Evolution method
    ap.add_argument("--method", type=str, default="auto",
                    choices=["auto", "krylov", "trotter"],
                    help="Evolution method: auto (trotter for N>=20), krylov, or trotter")
    ap.add_argument("--trotter-dt", type=float, default=0.05,
                    help="Trotter time step (smaller = more accurate but slower)")
    
    ap.add_argument("--out", type=str, default="results_v3_scalable.json")
    ap.add_argument("--progress", action="store_true")
    
    args = ap.parse_args()
    N = args.N
    
    def log(msg: str):
        if args.progress:
            print(msg, flush=True)
    
    log("=" * 70)
    log(f"HSF CONSTRAINT TEST V3 — SCALABLE (N={N}, dim=2^{N}={2**N:,})")
    log("=" * 70)
    
    # Memory estimate
    mem_state_mb = (2 ** N * 16) / (1024 ** 2)
    log(f"State vector memory: {mem_state_mb:.1f} MB")
    
    if N > 26:
        log("WARNING: N>26 may exceed available memory!")
    
    rng = np.random.default_rng(args.seed)
    
    log(f"\nBuilding {args.topology} graph...")
    t0 = time.time()
    edges = build_edges(N, args.topology, rng, rr_deg=args.rr_deg)
    log(f"  {len(edges)} edges, built in {time.time()-t0:.2f}s")
    
    # Decide evolution method
    use_trotter = (args.method == "trotter") or (args.method == "auto" and N >= 20)
    
    if use_trotter:
        log(f"\nUsing TROTTER evolution (dt={args.trotter_dt}) - memory efficient for large N")
        t0 = time.time()
        # Need a fresh RNG with same seed for reproducibility
        rng_evolver = np.random.default_rng(args.seed + 1000)
        evolver = build_trotter_evolver(
            N=N, edges=edges, rng=rng_evolver,
            onsite_scale=args.onsite_scale,
            edge_scale=args.edge_scale,
            dt=args.trotter_dt,
        )
        log(f"  Trotter evolver built in {time.time()-t0:.2f}s")
        log(f"  {len(evolver.onsite_gates)} onsite gates, {len(evolver.edge_gates)} edge gates")
    else:
        log(f"\nUsing KRYLOV evolution (expm_multiply)")
        log(f"Building sparse Hamiltonian...")
        t0 = time.time()
        H = build_sparse_2local_hamiltonian(
            N=N, edges=edges, rng=rng,
            onsite_scale=args.onsite_scale,
            edge_scale=args.edge_scale,
            progress_callback=log if args.progress else None,
        )
        log(f"  Built in {time.time()-t0:.1f}s")
        log(f"  H shape: {H.shape}, nnz: {H.nnz:,}")
        mem_H_mb = (H.data.nbytes + H.indices.nbytes + H.indptr.nbytes) / (1024 ** 2)
        log(f"  H memory: {mem_H_mb:.1f} MB")
        evolver = SparseEvolver(N=N, H=H)
    
    log(f"\nGenerating initial state...")
    psi_base = random_product_state(N, rng)
    
    times = np.linspace(0.0, args.tmax, args.nt)
    
    # Determine blockings
    if args.blockings:
        blocking_sizes = [int(x.strip()) for x in args.blockings.split(",")]
    else:
        # Default: all factors of N
        blocking_sizes = [b for b in range(1, N + 1) if N % b == 0]
    blocking_sizes = sorted(set(blocking_sizes))
    
    log(f"\nBlockings to test: {blocking_sizes}")
    log(f"Times: {len(times)} points in [0, {args.tmax}]")
    
    # Bandwidth values
    if args.bandwidth_sweep:
        bandwidth_values = [float(x.strip()) for x in args.bandwidth_sweep.split(",")]
    else:
        bandwidth_values = [args.bandwidth]
    
    all_results = []
    
    for bw in bandwidth_values:
        log(f"\n{'='*50}")
        log(f"BANDWIDTH = {bw}")
        log(f"{'='*50}")
        
        results_this_bw = {
            "meta": {
                "N": N,
                "dim": 2 ** N,
                "topology": args.topology,
                "n_edges": len(edges),
                "seed": args.seed,
                "tmax": args.tmax,
                "nt": args.nt,
                "bandwidth_capacity": bw,
                "speed_threshold": args.speed_threshold,
                "recover_threshold": args.recover_threshold,
            },
            "blockings": [],
        }
        
        for b in blocking_sizes:
            eff = bandwidth_efficiency(b, bw)
            log(f"\n  block_size={b} (n_blocks={N//b}, efficiency={eff:.2f})")
            
            t0 = time.time()
            out = evaluate_blocking_v3_scalable(
                N=N,
                qubit_edges=edges,
                evolver=evolver,
                psi_base=psi_base,
                source_qubit=args.source,
                times=times,
                block_size=b,
                speed_threshold=args.speed_threshold,
                recover_threshold=args.recover_threshold,
                d_remote_min=args.d_remote_min,
                t_late_frac=args.t_late_frac,
                min_blocks=args.min_blocks,
                bandwidth_capacity=bw,
                bandwidth_penalty_scale=args.bandwidth_penalty_scale,
                progress_callback=log if args.progress else None,
            )
            elapsed = time.time() - t0
            
            log(f"    score={out['score']:+.3f} (took {elapsed:.1f}s)")
            log(f"    RAW: recover={out['frac_remote_recover_late_RAW']:.3f}, "
                f"EFF: recover={out['frac_remote_recover_late_EFF']:.3f}")
            
            results_this_bw["blockings"].append(out)
        
        # Sort by score
        results_this_bw["blockings"] = sorted(
            results_this_bw["blockings"], key=lambda x: x["score"], reverse=True
        )
        results_this_bw["winner"] = results_this_bw["blockings"][0]
        
        all_results.append(results_this_bw)
    
    # Output
    if len(bandwidth_values) == 1:
        final = all_results[0]
    else:
        final = {
            "sweep_type": "bandwidth",
            "bandwidth_values": bandwidth_values,
            "results_by_bandwidth": all_results,
        }
    
    with open(args.out, "w") as f:
        json.dump(final, f, indent=2)
    
    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    
    for res in all_results:
        bw = res["meta"]["bandwidth_capacity"]
        winner = res["winner"]
        log(f"\nBandwidth={bw}:")
        log(f"  {'Block':>6} {'#Blks':>6} {'Effic':>6} {'Score':>8} {'RecRAW':>7} {'RecEFF':>7}")
        log("  " + "-" * 50)
        for r in res["blockings"]:
            log(f"  {r['block_size']:>6} {r['n_blocks']:>6} {r['bandwidth_efficiency']:>6.2f} "
                f"{r['score']:>+8.3f} {r['frac_remote_recover_late_RAW']:>7.3f} "
                f"{r['frac_remote_recover_late_EFF']:>7.3f}")
        
        if winner["block_size"] == 1:
            verdict = "✓ SPATIAL WINS"
        elif winner["block_size"] == N:
            verdict = "✗ TRIVIAL WINS"
        else:
            verdict = f"? INTERMEDIATE (b={winner['block_size']})"
        log(f"\n  → WINNER: block_size={winner['block_size']} {verdict}")
    
    log(f"\nWrote: {args.out}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())