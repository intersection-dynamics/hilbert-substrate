#!/usr/bin/env python3
"""
CUDA-accelerated Quantum Simulation for Heterogeneity in Information Propagation (HIP)

Uses PyTorch for GPU acceleration. Falls back to CPU if CUDA unavailable.

Usage:
    python hip_cuda_simulation.py --n_nodes 10 --times 0.5 1.0 2.0 3.0 4.0
    python hip_cuda_simulation.py --n_nodes 12 --edge_prob 0.3 --n_state_samples 20
    python hip_cuda_simulation.py --help

Memory requirements (approximate, for Hamiltonian + unitary + states):
    8 qubits:   ~50 MB
    10 qubits:  ~200 MB
    12 qubits:  ~800 MB
    14 qubits:  ~12 GB
    16 qubits:  ~200 GB (likely won't fit on GPU)

Author: Ben Bray
"""

import argparse
import time
import os
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import Dict, Tuple, List, Optional

# ============================================================
# DEVICE CONFIGURATION
# ============================================================

def get_device(force_cpu: bool = False) -> torch.device:
    """Get the best available device."""
    if force_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon
    else:
        return torch.device('cpu')


def print_device_info(device: torch.device):
    """Print information about the compute device."""
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == 'mps':
        print("  Apple Silicon GPU")
    else:
        print("  CPU computation (no GPU acceleration)")


# ============================================================
# PAULI MATRICES
# ============================================================

def get_paulis(device: torch.device, dtype: torch.dtype = torch.complex128):
    """Return Pauli matrices on the specified device."""
    I = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    return I, X, Y, Z


# ============================================================
# TENSOR PRODUCTS AND OPERATOR EMBEDDING
# ============================================================

def tensor_product(ops: List[torch.Tensor]) -> torch.Tensor:
    """Compute tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = torch.kron(result, op)
    return result


def single_site_operator(op: torch.Tensor, site: int, n_sites: int,
                         I: torch.Tensor) -> torch.Tensor:
    """Embed a single-site operator into the full Hilbert space."""
    ops = [I] * n_sites
    ops[site] = op
    return tensor_product(ops)


def two_site_operator(op1: torch.Tensor, op2: torch.Tensor,
                      site1: int, site2: int, n_sites: int,
                      I: torch.Tensor) -> torch.Tensor:
    """Embed a two-site operator into the full Hilbert space."""
    ops = [I] * n_sites
    ops[site1] = op1
    ops[site2] = op2
    return tensor_product(ops)


# ============================================================
# HAMILTONIAN CONSTRUCTION
# ============================================================

def build_heisenberg_hamiltonian(G: nx.Graph, device: torch.device,
                                  h: float = 0.5, J: float = 1.0) -> torch.Tensor:
    """
    Build Heisenberg Hamiltonian on a graph.

    H = h * sum_i Z_i + J * sum_{(i,j) in E} (X_i X_j + Y_i Y_j + Z_i Z_j)
    """
    n = G.number_of_nodes()
    dim = 2 ** n
    I, X, Y, Z = get_paulis(device)

    H = torch.zeros((dim, dim), dtype=torch.complex128, device=device)

    # Local terms
    for i in G.nodes():
        H += h * single_site_operator(Z, i, n, I)

    # Interaction terms
    for (i, j) in G.edges():
        for P in [X, Y, Z]:
            H += J * two_site_operator(P, P, i, j, n, I)

    return H


def build_ising_hamiltonian(G: nx.Graph, device: torch.device,
                            h: float = 0.5, J: float = 1.0) -> torch.Tensor:
    """
    Build transverse-field Ising Hamiltonian on a graph.

    H = h * sum_i X_i + J * sum_{(i,j) in E} Z_i Z_j
    """
    n = G.number_of_nodes()
    dim = 2 ** n
    I, X, Y, Z = get_paulis(device)

    H = torch.zeros((dim, dim), dtype=torch.complex128, device=device)

    # Transverse field
    for i in G.nodes():
        H += h * single_site_operator(X, i, n, I)

    # Ising interaction
    for (i, j) in G.edges():
        H += J * two_site_operator(Z, Z, i, j, n, I)

    return H


# ============================================================
# TIME EVOLUTION
# ============================================================

def time_evolution_operator(H: torch.Tensor, t: float) -> torch.Tensor:
    """Compute U(t) = exp(-i H t) using matrix exponential."""
    return torch.linalg.matrix_exp(-1j * H * t)


# ============================================================
# PARTIAL TRACE
# ============================================================

def partial_trace_single_site(rho: torch.Tensor, keep_site: int,
                               n_sites: int) -> torch.Tensor:
    """
    Compute reduced density matrix for a single site.

    Uses explicit summation - clear but not the fastest possible.
    For very large systems, a reshape-based approach would be faster.
    """
    device = rho.device
    dtype = rho.dtype

    rho_reduced = torch.zeros((2, 2), dtype=dtype, device=device)

    for a in range(2):
        for b in range(2):
            for basis_other in range(2 ** (n_sites - 1)):
                row_idx = 0
                col_idx = 0
                other_pos = 0

                for s in range(n_sites):
                    if s == keep_site:
                        row_idx += a * (2 ** (n_sites - 1 - s))
                        col_idx += b * (2 ** (n_sites - 1 - s))
                    else:
                        bit = (basis_other >> (n_sites - 2 - other_pos)) & 1
                        row_idx += bit * (2 ** (n_sites - 1 - s))
                        col_idx += bit * (2 ** (n_sites - 1 - s))
                        other_pos += 1

                rho_reduced[a, b] += rho[row_idx, col_idx]

    return rho_reduced


def partial_trace_single_site_fast(rho: torch.Tensor, keep_site: int,
                                    n_sites: int) -> torch.Tensor:
    """
    Fast partial trace using tensor reshaping.

    Much faster for large systems but trickier to understand.
    """
    device = rho.device
    dtype = rho.dtype

    result = torch.zeros((2, 2), dtype=dtype, device=device)

    n_other = n_sites - 1
    dim_other = 2 ** n_other

    for a in range(2):
        for b in range(2):
            total = torch.tensor(0.0, dtype=dtype, device=device)

            for other_config in range(dim_other):
                row_bits = []
                col_bits = []
                other_pos = 0

                for s in range(n_sites):
                    if s == keep_site:
                        row_bits.append(a)
                        col_bits.append(b)
                    else:
                        bit = (other_config >> (n_other - 1 - other_pos)) & 1
                        row_bits.append(bit)
                        col_bits.append(bit)
                        other_pos += 1

                row_idx = sum(bb * (2 ** (n_sites - 1 - i)) for i, bb in enumerate(row_bits))
                col_idx = sum(bb * (2 ** (n_sites - 1 - i)) for i, bb in enumerate(col_bits))

                total = total + rho[row_idx, col_idx]

            result[a, b] = total

    return result


# ============================================================
# TRACE DISTANCE
# ============================================================

def trace_distance(rho1: torch.Tensor, rho2: torch.Tensor) -> float:
    """Compute trace distance: D(rho1, rho2) = (1/2) ||rho1 - rho2||_1"""
    diff = rho1 - rho2
    eigenvalues = torch.linalg.eigvalsh(diff)
    return 0.5 * torch.sum(torch.abs(eigenvalues)).item()


# ============================================================
# RANDOM STATES AND PERTURBATIONS
# ============================================================

def random_product_state(n_sites: int, device: torch.device) -> torch.Tensor:
    """Generate a random product state as a density matrix."""
    psi = torch.tensor([1.0], dtype=torch.complex128, device=device)

    for _ in range(n_sites):
        theta = np.arccos(2 * np.random.rand() - 1)
        phi = 2 * np.pi * np.random.rand()

        qubit = torch.tensor([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2)
        ], dtype=torch.complex128, device=device)

        psi = torch.kron(psi, qubit)

    return torch.outer(psi, psi.conj())


def random_su2(device: torch.device, enforce_det_one: bool = True) -> torch.Tensor:
    """Generate a random 2x2 Haar unitary, optionally projected to SU(2).

    Notes:
      - QR + phase correction gives a Haar-distributed U(2) matrix.
      - If enforce_det_one=True, we divide by sqrt(det(U)) to force det(U)=1 (up to branch choice),
        yielding an SU(2) element.
    """
    z = (torch.randn(2, 2, device=device) + 1j * torch.randn(2, 2, device=device)) / np.sqrt(2)
    z = z.to(torch.complex128)
    q, r = torch.linalg.qr(z)
    d = torch.diag(r)
    ph = d / torch.abs(d)
    U = q @ torch.diag(ph)

    if enforce_det_one:
        detU = torch.det(U)
        # Guard against pathological numerical issues; detU should never be 0 for a unitary.
        if torch.abs(detU) > 0:
            U = U / torch.sqrt(detU)

    return U


# ============================================================
# PERMEABILITY CALCULATION
# ============================================================

def directed_permeability_sample(
    H: torch.Tensor,
    U: torch.Tensor,
    rho: torch.Tensor,
    source: int,
    target: int,
    n_sites: int,
    n_operator_samples: int,
    device: torch.device
) -> float:
    """
    Estimate directed permeability for a single initial state.

    U is precomputed time evolution operator.
    """
    I, X, Y, Z = get_paulis(device)

    # Evolve unperturbed state
    rho_t = U @ rho @ U.conj().T
    rho_j = partial_trace_single_site(rho_t, target, n_sites)

    max_dist = 0.0

    # Try Pauli perturbations
    for P in [X, Y, Z]:
        O_full = single_site_operator(P, source, n_sites, I)
        rho_perturbed = O_full @ rho @ O_full.conj().T
        rho_perturbed_t = U @ rho_perturbed @ U.conj().T
        rho_perturbed_j = partial_trace_single_site(rho_perturbed_t, target, n_sites)

        dist = trace_distance(rho_j, rho_perturbed_j)
        max_dist = max(max_dist, dist)

    # Sample random SU(2) operators
    for _ in range(n_operator_samples):
        O_local = random_su2(device)
        O_full = single_site_operator(O_local, source, n_sites, I)

        rho_perturbed = O_full @ rho @ O_full.conj().T
        rho_perturbed_t = U @ rho_perturbed @ U.conj().T
        rho_perturbed_j = partial_trace_single_site(rho_perturbed_t, target, n_sites)

        dist = trace_distance(rho_j, rho_perturbed_j)
        max_dist = max(max_dist, dist)

    return max_dist


def typical_permeability(
    H: torch.Tensor,
    U: torch.Tensor,
    G: nx.Graph,
    source: int,
    target: int,
    n_state_samples: int,
    n_operator_samples: int,
    device: torch.device
) -> float:
    """Estimate typical directed permeability."""
    n_sites = G.number_of_nodes()
    permeabilities = []

    for _ in range(n_state_samples):
        rho = random_product_state(n_sites, device)
        p = directed_permeability_sample(
            H, U, rho, source, target, n_sites, n_operator_samples, device
        )
        permeabilities.append(p)

    return np.mean(permeabilities)


def compute_all_edge_permeabilities(
    H: torch.Tensor,
    U: torch.Tensor,
    G: nx.Graph,
    n_state_samples: int,
    n_operator_samples: int,
    device: torch.device,
    verbose: bool = True,
    edges_order: Optional[List[Tuple[int, int]]] = None
) -> Dict[Tuple[int, int], float]:
    """Compute symmetric edge permeabilities for all edges."""
    edge_perms = {}
    edges = list(edges_order) if edges_order is not None else list(G.edges())

    for idx, (i, j) in enumerate(edges):
        p_ij = typical_permeability(H, U, G, i, j, n_state_samples, n_operator_samples, device)
        p_ji = typical_permeability(H, U, G, j, i, n_state_samples, n_operator_samples, device)
        edge_perms[(i, j)] = max(p_ij, p_ji)

        if verbose:
            print(f"    Edge ({i},{j}): P = {edge_perms[(i,j)]:.4f}  [{idx+1}/{len(edges)}]")

    return edge_perms


def compute_node_intensities(
    H: torch.Tensor,
    U: torch.Tensor,
    G: nx.Graph,
    n_state_samples: int,
    n_operator_samples: int,
    device: torch.device
) -> np.ndarray:
    """Compute outgoing information intensity for all nodes."""
    n_sites = G.number_of_nodes()
    intensities = np.zeros(n_sites)

    perm_cache = {}

    for i in G.nodes():
        for j in G.neighbors(i):
            if (i, j) not in perm_cache:
                perm_cache[(i, j)] = typical_permeability(
                    H, U, G, i, j, n_state_samples, n_operator_samples, device
                )
            intensities[i] += perm_cache[(i, j)]

    return intensities


def compute_HIP(edge_permeabilities: Dict[Tuple[int, int], float]) -> float:
    """Compute HIP diagnostic as variance of edge permeabilities."""
    values = list(edge_permeabilities.values())
    return np.var(values)


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation(
    n_nodes: int,
    edge_prob: float,
    times: np.ndarray,
    hamiltonian_type: str,
    h: float,
    J: float,
    n_state_samples: int,
    n_operator_samples: int,
    seed: int,
    device: torch.device,
    output_dir: str
):
    """Run full HIP simulation."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Estimate memory requirement
    dim = 2 ** n_nodes
    mem_estimate_gb = (dim * dim * 16 * 3) / 1e9  # 3 matrices, complex128
    print(f"\nEstimated GPU memory needed: ~{mem_estimate_gb:.1f} GB")

    if device.type == 'cuda':
        available = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem_estimate_gb > available * 0.8:
            print(f"WARNING: May exceed GPU memory ({available:.1f} GB available)")

    # Generate graph
    print(f"\nGenerating connected graph with {n_nodes} nodes...")
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    attempts = 0
    while not nx.is_connected(G) and attempts < 100:
        G = nx.erdos_renyi_graph(n_nodes, edge_prob)
        attempts += 1

    if not nx.is_connected(G):
        raise ValueError("Could not generate connected graph")

    print(f"Graph has {G.number_of_edges()} edges")

    # Freeze an explicit edge order for reproducibility (used for saving P_edge).
    edges = list(G.edges())
    edge_index = np.array(edges, dtype=np.int64)  # shape (m, 2), undirected edge list in fixed order
    P_edge = np.zeros((len(times), len(edges)), dtype=np.float64)  # shape (T, m)

    # Build Hamiltonian
    print(f"\nBuilding {hamiltonian_type} Hamiltonian (h={h}, J={J})...")
    t0 = time.time()

    if hamiltonian_type == "heisenberg":
        H = build_heisenberg_hamiltonian(G, device, h, J)
    else:
        H = build_ising_hamiltonian(G, device, h, J)

    print(f"Hamiltonian built in {time.time() - t0:.2f}s")
    print(f"Hilbert space dimension: {dim}")

    # Results storage
    all_edge_perms = {}
    all_intensities = []
    hip_values = []

    for t_idx, t in enumerate(times):
        print(f"\n{'='*50}")
        print(f"Time t = {t:.2f}  ({t_idx + 1}/{len(times)})")
        print('='*50)

        # Compute time evolution operator
        print("  Computing U(t)...")
        t0 = time.time()
        U = time_evolution_operator(H, t)
        print(f"  U(t) computed in {time.time() - t0:.2f}s")

        # Edge permeabilities
        print("  Computing edge permeabilities...")
        edge_perms = compute_all_edge_permeabilities(
            H, U, G, n_state_samples, n_operator_samples, device, verbose=True, edges_order=edges
        )
        all_edge_perms[t] = edge_perms
        # Save edge permeabilities in fixed edge order for reproducible plotting later
        P_edge[t_idx, :] = np.array([edge_perms[e] for e in edges], dtype=np.float64)

        hip = compute_HIP(edge_perms)
        hip_values.append(hip)
        print(f"  HIP(t={t:.2f}) = {hip:.6f}")

        # Node intensities
        print("  Computing node intensities...")
        intensities = compute_node_intensities(
            H, U, G, n_state_samples, n_operator_samples, device
        )
        all_intensities.append(intensities)

        # Clear GPU cache periodically
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    all_intensities = np.array(all_intensities)
    hip_values = np.array(hip_values)

    # ========================================
    # Generate figures
    # ========================================

    print(f"\n{'='*50}")
    print("Generating figures...")
    print('='*50)

    # Figure 1: Edge-level HIP at middle time
    t_mid = times[len(times) // 2]
    edge_perms_mid = all_edge_perms[t_mid]

    pos = nx.spring_layout(G, seed=seed)
    pos_arr = np.array([pos[i] for i in range(n_nodes)], dtype=np.float64)

    # Reuse the frozen edge order from the simulation
    edges = [tuple(e) for e in edge_index.tolist()]
    edge_colors = P_edge[len(times) // 2, :].tolist()

    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=400,
                           node_color='lightgray', edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=edge_colors,
                           edge_cmap=plt.cm.viridis, width=3,
                           edge_vmin=0, edge_vmax=max(edge_colors) * 1.1)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                norm=plt.Normalize(vmin=0, vmax=max(edge_colors) * 1.1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(r'Edge permeability $P_{\{i,j\}}(t^*)$', fontsize=11)

    ax.set_title(f'Edge-level HIP snapshot (N={n_nodes}, t={t_mid:.1f})', fontsize=12)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/figure1_edge_hip.png', dpi=200)
    fig.savefig(f'{output_dir}/figure1_edge_hip.pdf')
    plt.close()

    # Figure 2: Persistence time series
    mean_intensity = all_intensities.mean(axis=0)
    top_k = min(5, n_nodes)
    top_nodes = np.argsort(mean_intensity)[-top_k:]

    fig, ax = plt.subplots(figsize=(9, 5))
    for node in top_nodes:
        ax.plot(times, all_intensities[:, node], 'o-', label=f'Node {node}',
                markersize=5, linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(r'Outgoing intensity $I_i(t)$', fontsize=11)
    ax.set_title(f'Persistence of HIP (N={n_nodes}, top-{top_k} nodes)', fontsize=12)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/figure2_persistence.png', dpi=200)
    fig.savefig(f'{output_dir}/figure2_persistence.pdf')
    # Backwards/LaTeX-friendly alias
    fig.savefig(f'{output_dir}/figure2_persistence_timeseries.png', dpi=200)
    fig.savefig(f'{output_dir}/figure2_persistence_timeseries.pdf')
    plt.close()

    # Rank stability
    ranks = np.argsort(np.argsort(-all_intensities, axis=1), axis=1)
    stability = []
    for t_idx in range(len(times) - 1):
        r, _ = spearmanr(ranks[t_idx], ranks[t_idx + 1])
        stability.append(r)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times[:-1], stability, 'o-', color='steelblue', markersize=6, linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Spearman rank correlation', fontsize=11)
    ax.set_title(f'Rank stability (N={n_nodes})', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/figure3_rank_stability.png', dpi=200)
    fig.savefig(f'{output_dir}/figure3_rank_stability.pdf')
    plt.close()

    # HIP over time
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, hip_values, 'o-', color='darkgreen', markersize=6, linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(r'HIP$(t)$', fontsize=11)
    ax.set_title(f'HIP diagnostic (N={n_nodes})', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/figure4_hip_vs_time.png', dpi=200)
    fig.savefig(f'{output_dir}/figure4_hip_vs_time.pdf')
    plt.close()

    # Save numerical results (NOW includes edge_index + P_edge + pos for full figure provenance)
    np.savez(f'{output_dir}/results.npz',
             times=times,
             intensities=all_intensities,
             hip_values=hip_values,
             rank_stability=np.array(stability),
             n_nodes=n_nodes,
             edge_prob=edge_prob,
             hamiltonian_type=hamiltonian_type,
             h=h, J=J,
             edge_index=edge_index,
             P_edge=P_edge,
             pos=pos_arr)

    # ========================================
    # Summary
    # ========================================

    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print('='*50)
    print(f"N = {n_nodes} qubits, {G.number_of_edges()} edges")
    print(f"Hamiltonian: {hamiltonian_type} (h={h}, J={J})")
    print(f"Times: {times[0]:.2f} to {times[-1]:.2f} ({len(times)} points)")
    print(f"\nHIP statistics:")
    print(f"  Mean:  {np.mean(hip_values):.6f}")
    print(f"  Std:   {np.std(hip_values):.6f}")
    print(f"  Range: [{np.min(hip_values):.6f}, {np.max(hip_values):.6f}]")
    print(f"\nRank stability:")
    print(f"  Mean: {np.mean(stability):.3f}")
    print(f"  Min:  {np.min(stability):.3f}")
    print(f"\nOutput saved to: {output_dir}/")

    return {
        'graph': G,
        'times': times,
        'intensities': all_intensities,
        'hip_values': hip_values,
        'rank_stability': np.array(stability),
        'edge_permeabilities': all_edge_perms,
        'edge_index': edge_index,
        'P_edge': P_edge,
        'pos': pos_arr
    }


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='CUDA-accelerated HIP quantum simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--n_nodes', type=int, default=8,
                        help='Number of qubits (warning: memory scales as 4^n)')
    parser.add_argument('--edge_prob', type=float, default=0.4,
                        help='Edge probability for random graph')
    parser.add_argument('--times', type=float, nargs='+',
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                        help='Time points to evaluate')
    parser.add_argument('--hamiltonian', type=str, default='heisenberg',
                        choices=['heisenberg', 'ising'],
                        help='Hamiltonian type')
    parser.add_argument('--h', type=float, default=0.5,
                        help='Local field strength')
    parser.add_argument('--J', type=float, default=1.0,
                        help='Coupling strength')
    parser.add_argument('--n_state_samples', type=int, default=15,
                        help='Number of random states for typical permeability')
    parser.add_argument('--n_operator_samples', type=int, default=15,
                        help='Number of SU(2) samples for supremum')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='hip_results',
                        help='Output directory')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU computation')

    args = parser.parse_args()

    print("="*60)
    print("HIP QUANTUM SIMULATION (CUDA-accelerated)")
    print("="*60)

    device = get_device(force_cpu=args.cpu)
    print_device_info(device)

    run_simulation(
        n_nodes=args.n_nodes,
        edge_prob=args.edge_prob,
        times=np.array(args.times),
        hamiltonian_type=args.hamiltonian,
        h=args.h,
        J=args.J,
        n_state_samples=args.n_state_samples,
        n_operator_samples=args.n_operator_samples,
        seed=args.seed,
        device=device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
