"""
Step 2: Build random 2-local Hamiltonians on arbitrary graphs.
Adapts the Hamiltonian construction from constraint_emergence_test_v3.
"""

import numpy as np
from scipy import sparse
import networkx as nx
from typing import Tuple

# Pauli matrices
I = sparse.csr_matrix(np.eye(2))
X = sparse.csr_matrix(np.array([[0, 1], [1, 0]]))
Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]]))
Z = sparse.csr_matrix(np.array([[1, 0], [0, -1]]))
PAULIS = [I, X, Y, Z]
PAULI_LABELS = ['I', 'X', 'Y', 'Z']


def kron_n(ops: list) -> sparse.csr_matrix:
    """Compute tensor product of list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = sparse.kron(result, op, format='csr')
    return result


def single_site_op(N: int, site: int, pauli_idx: int) -> sparse.csr_matrix:
    """
    Build N-qubit operator with Pauli at one site, identity elsewhere.
    
    N: number of qubits
    site: which qubit (0-indexed)
    pauli_idx: 0=I, 1=X, 2=Y, 3=Z
    """
    ops = [I] * N
    ops[site] = PAULIS[pauli_idx]
    return kron_n(ops)


def two_site_op(N: int, site1: int, site2: int, p1: int, p2: int) -> sparse.csr_matrix:
    """
    Build N-qubit operator with Paulis at two sites, identity elsewhere.
    
    N: number of qubits
    site1, site2: which qubits (0-indexed)
    p1, p2: Pauli indices (0=I, 1=X, 2=Y, 3=Z)
    """
    ops = [I] * N
    ops[site1] = PAULIS[p1]
    ops[site2] = PAULIS[p2]
    return kron_n(ops)


def build_graph_hamiltonian(
    G: nx.Graph, 
    N: int,
    h_std: float = 0.25,
    J_std: float = 0.8,
    seed: int = 0
) -> sparse.csr_matrix:
    """
    Build random 2-local Hamiltonian on graph G.
    
    H = sum_i h_i · sigma_i + sum_(i,j)∈E sum_PQ J_ij^PQ sigma_i^P sigma_j^Q
    
    G: interaction graph (networkx)
    N: number of qubits (should equal number of nodes in G)
    h_std: std dev for on-site field coefficients
    J_std: std dev for interaction coefficients
    seed: random seed for reproducibility
    
    Returns: sparse Hamiltonian matrix (2^N x 2^N)
    """
    assert G.number_of_nodes() == N, f"Graph has {G.number_of_nodes()} nodes but N={N}"
    
    np.random.seed(seed)
    dim = 2**N
    H = sparse.csr_matrix((dim, dim), dtype=complex)
    
    # On-site terms: h_i · sigma_i for each qubit
    for i in range(N):
        for p in range(1, 4):  # X, Y, Z (skip I)
            coeff = np.random.normal(0, h_std)
            H = H + coeff * single_site_op(N, i, p)
    
    # Interaction terms: J_ij^PQ sigma_i^P sigma_j^Q for each edge
    for (i, j) in G.edges():
        for p in range(1, 4):  # X, Y, Z
            for q in range(1, 4):  # X, Y, Z
                coeff = np.random.normal(0, J_std)
                H = H + coeff * two_site_op(N, i, j, p, q)
    
    # Ensure Hermitian (should be by construction, but numerical safety)
    H = (H + H.conj().T) / 2
    
    return H


def verify_hamiltonian(H: sparse.csr_matrix, G: nx.Graph, N: int):
    """Verify Hamiltonian properties."""
    dim = 2**N
    
    print(f"\nHamiltonian verification (N={N} qubits):")
    print(f"  Shape: {H.shape}")
    print(f"  Expected: ({dim}, {dim})")
    print(f"  Non-zeros: {H.nnz}")
    print(f"  Sparsity: {H.nnz / dim**2:.2e}")
    
    # Check Hermitian
    diff = H - H.conj().T
    hermitian_err = sparse.linalg.norm(diff)
    print(f"  Hermitian error: {hermitian_err:.2e}")
    assert hermitian_err < 1e-10, "Hamiltonian is not Hermitian!"
    
    # Check trace (should be ~0 for traceless Paulis)
    trace = H.diagonal().sum()
    print(f"  Trace: {trace:.2e}")
    
    # Compute a few eigenvalues to verify spectrum
    if N <= 10:  # Only for small systems
        from scipy.sparse.linalg import eigsh
        eigs = eigsh(H, k=min(6, dim-2), which='SA', return_eigenvectors=False)
        print(f"  Lowest eigenvalues: {np.sort(eigs.real)[:4]}")
    
    print("  ✓ Hamiltonian verified")


if __name__ == "__main__":
    from lattice_graphs import generate_lattice_graph
    
    print("=" * 50)
    print("Testing Hamiltonian construction on lattices")
    print("=" * 50)
    
    # Small test: N=8 (2^8 = 256 dim)
    print("\n--- Small test: N=8 ---")
    
    # 1D chain
    G_1d = generate_lattice_graph((8,), periodic=True)
    H_1d = build_graph_hamiltonian(G_1d, N=8, seed=42)
    print(f"\n1D chain (8 sites, coord=2):")
    verify_hamiltonian(H_1d, G_1d, N=8)
    
    # 2D grid (degenerate: 4x2 has coord issues, use 2x4)
    G_2d = generate_lattice_graph((4, 2), periodic=True)
    H_2d = build_graph_hamiltonian(G_2d, N=8, seed=42)
    print(f"\n2D grid (4x2, coord={G_2d.degree(0)}):")
    verify_hamiltonian(H_2d, G_2d, N=8)
    
    # Medium test: N=12 (2^12 = 4096 dim)
    print("\n--- Medium test: N=12 ---")
    
    G_ring = generate_lattice_graph((12,), periodic=True)
    H_ring = build_graph_hamiltonian(G_ring, N=12, seed=42)
    print(f"\n1D ring (12 sites):")
    verify_hamiltonian(H_ring, G_ring, N=12)
    
    G_grid = generate_lattice_graph((4, 3), periodic=True)
    H_grid = build_graph_hamiltonian(G_grid, N=12, seed=42)
    print(f"\n2D grid (4x3, coord={G_grid.degree(0)}):")
    verify_hamiltonian(H_grid, G_grid, N=12)
    
    print("\n" + "=" * 50)
    print("Hamiltonian construction verified!")
    print("=" * 50)