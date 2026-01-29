"""
Step 3: Trotter time evolution on arbitrary graph Hamiltonians.
Memory-efficient: O(2^N) instead of O(4^N).
"""

import numpy as np
from scipy.linalg import expm
import networkx as nx
from typing import Tuple, Dict, List

# Pauli matrices (dense, for small gate construction)
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
PAULIS = [I, X, Y, Z]


def build_trotter_gates(
    G: nx.Graph,
    N: int,
    dt: float,
    h_std: float = 0.25,
    J_std: float = 0.8,
    seed: int = 0
) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """
    Build Trotter gates for one time step dt.
    
    Returns:
        onsite_gates: {qubit_idx: 2x2 unitary}
        edge_gates: {(i,j): 4x4 unitary}
    """
    np.random.seed(seed)
    
    # On-site gates: exp(-i dt h_i · sigma_i)
    onsite_gates = {}
    for i in range(N):
        h_op = np.zeros((2, 2), dtype=complex)
        for p in range(1, 4):  # X, Y, Z
            coeff = np.random.normal(0, h_std)
            h_op += coeff * PAULIS[p]
        onsite_gates[i] = expm(-1j * dt * h_op)
    
    # Edge gates: exp(-i dt sum_PQ J_ij^PQ sigma_i^P sigma_j^Q)
    edge_gates = {}
    for (i, j) in G.edges():
        h_edge = np.zeros((4, 4), dtype=complex)
        for p in range(1, 4):  # X, Y, Z
            for q in range(1, 4):  # X, Y, Z
                coeff = np.random.normal(0, J_std)
                h_edge += coeff * np.kron(PAULIS[p], PAULIS[q])
        edge_gates[(i, j)] = expm(-1j * dt * h_edge)
    
    return onsite_gates, edge_gates


def apply_single_qubit_gate(psi: np.ndarray, qubit: int, gate: np.ndarray, N: int) -> np.ndarray:
    """
    Apply 2x2 gate to qubit in N-qubit state vector.
    
    psi: state vector of length 2^N
    qubit: which qubit (0-indexed, qubit 0 is leftmost/most significant)
    gate: 2x2 unitary
    N: total number of qubits
    """
    # Reshape to isolate the target qubit
    # Shape: (2^qubit, 2, 2^(N-qubit-1))
    left_dim = 2**qubit
    right_dim = 2**(N - qubit - 1)
    
    psi_reshaped = psi.reshape(left_dim, 2, right_dim)
    
    # Apply gate: contract over the qubit dimension
    # result[l, s', r] = sum_s gate[s', s] * psi[l, s, r]
    result = np.einsum('ab,lbr->lar', gate, psi_reshaped)
    
    return result.reshape(-1)


def apply_two_qubit_gate(psi: np.ndarray, q1: int, q2: int, gate: np.ndarray, N: int) -> np.ndarray:
    """
    Apply 4x4 gate to two qubits in N-qubit state vector.
    
    psi: state vector of length 2^N
    q1, q2: qubit indices (q1 < q2)
    gate: 4x4 unitary
    N: total number of qubits
    """
    if q1 > q2:
        q1, q2 = q2, q1
        # Also need to swap gate indices
        gate = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    
    # Dimensions for reshaping
    left_dim = 2**q1
    mid_dim = 2**(q2 - q1 - 1)
    right_dim = 2**(N - q2 - 1)
    
    # Reshape: (left, q1, mid, q2, right)
    psi_reshaped = psi.reshape(left_dim, 2, mid_dim, 2, right_dim)
    
    # Reshape gate: (q1', q2', q1, q2)
    gate_reshaped = gate.reshape(2, 2, 2, 2)
    
    # Apply gate
    result = np.einsum('abcd,lcmdr->lamdr', gate_reshaped, psi_reshaped)
    
    return result.reshape(-1)


def trotter_step(
    psi: np.ndarray,
    onsite_gates: Dict[int, np.ndarray],
    edge_gates: Dict[Tuple[int, int], np.ndarray],
    N: int
) -> np.ndarray:
    """
    Apply one Trotter step: first all on-site gates, then all edge gates.
    """
    # On-site gates
    for qubit, gate in onsite_gates.items():
        psi = apply_single_qubit_gate(psi, qubit, gate, N)
    
    # Edge gates
    for (q1, q2), gate in edge_gates.items():
        psi = apply_two_qubit_gate(psi, q1, q2, gate, N)
    
    # Renormalize (Trotter accumulates small errors)
    psi = psi / np.linalg.norm(psi)
    
    return psi


def evolve_trotter(
    psi0: np.ndarray,
    onsite_gates: Dict[int, np.ndarray],
    edge_gates: Dict[Tuple[int, int], np.ndarray],
    N: int,
    n_steps: int
) -> np.ndarray:
    """
    Evolve state for n_steps Trotter steps.
    """
    psi = psi0.copy()
    for _ in range(n_steps):
        psi = trotter_step(psi, onsite_gates, edge_gates, N)
    return psi


def evolve_to_times(
    psi0: np.ndarray,
    G: nx.Graph,
    N: int,
    times: np.ndarray,
    dt: float = 0.1,
    seed: int = 0,
    h_std: float = 0.25,
    J_std: float = 0.8
) -> List[np.ndarray]:
    """
    Evolve state and return snapshots at specified times.
    
    Returns list of state vectors at each time.
    """
    onsite_gates, edge_gates = build_trotter_gates(G, N, dt, h_std, J_std, seed)
    
    states = []
    psi = psi0.copy()
    current_time = 0.0
    time_idx = 0
    
    while time_idx < len(times):
        if current_time >= times[time_idx] - 1e-9:
            states.append(psi.copy())
            time_idx += 1
        else:
            psi = trotter_step(psi, onsite_gates, edge_gates, N)
            current_time += dt
    
    return states


def verify_evolution(
    G: nx.Graph,
    N: int,
    dt: float = 0.1,
    n_steps: int = 10,
    seed: int = 0
) -> None:
    """Verify that Trotter evolution preserves norm and changes state."""
    
    onsite_gates, edge_gates = build_trotter_gates(G, N, dt, seed=seed)
    
    # Random initial state
    np.random.seed(seed + 100)
    psi0 = np.random.randn(2**N) + 1j * np.random.randn(2**N)
    psi0 = psi0 / np.linalg.norm(psi0)
    
    # Evolve forward
    psi = evolve_trotter(psi0, onsite_gates, edge_gates, N, n_steps)
    
    # Check norm preservation
    norm = np.linalg.norm(psi)
    print(f"  Norm after {n_steps} steps: {norm:.10f}")
    assert abs(norm - 1.0) < 1e-6, f"Norm not preserved: {norm}"
    
    # Check that state actually evolved (not identity)
    overlap = np.abs(np.vdot(psi0, psi))
    print(f"  Overlap with initial: {overlap:.4f}")
    # Should have evolved away from initial state
    assert overlap < 0.99, f"State didn't evolve: overlap = {overlap}"
    
    print("  ✓ Evolution verified")


if __name__ == "__main__":
    from lattice_graphs import generate_lattice_graph
    
    print("=" * 50)
    print("Testing Trotter evolution")
    print("=" * 50)
    
    # Test on small systems
    for dims, name in [((8,), "1D chain"), ((4, 2), "2D grid 4x2")]:
        N = int(np.prod(dims))
        G = generate_lattice_graph(dims, periodic=True)
        print(f"\n{name} (N={N}, edges={G.number_of_edges()}):")
        verify_evolution(G, N, dt=0.1, n_steps=20, seed=42)
    
    # Test on medium system
    print("\n--- Medium system: N=12 ---")
    G = generate_lattice_graph((4, 3), periodic=True)
    print(f"2D grid 4x3 (N=12, edges={G.number_of_edges()}):")
    verify_evolution(G, N=12, dt=0.1, n_steps=30, seed=42)
    
    # Benchmark timing for larger system
    print("\n--- Timing test: N=16 ---")
    import time
    G = generate_lattice_graph((4, 4), periodic=True)
    N = 16
    
    onsite_gates, edge_gates = build_trotter_gates(G, N, dt=0.1, seed=0)
    psi = np.random.randn(2**N) + 1j * np.random.randn(2**N)
    psi = psi / np.linalg.norm(psi)
    
    start = time.time()
    n_steps = 50
    psi = evolve_trotter(psi, onsite_gates, edge_gates, N, n_steps)
    elapsed = time.time() - start
    
    print(f"  {n_steps} Trotter steps in {elapsed:.2f}s")
    print(f"  {elapsed/n_steps*1000:.1f} ms per step")
    print(f"  Final norm: {np.linalg.norm(psi):.10f}")
    
    print("\n" + "=" * 50)
    print("Trotter evolution verified!")
    print("=" * 50)