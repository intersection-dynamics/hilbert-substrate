"""
Step 4: Compute influence via trace distance of reduced density matrices.

Influence I_b(t) = (1/2) || rho_b(t) - rho'_b(t) ||_1

where rho_b is the reduced density matrix of block b, and rho' is 
the state after a local perturbation.
"""

import numpy as np
from typing import List, Tuple, Dict
import networkx as nx


def partial_trace(psi: np.ndarray, keep_qubits: List[int], N: int) -> np.ndarray:
    """
    Compute reduced density matrix by tracing out qubits not in keep_qubits.
    
    psi: state vector of length 2^N
    keep_qubits: list of qubit indices to keep (0-indexed)
    N: total number of qubits
    
    Returns: reduced density matrix of shape (2^k, 2^k) where k = len(keep_qubits)
    """
    k = len(keep_qubits)
    trace_qubits = [q for q in range(N) if q not in keep_qubits]
    
    # Reshape state into tensor with one index per qubit
    psi_tensor = psi.reshape([2] * N)
    
    # Compute |psi><psi| as tensor
    rho_tensor = np.outer(psi, psi.conj()).reshape([2] * (2 * N))
    
    # Trace out unwanted qubits
    # For each traced qubit q, we contract indices q and q+N
    # We need to do this carefully, adjusting indices as we go
    
    # Sort trace_qubits in descending order to avoid index shifting issues
    for q in sorted(trace_qubits, reverse=True):
        # Contract index q with index q + N (in current tensor)
        # After each trace, the tensor shrinks
        n_remaining = rho_tensor.ndim // 2
        # Index q in first half, index q + n_remaining in second half
        rho_tensor = np.trace(rho_tensor, axis1=q, axis2=q + n_remaining)
    
    # Reshape to matrix
    dim_k = 2**k
    rho = rho_tensor.reshape(dim_k, dim_k)
    
    return rho


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute trace distance: D(rho1, rho2) = (1/2) ||rho1 - rho2||_1
    
    The trace norm ||A||_1 = sum of singular values = Tr(sqrt(A† A))
    """
    diff = rho1 - rho2
    # Singular values of diff
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(singular_values)


def compute_influence(
    psi: np.ndarray,
    psi_perturbed: np.ndarray,
    block_qubits: List[int],
    N: int
) -> float:
    """
    Compute influence of perturbation on a block.
    
    I_block = (1/2) || rho_block - rho'_block ||_1
    
    psi: unperturbed state
    psi_perturbed: state after local perturbation
    block_qubits: which qubits are in this block
    N: total qubits
    
    Returns: influence (between 0 and 1)
    """
    rho = partial_trace(psi, block_qubits, N)
    rho_pert = partial_trace(psi_perturbed, block_qubits, N)
    return trace_distance(rho, rho_pert)


def apply_pauli_perturbation(psi: np.ndarray, qubit: int, pauli: str, N: int) -> np.ndarray:
    """
    Apply Pauli operator to a single qubit.
    
    psi: state vector
    qubit: which qubit (0-indexed)
    pauli: 'X', 'Y', or 'Z'
    N: total qubits
    
    Returns: perturbed state (not normalized - Paulis are unitary)
    """
    # Pauli matrices
    paulis = {
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }
    P = paulis[pauli]
    
    # Reshape and apply
    left_dim = 2**qubit
    right_dim = 2**(N - qubit - 1)
    
    psi_reshaped = psi.reshape(left_dim, 2, right_dim)
    result = np.einsum('ab,lbr->lar', P, psi_reshaped)
    
    return result.reshape(-1)


def measure_all_influences(
    psi: np.ndarray,
    psi_perturbed: np.ndarray,
    blocks: List[List[int]],
    N: int
) -> Dict[int, float]:
    """
    Measure influence on all blocks.
    
    Returns: {block_idx: influence}
    """
    influences = {}
    for idx, block_qubits in enumerate(blocks):
        influences[idx] = compute_influence(psi, psi_perturbed, block_qubits, N)
    return influences


def create_contiguous_blocks(N: int, block_size: int) -> List[List[int]]:
    """
    Create contiguous blocks of qubits.
    
    E.g., N=12, block_size=3 -> [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
    """
    assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"
    n_blocks = N // block_size
    return [[block_size * i + j for j in range(block_size)] for i in range(n_blocks)]


def verify_partial_trace():
    """Verify partial trace implementation."""
    print("\n--- Verifying partial trace ---")
    
    # Test 1: 2-qubit product state
    # |00> -> tracing out qubit 1 should give |0><0|
    psi_00 = np.array([1, 0, 0, 0], dtype=complex)
    rho_0 = partial_trace(psi_00, [0], N=2)
    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    assert np.allclose(rho_0, expected), f"Product state test failed: {rho_0}"
    print("  ✓ Product state |00>: trace out qubit 1 gives |0><0|")
    
    # Test 2: Bell state |00> + |11>
    # Tracing out either qubit should give maximally mixed state
    psi_bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho_bell = partial_trace(psi_bell, [0], N=2)
    expected_mixed = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    assert np.allclose(rho_bell, expected_mixed), f"Bell state test failed: {rho_bell}"
    print("  ✓ Bell state: trace out one qubit gives maximally mixed")
    
    # Test 3: 3-qubit state, trace out middle qubit
    # |000> -> trace out qubit 1 -> |00><00| on qubits 0,2
    psi_000 = np.zeros(8, dtype=complex)
    psi_000[0] = 1.0
    rho_02 = partial_trace(psi_000, [0, 2], N=3)
    expected_02 = np.zeros((4, 4), dtype=complex)
    expected_02[0, 0] = 1.0
    assert np.allclose(rho_02, expected_02), f"3-qubit test failed: {rho_02}"
    print("  ✓ 3-qubit |000>: trace out middle gives |00><00|")
    
    # Test 4: Verify trace = 1
    np.random.seed(42)
    psi_rand = np.random.randn(16) + 1j * np.random.randn(16)
    psi_rand = psi_rand / np.linalg.norm(psi_rand)
    rho_partial = partial_trace(psi_rand, [1, 2], N=4)
    trace_val = np.trace(rho_partial)
    assert np.abs(trace_val - 1.0) < 1e-10, f"Trace not 1: {trace_val}"
    print(f"  ✓ Random state: reduced density matrix has trace = {trace_val.real:.10f}")
    
    print("  ✓ All partial trace tests passed")


def verify_influence():
    """Verify influence computation."""
    print("\n--- Verifying influence computation ---")
    
    N = 4
    
    # Test 1: No perturbation -> zero influence
    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1.0  # |0000>
    blocks = create_contiguous_blocks(N, block_size=2)
    influences = measure_all_influences(psi, psi, blocks, N)
    assert all(inf < 1e-10 for inf in influences.values()), f"Should be zero: {influences}"
    print("  ✓ No perturbation gives zero influence")
    
    # Test 2: Pauli X on qubit 0 of |0000> gives |1000>
    # Block 0 (qubits 0,1) should see maximum influence
    psi_pert = apply_pauli_perturbation(psi, qubit=0, pauli='X', N=N)
    influences = measure_all_influences(psi, psi_pert, blocks, N)
    print(f"  Influences after X on qubit 0: {influences}")
    assert influences[0] > 0.9, f"Block 0 should see large influence: {influences[0]}"
    assert influences[1] < 0.01, f"Block 1 should see no influence: {influences[1]}"
    print("  ✓ Perturbation on qubit 0 affects block 0, not block 1")
    
    # Test 3: Random state - perturbation should generally change reduced state
    np.random.seed(123)
    psi_rand = np.random.randn(2**N) + 1j * np.random.randn(2**N)
    psi_rand = psi_rand / np.linalg.norm(psi_rand)
    psi_rand_pert = apply_pauli_perturbation(psi_rand, qubit=0, pauli='X', N=N)
    influences_rand = measure_all_influences(psi_rand, psi_rand_pert, blocks, N)
    print(f"  Influences on random state: {influences_rand}")
    # Block 0 (containing qubit 0) should see significant influence
    assert influences_rand[0] > 0.1, f"Should affect block 0: {influences_rand[0]}"
    print("  ✓ Random state shows expected influence pattern")
    
    print("  ✓ All influence tests passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing influence computation")
    print("=" * 50)
    
    verify_partial_trace()
    verify_influence()
    
    # Integration test with Trotter evolution
    print("\n--- Integration test: Trotter + Influence ---")
    
    from lattice_graphs import generate_lattice_graph
    from trotter_evolution import build_trotter_gates, evolve_trotter
    
    # Setup: 8-qubit 1D chain
    N = 8
    G = generate_lattice_graph((8,), periodic=True)
    dt = 0.1
    
    # Build gates
    onsite, edge = build_trotter_gates(G, N, dt, seed=0)
    
    # Initial product state |00000000>
    psi0 = np.zeros(2**N, dtype=complex)
    psi0[0] = 1.0
    
    # Perturbed state: X on qubit 0
    psi0_pert = apply_pauli_perturbation(psi0, qubit=0, pauli='X', N=N)
    
    # Evolve both
    n_steps = 20
    psi_t = evolve_trotter(psi0, onsite, edge, N, n_steps)
    psi_t_pert = evolve_trotter(psi0_pert, onsite, edge, N, n_steps)
    
    # Measure influence on each qubit (block_size=1)
    blocks = create_contiguous_blocks(N, block_size=1)
    influences = measure_all_influences(psi_t, psi_t_pert, blocks, N)
    
    print(f"  After {n_steps} steps (t={n_steps*dt:.1f}):")
    print(f"  Influences by qubit: {[f'{influences[i]:.3f}' for i in range(N)]}")
    
    # Check that influence has spread from qubit 0
    max_inf = max(influences.values())
    n_affected = sum(1 for v in influences.values() if v > 0.01)
    print(f"  Max influence: {max_inf:.3f}")
    print(f"  Qubits affected (>0.01): {n_affected}")
    
    assert n_affected > 1, "Influence should spread beyond source qubit"
    print("  ✓ Influence spreads under Trotter evolution")
    
    print("\n" + "=" * 50)
    print("Influence computation verified!")
    print("=" * 50)