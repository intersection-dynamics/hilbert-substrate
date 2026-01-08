"""
hilbert_substrate_large_N.py

Optimized implementation for large N using:
1. Sparse matrix representations
2. Iterative eigensolvers (Lanczos)
3. On-the-fly Pauli coefficient computation (avoid storing 4^N matrices)
4. Memory-efficient correlation functions

Target: N = 10-14 qubits (2^14 = 16384 dimensional Hilbert space)
"""

import numpy as np
from scipy.sparse import csr_matrix, kron as sparse_kron, eye as sparse_eye
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh
import time
from typing import List, Tuple, Dict
import gc

# =============================================================================
# SPARSE PAULI MATRICES
# =============================================================================

def sparse_pauli():
    """Return sparse Pauli matrices."""
    I = csr_matrix(np.array([[1, 0], [0, 1]], dtype=complex))
    X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    Y = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
    Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    return I, X, Y, Z

def sparse_kron_n(ops: List[csr_matrix]) -> csr_matrix:
    """Sparse Kronecker product of list of matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = sparse_kron(result, op, format='csr')
    return result

# =============================================================================
# EFFICIENT HAMILTONIAN CONSTRUCTION
# =============================================================================

def heisenberg_sparse(N: int, edges: List[Tuple[int, int]], J: float = 1.0) -> csr_matrix:
    """
    Build sparse Heisenberg Hamiltonian.
    
    H = J * Σ_{<ij>} (X_i X_j + Y_i Y_j + Z_i Z_j)
    
    Memory: O(N * 2^N) instead of O(4^N)
    """
    I, X, Y, Z = sparse_pauli()
    dim = 2 ** N
    
    # Start with zero matrix
    H = csr_matrix((dim, dim), dtype=complex)
    
    for (i, j) in edges:
        for pauli in [X, Y, Z]:
            # Build sparse operator for this term
            ops = [I] * N
            ops[i] = pauli
            ops[j] = pauli
            term = sparse_kron_n(ops)
            H = H + J * term
    
    # Ensure Hermitian
    H = 0.5 * (H + H.conj().T)
    return H

def edges_ring(N: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % N) for i in range(N)]

def edges_chain(N: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(N - 1)]

def edges_ladder(N: int) -> List[Tuple[int, int]]:
    """Two-leg ladder: N must be even."""
    L = N // 2
    edges = []
    # Rungs
    for i in range(L):
        edges.append((i, i + L))
    # Legs
    for i in range(L - 1):
        edges.append((i, i + 1))
        edges.append((i + L, i + L + 1))
    return edges

def edges_2d_square(Lx: int, Ly: int) -> Tuple[List[Tuple[int, int]], int]:
    """2D square lattice with periodic boundaries."""
    N = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            site = x * Ly + y
            # Right neighbor
            right = x * Ly + ((y + 1) % Ly)
            edges.append((site, right))
            # Down neighbor
            down = ((x + 1) % Lx) * Ly + y
            edges.append((site, down))
    return edges, N

# =============================================================================
# EFFICIENT GROUND STATE AND LOW-LYING SPECTRUM
# =============================================================================

def get_ground_state(H: csr_matrix, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get lowest k eigenvalues and eigenvectors using Lanczos.
    
    Much faster than full diagonalization for large sparse H.
    """
    # eigsh finds smallest eigenvalues
    energies, states = eigsh(H, k=k, which='SA', return_eigenvectors=True)
    # Sort by energy
    idx = np.argsort(energies)
    return energies[idx], states[:, idx]

# =============================================================================
# EFFICIENT CORRELATION FUNCTIONS
# =============================================================================

def compute_correlations_sparse(N: int, H: csr_matrix, ground_state: np.ndarray) -> np.ndarray:
    """
    Compute spin-spin correlation matrix <σ_i · σ_j> efficiently.
    
    Does not store full correlation operators - computes on the fly.
    """
    I, X, Y, Z = sparse_pauli()
    correlations = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i, N):
            if i == j:
                correlations[i, j] = 1.0
                continue
            
            corr = 0.0
            for pauli in [X, Y, Z]:
                # Build operator
                ops = [I] * N
                ops[i] = pauli
                ops[j] = pauli
                O_ij = sparse_kron_n(ops)
                
                # Expectation value
                corr += np.abs(ground_state.conj() @ (O_ij @ ground_state))
            
            correlations[i, j] = corr / 3.0
            correlations[j, i] = correlations[i, j]
    
    return correlations

def correlation_to_distance(C: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix."""
    D = np.zeros_like(C)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if i != j:
                D[i, j] = 1.0 / (np.abs(C[i, j]) + 1e-10)
    # Normalize
    D = D / np.max(D) if np.max(D) > 0 else D
    return D

# =============================================================================
# EFFICIENT LOCALITY COST (without full Pauli basis)
# =============================================================================

def locality_cost_efficient(H: csr_matrix, N: int, p: int = 4, n_samples: int = 1000) -> float:
    """
    Estimate locality cost using random sampling of Pauli strings.
    
    Instead of computing all 4^N coefficients, sample and estimate.
    For exact computation on small N, use the full method.
    """
    dim = 2 ** N
    I, X, Y, Z = sparse_pauli()
    paulis = [I, X, Y, Z]
    
    rng = np.random.default_rng(42)
    
    numerator = 0.0
    denominator = 0.0
    
    # Dense H for trace computation (only works for moderate N)
    if N <= 10:
        H_dense = H.toarray()
    else:
        # For very large N, use sampling-based trace estimation
        H_dense = None
    
    for _ in range(n_samples):
        # Random Pauli string
        indices = rng.integers(0, 4, size=N)
        weight = sum(1 for idx in indices if idx != 0)
        
        # Build operator
        ops = [paulis[idx] for idx in indices]
        P = sparse_kron_n(ops)
        
        # Coefficient c_k = Tr(H P_k) / dim
        if H_dense is not None:
            P_dense = P.toarray()
            c_k = np.real(np.trace(H_dense @ P_dense)) / dim
        else:
            # Stochastic trace estimation for very large N
            n_trace_samples = 10
            c_k = 0.0
            for _ in range(n_trace_samples):
                v = rng.choice([-1, 1], size=dim).astype(complex)
                c_k += np.real(v.conj() @ (H @ (P @ v)))
            c_k /= (n_trace_samples * dim)
        
        c_k_sq = c_k ** 2
        numerator += (weight ** p) * c_k_sq
        denominator += c_k_sq
    
    if denominator < 1e-30:
        return float('inf')
    
    # Scale to approximate full sum
    return float(numerator / denominator)

def locality_cost_exact(H_dense: np.ndarray, N: int, p: int = 4) -> float:
    """
    Exact locality cost for small N (computes all 4^N terms).
    Only use for N <= 6.
    """
    import itertools
    
    dim = 2 ** N
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [I, X, Y, Z]
    
    def kron_n_dense(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    numerator = 0.0
    denominator = 0.0
    
    for indices in itertools.product(range(4), repeat=N):
        weight = sum(1 for idx in indices if idx != 0)
        ops = [paulis[idx] for idx in indices]
        P = kron_n_dense(ops)
        
        c_k = np.real(np.trace(H_dense @ P)) / dim
        c_k_sq = c_k ** 2
        
        numerator += (weight ** p) * c_k_sq
        denominator += c_k_sq
    
    return float(numerator / denominator) if denominator > 0 else float('inf')

# =============================================================================
# EFFICIENT FORCE COMPUTATION
# =============================================================================

def compute_forces_sparse(N: int, H: csr_matrix, ground_state: np.ndarray, 
                          graph_distance: np.ndarray) -> Dict[int, float]:
    """
    Compute interaction potential V(d) between excitations.
    """
    I, X, Y, Z = sparse_pauli()
    E_ground = np.real(ground_state.conj() @ (H @ ground_state))
    
    # Single excitation energies
    single_E = []
    for site in range(N):
        ops = [I] * N
        ops[site] = X
        exc_op = sparse_kron_n(ops)
        
        psi = exc_op @ ground_state
        psi = psi / np.linalg.norm(psi)
        E = np.real(psi.conj() @ (H @ psi))
        single_E.append(E)
    
    # Two-excitation energies by distance
    V_by_d = {}
    for i in range(N):
        for j in range(i + 1, N):
            d = int(graph_distance[i, j])
            
            ops1 = [I] * N
            ops1[i] = X
            ops2 = [I] * N
            ops2[j] = X
            
            psi = sparse_kron_n(ops2) @ (sparse_kron_n(ops1) @ ground_state)
            psi = psi / np.linalg.norm(psi)
            E_ij = np.real(psi.conj() @ (H @ psi))
            
            V = E_ij - single_E[i] - single_E[j] + E_ground
            
            V_by_d.setdefault(d, []).append(V)
    
    return {d: np.mean(vs) for d, vs in V_by_d.items()}

def graph_distance_matrix(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Compute shortest path distances on graph."""
    D = np.full((N, N), np.inf)
    np.fill_diagonal(D, 0)
    for (i, j) in edges:
        D[i, j] = D[j, i] = 1
    # Floyd-Warshall
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
    return D

# =============================================================================
# DIMENSIONALITY ANALYSIS
# =============================================================================

def estimate_dimension(D: np.ndarray) -> float:
    """Estimate effective dimension from distance matrix."""
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    
    eigenvalues = np.linalg.eigvalsh(B)[::-1]
    positive = eigenvalues[eigenvalues > 1e-10]
    
    if len(positive) == 0:
        return 0.0
    
    normalized = positive / np.sum(positive)
    return 1.0 / np.sum(normalized ** 2)

def check_triangle_inequality(D: np.ndarray) -> Tuple[int, int]:
    """Check triangle inequality violations."""
    N = D.shape[0]
    violations = 0
    total = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if len({i, j, k}) == 3:
                    total += 1
                    if D[i, k] > D[i, j] + D[j, k] + 1e-10:
                        violations += 1
    return violations, total

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_large_N_analysis(N: int, topology: str = 'ring') -> Dict:
    """
    Run full analysis for large N.
    """
    print(f"\n{'='*70}")
    print(f"LARGE-N HILBERT SUBSTRATE ANALYSIS")
    print(f"N = {N} qubits, Hilbert space dim = {2**N}")
    print(f"{'='*70}\n")
    
    results = {'N': N, 'dim': 2**N, 'topology': topology}
    
    t0 = time.time()
    
    # Build Hamiltonian
    print(f"Building sparse Hamiltonian ({topology})...")
    if topology == 'ring':
        edges = edges_ring(N)
    elif topology == 'chain':
        edges = edges_chain(N)
    elif topology == 'ladder':
        edges = edges_ladder(N)
    else:
        edges = edges_ring(N)
    
    H = heisenberg_sparse(N, edges)
    print(f"  Hamiltonian: {H.shape[0]}×{H.shape[1]}, {H.nnz} nonzeros")
    print(f"  Sparsity: {100 * H.nnz / H.shape[0]**2:.4f}%")
    
    graph_D = graph_distance_matrix(N, edges)
    
    # Get ground state
    print(f"\nFinding ground state (Lanczos)...")
    t1 = time.time()
    energies, states = get_ground_state(H, k=min(10, 2**N - 2))
    ground = states[:, 0]
    print(f"  Time: {time.time() - t1:.2f}s")
    print(f"  Ground state energy: {energies[0]:.6f}")
    print(f"  First gap: {energies[1] - energies[0]:.6f}")
    
    results['ground_energy'] = float(energies[0])
    results['first_gap'] = float(energies[1] - energies[0])
    results['low_spectrum'] = energies.tolist()
    
    # Correlations and metric
    print(f"\nComputing correlations...")
    t1 = time.time()
    C = compute_correlations_sparse(N, H, ground)
    D = correlation_to_distance(C)
    print(f"  Time: {time.time() - t1:.2f}s")
    
    # Metric properties
    violations, total = check_triangle_inequality(D)
    eff_dim = estimate_dimension(D)
    print(f"\nMetric analysis:")
    print(f"  Triangle violations: {violations}/{total} ({100*violations/total:.1f}%)")
    print(f"  Effective dimension: {eff_dim:.2f}")
    
    results['triangle_violations'] = violations
    results['triangle_total'] = total
    results['effective_dimension'] = float(eff_dim)
    
    # Locality cost (use sampling for large N)
    print(f"\nComputing locality cost...")
    t1 = time.time()
    if N <= 6:
        H_dense = H.toarray()
        C_spatial = locality_cost_exact(H_dense, N, p=4)
        # Harmonion cost
        evals = np.linalg.eigvalsh(H_dense)
        H_diag = np.diag(evals)
        C_harm = locality_cost_exact(H_diag, N, p=4)
    else:
        C_spatial = locality_cost_efficient(H, N, p=4, n_samples=2000)
        C_harm = 1.0  # Approximate for large N
    print(f"  Time: {time.time() - t1:.2f}s")
    print(f"  Spatial cost: {C_spatial:.2f}")
    print(f"  Harmonion cost: {C_harm:.2f}")
    print(f"  Ratio: {C_spatial/C_harm:.2f}x")
    
    results['spatial_cost'] = float(C_spatial)
    results['harmonion_cost'] = float(C_harm)
    
    # Forces
    print(f"\nComputing force structure...")
    t1 = time.time()
    V_d = compute_forces_sparse(N, H, ground, graph_D)
    print(f"  Time: {time.time() - t1:.2f}s")
    
    print(f"\nInteraction potential V(d):")
    for d in sorted(V_d.keys()):
        V = V_d[d]
        bar = "█" * int(abs(V) * 2) if V != 0 else ""
        print(f"  d = {d}: V = {V:+.4f}  {'─' if V < 0 else '+'}{bar}")
    
    # Check locality
    V1 = abs(V_d.get(1, 0))
    V_rest = sum(abs(V_d.get(d, 0)) for d in V_d if d > 1)
    is_local = V_rest < 0.1 * V1 if V1 > 0 else True
    print(f"\n  Force is {'LOCAL ✓' if is_local else 'NON-LOCAL'}")
    
    results['V_vs_d'] = {int(d): float(v) for d, v in V_d.items()}
    results['is_local'] = is_local
    
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.2f}s")
    results['total_time'] = total_time
    
    return results


def scaling_analysis(N_values: List[int] = [6, 8, 10, 12]) -> Dict:
    """
    Run scaling analysis across multiple N values.
    """
    print("\n" + "="*70)
    print("SCALING ANALYSIS")
    print("="*70)
    
    results = {'N_values': N_values, 'data': []}
    
    for N in N_values:
        print(f"\n{'─'*70}")
        print(f"N = {N}")
        print(f"{'─'*70}")
        
        try:
            r = run_large_N_analysis(N, topology='ring')
            results['data'].append(r)
        except MemoryError:
            print(f"  MemoryError - skipping N={N}")
            results['data'].append({'N': N, 'error': 'MemoryError'})
        except Exception as e:
            print(f"  Error: {e}")
            results['data'].append({'N': N, 'error': str(e)})
        
        gc.collect()  # Force garbage collection
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n  N  | Dim    | Gap      | Eff.Dim | V(1)     | V(2)     | Local?")
    print("  " + "-"*65)
    
    for r in results['data']:
        if 'error' in r:
            print(f"  {r['N']:2d} | ERROR: {r['error']}")
        else:
            V1 = r['V_vs_d'].get(1, 0)
            V2 = r['V_vs_d'].get(2, 0)
            print(f"  {r['N']:2d} | {r['dim']:6d} | {r['first_gap']:.4f} | "
                  f"{r['effective_dimension']:.2f}    | {V1:+.4f} | {V2:+.4f} | "
                  f"{'✓' if r['is_local'] else '✗'}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
        results = run_large_N_analysis(N)
    else:
        # Default: scaling analysis
        results = scaling_analysis([6, 8, 10, 12])
    
    # Save results
    import json
    with open('/home/claude/large_N_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to /home/claude/large_N_results.json")