"""
hilbert_substrate_analysis.py

Hilbert Substrate Framework: Numerical Analysis Package
========================================================

This script investigates the emergence of spacetime, particles, and forces
from the Hilbert Substrate Framework (Papers I, II, III by B. Bray).

USAGE:
    python hilbert_substrate_analysis.py --N 12 --topology ring
    python hilbert_substrate_analysis.py --scaling 6,8,10,12,14
    python hilbert_substrate_analysis.py --help

REQUIREMENTS:
    numpy, scipy (standard scientific Python stack)
    
    Install: pip install numpy scipy

WHAT THIS COMPUTES:
    1. Ground state and low-energy spectrum (Lanczos for large N)
    2. Correlation-based metric structure
    3. Triangle inequality tests (is it a valid metric?)
    4. Effective dimensionality (MDS analysis)
    5. Interaction potential V(d) between excitations
    6. Test of force locality (is V(d≥2) = 0?)

KEY FINDINGS FROM PAPER III:
    - Forces are EXACTLY LOCAL: V(d≥2) = 0 for all N tested
    - This is the emergence of spatial locality from accessibility constraints
    - The accessibility ratio grows exponentially with N

Author: Based on B. Bray's Hilbert Substrate Framework
"""

import argparse
import numpy as np
from scipy.sparse import csr_matrix, kron as sparse_kron
from scipy.sparse.linalg import eigsh
import time
import json
from typing import List, Tuple, Dict, Optional

# =============================================================================
# SPARSE MATRIX INFRASTRUCTURE
# =============================================================================

def sparse_pauli() -> Tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """Return sparse Pauli matrices I, X, Y, Z."""
    I = csr_matrix(np.array([[1, 0], [0, 1]], dtype=complex))
    X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    Y = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
    Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    return I, X, Y, Z


def sparse_kron_n(ops: List[csr_matrix]) -> csr_matrix:
    """Sparse Kronecker product of a list of matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = sparse_kron(result, op, format='csr')
    return result


# =============================================================================
# TOPOLOGY / GRAPH STRUCTURES
# =============================================================================

def edges_ring(N: int) -> List[Tuple[int, int]]:
    """1D ring with periodic boundary conditions."""
    return [(i, (i + 1) % N) for i in range(N)]


def edges_chain(N: int) -> List[Tuple[int, int]]:
    """1D chain with open boundary conditions."""
    return [(i, i + 1) for i in range(N - 1)]


def edges_ladder(N: int) -> List[Tuple[int, int]]:
    """Two-leg ladder. N must be even."""
    if N % 2 != 0:
        raise ValueError("Ladder requires even N")
    L = N // 2
    edges = []
    for i in range(L):
        edges.append((i, i + L))  # Rungs
    for i in range(L - 1):
        edges.append((i, i + 1))  # Top leg
        edges.append((i + L, i + L + 1))  # Bottom leg
    return edges


def edges_2d_torus(Lx: int, Ly: int) -> Tuple[List[Tuple[int, int]], int]:
    """2D square lattice with periodic boundaries (torus)."""
    N = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            site = x * Ly + y
            right = x * Ly + ((y + 1) % Ly)
            down = ((x + 1) % Lx) * Ly + y
            edges.append((site, right))
            edges.append((site, down))
    return edges, N


def graph_distance_matrix(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Compute shortest-path distances on the graph (Floyd-Warshall)."""
    D = np.full((N, N), np.inf)
    np.fill_diagonal(D, 0)
    for (i, j) in edges:
        D[i, j] = D[j, i] = 1
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
    return D


# =============================================================================
# HAMILTONIAN CONSTRUCTION
# =============================================================================

def heisenberg_hamiltonian(N: int, edges: List[Tuple[int, int]], 
                           J: float = 1.0) -> csr_matrix:
    """
    Build sparse Heisenberg XXX Hamiltonian.
    
    H = J * Σ_{<ij>} (X_i X_j + Y_i Y_j + Z_i Z_j)
    
    Memory: O(N * 2^N) - sparse representation
    """
    I, X, Y, Z = sparse_pauli()
    dim = 2 ** N
    H = csr_matrix((dim, dim), dtype=complex)
    
    for (i, j) in edges:
        for pauli in [X, Y, Z]:
            ops = [I] * N
            ops[i] = pauli
            ops[j] = pauli
            H = H + J * sparse_kron_n(ops)
    
    return 0.5 * (H + H.conj().T)


# =============================================================================
# EIGENSOLVERS
# =============================================================================

def get_ground_state(H: csr_matrix, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get lowest k eigenvalues and eigenvectors using Lanczos algorithm.
    Much faster than full diagonalization for large sparse matrices.
    """
    dim = H.shape[0]
    k = min(k, dim - 2)  # eigsh requires k < dim - 1
    
    energies, states = eigsh(H, k=k, which='SA', return_eigenvectors=True)
    idx = np.argsort(energies)
    return energies[idx], states[:, idx]


# =============================================================================
# CORRELATION FUNCTIONS
# =============================================================================

def compute_correlations(N: int, H: csr_matrix, 
                         ground_state: np.ndarray) -> np.ndarray:
    """
    Compute spin-spin correlation matrix ⟨σ_i · σ_j⟩.
    
    C[i,j] = (1/3) * Σ_α |⟨σ^α_i σ^α_j⟩|
    """
    I, X, Y, Z = sparse_pauli()
    C = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i, N):
            if i == j:
                C[i, j] = 1.0
                continue
            
            corr = 0.0
            for pauli in [X, Y, Z]:
                ops = [I] * N
                ops[i] = pauli
                ops[j] = pauli
                O = sparse_kron_n(ops)
                corr += np.abs(ground_state.conj() @ (O @ ground_state))
            
            C[i, j] = corr / 3.0
            C[j, i] = C[i, j]
    
    return C


def correlation_to_distance(C: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix: D = 1/C."""
    D = np.zeros_like(C)
    mask = (C > 1e-10) & (np.eye(C.shape[0]) == 0)
    D[mask] = 1.0 / C[mask]
    D = D / np.max(D) if np.max(D) > 0 else D
    return D


# =============================================================================
# METRIC ANALYSIS
# =============================================================================

def check_triangle_inequality(D: np.ndarray) -> Tuple[int, int, float]:
    """
    Check if distance matrix satisfies triangle inequality.
    Returns (violations, total_checks, max_violation).
    """
    N = D.shape[0]
    violations = 0
    total = 0
    max_viol = 0.0
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if len({i, j, k}) == 3:
                    total += 1
                    viol = D[i, k] - D[i, j] - D[j, k]
                    if viol > 1e-10:
                        violations += 1
                        max_viol = max(max_viol, viol)
    
    return violations, total, max_viol


def estimate_dimension_mds(D: np.ndarray) -> Dict:
    """
    Estimate effective dimension using MDS (multidimensional scaling).
    
    This embeds the distance matrix into Euclidean space and counts
    how many dimensions are needed.
    """
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    B = -0.5 * H @ (D ** 2) @ H  # Double-centered squared distances
    
    eigenvalues = np.linalg.eigvalsh(B)[::-1]
    positive = eigenvalues[eigenvalues > 1e-10]
    
    if len(positive) == 0:
        return {'dimension': 0, 'eigenvalues': eigenvalues.tolist()}
    
    # Participation ratio
    normalized = positive / np.sum(positive)
    pr = 1.0 / np.sum(normalized ** 2)
    
    # Cumulative variance
    cumvar = np.cumsum(positive) / np.sum(positive)
    dim_90 = int(np.searchsorted(cumvar, 0.9) + 1)
    dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    
    return {
        'participation_ratio': float(pr),
        'dim_90_variance': dim_90,
        'dim_95_variance': dim_95,
        'top_eigenvalues': positive[:5].tolist()
    }


# =============================================================================
# FORCE / INTERACTION ANALYSIS
# =============================================================================

def compute_interaction_potential(N: int, H: csr_matrix, 
                                   ground_state: np.ndarray,
                                   graph_D: np.ndarray) -> Dict[int, float]:
    """
    Compute interaction potential V(d) between two excitations.
    
    V(d) = E(i,j) - E(i) - E(j) + E_ground
    
    where E(i,j) is energy with excitations at sites i,j separated by
    graph distance d.
    
    THIS IS THE KEY QUANTITY: if V(d≥2) = 0, forces are local!
    """
    I, X, Y, Z = sparse_pauli()
    E_ground = float(np.real(ground_state.conj() @ (H @ ground_state)))
    
    # Single excitation energies
    single_E = []
    for site in range(N):
        ops = [I] * N
        ops[site] = X
        exc_op = sparse_kron_n(ops)
        psi = exc_op @ ground_state
        psi = psi / np.linalg.norm(psi)
        E = float(np.real(psi.conj() @ (H @ psi)))
        single_E.append(E)
    
    # Two-excitation energies grouped by graph distance
    V_by_d = {}
    
    for i in range(N):
        for j in range(i + 1, N):
            d = int(graph_D[i, j])
            
            ops1 = [I] * N
            ops1[i] = X
            ops2 = [I] * N
            ops2[j] = X
            
            psi = sparse_kron_n(ops2) @ (sparse_kron_n(ops1) @ ground_state)
            psi = psi / np.linalg.norm(psi)
            E_ij = float(np.real(psi.conj() @ (H @ psi)))
            
            V = E_ij - single_E[i] - single_E[j] + E_ground
            V_by_d.setdefault(d, []).append(V)
    
    # Average by distance
    return {d: float(np.mean(vs)) for d, vs in V_by_d.items()}


def check_force_locality(V_d: Dict[int, float], threshold: float = 0.1) -> bool:
    """
    Check if forces are local: V(d≥2) should be ~0.
    
    Returns True if forces are local.
    """
    V1 = abs(V_d.get(1, 0))
    V_rest = sum(abs(V_d.get(d, 0)) for d in V_d if d > 1)
    
    if V1 < 1e-10:
        return True  # No nearest-neighbor interaction either
    
    return V_rest < threshold * V1


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_analysis(N: int, topology: str = 'ring', verbose: bool = True) -> Dict:
    """
    Run complete Hilbert Substrate analysis for N qubits.
    
    Parameters:
        N: Number of qubits
        topology: 'ring', 'chain', or 'ladder'
        verbose: Print progress
    
    Returns:
        Dictionary with all results
    """
    results = {
        'N': N,
        'dim': 2**N,
        'topology': topology,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"HILBERT SUBSTRATE ANALYSIS: N = {N}, topology = {topology}")
        print(f"Hilbert space dimension: {2**N:,}")
        print(f"{'='*70}")
    
    t_start = time.time()
    
    # Build system
    if verbose:
        print(f"\n[1/5] Building Hamiltonian...")
    
    if topology == 'ring':
        edges = edges_ring(N)
    elif topology == 'chain':
        edges = edges_chain(N)
    elif topology == 'ladder':
        edges = edges_ladder(N)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    H = heisenberg_hamiltonian(N, edges)
    graph_D = graph_distance_matrix(N, edges)
    
    if verbose:
        print(f"    Hamiltonian: {H.shape[0]}×{H.shape[1]}, {H.nnz:,} nonzeros")
        print(f"    Sparsity: {100 * H.nnz / H.shape[0]**2:.4f}%")
    
    results['num_edges'] = len(edges)
    results['nnz'] = H.nnz
    
    # Ground state
    if verbose:
        print(f"\n[2/5] Finding ground state (Lanczos)...")
    
    t1 = time.time()
    energies, states = get_ground_state(H, k=min(10, 2**N - 2))
    ground = states[:, 0]
    
    results['ground_energy'] = float(energies[0])
    results['first_gap'] = float(energies[1] - energies[0]) if len(energies) > 1 else 0
    results['spectrum'] = energies.tolist()
    
    if verbose:
        print(f"    Time: {time.time() - t1:.2f}s")
        print(f"    Ground state energy: {energies[0]:.6f}")
        print(f"    First excitation gap: {results['first_gap']:.6f}")
    
    # Correlations and metric
    if verbose:
        print(f"\n[3/5] Computing correlations and metric...")
    
    t1 = time.time()
    C = compute_correlations(N, H, ground)
    D = correlation_to_distance(C)
    
    violations, total, max_viol = check_triangle_inequality(D)
    dim_data = estimate_dimension_mds(D)
    
    results['triangle_violations'] = violations
    results['triangle_total'] = total
    results['triangle_fraction'] = violations / total if total > 0 else 0
    results['max_triangle_violation'] = max_viol
    results['effective_dimension'] = dim_data
    
    if verbose:
        print(f"    Time: {time.time() - t1:.2f}s")
        print(f"    Triangle violations: {violations}/{total} ({100*violations/total:.1f}%)")
        print(f"    MDS dimension (participation ratio): {dim_data['participation_ratio']:.2f}")
    
    # Force structure - THE KEY RESULT
    if verbose:
        print(f"\n[4/5] Computing force structure...")
    
    t1 = time.time()
    V_d = compute_interaction_potential(N, H, ground, graph_D)
    is_local = check_force_locality(V_d)
    
    results['V_vs_d'] = {int(k): float(v) for k, v in V_d.items()}
    results['forces_are_local'] = is_local
    
    if verbose:
        print(f"    Time: {time.time() - t1:.2f}s")
        print(f"\n    Interaction potential V(d):")
        for d in sorted(V_d.keys()):
            V = V_d[d]
            bar = "█" * int(abs(V) * 2) if abs(V) > 0.01 else ""
            print(f"      d = {d}: V = {V:+.6f}  {bar}")
        
        print(f"\n    FORCES ARE {'LOCAL ✓' if is_local else 'NON-LOCAL ✗'}")
        if is_local:
            print(f"    → V(d≥2) ≈ 0: Only nearest neighbors interact!")
    
    # Summary
    results['total_time'] = time.time() - t_start
    
    if verbose:
        print(f"\n[5/5] Analysis complete in {results['total_time']:.2f}s")
    
    return results


def run_scaling(N_values: List[int], topology: str = 'ring') -> Dict:
    """Run analysis across multiple N values."""
    print("\n" + "="*70)
    print("SCALING ANALYSIS")
    print("="*70)
    
    all_results = {'N_values': N_values, 'topology': topology, 'data': []}
    
    for N in N_values:
        try:
            r = run_analysis(N, topology, verbose=True)
            all_results['data'].append(r)
        except MemoryError:
            print(f"\n*** MemoryError at N={N} - stopping ***")
            break
        except Exception as e:
            print(f"\n*** Error at N={N}: {e} ***")
            all_results['data'].append({'N': N, 'error': str(e)})
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n  N  |   Dim   |   Gap   | Tri.Viol | V(d=1)   | V(d=2)   | Local?")
    print("  " + "-"*67)
    
    for r in all_results['data']:
        if 'error' in r:
            print(f"  {r['N']:2d} | ERROR: {r['error']}")
        else:
            V1 = r['V_vs_d'].get(1, r['V_vs_d'].get('1', 0))
            V2 = r['V_vs_d'].get(2, r['V_vs_d'].get('2', 0))
            tri = r['triangle_fraction']
            local = "✓" if r['forces_are_local'] else "✗"
            print(f"  {r['N']:2d} | {r['dim']:7,} | {r['first_gap']:7.4f} | "
                  f"{100*tri:6.1f}%  | {V1:+8.4f} | {V2:+8.4f} | {local}")
    
    return all_results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hilbert Substrate Framework Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hilbert_substrate_analysis.py --N 10
  python hilbert_substrate_analysis.py --N 12 --topology chain
  python hilbert_substrate_analysis.py --scaling 6,8,10,12
  python hilbert_substrate_analysis.py --scaling 6,8,10,12,14,16 --output results.json
        """
    )
    
    parser.add_argument('--N', type=int, default=None,
                        help='Number of qubits for single analysis')
    parser.add_argument('--topology', type=str, default='ring',
                        choices=['ring', 'chain', 'ladder'],
                        help='Interaction topology (default: ring)')
    parser.add_argument('--scaling', type=str, default=None,
                        help='Comma-separated N values for scaling analysis')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    if args.scaling:
        N_values = [int(x.strip()) for x in args.scaling.split(',')]
        results = run_scaling(N_values, args.topology)
    elif args.N:
        results = run_analysis(args.N, args.topology, verbose=not args.quiet)
    else:
        # Default: small scaling analysis
        print("No arguments provided. Running default scaling analysis (N=6,8,10,12).")
        print("Use --help for options.\n")
        results = run_scaling([6, 8, 10, 12], args.topology)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()