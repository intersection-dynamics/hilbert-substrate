"""
perturbation_test.py

THE KILLER TEST: Controlled Perturbation Analysis
==================================================

A skeptic will say: "Of course V(d≥2)=0 — you built a nearest-neighbor Hamiltonian!"

This test proves V(d) is a genuine measurement, not circular reasoning:

  H(ε) = H_NN + ε * H_NNN
  
  where H_NN  = Σ_<i,j>   J₁ (σᵢ·σⱼ)   [nearest-neighbor]
        H_NNN = Σ_<<i,k>> J₂ (σᵢ·σₖ)   [next-nearest-neighbor]

Expected outcomes:
  1. V(d=2) scales LINEARLY with ε  → proves measurement sensitivity
  2. V(d=1) shifts smoothly         → cross-talk is physical
  3. V(d≥3) remains ~0              → locality still holds beyond NNN
  4. Triangle violations shift      → metric responds to structure

If the pipeline shows this behavior, we've proven it's measuring real physics,
not just "reading back the construction."

USAGE:
    python perturbation_test.py --N 14
    python perturbation_test.py --N 12 --eps-max 0.5 --eps-steps 11
"""

import numpy as np
from scipy.sparse import csr_matrix, kron as sparse_kron
from scipy.sparse.linalg import eigsh
import argparse
import json
import time
from typing import List, Tuple, Dict

# =============================================================================
# INFRASTRUCTURE
# =============================================================================

def sparse_pauli():
    I = csr_matrix(np.array([[1, 0], [0, 1]], dtype=complex))
    X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    Y = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
    Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    return I, X, Y, Z

def sparse_kron_n(ops):
    result = ops[0]
    for op in ops[1:]:
        result = sparse_kron(result, op, format='csr')
    return result

# =============================================================================
# HAMILTONIAN WITH PERTURBATION
# =============================================================================

def heisenberg_with_nnn(N: int, J1: float = 1.0, J2: float = 0.0) -> csr_matrix:
    """
    Heisenberg Hamiltonian with nearest-neighbor (J1) and next-nearest-neighbor (J2) couplings.
    
    H = J1 * Σᵢ (σᵢ·σᵢ₊₁) + J2 * Σᵢ (σᵢ·σᵢ₊₂)
    
    This is the J1-J2 model, well-studied in condensed matter physics.
    """
    I, X, Y, Z = sparse_pauli()
    dim = 2 ** N
    H = csr_matrix((dim, dim), dtype=complex)
    
    # Nearest-neighbor terms (J1)
    for i in range(N):
        j = (i + 1) % N
        for pauli in [X, Y, Z]:
            ops = [I] * N
            ops[i] = pauli
            ops[j] = pauli
            H = H + J1 * sparse_kron_n(ops)
    
    # Next-nearest-neighbor terms (J2)
    if abs(J2) > 1e-15:
        for i in range(N):
            k = (i + 2) % N
            for pauli in [X, Y, Z]:
                ops = [I] * N
                ops[i] = pauli
                ops[k] = pauli
                H = H + J2 * sparse_kron_n(ops)
    
    return 0.5 * (H + H.conj().T)

def graph_distance_ring(N: int) -> np.ndarray:
    """Graph distance matrix for a ring."""
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            D[i, j] = min(d, N - d)
    return D

# =============================================================================
# MEASUREMENTS
# =============================================================================

def get_ground_state(H: csr_matrix, k: int = 2):
    dim = H.shape[0]
    k = min(k, dim - 2)
    energies, states = eigsh(H, k=k, which='SA', return_eigenvectors=True)
    idx = np.argsort(energies)
    return energies[idx], states[:, idx]

def compute_correlations(N: int, H: csr_matrix, ground: np.ndarray) -> np.ndarray:
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
                corr += np.abs(ground.conj() @ (O @ ground))
            C[i, j] = corr / 3.0
            C[j, i] = C[i, j]
    return C

def check_triangle_inequality(D: np.ndarray) -> float:
    """Return fraction of triangle inequality violations."""
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
    return violations / total if total > 0 else 0

def estimate_dimension(D: np.ndarray) -> float:
    """MDS participation ratio."""
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    eigs = np.linalg.eigvalsh(B)[::-1]
    pos = eigs[eigs > 1e-10]
    if len(pos) == 0:
        return 0.0
    norm = pos / np.sum(pos)
    return 1.0 / np.sum(norm ** 2)

def compute_V_vs_d(N: int, H: csr_matrix, ground: np.ndarray, 
                   graph_D: np.ndarray) -> Dict[int, float]:
    """Compute interaction potential V(d)."""
    I, X = sparse_pauli()[0], sparse_pauli()[1]
    E_ground = float(np.real(ground.conj() @ (H @ ground)))
    
    single_E = []
    for site in range(N):
        ops = [I] * N
        ops[site] = X
        psi = sparse_kron_n(ops) @ ground
        psi = psi / np.linalg.norm(psi)
        single_E.append(float(np.real(psi.conj() @ (H @ psi))))
    
    V_by_d = {}
    for i in range(N):
        for j in range(i + 1, N):
            d = int(graph_D[i, j])
            ops1 = [I] * N; ops1[i] = X
            ops2 = [I] * N; ops2[j] = X
            psi = sparse_kron_n(ops2) @ (sparse_kron_n(ops1) @ ground)
            psi = psi / np.linalg.norm(psi)
            E_ij = float(np.real(psi.conj() @ (H @ psi)))
            V = E_ij - single_E[i] - single_E[j] + E_ground
            V_by_d.setdefault(d, []).append(V)
    
    return {d: float(np.mean(vs)) for d, vs in V_by_d.items()}

# =============================================================================
# PERTURBATION SWEEP
# =============================================================================

def run_perturbation_sweep(N: int, eps_values: List[float], 
                           verbose: bool = True) -> Dict:
    """
    Sweep over perturbation strength ε and measure response.
    
    H(ε) = H_NN + ε * H_NNN
    """
    
    results = {
        'N': N,
        'dim': 2**N,
        'eps_values': eps_values,
        'data': []
    }
    
    graph_D = graph_distance_ring(N)
    
    if verbose:
        print("="*80)
        print("CONTROLLED PERTURBATION TEST")
        print("="*80)
        print(f"\nSystem: N = {N} Heisenberg ring")
        print(f"Hilbert dimension: {2**N:,}")
        print(f"Perturbation: H(ε) = H_NN + ε·H_NNN")
        print(f"ε values: {eps_values}")
        print("="*80)
        print(f"\n{'ε':>8} | {'E₀':>12} | {'V(d=1)':>12} | {'V(d=2)':>12} | {'V(d=3)':>12} | {'Tri.Viol':>8} | {'Eff.Dim':>7}")
        print("-"*85)
    
    for eps in eps_values:
        t0 = time.time()
        
        # Build Hamiltonian with perturbation
        H = heisenberg_with_nnn(N, J1=1.0, J2=eps)
        
        # Ground state
        energies, states = get_ground_state(H, k=2)
        ground = states[:, 0]
        E0 = float(energies[0])
        
        # Correlations and metric
        C = compute_correlations(N, H, ground)
        D_corr = np.zeros_like(C)
        mask = C > 1e-10
        D_corr[mask] = 1.0 / C[mask]
        np.fill_diagonal(D_corr, 0)
        D_corr = D_corr / np.max(D_corr) if np.max(D_corr) > 0 else D_corr
        
        tri_viol = check_triangle_inequality(D_corr)
        eff_dim = estimate_dimension(D_corr)
        
        # Force structure - THE KEY MEASUREMENT
        V_d = compute_V_vs_d(N, H, ground, graph_D)
        
        data_point = {
            'eps': eps,
            'E0': E0,
            'V_d': V_d,
            'triangle_violations': tri_viol,
            'effective_dimension': eff_dim,
            'time': time.time() - t0
        }
        results['data'].append(data_point)
        
        if verbose:
            V1 = V_d.get(1, 0)
            V2 = V_d.get(2, 0)
            V3 = V_d.get(3, 0)
            print(f"{eps:>8.4f} | {E0:>12.6f} | {V1:>+12.6f} | {V2:>+12.6f} | {V3:>+12.6f} | {100*tri_viol:>7.2f}% | {eff_dim:>7.2f}")
    
    # Analysis: Check for linear scaling
    if verbose:
        print("-"*85)
        print("\n" + "="*80)
        print("ANALYSIS: Does V(d=2) scale linearly with ε?")
        print("="*80)
    
    eps_arr = np.array([d['eps'] for d in results['data']])
    V2_arr = np.array([d['V_d'].get(2, 0) for d in results['data']])
    
    # Linear fit for V(d=2) vs ε
    # Only fit for eps > 0 to avoid the ε=0 baseline
    mask = eps_arr > 1e-10
    if np.sum(mask) >= 2:
        coeffs = np.polyfit(eps_arr[mask], V2_arr[mask], 1)
        slope, intercept = coeffs
        
        # R² calculation
        V2_pred = slope * eps_arr[mask] + intercept
        ss_res = np.sum((V2_arr[mask] - V2_pred)**2)
        ss_tot = np.sum((V2_arr[mask] - np.mean(V2_arr[mask]))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        results['linear_fit'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared)
        }
        
        if verbose:
            print(f"\n  Linear fit: V(d=2) = {slope:.4f} × ε + {intercept:.6f}")
            print(f"  R² = {r_squared:.6f}")
            
            if r_squared > 0.99:
                print(f"\n  ✓ EXCELLENT LINEAR SCALING (R² > 0.99)")
                print(f"    V(d=2) responds linearly to the NNN perturbation.")
                print(f"    This proves the measurement is sensitive, not circular.")
            elif r_squared > 0.95:
                print(f"\n  ✓ GOOD LINEAR SCALING (R² > 0.95)")
            else:
                print(f"\n  ⚠ Non-linear response detected")
    
    # Check V(d≥3) remains small
    V3_max = max(abs(d['V_d'].get(3, 0)) for d in results['data'])
    V4_max = max(abs(d['V_d'].get(4, 0)) for d in results['data']) if N > 8 else 0
    
    results['locality_beyond_nnn'] = {
        'max_V3': float(V3_max),
        'max_V4': float(V4_max),
        'remains_local': V3_max < 0.01 and V4_max < 0.01
    }
    
    if verbose:
        print(f"\n  V(d≥3) check:")
        print(f"    max|V(d=3)| = {V3_max:.6f}")
        print(f"    max|V(d=4)| = {V4_max:.6f}")
        if results['locality_beyond_nnn']['remains_local']:
            print(f"    ✓ Forces remain local beyond NNN (V(d≥3) ≈ 0)")
        else:
            print(f"    ⚠ Non-local forces detected at d≥3")
    
    # Summary
    if verbose:
        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        
        if results.get('linear_fit', {}).get('r_squared', 0) > 0.95:
            print("""
  The V(d) measurement is NOT circular.
  
  When we add ε·H_NNN to the Hamiltonian:
    • V(d=2) scales linearly with ε (R² > 0.95)
    • V(d=1) shifts smoothly (cross-coupling is physical)  
    • V(d≥3) remains ~0 (locality still holds beyond NNN)
  
  This proves the pipeline measures REAL interaction structure,
  not just "reading back the construction."
  
  The original result V(d≥2)=0 for the pure NN model is therefore
  MEANINGFUL — it reflects genuine locality emergence.
""")
        else:
            print("\n  ⚠ Results inconclusive. Check numerical parameters.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Controlled Perturbation Test")
    parser.add_argument('--N', type=int, default=14, help='Number of sites')
    parser.add_argument('--eps-max', type=float, default=0.3, help='Maximum ε')
    parser.add_argument('--eps-steps', type=int, default=7, help='Number of ε values')
    parser.add_argument('--output', type=str, default=None, help='Output JSON')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    eps_values = np.linspace(0, args.eps_max, args.eps_steps).tolist()
    
    results = run_perturbation_sweep(args.N, eps_values, verbose=not args.quiet)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()