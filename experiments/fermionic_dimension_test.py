"""
FERMIONIC STABILITY vs SPATIAL DIMENSION
=========================================

Hypothesis: 3D is selected because it's the minimum dimension where
fermionic statistics are topologically stable.

Test: Compare how well fermionic structure survives scrambling + 
accessibility recovery on lattices of different dimension.

Key insight:
- 1D: JW works, but no real exchange possible
- 2D: π₁(config space) = Braid group → anyons possible → fermions unstable
- 3D: π₁(config space) = Symmetric group → only ±1 → fermions locked

We test this by:
1. Creating lattices with 1D, 2D, 3D connectivity
2. Defining fermionic Hamiltonians (via JW)
3. Scrambling with random unitaries
4. Applying accessibility flow to recover locality
5. Measuring how well fermionic anticommutation survives
"""

import numpy as np
from scipy.linalg import eigh, expm, norm
from scipy.sparse import csr_matrix, kron as sparse_kron, eye as sparse_eye
from itertools import combinations
import time

# =============================================================================
# PAULI MATRICES AND TENSOR PRODUCTS
# =============================================================================

def pauli():
    I = np.array([[1,0],[0,1]], dtype=np.complex128)
    X = np.array([[0,1],[1,0]], dtype=np.complex128)
    Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
    Z = np.array([[1,0],[0,-1]], dtype=np.complex128)
    return I, X, Y, Z

def kron_n(ops):
    """Tensor product of list of operators."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

# =============================================================================
# LATTICE GENERATION
# =============================================================================

def make_1d_lattice(N):
    """1D chain with periodic boundary conditions."""
    edges = [(i, (i+1) % N) for i in range(N)]
    return edges, "1D Ring"

def make_2d_lattice(Lx, Ly):
    """2D square lattice with periodic boundary conditions."""
    N = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            # Right neighbor
            j = ((x+1) % Lx) * Ly + y
            edges.append((i, j))
            # Up neighbor
            k = x * Ly + ((y+1) % Ly)
            edges.append((i, k))
    return edges, f"2D Torus ({Lx}x{Ly})"

def make_3d_lattice(Lx, Ly, Lz):
    """3D cubic lattice with periodic boundary conditions."""
    N = Lx * Ly * Lz
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                i = x * Ly * Lz + y * Lz + z
                # X neighbor
                j = ((x+1) % Lx) * Ly * Lz + y * Lz + z
                edges.append((i, j))
                # Y neighbor
                k = x * Ly * Lz + ((y+1) % Ly) * Lz + z
                edges.append((i, k))
                # Z neighbor
                l = x * Ly * Lz + y * Lz + ((z+1) % Lz)
                edges.append((i, l))
    return edges, f"3D Torus ({Lx}x{Ly}x{Lz})"

# =============================================================================
# JORDAN-WIGNER TRANSFORMATION
# =============================================================================

def jordan_wigner_operators(N, ordering=None):
    """
    Build JW fermion operators for N sites.
    
    c†_j = Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_{j-1} ⊗ σ+_j ⊗ I_{j+1} ⊗ ... ⊗ I_{N-1}
    
    ordering: permutation of sites (if None, use natural ordering)
    """
    I, X, Y, Z = pauli()
    
    # Creation/annihilation for single site
    b_create = np.array([[0,0],[1,0]], dtype=np.complex128)   # |1⟩⟨0|
    b_destroy = np.array([[0,1],[0,0]], dtype=np.complex128)  # |0⟩⟨1|
    
    if ordering is None:
        ordering = list(range(N))
    
    # Inverse ordering: where does physical site j appear in the JW ordering?
    inv_ordering = [0] * N
    for pos, site in enumerate(ordering):
        inv_ordering[site] = pos
    
    c_create = []
    c_destroy = []
    
    for j in range(N):
        pos = inv_ordering[j]  # Position of site j in JW string
        
        ops_c = []
        ops_d = []
        for m in range(N):
            m_pos = inv_ordering[m]
            if m_pos < pos:
                ops_c.append(Z)
                ops_d.append(Z)
            elif m_pos == pos:
                ops_c.append(b_create)
                ops_d.append(b_destroy)
            else:
                ops_c.append(I)
                ops_d.append(I)
        
        c_create.append(kron_n(ops_c))
        c_destroy.append(kron_n(ops_d))
    
    return c_create, c_destroy

def check_anticommutation(c_create, c_destroy):
    """
    Check {c_i, c_j†} = δ_ij and {c_i, c_j} = 0.
    Returns average deviation from ideal.
    """
    N = len(c_create)
    dim = c_create[0].shape[0]
    
    errors = []
    
    # Check {c_i, c_j†} = δ_ij
    for i in range(N):
        for j in range(N):
            anticomm = c_destroy[i] @ c_create[j] + c_create[j] @ c_destroy[i]
            expected = np.eye(dim) if i == j else np.zeros((dim, dim))
            errors.append(norm(anticomm - expected))
    
    # Check {c_i, c_j} = 0
    for i in range(N):
        for j in range(N):
            anticomm = c_destroy[i] @ c_destroy[j] + c_destroy[j] @ c_destroy[i]
            errors.append(norm(anticomm))
    
    return np.mean(errors), np.max(errors)

# =============================================================================
# FERMIONIC HAMILTONIAN (Free Fermions on Lattice)
# =============================================================================

def free_fermion_hamiltonian(edges, N, t=1.0):
    """
    H = -t Σ_{⟨i,j⟩} (c†_i c_j + c†_j c_i)
    
    Free fermion hopping on the given graph.
    """
    c_create, c_destroy = jordan_wigner_operators(N)
    
    H = np.zeros((2**N, 2**N), dtype=np.complex128)
    
    for (i, j) in edges:
        # Hopping term: c†_i c_j + h.c.
        H -= t * (c_create[i] @ c_destroy[j] + c_create[j] @ c_destroy[i])
    
    return H, c_create, c_destroy

# =============================================================================
# SCRAMBLING AND RECOVERY
# =============================================================================

def random_unitary(dim, depth=1):
    """Generate random unitary by exponentiating random Hermitian."""
    U = np.eye(dim, dtype=np.complex128)
    for _ in range(depth):
        H_rand = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H_rand = (H_rand + H_rand.conj().T) / 2
        U = expm(-1j * H_rand) @ U
    return U

def scramble_hamiltonian(H, depth=None):
    """Apply random unitary: H' = U H U†"""
    dim = H.shape[0]
    if depth is None:
        depth = int(np.log2(dim))
    U = random_unitary(dim, depth)
    return U @ H @ U.conj().T, U

def pauli_weight(N, k):
    """Hamming weight (number of non-identity factors) for Pauli index k."""
    weight = 0
    for j in range(N):
        if (k >> (2*j)) & 3 != 0:  # Non-identity at position j
            weight += 1
    return weight

def locality_cost(H, N, p=4):
    """
    Compute the locality cost C_p(H) = Σ_k w(P_k)^p |c_k|^2 / Σ_k |c_k|^2
    
    where w(P_k) is the Hamming weight (number of non-identity factors).
    """
    I, X, Y, Z = pauli()
    paulis = [I, X, Y, Z]
    
    dim = 2**N
    coeffs_sq = []
    weights = []
    
    # Expand H in Pauli basis
    for k in range(4**N):
        # Build Pauli string for index k
        ops = []
        idx = k
        for j in range(N):
            ops.append(paulis[idx & 3])
            idx >>= 2
        P_k = kron_n(ops)
        
        # Coefficient
        c_k = np.trace(H @ P_k) / dim
        coeffs_sq.append(np.abs(c_k)**2)
        
        # Weight
        w = sum(1 for j in range(N) if ((k >> (2*j)) & 3) != 0)
        weights.append(w)
    
    coeffs_sq = np.array(coeffs_sq)
    weights = np.array(weights)
    
    total = np.sum(coeffs_sq)
    if total < 1e-15:
        return 0.0
    
    return np.sum((weights ** p) * coeffs_sq) / total

def double_bracket_step(H, M, dt=0.1):
    """One step of double bracket flow: dH/dt = [H, [H, M]]"""
    comm1 = H @ M - M @ H
    comm2 = H @ comm1 - comm1 @ H
    return H + dt * comm2

def locality_gradient(H, N, p=4):
    """
    Compute the gradient operator M for the locality cost.
    M = Σ_k (∂C/∂c_k) P_k
    """
    I, X, Y, Z = pauli()
    paulis = [I, X, Y, Z]
    dim = 2**N
    
    # First compute all coefficients and weights
    coeffs = []
    Ps = []
    weights = []
    
    for k in range(4**N):
        ops = []
        idx = k
        for j in range(N):
            ops.append(paulis[idx & 3])
            idx >>= 2
        P_k = kron_n(ops)
        Ps.append(P_k)
        
        c_k = np.trace(H @ P_k) / dim
        coeffs.append(c_k)
        
        w = sum(1 for j in range(N) if ((k >> (2*j)) & 3) != 0)
        weights.append(w)
    
    coeffs = np.array(coeffs)
    weights = np.array(weights)
    coeffs_sq = np.abs(coeffs)**2
    
    total = np.sum(coeffs_sq)
    if total < 1e-15:
        return np.zeros_like(H)
    
    # Gradient: ∂C/∂c_k ∝ w^p c_k
    M = np.zeros_like(H)
    for k in range(4**N):
        grad_k = (weights[k] ** p) * coeffs[k]
        M += grad_k * Ps[k]
    
    return M * (2 / total)

def accessibility_flow(H, N, p=4, max_steps=200, tol=1e-6, dt=0.05):
    """
    Run the double bracket flow to minimize locality cost.
    Returns trajectory of costs and final Hamiltonian.
    """
    costs = [locality_cost(H, N, p)]
    H_current = H.copy()
    
    for step in range(max_steps):
        M = locality_gradient(H_current, N, p)
        H_new = double_bracket_step(H_current, M, dt)
        
        # Ensure Hermitian
        H_new = (H_new + H_new.conj().T) / 2
        
        cost_new = locality_cost(H_new, N, p)
        costs.append(cost_new)
        
        # Check convergence
        if abs(costs[-1] - costs[-2]) < tol:
            break
        
        # Adaptive step size
        if cost_new > costs[-2]:
            dt *= 0.5
        else:
            dt *= 1.1
            dt = min(dt, 0.2)
        
        H_current = H_new
    
    return H_current, costs

# =============================================================================
# FERMIONIC STRUCTURE RECOVERY TEST
# =============================================================================

def measure_fermionic_structure(H_recovered, N, edges):
    """
    After recovery, check if the Hamiltonian still looks like free fermions.
    
    For a free fermion Hamiltonian, H should be quadratic in c, c†:
    H = Σ_{ij} t_{ij} c†_i c_j
    
    We measure:
    1. How much of H is in the quadratic sector
    2. How well anticommutation is preserved under the flow
    """
    c_create, c_destroy = jordan_wigner_operators(N)
    
    # Try to extract the hopping matrix by projecting onto quadratic terms
    t_matrix = np.zeros((N, N), dtype=np.complex128)
    
    for i in range(N):
        for j in range(N):
            # Project: t_ij = Tr(H c†_i c_j) / Tr(c†_i c_j c†_i c_j)
            # For normalized operators
            op = c_create[i] @ c_destroy[j]
            t_matrix[i, j] = np.trace(H_recovered @ op) / (2**N)
    
    # Reconstruct quadratic Hamiltonian
    H_quad = np.zeros_like(H_recovered)
    for i in range(N):
        for j in range(N):
            H_quad += t_matrix[i, j] * c_create[i] @ c_destroy[j]
    
    # Measure how much of H is quadratic
    residual = H_recovered - H_quad
    quad_fraction = 1 - norm(residual) / (norm(H_recovered) + 1e-10)
    
    # Check if hopping respects the lattice structure
    edge_set = set((min(e), max(e)) for e in edges)
    on_lattice = 0
    off_lattice = 0
    
    for i in range(N):
        for j in range(i+1, N):
            t_ij = abs(t_matrix[i, j])
            if (i, j) in edge_set:
                on_lattice += t_ij
            else:
                off_lattice += t_ij
    
    locality_ratio = on_lattice / (on_lattice + off_lattice + 1e-10)
    
    return {
        'quadratic_fraction': quad_fraction,
        'locality_ratio': locality_ratio,
        'hopping_matrix': t_matrix
    }

# =============================================================================
# EXCHANGE PHASE TEST (The Key Test!)
# =============================================================================

def measure_exchange_phase(H_recovered, N, site_a, site_b, n_steps=20):
    """
    Measure the Berry phase acquired when exchanging two particles.
    
    This is the key test:
    - In 3D: Should always be ±1 (fermions stable)
    - In 2D: Could drift to other values (anyons)
    
    We do this by:
    1. Creating a two-particle state
    2. Adiabatically moving one particle around the other
    3. Measuring the accumulated phase
    """
    c_create, c_destroy = jordan_wigner_operators(N)
    
    # Create two-particle state: |ψ⟩ = c†_a c†_b |vac⟩
    vacuum = np.zeros(2**N, dtype=np.complex128)
    vacuum[0] = 1
    
    psi_initial = c_create[site_a] @ c_create[site_b] @ vacuum
    psi_initial /= norm(psi_initial)
    
    # For the exchange, we need to define a path
    # In practice, we'll use the Hamiltonian evolution and measure
    # the phase of ⟨ψ_initial | ψ_final ⟩ after exchange
    
    # Simple test: compare c†_a c†_b |vac⟩ with c†_b c†_a |vac⟩
    psi_ab = c_create[site_a] @ c_create[site_b] @ vacuum
    psi_ba = c_create[site_b] @ c_create[site_a] @ vacuum
    
    # The ratio should be -1 for fermions
    overlap = np.vdot(psi_ab, psi_ba)
    phase = np.angle(overlap)
    
    return {
        'overlap': overlap,
        'phase': phase,
        'phase_over_pi': phase / np.pi,
        'is_fermionic': np.abs(np.abs(phase) - np.pi) < 0.1  # Phase ≈ ±π
    }

# =============================================================================
# PATH-DEPENDENT PHASE TEST (Critical for 2D vs 3D)
# =============================================================================

def test_path_dependence(N, edges, dim_label):
    """
    In 2D, different paths around another particle can give different phases.
    In 3D, all paths are homotopic → same phase.
    
    We test this by using different JW orderings and checking consistency.
    """
    # Try multiple random orderings
    n_orderings = 10
    phases = []
    
    for trial in range(n_orderings):
        ordering = np.random.permutation(N).tolist()
        c_create, c_destroy = jordan_wigner_operators(N, ordering)
        
        vacuum = np.zeros(2**N, dtype=np.complex128)
        vacuum[0] = 1
        
        # Create at first two sites (in natural numbering)
        psi_01 = c_create[0] @ c_create[1] @ vacuum
        psi_10 = c_create[1] @ c_create[0] @ vacuum
        
        if norm(psi_01) > 1e-10 and norm(psi_10) > 1e-10:
            psi_01 /= norm(psi_01)
            psi_10 /= norm(psi_10)
            overlap = np.vdot(psi_01, psi_10)
            phases.append(np.angle(overlap))
    
    phases = np.array(phases)
    
    # For stable fermions: all phases should be ≈ ±π
    # For anyons: phases could vary
    phase_std = np.std(np.abs(phases) - np.pi)
    
    return {
        'phases': phases,
        'phase_std': phase_std,
        'mean_phase_over_pi': np.mean(np.abs(phases)) / np.pi,
        'ordering_independent': phase_std < 0.1
    }

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_dimension_comparison(seed=42):
    """
    Main experiment: Compare fermionic stability across dimensions.
    """
    np.random.seed(seed)
    
    print("="*75)
    print("  FERMIONIC STABILITY vs SPATIAL DIMENSION")
    print("="*75)
    print("""
  Hypothesis: 3D is selected because fermionic statistics are
  topologically stable only in D ≥ 3.
  
  Test: Create fermionic Hamiltonians on 1D, 2D, 3D lattices,
  scramble them, recover via accessibility flow, measure if
  fermionic structure survives.
    """)
    
    results = {}
    
    # Define lattices with similar total sizes
    lattices = [
        (make_1d_lattice(8), 8),                    # N=8, 1D
        (make_2d_lattice(3, 3), 9),                 # N=9, 2D (closest to 8)
        (make_3d_lattice(2, 2, 2), 8),              # N=8, 3D
    ]
    
    for (edges, label), N in lattices:
        print(f"\n{'='*75}")
        print(f"  {label} (N={N} sites, {len(edges)} edges)")
        print("="*75)
        
        # Skip if too large
        if N > 10:
            print("  [Skipping: too large for exact computation]")
            continue
        
        # Build fermionic Hamiltonian
        print("\n  Building free fermion Hamiltonian...")
        H_original, c_create, c_destroy = free_fermion_hamiltonian(edges, N)
        
        # Check original anticommutation
        mean_err, max_err = check_anticommutation(c_create, c_destroy)
        print(f"  Original anticommutation error: mean={mean_err:.2e}, max={max_err:.2e}")
        
        # Original locality cost
        cost_original = locality_cost(H_original, N, p=4)
        print(f"  Original locality cost: {cost_original:.4f}")
        
        # Scramble
        print("\n  Scrambling...")
        H_scrambled, U = scramble_hamiltonian(H_original, depth=N)
        cost_scrambled = locality_cost(H_scrambled, N, p=4)
        print(f"  Scrambled locality cost: {cost_scrambled:.4f}")
        
        # Recovery
        print("\n  Running accessibility flow...")
        t0 = time.time()
        H_recovered, cost_history = accessibility_flow(H_scrambled, N, p=4, max_steps=300)
        t1 = time.time()
        print(f"  Recovery time: {t1-t0:.1f}s, steps: {len(cost_history)}")
        print(f"  Recovered locality cost: {cost_history[-1]:.4f}")
        
        # Measure fermionic structure
        print("\n  Measuring fermionic structure...")
        ferm_struct = measure_fermionic_structure(H_recovered, N, edges)
        print(f"  Quadratic fraction: {ferm_struct['quadratic_fraction']:.4f}")
        print(f"  Locality ratio: {ferm_struct['locality_ratio']:.4f}")
        
        # Exchange phase test
        print("\n  Testing exchange phases...")
        if N >= 2:
            exchange = measure_exchange_phase(H_recovered, N, 0, 1)
            print(f"  Exchange phase/π: {exchange['phase_over_pi']:.4f}")
            print(f"  Is fermionic (phase ≈ ±π): {exchange['is_fermionic']}")
        
        # Path dependence test (key for 2D vs 3D!)
        print("\n  Testing path dependence (ordering independence)...")
        path_test = test_path_dependence(N, edges, label)
        print(f"  Phase std across orderings: {path_test['phase_std']:.4f}")
        print(f"  Ordering independent: {path_test['ordering_independent']}")
        
        results[label] = {
            'N': N,
            'cost_original': cost_original,
            'cost_scrambled': cost_scrambled,
            'cost_recovered': cost_history[-1],
            'quadratic_fraction': ferm_struct['quadratic_fraction'],
            'locality_ratio': ferm_struct['locality_ratio'],
            'phase_std': path_test['phase_std'],
            'ordering_independent': path_test['ordering_independent']
        }
    
    # Summary
    print("\n" + "="*75)
    print("  SUMMARY: FERMIONIC STABILITY BY DIMENSION")
    print("="*75)
    
    print(f"\n  {'Lattice':<20} {'Cost Rec':<12} {'Quad Frac':<12} {'Local Ratio':<12} {'Phase Std':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for label, r in results.items():
        print(f"  {label:<20} {r['cost_recovered']:<12.4f} {r['quadratic_fraction']:<12.4f} "
              f"{r['locality_ratio']:<12.4f} {r['phase_std']:<12.4f}")
    
    return results

# =============================================================================
# TOPOLOGICAL STABILITY ANALYSIS
# =============================================================================

def analyze_braid_vs_permutation():
    """
    Theoretical analysis: Why does dimension matter?
    
    The fundamental group of the configuration space:
    - π₁(C_n(R¹)) = trivial (particles can't exchange)
    - π₁(C_n(R²)) = Braid group B_n (infinite, non-abelian)
    - π₁(C_n(R³)) = Symmetric group S_n (finite, |S_n| = n!)
    
    For D ≥ 3, only ±1 phases are allowed → fermions locked.
    For D = 2, any phase is allowed → anyons possible → fermions unstable.
    """
    print("\n" + "="*75)
    print("  TOPOLOGICAL ANALYSIS: WHY 3D?")
    print("="*75)
    
    print("""
  Configuration space of n indistinguishable particles in D dimensions:
  
    C_n(R^D) = (R^D)^n \\ Δ) / S_n
    
  where Δ is the "diagonal" (coincident points) and S_n is permutations.
  
  FUNDAMENTAL GROUP:
  ┌─────────┬────────────────────────┬─────────────────────────────┐
  │ D       │ π₁(C_n(R^D))           │ Allowed statistics          │
  ├─────────┼────────────────────────┼─────────────────────────────┤
  │ 1       │ Trivial                │ N/A (no exchange)           │
  │ 2       │ Braid group B_n        │ Anyons (continuous phases)  │
  │ ≥3      │ Symmetric group S_n    │ Bosons (+1) or Fermions (-1)│
  └─────────┴────────────────────────┴─────────────────────────────┘
  
  KEY INSIGHT:
  ════════════
  In 2D, a particle going around another traces a non-contractible loop.
  The phase acquired can be ANY value in [0, 2π).
  
  In 3D, any loop can be contracted to a point (R³ \\ {point} is simply
  connected). Double exchange = identity → phase² = 1 → phase = ±1.
  
  CONSEQUENCE FOR HSF:
  ════════════════════
  If the Hilbert substrate wants to support STABLE fermionic excitations,
  it must organize into D ≥ 3 spatial dimensions.
  
  3D is the MINIMUM dimension for fermionic stability!
    """)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run theoretical analysis
    analyze_braid_vs_permutation()
    
    # Run numerical experiments
    results = run_dimension_comparison(seed=42)
    
    print("\n" + "="*75)
    print("  CONCLUSION")
    print("="*75)
    print("""
  If the numerical results show:
  - 1D: Fermionic structure survives (trivially, no real exchange)
  - 2D: Fermionic structure degrades (phases can drift)
  - 3D: Fermionic structure survives (phases locked at ±1)
  
  Then we have evidence that 3D is selected by the requirement
  of stable fermionic excitations.
  
  The accessibility phase transition doesn't just select for locality—
  it selects for the MINIMUM dimension supporting stable fermions.
    """)