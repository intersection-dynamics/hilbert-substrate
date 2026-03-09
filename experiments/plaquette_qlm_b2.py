"""
Task B2 (Redesigned): Quantum Link Model Verification
======================================================

WHY THE FIRST ATTEMPT FAILED:
For d_B=2, each bond is a spin-1/2. This is the MINIMUM representation
of SU(2). The classical Wilson action E = -β Re Tr(U_□) is only valid 
in the large-spin (continuum) limit. For spin-1/2 quantum links, we need
to compare with the QUANTUM LINK MODEL (QLM).

WHAT TO TEST INSTEAD:
1. GAUSS'S LAW: Does H_plaq commute with gauge generators at each site?
   G_a^(v) = Σ_{bonds at v} (±) σ_a^(bond) / 2
   If [G_a, H_plaq] = 0 for all a, v → gauge invariance confirmed

2. PLAQUETTE OPERATOR STRUCTURE: Compare our plaquette terms with the
   standard quantum link model plaquette operator:
   B_□ = S⁺₁S⁺₂S⁻₃S⁻₄ + h.c.  (for a square)
   B_△ = S⁺₁S⁺₂S⁻₃ + h.c.     (for a triangle)
   where S⁺ = (σ_x + iσ_y)/2

3. SPECTRUM: Compare eigenvalues of H_plaq with known QLM results

4. GAUGE TRANSFORMATION TEST: Apply local gauge rotations on bonds
   at a vertex and check H_plaq is invariant
"""

import numpy as np
from scipy.linalg import expm
from itertools import product as iprod
import matplotlib.pyplot as plt
from bond_hamiltonian_b1 import EchoLattice, decompose_in_pauli, hamming_weight
from bond_hamiltonian_final import exact_H_eff, high_order_perturbation

np.set_printoptions(precision=8, suppress=True, linewidth=120)

# Pauli matrices
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sp = (sx + 1j*sy) / 2  # S+ = |0><1| 
sm = (sx - 1j*sy) / 2  # S- = |1><0|


# ============================================================
# Build the standard QLM plaquette operator for comparison
# ============================================================

def qlm_plaquette_operator(n_bonds, orientations=None):
    """
    Build the quantum link model plaquette operator.
    
    For a loop of n bonds with orientations (±1):
    B_□ = Π_k S^{η_k}_k + h.c.
    
    where S^{+1} = S⁺ = (σ_x + iσ_y)/2 (forward link)
          S^{-1} = S⁻ = (σ_x - iσ_y)/2 (backward link)
    
    Standard convention: alternate forward/backward around loop.
    For triangle: +++, for square: ++-- 
    """
    if orientations is None:
        # Default: all forward for triangle, alternating for square
        if n_bonds == 3:
            orientations = [+1, +1, -1]  # Triangle: 0→1, 1→2, 2→0 (last reversed)
        elif n_bonds == 4:
            orientations = [+1, +1, -1, -1]  # Square: forward on 2, backward on 2
    
    d = 2**n_bonds
    
    # Build the product S^{η_1} ⊗ S^{η_2} ⊗ ... ⊗ S^{η_n}
    ops = {+1: sp, -1: sm}
    
    product = ops[orientations[0]].copy()
    for k in range(1, n_bonds):
        product = np.kron(product, ops[orientations[k]])
    
    # B = product + product†
    B = product + product.conj().T
    
    return B


def decompose_and_compare(H_our, B_qlm, n_bonds, label=""):
    """
    Decompose both operators in the Pauli basis and compare structure.
    """
    coeffs_our = decompose_in_pauli(H_our, n_bonds)
    coeffs_qlm = decompose_in_pauli(B_qlm, n_bonds)
    
    # Keep only n-body terms from our H
    plaq_our = {k: v for k, v in coeffs_our.items() if hamming_weight(k) == n_bonds}
    plaq_qlm = {k: v for k, v in coeffs_qlm.items() if hamming_weight(k) == n_bonds}
    
    print(f"\n  {label}")
    print(f"  {'Pauli term':<12} {'Our H_plaq':>14} {'QLM B_□':>14} {'Ratio':>12}")
    print(f"  {'-'*54}")
    
    all_keys = sorted(set(plaq_our.keys()) | set(plaq_qlm.keys()))
    
    our_vec = []
    qlm_vec = []
    
    for key in all_keys:
        c_our = plaq_our.get(key, 0)
        c_qlm = plaq_qlm.get(key, 0)
        
        if abs(c_our) > 1e-12 or abs(c_qlm) > 1e-12:
            ratio = c_our.real / c_qlm.real if abs(c_qlm) > 1e-12 else float('inf')
            print(f"  {key:<12} {c_our.real:>+14.8f} {c_qlm.real:>+14.8f} {ratio:>12.6f}")
        
        our_vec.append(c_our.real)
        qlm_vec.append(c_qlm.real)
    
    our_vec = np.array(our_vec)
    qlm_vec = np.array(qlm_vec)
    
    # Cosine similarity
    n1 = np.linalg.norm(our_vec)
    n2 = np.linalg.norm(qlm_vec)
    if n1 > 0 and n2 > 0:
        cos_sim = np.dot(our_vec, qlm_vec) / (n1 * n2)
        print(f"\n  Cosine similarity: {cos_sim:.6f}")
        
        # Best-fit scaling: our ≈ α * QLM
        alpha = np.dot(our_vec, qlm_vec) / np.dot(qlm_vec, qlm_vec) if np.dot(qlm_vec, qlm_vec) > 0 else 0
        residual = np.linalg.norm(our_vec - alpha * qlm_vec)
        print(f"  Best-fit scaling α: {alpha:.8f}")
        print(f"  Residual ||H_our - α·B_qlm||: {residual:.8f}")
        print(f"  Relative residual: {residual/n1:.6f}")
    
    return cos_sim if n1 > 0 and n2 > 0 else 0, our_vec, qlm_vec


# ============================================================
# TEST 1: Gauss's Law — [G_a, H_plaq] = 0
# ============================================================

def test_gauss_law(lattice, H_plaq):
    """
    Build Gauss law generators G_a^(v) for each vertex v and 
    check [G_a^(v), H_plaq] = 0.
    
    G_a^(v) = Σ_{bonds incident to v} η_{v,bond} σ_a^(bond) / 2
    
    where η = +1 if v is the "source" of the bond, -1 if "target".
    For undirected graphs, we assign arbitrary orientations.
    """
    n_bonds = lattice.n_bonds
    d = 2**n_bonds
    
    # Assign orientations: edge (i,j) goes from i to j
    # Bond b connecting (i,j): η_i = +1, η_j = -1
    
    paulis = [sx, sy, sz]
    pauli_names = ['x', 'y', 'z']
    
    max_commutator = 0
    all_commutators = []
    
    print(f"\n  Gauss's Law Test: [G_a^(v), H_plaq] = 0 ?")
    print(f"  {'Vertex':<8} {'σ_a':<5} {'||[G,H]||':>12} {'Status':>10}")
    print(f"  {'-'*40}")
    
    for v in range(lattice.n_sites):
        for a_idx, (pa, pa_name) in enumerate(zip(paulis, pauli_names)):
            # Build G_a^(v) = Σ_bonds η σ_a^(bond) / 2
            G = np.zeros((d, d), dtype=complex)
            
            for b_idx, (i, j) in enumerate(lattice.edges):
                if v == i or v == j:
                    eta = +1 if v == i else -1
                    
                    # σ_a on bond b_idx, identity on all others
                    op = None
                    for k in range(n_bonds):
                        local = pa if k == b_idx else I2
                        if op is None:
                            op = local
                        else:
                            op = np.kron(op, local)
                    
                    G += eta * op / 2.0
            
            # Commutator
            comm = G @ H_plaq - H_plaq @ G
            comm_norm = np.linalg.norm(comm)
            max_commutator = max(max_commutator, comm_norm)
            all_commutators.append(comm_norm)
            
            status = "✓" if comm_norm < 1e-10 else "✗" if comm_norm > 0.01 else "~"
            print(f"  v={v:<5} σ_{pa_name:<3} {comm_norm:>12.2e} {status:>10}")
    
    print(f"\n  Max ||[G,H]||: {max_commutator:.2e}")
    
    if max_commutator < 1e-8:
        print(f"  → GAUSS'S LAW SATISFIED: H_plaq is gauge-invariant ✓")
    else:
        print(f"  → Gauss's law violated (may need different orientation convention)")
    
    return max_commutator, all_commutators


# ============================================================
# TEST 2: Gauge transformation invariance
# ============================================================

def test_gauge_invariance(lattice, H_plaq, n_tests=50):
    """
    Apply a local gauge transformation at vertex v:
    V_gauge = Π_{bonds at v} exp(i θ_a σ_a^(bond) / 2)
    
    Check: V H_plaq V† = H_plaq
    
    This is the finite (non-infinitesimal) version of Gauss's law.
    """
    n_bonds = lattice.n_bonds
    d = 2**n_bonds
    
    max_diff = 0
    
    for _ in range(n_tests):
        # Random gauge transformation at a random vertex
        v = np.random.randint(lattice.n_sites)
        theta = np.random.randn(3)  # Random su(2) element
        
        # U_gauge = exp(i θ·σ/2) for the transformation
        U_local = expm(1j * (theta[0]*sx + theta[1]*sy + theta[2]*sz) / 2)
        
        # Build full gauge transformation on bond space
        V = np.eye(d, dtype=complex)
        
        for b_idx, (i, j) in enumerate(lattice.edges):
            if v == i or v == j:
                eta = +1 if v == i else -1
                
                # Apply U_local^η on bond b_idx
                U_bond = U_local if eta == +1 else U_local.conj().T
                
                # Embed in full space
                V_b = None
                for k in range(n_bonds):
                    local = U_bond if k == b_idx else I2
                    if V_b is None:
                        V_b = local
                    else:
                        V_b = np.kron(V_b, local)
                
                V = V_b @ V
        
        # Check invariance
        H_transformed = V @ H_plaq @ V.conj().T
        diff = np.linalg.norm(H_transformed - H_plaq)
        max_diff = max(max_diff, diff)
    
    return max_diff


# ============================================================  
# TEST 3: Try ALL orientation conventions for QLM comparison
# ============================================================

def find_best_qlm_match(H_plaq, n_bonds):
    """
    Try all possible link orientations to find the QLM plaquette
    that best matches our H_plaq.
    """
    best_sim = -1
    best_orient = None
    best_B = None
    
    for orient in iprod([+1, -1], repeat=n_bonds):
        B = qlm_plaquette_operator(n_bonds, list(orient))
        
        # Decompose both
        c_ours = decompose_in_pauli(H_plaq, n_bonds)
        c_qlm = decompose_in_pauli(B, n_bonds)
        
        # Keep only n-body terms
        our_plaq = {k: v.real for k, v in c_ours.items() if hamming_weight(k) == n_bonds}
        qlm_plaq = {k: v.real for k, v in c_qlm.items() if hamming_weight(k) == n_bonds}
        
        all_keys = sorted(set(our_plaq.keys()) | set(qlm_plaq.keys()))
        v1 = np.array([our_plaq.get(k, 0) for k in all_keys])
        v2 = np.array([qlm_plaq.get(k, 0) for k in all_keys])
        
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            sim = abs(np.dot(v1, v2) / (n1 * n2))
            if sim > best_sim:
                best_sim = sim
                best_orient = list(orient)
                best_B = B
    
    return best_sim, best_orient, best_B


# ============================================================
# TEST 4: Spectrum comparison  
# ============================================================

def compare_spectra(H_plaq, B_qlm, label=""):
    """Compare eigenvalue spectra of our plaquette vs QLM."""
    evals_ours = np.sort(np.linalg.eigvalsh(H_plaq))
    evals_qlm = np.sort(np.linalg.eigvalsh(B_qlm))
    
    # Normalize both to have same range
    r_ours = evals_ours[-1] - evals_ours[0]
    r_qlm = evals_qlm[-1] - evals_qlm[0]
    
    if r_ours > 1e-12:
        evals_ours_n = (evals_ours - evals_ours[0]) / r_ours
    else:
        evals_ours_n = np.zeros_like(evals_ours)
    
    if r_qlm > 1e-12:
        evals_qlm_n = (evals_qlm - evals_qlm[0]) / r_qlm
    else:
        evals_qlm_n = np.zeros_like(evals_qlm)
    
    # Degeneracy structure
    def degeneracies(evals, tol=1e-8):
        degs = []
        i = 0
        while i < len(evals):
            j = i
            while j < len(evals) and abs(evals[j] - evals[i]) < tol:
                j += 1
            degs.append(j - i)
            i = j
        return degs
    
    degs_ours = degeneracies(evals_ours)
    degs_qlm = degeneracies(evals_qlm)
    
    print(f"\n  {label}")
    print(f"  Our spectrum degeneracies: {degs_ours}")
    print(f"  QLM spectrum degeneracies: {degs_qlm}")
    print(f"  Match: {'YES' if degs_ours == degs_qlm else 'NO'}")
    
    return evals_ours, evals_qlm


# ============================================================
# TEST 5: Scaling with coupling and d_B
# ============================================================

def plaquette_strength_vs_params(n_sites, edges, n_bonds):
    """
    Study how the plaquette coefficient scales with:
    - coupling strength g
    - bond dimension d_B
    """
    # Coupling sweep (d_B = 2)
    couplings = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    plaq_norms = []
    
    for g in couplings:
        lat = EchoLattice(n_sites, edges, d_B=2)
        H = exact_H_eff(lat, coupling=g)
        coeffs = decompose_in_pauli(H, n_bonds)
        
        # Sum of plaquette term magnitudes
        plaq_strength = sum(abs(c)**2 for lab, c in coeffs.items() 
                          if hamming_weight(lab) == n_bonds)
        plaq_norms.append(np.sqrt(plaq_strength))
    
    plaq_norms = np.array(plaq_norms)
    
    # Fit power law
    mask = plaq_norms > 1e-12
    if np.sum(mask) > 3:
        log_g = np.log(couplings[mask])
        log_p = np.log(plaq_norms[mask])
        fit = np.polyfit(log_g, log_p, 1)
        power = fit[0]
    else:
        power = 0
    
    return couplings, plaq_norms, power


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)
    
    all_results = {}
    
    for graph_name, n_sites, edges, n_loop in [
        ('Triangle', 3, [(0,1),(1,2),(0,2)], 3),
        ('Square',   4, [(0,1),(1,2),(2,3),(0,3)], 4),
    ]:
        print(f"\n{'='*70}")
        print(f"B2: {graph_name} — QUANTUM LINK MODEL COMPARISON")
        print(f"{'='*70}")
        
        coupling = 0.4
        lattice = EchoLattice(n_sites, edges, d_B=2)
        n_bonds = lattice.n_bonds
        
        # Get exact H_eff and extract plaquette part
        H_eff = exact_H_eff(lattice, coupling=coupling)
        
        coeffs = decompose_in_pauli(H_eff, n_bonds)
        
        # Build plaquette-only operator
        d = 2**n_bonds
        H_plaq = np.zeros((d, d), dtype=complex)
        
        pauli_ops = {'I': I2, 'X': sx, 'Y': sy, 'Z': sz}
        plaq_coeffs = {}
        for lab, c in coeffs.items():
            if hamming_weight(lab) == n_bonds:
                op = None
                for ch in lab:
                    local = pauli_ops[ch]
                    if op is None:
                        op = local
                    else:
                        op = np.kron(op, local)
                H_plaq += c * op
                plaq_coeffs[lab] = c
        
        # Make Hermitian
        H_plaq = (H_plaq + H_plaq.conj().T) / 2.0
        
        print(f"\n  ||H_eff|| = {np.linalg.norm(H_eff):.6f}")
        print(f"  ||H_plaq|| = {np.linalg.norm(H_plaq):.6f}")
        print(f"  Plaquette fraction: {np.linalg.norm(H_plaq)/np.linalg.norm(H_eff)*100:.4f}%")
        
        print(f"\n  Plaquette terms:")
        for lab, c in sorted(plaq_coeffs.items(), key=lambda x: -abs(x[1])):
            print(f"    {lab}: {c.real:+.8f}")
        
        # ---- TEST 1: Gauss's Law ----
        print(f"\n{'='*50}")
        print(f"TEST 1: GAUSS'S LAW")
        max_comm, all_comms = test_gauss_law(lattice, H_plaq)
        
        # ---- TEST 2: Finite Gauge Invariance ----
        print(f"\n{'='*50}")
        print(f"TEST 2: FINITE GAUGE TRANSFORMATION INVARIANCE")
        max_diff = test_gauge_invariance(lattice, H_plaq, n_tests=100)
        print(f"  Max ||V H V† - H||: {max_diff:.2e}")
        if max_diff < 1e-8:
            print(f"  → GAUGE INVARIANT ✓")
        else:
            print(f"  → Not gauge invariant under standard convention")
            print(f"    (expected: plaquette may use modified gauge structure)")
        
        # ---- TEST 3: QLM Comparison ----
        print(f"\n{'='*50}")
        print(f"TEST 3: QUANTUM LINK MODEL COMPARISON")
        
        # Standard QLM plaquette
        B_std = qlm_plaquette_operator(n_bonds)
        print(f"\n  Standard QLM (default orientation):")
        sim_std, _, _ = decompose_and_compare(H_plaq, B_std, n_bonds, 
                                                "Standard orientation")
        
        # Find best orientation
        print(f"\n  Searching all {2**n_bonds} orientations...")
        best_sim, best_orient, best_B = find_best_qlm_match(H_plaq, n_bonds)
        print(f"  Best orientation: {best_orient}")
        print(f"  Best cosine similarity: {best_sim:.6f}")
        
        if best_B is not None:
            print(f"\n  Best-match QLM comparison:")
            decompose_and_compare(H_plaq, best_B, n_bonds, 
                                 f"Orientation {best_orient}")
        
        # ---- TEST 4: Spectrum ----
        print(f"\n{'='*50}")
        print(f"TEST 4: SPECTRUM COMPARISON")
        if best_B is not None:
            evals_ours, evals_qlm = compare_spectra(H_plaq, best_B, 
                                                     "Our H_plaq vs best QLM B")
        
        # ---- TEST 5: Coupling Scaling ----
        print(f"\n{'='*50}")
        print(f"TEST 5: COUPLING SCALING")
        gs, pnorms, power = plaquette_strength_vs_params(n_sites, edges, n_bonds)
        print(f"  Power law: ||H_plaq|| ∝ g^{power:.2f}")
        print(f"  Expected (Schrieffer-Wolff, {n_loop} edges): g^{2*n_loop} = g^{2*n_loop}")
        print(f"  (Each virtual hop contributes g^2/Δ)")
        
        all_results[graph_name] = {
            'lattice': lattice,
            'H_plaq': H_plaq,
            'plaq_coeffs': plaq_coeffs,
            'best_B': best_B,
            'best_orient': best_orient,
            'best_sim': best_sim,
            'max_comm': max_comm,
            'gauge_inv_diff': max_diff,
            'couplings': gs,
            'plaq_norms': pnorms,
            'power': power,
            'evals_ours': np.sort(np.linalg.eigvalsh(H_plaq)),
            'evals_qlm': np.sort(np.linalg.eigvalsh(best_B)) if best_B is not None else None,
        }
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    
    for row, (graph_name, n_loop) in enumerate([('Triangle', 3), ('Square', 4)]):
        r = all_results[graph_name]
        
        # Panel 1: Pauli coefficient comparison
        ax = axes[row, 0]
        
        coeffs_ours = decompose_in_pauli(r['H_plaq'], r['lattice'].n_bonds)
        coeffs_qlm = decompose_in_pauli(r['best_B'], r['lattice'].n_bonds)
        
        plaq_ours = {k: v.real for k, v in coeffs_ours.items() if hamming_weight(k) == n_loop}
        plaq_qlm = {k: v.real for k, v in coeffs_qlm.items() if hamming_weight(k) == n_loop}
        
        all_keys = sorted(set(plaq_ours.keys()) | set(plaq_qlm.keys()))
        v1 = [plaq_ours.get(k, 0) for k in all_keys]
        v2 = [plaq_qlm.get(k, 0) for k in all_keys]
        
        # Normalize QLM to match our scale
        alpha = np.dot(v1, v2) / np.dot(v2, v2) if np.dot(v2, v2) > 0 else 0
        v2_scaled = [alpha * x for x in v2]
        
        x = np.arange(len(all_keys))
        width = 0.35
        ax.bar(x - width/2, v1, width, color='#3498db', alpha=0.7, label='Our H_plaq')
        ax.bar(x + width/2, v2_scaled, width, color='#e74c3c', alpha=0.7, 
               label=f'QLM × {alpha:.2e}')
        ax.set_xticks(x)
        ax.set_xticklabels(all_keys, rotation=45, fontsize=7)
        ax.set_ylabel('Coefficient')
        ax.set_title(f'{graph_name}: Pauli Decomposition', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Spectrum
        ax = axes[row, 1]
        evals_o = r['evals_ours']
        evals_q = r['evals_qlm']
        
        if evals_o is not None and np.max(np.abs(evals_o)) > 1e-12:
            evals_on = evals_o / np.max(np.abs(evals_o))
        else:
            evals_on = evals_o
        
        if evals_q is not None and np.max(np.abs(evals_q)) > 1e-12:
            evals_qn = evals_q / np.max(np.abs(evals_q))
        else:
            evals_qn = evals_q
        
        if evals_q is not None:
            ax.plot(range(len(evals_on)), evals_on, 'o-', color='#3498db', 
                   markersize=5, label='Our H_plaq')
            ax.plot(range(len(evals_qn)), evals_qn, 's--', color='#e74c3c', 
                   markersize=5, label='QLM B_□')
        ax.set_xlabel('Level index')
        ax.set_ylabel('Normalized eigenvalue')
        ax.set_title(f'{graph_name}: Spectrum', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Gauss's law commutator
        ax = axes[row, 2]
        
        # Re-run for visualization
        n_bonds = r['lattice'].n_bonds
        d = 2**n_bonds
        paulis_list = [sx, sy, sz]
        comm_data = []
        labels_gl = []
        
        for v in range(r['lattice'].n_sites):
            for a_idx, pa in enumerate(paulis_list):
                G = np.zeros((d, d), dtype=complex)
                for b_idx, (i, j) in enumerate(r['lattice'].edges):
                    if v == i or v == j:
                        eta = +1 if v == i else -1
                        op = None
                        for k in range(n_bonds):
                            local = pa if k == b_idx else I2
                            op = local if op is None else np.kron(op, local)
                        G += eta * op / 2.0
                
                comm = G @ r['H_plaq'] - r['H_plaq'] @ G
                comm_data.append(np.linalg.norm(comm))
                labels_gl.append(f'v{v},{"xyz"[a_idx]}')
        
        colors_gl = ['#e74c3c' if c > 1e-6 else '#2ecc71' for c in comm_data]
        ax.bar(range(len(comm_data)), comm_data, color=colors_gl, alpha=0.7)
        ax.set_xticks(range(len(labels_gl)))
        ax.set_xticklabels(labels_gl, rotation=45, fontsize=7)
        ax.set_ylabel('||[G_a^(v), H_plaq]||')
        ax.set_title(f'{graph_name}: Gauss\'s Law', fontweight='bold')
        ax.set_yscale('log')
        ax.axhline(y=1e-10, color='green', linestyle='--', alpha=0.5, label='Machine ε')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Panel 4: Coupling scaling
        ax = axes[row, 3]
        mask = r['plaq_norms'] > 1e-14
        ax.loglog(r['couplings'][mask], r['plaq_norms'][mask], 'o-', 
                 color='#3498db', markersize=6, linewidth=2, label='Data')
        
        # Fit line
        if r['power'] != 0:
            g_fit = np.linspace(r['couplings'][mask][0], r['couplings'][mask][-1], 100)
            log_fit = np.polyfit(np.log(r['couplings'][mask]), 
                                np.log(r['plaq_norms'][mask]), 1)
            ax.loglog(g_fit, np.exp(log_fit[1]) * g_fit**log_fit[0], '--',
                     color='#e74c3c', linewidth=1.5,
                     label=f'Fit: g^{log_fit[0]:.1f}')
            
            # Expected
            ax.loglog(g_fit, np.exp(log_fit[1]) * g_fit**(2*n_loop) / 
                     g_fit[0]**(2*n_loop) * r['plaq_norms'][mask][0], ':',
                     color='gray', linewidth=1.5,
                     label=f'Expected: g^{2*n_loop}')
        
        ax.set_xlabel('Coupling g', fontsize=11)
        ax.set_ylabel('||H_plaq||', fontsize=11)
        ax.set_title(f'{graph_name}: Coupling Scaling', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Task B2: Quantum Link Model Verification of Plaquette Action',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('/home/claude/plaquette_qlm_b2.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print(f"\n{'='*70}")
    print(f"B2 SUMMARY")
    print(f"{'='*70}")
    
    for gname in ['Triangle', 'Square']:
        r = all_results[gname]
        n = len([e for e in r['lattice'].edges])
        print(f"\n{gname} ({n} bonds):")
        print(f"  QLM cosine similarity:  {r['best_sim']:.6f}")
        print(f"  Best orientation:       {r['best_orient']}")
        print(f"  Max Gauss commutator:   {r['max_comm']:.2e}")
        print(f"  Gauge invariance diff:  {r['gauge_inv_diff']:.2e}")
        print(f"  Coupling power law:     g^{r['power']:.1f} (expected g^{2*n})")
    
    print(f"\nFigure saved to plaquette_qlm_b2.png")