"""
Task B1 Final: Higher-Order Perturbation for Plaquette Emergence
=================================================================

Key insight from v3: 2nd-order only gives self-energy (same bond twice).
Bond-bond coupling requires going AROUND A LOOP:
  - Triangle plaquette: 3rd order (3 edges)
  - Square plaquette: 4th order (4 edges)

This matches the strong-coupling expansion of lattice gauge theory
where the Wilson plaquette action emerges at order β^n for an n-sided
plaquette.

Method: Compute H_eff to arbitrary order using recursive Schrieffer-Wolff,
then decompose to find plaquette terms.
"""

import numpy as np
from scipy.linalg import expm
from itertools import product as iprod
import matplotlib.pyplot as plt
from bond_hamiltonian_b1 import (EchoLattice, decompose_in_pauli, 
                                  hamming_weight)

np.set_printoptions(precision=6, suppress=True, linewidth=120)


def high_order_perturbation(lattice, coupling=0.5, max_order=6):
    """
    Compute H_eff to high order using the resolvent expansion:
    
    H_eff = P V P + P V G V P + P V G V G V P + ...
    
    where G = Q/(E_0 - H_0) is the resolvent in Q subspace.
    
    Each order n picks up processes where n edges are traversed.
    Bond-bond coupling through a shared site requires the 
    excitation to hop around a loop.
    """
    D = lattice.total_dim
    d_s = lattice.d_sites_total
    d_b = lattice.d_bonds_total
    
    # Site energies (distinct to avoid degeneracy)
    site_gaps = [2.0 + 0.37*i for i in range(lattice.n_sites)]
    
    # H_0 = site field
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    H_0 = np.zeros((D, D), dtype=complex)
    for i in range(lattice.n_sites):
        H_0 += site_gaps[i] * lattice._embed_operator(sz, [i])
    
    # V = coupling Hamiltonian
    V = lattice.build_full_hamiltonian(coupling)
    
    # Reference: all sites |0⟩
    E_0 = sum(site_gaps)
    
    # Projectors
    P = np.zeros((D, D), dtype=complex)
    for b in range(d_b):
        P[b, b] = 1.0
    Q = np.eye(D) - P
    
    # Resolvent in Q subspace
    H_0_diag = np.diag(H_0).real
    G = np.zeros((D, D), dtype=complex)
    for i in range(D):
        site_config = i // d_b
        if site_config != 0:
            denom = E_0 - H_0_diag[i]
            if abs(denom) > 1e-10:
                G[i, i] = 1.0 / denom
    
    # Compute H_eff order by order
    # H_eff^(n) = P V (G V)^{n-1} P
    
    results = {}
    
    # Start: P V
    PV = P @ V
    
    # Build (GV)^k for k = 0, 1, 2, ...
    GV = G @ V
    current = PV  # This is P V (GV)^0 = P V
    
    for order in range(1, max_order + 1):
        # H_eff^(order) = current @ ... @ P
        # At this point, current = P V (GV)^{order-1}
        
        H_eff_order_full = current @ P
        H_eff_order = H_eff_order_full[:d_b, :d_b]
        H_eff_order = (H_eff_order + H_eff_order.conj().T) / 2.0
        
        norm = np.linalg.norm(H_eff_order)
        
        # Decompose
        coeffs = decompose_in_pauli(H_eff_order, lattice.n_bonds)
        
        # Categorize by weight
        by_weight = {}
        total = sum(abs(c)**2 for c in coeffs.values()) if coeffs else 0
        for lab, c in coeffs.items():
            w = hamming_weight(lab)
            by_weight.setdefault(w, []).append((lab, c))
        
        # Check for multi-bond terms
        max_weight = max(by_weight.keys()) if by_weight else 0
        has_plaquette = False
        plaq_terms = []
        
        for w, terms in by_weight.items():
            if w >= 2:
                for lab, c in terms:
                    active = [i for i, ch in enumerate(lab) if ch != 'I']
                    # Check loop
                    site_count = {}
                    for a in active:
                        for s in lattice.edges[a]:
                            site_count[s] = site_count.get(s, 0) + 1
                    is_loop = (all(v == 2 for v in site_count.values()) 
                              and len(site_count) == len(active)
                              and len(active) >= 3)
                    if is_loop:
                        has_plaquette = True
                        plaq_terms.append((lab, c))
        
        results[order] = {
            'H_eff': H_eff_order,
            'norm': norm,
            'coeffs': coeffs,
            'by_weight': by_weight,
            'has_plaquette': has_plaquette,
            'plaq_terms': plaq_terms,
            'max_weight': max_weight,
        }
        
        # Advance: current = current @ GV
        current = current @ GV
    
    return results


def exact_H_eff(lattice, coupling=0.5):
    """
    Compute EXACT effective bond Hamiltonian using the full resolvent.
    
    H_eff_exact = P H P + P V Q (E_0 - Q H Q)^{-1} Q V P
    
    where H = H_0 + V is the full Hamiltonian.
    This sums all orders at once.
    """
    D = lattice.total_dim
    d_s = lattice.d_sites_total
    d_b = lattice.d_bonds_total
    
    site_gaps = [2.0 + 0.37*i for i in range(lattice.n_sites)]
    
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    H_0 = np.zeros((D, D), dtype=complex)
    for i in range(lattice.n_sites):
        H_0 += site_gaps[i] * lattice._embed_operator(sz, [i])
    
    V = lattice.build_full_hamiltonian(coupling)
    H_full = H_0 + V
    
    E_0 = sum(site_gaps)
    
    P = np.zeros((D, D), dtype=complex)
    for b in range(d_b):
        P[b, b] = 1.0
    Q = np.eye(D) - P
    
    # Q H Q
    QHQ = Q @ H_full @ Q
    
    # (E_0 I - QHQ)^{-1} in Q subspace
    # Eigendecompose QHQ
    evals, evecs = np.linalg.eigh(QHQ)
    
    resolvent = np.zeros((D, D), dtype=complex)
    for i in range(D):
        if abs(evals[i]) > 1e-12:  # Q subspace has nonzero eigenvalues
            resolvent += (1.0 / (E_0 - evals[i])) * np.outer(evecs[:, i], evecs[:, i].conj())
    
    # H_eff = P H P + P V Q resolvent Q V P
    PHP = P @ H_full @ P
    PVQ = P @ V @ Q
    QVP = Q @ V @ P
    
    H_eff_exact_full = PHP + PVQ @ resolvent @ QVP
    H_eff_exact = H_eff_exact_full[:d_b, :d_b]
    H_eff_exact = (H_eff_exact + H_eff_exact.conj().T) / 2.0
    
    return H_eff_exact


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)
    
    for graph_name, n_sites, edges, expected_plaq_order in [
        ('Triangle', 3, [(0,1),(1,2),(0,2)], 3),
        ('Square',   4, [(0,1),(1,2),(2,3),(0,3)], 4),
    ]:
        print(f"\n{'='*70}")
        print(f"GRAPH: {graph_name}")
        print(f"Expected plaquette order: {expected_plaq_order}")
        print(f"{'='*70}")
        
        lattice = EchoLattice(n_sites, edges, d_B=2)
        
        # High-order perturbation
        coupling = 0.3
        results = high_order_perturbation(lattice, coupling=coupling, max_order=6)
        
        print(f"\nOrder-by-order H_eff (coupling={coupling}):")
        print(f"{'Order':>6} {'||H||':>12} {'Max weight':>12} {'Plaquette?':>12}")
        print(f"{'-'*48}")
        
        for order in sorted(results.keys()):
            r = results[order]
            plaq = "YES" if r['has_plaquette'] else "no"
            print(f"{order:>6} {r['norm']:>12.6f} {r['max_weight']:>12} {plaq:>12}")
        
        # Show details at the order where plaquette first appears
        print(f"\n--- Detail at each order ---")
        for order in sorted(results.keys()):
            r = results[order]
            if r['norm'] < 1e-12:
                print(f"\n  Order {order}: zero")
                continue
            
            print(f"\n  Order {order} (||H||={r['norm']:.6f}):")
            for w in sorted(r['by_weight'].keys()):
                terms = r['by_weight'][w]
                total = sum(abs(c)**2 for c in r['coeffs'].values())
                pct = 100*sum(abs(c)**2 for _,c in terms)/total if total > 0 else 0
                
                weight_names = {0:'Identity', 1:'1-bond', 2:'2-bond', 
                               3:'3-bond', 4:'4-bond'}
                print(f"    Weight {w} ({weight_names.get(w, f'{w}-body')}): "
                      f"{len(terms)} terms, {pct:.1f}%")
                
                for lab, c in sorted(terms, key=lambda x: -abs(x[1]))[:4]:
                    active = [i for i, ch in enumerate(lab) if ch != 'I']
                    site_count = {}
                    for a in active:
                        for s in lattice.edges[a]:
                            site_count[s] = site_count.get(s, 0) + 1
                    is_loop = (all(v == 2 for v in site_count.values()) 
                              and len(site_count) == len(active)
                              and len(active) >= 3)
                    
                    # Check if 2-body terms are adjacent
                    is_adj = False
                    if len(active) == 2:
                        is_adj = bool(set(lattice.edges[active[0]]) & 
                                     set(lattice.edges[active[1]]))
                    
                    tag = ""
                    if is_loop: tag = " ← LOOP!"
                    elif is_adj: tag = " (adjacent)"
                    
                    print(f"      {lab}: {c.real:+.8f}{tag}")
        
        # Exact H_eff for comparison
        print(f"\n--- EXACT H_eff (all orders summed) ---")
        H_exact = exact_H_eff(lattice, coupling=coupling)
        coeffs_exact = decompose_in_pauli(H_exact, lattice.n_bonds)
        
        total_exact = sum(abs(c)**2 for c in coeffs_exact.values())
        by_w_exact = {}
        for lab, c in coeffs_exact.items():
            w = hamming_weight(lab)
            by_w_exact.setdefault(w, []).append((lab, c))
        
        print(f"  ||H_exact|| = {np.linalg.norm(H_exact):.6f}")
        for w in sorted(by_w_exact.keys()):
            terms = by_w_exact[w]
            pct = 100*sum(abs(c)**2 for _,c in terms)/total_exact if total_exact > 0 else 0
            print(f"  Weight {w} ({pct:.1f}%):")
            for lab, c in sorted(terms, key=lambda x: -abs(x[1]))[:6]:
                active = [i for i, ch in enumerate(lab) if ch != 'I']
                site_count = {}
                for a in active:
                    for s in lattice.edges[a]:
                        site_count[s] = site_count.get(s, 0) + 1
                is_loop = (all(v == 2 for v in site_count.values()) 
                          and len(site_count) == len(active)
                          and len(active) >= 3)
                tag = " ← LOOP!" if is_loop else ""
                
                # Check adjacency for 2-body
                if len(active) == 2:
                    is_adj = bool(set(lattice.edges[active[0]]) & 
                                 set(lattice.edges[active[1]]))
                    if is_adj: tag = " (adjacent bonds)"
                    else: tag = " (non-adjacent!)"
                
                print(f"    {lab}: {c.real:+.8f}{tag}")
    
    # ============================================================
    # FIGURE: Order-by-order emergence
    # ============================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Norm vs order for both graphs
    ax = axes[0]
    for gname, ns, es, po, color, marker in [
        ('Triangle', 3, [(0,1),(1,2),(0,2)], 3, '#e74c3c', 'o'),
        ('Square', 4, [(0,1),(1,2),(2,3),(0,3)], 4, '#3498db', 's'),
    ]:
        lat = EchoLattice(ns, es, d_B=2)
        res = high_order_perturbation(lat, coupling=0.3, max_order=6)
        
        orders = sorted(res.keys())
        norms = [res[o]['norm'] for o in orders]
        has_plaq = [res[o]['has_plaquette'] for o in orders]
        
        ax.semilogy(orders, [max(n, 1e-16) for n in norms], 
                    f'{marker}-', color=color, linewidth=2, markersize=8,
                    label=gname)
        
        # Mark where plaquette first appears
        for o, hp in zip(orders, has_plaq):
            if hp and res[o]['norm'] > 1e-12:
                ax.axvline(x=o, color=color, linestyle=':', alpha=0.5)
                ax.annotate(f'plaquette\nat order {o}',
                           xy=(o, res[o]['norm']), fontsize=9,
                           xytext=(o+0.3, res[o]['norm']*5),
                           arrowprops=dict(arrowstyle='->', color=color))
                break
    
    ax.set_xlabel('Perturbation Order', fontsize=12)
    ax.set_ylabel('||H_eff^(n)||', fontsize=12)
    ax.set_title('Perturbative Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Weight distribution for exact H_eff
    ax = axes[1]
    width = 0.35
    
    for offset, (gname, ns, es, color) in enumerate([
        ('Triangle', 3, [(0,1),(1,2),(0,2)], '#e74c3c'),
        ('Square', 4, [(0,1),(1,2),(2,3),(0,3)], '#3498db'),
    ]):
        lat = EchoLattice(ns, es, d_B=2)
        H_ex = exact_H_eff(lat, coupling=0.3)
        coeffs = decompose_in_pauli(H_ex, lat.n_bonds)
        
        total = sum(abs(c)**2 for c in coeffs.values())
        weights = {}
        for lab, c in coeffs.items():
            w = hamming_weight(lab)
            weights[w] = weights.get(w, 0) + abs(c)**2
        
        ws = sorted(weights.keys())
        fracs = [100*weights[w]/total for w in ws]
        
        ax.bar([w + (offset-0.5)*width for w in ws], fracs, width, 
               color=color, alpha=0.7, label=gname)
    
    ax.set_xlabel('Operator Weight (# active bonds)', fontsize=12)
    ax.set_ylabel('Weight fraction (%)', fontsize=12)
    ax.set_title('Exact H_eff: Weight Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Physical mechanism
    ax = axes[2]
    ax.axis('off')
    
    mechanism = """
    PLAQUETTE EMERGENCE MECHANISM
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Virtual site excitation goes
    AROUND A LOOP to return to
    the ground state sector:

    Order 1: bond → site flip
             (stays in Q subspace)
    
    Order 2: same-bond self-energy  
             (no bond-bond coupling)

    Order 3: triangle plaquette!
             site₀→site₁→site₂→site₀
             bonds pick up σ_α factors

    Order 4: square plaquette!
             four virtual hops around □

    This IS the strong-coupling 
    expansion of lattice gauge theory.

    H_plaq = Σ_□ β^n Tr(U₁U₂...Uₙ)
    
    where n = # edges in loop
    and β ∝ coupling / site_gap
    """
    ax.text(0.05, 0.95, mechanism, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Physical Mechanism', fontsize=13, fontweight='bold')
    
    plt.suptitle('Task B1: Plaquette Action from Higher-Order Echo Processes',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('/home/claude/bond_hamiltonian_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to bond_hamiltonian_final.png")