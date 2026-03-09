"""
Critical Follow-up: Single Hamiltonian Algebra Generation
==========================================================

The main experiment found su(d_B) in all cases, but it mixed
multiple Hamiltonian types including random ones. Random
Hamiltonians trivially generate the full algebra.

The physics question: Does a SINGLE physically-structured 
transmission Hamiltonian generate the full su(d_B) just by
varying the input site states?

If YES: Site-state diversity (the "matter" degrees of freedom)
is sufficient to generate the full gauge algebra on bonds.
This is physically meaningful — matter drives gauge fields.

If NO: We need multiple interaction types, which is a weaker result.
"""

import numpy as np
from scipy.linalg import expm, logm
from echo_algebra_extraction import (
    transmission_hamiltonian, extract_generators_infinitesimal,
    find_independent_generators, compute_structure_constants,
    identify_algebra, check_su2, PAULI, GELL_MANN,
    extract_kraus_operators
)

np.set_printoptions(precision=6, suppress=True, linewidth=120)


def single_hamiltonian_test(d_B, variant, n_samples=500):
    """Test if a single Hamiltonian type generates full su(d_B)."""
    print(f"\n  Hamiltonian type: {variant}, d_B = {d_B}")
    
    H = transmission_hamiltonian(d_B, variant)
    gens = extract_generators_infinitesimal(d_B, H, n_site_samples=n_samples)
    basis, sv = find_independent_generators(gens, d_B)
    
    n_found = len(basis)
    n_expected = d_B**2 - 1
    
    print(f"  Generators found: {n_found} / {n_expected} expected for su({d_B})")
    
    # Show singular value spectrum to see if there's a clear gap
    if len(sv) > 0:
        print(f"  Top singular values: {sv[:min(n_expected+3, len(sv))]}")
    
    return n_found, n_expected, basis, sv


def site_basis_states_test(d_B, variant='standard'):
    """
    Instead of random site states, use a systematic set:
    computational basis + superposition states.
    
    This tests whether a finite, deterministic set of site 
    configurations generates the full algebra.
    """
    print(f"\n  Systematic site states test, d_B = {d_B}")
    
    H = transmission_hamiltonian(d_B, variant)
    eps = 1e-4
    U = expm(-1j * eps * H)
    
    # Deterministic site states
    site_states = [
        np.array([1, 0], dtype=complex),                    # |0⟩
        np.array([0, 1], dtype=complex),                    # |1⟩
        np.array([1, 1], dtype=complex) / np.sqrt(2),       # |+⟩
        np.array([1, -1], dtype=complex) / np.sqrt(2),      # |-⟩
        np.array([1, 1j], dtype=complex) / np.sqrt(2),      # |+i⟩
        np.array([1, -1j], dtype=complex) / np.sqrt(2),     # |-i⟩
    ]
    
    generators = []
    for psi_a in site_states:
        for psi_b in site_states:
            kraus = extract_kraus_operators(U, d_B, psi_a, psi_b)
            for K in kraus:
                trace_part = np.trace(K) / d_B
                G = (K - trace_part * np.eye(d_B)) / eps
                if np.linalg.norm(G) > 1e-8:
                    G_herm = (G + G.conj().T) / 2.0
                    G_anti = (G - G.conj().T) / (2.0j)
                    if np.linalg.norm(G_herm) > 1e-8:
                        generators.append(G_herm / np.linalg.norm(G_herm))
                    if np.linalg.norm(G_anti) > 1e-8:
                        generators.append(G_anti / np.linalg.norm(G_anti))
    
    basis, sv = find_independent_generators(generators, d_B)
    n_found = len(basis)
    n_expected = d_B**2 - 1
    
    print(f"  From {len(site_states)}² = {len(site_states)**2} site configurations:")
    print(f"  Generators found: {n_found} / {n_expected}")
    
    return n_found, n_expected


def minimal_site_states_test(d_B, variant='standard'):
    """
    Find the MINIMUM number of site state pairs needed to generate
    the full algebra. Start with computational basis and add states
    until we have enough generators.
    """
    print(f"\n  Minimal site states test, d_B = {d_B}")
    
    H = transmission_hamiltonian(d_B, variant)
    eps = 1e-4
    U = expm(-1j * eps * H)
    n_expected = d_B**2 - 1
    
    # Start with just |0⟩, |1⟩ and add states incrementally
    base_states = [
        np.array([1, 0], dtype=complex),
        np.array([0, 1], dtype=complex),
    ]
    
    extra_states = [
        np.array([1, 1], dtype=complex) / np.sqrt(2),
        np.array([1, -1], dtype=complex) / np.sqrt(2),
        np.array([1, 1j], dtype=complex) / np.sqrt(2),
        np.array([1, -1j], dtype=complex) / np.sqrt(2),
    ]
    
    all_states = base_states.copy()
    
    for round_num in range(len(extra_states) + 1):
        generators = []
        for psi_a in all_states:
            for psi_b in all_states:
                kraus = extract_kraus_operators(U, d_B, psi_a, psi_b)
                for K in kraus:
                    trace_part = np.trace(K) / d_B
                    G = (K - trace_part * np.eye(d_B)) / eps
                    if np.linalg.norm(G) > 1e-8:
                        G_herm = (G + G.conj().T) / 2.0
                        G_anti = (G - G.conj().T) / (2.0j)
                        if np.linalg.norm(G_herm) > 1e-8:
                            generators.append(G_herm / np.linalg.norm(G_herm))
                        if np.linalg.norm(G_anti) > 1e-8:
                            generators.append(G_anti / np.linalg.norm(G_anti))
        
        basis, sv = find_independent_generators(generators, d_B)
        n_found = len(basis)
        
        state_labels = ['|0⟩', '|1⟩', '|+⟩', '|-⟩', '|+i⟩', '|-i⟩']
        used = state_labels[:len(all_states)]
        print(f"  {len(all_states)} states ({', '.join(used)}): {n_found} / {n_expected} generators")
        
        if n_found >= n_expected:
            print(f"  → Full algebra achieved with {len(all_states)} site states!")
            break
        
        if round_num < len(extra_states):
            all_states.append(extra_states[round_num])
    
    return n_found, len(all_states)


def commutator_closure_test(d_B, variant='standard'):
    """
    Even if a single H doesn't directly generate all d²-1 generators,
    the COMMUTATOR CLOSURE might. Take the generators we get directly,
    compute all commutators, and see if the algebra closes at su(d_B).
    
    This is physically important: echoes interact (commutator = 
    sequential operations), and interactions might generate new 
    generators not present in single transmissions.
    """
    print(f"\n  Commutator closure test, d_B = {d_B}")
    
    H = transmission_hamiltonian(d_B, variant)
    
    # Get generators from just computational basis states
    eps = 1e-4
    U = expm(-1j * eps * H)
    
    comp_states = [
        np.array([1, 0], dtype=complex),
        np.array([0, 1], dtype=complex),
    ]
    
    generators = []
    for psi_a in comp_states:
        for psi_b in comp_states:
            kraus = extract_kraus_operators(U, d_B, psi_a, psi_b)
            for K in kraus:
                trace_part = np.trace(K) / d_B
                G = (K - trace_part * np.eye(d_B)) / eps
                if np.linalg.norm(G) > 1e-8:
                    G_herm = (G + G.conj().T) / 2.0
                    G_anti = (G - G.conj().T) / (2.0j)
                    if np.linalg.norm(G_herm) > 1e-8:
                        generators.append(G_herm / np.linalg.norm(G_herm))
                    if np.linalg.norm(G_anti) > 1e-8:
                        generators.append(G_anti / np.linalg.norm(G_anti))
    
    basis, sv = find_independent_generators(generators, d_B)
    n_direct = len(basis)
    n_expected = d_B**2 - 1
    print(f"  Direct generators (comp basis only): {n_direct}")
    
    # Now close under commutators
    all_gens = list(basis)
    for iteration in range(5):
        new_gens = []
        for i in range(len(all_gens)):
            for j in range(i+1, len(all_gens)):
                comm = all_gens[i] @ all_gens[j] - all_gens[j] @ all_gens[i]
                comm_herm = -1j * comm  # [T_a, T_b] = if_{abc} T_c → T_c ~ -i[T_a,T_b]
                comm_herm = (comm_herm + comm_herm.conj().T) / 2.0
                comm_herm -= np.trace(comm_herm)/d_B * np.eye(d_B)
                if np.linalg.norm(comm_herm) > 1e-8:
                    new_gens.append(comm_herm / np.linalg.norm(comm_herm))
        
        all_gens.extend(new_gens)
        basis_closed, sv_closed = find_independent_generators(all_gens, d_B)
        n_closed = len(basis_closed)
        print(f"  After {iteration+1} rounds of commutators: {n_closed} generators")
        
        if n_closed >= n_expected:
            print(f"  → Full su({d_B}) achieved via commutator closure!")
            break
        
        all_gens = list(basis_closed)
    
    return n_direct, n_closed


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)
    
    print("="*70)
    print("CRITICAL TEST: SINGLE HAMILTONIAN ALGEBRA GENERATION")
    print("="*70)
    
    # Test 1: Does each Hamiltonian type alone generate the full algebra?
    print(f"\n{'='*60}")
    print("TEST 1: Single Hamiltonian, random site states")
    print(f"{'='*60}")
    
    results = {}
    for d_B in [2, 3]:
        for variant in ['standard', 'full']:
            n, exp, _, _ = single_hamiltonian_test(d_B, variant, n_samples=500)
            results[(d_B, variant)] = (n, exp)
    
    # Test 2: Systematic site states
    print(f"\n{'='*60}")
    print("TEST 2: Systematic (non-random) site states")
    print(f"{'='*60}")
    
    for d_B in [2, 3]:
        for variant in ['standard', 'full']:
            site_basis_states_test(d_B, variant)
    
    # Test 3: Minimal number of site states
    print(f"\n{'='*60}")
    print("TEST 3: Minimal site states for full algebra")
    print(f"{'='*60}")
    
    for d_B in [2, 3]:
        for variant in ['standard', 'full']:
            minimal_site_states_test(d_B, variant)
    
    # Test 4: Commutator closure
    print(f"\n{'='*60}")
    print("TEST 4: Commutator closure from computational basis")
    print(f"{'='*60}")
    
    for d_B in [2, 3]:
        for variant in ['standard', 'full']:
            commutator_closure_test(d_B, variant)
    
    # Summary
    print(f"\n{'='*70}")
    print("PHYSICAL INTERPRETATION")
    print(f"{'='*70}")
    print("""
Key question: Is the echo algebra su(d_B) a trivial result 
(random unitaries always generate full algebra) or physically 
meaningful (specific transmission structure + matter diversity 
generates gauge algebra)?

The answer depends on:
1. Whether a SINGLE physical Hamiltonian generates full su(d_B)
2. Whether matter diversity (different site states) is NECESSARY
3. Whether commutator closure (echo interactions) plays a role

If the standard Hamiltonian + computational basis states alone
generate a PROPER SUBALGEBRA that closes to full su(d_B) only
via commutators, this tells a beautiful story:
  - Individual transmissions carry limited gauge content
  - Echo INTERACTIONS generate the remaining generators
  - The full gauge group emerges from dynamics, not kinematics
""")