#!/usr/bin/env python3
"""
BOSONIC vs FERMIONIC STATISTICS: Emergence from the Hilbert Substrate
======================================================================

This script demonstrates that quantum statistics (Fermi-Dirac vs Bose-Einstein)
are NOT fundamental axioms, but emerge from how excitations are constructed
in Hilbert space.

Key insight: The Jordan-Wigner transformation shows that:
- WITH the Z-string → anticommutation → FERMIONS
- WITHOUT the Z-string → commutation → BOSONS

Both constructions operate on the SAME underlying Hilbert space (C²)^⊗N.

Author: Ben Bray
Part of: The Hilbert Substrate Framework, Paper III
"""

import numpy as np
from typing import List, Tuple

# =============================================================================
# PAULI MATRICES AND TENSOR PRODUCTS
# =============================================================================

def pauli() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the Pauli matrices I, X, Y, Z."""
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return I, X, Y, Z


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    """Compute tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


# =============================================================================
# FERMION OPERATORS (JORDAN-WIGNER)
# =============================================================================

def build_fermion_operators(N: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build fermionic creation and annihilation operators via Jordan-Wigner.
    
    The Jordan-Wigner transformation:
        c†_j = Z₀ ⊗ Z₁ ⊗ ... ⊗ Z_{j-1} ⊗ b†_j ⊗ I_{j+1} ⊗ ... ⊗ I_{N-1}
        
    The "string" of Z operators before site j is what creates the
    antisymmetric (fermionic) statistics.
    
    Parameters
    ----------
    N : int
        Number of sites (qubits)
        
    Returns
    -------
    c_create : list of np.ndarray
        Fermionic creation operators c†_j for j = 0, ..., N-1
    c_destroy : list of np.ndarray
        Fermionic annihilation operators c_j for j = 0, ..., N-1
    """
    I, X, Y, Z = pauli()
    
    # Local creation/annihilation (hard-core boson basis)
    # Convention: |0⟩ = empty, |1⟩ = occupied
    b_create = np.array([[0, 0], [1, 0]], dtype=np.complex128)   # |1⟩⟨0|
    b_destroy = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0⟩⟨1|
    
    c_create = []
    c_destroy = []
    
    for j in range(N):
        # Build operators with Z-string before site j
        ops_create = []
        ops_destroy = []
        for m in range(N):
            if m < j:
                # Z-string: these Z's create the fermionic sign
                ops_create.append(Z)
                ops_destroy.append(Z)
            elif m == j:
                # The actual creation/annihilation at site j
                ops_create.append(b_create)
                ops_destroy.append(b_destroy)
            else:
                # Identity after site j
                ops_create.append(I)
                ops_destroy.append(I)
        
        c_create.append(kron_n(ops_create))
        c_destroy.append(kron_n(ops_destroy))
    
    return c_create, c_destroy


# =============================================================================
# BOSON OPERATORS (NO STRING)
# =============================================================================

def build_boson_operators(N: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build bosonic creation and annihilation operators (local, no JW string).
    
    Without the Jordan-Wigner string:
        b†_j = I ⊗ I ⊗ ... ⊗ b†_j ⊗ ... ⊗ I
        
    These are purely local operators that commute at different sites,
    giving bosonic statistics.
    
    Note: These are "hard-core bosons" (at most 1 per site) because we're
    using spin-1/2 (qubit) degrees of freedom. In the dilute limit, they
    satisfy standard bosonic commutation relations.
    
    Parameters
    ----------
    N : int
        Number of sites (qubits)
        
    Returns
    -------
    b_create : list of np.ndarray
        Bosonic creation operators b†_j for j = 0, ..., N-1
    b_destroy : list of np.ndarray
        Bosonic annihilation operators b_j for j = 0, ..., N-1
    """
    I, X, Y, Z = pauli()
    
    # Local creation/annihilation
    b_create_local = np.array([[0, 0], [1, 0]], dtype=np.complex128)   # |1⟩⟨0|
    b_destroy_local = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0⟩⟨1|
    
    b_create = []
    b_destroy = []
    
    for j in range(N):
        # NO Z-string, just local operators
        ops_create = [b_create_local if m == j else I for m in range(N)]
        ops_destroy = [b_destroy_local if m == j else I for m in range(N)]
        
        b_create.append(kron_n(ops_create))
        b_destroy.append(kron_n(ops_destroy))
    
    return b_create, b_destroy


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def check_anticommutation(c_destroy: List[np.ndarray], 
                          c_create: List[np.ndarray]) -> np.ndarray:
    """
    Check fermionic anticommutation relations: {c_i, c_j†} = δ_ij
    
    Returns matrix of results (should be identity matrix).
    """
    N = len(c_create)
    dim = c_create[0].shape[0]
    results = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            # Anticommutator: {A, B} = AB + BA
            anticomm = c_destroy[i] @ c_create[j] + c_create[j] @ c_destroy[i]
            # Check if it equals δ_ij * I
            expected = np.eye(dim) if i == j else np.zeros((dim, dim))
            results[i, j] = 1 if np.allclose(anticomm, expected) else 0
    
    return results


def check_commutation_in_vacuum(b_destroy: List[np.ndarray], 
                                b_create: List[np.ndarray]) -> np.ndarray:
    """
    Check bosonic commutation relations in vacuum: [b_i, b_j†] = δ_ij
    
    For hard-core bosons: [b, b†] = 1 - 2n
    In the vacuum (n=0): [b, b†] = 1
    
    Returns matrix of vacuum expectation values.
    """
    N = len(b_create)
    dim = b_create[0].shape[0]
    
    # Vacuum state: |000...0⟩
    vacuum = np.zeros(dim, dtype=np.complex128)
    vacuum[0] = 1
    
    results = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            # Commutator: [A, B] = AB - BA
            comm = b_destroy[i] @ b_create[j] - b_create[j] @ b_destroy[i]
            # Vacuum expectation value
            results[i, j] = np.real(vacuum.conj() @ comm @ vacuum)
    
    return results


def check_exchange_statistics(ops_create: List[np.ndarray], 
                              site_a: int, site_b: int) -> float:
    """
    Check exchange statistics by creating particles in different orders.
    
    Compare: O†_b O†_a |vac⟩  vs  O†_a O†_b |vac⟩
    
    For fermions: ratio = -1 (antisymmetric)
    For bosons: ratio = +1 (symmetric)
    
    Returns the ratio (exchange phase).
    """
    dim = ops_create[0].shape[0]
    
    # Vacuum state
    vacuum = np.zeros(dim, dtype=np.complex128)
    vacuum[0] = 1
    
    # Create in order: a then b
    psi_ab = ops_create[site_b] @ ops_create[site_a] @ vacuum
    
    # Create in order: b then a
    psi_ba = ops_create[site_a] @ ops_create[site_b] @ vacuum
    
    # Find ratio where both are nonzero
    nonzero = np.abs(psi_ba) > 1e-10
    if np.any(nonzero):
        ratios = psi_ab[nonzero] / psi_ba[nonzero]
        return np.real(ratios[0])
    else:
        return 0.0


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """Run the complete demonstration of quantum statistics emergence."""
    
    print("=" * 70)
    print("  QUANTUM STATISTICS FROM THE HILBERT SUBSTRATE")
    print("=" * 70)
    
    N = 4
    I, X, Y, Z = pauli()
    
    print(f"\n  System: N = {N} qubits")
    print(f"  Hilbert space dimension: {2**N}")
    print(f"\n  Convention: |0⟩ = empty site, |1⟩ = occupied site")
    print(f"  Vacuum state: |{'0'*N}⟩ (all sites empty)")
    
    # =========================================================================
    # FERMIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  FERMIONS: Jordan-Wigner Construction")
    print("=" * 70)
    
    print("""
  The Jordan-Wigner transformation introduces a "string" of Z operators:
  
    c†_j = (Z₀ ⊗ Z₁ ⊗ ... ⊗ Z_{j-1}) ⊗ b†_j ⊗ I_{j+1} ⊗ ... ⊗ I_{N-1}
    
  This string is the key to fermionic statistics. When exchanging particles,
  the Z operators at intermediate sites contribute factors of -1.
""")
    
    c_create, c_destroy = build_fermion_operators(N)
    
    print("  Anticommutation relations {c_i, c_j†}:")
    anticomm_results = check_anticommutation(c_destroy, c_create)
    for i in range(N):
        row = "    "
        for j in range(N):
            symbol = "δ" if anticomm_results[i, j] == 1 and i == j else \
                     ("0" if anticomm_results[i, j] == 1 else "?")
            row += f"{symbol:>3}"
        print(row)
    
    print("\n  ✓ {c_i, c_j†} = δ_ij  (FERMIONIC anticommutation)")
    
    # =========================================================================
    # BOSONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  BOSONS: Local Construction (No String)")
    print("=" * 70)
    
    print("""
  Without the Jordan-Wigner string, operators are purely local:
  
    b†_j = I ⊗ I ⊗ ... ⊗ b†_j ⊗ ... ⊗ I
    
  Local operators at different sites commute, giving bosonic statistics.
  (These are "hard-core bosons" limited to 0 or 1 per site.)
""")
    
    b_create, b_destroy = build_boson_operators(N)
    
    print("  Commutation relations [b_i, b_j†] (in vacuum):")
    comm_results = check_commutation_in_vacuum(b_destroy, b_create)
    for i in range(N):
        row = "    "
        for j in range(N):
            row += f"{comm_results[i, j]:+3.0f}"
        print(row)
    
    print("\n  ✓ [b_i, b_j†] = δ_ij in vacuum (BOSONIC commutation)")
    
    # =========================================================================
    # EXCHANGE STATISTICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  THE KEY TEST: EXCHANGE STATISTICS")
    print("=" * 70)
    
    print("\n  Create particles at sites 0 and 2, compare ordering:")
    
    # Fermions
    f_exchange = check_exchange_statistics(c_create, 0, 2)
    print(f"\n  FERMIONS: c†_2 c†_0|vac⟩  vs  c†_0 c†_2|vac⟩")
    print(f"    Ratio: {f_exchange:+.0f}")
    print(f"    → ANTISYMMETRIC (exchange gives -1)")
    
    # Bosons
    b_exchange = check_exchange_statistics(b_create, 0, 2)
    print(f"\n  BOSONS: b†_2 b†_0|vac⟩  vs  b†_0 b†_2|vac⟩")
    print(f"    Ratio: {b_exchange:+.0f}")
    print(f"    → SYMMETRIC (exchange gives +1)")
    
    # =========================================================================
    # CONCLUSION
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    
    print("""
  ╔════════════════════════════════════════════════════════════════════╗
  ║                                                                    ║
  ║   SAME underlying Hilbert space: (C²)^⊗N                           ║
  ║                                                                    ║
  ║   FERMIONS                         BOSONS                          ║
  ║   ─────────────────────────────────────────────────────────        ║
  ║   c†_j includes Z-string           b†_j is local (no string)       ║
  ║   {c, c†} = δ                      [b, b†] = δ (in vacuum)         ║
  ║   Exchange: −1                     Exchange: +1                    ║
  ║   Fermi-Dirac statistics           Bose-Einstein statistics        ║
  ║                                                                    ║
  ║   The Z-string encodes TOPOLOGICAL information that                ║
  ║   produces fermionic statistics. It tracks the "parity"            ║
  ║   of particles to the left of each site.                           ║
  ║                                                                    ║
  ║   Without the string, excitations are BOSONIC because              ║
  ║   local operators at different sites naturally commute.            ║
  ║                                                                    ║
  ╚════════════════════════════════════════════════════════════════════╝

  QUANTUM STATISTICS IS NOT A FUNDAMENTAL AXIOM.
  It emerges from how excitations are constructed in Hilbert space.
  
  The Hilbert Substrate Framework shows that both fermions and bosons
  can be realized within the same underlying quantum system—the
  difference is purely in how we define the creation operators.
""")


if __name__ == "__main__":
    main()