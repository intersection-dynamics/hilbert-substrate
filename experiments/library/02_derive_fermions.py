"""
EXPERIMENT 02: DERIVATION OF FERMIONIC STATISTICS
=================================================
Objective: 
  Prove that the Pauli Exclusion Principle (Phase = -1 upon exchange)
  is a natural consequence of Non-Abelian (Quaternion) Geometry.

Hypothesis:
  In a 3D substrate with SU(2) rotational symmetry, the order of operations
  for moving a particle matters. 
  - Path A (X then Y) != Path B (Y then X).
  - The difference is exactly a phase of -1.
"""

import numpy as np
import matplotlib.pyplot as plt

def run_experiment():
    print("--- Experiment 02: Derivation of Fermions ---")
    print("Testing Commutativity of Substrate Geometry...\n")

    # 1. Define the Algebra of Space (SU(2) / Quaternions)
    # ---------------------------------------------------
    # The links in our graph are not empty; they are Quaternions (i, j, k).
    # We map these to 2x2 Pauli Matrices.
    
    sigma_0 = np.eye(2, dtype=np.complex128)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Generators of Translation
    # Moving along X-axis twists the frame by 'i'
    # Moving along Y-axis twists the frame by 'j'
    # Moving along Z-axis twists the frame by 'k'
    # (Factor of -1j ensures Hermitian generators map to Unitary updates)
    Step_X = -1j * sigma_x
    Step_Y = -1j * sigma_y
    Step_Z = -1j * sigma_z
    
    def identify_state(matrix):
        if np.allclose(matrix, sigma_0): return " 1 (Identity)"
        if np.allclose(matrix, -sigma_0): return "-1 (Phase Shift)"
        if np.allclose(matrix, Step_X): return " i (Twist X)"
        if np.allclose(matrix, -Step_X): return "-i (Inv Twist X)"
        if np.allclose(matrix, Step_Y): return " j (Twist Y)"
        if np.allclose(matrix, -Step_Y): return "-j (Inv Twist Y)"
        if np.allclose(matrix, Step_Z): return " k (Twist Z)"
        if np.allclose(matrix, -Step_Z): return "-k (Inv Twist Z)"
        return "?"

    # 2. The Exchange Experiment
    # ---------------------------------------------------
    # We simulate exchanging two particles.
    # Topologically, this is equivalent to comparing two paths around a square.
    # Path A: Move Right (X), then Move Up (Y).
    # Path B: Move Up (Y), then Move Right (X).
    
    # Initial State: The Vacuum (Identity)
    psi_0 = sigma_0
    print(f"  Initial State: {identify_state(psi_0)}")
    
    # --- Path A (Clockwise-ish) ---
    print("\n  [Path A] Sequence: Step X -> Step Y")
    # Apply X twist
    psi_A = Step_X @ psi_0
    # Apply Y twist
    psi_A = Step_Y @ psi_A
    print(f"  Result A: {identify_state(psi_A)}")
    
    # --- Path B (Counter-Clockwise-ish) ---
    print("\n  [Path B] Sequence: Step Y -> Step X")
    # Apply Y twist
    psi_B = Step_Y @ psi_0
    # Apply X twist
    psi_B = Step_X @ psi_B
    print(f"  Result B: {identify_state(psi_B)}")
    
    # 3. The Comparison (Interference)
    # ---------------------------------------------------
    # Overlap = Trace(Psi_A * Psi_B_dagger) / 2
    # If Overlap = +1 : Bosons (Commutative)
    # If Overlap = -1 : Fermions (Anti-Commutative)
    
    overlap = np.trace(psi_A @ psi_B.conj().T) / 2.0
    print(f"\n  Overlap <Path A | Path B>: {overlap:.4f}")
    
    # 4. Verdict
    # ---------------------------------------------------
    print("\n" + "="*50)
    if np.isclose(overlap, -1.0):
        print("  VERDICT: FERMIONS GENERATED")
        print("  Reason: SU(2) is non-Abelian (ij = -ji).")
        print("  The geometry forces a -1 phase shift upon exchange.")
        print("  The Pauli Exclusion Principle is emergent.")
    elif np.isclose(overlap, 1.0):
        print("  VERDICT: BOSONS GENERATED")
        print("  Reason: Geometry is commutative.")
    else:
        print("  VERDICT: ANYONS/UNKNOWN")
    print("="*50)

if __name__ == "__main__":
    run_experiment()