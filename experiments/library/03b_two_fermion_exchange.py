"""
EXPERIMENT 03: TWO-FERMION EXCHANGE IN THE SUBSTRATE
====================================================

Goal:
  Build a minimal *working* prototype that shows:
    - Two particles follow different exchange histories
    - The underlying SU(2)/quaternion geometry gives a relative phase of -1

This is essentially Experiment 02 (ij != ji) upgraded with:
    - Explicit "two particles" (A and B)
    - Explicit worldline histories on a square plaquette
    - Two different orderings of the SAME geometric moves

We are NOT enforcing antisymmetry by hand.
We are just letting the non-Abelian substrate tell us the phase.
"""

import numpy as np


# ---------------------------------------------------------------------
# 1. Substrate Geometry: SU(2) / Quaternions
# ---------------------------------------------------------------------

# Pauli matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j],
                    [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0],
                    [0, -1]], dtype=np.complex128)

# Generators of "translation" in each direction
# (Same convention as your 02_derive_fermions.py)
Step_X = -1j * sigma_x   # quaternion "i"
Step_Y = -1j * sigma_y   # quaternion "j"
Step_Z = -1j * sigma_z   # quaternion "k"

# For completeness, define inverses (Hermitian conjugates)
Step_X_dag = Step_X.conj().T
Step_Y_dag = Step_Y.conj().T
Step_Z_dag = Step_Z.conj().T


def identify_state(matrix: np.ndarray) -> str:
    """Rough labeling of where we are in quaternion space."""
    if np.allclose(matrix, sigma_0):   return " 1 (Identity)"
    if np.allclose(matrix, -sigma_0):  return "-1 (Global Phase)"
    if np.allclose(matrix, Step_X):    return " i (Twist X)"
    if np.allclose(matrix, -Step_X):   return "-i (Inv Twist X)"
    if np.allclose(matrix, Step_Y):    return " j (Twist Y)"
    if np.allclose(matrix, -Step_Y):   return "-j (Inv Twist Y)"
    if np.allclose(matrix, Step_Z):    return " k (Twist Z)"
    if np.allclose(matrix, -Step_Z):   return "-k (Inv Twist Z)"
    return "?(generic SU(2) element)"


# ---------------------------------------------------------------------
# 2. Minimal "Graph" and Worldlines
# ---------------------------------------------------------------------

# We'll use a 2x2 plaquette just to have something to talk about:
#
#   (0,1) ---- (1,1)
#     |          |
#   (0,0) ---- (1,0)
#
# Particle A and B start at opposite corners and do an exchange-ish loop.
# The *geometry* (SU(2) twists on each step) is what we actually care about.


# Directions on the square and their associated SU(2) step operators
DIR_TO_STEP = {
    "+x": Step_X,
    "-x": Step_X_dag,
    "+y": Step_Y,
    "-y": Step_Y_dag,
}


def apply_history(history, verbose=True):
    """
    Apply a time-ordered sequence of moves for two labeled particles.

    history: list of (who, direction) tuples, e.g.
        [("A", "+x"), ("B", "+y")]

    We don't enforce hard-core constraints; this is purely to track
    how the substrate's SU(2) frame twists as the two worldlines braid.
    """
    # Global "frame" state in SU(2)
    psi = sigma_0.copy()

    # Just for pretty-printing positions (not dynamically important to phase)
    pos_A = np.array([0, 0], dtype=int)  # A starts at (0,0)
    pos_B = np.array([1, 1], dtype=int)  # B starts at (1,1)

    def move(pos, direction):
        if direction == "+x":
            return pos + np.array([1, 0])
        if direction == "-x":
            return pos + np.array([-1, 0])
        if direction == "+y":
            return pos + np.array([0, 1])
        if direction == "-y":
            return pos + np.array([0, -1])
        raise ValueError(f"Unknown direction {direction}")

    if verbose:
        print("  Initial global frame:", identify_state(psi))
        print(f"  Start positions: A{tuple(pos_A)}, B{tuple(pos_B)}\n")

    for step_idx, (who, direction) in enumerate(history, start=1):
        U_step = DIR_TO_STEP[direction]

        # Update the global SU(2) frame
        psi = U_step @ psi

        # Update positions just for narration
        if who == "A":
            pos_A = move(pos_A, direction)
        elif who == "B":
            pos_B = move(pos_B, direction)
        else:
            raise ValueError("who must be 'A' or 'B'")

        if verbose:
            print(f"  Step {step_idx}: {who} moves {direction}")
            print(f"    Positions: A{tuple(pos_A)}, B{tuple(pos_B)}")
            print(f"    Frame: {identify_state(psi)}")

    if verbose:
        print("\n  Final frame matrix:\n", psi)
        print("  Final frame label:", identify_state(psi))
        print("-" * 60)

    return psi


# ---------------------------------------------------------------------
# 3. Define Two Exchange Histories
# ---------------------------------------------------------------------
#
# The trick:
#   - Both histories have the same *net* geometric motion around the square,
#     but the *order* in which A and B move is swapped.
#   - This is exactly the ij vs ji story dressed up as "two particles".
#
# History A:
#   1. A goes +x  (uses Step_X)
#   2. B goes +y  (uses Step_Y)
#
#   -> total frame operator U_A = Step_Y * Step_X
#
# History B:
#   1. B goes +y  (uses Step_Y)
#   2. A goes +x  (uses Step_X)
#
#   -> total frame operator U_B = Step_X * Step_Y
#
# Because SU(2) is non-Abelian, U_A = - U_B, so the overlap <A|B> = -1.
# We interpret that as a fermionic exchange phase emerging from geometry.


def run_experiment():
    print("=== Experiment 03: Two-Fermion Exchange in the Substrate ===\n")

    print("Step operators (quaternion links):")
    print("  Step_X (i):\n", Step_X)
    print("  Step_Y (j):\n", Step_Y)
    print("  Step_Z (k):\n", Step_Z)
    print("\nWe now define two different exchange histories:\n")

    # History A: A moves first, then B (X then Y)
    history_A = [
        ("A", "+x"),  # apply Step_X
        ("B", "+y"),  # apply Step_Y
    ]

    # History B: B moves first, then A (Y then X)
    history_B = [
        ("B", "+y"),  # apply Step_Y
        ("A", "+x"),  # apply Step_X
    ]

    print("[History A]  ('Clockwise-ish' exchange: A then B)\n")
    psi_A = apply_history(history_A, verbose=True)

    print("\n[History B]  ('Counterclockwise-ish' exchange: B then A)\n")
    psi_B = apply_history(history_B, verbose=True)

    # -----------------------------------------------------------------
    # 4. Compare the Two Final States
    # -----------------------------------------------------------------
    overlap = np.trace(psi_A @ psi_B.conj().T) / 2.0

    print("\n=== Interference Measurement ===")
    print("We treat the SU(2) matrices as normalized 'states' via:")
    print("  <A|B> = Tr(psi_A psi_B^†) / 2")
    print(f"\n  <History A | History B> = {overlap:.4f}")

    print("\n=== Verdict ===")
    if np.isclose(overlap, -1.0):
        print("  → FERMIONIC EXCHANGE PHASE DETECTED (−1).")
        print("    The non-Abelian substrate makes the two exchange")
        print("    histories differ by a global minus sign.")
    elif np.isclose(overlap, 1.0):
        print("  → BOSONIC (commutative) behavior ( +1 ).")
    else:
        print("  → Neither +1 nor −1 exactly; mixed / anyonic-like behavior.")

    print("\nDone.")


if __name__ == "__main__":
    run_experiment()
