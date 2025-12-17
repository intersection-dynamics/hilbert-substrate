"""
EXPERIMENT 03c: SU(2) Holonomy Diagnostic on the Full Substrate
===============================================================

Goal
----

Use the *real* CombinedSubstrate3D to reproduce the logic of the
03b_two_fermion_exchange toy model:

    - Pick a 2x2 plaquette in the x–y plane.
    - Define two short "exchange-like" histories of two labeled points A,B
      that differ only by the ORDER in which we move along +x and +y.
    - Along each history, accumulate a global SU(2) "frame" by multiplying
      the actual link matrices U_ij from the substrate.
    - Compare the two histories with the inner product

          <A|B> = Tr(psi_A psi_B^†) / 2,

      exactly as in 03b.

If the local gauge geometry behaves like the simple quaternion cartoon
(Step_X = i, Step_Y = j), then these two histories will tend to differ
by an overall minus sign, giving <A|B> ≈ -1. In general, we just report
whatever the substrate actually does.

We do *not* impose any special structure beyond what CombinedSubstrate3D
already defines. This is a *diagnostic*, not a constraint.
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Import the core substrate from engine/substrate_and_fermions.py
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from engine.substrate_and_fermions import CombinedSubstrate3D  # type: ignore

# GPU (CuPy) imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    raise RuntimeError("CuPy is required for this diagnostic.")


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    "substrate": {
        "L": 3,                 # must be >= 2; 3 keeps things small
        "eta_scalar": 0.6,
        "decay_scalar": 0.05,
        "eta_matrix": 1.0,
        "decay_matrix": 0.1,
        "E_link": 1.0,
        "plaquette_coupling": 0.2,
        "gauge_noise": 0.001,
        "relax_steps": 80,
        "relax_dt": 0.05,
    },
    "diagnostic": {
        # z-slice on which to place the 2x2 plaquette
        # If None, uses L//2.
        "z_slice": None,
        # Whether to print the individual step matrices.
        "verbose_steps": True,
    },
}


# ============================================================================
# Helper: build a 2x2 plaquette and histories
# ============================================================================

def _node_index(L: int, x: int, y: int, z: int) -> int:
    """Map (x,y,z) to a single node index i."""
    return x * L * L + y * L + z


def _coord(L: int, i: int):
    """Inverse of _node_index; returns (x,y,z)."""
    x = i // (L * L)
    y = (i // L) % L
    z = i % L
    return x, y, z


def _move_2d(pos, direction):
    """
    Move in the x–y plane by one step.

    pos: (x, y) numpy array
    direction: "+x", "-x", "+y", "-y"
    """
    if direction == "+x":
        return pos + np.array([1, 0])
    if direction == "-x":
        return pos + np.array([-1, 0])
    if direction == "+y":
        return pos + np.array([0, 1])
    if direction == "-y":
        return pos + np.array([0, -1])
    raise ValueError(f"Unknown direction {direction}")


def _get_oriented_U(substrate: CombinedSubstrate3D, i: int, j: int) -> cp.ndarray:
    """
    Fetch the oriented SU(2) matrix U_ij from the substrate.

    CombinedSubstrate3D stores undirected links (i<j) internally, but its
    private method _get_U(i,j) returns the correctly oriented matrix:

        U_ij if (i,j) stored,
        U_ji^† if (j,i) stored.
    """
    # Access the private helper directly; this is an internal diagnostic.
    return substrate._get_U(i, j)  # type: ignore[attr-defined]


# ============================================================================
# Main diagnostic
# ============================================================================

def main():
    sub_cfg = CONFIG["substrate"]
    diag_cfg = CONFIG["diagnostic"]

    L = sub_cfg["L"]
    assert L >= 2, "L must be >= 2 for a 2x2 plaquette."

    z_slice = diag_cfg["z_slice"]
    if z_slice is None:
        z_slice = L // 2
    z_slice = int(z_slice) % L

    verbose_steps = bool(diag_cfg["verbose_steps"])

    print("=== Experiment 03c: SU(2) Holonomy Diagnostic ===\n")

    # ------------------------------------------------------------------
    # 1. Build and relax the combined substrate
    # ------------------------------------------------------------------
    substrate = CombinedSubstrate3D(
        L=L,
        eta_scalar=sub_cfg["eta_scalar"],
        decay_scalar=sub_cfg["decay_scalar"],
        eta_matrix=sub_cfg["eta_matrix"],
        decay_matrix=sub_cfg["decay_matrix"],
        E_link=sub_cfg["E_link"],
        plaquette_coupling=sub_cfg["plaquette_coupling"],
        gauge_noise=sub_cfg["gauge_noise"],
    )

    print(f"Building CombinedSubstrate3D with L={L}, z-slice={z_slice}")
    print("Relaxing gauge+scalar+stiffness geometry...")
    _ = substrate.evolve(
        steps=sub_cfg["relax_steps"],
        dt=sub_cfg["relax_dt"],
        record_history=False,
    )
    print("Geometry relaxation complete.\n")

    # ------------------------------------------------------------------
    # 2. Define a 2x2 plaquette in the x–y plane at z = z_slice
    # ------------------------------------------------------------------
    # We reuse the cartoon layout from 03b:
    #
    #   (0,1,z) ---- (1,1,z)
    #     |            |
    #   (0,0,z) ---- (1,0,z)
    #
    # Particle A starts at (0,0,z), particle B at (1,1,z).
    #
    A_start = np.array([0, 0], dtype=int)
    B_start = np.array([1, 1], dtype=int)

    # Two histories differ only by ordering of +x / +y moves
    history_A = [
        ("A", "+x"),  # A moves +x first
        ("B", "+y"),  # then B moves +y
    ]
    history_B = [
        ("B", "+y"),  # B moves +y first
        ("A", "+x"),  # then A moves +x
    ]

    print("We define two 'exchange-like' SU(2) histories on the plaquette:\n")
    print("  History A: A then B")
    print("    Steps:", history_A)
    print("  History B: B then A")
    print("    Steps:", history_B)
    print()

    # ------------------------------------------------------------------
    # 3. Function to accumulate SU(2) frame along a history
    # ------------------------------------------------------------------

    def accumulate_frame(history):
        """
        Starting from identity, for each step:
          - determine which particle moves (A or B),
          - determine its current (x,y,z_slice),
          - determine the target (after +x / +y),
          - fetch oriented U_ij from the substrate,
          - multiply psi <- U_step @ psi.
        """
        pos_A = A_start.copy()
        pos_B = B_start.copy()
        psi = cp.eye(2, dtype=cp.complex128)

        if verbose_steps:
            print("  Initial frame:")
            print(psi.get())
            print(f"  Start A={tuple(pos_A)}, B={tuple(pos_B)}\n")

        for step_idx, (who, direction) in enumerate(history, start=1):
            if who == "A":
                pos_before = pos_A
            elif who == "B":
                pos_before = pos_B
            else:
                raise ValueError("who must be 'A' or 'B'")

            pos_after = _move_2d(pos_before, direction)

            # Wrap around the lattice (periodic BCs)
            x0 = int(pos_before[0] % L)
            y0 = int(pos_before[1] % L)
            x1 = int(pos_after[0] % L)
            y1 = int(pos_after[1] % L)

            i = _node_index(L, x0, y0, z_slice)
            j = _node_index(L, x1, y1, z_slice)

            U_step = _get_oriented_U(substrate, i, j)

            # Update global SU(2) frame
            psi = U_step @ psi

            # Update positions for narration
            if who == "A":
                pos_A = pos_after
            else:
                pos_B = pos_after

            if verbose_steps:
                print(f"  Step {step_idx}: {who} moves {direction}")
                print(f"    Node indices: i={i} ({x0},{y0},{z_slice}), "
                      f"j={j} ({x1},{y1},{z_slice})")
                print("    U_step:")
                print(U_step.get())
                print("    Updated frame psi:")
                print(psi.get())
                print(f"    Positions: A={tuple(pos_A)}, B={tuple(pos_B)}\n")

        return psi

    # ------------------------------------------------------------------
    # 4. Accumulate frames for both histories
    # ------------------------------------------------------------------
    print("=== History A (A then B) ===")
    psi_A = accumulate_frame(history_A)
    print("Final frame for History A:")
    print(psi_A.get())
    print()

    print("=== History B (B then A) ===")
    psi_B = accumulate_frame(history_B)
    print("Final frame for History B:")
    print(psi_B.get())
    print()

    # ------------------------------------------------------------------
    # 5. Interference-style inner product <A|B> = Tr(psi_A psi_B^†)/2
    # ------------------------------------------------------------------
    inner = psi_A @ psi_B.conj().T
    overlap = 0.5 * cp.trace(inner)
    overlap_val = complex(overlap.get())

    print("=== SU(2) Frame Overlap ===")
    print("We treat the SU(2) frames as normalized 'states' via:")
    print("  <A|B> = Tr(psi_A psi_B^†) / 2\n")
    print(f"  <History A | History B> = {overlap_val.real:+.4f}"
          f"{overlap_val.imag:+.4f}j")
    print()

    # Classification (just for narrative)
    if np.isclose(overlap_val, -1.0 + 0.0j, atol=1e-2):
        print("  → Near-fermionic SU(2) holonomy (−1-like overlap).")
        print("    The two histories differ by an approximate global minus sign.")
    elif np.isclose(overlap_val, 1.0 + 0.0j, atol=1e-2):
        print("  → Near-bosonic / commuting behavior (+1-like overlap).")
        print("    The two histories produce nearly the same SU(2) frame.")
    else:
        print("  → Generic non-Abelian holonomy (neither +1 nor −1).")
        print("    The two histories explore different orientations in SU(2).")

    print("\nDone.")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()
