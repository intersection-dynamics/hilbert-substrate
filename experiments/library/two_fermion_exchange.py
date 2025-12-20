"""
Two-Fermion Exchange Experiment on the Combined Substrate
========================================================

This script builds on two components:

  1) CombinedSubstrate3D from substrate_three_levels.py
     - 3D lattice with scalar feedback, SU(2) link feedback, and
       living-link-inspired stiffness (E_link).

  2) TwoFermionSector from two_fermion_sector.py
     - Antisymmetric two-fermion Hilbert space built from a given
       single-particle Hamiltonian H1.

Experiment Outline
------------------

1. Relax the combined substrate geometry for a short time to obtain
   a nontrivial but stable background (t_ij, U_ij, E_link).

2. Build the single-particle Hamiltonian H1 from this relaxed geometry.

3. Construct the antisymmetric two-fermion sector on this H1.

4. Prepare two localized fermions at opposite sides of a 2D slice:
     fermion A at (0,1,1), fermion B at (2,1,1) (spin up for both).

5. Define a discrete "braid" path:
   - A moves around the top of a 3x3 square (y increasing then decreasing).
   - B moves around the bottom of the same square (y decreasing then increasing).
   After all segments, the traps return to their original positions,
   but the worldlines of the two fermions have been exchanged.

6. Implement a steering potential:
   - At each segment s, add a diagonal potential V_s that creates two
     deep wells at the segment's target positions for A and B.
   - For that segment, keep H1 + V_s fixed and evolve the two-fermion
     state for several small time steps.

7. At the end of the loop, compare Ψ_final with Ψ_initial via the overlap
     overlap = ⟨Ψ_initial | Ψ_final⟩
   For an ideal fermion exchange, we expect overlap ≈ −1 (global minus sign).

The script also tracks:
   - norm(t)
   - diagonal occupation probability (Pauli hole)
   - |overlap(t) with Ψ_initial| at each segment endpoint
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import CombinedSubstrate3D and the two-fermion machinery
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

try:
    from substrate_three_levels import CombinedSubstrate3D
except ImportError as e:
    raise ImportError(
        "Could not import CombinedSubstrate3D from substrate_three_levels.py."
    ) from e

try:
    from two_fermion_sector import (
        TwoFermionSector,
        single_particle_index,
        build_site_map,
    )
except ImportError as e:
    raise ImportError(
        "Could not import TwoFermionSector helpers from two_fermion_sector.py."
    ) from e

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy import linalg as cpla
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    cpla = None
    GPU_AVAILABLE = False
    raise RuntimeError("CuPy is required for this experiment.")


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # Substrate (three mechanisms combined)
    "substrate": {
        "L": 3,
        "eta_scalar": 0.6,
        "decay_scalar": 0.05,
        "eta_matrix": 1.0,
        "decay_matrix": 0.1,
        "E_link": 1.0,
        "plaquette_coupling": 0.2,
        "gauge_noise": 0.001,
        "geom_relax_steps": 80,
        "geom_dt": 0.05,
    },

    # Exchange experiment
    "exchange": {
        # Path for fermion A and B on the x-y plane at fixed z=1
        # Each entry is (x, y, z)
        # Paths have the same length and describe a closed loop returning
        # to the starting trap positions, but with the worldlines exchanged.
        "path_A": [
            (0, 1, 1),
            (0, 2, 1),
            (1, 2, 1),
            (2, 2, 1),
            (2, 1, 1),
        ],
        "path_B": [
            (2, 1, 1),
            (2, 0, 1),
            (1, 0, 1),
            (0, 0, 1),
            (0, 1, 1),
        ],

        # Trap depth (negative = potential well)
        "trap_depth": -5.0,

        # Number of small time steps per segment
        "steps_per_segment": 10,

        # Time step for each small evolution
        "dt": 0.05,

        # Single-particle spin choices for the two fermions (0=up, 1=down)
        "spin_A": 0,
        "spin_B": 0,

        "plot": True,
    },
}


# ============================================================================
# Helper: build single-particle H1 from relaxed substrate
# ============================================================================

def build_single_particle_hamiltonian(substrate: CombinedSubstrate3D) -> cp.ndarray:
    """
    Extract the single-particle Hamiltonian from the combined substrate.

    This uses the same Hamiltonian the substrate uses for its 1-body ψ field,
    but here we interpret it as the operator for a single fermion.
    """
    H1 = substrate.hamiltonian()
    return H1.astype(cp.complex128)


# ============================================================================
# Helper: add a steering potential for two traps
# ============================================================================

def add_trap_potential(H1_base: cp.ndarray,
                       L: int,
                       pos_A,
                       pos_B,
                       trap_depth: float) -> cp.ndarray:
    """
    Construct H1 = H1_base + V, where V is a diagonal single-particle potential
    that creates wells of depth trap_depth at two lattice sites pos_A and pos_B.

    The potential is spin-independent: both spin components at those sites
    get the same shift.
    """
    N_sites = L ** 3
    # map (x,y,z) -> node index
    def node_index(x, y, z):
        return x * L * L + y * L + z

    idx_A = node_index(*pos_A)
    idx_B = node_index(*pos_B)

    V_sites = np.zeros(N_sites, dtype=np.float64)
    V_sites[idx_A] += trap_depth
    V_sites[idx_B] += trap_depth

    # Expand to spinful basis: each site has two spin states
    V_single = np.repeat(V_sites, 2)  # length 2*N_sites
    V_diag = cp.asarray(V_single, dtype=cp.complex128)

    H1_trapped = H1_base + cp.diag(V_diag)
    return H1_trapped


# ============================================================================
# Main experiment
# ============================================================================

def main():
    sub_cfg = CONFIG["substrate"]
    ex_cfg = CONFIG["exchange"]

    L = sub_cfg["L"]
    path_A = ex_cfg["path_A"]
    path_B = ex_cfg["path_B"]

    assert len(path_A) == len(path_B), "path_A and path_B must have same length."
    n_segments = len(path_A)

    trap_depth = float(ex_cfg["trap_depth"])
    steps_per_segment = int(ex_cfg["steps_per_segment"])
    dt = float(ex_cfg["dt"])

    spin_A = int(ex_cfg["spin_A"])
    spin_B = int(ex_cfg["spin_B"])

    print("============================================================")
    print("Two-Fermion Exchange Experiment")
    print("============================================================")
    print(f"Lattice size: L={L}")
    print(f"Segments in braid path: {n_segments}")
    print(f"Steps per segment: {steps_per_segment}, dt={dt}")
    print(f"Trap depth: {trap_depth}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # 1. Build and relax combined substrate geometry
    # ------------------------------------------------------------
    # Try the newer API (with plaquette_coupling and gauge_noise) first.
    # If the local CombinedSubstrate3D is an older version, fall back
    # to the simpler constructor without those extra keywords.
    try:
        substrate = CombinedSubstrate3D(
            L=L,
            eta_scalar=sub_cfg["eta_scalar"],
            decay_scalar=sub_cfg["decay_scalar"],
            eta_matrix=sub_cfg["eta_matrix"],
            decay_matrix=sub_cfg["decay_matrix"],
            E_link=sub_cfg["E_link"],
            plaquette_coupling=sub_cfg.get("plaquette_coupling", 0.0),
            gauge_noise=sub_cfg.get("gauge_noise", 0.0),
        )
    except TypeError:
        # Backwards-compatible path: older CombinedSubstrate3D
        substrate = CombinedSubstrate3D(
            L=L,
            eta_scalar=sub_cfg["eta_scalar"],
            decay_scalar=sub_cfg["decay_scalar"],
            eta_matrix=sub_cfg["eta_matrix"],
            decay_matrix=sub_cfg["decay_matrix"],
            E_link=sub_cfg["E_link"],
        )

    print("Relaxing combined substrate geometry...")
    _ = substrate.evolve(
        steps=sub_cfg["geom_relax_steps"],
        dt=sub_cfg["geom_dt"],
    )
    print("Geometry relaxation complete.")

    # ------------------------------------------------------------
    # 2. Single-particle Hamiltonian on this background
    # ------------------------------------------------------------
    H1_base = build_single_particle_hamiltonian(substrate)
    M = H1_base.shape[0]
    print(f"Single-particle dimension M = {M}")

    # ------------------------------------------------------------
    # 3. Build two-fermion sector for the *initial* Hamiltonian
    #    (we reuse the pair ordering across all segments)
    # ------------------------------------------------------------
    print("Constructing antisymmetric two-fermion sector (initial H1)...")
    base_sector = TwoFermionSector(H1_base)
    K = base_sector.K
    print(f"Two-fermion dimension K = {K}")

    # site map: single-particle index -> node index
    site_map = build_site_map(L)

    # Initial positions for the two fermions are the first elements of the paths
    pos_A0 = path_A[0]
    pos_B0 = path_B[0]

    a1 = single_particle_index(*pos_A0, spin_A, L)
    a2 = single_particle_index(*pos_B0, spin_B, L)
    print(f"Initial fermion A at {pos_A0}, spin={spin_A} -> a1={a1}")
    print(f"Initial fermion B at {pos_B0}, spin={spin_B} -> a2={a2}")

    psi0 = base_sector.make_localized_state(a1, a2)
    psi = psi0.copy()

    # Storage for diagnostics
    segment_times = []
    norms = []
    diag_probs = []
    overlaps_with_initial = []

    current_time = 0.0

    # ------------------------------------------------------------
    # 4. Loop over segments of the braid path
    # ------------------------------------------------------------
    for seg in range(n_segments):
        print(f"\nSegment {seg+1}/{n_segments}")
        pos_A_seg = path_A[seg]
        pos_B_seg = path_B[seg]
        print(f"  Traps at A={pos_A_seg}, B={pos_B_seg}")

        # Build H1 including traps for this segment
        H1_seg = add_trap_potential(
            H1_base, L, pos_A_seg, pos_B_seg, trap_depth
        )

        # Build new two-fermion sector for H1_seg, reusing the same
        # pair ordering implicitly (TwoFermionSector uses only M)
        sector_seg = TwoFermionSector(H1_seg)

        # Short-hand
        evals = sector_seg.evals
        evecs = sector_seg.evecs

        # Represent current psi in the eigenbasis of H2_seg
        c = evecs.conj().T @ psi  # shape (K,)

        # Evolve for steps_per_segment steps with constant H2_seg
        for step in range(steps_per_segment):
            # One time step
            phase = cp.exp(-1j * evals * dt)
            c = phase * c
            psi = evecs @ c
            current_time += dt

            # Diagnostics at every step
            norm = float(cp.linalg.norm(psi).get())
            p_diag = base_sector.diagonal_occupation_probability(
                psi, site_map
            )
            overlap_init = float(cp.vdot(psi0, psi).get())

            segment_times.append(current_time)
            norms.append(norm)
            diag_probs.append(p_diag)
            overlaps_with_initial.append(overlap_init)

        # end of segment loop

    # ------------------------------------------------------------
    # Final diagnostics
    # ------------------------------------------------------------
    segment_times = np.array(segment_times)
    norms = np.array(norms)
    diag_probs = np.array(diag_probs)
    overlaps_with_initial = np.array(overlaps_with_initial)

    print("\n============================================================")
    print("Exchange Experiment Summary")
    print("============================================================")
    print(f"Total evolution time: t = {current_time:.3f}")
    print(f"Norm: mean={norms.mean():.6f}, max|norm-1|={np.max(np.abs(norms-1)):.3e}")
    print(f"Diagonal occupation probability:")
    print(f"  max  = {diag_probs.max():.3e}")
    print(f"  mean = {diag_probs.mean():.3e}")
    print("Overlap with initial state Ψ(0):")
    print(f"  final overlap = {overlaps_with_initial[-1]:.6f}")
    print(f"  |final overlap| = {np.abs(overlaps_with_initial[-1]):.6f}")
    print("------------------------------------------------------------")
    print("For an ideal fermion exchange, we expect the final state to be")
    print("approximately -Ψ(0), so the overlap should be close to -1.")
    print("Diagonal occupation probability should remain near zero,")
    print("demonstrating Pauli exclusion throughout the braid.")
    print("============================================================")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    if ex_cfg.get("plot", True):
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))

        ax[0, 0].plot(segment_times, norms)
        ax[0, 0].set_title("Norm of Two-Fermion State")
        ax[0, 0].set_xlabel("t")
        ax[0, 0].set_ylabel("‖Ψ(t)‖")

        ax[0, 1].plot(segment_times, diag_probs)
        ax[0, 1].set_title("Diagonal Occupation Probability")
        ax[0, 1].set_xlabel("t")
        ax[0, 1].set_ylabel("P(both fermions on same site)")
        ax[0, 1].set_ylim(0, max(1e-3, diag_probs.max() * 1.2))

        ax[1, 0].plot(segment_times, overlaps_with_initial.real, label="Re ⟨Ψ(0)|Ψ(t)⟩")
        ax[1, 0].plot(segment_times, overlaps_with_initial.imag, label="Im ⟨Ψ(0)|Ψ(t)⟩")
        ax[1, 0].set_title("Overlap with Initial State")
        ax[1, 0].set_xlabel("t")
        ax[1, 0].set_ylabel("Overlap")
        ax[1, 0].legend()

        ax[1, 1].plot(segment_times, np.abs(overlaps_with_initial))
        ax[1, 1].set_title("|Overlap with Initial State|")
        ax[1, 1].set_xlabel("t")
        ax[1, 1].set_ylabel("|⟨Ψ(0)|Ψ(t)⟩|")
        ax[1, 1].set_ylim(0, 1.1)

        plt.tight_layout()
        plt.show()


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    main()
