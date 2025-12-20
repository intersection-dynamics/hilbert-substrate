"""
Two-Fermion Exchange Experiment (v3): Fast vs Adiabatic
========================================================

This experiment compares two exchange protocols on the SAME relaxed
CombinedSubstrate3D geometry:

  1. A "fast" exchange:
       - shallow traps
       - few steps per segment
       - relatively violent motion

  2. A more "adiabatic" exchange:
       - deeper traps
       - many steps per segment
       - slower motion

Both protocols:

  - use the same 3x3x3 lattice,
  - start from the same relaxed gauge+scalar+stiffness background,
  - use the same antisymmetric two-fermion sector (same basis ordering),
  - move two traps around a 3x3 square in the x–y plane at z=1:

     path_A: (0,1,1) → (0,2,1) → (1,2,1) → (2,2,1) → (2,1,1)
     path_B: (2,1,1) → (2,0,1) → (1,0,1) → (0,0,1) → (0,1,1)

We track, for each protocol:

  - Norm  ‖Ψ(t)‖
  - Diagonal occupation probability (Pauli hole)
  - Overlaps:
      O0(t)    = ⟨Ψ(0)|Ψ(t)⟩
      Oswap(t) = ⟨Ψ_swap(0)|Ψ(t)⟩
    where Ψ_swap(0) is the antisymmetric state with the localized
    fermions swapped (A↔B).
  - Two-fermion energy ⟨H₂(t)⟩ under the *current* segment Hamiltonian.

The goal is to see that:

  - Pauli exclusion and norm conservation are robust in both regimes.
  - The more adiabatic protocol keeps |O0|, |Oswap| closer to 1 and
    keeps energy excursions smaller, while the fast protocol shakes
    the state into excited configurations (smaller |O|, more energy
    variance).

Configuration is driven purely by the CONFIG dict below.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import core substrate + fermion machinery
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from engine.substrate_and_fermions import (
    CombinedSubstrate3D,
    TwoFermionSector,
    single_particle_index,
    build_site_map,
)

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy import linalg as cpla  # noqa: F401  (kept for completeness)
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
    # Common substrate config
    "substrate": {
        "L": 3,                 # 3×3×3 lattice (N=27, M=54, K≈1431)
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

    # Fermion + path parameters shared by both protocols
    "paths": {
        # Paths for fermion A and B in x–y plane at fixed z=1
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
        "spin_A": 0,
        "spin_B": 0,
    },

    # Two protocols: "fast" vs "adiabatic"
    "protocols": {
        "fast": {
            "label": "Fast (non-adiabatic)",
            "trap_depth": -5.0,
            "steps_per_segment": 10,
            "dt": 0.05,
        },
        "adiabatic": {
            "label": "Adiabatic-ish",
            "trap_depth": -20.0,
            "steps_per_segment": 50,
            "dt": 0.05,
        },
    },

    # Plotting toggle
    "plot": True,
}


# ============================================================================
# Helper: add a diagonal steering potential for two traps
# ============================================================================

def add_trap_potential(H1_base: cp.ndarray,
                       L: int,
                       pos_A,
                       pos_B,
                       trap_depth: float) -> cp.ndarray:
    """
    Construct H1 = H1_base + V, where V is a diagonal single-particle potential
    that creates wells of depth trap_depth at two lattice sites pos_A and pos_B.

    Potential is spin-independent: both spin components at a site get
    the same energy shift.
    """
    N_sites = L ** 3

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
# Core: run one exchange protocol
# ============================================================================

def run_exchange_protocol(
    name: str,
    label: str,
    H1_base: cp.ndarray,
    L: int,
    path_A,
    path_B,
    spin_A: int,
    spin_B: int,
    trap_depth: float,
    steps_per_segment: int,
    dt: float,
):
    """
    Run one exchange protocol (e.g., 'fast' or 'adiabatic') on a *fixed*
    background H1_base.

    Returns:
        results: dict with numpy arrays for:
            times, norms, diag_probs, energies,
            overlaps_initial, overlaps_swapped
    """
    print(f"\n============================================================")
    print(f"Protocol: {name}  ({label})")
    print("============================================================")
    print(f"Trap depth: {trap_depth}")
    print(f"Steps per segment: {steps_per_segment}, dt={dt}")
    print("------------------------------------------------------------")

    # Build reference two-fermion sector (basis & antisymmetry)
    sector_ref = TwoFermionSector(H1_base)
    K = sector_ref.K
    M = sector_ref.M
    print(f"Reference two-fermion sector: K={K}, M={M}")

    site_map = build_site_map(L)

    # Initial positions (first elements in paths)
    pos_A0 = path_A[0]
    pos_B0 = path_B[0]

    a1 = single_particle_index(*pos_A0, spin_A, L)
    a2 = single_particle_index(*pos_B0, spin_B, L)
    print(f"Initial fermion A at {pos_A0}, spin={spin_A} -> a1={a1}")
    print(f"Initial fermion B at {pos_B0}, spin={spin_B} -> a2={a2}")

    # Ψ(0) and Ψ_swap(0) in antisymmetric sector
    psi0 = sector_ref.make_localized_state(a1, a2)
    psi_swap0 = sector_ref.make_localized_state(a2, a1)

    # Current state
    psi = psi0.copy()

    # Storage
    times = []
    norms = []
    diag_probs = []
    energies = []
    overlaps_initial = []
    overlaps_swapped = []

    current_time = 0.0
    n_segments = len(path_A)
    assert len(path_B) == n_segments, "path_A and path_B must have same length."

    # Loop over segments
    for seg in range(n_segments):
        pos_A_seg = path_A[seg]
        pos_B_seg = path_B[seg]
        print(f"\nSegment {seg + 1}/{n_segments}")
        print(f"  Traps at A={pos_A_seg}, B={pos_B_seg}")

        # Build H1 including traps for this segment
        H1_seg = add_trap_potential(H1_base, L, pos_A_seg, pos_B_seg, trap_depth)

        # Two-fermion sector for this segment (same M→same K and basis ordering)
        sector_seg = TwoFermionSector(H1_seg)
        assert sector_seg.K == K, "Two-fermion dimension mismatch."

        evals = sector_seg.evals
        evecs = sector_seg.evecs
        H2_seg = sector_seg.H2_gpu

        # Represent psi in eigenbasis of H2_seg
        c = evecs.conj().T @ psi

        # Evolve for steps_per_segment microsteps
        for _ in range(steps_per_segment):
            phase = cp.exp(-1j * evals * dt)
            c = phase * c
            psi = evecs @ c
            current_time += dt

            # Diagnostics
            norm = float(cp.linalg.norm(psi).get())
            p_diag = sector_ref.diagonal_occupation_probability(psi, site_map)
            overlap_init = cp.vdot(psi0, psi).get()
            overlap_swap = cp.vdot(psi_swap0, psi).get()

            # Energy expectation ⟨H2⟩
            H2_psi = H2_seg @ psi
            E_val = cp.vdot(psi, H2_psi).real.get()
            E_val = float(E_val)

            times.append(current_time)
            norms.append(norm)
            diag_probs.append(p_diag)
            energies.append(E_val)
            overlaps_initial.append(overlap_init)
            overlaps_swapped.append(overlap_swap)

    # Pack into numpy arrays
    times = np.array(times, dtype=float)
    norms = np.array(norms, dtype=float)
    diag_probs = np.array(diag_probs, dtype=float)
    energies = np.array(energies, dtype=float)
    overlaps_initial = np.array(overlaps_initial, dtype=np.complex128)
    overlaps_swapped = np.array(overlaps_swapped, dtype=np.complex128)

    # Final overlaps
    O0_final = overlaps_initial[-1]
    Os_final = overlaps_swapped[-1]

    print("\nSummary for protocol:", name)
    print("------------------------------------------------------------")
    print(f"Total evolution time: t = {current_time:.3f}")
    print(f"Norm: mean={norms.mean():.6f}, max|norm-1|={np.max(np.abs(norms - 1.0)):.3e}")
    print("Diagonal occupation probability (Pauli hole):")
    print(f"  max  = {diag_probs.max():.3e}")
    print(f"  mean = {diag_probs.mean():.3e}")
    print("Energy ⟨H₂⟩:")
    print(f"  min  = {energies.min():.6f}")
    print(f"  max  = {energies.max():.6f}")
    print(f"  mean = {energies.mean():.6f}")
    print("Overlap with initial state Ψ(0):")
    print(f"  O0_final    = {O0_final.real:+.6f} {O0_final.imag:+.6f}i")
    print(f"  |O0_final|  = {np.abs(O0_final):.6f}")
    print("Overlap with swapped initial state Ψ_swap(0):")
    print(f"  Oswap_final   = {Os_final.real:+.6f} {Os_final.imag:+.6f}i")
    print(f"  |Oswap_final| = {np.abs(Os_final):.6f}")
    print("------------------------------------------------------------")

    return {
        "name": name,
        "label": label,
        "times": times,
        "norms": norms,
        "diag_probs": diag_probs,
        "energies": energies,
        "overlaps_initial": overlaps_initial,
        "overlaps_swapped": overlaps_swapped,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    sub_cfg = CONFIG["substrate"]
    paths_cfg = CONFIG["paths"]
    protos_cfg = CONFIG["protocols"]

    L = sub_cfg["L"]
    path_A = paths_cfg["path_A"]
    path_B = paths_cfg["path_B"]
    spin_A = int(paths_cfg["spin_A"])
    spin_B = int(paths_cfg["spin_B"])

    print("============================================================")
    print("Two-Fermion Exchange Experiment (v3): Fast vs Adiabatic")
    print("============================================================")
    print(f"Lattice size: L={L}")
    print(f"Number of path segments: {len(path_A)}")
    print(f"Fermion spins: A={spin_A}, B={spin_B}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # 1. Build and relax common substrate geometry
    # ------------------------------------------------------------
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

    print("Relaxing combined substrate geometry...")
    _ = substrate.evolve(
        steps=sub_cfg["relax_steps"],
        dt=sub_cfg["relax_dt"],
        record_history=False,
    )
    print("Geometry relaxation complete.")

    H1_base = substrate.hamiltonian()
    M = H1_base.shape[0]
    print(f"Single-particle dimension M = {M}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # 2. Run both protocols on the same H1_base
    # ------------------------------------------------------------
    results = {}

    for name, cfg in protos_cfg.items():
        res = run_exchange_protocol(
            name=name,
            label=cfg["label"],
            H1_base=H1_base,
            L=L,
            path_A=path_A,
            path_B=path_B,
            spin_A=spin_A,
            spin_B=spin_B,
            trap_depth=cfg["trap_depth"],
            steps_per_segment=cfg["steps_per_segment"],
            dt=cfg["dt"],
        )
        results[name] = res

    # ------------------------------------------------------------
    # 3. Combined plots
    # ------------------------------------------------------------
    if CONFIG.get("plot", True):
        # Use a consistent time axis: each protocol has its own times
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))

        colors = {
            "fast": "tab:red",
            "adiabatic": "tab:blue",
        }

        # Norms
        for name, res in results.items():
            ax[0, 0].plot(res["times"], res["norms"],
                          label=res["label"],
                          color=colors.get(name, None))
        ax[0, 0].set_title("Norm of Two-Fermion State")
        ax[0, 0].set_xlabel("t")
        ax[0, 0].set_ylabel("‖Ψ(t)‖")
        ax[0, 0].legend()

        # Diagonal occupation
        for name, res in results.items():
            ax[0, 1].plot(res["times"], res["diag_probs"],
                          label=res["label"],
                          color=colors.get(name, None))
        ax[0, 1].set_title("Diagonal Occupation Probability")
        ax[0, 1].set_xlabel("t")
        ax[0, 1].set_ylabel("P(both fermions on same site)")
        all_diag = np.concatenate([results[k]["diag_probs"] for k in results])
        ax[0, 1].set_ylim(0, max(1e-3, all_diag.max() * 1.2))
        ax[0, 1].legend()

        # Energy
        for name, res in results.items():
            ax[0, 2].plot(res["times"], res["energies"],
                          label=res["label"],
                          color=colors.get(name, None))
        ax[0, 2].set_title("Two-Fermion Energy ⟨H₂⟩")
        ax[0, 2].set_xlabel("t")
        ax[0, 2].set_ylabel("Energy")
        ax[0, 2].legend()

        # |O0(t)|
        for name, res in results.items():
            ax[1, 0].plot(res["times"],
                          np.abs(res["overlaps_initial"]),
                          label=res["label"],
                          color=colors.get(name, None))
        ax[1, 0].set_title("|⟨Ψ(0)|Ψ(t)⟩|")
        ax[1, 0].set_xlabel("t")
        ax[1, 0].set_ylabel("|O0(t)|")
        ax[1, 0].set_ylim(0, 1.1)
        ax[1, 0].legend()

        # |Oswap(t)|
        for name, res in results.items():
            ax[1, 1].plot(res["times"],
                          np.abs(res["overlaps_swapped"]),
                          label=res["label"],
                          color=colors.get(name, None))
        ax[1, 1].set_title("|⟨Ψ_swap(0)|Ψ(t)⟩|")
        ax[1, 1].set_xlabel("t")
        ax[1, 1].set_ylabel("|Oswap(t)|")
        ax[1, 1].set_ylim(0, 1.1)
        ax[1, 1].legend()

        # Overlap phases (arg O0)
        for name, res in results.items():
            O0 = res["overlaps_initial"]
            phases = np.angle(O0)
            ax[1, 2].plot(res["times"], phases,
                          label=res["label"],
                          color=colors.get(name, None))
        ax[1, 2].set_title("Phase arg ⟨Ψ(0)|Ψ(t)⟩")
        ax[1, 2].set_xlabel("t")
        ax[1, 2].set_ylabel("Phase (rad)")
        ax[1, 2].legend()

        plt.tight_layout()
        plt.show()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()
