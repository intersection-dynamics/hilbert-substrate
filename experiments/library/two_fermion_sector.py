"""
Two-Fermion Sector on the Combined Substrate
===========================================

This experiment script does the following:

1. Uses the CombinedSubstrate3D from substrate_three_levels.py
   to generate a 3D geometry with all three mechanisms active:
       - scalar feedback on t_ij
       - matrix (SU(2)) feedback on U_ij
       - living-link-inspired stiffness E_link

2. Extracts the single-particle Hamiltonian H1 from that geometry.

3. Builds the antisymmetric two-fermion sector:
       - Basis states |a,b> with a<b, where a,b label site×spin states.
       - Two-body Hamiltonian H2 = H1⊗I + I⊗H1, restricted to the
         antisymmetric subspace by construction.

4. Prepares an initial state with two localized fermions at distinct sites,
   evolves it under H2, and checks that the Pauli "hole" is preserved:
       - Probability of both fermions occupying the same site remains ~0.

This script is config-driven (no CLI) and assumes CuPy + GPU are available.
Place this file in the same directory as substrate_three_levels.py and run:

    c:\GitHub\hilbert_substrate>python experiments\two_fermion_sector.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make sure we can import CombinedSubstrate3D from substrate_three_levels.py
# regardless of whether we run from repo root or from experiments/.
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

try:
    from substrate_three_levels import CombinedSubstrate3D
except ImportError as e:
    raise ImportError(
        "Could not import CombinedSubstrate3D from substrate_three_levels.py. "
        "Please ensure that substrate_three_levels.py is in the same directory "
        "as this script."
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
    # Geometry / substrate config (three mechanisms combined)
    "substrate": {
        "L": 3,              # lattice size; keep small for 2-fermion sector
        "eta_scalar": 0.6,
        "decay_scalar": 0.05,
        "eta_matrix": 1.0,
        "decay_matrix": 0.1,
        "E_link": 1.0,
        "plaquette_coupling": 0.2,
        "gauge_noise": 0.001,
        "geom_relax_steps": 100,  # steps to relax geometry before snapshot
        "geom_dt": 0.05,
    },

    # Two-fermion sector config
    "two_fermion": {
        # Initial positions of the two fermions in lattice coordinates
        # (x, y, z) each in [0, L-1]
        "fermion_sites": [
            (0, 0, 0),
            (2, 2, 2),
        ],
        # Initial spin for each fermion: 0 = up, 1 = down
        "fermion_spins": [0, 0],

        # Time evolution parameters for the 2-fermion sector
        "total_time": 5.0,
        "n_time_steps": 100,

        # Whether to make plots
        "plot": True,
    },
}


# ============================================================================
# Helper: build single-particle Hamiltonian from combined substrate
# ============================================================================

def build_single_particle_hamiltonian(substrate: CombinedSubstrate3D) -> cp.ndarray:
    """
    Return the single-particle Hamiltonian H1 (shape MxM, complex128)
    from the given combined substrate.

    This simply delegates to substrate.hamiltonian(), which already
    encodes all three geometric mechanisms via t_links and U_links.

    The result is a CuPy array.
    """
    H1 = substrate.hamiltonian()
    # Ensure correct dtype
    H1 = H1.astype(cp.complex128)
    return H1


# ============================================================================
# Two-Fermion Sector Construction
# ============================================================================

class TwoFermionSector:
    """
    Antisymmetric two-fermion sector built from a single-particle Hamiltonian H1.

    Single-particle basis:
        a = 0..M-1, where M = 2 * N_sites (site × spin index).

    Two-fermion basis:
        |a,b>, a<b, with dimension K = M(M-1)/2.

    Hamiltonian:
        H2 = H1⊗I + I⊗H1, restricted to the antisymmetric basis by construction.
    """

    def __init__(self, H1: cp.ndarray):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for TwoFermionSector.")

        # Store single-particle Hamiltonian and its dimension
        self.H1_gpu = H1
        self.M = int(H1.shape[0])

        # Build antisymmetric basis: list of (a,b) with a<b,
        # and a mapping from (a,b) to basis index K.
        self.pairs = []
        self.pair_index = {}
        for a in range(self.M):
            for b in range(a + 1, self.M):
                k = len(self.pairs)
                self.pairs.append((a, b))
                self.pair_index[(a, b)] = k
        self.K = len(self.pairs)

        # Build H2 (on CPU for simplicity), then move to GPU
        self.H2_gpu = self._build_H2()

        # Diagonalize H2 once for efficient time evolution
        self.evals, self.evecs = cp.linalg.eigh(self.H2_gpu)

    def _build_H2(self) -> cp.ndarray:
        """
        Construct the antisymmetric 2-fermion Hamiltonian H2 from H1.

        H2 is built on CPU (NumPy) and then transferred to GPU.
        """
        H1_cpu = self.H1_gpu.get()
        M = self.M
        K = self.K

        H2_cpu = np.zeros((K, K), dtype=np.complex128)

        # Loop over basis states |a,b> with a<b
        for k, (a, b) in enumerate(self.pairs):
            # Moves from particle 1: a -> a'
            for a_prime in range(M):
                amp = H1_cpu[a_prime, a]
                if amp == 0:
                    continue
                if a_prime == b:
                    continue  # Pauli exclusion: cannot occupy same state
                if a_prime < b:
                    new_pair = (a_prime, b)
                    sign = 1.0
                else:
                    new_pair = (b, a_prime)
                    sign = -1.0
                k2 = self.pair_index[new_pair]
                H2_cpu[k2, k] += sign * amp

            # Moves from particle 2: b -> b'
            for b_prime in range(M):
                amp = H1_cpu[b_prime, b]
                if amp == 0:
                    continue
                if b_prime == a:
                    continue  # Pauli exclusion
                if a < b_prime:
                    new_pair = (a, b_prime)
                    sign = 1.0
                else:
                    new_pair = (b_prime, a)
                    sign = -1.0
                k2 = self.pair_index[new_pair]
                H2_cpu[k2, k] += sign * amp

        # Symmetrize to enforce Hermiticity
        H2_cpu = 0.5 * (H2_cpu + H2_cpu.conj().T)
        return cp.asarray(H2_cpu)

    # ----------------------------------------------------------------------

    def make_localized_state(self, site_spin1, site_spin2) -> cp.ndarray:
        """
        Construct an antisymmetric two-fermion state where
        fermion 1 is localized in state a1 and fermion 2 in state a2.

        site_spin1, site_spin2 are single-particle indices a in [0, M-1].
        """
        a1, a2 = site_spin1, site_spin2
        if a1 == a2:
            raise ValueError("Fermions cannot occupy the same single-particle state.")
        if a1 < a2:
            pair = (a1, a2)
            sign = 1.0
        else:
            pair = (a2, a1)
            sign = -1.0

        psi = cp.zeros(self.K, dtype=cp.complex128)
        k = self.pair_index[pair]
        # normalized antisymmetric state is already encoded in basis choice;
        # sign here accounts for original ordering of (a1,a2)
        psi[k] = sign
        # normalization (should already be 1)
        norm = cp.linalg.norm(psi)
        psi /= norm
        return psi

    # ----------------------------------------------------------------------

    def time_evolve(self, psi0: cp.ndarray, times: np.ndarray) -> cp.ndarray:
        """
        Time-evolve initial state psi0 for each t in times.

        Uses spectral decomposition: H2 = V diag(E) V^†.
        Returns an array psi_t of shape (len(times), K).
        """
        V = self.evecs
        E = self.evals
        # Project initial state into eigenbasis
        c0 = V.conj().T @ psi0  # shape (K,)

        psi_ts = []
        for t in times:
            phase = cp.exp(-1j * E * t)
            ct = phase * c0
            psi_t = V @ ct
            psi_ts.append(psi_t)
        psi_ts = cp.stack(psi_ts, axis=0)  # (T, K)
        return psi_ts

    # ----------------------------------------------------------------------

    def diagonal_occupation_probability(self, psi_t: cp.ndarray, site_map) -> float:
        """
        Compute the probability that both fermions occupy the same *spatial* site,
        summed over spin, for a given two-fermion state psi_t.

        site_map: array of length M mapping single-particle index a -> site index i.
                  (spin indices are distinguished by a but map to the same i.)

        Returns a scalar probability.
        """
        # For each spatial site i, sum amplitudes over spin combinations (i,s1), (i,s2).
        # For fermions, this should be ~0 for all t (Pauli hole).
        M = self.M
        K = self.K
        psi = psi_t  # shape (K,)

        prob = 0.0
        # Precompute all single-particle indices per site
        site_to_states = {}
        for a in range(M):
            i = site_map[a]
            site_to_states.setdefault(i, []).append(a)

        # For each site, consider all (a,b) pairs with same spatial site
        for i, states in site_to_states.items():
            S = states
            # pairs (a,b) with a<b and same site
            for idx_a in range(len(S)):
                for idx_b in range(idx_a + 1, len(S)):
                    a = S[idx_a]
                    b = S[idx_b]
                    # (a,b) appears as a basis pair if a<b; if not, swap
                    if a < b:
                        pair = (a, b)
                    else:
                        pair = (b, a)
                    k = self.pair_index[pair]
                    prob += float(cp.abs(psi[k]) ** 2)

        return prob


# ============================================================================
# Utility to map lattice (x,y,z,spin) to single-particle index a
# ============================================================================

def single_particle_index(x, y, z, s, L):
    """
    Map lattice coordinates (x,y,z) and spin s∈{0,1} to single-particle index a.
    This must match the ordering used in CombinedSubstrate3D.hamiltonian().
    There, spinor psi is shaped (N,2) with N=L^3, flattened as (N*2,).

    We take:
        node_index = x*L*L + y*L + z
        a = 2*node_index + s
    """
    node_idx = x * L * L + y * L + z
    return 2 * node_idx + s


def build_site_map(L):
    """
    Build site_map[a] = node_index for all single-particle states a.
    Used to detect both-fermions-on-same-site probability.
    """
    M = 2 * (L ** 3)
    site_map = np.zeros(M, dtype=np.int32)
    for x in range(L):
        for y in range(L):
            for z in range(L):
                node_idx = x * L * L + y * L + z
                for s in (0, 1):
                    a = 2 * node_idx + s
                    site_map[a] = node_idx
    return site_map


# ============================================================================
# Main experiment
# ============================================================================

def main():
    sub_cfg = CONFIG["substrate"]
    tf_cfg = CONFIG["two_fermion"]

    L = sub_cfg["L"]

    # --------------------------------------------------
    # 1. Build and relax the combined substrate geometry
    # --------------------------------------------------
    print("============================================================")
    print("Two-Fermion Sector on Combined Substrate")
    print("============================================================")
    print(f"Lattice size: L={L}")
    print("Building combined substrate and relaxing geometry...")

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

    # Relax geometry (but we ignore the geometry diagnostics here)
    _ = substrate.evolve(
        steps=sub_cfg["geom_relax_steps"],
        dt=sub_cfg["geom_dt"],
    )

    print("Geometry relaxed. Building single-particle Hamiltonian H1...")

    H1 = build_single_particle_hamiltonian(substrate)
    M = H1.shape[0]
    print(f"Single-particle dimension M = {M}")

    # --------------------------------------------------
    # 2. Build the antisymmetric two-fermion sector
    # --------------------------------------------------
    sector = TwoFermionSector(H1)
    K = sector.K
    print(f"Two-fermion antisymmetric dimension K = {K}")

    # --------------------------------------------------
    # 3. Initial state: two localized fermions
    # --------------------------------------------------
    sites = tf_cfg["fermion_sites"]
    spins = tf_cfg["fermion_spins"]
    if len(sites) != 2 or len(spins) != 2:
        raise ValueError("fermion_sites and fermion_spins must each have length 2.")

    (x1, y1, z1), (x2, y2, z2) = sites
    s1, s2 = spins

    a1 = single_particle_index(x1, y1, z1, s1, L)
    a2 = single_particle_index(x2, y2, z2, s2, L)
    print(f"Fermion 1 at (x,y,z,s)=({x1},{y1},{z1},{s1}) -> a1={a1}")
    print(f"Fermion 2 at (x,y,z,s)=({x2},{y2},{z2},{s2}) -> a2={a2}")

    psi0 = sector.make_localized_state(a1, a2)

    # --------------------------------------------------
    # 4. Time evolution and Pauli-hole diagnostic
    # --------------------------------------------------
    total_time = tf_cfg["total_time"]
    n_steps = tf_cfg["n_time_steps"]
    times = np.linspace(0.0, total_time, n_steps)

    print("Diagonalizing H2 and evolving two-fermion state...")
    psi_ts = sector.time_evolve(psi0, times)  # shape (T, K)

    # site_map: single-particle index a -> spatial node index i
    site_map = build_site_map(L)

    diag_probs = []
    norms = []
    for ti in range(n_steps):
        psi_t = psi_ts[ti]
        # Norm check
        norm = float(cp.linalg.norm(psi_t).get())
        norms.append(norm)
        # Diagonal occupation probability (both fermions on same site)
        p_diag = sector.diagonal_occupation_probability(psi_t, site_map)
        diag_probs.append(p_diag)

    norms = np.array(norms)
    diag_probs = np.array(diag_probs)

    print("------------------------------------------------------------")
    print("Two-Fermion Pauli-Hole Diagnostic")
    print("------------------------------------------------------------")
    print(f"Mean norm  = {norms.mean():.6f} (should be ~1)")
    print(f"Max |norm-1| over time = {np.max(np.abs(norms-1)):.3e}")
    print(f"Max diagonal occupation probability = {diag_probs.max():.3e}")
    print(f"Mean diagonal occupation probability = {diag_probs.mean():.3e}")
    print("------------------------------------------------------------")
    print("If the Pauli exclusion principle is correctly enforced,")
    print("the diagonal occupation probability should remain very close")
    print("to zero for all times (up to numerical error).")

    # --------------------------------------------------
    # 5. Plot (optional)
    # --------------------------------------------------
    if tf_cfg.get("plot", True):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].plot(times, norms)
        ax[0].set_title("Two-Fermion Norm vs Time")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("‖Ψ(t)‖")

        ax[1].plot(times, diag_probs)
        ax[1].set_title("Diagonal Occupation Probability")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("P(both fermions on same site)")
        ax[1].set_ylim(0, max(1e-3, diag_probs.max() * 1.2))

        plt.tight_layout()
        plt.show()


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    main()
