"""
Hilbert Substrate Core: Geometry + Fermions
===========================================

This module defines two core components:

1) CombinedSubstrate3D
   --------------------
   A 3D lattice substrate carrying:

     - Scalar link amplitudes t_ij  (real, scalar feedback)
     - SU(2) matrix link variables U_ij  (matrix feedback)
     - A stiffness scale E_link     (living-link inspired)

   The single-particle Hamiltonian is:

       H_ij = - t_ij U_ij

   acting on a 2-component spinor ψ_i on each site i.

   The geometry evolves via:

     - Scalar feedback:
         dt_ij/dt = η_s |ψ_i||ψ_j| - γ_s (t_ij - 1)

     - Matrix feedback (schematic):
         dU_ij/dt = η_m (traceless |ψ_i><ψ_j|)
                    - (γ_m + E_link)(U_ij - I_2)
                    - κ (P_ij - I_2)
                    + noise

       followed by projection back to SU(2).

   Here P_ij is a simple plaquette matrix containing link (i,j).
   All evolution is done on the GPU using CuPy.

2) TwoFermionSector
   -----------------
   An antisymmetric two-fermion sector built from a single-particle
   Hamiltonian H1 (e.g., from CombinedSubstrate3D).

   Single-particle basis indices: a = 0..M-1.

   Two-fermion basis states: |a,b> with a<b, dimension K = M(M-1)/2.

   The Hamiltonian is:

       H2 = H1 ⊗ I + I ⊗ H1

   restricted to the antisymmetric subspace by construction.

   This sector provides:

     - make_localized_state(a1, a2): an antisymmetric 2-fermion state
     - time_evolve(psi0, times): exact evolution using spectral decomposition
     - diagonal_occupation_probability(psi_t, site_map): probability that
       both fermions occupy the same spatial site (Pauli-hole diagnostic).

Config and Usage
----------------

This file is meant to be imported by experiment scripts, e.g.:

    from core.substrate_and_fermions import CombinedSubstrate3D, TwoFermionSector

The CONFIG block at the top provides a reference set of parameters.

Running this module directly (python substrate_and_fermions.py)
performs a small built-in sanity check:

  - builds a small substrate,
  - relaxes the geometry for a few steps,
  - builds H1,
  - builds a TwoFermionSector,
  - prepares two localized fermions,
  - evolves for a few time steps,
  - prints norm and Pauli-hole diagnostics.

No command-line arguments are used; everything is config-driven.
"""

import numpy as np

# ---------------------------------------------------------------------------
# GPU imports
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from cupyx.scipy import linalg as cpla
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    cpla = None
    GPU_AVAILABLE = False
    raise RuntimeError("CuPy is required for this module.")


# ============================================================================
# GLOBAL CONFIG (REFERENCE VALUES)
# ============================================================================

CONFIG = {
    "substrate": {
        "L": 4,
        "eta_scalar": 0.6,
        "decay_scalar": 0.05,
        "eta_matrix": 1.0,
        "decay_matrix": 0.1,
        "E_link": 1.0,
        "plaquette_coupling": 0.2,
        "gauge_noise": 0.001,
        "relax_steps": 40,
        "relax_dt": 0.05,
    },
    "fermions": {
        # Initial fermion positions (x,y,z) and spins s∈{0,1}
        "fermion1": {"pos": (0, 0, 0), "spin": 0},
        "fermion2": {"pos": (3, 3, 3), "spin": 0},
        "total_time": 1.0,
        "n_time_steps": 20,
    },
}


# ============================================================================
# SU(2) helpers
# ============================================================================

# Pauli matrices and identity on the GPU
SIGMA_X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
SIGMA_Y = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
SIGMA_Z = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)
I2 = cp.eye(2, dtype=cp.complex128)


def project_to_su2(U: "cp.ndarray") -> "cp.ndarray":
    """
    Project a 2x2 complex matrix onto SU(2) using polar decomposition.

    Steps:
        1. H = U† U
        2. H^{-1/2} via eigen-decomposition
        3. U_unitary = U H^{-1/2}
        4. Rescale to det(U_unitary) = 1
    """
    H = U.conj().T @ U
    evals, evecs = cp.linalg.eigh(H)
    evals = cp.maximum(evals, 1e-12)
    inv_sqrt = evecs @ cp.diag(1.0 / cp.sqrt(evals)) @ evecs.conj().T
    Uu = U @ inv_sqrt

    det = cp.linalg.det(Uu)
    det = cp.where(cp.abs(det) < 1e-12, 1.0 + 0j, det)
    Uu = Uu / cp.sqrt(det)
    return Uu


# ============================================================================
# 1. Combined 3D Substrate (scalar + SU(2) + stiffness)
# ============================================================================

class CombinedSubstrate3D:
    """
    Combined 3D substrate carrying scalar, matrix, and stiffness mechanisms.

    Sites:
        - 3D lattice of size L^3.
        - At each site i: two-component spinor ψ_i.

    Links:
        - For each undirected nearest-neighbor pair (i,j):
            t_ij  (real scalar)
            U_ij  (2x2 SU(2) matrix)

    Single-particle Hamiltonian:
        H_ij = - t_ij U_ij

    Geometry evolution per time step dt:
        - Scalar feedback on t_ij from |ψ_i||ψ_j|.
        - Matrix feedback on U_ij from spinor correlators.
        - Stiffness term ~ E_link (U_ij - I_2).
        - Plaquette coupling and small gauge noise (optional).
    """

    def __init__(
        self,
        L: int,
        eta_scalar: float,
        decay_scalar: float,
        eta_matrix: float,
        decay_matrix: float,
        E_link: float,
        plaquette_coupling: float = 0.0,
        gauge_noise: float = 0.0,
    ):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for CombinedSubstrate3D.")

        self.L = int(L)
        self.N = self.L ** 3

        self.eta_scalar = float(eta_scalar)
        self.decay_scalar = float(decay_scalar)

        self.eta_matrix = float(eta_matrix)
        self.decay_matrix = float(decay_matrix)
        self.E_link = float(E_link)
        self.kappa = float(plaquette_coupling)
        self.noise = float(gauge_noise)

        # Undirected neighbor list: store each link once with i<j
        self.links = []          # list of (i, j)
        self.t_links = {}        # (i, j) -> scalar t_ij
        self.U_links = {}        # (i, j) -> 2x2 cp.ndarray (SU(2))
        self._build_links()

        # Initial spinor field: Gaussian lump at center, spin up
        self.psi = self._init_spinor_wavepacket()

    # ----- lattice helpers -----

    def _idx(self, x: int, y: int, z: int) -> int:
        return x * self.L * self.L + y * self.L + z

    def _coord(self, idx: int):
        x = idx // (self.L * self.L)
        y = (idx // self.L) % self.L
        z = idx % self.L
        return x, y, z

    def _move(self, idx: int, dx: int, dy: int, dz: int) -> int:
        x, y, z = self._coord(idx)
        nx = (x + dx) % self.L
        ny = (y + dy) % self.L
        nz = (z + dz) % self.L
        return self._idx(nx, ny, nz)

    def _build_links(self):
        """
        Build undirected nearest-neighbor links. For each pair (i,j) with i<j,
        initialize t_ij = 1.0 and U_ij = identity.
        """
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    i = self._idx(x, y, z)
                    # 6 neighbors (x±1, y±1, z±1) with periodic BCs
                    for dx, dy, dz in [
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ]:
                        j = self._move(i, dx, dy, dz)
                        if i < j and (i, j) not in self.t_links:
                            self.links.append((i, j))
                            self.t_links[(i, j)] = 1.0
                            self.U_links[(i, j)] = I2.copy()

    def _init_spinor_wavepacket(self) -> "cp.ndarray":
        """
        Initialize ψ as a normalized Gaussian centered in the lattice, spin up.
        Shape: (2N,) (flattened).
        """
        psi_host = np.zeros((self.N, 2), dtype=np.complex128)
        c = self.L // 2
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._idx(x, y, z)
                    r2 = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
                    amp = np.exp(-r2 / 2.0)
                    psi_host[idx, 0] = amp  # spin up
                    psi_host[idx, 1] = 0.0  # spin down
        flat = psi_host.reshape(-1)
        flat /= np.linalg.norm(flat)
        return cp.asarray(flat)

    # ----- link access with orientation -----

    def _get_U(self, i: int, j: int) -> "cp.ndarray":
        """
        Return the 2x2 matrix U_ij for oriented link i->j.

        Internally we store only (i,j) with i<j.
        For j<i we return U_ji†.
        """
        if i < j:
            return self.U_links[(i, j)]
        elif j < i:
            return self.U_links[(j, i)].conj().T
        else:
            return I2

    def _get_t(self, i: int, j: int) -> float:
        """
        Return the scalar t_ij for undirected link {i,j}.
        """
        if i < j:
            return self.t_links[(i, j)]
        elif j < i:
            return self.t_links[(j, i)]
        else:
            return 0.0

    # ----- Hamiltonian and plaquette -----

    def hamiltonian(self) -> "cp.ndarray":
        """
        Build the single-particle Hamiltonian H of shape (2N, 2N).

        For each undirected link (i,j), we add:

            H_ij = - t_ij U_ij
            H_ji = - t_ij U_ij†
        """
        size = 2 * self.N
        H = cp.zeros((size, size), dtype=cp.complex128)
        for (i, j) in self.links:
            t_ij = self.t_links[(i, j)]
            U_ij = self.U_links[(i, j)]
            # i -> j
            H[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] -= t_ij * U_ij
            # j -> i (Hermitian conjugate)
            H[2 * j : 2 * j + 2, 2 * i : 2 * i + 2] -= t_ij * U_ij.conj().T
        return H

    def _plaquette_matrix(self, i: int, j: int) -> "cp.ndarray":
        """
        Return a simple plaquette matrix P_ij around the link (i,j).

        We choose one perpendicular plaquette depending on the direction
        of (i,j). If construction fails (e.g., missing links), returns I2.
        """
        x, y, z = self._coord(i)
        xv, yv, zv = self._coord(j)
        dx = (xv - x) % self.L
        dy = (yv - y) % self.L
        dz = (zv - z) % self.L

        if dx == self.L - 1:
            dx = -1
        if dy == self.L - 1:
            dy = -1
        if dz == self.L - 1:
            dz = -1

        # Choose a perpendicular direction and build a 4-link loop
        try:
            if dx != 0:
                # Link along x, plaquette in x-y plane
                a = self._move(i, 0, 1, 0)
                b = self._move(j, 0, 1, 0)
            elif dy != 0:
                # Link along y, plaquette in y-z plane
                a = self._move(i, 0, 0, 1)
                b = self._move(j, 0, 0, 1)
            else:
                # Link along z, plaquette in z-x plane
                a = self._move(i, 1, 0, 0)
                b = self._move(j, 1, 0, 0)

            Uia = self._get_U(i, a)
            Uab = self._get_U(a, b)
            Ubj = self._get_U(b, j)
            Uji = self._get_U(j, i)
            return Uia @ Uab @ Ubj @ Uji
        except KeyError:
            return I2

    # ----- evolution -----

    def evolve(self, steps: int, dt: float, record_history: bool = False):
        """
        Evolve the spinor field and geometry for a given number of steps.

        Args:
            steps: number of time steps (integer).
            dt:    time step (float).
            record_history: if True, returns a dict with observables
                            recorded at each step; otherwise returns None.

        Observables stored (if record_history):
            - energy
            - participation_ratio
            - mean_t
            - gauge_deviation  (mean ||U_ij - I||)
            - gauge_unit_error (mean ||U_ij† U_ij - I||)
        """
        steps = int(steps)
        dt = float(dt)

        history = None
        if record_history:
            history = {
                "energy": [],
                "participation_ratio": [],
                "mean_t": [],
                "gauge_deviation": [],
                "gauge_unit_error": [],
            }

        for _ in range(steps):
            # Build H and evolve ψ
            H = self.hamiltonian()
            Uop = cpla.expm(-1j * H * dt)
            self.psi = Uop @ self.psi

            # Reshape to (N,2) and get magnitudes
            psi_spinors = self.psi.reshape(self.N, 2)
            psi_mag = cp.sqrt(cp.sum(cp.abs(psi_spinors) ** 2, axis=1)).get()

            # --- scalar feedback on t_ij ---
            new_t = {}
            for (i, j) in self.links:
                corr = psi_mag[i] * psi_mag[j]
                t_val = self.t_links[(i, j)]
                dt_val = (
                    self.eta_scalar * corr
                    - self.decay_scalar * (t_val - 1.0)
                ) * dt
                new_t[(i, j)] = t_val + dt_val
            self.t_links = new_t

            # --- matrix feedback on U_ij ---
            for (i, j) in self.links:
                P_i = psi_spinors[i]
                P_j = psi_spinors[j]
                # rank-1 correlation
                corr = cp.outer(P_i, cp.conj(P_j))
                tr_corr = cp.trace(corr) / 2.0
                corr_su2 = corr - tr_corr * I2

                U_curr = self.U_links[(i, j)]

                # Plaquette term
                if self.kappa != 0.0:
                    P = self._plaquette_matrix(i, j)
                    delta_plaq = -self.kappa * (P - I2)
                else:
                    delta_plaq = 0.0 * U_curr

                # Gauge noise (small random SU(2) generator)
                if self.noise > 0.0:
                    xi = cp.random.normal(0.0, 1.0, 3)
                    Hnoise = xi[0] * SIGMA_X + xi[1] * SIGMA_Y + xi[2] * SIGMA_Z
                    delta_noise = 1j * self.noise * Hnoise @ U_curr
                else:
                    delta_noise = 0.0 * U_curr

                delta = (
                    self.eta_matrix * corr_su2
                    - (self.decay_matrix + self.E_link) * (U_curr - I2)
                    + delta_plaq
                ) * dt + delta_noise * np.sqrt(dt)

                U_trial = U_curr + delta
                U_new = project_to_su2(U_trial)
                self.U_links[(i, j)] = U_new

            # --- record observables if needed ---
            if record_history:
                # Energy and participation ratio
                E = float(cp.vdot(self.psi, H @ self.psi).real.get())
                PR = float((1.0 / cp.sum(cp.abs(self.psi) ** 4)).get())

                # Mean t
                mean_t = float(np.mean(list(self.t_links.values())))

                # Gauge diagnostics
                dev_sum = 0.0
                unit_sum = 0.0
                count = 0
                for (i, j), U in self.U_links.items():
                    diff_I = U - I2
                    dev_sum += float(cp.linalg.norm(diff_I).get())
                    UU = U.conj().T @ U
                    unit_diff = UU - I2
                    unit_sum += float(cp.linalg.norm(unit_diff).get())
                    count += 1
                gauge_dev = dev_sum / max(count, 1)
                gauge_unit_err = unit_sum / max(count, 1)

                history["energy"].append(E)
                history["participation_ratio"].append(PR)
                history["mean_t"].append(mean_t)
                history["gauge_deviation"].append(gauge_dev)
                history["gauge_unit_error"].append(gauge_unit_err)

        return history


# ============================================================================
# 2. Two-Fermion Sector (antisymmetric)
# ============================================================================

class TwoFermionSector:
    """
    Antisymmetric two-fermion sector built from a single-particle Hamiltonian H1.

    Single-particle basis indices:
        a = 0..M-1

    Two-fermion basis states:
        |a,b> with a<b

    Dimension:
        K = M(M-1)/2

    Hamiltonian:
        H2 = H1 ⊗ I + I ⊗ H1

    restricted to the antisymmetric subspace by construction.

    Methods:
        - make_localized_state(a1, a2)
        - time_evolve(psi0, times)
        - diagonal_occupation_probability(psi_t, site_map)
    """

    def __init__(self, H1: "cp.ndarray"):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for TwoFermionSector.")

        self.H1_gpu = H1.astype(cp.complex128)
        self.M = int(H1.shape[0])

        # Build antisymmetric basis: list of (a,b) with a<b
        self.pairs = []
        self.pair_index = {}
        for a in range(self.M):
            for b in range(a + 1, self.M):
                k = len(self.pairs)
                self.pairs.append((a, b))
                self.pair_index[(a, b)] = k
        self.K = len(self.pairs)

        # Build H2 on CPU, then move to GPU and diagonalize
        self.H2_gpu = self._build_H2()
        self.evals, self.evecs = cp.linalg.eigh(self.H2_gpu)

    def _build_H2(self) -> "cp.ndarray":
        """
        Construct the antisymmetric 2-fermion Hamiltonian H2 from H1.

        H2 is constructed on the CPU (NumPy) and then transferred to GPU.
        """
        H1_cpu = self.H1_gpu.get()
        M = self.M
        K = self.K

        H2_cpu = np.zeros((K, K), dtype=np.complex128)

        for k, (a, b) in enumerate(self.pairs):
            # Moves from particle 1: a -> a'
            for a_prime in range(M):
                amp = H1_cpu[a_prime, a]
                if amp == 0:
                    continue
                if a_prime == b:
                    continue
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
                    continue
                if a < b_prime:
                    new_pair = (a, b_prime)
                    sign = 1.0
                else:
                    new_pair = (b_prime, a)
                    sign = -1.0
                k2 = self.pair_index[new_pair]
                H2_cpu[k2, k] += sign * amp

        H2_cpu = 0.5 * (H2_cpu + H2_cpu.conj().T)
        return cp.asarray(H2_cpu)

    # ------------------------------------------------------------------

    def make_localized_state(self, a1: int, a2: int) -> "cp.ndarray":
        """
        Construct an antisymmetric two-fermion state with fermions
        in single-particle states a1 and a2.

        a1, a2 must be distinct integers in [0, M-1].
        """
        a1 = int(a1)
        a2 = int(a2)
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
        psi[k] = sign
        norm = cp.linalg.norm(psi)
        psi /= norm
        return psi

    # ------------------------------------------------------------------

    def time_evolve(self, psi0: "cp.ndarray", times: np.ndarray) -> "cp.ndarray":
        """
        Time-evolve the initial state psi0 for each t in times.

        Uses spectral decomposition: H2 = V diag(E) V^†.

        Returns:
            psi_ts: CuPy array of shape (T, K),
                    where psi_ts[t_index] is the state at times[t_index].
        """
        V = self.evecs
        E = self.evals
        c0 = V.conj().T @ psi0  # coefficients in eigenbasis

        psi_ts = []
        for t in times:
            phase = cp.exp(-1j * E * t)
            ct = phase * c0
            psi_t = V @ ct
            psi_ts.append(psi_t)
        psi_ts = cp.stack(psi_ts, axis=0)
        return psi_ts

    # ------------------------------------------------------------------

    def diagonal_occupation_probability(
        self,
        psi_t: "cp.ndarray",
        site_map: np.ndarray,
    ) -> float:
        """
        Compute the probability that both fermions occupy the same spatial site,
        summed over spin, for a given two-fermion state psi_t.

        site_map: array of length M mapping single-particle index a -> site index i.
        """
        psi = psi_t  # shape (K,)

        # Precompute single-particle indices per site
        M = self.M
        site_to_states = {}
        for a in range(M):
            i = int(site_map[a])
            site_to_states.setdefault(i, []).append(a)

        prob = 0.0
        for i, states in site_to_states.items():
            S = states
            for idx_a in range(len(S)):
                for idx_b in range(idx_a + 1, len(S)):
                    a = S[idx_a]
                    b = S[idx_b]
                    if a < b:
                        pair = (a, b)
                    else:
                        pair = (b, a)
                    k = self.pair_index[pair]
                    prob += float(cp.abs(psi[k]) ** 2)
        return prob


# ============================================================================
# Utility functions for mapping between lattice and single-particle indices
# ============================================================================

def single_particle_index(x: int, y: int, z: int, s: int, L: int) -> int:
    """
    Map lattice coordinates (x,y,z) and spin s∈{0,1} to a single-particle index a.

    Convention:
        node_index = x * L^2 + y * L + z
        a = 2 * node_index + s
    """
    node_idx = x * L * L + y * L + z
    return 2 * node_idx + s


def build_site_map(L: int) -> np.ndarray:
    """
    Build site_map[a] = node_index for all single-particle states a,
    with node_index in [0, L^3-1].
    """
    N = L ** 3
    M = 2 * N
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
# Built-in sanity check (executed when run directly)
# ============================================================================

def _sanity_check():
    """
    Small built-in check:

    1) Build a small CombinedSubstrate3D using CONFIG["substrate"].
    2) Relax geometry for a few steps.
    3) Build H1 from the relaxed geometry.
    4) Build a TwoFermionSector on H1.
    5) Prepare two localized fermions at distant sites.
    6) Evolve for a short time and print:
         - norm of the two-fermion state
         - diagonal occupation probability (Pauli-hole diagnostic)
    """
    sub_cfg = CONFIG["substrate"]
    f_cfg = CONFIG["fermions"]

    L = sub_cfg["L"]
    print("============================================================")
    print("Hilbert Substrate Core: Sanity Check")
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

    _ = substrate.evolve(
        steps=sub_cfg["relax_steps"],
        dt=sub_cfg["relax_dt"],
        record_history=False,
    )
    print("Geometry relaxation complete.")

    # Single-particle Hamiltonian
    H1 = substrate.hamiltonian()
    M = H1.shape[0]
    print(f"Single-particle dimension M = {M}")

    # Two-fermion sector
    sector = TwoFermionSector(H1)
    K = sector.K
    print(f"Two-fermion dimension K = {K}")

    # Initial localized fermions
    pos1 = f_cfg["fermion1"]["pos"]
    pos2 = f_cfg["fermion2"]["pos"]
    s1 = f_cfg["fermion1"]["spin"]
    s2 = f_cfg["fermion2"]["spin"]

    a1 = single_particle_index(*pos1, s1, L)
    a2 = single_particle_index(*pos2, s2, L)
    print(f"Fermion 1 at {pos1}, spin={s1} -> a1={a1}")
    print(f"Fermion 2 at {pos2}, spin={s2} -> a2={a2}")

    psi0 = sector.make_localized_state(a1, a2)

    # Time evolution
    total_time = f_cfg["total_time"]
    n_steps = f_cfg["n_time_steps"]
    times = np.linspace(0.0, total_time, n_steps)

    psi_ts = sector.time_evolve(psi0, times)
    site_map = build_site_map(L)

    norms = []
    diag_probs = []
    for ti in range(n_steps):
        psi_t = psi_ts[ti]
        norm = float(cp.linalg.norm(psi_t).get())
        p_diag = sector.diagonal_occupation_probability(psi_t, site_map)
        norms.append(norm)
        diag_probs.append(p_diag)

    norms = np.array(norms)
    diag_probs = np.array(diag_probs)

    print("------------------------------------------------------------")
    print("Two-Fermion Sector Diagnostics")
    print("------------------------------------------------------------")
    print(f"Mean norm  = {norms.mean():.6f}")
    print(f"Max |norm-1| over time = {np.max(np.abs(norms - 1.0)):.3e}")
    print(f"Max diagonal occupation probability = {diag_probs.max():.3e}")
    print(f"Mean diagonal occupation probability = {diag_probs.mean():.3e}")
    print("------------------------------------------------------------")
    print("If Pauli exclusion is correctly enforced, the diagonal")
    print("occupation probability should remain near machine precision.")
    print("============================================================")


if __name__ == "__main__":
    _sanity_check()
