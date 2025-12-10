"""
Substrate Framework: Three Mechanisms in One Script
===================================================

This module implements three levels of the Hilbert Substrate model:

1. ScalarSubstrate3D
   -------------------
   Scalar feedback on link amplitudes in a 3D lattice.
   - Degrees of freedom: complex scalar per site.
   - Geometry: real link weights t_ij between nearest neighbours.
   - Feedback: dt_ij/dt = eta * |psi_i||psi_j| - decay * (t_ij - 1).
   - Purpose: topological mass generation and self-trapping.

2. MatrixSubstrate3D
   -------------------
   SU(2)-valued link matrices coupling spinor fields in 3D.
   - Degrees of freedom: 2-component spinor per site.
   - Geometry: 2x2 complex link matrices U_ij.
   - Feedback: links updated from local spinor correlations
               (traceless SU(2) part), plus relaxation toward identity,
               with an explicit SU(2) projection to control drift.
   - Purpose: emergent gauge-like spin precession and flux.

3. LivingLink1D
   ---------------
   Microscopic living-link model in 1D.
   - Degrees of freedom: particle position + two-level links on each bond.
   - Hamiltonian: H = E_link * sum_j n_j - t_hop * sum_x hops that flip links.
   - Observable: effective hopping t_eff from ground-state bandwidth;
                 α(E_link) = t_eff / t_hop, compared to t_hop / E_link.
   - Purpose: emergent fine-structure-like coupling from vacuum stiffness.

4. CombinedSubstrate3D
   ---------------------
   Full 3D substrate carrying all three mechanisms in effective form:
   - Scalar amplitudes t_ij on links.
   - SU(2) matrices U_ij on the same links.
   - Spinor field ψ_i on sites.
   - Effective living-link stiffness E_link that penalizes deviations of U_ij
     from the identity and stabilizes the gauge sector.

   Hamiltonian: H_ij = - t_ij U_ij acting on ψ (spinor field).
   Feedback:
     - Scalar:  dt_ij/dt = η_s |ψ_i||ψ_j| - γ_s (t_ij - 1).
     - Matrix:  dU_ij/dt = η_m (traceless corr_ij)
                           - (γ_m + E_link)(U_ij - I_2),
       where corr_ij = |ψ_i⟩⟨ψ_j|. After each update U_ij is projected
       back to SU(2) via polar decomposition.

Both matrix-based models also track gauge diagnostics:

    gauge_dev       = mean ||U_ij - I_2||
    gauge_unit_err  = mean ||U_ij^† U_ij - I_2||

The script is driven entirely by the CONFIG dictionary defined below.
No command-line interface is required.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- GPU imports (for 3D scalar/matrix/combined models) --------
try:
    import cupy as cp
    from cupyx.scipy import linalg as cpla
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    cpla = None
    GPU_AVAILABLE = False

# ---------------- CPU imports (for LivingLink1D) -----------------------------
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
CONFIG = {
    # Which experiment to run: "scalar", "matrix", "living", or "combined"
    "experiment": "combined",

    # ScalarSubstrate3D parameters
    "scalar": {
        "L": 6,
        "eta": 0.8,
        "decay": 0.05,
        "steps": 200,
        "dt": 0.05,
        "plot": True,
    },

    # MatrixSubstrate3D parameters
    "matrix": {
        "L": 4,
        "eta": 1.5,
        "decay": 0.1,
        "steps": 150,
        "dt": 0.05,
        "plot": True,
    },

    # LivingLink1D parameters
    "living": {
        "L": 8,
        "t_hop": 1.0,
        "E_link_values": [0.5, 1, 2, 4, 8, 16, 32, 64],
        "num_eigs": 1,
        "plot": True,
        "figfile": "living_link_alpha.png",
    },

    # CombinedSubstrate3D parameters
    "combined": {
        "L": 4,
        "eta_scalar": 0.6,
        "decay_scalar": 0.05,
        "eta_matrix": 1.0,
        "decay_matrix": 0.1,
        "E_link": 1.0,      # less stiff than before -> nontrivial curvature
        "steps": 150,
        "dt": 0.05,
        "plot": True,
    },
}


# ============================================================================
# Helper: project a 2x2 CuPy matrix onto SU(2)
# ============================================================================
def project_to_su2_cp(U: "cp.ndarray") -> "cp.ndarray":
    """
    Project a 2x2 complex matrix onto SU(2) using polar decomposition.

    Steps:
        1. Compute H = U^† U.
        2. Form H^{-1/2} via eigen-decomposition.
        3. Set U_unitary = U H^{-1/2}.
        4. Rescale to det(U_unitary) = 1.

    This enforces approximate unitarity and det=1, stabilizing the gauge sector.
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
# 1. SCALAR SUBSTRATE (3D, GPU)
# ============================================================================
class ScalarSubstrate3D:
    """
    3D scalar substrate with feedback on link amplitudes.
    """

    def __init__(self, L, eta, decay):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for ScalarSubstrate3D.")

        self.L = int(L)
        self.N = self.L**3
        self.eta = float(eta)
        self.decay = float(decay)

        self.links = {}
        self.neighbors = [[] for _ in range(self.N)]
        self._build_lattice()
        self.psi = self._init_wavepacket()

    def _idx(self, x, y, z):
        return x * self.L * self.L + y * self.L + z

    def _build_lattice(self):
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    u = self._idx(x, y, z)
                    shifts = [
                        (-1, 0, 0), (1, 0, 0),
                        (0, -1, 0), (0, 1, 0),
                        (0, 0, -1), (0, 0, 1),
                    ]
                    for dx, dy, dz in shifts:
                        nx = (x + dx) % self.L
                        ny = (y + dy) % self.L
                        nz = (z + dz) % self.L
                        v = self._idx(nx, ny, nz)
                        self.neighbors[u].append(v)
                        if (u, v) not in self.links:
                            self.links[(u, v)] = 1.0

    def _init_wavepacket(self):
        psi_host = np.zeros(self.N, dtype=np.complex128)
        c = self.L // 2
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._idx(x, y, z)
                    r2 = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
                    psi_host[idx] = np.exp(-r2 / 2.0)
        psi_host /= np.linalg.norm(psi_host)
        return cp.asarray(psi_host)

    def hamiltonian(self):
        H = cp.zeros((self.N, self.N), dtype=cp.complex128)
        for (u, v), t_val in self.links.items():
            H[u, v] -= t_val
        return H

    def evolve(self, steps, dt):
        steps = int(steps)
        dt = float(dt)
        history = {"energy": [], "participation": []}

        for _ in range(steps):
            H = self.hamiltonian()
            U = cpla.expm(-1j * H * dt)
            self.psi = U @ self.psi

            psi_mag = cp.abs(self.psi).get()
            new_links = {}
            for (u, v), val in self.links.items():
                corr = psi_mag[u] * psi_mag[v]
                dv = (self.eta * corr - self.decay * (val - 1.0)) * dt
                new_links[(u, v)] = val + dv
            self.links = new_links

            E = float(cp.vdot(self.psi, H @ self.psi).real.get())
            PR = float((1.0 / cp.sum(cp.abs(self.psi) ** 4)).get())
            history["energy"].append(E)
            history["participation"].append(PR)

        return history


def run_scalar_experiment(cfg):
    model = ScalarSubstrate3D(cfg["L"], cfg["eta"], cfg["decay"])
    res = model.evolve(cfg["steps"], cfg["dt"])

    if cfg.get("plot", True):
        energy = np.array(res["energy"])
        pr = np.array(res["participation"])

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(pr)
        ax[0].set_title("Scalar: Participation Ratio")
        ax[0].set_xlabel("Step")
        ax[0].set_ylabel("PR")

        ax[1].plot(energy)
        ax[1].set_title("Scalar: Energy")
        ax[1].set_xlabel("Step")
        ax[1].set_ylabel("E")

        plt.tight_layout()
        plt.show()


# ============================================================================
# 2. MATRIX SUBSTRATE (3D, GPU) WITH SU(2) PROJECTION & DIAGNOSTICS
# ============================================================================
class MatrixSubstrate3D:
    """
    3D spinor substrate with matrix-valued links.
    """

    def __init__(self, L, eta, decay):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for MatrixSubstrate3D.")

        self.L = int(L)
        self.N = self.L**3
        self.eta = float(eta)
        self.decay = float(decay)

        self.links = {}
        self._build_links()
        self.psi = self._init_spinor_wavepacket()

    def _idx(self, x, y, z):
        return x * self.L * self.L + y * self.L + z

    def _build_links(self):
        I2 = cp.eye(2, dtype=cp.complex128)
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    u = self._idx(x, y, z)
                    for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        nx = (x + dx) % self.L
                        ny = (y + dy) % self.L
                        nz = (z + dz) % self.L
                        v = self._idx(nx, ny, nz)
                        self.links[(u, v)] = I2.copy()
                        self.links[(v, u)] = I2.copy()

    def _init_spinor_wavepacket(self):
        psi_host = np.zeros((self.N, 2), dtype=np.complex128)
        c = self.L // 2
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._idx(x, y, z)
                    r2 = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
                    amp = np.exp(-r2 / 2.0)
                    psi_host[idx] = [amp, 0.0]
        flat = psi_host.reshape(-1)
        flat /= np.linalg.norm(flat)
        return cp.asarray(flat)

    def hamiltonian(self):
        size = 2 * self.N
        H = cp.zeros((size, size), dtype=cp.complex128)
        for (u, v), U in self.links.items():
            ui, vi = 2 * u, 2 * v
            H[ui:ui+2, vi:vi+2] -= U
        return H

    def _wilson_loop_flux(self):
        c = self.L // 2
        p1 = self._idx(c, c, c)
        p2 = self._idx((c+1) % self.L, c, c)
        p3 = self._idx((c+1) % self.L, (c+1) % self.L, c)
        p4 = self._idx(c, (c+1) % self.L, c)
        W = (self.links[(p1, p2)]
             @ self.links[(p2, p3)]
             @ self.links[(p3, p4)]
             @ self.links[(p4, p1)])
        tr = cp.trace(W)
        return 2.0 - float(cp.abs(tr).get())

    def evolve(self, steps, dt):
        steps = int(steps)
        dt = float(dt)
        I2 = cp.eye(2, dtype=cp.complex128)

        fluxes = []
        spin_down_probs = []
        gauge_dev = []
        gauge_unit_err = []

        for _ in range(steps):
            H = self.hamiltonian()
            U_op = cpla.expm(-1j * H * dt)
            self.psi = U_op @ self.psi

            psi_spinors = self.psi.reshape(self.N, 2)

            # Update links
            for (u, v) in list(self.links.keys()):
                if u < v:
                    P_i = psi_spinors[u]
                    P_j = psi_spinors[v]
                    corr = cp.outer(P_i, cp.conj(P_j))

                    # Use only the traceless SU(2) part
                    tr_corr = cp.trace(corr) / 2.0
                    corr_su2 = corr - tr_corr * I2

                    U_curr = self.links[(u, v)]
                    delta = (self.eta * corr_su2
                             - self.decay * (U_curr - I2)) * dt
                    U_trial = U_curr + delta
                    U_new = project_to_su2_cp(U_trial)

                    self.links[(u, v)] = U_new
                    self.links[(v, u)] = U_new.conj().T

            # Diagnostics
            flux = self._wilson_loop_flux()
            fluxes.append(flux)

            total_prob = cp.sum(cp.abs(psi_spinors)**2, axis=0)
            spin_down_probs.append(float(total_prob[1].get()))

            # Gauge deviation and unitarity error
            dev_sum = 0.0
            unit_sum = 0.0
            count = 0
            for (u, v), U in self.links.items():
                if u < v:
                    diff_I = U - I2
                    dev_sum += float(cp.linalg.norm(diff_I).get())
                    UU = U.conj().T @ U
                    unit_diff = UU - I2
                    unit_sum += float(cp.linalg.norm(unit_diff).get())
                    count += 1
            gauge_dev.append(dev_sum / max(count, 1))
            gauge_unit_err.append(unit_sum / max(count, 1))

        return {
            "flux": fluxes,
            "spin_down": spin_down_probs,
            "gauge_dev": gauge_dev,
            "gauge_unit_error": gauge_unit_err,
        }


def run_matrix_experiment(cfg):
    model = MatrixSubstrate3D(cfg["L"], cfg["eta"], cfg["decay"])
    res = model.evolve(cfg["steps"], cfg["dt"])

    if cfg.get("plot", True):
        steps = np.arange(len(res["flux"]))
        flux = np.array(res["flux"])
        spin = np.array(res["spin_down"])
        gdev = np.array(res["gauge_dev"])
        gerr = np.array(res["gauge_unit_error"])

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].plot(steps, flux)
        ax[0, 0].set_title("Matrix: Wilson Loop Flux Proxy")
        ax[0, 0].set_xlabel("Step")
        ax[0, 0].set_ylabel("Flux")

        ax[0, 1].plot(steps, spin)
        ax[0, 1].set_title("Matrix: Spin-Down Probability")
        ax[0, 1].set_xlabel("Step")
        ax[0, 1].set_ylabel("P(↓)")

        ax[1, 0].plot(steps, gdev)
        ax[1, 0].set_title("Matrix: ⟨‖U−I‖⟩")
        ax[1, 0].set_xlabel("Step")
        ax[1, 0].set_ylabel("Deviation")

        ax[1, 1].plot(steps, gerr)
        ax[1, 1].set_title("Matrix: ⟨‖U†U−I‖⟩ (Unitarity Error)")
        ax[1, 1].set_xlabel("Step")
        ax[1, 1].set_ylabel("Error")

        plt.tight_layout()
        plt.show()


# ============================================================================
# 3. LIVING-LINK MODEL (1D, CPU)
# ============================================================================
class LivingLink1D:
    """
    Microscopic 1D living-link Hamiltonian.
    """

    def __init__(self, L, t_hop, E_link):
        self.L = int(L)
        self.num_links = self.L - 1
        self.link_dim = 2 ** self.num_links
        self.dim = self.L * self.link_dim

        self.t_hop = float(t_hop)
        self.E_link = float(E_link)

    def _index(self, x, link_state):
        return x * self.link_dim + link_state

    def build_hamiltonian(self):
        rows, cols, data = [], [], []
        for x in range(self.L):
            for link_state in range(self.link_dim):
                u = self._index(x, link_state)
                n_exc = bin(link_state).count("1")
                rows.append(u); cols.append(u); data.append(self.E_link * n_exc)
                if x < self.L-1:
                    flipped = link_state ^ (1 << x)
                    v = self._index(x+1, flipped)
                    rows.append(u); cols.append(v); data.append(-self.t_hop)
                if x > 0:
                    flipped = link_state ^ (1 << (x-1))
                    v = self._index(x-1, flipped)
                    rows.append(u); cols.append(v); data.append(-self.t_hop)
        H = csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))
        H = 0.5 * (H + H.T)
        return H

    def ground_state(self, num_eigs=1):
        H = self.build_hamiltonian()
        E, V = eigsh(H, k=num_eigs, which="SA")
        return float(E[0]), V[:, 0]

    @staticmethod
    def free_ground_energy(L, t_hop):
        return -2 * t_hop * np.cos(np.pi / (L+1))

    @staticmethod
    def vacuum_probability(psi, L, link_dim):
        psi = psi.reshape(L, link_dim)
        return float(np.sum(np.abs(psi[:, 0])**2))

    @staticmethod
    def effective_hopping(E_int, E_free, t_hop):
        return -E_int / (-E_free) * t_hop


def run_living_experiment(cfg):
    L = cfg["L"]
    t_hop = cfg["t_hop"]
    E_values = cfg["E_link_values"]
    num_eigs = cfg["num_eigs"]

    E_free = LivingLink1D.free_ground_energy(L, t_hop)
    records = []

    for E_link in E_values:
        model = LivingLink1D(L, t_hop, E_link)
        E0, psi0 = model.ground_state(num_eigs=num_eigs)
        t_eff = LivingLink1D.effective_hopping(E0, E_free, t_hop)
        alpha_meas = t_eff / t_hop
        alpha_th = t_hop / E_link
        P_vac = LivingLink1D.vacuum_probability(psi0, L, model.link_dim)
        records.append({
            "E_link": E_link,
            "E_ground": E0,
            "alpha_measured": alpha_meas,
            "alpha_theory": alpha_th,
            "vacuum_prob": P_vac,
        })

    if cfg.get("plot", True):
        E = np.array([r["E_link"] for r in records], dtype=float)
        a_meas = np.array([r["alpha_measured"] for r in records], dtype=float)
        a_th = np.array([r["alpha_theory"] for r in records], dtype=float)
        Pvac = np.array([r["vacuum_prob"] for r in records], dtype=float)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].loglog(E, a_meas, "o-", label="Measured α")
        ax[0].loglog(E, a_th, "--", label="t_hop / E_link")
        ax[0].set_xlabel("E_link")
        ax[0].set_ylabel("α")
        ax[0].legend()
        ax[0].grid(True, which="both")

        ax[1].semilogx(E, Pvac, "o-")
        ax[1].set_xlabel("E_link")
        ax[1].set_ylabel("P(vacuum)")
        ax[1].set_ylim(0, 1.05)
        ax[1].grid(True, which="both")

        plt.tight_layout()
        plt.show()

    return records


# ============================================================================
# 4. COMBINED SUBSTRATE (3D, GPU: scalar + matrix + stiffness + diagnostics)
# ============================================================================
class CombinedSubstrate3D:
    """
    Full 3D substrate carrying scalar, matrix, and effective stiffness.

    - Sites carry a 2-component spinor ψ_i.
    - Each oriented link (i -> j) carries:
        * a real scalar amplitude t_ij
        * a 2x2 complex matrix U_ij
    - Hamiltonian on spinors: H_ij = - t_ij U_ij.

    Feedback:
        - Scalar:  dt_ij/dt = η_scalar * |ψ_i||ψ_j| - decay_scalar * (t_ij - 1).
        - Matrix:  dU_ij/dt = η_matrix * (traceless corr_ij)
                               - (decay_matrix + E_link)(U_ij - I_2),
          then U_ij is projected to SU(2).

    Diagnostics:
        - energy, participation ratio (PR)
        - Wilson-loop flux proxy
        - mean t_ij
        - gauge_dev       = mean ||U_ij - I_2||
        - gauge_unit_err  = mean ||U_ij^† U_ij - I_2||
    """

    def __init__(self,
                 L,
                 eta_scalar,
                 decay_scalar,
                 eta_matrix,
                 decay_matrix,
                 E_link):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for CombinedSubstrate3D.")

        self.L = int(L)
        self.N = self.L**3

        self.eta_scalar = float(eta_scalar)
        self.decay_scalar = float(decay_scalar)
        self.eta_matrix = float(eta_matrix)
        self.decay_matrix = float(decay_matrix)
        self.E_link = float(E_link)

        self.t_links = {}
        self.U_links = {}
        self._build_links()
        self.psi = self._init_spinor_wavepacket()

    def _idx(self, x, y, z):
        return x * self.L * self.L + y * self.L + z

    def _build_links(self):
        I2 = cp.eye(2, dtype=cp.complex128)
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    u = self._idx(x, y, z)
                    shifts = [
                        (-1, 0, 0), (1, 0, 0),
                        (0, -1, 0), (0, 1, 0),
                        (0, 0, -1), (0, 0, 1),
                    ]
                    for dx, dy, dz in shifts:
                        nx = (x + dx) % self.L
                        ny = (y + dy) % self.L
                        nz = (z + dz) % self.L
                        v = self._idx(nx, ny, nz)
                        if (u, v) not in self.t_links:
                            self.t_links[(u, v)] = 1.0
                        if (u, v) not in self.U_links:
                            self.U_links[(u, v)] = I2.copy()

    def _init_spinor_wavepacket(self):
        psi_host = np.zeros((self.N, 2), dtype=np.complex128)
        c = self.L // 2
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._idx(x, y, z)
                    r2 = (x - c)**2 + (y - c)**2 + (z - c)**2
                    amp = np.exp(-r2 / 2.0)
                    psi_host[idx] = [amp, 0.0]
        flat = psi_host.reshape(-1)
        flat /= np.linalg.norm(flat)
        return cp.asarray(flat)

    def hamiltonian(self):
        size = 2 * self.N
        H = cp.zeros((size, size), dtype=cp.complex128)
        for (u, v), t_val in self.t_links.items():
            U = self.U_links[(u, v)]
            ui, vi = 2*u, 2*v
            H[ui:ui+2, vi:vi+2] -= t_val * U
        return H

    def _wilson_loop_flux(self):
        c = self.L // 2
        p1 = self._idx(c, c, c)
        p2 = self._idx((c+1) % self.L, c, c)
        p3 = self._idx((c+1) % self.L, (c+1) % self.L, c)
        p4 = self._idx(c, (c+1) % self.L, c)
        W = (self.U_links[(p1, p2)]
             @ self.U_links[(p2, p3)]
             @ self.U_links[(p3, p4)]
             @ self.U_links[(p4, p1)])
        tr = cp.trace(W)
        return 2.0 - float(cp.abs(tr).get())

    def evolve(self, steps, dt):
        steps = int(steps)
        dt = float(dt)
        I2 = cp.eye(2, dtype=cp.complex128)

        history = {
            "energy": [],
            "participation": [],
            "flux": [],
            "spin_down": [],
            "mean_t": [],
            "gauge_dev": [],
            "gauge_unit_error": [],
        }

        for _ in range(steps):
            H = self.hamiltonian()
            U_op = cpla.expm(-1j * H * dt)
            self.psi = U_op @ self.psi

            psi_spinors = self.psi.reshape(self.N, 2)
            psi_mag = cp.sqrt(cp.sum(cp.abs(psi_spinors)**2, axis=1)).get()

            # Scalar feedback on t_ij
            new_t = {}
            for (u, v), t_val in self.t_links.items():
                corr = psi_mag[u] * psi_mag[v]
                dt_val = (self.eta_scalar * corr
                          - self.decay_scalar * (t_val - 1.0)) * dt
                new_t[(u, v)] = t_val + dt_val
            self.t_links = new_t

            # Matrix feedback + stiffness + SU(2) projection
            for (u, v) in list(self.U_links.keys()):
                if u < v:
                    P_i = psi_spinors[u]
                    P_j = psi_spinors[v]
                    corr = cp.outer(P_i, cp.conj(P_j))

                    tr_corr = cp.trace(corr) / 2.0
                    corr_su2 = corr - tr_corr * I2

                    U_curr = self.U_links[(u, v)]
                    delta = (self.eta_matrix * corr_su2
                             - (self.decay_matrix + self.E_link)
                             * (U_curr - I2)) * dt
                    U_trial = U_curr + delta
                    U_new = project_to_su2_cp(U_trial)

                    self.U_links[(u, v)] = U_new
                    self.U_links[(v, u)] = U_new.conj().T

            # Diagnostics
            E = float(cp.vdot(self.psi, H @ self.psi).real.get())
            PR = float((1.0 / cp.sum(cp.abs(self.psi)**4)).get())
            flux = self._wilson_loop_flux()
            total_prob = cp.sum(cp.abs(psi_spinors)**2, axis=0)
            p_down = float(total_prob[1].get())
            mean_t = float(np.mean(list(self.t_links.values())))

            # Gauge deviation & unitarity error
            dev_sum = 0.0
            unit_sum = 0.0
            count = 0
            for (u, v), U in self.U_links.items():
                if u < v:
                    diff_I = U - I2
                    dev_sum += float(cp.linalg.norm(diff_I).get())
                    UU = U.conj().T @ U
                    unit_diff = UU - I2
                    unit_sum += float(cp.linalg.norm(unit_diff).get())
                    count += 1
            gauge_dev = dev_sum / max(count, 1)
            gauge_unit_err = unit_sum / max(count, 1)

            history["energy"].append(E)
            history["participation"].append(PR)
            history["flux"].append(flux)
            history["spin_down"].append(p_down)
            history["mean_t"].append(mean_t)
            history["gauge_dev"].append(gauge_dev)
            history["gauge_unit_error"].append(gauge_unit_err)

        return history


def run_combined_experiment(cfg):
    model = CombinedSubstrate3D(
        L=cfg["L"],
        eta_scalar=cfg["eta_scalar"],
        decay_scalar=cfg["decay_scalar"],
        eta_matrix=cfg["eta_matrix"],
        decay_matrix=cfg["decay_matrix"],
        E_link=cfg["E_link"],
    )
    res = model.evolve(cfg["steps"], cfg["dt"])

    if cfg.get("plot", True):
        steps = np.arange(len(res["energy"]))
        energy = np.array(res["energy"])
        pr = np.array(res["participation"])
        flux = np.array(res["flux"])
        mean_t = np.array(res["mean_t"])
        gdev = np.array(res["gauge_dev"])
        gerr = np.array(res["gauge_unit_error"])

        fig, ax = plt.subplots(3, 2, figsize=(11, 10))

        ax[0, 0].plot(steps, pr)
        ax[0, 0].set_title("Combined: Participation Ratio")
        ax[0, 0].set_xlabel("Step")
        ax[0, 0].set_ylabel("PR")

        ax[0, 1].plot(steps, energy)
        ax[0, 1].set_title("Combined: Energy")
        ax[0, 1].set_xlabel("Step")
        ax[0, 1].set_ylabel("E")

        ax[1, 0].plot(steps, flux)
        ax[1, 0].set_title("Combined: Wilson Loop Flux Proxy")
        ax[1, 0].set_xlabel("Step")
        ax[1, 0].set_ylabel("Flux")

        ax[1, 1].plot(steps, mean_t)
        ax[1, 1].set_title("Combined: Mean Link Amplitude ⟨t_ij⟩")
        ax[1, 1].set_xlabel("Step")
        ax[1, 1].set_ylabel("⟨t⟩")

        ax[2, 0].plot(steps, gdev)
        ax[2, 0].set_title("Combined: ⟨‖U−I‖⟩")
        ax[2, 0].set_xlabel("Step")
        ax[2, 0].set_ylabel("Deviation")

        ax[2, 1].plot(steps, gerr)
        ax[2, 1].set_title("Combined: ⟨‖U†U−I‖⟩ (Unitarity Error)")
        ax[2, 1].set_xlabel("Step")
        ax[2, 1].set_ylabel("Error")

        plt.tight_layout()
        plt.show()

    return res


# ============================================================================
# MAIN DISPATCH
# ============================================================================
if __name__ == "__main__":
    mode = CONFIG["experiment"].lower()
    if mode == "scalar":
        run_scalar_experiment(CONFIG["scalar"])
    elif mode == "matrix":
        run_matrix_experiment(CONFIG["matrix"])
    elif mode == "living":
        run_living_experiment(CONFIG["living"])
    elif mode == "combined":
        run_combined_experiment(CONFIG["combined"])
    else:
        raise ValueError(f"Unknown experiment mode: {CONFIG['experiment']}")
