# matrix_substrate_3d.py
# -----------------------------------------------------------------------------
# SUBSTRATE LEVEL 2: MATRIX GEOMETRY (GPU VERSION, CONFIG-DRIVEN)
# Physics: Electromagnetism, Spin, Gauge Invariance
# Mechanism: Links U_ij are SU(2) matrices evolving via spinor correlations.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
    from cupyx.scipy import linalg as cpla
except ImportError as e:
    raise ImportError(
        "This module requires CuPy and cupyx.scipy. "
        "Install a CUDA-enabled CuPy build, e.g. 'pip install cupy-cuda12x'."
    ) from e

# -------------------------------------------------------------------------
# CONFIG BLOCK - EDIT THIS, NO CLI NEEDED
# -------------------------------------------------------------------------
CONFIG = {
    "L": 4,          # Lattice size: L x L x L
    "eta": 1.5,      # Feedback strength for gauge links
    "decay": 0.1,    # Relaxation of links toward identity
    "steps": 150,    # Evolution steps
    "dt": 0.05,      # Time step
    "plot": True,    # Show flux and spin-mix plots
}

# Pauli Matrices on GPU
SIGMA = [
    cp.eye(2, dtype=cp.complex128),                # Identity
    cp.array([[0, 1], [1, 0]], dtype=cp.complex128),   # X
    cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128), # Y
    cp.array([[1, 0], [0, -1]], dtype=cp.complex128),   # Z
]


class MatrixSubstrate3D:
    def __init__(self, L=4, eta=1.5, decay=0.1):
        self.L = int(L)
        self.N = self.L**3
        self.eta = float(eta)
        self.decay = float(decay)

        # State: spinor at each site (2 components), stored as flat vector on GPU
        self.psi = self._init_spinor_wavepacket()

        # Geometry: links (u, v) -> 2x2 complex matrix (cp.ndarray)
        self.links = {}
        self._init_links()

    # ----------------------- GEOMETRY HELPERS ----------------------------

    def _get_idx(self, x, y, z):
        return x * self.L**2 + y * self.L + z

    def _init_links(self):
        """Initialize all links to Identity matrix (flat gauge field)."""
        I2 = cp.eye(2, dtype=cp.complex128)
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    u = self._get_idx(x, y, z)
                    shifts = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # forward links
                    for dx, dy, dz in shifts:
                        nx = (x + dx) % self.L
                        ny = (y + dy) % self.L
                        nz = (z + dz) % self.L
                        v = self._get_idx(nx, ny, nz)
                        self.links[(u, v)] = I2.copy()
                        self.links[(v, u)] = I2.copy()  # Hermitian counterpart

    # ----------------------- INITIAL STATE -------------------------------

    def _init_spinor_wavepacket(self):
        """Gaussian packet with Spin UP, on GPU."""
        psi_host = np.zeros((self.N, 2), dtype=np.complex128)
        center = self.L // 2
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._get_idx(x, y, z)
                    dist = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
                    amp = np.exp(-dist / 2.0)
                    psi_host[idx] = [amp, 0.0]  # pure spin up
        flat = psi_host.reshape(-1)
        flat /= np.linalg.norm(flat)
        return cp.asarray(flat)

    # ----------------------- HAMILTONIAN ---------------------------------

    def get_hamiltonian(self):
        """
        Builds 2N x 2N Hamiltonian with matrix hoppings on GPU.

        H_int = - sum_{<i,j>} c_i^† U_ij c_j
        """
        size = 2 * self.N
        H = cp.zeros((size, size), dtype=cp.complex128)
        for (u, v), U_mat in self.links.items():
            ui, vi = 2 * u, 2 * v
            H[ui:ui + 2, vi:vi + 2] -= U_mat
        return H

    # ----------------------- FLUX MEASUREMENT ----------------------------

    def measure_flux(self):
        """Calculates magnetic flux (Wilson Loop) through a center plaquette."""
        c = self.L // 2
        p1 = self._get_idx(c, c, c)
        p2 = self._get_idx(c + 1, c, c)
        p3 = self._get_idx(c + 1, c + 1, c)
        p4 = self._get_idx(c, c + 1, c)

        W = (
            self.links[(p1, p2)]
            @ self.links[(p2, p3)]
            @ self.links[(p3, p4)]
            @ self.links[(p4, p1)]
        )

        # Flux proxy: 2 - |Tr(W)| (0 when no flux, >0 when flux present)
        trace_val = cp.trace(W)
        flux_val = 2.0 - float(cp.abs(trace_val).get())
        return flux_val

    # ----------------------- EVOLUTION -----------------------------------

    def evolve(self, steps=100, dt=0.05):
        """
        Evolve spinor field and gauge links.

        Returns:
            fluxes: list[float]     (Wilson loop flux proxy)
            overlaps: list[float]   (total spin-down probability)
        """
        steps = int(steps)
        dt = float(dt)

        fluxes = []
        overlaps = []

        print(f"[MatrixSubstrate3D] GPU simulation: L={self.L}, N={self.N}, "
              f"steps={steps}, dt={dt}")

        for t in range(steps):
            # 1. Quantum step
            H = self.get_hamiltonian()
            U_op = cpla.expm(-1j * H * dt)
            self.psi = U_op @ self.psi

            # 2. Matrix feedback
            psi_spinors = self.psi.reshape((self.N, 2))
            for (u, v) in list(self.links.keys()):
                if u < v:  # update each pair once
                    P_i = psi_spinors[u]  # (2,)
                    P_j = psi_spinors[v]  # (2,)

                    # Outer product |i><j|
                    correlation = cp.outer(P_i, cp.conj(P_j))

                    U_curr = self.links[(u, v)]
                    delta = (self.eta * correlation
                             - self.decay * (U_curr - cp.eye(2, dtype=cp.complex128))) * dt

                    U_new = U_curr + delta

                    # (Optional) Enforce unitarity with a soft projection via SVD
                    # u_svd, _, vh_svd = cpla.svd(U_new)
                    # U_new = u_svd @ vh_svd

                    self.links[(u, v)] = U_new
                    self.links[(v, u)] = U_new.conj().T

            # 3. Data collection
            flux = self.measure_flux()
            fluxes.append(flux)

            total_prob = cp.sum(cp.abs(psi_spinors)**2, axis=0)  # (2,)
            spin_down = float(total_prob[1].get())
            overlaps.append(spin_down)

            if (t + 1) % max(1, steps // 10) == 0:
                print(f"  step {t+1}/{steps}: flux={flux:.4f}, "
                      f"P_down={spin_down:.4f}")

        return fluxes, overlaps


# -------------------------------------------------------------------------
# MAIN (CONFIG-DRIVEN)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    sim = MatrixSubstrate3D(
        L=CONFIG["L"],
        eta=CONFIG["eta"],
        decay=CONFIG["decay"],
    )
    flux, spin_mix = sim.evolve(
        steps=CONFIG["steps"],
        dt=CONFIG["dt"],
    )

    if CONFIG.get("plot", True):
        flux = np.array(flux)
        spin_mix = np.array(spin_mix)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(flux)
        plt.title("Emergent Magnetic Flux (Wilson Loop)")
        plt.xlabel("Step")
        plt.ylabel("Flux Proxy (2 - |Tr W|)")

        plt.subplot(1, 2, 2)
        plt.plot(spin_mix)
        plt.title("Spin Mixing (Down Probability)")
        plt.xlabel("Step")
        plt.ylabel("P(↓)")

        plt.tight_layout()
        plt.show()
