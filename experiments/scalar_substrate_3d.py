# scalar_substrate_3d.py
# -----------------------------------------------------------------------------
# SUBSTRATE LEVEL 1: SCALAR GEOMETRY (GPU VERSION, CONFIG-DRIVEN)
# Physics: Topological Mass & Gravity
# Mechanism: Link amplitudes t_ij evolve based on local particle density.
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
    "L": 6,          # Lattice size: L x L x L
    "eta": 0.8,      # Feedback strength (mass generation rate)
    "decay": 0.05,   # Vacuum relaxation rate
    "steps": 200,    # Number of evolution steps
    "dt": 0.05,      # Time step
    "plot": True,    # Show matplotlib plots when run as __main__
}


class ScalarSubstrate3D:
    def __init__(self, L=6, eta=0.8, decay=0.05):
        self.L = int(L)  # Lattice dimension (LxLxL)
        self.N = self.L**3
        self.eta = float(eta)      # Feedback strength (mass generation rate)
        self.decay = float(decay)  # Vacuum relaxation rate

        # Geometry: Links are stored in a dictionary (u, v) -> amplitude (float)
        # We initialize a flat vacuum (t=1.0)
        self.links = {}
        self.neighbors = [[] for _ in range(self.N)]
        self._initialize_lattice()

        # State: Gaussian wavepacket on GPU
        self.psi = self._init_wavepacket()  # cp.ndarray, shape (N,)

    # ----------------------- LATTICE GEOMETRY ----------------------------

    def _get_idx(self, x, y, z):
        return x * self.L * self.L + y * self.L + z

    def _initialize_lattice(self):
        """Builds a 3D cubic lattice with periodic boundaries."""
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    u = self._get_idx(x, y, z)
                    # 6 neighbors (Left, Right, Up, Down, Front, Back)
                    shifts = [
                        (-1, 0, 0), (1, 0, 0),
                        (0, -1, 0), (0, 1, 0),
                        (0, 0, -1), (0, 0, 1),
                    ]
                    for dx, dy, dz in shifts:
                        nx = (x + dx) % self.L
                        ny = (y + dy) % self.L
                        nz = (z + dz) % self.L
                        v = self._get_idx(nx, ny, nz)
                        self.neighbors[u].append(v)
                        # Initialize link amplitude t_0 = 1.0; store each directed link once
                        if (u, v) not in self.links:
                            self.links[(u, v)] = 1.0

    # ----------------------- INITIAL STATE -------------------------------

    def _init_wavepacket(self):
        """Creates a Gaussian bump in the center (on GPU)."""
        psi_host = np.zeros(self.N, dtype=np.complex128)
        center = self.L // 2
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._get_idx(x, y, z)
                    dist = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
                    psi_host[idx] = np.exp(-dist / 2.0)
        psi_host /= np.linalg.norm(psi_host)
        return cp.asarray(psi_host)

    # ----------------------- HAMILTONIAN ---------------------------------

    def get_hamiltonian(self):
        """
        Constructs the dense Hamiltonian H from current link amplitudes, on GPU.

        H_ij = -t_ij for each link (i -> j). Degrees: N x N complex matrix.
        """
        H = cp.zeros((self.N, self.N), dtype=cp.complex128)
        for (u, v), val in self.links.items():
            H[u, v] -= val  # H = - sum t_ij c_i^† c_j
        return H

    # ----------------------- EVOLUTION -----------------------------------

    def evolve(self, steps=100, dt=0.05):
        """
        Evolve state and geometry for a number of steps.

        Returns:
            history: dict with fields
                'energy': list[float]
                'participation': list[float]
        """
        steps = int(steps)
        dt = float(dt)

        history = {'energy': [], 'participation': []}

        print(f"[ScalarSubstrate3D] GPU simulation: L={self.L}, N={self.N}, "
              f"steps={steps}, dt={dt}")

        for t in range(steps):
            # 1. Quantum Step (Unitary Evolution) on GPU
            H = self.get_hamiltonian()
            U_op = cpla.expm(-1j * H * dt)  # full dense expm on GPU
            self.psi = U_op @ self.psi

            # 2. Geometric Step (Scalar Feedback) on CPU scalars but using GPU mags
            # dt_ij/dt = eta * |psi_i||psi_j| - decay * (t_ij - 1.0)
            psi_mag = cp.abs(self.psi)
            psi_mag_host = psi_mag.get()  # small overhead; links are only ~6N

            new_links = {}
            for (u, v), val in self.links.items():
                correlation = float(psi_mag_host[u] * psi_mag_host[v])
                delta = (self.eta * correlation - self.decay * (val - 1.0)) * dt
                new_links[(u, v)] = val + delta
            self.links = new_links

            # 3. Measurement
            E = float(cp.vdot(self.psi, H @ self.psi).real.get())
            PR = float((1.0 / cp.sum(cp.abs(self.psi) ** 4)).get())

            history['energy'].append(E)
            history['participation'].append(PR)

            if (t + 1) % max(1, steps // 10) == 0:
                print(f"  step {t+1}/{steps}: E={E:.4f}, PR={PR:.2f}")

        return history


# -------------------------------------------------------------------------
# MAIN (CONFIG-DRIVEN)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    sim = ScalarSubstrate3D(
        L=CONFIG["L"],
        eta=CONFIG["eta"],
        decay=CONFIG["decay"],
    )
    res = sim.evolve(
        steps=CONFIG["steps"],
        dt=CONFIG["dt"],
    )

    if CONFIG.get("plot", True):
        energy = np.array(res['energy'])
        pr = np.array(res['participation'])

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(pr)
        plt.title("Particle Size (Participation Ratio)")
        plt.xlabel("Step")
        plt.ylabel("PR")

        plt.subplot(1, 2, 2)
        plt.plot(energy)
        plt.title("System Energy (Mass Formation)")
        plt.xlabel("Step")
        plt.ylabel("Energy")

        plt.tight_layout()
        plt.show()
