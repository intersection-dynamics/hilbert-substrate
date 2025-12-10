"""
Substrate Framework Experiments
===============================

One file containing:
- ScalarSubstrate3D   (scalar feedback / mass)
- MatrixSubstrate3D   (matrix gauge feedback / spin & flux)
- LivingLink1D        (microscopic living links / alpha(E))
- CombinedSubstrate3D (scalar + matrix + effective stiffness on same lattice)

Driven entirely by CONFIG at top; no CLI flags.
"""

import numpy as np
import matplotlib.pyplot as plt

# GPU stack for 3D substrates
try:
    import cupy as cp
    from cupyx.scipy import linalg as cpla
    GPU_AVAILABLE = True
except Exception:
    cp = None
    cpla = None
    GPU_AVAILABLE = False

# CPU stack for living-link model
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "experiment": "combined",  # "scalar", "matrix", "living", "combined"

    "scalar": {
        "L": 6,
        "eta": 0.8,
        "decay": 0.05,
        "steps": 200,
        "dt": 0.05,
        "plot": True,
    },

    "matrix": {
        "L": 4,
        "eta": 1.5,
        "decay": 0.1,
        "steps": 150,
        "dt": 0.05,
        "plot": True,
    },

    "living": {
        "L": 8,
        "t_hop": 1.0,
        "E_link_values": [0.5, 1, 2, 4, 8, 16, 32, 64],
        "num_eigs": 1,
        "plot": True,
        "figfile": "living_link_alpha.png",
    },

    "combined": {
        "L": 4,
        "eta_scalar": 0.6,
        "decay_scalar": 0.05,
        "eta_matrix": 1.0,
        "decay_matrix": 0.1,
        "E_link": 8.0,   # effective stiffness scale (living-link inspired)
        "steps": 150,
        "dt": 0.05,
        "plot": True,
    },
}

# ---------------------------------------------------------------------------
# 1. ScalarSubstrate3D
# ---------------------------------------------------------------------------
class ScalarSubstrate3D:
    def __init__(self, L, eta, decay):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for ScalarSubstrate3D.")
        self.L = int(L)
        self.N = self.L ** 3
        self.eta = float(eta)
        self.decay = float(decay)
        self.links = {}
        self._build_lattice()
        self.psi = self._init_wavepacket()

    def _idx(self, x, y, z):
        return x * self.L * self.L + y * self.L + z

    def _build_lattice(self):
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    u = self._idx(x, y, z)
                    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                        nx = (x + dx) % self.L
                        ny = (y + dy) % self.L
                        nz = (z + dz) % self.L
                        v = self._idx(nx, ny, nz)
                        if (u, v) not in self.links:
                            self.links[(u, v)] = 1.0

    def _init_wavepacket(self):
        psi_host = np.zeros(self.N, dtype=np.complex128)
        c = self.L // 2
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._idx(x, y, z)
                    r2 = (x-c)**2 + (y-c)**2 + (z-c)**2
                    psi_host[idx] = np.exp(-r2/2.0)
        psi_host /= np.linalg.norm(psi_host)
        return cp.asarray(psi_host)

    def hamiltonian(self):
        H = cp.zeros((self.N, self.N), dtype=cp.complex128)
        for (u, v), t in self.links.items():
            H[u, v] -= t
        return H

    def evolve(self, steps, dt):
        steps = int(steps); dt = float(dt)
        history = {"energy": [], "PR": []}
        for _ in range(steps):
            H = self.hamiltonian()
            U = cpla.expm(-1j * H * dt)
            self.psi = U @ self.psi
            psi_mag = cp.abs(self.psi).get()
            new_links = {}
            for (u, v), t in self.links.items():
                corr = psi_mag[u] * psi_mag[v]
                dt_val = (self.eta * corr - self.decay * (t - 1.0)) * dt
                new_links[(u, v)] = t + dt_val
            self.links = new_links
            E = float(cp.vdot(self.psi, H @ self.psi).real.get())
            PR = float((1.0 / cp.sum(cp.abs(self.psi) ** 4)).get())
            history["energy"].append(E)
            history["PR"].append(PR)
        return history

# ---------------------------------------------------------------------------
# 2. MatrixSubstrate3D
# ---------------------------------------------------------------------------
class MatrixSubstrate3D:
    def __init__(self, L, eta, decay):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for MatrixSubstrate3D.")
        self.L = int(L)
        self.N = self.L ** 3
        self.eta = float(eta)
        self.decay = float(decay)
        self.links = {}
        self._build_links()
        self.psi = self._init_spinor_packet()

    def _idx(self, x, y, z):
        return x * self.L * self.L + y * self.L + z

    def _build_links(self):
        I2 = cp.eye(2, dtype=cp.complex128)
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    u = self._idx(x, y, z)
                    for dx, dy, dz in [(1,0,0),(0,1,0),(0,0,1)]:
                        nx = (x+dx) % self.L
                        ny = (y+dy) % self.L
                        nz = (z+dz) % self.L
                        v = self._idx(nx, ny, nz)
                        self.links[(u,v)] = I2.copy()
                        self.links[(v,u)] = I2.copy()

    def _init_spinor_packet(self):
        psi_host = np.zeros((self.N, 2), dtype=np.complex128)
        c = self.L // 2
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._idx(x, y, z)
                    r2 = (x-c)**2 + (y-c)**2 + (z-c)**2
                    amp = np.exp(-r2/2.0)
                    psi_host[idx] = [amp, 0.0]
        flat = psi_host.reshape(-1)
        flat /= np.linalg.norm(flat)
        return cp.asarray(flat)

    def hamiltonian(self):
        size = 2 * self.N
        H = cp.zeros((size, size), dtype=cp.complex128)
        for (u, v), U in self.links.items():
            ui, vi = 2*u, 2*v
            H[ui:ui+2, vi:vi+2] -= U
        return H

    def _flux_proxy(self):
        c = self.L // 2
        p1 = self._idx(c, c, c)
        p2 = self._idx((c+1)%self.L, c, c)
        p3 = self._idx((c+1)%self.L, (c+1)%self.L, c)
        p4 = self._idx(c, (c+1)%self.L, c)
        W = self.links[(p1,p2)] @ self.links[(p2,p3)] @ self.links[(p3,p4)] @ self.links[(p4,p1)]
        tr = cp.trace(W)
        return 2.0 - float(cp.abs(tr).get())

    def evolve(self, steps, dt):
        steps = int(steps); dt = float(dt)
        fluxes = []; spin_down = []
        for _ in range(steps):
            H = self.hamiltonian()
            Uop = cpla.expm(-1j * H * dt)
            self.psi = Uop @ self.psi
            psi_spin = self.psi.reshape(self.N, 2)
            for (u, v) in list(self.links.keys()):
                if u < v:
                    P_i = psi_spin[u]; P_j = psi_spin[v]
                    corr = cp.outer(P_i, cp.conj(P_j))
                    U_curr = self.links[(u,v)]
                    delta = (self.eta * corr - self.decay * (U_curr - cp.eye(2, dtype=cp.complex128))) * dt
                    U_new = U_curr + delta
                    self.links[(u,v)] = U_new
                    self.links[(v,u)] = U_new.conj().T
            fluxes.append(self._flux_proxy())
            total_prob = cp.sum(cp.abs(psi_spin)**2, axis=0)
            spin_down.append(float(total_prob[1].get()))
        return {"flux": fluxes, "spin_down": spin_down}

# ---------------------------------------------------------------------------
# 3. LivingLink1D (microscopic, CPU)
# ---------------------------------------------------------------------------
class LivingLink1D:
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
                # diagonal
                n_exc = bin(link_state).count("1")
                rows.append(u); cols.append(u); data.append(self.E_link * n_exc)
                # right hop
                if x < self.L-1:
                    flipped = link_state ^ (1 << x)
                    v = self._index(x+1, flipped)
                    rows.append(u); cols.append(v); data.append(-self.t_hop)
                # left hop
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
        return float(E[0]), V[:,0]

    @staticmethod
    def free_ground_energy(L, t_hop):
        return -2 * t_hop * np.cos(np.pi / (L+1))

    @staticmethod
    def vacuum_prob(psi, L, link_dim):
        psi = psi.reshape(L, link_dim)
        p0 = np.abs(psi[:,0])**2
        return float(np.sum(p0))

    @staticmethod
    def effective_hopping(E_int, E_free, t_hop):
        return -E_int / (-E_free) * t_hop

# ---------------------------------------------------------------------------
# 4. CombinedSubstrate3D (effective, scalar + matrix + stiffness)
# ---------------------------------------------------------------------------
class CombinedSubstrate3D:
    """
    Spinor field on 3D lattice with both scalar and matrix feedback.

    Links carry:
      - scalar amplitude t_ij (real)
      - 2x2 matrix U_ij (complex)
    The Hamiltonian couples them as H_ij = - t_ij U_ij.

    Scalar feedback uses total spinor magnitude per site.
    Matrix feedback uses spinor correlations, with an additional
    stiffness term proportional to E_link that penalizes deviations
    of U_ij from identity (effective living-link stiffness).
    """
    def __init__(self, L, eta_s, decay_s, eta_m, decay_m, E_link):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for CombinedSubstrate3D.")
        self.L = int(L)
        self.N = self.L ** 3
        self.eta_s = float(eta_s)
        self.decay_s = float(decay_s)
        self.eta_m = float(eta_m)
        self.decay_m = float(decay_m)
        self.E_link = float(E_link)

        self.t_links = {}
        self.U_links = {}
        self._build_links()
        self.psi = self._init_spinor_packet()

    def _idx(self, x, y, z):
        return x * self.L * self.L + y * self.L + z

    def _build_links(self):
        I2 = cp.eye(2, dtype=cp.complex128)
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    u = self._idx(x, y, z)
                    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                        nx = (x+dx) % self.L
                        ny = (y+dy) % self.L
                        nz = (z+dz) % self.L
                        v = self._idx(nx, ny, nz)
                        if (u,v) not in self.t_links:
                            self.t_links[(u,v)] = 1.0
                        if (u,v) not in self.U_links:
                            self.U_links[(u,v)] = I2.copy()

    def _init_spinor_packet(self):
        psi_host = np.zeros((self.N, 2), dtype=np.complex128)
        c = self.L // 2
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._idx(x, y, z)
                    r2 = (x-c)**2 + (y-c)**2 + (z-c)**2
                    amp = np.exp(-r2/2.0)
                    psi_host[idx] = [amp, 0.0]
        flat = psi_host.reshape(-1)
        flat /= np.linalg.norm(flat)
        return cp.asarray(flat)

    def hamiltonian(self):
        size = 2 * self.N
        H = cp.zeros((size, size), dtype=cp.complex128)
        for (u,v), t in self.t_links.items():
            U = self.U_links[(u,v)]
            ui, vi = 2*u, 2*v
            H[ui:ui+2, vi:vi+2] -= t * U
        return H

    def _flux_proxy(self):
        c = self.L // 2
        p1 = self._idx(c, c, c)
        p2 = self._idx((c+1)%self.L, c, c)
        p3 = self._idx((c+1)%self.L, (c+1)%self.L, c)
        p4 = self._idx(c, (c+1)%self.L, c)
        W = (
            self.U_links[(p1,p2)]
            @ self.U_links[(p2,p3)]
            @ self.U_links[(p3,p4)]
            @ self.U_links[(p4,p1)]
        )
        tr = cp.trace(W)
        return 2.0 - float(cp.abs(tr).get())

    def evolve(self, steps, dt):
        steps = int(steps); dt = float(dt)
        history = {
            "energy": [],
            "PR": [],
            "flux": [],
            "spin_down": [],
            "mean_t": [],
        }
        for _ in range(steps):
            H = self.hamiltonian()
            Uop = cpla.expm(-1j * H * dt)
            self.psi = Uop @ self.psi
            psi_spin = self.psi.reshape(self.N, 2)
            psi_mag = cp.sqrt(cp.sum(cp.abs(psi_spin)**2, axis=1)).get()

            # scalar feedback on t_links
            new_t = {}
            for (u,v), t in self.t_links.items():
                corr = psi_mag[u] * psi_mag[v]
                dt_val = (self.eta_s * corr - self.decay_s * (t - 1.0)) * dt
                new_t[(u,v)] = t + dt_val
            self.t_links = new_t

            # matrix feedback with stiffness
            for (u,v) in list(self.U_links.keys()):
                if u < v:
                    P_i = psi_spin[u]; P_j = psi_spin[v]
                    corr = cp.outer(P_i, cp.conj(P_j))
                    U_curr = self.U_links[(u,v)]
                    # extra stiffness term ~ E_link pushes toward identity
                    delta = (
                        self.eta_m * corr
                        - (self.decay_m + self.E_link) * (U_curr - cp.eye(2, dtype=cp.complex128))
                    ) * dt
                    U_new = U_curr + delta
                    self.U_links[(u,v)] = U_new
                    self.U_links[(v,u)] = U_new.conj().T

            # diagnostics
            E = float(cp.vdot(self.psi, H @ self.psi).real.get())
            PR = float((1.0 / cp.sum(cp.abs(self.psi)**4)).get())
            flux = self._flux_proxy()
            total_prob = cp.sum(cp.abs(psi_spin)**2, axis=0)
            p_down = float(total_prob[1].get())
            mean_t = np.mean(list(self.t_links.values()))

            history["energy"].append(E)
            history["PR"].append(PR)
            history["flux"].append(flux)
            history["spin_down"].append(p_down)
            history["mean_t"].append(mean_t)
        return history

# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
def run_scalar(cfg):
    model = ScalarSubstrate3D(cfg["L"], cfg["eta"], cfg["decay"])
    res = model.evolve(cfg["steps"], cfg["dt"])
    if cfg.get("plot", True):
        steps = np.arange(len(res["energy"]))
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].plot(steps, res["PR"])
        ax[0].set_xlabel("Step"); ax[0].set_ylabel("PR"); ax[0].set_title("Scalar: Participation")
        ax[1].plot(steps, res["energy"])
        ax[1].set_xlabel("Step"); ax[1].set_ylabel("Energy"); ax[1].set_title("Scalar: Energy")
        plt.tight_layout(); plt.show()

def run_matrix(cfg):
    model = MatrixSubstrate3D(cfg["L"], cfg["eta"], cfg["decay"])
    res = model.evolve(cfg["steps"], cfg["dt"])
    if cfg.get("plot", True):
        steps = np.arange(len(res["flux"]))
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].plot(steps, res["flux"])
        ax[0].set_xlabel("Step"); ax[0].set_ylabel("Flux proxy"); ax[0].set_title("Matrix: Flux")
        ax[1].plot(steps, res["spin_down"])
        ax[1].set_xlabel("Step"); ax[1].set_ylabel("P(down)"); ax[1].set_title("Matrix: Spin mixing")
        plt.tight_layout(); plt.show()

def run_living(cfg):
    L = cfg["L"]; t_hop = cfg["t_hop"]; E_vals = cfg["E_link_values"]
    E_free = LivingLink1D.free_ground_energy(L, t_hop)
    records = []
    for E_link in E_vals:
        model = LivingLink1D(L, t_hop, E_link)
        E0, psi0 = model.ground_state(num_eigs=cfg["num_eigs"])
        t_eff = LivingLink1D.effective_hopping(E0, E_free, t_hop)
        alpha_meas = t_eff / t_hop
        alpha_th = t_hop / E_link
        Pvac = LivingLink1D.vacuum_prob(psi0, L, model.link_dim)
        records.append({
            "E_link": E_link,
            "E_ground": E0,
            "alpha_measured": alpha_meas,
            "alpha_theory": alpha_th,
            "P_vac": Pvac,
        })
    if cfg.get("plot", True):
        E = np.array([r["E_link"] for r in records], dtype=float)
        a_meas = np.array([r["alpha_measured"] for r in records], dtype=float)
        a_th = np.array([r["alpha_theory"] for r in records], dtype=float)
        Pvac = np.array([r["P_vac"] for r in records], dtype=float)

        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].loglog(E, a_meas, "o-", label="Measured α")
        ax[0].loglog(E, a_th, "--", label="t_hop / E_link")
        ax[0].set_xlabel("E_link"); ax[0].set_ylabel("α"); ax[0].legend(); ax[0].grid(True, which="both")
        ax[1].semilogx(E, Pvac, "o-")
        ax[1].set_xlabel("E_link"); ax[1].set_ylabel("P(vacuum)"); ax[1].set_ylim(0,1.05); ax[1].grid(True, which="both")
        plt.tight_layout(); plt.show()
    return records

def run_combined(cfg):
    model = CombinedSubstrate3D(
        L=cfg["L"],
        eta_s=cfg["eta_scalar"],
        decay_s=cfg["decay_scalar"],
        eta_m=cfg["eta_matrix"],
        decay_m=cfg["decay_matrix"],
        E_link=cfg["E_link"],
    )
    res = model.evolve(cfg["steps"], cfg["dt"])
    if cfg.get("plot", True):
        steps = np.arange(len(res["energy"]))
        fig, ax = plt.subplots(2,2, figsize=(10,8))
        ax[0,0].plot(steps, res["PR"])
        ax[0,0].set_xlabel("Step"); ax[0,0].set_ylabel("PR"); ax[0,0].set_title("Combined: Participation")
        ax[0,1].plot(steps, res["energy"])
        ax[0,1].set_xlabel("Step"); ax[0,1].set_ylabel("Energy"); ax[0,1].set_title("Combined: Energy")
        ax[1,0].plot(steps, res["flux"])
        ax[1,0].set_xlabel("Step"); ax[1,0].set_ylabel("Flux proxy"); ax[1,0].set_title("Combined: Flux")
        ax[1,1].plot(steps, res["mean_t"])
        ax[1,1].set_xlabel("Step"); ax[1,1].set_ylabel("⟨t_ij⟩"); ax[1,1].set_title("Combined: Mean link amplitude")
        plt.tight_layout(); plt.show()
    return res

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mode = CONFIG["experiment"].lower()
    if mode == "scalar":
        run_scalar(CONFIG["scalar"])
    elif mode == "matrix":
        run_matrix(CONFIG["matrix"])
    elif mode == "living":
        run_living(CONFIG["living"])
    elif mode == "combined":
        run_combined(CONFIG["combined"])
    else:
        raise ValueError(f"Unknown experiment type: {CONFIG['experiment']}")
