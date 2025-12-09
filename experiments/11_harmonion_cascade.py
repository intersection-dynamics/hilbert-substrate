import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt


class SubstrateSim:
    def __init__(
        self,
        L: int = 11,
        Q: float = 1.0,
        g_init: float = 0.5,
        kappa: float = 0.015,
        g_max: float = 7.0,
        dS_trigger: float = 0.2,
        cooldown_steps: int = 3,
    ):
        """
        L              : linear lattice size (L^3 sites)
        Q              : monopole charge / coupling strength
        g_init         : initial gauge stiffness everywhere
        kappa          : feedback strength (how strongly commits stiffen geometry)
        g_max          : saturation scale for stiffness (logistic cap)
        dS_trigger     : entropy increase required to trigger a commit event
        cooldown_steps : minimum number of steps between commit events
        """
        self.L = L
        self.N = L**3
        self.Q = Q
        self.kappa = kappa
        self.g_max = g_max
        self.dS_trigger = dS_trigger
        self.cooldown_steps = cooldown_steps

        # Gauge stiffness field
        self.g_map = np.full((L, L, L), g_init, dtype=np.float64)

        # Derived Memory Limit (Eq. 1 from PDF)
        self.epsilon = np.sqrt(2.0 / self.N)

        # State Vector (Wavefunction)
        self.psi = np.zeros((L, L, L), dtype=np.complex128)
        self.initialize_wavepacket()

        # Entropy baseline and cooldown
        self.S_min = self.get_entropy()     # running minimum entropy
        self.commit_cooldown = 0            # steps remaining until next allowed commit

        # Metrics History
        self.history = {
            "time": [],
            "avg_g": [],
            "max_g": [],
            "entropy": [],
            "commits": [],
        }

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------
    def initialize_wavepacket(self):
        """Place a Gaussian wavepacket near the monopole (center)."""
        center = self.L // 2
        x, y, z = np.indices((self.L, self.L, self.L))
        r2 = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2

        # Start slightly dispersed to allow "hunting"
        self.psi = np.exp(-r2 / 4.0).astype(np.complex128)
        self.psi /= np.linalg.norm(self.psi)

    # -------------------------------------------------
    # Hamiltonian Construction
    # -------------------------------------------------
    def build_hamiltonian(self):
        """
        Constructs H = T + V

        T: Laplacian (3D 7-point stencil, hopping = -1)
        V: Potential well depth scaled by local gauge stiffness 'g'
           V(r) = -g(r) * Q / r
        """
        diagonals = []
        offsets = []

        # Main diagonal
        diagonals.append(np.full(self.N, 6.0))
        offsets.append(0)

        # X-hopping
        diagonals.append(np.full(self.N - 1, -1.0))
        offsets.append(1)
        diagonals.append(np.full(self.N - 1, -1.0))
        offsets.append(-1)

        # Y-hopping
        diagonals.append(np.full(self.N - self.L, -1.0))
        offsets.append(self.L)
        diagonals.append(np.full(self.N - self.L, -1.0))
        offsets.append(-self.L)

        # Z-hopping
        diagonals.append(np.full(self.N - self.L**2, -1.0))
        offsets.append(self.L**2)
        diagonals.append(np.full(self.N - self.L**2, -1.0))
        offsets.append(-self.L**2)

        T = sp.diags(diagonals, offsets, shape=(self.N, self.N))

        # Potential
        center = self.L // 2
        x, y, z = np.indices((self.L, self.L, self.L))
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)

        # Regularize the core
        r[center, center, center] = 0.5

        # V = - g(r) * Q / r
        V_grid = -self.g_map * (self.Q / r)
        V = sp.diags(V_grid.flatten(), 0, shape=(self.N, self.N))

        return T + V, V_grid.flatten()

    # -------------------------------------------------
    # Feedback / Commit Rule
    # -------------------------------------------------
    def harmonion_feedback(self, local_energies):
        """
        Harmonion feedback loop with logistic saturation:

        1. Detect amplitudes < epsilon (thermodynamically irrelevant tail).
        2. Discard them (Commit/Collapse).
        3. Stiffen 'g' locally, but with a logistic cap so that g <= g_max.

           Δg ∝ κ * (ΔE / ε) * (1 - g / g_max)_+

        Returns the number of committed sites.
        """
        psi_flat = self.psi.flatten()
        amplitudes = np.abs(psi_flat)

        # 1. Identify commits
        commit_mask = (amplitudes < self.epsilon) & (amplitudes > 1e-15)
        n_commits = int(np.sum(commit_mask))
        if n_commits == 0:
            return 0

        # 2. Energy drop from commits (probability * |local potential|)
        probs_dropped = amplitudes[commit_mask] ** 2
        energies_dropped = local_energies[commit_mask]
        delta_E = np.sum(probs_dropped * np.abs(energies_dropped))

        if delta_E <= 0.0:
            # Collapse tails anyway, but no geometry update
            psi_flat[commit_mask] = 0.0
            norm = np.linalg.norm(psi_flat)
            if norm > 0:
                psi_flat /= norm
            self.psi = psi_flat.reshape((self.L, self.L, self.L))
            return n_commits

        # 3. Base feedback strength from energy release
        base_feedback = self.kappa * (delta_E / self.epsilon)

        # 4. Apply local logistic saturation around commit sites
        commit_indices = np.where(commit_mask)[0]
        xyz = np.array(
            np.unravel_index(commit_indices, (self.L, self.L, self.L))
        ).T

        for cx, cy, cz in xyz:
            x_min, x_max = max(0, cx - 2), min(self.L, cx + 3)
            y_min, y_max = max(0, cy - 2), min(self.L, cy + 3)
            z_min, z_max = max(0, cz - 2), min(self.L, cz + 3)

            block = self.g_map[x_min:x_max, y_min:y_max, z_min:z_max]

            # Logistic saturation factor: (1 - g/g_max), clipped at 0
            growth_factor = 1.0 - (block / self.g_max)
            growth_factor = np.clip(growth_factor, 0.0, None)

            if np.all(growth_factor <= 0.0):
                continue

            self.g_map[x_min:x_max, y_min:y_max, z_min:z_max] += (
                base_feedback * growth_factor
            )

        # 5. Collapse tails and renormalize
        psi_flat[commit_mask] = 0.0
        norm = np.linalg.norm(psi_flat)
        if norm > 0:
            psi_flat /= norm
        self.psi = psi_flat.reshape((self.L, self.L, self.L))

        return n_commits

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------
    def get_entropy(self) -> float:
        """Spatial Shannon entropy."""
        probs = np.abs(self.psi.flatten()) ** 2
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        return float(-np.sum(probs * np.log(probs)))

    # -------------------------------------------------
    # Main Evolution Loop
    # -------------------------------------------------
    def run(self, steps: int = 200, dt: float = 0.1, print_every: int = 10):
        print(f"--- EXPERIMENT 11: Harmonion Genesis (Entropy-Gated) ---")
        print(f"Lattice: {self.L}^3 (N={self.N})")
        print(f"Memory Limit (epsilon): {self.epsilon:.5f}")
        print(f"Initial g: {self.g_map.mean():.3f}")
        print(f"Target Goldilocks Range: g ~ 5–7 (g_max = {self.g_max})")
        print(f"Feedback kappa: {self.kappa}")
        print(f"Entropy trigger ΔS: {self.dS_trigger}")
        print(f"Cooldown steps: {self.cooldown_steps}")
        print("-" * 70)

        for t in range(steps):
            # 1. Build Hamiltonian (geometry may change over time)
            H, local_energies = self.build_hamiltonian()

            # 2. Unitary evolution
            psi_flat = self.psi.flatten()
            psi_new = expm_multiply(-1j * H * dt, psi_flat)
            self.psi = psi_new.reshape((self.L, self.L, self.L))

            # 3. Entropy after unitary
            S_unitary = self.get_entropy()

            # 4. Decide whether to commit
            commits = 0
            S_final = S_unitary

            if (
                (S_unitary - self.S_min) > self.dS_trigger
                and self.commit_cooldown == 0
            ):
                # Fire a commit event
                commits = self.harmonion_feedback(local_energies)
                S_final = self.get_entropy()

                # Update running minimum entropy if we dropped lower
                if S_final < self.S_min:
                    self.S_min = S_final

                # Start cooldown
                self.commit_cooldown = self.cooldown_steps
            else:
                # No commit; relax cooldown if active
                if self.commit_cooldown > 0:
                    self.commit_cooldown -= 1

            # 5. Logging
            avg_g = float(np.mean(self.g_map))
            max_g = float(np.max(self.g_map))

            self.history["time"].append(t)
            self.history["avg_g"].append(avg_g)
            self.history["max_g"].append(max_g)
            self.history["entropy"].append(S_final)
            self.history["commits"].append(commits)

            if (t % print_every) == 0:
                print(
                    f"Step {t:3d} | "
                    f"S: {S_final:.3f} (min={self.S_min:.3f}) | "
                    f"Max g: {max_g:.3f} | "
                    f"Avg g: {avg_g:.3f} | "
                    f"Commits: {commits} | "
                    f"Cooldown: {self.commit_cooldown}"
                )

        return self.history


# -------------------------------------------------
# Execution
# -------------------------------------------------
if __name__ == "__main__":
    sim = SubstrateSim(
        L=11,
        Q=1.0,
        g_init=0.5,
        kappa=0.015,      # feedback strength
        g_max=7.0,        # logistic cap (Goldilocks top end)
        dS_trigger=0.2,   # how much S must rise to trigger a commit
        cooldown_steps=3, # min steps between commit events
    )

    data = sim.run(steps=200, dt=0.2, print_every=10)

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: The rise and saturation of structure (g)
    ax1.plot(
        data["time"],
        data["max_g"],
        label="Max Stiffness (Core)",
        color="crimson",
        linewidth=2,
    )
    ax1.plot(
        data["time"],
        data["avg_g"],
        label="Avg Stiffness (Background)",
        color="blue",
        linestyle="--",
    )
    ax1.axhline(
        y=5.0,
        color="green",
        linestyle=":",
        label="Goldilocks Zone Start (Q=5)",
    )
    ax1.set_ylabel("Gauge Stiffness (g)")
    ax1.set_title("Experiment 11: Emergent Stability via Harmonion Feedback")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: The thermodynamics (entropy & commits)
    ax2.plot(
        data["time"],
        data["entropy"],
        label="Spatial Entropy",
        color="purple",
    )
    ax2_twin = ax2.twinx()
    ax2_twin.bar(
        data["time"],
        data["commits"],
        alpha=0.3,
        color="gray",
        label="Memory Commits",
        width=1.0,
    )
    ax2.set_ylabel("Entropy (S)")
    ax2_twin.set_ylabel("Commits per Step")
    ax2.set_xlabel("Time Steps")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
