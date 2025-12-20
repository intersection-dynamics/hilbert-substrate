"""
Microscopic Living-Link Model (1D)
==================================

This module implements a minimal interacting quantum model in which a particle
moves on a 1D chain and its motion is coupled to two-level "living link"
degrees of freedom located on each bond.

The Hamiltonian is:

    H = H_link + H_hop

where

    H_link = E_link * sum_j n_j

and

    H_hop  = -t_hop * sum_x ( |x+1, ℓ ⊕ 2^x><x, ℓ|  +  |x-1, ℓ ⊕ 2^(x-1)><x, ℓ| ).

Each hop flips the state of the traversed link. Link excitations cost energy
E_link. This model allows us to quantify how vacuum stiffness renormalizes the
effective hopping amplitude of the particle.

The effective coupling α(E_link) is extracted by comparing the interacting
ground-state bandwidth to the free-particle bandwidth.

This file supports:
    - Ground-state energy computation (sparse eigenvalue solver)
    - Extraction of effective hopping t_eff(E_link)
    - Extraction of α(E_link) = t_eff / t_hop
    - Measurement of vacuum-sector probability
    - Sweep over E_link values and optional plotting

No external command-line interface is required. All configuration parameters
are supplied in the CONFIG dictionary below.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# ---------------------------------------------------------------------
# Configuration (edit as needed)
# ---------------------------------------------------------------------
CONFIG = {
    "L": 8,                            # number of lattice sites
    "t_hop": 1.0,                      # bare hopping amplitude
    "E_link_values": [0.5, 1, 2, 4, 8, 16, 32, 64],
    "num_eigs": 1,                     # compute lowest eigenvalue only
    "plot": True,                      # generate α(E_link) figure
    "figfile": "living_link_alpha.png"
}


# ---------------------------------------------------------------------
# Living-Link Model
# ---------------------------------------------------------------------
class LivingLink1D:
    """
    Microscopic living-link Hamiltonian on a 1D chain.

    Each bond hosts a two-level system. Hop operations flip the associated link.
    """

    def __init__(self, L, t_hop, E_link):
        """
        Parameters
        ----------
        L : int
            Number of lattice sites.
        t_hop : float
            Bare hopping amplitude.
        E_link : float
            Energy cost for a single link excitation.
        """
        self.L = L
        self.num_links = L - 1
        self.link_dim = 2 ** self.num_links
        self.dim = L * self.link_dim

        self.t_hop = float(t_hop)
        self.E_link = float(E_link)

    def _index(self, x, link_state):
        """Linear index for basis state |x, link_state>."""
        return x * self.link_dim + link_state

    def build_hamiltonian(self):
        """Construct the Hamiltonian matrix in CSR sparse format."""
        rows = []
        cols = []
        data = []

        for x in range(self.L):
            for link_state in range(self.link_dim):
                u = self._index(x, link_state)

                # Diagonal term: link excitation energy
                energy = self.E_link * bin(link_state).count("1")
                rows.append(u)
                cols.append(u)
                data.append(energy)

                # Hop right: x → x+1, flip link x
                if x < self.L - 1:
                    flipped = link_state ^ (1 << x)
                    v = self._index(x + 1, flipped)
                    rows.append(u)
                    cols.append(v)
                    data.append(-self.t_hop)

                # Hop left: x → x-1, flip link x-1
                if x > 0:
                    flipped = link_state ^ (1 << (x - 1))
                    v = self._index(x - 1, flipped)
                    rows.append(u)
                    cols.append(v)
                    data.append(-self.t_hop)

        H = csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))

        # Ensure Hermiticity
        H = 0.5 * (H + H.T)
        return H

    def ground_state(self, num_eigs=1):
        """
        Compute the lowest eigenvalue and eigenvector.

        Parameters
        ----------
        num_eigs : int
            Number of eigenvalues/vectors to compute (typically 1).

        Returns
        -------
        E0 : float
            Lowest eigenvalue.
        psi0 : ndarray
            Corresponding eigenvector.
        """
        H = self.build_hamiltonian()
        E, V = eigsh(H, k=num_eigs, which="SA")
        return float(E[0]), V[:, 0]

    @staticmethod
    def free_particle_ground_state_energy(L, t_hop):
        """
        Closed-form ground state energy for a free particle on an open chain.

        E0 = -2 t_hop cos(pi / (L+1))
        """
        return -2 * t_hop * np.cos(np.pi / (L + 1))

    @staticmethod
    def vacuum_probability(psi, L, link_dim):
        """Probability that all links are in the ground (unexcited) state."""
        psi = psi.reshape(L, link_dim)
        prob_x0 = np.abs(psi[:, 0]) ** 2
        return float(np.sum(prob_x0))

    @staticmethod
    def effective_hopping(E0_interacting, E0_free, t_hop):
        """
        Extract effective hopping by matching ground-state bandwidths.
        """
        return -E0_interacting / (-E0_free) * t_hop


# ---------------------------------------------------------------------
# Sweep and Analysis
# ---------------------------------------------------------------------
def run_sweep(config):
    L = config["L"]
    t_hop = config["t_hop"]
    E_values = config["E_link_values"]

    # Free-particle reference ground-state energy
    E0_free = LivingLink1D.free_particle_ground_state_energy(L, t_hop)

    results = []

    for E_link in E_values:
        model = LivingLink1D(L=L, t_hop=t_hop, E_link=E_link)
        E0_int, psi0 = model.ground_state(num_eigs=config["num_eigs"])
        t_eff = model.effective_hopping(E0_int, E0_free, t_hop)
        alpha_meas = t_eff / t_hop
        alpha_th = t_hop / E_link
        Pvac = model.vacuum_probability(psi0, L, model.link_dim)

        results.append({
            "E_link": E_link,
            "E_ground": E0_int,
            "alpha_measured": alpha_meas,
            "alpha_theory": alpha_th,
            "vacuum_prob": Pvac
        })

    return results


def plot_results(results, figfile):
    E = np.array([r["E_link"] for r in results])
    alpha_meas = np.array([r["alpha_measured"] for r in results])
    alpha_th = np.array([r["alpha_theory"] for r in results])

    plt.figure(figsize=(6, 5))
    plt.loglog(E, alpha_meas, "o-", label="Measured α")
    plt.loglog(E, alpha_th, "--", label="t_hop / E_link")
    plt.xlabel("E_link")
    plt.ylabel("α")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(figfile)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    results = run_sweep(CONFIG)

    if CONFIG["plot"]:
        plot_results(results, CONFIG["figfile"])
