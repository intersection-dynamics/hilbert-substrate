"""
EXPERIMENT 07: NUCLEAR DYNAMICS (FUSION & DECAY)
================================================
Objective: 
  Demonstrate high-energy nuclear phenomena emerging from the Substrate.

Hypothesis 1 (Fusion):
  - A proton approaching a nucleus faces a Coulomb barrier (1/r).
  - At short range, the Strong Force (Yukawa) pulls it in.
  - We should see Quantum Tunneling: wavepacket leaks into the well.

Hypothesis 2 (Weak Decay):
  - A Neutron is an excited topological state; Proton is the ground state.
  - A weak coupling 'g' allows transitions (Flavor Changing).
  - We should see spontaneous population transfer.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
import matplotlib.pyplot as plt

def run_fusion_demo():
    print("\n[Part 1] Simulating Nuclear Fusion (Tunneling)...")
    
    # 1. Setup Space (Radial Coordinate r)
    R_POINTS = 300
    dr = 0.05
    r = np.linspace(dr, R_POINTS*dr, R_POINTS)
    
    # 2. Build Potentials
    # Repulsive Coulomb Barrier (EM)
    V_coulomb = 1.0 / r
    
    # Attractive Strong Force (Short range well)
    # Strength=20, Range=1.5
    V_strong = -20.0 * np.exp(-r / 1.5)
    
    V_total = V_coulomb + V_strong
    
    # 3. Hamiltonian H = -1/2m d^2/dr^2 + V(r)
    # Finite Difference Laplacian
    diag = np.ones(R_POINTS) * (1.0 / dr**2)
    off = np.ones(R_POINTS-1) * (-0.5 / dr**2)
    H_kin = sp.diags([off, diag, off], [-1, 0, 1], shape=(R_POINTS, R_POINTS))
    H_pot = sp.diags([V_total], [0])
    H = H_kin + H_pot
    
    # 4. Initial State: Incoming Proton
    # Gaussian centered at r=10, moving LEFT (k < 0)
    r0 = 10.0
    k0 = -2.5
    psi = np.exp(-0.5 * ((r - r0)**2) / 0.5) * np.exp(1j * k0 * r)
    psi /= np.linalg.norm(psi)
    
    # 5. Evolve
    dt = 0.05
    steps = 150
    snapshots = []
    
    print("  > Proton approaching Coulomb Barrier...")
    for t in range(steps):
        psi = expm_multiply(-1j * H * dt, psi)
        if t % 30 == 0:
            snapshots.append(np.abs(psi)**2)

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot Potentials (Scaled for visibility)
    plt.plot(r, V_total, 'k--', linewidth=2, label="Effective Potential (EM + Strong)")
    plt.axhline(0, color='k', linewidth=0.5)
    
    # Plot Wavepackets
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
    for i, dens in enumerate(snapshots):
        # Offset them vertically to show time progression (Waterfall plot)
        plt.fill_between(r, dens*5 + i*0.5, i*0.5, color=colors[i], alpha=0.5, label=f"t={i*30}")
        
    plt.ylim(-2, 5)
    plt.xlim(0, 12)
    plt.title("Nuclear Fusion: Quantum Tunneling into the Strong Force Well")
    plt.xlabel("Separation (fm)")
    plt.ylabel("Probability + Time Offset")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("07_nuclear_fusion.png")
    print("  > Fusion result saved to 07_nuclear_fusion.png")

def run_decay_demo():
    print("\n[Part 2] Simulating Weak Decay (Neutron -> Proton)...")
    
    # 1. Internal Hilbert Space
    # |0> = Neutron (Higher Energy)
    # |1> = Proton (Lower Energy)
    dim = 2
    H_weak = np.zeros((dim, dim), dtype=complex)
    
    E_neutron = 1.0
    E_proton = 0.2
    
    H_weak[0,0] = E_neutron
    H_weak[1,1] = E_proton
    
    # 2. The Weak Coupling (Flavor Mixing)
    # This off-diagonal term allows identity change
    g_weak = 0.1
    H_weak[0,1] = g_weak
    H_weak[1,0] = g_weak
    
    # 3. Evolution
    # Start as pure Neutron
    psi = np.array([1.0, 0.0], dtype=complex)
    
    times = np.linspace(0, 40, 100)
    prob_n = []
    prob_p = []
    
    print("  > Neutron evolving under Weak Interaction...")
    for t in times:
        U = expm(-1j * H_weak * t)
        psi_t = U @ psi
        prob_n.append(np.abs(psi_t[0])**2)
        prob_p.append(np.abs(psi_t[1])**2)
        
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(times, prob_n, 'b-', linewidth=3, label="Neutron Population")
    plt.plot(times, prob_p, 'r--', linewidth=3, label="Proton Population")
    
    plt.title("Weak Interaction: Spontaneous Beta Decay Model")
    plt.xlabel("Time")
    plt.ylabel("Population Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("07_weak_decay.png")
    print("  > Decay result saved to 07_weak_decay.png")

if __name__ == "__main__":
    print("--- Experiment 07: High Energy Physics ---")
    run_fusion_demo()
    run_decay_demo()