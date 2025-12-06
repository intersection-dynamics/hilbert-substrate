"""
EXPERIMENT 03: DERIVATION OF INERTIA (F=ma)
===========================================
Objective: 
  Prove that 'Mass' is an emergent property derived from the 
  stiffness of the gauge field (Vacuum Energy).

Hypothesis:
  - We apply a constant Force (F) to a particle.
  - We vary the Gauge Stiffness (g).
  - We observe that Acceleration (a) decreases as g increases.
  - Conclusion: m_effective ~ g. Inertia is the 'drag' of history.
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import itertools

def run_experiment():
    print("--- Experiment 03: Derivation of Inertia ---")
    print("Hypothesis: Inertial Mass emerges from Gauge Field Stiffness.")

    # 1. Configuration
    # ----------------
    N_SITES = 6              # Length of 1D Universe
    N_LINKS = N_SITES - 1
    L_TRUNC = 1              # Gauge truncation (-1, 0, +1)
    FORCE = 0.5              # Constant 'Gravity' pulling right

    # 2. Basis Construction
    # ---------------------
    # State = (Position, Gauge_Configuration)
    states = []
    # Generate all possible link configurations
    gauge_configs = list(itertools.product(range(-L_TRUNC, L_TRUNC+1), repeat=N_LINKS))
    
    for pos in range(N_SITES):
        for g in gauge_configs:
            states.append((pos, g))
            
    dim = len(states)
    state_map = {s: i for i, s in enumerate(states)}
    print(f"  Hilbert Space Dimension: {dim}")

    # 3. Physics Engine
    # -----------------
    def build_hamiltonian(stiffness_g):
        H = np.zeros((dim, dim), dtype=np.complex128)
        
        for idx, (pos, links) in enumerate(states):
            # A. Potential Energy (Force)
            # V = -F * x
            pot_energy = -FORCE * pos
            H[idx, idx] += pot_energy
            
            # B. Gauge Strain Energy (The "Stiffness")
            # Cost to twist the vacuum: E = 0.5 * g * sum(l^2)
            strain = 0.5 * stiffness_g * sum([l**2 for l in links])
            H[idx, idx] += strain
            
            # C. Kinetic Energy (Gauged Hopping)
            # Moving modifies the link we cross (Peierls Substitution)
            if pos < N_SITES - 1:
                link_id = pos
                current_l = links[link_id]
                
                # Rule: Hop Right -> Decrease Link Flux (e^-iA)
                target_l = current_l - 1
                
                if abs(target_l) <= L_TRUNC:
                    new_links = list(links)
                    new_links[link_id] = target_l
                    new_state = (pos+1, tuple(new_links))
                    
                    if new_state in state_map:
                        j = state_map[new_state]
                        # Standard Hopping Amplitude t=1.0
                        H[j, idx] -= 1.0 
                        H[idx, j] -= 1.0 # Hermitian
        return H

    # 4. The Simulation Loop
    # ----------------------
    g_values = [0.0, 2.0, 10.0]
    results = {}
    times = np.linspace(0, 4, 40)
    
    print("  > Simulating for different vacuum stiffness values...")
    
    for g in g_values:
        H = build_hamiltonian(g)
        
        # Initial State: Particle at 0, Vacuum Links
        psi = np.zeros(dim, dtype=np.complex128)
        start_links = tuple([0]*N_LINKS)
        start_idx = state_map[(0, start_links)]
        psi[start_idx] = 1.0
        
        # Evolve
        vals, vecs = eigh(H)
        coeffs = vecs.conj().T @ psi
        
        positions = []
        for t in times:
            psi_t = vecs @ (coeffs * np.exp(-1j * vals * t))
            prob = np.abs(psi_t)**2
            
            # Calculate Expectation <X>
            avg_x = 0.0
            for i, p in enumerate(prob):
                avg_x += p * states[i][0]
            positions.append(avg_x)
            
        results[g] = positions

    # 5. Visualization
    # ----------------
    plt.figure(figsize=(10, 6))
    
    for g in g_values:
        pos_data = results[g]
        # Fit parabola x = 0.5 * a * t^2 to estimate 'a'?
        # Just plot trajectory.
        plt.plot(times, pos_data, linewidth=2.5, label=f"Stiffness g={g}")
        
    plt.title(f"Derivation of Inertia: Trajectories under Constant Force (F={FORCE})")
    plt.xlabel("Time")
    plt.ylabel("Particle Position <X>")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("03_derive_inertia.png")
    print("  > Result saved to 03_derive_inertia.png")

if __name__ == "__main__":
    run_experiment()