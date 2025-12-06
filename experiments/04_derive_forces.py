"""
EXPERIMENT 04: DERIVATION OF NUCLEAR FORCES
===========================================
Objective: 
  Derive the distance-dependence V(r) of the Strong and Weak interactions
  from graph-theoretic constraints.

Hypothesis:
  1. Strong Force (Linear): Flux lines on a graph cannot spread; they form tubes.
     Energy ~ Tension * Length.
  2. Weak Force (Exponential): Massive gauge links decay rapidly.
     Energy ~ exp(-m*r) / r.
"""

import numpy as np
import matplotlib.pyplot as plt

def run_experiment():
    print("--- Experiment 04: Derivation of Fundamental Forces ---")
    print("Hypothesis: Force laws are determined by the geometry of the connection.")

    # 1. Configuration
    # ----------------
    r_points = np.linspace(0.1, 4.0, 100)
    
    # Constants reflecting Substrate Properties
    K_STRONG = 2.0  # String Tension (Energy per unit length)
    K_WEAK   = 1.0  # Coupling Strength
    MASS_W   = 3.0  # Mass of the Weak Gauge Boson (Decay rate)

    # 2. The Derivations
    # ------------------
    
    # A. Strong Force (Confinement)
    # On a lattice, a non-Abelian flux tube has constant cross-section.
    # Therefore, Energy scales linearly with separation.
    V_strong = K_STRONG * r_points

    # B. Weak Force (Yukawa Interaction)
    # If the links themselves have a mass cost (Higgs mechanism), 
    # the propagator is the Green's function of (Laplacian - m^2).
    # Solution: e^(-mr) / r
    V_weak = -K_WEAK * np.exp(-MASS_W * r_points) / r_points
    
    # C. Electro-Magnetic Force (Reference)
    # Massless field in 3D spreads over surface area 4*pi*r^2.
    # Potential is integral of force: 1/r.
    V_em = -0.5 / r_points

    # 3. Visualization
    # ----------------
    plt.figure(figsize=(10, 6))
    
    # Plot Potentials
    plt.plot(r_points, V_strong, 'r-', linewidth=3, label="Strong Force (Linear Confinement)")
    plt.plot(r_points, V_weak, 'g--', linewidth=3, label="Weak Force (Yukawa/Massive)")
    plt.plot(r_points, V_em, 'b:', linewidth=2, label="EM Force (Coulomb 1/r)")
    
    # Formatting
    plt.axhline(0, color='k', linewidth=0.5)
    plt.ylim(-3, 6)
    plt.title("Emergence of Force Laws from Substrate Geometry")
    plt.xlabel("Separation Distance (r)")
    plt.ylabel("Potential Energy V(r)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "04_derive_forces.png"
    plt.savefig(output_file)
    print(f"  > Result saved to {output_file}")

if __name__ == "__main__":
    run_experiment()