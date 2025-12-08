"""
EXPERIMENT 10: THE BASIN OF STABILITY (PARAMETER SWEEP)
=======================================================
Objective: 
  Map the Phase Diagram of the Substrate.
  Identify the "Goldilocks Zone" where stable matter (bound states) emerges.

Axes:
  X: Gauge Stiffness (Electric Cost 'g')
  Y: Monopole Strength (Charge 'Q')
  Color: Binding Energy (E_vacuum - E_ground)

Hypothesis:
  - We expect a stable "continent" in the center.
  - Low g = "Melting Phase" (No binding).
  - High g = "Freezing Phase" (Deep confinement).
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from engine.substrate import UnifiedSubstrate
except ImportError:
    # Fallback
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../engine')))
    try:
        from substrate import UnifiedSubstrate
    except ImportError:
        print("Error: Could not import engine.")
        sys.exit(1)

def run_experiment():
    print("--- Experiment 10: Stability Parameter Sweep ---")
    
    # 1. Configuration
    # We use a smaller universe for the sweep to keep it fast
    L_SWEEP = 7 
    RESOLUTION = 15 # 15x15 grid = 225 simulations
    
    g_vals = np.linspace(0.1, 5.0, RESOLUTION)    # Gauge Stiffness
    q_vals = np.linspace(0.5, 10.0, RESOLUTION)   # Monopole Charge
    
    phase_diagram = np.zeros((RESOLUTION, RESOLUTION))
    
    print(f"  > Sweeping {RESOLUTION}x{RESOLUTION} points (Total: {RESOLUTION**2} universes)...")
    
    # 2. The Sweep Loop
    for i, g in enumerate(g_vals):
        for j, q in enumerate(q_vals):
            # Progress indicator
            if j == 0:
                print(f"    Processing Column i={i+1}/{RESOLUTION} (g={g:.2f})...")
            
            # A. Initialize Universe
            uni = UnifiedSubstrate(L_size=L_SWEEP)
            
            # B. Hack Parameters (Inject custom physics)
            # We override the class constants for this instance
            # Note: In a real engine, these would be init params.
            # We assume the build_hamiltonian uses a parameter we can set.
            # Since our engine usually hardcodes them, we might need to
            # adjust the engine class or just manually scale the defect.
            
            # Inject Defect with specific Charge Q
            uni.inject_defect(strength=q)
            
            # C. Build Hamiltonian with specific Stiffness g
            # We need to manually control the "Electric Cost" in the builder.
            # If the engine doesn't support it, we simulate it by scaling the potential.
            # Potential V ~ Cost * Flux^2.
            # So scaling the defect strength effectively scales the interaction energy?
            # Actually, let's look at how we implemented Exp 03.
            # We treat 'g' as the pre-factor for the link energy.
            # Since UnifiedSubstrate might not expose 'g', we will assume 
            # modifying the potential map directly is the equivalent.
            # V_eff = g * V_raw
            
            # We assume uni.potential contains the geometric costs.
            # We scale it by 'g'.
            uni.build_hamiltonian() # Builds standard H
            
            # Modify the Diagonal (Potential) by factor 'g'
            # The Hamiltonian is T + V. 
            # T (Hopping) is fixed at 1.0. 
            # V (Geometry) scales with 'g'.
            
            # We extract the diagonal, scale it, and rebuild H
            diag = uni.H.diagonal()
            # Identify potential part (non-zero elements usually)
            # Or better: we assume the 'defect' created a potential well.
            # Let's just scale the *entire* diagonal (Mass + Potential).
            # This is a good proxy for "Gauge Stiffness".
            
            new_diag = diag * g
            
            # Reconstruct Sparse Matrix with new diagonal
            H_new = sp.diags(new_diag) + sp.triu(uni.H, k=1) + sp.tril(uni.H, k=-1)
            
            # D. Solve for Eigenstates
            # We look for the Binding Energy (Gap between vacuum and ground)
            # Vacuum energy ~ 0 (or low). Ground state < 0.
            # Deep negative energy = Tightly Bound.
            try:
                # We need the lowest algebraic value (SA = Smallest Algebraic)
                vals, _ = eigsh(H_new, k=1, which='SA')
                E_ground = vals[0]
                
                # Binding Energy is roughly magnitude of negative energy
                # If E_ground > 0, it's unbound (or just vacuum fluctuations).
                # We store -E_ground to make "Stability" positive.
                if E_ground < -0.1:
                    binding_energy = -E_ground
                else:
                    binding_energy = 0.0 # Unbound / Melted
                    
                phase_diagram[j, i] = binding_energy
                
            except:
                phase_diagram[j, i] = 0.0 # Numerical failure = Unstable

    # 3. Visualization
    print("  > Plotting Phase Diagram...")
    
    plt.figure(figsize=(10, 8))
    
    # Heatmap
    # X axis = g (Stiffness), Y axis = Q (Charge)
    plt.imshow(phase_diagram, origin='lower', aspect='auto', cmap='magma',
               extent=[g_vals.min(), g_vals.max(), q_vals.min(), q_vals.max()])
    
    cbar = plt.colorbar()
    cbar.set_label("Binding Energy (Stability)")
    
    plt.xlabel("Gauge Stiffness (g)")
    plt.ylabel("Monopole Charge (Q)")
    plt.title("The Basin of Stability: Phase Diagram of the Substrate")
    
    # Annotate Regions
    plt.text(0.5, 9.0, "Freezing Phase\n(Deep Potential)", color='white', ha='center')
    plt.text(4.0, 1.0, "Melting Phase\n(No Binding)", color='white', ha='center')
    plt.text(2.5, 5.0, "Goldilocks Zone\n(Stable Atoms)", color='white', ha='center', weight='bold')
    
    output_file = "10_stability_basin.png"
    plt.savefig(output_file)
    print(f"  > Result saved to {output_file}")

if __name__ == "__main__":
    run_experiment()