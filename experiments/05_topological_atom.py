"""
EXPERIMENT 05: THE TOPOLOGICAL ATOM
===================================
Objective: 
  Demonstrate that atomic orbitals (s, p, d) emerge naturally 
  from a topological defect (Monopole) in the Substrate.

Hypothesis:
  - A 'Proton' is a topological knot in the gauge field (Berry Phase).
  - An electron moving in this field will quantize into Spherical Harmonics.
  - We should see:
    - State 0: Sphere (1s)
    - State 1-3: Dumbbells (2p)
    - State 4+: Cloverleaves (3d)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so we can import the engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from engine.substrate import UnifiedSubstrate
except ImportError:
    print("\n[!] Error: Could not import 'engine.substrate'.")
    print("    Ensure you are running this from the repo root or 'experiments/' folder.")
    print("    And that 'engine/substrate.py' exists.")
    sys.exit(1)

def run_experiment():
    print("--- Experiment 05: The Topological Atom ---")
    
    # 1. Initialize Universe (High Resolution for Orbitals)
    # ---------------------------------------------------
    # L=11 or 13 is good for seeing the d-orbital lobes.
    L_SIZE = 13
    uni = UnifiedSubstrate(L_size=L_SIZE)
    
    # 2. Inject The "Proton"
    # ----------------------
    # We create a Monopole Defect at the center.
    # Strength = 4.0 creates a deep enough well to bind d-orbitals.
    uni.inject_defect(strength=4.0)
    
    # 3. Build Physics
    # ----------------
    uni.build_hamiltonian()
    
    # 4. Solve for Orbitals
    # ---------------------
    # We want the first 9 states (1s, 3x 2p, 5x 3d)
    NUM_STATES = 9
    vals, vecs = uni.solve_eigenstates(k=NUM_STATES)
    
    print(f"\n  [Result] Energy Spectrum:\n  {vals}")

    # 5. Visualization (The Zoo)
    # --------------------------
    print("  > plotting orbital shapes...")
    
    # We plot Z-slices through the center to visualize the lobes.
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    center_z = L_SIZE // 2
    
    for i in range(NUM_STATES):
        # Load state into universe for density calculation
        uni.psi = vecs[:, i]
        
        # Calculate 3D Density
        prob = np.abs(uni.psi)**2
        dens = prob[0::2] + prob[1::2] # Sum spins
        grid = dens.reshape((L_SIZE, L_SIZE, L_SIZE)) # Map to grid indices
        
        # Slice
        slice_img = grid[:, :, center_z]
        
        # Plot
        ax = axes[i]
        ax.imshow(slice_img, origin='lower', cmap='magma', interpolation='gaussian')
        
        # Attempt to label based on index (Heuristic)
        label = f"E={vals[i]:.2f}"
        if i == 0: label += " (1s)"
        elif 1 <= i <= 3: label += " (2p)"
        elif i >= 4: label += " (3d?)"
        
        ax.set_title(label)
        ax.axis('off')

    plt.suptitle(f"Emergent Atomic Orbitals from Monopole (Strength={uni.MONOPOLE_STRENGTH})", y=0.95)
    plt.tight_layout()
    
    output_file = "05_topological_atom.png"
    plt.savefig(output_file)
    print(f"  > Result saved to {output_file}")

if __name__ == "__main__":
    run_experiment()