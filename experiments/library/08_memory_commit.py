"""
EXPERIMENT 08: THE THERMODYNAMICS OF TIME (DERIVED)
===================================================
Objective: 
  Demonstrate that the "Arrow of Time" and "Classical Fact" emerge 
  naturally from the finite information capacity of the Universe.

Derivation:
  - We do NOT assume a collapse parameter.
  - We derive the "Vacuum Noise Limit" (epsilon) from the system size N.
  - epsilon = sqrt(2 / N)
  - Any probability amplitude below this noise floor is indistinguishable 
    from vacuum fluctuations and is "garbage collected" (forgotten).

Hypothesis:
  - This derived limit will produce a self-regulating "Sawtooth" entropy cycle.
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

# Path hack to import engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from engine.substrate import UnifiedSubstrate
except ImportError:
    # Fallback if running directly in the folder
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../engine')))
    try:
        from substrate import UnifiedSubstrate
    except ImportError:
        print("Error: Could not import engine.substrate. Check your folder structure.")
        sys.exit(1)

def get_shannon_entropy(psi):
    """Calculates Spatial Shannon Entropy S = -Sum p * log(p)"""
    # 1. Project to Spatial Probability (Sum spins)
    prob_full = np.abs(psi)**2
    prob_spatial = prob_full[0::2] + prob_full[1::2]
    
    # 2. Normalize 
    norm = np.sum(prob_spatial)
    if norm < 1e-9: return 0.0
    p = prob_spatial / norm
    
    # 3. Calculate Entropy
    p_safe = p[p > 1e-15] # Avoid log(0)
    S = -np.sum(p_safe * np.log(p_safe))
    return S

def run_experiment():
    print("--- Experiment 08: Derived Memory & Entropy Cycles ---")
    
    # 1. Setup Universe
    # -----------------
    L = 60
    uni = UnifiedSubstrate(L_size=L)
    uni.build_hamiltonian()
    
    # 2. THE DERIVATION
    # -----------------
    # N_spatial = L^3. 
    # The vacuum noise floor scales as 1/sqrt(Vol) for coherent states.
    N_sites = L**3
    MEMORY_GRAIN = np.sqrt(2.0 / N_sites)
    
    print(f"  > Universe Size (N): {N_sites} sites")
    print(f"  > Derived Vacuum Noise Limit (epsilon): {MEMORY_GRAIN:.6f}")
    print("    (Amplitudes below this are thermodynamically free to be forgotten)")
    
    # 3. Initialize Wavepacket
    # ------------------------
    center = L // 2
    uni.set_wavepacket(center=[center, L//2, L//2], width=2.5, k_vec=[1.8, 0, 0])
    
    # 4. Evolution Loop
    # -----------------
    steps = 80
    dt = 0.4
    
    psi_sys = uni.psi.copy()
    
    entropy_history = []
    density_map = []
    
    print("  > Evolving System...")
    
    for t in range(steps):
        # A. Unitary Step (Dispersion / Possibility Generation)
        psi_sys = expm_multiply(-1j * uni.H * dt, psi_sys)
        
        # B. Measure Entropy (The "Cost" of the state)
        S_pre = get_shannon_entropy(psi_sys)
        
        # C. The Memory Commit (Garbage Collection)
        # -----------------------------------------
        prob_full = np.abs(psi_sys)**2
        prob_spatial = prob_full[0::2] + prob_full[1::2]
        
        # Check against DERIVED limit
        mask_spatial = prob_spatial > MEMORY_GRAIN
        
        # If the wave has "tails" that are just noise...
        if np.sum(mask_spatial) > 0:
            # Apply Filter to full spinor
            mask_full = np.zeros_like(prob_full, dtype=bool)
            mask_full[0::2] = mask_spatial
            mask_full[1::2] = mask_spatial
            
            # Forget the noise
            psi_sys[~mask_full] = 0.0
            
            # Re-Normalize (Commit to Fact)
            norm = np.linalg.norm(psi_sys)
            if norm > 1e-9:
                psi_sys /= norm
            
        # D. Store Data
        entropy_history.append(S_pre)
        
        # Save 1D projection for visualization
        p_viz_spatial = np.abs(psi_sys)**2
        p_viz_spatial = p_viz_spatial[0::2] + p_viz_spatial[1::2]
        p_1d = np.sum(p_viz_spatial.reshape(L,L,L), axis=(1,2))
        density_map.append(p_1d)

    # 5. Visualization
    # ----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: The Emergent Classical Trajectory
    ax1.imshow(np.array(density_map).T, aspect='auto', origin='lower', cmap='inferno')
    ax1.set_ylabel("Space (X)")
    ax1.set_title(f"Emergent Trajectory (Derived Grain $\epsilon$={MEMORY_GRAIN:.5f})")
    
    # Plot 2: The Thermodynamic Heartbeat
    ax2.plot(entropy_history, 'b-o', linewidth=2, label="Spatial Entropy (S)")
    
    ax2.set_ylabel("Shannon Entropy (Uncertainty)")
    ax2.set_xlabel("Time Step")
    ax2.set_title("The Thermodynamics of Time: Information Generation vs. Erasure")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_file = "08_derived_memory.png"
    plt.savefig(output_file)
    print(f"  > Result saved to {output_file}")

if __name__ == "__main__":
    run_experiment()