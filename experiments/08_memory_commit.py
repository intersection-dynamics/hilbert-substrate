import sys
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

# Add parent directory to path
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

def run_experiment():
    print("--- Experiment 08: Memory Resolution & The Emergence of Fact ---")
    print("Hypothesis: Time is a sequence of memory commits (Coarse-Graining).")
    
    # 1. Setup Universe
    L = 60
    uni = UnifiedSubstrate(L_size=L)
    uni.build_hamiltonian()
    
    # Physics Parameters
    # The 'Grain': The smallest probability the Substrate bothers to remember.
    MEMORY_GRAIN = 0.02 
    
    # 2. Initialize a Fuzzy Superposition
    # A wavepacket spreading out
    center = L // 2
    uni.set_wavepacket(center=[center, L//2, L//2], width=3.0, k_vec=[1.5, 0, 0])
    
    # 3. Evolution Loop
    # We compare "Unitary" (Standard QM) vs "Coarse Grained" (Your Theory)
    
    psi_unitary = uni.psi.copy()
    psi_memory  = uni.psi.copy()
    
    steps = 60
    dt = 0.5
    
    history_unitary = []
    history_memory = []
    commit_events = [] # The "Ticks" of time
    
    print(f"  > Simulating with Memory Grain threshold: {MEMORY_GRAIN}")
    
    for t in range(steps):
        # A. Standard Unitary Evolution (Infinite Resolution)
        # ---------------------------------------------------
        psi_unitary = expm_multiply(-1j * uni.H * dt, psi_unitary)
        
        # B. Coarse-Grained Evolution (The Commit)
        # ----------------------------------------
        # 1. Evolve potential (Calculation phase)
        psi_temp = expm_multiply(-1j * uni.H * dt, psi_memory)
        
        # 2. THE COMMIT (Resolution Event)
        # Calculate Probability Density (Summing Spin Up + Down)
        prob_full = np.abs(psi_temp)**2
        # Collapse spin dimensions to spatial dimension
        prob_spatial = prob_full[0::2] + prob_full[1::2]
        
        # Identify "Real" facts vs "Ghost" possibilities
        # Where is the spatial density high enough to warrant a memory slot?
        mask_spatial = prob_spatial > MEMORY_GRAIN
        
        # "N is counting resolution events"
        n_events = np.sum(mask_spatial) 
        commit_events.append(n_events)
        
        if n_events > 0:
            # Commit the Fact: Filter out low-probability sites
            # We must apply the mask to the FULL spinor (both up and down components)
            # Create full mask (interleaved)
            mask_full = np.zeros_like(prob_full, dtype=bool)
            mask_full[0::2] = mask_spatial
            mask_full[1::2] = mask_spatial
            
            psi_temp[~mask_full] = 0.0
            
            # Re-Normalize (The Substrate commits to this new reality)
            norm = np.linalg.norm(psi_temp)
            if norm > 1e-9:
                psi_memory = psi_temp / norm
            else:
                psi_memory = psi_temp # Should not happen if n_events > 0
        else:
            # Wave has spread too thin to be real!
            pass
            
        # Store for Viz
        # Project 3D -> 1D density
        
        # Unitary History
        p_u_full = np.abs(psi_unitary)**2
        p_u_spatial = p_u_full[0::2] + p_u_full[1::2]
        # Sum over Y and Z to get 1D profile along X
        p_u_1d = np.sum(p_u_spatial.reshape(L,L,L), axis=(1,2))
        history_unitary.append(p_u_1d)
        
        # Memory History
        p_m_full = np.abs(psi_memory)**2
        p_m_spatial = p_m_full[0::2] + p_m_full[1::2]
        p_m_1d = np.sum(p_m_spatial.reshape(L,L,L), axis=(1,2))
        history_memory.append(p_m_1d)

    # 4. Visualization
    # ----------------
    fig = plt.figure(figsize=(10, 8))
    
    # Plot 1: Standard QM (The Cloud)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(history_unitary, aspect='auto', origin='lower', cmap='Blues')
    ax1.set_title("Standard Unitary Physics\n(Infinite Memory / No Collapse)")
    ax1.set_ylabel("Time Steps")
    ax1.set_xlabel("Space")
    
    # Plot 2: Substrate Memory (The Trajectory)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(history_memory, aspect='auto', origin='lower', cmap='Reds')
    ax2.set_title(f"Coarse-Grained Physics\n(Grain={MEMORY_GRAIN}: Facts Only)")
    ax2.set_xlabel("Space")
    
    # Plot 3: The "Ticks" of Time
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(commit_events, 'k-o', linewidth=2)
    ax3.set_title("Resolution Events ('N') per Time Step")
    ax3.set_ylabel("Number of Commited Facts (N)")
    ax3.set_xlabel("Time Step")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = "08_memory_commit.png"
    plt.savefig(output_file)
    print(f"  > Result saved to {output_file}")

if __name__ == "__main__":
    run_experiment()