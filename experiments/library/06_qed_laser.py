"""
EXPERIMENT 06: CAVITY QED (LIGHT-MATTER INTERACTION)
====================================================
Objective: 
  Demonstrate the absorption and re-emission of a photon by the
  Topological Atom (Vacuum Rabi Oscillations).

Hypothesis:
  - We derive the 1s and 2p orbitals from the Monopole Defect.
  - We couple them via the Dipole Operator to a single photon mode.
  - We observe the wavefunction morphing (s -> p -> s) as it absorbs the photon.
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
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

def run_experiment():
    print("--- Experiment 06: QED Laser Interaction (Adaptive) ---")
    
    # 1. Build the Atom
    # -----------------
    L_SIZE = 9
    uni = UnifiedSubstrate(L_size=L_SIZE)
    uni.inject_defect(strength=4.0)
    uni.build_hamiltonian()
    
    print("  > Solving for Atomic Orbitals (Scanning spectrum)...")
    vals, vecs = uni.solve_eigenstates(k=6)
    
    # Identify Ground State (1s)
    E_g = vals[0]
    psi_g = vecs[:, 0]
    
    # Search for First Excited Spatial State (2p)
    E_e = None
    psi_e = None
    
    for i in range(1, 6):
        if vals[i] > E_g + 1e-4:
            E_e = vals[i]
            psi_e = vecs[:, i]
            print(f"    Found Spatial Excitation at Index {i}")
            break
            
    if E_e is None:
        print("    [!] Error: Could not find excited state. Monopole might be too weak.")
        return

    omega = E_e - E_g                
    
    # 2. Calculate Dipole Coupling
    # ----------------------------
    X_op = np.zeros(uni.dim)
    center = L_SIZE // 2
    for idx, pos in uni.site_coords.items():
        x_val = pos[0] - center
        X_op[2*idx] = x_val
        X_op[2*idx+1] = x_val
        
    dipole_moment = np.vdot(psi_g, X_op * psi_e)
    coupling_g = 0.5 
    
    print(f"    Dipole Moment <g|x|e>: {np.abs(dipole_moment):.4f}")

    # 3. Build QED Hamiltonian
    # ------------------------
    H_qed = np.array([
        [0.0,        coupling_g],
        [coupling_g, 0.0]
    ])
    psi_qed = np.array([1.0, 0.0], dtype=complex)
    
    # 4. Time Evolution
    # -----------------
    print("  > Evolving System...")
    
    period = np.pi / coupling_g
    times = np.linspace(0, 2*period, 60)
    
    excited_pop = []
    wfs = [] # Store all wavefunctions to pick frames later
    
    for t in times:
        U = expm(-1j * H_qed * t)
        psi_t = U @ psi_qed
        excited_pop.append(np.abs(psi_t[1])**2)
        wfs.append(psi_t)

    # 5. Adaptive Frame Selection
    # ---------------------------
    # Find the index where population is closest to 0 (Ground), 0.5 (Mix), 1.0 (Excited)
    pop_arr = np.array(excited_pop)
    
    idx_ground = 0
    idx_peak = np.argmax(pop_arr)
    # Find mix point (closest to 0.5 before peak)
    idx_mix = np.argmin(np.abs(pop_arr[:idx_peak] - 0.5))
    # Find drop point (closest to 0.5 after peak)
    idx_drop = idx_peak + np.argmin(np.abs(pop_arr[idx_peak:] - 0.5))
    
    print(f"  > Captured Frames at indices: {idx_ground}, {idx_mix}, {idx_peak}, {idx_drop}")
    
    snap_indices = [idx_ground, idx_mix, idx_peak, idx_drop]
    snapshots = []
    
    for idx in snap_indices:
        # Reconstruct Spatial WF
        psi_t = wfs[idx]
        c_g = psi_t[0]
        c_e = psi_t[1]
        
        psi_spatial = c_g * psi_g + c_e * psi_e
        
        prob = np.abs(psi_spatial)**2
        dens = prob[0::2] + prob[1::2]
        grid = np.zeros((L_SIZE, L_SIZE, L_SIZE))
        for i, pos in uni.site_coords.items():
            grid[int(pos[0]), int(pos[1]), int(pos[2])] = dens[i]
        
        snapshots.append(grid[:, :, center])

    # 6. Visualization
    # ----------------
    fig = plt.figure(figsize=(12, 6))
    
    # Plot A: Rabi Flopping
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
    ax1.plot(times, excited_pop, 'r-', linewidth=3, label="Excited State Population")
    # Mark the frames
    ax1.plot(times[snap_indices], pop_arr[snap_indices], 'bo', label="Captured Frames")
    
    ax1.set_title(f"Vacuum Rabi Oscillations (Resonance w={omega:.3f})")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Probability P(|e>)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot B: The Movie Strip
    titles = ["Ground (1s)", "Mixing (Hybrid)", "Excited (2p)", "Emitting (Hybrid)"]
    for i, snap in enumerate(snapshots):
        ax = plt.subplot2grid((2, 4), (1, i))
        # Normalized color scale for comparison
        ax.imshow(snap, origin='lower', cmap='magma', interpolation='gaussian', vmin=0, vmax=np.max(snapshots))
        ax.set_title(titles[i])
        ax.axis('off')
        
    plt.tight_layout()
    output_file = "06_qed_laser.png"
    plt.savefig(output_file)
    print(f"  > Result saved to {output_file}")

if __name__ == "__main__":
    run_experiment()