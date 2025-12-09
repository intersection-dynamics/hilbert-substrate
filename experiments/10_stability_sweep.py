import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

def run_sweep():
    print("--- EXPERIMENT 10 REVISITED: The Geometry of Trapping ---")
    print("Mapping the stability of Topological Knots (Solitons).")
    
    # 1. Configuration
    L = 15
    N = L**3
    res = 20
    
    # Sweep Parameters
    # g_peak: From Vacuum (1.0) to Strong Knot (5.0)
    g_vals = np.linspace(1.0, 5.0, res)
    # sigma: From Point Defect (0.5) to Large Blob (3.0)
    sigma_vals = np.linspace(0.5, 3.5, res)
    
    phase_map = np.zeros((res, res))
    
    # Pre-compute indices for speed
    c = L // 2
    x, y, z = np.indices((L, L, L))
    r_sq = (x - c)**2 + (y - c)**2 + (z - c)**2
    
    # 2. The Sweep
    print(f"Sweeping {res}x{res} geometries...")
    start_time = time.time()
    
    for i, sigma in enumerate(sigma_vals):
        # Pre-calculate Gaussian shape for this radius
        # Shape: 0 at inf, 1 at center (normalized later)
        gaussian_shape = np.exp(-r_sq / (2 * sigma**2))
        
        for j, g_peak in enumerate(g_vals):
            # A. Construct Geometry
            # g(r) = 1.0 + (g_peak - 1.0) * Gaussian(r)
            g_map = 1.0 + (g_peak - 1.0) * gaussian_shape
            
            # B. Build Hamiltonian (Broken Symmetry Logic)
            g_flat = g_map.flatten()
            shifts = [1, -1, L, -L, L**2, -L**2]
            diagonals = []
            offsets = []
            
            for shift in shifts:
                g_shifted = np.roll(g_flat, -shift)
                # Hopping scales with Connectivity
                bond_strength = -0.5 * (g_flat + g_shifted)
                diagonals.append(bond_strength)
                offsets.append(shift)
            
            T_off_diag = sp.diags(diagonals, offsets, shape=(N, N))
            
            # Fixed Site Energy (Vacuum Level)
            # This is the "Broken Symmetry" that creates the trap
            H_diagonal = sp.diags(np.full(N, 6.0), 0)
            
            H = H_diagonal + T_off_diag
            
            # C. Solve for Ground State
            try:
                # We want the lowest algebraic eigenvalue (SA)
                vals, _ = eigsh(H, k=1, which='SA')
                E_ground = vals[0]
                
                # In this model, Vacuum Energy ~ 0.
                # Bound states have E < 0.
                # We store magnitude of binding.
                if E_ground < -0.01:
                    phase_map[i, j] = -E_ground
                else:
                    phase_map[i, j] = 0.0 # Unbound
            except:
                phase_map[i, j] = 0.0

    print(f"Sweep complete in {time.time() - start_time:.1f}s.")

    # 3. Visualization
    plt.figure(figsize=(10, 8))
    
    # Plot Heatmap
    # X = g_peak, Y = sigma
    plt.imshow(phase_map, origin='lower', aspect='auto', cmap='magma',
               extent=[g_vals.min(), g_vals.max(), sigma_vals.min(), sigma_vals.max()])
    
    cbar = plt.colorbar()
    cbar.set_label("Binding Energy (Stability)")
    
    plt.xlabel("Peak Connectivity ($g_{max}$)")
    plt.ylabel("Knot Radius ($\sigma$)")
    plt.title("Phase Diagram of Topological Matter")
    
    # Annotate
    plt.text(2.0, 3.0, "Stable Solitons", color='white', ha='center', weight='bold')
    plt.text(1.5, 0.8, "Vacuum / Unbound", color='white', ha='center')
    
    # Mark Experiment 14 Result (Approx g=4.6, sigma~1.5)
    plt.plot(4.6, 1.5, 'gx', markersize=12, markeredgewidth=3, label="Exp 14 Result")
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    run_sweep()