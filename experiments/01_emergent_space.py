import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

def run_experiment():
    print("--- Experiment 01: Emergent Space (Lieb-Robinson Bounds) ---")
    print("Hypothesis: A finite 'Speed of Light' (c) emerges from local graph connectivity.")
    
    # 1. Setup the Universe (1D Chain for clear visualization)
    L_SIZE = 60
    sites = np.arange(L_SIZE)
    center = L_SIZE // 2
    
    # 2. Build Hamiltonian (Nearest Neighbor Hopping)
    # H = -t * sum( |i><i+1| + |i+1><i| )
    dim = L_SIZE
    H = sp.diags([1.0, 1.0], [-1, 1], shape=(dim, dim), format='csr') * -1.0
    
    # 3. Initialize State (Localized Excitation at Center)
    psi_0 = np.zeros(dim, dtype=complex)
    psi_0[center] = 1.0
    
    # 4. Time Evolution
    t_max = 30
    steps = 60
    times = np.linspace(0, t_max, steps)
    
    # We store the density profile over time: rho(x, t)
    density_map = np.zeros((steps, dim))
    
    print(f"  > Evolving state over {steps} time steps...")
    for i, t in enumerate(times):
        # Unitary Evolution: psi(t) = exp(-iHt) psi(0)
        # using expm_multiply is efficient for sparse matrices
        psi_t = expm_multiply(-1j * H * t, psi_0)
        
        # Measure Density
        density_map[i, :] = np.abs(psi_t)**2

    # 5. Analysis: Extract the Light Cone
    # We find the 'wavefront' position at each time step
    # Threshold = 1% of max density
    threshold = 0.01
    fronts = []
    valid_times = []
    
    for i, t in enumerate(times):
        # Scan from center outwards
        rho = density_map[i, :]
        # Find furthest index where rho > threshold
        indices = np.where(rho > threshold)[0]
        if len(indices) > 0:
            # Furthest distance from center
            dist = np.max(np.abs(indices - center))
            fronts.append(dist)
            valid_times.append(t)
            
    # Calculate effective speed of light (Slope)
    # Fit line: dist = c * t
    if len(valid_times) > 5:
        fit = np.polyfit(valid_times, fronts, 1)
        c_eff = fit[0]
        print(f"  > Emergent Speed of Light (c): {c_eff:.4f} sites/time")
    else:
        c_eff = 0
        print("  > Signal did not propagate far enough to measure c.")

    # 6. Visualization
    plt.figure(figsize=(10, 8))
    
    # Plot A: The Light Cone (Space-Time Heatmap)
    plt.subplot(2, 1, 1)
    plt.imshow(density_map, aspect='auto', origin='lower', 
               extent=[0, L_SIZE, 0, t_max], cmap='inferno')
    plt.colorbar(label="Probability Density")
    plt.title("Emergence of the Light Cone (Causal Horizon)")
    plt.ylabel("Time")
    plt.xlabel("Space (Lattice Sites)")
    
    # Plot B: Linearity Check
    plt.subplot(2, 1, 2)
    plt.plot(valid_times, fronts, 'bo-', label='Wavefront Position')
    
    # Plot Fit
    if len(valid_times) > 0:
        fit_line = [c_eff * t for t in valid_times]
        plt.plot(valid_times, fit_line, 'r--', label=f'Linear Fit (c={c_eff:.2f})')
        
    plt.title(f"Lieb-Robinson Bound: Linear Propagation Velocity")
    plt.xlabel("Time")
    plt.ylabel("Distance from Source")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("01_emergent_space.png")
    print("  > Result saved to 01_emergent_space.png")

if __name__ == "__main__":
    run_experiment()