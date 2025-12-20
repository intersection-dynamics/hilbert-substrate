import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt
import time

class BrokenSymmetrySweep:
    """
    Optimized simulation using Experiment 14 Physics (Fixed Site Energy).
    This correctly models the 'trap' depth needed for stability.
    """
    def __init__(self, mu, lam, L=9):
        self.L = L
        self.N = L**3
        self.g_map = np.ones((L, L, L), dtype=np.float64)
        
        self.mu = mu
        self.lam = lam
        self.epsilon_base = np.sqrt(2.0 / self.N)
        
        # Initialize Wavepacket
        c = self.L // 2
        x, y, z = np.indices((self.L, self.L, self.L))
        r2 = (x - c)**2 + (y - c)**2 + (z - c)**2
        self.psi = np.exp(-r2 / 4.0).astype(np.complex128)
        self.psi /= np.linalg.norm(self.psi)

    def build_hamiltonian(self):
        g_flat = self.g_map.flatten()
        shifts = [1, -1, self.L, -self.L, self.L**2, -self.L**2]
        diagonals = []
        offsets = []
        
        for shift in shifts:
            g_shifted = np.roll(g_flat, -shift)
            # Experiment 14 Logic: Variable Hopping
            bond_strength = -0.5 * (g_flat + g_shifted)
            diagonals.append(bond_strength)
            offsets.append(shift)

        T = sp.diags(diagonals, offsets, shape=(self.N, self.N))
        
        # Experiment 14 Logic: Fixed Site Energy (Broken Symmetry)
        # This creates the effective potential well.
        H_diag = sp.diags(np.full(self.N, 6.0), 0)
        
        return H_diag + T

    def run_check(self, steps=100):
        for _ in range(steps):
            H = self.build_hamiltonian()
            # Evolve
            self.psi = expm_multiply(-1j * H * 0.2, self.psi.flatten()).reshape((self.L, self.L, self.L))
            
            # Feedback
            amplitudes = np.abs(self.psi.flatten())
            commit_mask = amplitudes > self.epsilon_base
            
            if np.any(commit_mask):
                payment = amplitudes[commit_mask]**2
                current_g = self.g_map.flatten()[commit_mask]
                
                # Work Constraint
                delta_g = payment / (self.mu * np.maximum(current_g, 0.1))
                indices = np.where(commit_mask)[0]
                np.add.at(self.g_map.ravel(), indices, delta_g)

            # Decay
            self.g_map = 1.0 + (self.g_map - 1.0) * (1.0 - self.lam)
            
        # Metric: Peak Connectivity
        return np.max(self.g_map)

def run_sweep():
    print("--- EXPERIMENT 13b (Corrected): The Phase Diagram of Existence ---")
    print("Mapping the 'Death Line' using Broken Symmetry Physics...")
    
    res = 15 # Resolution
    mu_vals = np.linspace(0.1, 3.0, res)       # Work Cost
    lam_vals = np.linspace(0.001, 0.05, res)   # Decay Rate
    
    phase_map = np.zeros((res, res))
    
    start_time = time.time()
    
    for i, mu in enumerate(mu_vals):
        print(f"Row {i+1}/{res}...", end="\r")
        for j, lam in enumerate(lam_vals):
            sim = BrokenSymmetrySweep(mu=mu, lam=lam)
            final_g = sim.run_check(steps=100)
            
            # If g grows significantly (> 1.5), it's Matter.
            # If g stays near 1.0, it's Vacuum.
            phase_map[i, j] = final_g - 1.0
            
    print(f"\nSweep complete in {time.time() - start_time:.1f}s.")

    # Visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(phase_map, origin='lower', aspect='auto', cmap='inferno',
               extent=[lam_vals.min(), lam_vals.max(), mu_vals.min(), mu_vals.max()])
    
    plt.colorbar(label='Topological Density (g - 1)')
    plt.xlabel(r'Radiative Decay Rate ($\lambda$)')
    plt.ylabel(r'Geometric Work Cost ($\mu$)')
    plt.title('The Equation of State: Where Can Matter Exist?')
    
    # Plot Theoretical Limit curve roughly
    x_line = np.linspace(lam_vals.min(), lam_vals.max(), 100)
    y_line = 0.01 / x_line # Tune this constant based on visualization
    plt.plot(x_line, y_line, 'c--', linewidth=2, label='Stability Limit')
    plt.ylim(mu_vals.min(), mu_vals.max())
    
    plt.legend(loc='upper right')
    plt.savefig('fig3_phases.png')
    plt.show()

if __name__ == "__main__":
    run_sweep()