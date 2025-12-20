import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
import matplotlib.pyplot as plt

class DeuteriumFusion:
    def __init__(self, L=15, separation=4, total_mass=2.0):
        self.L = L
        self.N = L**3
        self.separation = separation
        self.total_mass = total_mass
        
        # --- Topology ---
        # Baseline Vacuum = 1.0
        self.g_map = np.ones((L, L, L), dtype=np.float64)
        
        # --- Physics (Broken Symmetry) ---
        self.mu = 1.0        # Work Cost
        self.lam = 0.005     # Decay
        self.epsilon_base = np.sqrt(2.0 / self.N)
        
        # Wavefunction
        self.psi = np.zeros((L, L, L), dtype=np.complex128)
        self.initialize_nucleons()
        
    def initialize_nucleons(self):
        c = self.L // 2
        offset = self.separation / 2.0
        x, y, z = np.indices((self.L, self.L, self.L))
        
        # Nucleon 1
        r1 = (x - (c - offset))**2 + (y - c)**2 + (z - c)**2
        # Nucleon 2
        r2 = (x - (c + offset))**2 + (y - c)**2 + (z - c)**2
        
        # Superposition of two packets
        # We start with localized packets
        psi_raw = np.exp(-r1 / 4.0) + np.exp(-r2 / 4.0)
        
        # NORMALIZE TO TOTAL MASS (Number of Particles)
        # If we are simulating 2 bosons, the integral of |psi|^2 should be 2.0
        current_mass = np.sum(np.abs(psi_raw)**2)
        scale_factor = np.sqrt(self.total_mass / current_mass)
        self.psi = psi_raw.astype(np.complex128) * scale_factor

    def build_hamiltonian(self):
        """
        Broken Symmetry Hamiltonian:
        Variable Hopping + Fixed Site Energy
        """
        g_flat = self.g_map.flatten()
        shifts = [1, -1, self.L, -self.L, self.L**2, -self.L**2]
        diagonals = []
        offsets = []
        
        for shift in shifts:
            g_shifted = np.roll(g_flat, -shift)
            # Attractive Kinetic Well
            bond_strength = -0.5 * (g_flat + g_shifted)
            diagonals.append(bond_strength)
            offsets.append(shift)

        T_off_diag = sp.diags(diagonals, offsets, shape=(self.N, self.N))
        
        # Fixed Vacuum Site Energy (6.0)
        H_diagonal = sp.diags(np.full(self.N, 6.0), 0)
        
        return H_diagonal + T_off_diag

    def run_stabilization(self, steps=250):
        for t in range(steps):
            H = self.build_hamiltonian()
            
            # Evolve
            psi_flat = expm_multiply(-1j * H * 0.1, self.psi.flatten())
            self.psi = psi_flat.reshape((self.L, self.L, self.L))
            
            # Renormalize to maintain Mass=2.0 (Unitary evolution usually preserves norm, 
            # but we explicitly enforce particle number conservation against numerical drift)
            current_mass = np.sum(np.abs(self.psi)**2)
            self.psi *= np.sqrt(self.total_mass / current_mass)
            
            # Feedback
            amplitudes = np.abs(self.psi.flatten())
            # Commit threshold scales with sqrt(N) but here we just check signal
            commit_mask = amplitudes > self.epsilon_base
            
            if np.any(commit_mask):
                payment = amplitudes[commit_mask]**2
                current_g = self.g_map.flatten()[commit_mask]
                
                # Work Constraint
                delta_g = payment / (self.mu * current_g)
                indices = np.where(commit_mask)[0]
                np.add.at(self.g_map.ravel(), indices, delta_g)

            # Decay
            self.g_map = 1.0 + (self.g_map - 1.0) * (1.0 - self.lam)
            
        # Return Ground State Energy of the final geometry
        H_final = self.build_hamiltonian()
        vals, _ = eigsh(H_final, k=1, which='SA')
        return vals[0]

def run_experiment():
    print("--- EXPERIMENT 12: Corrected Fusion Scan ---")
    print("Simulating interaction of 2 Particles (Total Mass = 2.0)")
    
    # 1. Establish Single Particle Baseline
    print("Calibrating Single Atom...")
    # Run simulation with Mass=1.0 to find energy of one isolated atom
    sim_single = DeuteriumFusion(L=13, separation=0, total_mass=1.0)
    E_single = sim_single.run_stabilization(steps=200)
    
    baseline = E_single * 2.0
    print(f"Energy of 1 Atom: {E_single:.4f}")
    print(f"Baseline (2 Isolated Atoms): {baseline:.4f}")
    
    # 2. Scan Separations
    separations = [2, 3, 4, 5, 6, 8]
    energies = []
    
    print("\nScanning Separations...")
    for d in separations:
        sim = DeuteriumFusion(L=13, separation=d, total_mass=2.0)
        E = sim.run_stabilization(steps=200)
        energies.append(E)
        print(f"d={d}: Energy={E:.4f} | Delta={E - baseline:.4f}")

    # 3. Visualization
    plt.figure(figsize=(9, 6))
    
    plt.plot(separations, energies, 'bo-', linewidth=2, label='Deuterium (2 Particles)')
    plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline (No Interaction)')
    
    plt.xlabel('Separation Distance')
    plt.ylabel('System Energy')
    plt.title('Emergence of the Strong Force (Fusion)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    min_E = min(energies)
    if min_E < baseline:
        print(f"\nSUCCESS: Fusion is Exothermic!")
        print(f"Binding Energy: {baseline - min_E:.4f}")
    else:
        print("\nRESULT: No Binding.")
        
    plt.show()

if __name__ == "__main__":
    run_experiment()