import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
import matplotlib.pyplot as plt

class TrueTopologicalMatter:
    def __init__(self, L=11):
        self.L = L
        self.N = L**3
        
        self.g_map = np.ones((L, L, L), dtype=np.float64)
        
        # Physics Constants
        self.mu = 1.0        # Back to standard cost (Stability should be robust now)
        self.lam = 0.005     # Decay
        self.alpha = 0.5     # Horizon
        self.epsilon_base = np.sqrt(2.0 / self.N)
        
        self.psi = np.zeros((L, L, L), dtype=np.complex128)
        self.initialize_wavepacket()
        self.history = {'time': [], 'max_g': [], 'entropy': [], 'energy': []}

    def initialize_wavepacket(self):
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
        
        # 1. Variable Hopping (The Geometry)
        for shift in shifts:
            g_shifted = np.roll(g_flat, -shift)
            # Bond strength scales with geometry
            # Negative sign is critical for attractive kinetics
            bond_strength = -0.5 * (g_flat + g_shifted)
            diagonals.append(bond_strength)
            offsets.append(shift)

        T_off_diag = sp.diags(diagonals, offsets, shape=(self.N, self.N))
        
        # 2. Fixed Site Energy (The "Broken Symmetry")
        # We fix the site energy to the Vacuum Baseline (degree=6)
        # This means regions with g > 1 will have net NEGATIVE energy.
        # Vacuum (g=1) -> Energy ~ 0
        # Matter (g>1) -> Energy < 0
        H_diagonal = sp.diags(np.full(self.N, 6.0), 0)
        
        return H_diagonal + T_off_diag

    def run(self, steps=300):
        print(f"--- EXPERIMENT 14: Broken Symmetry Topology ---")
        print("Model: Fixed Site Energy. Variable Hopping acts as Effective Potential.")
        
        for t in range(steps):
            H = self.build_hamiltonian()
            
            # Evolve
            psi_flat = expm_multiply(-1j * H * 0.1, self.psi.flatten()) # Slower dt for stability
            self.psi = psi_flat.reshape((self.L, self.L, self.L))
            
            # Feedback
            amplitudes = np.abs(self.psi.flatten())
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
            
            # Logs
            max_g = np.max(self.g_map)
            entropy = -np.sum(amplitudes**2 * np.log(amplitudes**2 + 1e-12))
            
            # Quick energy estimate (Expectation Value <psi|H|psi>)
            # This tracks the depth of the trap
            energy_exp = np.vdot(psi_flat, H @ psi_flat).real
            
            self.history['time'].append(t)
            self.history['max_g'].append(max_g)
            self.history['entropy'].append(entropy)
            self.history['energy'].append(energy_exp)
            
            if t % 25 == 0:
                print(f"Step {t:3d} | G: {max_g:.3f} | S: {entropy:.3f} | E: {energy_exp:.3f}")

        return self.history

    def verify_stability(self):
        print("\n--- Reality Check ---")
        H_final = self.build_hamiltonian()
        vals, vecs = eigsh(H_final, k=1, which='SA')
        
        E_ground = vals[0]
        psi_ground = vecs[:, 0]
        ipr = np.sum(np.abs(psi_ground)**4)
        
        print(f"Ground Energy: {E_ground:.4f}")
        print(f"Localization (IPR): {ipr:.4f}")
        
        if E_ground < -1.0 and ipr > 0.1:
            print("RESULT: STABLE. Deep Topological Well confirmed.")
        else:
            print("RESULT: UNSTABLE.")

if __name__ == "__main__":
    sim = TrueTopologicalMatter(L=11)
    data = sim.run(steps=300)
    sim.verify_stability()
    
    # Dual Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(data['time'], data['max_g'], 'r', label='Connectivity (g)')
    ax1.set_ylabel('g')
    ax1.legend()
    ax1.set_title('Experiment 14: Topology vs Energy')
    
    ax2.plot(data['time'], data['energy'], 'b', label='System Energy')
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Time')
    ax2.legend()
    plt.show()