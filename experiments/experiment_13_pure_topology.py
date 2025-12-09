import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
import matplotlib.pyplot as plt

class PureSubstrate:
    def __init__(self, L=11):
        self.L = L
        self.N = L**3
        
        # --- Topology State ---
        self.g_map = np.ones((L, L, L), dtype=np.float64)
        
        # --- Physics Constants (Tuned for Ignition) ---
        self.mu = 0.5        # LOWERED COST (Was 1.0) - Makes building geometry cheaper
        self.lam = 0.005     # Decay rate
        self.alpha = 0.5     # SNR Horizon
        
        self.epsilon_base = np.sqrt(2.0 / self.N)
        
        self.psi = np.zeros((L, L, L), dtype=np.complex128)
        self.initialize_wavepacket()
        
        self.history = {'time': [], 'max_g': [], 'entropy': []}

    def initialize_wavepacket(self):
        c = self.L // 2
        x, y, z = np.indices((self.L, self.L, self.L))
        r2 = (x - c)**2 + (y - c)**2 + (z - c)**2
        # Start tighter to boost initial density
        self.psi = np.exp(-r2 / 2.0).astype(np.complex128)
        self.psi /= np.linalg.norm(self.psi)

    def build_hamiltonian(self):
        g_flat = self.g_map.flatten()
        shifts = [1, -1, self.L, -self.L, self.L**2, -self.L**2]
        diagonals = []
        offsets = []
        
        for shift in shifts:
            g_shifted = np.roll(g_flat, -shift)
            # Symmetric variable hopping
            bond_strength = -0.5 * (g_flat + g_shifted)
            diagonals.append(bond_strength)
            offsets.append(shift)

        T_off_diag = sp.diags(diagonals, offsets, shape=(self.N, self.N))
        
        # Enforce Laplacian conservation (Zero-point energy consistency)
        row_sums = np.array(np.abs(T_off_diag.sum(axis=1))).flatten()
        H_diagonal = sp.diags(row_sums, 0)
        
        return H_diagonal + T_off_diag

    def run(self, steps=300):
        print(f"--- EXPERIMENT 13 (Fixed): Pure Topological Matter ---")
        print(f"Lattice: {self.L}^3 | Mu: {self.mu} | Lambda: {self.lam}")
        
        for t in range(steps):
            H = self.build_hamiltonian()
            
            # Unitary Evolution
            psi_flat = expm_multiply(-1j * H * 0.2, self.psi.flatten())
            self.psi = psi_flat.reshape((self.L, self.L, self.L))
            
            # Feedback Loop
            amplitudes = np.abs(self.psi.flatten())
            commit_mask = amplitudes > self.epsilon_base
            
            if np.any(commit_mask):
                payment = amplitudes[commit_mask]**2
                current_g = self.g_map.flatten()[commit_mask]
                
                # Work Constraint
                delta_g = payment / (self.mu * current_g)
                
                indices = np.where(commit_mask)[0]
                np.add.at(self.g_map.ravel(), indices, delta_g)

            # Dissipation
            self.g_map = 1.0 + (self.g_map - 1.0) * (1.0 - self.lam)
            
            # Logging
            max_g = np.max(self.g_map)
            entropy = -np.sum(amplitudes**2 * np.log(amplitudes**2 + 1e-12))
            
            self.history['time'].append(t)
            self.history['max_g'].append(max_g)
            self.history['entropy'].append(entropy)
            
            if t % 25 == 0:
                print(f"Step {t:3d} | Connectivity (g): {max_g:.3f} | Entropy: {entropy:.3f}")

        return self.history

    def verify_stability(self):
        print("\n--- Running Spectral Reality Check ---")
        H_final = self.build_hamiltonian()
        
        # BUG FIX: Correctly unpack eigenvalues AND eigenvectors
        vals, vecs = eigsh(H_final, k=1, which='SA')
        
        ground_energy = vals[0]
        psi_ground = vecs[:, 0]
        
        print(f"Final Ground State Energy: {ground_energy:.4f}")
        
        # Check Localization (Inverse Participation Ratio)
        ipr = np.sum(np.abs(psi_ground)**4)
        print(f"Inverse Participation Ratio (Localization): {ipr:.4f}")
        
        if ground_energy < -0.01 or ipr > 0.1:
             print("RESULT: STABLE. Soliton formed.")
        else:
             print("RESULT: UNSTABLE. Ghost state.")

if __name__ == "__main__":
    sim = PureSubstrate(L=11)
    data = sim.run(steps=300)
    sim.verify_stability()
    
    plt.plot(data['time'], data['max_g'])
    plt.title('Experiment 13: Connectivity Evolution')
    plt.show()