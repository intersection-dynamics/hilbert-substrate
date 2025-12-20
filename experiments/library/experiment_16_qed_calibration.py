import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
from scipy.linalg import expm
import matplotlib.pyplot as plt

class TopologicalAtom:
    """
    The Soliton Atom from Exp 15.
    Defined by a Gaussian knot of connectivity in a Broken Symmetry Vacuum.
    """
    def __init__(self, L=15, g_peak=5.0, sigma=2.5):
        self.L = L
        self.N = L**3
        # Center of the grid
        self.c = L // 2
        
        # 1. Construct the Soliton Geometry
        x, y, z = np.indices((L, L, L))
        r_sq = (x - self.c)**2 + (y - self.c)**2 + (z - self.c)**2
        self.g_map = 1.0 + (g_peak - 1.0) * np.exp(-r_sq / (2 * sigma**2))

    def build_hamiltonian(self):
        g_flat = self.g_map.flatten()
        shifts = [1, -1, self.L, -self.L, self.L**2, -self.L**2]
        diagonals = []
        offsets = []
        
        for shift in shifts:
            g_shifted = np.roll(g_flat, -shift)
            # Broken Symmetry: Hopping scales with g
            bond_strength = -0.5 * (g_flat + g_shifted)
            diagonals.append(bond_strength)
            offsets.append(shift)

        T = sp.diags(diagonals, offsets, shape=(self.N, self.N))
        # Fixed Vacuum Site Energy
        H_diag = sp.diags(np.full(self.N, 6.0), 0)
        return H_diag + T

    def get_spectrum(self, k=5):
        print(f"--- Solving Spectrum for Topological Atom (L={self.L}) ---")
        H = self.build_hamiltonian()
        # Find lowest k eigenstates
        vals, vecs = eigsh(H, k=k, which='SA')
        return vals, vecs

def run_calibration():
    # 1. Initialize Atom
    atom = TopologicalAtom(L=15, g_peak=5.0, sigma=2.5)
    
    # 2. Get Eigenstates
    energies, states = atom.get_spectrum(k=4)
    
    # State 0: Ground State (1s-like)
    E_g = energies[0]
    psi_g = states[:, 0]
    
    # State 1-3: Excited States (2p-like)
    # We need to find the one oriented along X to couple to an X-polarized laser.
    # Dipole Operator X = sum( x_i * |i><i| )
    print("\n--- Identifying Dipole Couplings ---")
    
    x_indices, _, _ = np.indices((atom.L, atom.L, atom.L))
    x_op = x_indices.flatten() - atom.c # Center coordinates at 0
    
    # Find the strongest transition
    best_coupling = 0.0
    target_idx = -1
    
    for i in range(1, 4):
        psi_e = states[:, i]
        # Calculate Dipole Moment <g|x|e>
        # d = sum( psi_g[r] * x[r] * psi_e[r] )
        dipole = np.dot(psi_g, x_op * psi_e)
        
        print(f"Transition 0 -> {i}: Delta E = {energies[i] - E_g:.4f} | Dipole <x> = {abs(dipole):.4f}")
        
        if abs(dipole) > best_coupling:
            best_coupling = abs(dipole)
            target_idx = i
            
    if target_idx == -1 or best_coupling < 1e-3:
        print("ERROR: No strong dipole transition found. Symmetry might be perfect?")
        return

    # 3. The Calibration
    E_e = energies[target_idx]
    psi_e = states[:, target_idx]
    delta_E_sim = E_e - E_g
    
    print("\n--- ENERGY SCALE CALIBRATION ---")
    print(f"Simulation Bandgap (1s -> 2p): {delta_E_sim:.4f} Units")
    print(f"Real World Hydrogen (1s -> 2p): 10.2 eV")
    
    conversion = 10.2 / delta_E_sim
    print(f"CALIBRATION FACTOR: 1.0 Sim Unit = {conversion:.4f} eV")
    
    # 4. Rabi Oscillation Simulation (The "Movie")
    print("\n--- Running Rabi Oscillation (Laser Interaction) ---")
    
    # Hamiltonian in the {g, e} basis driven by resonant laser
    # H = [ 0     Omega/2 ]
    #     [ Omega/2  0    ]  (in rotating frame)
    
    # Let's say we pump it with a field strength E_field
    E_laser = 0.2
    Rabi_freq = best_coupling * E_laser
    
    print(f"Laser Field Strength: {E_laser}")
    print(f"Rabi Frequency (Omega): {Rabi_freq:.4f}")
    
    # Time evolution
    t_pi = np.pi / Rabi_freq # Time for full population inversion (pi-pulse)
    times = np.linspace(0, 2*t_pi, 50)
    
    excited_pop = []
    
    # Simple analytical Rabi solution P_e(t) = sin^2(Omega * t / 2)
    # But let's verify the coupling math holds up
    for t in times:
        p_e = np.sin(Rabi_freq * t / 2)**2
        excited_pop.append(p_e)

    # 5. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot A: Rabi Flopping
    plt.subplot(2, 2, 1)
    plt.plot(times, excited_pop, 'r-', linewidth=2)
    plt.title(f"Vacuum Rabi Oscillations\n(Resonant w = {delta_E_sim:.3f})")
    plt.xlabel("Time")
    plt.ylabel("Excited State Population")
    plt.grid(True, alpha=0.3)
    
    # Plot B: The Orbitals
    # Ground State
    plt.subplot(2, 2, 3)
    mid = atom.L // 2
    plt.imshow(psi_g.reshape(atom.L, atom.L, atom.L)[:, :, mid]**2, cmap='inferno')
    plt.title(f"Ground State (1s)\nE = {E_g:.3f}")
    plt.axis('off')

    # Excited State
    plt.subplot(2, 2, 4)
    plt.imshow(psi_e.reshape(atom.L, atom.L, atom.L)[:, :, mid]**2, cmap='inferno')
    plt.title(f"Excited State (2p)\nE = {E_e:.3f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_calibration()