import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

class TopologicalAtom:
    def __init__(self, L=21, g_peak=4.5, sigma=2.5):
        """
        L: Lattice size (Needs to be odd to have a center point)
        g_peak: Peak connectivity at the core (from Exp 14 results)
        sigma: Radius of the knot
        """
        self.L = L
        self.N = L**3
        self.g_map = np.ones((L, L, L), dtype=np.float64)
        
        # 1. Create the Frozen Soliton Geometry
        # We manually impose the stable geometry found in Exp 14
        c = L // 2
        x, y, z = np.indices((L, L, L))
        r_sq = (x - c)**2 + (y - c)**2 + (z - c)**2
        
        # Gaussian knot
        self.g_map = 1.0 + (g_peak - 1.0) * np.exp(-r_sq / (2 * sigma**2))

    def build_hamiltonian(self):
        """
        Broken Symmetry Hamiltonian (Exp 14 Physics)
        """
        g_flat = self.g_map.flatten()
        shifts = [1, -1, self.L, -self.L, self.L**2, -self.L**2]
        diagonals = []
        offsets = []
        
        for shift in shifts:
            g_shifted = np.roll(g_flat, -shift)
            # Variable Hopping (Attractive)
            bond_strength = -0.5 * (g_flat + g_shifted)
            diagonals.append(bond_strength)
            offsets.append(shift)

        T_off_diag = sp.diags(diagonals, offsets, shape=(self.N, self.N))
        
        # Fixed Vacuum Site Energy (Broken Symmetry)
        H_diag = sp.diags(np.full(self.N, 6.0), 0)
        
        return H_diag + T_off_diag

    def solve_orbitals(self, num_modes=9):
        print(f"--- Solving Spectrum for Topological Atom (L={self.L}) ---")
        print("Diagonalizing Hamiltonian...")
        
        H = self.build_hamiltonian()
        
        # Solve for lowest energy eigenstates (Smallest Algebraic)
        vals, vecs = eigsh(H, k=num_modes, which='SA')
        
        return vals, vecs

def plot_orbitals(vals, vecs, L):
    """
    Visualize 2D cross-sections of the 3D orbitals.
    """
    # Number of orbitals to plot
    n = len(vals)
    grid_size = int(np.ceil(np.sqrt(n)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    center = L // 2
    
    print("\n--- Orbital Analysis ---")
    
    for i in range(n):
        E = vals[i]
        psi = vecs[:, i].reshape((L, L, L))
        
        # Take a slice through the center (Z-plane)
        psi_slice = psi[:, :, center]
        density = np.abs(psi_slice)**2
        
        # Normalize for visualization
        density /= np.max(density)
        
        ax = axes[i]
        im = ax.imshow(density, cmap='inferno', interpolation='bicubic', origin='lower')
        ax.set_title(f"State {i}: E = {E:.3f}")
        ax.axis('off')
        
        # Print symmetry guess
        print(f"State {i}: E={E:.4f}")

    plt.suptitle("Emergent Electron Orbitals from Topological Geometry", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 21^3 lattice for better resolution of p/d orbital lobes
    atom = TopologicalAtom(L=21, g_peak=5.0, sigma=3.0)
    
    eigenvalues, eigenvectors = atom.solve_orbitals(num_modes=9)
    plot_orbitals(eigenvalues, eigenvectors, atom.L)