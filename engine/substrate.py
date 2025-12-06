import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
import matplotlib.pyplot as plt

class UnifiedSubstrate:
    """
    THE RESONANCE ENGINE
    ====================
    A physics engine deriving emergent reality from a graph-based Hilbert Space.
    
    Axioms:
      1. Graph Realism: Space is a network of N sites.
      2. Gauge Geometry: Links are SU(2) matrices (Quaternions), not scalars.
      3. Unitary Evolution: Dynamics are defined by exp(-iHt).
    """
    
    def __init__(self, L_size=8, geometry='cube'):
        print(f"--- Initializing Unified Substrate (L={L_size}) ---")
        self.L = L_size
        self.N_sites = L_size**3
        # Spin-1/2 Basis: Dimension is 2 * Sites (Spin Up + Spin Down)
        self.dim = self.N_sites * 2 
        
        # Physics Constants
        self.HOPPING_AMP = 1.0
        self.MONOPOLE_STRENGTH = 0.0 # Default Vacuum (Flat)
        self.defect_pos = np.array([L_size//2, L_size//2, L_size//2])
        
        # SU(2) Generators (Pauli Matrices)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.Id = np.eye(2, dtype=complex)

        # 1. Build Topology
        self._build_graph()
        
        # 2. Initialize State
        self.psi = np.zeros(self.dim, dtype=np.complex128)
        self.H = None # Hamiltonian (Lazy load)

    def _idx(self, x, y, z):
        """Map 3D coordinates to linear index."""
        return x*self.L**2 + y*self.L + z

    def _build_graph(self):
        """Constructs the connectivity of 3D space."""
        print("  > Constructing Graph Topology...")
        self.adj = [[] for _ in range(self.N_sites)]
        self.site_coords = {}
        
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._idx(x, y, z)
                    self.site_coords[idx] = np.array([x, y, z])
                    
                    # 6 Neighbors (Open Boundary Conditions)
                    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if 0 <= nx < self.L and 0 <= ny < self.L and 0 <= nz < self.L:
                            n_idx = self._idx(nx, ny, nz)
                            dir_vec = np.array([dx, dy, dz])
                            self.adj[idx].append((n_idx, dir_vec))

    # =========================================================================
    # GAUGE GEOMETRY (The "Force" Layer)
    # =========================================================================
    
    def inject_defect(self, strength=2.0, position=None):
        """
        Creates a Topological Defect (Monopole/Proton) in the gauge field.
        This warps the connections between sites.
        """
        self.MONOPOLE_STRENGTH = strength
        if position is not None:
            self.defect_pos = np.array(position)
        print(f"  > Gauge Field Warped: Monopole Strength Q={strength}")

    def _get_link_matrix(self, r_vec, dir_vec):
        """
        Calculates the SU(2) Holonomy (Parallel Transport) for a link.
        This is where the "Forces" come from.
        """
        if self.MONOPOLE_STRENGTH == 0:
            return self.Id # Flat Space
            
        # Vector from defect center to the link
        rel_vec = r_vec - self.defect_pos
        dist = np.linalg.norm(rel_vec) + 1e-6
        n_i = rel_vec / dist
        
        # Vector to the destination site
        dest_vec = rel_vec + dir_vec
        n_j = dest_vec / (np.linalg.norm(dest_vec) + 1e-6)
        
        # Axis of Rotation (Cross Product of directions)
        k = np.cross(n_i, n_j)
        sin_theta = np.linalg.norm(k)
        cos_theta = np.dot(n_i, n_j)
        
        if sin_theta < 1e-9: 
            return self.Id
            
        # The 'Twist' Angle (Berry Phase)
        theta = np.arccos(np.clip(cos_theta, -1, 1)) * self.MONOPOLE_STRENGTH
        k_hat = k / sin_theta
        
        # SU(2) Rotation: exp(-i * theta/2 * sigma.k)
        gen = k_hat[0]*self.sigma_x + k_hat[1]*self.sigma_y + k_hat[2]*self.sigma_z
        U = np.cos(theta/2)*self.Id - 1j*np.sin(theta/2)*gen
        return U

    # =========================================================================
    # DYNAMICS (The "Matter" Layer)
    # =========================================================================

    def build_hamiltonian(self):
        """
        Compiles the Physics Engine.
        H = -t * Sum ( c_i^dag * U_ij * c_j )
        """
        print("  > compiling Hamiltonian from Gauge Constraints...")
        # Use LIL for efficient structure building
        H = sp.lil_matrix((self.dim, self.dim), dtype=np.complex128)
        
        for idx in range(self.N_sites):
            r_vec = self.site_coords[idx]
            
            # Iterate neighbors
            for n_idx, d_vec in self.adj[idx]:
                # Get the geometric connection
                U = self._get_link_matrix(r_vec, d_vec)
                
                # Fill 2x2 Spin Block
                # Mapping: Site i, Spin s -> Index 2*i + s
                base_i = 2 * idx
                base_j = 2 * n_idx
                
                # Term: -t * U_ij
                # We multiply by 0.5 because we visit every bond twice (i->j and j->i)
                term = -self.HOPPING_AMP * 0.5 * U
                
                for s1 in range(2):
                    for s2 in range(2):
                        H[base_i + s1, base_j + s2] += term[s1, s2]
        
        self.H = H.tocsr() # Convert to CSR for fast math
        print(f"  > Hamiltonian Built ({self.dim}x{self.dim}).")

    def solve_eigenstates(self, k=5):
        """
        Solves the time-independent Schrödinger equation H|psi> = E|psi>.
        Returns: energies, vectors
        """
        print(f"  > Solving for lowest {k} Energy Eigenstates (Orbitals)...")
        if self.H is None: self.build_hamiltonian()
        
        vals, vecs = eigsh(self.H, k=k, which='SA')
        return vals, vecs

    def evolve(self, dt=0.1, steps=10):
        """
        Evolves the current state forward in time: psi(t) -> exp(-iHt) psi(0).
        """
        if self.H is None: self.build_hamiltonian()
        
        # Use Krylov subspace method for unitary evolution
        # Note: -1j for Schrodinger eq.
        self.psi = expm_multiply(-1j * self.H * (dt * steps), self.psi)

    # =========================================================================
    # DIAGNOSTICS & IO
    # =========================================================================

    def set_wavepacket(self, center=None, width=2.0, k_vec=[0,0,0], spin=[1,0]):
        """Initializes psi as a Gaussian Wavepacket."""
        if center is None: center = self.defect_pos
        
        print(f"  > Initializing Wavepacket at {center}...")
        c = np.array(center)
        k = np.array(k_vec)
        
        psi_spatial = np.zeros(self.N_sites, dtype=complex)
        for idx, pos in self.site_coords.items():
            dist = np.linalg.norm(pos - c)
            phase = np.dot(k, pos)
            psi_spatial[idx] = np.exp(-dist**2 / (2*width**2)) * np.exp(1j * phase)
            
        # Assign spinor components
        self.psi[0::2] = psi_spatial * spin[0] # Up
        self.psi[1::2] = psi_spatial * spin[1] # Down
        
        # Normalize
        self.psi /= np.linalg.norm(self.psi)

    def plot_density(self, filename=None, slice_axis='z', title="Quantum Density"):
        """
        Visualizes the probability density |psi|^2.
        Sums over spin states.
        """
        # Calculate Scalar Density
        prob = np.abs(self.psi)**2
        dens = prob[0::2] + prob[1::2]
        
        # Gridify
        grid = np.zeros((self.L, self.L, self.L))
        for idx, pos in self.site_coords.items():
            grid[int(pos[0]), int(pos[1]), int(pos[2])] = dens[idx]
            
        # Slice
        mid = self.L // 2
        if slice_axis == 'z': slice_img = grid[:, :, mid]
        elif slice_axis == 'y': slice_img = grid[:, mid, :]
        else: slice_img = grid[mid, :, :]
        
        plt.figure(figsize=(6, 5))
        plt.imshow(slice_img, origin='lower', cmap='magma', interpolation='gaussian')
        plt.colorbar(label="Probability Density")
        plt.title(title)
        plt.xlabel("X (Lattice sites)")
        plt.ylabel("Y (Lattice sites)")
        
        if filename:
            plt.savefig(filename)
            print(f"  > Plot saved to {filename}")
        else:
            plt.show()

if __name__ == "__main__":
    # Self-Test
    sim = UnifiedSubstrate(L_size=9)
    sim.inject_defect(strength=4.0)
    sim.build_hamiltonian()
    vals, vecs = sim.solve_eigenstates(k=1)
    sim.psi = vecs[:, 0]
    sim.plot_density(title="Ground State (Self-Test)")