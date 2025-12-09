import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

class ComplexSubstrate:
    def __init__(self, L=15, mode='SCALAR'):
        self.L = L
        self.N = L**3
        self.mode = mode # 'SCALAR' or 'VECTOR'
        
        # --- The Graph ---
        # Instead of just float numbers, links are now COMPLEX
        # We store 3 forward links per site: (x+1, y+1, z+1)
        # Shape: (N, 3)
        # Init with 1.0 (Real) -> Flat Vacuum
        self.links = np.ones((self.N, 3), dtype=np.complex128)
        
        # Physics Constants
        self.mu = 0.5        # Work Cost
        self.lam = 0.05      # Decay Rate (Fast decay to see dynamics)
        self.dt = 0.1
        
        # Wavefunction
        self.psi = np.zeros(self.N, dtype=np.complex128)
        self.init_collision()
        
        self.history = {'dist': []}

    def init_collision(self):
        # Create two wavepackets moving toward center
        c = self.L // 2
        x, y, z = np.indices((self.L, self.L, self.L))
        
        # Particle 1 (Left, moving Right)
        # k-vector +1.5 in X direction
        r1 = (x - (c-3))**2 + (y - c)**2 + (z - c)**2
        psi1 = np.exp(-r1/3.0) * np.exp(1j * 1.5 * x)
        
        # Particle 2 (Right, moving Left)
        # k-vector -1.5 in X direction
        r2 = (x - (c+3))**2 + (y - c)**2 + (z - c)**2
        psi2 = np.exp(-r2/3.0) * np.exp(-1j * 1.5 * x)
        
        # Superposition
        self.psi = (psi1 + psi2).flatten()
        self.psi /= np.linalg.norm(self.psi)

    def build_hamiltonian(self):
            # Construct sparse matrix from Complex Links
            row_idx = []
            col_idx = []
            vals = []
            
            shifts = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] # X, Y, Z neighbors
            
            for dim, shift in enumerate(shifts):
                # Calculate neighbor indices with Periodic Boundary
                stride = [1, self.L, self.L**2][dim]
                
                link_vals = self.links[:, dim]
                
                # FIX: Convert .tocoo() immediately to get .row and .col attributes
                d = sp.diags([-link_vals], [stride], shape=(self.N, self.N)).tocoo()
                
                row_idx.extend(d.row)
                col_idx.extend(d.col)
                vals.extend(d.data)
                
                # Add Hermitian conjugate (Lower triangle)
                # Conjugate the values, swap rows/cols
                row_idx.extend(d.col)
                col_idx.extend(d.row)
                vals.extend(np.conj(d.data))

            # Fixed Site Energy (Vacuum Level)
            H_diag = sp.diags(np.full(self.N, 6.0), 0)
            
            H_off = sp.coo_matrix((vals, (row_idx, col_idx)), shape=(self.N, self.N))
            return H_diag + H_off
        
    def update_geometry(self):
        # 1. Get Correlations: psi_i * psi_j.conj()
        # We only need to compute this for the 3 forward directions
        
        psi_flat = self.psi
        
        # Update each dimension
        strides = [1, self.L, self.L**2]
        
        for dim, stride in enumerate(strides):
            # psi_neighbor is psi shifted by -stride (so index i holds psi_{i+stride})
            psi_neighbor = np.roll(psi_flat, -stride)
            
            # The correlation vector C_ij = psi_i * psi_j^*
            # Wait, kinetic term is c^dag_i c_j. We want the link to match the flow.
            # Flow is usually Im(psi* grad psi).
            # Let's try matching the unitary rotation:
            # Target = psi_i * psi_j.conj()
            
            correlation = psi_flat * np.conj(psi_neighbor)
            
            if self.mode == 'SCALAR':
                # Old Physics: Magnitude Only
                # We ignore the phase of the correlation.
                feedback = np.abs(correlation)
                
                # Apply to Magnitude of link, keep Phase 0
                # current_mag = np.abs(self.links[:, dim])
                # update = feedback / (self.mu * current_mag)
                # self.links[:, dim] += update (This is complex, keep it simple)
                
                # Simplified Scalar Update:
                # Grow real part based on density overlap
                self.links[:, dim] += 2.0 * feedback # Purely real growth
                
            elif self.mode == 'VECTOR':
                # New Physics: Full Complex Feedback
                # The link tries to become the correlation vector
                
                # "Hebbian Learning for Gauge Fields"
                # L_new = L_old + eta * (psi_i * psi_j^*)
                self.links[:, dim] += 2.0 * correlation
                
        # Decay
        # Everything decays back to 1.0 (Real)
        self.links = 1.0 + (self.links - 1.0) * (1.0 - self.lam)

    def run(self, steps=80):
        print(f"--- Running {self.mode} Physics ---")
        
        psi_t = []
        
        for t in range(steps):
            H = self.build_hamiltonian()
            
            # Evolve Wavefunction
            self.psi = expm_multiply(-1j * H * self.dt, self.psi)
            
            # Evolve Geometry
            self.update_geometry()
            
            # Measure: Peak separation
            # Reshape to 3D
            grid = np.abs(self.psi.reshape((self.L, self.L, self.L)))**2
            # Project to X axis
            x_profile = np.sum(grid, axis=(1, 2))
            
            # Find the two peaks
            # Simple hack: Split array in half, find max in left and right
            c = self.L // 2
            max_L = np.argmax(x_profile[:c])
            max_R = c + np.argmax(x_profile[c:])
            
            dist = max_R - max_L
            self.history['dist'].append(dist)
            
            # Log center intensity (Merger metric)
            center_int = x_profile[c]
            
        return self.history['dist']

def compare_modes():
    # 1. Run Scalar (Glue Universe)
    sim_scalar = ComplexSubstrate(L=35, mode='SCALAR')
    dist_scalar = sim_scalar.run()
    
    # 2. Run Vector (Electromagnetic Universe)
    sim_vector = ComplexSubstrate(L=35, mode='VECTOR')
    dist_vector = sim_vector.run()
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dist_scalar, 'r--', linewidth=2, label='Scalar Feedback (Density)')
    plt.plot(dist_vector, 'b-', linewidth=3, label='Vector Feedback (Phase)')
    
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Particle Separation (Lattice Sites)')
    plt.title('Experiment 17: Emergence of Repulsion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    compare_modes()