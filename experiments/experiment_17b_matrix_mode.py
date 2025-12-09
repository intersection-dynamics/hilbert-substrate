import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

class MatrixSubstrate:
    def __init__(self, L=11, internal_dim=2):
        self.L = L
        self.internal_dim = internal_dim
        self.sites = L**3
        self.N_tot = self.sites * internal_dim
        
        # --- The Matrix Graph ---
        # Links are now (N_sites, 3_directions, dim, dim)
        # We initialize with Identity Matrix (Vacuum passes all flavors equally)
        self.links = np.zeros((self.sites, 3, internal_dim, internal_dim), dtype=np.complex128)
        for i in range(internal_dim):
            self.links[:, :, i, i] = 1.0
            
        # Physics Constants
        self.lam = 0.05      # Decay
        self.dt = 0.1
        
        # Wavefunction: Shape (Sites, Internal_Dim)
        self.psi = np.zeros((self.sites, internal_dim), dtype=np.complex128)
        self.init_collision()
        
        self.history = {'overlap': []}

    def init_collision(self):
        c = self.L // 2
        x, y, z = np.indices((self.L, self.L, self.L))
        
        # Particle 1: "Spin Up" [1, 0], moving Right (+k)
        r1 = (x - (c-3))**2 + (y - c)**2 + (z - c)**2
        psi1_spatial = np.exp(-r1/3.0) * np.exp(1j * 1.5 * x)
        # Assign to channel 0
        self.psi[:, 0] += psi1_spatial.flatten()
        
        # Particle 2: "Spin Down" [0, 1], moving Left (-k)
        r2 = (x - (c+3))**2 + (y - c)**2 + (z - c)**2
        psi2_spatial = np.exp(-r2/3.0) * np.exp(-1j * 1.5 * x)
        # Assign to channel 1
        self.psi[:, 1] += psi2_spatial.flatten()
        
        # Normalize total state
        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        self.psi /= norm

    def build_hamiltonian(self):
        # We need to build a block matrix.
        # This is computationally heavier, so we define it carefully.
        
        row_idx = []
        col_idx = []
        vals = []
        
        # Stride in Site Index space
        site_strides = [1, self.L, self.L**2]
        
        # Pre-compute flattened site indices
        sites = np.arange(self.sites)
        
        for d, stride in enumerate(site_strides):
            # Target neighbor sites
            # (ignoring boundary wrapping for speed in this collision test)
            valid_mask = sites + stride < self.sites 
            # In a real grid, we'd mask X/Y/Z edges specifically, 
            # but for a center collision L=11, this linear clip is safe enough.
            
            src_sites = sites[valid_mask]
            dst_sites = sites[valid_mask] + stride
            
            # For each link, we have a (dim, dim) matrix
            link_matrices = self.links[src_sites, d] # Shape (Num_Links, 2, 2)
            
            # Add blocks to sparse matrix
            # H connects (src_site * dim + alpha) to (dst_site * dim + beta)
            
            for alpha in range(self.internal_dim):
                for beta in range(self.internal_dim):
                    # Elements U_{alpha, beta}
                    U_elems = link_matrices[:, alpha, beta]
                    
                    # Source Row (Global Index)
                    rows = src_sites * self.internal_dim + alpha
                    # Dest Col (Global Index)
                    cols = dst_sites * self.internal_dim + beta
                    
                    # Add -0.5 * U as hopping
                    row_idx.extend(rows)
                    col_idx.extend(cols)
                    vals.extend(-0.5 * U_elems)
                    
                    # Hermitian Conjugate (H_ji = U_dag)
                    # U_dag_{beta, alpha} = conj(U_{alpha, beta})
                    # So we map cols->rows, rows->cols, take conj
                    row_idx.extend(cols)
                    col_idx.extend(rows)
                    vals.extend(-0.5 * np.conj(U_elems))

        # Diagonal (Vacuum Energy)
        # All internal states have same vacuum energy
        H_diag = sp.diags(np.full(self.N_tot, 6.0), 0)
        
        H_off = sp.coo_matrix((vals, (row_idx, col_idx)), shape=(self.N_tot, self.N_tot))
        return H_diag + H_off

    def update_geometry(self):
        # MATRIX AGENCY:
        # The link U_{ij} updates to match the OUTER PRODUCT of the spinors
        # U_{ij} += eta * (psi_i * psi_j^dag)
        # This means the link "learns" the rotation required to turn state i into state j
        
        psi_flat = self.psi # Shape (Sites, 2)
        site_strides = [1, self.L, self.L**2]
        
        for d, stride in enumerate(site_strides):
            # Neighbors
            psi_src = psi_flat # psi_i
            psi_dst = np.roll(psi_flat, -stride, axis=0) # psi_j
            
            # Outer Product: psi_i (x) psi_j^*
            # Result is (Sites, 2, 2)
            # Using einsum for batch outer product
            # "sa, sb -> sab" where s=site, a/b=internal_dim
            correlation = np.einsum('sa,sb->sab', psi_src, np.conj(psi_dst))
            
            # Matrix Feedback
            # Note: We enforce a stronger learning rate for matrix features
            self.links[:, d] += 2.0 * correlation
            
        # Decay
        # Decays back to Identity Matrix (Vacuum), not Zero
        # Because vacuum allows free passage.
        identity = np.eye(self.internal_dim)[None, :, :]
        self.links = identity + (self.links - identity) * (1.0 - self.lam)

    def run(self, steps=60):
        print(f"--- Running MATRIX Mode (Non-Abelian Geometry) ---")
        
        overlaps = []
        
        for t in range(steps):
            H = self.build_hamiltonian()
            
            # Flatten psi for sparse matvec
            psi_vec = self.psi.flatten()
            psi_new = expm_multiply(-1j * H * self.dt, psi_vec)
            self.psi = psi_new.reshape((self.sites, self.internal_dim))
            
            self.update_geometry()
            
            # Measure: How much do they overlap?
            # Project spatial densities
            rho_up = np.abs(self.psi[:, 0])**2
            rho_down = np.abs(self.psi[:, 1])**2
            
            # Overlap integral
            overlap = np.sum(rho_up * rho_down)
            overlaps.append(overlap)
            
            if t % 10 == 0:
                print(f"Step {t}: Spatial Overlap = {overlap:.4f}")
            
        return overlaps

def run_experiment():
    sim = MatrixSubstrate(L=11, internal_dim=2)
    data = sim.run(steps=2000)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data, 'g-', linewidth=3, label='Spin Up / Spin Down Overlap')
    plt.title('Experiment 17b: Matrix Geometry (Ghost States)')
    plt.xlabel('Time')
    plt.ylabel('Spatial Overlap')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_experiment()