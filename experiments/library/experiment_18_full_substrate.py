"""
EXPERIMENT 18: THE FULL SUBSTRATE
=================================
The complete information-theoretic model:
- Matrix geometry (SU(2) links that learn from spinor correlations)
- Memory commit (derived threshold ε = √(2/N))
- Orbital spectroscopy and Rabi oscillations
- Coupling constant measurement

This is the test: does non-abelian feedback give α << 1?
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
import matplotlib.pyplot as plt

class FullSubstrate:
    def __init__(self, L=15):
        self.L = L
        self.sites = L**3
        self.internal_dim = 2  # Spinor
        self.N_tot = self.sites * self.internal_dim
        
        # --- Matrix Geometry ---
        # Links are SU(2) matrices: (sites, 3 directions, 2, 2)
        self.links = np.zeros((self.sites, 3, 2, 2), dtype=np.complex128)
        # Initialize to identity (vacuum)
        for i in range(2):
            self.links[:, :, i, i] = 1.0
        
        # --- Scalar Geometry (for comparison) ---
        self.g_map = np.ones((self.sites,), dtype=np.float64)
        
        # --- Physics Constants ---
        self.mu = 1.0           # Work cost
        self.lam_scalar = 0.005 # Scalar decay
        self.lam_matrix = 0.02  # Matrix decay (faster for stability)
        self.eta_scalar = 1.0   # Scalar learning rate
        self.eta_matrix = 0.5   # Matrix learning rate
        
        # Derived memory threshold
        self.epsilon = np.sqrt(2.0 / self.sites)
        
        # Wavefunction: (sites, 2) spinor
        self.psi = np.zeros((self.sites, 2), dtype=np.complex128)
        
        # History
        self.history = {
            'time': [], 'max_g': [], 'energy': [], 
            'entropy': [], 'link_norm': []
        }

    def init_gaussian(self, center=None, width=2.5, spin='up'):
        """Initialize a Gaussian wavepacket"""
        if center is None:
            center = self.L // 2
        
        x, y, z = np.indices((self.L, self.L, self.L))
        r2 = (x - center)**2 + (y - center)**2 + (z - center)**2
        spatial = np.exp(-r2 / (2 * width**2)).flatten()
        
        self.psi[:, :] = 0
        if spin == 'up':
            self.psi[:, 0] = spatial
        elif spin == 'down':
            self.psi[:, 1] = spatial
        else:  # superposition
            self.psi[:, 0] = spatial / np.sqrt(2)
            self.psi[:, 1] = spatial / np.sqrt(2)
        
        self.psi /= np.linalg.norm(self.psi)

    def build_hamiltonian(self, use_matrix=True):
        """
        Build Hamiltonian with both scalar and matrix geometry.
        
        H = Σ ε₀ n_i - Σ t(g_ij) [c†_i U_ij c_j + h.c.]
        
        - g_ij: scalar amplitude (controls hopping strength)
        - U_ij: matrix (controls spin rotation during hop)
        """
        row_idx = []
        col_idx = []
        vals = []
        
        site_strides = [1, self.L, self.L**2]
        sites = np.arange(self.sites)
        
        for d, stride in enumerate(site_strides):
            # Forward neighbors
            mask_fwd = (sites % (self.L if d == 0 else (self.L**2 if d == 1 else self.L**3))) < \
                       (self.L - 1 if d == 0 else (self.L**2 - self.L if d == 1 else self.L**3 - self.L**2))
            
            # Simplified: just avoid last layer in each direction
            if d == 0:
                mask_fwd = (sites % self.L) < (self.L - 1)
            elif d == 1:
                mask_fwd = ((sites // self.L) % self.L) < (self.L - 1)
            else:
                mask_fwd = (sites // (self.L**2)) < (self.L - 1)
            
            src = sites[mask_fwd]
            dst = src + stride
            
            # Scalar hopping amplitude
            g_src = self.g_map[src]
            g_dst = self.g_map[dst]
            t_hop = 0.5 * (g_src + g_dst)  # Average connectivity
            
            if use_matrix:
                # Matrix hopping: t * U
                U_matrices = self.links[src, d]  # (n_links, 2, 2)
                
                for alpha in range(2):
                    for beta in range(2):
                        U_elem = U_matrices[:, alpha, beta]
                        
                        rows = src * 2 + alpha
                        cols = dst * 2 + beta
                        
                        # Forward hop: -t * U
                        row_idx.extend(rows)
                        col_idx.extend(cols)
                        vals.extend(-t_hop * U_elem)
                        
                        # Backward hop: -t * U†
                        row_idx.extend(cols)
                        col_idx.extend(rows)
                        vals.extend(-t_hop * np.conj(U_elem))
            else:
                # Scalar only: no spin rotation
                for alpha in range(2):
                    rows = src * 2 + alpha
                    cols = dst * 2 + alpha
                    
                    row_idx.extend(rows)
                    col_idx.extend(cols)
                    vals.extend(-t_hop)
                    
                    row_idx.extend(cols)
                    col_idx.extend(rows)
                    vals.extend(-t_hop)
        
        # Diagonal: fixed site energy
        H_diag = sp.diags(np.full(self.N_tot, 6.0), 0)
        H_off = sp.coo_matrix((vals, (row_idx, col_idx)), shape=(self.N_tot, self.N_tot))
        
        return (H_diag + H_off).tocsr()

    def update_geometry(self, use_matrix=True):
        """Update both scalar and matrix geometry from wavefunction"""
        
        psi_flat = self.psi  # (sites, 2)
        amplitudes = np.sqrt(np.sum(np.abs(psi_flat)**2, axis=1))  # spatial density
        
        site_strides = [1, self.L, self.L**2]
        
        for d, stride in enumerate(site_strides):
            psi_src = psi_flat
            psi_dst = np.roll(psi_flat, -stride, axis=0)
            
            amp_src = amplitudes
            amp_dst = np.roll(amplitudes, -stride)
            
            # --- Scalar Feedback (density) ---
            density_product = amp_src * amp_dst
            mask = density_product > self.epsilon
            
            if np.any(mask):
                payment = density_product[mask]**2
                current_g = self.g_map[mask]
                delta_g = self.eta_scalar * payment / (self.mu * current_g)
                np.add.at(self.g_map, np.where(mask)[0], delta_g)
            
            if use_matrix:
                # --- Matrix Feedback (spinor correlation) ---
                # U_ij += η * |ψ_i⟩⟨ψ_j|
                correlation = np.einsum('sa,sb->sab', psi_src, np.conj(psi_dst))
                self.links[:, d] += self.eta_matrix * correlation
        
        # --- Scalar Decay (toward vacuum g=1) ---
        self.g_map = 1.0 + (self.g_map - 1.0) * (1.0 - self.lam_scalar)
        
        if use_matrix:
            # --- Matrix Decay (toward identity) ---
            identity = np.eye(2)[None, None, :, :]
            self.links = identity + (self.links - identity) * (1.0 - self.lam_matrix)

    def memory_commit(self):
        """Prune amplitudes below vacuum noise floor"""
        amplitudes = np.sqrt(np.sum(np.abs(self.psi)**2, axis=1))
        mask = amplitudes > self.epsilon
        
        if np.any(~mask):
            self.psi[~mask, :] = 0.0
            norm = np.linalg.norm(self.psi)
            if norm > 1e-12:
                self.psi /= norm

    def get_entropy(self):
        """Spatial Shannon entropy"""
        prob = np.sum(np.abs(self.psi)**2, axis=1)
        prob = prob / (np.sum(prob) + 1e-15)
        prob_safe = prob[prob > 1e-15]
        return -np.sum(prob_safe * np.log(prob_safe))

    def run_stabilization(self, steps=200, use_matrix=True, verbose=True):
        """Let the system find its stable geometry"""
        if verbose:
            print(f"--- Stabilization ({steps} steps, matrix={use_matrix}) ---")
        
        for t in range(steps):
            H = self.build_hamiltonian(use_matrix=use_matrix)
            
            psi_vec = self.psi.flatten()
            psi_new = expm_multiply(-1j * H * 0.1, psi_vec)
            self.psi = psi_new.reshape((self.sites, 2))
            
            # Renormalize
            self.psi /= np.linalg.norm(self.psi)
            
            # Geometry feedback
            self.update_geometry(use_matrix=use_matrix)
            
            # Memory commit
            self.memory_commit()
            
            # Log
            if verbose and t % 50 == 0:
                max_g = np.max(self.g_map)
                entropy = self.get_entropy()
                link_norm = np.mean(np.abs(self.links))
                print(f"  Step {t:3d} | g_max={max_g:.3f} | S={entropy:.3f} | |U|={link_norm:.3f}")
        
        return np.max(self.g_map)

    def solve_orbitals(self, num_modes=9, use_matrix=True):
        """Diagonalize Hamiltonian to get orbital spectrum"""
        print(f"--- Solving Orbitals (matrix={use_matrix}) ---")
        
        H = self.build_hamiltonian(use_matrix=use_matrix)
        vals, vecs = eigsh(H, k=num_modes, which='SA')
        
        # Sort by energy
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        print("Orbital Energies:")
        for i, E in enumerate(vals):
            print(f"  State {i}: E = {E:.4f}")
        
        return vals, vecs

    def run_rabi(self, vals, vecs, field_strength=0.2, steps=500):
        """
        Simulate Rabi oscillation between ground and first excited state.
        Returns Rabi frequency.
        """
        print(f"--- Rabi Oscillation (E_field={field_strength}) ---")
        
        E0, E1 = vals[0], vals[1]
        psi0, psi1 = vecs[:, 0], vecs[:, 1]
        
        omega_res = E1 - E0
        print(f"  Transition Energy: {omega_res:.4f}")
        
        # Dipole matrix element (x-direction)
        x_coords = np.arange(self.L)
        x_grid = np.tile(x_coords, self.L**2)
        x_op = np.repeat(x_grid, 2)  # Expand for spinor
        
        dipole = np.abs(np.vdot(psi1, x_op * psi0))
        print(f"  Dipole Element: {dipole:.4f}")
        
        # Initialize in ground state
        psi_vec = psi0.copy()
        
        H_base = self.build_hamiltonian(use_matrix=True)
        
        # Dipole operator as sparse
        H_dipole = sp.diags(field_strength * x_op, 0)
        
        excited_pop = []
        
        for t in range(steps):
            # Oscillating field at resonance
            phase = np.cos(omega_res * t * 0.1)
            H_total = H_base + phase * H_dipole
            
            psi_vec = expm_multiply(-1j * H_total * 0.1, psi_vec)
            psi_vec /= np.linalg.norm(psi_vec)
            
            # Population in excited state
            pop = np.abs(np.vdot(psi1, psi_vec))**2
            excited_pop.append(pop)
        
        # Extract Rabi frequency from oscillation
        excited_pop = np.array(excited_pop)
        
        # Find first maximum
        for i in range(1, len(excited_pop)-1):
            if excited_pop[i] > excited_pop[i-1] and excited_pop[i] > excited_pop[i+1]:
                if excited_pop[i] > 0.5:
                    T_rabi = 2 * i * 0.1  # Half period
                    omega_rabi = np.pi / T_rabi
                    print(f"  Rabi Frequency: {omega_rabi:.4f}")
                    print(f"  Rabi Period: {T_rabi:.2f}")
                    break
        else:
            omega_rabi = field_strength * dipole  # Estimate
            print(f"  Rabi Frequency (estimated): {omega_rabi:.4f}")
        
        return omega_res, dipole, omega_rabi, excited_pop

    def calculate_alpha(self, E_bound, E_bandwidth=12.0):
        """
        Calculate effective coupling constant.
        
        α = sqrt(2 * E_bound / E_bandwidth)
        
        For hydrogen: E_bound = 13.6 eV, E_bandwidth ~ mc² = 511 keV
        Gives α ~ 1/137
        
        For scalar substrate: E_bound ~ 6, E_bandwidth ~ 12
        Gave α ~ 0.84
        """
        alpha = np.sqrt(2 * np.abs(E_bound) / E_bandwidth)
        return alpha


def run_full_experiment():
    """
    The complete measurement:
    1. Stabilize matter in matrix geometry
    2. Solve orbital spectrum
    3. Run Rabi oscillation
    4. Calculate coupling constant
    5. Compare scalar vs matrix
    """
    
    print("=" * 60)
    print("EXPERIMENT 18: FULL SUBSTRATE - COUPLING CONSTANT MEASUREMENT")
    print("=" * 60)
    
    results = {}
    
    # ===== SCALAR MODE =====
    print("\n" + "="*60)
    print("PART 1: SCALAR GEOMETRY (density feedback)")
    print("="*60)
    
    sim_scalar = FullSubstrate(L=13)
    sim_scalar.init_gaussian(width=2.5, spin='up')
    
    sim_scalar.run_stabilization(steps=200, use_matrix=False)
    
    vals_s, vecs_s = sim_scalar.solve_orbitals(num_modes=5, use_matrix=False)
    
    E_ground_scalar = vals_s[0]
    E_bandwidth = 12.0  # Approximate bandwidth of tight-binding
    
    alpha_scalar = sim_scalar.calculate_alpha(E_ground_scalar, E_bandwidth)
    
    print(f"\n--- SCALAR RESULTS ---")
    print(f"  Ground State Energy: {E_ground_scalar:.4f}")
    print(f"  Bandwidth: {E_bandwidth:.4f}")
    print(f"  α (scalar) = {alpha_scalar:.4f}")
    
    results['scalar'] = {
        'E_ground': E_ground_scalar,
        'alpha': alpha_scalar,
        'energies': vals_s
    }
    
    # ===== MATRIX MODE =====
    print("\n" + "="*60)
    print("PART 2: MATRIX GEOMETRY (spinor correlation feedback)")
    print("="*60)
    
    sim_matrix = FullSubstrate(L=13)
    sim_matrix.init_gaussian(width=2.5, spin='up')
    
    sim_matrix.run_stabilization(steps=200, use_matrix=True)
    
    vals_m, vecs_m = sim_matrix.solve_orbitals(num_modes=5, use_matrix=True)
    
    E_ground_matrix = vals_m[0]
    
    alpha_matrix = sim_matrix.calculate_alpha(E_ground_matrix, E_bandwidth)
    
    print(f"\n--- MATRIX RESULTS ---")
    print(f"  Ground State Energy: {E_ground_matrix:.4f}")
    print(f"  Bandwidth: {E_bandwidth:.4f}")
    print(f"  α (matrix) = {alpha_matrix:.4f}")
    
    results['matrix'] = {
        'E_ground': E_ground_matrix,
        'alpha': alpha_matrix,
        'energies': vals_m
    }
    
    # ===== RABI OSCILLATION (Matrix Mode) =====
    print("\n" + "="*60)
    print("PART 3: RABI OSCILLATION")
    print("="*60)
    
    omega_res, dipole, omega_rabi, rabi_data = sim_matrix.run_rabi(
        vals_m, vecs_m, field_strength=0.2, steps=500
    )
    
    # Oscillator strength
    f_osc = (2.0 / 3.0) * omega_res * dipole**2
    print(f"  Oscillator Strength: {f_osc:.4f}")
    
    results['rabi'] = {
        'omega_res': omega_res,
        'dipole': dipole,
        'omega_rabi': omega_rabi,
        'f_osc': f_osc
    }
    
    # ===== COMPARISON =====
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"  α (scalar feedback):  {alpha_scalar:.4f}  [Expected: ~0.84, nuclear regime]")
    print(f"  α (matrix feedback):  {alpha_matrix:.4f}  [Target: ~0.007, EM regime]")
    print(f"  α (real QED):         0.0073  [1/137]")
    print(f"  α (real QCD):         ~1.0    [strong coupling]")
    
    if alpha_matrix < alpha_scalar:
        print(f"\n  RESULT: Matrix geometry WEAKENS coupling!")
        print(f"  Reduction factor: {alpha_scalar/alpha_matrix:.2f}x")
    else:
        print(f"\n  RESULT: Matrix geometry does not weaken coupling.")
    
    # ===== VISUALIZATION =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Orbital spectrum comparison
    ax1 = axes[0, 0]
    x_s = np.arange(len(vals_s))
    x_m = np.arange(len(vals_m)) + 0.3
    ax1.bar(x_s, vals_s, width=0.3, label='Scalar', color='red', alpha=0.7)
    ax1.bar(x_m, vals_m, width=0.3, label='Matrix', color='blue', alpha=0.7)
    ax1.set_xlabel('State')
    ax1.set_ylabel('Energy')
    ax1.set_title('Orbital Spectrum: Scalar vs Matrix')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rabi oscillation
    ax2 = axes[0, 1]
    ax2.plot(np.arange(len(rabi_data)) * 0.1, rabi_data, 'g-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Excited State Population')
    ax2.set_title(f'Rabi Oscillation (ω_res = {omega_res:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Coupling constant comparison
    ax3 = axes[1, 0]
    alphas = [alpha_scalar, alpha_matrix, 1.0, 1/137]
    labels = ['Scalar\n(this work)', 'Matrix\n(this work)', 'QCD\n(strong)', 'QED\n(EM)']
    colors = ['red', 'blue', 'orange', 'green']
    bars = ax3.bar(labels, alphas, color=colors, alpha=0.7)
    ax3.set_ylabel('Coupling Constant α')
    ax3.set_title('Coupling Strength Comparison')
    ax3.set_yscale('log')
    ax3.axhline(y=1/137, color='green', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Geometry visualization
    ax4 = axes[1, 1]
    g_1d = sim_matrix.g_map.reshape(13, 13, 13)[6, 6, :]
    link_strength = np.mean(np.abs(sim_matrix.links), axis=(1, 2, 3)).reshape(13, 13, 13)[6, 6, :]
    ax4.plot(g_1d, 'r-', linewidth=2, label='Scalar g')
    ax4.plot(link_strength, 'b-', linewidth=2, label='Matrix |U|')
    ax4.set_xlabel('Position (1D slice)')
    ax4.set_ylabel('Geometry Strength')
    ax4.set_title('Scalar vs Matrix Geometry Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('18_full_substrate.png', dpi=150)
    print(f"\nFigure saved: 18_full_substrate.png")
    plt.show()
    
    return results


if __name__ == "__main__":
    results = run_full_experiment()