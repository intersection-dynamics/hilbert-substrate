"""
Coincidence Test with SU(3) Gauge Links
========================================

Same question: does the substrate create forbidden regions?

Now with SU(3) instead of SU(2):
- 3-component "color" vectors at each site instead of 2-component spinors
- 3×3 SU(3) link matrices instead of 2×2 SU(2)
- 8 Gell-Mann generators instead of 3 Pauli matrices
- Z₃ center instead of Z₂

Does the richer gauge structure change the coincidence behavior?
"""

import numpy as np
import matplotlib.pyplot as plt

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy import linalg as cpla
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    cpla = None
    GPU_AVAILABLE = False
    print("Warning: CuPy not available, falling back to NumPy (will be slow)")


# =============================================================================
# SU(3) helpers
# =============================================================================

def make_gell_mann_matrices(xp):
    """
    Return the 8 Gell-Mann matrices (generators of SU(3)).
    """
    # λ₁
    l1 = xp.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=xp.complex128)
    
    # λ₂
    l2 = xp.array([
        [0, -1j, 0],
        [1j, 0, 0],
        [0, 0, 0]
    ], dtype=xp.complex128)
    
    # λ₃
    l3 = xp.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=xp.complex128)
    
    # λ₄
    l4 = xp.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ], dtype=xp.complex128)
    
    # λ₅
    l5 = xp.array([
        [0, 0, -1j],
        [0, 0, 0],
        [1j, 0, 0]
    ], dtype=xp.complex128)
    
    # λ₆
    l6 = xp.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ], dtype=xp.complex128)
    
    # λ₇
    l7 = xp.array([
        [0, 0, 0],
        [0, 0, -1j],
        [0, 1j, 0]
    ], dtype=xp.complex128)
    
    # λ₈
    l8 = xp.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -2]
    ], dtype=xp.complex128) / np.sqrt(3)
    
    return [l1, l2, l3, l4, l5, l6, l7, l8]


# Initialize based on GPU availability
if GPU_AVAILABLE:
    xp = cp
    GELL_MANN = make_gell_mann_matrices(cp)
    I3 = cp.eye(3, dtype=cp.complex128)
else:
    xp = np
    GELL_MANN = make_gell_mann_matrices(np)
    I3 = np.eye(3, dtype=np.complex128)


def project_to_su3(U):
    """
    Project a 3×3 complex matrix onto SU(3) using polar decomposition.
    
    Steps:
        1. H = U† U
        2. H^{-1/2} via eigen-decomposition
        3. U_unitary = U H^{-1/2}
        4. Rescale to det(U_unitary) = 1 (cube root for SU(3))
    """
    _xp = cp if GPU_AVAILABLE else np
    
    H = U.conj().T @ U
    evals, evecs = _xp.linalg.eigh(H)
    evals = _xp.maximum(evals, 1e-12)
    inv_sqrt = evecs @ _xp.diag(1.0 / _xp.sqrt(evals)) @ evecs.conj().T
    Uu = U @ inv_sqrt
    
    # For SU(3), we need det = 1, so divide by cube root of det
    det = _xp.linalg.det(Uu)
    det = _xp.where(_xp.abs(det) < 1e-12, 1.0 + 0j, det)
    # Cube root: det^(1/3)
    phase = _xp.angle(det) / 3.0
    mag = _xp.abs(det) ** (1.0/3.0)
    det_cbrt = mag * _xp.exp(1j * phase)
    Uu = Uu / det_cbrt
    
    return Uu


def traceless_projection_su3(M):
    """
    Project a 3×3 matrix onto the traceless part (su(3) Lie algebra).
    """
    _xp = cp if GPU_AVAILABLE else np
    tr = _xp.trace(M) / 3.0
    return M - tr * I3


# =============================================================================
# SU(3) Substrate with two-lump initialization
# =============================================================================

class TwoLumpSubstrateSU3:
    """
    3D substrate with SU(3) gauge links and 3-component color vectors.
    
    Initialized with two separated Gaussian lumps to test coincidence.
    """
    
    def __init__(
        self,
        L: int,
        lump1_center: tuple,
        lump2_center: tuple,
        lump_width: float = 1.0,
        eta_scalar: float = 0.6,
        decay_scalar: float = 0.05,
        eta_matrix: float = 1.0,
        decay_matrix: float = 0.1,
        E_link: float = 1.0,
        plaquette_coupling: float = 0.2,
        gauge_noise: float = 0.001,
    ):
        self.xp = cp if GPU_AVAILABLE else np
        
        self.L = int(L)
        self.N = self.L ** 3
        
        self.lump1_center = lump1_center
        self.lump2_center = lump2_center
        self.lump_width = lump_width
        
        self.eta_scalar = float(eta_scalar)
        self.decay_scalar = float(decay_scalar)
        self.eta_matrix = float(eta_matrix)
        self.decay_matrix = float(decay_matrix)
        self.E_link = float(E_link)
        self.kappa = float(plaquette_coupling)
        self.noise = float(gauge_noise)
        
        # Build links
        self.links = []
        self.t_links = {}
        self.U_links = {}  # Now 3×3 SU(3) matrices
        self._build_links()
        
        # Initialize with two lumps (3-component)
        self.psi = self._init_two_lumps()
    
    def _idx(self, x, y, z):
        return x * self.L * self.L + y * self.L + z
    
    def _coord(self, idx):
        x = idx // (self.L * self.L)
        y = (idx // self.L) % self.L
        z = idx % self.L
        return x, y, z
    
    def _move(self, idx, dx, dy, dz):
        x, y, z = self._coord(idx)
        nx = (x + dx) % self.L
        ny = (y + dy) % self.L
        nz = (z + dz) % self.L
        return self._idx(nx, ny, nz)
    
    def _build_links(self):
        """Build nearest-neighbor links with initial t=1, U=I₃."""
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    i = self._idx(x, y, z)
                    for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
                        j = self._move(i, dx, dy, dz)
                        if i < j:
                            self.links.append((i, j))
                            self.t_links[(i, j)] = 1.0
                            self.U_links[(i, j)] = I3.copy()
    
    def _init_two_lumps(self):
        """
        Initialize ψ as sum of two Gaussian lumps.
        Now 3-component: first color component only (like "red").
        """
        xp = self.xp
        psi = np.zeros((self.N, 3), dtype=np.complex128)
        
        c1 = np.array(self.lump1_center)
        c2 = np.array(self.lump2_center)
        sigma = self.lump_width
        
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    idx = self._idx(x, y, z)
                    pos = np.array([x, y, z])
                    
                    # Distance to each center (with periodic wrapping)
                    d1 = pos - c1
                    d2 = pos - c2
                    d1 = d1 - self.L * np.round(d1 / self.L)
                    d2 = d2 - self.L * np.round(d2 / self.L)
                    
                    r1_sq = np.sum(d1**2)
                    r2_sq = np.sum(d2**2)
                    
                    amp1 = np.exp(-r1_sq / (2 * sigma**2))
                    amp2 = np.exp(-r2_sq / (2 * sigma**2))
                    
                    # Both lumps in first color component
                    psi[idx, 0] = amp1 + amp2
                    psi[idx, 1] = 0.0
                    psi[idx, 2] = 0.0
        
        # Normalize
        flat = psi.reshape(-1)
        flat /= np.linalg.norm(flat)
        
        if GPU_AVAILABLE:
            return cp.asarray(flat)
        return flat
    
    def _get_U(self, i, j):
        if i < j:
            return self.U_links[(i, j)]
        elif j < i:
            return self.U_links[(j, i)].conj().T
        else:
            return I3
    
    def _get_t(self, i, j):
        if i < j:
            return self.t_links[(i, j)]
        elif j < i:
            return self.t_links[(j, i)]
        else:
            return 0.0
    
    def hamiltonian(self):
        """Build single-particle Hamiltonian H (now 3N × 3N)."""
        _xp = self.xp
        size = 3 * self.N  # 3 color components per site
        H = _xp.zeros((size, size), dtype=_xp.complex128)
        for (i, j) in self.links:
            t_ij = self.t_links[(i, j)]
            U_ij = self.U_links[(i, j)]
            # i -> j
            H[3*i:3*i+3, 3*j:3*j+3] -= t_ij * U_ij
            # j -> i (Hermitian conjugate)
            H[3*j:3*j+3, 3*i:3*i+3] -= t_ij * U_ij.conj().T
        return H
    
    def _plaquette_matrix(self, i, j):
        """Return plaquette matrix for link (i,j)."""
        x, y, z = self._coord(i)
        xv, yv, zv = self._coord(j)
        dx = (xv - x) % self.L
        dy = (yv - y) % self.L
        dz = (zv - z) % self.L
        
        if dx == self.L - 1: dx = -1
        if dy == self.L - 1: dy = -1
        if dz == self.L - 1: dz = -1
        
        try:
            if dx != 0:
                a = self._move(i, 0, 1, 0)
                b = self._move(j, 0, 1, 0)
            elif dy != 0:
                a = self._move(i, 0, 0, 1)
                b = self._move(j, 0, 0, 1)
            else:
                a = self._move(i, 1, 0, 0)
                b = self._move(j, 1, 0, 0)
            
            Uia = self._get_U(i, a)
            Uab = self._get_U(a, b)
            Ubj = self._get_U(b, j)
            Uji = self._get_U(j, i)
            return Uia @ Uab @ Ubj @ Uji
        except KeyError:
            return I3
    
    def compute_diagnostics(self):
        """Compute diagnostic quantities."""
        _xp = self.xp
        
        # Reshape psi to (N, 3)
        psi_colors = self.psi.reshape(self.N, 3)
        
        # Site amplitudes (sum over color)
        site_amp_sq = _xp.sum(_xp.abs(psi_colors)**2, axis=1)
        
        # Energy
        H = self.hamiltonian()
        energy = float(_xp.vdot(self.psi, H @ self.psi).real)
        
        # Participation ratio
        ipr = float(_xp.sum(site_amp_sq**2))
        PR = 1.0 / max(ipr, 1e-12)
        
        if GPU_AVAILABLE:
            site_amp_sq_np = site_amp_sq.get()
        else:
            site_amp_sq_np = site_amp_sq
        
        # Center of mass
        total_amp = np.sum(site_amp_sq_np)
        com = np.zeros(3)
        for idx in range(self.N):
            x, y, z = self._coord(idx)
            com += site_amp_sq_np[idx] * np.array([x, y, z])
        com /= max(total_amp, 1e-12)
        
        # Spread
        spread_sq = 0.0
        for idx in range(self.N):
            x, y, z = self._coord(idx)
            pos = np.array([x, y, z])
            d = pos - com
            d = d - self.L * np.round(d / self.L)
            spread_sq += site_amp_sq_np[idx] * np.sum(d**2)
        spread = np.sqrt(spread_sq / max(total_amp, 1e-12))
        
        # Overlap at midpoint
        mid = (np.array(self.lump1_center) + np.array(self.lump2_center)) / 2
        mid = np.round(mid).astype(int) % self.L
        mid_idx = self._idx(*mid)
        overlap_amp = float(site_amp_sq_np[mid_idx])
        
        # Mean link strength
        mean_t = float(np.mean(list(self.t_links.values())))
        
        # Gauge deviation from identity
        gauge_dev = 0.0
        count = 0
        for (i, j), U in self.U_links.items():
            diff = U - I3
            gauge_dev += float(_xp.linalg.norm(diff))
            count += 1
        gauge_dev /= max(count, 1)
        
        # Max site amplitude
        max_amp = float(_xp.max(site_amp_sq))
        
        # Color mixing: how much amplitude is in non-primary colors?
        color_0 = float(_xp.sum(_xp.abs(psi_colors[:, 0])**2))
        color_1 = float(_xp.sum(_xp.abs(psi_colors[:, 1])**2))
        color_2 = float(_xp.sum(_xp.abs(psi_colors[:, 2])**2))
        color_mixing = color_1 + color_2  # Amplitude that "leaked" to other colors
        
        return {
            'energy': energy,
            'participation_ratio': PR,
            'spread': spread,
            'overlap_amp': overlap_amp,
            'mean_t': mean_t,
            'gauge_deviation': gauge_dev,
            'max_amplitude': max_amp,
            'color_mixing': color_mixing,
        }
    
    def evolve_step(self, dt):
        """Single evolution step for psi and geometry."""
        _xp = self.xp
        
        # Evolve psi
        H = self.hamiltonian()
        if GPU_AVAILABLE:
            Uop = cpla.expm(-1j * H * dt)
        else:
            from scipy.linalg import expm
            Uop = expm(-1j * H * dt)
        self.psi = Uop @ self.psi
        
        # Reshape for feedback
        psi_colors = self.psi.reshape(self.N, 3)
        if GPU_AVAILABLE:
            psi_mag = _xp.sqrt(_xp.sum(_xp.abs(psi_colors)**2, axis=1)).get()
        else:
            psi_mag = np.sqrt(np.sum(np.abs(psi_colors)**2, axis=1))
        
        # Scalar feedback
        new_t = {}
        for (i, j) in self.links:
            corr = psi_mag[i] * psi_mag[j]
            t_val = self.t_links[(i, j)]
            dt_val = (self.eta_scalar * corr - self.decay_scalar * (t_val - 1.0)) * dt
            new_t[(i, j)] = t_val + dt_val
        self.t_links = new_t
        
        # Matrix feedback (now SU(3))
        for (i, j) in self.links:
            P_i = psi_colors[i]  # 3-component
            P_j = psi_colors[j]
            corr = _xp.outer(P_i, _xp.conj(P_j))  # 3×3
            corr_su3 = traceless_projection_su3(corr)
            
            U_curr = self.U_links[(i, j)]
            
            if self.kappa != 0.0:
                P = self._plaquette_matrix(i, j)
                delta_plaq = -self.kappa * (P - I3)
            else:
                delta_plaq = 0.0 * U_curr
            
            if self.noise > 0.0:
                # Random su(3) generator
                xi = _xp.random.normal(0.0, 1.0, 8)
                Hnoise = sum(xi[k] * GELL_MANN[k] for k in range(8))
                delta_noise = 1j * self.noise * Hnoise @ U_curr
            else:
                delta_noise = 0.0 * U_curr
            
            delta = (
                self.eta_matrix * corr_su3
                - (self.decay_matrix + self.E_link) * (U_curr - I3)
                + delta_plaq
            ) * dt + delta_noise * np.sqrt(dt)
            
            U_trial = U_curr + delta
            U_new = project_to_su3(U_trial)
            self.U_links[(i, j)] = U_new


# =============================================================================
# Main experiment
# =============================================================================

def run_coincidence_experiment_su3(
    L=6,
    separation=4,
    lump_width=1.0,
    n_steps=200,
    dt=0.05,
    plot=True,
):
    """
    Run the coincidence test with SU(3) gauge structure.
    """
    
    center = L // 2
    lump1 = (center - separation // 2, center, center)
    lump2 = (center + separation // 2, center, center)
    
    print("=" * 60)
    print("Coincidence Test: SU(3) Gauge Structure")
    print("=" * 60)
    print(f"Lattice: {L}×{L}×{L}")
    print(f"Lump 1 center: {lump1}")
    print(f"Lump 2 center: {lump2}")
    print(f"Initial separation: {separation}")
    print(f"Lump width: {lump_width}")
    print(f"Steps: {n_steps}, dt={dt}")
    print(f"Gauge group: SU(3)")
    print("-" * 60)
    
    substrate = TwoLumpSubstrateSU3(
        L=L,
        lump1_center=lump1,
        lump2_center=lump2,
        lump_width=lump_width,
        eta_scalar=0.6,
        decay_scalar=0.05,
        eta_matrix=1.0,
        decay_matrix=0.1,
        E_link=1.0,
        plaquette_coupling=0.2,
        gauge_noise=0.001,
    )
    
    # Storage
    times = [0.0]
    history = {key: [] for key in [
        'energy', 'participation_ratio', 'spread', 'overlap_amp',
        'mean_t', 'gauge_deviation', 'max_amplitude', 'color_mixing'
    ]}
    
    # Initial diagnostics
    diag = substrate.compute_diagnostics()
    for key in history:
        history[key].append(diag[key])
    
    print(f"Initial state:")
    print(f"  Energy: {diag['energy']:.4f}")
    print(f"  PR: {diag['participation_ratio']:.2f}")
    print(f"  Spread: {diag['spread']:.3f}")
    print(f"  Overlap amp: {diag['overlap_amp']:.6f}")
    print(f"  Color mixing: {diag['color_mixing']:.6f}")
    print()
    
    # Evolve
    print("Evolving...")
    for step in range(n_steps):
        substrate.evolve_step(dt)
        times.append((step + 1) * dt)
        
        diag = substrate.compute_diagnostics()
        for key in history:
            history[key].append(diag[key])
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: E={diag['energy']:.4f}, "
                  f"PR={diag['participation_ratio']:.2f}, "
                  f"overlap={diag['overlap_amp']:.6f}, "
                  f"color_mix={diag['color_mixing']:.6f}")
    
    times = np.array(times)
    for key in history:
        history[key] = np.array(history[key])
    
    # Final summary
    print()
    print("-" * 60)
    print("Final state:")
    print(f"  Energy: {history['energy'][-1]:.4f} (initial: {history['energy'][0]:.4f})")
    print(f"  PR: {history['participation_ratio'][-1]:.2f} (initial: {history['participation_ratio'][0]:.2f})")
    print(f"  Spread: {history['spread'][-1]:.3f} (initial: {history['spread'][0]:.3f})")
    print(f"  Overlap amp: {history['overlap_amp'][-1]:.6f} (initial: {history['overlap_amp'][0]:.6f})")
    print(f"  Mean t: {history['mean_t'][-1]:.4f}")
    print(f"  Gauge dev: {history['gauge_deviation'][-1]:.6f}")
    print(f"  Color mixing: {history['color_mixing'][-1]:.6f} (initial: {history['color_mixing'][0]:.6f})")
    print("=" * 60)
    
    # Analysis
    print()
    print("ANALYSIS: Signs of forbidden regions?")
    print("-" * 60)
    
    overlap_change = history['overlap_amp'][-1] - history['overlap_amp'][0]
    energy_change = history['energy'][-1] - history['energy'][0]
    
    if overlap_change > 0 and energy_change > 0:
        print(f"✓ Overlap increased ({overlap_change:+.6f}) AND energy increased ({energy_change:+.4f})")
        print("  -> Possible energy barrier for coincidence")
    elif overlap_change > 0 and energy_change < 0:
        print(f"✗ Overlap increased ({overlap_change:+.6f}) BUT energy decreased ({energy_change:+.4f})")
        print("  -> No energy barrier; coincidence is energetically favorable")
    else:
        print(f"? Overlap changed by {overlap_change:+.6f}, energy changed by {energy_change:+.4f}")
    
    spread_change = history['spread'][-1] - history['spread'][0]
    if spread_change < -0.1:
        print(f"✗ Spread decreased ({spread_change:+.3f}) -> lumps are merging, no repulsion")
    elif spread_change > 0.1:
        print(f"✓ Spread increased ({spread_change:+.3f}) -> possible repulsion keeping lumps apart")
    else:
        print(f"? Spread roughly constant ({spread_change:+.3f})")
    
    # Color mixing analysis
    color_change = history['color_mixing'][-1] - history['color_mixing'][0]
    if color_change > 0.01:
        print(f"! Color mixing increased ({color_change:+.6f}) -> SU(3) gauge dynamics active")
    else:
        print(f"  Color mixing minimal ({color_change:+.6f}) -> staying in original color sector")
    
    print()
    
    if plot:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        axes[0, 0].plot(times, history['energy'])
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Energy')
        axes[0, 0].set_title('Energy vs Time')
        
        axes[0, 1].plot(times, history['participation_ratio'])
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Participation Ratio')
        axes[0, 1].set_title('Delocalization')
        
        axes[0, 2].plot(times, history['spread'])
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Spread')
        axes[0, 2].set_title('Wavefunction Spread')
        
        axes[0, 3].plot(times, history['color_mixing'])
        axes[0, 3].set_xlabel('Time')
        axes[0, 3].set_ylabel('Non-primary color amplitude')
        axes[0, 3].set_title('Color Mixing (SU(3) effect)')
        
        axes[1, 0].plot(times, history['overlap_amp'])
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Amplitude at midpoint')
        axes[1, 0].set_title('Overlap Region Amplitude')
        
        axes[1, 1].plot(times, history['mean_t'])
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Mean link strength')
        axes[1, 1].set_title('Scalar Links')
        
        axes[1, 2].plot(times, history['gauge_deviation'])
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Mean ||U - I||')
        axes[1, 2].set_title('Gauge Field Deviation')
        
        axes[1, 3].plot(times, history['max_amplitude'])
        axes[1, 3].set_xlabel('Time')
        axes[1, 3].set_ylabel('Max site amplitude')
        axes[1, 3].set_title('Peak Concentration')
        
        plt.tight_layout()
        plt.savefig('coincidence_test_su3.png', dpi=150)
        print("Saved: coincidence_test_su3.png")
        plt.show()
    
    return times, history


if __name__ == "__main__":
    run_coincidence_experiment_su3(
        L=6,
        separation=4,
        lump_width=1.0,
        n_steps=200,
        dt=0.05,
        plot=True,
    )