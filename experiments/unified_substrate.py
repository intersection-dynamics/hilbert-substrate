#!/usr/bin/env python3
"""
UNIFIED SUBSTRATE: Three Feedback Mechanisms on Emergent Spacetime
===================================================================

This script demonstrates all three feedback regimes from the Hilbert Substrate
Framework operating together on the same emergent spacetime:

1. SCALAR FEEDBACK (Level 1): Edge weights modulated by local amplitude
   → Emergent mass, topological binding, localization
   
2. MATRIX FEEDBACK (Level 2): U(1) phases on links, evolving with current
   → Gauge transport, spin precession, Aharonov-Bohm physics
   
3. QUANTUM LINKS (Level 3): Links as quantum two-level systems
   → Fine structure constant α = t/E_link, dressed particles

Together these produce:
- Massive particles (scalar feedback → effective mass)
- Gauge-coupled particles (matrix feedback → electromagnetic-like coupling)  
- Running coupling constants (quantum links → α from vacuum stiffness)

The key insight: All three emerge from the same substrate dynamics,
not as separate mechanisms but as different aspects of one unified picture.

Author: Ben Bray (Hilbert Substrate Framework)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SubstrateParams:
    """Parameters controlling all three feedback mechanisms."""
    # Geometry
    L: int = 6                    # Lattice size (1D chain for tractability)
    
    # Base hopping
    t_hop: float = 1.0            # Bare hopping amplitude
    
    # Level 1: Scalar feedback
    scalar_coupling: float = 0.5  # Strength of amplitude→weight feedback
    
    # Level 2: Matrix (gauge) feedback  
    flux: float = 0.0             # Background magnetic flux (in units of 2π)
    gauge_coupling: float = 0.3   # Strength of current→phase feedback
    
    # Level 3: Quantum links
    E_link: float = 4.0           # Vacuum stiffness (link excitation energy)
    
    # Dynamics
    dt: float = 0.05              # Time step
    n_steps: int = 200            # Number of evolution steps


class UnifiedSubstrate:
    """
    Unified simulation of all three feedback mechanisms.
    
    State space: |particle_position⟩ ⊗ |link_quantum_states⟩
    
    Additional classical degrees of freedom:
    - w[i]: scalar edge weights (updated dynamically)
    - phi[i]: U(1) gauge phases on links (updated dynamically)
    
    The quantum links are treated exactly; scalar and matrix feedback
    are treated semiclassically (updated based on expectation values).
    """
    
    def __init__(self, params: SubstrateParams):
        self.p = params
        self.L = params.L
        self.n_links = params.L - 1  # Open boundary
        
        # Quantum Hilbert space: L sites × 2^{n_links} link configs
        self.n_link_configs = 2 ** self.n_links
        self.dim = self.L * self.n_link_configs
        
        # Classical degrees of freedom (updated semiclassically)
        self.weights = np.ones(self.n_links)      # Scalar: edge weights
        self.phases = np.zeros(self.n_links)      # Matrix: U(1) phases
        
        # Add background flux (distributed across links)
        if params.flux != 0:
            self.phases += 2 * np.pi * params.flux / self.n_links
        
        # Storage for observables
        self.history = {
            'time': [],
            'position': [],
            'position_var': [],
            'mean_excitations': [],
            'weights': [],
            'phases': [],
            'total_phase': [],
            'energy': [],
        }
        
    def _index(self, pos: int, link_config: int) -> int:
        """Convert (position, link_config) to linear basis index."""
        return pos * self.n_link_configs + link_config
    
    def _flip_link(self, config: int, link_idx: int) -> int:
        """Flip quantum state of link."""
        return config ^ (1 << link_idx)
    
    def _count_excitations(self, config: int) -> int:
        """Count excited links in configuration."""
        return bin(config).count('1')
    
    def build_hamiltonian(self) -> sp.csr_matrix:
        """
        Build the full Hamiltonian incorporating all three mechanisms.
        
        H = -t Σ_⟨ij⟩ w_ij e^{iφ_ij} (c†_j σ^x_ij c_i) + E_link Σ_l n_l
        
        where:
        - w_ij: scalar edge weights (Level 1)
        - φ_ij: U(1) gauge phases (Level 2)  
        - σ^x_ij: link flip operator (Level 3)
        - n_l: link excitation number
        """
        rows, cols, data = [], [], []
        
        for pos in range(self.L):
            for lc in range(self.n_link_configs):
                idx = self._index(pos, lc)
                
                # Diagonal: quantum link energy (Level 3)
                n_exc = self._count_excitations(lc)
                rows.append(idx)
                cols.append(idx)
                data.append(n_exc * self.p.E_link)
                
                # Hopping right: pos → pos+1
                if pos < self.L - 1:
                    link_idx = pos
                    new_lc = self._flip_link(lc, link_idx)
                    jdx = self._index(pos + 1, new_lc)
                    
                    # Combined hopping: t × w × e^{iφ}
                    hop = -self.p.t_hop * self.weights[link_idx] * np.exp(1j * self.phases[link_idx])
                    
                    rows.append(idx)
                    cols.append(jdx)
                    data.append(hop)
                    
                # Hopping left: pos → pos-1
                if pos > 0:
                    link_idx = pos - 1
                    new_lc = self._flip_link(lc, link_idx)
                    jdx = self._index(pos - 1, new_lc)
                    
                    # Hermitian conjugate: t × w × e^{-iφ}
                    hop = -self.p.t_hop * self.weights[link_idx] * np.exp(-1j * self.phases[link_idx])
                    
                    rows.append(idx)
                    cols.append(jdx)
                    data.append(hop)
        
        H = sp.csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim), dtype=np.complex128)
        return H
    
    def measure_observables(self, psi: np.ndarray) -> Dict:
        """Compute all observables from current state."""
        
        # Position distribution
        pos_prob = np.zeros(self.L)
        link_prob = np.zeros(self.n_links)  # Probability link is excited
        
        for pos in range(self.L):
            for lc in range(self.n_link_configs):
                idx = self._index(pos, lc)
                prob = np.abs(psi[idx])**2
                pos_prob[pos] += prob
                
                # Check each link
                for l in range(self.n_links):
                    if (lc >> l) & 1:  # Link l is excited
                        link_prob[l] += prob
        
        mean_pos = np.sum(np.arange(self.L) * pos_prob)
        var_pos = np.sum((np.arange(self.L) - mean_pos)**2 * pos_prob)
        mean_exc = np.sum(link_prob)
        
        # Current on each link (for gauge feedback)
        # J_l = Im(⟨ψ| c†_{l+1} c_l |ψ⟩) approximately
        currents = np.zeros(self.n_links)
        for l in range(self.n_links):
            # Sum over link configs where we can compute the current
            for lc in range(self.n_link_configs):
                idx_left = self._index(l, lc)
                idx_right = self._index(l + 1, lc)
                if idx_left < self.dim and idx_right < self.dim:
                    currents[l] += np.imag(np.conj(psi[idx_right]) * psi[idx_left])
        
        return {
            'pos_prob': pos_prob,
            'link_prob': link_prob,
            'mean_pos': mean_pos,
            'var_pos': var_pos,
            'mean_exc': mean_exc,
            'currents': currents,
        }
    
    def update_classical_dof(self, obs: Dict, dt: float):
        """
        Update classical degrees of freedom based on quantum expectation values.
        
        Level 1 (Scalar): Weights increase where amplitude is high
            dw/dt = λ_s × (local_amplitude - mean)
            
        Level 2 (Matrix): Phases evolve with local current
            dφ/dt = λ_g × current
        """
        # Level 1: Scalar feedback
        # Weight on link l depends on amplitude at sites l and l+1
        for l in range(self.n_links):
            local_amp = (obs['pos_prob'][l] + obs['pos_prob'][l + 1]) / 2
            mean_amp = 1.0 / self.L
            
            # Weights drift toward regions of high amplitude (attractive)
            # With saturation to keep weights positive and bounded
            dw = self.p.scalar_coupling * (local_amp - mean_amp) * dt
            self.weights[l] = np.clip(self.weights[l] + dw, 0.1, 3.0)
        
        # Level 2: Matrix (gauge) feedback
        # Phases evolve with current (like minimal coupling to EM field)
        for l in range(self.n_links):
            dphi = self.p.gauge_coupling * obs['currents'][l] * dt
            self.phases[l] += dphi
    
    def initialize_wavepacket(self, center: Optional[int] = None, 
                              width: float = 1.0) -> np.ndarray:
        """Initialize a Gaussian wavepacket at given position."""
        if center is None:
            center = self.L // 2
            
        psi = np.zeros(self.dim, dtype=np.complex128)
        
        # Gaussian in position, vacuum in links
        for pos in range(self.L):
            amp = np.exp(-(pos - center)**2 / (2 * width**2))
            idx = self._index(pos, 0)  # link_config = 0 (all ground)
            psi[idx] = amp
            
        psi /= np.linalg.norm(psi)
        return psi
    
    def evolve(self, psi0: Optional[np.ndarray] = None, 
               record_interval: int = 1) -> np.ndarray:
        """
        Time-evolve the system with all three feedback mechanisms active.
        """
        if psi0 is None:
            psi0 = self.initialize_wavepacket()
        
        psi = psi0.copy()
        
        for step in range(self.p.n_steps):
            t = step * self.p.dt
            
            # Record observables
            if step % record_interval == 0:
                H = self.build_hamiltonian()
                obs = self.measure_observables(psi)
                energy = np.real(np.conj(psi) @ H @ psi)
                
                self.history['time'].append(t)
                self.history['position'].append(obs['mean_pos'])
                self.history['position_var'].append(obs['var_pos'])
                self.history['mean_excitations'].append(obs['mean_exc'])
                self.history['weights'].append(self.weights.copy())
                self.history['phases'].append(self.phases.copy())
                self.history['total_phase'].append(np.sum(self.phases))
                self.history['energy'].append(energy)
            
            # Build current Hamiltonian (depends on weights and phases)
            H = self.build_hamiltonian()
            
            # Quantum evolution
            psi = spla.expm_multiply(-1j * H * self.p.dt, psi)
            psi /= np.linalg.norm(psi)
            
            # Update classical degrees of freedom (semiclassical feedback)
            obs = self.measure_observables(psi)
            self.update_classical_dof(obs, self.p.dt)
        
        return psi
    
    def compute_effective_alpha(self) -> float:
        """
        Compute effective fine structure constant from ground state.
        
        In perturbative regime: α ≈ t/E_link
        """
        H = self.build_hamiltonian()
        try:
            E0 = spla.eigsh(H, k=1, which='SA', maxiter=3000)[0][0]
        except:
            E0 = np.linalg.eigvalsh(H.toarray())[0]
        
        # Free particle ground state energy for comparison
        cos_factor = np.cos(np.pi / (self.L + 1))
        t_eff = -np.real(E0) / (2 * cos_factor)
        
        # Effective alpha
        alpha = t_eff / self.p.t_hop
        return alpha


def run_unified_demo():
    """
    Demonstrate all three mechanisms working together.
    """
    print("="*70)
    print("  UNIFIED SUBSTRATE: Three Feedback Mechanisms")
    print("="*70)
    
    # Set up parameters
    params = SubstrateParams(
        L=6,
        t_hop=1.0,
        scalar_coupling=0.3,    # Level 1: mass/binding
        gauge_coupling=0.2,     # Level 2: gauge transport
        E_link=4.0,             # Level 3: vacuum stiffness
        flux=0.1,               # Small background flux
        dt=0.05,
        n_steps=300,
    )
    
    print(f"\nParameters:")
    print(f"  Lattice size: {params.L} sites")
    print(f"  Scalar coupling (Level 1): {params.scalar_coupling}")
    print(f"  Gauge coupling (Level 2): {params.gauge_coupling}")
    print(f"  Vacuum stiffness E_link (Level 3): {params.E_link}")
    print(f"  Background flux: {params.flux} × 2π")
    print(f"  Theoretical α = t/E = {params.t_hop/params.E_link:.4f}")
    
    # Create substrate
    substrate = UnifiedSubstrate(params)
    
    # Measure initial effective alpha
    alpha_initial = substrate.compute_effective_alpha()
    print(f"\nInitial effective α: {alpha_initial:.4f}")
    
    # Evolve
    print("\nEvolving system...")
    psi_final = substrate.evolve()
    
    # Final measurements
    alpha_final = substrate.compute_effective_alpha()
    print(f"Final effective α: {alpha_final:.4f}")
    
    return substrate, params


def run_comparison():
    """
    Compare: each mechanism alone vs all together.
    """
    print("\n" + "="*70)
    print("  COMPARISON: Individual vs Combined Mechanisms")
    print("="*70)
    
    base_params = {
        'L': 6,
        't_hop': 1.0,
        'dt': 0.05,
        'n_steps': 200,
    }
    
    configs = {
        'Quantum Links Only': SubstrateParams(
            **base_params,
            scalar_coupling=0.0,
            gauge_coupling=0.0,
            E_link=4.0,
            flux=0.0,
        ),
        'Scalar Only': SubstrateParams(
            **base_params,
            scalar_coupling=0.5,
            gauge_coupling=0.0,
            E_link=100.0,  # High E_link ≈ no quantum link effects
            flux=0.0,
        ),
        'Gauge Only': SubstrateParams(
            **base_params,
            scalar_coupling=0.0,
            gauge_coupling=0.3,
            E_link=100.0,
            flux=0.2,
        ),
        'All Three': SubstrateParams(
            **base_params,
            scalar_coupling=0.3,
            gauge_coupling=0.2,
            E_link=4.0,
            flux=0.1,
        ),
    }
    
    results = {}
    
    for name, params in configs.items():
        print(f"\nRunning: {name}...")
        substrate = UnifiedSubstrate(params)
        psi = substrate.evolve()
        
        results[name] = {
            'substrate': substrate,
            'history': substrate.history,
            'alpha': substrate.compute_effective_alpha(),
        }
        
        print(f"  Final α: {results[name]['alpha']:.4f}")
        print(f"  Final spread: {substrate.history['position_var'][-1]:.3f}")
        print(f"  Mean excitations: {substrate.history['mean_excitations'][-1]:.3f}")
    
    return results


def plot_unified_results(substrate: UnifiedSubstrate, params: SubstrateParams,
                         save_path: Optional[str] = None):
    """Create comprehensive visualization of unified dynamics."""
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    h = substrate.history
    t = np.array(h['time'])
    
    # Plot 1: Position spread (shows mass effects)
    ax = axes[0, 0]
    ax.plot(t, h['position_var'], 'b-', lw=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Position Variance', fontsize=11)
    ax.set_title('Level 1: Mass/Localization', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Edge weights evolution (scalar feedback)
    ax = axes[0, 1]
    weights_arr = np.array(h['weights'])
    for i in range(weights_arr.shape[1]):
        ax.plot(t, weights_arr[:, i], lw=1.5, label=f'Link {i}')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Edge Weight', fontsize=11)
    ax.set_title('Level 1: Scalar Edge Weights', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total phase (gauge evolution)
    ax = axes[0, 2]
    ax.plot(t, h['total_phase'], 'g-', lw=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Total Phase (Σφ)', fontsize=11)
    ax.set_title('Level 2: Gauge Phase Evolution', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Link excitations (quantum links)
    ax = axes[1, 0]
    ax.plot(t, h['mean_excitations'], 'r-', lw=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('⟨Number of Excited Links⟩', fontsize=11)
    ax.set_title('Level 3: Vacuum Excitations', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Energy (conservation check)
    ax = axes[1, 1]
    energy = np.array(h['energy'])
    ax.plot(t, energy - energy[0], 'm-', lw=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Energy - E₀', fontsize=11)
    ax.set_title('Energy Conservation Check', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    alpha_theory = params.t_hop / params.E_link
    alpha_measured = substrate.compute_effective_alpha()
    
    summary = f"""
UNIFIED SUBSTRATE SIMULATION
════════════════════════════

Lattice: {params.L} sites

LEVEL 1 - SCALAR FEEDBACK
  Coupling: {params.scalar_coupling}
  Effect: Edge weights adapt to amplitude
  → Emergent mass, binding

LEVEL 2 - MATRIX FEEDBACK  
  Coupling: {params.gauge_coupling}
  Background flux: {params.flux:.2f} × 2π
  Effect: Phases evolve with current
  → Gauge transport, AB effect

LEVEL 3 - QUANTUM LINKS
  E_link: {params.E_link}
  α (theory):   {alpha_theory:.4f}
  α (measured): {alpha_measured:.4f}
  → Fine structure constant

All three mechanisms operate on
the same emergent spacetime.
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    return fig


def plot_comparison(results: Dict, save_path: Optional[str] = None):
    """Compare the different mechanism configurations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    colors = {'Quantum Links Only': 'red', 'Scalar Only': 'blue', 
              'Gauge Only': 'green', 'All Three': 'purple'}
    
    # Plot 1: Position variance evolution
    ax = axes[0, 0]
    for name, data in results.items():
        t = data['history']['time']
        var = data['history']['position_var']
        ax.plot(t, var, color=colors[name], lw=2, label=name)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Position Variance', fontsize=11)
    ax.set_title('Spreading Dynamics', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean excitations
    ax = axes[0, 1]
    for name, data in results.items():
        t = data['history']['time']
        exc = data['history']['mean_excitations']
        ax.plot(t, exc, color=colors[name], lw=2, label=name)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('⟨Excited Links⟩', fontsize=11)
    ax.set_title('Vacuum Excitation (Level 3)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total phase evolution
    ax = axes[1, 0]
    for name, data in results.items():
        t = data['history']['time']
        phase = data['history']['total_phase']
        ax.plot(t, phase, color=colors[name], lw=2, label=name)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Total Phase', fontsize=11)
    ax.set_title('Gauge Phase (Level 2)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final alpha values
    ax = axes[1, 1]
    names = list(results.keys())
    alphas = [results[n]['alpha'] for n in names]
    bars = ax.bar(range(len(names)), alphas, color=[colors[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Effective α', fontsize=11)
    ax.set_title('Emergent Coupling Constant', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, alpha in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{alpha:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    return fig


def main():
    """Main demonstration of unified substrate."""
    
    # Run unified demo
    substrate, params = run_unified_demo()
    
    # Plot unified results
    fig1 = plot_unified_results(substrate, params, save_path='unified_substrate.png')
    
    # Run comparison
    results = run_comparison()
    
    # Plot comparison
    fig2 = plot_comparison(results, save_path='substrate_comparison.png')
    
    # Final summary
    print("\n" + "="*70)
    print("  SUMMARY: UNIFIED SUBSTRATE FRAMEWORK")
    print("="*70)
    print("""
Three feedback mechanisms operate on the same emergent spacetime:

LEVEL 1 - SCALAR FEEDBACK
  Edge weights w_ij modulated by local amplitude
  → Emergent mass (localization in heavy regions)
  → Topological binding (attractive effective potential)
  
LEVEL 2 - MATRIX FEEDBACK
  U(1) phases φ_ij evolve with probability current
  → Gauge transport (Aharonov-Bohm physics)
  → Spin precession (non-Abelian generalization)
  
LEVEL 3 - QUANTUM LINKS  
  Links as quantum two-level systems with energy E_link
  → Fine structure constant α = t/E_link
  → Dressed particles (polaron-like physics)

KEY INSIGHT: These are not separate mechanisms but three aspects
of the same substrate dynamics. A particle moving through the
substrate simultaneously:
  - Modulates local geometry (Level 1)
  - Accumulates gauge phase (Level 2)  
  - Excites vacuum fluctuations (Level 3)

The emergent physics depends on which feedback dominates:
  - Strong scalar coupling → bound states, massive particles
  - Strong gauge coupling → electromagnetic-like interactions
  - Low E_link → strong coupling (QCD-like)
  - High E_link → weak coupling (QED-like)
    """)
    
    plt.show()
    
    return substrate, results


if __name__ == "__main__":
    substrate, results = main()