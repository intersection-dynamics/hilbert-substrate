"""
decoherence_watcher.py

A tool to watch decoherence happen and see what emerges.

We don't assume the factorization is "correct" - we watch the dynamics
and see what patterns stabilize, what pointer bases form, how correlations
spread and freeze.

The idea: decoherence isn't a one-time event. It's a sustained process
where pointer bases keep selecting each other. We want to see that.

Key diagnostics:
- Coherence decay in various bases
- Entanglement growth between subsystems
- Purity evolution of reduced density matrices
- Pointer basis emergence (which states become stable?)
- Correlation structure (what gets correlated with what?)
- "Vortex" detection (where does structure form and persist?)

"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from scipy.linalg import expm, logm
import warnings


# =============================================================================
# Basic quantum operations
# =============================================================================

def tensor(*args):
    """Tensor product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def partial_trace(rho: np.ndarray, dims: List[int], keep: List[int]) -> np.ndarray:
    """
    Partial trace of density matrix.
    
    Parameters
    ----------
    rho : density matrix
    dims : list of dimensions for each subsystem
    keep : list of indices of subsystems to keep
    
    Returns
    -------
    Reduced density matrix over kept subsystems
    """
    n_sites = len(dims)
    keep = sorted(keep)
    trace_out = [i for i in range(n_sites) if i not in keep]
    
    # Reshape into tensor with one index per subsystem (twice - row and col)
    shape = dims + dims
    rho_tensor = rho.reshape(shape)
    
    # Trace out unwanted subsystems (from highest index down)
    for i in sorted(trace_out, reverse=True):
        rho_tensor = np.trace(rho_tensor, axis1=i, axis2=i + len(dims) - len([j for j in trace_out if j > i]))
        # Adjust dims for remaining traces
    
    # Actually let me rewrite this more carefully
    # This is tricky - let me use a cleaner approach
    
    return _partial_trace_clean(rho, dims, keep)


def _partial_trace_clean(rho: np.ndarray, dims: List[int], keep: List[int]) -> np.ndarray:
    """
    Partial trace implementation.
    
    Uses reshape and einsum for clarity.
    """
    n = len(dims)
    keep = sorted(keep)
    trace_out = sorted([i for i in range(n) if i not in keep])
    
    if len(trace_out) == 0:
        return rho.copy()
    
    if len(keep) == 0:
        return np.array([[np.trace(rho)]])
    
    # Reshape rho into tensor with 2n indices
    # First n indices are "row" indices, next n are "column" indices
    shape = list(dims) + list(dims)
    rho_tensor = rho.reshape(shape)
    
    # Build einsum string
    # Input has indices: i0 i1 i2 ... j0 j1 j2 ...
    # For traced indices, we need ik = jk (contract them)
    # For kept indices, they stay separate
    
    n_total = 2 * n
    in_indices = list(range(n_total))
    
    # For traced-out indices, make the "column" index equal to "row" index
    for t in trace_out:
        in_indices[n + t] = t  # j_t = i_t (contract)
    
    # Output indices: kept row indices, then kept column indices
    out_indices = [i for i in keep] + [n + i for i in keep]
    
    # Convert to letters for einsum
    def idx_to_char(i):
        return chr(ord('a') + i) if i < 26 else chr(ord('A') + i - 26)
    
    in_str = ''.join(idx_to_char(i) for i in in_indices)
    out_str = ''.join(idx_to_char(i) for i in out_indices)
    
    einsum_str = f"{in_str}->{out_str}"
    
    result = np.einsum(einsum_str, rho_tensor)
    
    # Reshape to matrix
    kept_dims = [dims[i] for i in keep]
    d_keep = int(np.prod(kept_dims))
    return result.reshape(d_keep, d_keep)


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2) - 1 for pure, 1/d for maximally mixed."""
    return np.real(np.trace(rho @ rho))


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    """S = -Tr(rho log rho)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > eps]
    return -np.sum(eigenvalues * np.log(eigenvalues))


def coherence_l1(rho: np.ndarray) -> float:
    """L1 norm of off-diagonal elements - basis-dependent coherence measure."""
    return np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))


def coherence_in_basis(rho: np.ndarray, basis: np.ndarray) -> float:
    """Coherence measured in a specific basis (columns of basis matrix)."""
    rho_transformed = basis.conj().T @ rho @ basis
    return coherence_l1(rho_transformed)


# =============================================================================
# Pauli matrices and common operators
# =============================================================================

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = [I, X, Y, Z]


def pauli_string(indices: List[int], n_qubits: int) -> np.ndarray:
    """
    Build a Pauli string operator.
    indices[i] in {0,1,2,3} -> {I, X, Y, Z} on qubit i
    """
    ops = [PAULIS[indices[i]] if i < len(indices) else I for i in range(n_qubits)]
    return tensor(*ops)


# =============================================================================
# Hamiltonian builders
# =============================================================================

def heisenberg_chain(n: int, J: float = 1.0, periodic: bool = False) -> np.ndarray:
    """
    Heisenberg XXX chain: H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    """
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    
    pairs = list(range(n - 1))
    if periodic and n > 2:
        pairs.append((n-1, 0))  # wrap around
    
    for i in pairs:
        j = (i + 1) % n
        for P in [X, Y, Z]:
            ops = [I] * n
            ops[i] = P
            ops[j] = P
            H += J * tensor(*ops)
    
    return H


def ising_chain(n: int, J: float = 1.0, h: float = 0.5, periodic: bool = False) -> np.ndarray:
    """
    Transverse field Ising: H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
    """
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    
    # ZZ interactions
    pairs = list(range(n - 1))
    if periodic and n > 2:
        pairs.append(n - 1)
    
    for i in pairs:
        j = (i + 1) % n
        ops = [I] * n
        ops[i] = Z
        ops[j] = Z
        H -= J * tensor(*ops)
    
    # Transverse field
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= h * tensor(*ops)
    
    return H


def random_local_hamiltonian(n: int, k: int = 2, n_terms: int = None, 
                             strength: float = 1.0, seed: int = None) -> np.ndarray:
    """
    Random k-local Hamiltonian.
    
    Parameters
    ----------
    n : number of qubits
    k : locality (max number of qubits per term)
    n_terms : number of random terms (default: n * k)
    strength : overall scale
    seed : random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_terms is None:
        n_terms = n * k
    
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    
    for _ in range(n_terms):
        # Random subset of k qubits
        sites = np.random.choice(n, size=min(k, n), replace=False)
        # Random Paulis on those sites
        indices = [0] * n
        for s in sites:
            indices[s] = np.random.randint(1, 4)  # X, Y, or Z (not I)
        
        coeff = strength * np.random.randn()
        H += coeff * pauli_string(indices, n)
    
    return H


def system_environment_hamiltonian(n_sys: int, n_env: int, 
                                   J_sys: float = 1.0,
                                   J_env: float = 0.5,
                                   J_int: float = 0.3) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Build a Hamiltonian with explicit system-environment structure.
    
    Returns H, sys_indices, env_indices
    """
    n = n_sys + n_env
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    
    sys_indices = list(range(n_sys))
    env_indices = list(range(n_sys, n))
    
    # System internal dynamics (Heisenberg-like)
    for i in range(n_sys - 1):
        for P in [X, Y, Z]:
            ops = [I] * n
            ops[i] = P
            ops[i + 1] = P
            H += J_sys * tensor(*ops)
    
    # Environment internal dynamics
    for i in range(n_sys, n - 1):
        for P in [X, Y, Z]:
            ops = [I] * n
            ops[i] = P
            ops[i + 1] = P
            H += J_env * tensor(*ops)
    
    # System-environment coupling
    for i in range(n_sys):
        env_partner = n_sys + (i % n_env)
        ops = [I] * n
        ops[i] = Z
        ops[env_partner] = Z
        H += J_int * tensor(*ops)
    
    return H, sys_indices, env_indices


# =============================================================================
# State preparation
# =============================================================================

def basis_state(bits: List[int]) -> np.ndarray:
    """Computational basis state |bits>"""
    n = len(bits)
    idx = sum(b * 2**(n - 1 - i) for i, b in enumerate(bits))
    state = np.zeros(2**n, dtype=complex)
    state[idx] = 1.0
    return state


def plus_state(n: int) -> np.ndarray:
    """|+>^n - maximally coherent in computational basis"""
    return np.ones(2**n, dtype=complex) / np.sqrt(2**n)


def random_pure_state(n: int, seed: int = None) -> np.ndarray:
    """Haar-random pure state"""
    if seed is not None:
        np.random.seed(seed)
    state = np.random.randn(2**n) + 1j * np.random.randn(2**n)
    return state / np.linalg.norm(state)


def ghz_state(n: int) -> np.ndarray:
    """GHZ state (|00...0> + |11...1>)/sqrt(2)"""
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)
    state[-1] = 1.0 / np.sqrt(2)
    return state


def product_state(single_states: List[np.ndarray]) -> np.ndarray:
    """Tensor product of single-qubit states"""
    result = single_states[0]
    for s in single_states[1:]:
        result = np.kron(result, s)
    return result


# =============================================================================
# Evolution and measurement
# =============================================================================

def evolve(state: np.ndarray, H: np.ndarray, t: float) -> np.ndarray:
    """Unitary evolution: |psi(t)> = exp(-iHt)|psi(0)>"""
    U = expm(-1j * H * t)
    return U @ state


def evolve_density(rho: np.ndarray, H: np.ndarray, t: float) -> np.ndarray:
    """Evolution of density matrix"""
    U = expm(-1j * H * t)
    return U @ rho @ U.conj().T


# =============================================================================
# Decoherence diagnostics
# =============================================================================

@dataclass
class DecoherenceSnapshot:
    """Snapshot of decoherence diagnostics at one time."""
    time: float = 0.0
    
    # Global state properties
    global_purity: float = 1.0  # Should stay 1 for pure state evolution
    
    # Subsystem properties (for each tracked subsystem)
    subsystem_purities: Dict[str, float] = field(default_factory=dict)
    subsystem_entropies: Dict[str, float] = field(default_factory=dict)
    
    # Coherence in different bases
    coherences: Dict[str, float] = field(default_factory=dict)
    
    # Entanglement between subsystem pairs
    mutual_informations: Dict[str, float] = field(default_factory=dict)
    
    # Pointer basis diagnostics
    diagonal_weights: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Correlation matrix (if computed)
    correlation_matrix: Optional[np.ndarray] = None


@dataclass 
class DecoherenceHistory:
    """Full history of decoherence evolution."""
    snapshots: List[DecoherenceSnapshot] = field(default_factory=list)
    
    # Metadata
    n_qubits: int = 0
    dims: List[int] = field(default_factory=list)
    subsystem_labels: List[str] = field(default_factory=list)
    
    def times(self) -> np.ndarray:
        return np.array([s.time for s in self.snapshots])
    
    def get_series(self, attr: str, key: str = None) -> np.ndarray:
        """Extract a time series of some diagnostic."""
        if key is None:
            return np.array([getattr(s, attr) for s in self.snapshots])
        else:
            return np.array([getattr(s, attr).get(key, np.nan) for s in self.snapshots])


class DecoherenceWatcher:
    """
    Watches decoherence dynamics and tracks what emerges.
    """
    
    def __init__(self, n_qubits: int, H: np.ndarray, 
                 subsystems: Dict[str, List[int]] = None):
        """
        Parameters
        ----------
        n_qubits : number of qubits
        H : Hamiltonian
        subsystems : dict mapping names to lists of qubit indices
                     e.g. {'system': [0,1], 'environment': [2,3,4]}
        """
        self.n_qubits = n_qubits
        self.dims = [2] * n_qubits
        self.H = H
        
        if subsystems is None:
            # Default: each qubit is its own subsystem
            subsystems = {f'q{i}': [i] for i in range(n_qubits)}
        self.subsystems = subsystems
        
        self.history = DecoherenceHistory(
            n_qubits=n_qubits,
            dims=self.dims,
            subsystem_labels=list(subsystems.keys())
        )
        
        # Bases to track coherence in
        self.bases = {
            'computational': np.eye(2**n_qubits, dtype=complex),
            'hadamard': self._hadamard_basis(),
        }
    
    def _hadamard_basis(self) -> np.ndarray:
        """Hadamard-transformed basis"""
        H_single = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        H_full = H_single
        for _ in range(self.n_qubits - 1):
            H_full = np.kron(H_full, H_single)
        return H_full
    
    def snapshot(self, state: np.ndarray, t: float) -> DecoherenceSnapshot:
        """Take a snapshot of all decoherence diagnostics."""
        
        # Density matrix
        rho = np.outer(state, state.conj())
        
        snap = DecoherenceSnapshot()
        snap.time = t
        
        # Global purity (should be 1 for pure state)
        snap.global_purity = purity(rho)
        
        # Subsystem diagnostics
        for name, indices in self.subsystems.items():
            rho_sub = _partial_trace_clean(rho, self.dims, indices)
            snap.subsystem_purities[name] = purity(rho_sub)
            snap.subsystem_entropies[name] = von_neumann_entropy(rho_sub)
            
            # Diagonal weights in computational basis
            snap.diagonal_weights[name] = np.real(np.diag(rho_sub))
        
        # Coherence in different bases
        for basis_name, basis in self.bases.items():
            snap.coherences[basis_name] = coherence_in_basis(rho, basis)
        
        # Mutual information between subsystem pairs
        subsystem_names = list(self.subsystems.keys())
        for i, name_a in enumerate(subsystem_names):
            for name_b in subsystem_names[i+1:]:
                indices_a = self.subsystems[name_a]
                indices_b = self.subsystems[name_b]
                indices_ab = sorted(set(indices_a) | set(indices_b))
                
                S_a = snap.subsystem_entropies[name_a]
                S_b = snap.subsystem_entropies[name_b]
                
                rho_ab = _partial_trace_clean(rho, self.dims, indices_ab)
                S_ab = von_neumann_entropy(rho_ab)
                
                # I(A:B) = S(A) + S(B) - S(AB)
                MI = S_a + S_b - S_ab
                snap.mutual_informations[f'{name_a}:{name_b}'] = MI
        
        # Single-site correlation matrix
        snap.correlation_matrix = self._compute_correlation_matrix(rho)
        
        return snap
    
    def _compute_correlation_matrix(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute ZZ correlation matrix: C_ij = <Z_i Z_j> - <Z_i><Z_j>
        """
        n = self.n_qubits
        C = np.zeros((n, n))
        
        # Single-site expectations
        Z_exp = np.zeros(n)
        for i in range(n):
            ops = [I] * n
            ops[i] = Z
            Z_i = tensor(*ops)
            Z_exp[i] = np.real(np.trace(rho @ Z_i))
        
        # Two-site correlations
        for i in range(n):
            for j in range(n):
                if i == j:
                    C[i, j] = 1 - Z_exp[i]**2  # Variance
                else:
                    ops = [I] * n
                    ops[i] = Z
                    ops[j] = Z
                    ZZ = tensor(*ops)
                    ZZ_exp = np.real(np.trace(rho @ ZZ))
                    C[i, j] = ZZ_exp - Z_exp[i] * Z_exp[j]
        
        return C
    
    def watch(self, initial_state: np.ndarray, 
              t_max: float, n_steps: int = 100) -> DecoherenceHistory:
        """
        Watch decoherence unfold.
        
        Parameters
        ----------
        initial_state : initial pure state
        t_max : maximum time
        n_steps : number of time steps
        
        Returns
        -------
        DecoherenceHistory with all snapshots
        """
        times = np.linspace(0, t_max, n_steps)
        state = initial_state.copy()
        
        self.history = DecoherenceHistory(
            n_qubits=self.n_qubits,
            dims=self.dims,
            subsystem_labels=list(self.subsystems.keys())
        )
        
        # Initial snapshot
        self.history.snapshots.append(self.snapshot(state, 0))
        
        # Evolve and snapshot
        dt = times[1] - times[0] if len(times) > 1 else t_max
        U = expm(-1j * self.H * dt)
        
        for t in times[1:]:
            state = U @ state
            self.history.snapshots.append(self.snapshot(state, t))
        
        return self.history
    
    def find_pointer_basis(self, rho: np.ndarray, subsystem: str,
                           threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attempt to identify pointer states for a subsystem.
        
        These are eigenstates of the reduced density matrix that remain
        stable under further evolution (approximately).
        
        Returns eigenvalues and eigenvectors of reduced density matrix.
        """
        indices = self.subsystems[subsystem]
        rho_sub = _partial_trace_clean(rho, self.dims, indices)
        
        eigenvalues, eigenvectors = np.linalg.eigh(rho_sub)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Filter by threshold
        significant = eigenvalues > threshold
        
        return eigenvalues[significant], eigenvectors[:, significant]


# =============================================================================
# Visualization
# =============================================================================

def plot_decoherence_history(history: DecoherenceHistory, 
                             figsize: Tuple[int, int] = (14, 10)):
    """Plot the decoherence history."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    times = history.times()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Subsystem purities
    ax = axes[0, 0]
    for name in history.subsystem_labels:
        purities = history.get_series('subsystem_purities', name)
        ax.plot(times, purities, label=name)
    ax.set_xlabel('Time')
    ax.set_ylabel('Purity')
    ax.set_title('Subsystem Purities (1=pure, 0.5=mixed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Subsystem entropies
    ax = axes[0, 1]
    for name in history.subsystem_labels:
        entropies = history.get_series('subsystem_entropies', name)
        ax.plot(times, entropies, label=name)
    ax.set_xlabel('Time')
    ax.set_ylabel('von Neumann Entropy')
    ax.set_title('Subsystem Entropies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Coherences in different bases
    ax = axes[0, 2]
    for basis_name in ['computational', 'hadamard']:
        coh = history.get_series('coherences', basis_name)
        ax.plot(times, coh, label=basis_name)
    ax.set_xlabel('Time')
    ax.set_ylabel('L1 Coherence')
    ax.set_title('Coherence in Different Bases')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Mutual informations
    ax = axes[1, 0]
    mi_keys = list(history.snapshots[0].mutual_informations.keys())
    for key in mi_keys[:5]:  # Limit to avoid clutter
        mi = history.get_series('mutual_informations', key)
        ax.plot(times, mi, label=key)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mutual Information')
    ax.set_title('Entanglement Between Subsystems')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Correlation matrix at final time
    ax = axes[1, 1]
    final_corr = history.snapshots[-1].correlation_matrix
    im = ax.imshow(final_corr, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title(f'ZZ Correlations at t={times[-1]:.2f}')
    ax.set_xlabel('Qubit j')
    ax.set_ylabel('Qubit i')
    plt.colorbar(im, ax=ax)
    
    # 6. Global purity (sanity check - should be 1)
    ax = axes[1, 2]
    global_pur = history.get_series('global_purity')
    ax.plot(times, global_pur, 'k-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Global Purity')
    ax.set_title('Global Purity (should stay = 1)')
    ax.set_ylim(0.99, 1.01)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decoherence_history.png', dpi=150)
    plt.show()
    
    return fig


def plot_correlation_evolution(history: DecoherenceHistory,
                               n_snapshots: int = 6):
    """Plot how the correlation matrix evolves over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    times = history.times()
    indices = np.linspace(0, len(history.snapshots) - 1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for ax, idx in zip(axes, indices):
        snap = history.snapshots[idx]
        im = ax.imshow(snap.correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f't = {snap.time:.3f}')
        ax.set_xlabel('Qubit j')
        ax.set_ylabel('Qubit i')
    
    plt.suptitle('Evolution of ZZ Correlation Structure')
    plt.tight_layout()
    plt.savefig('correlation_evolution.png', dpi=150)
    plt.show()
    
    return fig


# =============================================================================
# Demo / Main
# =============================================================================

def demo_system_environment():
    """
    Demo: Watch decoherence in a system coupled to an environment.
    """
    print("=" * 60)
    print("DECOHERENCE WATCHER DEMO")
    print("System-Environment Setup")
    print("=" * 60)
    
    # Setup
    n_sys = 2
    n_env = 4
    n_total = n_sys + n_env
    
    print(f"\nSystem: {n_sys} qubits")
    print(f"Environment: {n_env} qubits")
    print(f"Total: {n_total} qubits")
    
    # Build Hamiltonian
    H, sys_idx, env_idx = system_environment_hamiltonian(
        n_sys, n_env,
        J_sys=1.0,   # System internal coupling
        J_env=0.5,   # Environment internal coupling  
        J_int=0.3    # System-environment coupling
    )
    
    print(f"\nHamiltonian built:")
    print(f"  System coupling: J_sys = 1.0")
    print(f"  Environment coupling: J_env = 0.5")
    print(f"  Interaction coupling: J_int = 0.3")
    
    # Define subsystems
    subsystems = {
        'system': sys_idx,
        'env': env_idx,
        'q0': [0],  # Track individual qubits too
        'q1': [1],
    }
    
    # Create watcher
    watcher = DecoherenceWatcher(n_total, H, subsystems)
    
    # Initial state: system in superposition, environment in ground state
    # |psi_0> = |+>|+> ⊗ |0>^n_env
    sys_state = plus_state(n_sys)
    env_state = basis_state([0] * n_env)
    initial_state = np.kron(sys_state, env_state)
    
    print(f"\nInitial state: |++> ⊗ |00...0>")
    print(f"  System: maximally coherent superposition")
    print(f"  Environment: ground state")
    
    # Watch decoherence
    print(f"\nWatching evolution...")
    t_max = 10.0
    n_steps = 200
    
    history = watcher.watch(initial_state, t_max=t_max, n_steps=n_steps)
    
    # Report
    print(f"\n{'='*60}")
    print("RESULTS")
    print("=" * 60)
    
    initial = history.snapshots[0]
    final = history.snapshots[-1]
    
    print(f"\n--- Purity ---")
    print(f"{'Subsystem':<15} {'Initial':>10} {'Final':>10} {'Change':>10}")
    for name in ['system', 'env']:
        p0 = initial.subsystem_purities[name]
        pf = final.subsystem_purities[name]
        print(f"{name:<15} {p0:>10.4f} {pf:>10.4f} {pf-p0:>+10.4f}")
    
    print(f"\n--- Entropy ---")
    print(f"{'Subsystem':<15} {'Initial':>10} {'Final':>10} {'Change':>10}")
    for name in ['system', 'env']:
        s0 = initial.subsystem_entropies[name]
        sf = final.subsystem_entropies[name]
        print(f"{name:<15} {s0:>10.4f} {sf:>10.4f} {sf-s0:>+10.4f}")
    
    print(f"\n--- Coherence (computational basis) ---")
    c0 = initial.coherences['computational']
    cf = final.coherences['computational']
    print(f"Initial: {c0:.4f}")
    print(f"Final:   {cf:.4f}")
    print(f"Decay:   {(c0-cf)/c0*100:.1f}%")
    
    print(f"\n--- Entanglement (Mutual Information) ---")
    mi_key = 'system:env'
    mi0 = initial.mutual_informations.get(mi_key, 0)
    mif = final.mutual_informations.get(mi_key, 0)
    print(f"System-Environment MI:")
    print(f"  Initial: {mi0:.4f}")
    print(f"  Final:   {mif:.4f}")
    
    print(f"\n--- Correlation Structure ---")
    print("Final ZZ correlation matrix:")
    print(np.array2string(final.correlation_matrix, precision=3, suppress_small=True))
    
    # Plot
    print("\nGenerating plots...")
    plot_decoherence_history(history)
    plot_correlation_evolution(history)
    
    print("\nDone! Check decoherence_history.png and correlation_evolution.png")
    
    return history


def demo_pointer_basis_formation():
    """
    Demo: Watch pointer basis form in a measurement-like setup.
    """
    print("=" * 60)
    print("POINTER BASIS FORMATION DEMO")
    print("=" * 60)
    
    # Simple setup: 1 system qubit, 3 environment qubits
    # Interaction designed to select Z basis as pointer
    n = 4
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    # System self-Hamiltonian (small, in X direction - tries to rotate)
    ops = [X, I, I, I]
    H += 0.1 * tensor(*ops)
    
    # Environment self-Hamiltonian
    for i in range(1, n):
        ops = [I] * n
        ops[i] = X
        H += 0.2 * tensor(*ops)
    
    # System-environment ZZ coupling (selects Z basis as pointer)
    for i in range(1, n):
        ops = [I] * n
        ops[0] = Z
        ops[i] = Z
        H += 0.5 * tensor(*ops)
    
    print("Setup:")
    print("  1 system qubit + 3 environment qubits")
    print("  System tries to rotate (X term)")
    print("  But ZZ coupling to environment selects Z as pointer basis")
    
    subsystems = {
        'system': [0],
        'env': [1, 2, 3],
    }
    
    watcher = DecoherenceWatcher(n, H, subsystems)
    
    # Initial state: system in |+>, environment in |000>
    initial_state = np.kron(
        (basis_state([0]) + basis_state([1])) / np.sqrt(2),
        basis_state([0, 0, 0])
    )
    
    print("\nInitial state: |+> ⊗ |000>")
    
    history = watcher.watch(initial_state, t_max=20.0, n_steps=300)
    
    # Check if pointer basis formed
    final_state = evolve(initial_state, H, 20.0)
    rho = np.outer(final_state, final_state.conj())
    
    eigenvalues, eigenvectors = watcher.find_pointer_basis(rho, 'system', threshold=0.01)
    
    print(f"\n--- Pointer Basis Analysis ---")
    print(f"Eigenvalues of reduced density matrix: {eigenvalues}")
    print(f"Number of significant pointer states: {len(eigenvalues)}")
    
    if len(eigenvalues) > 0:
        print("\nPointer states (in computational basis):")
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            print(f"  State {i}: weight={val:.4f}, |coeff|² = {np.abs(vec)**2}")
    
    plot_decoherence_history(history)
    
    return history


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DECOHERENCE WATCHER")
    print("Watching what emerges from unitary evolution")
    print("="*60 + "\n")
    
    # Run demos
    history1 = demo_system_environment()
    print("\n" + "="*60 + "\n")
    history2 = demo_pointer_basis_formation()