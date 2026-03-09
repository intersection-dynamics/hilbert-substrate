"""
Task B1: Effective Bond Hamiltonian
====================================

Goal: Derive the Hamiltonian governing bond-only dynamics by
tracing out site degrees of freedom.

Key physics question: When two bonds share a site, the site 
mediates an effective bond-bond interaction. Do these mediated
interactions produce PLAQUETTE TERMS (products around closed loops)?

If yes → lattice gauge theory action emerges from echo dynamics.

Method:
1. Build full H on a small lattice (sites + bonds)
2. Project onto a reference site state sector
3. Use 2nd-order perturbation theory for effective bond-bond couplings
4. Also: direct numerical evolution approach
5. Decompose H_eff into local terms and check for plaquette structure
"""

import numpy as np
from scipy.linalg import expm, logm
from itertools import product as iprod
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=120)


# ============================================================
# Lattice infrastructure (streamlined from echo_model_v2)
# ============================================================

class EchoLattice:
    """
    Quantum lattice with sites on vertices and bonds on edges.
    Handles Hilbert space bookkeeping and operator embedding.
    """
    
    def __init__(self, n_sites, edges, d_B=2):
        self.n_sites = n_sites
        self.edges = edges
        self.n_bonds = len(edges)
        self.d_B = d_B
        
        # Adjacency
        self.neighbors = {i: [] for i in range(n_sites)}
        self.edge_index = {}
        for idx, (i, j) in enumerate(edges):
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
            self.edge_index[(i, j)] = idx
            self.edge_index[(j, i)] = idx
        
        # Dimensions: [site_0, site_1, ..., bond_0, bond_1, ...]
        self.d_site = 2
        self.dims = [self.d_site] * n_sites + [d_B] * self.n_bonds
        self.n_subsystems = len(self.dims)
        self.total_dim = int(np.prod(self.dims))
        
        # Precompute strides for fast index manipulation
        self.strides = np.ones(self.n_subsystems, dtype=int)
        for k in range(self.n_subsystems - 2, -1, -1):
            self.strides[k] = self.strides[k + 1] * self.dims[k + 1]
        
        # Subsystem indices
        self.site_idx = list(range(n_sites))
        self.bond_idx = list(range(n_sites, n_sites + self.n_bonds))
        
        # Dimensions of site and bond sectors
        self.d_sites_total = 2**n_sites
        self.d_bonds_total = d_B**self.n_bonds
        
        print(f"Lattice: {n_sites} sites, {self.n_bonds} bonds (d_B={d_B})")
        print(f"  Site space: {self.d_sites_total}, Bond space: {self.d_bonds_total}")
        print(f"  Total: {self.total_dim}")
    
    def _flat_index(self, config):
        """Convert subsystem configuration tuple to flat index."""
        return sum(c * s for c, s in zip(config, self.strides))
    
    def _embed_operator(self, op, subsystem_indices):
        """Embed a local operator into the full Hilbert space."""
        D = self.total_dim
        n = self.n_subsystems
        
        d_local = int(np.prod([self.dims[i] for i in subsystem_indices]))
        other_indices = [i for i in range(n) if i not in subsystem_indices]
        
        full_op = np.zeros((D, D), dtype=complex)
        
        local_dims = [self.dims[i] for i in subsystem_indices]
        other_dims = [self.dims[i] for i in other_indices]
        
        for other_config in iprod(*[range(d) for d in other_dims]):
            for local_bra in iprod(*[range(d) for d in local_dims]):
                for local_ket in iprod(*[range(d) for d in local_dims]):
                    flat_bra = 0
                    flat_ket = 0
                    for k, idx in enumerate(subsystem_indices):
                        flat_bra += local_bra[k] * self.strides[idx]
                        flat_ket += local_ket[k] * self.strides[idx]
                    for k, idx in enumerate(other_indices):
                        flat_bra += other_config[k] * self.strides[idx]
                        flat_ket += other_config[k] * self.strides[idx]
                    
                    local_flat_bra = 0
                    local_flat_ket = 0
                    local_stride = 1
                    for k in range(len(subsystem_indices) - 1, -1, -1):
                        local_flat_bra += local_bra[k] * local_stride
                        local_flat_ket += local_ket[k] * local_stride
                        local_stride *= local_dims[k]
                    
                    full_op[flat_bra, flat_ket] += op[local_flat_bra, local_flat_ket]
        
        return full_op
    
    def build_edge_hamiltonian(self, site_a, site_b, coupling=1.0):
        """
        Build the transmission Hamiltonian for edge (site_a, site_b).
        Acts on site_a ⊗ bond_{ab} ⊗ site_b in full space.
        """
        d_B = self.d_B
        bond_num = self.edge_index[(site_a, site_b)]
        
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Bond operators (spin-(d_B-1)/2)
        if d_B == 2:
            Bx, By, Bz = sx.copy(), sy.copy(), sz.copy()
        else:
            s = (d_B - 1) / 2.0
            Bx = np.zeros((d_B, d_B), dtype=complex)
            By = np.zeros((d_B, d_B), dtype=complex)
            Bz = np.zeros((d_B, d_B), dtype=complex)
            for m_idx in range(d_B):
                m = s - m_idx
                if m_idx + 1 < d_B:
                    mp = m - 1
                    coeff = np.sqrt(s*(s+1) - m*mp) * 0.5
                    Bx[m_idx, m_idx+1] = coeff
                    Bx[m_idx+1, m_idx] = coeff
                    By[m_idx, m_idx+1] = -1j * coeff
                    By[m_idx+1, m_idx] = 1j * coeff
                Bz[m_idx, m_idx] = m
        
        # H = σ_x⊗B_x⊗σ_x + σ_y⊗B_y⊗σ_y + σ_z⊗B_z⊗σ_z
        d_local = 2 * d_B * 2
        H_local = (np.kron(np.kron(sx, Bx), sx) +
                   np.kron(np.kron(sy, By), sy) +
                   np.kron(np.kron(sz, Bz), sz))
        
        H_local *= coupling
        
        subsystems = [site_a, self.n_sites + bond_num, site_b]
        return self._embed_operator(H_local, subsystems)
    
    def build_full_hamiltonian(self, coupling=1.0):
        """Sum of edge Hamiltonians over all edges."""
        H = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        for (i, j) in self.edges:
            H += self.build_edge_hamiltonian(i, j, coupling)
        return H
    
    def partial_trace_sites(self, rho_full):
        """
        Trace out ALL site degrees of freedom, leaving bond-only density matrix.
        rho_full: full density matrix (total_dim × total_dim)
        Returns: bond density matrix (d_bonds_total × d_bonds_total)
        """
        d_s = self.d_sites_total
        d_b = self.d_bonds_total
        
        # Reshape: the full space is organized as sites ⊗ bonds
        # We need to carefully handle the index ordering
        # Our convention: dims = [site_0, ..., site_{N-1}, bond_0, ..., bond_{M-1}]
        # So the natural tensor product IS sites_first ⊗ bonds
        
        # Reshape to (d_sites, d_bonds, d_sites, d_bonds)
        rho_tensor = rho_full.reshape(d_s, d_b, d_s, d_b)
        
        # Trace over sites: sum over site indices
        rho_bonds = np.trace(rho_tensor, axis1=0, axis2=2)
        
        return rho_bonds
    
    def partial_trace_bonds(self, rho_full):
        """Trace out bonds, leaving site-only density matrix."""
        d_s = self.d_sites_total
        d_b = self.d_bonds_total
        rho_tensor = rho_full.reshape(d_s, d_b, d_s, d_b)
        rho_sites = np.einsum('ibjb->ij', rho_tensor)
        return rho_sites
    
    def project_sites(self, operator, site_state):
        """
        Project a full-space operator onto a fixed site state.
        |ψ_sites⟩⟨ψ_sites| ⊗ I_bonds  ·  O  ·  |ψ_sites⟩⟨ψ_sites| ⊗ I_bonds
        
        Returns: d_bonds × d_bonds operator on bond space alone.
        """
        d_s = self.d_sites_total
        d_b = self.d_bonds_total
        
        # site_state should be a d_sites_total vector
        assert len(site_state) == d_s
        
        # Reshape operator
        O_tensor = operator.reshape(d_s, d_b, d_s, d_b)
        
        # Project: H_eff[b1, b2] = Σ_{s1,s2} ψ*_{s1} O[s1,b1,s2,b2] ψ_{s2}
        H_eff = np.einsum('i,iajb,j->ab', site_state.conj(), O_tensor, site_state)
        
        return H_eff


# ============================================================
# Pauli basis decomposition for bond operators
# ============================================================

def pauli_basis(n_qubits):
    """Generate the n-qubit Pauli basis."""
    I = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    single = [I, sx, sy, sz]
    labels_single = ['I', 'X', 'Y', 'Z']
    
    basis = []
    labels = []
    for indices in iprod(range(4), repeat=n_qubits):
        op = single[indices[0]]
        lab = labels_single[indices[0]]
        for k in range(1, n_qubits):
            op = np.kron(op, single[indices[k]])
            lab += labels_single[indices[k]]
        basis.append(op)
        labels.append(lab)
    
    return basis, labels


def decompose_in_pauli(H, n_qubits):
    """Decompose operator in Pauli basis. Returns {label: coefficient}."""
    basis, labels = pauli_basis(n_qubits)
    d = 2**n_qubits
    coeffs = {}
    for op, lab in zip(basis, labels):
        c = np.trace(H @ op) / d
        if abs(c) > 1e-10:
            coeffs[lab] = c
    return coeffs


def hamming_weight(label):
    """Count non-identity factors in a Pauli string."""
    return sum(1 for c in label if c != 'I')


# ============================================================
# Generalized basis decomposition for d_B > 2
# ============================================================

def gell_mann_basis(d):
    """Generalized Gell-Mann matrices for su(d), plus identity."""
    basis = [np.eye(d, dtype=complex)]  # Identity
    labels = ['I']
    
    # Symmetric off-diagonal
    for j in range(d):
        for k in range(j+1, d):
            M = np.zeros((d, d), dtype=complex)
            M[j, k] = 1
            M[k, j] = 1
            basis.append(M)
            labels.append(f'S{j}{k}')
    
    # Antisymmetric off-diagonal
    for j in range(d):
        for k in range(j+1, d):
            M = np.zeros((d, d), dtype=complex)
            M[j, k] = -1j
            M[k, j] = 1j
            basis.append(M)
            labels.append(f'A{j}{k}')
    
    # Diagonal
    for l in range(1, d):
        M = np.zeros((d, d), dtype=complex)
        for j in range(l):
            M[j, j] = 1
        M[l, l] = -l
        M *= np.sqrt(2.0 / (l * (l + 1)))
        basis.append(M)
        labels.append(f'D{l}')
    
    return basis, labels


def decompose_bond_operator(H_bonds, d_B, n_bonds):
    """
    Decompose a bond-space operator into tensor products of 
    single-bond operators.
    """
    if d_B == 2:
        return decompose_in_pauli(H_bonds, n_bonds)
    
    # For d_B > 2, use generalized Gell-Mann basis on each bond
    single_basis, single_labels = gell_mann_basis(d_B)
    d_total = d_B**n_bonds
    
    coeffs = {}
    for indices in iprod(range(len(single_basis)), repeat=n_bonds):
        op = single_basis[indices[0]]
        lab = single_labels[indices[0]]
        for k in range(1, n_bonds):
            op = np.kron(op, single_basis[indices[k]])
            lab += '⊗' + single_labels[indices[k]]
        
        c = np.trace(H_bonds @ op) / d_total
        if abs(c) > 1e-10:
            coeffs[lab] = c
    
    return coeffs


def operator_locality(label, separator='⊗'):
    """Count how many factors are non-identity."""
    if separator in label:
        parts = label.split(separator)
    else:
        parts = list(label)
    return sum(1 for p in parts if p != 'I')


# ============================================================
# METHOD 1: Direct projection onto site ground state
# ============================================================

def method_projection(lattice, coupling=0.5):
    """
    Project the full Hamiltonian onto a fixed site sector.
    
    H_eff = ⟨ψ_sites| H |ψ_sites⟩  (operator on bond space)
    
    This is zeroth-order: just the "classical" bond Hamiltonian
    for fixed site configuration.
    """
    print(f"\n{'='*60}")
    print(f"METHOD 1: Direct Projection (zeroth order)")
    print(f"{'='*60}")
    
    H_full = lattice.build_full_hamiltonian(coupling)
    
    # Try several site reference states
    d_s = lattice.d_sites_total
    
    results = {}
    
    # All-zero state
    psi_0 = np.zeros(d_s, dtype=complex)
    psi_0[0] = 1.0
    H_eff_0 = lattice.project_sites(H_full, psi_0)
    results['|00...0⟩'] = H_eff_0
    
    # All-one state  
    psi_1 = np.zeros(d_s, dtype=complex)
    psi_1[-1] = 1.0
    H_eff_1 = lattice.project_sites(H_full, psi_1)
    results['|11...1⟩'] = H_eff_1
    
    # Equal superposition
    psi_plus = np.ones(d_s, dtype=complex) / np.sqrt(d_s)
    H_eff_plus = lattice.project_sites(H_full, psi_plus)
    results['|+...+⟩'] = H_eff_plus
    
    # Random state
    np.random.seed(42)
    psi_rand = np.random.randn(d_s) + 1j * np.random.randn(d_s)
    psi_rand /= np.linalg.norm(psi_rand)
    H_eff_rand = lattice.project_sites(H_full, psi_rand)
    results['random'] = H_eff_rand
    
    for name, H_eff in results.items():
        print(f"\n  Site state: {name}")
        print(f"  H_eff norm: {np.linalg.norm(H_eff):.6f}")
        print(f"  H_eff is Hermitian: {np.allclose(H_eff, H_eff.conj().T)}")
        
        # Spectrum
        evals = np.linalg.eigvalsh(H_eff)
        print(f"  Spectrum: [{evals[0]:.4f}, ..., {evals[-1]:.4f}]")
        print(f"  Bandwidth: {evals[-1] - evals[0]:.4f}")
    
    return results


# ============================================================
# METHOD 2: Second-order perturbation theory (Schrieffer-Wolff)
# ============================================================

def method_perturbative(lattice, coupling=0.5):
    """
    Treat site-bond coupling as perturbation.
    
    H = H_0 + V where:
    H_0 = 0 (or site-only terms)
    V = Σ_edges H_{site_a, bond, site_b}  (the coupling)
    
    Second-order effective Hamiltonian on bonds:
    H_eff^(2) = P V Q (E_0 - Q H_0 Q)^{-1} Q V P
    
    where P projects onto reference site state, Q = 1-P.
    
    This generates BOND-BOND interactions mediated by virtual 
    site excitations — exactly the mechanism for plaquette terms.
    """
    print(f"\n{'='*60}")
    print(f"METHOD 2: Second-Order Perturbation Theory")
    print(f"{'='*60}")
    
    d_s = lattice.d_sites_total
    d_b = lattice.d_bonds_total
    D = lattice.total_dim
    
    # Reference site state: all zeros
    psi_sites = np.zeros(d_s, dtype=complex)
    psi_sites[0] = 1.0
    
    # Build projectors
    # P = |ψ_sites⟩⟨ψ_sites| ⊗ I_bonds
    # In matrix form over full space:
    P = np.zeros((D, D), dtype=complex)
    for b1 in range(d_b):
        for b2 in range(d_b):
            row = 0 * d_b + b1  # site index 0 (|00...0⟩)
            col = 0 * d_b + b2
            P[row, col] = 1.0  # Note: psi_sites[0]=1, rest=0
    
    Q = np.eye(D, dtype=complex) - P
    
    # Build full Hamiltonian
    H_full = lattice.build_full_hamiltonian(coupling)
    
    # Zeroth order: H_0 = 0 (no on-site energy for simplicity)
    # So (E_0 - Q H_0 Q)^{-1} = just need to handle the energy denominators
    
    # For H_0 = 0, the resolvent is ill-defined (all energies degenerate)
    # Instead, use a small site energy to break degeneracy
    
    # Add a site field: H_0 = Δ Σ_i σ_z^(i) on sites
    Delta = 2.0  # energy gap for site excitations
    H_site = np.zeros((D, D), dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    for i in range(lattice.n_sites):
        H_site += Delta * lattice._embed_operator(sz, [i])
    
    # Now H = H_0 + V where H_0 = H_site, V = H_full (the coupling)
    # E_0 = ground state energy of H_0 in site sector = -N*Delta (all spins up = |0⟩)
    E_0 = -lattice.n_sites * Delta
    
    # Q H_0 Q
    QH0Q = Q @ H_site @ Q
    
    # Resolvent: (E_0 - Q H_0 Q)^{-1} restricted to Q subspace
    # Add small regularization
    reg = 1e-10
    resolvent_matrix = E_0 * Q - QH0Q
    # Only invert in Q subspace
    evals_r, evecs_r = np.linalg.eigh(resolvent_matrix)
    # Zero eigenvalues correspond to P subspace — skip those
    resolvent = np.zeros((D, D), dtype=complex)
    for i in range(D):
        if abs(evals_r[i]) > reg:
            resolvent += (1.0 / evals_r[i]) * np.outer(evecs_r[:, i], evecs_r[:, i].conj())
    
    # V = coupling Hamiltonian
    V = H_full
    
    # First order: H_eff^(1) = P V P (projected on bonds)
    H_eff_1_full = P @ V @ P
    # Extract bond-only part
    H_eff_1 = H_eff_1_full[:d_b, :d_b]  # top-left block (site=|0...0⟩)
    
    # Second order: H_eff^(2) = P V Q resolvent Q V P
    QVP = Q @ V @ P
    H_eff_2_full = P @ V @ resolvent @ V @ P
    H_eff_2 = H_eff_2_full[:d_b, :d_b]
    
    # Make Hermitian (numerical noise)
    H_eff_1 = (H_eff_1 + H_eff_1.conj().T) / 2.0
    H_eff_2 = (H_eff_2 + H_eff_2.conj().T) / 2.0
    
    print(f"\n  First-order H_eff norm: {np.linalg.norm(H_eff_1):.6f}")
    print(f"  Second-order H_eff norm: {np.linalg.norm(H_eff_2):.6f}")
    
    H_eff_total = H_eff_1 + H_eff_2
    print(f"  Total (1st+2nd) norm: {np.linalg.norm(H_eff_total):.6f}")
    
    return H_eff_1, H_eff_2, H_eff_total


# ============================================================
# METHOD 3: Numerical evolution + bond state tomography
# ============================================================

def method_numerical_evolution(lattice, coupling=0.5, t_max=2.0, n_steps=20):
    """
    Evolve the full system, trace out sites at each step, 
    and extract the effective bond dynamics.
    
    If bonds evolve under some H_eff, then:
    ρ_bonds(t) = e^{-iH_eff t} ρ_bonds(0) e^{iH_eff t}
    
    We can extract H_eff by fitting the short-time evolution.
    """
    print(f"\n{'='*60}")
    print(f"METHOD 3: Numerical Evolution + Tomography")
    print(f"{'='*60}")
    
    H_full = lattice.build_full_hamiltonian(coupling)
    d_b = lattice.d_bonds_total
    
    # Initial state: sites in |0...0⟩, bonds in a known state
    # Use several initial bond states to overdetermine H_eff
    
    dt = t_max / n_steps
    U_dt = expm(-1j * dt * H_full)
    
    # Strategy: for small dt, 
    # ρ_bonds(dt) ≈ ρ_bonds(0) - i dt [H_eff, ρ_bonds(0)]
    # So: dρ/dt|_{t=0} = -i [H_eff, ρ_bonds(0)]
    # Given multiple initial states, we can solve for H_eff
    
    # Use computational basis states for bonds
    H_eff_estimates = []
    
    d_s = lattice.d_sites_total
    D = lattice.total_dim
    
    for bond_init_idx in range(min(d_b, 6)):  # Try several initial bond states
        # Full initial state: |0...0⟩_sites ⊗ |k⟩_bonds
        psi_init = np.zeros(D, dtype=complex)
        psi_init[0 * d_b + bond_init_idx] = 1.0  # site=|0...0⟩, bond=|k⟩
        
        # Evolve one small step
        psi_dt = U_dt @ psi_init
        
        # Get bond density matrices
        rho_0 = lattice.partial_trace_sites(np.outer(psi_init, psi_init.conj()))
        rho_dt = lattice.partial_trace_sites(np.outer(psi_dt, psi_dt.conj()))
        
        # Finite difference
        drho = (rho_dt - rho_0) / dt
        
        # drho ≈ -i[H_eff, rho_0] + dissipative terms
        # For pure bond state |k⟩⟨k|: [H_eff, |k⟩⟨k|] gives off-diagonal info
        # H_eff[m,k] for m≠k can be extracted from drho[m,k]
        
    # Better approach: use the full time series for ONE state
    # and fit with matrix logarithm
    
    print(f"\n  Evolving from |0...0⟩_sites ⊗ |0...0⟩_bonds")
    psi = np.zeros(D, dtype=complex)
    psi[0] = 1.0
    
    rho_bonds_series = []
    times = []
    for step in range(n_steps + 1):
        t = step * dt
        rho_full = np.outer(psi, psi.conj())
        rho_bonds = lattice.partial_trace_sites(rho_full)
        rho_bonds_series.append(rho_bonds)
        times.append(t)
        
        if step < n_steps:
            psi = U_dt @ psi
    
    # Check: is the bond evolution approximately unitary?
    # If so, Tr(ρ²) should stay near 1
    purities = [np.trace(rho @ rho).real for rho in rho_bonds_series]
    print(f"  Bond purity over time: {purities[0]:.4f} → {purities[-1]:.4f}")
    print(f"  (1.0 = unitary, <1 = dissipative)")
    
    # Entropies
    entropies = []
    for rho in rho_bonds_series:
        evals = np.linalg.eigvalsh(rho)
        evals = evals[evals > 1e-15]
        entropies.append(-np.sum(evals * np.log2(np.clip(evals, 1e-15, 1))))
    
    print(f"  Bond entropy: {entropies[0]:.4f} → {entropies[-1]:.4f}")
    
    # Extract effective H from short-time evolution
    # Use first few time steps where purity is still high
    rho_0 = rho_bonds_series[0]
    rho_1 = rho_bonds_series[1]
    
    # drho/dt ≈ -i[H_eff, rho_0] at t=0
    drho = (rho_1 - rho_0) / dt
    
    # For rho_0 = |0⟩⟨0| (pure state), this gives:
    # H_eff[m,0] = i * drho[m,0] for m≠0 (off-diagonal)
    # But diagonal requires more work
    
    print(f"\n  Short-time bond evolution (drho/dt at t=0):")
    print(f"  ||drho/dt|| = {np.linalg.norm(drho):.6f}")
    
    # Extract H_eff by fitting: find H such that -i[H, ρ₀] ≈ dρ/dt
    # Vectorize and solve least squares
    d = d_b
    
    # -i[H, ρ] = -i(Hρ - ρH)
    # vec(-i[H,ρ]) = -i(I⊗H - H^T⊗I) vec(ρ)
    # This is a linear equation in the entries of H
    
    # Use multiple initial states for better conditioning
    A_rows = []
    b_rows = []
    
    for bond_k in range(min(d_b, d_b)):  # All computational basis states
        psi_init = np.zeros(D, dtype=complex)
        psi_init[0 * d_b + bond_k] = 1.0
        
        psi_dt = U_dt @ psi_init
        rho_0_k = lattice.partial_trace_sites(np.outer(psi_init, psi_init.conj()))
        rho_1_k = lattice.partial_trace_sites(np.outer(psi_dt, psi_dt.conj()))
        drho_k = (rho_1_k - rho_0_k) / dt
        
        # Build the linear system for H_eff
        # drho = -i(H_eff @ rho - rho @ H_eff)
        for m in range(d):
            for n in range(d):
                row = np.zeros(d*d, dtype=complex)
                for p in range(d):
                    # H[m,p] * rho[p,n]
                    row[m*d + p] += -1j * rho_0_k[p, n]
                    # -rho[m,p] * H[p,n]
                    row[p*d + n] += 1j * rho_0_k[m, p]
                A_rows.append(row)
                b_rows.append(drho_k[m, n])
    
    A = np.array(A_rows)
    b = np.array(b_rows)
    
    # Solve least squares
    H_eff_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    H_eff_num = H_eff_vec.reshape(d, d)
    
    # Make Hermitian
    H_eff_num = (H_eff_num + H_eff_num.conj().T) / 2.0
    
    print(f"\n  Extracted H_eff (numerical):")
    print(f"  ||H_eff|| = {np.linalg.norm(H_eff_num):.6f}")
    print(f"  Is Hermitian: {np.allclose(H_eff_num, H_eff_num.conj().T)}")
    
    # Spectrum
    evals = np.linalg.eigvalsh(H_eff_num)
    print(f"  Spectrum: {evals}")
    
    return H_eff_num, times, purities, entropies


# ============================================================
# ANALYSIS: Decompose H_eff and look for plaquette terms
# ============================================================

def analyze_bond_hamiltonian(H_eff, lattice):
    """
    Decompose H_eff into single-bond, two-bond, and higher terms.
    Check for plaquette (loop) structure.
    """
    print(f"\n{'='*60}")
    print(f"BOND HAMILTONIAN STRUCTURE ANALYSIS")
    print(f"{'='*60}")
    
    d_B = lattice.d_B
    n_bonds = lattice.n_bonds
    
    if d_B == 2:
        coeffs = decompose_in_pauli(H_eff, n_bonds)
        
        # Organize by locality
        by_weight = {}
        for label, c in sorted(coeffs.items(), key=lambda x: -abs(x[1])):
            w = hamming_weight(label)
            if w not in by_weight:
                by_weight[w] = []
            by_weight[w].append((label, c))
        
        total_weight = sum(abs(c)**2 for c in coeffs.values())
        
        for w in sorted(by_weight.keys()):
            terms = by_weight[w]
            weight_sum = sum(abs(c)**2 for _, c in terms)
            pct = 100 * weight_sum / total_weight if total_weight > 0 else 0
            
            label_map = {0: 'Identity', 1: 'Single-bond', 2: 'Two-bond (nearest?)', 
                        3: 'Three-bond', 4: 'Four-bond (plaquette?)'}
            print(f"\n  Weight {w} ({label_map.get(w, f'{w}-body')}): "
                  f"{len(terms)} terms, {pct:.1f}% of total weight")
            
            # Show top terms
            sorted_terms = sorted(terms, key=lambda x: -abs(x[1]))
            for label, c in sorted_terms[:5]:
                # Identify which bonds are involved
                active_bonds = [i for i, ch in enumerate(label) if ch != 'I']
                bond_edges = [lattice.edges[i] for i in active_bonds] if active_bonds else []
                print(f"    {label}: {c.real:+.6f} {'+' if c.imag >= 0 else ''}{c.imag:.6f}i"
                      f"  (bonds: {active_bonds}, edges: {bond_edges})")
            if len(sorted_terms) > 5:
                print(f"    ... and {len(sorted_terms)-5} more")
        
        # Check for plaquette structure specifically
        print(f"\n  --- PLAQUETTE CHECK ---")
        check_plaquette_terms(coeffs, lattice)
    
    else:
        coeffs = decompose_bond_operator(H_eff, d_B, n_bonds)
        print(f"  Total terms: {len(coeffs)}")
        
        # Organize by locality
        by_weight = {}
        for label, c in coeffs.items():
            w = operator_locality(label)
            if w not in by_weight:
                by_weight[w] = []
            by_weight[w].append((label, c))
        
        for w in sorted(by_weight.keys()):
            terms = by_weight[w]
            print(f"\n  Weight {w}: {len(terms)} terms")
            sorted_terms = sorted(terms, key=lambda x: -abs(x[1]))
            for label, c in sorted_terms[:3]:
                print(f"    {label}: {abs(c):.6f}")
    
    return coeffs


def check_plaquette_terms(coeffs, lattice):
    """
    Check if two-bond terms connect bonds that share a vertex,
    and if higher-order terms form loops (plaquettes).
    """
    n_bonds = lattice.n_bonds
    
    # Build bond adjacency: two bonds are adjacent if they share a site
    bond_adjacent = {}
    for b1 in range(n_bonds):
        e1 = set(lattice.edges[b1])
        for b2 in range(b1+1, n_bonds):
            e2 = set(lattice.edges[b2])
            shared = e1 & e2
            if shared:
                bond_adjacent[(b1, b2)] = shared
    
    print(f"  Bond adjacency (shared sites):")
    for (b1, b2), shared in bond_adjacent.items():
        print(f"    Bond {b1} ({lattice.edges[b1]}) -- Bond {b2} ({lattice.edges[b2]})"
              f" share site(s) {shared}")
    
    # Check if two-bond terms in H_eff correspond to adjacent bonds
    two_body_terms = {lab: c for lab, c in coeffs.items() if hamming_weight(lab) == 2}
    
    n_adjacent = 0
    n_nonadjacent = 0
    for label, c in two_body_terms.items():
        active = [i for i, ch in enumerate(label) if ch != 'I']
        if len(active) == 2:
            b1, b2 = active
            if (min(b1,b2), max(b1,b2)) in bond_adjacent:
                n_adjacent += 1
            else:
                n_nonadjacent += 1
    
    print(f"\n  Two-bond terms: {n_adjacent} adjacent, {n_nonadjacent} non-adjacent")
    if n_adjacent > 0 and n_nonadjacent == 0:
        print(f"  → ALL two-bond terms connect adjacent bonds (share a site)")
        print(f"  → This is nearest-neighbor coupling on the bond graph!")
    
    # Check for loops: in a square lattice, 4-bond terms around a plaquette
    # For triangle, 3-bond terms around the triangle
    print(f"\n  Looking for loop terms...")
    for w in range(3, n_bonds+1):
        w_terms = {lab: c for lab, c in coeffs.items() if hamming_weight(lab) == w}
        if w_terms:
            for label, c in sorted(w_terms.items(), key=lambda x: -abs(x[1]))[:3]:
                active = [i for i, ch in enumerate(label) if ch != 'I']
                edges_in_term = [lattice.edges[i] for i in active]
                
                # Check if these edges form a loop
                all_sites = set()
                for e in edges_in_term:
                    all_sites.update(e)
                
                # A loop: each site appears exactly twice
                site_count = {}
                for e in edges_in_term:
                    for s in e:
                        site_count[s] = site_count.get(s, 0) + 1
                
                is_loop = all(v == 2 for v in site_count.values()) and len(site_count) == w
                
                print(f"    {w}-bond term: bonds {active}, edges {edges_in_term}"
                      f" |c|={abs(c):.6f} {'← LOOP!' if is_loop else ''}")


# ============================================================
# EXPERIMENTS
# ============================================================

def experiment_triangle(d_B=2):
    """Triangle graph: simplest system with a loop."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: TRIANGLE GRAPH (3 sites, 3 bonds, d_B={d_B})")
    print(f"{'='*70}")
    print(f"This is the simplest system with a plaquette (the triangle itself).")
    
    edges = [(0, 1), (1, 2), (0, 2)]
    lattice = EchoLattice(3, edges, d_B=d_B)
    
    # Method 1: Direct projection
    proj_results = method_projection(lattice, coupling=0.5)
    
    # Method 2: Perturbative
    H1, H2, H_total_pert = method_perturbative(lattice, coupling=0.5)
    
    # Method 3: Numerical
    H_eff_num, times, purities, entropies = method_numerical_evolution(
        lattice, coupling=0.3, t_max=1.0, n_steps=50)
    
    # Analyze all three
    print(f"\n\n--- Analyzing PROJECTION H_eff ---")
    analyze_bond_hamiltonian(proj_results['|00...0⟩'], lattice)
    
    print(f"\n\n--- Analyzing PERTURBATIVE H_eff (2nd order) ---")
    analyze_bond_hamiltonian(H_total_pert, lattice)
    
    print(f"\n\n--- Analyzing NUMERICAL H_eff ---")
    analyze_bond_hamiltonian(H_eff_num, lattice)
    
    return lattice, proj_results, H_total_pert, H_eff_num, times, purities, entropies


def experiment_square(d_B=2):
    """Square graph: 4 sites, 4 bonds, one plaquette."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: SQUARE GRAPH (4 sites, 4 bonds, d_B={d_B})")
    print(f"{'='*70}")
    print(f"The square has exactly one plaquette — the canonical lattice gauge object.")
    
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    lattice = EchoLattice(4, edges, d_B=d_B)
    
    # Method 1
    proj_results = method_projection(lattice, coupling=0.5)
    
    # Method 2
    H1, H2, H_total_pert = method_perturbative(lattice, coupling=0.5)
    
    # Method 3
    H_eff_num, times, purities, entropies = method_numerical_evolution(
        lattice, coupling=0.3, t_max=1.0, n_steps=50)
    
    print(f"\n\n--- Analyzing PROJECTION H_eff ---")
    analyze_bond_hamiltonian(proj_results['|00...0⟩'], lattice)
    
    print(f"\n\n--- Analyzing PERTURBATIVE H_eff (2nd order) ---")
    analyze_bond_hamiltonian(H_total_pert, lattice)
    
    print(f"\n\n--- Analyzing NUMERICAL H_eff ---")
    analyze_bond_hamiltonian(H_eff_num, lattice)
    
    return lattice, proj_results, H_total_pert, H_eff_num, times, purities, entropies


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)
    
    # Triangle
    (lat_tri, proj_tri, pert_tri, num_tri, 
     times_tri, pur_tri, ent_tri) = experiment_triangle(d_B=2)
    
    # Square
    (lat_sq, proj_sq, pert_sq, num_sq,
     times_sq, pur_sq, ent_sq) = experiment_square(d_B=2)
    
    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Triangle
    ax = axes[0, 0]
    ax.plot(times_tri, pur_tri, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Bond Purity Tr(ρ²)')
    ax.set_title('Triangle: Bond Purity')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(times_tri, ent_tri, 'r-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Bond von Neumann Entropy')
    ax.set_title('Triangle: Site-Bond Entanglement')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    # Spectrum comparison
    methods = ['Projection', 'Perturbative', 'Numerical']
    H_effs = [proj_tri['|00...0⟩'], pert_tri, num_tri]
    for i, (name, H) in enumerate(zip(methods, H_effs)):
        evals = np.sort(np.linalg.eigvalsh(H))
        ax.plot(range(len(evals)), evals, 'o-', label=name, markersize=4)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Energy')
    ax.set_title('Triangle: H_eff Spectra')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Row 2: Square
    ax = axes[1, 0]
    ax.plot(times_sq, pur_sq, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Bond Purity Tr(ρ²)')
    ax.set_title('Square: Bond Purity')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(times_sq, ent_sq, 'r-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Bond von Neumann Entropy')
    ax.set_title('Square: Site-Bond Entanglement')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    H_effs_sq = [proj_sq['|00...0⟩'], pert_sq, num_sq]
    for i, (name, H) in enumerate(zip(methods, H_effs_sq)):
        evals = np.sort(np.linalg.eigvalsh(H))
        ax.plot(range(len(evals)), evals, 'o-', label=name, markersize=4)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Energy')
    ax.set_title('Square: H_eff Spectra')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Task B1: Effective Bond Hamiltonian', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/home/claude/bond_hamiltonian_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")