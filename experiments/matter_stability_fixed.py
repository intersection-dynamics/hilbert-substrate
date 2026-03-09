"""
Matter Stability Constraint: Hopping/Ising Ratio Determination
================================================================

The bandwidth constraint selects U(1) gauge structure (XX+YY+ZZ) over
cross terms, but overselects Ising (ZZ) because it has zero transport
velocity. A pure Ising universe is frozen — no particle transport,
no bound state formation, no observers.

This script tests the matter stability constraint: what is the minimum
hopping fraction (within the U(1) sector) that supports stable bound
states in 3D? This determines the kinetic/potential ratio that the
framework predicts.

Approach:
  1. Fix 3D cubic lattice (the dimensionality selected by HSF)
  2. Parameterize H = α(XX+YY) + (1-α)(ZZ) per bond (U(1) sector only)
  3. For each α, prepare two-particle initial state with particles adjacent
  4. Evolve under H, measure binding diagnostics:
     - Correlation function <n_i n_j> between initially adjacent sites
     - Mean separation of the two-particle wavefunction
     - Participation ratio (how spread out the state becomes)
     - Bound state energy gap (spectral signature of binding)
  5. Map α → binding strength to find critical α_c
  6. Below α_c: no stable matter. Above α_c: bound states form.
  7. α_c is the framework's prediction for the kinetic/potential ratio.

We also compare 1D, 2D, 3D to verify dimensional dependence.

Usage:
    python matter_stability.py --quick       # Fast test (small systems)
    python matter_stability.py --standard    # Main sweep
    python matter_stability.py --dimensional # Compare dimensions
    python matter_stability.py --full        # Everything + plots
"""

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, kron as sp_kron, eye as sp_eye
import time
import argparse
import json


# ============================================================
# PAULI OPERATORS
# ============================================================

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
Sp = (X + 1j * Y) / 2   # |1><0| = creation
Sm = (X - 1j * Y) / 2   # |0><1| = annihilation
n_op = (I2 - Z) / 2      # |1><1| = number operator


# ============================================================
# LATTICE CONSTRUCTION
# ============================================================

def build_lattice(dim, L):
    """
    Build a d-dimensional cubic lattice with side length L.
    Returns: N (number of sites), edges (list of (i,j) pairs),
             coord_map (site_index -> (x,y,z,...) tuple)
    
    Uses periodic boundary conditions.
    """
    if dim == 1:
        N = L
        edges = [(i, (i + 1) % N) for i in range(N)]
        coord_map = {i: (i,) for i in range(N)}
    
    elif dim == 2:
        N = L * L
        edges = []
        coord_map = {}
        for r in range(L):
            for c in range(L):
                idx = r * L + c
                coord_map[idx] = (r, c)
                # Right neighbor
                right = r * L + (c + 1) % L
                edges.append((idx, right))
                # Down neighbor
                down = ((r + 1) % L) * L + c
                edges.append((idx, down))
    
    elif dim == 3:
        N = L * L * L
        edges = []
        coord_map = {}
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    idx = x * L * L + y * L + z
                    coord_map[idx] = (x, y, z)
                    # +x neighbor
                    nx = ((x + 1) % L) * L * L + y * L + z
                    edges.append((idx, nx))
                    # +y neighbor
                    ny = x * L * L + ((y + 1) % L) * L + z
                    edges.append((idx, ny))
                    # +z neighbor
                    nz = x * L * L + y * L + ((z + 1) % L)
                    edges.append((idx, nz))
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    
    return N, edges, coord_map


def lattice_distance(coord1, coord2, L):
    """Manhattan distance with periodic boundaries."""
    d = 0
    for c1, c2 in zip(coord1, coord2):
        diff = abs(c1 - c2)
        d += min(diff, L - diff)
    return d


# ============================================================
# HAMILTONIAN CONSTRUCTION (2-PARTICLE SECTOR)
# ============================================================

def build_2particle_basis(N):
    """
    Build the 2-particle Fock basis for N sites.
    States are |i,j> with i < j (two fermions at sites i and j).
    Returns list of (i, j) pairs and dimension.
    """
    basis = []
    for i in range(N):
        for j in range(i + 1, N):
            basis.append((i, j))
    return basis


def build_u1_hamiltonian_2particle(N, edges, alpha, J=1.0):
    """
    Build the Hamiltonian in the 2-particle sector.
    
    H = α * J * Σ_{<ij>} (XX + YY)/2 + (1-α) * J * Σ_{<ij>} ZZ
    
    In fermion language (Jordan-Wigner):
      Hopping: (XX+YY)/2 = S+S- + S-S+ = c†_i c_j + c†_j c_i
      Ising:   ZZ = (2n_i - 1)(2n_j - 1) = 4 n_i n_j - 2n_i - 2n_j + 1
    
    In the 2-particle sector:
      - Hopping moves one particle to an adjacent empty site
      - Ising gives an energy shift when two particles are on connected sites
    
    Parameters:
        N: number of lattice sites
        edges: list of (i,j) bond pairs
        alpha: hopping fraction (0 = pure Ising, 1 = pure hopping)
        J: overall coupling strength
    """
    basis = build_2particle_basis(N)
    dim = len(basis)
    basis_idx = {(i, j): k for k, (i, j) in enumerate(basis)}
    
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Build adjacency set for quick lookup
    adj = set()
    for (i, j) in edges:
        adj.add((min(i, j), max(i, j)))
    
    t = alpha * J        # hopping amplitude
    V = (1 - alpha) * J  # Ising coupling
    
    for k, (s1, s2) in enumerate(basis):
        # ---- Ising (diagonal) contribution ----
        # ZZ on each bond: if both particles sit on a bond, energy = +4V
        # If one particle on each end of a bond: energy = +4V
        # The ZZ interaction is n_i n_j type (after constant shift)
        # We use the density-density form: V * n_i * n_j per bond
        # For our 2-particle state |s1, s2>:
        #   n_i n_j = 1 only if {i,j} = {s1,s2} and (i,j) is a bond
        
        pair = (min(s1, s2), max(s1, s2))
        if pair in adj:
            # Both particles on a bond → attractive interaction
            # Use negative V for attraction (bound states need V < 0 convention)
            # Actually, ZZ gives +V when both occupied, -V when one occupied
            # For nearest-neighbor density-density: H_int = V Σ n_i n_j
            # With V > 0 this is repulsive; V < 0 attractive
            # Physical convention: Ising ZZ is attractive for forming bound states
            H[k, k] += -V  # Attractive interaction when adjacent
        
        # ---- Hopping (off-diagonal) contribution ----
        # c†_a c_b + h.c. for each bond (a,b)
        # Acting on |s1, s2>: can hop s1 or s2 to an adjacent empty site
        
        if t > 0:
            for (a, b) in edges:
                # Try hopping particle at s1 from a to b
                if s1 == a and s2 != b:
                    new = tuple(sorted([b, s2]))
                    if new[0] != new[1]:  # no double occupancy
                        sign = fermion_sign(s1, s2, a, b)
                        idx_new = basis_idx.get(new)
                        if idx_new is not None:
                            H[idx_new, k] += t * sign
                            H[k, idx_new] += t * sign  # Hermitian
                
                if s1 == b and s2 != a:
                    new = tuple(sorted([a, s2]))
                    if new[0] != new[1]:
                        sign = fermion_sign(s1, s2, b, a)
                        idx_new = basis_idx.get(new)
                        if idx_new is not None:
                            H[idx_new, k] += t * sign
                            H[k, idx_new] += t * sign
                
                # Try hopping particle at s2
                if s2 == a and s1 != b:
                    new = tuple(sorted([s1, b]))
                    if new[0] != new[1]:
                        sign = fermion_sign(s1, s2, a, b)
                        idx_new = basis_idx.get(new)
                        if idx_new is not None:
                            H[idx_new, k] += t * sign
                            H[k, idx_new] += t * sign
                
                if s2 == b and s1 != a:
                    new = tuple(sorted([s1, a]))
                    if new[0] != new[1]:
                        sign = fermion_sign(s1, s2, b, a)
                        idx_new = basis_idx.get(new)
                        if idx_new is not None:
                            H[idx_new, k] += t * sign
                            H[k, idx_new] += t * sign
    
    # Remove double-counting from symmetric construction
    H = (H + H.conj().T) / 2
    
    return H, basis, basis_idx


def fermion_sign(s1, s2, site_from, site_to):
    """
    Compute the fermionic sign for hopping.
    For spinless fermions on a lattice, the sign depends on 
    the number of occupied sites between from and to.
    For our 2-particle states, this is simple.
    """
    # Simplified: for 2 particles, sign is +1 for nearest-neighbor hops
    # in most cases. Full JW string would need counting occupied sites
    # between from and to, but for NN hops on ordered basis this is ±1.
    # We use +1 here as a first approximation (hopping is real and positive).
    return 1.0


def build_u1_hamiltonian_2particle_v2(N, edges, alpha, J=1.0):
    """
    Cleaner construction: directly in the 2-particle basis.
    
    H = -t Σ_{<ij>} (c†_i c_j + h.c.) - V Σ_{<ij>} n_i n_j
    
    where t = α*J (hopping) and V = (1-α)*J (attractive interaction).
    
    The minus signs ensure:
      - Hopping lowers kinetic energy (particles want to delocalize)
      - Interaction is attractive (particles want to sit on adjacent sites)
    """
    basis = build_2particle_basis(N)
    dim = len(basis)
    basis_idx = {(i, j): k for k, (i, j) in enumerate(basis)}
    
    # Build adjacency set
    adj_set = set()
    adj_list = {i: [] for i in range(N)}
    for (a, b) in edges:
        adj_set.add((min(a, b), max(a, b)))
        adj_list[a].append(b)
        adj_list[b].append(a)
    
    t = alpha * J        # hopping amplitude
    V = (1 - alpha) * J  # interaction strength
    
    H = np.zeros((dim, dim), dtype=np.float64)
    
    for k, (s1, s2) in enumerate(basis):
        # --- Diagonal: interaction energy ---
        pair = (min(s1, s2), max(s1, s2))
        if pair in adj_set:
            H[k, k] -= V  # Attractive: lower energy when adjacent
        
        # --- Off-diagonal: hopping ---
        if t > 0:
            # Hop particle 1 (at s1) to each neighbor
            for nb in adj_list[s1]:
                if nb != s2:  # can't hop to occupied site
                    new = tuple(sorted([nb, s2]))
                    idx_new = basis_idx.get(new)
                    if idx_new is not None and idx_new != k:
                        H[k, idx_new] -= t
                        H[idx_new, k] -= t
            
            # Hop particle 2 (at s2) to each neighbor
            for nb in adj_list[s2]:
                if nb != s1:  # can't hop to occupied site
                    new = tuple(sorted([s1, nb]))
                    idx_new = basis_idx.get(new)
                    if idx_new is not None and idx_new != k:
                        H[k, idx_new] -= t
                        H[idx_new, k] -= t
    
    # Fix double counting
    H = H / 2
    
    return H, basis, basis_idx


# ============================================================
# BINDING DIAGNOSTICS
# ============================================================

def measure_binding_energy(H, basis, N, edges, n_states=10):
    """
    Compute the binding energy from the spectrum.
    
    Binding energy = E_threshold - E_ground
    where E_threshold is the bottom of the 2-particle continuum
    (two free particles at infinite separation).
    
    If binding energy > 0, bound states exist.
    """
    dim = H.shape[0]
    
    if dim <= 50:
        # Full diagonalization for small systems
        evals, evecs = np.linalg.eigh(H)
    else:
        # Sparse diagonalization
        k = min(n_states, dim - 2)
        try:
            H_sparse = csr_matrix(H)
            evals, evecs = eigsh(H_sparse, k=k, which='SA')
            sort_idx = np.argsort(evals)
            evals = evals[sort_idx]
            evecs = evecs[:, sort_idx]
        except Exception:
            evals, evecs = np.linalg.eigh(H)
    
    E_ground = evals[0]
    psi_ground = evecs[:, 0]
    
    # Estimate continuum threshold: energy of two well-separated particles
    # For large enough systems, this is approximately the energy of states
    # where particles are far apart
    # Simple estimate: look at the mean separation of each eigenstate
    adj_set = set()
    for (a, b) in edges:
        adj_set.add((min(a, b), max(a, b)))
    
    # Mean separation of ground state
    gs_sep = mean_separation(psi_ground, basis, N, edges)
    
    # Find the lowest energy state with large separation (continuum edge)
    max_sep = 0
    for (i, j) in basis:
        # Use lattice distance
        sep = abs(i - j)  # Simple 1D distance for estimation
        if sep > max_sep:
            max_sep = sep
    
    # Better approach: compute the single-particle spectrum,
    # then the 2-particle continuum bottom = 2 * E_single_ground
    # For the Hamiltonian H = -t Σ c†c - V Σ nn:
    # Single particle: H_1 = -t Σ_{<ij>} |i><j|
    # E_1_ground = -z*t where z = coordination number (for large L)
    
    z = len(edges) / N * 2  # average coordination number
    # Actually compute single-particle spectrum
    H_single = np.zeros((N, N), dtype=np.float64)
    for (a, b) in edges:
        H_single[a, b] -= alpha_from_t(H, basis, edges, N)
        H_single[b, a] -= alpha_from_t(H, basis, edges, N)
    
    E1_vals = np.linalg.eigvalsh(H_single)
    E_continuum = E1_vals[0] + E1_vals[1]  # Two lowest single-particle states
    
    binding_energy = E_continuum - E_ground
    
    return {
        'E_ground': E_ground,
        'E_continuum': E_continuum,
        'binding_energy': binding_energy,
        'bound': binding_energy > 1e-6,
        'gs_separation': gs_sep,
        'spectrum': evals[:min(10, len(evals))].tolist(),
    }


def alpha_from_t(H, basis, edges, N):
    """Extract the hopping parameter from H matrix."""
    # Look at a known hopping matrix element
    adj_list = {i: [] for i in range(N)}
    for (a, b) in edges:
        adj_list[a].append(b)
        adj_list[b].append(a)
    
    basis_idx = {(i, j): k for k, (i, j) in enumerate(basis)}
    
    # Find a hopping element
    for k, (s1, s2) in enumerate(basis):
        for nb in adj_list[s1]:
            if nb != s2:
                new = tuple(sorted([nb, s2]))
                idx_new = basis_idx.get(new)
                if idx_new is not None and idx_new != k:
                    val = abs(H[k, idx_new])
                    if val > 1e-10:
                        return val
    return 0.0


def measure_binding_simple(N, edges, alpha, J=1.0):
    """
    Simplified binding energy measurement.
    
    Constructs single-particle and two-particle Hamiltonians directly,
    compares E_2particle_ground vs 2*E_1particle_ground.
    
    Binding energy = 2*E_1 - E_2 > 0 means bound.
    """
    t = alpha * J
    V = (1 - alpha) * J
    
    # --- Single-particle Hamiltonian ---
    H1 = np.zeros((N, N), dtype=np.float64)
    for (a, b) in edges:
        H1[a, b] -= t
        H1[b, a] -= t
    
    E1_vals = np.linalg.eigvalsh(H1)
    E1_ground = E1_vals[0]
    E_two_free = 2 * E1_ground  # Two non-interacting particles
    
    # --- Two-particle Hamiltonian ---
    basis = build_2particle_basis(N)
    dim2 = len(basis)
    basis_idx = {(i, j): k for k, (i, j) in enumerate(basis)}
    
    adj_set = set()
    adj_list = {i: [] for i in range(N)}
    for (a, b) in edges:
        adj_set.add((min(a, b), max(a, b)))
        adj_list[a].append(b)
        adj_list[b].append(a)
    
    H2 = np.zeros((dim2, dim2), dtype=np.float64)
    
    for k, (s1, s2) in enumerate(basis):
        # Interaction: attractive when adjacent
        pair = (min(s1, s2), max(s1, s2))
        if pair in adj_set:
            H2[k, k] -= V
        
        # Hopping of particle 1
        if t > 0:
            for nb in adj_list[s1]:
                if nb != s2:
                    new = tuple(sorted([nb, s2]))
                    idx_new = basis_idx.get(new)
                    if idx_new is not None:
                        H2[k, idx_new] -= t
            
            # Hopping of particle 2
            for nb in adj_list[s2]:
                if nb != s1:
                    new = tuple(sorted([s1, nb]))
                    idx_new = basis_idx.get(new)
                    if idx_new is not None:
                        H2[k, idx_new] -= t
    
    # Symmetrize (fix any double counting)
    H2 = (H2 + H2.T) / 2
    
    # Diagonalize
    if dim2 <= 500:
        E2_vals = np.linalg.eigvalsh(H2)
    else:
        H2_sparse = csr_matrix(H2)
        E2_vals = eigsh(H2_sparse, k=min(20, dim2 - 2), which='SA',
                        return_eigenvectors=False)
        E2_vals = np.sort(E2_vals)
    
    E2_ground = E2_vals[0]
    
    # Binding energy: energy saved by binding vs two free particles
    binding_energy = E_two_free - E2_ground
    
    # Also get ground state wavefunction for separation analysis
    if dim2 <= 500:
        _, evecs = np.linalg.eigh(H2)
        psi_gs = evecs[:, 0]
    else:
        _, evecs = eigsh(csr_matrix(H2), k=1, which='SA')
        psi_gs = evecs[:, 0]
    
    # Mean separation in ground state
    gs_sep = mean_separation(psi_gs, basis, N, edges)
    
    # Participation ratio (inverse = how many basis states contribute)
    probs = np.abs(psi_gs)**2
    IPR = np.sum(probs**2)
    PR = 1.0 / IPR  # Number of effectively occupied basis states
    
    # Fraction of weight on adjacent pairs
    adj_weight = 0.0
    for k, (s1, s2) in enumerate(basis):
        pair = (min(s1, s2), max(s1, s2))
        if pair in adj_set:
            adj_weight += probs[k]
    
    # Energy gap to first excited state
    if len(E2_vals) > 1:
        gap = E2_vals[1] - E2_vals[0]
    else:
        gap = 0.0
    
    return {
        'alpha': alpha,
        'E_1particle': E1_ground,
        'E_two_free': E_two_free,
        'E_2particle': E2_ground,
        'binding_energy': binding_energy,
        'bound': binding_energy > 1e-6,
        'gs_separation': gs_sep,
        'participation_ratio': PR,
        'adjacent_weight': adj_weight,
        'gap': gap,
        'dim_2particle': dim2,
        'spectrum_low': E2_vals[:min(5, len(E2_vals))].tolist(),
    }


def mean_separation(psi, basis, N, edges):
    """
    Compute mean separation of two particles in state psi.
    Uses graph distance.
    """
    # Build adjacency for BFS
    adj = {i: [] for i in range(N)}
    for (a, b) in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    # Precompute all pairwise distances (BFS)
    dist = {}
    for i in range(N):
        d = {i: 0}
        queue = [i]
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in d:
                    d[nb] = d[node] + 1
                    queue.append(nb)
        dist[i] = d
    
    # Compute <d> = Σ |ψ_k|^2 * d(s1_k, s2_k)
    probs = np.abs(psi)**2
    mean_d = 0.0
    for k, (s1, s2) in enumerate(basis):
        mean_d += probs[k] * dist[s1].get(s2, N)
    
    return mean_d


# ============================================================
# DYNAMICAL BINDING TEST
# ============================================================

def test_dynamical_binding(N, edges, alpha, J=1.0, dt=0.05, t_max=10.0):
    """
    Prepare two adjacent particles, evolve, measure whether they stay bound.
    
    This tests dynamical stability: even if a bound state exists spectrally,
    does a "physical" initial condition (two adjacent particles) remain
    localized?
    """
    basis = build_2particle_basis(N)
    dim = len(basis)
    basis_idx = {(i, j): k for k, (i, j) in enumerate(basis)}
    
    adj_set = set()
    adj_list = {i: [] for i in range(N)}
    for (a, b) in edges:
        adj_set.add((min(a, b), max(a, b)))
        adj_list[a].append(b)
        adj_list[b].append(a)
    
    t_hop = alpha * J
    V = (1 - alpha) * J
    
    # Build H
    H = np.zeros((dim, dim), dtype=np.float64)
    for k, (s1, s2) in enumerate(basis):
        pair = (min(s1, s2), max(s1, s2))
        if pair in adj_set:
            H[k, k] -= V
        
        if t_hop > 0:
            for nb in adj_list[s1]:
                if nb != s2:
                    new = tuple(sorted([nb, s2]))
                    idx_new = basis_idx.get(new)
                    if idx_new is not None:
                        H[k, idx_new] -= t_hop
            
            for nb in adj_list[s2]:
                if nb != s1:
                    new = tuple(sorted([s1, nb]))
                    idx_new = basis_idx.get(new)
                    if idx_new is not None:
                        H[k, idx_new] -= t_hop
    
    H = (H + H.T) / 2
    
    # Initial state: two particles on an edge (site 0 and first neighbor)
    first_edge = edges[0]
    s1, s2 = min(first_edge), max(first_edge)
    init_idx = basis_idx.get((s1, s2))
    
    if init_idx is None:
        # Fallback
        init_idx = 0
    
    psi = np.zeros(dim, dtype=np.complex128)
    psi[init_idx] = 1.0
    
    # Time evolution
    U_dt = expm(-1j * dt * H)
    
    n_steps = int(t_max / dt)
    trajectory = {
        'times': [],
        'mean_separation': [],
        'adjacent_weight': [],
        'participation_ratio': [],
    }
    
    for step in range(n_steps + 1):
        t_val = step * dt
        
        probs = np.abs(psi)**2
        
        # Mean separation
        sep = 0.0
        adj_w = 0.0
        for k, (si, sj) in enumerate(basis):
            pair = (min(si, sj), max(si, sj))
            if pair in adj_set:
                adj_w += probs[k]
            # Use index distance as rough separation proxy
            sep += probs[k] * abs(si - sj)
        
        IPR = np.sum(probs**2)
        PR = 1.0 / IPR if IPR > 1e-30 else dim
        
        trajectory['times'].append(t_val)
        trajectory['mean_separation'].append(sep)
        trajectory['adjacent_weight'].append(adj_w)
        trajectory['participation_ratio'].append(PR)
        
        if step < n_steps:
            psi = U_dt @ psi
            psi /= np.linalg.norm(psi)
    
    # Binding assessment: does adjacent weight stay high?
    early_adj = np.mean(trajectory['adjacent_weight'][:5])
    late_adj = np.mean(trajectory['adjacent_weight'][-20:])
    
    # Does separation stay small?
    early_sep = np.mean(trajectory['mean_separation'][:5])
    late_sep = np.mean(trajectory['mean_separation'][-20:])
    
    dynamically_bound = late_adj > 0.3 and late_sep < 2 * early_sep
    
    return {
        'dynamically_bound': dynamically_bound,
        'early_adj_weight': early_adj,
        'late_adj_weight': late_adj,
        'early_separation': early_sep,
        'late_separation': late_sep,
        'adj_retention': late_adj / max(early_adj, 1e-10),
        'sep_growth': late_sep / max(early_sep, 1e-10),
        'trajectory': trajectory,
    }


# ============================================================
# MAIN EXPERIMENTS
# ============================================================

def sweep_hopping_ratio(dim, L, alpha_points=20, J=1.0, verbose=True):
    """
    Sweep the hopping fraction α and measure binding at each point.
    
    Parameters:
        dim: spatial dimension (1, 2, or 3)
        L: lattice side length
        alpha_points: number of α values to test
        J: coupling strength
    """
    N, edges, coord_map = build_lattice(dim, L)
    
    print(f"\n{'='*70}")
    print(f"HOPPING/ISING RATIO SWEEP")
    print(f"  Dimension: {dim}D, L={L}, N={N} sites")
    print(f"  Bonds: {len(edges)}")
    print(f"  2-particle basis: {N*(N-1)//2} states")
    print(f"  Alpha range: 0 to 1 ({alpha_points} points)")
    print(f"{'='*70}\n")
    
    alphas = np.linspace(0.0, 1.0, alpha_points + 1)
    results = []
    
    start = time.time()
    
    for idx, alpha in enumerate(alphas):
        elapsed = time.time() - start
        
        if verbose:
            print(f"  [{idx+1}/{len(alphas)}] α={alpha:.3f} "
                  f"(hop={alpha:.1%}, ising={1-alpha:.1%}) ... ", 
                  end="", flush=True)
        
        result = measure_binding_simple(N, edges, alpha, J)
        result['dim'] = dim
        result['L'] = L
        result['N'] = N
        results.append(result)
        
        if verbose:
            status = "BOUND" if result['bound'] else "unbound"
            print(f"E_bind={result['binding_energy']:+.6f}  "
                  f"<d>={result['gs_separation']:.3f}  "
                  f"adj_w={result['adjacent_weight']:.3f}  "
                  f"[{status}]  "
                  f"({elapsed:.1f}s)")
    
    total_time = time.time() - start
    
    # Find critical alpha: MAXIMUM hopping that still supports binding
    # (The physical question: how much kinetic energy can you add 
    #  before the potential can no longer hold particles together?)
    # 
    # Also exclude α=0 (pure Ising) from "dynamically bound" since
    # it's trivially frozen — PR=1 means only one basis state contributes.
    
    dynamically_bound = [r for r in results 
                         if r['bound'] and r['participation_ratio'] > 2.0]
    unbound_results = [r for r in results if not r['bound']]
    
    if dynamically_bound and unbound_results:
        alpha_max_bound = max(r['alpha'] for r in dynamically_bound)
        # Find first unbound alpha above the bound region
        first_unbound = min((r['alpha'] for r in unbound_results 
                            if r['alpha'] > alpha_max_bound), default=1.0)
        alpha_c = (alpha_max_bound + first_unbound) / 2  # Midpoint estimate
        alpha_c_lower = alpha_max_bound
        alpha_c_upper = first_unbound
    elif dynamically_bound:
        alpha_c = max(r['alpha'] for r in dynamically_bound)
        alpha_c_lower = alpha_c
        alpha_c_upper = 1.0
    else:
        alpha_c = None
        alpha_c_lower = 0.0
        alpha_c_upper = 0.0
    
    # Find optimal binding: maximum binding energy with nontrivial dynamics
    if dynamically_bound:
        best_bound = max(dynamically_bound, key=lambda r: r['binding_energy'])
        alpha_optimal = best_bound['alpha']
    else:
        alpha_optimal = None
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {dim}D Lattice (L={L}, N={N})")
    print(f"{'='*70}")
    
    print(f"\n{'α (hopping)':<15} {'E_bind':<12} {'<d>':<8} {'adj_w':<8} "
          f"{'PR':<8} {'gap':<10} {'Status':<10}")
    print("-" * 71)
    
    for r in results:
        if r['bound'] and r['participation_ratio'] <= 2.0:
            status = "  frozen"  # Trivially bound, no dynamics
        elif r['bound']:
            status = "★ BOUND"
        else:
            status = "  free"
        print(f"{r['alpha']:<15.3f} {r['binding_energy']:<+12.6f} "
              f"{r['gs_separation']:<8.3f} {r['adjacent_weight']:<8.3f} "
              f"{r['participation_ratio']:<8.2f} {r['gap']:<10.6f} {status}")
    
    print(f"\nCritical hopping fraction (max α with binding):")
    print(f"  α_c ∈ [{alpha_c_lower:.3f}, {alpha_c_upper:.3f}]")
    if alpha_c is not None:
        print(f"  → Max hopping for stable matter: α_c ≈ {alpha_c:.3f}")
        print(f"  → Force ratio at threshold: {alpha_c:.1%} kinetic / {1-alpha_c:.1%} potential")
        if alpha_optimal is not None:
            print(f"  → Strongest binding at α = {alpha_optimal:.3f} "
                  f"({alpha_optimal:.1%} kinetic / {1-alpha_optimal:.1%} potential)")
    else:
        print(f"  → No dynamically bound states found")
    
    print(f"\nTotal time: {total_time:.1f}s")
    
    return results, alpha_c


def compare_dimensions(L_by_dim=None, alpha_points=20, J=1.0, verbose=True):
    """
    Compare binding across 1D, 2D, 3D lattices.
    
    Tests the HSF prediction: 3D should be the first dimension
    supporting stable matter.
    """
    if L_by_dim is None:
        # Choose L so systems are tractable
        # 2-particle basis: N*(N-1)/2
        # 1D: L=12 → 66 states
        # 2D: L=4 → 16 sites → 120 states  
        # 3D: L=3 → 27 sites → 351 states
        L_by_dim = {1: 12, 2: 4, 3: 3}
    
    print(f"\n{'#'*70}")
    print(f"DIMENSIONAL COMPARISON: MATTER STABILITY")
    print(f"{'#'*70}")
    
    all_results = {}
    critical_alphas = {}
    
    for dim in sorted(L_by_dim.keys()):
        L = L_by_dim[dim]
        results, alpha_c = sweep_hopping_ratio(
            dim, L, alpha_points=alpha_points, J=J, verbose=verbose
        )
        all_results[dim] = results
        critical_alphas[dim] = alpha_c
    
    # Summary comparison
    print(f"\n{'#'*70}")
    print(f"DIMENSIONAL COMPARISON SUMMARY")
    print(f"{'#'*70}")
    
    print(f"\n{'Dim':<6} {'L':<4} {'N':<6} {'α_c':<10} {'Interpretation':<30}")
    print("-" * 56)
    
    for dim in sorted(L_by_dim.keys()):
        L = L_by_dim[dim]
        N = L**dim
        alpha_c = critical_alphas[dim]
        
        if alpha_c is not None:
            ising_frac = 1 - alpha_c
            interp = f"Bound for α≤{alpha_c:.3f} ({ising_frac:.0%} Ising)"
        else:
            interp = "No dynamical binding"
        
        ac_str = f"{alpha_c:.3f}" if alpha_c is not None else "N/A"
        print(f"{dim}D{'':<4} {L:<4} {N:<6} {ac_str:<10} {interp}")
    
    # Check HSF prediction
    print(f"\n--- HSF Matter Stability Prediction ---")
    for dim in sorted(L_by_dim.keys()):
        ac = critical_alphas[dim]
        results_dim = all_results[dim]
        # Find max binding with nontrivial dynamics (PR > 2)
        dynamic_bound = [r for r in results_dim 
                        if r['bound'] and r['participation_ratio'] > 2.0]
        if dynamic_bound:
            best = max(dynamic_bound, key=lambda r: r['binding_energy'])
            print(f"  {dim}D: Max dynamical binding E_bind={best['binding_energy']:.4f} "
                  f"at α={best['alpha']:.3f}, "
                  f"critical α_c≈{ac:.3f}" if ac else "")
        else:
            print(f"  {dim}D: No dynamical binding found")
    
    return all_results, critical_alphas


def fine_sweep_critical_region(dim, L, alpha_range, n_points=50, 
                                 J=1.0, verbose=True):
    """
    Fine-grained sweep around the critical hopping fraction.
    """
    N, edges, coord_map = build_lattice(dim, L)
    
    alpha_lo, alpha_hi = alpha_range
    alphas = np.linspace(alpha_lo, alpha_hi, n_points)
    
    print(f"\n{'='*70}")
    print(f"FINE SWEEP: {dim}D lattice, α ∈ [{alpha_lo:.3f}, {alpha_hi:.3f}]")
    print(f"  N={N} sites, {n_points} points")
    print(f"{'='*70}\n")
    
    results = []
    
    for idx, alpha in enumerate(alphas):
        result = measure_binding_simple(N, edges, alpha, J)
        result['dim'] = dim
        result['L'] = L
        results.append(result)
        
        if verbose and (idx % 10 == 0 or idx == len(alphas) - 1):
            status = "BOUND" if result['bound'] else "free"
            print(f"  α={alpha:.4f}: E_bind={result['binding_energy']:+.8f} [{status}]")
    
    # Find precise critical point: max alpha with dynamical binding
    dynamically_bound = [r for r in results 
                         if r['bound'] and r['participation_ratio'] > 2.0]
    if dynamically_bound:
        alpha_c = max(r['alpha'] for r in dynamically_bound)
        unbound_above = [r['alpha'] for r in results 
                        if not r['bound'] and r['alpha'] > alpha_c]
        if unbound_above:
            alpha_c = (alpha_c + min(unbound_above)) / 2
        print(f"\n  Critical α_c ≈ {alpha_c:.4f}")
        print(f"  Force ratio: {alpha_c:.2%} hopping / {1-alpha_c:.2%} Ising")
    else:
        alpha_c = None
        print(f"\n  No dynamically bound states in this range")
    
    return results, alpha_c


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(all_results, critical_alphas, output_dir='/home/claude'):
    """Generate publication-quality plots of binding results."""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # ---- Figure 1: Binding energy vs hopping fraction, all dimensions ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {1: '#E74C3C', 2: '#3498DB', 3: '#27AE60'}
    labels = {1: '1D', 2: '2D', 3: '3D'}
    
    # Panel 1: Binding energy
    ax = axes[0, 0]
    for dim in sorted(all_results.keys()):
        results = all_results[dim]
        alphas = [r['alpha'] for r in results]
        bindings = [r['binding_energy'] for r in results]
        ax.plot(alphas, bindings, 'o-', color=colors[dim], 
                label=f'{labels[dim]}', linewidth=2, markersize=5)
        
        ac = critical_alphas.get(dim)
        if ac is not None and ac <= 1.0:
            ax.axvline(x=ac, color=colors[dim], linestyle='--', alpha=0.5)
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Hopping Fraction α', fontsize=12)
    ax.set_ylabel('Binding Energy', fontsize=12)
    ax.set_title('Binding Energy vs Force Composition', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Ground state separation
    ax = axes[0, 1]
    for dim in sorted(all_results.keys()):
        results = all_results[dim]
        alphas = [r['alpha'] for r in results]
        seps = [r['gs_separation'] for r in results]
        ax.plot(alphas, seps, 'o-', color=colors[dim],
                label=f'{labels[dim]}', linewidth=2, markersize=5)
    
    ax.set_xlabel('Hopping Fraction α', fontsize=12)
    ax.set_ylabel('Mean Separation ⟨d⟩', fontsize=12)
    ax.set_title('Ground State Localization', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Adjacent weight
    ax = axes[1, 0]
    for dim in sorted(all_results.keys()):
        results = all_results[dim]
        alphas = [r['alpha'] for r in results]
        adj_ws = [r['adjacent_weight'] for r in results]
        ax.plot(alphas, adj_ws, 'o-', color=colors[dim],
                label=f'{labels[dim]}', linewidth=2, markersize=5)
    
    ax.set_xlabel('Hopping Fraction α', fontsize=12)
    ax.set_ylabel('Adjacent Pair Weight', fontsize=12)
    ax.set_title('Probability of Particles Being Adjacent', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Phase diagram summary
    ax = axes[1, 1]
    
    for dim in sorted(all_results.keys()):
        results = all_results[dim]
        alphas = [r['alpha'] for r in results]
        bound = [1 if r['bound'] else 0 for r in results]
        
        y_offset = dim * 0.3
        for a, b in zip(alphas, bound):
            marker = '★' if b else '○'
            c = colors[dim] if b else 'lightgray'
            ax.scatter([a], [y_offset], c=c, s=100 if b else 40,
                       marker='*' if b else 'o', zorder=5)
        
        ac = critical_alphas.get(dim)
        if ac is not None and ac <= 1.0:
            ax.axvline(x=ac, color=colors[dim], linestyle='--', alpha=0.5)
            ax.text(ac + 0.02, y_offset + 0.1, f'α_c={ac:.3f}',
                    color=colors[dim], fontsize=10)
    
    ax.set_yticks([d * 0.3 for d in sorted(all_results.keys())])
    ax.set_yticklabels([f'{labels[d]}' for d in sorted(all_results.keys())])
    ax.set_xlabel('Hopping Fraction α', fontsize=12)
    ax.set_title('Matter Stability Phase Diagram\n'
                 '(★ = bound, ○ = unbound)', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add regions
    ax.axvspan(0, 0.05, alpha=0.1, color='red', label='Frozen (pure Ising)')
    ax.axvspan(0.95, 1.0, alpha=0.1, color='blue', label='Free (pure hopping)')
    
    plt.suptitle('Matter Stability Constraint:\n'
                 'Minimum Hopping Fraction for Stable Binding',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/matter_stability_results.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/matter_stability_results.png")
    
    # ---- Figure 2: The emergence chain update ----
    fig, ax = plt.subplots(figsize=(16, 6))
    
    steps = [
        (0.05, 'Bare Hilbert\nSpace', '#2C3E50'),
        (0.2, 'Constraints\n(4 rules)', '#8E44AD'),
        (0.38, '3D Spatial\nStructure', '#2980B9'),
        (0.55, 'Fermionic\nMatter', '#27AE60'),
        (0.72, 'U(1) Gauge\n(XX+YY+ZZ)', '#E74C3C'),
        (0.88, 'Force Ratio\nα_c predicted', '#F39C12'),
    ]
    
    for x, text, color in steps:
        bbox = dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.15,
                    edgecolor=color, linewidth=2)
        ax.text(x, 0.5, text, ha='center', va='center', fontsize=11,
                bbox=bbox, fontweight='bold')
    
    # Arrows
    arrow_style = dict(arrowstyle='->', lw=2.5, color='#34495E')
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 0.06
        x2 = steps[i+1][0] - 0.06
        ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5), arrowprops=arrow_style)
    
    # Labels below
    mechanisms = [
        '', 'No-signal\nNo-forget\nNo-refold\nBandwidth',
        'Accessibility\ncollapse', 'Dimensional\nselection',
        'Bandwidth\nconstraint', 'Matter\nstability'
    ]
    for i, (x, _, _) in enumerate(steps):
        if mechanisms[i]:
            ax.text(x, 0.15, mechanisms[i], ha='center', va='center',
                    fontsize=8, color='gray', fontstyle='italic')
    
    # Add critical alpha annotation
    ac_3d = critical_alphas.get(3)
    if ac_3d is not None:
        ax.text(0.88, 0.25, f'α_c ≈ {ac_3d:.3f}\n'
                f'({ac_3d:.1%} hop / {1-ac_3d:.1%} Ising)',
                ha='center', fontsize=10, color='#E74C3C', fontweight='bold')
    
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 0.85)
    ax.axis('off')
    ax.set_title('HSF Emergence Chain: From Constraints to Force Ratios',
                 fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/emergence_chain_updated.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/emergence_chain_updated.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matter Stability Constraint: Hopping/Ising Ratio"
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 1D chain only')
    parser.add_argument('--standard', action='store_true',
                        help='Standard: sweep all dimensions')
    parser.add_argument('--dimensional', action='store_true',
                        help='Dimensional comparison (1D, 2D, 3D)')
    parser.add_argument('--fine', action='store_true',
                        help='Fine sweep around critical region')
    parser.add_argument('--full', action='store_true',
                        help='Full analysis with plots')
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--L', type=int, default=None)
    parser.add_argument('--points', type=int, default=20)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.quick:
        # Quick test on 1D chain
        results, alpha_c = sweep_hopping_ratio(
            dim=1, L=10, alpha_points=10, verbose=True
        )
    
    elif args.standard:
        # Standard sweep on specified dimension
        L = args.L or {1: 12, 2: 4, 3: 3}.get(args.dim, 3)
        results, alpha_c = sweep_hopping_ratio(
            dim=args.dim, L=L, alpha_points=args.points, verbose=True
        )
    
    elif args.dimensional:
        all_results, critical_alphas = compare_dimensions(
            alpha_points=args.points, verbose=True
        )
        try:
            plot_results(all_results, critical_alphas)
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    elif args.fine:
        # First do coarse sweep to find region
        L = args.L or {1: 12, 2: 4, 3: 3}.get(args.dim, 3)
        results, alpha_c_coarse = sweep_hopping_ratio(
            dim=args.dim, L=L, alpha_points=20, verbose=True
        )
        
        if alpha_c_coarse is not None:
            lo = max(0, alpha_c_coarse - 0.1)
            hi = min(1, alpha_c_coarse + 0.1)
            fine_results, alpha_c_fine = fine_sweep_critical_region(
                dim=args.dim, L=L, alpha_range=(lo, hi),
                n_points=50, verbose=True
            )
    
    elif args.full:
        # Complete analysis
        print("PHASE 1: Dimensional comparison")
        all_results, critical_alphas = compare_dimensions(
            alpha_points=args.points, verbose=True
        )
        
        # Fine sweep on 3D
        if critical_alphas.get(3) is not None:
            ac = critical_alphas[3]
            lo = max(0, ac - 0.15)
            hi = min(1, ac + 0.15)
            print(f"\nPHASE 2: Fine sweep on 3D (α ∈ [{lo:.3f}, {hi:.3f}])")
            fine_results, alpha_c_fine = fine_sweep_critical_region(
                dim=3, L=3, alpha_range=(lo, hi),
                n_points=50, verbose=True
            )
            if alpha_c_fine is not None:
                critical_alphas[3] = alpha_c_fine
        
        # Plots
        print("\nPHASE 3: Generating plots")
        try:
            plot_results(all_results, critical_alphas)
        except Exception as e:
            print(f"Plotting failed: {e}")
        
        # Final summary
        print(f"\n{'#'*70}")
        print(f"FINAL RESULTS: MATTER STABILITY CONSTRAINT")
        print(f"{'#'*70}")
        
        for dim in sorted(critical_alphas.keys()):
            ac = critical_alphas[dim]
            if ac is not None:
                print(f"\n  {dim}D: α_c = {ac:.4f}")
                print(f"       Hopping fraction: {ac:.2%}")
                print(f"       Ising fraction:   {1-ac:.2%}")
                print(f"       Kinetic/Potential: {ac/(1-ac):.4f}" if ac < 1 else "")
            else:
                print(f"\n  {dim}D: No bound states")
        
        print(f"\n  Framework prediction:")
        ac_3d = critical_alphas.get(3)
        if ac_3d is not None and ac_3d < 1:
            print(f"    The minimum hopping fraction for stable 3D matter is α_c ≈ {ac_3d:.4f}")
            print(f"    This sets the kinetic/potential ratio to {ac_3d/(1-ac_3d):.4f}")
            print(f"    (i.e., {ac_3d:.1%} kinetic exchange, {1-ac_3d:.1%} static potential)")
        else:
            print(f"    Could not determine critical ratio")
    
    else:
        # Default: dimensional comparison
        all_results, critical_alphas = compare_dimensions(
            alpha_points=args.points, verbose=True
        )
    
    total_time = time.time() - start_time
    print(f"\n\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    if args.output:
        def serialize(obj):
            # Convert NumPy scalar types (includes np.bool_, np.int64, np.float32, etc.)
            if isinstance(obj, np.generic):
                return obj.item()

            # NumPy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()

            # Containers
            if isinstance(obj, dict):
                return {str(k): serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]

            # Plain Python scalars
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj

            # Last resort: stringify anything else so the JSON export can't crash
            return str(obj)
        with open(args.output, 'w') as f:
            json.dump(serialize(locals().get('all_results', {})), f, indent=2)
        print(f"Saved to: {args.output}")