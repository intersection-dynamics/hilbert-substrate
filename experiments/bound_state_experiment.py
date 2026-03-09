"""
Bound State Formation Experiment
================================

The key question: In which dimension do localized excitations 
stay BOUND rather than dispersing?

Physical motivation:
- Bandwidth ≈ 6 means particles can interact with ~6 neighbors
- 1D only uses 2 of those slots - fermions work, but no stable orbits
- 3D uses all 6 - first dimension where bound states can form
- Once matter exists, no-refolding prevents further evolution

What we test:
1. Initialize two excitations at nearby sites
2. Evolve under local (JW fermionic) Hamiltonian  
3. Measure: do they STAY BOUND or DISPERSE?

Metrics:
- Mean separation: grows (dispersing) vs oscillates (bound)
- Localization (IPR): high (bound) vs low (spread out)
- Return probability: do they come back together?

Usage:
    python bound_state_experiment.py --test
    python bound_state_experiment.py --full --output results.json
"""

import numpy as np
from scipy.linalg import expm
import networkx as nx
from typing import Dict, List, Tuple
import json
import time

try:
    import cupy as cp
    GPU = True
    mempool = cp.get_default_memory_pool()
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    import numpy as cp
    GPU = False
    print("CPU mode")


def clear_gpu():
    if GPU:
        mempool.free_all_blocks()


# ============================================================
# LATTICE GRAPHS
# ============================================================

def make_lattice(dims, periodic=True):
    d = len(dims)
    N = int(np.prod(dims))
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    def to_coords(idx):
        c = []
        for dim in reversed(dims):
            c.append(idx % dim)
            idx //= dim
        return tuple(reversed(c))
    
    def to_idx(coords):
        idx = 0
        for i, c in enumerate(coords):
            idx = idx * dims[i] + c
        return idx
    
    # Store coordinate lookup
    G.graph['to_coords'] = to_coords
    G.graph['to_idx'] = to_idx
    G.graph['dims'] = dims
    
    for node in range(N):
        coords = list(to_coords(node))
        for axis in range(d):
            nc = coords.copy()
            nc[axis] = (coords[axis] + 1) % dims[axis]
            G.add_edge(node, to_idx(nc))
    
    return G


def lattice_distance(G, i, j):
    """Compute lattice distance (shortest path) between sites."""
    return nx.shortest_path_length(G, i, j)


def euclidean_distance(G, i, j):
    """Compute Euclidean distance in lattice coordinates."""
    to_coords = G.graph['to_coords']
    dims = G.graph['dims']
    c1 = np.array(to_coords(i))
    c2 = np.array(to_coords(j))
    
    # Handle periodic boundaries - find minimum image
    delta = c1 - c2
    for ax, d in enumerate(dims):
        if abs(delta[ax]) > d / 2:
            delta[ax] = d - abs(delta[ax])
    
    return np.sqrt(np.sum(delta**2))


# ============================================================
# PAULI OPERATORS
# ============================================================

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
Sp = (X + 1j*Y) / 2
Sm = (X - 1j*Y) / 2


# ============================================================
# JORDAN-WIGNER GATES (fermionic hopping)
# ============================================================

def make_jw_gates(G, N, dt, t_hop=1.0, U_onsite=0.0):
    """
    Jordan-Wigner fermionic hopping with optional on-site interaction.
    
    H = -t Σ (c†_i c_j + h.c.) + U Σ n_i n_j  (for neighbors)
    
    The U term creates effective attraction/repulsion.
    """
    edge = {}
    for (i, j) in G.edges():
        if i > j:
            i, j = j, i
        
        # Hopping: -t(σ+⊗σ- + σ-⊗σ+)
        hop = np.kron(Sp, Sm)
        h_2site = -t_hop * (hop + hop.conj().T)
        
        # Density-density interaction: U * n_i * n_j
        # n = (1 + Z) / 2, so n⊗n = (1+Z)⊗(1+Z)/4
        if U_onsite != 0:
            n = (I + Z) / 2
            h_2site += U_onsite * np.kron(n, n)
        
        edge[(i, j)] = cp.asarray(expm(-1j * dt * h_2site))
    
    # Identity on-site
    onsite = {i: cp.asarray(I) for i in range(N)}
    
    return onsite, edge


def make_attractive_gates(G, N, dt, t_hop=1.0, V_attract=-0.5):
    """
    Fermionic hopping with ATTRACTIVE interaction.
    This should favor bound state formation.
    """
    return make_jw_gates(G, N, dt, t_hop, U_onsite=V_attract)


# ============================================================
# TROTTER EVOLUTION
# ============================================================

def apply_1q(psi, q, gate, N):
    l, r = 2**q, 2**(N - q - 1)
    psi = psi.reshape(l, 2, r)
    psi = cp.einsum('ab,lbr->lar', gate, psi)
    return psi.reshape(-1)


def apply_2q(psi, q1, q2, gate, N):
    if q1 > q2:
        q1, q2 = q2, q1
        gate = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    
    l = 2**q1
    m = 2**(q2 - q1 - 1)
    r = 2**(N - q2 - 1)
    
    psi = psi.reshape(l, 2, m, 2, r)
    gate = gate.reshape(2, 2, 2, 2)
    psi = cp.einsum('abcd,lcmdr->lamdr', gate, psi)
    return psi.reshape(-1)


def trotter_step(psi, onsite, edge, N):
    for q, g in onsite.items():
        psi = apply_1q(psi, q, g, N)
    for (q1, q2), g in edge.items():
        psi = apply_2q(psi, q1, q2, g, N)
    return psi / cp.linalg.norm(psi)


# ============================================================
# TWO-PARTICLE STATES
# ============================================================

def create_two_excitation_state(N, site1, site2):
    """
    Create |1_site1, 1_site2⟩ - two excitations at given sites.
    
    In computational basis: |...1...1...⟩
    """
    psi = cp.zeros(2**N, dtype=cp.complex128)
    
    # Index where both site1 and site2 have spin up (|1⟩)
    idx = 2**(N - 1 - site1) + 2**(N - 1 - site2)
    psi[idx] = 1.0
    
    return psi


def create_nearby_pair(G, init_separation=1):
    """Find a pair of sites at given separation."""
    N = G.number_of_nodes()
    
    # Start from center-ish
    start = N // 3
    
    for j in range(N):
        if lattice_distance(G, start, j) == init_separation:
            return start, j
    
    # Fallback to any neighbor
    return start, list(G.neighbors(start))[0]


# ============================================================
# BOUND STATE METRICS
# ============================================================

def measure_separation_distribution(psi, G, N):
    """
    Compute the probability distribution over pair separations.
    
    For each basis state |i,j⟩ with two excitations at sites i,j,
    weight by |amplitude|² and bin by separation.
    """
    # Get probabilities
    if GPU:
        probs = cp.abs(psi)**2
        probs = probs.get()
    else:
        probs = np.abs(psi)**2
    
    # Find two-excitation basis states
    separations = []
    weights = []
    
    for idx in range(2**N):
        if probs[idx] < 1e-12:
            continue
        
        # Count excitations and their positions
        binary = format(idx, f'0{N}b')
        positions = [N - 1 - i for i, b in enumerate(binary) if b == '1']
        
        if len(positions) == 2:
            sep = lattice_distance(G, positions[0], positions[1])
            separations.append(sep)
            weights.append(probs[idx])
    
    if not separations:
        return {'mean': 0, 'std': 0, 'max_prob_sep': 0}
    
    separations = np.array(separations)
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    mean_sep = np.sum(separations * weights)
    var_sep = np.sum((separations - mean_sep)**2 * weights)
    std_sep = np.sqrt(var_sep)
    
    # Most probable separation
    max_idx = np.argmax(weights)
    max_prob_sep = separations[max_idx]
    
    return {
        'mean': mean_sep,
        'std': std_sep,
        'max_prob_sep': max_prob_sep,
        'total_weight': np.sum(weights),  # Should be ~1 if particle number conserved
    }


def measure_localization(psi, N):
    """
    Inverse Participation Ratio (IPR).
    
    IPR = Σ |ψ_i|^4
    
    High IPR = localized (few basis states)
    Low IPR = delocalized (spread over many states)
    
    For perfectly localized: IPR = 1
    For maximally spread: IPR = 1/dim
    """
    if GPU:
        probs = cp.abs(psi)**2
        ipr = float(cp.sum(probs**2).get())
    else:
        probs = np.abs(psi)**2
        ipr = np.sum(probs**2)
    
    return ipr


def measure_return_probability(psi, psi_initial, N):
    """
    Overlap with initial state: |⟨ψ_0|ψ(t)⟩|²
    
    High = particles return to initial configuration (bound orbit)
    Low = dispersed away
    """
    if GPU:
        overlap = cp.abs(cp.vdot(psi_initial, psi))**2
        return float(overlap.get())
    else:
        return float(np.abs(np.vdot(psi_initial, psi))**2)


def measure_binding(psi, G, N, init_sep):
    """
    Combined binding metric.
    
    A state is "bound" if:
    1. Mean separation doesn't grow much beyond initial
    2. Separation std is small (not spreading)
    3. Some probability of returning to initial configuration
    """
    sep = measure_separation_distribution(psi, G, N)
    ipr = measure_localization(psi, N)
    
    # Binding score: penalize if mean separation grew
    sep_growth = sep['mean'] / max(init_sep, 1)
    binding = 1.0 / (1.0 + sep_growth) + ipr * 10  # Weight IPR
    
    return {
        'mean_sep': sep['mean'],
        'std_sep': sep['std'],
        'ipr': ipr,
        'binding_score': binding,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_bound_state_test(
    dims: Tuple[int, ...],
    init_separation: int = 1,
    dt: float = 0.1,
    t_max: float = 5.0,
    n_snapshots: int = 21,
    interaction: str = 'attractive',  # 'free', 'attractive', 'repulsive'
):
    """
    Test bound state formation for one lattice configuration.
    """
    N = int(np.prod(dims))
    G = make_lattice(dims)
    coord = G.degree(0)
    
    # Create gates
    if interaction == 'free':
        onsite, edge = make_jw_gates(G, N, dt, t_hop=1.0, U_onsite=0.0)
    elif interaction == 'attractive':
        onsite, edge = make_attractive_gates(G, N, dt, t_hop=1.0, V_attract=-0.5)
    elif interaction == 'repulsive':
        onsite, edge = make_jw_gates(G, N, dt, t_hop=1.0, U_onsite=0.5)
    else:
        onsite, edge = make_jw_gates(G, N, dt, t_hop=1.0, U_onsite=0.0)
    
    # Find initial pair
    site1, site2 = create_nearby_pair(G, init_separation)
    actual_init_sep = lattice_distance(G, site1, site2)
    
    # Create initial state
    psi = create_two_excitation_state(N, site1, site2)
    psi_init = psi.copy()
    
    # Time evolution with snapshots
    times = np.linspace(0, t_max, n_snapshots)
    n_steps = int(t_max / dt)
    steps_per_snap = max(1, n_steps // (n_snapshots - 1))
    
    trajectory = []
    
    # Initial measurement
    binding = measure_binding(psi, G, N, actual_init_sep)
    ret_prob = measure_return_probability(psi, psi_init, N)
    trajectory.append({
        't': 0.0,
        'mean_sep': binding['mean_sep'],
        'std_sep': binding['std_sep'],
        'ipr': binding['ipr'],
        'return_prob': ret_prob,
    })
    
    # Evolve
    for snap in range(1, n_snapshots):
        for _ in range(steps_per_snap):
            psi = trotter_step(psi, onsite, edge, N)
        
        binding = measure_binding(psi, G, N, actual_init_sep)
        ret_prob = measure_return_probability(psi, psi_init, N)
        
        trajectory.append({
            't': times[snap],
            'mean_sep': binding['mean_sep'],
            'std_sep': binding['std_sep'],
            'ipr': binding['ipr'],
            'return_prob': ret_prob,
        })
    
    # Summary statistics
    mean_seps = [t['mean_sep'] for t in trajectory]
    iprs = [t['ipr'] for t in trajectory]
    ret_probs = [t['return_prob'] for t in trajectory]
    
    # Key metrics for bound state:
    # 1. Did separation stay bounded? (mean of mean_sep relative to init)
    avg_sep = np.mean(mean_seps)
    max_sep = np.max(mean_seps)
    sep_ratio = avg_sep / max(actual_init_sep, 1)
    
    # 2. Did IPR stay high? (localization preserved)
    avg_ipr = np.mean(iprs)
    final_ipr = iprs[-1]
    
    # 3. Any recurrence? (return probability has peaks)
    max_return = np.max(ret_probs[1:]) if len(ret_probs) > 1 else 0
    avg_return = np.mean(ret_probs)
    
    # Combined bound state score
    # High score = bound, Low score = dispersed
    bound_score = (1.0 / sep_ratio) + avg_ipr * 100 + max_return * 10
    
    del psi, psi_init, onsite, edge
    clear_gpu()
    
    return {
        'dims': list(dims),
        'dimension': len(dims),
        'N': N,
        'coord': coord,
        'init_separation': actual_init_sep,
        'interaction': interaction,
        
        # Separation metrics
        'avg_separation': avg_sep,
        'max_separation': max_sep,
        'sep_ratio': sep_ratio,
        
        # Localization metrics
        'avg_ipr': avg_ipr,
        'final_ipr': final_ipr,
        
        # Recurrence metrics
        'max_return_prob': max_return,
        'avg_return_prob': avg_return,
        
        # Combined score
        'bound_score': bound_score,
        
        # Full trajectory for plotting
        'trajectory': trajectory,
    }


def run_experiment(
    configs: List[Tuple[int, ...]],
    interactions: List[str] = ['free', 'attractive'],
    init_sep: int = 1,
    dt: float = 0.1,
    t_max: float = 5.0,
    output: str = None,
):
    """Run bound state experiment across configurations."""
    
    print("=" * 70)
    print("BOUND STATE FORMATION EXPERIMENT")
    print("=" * 70)
    print(f"Configs: {configs}")
    print(f"Interactions: {interactions}")
    print(f"Initial separation: {init_sep}")
    print(f"t_max: {t_max}")
    print("=" * 70)
    print(flush=True)
    
    jobs = [(dims, inter) for dims in configs for inter in interactions]
    print(f"Total jobs: {len(jobs)}\n")
    
    results = []
    start = time.time()
    
    for i, (dims, inter) in enumerate(jobs):
        print(f"[{i+1}/{len(jobs)}] {len(dims)}D {dims} {inter}...", end=" ", flush=True)
        r = run_bound_state_test(dims, init_sep, dt, t_max, interaction=inter)
        elapsed = time.time() - start
        print(f"bound={r['bound_score']:.2f} sep_ratio={r['sep_ratio']:.2f} "
              f"ipr={r['avg_ipr']:.4f} [total: {elapsed/60:.1f}min]")
        results.append(r)
    
    total_time = time.time() - start
    
    # Summary table
    print("\n" + "=" * 70)
    print("BOUND STATE SUMMARY")
    print("=" * 70)
    print(f"{'Config':<15} {'Inter':<12} {'SepRatio':<10} {'AvgIPR':<10} "
          f"{'MaxReturn':<10} {'BoundScore':<10}")
    print("-" * 70)
    
    for inter in interactions:
        for r in sorted([x for x in results if x['interaction'] == inter],
                       key=lambda x: x['dimension']):
            cfg = f"{r['dimension']}D {tuple(r['dims'])}"
            print(f"{cfg:<15} {r['interaction']:<12} {r['sep_ratio']:<10.3f} "
                  f"{r['avg_ipr']:<10.4f} {r['max_return_prob']:<10.4f} "
                  f"{r['bound_score']:<10.2f}")
        print()
    
    # Which dimension has best bound states?
    print("=" * 70)
    print("WINNER BY INTERACTION TYPE")
    print("=" * 70)
    for inter in interactions:
        sub = [r for r in results if r['interaction'] == inter]
        if sub:
            winner = max(sub, key=lambda x: x['bound_score'])
            print(f"{inter}: {winner['dimension']}D {tuple(winner['dims'])} "
                  f"(score={winner['bound_score']:.2f})")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    # Save
    if output:
        # Remove trajectory for cleaner output (can be large)
        results_clean = []
        for r in results:
            r_clean = {k: v for k, v in r.items() if k != 'trajectory'}
            r_clean['trajectory_summary'] = {
                'times': [t['t'] for t in r['trajectory']],
                'mean_seps': [t['mean_sep'] for t in r['trajectory']],
                'iprs': [t['ipr'] for t in r['trajectory']],
                'return_probs': [t['return_prob'] for t in r['trajectory']],
            }
            results_clean.append(r_clean)
        
        out_data = {
            'results': results_clean,
            'config': {
                'configs': [list(c) for c in configs],
                'interactions': interactions,
                'init_sep': init_sep,
                't_max': t_max,
                'dt': dt,
                'total_time_minutes': total_time / 60,
            }
        }
        with open(output, 'w') as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved to: {output}")
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser(description="Bound State Formation Experiment")
    p.add_argument('--test', action='store_true', help="N=8 quick test")
    p.add_argument('--small', action='store_true', help="N=12")
    p.add_argument('--medium', action='store_true', help="N=16")
    p.add_argument('--full', action='store_true', help="N=24")
    p.add_argument('--true3d', action='store_true', help="N=27 with true coord=6")
    p.add_argument('--interactions', type=str, default='free,attractive',
                   help="Comma-separated: free,attractive,repulsive")
    p.add_argument('--init-sep', type=int, default=1)
    p.add_argument('--tmax', type=float, default=5.0)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--output', type=str, default=None)
    
    args = p.parse_args()
    
    interactions = args.interactions.split(',')
    
    if args.test:
        configs = [(8,), (4, 2), (2, 2, 2)]
    elif args.small:
        configs = [(12,), (4, 3), (2, 2, 3)]
    elif args.medium:
        configs = [(16,), (4, 4), (2, 2, 4)]
    elif args.full:
        configs = [(24,), (6, 4), (2, 3, 4)]
    elif args.true3d:
        # N=27 with true coord=6 for 3D
        configs = [(27,), (9, 3), (3, 3, 3)]
    else:
        configs = [(8,), (4, 2), (2, 2, 2)]
    
    # Memory estimate
    max_N = max(np.prod(c) for c in configs)
    state_gb = (2**max_N * 16) / 1e9
    print(f"\nState vector: {state_gb:.2f} GB for N={max_N}")
    if GPU:
        free = cp.cuda.runtime.memGetInfo()[0] / 1e9
        print(f"GPU free: {free:.1f} GB")
    
    run_experiment(
        configs=configs,
        interactions=interactions,
        init_sep=args.init_sep,
        dt=args.dt,
        t_max=args.tmax,
        output=args.output,
    )