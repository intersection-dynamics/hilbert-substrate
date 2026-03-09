"""
Bound State Phase Diagram Sweep
===============================

Map the boundary where bound states form as a function of:
- N (system size): 8, 12, 16, 20, 24, 27
- Coordination number: 2, 3, 4, 5, 6, 7, 8 (using regular graphs)
- Interaction type: free, attractive, repulsive
- Interaction strength: 0.0 to 1.0

Goal: Find the ridge line near coord≈6 where bound states turn on.

Output: JSON data for plotting "Boundness vs (coord, N, interaction)"

Usage:
    python bound_state_sweep.py --quick          # Fast test
    python bound_state_sweep.py --small          # N up to 16
    python bound_state_sweep.py --full           # N up to 24
    python bound_state_sweep.py --comprehensive  # Full sweep
"""

import numpy as np
from scipy.linalg import expm
import networkx as nx
from typing import Dict, List, Tuple, Optional
import json
import time
import itertools

try:
    import cupy as cp
    GPU = True
    mempool = cp.get_default_memory_pool()
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(f"GPU: {gpu_name}")
except ImportError:
    import numpy as cp
    GPU = False
    print("CPU mode")


def clear_gpu():
    if GPU:
        mempool.free_all_blocks()


# ============================================================
# GRAPH GENERATION - Regular graphs with specified coordination
# ============================================================

def make_regular_graph(N: int, coord: int, seed: int = 0) -> Optional[nx.Graph]:
    """
    Create a random regular graph with N nodes and coordination number coord.
    
    For coord to work, N * coord must be even (handshaking lemma).
    Returns None if impossible.
    """
    if (N * coord) % 2 != 0:
        return None
    if coord >= N:
        return None
    
    try:
        G = nx.random_regular_graph(coord, N, seed=seed)
        return G
    except nx.NetworkXError:
        return None


def make_lattice_graph(dims: Tuple[int, ...], periodic: bool = True) -> nx.Graph:
    """Create a d-dimensional periodic lattice."""
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


def get_graph_for_config(N: int, coord: int, seed: int = 0) -> Optional[nx.Graph]:
    """
    Get a graph with N nodes and specified coordination.
    
    Prefers lattice structures when possible, falls back to random regular.
    Returns None if the exact coordination cannot be achieved.
    """
    G = None
    
    # Try lattice structures first (more physically meaningful)
    if coord == 2:
        # 1D chain - always works
        G = make_lattice_graph((N,))
    elif coord == 4:
        # 2D lattice - find factors where both dims >= 2
        for i in range(int(np.sqrt(N)) + 1, 1, -1):
            if N % i == 0:
                j = N // i
                if i >= 2 and j >= 2:
                    G = make_lattice_graph((i, j))
                    break
    elif coord == 6:
        # 3D lattice - need all dims >= 3 for true coord=6
        for i in range(3, int(N**(1/3)) + 2):
            if N % i == 0:
                rem = N // i
                for j in range(3, int(np.sqrt(rem)) + 2):
                    if rem % j == 0:
                        k = rem // j
                        if k >= 3:  # All dims must be >= 3
                            G = make_lattice_graph((i, j, k))
                            break
                if G is not None:
                    break
    
    # Check if lattice has correct coordination
    if G is not None:
        actual_coord = G.degree(0)
        if actual_coord == coord:
            return G
        # Lattice doesn't have right coordination, try random regular
        G = None
    
    # Fall back to random regular graph for any coord
    G = make_regular_graph(N, coord, seed)
    return G


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
# HAMILTONIAN GATES
# ============================================================

def make_gates(G: nx.Graph, N: int, dt: float, 
               interaction: str = 'free', strength: float = 0.5) -> Tuple[dict, dict]:
    """
    Build Trotter gates for various interaction types.
    
    interaction: 'free', 'attractive', 'repulsive', 'anisotropic'
    strength: interaction strength (0 to 1)
    """
    edge = {}
    
    for (i, j) in G.edges():
        if i > j:
            i, j = j, i
        
        # Base hopping term
        hop = np.kron(Sp, Sm)
        h_2site = -(hop + hop.conj().T)  # -t(c†c + h.c.) with t=1
        
        # Add interaction
        if interaction == 'attractive':
            n = (I + Z) / 2
            h_2site += -strength * np.kron(n, n)  # -V n_i n_j
        elif interaction == 'repulsive':
            n = (I + Z) / 2
            h_2site += strength * np.kron(n, n)   # +V n_i n_j
        elif interaction == 'anisotropic':
            # Anisotropic hopping - stronger in some directions
            # Use edge index to vary strength
            edge_idx = hash((i, j)) % 3
            aniso = 1.0 + strength * (edge_idx - 1) * 0.5  # 0.75 to 1.25
            h_2site = -aniso * (hop + hop.conj().T)
        
        edge[(i, j)] = cp.asarray(expm(-1j * dt * h_2site))
    
    onsite = {i: cp.asarray(I) for i in range(N)}
    return onsite, edge


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
# STATES AND MEASUREMENTS
# ============================================================

def create_two_excitation(N: int, site1: int, site2: int):
    """Create |1_site1, 1_site2⟩"""
    psi = cp.zeros(2**N, dtype=cp.complex128)
    idx = 2**(N - 1 - site1) + 2**(N - 1 - site2)
    psi[idx] = 1.0
    return psi


def find_adjacent_pair(G: nx.Graph) -> Tuple[int, int]:
    """Find a pair of adjacent sites."""
    edges = list(G.edges())
    if edges:
        return edges[0]
    return 0, 1


def measure_ipr(psi, N: int) -> float:
    """Inverse Participation Ratio - measures localization."""
    if GPU:
        probs = cp.abs(psi)**2
        return float(cp.sum(probs**2).get())
    else:
        probs = np.abs(psi)**2
        return np.sum(probs**2)


def measure_separation(psi, G: nx.Graph, N: int) -> dict:
    """Measure mean separation of two-excitation state."""
    if GPU:
        probs = cp.abs(psi)**2
        probs = probs.get()
    else:
        probs = np.abs(psi)**2
    
    # Precompute distances
    try:
        dist_dict = dict(nx.all_pairs_shortest_path_length(G))
    except:
        return {'mean': 1.0, 'std': 0.0}
    
    seps, weights = [], []
    for idx in range(2**N):
        if probs[idx] < 1e-12:
            continue
        binary = format(idx, f'0{N}b')
        positions = [N - 1 - i for i, b in enumerate(binary) if b == '1']
        if len(positions) == 2:
            sep = dist_dict[positions[0]].get(positions[1], N)
            seps.append(sep)
            weights.append(probs[idx])
    
    if not seps:
        return {'mean': 1.0, 'std': 0.0}
    
    seps = np.array(seps)
    weights = np.array(weights)
    weights /= weights.sum()
    
    mean_sep = np.sum(seps * weights)
    std_sep = np.sqrt(np.sum((seps - mean_sep)**2 * weights))
    
    return {'mean': mean_sep, 'std': std_sep}


# ============================================================
# SINGLE CONFIGURATION TEST
# ============================================================

def test_bound_state(
    N: int,
    coord: int,
    interaction: str = 'free',
    strength: float = 0.5,
    dt: float = 0.1,
    t_max: float = 5.0,
    n_snapshots: int = 11,
    seed: int = 0,
) -> Optional[dict]:
    """
    Test bound state formation for one configuration.
    
    Returns None if graph cannot be constructed.
    """
    G = get_graph_for_config(N, coord, seed)
    if G is None:
        return None
    
    actual_coord = G.degree(0) if G.number_of_nodes() > 0 else 0
    
    # Build gates
    onsite, edge = make_gates(G, N, dt, interaction, strength)
    
    # Initial state - adjacent pair
    site1, site2 = find_adjacent_pair(G)
    psi = create_two_excitation(N, site1, site2)
    
    # Evolve and measure
    n_steps = int(t_max / dt)
    steps_per_snap = max(1, n_steps // (n_snapshots - 1))
    
    iprs = [measure_ipr(psi, N)]
    
    for snap in range(1, n_snapshots):
        for _ in range(steps_per_snap):
            psi = trotter_step(psi, onsite, edge, N)
        iprs.append(measure_ipr(psi, N))
    
    # Final measurements
    final_sep = measure_separation(psi, G, N)
    
    # Summary metrics
    final_ipr = iprs[-1]
    avg_ipr = np.mean(iprs)
    min_ipr = np.min(iprs)
    
    # "Boundness" score: high final IPR + stable IPR + separation near 1
    ipr_stability = 1.0 - np.std(iprs)
    sep_penalty = max(0, final_sep['mean'] - 1.5)
    boundness = final_ipr * (0.5 + 0.5 * ipr_stability) * (1.0 / (1.0 + sep_penalty))
    
    del psi, onsite, edge
    clear_gpu()
    
    return {
        'N': N,
        'coord': actual_coord,
        'requested_coord': coord,
        'interaction': interaction,
        'strength': strength,
        'final_ipr': final_ipr,
        'avg_ipr': avg_ipr,
        'min_ipr': min_ipr,
        'ipr_stability': ipr_stability,
        'final_sep_mean': final_sep['mean'],
        'final_sep_std': final_sep['std'],
        'boundness': boundness,
        'seed': seed,
    }


# ============================================================
# PARAMETER SWEEP
# ============================================================

def run_sweep(
    N_values: List[int],
    coord_values: List[int],
    interactions: List[str],
    strengths: List[float],
    seeds: List[int] = [0],
    dt: float = 0.1,
    t_max: float = 5.0,
    output: str = None,
):
    """Run comprehensive parameter sweep."""
    
    print("=" * 70)
    print("BOUND STATE PHASE DIAGRAM SWEEP")
    print("=" * 70)
    print(f"N values: {N_values}")
    print(f"Coord values: {coord_values}")
    print(f"Interactions: {interactions}")
    print(f"Strengths: {strengths}")
    print(f"Seeds: {seeds}")
    print("=" * 70)
    
    # Generate all jobs
    jobs = list(itertools.product(N_values, coord_values, interactions, strengths, seeds))
    print(f"Total jobs: {len(jobs)}\n", flush=True)
    
    results = []
    skipped = 0
    start = time.time()
    
    for i, (N, coord, inter, strength, seed) in enumerate(jobs):
        # Progress
        elapsed = time.time() - start
        eta = (elapsed / (i + 1)) * (len(jobs) - i - 1) if i > 0 else 0
        print(f"[{i+1}/{len(jobs)}] N={N} coord={coord} {inter} s={strength:.1f} seed={seed}...",
              end=" ", flush=True)
        
        r = test_bound_state(N, coord, inter, strength, dt, t_max, seed=seed)
        
        if r is None:
            print("SKIP (no valid graph)")
            skipped += 1
            continue
        
        print(f"boundness={r['boundness']:.3f} ipr={r['final_ipr']:.3f} "
              f"[{elapsed/60:.1f}m, ETA {eta/60:.1f}m]")
        results.append(r)
    
    total_time = time.time() - start
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Boundness by (N, coord)")
    print("=" * 70)
    
    # Build pivot table
    pivot = {}
    for r in results:
        key = (r['N'], r['coord'])
        if key not in pivot:
            pivot[key] = []
        pivot[key].append(r['boundness'])
    
    # Print header
    coords_present = sorted(set(r['coord'] for r in results))
    Ns_present = sorted(set(r['N'] for r in results))
    
    print(f"{'N':<6}", end="")
    for c in coords_present:
        print(f"c={c:<6}", end="")
    print()
    print("-" * (6 + 8 * len(coords_present)))
    
    for N in Ns_present:
        print(f"{N:<6}", end="")
        for c in coords_present:
            key = (N, c)
            if key in pivot:
                val = np.mean(pivot[key])
                print(f"{val:<8.3f}", end="")
            else:
                print(f"{'--':<8}", end="")
        print()
    
    # Find ridge line (coord where boundness peaks for each N)
    print("\n" + "=" * 70)
    print("RIDGE LINE: Peak coordination for each N")
    print("=" * 70)
    for N in Ns_present:
        best_coord = None
        best_boundness = -1
        for c in coords_present:
            key = (N, c)
            if key in pivot:
                val = np.mean(pivot[key])
                if val > best_boundness:
                    best_boundness = val
                    best_coord = c
        if best_coord is not None:
            print(f"N={N}: peak at coord={best_coord} (boundness={best_boundness:.3f})")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Completed: {len(results)}, Skipped: {skipped}")
    
    # Save results
    if output:
        out_data = {
            'results': results,
            'config': {
                'N_values': N_values,
                'coord_values': coord_values,
                'interactions': interactions,
                'strengths': strengths,
                'seeds': seeds,
                't_max': t_max,
                'dt': dt,
                'total_time_minutes': total_time / 60,
            },
            'summary': {
                'pivot_means': {f"{k[0]}_{k[1]}": np.mean(v) for k, v in pivot.items()},
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
    
    p = argparse.ArgumentParser(description="Bound State Phase Diagram Sweep")
    p.add_argument('--quick', action='store_true', help="Quick test (N≤12)")
    p.add_argument('--small', action='store_true', help="Small sweep (N≤16)")
    p.add_argument('--medium', action='store_true', help="Medium sweep (N≤20)")
    p.add_argument('--full', action='store_true', help="Full sweep (N≤24)")
    p.add_argument('--comprehensive', action='store_true', help="Comprehensive (N≤27)")
    p.add_argument('--tmax', type=float, default=5.0)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--seeds', type=str, default="0")
    p.add_argument('--output', type=str, default=None)
    
    args = p.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    if args.quick:
        N_values = [8, 12]
        coord_values = [2, 3, 4, 5, 6]
        interactions = ['free', 'attractive']
        strengths = [0.5]
    elif args.small:
        N_values = [8, 12, 16]
        coord_values = [2, 3, 4, 5, 6, 7, 8]
        interactions = ['free', 'attractive', 'repulsive']
        strengths = [0.0, 0.5, 1.0]
    elif args.medium:
        N_values = [8, 12, 16, 20]
        coord_values = [2, 3, 4, 5, 6, 7, 8]
        interactions = ['free', 'attractive', 'repulsive']
        strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    elif args.full:
        N_values = [8, 12, 16, 20, 24]
        coord_values = [2, 3, 4, 5, 6, 7, 8]
        interactions = ['free', 'attractive', 'repulsive']
        strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    elif args.comprehensive:
        N_values = [8, 12, 16, 20, 24, 27]
        coord_values = [2, 3, 4, 5, 6, 7, 8]
        interactions = ['free', 'attractive', 'repulsive', 'anisotropic']
        strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        # Default: quick test
        N_values = [8, 12]
        coord_values = [2, 4, 6]
        interactions = ['free', 'attractive']
        strengths = [0.5]
    
    # Memory warning
    max_N = max(N_values)
    state_gb = (2**max_N * 16) / 1e9
    print(f"\nState vector: {state_gb:.2f} GB for N={max_N}")
    if GPU:
        free = cp.cuda.runtime.memGetInfo()[0] / 1e9
        print(f"GPU free: {free:.1f} GB")
        if state_gb * 3 > free:
            print("WARNING: May exceed VRAM for largest N!")
    print()
    
    run_sweep(
        N_values=N_values,
        coord_values=coord_values,
        interactions=interactions,
        strengths=strengths,
        seeds=seeds,
        dt=args.dt,
        t_max=args.tmax,
        output=args.output,
    )