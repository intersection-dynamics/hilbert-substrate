"""
Fermionic Coherence Experiment - CuPy GPU Accelerated
======================================================

GPU-accelerated version using CuPy for NVIDIA GPUs.
Expect 10-50x speedup over CPU numpy.

Requirements:
    pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version

Usage:
    python fermionic_coherence_gpu.py --test
    python fermionic_coherence_gpu.py --true3d --multi --n-pairs 5 --output results.json

Author: Ben Bray
"""

import numpy as np
from scipy.linalg import expm  # Keep on CPU - small matrices
import networkx as nx
from typing import Dict, List, Tuple
import json
import time

# Try to import CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    mempool = cp.get_default_memory_pool()
    print(f"CuPy available. GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9:.1f} GB")
except ImportError:
    print("CuPy not available, falling back to NumPy (CPU)")
    import numpy as cp
    GPU_AVAILABLE = False


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if GPU_AVAILABLE:
        mempool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()


# ============================================================
# LATTICE GRAPHS (CPU - graph operations are fast)
# ============================================================

def generate_lattice_graph(dims: Tuple[int, ...], periodic: bool = True) -> nx.Graph:
    d = len(dims)
    N = int(np.prod(dims))
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    def to_coords(idx):
        coords = []
        for dim in reversed(dims):
            coords.append(idx % dim)
            idx //= dim
        return tuple(reversed(coords))
    
    def to_idx(coords):
        idx = 0
        for i, c in enumerate(coords):
            idx = idx * dims[i] + c
        return idx
    
    for node in range(N):
        coords = list(to_coords(node))
        for axis in range(d):
            new_coords = coords.copy()
            if periodic:
                new_coords[axis] = (coords[axis] + 1) % dims[axis]
                G.add_edge(node, to_idx(new_coords))
    
    return G


def find_disjoint_pairs(G: nx.Graph, distance: int = 2, n_pairs: int = 3) -> List[Tuple[int, int]]:
    """Find vertex-disjoint pairs at specified distance."""
    N = G.number_of_nodes()
    distances = dict(nx.all_pairs_shortest_path_length(G))
    
    candidates = [(i, j) for i in range(N) for j in range(i+1, N)
                  if distances[i][j] == distance]
    
    if not candidates:
        candidates = list(G.edges())
    
    disjoint = []
    used_vertices = set()
    
    for (i, j) in candidates:
        if i not in used_vertices and j not in used_vertices:
            disjoint.append((i, j))
            used_vertices.add(i)
            used_vertices.add(j)
            if len(disjoint) >= n_pairs:
                break
    
    return disjoint


# ============================================================
# TROTTER GATES (CPU - small matrix exponentials)
# ============================================================

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = [I, X, Y, Z]


def build_trotter_gates(G: nx.Graph, N: int, dt: float,
                        h_std: float = 0.25, J_std: float = 0.8, seed: int = 0):
    """Build Trotter gates on CPU, then transfer to GPU."""
    np.random.seed(seed)
    
    # Build on CPU
    onsite_gates_cpu = {}
    for i in range(N):
        h_op = np.zeros((2, 2), dtype=np.complex128)
        for p in range(1, 4):
            h_op += np.random.normal(0, h_std) * PAULIS[p]
        onsite_gates_cpu[i] = expm(-1j * dt * h_op)
    
    edge_gates_cpu = {}
    for (i, j) in G.edges():
        h_edge = np.zeros((4, 4), dtype=np.complex128)
        for p in range(1, 4):
            for q in range(1, 4):
                h_edge += np.random.normal(0, J_std) * np.kron(PAULIS[p], PAULIS[q])
        edge_gates_cpu[(i, j)] = expm(-1j * dt * h_edge)
    
    # Transfer to GPU
    onsite_gates = {i: cp.asarray(g) for i, g in onsite_gates_cpu.items()}
    edge_gates = {k: cp.asarray(g) for k, g in edge_gates_cpu.items()}
    
    return onsite_gates, edge_gates


# ============================================================
# GPU-ACCELERATED TROTTER EVOLUTION
# ============================================================

def apply_single_qubit_gate(psi, qubit, gate, N):
    """Apply single-qubit gate using GPU einsum."""
    left_dim = 2**qubit
    right_dim = 2**(N - qubit - 1)
    psi_reshaped = psi.reshape(left_dim, 2, right_dim)
    result = cp.einsum('ab,lbr->lar', gate, psi_reshaped)
    return result.reshape(-1)


def apply_two_qubit_gate(psi, q1, q2, gate, N):
    """Apply two-qubit gate using GPU einsum."""
    if q1 > q2:
        q1, q2 = q2, q1
        gate = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    
    left_dim = 2**q1
    mid_dim = 2**(q2 - q1 - 1)
    right_dim = 2**(N - q2 - 1)
    
    psi_reshaped = psi.reshape(left_dim, 2, mid_dim, 2, right_dim)
    gate_reshaped = gate.reshape(2, 2, 2, 2)
    result = cp.einsum('abcd,lcmdr->lamdr', gate_reshaped, psi_reshaped)
    return result.reshape(-1)


def trotter_step(psi, onsite, edge, N):
    """One Trotter step - fully on GPU."""
    for qubit, gate in onsite.items():
        psi = apply_single_qubit_gate(psi, qubit, gate, N)
    for (q1, q2), gate in edge.items():
        psi = apply_two_qubit_gate(psi, q1, q2, gate, N)
    psi = psi / cp.linalg.norm(psi)
    return psi


# ============================================================
# FERMIONIC STATES AND MEASUREMENTS (GPU)
# ============================================================

def create_singlet_state(N: int, site1: int, site2: int):
    """Create singlet state on GPU."""
    psi = cp.zeros(2**N, dtype=cp.complex128)
    idx1 = 2**(N - 1 - site1)
    idx2 = 2**(N - 1 - site2)
    psi[idx1] = 1.0 / cp.sqrt(2)
    psi[idx2] = -1.0 / cp.sqrt(2)
    return psi


def partial_trace_2site(psi, site1: int, site2: int, N: int):
    """Compute 2-site reduced density matrix on GPU."""
    keep = sorted([site1, site2])
    trace_out = [q for q in range(N) if q not in keep]
    
    psi_tensor = psi.reshape([2] * N)
    new_order = keep + trace_out
    psi_reordered = cp.transpose(psi_tensor, new_order)
    
    dim_keep = 4
    dim_trace = 2**(N - 2)
    psi_matrix = psi_reordered.reshape(dim_keep, dim_trace)
    
    rho = psi_matrix @ psi_matrix.conj().T
    return rho


def exchange_expectation(rho_2site) -> float:
    """Compute SWAP expectation (GPU tensor, return CPU scalar)."""
    SWAP = cp.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=cp.complex128)
    
    result = cp.real(cp.trace(rho_2site @ SWAP))
    return float(result.get()) if GPU_AVAILABLE else float(result)


# ============================================================
# SINGLE CONFIGURATION EVALUATION
# ============================================================

def evaluate_config(
    dims: Tuple[int, ...],
    bandwidth: float,
    seed: int = 0,
    dt: float = 0.1,
    t_max: float = 3.0,
    pair_distance: int = 2,
    n_pairs: int = 5,
) -> Dict:
    """Evaluate fermionic coherence for one configuration."""
    
    start_time = time.time()
    
    d = len(dims)
    N = int(np.prod(dims))
    G = generate_lattice_graph(tuple(dims), periodic=True)
    coord = G.degree(0)
    
    # Memory check
    state_gb = (2**N * 16) / (1024**3)
    if GPU_AVAILABLE:
        free_mem = mempool.free_bytes() + (cp.cuda.runtime.memGetInfo()[0])
        free_gb = free_mem / (1024**3)
        if state_gb * 3 > free_gb:
            print(f"  WARNING: Need ~{state_gb*3:.1f}GB, only {free_gb:.1f}GB free")
    
    # Find disjoint pairs
    pairs = find_disjoint_pairs(G, distance=pair_distance, n_pairs=n_pairs)
    actual_n_pairs = len(pairs)
    
    if actual_n_pairs < 2:
        return {
            'dims': list(dims),
            'dimension': d,
            'N': N,
            'coordination': coord,
            'bandwidth': bandwidth,
            'n_pairs': actual_n_pairs,
            'error': 'insufficient_pairs',
            'fermionic_score': -999,
            'eval_time': time.time() - start_time,
        }
    
    # Build Trotter gates (transfers to GPU)
    onsite, edge = build_trotter_gates(G, N, dt, seed=seed)
    
    # Track each pair
    pair_results = []
    n_steps = int(t_max / dt)
    
    for pair_idx, (site1, site2) in enumerate(pairs):
        # Initialize singlet on GPU
        psi = create_singlet_state(N, site1, site2)
        
        # Initial exchange
        rho0 = partial_trace_2site(psi, site1, site2, N)
        init_exchange = exchange_expectation(rho0)
        
        # Evolve
        for step in range(n_steps):
            psi = trotter_step(psi, onsite, edge, N)
        
        # Final exchange
        rho_f = partial_trace_2site(psi, site1, site2, N)
        final_exchange = exchange_expectation(rho_f)
        
        stayed_fermionic = final_exchange < 0
        retention = min(1.0, abs(final_exchange / init_exchange)) if init_exchange < -0.5 else 0
        
        pair_results.append({
            'init_exchange': init_exchange,
            'final_exchange': final_exchange,
            'stayed_fermionic': stayed_fermionic,
            'retention': retention,
        })
        
        # Clear intermediate GPU memory
        del psi, rho0, rho_f
        clear_gpu_memory()
    
    # Aggregate
    n_stayed = sum(1 for p in pair_results if p['stayed_fermionic'])
    frac_stayed = n_stayed / actual_n_pairs
    mean_retention = np.mean([p['retention'] for p in pair_results])
    
    # Bandwidth
    bw_eff = min(1.0, bandwidth / coord) if coord > 0 else 1.0
    bw_penalty = max(0, (coord - bandwidth) / bandwidth) if bandwidth > 0 else 0.0
    
    raw_score = 2.0 * frac_stayed + 2.0 * mean_retention
    fermionic_score = raw_score * bw_eff - bw_penalty
    
    # Clean up GPU memory
    del onsite, edge
    clear_gpu_memory()
    
    eval_time = time.time() - start_time
    
    return {
        'dims': list(dims),
        'dimension': d,
        'N': N,
        'coordination': coord,
        'bandwidth': bandwidth,
        'n_pairs': actual_n_pairs,
        'n_stayed_fermionic': n_stayed,
        'frac_stayed_fermionic': frac_stayed,
        'mean_retention': mean_retention,
        'bw_efficiency': bw_eff,
        'raw_score': raw_score,
        'fermionic_score': fermionic_score,
        'eval_time': eval_time,
    }


# ============================================================
# EXPERIMENT RUNNER (Sequential - GPU doesn't parallelize well)
# ============================================================

def run_experiment(
    configs: List[Tuple[int, ...]],
    bandwidths: List[float],
    seeds: List[int] = [0],
    dt: float = 0.1,
    t_max: float = 3.0,
    pair_distance: int = 2,
    n_pairs: int = 5,
    output_file: str = None,
):
    """Run fermionic coherence experiment on GPU."""
    
    print("=" * 70)
    print("FERMIONIC COHERENCE EXPERIMENT (GPU)")
    print("=" * 70)
    print(f"Configs: {configs}")
    print(f"Bandwidths: {bandwidths}")
    print(f"Seeds: {seeds}")
    print(f"n_pairs: {n_pairs}")
    print(f"t_max: {t_max}")
    print(f"GPU: {GPU_AVAILABLE}")
    
    # Build job list
    jobs = []
    for B in bandwidths:
        for dims in configs:
            for seed in seeds:
                jobs.append((dims, B, seed))
    
    print(f"Total jobs: {len(jobs)}")
    print("=" * 70)
    print(flush=True)
    
    start = time.time()
    results = []
    
    for i, (dims, B, seed) in enumerate(jobs):
        print(f"[{i+1}/{len(jobs)}] {len(dims)}D {dims} B={B} seed={seed}...", 
              end=" ", flush=True)
        
        r = evaluate_config(
            dims, B, seed, dt, t_max, pair_distance, n_pairs
        )
        
        elapsed = time.time() - start
        print(f"score={r['fermionic_score']:.3f} ({r['eval_time']:.1f}s) "
              f"[total: {elapsed/60:.1f}min]", flush=True)
        
        results.append(r)
    
    total_time = time.time() - start
    
    # Phase diagram
    phase_diagram = {}
    for B in bandwidths:
        dim_scores = {}
        for r in results:
            if r['bandwidth'] == B and r.get('fermionic_score', -999) > -900:
                d = r['dimension']
                if d not in dim_scores:
                    dim_scores[d] = []
                dim_scores[d].append(r['fermionic_score'])
        
        dim_means = {d: np.mean(s) for d, s in dim_scores.items()}
        winner = max(dim_means.items(), key=lambda x: x[1])[0] if dim_means else 0
        phase_diagram[B] = {'winner': winner, 'scores': dim_means}
    
    # Print summary
    print("\n" + "=" * 70)
    print("PHASE DIAGRAM (Fermionic Coherence)")
    print("=" * 70)
    print(f"{'Bandwidth':<10} {'Winner':<8} {'1D':<12} {'2D':<12} {'3D':<12}")
    print("-" * 55)
    for B in bandwidths:
        w = phase_diagram[B]['winner']
        s = phase_diagram[B]['scores']
        print(f"{B:<10.1f} {w}D{'':<6} "
              f"{s.get(1, float('nan')):<12.4f} "
              f"{s.get(2, float('nan')):<12.4f} "
              f"{s.get(3, float('nan')):<12.4f}")
    
    # Detailed
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    for B in bandwidths:
        print(f"\n--- Bandwidth = {B} ---")
        B_results = [r for r in results if r['bandwidth'] == B]
        for r in sorted(B_results, key=lambda x: -x.get('fermionic_score', -999)):
            if 'error' in r:
                print(f"  {r['dimension']}D {tuple(r['dims'])}: ERROR - {r['error']}")
            else:
                print(f"  {r['dimension']}D {tuple(r['dims'])}: score={r['fermionic_score']:.4f}")
                print(f"      pairs: {r['n_pairs']}, stayed_fermionic: {r['n_stayed_fermionic']} "
                      f"({r['frac_stayed_fermionic']:.1%})")
                print(f"      mean_retention: {r['mean_retention']:.3f}, bw_eff: {r['bw_efficiency']:.2f}")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    # Save
    if output_file:
        output = {
            'phase_diagram': {str(k): v for k, v in phase_diagram.items()},
            'results': results,
            'config': {
                'configs': [list(c) for c in configs],
                'bandwidths': bandwidths,
                'seeds': seeds,
                't_max': t_max,
                'n_pairs': n_pairs,
                'gpu': GPU_AVAILABLE,
                'total_time_minutes': total_time / 60,
            }
        }
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to: {output_file}")
    
    return phase_diagram, results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fermionic Coherence (GPU)")
    parser.add_argument('--test', action='store_true', help="N=8 quick test")
    parser.add_argument('--small', action='store_true', help="N=16")
    parser.add_argument('--medium', action='store_true', help="N=20")
    parser.add_argument('--full', action='store_true', help="N=24")
    parser.add_argument('--true3d', action='store_true', help="N=27 (true 3D coord=6)")
    parser.add_argument('--multi', action='store_true', help="(ignored, always multi-pair)")
    parser.add_argument('--n-pairs', type=int, default=5)
    parser.add_argument('--bandwidths', type=str, default=None)
    parser.add_argument('--seeds', type=str, default="0")
    parser.add_argument('--tmax', type=float, default=3.0)
    parser.add_argument('--quick', action='store_true', help="Short t_max for validation")
    parser.add_argument('--pair-distance', type=int, default=2)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    if args.test:
        configs = [(8,), (4, 2), (2, 2, 2)]
        bandwidths = [2.0, 3.0, 4.0]
    elif args.small:
        configs = [(16,), (4, 4), (2, 2, 4)]
        bandwidths = [2.0, 4.0, 6.0]
    elif args.medium:
        configs = [(20,), (5, 4), (2, 2, 5)]
        bandwidths = [2.0, 4.0, 6.0]
    elif args.full:
        configs = [(24,), (6, 4), (2, 3, 4)]
        bandwidths = [2.0, 4.0, 5.0, 6.0, 8.0]
    elif args.true3d:
        configs = [(27,), (9, 3), (3, 3, 3)]
        bandwidths = [2.0, 4.0, 5.0, 6.0, 8.0]
    else:
        configs = [(8,), (4, 2), (2, 2, 2)]
        bandwidths = [2.0, 3.0, 4.0]
    
    if args.bandwidths:
        bandwidths = [float(b) for b in args.bandwidths.split(',')]
    
    t_max = args.tmax
    if args.quick:
        t_max = 0.5
        print("QUICK MODE: t_max=0.5")
    
    # Memory estimate
    max_N = max(np.prod(c) for c in configs)
    state_gb = (2**max_N * 16) / (1024**3)
    print(f"\nState vector size: {state_gb:.1f} GB for N={max_N}")
    print(f"Need ~{state_gb * 3:.1f} GB VRAM for computation")
    
    if GPU_AVAILABLE:
        free_gb = cp.cuda.runtime.memGetInfo()[0] / (1024**3)
        total_gb = cp.cuda.runtime.memGetInfo()[1] / (1024**3)
        print(f"GPU VRAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
        if state_gb * 3 > free_gb:
            print("WARNING: May run out of VRAM!")
    
    run_experiment(
        configs=configs,
        bandwidths=bandwidths,
        seeds=seeds,
        dt=0.1,
        t_max=t_max,
        pair_distance=args.pair_distance,
        n_pairs=args.n_pairs,
        output_file=args.output,
    )