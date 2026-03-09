"""
Comprehensive Fermionic Experiment - GPU + Trotter
===================================================

Three metrics with GPU acceleration via Trotter evolution.
Scales to N=24+ unlike full matrix expm.

Metrics:
1. SCRAMBLING RATE: Half-life of antisymmetry decay
2. JW FERMIONIC: Evolution under actual fermionic hopping
3. PATH CORRELATIONS: Survival across disjoint paths

Usage:
    python fermionic_comprehensive_gpu.py --test
    python fermionic_comprehensive_gpu.py --full --output results.json
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
    vram = cp.cuda.runtime.memGetInfo()[1] / 1e9
    print(f"VRAM: {vram:.1f} GB")
except ImportError:
    import numpy as cp
    GPU = False
    print("CPU mode (no CuPy)")


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
    
    for node in range(N):
        coords = list(to_coords(node))
        for axis in range(d):
            nc = coords.copy()
            nc[axis] = (coords[axis] + 1) % dims[axis]
            G.add_edge(node, to_idx(nc))
    
    return G


def disjoint_pairs(G, dist=2, n=5):
    N = G.number_of_nodes()
    dists = dict(nx.all_pairs_shortest_path_length(G))
    cands = [(i, j) for i in range(N) for j in range(i+1, N) if dists[i][j] == dist]
    if not cands:
        cands = list(G.edges())
    
    pairs, used = [], set()
    for (i, j) in cands:
        if i not in used and j not in used:
            pairs.append((i, j))
            used.update([i, j])
            if len(pairs) >= n:
                break
    return pairs


def count_disjoint_paths(G, s, t):
    try:
        return len(list(nx.node_disjoint_paths(G, s, t)))
    except:
        return 1


# ============================================================
# PAULI OPERATORS
# ============================================================

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = [I, X, Y, Z]
Sp = (X + 1j*Y) / 2  # Raising operator
Sm = (X - 1j*Y) / 2  # Lowering operator


# ============================================================
# TROTTER GATES
# ============================================================

def make_random_gates(G, N, dt, seed=0):
    """Random Pauli Hamiltonian Trotter gates."""
    np.random.seed(seed)
    
    onsite = {}
    for i in range(N):
        h = np.zeros((2, 2), dtype=np.complex128)
        for p in range(1, 4):
            h += np.random.normal(0, 0.25) * PAULIS[p]
        onsite[i] = cp.asarray(expm(-1j * dt * h))
    
    edge = {}
    for (i, j) in G.edges():
        h = np.zeros((4, 4), dtype=np.complex128)
        for p in range(1, 4):
            for q in range(1, 4):
                h += np.random.normal(0, 0.8) * np.kron(PAULIS[p], PAULIS[q])
        edge[(i, j)] = cp.asarray(expm(-1j * dt * h))
    
    return onsite, edge


def make_jw_gates(G, N, dt, t_hop=1.0):
    """
    Jordan-Wigner fermionic hopping gates.
    
    For adjacent sites i, j (i < j), the hopping term is:
    c†_i c_j + c†_j c_i = σ+_i (Z_string) σ-_j + h.c.
    
    For nearest neighbors on the lattice, we apply the gate
    to the 2-qubit subspace directly.
    """
    edge = {}
    for (i, j) in G.edges():
        if i > j:
            i, j = j, i
        
        # 2-site hopping Hamiltonian: -t(σ+⊗σ- + σ-⊗σ+)
        # In basis {|00⟩, |01⟩, |10⟩, |11⟩}
        hop = np.kron(Sp, Sm)  # |10⟩ → |01⟩
        h_2site = -t_hop * (hop + hop.conj().T)
        edge[(i, j)] = cp.asarray(expm(-1j * dt * h_2site))
    
    # Identity on-site (no chemical potential)
    onsite = {i: cp.asarray(I) for i in range(N)}
    
    return onsite, edge


# ============================================================
# TROTTER EVOLUTION
# ============================================================

def apply_1q(psi, q, gate, N):
    """Apply single-qubit gate."""
    l, r = 2**q, 2**(N - q - 1)
    psi = psi.reshape(l, 2, r)
    psi = cp.einsum('ab,lbr->lar', gate, psi)
    return psi.reshape(-1)


def apply_2q(psi, q1, q2, gate, N):
    """Apply two-qubit gate."""
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
    """One Trotter step."""
    for q, g in onsite.items():
        psi = apply_1q(psi, q, g, N)
    for (q1, q2), g in edge.items():
        psi = apply_2q(psi, q1, q2, g, N)
    return psi / cp.linalg.norm(psi)


# ============================================================
# STATES AND MEASUREMENT
# ============================================================

def singlet(N, s1, s2):
    """Create antisymmetric (singlet) state."""
    psi = cp.zeros(2**N, dtype=cp.complex128)
    psi[2**(N - 1 - s1)] = 1 / cp.sqrt(2)
    psi[2**(N - 1 - s2)] = -1 / cp.sqrt(2)
    return psi


def ptrace2(psi, s1, s2, N):
    """Partial trace to 2-site density matrix."""
    keep = sorted([s1, s2])
    trace = [q for q in range(N) if q not in keep]
    psi = psi.reshape([2] * N)
    psi = cp.transpose(psi, keep + trace)
    psi = psi.reshape(4, 2**(N - 2))
    return psi @ psi.conj().T


def swap_exp(rho):
    """Compute ⟨SWAP⟩. Returns -1 for antisymmetric, +1 for symmetric."""
    SWAP = cp.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=cp.complex128)
    val = cp.real(cp.trace(rho @ SWAP))
    return float(val.get()) if GPU else float(val)


# ============================================================
# METRIC 1: SCRAMBLING RATE
# ============================================================

def metric_scrambling(dims, s1, s2, onsite, edge, dt, t_max, n_pts=11):
    """
    Track ⟨SWAP⟩ decay under random Hamiltonian.
    Measure half-life (time to cross -0.5).
    
    Slower scrambling = better fermionic support.
    """
    N = int(np.prod(dims))
    times = np.linspace(0, t_max, n_pts)
    n_steps_total = int(t_max / dt)
    steps_per_pt = max(1, n_steps_total // (n_pts - 1))
    
    psi = singlet(N, s1, s2)
    swaps = [swap_exp(ptrace2(psi, s1, s2, N))]
    
    for pt in range(1, n_pts):
        for _ in range(steps_per_pt):
            psi = trotter_step(psi, onsite, edge, N)
        swaps.append(swap_exp(ptrace2(psi, s1, s2, N)))
    
    # Find half-life (when SWAP crosses -0.5)
    half_life = t_max
    for i, s in enumerate(swaps):
        if s > -0.5 and i > 0:
            t0, t1 = times[i-1], times[i]
            s0, s1v = swaps[i-1], s
            half_life = t0 + (t1 - t0) * (-0.5 - s0) / (s1v - s0 + 1e-10)
            break
    
    del psi
    clear_gpu()
    
    return {'half_life': half_life, 'final_swap': swaps[-1], 'trajectory': swaps}


# ============================================================
# METRIC 2: JW FERMIONIC EVOLUTION
# ============================================================

def metric_jw(dims, s1, s2, dt, t_max, n_pts=11):
    """
    Track ⟨SWAP⟩ under Jordan-Wigner fermionic hopping.
    
    This is the "natural" dynamics for fermions - the lattice
    topology directly affects transport.
    
    Higher frac_antisym = better fermionic support.
    """
    N = int(np.prod(dims))
    G = make_lattice(dims)
    onsite, edge = make_jw_gates(G, N, dt)
    
    times = np.linspace(0, t_max, n_pts)
    n_steps = int(t_max / dt)
    steps_per = max(1, n_steps // (n_pts - 1))
    
    psi = singlet(N, s1, s2)
    swaps = [swap_exp(ptrace2(psi, s1, s2, N))]
    
    for pt in range(1, n_pts):
        for _ in range(steps_per):
            psi = trotter_step(psi, onsite, edge, N)
        swaps.append(swap_exp(ptrace2(psi, s1, s2, N)))
    
    frac_antisym = np.mean([s < 0 for s in swaps])
    
    del psi, onsite, edge
    clear_gpu()
    
    return {
        'frac_antisym': frac_antisym,
        'mean_swap': np.mean(swaps),
        'final_swap': swaps[-1],
        'trajectory': swaps,
    }


# ============================================================
# METRIC 3: PATH CORRELATION SURVIVAL
# ============================================================

def metric_paths(dims, onsite, edge, dt, t_max, n_pairs=5):
    """
    Measure final ⟨SWAP⟩ for multiple disjoint pairs.
    
    In higher dimensions, disjoint paths allow independent
    evolution without interference.
    
    Higher frac_stayed + independence = better.
    """
    N = int(np.prod(dims))
    G = make_lattice(dims)
    pairs = disjoint_pairs(G, dist=2, n=n_pairs)
    if len(pairs) < 2:
        pairs = disjoint_pairs(G, dist=1, n=n_pairs)
    
    # Graph property: average disjoint paths
    avg_dpaths = np.mean([count_disjoint_paths(G, p[0], p[1]) for p in pairs]) if pairs else 1
    
    n_steps = int(t_max / dt)
    
    finals = []
    for (s1, s2) in pairs:
        psi = singlet(N, s1, s2)
        for _ in range(n_steps):
            psi = trotter_step(psi, onsite, edge, N)
        finals.append(swap_exp(ptrace2(psi, s1, s2, N)))
        del psi
        clear_gpu()
    
    n_stayed = sum(1 for f in finals if f < 0)
    std = np.std(finals) if len(finals) > 1 else 0
    
    return {
        'n_pairs': len(pairs),
        'disjoint_paths': avg_dpaths,
        'frac_stayed': n_stayed / len(pairs) if pairs else 0,
        'independence': 1 / (1 + std),
        'finals': finals,
    }


# ============================================================
# COMBINED EVALUATION
# ============================================================

def evaluate(dims, bandwidth, seed=0, dt=0.1, t_max=2.0, n_pairs=5):
    """Evaluate all three metrics for one configuration."""
    start = time.time()
    
    d = len(dims)
    N = int(np.prod(dims))
    G = make_lattice(dims)
    coord = G.degree(0)
    
    # Build random Hamiltonian gates
    onsite_rand, edge_rand = make_random_gates(G, N, dt, seed)
    
    # Get pairs for testing
    pairs = disjoint_pairs(G, dist=2, n=n_pairs)
    if len(pairs) < 2:
        pairs = disjoint_pairs(G, dist=1, n=n_pairs)
    
    # METRIC 1: Scrambling rate (average over a few pairs)
    test_pairs = pairs[:min(3, len(pairs))]
    scrams = [metric_scrambling(dims, p[0], p[1], onsite_rand, edge_rand, dt, t_max)
              for p in test_pairs]
    avg_half = np.mean([s['half_life'] for s in scrams])
    score_scram = avg_half / t_max  # Normalized: longer = better
    
    # METRIC 2: JW fermionic evolution
    jws = [metric_jw(dims, p[0], p[1], dt, t_max) for p in test_pairs]
    avg_jw_frac = np.mean([j['frac_antisym'] for j in jws])
    score_jw = avg_jw_frac
    
    # METRIC 3: Path correlations
    paths = metric_paths(dims, onsite_rand, edge_rand, dt, t_max, n_pairs)
    score_path = (paths['frac_stayed'] + paths['independence']) / 2
    
    # Bandwidth efficiency
    bw_eff = min(1.0, bandwidth / coord) if coord > 0 else 1.0
    bw_pen = max(0, (coord - bandwidth) / bandwidth) if bandwidth > 0 else 0
    
    # Combined score
    raw = score_scram + score_jw + score_path
    final = raw * bw_eff - bw_pen
    
    # Cleanup
    del onsite_rand, edge_rand
    clear_gpu()
    
    return {
        'dims': list(dims),
        'dimension': d,
        'N': N,
        'coord': coord,
        'bandwidth': bandwidth,
        'bw_eff': bw_eff,
        
        # Metric 1
        'avg_half_life': avg_half,
        'score_scram': score_scram,
        
        # Metric 2
        'avg_jw_frac': avg_jw_frac,
        'score_jw': score_jw,
        
        # Metric 3
        'disjoint_paths': paths['disjoint_paths'],
        'frac_stayed': paths['frac_stayed'],
        'independence': paths['independence'],
        'score_path': score_path,
        
        # Combined
        'raw': raw,
        'final': final,
        
        'eval_time': time.time() - start,
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run(configs, bandwidths, seeds=[0], dt=0.1, t_max=2.0, n_pairs=5, output=None):
    """Run comprehensive fermionic experiment."""
    
    print("=" * 70)
    print("COMPREHENSIVE FERMIONIC EXPERIMENT (GPU + Trotter)")
    print("=" * 70)
    print(f"Configs: {configs}")
    print(f"Bandwidths: {bandwidths}")
    print(f"Seeds: {seeds}")
    print(f"t_max: {t_max}, dt: {dt}")
    print("=" * 70)
    print(flush=True)
    
    jobs = [(d, B, s) for B in bandwidths for d in configs for s in seeds]
    print(f"Total jobs: {len(jobs)}\n")
    
    results = []
    start = time.time()
    
    for i, (dims, B, seed) in enumerate(jobs):
        print(f"[{i+1}/{len(jobs)}] {len(dims)}D {dims} B={B} seed={seed}...",
              end=" ", flush=True)
        r = evaluate(dims, B, seed, dt, t_max, n_pairs)
        elapsed = time.time() - start
        print(f"final={r['final']:.3f} ({r['eval_time']:.1f}s) [total: {elapsed/60:.1f}min]")
        results.append(r)
    
    total_time = time.time() - start
    
    # Build phase diagram
    print("\n" + "=" * 70)
    print("PHASE DIAGRAM")
    print("=" * 70)
    print(f"{'B':<6} {'Win':<5} {'1D':<10} {'2D':<10} {'3D':<10}")
    print("-" * 45)
    
    phase_diagram = {}
    for B in bandwidths:
        scores = {}
        for r in results:
            if r['bandwidth'] == B:
                d = r['dimension']
                if d not in scores:
                    scores[d] = []
                scores[d].append(r['final'])
        
        means = {d: np.mean(s) for d, s in scores.items()}
        winner = max(means, key=means.get) if means else 0
        phase_diagram[B] = {'winner': winner, 'scores': means}
        
        print(f"{B:<6.1f} {winner}D{'':<3} "
              f"{means.get(1, 0):<10.3f} "
              f"{means.get(2, 0):<10.3f} "
              f"{means.get(3, 0):<10.3f}")
    
    # Metric breakdown at max bandwidth
    max_B = max(bandwidths)
    print("\n" + "=" * 70)
    print(f"METRIC BREAKDOWN (B={max_B})")
    print("=" * 70)
    print(f"{'Config':<15} {'Coord':<6} {'Scram':<8} {'JW':<8} {'Path':<8} {'DPaths':<8} {'Raw':<8}")
    print("-" * 70)
    
    for r in sorted([x for x in results if x['bandwidth'] == max_B],
                    key=lambda x: -x['final']):
        cfg = f"{r['dimension']}D {tuple(r['dims'])}"
        print(f"{cfg:<15} {r['coord']:<6} "
              f"{r['score_scram']:<8.3f} {r['score_jw']:<8.3f} "
              f"{r['score_path']:<8.3f} {r['disjoint_paths']:<8.1f} {r['raw']:<8.3f}")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    # Save results
    if output:
        out_data = {
            'phase_diagram': {str(k): v for k, v in phase_diagram.items()},
            'results': results,
            'config': {
                'configs': [list(c) for c in configs],
                'bandwidths': bandwidths,
                'seeds': seeds,
                'dt': dt,
                't_max': t_max,
                'n_pairs': n_pairs,
                'total_time_minutes': total_time / 60,
            }
        }
        with open(output, 'w') as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved to: {output}")
    
    return phase_diagram, results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser(description="Comprehensive Fermionic Experiment (GPU)")
    p.add_argument('--test', action='store_true', help="N=8 quick test")
    p.add_argument('--small', action='store_true', help="N=16")
    p.add_argument('--medium', action='store_true', help="N=20")
    p.add_argument('--full', action='store_true', help="N=24")
    p.add_argument('--true3d', action='store_true', help="N=27 (true coord=6)")
    p.add_argument('--bandwidths', type=str, default=None)
    p.add_argument('--seeds', type=str, default="0")
    p.add_argument('--tmax', type=float, default=2.0)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--n-pairs', type=int, default=5)
    p.add_argument('--output', type=str, default=None)
    
    args = p.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    if args.test:
        configs = [(8,), (4, 2), (2, 2, 2)]
        bandwidths = [2.0, 4.0, 6.0]
    elif args.small:
        configs = [(16,), (4, 4), (2, 2, 4)]
        bandwidths = [2.0, 4.0, 6.0]
    elif args.medium:
        configs = [(20,), (5, 4), (2, 2, 5)]
        bandwidths = [2.0, 4.0, 5.0, 6.0]
    elif args.full:
        configs = [(24,), (6, 4), (2, 3, 4)]
        bandwidths = [2.0, 4.0, 5.0, 6.0, 8.0]
    elif args.true3d:
        configs = [(27,), (9, 3), (3, 3, 3)]
        bandwidths = [2.0, 4.0, 5.0, 6.0, 8.0]
    else:
        configs = [(8,), (4, 2), (2, 2, 2)]
        bandwidths = [2.0, 4.0, 6.0]
    
    if args.bandwidths:
        bandwidths = [float(b) for b in args.bandwidths.split(',')]
    
    # Memory estimate
    max_N = max(np.prod(c) for c in configs)
    state_gb = (2**max_N * 16) / 1e9
    print(f"\nState vector: {state_gb:.2f} GB for N={max_N}")
    if GPU:
        free = cp.cuda.runtime.memGetInfo()[0] / 1e9
        print(f"GPU free: {free:.1f} GB")
        if state_gb * 3 > free:
            print("WARNING: May exceed VRAM!")
    
    run(
        configs=configs,
        bandwidths=bandwidths,
        seeds=seeds,
        dt=args.dt,
        t_max=args.tmax,
        n_pairs=args.n_pairs,
        output=args.output,
    )