"""
Fermionic Coherence Experiment
==============================

Tests whether antisymmetric (fermionic) correlations survive better
in higher-dimensional geometries, validating the JW string hypothesis.

Key insight: In 3D, JW strings have more "room" (disjoint paths) to 
coexist without interference. This should manifest as better survival
of antisymmetric correlations under unitary dynamics.

Experiment:
1. Initialize antisymmetric pair states (singlet-like)
2. Evolve under Trotter dynamics  
3. Measure survival of antisymmetric correlations
4. Combine with bandwidth constraints
5. Compare across 1D, 2D, 3D topologies

Usage:
    python fermionic_coherence.py --test --workers 2
    python fermionic_coherence.py --true3d --workers 3 --output results.json

Author: Ben Bray
"""

import numpy as np
from scipy.linalg import expm
import networkx as nx
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
import time
import multiprocessing as mp

# ============================================================
# LATTICE GRAPHS
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


# ============================================================
# TROTTER EVOLUTION
# ============================================================

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
PAULIS = [I, X, Y, Z]


def build_trotter_gates(G: nx.Graph, N: int, dt: float,
                        h_std: float = 0.25, J_std: float = 0.8, seed: int = 0):
    np.random.seed(seed)
    
    onsite_gates = {}
    for i in range(N):
        h_op = np.zeros((2, 2), dtype=complex)
        for p in range(1, 4):
            h_op += np.random.normal(0, h_std) * PAULIS[p]
        onsite_gates[i] = expm(-1j * dt * h_op)
    
    edge_gates = {}
    for (i, j) in G.edges():
        h_edge = np.zeros((4, 4), dtype=complex)
        for p in range(1, 4):
            for q in range(1, 4):
                h_edge += np.random.normal(0, J_std) * np.kron(PAULIS[p], PAULIS[q])
        edge_gates[(i, j)] = expm(-1j * dt * h_edge)
    
    return onsite_gates, edge_gates


def apply_single_qubit_gate(psi, qubit, gate, N):
    left_dim = 2**qubit
    right_dim = 2**(N - qubit - 1)
    psi_reshaped = psi.reshape(left_dim, 2, right_dim)
    result = np.einsum('ab,lbr->lar', gate, psi_reshaped)
    return result.reshape(-1)


def apply_two_qubit_gate(psi, q1, q2, gate, N):
    if q1 > q2:
        q1, q2 = q2, q1
        gate = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    
    left_dim = 2**q1
    mid_dim = 2**(q2 - q1 - 1)
    right_dim = 2**(N - q2 - 1)
    
    psi_reshaped = psi.reshape(left_dim, 2, mid_dim, 2, right_dim)
    gate_reshaped = gate.reshape(2, 2, 2, 2)
    result = np.einsum('abcd,lcmdr->lamdr', gate_reshaped, psi_reshaped)
    return result.reshape(-1)


def trotter_step(psi, onsite, edge, N):
    for qubit, gate in onsite.items():
        psi = apply_single_qubit_gate(psi, qubit, gate, N)
    for (q1, q2), gate in edge.items():
        psi = apply_two_qubit_gate(psi, q1, q2, gate, N)
    return psi / np.linalg.norm(psi)


# ============================================================
# FERMIONIC STATE PREPARATION
# ============================================================

def create_singlet_state(N: int, site1: int, site2: int) -> np.ndarray:
    """
    Create a singlet (antisymmetric) state between two sites:
    |ψ⟩ = (|↑₁↓₂⟩ - |↓₁↑₂⟩) / √2
    
    In computational basis with |0⟩=↓, |1⟩=↑:
    |ψ⟩ = (|1_site1, 0_site2⟩ - |0_site1, 1_site2⟩) / √2
    """
    psi = np.zeros(2**N, dtype=complex)
    
    # |1_site1, 0_site2⟩: site1 is |1⟩, site2 is |0⟩, others are |0⟩
    idx1 = 2**(N - 1 - site1)  # Only site1 is 1
    
    # |0_site1, 1_site2⟩: site2 is |1⟩, site1 is |0⟩, others are |0⟩  
    idx2 = 2**(N - 1 - site2)  # Only site2 is 1
    
    psi[idx1] = 1.0 / np.sqrt(2)
    psi[idx2] = -1.0 / np.sqrt(2)  # Minus for antisymmetry
    
    return psi


# ============================================================
# ANTISYMMETRIC CORRELATION MEASURES
# ============================================================

def partial_trace_2site(psi: np.ndarray, site1: int, site2: int, N: int) -> np.ndarray:
    """
    Compute the 2-site reduced density matrix for sites site1 and site2.
    Returns a 4x4 matrix in the basis {|00⟩, |01⟩, |10⟩, |11⟩}.
    """
    keep = sorted([site1, site2])
    trace_out = [q for q in range(N) if q not in keep]
    
    psi_tensor = psi.reshape([2] * N)
    new_order = keep + trace_out
    psi_reordered = np.transpose(psi_tensor, new_order)
    
    dim_keep = 4  # 2^2
    dim_trace = 2**(N - 2)
    psi_matrix = psi_reordered.reshape(dim_keep, dim_trace)
    
    return psi_matrix @ psi_matrix.conj().T


def singlet_projection(rho_2site: np.ndarray) -> float:
    """
    Compute the projection of a 2-site density matrix onto the singlet state.
    
    Singlet = (|01⟩ - |10⟩)/√2
    In the basis {|00⟩, |01⟩, |10⟩, |11⟩}, singlet projector is:
    P_singlet = |singlet⟩⟨singlet| = 0.5 * (|01⟩⟨01| + |10⟩⟨10| - |01⟩⟨10| - |10⟩⟨01|)
    """
    # Singlet state in 4-element basis
    singlet = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    P_singlet = np.outer(singlet, singlet.conj())
    
    # Tr(rho * P_singlet)
    return np.real(np.trace(rho_2site @ P_singlet))


def exchange_expectation(rho_2site: np.ndarray) -> float:
    """
    Compute expectation value of the SWAP operator.
    
    For fermions (antisymmetric states): ⟨SWAP⟩ = -1
    For bosons (symmetric states): ⟨SWAP⟩ = +1
    For mixed states: -1 < ⟨SWAP⟩ < +1
    
    SWAP in basis {|00⟩, |01⟩, |10⟩, |11⟩}:
    SWAP|00⟩ = |00⟩, SWAP|01⟩ = |10⟩, SWAP|10⟩ = |01⟩, SWAP|11⟩ = |11⟩
    """
    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)
    
    return np.real(np.trace(rho_2site @ SWAP))


def concurrence(rho_2site: np.ndarray) -> float:
    """
    Compute concurrence (entanglement measure) for a 2-qubit state.
    Concurrence = 1 for maximally entangled (including singlet).
    """
    # Pauli Y tensor Y
    YY = np.array([
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0]
    ], dtype=complex)
    
    rho_tilde = YY @ rho_2site.conj() @ YY
    R = rho_2site @ rho_tilde
    
    eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
    sqrt_eigs = np.sqrt(eigenvalues)
    
    C = max(0, sqrt_eigs[0] - sqrt_eigs[1] - sqrt_eigs[2] - sqrt_eigs[3])
    return C


# ============================================================
# MULTI-PAIR FERMIONIC COHERENCE
# ============================================================

def create_multi_singlet_state(N: int, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """
    Create a product of singlet states across multiple disjoint pairs.
    This represents multiple simultaneous fermionic correlations.
    
    |ψ⟩ = |singlet₁⟩ ⊗ |singlet₂⟩ ⊗ ...
    
    For simplicity, we create a superposition where each term has one pair excited.
    """
    n_pairs = len(pairs)
    psi = np.zeros(2**N, dtype=complex)
    
    for (s1, s2) in pairs:
        # Add singlet contribution for this pair
        idx1 = 2**(N - 1 - s1)
        idx2 = 2**(N - 1 - s2)
        psi[idx1] += 1.0 / np.sqrt(2 * n_pairs)
        psi[idx2] += -1.0 / np.sqrt(2 * n_pairs)
    
    return psi / np.linalg.norm(psi)


def find_disjoint_pairs(G: nx.Graph, distance: int = 2, n_pairs: int = 3) -> List[Tuple[int, int]]:
    """
    Find vertex-disjoint pairs at specified distance.
    This is where JW strings matter - disjoint pairs can maintain independent correlations.
    """
    N = G.number_of_nodes()
    distances = dict(nx.all_pairs_shortest_path_length(G))
    
    # All pairs at target distance
    candidates = [(i, j) for i in range(N) for j in range(i+1, N)
                  if distances[i][j] == distance]
    
    if not candidates:
        # Fallback to edges
        candidates = list(G.edges())
    
    # Greedy selection of vertex-disjoint pairs
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


def measure_multi_pair_coherence(psi: np.ndarray, N: int, 
                                  pairs: List[Tuple[int, int]]) -> Dict:
    """
    Measure coherence preservation across multiple pairs.
    
    Key metrics:
    1. Average singlet fidelity across pairs
    2. Total antisymmetric character (sum of exchange expectations)
    3. Pairwise independence (low cross-correlations)
    """
    singlet_fidelities = []
    exchange_values = []
    
    for (s1, s2) in pairs:
        rho_2 = partial_trace_2site(psi, s1, s2, N)
        singlet_fidelities.append(singlet_projection(rho_2))
        exchange_values.append(exchange_expectation(rho_2))
    
    return {
        'mean_singlet_fidelity': np.mean(singlet_fidelities),
        'total_singlet_fidelity': np.sum(singlet_fidelities),
        'mean_exchange': np.mean(exchange_values),
        'total_fermionic_character': -np.sum(exchange_values),  # Negative because -1 is fermionic
        'individual_fidelities': singlet_fidelities,
    }


def evaluate_multi_pair(args_tuple):
    """
    Evaluate multi-pair fermionic coherence.
    Tests whether multiple JW strings can coexist.
    
    Key idea: Track exchange expectation for EACH pair independently,
    then measure how many pairs retain fermionic character (negative SWAP).
    """
    dims, bandwidth, seed, dt, t_max, pair_distance, n_pairs = args_tuple
    
    start_time = time.time()
    
    d = len(dims)
    N = int(np.prod(dims))
    G = generate_lattice_graph(tuple(dims), periodic=True)
    coord = G.degree(0)
    
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
    
    # Build Trotter gates
    onsite, edge = build_trotter_gates(G, N, dt, seed=seed)
    
    # Track each pair separately - initialize each as a singlet
    # and measure its exchange character over time
    pair_results = []
    
    for (site1, site2) in pairs:
        # Initialize THIS pair as a singlet (others in |0⟩)
        psi = create_singlet_state(N, site1, site2)
        
        # Initial exchange
        rho0 = partial_trace_2site(psi, site1, site2, N)
        init_exchange = exchange_expectation(rho0)
        
        # Evolve
        n_steps = int(t_max / dt)
        for _ in range(n_steps):
            psi = trotter_step(psi, onsite, edge, N)
        
        # Final exchange
        rho_f = partial_trace_2site(psi, site1, site2, N)
        final_exchange = exchange_expectation(rho_f)
        
        # Does it stay fermionic? (negative SWAP expectation)
        stayed_fermionic = final_exchange < 0
        fermionic_retention = min(1.0, abs(final_exchange / init_exchange)) if init_exchange < -0.5 else 0
        
        pair_results.append({
            'init_exchange': init_exchange,
            'final_exchange': final_exchange,
            'stayed_fermionic': stayed_fermionic,
            'retention': fermionic_retention,
        })
    
    # Aggregate: what fraction of pairs stayed fermionic?
    n_stayed = sum(1 for p in pair_results if p['stayed_fermionic'])
    frac_stayed = n_stayed / actual_n_pairs
    mean_retention = np.mean([p['retention'] for p in pair_results])
    
    # Bandwidth
    bw_eff = min(1.0, bandwidth / coord) if coord > 0 else 1.0
    bw_penalty = max(0, (coord - bandwidth) / bandwidth) if bandwidth > 0 else 0.0
    
    # Score: fraction of pairs that stay fermionic × retention × bandwidth
    raw_score = 2.0 * frac_stayed + 2.0 * mean_retention
    fermionic_score = raw_score * bw_eff - bw_penalty
    
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
        'pair_details': pair_results,
        'bw_efficiency': bw_eff,
        'raw_score': raw_score,
        'fermionic_score': fermionic_score,
        'eval_time': eval_time,
    }


def run_multi_pair_experiment(
    configs: List[Tuple[int, ...]],
    bandwidths: List[float],
    seeds: List[int] = [0],
    workers: int = 1,
    dt: float = 0.1,
    t_max: float = 3.0,
    pair_distance: int = 2,
    n_pairs: int = 3,
    output_file: str = None,
):
    """Run multi-pair fermionic coherence experiment."""
    
    print("=" * 70)
    print("MULTI-PAIR FERMIONIC COHERENCE EXPERIMENT")
    print("=" * 70)
    print(f"Configs: {configs}")
    print(f"Bandwidths: {bandwidths}")
    print(f"n_pairs: {n_pairs}")
    print(f"Workers: {workers}")
    
    # Build jobs
    jobs = []
    for B in bandwidths:
        for dims in configs:
            for seed in seeds:
                jobs.append((dims, B, seed, dt, t_max, pair_distance, n_pairs))
    
    print(f"Total jobs: {len(jobs)}")
    print("=" * 70)
    print(flush=True)
    
    start = time.time()
    
    if workers > 1:
        # Use imap_unordered for progress output
        results = []
        with mp.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(evaluate_multi_pair, jobs)):
                elapsed = time.time() - start
                print(f"[{i+1}/{len(jobs)}] {r['dimension']}D {tuple(r['dims'])} B={r['bandwidth']}: "
                      f"score={r['fermionic_score']:.3f} ({r['eval_time']:.1f}s) "
                      f"[elapsed: {elapsed/3600:.1f}h]", flush=True)
                results.append(r)
    else:
        results = []
        for i, job in enumerate(jobs):
            print(f"Job {i+1}/{len(jobs)}: {job[0]} B={job[1]}...", end=" ", flush=True)
            r = evaluate_multi_pair(job)
            print(f"score={r['fermionic_score']:.3f} ({r['eval_time']:.1f}s)")
            results.append(r)
    
    total_time = time.time() - start
    
    # Phase diagram
    phase_diagram = {}
    for B in bandwidths:
        dim_scores = {}
        for r in results:
            if r['bandwidth'] == B and r['fermionic_score'] > -900:
                d = r['dimension']
                if d not in dim_scores:
                    dim_scores[d] = []
                dim_scores[d].append(r['fermionic_score'])
        
        dim_means = {d: np.mean(s) for d, s in dim_scores.items()}
        winner = max(dim_means.items(), key=lambda x: x[1])[0] if dim_means else 0
        phase_diagram[B] = {'winner': winner, 'scores': dim_means}
    
    # Print
    print("\n" + "=" * 70)
    print("MULTI-PAIR PHASE DIAGRAM")
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
    print("DETAILED (Multi-pair)")
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
    
    print(f"\nTotal time: {total_time/3600:.2f} hours")
    
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
                'total_time_hours': total_time / 3600,
            }
        }
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to: {output_file}")
    
    return phase_diagram, results

def evaluate_single(args_tuple):
    """
    Evaluate fermionic coherence for one configuration.
    Designed for parallel execution.
    """
    dims, bandwidth, seed, dt, t_max, pair_distance = args_tuple
    
    start_time = time.time()
    
    d = len(dims)
    N = int(np.prod(dims))
    G = generate_lattice_graph(tuple(dims), periodic=True)
    coord = G.degree(0)
    
    # Find pairs at target distance
    distances = dict(nx.all_pairs_shortest_path_length(G))
    pairs_at_distance = []
    for i in range(N):
        for j in range(i+1, N):
            if distances[i][j] == pair_distance:
                pairs_at_distance.append((i, j))
    
    actual_distance = pair_distance
    if not pairs_at_distance:
        # Fallback to nearest neighbors
        actual_distance = 1
        pairs_at_distance = list(G.edges())
    
    # Sample pairs
    np.random.seed(seed + 500)
    n_sample = min(5, len(pairs_at_distance))
    sample_pairs = [pairs_at_distance[i] for i in 
                    np.random.choice(len(pairs_at_distance), n_sample, replace=False)]
    
    # Build Trotter gates
    onsite, edge = build_trotter_gates(G, N, dt, seed=seed)
    
    # Metrics storage
    all_singlet_init = []
    all_singlet_final = []
    all_exchange_init = []
    all_exchange_final = []
    all_conc_init = []
    all_conc_final = []
    
    for (site1, site2) in sample_pairs:
        # Initialize singlet state
        psi = create_singlet_state(N, site1, site2)
        
        # Initial metrics
        rho0 = partial_trace_2site(psi, site1, site2, N)
        all_singlet_init.append(singlet_projection(rho0))
        all_exchange_init.append(exchange_expectation(rho0))
        all_conc_init.append(concurrence(rho0))
        
        # Evolve
        n_steps = int(t_max / dt)
        for _ in range(n_steps):
            psi = trotter_step(psi, onsite, edge, N)
        
        # Final metrics
        rho_f = partial_trace_2site(psi, site1, site2, N)
        all_singlet_final.append(singlet_projection(rho_f))
        all_exchange_final.append(exchange_expectation(rho_f))
        all_conc_final.append(concurrence(rho_f))
    
    # Averages
    singlet_init = np.mean(all_singlet_init)
    singlet_final = np.mean(all_singlet_final)
    exchange_init = np.mean(all_exchange_init)
    exchange_final = np.mean(all_exchange_final)
    conc_init = np.mean(all_conc_init)
    conc_final = np.mean(all_conc_final)
    
    # Survival ratios
    singlet_survival = singlet_final / singlet_init if singlet_init > 0.01 else 0
    # For exchange, initial is -1, want it to stay negative
    exchange_survival = exchange_final / exchange_init if abs(exchange_init) > 0.01 else 0
    conc_survival = conc_final / conc_init if conc_init > 0.01 else 0
    
    # Bandwidth
    bw_eff = min(1.0, bandwidth / coord) if coord > 0 else 1.0
    bw_penalty = max(0, (coord - bandwidth) / bandwidth) if bandwidth > 0 else 0.0
    
    # Combined score
    raw_score = (
        2.0 * singlet_survival +  # Primary metric
        1.0 * conc_survival +     # Entanglement survival
        1.0 * exchange_survival   # Exchange character survival
    )
    fermionic_score = raw_score * bw_eff - bw_penalty
    
    eval_time = time.time() - start_time
    
    return {
        'dims': list(dims),
        'dimension': d,
        'N': N,
        'coordination': coord,
        'bandwidth': bandwidth,
        'pair_distance': actual_distance,
        'n_pairs': len(sample_pairs),
        'singlet_init': singlet_init,
        'singlet_final': singlet_final,
        'singlet_survival': singlet_survival,
        'exchange_init': exchange_init,
        'exchange_final': exchange_final,
        'exchange_survival': exchange_survival,
        'conc_init': conc_init,
        'conc_final': conc_final,
        'conc_survival': conc_survival,
        'bw_efficiency': bw_eff,
        'bw_penalty': bw_penalty,
        'raw_score': raw_score,
        'fermionic_score': fermionic_score,
        'eval_time': eval_time,
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    configs: List[Tuple[int, ...]],
    bandwidths: List[float],
    seeds: List[int] = [0],
    workers: int = 1,
    dt: float = 0.1,
    t_max: float = 3.0,
    pair_distance: int = 2,
    output_file: str = None,
):
    """Run fermionic coherence experiment."""
    
    print("=" * 70)
    print("FERMIONIC COHERENCE EXPERIMENT")
    print("=" * 70)
    print(f"Configs: {configs}")
    print(f"Bandwidths: {bandwidths}")
    print(f"Seeds: {seeds}")
    print(f"Workers: {workers}")
    print(f"Pair distance: {pair_distance}")
    print(f"t_max: {t_max}")
    
    # Build jobs
    jobs = []
    for B in bandwidths:
        for dims in configs:
            for seed in seeds:
                jobs.append((dims, B, seed, dt, t_max, pair_distance))
    
    print(f"Total jobs: {len(jobs)}")
    print("=" * 70)
    
    start = time.time()
    
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(evaluate_single, jobs)
    else:
        results = []
        for i, job in enumerate(jobs):
            print(f"Job {i+1}/{len(jobs)}: {job[0]} B={job[1]} seed={job[2]}...", 
                  end=" ", flush=True)
            r = evaluate_single(job)
            print(f"score={r['fermionic_score']:.3f} ({r['eval_time']:.1f}s)")
            results.append(r)
    
    total_time = time.time() - start
    
    # Phase diagram
    phase_diagram = {}
    for B in bandwidths:
        dim_scores = {}
        for r in results:
            if r['bandwidth'] == B:
                d = r['dimension']
                if d not in dim_scores:
                    dim_scores[d] = []
                dim_scores[d].append(r['fermionic_score'])
        
        dim_means = {d: np.mean(s) for d, s in dim_scores.items()}
        winner = max(dim_means.items(), key=lambda x: x[1])[0] if dim_means else 0
        phase_diagram[B] = {'winner': winner, 'scores': dim_means}
    
    # Print summary
    print("\n" + "=" * 70)
    print("PHASE DIAGRAM")
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
        for r in sorted(B_results, key=lambda x: -x['fermionic_score']):
            print(f"  {r['dimension']}D {tuple(r['dims'])}: score={r['fermionic_score']:.4f}")
            print(f"      singlet: {r['singlet_init']:.3f} → {r['singlet_final']:.3f} "
                  f"(survival={r['singlet_survival']:.3f})")
            print(f"      exchange: {r['exchange_init']:.3f} → {r['exchange_final']:.3f}")
            print(f"      concurrence: {r['conc_init']:.3f} → {r['conc_final']:.3f}")
    
    print(f"\nTotal time: {total_time/3600:.2f} hours")
    
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
                'pair_distance': pair_distance,
                'total_time_hours': total_time / 3600,
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
    
    parser = argparse.ArgumentParser(description="Fermionic Coherence Experiment")
    parser.add_argument('--test', action='store_true', help="N=8")
    parser.add_argument('--small', action='store_true', help="N=16")
    parser.add_argument('--medium', action='store_true', help="N=20")
    parser.add_argument('--full', action='store_true', help="N=24")
    parser.add_argument('--true3d', action='store_true', help="N=27")
    parser.add_argument('--multi', action='store_true', help="Multi-pair mode (JW strings)")
    parser.add_argument('--n-pairs', type=int, default=3, help="Number of disjoint pairs")
    parser.add_argument('--bandwidths', type=str, default=None)
    parser.add_argument('--seeds', type=str, default="0")
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--tmax', type=float, default=3.0)
    parser.add_argument('--pair-distance', type=int, default=2)
    parser.add_argument('--quick', action='store_true', help="Quick mode: shorter t_max for validation")
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
    
    # Quick mode: shorter evolution for validation
    t_max = args.tmax
    if args.quick:
        t_max = 0.5  # Much shorter - just to validate code runs
        print(f"QUICK MODE: t_max={t_max} (for validation only)")
    
    # Memory estimate
    max_N = max(np.prod(c) for c in configs)
    mem_gb = (2**max_N * 16 * 3) / (1024**3)
    print(f"\nMemory estimate: ~{mem_gb:.1f} GB/worker, {mem_gb*args.workers:.1f} GB total")
    
    if args.multi:
        # Multi-pair experiment (tests JW string coexistence)
        run_multi_pair_experiment(
            configs=configs,
            bandwidths=bandwidths,
            seeds=seeds,
            workers=args.workers,
            dt=0.1,
            t_max=args.tmax,
            pair_distance=args.pair_distance,
            n_pairs=args.n_pairs,
            output_file=args.output,
        )
    else:
        # Single-pair experiment
        run_experiment(
            configs=configs,
            bandwidths=bandwidths,
            seeds=seeds,
            workers=args.workers,
            dt=0.1,
            t_max=args.tmax,
            pair_distance=args.pair_distance,
            output_file=args.output,
        )