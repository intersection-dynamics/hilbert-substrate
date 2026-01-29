"""
HSF Dimensionality Selection Experiment
=======================================

Tests whether the four constraints (no-signaling, no-forgetting, 
no-refolding, finite bandwidth) select for 3D over other dimensions.

Usage:
    python hsf_dimensionality_experiment.py

Recommended test configurations:
    N=64:  1D(64), 2D(8x8), 3D(4x4x4)
    N=81:  1D(81), 2D(9x9), 3D(3x3x9), 4D(3x3x3x3)

Author: Ben Bray
"""

import numpy as np
from scipy.linalg import expm
import networkx as nx
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import json
import time

# ============================================================
# LATTICE GRAPHS
# ============================================================

def generate_lattice_graph(dims: Tuple[int, ...], periodic: bool = True) -> nx.Graph:
    """Generate d-dimensional periodic lattice graph."""
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
                neighbor = to_idx(new_coords)
                G.add_edge(node, neighbor)
            else:
                if coords[axis] + 1 < dims[axis]:
                    new_coords[axis] = coords[axis] + 1
                    neighbor = to_idx(new_coords)
                    G.add_edge(node, neighbor)
    
    return G


# ============================================================
# TROTTER EVOLUTION
# ============================================================

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
PAULIS = [I, X, Y, Z]


def build_trotter_gates(
    G: nx.Graph, N: int, dt: float,
    h_std: float = 0.25, J_std: float = 0.8, seed: int = 0
) -> Tuple[Dict, Dict]:
    """Build Trotter gates for one time step."""
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


def apply_single_qubit_gate(psi: np.ndarray, qubit: int, gate: np.ndarray, N: int) -> np.ndarray:
    """Apply 2x2 gate to qubit in N-qubit state."""
    left_dim = 2**qubit
    right_dim = 2**(N - qubit - 1)
    psi_reshaped = psi.reshape(left_dim, 2, right_dim)
    result = np.einsum('ab,lbr->lar', gate, psi_reshaped)
    return result.reshape(-1)


def apply_two_qubit_gate(psi: np.ndarray, q1: int, q2: int, gate: np.ndarray, N: int) -> np.ndarray:
    """Apply 4x4 gate to two qubits."""
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


def trotter_step(psi: np.ndarray, onsite: Dict, edge: Dict, N: int) -> np.ndarray:
    """Apply one Trotter step."""
    for qubit, gate in onsite.items():
        psi = apply_single_qubit_gate(psi, qubit, gate, N)
    for (q1, q2), gate in edge.items():
        psi = apply_two_qubit_gate(psi, q1, q2, gate, N)
    return psi / np.linalg.norm(psi)


# ============================================================
# INFLUENCE MEASUREMENT (MEMORY-EFFICIENT)
# ============================================================

def partial_trace_efficient(psi: np.ndarray, keep_qubits: List[int], N: int) -> np.ndarray:
    """
    Compute reduced density matrix WITHOUT forming full 2^N × 2^N matrix.
    
    Memory: O(2^N) for state + O(2^(2k)) for output, where k = len(keep_qubits)
    """
    k = len(keep_qubits)
    trace_qubits = sorted([q for q in range(N) if q not in keep_qubits])
    keep_qubits = sorted(keep_qubits)
    
    # Reshape psi into tensor with shape [2]*N
    psi_tensor = psi.reshape([2] * N)
    
    # Reorder axes: keep_qubits first, then trace_qubits
    new_order = keep_qubits + trace_qubits
    psi_reordered = np.transpose(psi_tensor, new_order)
    
    # Reshape to (2^k, 2^(N-k))
    dim_keep = 2**k
    dim_trace = 2**(N - k)
    psi_matrix = psi_reordered.reshape(dim_keep, dim_trace)
    
    # rho_reduced = sum over traced indices of |psi><psi|
    # rho[i,j] = sum_a psi[i,a] * psi[j,a].conj()
    # This is just psi_matrix @ psi_matrix.conj().T
    rho = psi_matrix @ psi_matrix.conj().T
    
    return rho


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Trace distance: D = (1/2) ||rho1 - rho2||_1"""
    diff = rho1 - rho2
    sv = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(sv)


def compute_influence(psi: np.ndarray, psi_pert: np.ndarray, block: List[int], N: int) -> float:
    """Compute influence of perturbation on a block."""
    rho = partial_trace_efficient(psi, block, N)
    rho_pert = partial_trace_efficient(psi_pert, block, N)
    return trace_distance(rho, rho_pert)


def apply_pauli_x(psi: np.ndarray, qubit: int, N: int) -> np.ndarray:
    """Apply Pauli X to qubit."""
    left_dim = 2**qubit
    right_dim = 2**(N - qubit - 1)
    psi_reshaped = psi.reshape(left_dim, 2, right_dim)
    result = np.einsum('ab,lbr->lar', X, psi_reshaped)
    return result.reshape(-1)


# ============================================================
# CONSTRAINT SCORING
# ============================================================

@dataclass
class ConstraintScores:
    dims: Tuple[int, ...]
    dimension: int
    N: int
    coordination: int
    n_edges: int
    bandwidth: float
    
    # Individual scores
    no_signaling_score: float
    no_forgetting_score: float
    no_refolding_score: float
    bandwidth_score: float
    
    # Raw metrics
    early_violation: float
    late_recovery: float
    recovery_strength: float
    bandwidth_efficiency: float
    
    # Composite
    composite_score: float
    
    # Timing
    eval_time: float
    
    def summary(self) -> str:
        return (
            f"{self.dimension}D {self.dims} (coord={self.coordination}): "
            f"score={self.composite_score:.3f} "
            f"[sig={self.no_signaling_score:.2f}, "
            f"fgt={self.no_forgetting_score:.2f}, "
            f"rfl={self.no_refolding_score:.2f}, "
            f"bw={self.bandwidth_score:.2f}] "
            f"({self.eval_time:.1f}s)"
        )


def evaluate_topology(
    dims: Tuple[int, ...],
    bandwidth: float,
    seed: int = 0,
    dt: float = 0.1,
    t_max: float = 3.0,
    n_time_points: int = 31,
    recovery_threshold: float = 0.03,
    speed_threshold: float = 0.02,
    verbose: bool = False,
) -> ConstraintScores:
    """
    Evaluate a topology under the four HSF constraints.
    
    Args:
        dims: lattice dimensions, e.g., (64,) for 1D, (8,8) for 2D, (4,4,4) for 3D
        bandwidth: bandwidth capacity B
        seed: random seed
        dt: Trotter step size
        t_max: maximum evolution time
        n_time_points: number of time samples
        recovery_threshold: influence threshold for "recovered"
        speed_threshold: influence threshold for signaling violation
        verbose: print progress
    """
    start_time = time.time()
    
    d = len(dims)
    N = int(np.prod(dims))
    G = generate_lattice_graph(dims, periodic=True)
    coord = G.degree(0)
    n_edges = G.number_of_edges()
    
    if verbose:
        print(f"    Building Trotter gates ({n_edges} edges)...")
    
    onsite, edge = build_trotter_gates(G, N, dt, seed=seed)
    
    # Initial state: |000...0>
    psi0 = np.zeros(2**N, dtype=complex)
    psi0[0] = 1.0
    
    # Perturbed state: X on qubit 0
    psi0_pert = apply_pauli_x(psi0, qubit=0, N=N)
    
    # Graph distances from source
    distances = dict(nx.single_source_shortest_path_length(G, 0))
    
    # Time evolution
    times = np.linspace(0, t_max, n_time_points)
    psi = psi0.copy()
    psi_pert = psi0_pert.copy()
    
    influence_history = {q: [] for q in range(N)}
    current_t = 0.0
    
    if verbose:
        print(f"    Evolving to t={t_max}...")
    
    for ti, t_target in enumerate(times):
        while current_t < t_target - 1e-9:
            psi = trotter_step(psi, onsite, edge, N)
            psi_pert = trotter_step(psi_pert, onsite, edge, N)
            current_t += dt
        
        # Measure influence on each qubit
        for q in range(N):
            inf = compute_influence(psi, psi_pert, [q], N)
            influence_history[q].append(inf)
        
        if verbose and (ti + 1) % 10 == 0:
            print(f"      t={current_t:.1f}/{t_max}")
    
    # === NO-SIGNALING SCORE ===
    t_early = 0.3
    early_idx = np.argmin(np.abs(times - t_early))
    early_violation = 0.0
    n_distant = sum(1 for q in range(N) if distances[q] > 1)
    for q in range(N):
        if distances[q] > 1:
            if influence_history[q][early_idx] > speed_threshold:
                early_violation += influence_history[q][early_idx]
    early_violation /= max(1, n_distant)
    no_signaling_score = -2.0 * early_violation
    
    # === NO-FORGETTING SCORE ===
    t_late_start = t_max * 0.5
    neighbors = [q for q in range(N) if distances[q] == 1]
    late_indices = [i for i, t in enumerate(times) if t >= t_late_start]
    
    recovery_count = 0
    recovery_sum = 0.0
    total = len(late_indices) * len(neighbors)
    
    for t_idx in late_indices:
        for q in neighbors:
            inf = influence_history[q][t_idx]
            recovery_sum += inf
            if inf > recovery_threshold:
                recovery_count += 1
    
    late_recovery = recovery_count / max(1, total)
    recovery_strength = recovery_sum / max(1, total)
    no_forgetting_score = 2.0 * late_recovery + 0.8 * np.tanh(5 * recovery_strength)
    
    # === NO-REFOLDING SCORE ===
    try:
        vertex_conn = nx.node_connectivity(G)
        diameter = nx.diameter(G)
        complexity = vertex_conn / max(1, diameter)
    except:
        complexity = 0.0
    
    no_refolding_score = -2.0 if complexity < 0.3 else 0.0
    
    # === BANDWIDTH SCORE ===
    bandwidth_efficiency = min(1.0, bandwidth / coord) if coord > 0 else 1.0
    bandwidth_penalty = max(0, (coord - bandwidth) / bandwidth) if bandwidth > 0 else 0.0
    bandwidth_score = -1.0 * bandwidth_penalty
    
    # === COMPOSITE SCORE ===
    effective_forgetting = no_forgetting_score * bandwidth_efficiency
    composite_score = (
        no_signaling_score +
        effective_forgetting +
        no_refolding_score +
        bandwidth_score
    )
    
    eval_time = time.time() - start_time
    
    return ConstraintScores(
        dims=dims,
        dimension=d,
        N=N,
        coordination=coord,
        n_edges=n_edges,
        bandwidth=bandwidth,
        no_signaling_score=no_signaling_score,
        no_forgetting_score=no_forgetting_score,
        no_refolding_score=no_refolding_score,
        bandwidth_score=bandwidth_score,
        early_violation=early_violation,
        late_recovery=late_recovery,
        recovery_strength=recovery_strength,
        bandwidth_efficiency=bandwidth_efficiency,
        composite_score=composite_score,
        eval_time=eval_time,
    )


# ============================================================
# EXPERIMENTS
# ============================================================

def run_dimensionality_experiment(
    configs: List[Tuple[int, ...]],
    bandwidths: List[float],
    seeds: List[int] = [0],
    t_max: float = 3.0,
    n_time_points: int = 31,
    dt: float = 0.1,
    output_file: str = None,
):
    """
    Run dimensionality selection experiment.
    
    Args:
        configs: list of dimension tuples, e.g., [(64,), (8,8), (4,4,4)]
        bandwidths: list of bandwidth values to test
        seeds: random seeds for averaging
        t_max: evolution time
        n_time_points: time samples
        dt: Trotter step
        output_file: JSON file to save results
    """
    print("=" * 70)
    print("HSF DIMENSIONALITY SELECTION EXPERIMENT")
    print("=" * 70)
    
    all_results = []
    phase_diagram = {}
    
    for B in bandwidths:
        print(f"\n{'='*70}")
        print(f"BANDWIDTH B = {B}")
        print(f"{'='*70}")
        
        dim_scores = {}  # {dimension: [scores across seeds]}
        
        for dims in configs:
            d = len(dims)
            N = int(np.prod(dims))
            
            print(f"\n  {d}D lattice: dims={dims}, N={N}")
            
            scores_for_config = []
            for seed in seeds:
                print(f"    seed={seed}...", end=" ", flush=True)
                score = evaluate_topology(
                    dims, B, seed=seed,
                    dt=dt, t_max=t_max, n_time_points=n_time_points
                )
                scores_for_config.append(score)
                print(f"score={score.composite_score:.3f} ({score.eval_time:.1f}s)")
                
                all_results.append({
                    'bandwidth': B,
                    'dims': dims,
                    'dimension': d,
                    'seed': seed,
                    **asdict(score)
                })
            
            # Average score for this dimension
            mean_score = np.mean([s.composite_score for s in scores_for_config])
            if d not in dim_scores:
                dim_scores[d] = []
            dim_scores[d].append(mean_score)
        
        # Find winner for this bandwidth
        dim_means = {d: np.mean(scores) for d, scores in dim_scores.items()}
        winner_dim = max(dim_means.items(), key=lambda x: x[1])[0]
        
        print(f"\n  --- B={B} Results ---")
        for d in sorted(dim_means.keys()):
            marker = " <-- WINNER" if d == winner_dim else ""
            print(f"    {d}D: mean_score = {dim_means[d]:.3f}{marker}")
        
        phase_diagram[B] = {
            'winner': winner_dim,
            'scores_by_dim': dim_means
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE DIAGRAM SUMMARY")
    print("=" * 70)
    print(f"{'Bandwidth':<12} {'Winner':<8} {'1D':<10} {'2D':<10} {'3D':<10} {'4D':<10}")
    print("-" * 70)
    for B in bandwidths:
        winner = phase_diagram[B]['winner']
        scores = phase_diagram[B]['scores_by_dim']
        row = f"{B:<12.1f} {winner}D{'':<6}"
        for d in [1, 2, 3, 4]:
            if d in scores:
                row += f" {scores[d]:<10.3f}"
            else:
                row += f" {'--':<10}"
        print(row)
    
    # Save results
    if output_file:
        output = {
            'phase_diagram': {str(k): v for k, v in phase_diagram.items()},
            'all_results': all_results,
            'config': {
                'configs': [list(c) for c in configs],
                'bandwidths': bandwidths,
                'seeds': seeds,
                't_max': t_max,
                'n_time_points': n_time_points,
                'dt': dt,
            }
        }
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    
    return phase_diagram, all_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HSF Dimensionality Selection Experiment")
    parser.add_argument('--test', action='store_true', help="Quick test with N=8 (~1MB)")
    parser.add_argument('--small', action='store_true', help="Small experiment with N=16 (~5MB)")
    parser.add_argument('--medium', action='store_true', help="Medium experiment with N=20 (~80MB)")
    parser.add_argument('--full', action='store_true', help="Full experiment with N=24 (~1.2GB)")
    parser.add_argument('--true3d', action='store_true', help="N=27 with true 3D coord=6 (~10GB)")
    parser.add_argument('--include-4d', action='store_true', help="Include 4D in comparison")
    parser.add_argument('--bandwidths', type=str, default=None, 
                        help="Comma-separated bandwidth values")
    parser.add_argument('--seeds', type=str, default="0",
                        help="Comma-separated seeds for averaging")
    parser.add_argument('--output', type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(',')]
    
    if args.test:
        # Quick test: N=8, ~1MB
        configs = [(8,), (4, 2)]
        bandwidths = [2.0, 3.0, 4.0]
        t_max, n_time_points, dt = 2.0, 21, 0.1
        
    elif args.small:
        # Small: N=16, ~5MB
        # 1D: coord=2, 2D(4x4): coord=4
        configs = [(16,), (4, 4)]
        bandwidths = [2.0, 3.0, 4.0, 5.0, 6.0]
        t_max, n_time_points, dt = 2.5, 26, 0.1
        
    elif args.medium:
        # Medium: N=20, ~80MB
        # 1D: coord=2, 2D(5x4): coord=4, 3D(2x2x5): coord=4 (dim-2 issue)
        configs = [(20,), (5, 4), (2, 2, 5)]
        bandwidths = [2.0, 3.0, 4.0, 5.0, 6.0]
        t_max, n_time_points, dt = 3.0, 31, 0.1
        
    elif args.full:
        # Full: N=24, ~1.2GB
        # 1D(24): coord=2, 2D(6x4): coord=4, 3D(2x3x4): coord=5
        configs = [(24,), (6, 4), (2, 3, 4)]
        bandwidths = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        t_max, n_time_points, dt = 3.0, 31, 0.1
        
        if args.include_4d:
            # 4D(2x2x2x3): coord=5 (dim-2 issues reduce it)
            configs.append((2, 2, 2, 3))
    
    elif args.true3d:
        # TRUE 3D: N=27, ~10GB
        # 1D(27): coord=2, 2D(9x3): coord=4, 3D(3x3x3): coord=6 TRUE!
        configs = [(27,), (9, 3), (3, 3, 3)]
        bandwidths = [2.0, 4.0, 6.0, 8.0]
        t_max, n_time_points, dt = 3.0, 31, 0.1
            
    else:
        # Default: same as test
        configs = [(8,), (4, 2)]
        bandwidths = [2.0, 3.0, 4.0]
        t_max, n_time_points, dt = 2.0, 21, 0.1
    
    # Override bandwidths if specified
    if args.bandwidths:
        bandwidths = [float(b) for b in args.bandwidths.split(',')]
    
    # Run experiment
    run_dimensionality_experiment(
        configs=configs,
        bandwidths=bandwidths,
        seeds=seeds,
        t_max=t_max,
        n_time_points=n_time_points,
        dt=dt,
        output_file=args.output,
    )