"""
Jordan-Wigner Inspired Fermionic Coherence Metrics
===================================================

The key insight: Fermionic statistics require maintaining non-local 
correlations (JW strings). In higher dimensions, there's more "room"
for these strings to coexist without interference.

New metrics:
1. Path multiplicity: How many independent paths connect distant sites?
2. String cost: How much does maintaining JW-like correlations cost?
3. Fermionic coherence: Can antisymmetric correlations survive at distance?

Author: Ben Bray
"""

import numpy as np
from scipy.linalg import expm
import networkx as nx
from typing import Dict, List, Tuple
from itertools import combinations
import time

# ============================================================
# LATTICE GRAPHS (from previous)
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
                G.add_edge(node, neighbor := to_idx(new_coords))
    
    return G


# ============================================================
# PATH MULTIPLICITY METRICS
# ============================================================

def count_shortest_paths(G: nx.Graph, source: int, target: int) -> int:
    """Count number of distinct shortest paths between two nodes."""
    try:
        paths = list(nx.all_shortest_paths(G, source, target))
        return len(paths)
    except nx.NetworkXNoPath:
        return 0


def path_multiplicity_score(G: nx.Graph, sample_pairs: int = 50) -> Dict:
    """
    Measure path diversity in the graph.
    
    Higher dimensions have more alternative paths between distant nodes,
    giving JW strings more room to maneuver.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    
    # Sample random pairs at various distances
    distances = dict(nx.all_pairs_shortest_path_length(G))
    
    # Group by distance
    pairs_by_dist = {}
    for i in nodes:
        for j, d in distances[i].items():
            if i < j:  # Avoid duplicates
                if d not in pairs_by_dist:
                    pairs_by_dist[d] = []
                pairs_by_dist[d].append((i, j))
    
    # For each distance, compute average path multiplicity
    path_mult_by_dist = {}
    for dist, pairs in pairs_by_dist.items():
        if dist == 0:
            continue
        # Sample if too many pairs
        sample = pairs[:min(sample_pairs, len(pairs))]
        multiplicities = [count_shortest_paths(G, i, j) for i, j in sample]
        path_mult_by_dist[dist] = {
            'mean': np.mean(multiplicities),
            'max': max(multiplicities),
            'n_pairs': len(sample)
        }
    
    # Overall score: sum of log(multiplicity) weighted by distance
    # This captures the "room for strings" at each scale
    total_score = 0
    for dist, data in path_mult_by_dist.items():
        # Weight by distance - longer strings benefit more from multiplicity
        total_score += dist * np.log1p(data['mean'])
    
    return {
        'by_distance': path_mult_by_dist,
        'total_score': total_score,
        'diameter': max(path_mult_by_dist.keys()) if path_mult_by_dist else 0
    }


def vertex_disjoint_paths(G: nx.Graph, source: int, target: int) -> int:
    """
    Count vertex-disjoint paths (Menger's theorem).
    
    This measures true independence - paths that don't share any nodes.
    Critical for JW strings: independent paths can carry fermionic
    correlations without interference.
    """
    if source == target:
        return 0
    try:
        # node_connectivity gives the min vertex cut = max disjoint paths
        return nx.node_connectivity(G, source, target)
    except:
        return 0


def disjoint_path_score(G: nx.Graph, sample_pairs: int = 30) -> Dict:
    """
    Measure vertex-disjoint path diversity.
    
    This is the key metric for fermionic coherence:
    More disjoint paths = more independent channels for JW strings.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    
    # Sample pairs at distance 2 (next-nearest neighbors)
    # This is where fermionic exchange becomes relevant
    distances = dict(nx.all_pairs_shortest_path_length(G))
    
    pairs_dist_2 = [(i, j) for i in nodes for j in nodes 
                    if i < j and distances[i][j] == 2]
    
    if not pairs_dist_2:
        return {'mean_disjoint': 0, 'score': 0}
    
    sample = pairs_dist_2[:min(sample_pairs, len(pairs_dist_2))]
    disjoint_counts = [vertex_disjoint_paths(G, i, j) for i, j in sample]
    
    mean_disjoint = np.mean(disjoint_counts)
    
    # Score: in d dimensions, expect ~d disjoint paths for adjacent pairs
    # This directly measures "room for JW strings"
    return {
        'mean_disjoint_paths': mean_disjoint,
        'max_disjoint_paths': max(disjoint_counts),
        'score': mean_disjoint  # Higher = better for fermions
    }


# ============================================================
# JORDAN-WIGNER STRING COST
# ============================================================

def jw_string_cost(G: nx.Graph, ordering: List[int] = None) -> Dict:
    """
    Estimate the cost of Jordan-Wigner strings in this geometry.
    
    JW transformation: c_j† = (∏_{k<j} σ_k^z) σ_j^+
    
    The "string" ∏_{k<j} σ_k^z creates non-local correlations.
    In 1D with natural ordering, strings are minimal.
    In higher D, any linear ordering creates long strings.
    
    We measure: average string length for pairs at graph distance d.
    Lower is better (more "local" fermions).
    """
    N = G.number_of_nodes()
    
    # Use natural ordering if not specified
    if ordering is None:
        ordering = list(range(N))
    
    # Position in ordering
    pos = {node: i for i, node in enumerate(ordering)}
    
    # Graph distances
    graph_dist = dict(nx.all_pairs_shortest_path_length(G))
    
    # For each pair, compare graph distance vs ordering distance (string length)
    string_overhead = []
    for i in range(N):
        for j in range(i+1, N):
            g_dist = graph_dist[i][j]
            o_dist = abs(pos[i] - pos[j])  # JW string length
            if g_dist > 0:
                overhead = o_dist / g_dist  # Ratio: how much longer is string vs graph?
                string_overhead.append((g_dist, overhead))
    
    # Group by graph distance
    by_dist = {}
    for g_dist, overhead in string_overhead:
        if g_dist not in by_dist:
            by_dist[g_dist] = []
        by_dist[g_dist].append(overhead)
    
    avg_by_dist = {d: np.mean(ohs) for d, ohs in by_dist.items()}
    
    # Total cost: average overhead weighted by distance
    total_cost = np.mean([oh for _, oh in string_overhead])
    
    return {
        'overhead_by_distance': avg_by_dist,
        'mean_overhead': total_cost,
        'cost_score': -total_cost  # Negative because lower overhead is better
    }


def optimal_jw_ordering(G: nx.Graph, n_attempts: int = 10) -> Tuple[List[int], float]:
    """
    Find a good ordering for JW transformation (minimizes average string length).
    
    Uses BFS from different starting nodes and picks the best.
    """
    N = G.number_of_nodes()
    best_ordering = list(range(N))
    best_cost = jw_string_cost(G, best_ordering)['mean_overhead']
    
    for start in range(min(n_attempts, N)):
        # BFS ordering from this start node
        ordering = list(nx.bfs_tree(G, start).nodes())
        if len(ordering) < N:
            # Add any disconnected nodes
            ordering += [n for n in range(N) if n not in ordering]
        
        cost = jw_string_cost(G, ordering)['mean_overhead']
        if cost < best_cost:
            best_cost = cost
            best_ordering = ordering
    
    return best_ordering, best_cost


# ============================================================
# FERMIONIC SUBSPACE COHERENCE
# ============================================================

def create_fermionic_pair_state(N: int, site1: int, site2: int) -> np.ndarray:
    """
    Create an antisymmetric (fermionic) superposition:
    |ψ⟩ = (|site1⟩ - |site2⟩) / √2
    
    This is the simplest state with fermionic exchange character.
    """
    psi = np.zeros(2**N, dtype=complex)
    # |site1⟩ = single excitation at site1
    psi[2**(N-1-site1)] = 1/np.sqrt(2)
    # -|site2⟩
    psi[2**(N-1-site2)] = -1/np.sqrt(2)
    return psi


def measure_antisymmetric_fidelity(psi: np.ndarray, N: int, site1: int, site2: int) -> float:
    """
    Measure how much the state retains antisymmetric character 
    between site1 and site2.
    
    F = |⟨ψ_antisym | ψ⟩|²
    """
    psi_target = create_fermionic_pair_state(N, site1, site2)
    return np.abs(np.vdot(psi_target, psi))**2


# ============================================================
# COMBINED SCORING WITH JW METRICS
# ============================================================

def compute_jw_aware_scores(dims: Tuple[int, ...]) -> Dict:
    """
    Compute all JW-inspired metrics for a lattice configuration.
    """
    d = len(dims)
    N = int(np.prod(dims))
    G = generate_lattice_graph(dims, periodic=True)
    coord = G.degree(0)
    
    print(f"  Analyzing {d}D {dims} (N={N}, coord={coord})...")
    
    results = {
        'dims': dims,
        'dimension': d,
        'N': N,
        'coordination': coord,
    }
    
    # 1. Path multiplicity
    t0 = time.time()
    path_mult = path_multiplicity_score(G)
    results['path_multiplicity'] = path_mult['total_score']
    results['diameter'] = path_mult['diameter']
    print(f"    Path multiplicity: {path_mult['total_score']:.2f} ({time.time()-t0:.1f}s)")
    
    # 2. Disjoint paths (key for JW)
    t0 = time.time()
    disjoint = disjoint_path_score(G)
    results['disjoint_paths'] = disjoint['mean_disjoint_paths']
    results['disjoint_score'] = disjoint['score']
    print(f"    Mean disjoint paths: {disjoint['mean_disjoint_paths']:.2f} ({time.time()-t0:.1f}s)")
    
    # 3. JW string cost
    t0 = time.time()
    ordering, jw_cost = optimal_jw_ordering(G)
    jw_data = jw_string_cost(G, ordering)
    results['jw_overhead'] = jw_cost
    results['jw_score'] = jw_data['cost_score']
    print(f"    JW string overhead: {jw_cost:.2f} ({time.time()-t0:.1f}s)")
    
    # Combined fermionic viability score
    # Higher disjoint paths + lower JW overhead = better for fermions
    results['fermionic_viability'] = (
        2.0 * disjoint['score'] +  # Disjoint paths matter most
        1.0 * jw_data['cost_score'] +  # Lower overhead is better
        0.5 * np.log1p(path_mult['total_score'])  # Path diversity helps
    )
    
    return results


def compare_dimensions_jw(configs: List[Tuple[int, ...]]) -> List[Dict]:
    """Compare lattice configurations using JW-aware metrics."""
    
    print("=" * 60)
    print("JORDAN-WIGNER AWARE ANALYSIS")
    print("=" * 60)
    
    results = []
    for dims in configs:
        r = compute_jw_aware_scores(dims)
        results.append(r)
    
    # Sort by fermionic viability
    results.sort(key=lambda x: x['fermionic_viability'], reverse=True)
    
    print("\n" + "=" * 60)
    print("RESULTS (sorted by fermionic viability)")
    print("=" * 60)
    print(f"{'Config':<15} {'Dim':<4} {'Coord':<6} {'Disjoint':<10} {'JW Cost':<10} {'Viability':<10}")
    print("-" * 60)
    for r in results:
        print(f"{str(r['dims']):<15} {r['dimension']:<4} {r['coordination']:<6} "
              f"{r['disjoint_paths']:<10.2f} {r['jw_overhead']:<10.2f} "
              f"{r['fermionic_viability']:<10.2f}")
    
    return results


# ============================================================
# THEORETICAL PREDICTIONS
# ============================================================

def theoretical_jw_analysis():
    """
    Theoretical analysis of JW strings in different dimensions.
    """
    print("\n" + "=" * 60)
    print("THEORETICAL JORDAN-WIGNER ANALYSIS")
    print("=" * 60)
    
    print("""
In d dimensions with coordination number z = 2d:

1D (z=2):
   - JW strings are 1D in 1D space
   - Strings FILL the space - every pair has a string between them
   - No room to route around: O(1) disjoint paths
   - Fermions exist but are "expensive" - no stable atoms
   
2D (z=4):  
   - JW strings are 1D in 2D space
   - Strings have room but CROSSINGS create sign problems
   - O(√N) disjoint paths available
   - Fermions possible (anyons too!) but 2D atoms are marginal
   
3D (z=6):
   - JW strings are 1D in 3D space  
   - Strings are MEASURE ZERO - can route around each other
   - O(N^{2/3}) disjoint paths available
   - Knots possible - topological stability
   - FIRST dimension with truly stable fermionic matter
   
4D+ (z=8+):
   - Even more room, but:
   - All knots can be untied (no topological stability)
   - Inverse-square forces don't bind (no stable orbits)
   - Matter is unstable or doesn't form

KEY INSIGHT: 3D is special because:
   1. Enough room for JW strings (unlike 1D, 2D)
   2. Knots provide topological stability (unlike 4D+)
   3. Inverse-square law allows stable orbits (unlike 4D+)
""")
    
    # Quantitative predictions
    print("\nQuantitative predictions for path multiplicity:")
    print("-" * 40)
    for d in range(1, 5):
        z = 2 * d
        # For distance-2 pairs in d dimensions:
        # Number of shortest paths ≈ z
        # Number of disjoint paths ≈ min(z, d)
        print(f"  {d}D: coordination={z}, disjoint paths ≈ {min(z, 2*d-1)}")


# ============================================================
# INTEGRATED SCORING: Original Constraints + JW Metrics
# ============================================================

def integrated_score(
    dims: Tuple[int, ...],
    bandwidth: float,
    jw_weight: float = 1.0,  # Weight for fermionic viability
) -> Dict:
    """
    Combine original four constraints with JW-aware fermionic viability.
    
    The key insight: the original scoring measures information recovery,
    but misses that 3D's "dilution" is actually GOOD for fermions
    because it provides room for JW strings.
    
    New composite:
        score = original_score + jw_weight * fermionic_viability * bandwidth_efficiency
    """
    d = len(dims)
    N = int(np.prod(dims))
    G = generate_lattice_graph(dims, periodic=True)
    coord = G.degree(0)
    
    # Bandwidth efficiency (from original)
    bandwidth_efficiency = min(1.0, bandwidth / coord) if coord > 0 else 1.0
    bandwidth_penalty = max(0, (coord - bandwidth) / bandwidth) if bandwidth > 0 else 0.0
    
    # JW metrics
    disjoint = disjoint_path_score(G)
    ordering, jw_cost = optimal_jw_ordering(G)
    jw_data = jw_string_cost(G, ordering)
    
    fermionic_viability = (
        2.0 * disjoint['score'] +
        1.0 * jw_data['cost_score']
    )
    
    # Effective fermionic score (penalized by bandwidth)
    effective_fermionic = fermionic_viability * bandwidth_efficiency
    
    # Combined score
    # The fermionic term captures what the original recovery metric missed:
    # In 3D, signal "dilutes" into more neighbors, but this ENABLES fermions
    combined_score = (
        jw_weight * effective_fermionic +
        (-1.0) * bandwidth_penalty
    )
    
    return {
        'dims': dims,
        'dimension': d,
        'coordination': coord,
        'bandwidth': bandwidth,
        'bandwidth_efficiency': bandwidth_efficiency,
        'bandwidth_penalty': bandwidth_penalty,
        'disjoint_paths': disjoint['score'],
        'jw_overhead': jw_cost,
        'fermionic_viability': fermionic_viability,
        'effective_fermionic': effective_fermionic,
        'combined_score': combined_score,
    }


def run_jw_integrated_experiment(
    configs: List[Tuple[int, ...]],
    bandwidths: List[float],
    jw_weight: float = 1.0,
):
    """Run experiment with JW-integrated scoring."""
    
    print("=" * 70)
    print("JW-INTEGRATED DIMENSIONALITY EXPERIMENT")
    print(f"JW weight: {jw_weight}")
    print("=" * 70)
    
    phase_diagram = {}
    
    for B in bandwidths:
        print(f"\n--- Bandwidth B = {B} ---")
        results = []
        for dims in configs:
            r = integrated_score(dims, B, jw_weight)
            results.append(r)
            print(f"  {r['dimension']}D {dims}: score={r['combined_score']:.3f} "
                  f"(disjoint={r['disjoint_paths']:.1f}, bw_eff={r['bandwidth_efficiency']:.2f})")
        
        # Find winner
        winner = max(results, key=lambda x: x['combined_score'])
        phase_diagram[B] = {
            'winner': winner['dimension'],
            'scores': {r['dimension']: r['combined_score'] for r in results}
        }
        print(f"  Winner: {winner['dimension']}D")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE DIAGRAM (JW-Integrated)")
    print("=" * 70)
    print(f"{'Bandwidth':<10} {'Winner':<8} {'1D':<10} {'2D':<10} {'3D':<10}")
    print("-" * 50)
    for B in bandwidths:
        w = phase_diagram[B]['winner']
        s = phase_diagram[B]['scores']
        print(f"{B:<10.1f} {w}D{'':<6} {s.get(1, 0):<10.3f} {s.get(2, 0):<10.3f} {s.get(3, 0):<10.3f}")
    
    return phase_diagram


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    # Theoretical background
    theoretical_jw_analysis()
    
    # Small system test (fast)
    print("\n" + "=" * 60)
    print("SMALL SYSTEM TEST (N=8)")
    print("=" * 60)
    configs_8 = [(8,), (4, 2), (2, 2, 2)]
    compare_dimensions_jw(configs_8)
    
    # Medium system (more meaningful)
    print("\n" + "=" * 60)
    print("MEDIUM SYSTEM TEST (N=16)")  
    print("=" * 60)
    configs_16 = [(16,), (4, 4), (2, 2, 4)]
    compare_dimensions_jw(configs_16)
    
    # True comparison (N=27 for real 3D coord=6)
    print("\n" + "=" * 60)
    print("TRUE 3D COMPARISON (N=27)")
    print("=" * 60)
    configs_27 = [(27,), (9, 3), (3, 3, 3)]
    compare_dimensions_jw(configs_27)
    
    # JW-Integrated experiment
    print("\n")
    run_jw_integrated_experiment(
        configs_27,
        bandwidths=[2.0, 4.0, 5.0, 6.0, 8.0],
        jw_weight=1.0
    )