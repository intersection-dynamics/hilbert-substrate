"""
hilbert_substrate_topologies.py

Extended Hilbert Substrate Analysis with Multiple Topologies
============================================================

Supports:
  - 1D: ring, chain, ladder
  - 2D: square_2d, square_2d_open, triangular_2d, honeycomb_2d, kagome_2d
  - 3D: cubic_3d, cubic_3d_open
  - Higher-D: hypercube
  - Random: random_regular, erdos_renyi, watts_strogatz
  - Special: complete, star, binary_tree

USAGE:
    python hilbert_substrate_topologies.py --topology square_2d --L 4
    python hilbert_substrate_topologies.py --topology cubic_3d --L 2
    python hilbert_substrate_topologies.py --topology hypercube --L 4
    python hilbert_substrate_topologies.py --topology random_regular --N 16 --degree 3
    python hilbert_substrate_topologies.py --compare-all --L 3

REQUIREMENTS:
    pip install numpy scipy
"""

import argparse
import numpy as np
from scipy.sparse import csr_matrix, kron as sparse_kron
from scipy.sparse.linalg import eigsh
import time
from typing import List, Tuple, Dict
import json

# =============================================================================
# SPARSE MATRIX INFRASTRUCTURE
# =============================================================================

def sparse_pauli():
    """Return sparse Pauli matrices I, X, Y, Z."""
    I = csr_matrix(np.array([[1, 0], [0, 1]], dtype=complex))
    X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    Y = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
    Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    return I, X, Y, Z


def sparse_kron_n(ops):
    """Sparse Kronecker product of a list of matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = sparse_kron(result, op, format='csr')
    return result


# =============================================================================
# 1D TOPOLOGIES
# =============================================================================

def edges_ring(N: int) -> List[Tuple[int, int]]:
    """1D ring with periodic boundary conditions."""
    return [(i, (i + 1) % N) for i in range(N)]


def edges_chain(N: int) -> List[Tuple[int, int]]:
    """1D chain with open boundary conditions."""
    return [(i, i + 1) for i in range(N - 1)]


def edges_ladder(N: int) -> List[Tuple[int, int]]:
    """Two-leg ladder. N must be even."""
    if N % 2 != 0:
        raise ValueError("Ladder requires even N")
    L = N // 2
    edges = []
    for i in range(L):
        edges.append((i, i + L))
    for i in range(L - 1):
        edges.append((i, i + 1))
        edges.append((i + L, i + L + 1))
    return edges


# =============================================================================
# 2D TOPOLOGIES
# =============================================================================

def edges_square_2d_torus(L: int) -> Tuple[List[Tuple[int, int]], int]:
    """2D square lattice L×L with periodic boundaries (torus)."""
    N = L * L
    edges = []
    for x in range(L):
        for y in range(L):
            site = x * L + y
            right = x * L + ((y + 1) % L)
            down = ((x + 1) % L) * L + y
            edges.append((site, right))
            edges.append((site, down))
    return edges, N


def edges_square_2d_open(L: int) -> Tuple[List[Tuple[int, int]], int]:
    """2D square lattice L×L with open boundaries."""
    N = L * L
    edges = []
    for x in range(L):
        for y in range(L):
            site = x * L + y
            if y + 1 < L:
                edges.append((site, x * L + (y + 1)))
            if x + 1 < L:
                edges.append((site, (x + 1) * L + y))
    return edges, N


def edges_triangular_2d(L: int) -> Tuple[List[Tuple[int, int]], int]:
    """2D triangular lattice L×L with periodic boundaries. Coordination 6."""
    N = L * L
    edges = []
    for x in range(L):
        for y in range(L):
            site = x * L + y
            edges.append((site, x * L + ((y + 1) % L)))
            edges.append((site, ((x + 1) % L) * L + y))
            edges.append((site, ((x + 1) % L) * L + ((y + 1) % L)))
    return edges, N


def edges_honeycomb_2d(L: int) -> Tuple[List[Tuple[int, int]], int]:
    """2D honeycomb lattice with 2*L*L sites. Coordination 3."""
    N = 2 * L * L
    edges = []
    for x in range(L):
        for y in range(L):
            A = 2 * (x * L + y)
            B = A + 1
            edges.append((A, B))
            B_right = 2 * (x * L + ((y + 1) % L)) + 1
            edges.append((A, B_right))
            B_down = 2 * (((x + 1) % L) * L + y) + 1
            edges.append((A, B_down))
    return edges, N


def edges_kagome_2d(L: int) -> Tuple[List[Tuple[int, int]], int]:
    """2D Kagome lattice with 3*L*L sites. Coordination 4."""
    N = 3 * L * L
    edges = []
    for x in range(L):
        for y in range(L):
            base = 3 * (x * L + y)
            s0, s1, s2 = base, base + 1, base + 2
            edges.append((s0, s1))
            edges.append((s1, s2))
            edges.append((s2, s0))
            right_base = 3 * (x * L + ((y + 1) % L))
            down_base = 3 * (((x + 1) % L) * L + y)
            edges.append((s1, right_base))
            edges.append((s2, down_base))
    return edges, N


# =============================================================================
# 3D TOPOLOGIES
# =============================================================================

def edges_cubic_3d_torus(L: int) -> Tuple[List[Tuple[int, int]], int]:
    """3D cubic lattice L×L×L with periodic boundaries."""
    N = L * L * L
    edges = []
    def site(x, y, z): return x * L * L + y * L + z
    for x in range(L):
        for y in range(L):
            for z in range(L):
                s = site(x, y, z)
                edges.append((s, site(x, y, (z + 1) % L)))
                edges.append((s, site(x, (y + 1) % L, z)))
                edges.append((s, site((x + 1) % L, y, z)))
    return edges, N


def edges_cubic_3d_open(L: int) -> Tuple[List[Tuple[int, int]], int]:
    """3D cubic lattice L×L×L with open boundaries."""
    N = L * L * L
    edges = []
    def site(x, y, z): return x * L * L + y * L + z
    for x in range(L):
        for y in range(L):
            for z in range(L):
                s = site(x, y, z)
                if z + 1 < L: edges.append((s, site(x, y, z + 1)))
                if y + 1 < L: edges.append((s, site(x, y + 1, z)))
                if x + 1 < L: edges.append((s, site(x + 1, y, z)))
    return edges, N


# =============================================================================
# HIGHER-D & SPECIAL TOPOLOGIES
# =============================================================================

def edges_hypercube(dim: int) -> Tuple[List[Tuple[int, int]], int]:
    """D-dimensional hypercube with 2^D vertices. Coordination D."""
    N = 2 ** dim
    edges = []
    for i in range(N):
        for bit in range(dim):
            j = i ^ (1 << bit)
            if i < j:
                edges.append((i, j))
    return edges, N


def edges_complete(N: int) -> List[Tuple[int, int]]:
    """Complete graph - all pairs connected."""
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def edges_star(N: int) -> List[Tuple[int, int]]:
    """Star graph - site 0 connected to all others."""
    return [(0, i) for i in range(1, N)]


def edges_binary_tree(depth: int) -> Tuple[List[Tuple[int, int]], int]:
    """Complete binary tree of given depth."""
    N = 2 ** (depth + 1) - 1
    edges = []
    for i in range((N - 1) // 2):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < N: edges.append((i, left))
        if right < N: edges.append((i, right))
    return edges, N


# =============================================================================
# RANDOM TOPOLOGIES
# =============================================================================

def edges_random_regular(N: int, degree: int, seed: int = 42) -> List[Tuple[int, int]]:
    """Random regular graph where each vertex has exactly 'degree' neighbors."""
    if (N * degree) % 2 != 0:
        raise ValueError("N * degree must be even")
    if degree >= N:
        raise ValueError("degree must be less than N")
    
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        stubs = []
        for i in range(N):
            stubs.extend([i] * degree)
        rng.shuffle(stubs)
        
        edges = set()
        valid = True
        for k in range(0, len(stubs), 2):
            a, b = stubs[k], stubs[k + 1]
            if a == b or (min(a, b), max(a, b)) in edges:
                valid = False
                break
            edges.add((min(a, b), max(a, b)))
        if valid:
            return sorted(list(edges))
    raise RuntimeError("Failed to generate random regular graph")


def edges_erdos_renyi(N: int, p: float = 0.3, seed: int = 42) -> List[Tuple[int, int]]:
    """Erdos-Renyi random graph G(N, p)."""
    rng = np.random.default_rng(seed)
    return [(i, j) for i in range(N) for j in range(i + 1, N) if rng.random() < p]


def edges_watts_strogatz(N: int, k: int = 4, p: float = 0.1, seed: int = 42) -> List[Tuple[int, int]]:
    """Watts-Strogatz small-world graph."""
    if k % 2 != 0:
        raise ValueError("k must be even")
    rng = np.random.default_rng(seed)
    edges = set()
    for i in range(N):
        for j in range(1, k // 2 + 1):
            edges.add((min(i, (i+j)%N), max(i, (i+j)%N)))
    edges_list = list(edges)
    for i, j in edges_list:
        if rng.random() < p:
            new_j = rng.integers(0, N)
            while new_j == i or (min(i, new_j), max(i, new_j)) in edges:
                new_j = rng.integers(0, N)
            edges.discard((i, j))
            edges.add((min(i, new_j), max(i, new_j)))
    return sorted(list(edges))


# =============================================================================
# TOPOLOGY FACTORY
# =============================================================================

def get_edges(topology: str, **kwargs) -> Tuple[List[Tuple[int, int]], int, Dict]:
    """Factory function to get edges for any topology."""
    info = {'topology': topology}
    
    if topology == 'ring':
        N = kwargs.get('N', kwargs.get('L', 16))
        return edges_ring(N), N, {**info, 'N': N}
    if topology == 'chain':
        N = kwargs.get('N', kwargs.get('L', 16))
        return edges_chain(N), N, {**info, 'N': N}
    if topology == 'ladder':
        N = kwargs.get('N', kwargs.get('L', 16))
        return edges_ladder(N), N, {**info, 'N': N}
    if topology in ['square_2d', 'square_2d_torus', 'torus_2d']:
        L = kwargs.get('L', 4)
        edges, N = edges_square_2d_torus(L)
        return edges, N, {**info, 'L': L, 'shape': f'{L}x{L}'}
    if topology == 'square_2d_open':
        L = kwargs.get('L', 4)
        edges, N = edges_square_2d_open(L)
        return edges, N, {**info, 'L': L, 'shape': f'{L}x{L}'}
    if topology == 'triangular_2d':
        L = kwargs.get('L', 4)
        edges, N = edges_triangular_2d(L)
        return edges, N, {**info, 'L': L, 'coordination': 6}
    if topology == 'honeycomb_2d':
        L = kwargs.get('L', 4)
        edges, N = edges_honeycomb_2d(L)
        return edges, N, {**info, 'L': L, 'coordination': 3}
    if topology == 'kagome_2d':
        L = kwargs.get('L', 3)
        edges, N = edges_kagome_2d(L)
        return edges, N, {**info, 'L': L, 'coordination': 4}
    if topology in ['cubic_3d', 'cubic_3d_torus']:
        L = kwargs.get('L', 2)
        edges, N = edges_cubic_3d_torus(L)
        return edges, N, {**info, 'L': L, 'shape': f'{L}³'}
    if topology == 'cubic_3d_open':
        L = kwargs.get('L', 2)
        edges, N = edges_cubic_3d_open(L)
        return edges, N, {**info, 'L': L, 'shape': f'{L}³'}
    if topology == 'hypercube':
        dim = kwargs.get('L', kwargs.get('dim', 4))
        edges, N = edges_hypercube(dim)
        return edges, N, {**info, 'dimension': dim}
    if topology == 'complete':
        N = kwargs.get('N', kwargs.get('L', 8))
        return edges_complete(N), N, {**info, 'N': N}
    if topology == 'star':
        N = kwargs.get('N', kwargs.get('L', 16))
        return edges_star(N), N, {**info, 'N': N}
    if topology == 'binary_tree':
        depth = kwargs.get('L', kwargs.get('depth', 3))
        edges, N = edges_binary_tree(depth)
        return edges, N, {**info, 'depth': depth}
    if topology == 'random_regular':
        N = kwargs.get('N', 16)
        degree = kwargs.get('degree', 3)
        edges = edges_random_regular(N, degree, kwargs.get('seed', 42))
        return edges, N, {**info, 'N': N, 'degree': degree}
    if topology == 'erdos_renyi':
        N = kwargs.get('N', 16)
        p = kwargs.get('p', 0.3)
        edges = edges_erdos_renyi(N, p, kwargs.get('seed', 42))
        return edges, N, {**info, 'N': N, 'p': p}
    if topology == 'watts_strogatz':
        N = kwargs.get('N', 16)
        k = kwargs.get('k', 4)
        p = kwargs.get('p', 0.1)
        edges = edges_watts_strogatz(N, k, p, kwargs.get('seed', 42))
        return edges, N, {**info, 'N': N, 'k': k, 'p': p}
    raise ValueError(f"Unknown topology: {topology}")


# =============================================================================
# PHYSICS COMPUTATIONS
# =============================================================================

def heisenberg_hamiltonian(N: int, edges: List[Tuple[int, int]], J: float = 1.0) -> csr_matrix:
    """Build sparse Heisenberg XXX Hamiltonian."""
    I, X, Y, Z = sparse_pauli()
    dim = 2 ** N
    H = csr_matrix((dim, dim), dtype=complex)
    for (i, j) in edges:
        for pauli in [X, Y, Z]:
            ops = [I] * N
            ops[i] = pauli
            ops[j] = pauli
            H = H + J * sparse_kron_n(ops)
    return 0.5 * (H + H.conj().T)


def get_ground_state(H: csr_matrix, k: int = 6):
    """Get lowest k eigenvalues/vectors using Lanczos."""
    dim = H.shape[0]
    k = min(k, dim - 2)
    energies, states = eigsh(H, k=k, which='SA', return_eigenvectors=True)
    idx = np.argsort(energies)
    return energies[idx], states[:, idx]


def graph_distance_matrix(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Shortest-path distances (Floyd-Warshall)."""
    D = np.full((N, N), np.inf)
    np.fill_diagonal(D, 0)
    for (i, j) in edges:
        D[i, j] = D[j, i] = 1
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
    return D


def compute_correlations(N: int, H: csr_matrix, ground: np.ndarray) -> np.ndarray:
    """Spin-spin correlations."""
    I, X, Y, Z = sparse_pauli()
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if i == j:
                C[i, j] = 1.0
                continue
            corr = 0.0
            for pauli in [X, Y, Z]:
                ops = [I] * N
                ops[i] = pauli
                ops[j] = pauli
                O = sparse_kron_n(ops)
                corr += np.abs(ground.conj() @ (O @ ground))
            C[i, j] = corr / 3.0
            C[j, i] = C[i, j]
    return C


def estimate_dimension(D: np.ndarray) -> float:
    """MDS-based effective dimension."""
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    eigs = np.linalg.eigvalsh(B)[::-1]
    pos = eigs[eigs > 1e-10]
    if len(pos) == 0: return 0.0
    norm = pos / np.sum(pos)
    return 1.0 / np.sum(norm ** 2)


def compute_forces(N: int, H: csr_matrix, ground: np.ndarray, graph_D: np.ndarray) -> Dict[int, float]:
    """Interaction potential V(d) vs graph distance."""
    I, X = sparse_pauli()[0], sparse_pauli()[1]
    E_ground = float(np.real(ground.conj() @ (H @ ground)))
    
    single_E = []
    for site in range(N):
        ops = [I] * N
        ops[site] = X
        psi = sparse_kron_n(ops) @ ground
        psi = psi / np.linalg.norm(psi)
        single_E.append(float(np.real(psi.conj() @ (H @ psi))))
    
    V_by_d = {}
    for i in range(N):
        for j in range(i + 1, N):
            d = int(graph_D[i, j])
            if d == np.inf: continue
            ops1 = [I] * N; ops1[i] = X
            ops2 = [I] * N; ops2[j] = X
            psi = sparse_kron_n(ops2) @ (sparse_kron_n(ops1) @ ground)
            psi = psi / np.linalg.norm(psi)
            E_ij = float(np.real(psi.conj() @ (H @ psi)))
            V = E_ij - single_E[i] - single_E[j] + E_ground
            V_by_d.setdefault(d, []).append(V)
    
    return {d: float(np.mean(vs)) for d, vs in V_by_d.items()}


def check_locality(V_d: Dict[int, float], threshold: float = 0.1) -> bool:
    """Check if forces are local."""
    V1 = abs(V_d.get(1, 0))
    V_rest = sum(abs(V_d.get(d, 0)) for d in V_d if d > 1)
    return V_rest < threshold * V1 if V1 > 1e-10 else True


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(topology: str, verbose: bool = True, **kwargs) -> Dict:
    """Run complete analysis for given topology."""
    edges, N, topo_info = get_edges(topology, **kwargs)
    
    results = {'topology': topology, 'N': N, 'dim': 2**N, 'num_edges': len(edges), 'info': topo_info}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TOPOLOGY: {topology}")
        print(f"Sites: {N}, Hilbert dim: {2**N:,}, Edges: {len(edges)}")
        print(f"{'='*70}")
    
    t_start = time.time()
    
    if verbose: print(f"\n[1/4] Building Hamiltonian...")
    H = heisenberg_hamiltonian(N, edges)
    graph_D = graph_distance_matrix(N, edges)
    results['coordination'] = 2 * len(edges) / N
    if verbose: print(f"    Coordination: {results['coordination']:.2f}")
    
    if verbose: print(f"\n[2/4] Ground state...")
    t1 = time.time()
    energies, states = get_ground_state(H, k=min(6, 2**N - 2))
    ground = states[:, 0]
    results['ground_energy'] = float(energies[0])
    results['gap'] = float(energies[1] - energies[0]) if len(energies) > 1 else 0
    if verbose: print(f"    Time: {time.time()-t1:.2f}s, E_0={energies[0]:.4f}, Gap={results['gap']:.4f}")
    
    if verbose: print(f"\n[3/4] Correlations...")
    t1 = time.time()
    C = compute_correlations(N, H, ground)
    D_corr = np.zeros_like(C)
    mask = C > 1e-10
    D_corr[mask] = 1.0 / C[mask]
    np.fill_diagonal(D_corr, 0)
    D_corr = D_corr / np.max(D_corr) if np.max(D_corr) > 0 else D_corr
    results['eff_dim'] = estimate_dimension(D_corr)
    if verbose: print(f"    Time: {time.time()-t1:.2f}s, Eff.dim: {results['eff_dim']:.2f}")
    
    if verbose: print(f"\n[4/4] Forces...")
    t1 = time.time()
    V_d = compute_forces(N, H, ground, graph_D)
    results['V_vs_d'] = {int(k): float(v) for k, v in V_d.items()}
    results['local'] = check_locality(V_d)
    
    if verbose:
        print(f"    Time: {time.time()-t1:.2f}s")
        print(f"\n    V(d):")
        for d in sorted(V_d.keys())[:8]:
            V = V_d[d]
            bar = "█" * int(abs(V) * 2) if abs(V) > 0.01 else ""
            print(f"      d={d}: V={V:+.6f}  {bar}")
        print(f"\n    FORCES: {'LOCAL ✓' if results['local'] else 'NON-LOCAL ✗'}")
    
    results['time'] = time.time() - t_start
    return results


def compare_topologies(topologies: List[Tuple[str, Dict]], verbose: bool = True):
    """Compare multiple topologies."""
    print("\n" + "="*80)
    print("TOPOLOGY COMPARISON")
    print("="*80)
    
    all_results = []
    for topo, kwargs in topologies:
        try:
            r = run_analysis(topo, verbose=verbose, **kwargs)
            all_results.append(r)
        except Exception as e:
            print(f"\n*** {topo}: {e} ***")
            all_results.append({'topology': topo, 'error': str(e)})
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Topology':<18} | {'N':>4} | {'Dim':>7} | {'Coord':>5} | {'E.Dim':>5} | {'V(1)':>8} | {'V(2)':>8} | Loc")
    print("-" * 82)
    
    for r in all_results:
        if 'error' in r:
            print(f"{r['topology']:<18} | ERROR")
        else:
            V1 = r['V_vs_d'].get(1, 0)
            V2 = r['V_vs_d'].get(2, 0)
            print(f"{r['topology']:<18} | {r['N']:>4} | {r['dim']:>7,} | {r['coordination']:>5.1f} | "
                  f"{r['eff_dim']:>5.1f} | {V1:>+8.3f} | {V2:>+8.3f} | {'✓' if r['local'] else '✗'}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Hilbert Substrate - Extended Topologies")
    parser.add_argument('--topology', type=str, default='ring')
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--L', type=int, default=None)
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--compare-all', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    kwargs = {'degree': args.degree}
    if args.N: kwargs['N'] = args.N
    if args.L: kwargs['L'] = args.L
    
    if args.compare_all:
        L = args.L or 3
        N = args.N or 12
        topologies = [
            ('ring', {'N': N}),
            ('chain', {'N': N}),
            ('square_2d', {'L': L}),
            ('triangular_2d', {'L': L}),
            ('honeycomb_2d', {'L': L}),
            ('cubic_3d', {'L': 2}),
            ('hypercube', {'L': 4}),
            ('random_regular', {'N': N, 'degree': 3}),
            ('complete', {'N': min(N, 8)}),
        ]
        results = compare_topologies(topologies)
    else:
        results = run_analysis(args.topology, **kwargs)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()