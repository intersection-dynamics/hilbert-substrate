"""
Step 1: Generate d-dimensional lattice graphs.
Verify coordination numbers and basic properties.
"""

import numpy as np
import networkx as nx
from typing import Tuple, List

def generate_lattice_graph(dims: Tuple[int, ...], periodic: bool = True) -> nx.Graph:
    """
    Generate a d-dimensional lattice graph.
    
    dims: tuple specifying size in each dimension, e.g., (4,4) for 2D, (2,3,4) for 3D
    periodic: whether to use periodic boundary conditions
    
    Returns: networkx Graph with N = prod(dims) nodes
    """
    d = len(dims)
    N = int(np.prod(dims))
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    def to_coords(idx):
        """Flat index -> d-dimensional coordinates"""
        coords = []
        for dim in reversed(dims):
            coords.append(idx % dim)
            idx //= dim
        return tuple(reversed(coords))
    
    def to_idx(coords):
        """d-dimensional coordinates -> flat index"""
        idx = 0
        for i, c in enumerate(coords):
            idx = idx * dims[i] + c
        return idx
    
    # Add edges for nearest neighbors in each dimension
    for node in range(N):
        coords = list(to_coords(node))
        for axis in range(d):
            # Forward neighbor only (to avoid double-counting)
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


def verify_lattice(dims: Tuple[int, ...], periodic: bool = True):
    """Verify lattice properties."""
    d = len(dims)
    N = int(np.prod(dims))
    G = generate_lattice_graph(dims, periodic)
    
    # For periodic: each axis contributes N edges, BUT if dim size is 2,
    # the "forward" and "backward" edges are the same, so only N/2 edges
    if periodic:
        expected_edges = 0
        for dim_size in dims:
            if dim_size == 2:
                expected_edges += N // 2  # Wrap-around creates duplicate
            else:
                expected_edges += N
        # Coordination: 2*d except dims of size 2 contribute only 1 neighbor
        expected_coord = sum(2 if s > 2 else 1 for s in dims)
    else:
        expected_edges = None
        expected_coord = None
    
    degrees = [G.degree(n) for n in G.nodes()]
    
    print(f"\n{d}D lattice: dims={dims}, N={N}, periodic={periodic}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Expected edges (periodic): {expected_edges}")
    print(f"  Degree range: {min(degrees)} - {max(degrees)}")
    print(f"  Expected coordination (periodic): {expected_coord}")
    
    if periodic:
        assert G.number_of_nodes() == N, f"Node count mismatch: {G.number_of_nodes()} != {N}"
        assert G.number_of_edges() == expected_edges, f"Edge count mismatch: {G.number_of_edges()} != {expected_edges}"
        assert all(deg == expected_coord for deg in degrees), f"Degree mismatch: got {set(degrees)}, expected {expected_coord}"
        print("  ✓ All checks passed")
    
    return G


if __name__ == "__main__":
    print("=" * 50)
    print("Verifying d-dimensional lattice generation")
    print("=" * 50)
    
    # N=64: Clean configs for 1D, 2D, 3D comparison
    print("\n--- N=64: Best for comparing 1D vs 2D vs 3D ---")
    test_configs_64 = [
        (64,),           # 1D: coord=2
        (8, 8),          # 2D: coord=4
        (4, 4, 4),       # 3D: coord=6 (true 3D!)
    ]
    
    for dims in test_configs_64:
        verify_lattice(dims, periodic=True)
    
    # N=81: Enables true 4D comparison
    print("\n--- N=81: Enables true 4D (3^4) ---")
    test_configs_81 = [
        (81,),           # 1D: coord=2
        (9, 9),          # 2D: coord=4
        (3, 3, 9),       # 3D: coord=6
        (3, 3, 3, 3),    # 4D: coord=8 (true 4D!)
    ]
    
    for dims in test_configs_81:
        verify_lattice(dims, periodic=True)
    
    print("\n" + "=" * 50)
    print("Key insight: coord = 2*d ONLY when all dims > 2")
    print("Recommended test sizes:")
    print("  N=64: compare 1D, 2D, 3D")
    print("  N=81: compare 1D, 2D, 3D, 4D")
    print("=" * 50)