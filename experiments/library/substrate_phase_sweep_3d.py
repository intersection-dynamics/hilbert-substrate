"""
substrate_phase_sweep_3d.py

Parameter sweep searching for emergent 3D geometry.
We vary Vacuum Stiffness (Beta) against a fixed "Volumetric" Gravity (Alpha).

Gravity Update:
- Old: Reward Triangles (2D Simplex).
- New: Reward Tetrahedra (3D Simplex).

Hypothesis:
- The "Critical Point" dimension should shift from ~2.0 to ~3.0.
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
N_NODES = 45           # Size of Universe
STEPS_PER_RUN = 1200   # Monte Carlo steps
BETA_VALUES = np.linspace(0.0, 6.0, 25) # Extended range
ALPHA_GRAVITY = 1.0    # Fixed attractive force for Tetrahedra
TEMP = 0.35            # Slightly lower temp to help "freeze" complex shapes

def count_tetrahedra(G):
    """
    Counts the number of 4-cliques (Tetrahedra) in the graph.
    Algorithm: 
    1. Iterate over all edges (u, v).
    2. Find common neighbors (CN).
    3. Count edges existing within the subgraph induced by CN.
    4. Total count is Sum(Edges_in_CN) / 6 (since each tet has 6 edges).
    """
    if G.number_of_edges() < 6:
        return 0
    
    total_cn_edges = 0
    # Pre-compute adjacency for speed
    adj = {n: set(G[n]) for n in G}
    
    for u, v in G.edges():
        # Nodes connected to BOTH u and v
        common = list(adj[u].intersection(adj[v]))
        if len(common) < 2:
            continue
            
        # Count edges between common neighbors
        # This checks if w, z in Common are connected
        # If yes, u-v-w-z forms a tetrahedron
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                w, z = common[i], common[j]
                if G.has_edge(w, z):
                    total_cn_edges += 1
                    
    return total_cn_edges / 6.0

def get_energies(G, alpha, beta):
    if not nx.is_connected(G):
        return float('inf'), float('inf')

    # 1. Gravitational Potential (Volumetric)
    tet_count = count_tetrahedra(G)
    E_grav = -alpha * tet_count

    # 2. Casimir Potential (Spectral ZPE)
    evals = nx.laplacian_spectrum(G)
    evals = evals[evals > 1e-5]
    E_zpe = beta * np.sum(np.sqrt(evals))

    return E_grav, E_zpe

def get_spectral_dimension(G, t_probe=2.0):
    evals = nx.laplacian_spectrum(G)
    dt = 0.5
    K_t1 = np.sum(np.exp(-evals * t_probe))
    K_t2 = np.sum(np.exp(-evals * (t_probe + dt)))
    
    # Avoid log(0)
    if K_t1 <= 0 or K_t2 <= 0: return 0.0
    
    slope = (np.log(K_t2) - np.log(K_t1)) / (np.log(t_probe + dt) - np.log(t_probe))
    return -2 * slope

def run_simulation(alpha, beta):
    # Initialize: Random Hot Universe
    G = nx.erdos_renyi_graph(N_NODES, p=0.20)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(N_NODES, p=0.20)
        
    E_grav, E_zpe = get_energies(G, alpha, beta)
    E_total = E_grav + E_zpe
    
    # Evolution Loop
    for step in range(STEPS_PER_RUN):
        G_new = G.copy()
        nodes = list(G_new.nodes())
        
        # PROPOSAL
        if np.random.rand() < 0.5:
            # ADD EDGE: Try to form a tetrahedron?
            # Prefer adding edges between common neighbors of an existing edge
            # This "completes" triangles into tetrahedra
            # Simplest heuristic: Pick random u, pick random neighbor v, pick random neighbor w...
            # Let's stick to the "Close Triangle" heuristic, it eventually builds tets.
            u = np.random.choice(nodes)
            nbrs = list(G_new.neighbors(u))
            if len(nbrs) >= 2:
                v, w = np.random.choice(nbrs, 2, replace=False)
                if not G_new.has_edge(v, w):
                    G_new.add_edge(v, w)
        else:
            # REMOVE EDGE
            edges = list(G_new.edges())
            if len(edges) > 0:
                u, v = edges[np.random.randint(len(edges))]
                G_new.remove_edge(u, v)
        
        # ACCEPT/REJECT
        if nx.is_connected(G_new):
            # Optimization: Only full calc if we need it? 
            # For N=45, full calc is fast enough.
            E_g_new, E_z_new = get_energies(G_new, alpha, beta)
            E_tot_new = E_g_new + E_z_new
            dE = E_tot_new - E_total
            
            if dE < 0 or np.random.rand() < np.exp(-dE / TEMP):
                G = G_new
                E_total = E_tot_new

    dim = get_spectral_dimension(G)
    avg_degree = 2 * G.number_of_edges() / N_NODES
    return dim, avg_degree

# --- Main Sweep ---
if __name__ == "__main__":
    print(f"Starting 3D Sweep: Alpha={ALPHA_GRAVITY} (Tetrahedra)")
    print(f"{'Beta':<10} | {'Dim':<10} | {'AvgDeg':<10}")
    print("-" * 35)

    results = []
    for beta in BETA_VALUES:
        dim, deg = run_simulation(ALPHA_GRAVITY, beta)
        results.append({"beta": beta, "dim": dim, "deg": deg})
        print(f"{beta:<10.2f} | {dim:<10.4f} | {deg:<10.2f}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv("phase_diagram_3d.csv", index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['beta'], df['dim'], 'o-', color='darkorange', label='Spectral Dimension')
    plt.axhline(4.0, color='gray', linestyle=':', alpha=0.5, label='4D')
    plt.axhline(3.0, color='r', linestyle='--', alpha=0.8, label='3D (Target)')
    plt.axhline(2.0, color='g', linestyle='--', alpha=0.5, label='2D')
    plt.xlabel("Vacuum Stiffness (Beta)")
    plt.ylabel("Emergent Dimension")
    plt.title("Phase Transition: Searching for 3D Geometry")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()