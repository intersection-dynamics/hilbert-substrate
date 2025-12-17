"""
substrate_hero_run.py

A high-resolution simulation to confirm 3D emergence.
Uses N=200 nodes at the critical phase transition point (Beta=2.5).

Goals:
1. Overcome Surface Effects to measure Bulk Dimension.
2. Visualize the emergent structure in 3D.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
N_NODES = 200          # Large enough to have a "Bulk"
STEPS = 3000           # Allow time to anneal
ALPHA = 1.0            # Gravity (Tetrahedra)
BETA = 2.5             # Critical Point found in sweep
TEMP = 0.3             # Low temp to freeze the core

def count_tetrahedra(G):
    # Fast clique counter
    return sum(nx.triangles(G).values()) / 3  # Approximation for speed? 
    # Wait, triangles != tetrahedra. 
    # For N=200, the full tetrahedra count is expensive (O(N^4)).
    # Let's use the optimized local clique check.
    if G.number_of_edges() < 6: return 0
    count = 0
    adj = {n: set(G[n]) for n in G}
    for u, v in G.edges():
        common = list(adj[u].intersection(adj[v]))
        if len(common) < 2: continue
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                if G.has_edge(common[i], common[j]):
                    count += 1
    return count / 6.0

def get_energies(G):
    if not nx.is_connected(G): return 1e9, 1e9
    
    # 1. Gravity (Tetrahedra)
    E_grav = -ALPHA * count_tetrahedra(G)
    
    # 2. ZPE (Spectral)
    evals = nx.laplacian_spectrum(G)
    evals = evals[evals > 1e-5]
    E_zpe = BETA * np.sum(np.sqrt(evals))
    
    return E_grav, E_zpe

def get_dimension(G):
    evals = nx.laplacian_spectrum(G)
    t1, t2 = 2.0, 2.5
    K1 = np.sum(np.exp(-evals * t1))
    K2 = np.sum(np.exp(-evals * t2))
    slope = (np.log(K2) - np.log(K1)) / (np.log(t2) - np.log(t1))
    return -2 * slope

# --- Simulation ---
print(f"Initializing Hero Run (N={N_NODES})...")
G = nx.erdos_renyi_graph(N_NODES, 0.15)
while not nx.is_connected(G): G = nx.erdos_renyi_graph(N_NODES, 0.15)

E_g, E_z = get_energies(G)
E_tot = E_g + E_z

print(f"Annealing geometry (Steps={STEPS})...")
for step in range(STEPS):
    G_new = G.copy()
    
    # Proposal: Bias towards Tetrahedra
    if np.random.rand() < 0.5:
        # Add Edge
        nodes = list(G.nodes())
        u = np.random.choice(nodes)
        nbrs = list(G.neighbors(u))
        if len(nbrs) >= 2:
            v, w = np.random.choice(nbrs, 2, replace=False)
            if not G_new.has_edge(v, w): G_new.add_edge(v, w)
    else:
        # Remove Edge
        edges = list(G.edges())
        if len(edges) > 0:
            u, v = edges[np.random.randint(len(edges))]
            G_new.remove_edge(u, v)
            
    if nx.is_connected(G_new):
        Eg_new, Ez_new = get_energies(G_new)
        Et_new = Eg_new + Ez_new
        dE = Et_new - E_tot
        
        if dE < 0 or np.random.rand() < np.exp(-dE/TEMP):
            G = G_new
            E_tot = Et_new
            
    if step % 100 == 0:
        print(f"Step {step}: Dim={get_dimension(G):.2f}, Tetrahedra={int(-E_g)}")

# --- Results ---
final_dim = get_dimension(G)
print(f"\nFinal Spectral Dimension: {final_dim:.4f}")

# --- Visualization ---
print("Generating 3D Visualization...")
# Use Spectral Layout (uses Laplacian eigenvectors -> intrinsic geometry)
pos = nx.spectral_layout(G, dim=3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

xs = [pos[n][0] for n in G.nodes()]
ys = [pos[n][1] for n in G.nodes()]
zs = [pos[n][2] for n in G.nodes()]

# Color by distance from center
center = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
dists = [np.linalg.norm(np.array(pos[n]) - center) for n in G.nodes()]

ax.scatter(xs, ys, zs, c=dists, cmap='plasma', s=40, alpha=0.8)

# Draw edges (lightly)
for u, v in G.edges():
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], 
            color='gray', alpha=0.1)

ax.set_title(f"Emergent Geometry (N={N_NODES})\nDim={final_dim:.2f}")
plt.savefig("hero_geometry_3d.png")
plt.show()