"""
substrate_hero_annealing.py

Simulated Annealing run to crystallize 3D geometry from a random substrate.
Solves the "Freezing" problem by starting at High Temp and cooling down.

Mechanism:
1. Start with Hot Dense Graph (High Entropy).
2. T_High: Allows breaking 'bad' tetrahedra to escape local minima.
3. T_Low: Freezes the system into the lowest energy configuration (3D Lattice).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
N_NODES = 200
STEPS = 5000           # Longer run for proper cooling
ALPHA = 1.0            # Gravity (Tetrahedra)
BETA = 2.5             # Critical ZPE (from your sweep)

# ANNEALING SCHEDULE
T_START = 10.0         # Hot! (Melts the random graph)
T_END = 0.1            # Cold! (Freezes the geometry)

def count_tetrahedra(G):
    # Optimized Local Clique Counting
    if G.number_of_edges() < 6: return 0
    count = 0
    adj = {n: set(G[n]) for n in G}
    # Iterate over edges to find 4-cliques (u-v plus two common neighbors connected to each other)
    for u, v in G.edges():
        common = list(adj[u].intersection(adj[v]))
        if len(common) < 2: continue
        # Check all pairs in common
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                if G.has_edge(common[i], common[j]):
                    count += 1
    # Each tetrahedron has 6 edges, so we count it 6 times in this loop
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
    # Probing at t=2.0 was too slow for dense graphs (Dim=0).
    # We use a slightly faster probe time to capture the "melting" phase.
    evals = nx.laplacian_spectrum(G)
    t1, t2 = 1.0, 1.5
    K1 = np.sum(np.exp(-evals * t1))
    K2 = np.sum(np.exp(-evals * t2))
    if K1 <= 1e-9 or K2 <= 1e-9: return 0.0
    slope = (np.log(K2) - np.log(K1)) / (np.log(t2) - np.log(t1))
    return -2 * slope

# --- Simulation ---
print(f"Initializing Annealing Run (N={N_NODES})...")
# Start slightly sparser to help the ZPE
G = nx.erdos_renyi_graph(N_NODES, 0.12)
while not nx.is_connected(G): G = nx.erdos_renyi_graph(N_NODES, 0.12)

E_g, E_z = get_energies(G)
E_tot = E_g + E_z

history_dim = []
history_temp = []
history_tet = []

print(f"Melting and Cooling (Steps={STEPS}, T={T_START}->{T_END})...")

for step in range(STEPS):
    # 1. Update Temperature
    progress = step / STEPS
    # Exponential cooling schedule
    current_temp = T_START * (T_END / T_START) ** progress
    
    G_new = G.copy()
    
    # 2. Proposal (Bias to Pruning early on?)
    # Let's keep 50/50 add/remove to be unbiased
    if np.random.rand() < 0.5:
        # Add Edge (Try to form Tetrahedron)
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
            
    # 3. Acceptance
    if nx.is_connected(G_new):
        Eg_new, Ez_new = get_energies(G_new)
        Et_new = Eg_new + Ez_new
        dE = Et_new - E_tot
        
        # Metropolis with Variable Temp
        if dE < 0 or np.random.rand() < np.exp(-dE / current_temp):
            G = G_new
            E_tot = Et_new
            E_g = Eg_new # Track for logging
            
    # 4. Logging
    if step % 100 == 0:
        dim = get_dimension(G)
        history_dim.append(dim)
        history_temp.append(current_temp)
        history_tet.append(abs(E_g))
        print(f"Step {step}: T={current_temp:.2f} | Dim={dim:.2f} | Tets={int(abs(E_g))} | Edges={G.number_of_edges()}")

# --- Visualization ---
final_dim = get_dimension(G)
print(f"\nFinal Spectral Dimension: {final_dim:.4f}")

# 1. Evolution Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history_dim, 'b')
plt.title("Dimension Evolution")
plt.xlabel("Time (x100)")
plt.axhline(3.0, color='r', linestyle='--')

plt.subplot(1, 3, 2)
plt.plot(history_tet, 'orange')
plt.title("Tetrahedra Count")

plt.subplot(1, 3, 3)
plt.plot(history_temp, 'r')
plt.title("Temperature")
plt.savefig("annealing_stats.png")

# 2. Geometry Plot
print("Generating 3D Visualization...")
pos = nx.spectral_layout(G, dim=3)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
xs = [pos[n][0] for n in G.nodes()]
ys = [pos[n][1] for n in G.nodes()]
zs = [pos[n][2] for n in G.nodes()]
center = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
dists = [np.linalg.norm(np.array(pos[n]) - center) for n in G.nodes()]
ax.scatter(xs, ys, zs, c=dists, cmap='plasma', s=40, alpha=0.8)
for u, v in G.edges():
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], color='gray', alpha=0.1)
ax.set_title(f"Crystallized Geometry (N={N_NODES})\nDim={final_dim:.2f}")
plt.savefig("hero_geometry_annealed.png")
print("Done.")