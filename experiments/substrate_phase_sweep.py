"""
substrate_phase_sweep.py

Parameter sweep to find the "Goldilocks Zone" of emergent geometry.
We vary the Zero-Point Energy (ZPE) strength (Beta) against a fixed
Gravitational Clustering strength (Alpha).

Hypothesis:
- Low Beta: Gravity wins -> Universe collapses to a point (Dim ~ 0 or Inf).
- High Beta: ZPE wins -> Universe explodes into dust/tree (Dim ~ 1).
- Critical Beta: Balanced forces -> Emergent Manifold (Dim ~ 3).

Usage:
    python substrate_phase_sweep.py
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- Configuration ---
N_NODES = 40           # Keep small (30-50) for speed (Eigenvalues are O(N^3))
STEPS_PER_RUN = 1000   # Monte Carlo steps to equilibrate
BETA_VALUES = np.linspace(0.0, 4.0, 20) # Sweep ZPE strength from 0 to 4
ALPHA_GRAVITY = 1.0    # Fixed attractive force

# Temperature for Metropolis-Hastings (allows escaping local minima)
TEMP = 0.4

def get_energies(G, alpha, beta):
    """
    Calculate the two competing potentials of the vacuum.
    1. Gravity (Attractive): Wants to form triangles (cluster).
       E_grav = - alpha * (Number of Triangles)
    2. ZPE (Repulsive): Wants to lower vibrational frequencies.
       E_zpe = + beta * Sum(sqrt(Laplacian_Eigenvalues))
    """
    if not nx.is_connected(G):
        return float('inf'), float('inf')

    # 1. Gravitational Potential (Clustering)
    # Trace(A^3) / 6 is number of triangles
    tri = sum(nx.triangles(G).values()) / 3
    E_grav = -alpha * tri

    # 2. Casimir Potential (Spectral ZPE)
    # We use the combinatorial Laplacian (L = D - A)
    # The 'frequencies' of the vacuum are sqrt(eigenvalues)
    evals = nx.laplacian_spectrum(G)
    # Filter small numerical noise near 0
    evals = evals[evals > 1e-5]
    E_zpe = beta * np.sum(np.sqrt(evals))

    return E_grav, E_zpe

def get_spectral_dimension(G, t_probe=2.0):
    """
    Measure effective dimension d_s via Heat Kernel return probability P(t).
    P(t) ~ t^(-d_s/2)  =>  d_s = -2 * slope of log(P(t))
    """
    evals = nx.laplacian_spectrum(G)
    # Measure slope between t_probe and t_probe + dt
    dt = 0.5
    K_t1 = np.sum(np.exp(-evals * t_probe))
    K_t2 = np.sum(np.exp(-evals * (t_probe + dt)))
    
    slope = (np.log(K_t2) - np.log(K_t1)) / (np.log(t_probe + dt) - np.log(t_probe))
    return -2 * slope

def run_simulation(alpha, beta):
    # Initialize: Random Hot Universe
    G = nx.erdos_renyi_graph(N_NODES, p=0.15)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(N_NODES, p=0.15)
        
    E_grav, E_zpe = get_energies(G, alpha, beta)
    E_total = E_grav + E_zpe
    
    # Evolution Loop
    for step in range(STEPS_PER_RUN):
        G_new = G.copy()
        nodes = list(G_new.nodes())
        
        # PROPOSAL: 50% Add Local Link (Triangle), 50% Remove Random Link
        if np.random.rand() < 0.5:
            # Try to add edge to close a triangle (Gravity assist)
            u = np.random.choice(nodes)
            nbrs = list(G_new.neighbors(u))
            if len(nbrs) >= 2:
                v, w = np.random.choice(nbrs, 2, replace=False)
                if not G_new.has_edge(v, w):
                    G_new.add_edge(v, w)
        else:
            # Try to remove random edge (ZPE assist)
            edges = list(G_new.edges())
            if len(edges) > 0:
                u, v = edges[np.random.randint(len(edges))]
                G_new.remove_edge(u, v)
        
        # ACCEPT/REJECT
        if nx.is_connected(G_new):
            E_g_new, E_z_new = get_energies(G_new, alpha, beta)
            E_tot_new = E_g_new + E_z_new
            dE = E_tot_new - E_total
            
            # Metropolis Rule
            if dE < 0 or np.random.rand() < np.exp(-dE / TEMP):
                G = G_new
                E_total = E_tot_new

    # Final Measurement
    final_dim = get_spectral_dimension(G)
    avg_degree = 2 * len(G.edges()) / N_NODES
    return final_dim, avg_degree

# --- Main Sweep ---
if __name__ == "__main__":
    results = []
    print(f"Starting Sweep: Alpha={ALPHA_GRAVITY}, Beta range={BETA_VALUES[0]}-{BETA_VALUES[-1]}")
    print(f"{'Beta':<10} | {'Dim':<10} | {'AvgDeg':<10}")
    print("-" * 35)

    for beta in BETA_VALUES:
        dim, deg = run_simulation(ALPHA_GRAVITY, beta)
        results.append({
            "beta": beta,
            "alpha": ALPHA_GRAVITY,
            "dimension": dim,
            "avg_degree": deg
        })
        print(f"{beta:<10.2f} | {dim:<10.4f} | {deg:<10.2f}")

    # Save Results
    df = pd.DataFrame(results)
    filename = "phase_diagram.csv"
    df.to_csv(filename, index=False)
    print(f"\nSweep complete. Data saved to {filename}")

    # Quick Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['beta'], df['dimension'], 'o-', color='purple', label='Spectral Dimension')
    plt.axhline(3.0, color='r', linestyle='--', alpha=0.5, label='3D')
    plt.axhline(2.0, color='g', linestyle='--', alpha=0.5, label='2D')
    plt.axhline(1.0, color='b', linestyle='--', alpha=0.5, label='1D')
    plt.xlabel(f"Vacuum Stiffness (Beta) [Fixed Gravity Alpha={ALPHA_GRAVITY}]")
    plt.ylabel("Emergent Dimension")
    plt.title("Phase Transition: From Black Hole to Spacetime")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()