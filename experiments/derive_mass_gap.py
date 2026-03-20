#!/usr/bin/env python3
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Import your existing simulation engine!
import gpu_mesoscape_organizer_collective_response as meso

def calculate_mi_core_to_node(psi, core_nodes, target_node, n_sites, xp):
    """Calculates Mutual Information between the 2-node core and a single target node."""
    # Trace out everything except the core and the target node
    keep_all = core_nodes + [target_node]
    rho_all = meso.partial_trace_keep(psi, keep_all, n_sites, xp)
    rho_core = meso.partial_trace_keep(psi, core_nodes, n_sites, xp)
    rho_target = meso.partial_trace_keep(psi, [target_node], n_sites, xp)
    
    # S(core) + S(target) - S(core U target)
    s_core = meso.von_neumann_entropy(rho_core, xp)
    s_target = meso.von_neumann_entropy(rho_target, xp)
    s_all = meso.von_neumann_entropy(rho_all, xp)
    
    return float(s_core + s_target - s_all)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--target-step", type=int, default=516, help="The step where the [8, 10] core crystallized")
    args = ap.parse_args()

    # Match the exact configuration of your N=12 run
    cfg = meso.SimConfig(
        n_max=12, n_init=2, seed=0, local_scale=0.10, pair_scale=0.16, 
        spawn_pair_scale=0.12, total_steps=args.target_step, dt=0.2, 
        eval_every=12, lookahead_windows=3, settling_windows=2, 
        fission_fraction=0.30, candidate_fraction=0.45, birth_score_floor=0.015, 
        decay_mi_threshold=0.05, decay_corr_threshold=0.07, 
        neighborhood_bonus_weight=0.18, shell_bonus_weight=0.20, 
        mi_survival_floor=0.076, corr_survival_floor=0.086, 
        persist_windows_required=2, persist_entropy_threshold=0.06, 
        persist_mean_mi_threshold=0.07, persist_triangle_threshold=1, 
        device=args.device
    )

    print(f"Re-simulating mesoscape up to step {args.target_step} to measure Mass Gap...\n")
    xp, using_gpu, GM, snapshots, states = meso.simulate_mesoscape(cfg, progress=True)
    
    # Extract the final state at the target step
    final_state = states[-1]
    psi = final_state["psi"]
    active_nodes = final_state["active_nodes"]
    active_edges = final_state["active_edges"]
    n_sites = cfg.n_max
    
    # Identify the dominant core
    core_data = meso.dominant_core_snapshot(psi, active_nodes, active_edges, final_state["edge_strengths"], GM, xp, n_sites)
    core_nodes = list(core_data["core_pair"])
    print(f"\nTarget Core locked at nodes: {core_nodes}")

    # Build the graph to find topological distances (r)
    G = nx.Graph()
    G.add_nodes_from(active_nodes)
    G.add_edges_from(active_edges)

    distances = []
    mutual_informations = []

    print("\nMeasuring Correlation Length (MI decay across the substrate)...")
    for node in active_nodes:
        if node in core_nodes:
            continue
            
        # Distance is the shortest path from the target node to *either* core node
        if nx.has_path(G, node, core_nodes[0]) and nx.has_path(G, node, core_nodes[1]):
            d1 = nx.shortest_path_length(G, node, core_nodes[0])
            d2 = nx.shortest_path_length(G, node, core_nodes[1])
            r = min(d1, d2)
            
            mi = calculate_mi_core_to_node(psi, core_nodes, node, n_sites, xp)
            
            if mi > 1e-10:  # Ignore total zeroes to safely take the logarithm
                distances.append(r)
                mutual_informations.append(mi)
                print(f" Node {node:2d} | Distance: {r} links | MI: {mi:.6e}")

    # Group by distance and average the MI
    dist_dict = {}
    for r, mi in zip(distances, mutual_informations):
        dist_dict.setdefault(r, []).append(mi)
        
    unique_r = sorted(list(dist_dict.keys()))
    avg_mi = [np.mean(dist_dict[r]) for r in unique_r]
    log_avg_mi = np.log(avg_mi)

    # Perform Log-Linear Regression: ln(MI) = (-1/xi) * r + C
    slope, intercept, r_value, p_value, std_err = linregress(unique_r, log_avg_mi)
    
    xi = -1.0 / slope if slope < 0 else float('inf')
    mass_gap = 1.0 / xi if xi != float('inf') else 0.0

    print("=" * 64)
    print("DIMENSIONAL TRANSMUTATION: MASS GAP DERIVATION")
    print("=" * 64)
    print(f"R-squared of fit          : {r_value**2:.4f}")
    print(f"Correlation Length (xi)   : {xi:.4f} links")
    print(f"Dimensionless Mass Gap    : {mass_gap:.4f}")
    print("=" * 64)

    # Plot the exponential decay
    plt.figure(figsize=(10, 6))
    plt.scatter(unique_r, log_avg_mi, color='blue', label='Data (Avg MI)', zorder=5)
    plt.plot(unique_r, intercept + slope * np.array(unique_r), color='red', linestyle='--', label=f'Fit: $\\xi$ = {xi:.2f}')
    
    plt.title("Correlation Length: The Stickiness of the Mesoscape", fontsize=16)
    plt.xlabel("Distance from Core ($r$ in links)", fontsize=14)
    plt.ylabel("Log Mutual Information $\\ln(MI)$", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("correlation_length_decay.png", dpi=300)
    print("Saved plot to 'correlation_length_decay.png'")

if __name__ == "__main__":
    main()