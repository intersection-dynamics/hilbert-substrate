#!/usr/bin/env python3
import json
import networkx as nx

def calculate_physical_scale(json_path="gpu_mesoscape_organizer_collective_response.json"):
    print(f"Loading collective response data from {json_path}...\n")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{json_path}'. Make sure the simulation has finished running.")
        return

    summary = data.get("response_summary", {})
    partition = data.get("partition", {})
    
    # 1. Delta Tau (Simulated Time to peak mutual information shift)
    delta_tau_steps = summary.get("argmax_abs_core_shell_mi_shift_step", 0)
    
    if delta_tau_steps == 0:
        print("Warning: Wave peak detected at step 0. Defaulting to 1 to prevent division by zero.")
        delta_tau_steps = 1

    # 2. Delta n (Simulated Distance)
    edges = partition.get("organizer_edges", [])
    G = nx.Graph()
    G.add_edges_from(edges)
    
    left_nodes = partition.get("left_nodes", [])
    core_nodes = partition.get("core_nodes", [])
    
    # BUG FIX: Ensure the origin nodes don't overlap with the core nodes
    valid_left_nodes = [n for n in left_nodes if n not in core_nodes]
    
    distances = []
    for l_node in valid_left_nodes:
        for c_node in core_nodes:
            if nx.has_path(G, l_node, c_node):
                distances.append(nx.shortest_path_length(G, l_node, c_node))
    
    if distances:
        delta_n_links = min(distances)
    else:
        print("Warning: No path found between the poke origin and the core. Defaulting distance to 1 link.")
        delta_n_links = 1
    
    # Simulated Velocity (links per step)
    v_sim = delta_n_links / delta_tau_steps
    
    # 3. Physics Mapping
    if nx.is_empty(G):
        print("Error: Graph is empty.")
        return
        
    # Get the diameter to map to a physical wavelength
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        # If the graph fragmented, use the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        diameter = nx.diameter(G_sub)

    # Prevent division by zero if the graph is just a single node
    diameter = max(1, diameter)

    # Constants
    gamma_wavelength_m = 1e-12       # 1 picometer (meters)
    speed_of_light_mps = 299792458.0 # c (meters per second)
    planck_time_s = 5.39e-44         # t_p (seconds)
    
    # Physical length of one spatial link (a)
    link_size_a = gamma_wavelength_m / diameter
    
    # Physical duration of one simulation step (dt)
    # c = v_sim * (a / dt)  =>  dt = (v_sim * a) / c
    dt_seconds = (v_sim * link_size_a) / speed_of_light_mps
    
    print("=" * 64)
    print("LIEB-ROBINSON WAVE PROPAGATION RESULTS")
    print("=" * 64)
    print(f"Simulated Distance traveled  : {delta_n_links} links")
    print(f"Simulated Time to peak       : {delta_tau_steps} dt steps")
    print(f"Simulated Wave Velocity      : {v_sim:.4f} links/step")
    print("-" * 64)
    print("PHYSICAL SCALE DERIVATION")
    print("-" * 64)
    print(f"Assumed Wavelength (Gamma)   : {gamma_wavelength_m:.2e} meters")
    print(f"Emergent Spatial Link Size   : {link_size_a:.4e} meters")
    print(f"Derived Time Step (dt)       : {dt_seconds:.4e} seconds")
    print("-" * 64)
    
    ratio = dt_seconds / planck_time_s
    print(f"Conclusion: One dt step in your mesoscape is roughly")
    print(f"{ratio:.2e} times the Planck Time.")
    print("=" * 64)

if __name__ == "__main__":
    calculate_physical_scale()