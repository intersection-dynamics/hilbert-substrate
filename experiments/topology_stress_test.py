import numpy as np
from scipy.linalg import expm, norm
import itertools
import time
import pandas as pd
import networkx as nx

def run_topology_stress_test():
    print("--- TOPOLOGY STRESS TEST: RING, GRID, RANDOM ---")
    
    # Configuration
    # We run N=4, 5, 6 to catch the transition and accommodate geometries like 2x3 grid
    scenarios = [
        {"N": 4, "Topo": "Ring"},
        {"N": 4, "Topo": "Grid 2x2"}, 
        {"N": 4, "Topo": "Complete (Random Reg d=3)"},
        
        {"N": 5, "Topo": "Ring"},
        {"N": 5, "Topo": "Random Connected"}, # N=5 regular d=3 impossible
        
        {"N": 6, "Topo": "Ring"},
        {"N": 6, "Topo": "Grid 2x3"},
        {"N": 6, "Topo": "Random Regular d=3"},
    ]
    
    restarts = 3
    results = []

    # --- Pre-compute Basis per N to save time ---
    basis_cache = {}
    
    for N in [4, 5, 6]:
        print(f"Generating Basis N={N}...")
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        Paulis = [I, X, Y, Z]
        
        mats = []
        weights = []
        for indices in itertools.product(range(4), repeat=N):
            term = Paulis[indices[0]]
            for k in range(1, N): term = np.kron(term, Paulis[indices[k]])
            w = sum(1 for i in indices if i != 0)
            mats.append(term)
            weights.append(w)
        basis_cache[N] = (np.array(mats), np.array(weights), 2**N)

    # --- Helper Functions ---
    def get_cost_and_grad(H, mats, weights, dim):
        coeffs = np.real(np.einsum('ij,kji->k', H, mats)) / dim
        norm_sq = np.sum(coeffs**2)
        w4 = weights ** 4
        cost = np.sum(w4 * coeffs**2) / norm_sq
        
        grad_coeffs = 2 * w4 * coeffs / norm_sq
        M = np.tensordot(grad_coeffs, mats, axes=([0],[0]))
        K = 1j * (H @ M - M @ H)
        return cost, K

    def build_hamiltonian(N, topo_type, mats, weights, dim):
        # Use NetworkX to generate graph
        if topo_type == "Ring":
            G = nx.cycle_graph(N)
        elif topo_type == "Grid 2x2":
            G = nx.grid_2d_graph(2, 2)
            G = nx.convert_node_labels_to_integers(G)
        elif topo_type == "Grid 2x3":
            G = nx.grid_2d_graph(2, 3)
            G = nx.convert_node_labels_to_integers(G)
        elif "Complete" in topo_type:
            G = nx.complete_graph(N)
        elif "Random Regular" in topo_type:
            G = nx.random_regular_graph(3, N, seed=42)
        elif "Random Connected" in topo_type:
            # Erdos-Renyi that ensures connectivity
            while True:
                G = nx.erdos_renyi_graph(N, 0.6)
                if nx.is_connected(G): break
        else:
            raise ValueError(f"Unknown topology: {topo_type}")
            
        # Build H
        H = np.zeros((dim, dim), dtype=complex)
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        def add_term(op, i, j):
            ops = [I]*N; ops[i]=op; ops[j]=op
            t = ops[0]
            for k in range(1,N): t = np.kron(t, ops[k])
            return t
            
        for u, v in G.edges():
            H += add_term(X, u, v) + add_term(Y, u, v) + add_term(Z, u, v)
            
        return H

    # --- Main Loop ---
    for scen in scenarios:
        N = scen["N"]
        topo = scen["Topo"]
        print(f"\nRunning {topo} (N={N})...")
        
        mats, weights, dim = basis_cache[N]
        
        # 1. Targets
        H_target = build_hamiltonian(N, topo, mats, weights, dim)
        c_spatial, _ = get_cost_and_grad(H_target, mats, weights, dim)
        
        # Eigen check
        evals, evecs = np.linalg.eigh(H_target)
        H_diag = np.diag(evals)
        # Transform back to check true cost (sanity check)
        # H_back = evecs @ H_diag @ evecs.conj().T
        # c_sanity, _ = get_cost_and_grad(H_back, mats, weights, dim)
        
        # Harmonion Theoretical Cost (if we could live in eigenbasis)
        # We compute cost of Diagonal matrix directly in Pauli basis
        # Note: As discussed, this is "Cheating" but represents the "Stationarity Cost"
        c_harmonion, _ = get_cost_and_grad(H_diag, mats, weights, dim)
        
        print(f"  Target Spatial: {c_spatial:.2f} | Target Harmonion (Stationary): {c_harmonion:.2f}")

        # 2. Restarts
        spatial_hits = 0
        mean_cost = 0
        
        for r in range(restarts):
            # Scramble
            np.random.seed(r * 100 + N)
            G_rand = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            G_rand = G_rand + G_rand.conj().T
            U_scram = expm(1j * G_rand * 0.2)
            H_scrambled = U_scram @ H_target @ U_scram.conj().T
            
            # Optimize
            H_curr = H_scrambled.copy()
            lr = 0.1
            for step in range(80): # Shortened slightly for speed
                cost_old, K = get_cost_and_grad(H_curr, mats, weights, dim)
                if norm(K) < 1e-6: break
                K_dir = K / norm(K)
                
                alpha = lr
                accepted = False
                for _ in range(5):
                    U_step = expm(-1j * alpha * K_dir)
                    H_try = U_step @ H_curr @ U_step.conj().T
                    c_try, _ = get_cost_and_grad(H_try, mats, weights, dim)
                    if c_try < cost_old:
                        H_curr = H_try
                        cost_old = c_try
                        accepted = True
                        break
                    else:
                        alpha *= 0.5
                if not accepted: lr *= 0.5
            
            mean_cost += cost_old
            # Check if basin is spatial (within 10% or absolute tolerance)
            if abs(cost_old - c_spatial) < 2.0:
                spatial_hits += 1
            
            print(f"    Run {r}: {cost_old:.2f}")

        mean_cost /= restarts
        results.append({
            "N": N,
            "Topo": topo,
            "Spatial_Cost": c_spatial,
            "Harmonion_Cost": c_harmonion,
            "Mean_Rec_Cost": mean_cost,
            "Spatial_Success": f"{spatial_hits}/{restarts}"
        })

    return pd.DataFrame(results)

df_topo = run_topology_stress_test()
print("\n--- FINAL TOPOLOGY STRESS TEST RESULTS ---")
print(df_topo)