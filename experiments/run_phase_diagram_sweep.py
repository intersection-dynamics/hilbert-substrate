import numpy as np
from scipy.linalg import expm, norm
import itertools
import time
import pandas as pd
import networkx as nx

def run_phase_diagram_sweep(N=4):
    print(f"--- PHASE DIAGRAM SWEEP (N={N}, Topology=Ring) ---")
    
    # 1. Setup Basis
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
    mats = np.array(mats)
    weights = np.array(weights)
    dim = 2**N

    # 2. Helper with Variable Power
    def get_cost_and_grad(H, p):
        coeffs = np.real(np.einsum('ij,kji->k', H, mats)) / dim
        norm_sq = np.sum(coeffs**2)
        w_p = weights ** p
        cost = np.sum(w_p * coeffs**2) / norm_sq
        
        grad_coeffs = 2 * w_p * coeffs / norm_sq
        M = np.tensordot(grad_coeffs, mats, axes=([0],[0]))
        K = 1j * (H @ M - M @ H)
        return cost, K

    # 3. Target Hamiltonian (Ring)
    H_target = np.zeros((dim, dim), dtype=complex)
    def add_term(op, i, j):
        ops = [I]*N; ops[i]=op; ops[j]=op
        t = ops[0]
        for k in range(1,N): t = np.kron(t, ops[k])
        return t
    
    # Ring Topology
    for i in range(N):
        j = (i + 1) % N
        H_target += add_term(X, i, j) + add_term(Y, i, j) + add_term(Z, i, j)
        
    # Eigenbasis for reference
    evals, evecs = np.linalg.eigh(H_target)
    H_diag = np.diag(evals)

    # 4. Sweep
    powers = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    results = []
    
    for p in powers:
        print(f"Testing penalty w^{p}...")
        
        # Baselines
        c_spatial, _ = get_cost_and_grad(H_target, p)
        # Theoretically ideal cost (if we could reach eigenbasis)
        c_harmonion, _ = get_cost_and_grad(H_diag, p)
        
        # Optimization (3 restarts to be safe)
        best_recovered = 1e9
        
        for r in range(3):
            # Scramble
            np.random.seed(42 + int(p)*10 + r)
            G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            G = G + G.conj().T
            U_scram = expm(1j * G * 0.2)
            H_curr = U_scram @ H_target @ U_scram.conj().T
            
            # Descent
            lr = 0.1
            for step in range(100):
                cost_old, K = get_cost_and_grad(H_curr, p)
                if norm(K) < 1e-6: break
                K_dir = K / norm(K)
                
                alpha = lr
                accepted = False
                for _ in range(5):
                    U_step = expm(-1j * alpha * K_dir)
                    H_try = U_step @ H_curr @ U_step.conj().T
                    c_try, _ = get_cost_and_grad(H_try, p)
                    if c_try < cost_old:
                        H_curr = H_try
                        cost_old = c_try
                        accepted = True
                        break
                    else:
                        alpha *= 0.5
                if not accepted: lr *= 0.5
            
            if cost_old < best_recovered:
                best_recovered = cost_old
        
        # Classification
        # Did we beat spatial?
        # Note: If best_recovered is significantly lower than c_spatial, we escaped.
        # If best_recovered ~= c_spatial, we are trapped.
        status = "Trapped (Space)"
        if best_recovered < c_spatial * 0.8: # Threshold for breaking out
            status = "Escaped (Harmonion)"
        elif best_recovered > c_spatial * 1.5:
             status = "Lost (Random)"

        results.append({
            "Power": p,
            "Spatial_Cost": c_spatial,
            "Harmonion_Cost": c_harmonion,
            "Recovered_Cost": best_recovered,
            "Status": status
        })

    return pd.DataFrame(results)

df_phase = run_phase_diagram_sweep(N=4)
print("\n--- PHASE DIAGRAM DATA ---")
print(df_phase)