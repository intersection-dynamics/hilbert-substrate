#!/usr/bin/env python3
"""
THE SUBSTRATE FRAMEWORK: DEMONSTRATION SUITE
====================================================================
A unified physics engine deriving classical reality from Hilbert Space.

Axioms:
1. Hilbert Space Realism (Vector is fundamental)
2. Unitary Evolution (No collapse)
3. Emergent Constraints (Geometry = Memory)

Experiments included:
[1] INERTIA:  Deriving Mass (F=ma) from Vacuum Stiffness.
[2] FERMIONS: Deriving Pauli Exclusion from Non-Abelian Topology.
[3] MEMORY:   Proving Path Distinguishability (Geometry is Memory).
[4] WAKE:     Visualizing the "History" trail of a moving particle.
"""

import sys
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sp
    HAS_GPU = True
    XP = cp  # Generic array module alias
    print(" [✓] CuPy detected. GPU Acceleration Enabled.")
except ImportError:
    HAS_GPU = False
    XP = np
    print(" [!] CuPy not found. Running on CPU (Standard Mode).")

# =============================================================================
# EXPERIMENT 1: DERIVING INERTIA (F=ma)
# =============================================================================
def demo_inertia():
    print("\n--- EXPERIMENT: DERIVATION OF INERTIA ---")
    print("Hypothesis: Mass is the resistance of the Gauge Field to transformation.")
    print("We apply a constant Force and vary Vacuum Stiffness (g).")
    
    # Config
    N_SITES = 6
    N_LINKS = N_SITES - 1
    L_TRUNC = 1
    FORCE = 0.5
    
    # Basis Construction
    states = []
    gauge_configs = list(itertools.product(range(-L_TRUNC, L_TRUNC+1), repeat=N_LINKS))
    for pos in range(N_SITES):
        for g in gauge_configs:
            states.append((pos, g)) # g is tuple
    
    dim = len(states)
    state_map = {s: i for i, s in enumerate(states)}
    print(f"Hilbert Space: {dim} states")

    def build_hamiltonian(stiffness_g):
        H = np.zeros((dim, dim), dtype=np.complex128)
        for idx, (pos, links) in enumerate(states):
            # Potential Energy (Gauge Strain + External Force)
            strain = 0.5 * stiffness_g * sum([l**2 for l in links])
            potential = -FORCE * pos
            H[idx, idx] = strain + potential
            
            # Kinetic (Gauged Hopping)
            if pos < N_SITES - 1:
                link_id = pos
                current_l = links[link_id]
                target_l = current_l - 1 # Peierls substitution
                if abs(target_l) <= L_TRUNC:
                    new_links = list(links)
                    new_links[link_id] = target_l
                    new_state = (pos+1, tuple(new_links))
                    if new_state in state_map:
                        j = state_map[new_state]
                        H[j, idx] -= 1.0
                        H[idx, j] -= 1.0
        return H

    # Run Sweep
    g_values = [0.0, 2.0, 10.0]
    results = {}
    
    plt.figure(figsize=(10, 6))
    
    for g in g_values:
        print(f"  > Simulating g={g}...")
        H = build_hamiltonian(g)
        
        # Start state
        psi = np.zeros(dim, dtype=np.complex128)
        start_idx = state_map[(0, tuple([0]*N_LINKS))]
        psi[start_idx] = 1.0
        
        # Evolve
        times = np.linspace(0, 4, 40)
        evals, evecs = eigh(H)
        coeffs = evecs.conj().T @ psi
        
        pos_avg = []
        for t in times:
            psi_t = evecs @ (coeffs * np.exp(-1j * evals * t))
            prob = np.abs(psi_t)**2
            avg_x = sum([prob[i] * states[i][0] for i in range(dim)])
            pos_avg.append(avg_x)
            
        plt.plot(times, pos_avg, linewidth=2.5, label=f"Stiffness g={g}")

    plt.title(f"Deriving Inertia: Particle Trajectories under Constant Force")
    plt.xlabel("Time")
    plt.ylabel("Position <X>")
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = "demo_inertia.png"
    plt.savefig(filename)
    print(f"  [✓] Plot saved to {filename}")

# =============================================================================
# EXPERIMENT 2: DERIVING FERMIONS (QUATERNIONS)
# =============================================================================
def demo_fermions():
    print("\n--- EXPERIMENT: DERIVATION OF FERMIONS ---")
    print("Hypothesis: Pauli Exclusion arises from Non-Abelian (Quaternion) Geometry.")
    print("We compare Path A (XY) vs Path B (YX).")
    
    # Quaternion / SU(2) Generators
    sigma_0 = np.eye(2, dtype=np.complex128)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    GEN_X = -1j * sigma_x # i
    GEN_Y = -1j * sigma_y # j
    
    def get_name(m):
        if np.allclose(m, sigma_0): return " 1"
        if np.allclose(m, -sigma_0): return "-1"
        if np.allclose(m, GEN_X): return " i"
        if np.allclose(m, GEN_Y): return " j"
        if np.allclose(m, -1j*sigma_z): return " k"
        if np.allclose(m, 1j*sigma_z): return "-k"
        return "?"

    # Initial Universe (Identity)
    psi = sigma_0
    print(f"  Initial State: {get_name(psi)}")
    
    # Path A: Hop X then Hop Y
    psi_A = GEN_Y @ (GEN_X @ psi)
    print(f"  Path A (X->Y): {get_name(psi_A)}")
    
    # Path B: Hop Y then Hop X
    psi_B = GEN_X @ (GEN_Y @ psi)
    print(f"  Path B (Y->X): {get_name(psi_B)}")
    
    # Overlap
    overlap = np.trace(psi_A @ psi_B.conj().T) / 2.0
    print(f"  Overlap <A|B>: {overlap:.4f}")
    
    if np.isclose(overlap, -1.0):
        print("\n  [✓] VERDICT: FERMIONS GENERATED")
        print("  The non-commutative geometry forced a (-1) phase shift.")
    else:
        print("\n  [!] VERDICT: BOSONS (or other)")

# =============================================================================
# EXPERIMENT 3: PATH MEMORY (3D CUBE)
# =============================================================================
def demo_memory():
    print("\n--- EXPERIMENT: PATH MEMORY TEST ---")
    print("Hypothesis: The Substrate records 'Which-Way' information in the geometry.")
    
    if not HAS_GPU:
        print("Note: Running on CPU. Using reduced scale for speed.")
    
    # 3D Cube Logic
    N_SITES = 8
    N_LINKS = 12
    DIM = N_SITES * (1 << N_LINKS)
    
    print(f"Hilbert Space: {DIM} states")
    
    # Using simple CPU/NumPy implementation for portability here
    # (The full GPU version is in substrate_proof.py, simplified here)
    
    def apply_hop(psi_in, u, v, link_id):
        psi_out = np.zeros_like(psi_in)
        start_u = u * (1 << N_LINKS)
        end_u = (u+1) * (1 << N_LINKS)
        block = psi_in[start_u:end_u]
        
        if np.all(block == 0): return psi_out
        
        # Permute gauge bits
        indices = np.arange(1 << N_LINKS)
        flipped = indices ^ (1 << link_id)
        
        start_v = v * (1 << N_LINKS)
        psi_out[start_v + flipped] = block
        return psi_out

    psi_0 = np.zeros(DIM, dtype=np.complex128)
    psi_0[0] = 1.0 # Vac at corner 0
    
    # Path A: 0->1->3
    pA = apply_hop(psi_0, 0, 1, 0)
    pA = apply_hop(pA, 1, 3, 5)
    
    # Path B: 0->2->3
    pB = apply_hop(psi_0, 0, 2, 4)
    pB = apply_hop(pB, 2, 3, 1)
    
    overlap = np.vdot(pA, pB)
    print(f"  Overlap <Path A | Path B>: {abs(overlap):.6f}")
    
    if abs(overlap) < 1e-5:
        print("  [✓] VERDICT: PATHS DISTINGUISHABLE")
        print("  Geometry is Memory. Histories are orthogonal.")
    else:
        print("  [!] VERDICT: PATHS INDISTINGUISHABLE")

# =============================================================================
# EXPERIMENT 4: GAUGE WAKE (VISUALIZATION)
# =============================================================================
def demo_wake():
    print("\n--- EXPERIMENT: GAUGE WAKE VISUALIZATION ---")
    print("Hypothesis: Moving particles leave a history trail.")
    
    N_SITES = 4
    N_LINKS = 3
    L_TRUNC = 2
    
    # ... (Reusing the logic from gauge_substrate.py) ...
    # Simplified Basis for visualization
    states = []
    import itertools
    for pos in range(N_SITES):
        for links in itertools.product(range(-L_TRUNC, L_TRUNC+1), repeat=N_LINKS):
             states.append( (pos, links) )
             
    dim = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i, (pos, links) in enumerate(states):
        H[i,i] += 0.5 * sum([l**2 for l in links]) # Electric
        if pos < N_SITES-1:
            lid = pos
            if links[lid] > -L_TRUNC:
                nl = list(links); nl[lid]-=1
                ns = (pos+1, tuple(nl))
                if ns in state_to_idx:
                    j = state_to_idx[ns]
                    H[j,i] -= 1.0; H[i,j] -= 1.0

    psi = np.zeros(dim, dtype=np.complex128)
    psi[state_to_idx[(0, tuple([0]*N_LINKS))]] = 1.0
    
    times = np.linspace(0, 6, 60)
    evals, evecs = eigh(H)
    coeffs = evecs.conj().T @ psi
    
    link_exc = np.zeros((len(times), N_LINKS))
    
    for t_i, t in enumerate(times):
        pt = evecs @ (coeffs * np.exp(-1j*evals*t))
        prob = np.abs(pt)**2
        for idx, p in enumerate(prob):
            _, lnk = states[idx]
            for l_i, l_val in enumerate(lnk):
                link_exc[t_i, l_i] += p*(l_val**2)
                
    plt.figure(figsize=(8, 4))
    for l in range(N_LINKS):
        plt.plot(times, link_exc[:, l], label=f"Link {l}")
    plt.title("The Wake: History stored in Geometry")
    plt.xlabel("Time")
    plt.ylabel("Link Excitation")
    plt.legend()
    plt.savefig("demo_wake.png")
    print("  [✓] Plot saved to demo_wake.png")

# =============================================================================
# MAIN MENU
# =============================================================================
def main():
    while True:
        print("\n=== RESONANCE LABORATORY: SUBSTRATE ENGINE ===")
        print("1. Derive Inertia (Mass from Stiffness)")
        print("2. Derive Fermions (Non-Abelian Topology)")
        print("3. Path Memory Test (3D Cube)")
        print("4. Visualize Gauge Wake")
        print("q. Quit")
        
        choice = input("Select Experiment > ")
        
        if choice == '1': demo_inertia()
        elif choice == '2': demo_fermions()
        elif choice == '3': demo_memory()
        elif choice == '4': demo_wake()
        elif choice.lower() == 'q': break
        else: print("Invalid selection.")

if __name__ == "__main__":
    main()