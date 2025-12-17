import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import itertools
import time

def run_experiment_3():
    print("="*60)
    print("      EXPERIMENT 3: THE BATTLE OF THE BASES")
    print("      Hypothesis: Space emerges via Environmental Selection")
    print("="*60)
    
    # --- CONFIGURATION ---
    N = 4                   # System size (qubits)
    dim_sys = 2**N
    coupling_strength = 5.0 # Strong system-environment coupling
    
    # --- 1. DEFINITIONS & OPERATORS ---
    print("\n[1] Initializing Physics Engine...")
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Paulis = [I, X, Y, Z]

    def kron_list(ops):
        """Helper to compute Kronecker product of a list of matrices"""
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    # Generate Pauli Basis for Cost Function Calculation
    # (Used to evaluate locality)
    pauli_basis_matrices = []
    pauli_basis_weights = []
    for indices in itertools.product(range(4), repeat=N):
        term = kron_list([Paulis[i] for i in indices])
        weight = sum(1 for i in indices if i != 0)
        pauli_basis_matrices.append(term)
        pauli_basis_weights.append(weight)
    pauli_basis_weights = np.array(pauli_basis_weights)

    # Locality Cost Function L[phi; H]
    def get_locality_cost(H_mat):
        # Project H onto Pauli Basis to get coefficients c_alpha
        coeffs = np.array([np.real(np.trace(H_mat @ P))/dim_sys for P in pauli_basis_matrices])
        norm_sq = np.sum(coeffs**2)
        if norm_sq < 1e-12: return 0.0
        # Cost = Sum(weight^4 * c_alpha^2) / Sum(c_alpha^2)
        return np.sum((pauli_basis_weights ** 4) * (coeffs ** 2)) / norm_sq

    # --- 2. PREPARE HAMILTONIANS ---
    print("[2] Constructing Substrate Hamiltonians...")
    
    # System Hamiltonian: 1D Heisenberg Chain (The "Laws of Physics")
    H_sys = np.zeros((dim_sys, dim_sys), dtype=complex)
    for i in range(N - 1):
        ops_X = [I]*N; ops_X[i]=X; ops_X[i+1]=X
        ops_Y = [I]*N; ops_Y[i]=Y; ops_Y[i+1]=Y
        ops_Z = [I]*N; ops_Z[i]=Z; ops_Z[i+1]=Z
        H_sys += kron_list(ops_X) + kron_list(ops_Y) + kron_list(ops_Z)

    print(f"    > Spatial Basis Cost: {get_locality_cost(H_sys):.4f}")

    # Scramble the Hamiltonian (Simulate the raw, unstructured substrate)
    np.random.seed(42)
    G = np.random.randn(dim_sys, dim_sys) + 1j * np.random.randn(dim_sys, dim_sys)
    G = G + G.conj().T
    U_scramble = expm(1j * G * 0.2)
    H_scrambled = U_scramble @ H_sys @ U_scramble.conj().T
    
    print(f"    > Scrambled Cost:     {get_locality_cost(H_scrambled):.4f}")

    # --- 3. RECOVER OPTIMAL (HARMONION) BASIS ---
    print("\n[3] Optimizing Factorization (Seeking Harmonions)...")
    
    # We seek U_opt such that H_rec = U_opt H_scr U_opt^dag is maximally local.
    # Generators for the optimization (SU(2^N) algebra, excluding Identity)
    generators = pauli_basis_matrices[1:] 

    def objective(params):
        A = np.zeros((dim_sys, dim_sys), dtype=complex)
        for k, th in enumerate(params):
            A += th * generators[k]
        U = expm(-1j * A)
        H_trial = U @ H_scrambled @ U.conj().T
        return get_locality_cost(H_trial)

    # Run Optimization (BFGS)
    # Note: For production, maxiter can be increased, but 60 is usually sufficient for N=3
    x0 = np.zeros(len(generators))
    start_time = time.time()
    res = minimize(objective, x0, method='BFGS', options={'maxiter': 60})
    print(f"    > Optimization converged in {time.time()-start_time:.1f}s")
    print(f"    > Harmonion Basis Cost: {res.fun:.4f}")

    # Reconstruct the Transform V
    # V maps the Spatial Frame -> Harmonion Frame
    params_opt = res.x
    A_opt = np.zeros((dim_sys, dim_sys), dtype=complex)
    for k, th in enumerate(params_opt):
        A_opt += th * generators[k]
    U_opt = expm(-1j * A_opt)
    
    # The total transformation from Spatial to Recovered Frame
    V = U_opt @ U_scramble 

    # --- 4. THE DEATHMATCH (DECOHERENCE SIMULATION) ---
    print("\n[4] Simulating Environmental Monitoring...")
    
    # Define Environment: N qubits (one for each system qubit)
    N_tot = 2 * N
    dim_tot = 2**N_tot
    
    # Interaction Hamiltonian: H_int = g * Sum (Z_sys_i (x) Z_env_i)
    # This interaction "measures" the system in the Z-basis (Spatial)
    H_int = np.zeros((dim_tot, dim_tot), dtype=complex)
    
    for i in range(N):
        ops = [I] * N_tot
        ops[i] = Z          # System qubit i
        ops[i + N] = Z      # Environment qubit i
        H_int += coupling_strength * kron_list(ops)

    # Total Hamiltonian for Evolution
    H_sys_ext = np.kron(H_sys, np.eye(2**N)) # System H acts only on System
    H_total = H_sys_ext + H_int

    # Prepare Initial States
    # 1. Spatial State: |100> (Localized in Space)
    psi_space_sys = np.zeros(dim_sys, dtype=complex)
    psi_space_sys[4] = 1.0 
    
    # 2. Harmonion State: |100>_harm (Localized in Harmonion Frame)
    # In the spatial frame, this is V^dag |100>
    psi_harm_sys = V.conj().T @ psi_space_sys
    
    # Environment State: |+++> (Maximally sensitive)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi_env = kron_list([plus]*N)
    
    # Full Initial States
    psi_A = np.kron(psi_space_sys, psi_env)
    psi_B = np.kron(psi_harm_sys, psi_env)
    
    # --- 5. RUN EVOLUTION ---
    times = np.linspace(0, 0.5, 6)
    
    print("\n    Results: Purity vs Time")
    print(f"    {'Time':<6} | {'Purity (Spatial)':<18} | {'Purity (Harmonion)':<18}")
    print("    " + "-"*50)
    
    results = []

    for t in times:
        U_t = expm(-1j * H_total * t)
        
        # Evolve State A (Spatial)
        psi_A_t = U_t @ psi_A
        # Partial Trace to get Reduced Rho
        psi_A_reshaped = psi_A_t.reshape((dim_sys, 2**N)) 
        rho_A = psi_A_reshaped @ psi_A_reshaped.conj().T
        purity_A = np.real(np.trace(rho_A @ rho_A))
        
        # Evolve State B (Harmonion)
        psi_B_t = U_t @ psi_B
        psi_B_reshaped = psi_B_t.reshape((dim_sys, 2**N))
        rho_B = psi_B_reshaped @ psi_B_reshaped.conj().T
        purity_B = np.real(np.trace(rho_B @ rho_B))
        
        print(f"    {t:.2f}   | {purity_A:.4f}             | {purity_B:.4f}")
        results.append((t, purity_A, purity_B))

    # --- 6. CONCLUSION ---
    print("\n[5] Analysis")
    final_space = results[-1][1]
    final_harm = results[-1][2]
    
    print(f"    Final Spatial Purity:   {final_space:.4f}")
    print(f"    Final Harmonion Purity: {final_harm:.4f}")
    
    if final_space > final_harm * 2.0:
        print("\n    CONCLUSION: CONFIRMED.")
        print("    The Spatial basis survived. The Harmonion basis was destroyed.")
        print("    Space is the 'Pointer Basis' selected by the environment.")
    else:
        print("\n    CONCLUSION: INCONCLUSIVE.")

if __name__ == "__main__":
    run_experiment_3()