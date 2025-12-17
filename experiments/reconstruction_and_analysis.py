import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm, norm
import itertools
import time

def run_humpty_dumpty_analysis():
    # --- Configuration ---
    N = 3  # Qubits
    np.set_printoptions(precision=2, suppress=True)
    print(f"--- HUMPTY DUMPTY: GEOMETRY RECOVERY (N={N}) ---")

    # --- 1. Basic Setup ---
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Paulis = [I, X, Y, Z]
    
    # Generate Basis and Precompute Weights
    pauli_basis_matrices = []
    pauli_basis_weights = []   # |S|
    pauli_basis_indices = []   # List of active sites for each term
    
    print("Generating Operator Basis...")
    for indices in itertools.product(range(4), repeat=N):
        term = Paulis[indices[0]]
        for k in range(1, N):
            term = np.kron(term, Paulis[indices[k]])
        
        # Identify support: which indices are not Identity (0)
        active_sites = [i for i, op_idx in enumerate(indices) if op_idx != 0]
        weight = len(active_sites)
        
        pauli_basis_matrices.append(term)
        pauli_basis_weights.append(weight)
        pauli_basis_indices.append(active_sites)

    pauli_basis_weights = np.array(pauli_basis_weights)

    # --- 2. The Locality Cost Function (L) ---
    def get_coeffs(H_matrix):
        dim = 2**N
        coeffs = []
        for P in pauli_basis_matrices:
            # Hilbert-Schmidt inner product: Tr(H P) / dim
            c = np.real(np.trace(H_matrix @ P)) / dim
            coeffs.append(c)
        return np.array(coeffs)

    def get_locality_cost_from_coeffs(coeffs):
        # Normalize
        norm_sq = np.sum(coeffs**2)
        if norm_sq < 1e-12: return 0.0
        # Cost = Sum (|S|^4 * c_S^2) / Norm
        cost = np.sum((pauli_basis_weights ** 4) * (coeffs ** 2)) / norm_sq
        return cost

    # --- 3. Geometric Analysis Tool (Eq. 28) ---
    def get_interaction_matrix(H_matrix):
        """
        Computes the interaction weight w_ij between factors i and j.
        w_ij = Sum_{S containing i,j} ||H_S||
        """
        coeffs = get_coeffs(H_matrix)
        w_matrix = np.zeros((N, N))
        
        # We iterate through all Pauli terms
        for idx, c in enumerate(coeffs):
            if abs(c) < 1e-5: continue # Skip zero terms
            
            sites = pauli_basis_indices[idx]
            # If term acts on >= 2 sites, it contributes to edges
            if len(sites) >= 2:
                # Add contribution to all pairs in this term
                term_magnitude = abs(c) # ||H_S||
                for i in range(len(sites)):
                    for j in range(i + 1, len(sites)):
                        u, v = sites[i], sites[j]
                        w_matrix[u, v] += term_magnitude
                        w_matrix[v, u] += term_magnitude
                        
        return w_matrix

    # --- 4. Build Target (Heisenberg Chain) ---
    print("Building Target 1D Chain...")
    H_target = np.zeros((2**N, 2**N), dtype=complex)
    
    def add_term(op, i, j):
        ops = [I] * N
        ops[i] = op; ops[j] = op
        term = ops[0]
        for k in range(1, N):
            term = np.kron(term, ops[k])
        return term

    # Chain: 0-1, 1-2, etc.
    for i in range(N - 1):
        H_target += add_term(X, i, i+1) + add_term(Y, i, i+1) + add_term(Z, i, i+1)

    # --- 5. Scramble ---
    print("Scrambling Geometry...")
    np.random.seed(42)
    G = np.random.randn(2**N, 2**N) + 1j * np.random.randn(2**N, 2**N)
    G = G + G.conj().T
    U_scramble = expm(1j * G * 0.2) # Stronger scramble this time?
    H_scrambled = U_scramble @ H_target @ U_scramble.conj().T

    # --- 6. Recover (Optimize) ---
    print("Running Optimization to Recover Factorization...")
    generators = pauli_basis_matrices[1:] # SU(2^N) generators
    
    def objective(params):
        A = np.zeros((2**N, 2**N), dtype=complex)
        for k, theta in enumerate(params):
            A += theta * generators[k]
        U_guess = expm(-1j * A)
        H_trial = U_guess @ H_scrambled @ U_guess.conj().T
        coeffs = get_coeffs(H_trial)
        return get_locality_cost_from_coeffs(coeffs)

    x0 = np.zeros(len(generators))
    # BFGS optimization
    res = minimize(objective, x0, method='BFGS', options={'maxiter': 150})
    
    # Reconstruct Final Result
    params_opt = res.x
    A_opt = np.zeros((2**N, 2**N), dtype=complex)
    for k, theta in enumerate(params_opt):
        A_opt += theta * generators[k]
    U_opt = expm(-1j * A_opt)
    H_recovered = U_opt @ H_scrambled @ U_opt.conj().T

    # --- 7. ANALYSIS & OUTPUT ---
    print("\n" + "="*40)
    print("       GEOMETRY ANALYSIS (Interaction Graph w_ij)")
    print("="*40)

    def print_matrix(name, M):
        print(f"\n--- {name} Interaction Matrix ---")
        print("   " + "  ".join([f"Q{i}" for i in range(N)]))
        for i in range(N):
            row_str = f"Q{i} "
            for j in range(N):
                val = M[i,j]
                if i == j: char = " -  "
                elif val < 0.1: char = " .  " # Empty/Weak
                elif val < 1.0: char = f"{val:.1f} " # Weak coupling
                else: char = f"**{val:.1f}" # Strong coupling
                row_str += char
            print(row_str)

    # Analyze all three
    W_target = get_interaction_matrix(H_target)
    W_scrambled = get_interaction_matrix(H_scrambled)
    W_recovered = get_interaction_matrix(H_recovered)

    print_matrix("TARGET (Original Chain)", W_target)
    print_matrix("SCRAMBLED (Hidden Geometry)", W_scrambled)
    print_matrix("RECOVERED (Emergent Geometry)", W_recovered)

    print("\n" + "-"*40)
    print(f"Original Cost:  {get_locality_cost_from_coeffs(get_coeffs(H_target)):.4f}")
    print(f"Scrambled Cost: {get_locality_cost_from_coeffs(get_coeffs(H_scrambled)):.4f}")
    print(f"Recovered Cost: {res.fun:.4f}")
    
    if res.fun < 5.0:
        print("\nINTERPRETATION: The recovered cost is very low (<5).")
        print("This suggests the algorithm decoupled the system into free particles")
        print("or nearly non-interacting subsystems.")
    else:
        print("\nINTERPRETATION: The algorithm recovered a connected structure.")

if __name__ == "__main__":
    run_humpty_dumpty_analysis()