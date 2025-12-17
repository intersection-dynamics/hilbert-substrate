import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import itertools
import time

def run_humpty_dumpty_test():
    # --- Configuration ---
    N = 3  # Keep small (3-4) for fast convergence. N=5+ requires GPU/PyTorch.
    print(f"--- THE HUMPTY DUMPTY TEST (N={N} Qubits) ---")

    # --- 1. Basic Operators ---
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Paulis = [I, X, Y, Z]
    
    # Generate all N-qubit Pauli strings
    print("Generating Pauli Basis...")
    pauli_basis_matrices = []
    pauli_basis_weights = []
    
    # Loop 4^N times to create the full operator basis
    for indices in itertools.product(range(4), repeat=N):
        # Tensor product: P_1 (x) P_2 (x) ...
        term = Paulis[indices[0]]
        for k in range(1, N):
            term = np.kron(term, Paulis[indices[k]])
        
        # Calculate Weight: Count non-Identity factors
        weight = sum(1 for i in indices if i != 0)
        
        pauli_basis_matrices.append(term)
        pauli_basis_weights.append(weight)

    pauli_basis_weights = np.array(pauli_basis_weights)

    # --- 2. Define Locality Cost Function ---
    def get_locality_cost(H_matrix):
        """
        Calculates L[phi; H] based on Eq 13 of 'From Hamiltonian to Geometry'.
        Cost = Sum (weight^4 * coefficient^2)
        We use weight^4 to heavily penalize non-local terms.
        """
        dim = 2**N
        # Project H onto Pauli Basis to get coefficients c_alpha
        # c_alpha = Tr(H * P_alpha) / dim
        # We compute this via Hilbert-Schmidt inner product
        coeffs = []
        for P in pauli_basis_matrices:
            c = np.real(np.trace(H_matrix @ P)) / dim
            coeffs.append(c)
        
        coeffs = np.array(coeffs)
        
        # Normalize to ensure scale invariance
        norm_sq = np.sum(coeffs**2)
        if norm_sq < 1e-12: return 0.0
        
        # The Cost Function
        cost = np.sum((pauli_basis_weights ** 4) * (coeffs ** 2)) / norm_sq
        return cost

    # --- 3. Construct Target Hamiltonian (Heisenberg Chain) ---
    print("Building 1D Heisenberg Chain...")
    H_target = np.zeros((2**N, 2**N), dtype=complex)
    
    # Helper for 2-body terms
    def add_term(op, i, j):
        ops = [I] * N
        ops[i] = op
        ops[j] = op
        term = ops[0]
        for k in range(1, N):
            term = np.kron(term, ops[k])
        return term

    # Open boundary conditions: H = Sum (XX + YY + ZZ)
    for i in range(N - 1):
        H_target += add_term(X, i, i+1)
        H_target += add_term(Y, i, i+1)
        H_target += add_term(Z, i, i+1)

    c_original = get_locality_cost(H_target)
    print(f"[Target] Heisenberg Locality Cost: {c_original:.4f}")

    # --- 4. Scramble the Hamiltonian ---
    print("Scrambling with random unitary...")
    np.random.seed(42) # Fixed seed for reproducibility
    
    # Random Hermitian generator
    G = np.random.randn(2**N, 2**N) + 1j * np.random.randn(2**N, 2**N)
    G = G + G.conj().T
    U_scramble = expm(1j * G * 0.1) # Moderate scramble
    
    H_scrambled = U_scramble @ H_target @ U_scramble.conj().T
    
    c_scrambled = get_locality_cost(H_scrambled)
    print(f"[Scrambled] Hamiltonian Cost:      {c_scrambled:.4f}")

    # --- 5. Optimization (The Recovery) ---
    print("Attempting algorithmic recovery (optimizing factorization)...")
    
    # Parameterize unitary U = exp(-i * Sum theta_k P_k)
    # We use the Pauli basis (excluding Identity) as generators for SU(2^N)
    generators = pauli_basis_matrices[1:] 
    num_params = len(generators)
    
    def objective_function(params):
        # 1. Construct Unitary from params
        # A = Sum theta_k P_k
        A = np.zeros((2**N, 2**N), dtype=complex)
        for k, theta in enumerate(params):
            A += theta * generators[k]
        
        U_guess = expm(-1j * A)
        
        # 2. Transform Hamiltonian
        H_trial = U_guess @ H_scrambled @ U_guess.conj().T
        
        # 3. Compute Cost
        return get_locality_cost(H_trial)

    # Initial guess: zero parameters (Identity)
    x0 = np.zeros(num_params)
    
    # Run Optimizer (BFGS is standard for smooth landscapes)
    start_time = time.time()
    res = minimize(objective_function, x0, method='BFGS', 
                   options={'maxiter': 200, 'disp': True})
    
    print(f"Optimization finished in {time.time()-start_time:.1f}s")
    print(f"Success: {res.success}")
    print(f"[Recovered] Hamiltonian Cost:      {res.fun:.4f}")

    # --- Conclusion ---
    print("\n--- RESULTS ---")
    print(f"Target Cost:    {c_original:.4f}")
    print(f"Scrambled Cost: {c_scrambled:.4f}")
    print(f"Recovered Cost: {res.fun:.4f}")
    
    if res.fun < c_scrambled:
        print("\nSUCCESS: The algorithm successfully reduced the locality cost,")
        print("effectively 'discovering' a more local factorization than the scrambled one.")
    else:
        print("\nFAIL: Optimization got stuck.")

if __name__ == "__main__":
    run_humpty_dumpty_test()