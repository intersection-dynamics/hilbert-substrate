import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import itertools

def analyze_recovered_unitary():
    print("--- EXTRACTING THE RECOVERED UNITARY ---")
    N = 3
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    # --- 1. Setup ---
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Paulis = [I, X, Y, Z]
    
    # Basis generation
    pauli_basis_matrices = []
    pauli_basis_weights = []
    for indices in itertools.product(range(4), repeat=N):
        term = Paulis[indices[0]]
        for k in range(1, N):
            term = np.kron(term, Paulis[indices[k]])
        weight = sum(1 for i in indices if i != 0)
        pauli_basis_matrices.append(term)
        pauli_basis_weights.append(weight)
    pauli_basis_weights = np.array(pauli_basis_weights)

    # Cost Function
    def get_coeffs(H_matrix):
        dim = 2**N
        coeffs = []
        for P in pauli_basis_matrices:
            c = np.real(np.trace(H_matrix @ P)) / dim
            coeffs.append(c)
        return np.array(coeffs)

    def get_locality_cost(H_matrix):
        coeffs = get_coeffs(H_matrix)
        norm_sq = np.sum(coeffs**2)
        if norm_sq < 1e-12: return 0.0
        return np.sum((pauli_basis_weights ** 4) * (coeffs ** 2)) / norm_sq

    # --- 2. Build & Scramble ---
    print("Recreating Scrambled Hamiltonian...")
    H_target = np.zeros((2**N, 2**N), dtype=complex)
    # Heisenberg Chain
    def add_term(op, i, j):
        ops = [I]*N
        ops[i]=op; ops[j]=op
        term = ops[0]
        for k in range(1,N): term = np.kron(term, ops[k])
        return term
    for i in range(N-1):
        H_target += add_term(X, i, i+1) + add_term(Y, i, i+1) + add_term(Z, i, i+1)

    # Scramble
    np.random.seed(42)
    G = np.random.randn(2**N, 2**N) + 1j * np.random.randn(2**N, 2**N)
    G = G + G.conj().T
    U_scramble = expm(1j * G * 0.2)
    H_scrambled = U_scramble @ H_target @ U_scramble.conj().T

    # --- 3. Optimize (Recover) ---
    print("Running Optimization...")
    generators = pauli_basis_matrices[1:] 
    
    def objective(params):
        A = np.zeros((2**N, 2**N), dtype=complex)
        for k, theta in enumerate(params):
            A += theta * generators[k]
        U_guess = expm(-1j * A)
        H_trial = U_guess @ H_scrambled @ U_guess.conj().T
        return get_locality_cost(H_trial)

    x0 = np.zeros(len(generators))
    # Use fewer iterations if just checking, but enough to get the result
    res = minimize(objective, x0, method='BFGS', options={'maxiter': 100})
    
    # Get U_opt
    params_opt = res.x
    A_opt = np.zeros((2**N, 2**N), dtype=complex)
    for k, theta in enumerate(params_opt):
        A_opt += theta * generators[k]
    U_opt = expm(-1j * A_opt)

    # --- 4. Analyze V = U_opt * U_scramble ---
    # V maps (Target Frame) -> (Recovered Frame)
    # Because H_rec = U_opt H_scr U_opt^dag 
    #              = U_opt (U_scr H_tar U_scr^dag) U_opt^dag
    #              = (U_opt U_scr) H_tar (U_opt U_scr)^dag
    # Let V = U_opt @ U_scramble. 
    # If H_rec is simpler than H_tar, then V is NOT Identity. 
    # V is the transformation from the "Spatial Site Basis" to the "Quasiparticle Basis".
    
    V = U_opt @ U_scramble
    
    print("\nOptimization Result:")
    print(f"Recovered Cost: {res.fun:.4f}")
    
    # Let's inspect what the "Recovered Basis States" look like in the "Target (Site) Basis".
    # The basis states of the recovered frame are |k>_rec.
    # In the target frame, these are |psi_k> = V^dagger |k>.
    # Why? H_rec = V H_tar V^dag. 
    # If H_rec |k> = E |k>, then V H_tar V^dag |k> = E |k> 
    # => H_tar (V^dag |k>) = E (V^dag |k>).
    # So V^dag |k> are the eigenstates of H_tar (approx) or the basis states of the simple form.
    
    print("\n--- BASIS ANALYSIS ---")
    print("Mapping: Basis State (Rec) -> State Vector (Target/Site Frame)")
    
    basis_labels = ["000", "001", "010", "011", "100", "101", "110", "111"]
    
    for k in range(2**N):
        # Create computational basis vector |k>
        vec_rec = np.zeros(2**N, dtype=complex)
        vec_rec[k] = 1.0
        
        # Transform to Target Frame
        vec_target = V.conj().T @ vec_rec # V^dagger |k>
        
        # Clean up for printing (remove small phases/values)
        def clean_complex(z):
            if abs(z) < 1e-3: return 0j
            return z
        
        # Identify significant components
        components = []
        for i in range(2**N):
            val = vec_target[i]
            if abs(val) > 0.1:
                components.append(f"{abs(val):.2f}*|{basis_labels[i]}>")
        
        comp_str = " + ".join(components)
        print(f"|{basis_labels[k]}_rec>  =>  {comp_str}")

    # Check for Singlet/Bell structure?
    # Singlet on 0-1: (|01> - |10>)/sqrt(2) approx 0.7
    # Triplet: (|01> + |10>)
    
    # We can also check the matrix V itself roughly
    print("\n--- V_eff (Modulus) ---")
    print(np.abs(V))

    return

analyze_recovered_unitary()