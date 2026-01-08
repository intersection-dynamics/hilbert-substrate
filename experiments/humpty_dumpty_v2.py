import numpy as np
from scipy.linalg import expm, norm
import itertools
import time

def run_humpty_dumpty_corrected(N=6):
    print(f"--- HUMPTY DUMPTY CORRECTED (N={N}) ---")
    
    # 1. Setup Pauli Basis
    start_time = time.time()
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Paulis = [I, X, Y, Z]
    
    pauli_mats = []
    pauli_weights = []
    
    # Generate Basis
    for indices in itertools.product(range(4), repeat=N):
        term = Paulis[indices[0]]
        for k in range(1, N):
            term = np.kron(term, Paulis[indices[k]])
        w = sum(1 for i in indices if i != 0)
        pauli_mats.append(term)
        pauli_weights.append(w)
        
    pauli_mats = np.array(pauli_mats)
    pauli_weights = np.array(pauli_weights)
    dim = 2**N
    print(f"Basis generated in {time.time() - start_time:.2f}s")

    # 2. Cost Function Engine
    def get_cost(H_in):
        # Decompose
        coeffs = np.real(np.einsum('ij,kji->k', H_in, pauli_mats)) / dim
        # Cost
        norm_sq = np.sum(coeffs**2)
        w4 = pauli_weights ** 4
        cost = np.sum(w4 * coeffs**2) / norm_sq
        return cost

    # 3. Build Spatial Hamiltonian
    print("Building Target (Spatial)...")
    H_target = np.zeros((dim, dim), dtype=complex)
    def add_term(op, i, j):
        ops = [I]*N; ops[i]=op; ops[j]=op
        t = ops[0]
        for k in range(1,N): t = np.kron(t, ops[k])
        return t
    
    for i in range(N-1):
        H_target += add_term(X,i,i+1) + add_term(Y,i,i+1) + add_term(Z,i,i+1)
        
    c_spatial = get_cost(H_target)
    print(f"Spatial Cost: {c_spatial:.4f}")

    # 4. The Eigenbasis Check (CORRECTED OPTION A)
    print("Checking Eigenbasis Consistency...")
    evals, evecs = np.linalg.eigh(H_target)
    
    # H_diag is the Hamiltonian in the eigenbasis
    H_diag = np.diag(evals)
    
    # WE MUST TRANSFORM BACK TO COMPUTE COST IN PAULI BASIS
    H_back = evecs @ H_diag @ evecs.conj().T
    
    c_eigen_check = get_cost(H_back)
    print(f"Eigenbasis Check (H_back): {c_eigen_check:.4f}")
    
    if abs(c_spatial - c_eigen_check) < 1e-4:
        print(">> SANITY CHECK PASSED: The Physics is consistent.")
    else:
        print(">> SANITY CHECK FAILED.")

if __name__ == "__main__":
    run_humpty_dumpty_corrected(N=6)