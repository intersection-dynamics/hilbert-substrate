"""
Ouroboros.py
The Hilbert Substrate Laboratory

Purpose:
  To numerically map the 'Tradeoff Functional' from the Hilbert Substrate Framework.
  It tests whether the self-consistent feedback loop between Dynamical Simplicity (alpha)
  and Observational Stability (beta) converges to a fixed point (Particle) or enters
  a limit cycle (Wave).

Method:
  1. Define a 3-qubit local Hamiltonian (Heisenberg Chain).
  2. Optimize a Unitary U (representing the factorization) to minimize Cost J.
  3. J = alpha * Locality_Cost + beta * Stability_Cost.
  4. CRITICAL STEP: 'beta' is updated dynamically based on the system's current localization.
     - If the system is Spatial (localized), beta INCREASES (Environment sees it).
     - If the system is Harmonion (delocalized), beta DECREASES (Environment loses it).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import itertools
import sys

# --- CONFIGURATION ---
NUM_QUBITS = 3
DIM = 2**NUM_QUBITS
MAX_ITERATIONS = 20  # How many Ouroboros loops to run
CONVERGENCE_THRESHOLD = 1e-4

# --- 1. QUANTUM PHYSICS ENGINE ---

def get_pauli_basis(n):
    """Generates Pauli matrices and their locality weights."""
    # I, X, Y, Z
    paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), 
              np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    
    basis_mats = []
    weights = []
    
    for p in itertools.product(range(4), repeat=n):
        weight = sum(1 for idx in p if idx != 0) # Hamming weight
        
        mat = paulis[p[0]]
        for i in range(1, n):
            mat = np.kron(mat, paulis[p[i]])
        
        basis_mats.append(mat)
        weights.append(weight)
        
    return basis_mats, weights

def create_heisenberg_hamiltonian(n):
    """Creates a standard 1D Heisenberg chain (Local)."""
    # H = sum (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    
    ops = [sx, sy, sz]
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    for i in range(n - 1): # Open boundary conditions
        for op in ops:
            # Construct op_i * op_{i+1}
            term_list = [np.eye(2)] * n
            term_list[i] = op
            term_list[i+1] = op
            
            term = term_list[0]
            for k in range(1, n):
                term = np.kron(term, term_list[k])
            H += term
    return H

# --- 2. COST FUNCTIONS ---

def locality_cost(H_rotated, basis_mats, weights):
    """
    Measures Dynamical Simplicity (L).
    Penalizes terms with high Pauli weight (non-local interactions).
    """
    cost = 0.0
    # Project H onto Pauli basis
    # Trace(A @ B) / dim gives coefficient
    for mat, w in zip(basis_mats, weights):
        if w == 0: continue # Ignore identity
        
        coeff = np.abs(np.trace(mat @ H_rotated) / DIM)
        
        # Heuristic: Exponential penalty for non-locality
        # This matches the 'locality cost' definition in the paper
        penalty = np.exp(w - 1) if w > 1 else 1.0
        cost += (coeff**2) * penalty
        
    return cost

def participation_ratio(U):
    """
    Measures how 'delocalized' the basis is relative to the computational basis.
    Inverse Participation Ratio (IPR).
    High IPR = Localized (Spatial)
    Low IPR = Delocalized (Harmonion)
    """
    # We look at the columns of U. If U is Identity, IPR is max.
    # If U is Fourier Transform, IPR is min.
    # We average the IPR of the basis vectors.
    ipr_sum = 0
    for i in range(DIM):
        # Column i is the i-th basis vector
        psi = U[:, i]
        # Sum of fourth powers of amplitudes measures localization
        ipr = np.sum(np.abs(psi)**4)
        ipr_sum += ipr
    
    return ipr_sum / DIM

def stability_cost(U):
    """
    Measures Observational Stability (D).
    We want to MAXIMIZE localization (IPR).
    So Cost = 1 / IPR.
    """
    ipr = participation_ratio(U)
    # Avoid division by zero, though IPR >= 1/DIM
    return 1.0 / ipr

# --- 3. OPTIMIZATION ---

def parameterize_unitary(params, generators):
    """Reconstructs U from parameters using a simplified generator set."""
    U = np.eye(DIM, dtype=complex)
    # Using a Trotterized expansion for speed, or simple linear sum exponentiation
    # params is a vector of coefficients for the generators
    H_gen = np.zeros((DIM, DIM), dtype=complex)
    for p, G in zip(params, generators):
        H_gen += p * G
    
    return expm(-1j * H_gen)

def objective(params, generators, H_orig, basis_mats, weights, alpha, beta):
    U = parameterize_unitary(params, generators)
    
    # Transform H into the new factorization frame
    H_new = U @ H_orig @ U.conj().T
    
    L = locality_cost(H_new, basis_mats, weights)
    D = stability_cost(U)
    
    return alpha * L + beta * D

# --- 4. THE OUROBOROS LOOP ---

def main():
    print("--- Ouroboros.py: The Hilbert Substrate Laboratory ---")
    print(f"System: {NUM_QUBITS} Qubits")
    
    # Initialize Physics
    basis_mats, weights = get_pauli_basis(NUM_QUBITS)
    H_spatial = create_heisenberg_hamiltonian(NUM_QUBITS)
    
    # Create generators for the unitary optimization (Generalized Pauli rotations)
    # We use a subset of 2-local Paulis to define the rotation manifold
    generators = [mat for mat, w in zip(basis_mats, weights) if 0 < w <= 2]
    num_params = len(generators)
    
    # Starting State: Random Factorization
    # We start with params = 0 (Spatial Basis) to see if it holds,
    # OR start random to see if it finds it. Let's start slight perturbed.
    current_params = np.random.normal(0, 0.1, num_params)
    
    # Hyperparameters
    alpha = 1.0  # Simplicity is constant physics
    
    # Data Tracking
    betas = []
    costs = []

    print("\n[!] Initiating Self-Consistent Feedback Loop...")
    
    for t in range(MAX_ITERATIONS):
        # 1. CONSTRUCT CURRENT REALITY
        U_current = parameterize_unitary(current_params, generators)
        
        # 2. MEASURE VISIBILITY (Calculate Self-Consistent Beta)
        # How "Spatial" is the current factorization?
        # IPR ranges from 1/DIM (wave) to 1.0 (particle)
        current_ipr = participation_ratio(U_current)
        
        # FEEDBACK LAW:
        # If IPR is high (Localized), the environment couples strongly -> Beta High.
        # If IPR is low (Delocalized), the environment decouples -> Beta Low.
        # We amplify this to see the transition.
        current_beta = 5.0 * (current_ipr**2) 
        
        betas.append(current_beta)
        
        print(f"Cycle {t+1}: Localization(IPR)={current_ipr:.4f} -> Pressure(Beta)={current_beta:.4f}")
        
        # 3. OPTIMIZE FOR NEXT INSTANT
        # The substrate re-arranges to minimize the NEW cost function
        res = minimize(objective, current_params, 
                       args=(generators, H_spatial, basis_mats, weights, alpha, current_beta),
                       method='COBYLA', # Gradient-free usually safer for complex landscapes
                       options={'maxiter': 50, 'tol': 1e-3})
        
        new_params = res.x
        
        # 4. CHECK MOVEMENT (Dynamics)
        # Euclidean distance in parameter space (approximate metric on manifold)
        delta = np.linalg.norm(new_params - current_params)
        print(f"      -> Substrate Shift (Delta): {delta:.6f}")
        
        if delta < CONVERGENCE_THRESHOLD:
            print(f"\n[+] CONVERGENCE ACHIEVED at Cycle {t+1}")
            print("    The Ouroboros has caught its tail.")
            print("    Status: STABLE PARTICLE / GEOMETRY EMERGED.")
            break
            
        current_params = new_params
    else:
        print(f"\n[-] NO CONVERGENCE after {MAX_ITERATIONS} cycles.")
        print("    The system is oscillating or chaotic.")
        print("    Status: LIMIT CYCLE / WAVE / TIME CRYSTAL.")

if __name__ == "__main__":
    main()