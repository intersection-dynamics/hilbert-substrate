"""
OuroborosChain_v2.py
The Hilbert Substrate Laboratory: Experiment 2 (Gravity & Event Horizons)

Purpose:
  To test the hypothesis that 'Mass' (High Observational Stability) creates
  'Spacetime Curvature' via a Topological Break (Event Horizon).

Method:
  1. Create a 1D chain of Ouroboros Cells.
  2. Sweep through increasing 'Clamping Pressures' on the middle cell.
  3. Use the TOPOLOGICAL METRIC: Distance = 1 / (Entanglement Shortcuts).
  4. Watch for the 'Event Horizon' where distance diverges to infinity.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import itertools

# --- CONFIGURATION ---
NUM_QUBITS_PER_CELL = 3
CHAIN_LENGTH = 5
DIM = 2**NUM_QUBITS_PER_CELL
MAX_ITERATIONS = 20
CONVERGENCE_THRESHOLD = 1e-3

# --- 1. PHYSICS ENGINE ---

def get_pauli_basis(n):
    paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), 
              np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    basis_mats, weights = [], []
    for p in itertools.product(range(4), repeat=n):
        weight = sum(1 for idx in p if idx != 0)
        mat = paulis[p[0]]
        for i in range(1, n):
            mat = np.kron(mat, paulis[p[i]])
        basis_mats.append(mat)
        weights.append(weight)
    return basis_mats, weights

def create_heisenberg_hamiltonian(n):
    ops = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n - 1):
        for op in ops:
            term = [np.eye(2)] * n
            term[i] = op
            term[i+1] = op
            mat = term[0]
            for k in range(1, n): mat = np.kron(mat, term[k])
            H += mat
    return H

def participation_ratio(U):
    ipr_sum = 0
    for i in range(DIM):
        ipr_sum += np.sum(np.abs(U[:, i])**4)
    return ipr_sum / DIM

def locality_cost(H_rotated, basis_mats, weights):
    cost = 0.0
    for mat, w in zip(basis_mats, weights):
        if w == 0: continue
        coeff = np.abs(np.trace(mat @ H_rotated) / DIM)
        penalty = np.exp(w - 1) if w > 1 else 1.0
        cost += (coeff**2) * penalty
    return cost

def stability_cost(U):
    # Cost to satisfy the environment: Maximize Localization (IPR)
    return 1.0 / participation_ratio(U)

def parameterize_unitary(params, generators):
    H_gen = np.zeros((DIM, DIM), dtype=complex)
    for p, G in zip(params, generators): H_gen += p * G
    return expm(-1j * H_gen)

def objective(params, generators, H_orig, basis_mats, weights, alpha, beta):
    U = parameterize_unitary(params, generators)
    H_new = U @ H_orig @ U.conj().T
    return alpha * locality_cost(H_new, basis_mats, weights) + beta * stability_cost(U)

# --- 2. THE CELL CLASS (Updated Metric) ---

class SubstrateCell:
    def __init__(self, cell_id, is_mass=False, mass_pressure=10.0):
        self.id = cell_id
        self.is_mass = is_mass
        self.mass_pressure = mass_pressure # Configurable pressure
        self.basis_mats, self.weights = get_pauli_basis(NUM_QUBITS_PER_CELL)
        self.H_spatial = create_heisenberg_hamiltonian(NUM_QUBITS_PER_CELL)
        self.generators = [mat for mat, w in zip(self.basis_mats, self.weights) if 0 < w <= 2]
        self.params = np.random.normal(0, 0.1, len(self.generators))
        self.current_ipr = 0.0
        self.metric_length = 0.0
        
    def relax(self, alpha=1.0):
        # 1. Determine Pressure (Beta)
        U_curr = parameterize_unitary(self.params, self.generators)
        self.current_ipr = participation_ratio(U_curr)
        
        # If Mass, use external clamping. If Vacuum, use self-consistent feedback.
        current_beta = self.mass_pressure if self.is_mass else 5.0 * (self.current_ipr**2)
            
        # 2. Optimize
        res = minimize(objective, self.params, 
                       args=(self.generators, self.H_spatial, self.basis_mats, self.weights, alpha, current_beta),
                       method='COBYLA', options={'maxiter': 40, 'tol': 1e-3})
        self.params = res.x
        
        # 3. CALCULATE TOPOLOGICAL METRIC
        # Distance = 1 / (Entanglement Shortcuts)
        shortcuts = 1.0 - self.current_ipr
        # Avoid division by zero with epsilon
        self.metric_length = 1.0 / (shortcuts + 1e-6)
        
        return self.metric_length

# --- 3. THE EXPERIMENT ---

def run_chain_experiment():
    print("--- Ouroboros Experiment 2: Event Horizons ---")
    
    # 1. Establish Vacuum Baseline
    print("\n[1] Calibrating VACUUM Baseline...")
    vacuum_chain = [SubstrateCell(i) for i in range(CHAIN_LENGTH)]
    for _ in range(10): 
        [c.relax() for c in vacuum_chain]
    
    vacuum_dist = sum(c.metric_length for c in vacuum_chain)
    print(f"    -> Vacuum Metric Distance: {vacuum_dist:.4f}")
    
    # 2. The Black Hole Sweep
    pressures = [10.0, 50.0, 100.0, 500.0]
    
    print("\n[2] Initiating Gravity Sweep...")
    for p in pressures:
        gravity_chain = [SubstrateCell(i, is_mass=(i==2), mass_pressure=p) for i in range(CHAIN_LENGTH)]
        
        # Relax chain
        for _ in range(10): 
            [c.relax() for c in gravity_chain]
            
        gravity_dist = sum(c.metric_length for c in gravity_chain)
        stretch = ((gravity_dist - vacuum_dist) / vacuum_dist) * 100
        
        # Analyze Center Cell (The Singularity)
        center_ipr = gravity_chain[2].current_ipr
        
        print(f"\n--- Clamping Pressure: {p} ---")
        print(f"    Mass Cell IPR: {center_ipr:.6f} (1.0 = Pure Spatial)")
        print(f"    Total Distance: {gravity_dist:.4f}")
        print(f"    Time Dilation: +{stretch:.2f}%")
        
        if stretch > 500:
            print("    [!!!] CRITICALITY: EVENT HORIZON DETECTED.")

if __name__ == "__main__":
    run_chain_experiment()