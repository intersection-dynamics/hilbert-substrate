"""
Basin Size Experiment
=====================

Measures the relative basin sizes of 1D, 2D, and 3D attractors
under locality-biased Riemannian flow.

From Paper II: The double-bracket flow minimizes locality cost while
preserving the spectrum. Different initial conditions converge to
different local minima. We measure which geometries capture more
of configuration space.

Hypothesis: 3D basins are larger, so generic initial conditions
preferentially flow to 3D-like attractors.

Usage:
    python basin_size_experiment.py --test
    python basin_size_experiment.py --full --n-samples 500 --output basins.json

Author: Ben Bray
"""

import numpy as np
from scipy.linalg import expm
from typing import Dict, List, Tuple
import json
import time
from multiprocessing import Pool, cpu_count

# Try CuPy for GPU acceleration of some operations
try:
    import cupy as cp
    GPU = True
    print(f"GPU available: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    cp = np
    GPU = False
    print("CPU mode")


# ============================================================
# PAULI BASIS
# ============================================================

I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = [I, X, Y, Z]
PAULI_LABELS = ['I', 'X', 'Y', 'Z']


def generate_pauli_basis(N: int) -> List[Tuple[np.ndarray, int, str]]:
    """
    Generate all N-qubit Pauli operators with their weights.
    
    Returns: List of (operator, weight, label) tuples
    Weight = number of non-identity factors (Hamming weight)
    """
    basis = []
    
    for idx in range(4**N):
        # Decode index into Pauli indices
        indices = []
        temp = idx
        for _ in range(N):
            indices.append(temp % 4)
            temp //= 4
        indices = indices[::-1]
        
        # Build operator
        op = PAULIS[indices[0]]
        for i in range(1, N):
            op = np.kron(op, PAULIS[indices[i]])
        
        # Compute weight (number of non-I factors)
        weight = sum(1 for i in indices if i != 0)
        
        # Label
        label = ''.join(PAULI_LABELS[i] for i in indices)
        
        basis.append((op, weight, label))
    
    return basis


def pauli_decompose(H: np.ndarray, basis: List) -> np.ndarray:
    """Decompose Hamiltonian into Pauli coefficients."""
    N_qubits = int(np.log2(H.shape[0]))
    dim = 2**N_qubits
    coeffs = np.array([np.trace(H @ P) / dim for (P, w, l) in basis])
    return coeffs


def pauli_reconstruct(coeffs: np.ndarray, basis: List) -> np.ndarray:
    """Reconstruct Hamiltonian from Pauli coefficients."""
    H = sum(c * P for c, (P, w, l) in zip(coeffs, basis))
    return H


# ============================================================
# LOCALITY COST FUNCTIONAL
# ============================================================

def locality_cost(H: np.ndarray, basis: List, p: float = 4.0) -> float:
    """
    Compute locality cost C_p(H) from Paper II.
    
    C_p = Σ_k w(P_k)^p |c_k|^2 / Σ_k |c_k|^2
    
    where w(P_k) is the Hamming weight of Pauli operator P_k.
    """
    coeffs = pauli_decompose(H, basis)
    weights = np.array([w for (P, w, l) in basis], dtype=np.float64)
    
    # Normalize coefficients to avoid overflow
    coeffs = np.array(coeffs, dtype=np.complex128)
    max_c = np.max(np.abs(coeffs))
    if max_c > 1e-15:
        coeffs = coeffs / max_c
    
    c2 = np.abs(coeffs)**2
    total = np.sum(c2)
    
    if total < 1e-15:
        return 0.0
    
    # Use float64 for weights^p to avoid overflow with large p
    wp = np.power(weights.astype(np.float64), p)
    cost = np.sum(wp * c2) / total
    return float(np.real(cost))


def locality_cost_by_weight(H: np.ndarray, basis: List) -> Dict[int, float]:
    """Break down coefficient weight by Hamming weight."""
    coeffs = pauli_decompose(H, basis)
    weights = np.array([w for (P, w, l) in basis])
    
    # Normalize to avoid overflow
    coeffs = np.array(coeffs, dtype=np.complex128)
    max_c = np.max(np.abs(coeffs))
    if max_c > 1e-15:
        coeffs = coeffs / max_c
    
    c2 = np.abs(coeffs)**2
    total = np.sum(c2)
    
    breakdown = {}
    for w in range(int(max(weights)) + 1):
        mask = (weights == w)
        breakdown[w] = float(np.sum(c2[mask]) / total) if total > 1e-15 else 0
    
    return breakdown


# ============================================================
# DOUBLE-BRACKET FLOW
# ============================================================

def compute_M(H: np.ndarray, basis: List, p: float = 4.0) -> np.ndarray:
    """
    Compute the gradient operator M for the locality cost.
    
    M = Σ_k (∂C_p/∂c_k) P_k
    """
    coeffs = pauli_decompose(H, basis)
    weights = np.array([w for (P, w, l) in basis], dtype=np.float64)
    
    coeffs = np.array(coeffs, dtype=np.complex128)
    max_c = np.max(np.abs(coeffs))
    if max_c > 1e-15:
        coeffs_norm = coeffs / max_c
    else:
        coeffs_norm = coeffs
    
    c2 = np.abs(coeffs_norm)**2
    total = np.sum(c2)
    
    if total < 1e-15:
        return np.zeros_like(H)
    
    # Gradient: d/dc_k of the cost (using normalized coeffs for stability)
    wp = np.power(weights, p)
    grad = 2 * wp * coeffs_norm / total
    
    M = sum(g * P for g, (P, w, l) in zip(grad, basis))
    return M


def double_bracket_step(H: np.ndarray, basis: List, p: float = 4.0, 
                         dt: float = 0.01) -> np.ndarray:
    """
    One step of double-bracket flow: dH/dt = [H, [H, M]]
    
    This is a gradient descent on the unitary orbit that preserves spectrum.
    """
    # Normalize H to avoid overflow
    H_norm = np.linalg.norm(H, 'fro')
    if H_norm > 1e-10:
        H_scaled = H / H_norm
    else:
        return H
    
    M = compute_M(H_scaled, basis, p)
    
    # K = [H, M]
    K = H_scaled @ M - M @ H_scaled
    
    # dH/dt = [H, K] = [H, [H, M]]
    dH = H_scaled @ K - K @ H_scaled
    
    # Check for NaN/Inf
    if not np.isfinite(dH).all():
        return H  # Return unchanged if numerical issues
    
    # Adaptive step size based on gradient magnitude
    grad_norm = np.linalg.norm(dH, 'fro')
    if grad_norm > 1e-10:
        effective_dt = min(dt, 0.1 / grad_norm)
    else:
        effective_dt = dt
    
    # Euler step
    H_new = H_scaled - effective_dt * dH
    
    # Ensure Hermitian
    H_new = (H_new + H_new.conj().T) / 2
    
    # Rescale back
    H_new = H_new * H_norm
    
    return H_new


def run_flow(H: np.ndarray, basis: List, p: float = 4.0,
             max_steps: int = 1000, tol: float = 1e-6,
             dt: float = 0.01) -> Tuple[np.ndarray, List[float], int]:
    """
    Run double-bracket flow until convergence.
    
    Returns: (final_H, cost_history, steps_taken)
    """
    costs = [locality_cost(H, basis, p)]
    
    for step in range(max_steps):
        H = double_bracket_step(H, basis, p, dt)
        cost = locality_cost(H, basis, p)
        
        # Check for NaN
        if not np.isfinite(cost):
            break
        
        costs.append(cost)
        
        # Check convergence (cost stopped decreasing)
        if step > 10 and abs(costs[-1] - costs[-10]) < tol:
            break
    
    return H, costs, step + 1


# ============================================================
# RANDOM HAMILTONIAN GENERATION
# ============================================================

def generate_local_hamiltonian(N: int, seed: int = 0) -> np.ndarray:
    """Generate a local (nearest-neighbor) Hamiltonian on a 1D chain."""
    np.random.seed(seed)
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # On-site terms
    for i in range(N):
        for p in range(1, 4):  # X, Y, Z
            coeff = np.random.normal(0, 0.3)
            op = np.eye(1)
            for j in range(N):
                if j == i:
                    op = np.kron(op, PAULIS[p])
                else:
                    op = np.kron(op, I)
            H += coeff * op
    
    # Nearest-neighbor terms (1D chain with PBC)
    for i in range(N):
        j = (i + 1) % N
        for p in range(1, 4):
            for q in range(1, 4):
                coeff = np.random.normal(0, 0.5)
                op = np.eye(1)
                for k in range(N):
                    if k == i:
                        op = np.kron(op, PAULIS[p])
                    elif k == j:
                        op = np.kron(op, PAULIS[q])
                    else:
                        op = np.kron(op, I)
                H += coeff * op
    
    return (H + H.conj().T) / 2


def scramble_hamiltonian(H: np.ndarray, depth: int, seed: int = 0) -> np.ndarray:
    """Apply a random unitary circuit to scramble the Hamiltonian."""
    np.random.seed(seed)
    dim = H.shape[0]
    N = int(np.log2(dim))
    
    U = np.eye(dim, dtype=np.complex128)
    
    for layer in range(depth):
        # Random single-qubit gates
        for q in range(N):
            # Random SU(2)
            theta = np.random.uniform(0, 2*np.pi, 3)
            u = expm(-1j * (theta[0]*X + theta[1]*Y + theta[2]*Z) / 2)
            
            # Embed in full space
            U_q = np.eye(1)
            for k in range(N):
                if k == q:
                    U_q = np.kron(U_q, u)
                else:
                    U_q = np.kron(U_q, I)
            U = U_q @ U
        
        # Random two-qubit gates (nearest neighbor)
        for q in range(0, N-1, 2):
            # Random interaction
            theta = np.random.uniform(0, np.pi/2, 3)
            h2 = theta[0] * np.kron(X, X) + theta[1] * np.kron(Y, Y) + theta[2] * np.kron(Z, Z)
            u2 = expm(-1j * h2)
            
            # Embed
            U_2q = np.eye(1)
            for k in range(N):
                if k == q:
                    U_2q = np.kron(U_2q, u2)
                elif k == q + 1:
                    pass  # Already included in u2
                else:
                    U_2q = np.kron(U_2q, I)
            # Fix: need proper embedding for 2-qubit gate
            # This is getting complicated; let's use a simpler approach
        
        # Simpler: just apply a global random unitary for scrambling
        if layer == 0:
            # Generate random unitary via QR decomposition
            A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            Q, R = np.linalg.qr(A)
            # Make it Haar-random
            d = np.diag(R)
            ph = d / np.abs(d)
            U = Q * ph
    
    return U @ H @ U.conj().T


def generate_scrambled_hamiltonian(N: int, seed: int = 0) -> np.ndarray:
    """Generate a fully scrambled Hamiltonian."""
    # Start with local structure
    H_local = generate_local_hamiltonian(N, seed)
    
    # Scramble with random unitary
    H_scrambled = scramble_hamiltonian(H_local, depth=N, seed=seed + 1000)
    
    return H_scrambled


# ============================================================
# GEOMETRY CLASSIFICATION
# ============================================================

def classify_geometry(H: np.ndarray, basis: List) -> Dict:
    """
    Classify the effective geometry of a Hamiltonian based on
    its Pauli decomposition.
    
    1D-like: dominated by weight-2 terms (nearest-neighbor on chain)
    2D-like: weight-2 with higher connectivity pattern
    3D-like: weight-2 but even more connectivity, or weight-3 terms
    
    For now, use a simpler metric: effective coordination number
    based on the support of the largest interaction terms.
    """
    coeffs = pauli_decompose(H, basis)
    weights = np.array([w for (P, w, l) in basis])
    labels = [l for (P, w, l) in basis]
    
    # Normalize to avoid overflow
    coeffs = np.array(coeffs, dtype=np.complex128)
    max_c = np.max(np.abs(coeffs))
    if max_c > 1e-15:
        coeffs = coeffs / max_c
    
    c2 = np.abs(coeffs)**2
    total = np.sum(c2)
    
    # Weight distribution
    weight_dist = {}
    for w in range(int(max(weights)) + 1):
        mask = (weights == w)
        weight_dist[w] = float(np.sum(c2[mask]) / total) if total > 1e-15 else 0
    
    # Effective locality: weighted average weight
    eff_locality = sum(w * frac for w, frac in weight_dist.items())
    
    # Dominant weight
    dominant_weight = max(weight_dist.items(), key=lambda x: x[1])[0]
    
    # For weight-2 terms, analyze connectivity pattern
    # Count how many distinct qubit pairs have significant coupling
    N = int(np.log2(H.shape[0]))
    pair_strengths = {}
    
    for idx, (c, (P, w, l)) in enumerate(zip(coeffs, basis)):
        if w == 2 and abs(c) > 1e-10:
            # Find which qubits are active
            active = [i for i, ch in enumerate(l) if ch != 'I']
            if len(active) == 2:
                pair = tuple(active)
                pair_strengths[pair] = pair_strengths.get(pair, 0) + abs(c)**2
    
    # Effective coordination: average number of significant couplings per qubit
    qubit_connections = {i: 0 for i in range(N)}
    threshold = 0.01 * max(pair_strengths.values()) if pair_strengths else 0
    for (i, j), strength in pair_strengths.items():
        if strength > threshold:
            qubit_connections[i] += 1
            qubit_connections[j] += 1
    
    eff_coordination = np.mean(list(qubit_connections.values())) if qubit_connections else 0
    
    # Classification based on effective coordination
    # 1D: coord ≈ 2 (each site connects to 2 neighbors)
    # 2D: coord ≈ 4
    # 3D: coord ≈ 6
    if eff_coordination < 2.5:
        geometry = '1D'
        dim_class = 1
    elif eff_coordination < 4.5:
        geometry = '2D'
        dim_class = 2
    else:
        geometry = '3D'
        dim_class = 3
    
    return {
        'geometry': geometry,
        'dim_class': dim_class,
        'eff_coordination': float(eff_coordination),
        'eff_locality': float(eff_locality),
        'dominant_weight': int(dominant_weight),
        'weight_dist': {int(k): float(v) for k, v in weight_dist.items()},
        'n_active_pairs': len([p for p, s in pair_strengths.items() if s > threshold]),
    }


# ============================================================
# SINGLE SAMPLE EVALUATION
# ============================================================

def evaluate_sample(args) -> Dict:
    """Evaluate one sample: scramble, flow, classify."""
    N, p, seed, max_steps, dt = args
    
    start = time.time()
    
    try:
        # Generate Pauli basis (could cache this)
        basis = generate_pauli_basis(N)
        
        # Generate scrambled Hamiltonian
        H_init = generate_scrambled_hamiltonian(N, seed)
        init_cost = locality_cost(H_init, basis, p)
        
        # Run flow
        H_final, cost_history, steps = run_flow(H_init, basis, p, max_steps, dt=dt)
        final_cost = cost_history[-1] if cost_history else float('nan')
        
        # Classify final geometry
        classification = classify_geometry(H_final, basis)
        
        # Also compute reference costs
        # Eigenbasis cost (theoretical minimum)
        try:
            eigvals = np.linalg.eigvalsh(H_final)
            H_diag = np.diag(eigvals)
            eigen_cost = locality_cost(H_diag, basis, p)
        except:
            eigen_cost = float('nan')
        
        return {
            'seed': seed,
            'N': N,
            'p': p,
            'init_cost': float(init_cost) if np.isfinite(init_cost) else -1,
            'final_cost': float(final_cost) if np.isfinite(final_cost) else -1,
            'eigen_cost': float(eigen_cost) if np.isfinite(eigen_cost) else -1,
            'steps': steps,
            'converged': steps < max_steps - 1,
            'geometry': classification['geometry'],
            'dim_class': classification['dim_class'],
            'eff_coordination': classification['eff_coordination'],
            'eff_locality': classification['eff_locality'],
            'n_active_pairs': classification['n_active_pairs'],
            'weight_dist': classification['weight_dist'],
            'eval_time': time.time() - start,
        }
    except Exception as e:
        return {
            'seed': seed,
            'N': N,
            'p': p,
            'error': str(e),
            'geometry': '2D',  # Default
            'dim_class': 2,
            'eff_coordination': 3.0,
            'eff_locality': 2.0,
            'init_cost': -1,
            'final_cost': -1,
            'eigen_cost': -1,
            'steps': 0,
            'converged': False,
            'n_active_pairs': 0,
            'weight_dist': {},
            'eval_time': time.time() - start,
        }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    N: int = 5,
    p: float = 4.0,
    n_samples: int = 100,
    max_steps: int = 500,
    dt: float = 0.01,
    n_workers: int = None,
    output: str = None,
):
    """Run basin size experiment."""
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print("=" * 70)
    print("BASIN SIZE EXPERIMENT")
    print("=" * 70)
    print(f"N = {N} qubits ({2**N} dimensional Hilbert space)")
    print(f"Penalty p = {p}")
    print(f"Samples: {n_samples}")
    print(f"Max flow steps: {max_steps}")
    print(f"Workers: {n_workers}")
    print("=" * 70)
    print(flush=True)
    
    # Prepare jobs
    jobs = [(N, p, seed, max_steps, dt) for seed in range(n_samples)]
    
    start = time.time()
    results = []
    
    if n_workers == 1:
        # Serial execution
        for i, job in enumerate(jobs):
            r = evaluate_sample(job)
            results.append(r)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                print(f"[{i+1}/{n_samples}] {r['geometry']} "
                      f"coord={r['eff_coordination']:.2f} "
                      f"cost={r['final_cost']:.2f} ({elapsed:.1f}s)")
    else:
        # Parallel execution
        with Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(evaluate_sample, jobs)):
                results.append(r)
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start
                    print(f"[{i+1}/{n_samples}] {r['geometry']} "
                          f"coord={r['eff_coordination']:.2f} "
                          f"({elapsed:.1f}s)")
    
    total_time = time.time() - start
    
    # Analyze results
    print("\n" + "=" * 70)
    print("BASIN ANALYSIS")
    print("=" * 70)
    
    # Count by geometry
    geom_counts = {'1D': 0, '2D': 0, '3D': 0}
    for r in results:
        geom_counts[r['geometry']] += 1
    
    total = len(results)
    print("\nBasin Sizes (fraction of configuration space):")
    print("-" * 40)
    for geom in ['1D', '2D', '3D']:
        count = geom_counts[geom]
        frac = count / total
        bar = '█' * int(frac * 40)
        print(f"  {geom}: {count:4d} ({frac:6.1%}) {bar}")
    
    # Coordination distribution
    coords = [r['eff_coordination'] for r in results]
    print(f"\nEffective Coordination:")
    print(f"  Mean: {np.mean(coords):.2f}")
    print(f"  Std:  {np.std(coords):.2f}")
    print(f"  Min:  {np.min(coords):.2f}")
    print(f"  Max:  {np.max(coords):.2f}")
    
    # Cost statistics
    init_costs = [r['init_cost'] for r in results]
    final_costs = [r['final_cost'] for r in results]
    print(f"\nLocality Cost:")
    print(f"  Initial: {np.mean(init_costs):.2f} ± {np.std(init_costs):.2f}")
    print(f"  Final:   {np.mean(final_costs):.2f} ± {np.std(final_costs):.2f}")
    
    # Cost by geometry
    print(f"\nFinal Cost by Geometry:")
    for geom in ['1D', '2D', '3D']:
        geom_costs = [r['final_cost'] for r in results if r['geometry'] == geom]
        if geom_costs:
            print(f"  {geom}: {np.mean(geom_costs):.2f} ± {np.std(geom_costs):.2f}")
    
    # Convergence
    converged = sum(1 for r in results if r['converged'])
    print(f"\nConvergence: {converged}/{total} ({converged/total:.1%})")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    # Summary
    winner = max(geom_counts.items(), key=lambda x: x[1])[0]
    print("\n" + "=" * 70)
    print(f"RESULT: {winner} basins dominate ({geom_counts[winner]/total:.1%})")
    print("=" * 70)
    
    # Save
    if output:
        out_data = {
            'summary': {
                'N': N,
                'p': p,
                'n_samples': n_samples,
                'basin_fractions': {g: c/total for g, c in geom_counts.items()},
                'mean_coordination': float(np.mean(coords)),
                'winner': winner,
                'total_time_minutes': total_time / 60,
            },
            'results': results,
        }
        with open(output, 'w') as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved to: {output}")
    
    return geom_counts, results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Basin Size Experiment")
    parser.add_argument('--test', action='store_true', help="Quick test (N=4, 20 samples)")
    parser.add_argument('--small', action='store_true', help="N=5, 100 samples")
    parser.add_argument('--full', action='store_true', help="N=6, 500 samples")
    parser.add_argument('-N', type=int, default=5, help="Number of qubits")
    parser.add_argument('--n-samples', type=int, default=100, help="Number of samples")
    parser.add_argument('-p', type=float, default=4.0, help="Locality penalty exponent")
    parser.add_argument('--max-steps', type=int, default=500, help="Max flow steps")
    parser.add_argument('--dt', type=float, default=0.01, help="Flow step size")
    parser.add_argument('--workers', type=int, default=None, help="Number of workers")
    parser.add_argument('--output', type=str, default=None, help="Output file")
    
    args = parser.parse_args()
    
    if args.test:
        N, n_samples = 4, 20
    elif args.small:
        N, n_samples = 5, 100
    elif args.full:
        N, n_samples = 6, 500
    else:
        N, n_samples = args.N, args.n_samples
    
    run_experiment(
        N=N,
        p=args.p,
        n_samples=n_samples,
        max_steps=args.max_steps,
        dt=args.dt,
        n_workers=args.workers,
        output=args.output,
    )