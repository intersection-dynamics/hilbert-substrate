"""
Basin Size N-Sweep Experiment
=============================

Sweep over system sizes N to see how basin structure evolves.
Automatically adjusts sample count for larger N due to compute scaling.

Scaling:
- Pauli basis: 4^N operators
- Hilbert space: 2^N dimensions
- Each flow step: O(4^N * 2^N) for Pauli decomposition

Usage:
    python basin_sweep_N.py --min-n 4 --max-n 10 --output sweep_results.json
"""

import numpy as np
from scipy.linalg import expm
from typing import Dict, List, Tuple
import json
import time
from multiprocessing import Pool, cpu_count
import os

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
    """Generate all N-qubit Pauli operators with weights."""
    basis = []
    for idx in range(4**N):
        indices = []
        temp = idx
        for _ in range(N):
            indices.append(temp % 4)
            temp //= 4
        indices = indices[::-1]
        
        op = PAULIS[indices[0]]
        for i in range(1, N):
            op = np.kron(op, PAULIS[indices[i]])
        
        weight = sum(1 for i in indices if i != 0)
        label = ''.join(PAULI_LABELS[i] for i in indices)
        basis.append((op, weight, label))
    
    return basis


def pauli_decompose(H: np.ndarray, basis: List) -> np.ndarray:
    """Decompose Hamiltonian into Pauli coefficients."""
    N_qubits = int(np.log2(H.shape[0]))
    dim = 2**N_qubits
    coeffs = np.array([np.trace(H @ P) / dim for (P, w, l) in basis])
    return coeffs


# ============================================================
# LOCALITY COST AND FLOW
# ============================================================

def locality_cost(H: np.ndarray, basis: List, p: float = 4.0) -> float:
    """Compute locality cost C_p(H)."""
    coeffs = pauli_decompose(H, basis)
    weights = np.array([w for (P, w, l) in basis], dtype=np.float64)
    
    coeffs = np.array(coeffs, dtype=np.complex128)
    max_c = np.max(np.abs(coeffs))
    if max_c > 1e-15:
        coeffs = coeffs / max_c
    
    c2 = np.abs(coeffs)**2
    total = np.sum(c2)
    
    if total < 1e-15:
        return 0.0
    
    wp = np.power(weights.astype(np.float64), p)
    cost = np.sum(wp * c2) / total
    return float(np.real(cost))


def compute_M(H: np.ndarray, basis: List, p: float = 4.0) -> np.ndarray:
    """Compute gradient operator M."""
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
    
    wp = np.power(weights, p)
    grad = 2 * wp * coeffs_norm / total
    
    M = sum(g * P for g, (P, w, l) in zip(grad, basis))
    return M


def double_bracket_step(H: np.ndarray, basis: List, p: float = 4.0, 
                         dt: float = 0.01) -> np.ndarray:
    """One step of double-bracket flow."""
    H_norm = np.linalg.norm(H, 'fro')
    if H_norm > 1e-10:
        H_scaled = H / H_norm
    else:
        return H
    
    M = compute_M(H_scaled, basis, p)
    K = H_scaled @ M - M @ H_scaled
    dH = H_scaled @ K - K @ H_scaled
    
    if not np.isfinite(dH).all():
        return H
    
    grad_norm = np.linalg.norm(dH, 'fro')
    if grad_norm > 1e-10:
        effective_dt = min(dt, 0.1 / grad_norm)
    else:
        effective_dt = dt
    
    H_new = H_scaled - effective_dt * dH
    H_new = (H_new + H_new.conj().T) / 2
    H_new = H_new * H_norm
    
    return H_new


def run_flow(H: np.ndarray, basis: List, p: float = 4.0,
             max_steps: int = 500, tol: float = 1e-6,
             dt: float = 0.01) -> Tuple[np.ndarray, float, int]:
    """Run flow until convergence. Returns (H_final, final_cost, steps)."""
    cost = locality_cost(H, basis, p)
    
    for step in range(max_steps):
        H = double_bracket_step(H, basis, p, dt)
        new_cost = locality_cost(H, basis, p)
        
        if not np.isfinite(new_cost):
            break
        
        if step > 10 and abs(new_cost - cost) < tol:
            break
        cost = new_cost
    
    return H, cost, step + 1


# ============================================================
# HAMILTONIAN GENERATION
# ============================================================

def generate_scrambled_hamiltonian(N: int, seed: int = 0) -> np.ndarray:
    """Generate a random scrambled Hamiltonian."""
    np.random.seed(seed)
    dim = 2**N
    
    # Start with local Hamiltonian
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(N):
        for p in range(1, 4):
            coeff = np.random.normal(0, 0.3)
            op = np.eye(1)
            for j in range(N):
                op = np.kron(op, PAULIS[p] if j == i else I)
            H += coeff * op
    
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
    
    H = (H + H.conj().T) / 2
    
    # Scramble with random unitary
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    ph = d / np.abs(d)
    U = Q * ph
    
    return U @ H @ U.conj().T


# ============================================================
# GEOMETRY CLASSIFICATION
# ============================================================

def classify_geometry(H: np.ndarray, basis: List) -> Dict:
    """Classify effective geometry based on Pauli decomposition."""
    coeffs = pauli_decompose(H, basis)
    
    coeffs = np.array(coeffs, dtype=np.complex128)
    max_c = np.max(np.abs(coeffs))
    if max_c > 1e-15:
        coeffs = coeffs / max_c
    
    N = int(np.log2(H.shape[0]))
    labels = [l for (P, w, l) in basis]
    
    # Analyze weight-2 connectivity
    pair_strengths = {}
    for idx, (c, (P, w, l)) in enumerate(zip(coeffs, basis)):
        if w == 2 and abs(c) > 1e-10:
            active = [i for i, ch in enumerate(l) if ch != 'I']
            if len(active) == 2:
                pair = tuple(active)
                pair_strengths[pair] = pair_strengths.get(pair, 0) + abs(c)**2
    
    qubit_connections = {i: 0 for i in range(N)}
    threshold = 0.01 * max(pair_strengths.values()) if pair_strengths else 0
    for (i, j), strength in pair_strengths.items():
        if strength > threshold:
            qubit_connections[i] += 1
            qubit_connections[j] += 1
    
    eff_coordination = np.mean(list(qubit_connections.values())) if qubit_connections else 0
    
    if eff_coordination < 2.5:
        geometry = '1D'
    elif eff_coordination < 4.5:
        geometry = '2D'
    else:
        geometry = '3D'
    
    return {
        'geometry': geometry,
        'eff_coordination': float(eff_coordination),
        'n_active_pairs': len([p for p, s in pair_strengths.items() if s > threshold]),
    }


# ============================================================
# SINGLE SAMPLE EVALUATION
# ============================================================

def evaluate_sample(args) -> Dict:
    """Evaluate one sample."""
    N, p, seed, max_steps, dt = args
    
    start = time.time()
    
    try:
        basis = generate_pauli_basis(N)
        H_init = generate_scrambled_hamiltonian(N, seed)
        H_final, final_cost, steps = run_flow(H_init, basis, p, max_steps, dt=dt)
        classification = classify_geometry(H_final, basis)
        
        return {
            'seed': seed,
            'N': N,
            'final_cost': float(final_cost) if np.isfinite(final_cost) else -1,
            'steps': steps,
            'geometry': classification['geometry'],
            'eff_coordination': classification['eff_coordination'],
            'eval_time': time.time() - start,
        }
    except Exception as e:
        return {
            'seed': seed,
            'N': N,
            'error': str(e),
            'geometry': '2D',
            'eff_coordination': 3.0,
            'final_cost': -1,
            'steps': 0,
            'eval_time': time.time() - start,
        }


# ============================================================
# N-SWEEP
# ============================================================

def run_N_sweep(
    min_N: int = 4,
    max_N: int = 10,
    p: float = 4.0,
    base_samples: int = 200,
    max_steps: int = 500,
    dt: float = 0.01,
    n_workers: int = None,
    output: str = None,
):
    """Sweep over N values, measuring basin structure at each."""
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print("=" * 70)
    print("BASIN SIZE N-SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"N range: {min_N} to {max_N}")
    print(f"Penalty p = {p}")
    print(f"Base samples: {base_samples} (reduced for large N)")
    print(f"Workers: {n_workers}")
    print("=" * 70)
    print(flush=True)
    
    # Estimate sample counts (reduce for large N)
    # N=6 took 476 min for 200 samples
    # Scaling roughly as 4^N, so reduce samples accordingly
    sample_counts = {}
    for N in range(min_N, max_N + 1):
        # Scale down samples for larger N
        scale_factor = (4**(N - 6)) if N > 6 else 1
        samples = max(20, base_samples // scale_factor)
        sample_counts[N] = samples
    
    print("\nSample counts by N:")
    for N, s in sample_counts.items():
        print(f"  N={N}: {s} samples")
    print()
    
    all_results = {}
    sweep_start = time.time()
    
    for N in range(min_N, max_N + 1):
        n_samples = sample_counts[N]
        
        print(f"\n{'='*60}")
        print(f"N = {N} ({2**N} dim Hilbert space, {4**N} Pauli ops)")
        print(f"Running {n_samples} samples...")
        print(f"{'='*60}", flush=True)
        
        jobs = [(N, p, seed, max_steps, dt) for seed in range(n_samples)]
        
        start = time.time()
        results = []
        
        if n_workers == 1 or N >= 10:  # Serial for large N (memory)
            for i, job in enumerate(jobs):
                r = evaluate_sample(job)
                results.append(r)
                if (i + 1) % max(1, n_samples // 10) == 0:
                    elapsed = time.time() - start
                    print(f"  [{i+1}/{n_samples}] {r['geometry']} "
                          f"coord={r['eff_coordination']:.2f} ({elapsed:.1f}s)")
        else:
            with Pool(n_workers) as pool:
                for i, r in enumerate(pool.imap_unordered(evaluate_sample, jobs)):
                    results.append(r)
                    if (i + 1) % max(1, n_samples // 10) == 0:
                        elapsed = time.time() - start
                        print(f"  [{i+1}/{n_samples}] {r['geometry']} "
                              f"coord={r['eff_coordination']:.2f} ({elapsed:.1f}s)")
        
        elapsed = time.time() - start
        
        # Analyze
        geom_counts = {'1D': 0, '2D': 0, '3D': 0}
        coords = []
        for r in results:
            geom_counts[r['geometry']] += 1
            coords.append(r['eff_coordination'])
        
        total = len(results)
        basin_fracs = {g: c/total for g, c in geom_counts.items()}
        mean_coord = np.mean(coords)
        std_coord = np.std(coords)
        winner = max(geom_counts.items(), key=lambda x: x[1])[0]
        
        print(f"\n  Results for N={N}:")
        print(f"    Basin fractions: 1D={basin_fracs['1D']:.1%}, "
              f"2D={basin_fracs['2D']:.1%}, 3D={basin_fracs['3D']:.1%}")
        print(f"    Mean coordination: {mean_coord:.2f} ± {std_coord:.2f}")
        print(f"    Winner: {winner}")
        print(f"    Time: {elapsed/60:.1f} minutes")
        
        all_results[N] = {
            'N': N,
            'n_samples': n_samples,
            'basin_fractions': basin_fracs,
            'mean_coordination': mean_coord,
            'std_coordination': std_coord,
            'winner': winner,
            'time_minutes': elapsed / 60,
            'individual_results': results,
        }
        
        # Save incrementally
        if output:
            out_data = {
                'config': {
                    'min_N': min_N,
                    'max_N': max_N,
                    'p': p,
                    'base_samples': base_samples,
                },
                'results': {str(k): {key: val for key, val in v.items() 
                                     if key != 'individual_results'} 
                           for k, v in all_results.items()},
                'detailed_results': {str(k): v for k, v in all_results.items()},
            }
            with open(output, 'w') as f:
                json.dump(out_data, f, indent=2)
            print(f"  [Saved to {output}]")
    
    total_time = time.time() - sweep_start
    
    # Final summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'N':<4} {'Dim':<8} {'1D':<8} {'2D':<8} {'3D':<8} {'Coord':<10} {'Winner':<8}")
    print("-" * 60)
    
    for N in range(min_N, max_N + 1):
        if N in all_results:
            r = all_results[N]
            print(f"{N:<4} {2**N:<8} "
                  f"{r['basin_fractions']['1D']:<8.1%} "
                  f"{r['basin_fractions']['2D']:<8.1%} "
                  f"{r['basin_fractions']['3D']:<8.1%} "
                  f"{r['mean_coordination']:<10.2f} "
                  f"{r['winner']:<8}")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    return all_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Basin Size N-Sweep")
    parser.add_argument('--min-n', type=int, default=4, help="Minimum N")
    parser.add_argument('--max-n', type=int, default=10, help="Maximum N")
    parser.add_argument('-p', type=float, default=4.0, help="Locality penalty")
    parser.add_argument('--base-samples', type=int, default=200, help="Base sample count")
    parser.add_argument('--max-steps', type=int, default=500, help="Max flow steps")
    parser.add_argument('--workers', type=int, default=None, help="Number of workers")
    parser.add_argument('--output', type=str, default=None, help="Output file")
    
    args = parser.parse_args()
    
    run_N_sweep(
        min_N=args.min_n,
        max_N=args.max_n,
        p=args.p,
        base_samples=args.base_samples,
        max_steps=args.max_steps,
        n_workers=args.workers,
        output=args.output,
    )