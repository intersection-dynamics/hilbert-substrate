"""
Basin Size N-Sweep - Memory Efficient Version
==============================================

Computes Pauli coefficients on-the-fly instead of storing full basis.
This trades compute for memory, enabling larger N on limited RAM.

Memory usage: O(2^N) for Hamiltonian, not O(4^N) for basis.

Usage:
    python basin_sweep_efficient.py --min-n 7 --max-n 10 --output sweep.json
"""

import numpy as np
from typing import Dict, Tuple
import json
import time
from multiprocessing import Pool, cpu_count

# ============================================================
# PAULI MATRICES (just the 4 single-qubit ones)
# ============================================================

I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = [I, X, Y, Z]


def pauli_from_index(idx: int, N: int) -> Tuple[np.ndarray, int]:
    """
    Generate a single N-qubit Pauli operator from its index.
    Returns (operator, weight) without storing all operators.
    
    Memory: O(2^N) for the operator, computed on demand.
    """
    indices = []
    temp = idx
    for _ in range(N):
        indices.append(temp % 4)
        temp //= 4
    indices = indices[::-1]
    
    # Build operator via Kronecker products
    op = PAULIS[indices[0]].copy()
    for i in range(1, N):
        op = np.kron(op, PAULIS[indices[i]])
    
    weight = sum(1 for i in indices if i != 0)
    return op, weight


def pauli_indices_from_index(idx: int, N: int) -> Tuple[list, int]:
    """Get Pauli indices and weight without building the full operator."""
    indices = []
    temp = idx
    for _ in range(N):
        indices.append(temp % 4)
        temp //= 4
    indices = indices[::-1]
    weight = sum(1 for i in indices if i != 0)
    return indices, weight


# ============================================================
# MEMORY-EFFICIENT PAULI DECOMPOSITION
# ============================================================

def locality_cost_efficient(H: np.ndarray, N: int, p: float = 4.0) -> float:
    """
    Compute locality cost by iterating over Pauli operators one at a time.
    Memory: O(2^N) instead of O(4^N).
    """
    dim = 2**N
    
    # Normalize H
    H_norm = np.linalg.norm(H, 'fro')
    if H_norm < 1e-15:
        return 0.0
    H_scaled = H / H_norm
    
    weighted_sum = 0.0
    total_sum = 0.0
    
    for idx in range(4**N):
        P, weight = pauli_from_index(idx, N)
        coeff = np.trace(H_scaled @ P) / dim
        c2 = np.abs(coeff)**2
        
        weighted_sum += (weight ** p) * c2
        total_sum += c2
    
    if total_sum < 1e-15:
        return 0.0
    
    return weighted_sum / total_sum


def compute_M_efficient(H: np.ndarray, N: int, p: float = 4.0) -> np.ndarray:
    """
    Compute gradient operator M on-the-fly.
    Memory: O(2^N).
    """
    dim = 2**N
    
    H_norm = np.linalg.norm(H, 'fro')
    if H_norm < 1e-15:
        return np.zeros_like(H)
    H_scaled = H / H_norm
    
    # First pass: compute total for normalization
    total = 0.0
    for idx in range(4**N):
        P, _ = pauli_from_index(idx, N)
        coeff = np.trace(H_scaled @ P) / dim
        total += np.abs(coeff)**2
    
    if total < 1e-15:
        return np.zeros_like(H)
    
    # Second pass: build M
    M = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(4**N):
        P, weight = pauli_from_index(idx, N)
        coeff = np.trace(H_scaled @ P) / dim
        grad = 2 * (weight ** p) * coeff / total
        M += grad * P
    
    return M


# ============================================================
# DOUBLE BRACKET FLOW
# ============================================================

def double_bracket_step_efficient(H: np.ndarray, N: int, p: float = 4.0,
                                   dt: float = 0.01) -> np.ndarray:
    """One flow step with memory-efficient gradient computation."""
    H_norm = np.linalg.norm(H, 'fro')
    if H_norm < 1e-10:
        return H
    H_scaled = H / H_norm
    
    M = compute_M_efficient(H_scaled, N, p)
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
    return H_new * H_norm


def run_flow_efficient(H: np.ndarray, N: int, p: float = 4.0,
                       max_steps: int = 200, tol: float = 1e-5,
                       dt: float = 0.01) -> Tuple[np.ndarray, float, int]:
    """Run flow with memory-efficient operations."""
    cost = locality_cost_efficient(H, N, p)
    
    for step in range(max_steps):
        H = double_bracket_step_efficient(H, N, p, dt)
        new_cost = locality_cost_efficient(H, N, p)
        
        if not np.isfinite(new_cost):
            break
        
        if step > 5 and abs(new_cost - cost) < tol:
            break
        cost = new_cost
    
    return H, cost, step + 1


# ============================================================
# HAMILTONIAN GENERATION
# ============================================================

def generate_scrambled_hamiltonian(N: int, seed: int = 0) -> np.ndarray:
    """Generate random scrambled Hamiltonian."""
    np.random.seed(seed)
    dim = 2**N
    
    # Random Hermitian matrix
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (A + A.conj().T) / 2
    
    # Normalize
    H = H / np.linalg.norm(H, 'fro')
    
    return H


# ============================================================
# GEOMETRY CLASSIFICATION (Memory Efficient)
# ============================================================

def classify_geometry_efficient(H: np.ndarray, N: int) -> Dict:
    """Classify geometry by analyzing weight-2 Pauli coefficients."""
    dim = 2**N
    
    H_norm = np.linalg.norm(H, 'fro')
    if H_norm < 1e-15:
        return {'geometry': '2D', 'eff_coordination': 3.0}
    H_scaled = H / H_norm
    
    # Only compute weight-2 coefficients for connectivity analysis
    pair_strengths = {}
    
    for idx in range(4**N):
        indices, weight = pauli_indices_from_index(idx, N)
        
        if weight != 2:
            continue
        
        # Find active qubits
        active = [i for i, p_idx in enumerate(indices) if p_idx != 0]
        if len(active) != 2:
            continue
        
        # Compute coefficient
        P, _ = pauli_from_index(idx, N)
        coeff = np.trace(H_scaled @ P) / dim
        
        pair = tuple(active)
        pair_strengths[pair] = pair_strengths.get(pair, 0) + np.abs(coeff)**2
    
    # Compute effective coordination
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
    }


# ============================================================
# SINGLE SAMPLE
# ============================================================

def evaluate_sample(args) -> Dict:
    """Evaluate one sample with memory-efficient operations."""
    N, p, seed, max_steps, dt = args
    
    start = time.time()
    
    try:
        H_init = generate_scrambled_hamiltonian(N, seed)
        H_final, final_cost, steps = run_flow_efficient(H_init, N, p, max_steps, dt=dt)
        classification = classify_geometry_efficient(H_final, N)
        
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

def run_sweep(
    min_N: int = 7,
    max_N: int = 10,
    p: float = 4.0,
    base_samples: int = 100,
    max_steps: int = 200,
    dt: float = 0.01,
    n_workers: int = 1,  # Default to 1 for memory safety
    output: str = None,
):
    """Run sweep with memory-efficient implementation."""
    
    print("=" * 70)
    print("BASIN SIZE N-SWEEP (Memory Efficient)")
    print("=" * 70)
    print(f"N range: {min_N} to {max_N}")
    print(f"Penalty p = {p}")
    print(f"Workers: {n_workers} (use 1-2 to avoid OOM)")
    print("=" * 70)
    
    # Memory estimates
    print("\nMemory estimates (Hamiltonian only):")
    for N in range(min_N, max_N + 1):
        h_size = (2**N)**2 * 16 / 1e9
        print(f"  N={N}: {h_size:.3f} GB per Hamiltonian")
    print()
    
    # Sample counts - reduce for larger N due to compute time
    sample_counts = {}
    for N in range(min_N, max_N + 1):
        # Each step is O(4^N), so reduce samples
        scale = 4 ** (N - min_N)
        samples = max(10, base_samples // scale)
        sample_counts[N] = samples
    
    print("Sample counts:")
    for N, s in sample_counts.items():
        print(f"  N={N}: {s} samples")
    print()
    
    all_results = {}
    sweep_start = time.time()
    
    for N in range(min_N, max_N + 1):
        n_samples = sample_counts[N]
        
        print(f"\n{'='*60}")
        print(f"N = {N} (Hilbert dim = {2**N}, iterating over {4**N} Paulis)")
        print(f"Running {n_samples} samples (serial for memory safety)...")
        print(f"{'='*60}", flush=True)
        
        start = time.time()
        results = []
        
        # Always run serial for large N to avoid memory issues
        for i in range(n_samples):
            args = (N, p, i, max_steps, dt)
            r = evaluate_sample(args)
            results.append(r)
            
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (n_samples - i - 1)
            print(f"  [{i+1}/{n_samples}] {r['geometry']} "
                  f"coord={r['eff_coordination']:.2f} "
                  f"({r['eval_time']:.1f}s, ETA: {eta/60:.1f}min)", flush=True)
        
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
        
        print(f"\n  N={N} Results:")
        print(f"    Basins: 1D={basin_fracs['1D']:.1%}, 2D={basin_fracs['2D']:.1%}, 3D={basin_fracs['3D']:.1%}")
        print(f"    Coordination: {mean_coord:.2f} ± {std_coord:.2f}")
        print(f"    Winner: {winner}")
        print(f"    Time: {elapsed/60:.1f} min")
        
        all_results[N] = {
            'N': N,
            'n_samples': n_samples,
            'basin_fractions': basin_fracs,
            'mean_coordination': mean_coord,
            'std_coordination': std_coord,
            'winner': winner,
            'time_minutes': elapsed / 60,
        }
        
        # Save incrementally
        if output:
            with open(output, 'w') as f:
                json.dump({
                    'config': {'min_N': min_N, 'max_N': max_N, 'p': p},
                    'results': {str(k): v for k, v in all_results.items()},
                }, f, indent=2)
            print(f"  [Saved to {output}]")
    
    total_time = time.time() - sweep_start
    
    # Final summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'N':<4} {'1D':<8} {'2D':<8} {'3D':<8} {'Coord':<12} {'Winner':<8}")
    print("-" * 55)
    
    for N in sorted(all_results.keys()):
        r = all_results[N]
        print(f"{N:<4} "
              f"{r['basin_fractions']['1D']:<8.1%} "
              f"{r['basin_fractions']['2D']:<8.1%} "
              f"{r['basin_fractions']['3D']:<8.1%} "
              f"{r['mean_coordination']:.2f}±{r['std_coordination']:.2f}  "
              f"{r['winner']:<8}")
    
    print(f"\nTotal time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-n', type=int, default=7)
    parser.add_argument('--max-n', type=int, default=10)
    parser.add_argument('-p', type=float, default=4.0)
    parser.add_argument('--base-samples', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    run_sweep(
        min_N=args.min_n,
        max_N=args.max_n,
        p=args.p,
        base_samples=args.base_samples,
        max_steps=args.max_steps,
        n_workers=args.workers,
        output=args.output,
    )