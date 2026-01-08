"""
scramble_recover_parallel.py
============================
Accelerated via Multi-Chain Parallelism (Standard NumPy).
Bypasses Numba/JAX compatibility issues on Windows Store Python.

ACCELERATION STRATEGY:
- Uses your 12-core CPU to run multiple independent Strobe optimization 
  chains in parallel for the SAME seed.
- Returns the best result found by any of the cores.
- Math is optimized using raw NumPy broadcasting where possible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.linalg import expm

# =============================================================================
# Configuration
# =============================================================================

def set_thread_env(threads: int = 1) -> None:
    # Limit BLAS threads per worker to avoid oversubscription
    n = str(int(threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n

# =============================================================================
# Core Math (Optimized NumPy)
# =============================================================================

def hermitianize(H: np.ndarray) -> np.ndarray:
    return 0.5 * (H + H.conj().T)

def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def dense_pauli():
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return I, X, Y, Z

def spin_ring_dense(N: int, model: str = "xxx", J: float = 1.0, Delta: float = 1.0) -> np.ndarray:
    model = model.lower()
    I, X, Y, Z = dense_pauli()
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(N):
        j = (i + 1) % N
        for P in (X, Y):
            ops = [I] * N
            ops[i] = P
            ops[j] = P
            H += J * kron_n(ops)
        if abs(Delta) > 0 and model != "xx":
            ops = [I] * N
            ops[i] = Z
            ops[j] = Z
            H += (J * Delta) * kron_n(ops)
    return hermitianize(H)

def build_global_scrambler(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 2**N
    Z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.where(np.abs(d) > 0, np.abs(d), 1.0)
    return Q * ph

def scramble_hamiltonian(H: np.ndarray, U: np.ndarray) -> np.ndarray:
    return hermitianize(U @ H @ U.conj().T)

# =============================================================================
# Optimization Objectives (Inlined for speed)
# =============================================================================

def two_qubit_reduced_operator(H: np.ndarray, N: int, q1: int, q2: int) -> np.ndarray:
    # Optimized reshape/transpose
    if q1 > q2: q1, q2 = q2, q1
    
    # Dimensions
    dim = H.shape[0]
    shape = (2,) * (2 * N)
    H_reshaped = H.reshape(shape)
    
    # We want to trace out everything EXCEPT q1 and q2 (row) and q1 and q2 (col)
    keep_rows = [q1, q2]
    keep_cols = [N + q1, N + q2]
    trace_rows = [i for i in range(N) if i not in keep_rows]
    trace_cols = [N + i for i in trace_rows]
    
    # Move kept indices to front
    perm = keep_rows + keep_cols + trace_rows + trace_cols
    H_p = H_reshaped.transpose(perm)
    
    # Shape is (2,2, 2,2, 2^(N-2), 2^(N-2))
    rest_dim = 1 << (N - 2)
    H_p = H_p.reshape(4, 4, rest_dim, rest_dim)
    
    # Trace over the last two dimensions (rest_dim)
    # This computes sum_k H[a, b, k, k]
    return np.trace(H_p, axis1=2, axis2=3)

def objective_sparse_ratio(H: np.ndarray, N: int) -> float:
    num = 0.0
    sq_sum = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            h_red = two_qubit_reduced_operator(H, N, i, j)
            val = np.linalg.norm(h_red) # Frobenius
            num += val
            sq_sum += val*val
            
    den = math.sqrt(sq_sum + 1e-15)
    return num / den

# =============================================================================
# Strobe Update (Inlined)
# =============================================================================

def apply_two_qubit_conjugation(H: np.ndarray, N: int, q1: int, q2: int, U2: np.ndarray) -> np.ndarray:
    if q1 > q2: q1, q2 = q2, q1
    
    dim = H.shape[0]
    shape = (2,) * (2 * N)
    H_reshaped = H.reshape(shape)
    
    # Permute to isolate q1, q2 indices at start of row and col sets
    others = [i for i in range(N) if i not in (q1, q2)]
    
    # Permute: [q1, q2, others, N+q1, N+q2, N+others]
    perm = [q1, q2] + others + [N+q1, N+q2] + [N+i for i in others]
    H_p = H_reshaped.transpose(perm)
    
    # View as (4, rest, 4, rest)
    rest_dim = 1 << (N - 2)
    H_view = H_p.reshape(4, rest_dim, 4, rest_dim)
    
    # Update logic: U H U^dag
    # We contract U (4x4) with H (4,...) on axis 0
    # Then contract with U^dag on axis 2
    
    Ud = U2.conj().T
    
    # U @ H_view along axis 0
    tmp = np.tensordot(U2, H_view, axes=([1], [0])) # -> (4, rest, 4, rest)
    
    # tmp @ U^dag along axis 2
    # tmp is (row_pair, row_rest, col_pair, col_rest)
    # Ud is (new_col_pair, old_col_pair)
    # We contract tmp axis 2 with Ud axis 1
    out = np.tensordot(tmp, Ud, axes=([2], [1])) # -> (4, rest, rest, 4)
    
    # Result is (row_pair, row_rest, col_rest, col_pair_new)
    # We need (row_pair, row_rest, col_pair_new, col_rest) to match perm
    out = out.transpose(0, 1, 3, 2)
    
    # Restore shape and undo permutation
    out = out.reshape(shape)
    inv_perm = np.argsort(perm)
    out = out.transpose(inv_perm)
    
    return out.reshape(dim, dim)

def random_small_gate(rng, eps):
    X = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2.0)
    A = 0.5 * (X + X.conj().T)
    return expm(1j * eps * A)

# =============================================================================
# Worker Process
# =============================================================================

def strobe_worker(args):
    """
    Runs one independent chain of Strobe optimization.
    """
    (H_start, N, cfg, seed_offset) = args
    
    # Unique seed for this chain
    rng = np.random.default_rng(seed_offset)
    
    H = H_start.copy()
    current_cost = objective_sparse_ratio(H, N)
    
    best_H = H.copy()
    best_cost = current_cost
    
    temp = cfg["temp"]
    decay = cfg["temp_decay"]
    gate_eps = cfg["gate_eps"]
    cycles = cfg["cycles"]
    
    edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for step in range(cycles):
        # Propose
        q1, q2 = edges[rng.integers(0, len(edges))]
        U2 = random_small_gate(rng, gate_eps)
        
        H_cand = apply_two_qubit_conjugation(H, N, q1, q2, U2)
        cand_cost = objective_sparse_ratio(H_cand, N)
        
        dE = cand_cost - current_cost
        
        # Metropolis
        if dE <= 0 or (temp > 1e-12 and rng.random() < math.exp(-dE / temp)):
            H = H_cand
            current_cost = cand_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_H = H.copy()
                
        temp *= decay
        
    return best_H, best_cost

# =============================================================================
# Main Driver
# =============================================================================

def run_multi_chain(payload: Dict[str, Any], num_chains: int) -> Dict[str, Any]:
    N = int(payload["N"])
    seed = int(payload["seed"])
    
    # 1. Setup (Main Thread)
    H_spatial = spin_ring_dense(N, model=payload["model"], Delta=payload["Delta"])
    U = build_global_scrambler(N, seed)
    H_scrambled = scramble_hamiltonian(H_spatial, U)
    
    init_cost = objective_sparse_ratio(H_scrambled, N)
    
    # 2. Parallel Chains
    strobe_cfg = payload["strobe_cfg"]
    
    worker_args = []
    for k in range(num_chains):
        # Each worker gets a different seed offset
        args = (H_scrambled, N, strobe_cfg, seed + 1000 + k)
        worker_args.append(args)
        
    best_overall_H = H_scrambled
    best_overall_cost = init_cost
    
    print(f"  > Launching {num_chains} parallel Strobe chains on 12 cores...", flush=True)
    t0_chains = time.time()
    
    # Use ProcessPoolExecutor to map workers to cores
    with ProcessPoolExecutor(max_workers=num_chains) as ex:
        futures = [ex.submit(strobe_worker, arg) for arg in worker_args]
        
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                h_res, cost_res = fut.result()
                if cost_res < best_overall_cost:
                    best_overall_cost = cost_res
                    best_overall_H = h_res
                if completed % 2 == 0:
                    print(f"    [Chain {completed}/{num_chains}] done. Best cost so far: {best_overall_cost:.5f}", flush=True)
            except Exception as e:
                print(f"Worker failed: {e}")

    dt_chains = time.time() - t0_chains
    
    # 3. Metrics (Final check)
    vals = []
    for i in range(N):
        for j in range(i+1, N):
            h_red = two_qubit_reduced_operator(best_overall_H, N, i, j)
            vals.append(np.linalg.norm(h_red))
    
    vals = np.array(vals)
    total_w = np.sum(vals)
    vals.sort()
    # Sum of top N strongest edges
    top_w = np.sum(vals[::-1][:N]) 
    top_share = top_w / (total_w + 1e-12)
    
    return {
        "seed": seed,
        "init_cost": init_cost,
        "final_cost": best_overall_cost,
        "sparse_reduction": init_cost / (best_overall_cost + 1e-12),
        "locality_recovered": bool(top_share > 0.7),
        "topN_share": top_share,
        "wall_time_chains": dt_chains
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    # Default to 12 jobs for your 12-core CPU
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--cycles", type=int, default=8000)
    args = parser.parse_args()
    
    payload = {
        "N": args.N,
        "seed": args.seed,
        "model": "xxx",
        "Delta": 1.0,
        "strobe_cfg": {
            "cycles": args.cycles, 
            "temp": 0.05, 
            "temp_decay": 0.9995, 
            "gate_eps": 0.05
        }
    }
    
    print(f"Starting Multi-Chain Recovery (N={args.N}, Cores={args.jobs})...")
    t0 = time.time()
    
    res = run_multi_chain(payload, num_chains=args.jobs)
    
    print("\nResults:")
    print(f"Seed: {res['seed']}")
    print(f"Reduction: {res['sparse_reduction']:.2f}x")
    print(f"Recovered? {'YES' if res['locality_recovered'] else 'NO'} (TopN Share: {res['topN_share']:.2f})")
    print(f"Time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    # Windows requires this guard for multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()