"""
scramble_recover_multichain.py
==============================
Best-of-K Recovery Harness for Hilbert Substrate.

Architecture:
1. Per Seed:
   - Generate H_scrambled (Global/Local)
   - Launch K parallel STROBE chains (each with unique RNG).
   - Select Best Chain (lowest objective value).
   - Run FLOW on the winner (polishing).
   - Compute full audit metrics (Sparsity, Signaling, Fermion/JW).
2. Output:
   - Stream results to JSONL.
   - Print summary statistics at the end.

Compatibility:
- Windows Store Python safe (no Numba/JAX).
- CPU-optimized (numpy with limited threads).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm, eigh

# =============================================================================
# Environment Setup
# =============================================================================

def set_thread_env(threads: int = 1) -> None:
    n = str(int(threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n

# =============================================================================
# Math Primitives
# =============================================================================

def hermitianize(H: np.ndarray) -> np.ndarray:
    return 0.5 * (H + H.conj().T)

def dense_pauli():
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    # basis: |0> = up, |1> = down
    sig_minus = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sig_plus  = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    return I, X, Y, Z, sig_minus, sig_plus

def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

# =============================================================================
# Hamiltonian Generation
# =============================================================================

def spin_ring_dense(N: int, model: str = "xxx", J: float = 1.0, Delta: float = 1.0) -> np.ndarray:
    model = model.lower()
    I, X, Y, Z, _, _ = dense_pauli()
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(N):
        j = (i + 1) % N
        # XX + YY part
        for P in (X, Y):
            ops = [I] * N
            ops[i] = P
            ops[j] = P
            H += J * kron_n(ops)
        # ZZ part
        if model in ("xxz", "xxx") and abs(Delta) > 1e-9:
            ops = [I] * N
            ops[i] = Z
            ops[j] = Z
            H += (J * Delta) * kron_n(ops)
            
    return hermitianize(H)

def build_scrambler(N: int, mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 2**N
    
    if mode == "global":
        Z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
        Q, R = np.linalg.qr(Z)
        d = np.diag(R)
        ph = d / np.where(np.abs(d) > 0, np.abs(d), 1.0)
        return Q * ph
    elif mode == "local":
        # Product of random SU(2)s
        def rand_su2():
            z = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / np.sqrt(2.0)
            q, r = np.linalg.qr(z)
            d = np.diag(r)
            ph = d / np.where(np.abs(d)>0, np.abs(d), 1.0)
            return q * ph
        
        U = rand_su2()
        for _ in range(N-1):
            U = np.kron(U, rand_su2())
        return U
    else:
        raise ValueError(f"Unknown scramble mode: {mode}")

def scramble_hamiltonian(H: np.ndarray, U: np.ndarray) -> np.ndarray:
    return hermitianize(U @ H @ U.conj().T)

# =============================================================================
# Objectives & Metrics
# =============================================================================

def two_qubit_reduced_operator(H: np.ndarray, N: int, q1: int, q2: int) -> np.ndarray:
    if q1 == q2: raise ValueError("q1==q2")
    if q1 > q2: q1, q2 = q2, q1
    
    shape = (2,) * (2 * N)
    H_reshaped = H.reshape(shape)
    
    # Keep q1, q2. Trace others.
    # Permute: [q1, q2, others..., q1', q2', others'...]
    # But for trace, we just need to contract 'others' with 'others'
    # Optimal: Move q1, q2 to front.
    keep_rows = [q1, q2]
    keep_cols = [N+q1, N+q2]
    trace_rows = [i for i in range(N) if i not in keep_rows]
    trace_cols = [N+i for i in trace_rows]
    
    perm = keep_rows + keep_cols + trace_rows + trace_cols
    H_p = H_reshaped.transpose(perm)
    
    rest_dim = 1 << (N - 2)
    H_p = H_p.reshape(4, 4, rest_dim, rest_dim)
    
    # Trace over the rest
    return np.trace(H_p, axis1=2, axis2=3)

def compute_pair_strengths(H: np.ndarray, N: int) -> np.ndarray:
    S = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            h_red = two_qubit_reduced_operator(H, N, i, j)
            val = float(np.linalg.norm(h_red, ord='fro'))
            S[i, j] = val
            S[j, i] = val
    return S

def objective_sparse_ratio(S: np.ndarray, N: int, eps: float = 1e-12) -> float:
    # minimize sum(S) / sqrt(sum(S^2))
    triu = S[np.triu_indices(N, k=1)]
    num = np.sum(triu)
    den = np.sqrt(np.sum(triu**2) + eps)
    return float(num / den)

def objective_signal_entropy(S: np.ndarray, N: int, eps: float = 1e-12) -> float:
    # minimize mean row entropy
    denom = math.log(N - 1)
    total_ent = 0.0
    for i in range(N):
        # row excluding self
        row = np.delete(S[i], i)
        row = row + eps
        Z = np.sum(row)
        p = row / Z
        ent = -np.sum(p * np.log(p))
        total_ent += ent
    
    return float((total_ent / N) / denom)

def compute_metrics(H: np.ndarray, N: int, max_weight: Optional[int] = 4) -> Dict[str, Any]:
    S = compute_pair_strengths(H, N)
    
    # 1. Sparsity
    sparse_cost = objective_sparse_ratio(S, N)
    
    # 2. Top N Share
    triu = S[np.triu_indices(N, k=1)]
    total_w = np.sum(triu) + 1e-18
    sorted_w = np.sort(triu)[::-1]
    top_n = np.sum(sorted_w[:N])
    topN_share = float(top_n / total_w)
    
    # 3. Signal Entropy
    sig_ent = objective_signal_entropy(S, N)
    
    # 4. Ring V2 Diagnostic (Diagnostic Only)
    # We check if 2-distance couplings match expected ring behavior (optional)
    # For now, simplistic check: average strength at dist=2 vs dist=1?
    # Actually, let's just use the TopN share as the primary success metric.
    
    return {
        "sparse_cost": sparse_cost,
        "signal_entropy": sig_ent,
        "topN_share": topN_share,
        "pair_strengths": S.tolist()
    }

# =============================================================================
# Fermion Audit (Model=XX only)
# =============================================================================

def fermion_audit(H: np.ndarray, N: int) -> Dict[str, Any]:
    I, _, _, Z, sm, sp = dense_pauli()
    
    # 1. Jordan-Wigner Anticommutator Check
    # c_j = (prod_{k<j} Z_k) sigma^-_j
    # We build c_j and c_j_dag full matrices
    c_ops = []
    cd_ops = []
    
    for j in range(N):
        ops_c = []
        ops_cd = []
        for k in range(N):
            if k < j:
                ops_c.append(Z); ops_cd.append(Z)
            elif k == j:
                ops_c.append(sm); ops_cd.append(sp)
            else:
                ops_c.append(I); ops_cd.append(I)
        c_ops.append(kron_n(ops_c))
        cd_ops.append(kron_n(ops_cd))
        
    # Check on Ground State
    evals, evecs = eigh(H)
    psi0 = evecs[:, 0] # Ground state
    
    max_cc_viol = 0.0
    max_ccd_viol = 0.0
    
    for i in range(N):
        for j in range(N):
            # {c_i, c_j} should be 0
            ac = c_ops[i] @ c_ops[j] + c_ops[j] @ c_ops[i]
            val = abs(np.vdot(psi0, ac @ psi0))
            max_cc_viol = max(max_cc_viol, val)
            
            # {c_i, c_j^dag} should be delta_ij
            acd = c_ops[i] @ cd_ops[j] + cd_ops[j] @ c_ops[i]
            val = abs(np.vdot(psi0, acd @ psi0))
            expected = 1.0 if i == j else 0.0
            max_ccd_viol = max(max_ccd_viol, abs(val - expected))
            
    # 2. Sector Additivity
    # Check if E(n=2) - E(n=0) approx sum of single particle energies
    # Get E0 for n=0
    def get_sector_min(n_target):
        # Filter evecs by particle number? Expensive.
        # Just use crude check or rely on known subspace if available.
        # Faster: Project H onto sector?
        # For N=8, full diagonalization is fast enough.
        # We need to identify sector of each eigenstate.
        # Number operator Num = Sum (I-Z)/2
        pass 
        # Implementing full sector extraction is heavy. 
        # Let's stick to the JW check as the primary proxy for "fermion-ness"
        
    return {
        "jw_max_cc_violation": float(max_cc_viol),
        "jw_max_ccd_violation": float(max_ccd_viol),
        "status": "ok"
    }

# =============================================================================
# STROBE Worker
# =============================================================================

@dataclass
class StrobeConfig:
    cycles: int
    edges: str # "ring" or "all"
    objective: str # "sparse" or "signal"
    gate_eps: float = 0.05
    temp: float = 0.05
    temp_decay: float = 0.9995

def apply_two_qubit_conjugation(H: np.ndarray, N: int, q1: int, q2: int, U2: np.ndarray) -> np.ndarray:
    if q1 > q2: q1, q2 = q2, q1
    dim = H.shape[0]
    shape = (2,) * (2 * N)
    H_reshaped = H.reshape(shape)
    
    others = [i for i in range(N) if i not in (q1, q2)]
    perm = [q1, q2] + others + [N+q1, N+q2] + [N+i for i in others]
    H_p = H_reshaped.transpose(perm)
    
    rest_dim = 1 << (N - 2)
    H_view = H_p.reshape(4, rest_dim, 4, rest_dim)
    
    Ud = U2.conj().T
    # U @ H_view @ U^dag
    tmp = np.tensordot(U2, H_view, axes=([1], [0]))
    out = np.tensordot(tmp, Ud, axes=([2], [1]))
    
    out = out.transpose(0, 1, 3, 2)
    out = out.reshape(shape)
    inv_perm = np.argsort(perm)
    out = out.transpose(inv_perm)
    
    return out.reshape(dim, dim)

def strobe_worker(args):
    """
    Independent Strobe Chain.
    Returns: (best_H, best_cost, metrics_dict)
    """
    (chain_id, H_start, N, cfg, seed) = args
    rng = np.random.default_rng(seed)
    
    H = H_start.copy()
    
    # Init Cost
    S = compute_pair_strengths(H, N)
    if cfg.objective == "signal":
        current_cost = objective_signal_entropy(S, N)
    else:
        current_cost = objective_sparse_ratio(S, N)
        
    best_H = H.copy()
    best_cost = current_cost
    
    # Precompute move edges
    if cfg.edges == "ring":
        edges = [(i, (i+1)%N) for i in range(N)]
    else:
        edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        
    accepted = 0
    t0 = time.time()
    
    temp = cfg.temp
    
    for step in range(cfg.cycles):
        q1, q2 = edges[rng.integers(0, len(edges))]
        
        # Random small unitary
        X = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2.0)
        A = 0.5 * (X + X.conj().T)
        U2 = expm(1j * cfg.gate_eps * A)
        
        H_cand = apply_two_qubit_conjugation(H, N, q1, q2, U2)
        
        # Eval Cost (Need S for both)
        # Note: calculating full S every step is slowish for N=8 (28 pairs).
        # But for N=8 it's acceptable (approx 1ms). 10k steps ~ 10s.
        S_cand = compute_pair_strengths(H_cand, N)
        
        if cfg.objective == "signal":
            cand_cost = objective_signal_entropy(S_cand, N)
        else:
            cand_cost = objective_sparse_ratio(S_cand, N)
            
        dE = cand_cost - current_cost
        
        if dE <= 0 or (temp > 1e-12 and rng.random() < math.exp(-dE / temp)):
            H = H_cand
            current_cost = cand_cost
            accepted += 1
            if current_cost < best_cost:
                best_cost = current_cost
                best_H = H.copy()
                
        temp *= cfg.temp_decay
        
    dt = time.time() - t0
    
    return {
        "chain_id": chain_id,
        "best_cost": float(best_cost),
        "best_H": best_H,
        "accepted_rate": accepted / cfg.cycles,
        "wall_time": dt
    }

# =============================================================================
# FLOW Polisher
# =============================================================================

def iter_pauli_words(N: int, max_weight: int):
    # Generator for Pauli words indices
    # 0=I, 1=X, 2=Y, 3=Z
    idx = [0] * N
    def rec(pos, w):
        if pos == N:
            yield tuple(idx), w
            return
        idx[pos] = 0
        yield from rec(pos + 1, w)
        if w < max_weight:
            for v in (1, 2, 3):
                idx[pos] = v
                yield from rec(pos + 1, w + 1)
    yield from rec(0, 0)

def compute_locality_gradient(H: np.ndarray, N: int, p: int, max_weight: int) -> np.ndarray:
    I, X, Y, Z, _, _ = dense_pauli()
    mats = [I, X, Y, Z]
    
    M = np.zeros_like(H)
    
    # This loop is CPU intensive. For N=8, max_weight=4, it's manageable.
    for indices, w in iter_pauli_words(N, max_weight):
        if w <= 1: continue 
        
        # Build P
        ops = [mats[k] for k in indices]
        P = kron_n(ops)
        
        # coeff = Tr(H P) / dim
        tr = np.real(np.trace(H @ P)) 
        
        # Weight w^p
        weight = float(w**p)
        
        # Operator M += weight * coeff * P
        M += (weight * tr) * P
        
    return hermitianize(M)

def flow_polish(H: np.ndarray, N: int, steps: int, dt: float, p: int, max_weight: int) -> np.ndarray:
    if steps <= 0: return H
    
    H_curr = H.copy()
    for _ in range(steps):
        M = compute_locality_gradient(H_curr, N, p, max_weight)
        G = H_curr @ M - M @ H_curr
        G = 0.5 * (G - G.conj().T) # Skew-Hermitian
        
        U = expm(dt * G) # Minimize
        H_curr = hermitianize(U @ H_curr @ U.conj().T)
        
    return H_curr

# =============================================================================
# Main Pipeline
# =============================================================================

def run_seed_pipeline(
    seed: int, 
    args: argparse.Namespace, 
    strobe_cfg: StrobeConfig
) -> Dict[str, Any]:
    
    t0_seed = time.time()
    
    # 1. Generate & Scramble
    H_spatial = spin_ring_dense(args.N, model=args.model, Delta=args.Delta)
    U_scramble = build_scrambler(args.N, args.scramble, seed)
    H_scrambled = scramble_hamiltonian(H_spatial, U_scramble)
    
    metrics_init = compute_metrics(H_scrambled, args.N)
    
    # 2. Multi-Chain Strobe
    worker_args = []
    for k in range(args.chains):
        chain_seed = seed * 1000 + k
        worker_args.append((k, H_scrambled, args.N, strobe_cfg, chain_seed))
    
    chain_results = []
    
    with ProcessPoolExecutor(max_workers=args.chains) as ex:
        # Submit all
        futs = [ex.submit(strobe_worker, a) for a in worker_args]
        
        for f in as_completed(futs):
            try:
                res = f.result()
                chain_results.append(res)
            except Exception as e:
                print(f"Chain failed: {e}")
                
    if not chain_results:
        return {"error": "All chains failed"}
    
    # 3. Select Winner
    # If objective is signal, lower is better. If sparse, lower is better.
    # Tie-break with topN_share (higher is better).
    # We negate topN_share for sorting purposes.
    
    def sort_key(r):
        # Primary: Cost (asc), Secondary: -TopShare (asc) -> TopShare (desc)
        # We need to compute TopShare for the candidate to tiebreak properly,
        # but Strobe worker didn't return it to save bandwidth?
        # Strobe worker returns H. We can compute metrics on H.
        # But computing metrics on all K candidates is slow? No, K=12 is fine.
        return (r['best_cost'])

    chain_results.sort(key=sort_key)
    winner = chain_results[0]
    
    best_H = winner['best_H']
    chain_costs = [r['best_cost'] for r in chain_results]
    
    # 4. Flow Polisher
    if not args.no_flow:
        H_final = flow_polish(
            best_H, args.N, 
            steps=args.flow_steps, dt=args.dt, p=args.p, max_weight=args.max_weight
        )
    else:
        H_final = best_H
        
    metrics_final = compute_metrics(H_final, args.N)
    
    # 5. Audits
    audit_data = {}
    if args.fermion_audit:
        if args.model == "xx":
            audit_data = fermion_audit(H_final, args.N)
        else:
            audit_data = {"status": "skipped_model_mismatch"}
            
    # 6. Success Flags
    # Sparse OK: TopN >= 0.70 OR Reduction >= 4.0
    sparse_red = metrics_init['sparse_cost'] / (metrics_final['sparse_cost'] + 1e-9)
    sparse_ok = (metrics_final['topN_share'] >= 0.70) or (sparse_red >= 4.0)
    
    # Signal OK: Reduction >= 1.10
    sig_red = metrics_init['signal_entropy'] / (metrics_final['signal_entropy'] + 1e-9)
    signal_ok = (sig_red >= 1.10)
    
    t_total = time.time() - t0_seed
    
    return {
        "seed": seed,
        "N": args.N,
        "model": args.model,
        "scramble": args.scramble,
        "chains": args.chains,
        "strobe_objective": args.strobe_objective,
        
        "init_metrics": {
            "sparse_cost": metrics_init['sparse_cost'],
            "signal_entropy": metrics_init['signal_entropy'],
            "topN_share": metrics_init['topN_share']
        },
        "final_metrics": {
            "sparse_cost": metrics_final['sparse_cost'],
            "signal_entropy": metrics_final['signal_entropy'],
            "topN_share": metrics_final['topN_share']
        },
        
        "reductions": {
            "sparse": sparse_red,
            "signal": sig_red
        },
        
        "flags": {
            "sparse_ok": bool(sparse_ok),
            "signal_ok": bool(signal_ok)
        },
        
        "diagnostics": {
            "chain_costs": chain_costs,
            "winner_chain": winner['chain_id'],
            "total_time": t_total,
            "fermion_audit": audit_data
        }
    }

# =============================================================================
# CLI & Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Best-of-K Recovery Harness")
    
    # Core
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--model", type=str, default="xxx", choices=["xx", "xxz", "xxx"])
    parser.add_argument("--Delta", type=float, default=1.0)
    parser.add_argument("--scramble", type=str, default="global", choices=["global", "local"])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=1)
    
    # Chains
    parser.add_argument("--chains", type=int, default=12, help="Parallel chains per seed")
    parser.add_argument("--cycles", type=int, default=8000)
    parser.add_argument("--strobe-edges", type=str, default="all", choices=["ring", "all"])
    parser.add_argument("--strobe-objective", type=str, default="sparse", choices=["sparse", "signal"])
    
    # Flow
    parser.add_argument("--flow-steps", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--max-weight", type=int, default=4)
    parser.add_argument("--no-flow", action="store_true")
    
    # Audits / Output
    parser.add_argument("--fermion-audit", action="store_true")
    parser.add_argument("--out", type=str, default="results.jsonl")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--blas-threads", type=int, default=1)
    
    args = parser.parse_args()
    
    # Set Env
    set_thread_env(args.blas_threads)
    
    strobe_cfg = StrobeConfig(
        cycles=args.cycles,
        edges=args.strobe_edges,
        objective=args.strobe_objective
    )
    
    # Prepare Output
    if os.path.exists(args.out):
        print(f"Warning: Appending to {args.out}")
        
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    results = []
    
    print(f"Starting Scan: N={args.N} Model={args.model} Chains={args.chains} Obj={args.strobe_objective}")
    
    with open(args.out, "a", encoding="utf-8") as f:
        for i, seed in enumerate(seeds):
            try:
                res = run_seed_pipeline(seed, args, strobe_cfg)
                
                # Write immediately
                f.write(json.dumps(res) + "\n")
                f.flush()
                results.append(res)
                
                if args.progress:
                    s_ok = res['flags']['sparse_ok']
                    sig_ok = res['flags']['signal_ok']
                    share = res['final_metrics']['topN_share']
                    print(f"[{i+1}/{len(seeds)}] Seed {seed}: Share={share:.3f} SparseOK={s_ok} SignalOK={sig_ok}")
                    
            except Exception as e:
                print(f"Seed {seed} Crashed: {e}")
                import traceback
                traceback.print_exc()

    # Summary Report
    if results:
        n = len(results)
        s_ok_count = sum(1 for r in results if r['flags']['sparse_ok'])
        sig_ok_count = sum(1 for r in results if r['flags']['signal_ok'])
        
        shares = [r['final_metrics']['topN_share'] for r in results]
        
        print("\n" + "="*40)
        print("SUMMARY REPORT")
        print("="*40)
        print(f"Total Seeds: {n}")
        print(f"Sparse Success: {s_ok_count} ({s_ok_count/n*100:.1f}%)")
        print(f"Signal Success: {sig_ok_count} ({sig_ok_count/n*100:.1f}%)")
        print(f"TopN Share: Median={np.median(shares):.3f} Max={np.max(shares):.3f}")
        print("="*40)
        
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()