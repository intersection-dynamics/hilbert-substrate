#!/usr/bin/env python3
"""
Dense-to-Sparse Geometry Recovery Test
=======================================

Tests which sparse geometry is most accessible from a dense local configuration.

Protocol:
1. Start with local Hamiltonian on complete graph (all pairs coupled)
2. Apply dense local scrambling (random 2-qubit gates on ALL pairs)
3. Attempt recovery to different SPARSE geometries (1D, 2D, 3D)
4. Compare recovery success across geometries

This is accessibility-respecting: only 2-qubit gates, no global Haar unitaries.
We're testing which sparse structure is easiest to reach from the "hot" dense region.

Hypothesis: If 3D has the deepest/widest basin, recovery to 3D should succeed
better than 2D, which should succeed better than 1D.

Usage:
  python dense_to_sparse_test.py --n_qubits 8 --targets 1d 3d
  python dense_to_sparse_test.py --n_qubits 9 --targets 1d 2d
  python dense_to_sparse_test.py --n_qubits 27 --targets 1d 3d --workers 8

Valid targets:
  1d: any n >= 3
  2d: n must be perfect square (4, 9, 16, 25, ...)
  3d: n must be perfect cube (8, 27, 64, ...)

Author: Ben Bray
Date: January 2026
"""

import numpy as np
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# =============================================================================
# BASIC INFRASTRUCTURE
# =============================================================================

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def random_su4(rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    Q *= np.exp(-1j * np.angle(np.diag(R)))
    return Q


def embed_2q_gate(n: int, i: int, j: int, U4: np.ndarray) -> np.ndarray:
    dim = 2 ** n
    rest = [k for k in range(n) if k not in (i, j)]
    perm = [i, j] + rest
    
    P = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        bits = [(b >> k) & 1 for k in range(n)]
        new_bits = [bits[p] for p in perm]
        new_b = sum(new_bits[k] << k for k in range(n))
        P[new_b, b] = 1.0
    
    U_big = np.kron(U4, np.eye(2 ** (n - 2), dtype=complex))
    return P.conj().T @ U_big @ P


# =============================================================================
# GRAPH TOPOLOGIES
# =============================================================================

def complete_graph(n: int) -> List[Tuple[int, int]]:
    """All pairs - complete graph."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def ring_1d(n: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def lattice_2d(n: int) -> List[Tuple[int, int]]:
    side = int(np.sqrt(n))
    edges = []
    for x in range(side):
        for y in range(side):
            idx = x + y * side
            edges.append((idx, ((x + 1) % side) + y * side))
            edges.append((idx, x + ((y + 1) % side) * side))
    return edges


def lattice_3d(n: int) -> List[Tuple[int, int]]:
    side = int(round(n ** (1/3)))
    edges = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                idx = x + y * side + z * side * side
                edges.append((idx, ((x + 1) % side) + y * side + z * side * side))
                edges.append((idx, x + ((y + 1) % side) * side + z * side * side))
                edges.append((idx, x + y * side + ((z + 1) % side) * side * side))
    return edges


def get_edges(geometry: str, n: int) -> List[Tuple[int, int]]:
    if geometry == "complete":
        return complete_graph(n)
    elif geometry == "1d":
        return ring_1d(n)
    elif geometry == "2d":
        return lattice_2d(n)
    elif geometry == "3d":
        return lattice_3d(n)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def geometry_info(geometry: str, n: int) -> Dict:
    """Return info about geometry."""
    edges = get_edges(geometry, n)
    if geometry == "1d":
        dim, coord = 1, n
    elif geometry == "2d":
        side = int(np.sqrt(n))
        dim, coord = 2, side
    elif geometry == "3d":
        side = int(round(n ** (1/3)))
        dim, coord = 3, side
    else:
        dim, coord = 0, n
    return {"edges": len(edges), "dimension": dim, "coordination": 2 * dim if dim > 0 else n-1}


# =============================================================================
# HAMILTONIAN AND LOCALITY METRIC
# =============================================================================

def build_xx_ham(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i, j in edges:
        ops_x = [I2] * n
        ops_x[i], ops_x[j] = X, X
        ops_y = [I2] * n
        ops_y[i], ops_y[j] = Y, Y
        H -= 0.5 * (kron_n(ops_x) + kron_n(ops_y))
    return 0.5 * (H + H.conj().T)


def locality_score(H: np.ndarray, n: int, target_edges: List[Tuple[int, int]]) -> float:
    """
    Fraction of 2-body XX+YY weight on target edges vs ALL pairs.
    """
    dim = 2 ** n
    edge_set = set(target_edges) | set((j, i) for i, j in target_edges)
    
    on_target = 0.0
    off_target = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            ops_xx = [I2] * n
            ops_xx[i], ops_xx[j] = X, X
            Pxx = kron_n(ops_xx)
            cxx = np.vdot(Pxx.ravel(), H.ravel()) / dim
            
            ops_yy = [I2] * n
            ops_yy[i], ops_yy[j] = Y, Y
            Pyy = kron_n(ops_yy)
            cyy = np.vdot(Pyy.ravel(), H.ravel()) / dim
            
            w = float(np.abs(cxx)**2 + np.abs(cyy)**2)
            if (i, j) in edge_set:
                on_target += w
            else:
                off_target += w
    
    total = on_target + off_target
    if total < 1e-15:
        return 0.0
    return on_target / total


# =============================================================================
# DENSE SCRAMBLING
# =============================================================================

def dense_scramble(H: np.ndarray, n: int, depth: int, 
                   rng: np.random.Generator) -> np.ndarray:
    """
    Apply random 2-qubit gates on ALL pairs (complete graph).
    Each layer applies one random SU(4) to each pair.
    This is still 2-local (accessibility-respecting) but maximally connected.
    """
    all_pairs = complete_graph(n)
    H_cur = H.copy()
    
    for _ in range(depth):
        rng.shuffle(all_pairs)
        for (i, j) in all_pairs:
            U = embed_2q_gate(n, i, j, random_su4(rng))
            H_cur = U @ H_cur @ U.conj().T
    
    return H_cur


# =============================================================================
# SPARSE RECOVERY
# =============================================================================

def sparse_recovery(H: np.ndarray, n: int, target_edges: List[Tuple[int, int]],
                    rng: np.random.Generator,
                    sweeps: int = 15,
                    trials_per_edge: int = 25) -> Tuple[np.ndarray, float, List[float]]:
    """
    Recover locality to a SPARSE target geometry using only target edge gates.
    """
    H_cur = H.copy()
    best_score = locality_score(H_cur, n, target_edges)
    history = [best_score]
    
    for sweep in range(sweeps):
        improved = False
        for (i, j) in target_edges:
            edge_best_score = best_score
            edge_best_H = H_cur
            
            for _ in range(trials_per_edge):
                U = embed_2q_gate(n, i, j, random_su4(rng))
                H_new = U @ H_cur @ U.conj().T
                score = locality_score(H_new, n, target_edges)
                if score > edge_best_score + 1e-9:
                    edge_best_score = score
                    edge_best_H = H_new
            
            if edge_best_score > best_score + 1e-9:
                H_cur = edge_best_H
                best_score = edge_best_score
                improved = True
        
        history.append(best_score)
        if not improved:
            break
    
    return H_cur, best_score, history


# =============================================================================
# JORDAN-WIGNER TEST
# =============================================================================

def jw_annihilation(n: int, j: int) -> np.ndarray:
    ops = [Z] * j + [(X + 1j * Y) / 2] + [I2] * (n - j - 1)
    return kron_n(ops)


def test_jw_anticommutation(n: int) -> Dict[str, float]:
    """
    Test fermionic anticommutation. Skip for n > 12 (memory constraints).
    """
    if n > 12:
        return {"max_cc_violation": "skipped (n>12)", "max_ccdag_violation": "skipped (n>12)"}
    
    c_ops = [jw_annihilation(n, j) for j in range(n)]
    dim = 2**n
    I_full = np.eye(dim, dtype=complex)
    
    max_cc = 0.0
    max_ccdag = 0.0
    
    for i in range(n):
        for j in range(n):
            anticomm = c_ops[i] @ c_ops[j] + c_ops[j] @ c_ops[i]
            max_cc = max(max_cc, np.linalg.norm(anticomm, 'fro'))
            
            anticomm_dag = c_ops[i] @ c_ops[j].conj().T + c_ops[j].conj().T @ c_ops[i]
            target = I_full if i == j else np.zeros_like(I_full)
            max_ccdag = max(max_ccdag, np.linalg.norm(anticomm_dag - target, 'fro'))
    
    return {"max_cc_violation": float(max_cc), "max_ccdag_violation": float(max_ccdag)}


# =============================================================================
# SINGLE RUN
# =============================================================================

@dataclass 
class SingleResult:
    target_geometry: str
    n_qubits: int
    scramble_depth: int
    seed: int
    initial_score: float  # Score before scrambling (should be ~edge_frac)
    scrambled_score: float  # Score right after dense scrambling
    recovered_score: float  # Score after recovery
    n_target_edges: int
    n_total_pairs: int
    recovery_history: List[float]


def run_single(target_geometry: str, n_qubits: int, scramble_depth: int, seed: int,
               recovery_sweeps: int, trials_per_edge: int) -> SingleResult:
    """
    Run one test: dense scramble -> sparse recovery to target geometry.
    """
    rng = np.random.default_rng(seed)
    
    # Target edges (sparse geometry)
    target_edges = get_edges(target_geometry, n_qubits)
    n_total_pairs = n_qubits * (n_qubits - 1) // 2
    
    # Start with XX Hamiltonian on COMPLETE graph (dense)
    complete_edges = complete_graph(n_qubits)
    H0 = build_xx_ham(n_qubits, complete_edges)
    
    # Initial score relative to target (will be low since H is on complete graph)
    init_score = locality_score(H0, n_qubits, target_edges)
    
    # Dense scramble (2-local on all pairs)
    H_scr = dense_scramble(H0, n_qubits, scramble_depth, rng)
    scr_score = locality_score(H_scr, n_qubits, target_edges)
    
    # Recover to sparse target geometry
    _, rec_score, history = sparse_recovery(
        H_scr, n_qubits, target_edges, rng,
        sweeps=recovery_sweeps,
        trials_per_edge=trials_per_edge
    )
    
    return SingleResult(
        target_geometry=target_geometry,
        n_qubits=n_qubits,
        scramble_depth=scramble_depth,
        seed=seed,
        initial_score=float(init_score),
        scrambled_score=float(scr_score),
        recovered_score=float(rec_score),
        n_target_edges=len(target_edges),
        n_total_pairs=n_total_pairs,
        recovery_history=[float(x) for x in history]
    )


def _run_single_wrapper(args):
    return run_single(*args)


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_geometry(target: str, n_qubits: int, seeds: List[int],
                     scramble_depth: int, recovery_sweeps: int,
                     trials_per_edge: int, workers: int,
                     verbose: bool = True) -> Dict:
    """
    Test recovery to one target geometry across multiple seeds.
    """
    tasks = [(target, n_qubits, scramble_depth, s, recovery_sweeps, trials_per_edge)
             for s in seeds]
    
    if verbose:
        print(f"  Running {len(tasks)} seeds...")
    
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_single_wrapper, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
    
    # Aggregate
    init_scores = [r.initial_score for r in results]
    scr_scores = [r.scrambled_score for r in results]
    rec_scores = [r.recovered_score for r in results]
    
    # Theoretical max: fraction of pairs that are target edges
    edge_frac = results[0].n_target_edges / results[0].n_total_pairs
    
    return {
        "target_geometry": target,
        "n_qubits": n_qubits,
        "scramble_depth": scramble_depth,
        "n_target_edges": results[0].n_target_edges,
        "n_total_pairs": results[0].n_total_pairs,
        "edge_fraction": float(edge_frac),
        "mean_initial_score": float(np.mean(init_scores)),
        "mean_scrambled_score": float(np.mean(scr_scores)),
        "std_scrambled_score": float(np.std(scr_scores)),
        "mean_recovered_score": float(np.mean(rec_scores)),
        "std_recovered_score": float(np.std(rec_scores)),
        "recovery_gain": float(np.mean(rec_scores) - np.mean(scr_scores)),
        "normalized_recovery": float(np.mean(rec_scores) / edge_frac) if edge_frac > 0 else 0,
        "per_seed": [
            {
                "seed": r.seed,
                "scrambled": r.scrambled_score,
                "recovered": r.recovered_score,
                "gain": r.recovered_score - r.scrambled_score
            }
            for r in sorted(results, key=lambda x: x.seed)
        ]
    }


# =============================================================================
# MAIN
# =============================================================================

def validate_geometry(geometry: str, n_qubits: int) -> bool:
    if geometry == "1d":
        return n_qubits >= 3
    elif geometry == "2d":
        side = int(np.sqrt(n_qubits))
        return side * side == n_qubits and side >= 2
    elif geometry == "3d":
        side = int(round(n_qubits ** (1/3)))
        return side ** 3 == n_qubits and side >= 2
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Dense-to-sparse geometry recovery test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dense_to_sparse_test.py --n_qubits 8 --targets 1d 3d
  python dense_to_sparse_test.py --n_qubits 9 --targets 1d 2d
  python dense_to_sparse_test.py --n_qubits 16 --targets 1d 2d

Valid targets for n_qubits:
  1d: any n >= 3
  2d: perfect squares (4, 9, 16, ...)
  3d: perfect cubes (8, 27, 64, ...)

NOTE: Memory usage scales as 2^(2n). Max practical n:
  n=12: ~500 MB
  n=14: ~8 GB  
  n=16: ~128 GB (needs large RAM)
  n>16: requires sparse methods (not implemented here)
        """
    )
    
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--targets", type=str, nargs="+", default=["1d", "3d"],
                        help="Target geometries to recover to")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--depth", type=int, default=3,
                        help="Dense scrambling depth (layers of all-pairs gates)")
    parser.add_argument("--recovery_sweeps", type=int, default=15)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="dense_to_sparse_results.json")
    
    args = parser.parse_args()
    
    # Memory check
    dim = 2 ** args.n_qubits
    mem_gb = (dim * dim * 16) / (1024**3)  # complex128 = 16 bytes
    if args.n_qubits > 14:
        print(f"WARNING: n_qubits={args.n_qubits} requires ~{mem_gb:.0f} GB RAM per matrix")
        print("         This will likely crash. Use n_qubits <= 12 for safety.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            return
    
    args = parser.parse_args()
    
    # Validate
    for t in args.targets:
        if not validate_geometry(t, args.n_qubits):
            print(f"Error: n_qubits={args.n_qubits} incompatible with {t}")
            print("  1d: any n >= 3")
            print("  2d: perfect square (4, 9, 16, ...)")
            print("  3d: perfect cube (8, 27, 64, ...)")
            return
    
    n_pairs = args.n_qubits * (args.n_qubits - 1) // 2
    
    print("=" * 65)
    print("DENSE-TO-SPARSE GEOMETRY RECOVERY TEST")
    print("=" * 65)
    print(f"n_qubits:         {args.n_qubits}")
    print(f"total pairs:      {n_pairs}")
    print(f"targets:          {args.targets}")
    print(f"scramble depth:   {args.depth}")
    print(f"seeds:            {args.seeds}")
    print(f"recovery_sweeps:  {args.recovery_sweeps}")
    print(f"trials/edge:      {args.trials}")
    print(f"workers:          {args.workers}")
    print("=" * 65)
    
    # Target geometry info
    print("\nTarget geometry info:")
    for t in args.targets:
        info = geometry_info(t, args.n_qubits)
        frac = info['edges'] / n_pairs
        print(f"  {t}: {info['edges']} edges ({frac:.1%} of pairs), coord={info['coordination']}")
    
    # JW test
    print("\n[1/2] Testing Jordan-Wigner anticommutation...")
    jw_result = test_jw_anticommutation(args.n_qubits)
    print(f"  max ||{{c_i, c_j}}||:        {jw_result['max_cc_violation']:.2e}")
    print(f"  max ||{{c_i, c_j†}} - δ||:   {jw_result['max_ccdag_violation']:.2e}")
    
    # Recovery tests
    print(f"\n[2/2] Testing recovery to sparse geometries...")
    
    all_results = {
        "metadata": {
            "n_qubits": args.n_qubits,
            "n_total_pairs": n_pairs,
            "targets": args.targets,
            "scramble_depth": args.depth,
            "seeds": args.seeds,
            "recovery_sweeps": args.recovery_sweeps,
            "trials_per_edge": args.trials,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "jw_anticommutation": jw_result,
        "recovery_results": {}
    }
    
    for target in args.targets:
        print(f"\n--- Recovery to {target.upper()} ---")
        t0 = time.time()
        
        result = analyze_geometry(
            target=target,
            n_qubits=args.n_qubits,
            seeds=args.seeds,
            scramble_depth=args.depth,
            recovery_sweeps=args.recovery_sweeps,
            trials_per_edge=args.trials,
            workers=args.workers,
            verbose=True
        )
        
        elapsed = time.time() - t0
        print(f"  Target edges:     {result['n_target_edges']} ({result['edge_fraction']:.1%} of pairs)")
        print(f"  Scrambled score:  {result['mean_scrambled_score']:.4f} ± {result['std_scrambled_score']:.4f}")
        print(f"  Recovered score:  {result['mean_recovered_score']:.4f} ± {result['std_recovered_score']:.4f}")
        print(f"  Recovery gain:    {result['recovery_gain']:+.4f}")
        print(f"  Normalized:       {result['normalized_recovery']:.2f}x edge fraction")
        print(f"  Time:             {elapsed:.1f}s")
        
        all_results["recovery_results"][target] = result
    
    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY: Recovery Success by Geometry")
    print("=" * 65)
    print(f"{'Geometry':<10} {'Edges':<8} {'Scrambled':<12} {'Recovered':<12} {'Gain':<10} {'Normalized':<10}")
    print("-" * 62)
    
    for target in args.targets:
        r = all_results["recovery_results"][target]
        print(f"{target:<10} {r['n_target_edges']:<8} {r['mean_scrambled_score']:<12.4f} "
              f"{r['mean_recovered_score']:<12.4f} {r['recovery_gain']:<+10.4f} {r['normalized_recovery']:<10.2f}")
    
    # Interpretation
    print("\nInterpretation:")
    print("  'Normalized' = recovered_score / edge_fraction")
    print("  Values > 1.0 mean recovery concentrates weight on target edges")
    print("  Higher normalized score = geometry is more accessible from dense state")
    
    # Which won?
    best = max(args.targets, key=lambda t: all_results["recovery_results"][t]["normalized_recovery"])
    print(f"\n  Best recovery: {best.upper()}")
    
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()