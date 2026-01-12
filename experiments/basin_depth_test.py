#!/usr/bin/env python3
"""
Basin Depth Test: Accessibility-Respecting Protocol
====================================================

Tests whether higher-dimensional geometries have deeper accessible basins
using LOCAL scrambling (stays within accessible region per Paper II).

Protocol:
1. Start with local Hamiltonian on geometry G
2. Apply LOCAL scrambling (random gates on G's edges) of depth D
3. Attempt recovery back to G using local gates
4. Find critical D* where recovery fails
5. Compare D* across geometries

If 3D has deeper basin: D*(3D) > D*(2D) > D*(1D)

This addresses the referee critique: we no longer use global scrambling
which violates accessibility constraints.

Usage:
  python basin_depth_test.py --n_qubits 9 --geometries 1d 2d --seeds 1 2 3 4 5
  python basin_depth_test.py --n_qubits 8 --geometries 1d 3d --max_depth 20

Valid configurations:
  1d: any n >= 3
  2d: n must be perfect square (4, 9, 16, 25, ...)
  3d: n must be perfect cube (8, 27, 64, ...)

Author: Ben Bray
Date: January 2026
"""

import numpy as np
from numpy.linalg import eigh
import json
import argparse
from dataclasses import dataclass, asdict
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
    """Tensor product of list of operators."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def random_su4(rng: np.random.Generator) -> np.ndarray:
    """Haar-random SU(4) matrix."""
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    Q *= np.exp(-1j * np.angle(np.diag(R)))
    return Q


def embed_2q_gate(n: int, i: int, j: int, U4: np.ndarray) -> np.ndarray:
    """Embed 4x4 unitary acting on qubits i,j into 2^n dimensional space."""
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

def ring_1d(n: int) -> List[Tuple[int, int]]:
    """1D ring with periodic boundary."""
    return [(i, (i + 1) % n) for i in range(n)]


def lattice_2d(lx: int, ly: int) -> List[Tuple[int, int]]:
    """2D torus (periodic in both directions)."""
    edges = []
    for x in range(lx):
        for y in range(ly):
            idx = x + y * lx
            edges.append((idx, ((x + 1) % lx) + y * lx))
            edges.append((idx, x + ((y + 1) % ly) * lx))
    return edges


def lattice_3d(l: int) -> List[Tuple[int, int]]:
    """3D torus with side length l."""
    edges = []
    for x in range(l):
        for y in range(l):
            for z in range(l):
                idx = x + y * l + z * l * l
                edges.append((idx, ((x + 1) % l) + y * l + z * l * l))
                edges.append((idx, x + ((y + 1) % l) * l + z * l * l))
                edges.append((idx, x + y * l + ((z + 1) % l) * l * l))
    return edges


def get_edges(geometry: str, n_qubits: int) -> List[Tuple[int, int]]:
    """Get edge list for specified geometry."""
    if geometry == "1d":
        return ring_1d(n_qubits)
    elif geometry == "2d":
        side = int(np.sqrt(n_qubits))
        return lattice_2d(side, side)
    elif geometry == "3d":
        side = int(round(n_qubits ** (1/3)))
        return lattice_3d(side)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


# =============================================================================
# HAMILTONIAN AND LOCALITY METRIC
# =============================================================================

def build_xx_ham(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """XX Hamiltonian: H = -1/2 sum_{<ij>} (X_i X_j + Y_i Y_j)"""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i, j in edges:
        ops_x = [I2] * n
        ops_x[i], ops_x[j] = X, X
        ops_y = [I2] * n
        ops_y[i], ops_y[j] = Y, Y
        H -= 0.5 * (kron_n(ops_x) + kron_n(ops_y))
    return 0.5 * (H + H.conj().T)


def locality_score(H: np.ndarray, n: int, edges: List[Tuple[int, int]]) -> float:
    """
    Fraction of 2-body XX+YY weight on target edges vs all pairs.
    Returns value in [0, 1] where 1 = perfectly local.
    """
    dim = 2 ** n
    edge_set = set(edges) | set((j, i) for i, j in edges)
    
    on_edge = 0.0
    off_edge = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            # XX coefficient
            ops_xx = [I2] * n
            ops_xx[i], ops_xx[j] = X, X
            Pxx = kron_n(ops_xx)
            cxx = np.vdot(Pxx.ravel(), H.ravel()) / dim
            
            # YY coefficient
            ops_yy = [I2] * n
            ops_yy[i], ops_yy[j] = Y, Y
            Pyy = kron_n(ops_yy)
            cyy = np.vdot(Pyy.ravel(), H.ravel()) / dim
            
            w = float(np.abs(cxx)**2 + np.abs(cyy)**2)
            if (i, j) in edge_set:
                on_edge += w
            else:
                off_edge += w
    
    total = on_edge + off_edge
    if total < 1e-15:
        return 0.0
    return on_edge / total


# =============================================================================
# LOCAL SCRAMBLING (ACCESSIBILITY-RESPECTING)
# =============================================================================

def local_scramble(H: np.ndarray, n: int, edges: List[Tuple[int, int]], 
                   depth: int, rng: np.random.Generator) -> np.ndarray:
    """
    Apply 'depth' layers of random 2-qubit gates on graph edges.
    This is LOCAL scrambling - stays within the accessible basin.
    Each layer applies one random SU(4) to each edge.
    """
    H_cur = H.copy()
    edge_list = list(edges)
    
    for _ in range(depth):
        rng.shuffle(edge_list)
        for (i, j) in edge_list:
            U = embed_2q_gate(n, i, j, random_su4(rng))
            H_cur = U @ H_cur @ U.conj().T
    
    return H_cur


# =============================================================================
# LOCAL RECOVERY
# =============================================================================

def local_recovery(H: np.ndarray, n: int, edges: List[Tuple[int, int]],
                   rng: np.random.Generator, 
                   sweeps: int = 10, 
                   trials_per_edge: int = 20) -> Tuple[np.ndarray, float, List[float]]:
    """
    Attempt to recover locality using random local gates.
    Greedy: accept any gate that improves locality score.
    
    Returns: (H_recovered, final_score, score_history)
    """
    H_cur = H.copy()
    best_score = locality_score(H_cur, n, edges)
    history = [best_score]
    
    for sweep in range(sweeps):
        improved = False
        for (i, j) in edges:
            edge_best_score = best_score
            edge_best_H = H_cur
            
            for _ in range(trials_per_edge):
                U = embed_2q_gate(n, i, j, random_su4(rng))
                H_new = U @ H_cur @ U.conj().T
                score = locality_score(H_new, n, edges)
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
    """Jordan-Wigner annihilation operator c_j."""
    ops = [Z] * j + [(X + 1j * Y) / 2] + [I2] * (n - j - 1)
    return kron_n(ops)


def test_jw_anticommutation(n: int) -> Dict[str, float]:
    """
    Test fermionic anticommutation relations.
    {c_i, c_j} = 0 and {c_i, c_j†} = δ_ij
    Returns max violations (should be ~machine epsilon).
    """
    c_ops = [jw_annihilation(n, j) for j in range(n)]
    dim = 2**n
    I_full = np.eye(dim, dtype=complex)
    
    max_cc = 0.0
    max_ccdag = 0.0
    
    for i in range(n):
        for j in range(n):
            # {c_i, c_j} should be 0
            anticomm = c_ops[i] @ c_ops[j] + c_ops[j] @ c_ops[i]
            max_cc = max(max_cc, np.linalg.norm(anticomm, 'fro'))
            
            # {c_i, c_j†} should be δ_ij * I
            anticomm_dag = c_ops[i] @ c_ops[j].conj().T + c_ops[j].conj().T @ c_ops[i]
            target = I_full if i == j else np.zeros_like(I_full)
            max_ccdag = max(max_ccdag, np.linalg.norm(anticomm_dag - target, 'fro'))
    
    return {"max_cc_violation": float(max_cc), "max_ccdag_violation": float(max_ccdag)}


# =============================================================================
# SINGLE RUN
# =============================================================================

@dataclass
class SingleRunResult:
    geometry: str
    n_qubits: int
    depth: int
    seed: int
    initial_locality: float
    scrambled_locality: float
    recovered_locality: float
    recovery_ratio: float
    n_edges: int


def run_single(geometry: str, n_qubits: int, depth: int, seed: int,
               recovery_sweeps: int, trials_per_edge: int) -> SingleRunResult:
    """Run one (geometry, depth, seed) configuration."""
    rng = np.random.default_rng(seed)
    edges = get_edges(geometry, n_qubits)
    
    H0 = build_xx_ham(n_qubits, edges)
    init_loc = locality_score(H0, n_qubits, edges)
    
    H_scr = local_scramble(H0, n_qubits, edges, depth, rng)
    scr_loc = locality_score(H_scr, n_qubits, edges)
    
    _, rec_loc, _ = local_recovery(H_scr, n_qubits, edges, rng,
                                    sweeps=recovery_sweeps, 
                                    trials_per_edge=trials_per_edge)
    
    return SingleRunResult(
        geometry=geometry,
        n_qubits=n_qubits,
        depth=depth,
        seed=seed,
        initial_locality=float(init_loc),
        scrambled_locality=float(scr_loc),
        recovered_locality=float(rec_loc),
        recovery_ratio=float(rec_loc / init_loc) if init_loc > 0 else 0.0,
        n_edges=len(edges)
    )


def _run_single_wrapper(args):
    """Wrapper for parallel execution."""
    return run_single(*args)


# =============================================================================
# BASIN DEPTH ANALYSIS
# =============================================================================

def analyze_basin_depth(geometry: str, n_qubits: int, 
                        seeds: List[int],
                        max_depth: int,
                        depth_step: int,
                        recovery_sweeps: int,
                        trials_per_edge: int,
                        workers: int,
                        verbose: bool = True) -> Dict:
    """
    Sweep over scrambling depths to find critical depth D*.
    """
    depths = list(range(0, max_depth + 1, depth_step))
    
    tasks = [
        (geometry, n_qubits, d, s, recovery_sweeps, trials_per_edge)
        for d in depths for s in seeds
    ]
    
    if verbose:
        print(f"  Running {len(tasks)} configurations...")
    
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_single_wrapper, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
    
    by_depth = {}
    for r in results:
        d = r.depth
        if d not in by_depth:
            by_depth[d] = []
        by_depth[d].append(r)
    
    depth_stats = []
    for d in sorted(by_depth.keys()):
        runs = by_depth[d]
        ratios = [r.recovery_ratio for r in runs]
        depth_stats.append({
            "depth": d,
            "mean_recovery_ratio": float(np.mean(ratios)),
            "std_recovery_ratio": float(np.std(ratios)),
            "min_recovery_ratio": float(np.min(ratios)),
            "mean_scrambled_locality": float(np.mean([r.scrambled_locality for r in runs])),
            "mean_recovered_locality": float(np.mean([r.recovered_locality for r in runs])),
            "n_runs": len(runs)
        })
    
    # D* = first depth where mean ratio < threshold
    threshold = 0.7
    d_star = max_depth
    for ds in depth_stats:
        if ds["mean_recovery_ratio"] < threshold:
            d_star = ds["depth"]
            break
    
    edges = get_edges(geometry, n_qubits)
    
    return {
        "geometry": geometry,
        "n_qubits": n_qubits,
        "n_edges": len(edges),
        "d_star": d_star,
        "threshold": threshold,
        "depth_stats": depth_stats
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
        description="Basin depth test with accessibility-respecting local scrambling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python basin_depth_test.py --n_qubits 9 --geometries 1d 2d
  python basin_depth_test.py --n_qubits 8 --geometries 1d 3d
  python basin_depth_test.py --n_qubits 16 --geometries 1d 2d --seeds 1 2 3

Valid n_qubits:
  1d: any n >= 3
  2d: perfect squares (4, 9, 16, 25, ...)
  3d: perfect cubes (8, 27, 64, ...)
        """
    )
    
    parser.add_argument("--n_qubits", type=int, default=9)
    parser.add_argument("--geometries", type=str, nargs="+", default=["1d", "2d"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--depth_step", type=int, default=2)
    parser.add_argument("--recovery_sweeps", type=int, default=10)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="basin_depth_results.json")
    
    args = parser.parse_args()
    
    for g in args.geometries:
        if not validate_geometry(g, args.n_qubits):
            print(f"Error: n_qubits={args.n_qubits} incompatible with {g}")
            return
    
    print("=" * 65)
    print("BASIN DEPTH TEST: Accessibility-Respecting Protocol")
    print("=" * 65)
    print(f"n_qubits:        {args.n_qubits}")
    print(f"geometries:      {args.geometries}")
    print(f"seeds:           {args.seeds}")
    print(f"max_depth:       {args.max_depth}")
    print(f"recovery_sweeps: {args.recovery_sweeps}")
    print(f"trials/edge:     {args.trials}")
    print(f"workers:         {args.workers}")
    print("=" * 65)
    
    print("\n[1/2] Testing Jordan-Wigner anticommutation...")
    jw_result = test_jw_anticommutation(args.n_qubits)
    print(f"  max ||{{c_i, c_j}}||:        {jw_result['max_cc_violation']:.2e}")
    print(f"  max ||{{c_i, c_j†}} - δ||:   {jw_result['max_ccdag_violation']:.2e}")
    
    print(f"\n[2/2] Measuring basin depths...")
    
    all_results = {
        "metadata": {
            "n_qubits": args.n_qubits,
            "geometries": args.geometries,
            "seeds": args.seeds,
            "max_depth": args.max_depth,
            "depth_step": args.depth_step,
            "recovery_sweeps": args.recovery_sweeps,
            "trials_per_edge": args.trials,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "jw_anticommutation": jw_result,
        "basin_depth": {}
    }
    
    for geom in args.geometries:
        print(f"\n--- {geom.upper()} geometry ---")
        t0 = time.time()
        
        result = analyze_basin_depth(
            geometry=geom,
            n_qubits=args.n_qubits,
            seeds=args.seeds,
            max_depth=args.max_depth,
            depth_step=args.depth_step,
            recovery_sweeps=args.recovery_sweeps,
            trials_per_edge=args.trials,
            workers=args.workers,
            verbose=True
        )
        
        elapsed = time.time() - t0
        print(f"  Edges: {result['n_edges']}")
        print(f"  D* (critical depth): {result['d_star']}")
        print(f"  Time: {elapsed:.1f}s")
        
        all_results["basin_depth"][geom] = result
    
    print("\n" + "=" * 65)
    print("SUMMARY: Critical Depths D*")
    print("=" * 65)
    print(f"{'Geometry':<12} {'Edges':<8} {'D*':<8}")
    print("-" * 28)
    for geom in args.geometries:
        r = all_results["basin_depth"][geom]
        print(f"{geom:<12} {r['n_edges']:<8} {r['d_star']:<8}")
    
    print("\nInterpretation:")
    print("  D* = max local scrambling depth from which recovery succeeds")
    print("  Higher D* = deeper/more robust basin")
    
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()