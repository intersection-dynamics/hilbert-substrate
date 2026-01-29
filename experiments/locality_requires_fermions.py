#!/usr/bin/env python3
"""
Experiment 1: Does Locality Require Fermions?
==============================================

Tests whether the accessibility collapse (locality emergence) depends on
fermionic structure or is a generic feature of local Hamiltonians.

Paper II showed that for XX model at N >= 5, scrambled states get trapped
in the spatial basin and cannot reach the delocalized Harmonion basis.
But the XX model maps to free fermions via Jordan-Wigner.

Question: Is fermionic structure necessary for locality emergence?

Models tested:
  XX:         H = -Σ (XiXj + YiYj)     → Maps to free fermions
  ZZ:         H = -Σ ZiZj              → Classical Ising, no fermions
  Heisenberg: H = -Σ (XiXj + YiYj + ZiZj) → Interacting spins
  Random:     H = Σ random 2-local     → No special structure

Protocol:
  1. Build local Hamiltonian on 1D ring
  2. Apply global scrambling (cross accessibility barrier)
  3. Attempt locality recovery using local gates
  4. Measure final locality score
  5. Compare across models and system sizes N

Possible outcomes:
  A: Only XX shows strong locality recovery → Fermions required
  B: All models show similar recovery → Locality is generic
  C: Different critical N per model → Fermions affect scaling

Usage:
  python locality_requires_fermions.py --n_min 4 --n_max 10
  python locality_requires_fermions.py --n_values 4 6 8 10 --seeds 1 2 3 4 5

Author: Ben Bray
Date: January 2026
"""

import numpy as np
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# =============================================================================
# PAULI INFRASTRUCTURE
# =============================================================================

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = [I2, X, Y, Z]


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def pauli_string(n: int, positions: List[int], paulis: List[np.ndarray]) -> np.ndarray:
    """Build n-qubit Pauli string with specified paulis at positions."""
    ops = [I2] * n
    for pos, p in zip(positions, paulis):
        ops[pos] = p
    return kron_n(ops)


def random_su4(rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    Q *= np.exp(-1j * np.angle(np.diag(R)))
    return Q


def embed_2q_gate(n: int, i: int, j: int, U4: np.ndarray) -> np.ndarray:
    """Embed 4x4 unitary on qubits i,j into 2^n space."""
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
# HAMILTONIANS
# =============================================================================

def ring_edges(n: int) -> List[Tuple[int, int]]:
    """1D ring with periodic boundary conditions."""
    return [(i, (i + 1) % n) for i in range(n)]


def build_xx_hamiltonian(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    XX model: H = -Σ (XiXj + YiYj) / 2
    Maps to free fermions via Jordan-Wigner.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i, j in edges:
        H -= 0.5 * pauli_string(n, [i, j], [X, X])
        H -= 0.5 * pauli_string(n, [i, j], [Y, Y])
    return 0.5 * (H + H.conj().T)


def build_zz_hamiltonian(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    ZZ (Ising) model: H = -Σ ZiZj
    Classical, diagonal in Z basis, no fermion mapping.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i, j in edges:
        H -= pauli_string(n, [i, j], [Z, Z])
    return 0.5 * (H + H.conj().T)


def build_heisenberg_hamiltonian(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Heisenberg model: H = -Σ (XiXj + YiYj + ZiZj)
    Interacting spins with SU(2) symmetry.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i, j in edges:
        H -= pauli_string(n, [i, j], [X, X])
        H -= pauli_string(n, [i, j], [Y, Y])
        H -= pauli_string(n, [i, j], [Z, Z])
    return 0.5 * (H + H.conj().T)


def build_random_2local_hamiltonian(n: int, edges: List[Tuple[int, int]], 
                                     rng: np.random.Generator) -> np.ndarray:
    """
    Random 2-local Hamiltonian: H = Σ_{ij} Σ_{ab} c_{ij,ab} Pi^a Pj^b
    No special algebraic structure.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    for i, j in edges:
        # Random coefficients for all 9 non-identity Pauli pairs
        for a in range(1, 4):  # X, Y, Z
            for b in range(1, 4):
                coeff = rng.standard_normal()
                H += coeff * pauli_string(n, [i, j], [PAULIS[a], PAULIS[b]])
    
    return 0.5 * (H + H.conj().T)


HAMILTONIANS = {
    "XX": build_xx_hamiltonian,
    "ZZ": build_zz_hamiltonian,
    "Heisenberg": build_heisenberg_hamiltonian,
    "Random": build_random_2local_hamiltonian
}


# =============================================================================
# LOCALITY COST (from Paper II)
# =============================================================================

def pauli_weight(indices: Tuple[int, ...]) -> int:
    """Number of non-identity Paulis."""
    return len(indices)


def locality_cost(H: np.ndarray, n: int, p: float = 4.0) -> float:
    """
    Locality cost C_p(H) from Paper II.
    C_p = Σ_k w(P_k)^p |c_k|^2 / Σ_k |c_k|^2
    where w(P_k) is the Hamming weight (number of non-identity factors).
    
    Lower = more local. Minimum is achieved in eigenbasis (Harmonion).
    Spatial Hamiltonians have cost = 2^p (all terms are weight-2).
    """
    dim = 2 ** n
    
    # Compute Pauli decomposition coefficients
    weights = []
    coeffs_sq = []
    
    # Iterate over all Pauli strings
    for idx in range(4 ** n):
        # Decode index to Pauli indices
        pauli_indices = []
        temp = idx
        ops = []
        weight = 0
        for q in range(n):
            p_idx = temp % 4
            temp //= 4
            ops.append(PAULIS[p_idx])
            if p_idx != 0:  # Non-identity
                weight += 1
        
        P = kron_n(ops)
        c = np.vdot(P.ravel(), H.ravel()) / dim
        c_sq = float(np.abs(c) ** 2)
        
        if c_sq > 1e-20:
            weights.append(weight)
            coeffs_sq.append(c_sq)
    
    # Compute weighted cost
    total = sum(coeffs_sq)
    if total < 1e-15:
        return 0.0
    
    cost = sum(w**p * c for w, c in zip(weights, coeffs_sq)) / total
    return cost


def locality_cost_fast(H: np.ndarray, n: int, p: float = 4.0) -> float:
    """
    Faster locality cost - only compute weight-1 and weight-2 terms.
    For local Hamiltonians, these dominate.
    """
    dim = 2 ** n
    
    weight_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}  # Will accumulate |c|^2 by weight
    
    # Weight 0: Identity
    c = np.trace(H) / dim
    weight_sums[0] = float(np.abs(c) ** 2)
    
    # Weight 1: Single Paulis
    for i in range(n):
        for p_idx in range(1, 4):
            P = pauli_string(n, [i], [PAULIS[p_idx]])
            c = np.vdot(P.ravel(), H.ravel()) / dim
            weight_sums[1] += float(np.abs(c) ** 2)
    
    # Weight 2: Pairs
    for i in range(n):
        for j in range(i + 1, n):
            for a in range(1, 4):
                for b in range(1, 4):
                    P = pauli_string(n, [i, j], [PAULIS[a], PAULIS[b]])
                    c = np.vdot(P.ravel(), H.ravel()) / dim
                    weight_sums[2] += float(np.abs(c) ** 2)
    
    # Everything else is weight >= 3
    total_low = sum(weight_sums.values())
    total_all = np.sum(np.abs(H) ** 2) / dim  # Parseval: sum of |c|^2 = ||H||^2 / dim
    weight_sums[3] = float(total_all - total_low)
    
    total = sum(weight_sums.values())
    if total < 1e-15:
        return 0.0
    
    # Use average weight 3.5 for higher-order terms (conservative estimate)
    cost = sum((w ** p) * weight_sums[w] for w in [0, 1, 2])
    cost += (3.5 ** p) * max(0, weight_sums[3])
    
    return cost / total


# =============================================================================
# SCRAMBLING AND RECOVERY
# =============================================================================

def global_scramble(H: np.ndarray, n: int, depth: int, 
                    rng: np.random.Generator) -> np.ndarray:
    """
    Global scrambling: random 2-qubit gates on ALL pairs.
    This crosses the accessibility barrier (Paper II).
    """
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    H_cur = H.copy()
    
    for _ in range(depth):
        rng.shuffle(all_pairs)
        for (i, j) in all_pairs:
            U = embed_2q_gate(n, i, j, random_su4(rng))
            H_cur = U @ H_cur @ U.conj().T
    
    return H_cur


def local_recovery(H: np.ndarray, n: int, edges: List[Tuple[int, int]],
                   rng: np.random.Generator,
                   sweeps: int = 20,
                   trials_per_edge: int = 30) -> Tuple[np.ndarray, float, List[float]]:
    """
    Attempt to recover locality using only local (edge) gates.
    Greedy optimization: accept any gate that reduces locality cost.
    """
    H_cur = H.copy()
    best_cost = locality_cost_fast(H_cur, n)
    history = [best_cost]
    
    for sweep in range(sweeps):
        improved = False
        for (i, j) in edges:
            edge_best_cost = best_cost
            edge_best_H = H_cur
            
            for _ in range(trials_per_edge):
                U = embed_2q_gate(n, i, j, random_su4(rng))
                H_new = U @ H_cur @ U.conj().T
                cost = locality_cost_fast(H_new, n)
                if cost < edge_best_cost - 1e-9:
                    edge_best_cost = cost
                    edge_best_H = H_new
            
            if edge_best_cost < best_cost - 1e-9:
                H_cur = edge_best_H
                best_cost = edge_best_cost
                improved = True
        
        history.append(best_cost)
        if not improved:
            break
    
    return H_cur, best_cost, history


# =============================================================================
# EIGENBASIS COST (theoretical minimum)
# =============================================================================

def eigenbasis_cost(H: np.ndarray, n: int, p: float = 4.0) -> float:
    """
    Cost in the eigenbasis (Harmonion basis).
    This is the theoretical global minimum - diagonal Hamiltonian.
    """
    eigenvalues, _ = np.linalg.eigh(H)
    
    # In eigenbasis, H is diagonal, so only weight-N terms contribute
    # Actually, compute it properly by diagonalizing and measuring cost
    dim = 2 ** n
    H_diag = np.diag(eigenvalues)
    return locality_cost_fast(H_diag, n, p)


# =============================================================================
# SINGLE RUN
# =============================================================================

@dataclass
class SingleResult:
    model: str
    n_qubits: int
    seed: int
    initial_cost: float
    scrambled_cost: float
    recovered_cost: float
    eigenbasis_cost: float
    spatial_cost: float  # 2^p for p=4 → 16
    recovery_ratio: float  # How much of the gap was closed
    converged_to_spatial: bool  # Did it land near spatial cost?


def run_single(model: str, n_qubits: int, seed: int,
               scramble_depth: int = 3,
               recovery_sweeps: int = 20,
               trials_per_edge: int = 30,
               p: float = 4.0) -> SingleResult:
    """Run one (model, n, seed) configuration."""
    rng = np.random.default_rng(seed)
    edges = ring_edges(n_qubits)
    
    # Build Hamiltonian
    if model == "Random":
        H0 = build_random_2local_hamiltonian(n_qubits, edges, rng)
    else:
        H0 = HAMILTONIANS[model](n_qubits, edges)
    
    # Reference costs
    init_cost = locality_cost_fast(H0, n_qubits, p)
    eigen_cost = eigenbasis_cost(H0, n_qubits, p)
    spatial_cost = 2 ** p  # All weight-2 terms → cost = 2^p
    
    # Global scramble
    H_scr = global_scramble(H0, n_qubits, scramble_depth, rng)
    scr_cost = locality_cost_fast(H_scr, n_qubits, p)
    
    # Local recovery
    _, rec_cost, _ = local_recovery(H_scr, n_qubits, edges, rng,
                                     sweeps=recovery_sweeps,
                                     trials_per_edge=trials_per_edge)
    
    # Metrics
    # Recovery ratio: how much of (scrambled - eigenbasis) gap was closed
    gap = scr_cost - eigen_cost
    if gap > 1e-9:
        recovery_ratio = (scr_cost - rec_cost) / gap
    else:
        recovery_ratio = 0.0
    
    # Did it converge to spatial basin?
    converged_to_spatial = rec_cost < spatial_cost * 1.5  # Within 50% of spatial cost
    
    return SingleResult(
        model=model,
        n_qubits=n_qubits,
        seed=seed,
        initial_cost=float(init_cost),
        scrambled_cost=float(scr_cost),
        recovered_cost=float(rec_cost),
        eigenbasis_cost=float(eigen_cost),
        spatial_cost=float(spatial_cost),
        recovery_ratio=float(recovery_ratio),
        converged_to_spatial=converged_to_spatial
    )


def _run_wrapper(args):
    return run_single(*args)


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_model(model: str, n_values: List[int], seeds: List[int],
                  scramble_depth: int, recovery_sweeps: int,
                  trials_per_edge: int, workers: int,
                  verbose: bool = True) -> Dict:
    """Analyze one model across system sizes."""
    
    tasks = [(model, n, s, scramble_depth, recovery_sweeps, trials_per_edge)
             for n in n_values for s in seeds]
    
    if verbose:
        print(f"  Running {len(tasks)} configurations...")
    
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_wrapper, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
    
    # Aggregate by N
    by_n = {}
    for r in results:
        n = r.n_qubits
        if n not in by_n:
            by_n[n] = []
        by_n[n].append(r)
    
    n_stats = []
    for n in sorted(by_n.keys()):
        runs = by_n[n]
        n_stats.append({
            "n_qubits": n,
            "mean_initial_cost": float(np.mean([r.initial_cost for r in runs])),
            "mean_scrambled_cost": float(np.mean([r.scrambled_cost for r in runs])),
            "mean_recovered_cost": float(np.mean([r.recovered_cost for r in runs])),
            "std_recovered_cost": float(np.std([r.recovered_cost for r in runs])),
            "mean_eigenbasis_cost": float(np.mean([r.eigenbasis_cost for r in runs])),
            "spatial_cost": runs[0].spatial_cost,
            "mean_recovery_ratio": float(np.mean([r.recovery_ratio for r in runs])),
            "fraction_converged_spatial": float(np.mean([r.converged_to_spatial for r in runs])),
            "n_runs": len(runs)
        })
    
    return {
        "model": model,
        "n_stats": n_stats
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test whether locality emergence requires fermionic structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  XX         - Free fermions via Jordan-Wigner (σxσx + σyσy)
  ZZ         - Classical Ising, no fermion mapping (σzσz)
  Heisenberg - Interacting spins with SU(2) symmetry
  Random     - Random 2-local, no special structure

Examples:
  python locality_requires_fermions.py --n_values 4 5 6 7 8
  python locality_requires_fermions.py --models XX ZZ --n_values 4 6 8 10
  python locality_requires_fermions.py --n_min 4 --n_max 10 --n_step 2
        """
    )
    
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["XX", "ZZ", "Heisenberg", "Random"],
                        help="Models to test")
    parser.add_argument("--n_values", type=int, nargs="+", default=None,
                        help="Specific N values to test")
    parser.add_argument("--n_min", type=int, default=4)
    parser.add_argument("--n_max", type=int, default=10)
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--scramble_depth", type=int, default=3)
    parser.add_argument("--recovery_sweeps", type=int, default=20)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="locality_fermion_results.json")
    
    args = parser.parse_args()
    
    # Determine N values
    if args.n_values:
        n_values = args.n_values
    else:
        n_values = list(range(args.n_min, args.n_max + 1, args.n_step))
    
    # Memory check
    max_n = max(n_values)
    if max_n > 12:
        mem_gb = (2**max_n)**2 * 16 / 1e9
        print(f"WARNING: N={max_n} requires ~{mem_gb:.1f} GB RAM")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            return
    
    print("=" * 70)
    print("EXPERIMENT 1: Does Locality Emergence Require Fermionic Structure?")
    print("=" * 70)
    print(f"Models:          {args.models}")
    print(f"N values:        {n_values}")
    print(f"Seeds:           {args.seeds}")
    print(f"Scramble depth:  {args.scramble_depth}")
    print(f"Recovery sweeps: {args.recovery_sweeps}")
    print(f"Trials/edge:     {args.trials}")
    print(f"Workers:         {args.workers}")
    print("=" * 70)
    
    all_results = {
        "metadata": {
            "models": args.models,
            "n_values": n_values,
            "seeds": args.seeds,
            "scramble_depth": args.scramble_depth,
            "recovery_sweeps": args.recovery_sweeps,
            "trials_per_edge": args.trials,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "model_results": {}
    }
    
    for model in args.models:
        print(f"\n{'='*70}")
        print(f"Testing {model} model")
        print(f"{'='*70}")
        t0 = time.time()
        
        result = analyze_model(
            model=model,
            n_values=n_values,
            seeds=args.seeds,
            scramble_depth=args.scramble_depth,
            recovery_sweeps=args.recovery_sweeps,
            trials_per_edge=args.trials,
            workers=args.workers,
            verbose=True
        )
        
        elapsed = time.time() - t0
        
        print(f"\n  {'N':<4} {'Initial':<10} {'Scrambled':<10} {'Recovered':<10} {'Eigenbasis':<10} {'Spatial%':<10}")
        print("  " + "-" * 54)
        for ns in result["n_stats"]:
            spatial_frac = ns["fraction_converged_spatial"] * 100
            print(f"  {ns['n_qubits']:<4} {ns['mean_initial_cost']:<10.2f} "
                  f"{ns['mean_scrambled_cost']:<10.2f} {ns['mean_recovered_cost']:<10.2f} "
                  f"{ns['mean_eigenbasis_cost']:<10.2f} {spatial_frac:<10.0f}%")
        
        print(f"\n  Time: {elapsed:.1f}s")
        
        all_results["model_results"][model] = result
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Recovery to Spatial Basin by Model and N")
    print("=" * 70)
    print(f"(Spatial cost = 16.0 for p=4. 'Converged' = recovered < 24)")
    print()
    
    # Header
    header = f"{'N':<4}"
    for model in args.models:
        header += f" {model:<12}"
    print(header)
    print("-" * (4 + 13 * len(args.models)))
    
    # Data rows
    for n in n_values:
        row = f"{n:<4}"
        for model in args.models:
            stats = all_results["model_results"][model]["n_stats"]
            ns = next((s for s in stats if s["n_qubits"] == n), None)
            if ns:
                row += f" {ns['mean_recovered_cost']:<12.2f}"
            else:
                row += f" {'N/A':<12}"
        print(row)
    
    print()
    print("Interpretation:")
    print("  - If only XX shows low recovered cost at large N → Fermions required")
    print("  - If all models show similar recovery → Locality is generic")
    print("  - If different critical N → Fermions affect scaling but not phenomenon")
    
    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()