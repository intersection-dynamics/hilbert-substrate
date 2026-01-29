#!/usr/bin/env python3
"""
Experiment 3: Is Qubit Factorization Preferred?
================================================

Critical question: Have we been assuming what we wanted to prove?

Every experiment so far used qubits (d=2), which automatically gives
Jordan-Wigner structure. But a Hilbert space of dimension D=64 could
be factorized many ways:

  - 6 qubits:    2⊗2⊗2⊗2⊗2⊗2  → Has JW fermions
  - 3 ququarts:  4⊗4⊗4          → No JW structure  
  - 2 systems:   8⊗8            → No JW structure
  - Mixed:       2⊗4⊗8          → Partial structure

The test: Given a "generic" Hamiltonian, which factorization does
locality-biased dynamics prefer?

Protocol:
  1. Generate a random Hermitian matrix H in D dimensions
  2. For each candidate factorization F:
     - Interpret H as an operator on F's tensor structure
     - Measure locality cost in F
     - Apply locality recovery in F
     - Measure final locality cost
  3. Compare: which F gives lowest cost / best recovery?

If qubits consistently win → Fermionic structure preferred by constraints
If all F are equivalent → We've been putting fermions in by hand

Usage:
  python factorization_preference.py --dim 64
  python factorization_preference.py --dim 32 --seeds 1 2 3 4 5

Author: Ben Bray
Date: January 2026
"""

import numpy as np
from itertools import product
from typing import List, Tuple, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import argparse
import time

# =============================================================================
# FACTORIZATION UTILITIES
# =============================================================================

def get_factorizations(D: int) -> List[Tuple[int, ...]]:
    """
    Get all nontrivial factorizations of D into factors >= 2.
    Returns list of tuples, e.g., for D=8: [(2,2,2), (2,4), (4,2), (8,)]
    """
    def factor(n, min_factor=2):
        if n == 1:
            return [()]
        results = []
        for f in range(min_factor, n + 1):
            if n % f == 0:
                for rest in factor(n // f, f):
                    results.append((f,) + rest)
        return results
    
    facts = factor(D)
    # Filter: at least 2 factors (so we have tensor structure)
    facts = [f for f in facts if len(f) >= 2]
    # Also include all permutations (2⊗4 vs 4⊗2 matter for locality)
    expanded = set()
    for f in facts:
        # Add all unique permutations
        from itertools import permutations
        for p in permutations(f):
            expanded.add(p)
    return sorted(expanded, key=lambda x: (len(x), x))


def factorization_label(f: Tuple[int, ...]) -> str:
    """Human-readable label for factorization."""
    return "⊗".join(str(d) for d in f)


# =============================================================================
# PAULI-LIKE OPERATORS FOR ARBITRARY LOCAL DIMENSION
# =============================================================================

def generalized_paulis(d: int) -> List[np.ndarray]:
    """
    Generate generalized Pauli operators for dimension d.
    These form a basis for d×d Hermitian matrices.
    
    For d=2: standard Paulis {I, X, Y, Z}
    For d>2: generalized Gell-Mann matrices
    """
    paulis = [np.eye(d, dtype=complex)]  # Identity
    
    # Off-diagonal symmetric (like X)
    for j in range(d):
        for k in range(j + 1, d):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = 1
            m[k, j] = 1
            paulis.append(m)
    
    # Off-diagonal antisymmetric (like Y)
    for j in range(d):
        for k in range(j + 1, d):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = -1j
            m[k, j] = 1j
            paulis.append(m)
    
    # Diagonal (like Z, generalized)
    for l in range(1, d):
        m = np.zeros((d, d), dtype=complex)
        norm = np.sqrt(2 / (l * (l + 1)))
        for j in range(l):
            m[j, j] = norm
        m[l, l] = -l * norm
        paulis.append(m)
    
    return paulis


# =============================================================================
# TENSOR STRUCTURE AND LOCALITY
# =============================================================================

def kron_list(ops: List[np.ndarray]) -> np.ndarray:
    """Tensor product of list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def pauli_weight(indices: Tuple[int, ...], local_dims: Tuple[int, ...]) -> int:
    """
    Compute weight (number of non-identity factors) for a Pauli string.
    indices[i] = 0 means identity on site i.
    """
    return sum(1 for idx in indices if idx != 0)


def decompose_hamiltonian(H: np.ndarray, factorization: Tuple[int, ...]) -> Dict:
    """
    Decompose H into generalized Pauli basis for given factorization.
    Returns dict mapping Pauli index tuple to coefficient.
    """
    D = H.shape[0]
    n_sites = len(factorization)
    
    # Generate local Pauli bases
    local_paulis = [generalized_paulis(d) for d in factorization]
    
    coeffs = {}
    
    # Iterate over all Pauli strings
    ranges = [range(len(lp)) for lp in local_paulis]
    for indices in product(*ranges):
        # Build the Pauli string
        ops = [local_paulis[site][idx] for site, idx in enumerate(indices)]
        P = kron_list(ops)
        
        # Coefficient: c = Tr(P† H) / D
        c = np.trace(P.conj().T @ H) / D
        
        if np.abs(c) > 1e-12:
            coeffs[indices] = complex(c)
    
    return coeffs


def locality_cost_from_decomposition(coeffs: Dict, factorization: Tuple[int, ...], 
                                      p: float = 2.0) -> float:
    """
    Compute locality cost from Pauli decomposition.
    C_p = Σ w(P)^p |c_P|^2 / Σ |c_P|^2
    """
    total_weight = 0.0
    total_norm = 0.0
    
    for indices, c in coeffs.items():
        w = pauli_weight(indices, factorization)
        c_sq = np.abs(c) ** 2
        total_weight += (w ** p) * c_sq
        total_norm += c_sq
    
    if total_norm < 1e-15:
        return 0.0
    return total_weight / total_norm


def locality_cost(H: np.ndarray, factorization: Tuple[int, ...], p: float = 2.0) -> float:
    """Compute locality cost of H in given factorization."""
    coeffs = decompose_hamiltonian(H, factorization)
    return locality_cost_from_decomposition(coeffs, factorization, p)


# =============================================================================
# RANDOM HAMILTONIANS
# =============================================================================

def random_hermitian(D: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random Hermitian matrix."""
    A = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    return (A + A.conj().T) / 2


def random_local_hamiltonian(factorization: Tuple[int, ...], 
                              rng: np.random.Generator,
                              max_weight: int = 2) -> np.ndarray:
    """
    Generate random Hamiltonian that is LOCAL in given factorization.
    Only includes terms up to max_weight body.
    """
    D = np.prod(factorization)
    n_sites = len(factorization)
    H = np.zeros((D, D), dtype=complex)
    
    local_paulis = [generalized_paulis(d) for d in factorization]
    
    # Add random terms up to max_weight
    ranges = [range(len(lp)) for lp in local_paulis]
    for indices in product(*ranges):
        w = pauli_weight(indices, factorization)
        if 0 < w <= max_weight:
            ops = [local_paulis[site][idx] for site, idx in enumerate(indices)]
            P = kron_list(ops)
            coeff = rng.standard_normal()
            H += coeff * P
    
    return (H + H.conj().T) / 2


# =============================================================================
# LOCAL RECOVERY (factorization-aware)
# =============================================================================

def random_local_unitary(factorization: Tuple[int, ...], site: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Generate random unitary acting on one site, embedded in full space."""
    D = int(np.prod(factorization))
    d = factorization[site]
    
    # Random SU(d)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    U_local = Q * np.exp(-1j * np.angle(np.diag(R)))
    
    # Embed in full space
    dims_before = int(np.prod(factorization[:site])) if site > 0 else 1
    dims_after = int(np.prod(factorization[site+1:])) if site < len(factorization)-1 else 1
    
    U = np.kron(np.kron(np.eye(dims_before), U_local), np.eye(dims_after))
    return U


def random_two_site_unitary(factorization: Tuple[int, ...], site1: int, site2: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Generate random unitary acting on two sites."""
    D = int(np.prod(factorization))
    d1, d2 = factorization[site1], factorization[site2]
    d_pair = d1 * d2
    
    # Random SU(d_pair)
    A = rng.standard_normal((d_pair, d_pair)) + 1j * rng.standard_normal((d_pair, d_pair))
    Q, R = np.linalg.qr(A)
    U_local = Q * np.exp(-1j * np.angle(np.diag(R)))
    
    # Embedding is complex for non-adjacent sites, use permutation approach
    # For simplicity, only do adjacent sites
    if site2 != site1 + 1:
        # For non-adjacent, fall back to single-site
        return random_local_unitary(factorization, site1, rng)
    
    dims_before = int(np.prod(factorization[:site1])) if site1 > 0 else 1
    dims_after = int(np.prod(factorization[site2+1:])) if site2 < len(factorization)-1 else 1
    
    U = np.kron(np.kron(np.eye(dims_before), U_local), np.eye(dims_after))
    return U


def local_recovery_in_factorization(H: np.ndarray, factorization: Tuple[int, ...],
                                     rng: np.random.Generator,
                                     sweeps: int = 10,
                                     trials_per_site: int = 20,
                                     p: float = 2.0) -> Tuple[np.ndarray, float]:
    """
    Attempt locality recovery using local unitaries in given factorization.
    """
    H_cur = H.copy()
    best_cost = locality_cost(H_cur, factorization, p)
    n_sites = len(factorization)
    
    for sweep in range(sweeps):
        improved = False
        
        # Single-site moves
        for site in range(n_sites):
            site_best_cost = best_cost
            site_best_H = H_cur
            
            for _ in range(trials_per_site):
                U = random_local_unitary(factorization, site, rng)
                H_new = U @ H_cur @ U.conj().T
                cost = locality_cost(H_new, factorization, p)
                if cost < site_best_cost - 1e-9:
                    site_best_cost = cost
                    site_best_H = H_new
            
            if site_best_cost < best_cost - 1e-9:
                H_cur = site_best_H
                best_cost = site_best_cost
                improved = True
        
        # Two-site moves (adjacent only)
        for site in range(n_sites - 1):
            site_best_cost = best_cost
            site_best_H = H_cur
            
            for _ in range(trials_per_site):
                U = random_two_site_unitary(factorization, site, site + 1, rng)
                H_new = U @ H_cur @ U.conj().T
                cost = locality_cost(H_new, factorization, p)
                if cost < site_best_cost - 1e-9:
                    site_best_cost = cost
                    site_best_H = H_new
            
            if site_best_cost < best_cost - 1e-9:
                H_cur = site_best_H
                best_cost = site_best_cost
                improved = True
        
        if not improved:
            break
    
    return H_cur, best_cost


# =============================================================================
# SINGLE RUN
# =============================================================================

@dataclass
class FactorizationResult:
    factorization: Tuple[int, ...]
    label: str
    n_sites: int
    local_dims: List[int]
    initial_cost: float
    recovered_cost: float
    cost_reduction: float
    has_qubit_structure: bool  # All local dims = 2?


def analyze_factorization(H: np.ndarray, factorization: Tuple[int, ...],
                          rng: np.random.Generator,
                          sweeps: int = 10,
                          trials_per_site: int = 20,
                          p: float = 2.0) -> FactorizationResult:
    """Analyze one factorization."""
    init_cost = locality_cost(H, factorization, p)
    _, rec_cost = local_recovery_in_factorization(
        H, factorization, rng, sweeps, trials_per_site, p
    )
    
    return FactorizationResult(
        factorization=factorization,
        label=factorization_label(factorization),
        n_sites=len(factorization),
        local_dims=list(factorization),
        initial_cost=float(init_cost),
        recovered_cost=float(rec_cost),
        cost_reduction=float(init_cost - rec_cost),
        has_qubit_structure=all(d == 2 for d in factorization)
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(D: int, seed: int, 
                   H_type: str = "random",
                   source_factorization: Tuple[int, ...] = None,
                   sweeps: int = 10,
                   trials_per_site: int = 20,
                   p: float = 2.0) -> Dict:
    """
    Run full experiment for dimension D.
    
    H_type: 
      "random" - completely random Hermitian
      "local_in_source" - random but local in source_factorization
    """
    rng = np.random.default_rng(seed)
    
    # Generate Hamiltonian
    if H_type == "random":
        H = random_hermitian(D, rng)
    elif H_type == "local_in_source":
        H = random_local_hamiltonian(source_factorization, rng, max_weight=2)
    else:
        raise ValueError(f"Unknown H_type: {H_type}")
    
    # Get all factorizations
    factorizations = get_factorizations(D)
    
    # Analyze each
    results = []
    for f in factorizations:
        # Use fresh rng for each to ensure reproducibility
        f_rng = np.random.default_rng(seed + hash(f) % (2**31))
        result = analyze_factorization(H, f, f_rng, sweeps, trials_per_site, p)
        results.append(result)
    
    # Sort by recovered cost
    results = sorted(results, key=lambda r: r.recovered_cost)
    
    return {
        "D": D,
        "seed": seed,
        "H_type": H_type,
        "source_factorization": source_factorization,
        "results": [
            {
                "factorization": r.factorization,
                "label": r.label,
                "n_sites": r.n_sites,
                "local_dims": r.local_dims,
                "initial_cost": r.initial_cost,
                "recovered_cost": r.recovered_cost,
                "cost_reduction": r.cost_reduction,
                "has_qubit_structure": r.has_qubit_structure
            }
            for r in results
        ],
        "winner": results[0].label,
        "winner_is_qubit": results[0].has_qubit_structure
    }


def _run_wrapper(args):
    return run_experiment(*args)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test whether qubit factorization is preferred",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests whether locality-biased dynamics prefer qubit (d=2) factorization
over other tensor structures.

Examples:
  python factorization_preference.py --dim 16
  python factorization_preference.py --dim 64 --seeds 1 2 3
  python factorization_preference.py --dim 32 --H_type local_qubits
        """
    )
    
    parser.add_argument("--dim", type=int, default=16,
                        help="Hilbert space dimension (default: 16)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Random seeds")
    parser.add_argument("--H_type", type=str, default="random",
                        choices=["random", "local_qubits", "local_ququarts"],
                        help="Type of Hamiltonian")
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--p", type=float, default=2.0,
                        help="Locality penalty power")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=str, default="factorization_results.json")
    
    args = parser.parse_args()
    
    D = args.dim
    
    # Determine source factorization if needed
    if args.H_type == "local_qubits":
        n_qubits = int(np.log2(D))
        if 2**n_qubits != D:
            print(f"Error: D={D} is not a power of 2, can't use local_qubits")
            return
        source_fact = tuple([2] * n_qubits)
        H_type = "local_in_source"
    elif args.H_type == "local_ququarts":
        n_ququarts = int(np.log(D) / np.log(4))
        if 4**n_ququarts != D:
            print(f"Error: D={D} is not a power of 4, can't use local_ququarts")
            return
        source_fact = tuple([4] * n_ququarts)
        H_type = "local_in_source"
    else:
        source_fact = None
        H_type = "random"
    
    print("=" * 70)
    print("EXPERIMENT 3: Is Qubit Factorization Preferred?")
    print("=" * 70)
    print(f"Hilbert space dim: {D}")
    print(f"H_type:            {args.H_type}")
    if source_fact:
        print(f"Source factorization: {factorization_label(source_fact)}")
    print(f"Seeds:             {args.seeds}")
    print(f"Penalty power p:   {args.p}")
    print("=" * 70)
    
    # List factorizations
    factorizations = get_factorizations(D)
    print(f"\nFactorizations of {D}:")
    for f in factorizations:
        qubit = " [QUBIT]" if all(d == 2 for d in f) else ""
        print(f"  {factorization_label(f)}{qubit}")
    
    # Run experiments
    print(f"\nRunning {len(args.seeds)} seeds...")
    
    all_results = {
        "metadata": {
            "D": D,
            "H_type": args.H_type,
            "source_factorization": source_fact,
            "seeds": args.seeds,
            "p": args.p,
            "factorizations": [factorization_label(f) for f in factorizations],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "seed_results": []
    }
    
    # Run seeds in parallel
    tasks = [(D, seed, H_type, source_fact, args.sweeps, args.trials, args.p)
             for seed in args.seeds]
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_wrapper, t) for t in tasks]
        for f in as_completed(futures):
            result = f.result()
            all_results["seed_results"].append(result)
            print(f"  Seed {result['seed']}: winner = {result['winner']} "
                  f"{'(QUBIT)' if result['winner_is_qubit'] else ''}")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("RESULTS BY FACTORIZATION (averaged over seeds)")
    print("=" * 70)
    
    # Collect costs by factorization
    fact_costs = {}
    for f in factorizations:
        label = factorization_label(f)
        costs = []
        for sr in all_results["seed_results"]:
            for r in sr["results"]:
                if r["label"] == label:
                    costs.append(r["recovered_cost"])
        if costs:
            fact_costs[label] = {
                "mean": np.mean(costs),
                "std": np.std(costs),
                "is_qubit": all(d == 2 for d in f)
            }
    
    # Sort by mean cost
    sorted_facts = sorted(fact_costs.items(), key=lambda x: x[1]["mean"])
    
    print(f"{'Factorization':<20} {'Mean Cost':<12} {'Std':<12} {'Type':<10}")
    print("-" * 54)
    for label, data in sorted_facts:
        ftype = "QUBIT" if data["is_qubit"] else ""
        print(f"{label:<20} {data['mean']:<12.4f} {data['std']:<12.4f} {ftype:<10}")
    
    # Count wins
    qubit_wins = sum(1 for sr in all_results["seed_results"] if sr["winner_is_qubit"])
    total = len(all_results["seed_results"])
    
    print(f"\nQubit factorization won: {qubit_wins}/{total} seeds")
    
    all_results["summary"] = {
        "qubit_wins": qubit_wins,
        "total_seeds": total,
        "qubit_win_rate": qubit_wins / total,
        "best_factorization": sorted_facts[0][0],
        "best_is_qubit": sorted_facts[0][1]["is_qubit"]
    }
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if sorted_facts[0][1]["is_qubit"]:
        print("✓ Qubit factorization IS preferred by locality dynamics")
        print("  → Fermionic structure may be selected by constraints")
    else:
        print("✗ Qubit factorization is NOT preferred")
        print("  → Our fermionic results may be an artifact of assuming qubits")
    
    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()