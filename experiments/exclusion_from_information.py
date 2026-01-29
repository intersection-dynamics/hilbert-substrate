#!/usr/bin/env python3
"""
Experiment 2: Does Exclusion Emerge from No-Forgetting?
========================================================

The claim: Fermionic exclusion exists to preserve information.
- Two fermions can't occupy the same state
- If they could, individual identity information would be lost
- Exclusion = the universe's defense against information erasure

This experiment tests whether "collision avoidance" correlates with
information preservation under unitary dynamics.

Setup:
  - Initialize 2 distinguishable excitations at known positions
  - Each carries "identity" information (which excitation is which)
  - Evolve with Hamiltonians that favor bunching (attractive) vs spreading (repulsive)
  - Measure: does the system preserve identity information?

Key insight: In a spin-1/2 chain, "double occupancy" is impossible by construction.
But we CAN measure:
  1. How close excitations get (collision proximity)
  2. Whether we can still track "which is which" (identity preservation)
  3. The relationship between these two quantities

Models tested:
  XXZ: H = -Σ (XiXj + YiYj + Δ ZiZj)
    Δ > 0: repulsive (antiferromagnetic), excitations avoid each other
    Δ = 0: free fermions (XX), excitations pass through each other
    Δ < 0: attractive (ferromagnetic), excitations should bunch...but can they?

Prediction:
  If exclusion is fundamental to information preservation:
    - Even with strong attraction (Δ << 0), excitations maintain separation
    - Identity information remains accessible locally
  If exclusion is just a model artifact:
    - Attractive interactions should maximize proximity
    - Identity information should scramble to global correlations

Usage:
  python exclusion_from_information.py --n_qubits 12 --deltas -2 -1 0 1 2
  python exclusion_from_information.py --n_qubits 10 --time_steps 100

Author: Ben Bray
Date: January 2026
"""

import numpy as np
from scipy.stats import entropy
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# =============================================================================
# INFRASTRUCTURE
# =============================================================================

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Ladder operators
Sp = (X + 1j * Y) / 2  # |1><0| raises
Sm = (X - 1j * Y) / 2  # |0><1| lowers


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def pauli_on_site(n: int, site: int, pauli: np.ndarray) -> np.ndarray:
    """Single Pauli on one site."""
    ops = [I2] * n
    ops[site] = pauli
    return kron_n(ops)


def two_site_op(n: int, i: int, j: int, op_i: np.ndarray, op_j: np.ndarray) -> np.ndarray:
    """Operator on two sites."""
    ops = [I2] * n
    ops[i] = op_i
    ops[j] = op_j
    return kron_n(ops)


# =============================================================================
# XXZ HAMILTONIAN
# =============================================================================

def build_xxz_hamiltonian(n: int, delta: float, periodic: bool = False) -> np.ndarray:
    """
    XXZ model: H = -Σ (XiXj + YiYj + Δ ZiZj)
    
    Δ > 0: antiferromagnetic (repulsive for excitations)
    Δ = 0: XX model (free fermions)
    Δ < 0: ferromagnetic (attractive for excitations)
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Determine edges
    edges = [(i, i + 1) for i in range(n - 1)]
    if periodic:
        edges.append((n - 1, 0))
    
    for i, j in edges:
        H -= two_site_op(n, i, j, X, X)
        H -= two_site_op(n, i, j, Y, Y)
        H -= delta * two_site_op(n, i, j, Z, Z)
    
    return H


# =============================================================================
# INITIAL STATES: TWO DISTINGUISHABLE EXCITATIONS
# =============================================================================

def computational_basis_state(n: int, config: List[int]) -> np.ndarray:
    """
    Create computational basis state.
    config[i] = 0 or 1 for each site.
    """
    dim = 2 ** n
    idx = sum(c << i for i, c in enumerate(config))
    psi = np.zeros(dim, dtype=complex)
    psi[idx] = 1.0
    return psi


def two_excitation_state(n: int, pos1: int, pos2: int) -> np.ndarray:
    """
    State with excitations (spin up) at pos1 and pos2, rest spin down.
    |↓↓...↑...↑...↓↓⟩
    """
    config = [0] * n
    config[pos1] = 1
    config[pos2] = 1
    return computational_basis_state(n, config)


def labeled_excitation_state(n: int, pos1: int, pos2: int) -> np.ndarray:
    """
    Create state where excitations are "labeled" by being in a superposition
    that encodes their identity.
    
    |ψ⟩ = (|↑₁↑₂⟩ + i|↑₂↑₁⟩) / √2
    
    This is antisymmetric under exchange - fermionic!
    The phase encodes "which is which".
    """
    # For now, just use the simple two-excitation state
    # The "labeling" comes from knowing the initial positions
    return two_excitation_state(n, pos1, pos2)


# =============================================================================
# OBSERVABLES
# =============================================================================

def site_occupation(psi: np.ndarray, n: int, site: int) -> float:
    """
    Expectation value of occupation at site: ⟨n_i⟩ = (1 + ⟨Z_i⟩) / 2
    """
    Z_op = pauli_on_site(n, site, Z)
    exp_Z = np.real(np.vdot(psi, Z_op @ psi))
    return (1 + exp_Z) / 2


def occupation_profile(psi: np.ndarray, n: int) -> np.ndarray:
    """Get occupation at all sites."""
    return np.array([site_occupation(psi, n, i) for i in range(n)])


def excitation_distance(psi: np.ndarray, n: int) -> float:
    """
    Expected distance between excitations.
    Computed as average |i - j| weighted by probability of finding
    excitations at sites i and j.
    """
    # Get two-point correlations ⟨n_i n_j⟩
    total_dist = 0.0
    total_prob = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Probability of excitations at both i and j
            # |↑_i ↑_j⟩⟨↑_i ↑_j|
            proj = np.zeros(2**n, dtype=complex)
            for basis_idx in range(2**n):
                # Check if this basis state has excitations at i and j
                bit_i = (basis_idx >> i) & 1
                bit_j = (basis_idx >> j) & 1
                if bit_i == 1 and bit_j == 1:
                    proj[basis_idx] = psi[basis_idx]
            
            prob = np.real(np.vdot(proj, proj))
            if prob > 1e-12:
                total_dist += prob * (j - i)
                total_prob += prob
    
    if total_prob < 1e-12:
        return 0.0
    return total_dist / total_prob


def excitation_spread(psi: np.ndarray, n: int) -> float:
    """
    Standard deviation of excitation positions.
    Measures how "spread out" the excitations are.
    """
    occ = occupation_profile(psi, n)
    positions = np.arange(n)
    
    # Normalize occupation to probability
    total = np.sum(occ)
    if total < 1e-12:
        return 0.0
    prob = occ / total
    
    mean_pos = np.sum(positions * prob)
    var_pos = np.sum((positions - mean_pos)**2 * prob)
    return np.sqrt(var_pos)


def nearest_neighbor_correlation(psi: np.ndarray, n: int) -> float:
    """
    Average ⟨n_i n_{i+1}⟩ - measures how often excitations are adjacent.
    Higher = more bunching.
    """
    total = 0.0
    for i in range(n - 1):
        # Project onto states with both i and i+1 occupied
        proj = np.zeros(2**n, dtype=complex)
        for basis_idx in range(2**n):
            if ((basis_idx >> i) & 1) == 1 and ((basis_idx >> (i+1)) & 1) == 1:
                proj[basis_idx] = psi[basis_idx]
        total += np.real(np.vdot(proj, proj))
    
    return total / (n - 1)


def center_of_mass(psi: np.ndarray, n: int) -> float:
    """Center of mass of excitation distribution."""
    occ = occupation_profile(psi, n)
    positions = np.arange(n)
    total = np.sum(occ)
    if total < 1e-12:
        return n / 2
    return np.sum(positions * occ) / total


# =============================================================================
# INFORMATION MEASURES
# =============================================================================

def reduced_density_matrix(psi: np.ndarray, n: int, sites: List[int]) -> np.ndarray:
    """
    Compute reduced density matrix for specified sites.
    Traces out all other sites.
    """
    dim = 2 ** n
    n_keep = len(sites)
    dim_keep = 2 ** n_keep
    
    # Build the reduced density matrix
    rho = np.zeros((dim_keep, dim_keep), dtype=complex)
    
    sites_set = set(sites)
    traced_sites = [i for i in range(n) if i not in sites_set]
    n_traced = len(traced_sites)
    dim_traced = 2 ** n_traced
    
    for i_keep in range(dim_keep):
        for j_keep in range(dim_keep):
            for k_traced in range(dim_traced):
                # Build full basis indices
                i_full = 0
                j_full = 0
                
                # Place kept bits
                for bit_pos, site in enumerate(sites):
                    i_full |= ((i_keep >> bit_pos) & 1) << site
                    j_full |= ((j_keep >> bit_pos) & 1) << site
                
                # Place traced bits (same for both)
                for bit_pos, site in enumerate(traced_sites):
                    bit_val = (k_traced >> bit_pos) & 1
                    i_full |= bit_val << site
                    j_full |= bit_val << site
                
                rho[i_keep, j_keep] += psi[i_full] * np.conj(psi[j_full])
    
    return rho


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy S = -Tr(ρ log ρ)."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return -np.sum(eigenvalues * np.log(eigenvalues))


def half_chain_entanglement(psi: np.ndarray, n: int) -> float:
    """Entanglement entropy between left and right halves."""
    left_sites = list(range(n // 2))
    rho_left = reduced_density_matrix(psi, n, left_sites)
    return von_neumann_entropy(rho_left)


def local_information(psi: np.ndarray, n: int, initial_positions: Tuple[int, int]) -> float:
    """
    Measure how much information about initial positions is still
    accessible from LOCAL measurements.
    
    We compute the trace distance between the reduced density matrices
    of regions near the initial positions.
    
    High value = initial positions still distinguishable locally
    Low value = information has scrambled globally
    """
    pos1, pos2 = initial_positions
    
    # Get local density matrices near initial positions
    # Use 3-site windows centered on initial positions
    window = 1
    sites1 = [max(0, pos1 - window), pos1, min(n-1, pos1 + window)]
    sites2 = [max(0, pos2 - window), pos2, min(n-1, pos2 + window)]
    
    # Remove duplicates and sort
    sites1 = sorted(set(sites1))
    sites2 = sorted(set(sites2))
    
    # Compute reduced density matrices
    rho1 = reduced_density_matrix(psi, n, sites1)
    rho2 = reduced_density_matrix(psi, n, sites2)
    
    # Trace distance: (1/2)||ρ1 - ρ2||_1
    # High trace distance = regions are distinguishable
    diff = rho1 - rho2 if rho1.shape == rho2.shape else None
    if diff is None:
        return 0.0
    
    eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
    trace_norm = np.sum(np.sqrt(np.maximum(eigenvalues, 0)))
    return trace_norm / 2


def position_fidelity(psi: np.ndarray, n: int, initial_positions: Tuple[int, int]) -> float:
    """
    Fidelity with initial state - how much do we still "know" the positions?
    """
    psi_init = two_excitation_state(n, initial_positions[0], initial_positions[1])
    return np.abs(np.vdot(psi_init, psi)) ** 2


# =============================================================================
# TIME EVOLUTION
# =============================================================================

def diagonalize_hamiltonian(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize H once: H = V @ diag(E) @ V†
    Returns (eigenvalues, eigenvectors)
    """
    E, V = np.linalg.eigh(H)
    return E, V


def evolve_state_eigen(psi: np.ndarray, E: np.ndarray, V: np.ndarray, t: float) -> np.ndarray:
    """
    Evolve state using precomputed eigendecomposition.
    |ψ(t)⟩ = V @ diag(e^{-iEt}) @ V† |ψ(0)⟩
    
    Much more memory efficient than scipy.expm.
    """
    # Transform to eigenbasis
    psi_eigen = V.conj().T @ psi
    # Apply phase evolution
    phases = np.exp(-1j * E * t)
    psi_eigen_t = phases * psi_eigen
    # Transform back
    return V @ psi_eigen_t


# =============================================================================
# SINGLE RUN
# =============================================================================

@dataclass
class TimePoint:
    time: float
    occupation_profile: List[float]
    excitation_distance: float
    excitation_spread: float
    nn_correlation: float
    half_chain_entropy: float
    position_fidelity: float
    center_of_mass: float


@dataclass
class RunResult:
    n_qubits: int
    delta: float
    initial_separation: int
    time_points: List[TimePoint]
    # Summary statistics
    min_distance: float
    max_nn_correlation: float
    final_entropy: float
    final_fidelity: float


def run_single(n_qubits: int, delta: float, initial_separation: int,
               max_time: float = 10.0, n_steps: int = 50) -> RunResult:
    """
    Run one (n, Δ, separation) configuration.
    """
    # Initial state: two excitations separated by initial_separation
    center = n_qubits // 2
    pos1 = center - initial_separation // 2
    pos2 = center + (initial_separation + 1) // 2
    
    # Ensure valid positions
    pos1 = max(0, min(n_qubits - 1, pos1))
    pos2 = max(0, min(n_qubits - 1, pos2))
    if pos1 == pos2:
        pos2 = min(pos1 + 1, n_qubits - 1)
    
    psi0 = two_excitation_state(n_qubits, pos1, pos2)
    initial_positions = (pos1, pos2)
    
    # Build Hamiltonian and diagonalize ONCE
    H = build_xxz_hamiltonian(n_qubits, delta, periodic=False)
    E, V = diagonalize_hamiltonian(H)
    
    # Time evolution
    times = np.linspace(0, max_time, n_steps)
    time_points = []
    
    min_dist = float('inf')
    max_nn = 0.0
    
    for t in times:
        psi_t = evolve_state_eigen(psi0, E, V, t)
        
        occ = occupation_profile(psi_t, n_qubits)
        dist = excitation_distance(psi_t, n_qubits)
        spread = excitation_spread(psi_t, n_qubits)
        nn_corr = nearest_neighbor_correlation(psi_t, n_qubits)
        entropy = half_chain_entanglement(psi_t, n_qubits)
        fidelity = position_fidelity(psi_t, n_qubits, initial_positions)
        com = center_of_mass(psi_t, n_qubits)
        
        min_dist = min(min_dist, dist)
        max_nn = max(max_nn, nn_corr)
        
        time_points.append(TimePoint(
            time=float(t),
            occupation_profile=[float(x) for x in occ],
            excitation_distance=float(dist),
            excitation_spread=float(spread),
            nn_correlation=float(nn_corr),
            half_chain_entropy=float(entropy),
            position_fidelity=float(fidelity),
            center_of_mass=float(com)
        ))
    
    final = time_points[-1]
    
    return RunResult(
        n_qubits=n_qubits,
        delta=delta,
        initial_separation=initial_separation,
        time_points=time_points,
        min_distance=float(min_dist),
        max_nn_correlation=float(max_nn),
        final_entropy=float(final.half_chain_entropy),
        final_fidelity=float(final.position_fidelity)
    )


def _run_wrapper(args):
    return run_single(*args)


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_delta(n_qubits: int, delta: float, separations: List[int],
                  max_time: float, n_steps: int, workers: int) -> Dict:
    """Analyze one Δ value across initial separations."""
    
    tasks = [(n_qubits, delta, sep, max_time, n_steps) for sep in separations]
    
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_wrapper, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
    
    # Sort by separation
    results = sorted(results, key=lambda r: r.initial_separation)
    
    return {
        "delta": delta,
        "n_qubits": n_qubits,
        "separations": [r.initial_separation for r in results],
        "min_distances": [r.min_distance for r in results],
        "max_nn_correlations": [r.max_nn_correlation for r in results],
        "final_entropies": [r.final_entropy for r in results],
        "final_fidelities": [r.final_fidelity for r in results],
        "time_series": {
            r.initial_separation: {
                "times": [tp.time for tp in r.time_points],
                "distances": [tp.excitation_distance for tp in r.time_points],
                "nn_correlations": [tp.nn_correlation for tp in r.time_points],
                "entropies": [tp.half_chain_entropy for tp in r.time_points],
                "fidelities": [tp.position_fidelity for tp in r.time_points]
            }
            for r in results
        }
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test whether exclusion emerges from information preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
XXZ Model: H = -Σ (XiXj + YiYj + Δ ZiZj)

  Δ > 0: Repulsive (antiferromagnetic)
  Δ = 0: Free fermions (XX model)  
  Δ < 0: Attractive (ferromagnetic)

Key question: With strong attraction (Δ << 0), do excitations:
  A) Collapse together? (no intrinsic exclusion)
  B) Maintain minimum separation? (exclusion from information preservation)

Examples:
  python exclusion_from_information.py --n_qubits 12 --deltas -2 -1 0 1 2
  python exclusion_from_information.py --n_qubits 10 --max_time 20 --n_steps 100
        """
    )
    
    parser.add_argument("--n_qubits", type=int, default=12)
    parser.add_argument("--deltas", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    parser.add_argument("--separations", type=int, nargs="+", default=None,
                        help="Initial separations to test (default: 2, 4, 6)")
    parser.add_argument("--max_time", type=float, default=10.0)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers (default: 2, keep low for memory)")
    parser.add_argument("--output", type=str, default="exclusion_information_results.json")
    
    args = parser.parse_args()
    
    # Default separations
    if args.separations is None:
        args.separations = [2, 4, min(6, args.n_qubits - 2)]
    
    # Memory check
    if args.n_qubits > 14:
        mem_gb = (2**args.n_qubits)**2 * 16 / 1e9
        print(f"WARNING: N={args.n_qubits} requires ~{mem_gb:.1f} GB RAM")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            return
    
    print("=" * 70)
    print("EXPERIMENT 2: Does Exclusion Emerge from Information Preservation?")
    print("=" * 70)
    print(f"n_qubits:     {args.n_qubits}")
    print(f"Δ values:     {args.deltas}")
    print(f"separations:  {args.separations}")
    print(f"max_time:     {args.max_time}")
    print(f"n_steps:      {args.n_steps}")
    print("=" * 70)
    
    print("\nPhysics reminder:")
    print("  Δ < 0: Attractive - excitations SHOULD bunch if no exclusion")
    print("  Δ = 0: Free fermions - excitations pass through each other")
    print("  Δ > 0: Repulsive - excitations avoid each other")
    print()
    
    all_results = {
        "metadata": {
            "n_qubits": args.n_qubits,
            "deltas": args.deltas,
            "separations": args.separations,
            "max_time": args.max_time,
            "n_steps": args.n_steps,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "delta_results": {}
    }
    
    for delta in args.deltas:
        delta_label = f"Δ={delta:+.1f}"
        interaction = "attractive" if delta < 0 else ("repulsive" if delta > 0 else "free")
        print(f"\n--- Testing {delta_label} ({interaction}) ---")
        t0 = time.time()
        
        result = analyze_delta(
            n_qubits=args.n_qubits,
            delta=delta,
            separations=args.separations,
            max_time=args.max_time,
            n_steps=args.n_steps,
            workers=args.workers
        )
        
        elapsed = time.time() - t0
        
        print(f"  {'Sep':<6} {'Min Dist':<10} {'Max NN':<10} {'Final S':<10} {'Final F':<10}")
        print("  " + "-" * 46)
        for i, sep in enumerate(result["separations"]):
            print(f"  {sep:<6} {result['min_distances'][i]:<10.3f} "
                  f"{result['max_nn_correlations'][i]:<10.3f} "
                  f"{result['final_entropies'][i]:<10.3f} "
                  f"{result['final_fidelities'][i]:<10.3f}")
        
        print(f"  Time: {elapsed:.1f}s")
        
        all_results["delta_results"][str(delta)] = result
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Minimum Distance Achieved (lower = more bunching)")
    print("=" * 70)
    
    header = f"{'Δ':<8}"
    for sep in args.separations:
        header += f" Sep={sep:<6}"
    print(header)
    print("-" * (8 + 10 * len(args.separations)))
    
    for delta in args.deltas:
        row = f"{delta:<+8.1f}"
        result = all_results["delta_results"][str(delta)]
        for min_d in result["min_distances"]:
            row += f" {min_d:<9.3f}"
        print(row)
    
    print("\nInterpretation:")
    print("  - If min_distance has a FLOOR across all Δ → exclusion is intrinsic")
    print("  - If Δ<0 gives smaller min_distance than Δ>0 → exclusion depends on interactions")
    print("  - If min_distance → 1 for Δ<<0 → hard-core (spin) constraint, not information-theoretic")
    
    # Key test: compare most attractive to free fermions
    if -2.0 in args.deltas and 0.0 in args.deltas:
        attr = all_results["delta_results"][str(-2.0)]["min_distances"]
        free = all_results["delta_results"][str(0.0)]["min_distances"]
        print(f"\n  Attractive (Δ=-2) vs Free (Δ=0) min distances:")
        for i, sep in enumerate(args.separations):
            diff = attr[i] - free[i]
            print(f"    Sep={sep}: {attr[i]:.3f} vs {free[i]:.3f} (diff: {diff:+.3f})")
    
    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()