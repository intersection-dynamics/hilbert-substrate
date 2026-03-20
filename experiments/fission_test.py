"""
Fission Test: Can one subsystem become two?
============================================

The question: start with ONE thing (a monolithic quantum state with
high entanglement across every possible split). Evolve it. Does it
ever spontaneously become TWO things?

Method: at each timestep, compute entanglement entropy for ALL
bipartitions of the qubits. If any bipartition develops low entropy,
the system has fissioned — one thing became two.

Scenarios:
  1. MONOLITH → LOCAL H: Haar random state (one blob) evolving under
     Heisenberg chain. Can locality create a split?

  2. QUENCH: Ground state of all-to-all H (deeply entangled monolith),
     suddenly evolve under local H. Does the monolith crack?

  3. PRODUCT → LOCAL H: N independent qubits evolving under Heisenberg.
     They fuse. Do they ever re-fission?

  4. STRUCTURED → RANDOM H: Ground state of Heisenberg (has structure),
     quenched to random H. Does structure dissolve? Reform differently?
"""

import numpy as np
from scipy.linalg import eigh, expm
from itertools import combinations
import json
import argparse
import time
import os


def haar_random_state(D):
    """Haar-random pure state — the purest monolith."""
    psi = np.random.randn(D) + 1j * np.random.randn(D)
    return psi / np.linalg.norm(psi)


def heisenberg_chain(n_qubits):
    """Nearest-neighbor Heisenberg XXX with periodic BC."""
    D = 2 ** n_qubits
    H = np.zeros((D, D), dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        for pauli in [sx, sy, sz]:
            ops = [I2] * n_qubits
            ops[i] = pauli
            ops[j] = pauli
            term = ops[0]
            for k in range(1, n_qubits):
                term = np.kron(term, ops[k])
            H += term
    return H


def all_to_all_hamiltonian(n_qubits):
    """All-to-all Heisenberg coupling — makes deeply entangled ground state."""
    D = 2 ** n_qubits
    H = np.zeros((D, D), dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            for pauli in [sx, sy, sz]:
                ops = [I2] * n_qubits
                ops[i] = pauli
                ops[j] = pauli
                term = ops[0]
                for k in range(1, n_qubits):
                    term = np.kron(term, ops[k])
                H += term
    return H


def random_gue(D, scale=1.0):
    """Random GUE Hamiltonian."""
    A = np.random.randn(D, D) + 1j * np.random.randn(D, D)
    H = (A + A.conj().T) / 2
    H *= scale / np.linalg.norm(H, 'fro') * D
    return H


def ground_state(H):
    """Ground state of H."""
    evals, evecs = eigh(H)
    return evecs[:, 0]


def all_bipartitions(n_qubits):
    """All bipartitions of n qubits into two non-empty groups.
    Returns list of (A_indices, B_indices) without duplicates."""
    qubits = list(range(n_qubits))
    bipartitions = []
    seen = set()
    for size_A in range(1, n_qubits):
        for combo in combinations(qubits, size_A):
            A = list(combo)
            B = [q for q in qubits if q not in combo]
            # Canonical form: use the smaller set, or lexicographically first
            key = frozenset(A)
            complement = frozenset(B)
            if complement in seen:
                continue
            seen.add(key)
            bipartitions.append((A, B))
    return bipartitions


def bipartition_entropy(psi, A_qubits, B_qubits, n_qubits):
    """Entanglement entropy for bipartition A|B of a pure state.
    
    Reshapes state into tensor, transposes A-qubits first, SVD.
    """
    n_A = len(A_qubits)
    n_B = len(B_qubits)
    d_A = 2 ** n_A
    d_B = 2 ** n_B

    # Reshape psi into (2,2,...,2) tensor
    psi_tensor = psi.reshape([2] * n_qubits)

    # Transpose: A-qubits first, then B-qubits
    order = A_qubits + B_qubits
    psi_tensor = np.transpose(psi_tensor, order)

    # Reshape into (d_A, d_B) matrix
    psi_mat = psi_tensor.reshape(d_A, d_B)

    # SVD
    s = np.linalg.svd(psi_mat, compute_uv=False)
    s = s[s > 1e-15]
    s2 = s ** 2
    return float(-np.sum(s2 * np.log2(s2 + 1e-30)))


def compute_all_entropies(psi, bipartitions, n_qubits):
    """Compute entanglement entropy for all bipartitions."""
    entropies = []
    for A, B in bipartitions:
        S = bipartition_entropy(psi, A, B, n_qubits)
        S_max = min(len(A), len(B))  # log2(2^min) = min
        entropies.append({
            'S': S,
            'S_max': S_max,
            'S_frac': S / S_max if S_max > 0 else 0.0,
        })
    return entropies


def subsystem_count(entropies, threshold=0.3):
    """Count how many bipartitions have normalized entropy below threshold.
    
    A rough measure of 'fissibility': how many ways can this state
    be described as approximately two independent things?
    """
    n_low = sum(1 for e in entropies if e['S_frac'] < threshold)
    return n_low


def count_subsystems(all_S, bipartitions, n_qubits, threshold=0.3):
    """Estimate subsystem count from bipartition entropies.
    
    Strategy: find the best (lowest entropy) bipartition.
    If it's below threshold, that's 2 subsystems.
    Then check if each half could further split by looking at
    bipartitions that refine the best one.
    
    Returns (n_subsystems, best_split_info).
    """
    n_bip = len(all_S)
    
    # Find minimum S/S_max bipartition
    min_idx = min(range(n_bip), key=lambda i: all_S[i]['S_frac'])
    best = all_S[min_idx]
    
    if best['S_frac'] >= threshold:
        return 1, None  # monolith
    
    A, B = bipartitions[min_idx]
    
    # Check if A further splits: find bipartitions where one side is a
    # strict subset of A and the other side contains all of B plus the rest of A
    n_sub = 2
    for side in [A, B]:
        if len(side) < 2:
            continue
        # Look for a bipartition that splits 'side' while keeping the
        # complement together. This means: find a bipartition (X, Y) where
        # X ⊂ side and Y ⊃ complement
        side_set = set(side)
        best_sub = 1.0
        for i, (Ai, Bi) in enumerate(bipartitions):
            Ai_set = set(Ai)
            Bi_set = set(Bi)
            # Check if this bipartition is a refinement of 'side'
            if Ai_set < side_set and Bi_set > (set(range(n_qubits)) - side_set):
                best_sub = min(best_sub, all_S[i]['S_frac'])
            elif Bi_set < side_set and Ai_set > (set(range(n_qubits)) - side_set):
                best_sub = min(best_sub, all_S[i]['S_frac'])
        if best_sub < threshold:
            n_sub += 1  # this half further splits
    
    return n_sub, {'A': A, 'B': B, 'S_frac': best['S_frac']}


def run_scenario(name, psi0, H_evolve, n_qubits, bipartitions, times, threshold=0.3):
    """Run one scenario: evolve psi0 under H_evolve, track fission."""
    D = 2 ** n_qubits
    n_bip = len(bipartitions)

    # Diagonalize evolution Hamiltonian
    evals, evecs = eigh(H_evolve)
    psi_eig = evecs.conj().T @ psi0

    # Storage
    min_S_frac_series = []
    min_S_series = []
    min_bip_series = []
    subsystem_count_series = []
    n_subsystems_series = []
    mean_S_frac_series = []

    t_start = time.time()
    for ti, t in enumerate(times):
        # Evolve
        phases = np.exp(-1j * evals * t)
        psi_t = evecs @ (phases * psi_eig)

        # All bipartition entropies
        all_S = compute_all_entropies(psi_t, bipartitions, n_qubits)

        # Find minimum
        min_idx = min(range(n_bip), key=lambda i: all_S[i]['S_frac'])
        min_S_frac = all_S[min_idx]['S_frac']
        min_S = all_S[min_idx]['S']
        min_bip = min_idx

        # Count low-entropy bipartitions
        n_low = subsystem_count(all_S, threshold)

        # Subsystem count from bipartition entropies
        n_sub, _ = count_subsystems(all_S, bipartitions, n_qubits, threshold)

        min_S_frac_series.append(float(min_S_frac))
        min_S_series.append(float(min_S))
        min_bip_series.append(int(min_bip))
        subsystem_count_series.append(int(n_low))
        n_subsystems_series.append(int(n_sub))
        mean_S_frac_series.append(float(np.mean([e['S_frac'] for e in all_S])))

        if ti % 25 == 0 or ti == len(times) - 1:
            A, B = bipartitions[min_bip]
            elapsed = time.time() - t_start
            print(f"  t={t:6.3f}: min S/S_max={min_S_frac:.4f} "
                  f"at ({A}|{B})  "
                  f"n_sub={n_sub}  n_low={n_low}  "
                  f"mean={mean_S_frac_series[-1]:.4f}  [{elapsed:.1f}s]")

    # Initial state diagnostics
    init_S = compute_all_entropies(psi0, bipartitions, n_qubits)
    init_min_frac = min(e['S_frac'] for e in init_S)
    init_mean_frac = np.mean([e['S_frac'] for e in init_S])

    result = {
        'name': name,
        'n_qubits': n_qubits,
        'n_bipartitions': n_bip,
        'threshold': threshold,
        'initial_state': {
            'min_S_frac': float(init_min_frac),
            'mean_S_frac': float(init_mean_frac),
            'n_subsystems': count_subsystems(init_S, bipartitions, n_qubits, threshold)[0],
        },
        'timeseries': {
            'times': [float(t) for t in times],
            'min_S_frac': min_S_frac_series,
            'min_S': min_S_series,
            'min_bipartition_idx': min_bip_series,
            'n_low_entropy_bipartitions': subsystem_count_series,
            'n_subsystems': n_subsystems_series,
            'mean_S_frac': mean_S_frac_series,
        },
    }

    # Summary
    ever_fissioned = any(n > 1 for n in n_subsystems_series[1:])  # skip t=0
    max_subsystems = max(n_subsystems_series)
    min_ever = min(min_S_frac_series[1:]) if len(min_S_frac_series) > 1 else min_S_frac_series[0]

    result['summary'] = {
        'ever_fissioned': ever_fissioned,
        'max_subsystems': max_subsystems,
        'min_S_frac_ever': float(min_ever),
        'mean_n_subsystems': float(np.mean(n_subsystems_series)),
        'fission_fraction': float(np.mean([1 if n > 1 else 0
                                            for n in n_subsystems_series[1:]])),
    }

    return result


def print_summary(result):
    """Print human-readable summary."""
    s = result['summary']
    i = result['initial_state']
    print(f"\n  {'─'*50}")
    print(f"  {result['name']}")
    print(f"  {'─'*50}")
    print(f"  Initial: {i['n_subsystems']} subsystem(s), "
          f"min S/S_max = {i['min_S_frac']:.4f}, "
          f"mean S/S_max = {i['mean_S_frac']:.4f}")
    print(f"  Ever fissioned? {'YES' if s['ever_fissioned'] else 'NO'}")
    print(f"  Max subsystems reached: {s['max_subsystems']}")
    print(f"  Mean subsystem count: {s['mean_n_subsystems']:.2f}")
    print(f"  Fraction of time fissioned: {s['fission_fraction']*100:.1f}%")
    print(f"  Lowest min(S/S_max) ever: {s['min_S_frac_ever']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Fission test: can one subsystem become two?')
    parser.add_argument('--nqubits', type=int, default=6)
    parser.add_argument('--ntimes', type=int, default=200,
                        help='Number of time steps')
    parser.add_argument('--tmax', type=float, default=15.0)
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='S/S_max below which we call it a split')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    N = args.nqubits
    D = 2 ** N

    print(f"FISSION TEST: CAN ONE SUBSYSTEM BECOME TWO?")
    print(f"{'='*60}")
    print(f"  N = {N} qubits, D = {D}")
    print(f"  Threshold for split: S/S_max < {args.threshold}")
    print(f"  Time steps: {args.ntimes}, t_max = {args.tmax}")

    # Enumerate all bipartitions
    bipartitions = all_bipartitions(N)
    print(f"  Bipartitions: {len(bipartitions)}")
    print()

    times = np.linspace(0, args.tmax, args.ntimes)

    # Build Hamiltonians
    print("Building Hamiltonians...")
    H_local = heisenberg_chain(N)
    H_alltoall = all_to_all_hamiltonian(N)
    H_random = random_gue(D, scale=np.linalg.norm(H_local, 'fro'))

    # Build initial states
    print("Building initial states...")
    psi_haar = haar_random_state(D)
    psi_gs_alltoall = ground_state(H_alltoall)
    psi_gs_local = ground_state(H_local)
    # Product state: alternating |010101...⟩
    idx = 0
    for q in range(N):
        idx |= (q % 2) << (N - 1 - q)
    psi_product = np.zeros(D, dtype=complex)
    psi_product[idx] = 1.0

    results = {}

    # === Scenario 1: Monolith → Local ===
    print(f"\n{'='*60}")
    print("SCENARIO 1: MONOLITH → LOCAL")
    print("  Haar random state → evolve under Heisenberg chain")
    print("  Can local dynamics crack a monolith?")
    print(f"{'='*60}")
    r1 = run_scenario("Haar random → Heisenberg",
                       psi_haar, H_local, N, bipartitions, times, args.threshold)
    print_summary(r1)
    results['monolith_to_local'] = r1

    # === Scenario 2: Quench ===
    print(f"\n{'='*60}")
    print("SCENARIO 2: QUENCH")
    print("  Ground state of all-to-all H → evolve under local H")
    print("  Does a deeply entangled blob develop spatial structure?")
    print(f"{'='*60}")
    r2 = run_scenario("All-to-all GS → Heisenberg quench",
                       psi_gs_alltoall, H_local, N, bipartitions, times, args.threshold)
    print_summary(r2)
    results['quench'] = r2

    # === Scenario 3: Fusion then re-fission? ===
    print(f"\n{'='*60}")
    print("SCENARIO 3: PRODUCT → LOCAL")
    print("  N independent qubits → evolve under Heisenberg")
    print("  They fuse. Do they ever re-fission?")
    print(f"{'='*60}")
    r3 = run_scenario("Product state → Heisenberg",
                       psi_product, H_local, N, bipartitions, times, args.threshold)
    print_summary(r3)
    results['product_to_local'] = r3

    # === Scenario 4: Structured → Random ===
    print(f"\n{'='*60}")
    print("SCENARIO 4: STRUCTURED → RANDOM")
    print("  Heisenberg ground state → evolve under random GUE")
    print("  Does existing structure dissolve? Reform differently?")
    print(f"{'='*60}")
    r4 = run_scenario("Heisenberg GS → GUE quench",
                       psi_gs_local, H_random, N, bipartitions, times, args.threshold)
    print_summary(r4)
    results['structured_to_random'] = r4

    # === VERDICT ===
    print(f"\n{'='*60}")
    print("VERDICT: DOES HILBERT SPACE ALLOW FISSION?")
    print(f"{'='*60}")
    print(f"\n  {'Scenario':<40} {'Fission?':>8} {'Max sub':>8} {'Time%':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for key, r in results.items():
        s = r['summary']
        tag = "YES" if s['ever_fissioned'] else "NO"
        print(f"  {r['name']:<40} {tag:>8} {s['max_subsystems']:>8} "
              f"{s['fission_fraction']*100:>7.1f}%")

    # Save
    os.makedirs('hsf_out', exist_ok=True)
    outfile = f'hsf_out/fission_test_n{N}.json'

    # Strip heavy timeseries for file size
    output = {
        'params': vars(args),
        'n_bipartitions': len(bipartitions),
        'bipartitions': [{'A': A, 'B': B} for A, B in bipartitions],
        'results': results,
    }
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {outfile}")


if __name__ == '__main__':
    main()