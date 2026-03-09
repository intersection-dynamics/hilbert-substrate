"""
Factorization Churn Test
========================
Does a single undivided system ever spontaneously split into two?

Start with a Haar-random pure state — no preferred factorization,
near-maximal entropy in every bipartition. Evolve under H.

At each timestep, scan over factorizations and ask:
  which bipartition has minimum entanglement entropy?

If a factorization "crystallizes," its entropy dips well below the
background. Track the winner's identity over time to see churn.

Two Hamiltonians compared:
  (A) Heisenberg chain — has a natural factorization (qubit tensor product)
  (B) Random GUE — no preferred factorization

Prediction if boiling-pot picture is correct:
  - Under structured H: natural factorization wins disproportionately
  - Under random H: no factorization is preferred, pure noise
  - Winner identity changes over time (churn), but returns to natural
"""

import numpy as np
from scipy.linalg import eigh
import json
import argparse
import time
import os


def random_pure_state(D):
    """Haar-random pure state."""
    psi = np.random.randn(D) + 1j * np.random.randn(D)
    return psi / np.linalg.norm(psi)


def heisenberg_chain(n_qubits):
    """Heisenberg XXX chain with periodic boundary conditions."""
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


def random_gue(D):
    """Random GUE Hamiltonian (no preferred factorization)."""
    A = np.random.randn(D, D) + 1j * np.random.randn(D, D)
    H = (A + A.conj().T) / 2
    # Normalize so spectral width ~ same as Heisenberg
    H *= np.sqrt(D) / np.linalg.norm(H, 'fro') * D
    return H


def random_unitary(D):
    """Haar-random unitary (defines a factorization)."""
    Z = np.random.randn(D, D) + 1j * np.random.randn(D, D)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph[np.newaxis, :]


def entanglement_entropy(psi_rot, d_A, d_B):
    """Von Neumann entropy of subsystem A given rotated state vector."""
    psi_mat = psi_rot.reshape(d_A, d_B)
    s = np.linalg.svd(psi_mat, compute_uv=False)
    s = s[s > 1e-15]
    s2 = s ** 2
    return -np.sum(s2 * np.log2(s2 + 1e-30))


def run_experiment(H, factorization_bases, d_A, d_B, psi0, times, label):
    """
    Evolve psi0 under H, compute entanglement entropy under each
    factorization at each timestep.
    
    factorization_bases[i] = U_i^dagger  (pre-transposed for speed)
    factorization 0 = natural (identity)
    """
    D = len(psi0)
    n_fact = len(factorization_bases)
    n_times = len(times)

    # Diagonalize H once
    evals, evecs = eigh(H)
    # psi in eigenbasis
    psi_eig = evecs.conj().T @ psi0

    # Pre-compute factorization rotations in eigenbasis
    # For factorization i: rotated state = U_i^dag @ psi
    # = U_i^dag @ evecs @ diag(e^{-iEt}) @ evecs^dag @ psi0
    # = (U_i^dag @ evecs) @ (e^{-iEt} * psi_eig)
    # Pre-compute M_i = U_i^dag @ evecs
    print(f"  Pre-computing {n_fact} rotation matrices...")
    M = np.array([Udag @ evecs for Udag in factorization_bases])  # (n_fact, D, D)

    # Storage
    all_entropies = np.zeros((n_times, n_fact))

    t_start = time.time()
    for ti, t in enumerate(times):
        # Evolve: psi_eig(t) = e^{-iEt} * psi_eig(0)
        phases = np.exp(-1j * evals * t)
        psi_eig_t = phases * psi_eig  # element-wise

        # Entropy for each factorization
        for fi in range(n_fact):
            psi_rot = M[fi] @ psi_eig_t
            all_entropies[ti, fi] = entanglement_entropy(psi_rot, d_A, d_B)

        if (ti + 1) % 20 == 0 or ti == 0 or ti == n_times - 1:
            winner = np.argmin(all_entropies[ti])
            elapsed = time.time() - t_start
            print(f"  [{label}] t={t:.3f}: winner=fact{winner} "
                  f"S_min={all_entropies[ti, winner]:.4f} "
                  f"S_nat={all_entropies[ti, 0]:.4f} "
                  f"S_mean={np.mean(all_entropies[ti, 1:]):.4f} "
                  f"[{elapsed:.1f}s]")

    return all_entropies


def analyze_churn(all_entropies, times, n_random, S_max, label):
    """Analyze factorization churn from entropy timeseries."""
    n_times, n_fact = all_entropies.shape

    winners = np.argmin(all_entropies, axis=1)
    natural_entropies = all_entropies[:, 0]
    random_entropies = all_entropies[:, 1:]

    # --- Churn statistics ---
    unique_winners = set(winners.tolist())
    natural_wins = np.sum(winners == 0)
    transitions = np.sum(np.diff(winners) != 0)

    # Reign lengths
    reigns = []
    current = winners[0]
    length = 1
    for w in winners[1:]:
        if w == current:
            length += 1
        else:
            reigns.append((int(current), length))
            current = w
            length = 1
    reigns.append((int(current), length))

    avg_reign = np.mean([r[1] for r in reigns])
    max_reign = max(reigns, key=lambda r: r[1])

    # --- Crystallization depth ---
    # How many sigma below mean is the minimum at each timestep?
    random_means = np.mean(random_entropies, axis=1)
    random_stds = np.std(random_entropies, axis=1)
    min_per_step = np.min(all_entropies, axis=1)
    # Sigma below mean (positive = below mean)
    crystallization_depth = (random_means - min_per_step) / (random_stds + 1e-15)

    # Natural factorization depth
    natural_depth = (random_means - natural_entropies) / (random_stds + 1e-15)

    # Natural rank at each timestep (0 = best, n_random = worst)
    natural_ranks = np.sum(random_entropies < natural_entropies[:, None], axis=1)

    # --- Compile ---
    result = {
        'label': label,
        'n_timesteps': n_times,
        'n_factorizations': n_fact,
        'S_max': float(S_max),
        'churn': {
            'distinct_winners': len(unique_winners),
            'natural_wins': int(natural_wins),
            'natural_win_fraction': float(natural_wins / n_times),
            'expected_win_fraction': float(1.0 / n_fact),
            'transition_rate': float(transitions / (n_times - 1)),
            'avg_reign_length': float(avg_reign),
            'longest_reign': {
                'factorization': int(max_reign[0]),
                'length': int(max_reign[1]),
                'is_natural': max_reign[0] == 0,
            },
        },
        'crystallization': {
            'mean_depth_sigma': float(np.mean(crystallization_depth)),
            'max_depth_sigma': float(np.max(crystallization_depth)),
            'natural_mean_depth_sigma': float(np.mean(natural_depth)),
        },
        'entropy_stats': {
            'natural_mean': float(np.mean(natural_entropies)),
            'natural_std': float(np.std(natural_entropies)),
            'random_grand_mean': float(np.mean(random_entropies)),
            'random_grand_std': float(np.std(random_entropies.ravel())),
            'natural_rank_mean': float(np.mean(natural_ranks)),
            'natural_rank_percentile': float(np.mean(natural_ranks) / n_random * 100),
        },
        'timeseries': {
            'times': times.tolist(),
            'natural_entropy': natural_entropies.tolist(),
            'random_mean': random_means.tolist(),
            'random_min': np.min(random_entropies, axis=1).tolist(),
            'random_max': np.max(random_entropies, axis=1).tolist(),
            'winner_id': winners.tolist(),
            'winner_entropy': min_per_step.tolist(),
            'crystallization_depth': crystallization_depth.tolist(),
            'natural_depth': natural_depth.tolist(),
            'natural_rank': natural_ranks.tolist(),
        },
    }

    return result


def print_report(result):
    """Print human-readable report."""
    label = result['label']
    c = result['churn']
    cr = result['crystallization']
    es = result['entropy_stats']
    S_max = result['S_max']

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print(f"\n  CHURN:")
    print(f"    Distinct winners: {c['distinct_winners']} / {result['n_factorizations']}")
    print(f"    Natural wins: {c['natural_wins']}/{result['n_timesteps']} "
          f"= {c['natural_win_fraction']*100:.1f}%  "
          f"(chance = {c['expected_win_fraction']*100:.2f}%)")
    print(f"    Transition rate: {c['transition_rate']*100:.1f}% of steps")
    print(f"    Avg reign: {c['avg_reign_length']:.1f} steps")
    lr = c['longest_reign']
    nat_tag = " ← NATURAL" if lr['is_natural'] else ""
    print(f"    Longest reign: fact{lr['factorization']} "
          f"for {lr['length']} steps{nat_tag}")

    print(f"\n  CRYSTALLIZATION (sigma below random mean):")
    print(f"    Any factorization: mean={cr['mean_depth_sigma']:.2f}σ, "
          f"max={cr['max_depth_sigma']:.2f}σ")
    print(f"    Natural factorization: mean={cr['natural_mean_depth_sigma']:.2f}σ")

    print(f"\n  ENTROPY (S_max = {S_max:.3f}):")
    print(f"    Natural: {es['natural_mean']:.4f} ± {es['natural_std']:.4f} "
          f"({es['natural_mean']/S_max*100:.1f}% of max)")
    print(f"    Random:  {es['random_grand_mean']:.4f} ± {es['random_grand_std']:.4f} "
          f"({es['random_grand_mean']/S_max*100:.1f}% of max)")
    print(f"    Natural rank: {es['natural_rank_mean']:.1f} "
          f"(percentile {es['natural_rank_percentile']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Test factorization churn: does a monolith spontaneously split?')
    parser.add_argument('--nqubits', type=int, default=5)
    parser.add_argument('--nfact', type=int, default=200,
                        help='Number of random factorizations to sample')
    parser.add_argument('--ntimes', type=int, default=100,
                        help='Number of time steps')
    parser.add_argument('--tmax', type=float, default=10.0,
                        help='Maximum evolution time')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    N = args.nqubits
    D = 2 ** N

    # Bipartition: split qubits as evenly as possible
    n_A = N // 2
    n_B = N - n_A
    d_A = 2 ** n_A
    d_B = 2 ** n_B

    S_max = np.log2(min(d_A, d_B))

    print(f"FACTORIZATION CHURN TEST")
    print(f"{'='*70}")
    print(f"  N = {N} qubits, D = {D}")
    print(f"  Bipartition: {d_A} × {d_B} (S_max = {S_max:.3f} bits)")
    print(f"  Factorizations: 1 natural + {args.nfact} random")
    print(f"  Time steps: {args.ntimes}, t_max = {args.tmax}")
    print(f"  Seed: {args.seed}")
    print()

    # Time grid
    times = np.linspace(0, args.tmax, args.ntimes)

    # --- Generate factorizations ---
    # Natural = identity (standard qubit tensor product)
    # Random = Haar-random unitaries
    # Store as U^dagger for speed
    print("Generating factorizations...")
    fact_bases = [np.eye(D, dtype=complex)]  # natural
    for i in range(args.nfact):
        U = random_unitary(D)
        fact_bases.append(U.conj().T)  # store U^dag directly
    print(f"  Done: {len(fact_bases)} factorizations")

    # --- Initial state: Haar-random (no preferred factorization) ---
    psi0 = random_pure_state(D)

    # Verify initial state is unbiased
    init_entropies = np.array([
        entanglement_entropy(Udag @ psi0, d_A, d_B)
        for Udag in fact_bases
    ])
    print(f"\nInitial state entropy check:")
    print(f"  Natural: {init_entropies[0]:.4f}")
    print(f"  Random mean: {np.mean(init_entropies[1:]):.4f} "
          f"± {np.std(init_entropies[1:]):.4f}")
    print(f"  S_max: {S_max:.4f}")
    print(f"  Page value: ~{S_max - 1/(2*np.log(2)*d_A):.4f}")

    # === EXPERIMENT A: Structured Hamiltonian (Heisenberg) ===
    print(f"\n{'='*70}")
    print("EXPERIMENT A: Heisenberg chain (structured)")
    print(f"{'='*70}")
    H_struct = heisenberg_chain(N)
    S_struct = run_experiment(H_struct, fact_bases, d_A, d_B, psi0, times,
                              "Heisenberg")
    result_A = analyze_churn(S_struct, times, args.nfact, S_max, "Heisenberg chain")
    print_report(result_A)

    # === EXPERIMENT B: Random Hamiltonian (GUE) ===
    print(f"\n{'='*70}")
    print("EXPERIMENT B: Random GUE Hamiltonian (no structure)")
    print(f"{'='*70}")
    H_rand = random_gue(D)
    S_rand = run_experiment(H_rand, fact_bases, d_A, d_B, psi0, times,
                             "GUE")
    result_B = analyze_churn(S_rand, times, args.nfact, S_max, "Random GUE")
    print_report(result_B)

    # === EXPERIMENT C: Structured H + product initial state (natural basis) ===
    print(f"\n{'='*70}")
    print("EXPERIMENT C: Heisenberg + product state in NATURAL factorization")
    print(f"{'='*70}")
    # Product state: |01010...⟩ (alternating up/down) — has nontrivial dynamics
    # This is NOT an eigenstate of H, so it will evolve
    idx = 0
    for q in range(N):
        idx |= (q % 2) << (N - 1 - q)  # alternating 0,1,0,1,...
    psi0_product = np.zeros(D, dtype=complex)
    psi0_product[idx] = 1.0
    # Verify it's not an eigenstate
    Hpsi = H_struct @ psi0_product
    overlap = abs(np.vdot(psi0_product, Hpsi))
    norm_Hpsi = np.linalg.norm(Hpsi)
    print(f"  Product state |{''.join(str((q%2)) for q in range(N))}⟩")
    print(f"  ⟨ψ|H|ψ⟩ = {np.vdot(psi0_product, Hpsi).real:.4f}, "
          f"||H|ψ⟩|| = {norm_Hpsi:.4f}, "
          f"eigenstate? {'YES' if abs(norm_Hpsi - overlap) < 1e-10 else 'NO'}")
    S_prod = run_experiment(H_struct, fact_bases, d_A, d_B, psi0_product, times,
                             "Product-natural")
    result_C = analyze_churn(S_prod, times, args.nfact, S_max, "Product state (natural basis)")
    print_report(result_C)

    # === EXPERIMENT D: Structured H + product state in RANDOM factorization ===
    print(f"\n{'='*70}")
    print("EXPERIMENT D: Heisenberg + product state in RANDOM factorization")
    print(f"{'='*70}")
    # Product state in a random basis — zero entropy in some random factorization,
    # generic entropy in natural factorization
    # Use factorization #1 (the first random one)
    U_rand = fact_bases[1].conj().T  # undo the dagger to get U
    psi0_rand_product = U_rand[:, 0]  # first column = product state in that basis
    S_quench = run_experiment(H_struct, fact_bases, d_A, d_B, psi0_rand_product, times,
                               "Product-random")
    result_D = analyze_churn(S_quench, times, args.nfact, S_max,
                              "Product state (random basis) → Heisenberg quench")
    print_report(result_D)

    # === COMPARISON ===
    print(f"\n{'='*70}")
    print("COMPARISON: Does structure create calm spots?")
    print(f"{'='*70}")
    chance = result_A['churn']['expected_win_fraction']

    print(f"\n  {'Experiment':<45} {'Nat win%':>8} {'Nat depth':>10} {'Nat rank%':>10}")
    print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*10}")
    for label, r in [('A: Random ψ + Heisenberg H', result_A),
                     ('B: Random ψ + GUE H', result_B),
                     ('C: Product(natural) ψ + Heisenberg H', result_C),
                     ('D: Product(random) ψ + Heisenberg H', result_D)]:
        wf = r['churn']['natural_win_fraction']
        nd = r['crystallization']['natural_mean_depth_sigma']
        nr = r['entropy_stats']['natural_rank_percentile']
        print(f"  {label:<45} {wf*100:>7.1f}% {nd:>9.2f}σ {nr:>9.1f}%")
    print(f"  {'Chance':<45} {chance*100:>7.2f}%")

    # --- Save ---
    os.makedirs('hsf_out', exist_ok=True)
    outfile = f'hsf_out/factorization_churn_n{N}.json'
    output = {
        'params': vars(args),
        'd_A': d_A, 'd_B': d_B, 'D': D, 'S_max': S_max,
        'experiment_A': result_A,
        'experiment_B': result_B,
        'experiment_C': result_C,
        'experiment_D': result_D,
    }
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {outfile}")


if __name__ == '__main__':
    main()