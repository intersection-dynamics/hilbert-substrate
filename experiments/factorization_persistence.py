#!/usr/bin/env python3
"""
factorization_persistence.py
=============================
HSF — Do Some Factorizations Survive the Boiling?

THE QUESTION:
  A bare Hilbert space of dimension D has no preferred factorization.
  Under generic unitary dynamics, do some factorizations show persistent
  low entanglement while others scramble to thermal entropy?

  If yes: which factorizations persist, and do they have HSF-constraint
  structure (locality, no-signaling boundaries, information preservation)?

METHOD:
  1. Fix a Hamiltonian H on a D-dimensional Hilbert space.
  2. Sample many candidate factorizations, each defined by a unitary
     basis change: H' = U H U†, then factorize as d_A ⊗ d_B.
  3. For each factorization, evolve a set of product initial states
     and track the entanglement entropy S(ρ_A)(t).
  4. Measure:
     - Entropy growth rate (how fast the factorization scrambles)
     - Locality cost (how "local" H looks in that factorization)
     - Peak entropy (how close to maximally mixed)
  5. Correlate: do low-locality-cost factorizations show slow scrambling?

THE PREDICTION:
  If the "boiling pot" picture is correct:
  - Most factorizations scramble fast (entropy → max quickly)
  - A few factorizations show persistently low entropy
  - Those persistent factorizations are the ones where H is local
  - The correlation between locality cost and scrambling rate is STRONG

SYSTEM:
  D = 2^n qubits with a structured Hamiltonian that has a "natural"
  factorization (nearest-neighbor chain) buried under a random basis change.
  We then search through factorizations to see if the dynamics reveals
  the natural one, as if the Hilbert space is "discovering" its own structure.

DEPENDENCIES: numpy, scipy
RUN: python factorization_persistence.py
"""

import numpy as np
from scipy.linalg import expm, logm
import time
import json
import os


# ═══════════════════════════════════════════════════════════════════════
#  Hamiltonian construction
# ═══════════════════════════════════════════════════════════════════════

def pauli():
    """Standard Pauli matrices."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def heisenberg_chain(n_qubits, periodic=False):
    """
    Nearest-neighbor Heisenberg chain on n qubits.
    H = Σ_{<i,j>} (X_i X_j + Y_i Y_j + Z_i Z_j)
    """
    D = 2 ** n_qubits
    H = np.zeros((D, D), dtype=complex)
    I, X, Y, Z = pauli()
    
    pairs = list(zip(range(n_qubits - 1), range(1, n_qubits)))
    if periodic and n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    
    for i, j in pairs:
        for P in [X, Y, Z]:
            term = np.eye(1, dtype=complex)
            for k in range(n_qubits):
                if k == i or k == j:
                    term = np.kron(term, P)
                else:
                    term = np.kron(term, I)
            H += term
    
    return H


def random_hamiltonian(D, sparsity=1.0):
    """Random Hermitian matrix (GUE-like)."""
    H = np.random.randn(D, D) + 1j * np.random.randn(D, D)
    H = (H + H.conj().T) / 2
    return H


# ═══════════════════════════════════════════════════════════════════════
#  Factorization tools
# ═══════════════════════════════════════════════════════════════════════

def random_unitary(D):
    """Haar-random unitary via QR decomposition."""
    Z = (np.random.randn(D, D) + 1j * np.random.randn(D, D)) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    # Fix phases to make it Haar
    d = np.diag(R)
    ph = d / np.abs(d)
    Q = Q @ np.diag(ph)
    return Q


def entanglement_entropy(psi, d_A, d_B):
    """
    Von Neumann entropy of subsystem A for a pure state |psi⟩
    in the factorization H = C^{d_A} ⊗ C^{d_B}.
    """
    psi_mat = psi.reshape(d_A, d_B)
    sv = np.linalg.svd(psi_mat, compute_uv=False)
    sv = sv[sv > 1e-15]
    probs = sv ** 2
    probs = probs / probs.sum()  # normalize
    return -np.sum(probs * np.log(probs + 1e-30))


def max_entropy(d_A, d_B):
    """Maximum entanglement entropy for the smaller subsystem."""
    return np.log(min(d_A, d_B))


# ═══════════════════════════════════════════════════════════════════════
#  Locality cost of H in a given factorization
# ═══════════════════════════════════════════════════════════════════════

def locality_cost(H, d_A, d_B):
    """
    Measure how "local" H looks in the factorization d_A ⊗ d_B.
    
    Decompose H = Σ_{a,b} c_{ab} (A_a ⊗ B_b) where {A_a} and {B_b}
    are orthonormal operator bases for the two subsystems.
    
    Locality cost = fraction of H's Frobenius norm carried by
    non-product terms (interaction terms where both a≠identity AND b≠identity).
    
    Returns value in [0, 1]:
      0 = perfectly local (H = H_A ⊗ I + I ⊗ H_B)
      1 = maximally nonlocal (all weight on interaction terms)
    """
    D = d_A * d_B
    
    # Reshape H as a d_A² × d_B² matrix of coefficients
    # H_{(i,j),(k,l)} = H_{i*d_B+j, k*d_B+l}
    # Treat as operator on A⊗B, decompose via partial traces
    
    H_sq_total = np.trace(H @ H.conj().T).real
    if H_sq_total < 1e-20:
        return 0.0
    
    # H_A = Tr_B(H) / d_B  (reduced Hamiltonian on A)
    H_mat = H.reshape(d_A, d_B, d_A, d_B)
    H_A = np.trace(H_mat, axis1=1, axis2=3) / d_B  # shape (d_A, d_A)
    H_B = np.trace(H_mat, axis1=0, axis2=2) / d_A  # shape (d_B, d_B)
    
    # Local part: H_local = H_A ⊗ I_B + I_A ⊗ H_B
    H_local = (np.kron(H_A, np.eye(d_B)) + np.kron(np.eye(d_A), H_B))
    
    H_interaction = H - H_local
    interaction_sq = np.trace(H_interaction @ H_interaction.conj().T).real
    
    return interaction_sq / H_sq_total


# ═══════════════════════════════════════════════════════════════════════
#  Core experiment: measure persistence of a factorization
# ═══════════════════════════════════════════════════════════════════════

def measure_factorization(H, U_fact, d_A, d_B, n_times=50, dt=0.1,
                           n_initial_states=10):
    """
    Measure the persistence of a factorization defined by basis change U_fact.
    
    In the rotated basis (U_fact), the Hilbert space is read as C^{d_A} ⊗ C^{d_B}.
    We evolve product states (in this factorization) under H and track entropy.
    
    Returns:
      - entropy trajectory (averaged over initial states)
      - scrambling rate (initial slope of entropy growth)
      - time-averaged entropy
      - locality cost of H in this factorization
    """
    D = d_A * d_B
    
    # H in the rotated basis
    H_rot = U_fact.conj().T @ H @ U_fact
    
    # Locality cost
    loc_cost = locality_cost(H_rot, d_A, d_B)
    
    # Time evolution operator
    times = np.arange(n_times) * dt
    
    # Generate product initial states in the rotated basis
    entropy_curves = []
    
    for _ in range(n_initial_states):
        # Random product state in the factorization basis
        psi_A = np.random.randn(d_A) + 1j * np.random.randn(d_A)
        psi_A /= np.linalg.norm(psi_A)
        psi_B = np.random.randn(d_B) + 1j * np.random.randn(d_B)
        psi_B /= np.linalg.norm(psi_B)
        psi_product = np.kron(psi_A, psi_B)
        
        # Transform to the computational basis for evolution
        psi = U_fact @ psi_product
        
        # Evolve and measure entropy at each time step
        S_t = np.zeros(n_times)
        for ti, t in enumerate(times):
            if ti == 0:
                psi_t = psi.copy()
            else:
                # Incremental evolution
                psi_t = expm(-1j * H * dt) @ psi_t
                psi_t /= np.linalg.norm(psi_t)
            
            # Transform back to factorization basis to measure entropy
            psi_fact = U_fact.conj().T @ psi_t
            S_t[ti] = entanglement_entropy(psi_fact, d_A, d_B)
        
        entropy_curves.append(S_t)
    
    entropy_avg = np.mean(entropy_curves, axis=0)
    S_max = max_entropy(d_A, d_B)
    
    # Scrambling rate: slope of entropy in the early linear regime
    # Use first 20% of time points
    n_early = max(3, n_times // 5)
    if entropy_avg[n_early] > entropy_avg[0]:
        scrambling_rate = (entropy_avg[n_early] - entropy_avg[0]) / (times[n_early] - times[0])
    else:
        scrambling_rate = 0.0
    
    # Time-averaged entropy (normalized by maximum)
    avg_entropy_normalized = np.mean(entropy_avg) / S_max if S_max > 0 else 0
    
    # Late-time entropy (normalized)
    late_entropy = np.mean(entropy_avg[-n_early:]) / S_max if S_max > 0 else 0
    
    return {
        'locality_cost': float(loc_cost),
        'scrambling_rate': float(scrambling_rate),
        'avg_entropy_norm': float(avg_entropy_normalized),
        'late_entropy_norm': float(late_entropy),
        'entropy_curve': [float(x) for x in entropy_avg],
        'max_entropy': float(S_max),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 1: Hidden structure discovery
# ═══════════════════════════════════════════════════════════════════════

def experiment_hidden_structure(n_qubits=4, n_factorizations=200,
                                n_times=60, dt=0.15):
    """
    A Heisenberg chain has a natural spatial factorization. We hide it
    under a random basis change, then search through factorizations to
    see if the dynamics reveals the natural one.
    
    Like the Hilbert space "discovering" its own structure through the
    boiling process.
    """
    D = 2 ** n_qubits
    d_A = 2 ** (n_qubits // 2)
    d_B = D // d_A
    
    print(f"\n{'═' * 70}")
    print(f"  EXPERIMENT 1: Hidden Structure Discovery")
    print(f"  {n_qubits} qubits, D={D}, factorization {d_A}×{d_B}")
    print(f"  {n_factorizations} candidate factorizations")
    print(f"{'═' * 70}")
    
    # Build a Heisenberg chain and scramble it
    H_spatial = heisenberg_chain(n_qubits, periodic=True)
    U_scramble = random_unitary(D)
    H = U_scramble @ H_spatial @ U_scramble.conj().T
    
    # The "natural" factorization is U_scramble applied to the
    # standard qubit factorization (first n/2 qubits ⊗ last n/2 qubits)
    
    results = []
    
    # Test the NATURAL factorization (the one H was built in)
    print(f"  Testing natural factorization...")
    r_natural = measure_factorization(H, U_scramble, d_A, d_B,
                                       n_times=n_times, dt=dt)
    r_natural['type'] = 'natural'
    results.append(r_natural)
    print(f"    Locality cost: {r_natural['locality_cost']:.4f}")
    print(f"    Scrambling rate: {r_natural['scrambling_rate']:.4f}")
    print(f"    Late entropy (norm): {r_natural['late_entropy_norm']:.4f}")
    
    # Test the IDENTITY factorization (random, unrelated to H)
    print(f"  Testing identity (computational) basis factorization...")
    r_comp = measure_factorization(H, np.eye(D), d_A, d_B,
                                    n_times=n_times, dt=dt)
    r_comp['type'] = 'computational'
    results.append(r_comp)
    print(f"    Locality cost: {r_comp['locality_cost']:.4f}")
    print(f"    Scrambling rate: {r_comp['scrambling_rate']:.4f}")
    print(f"    Late entropy (norm): {r_comp['late_entropy_norm']:.4f}")
    
    # Test many RANDOM factorizations
    print(f"  Testing {n_factorizations} random factorizations...")
    t0 = time.time()
    for i in range(n_factorizations):
        U_rand = random_unitary(D)
        r = measure_factorization(H, U_rand, d_A, d_B,
                                   n_times=n_times, dt=dt,
                                   n_initial_states=5)
        r['type'] = 'random'
        results.append(r)
        
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_factorizations} done ({time.time()-t0:.1f}s)")
    
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 2: Structured vs random Hamiltonian
# ═══════════════════════════════════════════════════════════════════════

def experiment_structured_vs_random(n_qubits=4, n_factorizations=200,
                                     n_times=60, dt=0.15):
    """
    Compare factorization persistence under:
    (a) A structured Hamiltonian (Heisenberg chain) — should have persistent factorizations
    (b) A random Hamiltonian (GUE) — should scramble ALL factorizations equally
    
    If the boiling picture is correct: structured H creates "islands of calm"
    in factorization space, while random H is uniformly turbulent.
    """
    D = 2 ** n_qubits
    d_A = 2 ** (n_qubits // 2)
    d_B = D // d_A
    
    print(f"\n{'═' * 70}")
    print(f"  EXPERIMENT 2: Structured vs Random Hamiltonian")
    print(f"  {n_qubits} qubits, D={D}, factorization {d_A}×{d_B}")
    print(f"{'═' * 70}")
    
    all_results = {}
    
    for h_type, H in [('structured', heisenberg_chain(n_qubits, periodic=True)),
                       ('random', random_hamiltonian(D))]:
        
        # Normalize so both have similar spectral range
        evals = np.linalg.eigvalsh(H)
        H = H / (evals[-1] - evals[0]) * 4 * n_qubits
        
        print(f"\n  --- {h_type.upper()} Hamiltonian ---")
        
        locs = []
        rates = []
        lates = []
        
        for i in range(n_factorizations):
            U = random_unitary(D)
            r = measure_factorization(H, U, d_A, d_B,
                                       n_times=n_times, dt=dt,
                                       n_initial_states=5)
            locs.append(r['locality_cost'])
            rates.append(r['scrambling_rate'])
            lates.append(r['late_entropy_norm'])
        
        locs = np.array(locs)
        rates = np.array(rates)
        lates = np.array(lates)
        
        # Correlation between locality cost and scrambling
        corr_rate = np.corrcoef(locs, rates)[0, 1]
        corr_late = np.corrcoef(locs, lates)[0, 1]
        
        # Distribution of locality costs
        print(f"    Locality cost:  mean={locs.mean():.4f}  std={locs.std():.4f}"
              f"  min={locs.min():.4f}  max={locs.max():.4f}")
        print(f"    Scrambling rate: mean={rates.mean():.4f}  std={rates.std():.4f}")
        print(f"    Late entropy:   mean={lates.mean():.4f}  std={lates.std():.4f}")
        print(f"    Corr(locality, scrambling rate):  r = {corr_rate:+.4f}")
        print(f"    Corr(locality, late entropy):     r = {corr_late:+.4f}")
        
        # Is there a "persistent" tail?
        threshold = np.percentile(lates, 10)
        n_persistent = int(np.sum(lates < threshold))
        persistent_loc = locs[lates < threshold].mean()
        bulk_loc = locs[lates >= threshold].mean()
        
        print(f"    Persistent tail (bottom 10%): avg locality = {persistent_loc:.4f}")
        print(f"    Bulk (top 90%):              avg locality = {bulk_loc:.4f}")
        
        all_results[h_type] = {
            'locality_costs': [float(x) for x in locs],
            'scrambling_rates': [float(x) for x in rates],
            'late_entropies': [float(x) for x in lates],
            'corr_locality_scrambling': float(corr_rate),
            'corr_locality_late_entropy': float(corr_late),
            'persistent_avg_locality': float(persistent_loc),
            'bulk_avg_locality': float(bulk_loc),
        }
    
    return all_results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 3: Persistence as a function of constraint satisfaction
# ═══════════════════════════════════════════════════════════════════════

def experiment_constraint_correlation(n_qubits=4, n_factorizations=300,
                                       n_times=60, dt=0.15):
    """
    For each candidate factorization, measure MULTIPLE HSF-relevant
    quantities and check which predict persistence:
    
    1. Locality cost (how local H looks) — proxy for no-signaling
    2. Entropy production rate — proxy for information scrambling
    3. Mutual information persistence — proxy for no-forgetting
    4. Spectral gap of reduced dynamics — proxy for finite bandwidth
    """
    D = 2 ** n_qubits
    d_A = 2 ** (n_qubits // 2)
    d_B = D // d_A
    
    print(f"\n{'═' * 70}")
    print(f"  EXPERIMENT 3: Constraint Correlation Analysis")
    print(f"  Which properties predict factorization persistence?")
    print(f"  {n_qubits} qubits, D={D}, {n_factorizations} factorizations")
    print(f"{'═' * 70}")
    
    H = heisenberg_chain(n_qubits, periodic=True)
    U_scramble = random_unitary(D)
    H = U_scramble @ H @ U_scramble.conj().T
    
    data = {
        'locality_cost': [],
        'scrambling_rate': [],
        'late_entropy': [],
        'mutual_info_persistence': [],
        'spectral_structure': [],
    }
    
    print(f"  Sampling factorizations...")
    t0 = time.time()
    
    # Always include the natural factorization
    test_unitaries = [U_scramble]  # index 0 = natural
    for _ in range(n_factorizations - 1):
        test_unitaries.append(random_unitary(D))
    
    for idx, U in enumerate(test_unitaries):
        H_rot = U.conj().T @ H @ U
        
        # 1. Locality cost
        loc = locality_cost(H_rot, d_A, d_B)
        data['locality_cost'].append(loc)
        
        # 2-3. Entropy evolution and scrambling
        r = measure_factorization(H, U, d_A, d_B,
                                   n_times=n_times, dt=dt,
                                   n_initial_states=8)
        data['scrambling_rate'].append(r['scrambling_rate'])
        data['late_entropy'].append(r['late_entropy_norm'])
        
        # 4. Mutual information persistence
        # Measure MI at early and late times
        times = np.arange(n_times) * dt
        psi_A = np.random.randn(d_A) + 1j * np.random.randn(d_A)
        psi_A /= np.linalg.norm(psi_A)
        psi_B = np.random.randn(d_B) + 1j * np.random.randn(d_B)
        psi_B /= np.linalg.norm(psi_B)
        psi = U @ np.kron(psi_A, psi_B)
        
        U_dt = expm(-1j * H * dt)
        psi_t = psi.copy()
        S_early = 0.0
        S_late = 0.0
        for ti in range(n_times):
            if ti > 0:
                psi_t = U_dt @ psi_t
                psi_t /= np.linalg.norm(psi_t)
            psi_f = U.conj().T @ psi_t
            S = entanglement_entropy(psi_f, d_A, d_B)
            if ti == n_times // 5:
                S_early = S
            if ti == n_times - 1:
                S_late = S
        
        # MI persistence = how much the entropy stays low relative to max
        S_m = max_entropy(d_A, d_B)
        mi_persist = 1.0 - S_late / S_m if S_m > 0 else 0
        data['mutual_info_persistence'].append(float(mi_persist))
        
        # 5. Spectral structure: how gapped is the interaction part?
        H_mat = H_rot.reshape(d_A, d_B, d_A, d_B)
        H_A = np.trace(H_mat, axis1=1, axis2=3) / d_B
        H_B = np.trace(H_mat, axis1=0, axis2=2) / d_A
        H_int = H_rot - np.kron(H_A, np.eye(d_B)) - np.kron(np.eye(d_A), H_B)
        int_norm = np.linalg.norm(H_int, 'fro')
        full_norm = np.linalg.norm(H_rot, 'fro')
        spectral_ratio = int_norm / full_norm if full_norm > 0 else 0
        data['spectral_structure'].append(float(spectral_ratio))
        
        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{n_factorizations} done ({time.time()-t0:.1f}s)")
    
    # Analysis
    print(f"\n  Analysis:")
    
    locs = np.array(data['locality_cost'])
    rates = np.array(data['scrambling_rate'])
    lates = np.array(data['late_entropy'])
    mi_p = np.array(data['mutual_info_persistence'])
    spec = np.array(data['spectral_structure'])
    
    predictors = [
        ('Locality cost', locs),
        ('Spectral ratio', spec),
    ]
    
    outcomes = [
        ('Scrambling rate', rates),
        ('Late entropy', lates),
        ('MI persistence', mi_p),
    ]
    
    print(f"\n  Correlation matrix (predictors → persistence measures):")
    print(f"  {'':>20}", end='')
    for name, _ in outcomes:
        print(f"  {name:>16}", end='')
    print()
    
    corr_results = {}
    for p_name, p_data in predictors:
        print(f"  {p_name:>20}", end='')
        for o_name, o_data in outcomes:
            r = np.corrcoef(p_data, o_data)[0, 1]
            print(f"  {r:>+16.4f}", end='')
            corr_results[f'{p_name}_vs_{o_name}'] = float(r)
        print()
    
    # Natural factorization vs bulk
    print(f"\n  Natural factorization (index 0) vs bulk:")
    print(f"    Locality cost: {locs[0]:.4f}  (bulk mean: {locs[1:].mean():.4f},"
          f" percentile: {(locs[1:] < locs[0]).mean()*100:.1f}%)")
    print(f"    Late entropy:  {lates[0]:.4f}  (bulk mean: {lates[1:].mean():.4f},"
          f" percentile: {(lates[1:] < lates[0]).mean()*100:.1f}%)")
    print(f"    MI persistence: {mi_p[0]:.4f}  (bulk mean: {mi_p[1:].mean():.4f},"
          f" percentile: {(mi_p[1:] > mi_p[0]).mean()*100:.1f}%)")
    
    data['correlations'] = corr_results
    data['natural_index'] = 0
    
    return data


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='HSF Factorization Persistence')
    parser.add_argument('--nqubits', type=int, default=5, help='Number of qubits (default: 5)')
    parser.add_argument('--nfact', type=int, default=150, help='Number of random factorizations (default: 150)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    np.random.seed(args.seed)
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  HSF: Factorization Persistence in a Boiling Hilbert Space         ║")
    print("║  Do some factorizations survive generic dynamics?                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    n_qubits = args.nqubits
    n_fact = args.nfact
    n_t = 50
    dt_val = 0.12
    
    # ─── Experiment 1 ──────────────────────────────────────────────
    results_1 = experiment_hidden_structure(
        n_qubits=n_qubits, n_factorizations=n_fact, n_times=n_t, dt=dt_val)
    
    # Quick analysis
    natural = results_1[0]
    comp = results_1[1]
    randoms = results_1[2:]
    
    random_locs = [r['locality_cost'] for r in randoms]
    random_lates = [r['late_entropy_norm'] for r in randoms]
    
    print(f"\n  ── Experiment 1 Summary ──")
    print(f"  Natural factorization:")
    print(f"    Locality cost:    {natural['locality_cost']:.4f}"
          f"  (random mean: {np.mean(random_locs):.4f})")
    print(f"    Late entropy:     {natural['late_entropy_norm']:.4f}"
          f"  (random mean: {np.mean(random_lates):.4f})")
    print(f"  Computational basis (unrelated):")
    print(f"    Locality cost:    {comp['locality_cost']:.4f}")
    print(f"    Late entropy:     {comp['late_entropy_norm']:.4f}")
    
    nat_percentile = np.mean([l > natural['late_entropy_norm'] for l in random_lates]) * 100
    print(f"  Natural factorization has lower entropy than {nat_percentile:.1f}% of random")
    
    corr = np.corrcoef(random_locs, random_lates)[0, 1]
    print(f"  Correlation(locality cost, late entropy): r = {corr:+.4f}")
    
    # ─── Experiment 2 ──────────────────────────────────────────────
    results_2 = experiment_structured_vs_random(
        n_qubits=n_qubits, n_factorizations=n_fact, n_times=n_t, dt=dt_val)
    
    # ─── Experiment 3 ──────────────────────────────────────────────
    results_3 = experiment_constraint_correlation(
        n_qubits=n_qubits, n_factorizations=n_fact, n_times=n_t, dt=dt_val)
    
    # ─── Final verdict ─────────────────────────────────────────────
    elapsed = time.time() - t_start
    
    print(f"\n\n{'═' * 70}")
    print(f"  VERDICT: DOES THE BOILING POT HAVE CALM SPOTS?")
    print(f"{'═' * 70}")
    
    # Key metrics
    struct_corr = results_2['structured']['corr_locality_late_entropy']
    random_corr = results_2['random']['corr_locality_late_entropy']
    struct_spread = np.std(results_2['structured']['late_entropies'])
    random_spread = np.std(results_2['random']['late_entropies'])
    
    print(f"\n  1. Does locality predict persistence?")
    print(f"     Structured H: r(locality, late entropy) = {struct_corr:+.4f}"
          f"  {'YES' if abs(struct_corr) > 0.3 else 'WEAK'}")
    print(f"     Random H:     r(locality, late entropy) = {random_corr:+.4f}"
          f"  {'YES' if abs(random_corr) > 0.3 else 'NO (expected)'}")
    
    print(f"\n  2. Is there spread in persistence (some survive, others don't)?")
    print(f"     Structured H: std(late entropy) = {struct_spread:.4f}"
          f"  {'YES — spread exists' if struct_spread > 0.05 else 'NO — uniform scrambling'}")
    print(f"     Random H:     std(late entropy) = {random_spread:.4f}"
          f"  {'YES' if random_spread > 0.05 else 'NO — uniform (expected)'}")
    
    print(f"\n  3. Can dynamics discover the natural factorization?")
    print(f"     Natural factorization at percentile {nat_percentile:.1f}% of late entropy")
    print(f"     {'YES — natural factorization is distinctly persistent' if nat_percentile > 80 else 'UNCLEAR'}")
    
    print(f"\n  Total runtime: {elapsed:.1f}s")
    
    # Save
    os.makedirs('hsf_out', exist_ok=True)
    output = {
        'experiment_1_hidden_structure': {
            'natural': natural,
            'computational': comp,
            'n_random': len(randoms),
            'random_locality_mean': float(np.mean(random_locs)),
            'random_late_entropy_mean': float(np.mean(random_lates)),
            'natural_percentile': float(nat_percentile),
            'correlation': float(corr),
        },
        'experiment_2_structured_vs_random': results_2,
        'experiment_3_constraints': {
            'correlations': results_3['correlations'],
            'natural_locality': float(results_3['locality_cost'][0]),
            'natural_late_entropy': float(results_3['late_entropy'][0]),
        },
        'runtime': elapsed,
    }
    
    outpath = f'hsf_out/factorization_persistence_n{n_qubits}.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()