"""
Force Emergence from Accessibility Collapse
=============================================

After the double bracket flow traps the system in the spatial basin,
decompose the recovered Hamiltonian in the Pauli basis and catalog
the interaction terms by weight and operator content.

Key question: does the interaction structure depend on the starting
Hamiltonian, or does accessibility collapse produce a universal 
force structure?

Usage:
    python force_emergence.py --quick       # N=4, fast test
    python force_emergence.py --standard    # N=5, main result
    python force_emergence.py --full        # N=5,6 multiple seeds
"""

import numpy as np
from scipy.linalg import expm
from itertools import product as iprod
from collections import defaultdict
import json
import time
import argparse

# ============================================================
# PAULI BASIS
# ============================================================

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
PAULI_LABELS = ['I', 'X', 'Y', 'Z']


def pauli_basis(N: int):
    """
    Generate the full N-qubit Pauli basis.
    
    Returns list of (label_string, matrix, weight) tuples.
    Weight = number of non-identity factors (Hamming weight).
    """
    basis = []
    for indices in iprod(range(4), repeat=N):
        label = ''.join(PAULI_LABELS[i] for i in indices)
        weight = sum(1 for i in indices if i != 0)
        
        mat = np.eye(1, dtype=np.complex128)
        for i in indices:
            mat = np.kron(mat, [I, X, Y, Z][i])
        
        basis.append((label, mat, weight))
    
    return basis


def decompose_hamiltonian(H: np.ndarray, N: int, basis=None):
    """
    Decompose H in the Pauli basis.
    
    H = sum_k c_k P_k where c_k = Tr(H P_k) / 2^N
    
    Returns dict: label -> (coefficient, weight)
    """
    dim = 2**N
    if basis is None:
        basis = pauli_basis(N)
    
    coeffs = {}
    for label, P, weight in basis:
        c = np.trace(H @ P) / dim
        if abs(c) > 1e-12:
            coeffs[label] = (complex(c), weight)
    
    return coeffs


def classify_interactions(coeffs: dict, N: int):
    """
    Classify Pauli decomposition by weight and operator content.
    
    Returns structured summary of the force content.
    """
    by_weight = defaultdict(list)
    
    for label, (c, weight) in coeffs.items():
        # Find which sites are active
        active_sites = [i for i, ch in enumerate(label) if ch != 'I']
        active_ops = ''.join(label[i] for i in active_sites)
        
        by_weight[weight].append({
            'label': label,
            'coeff': c,
            'magnitude': abs(c),
            'sites': active_sites,
            'operators': active_ops,
        })
    
    # Sort each weight class by magnitude
    for w in by_weight:
        by_weight[w].sort(key=lambda x: -x['magnitude'])
    
    return dict(by_weight)


def interaction_signature(classified: dict, N: int):
    """
    Extract the force signature from weight-2 terms.
    
    For each pair of neighboring sites, determine the operator 
    content (XX, YY, ZZ, XY, etc.) and relative strengths.
    """
    if 2 not in classified:
        return {'pairs': {}, 'summary': 'No weight-2 terms found'}
    
    # Group weight-2 terms by site pair
    pairs = defaultdict(dict)
    for term in classified[2]:
        pair = tuple(term['sites'])
        ops = term['operators']
        pairs[pair][ops] = {
            'coeff': term['coeff'],
            'magnitude': term['magnitude'],
        }
    
    # For each pair, compute symmetry diagnostics
    pair_summaries = {}
    for pair, ops in pairs.items():
        summary = {}
        
        # Extract all 9 possible components
        comp_names = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
        mags = {name: ops.get(name, {}).get('magnitude', 0) for name in comp_names}
        
        xx, yy, zz = mags['XX'], mags['YY'], mags['ZZ']
        
        # Symmetric part: XX, YY, ZZ
        symmetric = xx + yy + zz
        # Cross terms
        cross = sum(mags[k] for k in comp_names if k not in ['XX', 'YY', 'ZZ'])
        total = symmetric + cross
        
        if total < 1e-12:
            continue
        
        summary['total_magnitude'] = total
        summary['symmetric_fraction'] = symmetric / total
        summary['cross_fraction'] = cross / total
        summary['components'] = {k: v for k, v in mags.items() if v > 1e-12}
        summary['component_coeffs'] = {
            k: complex(v['coeff']) for k, v in ops.items()
        }
        
        # Symmetry classification
        if xx > 1e-12 and yy > 1e-12:
            xx_yy_ratio = min(xx, yy) / max(xx, yy)
        else:
            xx_yy_ratio = 0.0
        
        diag_nonzero = [d for d in [xx, yy, zz] if d > 1e-12]
        if len(diag_nonzero) >= 2:
            su2_ratio = min(diag_nonzero) / max(diag_nonzero)
        else:
            su2_ratio = 0.0
        
        hopping_strength = (xx + yy) / 2
        ising_strength = zz
        
        summary['xx_yy_ratio'] = xx_yy_ratio
        summary['su2_ratio'] = su2_ratio
        summary['hopping_strength'] = hopping_strength
        summary['ising_strength'] = ising_strength
        
        if hopping_strength > 1e-10:
            summary['ising_to_hopping'] = ising_strength / hopping_strength
        
        # Classify
        if xx_yy_ratio > 0.95 and su2_ratio > 0.95 and cross / total < 0.05:
            summary['symmetry'] = 'SU(2) Heisenberg'
        elif xx_yy_ratio > 0.95 and cross / total < 0.05:
            summary['symmetry'] = 'U(1) (XX+YY type)'
        elif zz > 0.9 * total:
            summary['symmetry'] = 'Ising (ZZ only)'
        elif cross > 0.5 * total:
            summary['symmetry'] = 'Dzyaloshinskii-Moriya (cross-dominated)'
        else:
            summary['symmetry'] = 'Mixed/Other'
        
        pair_summaries[str(pair)] = summary
    
    # Global summary
    symmetries = [s.get('symmetry', 'Unknown') for s in pair_summaries.values()]
    symmetry_counts = defaultdict(int)
    for s in symmetries:
        symmetry_counts[s] += 1
    
    hop_ratios = [s.get('ising_to_hopping', None) for s in pair_summaries.values()]
    hop_ratios = [r for r in hop_ratios if r is not None]
    
    return {
        'pairs': pair_summaries,
        'symmetry_counts': dict(symmetry_counts),
        'mean_ising_to_hopping': float(np.mean(hop_ratios)) if hop_ratios else None,
        'std_ising_to_hopping': float(np.std(hop_ratios)) if hop_ratios else None,
    }


# ============================================================
# HAMILTONIANS - Various seeds to test universality
# ============================================================

def make_heisenberg_1d(N: int, J: float = 1.0):
    """1D Heisenberg chain with periodic boundaries."""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(N):
        j = (i + 1) % N
        for pauli in [X, Y, Z]:
            term = np.eye(1)
            for k in range(N):
                if k == i or k == j:
                    term = np.kron(term, pauli)
                else:
                    term = np.kron(term, I)
            H += J * term
    return H


def make_ising_transverse(N: int, J: float = 1.0, h: float = 0.5):
    """Transverse field Ising model."""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(N):
        j = (i + 1) % N
        term = np.eye(1)
        for k in range(N):
            if k == i or k == j:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += J * term
    
    for i in range(N):
        term = np.eye(1)
        for k in range(N):
            if k == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += h * term
    return H


def make_xy_model(N: int, J: float = 1.0):
    """XY model - U(1) symmetric."""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(N):
        j = (i + 1) % N
        for pauli in [X, Y]:
            term = np.eye(1)
            for k in range(N):
                if k == i or k == j:
                    term = np.kron(term, pauli)
                else:
                    term = np.kron(term, I)
            H += J * term
    return H


def make_random_local(N: int, seed: int = 42):
    """
    Random local Hamiltonian with nearest-neighbor terms.
    No particular symmetry imposed.
    """
    rng = np.random.RandomState(seed)
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    paulis_noid = [X, Y, Z]
    
    # Random on-site terms
    for i in range(N):
        for p in paulis_noid:
            coeff = rng.randn()
            term = np.eye(1)
            for k in range(N):
                if k == i:
                    term = np.kron(term, p)
                else:
                    term = np.kron(term, I)
            H += coeff * term
    
    # Random nearest-neighbor terms
    for i in range(N):
        j = (i + 1) % N
        for p1 in paulis_noid:
            for p2 in paulis_noid:
                coeff = rng.randn() * 0.5
                term = np.eye(1)
                for k in range(N):
                    if k == i:
                        term = np.kron(term, p1)
                    elif k == j:
                        term = np.kron(term, p2)
                    else:
                        term = np.kron(term, I)
                H += coeff * term
    return H


def make_hubbard_1d(N: int, t: float = 1.0, U: float = 0.5):
    """
    1D Hubbard-like model on qubits.
    Hopping + nearest-neighbor density-density.
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    Sp = (X + 1j*Y) / 2
    Sm = (X - 1j*Y) / 2
    n_op = (I - Z) / 2  # |1><1| particle number

    for i in range(N):
        j = (i + 1) % N
        # Hopping
        term = np.eye(1)
        for k in range(N):
            if k == i:
                term = np.kron(term, Sp)
            elif k == j:
                term = np.kron(term, Sm)
            else:
                term = np.kron(term, I)
        H += -t * (term + term.conj().T)
        
        # Density-density
        term = np.eye(1)
        for k in range(N):
            if k == i or k == j:
                term = np.kron(term, n_op)
            else:
                term = np.kron(term, I)
        H += U * term
    return H


HAMILTONIANS = {
    'heisenberg': make_heisenberg_1d,
    'ising': make_ising_transverse,
    'xy': make_xy_model,
    'random_local': make_random_local,
    'hubbard': make_hubbard_1d,
}


# ============================================================
# SCRAMBLING
# ============================================================

def scramble(H: np.ndarray, depth: int, N: int, seed: int = 0):
    """
    Apply a random unitary circuit to scramble H -> U H U†.
    """
    rng = np.random.RandomState(seed)
    dim = 2**N
    U_total = np.eye(dim, dtype=np.complex128)
    
    for layer in range(depth):
        offset = layer % 2
        for i in range(offset, N - 1, 2):
            j = i + 1
            # Random SU(4) via QR
            raw = rng.randn(4, 4) + 1j * rng.randn(4, 4)
            Q, _ = np.linalg.qr(raw)
            
            # Embed in full Hilbert space
            U_gate = np.eye(dim, dtype=np.complex128)
            for a in range(dim):
                for b in range(dim):
                    ai = (a >> (N - 1 - i)) & 1
                    aj = (a >> (N - 1 - j)) & 1
                    bi = (b >> (N - 1 - i)) & 1
                    bj = (b >> (N - 1 - j)) & 1
                    
                    a_rest = a & ~((1 << (N - 1 - i)) | (1 << (N - 1 - j)))
                    b_rest = b & ~((1 << (N - 1 - i)) | (1 << (N - 1 - j)))
                    
                    if a_rest == b_rest:
                        U_gate[a, b] = Q[2*ai + aj, 2*bi + bj]
                    else:
                        U_gate[a, b] = 0.0
            
            U_total = U_gate @ U_total
    
    return U_total @ H @ U_total.conj().T


# ============================================================
# DOUBLE BRACKET FLOW
# ============================================================

def locality_cost(H: np.ndarray, N: int, p: float = 4.0, basis=None):
    """
    Locality cost C_p(H) = sum w(P)^p |c_k|^2 / sum |c_k|^2.
    """
    if basis is None:
        basis = pauli_basis(N)
    
    dim = 2**N
    num = 0.0
    den = 0.0
    
    for label, P, weight in basis:
        c = np.trace(H @ P) / dim
        c2 = abs(c)**2
        den += c2
        num += (weight**p) * c2
    
    return num / den if den > 0 else 0.0


def locality_gradient(H: np.ndarray, N: int, p: float = 4.0, basis=None):
    """
    Gradient operator M for the double bracket flow.
    """
    if basis is None:
        basis = pauli_basis(N)
    
    dim = 2**N
    M = np.zeros_like(H)
    den = 0.0
    
    terms = []
    for label, P, weight in basis:
        c = np.trace(H @ P) / dim
        c2 = abs(c)**2
        den += c2
        if weight > 0:
            terms.append((c, P, weight))
    
    for c, P, weight in terms:
        M += (weight**p) * c * P
    
    M *= 2.0 / den
    return M


def double_bracket_step(H: np.ndarray, M: np.ndarray, eta: float):
    """
    One step: H -> exp(eta*K) H exp(-eta*K), K = [H, M].
    Isospectral by construction.
    """
    K = H @ M - M @ H
    eK = expm(eta * K)
    eKd = expm(-eta * K)
    return eK @ H @ eKd


def run_flow(H: np.ndarray, N: int, p: float = 4.0,
             max_steps: int = 500, eta: float = 0.01,
             tol: float = 1e-8, verbose: bool = False):
    """
    Run the double bracket flow to minimize locality cost.
    """
    basis = pauli_basis(N)
    
    costs = []
    cost = locality_cost(H, N, p, basis)
    costs.append(cost)
    
    if verbose:
        print(f"  Step 0: cost = {cost:.6f}")
    
    H_current = H.copy()
    
    for step in range(1, max_steps + 1):
        M = locality_gradient(H_current, N, p, basis)
        
        # Backtracking line search
        current_cost = cost
        eta_try = eta
        
        for _ in range(10):
            H_try = double_bracket_step(H_current, M, eta_try)
            new_cost = locality_cost(H_try, N, p, basis)
            
            if new_cost < current_cost:
                break
            eta_try *= 0.5
        else:
            if verbose:
                print(f"  Step {step}: converged (no improvement)")
            break
        
        H_current = H_try
        cost = new_cost
        costs.append(cost)
        
        if verbose and step % 50 == 0:
            print(f"  Step {step}: cost = {cost:.6f}")
        
        if len(costs) >= 2 and abs(costs[-1] - costs[-2]) < tol:
            if verbose:
                print(f"  Step {step}: converged (delta < {tol})")
            break
    
    return H_current, costs


# ============================================================
# SPECTRAL VERIFICATION
# ============================================================

def verify_isospectrality(H_original, H_recovered, tol=1e-6):
    """Verify that the flow preserved the spectrum."""
    eigs_orig = np.sort(np.real(np.linalg.eigvalsh(H_original)))
    eigs_rec = np.sort(np.real(np.linalg.eigvalsh(H_recovered)))
    max_diff = np.max(np.abs(eigs_orig - eigs_rec))
    return max_diff < tol, max_diff


# ============================================================
# WEIGHT SPECTRUM
# ============================================================

def weight_spectrum(coeffs: dict):
    """Total |c|^2 at each locality weight, normalized."""
    spectrum = defaultdict(float)
    for label, (c, weight) in coeffs.items():
        spectrum[weight] += abs(c)**2
    
    total = sum(spectrum.values())
    return {w: v/total for w, v in sorted(spectrum.items())}


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(N: int, hamiltonian_name: str, 
                   scramble_depth: int = None,
                   scramble_seed: int = 0,
                   p: float = 4.0,
                   max_steps: int = 500,
                   eta: float = 0.01,
                   verbose: bool = True):
    """
    Full pipeline: build -> scramble -> flow -> decompose -> classify.
    """
    if scramble_depth is None:
        scramble_depth = N
    
    print(f"\n{'='*60}")
    print(f"Hamiltonian: {hamiltonian_name}, N={N}")
    print(f"Scramble: depth={scramble_depth}, seed={scramble_seed}")
    print(f"Flow: p={p}, max_steps={max_steps}")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    # Build
    if hamiltonian_name == 'random_local':
        H_local = HAMILTONIANS[hamiltonian_name](N, seed=scramble_seed + 100)
    else:
        H_local = HAMILTONIANS[hamiltonian_name](N)
    
    # Original decomposition
    print("\n--- Original Hamiltonian ---")
    basis = pauli_basis(N)
    coeffs_orig = decompose_hamiltonian(H_local, N, basis)
    ws_orig = weight_spectrum(coeffs_orig)
    print(f"Weight spectrum: {', '.join(f'w{k}={v:.4f}' for k, v in ws_orig.items())}")
    orig_cost = locality_cost(H_local, N, p, basis)
    print(f"Locality cost (p={p}): {orig_cost:.4f}")
    
    # Scramble
    print(f"\n--- Scrambling (depth={scramble_depth}) ---")
    H_scrambled = scramble(H_local, scramble_depth, N, scramble_seed)
    
    coeffs_scr = decompose_hamiltonian(H_scrambled, N, basis)
    ws_scr = weight_spectrum(coeffs_scr)
    print(f"Weight spectrum: {', '.join(f'w{k}={v:.4f}' for k, v in ws_scr.items())}")
    scr_cost = locality_cost(H_scrambled, N, p, basis)
    print(f"Locality cost (p={p}): {scr_cost:.4f}")
    
    iso_ok, iso_diff = verify_isospectrality(H_local, H_scrambled)
    print(f"Isospectrality: {'PASS' if iso_ok else 'FAIL'} (max diff = {iso_diff:.2e})")
    
    # Double bracket flow
    print(f"\n--- Double Bracket Flow ---")
    H_recovered, cost_trajectory = run_flow(
        H_scrambled, N, p, max_steps, eta, verbose=verbose
    )
    
    final_cost = cost_trajectory[-1]
    print(f"Final cost: {final_cost:.6f} (original: {orig_cost:.4f})")
    print(f"Flow steps: {len(cost_trajectory) - 1}")
    
    iso_ok, iso_diff = verify_isospectrality(H_local, H_recovered)
    print(f"Isospectrality: {'PASS' if iso_ok else 'FAIL'} (max diff = {iso_diff:.2e})")
    
    # Decompose recovered Hamiltonian
    print(f"\n--- Recovered Hamiltonian Decomposition ---")
    coeffs_rec = decompose_hamiltonian(H_recovered, N, basis)
    ws_rec = weight_spectrum(coeffs_rec)
    print(f"Weight spectrum: {', '.join(f'w{k}={v:.4f}' for k, v in ws_rec.items())}")
    
    # Classify
    classified = classify_interactions(coeffs_rec, N)
    
    for w in sorted(classified.keys()):
        terms = classified[w]
        total_mag = sum(t['magnitude'] for t in terms)
        print(f"\n  Weight {w}: {len(terms)} terms, total magnitude = {total_mag:.6f}")
        
        if w <= 2:
            for t in terms[:10]:
                c = t['coeff']
                print(f"    {t['label']}: {c.real:+.6f} {'+' if c.imag >= 0 else ''}{c.imag:.6f}j "
                      f"(|c|={t['magnitude']:.6f})")
    
    # Force signature
    print(f"\n--- Force Signature (Weight-2 Analysis) ---")
    sig = interaction_signature(classified, N)
    
    if sig['symmetry_counts']:
        print(f"Symmetry classification across pairs:")
        for sym, count in sig['symmetry_counts'].items():
            print(f"  {sym}: {count} pairs")
    
    if sig['mean_ising_to_hopping'] is not None:
        print(f"Mean Ising/Hopping ratio: {sig['mean_ising_to_hopping']:.4f} "
              f"+/- {sig['std_ising_to_hopping']:.4f}")
    
    for pair_str, psummary in sig['pairs'].items():
        sym = psummary.get('symmetry', 'Unknown')
        comps = psummary.get('components', {})
        print(f"\n  Pair {pair_str} [{sym}]:")
        for op, mag in sorted(comps.items(), key=lambda x: -x[1]):
            print(f"    {op}: {mag:.6f}")
    
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    
    return {
        'hamiltonian': hamiltonian_name,
        'N': N,
        'scramble_depth': scramble_depth,
        'scramble_seed': scramble_seed,
        'p': p,
        'cost_original': orig_cost,
        'cost_scrambled': scr_cost,
        'cost_recovered': final_cost,
        'flow_steps': len(cost_trajectory) - 1,
        'weight_spectrum_original': ws_orig,
        'weight_spectrum_scrambled': ws_scr,
        'weight_spectrum_recovered': ws_rec,
        'force_signature': sig,
        'isospectrality_diff': iso_diff,
        'elapsed_seconds': elapsed,
    }


# ============================================================
# UNIVERSALITY TEST
# ============================================================

def test_universality(N: int, p: float = 4.0, scramble_seeds=None,
                      max_steps: int = 500, verbose: bool = False):
    """
    Test whether different starting Hamiltonians produce
    the same force structure after accessibility collapse.
    """
    if scramble_seeds is None:
        scramble_seeds = [0, 1, 2]
    
    print("\n" + "=" * 70)
    print("UNIVERSALITY TEST")
    print(f"N={N}, p={p}, seeds={scramble_seeds}")
    print("=" * 70)
    
    all_results = []
    
    for ham_name in HAMILTONIANS:
        for seed in scramble_seeds:
            result = run_experiment(
                N=N,
                hamiltonian_name=ham_name,
                scramble_seed=seed,
                p=p,
                max_steps=max_steps,
                verbose=verbose,
            )
            all_results.append(result)
    
    # Comparison table
    print("\n" + "=" * 70)
    print("UNIVERSALITY COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Hamiltonian':<20} {'Seed':<6} {'Final Cost':<12} "
          f"{'Dominant Symmetry':<25} {'Ising/Hop':<12} "
          f"{'w1 frac':<10} {'w2 frac':<10} {'w3+ frac':<10}")
    print("-" * 115)
    
    for r in all_results:
        sym_counts = r['force_signature'].get('symmetry_counts', {})
        dominant_sym = max(sym_counts, key=sym_counts.get) if sym_counts else 'N/A'
        ih_ratio = r['force_signature'].get('mean_ising_to_hopping', None)
        ih_str = f"{ih_ratio:.4f}" if ih_ratio is not None else "N/A"
        
        ws = r['weight_spectrum_recovered']
        w1 = ws.get(1, 0)
        w2 = ws.get(2, 0)
        w3plus = sum(v for k, v in ws.items() if k >= 3)
        
        print(f"{r['hamiltonian']:<20} {r['scramble_seed']:<6} "
              f"{r['cost_recovered']:<12.4f} {dominant_sym:<25} "
              f"{ih_str:<12} {w1:<10.4f} {w2:<10.4f} {w3plus:<10.4f}")
    
    # Operator content comparison
    print("\n--- Weight-2 Operator Content Comparison ---")
    for r in all_results:
        sig = r['force_signature']
        if not sig['pairs']:
            continue
        
        op_totals = defaultdict(float)
        for pair_str, psummary in sig['pairs'].items():
            for op, mag in psummary.get('components', {}).items():
                op_totals[op] += mag
        
        total = sum(op_totals.values())
        if total > 0:
            op_fracs = {k: v/total for k, v in op_totals.items()}
            top_ops = sorted(op_fracs.items(), key=lambda x: -x[1])[:5]
            ops_str = ', '.join(f"{op}={frac:.3f}" for op, frac in top_ops)
            print(f"  {r['hamiltonian']:<16} seed={r['scramble_seed']}: {ops_str}")
    
    return all_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Force Emergence from Accessibility Collapse"
    )
    parser.add_argument('--quick', action='store_true', help='N=4 single test')
    parser.add_argument('--standard', action='store_true', help='N=5 universality')
    parser.add_argument('--full', action='store_true', help='N=5,6 universality')
    parser.add_argument('--N', type=int, default=None, help='System size')
    parser.add_argument('--p', type=float, default=4.0, help='Locality penalty')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--eta', type=float, default=0.01, help='Step size')
    parser.add_argument('--seeds', type=str, default='0,1,2')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    
    if args.quick:
        result = run_experiment(
            N=4, hamiltonian_name='heisenberg',
            scramble_seed=0, p=args.p, max_steps=args.max_steps,
            verbose=True
        )
        results = [result]
    
    elif args.standard:
        results = test_universality(
            N=5, p=args.p, scramble_seeds=seeds,
            max_steps=args.max_steps, verbose=args.verbose
        )
    
    elif args.full:
        all_results = []
        for N in [5, 6]:
            r = test_universality(
                N=N, p=args.p, scramble_seeds=seeds,
                max_steps=args.max_steps, verbose=args.verbose
            )
            all_results.extend(r)
        results = all_results
    
    elif args.N:
        results = test_universality(
            N=args.N, p=args.p, scramble_seeds=seeds,
            max_steps=args.max_steps, verbose=args.verbose
        )
    
    else:
        result = run_experiment(
            N=4, hamiltonian_name='heisenberg',
            scramble_seed=0, p=args.p, max_steps=200,
            verbose=True
        )
        results = [result]
    
    # Save
    if args.output:
        def to_serializable(obj):
            if isinstance(obj, complex):
                return {'real': obj.real, 'imag': obj.imag}
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            return obj
        
        with open(args.output, 'w') as f:
            json.dump(to_serializable(results), f, indent=2)
        print(f"\nSaved to: {args.output}")
    
    print("\nDone.")