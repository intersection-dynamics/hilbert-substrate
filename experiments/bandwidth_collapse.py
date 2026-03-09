"""
Bandwidth-Aware Accessibility Collapse
========================================

Modified double bracket flow where the cost functional includes
both spatial locality (weight-based) and transport efficiency
(operator-content-based).

Key idea:
  Standard DBF cost: C_spatial = Σ w(P_k)^p |c_k|^2 / Σ |c_k|^2
  - Only sees how many sites an operator touches
  - Blind to operator content (XX vs ZY both have weight 2)
  
  New transport cost: C_transport = Σ v(P_k) |c_k|^2 / Σ |c_k|^2
  - v(P_k) = transport velocity associated with that operator type
  - From our data: XX,YY ~ 3.93, ZZ ~ 0, cross ~ 4.18
  
  Combined: C_total = C_spatial + lambda * C_transport
  
  The double bracket flow on C_total simultaneously:
  1. Pushes weight into low-body terms (locality)
  2. Pushes weight-2 content toward low-transport operators (bandwidth)
  
  Prediction: the flow should produce Hopping + Ising with suppressed
  cross terms, i.e. U(1) gauge structure.

Usage:
    python bandwidth_collapse.py --quick      # N=4, single test
    python bandwidth_collapse.py --standard   # N=5, multiple Hamiltonians
    python bandwidth_collapse.py --sweep      # Sweep over lambda values
"""

import numpy as np
from scipy.linalg import expm
from itertools import product as iprod
import time
import argparse
import json


# ============================================================
# PAULI INFRASTRUCTURE
# ============================================================

I2 = np.eye(2, dtype=np.complex128)
PAULIS = {
    'I': np.eye(2, dtype=np.complex128),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
}
PAULI_LABELS = ['I', 'X', 'Y', 'Z']


def build_pauli_basis(N):
    """
    Build the full N-qubit Pauli basis.
    Returns list of (label_tuple, matrix, weight) for all 4^N elements.
    """
    basis = []
    for labels in iprod(PAULI_LABELS, repeat=N):
        mat = np.eye(1, dtype=np.complex128)
        for l in labels:
            mat = np.kron(mat, PAULIS[l])
        weight = sum(1 for l in labels if l != 'I')
        basis.append((labels, mat, weight))
    return basis


def pauli_decompose(H, basis, N):
    """
    Decompose H in the Pauli basis.
    Returns coefficients c_k such that H = Σ c_k P_k.
    c_k = Tr(H P_k) / 2^N
    """
    dim = 2**N
    coeffs = np.zeros(len(basis), dtype=np.complex128)
    for k, (labels, Pk, w) in enumerate(basis):
        coeffs[k] = np.trace(H @ Pk) / dim
    return coeffs


def pauli_reconstruct(coeffs, basis):
    """Reconstruct H from Pauli coefficients."""
    H = np.zeros_like(basis[0][1])
    for k, (labels, Pk, w) in enumerate(basis):
        if abs(coeffs[k]) > 1e-15:
            H += coeffs[k] * Pk
    return H


# ============================================================
# TRANSPORT VELOCITY ASSIGNMENT
# ============================================================

def get_transport_velocity(labels):
    """
    Assign a transport velocity to a Pauli operator based on its
    weight-2 content.
    
    For weight != 2, returns 0 (only weight-2 terms carry forces).
    
    For weight-2 terms:
      - Matched pairs (XX, YY, ZZ): based on measured LR velocities
        XX, YY -> hopping velocity (3.93)
        ZZ -> Ising velocity (0.0)
      - Cross pairs (XY, XZ, YX, YZ, ZX, ZY): cross velocity (4.18)
    
    These values come from the bandwidth force selection experiment.
    """
    # Find non-identity positions
    non_id = [(i, labels[i]) for i in range(len(labels)) if labels[i] != 'I']
    
    if len(non_id) != 2:
        return 0.0  # Only assign transport cost to weight-2
    
    _, op1 = non_id[0]
    _, op2 = non_id[1]
    
    # Velocity assignments from measured data
    if op1 == op2:
        # Matched pair
        if op1 in ('X', 'Y'):
            return 3.93   # Hopping channel
        else:  # Z
            return 0.0    # Ising channel (no transport)
    else:
        # Cross pair
        return 4.18       # Cross channel


def get_operator_class(labels):
    """
    Classify a weight-2 Pauli operator.
    Returns: 'hopping', 'ising', 'cross', or 'other'
    """
    non_id = [(i, labels[i]) for i in range(len(labels)) if labels[i] != 'I']
    
    if len(non_id) != 2:
        return 'other'
    
    _, op1 = non_id[0]
    _, op2 = non_id[1]
    
    if op1 == op2:
        if op1 in ('X', 'Y'):
            return 'hopping'
        else:
            return 'ising'
    else:
        return 'cross'


# ============================================================
# COST FUNCTIONALS
# ============================================================

def cost_spatial(coeffs, basis, p=4):
    """
    Standard locality cost from Paper II.
    C_spatial = Σ w(P_k)^p |c_k|^2 / Σ |c_k|^2
    """
    c2 = np.abs(coeffs)**2
    total = np.sum(c2)
    if total < 1e-30:
        return 0.0
    
    weighted = sum(w**p * c2[k] for k, (_, _, w) in enumerate(basis))
    return (weighted / total).real


def cost_transport(coeffs, basis):
    """
    Transport cost: penalizes high-velocity operator types.
    C_transport = Σ v(P_k) |c_k|^2 / Σ |c_k|^2
    """
    c2 = np.abs(coeffs)**2
    total = np.sum(c2)
    if total < 1e-30:
        return 0.0
    
    weighted = sum(get_transport_velocity(labels) * c2[k] 
                   for k, (labels, _, _) in enumerate(basis))
    return (weighted / total).real


def cost_combined(coeffs, basis, p=4, lam=1.0):
    """
    Combined cost: spatial + lambda * transport.
    """
    cs = cost_spatial(coeffs, basis, p)
    ct = cost_transport(coeffs, basis)
    return cs + lam * ct


# ============================================================
# GRADIENT COMPUTATION
# ============================================================

def compute_gradient_operator(H, coeffs, basis, N, p=4, lam=0.0):
    """
    Compute the gradient operator M for the double bracket flow.
    
    For C_total = C_spatial + lambda * C_transport:
    
    M = Σ_k (∂C_total/∂c_k) P_k
    
    where ∂C/∂c_k = 2 * c_k * [penalty(k) - C] / (Σ |c_j|^2)
    
    and penalty(k) = w(P_k)^p + lambda * v(P_k)
    """
    c2 = np.abs(coeffs)**2
    total = np.sum(c2)
    if total < 1e-30:
        return np.zeros_like(H)
    
    # Current cost value
    C_current = cost_combined(coeffs, basis, p, lam)
    
    # Build gradient operator
    dim = H.shape[0]
    M = np.zeros((dim, dim), dtype=np.complex128)
    
    for k, (labels, Pk, w) in enumerate(basis):
        if abs(coeffs[k]) < 1e-15:
            continue
        
        # Total penalty for this term
        penalty_k = w**p + lam * get_transport_velocity(labels)
        
        # Gradient coefficient
        grad_k = 2.0 * coeffs[k] * (penalty_k - C_current) / total
        
        M += grad_k * Pk
    
    return M


# ============================================================
# DOUBLE BRACKET FLOW
# ============================================================

def double_bracket_step(H, M, dt):
    """
    One step of the double bracket flow:
    H(t+dt) ≈ exp(dt * K) H exp(-dt * K)
    where K = [H, M]
    """
    K = H @ M - M @ H  # [H, M]
    
    # Generate unitary
    U = expm(dt * K)
    
    return U @ H @ U.conj().T


def run_flow(H_init, N, basis, p=4, lam=0.0, n_steps=500,
             dt_init=0.01, adaptive=True, verbose=True):
    """
    Run the double bracket flow with combined cost.
    
    Returns trajectory of cost values and final Hamiltonian.
    """
    H = H_init.copy()
    
    trajectory = {
        'cost_total': [],
        'cost_spatial': [],
        'cost_transport': [],
        'weight_spectrum': [],
        'operator_content': [],
        'steps': [],
    }
    
    dt = dt_init
    prev_cost = None
    stall_count = 0
    
    for step in range(n_steps):
        # Decompose current H
        coeffs = pauli_decompose(H, basis, N)
        
        # Compute costs
        c_spatial = cost_spatial(coeffs, basis, p)
        c_transport = cost_transport(coeffs, basis)
        c_total = c_spatial + lam * c_transport
        
        # Weight spectrum
        c2 = np.abs(coeffs)**2
        total_c2 = np.sum(c2)
        w_fracs = {}
        for k, (labels, _, w) in enumerate(basis):
            w_fracs[w] = w_fracs.get(w, 0) + c2[k]
        w_fracs = {w: f/total_c2 for w, f in w_fracs.items()}
        
        # Operator content of weight-2 terms
        op_content = {'hopping': 0, 'ising': 0, 'cross': 0}
        w2_total = 0
        for k, (labels, _, w) in enumerate(basis):
            if w == 2:
                cls = get_operator_class(labels)
                if cls in op_content:
                    op_content[cls] += c2[k]
                    w2_total += c2[k]
        if w2_total > 0:
            op_content = {k: v/w2_total for k, v in op_content.items()}
        
        # Record
        trajectory['cost_total'].append(c_total)
        trajectory['cost_spatial'].append(c_spatial)
        trajectory['cost_transport'].append(c_transport)
        trajectory['weight_spectrum'].append(dict(w_fracs))
        trajectory['operator_content'].append(dict(op_content))
        trajectory['steps'].append(step)
        
        # Logging
        if verbose and (step % 50 == 0 or step == n_steps - 1):
            w1 = w_fracs.get(1, 0)
            w2 = w_fracs.get(2, 0)
            w3p = sum(v for k, v in w_fracs.items() if k >= 3)
            print(f"  Step {step:4d}: C_total={c_total:.4f} "
                  f"(spatial={c_spatial:.4f}, transport={c_transport:.4f}) | "
                  f"w1={w1:.3f} w2={w2:.3f} w3+={w3p:.3f} | "
                  f"hop={op_content.get('hopping',0):.3f} "
                  f"ising={op_content.get('ising',0):.3f} "
                  f"cross={op_content.get('cross',0):.3f} | "
                  f"dt={dt:.5f}")
        
        # Check convergence
        if prev_cost is not None:
            improvement = prev_cost - c_total
            if improvement < 1e-8:
                stall_count += 1
                if adaptive:
                    dt *= 0.8
                if stall_count > 20:
                    if verbose:
                        print(f"  Converged at step {step} (stalled)")
                    break
            else:
                stall_count = 0
                if adaptive and improvement > 0.01:
                    dt = min(dt * 1.1, 0.1)
        
        prev_cost = c_total
        
        # Compute gradient and step
        M = compute_gradient_operator(H, coeffs, basis, N, p, lam)
        
        # Backtracking line search
        H_new = double_bracket_step(H, M, dt)
        coeffs_new = pauli_decompose(H_new, basis, N)
        c_new = cost_combined(coeffs_new, basis, p, lam)
        
        attempts = 0
        while c_new > c_total and attempts < 10:
            dt *= 0.5
            H_new = double_bracket_step(H, M, dt)
            coeffs_new = pauli_decompose(H_new, basis, N)
            c_new = cost_combined(coeffs_new, basis, p, lam)
            attempts += 1
        
        if c_new <= c_total:
            H = H_new
        else:
            # Flow stuck
            if verbose and step % 100 == 0:
                print(f"  Step {step}: stuck, dt={dt:.2e}")
    
    return H, trajectory


# ============================================================
# SCRAMBLING
# ============================================================

def scramble_hamiltonian(H, N, depth=None, seed=None):
    """
    Scramble H by applying a random unitary circuit.
    H' = U H U†
    """
    if seed is not None:
        np.random.seed(seed)
    
    if depth is None:
        depth = N
    
    dim = 2**N
    U_total = np.eye(dim, dtype=np.complex128)
    
    for layer in range(depth):
        # Random 2-qubit gates on neighboring pairs
        for i in range(0, N - 1, 2 if layer % 2 == 0 else 1):
            j = (i + 1) % N
            if j == i:
                continue
            
            # Random SU(4) gate
            G = random_su(4, seed=None)
            
            # Embed in full space
            U_gate = embed_2site_gate(G, i, j, N)
            U_total = U_gate @ U_total
    
    return U_total @ H @ U_total.conj().T


def random_su(d, seed=None):
    """Random element of SU(d) via QR decomposition."""
    if seed is not None:
        np.random.seed(seed)
    
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    Q, R = np.linalg.qr(A)
    # Fix phases to get uniform Haar measure
    D = np.diag(R)
    Q = Q @ np.diag(D / np.abs(D))
    # Ensure det = 1
    Q = Q / np.linalg.det(Q)**(1/d)
    return Q


def embed_2site_gate(G, i, j, N):
    """Embed a 4x4 gate acting on sites i,j into the full 2^N space."""
    dim = 2**N
    U = np.zeros((dim, dim), dtype=np.complex128)
    
    for a in range(dim):
        ai = (a >> (N - 1 - i)) & 1
        aj = (a >> (N - 1 - j)) & 1
        a_rest = a ^ (ai << (N - 1 - i)) ^ (aj << (N - 1 - j))
        
        for bi in range(2):
            for bj in range(2):
                b = a_rest | (bi << (N - 1 - i)) | (bj << (N - 1 - j))
                U[b, a] = G[2*bi + bj, 2*ai + aj]
    
    return U


# ============================================================
# SEED HAMILTONIANS
# ============================================================

def make_heisenberg_chain(N, J=1.0, periodic=True):
    """1D Heisenberg chain."""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    n_edges = N if periodic else N - 1
    for i in range(n_edges):
        j = (i + 1) % N
        for pauli in ['X', 'Y', 'Z']:
            H += J * make_2site_pauli(N, i, j, pauli, pauli)
    
    return H


def make_ising_chain(N, J=1.0, h=0.5, periodic=True):
    """Transverse field Ising model."""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    n_edges = N if periodic else N - 1
    for i in range(n_edges):
        j = (i + 1) % N
        H += J * make_2site_pauli(N, i, j, 'Z', 'Z')
    
    for i in range(N):
        H += h * make_1site_pauli(N, i, 'X')
    
    return H


def make_xy_chain(N, J=1.0, periodic=True):
    """XY model."""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    n_edges = N if periodic else N - 1
    for i in range(n_edges):
        j = (i + 1) % N
        H += J * make_2site_pauli(N, i, j, 'X', 'X')
        H += J * make_2site_pauli(N, i, j, 'Y', 'Y')
    
    return H


def make_random_local(N, seed=42):
    """Random local Hamiltonian (weight-1 and weight-2 terms)."""
    np.random.seed(seed)
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Random on-site terms
    for i in range(N):
        for p in ['X', 'Y', 'Z']:
            H += np.random.randn() * make_1site_pauli(N, i, p)
    
    # Random nearest-neighbor terms
    for i in range(N):
        j = (i + 1) % N
        for p1 in ['X', 'Y', 'Z']:
            for p2 in ['X', 'Y', 'Z']:
                H += np.random.randn() * 0.5 * make_2site_pauli(N, i, j, p1, p2)
    
    return H


def make_2site_pauli(N, i, j, p1, p2):
    """Build σ_p1^i ⊗ σ_p2^j operator."""
    ops = [I2] * N
    ops[i] = PAULIS[p1]
    ops[j] = PAULIS[p2]
    
    result = ops[0]
    for k in range(1, N):
        result = np.kron(result, ops[k])
    return result


def make_1site_pauli(N, i, p):
    """Build σ_p^i operator."""
    ops = [I2] * N
    ops[i] = PAULIS[p]
    
    result = ops[0]
    for k in range(1, N):
        result = np.kron(result, ops[k])
    return result


# ============================================================
# ANALYSIS
# ============================================================

def analyze_force_content(H, basis, N, label=""):
    """
    Detailed analysis of the weight-2 operator content.
    """
    coeffs = pauli_decompose(H, basis, N)
    c2 = np.abs(coeffs)**2
    total = np.sum(c2)
    
    # Weight spectrum
    w_spec = {}
    for k, (labels, _, w) in enumerate(basis):
        w_spec[w] = w_spec.get(w, 0) + c2[k]
    w_spec = {w: f/total for w, f in sorted(w_spec.items())}
    
    # Weight-2 breakdown
    w2_breakdown = {}
    w2_total = 0
    for k, (labels, _, w) in enumerate(basis):
        if w == 2:
            non_id = [l for l in labels if l != 'I']
            op_type = ''.join(non_id)
            w2_breakdown[op_type] = w2_breakdown.get(op_type, 0) + c2[k]
            w2_total += c2[k]
    
    if w2_total > 0:
        w2_breakdown = {k: v/w2_total for k, v in sorted(w2_breakdown.items())}
    
    # Channel classification
    hop_frac = sum(v for k, v in w2_breakdown.items() if k in ('XX', 'YY'))
    ising_frac = w2_breakdown.get('ZZ', 0)
    cross_frac = sum(v for k, v in w2_breakdown.items() 
                     if k not in ('XX', 'YY', 'ZZ'))
    
    # Nearest-neighbor vs longer-range in weight-2
    nn_frac = 0
    lr_frac = 0
    for k, (labels, _, w) in enumerate(basis):
        if w == 2:
            sites = [i for i, l in enumerate(labels) if l != 'I']
            if len(sites) == 2:
                dist = min(abs(sites[1] - sites[0]), 
                          N - abs(sites[1] - sites[0]))
                if dist == 1:
                    nn_frac += c2[k]
                else:
                    lr_frac += c2[k]
    
    if w2_total > 0:
        nn_frac /= w2_total
        lr_frac /= w2_total
    
    print(f"\n{'='*60}")
    print(f"FORCE ANALYSIS: {label}")
    print(f"{'='*60}")
    print(f"\nWeight spectrum:")
    for w, f in w_spec.items():
        bar = '█' * int(f * 40)
        print(f"  w={w}: {f:.4f}  {bar}")
    
    print(f"\nWeight-2 operator content:")
    for op, f in sorted(w2_breakdown.items(), key=lambda x: -x[1]):
        bar = '█' * int(f * 40)
        print(f"  {op}: {f:.4f}  {bar}")
    
    print(f"\nChannel summary:")
    print(f"  Hopping (XX+YY): {hop_frac:.4f}")
    print(f"  Ising (ZZ):      {ising_frac:.4f}")
    print(f"  Cross (XY etc):  {cross_frac:.4f}")
    
    print(f"\nLocality of weight-2:")
    print(f"  Nearest-neighbor: {nn_frac:.4f}")
    print(f"  Longer-range:     {lr_frac:.4f}")
    
    # Symmetry check: is hopping + ising >> cross?
    u1_score = (hop_frac + ising_frac) - cross_frac
    print(f"\nU(1) score (hop+ising-cross): {u1_score:.4f}")
    print(f"  > 0 means U(1)-like, < 0 means cross-dominated")
    
    return {
        'weight_spectrum': w_spec,
        'w2_breakdown': w2_breakdown,
        'hopping': hop_frac,
        'ising': ising_frac,
        'cross': cross_frac,
        'nn_frac': nn_frac,
        'u1_score': u1_score,
    }


# ============================================================
# MAIN EXPERIMENTS
# ============================================================

def experiment_single(N, ham_type='heisenberg', p=4, lam=1.0, 
                      seed=0, n_steps=500, verbose=True):
    """
    Single experiment: scramble, flow with combined cost, analyze.
    """
    print(f"\n{'#'*70}")
    print(f"EXPERIMENT: N={N}, {ham_type}, p={p}, lambda={lam}, seed={seed}")
    print(f"{'#'*70}")
    
    # Build Pauli basis
    basis = build_pauli_basis(N)
    print(f"Pauli basis: {len(basis)} elements")
    
    # Build seed Hamiltonian
    if ham_type == 'heisenberg':
        H_seed = make_heisenberg_chain(N)
    elif ham_type == 'ising':
        H_seed = make_ising_chain(N)
    elif ham_type == 'xy':
        H_seed = make_xy_chain(N)
    elif ham_type == 'random':
        H_seed = make_random_local(N, seed=seed+100)
    else:
        raise ValueError(f"Unknown Hamiltonian type: {ham_type}")
    
    # Analyze original
    print("\n--- ORIGINAL ---")
    orig_analysis = analyze_force_content(H_seed, basis, N, f"Original {ham_type}")
    
    # Scramble
    H_scrambled = scramble_hamiltonian(H_seed, N, depth=2*N, seed=seed)
    print("\n--- SCRAMBLED ---")
    scr_analysis = analyze_force_content(H_scrambled, basis, N, "Scrambled")
    
    # Run flow WITHOUT transport cost (lambda=0, standard Paper II)
    print(f"\n{'='*60}")
    print(f"FLOW 1: Standard spatial-only (lambda=0)")
    print(f"{'='*60}")
    H_spatial, traj_spatial = run_flow(
        H_scrambled, N, basis, p=p, lam=0.0, 
        n_steps=n_steps, verbose=verbose
    )
    spatial_analysis = analyze_force_content(
        H_spatial, basis, N, "Recovered (spatial only)")
    
    # Run flow WITH transport cost
    print(f"\n{'='*60}")
    print(f"FLOW 2: Bandwidth-aware (lambda={lam})")
    print(f"{'='*60}")
    H_combined, traj_combined = run_flow(
        H_scrambled, N, basis, p=p, lam=lam,
        n_steps=n_steps, verbose=verbose
    )
    combined_analysis = analyze_force_content(
        H_combined, basis, N, f"Recovered (lambda={lam})")
    
    # Comparison
    print(f"\n{'='*60}")
    print(f"COMPARISON: Does bandwidth constraint change force content?")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<25} {'Scrambled':<12} {'Spatial-only':<12} {'Bandwidth':<12}")
    print("-" * 61)
    
    for name, scr, spa, comb in [
        ('Hopping (XX+YY)', scr_analysis['hopping'], 
         spatial_analysis['hopping'], combined_analysis['hopping']),
        ('Ising (ZZ)', scr_analysis['ising'],
         spatial_analysis['ising'], combined_analysis['ising']),
        ('Cross (XY etc)', scr_analysis['cross'],
         spatial_analysis['cross'], combined_analysis['cross']),
        ('U(1) score', scr_analysis['u1_score'],
         spatial_analysis['u1_score'], combined_analysis['u1_score']),
        ('NN fraction', scr_analysis['nn_frac'],
         spatial_analysis['nn_frac'], combined_analysis['nn_frac']),
    ]:
        print(f"{name:<25} {scr:<12.4f} {spa:<12.4f} {comb:<12.4f}")
    
    # Did bandwidth improve U(1) score?
    delta_u1 = combined_analysis['u1_score'] - spatial_analysis['u1_score']
    print(f"\nU(1) score change from bandwidth: {delta_u1:+.4f}")
    if delta_u1 > 0.01:
        print("  → Bandwidth constraint DOES select toward U(1)")
    elif delta_u1 < -0.01:
        print("  → Bandwidth constraint selects AWAY from U(1) (!)")
    else:
        print("  → Bandwidth constraint has minimal effect on gauge structure")
    
    return {
        'orig': orig_analysis,
        'scrambled': scr_analysis,
        'spatial': spatial_analysis,
        'combined': combined_analysis,
        'traj_spatial': traj_spatial,
        'traj_combined': traj_combined,
    }


def experiment_universality(N, p=4, lam=1.0, n_steps=500, 
                             seeds=[0, 1, 2], verbose=True):
    """
    Run across multiple Hamiltonians and seeds.
    Test whether bandwidth selects U(1) universally.
    """
    ham_types = ['heisenberg', 'ising', 'xy', 'random']
    
    all_results = []
    
    for ham_type in ham_types:
        for seed in seeds:
            result = experiment_single(
                N, ham_type, p, lam, seed, n_steps, verbose
            )
            
            all_results.append({
                'ham_type': ham_type,
                'seed': seed,
                'spatial_u1': result['spatial']['u1_score'],
                'combined_u1': result['combined']['u1_score'],
                'spatial_cross': result['spatial']['cross'],
                'combined_cross': result['combined']['cross'],
                'spatial_hopping': result['spatial']['hopping'],
                'combined_hopping': result['combined']['hopping'],
                'spatial_ising': result['spatial']['ising'],
                'combined_ising': result['combined']['ising'],
            })
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"UNIVERSALITY SUMMARY (N={N}, p={p}, lambda={lam})")
    print(f"{'#'*70}")
    
    print(f"\n{'Ham':<12} {'Seed':<6} {'Cross(sp)':<10} {'Cross(bw)':<10} "
          f"{'U1(sp)':<10} {'U1(bw)':<10} {'ΔU1':<10}")
    print("-" * 68)
    
    for r in all_results:
        delta = r['combined_u1'] - r['spatial_u1']
        marker = "✓" if delta > 0.01 else "~" if abs(delta) < 0.01 else "✗"
        print(f"{r['ham_type']:<12} {r['seed']:<6} "
              f"{r['spatial_cross']:<10.4f} {r['combined_cross']:<10.4f} "
              f"{r['spatial_u1']:<10.4f} {r['combined_u1']:<10.4f} "
              f"{delta:<+10.4f} {marker}")
    
    # Aggregate
    spatial_u1s = [r['spatial_u1'] for r in all_results]
    combined_u1s = [r['combined_u1'] for r in all_results]
    spatial_cross = [r['spatial_cross'] for r in all_results]
    combined_cross = [r['combined_cross'] for r in all_results]
    
    print(f"\nAverages:")
    print(f"  Spatial-only  U(1) score: {np.mean(spatial_u1s):.4f} ± {np.std(spatial_u1s):.4f}")
    print(f"  Bandwidth     U(1) score: {np.mean(combined_u1s):.4f} ± {np.std(combined_u1s):.4f}")
    print(f"  Spatial-only  cross frac: {np.mean(spatial_cross):.4f} ± {np.std(spatial_cross):.4f}")
    print(f"  Bandwidth     cross frac: {np.mean(combined_cross):.4f} ± {np.std(combined_cross):.4f}")
    
    improved = sum(1 for r in all_results 
                   if r['combined_u1'] > r['spatial_u1'] + 0.01)
    print(f"\n  Bandwidth improved U(1): {improved}/{len(all_results)} cases")
    
    return all_results


def experiment_lambda_sweep(N, ham_type='heisenberg', p=4, seed=0,
                             n_steps=300, verbose=False):
    """
    Sweep over lambda values to map the effect of bandwidth strength.
    """
    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    basis = build_pauli_basis(N)
    
    # Build and scramble
    if ham_type == 'heisenberg':
        H_seed = make_heisenberg_chain(N)
    elif ham_type == 'ising':
        H_seed = make_ising_chain(N)
    elif ham_type == 'xy':
        H_seed = make_xy_chain(N)
    else:
        H_seed = make_random_local(N, seed=seed+100)
    
    H_scrambled = scramble_hamiltonian(H_seed, N, depth=2*N, seed=seed)
    
    results = []
    
    for lam in lambdas:
        print(f"\n--- Lambda = {lam} ---")
        
        H_final, traj = run_flow(
            H_scrambled, N, basis, p=p, lam=lam,
            n_steps=n_steps, verbose=verbose
        )
        
        analysis = analyze_force_content(H_final, basis, N, f"lambda={lam}")
        
        results.append({
            'lambda': lam,
            'hopping': analysis['hopping'],
            'ising': analysis['ising'],
            'cross': analysis['cross'],
            'u1_score': analysis['u1_score'],
            'nn_frac': analysis['nn_frac'],
            'final_cost_spatial': traj['cost_spatial'][-1],
            'final_cost_transport': traj['cost_transport'][-1],
        })
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"LAMBDA SWEEP SUMMARY (N={N}, {ham_type}, seed={seed})")
    print(f"{'#'*70}")
    
    print(f"\n{'Lambda':<10} {'Hopping':<10} {'Ising':<10} {'Cross':<10} "
          f"{'U1 score':<10} {'C_spatial':<12} {'C_transport':<12}")
    print("-" * 74)
    
    for r in results:
        print(f"{r['lambda']:<10.1f} {r['hopping']:<10.4f} {r['ising']:<10.4f} "
              f"{r['cross']:<10.4f} {r['u1_score']:<+10.4f} "
              f"{r['final_cost_spatial']:<12.4f} {r['final_cost_transport']:<12.4f}")
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bandwidth-Aware Accessibility Collapse"
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: N=4, single Hamiltonian')
    parser.add_argument('--standard', action='store_true',
                        help='Standard: N=5, universality test')
    parser.add_argument('--sweep', action='store_true',
                        help='Lambda sweep: vary bandwidth strength')
    parser.add_argument('--full', action='store_true',
                        help='Full: N=5, universality + lambda sweep')
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--p', type=int, default=4)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--ham', type=str, default='heisenberg')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.quick:
        result = experiment_single(
            N=4, ham_type='heisenberg', p=4, lam=1.0,
            seed=0, n_steps=300, verbose=True
        )
    
    elif args.standard:
        N = args.N or 5
        result = experiment_universality(
            N=N, p=args.p, lam=args.lam,
            n_steps=args.steps, seeds=[0, 1], verbose=True
        )
    
    elif args.sweep:
        N = args.N or 5
        result = experiment_lambda_sweep(
            N=N, ham_type=args.ham, p=args.p, seed=args.seed,
            n_steps=args.steps, verbose=False
        )
    
    elif args.full:
        N = args.N or 5
        print("PHASE 1: Universality test")
        uni_result = experiment_universality(
            N=N, p=args.p, lam=args.lam,
            n_steps=args.steps, seeds=[0, 1], verbose=True
        )
        
        print("\n\nPHASE 2: Lambda sweep")
        lam_result = experiment_lambda_sweep(
            N=N, ham_type='heisenberg', p=args.p, seed=0,
            n_steps=args.steps, verbose=False
        )
        
        result = {'universality': uni_result, 'lambda_sweep': lam_result}
    
    else:
        # Custom single run
        N = args.N or 5
        result = experiment_single(
            N=N, ham_type=args.ham, p=args.p, lam=args.lam,
            seed=args.seed, n_steps=args.steps, verbose=True
        )
    
    total_time = time.time() - start_time
    print(f"\n\nTotal runtime: {total_time/60:.1f} minutes")
    
    if args.output:
        def serialize(obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            return obj
        
        with open(args.output, 'w') as f:
            json.dump(serialize(result), f, indent=2)
        print(f"Saved to: {args.output}")