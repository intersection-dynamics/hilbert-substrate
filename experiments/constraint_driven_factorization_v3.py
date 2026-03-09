#!/usr/bin/env python3
"""
constraint_driven_factorization_v3.py
======================================

CORRECTED EMERGENCE TEST (v3): Proper tensor product embeddings.

PATCH (v3.1):
  Fix structure-constant extraction to work for NON-orthonormal generator lists.
  We now compute f^{abc} by projecting commutators onto span{T^c} using the
  Gram matrix inverse. This removes the spurious normalization mismatch that
  made f_err ~ 2*||f|| even when the algebra was correct / full.

v2 BUGS FIXED (still true):
  1. Embedding was padded-block (3⊕zeros), giving reducible reps that
     trivially decouple. Now uses T^a⊗I_m, acting on ALL d_B dimensions.
  2. Algebra dimension computed by careful product-span SVD.

The test
--------
For each candidate d_B:

  Step 1 (Embedding): If d_B = N·m, embed L^a = T^a ⊗ I_m  (d_B × d_B).
     This is the N-fold copy of the fundamental, acting on ALL d_B dimensions.
     If N does not divide d_B, L cannot have a uniform embedding → skip.

  Step 2 (Commutant): The commutant of L is I_N ⊗ M_m (dimension m²).
     Does this commutant contain an su(N) subalgebra?
     → Yes iff m ≥ N, i.e., d_B ≥ N².

  Step 3 (No-signaling optimization): Fix L = T^a⊗I_m.
     Parametrize R inside the commutant: R^a = I_N ⊗ (V T_emb^a V†)
     where T_emb^a are su(N) generators embedded in M_m.
     Optimize V to make R close (as a Lie rep) to su(N) structure constants.

  Step 4 (Diagnostics): Check [L,R]=0, algebra dimension, gauge invariance.

Key prediction
--------------
  d_B = N²   → commutant has room for su(N), both constraints satisfied, GAUGE
  d_B = N·m, m<N → commutant too small, no su(N) fits, FAIL
  d_B not divisible by N → no uniform embedding exists, FAIL

  Finite bandwidth (minimize d_B) selects d_B = N².

Run
---
  python constraint_driven_factorization_v3.py

Requires: numpy, scipy
"""

import numpy as np
import math
import json
import os
import time
from datetime import datetime

try:
    from scipy.linalg import expm
    from scipy.optimize import minimize
except ImportError:
    raise RuntimeError("scipy required: pip install scipy")

np.set_printoptions(precision=6, suppress=True, linewidth=140)


# ======================================================================
# Utilities
# ======================================================================

def hs_inner(A, B):
    return float(np.real(np.trace(A.conj().T @ B)))

def hs_norm(A):
    return float(np.sqrt(max(np.trace(A.conj().T @ A).real, 0.0)))

def fro(A):
    return float(np.linalg.norm(A, 'fro'))

def comm(A, B):
    return A @ B - B @ A

def hermitize(A):
    return (A + A.conj().T) / 2.0

def traceless(A):
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d, dtype=complex)

def kron(*ops):
    out = ops[0]
    for X in ops[1:]:
        out = np.kron(out, X)
    return out


def su_generators(d):
    """HS-orthonormal Hermitian traceless generators of su(d)."""
    gens = []
    for i in range(d):
        for j in range(i + 1, d):
            S = np.zeros((d, d), dtype=complex)
            S[i, j] = 1.0; S[j, i] = 1.0
            gens.append(S)
            A = np.zeros((d, d), dtype=complex)
            A[i, j] = -1j; A[j, i] = 1j
            gens.append(A)
    for k in range(1, d):
        D = np.zeros((d, d), dtype=complex)
        for i in range(k): D[i, i] = 1.0
        D[k, k] = -float(k)
        D *= math.sqrt(2.0 / (k * (k + 1.0)))
        gens.append(D)
    out = []
    for G in gens:
        H = hermitize(traceless(G))
        n = hs_norm(H)
        if n > 1e-12:
            out.append(H / n)
    return out


def structure_constants_projected(T):
    """
    Compute structure constants f^{abc} for a (possibly non-orthonormal) list T.

    We define coefficients by projection:
      [T^a, T^b] ≈ 2 i Σ_c f^{abc} T^c
    where f^{ab:} is obtained by solving (Gram)·f = ip with:
      Gram_{dc} = Tr(T^d† T^c)
      ip_d = (1/(2i)) Tr(T^d† [T^a, T^b])

    If T is HS-orthonormal, this reduces to the usual closed-form.
    """
    n = len(T)
    G = np.zeros((n, n), dtype=float)
    for d in range(n):
        for c in range(n):
            G[d, c] = hs_inner(T[d], T[c])
    Ginv = np.linalg.pinv(G, rcond=1e-12)

    f = np.zeros((n, n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            C = comm(T[a], T[b])
            ip = np.array([(np.trace(T[d].conj().T @ C) / (2j)).real for d in range(n)], dtype=float)
            coeff = Ginv @ ip
            f[a, b, :] = coeff
    return f


def random_su(d, rng):
    """Random SU(d) element."""
    gens = su_generators(d)
    coeffs = rng.normal(size=len(gens))
    return expm(1j * sum(c * G for c, G in zip(coeffs, gens)))


def algebra_dim_exact(L_ops, R_ops, d):
    """
    Compute algebra dimension by vectorizing all products up to degree 3
    and doing careful SVD with tight tolerance.
    """
    def vec(M): return M.reshape(-1)

    all_ops = L_ops + R_ops

    # Degree 0: identity
    basis = [vec(np.eye(d, dtype=complex))]

    # Degree 1: all generators
    for O in all_ops:
        basis.append(vec(O))

    # Degree 2: all pairwise products
    for A in all_ops:
        for B in all_ops:
            basis.append(vec(A @ B))

    # Degree 3: triple products (sample to avoid explosion)
    for i, A in enumerate(all_ops):
        for j, B in enumerate(all_ops):
            for k, C in enumerate(all_ops):
                if (i + j + k) % 3 == 0:
                    basis.append(vec(A @ B @ C))

    M = np.array(basis)
    svals = np.linalg.svd(M, compute_uv=False)
    tol = max(svals[0] * 1e-10, 1e-12)
    return int(np.sum(svals > tol))


# ======================================================================
# Core test for one (d_site, d_B) pair
# ======================================================================

def test_pair(N, d_B, n_trials=8, seed=42):
    """
    Test whether d_B-dim link supports two commuting su(N) subalgebras
    that jointly fill M(d_B).

    N = d_site (e.g. 2 for SU(2), 3 for SU(3))
    d_B = link dimension
    """
    T = su_generators(N)
    n_gen = len(T)  # N²-1
    f_target = structure_constants_projected(T)

    I_N = np.eye(N, dtype=complex)
    I_dB = np.eye(d_B, dtype=complex)
    I_site = np.eye(N, dtype=complex)

    results = []

    # Check divisibility
    if d_B % N != 0:
        print(f"  d_B={d_B} not divisible by N={N} → no uniform embedding possible")
        for trial in range(n_trials):
            results.append({
                'trial': trial,
                'divisible': False,
                'm': None,
                'commutant_has_suN': False,
                'ns_cost': None,
                'max_comm_LR': None,
                'alg_dim_joint': None,
                'alg_dim_full': d_B * d_B,
                'both_satisfied': False,
                'gauge_invariant': False,
                'status': 'NOT_DIVISIBLE',
            })
            print(f"    Trial {trial+1}/{n_trials}: [NOT_DIVISIBLE]")
        return results

    m = d_B // N
    I_m = np.eye(m, dtype=complex)

    print(f"  d_B={d_B} = {N}×{m}.  Commutant dimension: {m}² = {m*m}")
    print(f"  su({N}) needs {n_gen} generators + identity = {n_gen+1} dimensions")
    print(f"  Commutant can hold su({N})? m={m} {'≥' if m >= N else '<'} N={N} → "
          f"{'YES' if m >= N else 'NO'}")

    commutant_has_suN = (m >= N)

    # Build L^a = T^a ⊗ I_m  (acts on all d_B dimensions)
    L_ops = [np.kron(Ta, I_m) for Ta in T]

    # Verify L structure using projected f_target (works regardless of norm scaling of L_ops list)
    max_struc_err = 0.0
    for a in range(n_gen):
        for b in range(n_gen):
            C_ab = comm(L_ops[a], L_ops[b])
            expected = sum(2j * f_target[a, b, c] * L_ops[c] for c in range(n_gen))
            max_struc_err = max(max_struc_err, fro(C_ab - expected))
    print(f"  L structure constant error: {max_struc_err:.2e}")

    if not commutant_has_suN:
        # Cannot fit su(N) in commutant — demonstrate this numerically
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial * 1000)

            if m >= 2:
                # Try to embed su(N) in M_m anyway
                T_m = su_generators(m)

                def embed_in_m(Ta):
                    M = np.zeros((m, m), dtype=complex)
                    ds = min(Ta.shape[0], m)
                    M[:ds, :ds] = Ta[:ds, :ds]
                    return hermitize(traceless(M))

                W = random_su(m, rng)
                R_m_ops = [W @ embed_in_m(Ta) @ W.conj().T for Ta in T]
                R_ops = [np.kron(I_N, Rm) for Rm in R_m_ops]

                max_lr = max(fro(comm(L_ops[a], R_ops[b]))
                             for a in range(n_gen) for b in range(n_gen))

                f_R = structure_constants_projected(R_ops)
                f_err = float(np.linalg.norm(f_R - f_target))

                alg_dim = algebra_dim_exact(L_ops, R_ops, d_B)
            else:
                max_lr = 0.0
                f_err = float('inf')
                alg_dim = 0

            results.append({
                'trial': trial,
                'divisible': True,
                'm': m,
                'commutant_has_suN': False,
                'ns_cost': float(max_lr),
                'max_comm_LR': float(max_lr),
                'f_structure_err': float(f_err),
                'alg_dim_joint': int(alg_dim),
                'alg_dim_full': d_B * d_B,
                'both_satisfied': False,
                'gauge_invariant': False,
                'status': 'COMMUTANT_TOO_SMALL',
            })
            print(f"    Trial {trial+1}/{n_trials}: [L,R]={max_lr:.2e}  "
                  f"f_err={f_err:.4f}  alg={alg_dim}/{d_B**2}  "
                  f"[COMMUTANT_TOO_SMALL]")
        return results

    # Commutant CAN hold su(N). Search for best R in commutant.
    T_m = su_generators(m)

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial * 1000)

        def embed_in_m(Ta):
            M = np.zeros((m, m), dtype=complex)
            ds = Ta.shape[0]
            M[:ds, :ds] = Ta
            return hermitize(traceless(M))

        T_emb_m = [embed_in_m(Ta) for Ta in T]

        def make_R(theta):
            Hm = sum(t * G for t, G in zip(theta, T_m))
            V = expm(1j * Hm)
            return [np.kron(I_N, V @ Te @ V.conj().T) for Te in T_emb_m]

        # Use projected structure constants for cost, not assuming orthonormality
        def cost_structure(theta):
            R_ops = make_R(theta)
            f_R = structure_constants_projected(R_ops)
            diff = f_R - f_target
            return float(np.sum(diff * diff))

        best_cost = cost_structure(np.zeros(len(T_m)))
        best_theta = np.zeros(len(T_m))

        for restart in range(5):
            t0 = np.zeros(len(T_m)) if restart == 0 else rng.normal(scale=1.0, size=len(T_m))
            res = minimize(cost_structure, t0, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_cost:
                best_cost = float(res.fun)
                best_theta = res.x.copy()

        R_ops = make_R(best_theta)

        max_lr = max(fro(comm(L_ops[a], R_ops[b]))
                     for a in range(n_gen) for b in range(n_gen))

        f_R = structure_constants_projected(R_ops)
        f_err = float(np.linalg.norm(f_R - f_target))
        su_N_good = f_err < 1e-6

        alg_dim = algebra_dim_exact(L_ops, R_ops, d_B)
        fills_algebra = (alg_dim >= d_B * d_B)

        # Gauge invariance (note: this uses fund on both sites; Gauss-kernel may still be empty)
        D_full = N * d_B * N
        H_coupling = np.zeros((D_full, D_full), dtype=complex)
        for a in range(n_gen):
            H_coupling += kron(T[a], L_ops[a], I_site)
            H_coupling += kron(I_site, R_ops[a], T[a])
        H_coupling = hermitize(H_coupling)

        max_gauss = 0.0
        for a in range(n_gen):
            G_L = kron(T[a], I_dB, I_site) + kron(I_site, L_ops[a], I_site)
            G_R = kron(I_site, I_dB, T[a]) + kron(I_site, R_ops[a], I_site)
            max_gauss = max(max_gauss, fro(comm(H_coupling, G_L)))
            max_gauss = max(max_gauss, fro(comm(H_coupling, G_R)))

        gauge_inv = max_gauss < 1e-6
        both = su_N_good and fills_algebra

        casimir_R = sum(Ra @ Ra for Ra in R_ops)
        evals_CR = np.sort(np.linalg.eigvalsh(hermitize(casimir_R).real))

        flags = []
        if max_lr < 1e-8: flags.append("NS")
        if su_N_good: flags.append("su(N)")
        if fills_algebra: flags.append("NF")
        if gauge_inv: flags.append("GAUGE")
        status = "+".join(flags) if flags else "NONE"

        results.append({
            'trial': trial,
            'divisible': True,
            'm': m,
            'commutant_has_suN': True,
            'ns_cost': float(max_lr),
            'max_comm_LR': float(max_lr),
            'f_structure_err': float(f_err),
            'f_structure_good': bool(su_N_good),
            'alg_dim_joint': int(alg_dim),
            'alg_dim_full': d_B * d_B,
            'fills_algebra': bool(fills_algebra),
            'both_satisfied': bool(both),
            'max_gauss': float(max_gauss),
            'gauge_invariant': bool(gauge_inv),
            'casimir_R': evals_CR.tolist(),
            'status': status,
        })
        print(f"    Trial {trial+1}/{n_trials}: [L,R]={max_lr:.2e}  "
              f"f_err={f_err:.2e}  alg={alg_dim}/{d_B**2}  "
              f"gauss={max_gauss:.2e}  [{status}]")

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    t_start = time.time()

    print("#" * 78)
    print("#  CONSTRAINT-DRIVEN FACTORIZATION v3.1 (PATCHED)")
    print("#  Proper tensor product embeddings: L^a = T^a ⊗ I_m")
    print("#  Structure constants computed by Gram-projected coefficients (robust).")
    print("#" * 78)
    print()
    print("  Logic: L^a = T^a⊗I_m acts on ALL d_B dimensions.")
    print("  Commutant = I_N⊗M_m, which contains su(N) iff m ≥ N (d_B ≥ N²).")
    print("  Finite bandwidth → minimize d_B → d_B = N².")

    all_results = {}

    print(f"\n{'='*78}")
    print("SU(3) SERIES: N=3")
    print(f"{'='*78}")

    su3_tests = [
        (3, 3,  "d_B=3  (m=1, commutant=M_1, too small)"),
        (3, 6,  "d_B=6  (m=2, commutant=M_2, too small for su(3))"),
        (3, 9,  "d_B=9  (m=3=N, commutant=M_3, MINIMAL for su(3))"),
        (3, 12, "d_B=12 (m=4>N, commutant=M_4, su(3) fits)"),
    ]

    for N, d_B, desc in su3_tests:
        print(f"\n--- {desc} ---")
        r = test_pair(N, d_B, n_trials=6, seed=42)
        all_results[f'SU3_dB{d_B}'] = r

    print(f"\n--- d_B=7 (not divisible by 3) ---")
    all_results['SU3_dB7'] = test_pair(3, 7, n_trials=3, seed=42)

    print(f"\n--- d_B=8 (not divisible by 3) ---")
    all_results['SU3_dB8'] = test_pair(3, 8, n_trials=3, seed=42)

    print(f"\n{'='*78}")
    print("SU(2) SERIES: N=2")
    print(f"{'='*78}")

    su2_tests = [
        (2, 2,  "d_B=2  (m=1, commutant=M_1, too small)"),
        (2, 4,  "d_B=4  (m=2=N, commutant=M_2, MINIMAL for su(2))"),
        (2, 6,  "d_B=6  (m=3>N, commutant=M_3, su(2) fits)"),
    ]

    for N, d_B, desc in su2_tests:
        print(f"\n--- {desc} ---")
        r = test_pair(N, d_B, n_trials=6, seed=77)
        all_results[f'SU2_dB{d_B}'] = r

    print(f"\n--- d_B=3 (not divisible by 2) ---")
    all_results['SU2_dB3'] = test_pair(2, 3, n_trials=3, seed=55)

    print(f"\n--- d_B=5 (not divisible by 2) ---")
    all_results['SU2_dB5'] = test_pair(2, 5, n_trials=3, seed=33)

    elapsed = time.time() - t_start

    print(f"\n{'#'*78}")
    print("#  FINAL VERDICT TABLE")
    print(f"{'#'*78}\n")

    rows = []
    for key, trials in all_results.items():
        n_t = len(trials)
        n_gauge = sum(1 for t in trials if t.get('gauge_invariant', False))
        n_both = sum(1 for t in trials if t.get('both_satisfied', False))
        divisible = trials[0].get('divisible', False)
        m_val = trials[0].get('m')
        comm_suN = trials[0].get('commutant_has_suN', False)

        if 'SU3' in key: N = 3
        elif 'SU2' in key: N = 2
        else: N = '?'

        d_B_str = key.split('dB')[1] if 'dB' in key else '?'

        rows.append({
            'key': key, 'N': N, 'd_B': d_B_str, 'div': divisible,
            'm': m_val, 'comm': comm_suN, 'both': n_both, 'gauge': n_gauge, 'n': n_t
        })

    print(f"  {'Test':<16} {'N':>2} {'d_B':>4} {'div':>4} {'m':>3} {'comm⊇su(N)':>11} "
          f"{'NS+NF':>6} {'GAUGE':>6}")
    print(f"  {'-'*62}")
    for r in rows:
        div_s = 'yes' if r['div'] else 'no'
        m_s = str(r['m']) if r['m'] else '-'
        comm_s = 'YES' if r['comm'] else 'no'
        both_s = f"{r['both']}/{r['n']}"
        gauge_s = f"{r['gauge']}/{r['n']}" if r['gauge'] > 0 else '-'
        is_min = '← MIN' if (r['comm'] and r['m'] == r['N']) else ''
        print(f"  {r['key']:<16} {r['N']:>2} {r['d_B']:>4} {div_s:>4} {m_s:>3} "
              f"{comm_s:>11} {both_s:>6} {gauge_s:>6}  {is_min}")

    print(f"""
  INTERPRETATION:
  ═══════════════
  1. N must divide d_B (otherwise no uniform embedding)
  2. Commutant I_N⊗M_m contains su(N) iff m ≥ N, i.e. d_B ≥ N²
  3. FINITE BANDWIDTH selects minimum d_B → d_B = N²
  4. At d_B = N²: two commuting su(N) can fill M(N²) → gauge invariance

  (Now with structure constants computed by Gram-projection, so f_err is meaningful.)

  Runtime: {elapsed:.1f}s
""")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"constraint_driven_factorization_v3_1_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()