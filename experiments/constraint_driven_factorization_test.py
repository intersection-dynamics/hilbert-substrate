#!/usr/bin/env python3
"""
constraint_driven_factorization_test.py  (REWRITE for Target B)
================================================================

Target B (strong target):
  Does the constraint engine (implemented here as "no-signaling optimization")
  drive a generic d_link=9 link toward:

    (i) true 3⊗3 tensor factorization on the link, and
    (ii) singlet-admitting 3⊗3bar vertex pairing (nonempty Gauss kernel),
         compatible with a gauge-invariant coupling Hamiltonian?

What this script does
---------------------
For each trial (d_link=9, d_site=3):
1) Randomly embed su(3) into the link as L^a (fixed).
2) Randomly embed su(3) into the link as R^a (variable embedding).
3) Optimize R embedding to minimize NO-SIGNALING cost:
       cost = Σ_{a,b} || [L^a, R^b] ||_F^2
4) After optimization:
   A) Commutativity: max ||[L^a, R^b]||
   B) Factorization signature via commutants:
      - dim Comm(L)  (complex dimension)
      - dim Comm(R)
      - dim Comm({L,R})  (should be ~1 for irreducible tensor factorization)
      For a true 3⊗3 decomposition with L ~ su(3)⊗I, Comm(L) should have dim 9.

   C) Gauss kernel existence (singlet-admitting pairing):
      We sweep representation conventions:
        site in {fund, anti}  where anti(T)=-(T^T)
        link-end in {fund, anti} applied to L or R in the Gauss generators
      We test if there exists ANY convention where:
        - H_coupling commutes with Gauss generators (numerical),
        - and dim ker(G^2) > 0.
      This directly tests the 3⊗3bar singlet condition.

Controls:
- d_link=8 is included as a *soft control*:
  it may find commuting subalgebras, but it should *not* show Comm(L)≈9
  for a 3⊗3 factorization, and typically Gauss-kernel will not match the
  composite-link singlet pattern.

Requires: numpy, scipy
Run:
  python constraint_driven_factorization_test.py
"""

import numpy as np
import math
import json
import time
import os
from datetime import datetime

try:
    from scipy.linalg import expm
    from scipy.optimize import minimize
except ImportError:
    raise RuntimeError("scipy required: pip install scipy")

np.set_printoptions(precision=8, suppress=True, linewidth=140)


# ======================================================================
# Utilities
# ======================================================================

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


# ======================================================================
# su(d) generators (HS-orthonormal, Hermitian, traceless)
# ======================================================================

def su_generators(d):
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
        for i in range(k):
            D[i, i] = 1.0
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


def random_su_element(d, rng):
    """Random element of SU(d) via exponential of random anti-Hermitian."""
    gens = su_generators(d)
    coeffs = rng.normal(size=len(gens))
    H = sum(c * G for c, G in zip(coeffs, gens))  # Hermitian
    return expm(1j * H)


def embed_site_gen(Ta, d_big):
    """Embed d_site×d_site generator into d_big×d_big (top-left block)."""
    M = np.zeros((d_big, d_big), dtype=complex)
    d_s = Ta.shape[0]
    M[:d_s, :d_s] = Ta
    return hermitize(traceless(M))


# ======================================================================
# Correct closure test (Hermitian basis): project Y=-i[T,T] into span
# ======================================================================

def closure_residual_correct(basis_ops):
    k = len(basis_ops)
    if k == 0:
        return 0.0

    # Gram and pseudo-inverse
    G = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            G[i, j] = float(np.real(np.trace(basis_ops[i].conj().T @ basis_ops[j])))
    Ginv = np.linalg.pinv(G, rcond=1e-12)

    def project(Y):
        ip = np.array([float(np.real(np.trace(basis_ops[i].conj().T @ Y))) for i in range(k)], dtype=float)
        alpha = Ginv @ ip
        P = np.zeros_like(Y)
        for i in range(k):
            P += alpha[i] * basis_ops[i]
        return P

    vals = []
    for a in range(k):
        for b in range(k):
            Y = (-1.0j) * comm(basis_ops[a], basis_ops[b])  # Hermitian
            P = project(Y)
            vals.append(fro(Y - P))
    return float(np.mean(np.array(vals, dtype=float)))


# ======================================================================
# Commutant dimension via linear nullspace of commutator constraints
# ======================================================================

def commutant_dimension(ops, tol=1e-8):
    """
    Compute complex dimension of commutant:
      Comm(ops) = { X : [X, A]=0 for all A in ops }.

    We represent X as vec(X) and enforce (I⊗A - A^T⊗I) vec(X)=0,
    stack for all A, then compute nullity via SVD.
    """
    if len(ops) == 0:
        return 0

    d = ops[0].shape[0]
    I = np.eye(d, dtype=complex)

    blocks = []
    for A in ops:
        K = np.kron(I, A) - np.kron(A.T, I)
        blocks.append(K)

    M = np.vstack(blocks)  # shape (m*d^2, d^2)

    # SVD on complex matrix: use np.linalg.svd (works on complex)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    # nullity = count singular values <= tol * s0
    s0 = s[0] if s.size else 1.0
    thresh = tol * s0
    nullity = int(np.sum(s <= thresh))
    return nullity


# ======================================================================
# Gauss kernel test: sweep rep conventions to find nonempty ker(G^2)
# ======================================================================

def rep_variant(Ta, which):
    """
    which in {'fund', 'anti'}:
      fund: Ta
      anti: -(Ta^T)  (anti-fund for Hermitian basis)
    """
    if which == "fund":
        return Ta
    if which == "anti":
        return -(Ta.T)
    raise ValueError("rep variant must be 'fund' or 'anti'")


def gauge_and_gauss_kernel_test(L_ops, R_ops, d_site, tol_comm=1e-8, tol_kernel=1e-8):
    """
    Build a two-leg coupling Hamiltonian and Gauss generators under all combinations:
      site_A_rep in {fund, anti}
      site_B_rep in {fund, anti}
      link_left_rep in {fund, anti} applied to L_ops
      link_right_rep in {fund, anti} applied to R_ops

    Check:
      - gauge invariance: max ||[H, G]|| < tol_comm
      - Gauss kernel: dim ker(G^2) > 0 under tol_kernel

    Returns best combo (max_comm, gauss_dim, smallest_eigs[:6]) and whether any passed.
    """
    T_site_f = su_generators(d_site)
    T_site_af = [-(X.T) for X in T_site_f]

    reps = ["fund", "anti"]

    I_site = np.eye(d_site, dtype=complex)
    d_link = L_ops[0].shape[0]
    I_link = np.eye(d_link, dtype=complex)
    D_full = d_site * d_link * d_site

    best = None

    for repA in reps:
        for repB in reps:
            T_A = T_site_f if repA == "fund" else T_site_af
            T_B = T_site_f if repB == "fund" else T_site_af

            for repL in reps:
                for repR in reps:
                    # Apply rep choice to link endpoint generators
                    L_rep = [rep_variant(L_ops[a], repL) for a in range(len(T_site_f))]
                    R_rep = [rep_variant(R_ops[a], repR) for a in range(len(T_site_f))]

                    # Build coupling Hamiltonian
                    H = np.zeros((D_full, D_full), dtype=complex)
                    for a in range(len(T_site_f)):
                        H += kron(T_A[a], L_rep[a], I_site)
                        H += kron(I_site, R_rep[a], T_B[a])
                    H = hermitize(H)

                    # Build Gauss generators
                    max_comm = 0.0
                    GL = []
                    GR = []
                    for a in range(len(T_site_f)):
                        G_L = kron(T_A[a], I_link, I_site) + kron(I_site, L_rep[a], I_site)
                        G_R = kron(I_site, R_rep[a], I_site) + kron(I_site, I_link, T_B[a])
                        GL.append(G_L)
                        GR.append(G_R)
                        max_comm = max(max_comm, fro(comm(H, G_L)))
                        max_comm = max(max_comm, fro(comm(H, G_R)))

                    # G^2 and kernel size
                    G2 = np.zeros((D_full, D_full), dtype=complex)
                    for a in range(len(T_site_f)):
                        G2 += GL[a] @ GL[a] + GR[a] @ GR[a]
                    G2 = hermitize(G2)
                    evals = np.linalg.eigvalsh(G2.real)
                    gauss_dim = int(np.sum(np.abs(evals) < tol_kernel))

                    ok_gauge = (max_comm < tol_comm)
                    ok_kernel = (gauss_dim > 0)

                    rec = {
                        "repA": repA, "repB": repB,
                        "repL": repL, "repR": repR,
                        "max_comm": float(max_comm),
                        "gauss_dim": int(gauss_dim),
                        "smallest_eigs": [float(x) for x in evals[:6]],
                        "ok_gauge": bool(ok_gauge),
                        "ok_kernel": bool(ok_kernel),
                        "passes": bool(ok_gauge and ok_kernel),
                    }

                    if best is None:
                        best = rec
                    else:
                        # rank by: passes, then larger gauss_dim, then smaller max_comm, then smaller smallest eig
                        def key(r):
                            return (r["passes"], r["gauss_dim"], -r["max_comm"])
                        if key(rec) > key(best):
                            best = rec

    return best


# ======================================================================
# Core test: no-signaling optimization and B-target diagnostics
# ======================================================================

def no_signaling_optimization(d_link, d_site, n_trials=12, seed=42):
    """
    For each trial:
      - Random L embedding fixed
      - Random R embedding optimized to commute with L (no-signaling)
      - Then check:
          (B1) commutant signature for 3⊗3 on d=9
          (B2) existence of gauge+Gauss-kernel pairing (3⊗3bar)
    """
    rng = np.random.default_rng(seed)
    T_site = su_generators(d_site)
    T_link = su_generators(d_link)
    n_site_gen = len(T_site)
    n_link_gen = len(T_link)

    results = []

    for trial in range(n_trials):
        trial_seed = seed + trial * 1000
        rng_trial = np.random.default_rng(trial_seed)

        W_L = random_su_element(d_link, rng_trial)
        W_R_init = random_su_element(d_link, rng_trial)

        T_embedded = [embed_site_gen(Ta, d_link) for Ta in T_site]

        # L^a fixed
        L_ops = [W_L @ Te @ W_L.conj().T for Te in T_embedded]

        # R^a(theta)
        def make_R_ops(theta):
            H_param = sum(t * G for t, G in zip(theta, T_link))
            V = W_R_init @ expm(1j * H_param)
            return [V @ Te @ V.conj().T for Te in T_embedded]

        def cost_no_signaling(theta):
            R_ops = make_R_ops(theta)
            total = 0.0
            for a in range(n_site_gen):
                for b in range(n_site_gen):
                    C = comm(L_ops[a], R_ops[b])
                    total += np.sum(np.abs(C) ** 2).real
            return float(total)

        theta0 = np.zeros(n_link_gen)
        cost0 = cost_no_signaling(theta0)

        best_cost = cost0
        best_theta = theta0.copy()

        for restart in range(3):
            t0 = np.zeros(n_link_gen) if restart == 0 else rng_trial.normal(scale=0.5, size=n_link_gen)
            res = minimize(cost_no_signaling, t0, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_cost:
                best_cost = float(res.fun)
                best_theta = res.x.copy()

        R_ops_final = make_R_ops(best_theta)

        # Commutativity diagnostic
        max_comm = 0.0
        for a in range(n_site_gen):
            for b in range(n_site_gen):
                max_comm = max(max_comm, fro(comm(L_ops[a], R_ops_final[b])))

        # Factorization signature: commutant dimensions
        dim_comm_L = commutant_dimension(L_ops, tol=1e-8)
        dim_comm_R = commutant_dimension(R_ops_final, tol=1e-8)
        dim_comm_both = commutant_dimension(L_ops + R_ops_final, tol=1e-8)

        # Closure sanity (not required, but informative)
        closure_L = closure_residual_correct(L_ops)
        closure_R = closure_residual_correct(R_ops_final)

        # Gauss kernel test (sweeps rep conventions to find any singlet-admitting pairing)
        best_gauss = gauge_and_gauss_kernel_test(L_ops, R_ops_final, d_site=d_site, tol_comm=1e-8, tol_kernel=1e-8)

        results.append({
            "trial": trial,
            "cost_initial": float(cost0),
            "cost_final": float(best_cost),
            "max_comm_LR": float(max_comm),
            "closure_L_mean": float(closure_L),
            "closure_R_mean": float(closure_R),
            "commutant_dims": {
                "Comm(L)_dim": int(dim_comm_L),
                "Comm(R)_dim": int(dim_comm_R),
                "Comm(L,R)_dim": int(dim_comm_both),
            },
            "gauss_best": best_gauss,
        })

        # Print status line
        commute_ok = best_cost < 1e-10
        # For true 3⊗3 with irreducible su(3) action, expect Comm(L)≈9 and Comm(L,R)≈1
        factor_ok = (d_link == 9 and dim_comm_L >= 9 and dim_comm_R >= 9 and dim_comm_both <= 2)
        gauss_ok = bool(best_gauss["passes"])

        status = []
        if commute_ok: status.append("COMMUTE")
        if factor_ok: status.append("FACTORSIG")
        if gauss_ok: status.append("GAUSS_KERNEL")
        if not status: status.append("no")

        print(f"    Trial {trial+1:>2}/{n_trials}: cost {cost0:.2e} -> {best_cost:.2e}  "
              f"max[L,R]={max_comm:.2e}  Comm(L)={dim_comm_L} Comm(R)={dim_comm_R} Comm(L,R)={dim_comm_both}  "
              f"GaussBest(dim={best_gauss['gauss_dim']}, maxComm={best_gauss['max_comm']:.1e}, reps={best_gauss['repA']}/{best_gauss['repL']} | {best_gauss['repR']}/{best_gauss['repB']})  "
              f"[{'+'.join(status)}]")

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    t_start = time.time()

    print("#" * 78)
    print("#  CONSTRAINT-DRIVEN FACTORIZATION (Target B)")
    print("#  Does no-signaling drive toward 3⊗3 AND singlet-admitting 3⊗3bar Gauss kernel?")
    print("#" * 78)

    all_results = {}

    # TEST: d_link = 9, d_site = 3
    print(f"\n{'='*78}")
    print("TEST: d_link = 9, d_site = 3")
    print("  Target B success requires:")
    print("   (1) [L,R]≈0  (no-signaling)")
    print("   (2) Comm(L)≈9 and Comm(L,R)≈1  (3⊗3 factorization signature)")
    print("   (3) exists rep pairing with Gauss kernel dim>0 (3⊗3bar singlet at vertices)")
    print(f"{'='*78}\n")

    r9 = no_signaling_optimization(d_link=9, d_site=3, n_trials=12, seed=42)
    all_results["d9"] = r9

    n_commute = sum(1 for r in r9 if r["cost_final"] < 1e-10)
    n_factor = sum(1 for r in r9 if (r["commutant_dims"]["Comm(L)_dim"] >= 9 and r["commutant_dims"]["Comm(R)_dim"] >= 9 and r["commutant_dims"]["Comm(L,R)_dim"] <= 2))
    n_gauss = sum(1 for r in r9 if r["gauss_best"]["passes"])

    print(f"\nSUMMARY d_link=9:")
    print(f"  commute successes: {n_commute}/{len(r9)}")
    print(f"  factorization-signature successes (Comm dims): {n_factor}/{len(r9)}")
    print(f"  Gauss-kernel successes (some 3⊗3bar pairing exists): {n_gauss}/{len(r9)}")

    # SOFT CONTROL: d_link = 8 (not N^2)
    print(f"\n{'='*78}")
    print("SOFT CONTROL: d_link = 8, d_site = 3")
    print("  This may still find commuting subalgebras (they can hide on subspaces),")
    print("  but it should NOT show the 3⊗3 commutant signature Comm(L)≈9.")
    print(f"{'='*78}\n")

    r8 = no_signaling_optimization(d_link=8, d_site=3, n_trials=6, seed=99)
    all_results["d8"] = r8
    n_commute_8 = sum(1 for r in r8 if r["cost_final"] < 1e-10)
    n_factor_8 = sum(1 for r in r8 if (r["commutant_dims"]["Comm(L)_dim"] >= 9 and r["commutant_dims"]["Comm(L,R)_dim"] <= 2))
    n_gauss_8 = sum(1 for r in r8 if r["gauss_best"]["passes"])

    print(f"\nSUMMARY d_link=8:")
    print(f"  commute successes: {n_commute_8}/{len(r8)}")
    print(f"  factorization-signature successes (should be ~0): {n_factor_8}/{len(r8)}")
    print(f"  Gauss-kernel successes: {n_gauss_8}/{len(r8)}")

    elapsed = time.time() - t_start

    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"constraint_driven_factorization_{tag}.json")

    def clean(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    summary = {
        "target": "B",
        "criteria": {
            "commute_cost_thresh": 1e-10,
            "commutant_signature": "for 3⊗3: Comm(L) dim≈9 and Comm(L,R) dim≈1",
            "gauss_kernel": "exists rep pairing with dim ker(G^2)>0",
        },
        "aggregate": {
            "d9": {"commute": n_commute, "factor_sig": n_factor, "gauss_kernel": n_gauss, "n_trials": len(r9)},
            "d8": {"commute": n_commute_8, "factor_sig": n_factor_8, "gauss_kernel": n_gauss_8, "n_trials": len(r8)},
        },
        "details": clean(all_results),
        "runtime_s": elapsed,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRuntime: {elapsed:.1f}s")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()