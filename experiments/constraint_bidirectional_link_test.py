#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constraint_bidirectional_link_test.py
====================================

Goal (Target B, generalized):
-----------------------------
Test whether the constraint engine (here: no-signaling commutativity optimizer)
naturally yields a *bidirectional link Hilbert space* in the strong algebraic sense:

  H_link ≅ V ⊗ Vbar    with independent tensor factors,

operationalized as:
  - Two commuting endpoint actions L^a, R^a on H_link
  - Commutant signature for a true tensor-factor action:
        dim Comm(L)   = N^2
        dim Comm(R)   = N^2
        dim Comm(L,R) = 1
    where d_link = N^2.

Also test (independently) whether the singlet-admitting conjugate vertex pairing exists:
  site_A = anti-fund, link_left = fund  -> (Vbar ⊗ V) contains singlet
  link_right = anti-fund, site_B = fund -> (Vbar ⊗ V) contains singlet

and compute dim ker(G^2) for the corresponding Gauss generators.

Important:
----------
- This does NOT add "Gauss kernel pressure" or "irreducibility pressure" to the optimizer.
- It simply checks whether those features appear *as a consequence* of no-signaling optimization.

Dependencies: numpy, scipy

Examples:
---------
python constraint_bidirectional_link_test.py --d_link 9 --trials 12
python constraint_bidirectional_link_test.py --scan_squares 4,9,16,25 --trials 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.linalg import expm
    from scipy.optimize import minimize
except Exception as e:
    raise RuntimeError("This script requires scipy (scipy.linalg.expm, scipy.optimize.minimize).") from e


# =============================================================================
# Basic linear algebra helpers
# =============================================================================

def hermitize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)

def traceless(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d, dtype=complex)

def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))

def hs_inner(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.real(np.trace(A.conj().T @ B)))

def hs_norm(A: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, hs_inner(A, A))))

def normalize_hs(A: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = hs_norm(A)
    return A.copy() if n < eps else (A / n)

def kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for X in ops[1:]:
        out = np.kron(out, X)
    return out


# =============================================================================
# su(d) Hermitian HS-orthonormal generators (fundamental)
# =============================================================================

def su_generators(d: int) -> List[np.ndarray]:
    gens = []

    # symmetric and antisymmetric off-diagonals
    for i in range(d):
        for j in range(i + 1, d):
            S = np.zeros((d, d), dtype=complex)
            S[i, j] = 1.0
            S[j, i] = 1.0
            gens.append(S)

            A = np.zeros((d, d), dtype=complex)
            A[i, j] = -1j
            A[j, i] = 1j
            gens.append(A)

    # diagonal traceless
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
        out.append(normalize_hs(H))
    return out


def anti_fund(Ta: np.ndarray) -> np.ndarray:
    """Anti-fund rep for Hermitian generators: Tbar^a = -(T^a)^T"""
    return -(Ta.T)


# =============================================================================
# Random SU(d) element
# =============================================================================

def random_su_element(d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build a random unitary (close to Haar-ish) via exp(iH) where H is random Hermitian.
    """
    basis = su_generators(d)
    coeffs = rng.normal(size=len(basis))
    H = np.zeros((d, d), dtype=complex)
    for c, G in zip(coeffs, basis):
        H += c * G
    return expm(1j * hermitize(traceless(H)))


# =============================================================================
# Embed site generators into the link space (top-left N×N block)
# =============================================================================

def embed_site_gen_into_link(Ta: np.ndarray, d_link: int) -> np.ndarray:
    """
    Place Ta (N×N) into top-left corner of d_link×d_link and traceless/hermitize.
    This is a generic embedding used for the no-signaling optimization.
    """
    N = Ta.shape[0]
    M = np.zeros((d_link, d_link), dtype=complex)
    M[:N, :N] = Ta
    return hermitize(traceless(M))


# =============================================================================
# Commutant dimension: nullspace of commutator constraints (complex dimension)
# =============================================================================

def commutant_dimension(ops: List[np.ndarray], tol: float = 1e-8) -> int:
    """
    Comm(ops) = {X : [X,A]=0 ∀A∈ops}.
    Linear constraints on vec(X):
      (I⊗A - A^T⊗I) vec(X) = 0
    Stack and compute nullity.
    """
    if not ops:
        return 0
    d = ops[0].shape[0]
    I = np.eye(d, dtype=complex)

    blocks = []
    for A in ops:
        blocks.append(np.kron(I, A) - np.kron(A.T, I))
    M = np.vstack(blocks)

    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s0 = s[0] if s.size else 1.0
    thresh = tol * s0
    nullity = int(np.sum(s <= thresh))
    return nullity


# =============================================================================
# No-signaling optimization: find R embedding that commutes with fixed L embedding
# =============================================================================

def optimize_commuting_endpoints(N: int, d_link: int, rng: np.random.Generator,
                                 restarts: int = 3, maxiter: int = 2000) -> Dict:
    """
    Choose fixed L embedding and optimize R embedding to minimize:
      cost = Σ_{a,b} ||[L^a, R^b]||_F^2
    where L^a, R^b are embedded copies of su(N) generators in d_link.
    """
    T_site = su_generators(N)
    T_link_basis = su_generators(d_link)

    T_emb = [embed_site_gen_into_link(Ta, d_link) for Ta in T_site]
    k = len(T_site)
    m = len(T_link_basis)

    W_L = random_su_element(d_link, rng)
    W_R0 = random_su_element(d_link, rng)

    L_ops = [W_L @ Te @ W_L.conj().T for Te in T_emb]

    def make_R_ops(theta: np.ndarray) -> List[np.ndarray]:
        H = np.zeros((d_link, d_link), dtype=complex)
        for t, G in zip(theta, T_link_basis):
            H += t * G
        V = W_R0 @ expm(1j * hermitize(traceless(H)))
        return [V @ Te @ V.conj().T for Te in T_emb]

    def cost(theta: np.ndarray) -> float:
        R_ops = make_R_ops(theta)
        tot = 0.0
        for a in range(k):
            for b in range(k):
                C = comm(L_ops[a], R_ops[b])
                tot += float(np.sum(np.abs(C) ** 2).real)
        return float(tot)

    theta0 = np.zeros(m, dtype=float)
    best_theta = theta0.copy()
    best_cost = cost(theta0)
    best_res = None

    for r in range(restarts):
        start = theta0 if r == 0 else rng.normal(scale=0.5, size=m)
        res = minimize(cost, start, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-15, "gtol": 1e-12})
        if float(res.fun) < best_cost:
            best_cost = float(res.fun)
            best_theta = res.x.copy()
            best_res = res

    R_ops = make_R_ops(best_theta)

    # summary comm stats
    max_comm = 0.0
    for a in range(k):
        for b in range(k):
            max_comm = max(max_comm, fro(comm(L_ops[a], R_ops[b])))

    return {
        "L_ops": L_ops,
        "R_ops": R_ops,
        "cost_final": best_cost,
        "max_comm_LR": float(max_comm),
        "optimizer": {
            "restarts": int(restarts),
            "maxiter": int(maxiter),
            "best_nit": int(best_res.nit) if best_res is not None else None,
            "best_success": bool(best_res.success) if best_res is not None else None,
        }
    }


# =============================================================================
# Gauss kernel test for the conjugate pairing (siteA anti, linkL fund, linkR anti, siteB fund)
# =============================================================================

def gauss_kernel_for_conjugate_pairing(N: int, L_ops: List[np.ndarray], R_ops: List[np.ndarray],
                                       tol_comm: float = 1e-8, tol_kernel: float = 1e-8) -> Dict:
    """
    Use the *specific* singlet-admitting pairing:
      site_A: anti-fund (Tbar)
      link_left: fund action (use L_ops)
      link_right: anti-fund action (use anti(R_ops))
      site_B: fund (T)

    Build H = Σ_a [ T_A^a ⊗ L^a ⊗ I + I ⊗ Rbar^a ⊗ T_B^a ]
    and Gauss generators:
      G_L^a = T_A^a ⊗ I ⊗ I + I ⊗ L^a ⊗ I
      G_R^a = I ⊗ Rbar^a ⊗ I + I ⊗ I ⊗ T_B^a

    Return:
      - gauge invariance commutators
      - dim ker(G^2)
      - smallest eigenvalues of G^2
    """
    d_link = L_ops[0].shape[0]
    T_f = su_generators(N)
    T_a = [anti_fund(X) for X in T_f]

    I_site = np.eye(N, dtype=complex)
    I_link = np.eye(d_link, dtype=complex)

    k = len(T_f)
    D = N * d_link * N

    # Represent link-right as anti-fund relative to R_ops
    Rbar_ops = [anti_fund(X) for X in R_ops]  # -(R^T)

    # Build coupling Hamiltonian
    H = np.zeros((D, D), dtype=complex)
    for a in range(k):
        H += kron(T_a[a], L_ops[a], I_site)
        H += kron(I_site, Rbar_ops[a], T_f[a])
    H = hermitize(H)

    # Build Gauss generators and gauge commutators
    max_comm = 0.0
    GL = []
    GR = []
    for a in range(k):
        G_L = kron(T_a[a], I_link, I_site) + kron(I_site, L_ops[a], I_site)
        G_R = kron(I_site, Rbar_ops[a], I_site) + kron(I_site, I_link, T_f[a])
        GL.append(G_L)
        GR.append(G_R)
        max_comm = max(max_comm, fro(comm(H, G_L)))
        max_comm = max(max_comm, fro(comm(H, G_R)))

    ok_gauge = (max_comm < tol_comm)

    # G^2 and kernel
    G2 = np.zeros((D, D), dtype=complex)
    for a in range(k):
        G2 += GL[a] @ GL[a] + GR[a] @ GR[a]
    G2 = hermitize(G2)
    evals = np.linalg.eigvalsh(G2.real)
    gauss_dim = int(np.sum(np.abs(evals) < tol_kernel))

    return {
        "max_comm_H_G": float(max_comm),
        "passes_gauge_invariance": bool(ok_gauge),
        "gauss_dim": int(gauss_dim),
        "smallest_eigs": [float(x) for x in evals[:min(10, evals.size)]],
        "D_total": int(D),
        "tol_comm": float(tol_comm),
        "tol_kernel": float(tol_kernel),
    }


# =============================================================================
# Driver
# =============================================================================

def perfect_square_root(d: int) -> Optional[int]:
    r = int(round(math.sqrt(d)))
    return r if r * r == d else None


def run_for_d_link(d_link: int, trials: int, seed: int, restarts: int, maxiter: int) -> Dict:
    N = perfect_square_root(d_link)
    if N is None:
        return {"d_link": d_link, "ok": False, "reason": "d_link is not a perfect square; no N⊗N factorization possible."}

    rng = np.random.default_rng(seed)
    per_trial = []

    for t in range(trials):
        rng_t = np.random.default_rng(seed + 1000 * t + 7)

        opt = optimize_commuting_endpoints(N=N, d_link=d_link, rng=rng_t, restarts=restarts, maxiter=maxiter)
        L_ops = opt["L_ops"]
        R_ops = opt["R_ops"]

        # Commutant signature
        dim_comm_L = commutant_dimension(L_ops, tol=1e-8)
        dim_comm_R = commutant_dimension(R_ops, tol=1e-8)
        dim_comm_both = commutant_dimension(L_ops + R_ops, tol=1e-8)

        # Strong factorization success condition
        factor_sig_ok = (dim_comm_L == N * N) and (dim_comm_R == N * N) and (dim_comm_both == 1)

        # Gauss kernel test for conjugate pairing
        gauss = gauss_kernel_for_conjugate_pairing(N=N, L_ops=L_ops, R_ops=R_ops, tol_comm=1e-8, tol_kernel=1e-8)

        per_trial.append({
            "trial": t,
            "N": N,
            "cost_final": opt["cost_final"],
            "max_comm_LR": opt["max_comm_LR"],
            "commutants": {
                "Comm(L)": dim_comm_L,
                "Comm(R)": dim_comm_R,
                "Comm(L,R)": dim_comm_both,
                "expected_Comm(L)": N * N,
                "expected_Comm(R)": N * N,
                "expected_Comm(L,R)": 1,
                "factor_sig_ok": bool(factor_sig_ok),
            },
            "gauss_conjugate_pairing": gauss,
            "optimizer": opt["optimizer"],
        })

        tag = []
        if opt["cost_final"] < 1e-10: tag.append("COMMUTE")
        if factor_sig_ok: tag.append("FACTORSIG")
        if gauss["passes_gauge_invariance"] and gauss["gauss_dim"] > 0: tag.append("GAUSS_KERNEL")
        if not tag: tag.append("no")
        print(f"  Trial {t+1:>2}/{trials}  d_link={d_link} (N={N})  "
              f"cost={opt['cost_final']:.2e}  max[L,R]={opt['max_comm_LR']:.2e}  "
              f"Comm(L)={dim_comm_L} Comm(R)={dim_comm_R} Comm(L,R)={dim_comm_both}  "
              f"Gauss(dim={gauss['gauss_dim']}, maxComm={gauss['max_comm_H_G']:.1e})  "
              f"[{'+'.join(tag)}]")

    # Aggregate
    n_commute = sum(1 for r in per_trial if r["cost_final"] < 1e-10)
    n_factor = sum(1 for r in per_trial if r["commutants"]["factor_sig_ok"])
    n_gauss = sum(1 for r in per_trial if (r["gauss_conjugate_pairing"]["passes_gauge_invariance"] and r["gauss_conjugate_pairing"]["gauss_dim"] > 0))

    return {
        "d_link": d_link,
        "N": N,
        "ok": True,
        "aggregate": {
            "trials": trials,
            "commute_successes": n_commute,
            "factor_signature_successes": n_factor,
            "gauss_kernel_successes": n_gauss,
        },
        "trials": per_trial,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_link", type=int, default=9, help="Link dimension to test (must be perfect square for N⊗N).")
    ap.add_argument("--scan_squares", type=str, default="", help="Comma list like 4,9,16,25 to test multiple d_link values.")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=2000)
    args = ap.parse_args()

    d_links = []
    if args.scan_squares.strip():
        d_links = [int(x.strip()) for x in args.scan_squares.split(",") if x.strip()]
    else:
        d_links = [int(args.d_link)]

    print("=" * 78)
    print("BIDIRECTIONAL LINK TEST (Target B, generalized to any N⊗N)")
    print("Criteria:")
    print("  - COMMUTE: no-signaling optimizer achieves [L,R]≈0")
    print("  - FACTORSIG: Comm(L)=N^2, Comm(R)=N^2, Comm(L,R)=1 (d_link=N^2)")
    print("  - GAUSS_KERNEL: conjugate pairing yields dim ker(G^2)>0")
    print("=" * 78)

    all_results = []
    t0 = time.time()

    for d_link in d_links:
        print(f"\nTesting d_link={d_link} ...")
        res = run_for_d_link(
            d_link=d_link,
            trials=int(args.trials),
            seed=int(args.seed),
            restarts=int(args.restarts),
            maxiter=int(args.maxiter),
        )
        all_results.append(res)

        if res.get("ok"):
            agg = res["aggregate"]
            print(f"SUMMARY d_link={d_link} (N={res['N']}): "
                  f"commute {agg['commute_successes']}/{agg['trials']}, "
                  f"factorSig {agg['factor_signature_successes']}/{agg['trials']}, "
                  f"GaussKernel {agg['gauss_kernel_successes']}/{agg['trials']}")
        else:
            print(f"SKIP d_link={d_link}: {res.get('reason')}")

    elapsed = time.time() - t0

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"constraint_bidirectional_link_{tag}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"runtime_s": elapsed, "results": all_results}, f, indent=2)

    print(f"\nRuntime: {elapsed:.1f}s")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()