#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_link_qutrit_npz_diagnostic_v6.py
=====================================

Purpose
-------
Diagnose qutrit-link extracted endpoint generators L[a], R[a] (8 of 3x3 each)
from your NPZ (aligned/mixed). This script focuses on the key structural issue:

- A genuine gauge link needs two commuting endpoint actions on the link:
    [L^a, R^b] = 0  for all a,b
  With link Hilbert space C^3 and both acting nontrivially on the same space,
  this is generically impossible.

What this script reports
------------------------
A) Basic su(3) closure + invariants for each side (L and R).
B) Cross-endpoint commutator stats: norms of [L^a, R^b].
C) "Conjugate rep" sanity: compare R to -L^T under best generator-space rotation.
D) Virtual link-promotion test:
   - Promote link to 9D: H_E ⊗ H_E
   - Define L_eff[a] = L[a] ⊗ I
            R_eff[a] = I ⊗ (-R[a].T)
     (right endpoint in anti-fund)
   - Build singlet projectors at vertices (like v5) and test Gauss commutators.
   This answers: "If the link had room (dim 9), would these endpoints be gauge-capable?"

Usage (Windows one-liner)
-------------------------
python gauss_link_qutrit_npz_diagnostic_v6.py --npz echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz --mode aligned
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.linalg import expm
    from scipy.optimize import minimize
except Exception as e:
    raise RuntimeError("This script requires scipy (scipy.linalg.expm, scipy.optimize.minimize).") from e


# -------------------------
# Utilities
# -------------------------

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def hermitize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)


def traceless(A: np.ndarray) -> np.ndarray:
    return A - np.trace(A) * np.eye(A.shape[0], dtype=A.dtype) / A.shape[0]


def hs_inner(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.real(np.trace(A.conj().T @ B)))


def hs_norm(A: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, hs_inner(A, A))))


def normalize_hs(A: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = hs_norm(A)
    if n < eps:
        return A.copy()
    return A / n


def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def kron(*args: np.ndarray) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for A in args:
        out = np.kron(out, A)
    return out


# -------------------------
# su(3) basis (HS-orthonormal)
# -------------------------

def su_generators_gellmann(d: int) -> List[np.ndarray]:
    gens: List[np.ndarray] = []

    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = 1.0
            M[j, i] = 1.0
            gens.append(M)

    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = -1.0j
            M[j, i] = 1.0j
            gens.append(M)

    for k in range(1, d):
        M = np.zeros((d, d), dtype=complex)
        for i in range(k):
            M[i, i] = 1.0
        M[k, k] = -float(k)
        gens.append(M)

    out = []
    for G in gens:
        X = traceless(hermitize(G))
        out.append(X / max(1e-15, hs_norm(X)))

    if len(out) != d * d - 1:
        raise RuntimeError("bad generator count")
    return out


# -------------------------
# Structure constants + adjoint rep
# -------------------------

def structure_constants(T: List[np.ndarray]) -> np.ndarray:
    n = len(T)
    f = np.zeros((n, n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            C = comm(T[a], T[b])
            for c in range(n):
                val = (1.0 / (2.0j)) * np.trace(C @ T[c])
                f[a, b, c] = float(np.real(val))
    return f


def adjoint_matrices(f: np.ndarray) -> List[np.ndarray]:
    return [np.array(f[a, :, :], dtype=float) for a in range(f.shape[0])]


def f_invariants(f: np.ndarray) -> Dict[str, float]:
    return {
        "f_fro": float(np.linalg.norm(f.ravel())),
        "f_maxabs": float(np.max(np.abs(f))),
    }


# -------------------------
# Generator-space SO(8) alignment
# -------------------------

def params_dim(n: int) -> int:
    return n * (n - 1) // 2


def skew_from_params(p: np.ndarray, n: int) -> np.ndarray:
    K = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = p[idx]
            K[j, i] = -p[idx]
            idx += 1
    return K


def align_adjoint(A_list: List[np.ndarray], B_list: List[np.ndarray],
                  restarts: int, maxiter: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    n = A_list[0].shape[0]
    A = np.stack(A_list, axis=0)
    B = np.stack(B_list, axis=0)
    k = A.shape[0]

    def objective(p: np.ndarray) -> float:
        K = skew_from_params(p, n)
        O = expm(K)
        OT = O.T
        s = 0.0
        for i in range(k):
            D = A[i] - (O @ B[i] @ OT)
            s += float(np.sum(np.abs(D) ** 2))
        return s

    best = {"val": float("inf"), "O": np.eye(n), "nit": None, "success": False}

    starts: List[np.ndarray] = [np.zeros(params_dim(n), dtype=float)]
    for _ in range(max(0, restarts - 1)):
        starts.append(rng.normal(scale=0.25, size=params_dim(n)))

    for p0 in starts:
        res = minimize(objective, p0, method="L-BFGS-B", options={"maxiter": int(maxiter), "ftol": 1e-12})
        val = float(res.fun)
        if val < best["val"]:
            K = skew_from_params(res.x, n)
            O = expm(K)
            U, _, Vt = np.linalg.svd(O, full_matrices=False)
            O = U @ Vt
            best = {"val": val, "O": O, "nit": int(res.nit), "success": bool(res.success)}
    return best


def rotate_generators(T: List[np.ndarray], O: np.ndarray) -> List[np.ndarray]:
    # NO Gram–Schmidt: preserve meaning of indices
    n = len(T)
    out: List[np.ndarray] = []
    for a in range(n):
        M = np.zeros_like(T[0])
        for i in range(n):
            M += float(O[a, i]) * T[i]
        out.append(normalize_hs(traceless(hermitize(M))))
    return out


# -------------------------
# Load NPZ bases
# -------------------------

def load_npz_bases(path: str, mode: str) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    if mode == "aligned":
        kL = "basis_left_aligned"
        kR = "basis_right_aligned"
    elif mode == "mixed":
        kL = "basis_left_mixed"
        kR = "basis_right_mixed"
    else:
        raise ValueError("mode must be aligned or mixed")

    if kL not in data or kR not in data:
        raise KeyError(f"NPZ missing keys. Needed {kL},{kR}. Present keys: {keys}")

    L_arr = data[kL]
    R_arr = data[kR]

    if L_arr.shape != (8, 3, 3) or R_arr.shape != (8, 3, 3):
        raise ValueError(f"Unexpected shapes: L={L_arr.shape} R={R_arr.shape} (expected (8,3,3))")

    L = [normalize_hs(traceless(hermitize(np.array(L_arr[a], dtype=complex)))) for a in range(8)]
    R = [normalize_hs(traceless(hermitize(np.array(R_arr[a], dtype=complex)))) for a in range(8)]

    meta = {"npz_keys": keys, "used_keys": {"L": kL, "R": kR}}
    return L, R, meta


# -------------------------
# Diagnostics
# -------------------------

def commutator_grid_stats(L: List[np.ndarray], R: List[np.ndarray]) -> Dict[str, float]:
    vals = []
    for a in range(len(L)):
        for b in range(len(R)):
            vals.append(hs_norm(comm(L[a], R[b])))
    v = np.array(vals, dtype=float)
    return {
        "max": float(v.max()),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "min": float(v.min()),
    }


def best_conjugate_rep_match(L: List[np.ndarray], R: List[np.ndarray],
                            restarts: int, maxiter: int, seed: int) -> Dict:
    """
    Check whether R looks like the conjugate rep of L (anti-fund for Hermitian basis):
      anti(L[a]) ~ - L[a]^T
    We test alignment between f(L) and f(anti(R)) in adjoint space.
    """
    fL = structure_constants(L)
    adL = adjoint_matrices(fL)

    R_conj = [normalize_hs(traceless(hermitize(-X.T))) for X in R]
    fRc = structure_constants(R_conj)
    adRc = adjoint_matrices(fRc)

    align = align_adjoint(adL, adRc, restarts=restarts, maxiter=maxiter, seed=seed)
    return {
        "align_obj": float(align["val"]),
        "success": bool(align["success"]),
        "nit": align["nit"],
    }


# -------------------------
# Virtual link promotion (dim 9) + singlet Gauss test
# -------------------------

def singlet_projector_dim3() -> np.ndarray:
    d = 3
    omega = np.zeros((d * d, 1), dtype=complex)
    for i in range(d):
        omega[i * d + i, 0] = 1.0
    P = (omega @ omega.conj().T) / float(d)
    return hermitize(P)


def gauss_singlet_test_virtual(L: List[np.ndarray], R: List[np.ndarray]) -> Dict:
    """
    Promote link to H_E ⊗ H_E (dim 9).
    Use L_eff = L ⊗ I (fund on left factor)
        R_eff = I ⊗ (-R^T) (anti-fund on right factor)
    Then test invariance of H = P_left ⊗ P_right (v5 style) under Gauss.

    Factors:
      0: matter_L in 3bar
      1: link_left_factor in 3          (uses L)
      2: link_right_factor in 3bar      (uses -R^T)
      3: matter_R in 3
    """
    d = 3
    I = np.eye(d, dtype=complex)

    # Site su(3)
    Q = su_generators_gellmann(3)
    Q_af = [-(X.T) for X in Q]  # anti-fund

    # Promoted link endpoint actions
    L_eff = [np.kron(L[a], I) for a in range(8)]
    R_eff = [np.kron(I, -R[a].T) for a in range(8)]

    # Build H = P ⊗ P on (0,1) and (2,3)
    P = singlet_projector_dim3()  # 9x9
    H = kron(P, P)                # 81x81
    H = hermitize(H)
    H /= max(1e-15, hs_norm(H))

    # Build Gauss generators (left: factor0+factor1, right: factor2+factor3)
    I3 = I
    def embed4(A: np.ndarray, factor: int) -> np.ndarray:
        mats = [I3, I3, I3, I3]
        mats[factor] = A
        return kron(*mats)

    left_vals = []
    right_vals = []
    for a in range(8):
        G_left = embed4(Q_af[a], 0) + embed4(L_eff[a], 1)
        G_right = embed4(R_eff[a], 2) + embed4(Q[a], 3)
        left_vals.append(hs_norm(comm(H, G_left)))
        right_vals.append(hs_norm(comm(H, G_right)))

    left = np.array(left_vals, dtype=float)
    right = np.array(right_vals, dtype=float)

    return {
        "left_max": float(left.max()),
        "right_max": float(right.max()),
        "left_vec": left.tolist(),
        "right_vec": right.tolist(),
        "passes_tol_1e-10": bool(max(left.max(), right.max()) <= 1e-10),
        "note": "If this passes, your extracted (L,R) are compatible with gauge structure AFTER promoting link to dim 9 (two-qutrit link)."
    }


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--mode", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    L, R, meta = load_npz_bases(args.npz, args.mode)

    fL = structure_constants(L)
    fR = structure_constants(R)

    # Cross commutator grid is the core "can both endpoints live on C^3?" test
    cross = commutator_grid_stats(L, R)

    # Conjugate-rep plausibility
    conj_match = best_conjugate_rep_match(L, R, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 33)

    # Virtual promotion test
    virtual = gauss_singlet_test_virtual(L, R)

    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "npz": args.npz,
            "mode": args.mode,
            "seed": int(args.seed),
            "restarts": int(args.restarts),
            "maxiter": int(args.maxiter),
            **meta,
        },
        "shapes": {
            "L": [int(x) for x in L[0].shape],
            "R": [int(x) for x in R[0].shape],
            "n_gen": 8,
        },
        "invariants": {
            "L": f_invariants(fL),
            "R": f_invariants(fR),
        },
        "tests": {
            "cross_commutators_[L^a,R^b]_HSnorm": cross,
            "conjugate_rep_alignment_R_to_-R^T_vs_L": conj_match,
            "virtual_link_promotion_dim9_singlet_gauss_test": virtual,
        },
        "interpretation": {
            "if_cross_max_is_O(1)": "Then a single qutrit link cannot host two commuting endpoint actions; gauge-link requires larger link register (e.g. dim 9).",
            "if_virtual_promotion_passes": "Then your extracted endpoints are 'gauge-compatible' once link is promoted to 9D, supporting the idea that a constraint mode (e.g., bandwidth/compression) is keeping the link too small.",
        }
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()