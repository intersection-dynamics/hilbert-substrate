#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_link_qutrit_npz_diagnostic_v6_1.py

Fixes v6:
- Corrected virtual promotion embedding: total space is (3,9,9,3) -> 729 dim
- Correctly places singlet projectors inside 9D link factors by tensored identities:
    P_left  acts on (3bar ⊗ (3 ⊗ 3)) as (P_singlet(3bar⊗3) ⊗ I3)
    P_right acts on ((3 ⊗ 3bar) ⊗ 3) as (I3 ⊗ P_singlet(3bar⊗3))
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
    fL = structure_constants(L)
    adL = adjoint_matrices(fL)

    R_conj = [normalize_hs(traceless(hermitize(-X.T))) for X in R]
    fRc = structure_constants(R_conj)
    adRc = adjoint_matrices(fRc)

    align = align_adjoint(adL, adRc, restarts=restarts, maxiter=maxiter, seed=seed)
    return {"align_obj": float(align["val"]), "success": bool(align["success"]), "nit": align["nit"]}


# -------------------------
# Virtual link promotion (corrected): dims (3,9,9,3) -> 729
# -------------------------

def singlet_projector_dim3() -> np.ndarray:
    d = 3
    omega = np.zeros((d * d, 1), dtype=complex)
    for i in range(d):
        omega[i * d + i, 0] = 1.0
    return hermitize((omega @ omega.conj().T) / float(d))


def embed_4factor(op: np.ndarray, dims: Tuple[int, int, int, int], which: int) -> np.ndarray:
    """Embed op on factor 'which' inside dims=(d0,d1,d2,d3)."""
    mats = []
    for i, d in enumerate(dims):
        mats.append(np.eye(d, dtype=complex))
    mats[which] = op
    return kron(*mats)


def gauss_singlet_test_virtual(L: List[np.ndarray], R: List[np.ndarray]) -> Dict:
    """
    Factors and dims:
      0: matter_L (3bar)    dim 3
      1: link_left (3⊗3)    dim 9    uses L_eff[a] = L[a] ⊗ I3
      2: link_right (3⊗3bar)dim 9    uses R_eff[a] = I3 ⊗ (-R[a].T)
      3: matter_R (3)       dim 3
    Total dim 729.
    """
    d = 3
    I3 = np.eye(d, dtype=complex)
    dims = (3, 9, 9, 3)

    # Site su(3) in fund and anti-fund
    Q = su_generators_gellmann(3)
    Q_af = [-(X.T) for X in Q]  # anti-fund

    # Promoted link endpoint actions (9x9)
    L_eff = [np.kron(L[a], I3) for a in range(8)]        # (3⊗3) with action on first factor
    R_eff = [np.kron(I3, -R[a].T) for a in range(8)]     # (3⊗3bar) with action on second factor as anti-fund

    # Build singlet projectors inside the 9D link factors:
    # Left invariant pair is (matter_L 3bar) ⊗ (link_left fund-3 factor).
    # link_left factor is 3 ⊗ 3, so projector is P(3bar⊗3) ⊗ I3.
    Psing = singlet_projector_dim3()     # 9x9 acting on (3bar⊗3)
    P_left_0_1 = kron(Psing, I3)         # (3bar⊗3) ⊗ 3  => 27x27 on factors (0,1)
    # Right invariant pair is (link_right 3bar factor) ⊗ (matter_R 3).
    # link_right factor is 3 ⊗ 3bar, so projector is I3 ⊗ P(3bar⊗3) on (factor2,factor3) ordering:
    P_right_2_3 = kron(I3, Psing)        # 3 ⊗ (3bar⊗3) => 27x27 on factors (2,3)

    # Full H = P_left ⊗ P_right on (0,1,2,3)
    H = kron(P_left_0_1, P_right_2_3)    # 729x729
    H = hermitize(H)
    H /= max(1e-15, hs_norm(H))

    # Gauss generators:
    # Left: Q_af on factor0 plus L_eff on factor1
    # Right: R_eff on factor2 plus Q on factor3
    left_vals = []
    right_vals = []
    for a in range(8):
        G_left = embed_4factor(Q_af[a], dims, 0) + embed_4factor(L_eff[a], dims, 1)
        G_right = embed_4factor(R_eff[a], dims, 2) + embed_4factor(Q[a], dims, 3)
        left_vals.append(hs_norm(comm(H, G_left)))
        right_vals.append(hs_norm(comm(H, G_right)))

    left = np.array(left_vals, dtype=float)
    right = np.array(right_vals, dtype=float)

    return {
        "dims": {"total": 729, "factors": dims},
        "left_max": float(left.max()),
        "right_max": float(right.max()),
        "left_vec": left.tolist(),
        "right_vec": right.tolist(),
        "passes_tol_1e-10": bool(max(left.max(), right.max()) <= 1e-10),
        "note": "Corrected virtual promotion: tests whether extracted endpoints become gauge-compatible when link is promoted to 9D (two-qutrit) with right endpoint treated as anti-fund."
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

    cross = commutator_grid_stats(L, R)
    conj_match = best_conjugate_rep_match(L, R, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 33)
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
        "shapes": {"L": [3, 3], "R": [3, 3], "n_gen": 8},
        "invariants": {"L": f_invariants(fL), "R": f_invariants(fR)},
        "tests": {
            "cross_commutators_[L^a,R^b]_HSnorm": cross,
            "conjugate_rep_alignment_R_to_-R^T_vs_L": conj_match,
            "virtual_link_promotion_dim9_singlet_gauss_test": virtual,
        },
        "interpretation": {
            "cross_commutators": "If max is not near 0, qutrit link cannot host two commuting endpoint actions -> needs larger link register for gauge links.",
            "virtual_promotion": "If passes, your extracted endpoints are gauge-compatible once link is promoted; suggests a constraint mode is keeping link too small."
        }
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()