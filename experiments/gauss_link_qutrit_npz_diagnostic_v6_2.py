#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_link_qutrit_npz_diagnostic_v6_2.py

Adds vs v6.1:
- Writes JSON to disk under hsf_out/gauss_diag/...
- Extra diagnostics:
  * Gram matrix Tr(Ta Tb) for L and R (orthonormality check)
  * Per-generator HS norms
  * Closure residual: ||[Ta,Tb] - projection_onto_span|| (mean/max)
  * Uses basis_both_{aligned,mixed} (if present) to analyze relationship:
      - best generator-space SO(8) map from both->L and both->R (via adjoint alignment)
      - compares those maps, gives a sense if L and R are just different slicings

Run:
python gauss_link_qutrit_npz_diagnostic_v6_2.py --npz <file.npz> --mode aligned
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
# Generator-space SO(8) alignment (adjoint)
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


# -------------------------
# Load NPZ bases
# -------------------------

def _get_key(mode: str, which: str) -> str:
    return f"basis_{which}_{mode}"


def load_npz_bases(path: str, mode: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray] | None, Dict]:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    kL = _get_key(mode, "left")
    kR = _get_key(mode, "right")
    kB = _get_key(mode, "both")

    if kL not in data or kR not in data:
        raise KeyError(f"NPZ missing keys. Needed {kL},{kR}. Present keys: {keys}")

    L_arr = data[kL]
    R_arr = data[kR]
    if L_arr.shape != (8, 3, 3) or R_arr.shape != (8, 3, 3):
        raise ValueError(f"Unexpected shapes: L={L_arr.shape} R={R_arr.shape} (expected (8,3,3))")

    L = [normalize_hs(traceless(hermitize(np.array(L_arr[a], dtype=complex)))) for a in range(8)]
    R = [normalize_hs(traceless(hermitize(np.array(R_arr[a], dtype=complex)))) for a in range(8)]

    B = None
    if kB in data:
        B_arr = data[kB]
        if B_arr.shape == (8, 3, 3):
            B = [normalize_hs(traceless(hermitize(np.array(B_arr[a], dtype=complex)))) for a in range(8)]

    meta = {"npz_keys": keys, "used_keys": {"L": kL, "R": kR, "both": (kB if kB in keys else None)}}
    return L, R, B, meta


# -------------------------
# Extra diagnostics
# -------------------------

def gram_matrix(T: List[np.ndarray]) -> np.ndarray:
    n = len(T)
    G = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            G[a, b] = hs_inner(T[a], T[b])
    return G


def ortho_stats(G: np.ndarray) -> Dict[str, float]:
    off = G - np.diag(np.diag(G))
    return {
        "diag_mean": float(np.mean(np.diag(G))),
        "diag_min": float(np.min(np.diag(G))),
        "diag_max": float(np.max(np.diag(G))),
        "offdiag_maxabs": float(np.max(np.abs(off))),
        "offdiag_meanabs": float(np.mean(np.abs(off))),
    }


def closure_residual(T: List[np.ndarray]) -> Dict[str, float]:
    """
    For each (a,b), compute commutator C=[Ta,Tb].
    Project C onto span{Tc} using HS inner products (assuming near-orthonormal),
    and compute residual norm.
    """
    n = len(T)
    # build Gram and its inverse for robust projection even if basis not perfect
    G = gram_matrix(T)
    Ginv = np.linalg.pinv(G, rcond=1e-12)

    def project(C: np.ndarray) -> np.ndarray:
        # coefficients alpha_c = sum_d Ginv[c,d] <T_d, C>
        ip = np.array([hs_inner(T[d], C) for d in range(n)], dtype=float)
        alpha = Ginv @ ip
        P = np.zeros_like(C)
        for c in range(n):
            P += alpha[c] * T[c]
        return P

    res = []
    for a in range(n):
        for b in range(n):
            C = comm(T[a], T[b])
            P = project(C)
            res.append(hs_norm(C - P))

    v = np.array(res, dtype=float)
    return {"max": float(v.max()), "median": float(np.median(v)), "mean": float(v.mean())}


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


def compare_to_basis_both(B: List[np.ndarray], X: List[np.ndarray],
                          restarts: int, maxiter: int, seed: int) -> Dict:
    """
    Compare two su(3) bases B and X via adjoint alignment objective.
    """
    fB = structure_constants(B)
    fX = structure_constants(X)
    adB = adjoint_matrices(fB)
    adX = adjoint_matrices(fX)
    align = align_adjoint(adB, adX, restarts=restarts, maxiter=maxiter, seed=seed)
    return {"align_obj": float(align["val"]), "success": bool(align["success"]), "nit": align["nit"]}


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
    ap.add_argument("--outdir", type=str, default=os.path.join("hsf_out", "gauss_diag"))
    args = ap.parse_args()

    L, R, B, meta = load_npz_bases(args.npz, args.mode)

    fL = structure_constants(L)
    fR = structure_constants(R)

    GL = gram_matrix(L)
    GR = gram_matrix(R)

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
        "diagnostics": {
            "L_hs_norms": [hs_norm(x) for x in L],
            "R_hs_norms": [hs_norm(x) for x in R],
            "L_gram_stats": ortho_stats(GL),
            "R_gram_stats": ortho_stats(GR),
            "L_closure_residual": closure_residual(L),
            "R_closure_residual": closure_residual(R),
        },
        "tests": {
            "cross_commutators_[L^a,R^b]_HSnorm": commutator_grid_stats(L, R),
            "conjugate_rep_alignment_R_to_-R^T_vs_L": best_conjugate_rep_match(
                L, R, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 33
            ),
        },
        "both_basis_analysis": None,
        "interpretation": {
            "cross_commutators": "If max is O(1), L and R do not commute on the qutrit link -> cannot be genuine left/right gauge endpoint actions on the same C^3.",
            "next": "If basis_both_* exists, compare it to L and R. If both is close to each, L/R are two slices of one local su(3) action (echo), not commuting endpoint actions (gauge).",
        }
    }

    if B is not None:
        out["both_basis_analysis"] = {
            "align_both_to_L": compare_to_basis_both(B, L, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 101),
            "align_both_to_R": compare_to_basis_both(B, R, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 202),
        }

    # write JSON file
    ensure_dir(args.outdir)
    fname = f"{out['meta']['timestamp']}_qutrit_link_diag_{args.mode}.json"
    fpath = os.path.join(args.outdir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # print plus tell user where it went
    print(json.dumps(out, indent=2))
    print(f"\n[SAVED] {fpath}")


if __name__ == "__main__":
    main()