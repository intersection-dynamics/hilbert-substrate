#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_structure_constants_diagnostic_su3_v1_1.py
================================================

Fixes vs v1
-----------
1) IMPORTANT: Removed post-rotation Gram–Schmidt in rotate_generators().
   Gram–Schmidt can silently change the generator-index correspondence and
   defeat the purpose of applying an adjoint-space alignment O.

2) IMPORTANT: Removed Gram–Schmidt in apply_variant().
   Variants (neg/transpose/conj) should preserve orthonormality in exact math.
   We only do per-generator numerical cleanup: hermitize + traceless + normalize.

3) Derived local dimension d from Q[0].shape[0] instead of hard-coding d=3 in the
   gauge Hamiltonian / Gauss tests.

4) Added explicit "conj" and "neg_conj" variants (kept transpose variants for
   backwards compatibility).

5) Made align_adjoint() objective slightly more robust: uses sum(|D|^2).

Purpose
-------
Given your extracted su(3) link endpoint bases (L^a, R^a) from the LR NPZ and the
canonical site su(3) basis Q^a (HS-orthonormal Gell-Mann-like), this script:

1) Computes structure constants tensors f^{abc} for Q, L, R using:
      f^{abc} = (1/(2i)) Tr( [T^a, T^b] T^c )

2) Computes simple invariants and mismatch scores:
   - ||f||_F, max|f|
   - ad-casimir traces: tr(ad_a^T ad_a) etc.
   - "best orthogonal conjugacy" objective between adjoint sets:
        minimize_O sum_a || adQ[a] - O adX[a] O^T ||_F^2

3) Tests "representation variants" for link generators:
   For each of L and R, we consider variants of the matrices before doing
   adjoint alignment, then rotate generators in generator space, and finally
   test Gauss invariance for a simple gauge-covariant H.

Usage
-----
python gauss_structure_constants_diagnostic_su3_v1_1.py --lr_npz path/to/lr.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# scipy is used for expm + minimize
try:
    from scipy.linalg import expm
    from scipy.optimize import minimize
except Exception as e:
    raise RuntimeError("This script requires scipy (scipy.linalg.expm, scipy.optimize.minimize).") from e


# -------------------------
# Small utilities
# -------------------------

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def hermitize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)


def traceless(A: np.ndarray) -> np.ndarray:
    return A - np.trace(A) * np.eye(A.shape[0], dtype=A.dtype) / A.shape[0]


def hs_inner(A: np.ndarray, B: np.ndarray) -> float:
    # Hilbert–Schmidt inner product (real part)
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


def kron(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    # np.kron is fine here; three-factor helper keeps callsites readable
    return np.kron(np.kron(A, B), C)


# -------------------------
# Canonical su(3) basis (HS-orthonormal)
# -------------------------

def su_generators_gellmann(d: int) -> List[np.ndarray]:
    """
    Construct a Gell-Mann-like Hermitian traceless basis for su(d), then
    HS-orthonormalize by direct construction (not Gram–Schmidt).

    For d=3 returns 8 generators, HS-normalized to 1.
    """
    if d < 2:
        raise ValueError("d must be >= 2")

    gens: List[np.ndarray] = []

    # Symmetric off-diagonals: (E_ij + E_ji)
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = 1.0
            M[j, i] = 1.0
            gens.append(M)

    # Anti-symmetric off-diagonals: -i(E_ij - E_ji)
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = -1.0j
            M[j, i] = 1.0j
            gens.append(M)

    # Diagonal traceless generators
    for k in range(1, d):
        M = np.zeros((d, d), dtype=complex)
        # entries 0..k-1 are 1, entry k is -k
        for i in range(k):
            M[i, i] = 1.0
        M[k, k] = -float(k)
        gens.append(M)

    # Clean + HS normalize each. This construction is already orthogonal;
    # normalization makes it HS-orthonormal.
    out: List[np.ndarray] = []
    for G in gens:
        X = normalize_hs(traceless(hermitize(G)))
        out.append(X)

    # sanity: expected count is d^2 - 1
    if len(out) != d * d - 1:
        raise RuntimeError("Unexpected generator count in su_generators_gellmann")
    return out


# -------------------------
# Load LR bases from NPZ
# -------------------------

def load_lr_bases_npz(path: str, echo_model: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Expected npz format: must contain arrays for L and R bases.
    We keep this flexible by trying a few likely keys.

    echo_model:
      "aligned": prefer keys with "aligned" in name if present
      "mixed":   prefer keys with "mixed" in name if present
    """
    data = np.load(path, allow_pickle=True)

    # Candidate keys to try in order (you can extend if your NPZ differs)
    # We try to be forgiving but explicit.
    aligned_L_keys = ["L_aligned", "L_basis_aligned", "L_basis"]
    aligned_R_keys = ["R_aligned", "R_basis_aligned", "R_basis"]
    mixed_L_keys = ["L_mixed", "L_basis_mixed", "L_basis"]
    mixed_R_keys = ["R_mixed", "R_basis_mixed", "R_basis"]

    if echo_model == "aligned":
        L_keys, R_keys = aligned_L_keys, aligned_R_keys
    elif echo_model == "mixed":
        L_keys, R_keys = mixed_L_keys, mixed_R_keys
    else:
        raise ValueError("echo_model must be 'aligned' or 'mixed'")

    def pick(keys: List[str]) -> np.ndarray:
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"Could not find any of keys {keys} in NPZ. Present keys: {list(data.keys())}")

    L_arr = pick(L_keys)
    R_arr = pick(R_keys)

    # Convert to list of matrices
    L = [np.array(L_arr[i], dtype=complex) for i in range(L_arr.shape[0])]
    R = [np.array(R_arr[i], dtype=complex) for i in range(R_arr.shape[0])]

    # Numerical cleanup per-generator (but DO NOT re-mix the basis)
    L = [normalize_hs(traceless(hermitize(X))) for X in L]
    R = [normalize_hs(traceless(hermitize(X))) for X in R]

    return L, R


# -------------------------
# Structure constants + adjoint representation
# -------------------------

def structure_constants(T: List[np.ndarray]) -> np.ndarray:
    """
    Compute f^{abc} via:
       f^{abc} = (1/(2i)) Tr([T^a, T^b] T^c)
    Assumes T is a Hermitian traceless basis; scale depends on normalization.
    """
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
    """
    (ad_a)_{bc} = f^{abc}
    """
    n = f.shape[0]
    ad = []
    for a in range(n):
        ad.append(np.array(f[a, :, :], dtype=float))
    return ad


def f_invariants(f: np.ndarray) -> Dict[str, float]:
    return {
        "f_fro": float(np.linalg.norm(f.ravel())),
        "f_maxabs": float(np.max(np.abs(f))),
    }


def ad_invariants(ad: List[np.ndarray]) -> Dict[str, float]:
    # Simple invariants: per-generator Fro norms, summary stats.
    fro = np.array([np.linalg.norm(A) for A in ad], dtype=float)
    tr = np.array([np.trace(A.T @ A) for A in ad], dtype=float)  # same as Fro^2
    return {
        "ad_fro_mean": float(np.mean(fro)),
        "ad_fro_min": float(np.min(fro)),
        "ad_fro_max": float(np.max(fro)),
        "ad_trace_mean": float(np.mean(tr)),
        "ad_trace_min": float(np.min(tr)),
        "ad_trace_max": float(np.max(tr)),
    }


# -------------------------
# Adjoint alignment: minimize sum_a ||A_a - O B_a O^T||_F^2 over O ∈ SO(n)
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
            # Robust squared Frobenius norm (works even if tiny imag drift appears)
            s += float(np.sum(np.abs(D) ** 2))
        return s

    best = {"val": float("inf"), "O": np.eye(n), "nit": None, "success": False, "p": None}

    starts: List[np.ndarray] = [np.zeros(params_dim(n), dtype=float)]
    for _ in range(max(0, restarts - 1)):
        starts.append(rng.normal(scale=0.25, size=params_dim(n)))

    for p0 in starts:
        res = minimize(objective, p0, method="L-BFGS-B", options={"maxiter": int(maxiter), "ftol": 1e-12})
        val = float(res.fun)
        if val < best["val"]:
            K = skew_from_params(res.x, n)
            O = expm(K)
            # Clean numerical drift to nearest orthogonal
            U, _, Vt = np.linalg.svd(O, full_matrices=False)
            O = U @ Vt
            best = {"val": val, "O": O, "nit": int(res.nit), "success": bool(res.success), "p": res.x.tolist()}
    return best


# -------------------------
# Rotate generators in generator space (NO Gram–Schmidt!)
# -------------------------

def rotate_generators(T: List[np.ndarray], O: np.ndarray) -> List[np.ndarray]:
    """
    Generator-space rotation: T'_a = sum_i O[a,i] T_i

    IMPORTANT:
    Do NOT Gram–Schmidt here. Gram–Schmidt re-mixes and can destroy the
    intended a-index correspondence needed for gauge contractions.
    """
    n = len(T)
    out: List[np.ndarray] = []
    for a in range(n):
        M = np.zeros_like(T[0])
        for i in range(n):
            M += float(O[a, i]) * T[i]
        out.append(normalize_hs(traceless(hermitize(M))))
    return out


# -------------------------
# Gauss test for H_gauge
# -------------------------

def build_H_gauge(Q: List[np.ndarray], L: List[np.ndarray], R: List[np.ndarray]) -> np.ndarray:
    d = int(Q[0].shape[0])
    I = np.eye(d, dtype=complex)
    H = np.zeros((d * d * d, d * d * d), dtype=complex)
    for a in range(len(Q)):
        H += kron(Q[a], L[a], I) + kron(I, R[a], Q[a])
    H = hermitize(H)
    nrm = hs_norm(H)
    if nrm > 0:
        H = H / nrm
    return H


def gauss_commutators(H: np.ndarray, Q: List[np.ndarray], L: List[np.ndarray], R: List[np.ndarray]) -> Dict[str, float]:
    d = int(Q[0].shape[0])
    I = np.eye(d, dtype=complex)

    left = []
    right = []
    for a in range(len(Q)):
        G_left = kron(Q[a], I, I) + kron(I, L[a], I)
        G_right = kron(I, I, Q[a]) - kron(I, R[a], I)
        left.append(hs_norm(comm(H, G_left)))
        right.append(hs_norm(comm(H, G_right)))

    left = np.array(left, dtype=float)
    right = np.array(right, dtype=float)

    return {
        "gauss_left_max": float(np.max(left)),
        "gauss_right_max": float(np.max(right)),
        "gauss_left_median": float(np.median(left)),
        "gauss_right_median": float(np.median(right)),
        "gauss_left_mean": float(np.mean(left)),
        "gauss_right_mean": float(np.mean(right)),
        "gauss_left_vec": left.tolist(),
        "gauss_right_vec": right.tolist(),
    }


# -------------------------
# Representation variants
# -------------------------

def apply_variant(T: List[np.ndarray], variant: str) -> List[np.ndarray]:
    """
    Apply a simple matrix-space variant to the generators without re-mixing them.

    Kept variants:
      orig, neg, transpose, neg_transpose

    Added (clearer):
      conj, neg_conj
    """
    if variant == "orig":
        out = [X.copy() for X in T]
    elif variant == "neg":
        out = [(-X).copy() for X in T]
    elif variant == "transpose":
        out = [X.T.copy() for X in T]
    elif variant == "neg_transpose":
        out = [(-X.T).copy() for X in T]
    elif variant == "conj":
        out = [X.conj().copy() for X in T]
    elif variant == "neg_conj":
        out = [(-X.conj()).copy() for X in T]
    else:
        raise ValueError(f"unknown variant: {variant}")

    # Per-generator cleanup only (NO Gram–Schmidt)
    out = [normalize_hs(traceless(hermitize(X))) for X in out]
    return out


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_npz", type=str, required=True)
    ap.add_argument("--echo_model", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Q = su_generators_gellmann(3)
    L_raw, R_raw = load_lr_bases_npz(args.lr_npz, args.echo_model)

    fQ = structure_constants(Q)
    adQ = adjoint_matrices(fQ)

    base = {
        "Q": {"f": f_invariants(fQ), "ad": ad_invariants(adQ)},
    }

    # Expanded variants list (keeps prior ones, adds conj explicitly)
    variants = ["orig", "neg", "transpose", "neg_transpose", "conj", "neg_conj"]
    combo_results = []

    # baseline (no alignment, no variants)
    H0 = build_H_gauge(Q, L_raw, R_raw)
    g0 = gauss_commutators(H0, Q, L_raw, R_raw)

    for vL in variants:
        L_v = apply_variant(L_raw, vL)
        fL = structure_constants(L_v)
        adL = adjoint_matrices(fL)
        alignL = align_adjoint(adQ, adL, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 101)

        for vR in variants:
            R_v = apply_variant(R_raw, vR)
            fR = structure_constants(R_v)
            adR = adjoint_matrices(fR)
            alignR = align_adjoint(adQ, adR, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 202)

            # rotate to site-index basis (do not re-orthonormalize / re-mix)
            OL = np.array(alignL["O"], dtype=float)
            OR = np.array(alignR["O"], dtype=float)
            L_al = rotate_generators(L_v, OL)
            R_al = rotate_generators(R_v, OR)

            H = build_H_gauge(Q, L_al, R_al)
            g = gauss_commutators(H, Q, L_al, R_al)

            combo_results.append({
                "L_variant": vL,
                "R_variant": vR,
                "align_obj_L": float(alignL["val"]),
                "align_obj_R": float(alignR["val"]),
                "gauss_left_max": g["gauss_left_max"],
                "gauss_right_max": g["gauss_right_max"],
                "gauss_left_median": g["gauss_left_median"],
                "gauss_right_median": g["gauss_right_median"],
                "passes_tol": bool((g["gauss_left_max"] <= args.tol) and (g["gauss_right_max"] <= args.tol)),
            })

    # sort: best (min max(G_left_max, G_right_max))
    combo_results_sorted = sorted(combo_results, key=lambda r: max(r["gauss_left_max"], r["gauss_right_max"]))
    best = combo_results_sorted[0] if combo_results_sorted else None

    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "lr_npz": args.lr_npz,
            "echo_model": args.echo_model,
            "tol": float(args.tol),
            "restarts": int(args.restarts),
            "maxiter": int(args.maxiter),
            "seed": int(args.seed),
        },
        "baseline_unaligned": {
            "gauss": g0,
        },
        "invariants_site": base["Q"],
        "variant_sweep_summary": {
            "variants_tested": variants,
            "n_combos": int(len(combo_results)),
            "best": best,
        },
        "variant_sweep_ranked": combo_results_sorted,
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()