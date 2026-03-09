#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_structure_constants_diagnostic_su3_v1_2.py
================================================

Changes vs v1_1
---------------
1) --lr_npz can now be:
   - a .npz file path
   - a directory path containing .npz files
   The script will auto-select an NPZ from a directory using a score heuristic
   (prefers names containing 'lr', 'link', 'bond', 'echo', 'gauss', 'aligned',
   and then prefers most-recent mtime).

2) Added --list_npz to print candidate NPZ files found in a directory and exit.

3) Improved error messages for missing keys / empty directories.

Everything else matches v1_1: no Gram–Schmidt after rotations; d inferred; variants include conj.

Usage examples (Windows)
-----------------------
# Option A: pass a specific npz file
python gauss_structure_constants_diagnostic_su3_v1_2.py --lr_npz C:\\GitHub\\hilbert_substrate\\experiments\\hsf_out\\some_lr_artifact.npz

# Option B: pass a directory (auto-pick best)
python gauss_structure_constants_diagnostic_su3_v1_2.py --lr_npz C:\\GitHub\\hilbert_substrate\\experiments\\hsf_out

# Option C: list candidates first
python gauss_structure_constants_diagnostic_su3_v1_2.py --lr_npz C:\\GitHub\\hilbert_substrate\\experiments\\hsf_out --list_npz
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
# Small utilities
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


def kron(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(A, B), C)


# -------------------------
# Canonical su(3) basis (HS-orthonormal)
# -------------------------

def su_generators_gellmann(d: int) -> List[np.ndarray]:
    if d < 2:
        raise ValueError("d must be >= 2")

    gens: List[np.ndarray] = []

    # Symmetric off-diagonals
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = 1.0
            M[j, i] = 1.0
            gens.append(M)

    # Anti-symmetric off-diagonals
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = -1.0j
            M[j, i] = 1.0j
            gens.append(M)

    # Diagonals
    for k in range(1, d):
        M = np.zeros((d, d), dtype=complex)
        for i in range(k):
            M[i, i] = 1.0
        M[k, k] = -float(k)
        gens.append(M)

    out: List[np.ndarray] = []
    for G in gens:
        out.append(normalize_hs(traceless(hermitize(G))))

    if len(out) != d * d - 1:
        raise RuntimeError("Unexpected generator count in su_generators_gellmann")
    return out


# -------------------------
# NPZ selection helpers
# -------------------------

def is_npz_file(p: str) -> bool:
    return os.path.isfile(p) and p.lower().endswith(".npz")


def list_npz_in_dir(d: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(d):
        return out
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if is_npz_file(p):
            out.append(p)
    out.sort()
    return out


def score_npz_path(p: str) -> Tuple[int, float]:
    """
    Higher score is better. Returned tuple (score, mtime).
    """
    name = os.path.basename(p).lower()
    score = 0

    # Strong signals
    for kw, w in [
        ("lr", 50),
        ("link", 35),
        ("bond", 25),
        ("echo", 20),
        ("gauss", 20),
        ("aligned", 10),
        ("basis", 8),
        ("gener", 6),
        ("su3", 6),
        ("su(3", 6),
    ]:
        if kw in name:
            score += w

    # Penalize "summary" or "pooled" style npzs that are less likely to contain bases
    for kw, w in [
        ("summary", -10),
        ("pooled", -10),
        ("metrics", -6),
        ("plot", -6),
    ]:
        if kw in name:
            score += w

    try:
        mtime = os.path.getmtime(p)
    except Exception:
        mtime = 0.0

    return score, mtime


def resolve_npz_path(user_path: str) -> Tuple[str, List[str]]:
    """
    Returns (selected_npz, candidates_list).
    If user_path is a file, selected_npz is that file.
    If user_path is a directory, selects the best candidate.
    """
    p = os.path.expandvars(os.path.expanduser(user_path))

    if is_npz_file(p):
        return p, [p]

    if os.path.isdir(p):
        candidates = list_npz_in_dir(p)
        if not candidates:
            raise FileNotFoundError(f"No .npz files found in directory: {p}")

        # Rank by (score, mtime) descending
        ranked = sorted(candidates, key=lambda x: score_npz_path(x), reverse=True)
        return ranked[0], ranked

    # If it's neither a file nor a directory, give a sharp message.
    raise FileNotFoundError(f"--lr_npz path not found or not accessible: {p}")


# -------------------------
# Load LR bases from NPZ
# -------------------------

def load_lr_bases_npz(path: str, echo_model: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    data = np.load(path, allow_pickle=True)

    aligned_L_keys = ["L_aligned", "L_basis_aligned", "L_basis", "L"]
    aligned_R_keys = ["R_aligned", "R_basis_aligned", "R_basis", "R"]
    mixed_L_keys = ["L_mixed", "L_basis_mixed", "L_basis", "L"]
    mixed_R_keys = ["R_mixed", "R_basis_mixed", "R_basis", "R"]

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
        raise KeyError(
            f"Could not find any of keys {keys} in NPZ.\n"
            f"NPZ keys present: {list(data.keys())}"
        )

    L_arr = pick(L_keys)
    R_arr = pick(R_keys)

    if L_arr.ndim < 3 or R_arr.ndim < 3:
        raise ValueError(
            f"Expected L/R arrays with shape (n, d, d). Got L:{L_arr.shape} R:{R_arr.shape}"
        )

    L = [np.array(L_arr[i], dtype=complex) for i in range(L_arr.shape[0])]
    R = [np.array(R_arr[i], dtype=complex) for i in range(R_arr.shape[0])]

    # Per-generator cleanup only
    L = [normalize_hs(traceless(hermitize(X))) for X in L]
    R = [normalize_hs(traceless(hermitize(X))) for X in R]

    return L, R


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
    n = f.shape[0]
    return [np.array(f[a, :, :], dtype=float) for a in range(n)]


def f_invariants(f: np.ndarray) -> Dict[str, float]:
    return {"f_fro": float(np.linalg.norm(f.ravel())), "f_maxabs": float(np.max(np.abs(f)))}


def ad_invariants(ad: List[np.ndarray]) -> Dict[str, float]:
    fro = np.array([np.linalg.norm(A) for A in ad], dtype=float)
    tr = np.array([np.trace(A.T @ A) for A in ad], dtype=float)
    return {
        "ad_fro_mean": float(np.mean(fro)),
        "ad_fro_min": float(np.min(fro)),
        "ad_fro_max": float(np.max(fro)),
        "ad_trace_mean": float(np.mean(tr)),
        "ad_trace_min": float(np.min(tr)),
        "ad_trace_max": float(np.max(tr)),
    }


# -------------------------
# Adjoint alignment
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

    best = {"val": float("inf"), "O": np.eye(n), "nit": None, "success": False, "p": None}

    starts: List[np.ndarray] = [np.zeros(params_dim(n), dtype=float)]
    for _ in range(max(0, restarts - 1)):
        starts.append(rng.normal(scale=0.25, size=params_dim(n)))

    for p0 in starts:
        res = minimize(objective, p0, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-12})
        val = float(res.fun)
        if val < best["val"]:
            K = skew_from_params(res.x, n)
            O = expm(K)
            # Snap to nearest orthogonal
            U, _, Vt = np.linalg.svd(O, full_matrices=False)
            O = U @ Vt
            best = {"val": val, "O": O, "nit": int(res.nit), "success": bool(res.success), "p": res.x.tolist()}
    return best


# -------------------------
# Rotate generators in generator space (NO Gram–Schmidt)
# -------------------------

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
# Gauge Hamiltonian + Gauss commutators
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

    return [normalize_hs(traceless(hermitize(X))) for X in out]


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_npz", type=str, required=True,
                    help="Path to an LR .npz file OR a directory containing .npz artifacts.")
    ap.add_argument("--echo_model", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list_npz", action="store_true",
                    help="If --lr_npz is a directory, list candidate .npz files and exit.")
    args = ap.parse_args()

    selected_npz, candidates = resolve_npz_path(args.lr_npz)

    if args.list_npz:
        if os.path.isdir(os.path.expanduser(os.path.expandvars(args.lr_npz))):
            print("NPZ candidates (ranked best-first):")
            ranked = sorted(candidates, key=lambda x: score_npz_path(x), reverse=True)
            for p in ranked:
                sc, mt = score_npz_path(p)
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt)) if mt > 0 else "?"
                print(f"  score={sc:4d}  mtime={ts}  {p}")
        else:
            print(f"--lr_npz is a file: {selected_npz}")
        return

    # Load
    Q = su_generators_gellmann(3)
    L_raw, R_raw = load_lr_bases_npz(selected_npz, args.echo_model)

    # Site invariants
    fQ = structure_constants(Q)
    adQ = adjoint_matrices(fQ)

    # Baseline (no alignment, no variants)
    H0 = build_H_gauge(Q, L_raw, R_raw)
    g0 = gauss_commutators(H0, Q, L_raw, R_raw)

    variants = ["orig", "neg", "transpose", "neg_transpose", "conj", "neg_conj"]
    combo_results = []

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

    combo_results_sorted = sorted(combo_results, key=lambda r: max(r["gauss_left_max"], r["gauss_right_max"]))
    best = combo_results_sorted[0] if combo_results_sorted else None

    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "lr_npz_input": args.lr_npz,
            "lr_npz_selected": selected_npz,
            "echo_model": args.echo_model,
            "tol": float(args.tol),
            "restarts": int(args.restarts),
            "maxiter": int(args.maxiter),
            "seed": int(args.seed),
        },
        "baseline_unaligned": {"gauss": g0},
        "invariants_site": {"f": f_invariants(fQ), "ad": ad_invariants(adQ)},
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