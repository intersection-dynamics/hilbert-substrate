#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_structure_constants_diagnostic_su3_v1_3.py
================================================

Goal: zero-friction run from the script's home directory.

Default behavior
----------------
- No --lr_npz required.
- Script looks for candidate NPZs in:
    1) <script_dir>/hsf_out/** (recursive)
    2) <script_dir>/*.npz      (non-recursive)
- It selects the "best" NPZ by OPENING each NPZ and scoring it by keys + shapes:
    - Must contain plausible L/R basis arrays with shape (n, d, d)
    - Prefers keys: L_aligned/R_aligned, L_basis/R_basis, L_mixed/R_mixed
    - Prefers n == 8, d == 3 (su3/qutrit)
    - Then prefers most-recent modification time

Override behavior
-----------------
- You can still pass --lr_npz FILE or DIRECTORY if you want.
  If directory, it scans recursively for *.npz.

Example
-------
python gauss_structure_constants_diagnostic_su3_v1_3.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Optional

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


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# -------------------------
# Canonical su(3) basis (HS-orthonormal)
# -------------------------

def su_generators_gellmann(d: int) -> List[np.ndarray]:
    if d < 2:
        raise ValueError("d must be >= 2")

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

    out: List[np.ndarray] = []
    for G in gens:
        out.append(normalize_hs(traceless(hermitize(G))))

    if len(out) != d * d - 1:
        raise RuntimeError("Unexpected generator count in su_generators_gellmann")
    return out


# -------------------------
# NPZ discovery + scoring by actual contents
# -------------------------

# Key preference order (left/right in lockstep)
KEY_SETS = [
    (["L_aligned", "L_basis_aligned"], ["R_aligned", "R_basis_aligned"]),
    (["L_mixed", "L_basis_mixed"], ["R_mixed", "R_basis_mixed"]),
    (["L_basis", "L"], ["R_basis", "R"]),
]


def iter_npz_paths(root: str, recursive: bool = True) -> List[str]:
    out: List[str] = []
    if os.path.isfile(root) and root.lower().endswith(".npz"):
        return [root]
    if os.path.isdir(root):
        if recursive:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith(".npz"):
                        out.append(os.path.join(dirpath, fn))
        else:
            for fn in os.listdir(root):
                p = os.path.join(root, fn)
                if os.path.isfile(p) and fn.lower().endswith(".npz"):
                    out.append(p)
    out.sort()
    return out


def try_extract_LR_from_npz(npz_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Returns (L_arr, R_arr, which_keyset) or (None, None, reason).
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None, None, f"load_error: {e}"

    keys = set(list(data.keys()))
    for (L_keys, R_keys) in KEY_SETS:
        lk = next((k for k in L_keys if k in keys), None)
        rk = next((k for k in R_keys if k in keys), None)
        if lk is None or rk is None:
            continue
        L = data[lk]
        R = data[rk]
        # Expect (n, d, d)
        if not (hasattr(L, "ndim") and hasattr(R, "ndim")):
            continue
        if L.ndim < 3 or R.ndim < 3:
            continue
        if L.shape[-1] != L.shape[-2] or R.shape[-1] != R.shape[-2]:
            continue
        if L.shape[-1] != R.shape[-1]:
            continue
        return L, R, f"{lk}/{rk}"

    return None, None, f"missing_LR_keys: {sorted(list(keys))[:20]}{'...' if len(keys) > 20 else ''}"


def score_npz_by_contents(npz_path: str) -> Tuple[int, float, str]:
    """
    Score by actual NPZ contents. Higher is better.
    Returns (score, mtime, note).
    """
    mtime = 0.0
    try:
        mtime = os.path.getmtime(npz_path)
    except Exception:
        pass

    L, R, note = try_extract_LR_from_npz(npz_path)
    if L is None or R is None:
        return -10_000, mtime, note

    n = int(L.shape[0])
    d = int(L.shape[-1])

    score = 0
    # Keyset preference implied by note
    if "aligned" in note:
        score += 500
    elif "mixed" in note:
        score += 300
    else:
        score += 150

    # Shape preferences
    if d == 3:
        score += 300
    else:
        score += 50

    if n == 8:
        score += 200
    else:
        score += max(0, 120 - abs(n - 8) * 10)

    # Sanity: L/R same shape
    if tuple(L.shape) == tuple(R.shape):
        score += 50

    # Filename weak hint (tiny weight only)
    base = os.path.basename(npz_path).lower()
    for kw, w in [("lr", 20), ("link", 10), ("echo", 10), ("gauss", 10), ("basis", 8)]:
        if kw in base:
            score += w

    return score, mtime, note


def auto_select_npz(user_override: Optional[str]) -> Tuple[str, List[Dict]]:
    """
    If user_override is None:
        scan <script_dir>/hsf_out recursively + <script_dir>/*.npz
    Else:
        if file -> use it
        if dir  -> scan recursively within it

    Returns selected_path and debug list for transparency.
    """
    if user_override:
        root = os.path.expandvars(os.path.expanduser(user_override))
        candidates = iter_npz_paths(root, recursive=True)
        if not candidates:
            raise FileNotFoundError(f"No .npz files found under: {root}")
    else:
        sd = script_dir()
        candidates = []
        # Prefer hsf_out tree
        candidates += iter_npz_paths(os.path.join(sd, "hsf_out"), recursive=True)
        # Also look for npz next to script
        candidates += iter_npz_paths(sd, recursive=False)
        # De-dup
        candidates = sorted(list(dict.fromkeys(candidates)))
        if not candidates:
            raise FileNotFoundError(
                f"No .npz files found in:\n"
                f"  {os.path.join(sd, 'hsf_out')} (recursive)\n"
                f"  {sd} (non-recursive)"
            )

    scored = []
    for p in candidates:
        sc, mt, note = score_npz_by_contents(p)
        scored.append({
            "path": p,
            "score": int(sc),
            "mtime": float(mt),
            "mtime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt)) if mt > 0 else "?",
            "note": note,
        })

    scored_sorted = sorted(scored, key=lambda r: (r["score"], r["mtime"]), reverse=True)
    best = scored_sorted[0]
    if best["score"] < -1000:
        # We found NPZs but none matched L/R basis keys/shapes.
        raise RuntimeError(
            "Found .npz files, but none contained recognizable L/R basis arrays.\n"
            "Top candidates:\n" +
            "\n".join([f"  score={r['score']} {r['path']}  ({r['note']})" for r in scored_sorted[:10]])
        )

    return best["path"], scored_sorted


# -------------------------
# Load LR bases (given a selected NPZ)
# -------------------------

def load_lr_bases_npz(npz_path: str, echo_model: str) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
    data = np.load(npz_path, allow_pickle=True)
    keys = set(list(data.keys()))

    if echo_model == "aligned":
        keysets = [
            (["L_aligned", "L_basis_aligned"], ["R_aligned", "R_basis_aligned"]),
            (["L_basis", "L"], ["R_basis", "R"]),
            (["L_mixed", "L_basis_mixed"], ["R_mixed", "R_basis_mixed"]),
        ]
    else:  # mixed
        keysets = [
            (["L_mixed", "L_basis_mixed"], ["R_mixed", "R_basis_mixed"]),
            (["L_basis", "L"], ["R_basis", "R"]),
            (["L_aligned", "L_basis_aligned"], ["R_aligned", "R_basis_aligned"]),
        ]

    lk = rk = None
    for (L_keys, R_keys) in keysets:
        lk = next((k for k in L_keys if k in keys), None)
        rk = next((k for k in R_keys if k in keys), None)
        if lk and rk:
            break

    if not lk or not rk:
        raise KeyError(f"No suitable L/R keys found for echo_model={echo_model}. Keys present: {sorted(list(keys))}")

    L_arr = data[lk]
    R_arr = data[rk]

    if L_arr.ndim < 3 or R_arr.ndim < 3:
        raise ValueError(f"Expected L/R arrays with shape (n,d,d). Got L:{L_arr.shape} R:{R_arr.shape}")

    L = [normalize_hs(traceless(hermitize(np.array(L_arr[i], dtype=complex)))) for i in range(L_arr.shape[0])]
    R = [normalize_hs(traceless(hermitize(np.array(R_arr[i], dtype=complex)))) for i in range(R_arr.shape[0])]

    return L, R, f"{lk}/{rk}"


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
            U, _, Vt = np.linalg.svd(O, full_matrices=False)
            O = U @ Vt
            best = {"val": val, "O": O, "nit": int(res.nit), "success": bool(res.success), "p": res.x.tolist()}
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
    ap.add_argument("--lr_npz", type=str, default=None,
                    help="Optional: specific .npz file or directory to scan. If omitted, auto-scans script_dir/hsf_out.")
    ap.add_argument("--echo_model", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show_pick", action="store_true",
                    help="Print the chosen NPZ and the top scored candidates.")
    args = ap.parse_args()

    chosen_npz, scored = auto_select_npz(args.lr_npz)

    if args.show_pick:
        print(f"[auto-pick] using: {chosen_npz}")
        print("[auto-pick] top candidates:")
        for r in scored[:10]:
            print(f"  score={r['score']:6d}  mtime={r['mtime_str']}  {r['path']}  ({r['note']})")

    Q = su_generators_gellmann(3)
    L_raw, R_raw, used_keys = load_lr_bases_npz(chosen_npz, args.echo_model)

    fQ = structure_constants(Q)
    adQ = adjoint_matrices(fQ)

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
            "script_dir": script_dir(),
            "lr_npz_override": args.lr_npz,
            "lr_npz_selected": chosen_npz,
            "npz_used_keys": used_keys,
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