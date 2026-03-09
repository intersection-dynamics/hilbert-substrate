#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_link_npz_diagnostic_v7_3_subalgebra.py
===========================================

v7.3 (option A): Find an su(3)-like 8-generator subalgebra as a *subset* of a larger
extracted k-generator basis (e.g. k=18) on each endpoint.

Why this exists
---------------
Your link9 extraction often yields k>8 (e.g. 18). Naively truncating to the
top-8 SVD directions produces 8 generators that are NOT su(3) (closure residual large,
f_fro far from canonical). So instead we:

- search for an 8-element subset inside the k basis that is as close as possible
  to an su(3) Hermitian basis:
    * corrected closure residual small
    * f_fro close to canonical su(3) value for HS-orthonormal basis (~3.4641016)

Then we test whether the chosen L8 and R8 behave more gauge-like:
- cross-commutator grid/energy
- commuting-spectrum between the two 8-sets

NPZ keys
--------
Prefers full keys if present:
  basis_left_<mode>_full, basis_right_<mode>_full
else uses:
  basis_left_<mode>, basis_right_<mode>

Optionally uses basis_both_<mode>_full for reference (not required).

Outputs
-------
JSON:
  hsf_out/gauss_diag/<timestamp>_link_diag_<mode>_v7_3.json
NPZ artifacts:
  hsf_out/gauss_diag/<timestamp>_link_diag_<mode>_v7_3_artifacts.npz

Artifacts include:
  L8, R8 (8,d,d)  selected subsets
  L8_idx, R8_idx (8,) indices into the original k basis
  metrics

Run
---
python gauss_link_npz_diagnostic_v7_3_subalgebra.py --npz <file.npz> --mode aligned
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np


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
    d = A.shape[0]
    return A - np.trace(A) * np.eye(d, dtype=A.dtype) / d


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


# -------------------------
# Load NPZ bases (k,d agnostic), with preference for *_full
# -------------------------

def _key(mode: str, which: str, full: bool) -> str:
    return f"basis_{which}_{mode}_full" if full else f"basis_{which}_{mode}"


def _load_basis(data: np.lib.npyio.NpzFile, key: str) -> List[np.ndarray]:
    arr = data[key]
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Key {key} has shape {arr.shape}; expected (k,d,d).")
    k = int(arr.shape[0])
    d = int(arr.shape[1])
    out = []
    for a in range(k):
        X = np.array(arr[a], dtype=complex)
        out.append(normalize_hs(traceless(hermitize(X))))
    return out


def load_npz_lr(path: str, mode: str) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    # Prefer full keys if present
    kL_full = _key(mode, "left", True)
    kR_full = _key(mode, "right", True)
    use_full = (kL_full in keys) and (kR_full in keys)

    kL = kL_full if use_full else _key(mode, "left", False)
    kR = kR_full if use_full else _key(mode, "right", False)

    if kL not in keys or kR not in keys:
        raise KeyError(f"Missing L/R keys. Needed {kL},{kR}. Present keys: {keys}")

    L = _load_basis(data, kL)
    R = _load_basis(data, kR)

    if len(L) != len(R) or L[0].shape != R[0].shape:
        raise ValueError(f"L/R mismatch: kL={len(L)} kR={len(R)} dL={L[0].shape} dR={R[0].shape}")

    meta = {
        "npz_keys": keys,
        "used_keys": {"L": kL, "R": kR},
        "used_full": bool(use_full),
    }
    return L, R, meta


# -------------------------
# su(3) structure constants and corrected closure residual
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


def f_fro(f: np.ndarray) -> float:
    return float(np.linalg.norm(f.ravel()))


def gram_matrix(T: List[np.ndarray]) -> np.ndarray:
    n = len(T)
    G = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            G[a, b] = hs_inner(T[a], T[b])
    return G


def closure_residual_corrected(T: List[np.ndarray]) -> float:
    """
    Returns mean residual norm over all (a,b):
      Y = -i[T_a,T_b] projected onto span{T_c}
    """
    n = len(T)
    G = gram_matrix(T)
    Ginv = np.linalg.pinv(G, rcond=1e-12)

    def project(Y: np.ndarray) -> np.ndarray:
        ip = np.array([hs_inner(T[d], Y) for d in range(n)], dtype=float)
        alpha = Ginv @ ip
        P = np.zeros_like(Y)
        for c in range(n):
            P += alpha[c] * T[c]
        return P

    res = []
    for a in range(n):
        for b in range(n):
            Y = (-1.0j) * comm(T[a], T[b])
            P = project(Y)
            res.append(hs_norm(Y - P))
    return float(np.mean(np.array(res, dtype=float)))


# -------------------------
# Cross commutators and commuting spectrum (8-set)
# -------------------------

def commutator_grid(L: List[np.ndarray], R: List[np.ndarray]) -> np.ndarray:
    n = len(L)
    M = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            M[a, b] = hs_norm(comm(L[a], R[b]))
    return M


def cross_energy(M: np.ndarray) -> float:
    return float(np.sum(M * M))


def commuting_quadratic_form(A: List[np.ndarray], B: List[np.ndarray]) -> np.ndarray:
    n = len(A)
    M = np.zeros((n, n), dtype=float)
    C = [[comm(A[a], B[b]) for b in range(n)] for a in range(n)]
    for a in range(n):
        for ap in range(n):
            s = 0.0
            for b in range(n):
                s += hs_inner(C[a][b], C[ap][b])
            M[a, ap] = float(s)
    return 0.5 * (M + M.T)


def commuting_dim_est(L: List[np.ndarray], R: List[np.ndarray], eps: float) -> Dict:
    ML = commuting_quadratic_form(L, R)
    MR = commuting_quadratic_form(R, L)
    eL = np.sort(np.real(np.linalg.eigvalsh(ML)))
    eR = np.sort(np.real(np.linalg.eigvalsh(MR)))
    return {
        "eps": float(eps),
        "L_commuting_dim_est": int(np.sum(eL <= eps)),
        "R_commuting_dim_est": int(np.sum(eR <= eps)),
        "L_min_eigs": eL[:3].tolist(),
        "R_min_eigs": eR[:3].tolist(),
    }


# -------------------------
# Subset search for su(3)-like 8-set
# -------------------------

SU3_F_FRO_TARGET = 3.4641016151377544  # canonical for HS-orthonormal su(3)


def score_subset(basis: List[np.ndarray], idx: List[int],
                 w_close: float = 1.0, w_f: float = 1.0) -> Dict:
    """
    Lower score is better.
    """
    T = [basis[i] for i in idx]
    # ensure HS normalization (already should be, but safe)
    T = [normalize_hs(traceless(hermitize(X))) for X in T]
    cl = closure_residual_corrected(T)
    f = structure_constants(T)
    ff = f_fro(f)
    score = (w_close * cl) + (w_f * abs(ff - SU3_F_FRO_TARGET))
    return {"score": float(score), "closure_mean": float(cl), "f_fro": float(ff)}


def greedy_su3_subset(basis: List[np.ndarray], seed: int = 0,
                      n_trials: int = 30,
                      w_close: float = 1.0, w_f: float = 1.0) -> Dict:
    """
    Greedy forward selection with random restarts:
    - start with a random element
    - iteratively add the element that gives the best score at size s+1
    - keep best of n_trials
    """
    rng = np.random.default_rng(seed)
    k = len(basis)
    if k < 8:
        raise ValueError(f"Need k>=8 to pick an 8-subset; got {k}")

    best = None

    for t in range(n_trials):
        start = int(rng.integers(0, k))
        idx = [start]

        while len(idx) < 8:
            candidates = [i for i in range(k) if i not in idx]
            best_i = None
            best_s = None
            for i in candidates:
                s = score_subset(basis, idx + [i], w_close=w_close, w_f=w_f)
                if best_s is None or s["score"] < best_s["score"]:
                    best_s = s
                    best_i = i
            idx.append(best_i)

        final = score_subset(basis, idx, w_close=w_close, w_f=w_f)
        rec = {"idx": idx, **final}

        if best is None or rec["score"] < best["score"]:
            best = rec

    return best


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--mode", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--outdir", type=str, default=os.path.join("hsf_out", "gauss_diag"))
    ap.add_argument("--eps_commute", type=float, default=1e-6)
    ap.add_argument("--trials", type=int, default=40, help="greedy restarts per side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w_close", type=float, default=1.0)
    ap.add_argument("--w_f", type=float, default=1.0)
    args = ap.parse_args()

    L, R, meta = load_npz_lr(args.npz, args.mode)
    k = len(L)
    d = int(L[0].shape[0])

    # Find best su(3)-like 8-subset on each side
    bestL = greedy_su3_subset(L, seed=args.seed + 101, n_trials=args.trials, w_close=args.w_close, w_f=args.w_f)
    bestR = greedy_su3_subset(R, seed=args.seed + 202, n_trials=args.trials, w_close=args.w_close, w_f=args.w_f)

    L8_idx = bestL["idx"]
    R8_idx = bestR["idx"]
    L8 = [L[i] for i in L8_idx]
    R8 = [R[i] for i in R8_idx]

    # Evaluate cross/gauge-ish metrics on selected 8-sets
    M = commutator_grid(L8, R8)
    E = cross_energy(M)
    commute = commuting_dim_est(L8, R8, eps=args.eps_commute)

    # Output
    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "npz": args.npz,
            "mode": args.mode,
            "params": {
                "eps_commute": float(args.eps_commute),
                "trials": int(args.trials),
                "seed": int(args.seed),
                "w_close": float(args.w_close),
                "w_f": float(args.w_f),
            },
            **meta,
        },
        "shapes": {"k_full": int(k), "d_link": int(d)},
        "su3_target": {"f_fro_target": SU3_F_FRO_TARGET},
        "selected_subalgebra": {
            "L": {"idx": L8_idx, "score": bestL["score"], "closure_mean": bestL["closure_mean"], "f_fro": bestL["f_fro"]},
            "R": {"idx": R8_idx, "score": bestR["score"], "closure_mean": bestR["closure_mean"], "f_fro": bestR["f_fro"]},
        },
        "cross_on_selected_8sets": {
            "cross_energy_sum_sq": float(E),
            "cross_comm_max": float(np.max(M)),
            "cross_comm_median": float(np.median(M)),
            "cross_comm_mean": float(np.mean(M)),
            "cross_comm_min": float(np.min(M)),
        },
        "commuting_content_on_selected_8sets": commute,
        "note": (
            "This is v7.3 option-A: greedy discrete 8-subset search. "
            "If it finds L/R 8-sets with f_fro≈3.464 and small closure residual, "
            "then you truly have su(3) subalgebras inside the k>8 span."
        ),
    }

    ensure_dir(args.outdir)
    base = f"{out['meta']['timestamp']}_link_diag_{args.mode}_v7_3"
    json_path = os.path.join(args.outdir, base + ".json")
    npz_path = os.path.join(args.outdir, base + "_artifacts.npz")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    np.savez_compressed(
        npz_path,
        L8=np.stack(L8, axis=0),
        R8=np.stack(R8, axis=0),
        L8_idx=np.array(L8_idx, dtype=int),
        R8_idx=np.array(R8_idx, dtype=int),
        cross_comm_grid=M,
    )

    print(json.dumps(out, indent=2))
    print(f"\n[SAVED] {json_path}")
    print(f"[SAVED] {npz_path}")


if __name__ == "__main__":
    main()