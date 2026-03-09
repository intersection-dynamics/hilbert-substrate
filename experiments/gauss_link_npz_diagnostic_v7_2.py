#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_link_npz_diagnostic_v7_2.py
================================

v7.2 builds on v7.1 (k,d arbitrary) and adds *actionable* structure extraction:

1) Extract commuting (near-null) directions
------------------------------------------
Given L basis {L_a} and R basis {R_b} of size k on link Hilbert space C^d, define

  For v in R^k, X_L(v) = sum_a v_a L_a
  Q_L(v) = sum_b || [X_L(v), R_b] ||_HS^2 = v^T M_L v

Likewise define M_R for combinations of R commuting with all L.

We:
- compute eigen-decomposition of M_L and M_R,
- extract the smallest-eigenvalue eigenvectors (commuting candidates),
- report "commuting_dim_est" for a user-chosen tolerance eps_commute,
- export the commuting operators X_L, X_R.

2) Centrality tests
-------------------
For commuting candidates X_L:
- test how much X_L commutes with *L itself*:
    C_self_L = mean_a ||[X_L, L_a]||_HS
and similarly for X_R with R.

This tells you whether the commuting direction is:
- a shared center between endpoints only (commutes with opposite side),
- or an actual center of the endpoint algebra itself.

3) Approximate maximal commuting subspace dimension vs tolerance
---------------------------------------------------------------
We compute a "commutation Gram" K (k x k) between combinations of L that commute with all R:

- From eigvals of M_L: directions with eig <= eps are those commuting with all R (in HS^2 sense).
We report dim(eig<=eps) for eps in a sweep (log-spaced), same for M_R.

This is the clean diagnostic for whether you're growing toward an 8D commuting block.

4) Optional: "best pair" commuting vectors
------------------------------------------
We also compute the best pair (v_L, v_R) that minimize
    || [X_L(v_L), X_R(v_R)] ||_HS
subject to ||v||=1
within the extracted commuting subspaces (if any).

Outputs
-------
JSON summary:
  hsf_out/gauss_diag/<timestamp>_link_diag_<mode>_v7_2.json

NPZ artifacts (operators and vectors):
  hsf_out/gauss_diag/<timestamp>_link_diag_<mode>_v7_2_artifacts.npz

NPZ contains:
  vL_min, vR_min                   (k,)
  XL_min, XR_min                   (d,d)
  eigs_ML, eigs_MR                 (k,)
  V_ML, V_MR                       (k,k) eigenvectors (columns)
  dims_vs_eps_L, dims_vs_eps_R     (n_eps,2) columns [eps,dim]
  (plus some commutator stats)

Usage
-----
python gauss_link_npz_diagnostic_v7_2.py --npz <file.npz> --mode aligned
python gauss_link_npz_diagnostic_v7_2.py --npz <file.npz> --mode mixed

Recommended for your current link9 run:
python gauss_link_npz_diagnostic_v7_2.py --npz echo_algebra_step1_link9_su3_LR_bases_20260221_210222.npz --mode aligned

Notes
-----
- This script assumes L and R have the same k and same d.
- It does *not* assume k=8.
- It does *not* assume closure (indeed, k=18 likely includes non-closed operator content).
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
# Load NPZ bases (k,d agnostic)
# -------------------------

def _get_key(mode: str, which: str) -> str:
    return f"basis_{which}_{mode}"


def _load_basis(data: np.lib.npyio.NpzFile, key: str) -> List[np.ndarray]:
    arr = data[key]
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Key {key} has shape {arr.shape}; expected (k,d,d).")
    k = int(arr.shape[0])
    d = int(arr.shape[1])
    out = []
    for a in range(k):
        X = np.array(arr[a], dtype=complex)
        if X.shape != (d, d):
            raise ValueError("shape mismatch inside basis array")
        out.append(normalize_hs(traceless(hermitize(X))))
    return out


def load_npz_bases(path: str, mode: str) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]], Dict]:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    kL = _get_key(mode, "left")
    kR = _get_key(mode, "right")
    kB = _get_key(mode, "both")

    if kL not in data or kR not in data:
        raise KeyError(f"NPZ missing keys. Needed {kL},{kR}. Present keys: {keys}")

    L = _load_basis(data, kL)
    R = _load_basis(data, kR)

    if len(L) != len(R) or L[0].shape != R[0].shape:
        raise ValueError(f"Left/right mismatch: kL={len(L)} kR={len(R)} dL={L[0].shape} dR={R[0].shape}")

    B = None
    if kB in data:
        try:
            Bb = _load_basis(data, kB)
            if len(Bb) == len(L) and Bb[0].shape == L[0].shape:
                B = Bb
        except Exception:
            B = None

    meta = {"npz_keys": keys, "used_keys": {"L": kL, "R": kR, "both": (kB if kB in keys else None)}}
    return L, R, B, meta


# -------------------------
# Commuting quadratic forms and spectra
# -------------------------

def commuting_quadratic_form(A: List[np.ndarray], B: List[np.ndarray]) -> np.ndarray:
    """
    Build M such that for v in R^k, X = sum_a v_a A[a]:
      Q(v) = sum_b ||[X, B[b]]||_HS^2 = v^T M v
    """
    k = len(A)
    if len(B) != k:
        raise ValueError("Expected same k on both sides for quadratic form construction.")
    M = np.zeros((k, k), dtype=float)

    # C[a][b] = [A[a], B[b]]
    C = [[comm(A[a], B[b]) for b in range(k)] for a in range(k)]
    for a in range(k):
        for ap in range(k):
            s = 0.0
            for b in range(k):
                s += hs_inner(C[a][b], C[ap][b])
            M[a, ap] = float(s)
    return 0.5 * (M + M.T)


def eigen_decomp_sym(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (eigs_sorted, V_sorted) where columns of V_sorted are eigenvectors.
    """
    eigs, V = np.linalg.eigh(M)
    idx = np.argsort(np.real(eigs))
    eigs = np.real(eigs[idx])
    V = np.real(V[:, idx])  # should be real symmetric input => real eigvecs
    return eigs, V


def dims_vs_eps(eigs: np.ndarray, eps_list: np.ndarray) -> np.ndarray:
    out = np.zeros((eps_list.size, 2), dtype=float)
    for i, eps in enumerate(eps_list):
        out[i, 0] = float(eps)
        out[i, 1] = float(np.sum(eigs <= eps))
    return out


def build_combo_operator(basis: List[np.ndarray], v: np.ndarray) -> np.ndarray:
    X = np.zeros_like(basis[0])
    for a in range(len(basis)):
        X += float(v[a]) * basis[a]
    X = hermitize(traceless(X))
    # DO NOT renormalize; the magnitude conveys something. But we can optionally HS-normalize for reporting:
    return X


def commutes_with_set(X: np.ndarray, basis: List[np.ndarray]) -> Dict[str, float]:
    vals = np.array([hs_norm(comm(X, A)) for A in basis], dtype=float)
    return {
        "max": float(vals.max()) if vals.size else 0.0,
        "mean": float(vals.mean()) if vals.size else 0.0,
        "median": float(np.median(vals)) if vals.size else 0.0,
    }


# -------------------------
# Within commuting subspaces: best pair minimizing ||[XL, XR]||
# -------------------------

def best_pair_commutator(L: List[np.ndarray], R: List[np.ndarray],
                         V_L: np.ndarray, V_R: np.ndarray,
                         nL: int, nR: int) -> Dict:
    """
    Search best pair within the first nL commuting eigenvectors for L
    and first nR commuting eigenvectors for R, minimizing ||[XL, XR]||_HS.

    Uses brute force over bases + small random combinations for robustness.
    """
    k = len(L)
    rng = np.random.default_rng(0)

    # Candidate vectors: basis eigenvectors + a few random combos within subspace
    cand_L = []
    cand_R = []

    for i in range(nL):
        v = V_L[:, i]
        cand_L.append(v / max(1e-15, np.linalg.norm(v)))
    for i in range(nR):
        v = V_R[:, i]
        cand_R.append(v / max(1e-15, np.linalg.norm(v)))

    # add random combos
    for _ in range(50):
        if nL > 0:
            a = rng.normal(size=nL)
            v = (V_L[:, :nL] @ a)
            v = v / max(1e-15, np.linalg.norm(v))
            cand_L.append(v)
        if nR > 0:
            b = rng.normal(size=nR)
            w = (V_R[:, :nR] @ b)
            w = w / max(1e-15, np.linalg.norm(w))
            cand_R.append(w)

    best = {"score": float("inf"), "vL": None, "vR": None}

    for v in cand_L:
        XL = build_combo_operator(L, v)
        for w in cand_R:
            XR = build_combo_operator(R, w)
            score = hs_norm(comm(XL, XR))
            if score < best["score"]:
                best = {"score": float(score), "vL": v.copy(), "vR": w.copy()}

    if best["vL"] is None:
        return {"score": None}

    XL = build_combo_operator(L, best["vL"])
    XR = build_combo_operator(R, best["vR"])

    return {
        "score": float(best["score"]),
        "vL": best["vL"],
        "vR": best["vR"],
        "XL": XL,
        "XR": XR,
        "XL_norm": float(hs_norm(XL)),
        "XR_norm": float(hs_norm(XR)),
    }


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--mode", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--outdir", type=str, default=os.path.join("hsf_out", "gauss_diag"))
    ap.add_argument("--eps_commute", type=float, default=1e-6)
    ap.add_argument("--eps_sweep_min", type=float, default=1e-14)
    ap.add_argument("--eps_sweep_max", type=float, default=1e-2)
    ap.add_argument("--eps_sweep_n", type=int, default=25)
    args = ap.parse_args()

    L, R, B, meta = load_npz_bases(args.npz, args.mode)
    k = len(L)
    d = int(L[0].shape[0])

    # Build commuting quadratic forms
    ML = commuting_quadratic_form(L, R)
    MR = commuting_quadratic_form(R, L)

    eigL, VL = eigen_decomp_sym(ML)
    eigR, VR = eigen_decomp_sym(MR)

    # Commuting dimensions at eps_commute
    dimL = int(np.sum(eigL <= args.eps_commute))
    dimR = int(np.sum(eigR <= args.eps_commute))

    # Extract smallest-eigenvalue commuting directions (always take the minimum)
    vL_min = VL[:, 0]
    vR_min = VR[:, 0]
    vL_min = vL_min / max(1e-15, np.linalg.norm(vL_min))
    vR_min = vR_min / max(1e-15, np.linalg.norm(vR_min))

    XL_min = build_combo_operator(L, vL_min)
    XR_min = build_combo_operator(R, vR_min)

    # Centrality checks
    XL_vs_R = commutes_with_set(XL_min, R)   # should be tiny if eig is tiny
    XR_vs_L = commutes_with_set(XR_min, L)

    XL_vs_L = commutes_with_set(XL_min, L)
    XR_vs_R = commutes_with_set(XR_min, R)

    # Sweep eps -> commuting dim
    eps_list = np.logspace(np.log10(args.eps_sweep_min), np.log10(args.eps_sweep_max), args.eps_sweep_n)
    dims_eps_L = dims_vs_eps(eigL, eps_list)
    dims_eps_R = dims_vs_eps(eigR, eps_list)

    # Best pair commutator within commuting subspaces (if any)
    # If dim=0 at eps_commute, still use 1 to include the minimal eigenvector.
    nL = max(1, dimL)
    nR = max(1, dimR)
    bestpair = best_pair_commutator(L, R, VL, VR, nL=nL, nR=nR)

    # Optional: if B exists and matches, compute its ML/MR too (informational)
    both_info = None
    if B is not None and len(B) == k and B[0].shape == (d, d):
        MB_L = commuting_quadratic_form(B, R)
        MB_R = commuting_quadratic_form(B, L)
        eBL, _ = eigen_decomp_sym(MB_L)
        eBR, _ = eigen_decomp_sym(MB_R)
        both_info = {
            "both_vs_R_min_eig": float(eBL[0]),
            "both_vs_L_min_eig": float(eBR[0]),
        }

    # Package JSON (avoid dumping huge matrices fully)
    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "npz": args.npz,
            "mode": args.mode,
            "params": {
                "eps_commute": float(args.eps_commute),
                "eps_sweep_min": float(args.eps_sweep_min),
                "eps_sweep_max": float(args.eps_sweep_max),
                "eps_sweep_n": int(args.eps_sweep_n),
            },
            **meta,
        },
        "shapes": {"k_generators": int(k), "d_link": int(d)},
        "commuting_spectrum": {
            "eps_commute": float(args.eps_commute),
            "L_commuting_dim_est": dimL,
            "R_commuting_dim_est": dimR,
            "L_min_eigs": eigL[:min(10, k)].tolist(),
            "R_min_eigs": eigR[:min(10, k)].tolist(),
        },
        "commuting_direction_min": {
            "vL_min": vL_min.tolist(),
            "vR_min": vR_min.tolist(),
            "XL_min_norm": float(hs_norm(XL_min)),
            "XR_min_norm": float(hs_norm(XR_min)),
            "XL_commutes_with_R": XL_vs_R,
            "XR_commutes_with_L": XR_vs_L,
            "XL_commutes_with_L": XL_vs_L,
            "XR_commutes_with_R": XR_vs_R,
            "commutator_XL_XR_norm": float(hs_norm(comm(XL_min, XR_min))),
        },
        "dims_vs_eps": {
            "L": dims_eps_L.tolist(),
            "R": dims_eps_R.tolist(),
            "note": "Rows are [eps, dim(eig<=eps)]. Use this to see if commuting dimension grows toward 8 as you change generation settings.",
        },
        "best_pair_within_commuting_subspaces": {
            "nL_used": int(nL),
            "nR_used": int(nR),
            "score_commutator_norm": bestpair.get("score"),
            "XL_norm": bestpair.get("XL_norm"),
            "XR_norm": bestpair.get("XR_norm"),
        },
        "basis_both_info": both_info,
    }

    # Save artifacts NPZ (operators/vectors/eigs)
    ensure_dir(args.outdir)
    base = f"{out['meta']['timestamp']}_link_diag_{args.mode}_v7_2"
    json_path = os.path.join(args.outdir, base + ".json")
    npz_path = os.path.join(args.outdir, base + "_artifacts.npz")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Save big arrays in NPZ (compressed)
    np.savez_compressed(
        npz_path,
        vL_min=vL_min,
        vR_min=vR_min,
        XL_min=XL_min,
        XR_min=XR_min,
        eigs_ML=eigL,
        eigs_MR=eigR,
        V_ML=VL,
        V_MR=VR,
        dims_vs_eps_L=dims_eps_L,
        dims_vs_eps_R=dims_eps_R,
        # best pair if available
        best_vL=(bestpair["vL"] if "vL" in bestpair and bestpair["vL"] is not None else np.zeros(k)),
        best_vR=(bestpair["vR"] if "vR" in bestpair and bestpair["vR"] is not None else np.zeros(k)),
        best_XL=(bestpair["XL"] if "XL" in bestpair and bestpair["XL"] is not None else np.zeros((d, d), dtype=complex)),
        best_XR=(bestpair["XR"] if "XR" in bestpair and bestpair["XR"] is not None else np.zeros((d, d), dtype=complex)),
    )

    print(json.dumps(out, indent=2))
    print(f"\n[SAVED] {json_path}")
    print(f"[SAVED] {npz_path}")


if __name__ == "__main__":
    main()