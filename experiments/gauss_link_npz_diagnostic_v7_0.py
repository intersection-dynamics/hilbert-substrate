#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_link_npz_diagnostic_v7_0.py
================================

Generalizes v6.3 to arbitrary link dimension d_link (e.g. 3, 9, 27...).

What it expects
---------------
An NPZ containing at least:
  basis_left_<mode>   : shape (8, d, d)
  basis_right_<mode>  : shape (8, d, d)

Optionally:
  basis_both_<mode>   : shape (8, d, d)

where <mode> is "aligned" or "mixed".

What it does
------------
A) Cleans & HS-normalizes each generator:
     T[a] := normalize_hs(traceless(hermitize(T[a])))
   then checks orthonormality (Gram) and HS norms.

B) su(3) closure diagnostics (corrected):
   - Computes structure constants:
       f^{abc} = (1/(2i)) Tr( [T^a, T^b] T^c )
   - Computes corrected closure residual by projecting:
       Y = -i [T^a, T^b]    (Hermitian)
     onto span{T^c} using Gram pseudo-inverse.

C) Cross-endpoint diagnostics:
   - Grid of commutator norms M[a,b] = ||[L[a], R[b]]||_HS
   - Cross energy: E = sum_{a,b} M[a,b]^2

D) Best-effort generator-space rotations O_L, O_R ∈ SO(8) to minimize cross energy:
       E(O_L, O_R) = sum_{a,b} ||[L', R']||^2
     using block-coordinate descent with Lie-parameterized updates.

E) "Commuting content" spectra (raw and optimized):
   - Builds quadratic form M_L where for X = sum_a v_a L[a]:
       Q(v) = sum_b ||[X, R[b]]||^2 = v^T M_L v
     Eigenvalues near 0 indicate directions in span(L) commuting with all R.
   - Same for span(R) commuting with all L.

F) Writes JSON output to disk:
     hsf_out/gauss_diag/<timestamp>_link_diag_<mode>_v7_0.json

Usage
-----
python gauss_link_npz_diagnostic_v7_0.py --npz yourfile.npz --mode aligned
python gauss_link_npz_diagnostic_v7_0.py --npz yourfile.npz --mode mixed

Tuning
------
--restarts   : random restarts for O_L/O_R optimization
--outer      : outer block-coordinate iterations
--maxiter    : L-BFGS inner iterations
--eps_commute: eigenvalue threshold to count "commuting dimensions"

Notes
-----
- This script is dimension-agnostic: it never assumes d=3 beyond "8 generators" for su(3).
- If you move to link dim 9 (two-qutrit), the same harness will work unchanged.
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
# Gram / Orthonormality diagnostics
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


# -------------------------
# su(3) structure constants and invariants
# -------------------------

def structure_constants(T: List[np.ndarray]) -> np.ndarray:
    """
    For Hermitian basis T_a, define:
      f^{abc} = (1/(2i)) Tr( [T^a, T^b] T^c )

    If basis is HS-orthonormal, then:
      -i [T^a, T^b] = sum_c f^{abc} T^c
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


def f_invariants(f: np.ndarray) -> Dict[str, float]:
    return {
        "f_fro": float(np.linalg.norm(f.ravel())),
        "f_maxabs": float(np.max(np.abs(f))),
    }


def closure_residual_corrected(T: List[np.ndarray]) -> Dict[str, float]:
    """
    Corrected closure residual for Hermitian basis:
    project Y = -i [T_a, T_b] (Hermitian) onto span{T_c}.
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
            Y = (-1.0j) * comm(T[a], T[b])  # Hermitian
            P = project(Y)
            res.append(hs_norm(Y - P))

    v = np.array(res, dtype=float)
    return {"max": float(v.max()), "median": float(np.median(v)), "mean": float(np.mean(v))}


# -------------------------
# Cross commutators
# -------------------------

def commutator_grid(L: List[np.ndarray], R: List[np.ndarray]) -> np.ndarray:
    n = len(L)
    m = len(R)
    M = np.zeros((n, m), dtype=float)
    for a in range(n):
        for b in range(m):
            M[a, b] = hs_norm(comm(L[a], R[b]))
    return M


def commutator_grid_stats(M: np.ndarray) -> Dict[str, float]:
    v = M.ravel()
    return {
        "max": float(np.max(v)),
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "min": float(np.min(v)),
    }


def cross_energy_from_grid(M: np.ndarray) -> float:
    return float(np.sum(M * M))


# -------------------------
# Generator-space rotations O in SO(8)
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


def nearest_orthogonal(O: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(O, full_matrices=False)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def rotate_generators(T: List[np.ndarray], O: np.ndarray) -> List[np.ndarray]:
    """
    T'_a = sum_i O[a,i] T_i
    Per-generator numerical cleanup only; no Gram-Schmidt.
    """
    n = len(T)
    out: List[np.ndarray] = []
    for a in range(n):
        M = np.zeros_like(T[0])
        for i in range(n):
            M += float(O[a, i]) * T[i]
        out.append(normalize_hs(traceless(hermitize(M))))
    return out


def optimize_cross_commutators(L: List[np.ndarray], R: List[np.ndarray],
                               restarts: int, maxiter: int,
                               n_outer: int, seed: int) -> Dict:
    """
    Block-coordinate descent over (O_L, O_R) in SO(8) to minimize
      E = sum_{a,b} ||[L'_a, R'_b]||_HS^2
    """
    rng = np.random.default_rng(seed)
    n = len(L)
    if n != 8 or len(R) != 8:
        raise ValueError(f"Expected 8 generators per side (su(3)); got L={len(L)} R={len(R)}")

    def make_start() -> np.ndarray:
        p = rng.normal(scale=0.25, size=params_dim(n))
        O = expm(skew_from_params(p, n))
        return nearest_orthogonal(O)

    starts = [(np.eye(n), np.eye(n))] + [(make_start(), make_start()) for _ in range(max(0, restarts - 1))]

    def energy_from(O_L: np.ndarray, O_R: np.ndarray) -> float:
        Lr = rotate_generators(L, O_L)
        Rr = rotate_generators(R, O_R)
        M = commutator_grid(Lr, Rr)
        return cross_energy_from_grid(M)

    best = {"E": float("inf"), "O_L": np.eye(n), "O_R": np.eye(n), "history": None}

    for (O_L0, O_R0) in starts:
        O_L = O_L0.copy()
        O_R = O_R0.copy()
        hist = []

        for outer in range(n_outer):
            # Optimize O_L with O_R fixed, via left-multiplicative update expm(K) @ O_L
            def obj_L(p: np.ndarray) -> float:
                O = nearest_orthogonal(expm(skew_from_params(p, n)) @ O_L)
                return energy_from(O, O_R)

            resL = minimize(obj_L, np.zeros(params_dim(n)), method="L-BFGS-B",
                            options={"maxiter": int(maxiter), "ftol": 1e-12})
            O_L = nearest_orthogonal(expm(skew_from_params(resL.x, n)) @ O_L)

            # Optimize O_R with O_L fixed
            def obj_R(p: np.ndarray) -> float:
                O = nearest_orthogonal(expm(skew_from_params(p, n)) @ O_R)
                return energy_from(O_L, O)

            resR = minimize(obj_R, np.zeros(params_dim(n)), method="L-BFGS-B",
                            options={"maxiter": int(maxiter), "ftol": 1e-12})
            O_R = nearest_orthogonal(expm(skew_from_params(resR.x, n)) @ O_R)

            E = energy_from(O_L, O_R)
            hist.append({"outer": int(outer), "E": float(E)})

            if outer >= 2 and abs(hist[-1]["E"] - hist[-2]["E"]) < 1e-10:
                break

        E_final = hist[-1]["E"] if hist else energy_from(O_L, O_R)
        if E_final < best["E"]:
            best = {"E": float(E_final), "O_L": O_L, "O_R": O_R, "history": hist}

    return best


# -------------------------
# Commuting-content spectra
# -------------------------

def commuting_quadratic_form(A: List[np.ndarray], B: List[np.ndarray]) -> np.ndarray:
    """
    Build M such that for v in R^8, X = sum_a v_a A[a]:
      Q(v) = sum_b ||[X, B[b]]||_HS^2 = v^T M v
    """
    n = len(A)
    if n != 8 or len(B) != 8:
        raise ValueError("Expected 8 generators for su(3) diagnostics.")
    M = np.zeros((n, n), dtype=float)

    C = [[comm(A[a], B[b]) for b in range(n)] for a in range(n)]
    for a in range(n):
        for ap in range(n):
            s = 0.0
            for b in range(n):
                s += hs_inner(C[a][b], C[ap][b])
            M[a, ap] = float(s)
    return 0.5 * (M + M.T)


def commuting_spectrum(L: List[np.ndarray], R: List[np.ndarray], eps: float) -> Dict:
    ML = commuting_quadratic_form(L, R)
    MR = commuting_quadratic_form(R, L)
    evals_L = np.sort(np.real(np.linalg.eigvalsh(ML)))
    evals_R = np.sort(np.real(np.linalg.eigvalsh(MR)))
    return {
        "eps": float(eps),
        "L_vs_R_eigs": evals_L.tolist(),
        "R_vs_L_eigs": evals_R.tolist(),
        "L_commuting_dim_est": int(np.sum(evals_L <= eps)),
        "R_commuting_dim_est": int(np.sum(evals_R <= eps)),
        "L_min_eigs": evals_L[:3].tolist(),
        "R_min_eigs": evals_R[:3].tolist(),
    }


# -------------------------
# Load NPZ bases (dimension-agnostic)
# -------------------------

def _get_key(mode: str, which: str) -> str:
    return f"basis_{which}_{mode}"


def _load_basis(data: np.lib.npyio.NpzFile, key: str) -> List[np.ndarray]:
    arr = data[key]
    if arr.ndim != 3 or arr.shape[0] != 8 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Key {key} has shape {arr.shape}; expected (8,d,d).")
    d = int(arr.shape[1])
    out = []
    for a in range(8):
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

    # Optional "both"
    B = None
    if kB in data:
        try:
            B = _load_basis(data, kB)
        except Exception:
            B = None

    meta = {"npz_keys": keys, "used_keys": {"L": kL, "R": kR, "both": (kB if kB in keys else None)}}
    return L, R, B, meta


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--mode", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=os.path.join("hsf_out", "gauss_diag"))
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--outer", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--eps_commute", type=float, default=1e-6)
    args = ap.parse_args()

    L, R, B, meta = load_npz_bases(args.npz, args.mode)
    d_link = int(L[0].shape[0])

    # Orthonormality
    GL = gram_matrix(L)
    GR = gram_matrix(R)

    # Invariants + closure residual (correct)
    fL = structure_constants(L)
    fR = structure_constants(R)
    closure_L = closure_residual_corrected(L)
    closure_R = closure_residual_corrected(R)

    # Cross commutators raw
    M_raw = commutator_grid(L, R)
    raw_stats = commutator_grid_stats(M_raw)
    E_raw = cross_energy_from_grid(M_raw)

    # Commuting spectrum raw
    spec_raw = commuting_spectrum(L, R, eps=args.eps_commute)

    # Optimize
    opt = optimize_cross_commutators(L, R, restarts=args.restarts, maxiter=args.maxiter,
                                     n_outer=args.outer, seed=args.seed + 17)
    O_L = opt["O_L"]
    O_R = opt["O_R"]
    L_opt = rotate_generators(L, O_L)
    R_opt = rotate_generators(R, O_R)

    M_opt = commutator_grid(L_opt, R_opt)
    opt_stats = commutator_grid_stats(M_opt)
    E_opt = cross_energy_from_grid(M_opt)

    # Commuting spectrum optimized
    spec_opt = commuting_spectrum(L_opt, R_opt, eps=args.eps_commute)

    # Optional basis_both
    both_info = None
    if B is not None:
        GB = gram_matrix(B)
        fB = structure_constants(B)
        both_info = {
            "d_link": int(B[0].shape[0]),
            "gram_stats": ortho_stats(GB),
            "invariants": f_invariants(fB),
            "closure_residual_corrected": closure_residual_corrected(B),
        }

    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "npz": args.npz,
            "mode": args.mode,
            "seed": int(args.seed),
            "params": {
                "restarts": int(args.restarts),
                "outer": int(args.outer),
                "maxiter": int(args.maxiter),
                "eps_commute": float(args.eps_commute),
            },
            **meta,
        },
        "shapes": {"n_gen": 8, "d_link": int(d_link)},
        "orthonormality": {
            "L_gram_stats": ortho_stats(GL),
            "R_gram_stats": ortho_stats(GR),
            "L_hs_norms": [hs_norm(x) for x in L],
            "R_hs_norms": [hs_norm(x) for x in R],
        },
        "su3_invariants": {
            "L": f_invariants(fL),
            "R": f_invariants(fR),
        },
        "closure_residual_corrected": {
            "L": closure_L,
            "R": closure_R,
            "note": "Projects (-i[T_a,T_b]) onto span{T_c} (Hermitian closure test).",
        },
        "cross_commutators": {
            "raw": {"stats": raw_stats, "energy_sum_sq": float(E_raw)},
            "optimized": {
                "stats": opt_stats,
                "energy_sum_sq": float(E_opt),
                "improvement_factor": float(E_raw / max(1e-15, E_opt)),
                "history": opt["history"],
            },
            "note": "If E_opt ~ E_raw and both O(1), there is no SO(8) endpoint relabeling that yields commuting L/R on this link space.",
        },
        "commuting_content": {
            "raw": spec_raw,
            "optimized": spec_opt,
            "note": "Eigenvalues near 0 indicate directions that commute with the opposite endpoint set.",
        },
        "basis_both": both_info,
    }

    # Save JSON
    ensure_dir(args.outdir)
    fname = f"{out['meta']['timestamp']}_link_diag_{args.mode}_v7_0.json"
    fpath = os.path.join(args.outdir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\n[SAVED] {fpath}")


if __name__ == "__main__":
    main()