#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_link_qutrit_npz_diagnostic_v6_3.py
=======================================

v6.3 goals (from our fork)
--------------------------
1) Fix closure residual measurement:
   - For Hermitian generators {T_a}, commutator [T_a, T_b] is anti-Hermitian.
   - The su(3) Lie closure in the Hermitian basis is expressed via:
         -i [T_a, T_b] = sum_c f^{abc} T_c
   - Therefore, closure residual must project (-i [T_a, T_b]) onto span{T_c},
     not [T_a, T_b].

2) Measure "commuting content" between extracted endpoints:
   - Compute the full 8x8 grid of commutator norms ||[L_a, R_b]||_HS.
   - Compute best-effort generator-space rotations O_L, O_R in SO(8) that
     minimize the cross-commutator energy:
         E(O_L, O_R) = sum_{a,b} || [L'_a, R'_b] ||_HS^2
     where L' = rotate(L, O_L), R' = rotate(R, O_R).

3) Estimate how much of L can be made to commute with R (and vice versa):
   - Treat the map v (in generator space) -> operator X = sum_a v_a L_a.
   - Define a quadratic form:
         Q_L(v) = sum_b || [X, R_b] ||_HS^2  = v^T M_L v
     and similarly Q_R(w) = w^T M_R w.
   - Eigenvalues near 0 correspond to directions in span(L) that (approximately)
     commute with all R_b. We report the smallest eigenvalues + counts below eps.

This script:
- Loads L/R (and optional "both") from the NPZ in aligned or mixed mode.
- Computes invariants, orthonormality (Gram), corrected closure residual,
  raw cross-commutator stats.
- Runs a block-coordinate optimization over (O_L, O_R) using Lie algebra
  parameterization O=expm(K) with K skew-symmetric, minimizing E.
- Computes commuting-content eigen-spectra before and after optimization.
- Writes JSON to disk under hsf_out/gauss_diag/

Usage
-----
python gauss_link_qutrit_npz_diagnostic_v6_3.py --npz echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz --mode aligned
python gauss_link_qutrit_npz_diagnostic_v6_3.py --npz ... --mode mixed

Notes
-----
- This is purely diagnostic; it does not claim gauge structure exists on C^3.
- If optimized cross-commutator energy remains O(1), that's strong evidence
  a qutrit link cannot host a commuting endpoint split in your extracted regime.
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


# -------------------------
# Structure constants + adjoint rep
# -------------------------

def structure_constants(T: List[np.ndarray]) -> np.ndarray:
    """
    For Hermitian basis T_a, define:
      f^{abc} = (1/(2i)) Tr( [T^a, T^b] T^c )
    so that -i [T^a, T^b] = sum_c f^{abc} T^c  when basis is HS-orthonormal.
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


# -------------------------
# Gram / ortho diagnostics
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
# Corrected closure residual: project (-i [Ta,Tb]) onto span{Tc}
# -------------------------

def closure_residual_corrected(T: List[np.ndarray]) -> Dict[str, float]:
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
            C = comm(T[a], T[b])          # anti-Hermitian
            Y = (-1.0j) * C               # Hermitian
            P = project(Y)
            res.append(hs_norm(Y - P))

    v = np.array(res, dtype=float)
    return {"max": float(v.max()), "median": float(np.median(v)), "mean": float(v.mean())}


# -------------------------
# Cross commutator grid
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


def cross_energy(L: List[np.ndarray], R: List[np.ndarray]) -> float:
    M = commutator_grid(L, R)
    return float(np.sum(M * M))


# -------------------------
# Generator-space rotations O in SO(8): parameterize by expm(K), K skew
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
    # ensure det +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def rotate_generators(T: List[np.ndarray], O: np.ndarray) -> List[np.ndarray]:
    # No Gram-Schmidt; preserve index semantics
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
    Block-coordinate descent:
      - optimize O_L with O_R fixed
      - optimize O_R with O_L fixed
    minimizing E = sum_{a,b} ||[L'_a, R'_b]||^2.

    Returns best found O_L, O_R and energies.
    """
    rng = np.random.default_rng(seed)
    n = len(L)
    assert n == 8 and len(R) == 8

    def make_start() -> np.ndarray:
        p = rng.normal(scale=0.25, size=params_dim(n))
        K = skew_from_params(p, n)
        O = expm(K)
        return nearest_orthogonal(O)

    starts = [(np.eye(n), np.eye(n))] + [(make_start(), make_start()) for _ in range(max(0, restarts - 1))]

    best = {
        "E": float("inf"),
        "O_L": np.eye(n),
        "O_R": np.eye(n),
        "history": None,
    }

    # objective given O_L and O_R, by rotating once
    def energy_from(O_L: np.ndarray, O_R: np.ndarray) -> float:
        Lr = rotate_generators(L, O_L)
        Rr = rotate_generators(R, O_R)
        return cross_energy(Lr, Rr)

    for (O_L0, O_R0) in starts:
        O_L = O_L0.copy()
        O_R = O_R0.copy()
        hist = []

        for outer in range(n_outer):
            # Optimize O_L with O_R fixed
            def obj_L(p: np.ndarray) -> float:
                K = skew_from_params(p, n)
                O = nearest_orthogonal(expm(K) @ O_L)
                return energy_from(O, O_R)

            resL = minimize(obj_L, np.zeros(params_dim(n)), method="L-BFGS-B",
                            options={"maxiter": int(maxiter), "ftol": 1e-12})
            K = skew_from_params(resL.x, n)
            O_L = nearest_orthogonal(expm(K) @ O_L)

            # Optimize O_R with O_L fixed
            def obj_R(p: np.ndarray) -> float:
                K = skew_from_params(p, n)
                O = nearest_orthogonal(expm(K) @ O_R)
                return energy_from(O_L, O)

            resR = minimize(obj_R, np.zeros(params_dim(n)), method="L-BFGS-B",
                            options={"maxiter": int(maxiter), "ftol": 1e-12})
            K = skew_from_params(resR.x, n)
            O_R = nearest_orthogonal(expm(K) @ O_R)

            E = energy_from(O_L, O_R)
            hist.append({"outer": int(outer), "E": float(E)})

            # early stop if improvement stalls
            if outer >= 2 and abs(hist[-1]["E"] - hist[-2]["E"]) < 1e-10:
                break

        E_final = hist[-1]["E"] if hist else energy_from(O_L, O_R)
        if E_final < best["E"]:
            best = {"E": float(E_final), "O_L": O_L, "O_R": O_R, "history": hist}

    return best


# -------------------------
# Commuting-content spectra
# -------------------------

def commuting_quadratic_form_L_vs_R(L: List[np.ndarray], R: List[np.ndarray]) -> np.ndarray:
    """
    Build M_L such that for v in R^8, X = sum_a v_a L_a:
      Q_L(v) = sum_b ||[X, R_b]||_HS^2 = v^T M_L v
    """
    n = len(L)
    M = np.zeros((n, n), dtype=float)

    # Precompute commutators of basis elements with R_b:
    # C_{a,b} = [L_a, R_b], treat as matrix operator
    C = [[comm(L[a], R[b]) for b in range(n)] for a in range(n)]

    # Q(v) = sum_b || sum_a v_a C_{a,b} ||^2
    # Expand: v^T M v with M_{a,a'} = sum_b <C_{a,b}, C_{a',b}>
    for a in range(n):
        for ap in range(n):
            s = 0.0
            for b in range(n):
                s += hs_inner(C[a][b], C[ap][b])
            M[a, ap] = float(s)
    # Symmetrize numerically
    M = 0.5 * (M + M.T)
    return M


def commuting_spectrum(L: List[np.ndarray], R: List[np.ndarray], eps: float = 1e-6) -> Dict:
    ML = commuting_quadratic_form_L_vs_R(L, R)
    MR = commuting_quadratic_form_L_vs_R(R, L)  # swap roles
    evals_L = np.linalg.eigvalsh(ML)
    evals_R = np.linalg.eigvalsh(MR)
    evals_L = np.sort(np.real(evals_L))
    evals_R = np.sort(np.real(evals_R))
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
# Load NPZ
# -------------------------

def _get_key(mode: str, which: str) -> str:
    return f"basis_{which}_{mode}"


def load_npz_bases(path: str, mode: str) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]], Dict]:
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
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--mode", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=os.path.join("hsf_out", "gauss_diag"))
    ap.add_argument("--restarts", type=int, default=8, help="number of random restarts for O_L/O_R optimization")
    ap.add_argument("--maxiter", type=int, default=200, help="max iter per inner L-BFGS solve")
    ap.add_argument("--outer", type=int, default=6, help="outer block-coordinate iterations")
    ap.add_argument("--eps_commute", type=float, default=1e-6, help="eigenvalue threshold for commuting dim estimate")
    args = ap.parse_args()

    L, R, B, meta = load_npz_bases(args.npz, args.mode)

    # Invariants + ortho
    fL = structure_constants(L)
    fR = structure_constants(R)
    GL = gram_matrix(L)
    GR = gram_matrix(R)

    # Corrected closure residuals
    closure_L = closure_residual_corrected(L)
    closure_R = closure_residual_corrected(R)

    # Raw cross commutators
    M_raw = commutator_grid(L, R)
    raw_stats = commutator_grid_stats(M_raw)
    E_raw = float(np.sum(M_raw * M_raw))

    # Commuting spectrum before
    spec_raw = commuting_spectrum(L, R, eps=args.eps_commute)

    # Optimize rotations to reduce cross-commutators
    opt = optimize_cross_commutators(L, R, restarts=args.restarts, maxiter=args.maxiter,
                                     n_outer=args.outer, seed=args.seed + 17)
    O_L = opt["O_L"]
    O_R = opt["O_R"]
    L_opt = rotate_generators(L, O_L)
    R_opt = rotate_generators(R, O_R)

    M_opt = commutator_grid(L_opt, R_opt)
    opt_stats = commutator_grid_stats(M_opt)
    E_opt = float(np.sum(M_opt * M_opt))

    # Commuting spectrum after
    spec_opt = commuting_spectrum(L_opt, R_opt, eps=args.eps_commute)

    # Optional comparison of "both" to L/R via f-invariants (not full alignment here)
    both_info = None
    if B is not None:
        fB = structure_constants(B)
        GB = gram_matrix(B)
        both_info = {
            "invariants": f_invariants(fB),
            "gram_stats": ortho_stats(GB),
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
                "maxiter": int(args.maxiter),
                "outer": int(args.outer),
                "eps_commute": float(args.eps_commute),
            },
            **meta,
        },
        "shapes": {"L": [3, 3], "R": [3, 3], "n_gen": 8},
        "orthonormality": {
            "L_gram_stats": ortho_stats(GL),
            "R_gram_stats": ortho_stats(GR),
        },
        "invariants": {
            "L": f_invariants(fL),
            "R": f_invariants(fR),
        },
        "closure_residual_corrected": {
            "L": closure_L,
            "R": closure_R,
            "note": "This projects (-i[T_a,T_b]) onto span{T_c}, which is the correct Hermitian-space closure test.",
        },
        "cross_commutators": {
            "raw": {
                "stats": raw_stats,
                "energy_sum_sq": float(E_raw),
            },
            "optimized": {
                "stats": opt_stats,
                "energy_sum_sq": float(E_opt),
                "improvement_factor": float(E_raw / max(1e-15, E_opt)),
                "history": opt["history"],
            },
            "note": "If optimized energy remains O(1), there is no SO(8) relabeling that makes L/R approximately commuting on a qutrit link.",
        },
        "commuting_content": {
            "raw": spec_raw,
            "optimized": spec_opt,
            "note": "Eigenvalues near 0 indicate directions in span(L) (or span(R)) that commute with the other endpoint set.",
        },
        "basis_both": both_info,
    }

    # Save JSON
    ensure_dir(args.outdir)
    fname = f"{out['meta']['timestamp']}_qutrit_link_diag_{args.mode}_v6_3.json"
    fpath = os.path.join(args.outdir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\n[SAVED] {fpath}")


if __name__ == "__main__":
    main()