#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_alignment_su3_v1.py
========================

Purpose
-------
Your Gauss-covariant Hamiltonian failed because the extracted link generators (L^a, R^a)
are an su(3) basis that is *rotated* relative to the site su(3) basis Q^a. Contracting
index-by-index assumes the same basis ordering/identification, which is false.

This script:
  1) Loads extracted (L^a, R^a) from your LR NPZ.
  2) Builds the canonical site basis Q^a (HS-orthonormal Gell-Mann-like).
  3) Computes structure constants f^{abc} for Q, L, R:
        f^{abc} = (1/(2i)) Tr( [T^a, T^b] T^c )
     with HS-orthonormal Hermitian traceless generators.
  4) Solves for an orthogonal alignment matrix O (8x8) such that, for the adjoint matrices
        (ad_a)_{bc} = f^{abc},
     we have approximately:
        ad_site_a  ≈  O ad_link_a O^T   for all a
     (a joint orthogonal Procrustes problem with shared O).
  5) Applies O to rotate the link generators into the site basis:
        L_aligned^a = Σ_i O_{a i} L^i
        R_aligned^a = Σ_i O_{a i} R^i
     (we solve separate O_L and O_R, because left/right extractions can differ by a rotation).
  6) Rebuilds the gauge-covariant Hamiltonian:
        H = Σ_a [ Q_x^a ⊗ L_aligned^a ⊗ I  +  I ⊗ R_aligned^a ⊗ Q_y^a ]
     and re-evaluates Gauss commutators.

Run (single-line Windows)
-------------------------
Aligned:
python gauss_alignment_su3_v1.py --lr_npz "C:\GitHub\hilbert_substrate\experiments\hsf_out\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz" --echo_model aligned --tol 1e-10 --restarts 6 --maxiter 400

Mixed:
python gauss_alignment_su3_v1.py --lr_npz "C:\GitHub\hilbert_substrate\experiments\hsf_out\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz" --echo_model mixed --tol 1e-10 --restarts 6 --maxiter 400

Outputs
-------
Writes JSON to ./hsf_out/gauss_alignment_su3_v1_<timestamp>.json

Notes
-----
- Requires scipy for expm + optimize.
- If you get a local minimum, increase --restarts (e.g. 20).
"""

import os
import json
import math
import argparse
from datetime import datetime

import numpy as np

try:
    from scipy.linalg import expm
    from scipy.optimize import minimize
except Exception as e:
    raise RuntimeError("scipy is required (scipy.linalg.expm and scipy.optimize.minimize).") from e


# -------------------------
# Utils
# -------------------------

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_out_dir() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def hermitize(A: np.ndarray) -> np.ndarray:
    return (A + A.conj().T) / 2.0

def traceless(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d, dtype=complex)

def hs_inner(A: np.ndarray, B: np.ndarray) -> complex:
    return np.trace(A.conj().T @ B)

def hs_norm(A: np.ndarray) -> float:
    return float(np.sqrt(max(hs_inner(A, A).real, 0.0)))

def normalize_hs(A: np.ndarray, tol: float = 1e-30) -> np.ndarray:
    n = hs_norm(A)
    if n < tol:
        return A.copy()
    return A / n

def gram_schmidt_hs(basis, tol=1e-12):
    out = []
    for A in basis:
        B = A.copy()
        for Q in out:
            B -= hs_inner(Q, B) * Q
        n = hs_norm(B)
        if n > tol:
            out.append(B / n)
    return out

def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for X in ops[1:]:
        out = np.kron(out, X)
    return out

def skew_from_params(p: np.ndarray, n: int) -> np.ndarray:
    """
    Map length n(n-1)/2 parameter vector to n×n real skew-symmetric matrix.
    Order: fill upper triangle row-major.
    """
    K = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = p[idx]
            K[j, i] = -p[idx]
            idx += 1
    return K

def params_dim(n: int) -> int:
    return (n * (n - 1)) // 2


# -------------------------
# su(d) generators (HS-orthonormal, Hermitian, traceless)
# -------------------------

def su_generators_gellmann(d: int):
    gens = []

    for i in range(d):
        for j in range(i + 1, d):
            S = np.zeros((d, d), dtype=complex)
            S[i, j] = 1.0
            S[j, i] = 1.0
            gens.append(S)

            A = np.zeros((d, d), dtype=complex)
            A[i, j] = -1j
            A[j, i] = 1j
            gens.append(A)

    for k in range(1, d):
        D = np.zeros((d, d), dtype=complex)
        for i in range(k):
            D[i, i] = 1.0
        D[k, k] = -float(k)
        D = D * math.sqrt(2.0 / (k * (k + 1.0)))
        gens.append(D)

    gens = [normalize_hs(traceless(hermitize(G))) for G in gens]
    gens = gram_schmidt_hs(gens, tol=1e-12)
    return gens


# -------------------------
# Load extracted L/R bases from NPZ
# -------------------------

def load_lr_bases_npz(npz_path: str, echo_model: str):
    data = np.load(npz_path)
    keyL = f"basis_left_{echo_model}"
    keyR = f"basis_right_{echo_model}"
    if keyL not in data.files or keyR not in data.files:
        raise RuntimeError(f"NPZ missing keys {keyL} and/or {keyR}. Found: {data.files}")

    L_ops = data[keyL]
    R_ops = data[keyR]

    L_ops = [normalize_hs(traceless(hermitize(L_ops[i]))) for i in range(L_ops.shape[0])]
    R_ops = [normalize_hs(traceless(hermitize(R_ops[i]))) for i in range(R_ops.shape[0])]
    return L_ops, R_ops


# -------------------------
# Structure constants / adjoint matrices
# -------------------------

def structure_constants(T: list[np.ndarray]) -> np.ndarray:
    """
    Returns f[a,b,c] real tensor using:
      f^{abc} = (1/(2i)) Tr( [T^a, T^b] T^c )

    Assumes HS-orthonormal basis: Tr(Ta Tb) = δ_ab
    """
    n = len(T)
    f = np.zeros((n, n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            C = comm(T[a], T[b])
            for c in range(n):
                val = (1.0 / (2.0j)) * np.trace(C @ T[c])
                f[a, b, c] = float(np.real_if_close(val).real)
    return f

def adjoint_matrices(f: np.ndarray) -> list[np.ndarray]:
    """
    ad[a] is n×n matrix with (ad[a])_{b c} = f[a,b,c]
    """
    n = f.shape[0]
    return [f[a, :, :].copy() for a in range(n)]


# -------------------------
# Alignment solve: find O minimizing sum_a ||A_a - O B_a O^T||_F^2
# -------------------------

def align_adjoint(A_list: list[np.ndarray], B_list: list[np.ndarray],
                  restarts: int, maxiter: int, seed: int):
    """
    Returns best O (8x8) and objective.
    """
    rng = np.random.default_rng(seed)
    n = A_list[0].shape[0]
    assert all(M.shape == (n, n) for M in A_list)
    assert all(M.shape == (n, n) for M in B_list)

    # Pre-pack arrays for speed
    A = np.stack(A_list, axis=0)  # (k,n,n)
    B = np.stack(B_list, axis=0)  # (k,n,n)
    k = A.shape[0]

    def objective(p):
        K = skew_from_params(p, n)
        O = expm(K)  # orthogonal (up to numeric drift)
        # Compute sum Frobenius
        s = 0.0
        OT = O.T
        for i in range(k):
            D = A[i] - (O @ B[i] @ OT)
            s += float(np.sum(D * D))
        return s

    best = {"val": float("inf"), "O": np.eye(n), "p": None, "nit": None, "success": False}

    # restarts: include identity as one start
    starts = []
    starts.append(np.zeros(params_dim(n), dtype=float))
    for _ in range(max(0, restarts - 1)):
        # random skew small
        p0 = rng.normal(scale=0.2, size=params_dim(n))
        starts.append(p0)

    for idx, p0 in enumerate(starts):
        res = minimize(
            objective,
            p0,
            method="L-BFGS-B",
            options={"maxiter": int(maxiter), "ftol": 1e-12}
        )
        val = float(res.fun)
        if val < best["val"]:
            K = skew_from_params(res.x, n)
            O = expm(K)
            # re-orthonormalize via SVD to clean numerical drift
            U, _, Vt = np.linalg.svd(O, full_matrices=False)
            O = U @ Vt
            best = {"val": val, "O": O, "p": res.x.tolist(), "nit": int(res.nit), "success": bool(res.success)}

    return best


# -------------------------
# Rotate generator lists by O
# -------------------------

def rotate_generators(T: list[np.ndarray], O: np.ndarray) -> list[np.ndarray]:
    """
    New basis:
      T_new[a] = Σ_i O[a,i] T[i]
    """
    n = len(T)
    out = []
    for a in range(n):
        M = np.zeros_like(T[0])
        for i in range(n):
            M += float(O[a, i]) * T[i]
        out.append(normalize_hs(traceless(hermitize(M))))
    # Optional: re-orthonormalize (tiny drift)
    out = gram_schmidt_hs(out, tol=1e-12)
    return out


# -------------------------
# Gauss commutators for H_gauge
# -------------------------

def gauss_commutators(H: np.ndarray, Q: list[np.ndarray], L: list[np.ndarray], R: list[np.ndarray]):
    """
    dims 3 ⊗ 3 ⊗ 3 in ordering x ⊗ link ⊗ y.

    G_left^a  = Q_x^a ⊗ I ⊗ I  +  I ⊗ L^a ⊗ I
    G_right^a = I ⊗ I ⊗ Q_y^a  -  I ⊗ R^a ⊗ I
    """
    d = 3
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
        "gauss_left_mean": float(np.mean(left)),
        "gauss_left_median": float(np.median(left)),
        "gauss_right_max": float(np.max(right)),
        "gauss_right_mean": float(np.mean(right)),
        "gauss_right_median": float(np.median(right)),
        "per_generator_left": [float(x) for x in left.tolist()],
        "per_generator_right": [float(x) for x in right.tolist()],
    }

def build_H_gauge(Q: list[np.ndarray], L: list[np.ndarray], R: list[np.ndarray]):
    d = 3
    I = np.eye(d, dtype=complex)
    H = np.zeros((d*d*d, d*d*d), dtype=complex)
    for a in range(len(Q)):
        H += kron(Q[a], L[a], I) + kron(I, R[a], Q[a])
    H = hermitize(H)
    nrm = hs_norm(H)
    if nrm > 0:
        H = H / nrm
    return H


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_npz", type=str, required=True)
    ap.add_argument("--echo_model", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Site generators
    Q = su_generators_gellmann(3)
    if len(Q) != 8:
        raise RuntimeError("Expected 8 site generators for su(3).")

    # Link extracted generators
    L_raw, R_raw = load_lr_bases_npz(args.lr_npz, args.echo_model)
    if len(L_raw) != 8 or len(R_raw) != 8:
        raise RuntimeError("Expected 8 link generators for L and R.")

    # Baseline (unaligned) Gauss
    H0 = build_H_gauge(Q, L_raw, R_raw)
    gauss0 = gauss_commutators(H0, Q, L_raw, R_raw)

    # Compute adjoint matrices
    fQ = structure_constants(Q)
    fL = structure_constants(L_raw)
    fR = structure_constants(R_raw)

    adQ = adjoint_matrices(fQ)
    adL = adjoint_matrices(fL)
    adR = adjoint_matrices(fR)

    # Align L to Q and R to Q independently
    bestL = align_adjoint(adQ, adL, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 11)
    bestR = align_adjoint(adQ, adR, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 29)

    OL = np.array(bestL["O"], dtype=float)
    OR = np.array(bestR["O"], dtype=float)

    # Rotate generators
    L_aligned = rotate_generators(L_raw, OL)
    R_aligned = rotate_generators(R_raw, OR)

    # Aligned Gauss
    H1 = build_H_gauge(Q, L_aligned, R_aligned)
    gauss1 = gauss_commutators(H1, Q, L_aligned, R_aligned)

    passes = (gauss1["gauss_left_max"] <= args.tol) and (gauss1["gauss_right_max"] <= args.tol)

    # Report
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
            "gauss": gauss0,
        },
        "alignment": {
            "L": {k: bestL[k] for k in bestL if k != "O"},
            "R": {k: bestR[k] for k in bestR if k != "O"},
            "O_L": OL.tolist(),
            "O_R": OR.tolist(),
        },
        "after_alignment": {
            "gauss": gauss1,
            "passes_tol": bool(passes),
        },
    }

    # Print summary
    print("============================================================")
    print("SU(3) ALIGNMENT + GAUSS TEST")
    print("------------------------------------------------------------")
    print(f"echo_model={args.echo_model}  tol={args.tol:.2e}  restarts={args.restarts}  maxiter={args.maxiter}")
    print("Baseline (unaligned) H_gauge:")
    print(f"  left : max={gauss0['gauss_left_max']:.3e}  median={gauss0['gauss_left_median']:.3e}  mean={gauss0['gauss_left_mean']:.3e}")
    print(f"  right: max={gauss0['gauss_right_max']:.3e}  median={gauss0['gauss_right_median']:.3e}  mean={gauss0['gauss_right_mean']:.3e}")
    print("Alignment objective (lower is better):")
    print(f"  L: val={bestL['val']:.6e}  nit={bestL['nit']}  success={bestL['success']}")
    print(f"  R: val={bestR['val']:.6e}  nit={bestR['nit']}  success={bestR['success']}")
    print("After alignment H_gauge:")
    print(f"  left : max={gauss1['gauss_left_max']:.3e}  median={gauss1['gauss_left_median']:.3e}  mean={gauss1['gauss_left_mean']:.3e}")
    print(f"  right: max={gauss1['gauss_right_max']:.3e}  median={gauss1['gauss_right_median']:.3e}  mean={gauss1['gauss_right_mean']:.3e}")
    print(f"  passes_tol={passes}")
    print("============================================================")

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"gauss_alignment_su3_v1_{out['meta']['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()