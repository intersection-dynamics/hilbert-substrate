#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_structure_constants_diagnostic_su3_v1.py
=============================================

What this does
--------------
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
   For each of L and R, we consider these variants of the generator list:
     - orig:          T
     - neg:           -T
     - transpose:     T^T
     - neg_transpose: -T^T

   (For su(N), the conjugate fundamental representation corresponds to -T^T.)

4) For each (L_variant, R_variant), it:
   - aligns L_variant to Q via adjoint-Procrustes (find O_L)
   - aligns R_variant to Q via adjoint-Procrustes (find O_R)
   - rotates generators into "site-index basis" using O_L / O_R
   - builds the gauge-covariant Hamiltonian:
        H = Σ_a [ Q_x^a ⊗ L_aligned^a ⊗ I + I ⊗ R_aligned^a ⊗ Q_y^a ]
   - measures Gauss commutators:
        [H, G_left^a],  G_left^a = Q_x^a + L^a
        [H, G_right^a], G_right^a = Q_y^a - R^a

This tells you *exactly* whether you're facing:
  - mere basis relabeling,
  - conjugate-rep issue (sign / transpose),
  - or a deeper structure-constants mismatch.

Run (single-line Windows)
-------------------------
python gauss_structure_constants_diagnostic_su3_v1.py --lr_npz "C:\GitHub\hilbert_substrate\experiments\hsf_out\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz" --echo_model aligned --tol 1e-10 --restarts 10 --maxiter 600

Then:
python gauss_structure_constants_diagnostic_su3_v1.py --lr_npz "C:\GitHub\hilbert_substrate\experiments\hsf_out\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz" --echo_model mixed --tol 1e-10 --restarts 10 --maxiter 600

Outputs
-------
Writes JSON to ./hsf_out/gauss_structure_constants_diagnostic_su3_v1_<timestamp>.json
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

def params_dim(n: int) -> int:
    return (n * (n - 1)) // 2

def skew_from_params(p: np.ndarray, n: int) -> np.ndarray:
    K = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = p[idx]
            K[j, i] = -p[idx]
            idx += 1
    return K


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
    return [f[a, :, :].copy() for a in range(f.shape[0])]

def f_invariants(f: np.ndarray) -> dict:
    # Frobenius norm, max abs, antisymmetry check stats
    fn = float(np.linalg.norm(f.reshape(-1)))
    mx = float(np.max(np.abs(f)))
    # antisymmetry in (a,b): f^{ab c} + f^{ba c} should be ~0
    anti = f + np.transpose(f, (1, 0, 2))
    anti_rms = float(np.sqrt(np.mean(anti * anti)))
    return {"f_fro": fn, "f_max_abs": mx, "antisym_rms_ab": anti_rms}

def ad_invariants(ad_list: list[np.ndarray]) -> dict:
    # for compact real ad matrices: tr(ad_a^T ad_a) should be constant across a
    vals = []
    for A in ad_list:
        vals.append(float(np.trace(A.T @ A)))
    vals = np.array(vals, dtype=float)
    return {
        "tr_adTad_mean": float(np.mean(vals)),
        "tr_adTad_min": float(np.min(vals)),
        "tr_adTad_max": float(np.max(vals)),
        "tr_adTad_std": float(np.std(vals)),
        "per_a": [float(x) for x in vals.tolist()],
    }


# -------------------------
# Alignment solve: find O minimizing sum_a ||A_a - O B_a O^T||_F^2
# -------------------------

def align_adjoint(A_list: list[np.ndarray], B_list: list[np.ndarray],
                  restarts: int, maxiter: int, seed: int):
    rng = np.random.default_rng(seed)
    n = A_list[0].shape[0]
    A = np.stack(A_list, axis=0)
    B = np.stack(B_list, axis=0)
    k = A.shape[0]

    def objective(p):
        K = skew_from_params(p, n)
        O = expm(K)
        OT = O.T
        s = 0.0
        for i in range(k):
            D = A[i] - (O @ B[i] @ OT)
            s += float(np.sum(D * D))
        return s

    best = {"val": float("inf"), "O": np.eye(n), "nit": None, "success": False, "p": None}

    starts = [np.zeros(params_dim(n), dtype=float)]
    for _ in range(max(0, restarts - 1)):
        starts.append(rng.normal(scale=0.25, size=params_dim(n)))

    for p0 in starts:
        res = minimize(objective, p0, method="L-BFGS-B", options={"maxiter": int(maxiter), "ftol": 1e-12})
        val = float(res.fun)
        if val < best["val"]:
            K = skew_from_params(res.x, n)
            O = expm(K)
            # clean numerical drift to nearest orthogonal
            U, _, Vt = np.linalg.svd(O, full_matrices=False)
            O = U @ Vt
            best = {"val": val, "O": O, "nit": int(res.nit), "success": bool(res.success), "p": res.x.tolist()}
    return best

def rotate_generators(T: list[np.ndarray], O: np.ndarray) -> list[np.ndarray]:
    n = len(T)
    out = []
    for a in range(n):
        M = np.zeros_like(T[0])
        for i in range(n):
            M += float(O[a, i]) * T[i]
        out.append(normalize_hs(traceless(hermitize(M))))
    out = gram_schmidt_hs(out, tol=1e-12)
    return out


# -------------------------
# Gauss test for H_gauge
# -------------------------

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

def gauss_commutators(H: np.ndarray, Q: list[np.ndarray], L: list[np.ndarray], R: list[np.ndarray]) -> dict:
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
        "gauss_left_median": float(np.median(left)),
        "gauss_left_mean": float(np.mean(left)),
        "gauss_right_max": float(np.max(right)),
        "gauss_right_median": float(np.median(right)),
        "gauss_right_mean": float(np.mean(right)),
        "per_generator_left": [float(x) for x in left.tolist()],
        "per_generator_right": [float(x) for x in right.tolist()],
    }


# -------------------------
# Variants (conjugate rep tests)
# -------------------------

def apply_variant(T: list[np.ndarray], variant: str) -> list[np.ndarray]:
    if variant == "orig":
        out = [X.copy() for X in T]
    elif variant == "neg":
        out = [(-X).copy() for X in T]
    elif variant == "transpose":
        out = [X.T.copy() for X in T]
    elif variant == "neg_transpose":
        out = [(-X.T).copy() for X in T]
    else:
        raise ValueError("unknown variant")
    out = [normalize_hs(traceless(hermitize(X))) for X in out]
    out = gram_schmidt_hs(out, tol=1e-12)
    return out


# -------------------------
# Main
# -------------------------

def main():
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

    variants = ["orig", "neg", "transpose", "neg_transpose"]
    combo_results = []

    # quick baseline (no variant changes)
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

            # rotate to site-index basis
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
    best = combo_results_sorted[0]

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
            "n_combos": len(combo_results_sorted),
            "best_combo": best,
            "top5": combo_results_sorted[:5],
        },
        "all_combos": combo_results_sorted,
        "interpretation_hint": (
            "If any combo passes_tol, the obstruction was a rep/basis convention (often neg_transpose on one end). "
            "If none pass_tol and align_obj_* stay O(1) not ~0, then the link su(3) has structure constants "
            "not orthogonally equivalent to the site basis -> representation mismatch / non-gauge embedding."
        ),
    }

    # Print concise
    print("============================================================")
    print("SU(3) STRUCTURE-CONSTANTS / GAUSS DIAGNOSTIC (v1)")
    print("------------------------------------------------------------")
    print(f"echo_model={args.echo_model}  tol={args.tol:.2e}  restarts={args.restarts}  maxiter={args.maxiter}")
    print("Baseline (unaligned) H_gauge:")
    print(f"  left_max={g0['gauss_left_max']:.6f}  right_max={g0['gauss_right_max']:.6f}")
    print("Best variant combo (after alignment):")
    print(f"  L={best['L_variant']}  R={best['R_variant']}")
    print(f"  align_obj_L={best['align_obj_L']:.6e}  align_obj_R={best['align_obj_R']:.6e}")
    print(f"  gauss_left_max={best['gauss_left_max']:.6f}  gauss_right_max={best['gauss_right_max']:.6f}")
    print(f"  passes_tol={best['passes_tol']}")
    print("Top 5 combos (minimize max(gauss_left_max, gauss_right_max)):")
    for i, r in enumerate(combo_results_sorted[:5], start=1):
        m = max(r["gauss_left_max"], r["gauss_right_max"])
        print(f"  #{i}: L={r['L_variant']:<13s} R={r['R_variant']:<13s}  "
              f"maxGauss={m:.6f}  (L={r['gauss_left_max']:.6f}, R={r['gauss_right_max']:.6f})")
    print("============================================================")

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"gauss_structure_constants_diagnostic_su3_v1_{out['meta']['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()