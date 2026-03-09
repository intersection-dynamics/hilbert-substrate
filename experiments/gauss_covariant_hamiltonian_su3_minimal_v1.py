#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_covariant_hamiltonian_su3_minimal_v1.py
============================================

Minimal test: build the *explicit* Gauss-covariant local Hamiltonian for an oriented
link x -> y using your extracted su(3) link endpoint bases (L^a, R^a), and verify:

  [H, G_left^a]  ~ 0   where G_left^a  = Q_x^a + L^a
  [H, G_right^a] ~ 0   where G_right^a = Q_y^a - R^a

Construction
------------
Given:
  - Q_x^a, Q_y^a: su(3) generators on the qutrit sites (HS-orthonormal)
  - L^a, R^a: extracted su(3) endpoint generators on the bond register (qutrit)

Define the gauge-covariant coupling (local "matter-link" coupling):
  H = sum_a [  Q_x^a ⊗ L^a ⊗ I_y  +  I_x ⊗ R^a ⊗ Q_y^a  ]

Optionally also test the (wrong) symmetric echo ansatz for comparison:
  H_echo = sum_a Q_x^a ⊗ B^a ⊗ Q_y^a   (with B^a chosen from bond basis)

Usage (single-line Windows)
---------------------------
Aligned:
python gauss_covariant_hamiltonian_su3_minimal_v1.py --lr_npz "C:\GitHub\hilbert_substrate\experiments\hsf_out\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz" --echo_model aligned --tol 1e-10

Mixed:
python gauss_covariant_hamiltonian_su3_minimal_v1.py --lr_npz "C:\GitHub\hilbert_substrate\experiments\hsf_out\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz" --echo_model mixed --tol 1e-10

Outputs
-------
Writes JSON to ./hsf_out/gauss_covariant_hamiltonian_su3_minimal_v1_<timestamp>.json
"""

import os
import json
import math
import argparse
from datetime import datetime

import numpy as np


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
    for A in ops[1:]:
        out = np.kron(out, A)
    return out


# -------------------------
# su(3) basis on qutrit site (HS-orthonormal Gell-Mann-like)
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
# Gauss commutator diagnostics
# -------------------------

def gauss_commutators(H: np.ndarray, Q: list[np.ndarray], L: list[np.ndarray], R: list[np.ndarray]):
    """
    For dims 3 ⊗ 3 ⊗ 3 in ordering x ⊗ link ⊗ y.

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


# -------------------------
# Build Hamiltonians
# -------------------------

def build_H_gauge(Q: list[np.ndarray], L: list[np.ndarray], R: list[np.ndarray], lam: float = 1.0):
    """
    H_gauge = Σ_a [ Q_x^a ⊗ L^a ⊗ I  +  I ⊗ R^a ⊗ Q_y^a ]
    """
    d = 3
    I = np.eye(d, dtype=complex)

    H = np.zeros((d*d*d, d*d*d), dtype=complex)
    for a in range(len(Q)):
        H += kron(Q[a], L[a], I) + kron(I, R[a], Q[a])

    H = hermitize(H)
    # scale for comparability; doesn't affect commutator being zero
    n = hs_norm(H)
    if n > 0:
        H = (lam / n) * H
    return H

def build_H_echo_like(Q: list[np.ndarray], B: list[np.ndarray]):
    """
    For comparison only:
      H_echo = Σ_a Q_x^a ⊗ B^a ⊗ Q_y^a
    """
    d = 3
    H = np.zeros((d*d*d, d*d*d), dtype=complex)
    for a in range(len(Q)):
        H += kron(Q[a], B[a], Q[a])
    H = hermitize(H)
    n = hs_norm(H)
    if n > 0:
        H = H / n
    return H


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_npz", type=str, required=True)
    ap.add_argument("--echo_model", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--no_compare_echo", action="store_true", help="skip echo-like comparison Hamiltonian")
    args = ap.parse_args()

    Q = su_generators_gellmann(3)   # site generators (8)
    L, R = load_lr_bases_npz(args.lr_npz, args.echo_model)

    if len(L) != 8 or len(R) != 8 or len(Q) != 8:
        raise RuntimeError(f"Expected 8 generators; got Q={len(Q)} L={len(L)} R={len(R)}")

    H_gauge = build_H_gauge(Q, L, R, lam=1.0)
    diag_gauge = gauss_commutators(H_gauge, Q, L, R)

    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "lr_npz": args.lr_npz,
            "echo_model": args.echo_model,
            "tol": float(args.tol),
        },
        "gauge_covariant": {
            "diagnostics": diag_gauge,
            "passes_tol": bool((diag_gauge["gauss_left_max"] <= args.tol) and (diag_gauge["gauss_right_max"] <= args.tol)),
        },
        "echo_like": None
    }

    if not args.no_compare_echo:
        # Use bond basis as B^a. We'll compare both choices:
        #  - B = L (left basis)
        #  - B = R (right basis)
        H_echo_L = build_H_echo_like(Q, L)
        H_echo_R = build_H_echo_like(Q, R)
        diag_echo_L = gauss_commutators(H_echo_L, Q, L, R)
        diag_echo_R = gauss_commutators(H_echo_R, Q, L, R)

        out["echo_like"] = {
            "using_B_equals_L": {"diagnostics": diag_echo_L},
            "using_B_equals_R": {"diagnostics": diag_echo_R},
        }

    # Print
    print("============================================================")
    print("GAUSS-COVARIANT HAMILTONIAN MINIMAL TEST (SU3)")
    print("------------------------------------------------------------")
    print(f"echo_model={args.echo_model}  tol={args.tol:.2e}")
    print("H_gauge:")
    print(f"  left : max={diag_gauge['gauss_left_max']:.3e}  median={diag_gauge['gauss_left_median']:.3e}  mean={diag_gauge['gauss_left_mean']:.3e}")
    print(f"  right: max={diag_gauge['gauss_right_max']:.3e}  median={diag_gauge['gauss_right_median']:.3e}  mean={diag_gauge['gauss_right_mean']:.3e}")
    print(f"  passes_tol={out['gauge_covariant']['passes_tol']}")
    if out["echo_like"] is not None:
        deL = out["echo_like"]["using_B_equals_L"]["diagnostics"]
        deR = out["echo_like"]["using_B_equals_R"]["diagnostics"]
        print("------------------------------------------------------------")
        print("Echo-like comparison:")
        print(f"  B=L: left_max={deL['gauss_left_max']:.3e}  right_max={deL['gauss_right_max']:.3e}")
        print(f"  B=R: left_max={deR['gauss_left_max']:.3e}  right_max={deR['gauss_right_max']:.3e}")
    print("============================================================")

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"gauss_covariant_hamiltonian_su3_minimal_v1_{out['meta']['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()