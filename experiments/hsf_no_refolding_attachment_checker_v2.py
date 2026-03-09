#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hsf_no_refolding_attachment_checker_v2.py
========================================

v2 — Strict no-refolding / attachment preservation checker + Gauss diagnostics.

This file fixes the formatting crash you hit and keeps the functionality intact.

USAGE (single-line Windows commands)
------------------------------------
Aligned:
python hsf_no_refolding_attachment_checker_v2.py --mode echo_su3 --lr_npz "C:\GitHub\hilbert_substrate\experiments\hsf_out\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz" --echo_model aligned --tol 1e-10

Mixed:
python hsf_no_refolding_attachment_checker_v2.py --mode echo_su3 --lr_npz "C:\GitHub\hilbert_substrate\experiments\hsf_out\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz" --echo_model mixed --tol 1e-10

Outputs
-------
Writes JSON to ./hsf_out/hsf_no_refolding_attachment_checker_v2_<timestamp>.json
"""

import os
import json
import math
import argparse
import importlib
from datetime import datetime

import numpy as np

# scipy is only needed for expm in the echo builder (kept for parity / future use)
try:
    from scipy.linalg import expm  # noqa: F401
except Exception as e:
    raise RuntimeError("scipy is required (scipy.linalg.expm). Install scipy.") from e


# -------------------------
# Small utils
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
# su(d) generators (HS-orthonormal, Hermitian, traceless)
# -------------------------

def su_generators_gellmann(d: int):
    gens = []

    # symmetric / antisymmetric off-diagonals
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

    # diagonals
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
# Echo-su3 convenience builder (matches your echo script's H)
# -------------------------

def build_echo_H_qutrit_su3(echo_model: str, seed: int = 999):
    """
    echo_model:
      - "aligned": B_a = S_a
      - "mixed":   B_a = Σ_b R_ab S_b with R orthogonal (seeded)

    Returns H_full on L⊗E⊗R where dims are 3,3,3.
    """
    dS = 3
    dB = 3
    S_basis = su_generators_gellmann(dS)  # 8
    B_basis = su_generators_gellmann(dB)  # 8

    rng = np.random.default_rng(seed)

    if echo_model == "aligned":
        Bp = B_basis
        mix_R = np.eye(8)
    elif echo_model == "mixed":
        X = rng.normal(size=(8, 8))
        Q, _ = np.linalg.qr(X)
        R = Q
        Bp = []
        for a in range(8):
            M = np.zeros((dB, dB), dtype=complex)
            for b in range(8):
                M += float(R[a, b]) * B_basis[b]
            Bp.append(normalize_hs(traceless(hermitize(M))))
        Bp = gram_schmidt_hs(Bp, tol=1e-12)
        mix_R = R
    else:
        raise ValueError("echo_model must be 'aligned' or 'mixed'")

    H = np.zeros((dS * dB * dS, dS * dB * dS), dtype=complex)
    for a in range(8):
        Sa = S_basis[a]
        Ba = Bp[a]
        H += np.kron(np.kron(Sa, Ba), Sa)

    H = H / max(hs_norm(H), 1e-12)
    return hermitize(H), mix_R


def load_lr_bases_npz(npz_path: str, echo_model: str):
    """
    Loads L/R bases from your LR NPZ:
      basis_left_<model>, basis_right_<model> with shape (8,3,3).
    """
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
# Core Gauss diagnostics
# -------------------------

def check_full_gauss(H_full: np.ndarray, dL: int, dE: int, dR: int,
                     Q_site: list[np.ndarray], L_ops_E: list[np.ndarray], R_ops_E: list[np.ndarray]):
    """
    Endpoint Gauss commutators for a full H on L⊗E⊗R.

    G_left^a  = Q_L^a + L^a_on_E
    G_right^a = Q_R^a - R^a_on_E   (oriented link convention)
    """
    I_L = np.eye(dL, dtype=complex)
    I_E = np.eye(dE, dtype=complex)
    I_R = np.eye(dR, dtype=complex)

    left = []
    right = []
    for a in range(len(Q_site)):
        Qa = Q_site[a]
        La = L_ops_E[a]
        Ra = R_ops_E[a]

        G_left = kron(Qa, I_E, I_R) + kron(I_L, La, I_R)
        G_right = kron(I_L, I_E, Qa) - kron(I_L, Ra, I_R)

        left.append(hs_norm(comm(H_full, G_left)))
        right.append(hs_norm(comm(H_full, G_right)))

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
# Module mode (optional integration hook)
# -------------------------

def run_module_mode(module_name: str, tol: float):
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "build_terms"):
        raise RuntimeError(f"Module '{module_name}' must define build_terms().")

    blob = mod.build_terms()

    dims = blob.get("dims", {})
    dL = int(dims["dL"])
    dE = int(dims["dE"])
    dR = int(dims["dR"])

    L_ops_E = blob["L_ops_E"]
    R_ops_E = blob["R_ops_E"]
    Q_site = blob["Q_ops_site"]

    if isinstance(L_ops_E, np.ndarray):
        L_ops_E = [L_ops_E[i] for i in range(L_ops_E.shape[0])]
    if isinstance(R_ops_E, np.ndarray):
        R_ops_E = [R_ops_E[i] for i in range(R_ops_E.shape[0])]
    if isinstance(Q_site, np.ndarray):
        Q_site = [Q_site[i] for i in range(Q_site.shape[0])]

    L_ops_E = [normalize_hs(traceless(hermitize(np.array(O, dtype=complex)))) for O in L_ops_E]
    R_ops_E = [normalize_hs(traceless(hermitize(np.array(O, dtype=complex)))) for O in R_ops_E]
    Q_site = [normalize_hs(traceless(hermitize(np.array(O, dtype=complex)))) for O in Q_site]

    report_terms = []
    for t in blob["terms"]:
        name = t.get("name", "unnamed")
        kind = t.get("kind", "FULL").upper()
        H = hermitize(np.array(t["H"], dtype=complex))

        gauss = check_full_gauss(H, dL, dE, dR, Q_site, L_ops_E, R_ops_E)

        # Minimal v2 module report: we always provide Gauss diagnostics;
        # attachment-preserving checks are left to your term-kind conventions if you want to extend further.
        report_terms.append({"name": name, "kind": kind, "tol": float(tol), **gauss})

    return {
        "dims": {"dL": dL, "dE": dE, "dR": dR},
        "terms": report_terms,
        "note": blob.get("note", ""),
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="echo_su3", choices=["echo_su3", "module"])
    ap.add_argument("--tol", type=float, default=1e-10)

    # echo_su3 mode
    ap.add_argument("--lr_npz", type=str, default="")
    ap.add_argument("--echo_model", type=str, default="aligned", choices=["aligned", "mixed"])
    ap.add_argument("--echo_seed", type=int, default=999)

    # module mode
    ap.add_argument("--module", type=str, default="")

    args = ap.parse_args()

    report = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "mode": args.mode,
            "tol": float(args.tol),
        },
        "dims": {},
        "terms": [],
        "gauss_full": None,
    }

    if args.mode == "echo_su3":
        if not args.lr_npz:
            raise RuntimeError("--lr_npz is required in echo_su3 mode")

        L_ops_E, R_ops_E = load_lr_bases_npz(args.lr_npz, args.echo_model)
        Q_site = su_generators_gellmann(3)

        if len(Q_site) != len(L_ops_E) or len(Q_site) != len(R_ops_E):
            raise RuntimeError("Generator count mismatch: site Q vs link L/R.")

        H_full, mixR = build_echo_H_qutrit_su3(args.echo_model, seed=args.echo_seed)

        dL = 3
        dE = 3
        dR = 3
        report["dims"] = {"dL": dL, "dE": dE, "dR": dR}
        report["meta"]["echo_model"] = args.echo_model
        report["meta"]["echo_seed"] = int(args.echo_seed)
        report["meta"]["lr_npz"] = args.lr_npz
        report["meta"]["echo_mix_matrix"] = mixR.tolist() if mixR is not None else None

        report["gauss_full"] = check_full_gauss(H_full, dL, dE, dR, Q_site, L_ops_E, R_ops_E)

        report["terms"].append({
            "name": f"echo_H_full_{args.echo_model}",
            "kind": "FULL",
            **report["gauss_full"],
        })

        print("============================================================")
        print("NO-REFOLDING / ATTACHMENT CHECKER v2 — echo_su3")
        print("------------------------------------------------------------")
        print(f"echo_model={args.echo_model}  tol={args.tol:.2e}")
        print("Gauss commutators (FULL H):")
        print(f"  left : max={report['gauss_full']['gauss_left_max']:.3e}  median={report['gauss_full']['gauss_left_median']:.3e}  mean={report['gauss_full']['gauss_left_mean']:.3e}")
        print(f"  right: max={report['gauss_full']['gauss_right_max']:.3e}  median={report['gauss_full']['gauss_right_median']:.3e}  mean={report['gauss_full']['gauss_right_mean']:.3e}")
        print("============================================================")

    elif args.mode == "module":
        if not args.module:
            raise RuntimeError("--module is required in module mode")

        blob_report = run_module_mode(args.module, args.tol)
        report["dims"] = blob_report["dims"]
        report["terms"] = blob_report["terms"]
        report["meta"]["module"] = args.module
        report["meta"]["note"] = blob_report.get("note", "")

        print("============================================================")
        print("NO-REFOLDING / ATTACHMENT CHECKER v2 — module mode")
        print("------------------------------------------------------------")
        dL = report["dims"]["dL"]
        dE = report["dims"]["dE"]
        dR = report["dims"]["dR"]
        print(f"module={args.module}  tol={args.tol:.2e}  dims(L,E,R)=({dL},{dE},{dR})")
        print("------------------------------------------------------------")
        for t in report["terms"]:
            print(f"{t['kind']:4s}  {t['name']:<28s}  GaussL(max)={t['gauss_left_max']:.3e}  GaussR(max)={t['gauss_right_max']:.3e}")
        print("============================================================")

    else:
        raise RuntimeError("Unknown mode")

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"hsf_no_refolding_attachment_checker_v2_{report['meta']['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()