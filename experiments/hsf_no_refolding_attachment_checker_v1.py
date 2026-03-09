#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hsf_no_refolding_attachment_checker_v1.py
========================================

Goal
----
Given a candidate microscopic Hamiltonian (or set of terms) acting on a local neighborhood
(site-link-site, plaquette patch, etc.), classify each term as:

  (A) attachment-preserving  (no-refolding compatible)
      -> commutes with ALL left/right boundary generators on the link
         [h, L^a] = 0 and [h, R^a] = 0  (within the local support)

  (B) boundary-blending / refolding-like
      -> violates at least one commutator above

This operationalizes the proposed link between the HSF "no-refolding" constraint and
a superselection rule separating E_L and E_R sectors.

How it works
------------
1) Build a reference link-topology Hilbert space E in the boundary-inheritance form:

       E ≅ E_L ⊗ E_R ⊗ core
       dim(E_L)=d, dim(E_R)=d, dim(core)=core_dim

   and optionally scramble by a random unitary S so the basis is "topological".

2) Construct canonical commuting su(d) boundary generators on E:

       L^a = T^a ⊗ I ⊗ I
       R^a = I ⊗ (T^a)^T ⊗ I
   (then conjugated by S if scrambling is enabled)

3) Accept an input Hamiltonian specification in one of two modes:

   Mode 1 (JSON): Provide a JSON file that lists operator terms as Kronecker products
                  from named local operator dictionaries (site generators and link ops).

   Mode 2 (Import): You can point this script at a Python module that defines a function
                    build_terms() returning a list of numpy arrays (full matrices) for
                    the local neighborhood. This is the easiest way to integrate with your repo.

Outputs
-------
JSON report in ./hsf_out/hsf_no_refolding_attachment_checker_v1_<timestamp>.json

Run examples (Windows one-liners)
---------------------------------
A) Check built-in demo terms:
python hsf_no_refolding_attachment_checker_v1.py --demo --d 3 --core_dim 1 --scramble --seed 0

B) Check a custom Python module (must be importable):
python hsf_no_refolding_attachment_checker_v1.py --module my_hsf_terms_module --d 3 --core_dim 1 --scramble --seed 0

Module contract
---------------
Your module must define:

    def build_terms(d: int, core_dim: int, scramble: bool, seed: int) -> dict

returning a dict with:
    {
      "terms": [
         {"name": "...", "H": <np.ndarray>},  # matrix on local neighborhood Hilbert space
         ...
      ],
      "dims": {"dL": int, "dE": int, "dR": int},  # local factor dims; must include the link E in middle
      "note": "optional"
    }

The matrices must be sized (dL*dE*dR) x (dL*dE*dR) with ordering L ⊗ E ⊗ R.

Classification criteria
-----------------------
For each term h:

  compute max_a || [h, I_L ⊗ L^a ⊗ I_R] ||_HS
  compute max_a || [h, I_L ⊗ R^a ⊗ I_R] ||_HS

  If both maxima <= tol => attachment-preserving.

Default tol = 1e-10.

Dependencies
------------
numpy, scipy
"""

import os
import json
import math
import argparse
import importlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.linalg import expm


# -------------------------
# Utilities
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

def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(X)
    diag = np.diag(R)
    ph = diag / np.where(np.abs(diag) > 0, np.abs(diag), 1.0)
    Q = Q * ph
    return Q

def kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out


# -------------------------
# su(d) generators
# -------------------------

def su_generators(d: int):
    gens = []
    for i in range(d):
        for j in range(i + 1, d):
            S = np.zeros((d, d), dtype=complex)
            S[i, j] = 1.0
            S[j, i] = 1.0

            A = np.zeros((d, d), dtype=complex)
            A[i, j] = -1j
            A[j, i] = 1j

            gens.append(S)
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
# Build canonical boundary generators on E
# -------------------------

@dataclass
class BoundaryGenPack:
    L_ops: list[np.ndarray]   # on E
    R_ops: list[np.ndarray]   # on E
    scramble_U: np.ndarray    # S on E
    dims: dict                # {"d":..., "core_dim":..., "dE":...}


def build_boundary_generators(d: int, core_dim: int, scramble: bool, seed: int) -> BoundaryGenPack:
    """
    E = E_L ⊗ E_R ⊗ core with dims d, d, core_dim.
    L^a = T^a ⊗ I ⊗ I
    R^a = I ⊗ (T^a)^T ⊗ I
    then optionally conjugate by S.
    """
    rng = np.random.default_rng(seed)
    T = su_generators(d)
    I_d = np.eye(d, dtype=complex)
    I_c = np.eye(core_dim, dtype=complex)

    dE = d * d * core_dim

    L_ops = [kron(Ta, I_d, I_c) for Ta in T]
    R_ops = [kron(I_d, Ta.T, I_c) for Ta in T]

    if scramble:
        S = random_unitary(dE, rng)
        L_ops = [S @ O @ S.conj().T for O in L_ops]
        R_ops = [S @ O @ S.conj().T for O in R_ops]
    else:
        S = np.eye(dE, dtype=complex)

    L_ops = [normalize_hs(traceless(hermitize(O))) for O in L_ops]
    R_ops = [normalize_hs(traceless(hermitize(O))) for O in R_ops]

    return BoundaryGenPack(
        L_ops=L_ops,
        R_ops=R_ops,
        scramble_U=S,
        dims={"d": d, "core_dim": core_dim, "dE": dE}
    )


# -------------------------
# Term classification
# -------------------------

def classify_term(term_H: np.ndarray, dL: int, dE: int, dR: int,
                  L_ops_E: list[np.ndarray], R_ops_E: list[np.ndarray],
                  tol: float):
    """
    term_H: matrix on L⊗E⊗R.
    Build embedded boundary generators:
      G_L^a = I_L ⊗ L^a ⊗ I_R
      G_R^a = I_L ⊗ R^a ⊗ I_R

    Return commutator norms and pass/fail.
    """
    I_L = np.eye(dL, dtype=complex)
    I_R = np.eye(dR, dtype=complex)

    # compute max comm norm
    max_L = 0.0
    max_R = 0.0

    for O in L_ops_E:
        G = kron(I_L, O, I_R)
        n = hs_norm(comm(term_H, G))
        if n > max_L:
            max_L = n

    for O in R_ops_E:
        G = kron(I_L, O, I_R)
        n = hs_norm(comm(term_H, G))
        if n > max_R:
            max_R = n

    ok = (max_L <= tol) and (max_R <= tol)
    return {
        "max_comm_left": float(max_L),
        "max_comm_right": float(max_R),
        "tol": float(tol),
        "attachment_preserving": bool(ok),
    }


# -------------------------
# Demo term set
# -------------------------

def demo_terms(d: int, core_dim: int, scramble: bool, seed: int):
    """
    Build a few illustrative terms on L⊗E⊗R:
      - term1: boundary-preserving coupling (acts only on E core) => should pass
      - term2: coupling that acts only on E_L (commutes with R but not necessarily L) => should fail
      - term3: explicit boundary-mixing Hamiltonian on E (E_L ⊗ E_R coupling) => should fail

    NOTE: These are sanity checks of the checker.
    """
    rng = np.random.default_rng(seed + 999)
    T = su_generators(d)
    I_d = np.eye(d, dtype=complex)
    I_c = np.eye(core_dim, dtype=complex)

    dE = d * d * core_dim
    dL = d
    dR = d

    # E factorization (pre-scramble) for building terms
    if scramble:
        S = random_unitary(dE, rng)
    else:
        S = np.eye(dE, dtype=complex)

    def conjE(Oe):
        return S @ Oe @ S.conj().T

    # (1) Pure core operator (identity on E_L and E_R)
    # This should commute with both boundary algebras.
    # Build a random traceless Hermitian on core and lift.
    if core_dim > 1:
        X = rng.normal(size=(core_dim, core_dim)) + 1j * rng.normal(size=(core_dim, core_dim))
        Hc = traceless(hermitize(X))
        Hc = normalize_hs(Hc)
        H_E_core = kron(I_d, I_d, Hc)
    else:
        H_E_core = np.zeros((dE, dE), dtype=complex)

    term1 = kron(np.eye(dL, dtype=complex), conjE(H_E_core), np.eye(dR, dtype=complex))

    # (2) Operator on E_L only (will generally not commute with L^a)
    H_EL = kron(T[0], I_d, I_c)
    term2 = kron(np.eye(dL, dtype=complex), conjE(H_EL), np.eye(dR, dtype=complex))

    # (3) Mixing between E_L and E_R
    H_mix = kron(T[0], T[0], I_c)
    term3 = kron(np.eye(dL, dtype=complex), conjE(H_mix), np.eye(dR, dtype=complex))

    return {
        "dims": {"dL": dL, "dE": dE, "dR": dR},
        "terms": [
            {"name": "demo_core_only", "H": term1},
            {"name": "demo_E_left_only", "H": term2},
            {"name": "demo_E_mix", "H": term3},
        ],
        "note": "demo terms for attachment checker",
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--core_dim", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scramble", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--demo", action="store_true", help="run built-in demo term set")
    ap.add_argument("--module", type=str, default="", help="python module name providing build_terms()")
    args = ap.parse_args()

    # boundary generators on E
    pack = build_boundary_generators(d=args.d, core_dim=args.core_dim, scramble=args.scramble, seed=args.seed)

    # load terms
    if args.demo:
        blob = demo_terms(d=args.d, core_dim=args.core_dim, scramble=args.scramble, seed=args.seed)
    elif args.module:
        mod = importlib.import_module(args.module)
        if not hasattr(mod, "build_terms"):
            raise RuntimeError(f"Module '{args.module}' must define build_terms(d, core_dim, scramble, seed)")
        blob = mod.build_terms(d=args.d, core_dim=args.core_dim, scramble=args.scramble, seed=args.seed)
    else:
        raise RuntimeError("Provide --demo or --module <name>")

    dims = blob.get("dims", {})
    dL = int(dims.get("dL", args.d))
    dE = int(dims.get("dE", pack.dims["dE"]))
    dR = int(dims.get("dR", args.d))

    if dE != pack.dims["dE"]:
        raise RuntimeError(
            f"dE mismatch: term set dE={dE} but boundary generator pack expects dE={pack.dims['dE']} "
            f"(d={args.d}, core_dim={args.core_dim})."
        )

    report = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "d": args.d,
            "core_dim": args.core_dim,
            "seed": args.seed,
            "scramble": bool(args.scramble),
            "tol": float(args.tol),
            "source": "demo" if args.demo else f"module:{args.module}",
            "note": blob.get("note", ""),
        },
        "dims": {"dL": dL, "dE": dE, "dR": dR},
        "terms": []
    }

    # classify
    for item in blob["terms"]:
        name = item.get("name", "unnamed")
        H = item["H"]
        H = hermitize(np.array(H, dtype=complex))

        res = classify_term(
            term_H=H,
            dL=dL, dE=dE, dR=dR,
            L_ops_E=pack.L_ops,
            R_ops_E=pack.R_ops,
            tol=args.tol
        )
        report["terms"].append({"name": name, **res})

    # summary
    n_ok = sum(1 for t in report["terms"] if t["attachment_preserving"])
    n_tot = len(report["terms"])
    report["summary"] = {
        "n_terms": int(n_tot),
        "n_attachment_preserving": int(n_ok),
        "n_boundary_blending": int(n_tot - n_ok),
    }

    # print
    print("============================================================")
    print("NO-REFOLDING ATTACHMENT CHECKER")
    print("------------------------------------------------------------")
    print(f"d={args.d} core_dim={args.core_dim} dE={dE} scramble={args.scramble} tol={args.tol:.2e}")
    print(f"source: {report['meta']['source']}")
    print("------------------------------------------------------------")
    for t in report["terms"]:
        flag = "PASS" if t["attachment_preserving"] else "FAIL"
        print(f"{flag:4s}  {t['name']:<28s}  max_comm_left={t['max_comm_left']:.3e}  max_comm_right={t['max_comm_right']:.3e}")
    print("------------------------------------------------------------")
    print(f"Attachment-preserving: {n_ok}/{n_tot}")
    print("============================================================")

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"hsf_no_refolding_attachment_checker_v1_{report['meta']['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()