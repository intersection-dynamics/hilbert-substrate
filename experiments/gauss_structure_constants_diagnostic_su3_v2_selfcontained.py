#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_structure_constants_diagnostic_su3_v2_selfcontained.py
===========================================================

Self-contained diagnostic: generates its own Q, L, R data (no NPZ required).

Scenarios
---------
--scenario good
    Build a guaranteed Gauss-invariant situation using the same su(3) basis on
    site + link endpoints (up to safe per-generator cleanup).

--scenario bad_adjoint
    Build L/R by taking the correct basis but applying a random orthogonal
    rotation *in generator space* AND then a random permutation/sign flips.
    Algebra closes perfectly (Echo SU(3) style), but Gauss invariance fails
    unless alignment recovers the mapping.

--scenario bad_embed
    Build L/R as an su(3) basis in a *different 3D representation embedding*
    relative to site charges (still d=3 matrices, still closes su(3)), but
    constructed so that no adjoint-space orthogonal alignment can make the
    endpoint actions match the site Q^a pairing required for Gauss.
    This is the "representation mismatch" toy analogue of your obstruction.

What it does
------------
1) Constructs site generators Q^a (HS-orthonormal su(3) basis).
2) Constructs endpoint generators L^a, R^a based on scenario.
3) Computes structure constants f^{abc} for Q, L, R and adjoint matrices ad_a.
4) Finds best O∈SO(8) aligning ad(L) to ad(Q), and ad(R) to ad(Q).
5) Rotates generators by those O (without Gram-Schmidt).
6) Builds gauge-covariant H = Σ_a Q[a]⊗L[a]⊗I + I⊗R[a]⊗Q[a]
7) Reports ||[H, G_left^a]|| and ||[H, G_right^a]||.

Usage
-----
python gauss_structure_constants_diagnostic_su3_v2_selfcontained.py --scenario good
python gauss_structure_constants_diagnostic_su3_v2_selfcontained.py --scenario bad_adjoint
python gauss_structure_constants_diagnostic_su3_v2_selfcontained.py --scenario bad_embed
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

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


def kron3(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(A, B), C)


# -------------------------
# su(d) basis (Gell-Mann-like), HS-orthonormal
# -------------------------

def su_generators_gellmann(d: int) -> List[np.ndarray]:
    if d < 2:
        raise ValueError("d must be >= 2")

    gens: List[np.ndarray] = []

    # Symmetric off-diagonals
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = 1.0
            M[j, i] = 1.0
            gens.append(M)

    # Anti-symmetric off-diagonals
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = -1.0j
            M[j, i] = 1.0j
            gens.append(M)

    # Diagonals
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
        raise RuntimeError("Unexpected generator count.")
    return out


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
    return [np.array(f[a, :, :], dtype=float) for a in range(f.shape[0])]


def f_invariants(f: np.ndarray) -> Dict[str, float]:
    return {
        "f_fro": float(np.linalg.norm(f.ravel())),
        "f_maxabs": float(np.max(np.abs(f))),
    }


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
# SO(n) parameterization and adjoint alignment
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
        res = minimize(objective, p0, method="L-BFGS-B", options={"maxiter": int(maxiter), "ftol": 1e-12})
        val = float(res.fun)
        if val < best["val"]:
            K = skew_from_params(res.x, n)
            O = expm(K)
            # project to nearest orthogonal
            U, _, Vt = np.linalg.svd(O, full_matrices=False)
            O = U @ Vt
            best = {"val": val, "O": O, "nit": int(res.nit), "success": bool(res.success), "p": res.x.tolist()}
    return best


# -------------------------
# Generator-space rotation (NO Gram–Schmidt)
# -------------------------

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
        H += kron3(Q[a], L[a], I) + kron3(I, R[a], Q[a])
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
        G_left = kron3(Q[a], I, I) + kron3(I, L[a], I)
        G_right = kron3(I, I, Q[a]) - kron3(I, R[a], I)
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
# Scenario generators
# -------------------------

def random_SO(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal with det +1 via QR + det correction."""
    M = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(M)
    # make diagonal of R positive
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    # det correction to force +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def perm_sign_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation with random sign flips; orthogonal with det ±1 (we don't care here)."""
    P = np.eye(n)
    perm = rng.permutation(n)
    P = P[perm, :]
    signs = rng.choice([-1.0, 1.0], size=n)
    P = (signs[:, None]) * P
    return P


def conjugate_basis(T: List[np.ndarray], U: np.ndarray) -> List[np.ndarray]:
    """Conjugate each generator: T'_a = U T_a U†."""
    out = []
    for X in T:
        Y = U @ X @ U.conj().T
        out.append(normalize_hs(traceless(hermitize(Y))))
    return out


def scenario_good(Q: List[np.ndarray], rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Use same basis (optionally with tiny numeric noise).
    L = [X.copy() for X in Q]
    R = [X.copy() for X in Q]
    return L, R


def scenario_bad_adjoint(Q: List[np.ndarray], rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # L and R are correct su(3), but scrambled in generator space.
    OL = random_SO(len(Q), rng)
    OR = random_SO(len(Q), rng)
    PL = perm_sign_matrix(len(Q), rng)
    PR = perm_sign_matrix(len(Q), rng)
    L = rotate_generators(Q, PL @ OL)  # mix + scramble
    R = rotate_generators(Q, PR @ OR)
    return L, R


def scenario_bad_embed(Q: List[np.ndarray], rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Representation mismatch toy:
    - Start from Q (fundamental rep).
    - Conjugate by a *fixed* U for L, and a different U for R.
    - Then additionally scramble generator labels differently for L and R.

    Conjugation alone would still be representationally equivalent, but the key is:
    Gauss pairing needs a *shared* generator labeling across Q-L and Q-R contractions.
    We deliberately create L/R with incompatible labelings that can't be repaired by
    a single adjoint O that also preserves the needed endpoint pairing.

    In practice: this produces "Echo SU(3)" (closure) but persistent Gauss failure.
    """
    d = Q[0].shape[0]
    # Two unrelated SU(3) unitaries
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    B = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    # QR -> unitary
    U1, _ = np.linalg.qr(A)
    U2, _ = np.linalg.qr(B)
    # Normalize determinant to 1 (SU(3)-ish)
    U1 = U1 / np.linalg.det(U1) ** (1 / d)
    U2 = U2 / np.linalg.det(U2) ** (1 / d)

    L0 = conjugate_basis(Q, U1)
    R0 = conjugate_basis(Q, U2)

    # Now apply different nontrivial generator-space scrambles to L and R
    # that make the required a-index pairing incompatible across endpoints.
    OL = random_SO(len(Q), rng)
    OR = random_SO(len(Q), rng)
    PL = perm_sign_matrix(len(Q), rng)
    PR = perm_sign_matrix(len(Q), rng)

    L = rotate_generators(L0, PL @ OL)
    R = rotate_generators(R0, PR @ OR)
    return L, R


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, default="good", choices=["good", "bad_adjoint", "bad_embed"])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    Q = su_generators_gellmann(3)

    if args.scenario == "good":
        L_raw, R_raw = scenario_good(Q, rng)
    elif args.scenario == "bad_adjoint":
        L_raw, R_raw = scenario_bad_adjoint(Q, rng)
    elif args.scenario == "bad_embed":
        L_raw, R_raw = scenario_bad_embed(Q, rng)
    else:
        raise RuntimeError("unknown scenario")

    # Compute f + ad for each
    fQ = structure_constants(Q)
    fL = structure_constants(L_raw)
    fR = structure_constants(R_raw)

    adQ = adjoint_matrices(fQ)
    adL = adjoint_matrices(fL)
    adR = adjoint_matrices(fR)

    # Align adjoint reps
    alignL = align_adjoint(adQ, adL, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 101)
    alignR = align_adjoint(adQ, adR, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 202)

    OL = np.array(alignL["O"], dtype=float)
    OR = np.array(alignR["O"], dtype=float)

    # Rotate generators (keep index mapping; no Gram-Schmidt)
    L_al = rotate_generators(L_raw, OL)
    R_al = rotate_generators(R_raw, OR)

    # Gauge test
    H0 = build_H_gauge(Q, L_raw, R_raw)
    g0 = gauss_commutators(H0, Q, L_raw, R_raw)

    H = build_H_gauge(Q, L_al, R_al)
    g = gauss_commutators(H, Q, L_al, R_al)

    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "scenario": args.scenario,
            "seed": int(args.seed),
            "tol": float(args.tol),
            "restarts": int(args.restarts),
            "maxiter": int(args.maxiter),
        },
        "invariants": {
            "Q": {"f": f_invariants(fQ), "ad": ad_invariants(adQ)},
            "L_raw": {"f": f_invariants(fL), "ad": ad_invariants(adL)},
            "R_raw": {"f": f_invariants(fR), "ad": ad_invariants(adR)},
        },
        "alignment": {
            "L": {"obj": float(alignL["val"]), "success": bool(alignL["success"]), "nit": alignL["nit"]},
            "R": {"obj": float(alignR["val"]), "success": bool(alignR["success"]), "nit": alignR["nit"]},
        },
        "gauss": {
            "baseline_raw": g0,
            "after_adjoint_alignment": g,
            "passes_tol_after": bool((g["gauss_left_max"] <= args.tol) and (g["gauss_right_max"] <= args.tol)),
        },
        "notes": {
            "interpretation_hint": (
                "good: should pass (Gauss commutators ~ numerical noise). "
                "bad_adjoint: often fails raw but improves after alignment. "
                "bad_embed: closes su(3) but tends to remain O(1) after alignment (toy obstruction)."
            )
        }
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()