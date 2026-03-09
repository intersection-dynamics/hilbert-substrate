#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_structure_constants_diagnostic_su3_v3_selfcontained_link9.py
==================================================================

Key fix vs v2.1:
- The link Hilbert space must support *commuting* left and right SU(3) actions.
  With link dimension 3 and L=R=Q on that same space, [L^a, R^b] != 0, so
  Gauss invariance fails even in the "good" case.

This version uses a minimal finite link register:
  link = C^3 ⊗ C^3  (dimension 9)
with
  L^a = Q^a ⊗ I
  R^a = I ⊗ Q^a
so [L^a, R^b] = 0 exactly.

Scenarios
---------
--scenario good
    Uses the commuting construction above: should PASS (commutators ~ numeric noise).

--scenario bad_adjoint
    Takes L/R and scrambles them in generator space independently; alignment should fix it.

--scenario bad_embed
    Conjugates the *link-side* reps by unrelated unitaries in each factor and then scrambles
    indices; this can preserve closure but can break the index pairing needed for Gauss.

Usage (Windows one-liner)
-------------------------
python gauss_structure_constants_diagnostic_su3_v3_selfcontained_link9.py --scenario good
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


def kron(*args: np.ndarray) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for A in args:
        out = np.kron(out, A)
    return out


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
            U, _, Vt = np.linalg.svd(O, full_matrices=False)
            O = U @ Vt
            best = {"val": val, "O": O, "nit": int(res.nit), "success": bool(res.success), "p": res.x.tolist()}
    return best


def rotate_generators(T: List[np.ndarray], O: np.ndarray) -> List[np.ndarray]:
    # NO Gram–Schmidt: preserve index meaning
    n = len(T)
    out: List[np.ndarray] = []
    for a in range(n):
        M = np.zeros_like(T[0])
        for i in range(n):
            M += float(O[a, i]) * T[i]
        out.append(normalize_hs(traceless(hermitize(M))))
    return out


# -------------------------
# Link construction (dimension 9) with commuting L/R
# -------------------------

def make_link_actions_from_site(Q_site: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    site reps: Q_site acts on C^3
    link reps: act on C^3 ⊗ C^3 (dim 9)
      L^a = Q^a ⊗ I
      R^a = I ⊗ Q^a
    => [L^a, R^b] = 0 exactly
    """
    d = Q_site[0].shape[0]
    I = np.eye(d, dtype=complex)
    L = [normalize_hs(traceless(hermitize(np.kron(Q_site[a], I)))) for a in range(len(Q_site))]
    R = [normalize_hs(traceless(hermitize(np.kron(I, Q_site[a])))) for a in range(len(Q_site))]
    return L, R


# -------------------------
# Gauge Hamiltonian + Gauss commutators
# -------------------------

def build_H_gauge(Qsite: List[np.ndarray], L: List[np.ndarray], R: List[np.ndarray]) -> np.ndarray:
    """
    Hilbert space ordering:
      (left matter) ⊗ (link) ⊗ (right matter)
    Dimensions:
      d_m = 3
      d_link = 9
      total = 3 * 9 * 3 = 81
    """
    d_m = int(Qsite[0].shape[0])
    d_link = int(L[0].shape[0])
    I_m = np.eye(d_m, dtype=complex)
    I_link = np.eye(d_link, dtype=complex)

    H = np.zeros((d_m * d_link * d_m, d_m * d_link * d_m), dtype=complex)
    for a in range(len(Qsite)):
        H += kron(Qsite[a], L[a], I_m) + kron(I_m, R[a], Qsite[a])
    H = hermitize(H)
    nrm = hs_norm(H)
    if nrm > 0:
        H = H / nrm
    return H


def gauss_commutators(H: np.ndarray, Qsite: List[np.ndarray], L: List[np.ndarray], R: List[np.ndarray],
                      sL: int, sR: int) -> Dict[str, float]:
    """
    Gauss generators:
      G_left^a  = Q_left^a  + sL * L^a
      G_right^a = Q_right^a + sR * R^a
    With commuting L/R on the link, the correct convention should give ~0 in "good".
    """
    d_m = int(Qsite[0].shape[0])
    d_link = int(L[0].shape[0])
    I_m = np.eye(d_m, dtype=complex)
    I_link = np.eye(d_link, dtype=complex)

    left_vals = []
    right_vals = []
    for a in range(len(Qsite)):
        G_left = kron(Qsite[a], I_link, I_m) + sL * kron(I_m, L[a], I_m)
        G_right = kron(I_m, I_link, Qsite[a]) + sR * kron(I_m, R[a], I_m)
        left_vals.append(hs_norm(comm(H, G_left)))
        right_vals.append(hs_norm(comm(H, G_right)))

    left = np.array(left_vals, dtype=float)
    right = np.array(right_vals, dtype=float)
    return {
        "sL": int(sL),
        "sR": int(sR),
        "gauss_left_max": float(np.max(left)),
        "gauss_right_max": float(np.max(right)),
        "score_maxmax": float(max(np.max(left), np.max(right))),
        "gauss_left_vec": left.tolist(),
        "gauss_right_vec": right.tolist(),
    }


def gauss_best(H: np.ndarray, Qsite: List[np.ndarray], L: List[np.ndarray], R: List[np.ndarray]) -> Dict:
    table = []
    for sL in (+1, -1):
        for sR in (+1, -1):
            table.append(gauss_commutators(H, Qsite, L, R, sL=sL, sR=sR))
    table = sorted(table, key=lambda x: x["score_maxmax"])
    return {"best": table[0], "all": table}


# -------------------------
# Scenarios
# -------------------------

def random_SO(n: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(n, n))
    Qm, Rm = np.linalg.qr(M)
    s = np.sign(np.diag(Rm))
    s[s == 0] = 1.0
    Qm = Qm * s
    if np.linalg.det(Qm) < 0:
        Qm[:, 0] *= -1.0
    return Qm


def perm_sign_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    P = np.eye(n)
    perm = rng.permutation(n)
    P = P[perm, :]
    signs = rng.choice([-1.0, 1.0], size=n)
    return (signs[:, None]) * P


def conjugate_on_factor_link(gen_list: List[np.ndarray], d: int, factor: int, U: np.ndarray) -> List[np.ndarray]:
    """
    Conjugate generators on link space C^d ⊗ C^d by acting U on one factor.
    factor=0 -> U ⊗ I
    factor=1 -> I ⊗ U
    """
    I = np.eye(d, dtype=complex)
    if factor == 0:
        W = np.kron(U, I)
    elif factor == 1:
        W = np.kron(I, U)
    else:
        raise ValueError("factor must be 0 or 1")
    out = []
    for X in gen_list:
        Y = W @ X @ W.conj().T
        out.append(normalize_hs(traceless(hermitize(Y))))
    return out


def scenario_good(Qsite: List[np.ndarray], rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    return make_link_actions_from_site(Qsite)


def scenario_bad_adjoint(Qsite: List[np.ndarray], rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    L, R = make_link_actions_from_site(Qsite)
    OL = random_SO(len(Qsite), rng)
    OR = random_SO(len(Qsite), rng)
    PL = perm_sign_matrix(len(Qsite), rng)
    PR = perm_sign_matrix(len(Qsite), rng)
    return rotate_generators(L, PL @ OL), rotate_generators(R, PR @ OR)


def scenario_bad_embed(Qsite: List[np.ndarray], rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    d = Qsite[0].shape[0]
    L, R = make_link_actions_from_site(Qsite)

    # random unitaries on each factor
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    B = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    U1, _ = np.linalg.qr(A)
    U2, _ = np.linalg.qr(B)
    U1 = U1 / (np.linalg.det(U1) ** (1 / d))
    U2 = U2 / (np.linalg.det(U2) ** (1 / d))

    # Conjugate L on factor-0, R on factor-1 (keeps [L,R]=0 but changes embeddings)
    Lc = conjugate_on_factor_link(L, d, factor=0, U=U1)
    Rc = conjugate_on_factor_link(R, d, factor=1, U=U2)

    # then scramble indices differently
    OL = random_SO(len(Qsite), rng)
    OR = random_SO(len(Qsite), rng)
    PL = perm_sign_matrix(len(Qsite), rng)
    PR = perm_sign_matrix(len(Qsite), rng)
    return rotate_generators(Lc, PL @ OL), rotate_generators(Rc, PR @ OR)


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

    Qsite = su_generators_gellmann(3)

    if args.scenario == "good":
        L_raw, R_raw = scenario_good(Qsite, rng)
    elif args.scenario == "bad_adjoint":
        L_raw, R_raw = scenario_bad_adjoint(Qsite, rng)
    else:
        L_raw, R_raw = scenario_bad_embed(Qsite, rng)

    # invariants (note L/R are 9x9 now)
    fQ = structure_constants(Qsite)
    fL = structure_constants(L_raw)
    fR = structure_constants(R_raw)
    adQ = adjoint_matrices(fQ)
    adL = adjoint_matrices(fL)
    adR = adjoint_matrices(fR)

    # align in adjoint space
    alignL = align_adjoint(adQ, adL, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 101)
    alignR = align_adjoint(adQ, adR, restarts=args.restarts, maxiter=args.maxiter, seed=args.seed + 202)

    OL = np.array(alignL["O"], dtype=float)
    OR = np.array(alignR["O"], dtype=float)

    L_al = rotate_generators(L_raw, OL)
    R_al = rotate_generators(R_raw, OR)

    # build H and test Gauss for all sign conventions
    H_raw = build_H_gauge(Qsite, L_raw, R_raw)
    H_al = build_H_gauge(Qsite, L_al, R_al)

    g_raw = gauss_best(H_raw, Qsite, L_raw, R_raw)
    g_al = gauss_best(H_al, Qsite, L_al, R_al)

    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "scenario": args.scenario,
            "seed": int(args.seed),
            "tol": float(args.tol),
            "restarts": int(args.restarts),
            "maxiter": int(args.maxiter),
            "dims": {"matter": 3, "link": int(L_raw[0].shape[0])},
        },
        "invariants": {
            "Qsite": {"f": f_invariants(fQ), "ad": ad_invariants(adQ)},
            "L_raw": {"f": f_invariants(fL), "ad": ad_invariants(adL)},
            "R_raw": {"f": f_invariants(fR), "ad": ad_invariants(adR)},
        },
        "alignment": {
            "L": {"obj": float(alignL["val"]), "success": bool(alignL["success"]), "nit": alignL["nit"]},
            "R": {"obj": float(alignR["val"]), "success": bool(alignR["success"]), "nit": alignR["nit"]},
        },
        "gauss": {
            "raw_best": g_raw["best"],
            "aligned_best": g_al["best"],
            "passes_tol_aligned": bool(g_al["best"]["score_maxmax"] <= args.tol),
            "tested_raw": g_raw["all"],
            "tested_aligned": g_al["all"],
        },
        "notes": {
            "core_point": "Link uses dim=9 so left/right actions commute. If your real extracted link lives in dim=3, that alone can obstruct Gauss invariance."
        }
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()