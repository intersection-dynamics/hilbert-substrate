#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, os, time
from typing import List, Dict, Tuple
import numpy as np

from scipy.linalg import expm


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


def su_generators_gellmann(d: int) -> List[np.ndarray]:
    gens: List[np.ndarray] = []
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = 1.0
            M[j, i] = 1.0
            gens.append(M)
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = -1.0j
            M[j, i] = 1.0j
            gens.append(M)
    for k in range(1, d):
        M = np.zeros((d, d), dtype=complex)
        for i in range(k):
            M[i, i] = 1.0
        M[k, k] = -float(k)
        gens.append(M)
    out = [normalize_hs(traceless(hermitize(G))) for G in gens]
    if len(out) != d * d - 1:
        raise RuntimeError("bad generator count")
    return out


def make_link_actions(Qsite: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    d = Qsite[0].shape[0]
    I = np.eye(d, dtype=complex)
    # IMPORTANT: do NOT HS-normalize here; keep the same scale as Qsite so covariance matches.
    L = [np.kron(Qsite[a], I) for a in range(len(Qsite))]
    R = [np.kron(I, Qsite[a]) for a in range(len(Qsite))]
    # tiny cleanup only
    L = [traceless(hermitize(X)) for X in L]
    R = [traceless(hermitize(X)) for X in R]
    return L, R


def random_su_from_basis(T: List[np.ndarray], rng: np.random.Generator, scale: float = 0.5) -> np.ndarray:
    theta = rng.normal(size=len(T)) * scale
    A = np.zeros_like(T[0])
    for a in range(len(T)):
        A += theta[a] * T[a]
    return traceless(hermitize(A))


def build_link_transporter(Qsite: List[np.ndarray], rng: np.random.Generator, scale: float = 0.5) -> np.ndarray:
    """
    Build a fundamental SU(3) element U_fund = exp(i * sum theta_a Q^a).
    """
    A = random_su_from_basis(Qsite, rng, scale=scale)
    U = expm(1.0j * A)
    return U


def build_hopping_H(Qsite: List[np.ndarray], L: List[np.ndarray], R: List[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    Hilbert space:
      (left matter d=3) ⊗ (link d_link=9) ⊗ (right matter d=3)

    We create a gauge-covariant hopping term using a transporter U_fund that acts
    between right/left matter indices, while link transforms with L/R.
    Minimal construction:
      H = sum_{i,j} ( |i><j|_L ⊗ U_link(i,j) ⊗ |j><i|_R ) + h.c.

    Where U_link(i,j) is implemented on the link as (U_fund ⊗ I) acting on factor-0.
    """
    d = Qsite[0].shape[0]
    I = np.eye(d, dtype=complex)
    d_link = L[0].shape[0]
    I_link = np.eye(d_link, dtype=complex)

    U_fund = build_link_transporter(Qsite, rng, scale=0.8)

    # Embed U on the link so it transforms by left action on factor-0 and right action on factor-1.
    # For this test we use U_link = U_fund ⊗ U_fund^†, which is the cleanest "bi-fundamental".
    U_link = np.kron(U_fund, U_fund.conj().T)

    # Build hopping operator
    H = np.zeros((d * d_link * d, d * d_link * d), dtype=complex)

    for i in range(d):
        for j in range(d):
            E_L = np.zeros((d, d), dtype=complex)
            E_L[i, j] = 1.0
            E_R = np.zeros((d, d), dtype=complex)
            E_R[j, i] = 1.0  # opposite to make bilinear invariant
            H += kron(E_L, U_link, E_R)

    H = hermitize(H)
    H /= max(1e-15, hs_norm(H))
    return H


def gauss_best(H: np.ndarray, Qsite: List[np.ndarray], L: List[np.ndarray], R: List[np.ndarray]) -> Dict:
    d = Qsite[0].shape[0]
    d_link = L[0].shape[0]
    I_m = np.eye(d, dtype=complex)
    I_link = np.eye(d_link, dtype=complex)

    def one(sL: int, sR: int) -> Dict:
        left = []
        right = []
        for a in range(len(Qsite)):
            G_left = kron(Qsite[a], I_link, I_m) + sL * kron(I_m, L[a], I_m)
            G_right = kron(I_m, I_link, Qsite[a]) + sR * kron(I_m, R[a], I_m)
            left.append(hs_norm(comm(H, G_left)))
            right.append(hs_norm(comm(H, G_right)))
        left = np.array(left)
        right = np.array(right)
        return {
            "sL": int(sL), "sR": int(sR),
            "left_max": float(left.max()), "right_max": float(right.max()),
            "score": float(max(left.max(), right.max()))
        }

    allc = [one(sL, sR) for sL in (+1, -1) for sR in (+1, -1)]
    allc = sorted(allc, key=lambda x: x["score"])
    return {"best": allc[0], "all": allc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-10)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    Qsite = su_generators_gellmann(3)
    L, R = make_link_actions(Qsite)

    H = build_hopping_H(Qsite, L, R, rng)
    g = gauss_best(H, Qsite, L, R)

    out = {
        "meta": {"script": os.path.basename(__file__), "timestamp": now_tag(), "seed": args.seed, "tol": args.tol},
        "dims": {"matter": 3, "link": int(L[0].shape[0])},
        "gauss": g,
        "passes": bool(g["best"]["score"] <= args.tol),
        "note": "This tests a gauge-covariant hopping-like term instead of Q·L + R·Q."
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()