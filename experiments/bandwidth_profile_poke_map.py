#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bandwidth_profile_poke_map.py
=============================

A single self-contained script that:

(a) does a poke sweep on site A (uses su(d_site) Hermitian generators),
(b) constructs the poke→endpoint influence map (responses on link endpoint observables),
(c) prints a one-line "bandwidth profile" = top singular values of that map.

No "bits/sec" language. Bandwidth here means: how many independent poke directions on A
produce independent, significant response directions on the link endpoint interface.

Model
-----
System Hilbert space:
  H = H_A(d_site) ⊗ H_link(d_link) ⊗ H_B(d_site)

We choose a baseline product state |psiA>⊗|psiL>⊗|psiB>.
For each poke generator P_a on site A:
  - apply a small unitary U_poke = exp(-i * eta * P_a) to site A
  - evolve under system coupling U = exp(-i * dt * H_coupling)
  - measure the change in expectations of link endpoint observables {O_j}:
      r_j(a) = <O_j>_poked_after - <O_j>_baseline_after
Stack r(:,a) into response matrix R (n_obs x n_pokes), compute singular values s.

Outputs
-------
- Prints top singular values in one line (the "bandwidth profile")
- Writes JSON with details into ./hsf_out/bandwidth/

Usage examples
--------------
# Composite link (9=3⊗3), gauge-like two-leg coupling with singlet-admitting pairing
python bandwidth_profile_poke_map.py --d_site 3 --d_link 9 --model two_leg_conjugate

# Simple link (3), two-leg style isn't available; use single_leg baseline
python bandwidth_profile_poke_map.py --d_site 3 --d_link 3 --model single_leg

Notes
-----
- If d_link is a perfect square N^2, we define link endpoint observables as:
    O_j = T^j ⊗ I_N   (left endpoint on V factor), using su(N) generators.
  This matches "bidirectional link = V⊗Vbar" algebraically.
- If d_link is not a perfect square, we fall back to using su(d_link) generators
  but only the first (d_site^2-1) of them as endpoint observables.

Dependencies: numpy, scipy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np

try:
    from scipy.linalg import expm
except Exception as e:
    raise RuntimeError("This script requires scipy (scipy.linalg.expm).") from e


# ---------------------------
# Basic helpers
# ---------------------------

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def hermitize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)

def traceless(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d, dtype=complex)

def hs_inner(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.real(np.trace(A.conj().T @ B)))

def hs_norm(A: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, hs_inner(A, A))))

def normalize_hs(A: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = hs_norm(A)
    return A.copy() if n < eps else (A / n)

def kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for X in ops[1:]:
        out = np.kron(out, X)
    return out

def haar_state(d: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=d) + 1j * rng.normal(size=d)
    n = np.linalg.norm(z)
    if n < 1e-30:
        z[0] = 1.0
        n = 1.0
    return z / n

def perfect_square_root(d: int) -> int | None:
    r = int(round(math.sqrt(d)))
    return r if r * r == d else None


# ---------------------------
# su(d) generators (HS-orthonormal Hermitian traceless)
# ---------------------------

def su_generators(d: int) -> List[np.ndarray]:
    gens = []

    # symmetric and antisymmetric off-diagonal
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

    # diagonal traceless
    for k in range(1, d):
        D = np.zeros((d, d), dtype=complex)
        for i in range(k):
            D[i, i] = 1.0
        D[k, k] = -float(k)
        D *= math.sqrt(2.0 / (k * (k + 1.0)))
        gens.append(D)

    out = []
    for G in gens:
        H = hermitize(traceless(G))
        out.append(normalize_hs(H))
    return out

def anti_fund(Ta: np.ndarray) -> np.ndarray:
    # Anti-fundamental rep for Hermitian basis
    return -(Ta.T)


# ---------------------------
# Build coupling Hamiltonians
# ---------------------------

def build_coupling_H(d_site: int, d_link: int, model: str) -> Tuple[np.ndarray, Dict]:
    """
    Returns H on (siteA ⊗ link ⊗ siteB).

    model options:
      - single_leg:
          H = Σ_a  T_A^a ⊗ L^a ⊗ T_B^a
        where L^a are embedded site generators into link (top-left d_site block), then conjugate by identity.
        (Works for any d_link >= d_site.)

      - two_leg_conjugate:  (best match to your composite-link gauge construction)
          Requires d_link = N^2 and uses link = N⊗N with:
            left endpoint uses fund:   T ⊗ I
            right endpoint uses anti:  I ⊗ (-(T^T))
          And chooses singlet-admitting pairing:
            siteA anti-fund, siteB fund
          H = Σ_a [ T_Abar^a ⊗ (T^a⊗I) ⊗ I  +  I ⊗ (I⊗Tbar^a) ⊗ T_B^a ]

      - two_leg_fund:
          Like two_leg_conjugate but uses fund on both link factors and fund on both sites
          (often gauge-invariant but Gauss kernel may be empty).
    """
    N = perfect_square_root(d_link)
    T_site = su_generators(d_site)
    k = len(T_site)
    I_site = np.eye(d_site, dtype=complex)
    I_link = np.eye(d_link, dtype=complex)

    if model == "single_leg":
        # Embed su(d_site) into link as top-left block
        L_ops = []
        for a in range(k):
            M = np.zeros((d_link, d_link), dtype=complex)
            M[:d_site, :d_site] = T_site[a]
            L_ops.append(hermitize(traceless(M)))
        H = np.zeros((d_site * d_link * d_site, d_site * d_link * d_site), dtype=complex)
        for a in range(k):
            H += kron(T_site[a], L_ops[a], T_site[a])
        H = hermitize(H)
        meta = {"model": model, "d_site": d_site, "d_link": d_link, "N": None}
        return H, meta

    if model in ("two_leg_conjugate", "two_leg_fund"):
        if N is None:
            raise ValueError(f"{model} requires d_link to be a perfect square (N^2). Got d_link={d_link}.")
        if d_site != N:
            # We keep it strict: the site rep dimension matches the link factor dimension
            raise ValueError(f"{model} requires d_site == N where d_link=N^2. Got d_site={d_site}, N={N}.")

        I_N = np.eye(N, dtype=complex)
        T_f = su_generators(N)
        T_af = [anti_fund(X) for X in T_f]

        # Link endpoint operators on d_link = N⊗N
        link_left_f = [kron(T_f[a], I_N) for a in range(k)]
        link_right_f = [kron(I_N, T_f[a]) for a in range(k)]
        link_right_af = [kron(I_N, T_af[a]) for a in range(k)]

        # Site reps
        if model == "two_leg_conjugate":
            T_A = T_af  # siteA anti
            T_B = T_f   # siteB fund
            link_left = link_left_f
            link_right = link_right_af
        else:
            T_A = T_f
            T_B = T_f
            link_left = link_left_f
            link_right = link_right_f

        D = d_site * d_link * d_site
        H = np.zeros((D, D), dtype=complex)
        for a in range(k):
            H += kron(T_A[a], link_left[a], I_site)
            H += kron(I_site, link_right[a], T_B[a])
        H = hermitize(H)
        meta = {"model": model, "d_site": d_site, "d_link": d_link, "N": N}
        return H, meta

    raise ValueError(f"Unknown model: {model}")


# ---------------------------
# Define link endpoint observables (what we measure on the link)
# ---------------------------

def link_endpoint_observables(d_site: int, d_link: int) -> Tuple[List[np.ndarray], Dict]:
    """
    Returns a list of endpoint observables O_j on the link to measure response.
    Default: left endpoint generators.

    If d_link = N^2:
      O_j = T^j ⊗ I_N  with T^j su(N) and N=d_site preferred.

    Else:
      use first (d_site^2-1) su(d_link) generators as a generic measurement set.
    """
    N = perfect_square_root(d_link)
    if N is not None:
        I_N = np.eye(N, dtype=complex)
        T_N = su_generators(N)
        O = [kron(T_N[j], I_N) for j in range(len(T_N))]
        meta = {"type": "factor_left", "N": N, "count": len(O)}
        return O, meta

    # fallback
    T_L = su_generators(d_link)
    count = min(len(T_L), d_site * d_site - 1)
    O = T_L[:count]
    meta = {"type": "fallback_su(d_link)", "N": None, "count": len(O)}
    return O, meta


# ---------------------------
# Expectation / evolution
# ---------------------------

def expval_link(O_link: np.ndarray, psi_full: np.ndarray, d_site: int, d_link: int) -> float:
    """
    Compute <psi| I ⊗ O_link ⊗ I |psi>
    """
    I_site = np.eye(d_site, dtype=complex)
    Op = kron(I_site, O_link, I_site)
    return float(np.real(psi_full.conj().T @ (Op @ psi_full)))

def apply_unitary_on_siteA(U_A: np.ndarray, psi_full: np.ndarray, d_site: int, d_link: int) -> np.ndarray:
    I_link = np.eye(d_link, dtype=complex)
    I_site = np.eye(d_site, dtype=complex)
    U_full = kron(U_A, I_link, I_site)
    return U_full @ psi_full

def evolve(U: np.ndarray, psi_full: np.ndarray) -> np.ndarray:
    return U @ psi_full


# ---------------------------
# Main bandwidth profile computation
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_site", type=int, default=3)
    ap.add_argument("--d_link", type=int, default=9)
    ap.add_argument("--model", type=str, default="two_leg_conjugate",
                    choices=["single_leg", "two_leg_conjugate", "two_leg_fund"])
    ap.add_argument("--dt", type=float, default=0.10, help="Coupling evolution step size")
    ap.add_argument("--eta", type=float, default=0.10, help="Poke strength on site A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=8, help="How many singular values to print")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Build coupling and evolution operator
    H, metaH = build_coupling_H(args.d_site, args.d_link, args.model)
    U = expm(-1j * args.dt * H)

    # Choose baseline product state
    psiA = haar_state(args.d_site, rng)
    psiL = haar_state(args.d_link, rng)
    psiB = haar_state(args.d_site, rng)
    psi0 = np.kron(np.kron(psiA, psiL), psiB)

    # Choose poke generators on site A
    pokes = su_generators(args.d_site)  # k_pokes = d_site^2 - 1
    k_pokes = len(pokes)

    # Choose endpoint observables on link
    obs, metaO = link_endpoint_observables(args.d_site, args.d_link)
    n_obs = len(obs)

    # Baseline after coupling evolution
    psi_base = evolve(U, psi0)

    # Measure baseline endpoint expectations
    base_vals = np.array([expval_link(O, psi_base, args.d_site, args.d_link) for O in obs], dtype=float)

    # Response matrix R: shape (n_obs, k_pokes)
    R = np.zeros((n_obs, k_pokes), dtype=float)

    for a, P in enumerate(pokes):
        U_poke = expm(-1j * args.eta * P)          # poke on site A
        psi_p = apply_unitary_on_siteA(U_poke, psi0, args.d_site, args.d_link)
        psi_p_after = evolve(U, psi_p)

        vals = np.array([expval_link(O, psi_p_after, args.d_site, args.d_link) for O in obs], dtype=float)
        R[:, a] = (vals - base_vals)

    # Singular values (bandwidth profile)
    # We only care about relative profile; SVD on real response matrix
    _, s, _ = np.linalg.svd(R, full_matrices=False)

    topk = min(args.topk, s.size)
    profile = s[:topk].tolist()

    # One-line output
    tag = now_tag()
    print(f"[BANDWIDTH PROFILE] model={args.model} d_site={args.d_site} d_link={args.d_link} dt={args.dt:g} eta={args.eta:g} seed={args.seed} :: "
          f"singular_values_top{topk} = " + ", ".join(f"{x:.6g}" for x in profile))

    # Save JSON
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsf_out", "bandwidth")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{tag}_bandwidth_profile.json")
    out = {
        "timestamp": tag,
        "meta": {
            "script": os.path.basename(__file__),
            "model": args.model,
            "d_site": args.d_site,
            "d_link": args.d_link,
            "dt": args.dt,
            "eta": args.eta,
            "seed": args.seed,
        },
        "coupling": metaH,
        "endpoint_observables": metaO,
        "shapes": {"R": [int(n_obs), int(k_pokes)]},
        "singular_values": s.tolist(),
        "singular_values_topk": profile,
        "note": "Higher singular values = more independent poke directions on A imprint strongly into link endpoint observables in one step.",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()