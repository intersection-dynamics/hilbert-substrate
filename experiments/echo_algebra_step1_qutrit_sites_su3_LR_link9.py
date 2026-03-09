#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
echo_algebra_step1_qutrit_sites_su3_LR_link9.py
==============================================

Goal
----
Generate the *next dataset* you need: endpoint SU(3) generators acting on a
*larger* bond/link Hilbert space (dimension 9), so that a genuine split into
two commuting endpoint actions is possible in principle.

This is a link-promotion version of:
  echo_algebra_step1_qutrit_sites_su3_LR.py

Key change
----------
Bond register is now:
  H_E ≅ C^9 ≅ C^3 ⊗ C^3

and we explicitly *build the Hamiltonian* so that the bond supports two commuting
SU(3) actions:
  - left bond action uses generators (Q_a ⊗ I)
  - right bond action uses generators (I ⊗ Q_a)

Coupling styles
---------------
1) two_leg  (DEFAULT, recommended)
   H = sum_a  S_a ⊗ (B_L[a]) ⊗ I   +   sum_a  I ⊗ (B_R[a]) ⊗ S_a
   where:
     B_L[a] = Q_a ⊗ I
     B_R[a] = I ⊗ Q_a
   This is the minimal structure that allows extracted L/R to commute on the bond.

2) single_leg (legacy / comparison)
   H = sum_a  S_a ⊗ (B[a]) ⊗ S_a
   where B[a] is built from Q_a ⊗ I (so bond action is only one SU(3)).
   This is expected to reproduce the qutrit-link “48 wall” type behavior (no commuting split).

Mixing modes
------------
model = aligned:
  uses canonical Q_a on the relevant bond factor(s)

model = mixed:
  applies a fixed random orthogonal mixing in generator space separately to the
  left and right bond actions (seeded), i.e.
     B_L' = O_L · B_L
     B_R' = O_R · B_R
  This keeps each endpoint action su(3), but changes the embedding.

Echo extraction (unchanged)
---------------------------
We compute Kraus operators for the bond channel induced by fixing site states.
We reduce Kraus -> generator using:
  - dominant : pick the Kraus with max Frobenius weight (legacy)
  - weighted : weighted sum over all Kraus anti-Hermitian parts (recommended)

We then:
  - collect generator pool, estimate span with SVD,
  - reconstruct HS-orthonormal basis for each "both/left/right" extraction,
  - report closure and cross commutators,
  - write JSON + NPZ with arrays shaped (8, 9, 9) when successful.

Outputs
-------
  ./hsf_out/echo_algebra_step1_link9_su3_LR_<timestamp>.json
  ./hsf_out/echo_algebra_step1_link9_su3_LR_bases_<timestamp>.npz

Run (Windows one-liner)
-----------------------
python echo_algebra_step1_qutrit_sites_su3_LR_link9.py

Then run your v7.0 harness on the emitted NPZ:
python gauss_link_npz_diagnostic_v7_0.py --npz <that_npz> --mode aligned

Dependencies: numpy, scipy
"""

import os
import math
import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    from scipy.linalg import expm
except Exception as e:
    raise RuntimeError("scipy is required (scipy.linalg.expm). Install scipy.") from e


# -------------------------
# Small utilities
# -------------------------

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_out_dir() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def hs_inner(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.trace(A.conj().T @ B).real)


def hs_norm(A: np.ndarray) -> float:
    return math.sqrt(max(hs_inner(A, A), 0.0))


def hermitize(A: np.ndarray) -> np.ndarray:
    return (A + A.conj().T) / 2.0


def traceless(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d, dtype=complex)


def normalize_hs(A: np.ndarray) -> np.ndarray:
    n = hs_norm(A)
    if n < 1e-30:
        return A.copy()
    return A / n


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def gram_schmidt_hs(basis, tol=1e-10):
    out = []
    for A in basis:
        B = A.copy()
        for Q in out:
            B -= hs_inner(Q, B) * Q
        n = hs_norm(B)
        if n > tol:
            out.append(B / n)
    return out


def embed_real_coords_hermitian(A: np.ndarray) -> np.ndarray:
    """
    Map Hermitian dxd matrix to a real vector of length d^2:
      [Re(diag), Re(upper), Im(upper)]
    """
    d = A.shape[0]
    v = []
    for i in range(d):
        v.append(float(A[i, i].real))
    for i in range(d):
        for j in range(i + 1, d):
            v.append(float(A[i, j].real))
            v.append(float(A[i, j].imag))
    return np.array(v, dtype=float)


def reconstruct_from_real_coords(v: np.ndarray, d: int) -> np.ndarray:
    A = np.zeros((d, d), dtype=complex)
    idx = 0
    for i in range(d):
        A[i, i] = v[idx]
        idx += 1
    for i in range(d):
        for j in range(i + 1, d):
            re = v[idx]
            im = v[idx + 1]
            idx += 2
            A[i, j] = re + 1j * im
            A[j, i] = re - 1j * im
    return A


def haar_random_state(d: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=d) + 1j * rng.normal(size=d)
    n = np.linalg.norm(z)
    if n < 1e-30:
        z[0] = 1.0
        n = 1.0
    return z / n


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    # allow det=-1; we only need orthogonality in generator mixing
    return Q


# -------------------------
# su(3) generators (HS-orthonormal)
# -------------------------

def su_generators_gellmann(d: int):
    """
    HS-orthonormal Hermitian traceless su(d) basis.
    For d=3 returns 8 generators.
    """
    gens = []

    # symmetric + antisymmetric off-diagonals
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

    # diagonal traceless (d-1)
    for k in range(1, d):
        D = np.zeros((d, d), dtype=complex)
        for i in range(k):
            D[i, i] = 1.0
        D[k, k] = -float(k)
        D = D * math.sqrt(2.0 / (k * (k + 1.0)))
        gens.append(D)

    out = []
    for G in gens:
        H = hermitize(G)
        H = traceless(H)
        H = normalize_hs(H)
        out.append(H)

    out = gram_schmidt_hs(out, tol=1e-12)
    return out


# -------------------------
# Build Hamiltonian H on (siteL ⊗ bond ⊗ siteR)
# -------------------------

def mix_generators_orth(basis: list[np.ndarray], O: np.ndarray) -> list[np.ndarray]:
    """
    Given basis {B_b}, return B'_a = sum_b O[b,a] B_b (column convention),
    then hermitize/traceless/normalize each.
    """
    d = basis[0].shape[0]
    out = []
    for a in range(8):
        M = np.zeros((d, d), dtype=complex)
        for b in range(8):
            M += O[b, a] * basis[b]
        M = hermitize(M)
        M = traceless(M)
        M = normalize_hs(M)
        out.append(M)
    return out


def build_H_qutrit_sites_link9_su3(model: str,
                                  coupling: str = "two_leg",
                                  seed: int = 999):
    """
    Sites:
      dS = 3 (qutrit)
    Bond/Link:
      dB = 9 = 3⊗3

    coupling:
      two_leg  : H = Σ_a  S_a ⊗ B_L[a] ⊗ I  +  Σ_a  I ⊗ B_R[a] ⊗ S_a
      single_leg: H = Σ_a S_a ⊗ B[a] ⊗ S_a  (legacy comparison; uses B = Q⊗I)

    model:
      aligned: canonical su(3) basis on each bond factor
      mixed  : fixed random orthogonal mixing on generator index separately for B_L and B_R
    """
    dS = 3
    dB_factor = 3
    dB = dB_factor * dB_factor  # 9

    S_basis = su_generators_gellmann(dS)          # 8
    Q = su_generators_gellmann(dB_factor)         # 8 on each factor
    I3 = np.eye(dB_factor, dtype=complex)
    IS = np.eye(dS, dtype=complex)

    # Bond endpoint actions as 9x9 generators
    B_L = [np.kron(Q[a], I3) for a in range(8)]   # acts on left bond factor
    B_R = [np.kron(I3, Q[a]) for a in range(8)]   # acts on right bond factor

    # Clean/normalize (keep HS scale stable)
    B_L = [normalize_hs(traceless(hermitize(X))) for X in B_L]
    B_R = [normalize_hs(traceless(hermitize(X))) for X in B_R]

    rng = np.random.default_rng(seed)
    mix_L = np.eye(8, dtype=float)
    mix_R = np.eye(8, dtype=float)

    if model == "mixed":
        mix_L = random_orthogonal(8, rng)
        mix_R = random_orthogonal(8, rng)
        B_L = mix_generators_orth(B_L, mix_L)
        B_R = mix_generators_orth(B_R, mix_R)
    elif model != "aligned":
        raise ValueError("model must be 'aligned' or 'mixed'")

    # Build H on (siteL ⊗ bond ⊗ siteR) with dims (3,9,3)
    H = np.zeros((dS * dB * dS, dS * dB * dS), dtype=complex)

    if coupling == "two_leg":
        # Σ_a S_a ⊗ B_L[a] ⊗ I  +  Σ_a I ⊗ B_R[a] ⊗ S_a
        for a in range(8):
            Sa = S_basis[a]
            H += np.kron(np.kron(Sa, B_L[a]), IS)
            H += np.kron(np.kron(IS, B_R[a]), Sa)

    elif coupling == "single_leg":
        # Legacy: Σ_a S_a ⊗ (B_L[a]) ⊗ S_a  (only one bond action effectively)
        for a in range(8):
            Sa = S_basis[a]
            H += np.kron(np.kron(Sa, B_L[a]), Sa)
    else:
        raise ValueError("coupling must be 'two_leg' or 'single_leg'")

    # Gentle normalization for numerical stability
    H = H / max(hs_norm(H), 1e-12)

    return H, {"mix_L_det": float(np.linalg.det(mix_L)),
               "mix_R_det": float(np.linalg.det(mix_R)),
               "coupling": coupling,
               "dB": dB}


# -------------------------
# Echo channel Kraus extraction for qutrit sites (dS=3, dB variable)
# -------------------------

def extract_kraus_qutrit(U_full: np.ndarray, dB: int, psi_left: np.ndarray, psi_right: np.ndarray):
    """
    Given U acting on (siteL ⊗ bond ⊗ siteR), and fixed site input state |psiL>|psiR|,
    build Kraus operators K_{mn} acting on the bond, where m is left site output basis index,
    n is right site output basis index.

    Returns list of 9 Kraus ops (m,n in {0,1,2}) each of shape (dB,dB).
    """
    dS = 3
    d_full = dS * dB * dS

    embed = np.zeros((d_full, dB), dtype=complex)
    # embed: bond basis -> full basis with sites fixed to psi_left, psi_right
    for b in range(dB):
        for a in range(dS):
            for c in range(dS):
                row = a * (dB * dS) + b * dS + c
                embed[row, b] = psi_left[a] * psi_right[c]

    U_embed = U_full @ embed

    kraus_ops = []
    for m in range(dS):
        for n in range(dS):
            K = np.zeros((dB, dB), dtype=complex)
            for b_out in range(dB):
                row = m * (dB * dS) + b_out * dS + n
                for b_in in range(dB):
                    K[b_out, b_in] = U_embed[row, b_in]
            kraus_ops.append(K)
    return kraus_ops


def generator_from_kraus(K: np.ndarray, eps: float) -> np.ndarray:
    """
    For small eps, a near-identity Kraus branch behaves like exp(-i eps H_eff) up to scaling.
    Use the anti-Hermitian part / eps as a generator proxy.
    """
    H_eff = (K - K.conj().T) / (2.0j * eps)
    H_eff = hermitize(H_eff)
    H_eff = traceless(H_eff)
    return H_eff


def reduce_kraus_to_generator(Ks, eps: float, mode: str = "weighted") -> np.ndarray:
    """
    mode:
      dominant : choose max Frobenius-weight Kraus (legacy)
      weighted : weighted sum over ALL Kraus generator proxies (recommended)
    """
    if mode not in ("dominant", "weighted"):
        raise ValueError("reduce mode must be 'dominant' or 'weighted'")

    weights = np.array([np.linalg.norm(K, "fro") ** 2 for K in Ks], dtype=float)
    if np.all(weights <= 0):
        return np.zeros_like(Ks[0])

    if mode == "dominant":
        Kd = Ks[int(np.argmax(weights))]
        return generator_from_kraus(Kd, eps)

    w = weights / (weights.sum() + 1e-300)
    G = np.zeros_like(Ks[0])
    for wi, Ki in zip(w, Ks):
        G += wi * generator_from_kraus(Ki, eps)
    G = hermitize(G)
    G = traceless(G)
    return G


# -------------------------
# Span extraction
# -------------------------

@dataclass
class SweepConfig:
    model: str
    coupling: str
    eps: float
    n_samples: int
    seed: int = 12345
    bond_seed: int = 999
    svd_tol_rel: float = 1e-6
    gs_tol: float = 1e-10
    reduce_mode: str = "weighted"   # "weighted" recommended
    fixed_state_mode: str = "haar"  # "haar" or "zero"
    max_keep: int = 20000


def _fixed_state(d: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "zero":
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1.0
        return psi
    if mode == "haar":
        return haar_random_state(d, rng)
    raise ValueError("fixed_state_mode must be 'haar' or 'zero'")


def extract_span_basis(cfg: SweepConfig, vary: str):
    """
    vary:
      both  : vary psiL and psiR
      left  : vary psiL, hold psiR fixed
      right : vary psiR, hold psiL fixed
    """
    dS = 3
    dB = 9
    rng = np.random.default_rng(cfg.seed)

    H, meta = build_H_qutrit_sites_link9_su3(cfg.model, coupling=cfg.coupling, seed=cfg.bond_seed)
    U = expm(-1j * cfg.eps * H)

    # fixed partner state (deterministic given seed)
    rng_fix = np.random.default_rng(cfg.seed + 777)
    psiL_fix = _fixed_state(dS, rng_fix, cfg.fixed_state_mode)
    psiR_fix = _fixed_state(dS, rng_fix, cfg.fixed_state_mode)

    pool = []
    for _ in range(cfg.n_samples):
        if vary == "both":
            psiL = haar_random_state(dS, rng)
            psiR = haar_random_state(dS, rng)
        elif vary == "left":
            psiL = haar_random_state(dS, rng)
            psiR = psiR_fix
        elif vary == "right":
            psiL = psiL_fix
            psiR = haar_random_state(dS, rng)
        else:
            raise ValueError("vary must be 'both', 'left', or 'right'")

        Ks = extract_kraus_qutrit(U, dB, psiL, psiR)
        G = reduce_kraus_to_generator(Ks, cfg.eps, mode=cfg.reduce_mode)
        n = hs_norm(G)
        if n > 1e-12:
            pool.append(G / n)
            if len(pool) >= cfg.max_keep:
                break

    if not pool:
        return [], np.array([]), 0, meta

    V = np.stack([embed_real_coords_hermitian(A) for A in pool], axis=0)
    _, S, Vh = np.linalg.svd(V, full_matrices=False)

    if S.size == 0:
        return [], S, 0, meta

    thresh = cfg.svd_tol_rel * S[0]
    dim_est = int(np.sum(S > thresh))
    basis_vecs = Vh[:dim_est, :]

    basis = []
    for v in basis_vecs:
        A = reconstruct_from_real_coords(v, dB)
        A = hermitize(A)
        A = traceless(A)
        A = normalize_hs(A)
        if hs_norm(A) > 1e-12:
            basis.append(A)

    basis = gram_schmidt_hs(basis, tol=cfg.gs_tol)
    return basis, S, len(basis), meta


# -------------------------
# Diagnostics
# -------------------------

def closure_residual_stats(basis):
    """
    Project each commutator [Ti,Tj] onto span(basis) and report relative residuals.
    NOTE: This is the original diagnostic (not the -i corrected one).
    It's still useful as a rough stability check.
    """
    if len(basis) == 0:
        return {"pairs": 0}

    rel_resids = []
    k = len(basis)
    for i in range(k):
        for j in range(i + 1, k):
            C = commutator(basis[i], basis[j])
            Cn = hs_norm(C)
            if Cn < 1e-14:
                continue
            P = np.zeros_like(C)
            for t in basis:
                P += hs_inner(t, C) * t
            R = C - P
            rn = hs_norm(R)
            rel_resids.append(rn / (Cn + 1e-30))

    if not rel_resids:
        return {"pairs": 0}

    rel = np.array(rel_resids, dtype=float)
    return {
        "pairs": int(rel.size),
        "rel_min": float(rel.min()),
        "rel_med": float(np.median(rel)),
        "rel_max": float(rel.max()),
        "rel_mean": float(rel.mean()),
    }


def cross_comm_stats(L_basis, R_basis):
    """
    Measure how close [L_a, R_b] is to zero for all pairs.
    """
    if not L_basis or not R_basis:
        return {"pairs": 0}

    vals = []
    for La in L_basis:
        Ln = hs_norm(La)
        for Rb in R_basis:
            Rn = hs_norm(Rb)
            C = commutator(La, Rb)
            Cn = hs_norm(C)
            vals.append(Cn / (Ln * Rn + 1e-300))

    arr = np.array(vals, dtype=float)
    return {
        "pairs": int(arr.size),
        "rel_min": float(arr.min()),
        "rel_med": float(np.median(arr)),
        "rel_max": float(arr.max()),
        "rel_mean": float(arr.mean()),
    }


def print_result_line(tag: str, res: dict):
    print("-" * 78)
    print(tag)
    print(f"  model={res['model']}  coupling={res['coupling']}  eps={res['eps']:.1e}  samples={res['n_samples']}  reduce={res['reduce_mode']}  fixed={res['fixed_state_mode']}")
    print(f"  basis_dim={res['basis_dim']}  (target 8)")
    if res["sv_top5"]:
        print(f"  top singulars: {', '.join(f'{x:.3g}' for x in res['sv_top5'])}")
    cl = res["closure"]
    if cl.get("pairs", 0) > 0:
        print(f"  closure pairs={cl['pairs']}  rel_med={cl['rel_med']:.3e}  rel_max={cl['rel_max']:.3e}")
    else:
        print("  closure: (insufficient pairs / degenerate commutators)")


# -------------------------
# Main
# -------------------------

def main():
    out_dir = ensure_out_dir()
    print("=" * 78)
    print("STEP 1 Diagnostic — Qutrit sites probing SU(3) bond")
    print("LINK PROMOTION: dB = 9 = 3⊗3, with explicit TWO-LEG couplings for commuting endpoints.")
    print("=" * 78)
    print(f"outputs: {out_dir}")

    # You can safely start with just two_leg; single_leg is included for comparison.
    models = ["aligned", "mixed"]
    couplings = ["two_leg"]   # change to ["two_leg", "single_leg"] if you want comparison runs
    eps_list = [1e-4, 3e-4, 1e-3]
    n_samples = 20000

    base_cfg = dict(
        n_samples=n_samples,
        seed=12345,
        bond_seed=999,
        svd_tol_rel=1e-6,
        gs_tol=1e-10,
        reduce_mode="weighted",
        fixed_state_mode="haar",
        max_keep=n_samples,
    )

    all_results = []
    best_cache = {}  # (model, coupling, vary) -> (score, basis)

    for model in models:
        for coupling in couplings:
            for eps in eps_list:
                cfg = SweepConfig(model=model, coupling=coupling, eps=eps, **base_cfg)

                # BOTH span
                basis_B, S_B, dim_B, metaH = extract_span_basis(cfg, vary="both")
                res_B = {
                    "kind": "both",
                    "model": model,
                    "coupling": coupling,
                    "eps": eps,
                    "n_samples": n_samples,
                    "seed": cfg.seed,
                    "bond_seed": cfg.bond_seed,
                    "reduce_mode": cfg.reduce_mode,
                    "fixed_state_mode": cfg.fixed_state_mode,
                    "expected_dim": 8,
                    "basis_dim": dim_B,
                    "sv_top5": [float(x) for x in (S_B[:5] if S_B.size else [])],
                    "closure": closure_residual_stats(basis_B),
                    "meta": metaH,
                }
                print_result_line("[BOTH] span extraction", res_B)
                all_results.append(res_B)

                # LEFT span (vary left site)
                basis_L, S_L, dim_L, _ = extract_span_basis(cfg, vary="left")
                res_L = {
                    "kind": "left",
                    "model": model,
                    "coupling": coupling,
                    "eps": eps,
                    "n_samples": n_samples,
                    "seed": cfg.seed,
                    "bond_seed": cfg.bond_seed,
                    "reduce_mode": cfg.reduce_mode,
                    "fixed_state_mode": cfg.fixed_state_mode,
                    "expected_dim": 8,
                    "basis_dim": dim_L,
                    "sv_top5": [float(x) for x in (S_L[:5] if S_L.size else [])],
                    "closure": closure_residual_stats(basis_L),
                    "meta": metaH,
                }
                print_result_line("[LEFT] span extraction (candidate L^a)", res_L)
                all_results.append(res_L)

                # RIGHT span (vary right site)
                basis_R, S_R, dim_R, _ = extract_span_basis(cfg, vary="right")
                res_R = {
                    "kind": "right",
                    "model": model,
                    "coupling": coupling,
                    "eps": eps,
                    "n_samples": n_samples,
                    "seed": cfg.seed,
                    "bond_seed": cfg.bond_seed,
                    "reduce_mode": cfg.reduce_mode,
                    "fixed_state_mode": cfg.fixed_state_mode,
                    "expected_dim": 8,
                    "basis_dim": dim_R,
                    "sv_top5": [float(x) for x in (S_R[:5] if S_R.size else [])],
                    "closure": closure_residual_stats(basis_R),
                    "meta": metaH,
                }
                print_result_line("[RIGHT] span extraction (candidate R^a)", res_R)
                all_results.append(res_R)

                # CROSS commutator diagnostic
                cross = cross_comm_stats(basis_L, basis_R)
                print("-" * 78)
                print("[CROSS] left/right independence check: [L_a, R_b] ~ 0")
                if cross.get("pairs", 0) > 0:
                    print(f"  pairs={cross['pairs']}  rel_med={cross['rel_med']:.3e}  rel_max={cross['rel_max']:.3e}  rel_mean={cross['rel_mean']:.3e}")
                else:
                    print("  (insufficient pairs / empty basis)")
                all_results.append({
                    "kind": "cross",
                    "model": model,
                    "coupling": coupling,
                    "eps": eps,
                    "n_samples": n_samples,
                    "reduce_mode": cfg.reduce_mode,
                    "fixed_state_mode": cfg.fixed_state_mode,
                    "cross": cross,
                })

                # cache best bases per (model, coupling, kind) for NPZ
                def score(res):
                    cl_med = res["closure"].get("rel_med", 1e9)
                    return (res["basis_dim"], -cl_med)

                for (kind, basis, res) in [
                    ("both", basis_B, res_B),
                    ("left", basis_L, res_L),
                    ("right", basis_R, res_R),
                ]:
                    key = (model, coupling, kind)
                    sc = score(res)
                    if key not in best_cache or sc > best_cache[key][0]:
                        best_cache[key] = (sc, basis)

    # Save JSON
    tag = now_tag()
    out_json = os.path.join(out_dir, f"echo_algebra_step1_link9_su3_LR_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("=" * 78)
    print(f"[saved] {out_json}")

    # Save NPZ (best bases)
    npz_path = os.path.join(out_dir, f"echo_algebra_step1_link9_su3_LR_bases_{tag}.npz")
    npz_dict = {}
    for (model, coupling, kind), (_sc, basis) in best_cache.items():
        if not basis:
            continue
        key = f"basis_{kind}_{model}" if coupling == "two_leg" else f"basis_{kind}_{model}_{coupling}"
        # Expect 8 generators; if dim != 8, still save for inspection
        npz_dict[key] = np.stack(basis, axis=0)

    if npz_dict:
        np.savez(npz_path, **npz_dict)
        print(f"[saved] {npz_path}")
        print("  saved arrays:", ", ".join(sorted(npz_dict.keys())))
    else:
        print("[warn] no bases to save (all empty)")

    print("=" * 78)
    print("Interpretation guide:")
    print("  - Success signal: basis_dim_left==8 AND basis_dim_right==8, and cross commutators are small:")
    print("      rel_max([L_a,R_b]) ≪ 1 (ideally ~1e-2 or better)")
    print("  - If two_leg still yields large cross commutators, increase n_samples and/or try eps_list finer.")
    print("=" * 78)


if __name__ == "__main__":
    main()