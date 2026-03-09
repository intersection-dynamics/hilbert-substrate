#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
echo_algebra_step1_qutrit_sites_su3_LR.py
========================================

Goal
----
Extend the Step-1 qutrit-site SU(3) echo-algebra diagnostic to explicitly extract
TWO commuting SU(3) actions on the SAME bond Hilbert space:

  - L^a : "left-end" action (vary left site state, hold right fixed)
  - R^a : "right-end" action (vary right site state, hold left fixed)

and verify the defining relations (numerically):

  [L^a, L^b] closes on span(L)          (Lie closure)
  [R^a, R^b] closes on span(R)          (Lie closure)
  [L^a, R^b] ≈ 0 for all a,b            (left/right independence)

This is the missing ingredient for a real Gauss-law test:
  G_x^a = Q_x^a + sum_out L_link^a - sum_in R_link^a

Model
-----
Bond: dB=3 (qutrit bond register).
Sites: dS=3 (qutrit sites) on both sides of the bond.

Coupling options:
  aligned: H = sum_a  S_a ⊗ B_a ⊗ S_a
  mixed  : H = sum_a  S_a ⊗ B'_a ⊗ S_a   with B' = O B (fixed orthogonal mix)

Echo extraction (channel -> generators)
--------------------------------------
We compute Kraus operators K_{mn} for the bond channel induced by fixing site states.
We map Kraus -> generator(s) using one of:
  - dominant : pick the Kraus with maximum Frobenius weight (legacy behavior)
  - weighted : use a weighted sum over all Kraus anti-Hermitian parts (recommended)

Then we:
  - collect a pool of normalized generators
  - SVD on real Hermitian coordinates -> estimate span dimension
  - reconstruct an HS-orthonormal basis
  - check closure residual stats

Outputs
-------
  ./hsf_out/echo_algebra_step1_qutrit_su3_LR_<timestamp>.json
  ./hsf_out/echo_algebra_step1_qutrit_su3_LR_bases_<timestamp>.npz

Run (Windows one-liner)
-----------------------
python echo_algebra_step1_qutrit_sites_su3_LR.py

You can tweak:
  eps_list, n_samples, reduce_mode ("weighted" recommended)

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
    This is stable for SVD span estimation.
    """
    d = A.shape[0]
    v = []
    # diag (real)
    for i in range(d):
        v.append(float(A[i, i].real))
    # upper triangle
    for i in range(d):
        for j in range(i + 1, d):
            v.append(float(A[i, j].real))
            v.append(float(A[i, j].imag))
    return np.array(v, dtype=float)


def reconstruct_from_real_coords(v: np.ndarray, d: int) -> np.ndarray:
    """
    Inverse of embed_real_coords_hermitian (up to Hermitian).
    """
    A = np.zeros((d, d), dtype=complex)
    idx = 0
    # diag
    for i in range(d):
        A[i, i] = v[idx]
        idx += 1
    # upper
    for i in range(d):
        for j in range(i + 1, d):
            re = v[idx]
            im = v[idx + 1]
            idx += 2
            A[i, j] = re + 1j * im
            A[j, i] = re - 1j * im
    return A


def haar_random_state(d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Haar-random pure state in C^d via complex normal + normalize.
    """
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

def build_H_qutrit_su3(model: str, seed: int = 999):
    """
    aligned:
      H = sum_a S_a ⊗ B_a ⊗ S_a

    mixed:
      H = sum_a S_a ⊗ B'_a ⊗ S_a, where B' = O B for random orthogonal O (seeded)
    """
    dS = 3
    dB = 3

    S_basis = su_generators_gellmann(dS)  # 8
    B_basis = su_generators_gellmann(dB)  # 8

    rng = np.random.default_rng(seed)
    mix_R = np.eye(8, dtype=float)

    if model == "mixed":
        mix_R = random_orthogonal(8, rng)
        B_mixed = []
        for a in range(8):
            M = np.zeros((dB, dB), dtype=complex)
            for b in range(8):
                M += mix_R[b, a] * B_basis[b]
            M = hermitize(M)
            M = traceless(M)
            M = normalize_hs(M)
            B_mixed.append(M)
        B_basis = B_mixed

    elif model != "aligned":
        raise ValueError("model must be 'aligned' or 'mixed'")

    H = np.zeros((dS * dB * dS, dS * dB * dS), dtype=complex)
    for a in range(8):
        Sa = S_basis[a]
        Ba = B_basis[a]
        H += np.kron(np.kron(Sa, Ba), Sa)

    # gentle normalization for numerical stability
    H = H / max(hs_norm(H), 1e-12)

    return H, mix_R


# -------------------------
# Echo channel Kraus extraction for qutrit sites
# -------------------------

def extract_kraus_qutrit(U_full: np.ndarray, dB: int, psi_left: np.ndarray, psi_right: np.ndarray):
    """
    Given U acting on (siteL ⊗ bond ⊗ siteR), and fixed site input state |psiL>|psiR|,
    build Kraus operators K_{mn} acting on the bond, where m is left site output basis index,
    n is right site output basis index.
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
        G = generator_from_kraus(Kd, eps)
        return G

    # weighted
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
    eps: float
    n_samples: int
    seed: int = 12345
    bond_seed: int = 999
    svd_tol_rel: float = 1e-6
    gs_tol: float = 1e-10
    reduce_mode: str = "weighted"   # "weighted" recommended
    fixed_state_mode: str = "haar"  # "haar" or "zero"
    max_keep: int = 20000           # safety cap on pool size


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
      both  : vary psiL and psiR (original)
      left  : vary psiL, hold psiR fixed
      right : vary psiR, hold psiL fixed
    """
    dB = 3
    rng = np.random.default_rng(cfg.seed)

    H, mix_R = build_H_qutrit_su3(cfg.model, seed=cfg.bond_seed)
    U = expm(-1j * cfg.eps * H)

    # fixed partner state (deterministic given seed)
    rng_fix = np.random.default_rng(cfg.seed + 777)
    psiL_fix = _fixed_state(3, rng_fix, cfg.fixed_state_mode)
    psiR_fix = _fixed_state(3, rng_fix, cfg.fixed_state_mode)

    pool = []
    for _ in range(cfg.n_samples):
        if vary == "both":
            psiL = haar_random_state(3, rng)
            psiR = haar_random_state(3, rng)
        elif vary == "left":
            psiL = haar_random_state(3, rng)
            psiR = psiR_fix
        elif vary == "right":
            psiL = psiL_fix
            psiR = haar_random_state(3, rng)
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
        return [], np.array([]), 0, mix_R

    V = np.stack([embed_real_coords_hermitian(A) for A in pool], axis=0)
    _, S, Vh = np.linalg.svd(V, full_matrices=False)

    if S.size == 0:
        return [], S, 0, mix_R

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
    return basis, S, len(basis), mix_R


# -------------------------
# Diagnostics
# -------------------------

def closure_residual_stats(basis):
    """
    Project each commutator [Ti,Tj] onto span(basis) and report relative residuals.
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
    Report normalized Frobenius/HS magnitudes.
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
    print(f"  model={res['model']}  eps={res['eps']:.1e}  samples={res['n_samples']}  reduce={res['reduce_mode']}  fixed={res['fixed_state_mode']}")
    print(f"  basis_dim={res['basis_dim']}  (expected 8)")
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
    print("STEP 1 Diagnostic — Qutrit sites probing SU(3) bond (dS=3, dB=3)")
    print("NOW WITH explicit LEFT/RIGHT span extraction for link-end generators (L^a, R^a).")
    print("=" * 78)
    print(f"outputs: {out_dir}")

    models = ["aligned", "mixed"]
    eps_list = [1e-4, 3e-4, 1e-3]
    n_samples = 20000

    # Recommended defaults for LR separation:
    # - weighted reduction stabilizes the extracted generator as a channel "direction"
    # - fixed partner state via Haar draw avoids privileging computational basis too much
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
    best_cache = {}  # (model, vary) -> (score, basis)

    for model in models:
        for eps in eps_list:
            cfg = SweepConfig(model=model, eps=eps, **base_cfg)

            # Full span (both)
            basis_B, S_B, dim_B, mix_R = extract_span_basis(cfg, vary="both")
            res_B = {
                "kind": "both",
                "model": model,
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
                "meta": {"mix_R_det": float(np.linalg.det(mix_R))},
            }
            print_result_line("[BOTH] span extraction", res_B)
            all_results.append(res_B)

            # Left span (vary left)
            basis_L, S_L, dim_L, _ = extract_span_basis(cfg, vary="left")
            res_L = {
                "kind": "left",
                "model": model,
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
                "meta": {"mix_R_det": float(np.linalg.det(mix_R))},
            }
            print_result_line("[LEFT] span extraction (candidate L^a)", res_L)
            all_results.append(res_L)

            # Right span (vary right)
            basis_R, S_R, dim_R, _ = extract_span_basis(cfg, vary="right")
            res_R = {
                "kind": "right",
                "model": model,
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
                "meta": {"mix_R_det": float(np.linalg.det(mix_R))},
            }
            print_result_line("[RIGHT] span extraction (candidate R^a)", res_R)
            all_results.append(res_R)

            # Cross commutator diagnostic between extracted L and R spans
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
                "eps": eps,
                "n_samples": n_samples,
                "reduce_mode": cfg.reduce_mode,
                "fixed_state_mode": cfg.fixed_state_mode,
                "cross": cross,
            })

            # Cache best bases per (model, kind) for NPZ
            def score(res):
                # prefer full dimension, then smaller closure median residual
                cl_med = res["closure"].get("rel_med", 1e9)
                return (res["basis_dim"], -cl_med)

            for (kind, basis, res) in [
                ("both", basis_B, res_B),
                ("left", basis_L, res_L),
                ("right", basis_R, res_R),
            ]:
                key = (model, kind)
                sc = score(res)
                if key not in best_cache or sc > best_cache[key][0]:
                    best_cache[key] = (sc, basis)

    # Save JSON
    tag = now_tag()
    out_json = os.path.join(out_dir, f"echo_algebra_step1_qutrit_su3_LR_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("=" * 78)
    print(f"[saved] {out_json}")

    # Save NPZ (best bases)
    npz_path = os.path.join(out_dir, f"echo_algebra_step1_qutrit_su3_LR_bases_{tag}.npz")
    npz_dict = {}
    for (model, kind), (_sc, basis) in best_cache.items():
        if not basis:
            continue
        npz_dict[f"basis_{kind}_{model}"] = np.stack(basis, axis=0)

    if npz_dict:
        np.savez(npz_path, **npz_dict)
        print(f"[saved] {npz_path}")
        print("  saved arrays:", ", ".join(sorted(npz_dict.keys())))
    else:
        print("[warn] no bases to save (all empty)")

    print("=" * 78)
    print("Interpretation guide:")
    print("  - PASS(L): basis_dim_left==8 with good closure -> you have an explicit candidate L^a set.")
    print("  - PASS(R): basis_dim_right==8 with good closure -> you have an explicit candidate R^a set.")
    print("  - CRITICAL: cross commutators small: rel_max([L_a,R_b]) ≪ 1 (ideally ~1e-2 or better).")
    print("  - If cross commutators are not small, try:")
    print("      * reduce_mode='weighted' (already default),")
    print("      * fixed_state_mode='haar' vs 'zero',")
    print("      * increase n_samples,")
    print("      * tighten/loosen svd_tol_rel,")
    print("      * or move away from 'dominant Kraus' entirely (channel-level generator fit).")
    print("=" * 78)


if __name__ == "__main__":
    main()
