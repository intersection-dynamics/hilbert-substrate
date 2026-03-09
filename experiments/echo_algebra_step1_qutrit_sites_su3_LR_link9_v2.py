#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
echo_algebra_step1_qutrit_sites_su3_LR_link9_v2.py
==================================================

What this fixes
---------------
Your link9 run produced bases with k=18, so v7.0 (which expects k=8 for SU(3))
refused the NPZ.

This v2 generator script saves BOTH:
- the FULL basis it finds (k_full, e.g. 18) under keys ending with "_full"
- a TRUNCATED k=8 version under the STANDARD keys expected by v7.0:
    basis_left_aligned  -> (8, 9, 9)
    basis_right_aligned -> (8, 9, 9)
    basis_both_aligned  -> (8, 9, 9)
(and same for mixed)

So you can immediately run:
  python gauss_link_npz_diagnostic_v7_0.py --npz <npz> --mode aligned

Outputs
-------
  hsf_out/echo_algebra_step1_link9_su3_LR_<timestamp>.json
  hsf_out/echo_algebra_step1_link9_su3_LR_bases_<timestamp>.npz

Run (Windows one-liner)
-----------------------
python echo_algebra_step1_qutrit_sites_su3_LR_link9_v2.py
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
# Utilities
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
    """Hermitian dxd -> real vector of length d^2: [Re diag, Re upper, Im upper]."""
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
    return Q


# -------------------------
# su(3) generators (HS-orthonormal)
# -------------------------

def su_generators_gellmann(d: int):
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

    out = []
    for G in gens:
        H = hermitize(G)
        H = traceless(H)
        H = normalize_hs(H)
        out.append(H)

    out = gram_schmidt_hs(out, tol=1e-12)
    return out


# -------------------------
# Build H on (siteL ⊗ bond ⊗ siteR) with link dim 9
# -------------------------

def mix_generators_orth(basis: list[np.ndarray], O: np.ndarray) -> list[np.ndarray]:
    d = basis[0].shape[0]
    out = []
    k = len(basis)
    for a in range(k):
        M = np.zeros((d, d), dtype=complex)
        for b in range(k):
            M += O[b, a] * basis[b]
        M = hermitize(M)
        M = traceless(M)
        M = normalize_hs(M)
        out.append(M)
    return out


def build_H_qutrit_sites_link9_su3(model: str,
                                  coupling: str = "two_leg",
                                  seed: int = 999):
    dS = 3
    dB_factor = 3
    dB = 9

    S_basis = su_generators_gellmann(dS)          # 8
    Q = su_generators_gellmann(dB_factor)         # 8 on each factor
    I3 = np.eye(dB_factor, dtype=complex)
    IS = np.eye(dS, dtype=complex)

    B_L = [np.kron(Q[a], I3) for a in range(8)]   # 9x9
    B_R = [np.kron(I3, Q[a]) for a in range(8)]   # 9x9
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

    H = np.zeros((dS * dB * dS, dS * dB * dS), dtype=complex)

    if coupling == "two_leg":
        for a in range(8):
            Sa = S_basis[a]
            H += np.kron(np.kron(Sa, B_L[a]), IS)
            H += np.kron(np.kron(IS, B_R[a]), Sa)
    elif coupling == "single_leg":
        for a in range(8):
            Sa = S_basis[a]
            H += np.kron(np.kron(Sa, B_L[a]), Sa)
    else:
        raise ValueError("coupling must be 'two_leg' or 'single_leg'")

    H = H / max(hs_norm(H), 1e-12)

    return H, {"mix_L_det": float(np.linalg.det(mix_L)),
               "mix_R_det": float(np.linalg.det(mix_R)),
               "coupling": coupling,
               "dB": dB}


# -------------------------
# Kraus extraction
# -------------------------

def extract_kraus_qutrit(U_full: np.ndarray, dB: int, psi_left: np.ndarray, psi_right: np.ndarray):
    dS = 3
    d_full = dS * dB * dS

    embed = np.zeros((d_full, dB), dtype=complex)
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
    H_eff = (K - K.conj().T) / (2.0j * eps)
    H_eff = hermitize(H_eff)
    H_eff = traceless(H_eff)
    return H_eff


def reduce_kraus_to_generator(Ks, eps: float, mode: str = "weighted") -> np.ndarray:
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
    G = hermitize(traceless(G))
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
    reduce_mode: str = "weighted"
    fixed_state_mode: str = "haar"
    max_keep: int = 20000
    k_save: int = 8  # <= THIS is the key v2 feature


def _fixed_state(d: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "zero":
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1.0
        return psi
    if mode == "haar":
        return haar_random_state(d, rng)
    raise ValueError("fixed_state_mode must be 'haar' or 'zero'")


def extract_span_basis(cfg: SweepConfig, vary: str):
    dS = 3
    dB = 9
    rng = np.random.default_rng(cfg.seed)

    H, meta = build_H_qutrit_sites_link9_su3(cfg.model, coupling=cfg.coupling, seed=cfg.bond_seed)
    U = expm(-1j * cfg.eps * H)

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
        return [], np.array([]), 0, 0, meta

    V = np.stack([embed_real_coords_hermitian(A) for A in pool], axis=0)
    _, S, Vh = np.linalg.svd(V, full_matrices=False)

    if S.size == 0:
        return [], S, 0, 0, meta

    thresh = cfg.svd_tol_rel * S[0]
    dim_est = int(np.sum(S > thresh))

    # FULL basis vectors (dim_est)
    basis_vecs_full = Vh[:dim_est, :]
    # TRUNCATED basis vectors (k_save)
    dim_trunc = int(min(dim_est, cfg.k_save))
    basis_vecs_trunc = Vh[:dim_trunc, :]

    def build_basis(vecs):
        basis = []
        for v in vecs:
            A = reconstruct_from_real_coords(v, dB)
            A = normalize_hs(traceless(hermitize(A)))
            if hs_norm(A) > 1e-12:
                basis.append(A)
        basis = gram_schmidt_hs(basis, tol=cfg.gs_tol)
        return basis

    basis_full = build_basis(basis_vecs_full)
    basis_trunc = build_basis(basis_vecs_trunc)

    return basis_trunc, basis_full, S, len(basis_trunc), len(basis_full), meta


# -------------------------
# Main
# -------------------------

def main():
    out_dir = ensure_out_dir()
    tag = now_tag()

    print("=" * 78)
    print("STEP 1 Link9 generator (v2): saves k=8 bases for v7.0 AND full-k bases.")
    print("=" * 78)
    print(f"outputs: {out_dir}")

    models = ["aligned", "mixed"]
    coupling = "two_leg"
    eps_list = [1e-4, 3e-4, 1e-3]
    n_samples = 20000

    cfg_common = dict(
        coupling=coupling,
        n_samples=n_samples,
        seed=12345,
        bond_seed=999,
        svd_tol_rel=1e-6,
        gs_tol=1e-10,
        reduce_mode="weighted",
        fixed_state_mode="haar",
        max_keep=n_samples,
        k_save=8,
    )

    all_results = []
    # We keep the best run per (model, kind) separately for truncated and full
    best_trunc = {}  # (model, kind) -> (basis, score, eps)
    best_full = {}   # (model, kind) -> (basis, score, eps)

    def score(basis_trunc, basis_full, S):
        # Prefer having >=8 in full, then >=8 in trunc, then stronger singular separation.
        s0 = float(S[0]) if S.size else 0.0
        return (len(basis_full) >= 8, len(basis_trunc) >= 8, s0)

    for model in models:
        for eps in eps_list:
            cfg = SweepConfig(model=model, eps=eps, **cfg_common)

            for kind in ("both", "left", "right"):
                basis_tr, basis_full, S, dim_tr, dim_full, metaH = extract_span_basis(cfg, vary=kind)
                rec = {
                    "model": model,
                    "kind": kind,
                    "eps": eps,
                    "n_samples": n_samples,
                    "dim_trunc": dim_tr,
                    "dim_full": dim_full,
                    "sv_top5": [float(x) for x in (S[:5] if S.size else [])],
                    "meta": metaH,
                }
                all_results.append(rec)

                sc = score(basis_tr, basis_full, S)

                key = (model, kind)
                if key not in best_trunc or sc > best_trunc[key][1]:
                    best_trunc[key] = (basis_tr, sc, eps)
                if key not in best_full or sc > best_full[key][1]:
                    best_full[key] = (basis_full, sc, eps)

                print("-" * 78)
                print(f"{model.upper()} eps={eps:.1e} kind={kind}: trunc_dim={dim_tr} full_dim={dim_full}  topS={rec['sv_top5'][:3]}")

    # Save JSON summary
    out_json = os.path.join(out_dir, f"echo_algebra_step1_link9_su3_LR_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("=" * 78)
    print(f"[saved] {out_json}")

    # Save NPZ with BOTH truncated and full
    npz_path = os.path.join(out_dir, f"echo_algebra_step1_link9_su3_LR_bases_{tag}.npz")
    npz_dict = {}

    for (model, kind), (basis, _sc, _eps) in best_trunc.items():
        if len(basis) == 0:
            continue
        # STANDARD KEYS expected by v7.0 (k=8)
        npz_dict[f"basis_{kind}_{model}"] = np.stack(basis, axis=0)

    for (model, kind), (basis, _sc, _eps) in best_full.items():
        if len(basis) == 0:
            continue
        # FULL KEYS for advanced harnesses (k can be >8)
        npz_dict[f"basis_{kind}_{model}_full"] = np.stack(basis, axis=0)

    if npz_dict:
        np.savez(npz_path, **npz_dict)
        print(f"[saved] {npz_path}")
        print("  saved arrays:", ", ".join(sorted(npz_dict.keys())))
    else:
        print("[warn] no bases to save (all empty)")

    print("=" * 78)
    print("Next step:")
    print(f"  python gauss_link_npz_diagnostic_v7_0.py --npz {os.path.basename(npz_path)} --mode aligned")
    print("=" * 78)


if __name__ == "__main__":
    main()