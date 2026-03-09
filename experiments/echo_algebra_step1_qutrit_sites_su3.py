#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
echo_algebra_step1_qutrit_sites_su3.py
=====================================

Step-1 decisive diagnostic:
  Does the echo-algebra mechanism recover FULL su(3) on a d_B=3 bond
  when the *sites are also qutrits* (d_S=3), i.e. the probe matches the bond?

Coupling:
  H = sum_{a=1..8}  S_a ⊗ B_a ⊗ S_a        ("aligned" coupling)

Optionally:
  H = sum_a         S_a ⊗ B'_a ⊗ S_a       ("mixed" coupling)
with {B'_a} an orthogonal mixing of the su(3) basis (fixed by seed).

Echo extraction:
  - compute Kraus operators K_{mn} (m,n in {0,1,2})
  - choose dominant Kraus by Frobenius weight
  - map to Hermitian traceless generator from anti-Hermitian part / eps
  - SVD the span -> basis_dim
  - check Lie-closure residuals

Run:
  python echo_algebra_step1_qutrit_sites_su3.py

Outputs:
  ./hsf_out/echo_algebra_step1_qutrit_su3_<timestamp>.json
  ./hsf_out/echo_algebra_step1_qutrit_su3_best_bases_<timestamp>.npz

Dependencies:
  numpy, scipy
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


def embed_real_coords_hermitian(A: np.ndarray) -> np.ndarray:
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


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def su_generators_gellmann(d: int):
    """
    Hermitian traceless su(d) generators, HS-orthonormal: Tr(Ta Tb)=δ_ab.
    Generalized Gell-Mann construction.
    """
    gens = []

    # Off-diagonal symmetric and antisymmetric
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

    # Diagonal traceless (d-1)
    for k in range(1, d):
        D = np.zeros((d, d), dtype=complex)
        for i in range(k):
            D[i, i] = 1.0
        D[k, k] = -float(k)
        D = D * math.sqrt(2.0 / (k * (k + 1.0)))
        gens.append(D)

    out = []
    for G in gens:
        out.append(normalize_hs(traceless(hermitize(G))))

    out = gram_schmidt_hs(out, tol=1e-12)
    return out


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def haar_random_state(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / (np.linalg.norm(v) + 1e-30)


def build_H_qutrit_su3(model: str, seed: int = 999):
    """
    model:
      - "aligned": B_a = S_a basis (same basis)
      - "mixed":   B_a = sum_b R_{ab} basis_b with random orthogonal R (fixed seed)
    """
    dS = 3
    dB = 3
    S_basis = su_generators_gellmann(dS)  # 8
    B_basis = su_generators_gellmann(dB)  # 8

    rng = np.random.default_rng(seed)

    if model == "aligned":
        Bp = B_basis
        mix_R = np.eye(8)
    elif model == "mixed":
        R = random_orthogonal(8, rng)
        Bp = []
        for a in range(8):
            M = np.zeros((dB, dB), dtype=complex)
            for b in range(8):
                M += float(R[a, b]) * B_basis[b]
            Bp.append(normalize_hs(traceless(hermitize(M))))
        Bp = gram_schmidt_hs(Bp, tol=1e-12)
        mix_R = R
    else:
        raise ValueError("model must be 'aligned' or 'mixed'")

    H = np.zeros((dS * dB * dS, dS * dB * dS), dtype=complex)
    for a in range(8):
        Sa = S_basis[a]
        Ba = Bp[a]
        H += np.kron(np.kron(Sa, Ba), Sa)

    # Gentle normalization for numerical stability
    H = H / max(hs_norm(H), 1e-12)

    return H, mix_R


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


@dataclass
class SweepConfig:
    model: str
    eps: float
    n_samples: int
    seed: int = 12345
    bond_seed: int = 999
    svd_tol_rel: float = 1e-6
    gs_tol: float = 1e-10


def extract_span_basis(cfg: SweepConfig):
    dB = 3
    rng = np.random.default_rng(cfg.seed)

    H, mix_R = build_H_qutrit_su3(cfg.model, seed=cfg.bond_seed)
    U = expm(-1j * cfg.eps * H)

    pool = []
    for _ in range(cfg.n_samples):
        psiL = haar_random_state(3, rng)
        psiR = haar_random_state(3, rng)
        Ks = extract_kraus_qutrit(U, dB, psiL, psiR)

        weights = [np.linalg.norm(K, "fro")**2 for K in Ks]
        Kd = Ks[int(np.argmax(weights))]

        G = generator_from_kraus(Kd, cfg.eps)
        n = hs_norm(G)
        if n > 1e-12:
            pool.append(G / n)

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


def closure_residual_stats(basis):
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
    }


def print_result(res):
    print("-" * 78)
    print(f"model={res['model']}  eps={res['eps']:.1e}  samples={res['n_samples']}  bond_seed={res['bond_seed']}")
    print(f"  basis_dim={res['basis_dim']} (expected 8)")
    if res["sv_top5"]:
        print(f"  top singulars: {', '.join(f'{x:.3g}' for x in res['sv_top5'])}")
    cl = res["closure"]
    if cl.get("pairs", 0) > 0:
        print(f"  closure pairs={cl['pairs']}  rel_med={cl['rel_med']:.3e}  rel_max={cl['rel_max']:.3e}")
    else:
        print("  closure: (insufficient pairs / degenerate commutators)")


def main():
    out_dir = ensure_out_dir()
    print("=" * 78)
    print("STEP 1 Diagnostic — Qutrit sites probing SU(3) bond (dS=3, dB=3)")
    print("=" * 78)
    print(f"outputs: {out_dir}")

    models = ["aligned", "mixed"]
    eps_list = [1e-4, 3e-4, 1e-3]
    n_samples = 20000

    all_results = []
    best_basis_cache = {}

    for model in models:
        for eps in eps_list:
            cfg = SweepConfig(
                model=model,
                eps=eps,
                n_samples=n_samples,
                seed=12345,
                bond_seed=999,
                svd_tol_rel=1e-6,
                gs_tol=1e-10,
            )
            basis, S, dim_est, mix_R = extract_span_basis(cfg)
            res = {
                "model": model,
                "eps": eps,
                "n_samples": n_samples,
                "seed": cfg.seed,
                "bond_seed": cfg.bond_seed,
                "expected_dim": 8,
                "basis_dim": dim_est,
                "sv_top5": [float(x) for x in (S[:5] if S.size else [])],
                "closure": closure_residual_stats(basis),
                "meta": {"mix_R_det": float(np.linalg.det(mix_R))},
            }
            print_result(res)
            all_results.append(res)

            key = (model,)
            score = (res["basis_dim"], -res["closure"].get("rel_med", 1e9))
            if key not in best_basis_cache or score > best_basis_cache[key][0]:
                best_basis_cache[key] = (score, basis)

    out_json = os.path.join(out_dir, f"echo_algebra_step1_qutrit_su3_{now_tag()}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 78)
    print(f"[saved] {out_json}")

    npz_path = os.path.join(out_dir, f"echo_algebra_step1_qutrit_su3_best_bases_{now_tag()}.npz")
    npz_dict = {}
    for (model,), (_score, basis) in best_basis_cache.items():
        if not basis:
            continue
        npz_dict[f"basis_qutrit_{model}"] = np.stack(basis, axis=0)
    if npz_dict:
        np.savez(npz_path, **npz_dict)
        print(f"[saved] {npz_path}")
    else:
        print("[warn] no bases to save (all empty)")

    print("=" * 78)
    print("Interpretation guide:")
    print("  - PASS: basis_dim==8 with small closure residuals -> qubit-site ceiling confirmed.")
    print("  - If basis_dim stays 7 -> obstruction is likely in echo extraction (dominant Kraus) or deeper symmetry.")
    print("=" * 78)


if __name__ == "__main__":
    main()
