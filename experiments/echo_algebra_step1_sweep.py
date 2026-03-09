#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
echo_algebra_step1_sweep.py
==========================

Step-1 validation for HSF / Echo Algebra claim:

Goal:
  Verify that the bond "echo algebra" extracted from the microscopic single-bond
  echo mechanism has:
    (1) the correct *dimension*  dim = d_B^2 - 1
    (2) *Lie closure* under commutators (within numerical tolerance)
    (3) (optional) su(2) structure constant sanity checks

This script avoids the failure mode of overly strict gating rejecting all samples.
Instead it estimates the *linear span* of effective Hermitian traceless generators
derived from dominant Kraus operators across random boundary site states.

Default sweep:
  d_B in {2,3}
  variant in {"standard","full"}
  eps in {1e-4, 3e-4, 1e-3}

Outputs:
  ./hsf_out/echo_algebra_step1_sweep_<timestamp>.json
  ./hsf_out/echo_algebra_step1_best_bases_<timestamp>.npz

Run (Windows):
  python echo_algebra_step1_sweep.py
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


# -------------------------
# Model: site ⊗ bond ⊗ site
# -------------------------

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def su2_irrep_generators(d_B: int):
    if d_B == 2:
        return (X2.copy(), Y2.copy(), Z2.copy())

    s = (d_B - 1) / 2.0
    Jx = np.zeros((d_B, d_B), dtype=complex)
    Jy = np.zeros((d_B, d_B), dtype=complex)
    Jz = np.zeros((d_B, d_B), dtype=complex)

    for m_idx in range(d_B):
        m = s - m_idx
        Jz[m_idx, m_idx] = m
        if m_idx + 1 < d_B:
            mp = m - 1
            coeff = np.sqrt(s * (s + 1) - m * mp) * 0.5
            Jx[m_idx, m_idx + 1] = coeff
            Jx[m_idx + 1, m_idx] = coeff
            Jy[m_idx, m_idx + 1] = -1j * coeff
            Jy[m_idx + 1, m_idx] = 1j * coeff

    return (Jx, Jy, Jz)


def transmission_hamiltonian_single_bond(d_B: int, variant: str):
    Bx, By, Bz = su2_irrep_generators(d_B)
    if variant == "standard":
        H = (np.kron(np.kron(X2, Bx), X2) +
             np.kron(np.kron(Z2, Bz), Z2) +
             0.3 * np.kron(np.kron(Y2, np.eye(d_B, dtype=complex)), Y2))
    elif variant == "full":
        H = (np.kron(np.kron(X2, Bx), X2) +
             np.kron(np.kron(Y2, By), Y2) +
             np.kron(np.kron(Z2, Bz), Z2))
    else:
        raise ValueError("variant must be 'standard' or 'full'")
    return H


def haar_random_qubit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    return v / (np.linalg.norm(v) + 1e-30)


def extract_kraus(U_full: np.ndarray, d_B: int, psi_left: np.ndarray, psi_right: np.ndarray):
    d_full = 2 * d_B * 2
    embed = np.zeros((d_full, d_B), dtype=complex)
    for b in range(d_B):
        for a in range(2):
            for c in range(2):
                row = a * (d_B * 2) + b * 2 + c
                embed[row, b] = psi_left[a] * psi_right[c]

    U_embed = U_full @ embed

    kraus_ops = []
    for m in range(2):
        for n in range(2):
            K = np.zeros((d_B, d_B), dtype=complex)
            for b_out in range(d_B):
                row = m * (d_B * 2) + b_out * 2 + n
                for b_in in range(d_B):
                    K[b_out, b_in] = U_embed[row, b_in]
            kraus_ops.append(K)
    return kraus_ops


def generator_from_kraus(K: np.ndarray, eps: float) -> np.ndarray:
    H_eff = (K - K.conj().T) / (2.0j * eps)
    H_eff = hermitize(H_eff)
    H_eff = traceless(H_eff)
    return H_eff


# -------------------------
# Span extraction + closure tests
# -------------------------

@dataclass
class SweepConfig:
    d_B: int
    variant: str
    eps: float
    n_samples: int
    seed: int = 0
    svd_tol_rel: float = 1e-6
    gs_tol: float = 1e-10


def extract_span_basis(cfg: SweepConfig):
    rng = np.random.default_rng(cfg.seed)

    H = transmission_hamiltonian_single_bond(cfg.d_B, cfg.variant)
    U = expm(-1j * cfg.eps * H)

    pool = []
    for _ in range(cfg.n_samples):
        psiL = haar_random_qubit(rng)
        psiR = haar_random_qubit(rng)
        Ks = extract_kraus(U, cfg.d_B, psiL, psiR)

        weights = [np.linalg.norm(K, "fro")**2 for K in Ks]
        Kd = Ks[int(np.argmax(weights))]

        G = generator_from_kraus(Kd, cfg.eps)
        n = hs_norm(G)
        if n > 1e-12:
            pool.append(G / n)

    if not pool:
        return [], np.array([]), 0

    V = np.stack([embed_real_coords_hermitian(A) for A in pool], axis=0)
    _, S, Vh = np.linalg.svd(V, full_matrices=False)

    if S.size == 0:
        return [], S, 0

    thresh = cfg.svd_tol_rel * S[0]
    dim_est = int(np.sum(S > thresh))
    basis_vecs = Vh[:dim_est, :]

    basis = []
    for v in basis_vecs:
        A = reconstruct_from_real_coords(v, cfg.d_B)
        A = hermitize(A)
        A = traceless(A)
        A = normalize_hs(A)
        if hs_norm(A) > 1e-12:
            basis.append(A)

    basis = gram_schmidt_hs(basis, tol=cfg.gs_tol)
    return basis, S, len(basis)


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def closure_residual_stats(basis):
    if len(basis) == 0:
        return {"pairs": 0}

    rel_resids = []
    abs_resids = []
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
            abs_resids.append(rn)

    if not rel_resids:
        return {"pairs": 0}

    rel = np.array(rel_resids, dtype=float)
    ab = np.array(abs_resids, dtype=float)
    return {
        "pairs": int(rel.size),
        "rel_min": float(rel.min()),
        "rel_med": float(np.median(rel)),
        "rel_max": float(rel.max()),
        "abs_med": float(np.median(ab)),
    }


def structure_constants_su2_sanity(basis):
    if len(basis) != 3:
        return {"ok": False}

    f = np.zeros((3, 3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                val = -1j * np.trace(basis[a].conj().T @ commutator(basis[b], basis[c]))
                f[a, b, c] = float(val.real)

    anti_err = np.max(np.abs(f + np.transpose(f, (0, 2, 1))))
    fnorm = float(np.linalg.norm(f))
    return {"ok": True, "antisym_err_bc": float(anti_err), "f_fro_norm": fnorm}


def expected_dim_su(d_B: int) -> int:
    return d_B * d_B - 1


def run_one(cfg: SweepConfig):
    basis, S, dim_est = extract_span_basis(cfg)
    exp_dim = expected_dim_su(cfg.d_B)
    closure = closure_residual_stats(basis)
    su2 = {}
    if cfg.d_B == 2:
        su2 = structure_constants_su2_sanity(basis)

    result = {
        "d_B": cfg.d_B,
        "variant": cfg.variant,
        "eps": cfg.eps,
        "n_samples": cfg.n_samples,
        "seed": cfg.seed,
        "expected_dim": exp_dim,
        "basis_dim": dim_est,
        "sv_top5": [float(x) for x in (S[:5] if S.size else [])],
        "closure": closure,
        "su2_sanity": su2,
    }
    return result, basis


def print_result(res):
    print("-" * 78)
    print(f"d_B={res['d_B']}  variant={res['variant']}  eps={res['eps']:.1e}  samples={res['n_samples']}")
    print(f"  basis_dim={res['basis_dim']} (expected {res['expected_dim']})")
    if res["sv_top5"]:
        print(f"  top singulars: {', '.join(f'{x:.3g}' for x in res['sv_top5'])}")
    cl = res["closure"]
    if cl.get("pairs", 0) > 0:
        print(f"  closure pairs={cl['pairs']}  rel_med={cl['rel_med']:.3e}  rel_max={cl['rel_max']:.3e}")
    else:
        print("  closure: (insufficient pairs / degenerate commutators)")
    if res["d_B"] == 2 and res["su2_sanity"].get("ok", False):
        s2 = res["su2_sanity"]
        print(f"  su2 sanity: antisym_err_bc={s2['antisym_err_bc']:.3e}  ||f||_F={s2['f_fro_norm']:.3g}")


def main():
    out_dir = ensure_out_dir()
    print("=" * 78)
    print("STEP 1 — Echo Algebra Identification Sweep")
    print("=" * 78)
    print(f"outputs: {out_dir}")

    dBs = [2, 3]
    variants = ["standard", "full"]
    eps_list = [1e-4, 3e-4, 1e-3]
    n_samples_map = {2: 4000, 3: 12000}

    all_results = []
    best_basis_cache = {}

    for d_B in dBs:
        for variant in variants:
            for eps in eps_list:
                cfg = SweepConfig(
                    d_B=d_B,
                    variant=variant,
                    eps=eps,
                    n_samples=n_samples_map[d_B],
                    seed=12345,
                    svd_tol_rel=1e-6,
                    gs_tol=1e-10,
                )
                res, basis = run_one(cfg)
                print_result(res)
                all_results.append(res)

                key = (d_B, variant)
                score = (res["basis_dim"], -res["closure"].get("rel_med", 1e9))
                if key not in best_basis_cache or score > best_basis_cache[key][0]:
                    best_basis_cache[key] = (score, basis)

    out_json = os.path.join(out_dir, f"echo_algebra_step1_sweep_{now_tag()}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 78)
    print(f"[saved] {out_json}")

    npz_path = os.path.join(out_dir, f"echo_algebra_step1_best_bases_{now_tag()}.npz")
    npz_dict = {}
    for (d_B, variant), (_score, basis) in best_basis_cache.items():
        if not basis:
            continue
        npz_dict[f"basis_dB{d_B}_{variant}"] = np.stack(basis, axis=0)
    if npz_dict:
        np.savez(npz_path, **npz_dict)
        print(f"[saved] {npz_path}")
    else:
        print("[warn] no bases to save (all empty)")

    print("=" * 78)
    print("Interpretation guide:")
    print("  - basis_dim == d_B^2 - 1 is the Step-1 pass condition.")
    print("  - closure rel_med/rel_max should be small (<< 1) if it's a Lie algebra.")
    print("  - If 'standard' fails but 'full' passes, anisotropy was truncating the algebra.")
    print("  - For d_B=3, you want basis_dim=8; if you get 3 or 5, you're seeing a subalgebra.")
    print("=" * 78)


if __name__ == "__main__":
    main()
