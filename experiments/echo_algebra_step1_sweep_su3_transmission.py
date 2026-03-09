#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
echo_algebra_step1_sweep_su3_transmission.py
===========================================

Step-1 (Echo algebra identification) — SU(3)-based transmission Hamiltonian.

Why this exists
---------------
Your previous Step-1 sweep used su(2) *irreps* embedded in d_B=3 (spin-1),
so the microscopic bond operators never spanned su(3). The sweep correctly
reported basis_dim < 8.

This script replaces that with a genuinely su(3)-capable bond coupling:

  H = X ⊗ Bx ⊗ X  +  Y ⊗ By ⊗ Y  +  Z ⊗ Bz ⊗ Z

where Bx, By, Bz are *generic Hermitian traceless su(3) elements* formed as
random linear combinations of the 8 Gell-Mann generators (HS-orthonormalized).

Key idea:
  Two generic elements of su(3) generate all of su(3) under commutators
  (with probability 1 in a continuous ensemble). Using three (Bx,By,Bz)
  makes that overwhelmingly likely, and fully compatible with qubit sites.

What it does
------------
For d_B in {2,3} it runs a small sweep over eps and prints:
  - extracted echo span dimension (basis_dim)
  - Lie-closure residual stats for the extracted basis
  - (for d_B=2) su(2) structure-constant sanity check

For d_B=3 it specifically tests:
  - "su3_random": Bx,By,Bz are generic su(3) elements (this is the new thing)
and optionally compares to:
  - "su2_irrep": old behavior (for reference)

Outputs
-------
Writes:
  ./hsf_out/echo_algebra_step1_su3trans_<timestamp>.json
  ./hsf_out/echo_algebra_step1_su3trans_best_bases_<timestamp>.npz
and prints a concise console report.

Run (Windows):
  python echo_algebra_step1_sweep_su3_transmission.py

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
# Site ops (qubits)
# -------------------------

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def haar_random_qubit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    return v / (np.linalg.norm(v) + 1e-30)


# -------------------------
# su(d) generator bases (generalized Gell-Mann)
# -------------------------

def su_generators_gellmann(d: int):
    """
    Hermitian traceless su(d) generators, HS-orthonormal: Tr(Ta Tb)=δ_ab.
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
        G = hermitize(G)
        G = traceless(G)
        out.append(normalize_hs(G))

    out = gram_schmidt_hs(out, tol=1e-12)
    return out


def random_su_elements(d: int, rng: np.random.Generator, n: int = 3):
    """
    Create n generic su(d) Hermitian traceless elements as random combos of su(d) basis,
    then HS-orthonormalize them.
    """
    basis = su_generators_gellmann(d)
    if len(basis) != d * d - 1:
        raise RuntimeError(f"Internal error: su({d}) basis size {len(basis)} != {d*d-1}")

    elems = []
    for _ in range(n):
        coeff = rng.normal(size=len(basis))
        M = np.zeros((d, d), dtype=complex)
        for c, T in zip(coeff, basis):
            M += float(c) * T
        M = hermitize(M)
        M = traceless(M)
        M = normalize_hs(M)
        elems.append(M)

    elems = gram_schmidt_hs(elems, tol=1e-12)
    if len(elems) < n:
        return random_su_elements(d, rng, n=n)
    return elems


# -------------------------
# Transmission Hamiltonians
# -------------------------

def su2_irrep_generators(d_B: int):
    """Old behavior: su(2) irrep embedded in d_B."""
    if d_B == 2:
        return (normalize_hs(X2.copy()), normalize_hs(Y2.copy()), normalize_hs(Z2.copy()))

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

    return (normalize_hs(Jx), normalize_hs(Jy), normalize_hs(Jz))


def transmission_hamiltonian_single_bond(d_B: int, model: str, variant: str, rng: np.random.Generator):
    """
    H on C^2 ⊗ C^{d_B} ⊗ C^2.

    model:
      - "su2_irrep": old (Bx,By,Bz) from su(2) irrep inside d_B
      - "su3_random": for d_B=3 only: (Bx,By,Bz) are generic su(3) elements

    variant:
      - "standard": X⊗Bx⊗X + Z⊗Bz⊗Z + 0.3*Y⊗I⊗Y
      - "full":     X⊗Bx⊗X + Y⊗By⊗Y + Z⊗Bz⊗Z
    """
    if model == "su2_irrep":
        Bx, By, Bz = su2_irrep_generators(d_B)
    elif model == "su3_random":
        if d_B != 3:
            raise ValueError("model 'su3_random' is intended for d_B=3")
        Bx, By, Bz = random_su_elements(3, rng, n=3)
    else:
        raise ValueError("model must be 'su2_irrep' or 'su3_random'")

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
    return H, (Bx, By, Bz)


# -------------------------
# Kraus extraction + generator pool
# -------------------------

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
    model: str
    variant: str
    eps: float
    n_samples: int
    seed: int = 0
    svd_tol_rel: float = 1e-6
    gs_tol: float = 1e-10
    bond_seed: int = 999  # fixes Bx,By,Bz for su3_random


def extract_span_basis(cfg: SweepConfig):
    rng = np.random.default_rng(cfg.seed)
    rng_bond = np.random.default_rng(cfg.bond_seed)

    H, (Bx, By, Bz) = transmission_hamiltonian_single_bond(cfg.d_B, cfg.model, cfg.variant, rng_bond)
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
        return [], np.array([]), 0, (Bx, By, Bz)

    V = np.stack([embed_real_coords_hermitian(A) for A in pool], axis=0)
    _, S, Vh = np.linalg.svd(V, full_matrices=False)

    if S.size == 0:
        return [], S, 0, (Bx, By, Bz)

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
    return basis, S, len(basis), (Bx, By, Bz)


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


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
    basis, S, dim_est, (Bx, By, Bz) = extract_span_basis(cfg)
    exp_dim = expected_dim_su(cfg.d_B)
    closure = closure_residual_stats(basis)
    su2 = {}
    if cfg.d_B == 2:
        su2 = structure_constants_su2_sanity(basis)

    def fp(M):
        return float(np.linalg.norm(M, "fro"))

    result = {
        "d_B": cfg.d_B,
        "model": cfg.model,
        "variant": cfg.variant,
        "eps": cfg.eps,
        "n_samples": cfg.n_samples,
        "seed": cfg.seed,
        "bond_seed": cfg.bond_seed,
        "expected_dim": exp_dim,
        "basis_dim": dim_est,
        "sv_top5": [float(x) for x in (S[:5] if S.size else [])],
        "closure": closure,
        "su2_sanity": su2,
        "Bx_fro": fp(Bx),
        "By_fro": fp(By),
        "Bz_fro": fp(Bz),
    }
    return result, basis


def print_result(res):
    print("-" * 78)
    print(f"d_B={res['d_B']}  model={res['model']}  variant={res['variant']}  eps={res['eps']:.1e}  samples={res['n_samples']}")
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
    print("STEP 1 — SU(3)-based Transmission Hamiltonian Sweep")
    print("=" * 78)
    print(f"outputs: {out_dir}")

    dBs = [2, 3]
    models_for = {
        2: ["su2_irrep"],
        3: ["su2_irrep", "su3_random"],
    }
    variants = ["standard", "full"]
    eps_list = [1e-4, 3e-4, 1e-3]
    n_samples_map = {2: 4000, 3: 14000}

    all_results = []
    best_basis_cache = {}

    for d_B in dBs:
        for model in models_for[d_B]:
            for variant in variants:
                for eps in eps_list:
                    cfg = SweepConfig(
                        d_B=d_B,
                        model=model,
                        variant=variant,
                        eps=eps,
                        n_samples=n_samples_map[d_B],
                        seed=12345,
                        bond_seed=999,
                        svd_tol_rel=1e-6,
                        gs_tol=1e-10,
                    )
                    res, basis = run_one(cfg)
                    print_result(res)
                    all_results.append(res)

                    key = (d_B, model, variant)
                    score = (res["basis_dim"], -res["closure"].get("rel_med", 1e9))
                    if key not in best_basis_cache or score > best_basis_cache[key][0]:
                        best_basis_cache[key] = (score, basis)

    out_json = os.path.join(out_dir, f"echo_algebra_step1_su3trans_{now_tag()}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 78)
    print(f"[saved] {out_json}")

    npz_path = os.path.join(out_dir, f"echo_algebra_step1_su3trans_best_bases_{now_tag()}.npz")
    npz_dict = {}
    for (d_B, model, variant), (_score, basis) in best_basis_cache.items():
        if not basis:
            continue
        npz_dict[f"basis_dB{d_B}_{model}_{variant}"] = np.stack(basis, axis=0)
    if npz_dict:
        np.savez(npz_path, **npz_dict)
        print(f"[saved] {npz_path}")
    else:
        print("[warn] no bases to save (all empty)")

    print("=" * 78)
    print("Interpretation guide:")
    print("  - Target: d_B=3, model=su3_random, variant=full -> basis_dim should hit 8.")
    print("  - If it hits 8 with small closure residuals, Step-1 SU(3) is confirmed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
