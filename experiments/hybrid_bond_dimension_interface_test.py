#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hybrid_bond_dimension_interface_test.py
======================================

Minimal "hybrid bond dimension" interface test for HSF echo algebra.

Builds a tiny 2-plaquette lattice with mixed bond Hilbert dimensions (d_B),
runs a local (single-bond) echo-algebra extraction on each edge, and summarizes
domain vs interface behavior.

Graph (two squares side-by-side):
   (0)---(1)---(2)
    |     |     |
   (3)---(4)---(5)

Left plaquette edges:  (0,1), (1,4), (3,4), (0,3)
Right plaquette edges: (1,2), (2,5), (4,5), (1,4)   [shares (1,4)]

Bond dimension assignment:
  - Left-region edges: d_B=2
  - Right-region edges: d_B=3
  - Shared edge (1,4): d_B=2   -> right plaquette is mixed (interface)

Key design choice:
  Sites are qutrits (d_S=3) globally, so a single site representation can couple
  to both d_B=2 and d_B=3 edges.
  - For d_B=2 edges: use an su(2) subalgebra embedded in su(3) on qutrit sites.
  - For d_B=3 edges: use full su(3) basis on sites and bonds.

Run (Windows):
  python hybrid_bond_dimension_interface_test.py

Outputs:
  ./hsf_out/hybrid_bond_interface_<timestamp>.json

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


def expected_dim_su(d: int) -> int:
    return d * d - 1


def haar_random_state(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / (np.linalg.norm(v) + 1e-30)


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

    out = [normalize_hs(traceless(hermitize(G))) for G in gens]
    out = gram_schmidt_hs(out, tol=1e-12)
    return out


def su2_in_su3_site_generators():
    X = np.zeros((3, 3), dtype=complex)
    Y = np.zeros((3, 3), dtype=complex)
    Z = np.zeros((3, 3), dtype=complex)
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    Y[0, 1] = -1j
    Y[1, 0] = 1j
    Z[0, 0] = 1.0
    Z[1, 1] = -1.0
    return (normalize_hs(X), normalize_hs(Y), normalize_hs(Z))


def H_single_bond(dS: int, dB: int, model: str):
    assert dS == 3, "This script assumes qutrit sites (dS=3)."

    if model == "edge_su2":
        assert dB == 2
        Sx, Sy, Sz = su2_in_su3_site_generators()

        X2 = np.array([[0, 1], [1, 0]], dtype=complex)
        Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
        Bx, By, Bz = normalize_hs(X2), normalize_hs(Y2), normalize_hs(Z2)

        H = (np.kron(np.kron(Sx, Bx), Sx) +
             np.kron(np.kron(Sy, By), Sy) +
             np.kron(np.kron(Sz, Bz), Sz))

    elif model == "edge_su3":
        assert dB == 3
        S_basis = su_generators_gellmann(3)  # 8
        B_basis = su_generators_gellmann(3)  # 8
        H = np.zeros((dS * dB * dS, dS * dB * dS), dtype=complex)
        for a in range(8):
            H += np.kron(np.kron(S_basis[a], B_basis[a]), S_basis[a])

    else:
        raise ValueError("model must be 'edge_su2' or 'edge_su3'")

    H = H / max(hs_norm(H), 1e-12)
    return H


def extract_kraus(U_full: np.ndarray, dS: int, dB: int, psiL: np.ndarray, psiR: np.ndarray):
    d_full = dS * dB * dS
    embed = np.zeros((d_full, dB), dtype=complex)

    for b in range(dB):
        for a in range(dS):
            for c in range(dS):
                row = a * (dB * dS) + b * dS + c
                embed[row, b] = psiL[a] * psiR[c]

    U_embed = U_full @ embed

    Ks = []
    for m in range(dS):
        for n in range(dS):
            K = np.zeros((dB, dB), dtype=complex)
            for b_out in range(dB):
                row = m * (dB * dS) + b_out * dS + n
                for b_in in range(dB):
                    K[b_out, b_in] = U_embed[row, b_in]
            Ks.append(K)
    return Ks


def generator_from_kraus(K: np.ndarray, eps: float) -> np.ndarray:
    H_eff = (K - K.conj().T) / (2.0j * eps)
    H_eff = hermitize(H_eff)
    H_eff = traceless(H_eff)
    return H_eff


@dataclass
class EdgeEchoCfg:
    dS: int
    dB: int
    model: str
    eps: float
    n_samples: int
    seed: int = 12345
    svd_tol_rel: float = 1e-6
    gs_tol: float = 1e-10


def extract_echo_span_basis(cfg: EdgeEchoCfg):
    rng = np.random.default_rng(cfg.seed)

    H = H_single_bond(cfg.dS, cfg.dB, cfg.model)
    U = expm(-1j * cfg.eps * H)

    pool = []
    for _ in range(cfg.n_samples):
        psiL = haar_random_state(cfg.dS, rng)
        psiR = haar_random_state(cfg.dS, rng)
        Ks = extract_kraus(U, cfg.dS, cfg.dB, psiL, psiR)

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
        A = reconstruct_from_real_coords(v, cfg.dB)
        A = hermitize(A)
        A = traceless(A)
        A = normalize_hs(A)
        if hs_norm(A) > 1e-12:
            basis.append(A)

    basis = gram_schmidt_hs(basis, tol=cfg.gs_tol)
    return basis, S, len(basis)


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


def build_two_plaquette_graph():
    sites = list(range(6))
    edges = [
        (0, 1), (1, 2),
        (3, 4), (4, 5),
        (0, 3), (1, 4), (2, 5),
    ]
    edges = [tuple(sorted(e)) for e in edges]

    left = [tuple(sorted(e)) for e in [(0, 1), (1, 4), (3, 4), (0, 3)]]
    right = [tuple(sorted(e)) for e in [(1, 2), (2, 5), (4, 5), (1, 4)]]

    plaquettes = {"left": left, "right": right}
    return sites, edges, plaquettes


def assign_bond_dimensions(edges):
    left_edges = {(0, 1), (0, 3), (3, 4)}
    shared = (1, 4)

    dB = {}
    for e in edges:
        if e in left_edges or e == shared:
            dB[e] = 2
        else:
            dB[e] = 3
    return dB


def main():
    out_dir = ensure_out_dir()
    tag = now_tag()

    dS = 3
    eps = 3e-4
    n_samples = 12000
    seed = 12345

    sites, edges, plaquettes = build_two_plaquette_graph()
    dB_map = assign_bond_dimensions(edges)

    print("=" * 78)
    print("HYBRID BOND DIMENSION INTERFACE TEST")
    print("=" * 78)
    print(f"Graph: 2 plaquettes (left, right), 6 sites, 7 edges")
    print(f"Sites: dS={dS} globally")
    print(f"Echo extraction: eps={eps}  samples/edge={n_samples}")
    print(f"outputs: {out_dir}")
    print("-" * 78)

    edge_reports = []
    for e in edges:
        dB = dB_map[e]
        model = "edge_su2" if dB == 2 else "edge_su3"

        cfg = EdgeEchoCfg(
            dS=dS,
            dB=dB,
            model=model,
            eps=eps,
            n_samples=n_samples,
            seed=seed,
            svd_tol_rel=1e-6,
            gs_tol=1e-10,
        )
        basis, S, dim_est = extract_echo_span_basis(cfg)
        rep = {
            "edge": list(e),
            "dB": dB,
            "model": model,
            "expected_dim": expected_dim_su(dB),
            "basis_dim": dim_est,
            "sv_top5": [float(x) for x in (S[:5] if S.size else [])],
            "closure": closure_residual_stats(basis),
        }
        edge_reports.append(rep)

        print(f"edge {e}  dB={dB}  -> basis_dim={dim_est} (exp {rep['expected_dim']})")

    plaq_reports = []
    for name, pl_edges in plaquettes.items():
        dims = [dB_map[tuple(sorted(e))] for e in pl_edges]
        uniform = (len(set(dims)) == 1)
        cap = 1
        for d in dims:
            cap *= expected_dim_su(d)
        plaq_reports.append({
            "plaquette": name,
            "edges": [list(tuple(sorted(e))) for e in pl_edges],
            "dB_list": dims,
            "uniform_dB": uniform,
            "loop_capacity_proxy": int(cap),
        })

    report = {
        "tag": tag,
        "config": {"dS": dS, "eps": eps, "samples_per_edge": n_samples, "seed": seed},
        "graph": {"sites": sites, "edges": [list(e) for e in edges], "plaquettes": plaq_reports},
        "edge_echo": edge_reports,
        "interface": {
            "shared_edge": [1, 4],
            "comment": (
                "Right plaquette is mixed because edge (1,4) is dB=2. "
                "This file is a first-pass sanity check that local echo algebras "
                "still look like su(2) on dB=2 edges and su(3) on dB=3 edges with qutrit sites."
            ),
        },
    }

    out_json = os.path.join(out_dir, f"hybrid_bond_interface_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("-" * 78)
    print("PLAQUETTES:")
    for p in plaq_reports:
        print(f"  {p['plaquette']}: dB={p['dB_list']}  uniform={p['uniform_dB']}  capacity~{p['loop_capacity_proxy']}")
    print("-" * 78)
    print(f"[saved] {out_json}")
    print("=" * 78)


if __name__ == "__main__":
    main()
