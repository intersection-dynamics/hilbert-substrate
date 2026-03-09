#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hybrid_sw_plaquette_interface_test.py
====================================

Hybrid-bond-dimension B1→B2 interface test (numerical SW via projected evolution).

Goal
----
Compare plaquette ("loop") terms in the bond-only effective Hamiltonian for:
  - Left plaquette: uniform d_B=2 on all 4 edges  (SU(2)-like)
  - Right plaquette: mixed d_B = [3,3,3,2] with one d_B=2 interface edge

We compute an *effective bond Hamiltonian* by:
  1) Building a full Hamiltonian H on (sites ⊗ bonds) for the plaquette subgraph
     with a site gap Δ (H0) and couplings g (V).
  2) Computing the projected evolution:
        U_eff(t) = P exp(-i t H) P   (restricted to the site-ground manifold)
     where P = |0...0><0...0|_sites ⊗ I_bonds.
  3) Defining the bond-only effective Hamiltonian:
        H_eff = (i/t) logm(U_eff)
  4) Decomposing H_eff into local operator bases and measuring the total weight
     in *weight-4* (all-bond) terms as a "loop-term" proxy.

This is an "all-orders" numerical SW surrogate (small t + large Δ).
It avoids explicitly coding 4th-order perturbation while still exposing
plaquette terms if they exist.

Design choices
--------------
- Sites are qutrits (d_S=3).
  * For d_B=2 edges: site coupling uses embedded su(2) in su(3).
  * For d_B=3 edges: site coupling uses full su(3) Gell-Mann generators.

- Bond operator bases for decomposition:
  * d=2 bonds: {I, X, Y, Z} HS-orthonormal
  * d=3 bonds: {I, 8 Gell-Mann} HS-orthonormal

Outputs
-------
- JSON: ./hsf_out/hybrid_sw_plaquette_interface_<timestamp>.json
- Console summary

Run (Windows):
  python hybrid_sw_plaquette_interface_test.py

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
    from scipy.sparse import csr_matrix
    from scipy.sparse import kron as skron
    from scipy.sparse.linalg import expm_multiply
    from scipy.linalg import logm
except Exception as e:
    raise RuntimeError("This script requires scipy (sparse + logm). Install scipy.") from e


# -------------------------
# Small helpers
# -------------------------

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_out_dir() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def hs_inner(A: np.ndarray, B: np.ndarray) -> complex:
    return np.trace(A.conj().T @ B)


def hs_norm(A: np.ndarray) -> float:
    return float(np.sqrt(max(hs_inner(A, A).real, 0.0)))


def hermitize(A: np.ndarray) -> np.ndarray:
    return (A + A.conj().T) / 2.0


def traceless(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d, dtype=complex)


def normalize_hs_dense(A: np.ndarray) -> np.ndarray:
    n = hs_norm(A)
    if n < 1e-30:
        return A.copy()
    return A / n


def gram_schmidt_hs_dense(basis, tol=1e-12):
    out = []
    for A in basis:
        B = A.copy()
        for Q in out:
            B -= hs_inner(Q, B) * Q
        n = hs_norm(B)
        if n > tol:
            out.append(B / n)
    return out


# -------------------------
# Operator libraries
# -------------------------

def su_generators_gellmann(d: int):
    """HS-orthonormal Hermitian traceless su(d) basis."""
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

    out = [normalize_hs_dense(traceless(hermitize(G))) for G in gens]
    out = gram_schmidt_hs_dense(out, tol=1e-12)
    return out


def pauli_hs_basis():
    """HS-orthonormal {I,X,Y,Z} on d=2."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I / math.sqrt(2), X / math.sqrt(2), Y / math.sqrt(2), Z / math.sqrt(2)]


def su2_in_su3_site_generators_hs():
    """HS-orthonormal su(2) acting on |0>,|1> of qutrit; |2> is spectator."""
    X = np.zeros((3, 3), dtype=complex)
    Y = np.zeros((3, 3), dtype=complex)
    Z = np.zeros((3, 3), dtype=complex)
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    Y[0, 1] = -1j
    Y[1, 0] = 1j
    Z[0, 0] = 1.0
    Z[1, 1] = -1.0
    return (normalize_hs_dense(X), normalize_hs_dense(Y), normalize_hs_dense(Z))


# -------------------------
# Sparse building blocks
# -------------------------

def eye_sp(d: int) -> csr_matrix:
    return csr_matrix(np.eye(d, dtype=complex))


def build_site_gap_H0(dS: int, n_sites: int, bond_dims: list[int]) -> csr_matrix:
    """
    H0 = Σ_s (I - |0><0|)_s ⊗ I_bonds   (Δ applied outside).
    """
    P0 = np.zeros((dS, dS), dtype=complex)
    P0[0, 0] = 1.0
    Q = csr_matrix(np.eye(dS, dtype=complex) - P0)

    dims = [dS] * n_sites + bond_dims
    dim_full = int(np.prod(dims))
    H0 = csr_matrix((dim_full, dim_full), dtype=complex)

    for s in range(n_sites):
        term = None
        for k, d in enumerate(dims):
            A = Q if k == s else eye_sp(d)
            term = A if term is None else skron(term, A, format="csr")
        H0 = H0 + term

    return H0


def build_edge_coupling_terms(dS: int, dB: int):
    """
    Returns list of (S_L, B, S_R) dense ops for an edge term.
    - dB=2: sum over 3 embedded su(2) generators aligned with Pauli X/Y/Z.
    - dB=3: sum over 8 aligned su(3) generators.
    """
    if dS != 3:
        raise ValueError("This script assumes qutrit sites dS=3.")

    if dB == 2:
        Sx, Sy, Sz = su2_in_su3_site_generators_hs()
        _, X, Y, Z = pauli_hs_basis()
        return [(Sx, X, Sx), (Sy, Y, Sy), (Sz, Z, Sz)]

    if dB == 3:
        S_basis = su_generators_gellmann(3)
        B_basis = su_generators_gellmann(3)
        return [(S_basis[a], B_basis[a], S_basis[a]) for a in range(8)]

    raise ValueError("dB must be 2 or 3 for this hybrid test.")


@dataclass
class PlaquetteSpec:
    name: str
    dS: int
    site_ids: list[int]
    bond_edges: list[tuple[int, int]]
    bond_dB: list[int]
    Delta: float = 6.0
    g: float = 0.30
    t: float = 0.10
    eps_logm_guard: float = 1e-12


def bond_basis_size(bond_dims: list[int]) -> int:
    d = 1
    for x in bond_dims:
        d *= x
    return d


def build_full_H(spec: PlaquetteSpec) -> csr_matrix:
    """
    Full H = Δ H0 + g V on (sites ⊗ bonds).
    Tensor factor order: [sites..., bonds...]
    """
    n_sites = len(spec.site_ids)
    bond_dims = spec.bond_dB

    dims = [spec.dS] * n_sites + bond_dims
    dim_full = int(np.prod(dims))

    H0 = build_site_gap_H0(spec.dS, n_sites, bond_dims) * spec.Delta
    V = csr_matrix((dim_full, dim_full), dtype=complex)

    site_to_idx = {sid: i for i, sid in enumerate(spec.site_ids)}

    for b_idx, ((u, v), dB) in enumerate(zip(spec.bond_edges, spec.bond_dB)):
        iu = site_to_idx[u]
        iv = site_to_idx[v]
        ib = n_sites + b_idx

        couplings = build_edge_coupling_terms(spec.dS, dB)
        for SL, Bb, SR in couplings:
            SLs = csr_matrix(SL)
            BBs = csr_matrix(Bb)
            SRs = csr_matrix(SR)

            term = None
            for k, d in enumerate(dims):
                if k == iu:
                    A = SLs
                elif k == iv:
                    A = SRs
                elif k == ib:
                    A = BBs
                else:
                    A = eye_sp(d)
                term = A if term is None else skron(term, A, format="csr")

            V = V + term

    return H0 + (spec.g * V)


# -------------------------
# Projected evolution and H_eff
# -------------------------

def site_ground_projector_indices(dS: int, n_sites: int, bond_dims: list[int]) -> np.ndarray:
    """
    Indices for |0...0>_sites ⊗ |b>_bonds for all computational bond states |b>.
    With tensor order [sites..., bonds...], these are exactly 0..(bdim-1).
    """
    bdim = bond_basis_size(bond_dims)
    return np.arange(bdim, dtype=np.int64)


def compute_U_eff(spec: PlaquetteSpec, H: csr_matrix) -> np.ndarray:
    """
    U_eff = P exp(-i t H) P on bond manifold by evolving each basis vector.
    """
    n_sites = len(spec.site_ids)
    bond_dims = spec.bond_dB
    bdim = bond_basis_size(bond_dims)

    dims = [spec.dS] * n_sites + bond_dims
    dim_full = int(np.prod(dims))

    idxP = site_ground_projector_indices(spec.dS, n_sites, bond_dims)
    A = (-1j * spec.t) * H

    Ueff = np.zeros((bdim, bdim), dtype=complex)
    for j in range(bdim):
        v = np.zeros(dim_full, dtype=complex)
        v[idxP[j]] = 1.0
        w = expm_multiply(A, v)
        Ueff[:, j] = w[idxP]

    return Ueff


def safe_logm(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Projection makes U_eff slightly non-unitary. We polar-reunitarize:
      W = U (U†U)^(-1/2), then take logm(W).
    """
    M = U.conj().T @ U
    evals, evecs = np.linalg.eigh(hermitize(M))
    evals = np.maximum(evals, eps)
    Minvhalf = (evecs * (1.0 / np.sqrt(evals))) @ evecs.conj().T
    W = U @ Minvhalf
    return logm(W)


def compute_H_eff_from_Ueff(spec: PlaquetteSpec, Ueff: np.ndarray) -> np.ndarray:
    L = safe_logm(Ueff, eps=spec.eps_logm_guard)
    Heff = (1j / spec.t) * L
    return hermitize(Heff)


# -------------------------
# Decomposition and loop proxy
# -------------------------

def local_basis_for_dim(d: int):
    """
    HS-orthonormal basis with identity first.
      d=2: 4 matrices
      d=3: 9 matrices
    """
    if d == 2:
        return pauli_hs_basis()
    if d == 3:
        I = np.eye(3, dtype=complex) / math.sqrt(3)
        gens = su_generators_gellmann(3)  # 8
        return [I] + gens
    raise ValueError("Only d=2 or d=3 supported.")


def weight4_loop_power(Heff: np.ndarray, bond_dims: list[int], max_terms_report: int = 12):
    """
    Total HS power in weight-4 (non-identity on all 4 bonds) terms.
    Uses tensor contractions (fast enough for 2/3 hybrid).
    """
    assert len(bond_dims) == 4

    bases = [local_basis_for_dim(d) for d in bond_dims]
    nonid_ranges = [range(1, 4) if d == 2 else range(1, 9) for d in bond_dims]

    d1, d2, d3, d4 = bond_dims
    Ht = Heff.reshape(d1, d2, d3, d4, d1, d2, d3, d4)

    total_power = float(np.trace(Heff.conj().T @ Heff).real)

    loop_terms = []
    loop_power = 0.0

    for i1 in nonid_ranges[0]:
        B1 = bases[0][i1]
        for i2 in nonid_ranges[1]:
            B2 = bases[1][i2]
            for i3 in nonid_ranges[2]:
                B3 = bases[2][i3]
                for i4 in nonid_ranges[3]:
                    B4 = bases[3][i4]

                    # c = Tr((B1⊗B2⊗B3⊗B4)† Heff)
                    T = Ht
                    T = np.tensordot(np.conj(B1), T, axes=([0, 1], [0, 4]))
                    T = np.tensordot(np.conj(B2), T, axes=([0, 1], [0, 3]))
                    T = np.tensordot(np.conj(B3), T, axes=([0, 1], [0, 2]))
                    c = np.tensordot(np.conj(B4), T, axes=([0, 1], [0, 1]))

                    c = complex(c)
                    p = abs(c) ** 2
                    loop_power += p
                    if p > 1e-10:
                        loop_terms.append((p, (i1, i2, i3, i4), c))

    loop_terms.sort(key=lambda x: x[0], reverse=True)
    top = [
        {
            "power": float(p),
            "indices": list(idx),
            "coeff_re": float(c.real),
            "coeff_im": float(c.imag),
        }
        for (p, idx, c) in loop_terms[:max_terms_report]
    ]

    return {
        "loop_power": float(loop_power),
        "total_power": float(total_power),
        "loop_fraction": float(loop_power / (total_power + 1e-30)),
        "top_terms": top,
        "count_terms": int(len(loop_terms)),
    }


def summarize_edge_dims(bond_dims: list[int]) -> str:
    return "[" + ",".join(str(d) for d in bond_dims) + "]"


# -------------------------
# Main
# -------------------------

def main():
    out_dir = ensure_out_dir()
    tag = now_tag()

    # You can sweep these if needed.
    Delta = 6.0
    g = 0.30
    t = 0.10

    # Left plaquette: sites {0,1,3,4}, all dB=2
    left = PlaquetteSpec(
        name="left_uniform_SU2",
        dS=3,
        site_ids=[0, 1, 3, 4],
        bond_edges=[(0, 1), (1, 4), (3, 4), (0, 3)],
        bond_dB=[2, 2, 2, 2],
        Delta=Delta,
        g=g,
        t=t,
    )

    # Right plaquette: sites {1,2,4,5}, bonds [3,3,3,2] with interface edge (1,4)
    right = PlaquetteSpec(
        name="right_mixed_SU3_SU2",
        dS=3,
        site_ids=[1, 2, 4, 5],
        bond_edges=[(1, 2), (2, 5), (4, 5), (1, 4)],
        bond_dB=[3, 3, 3, 2],
        Delta=Delta,
        g=g,
        t=t,
    )

    specs = [left, right]

    print("=" * 78)
    print("HYBRID SW PLAQUETTE INTERFACE TEST")
    print("=" * 78)
    print(f"Controls: dS=3, Δ={Delta}, g={g}, t={t}")
    print("Method: U_eff = P exp(-i t H) P, H_eff = (i/t) logm(U_eff)")
    print(f"outputs: {out_dir}")
    print("-" * 78)

    results = {
        "tag": tag,
        "controls": {"dS": 3, "Delta": Delta, "g": g, "t": t},
        "plaquettes": [],
    }

    for spec in specs:
        print(f"[{spec.name}] sites={spec.site_ids} bonds={spec.bond_edges} dB={summarize_edge_dims(spec.bond_dB)}")

        H = build_full_H(spec)
        Ueff = compute_U_eff(spec, H)
        Heff = compute_H_eff_from_Ueff(spec, Ueff)
        loop = weight4_loop_power(Heff, spec.bond_dB, max_terms_report=10)

        nonunit = float(np.linalg.norm(Ueff.conj().T @ Ueff - np.eye(Ueff.shape[0]), ord="fro"))
        herm_err = float(np.linalg.norm(Heff - Heff.conj().T, ord="fro"))

        rep = {
            "name": spec.name,
            "site_ids": spec.site_ids,
            "bond_edges": [list(e) for e in spec.bond_edges],
            "bond_dB": spec.bond_dB,
            "dim_bond": int(bond_basis_size(spec.bond_dB)),
            "Ueff_nonunit_fro": nonunit,
            "Heff_herm_err_fro": herm_err,
            "loop_proxy": loop,
        }
        results["plaquettes"].append(rep)

        print(f"  dim(bond)={rep['dim_bond']}  ||U†U-I||_F={nonunit:.3e}  herm_err={herm_err:.3e}")
        print(f"  loop_fraction(weight4)={loop['loop_fraction']:.6g}  loop_power={loop['loop_power']:.6g}  total={loop['total_power']:.6g}")

        if loop["top_terms"]:
            print("  top weight-4 terms (power, indices):")
            for k, tt in enumerate(loop["top_terms"][:5]):
                print(f"    {k+1:2d}. p={tt['power']:.3e} idx={tt['indices']} c=({tt['coeff_re']:.3e}+{tt['coeff_im']:.3e}i)")
        print("-" * 78)

    lf = results["plaquettes"][0]["loop_proxy"]["loop_fraction"]
    rf = results["plaquettes"][1]["loop_proxy"]["loop_fraction"]
    ratio = float(rf / (lf + 1e-30))
    results["interface_summary"] = {
        "left_loop_fraction": float(lf),
        "right_loop_fraction": float(rf),
        "right_over_left": ratio,
        "interpretation": (
            "If right_over_left << 1, the mixed plaquette suppresses all-bond (loop-like) terms relative to the uniform SU(2) plaquette. "
            "If comparable, loop structure survives the interface edge in this SW-projected evolution."
        ),
    }

    out_json = os.path.join(out_dir, f"hybrid_sw_plaquette_interface_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("INTERFACE SUMMARY:")
    print(f"  left loop_fraction  = {lf:.6g}")
    print(f"  right loop_fraction = {rf:.6g}")
    print(f"  right/left          = {ratio:.6g}")
    print(f"[saved] {out_json}")
    print("=" * 78)
    print("Notes:")
    print("  - Increase Δ or reduce g to push deeper into the SW regime (cleaner separation).")
    print("  - Reduce t if Ueff is far from unitary after projection; t ~ 0.05 can help.")
    print("=" * 78)


if __name__ == "__main__":
    main()
