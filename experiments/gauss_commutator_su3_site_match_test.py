#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_commutator_su3_site_match_test.py
======================================

Purpose
-------
Test whether "site/bond representation match" improves emergent gauge invariance
in the bond-only effective Hamiltonian H_eff produced by a numerical SW surrogate.

We compare:
  - d_site = 2 (qubit sites) with d_B = 3 (qutrit bonds)
  - d_site = 3 (qutrit sites) with d_B = 3 (qutrit bonds)

Under two coupling styles:
  - aligned: site generators couple to matching bond generators (first k)
  - mixed  : site generators couple to random orthogonal mixture of bond generators

Pipeline
--------
1) Build full Hamiltonian H on (sites ⊗ bonds):
     H = Δ * H0 + g * V
   where H0 penalizes site excitations away from |0>.

2) Project to site-ground manifold P = |0...0><0...0|_sites ⊗ I_bonds:
     U_eff = P exp(-i t H) P   (computed on bond basis)

3) Polar-reunitarize U_eff and take:
     H_eff = (i/t) logm(U_eff_unitary)

4) Define bond-only Gauss generators on each vertex x:
     G_x^a = (+T^a on bond_out) + (-T^a on bond_in)
   (orientation is set by plaquette ordering)

5) Compute normalized commutator norms:
     eps(x,a) = ||[G_x^a, H_eff]||_F / (||G_x^a||_F ||H_eff||_F)

Outputs
-------
- JSON summary in ./hsf_out/gauss_commutator_su3_<timestamp>.json
- Console summary

Run (Windows):
  python gauss_commutator_su3_site_match_test.py --Delta 6,8,10 --g 0.30 --t 0.10 --seed 0

Dependencies:
  numpy, scipy
"""

import os
import math
import json
import argparse
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
# Utilities
# -------------------------

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_out_dir() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def hermitize(A: np.ndarray) -> np.ndarray:
    return (A + A.conj().T) / 2.0

def hs_inner(A: np.ndarray, B: np.ndarray) -> complex:
    return np.trace(A.conj().T @ B)

def hs_norm(A: np.ndarray) -> float:
    return float(np.sqrt(max(hs_inner(A, A).real, 0.0)))

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

def eye_sp(d: int) -> csr_matrix:
    return csr_matrix(np.eye(d, dtype=complex))


# -------------------------
# Generator libraries
# -------------------------

def su_generators_gellmann(d: int):
    """
    HS-orthonormal Hermitian traceless su(d) basis.
    For d=3 this yields 8 generators.
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

    out = [normalize_hs_dense(traceless(hermitize(G))) for G in gens]
    out = gram_schmidt_hs_dense(out, tol=1e-12)
    return out

def su2_pauli_generators_hs():
    """
    HS-orthonormal su(2) generators on a qubit (3 operators).
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    # normalize to HS norm 1
    return [normalize_hs_dense(X), normalize_hs_dense(Y), normalize_hs_dense(Z)]

def su2_in_su3_site_generators_hs():
    """
    HS-orthonormal su(2) acting on |0>,|1> of a qutrit; |2> spectator.
    Useful if you want a 3-channel site algebra inside su(3).
    """
    X = np.zeros((3, 3), dtype=complex)
    Y = np.zeros((3, 3), dtype=complex)
    Z = np.zeros((3, 3), dtype=complex)
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    Y[0, 1] = -1j
    Y[1, 0] = 1j
    Z[0, 0] = 1.0
    Z[1, 1] = -1.0
    return [normalize_hs_dense(X), normalize_hs_dense(Y), normalize_hs_dense(Z)]

def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random Haar-ish orthogonal via QR on Gaussian matrix.
    """
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    # fix sign ambiguity to make it deterministic-ish
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q


# -------------------------
# Model construction
# -------------------------

@dataclass
class CaseSpec:
    name: str
    d_site: int = 3
    d_bond: int = 3
    Delta: float = 6.0
    g: float = 0.30
    t: float = 0.10
    coupling: str = "aligned"   # aligned | mixed
    seed: int = 0
    eps_logm_guard: float = 1e-12

def bond_basis_size(n_bonds: int, d_bond: int) -> int:
    return int(d_bond ** n_bonds)

def full_dim(n_sites: int, n_bonds: int, d_site: int, d_bond: int) -> int:
    return int((d_site ** n_sites) * (d_bond ** n_bonds))

def build_site_gap_H0(d_site: int, n_sites: int, n_bonds: int, d_bond: int) -> csr_matrix:
    """
    H0 = Σ_s (I - |0><0|)_s ⊗ I_bonds   (Δ applied outside).
    """
    P0 = np.zeros((d_site, d_site), dtype=complex)
    P0[0, 0] = 1.0
    Q = csr_matrix(np.eye(d_site, dtype=complex) - P0)

    dims = [d_site] * n_sites + [d_bond] * n_bonds
    dim_full = int(np.prod(dims))
    H0 = csr_matrix((dim_full, dim_full), dtype=complex)

    for s in range(n_sites):
        term = None
        for k, d in enumerate(dims):
            A = Q if k == s else eye_sp(d)
            term = A if term is None else skron(term, A, format="csr")
        H0 = H0 + term

    return H0

def build_edge_couplings(d_site: int, d_bond: int, coupling: str, rng: np.random.Generator):
    """
    Returns list of (S_site, B_bond, S_site_other_end) dense ops to sum for one edge.

    For d_bond=3, bond has 8 generators. Site has:
      - d_site=3 => 8 generators (full su(3))
      - d_site=2 => 3 generators (su(2))

    aligned:
      S_a couples to B_a for a=0..k-1
    mixed:
      S_a couples to sum_b O[b,a] B_b where O is random orthogonal in bond-gen space
      (only first k columns used), so if k=8 you span all 8 bond directions.
    """
    if d_bond != 3:
        raise ValueError("This script is specialized to d_bond=3 (SU(3) bonds).")

    bond_gens = su_generators_gellmann(3)  # 8

    if d_site == 3:
        site_gens = su_generators_gellmann(3)  # 8
    elif d_site == 2:
        site_gens = su2_pauli_generators_hs()  # 3
    else:
        raise ValueError("d_site must be 2 or 3.")

    k = len(site_gens)
    nB = len(bond_gens)

    if coupling == "aligned":
        # couple first k directions
        return [(site_gens[a], bond_gens[a], site_gens[a]) for a in range(k)]

    if coupling == "mixed":
        O = random_orthogonal(nB, rng)  # 8x8
        out = []
        for a in range(k):
            Bb = np.zeros((3, 3), dtype=complex)
            for b in range(nB):
                Bb += O[b, a] * bond_gens[b]
            Bb = normalize_hs_dense(traceless(hermitize(Bb)))
            out.append((site_gens[a], Bb, site_gens[a]))
        return out

    raise ValueError("coupling must be 'aligned' or 'mixed'.")


def build_full_H_plaquette(spec: CaseSpec) -> csr_matrix:
    """
    Single plaquette:
      sites: 0,1,2,3
      bonds: e0=(0-1), e1=(1-2), e2=(2-3), e3=(3-0)
    Tensor order: [sites..., bonds...]

    H = Δ H0 + g V
    V = Σ_edges Σ_a  (S_u^a ⊗ B_e^a ⊗ S_v^a)
    """
    n_sites = 4
    n_bonds = 4
    dims = [spec.d_site] * n_sites + [spec.d_bond] * n_bonds
    dim = int(np.prod(dims))

    H0 = build_site_gap_H0(spec.d_site, n_sites, n_bonds, spec.d_bond) * spec.Delta
    V = csr_matrix((dim, dim), dtype=complex)

    edges = [
        (0, 1, 0),  # (u, v, bond_index)
        (1, 2, 1),
        (2, 3, 2),
        (3, 0, 3),
    ]

    rng = np.random.default_rng(spec.seed)
    couplings = build_edge_couplings(spec.d_site, spec.d_bond, spec.coupling, rng)

    for (u, v, b_idx) in edges:
        iu = u
        iv = v
        ib = n_sites + b_idx

        for (Su, Bb, Sv) in couplings:
            Su_sp = csr_matrix(Su)
            Sv_sp = csr_matrix(Sv)
            Bb_sp = csr_matrix(Bb)

            term = None
            for k, d in enumerate(dims):
                if k == iu:
                    A = Su_sp
                elif k == iv:
                    A = Sv_sp
                elif k == ib:
                    A = Bb_sp
                else:
                    A = eye_sp(d)
                term = A if term is None else skron(term, A, format="csr")

            V = V + term

    return H0 + (spec.g * V)


# -------------------------
# SW surrogate: U_eff and H_eff
# -------------------------

def site_ground_projector_indices(d_site: int, n_sites: int, n_bonds: int, d_bond: int) -> np.ndarray:
    """
    Indices for |0...0>_sites ⊗ |b>_bonds for all computational bond states |b>.
    With tensor order [sites..., bonds...], these are exactly 0..(bdim-1).
    """
    bdim = bond_basis_size(n_bonds, d_bond)
    return np.arange(bdim, dtype=np.int64)

def compute_U_eff(spec: CaseSpec, H: csr_matrix) -> np.ndarray:
    """
    U_eff = P exp(-i t H) P on bond manifold by evolving each bond basis vector
    embedded in the full space.
    """
    n_sites = 4
    n_bonds = 4
    bdim = bond_basis_size(n_bonds, spec.d_bond)
    dim = full_dim(n_sites, n_bonds, spec.d_site, spec.d_bond)

    idxP = site_ground_projector_indices(spec.d_site, n_sites, n_bonds, spec.d_bond)
    A = (-1j * spec.t) * H

    Ueff = np.zeros((bdim, bdim), dtype=complex)
    for j in range(bdim):
        v = np.zeros(dim, dtype=complex)
        v[idxP[j]] = 1.0
        w = expm_multiply(A, v)
        Ueff[:, j] = w[idxP]
    return Ueff

def safe_logm(U: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, dict]:
    """
    Projection makes U_eff slightly non-unitary. Polar-reunitarize:
      W = U (U†U)^(-1/2), then L=logm(W).

    Returns (L, diagnostics).
    """
    M = U.conj().T @ U
    Mh = hermitize(M)
    evals, evecs = np.linalg.eigh(Mh)
    evals_clipped = np.maximum(evals, eps)
    Minvhalf = (evecs * (1.0 / np.sqrt(evals_clipped))) @ evecs.conj().T
    W = U @ Minvhalf

    diag = {
        "proj_nonunitarity_fro": float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]), ord="fro")),
        "min_eig_M": float(np.min(evals).real),
        "min_eig_M_clipped": float(np.min(evals_clipped).real),
        "max_eig_M": float(np.max(evals).real),
    }
    return logm(W), diag

def compute_H_eff(spec: CaseSpec, Ueff: np.ndarray) -> tuple[np.ndarray, dict]:
    L, diag = safe_logm(Ueff, eps=spec.eps_logm_guard)
    Heff = (1j / spec.t) * L
    Heff = hermitize(Heff)
    diag["Heff_herm_err_fro"] = float(np.linalg.norm(Heff - Heff.conj().T, ord="fro"))
    diag["Heff_norm_fro"] = float(np.linalg.norm(Heff, ord="fro"))
    return Heff, diag


# -------------------------
# Gauss commutator test (bond-only)
# -------------------------

def kron_dense(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)

def embed_onebond_op(op: np.ndarray, which: int, n_bonds: int, d_bond: int) -> np.ndarray:
    """
    Embed a single-bond operator into the full bond Hilbert space (dense).
    bond ordering: [e0,e1,e2,e3]
    """
    out = None
    for i in range(n_bonds):
        A = op if i == which else np.eye(d_bond, dtype=complex)
        out = A if out is None else kron_dense(out, A)
    return out

def build_gauss_generators_su3_on_bonds():
    """
    Returns 8 HS-orthonormal su(3) generators on a single bond.
    """
    return su_generators_gellmann(3)  # 8

def gauss_operators_for_plaquette_bonds():
    """
    Plaquette orientation:
      e0: 0 -> 1
      e1: 1 -> 2
      e2: 2 -> 3
      e3: 3 -> 0

    At each vertex x, use:
      G_x^a = +T^a(outgoing bond) - T^a(incoming bond)
    """
    # outgoing/incoming bond indices for each vertex
    # vertex 0: outgoing e0, incoming e3
    # vertex 1: outgoing e1, incoming e0
    # vertex 2: outgoing e2, incoming e1
    # vertex 3: outgoing e3, incoming e2
    return {
        0: (0, 3),
        1: (1, 0),
        2: (2, 1),
        3: (3, 2),
    }

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def gauss_commutator_metrics(Heff: np.ndarray, d_bond: int = 3) -> dict:
    """
    Compute normalized commutator norms for all vertices and all 8 generators.
    """
    n_bonds = 4
    bond_gens = build_gauss_generators_su3_on_bonds()  # 8
    vmap = gauss_operators_for_plaquette_bonds()

    Heff_norm = np.linalg.norm(Heff, ord="fro")
    if Heff_norm < 1e-30:
        return {"error": "Heff Frobenius norm is ~0; cannot evaluate."}

    eps_list = []
    per_vertex = {}

    for vx in range(4):
        out_b, in_b = vmap[vx]
        eps_a = []
        for a in range(8):
            T = bond_gens[a]
            G = embed_onebond_op(T, out_b, n_bonds, d_bond) - embed_onebond_op(T, in_b, n_bonds, d_bond)
            G_norm = np.linalg.norm(G, ord="fro")
            C = commutator(G, Heff)
            C_norm = np.linalg.norm(C, ord="fro")
            eps = float(C_norm / (G_norm * Heff_norm + 1e-300))
            eps_list.append(eps)
            eps_a.append(eps)
        per_vertex[str(vx)] = {
            "eps_a": eps_a,
            "eps_max": float(np.max(eps_a)),
            "eps_mean": float(np.mean(eps_a)),
        }

    eps_arr = np.array(eps_list, dtype=float)
    return {
        "eps_all": eps_list,
        "eps_max": float(np.max(eps_arr)),
        "eps_mean": float(np.mean(eps_arr)),
        "eps_median": float(np.median(eps_arr)),
        "per_vertex": per_vertex,
    }


# -------------------------
# Runner
# -------------------------

def run_one_case(spec: CaseSpec) -> dict:
    H = build_full_H_plaquette(spec)
    Ueff = compute_U_eff(spec, H)
    Heff, diag = compute_H_eff(spec, Ueff)

    gauss = gauss_commutator_metrics(Heff, d_bond=spec.d_bond)

    out = {
        "case": {
            "name": spec.name,
            "d_site": spec.d_site,
            "d_bond": spec.d_bond,
            "Delta": spec.Delta,
            "g": spec.g,
            "t": spec.t,
            "coupling": spec.coupling,
            "seed": spec.seed,
            "eps_logm_guard": spec.eps_logm_guard,
        },
        "logm_diag": diag,
        "gauss": gauss,
    }
    return out

def parse_csv_floats(s: str) -> list[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Delta", type=str, default="6.0",
                    help="Comma-separated Δ values, e.g. 6,8,10")
    ap.add_argument("--g", type=float, default=0.30)
    ap.add_argument("--t", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coupling", type=str, default="both",
                    help="aligned | mixed | both")
    ap.add_argument("--eps_logm_guard", type=float, default=1e-12)
    args = ap.parse_args()

    Deltas = parse_csv_floats(args.Delta)
    if not Deltas:
        Deltas = [6.0]

    coupling_modes = []
    if args.coupling == "both":
        coupling_modes = ["aligned", "mixed"]
    else:
        coupling_modes = [args.coupling]

    results = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
        },
        "runs": [],
    }

    for coupling in coupling_modes:
        for Delta in Deltas:
            # qubit site
            spec2 = CaseSpec(
                name=f"qubit_site_dS2_dB3_{coupling}_Delta{Delta}",
                d_site=2, d_bond=3,
                Delta=Delta, g=args.g, t=args.t,
                coupling=coupling,
                seed=args.seed,
                eps_logm_guard=args.eps_logm_guard,
            )
            # qutrit site
            spec3 = CaseSpec(
                name=f"qutrit_site_dS3_dB3_{coupling}_Delta{Delta}",
                d_site=3, d_bond=3,
                Delta=Delta, g=args.g, t=args.t,
                coupling=coupling,
                seed=args.seed,
                eps_logm_guard=args.eps_logm_guard,
            )

            r2 = run_one_case(spec2)
            r3 = run_one_case(spec3)

            results["runs"].append({
                "coupling": coupling,
                "Delta": Delta,
                "qubit_site": r2,
                "qutrit_site": r3,
                "improvement_ratio_eps_max": (
                    (r3["gauss"]["eps_max"] / r2["gauss"]["eps_max"])
                    if ("eps_max" in r2["gauss"] and r2["gauss"]["eps_max"] > 0)
                    else None
                ),
                "improvement_ratio_eps_mean": (
                    (r3["gauss"]["eps_mean"] / r2["gauss"]["eps_mean"])
                    if ("eps_mean" in r2["gauss"] and r2["gauss"]["eps_mean"] > 0)
                    else None
                ),
            })

            # Console summary (compact)
            print("------------------------------------------------------------")
            print(f"Δ={Delta:.3f}  coupling={coupling}  g={args.g:.3f}  t={args.t:.3f}")
            print(f"  qubit-site  eps_max={r2['gauss'].get('eps_max', None)}  eps_mean={r2['gauss'].get('eps_mean', None)}")
            print(f"  qutrit-site eps_max={r3['gauss'].get('eps_max', None)}  eps_mean={r3['gauss'].get('eps_mean', None)}")
            print(f"  ratio (qutrit/qubit): max={results['runs'][-1]['improvement_ratio_eps_max']}  mean={results['runs'][-1]['improvement_ratio_eps_mean']}")

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"gauss_commutator_su3_{results['meta']['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("============================================================")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
