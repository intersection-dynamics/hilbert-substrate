#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_commutator_su3_v5_echo_LR_basis_compare.py
===============================================

What v5 adds vs v4
------------------
v4 already did the *right* doubled-link Gauss test (commuting L/R via operator-space link),
and compared commutators on:
  - U_super = W ⊗ W*
  - H_super = Heff ⊗ I - I ⊗ Heff^T

But v4 only used ONE SU(3) generator basis at a time (either Gell-Mann or NPZ).

v5 runs BOTH bases side-by-side in ONE sweep and prints / saves:

  - epsU_mean, epsU_max  for generator_source = gellmann
  - epsU_mean, epsU_max  for generator_source = npz (echo basis_both_<model>)
  - same for epsH_mean, epsH_max

This directly answers: "does using the echo-derived SU(3) basis (from LR step) improve Gauss commutators,
especially in the mixed coupling case?"

Important technical note
------------------------
We intentionally DO NOT use basis_left / basis_right from the LR extractor because those do NOT commute
when interpreted as operators on a single 3D bond Hilbert space (your cross-comm stats were O(1)).
Instead we do the *correct* thing:
  - choose a su(3) basis {T^a} on C^3 (either canonical or echo-derived)
  - define exact commuting link-end actions on operator-space link C^3 ⊗ C^3 (dim 9):
      L^a = T^a ⊗ I
      R^a = I ⊗ (T^a)^T
This is the minimal faithful representation where left/right commute.

Run (Windows, one line)
-----------------------
python gauss_commutator_su3_v5_echo_LR_basis_compare.py --Delta 6,8,10,14 --g 0.30 --t 0.10 --seed 0 --coupling both --npz hsf_out\\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz

If you omit --npz, the script will still run, but the "npz" generator_source will be skipped.

Outputs
-------
./hsf_out/gauss_commutator_su3_v5_<timestamp>.json

Dependencies: numpy, scipy
"""

import os
import math
import json
import argparse
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import kron as skron
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import logm


# -------------------------
# Utilities
# -------------------------

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_out_dir() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def hermitize_dense(A: np.ndarray) -> np.ndarray:
    return (A + A.conj().T) / 2.0

def hs_inner(A: np.ndarray, B: np.ndarray) -> complex:
    return np.trace(A.conj().T @ B)

def hs_norm_dense(A: np.ndarray) -> float:
    return float(np.sqrt(max(hs_inner(A, A).real, 0.0)))

def traceless(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d, dtype=complex)

def normalize_hs_dense(A: np.ndarray) -> np.ndarray:
    n = hs_norm_dense(A)
    if n < 1e-30:
        return A.copy()
    return A / n

def gram_schmidt_hs_dense(basis, tol=1e-12):
    out = []
    for A in basis:
        B = A.copy()
        for Q in out:
            B -= hs_inner(Q, B) * Q
        n = hs_norm_dense(B)
        if n > tol:
            out.append(B / n)
    return out

def eye_sp(d: int) -> csr_matrix:
    return csr_matrix(np.eye(d, dtype=complex))

def fro_norm_sp(A: csr_matrix) -> float:
    if A.nnz == 0:
        return 0.0
    return float(np.sqrt(np.sum(np.abs(A.data) ** 2)))

def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q


# -------------------------
# su(3) generators
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

    out = [normalize_hs_dense(traceless(hermitize_dense(G))) for G in gens]
    out = gram_schmidt_hs_dense(out, tol=1e-12)
    return out

def load_su3_basis_from_npz(npz_path: str, model: str):
    """
    Loads basis_both_<model> from NPZ produced by echo_algebra_step1_qutrit_su3_LR script.
    Expect shape (8,3,3) (or >=8).
    """
    data = np.load(npz_path)
    key = f"basis_both_{model}"
    if key not in data:
        raise KeyError(f"NPZ missing key '{key}'. Available keys (first 30): {list(data.keys())[:30]}")
    arr = data[key]
    if arr.shape[0] < 8 or arr.shape[1:] != (3, 3):
        raise ValueError(f"{key} must have shape (>=8,3,3), got {arr.shape}")

    basis = [arr[i].astype(complex) for i in range(8)]
    basis = [normalize_hs_dense(traceless(hermitize_dense(B))) for B in basis]
    basis = gram_schmidt_hs_dense(basis, tol=1e-12)
    if len(basis) < 8:
        raise RuntimeError(f"After orthonormalization got {len(basis)} < 8 generators from NPZ.")
    return basis[:8]


# -------------------------
# Geometry (plaquette)
# -------------------------

N_SITES = 4
N_BONDS = 4

EDGES = [
    (0, 1, 0),
    (1, 2, 1),
    (2, 3, 2),
    (3, 0, 3),
]

VERTEX_OUT_IN = {
    0: (0, 3),
    1: (1, 0),
    2: (2, 1),
    3: (3, 2),
}


# -------------------------
# Microscopic model
# -------------------------

@dataclass
class CaseSpec:
    name: str
    d_site: int = 3
    d_bond: int = 3
    Delta: float = 6.0
    g: float = 0.30
    t: float = 0.10
    coupling: str = "aligned"
    seed: int = 0
    eps_logm_guard: float = 1e-12

def build_edge_couplings(d_site: int, d_bond: int, coupling: str, rng: np.random.Generator):
    if d_bond != 3:
        raise ValueError("v5 assumes d_bond=3 (SU(3) bonds).")

    bond_gens = su_generators_gellmann(3)  # 8

    if d_site == 3:
        site_gens = su_generators_gellmann(3)  # 8
    elif d_site == 2:
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        site_gens = [normalize_hs_dense(X), normalize_hs_dense(Y), normalize_hs_dense(Z)]
    else:
        raise ValueError("d_site must be 2 or 3.")

    k = len(site_gens)
    nB = len(bond_gens)

    if coupling == "aligned":
        return [(site_gens[a], bond_gens[a], site_gens[a]) for a in range(k)]

    if coupling == "mixed":
        O = random_orthogonal(nB, rng)
        out = []
        for a in range(k):
            Bb = np.zeros((3, 3), dtype=complex)
            for b in range(nB):
                Bb += O[b, a] * bond_gens[b]
            Bb = normalize_hs_dense(traceless(hermitize_dense(Bb)))
            out.append((site_gens[a], Bb, site_gens[a]))
        return out

    raise ValueError("coupling must be 'aligned' or 'mixed'.")

def build_site_gap_H0(d_site: int, d_bond: int) -> csr_matrix:
    P0 = np.zeros((d_site, d_site), dtype=complex)
    P0[0, 0] = 1.0
    Q = csr_matrix(np.eye(d_site, dtype=complex) - P0)

    dims = [d_site] * N_SITES + [d_bond] * N_BONDS
    dim_full = int(np.prod(dims))
    H0 = csr_matrix((dim_full, dim_full), dtype=complex)

    for s in range(N_SITES):
        term = None
        for k, d in enumerate(dims):
            A = Q if k == s else eye_sp(d)
            term = A if term is None else skron(term, A, format="csr")
        H0 = H0 + term

    return H0

def build_full_H_plaquette(spec: CaseSpec) -> csr_matrix:
    dims = [spec.d_site] * N_SITES + [spec.d_bond] * N_BONDS
    dim = int(np.prod(dims))

    H0 = build_site_gap_H0(spec.d_site, spec.d_bond) * spec.Delta
    V = csr_matrix((dim, dim), dtype=complex)

    rng = np.random.default_rng(spec.seed)
    couplings = build_edge_couplings(spec.d_site, spec.d_bond, spec.coupling, rng)

    for (u, v, b_idx) in EDGES:
        iu = u
        iv = v
        ib = N_SITES + b_idx

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
# Bond-only effective W and Heff
# -------------------------

def site_ground_projector_indices(d_site: int, d_bond: int) -> np.ndarray:
    bdim = int(d_bond ** N_BONDS)  # 81
    return np.arange(bdim, dtype=np.int64)

def compute_Ueff_bonds(spec: CaseSpec, H: csr_matrix) -> np.ndarray:
    d_site = spec.d_site
    d_bond = spec.d_bond
    bdim = int(d_bond ** N_BONDS)
    dim_full = int((d_site ** N_SITES) * (d_bond ** N_BONDS))

    idxP = site_ground_projector_indices(d_site, d_bond)
    A = (-1j * spec.t) * H

    Ueff = np.zeros((bdim, bdim), dtype=complex)
    for j in range(bdim):
        v = np.zeros(dim_full, dtype=complex)
        v[idxP[j]] = 1.0
        w = expm_multiply(A, v)
        Ueff[:, j] = w[idxP]
    return Ueff

def polar_reunitarize(U: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, dict]:
    M = U.conj().T @ U
    Mh = hermitize_dense(M)
    evals, evecs = np.linalg.eigh(Mh)
    evals_clipped = np.maximum(evals, eps)
    Minvhalf = (evecs * (1.0 / np.sqrt(evals_clipped))) @ evecs.conj().T
    W = U @ Minvhalf
    diag = {
        "proj_nonunitarity_fro": float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]), ord="fro")),
        "min_eig_M": float(np.min(evals).real),
        "max_eig_M": float(np.max(evals).real),
    }
    return W, diag

def compute_Heff_from_W(spec: CaseSpec, W: np.ndarray) -> tuple[np.ndarray, dict]:
    L = logm(W)
    Heff = (1j / spec.t) * L
    Heff = hermitize_dense(Heff)
    diag = {
        "Heff_norm_fro": float(np.linalg.norm(Heff, ord="fro")),
        "Heff_herm_err_fro": float(np.linalg.norm(Heff - Heff.conj().T, ord="fro")),
    }
    return Heff, diag


# -------------------------
# Doubled-space objects (dim 6561)
# -------------------------

def lift_U_super_sparse(W: np.ndarray) -> csr_matrix:
    # U_super = W ⊗ W*
    return skron(csr_matrix(W), csr_matrix(W.conj()), format="csr")

def lift_H_super_sparse(Heff: np.ndarray) -> csr_matrix:
    # H_super = Heff ⊗ I - I ⊗ Heff^T
    d = Heff.shape[0]
    Heff_sp = csr_matrix(Heff)
    I_sp = eye_sp(d)
    return skron(Heff_sp, I_sp, format="csr") - skron(I_sp, csr_matrix(Heff.T), format="csr")


# -------------------------
# True link-end Gauss generators (commuting L/R) on doubled link Hilbert
# -------------------------

def embed_one_link_op_doubled(op9: csr_matrix, which_link: int) -> csr_matrix:
    d_link = 9
    term = None
    for i in range(N_BONDS):
        A = op9 if i == which_link else eye_sp(d_link)
        term = A if term is None else skron(term, A, format="csr")
    return term

def build_LR_ops_on_one_link(T_basis_3x3):
    """
    On one operator-space link (dim 9 = 3*3):
      L^a = T^a ⊗ I
      R^a = I ⊗ (T^a)^T
    """
    I3 = eye_sp(3)
    L_ops = []
    R_ops = []
    for Ta in T_basis_3x3:
        Ta_sp = csr_matrix(Ta)
        L_ops.append(skron(Ta_sp, I3, format="csr"))
        R_ops.append(skron(I3, csr_matrix(Ta.T), format="csr"))
    return L_ops, R_ops

def build_gauss_ops_doubled(T_basis_3x3):
    L_ops_1, R_ops_1 = build_LR_ops_on_one_link(T_basis_3x3)
    G = {}
    for x in range(N_SITES):
        out_link, in_link = VERTEX_OUT_IN[x]
        for a in range(8):
            L_full = embed_one_link_op_doubled(L_ops_1[a], out_link)
            R_full = embed_one_link_op_doubled(R_ops_1[a], in_link)
            G[(x, a)] = L_full - R_full
    return G


# -------------------------
# Metrics on a target operator (U_super or H_super)
# -------------------------

def commutator_sp(A: csr_matrix, B: csr_matrix) -> csr_matrix:
    return (A @ B) - (B @ A)

def comm_metrics_for_target(target: csr_matrix, G_ops: dict) -> dict:
    Tn = fro_norm_sp(target)
    if Tn < 1e-30:
        return {"error": "target Frobenius norm ~0"}

    eps_all = []
    per_vertex = {}

    for x in range(N_SITES):
        eps_a = []
        for a in range(8):
            G = G_ops[(x, a)]
            Gn = fro_norm_sp(G)
            C = commutator_sp(G, target)
            Cn = fro_norm_sp(C)
            eps = float(Cn / (Gn * Tn + 1e-300))
            eps_all.append(eps)
            eps_a.append(eps)

        per_vertex[str(x)] = {
            "eps_a": eps_a,
            "eps_max": float(np.max(eps_a)),
            "eps_mean": float(np.mean(eps_a)),
        }

    arr = np.array(eps_all, dtype=float)
    return {
        "eps_all": eps_all,
        "eps_max": float(np.max(arr)),
        "eps_mean": float(np.mean(arr)),
        "eps_median": float(np.median(arr)),
        "target_norm_fro": float(Tn),
        "per_vertex": per_vertex,
    }


# -------------------------
# Run one micro-case, but evaluate multiple generator bases
# -------------------------

def eval_case_with_basis(spec: CaseSpec, T_basis_3x3) -> dict:
    """
    Computes W, Heff once (for this spec), then evaluates Gauss commutators using provided T_basis.
    """
    H_full = build_full_H_plaquette(spec)
    Ueff = compute_Ueff_bonds(spec, H_full)
    W, polar_diag = polar_reunitarize(Ueff, eps=spec.eps_logm_guard)
    Heff, heff_diag = compute_Heff_from_W(spec, W)

    U_super = lift_U_super_sparse(W)
    H_super = lift_H_super_sparse(Heff)

    G_ops = build_gauss_ops_doubled(T_basis_3x3)

    U_metrics = comm_metrics_for_target(U_super, G_ops)
    H_metrics = comm_metrics_for_target(H_super, G_ops)

    return {
        "polar_diag": polar_diag,
        "heff_diag": heff_diag,
        "U_metrics": U_metrics,
        "H_metrics": H_metrics,
    }


# -------------------------
# CLI
# -------------------------

def parse_csv_floats(s: str) -> list[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def fmt(x):
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.6g}"
    return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Delta", type=str, default="6.0", help="Comma-separated Δ values, e.g. 6,8,10,14")
    ap.add_argument("--g", type=float, default=0.30)
    ap.add_argument("--t", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coupling", type=str, default="both", help="aligned | mixed | both")
    ap.add_argument("--eps_logm_guard", type=float, default=1e-12)
    ap.add_argument("--npz", type=str, default="", help="NPZ with echo bases (basis_both_aligned/mixed). If omitted, NPZ comparison is skipped.")
    args = ap.parse_args()

    Deltas = parse_csv_floats(args.Delta) or [6.0]
    coupling_modes = ["aligned", "mixed"] if args.coupling == "both" else [args.coupling]

    tag = now_tag()
    results = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": tag,
            "npz": args.npz if args.npz else None,
            "note": "v5 compares Gauss commutators using generator_source=gellmann vs generator_source=npz (echo basis_both).",
        },
        "runs": [],
    }

    for coupling in coupling_modes:
        # generator sources available
        bases = []
        bases.append(("gellmann", su_generators_gellmann(3)))

        if args.npz:
            try:
                bases.append(("npz", load_su3_basis_from_npz(args.npz, model=coupling)))
            except Exception as e:
                print(f"[warn] could not load NPZ basis for model={coupling}: {e}")

        for Delta in Deltas:
            # Build micro specs (qubit-site and qutrit-site)
            spec_qubit = CaseSpec(
                name=f"qubit_site_dS2_dB3_{coupling}_Delta{Delta}",
                d_site=2, d_bond=3, Delta=Delta, g=args.g, t=args.t,
                coupling=coupling, seed=args.seed, eps_logm_guard=args.eps_logm_guard,
            )
            spec_qutrit = CaseSpec(
                name=f"qutrit_site_dS3_dB3_{coupling}_Delta{Delta}",
                d_site=3, d_bond=3, Delta=Delta, g=args.g, t=args.t,
                coupling=coupling, seed=args.seed, eps_logm_guard=args.eps_logm_guard,
            )

            rec = {
                "coupling": coupling,
                "Delta": Delta,
                "cases": {
                    "qubit_site": {
                        "spec": spec_qubit.__dict__,
                        "by_generator_source": {}
                    },
                    "qutrit_site": {
                        "spec": spec_qutrit.__dict__,
                        "by_generator_source": {}
                    }
                },
                "comparisons": {}
            }

            # Evaluate both sites for each generator source
            for (src, Tbasis) in bases:
                rQ = eval_case_with_basis(spec_qubit, Tbasis)
                rT = eval_case_with_basis(spec_qutrit, Tbasis)
                rec["cases"]["qubit_site"]["by_generator_source"][src] = rQ
                rec["cases"]["qutrit_site"]["by_generator_source"][src] = rT

            # Produce "before vs after" within each site (npz vs gellmann), if available
            if "npz" in rec["cases"]["qutrit_site"]["by_generator_source"]:
                for site_key in ("qubit_site", "qutrit_site"):
                    g = rec["cases"][site_key]["by_generator_source"]["gellmann"]
                    n = rec["cases"][site_key]["by_generator_source"]["npz"]
                    rec["comparisons"][f"{site_key}_npz_over_gellmann"] = {
                        "U_eps_mean_ratio": (
                            n["U_metrics"]["eps_mean"] / g["U_metrics"]["eps_mean"]
                            if g["U_metrics"].get("eps_mean", 0) > 0 else None
                        ),
                        "U_eps_max_ratio": (
                            n["U_metrics"]["eps_max"] / g["U_metrics"]["eps_max"]
                            if g["U_metrics"].get("eps_max", 0) > 0 else None
                        ),
                        "H_eps_mean_ratio": (
                            n["H_metrics"]["eps_mean"] / g["H_metrics"]["eps_mean"]
                            if g["H_metrics"].get("eps_mean", 0) > 0 else None
                        ),
                        "H_eps_max_ratio": (
                            n["H_metrics"]["eps_max"] / g["H_metrics"]["eps_max"]
                            if g["H_metrics"].get("eps_max", 0) > 0 else None
                        ),
                    }

            # Also keep the "qutrit over qubit" ratios for each generator source
            for (src, _Tb) in bases:
                q = rec["cases"]["qubit_site"]["by_generator_source"][src]
                t = rec["cases"]["qutrit_site"]["by_generator_source"][src]
                rec["comparisons"][f"qutrit_over_qubit_{src}"] = {
                    "U_eps_mean_ratio": (
                        t["U_metrics"]["eps_mean"] / q["U_metrics"]["eps_mean"]
                        if q["U_metrics"].get("eps_mean", 0) > 0 else None
                    ),
                    "U_eps_max_ratio": (
                        t["U_metrics"]["eps_max"] / q["U_metrics"]["eps_max"]
                        if q["U_metrics"].get("eps_max", 0) > 0 else None
                    ),
                    "H_eps_mean_ratio": (
                        t["H_metrics"]["eps_mean"] / q["H_metrics"]["eps_mean"]
                        if q["H_metrics"].get("eps_mean", 0) > 0 else None
                    ),
                    "H_eps_max_ratio": (
                        t["H_metrics"]["eps_max"] / q["H_metrics"]["eps_max"]
                        if q["H_metrics"].get("eps_max", 0) > 0 else None
                    ),
                }

            results["runs"].append(rec)

            # Console report (compact)
            print("--------------------------------------------------------------------------------")
            print(f"[v5] Δ={Delta:.3f} coupling={coupling} g={args.g:.3f} t={args.t:.3f}")
            for src, _Tb in bases:
                q = rec["cases"]["qubit_site"]["by_generator_source"][src]
                t_ = rec["cases"]["qutrit_site"]["by_generator_source"][src]
                comp = rec["comparisons"].get(f"qutrit_over_qubit_{src}", {})
                print(f"  generators={src}")
                print(f"    U_super: qubit mean={fmt(q['U_metrics'].get('eps_mean'))} max={fmt(q['U_metrics'].get('eps_max'))} | "
                      f"qutrit mean={fmt(t_['U_metrics'].get('eps_mean'))} max={fmt(t_['U_metrics'].get('eps_max'))} | "
                      f"ratio mean={fmt(comp.get('U_eps_mean_ratio'))} max={fmt(comp.get('U_eps_max_ratio'))}")
                print(f"    H_super: qubit mean={fmt(q['H_metrics'].get('eps_mean'))} max={fmt(q['H_metrics'].get('eps_max'))} | "
                      f"qutrit mean={fmt(t_['H_metrics'].get('eps_mean'))} max={fmt(t_['H_metrics'].get('eps_max'))} | "
                      f"ratio mean={fmt(comp.get('H_eps_mean_ratio'))} max={fmt(comp.get('H_eps_max_ratio'))}")

            if "npz" in rec["cases"]["qutrit_site"]["by_generator_source"]:
                bq = rec["comparisons"].get("qutrit_site_npz_over_gellmann", {})
                print("  echo-basis impact on qutrit-site (npz/gellmann): "
                      f"U_mean={fmt(bq.get('U_eps_mean_ratio'))} U_max={fmt(bq.get('U_eps_max_ratio'))} "
                      f"H_mean={fmt(bq.get('H_eps_mean_ratio'))} H_max={fmt(bq.get('H_eps_max_ratio'))}")

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"gauss_commutator_su3_v5_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("================================================================================")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()