#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_commutator_su3_v3_doubled_link.py
======================================

What v3 does (and why)
----------------------
Your v2 Gauss test still used a surrogate "single-action" bond generator T^a on a dB=3 link.
But a genuine SU(3) gauge link requires TWO commuting end-actions:

  L^a  (left action at source end)
  R^a  (right action at target end)
  with [L^a, R^b] = 0.

A single qutrit bond Hilbert C^3 cannot host two commuting su(3) operator algebras as ordinary
operators on the same 3D space. The standard fix is to represent a link as an *operator space*
(aka doubled / Liouville / Choi / "matrix element") Hilbert of dimension d^2.

For dB=3:
  link Hilbert (doubled) = C^3 ⊗ C^3  (dim 9)
  L^a = T^a ⊗ I
  R^a = I ⊗ (T^a)^T
These commute exactly.

What this script tests
----------------------
We keep your microscopic model exactly like v1 (sites + fundamental bonds),
compute the *bond-only effective unitary* on the plaquette:

  U_eff_bonds  on H_bonds = (C^3)^{⊗4}  (dim 81)

Then we compute the effective Hamiltonian H_eff_bonds via:
  U_eff = P exp(-i t H_full) P
  polar reunitarize -> W
  H_eff_bonds = (i/t) logm(W)

Now lift this to the doubled (operator-space) representation on the same 4 links:
  H_eff_super = H_eff_bonds ⊗ I - I ⊗ (H_eff_bonds)^T
This generates conjugation:
  X -> e^{-i t H_eff_bonds} X e^{+i t H_eff_bonds}

Finally we build true Gauss generators on the doubled link Hilbert:
  For each vertex x and generator a=0..7:
    G_x^a = + L_link(outgoing, a)  - R_link(incoming, a)
(using plaquette orientation)
and compute:
  eps(x,a) = ||[G_x^a, H_eff_super]||_F / (||G_x^a||_F ||H_eff_super||_F)

This is a *pure-gauge* invariance test (no matter charge term Q included), appropriate for the
bond-only effective dynamics you are currently extracting.

Optional: use echo-basis generators from your NPZ
-------------------------------------------------
If you pass --npz path/to/echo_algebra_step1_qutrit_su3_LR_bases_*.npz,
the script will use basis_both_<model> (aligned/mixed) as the su(3) generator set T^a
instead of the canonical Gell-Mann basis.

Run (Windows one-liners)
------------------------
1) Canonical SU(3) generators:
python gauss_commutator_su3_v3_doubled_link.py --Delta 6,8,10,14 --g 0.30 --t 0.10 --seed 0 --coupling both

2) Use your extracted basis from NPZ:
python gauss_commutator_su3_v3_doubled_link.py --Delta 6,8,10,14 --g 0.30 --t 0.10 --seed 0 --coupling both --npz hsf_out\\echo_algebra_step1_qutrit_su3_LR_bases_20260219_144601.npz

Outputs
-------
Writes JSON to ./hsf_out/gauss_commutator_su3_v3_<timestamp>.json

Dependencies: numpy, scipy
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
# Small utilities
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
    # Frobenius norm from sparse data
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
# su(3) generator basis
# -------------------------

def su_generators_gellmann(d: int):
    """
    HS-orthonormal Hermitian traceless su(d) basis.
    For d=3 yields 8.
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

    out = [normalize_hs_dense(traceless(hermitize_dense(G))) for G in gens]
    out = gram_schmidt_hs_dense(out, tol=1e-12)
    return out

def load_su3_basis_from_npz(npz_path: str, model: str):
    """
    Loads basis_both_<model> from the NPZ produced by echo_algebra_step1_qutrit_sites_su3_LR.py.
    Expects array shape (8,3,3).
    """
    data = np.load(npz_path)
    key = f"basis_both_{model}"
    if key not in data:
        raise KeyError(f"NPZ missing key '{key}'. Available: {list(data.keys())[:20]}")
    arr = data[key]
    if arr.shape[0] < 8 or arr.shape[1:] != (3, 3):
        raise ValueError(f"{key} must have shape (>=8,3,3), got {arr.shape}")
    basis = [arr[i].astype(complex) for i in range(8)]
    # re-orthonormalize defensively
    basis = [normalize_hs_dense(traceless(hermitize_dense(B))) for B in basis]
    basis = gram_schmidt_hs_dense(basis, tol=1e-12)
    if len(basis) < 8:
        raise RuntimeError(f"After orthonormalization, got only {len(basis)} generators from NPZ.")
    return basis[:8]


# -------------------------
# Geometry (plaquette)
# -------------------------

N_SITES = 4
N_BONDS = 4

# edges (u,v,bond_index) with orientation u -> v
EDGES = [
    (0, 1, 0),
    (1, 2, 1),
    (2, 3, 2),
    (3, 0, 3),
]

# for each vertex x: outgoing bond index, incoming bond index
VERTEX_OUT_IN = {
    0: (0, 3),
    1: (1, 0),
    2: (2, 1),
    3: (3, 2),
}


# -------------------------
# Microscopic model (same as v1)
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

def build_edge_couplings(d_site: int, d_bond: int, coupling: str, rng: np.random.Generator):
    """
    For d_bond=3, bond gens are 8.
    For d_site:
      d_site=3 => 8 gens
      d_site=2 => 3 gens (Paulis), coupled to first 3 bond gens.
    """
    if d_bond != 3:
        raise ValueError("This v3 script assumes d_bond=3 (SU(3) fundamental bonds).")

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
    """
    H0 = Σ_s (I - |0><0|)_s ⊗ I_bonds
    """
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
    """
    H = Δ H0 + g V
    V = Σ_edges Σ_a (S_u^a ⊗ B_e^a ⊗ S_v^a)
    Tensor order: [sites..., bonds...]
    """
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
# Bond-only projection and Heff_bonds
# -------------------------

def site_ground_projector_indices(d_site: int, d_bond: int) -> np.ndarray:
    """
    Indices for |0...0>_sites ⊗ |b>_bonds for all bond computational states.
    With tensor order [sites..., bonds...], these are exactly 0..(bdim-1).
    """
    bdim = int(d_bond ** N_BONDS)
    return np.arange(bdim, dtype=np.int64)

def compute_Ueff_bonds(spec: CaseSpec, H: csr_matrix) -> np.ndarray:
    """
    U_eff = P exp(-i t H) P on the bond manifold (sites frozen to |0>).
    """
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
    """
    W = U (U†U)^(-1/2)
    """
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

def compute_Heff_bonds_from_W(spec: CaseSpec, W: np.ndarray) -> np.ndarray:
    """
    Heff_bonds = (i/t) logm(W), Hermitianized.
    """
    L = logm(W)
    Heff = (1j / spec.t) * L
    Heff = hermitize_dense(Heff)
    return Heff


# -------------------------
# Lift to doubled (operator-space) Hamiltonian
# -------------------------

def lift_to_super_hamiltonian_sparse(Heff_bonds: np.ndarray) -> csr_matrix:
    """
    For unitary evolution generated by Heff_bonds:
      X -> e^{-itHeff} X e^{+itHeff}
    vec(X) evolves with:
      H_super = Heff ⊗ I - I ⊗ (Heff)^T

    This is the natural doubled-link (Liouville) Hamiltonian.
    """
    d = Heff_bonds.shape[0]  # should be 81
    Heff_sp = csr_matrix(Heff_bonds)
    I_sp = eye_sp(d)
    # H ⊗ I - I ⊗ H^T
    H_super = skron(Heff_sp, I_sp, format="csr") - skron(I_sp, csr_matrix(Heff_bonds.T), format="csr")
    return H_super


# -------------------------
# True link-end generators on doubled link Hilbert
# -------------------------

def embed_one_link_op_doubled(op9: csr_matrix, which_link: int) -> csr_matrix:
    """
    Embed a 9x9 operator on one link into the full doubled plaquette Hilbert:
      H_link_doubled = (C^3 ⊗ C^3)^{⊗4} has dimension 9^4 = 6561
    link order: e0,e1,e2,e3
    """
    d_link = 9
    term = None
    for i in range(N_BONDS):
        A = op9 if i == which_link else eye_sp(d_link)
        term = A if term is None else skron(term, A, format="csr")
    return term

def build_LR_ops_on_one_link(T_basis_3x3):
    """
    Given su(3) generators T^a on C^3 (Hermitian),
    return L^a and R^a on C^3 ⊗ C^3 (dim 9):

      L^a = T^a ⊗ I
      R^a = I ⊗ (T^a)^T

    NOTE: transpose (not dagger) is the standard vec convention for right multiplication.
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
    """
    Build G_x^a on full doubled plaquette Hilbert (dim 6561):
      G_x^a = + L_out(link_out)^a  - R_in(link_in)^a
    using vertex outgoing/incoming links from plaquette orientation.
    """
    L_ops_1, R_ops_1 = build_LR_ops_on_one_link(T_basis_3x3)  # lists of 8 (9x9) sparse

    G = {}
    for x in range(N_SITES):
        out_link, in_link = VERTEX_OUT_IN[x]
        for a in range(8):
            L_full = embed_one_link_op_doubled(L_ops_1[a], out_link)
            R_full = embed_one_link_op_doubled(R_ops_1[a], in_link)
            G[(x, a)] = L_full - R_full
    return G


# -------------------------
# Metrics
# -------------------------

def commutator_sp(A: csr_matrix, B: csr_matrix) -> csr_matrix:
    return (A @ B) - (B @ A)

def gauss_commutator_metrics(H_super: csr_matrix, G_ops: dict) -> dict:
    Hn = fro_norm_sp(H_super)
    if Hn < 1e-30:
        return {"error": "H_super Frobenius norm is ~0; cannot evaluate."}

    eps_all = []
    per_vertex = {}

    for x in range(N_SITES):
        eps_a = []
        for a in range(8):
            G = G_ops[(x, a)]
            Gn = fro_norm_sp(G)
            C = commutator_sp(G, H_super)
            Cn = fro_norm_sp(C)
            eps = float(Cn / (Gn * Hn + 1e-300))
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
        "per_vertex": per_vertex,
    }


# -------------------------
# Run one case
# -------------------------

def run_one_case(spec: CaseSpec, T_basis_3x3) -> dict:
    H_full = build_full_H_plaquette(spec)
    Ueff = compute_Ueff_bonds(spec, H_full)

    W, diag_polar = polar_reunitarize(Ueff, eps=spec.eps_logm_guard)
    Heff_bonds = compute_Heff_bonds_from_W(spec, W)

    # Lift to doubled (operator-space) Hamiltonian
    H_super = lift_to_super_hamiltonian_sparse(Heff_bonds)

    # Build true Gauss generators on doubled plaquette
    G_ops = build_gauss_ops_doubled(T_basis_3x3)

    gauss = gauss_commutator_metrics(H_super, G_ops)

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
        "polar_diag": diag_polar,
        "Heff_bonds_norm_fro": float(np.linalg.norm(Heff_bonds, ord="fro")),
        "H_super_norm_fro": float(fro_norm_sp(H_super)),
        "gauss": gauss,
    }
    return out


# -------------------------
# CLI
# -------------------------

def parse_csv_floats(s: str) -> list[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Delta", type=str, default="6.0", help="Comma-separated Δ values, e.g. 6,8,10,14")
    ap.add_argument("--g", type=float, default=0.30)
    ap.add_argument("--t", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coupling", type=str, default="both", help="aligned | mixed | both")
    ap.add_argument("--eps_logm_guard", type=float, default=1e-12)
    ap.add_argument("--npz", type=str, default="", help="Optional NPZ with echo bases to use (basis_both_aligned/mixed)")
    args = ap.parse_args()

    Deltas = parse_csv_floats(args.Delta) or [6.0]
    coupling_modes = ["aligned", "mixed"] if args.coupling == "both" else [args.coupling]

    results = {
        "meta": {
            "script": os.path.basename(__file__),
            "timestamp": now_tag(),
            "npz": args.npz if args.npz else None,
            "note": "v3 doubled-link Gauss test: pure-gauge invariance on lifted Liouville Hamiltonian",
        },
        "runs": [],
    }

    for coupling in coupling_modes:
        # Choose generator basis T^a on C^3
        if args.npz:
            T_basis = load_su3_basis_from_npz(args.npz, model=coupling)
            gen_src = f"npz:{os.path.basename(args.npz)}:{coupling}"
        else:
            T_basis = su_generators_gellmann(3)
            gen_src = "gellmann"

        for Delta in Deltas:
            spec2 = CaseSpec(
                name=f"qubit_site_dS2_dB3_{coupling}_Delta{Delta}",
                d_site=2, d_bond=3, Delta=Delta,
                g=args.g, t=args.t, coupling=coupling,
                seed=args.seed, eps_logm_guard=args.eps_logm_guard,
            )
            spec3 = CaseSpec(
                name=f"qutrit_site_dS3_dB3_{coupling}_Delta{Delta}",
                d_site=3, d_bond=3, Delta=Delta,
                g=args.g, t=args.t, coupling=coupling,
                seed=args.seed, eps_logm_guard=args.eps_logm_guard,
            )

            r2 = run_one_case(spec2, T_basis)
            r3 = run_one_case(spec3, T_basis)

            runrec = {
                "coupling": coupling,
                "Delta": Delta,
                "generator_source": gen_src,
                "qubit_site": r2,
                "qutrit_site": r3,
                "ratio_eps_max_qutrit_over_qubit": (
                    (r3["gauss"]["eps_max"] / r2["gauss"]["eps_max"])
                    if ("eps_max" in r2["gauss"] and r2["gauss"]["eps_max"] > 0)
                    else None
                ),
                "ratio_eps_mean_qutrit_over_qubit": (
                    (r3["gauss"]["eps_mean"] / r2["gauss"]["eps_mean"])
                    if ("eps_mean" in r2["gauss"] and r2["gauss"]["eps_mean"] > 0)
                    else None
                ),
            }
            results["runs"].append(runrec)

            # Console summary
            print("------------------------------------------------------------")
            print(f"[v3] Δ={Delta:.3f} coupling={coupling} g={args.g:.3f} t={args.t:.3f}  generators={gen_src}")
            print(f"  qubit-site  eps_max={r2['gauss'].get('eps_max', None)}  eps_mean={r2['gauss'].get('eps_mean', None)}")
            print(f"  qutrit-site eps_max={r3['gauss'].get('eps_max', None)}  eps_mean={r3['gauss'].get('eps_mean', None)}")
            print(f"  ratio(qutrit/qubit): max={runrec['ratio_eps_max_qutrit_over_qubit']}  mean={runrec['ratio_eps_mean_qutrit_over_qubit']}")

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"gauss_commutator_su3_v3_{results['meta']['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("============================================================")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
