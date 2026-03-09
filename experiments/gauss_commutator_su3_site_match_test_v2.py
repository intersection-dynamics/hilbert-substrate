#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauss_commutator_su3_site_match_test_v2.py
=========================================

What this v2 fixes vs v1
------------------------
v1 projected ALL sites to |0>, so the Gauss operator could not include matter charge Q_x^a.
That makes it impossible to test the "d_site = d_B fixes Gauss" hypothesis, because the fix
lives in the cancellation between Q_x^a and adjacent bond-end flux terms.

v2 instead projects onto a SMALL low-energy subspace that STILL CONTAINS site degrees of freedom:
  - keep all bond states (d_B^4)
  - keep site states with AT MOST ONE site excited away from |0>
    (this keeps the Hilbert size manageable, but still lets Q_x^a act nontrivially)

Then it computes an effective Hamiltonian on that projected subspace:
  U_eff = P exp(-i t H) P
  (polar reunitarize) -> logm -> H_eff

And tests a *matter+flux* Gauss operator (surrogate):
  G_x^a = Q_x^a + T^a(out bond) - T^a(in bond)

Notes / limitations (explicit)
------------------------------
1) This is a *surrogate* Gauss law because our bonds are modeled as single d_B registers.
   A full lattice gauge theory distinguishes left/right actions (L^a and R^a) on each link end,
   which requires a link Hilbert with both actions realized consistently. If your echo-bond
   construction already provides L/R operators, you should swap them in.
   For now we use the same su(3) generator on the bond register with orientation signs.

2) The projection subspace ("<= 1 site excitation") is chosen to be:
   - large enough to include matter charge operators Q_x^a
   - small enough to keep U_eff and H_eff practical
   If you want "2 excitations" (stronger test), it is an easy extension.

Run (Windows, one line)
-----------------------
python gauss_commutator_su3_site_match_test_v2.py --Delta 6,8,10,14 --g 0.30 --t 0.10 --seed 0 --coupling both

Outputs
-------
Writes JSON to ./hsf_out/gauss_commutator_su3_v2_<timestamp>.json
and prints a compact comparison summary.

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
# Helpers
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

def commutator_dense(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q


# -------------------------
# su(d) generators
# -------------------------

def su_generators_gellmann(d: int):
    """
    HS-orthonormal Hermitian traceless su(d) basis.
    For d=3 this yields 8 generators.
    """
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

def su2_pauli_generators_hs():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [normalize_hs_dense(X), normalize_hs_dense(Y), normalize_hs_dense(Z)]


# -------------------------
# Model spec
# -------------------------

@dataclass
class CaseSpec:
    name: str
    d_site: int = 3
    d_bond: int = 3
    Delta: float = 6.0
    g: float = 0.30
    t: float = 0.10
    coupling: str = "aligned"  # aligned | mixed
    seed: int = 0
    eps_logm_guard: float = 1e-12
    max_site_excitations: int = 1  # v2 key: keep <= this many excitations away from |0>


# -------------------------
# Geometry: plaquette
# -------------------------

N_SITES = 4
N_BONDS = 4

# Edges (u, v, bond_index) with orientation u -> v:
EDGES = [
    (0, 1, 0),  # e0
    (1, 2, 1),  # e1
    (2, 3, 2),  # e2
    (3, 0, 3),  # e3
]

# For each vertex x: outgoing bond, incoming bond (by the above orientation)
VERTEX_OUT_IN = {
    0: (0, 3),
    1: (1, 0),
    2: (2, 1),
    3: (3, 2),
}


# -------------------------
# Coupling construction
# -------------------------

def build_edge_couplings(d_site: int, d_bond: int, coupling: str, rng: np.random.Generator):
    """
    Returns list of (S_site, B_bond, S_site_other_end) dense ops to sum for one edge.

    Bond is SU(3) => 8 generators.
    Site:
      d_site=3 => 8 generators
      d_site=2 => 3 generators
    """
    if d_bond != 3:
        raise ValueError("v2 script currently assumes d_bond=3 (SU(3) bond registers).")

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
        return [(site_gens[a], bond_gens[a], site_gens[a]) for a in range(k)]

    if coupling == "mixed":
        O = random_orthogonal(nB, rng)  # 8x8
        out = []
        for a in range(k):
            Bb = np.zeros((3, 3), dtype=complex)
            for b in range(nB):
                Bb += O[b, a] * bond_gens[b]
            Bb = normalize_hs_dense(traceless(hermitize_dense(Bb)))
            out.append((site_gens[a], Bb, site_gens[a]))
        return out

    raise ValueError("coupling must be 'aligned' or 'mixed'.")


# -------------------------
# Sparse operator assembly utilities
# -------------------------

def embed_factor_op_sparse(op_sp: csr_matrix, which: int, dims: list[int]) -> csr_matrix:
    """
    Embed op_sp acting on factor 'which' into full tensor product space defined by dims.
    Uses kron chaining (CSR).
    """
    term = None
    for k, d in enumerate(dims):
        A = op_sp if k == which else eye_sp(d)
        term = A if term is None else skron(term, A, format="csr")
    return term

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
        H0 = H0 + embed_factor_op_sparse(Q, s, dims)

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
# Projection subspace: <= max_site_excitations away from |0>
# -------------------------

def unravel_index(idx: int, dims: list[int]) -> list[int]:
    out = []
    for d in reversed(dims):
        out.append(idx % d)
        idx //= d
    return list(reversed(out))

def ravel_index(letters: list[int], dims: list[int]) -> int:
    idx = 0
    for x, d in zip(letters, dims):
        idx = idx * d + x
    return idx

def build_projected_basis_indices(d_site: int, d_bond: int, max_site_ex: int) -> list[int]:
    """
    Basis indices in full space for all states with:
      - bonds: any values (0..d_bond-1 on each of 4 bonds)
      - sites: excitations count <= max_site_ex, where excitation means site != 0
    """
    dims = [d_site] * N_SITES + [d_bond] * N_BONDS

    # enumerate site configurations with <= max_site_ex excitations
    site_confs = []
    # brute force is fine: d_site^4 is 16 or 81
    for s0 in range(d_site):
        for s1 in range(d_site):
            for s2 in range(d_site):
                for s3 in range(d_site):
                    ex = int(s0 != 0) + int(s1 != 0) + int(s2 != 0) + int(s3 != 0)
                    if ex <= max_site_ex:
                        site_confs.append([s0, s1, s2, s3])

    # enumerate all bond configurations
    bond_confs = []
    for b0 in range(d_bond):
        for b1 in range(d_bond):
            for b2 in range(d_bond):
                for b3 in range(d_bond):
                    bond_confs.append([b0, b1, b2, b3])

    idxs = []
    for sc in site_confs:
        for bc in bond_confs:
            letters = sc + bc
            idxs.append(ravel_index(letters, dims))

    return idxs


# -------------------------
# Effective evolution on projected basis
# -------------------------

def safe_logm(U: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, dict]:
    """
    Polar reunitarize U then logm.
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
        "min_eig_M_clipped": float(np.min(evals_clipped).real),
        "max_eig_M": float(np.max(evals).real),
    }
    return logm(W), diag

def compute_U_eff_on_subspace(spec: CaseSpec, H: csr_matrix, sub_idx: list[int]) -> np.ndarray:
    """
    U_eff = P exp(-i t H) P on the chosen subspace basis.

    We do it column-by-column using expm_multiply on full space, then restrict rows to subspace.
    """
    dim_full = H.shape[0]
    K = len(sub_idx)
    sub_idx_arr = np.array(sub_idx, dtype=np.int64)

    A = (-1j * spec.t) * H

    Ueff = np.zeros((K, K), dtype=complex)
    for j in range(K):
        v = np.zeros(dim_full, dtype=complex)
        v[sub_idx_arr[j]] = 1.0
        w = expm_multiply(A, v)
        Ueff[:, j] = w[sub_idx_arr]
    return Ueff

def compute_H_eff(spec: CaseSpec, Ueff: np.ndarray) -> tuple[np.ndarray, dict]:
    L, diag = safe_logm(Ueff, eps=spec.eps_logm_guard)
    Heff = (1j / spec.t) * L
    Heff = hermitize_dense(Heff)
    diag["Heff_herm_err_fro"] = float(np.linalg.norm(Heff - Heff.conj().T, ord="fro"))
    diag["Heff_norm_fro"] = float(np.linalg.norm(Heff, ord="fro"))
    return Heff, diag


# -------------------------
# Projected Gauss operators (matter + flux surrogate)
# -------------------------

def project_operator_to_subspace(op_full: csr_matrix, sub_idx: list[int]) -> np.ndarray:
    """
    Compute the dense projected operator:
      O_proj = P O P
    where P selects basis vectors indexed by sub_idx.
    """
    dim_full = op_full.shape[0]
    K = len(sub_idx)
    sub_idx_arr = np.array(sub_idx, dtype=np.int64)

    Oproj = np.zeros((K, K), dtype=complex)
    for j in range(K):
        v = np.zeros(dim_full, dtype=complex)
        v[sub_idx_arr[j]] = 1.0
        w = op_full @ v
        Oproj[:, j] = w[sub_idx_arr]
    return Oproj

def build_Q_site_ops(spec: CaseSpec):
    """
    Matter charge generators Q_x^a on site x.

    For d_site=3: use 8 su(3) generators.
    For d_site=2: embed su(2) generators (3) as-is; for the Gauss test we will only
                 compare across the *8* bond directions by padding Q with zeros for a>=3.
                 This is intentional: it shows mismatch cannot cancel all 8 flux directions.
    """
    if spec.d_site == 3:
        Qloc = su_generators_gellmann(3)  # 8
        return Qloc, 8
    elif spec.d_site == 2:
        Qloc = su2_pauli_generators_hs()  # 3
        return Qloc, 3
    else:
        raise ValueError("d_site must be 2 or 3.")

def build_T_bond_ops():
    """
    Flux generators on a bond register (SU(3)).
    """
    return su_generators_gellmann(3)  # 8

def build_full_Gauss_ops_projected(spec: CaseSpec, sub_idx: list[int]) -> dict:
    """
    Build projected Gauss operators G_x^a = Q_x^a + T^a(out) - T^a(in) on the projected subspace.

    Returns dict keyed by (x,a) with dense matrices.
    """
    dims = [spec.d_site] * N_SITES + [spec.d_bond] * N_BONDS

    Qloc_list, q_count = build_Q_site_ops(spec)
    Tloc_list = build_T_bond_ops()  # 8 always

    # Prebuild full-space sparse embeddings for bond T^a on each bond
    bond_T_full = {}  # (bond_index, a) -> csr
    for b in range(N_BONDS):
        factor = N_SITES + b
        for a in range(8):
            bond_T_full[(b, a)] = embed_factor_op_sparse(csr_matrix(Tloc_list[a]), factor, dims)

    # Prebuild full-space sparse embeddings for site Q^a on each site
    # For d_site=2, only a<3 exist; for a>=3 we treat Q=0 (mismatch).
    site_Q_full = {}  # (site, a) -> csr or None
    for x in range(N_SITES):
        for a in range(8):
            if a < q_count:
                site_Q_full[(x, a)] = embed_factor_op_sparse(csr_matrix(Qloc_list[a]), x, dims)
            else:
                site_Q_full[(x, a)] = None

    Gproj = {}
    for x in range(N_SITES):
        out_b, in_b = VERTEX_OUT_IN[x]
        for a in range(8):
            G_full = bond_T_full[(out_b, a)] - bond_T_full[(in_b, a)]
            if site_Q_full[(x, a)] is not None:
                G_full = G_full + site_Q_full[(x, a)]
            # Project to subspace
            Gproj[(x, a)] = project_operator_to_subspace(G_full, sub_idx)
    return Gproj


# -------------------------
# Metrics
# -------------------------

def gauss_commutator_metrics(Heff: np.ndarray, Gproj: dict) -> dict:
    """
    Compute normalized commutator norms for all vertices x and generators a:

      eps(x,a) = ||[G_x^a, H_eff]||_F / (||G_x^a||_F ||H_eff||_F)
    """
    Hn = np.linalg.norm(Heff, ord="fro")
    if Hn < 1e-30:
        return {"error": "Heff Frobenius norm is ~0; cannot evaluate."}

    eps_all = []
    per_vertex = {}

    for x in range(N_SITES):
        eps_a = []
        for a in range(8):
            G = Gproj[(x, a)]
            Gn = np.linalg.norm(G, ord="fro")
            C = commutator_dense(G, Heff)
            Cn = np.linalg.norm(C, ord="fro")
            eps = float(Cn / (Gn * Hn + 1e-300))
            eps_all.append(eps)
            eps_a.append(eps)
        per_vertex[str(x)] = {
            "eps_a": eps_a,
            "eps_max": float(np.max(eps_a)),
            "eps_mean": float(np.mean(eps_a)),
        }

    eps_arr = np.array(eps_all, dtype=float)
    return {
        "eps_all": eps_all,
        "eps_max": float(np.max(eps_arr)),
        "eps_mean": float(np.mean(eps_arr)),
        "eps_median": float(np.median(eps_arr)),
        "per_vertex": per_vertex,
    }


# -------------------------
# Runner
# -------------------------

def run_one_case(spec: CaseSpec) -> dict:
    # Full model
    H = build_full_H_plaquette(spec)

    # Projection basis indices
    sub_idx = build_projected_basis_indices(spec.d_site, spec.d_bond, spec.max_site_excitations)
    K = len(sub_idx)

    # Effective evolution on subspace
    Ueff = compute_U_eff_on_subspace(spec, H, sub_idx)
    Heff, logdiag = compute_H_eff(spec, Ueff)

    # Build projected Gauss operators (matter+flux)
    Gproj = build_full_Gauss_ops_projected(spec, sub_idx)

    # Metrics
    gauss = gauss_commutator_metrics(Heff, Gproj)

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
            "max_site_excitations": spec.max_site_excitations,
            "subspace_dim": K,
        },
        "logm_diag": logdiag,
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
    ap.add_argument("--Delta", type=str, default="6.0", help="Comma-separated Δ values, e.g. 6,8,10,14")
    ap.add_argument("--g", type=float, default=0.30)
    ap.add_argument("--t", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coupling", type=str, default="both", help="aligned | mixed | both")
    ap.add_argument("--eps_logm_guard", type=float, default=1e-12)
    ap.add_argument("--max_site_excitations", type=int, default=1, help="Keep site states with <= this many excitations")
    args = ap.parse_args()

    Deltas = parse_csv_floats(args.Delta)
    if not Deltas:
        Deltas = [6.0]

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
            # qubit site (mismatch)
            spec2 = CaseSpec(
                name=f"qubit_site_dS2_dB3_{coupling}_Delta{Delta}",
                d_site=2, d_bond=3,
                Delta=Delta, g=args.g, t=args.t,
                coupling=coupling,
                seed=args.seed,
                eps_logm_guard=args.eps_logm_guard,
                max_site_excitations=args.max_site_excitations,
            )
            # qutrit site (match)
            spec3 = CaseSpec(
                name=f"qutrit_site_dS3_dB3_{coupling}_Delta{Delta}",
                d_site=3, d_bond=3,
                Delta=Delta, g=args.g, t=args.t,
                coupling=coupling,
                seed=args.seed,
                eps_logm_guard=args.eps_logm_guard,
                max_site_excitations=args.max_site_excitations,
            )

            r2 = run_one_case(spec2)
            r3 = run_one_case(spec3)

            runrec = {
                "coupling": coupling,
                "Delta": Delta,
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
            print(f"Δ={Delta:.3f} coupling={coupling} g={args.g:.3f} t={args.t:.3f}  (<= {args.max_site_excitations} site excitations)")
            print(f"  qubit-site  K={r2['case']['subspace_dim']}  eps_max={r2['gauss'].get('eps_max', None)}  eps_mean={r2['gauss'].get('eps_mean', None)}")
            print(f"  qutrit-site K={r3['case']['subspace_dim']}  eps_max={r3['gauss'].get('eps_max', None)}  eps_mean={r3['gauss'].get('eps_mean', None)}")
            print(f"  ratio(qutrit/qubit): max={runrec['ratio_eps_max_qutrit_over_qubit']}  mean={runrec['ratio_eps_mean_qutrit_over_qubit']}")

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"gauss_commutator_su3_v2_{results['meta']['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("============================================================")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
