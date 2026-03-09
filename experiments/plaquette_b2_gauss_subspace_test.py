#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plaquette_b2_gauss_subspace_test.py
===================================

B1 → B2 decisive tests (Path A):

1) Does the *dynamical* ground state of the full echo Hamiltonian live in/near
   the Gauss-law (gauge) constraint surface?

   - Build full H = H0 + g V on sites⊗bonds (small graphs: triangle/square).
   - Compute ground state |Ψ0>.
   - Reduce to bonds: ρ_b = Tr_sites |Ψ0><Ψ0|.
   - Build Gauss generators G_s^a on bonds only, and constraint operator:
       C = Σ_{s,a} (G_s^a)^2
   - Measure:
       ⟨C⟩ = Tr(ρ_b C)
       overlap with "Gauss subspace" = Tr(ρ_b P_phys)

   Note: exact kernel of all G_s^a may be trivial for this simplified bond-qubit
   representation. So we define P_phys as the low-violation subspace of C:
   eigenvectors with eigenvalue <= eps_abs, and if none exist, the lowest-k
   eigenvectors (k selectable).

2) Does the B1 loop-term (SW order = cycle length) preserve that subspace?

   The non-vacuous invariance test is *leakage*:
       leakage = || (I - P_phys) H_loop P_phys ||_F / ||H_loop||_F
   If leakage is small, H_loop maps the constraint surface to itself,
   i.e., gauge invariance holds on physical states even if [H, G] ≠ 0 on the full space.

Outputs:
- Console report for Triangle (3 sites/3 bonds) and Square (4 sites/4 bonds)
- Saves one text summary into ./hsf_out

Run:
  python plaquette_b2_gauss_subspace_test.py
"""

import os
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh

# ----------------------------
# Basic operators
# ----------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def embed_qubit_op(op, which, n_qubits):
    ops = [I2] * n_qubits
    ops[which] = op
    return kron_all(ops)


def embed_bond_op(op, which_bond, n_bonds):
    return embed_qubit_op(op, which_bond, n_bonds)


def partial_trace_sites(rho, d_sites, d_bonds):
    rho = rho.reshape(d_sites, d_bonds, d_sites, d_bonds)
    out = np.zeros((d_bonds, d_bonds), dtype=complex)
    for i in range(d_sites):
        out += rho[i, :, i, :]
    return out


# ----------------------------
# Graph/Lattice model
# ----------------------------

@dataclass
class GraphSpec:
    name: str
    n_sites: int
    edges: list  # list of tuples (u, v) with bond index = position in list
    cycle_len: int

    @property
    def n_bonds(self):
        return len(self.edges)

    def incidence(self):
        inc = [[] for _ in range(self.n_sites)]
        for b, (u, v) in enumerate(self.edges):
            # orient u -> v
            inc[u].append((b, +1))
            inc[v].append((b, -1))
        return inc


TRI = GraphSpec("Triangle", 3, [(0, 1), (1, 2), (0, 2)], 3)
SQR = GraphSpec("Square", 4, [(0, 1), (1, 2), (2, 3), (0, 3)], 4)


# ----------------------------
# Full echo Hamiltonian (sites ⊗ bonds)
# ----------------------------

def build_H0_sites(n_sites, gaps):
    # H0 = - Σ_i Δ_i Z_i so |0...0> is ground with E0 = -Σ Δ_i
    H0 = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for i in range(n_sites):
        H0 += (-gaps[i]) * embed_qubit_op(Z, i, n_sites)
    return H0


def build_V_sites_bonds(spec: GraphSpec, n_sites, n_bonds):
    """
    Transmission coupling for each edge (u)-(bond b)-(v):

      V_b = X_u ⊗ X_b ⊗ X_v  +  Z_u ⊗ Z_b ⊗ Z_v

    on (sites ⊗ bonds).
    """
    d_sites = 2**n_sites
    d_bonds = 2**n_bonds
    D = d_sites * d_bonds
    V = np.zeros((D, D), dtype=complex)

    def embed_site_full(op_site, site_idx):
        return np.kron(embed_qubit_op(op_site, site_idx, n_sites), np.eye(d_bonds, dtype=complex))

    def embed_bond_full(op_bond, bond_idx):
        return np.kron(np.eye(d_sites, dtype=complex), embed_bond_op(op_bond, bond_idx, n_bonds))

    for b, (u, v) in enumerate(spec.edges):
        term_x = embed_site_full(X, u) @ embed_bond_full(X, b) @ embed_site_full(X, v)
        term_z = embed_site_full(Z, u) @ embed_bond_full(Z, b) @ embed_site_full(Z, v)
        V += term_x + term_z

    return V


# ----------------------------
# SW effective Hamiltonian on bonds at order n (clean)
# ----------------------------

def build_projectors(n_sites, n_bonds):
    d_sites = 2**n_sites
    d_bonds = 2**n_bonds
    D = d_sites * d_bonds

    psi0 = np.zeros(d_sites, dtype=complex)
    psi0[0] = 1.0
    P_sites = np.outer(psi0, psi0.conj())
    P = np.kron(P_sites, np.eye(d_bonds, dtype=complex))
    Q = np.eye(D, dtype=complex) - P
    return P, Q


def resolvent_Q(E0, H0_full, Q):
    D = H0_full.shape[0]
    Q_idx = np.where(np.abs(np.diag(Q)) > 0.5)[0]
    if Q_idx.size == 0:
        raise RuntimeError("Q subspace empty")

    QH0Q = H0_full[np.ix_(Q_idx, Q_idx)]
    A = (E0 * np.eye(Q_idx.size, dtype=complex) - QH0Q)

    evals, evecs = eigh(A)
    inv = np.zeros_like(A)
    for i, lam in enumerate(evals):
        if abs(lam) < 1e-12:
            continue
        inv += (1.0 / lam) * np.outer(evecs[:, i], evecs[:, i].conj())

    G = np.zeros((D, D), dtype=complex)
    G[np.ix_(Q_idx, Q_idx)] = inv
    return G


def heff_order_on_bonds(spec: GraphSpec, gaps, g=0.3, order=None):
    n_sites = spec.n_sites
    n_bonds = spec.n_bonds
    d_sites = 2**n_sites
    d_bonds = 2**n_bonds

    if order is None:
        order = spec.cycle_len

    H0_sites = build_H0_sites(n_sites, gaps)
    H0_full = np.kron(H0_sites, np.eye(d_bonds, dtype=complex))
    V_full = build_V_sites_bonds(spec, n_sites, n_bonds)

    P, Q = build_projectors(n_sites, n_bonds)
    E0 = -float(np.sum(gaps))

    G = resolvent_Q(E0, H0_full, Q)

    V = g * V_full
    PVQ = P @ V @ Q
    QVP = Q @ V @ P
    QVQ = Q @ V @ Q

    if order == 1:
        Heff_full = P @ V @ P
    elif order == 2:
        Heff_full = PVQ @ G @ QVP
    else:
        T = G.copy()
        for _ in range(order - 2):
            T = T @ (QVQ @ G)
        Heff_full = PVQ @ T @ QVP

    # bond-only block (site ground sector)
    Hb = Heff_full[:d_bonds, :d_bonds].copy()
    Hb = (Hb + Hb.conj().T) / 2.0
    Hb_id = Hb - (np.trace(Hb) / d_bonds) * np.eye(d_bonds, dtype=complex)
    return Hb, Hb_id


# ----------------------------
# Pauli decomposition on bonds (loop extraction)
# ----------------------------

def pauli_strings(n_bonds):
    labels = ["I", "X", "Y", "Z"]
    mats = [I2, X, Y, Z]
    out = []
    for idx in range(4**n_bonds):
        tmp = idx
        ops = []
        lab = []
        for _ in range(n_bonds):
            tmp, r = divmod(tmp, 4)
            ops.append(mats[r])
            lab.append(labels[r])
        out.append(("".join(reversed(lab)), kron_all(list(reversed(ops)))))
    return out


def pauli_coeffs(Hb, n_bonds):
    d = 2**n_bonds
    coeffs = {}
    for lab, Pm in pauli_strings(n_bonds):
        c = np.trace(Pm.conj().T @ Hb) / d
        if abs(c) > 1e-12:
            coeffs[lab] = complex(c)
    return coeffs


def operator_weight(label):
    return sum(1 for ch in label if ch != "I")


def is_loop_support(label, spec: GraphSpec):
    # For these graphs, the plaquette is the full edge set.
    return operator_weight(label) == spec.cycle_len and all(label[i] != "I" for i in range(spec.n_bonds))


def loop_operator_from_coeffs(coeffs, spec: GraphSpec):
    n_bonds = spec.n_bonds
    d = 2**n_bonds
    Pmap = {lab: op for lab, op in pauli_strings(n_bonds)}

    Hloop = np.zeros((d, d), dtype=complex)
    loop_pow = 0.0
    total_pow = 0.0

    for lab, c in coeffs.items():
        total_pow += (abs(c) ** 2)
        if is_loop_support(lab, spec):
            Hloop += c * Pmap[lab]
            loop_pow += (abs(c) ** 2)

    loop_amp = math.sqrt(loop_pow) if loop_pow > 0 else 0.0
    loop_frac = (loop_pow / total_pow) if total_pow > 0 else 0.0
    return Hloop, loop_amp, loop_frac


# ----------------------------
# Gauss generators + constraint operator on bonds
# ----------------------------

def gauss_generators(spec: GraphSpec):
    """
    Naive Gauss generators on bond Hilbert space:
      G_s^a = Σ_{b incident on s} sign(s,b) * σ_b^a
    with sign +1 at first endpoint, -1 at second (edge orientation).
    """
    n_bonds = spec.n_bonds
    inc = spec.incidence()

    G = {}
    for s in range(spec.n_sites):
        for ax, op in [("X", X), ("Y", Y), ("Z", Z)]:
            M = np.zeros((2**n_bonds, 2**n_bonds), dtype=complex)
            for (b, sign) in inc[s]:
                M += sign * embed_bond_op(op, b, n_bonds)
            G[(s, ax)] = M
    return G


def constraint_operator_C(spec: GraphSpec, G):
    C = np.zeros((2**spec.n_bonds, 2**spec.n_bonds), dtype=complex)
    for M in G.values():
        C += M @ M
    C = (C + C.conj().T) / 2.0
    return C


def physical_projector_from_C(C, eps_abs=1e-10, fallback_k=4):
    """
    Define P_phys as exact kernel of C (eigs <= eps_abs). If empty, use lowest-k eigvecs.
    """
    evals, evecs = eigh(C)
    idx = np.argsort(evals.real)
    evals = evals.real[idx]
    evecs = evecs[:, idx]

    mask = evals <= eps_abs
    k = int(np.sum(mask))
    if k == 0:
        k = min(fallback_k, C.shape[0])
        mode = f"approx_lowest_{k}"
        vecs = evecs[:, :k]
    else:
        mode = f"exact_kernel_{k}"
        vecs = evecs[:, mask]

    P = vecs @ vecs.conj().T
    P = (P + P.conj().T) / 2.0
    return P, evals, k, mode


def leakage_ratio(H, P):
    I = np.eye(H.shape[0], dtype=complex)
    Q = I - P
    num = norm(Q @ H @ P, ord="fro")
    den = norm(H, ord="fro") + 1e-30
    return float(num / den)


# ----------------------------
# Ground-state overlap test
# ----------------------------

def ground_state_full(H):
    evals, evecs = eigh(H)
    i0 = int(np.argmin(evals.real))
    return float(evals.real[i0]), evecs[:, i0]


def bond_reduced_from_state(psi_full, n_sites, n_bonds):
    d_sites = 2**n_sites
    d_bonds = 2**n_bonds
    rho = np.outer(psi_full, psi_full.conj())
    rho_b = partial_trace_sites(rho, d_sites, d_bonds)
    rho_b = (rho_b + rho_b.conj().T) / 2.0
    return rho_b


# ----------------------------
# Main report
# ----------------------------

def report(spec: GraphSpec, g=0.3, gaps=None, eps_abs=1e-10, fallback_k=4):
    if gaps is None:
        gaps = [2.0] * spec.n_sites

    print("\n" + "=" * 78)
    print(f"{spec.name} — Path A tests (Gauss subspace + invariance)")
    print("=" * 78)
    print(f"sites={spec.n_sites} bonds={spec.n_bonds} cycle_len={spec.cycle_len} g={g} gaps={gaps}")

    n_sites = spec.n_sites
    n_bonds = spec.n_bonds
    d_bonds = 2**n_bonds

    # Full Hamiltonian
    H0_sites = build_H0_sites(n_sites, gaps)
    H0_full = np.kron(H0_sites, np.eye(d_bonds, dtype=complex))
    V_full = build_V_sites_bonds(spec, n_sites, n_bonds)
    H_full = H0_full + g * V_full

    # Constraint on bonds
    G = gauss_generators(spec)
    C = constraint_operator_C(spec, G)
    P_phys, evalsC, k_used, mode = physical_projector_from_C(C, eps_abs=eps_abs, fallback_k=fallback_k)

    print(f"\nC spectrum: min={evalsC[0]:.6g} next={evalsC[1]:.6g} median={np.median(evalsC):.6g} max={evalsC[-1]:.6g}")
    print(f"P_phys: {mode} dim={k_used}/{d_bonds}  eps_abs={eps_abs:g}")

    # Ground state overlap
    E0, psi0 = ground_state_full(H_full)
    rho_b = bond_reduced_from_state(psi0, n_sites, n_bonds)

    expC = float(np.trace(rho_b @ C).real)
    overlap = float(np.trace(rho_b @ P_phys).real)

    print(f"\nFull H ground energy: {E0:.6f}")
    print(f"Bond <C>:              {expC:.6e}")
    print(f"Bond overlap Tr(ρ P):  {overlap:.6f}")

    # B1 loop term (order = cycle length)
    _, Hb_id = heff_order_on_bonds(spec, gaps=gaps, g=g, order=spec.cycle_len)
    coeffs = pauli_coeffs(Hb_id, n_bonds)
    Hloop, loop_amp, loop_frac = loop_operator_from_coeffs(coeffs, spec)

    leak_loop = leakage_ratio(Hloop, P_phys)
    leak_all = leakage_ratio(Hb_id, P_phys)

    print(f"\nB1 SW order n={spec.cycle_len} (identity-subtracted):")
    print(f"  loop_amp (RMS): {loop_amp:.6e}")
    print(f"  loop_frac:      {loop_frac:.6f}")
    print(f"  leakage(loop):  {leak_loop:.6f}")
    print(f"  leakage(all):   {leak_all:.6f}")

    loop_terms = [(lab, c) for lab, c in coeffs.items() if is_loop_support(lab, spec)]
    loop_terms.sort(key=lambda t: abs(t[1]), reverse=True)
    print("\nTop loop-support Pauli terms:")
    if loop_terms:
        for lab, c in loop_terms[:10]:
            print(f"  {lab}: {c.real:+.6e}{c.imag:+.6e}i |c|={abs(c):.6e}")
    else:
        print("  (none above tolerance)")

    return {
        "spec": spec.name,
        "mode": mode,
        "k_used": k_used,
        "E0": E0,
        "expC": expC,
        "overlap": overlap,
        "loop_amp": loop_amp,
        "loop_frac": loop_frac,
        "leak_loop": leak_loop,
        "leak_all": leak_all,
        "evalsC": evalsC,
    }


def main():
    g = 0.3
    gaps_tri = [2.0, 2.0, 2.0]
    gaps_sqr = [2.0, 2.0, 2.0, 2.0]

    # eps_abs: if exact kernel exists, it'll be picked up here.
    # fallback_k: if no exact kernel, pick lowest-k eigenvectors of C as "near-Gauss" surface.
    eps_abs = 1e-10
    fallback_k = 4

    rep_tri = report(TRI, g=g, gaps=gaps_tri, eps_abs=eps_abs, fallback_k=fallback_k)
    rep_sqr = report(SQR, g=g, gaps=gaps_sqr, eps_abs=eps_abs, fallback_k=fallback_k)

    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gauss_subspace_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for rep in (rep_tri, rep_sqr):
            f.write(f"{rep['spec']}\n")
            f.write(f"  P_phys={rep['mode']} dim={rep['k_used']}\n")
            f.write(f"  E0={rep['E0']:.6f}\n")
            f.write(f"  <C>={rep['expC']:.6e}  overlap={rep['overlap']:.6f}\n")
            f.write(f"  loop_amp={rep['loop_amp']:.6e} loop_frac={rep['loop_frac']:.6f}\n")
            f.write(f"  leakage(loop)={rep['leak_loop']:.6f} leakage(all)={rep['leak_all']:.6f}\n")
            f.write(f"  C_min={rep['evalsC'][0]:.6g}  C_max={rep['evalsC'][-1]:.6g}\n")
            f.write("\n")

    print(f"\n[saved summary] {out_path}")


if __name__ == "__main__":
    main()
