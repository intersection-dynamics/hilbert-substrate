#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plaquette_b2_gauss_from_echo_algebra.py
======================================

Goal:
Derive Gauss generators from the *actual extracted echo algebra* (Task A),
instead of assuming naive Pauli sums.

Pipeline:
1) Extract bond-algebra generators {T^a} on bond Hilbert space C^{d_B}
   using strict "dominant Kraus + near-unitary" filter.
2) Orthonormalize in Hilbert-Schmidt inner product.
3) Build Gauss generators on a multi-bond graph:
      G_s^a = sum_{b incident to s} sign(s,b) * T_b^a
4) Build constraint operator C = sum_{s,a} (G_s^a)^2 and form P_phys from low-C subspace.
5) Compute:
   - commutator ratios with H_loop (B1 SW loop term at order = cycle length)
   - ground state overlap of full echo Hamiltonian with P_phys (bond-reduced)
   - leakage of H_loop across P_phys boundary.

This is the decisive "does gauge invariance exist in the algebra you actually have?" test.

Default uses d_B=2 (qubit links), matching your current B1 loop extraction assumptions.
"""

import os
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm, eigh

# ----------------------------
# Base operators
# ----------------------------

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def partial_trace_sites(rho, d_sites, d_bonds):
    rho = rho.reshape(d_sites, d_bonds, d_sites, d_bonds)
    out = np.zeros((d_bonds, d_bonds), dtype=complex)
    for i in range(d_sites):
        out += rho[i, :, i, :]
    return out


# ----------------------------
# Graph specs
# ----------------------------

@dataclass
class GraphSpec:
    name: str
    n_sites: int
    edges: list  # list of tuples (u, v), bond index = position
    cycle_len: int

    @property
    def n_bonds(self):
        return len(self.edges)

    def incidence(self):
        inc = [[] for _ in range(self.n_sites)]
        for b, (u, v) in enumerate(self.edges):
            inc[u].append((b, +1))  # orient u -> v
            inc[v].append((b, -1))
        return inc


TRI = GraphSpec("Triangle", 3, [(0, 1), (1, 2), (0, 2)], 3)
SQR = GraphSpec("Square", 4, [(0, 1), (1, 2), (2, 3), (0, 3)], 4)


# ----------------------------
# Task A: Extract echo algebra generators on a single bond
# ----------------------------

def haar_random_qubit(rng):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    return v / (np.linalg.norm(v) + 1e-30)


def su2_irrep_generators(d_B):
    """
    SU(2) spin-s irrep generators (Hermitian), dimension d_B.
    For d_B=2 this matches Pauli/2 up to convention.
    """
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


def transmission_hamiltonian_single_bond(d_B, variant="standard"):
    """
    H on C^2 ⊗ C^{d_B} ⊗ C^2.
    'standard' matches your usual: X-Bx-X + Z-Bz-Z + (small) Y-I-Y.
    """
    sx, sy, sz = X2, Y2, Z2
    IB = np.eye(d_B, dtype=complex)

    Bx, By, Bz = su2_irrep_generators(d_B)

    if variant == "standard":
        H = (np.kron(np.kron(sx, Bx), sx) +
             np.kron(np.kron(sz, Bz), sz) +
             0.3 * np.kron(np.kron(sy, IB), sy))
    elif variant == "full":
        H = (np.kron(np.kron(sx, Bx), sx) +
             np.kron(np.kron(sy, By), sy) +
             np.kron(np.kron(sz, Bz), sz))
    else:
        raise ValueError("variant must be 'standard' or 'full'")
    return H


def extract_kraus(U_full, d_B, psi_a, psi_b):
    """
    Kraus operators on bond space induced by U_full and projecting sites onto computational basis.
    """
    d_full = 2 * d_B * 2
    embed = np.zeros((d_full, d_B), dtype=complex)
    for b in range(d_B):
        for a in range(2):
            for c in range(2):
                row = a * (d_B * 2) + b * 2 + c
                embed[row, b] = psi_a[a] * psi_b[c]
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


def strict_echo_generators(d_B, n_site_samples=1200, eps=1e-4,
                           unitary_tol=1e-6, leakage_tol=1e-6,
                           variant="standard", seed=0):
    """
    Strict generator extraction:
    - keep samples where sum K†K ≈ I and one Kraus dominates
    - define effective Hermitian generator from dominant Kraus's anti-Hermitian part
    Returns list of Hermitian traceless generators (not yet orthonormal).
    """
    rng = np.random.default_rng(seed)
    H = transmission_hamiltonian_single_bond(d_B, variant=variant)
    U = expm(-1j * eps * H)
    I = np.eye(d_B, dtype=complex)

    gens = []
    kept = 0
    rej = 0

    for _ in range(n_site_samples):
        psi_a = haar_random_qubit(rng)
        psi_b = haar_random_qubit(rng)
        Ks = extract_kraus(U, d_B, psi_a, psi_b)

        comp = np.zeros((d_B, d_B), dtype=complex)
        weights = []
        for K in Ks:
            comp += K.conj().T @ K
            weights.append(np.linalg.norm(K, "fro")**2)

        comp_err = np.linalg.norm(comp - I)
        dom = int(np.argmax(weights))
        leakage = 1.0 - (weights[dom] / (sum(weights) + 1e-30))

        if comp_err > unitary_tol or leakage > leakage_tol:
            rej += 1
            continue

        Kd = Ks[dom]
        # Hermitian generator from dominant Kraus
        H_eff = (Kd - Kd.conj().T) / (2.0j * eps)
        H_eff = (H_eff + H_eff.conj().T) / 2.0
        H_eff -= (np.trace(H_eff) / d_B) * I

        nrm = np.linalg.norm(H_eff)
        if nrm > 1e-12:
            gens.append(H_eff / nrm)
            kept += 1

    print(f"[TaskA] strict samples kept={kept} rejected={rej} (unitary_tol={unitary_tol}, leakage_tol={leakage_tol})")
    return gens


def hs_orthonormal_basis(ops, d_B, tol=1e-6):
    """
    From a pool of Hermitian traceless ops, extract an HS-orthonormal basis
    using SVD on real coordinate embedding (robust).
    """
    if not ops:
        return []

    # Real coordinate embedding of Hermitian matrices:
    # diag real + upper triangle real/imag
    vecs = []
    for A in ops:
        A = (A + A.conj().T) / 2.0
        A -= (np.trace(A) / d_B) * np.eye(d_B, dtype=complex)
        v = []
        for i in range(d_B):
            v.append(A[i, i].real)
        for i in range(d_B):
            for j in range(i + 1, d_B):
                v.append(A[i, j].real)
                v.append(A[i, j].imag)
        vecs.append(v)

    V = np.array(vecs, dtype=float)
    _, S, Vh = np.linalg.svd(V, full_matrices=False)
    n_ind = int(np.sum(S > tol * (S[0] if S.size else 1.0)))
    basis_vecs = Vh[:n_ind]

    basis = []
    for v in basis_vecs:
        A = np.zeros((d_B, d_B), dtype=complex)
        idx = 0
        for i in range(d_B):
            A[i, i] = v[idx]
            idx += 1
        for i in range(d_B):
            for j in range(i + 1, d_B):
                A[i, j] = v[idx] + 1j * v[idx + 1]
                A[j, i] = v[idx] - 1j * v[idx + 1]
                idx += 2
        A -= (np.trace(A) / d_B) * np.eye(d_B, dtype=complex)

        # HS normalize: <A,A> = Tr(A†A)
        hs = np.trace(A.conj().T @ A).real
        if hs > 1e-12:
            basis.append(A / math.sqrt(hs))

    # Gram-Schmidt polish
    ortho = []
    for A in basis:
        B = A.copy()
        for P in ortho:
            B -= (np.trace(P.conj().T @ B).real) * P
        hs = np.trace(B.conj().T @ B).real
        if hs > 1e-12:
            ortho.append(B / math.sqrt(hs))

    return ortho


# ----------------------------
# Full echo Hamiltonian (sites ⊗ bonds) for d_B=2 only (current B1 machinery)
# ----------------------------

def embed_site_op(op, site_idx, n_sites, d_bonds):
    return np.kron(embed_qubit_op(op, site_idx, n_sites), np.eye(d_bonds, dtype=complex))


def embed_bond_qubit_op(op, bond_idx, n_bonds, d_sites):
    return np.kron(np.eye(d_sites, dtype=complex), embed_qubit_op(op, bond_idx, n_bonds))


def build_H0_sites(n_sites, gaps):
    H0 = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for i in range(n_sites):
        H0 += (-gaps[i]) * embed_qubit_op(Z2, i, n_sites)  # |0...0> ground
    return H0


def build_V_sites_bonds_qubitlinks(spec: GraphSpec, n_sites, n_bonds):
    d_sites = 2**n_sites
    d_bonds = 2**n_bonds
    D = d_sites * d_bonds
    V = np.zeros((D, D), dtype=complex)

    for b, (u, v) in enumerate(spec.edges):
        term_x = embed_site_op(X2, u, n_sites, d_bonds) @ embed_bond_qubit_op(X2, b, n_bonds, d_sites) @ embed_site_op(X2, v, n_sites, d_bonds)
        term_z = embed_site_op(Z2, u, n_sites, d_bonds) @ embed_bond_qubit_op(Z2, b, n_bonds, d_sites) @ embed_site_op(Z2, v, n_sites, d_bonds)
        V += term_x + term_z

    return V


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
    V_full = build_V_sites_bonds_qubitlinks(spec, n_sites, n_bonds)

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

    Hb = Heff_full[:d_bonds, :d_bonds].copy()
    Hb = (Hb + Hb.conj().T) / 2.0
    Hb_id = Hb - (np.trace(Hb) / d_bonds) * np.eye(d_bonds, dtype=complex)
    return Hb_id


# ----------------------------
# Pauli decomposition + loop extraction (qubit links)
# ----------------------------

def pauli_strings(n_bonds):
    labels = ["I", "X", "Y", "Z"]
    mats = [I2, X2, Y2, Z2]
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
    # on these graphs, a plaquette uses all bonds
    return operator_weight(label) == spec.cycle_len and all(label[i] != "I" for i in range(spec.n_bonds))


def loop_operator_from_coeffs(coeffs, spec: GraphSpec):
    n_bonds = spec.n_bonds
    Pmap = {lab: op for lab, op in pauli_strings(n_bonds)}
    d = 2**n_bonds

    Hloop = np.zeros((d, d), dtype=complex)
    loop_pow = 0.0
    tot_pow = 0.0

    for lab, c in coeffs.items():
        tot_pow += abs(c)**2
        if is_loop_support(lab, spec):
            Hloop += c * Pmap[lab]
            loop_pow += abs(c)**2

    loop_frac = (loop_pow / tot_pow) if tot_pow > 0 else 0.0
    loop_amp = math.sqrt(loop_pow) if loop_pow > 0 else 0.0
    return Hloop, loop_amp, loop_frac


# ----------------------------
# Gauss from extracted echo algebra basis (key change)
# ----------------------------

def embed_bond_generator(T, bond_idx, n_bonds, d_B):
    """
    Embed a d_B×d_B operator on one bond within (d_B^n_bonds) bond Hilbert space.
    """
    ops = []
    for b in range(n_bonds):
        ops.append(T if b == bond_idx else np.eye(d_B, dtype=complex))
    return kron_all(ops)


def gauss_generators_from_basis(spec: GraphSpec, basis_T):
    """
    G_s^a = Σ_{b incident} sign(s,b) * T_b^a
    basis_T: list of HS-orthonormal generators on single bond space (d_B×d_B)
    """
    n_bonds = spec.n_bonds
    d_B = basis_T[0].shape[0]
    inc = spec.incidence()

    G = {}  # key = (site, a_index)
    for s in range(spec.n_sites):
        for a, Ta in enumerate(basis_T):
            M = np.zeros((d_B**n_bonds, d_B**n_bonds), dtype=complex)
            for (b, sign) in inc[s]:
                M += sign * embed_bond_generator(Ta, b, n_bonds, d_B)
            G[(s, a)] = M
    return G


def constraint_operator_C(G):
    dim = next(iter(G.values())).shape[0]
    C = np.zeros((dim, dim), dtype=complex)
    for M in G.values():
        C += M @ M
    C = (C + C.conj().T) / 2.0
    return C


def physical_projector_from_C(C, eps_abs=1e-10, fallback_k=4):
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


def commutator_ratio(A, B):
    num = norm(A @ B - B @ A, ord="fro")
    den = (norm(A, ord="fro") * norm(B, ord="fro")) + 1e-30
    return float(num / den)


def leakage_ratio(H, P):
    I = np.eye(H.shape[0], dtype=complex)
    Q = I - P
    num = norm(Q @ H @ P, ord="fro")
    den = norm(H, ord="fro") + 1e-30
    return float(num / den)


# ----------------------------
# Ground state test (full echo H) for d_B=2 links
# ----------------------------

def ground_state_full(H):
    evals, evecs = eigh(H)
    i0 = int(np.argmin(evals.real))
    return float(evals.real[i0]), evecs[:, i0]


def bond_reduced_state(psi_full, n_sites, n_bonds):
    d_sites = 2**n_sites
    d_bonds = 2**n_bonds
    rho = np.outer(psi_full, psi_full.conj())
    rho_b = partial_trace_sites(rho, d_sites, d_bonds)
    rho_b = (rho_b + rho_b.conj().T) / 2.0
    return rho_b


def full_echo_H_qubitlinks(spec: GraphSpec, gaps, g=0.3):
    n_sites = spec.n_sites
    n_bonds = spec.n_bonds
    d_sites = 2**n_sites
    d_bonds = 2**n_bonds

    H0_sites = build_H0_sites(n_sites, gaps)
    H0_full = np.kron(H0_sites, np.eye(d_bonds, dtype=complex))
    V_full = build_V_sites_bonds_qubitlinks(spec, n_sites, n_bonds)
    return H0_full + g * V_full


# ----------------------------
# Main report
# ----------------------------

def run_graph(spec: GraphSpec, basis_T, g=0.3, gaps=None, eps_abs=1e-10, fallback_k=4):
    if gaps is None:
        gaps = [2.0] * spec.n_sites

    print("\n" + "=" * 84)
    print(f"{spec.name}: Gauss from Echo Algebra Basis (Task A-derived) — B1→B2 diagnostic")
    print("=" * 84)
    print(f"sites={spec.n_sites} bonds={spec.n_bonds} cycle_len={spec.cycle_len} g={g} gaps={gaps}")
    print(f"bond basis size (dim su(d_B)) = {len(basis_T)}  d_B={basis_T[0].shape[0]}")

    # Build Gauss from extracted basis
    G = gauss_generators_from_basis(spec, basis_T)
    C = constraint_operator_C(G)
    P_phys, evalsC, k_used, mode = physical_projector_from_C(C, eps_abs=eps_abs, fallback_k=fallback_k)

    print(f"\nC spectrum: min={evalsC[0]:.6g} next={evalsC[1]:.6g} median={np.median(evalsC):.6g} max={evalsC[-1]:.6g}")
    print(f"P_phys: {mode} dim={k_used}/{C.shape[0]} eps_abs={eps_abs:g}")

    # B1 loop operator at first plaquette order (cycle length), using qubit-link SW machinery
    Hb_id = heff_order_on_bonds(spec, gaps=gaps, g=g, order=spec.cycle_len)
    coeffs = pauli_coeffs(Hb_id, spec.n_bonds)
    Hloop, loop_amp, loop_frac = loop_operator_from_coeffs(coeffs, spec)

    print(f"\nB1 loop (SW order n={spec.cycle_len}, identity-subtracted):")
    print(f"  loop_amp (RMS): {loop_amp:.6e}")
    print(f"  loop_frac:      {loop_frac:.6f}")

    # Commutators with derived Gauss generators
    # (Report worst-case over all sites and a’s)
    ratios = []
    for (s, a), M in G.items():
        r = commutator_ratio(Hloop, M)
        ratios.append(r)
    ratios = np.array(ratios, dtype=float)
    print(f"\nCommutator ratios r = ||[Hloop,G]||/(||Hloop|| ||G||):")
    print(f"  min={ratios.min():.6f}  median={np.median(ratios):.6f}  max={ratios.max():.6f}")

    # Subspace invariance (leakage)
    leak_loop = leakage_ratio(Hloop, P_phys)
    leak_all = leakage_ratio(Hb_id, P_phys)
    print(f"\nLeakage ratios (subspace invariance):")
    print(f"  leakage(loop) = ||(I-P)Hloop P|| / ||Hloop||  = {leak_loop:.6f}")
    print(f"  leakage(all)  = ||(I-P)Heff  P|| / ||Heff||   = {leak_all:.6f}")

    # Ground state overlap test (full echo H, bond reduced)
    H_full = full_echo_H_qubitlinks(spec, gaps=gaps, g=g)
    E0, psi0 = ground_state_full(H_full)
    rho_b = bond_reduced_state(psi0, spec.n_sites, spec.n_bonds)
    expC = float(np.trace(rho_b @ C).real)
    overlap = float(np.trace(rho_b @ P_phys).real)

    print(f"\nGround state dynamical selection:")
    print(f"  full ground energy E0 = {E0:.6f}")
    print(f"  bond <C>              = {expC:.6e}")
    print(f"  bond overlap Tr(rho P)= {overlap:.6f}")

    return {
        "name": spec.name,
        "C_min": float(evalsC[0]),
        "C_max": float(evalsC[-1]),
        "P_mode": mode,
        "P_dim": int(k_used),
        "loop_amp": float(loop_amp),
        "loop_frac": float(loop_frac),
        "comm_min": float(ratios.min()),
        "comm_med": float(np.median(ratios)),
        "comm_max": float(ratios.max()),
        "leak_loop": float(leak_loop),
        "leak_all": float(leak_all),
        "E0": float(E0),
        "expC": float(expC),
        "overlap": float(overlap),
    }


def main():
    # Keep d_B=2 here because B1 loop extraction uses qubit-link SW & Pauli decomposition.
    # If you later generalize B1 loop extraction to d_B>2, this same Gauss pipeline works.
    d_B = 2

    # Task A extraction
    pool = strict_echo_generators(
        d_B=d_B,
        n_site_samples=1400,
        eps=1e-4,
        unitary_tol=1e-6,
        leakage_tol=1e-6,
        variant="standard",
        seed=123,
    )
    basis_T = hs_orthonormal_basis(pool, d_B=d_B, tol=1e-6)

    expected = d_B**2 - 1
    print(f"[TaskA] extracted HS-orthonormal basis size = {len(basis_T)} (expected {expected})")
    if len(basis_T) != expected:
        print("  [WARN] basis size mismatch; results may be inconclusive. Try increasing samples or loosening tol slightly.")

    g = 0.3
    gaps_tri = [2.0, 2.0, 2.0]
    gaps_sqr = [2.0, 2.0, 2.0, 2.0]

    eps_abs = 1e-10
    fallback_k = 4

    rep_tri = run_graph(TRI, basis_T, g=g, gaps=gaps_tri, eps_abs=eps_abs, fallback_k=fallback_k)
    rep_sqr = run_graph(SQR, basis_T, g=g, gaps=gaps_sqr, eps_abs=eps_abs, fallback_k=fallback_k)

    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gauss_from_echo_algebra_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for rep in (rep_tri, rep_sqr):
            f.write(rep["name"] + "\n")
            for k, v in rep.items():
                if k == "name":
                    continue
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
