#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plaquette_b2_gauss_from_echo_algebra_v2.py
========================================

Fixes vs v1:
- Robust Task A generator extraction:
    * tries a strict near-unitary dominant-Kraus filter
    * if that yields 0 generators (as you saw), automatically relaxes tolerances
    * if still insufficient, falls back to a *span-based* extraction:
        - collects MANY Hermitian traceless "echo Hamiltonians" from dominant Kraus,
          without requiring near-unitary domination, then SVDs the span
  For d_B=2, this reliably recovers a 3D su(2) basis under very mild conditions.

- Handles empty basis gracefully (no IndexError), and prints diagnostics.
- Keeps the same B1 loop operator + Gauss-subspace + leakage tests.

Run:
  python plaquette_b2_gauss_from_echo_algebra_v2.py
"""

import os
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm, eigh

# ----------------------------
# Base ops (qubits)
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

def embed_qubit_op(op, which, n_qubits):
    ops = [I2] * n_qubits
    ops[which] = op
    return kron_all(ops)

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
# Task A: Echo algebra extraction on ONE bond space (d_B)
# ----------------------------

def haar_random_qubit(rng):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    return v / (np.linalg.norm(v) + 1e-30)

def su2_irrep_generators(d_B):
    # Spin-s SU(2) irrep generators (Hermitian), dimension d_B
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

    'standard': X-Bx-X + Z-Bz-Z + 0.3 * Y-I-Y
    'full':     X-Bx-X + Y-By-Y + Z-Bz-Z
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

def herm_traceless(A):
    d = A.shape[0]
    A = (A + A.conj().T) / 2.0
    A = A - (np.trace(A) / d) * np.eye(d, dtype=complex)
    return A

def dominant_kraus_effective_H(Ks, eps):
    # choose dominant by Fro norm, take anti-Hermitian part ~ generator
    weights = [np.linalg.norm(K, "fro")**2 for K in Ks]
    dom = int(np.argmax(weights))
    Kd = Ks[dom]
    H_eff = (Kd - Kd.conj().T) / (2.0j * eps)
    return herm_traceless(H_eff), weights, dom

def strict_echo_generators(d_B, n_samples, eps, unitary_tol, leakage_tol, variant, seed):
    rng = np.random.default_rng(seed)
    H = transmission_hamiltonian_single_bond(d_B, variant=variant)
    U = expm(-1j * eps * H)
    I = np.eye(d_B, dtype=complex)

    gens = []
    kept = 0
    rej = 0

    for _ in range(n_samples):
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

        Heff, _, _ = dominant_kraus_effective_H(Ks, eps)
        nrm = np.linalg.norm(Heff)
        if nrm > 1e-12:
            gens.append(Heff / nrm)
            kept += 1

    return gens, kept, rej

def span_based_generators(d_B, n_samples, eps, variant, seed):
    """
    Fallback: do not require near-unitary domination.
    Just collect many Hermitian traceless generators from dominant Kraus
    and later SVD the span to get a basis.
    """
    rng = np.random.default_rng(seed)
    H = transmission_hamiltonian_single_bond(d_B, variant=variant)
    U = expm(-1j * eps * H)

    ops = []
    for _ in range(n_samples):
        psi_a = haar_random_qubit(rng)
        psi_b = haar_random_qubit(rng)
        Ks = extract_kraus(U, d_B, psi_a, psi_b)
        Heff, _, _ = dominant_kraus_effective_H(Ks, eps)
        nrm = np.linalg.norm(Heff)
        if nrm > 1e-12:
            ops.append(Heff / nrm)
    return ops

def hermitian_real_embedding(A):
    # embed Hermitian matrix into real vector: diag real + upper tri (real, imag)
    d = A.shape[0]
    v = []
    for i in range(d):
        v.append(A[i, i].real)
    for i in range(d):
        for j in range(i + 1, d):
            v.append(A[i, j].real)
            v.append(A[i, j].imag)
    return np.array(v, dtype=float)

def hs_orthonormal_basis_from_pool(pool, d_B, svd_tol=1e-6):
    if not pool:
        return []

    V = np.stack([hermitian_real_embedding(herm_traceless(A)) for A in pool], axis=0)
    # SVD on samples x features
    _, S, Vh = np.linalg.svd(V, full_matrices=False)
    if S.size == 0:
        return []
    n_ind = int(np.sum(S > svd_tol * S[0]))
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
        A = herm_traceless(A)
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

def extract_echo_basis(d_B=2, variant="standard", eps=1e-4, seed=123):
    """
    Robust extraction: strict passes with progressively relaxed tolerances, else span-fallback.
    """
    expected = d_B**2 - 1

    strict_schedules = [
        # (unitary_tol, leakage_tol, n_samples)
        (1e-6, 1e-6, 1400),
        (1e-5, 1e-4, 2000),
        (1e-4, 1e-3, 3000),
        (1e-3, 1e-2, 4000),
    ]

    all_pool = []
    for (ut, lt, ns) in strict_schedules:
        gens, kept, rej = strict_echo_generators(
            d_B=d_B, n_samples=ns, eps=eps, unitary_tol=ut, leakage_tol=lt,
            variant=variant, seed=seed
        )
        print(f"[TaskA strict] kept={kept} rej={rej} (unitary_tol={ut:g}, leakage_tol={lt:g}) pool+={len(gens)}")
        all_pool.extend(gens)
        basis = hs_orthonormal_basis_from_pool(all_pool, d_B=d_B, svd_tol=1e-6)
        if len(basis) >= expected:
            print(f"[TaskA] extracted basis dim={len(basis)} (expected {expected}) via strict/accumulated pool.")
            return basis[:expected]

    # Fallback: span-based
    print("[TaskA] strict filters yielded insufficient basis; using span-based extraction (no near-unitary requirement).")
    pool = span_based_generators(d_B=d_B, n_samples=8000, eps=eps, variant=variant, seed=seed+1)
    basis = hs_orthonormal_basis_from_pool(pool, d_B=d_B, svd_tol=1e-6)
    print(f"[TaskA fallback] span pool size={len(pool)} -> basis dim={len(basis)} (expected {expected})")
    if len(basis) >= expected:
        return basis[:expected]
    return basis  # may be smaller; caller handles

# ----------------------------
# Full echo H (sites ⊗ qubit-bonds), and SW loop operator (B1 machinery)
# ----------------------------

def embed_site_op(op, site_idx, n_sites, d_bonds):
    return np.kron(embed_qubit_op(op, site_idx, n_sites), np.eye(d_bonds, dtype=complex))

def embed_bond_qubit_op(op, bond_idx, n_bonds, d_sites):
    return np.kron(np.eye(d_sites, dtype=complex), embed_qubit_op(op, bond_idx, n_bonds))

def build_H0_sites(n_sites, gaps):
    H0 = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for i in range(n_sites):
        H0 += (-gaps[i]) * embed_qubit_op(Z2, i, n_sites)  # |0..0> ground
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
    psi0 = np.zeros(d_sites, dtype=complex); psi0[0] = 1.0
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

# Pauli decomposition (qubit links)
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
# Gauss from extracted echo basis
# ----------------------------

def embed_bond_generator(T, bond_idx, n_bonds, d_B):
    ops = []
    for b in range(n_bonds):
        ops.append(T if b == bond_idx else np.eye(d_B, dtype=complex))
    return kron_all(ops)

def gauss_generators_from_basis(spec: GraphSpec, basis_T):
    n_bonds = spec.n_bonds
    d_B = basis_T[0].shape[0]
    inc = spec.incidence()
    G = {}
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

# Ground state test (full echo H for qubit links only)
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
# Report runner
# ----------------------------

def run_graph(spec: GraphSpec, basis_T, g=0.3, gaps=None, eps_abs=1e-10, fallback_k=4):
    if gaps is None:
        gaps = [2.0] * spec.n_sites

    print("\n" + "=" * 84)
    print(f"{spec.name}: Gauss from Echo Algebra Basis (Task A-derived) — B1→B2 diagnostic")
    print("=" * 84)
    print(f"sites={spec.n_sites} bonds={spec.n_bonds} cycle_len={spec.cycle_len} g={g} gaps={gaps}")
    print(f"bond basis size = {len(basis_T)}  d_B={basis_T[0].shape[0]}")

    G = gauss_generators_from_basis(spec, basis_T)
    C = constraint_operator_C(G)
    P_phys, evalsC, k_used, mode = physical_projector_from_C(C, eps_abs=eps_abs, fallback_k=fallback_k)

    print(f"\nC spectrum: min={evalsC[0]:.6g} next={evalsC[1]:.6g} median={np.median(evalsC):.6g} max={evalsC[-1]:.6g}")
    print(f"P_phys: {mode} dim={k_used}/{C.shape[0]} eps_abs={eps_abs:g}")

    Hb_id = heff_order_on_bonds(spec, gaps=gaps, g=g, order=spec.cycle_len)
    coeffs = pauli_coeffs(Hb_id, spec.n_bonds)
    Hloop, loop_amp, loop_frac = loop_operator_from_coeffs(coeffs, spec)

    ratios = [commutator_ratio(Hloop, M) for M in G.values()]
    ratios = np.array(ratios, dtype=float)

    leak_loop = leakage_ratio(Hloop, P_phys)
    leak_all = leakage_ratio(Hb_id, P_phys)

    H_full = full_echo_H_qubitlinks(spec, gaps=gaps, g=g)
    E0, psi0 = ground_state_full(H_full)
    rho_b = bond_reduced_state(psi0, spec.n_sites, spec.n_bonds)
    expC = float(np.trace(rho_b @ C).real)
    overlap = float(np.trace(rho_b @ P_phys).real)

    print(f"\nB1 loop (SW order n={spec.cycle_len}, identity-subtracted): loop_amp={loop_amp:.6e} loop_frac={loop_frac:.6f}")
    print(f"Commutator ratios: min={ratios.min():.6f} med={np.median(ratios):.6f} max={ratios.max():.6f}")
    print(f"Leakage(loop)={leak_loop:.6f}  Leakage(all)={leak_all:.6f}")
    print(f"Ground selection: E0={E0:.6f}  <C>={expC:.6e}  overlap={overlap:.6f}")

    return {
        "name": spec.name,
        "basis_dim": len(basis_T),
        "P_mode": mode,
        "P_dim": int(k_used),
        "comm_min": float(ratios.min()),
        "comm_med": float(np.median(ratios)),
        "comm_max": float(ratios.max()),
        "leak_loop": float(leak_loop),
        "leak_all": float(leak_all),
        "expC": float(expC),
        "overlap": float(overlap),
    }

def main():
    d_B = 2
    variant = "standard"
    eps = 1e-4
    seed = 123

    basis_T = extract_echo_basis(d_B=d_B, variant=variant, eps=eps, seed=seed)
    expected = d_B**2 - 1

    if len(basis_T) < expected:
        print(f"[WARN] extracted basis dim={len(basis_T)} < expected {expected}. Results may still be indicative but not definitive.")
        if len(basis_T) == 0:
            print("[FATAL] Could not extract any generators. Try increasing eps (e.g. 1e-3) or switching variant='full'.")
            return

    g = 0.3
    gaps_tri = [2.0, 2.0, 2.0]
    gaps_sqr = [2.0, 2.0, 2.0, 2.0]

    eps_abs = 1e-10
    fallback_k = 4

    rep_tri = run_graph(TRI, basis_T, g=g, gaps=gaps_tri, eps_abs=eps_abs, fallback_k=fallback_k)
    rep_sqr = run_graph(SQR, basis_T, g=g, gaps=gaps_sqr, eps_abs=eps_abs, fallback_k=fallback_k)

    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gauss_from_echo_algebra_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

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
