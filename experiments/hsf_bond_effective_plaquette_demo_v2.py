#!/usr/bin/env python
"""
HSF Task B1 — Effective Bond Hamiltonian & Plaquette Emergence (Clean Rewrite v2)
================================================================================

This script is a *single, self-contained* replacement for the older B1 scripts.

What it does (honestly, and in the right order):
  1) Builds a small "site + bond" echo lattice (triangle, square) with d_B=2 bonds.
  2) Defines H = H0 + V where:
        H0 = - Σ_i Δ_i σz(site_i)           (|0...0> is the site ground sector)
        V  = g Σ_edges [ σx(a)⊗Bx(edge)⊗σx(b) + σz(a)⊗Bz(edge)⊗σz(b) ]
  3) Defines the *correct* projector:
        P = |0...0><0...0|_sites ⊗ I_bonds
        Q = I - P
  4) Computes:
        (a) Order-by-order Schrieffer–Wolff / resolvent expansion contributions
            that return to P only after n V-insertions:
               H_eff^(1) = P V P
               H_eff^(2) = P V Q G Q V P
               H_eff^(n) = P V Q (G Q V Q)^{n-2} G Q V P   for n>=2
            where G = (E0 I_Q - Q H0 Q)^(-1) on the Q-subspace.
        (b) "Exact" all-orders effective Hamiltonian:
               H_eff_exact = P H P + P V Q (E0 I_Q - Q H Q)^(-1) Q V P
            constructed *strictly* on Q indices to avoid P-contamination.
  5) Decomposes bond-space operators into Pauli strings (d_B=2) and:
        - subtracts the identity component before reporting weights
        - reports connected multi-bond weight fractions
        - detects loop terms (triangle: weight-3 loop, square: weight-4 loop)
  6) Saves a single figure into ./hsf_out next to this file.

Key rewrite vs your previous demo:
  - Identity component is removed before Pauli weight reporting (no more "99% weight-0" confusion).
  - "Exact" resolvent is computed on the Q-subspace indices only (clean).
  - Prints top loop-like Pauli strings and their coefficients.

Run:
  python hsf_bond_effective_plaquette_demo_v2.py

"""

import os
from datetime import datetime
from itertools import product as iprod

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=140)


# ------------------------------------------------------------
# Basic operators (qubits)
# ------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


# ------------------------------------------------------------
# Helpers: Pauli-string decomposition on n qubits (bonds)
# ------------------------------------------------------------
def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def hamming_weight(label: str) -> int:
    return sum(1 for ch in label if ch != "I")


def decompose_in_pauli(H: np.ndarray, n_qubits: int) -> dict:
    """
    Decompose a 2^n x 2^n Hermitian operator into Pauli strings.
    Returns dict label -> coefficient, with the convention:
       H = Σ_label c_label P_label
    where c_label = Tr(P_label H) / 2^n (since Tr(P_a P_b) = 2^n δ_ab).
    """
    d = 2 ** n_qubits
    assert H.shape == (d, d)
    coeffs = {}

    labels = list(iprod("IXYZ", repeat=n_qubits))
    for tup in labels:
        label = "".join(tup)
        Pmats = [PAULI[ch] for ch in label]
        P = kron_all(Pmats)
        c = np.trace(P.conj().T @ H) / d
        # keep only meaningful
        if abs(c) > 1e-12:
            coeffs[label] = c
    return coeffs


def subtract_identity(H: np.ndarray) -> np.ndarray:
    d = H.shape[0]
    return H - (np.trace(H) / d) * np.eye(d, dtype=complex)


# ------------------------------------------------------------
# Echo lattice: sites are qubits, bonds are qubits (d_B=2)
# ------------------------------------------------------------
class EchoLattice:
    """
    Basis ordering:
      full_index = site_index * d_bonds + bond_index
    where:
      site_index in [0, 2^n_sites)
      bond_index in [0, 2^n_bonds)
    """

    def __init__(self, n_sites: int, edges: list[tuple[int, int]], d_B: int = 2):
        assert d_B == 2, "This B1 demo supports d_B=2 bonds (Pauli decomposition)."
        self.n_sites = n_sites
        self.edges = edges  # list of (site_a, site_b) for each bond index
        self.n_bonds = len(edges)
        self.d_B = d_B

        self.d_sites_total = 2 ** n_sites
        self.d_bonds_total = d_B ** self.n_bonds
        self.total_dim = self.d_sites_total * self.d_bonds_total

    def embed_site_op(self, op2: np.ndarray, site_i: int) -> np.ndarray:
        mats = []
        for i in range(self.n_sites):
            mats.append(op2 if i == site_i else I2)
        op_sites = kron_all(mats)
        return np.kron(op_sites, np.eye(self.d_bonds_total, dtype=complex))

    def embed_bond_op(self, op2: np.ndarray, bond_i: int) -> np.ndarray:
        mats = []
        for b in range(self.n_bonds):
            mats.append(op2 if b == bond_i else I2)
        op_bonds = kron_all(mats)
        return np.kron(np.eye(self.d_sites_total, dtype=complex), op_bonds)

    def edge_coupling(self, bond_i: int) -> np.ndarray:
        """
        V_edge = X(site_a) ⊗ X(bond_i) ⊗ X(site_b) + Z(site_a) ⊗ Z(bond_i) ⊗ Z(site_b)
        """
        a, b = self.edges[bond_i]
        term_x = self.embed_site_op(X, a) @ self.embed_bond_op(X, bond_i) @ self.embed_site_op(X, b)
        term_z = self.embed_site_op(Z, a) @ self.embed_bond_op(Z, bond_i) @ self.embed_site_op(Z, b)
        return term_x + term_z

    def build_V(self, g: float) -> np.ndarray:
        V = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        for bi in range(self.n_bonds):
            V += g * self.edge_coupling(bi)
        return V

    def build_H0(self, gaps: list[float]) -> np.ndarray:
        """
        H0 = - Σ_i Δ_i Z(site_i)   so |0...0> has energy E0 = -Σ Δ_i
        """
        assert len(gaps) == self.n_sites
        H0 = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        for i, dlt in enumerate(gaps):
            H0 += (-dlt) * self.embed_site_op(Z, i)
        return H0


# ------------------------------------------------------------
# Projectors and subspace indexing
# ------------------------------------------------------------
def projector_P_Q(lattice: EchoLattice):
    """
    P = |0...0><0...0|_sites ⊗ I_bonds
    Q = I - P
    """
    d_s = lattice.d_sites_total
    d_b = lattice.d_bonds_total
    D = lattice.total_dim

    psi0_sites = np.zeros(d_s, dtype=complex)
    psi0_sites[0] = 1.0  # |0...0>
    P_sites = np.outer(psi0_sites, psi0_sites.conj())

    P = np.kron(P_sites, np.eye(d_b, dtype=complex))
    Q = np.eye(D, dtype=complex) - P

    # Q-subspace indices (basis states not in site_index=0 sector)
    # With our ordering, indices 0..d_b-1 correspond to site_index=0.
    Q_idx = np.arange(D, dtype=int)[d_b:]
    P_idx = np.arange(d_b, dtype=int)
    return P, Q, P_idx, Q_idx


# ------------------------------------------------------------
# Clean resolvents (diagonal H0 and full QHQ)
# ------------------------------------------------------------
def resolvent_G_from_H0(lattice: EchoLattice, H0: np.ndarray, E0: float, Q_idx: np.ndarray) -> np.ndarray:
    """
    G = (E0 I_Q - Q H0 Q)^(-1) on Q-subspace.
    Since H0 is diagonal in computational basis here, we can invert elementwise.
    """
    D = lattice.total_dim
    G = np.zeros((D, D), dtype=complex)

    diag = np.diag(H0).real
    for i in Q_idx:
        denom = (E0 - diag[i])
        if abs(denom) > 1e-12:
            G[i, i] = 1.0 / denom
    return G


def resolvent_R_from_QHQ(E0: float, QHQ: np.ndarray, Q_idx: np.ndarray) -> np.ndarray:
    """
    Build R_Q = (E0 I_Q - QHQ_Q)^(-1) strictly on Q indices.
    Returns full-size D×D matrix with only Q×Q block populated.
    """
    D = QHQ.shape[0]
    QHQ_Q = QHQ[np.ix_(Q_idx, Q_idx)]
    evals, evecs = np.linalg.eigh(QHQ_Q)

    # Invert (E0 - evals)
    inv = np.zeros_like(evals, dtype=float)
    for k, lam in enumerate(evals):
        denom = (E0 - lam)
        inv[k] = 0.0 if abs(denom) < 1e-12 else (1.0 / denom)

    R_Q = (evecs * inv) @ evecs.conj().T  # evecs diag(inv) evecs†

    R_full = np.zeros((D, D), dtype=complex)
    R_full[np.ix_(Q_idx, Q_idx)] = R_Q
    return R_full


# ------------------------------------------------------------
# Effective Hamiltonians
# ------------------------------------------------------------
def Heff_orders(lattice: EchoLattice, gaps: list[float], g: float, max_order: int):
    """
    Returns dict order -> Heff^(order) on bond Hilbert space (d_b×d_b),
    using the clean chain:
      Heff^(1) = P V P
      Heff^(2) = P V Q G Q V P
      Heff^(n) = P V Q (G Q V Q)^{n-2} G Q V P
    """
    H0 = lattice.build_H0(gaps)
    V = lattice.build_V(g)

    P, Q, P_idx, Q_idx = projector_P_Q(lattice)
    E0 = -sum(gaps)  # energy of |0...0> under H0

    G = resolvent_G_from_H0(lattice, H0, E0, Q_idx)

    # Precompute blocks
    PVQ = P @ V @ Q
    QVP = Q @ V @ P
    QVQ = Q @ V @ Q

    # "A" operator inside the chain
    A = G @ QVQ

    results = {}
    for n in range(1, max_order + 1):
        if n == 1:
            Heff_full = P @ V @ P
        elif n == 2:
            Heff_full = PVQ @ G @ QVP
        else:
            # PVQ * (A)^(n-2) * G * QVP
            mid = np.eye(lattice.total_dim, dtype=complex)
            for _ in range(n - 2):
                mid = mid @ A
            Heff_full = PVQ @ mid @ G @ QVP

        # Restrict to bond block (site ground sector corresponds to first d_b indices)
        d_b = lattice.d_bonds_total
        Hb = Heff_full[np.ix_(P_idx, P_idx)].copy()
        Hb = (Hb + Hb.conj().T) / 2.0
        results[n] = Hb

    return results


def Heff_exact(lattice: EchoLattice, gaps: list[float], g: float):
    """
    Exact all-orders effective Hamiltonian:
      Heff = P H P + P V Q (E0 I_Q - Q H Q)^(-1) Q V P
    computed with a clean Q-subspace resolvent.
    """
    H0 = lattice.build_H0(gaps)
    V = lattice.build_V(g)
    H = H0 + V

    P, Q, P_idx, Q_idx = projector_P_Q(lattice)
    E0 = -sum(gaps)

    QHQ = Q @ H @ Q
    R = resolvent_R_from_QHQ(E0, QHQ, Q_idx)

    PHP = P @ H @ P
    PVQ = P @ V @ Q
    QVP = Q @ V @ P

    Heff_full = PHP + PVQ @ R @ QVP
    d_b = lattice.d_bonds_total
    Hb = Heff_full[np.ix_(P_idx, P_idx)].copy()
    Hb = (Hb + Hb.conj().T) / 2.0
    return Hb


# ------------------------------------------------------------
# Loop/plaquette detection on bond graph
# ------------------------------------------------------------
def is_loop_term(label: str, edges: list[tuple[int, int]]) -> bool:
    active = [i for i, ch in enumerate(label) if ch != "I"]
    if len(active) < 3:
        return False

    site_count = {}
    for bi in active:
        a, b = edges[bi]
        site_count[a] = site_count.get(a, 0) + 1
        site_count[b] = site_count.get(b, 0) + 1

    # Loop criterion: each site appears exactly twice and number of sites equals number of edges in active set
    # (simple cycle)
    if not all(v == 2 for v in site_count.values()):
        return False
    if len(site_count) != len(active):
        return False
    return True


def summarize_pauli_weights(Hb: np.ndarray, n_bonds: int, edges: list[tuple[int, int]], top_k: int = 10):
    """
    Returns:
      weights: dict weight -> fraction (%), excluding identity shift (trace removed)
      loop_terms: list of (label, coeff) with loop-like structure
    """
    Hb2 = subtract_identity(Hb)
    coeffs = decompose_in_pauli(Hb2, n_bonds)

    total = sum((abs(c) ** 2) for c in coeffs.values())
    if total < 1e-30:
        return {}, [], {}

    weight_power = {}
    loop_terms = []
    for lab, c in coeffs.items():
        w = hamming_weight(lab)
        weight_power[w] = weight_power.get(w, 0.0) + (abs(c) ** 2)
        if is_loop_term(lab, edges):
            loop_terms.append((lab, c))

    # Convert to percentages
    weights_pct = {w: 100.0 * p / total for w, p in weight_power.items()}

    # Top terms for debugging
    top_terms = sorted(coeffs.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    top_terms = {lab: c for lab, c in top_terms}

    loop_terms = sorted(loop_terms, key=lambda kv: -abs(kv[1]))[:top_k]
    return weights_pct, loop_terms, top_terms


# ------------------------------------------------------------
# Main demo (triangle + square)
# ------------------------------------------------------------
def run_graph(name: str, n_sites: int, edges: list[tuple[int, int]], g: float, max_order: int):
    lattice = EchoLattice(n_sites=n_sites, edges=edges, d_B=2)

    # Slightly non-degenerate gaps help avoid accidental degeneracy in denominators
    gaps = [2.0 + 0.37 * i for i in range(n_sites)]

    # Order-by-order
    orders = Heff_orders(lattice, gaps=gaps, g=g, max_order=max_order)

    # Detect first order where a loop term appears (by Pauli decomposition after identity subtraction)
    first_loop_order = None
    norms = []
    loop_power = []

    for n in range(1, max_order + 1):
        Hb = orders[n]
        norms.append(np.linalg.norm(subtract_identity(Hb)))

        w_pct, loop_terms, _ = summarize_pauli_weights(Hb, lattice.n_bonds, edges)
        lp = 0.0
        if loop_terms:
            # compute loop power fraction
            Hb2 = subtract_identity(Hb)
            coeffs = decompose_in_pauli(Hb2, lattice.n_bonds)
            total = sum(abs(c) ** 2 for c in coeffs.values())
            loop_sum = 0.0
            for lab, c in coeffs.items():
                if is_loop_term(lab, edges):
                    loop_sum += abs(c) ** 2
            lp = (loop_sum / total) if total > 1e-30 else 0.0

            if first_loop_order is None and lp > 1e-6:
                first_loop_order = n
        loop_power.append(lp)

    # Exact
    Hb_exact = Heff_exact(lattice, gaps=gaps, g=g)
    w_exact, loop_terms_exact, top_terms_exact = summarize_pauli_weights(Hb_exact, lattice.n_bonds, edges)

    print(f"\n{'='*78}")
    print(f"{name.upper()}  (sites={n_sites}, bonds={len(edges)}, g={g})")
    print(f"{'='*78}")
    print(f"  Gaps Δ_i: {gaps}")
    print(f"  Exact Heff ||·|| (identity-subtracted): {np.linalg.norm(subtract_identity(Hb_exact)):.6e}")

    # Print order table
    print("\n  Order-by-order (identity-subtracted norms and loop fraction):")
    print(f"  {'n':>3}  {'||Heff^(n)||':>14}  {'loop_frac':>10}")
    print("  " + "-" * 34)
    for n in range(1, max_order + 1):
        print(f"  {n:>3}  {norms[n-1]:>14.6e}  {loop_power[n-1]:>10.3e}")

    if first_loop_order is None:
        print("\n  First loop: not detected (loop_frac never exceeded threshold)")
    else:
        print(f"\n  First loop detected at perturbation order n = {first_loop_order}")

    # Exact weight distribution
    print("\n  Exact Heff (identity-subtracted) Pauli weight distribution:")
    for w in sorted(w_exact.keys()):
        print(f"    weight {w}: {w_exact[w]:6.2f}%")

    if loop_terms_exact:
        print("\n  Exact loop-like terms (top):")
        for lab, c in loop_terms_exact[:8]:
            print(f"    {lab}: {c.real:+.6e}{c.imag:+.6e}j  |c|={abs(c):.3e}")
    else:
        print("\n  Exact loop-like terms: none above threshold (after identity subtraction).")

    print("\n  Exact top Pauli terms (after identity subtraction):")
    for lab, c in list(top_terms_exact.items())[:8]:
        tag = " ← LOOP" if is_loop_term(lab, edges) else ""
        print(f"    {lab}: {c.real:+.6e}{c.imag:+.6e}j  |c|={abs(c):.3e}{tag}")

    return {
        "name": name,
        "orders": list(range(1, max_order + 1)),
        "norms": norms,
        "loop_power": loop_power,
        "exact_weights": w_exact,
        "first_loop_order": first_loop_order if first_loop_order is not None else -1,
    }


def make_figure(results_tri, results_sq, out_png: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: norms vs order
    ax = axes[0]
    ax.semilogy(results_tri["orders"], [max(x, 1e-18) for x in results_tri["norms"]], marker="o", label="Triangle")
    ax.semilogy(results_sq["orders"], [max(x, 1e-18) for x in results_sq["norms"]], marker="s", label="Square")
    ax.set_title("Identity-subtracted ||H_eff^(n)|| vs order")
    ax.set_xlabel("Perturbation order n")
    ax.set_ylabel("||H_eff^(n) - tr/ d · I||")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Mark first-loop order if detected
    if results_tri["first_loop_order"] > 0:
        ax.axvline(results_tri["first_loop_order"], linestyle=":", alpha=0.5)
        ax.annotate("triangle loop", xy=(results_tri["first_loop_order"], results_tri["norms"][results_tri["first_loop_order"]-1]),
                    xytext=(results_tri["first_loop_order"]+0.3, results_tri["norms"][results_tri["first_loop_order"]-1]*4),
                    arrowprops=dict(arrowstyle="->"))
    if results_sq["first_loop_order"] > 0:
        ax.axvline(results_sq["first_loop_order"], linestyle=":", alpha=0.5)
        ax.annotate("square loop", xy=(results_sq["first_loop_order"], results_sq["norms"][results_sq["first_loop_order"]-1]),
                    xytext=(results_sq["first_loop_order"]+0.3, results_sq["norms"][results_sq["first_loop_order"]-1]*4),
                    arrowprops=dict(arrowstyle="->"))

    # Panel 2: loop fraction vs order
    ax = axes[1]
    ax.plot(results_tri["orders"], results_tri["loop_power"], marker="o", label="Triangle")
    ax.plot(results_sq["orders"], results_sq["loop_power"], marker="s", label="Square")
    ax.set_title("Loop power fraction vs order (Pauli, identity-subtracted)")
    ax.set_xlabel("Perturbation order n")
    ax.set_ylabel("Σ_loop |c|^2 / Σ_all |c|^2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 3: exact weight distributions (excluding identity already)
    ax = axes[2]
    # collect up to weight 4
    weights = sorted(set(results_tri["exact_weights"].keys()) | set(results_sq["exact_weights"].keys()))
    tri_vals = [results_tri["exact_weights"].get(w, 0.0) for w in weights]
    sq_vals = [results_sq["exact_weights"].get(w, 0.0) for w in weights]
    x = np.arange(len(weights))
    width = 0.35
    ax.bar(x - width/2, tri_vals, width, label="Triangle")
    ax.bar(x + width/2, sq_vals, width, label="Square")
    ax.set_xticks(x, [str(w) for w in weights])
    ax.set_title("Exact H_eff Pauli weights (identity-subtracted)")
    ax.set_xlabel("Operator weight (# active bonds)")
    ax.set_ylabel("Weight fraction (%)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.suptitle("HSF Task B1 — Clean Plaquette Extraction (v2)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[saved figure] {out_png}")


def main():
    g = 0.3
    max_order = 8

    tri = run_graph("Triangle", n_sites=3, edges=[(0, 1), (1, 2), (0, 2)], g=g, max_order=max_order)
    sq  = run_graph("Square",   n_sites=4, edges=[(0, 1), (1, 2), (2, 3), (0, 3)], g=g, max_order=max_order)

    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"bond_effective_plaquette_demo_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    make_figure(tri, sq, out_png)


if __name__ == "__main__":
    main()
