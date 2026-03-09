#!/usr/bin/env python
"""
HSF Task B1 — Locking Down the Plaquette Mechanism (v3)
=======================================================

This script is a *single, self-contained* demonstration that:
  1) Implements a clean Schrieffer–Wolff / resolvent expansion for the bond-only
     effective Hamiltonian with the correct ground sector and projector.
  2) Detects loop/plaquette operators in the bond Hamiltonian via Pauli-string
     decomposition (d_B=2).
  3) *Locks the mechanism down* by verifying the strong-coupling scalings:
        loop_amplitude ∝ g^n
        loop_amplitude ∝ Δ^{-(n-1)}
     where n is the cycle length (triangle: n=3, square: n=4).

What you get (paper-grade diagnostics):
  - Order-of-appearance: first loop power shows at n=cycle length.
  - Loop isolation: identity-subtracted Pauli decomposition.
  - Scaling fits: log-log slopes vs g and vs Δ-scale.
  - Prints top loop Pauli strings at the first loop order.
  - Saves a single figure into ./hsf_out next to this file.

Run:
  python hsf_bond_effective_plaquette_demo_v3.py

"""

import os
from datetime import datetime
from itertools import product as iprod

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=140)

# -------------------------
# Qubit operators
# -------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def hamming_weight(label: str) -> int:
    return sum(1 for ch in label if ch != "I")


def subtract_identity(H: np.ndarray) -> np.ndarray:
    d = H.shape[0]
    return H - (np.trace(H) / d) * np.eye(d, dtype=complex)


def decompose_in_pauli(H: np.ndarray, n_qubits: int) -> dict:
    """Pauli decomposition for n-qubit operator (Hermitian assumed).

    Convention:
      H = Σ c_label P_label
      c_label = Tr(P_label H) / 2^n
    """
    d = 2 ** n_qubits
    assert H.shape == (d, d)

    coeffs = {}
    for tup in iprod("IXYZ", repeat=n_qubits):
        label = "".join(tup)
        Pmats = [PAULI[ch] for ch in label]
        P = kron_all(Pmats)
        c = np.trace(P.conj().T @ H) / d
        if abs(c) > 1e-14:
            coeffs[label] = c
    return coeffs


# -------------------------
# Echo lattice (sites qubits, bonds qubits)
# -------------------------
class EchoLattice:
    """Basis ordering: full_index = site_index * d_bonds + bond_index."""

    def __init__(self, n_sites: int, edges: list[tuple[int, int]]):
        self.n_sites = n_sites
        self.edges = edges
        self.n_bonds = len(edges)

        self.d_sites_total = 2 ** n_sites
        self.d_bonds_total = 2 ** self.n_bonds
        self.total_dim = self.d_sites_total * self.d_bonds_total

    def embed_site_op(self, op2: np.ndarray, site_i: int) -> np.ndarray:
        mats = [(op2 if i == site_i else I2) for i in range(self.n_sites)]
        op_sites = kron_all(mats)
        return np.kron(op_sites, np.eye(self.d_bonds_total, dtype=complex))

    def embed_bond_op(self, op2: np.ndarray, bond_i: int) -> np.ndarray:
        mats = [(op2 if b == bond_i else I2) for b in range(self.n_bonds)]
        op_bonds = kron_all(mats)
        return np.kron(np.eye(self.d_sites_total, dtype=complex), op_bonds)

    def edge_coupling(self, bond_i: int) -> np.ndarray:
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
        """H0 = - Σ Δ_i Z(site_i) so |0...0> is ground with E0=-ΣΔ."""
        assert len(gaps) == self.n_sites
        H0 = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        for i, dlt in enumerate(gaps):
            H0 += (-dlt) * self.embed_site_op(Z, i)
        return H0


def projector_P_Q(lattice: EchoLattice):
    """P = |0...0><0...0|_sites ⊗ I_bonds; Q = I-P; plus index lists."""
    d_s = lattice.d_sites_total
    d_b = lattice.d_bonds_total
    D = lattice.total_dim

    psi0 = np.zeros(d_s, dtype=complex)
    psi0[0] = 1.0
    P_sites = np.outer(psi0, psi0.conj())
    P = np.kron(P_sites, np.eye(d_b, dtype=complex))
    Q = np.eye(D, dtype=complex) - P

    P_idx = np.arange(d_b, dtype=int)
    Q_idx = np.arange(D, dtype=int)[d_b:]
    return P, Q, P_idx, Q_idx


def resolvent_G_from_H0(lattice: EchoLattice, H0: np.ndarray, E0: float, Q_idx: np.ndarray) -> np.ndarray:
    """G = (E0 I_Q - Q H0 Q)^(-1) on Q-subspace; H0 diagonal => invert elementwise."""
    D = lattice.total_dim
    G = np.zeros((D, D), dtype=complex)
    diag = np.diag(H0).real
    for i in Q_idx:
        denom = (E0 - diag[i])
        if abs(denom) > 1e-14:
            G[i, i] = 1.0 / denom
    return G


def Heff_orders(lattice: EchoLattice, gaps: list[float], g: float, max_order: int) -> dict[int, np.ndarray]:
    """Clean SW/resolvent chain returning bond-space Heff^(n)."""
    H0 = lattice.build_H0(gaps)
    V = lattice.build_V(g)

    P, Q, P_idx, Q_idx = projector_P_Q(lattice)
    E0 = -sum(gaps)
    G = resolvent_G_from_H0(lattice, H0, E0, Q_idx)

    PVQ = P @ V @ Q
    QVP = Q @ V @ P
    QVQ = Q @ V @ Q
    A = G @ QVQ

    results = {}
    for n in range(1, max_order + 1):
        if n == 1:
            Heff_full = P @ V @ P
        elif n == 2:
            Heff_full = PVQ @ G @ QVP
        else:
            mid = np.eye(lattice.total_dim, dtype=complex)
            for _ in range(n - 2):
                mid = mid @ A
            Heff_full = PVQ @ mid @ G @ QVP

        d_b = lattice.d_bonds_total
        Hb = Heff_full[np.ix_(P_idx, P_idx)].copy()
        Hb = (Hb + Hb.conj().T) / 2.0
        results[n] = Hb

    return results


# -------------------------
# Loop detection and extraction
# -------------------------
def is_simple_cycle(active_bonds: list[int], edges: list[tuple[int, int]]) -> bool:
    if len(active_bonds) < 3:
        return False
    site_count = {}
    for bi in active_bonds:
        a, b = edges[bi]
        site_count[a] = site_count.get(a, 0) + 1
        site_count[b] = site_count.get(b, 0) + 1
    # each site degree 2 and #sites == #edges (cycle)
    if not all(v == 2 for v in site_count.values()):
        return False
    if len(site_count) != len(active_bonds):
        return False
    return True


def is_loop_term(label: str, edges: list[tuple[int, int]]) -> bool:
    active = [i for i, ch in enumerate(label) if ch != "I"]
    return is_simple_cycle(active, edges)


def loop_amplitude_from_pauli(Hb: np.ndarray, n_bonds: int, edges: list[tuple[int, int]]):
    """Return (loop_amp, total_amp, loop_frac, top_loop_terms, top_terms).

    loop_amp := sqrt( Σ_loop |c|^2 ) for identity-subtracted Hb.
    total_amp := sqrt( Σ_all |c|^2 ) for identity-subtracted Hb.
    """
    Hb2 = subtract_identity(Hb)
    coeffs = decompose_in_pauli(Hb2, n_bonds)

    total_pow = 0.0
    loop_pow = 0.0
    loop_terms = []

    for lab, c in coeffs.items():
        p = abs(c) ** 2
        total_pow += p
        if is_loop_term(lab, edges):
            loop_pow += p
            loop_terms.append((lab, c))

    total_amp = np.sqrt(total_pow) if total_pow > 0 else 0.0
    loop_amp = np.sqrt(loop_pow) if loop_pow > 0 else 0.0
    loop_frac = (loop_pow / total_pow) if total_pow > 1e-30 else 0.0

    loop_terms = sorted(loop_terms, key=lambda kv: -abs(kv[1]))
    top_terms = sorted(coeffs.items(), key=lambda kv: -abs(kv[1]))

    return loop_amp, total_amp, loop_frac, loop_terms[:12], top_terms[:12]


# -------------------------
# Scaling fits
# -------------------------
def fit_loglog(x: np.ndarray, y: np.ndarray):
    """Fit y ~ a x^m in log-log; returns (m, a)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan
    m, b = np.polyfit(np.log(x), np.log(y), 1)
    a = np.exp(b)
    return m, a


# -------------------------
# Main “lock it down” routine
# -------------------------
def analyze_graph(name: str, n_sites: int, edges: list[tuple[int, int]], cycle_len: int,
                  g0: float = 0.3, max_order: int = 8,
                  g_sweep=(0.08, 0.12, 0.18, 0.27, 0.40, 0.60),
                  delta_scales=(0.8, 1.0, 1.3, 1.7, 2.2)):

    lattice = EchoLattice(n_sites=n_sites, edges=edges)

    base_gaps = [2.0 + 0.37 * i for i in range(n_sites)]

    # --- Order of appearance check at g0
    orders = Heff_orders(lattice, gaps=base_gaps, g=g0, max_order=max_order)
    order_norms = []
    order_loop_frac = []
    for n in range(1, max_order + 1):
        Hb = orders[n]
        loop_amp, total_amp, loop_frac, _, _ = loop_amplitude_from_pauli(Hb, lattice.n_bonds, edges)
        order_norms.append(np.linalg.norm(subtract_identity(Hb)))
        order_loop_frac.append(loop_frac)

    # capture top loop terms at the *first* cycle_len order
    loop_amp_n, total_amp_n, loop_frac_n, loop_terms_n, top_terms_n = loop_amplitude_from_pauli(
        orders[cycle_len], lattice.n_bonds, edges
    )

    print(f"\n{'='*88}\n{name}  (cycle_len={cycle_len}, sites={n_sites}, bonds={len(edges)})\n{'='*88}")
    print(f"Base gaps Δ_i: {base_gaps}  (E0 = {-sum(base_gaps):.3f})")
    print(f"At g={g0}: loop_frac by order (1..{max_order}):")
    print("  " + ", ".join([f"{lf:.3f}" for lf in order_loop_frac]))
    print(f"First-cycle order n={cycle_len}: loop_amp={loop_amp_n:.3e}, total_amp={total_amp_n:.3e}, loop_frac={loop_frac_n:.3f}")

    print("\nTop LOOP Pauli strings at first-cycle order (identity-subtracted):")
    if not loop_terms_n:
        print("  (none)")
    else:
        for lab, c in loop_terms_n[:8]:
            print(f"  {lab}: {c.real:+.3e}{c.imag:+.3e}j  |c|={abs(c):.3e}")

    # --- g scaling: loop_amp at n=cycle_len vs g
    loop_vs_g = []
    for g in g_sweep:
        Hb = Heff_orders(lattice, gaps=base_gaps, g=g, max_order=cycle_len)[cycle_len]
        loop_amp, _, _, _, _ = loop_amplitude_from_pauli(Hb, lattice.n_bonds, edges)
        loop_vs_g.append(loop_amp)

    slope_g, a_g = fit_loglog(np.array(g_sweep), np.array(loop_vs_g))

    # --- Δ scaling: scale all gaps by s, loop_amp at fixed g0
    loop_vs_ds = []
    for s in delta_scales:
        gaps = [s * x for x in base_gaps]
        Hb = Heff_orders(lattice, gaps=gaps, g=g0, max_order=cycle_len)[cycle_len]
        loop_amp, _, _, _, _ = loop_amplitude_from_pauli(Hb, lattice.n_bonds, edges)
        loop_vs_ds.append(loop_amp)

    slope_d, a_d = fit_loglog(np.array(delta_scales), np.array(loop_vs_ds))

    print(f"\nScaling fits for loop amplitude (RMS loop coeffs, identity-subtracted):")
    print(f"  loop_amp vs g: slope ≈ {slope_g:.3f}  (expected {cycle_len})")
    print(f"  loop_amp vs Δ_scale: slope ≈ {slope_d:.3f}  (expected {-(cycle_len-1)})")

    return {
        "name": name,
        "cycle_len": cycle_len,
        "orders": np.arange(1, max_order + 1),
        "order_norms": np.array(order_norms, dtype=float),
        "order_loop_frac": np.array(order_loop_frac, dtype=float),
        "g_sweep": np.array(g_sweep, dtype=float),
        "loop_vs_g": np.array(loop_vs_g, dtype=float),
        "g_slope": float(slope_g),
        "delta_scales": np.array(delta_scales, dtype=float),
        "loop_vs_ds": np.array(loop_vs_ds, dtype=float),
        "d_slope": float(slope_d),
    }


def make_figure(res_tri, res_sq, out_png: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: order-of-appearance diagnostics
    for col, res in enumerate([res_tri, res_sq]):
        ax0 = axes[0, col]
        ax0.semilogy(res["orders"], np.maximum(res["order_norms"], 1e-18), marker="o")
        ax0.set_title(f"{res['name']}: ||H_eff^(n)|| (id-sub)")
        ax0.set_xlabel("order n")
        ax0.set_ylabel("norm")
        ax0.grid(True, alpha=0.3)
        ax0.axvline(res["cycle_len"], linestyle=":", alpha=0.6)

        ax1 = axes[0, col + 1] if col == 0 else axes[0, col]  # no-op, kept for readability

    # Replace with dedicated loop-fraction plot across both
    ax = axes[0, 2]
    ax.plot(res_tri["orders"], res_tri["order_loop_frac"], marker="o", label="Triangle")
    ax.plot(res_sq["orders"], res_sq["order_loop_frac"], marker="s", label="Square")
    ax.set_title("Loop power fraction vs order")
    ax.set_xlabel("order n")
    ax.set_ylabel("Σ_loop |c|^2 / Σ_all |c|^2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Row 2: scaling fits
    ax = axes[1, 0]
    ax.loglog(res_tri["g_sweep"], np.maximum(res_tri["loop_vs_g"], 1e-18), marker="o", label=f"Triangle slope {res_tri['g_slope']:.2f}")
    ax.loglog(res_sq["g_sweep"], np.maximum(res_sq["loop_vs_g"], 1e-18), marker="s", label=f"Square slope {res_sq['g_slope']:.2f}")
    ax.set_title("Loop amplitude vs g (log-log)")
    ax.set_xlabel("g")
    ax.set_ylabel("loop_amp")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.loglog(res_tri["delta_scales"], np.maximum(res_tri["loop_vs_ds"], 1e-18), marker="o", label=f"Triangle slope {res_tri['d_slope']:.2f}")
    ax.loglog(res_sq["delta_scales"], np.maximum(res_sq["loop_vs_ds"], 1e-18), marker="s", label=f"Square slope {res_sq['d_slope']:.2f}")
    ax.set_title("Loop amplitude vs Δ_scale (log-log)")
    ax.set_xlabel("Δ_scale")
    ax.set_ylabel("loop_amp")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    ax.axis("off")
    text = (
        "MECHANISM LOCKDOWN\n\n"
        "Compute bond-only Heff via clean SW/resolvent chain:\n"
        "  H0 = -Σ Δ_i Z_i  (|0..0> is ground)\n"
        "  P = |0..0><0..0| ⊗ I_bonds\n"
        "  Heff^(n) = P V Q (G QVQ)^(n-2) G Q V P\n\n"
        "Loop amplitude extracted as RMS of loop Pauli coeffs\n"
        "after identity subtraction.\n\n"
        "Expected strong-coupling scalings:\n"
        "  triangle (n=3): loop_amp ∝ g^3 · Δ^-2\n"
        "  square   (n=4): loop_amp ∝ g^4 · Δ^-3\n"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#fff8d5", alpha=0.9))

    plt.suptitle("HSF Task B1 — Plaquette Mechanism Lockdown (v3)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[saved figure] {out_png}")


def main():
    # Triangle: 3-cycle
    res_tri = analyze_graph(
        name="Triangle",
        n_sites=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        cycle_len=3,
        g0=0.3,
        max_order=8,
        g_sweep=(0.06, 0.09, 0.14, 0.21, 0.32, 0.48),
        delta_scales=(0.7, 1.0, 1.4, 2.0, 2.8),
    )

    # Square: 4-cycle
    res_sq = analyze_graph(
        name="Square",
        n_sites=4,
        edges=[(0, 1), (1, 2), (2, 3), (0, 3)],
        cycle_len=4,
        g0=0.3,
        max_order=8,
        g_sweep=(0.06, 0.09, 0.14, 0.21, 0.32, 0.48),
        delta_scales=(0.7, 1.0, 1.4, 2.0, 2.8),
    )

    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"bond_effective_plaquette_demo_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    make_figure(res_tri, res_sq, out_png)


if __name__ == "__main__":
    main()
