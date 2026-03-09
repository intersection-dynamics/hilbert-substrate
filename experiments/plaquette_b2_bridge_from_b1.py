#!/usr/bin/env python
"""
plaquette_b2_bridge_from_b1.py
==============================

Purpose
-------
Bridge Task B1 -> Task B2 in a *single, self-contained* script.

B1 (already locked down):
  - Clean Schrieffer–Wolff / resolvent chain produces a bond-only H_eff^(n)
  - Loop/plaquette terms first appear at order n = cycle length
  - Loop amplitude scales as g^n / Δ^(n-1)

B2 (what this script tests/quantifies):
  1) Wilson-subspace structure:
     - Extract the identity-subtracted loop operator H_loop from H_eff^(n=cycle_len)
     - Decompose into Pauli strings (d_B=2 links)
     - Report fraction of loop power in the "magnetic/Wilson-like" XY subspace
       (loop terms with only X/Y on active bonds, no Z).

  2) (Approx) Gauge invariance diagnostics:
     - Define simple Gauss generators at each site:
         G_s^a = sum_{b incident on s} η_{s,b} σ_b^a
       with orientation signs η derived from edge ordering.
     - Compute commutator norm ratios:
         r = ||[H_loop, G_s^a]||_F / (||H_loop||_F * ||G_s^a||_F)
       Small r suggests gauge invariance under these generators.
       Large but structured r suggests gauge-fixed residual.

Notes
-----
- This is a *bridge* diagnostic, not a full lattice gauge theory proof.
- d_B=2 (bond qubits) is assumed for Pauli decomposition.

Run (Windows):
  python plaquette_b2_bridge_from_b1.py

Outputs:
  - Prints a compact B2 bridge report for triangle and square.
  - Saves a summary PNG into ./hsf_out next to this file.

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
PAULI_LABELS = ["I", "X", "Y", "Z"]


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def subtract_identity(H: np.ndarray) -> np.ndarray:
    d = H.shape[0]
    return H - (np.trace(H) / d) * np.eye(d, dtype=complex)


def decompose_in_pauli(H: np.ndarray, n_qubits: int, tol: float = 1e-14) -> dict:
    """Pauli decomposition for n-qubit operator.

    Convention:
      H = Σ c_label P_label
      c_label = Tr(P_label H) / 2^n
    """
    d = 2 ** n_qubits
    assert H.shape == (d, d)

    coeffs = {}
    for tup in iprod("IXYZ", repeat=n_qubits):
        label = "".join(tup)
        P = kron_all([PAULI[ch] for ch in label])
        c = np.trace(P.conj().T @ H) / d
        if abs(c) > tol:
            coeffs[label] = c
    return coeffs


def pauli_op_from_label(label: str) -> np.ndarray:
    return kron_all([PAULI[ch] for ch in label])


def fro_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


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

        # incident bonds per site
        self.incident = [[] for _ in range(n_sites)]
        for bi, (a, b) in enumerate(edges):
            self.incident[a].append(bi)
            self.incident[b].append(bi)

        # orientation sign η_{s,b}: +1 if s is "tail" (min endpoint), -1 if "head" (max endpoint)
        self.eta = np.zeros((n_sites, self.n_bonds), dtype=int)
        for bi, (a, b) in enumerate(edges):
            tail, head = (a, b) if a < b else (b, a)
            self.eta[tail, bi] = +1
            self.eta[head, bi] = -1

    def embed_site_op(self, op2: np.ndarray, site_i: int) -> np.ndarray:
        mats = [(op2 if i == site_i else I2) for i in range(self.n_sites)]
        op_sites = kron_all(mats)
        return np.kron(op_sites, np.eye(self.d_bonds_total, dtype=complex))

    def embed_bond_op(self, op2: np.ndarray, bond_i: int) -> np.ndarray:
        mats = [(op2 if b == bond_i else I2) for b in range(self.n_bonds)]
        op_bonds = kron_all(mats)
        return np.kron(np.eye(self.d_sites_total, dtype=complex), op_bonds)

    def edge_coupling(self, bond_i: int) -> np.ndarray:
        """Simple transmission edge term: X-X-X + Z-Z-Z across site-bond-site."""
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


def Heff_order(lattice: EchoLattice, gaps: list[float], g: float, order_n: int) -> np.ndarray:
    """Return bond-space Heff^(n) via clean SW/resolvent chain."""
    H0 = lattice.build_H0(gaps)
    V = lattice.build_V(g)

    P, Q, P_idx, Q_idx = projector_P_Q(lattice)
    E0 = -sum(gaps)
    G = resolvent_G_from_H0(lattice, H0, E0, Q_idx)

    PVQ = P @ V @ Q
    QVP = Q @ V @ P
    QVQ = Q @ V @ Q
    A = G @ QVQ

    if order_n == 1:
        Heff_full = P @ V @ P
    elif order_n == 2:
        Heff_full = PVQ @ G @ QVP
    else:
        mid = np.eye(lattice.total_dim, dtype=complex)
        for _ in range(order_n - 2):
            mid = mid @ A
        Heff_full = PVQ @ mid @ G @ QVP

    Hb = Heff_full[np.ix_(P_idx, P_idx)].copy()
    Hb = (Hb + Hb.conj().T) / 2.0
    return Hb


# -------------------------
# Loop detection
# -------------------------
def is_simple_cycle(active_bonds: list[int], edges: list[tuple[int, int]]) -> bool:
    if len(active_bonds) < 3:
        return False
    site_count = {}
    for bi in active_bonds:
        a, b = edges[bi]
        site_count[a] = site_count.get(a, 0) + 1
        site_count[b] = site_count.get(b, 0) + 1
    if not all(v == 2 for v in site_count.values()):
        return False
    if len(site_count) != len(active_bonds):
        return False
    return True


def is_loop_term(label: str, edges: list[tuple[int, int]]) -> bool:
    active = [i for i, ch in enumerate(label) if ch != "I"]
    return is_simple_cycle(active, edges)


def is_xy_only(label: str) -> bool:
    """True if all non-identity characters are X or Y."""
    for ch in label:
        if ch == "I":
            continue
        if ch not in ("X", "Y"):
            return False
    return True


def extract_loop_operator(Hb: np.ndarray, n_bonds: int, edges: list[tuple[int, int]], tol: float = 1e-14):
    """Return loop-only operator H_loop and coefficient dictionaries."""
    Hb2 = subtract_identity(Hb)
    coeffs = decompose_in_pauli(Hb2, n_bonds, tol=tol)

    loop_coeffs = {}
    xy_loop_coeffs = {}
    total_pow = 0.0
    loop_pow = 0.0
    xy_loop_pow = 0.0

    for lab, c in coeffs.items():
        p = abs(c) ** 2
        total_pow += p
        if is_loop_term(lab, edges):
            loop_coeffs[lab] = c
            loop_pow += p
            if is_xy_only(lab):
                xy_loop_coeffs[lab] = c
                xy_loop_pow += p

    # build operators
    d = 2 ** n_bonds
    H_loop = np.zeros((d, d), dtype=complex)
    H_xy = np.zeros((d, d), dtype=complex)
    for lab, c in loop_coeffs.items():
        H_loop += c * pauli_op_from_label(lab)
    for lab, c in xy_loop_coeffs.items():
        H_xy += c * pauli_op_from_label(lab)

    # enforce Hermitian (numerical)
    H_loop = (H_loop + H_loop.conj().T) / 2.0
    H_xy = (H_xy + H_xy.conj().T) / 2.0

    loop_amp = math.sqrt(loop_pow) if loop_pow > 0 else 0.0
    total_amp = math.sqrt(total_pow) if total_pow > 0 else 0.0
    loop_frac = (loop_pow / total_pow) if total_pow > 1e-30 else 0.0
    xy_frac_in_loop = (xy_loop_pow / loop_pow) if loop_pow > 1e-30 else 0.0

    top_loop = sorted(loop_coeffs.items(), key=lambda kv: -abs(kv[1]))[:12]
    top_xy = sorted(xy_loop_coeffs.items(), key=lambda kv: -abs(kv[1]))[:12]

    return {
        "Hb_idsub": Hb2,
        "coeffs": coeffs,
        "H_loop": H_loop,
        "H_xy_loop": H_xy,
        "loop_amp": loop_amp,
        "total_amp": total_amp,
        "loop_frac_total": loop_frac,
        "xy_frac_within_loop": xy_frac_in_loop,
        "top_loop_terms": top_loop,
        "top_xy_loop_terms": top_xy,
    }


# -------------------------
# Gauss generators and commutator tests
# -------------------------
def gauss_generators(lattice: EchoLattice):
    """Return dict: (site, axis) -> operator on bond space."""
    n = lattice.n_bonds
    d = 2 ** n
    gens = {}
    for s in range(lattice.n_sites):
        for axis, op in [("X", X), ("Y", Y), ("Z", Z)]:
            G = np.zeros((d, d), dtype=complex)
            for bi in lattice.incident[s]:
                sign = lattice.eta[s, bi]
                mats = [(op if b == bi else I2) for b in range(n)]
                G += sign * kron_all(mats)
            G = (G + G.conj().T) / 2.0
            gens[(s, axis)] = G
    return gens


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def comm_ratio(H: np.ndarray, G: np.ndarray) -> float:
    num = fro_norm(commutator(H, G))
    den = fro_norm(H) * fro_norm(G) + 1e-30
    return num / den


# -------------------------
# One graph report
# -------------------------
def report_graph(name: str, n_sites: int, edges: list[tuple[int, int]], cycle_len: int,
                 g: float = 0.3, gaps=None, tol: float = 1e-14):
    lattice = EchoLattice(n_sites=n_sites, edges=edges)
    if gaps is None:
        gaps = [2.0 + 0.37 * i for i in range(n_sites)]

    Hb = Heff_order(lattice, gaps=gaps, g=g, order_n=cycle_len)
    ex = extract_loop_operator(Hb, lattice.n_bonds, edges, tol=tol)

    H_loop = ex["H_loop"]
    H_xy = ex["H_xy_loop"]

    # commutator diagnostics
    gens = gauss_generators(lattice)
    ratios = {(s, a): comm_ratio(H_loop, G) for (s, a), G in gens.items()}

    # summarize
    max_r = max(ratios.values()) if ratios else float("nan")
    med_r = float(np.median(list(ratios.values()))) if ratios else float("nan")

    # overlap between loop and XY-loop subspace operator
    # (power fraction is already computed coefficient-wise; this overlap is operator-level)
    ov = abs(np.trace(H_xy.conj().T @ H_loop)) / (fro_norm(H_xy) * fro_norm(H_loop) + 1e-30)

    print(f"\n{'='*96}\n{name}  B1→B2 bridge  (cycle_len={cycle_len}, sites={n_sites}, bonds={len(edges)})\n{'='*96}")
    print(f"g={g}  gaps={gaps}  E0={-sum(gaps):.3f}")
    print(f"Loop amplitude (RMS coeffs): {ex['loop_amp']:.3e}")
    print(f"Loop fraction of total non-id power: {ex['loop_frac_total']:.3f}")
    print(f"XY-only fraction within loop power (Wilson-like magnetic sector proxy): {ex['xy_frac_within_loop']:.3f}")
    print(f"Operator overlap |<H_xy|H_loop>| (normalized Frobenius): {ov:.3f}")

    print("\nTop LOOP Pauli strings (identity-subtracted):")
    for lab, c in ex["top_loop_terms"][:8]:
        print(f"  {lab}: {c.real:+.3e}{c.imag:+.3e}j  |c|={abs(c):.3e}")

    if ex["top_xy_loop_terms"]:
        print("\nTop XY-LOOP Pauli strings:")
        for lab, c in ex["top_xy_loop_terms"][:8]:
            print(f"  {lab}: {c.real:+.3e}{c.imag:+.3e}j  |c|={abs(c):.3e}")

    print("\nGauss commutator ratios r = ||[H_loop,G_s^a]||/(||H_loop||·||G||):")
    for s in range(n_sites):
        row = []
        for a in ("X", "Y", "Z"):
            row.append(f"{ratios[(s,a)]:.3e}")
        print(f"  site {s}:  X {row[0]}   Y {row[1]}   Z {row[2]}")
    print(f"Summary: median r={med_r:.3e}, max r={max_r:.3e}")

    return {
        "name": name,
        "H_loop": H_loop,
        "H_xy": H_xy,
        "ratios": ratios,
        "loop_frac_total": ex["loop_frac_total"],
        "xy_frac_within_loop": ex["xy_frac_within_loop"],
        "overlap": ov,
    }


def make_summary_figure(rep_tri, rep_sq, out_png: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # plot commutator ratios per site/axis
    def plot_comm(ax, rep, title):
        n_sites = len({s for (s, a) in rep["ratios"].keys()})
        xs = np.arange(n_sites)
        for a, marker in [("X", "o"), ("Y", "s"), ("Z", "^")]:
            ys = [rep["ratios"][(s, a)] for s in range(n_sites)]
            ax.plot(xs, ys, marker=marker, label=a)
        ax.set_yscale("log")
        ax.set_xlabel("site")
        ax.set_ylabel("commutator ratio r")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

    plot_comm(axes[0], rep_tri, "Triangle: commutator ratios")
    plot_comm(axes[1], rep_sq, "Square: commutator ratios")

    fig.suptitle("HSF B1→B2 Bridge — Loop Operator vs Gauss Generators (diagnostic)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # Graphs
    tri_edges = [(0, 1), (1, 2), (0, 2)]
    sq_edges = [(0, 1), (1, 2), (2, 3), (0, 3)]

    # Parameters (keep consistent with your B1 lockdown)
    g = 0.3
    gaps_tri = [2.0 + 0.37 * i for i in range(3)]
    gaps_sq = [2.0 + 0.37 * i for i in range(4)]

    rep_tri = report_graph("Triangle", 3, tri_edges, cycle_len=3, g=g, gaps=gaps_tri)
    rep_sq = report_graph("Square", 4, sq_edges, cycle_len=4, g=g, gaps=gaps_sq)

    # Save figure
    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"plaquette_b2_bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    make_summary_figure(rep_tri, rep_sq, out_png)
    print(f"\n[saved] {out_png}")


if __name__ == "__main__":
    main()
