#!/usr/bin/env python
"""
HSF — Effective Bond Hamiltonian + Plaquette Emergence (Single Script)
=====================================================================

This script replaces the older B1 "bond_hamiltonian_*" family with ONE
clean, self-contained implementation that:

1) Builds a site+bond Hilbert space for a given graph (Triangle/Square).
2) Defines a gapped site Hamiltonian H0 with |0...0> as the true ground sector.
3) Defines a transmission coupling V that entangles sites and bonds.
4) Constructs the *correct* projector:
       P = |0...0><0...0|_sites ⊗ I_bonds
5) Computes:
   - Perturbative effective bond Hamiltonians order-by-order:
       H_eff^(1) = P V P
       H_eff^(2) = P V Q G Q V P
       H_eff^(3) = P V Q G Q V Q G Q V P
       ...
     where G = (E0 I - Q H0 Q)^(-1) on the Q subspace.
   - Exact effective Hamiltonian (all orders summed) via full resolvent:
       H_eff_exact = P H P + P V Q (E0 I - Q H Q)^(-1) Q V P
     with Q-subspace inversion performed *only on the Q indices*.

6) Decomposes H_eff into Pauli strings on bonds (d_B=2) and detects loop/plaquette terms.

Outputs:
- Console summary for Triangle and Square
- One PNG figure saved to ./hsf_out/

Run (Windows, single-line):
  python hsf_bond_effective_plaquette_demo.py

Notes:
- Pauli decomposition is implemented for bond dimension d_B=2 (qubit bonds).
- The "plaquette appears at order n" expectation is:
    Triangle (3-cycle): order 3
    Square   (4-cycle): order 4
"""

import os
from dataclasses import dataclass
from datetime import datetime
from itertools import product
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# -----------------------------
# Utilities: Pauli basis (d_B=2)
# -----------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}
PAULI_KEYS = ("I", "X", "Y", "Z")


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def hamming_weight(label: str) -> int:
    return sum(1 for ch in label if ch != "I")


def decompose_in_pauli(H_bonds: np.ndarray, n_bonds: int, tol: float = 1e-12):
    """
    Decompose a (2^n_bonds x 2^n_bonds) Hermitian into Pauli-string coefficients:
      H = sum_s c_s * P_s
    where P_s is tensor product of {I,X,Y,Z}.
    Coeff:
      c_s = Tr(P_s^\dagger H) / 2^n
    """
    dim = 2 ** n_bonds
    assert H_bonds.shape == (dim, dim)
    coeffs = {}
    norm = 2 ** n_bonds
    for key_tuple in product(PAULI_KEYS, repeat=n_bonds):
        label = "".join(key_tuple)
        P = kron_all([PAULI[k] for k in key_tuple])
        c = np.trace(P.conj().T @ H_bonds) / norm
        if abs(c) > tol:
            # keep real part if tiny imag
            if abs(c.imag) < 1e-10:
                c = c.real
            coeffs[label] = c
    return coeffs


# -----------------------------
# Lattice model
# -----------------------------
@dataclass
class EchoLattice:
    """
    Graph defined by sites and undirected edges. Each edge is a bond qubit.
    Full Hilbert space: H_sites (2^n_sites) ⊗ H_bonds (2^n_bonds).
    """
    n_sites: int
    edges: list  # list of tuples (u,v) length n_bonds
    d_B: int = 2

    def __post_init__(self):
        if self.d_B != 2:
            raise ValueError("This demo script supports d_B=2 only (Pauli decomposition).")
        self.n_bonds = len(self.edges)
        self.d_sites_total = 2 ** self.n_sites
        self.d_bonds_total = 2 ** self.n_bonds
        self.total_dim = self.d_sites_total * self.d_bonds_total

    def _embed_site_op(self, op2: np.ndarray, site_idx: int) -> np.ndarray:
        mats = []
        for i in range(self.n_sites):
            mats.append(op2 if i == site_idx else I2)
        op_sites = kron_all(mats)
        return np.kron(op_sites, np.eye(self.d_bonds_total, dtype=complex))

    def _embed_bond_op(self, op2: np.ndarray, bond_idx: int) -> np.ndarray:
        mats = []
        for b in range(self.n_bonds):
            mats.append(op2 if b == bond_idx else I2)
        op_bonds = kron_all(mats)
        return np.kron(np.eye(self.d_sites_total, dtype=complex), op_bonds)

    def transmission_edge_hamiltonian(self, bond_idx: int, coupling: float = 1.0) -> np.ndarray:
        """
        Isotropic SU(2)-like coupling:
          H_edge = sx(u) Bx(e) sx(v) + sy(u) By(e) sy(v) + sz(u) Bz(e) sz(v)
        with bond operators equal to Pauli on the bond qubit.

        This is the cleanest baseline for seeing loop-mediated higher-order terms.
        """
        u, v = self.edges[bond_idx]
        Hx = self._embed_site_op(X, u) @ self._embed_bond_op(X, bond_idx) @ self._embed_site_op(X, v)
        Hy = self._embed_site_op(Y, u) @ self._embed_bond_op(Y, bond_idx) @ self._embed_site_op(Y, v)
        Hz = self._embed_site_op(Z, u) @ self._embed_bond_op(Z, bond_idx) @ self._embed_site_op(Z, v)
        return coupling * (Hx + Hy + Hz)

    def build_V(self, coupling: float = 0.3) -> np.ndarray:
        V = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        for b in range(self.n_bonds):
            V += self.transmission_edge_hamiltonian(b, coupling=coupling)
        return V


# -----------------------------
# Projectors and H0
# -----------------------------
def build_P_Q(lattice: EchoLattice):
    """
    P = |0...0><0...0|_sites ⊗ I_bonds (CORRECT)
    """
    d_s = lattice.d_sites_total
    d_b = lattice.d_bonds_total
    psi0 = np.zeros(d_s, dtype=complex)
    psi0[0] = 1.0
    P_sites = np.outer(psi0, psi0.conj())  # rank-1
    P = np.kron(P_sites, np.eye(d_b, dtype=complex))
    Q = np.eye(lattice.total_dim, dtype=complex) - P
    # Sanity
    err = np.linalg.norm(P @ P - P)
    if err > 1e-10:
        raise RuntimeError(f"Projector failed sanity: ||P^2-P||={err:.2e}")
    return P, Q


def build_H0_sites(lattice: EchoLattice, site_gaps=None):
    """
    H0 = - sum_i Δ_i σz(i)  (so |0...0> is ground)
    Energy per site in |0> is -Δ_i (since σz|0>=|0>).
    """
    if site_gaps is None:
        # slight staggering to lift degeneracy (optional but helpful)
        site_gaps = [2.0 + 0.37 * i for i in range(lattice.n_sites)]
    H0 = np.zeros((lattice.total_dim, lattice.total_dim), dtype=complex)
    for i, d in enumerate(site_gaps):
        H0 += (-d) * lattice._embed_site_op(Z, i)
    # Ground energy for |0...0> is -sum Δ_i (since each site contributes -Δ_i)
    E0 = -float(np.sum(site_gaps))
    return H0, E0, site_gaps


def diag_resolvent_G(lattice: EchoLattice, Q: np.ndarray, H0: np.ndarray, E0: float, eps: float = 1e-12):
    """
    Since H0 is diagonal in computational basis (only σz terms), build:
      G = Q * diag(1/(E0 - H0_diag)) * Q
    on the full space, but only nonzero on Q subspace.
    """
    H0_diag = np.diag(H0).real
    G = np.zeros((lattice.total_dim, lattice.total_dim), dtype=complex)
    q_mask = np.abs(np.diag(Q) - 1.0) < 1e-14  # Q diagonal is 1 on Q indices
    idxs = np.where(q_mask)[0]
    for i in idxs:
        denom = (E0 - H0_diag[i])
        if abs(denom) > eps:
            G[i, i] = 1.0 / denom
    return G


# -----------------------------
# Perturbative H_eff
# -----------------------------
def perturbative_orders(lattice: EchoLattice, coupling: float = 0.3, max_order: int = 6):
    """
    Compute H_eff^(n) for n=1..max_order where:
      n=1: P V P
      n=2: P V Q G Q V P
      n=3: P V Q G Q V Q G Q V P
      ...
    Return dict(order -> H_eff_on_bonds)
    """
    P, Q = build_P_Q(lattice)
    H0, E0, site_gaps = build_H0_sites(lattice)
    V = lattice.build_V(coupling=coupling)
    G = diag_resolvent_G(lattice, Q, H0, E0)

    d_b = lattice.d_bonds_total
    p_idx = np.arange(d_b)  # site ground sector indices are first d_b states (site_config=0)
    # NOTE: this is now safe because P was built as |0..0><0..0| ⊗ I, matching ordering.

    PVQ = P @ V @ Q
    QVP = Q @ V @ P
    QVQ = Q @ V @ Q

    results = {}

    # Order 1
    H1_full = P @ V @ P
    H1 = H1_full[np.ix_(p_idx, p_idx)]
    H1 = (H1 + H1.conj().T) / 2.0
    results[1] = H1

    # Higher orders
    for n in range(2, max_order + 1):
        # PVQ @ (G @ QVQ)^(n-2) @ G @ QVP
        term = PVQ
        if n >= 3:
            block = G @ QVQ
            for _ in range(n - 2):
                term = term @ block
        term = term @ G @ QVP

        Hn_full = term
        Hn = Hn_full[np.ix_(p_idx, p_idx)]
        Hn = (Hn + Hn.conj().T) / 2.0
        results[n] = Hn

    return results, {"E0": E0, "site_gaps": site_gaps}


# -----------------------------
# Exact H_eff (all orders)
# -----------------------------
def exact_H_eff(lattice: EchoLattice, coupling: float = 0.3):
    """
    Exact effective Hamiltonian (Feshbach / resolvent form):
      H_eff = P H P + P V Q (E0 I - Q H Q)^(-1) Q V P
    with inversion performed only in the Q-subspace.

    We use H = H0 + V. H0 is diagonal; V connects sectors.
    """
    P, Q = build_P_Q(lattice)
    H0, E0, site_gaps = build_H0_sites(lattice)
    V = lattice.build_V(coupling=coupling)
    H = H0 + V

    d_b = lattice.d_bonds_total
    p_idx = np.arange(d_b)

    # Build Q-subspace indices
    q_mask = np.abs(np.diag(Q) - 1.0) < 1e-14
    q_idx = np.where(q_mask)[0]

    # Restricted blocks
    QHQ = H[np.ix_(q_idx, q_idx)]
    PVQ = V[np.ix_(p_idx, q_idx)]
    QVP = V[np.ix_(q_idx, p_idx)]
    PHP = H[np.ix_(p_idx, p_idx)]

    # Invert (E0 I - QHQ) via eigendecomposition (stable for small sizes)
    evals, evecs = eigh(QHQ)
    denom = (E0 - evals)
    # Regularize tiny denominators
    denom[np.abs(denom) < 1e-12] = np.sign(denom[np.abs(denom) < 1e-12]) * 1e-12

    inv = (evecs * (1.0 / denom)) @ evecs.conj().T  # (E0 I - QHQ)^-1

    Heff = PHP + PVQ @ inv @ QVP
    Heff = (Heff + Heff.conj().T) / 2.0
    return Heff


# -----------------------------
# Plaquette/loop detection on Pauli strings
# -----------------------------
def is_loop_term(lattice: EchoLattice, active_bonds):
    """
    active_bonds: list of bond indices participating in the operator.
    For a loop plaquette on an undirected graph:
      - each involved site should have degree 2 within this set
      - number of involved sites equals number of active bonds
      - at least 3 bonds
    """
    if len(active_bonds) < 3:
        return False
    site_count = {}
    for b in active_bonds:
        u, v = lattice.edges[b]
        site_count[u] = site_count.get(u, 0) + 1
        site_count[v] = site_count.get(v, 0) + 1
    if any(c != 2 for c in site_count.values()):
        return False
    if len(site_count) != len(active_bonds):
        return False
    return True


def summarize_pauli_structure(lattice: EchoLattice, H_bonds: np.ndarray, top_k=8, tol=1e-12):
    coeffs = decompose_in_pauli(H_bonds, lattice.n_bonds, tol=tol)
    total_w = float(np.sum([abs(c) ** 2 for c in coeffs.values()])) if coeffs else 0.0

    by_w = {}
    loops = []
    for lab, c in coeffs.items():
        w = hamming_weight(lab)
        by_w.setdefault(w, []).append((lab, c))

        if w >= 3:
            active = [i for i, ch in enumerate(lab) if ch != "I"]
            if is_loop_term(lattice, active):
                loops.append((lab, c))

    # Sort each weight bucket by magnitude
    for w in by_w:
        by_w[w].sort(key=lambda t: -abs(t[1]))

    loops.sort(key=lambda t: -abs(t[1]))

    return {
        "coeffs": coeffs,
        "total_weight": total_w,
        "by_weight": by_w,
        "loop_terms": loops,
    }


# -----------------------------
# Run experiments + plot
# -----------------------------
def run_graph(graph_name, n_sites, edges, expected_plaq_order, coupling=0.3, max_order=6):
    lat = EchoLattice(n_sites=n_sites, edges=edges, d_B=2)

    print("\n" + "=" * 78)
    print(f"GRAPH: {graph_name}  |  sites={n_sites} bonds={len(edges)}  |  coupling={coupling}")
    print("=" * 78)
    print(f"Expected first plaquette order: {expected_plaq_order}")

    orders, meta = perturbative_orders(lat, coupling=coupling, max_order=max_order)
    He_exact = exact_H_eff(lat, coupling=coupling)

    # Order-by-order summary
    print("\nOrder-by-order perturbative H_eff:")
    print(f"{'order':>5} {'||H||':>12} {'max_w':>6} {'loop?':>7}   top loop term (if any)")
    print("-" * 78)

    first_loop = None
    norms = []
    loop_flags = []
    max_ws = []

    for n in range(1, max_order + 1):
        Hn = orders[n]
        info = summarize_pauli_structure(lat, Hn, tol=1e-12)
        norm = float(np.linalg.norm(Hn))
        norms.append(norm)

        max_w = max(info["by_weight"].keys()) if info["by_weight"] else 0
        max_ws.append(max_w)

        has_loop = len(info["loop_terms"]) > 0
        loop_flags.append(has_loop)
        if has_loop and first_loop is None and norm > 1e-12:
            first_loop = n

        loop_str = "YES" if has_loop else "no"
        top_loop = ""
        if has_loop:
            lab, c = info["loop_terms"][0]
            top_loop = f"{lab}  c={c:+.3e}"

        print(f"{n:>5} {norm:>12.6f} {max_w:>6} {loop_str:>7}   {top_loop}")

    # Exact summary
    print("\nExact H_eff (all orders):")
    ex_info = summarize_pauli_structure(lat, He_exact, tol=1e-12)
    ex_norm = float(np.linalg.norm(He_exact))
    ex_max_w = max(ex_info["by_weight"].keys()) if ex_info["by_weight"] else 0
    ex_has_loop = len(ex_info["loop_terms"]) > 0
    print(f"  ||H_exact|| = {ex_norm:.6f}  |  max_weight = {ex_max_w}  |  has_loop = {ex_has_loop}")

    if ex_has_loop:
        print("  Top loop terms:")
        for (lab, c) in ex_info["loop_terms"][:5]:
            print(f"    {lab}: {c:+.6e}")

    if first_loop is None:
        print("\n[WARN] No loop/plaquette term detected up to max_order. Increase max_order or check coupling/gaps.")
    else:
        print(f"\nFirst detected loop term at perturbative order: {first_loop} (expected {expected_plaq_order})")

    return {
        "lattice": lat,
        "orders": orders,
        "exact": He_exact,
        "norms": norms,
        "loop_flags": loop_flags,
        "max_ws": max_ws,
        "first_loop": first_loop,
        "expected_loop": expected_plaq_order,
    }


def make_figure(tri_res, sq_res, coupling):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: norms vs order (log)
    ax = axes[0]
    for res, color, marker, name in [
        (tri_res, "#e74c3c", "o", "Triangle"),
        (sq_res, "#3498db", "s", "Square"),
    ]:
        orders = np.arange(1, len(res["norms"]) + 1)
        norms = np.array(res["norms"])
        norms = np.maximum(norms, 1e-16)
        ax.semilogy(orders, norms, marker=marker, color=color, linewidth=2, label=name)

        # mark first loop
        if res["first_loop"] is not None:
            ax.axvline(res["first_loop"], color=color, linestyle=":", alpha=0.5)
            ax.annotate(
                f"loop @ {res['first_loop']}",
                xy=(res["first_loop"], norms[res["first_loop"] - 1]),
                xytext=(res["first_loop"] + 0.2, norms[res["first_loop"] - 1] * 4),
                arrowprops=dict(arrowstyle="->", color=color),
                fontsize=10,
            )

    ax.set_xlabel("Perturbation order n")
    ax.set_ylabel(r"$\|H_{\mathrm{eff}}^{(n)}\|$")
    ax.set_title("Perturbative magnitude vs order")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: weight fractions for exact H_eff
    ax = axes[1]
    width = 0.35
    for offset, (res, color, name) in enumerate([
        (tri_res, "#e74c3c", "Triangle"),
        (sq_res, "#3498db", "Square"),
    ]):
        lat = res["lattice"]
        info = summarize_pauli_structure(lat, res["exact"], tol=1e-12)
        coeffs = info["coeffs"]
        total = float(np.sum([abs(c) ** 2 for c in coeffs.values()])) if coeffs else 1.0

        wsum = {}
        for lab, c in coeffs.items():
            w = hamming_weight(lab)
            wsum[w] = wsum.get(w, 0.0) + float(abs(c) ** 2)

        ws = sorted(wsum.keys())
        fracs = [100.0 * wsum[w] / total for w in ws]
        xs = [w + (offset - 0.5) * width for w in ws]
        ax.bar(xs, fracs, width=width, color=color, alpha=0.7, label=name)

    ax.set_xlabel("Operator weight (# active bonds)")
    ax.set_ylabel("Weight fraction (%)")
    ax.set_title("Exact $H_{eff}$ Pauli-weight distribution")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    # Panel 3: mechanism text
    ax = axes[2]
    ax.axis("off")
    txt = f"""
PLAQUETTE EMERGENCE (clean version)
----------------------------------

We define:

  H = H0 + V
  H0 = -Σ_i Δ_i σz(i)   (|0..0> is ground)
  P  = |0..0><0..0| ⊗ I_bonds
  Q  = I - P

Perturbative expansion:

  H_eff^(1) = P V P
  H_eff^(2) = P V Q G Q V P
  H_eff^(3) = P V Q G Q V Q G Q V P
  ...
  G = (E0 I - Q H0 Q)^(-1)

Loop/plaquette terms require the
virtual excitation to traverse a cycle:

  Triangle: first appears at order 3
  Square:   first appears at order 4

Coupling used: g = {coupling}
"""
    ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Mechanism", fontweight="bold")

    plt.suptitle("HSF Task B1 — Effective Bond Hamiltonian & Plaquette Emergence", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def main():
    np.random.seed(42)

    coupling = 0.30
    max_order = 6

    tri = run_graph("Triangle", 3, [(0, 1), (1, 2), (0, 2)], expected_plaq_order=3, coupling=coupling, max_order=max_order)
    sq = run_graph("Square", 4, [(0, 1), (1, 2), (2, 3), (0, 3)], expected_plaq_order=4, coupling=coupling, max_order=max_order)

    fig = make_figure(tri, sq, coupling=coupling)

    # Save figure in a portable location
    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bond_effective_plaquette_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
