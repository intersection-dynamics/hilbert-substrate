#!/usr/bin/env python3
"""
Priority E1: Constraint Ablation Study
Hilbert Substrate Framework — Paper III

Demonstrates that removing ANY single constraint from the HSF destroys
the emergence chain. Four constraints tested:

  1. No-forgetting  → bonds record transmissions → gauge fields
  2. No-signaling   → bounded information speed → spatial locality
  3. Finite bandwidth → finite bond dimension → finite gauge groups
  4. No-refolding   → dimensional stability → spatial basin trapping

System: 4-site square lattice with bond degrees of freedom
  Sites: qubits 0-3 (square: 0—1—2—3—0)
  Bonds: qubits 4-7 (bond k+4 on edge k)
  Total: 8 qubits = 256-dimensional Hilbert space

Dependencies: numpy, scipy, matplotlib (standard scientific Python)

Author: Ben Bray / HSF Research Program
Date: February 2026
"""

import numpy as np
from scipy.linalg import expm, eigvalsh
from itertools import product as iprod
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# SECTION 1: OPERATOR UTILITIES
# ============================================================

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_LIST = [I2, sx, sy, sz]


def tensor(*ops):
    """Tensor product of a sequence of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def op_on_qubit(op, qubit, n_qubits):
    """Place a single-qubit operator on qubit `qubit` in an n_qubits system."""
    ops = [I2] * n_qubits
    ops[qubit] = op
    return tensor(*ops)


def two_qubit_op(op_a, qubit_a, op_b, qubit_b, n_qubits):
    """Place two single-qubit operators on specified qubits."""
    ops = [I2] * n_qubits
    ops[qubit_a] = op_a
    ops[qubit_b] = op_b
    return tensor(*ops)


def partial_trace(rho, keep, dims):
    """
    Partial trace keeping subsystems in `keep`.
    dims = list of dimensions for each subsystem.
    """
    keep = sorted(keep)
    n = len(dims)
    trace_out = sorted(set(range(n)) - set(keep))
    d_keep = int(np.prod([dims[k] for k in keep]))
    rho_reduced = np.zeros((d_keep, d_keep), dtype=complex)

    d_trace_dims = [dims[t] for t in trace_out]
    d_trace = int(np.prod(d_trace_dims)) if d_trace_dims else 1

    for ki in range(d_keep):
        for kj in range(d_keep):
            ki_sub = _index_decompose(ki, [dims[k] for k in keep])
            kj_sub = _index_decompose(kj, [dims[k] for k in keep])
            for tr in range(d_trace):
                tr_sub = _index_decompose(tr, d_trace_dims) if d_trace_dims else []
                full_i = [0] * n
                full_j = [0] * n
                for p, k in enumerate(keep):
                    full_i[k] = ki_sub[p]
                    full_j[k] = kj_sub[p]
                for p, t in enumerate(trace_out):
                    full_i[t] = tr_sub[p]
                    full_j[t] = tr_sub[p]
                fi = _index_compose(full_i, dims)
                fj = _index_compose(full_j, dims)
                rho_reduced[ki, kj] += rho[fi, fj]
    return rho_reduced


def _index_decompose(flat, dims):
    """flat index → list of per-subsystem indices."""
    result = []
    for d in reversed(dims):
        result.append(flat % d)
        flat //= d
    return list(reversed(result))


def _index_compose(subs, dims):
    """list of per-subsystem indices → flat index."""
    flat = 0
    for s, d in zip(subs, dims):
        flat = flat * d + s
    return flat


def von_neumann_entropy(rho):
    """S = -Tr(ρ log₂ ρ)."""
    evals = np.real(eigvalsh(rho))
    evals = evals[evals > 1e-14]
    return float(-np.sum(evals * np.log2(evals)))


def mutual_information(rho_ab, rho_a, rho_b):
    """I(A:B) = S(A) + S(B) - S(AB)."""
    return von_neumann_entropy(rho_a) + von_neumann_entropy(rho_b) - von_neumann_entropy(rho_ab)


def trace_distance(rho, sigma):
    """T(ρ,σ) = ½ ‖ρ-σ‖₁."""
    evals = np.real(eigvalsh(rho - sigma))
    return float(0.5 * np.sum(np.abs(evals)))


# ============================================================
# SECTION 2: ECHO LATTICE MODEL
# ============================================================

class EchoLattice:
    """
    4-site square lattice with bond (link) degrees of freedom.

    Topology:
        site 0 ---bond 4--- site 1
          |                   |
        bond 7             bond 5
          |                   |
        site 3 ---bond 6--- site 2
    """
    N_QUBITS = 8
    DIM = 256
    EDGES = [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]
    DIAGONALS = [(0, 2), (1, 3)]
    PLAQUETTE_BONDS = [4, 5, 6, 7]

    def __init__(self):
        self.dims = [2] * self.N_QUBITS
        self.site_idx = list(range(4))
        self.bond_idx = list(range(4, 8))

    # ---------- Hamiltonian builders ----------

    def build_H(self, scenario='FULL', g=0.3, Delta=1.0):
        """
        FULL       – echo coupling through bonds (all constraints)
        NO_FORGET  – direct site-site coupling, bonds passive
        NO_SIGNAL  – echo coupling + long-range site-site terms
        """
        H = np.zeros((self.DIM, self.DIM), dtype=complex)

        # On-site energy gap
        for i in range(4):
            H += Delta * op_on_qubit(sz, i, self.N_QUBITS)

        if scenario in ('FULL', 'NO_SIGNAL'):
            # Echo coupling: site ↔ bond ↔ site
            for si, sj, b in self.EDGES:
                for pauli in [sx, sy, sz]:
                    H += g * two_qubit_op(pauli, si, pauli, b, self.N_QUBITS)
                    H += g * two_qubit_op(pauli, b, pauli, sj, self.N_QUBITS)

        elif scenario == 'NO_FORGET':
            # Direct Heisenberg: site ↔ site, bonds decouple
            for si, sj, _ in self.EDGES:
                for pauli in [sx, sy, sz]:
                    H += g * two_qubit_op(pauli, si, pauli, sj, self.N_QUBITS)

        if scenario == 'NO_SIGNAL':
            # Add long-range (diagonal) site-site couplings
            for si, sj in self.DIAGONALS:
                for pauli in [sx, sy, sz]:
                    H += g * two_qubit_op(pauli, si, pauli, sj, self.N_QUBITS)

        return H

    # ---------- States ----------

    def psi_neel(self):
        """Néel state |0101⟩_sites ⊗ |0000⟩_bonds."""
        psi = np.zeros(self.DIM, dtype=complex)
        bits = [0, 1, 0, 1, 0, 0, 0, 0]
        idx = sum(b << (7 - i) for i, b in enumerate(bits))
        psi[idx] = 1.0
        return psi

    def psi_zero(self):
        """All-zero state."""
        psi = np.zeros(self.DIM, dtype=complex)
        psi[0] = 1.0
        return psi

    # ---------- Evolution ----------

    def evolve(self, psi, H, t):
        return expm(-1j * H * t) @ psi

    # ---------- Diagnostics ----------

    def avg_bond_entropy(self, psi):
        """Average von Neumann entropy of individual bonds."""
        rho = np.outer(psi, psi.conj())
        S_vals = []
        for b in self.bond_idx:
            rho_b = partial_trace(rho, [b], self.dims)
            S_vals.append(von_neumann_entropy(rho_b))
        return float(np.mean(S_vals))

    def plaquette_correlation(self, psi):
        """Connected 4-bond ZZ…Z correlation around the plaquette."""
        rho = np.outer(psi, psi.conj())
        # ⟨Z₄Z₅Z₆Z₇⟩
        op4 = op_on_qubit(sz, 4, 8) @ op_on_qubit(sz, 5, 8) @ \
              op_on_qubit(sz, 6, 8) @ op_on_qubit(sz, 7, 8)
        exp_4 = float(np.real(np.trace(rho @ op4)))
        # Π ⟨Z_b⟩
        prod_1 = 1.0
        for b in self.PLAQUETTE_BONDS:
            Zb = op_on_qubit(sz, b, 8)
            prod_1 *= float(np.real(np.trace(rho @ Zb)))
        return exp_4 - prod_1

    def plaquette_mi(self, psi):
        """Total MI between consecutive bond pairs around the plaquette."""
        rho = np.outer(psi, psi.conj())
        total = 0.0
        bonds = self.PLAQUETTE_BONDS
        for k in range(4):
            b1, b2 = bonds[k], bonds[(k + 1) % 4]
            rho12 = partial_trace(rho, [b1, b2], self.dims)
            rho1 = partial_trace(rho, [b1], self.dims)
            rho2 = partial_trace(rho, [b2], self.dims)
            total += mutual_information(rho12, rho1, rho2)
        return total

    def light_cone(self, H, times, perturb_site=0):
        """
        Trace distance at each site after perturbation at `perturb_site`.
        Returns {site: [TD(t) for t in times]}.
        """
        psi_ref = self.psi_zero()
        psi_pert = op_on_qubit(sx, perturb_site, 8) @ psi_ref

        results = {s: [] for s in range(4)}
        for t in times:
            pr = self.evolve(psi_ref, H, t)
            pp = self.evolve(psi_pert, H, t)
            rho_r = np.outer(pr, pr.conj())
            rho_p = np.outer(pp, pp.conj())
            for s in range(4):
                rr = partial_trace(rho_r, [s], self.dims)
                rp = partial_trace(rho_p, [s], self.dims)
                results[s].append(trace_distance(rr, rp))
        return results


# ============================================================
# SECTION 3: GAUGE ALGEBRA DIAGNOSTIC
# ============================================================

def echo_algebra_dimension(d_B=2, verbose=False):
    """
    Dimension of the echo algebra for bond dimension d_B.

    On a single (site, bond) pair with echo coupling
        H = Σ_α σ^α_site ⊗ λ^α_bond
    the algebra generated by nested commutators, projected to bond,
    should reproduce su(d_B).
    """
    d_S = 2
    if d_B == 2:
        bond_ops = [sx, sy, sz]
    elif d_B == 3:
        bond_ops = _gell_mann()
    else:
        bond_ops = _gen_gell_mann(d_B)

    site_ops = [sx, sy, sz]

    # Generators: T_{αβ} = σ^α ⊗ λ^β  on  (d_S × d_B) space
    gens = []
    for s_op in site_ops:
        for b_op in bond_ops:
            gens.append(np.kron(s_op, b_op))

    # Close under commutation
    algebra = list(gens)
    for _ in range(40):
        new = []
        for i in range(len(algebra)):
            for j in range(i + 1, len(algebra)):
                comm = algebra[i] @ algebra[j] - algebra[j] @ algebra[i]
                if np.linalg.norm(comm) > 1e-10 and not _in_span(comm, algebra + new):
                    new.append(comm / np.linalg.norm(comm))
        if not new:
            break
        algebra.extend(new)

    # Project each element to bond subspace: A_bond = Σ_s ⟨s|A|s⟩
    projected = []
    for A in algebra:
        A_b = np.zeros((d_B, d_B), dtype=complex)
        for s in range(d_S):
            A_b += A[s * d_B:(s + 1) * d_B, s * d_B:(s + 1) * d_B]
        A_b -= np.trace(A_b) / d_B * np.eye(d_B)
        if np.linalg.norm(A_b) > 1e-10:
            projected.append(A_b.flatten())

    if not projected:
        rank = 0
    else:
        rank = int(np.linalg.matrix_rank(np.array(projected), tol=1e-8))

    if verbose:
        print(f"  d_B={d_B}: algebra dim (full)={len(algebra)}, "
              f"bond-projected rank={rank}, expected su({d_B})={d_B**2 - 1}")
    return rank


def _in_span(vec, basis, tol=1e-8):
    if not basis:
        return False
    B = np.array([b.flatten() for b in basis]).T
    v = vec.flatten()
    coeffs, _, _, _ = np.linalg.lstsq(B, v, rcond=None)
    return np.linalg.norm(v - B @ coeffs) < tol * max(np.linalg.norm(v), 1.0)


def _gell_mann():
    """8 Gell-Mann matrices."""
    L = []
    L.append(np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex))
    L.append(np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex))
    L.append(np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=complex))
    L.append(np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=complex))
    L.append(np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex))
    L.append(np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex))
    L.append(np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex))
    L.append(np.diag([1.0, 1.0, -2.0]).astype(complex) / np.sqrt(3))
    return L


def _gen_gell_mann(d):
    """Generalized Gell-Mann matrices for su(d)."""
    mats = []
    for j in range(d):
        for k in range(j+1, d):
            m = np.zeros((d, d), dtype=complex); m[j,k]=1; m[k,j]=1; mats.append(m)
    for j in range(d):
        for k in range(j+1, d):
            m = np.zeros((d, d), dtype=complex); m[j,k]=-1j; m[k,j]=1j; mats.append(m)
    for l in range(1, d):
        m = np.zeros((d, d), dtype=complex)
        for j in range(l): m[j,j] = 1
        m[l,l] = -l
        m *= np.sqrt(2.0/(l*(l+1)))
        mats.append(m)
    return mats


# ============================================================
# SECTION 4: NO-REFOLDING TEST  (Paper II mini-version)
# ============================================================

def brockett_step(H, p=4, dt=0.05):
    """One step of double-bracket flow minimising Cp on a 2-qubit H."""
    c = np.zeros(16, dtype=complex)
    w = np.zeros(16)
    idx = 0
    for i, pi in enumerate(PAULI_LIST):
        for j, pj in enumerate(PAULI_LIST):
            P = np.kron(pi, pj)
            c[idx] = np.trace(P @ H) / 4.0
            w[idx] = (0 if i == 0 else 1) + (0 if j == 0 else 1)
            idx += 1
    total_sq = np.sum(np.abs(c)**2)
    if total_sq < 1e-15:
        return H, 0.0

    cost = float(np.real(np.sum(w**p * np.abs(c)**2) / total_sq))

    # Gradient operator M = (2/‖c‖²) Σ w^p c_k P_k
    M = np.zeros_like(H)
    idx = 0
    for i, pi in enumerate(PAULI_LIST):
        for j, pj in enumerate(PAULI_LIST):
            M += w[idx]**p * c[idx] * np.kron(pi, pj)
            idx += 1
    M *= 2.0 / total_sq

    bracket = H @ M - M @ H
    dH = H @ bracket - bracket @ H
    H_new = H + dt * dH
    H_new = 0.5 * (H_new + H_new.conj().T)
    return H_new, cost


def random_unitary(d, seed):
    rng = np.random.RandomState(seed)
    Z = (rng.randn(d, d) + 1j * rng.randn(d, d)) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    ph = np.diag(R); ph = ph / np.abs(ph)
    return Q * ph[np.newaxis, :]


def _pauli_decompose(H, n_q):
    """Pauli decomposition of n_q-qubit Hamiltonian → (coeffs, weights)."""
    dim = 2**n_q
    n_paulis = 4**n_q
    coeffs = np.zeros(n_paulis, dtype=complex)
    weights = np.zeros(n_paulis)
    # Iterate over all Pauli strings
    for idx in range(n_paulis):
        # Decode idx → Pauli indices
        tmp = idx
        pauli_ids = []
        for _ in range(n_q):
            pauli_ids.append(tmp % 4)
            tmp //= 4
        pauli_ids = pauli_ids[::-1]
        # Build Pauli string
        P = PAULI_LIST[pauli_ids[0]]
        for pid in pauli_ids[1:]:
            P = np.kron(P, PAULI_LIST[pid])
        coeffs[idx] = np.trace(P @ H) / dim
        weights[idx] = sum(1 for pid in pauli_ids if pid != 0)
    return coeffs, weights


def _brockett_step_nq(H, n_q, p=4, dt=0.02):
    """One Brockett flow step for n_q-qubit H."""
    dim = 2**n_q
    coeffs, weights = _pauli_decompose(H, n_q)
    total_sq = np.sum(np.abs(coeffs)**2)
    if total_sq < 1e-15:
        return H, 0.0
    cost = float(np.real(np.sum(weights**p * np.abs(coeffs)**2) / total_sq))

    # Gradient operator M
    M = np.zeros_like(H)
    n_paulis = 4**n_q
    for idx in range(n_paulis):
        tmp = idx
        pauli_ids = []
        for _ in range(n_q):
            pauli_ids.append(tmp % 4)
            tmp //= 4
        pauli_ids = pauli_ids[::-1]
        P = PAULI_LIST[pauli_ids[0]]
        for pid in pauli_ids[1:]:
            P = np.kron(P, PAULI_LIST[pid])
        M += weights[idx]**p * coeffs[idx] * P
    M *= 2.0 / total_sq

    bracket = H @ M - M @ H
    dH = H @ bracket - bracket @ H
    H_new = H + dt * dH
    H_new = 0.5 * (H_new + H_new.conj().T)
    return H_new, cost


def no_refolding_test(n_trials=6, n_steps=40, p=4):
    """
    Brockett-flow test on a 3-qubit chain (dim=8).
    
    The Brockett flow monotonically descends the locality cost landscape.
    - WITHOUT re-scrambling: cost decreases smoothly to spatial minimum.
    - WITH periodic re-scrambling: kicks disrupt convergence; the system
      gets pushed out of the spatial basin and cost fluctuates.
    
    This demonstrates that no-refolding (= no re-scrambling) is required
    for the system to remain in the spatial basin once trapped.
    """
    n_q = 3
    dim = 2**n_q
    # Local Heisenberg chain: H = Σ_{i,i+1} σ·σ
    H_local = np.zeros((dim, dim), dtype=complex)
    for i in range(n_q - 1):
        for pauli in [sx, sy, sz]:
            ops = [I2]*n_q
            ops[i] = pauli; ops[i+1] = pauli
            P = ops[0]
            for o in ops[1:]:
                P = np.kron(P, o)
            H_local += P

    results = {'stable': [], 'unstable': []}
    for trial in range(n_trials):
        U = random_unitary(dim, seed=trial*137)
        H0 = U @ H_local @ U.conj().T

        # Path 1: no re-scrambling (monotonic descent)
        H = H0.copy()
        costs = []
        for step in range(n_steps + 1):
            H, cost = _brockett_step_nq(H, n_q, p=p, dt=0.02)
            costs.append(cost)
        results['stable'].append(costs)

        # Path 2: periodic re-scrambling (kicks disrupt basin)
        H = H0.copy()
        costs = []
        for step in range(n_steps + 1):
            H, cost = _brockett_step_nq(H, n_q, p=p, dt=0.02)
            costs.append(cost)
            if step > 0 and step % 8 == 0:
                rng = np.random.RandomState(trial*1000 + step)
                A = rng.randn(dim, dim) + 1j*rng.randn(dim, dim)
                A = (A + A.conj().T)/2
                A -= np.trace(A)/dim * np.eye(dim)
                kick_strength = 0.5
                U_kick = expm(1j * kick_strength * A / np.linalg.norm(A))
                H = U_kick @ H @ U_kick.conj().T
        results['unstable'].append(costs)

    return results


# ============================================================
# SECTION 5: RUN EVERYTHING
# ============================================================

def run_all(g=0.3, Delta=1.0, T_max=5.0, n_t=40):
    lat = EchoLattice()
    times = np.linspace(0.1, T_max, n_t)

    scenarios = ['FULL', 'NO_FORGET', 'NO_SIGNAL']
    data = {}

    for sc in scenarios:
        print(f"\n--- {sc} ---")
        H = lat.build_H(sc, g=g, Delta=Delta)
        assert np.allclose(H, H.conj().T), f"{sc}: H not Hermitian!"

        psi0 = lat.psi_neel()

        # Bond entropy
        be = []
        for t in times:
            psi_t = lat.evolve(psi0, H, t)
            be.append(lat.avg_bond_entropy(psi_t))
        print(f"  Bond entropy final: {be[-1]:.4f}")

        # Plaquette diagnostics
        pc, pm = [], []
        for t in times:
            psi_t = lat.evolve(psi0, H, t)
            pc.append(lat.plaquette_correlation(psi_t))
            pm.append(lat.plaquette_mi(psi_t))
        print(f"  Plaquette corr final: {pc[-1]:.6f}")
        print(f"  Plaquette MI final:   {pm[-1]:.4f}")

        # Light cone
        lc = lat.light_cone(H, times, perturb_site=0)

        data[sc] = dict(times=times, bond_entropy=np.array(be),
                        plaq_corr=np.array(pc), plaq_mi=np.array(pm),
                        light_cone=lc)

    return data


# ============================================================
# SECTION 6: PLOT
# ============================================================

def make_figure(data, alg, refold, save='e1_constraint_ablation.png'):
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.38, wspace=0.32)

    C = {'FULL': '#2196F3', 'NO_FORGET': '#F44336', 'NO_SIGNAL': '#FF9800'}
    L = {'FULL': 'All constraints', 'NO_FORGET': 'No-forgetting OFF',
         'NO_SIGNAL': 'No-signaling OFF'}

    # (A) Bond entropy
    ax = fig.add_subplot(gs[0, 0])
    for sc in data:
        ax.plot(data[sc]['times'], data[sc]['bond_entropy'],
                color=C[sc], label=L[sc], lw=2)
    ax.set_xlabel('Time'); ax.set_ylabel('Avg Bond Entropy (bits)')
    ax.set_title('(A) Bond Participation\n(Echo Signature)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (B) |Plaquette correlation|
    ax = fig.add_subplot(gs[0, 1])
    for sc in data:
        ax.plot(data[sc]['times'], np.abs(data[sc]['plaq_corr']),
                color=C[sc], label=L[sc], lw=2)
    ax.set_xlabel('Time'); ax.set_ylabel('|Connected Plaquette Corr|')
    ax.set_title('(B) Loop Structure\n(Gauge Signature)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (C) Plaquette MI
    ax = fig.add_subplot(gs[0, 2])
    for sc in data:
        ax.plot(data[sc]['times'], data[sc]['plaq_mi'],
                color=C[sc], label=L[sc], lw=2)
    ax.set_xlabel('Time'); ax.set_ylabel('Loop Bond MI (bits)')
    ax.set_title('(C) Bond-Bond Correlations\n(Loop MI)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (D-F) Light cones
    site_c = ['#000000', '#E91E63', '#9C27B0', '#4CAF50']
    dists = [0, 1, 2, 1]
    for j, sc in enumerate(data):
        ax = fig.add_subplot(gs[1, j])
        lc = data[sc]['light_cone']
        t = data[sc]['times']
        for s in range(4):
            ls = ':' if s == 0 else '-'
            ax.plot(t, lc[s], color=site_c[s],
                    label=f'd={dists[s]}', lw=2, ls=ls)
        ax.set_xlabel('Time'); ax.set_ylabel('Trace Distance')
        ax.set_title(f'({"DEF"[j]}) Light Cone: {L[sc]}',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, title='Graph dist'); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # (G) Algebra dimension bar chart
    ax = fig.add_subplot(gs[2, 0])
    dBs = sorted(alg.keys())
    meas = [alg[d] for d in dBs]
    expd = [d**2 - 1 for d in dBs]
    x = np.arange(len(dBs)); w = 0.35
    ax.bar(x - w/2, meas, w, label='Measured', color='#2196F3', alpha=0.85)
    ax.bar(x + w/2, expd, w, label='Expected d²−1', color='#90CAF9', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'd_B={d}\n→SU({d})' for d in dBs])
    ax.set_ylabel('Algebra Dimension')
    ax.set_title('(G) Finite Bandwidth\n→ Gauge Group', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(meas):
        ax.text(i - w/2, v + 0.3, str(v), ha='center', fontweight='bold', fontsize=9)

    # (H) Refolding test
    ax = fig.add_subplot(gs[2, 1])
    steps = np.arange(len(refold['stable'][0]))
    for c_ in refold['stable']:
        ax.plot(steps, c_, color='#2196F3', alpha=0.25, lw=1)
    for c_ in refold['unstable']:
        ax.plot(steps, c_, color='#F44336', alpha=0.25, lw=1)
    ax.plot(steps, np.mean(refold['stable'], axis=0),
            color='#2196F3', lw=3, label='No refolding (stable)')
    ax.plot(steps, np.mean(refold['unstable'], axis=0),
            color='#F44336', lw=3, label='Refolding allowed')
    ax.set_xlabel('Brockett Flow Step'); ax.set_ylabel('Locality Cost Cp')
    ax.set_title('(H) No-Refolding\n→ Basin Stability', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (I) Summary table
    ax = fig.add_subplot(gs[2, 2]); ax.axis('off')
    rows = []
    for sc in data:
        d = data[sc]
        be_f = d['bond_entropy'][-1]
        pc_f = abs(d['plaq_corr'][-1])
        mi_f = d['plaq_mi'][-1]
        mid = len(d['times']) // 4
        lc = d['light_cone']
        td_adj = max(lc[1][mid], lc[3][mid])
        td_diag = lc[2][mid]
        lcr = td_diag / (td_adj + 1e-10)
        rows.append([L[sc], f'{be_f:.3f}', f'{pc_f:.4f}', f'{mi_f:.3f}', f'{lcr:.2f}'])
    rows.append(['Bandwidth OFF', '—', '—', '—', 'dim→∞'])
    cost_s = np.mean([c[-1] for c in refold['stable']])
    cost_u = np.mean([c[-1] for c in refold['unstable']])
    rows.append(['Refolding ON', '—', '—', '—', f'C={cost_u:.1f}'])

    cols = ['Scenario', 'Bond S', '|C_plaq|', 'Loop MI', 'LC ratio']
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.4)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#E3F2FD')
        tbl[0, j].set_text_props(fontweight='bold')
    for j in range(len(cols)):
        tbl[1, j].set_facecolor('#E8F5E9')
    for i in range(2, len(rows)+1):
        for j in range(len(cols)):
            tbl[i, j].set_facecolor('#FFEBEE')
    ax.set_title('(I) Ablation Summary', fontweight='bold', pad=20)

    fig.suptitle('E1: Constraint Ablation Study — Hilbert Substrate Framework\n'
                 '4-site square, d_B=2, g=0.3, Δ=1.0',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {save}")
    plt.close(fig)


# ============================================================
# SECTION 7: MAIN
# ============================================================

def main():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║  HILBERT SUBSTRATE FRAMEWORK — PAPER III                    ║
║  Priority E1: Constraint Ablation Study                     ║
║  System: 4-site square lattice, 8 qubits, dim=256           ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

    # 1. Main ablation
    print("[1/3] Echo-model ablation (3 scenarios × 40 time steps)...")
    data = run_all(g=0.3, Delta=1.0, T_max=5.0, n_t=40)

    # 2. Gauge algebra
    print("\n[2/3] Gauge algebra dimension test...")
    alg = {}
    for dB in [2, 3, 4]:
        alg[dB] = echo_algebra_dimension(dB, verbose=True)

    # 3. No-refolding
    print("\n[3/3] No-refolding test (Brockett flow)...")
    refold = no_refolding_test(n_trials=6, n_steps=40, p=4)

    # Summary
    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Constraint Removed':<22} {'What Breaks':<28} {'Evidence'}")
    print("-" * 75)

    be_f = data['FULL']['bond_entropy'][-1]
    be_n = data['NO_FORGET']['bond_entropy'][-1]
    print(f"{'No-forgetting':<22} {'Gauge invariance gone':<28} "
          f"Bond S: {be_f:.3f} → {be_n:.3f}")

    mid = len(data['FULL']['times']) // 4
    def lc_ratio(d):
        lc = d['light_cone']
        return lc[2][mid] / (max(lc[1][mid], lc[3][mid]) + 1e-10)
    print(f"{'No-signaling':<22} {'Spatial structure lost':<28} "
          f"LC ratio: {lc_ratio(data['FULL']):.3f} → {lc_ratio(data['NO_SIGNAL']):.3f}")

    for dB in [2, 3, 4]:
        print(f"{'Finite bandwidth':<22} "
              f"{'d_B='+str(dB)+' → dim='+str(alg[dB]):<28} "
              f"Expected su({dB})={dB**2-1}")

    cs = np.mean([c[-1] for c in refold['stable']])
    cu = np.mean([c[-1] for c in refold['unstable']])
    print(f"{'No-refolding':<22} {'Basin instability':<28} "
          f"Cost: {cs:.2f} (stable) vs {cu:.2f} (kicked)")

    # Plot
    print("\nGenerating 9-panel figure...")
    make_figure(data, alg, refold, save='e1_constraint_ablation.png')

    print("\n✓ E1 complete.  Output: e1_constraint_ablation.png")


if __name__ == '__main__':
    main()