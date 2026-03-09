"""
Echo Algebra Extraction (Clean / Less Foolable)
==============================================

Key upgrades vs original:
- Adds strict generator extraction that only uses near-unitary, single-Kraus-dominated samples.
- Headline experiments do NOT mix in random Hamiltonians.
- Random Hamiltonians are available as an optional control only.
"""

import numpy as np
from scipy.linalg import expm, logm

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ============================================================
# Reference generators
# ============================================================

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [sigma_x, sigma_y, sigma_z]

def gell_mann_matrices():
    gm = []
    gm.append(np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex))
    gm.append(np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex))
    gm.append(np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=complex))
    gm.append(np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=complex))
    gm.append(np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex))
    gm.append(np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex))
    gm.append(np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex))
    gm.append(np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=complex) / np.sqrt(3))
    return gm

GELL_MANN = gell_mann_matrices()

# ============================================================
# Transmission Hamiltonians
# ============================================================

def transmission_hamiltonian(d_B, variant="standard", rng=None):
    """
    Returns H on C^2 ⊗ C^{d_B} ⊗ C^2.
    NOTE: for d_B>2 this uses an SU(2) spin-s irrep embedding (as in your original).
    """
    if rng is None:
        rng = np.random.default_rng()

    sx, sy, sz = sigma_x, sigma_y, sigma_z
    I2 = np.eye(2, dtype=complex)
    IB = np.eye(d_B, dtype=complex)

    # Bond operators
    if d_B == 2:
        Bx, By, Bz = sx.copy(), sy.copy(), sz.copy()
    else:
        # spin-s SU(2) irrep (dimension d_B)
        s = (d_B - 1) / 2.0
        Bx = np.zeros((d_B, d_B), dtype=complex)
        By = np.zeros((d_B, d_B), dtype=complex)
        Bz = np.zeros((d_B, d_B), dtype=complex)
        for m_idx in range(d_B):
            m = s - m_idx
            if m_idx + 1 < d_B:
                mp = m - 1
                coeff = np.sqrt(s*(s+1) - m*mp) * 0.5
                Bx[m_idx, m_idx+1] = coeff
                Bx[m_idx+1, m_idx] = coeff
                By[m_idx, m_idx+1] = -1j * coeff
                By[m_idx+1, m_idx] = 1j * coeff
            Bz[m_idx, m_idx] = m

    if variant == "standard":
        H = (np.kron(np.kron(sx, Bx), sx) +
             np.kron(np.kron(sz, Bz), sz) +
             0.3 * np.kron(np.kron(sy, IB), sy))
    elif variant == "full":
        H = (np.kron(np.kron(sx, Bx), sx) +
             np.kron(np.kron(sy, By), sy) +
             np.kron(np.kron(sz, Bz), sz))
    elif variant == "random":
        d_full = 2 * d_B * 2
        A = rng.normal(size=(d_full, d_full)) + 1j * rng.normal(size=(d_full, d_full))
        H = (A + A.conj().T) / 2.0
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return H

# ============================================================
# Kraus extraction
# ============================================================

def extract_kraus_operators(U_full, d_B, psi_a, psi_b):
    """
    K_{mn} = (⟨m_a| ⊗ I_B ⊗ ⟨n_b|) U (|ψ_a⟩ ⊗ I_B ⊗ |ψ_b⟩)
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

# ============================================================
# Generator extraction (lenient vs strict)
# ============================================================

def haar_random_qubit(rng):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    return v / np.linalg.norm(v)

def extract_generators_infinitesimal_lenient(d_B, H, n_site_samples=200, eps=1e-4, rng=None):
    """
    Your original philosophy: take all Kraus operators, subtract trace,
    divide by eps, then split into Hermitian + anti-Hermitian parts.
    Useful for exploration, but can inflate the span if dynamics is dissipative.
    """
    if rng is None:
        rng = np.random.default_rng()

    U = expm(-1j * eps * H)
    I = np.eye(d_B, dtype=complex)
    generators = []

    for _ in range(n_site_samples):
        psi_a = haar_random_qubit(rng)
        psi_b = haar_random_qubit(rng)
        kraus = extract_kraus_operators(U, d_B, psi_a, psi_b)

        for K in kraus:
            trace_part = np.trace(K) / d_B
            G = (K - trace_part * I) / eps

            if np.linalg.norm(G) > 1e-8:
                G_herm = (G + G.conj().T) / 2.0
                G_anti = (G - G.conj().T) / (2.0j)

                if np.linalg.norm(G_herm) > 1e-8:
                    generators.append(G_herm / np.linalg.norm(G_herm))
                if np.linalg.norm(G_anti) > 1e-8:
                    generators.append(G_anti / np.linalg.norm(G_anti))

    return generators

def extract_generators_infinitesimal_strict(d_B, H, n_site_samples=200, eps=1e-4,
                                           unitary_tol=1e-6, leakage_tol=1e-6, rng=None, verbose=True):
    """
    Less foolable:
    - Keep only samples where Σ K†K ≈ I (numerically) AND one Kraus dominates.
    - Use only dominant Kraus to define an effective Hermitian generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    U = expm(-1j * eps * H)
    I = np.eye(d_B, dtype=complex)

    generators = []
    accepted = 0
    rejected = 0

    for _ in range(n_site_samples):
        psi_a = haar_random_qubit(rng)
        psi_b = haar_random_qubit(rng)
        kraus = extract_kraus_operators(U, d_B, psi_a, psi_b)

        comp = np.zeros((d_B, d_B), dtype=complex)
        weights = []
        for K in kraus:
            comp += K.conj().T @ K
            weights.append(np.linalg.norm(K, "fro")**2)

        comp_err = np.linalg.norm(comp - I)
        dom = int(np.argmax(weights))
        leakage = 1.0 - (weights[dom] / (sum(weights) + 1e-30))

        if comp_err > unitary_tol or leakage > leakage_tol:
            rejected += 1
            continue

        K_dom = kraus[dom]
        H_eff = (K_dom - K_dom.conj().T) / (2.0j * eps)
        H_eff = (H_eff + H_eff.conj().T) / 2.0
        H_eff -= (np.trace(H_eff) / d_B) * I

        nrm = np.linalg.norm(H_eff)
        if nrm > 1e-10:
            generators.append(H_eff / nrm)
            accepted += 1

    if verbose:
        print(f"  [strict] accepted={accepted} rejected={rejected} "
              f"(unitary_tol={unitary_tol}, leakage_tol={leakage_tol})")

    return generators

# ============================================================
# Independence + structure constants
# ============================================================

def find_independent_generators(generators, d_B, tol=1e-6):
    if not generators:
        return [], np.array([])

    vectors = []
    for G in generators:
        G = (G + G.conj().T) / 2.0
        G -= np.trace(G) / d_B * np.eye(d_B)

        v = []
        for i in range(d_B):
            v.append(G[i, i].real)
        for i in range(d_B):
            for j in range(i+1, d_B):
                v.append(G[i, j].real)
                v.append(G[i, j].imag)
        vectors.append(v)

    V = np.array(vectors)
    _, S, Vh = np.linalg.svd(V, full_matrices=False)

    # Relative threshold
    n_indep = int(np.sum(S > tol * (S[0] if S.size else 1.0)))
    basis_vectors = Vh[:n_indep]

    basis = []
    for v in basis_vectors:
        G = np.zeros((d_B, d_B), dtype=complex)
        idx = 0
        for i in range(d_B):
            G[i, i] = v[idx]
            idx += 1
        for i in range(d_B):
            for j in range(i+1, d_B):
            # upper triangle
                G[i, j] = v[idx] + 1j * v[idx+1]
                G[j, i] = v[idx] - 1j * v[idx+1]
                idx += 2

        G -= (np.trace(G) / d_B) * np.eye(d_B)
        nrm = np.linalg.norm(G)
        if nrm > 1e-12:
            basis.append(G / nrm)

    return basis, S

def compute_structure_constants(basis):
    n = len(basis)

    # Gram-Schmidt orthonormalize
    ortho = []
    for G in basis:
        v = G.copy()
        for prev in ortho:
            overlap = np.trace(prev.conj().T @ v).real
            v -= overlap * prev
        norm = np.sqrt(np.trace(v.conj().T @ v).real)
        if norm > 1e-12:
            ortho.append(v / norm)

    n = len(ortho)
    f = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            comm = ortho[a] @ ortho[b] - ortho[b] @ ortho[a]
            for c in range(n):
                f[a, b, c] = (-1j * np.trace(ortho[c].conj().T @ comm)).real
    return f, ortho

# ============================================================
# Small “paper-grade” experiments
# ============================================================

def run_dimension_test(d_B, variant="standard", use_strict=True, n_site_samples=500, tol=1e-6, include_random=False, seed=0):
    rng = np.random.default_rng(seed)
    H = transmission_hamiltonian(d_B, variant, rng=rng)

    if use_strict:
        gens = extract_generators_infinitesimal_strict(d_B, H, n_site_samples=n_site_samples, rng=rng)
    else:
        gens = extract_generators_infinitesimal_lenient(d_B, H, n_site_samples=n_site_samples, rng=rng)

    basis, sv = find_independent_generators(gens, d_B, tol=tol)
    found = len(basis)
    expected = d_B**2 - 1

    print(f"\n[d_B={d_B} variant={variant} strict={use_strict}] found={found} expected={expected}")
    if sv.size:
        print("  top singular values:", sv[:min(len(sv), expected+3)])

    # Optional random control (reported separately)
    if include_random:
        Hr = transmission_hamiltonian(d_B, "random", rng=rng)
        gens_r = extract_generators_infinitesimal_lenient(d_B, Hr, n_site_samples=max(200, n_site_samples//2), rng=rng)
        basis_r, _ = find_independent_generators(gens_r, d_B, tol=tol)
        print(f"  [random control] found={len(basis_r)} expected={expected}")

    return found, expected, basis, sv

if __name__ == "__main__":
    # Headline: structured Hamiltonian only (no random mixing)
    for d_B in [2, 3, 4]:
        run_dimension_test(d_B, variant="standard", use_strict=True, n_site_samples=800, tol=1e-6, include_random=False, seed=42)
