#!/usr/bin/env python3
"""
Entanglement Precedes Locality: Experimental Suite (REWRITTEN + PARALLEL)
========================================================================

Adds:
- Parallel per-seed execution via multiprocessing (Windows-safe, spawn).
- --jobs controls worker count.
- Optional per-seed JSON outputs in --outdir.

Key fixes vs original legacy script:
- Tracks and applies the actual recovery circuit to the scrambled state:
    psi_after = U_rec @ psi_scr
- Sector additivity robust + explicit:
    * commutator ||[H, Z_total]||/||H|| decides if sectors are defined
    * if defined, diagonalize inside exact magnetization subspaces
- Horodecki-optimized CHSH S_max per pair

Author: Ben Bray
Rewrite: Jan 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from numpy.linalg import eigh


# =============================================================================
# BASIC OPERATORS / UTILITIES
# =============================================================================

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_list(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def single_qubit_op(n_qubits: int, site: int, op2: np.ndarray) -> np.ndarray:
    ops = [I2] * n_qubits
    ops[site] = op2
    return kron_list(ops)


def two_qubit_op(n_qubits: int, i: int, j: int, op_i: np.ndarray, op_j: np.ndarray) -> np.ndarray:
    ops = [I2] * n_qubits
    ops[i] = op_i
    ops[j] = op_j
    return kron_list(ops)


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    ph = d / np.abs(d)
    Q = Q * ph.conj()
    return Q


def random_su4(rng: np.random.Generator) -> np.ndarray:
    U = random_unitary(4, rng)
    det = np.linalg.det(U)
    U = U / det ** (1 / 4)
    return U


def embed_two_qubit_unitary(n_qubits: int, i: int, j: int, U4: np.ndarray) -> np.ndarray:
    if i == j:
        raise ValueError("i and j must differ")
    if U4.shape != (4, 4):
        raise ValueError("U4 must be 4x4")

    n = n_qubits
    dim = 2 ** n

    rest = [k for k in range(n) if k not in (i, j)]
    perm = [i, j] + rest

    P = np.zeros((dim, dim), dtype=complex)
    for basis in range(dim):
        bits = [(basis >> k) & 1 for k in range(n)]
        new_bits = [bits[p] for p in perm]
        new_basis = sum((new_bits[k] << k) for k in range(n))
        P[new_basis, basis] = 1.0

    U_big = np.kron(U4, np.eye(2 ** (n - 2), dtype=complex))
    return P.conj().T @ U_big @ P


def ground_state(H: np.ndarray) -> Tuple[float, np.ndarray]:
    evals, evecs = eigh(H)
    psi0 = evecs[:, 0]
    psi0 = psi0 / np.linalg.norm(psi0)
    return float(np.real(evals[0])), psi0


# =============================================================================
# GRAPH HELPERS / LOCALITY METRIC
# =============================================================================

def ring_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def chain_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def random_edges(n: int, m: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = set()
    while len(edges) < m:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        edges.add((a, b))
    return list(edges)


def build_xx_hamiltonian(n_qubits: int, edges: List[Tuple[int, int]], J: float = 1.0) -> np.ndarray:
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for i, j in edges:
        H += (-J / 2.0) * (two_qubit_op(n_qubits, i, j, X, X) + two_qubit_op(n_qubits, i, j, Y, Y))
    H = 0.5 * (H + H.conj().T)
    return H


def locality_fraction(H: np.ndarray, n_qubits: int, edges: List[Tuple[int, int]]) -> float:
    n = n_qubits
    edge_set = set(edges) | set((b, a) for a, b in edges)

    total_w = 0.0
    local_w = 0.0

    labels = ["I", "X", "Y", "Z"]
    dim = 2 ** n

    single_ops = {(site, lab): single_qubit_op(n, site, PAULIS[lab]) for site in range(n) for lab in PAULIS.keys()}

    def coeff_of(op: np.ndarray) -> complex:
        return np.trace(op.conj().T @ H) / dim

    for i in range(n):
        for j in range(i + 1, n):
            for a_lab in labels:
                for b_lab in labels:
                    if a_lab == "I" and b_lab == "I":
                        continue
                    if a_lab == "I" or b_lab == "I":
                        continue
                    op = single_ops[(i, a_lab)] @ single_ops[(j, b_lab)]
                    c = coeff_of(op)
                    w = float(np.abs(c) ** 2)
                    total_w += w
                    if (i, j) in edge_set:
                        local_w += w

    if total_w < 1e-15:
        return 0.0
    return local_w / total_w


# =============================================================================
# ENTANGLEMENT METRICS
# =============================================================================

def reduced_density_matrix(psi: np.ndarray, n_qubits: int, keep: List[int]) -> np.ndarray:
    keep = list(keep)
    n = n_qubits
    dim = 2 ** n
    if psi.shape != (dim,):
        raise ValueError("psi must be a state vector of size 2^n")

    rho_full = np.outer(psi, psi.conj())
    traced = [q for q in range(n) if q not in keep]
    perm = keep + traced

    rho_t = rho_full.reshape([2] * n + [2] * n)
    ket_axes = perm
    bra_axes = [q + n for q in perm]
    rho_t = np.transpose(rho_t, axes=ket_axes + bra_axes)

    k = len(keep)
    rho_t = rho_t.reshape(2 ** k, 2 ** (n - k), 2 ** k, 2 ** (n - k))
    rho_red = np.einsum("a e b e -> a b", rho_t)
    return rho_red


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    evals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    evals = np.real(evals)
    evals = np.clip(evals, 0.0, 1.0)
    evals = evals[evals > eps]
    if evals.size == 0:
        return 0.0
    return float(-np.sum(evals * np.log(evals)))


def mean_bipartite_entropy(psi: np.ndarray, n_qubits: int) -> float:
    ent = []
    for i in range(n_qubits):
        rho_i = reduced_density_matrix(psi, n_qubits, [i])
        ent.append(von_neumann_entropy(rho_i))
    return float(np.mean(ent))


def mutual_information_matrix(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    n = n_qubits
    S1 = [von_neumann_entropy(reduced_density_matrix(psi, n, [i])) for i in range(n)]
    MI = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rho_ij = reduced_density_matrix(psi, n, [i, j])
            Sij = von_neumann_entropy(rho_ij)
            MI[i, j] = S1[i] + S1[j] - Sij
            MI[j, i] = MI[i, j]
    return MI


def entanglement_graph_compatibility(MI: np.ndarray, edges: List[Tuple[int, int]]) -> float:
    n = MI.shape[0]
    edge_set = set(edges) | set((j, i) for i, j in edges)

    edge_MI = []
    nonedge_MI = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in edge_set:
                edge_MI.append(MI[i, j])
            else:
                nonedge_MI.append(MI[i, j])

    if not edge_MI or not nonedge_MI:
        return 0.0

    mean_edge = float(np.mean(edge_MI))
    mean_nonedge = float(np.mean(nonedge_MI))
    if mean_nonedge < 1e-12:
        return mean_edge
    return mean_edge / (mean_edge + mean_nonedge)


def corrcoef_flat(A: np.ndarray, B: np.ndarray) -> float:
    n = A.shape[0]
    a = []
    b = []
    for i in range(n):
        for j in range(i + 1, n):
            a.append(A[i, j])
            b.append(B[i, j])
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# =============================================================================
# RECOVERY (TRACK THE CIRCUIT!)
# =============================================================================

@dataclass
class RecoveryResult:
    H_rec: np.ndarray
    U_rec: np.ndarray
    local_fraction_history: List[float]


def locality_recovery(
    H_start: np.ndarray,
    n_qubits: int,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
    sweeps: int = 20,
    trials_per_edge: int = 30,
    verbose: bool = True,
) -> RecoveryResult:
    H = H_start.copy()
    dim = 2 ** n_qubits
    U_total = np.eye(dim, dtype=complex)
    history = []

    def score(H_: np.ndarray) -> float:
        return locality_fraction(H_, n_qubits, edges)

    best_score = score(H)
    history.append(best_score)

    for s in range(sweeps):
        improved = False
        for (i, j) in edges:
            edge_best_score = best_score
            edge_best_H = H
            edge_best_U = None

            for _ in range(trials_per_edge):
                U4 = random_su4(rng)
                U = embed_two_qubit_unitary(n_qubits, i, j, U4)
                Hp = U @ H @ U.conj().T
                sc = score(Hp)
                if sc > edge_best_score + 1e-12:
                    edge_best_score = sc
                    edge_best_H = Hp
                    edge_best_U = U

            if edge_best_U is not None:
                H = edge_best_H
                U_total = edge_best_U @ U_total
                best_score = edge_best_score
                improved = True

        history.append(best_score)
        if verbose:
            print(f"  Sweep {s}: local fraction = {best_score:.6f}")
        if not improved:
            break

    return RecoveryResult(H_rec=H, U_rec=U_total, local_fraction_history=history)


# =============================================================================
# SECTOR ADDITIVITY (ROBUST + EXPLICIT)
# =============================================================================

def z_total_operator(n_qubits: int) -> np.ndarray:
    return sum(single_qubit_op(n_qubits, i, Z) for i in range(n_qubits))


def commutator_relative_norm(H: np.ndarray, A: np.ndarray) -> float:
    C = H @ A - A @ H
    num = np.linalg.norm(C, ord="fro")
    den = np.linalg.norm(H, ord="fro")
    if den < 1e-15:
        return float("inf")
    return float(num / den)


def basis_indices_with_k_ones(n_qubits: int, k: int) -> np.ndarray:
    dim = 2 ** n_qubits
    out = []
    for b in range(dim):
        if int(b).bit_count() == k:
            out.append(b)
    return np.array(out, dtype=int)


def sector_eigenvalues_by_k(H: np.ndarray, n_qubits: int, k: int) -> np.ndarray:
    idx = basis_indices_with_k_ones(n_qubits, k)
    if idx.size == 0:
        return np.array([], dtype=float)
    Hk = H[np.ix_(idx, idx)]
    evals = np.linalg.eigvalsh(0.5 * (Hk + Hk.conj().T))
    return np.real(evals)


def sector_additivity_error(
    H: np.ndarray,
    n_qubits: int,
    comm_tol: float = 1e-10,
) -> Tuple[float, float]:
    Ztot = z_total_operator(n_qubits)
    comm_rel = commutator_relative_norm(H, Ztot)

    if comm_rel > comm_tol:
        return (np.nan, comm_rel)

    eps_1 = np.sort(sector_eigenvalues_by_k(H, n_qubits, k=1))
    E_2 = np.sort(sector_eigenvalues_by_k(H, n_qubits, k=2))

    if eps_1.size == 0 or E_2.size == 0:
        return (np.nan, comm_rel)

    predicted = []
    for i in range(len(eps_1)):
        for j in range(i + 1, len(eps_1)):
            predicted.append(eps_1[i] + eps_1[j])
    predicted = np.sort(np.array(predicted, dtype=float))

    if predicted.shape != E_2.shape:
        return (np.nan, comm_rel)

    err = float(np.sqrt(np.mean((E_2 - predicted) ** 2)))
    return (err, comm_rel)


# =============================================================================
# JORDAN-WIGNER "QUADRATIC FRACTION"
# =============================================================================

def jw_fermion_ops(n_qubits: int) -> List[np.ndarray]:
    ops = []
    for j in range(n_qubits):
        prefix = [Z] * j
        mid = (X + 1j * Y) / 2.0
        suffix = [I2] * (n_qubits - j - 1)
        ops.append(kron_list(prefix + [mid] + suffix))
    return ops


def jw_anticomm_violations(c_ops: List[np.ndarray]) -> Dict[str, float]:
    n = len(c_ops)
    dim = c_ops[0].shape[0]
    I = np.eye(dim, dtype=complex)

    max_cc = 0.0
    max_ccdag = 0.0
    for i in range(n):
        for j in range(n):
            cc = c_ops[i] @ c_ops[j] + c_ops[j] @ c_ops[i]
            max_cc = max(max_cc, float(np.linalg.norm(cc, ord="fro")))
            ccdag = c_ops[i] @ c_ops[j].conj().T + c_ops[j].conj().T @ c_ops[i]
            diff = (ccdag - I) if i == j else ccdag
            max_ccdag = max(max_ccdag, float(np.linalg.norm(diff, ord="fro")))
    return {"max_cc_violation": max_cc, "max_ccdag_violation": max_ccdag}


def quadratic_fraction_jw(H: np.ndarray, c_ops: List[np.ndarray]) -> float:
    dim = H.shape[0]
    basis = []
    for i in range(len(c_ops)):
        for j in range(len(c_ops)):
            basis.append(c_ops[i].conj().T @ c_ops[j])

    coeffs = []
    for B in basis:
        coeffs.append(np.trace(B.conj().T @ H) / dim)
    coeffs = np.array(coeffs, dtype=complex)

    Hproj = np.zeros_like(H)
    for c, B in zip(coeffs, basis):
        Hproj += c * B

    num = np.linalg.norm(Hproj, ord="fro") ** 2
    den = np.linalg.norm(H, ord="fro") ** 2
    if den < 1e-15:
        return 0.0
    return float(num / den)


# =============================================================================
# BELL / CHSH (HORODECKI-OPTIMIZED)
# =============================================================================

def horodecki_chsh_smax_for_pair(psi: np.ndarray, n_qubits: int, i: int, j: int) -> float:
    rho = reduced_density_matrix(psi, n_qubits, [i, j])
    sigmas = [X, Y, Z]
    T = np.zeros((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            op = np.kron(sigmas[a], sigmas[b])
            T[a, b] = float(np.real(np.trace(rho @ op)))
    M = T.T @ T
    evals = np.sort(np.real(np.linalg.eigvalsh(M)))[::-1]
    m1 = max(0.0, float(evals[0]))
    m2 = max(0.0, float(evals[1]))
    return float(2.0 * math.sqrt(m1 + m2))


def max_pairwise_chsh(psi: np.ndarray, n_qubits: int) -> Tuple[float, float]:
    vals = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            vals.append(horodecki_chsh_smax_for_pair(psi, n_qubits, i, j))
    vals = np.array(vals, dtype=float)
    return float(np.max(vals)), float(np.mean(vals))


# =============================================================================
# EXPERIMENTS
# =============================================================================

def scramble_hamiltonian_and_state(H0: np.ndarray, psi0: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = H0.shape[0]
    U = random_unitary(dim, rng)
    H_scr = U @ H0 @ U.conj().T
    psi_scr = U @ psi0
    psi_scr = psi_scr / np.linalg.norm(psi_scr)
    H_scr = 0.5 * (H_scr + H_scr.conj().T)
    return H_scr, psi_scr, U


def run_experiment1(n_qubits: int, graph: str, seed: int,
                    recovery_sweeps: int = 20, trials_per_edge: int = 30,
                    verbose: bool = True) -> Dict:
    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Entanglement through Scramble/Recovery (CIRCUIT-TRACKED)")
        print(f"n_qubits={n_qubits}, graph={graph}, seed={seed}")
        print("=" * 60)

    rng = np.random.default_rng(seed)

    if graph == "ring":
        edges = ring_edges(n_qubits)
    elif graph == "chain":
        edges = chain_edges(n_qubits)
    else:
        raise ValueError("graph must be ring or chain")

    H0 = build_xx_hamiltonian(n_qubits, edges)
    _, psi0 = ground_state(H0)

    init_local = locality_fraction(H0, n_qubits, edges)
    init_S = mean_bipartite_entropy(psi0, n_qubits)

    H_scr, psi_scr, _ = scramble_hamiltonian_and_state(H0, psi0, rng)
    scr_local = locality_fraction(H_scr, n_qubits, edges)
    scr_S = mean_bipartite_entropy(psi_scr, n_qubits)

    if verbose:
        print("\nInitial state:")
        print(f"  Local fraction: {init_local:.6f}")
        print(f"  Mean bipartite entropy: {init_S:.4f}")

        print("\nScrambled state:")
        print(f"  Local fraction: {scr_local:.6f}")
        print(f"  Mean bipartite entropy: {scr_S:.4f}")

        print("\nRunning locality recovery (tracking recovery unitary)...")

    rec = locality_recovery(
        H_start=H_scr,
        n_qubits=n_qubits,
        edges=edges,
        rng=rng,
        sweeps=recovery_sweeps,
        trials_per_edge=trials_per_edge,
        verbose=verbose,
    )

    H_rec = rec.H_rec
    psi_after = rec.U_rec @ psi_scr
    psi_after = psi_after / np.linalg.norm(psi_after)

    _, psi_gs_rec = ground_state(H_rec)

    rec_local = locality_fraction(H_rec, n_qubits, edges)
    after_S = mean_bipartite_entropy(psi_after, n_qubits)
    gsrec_S = mean_bipartite_entropy(psi_gs_rec, n_qubits)

    MI_scr = mutual_information_matrix(psi_scr, n_qubits)
    MI_after = mutual_information_matrix(psi_after, n_qubits)
    MI_corr = corrcoef_flat(MI_scr, MI_after)

    overlap = float(np.abs(np.vdot(psi_gs_rec, psi_after)))

    if verbose:
        print("\nRecovered (Hamiltonian):")
        print(f"  Local fraction: {rec_local:.6f}")
        print("\nRecovered (State via circuit):")
        print(f"  Mean bipartite entropy: {after_S:.4f}")
        print("\nGround state of recovered Hamiltonian (diagnostic):")
        print(f"  Mean bipartite entropy: {gsrec_S:.4f}")
        print(f"  |<psi_after | gs(H_rec)>|: {overlap:.6f}")

        print("\n--- KEY FINDING ---")
        print(f"Entropy change (scrambled vs initial): {scr_S - init_S:+.4f}")
        print(f"Entropy change (after-recovery-circuit vs scrambled): {after_S - scr_S:+.4f}")
        print(f"MI correlation (scrambled vs after-recovery-circuit): {MI_corr:.4f}")

    return {
        "seed": seed,
        "initial_local": float(init_local),
        "scrambled_local": float(scr_local),
        "recovered_local": float(rec_local),
        "initial_entropy_mean": float(init_S),
        "scrambled_entropy_mean": float(scr_S),
        "recovered_entropy_mean": float(after_S),
        "recovered_entropy_gsHrec_mean": float(gsrec_S),
        "MI_correlation_scrambled_recovered": float(MI_corr),
        "recovery_state_overlap_with_gs": float(overlap),
        "recovery_local_fraction_history": [float(x) for x in rec.local_fraction_history],
    }


def run_experiment2(n_qubits: int, seed: int,
                    recovery_sweeps: int = 20, trials_per_edge: int = 30,
                    verbose: bool = True) -> Dict:
    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Entanglement Constrains Geometry (STATE-CONSISTENT)")
        print(f"n_qubits={n_qubits}, seed={seed}")
        print("=" * 60)

    rng = np.random.default_rng(seed)

    source_edges = ring_edges(n_qubits)
    H0 = build_xx_hamiltonian(n_qubits, source_edges)
    _, psi0 = ground_state(H0)

    H_scr, psi_scr, _ = scramble_hamiltonian_and_state(H0, psi0, rng)
    MI_scr = mutual_information_matrix(psi_scr, n_qubits)

    targets = [
        ("ring", ring_edges(n_qubits)),
        ("chain", chain_edges(n_qubits)),
        ("random1", random_edges(n_qubits, m=n_qubits, rng=rng)),
        ("random2", random_edges(n_qubits, m=n_qubits, rng=rng)),
    ]

    final_locals = []
    compat = []

    for name, edges in targets:
        if verbose:
            print(f"\nRecovery to {name}...")
        rec = locality_recovery(
            H_start=H_scr,
            n_qubits=n_qubits,
            edges=edges,
            rng=rng,
            sweeps=recovery_sweeps,
            trials_per_edge=trials_per_edge,
            verbose=verbose,
        )

        comp = entanglement_graph_compatibility(MI_scr, edges)
        lf = locality_fraction(rec.H_rec, n_qubits, edges)

        if verbose:
            print(f"  Final local fraction: {lf:.4f}")
            print(f"  MI compatibility (from scrambled state): {comp:.4f}")

        final_locals.append(float(lf))
        compat.append(float(comp))

    corr = float(np.corrcoef(np.array(compat), np.array(final_locals))[0, 1]) if np.std(compat) > 1e-12 else 0.0

    if verbose:
        print("\n--- KEY FINDING ---")
        print(f"Correlation(MI compatibility, recovery success): {corr:.4f}")

    return {
        "seed": seed,
        "target_graphs": [t[0] for t in targets],
        "final_local_fractions": final_locals,
        "mi_compatibility": compat,
        "correlation": corr,
    }


def run_experiment3(n_qubits: int, seed: int,
                    recovery_sweeps: int = 20, trials_per_edge: int = 30,
                    verbose: bool = True) -> Dict:
    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Jordan-Wigner Survival + Sector Additivity (ROBUST)")
        print(f"n_qubits={n_qubits}, seed={seed}")
        print("=" * 60)

    rng = np.random.default_rng(seed)

    edges = ring_edges(n_qubits)
    H0 = build_xx_hamiltonian(n_qubits, edges)
    _, psi0 = ground_state(H0)

    H_scr, _, _ = scramble_hamiltonian_and_state(H0, psi0, rng)

    if verbose:
        print("\nRunning locality recovery (tracking unitary)...")

    rec = locality_recovery(
        H_start=H_scr,
        n_qubits=n_qubits,
        edges=edges,
        rng=rng,
        sweeps=recovery_sweeps,
        trials_per_edge=trials_per_edge,
        verbose=verbose,
    )
    H_rec = rec.H_rec

    c_ops = jw_fermion_ops(n_qubits)

    anti0 = jw_anticomm_violations(c_ops)
    antiS = jw_anticomm_violations(c_ops)
    antiR = jw_anticomm_violations(c_ops)

    Q0 = quadratic_fraction_jw(H0, c_ops)
    QS = quadratic_fraction_jw(H_scr, c_ops)
    QR = quadratic_fraction_jw(H_rec, c_ops)

    add0, comm0 = sector_additivity_error(H0, n_qubits)
    addS, commS = sector_additivity_error(H_scr, n_qubits)
    addR, commR = sector_additivity_error(H_rec, n_qubits)

    if verbose:
        print("\nInitial (local) Hamiltonian:")
        print(f"  Anticomm violations: {anti0}")
        print(f"  Quadratic fraction Q: {Q0:.6f}")
        print(f"  Sector additivity error: {add0}  (comm ||[H,Z]||/||H|| = {comm0:.3e})")

        print("\nScrambled (nonlocal) Hamiltonian:")
        print(f"  Anticomm violations: {antiS}")
        print(f"  Quadratic fraction Q: {QS:.6f}")
        print(f"  Sector additivity error: {addS}  (comm ||[H,Z]||/||H|| = {commS:.3e})")

        print("\nRecovered Hamiltonian:")
        print(f"  Anticomm violations: {antiR}")
        print(f"  Quadratic fraction Q: {QR:.6f}")
        print(f"  Sector additivity error: {addR}  (comm ||[H,Z]||/||H|| = {commR:.3e})")

    return {
        "seed": seed,
        "initial_Q": float(Q0),
        "scrambled_Q": float(QS),
        "recovered_Q": float(QR),
        "initial_anticomm_cc": float(anti0["max_cc_violation"]),
        "scrambled_anticomm_cc": float(antiS["max_cc_violation"]),
        "recovered_anticomm_cc": float(antiR["max_cc_violation"]),
        "initial_sector_additivity": None if np.isnan(add0) else float(add0),
        "scrambled_sector_additivity": None if np.isnan(addS) else float(addS),
        "recovered_sector_additivity": None if np.isnan(addR) else float(addR),
        "initial_comm_rel": float(comm0),
        "scrambled_comm_rel": float(commS),
        "recovered_comm_rel": float(commR),
    }


def run_experiment4(n_qubits: int, seed: int,
                    recovery_sweeps: int = 20, trials_per_edge: int = 30,
                    verbose: bool = True) -> Dict:
    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: Bell Correlations (HORODECKI OPTIMIZED)")
        print(f"n_qubits={n_qubits}, seed={seed}")
        print("=" * 60)

    rng = np.random.default_rng(seed)

    edges = ring_edges(n_qubits)
    H0 = build_xx_hamiltonian(n_qubits, edges)
    _, psi0 = ground_state(H0)

    H_scr, psi_scr, _ = scramble_hamiltonian_and_state(H0, psi0, rng)

    if verbose:
        print("\nRunning locality recovery (tracking unitary)...")

    rec = locality_recovery(
        H_start=H_scr,
        n_qubits=n_qubits,
        edges=edges,
        rng=rng,
        sweeps=recovery_sweeps,
        trials_per_edge=trials_per_edge,
        verbose=verbose,
    )
    psi_after = rec.U_rec @ psi_scr
    psi_after = psi_after / np.linalg.norm(psi_after)

    init_max, init_mean = max_pairwise_chsh(psi0, n_qubits)
    scr_max, scr_mean = max_pairwise_chsh(psi_scr, n_qubits)
    aft_max, aft_mean = max_pairwise_chsh(psi_after, n_qubits)

    def smax_matrix(psi: np.ndarray) -> np.ndarray:
        n = n_qubits
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = horodecki_chsh_smax_for_pair(psi, n, i, j)
                M[j, i] = M[i, j]
        return M

    S_scr = smax_matrix(psi_scr)
    S_aft = smax_matrix(psi_after)
    corr = corrcoef_flat(S_scr, S_aft)

    if verbose:
        print("\nClassical bound: 2.0, Tsirelson: 2.8284")
        print("\nInitial state (Horodecki):")
        print(f"  Max S_max:  {init_max:.4f}")
        print(f"  Mean S_max: {init_mean:.4f}")
        print(f"  Bell violation present? {init_max > 2.0}")

        print("\nScrambled state (Horodecki):")
        print(f"  Max S_max:  {scr_max:.4f}")
        print(f"  Mean S_max: {scr_mean:.4f}")
        print(f"  Bell violation present? {scr_max > 2.0}")

        print("\nAfter recovery circuit (Horodecki):")
        print(f"  Max S_max:  {aft_max:.4f}")
        print(f"  Mean S_max: {aft_mean:.4f}")
        print(f"  Bell violation present? {aft_max > 2.0}")

        print("\n--- KEY FINDING ---")
        print("CHSH correlation (scrambled↔after recovery circuit):")
        print(f"  Correlation coefficient: {corr:.4f}")

    return {
        "seed": seed,
        "initial_max_CHSH": float(init_max),
        "scrambled_max_CHSH": float(scr_max),
        "recovered_max_CHSH": float(aft_max),
        "initial_mean_CHSH": float(init_mean),
        "scrambled_mean_CHSH": float(scr_mean),
        "recovered_mean_CHSH": float(aft_mean),
        "CHSH_correlation": float(corr),
    }


# =============================================================================
# PER-SEED RUNNER (FOR PARALLELISM)
# =============================================================================

def run_one_seed(payload: Dict) -> Dict:
    """
    Worker entry point. Must be top-level function for Windows spawn.
    """
    n_qubits = int(payload["n_qubits"])
    seed = int(payload["seed"])
    recovery_sweeps = int(payload["recovery_sweeps"])
    trials_per_edge = int(payload["trials_per_edge"])
    outdir = payload.get("outdir")
    quiet = bool(payload.get("quiet", False))

    # In parallel mode, default to less noisy output to avoid interleaving.
    verbose = not quiet

    if verbose:
        print("\n" + "#" * 70)
        print(f"# SEED: {seed}")
        print("#" * 70)

    results = {
        "seed": seed,
        "experiment1": run_experiment1(n_qubits, "ring", seed, recovery_sweeps, trials_per_edge, verbose=verbose),
        "experiment2": run_experiment2(n_qubits, seed, recovery_sweeps, trials_per_edge, verbose=verbose),
        "experiment3": run_experiment3(n_qubits, seed, recovery_sweeps, trials_per_edge, verbose=verbose),
        "experiment4": run_experiment4(n_qubits, seed, recovery_sweeps, trials_per_edge, verbose=verbose),
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"seed_{seed}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def merge_seed_results(n_qubits: int, seeds: List[int], per_seed: List[Dict]) -> Dict:
    """
    Convert per-seed dicts into the legacy-ish structure:
      experiment1: [ ... per seed ... ]
      ...
    """
    per_seed_sorted = sorted(per_seed, key=lambda d: d["seed"])

    out = {
        "metadata": {
            "n_qubits": n_qubits,
            "seeds": [d["seed"] for d in per_seed_sorted],
        },
        "experiment1": [],
        "experiment2": [],
        "experiment3": [],
        "experiment4": [],
    }

    for d in per_seed_sorted:
        out["experiment1"].append(d["experiment1"])
        out["experiment2"].append(d["experiment2"])
        out["experiment3"].append(d["experiment3"])
        out["experiment4"].append(d["experiment4"])

    return out


# =============================================================================
# SUITE RUNNER + CLI
# =============================================================================

def run_all_experiments(
    n_qubits: int,
    seeds: List[int],
    output_file: str,
    recovery_sweeps: int = 20,
    trials_per_edge: int = 30,
    jobs: int = 1,
    outdir: Optional[str] = None,
    quiet_workers: bool = True,
) -> Dict:
    print("=" * 70)
    print("ENTANGLEMENT PRECEDES LOCALITY: FULL EXPERIMENTAL SUITE (PARALLEL)")
    print("=" * 70)
    print(f"n_qubits: {n_qubits}")
    print(f"seeds: {seeds}")
    print(f"output: {output_file}")
    print(f"jobs: {jobs}")
    if outdir:
        print(f"per-seed outdir: {outdir}")
    print("=" * 70)

    # Serial (simple, still supported)
    if jobs <= 1:
        per_seed_results = []
        for seed in seeds:
            per_seed_results.append(
                run_one_seed({
                    "n_qubits": n_qubits,
                    "seed": seed,
                    "recovery_sweeps": recovery_sweeps,
                    "trials_per_edge": trials_per_edge,
                    "outdir": outdir,
                    "quiet": False,
                })
            )
        merged = merge_seed_results(n_qubits, seeds, per_seed_results)
    else:
        # Parallel per-seed
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        tasks = [{
            "n_qubits": n_qubits,
            "seed": seed,
            "recovery_sweeps": recovery_sweeps,
            "trials_per_edge": trials_per_edge,
            "outdir": outdir,
            "quiet": quiet_workers,
        } for seed in seeds]

        with ctx.Pool(processes=jobs) as pool:
            per_seed_results = pool.map(run_one_seed, tasks)

        merged = merge_seed_results(n_qubits, seeds, per_seed_results)

    # Quick summary
    e1 = merged["experiment1"]
    print("\n" + "-" * 70)
    print("SUMMARY (means over seeds)")
    print("-" * 70)
    print(f"  Initial local frac:    {np.mean([d['initial_local'] for d in e1]):.4f}")
    print(f"  Scrambled local frac:  {np.mean([d['scrambled_local'] for d in e1]):.4f}")
    print(f"  Recovered local frac:  {np.mean([d['recovered_local'] for d in e1]):.4f}")
    print(f"  Initial entropy mean:  {np.mean([d['initial_entropy_mean'] for d in e1]):.4f}")
    print(f"  Scrambled entropy:     {np.mean([d['scrambled_entropy_mean'] for d in e1]):.4f}")
    print(f"  After-circuit entropy: {np.mean([d['recovered_entropy_mean'] for d in e1]):.4f}")
    print(f"  MI corr (scr↔after):   {np.mean([d['MI_correlation_scrambled_recovered'] for d in e1]):.4f}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return merged


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entanglement Precedes Locality: experimental suite (rewritten + parallel)."
    )
    p.add_argument("--n_qubits", type=int, default=6, help="Number of qubits")
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456], help="RNG seeds")
    p.add_argument("--output", type=str, default="entanglement_locality_results.json", help="Output JSON file")
    p.add_argument("--experiment", type=int, default=0, choices=[0, 1, 2, 3, 4],
                   help="0=all, 1..4 run a single experiment (serial)")
    p.add_argument("--recovery-sweeps", type=int, default=20, help="Max recovery sweeps")
    p.add_argument("--trials-per-edge", type=int, default=30, help="Random SU(4) trials per edge per sweep")

    # Parallel controls
    p.add_argument("--jobs", type=int, default=1, help="Number of worker processes (seeds in parallel)")
    p.add_argument("--outdir", type=str, default="", help="Optional directory for per-seed JSON outputs")
    p.add_argument("--no-quiet-workers", action="store_true", help="Do not quiet worker printing (can interleave output)")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    outdir = args.outdir.strip() if args.outdir.strip() else None
    quiet_workers = not args.no_quiet_workers

    if args.experiment == 0:
        run_all_experiments(
            n_qubits=args.n_qubits,
            seeds=args.seeds,
            output_file=args.output,
            recovery_sweeps=args.recovery_sweeps,
            trials_per_edge=args.trials_per_edge,
            jobs=args.jobs,
            outdir=outdir,
            quiet_workers=quiet_workers,
        )
    else:
        # Single-experiment mode is kept serial to keep output readable and avoid partial merges.
        # If you want parallel single-experiment, we can add it, but the suite is the main target.
        if args.experiment == 1:
            for seed in args.seeds:
                run_experiment1(args.n_qubits, "ring", seed, args.recovery_sweeps, args.trials_per_edge, verbose=True)
        elif args.experiment == 2:
            for seed in args.seeds:
                run_experiment2(args.n_qubits, seed, args.recovery_sweeps, args.trials_per_edge, verbose=True)
        elif args.experiment == 3:
            for seed in args.seeds:
                run_experiment3(args.n_qubits, seed, args.recovery_sweeps, args.trials_per_edge, verbose=True)
        elif args.experiment == 4:
            for seed in args.seeds:
                run_experiment4(args.n_qubits, seed, args.recovery_sweeps, args.trials_per_edge, verbose=True)


if __name__ == "__main__":
    main()
