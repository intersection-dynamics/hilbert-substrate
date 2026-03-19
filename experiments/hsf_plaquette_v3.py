#!/usr/bin/env python3
"""
HSF Plaquette Spectrum — Production v3
========================================
Fix v2 failure: randomized polynomial filter gave 29,629 false singlets for 4v8l.

Root cause: polynomial filter couldn't cleanly separate ~6,750 null vectors from
~44,000 non-null vectors in the 50,625-dim Cartan sector.

Fix: Use sparse LU shift-invert eigsh for vertex 0. This gives exact eigenvalues
and clean separation. splu on a 50,625 sparse matrix uses ~2-4 GB for LU factors.

Strategy:
  Vertex 0: sparse eigsh with shift-invert via splu(Cv + sigma*I)
  Vertices 1+: dense eigh on progressively smaller projected subspaces

Usage: python hsf_plaquette_v3.py [lattice_names...]
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu, LinearOperator
from itertools import product as iterproduct
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import time
import json
import sys
import gc

# ============================================================
# SU(3) generators
# ============================================================
def su3_generators():
    T = np.zeros((8, 3, 3), dtype=complex)
    T[0][0,1] = T[0][1,0] = 0.5
    T[1][0,1] = -0.5j; T[1][1,0] = 0.5j
    T[2][0,0] = 0.5; T[2][1,1] = -0.5
    T[3][0,2] = T[3][2,0] = 0.5
    T[4][0,2] = -0.5j; T[4][2,0] = 0.5j
    T[5][1,2] = T[5][2,1] = 0.5
    T[6][1,2] = -0.5j; T[6][2,1] = 0.5j
    T[7][0,0] = T[7][1,1] = 1/(2*np.sqrt(3))
    T[7][2,2] = -1/np.sqrt(3)
    return T

T_FUND = su3_generators()
T_CONJ = np.array([-T_FUND[a].conj() for a in range(8)])

FUND_NZ = []
CONJ_NZ = []
for a in range(8):
    FUND_NZ.append([(r,c,T_FUND[a][r,c]) for r in range(3) for c in range(3) if abs(T_FUND[a][r,c])>1e-15])
    CONJ_NZ.append([(r,c,T_CONJ[a][r,c]) for r in range(3) for c in range(3) if abs(T_CONJ[a][r,c])>1e-15])

W3_T3 = np.array([0.5, -0.5, 0.0])
W3_T8 = np.array([1/(2*np.sqrt(3)), 1/(2*np.sqrt(3)), -1/np.sqrt(3)])
W3BAR_T3 = -W3_T3
W3BAR_T8 = -W3_T8

# ============================================================
# Lattice definitions
# ============================================================
def make_lattice(name):
    if name == "2v4l":
        return 2, [(0,1),(0,1),(1,0),(1,0)], "2v, 4 links (L=3)"
    elif name == "3v6l":
        return 3, [(0,1),(1,0),(1,2),(2,1),(2,0),(0,2)], "3v, 6 links (L=4)"
    elif name == "4v8l":
        return 4, [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,0),(0,3)], "Square: 4v, 8 links (L=5)"
    elif name == "5v10l":
        return 5, [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,4),(4,3),(4,0),(0,4)], "Pentagon: 5v, 10 links (L=6)"
    else:
        raise ValueError(f"Unknown: {name}")

def lattice_info(n_vertices, edges):
    n_links = len(edges)
    L = n_links - n_vertices + 1
    outgoing = {v: [] for v in range(n_vertices)}
    incoming = {v: [] for v in range(n_vertices)}
    for l, (src, dst) in enumerate(edges):
        outgoing[src].append(l)
        incoming[dst].append(l)
    plaquettes = []
    for l1, (s1, d1) in enumerate(edges):
        for l2, (s2, d2) in enumerate(edges):
            if l2 <= l1: continue
            if s1 == d2 and d1 == s2:
                plaquettes.append((l1, l2, s1, d1))
    return n_links, L, outgoing, incoming, plaquettes

# ============================================================
# Parallel Cartan filtering
# ============================================================
def _cartan_worker(args):
    chunk_start, chunk_end, n_links, n_vertices, out_lists, in_lists = args
    results = []
    state_len = 2 * n_links
    for idx in range(chunk_start, chunk_end):
        vals = []
        tmp = idx
        for _ in range(state_len):
            vals.append(tmp % 3)
            tmp //= 3
        ok = True
        for v in range(n_vertices):
            s3 = sum(W3_T3[vals[2*l]] for l in out_lists[v]) + sum(W3BAR_T3[vals[2*l+1]] for l in in_lists[v])
            if abs(s3) > 1e-10: ok = False; break
            s8 = sum(W3_T8[vals[2*l]] for l in out_lists[v]) + sum(W3BAR_T8[vals[2*l+1]] for l in in_lists[v])
            if abs(s8) > 1e-10: ok = False; break
        if ok:
            results.append(tuple(vals))
    return results

def cartan_filter(n_vertices, n_links, outgoing, incoming, n_workers):
    total = 3 ** (2 * n_links)
    out_lists = [list(outgoing[v]) for v in range(n_vertices)]
    in_lists = [list(incoming[v]) for v in range(n_vertices)]
    chunk_size = max(1, total // (n_workers * 8))
    chunks = [(s, min(s+chunk_size, total), n_links, n_vertices, out_lists, in_lists)
              for s in range(0, total, chunk_size)]
    print(f"  {total:,} configs, {len(chunks)} chunks, {n_workers} workers...", flush=True)
    with Pool(n_workers) as pool:
        results = pool.map(_cartan_worker, chunks)
    return [s for chunk in results for s in chunk]

# ============================================================
# Per-vertex Casimir (sparse)
# ============================================================
def build_vertex_casimir(cartan_states, state_to_idx, v, outgoing, incoming, n_cartan):
    """Build Casimir C_v = Σ_a (T^a_total)^2 as sparse matrix on Cartan sector.

    CRITICAL: T^a can map Cartan states to non-Cartan states. These out-of-sector
    transitions contribute to the Casimir norm (diagonal elements) even though
    the target state is outside our basis. We track ALL transitions via inv_idx
    keyed by target state (tuple or int), then compute C = A†A correctly.
    """
    out_info = [(2*l, FUND_NZ) for l in outgoing[v]]
    in_info = [(2*l+1, CONJ_NZ) for l in incoming[v]]
    all_info = out_info + in_info

    coo_r, coo_c, coo_v = [], [], []
    for a in range(8):
        # inv_idx: target_key -> [(source_ci, amplitude), ...]
        # Keys are int (in-Cartan) or tuple (out-of-Cartan)
        inv_idx = defaultdict(list)
        for ci in range(n_cartan):
            s = cartan_states[ci]
            img = {}
            for (col, nz_list) in all_info:
                k = s[col]
                for (r, c, val) in nz_list[a]:
                    if c != k: continue
                    s_new = list(s); s_new[col] = r; key = tuple(s_new)
                    if key in state_to_idx:
                        img[state_to_idx[key]] = img.get(state_to_idx[key], 0) + val
                    else:
                        img[key] = img.get(key, 0) + val
            for key, val in img.items():
                inv_idx[key].append((ci, val))

        # Casimir = Σ_a (T^a)†(T^a) = Σ_a A†A
        # For each target state, the column of A has entries (ci, val).
        # (A†A)_{ci,cj} = Σ_target conj(A_{target,ci}) * A_{target,cj}
        for key, entries in inv_idx.items():
            for ii in range(len(entries)):
                ci, vi = entries[ii]
                coo_r.append(ci); coo_c.append(ci); coo_v.append(abs(vi)**2)
                for jj in range(ii+1, len(entries)):
                    cj, vj = entries[jj]
                    val = np.conj(vi) * vj
                    if abs(val) > 1e-15:
                        coo_r.append(ci); coo_c.append(cj); coo_v.append(val)
                        coo_r.append(cj); coo_c.append(ci); coo_v.append(np.conj(val))

    Cv = sparse.coo_matrix((coo_v, (coo_r, coo_c)), shape=(n_cartan, n_cartan)).tocsr()
    return (Cv + Cv.conj().T) / 2


def find_nullspace_sparse(Cv_sparse, n):
    """
    Find null space of sparse PSD matrix using shift-invert eigsh.

    Strategy:
      1. Factor (Cv + sigma*I) with splu for shift-invert
      2. Use geometric doubling: start with k=500, double if all are null
      3. Stop when we find the boundary between null and non-null eigenvalues
    """
    sigma = 1e-3

    # Build shift-invert operator via sparse LU
    print(f"    Building splu of (Cv + {sigma}*I), n={n}...", flush=True)
    t0 = time.time()
    Cv_shifted = Cv_sparse + sigma * sparse.eye(n, format='csr')
    try:
        lu = splu(Cv_shifted.tocsc())
        print(f"    splu done ({time.time()-t0:.1f}s), nnz(L+U)={lu.nnz:,}", flush=True)
    except MemoryError:
        print(f"    splu OOM! Falling back to iterative approach.", flush=True)
        return _find_nullspace_iterative(Cv_sparse, n)

    OPinv = LinearOperator((n, n), matvec=lu.solve, dtype=complex)

    # Geometric doubling: find all null eigenvalues
    k = min(500, n - 1)
    threshold = 0.1  # eigenvalues < 0.1 are null (gap is ~3.0)
    all_null_vecs = None

    # Memory cap: eigsh uses ncv*n*16 bytes for Lanczos basis + k*n*16 for eigvecs
    # Cap at 16 GB total for eigsh work arrays
    max_k_for_memory = int(12e9 / (n * 16))  # ~12 GB for all arrays
    print(f"    Memory cap: max k ≈ {max_k_for_memory} for n={n}", flush=True)

    while k < n - 1:
        k = min(k, max_k_for_memory, n - 1)
        ncv = min(k + 100, 2 * k + 1, n - 1)
        print(f"    eigsh shift-invert: k={k}, ncv={ncv}...", flush=True)
        t0 = time.time()
        try:
            evals, evecs = eigsh(Cv_sparse, k=k, sigma=0, OPinv=OPinv,
                                  ncv=ncv, tol=1e-10, maxiter=5000)
        except Exception as e:
            print(f"    eigsh failed ({e}), relaxing tolerance...", flush=True)
            evals, evecs = eigsh(Cv_sparse, k=min(k, n-1), sigma=0, OPinv=OPinv,
                                  ncv=ncv, tol=1e-6, maxiter=10000)
        dt = time.time() - t0

        null_mask = np.abs(evals) < threshold
        n_null = int(np.sum(null_mask))
        n_nonzero = k - n_null

        # Show eigenvalue distribution
        evals_sorted = np.sort(np.abs(evals))
        print(f"    k={k}: {n_null} null, {n_nonzero} nonzero ({dt:.1f}s)", flush=True)
        if n_nonzero > 0:
            print(f"    Eigenvalues: [{evals_sorted[0]:.2e} ... "
                  f"{evals_sorted[max(n_null-1,0)]:.2e}] | "
                  f"[{evals_sorted[n_null]:.2e} ... {evals_sorted[-1]:.2e}]", flush=True)
        else:
            print(f"    All {k} eigenvalues < threshold!", flush=True)

        if n_nonzero >= 10:
            # We've found the boundary cleanly — take null vectors
            all_null_vecs = evecs[:, null_mask]
            break
        elif n_nonzero > 0:
            # Found boundary but few nonzero — get more for safety
            all_null_vecs = evecs[:, null_mask]
            print(f"    Found boundary with thin margin ({n_nonzero} nonzero). "
                  f"Accepting {n_null} null vectors.", flush=True)
            break
        else:
            # All null — need more eigenvalues
            k = min(k * 2, n - 1)
            if k >= n - 1:
                all_null_vecs = evecs[:, null_mask]
                print(f"    Reached k=n-1, all null.", flush=True)
                break

    del lu, OPinv; gc.collect()

    if all_null_vecs is None or all_null_vecs.shape[1] == 0:
        print(f"    WARNING: No null vectors found!", flush=True)
        return np.zeros((n, 0), dtype=complex)

    # Verify
    residual = np.max(np.linalg.norm(Cv_sparse @ all_null_vecs, axis=0))
    print(f"    Null space: dim={all_null_vecs.shape[1]}, "
          f"max_residual={residual:.2e}", flush=True)

    P_null, _ = np.linalg.qr(all_null_vecs, mode='reduced')
    return P_null


def _find_nullspace_iterative(Cv_sparse, n):
    """Fallback: block eigsh without shift-invert, for when splu runs OOM."""
    print(f"    Iterative eigsh fallback...", flush=True)
    # Geometric doubling without shift-invert
    k = min(500, n - 1)
    threshold = 0.1

    while k < n - 1:
        print(f"    eigsh SM: k={k}...", flush=True)
        evals, evecs = eigsh(Cv_sparse, k=k, which='SM', tol=1e-8, maxiter=10000)
        null_mask = np.abs(evals) < threshold
        n_null = int(np.sum(null_mask))
        if n_null < k:
            P_null = evecs[:, null_mask]
            break
        k = min(k * 2, n - 1)
    else:
        P_null = evecs[:, null_mask]

    residual = np.max(np.linalg.norm(Cv_sparse @ P_null, axis=0)) if P_null.shape[1] > 0 else 0
    print(f"    Fallback null space: dim={P_null.shape[1]}, residual={residual:.2e}", flush=True)

    P_null, _ = np.linalg.qr(P_null, mode='reduced')
    return P_null


# ============================================================
# Iterative singlet sector finder
# ============================================================
def find_singlet_sector(cartan_states, state_to_idx, n_vertices, outgoing, incoming, n_cartan):
    print(f"\n  Iterative null-space intersection:", flush=True)
    P = None

    for v in range(n_vertices):
        t0 = time.time()
        current_dim = P.shape[1] if P is not None else n_cartan
        Cv = build_vertex_casimir(cartan_states, state_to_idx, v, outgoing, incoming, n_cartan)
        print(f"    Vertex {v}: Casimir built (nnz={Cv.nnz:,})", flush=True)

        if P is None:
            if n_cartan <= 10000:
                # Dense path
                Cv_d = Cv.toarray(); Cv_d = (Cv_d + Cv_d.conj().T) / 2
                evals, evecs = np.linalg.eigh(Cv_d)
                P = evecs[:, np.abs(evals) < 1e-8]
                del Cv_d
            else:
                # Large sparse path: shift-invert eigsh
                P = find_nullspace_sparse(Cv, n_cartan)
        else:
            Cv_sub = P.conj().T @ (Cv @ P)
            Cv_sub = np.asarray(Cv_sub)
            Cv_sub = (Cv_sub + Cv_sub.conj().T) / 2
            evals, evecs = np.linalg.eigh(Cv_sub)
            P = P @ evecs[:, np.abs(evals) < 1e-8]

        del Cv; gc.collect()
        new_dim = P.shape[1]
        print(f"    Vertex {v}: {current_dim} → {new_dim} ({time.time()-t0:.1f}s)", flush=True)

        if new_dim == 0:
            return 0, None

    P, _ = np.linalg.qr(P, mode='reduced')
    return P.shape[1], P


# ============================================================
# Plaquette Hamiltonian (sparse)
# ============================================================
def build_plaquette_H(cartan_states, state_to_idx, plaquettes, edges, n_cartan):
    coo_r, coo_c, coo_v = [], [], []
    for p_idx, (l1, l2, vA, vB) in enumerate(plaquettes):
        for (colA, nzA, colB, nzB) in [
            (2*l1, FUND_NZ, 2*l2+1, CONJ_NZ),
            (2*l1+1, CONJ_NZ, 2*l2, FUND_NZ),
        ]:
            for a in range(8):
                for ci in range(n_cartan):
                    s = cartan_states[ci]
                    kA = s[colA]; kB = s[colB]
                    for (rA, cA, vA_val) in nzA[a]:
                        if cA != kA: continue
                        for (rB, cB, vB_val) in nzB[a]:
                            if cB != kB: continue
                            s_new = list(s)
                            s_new[colA] = rA; s_new[colB] = rB
                            key = tuple(s_new)
                            if key in state_to_idx:
                                coo_r.append(state_to_idx[key])
                                coo_c.append(ci)
                                coo_v.append(vA_val * vB_val)
        print(f"    Plaquette {p_idx} (links {l1},{l2}) done", flush=True)
    H = sparse.coo_matrix((coo_v, (coo_r, coo_c)), shape=(n_cartan, n_cartan)).tocsr()
    return (H + H.conj().T) / 2


# ============================================================
# Entanglement
# ============================================================
def bipartite_entropy(psi, cartan_states, cols_A, cols_B):
    groups = {}
    for ci, s in enumerate(cartan_states):
        kA = tuple(s[c] for c in cols_A)
        kB = tuple(s[c] for c in cols_B)
        groups.setdefault(kA, {})[kB] = groups.get(kA, {}).get(kB, 0) + psi[ci]
    A_keys = sorted(groups); B_keys = sorted(set(kb for g in groups.values() for kb in g))
    C = np.zeros((len(A_keys), len(B_keys)), dtype=complex)
    Ai = {k:i for i,k in enumerate(A_keys)}; Bi = {k:i for i,k in enumerate(B_keys)}
    for ka in groups:
        for kb in groups[ka]:
            C[Ai[ka], Bi[kb]] = groups[ka][kb]
    sv = np.linalg.svd(C, compute_uv=False)
    sv = sv[sv > 1e-12]; p = sv**2
    return -np.sum(p * np.log2(p + 1e-30)), len(sv)


# ============================================================
# Main analysis
# ============================================================
def analyze(name, n_workers=None):
    if n_workers is None:
        n_workers = min(cpu_count(), 12)

    n_vertices, edges, desc = make_lattice(name)
    n_links, L, outgoing, incoming, plaquettes = lattice_info(n_vertices, edges)
    total_dim = 9 ** n_links
    state_len = 2 * n_links

    print(f"\n{'#'*70}")
    print(f"# {desc}")
    print(f"# V={n_vertices}, E={n_links}, L={L}, dim={total_dim:,}, plaq={len(plaquettes)}")
    print(f"{'#'*70}", flush=True)

    if total_dim > 5_000_000_000:
        print(f"  Too large."); return None

    # Cartan
    print(f"\n[1/4] Cartan...", flush=True)
    t0 = time.time()
    cartan_states = cartan_filter(n_vertices, n_links, outgoing, incoming, n_workers)
    n_cartan = len(cartan_states)
    print(f"  Cartan: {n_cartan:,} ({time.time()-t0:.1f}s)", flush=True)
    if n_cartan == 0: return None
    state_to_idx = {s: i for i, s in enumerate(cartan_states)}

    # Singlet
    print(f"\n[2/4] Singlet sector...", flush=True)
    t0 = time.time()
    n_singlet, P = find_singlet_sector(cartan_states, state_to_idx, n_vertices,
                                        outgoing, incoming, n_cartan)
    print(f"  *** SINGLET: {n_singlet} *** ({time.time()-t0:.1f}s)", flush=True)

    if n_singlet < 2:
        return {'name': name, 'desc': desc, 'n_singlet': n_singlet,
                'n_vertices': n_vertices, 'n_links': n_links, 'n_loops': L,
                'n_plaquettes': len(plaquettes), 'cartan_dim': n_cartan, 'total_dim': total_dim}

    # Plaquette H
    print(f"\n[3/4] Plaquette Hamiltonian...", flush=True)
    t0 = time.time()
    H_sp = build_plaquette_H(cartan_states, state_to_idx, plaquettes, edges, n_cartan)
    print(f"  Built ({time.time()-t0:.1f}s), nnz={H_sp.nnz:,}", flush=True)

    H_s = np.asarray(P.conj().T @ (H_sp @ P))
    H_s = (H_s + H_s.conj().T) / 2
    energies, energy_vecs = np.linalg.eigh(H_s)

    levels = []
    current = [energies[0]]
    for e in energies[1:]:
        if abs(e - current[-1]) < 1e-6: current.append(e)
        else: levels.append(current); current = [e]
    levels.append(current)
    degs = [len(lev) for lev in levels]
    gap = levels[1][0] - levels[0][0] if len(levels) > 1 else 0

    # Print spectrum
    print(f"\n{'='*70}")
    print(f"  SPECTRUM: {desc}")
    print(f"{'='*70}")
    print(f"  Singlet: {n_singlet}, Levels: {len(levels)}")
    for i, lev in enumerate(levels):
        print(f"    Level {i}: E = {lev[0]:+12.8f}, deg = {len(lev)}")
    print(f"\n  E₀ = {levels[0][0]:+.8f} (deg {len(levels[0])})")
    if gap > 0:
        print(f"  ΔE = {gap:.8f}")
        print(f"  E₁ = {levels[1][0]:+.8f} (deg {len(levels[1])})")

    # Symmetry
    print(f"\n  Symmetries:", flush=True)
    link_groups = {}
    for l, (s, d) in enumerate(edges):
        link_groups.setdefault((s,d), []).append(l)
    for (s, d), group in link_groups.items():
        if len(group) < 2: continue
        la, lb = group[0], group[1]
        rows, cols, vals = [], [], []
        for ci in range(n_cartan):
            st = list(cartan_states[ci])
            sn = list(st)
            sn[2*la], sn[2*la+1] = st[2*lb], st[2*lb+1]
            sn[2*lb], sn[2*lb+1] = st[2*la], st[2*la+1]
            key = tuple(sn)
            if key in state_to_idx:
                rows.append(state_to_idx[key]); cols.append(ci); vals.append(1.0)
        S_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cartan, n_cartan))
        S_s = np.asarray(P.conj().T @ (S_mat @ P))
        comm = H_s @ S_s - S_s @ H_s
        cn = np.linalg.norm(comm)
        print(f"    Swap l{la}↔l{lb} ({s}→{d}): ||[H,S]||={cn:.2e}")
        if cn < 1e-8:
            for k in range(min(n_singlet, 20)):
                v = energy_vecs[:, k]
                exp = (v.conj() @ S_s @ v).real
                print(f"      E_{k}={energies[k]:+10.6f} <swap>={exp:+.4f}")

    # Entanglement
    print(f"\n  Entanglement:", flush=True)
    n_half = n_links // 2
    cA = [c for l in range(n_half) for c in [2*l, 2*l+1]]
    cB = [c for l in range(n_half, n_links) for c in [2*l, 2*l+1]]
    for i in range(min(n_singlet, 10)):
        psi = P @ energy_vecs[:, i]
        S, sr = bipartite_entropy(psi, cartan_states, cA, cB)
        print(f"    E_{i}={energies[i]:+10.6f}: S={S:.4f} bits, rank={sr}")

    # Binomial check
    from math import comb
    n_lev = len(degs) - 1
    binom = [comb(n_lev, k) for k in range(n_lev+1)]
    is_binom = (degs == binom)
    print(f"\n  Degeneracies: {degs}")
    print(f"  Binomial C({n_lev},k): {binom}")
    print(f"  Match: {'YES ✓' if is_binom else 'NO ✗'}")

    return {
        'name': name, 'desc': desc,
        'n_vertices': n_vertices, 'n_links': n_links, 'n_loops': L,
        'n_plaquettes': len(plaquettes), 'total_dim': total_dim,
        'cartan_dim': n_cartan, 'n_singlet': n_singlet,
        'energies': energies.tolist(),
        'levels': [(lev[0], len(lev)) for lev in levels],
        'gap': float(gap), 'degeneracies': degs, 'is_binomial': is_binom,
    }


def main():
    print("=" * 70)
    print("HSF PLAQUETTE SPECTRUM — PRODUCTION v3")
    print("=" * 70)
    n_workers = min(cpu_count(), 12)
    print(f"CPUs: {cpu_count()}, workers: {n_workers}", flush=True)

    names = sys.argv[1:] if len(sys.argv) > 1 else ["2v4l", "3v6l", "4v8l"]

    t_total = time.time()
    results = []
    for name in names:
        try:
            r = analyze(name, n_workers)
            if r: results.append(r)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback; traceback.print_exc()
        gc.collect()

    print(f"\n{'#'*70}")
    print(f"# SUMMARY")
    print(f"{'#'*70}")
    print(f"{'Name':>8} {'V':>2} {'E':>2} {'L':>2} {'P':>2} {'Cartan':>10} "
          f"{'Sing':>6} {'Gap':>8} {'Degens':>25} {'Binom':>6}")
    print("-" * 80)
    for r in results:
        g = f"{r.get('gap',0):.4f}" if r.get('gap',0) > 0 else "---"
        d = str(r.get('degeneracies','?'))
        b = "✓" if r.get('is_binomial') else ""
        print(f"{r['name']:>8} {r['n_vertices']:>2} {r['n_links']:>2} {r['n_loops']:>2} "
              f"{r['n_plaquettes']:>2} {r['cartan_dim']:>10,} {r['n_singlet']:>6} "
              f"{g:>8} {d:>25} {b:>6}")

    with open("hsf_plaquette_v3.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to hsf_plaquette_v3.json")
    print(f"Total: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    main()