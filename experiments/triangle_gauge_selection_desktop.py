#!/usr/bin/env python3
"""
triangle_gauge_selection_desktop.py
====================================
HSF Paper III — Gauge Group Selection (Optimized for Desktop)

Optimized for 12-core / 32 GB systems. Key improvements over the
container version:
  - Upfront memory estimates before committing to computation
  - Shift-invert eigsh for faster near-zero eigenvalue search
  - Chunked sparse construction to limit peak memory
  - Vertex-only mode (--vertex-only) for the rep-theory proof
  - Adjustable N range (--nmax)

RESOURCE ESTIMATES:
  N=2:    512 dim,       ~1 MB,    <1s
  N=3:    19,683 dim,    ~10 MB,   ~5s
  N=4:    262,144 dim,   ~200 MB,  ~2 min
  N=5:    1,953,125 dim, ~3 GB,    ~30 min  (feasible on 32 GB)
  N=6:    10,077,696 dim, ~25 GB,  ~hours   (tight on 32 GB)

USAGE:
  python triangle_gauge_selection_desktop.py              # N=2..5
  python triangle_gauge_selection_desktop.py --nmax 6     # include N=6 (needs ~25 GB)
  python triangle_gauge_selection_desktop.py --vertex-only --nmax 12  # rep theory only
  python triangle_gauge_selection_desktop.py --nmax 5 --skip-vertex   # lattice only

DEPENDENCIES: numpy, scipy
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
import time
import json
import os
import sys
import argparse
import gc


# ═══════════════════════════════════════════════════════════════════════
#  SU(N) generators for general N
# ═══════════════════════════════════════════════════════════════════════

def sun_generators(N):
    """Generalized Gell-Mann matrices / 2."""
    gens = []
    for j in range(N):
        for k in range(j + 1, N):
            T = np.zeros((N, N), dtype=complex)
            T[j, k] = 0.5; T[k, j] = 0.5
            gens.append(T)
            T = np.zeros((N, N), dtype=complex)
            T[j, k] = -0.5j; T[k, j] = 0.5j
            gens.append(T)
    for l in range(1, N):
        T = np.zeros((N, N), dtype=complex)
        norm = 1.0 / np.sqrt(2.0 * l * (l + 1))
        for j in range(l):
            T[j, j] = norm
        T[l, l] = -l * norm
        gens.append(T)
    return np.array(gens)


# ═══════════════════════════════════════════════════════════════════════
#  Resource estimation
# ═══════════════════════════════════════════════════════════════════════

def estimate_resources(N):
    """Estimate memory and time for a triangle lattice test at given N."""
    total_dim = N ** 9
    n_gen = N * N - 1

    # Hamiltonian: ~6 * n_gen coupling terms, each with nnz ~ total_dim * N
    # Each sparse matrix entry = 16 bytes (complex128) + 8 bytes (indices)
    nnz_per_term = total_dim * N  # rough upper bound
    n_terms = 6 * n_gen
    # Final H has fewer nnz due to overlap, estimate ~5x single term
    h_nnz = min(nnz_per_term * 5, total_dim * total_dim)
    h_mem_bytes = h_nnz * 24  # 16 for value + 4+4 for indices

    # G² is denser, roughly 2-3x H
    g2_mem_bytes = h_mem_bytes * 3

    # Gauss generators: 3 * n_gen sparse matrices
    gauss_mem = 3 * n_gen * nnz_per_term * 24

    # eigsh workspace: ~10 * total_dim * 16 bytes
    eig_mem = 10 * total_dim * 16

    total_mem = h_mem_bytes + g2_mem_bytes + gauss_mem + eig_mem

    # Time estimate (rough, calibrated from N=3,4 runs)
    if N <= 3:
        est_time = 10
    elif N == 4:
        est_time = 120
    elif N == 5:
        est_time = 1800
    else:
        est_time = 7200 * (N / 6) ** 3

    return {
        'N': N,
        'total_dim': total_dim,
        'n_generators': n_gen,
        'est_memory_gb': total_mem / 1e9,
        'est_time_minutes': est_time / 60,
        'h_nnz_estimate': h_nnz,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Sparse tensor-product utilities (memory-optimized)
# ═══════════════════════════════════════════════════════════════════════

def sparse_kron_chain(ops_and_eyes):
    """
    Build a kronecker product of a list of (sparse) matrices efficiently.
    Processes left-to-right to keep intermediate results as small as possible.
    """
    result = ops_and_eyes[0]
    for mat in ops_and_eyes[1:]:
        result = sparse.kron(result, mat, format='csr')
    return result


def sparse_single_factor(op_dense, factor_idx, dims):
    """Embed local operator on one tensor factor (sparse)."""
    n = len(dims)
    mats = []
    for i in range(n):
        if i == factor_idx:
            mats.append(sparse.csr_matrix(op_dense))
        else:
            mats.append(sparse.eye(dims[i], format='csr'))
    return sparse_kron_chain(mats)


def sparse_two_factor(opA, fA, opB, fB, dims):
    """Embed product of two local operators (sparse)."""
    n = len(dims)
    mats = []
    for i in range(n):
        if i == fA:
            mats.append(sparse.csr_matrix(opA))
        elif i == fB:
            mats.append(sparse.csr_matrix(opB))
        else:
            mats.append(sparse.eye(dims[i], format='csr'))
    return sparse_kron_chain(mats)


# ═══════════════════════════════════════════════════════════════════════
#  Vertex-level representation theory (cheap, works for any N)
# ═══════════════════════════════════════════════════════════════════════

def vertex_singlet_count(N):
    """
    Count singlets in N⊗N⊗N for SU(N) via total Casimir on C^{N³}.
    """
    T_gens = sun_generators(N)
    n_gen = T_gens.shape[0]
    d = N ** 3

    C = np.zeros((d, d), dtype=complex)
    I_N = np.eye(N)
    for a in range(n_gen):
        T = T_gens[a]
        T_total = (np.kron(np.kron(T, I_N), I_N) +
                   np.kron(np.kron(I_N, T), I_N) +
                   np.kron(np.kron(I_N, I_N), T))
        C += T_total @ T_total

    evals = np.linalg.eigvalsh(C)
    n_singlets = int(np.sum(np.abs(evals) < 1e-8))

    # Extract representation content
    unique_levels = []
    sorted_e = np.sort(evals)
    i = 0
    while i < len(sorted_e):
        val = sorted_e[i]
        count = int(np.sum(np.abs(sorted_e - val) < 0.05))
        unique_levels.append((float(val), int(count)))
        i += count

    return n_singlets, unique_levels


# ═══════════════════════════════════════════════════════════════════════
#  Full triangle lattice test
# ═══════════════════════════════════════════════════════════════════════

def run_triangle(N, verbose=True):
    """Full triangle lattice test for SU(N)."""

    T_gens = sun_generators(N)
    n_gen = T_gens.shape[0]
    n_factors = 9
    dims = [N] * n_factors
    total_dim = N ** n_factors

    # Factor indices
    iA, iAB_L, iAB_R = 0, 1, 2
    iB, iBC_L, iBC_R = 3, 4, 5
    iC, iCA_L, iCA_R = 6, 7, 8

    couplings = [
        (iA, iAB_L), (iAB_R, iB),
        (iB, iBC_L), (iBC_R, iC),
        (iC, iCA_L), (iCA_R, iA),
    ]

    gauss_vertex_factors = {
        'A': [iA, iAB_L, iCA_R],
        'B': [iB, iAB_R, iBC_L],
        'C': [iC, iBC_R, iCA_L],
    }

    # ─── Build Hamiltonian ─────────────────────────────────────────
    if verbose:
        print(f"  [{_ts()}] Building Hamiltonian ({len(couplings)*n_gen} terms)...")
    t0 = time.time()

    H = sparse.csr_matrix((total_dim, total_dim), dtype=complex)
    for ci, (fA, fB) in enumerate(couplings):
        for a in range(n_gen):
            H = H + sparse_two_factor(T_gens[a], fA, T_gens[a], fB, dims)
        if verbose:
            print(f"         coupling {ci+1}/{len(couplings)} done  "
                  f"({time.time()-t0:.1f}s, nnz={H.nnz:,})")

    H = 0.5 * (H + H.conj().T)
    if verbose:
        mem_mb = (H.data.nbytes + H.indices.nbytes + H.indptr.nbytes) / 1e6
        print(f"  [{_ts()}] H built: nnz={H.nnz:,}, mem={mem_mb:.0f} MB")

    # ─── Build Gauss generators ────────────────────────────────────
    if verbose:
        print(f"  [{_ts()}] Building Gauss generators ({3*n_gen} operators)...")
    t0 = time.time()

    G = {}
    for v, factors in gauss_vertex_factors.items():
        for a in range(n_gen):
            Ga = sparse.csr_matrix((total_dim, total_dim), dtype=complex)
            for f in factors:
                Ga = Ga + sparse_single_factor(T_gens[a], f, dims)
            G[(v, a)] = Ga
    if verbose:
        print(f"  [{_ts()}] Done in {time.time()-t0:.1f}s")

    # ─── Test 1: Gauge invariance ──────────────────────────────────
    if verbose:
        print(f"  [{_ts()}] Testing gauge invariance...")
    max_comm = 0.0
    for v in ['A', 'B', 'C']:
        for a in range(n_gen):
            comm = H @ G[(v, a)] - G[(v, a)] @ H
            max_comm = max(max_comm, spla.norm(comm))
    gauge_ok = max_comm < 1e-10
    if verbose:
        print(f"    max ||[H,G]|| = {max_comm:.3e}  →  {'✓' if gauge_ok else '✗'}")

    # ─── Test 2: Gauss subspace ────────────────────────────────────
    if verbose:
        print(f"  [{_ts()}] Building G²...")
    t0 = time.time()
    G2 = sparse.csr_matrix((total_dim, total_dim), dtype=complex)
    for v in ['A', 'B', 'C']:
        for a in range(n_gen):
            Ga = G[(v, a)]
            G2 = G2 + Ga @ Ga
    G2 = 0.5 * (G2 + G2.conj().T)
    if verbose:
        mem_mb = (G2.data.nbytes + G2.indices.nbytes + G2.indptr.nbytes) / 1e6
        print(f"  [{_ts()}] G² built in {time.time()-t0:.1f}s: nnz={G2.nnz:,}, mem={mem_mb:.0f} MB")

    # Free Gauss generators to save memory before eigsh
    del G
    gc.collect()

    if verbose:
        print(f"  [{_ts()}] Finding lowest eigenvalues of G² (shift-invert)...")
    t0 = time.time()
    n_eigs = min(60, total_dim - 2)

    try:
        # Shift-invert mode: find eigenvalues near sigma=0
        # Much faster for finding near-zero eigenvalues in large matrices
        eigvals, eigvecs = spla.eigsh(G2, k=n_eigs, sigma=0.0,
                                       which='LM', tol=1e-10,
                                       maxiter=5000)
    except Exception as e:
        if verbose:
            print(f"    Shift-invert failed ({e}), falling back to SM mode...")
        eigvals, eigvecs = spla.eigsh(G2, k=n_eigs, which='SM',
                                       tol=1e-10, maxiter=5000)

    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if verbose:
        print(f"  [{_ts()}] eigsh done in {time.time()-t0:.1f}s")

    tol = 1e-6
    n_singlets = int(np.sum(eigvals < tol))

    # Show spectrum structure
    unique_levels = []
    i = 0
    while i < len(eigvals) and len(unique_levels) < 8:
        val = eigvals[i]
        count = int(np.sum(np.abs(eigvals - val) < 0.01))
        unique_levels.append((float(val), int(count)))
        i += count

    if verbose:
        print(f"\n    G² spectrum (lowest levels):")
        for val, deg in unique_levels:
            tag = "  ← SINGLET" if val < tol else ""
            print(f"      λ = {val:10.4f}  (degeneracy {deg}){tag}")
        print(f"\n    Gauss singlets: {n_singlets}  →  "
              f"{'✓ NONEMPTY' if n_singlets > 0 else '✗ EMPTY'}")

    # ─── Test 3: Ground state in Gauss sector ──────────────────────
    gs_energy = None
    gauss_gs_energy = None
    gs_overlap = 0.0
    gs_in_gauss = False
    gap = None

    if n_singlets > 0:
        singlet_vecs = eigvecs[:, eigvals < tol]
        HV = H @ singlet_vecs
        H_proj = singlet_vecs.conj().T @ HV
        H_proj = 0.5 * (H_proj + H_proj.conj().T)
        evals_gauss = np.linalg.eigvalsh(H_proj)
        gauss_gs_energy = float(evals_gauss[0])

        if verbose:
            print(f"  [{_ts()}] Finding full ground state...")
        n_full = min(10, total_dim - 2)
        evals_full, evecs_full = spla.eigsh(H, k=n_full, which='SA', tol=1e-10)
        evals_full_sorted = np.sort(evals_full)
        gs_energy = float(evals_full_sorted[0])
        gap = float(evals_full_sorted[1] - evals_full_sorted[0]) if len(evals_full_sorted) > 1 else None

        gs_vec = evecs_full[:, np.argmin(evals_full)]
        gs_overlap = float(np.sum(np.abs(singlet_vecs.conj().T @ gs_vec)**2))
        gs_in_gauss = gs_overlap > 0.99

        if verbose:
            print(f"    Gauss GS energy: {gauss_gs_energy:+.6f}")
            print(f"    Full GS energy:  {gs_energy:+.6f}")
            print(f"    Gap: {gap:.4f}")
            print(f"    GS overlap: {gs_overlap:.8f}  →  "
                  f"{'✓ GS IN GAUSS' if gs_in_gauss else '✗'}")

    return {
        'N': int(N),
        'gauge_group': f'SU({N})',
        'n_generators': int(n_gen),
        'link_dim': int(N**2),
        'total_dim': int(total_dim),
        'computed': True,
        'gauge_invariant': bool(gauge_ok),
        'max_gauge_commutator': float(max_comm),
        'n_singlets': int(n_singlets),
        'g2_spectrum': [(float(v), int(d)) for v, d in unique_levels],
        'g2_lowest_eigenvalues': [float(x) for x in eigvals[:20]],
        'gs_energy_full': gs_energy,
        'gs_energy_gauss': gauss_gs_energy,
        'energy_gap': gap,
        'gs_overlap_with_gauss': float(gs_overlap),
        'gs_in_gauss': bool(gs_in_gauss),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

_t0_global = time.time()
def _ts():
    return f"{time.time() - _t0_global:6.1f}s"


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    global _t0_global
    _t0_global = time.time()

    parser = argparse.ArgumentParser(description='HSF Triangle Gauge Group Selection')
    parser.add_argument('--nmin', type=int, default=2, help='Minimum N (default: 2)')
    parser.add_argument('--nmax', type=int, default=5, help='Maximum N for lattice test (default: 5)')
    parser.add_argument('--vertex-max', type=int, default=None,
                        help='Maximum N for vertex rep theory (default: nmax+3)')
    parser.add_argument('--vertex-only', action='store_true',
                        help='Only run vertex-level rep theory (fast, any N)')
    parser.add_argument('--skip-vertex', action='store_true',
                        help='Skip vertex test, run lattice only')
    parser.add_argument('--mem-limit-gb', type=float, default=28.0,
                        help='Memory limit in GB (default: 28, leaves headroom on 32 GB)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompts')
    args = parser.parse_args()

    vertex_max = args.vertex_max or (args.nmax + 3)

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  HSF Paper III: Gauge Group Selection — Desktop Edition             ║")
    print("║  Triangle lattice with composite links: which SU(N) is selected?   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    all_results = {
        'vertex': {},
        'triangle': {},
    }

    # ─── Part 1: Vertex rep theory ─────────────────────────────────
    if not args.skip_vertex:
        print(f"\n{'═' * 70}")
        print(f"  PART 1: Vertex Representation Theory  (N=2..{vertex_max})")
        print(f"  Singlets in N⊗N⊗N for SU(N)?")
        print(f"{'═' * 70}")

        for N in range(2, vertex_max + 1):
            t0 = time.time()
            n_sing, levels = vertex_singlet_count(N)
            dt = time.time() - t0
            marker = " ★" if n_sing > 0 else ""
            decomp = ", ".join([f"C={v:.2f}(×{c})" for v, c in levels])
            print(f"  SU({N:2d}): singlets={n_sing}{marker}  ({dt:.2f}s)  [{decomp}]")
            all_results['vertex'][str(N)] = {
                'n_singlets': n_sing,
                'decomposition': levels,
            }

        # Quick summary
        winners = [N for N in range(2, vertex_max+1) if all_results['vertex'][str(N)]['n_singlets'] > 0]
        print(f"\n  Singlet-bearing groups: {winners if winners else 'NONE except N=3'}")

    if args.vertex_only:
        _save_results(all_results)
        return

    # ─── Part 2: Resource check ────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  PART 2: Full Triangle Lattice  (N={args.nmin}..{args.nmax})")
    print(f"{'═' * 70}")

    print(f"\n  Resource estimates:")
    print(f"  {'N':>4}  {'dim(H)':>14}  {'Est. Memory':>12}  {'Est. Time':>12}  Feasible?")
    print(f"  {'─'*4}  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*10}")

    run_list = []
    for N in range(args.nmin, args.nmax + 1):
        est = estimate_resources(N)
        feasible = est['est_memory_gb'] < args.mem_limit_gb
        tag = "✓" if feasible else f"✗ (>{args.mem_limit_gb:.0f} GB)"
        print(f"  {N:>4}  {est['total_dim']:>14,}  {est['est_memory_gb']:>9.1f} GB"
              f"  {est['est_time_minutes']:>9.1f} min  {tag}")
        if feasible:
            run_list.append(N)

    if not run_list:
        print("\n  No feasible N values. Try --nmax with a smaller value or increase --mem-limit-gb.")
        _save_results(all_results)
        return

    skipped = [N for N in range(args.nmin, args.nmax + 1) if N not in run_list]
    if skipped:
        print(f"\n  ⚠ Skipping N={skipped} (exceeds memory limit)")

    if not args.yes:
        print(f"\n  Will run: N={run_list}")
        print(f"  Press Enter to continue (or Ctrl-C to abort)...", end='', flush=True)
        try:
            input()
        except KeyboardInterrupt:
            print("\n  Aborted.")
            return

    # ─── Part 3: Run tests ─────────────────────────────────────────
    for N in run_list:
        print(f"\n{'─' * 70}")
        print(f"  SU({N})  |  d_B={N**2}  |  dim(H)={N**9:,}  |  gens={N**2-1}")
        print(f"{'─' * 70}")

        t0 = time.time()
        result = run_triangle(N)
        result['wall_time_seconds'] = time.time() - t0
        all_results['triangle'][str(N)] = result

        print(f"\n  SU({N}) completed in {result['wall_time_seconds']:.1f}s")
        gc.collect()

    # ─── Final summary ─────────────────────────────────────────────
    elapsed = time.time() - _t0_global

    print(f"\n\n{'═' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'═' * 70}")

    if all_results['vertex']:
        print(f"\n  Vertex test (N⊗N⊗N singlets):")
        print(f"  {'N':>4}  {'Singlets':>9}")
        print(f"  {'─'*4}  {'─'*9}")
        for N_str, vr in sorted(all_results['vertex'].items(), key=lambda x: int(x[0])):
            marker = "  ★" if vr['n_singlets'] > 0 else ""
            print(f"  {N_str:>4}  {vr['n_singlets']:>9}{marker}")

    if all_results['triangle']:
        print(f"\n  Triangle lattice:")
        print(f"  {'N':>4}  {'dim(H)':>12}  {'Gauge':>6}  {'Singlets':>9}  {'GS∈Gauss':>9}  {'Time':>8}")
        print(f"  {'─'*4}  {'─'*12}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}")
        for N_str, tr in sorted(all_results['triangle'].items(), key=lambda x: int(x[0])):
            g = '✓' if tr['gauge_invariant'] else '✗'
            s = str(tr['n_singlets'])
            gs = '✓' if tr['gs_in_gauss'] else '✗'
            wt = f"{tr.get('wall_time_seconds', 0):.0f}s"
            marker = "  ★" if tr['n_singlets'] > 0 and tr['gs_in_gauss'] else ""
            print(f"  {N_str:>4}  {tr['total_dim']:>12,}  {g:>6}  {s:>9}  {gs:>9}  {wt:>8}{marker}")

    print(f"\n  Total runtime: {elapsed:.1f}s")
    _save_results(all_results)


def _save_results(results):
    os.makedirs('hsf_out', exist_ok=True)
    outpath = 'hsf_out/triangle_gauge_selection.json'

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, tuple):
            return [clean(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(outpath, 'w') as f:
        json.dump(clean(results), f, indent=2)
    print(f"  Results saved to {outpath}")


if __name__ == '__main__':
    main()