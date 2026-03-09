#!/usr/bin/env python3
"""
triangle_gauge_selection_v2.py
==============================
HSF Paper III — Gauge Group Selection (Matrix-Free Swap Formulation)

MATHEMATICAL BASIS:
  For SU(N) generators normalized as Tr(T^a T^b) = δ^{ab}/2:

      Σ_a T^a ⊗ T^a = (P - I/N) / 2

  where P is the swap (permutation) operator. This identity converts
  ALL matrix operations into tensor axis transpositions:

      H·v  = (1/2) Σ_{6 couplings} swap(v, f1, f2) - (3/N)·v
      G²·v = c·v + Σ_{9 vertex-pairs} swap(v, pair)

  where c = 9(N²-3)/(2N). Each "swap" is just np.swapaxes on the
  vector reshaped as an order-9 tensor.

MEMORY FOOTPRINT:
  Only eigsh workspace vectors + the input/output vector.
  No sparse matrices stored at all.

  N=2:    512 dim,          8 KB/vec,   trivial
  N=3:    19,683 dim,     315 KB/vec,   trivial
  N=4:    262,144 dim,      4 MB/vec,   seconds
  N=5:  1,953,125 dim,     31 MB/vec,   ~2 min     (needs ~1 GB)
  N=6: 10,077,696 dim,    161 MB/vec,   ~15 min    (needs ~5 GB)
  N=7: 40,353,607 dim,    645 MB/vec,   ~1-2 hr    (needs ~15 GB)

USAGE:
  python triangle_gauge_selection_v2.py                    # N=2..5
  python triangle_gauge_selection_v2.py --nmax 6           # include N=6
  python triangle_gauge_selection_v2.py --nmax 7           # push to N=7 (needs ~15 GB)
  python triangle_gauge_selection_v2.py --vertex-only --nmax 15

DEPENDENCIES: numpy, scipy
"""

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
import time
import json
import os
import argparse
import gc


# ═══════════════════════════════════════════════════════════════════════
#  System geometry: triangle lattice factor layout
# ═══════════════════════════════════════════════════════════════════════
#
#  9 tensor factors (each of dimension N):
#    A(0)  AB_L(1)  AB_R(2)  B(3)  BC_L(4)  BC_R(5)  C(6)  CA_L(7)  CA_R(8)
#
#  Hamiltonian couplings (site ↔ adjacent link factor):
COUPLING_PAIRS = [(0, 1), (2, 3), (3, 4), (5, 6), (6, 7), (8, 0)]
#
#  Gauss generator vertex structure (site + 2 link factors):
VERTEX_FACTORS = {
    'A': [0, 1, 8],  # site A + left(AB) + right(CA)
    'B': [3, 2, 4],  # site B + right(AB) + left(BC)
    'C': [6, 5, 7],  # site C + right(BC) + left(CA)
}
#
#  Vertex pairs for G² (all pairs within each vertex):
VERTEX_PAIRS = []
for factors in VERTEX_FACTORS.values():
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            VERTEX_PAIRS.append((factors[i], factors[j]))
# = [(0,1),(0,8),(1,8), (3,2),(3,4),(2,4), (6,5),(6,7),(5,7)]
assert len(VERTEX_PAIRS) == 9


# ═══════════════════════════════════════════════════════════════════════
#  Swap-based matrix-free operators
# ═══════════════════════════════════════════════════════════════════════

def make_H_matvec(N):
    """
    H·v = (1/2) Σ_{6 couplings} swap(v, f1, f2) - (3/N)·v
    
    Derivation: each coupling is Σ_a T^a_{f1} T^a_{f2} = (P_{f1,f2} - I/N)/2.
    Summing 6 couplings: H = (1/2)Σ P - 6/(2N) I = (1/2)Σ P - (3/N) I.
    """
    shape = (N,) * 9
    scalar = -3.0 / N
    
    def matvec(v):
        vt = v.reshape(shape)
        result = scalar * v.copy()
        for f1, f2 in COUPLING_PAIRS:
            result += 0.5 * np.ascontiguousarray(np.swapaxes(vt, f1, f2)).ravel()
        return result
    
    return matvec


def make_G2_matvec(N):
    """
    G²·v = c·v + Σ_{9 vertex-pairs} swap(v, pair)
    
    where c = 9(N²-3)/(2N).
    
    Derivation: at each vertex with factors {f1,f2,f3},
      Σ_a (G^a_v)² = Σ_a (T_{f1} + T_{f2} + T_{f3})²
        = 3·C₂·I + 2·Σ_{pairs} (P_{pair} - I/N)/2
        = 3(N²-1)/(2N)·I + Σ_{3 pairs} P - 3/N·I
        = [3(N²-3)/(2N)]·I + Σ_{3 pairs} P
    
    Summing 3 vertices:
      G² = [9(N²-3)/(2N)]·I + Σ_{9 pairs} P
    """
    shape = (N,) * 9
    scalar = 9.0 * (N * N - 3) / (2.0 * N)
    
    def matvec(v):
        vt = v.reshape(shape)
        result = scalar * v.copy()
        for f1, f2 in VERTEX_PAIRS:
            result += np.ascontiguousarray(np.swapaxes(vt, f1, f2)).ravel()
        return result
    
    return matvec


# ═══════════════════════════════════════════════════════════════════════
#  Tensor-contraction gauge invariance test
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


def apply_generator(v_tensor, T_a, factor_idx):
    """Apply T^a on one tensor factor: (T^a)_{factor} · v."""
    v_moved = np.moveaxis(v_tensor, factor_idx, -1)
    result = v_moved @ T_a.T  # correct for general complex T
    return np.moveaxis(result, -1, factor_idx)


def test_gauge_invariance_stochastic(N, n_samples=5):
    """
    Test [H, G^a_v] = 0 using random vectors.
    
    For each random v, compute:
      ||H(G^a·v) - G^a(H·v)|| / ||v||
    
    If gauge invariance holds, this should be ~machine epsilon.
    """
    shape = (N,) * 9
    total_dim = N ** 9
    T_gens = sun_generators(N)
    n_gen = T_gens.shape[0]
    H_mv = make_H_matvec(N)
    
    max_violation = 0.0
    
    for sample in range(n_samples):
        # Random complex unit vector
        v = np.random.randn(total_dim) + 1j * np.random.randn(total_dim)
        v /= np.linalg.norm(v)
        
        Hv = H_mv(v)
        
        for vertex, factors in VERTEX_FACTORS.items():
            for a in range(n_gen):
                # G^a · v
                vt = v.reshape(shape)
                Gav = np.zeros(total_dim, dtype=complex)
                for f in factors:
                    Gav += apply_generator(vt, T_gens[a], f).ravel()
                
                # G^a · (H·v)
                Hvt = Hv.reshape(shape)
                GaHv = np.zeros(total_dim, dtype=complex)
                for f in factors:
                    GaHv += apply_generator(Hvt, T_gens[a], f).ravel()
                
                # H · (G^a · v)
                HGav = H_mv(Gav)
                
                violation = np.linalg.norm(HGav - GaHv)
                max_violation = max(max_violation, violation)
    
    return max_violation


# ═══════════════════════════════════════════════════════════════════════
#  Vertex representation theory (cheap, any N)
# ═══════════════════════════════════════════════════════════════════════

def vertex_singlet_count(N):
    """Count singlets in N⊗N⊗N for SU(N) via total Casimir."""
    T_gens = sun_generators(N)
    n_gen = T_gens.shape[0]
    d = N ** 3
    I_N = np.eye(N)
    
    C = np.zeros((d, d), dtype=complex)
    for a in range(n_gen):
        T = T_gens[a]
        T_total = (np.kron(np.kron(T, I_N), I_N) +
                   np.kron(np.kron(I_N, T), I_N) +
                   np.kron(np.kron(I_N, I_N), T))
        C += T_total @ T_total
    
    evals = np.linalg.eigvalsh(C)
    n_singlets = int(np.sum(np.abs(evals) < 1e-8))
    
    # Representation content
    levels = []
    sorted_e = np.sort(evals)
    i = 0
    while i < len(sorted_e):
        val = sorted_e[i]
        count = int(np.sum(np.abs(sorted_e - val) < 0.05))
        levels.append((float(val), int(count)))
        i += count
    
    return n_singlets, levels


# ═══════════════════════════════════════════════════════════════════════
#  Verification against explicit sparse computation
# ═══════════════════════════════════════════════════════════════════════

def verify_swap_formula(N):
    """
    For small N, verify the swap formula matches the explicit sparse
    computation. Returns max discrepancy.
    """
    from scipy import sparse
    
    total_dim = N ** 9
    if total_dim > 50000:
        return None  # too large for explicit
    
    shape = (N,) * 9
    T_gens = sun_generators(N)
    n_gen = T_gens.shape[0]
    dims = [N] * 9
    
    # Build explicit H
    def sp_embed2(opA, fA, opB, fB):
        mats = []
        for i in range(9):
            if i == fA: mats.append(sparse.csr_matrix(opA))
            elif i == fB: mats.append(sparse.csr_matrix(opB))
            else: mats.append(sparse.eye(N, format='csr'))
        result = mats[0]
        for m in mats[1:]:
            result = sparse.kron(result, m, format='csr')
        return result
    
    def sp_embed1(op, f):
        mats = []
        for i in range(9):
            if i == f: mats.append(sparse.csr_matrix(op))
            else: mats.append(sparse.eye(N, format='csr'))
        result = mats[0]
        for m in mats[1:]:
            result = sparse.kron(result, m, format='csr')
        return result
    
    H_explicit = sparse.csr_matrix((total_dim, total_dim), dtype=complex)
    for f1, f2 in COUPLING_PAIRS:
        for a in range(n_gen):
            H_explicit += sp_embed2(T_gens[a], f1, T_gens[a], f2)
    H_explicit = 0.5 * (H_explicit + H_explicit.conj().T)
    
    G2_explicit = sparse.csr_matrix((total_dim, total_dim), dtype=complex)
    for v, factors in VERTEX_FACTORS.items():
        for a in range(n_gen):
            Ga = sparse.csr_matrix((total_dim, total_dim), dtype=complex)
            for f in factors:
                Ga += sp_embed1(T_gens[a], f)
            G2_explicit += Ga @ Ga
    G2_explicit = 0.5 * (G2_explicit + G2_explicit.conj().T)
    
    # Compare on random vectors
    H_swap = make_H_matvec(N)
    G2_swap = make_G2_matvec(N)
    
    max_H_err = 0.0
    max_G2_err = 0.0
    for _ in range(5):
        v = np.random.randn(total_dim) + 1j * np.random.randn(total_dim)
        v /= np.linalg.norm(v)
        
        Hv_explicit = H_explicit @ v
        Hv_swap = H_swap(v)
        max_H_err = max(max_H_err, np.linalg.norm(Hv_explicit - Hv_swap))
        
        G2v_explicit = G2_explicit @ v
        G2v_swap = G2_swap(v)
        max_G2_err = max(max_G2_err, np.linalg.norm(G2v_explicit - G2v_swap))
    
    return max_H_err, max_G2_err


# ═══════════════════════════════════════════════════════════════════════
#  Full triangle test
# ═══════════════════════════════════════════════════════════════════════

def run_triangle(N, verify=False, n_eigs=20):
    """Full triangle test for SU(N), matrix-free."""
    
    total_dim = N ** 9
    n_gen = N * N - 1
    
    print(f"\n{'═' * 70}")
    print(f"  SU({N})  |  d_B={N**2}  |  dim(H)={total_dim:,}  |  gens={n_gen}")
    vec_mb = total_dim * 16 / 1e6
    print(f"  Vector size: {vec_mb:.1f} MB  |  eigsh workspace: ~{vec_mb * (n_eigs + 5):.0f} MB")
    print(f"{'═' * 70}")
    
    # ─── Optional: verify swap formula ─────────────────────────────
    if verify and total_dim <= 50000:
        print(f"  Verifying swap formula against explicit sparse...")
        h_err, g2_err = verify_swap_formula(N)
        print(f"    H  max error: {h_err:.2e}")
        print(f"    G² max error: {g2_err:.2e}")
        assert h_err < 1e-10 and g2_err < 1e-10, "Swap formula verification FAILED!"
        print(f"    ✓ Verified")
    
    # ─── Gauge invariance (stochastic) ─────────────────────────────
    print(f"  Testing gauge invariance (stochastic, 3 samples)...")
    t0 = time.time()
    max_viol = test_gauge_invariance_stochastic(N, n_samples=3)
    gauge_ok = max_viol < 1e-10
    print(f"    max ||[H,G]·v|| = {max_viol:.3e}  →  {'✓' if gauge_ok else '✗'}"
          f"  ({time.time()-t0:.1f}s)")
    
    # ─── G² eigenvalues ────────────────────────────────────────────
    G2_mv = make_G2_matvec(N)
    G2_op = LinearOperator((total_dim, total_dim), matvec=G2_mv, dtype=complex)
    
    actual_n_eigs = min(n_eigs, total_dim - 2)
    
    print(f"  Finding {actual_n_eigs} lowest G² eigenvalues (matrix-free eigsh)...")
    t0 = time.time()
    
    try:
        eigvals, eigvecs = eigsh(G2_op, k=actual_n_eigs, which='SM',
                                  tol=1e-10, maxiter=3000)
    except Exception as e:
        print(f"    eigsh failed: {e}")
        print(f"    Trying with looser tolerance...")
        eigvals, eigvecs = eigsh(G2_op, k=actual_n_eigs, which='SM',
                                  tol=1e-6, maxiter=5000)
    
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    dt_eig = time.time() - t0
    print(f"    Done in {dt_eig:.1f}s")
    
    tol = 1e-4
    n_singlets = int(np.sum(eigvals < tol))
    
    # Spectrum structure
    levels = []
    i = 0
    while i < len(eigvals) and len(levels) < 8:
        val = eigvals[i]
        count = int(np.sum(np.abs(eigvals - val) < 0.05))
        levels.append((float(val), int(count)))
        i += count
    
    print(f"\n    G² spectrum:")
    for val, deg in levels:
        tag = "  ← SINGLET" if val < tol else ""
        print(f"      λ = {val:12.6f}  (×{deg}){tag}")
    
    print(f"\n    Singlets: {n_singlets}  →  {'✓ NONEMPTY' if n_singlets > 0 else '✗ EMPTY'}")
    
    # ─── Ground state ──────────────────────────────────────────────
    gs_energy = None
    gauss_gs_energy = None
    gs_overlap = 0.0
    gs_in_gauss = False
    gap = None
    
    if n_singlets > 0:
        singlet_vecs = eigvecs[:, eigvals < tol]
        
        # Project H into Gauss sector
        H_mv = make_H_matvec(N)
        H_singlet = np.zeros((n_singlets, n_singlets), dtype=complex)
        for i in range(n_singlets):
            Hvi = H_mv(singlet_vecs[:, i])
            for j in range(n_singlets):
                H_singlet[i, j] = singlet_vecs[:, j].conj() @ Hvi
        H_singlet = 0.5 * (H_singlet + H_singlet.conj().T)
        evals_gauss = np.linalg.eigvalsh(H_singlet)
        gauss_gs_energy = float(evals_gauss[0])
        
        # Full ground state
        print(f"  Finding full ground state...")
        H_op = LinearOperator((total_dim, total_dim), matvec=H_mv, dtype=complex)
        n_full = min(6, total_dim - 2)
        evals_full, evecs_full = eigsh(H_op, k=n_full, which='SA',
                                        tol=1e-10, maxiter=3000)
        evals_full_s = np.sort(evals_full)
        gs_energy = float(evals_full_s[0])
        gap = float(evals_full_s[1] - evals_full_s[0]) if len(evals_full_s) > 1 else None
        
        gs_vec = evecs_full[:, np.argmin(evals_full)]
        gs_overlap = float(np.sum(np.abs(singlet_vecs.conj().T @ gs_vec)**2))
        gs_in_gauss = gs_overlap > 0.99
        
        print(f"    Gauss GS:   E = {gauss_gs_energy:+.6f}")
        print(f"    Full GS:    E = {gs_energy:+.6f}  (gap = {gap:.4f})")
        print(f"    Overlap:    {gs_overlap:.10f}  →  "
              f"{'✓ GS IN GAUSS' if gs_in_gauss else '✗'}")
    
    return {
        'N': int(N),
        'gauge_group': f'SU({N})',
        'n_generators': int(n_gen),
        'link_dim': int(N**2),
        'total_dim': int(total_dim),
        'gauge_invariant': bool(gauge_ok),
        'max_gauge_violation': float(max_viol),
        'n_singlets': int(n_singlets),
        'g2_spectrum': [(float(v), int(d)) for v, d in levels],
        'g2_lowest_eigenvalues': [float(x) for x in eigvals[:20]],
        'gs_energy_full': gs_energy,
        'gs_energy_gauss': gauss_gs_energy,
        'energy_gap': gap,
        'gs_overlap': float(gs_overlap),
        'gs_in_gauss': bool(gs_in_gauss),
        'eigsh_time_seconds': float(dt_eig),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Resource estimation
# ═══════════════════════════════════════════════════════════════════════

def estimate_memory_gb(N, n_eigs=20):
    """Estimate total memory for matrix-free approach."""
    total_dim = N ** 9
    vec_bytes = total_dim * 16  # complex128
    # eigsh needs ~(n_eigs + 5) vectors
    n_vecs = n_eigs + 10  # some extra for workspace
    # Plus ~3 temporary vectors during matvec
    total = vec_bytes * (n_vecs + 3)
    return total / 1e9


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='HSF Triangle Gauge Selection (Matrix-Free)')
    parser.add_argument('--nmin', type=int, default=2)
    parser.add_argument('--nmax', type=int, default=5)
    parser.add_argument('--vertex-max', type=int, default=None)
    parser.add_argument('--vertex-only', action='store_true')
    parser.add_argument('--skip-vertex', action='store_true')
    parser.add_argument('--verify', action='store_true',
                        help='Verify swap formula against explicit sparse for small N')
    parser.add_argument('--n-eigs', type=int, default=20,
                        help='Number of eigenvalues to find (reduce for large N)')
    parser.add_argument('--mem-limit-gb', type=float, default=28.0)
    parser.add_argument('--yes', '-y', action='store_true')
    args = parser.parse_args()
    
    vertex_max = args.vertex_max or max(args.nmax + 3, 8)
    t_start = time.time()
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  HSF Paper III: Gauge Group Selection — Matrix-Free Edition         ║")
    print("║  Swap-operator identity eliminates all sparse matrices              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    results = {'vertex': {}, 'triangle': {}}
    
    # ─── Vertex rep theory ─────────────────────────────────────────
    if not args.skip_vertex:
        print(f"\n{'═' * 70}")
        print(f"  VERTEX REPRESENTATION THEORY  (N=2..{vertex_max})")
        print(f"  Singlets in N⊗N⊗N for SU(N)?")
        print(f"{'═' * 70}\n")
        
        print(f"  {'N':>4}  {'Singlets':>9}  Decomposition")
        print(f"  {'─'*4}  {'─'*9}  {'─'*45}")
        
        for N in range(2, vertex_max + 1):
            n_sing, levels = vertex_singlet_count(N)
            marker = " ★" if n_sing > 0 else ""
            decomp = ", ".join([f"C={v:.2f}(×{c})" for v, c in levels])
            print(f"  {N:>4}  {n_sing:>9}{marker}  {decomp}")
            results['vertex'][str(N)] = {'n_singlets': n_sing, 'decomposition': levels}
    
    if args.vertex_only:
        _save(results, time.time() - t_start)
        return
    
    # ─── Resource check ────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  TRIANGLE LATTICE  (N={args.nmin}..{args.nmax})")
    print(f"  Using matrix-free swap operators — no sparse matrices stored")
    print(f"{'═' * 70}\n")
    
    print(f"  {'N':>4}  {'dim(H)':>14}  {'Vec size':>10}  {'Est. RAM':>10}  OK?")
    print(f"  {'─'*4}  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*5}")
    
    run_list = []
    for N in range(args.nmin, args.nmax + 1):
        td = N ** 9
        vec_mb = td * 16 / 1e6
        mem_gb = estimate_memory_gb(N, args.n_eigs)
        ok = mem_gb < args.mem_limit_gb
        tag = "✓" if ok else f"✗ >{args.mem_limit_gb:.0f}GB"
        print(f"  {N:>4}  {td:>14,}  {vec_mb:>8.1f}MB  {mem_gb:>8.1f}GB  {tag}")
        if ok:
            run_list.append(N)
    
    if not run_list:
        print("\n  No feasible N. Try --n-eigs 10 or --mem-limit-gb higher.")
        _save(results, time.time() - t_start)
        return
    
    skipped = set(range(args.nmin, args.nmax + 1)) - set(run_list)
    if skipped:
        print(f"\n  ⚠ Skipping N={sorted(skipped)} (over memory limit)")
        print(f"  Tip: --n-eigs 10 reduces workspace significantly")
    
    if not args.yes:
        print(f"\n  Will run: {run_list}. Press Enter to continue...", end='', flush=True)
        try:
            input()
        except KeyboardInterrupt:
            print("\n  Aborted.")
            return
    
    # ─── Run ───────────────────────────────────────────────────────
    for N in run_list:
        t0 = time.time()
        r = run_triangle(N, verify=args.verify, n_eigs=args.n_eigs)
        r['wall_time_seconds'] = time.time() - t0
        results['triangle'][str(N)] = r
        gc.collect()
    
    # ─── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t_start
    
    print(f"\n\n{'═' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'═' * 70}")
    
    if results['vertex']:
        winners = [N for N in range(2, vertex_max+1)
                   if results['vertex'].get(str(N), {}).get('n_singlets', 0) > 0]
        print(f"\n  Vertex: singlet only for N={winners}")
    
    if results['triangle']:
        print(f"\n  {'N':>4}  {'dim(H)':>12}  {'Gauge':>6}  {'Sing.':>6}  {'GS∈G':>6}  {'Time':>8}")
        print(f"  {'─'*4}  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")
        for N_str in sorted(results['triangle'], key=int):
            r = results['triangle'][N_str]
            g = '✓' if r['gauge_invariant'] else '✗'
            s = str(r['n_singlets'])
            gs = '✓' if r['gs_in_gauss'] else '—'
            wt = f"{r['wall_time_seconds']:.0f}s"
            star = "  ★" if r['n_singlets'] > 0 and r['gs_in_gauss'] else ""
            print(f"  {N_str:>4}  {r['total_dim']:>12,}  {g:>6}  {s:>6}  {gs:>6}  {wt:>8}{star}")
    
    print(f"\n  Total: {elapsed:.1f}s")
    _save(results, elapsed)


def _save(results, elapsed):
    os.makedirs('hsf_out', exist_ok=True)
    path = 'hsf_out/triangle_gauge_selection_v2.json'
    
    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    
    results['runtime_seconds'] = elapsed
    with open(path, 'w') as f:
        json.dump(clean(results), f, indent=2)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    main()