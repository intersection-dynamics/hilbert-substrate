#!/usr/bin/env python3
"""
HSF Gravity: Ripple Asymmetry on Full N=14 Mesoscape (GPU)
============================================================
Matrix-free GPU implementation using CuPy.

State: (3,)^14 tensor on GPU, D = 3^14 = 4,782,969
Hamiltonian applied via tensor contractions (no explicit matrix).
Ground state via Lanczos eigsh with LinearOperator.
Time evolution via Lanczos-based Krylov expm_multiply.

Memory estimate:
  1 state vector: 73 MB
  eigsh with k=10, ncv=40: ~3 GB workspace
  Total: ~5-6 GB GPU RAM

Requirements: cupy, scipy
Usage: python gpu_gravity_ripple_N14.py [--seed 0] [--n-seeds 4]
"""

import numpy as np
import argparse
import json
import time
import os

try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
    from cupyx.scipy.sparse.linalg import LinearOperator as CpLinearOperator
    HAS_CUPY = True
    xp = cp
    print("Using CuPy (GPU)")
except ImportError:
    HAS_CUPY = False
    xp = np
    print("CuPy not found, falling back to NumPy (CPU)")

from scipy.sparse.linalg import eigsh, expm_multiply, LinearOperator


# ============================================================
# SU(3) generators on GPU
# ============================================================
def gellmann(xp):
    i = 1j
    out = []
    out.append(xp.array([[0,1,0],[1,0,0],[0,0,0]], dtype=xp.complex128) / 2)
    out.append(xp.array([[0,-i,0],[i,0,0],[0,0,0]], dtype=xp.complex128) / 2)
    out.append(xp.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=xp.complex128) / 2)
    out.append(xp.array([[0,0,1],[0,0,0],[1,0,0]], dtype=xp.complex128) / 2)
    out.append(xp.array([[0,0,-i],[0,0,0],[i,0,0]], dtype=xp.complex128) / 2)
    out.append(xp.array([[0,0,0],[0,0,1],[0,1,0]], dtype=xp.complex128) / 2)
    out.append(xp.array([[0,0,0],[0,0,-i],[0,i,0]], dtype=xp.complex128) / 2)
    out.append((1.0/xp.sqrt(xp.float64(3.0))) *
               xp.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=xp.complex128) / 2)
    return out


# ============================================================
# N=14 mesoscape topology
# ============================================================
N_SITES = 14
D_TOTAL = 3 ** N_SITES  # 4,782,969
SHAPE = (3,) * N_SITES

# Full edge list from the N=14 fission cascade
EDGES = [
    (0,1), (0,2), (0,5), (0,6),
    (1,2), (1,3), (1,4), (1,6), (1,7), (1,8),
    (2,3), (2,4), (2,5), (2,7),
    (6,8), (6,9), (6,10),
    (8,9), (8,10), (8,11),
    (10,11), (10,12),
    (11,12), (11,13),
    (12,13),
]

# Node classification
DENSE_NODES = {0, 1, 2, 3, 4, 5, 7}    # old core, high connectivity
BRIDGE_NODES = {6, 8}                     # connecting hubs
SPARSE_NODES = {9, 10, 11, 12, 13}       # growth frontier


def edge_region(i, j):
    ns = {i, j}
    if ns <= DENSE_NODES:
        return "dense_internal"
    if ns <= SPARSE_NODES:
        return "sparse_internal"
    if ns & DENSE_NODES and ns & BRIDGE_NODES:
        return "dense_bridge"
    if ns & BRIDGE_NODES and ns & SPARSE_NODES:
        return "sparse_bridge"
    if ns <= BRIDGE_NODES:
        return "bridge_internal"
    if ns & DENSE_NODES and ns & SPARSE_NODES:
        return "cross"
    return "other"


# ============================================================
# Matrix-free Hamiltonian application
# ============================================================
def apply_one_body(psi_tensor, op, site, xp):
    """Apply 3x3 operator to one site of (3,)^N tensor."""
    y = xp.moveaxis(psi_tensor, site, 0)
    y = xp.tensordot(op, y, axes=([1], [0]))
    return xp.moveaxis(y, 0, site)


def apply_edge(psi_tensor, GM, i, j, xp):
    """Apply Σ_a T^a_i T^a_j to state tensor."""
    out = xp.zeros_like(psi_tensor)
    for a in range(8):
        tmp = apply_one_body(psi_tensor, GM[a], i, xp)
        tmp = apply_one_body(tmp, GM[a], j, xp)
        out = out + tmp
    return out


def apply_H(psi_flat, edges, strengths, GM, xp, shape):
    """Apply full Hamiltonian to flat state vector."""
    psi_t = psi_flat.reshape(shape)
    out = xp.zeros_like(psi_t)
    for (i, j), g in zip(edges, strengths):
        out = out + g * apply_edge(psi_t, GM, i, j, xp)
    return out.reshape(-1)


def edge_expectation(psi_flat, GM, i, j, xp, shape):
    """Compute ⟨ψ|H_ij|ψ⟩."""
    psi_t = psi_flat.reshape(shape)
    Hpsi = apply_edge(psi_t, GM, i, j, xp)
    return float(xp.real(xp.vdot(psi_flat, Hpsi.reshape(-1))))


# ============================================================
# Krylov time evolution
# ============================================================
def krylov_evolve(psi, dt, n_steps, edges, strengths, GM, xp, shape, n_krylov=20):
    """
    Lanczos-based time evolution: |ψ(t+dt)⟩ = exp(-iH dt)|ψ(t)⟩
    Uses the Krylov subspace method for matrix-free expm.
    """
    for step in range(n_steps):
        # Build Krylov subspace
        V = xp.zeros((len(psi), n_krylov), dtype=psi.dtype)
        T_mat = xp.zeros((n_krylov, n_krylov), dtype=psi.dtype)
        
        V[:, 0] = psi / xp.linalg.norm(psi)
        w = apply_H(V[:, 0], edges, strengths, GM, xp, shape)
        alpha = xp.real(xp.vdot(V[:, 0], w))
        T_mat[0, 0] = alpha
        w = w - alpha * V[:, 0]
        
        for j in range(1, n_krylov):
            beta = xp.linalg.norm(w)
            if float(beta) < 1e-14:
                # Krylov space exhausted
                n_krylov_actual = j
                break
            V[:, j] = w / beta
            T_mat[j-1, j] = beta
            T_mat[j, j-1] = beta
            w = apply_H(V[:, j], edges, strengths, GM, xp, shape)
            alpha = xp.real(xp.vdot(V[:, j], w))
            T_mat[j, j] = alpha
            w = w - alpha * V[:, j] - beta * V[:, j-1]
            # Re-orthogonalize
            for k in range(j+1):
                w = w - xp.vdot(V[:, k], w) * V[:, k]
        else:
            n_krylov_actual = n_krylov
        
        # Diagonalize T (small matrix, can do on CPU)
        T_small = T_mat[:n_krylov_actual, :n_krylov_actual]
        if HAS_CUPY:
            T_np = cp.asnumpy(T_small)
        else:
            T_np = np.array(T_small)
        T_np = np.real((T_np + T_np.conj().T) / 2)
        evals_k, evecs_k = np.linalg.eigh(T_np)
        
        # exp(-i T dt) applied to e_0
        phases = np.exp(-1j * evals_k * dt)
        y = evecs_k @ (phases * evecs_k[0, :].conj())
        
        if HAS_CUPY:
            y_gpu = cp.asarray(y)
        else:
            y_gpu = y
        
        psi = V[:, :n_krylov_actual] @ y_gpu
        psi = psi / xp.linalg.norm(psi)
    
    return psi


# ============================================================
# Main experiment
# ============================================================
def run_gravity_test(seed, GM, xp):
    print(f"\n{'─'*60}")
    print(f"  Seed: {seed}")
    print(f"{'─'*60}")
    
    rng = np.random.default_rng(seed)
    strengths = [float(rng.uniform(0.5, 1.5)) for _ in EDGES]
    
    # Build LinearOperator for eigsh
    def matvec(v):
        if HAS_CUPY:
            v_gpu = cp.asarray(v)
            result = apply_H(v_gpu, EDGES, strengths, GM, xp, SHAPE)
            return cp.asnumpy(result)
        else:
            return apply_H(v, EDGES, strengths, GM, xp, SHAPE)
    
    H_op = LinearOperator((D_TOTAL, D_TOTAL), matvec=matvec, dtype=np.complex128)
    
    # Find ground state
    print(f"  Finding ground state (D={D_TOTAL:,})...", flush=True)
    t0 = time.time()
    evals, evecs = eigsh(H_op, k=6, which='SA', tol=1e-8, maxiter=500)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    gs_np = evecs[:, 0]
    E0 = evals[0]
    gap = evals[1] - evals[0]
    print(f"  E0 = {E0:.6f}, gap = {gap:.6f} ({time.time()-t0:.1f}s)")
    
    if HAS_CUPY:
        gs = cp.asarray(gs_np)
    else:
        gs = gs_np
    
    # GS edge expectations
    print(f"  Computing GS edge expectations...", flush=True)
    t0 = time.time()
    gs_ev = {}
    for (i, j) in EDGES:
        gs_ev[(i,j)] = edge_expectation(gs, GM, i, j, xp, SHAPE)
    print(f"  Done ({time.time()-t0:.1f}s)")
    
    # Ripple propagation from boundary edges
    # Sources: edges connecting bridge to sparse region
    boundary_edges = [(i,j) for i,j in EDGES 
                      if edge_region(i,j) in ("sparse_bridge", "bridge_internal")]
    
    print(f"  Boundary source edges: {boundary_edges}")
    
    t_max = 12.0
    dt_step = 0.5
    n_steps_total = int(t_max / dt_step)
    sample_times = list(range(0, n_steps_total + 1, 2))  # sample every 2 steps
    
    source_results = {}
    
    for src_i, src_j in boundary_edges:
        src_region = edge_region(src_i, src_j)
        print(f"\n    Source: ({src_i},{src_j}) [{src_region}]", flush=True)
        
        # Create excitation
        psi_excited = apply_edge(gs.reshape(SHAPE), GM, src_i, src_j, xp).reshape(-1)
        norm_exc = float(xp.linalg.norm(psi_excited))
        psi_excited = psi_excited / norm_exc
        
        # Track edge expectations over time
        t0 = time.time()
        psi_t = psi_excited.copy()
        
        peak_delta = {e: 0.0 for e in EDGES}
        mean_delta = {e: 0.0 for e in EDGES}
        n_samples = 0
        
        for step in range(n_steps_total + 1):
            if step in sample_times:
                # Measure all edge expectations
                for (i, j) in EDGES:
                    ev = edge_expectation(psi_t, GM, i, j, xp, SHAPE)
                    d = abs(ev - gs_ev[(i,j)])
                    peak_delta[(i,j)] = max(peak_delta[(i,j)], d)
                    mean_delta[(i,j)] += d
                n_samples += 1
            
            if step < n_steps_total:
                psi_t = krylov_evolve(psi_t, dt_step, 1, EDGES, strengths,
                                       GM, xp, SHAPE, n_krylov=25)
        
        for e in EDGES:
            mean_delta[e] /= max(n_samples, 1)
        
        elapsed = time.time() - t0
        print(f"    Evolution done ({elapsed:.1f}s, {n_samples} samples)")
        
        # Classify and report
        dense_peaks = []
        sparse_peaks = []
        
        print(f"    {'Edge':>8} {'Region':>18} {'Peak |δ|':>10} {'Mean |δ|':>10}")
        for (i,j) in EDGES:
            if (i,j) == (src_i, src_j):
                continue
            region = edge_region(i, j)
            tag = ""
            if "dense" in region:
                dense_peaks.append(peak_delta[(i,j)])
                tag = " ← dense"
            elif "sparse" in region:
                sparse_peaks.append(peak_delta[(i,j)])
                tag = " ← sparse"
            print(f"    ({i:>2},{j:>2}) {region:>18} "
                  f"{peak_delta[(i,j)]:>10.6f} {mean_delta[(i,j)]:>10.6f}{tag}")
        
        ratio = None
        if dense_peaks and sparse_peaks:
            md = np.mean(dense_peaks)
            ms = np.mean(sparse_peaks)
            ratio = md / (ms + 1e-12)
            print(f"\n    Dense-ward: {md:.6f} (n={len(dense_peaks)})")
            print(f"    Sparse-ward: {ms:.6f} (n={len(sparse_peaks)})")
            print(f"    Ratio: {ratio:.3f}")
        
        source_results[f"({src_i},{src_j})"] = {
            "source": [src_i, src_j],
            "region": src_region,
            "peak_delta": {str(k): v for k, v in peak_delta.items()},
            "dense_peaks": dense_peaks,
            "sparse_peaks": sparse_peaks,
            "ratio": ratio,
        }
    
    return source_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    
    print("=" * 70)
    print("HSF GRAVITY: RIPPLE ASYMMETRY ON N=14 MESOSCAPE (GPU)")
    print("=" * 70)
    print(f"\nTopology: {N_SITES} sites, {len(EDGES)} edges, D = {D_TOTAL:,}")
    print(f"Edges: {EDGES}")
    print(f"\nNode regions:")
    print(f"  Dense:  {sorted(DENSE_NODES)}")
    print(f"  Bridge: {sorted(BRIDGE_NODES)}")
    print(f"  Sparse: {sorted(SPARSE_NODES)}")
    
    print(f"\nEdge classification:")
    for (i,j) in EDGES:
        print(f"  ({i:>2},{j:>2}): {edge_region(i,j)}")
    
    if HAS_CUPY:
        try:
            dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            print(f"\nGPU: {dev.get('name', b'unknown').decode() if isinstance(dev.get('name', b''), bytes) else dev.get('name', 'unknown')}")
        except Exception:
            print(f"\nGPU: device {cp.cuda.Device().id}")
        try:
            mem = cp.cuda.Device().mem_info
            print(f"GPU memory: {mem[0]/1e9:.1f} GB free / {mem[1]/1e9:.1f} GB total")
        except Exception:
            pass
    
    GM = gellmann(xp)
    
    all_results = {}
    all_ratios = []
    
    t_total = time.time()
    for s in range(args.n_seeds):
        seed = args.seed + s * 1000
        result = run_gravity_test(seed, GM, xp)
        all_results[f"seed_{seed}"] = result
        
        for key, r in result.items():
            if r["ratio"] is not None:
                all_ratios.append(r["ratio"])
    
    # Aggregate
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS ({len(all_ratios)} measurements)")
    print(f"{'='*70}")
    if all_ratios:
        print(f"\n  Dense/sparse ratios: {['%.3f' % r for r in all_ratios]}")
        print(f"  Mean: {np.mean(all_ratios):.3f} ± {np.std(all_ratios):.3f}")
        print(f"  Median: {np.median(all_ratios):.3f}")
        print(f"  Min: {min(all_ratios):.3f}, Max: {max(all_ratios):.3f}")
        above_1 = sum(1 for r in all_ratios if r > 1.0)
        print(f"  Fraction > 1.0 (gravitational): {above_1}/{len(all_ratios)}")
        
        mean_r = np.mean(all_ratios)
        if mean_r > 1.15 and above_1 > len(all_ratios) * 0.7:
            print(f"\n  *** GRAVITATIONAL SIGNAL ***")
            print(f"  Ripples propagate {mean_r:.2f}x stronger toward dense regions.")
        elif mean_r < 0.85:
            print(f"\n  *** ANTI-GRAVITATIONAL ***")
        else:
            print(f"\n  Inconclusive at current statistics.")
            print(f"  Try more seeds (--n-seeds 8) for better statistics.")
    
    elapsed = time.time() - t_total
    print(f"\n  Total time: {elapsed:.1f}s")
    
    # Save
    outfile = args.json_out or "hsf_out/gpu_gravity_ripple_N14.json"
    os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else "hsf_out",
                exist_ok=True)
    output = {
        "config": {
            "n_sites": N_SITES,
            "d_total": D_TOTAL,
            "edges": EDGES,
            "n_seeds": args.n_seeds,
            "base_seed": args.seed,
        },
        "all_ratios": all_ratios,
        "mean_ratio": float(np.mean(all_ratios)) if all_ratios else None,
        "std_ratio": float(np.std(all_ratios)) if all_ratios else None,
        "seed_results": {k: {sk: {kk: vv for kk, vv in sv.items() if kk != "peak_delta"}
                             for sk, sv in v.items()}
                        for k, v in all_results.items()},
        "elapsed_seconds": elapsed,
    }
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {outfile}")


if __name__ == "__main__":
    main()