#!/usr/bin/env python3
"""
HIP: Capacity vs Typicality
============================

Computes BOTH diagnostics on the same graph:

1. CAPACITY: C_{i->j}(t) = sup_{ψ,O} D(ρ_j, ρ'_j)
   - What's the best-case information transfer?
   - Computed via variational optimization (JAX)

2. TYPICALITY: T_{i->j}(t) = 𝔼_ψ [max_O D(ρ_j, ρ'_j)]
   - What's the average-case information transfer?
   - Approximated via sampling

3. GAP: G_{i->j}(t) = C - T
   - How much fine-tuning is needed?
   - Small gap = robust pathway
   - Large gap = fragile channel

Outputs comparison statistics and visualizations.
"""

# Suppress warnings before importing JAX
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=true'

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

# JAX for capacity computation
import jax
jax.config.update("jax_enable_x64", True)
import logging
logging.getLogger('jax').setLevel(logging.ERROR)

import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.linalg import expm as jax_expm
import optax


# =============================================================================
# COMMON: Pauli matrices and graph utilities
# =============================================================================

I2_np = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X_np = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y_np = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z_np = np.array([[1, 0], [0, -1]], dtype=np.complex128)

I2_jax = jnp.array(I2_np)
X_jax = jnp.array(X_np)
Y_jax = jnp.array(Y_np)
Z_jax = jnp.array(Z_np)
PAULIS_JAX = [I2_jax, X_jax, Y_jax, Z_jax]


def make_graph(graph_type: str, n: int, seed: int = 0):
    if graph_type == "ring":
        G = nx.cycle_graph(n)
    elif graph_type == "line":
        G = nx.path_graph(n)
    elif graph_type == "grid":
        side = int(math.ceil(math.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
        nodes = list(G.nodes())[:n]
        G = G.subgraph(nodes).copy()
        G = nx.convert_node_labels_to_integers(G)
    elif graph_type == "random":
        G = nx.erdos_renyi_graph(n, 0.3, seed=seed)
        if not nx.is_connected(G):
            for i in range(n - 1):
                G.add_edge(i, i + 1)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    return G


# =============================================================================
# CAPACITY: Variational optimization (JAX)
# =============================================================================

def kron_list_jax(mats):
    out = mats[0]
    for m in mats[1:]:
        out = jnp.kron(out, m)
    return out


def embed_single_site_jax(op2, n_qubits, site):
    mats = [I2_jax] * n_qubits
    mats[site] = op2
    return kron_list_jax(mats)


def embed_two_site_jax(op4, n_qubits, site_a, site_b):
    out = jnp.zeros((2**n_qubits, 2**n_qubits), dtype=jnp.complex128)
    for m in range(4):
        for n in range(4):
            basis = jnp.kron(PAULIS_JAX[m], PAULIS_JAX[n])
            c = 0.25 * jnp.trace(basis.conj().T @ op4)
            if jnp.abs(c) > 1e-14:
                Em = embed_single_site_jax(PAULIS_JAX[m], n_qubits, site_a)
                En = embed_single_site_jax(PAULIS_JAX[n], n_qubits, site_b)
                out = out + c * (Em @ En)
    return 0.5 * (out + out.conj().T)


def build_hamiltonian_jax(G, n_qubits, J=1.0, h=0.2):
    dim = 2 ** n_qubits
    H = jnp.zeros((dim, dim), dtype=jnp.complex128)
    for i in range(n_qubits):
        H = H + h * embed_single_site_jax(Z_jax, n_qubits, i)
    HH = J * (jnp.kron(X_jax, X_jax) + jnp.kron(Y_jax, Y_jax) + jnp.kron(Z_jax, Z_jax))
    for (i, j) in G.edges():
        H = H + embed_two_site_jax(HH, n_qubits, i, j)
    return 0.5 * (H + H.conj().T)


def reduced_rho_jax(psi, n_qubits, target):
    tensor = psi.reshape((2,) * n_qubits)
    perm = [target] + [k for k in range(n_qubits) if k != target]
    tensor = jnp.transpose(tensor, perm)
    A = tensor.reshape((2, 2**(n_qubits - 1)))
    return A @ A.conj().T


def trace_distance_jax(rho, sigma):
    s = jnp.linalg.svd(rho - sigma, compute_uv=False)
    return 0.5 * jnp.sum(s)


def capacity_objective(state_real, state_imag, op_params, U_t, n_qubits, source, target):
    psi = state_real + 1j * state_imag
    psi = psi / (jnp.linalg.norm(psi) + 1e-12)
    
    H_op = op_params[0] * X_jax + op_params[1] * Y_jax + op_params[2] * Z_jax
    O_i = jax_expm(1j * H_op)
    O_i_full = embed_single_site_jax(O_i, n_qubits, source)
    
    phi = U_t @ psi
    phi_pert = U_t @ (O_i_full @ psi)
    
    rho_j = reduced_rho_jax(phi, n_qubits, target)
    rho_j_pert = reduced_rho_jax(phi_pert, n_qubits, target)
    
    return -trace_distance_jax(rho_j_pert, rho_j)


def compute_capacity(U_t, n_qubits, source, target, n_restarts=8, max_iters=200, lr=0.05, rng_key=None):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    dim = 2 ** n_qubits
    
    @jit
    def loss_fn(params):
        return capacity_objective(params['sr'], params['si'], params['op'], 
                                  U_t, n_qubits, source, target)
    
    grad_fn = jit(grad(loss_fn))
    best = 0.0
    
    for _ in range(n_restarts):
        rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
        params = {
            'sr': jax.random.normal(k1, (dim,)),
            'si': jax.random.normal(k2, (dim,)),
            'op': jax.random.uniform(k3, (3,), minval=-np.pi, maxval=np.pi)
        }
        
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        
        for _ in range(max_iters):
            grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        
        val = -float(loss_fn(params))
        if val > best:
            best = val
    
    return best


# =============================================================================
# TYPICALITY: Sampling (NumPy)
# =============================================================================

def random_product_state_np(n_qubits, rng):
    singles = []
    for _ in range(n_qubits):
        theta = np.arccos(1 - 2 * rng.random())
        phi = 2 * np.pi * rng.random()
        singles.append(np.array([np.cos(theta/2), np.sin(theta/2) * np.exp(1j*phi)], dtype=np.complex128))
    psi = singles[0]
    for s in singles[1:]:
        psi = np.kron(psi, s)
    return psi


def reduced_rho_np(psi, n_qubits, target):
    tensor = psi.reshape((2,) * n_qubits)
    perm = [target] + [k for k in range(n_qubits) if k != target]
    tensor = np.transpose(tensor, perm)
    A = tensor.reshape((2, 2**(n_qubits - 1)))
    return A @ A.conj().T


def trace_distance_np(rho, sigma):
    s = np.linalg.svd(rho - sigma, compute_uv=False)
    return 0.5 * np.sum(np.abs(s))


def apply_local_op_np(psi, Op, n_qubits, site):
    tensor = psi.reshape((2,) * n_qubits)
    perm = [site] + [k for k in range(n_qubits) if k != site]
    inv_perm = [0] * n_qubits
    for i, p in enumerate(perm):
        inv_perm[p] = i
    tensor = np.transpose(tensor, perm)
    shape = tensor.shape
    tensor = (Op @ tensor.reshape(2, -1)).reshape(shape)
    tensor = np.transpose(tensor, inv_perm)
    return tensor.reshape(-1)


def compute_typicality(w, V, t, n_qubits, source, target, n_states=50, n_ops=10, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    
    ops = [X_np, Y_np, Z_np]
    for _ in range(max(0, n_ops - 3)):
        axis = rng.standard_normal(3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(angle/2), np.sin(angle/2)
        ops.append(c * I2_np + 1j * s * (axis[0]*X_np + axis[1]*Y_np + axis[2]*Z_np))
    
    def evolve(psi):
        return V @ (np.exp(-1j * w * t) * (V.conj().T @ psi))
    
    total = 0.0
    for _ in range(n_states):
        psi0 = random_product_state_np(n_qubits, rng)
        psi_t = evolve(psi0)
        rho_base = reduced_rho_np(psi_t, n_qubits, target)
        
        best_this_state = 0.0
        for Op in ops:
            psi0_p = apply_local_op_np(psi0, Op, n_qubits, source)
            psi_t_p = evolve(psi0_p)
            rho_p = reduced_rho_np(psi_t_p, n_qubits, target)
            D = trace_distance_np(rho_p, rho_base)
            if D > best_this_state:
                best_this_state = D
        
        total += best_this_state
    
    return total / n_states


# =============================================================================
# MAIN: Compute both and compare
# =============================================================================

@dataclass
class ComparisonResult:
    times: List[float]
    capacity: Dict[str, List[float]]      # edge -> [C at each time]
    typicality: Dict[str, List[float]]    # edge -> [T at each time]
    gap: Dict[str, List[float]]           # edge -> [G at each time]
    hip_var_capacity: List[float]
    hip_var_typicality: List[float]
    hip_var_gap: List[float]
    mean_capacity: List[float]
    mean_typicality: List[float]
    mean_gap: List[float]
    config: dict


def compute_both(G, n_qubits, times, J=1.0, h=0.2,
                 cap_restarts=8, cap_iters=200,
                 typ_states=50, typ_ops=10,
                 seed=0, verbose=True):
    
    edges = list(G.edges())
    nodes = list(G.nodes())
    
    # Build Hamiltonian (JAX version for capacity)
    if verbose:
        print("Building Hamiltonian (JAX)...", end=" ", flush=True)
    t0 = time.time()
    H_jax = build_hamiltonian_jax(G, n_qubits, J, h)
    if verbose:
        print(f"done ({time.time()-t0:.1f}s)")
    
    # Diagonalize for typicality (NumPy)
    if verbose:
        print("Diagonalizing (NumPy)...", end=" ", flush=True)
    t0 = time.time()
    H_np = np.array(H_jax)
    w, V = np.linalg.eigh(H_np)
    if verbose:
        print(f"done ({time.time()-t0:.1f}s)")
    
    # Storage
    C_by_edge = {f"{u}-{v}": [] for (u, v) in edges}
    T_by_edge = {f"{u}-{v}": [] for (u, v) in edges}
    G_by_edge = {f"{u}-{v}": [] for (u, v) in edges}
    
    hip_var_C, hip_var_T, hip_var_G = [], [], []
    mean_C, mean_T, mean_G = [], [], []
    
    rng_np = np.random.default_rng(seed)
    
    for t_idx, t in enumerate(times):
        if verbose:
            print(f"\n{'='*60}")
            print(f"TIME t = {t}")
            print('='*60)
        
        # Evolution operator for capacity
        U_t = jax_expm(-1j * H_jax * t)
        
        # Compute both for each edge
        C_this_t = {}
        T_this_t = {}
        
        for (u, v) in edges:
            edge_key = f"{u}-{v}"
            
            # --- CAPACITY ---
            if verbose:
                print(f"\n  Edge {u}-{v}:")
                print(f"    Capacity...", end=" ", flush=True)
            
            t0 = time.time()
            rng_key = jax.random.PRNGKey(seed + t_idx * 1000 + u * 100 + v)
            
            # Both directions, take max
            c_uv = compute_capacity(U_t, n_qubits, u, v, cap_restarts, cap_iters, rng_key=rng_key)
            rng_key, _ = jax.random.split(rng_key)
            c_vu = compute_capacity(U_t, n_qubits, v, u, cap_restarts, cap_iters, rng_key=rng_key)
            C = max(c_uv, c_vu)
            
            if verbose:
                print(f"{C:.4f} ({time.time()-t0:.1f}s)")
            
            # --- TYPICALITY ---
            if verbose:
                print(f"    Typicality...", end=" ", flush=True)
            
            t0 = time.time()
            t_uv = compute_typicality(w, V, t, n_qubits, u, v, typ_states, typ_ops, rng_np)
            t_vu = compute_typicality(w, V, t, n_qubits, v, u, typ_states, typ_ops, rng_np)
            T = max(t_uv, t_vu)
            
            if verbose:
                print(f"{T:.4f} ({time.time()-t0:.1f}s)")
            
            # --- GAP ---
            gap = C - T
            if verbose:
                print(f"    Gap: {gap:.4f}")
            
            C_this_t[edge_key] = C
            T_this_t[edge_key] = T
            
            C_by_edge[edge_key].append(C)
            T_by_edge[edge_key].append(T)
            G_by_edge[edge_key].append(gap)
        
        # Compute HIP variance for this time
        C_vals = list(C_this_t.values())
        T_vals = list(T_this_t.values())
        G_vals = [C_this_t[e] - T_this_t[e] for e in C_this_t]
        
        def variance(x):
            m = sum(x) / len(x)
            return sum((v - m)**2 for v in x) / len(x)
        
        hip_var_C.append(variance(C_vals))
        hip_var_T.append(variance(T_vals))
        hip_var_G.append(variance(G_vals))
        
        mean_C.append(sum(C_vals) / len(C_vals))
        mean_T.append(sum(T_vals) / len(T_vals))
        mean_G.append(sum(G_vals) / len(G_vals))
        
        if verbose:
            print(f"\n  SUMMARY t={t}:")
            print(f"    Capacity:   mean={mean_C[-1]:.4f}, HIP_var={hip_var_C[-1]:.6f}")
            print(f"    Typicality: mean={mean_T[-1]:.4f}, HIP_var={hip_var_T[-1]:.6f}")
            print(f"    Gap:        mean={mean_G[-1]:.4f}, HIP_var={hip_var_G[-1]:.6f}")
    
    return ComparisonResult(
        times=times,
        capacity=C_by_edge,
        typicality=T_by_edge,
        gap=G_by_edge,
        hip_var_capacity=hip_var_C,
        hip_var_typicality=hip_var_T,
        hip_var_gap=hip_var_G,
        mean_capacity=mean_C,
        mean_typicality=mean_T,
        mean_gap=mean_G,
        config={
            'n_qubits': n_qubits,
            'n_edges': len(edges),
            'cap_restarts': cap_restarts,
            'cap_iters': cap_iters,
            'typ_states': typ_states,
            'typ_ops': typ_ops,
            'seed': seed
        }
    )


def plot_comparison(G, result, outdir):
    import matplotlib.pyplot as plt
    
    os.makedirs(outdir, exist_ok=True)
    
    # Summary plot: HIP variance for all three
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(result.times, result.hip_var_capacity, 'o-', lw=2, ms=8, color='blue')
    axes[0].set_xlabel("Time t")
    axes[0].set_ylabel("HIP Variance")
    axes[0].set_title("CAPACITY\n(uniform = structure invisible)")
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    axes[1].plot(result.times, result.hip_var_typicality, 'o-', lw=2, ms=8, color='green')
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("HIP Variance")
    axes[1].set_title("TYPICALITY\n(heterogeneous = structure visible)")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    axes[2].plot(result.times, result.hip_var_gap, 'o-', lw=2, ms=8, color='red')
    axes[2].set_xlabel("Time t")
    axes[2].set_ylabel("HIP Variance")
    axes[2].set_title("GAP (C - T)\n(accessibility structure)")
    axes[2].grid(True, alpha=0.3)
    axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    fig.suptitle("HIP: Capacity vs Typicality", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hip_comparison.png"), dpi=150)
    plt.close(fig)
    
    # Mean values comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.times, result.mean_capacity, 'o-', lw=2, ms=8, label='Capacity (sup)', color='blue')
    ax.plot(result.times, result.mean_typicality, 's-', lw=2, ms=8, label='Typicality (avg)', color='green')
    ax.fill_between(result.times, result.mean_typicality, result.mean_capacity, alpha=0.3, color='red', label='Gap')
    ax.set_xlabel("Time t")
    ax.set_ylabel("Permeability")
    ax.set_title("Capacity vs Typicality: The Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "capacity_vs_typicality.png"), dpi=150)
    plt.close(fig)
    
    # Per-edge comparison at final time
    edges = list(result.capacity.keys())
    C_final = [result.capacity[e][-1] for e in edges]
    T_final = [result.typicality[e][-1] for e in edges]
    G_final = [result.gap[e][-1] for e in edges]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(edges))
    width = 0.35
    
    ax.bar(x - width/2, C_final, width, label='Capacity', color='blue', alpha=0.7)
    ax.bar(x + width/2, T_final, width, label='Typicality', color='green', alpha=0.7)
    
    ax.set_xlabel("Edge")
    ax.set_ylabel("Permeability")
    ax.set_title(f"Per-Edge Comparison (t = {result.times[-1]})")
    ax.set_xticks(x)
    ax.set_xticklabels(edges, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "per_edge_comparison.png"), dpi=150)
    plt.close(fig)
    
    print(f"Plots saved to {outdir}/")


def main():
    parser = argparse.ArgumentParser(description="HIP: Capacity vs Typicality")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--graph", type=str, default="random", choices=["ring", "line", "grid", "random"])
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.2)
    parser.add_argument("--times", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--cap-restarts", type=int, default=6)
    parser.add_argument("--cap-iters", type=int, default=150)
    parser.add_argument("--typ-states", type=int, default=50)
    parser.add_argument("--typ-ops", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs_comparison")
    parser.add_argument("--no-plots", action="store_true")
    
    args = parser.parse_args()
    times = [float(x.strip()) for x in args.times.split(",")]
    
    print("=" * 60)
    print("  HIP: CAPACITY vs TYPICALITY")
    print("=" * 60)
    print(f"N = {args.n}, graph = {args.graph}")
    print(f"times = {times}")
    print(f"Capacity: {args.cap_restarts} restarts, {args.cap_iters} iters")
    print(f"Typicality: {args.typ_states} states, {args.typ_ops} ops")
    print("=" * 60)
    
    G = make_graph(args.graph, args.n, seed=args.seed)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    result = compute_both(
        G, args.n, times, args.J, args.h,
        args.cap_restarts, args.cap_iters,
        args.typ_states, args.typ_ops,
        args.seed, verbose=True
    )
    
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    if not args.no_plots:
        plot_comparison(G, result, args.out)
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Time':<8} {'C_var':<12} {'T_var':<12} {'G_var':<12} {'mean_C':<10} {'mean_T':<10} {'mean_G':<10}")
    print("-" * 74)
    for i, t in enumerate(times):
        print(f"{t:<8.2f} {result.hip_var_capacity[i]:<12.6f} {result.hip_var_typicality[i]:<12.6f} "
              f"{result.hip_var_gap[i]:<12.6f} {result.mean_capacity[i]:<10.4f} "
              f"{result.mean_typicality[i]:<10.4f} {result.mean_gap[i]:<10.4f}")
    print("=" * 60)
    
    print("\nKEY INSIGHT:")
    if result.hip_var_capacity[-1] < 0.001 and result.hip_var_typicality[-1] > result.hip_var_capacity[-1] * 10:
        print("  ✓ Capacity is UNIFORM (all edges can carry information)")
        print("  ✓ Typicality is HETEROGENEOUS (some edges are easier than others)")
        print("  → The structure lives in ACCESSIBILITY, not CAPACITY")
    else:
        print("  Results inconclusive - check the plots")


if __name__ == "__main__":
    main()