#!/usr/bin/env python3
"""
HIP Typicality (JAX/GPU)
========================

GPU-accelerated version using JAX.
Falls back to CPU if no GPU available.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import networkx as nx

# JAX setup
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

# Check for GPU
print(f"JAX devices: {jax.devices()}")

jax.config.update("jax_enable_x64", True)


# Pauli matrices
I2 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = [I2, X, Y, Z]


def kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def embed_single_site(op2, n_qubits, site):
    mats = [I2] * n_qubits
    mats[site] = op2
    return kron_list(mats)


def embed_two_site(op4, n_qubits, site_a, site_b):
    out = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for m in range(4):
        for n in range(4):
            basis = np.kron(PAULIS[m], PAULIS[n])
            c = 0.25 * np.trace(basis.conj().T @ op4)
            if np.abs(c) > 1e-14:
                Em = embed_single_site(PAULIS[m], n_qubits, site_a)
                En = embed_single_site(PAULIS[n], n_qubits, site_b)
                out = out + c * (Em @ En)
    return 0.5 * (out + out.conj().T)


def build_hamiltonian(G, n_qubits, J=1.0, h=0.2):
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n_qubits):
        H += h * embed_single_site(Z, n_qubits, i)
    HH = J * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z))
    for (i, j) in G.edges():
        H += embed_two_site(HH, n_qubits, i, j)
    return 0.5 * (H + H.conj().T)


def make_graph(graph_type, n, seed=0):
    if graph_type == "ring":
        return nx.cycle_graph(n)
    elif graph_type == "line":
        return nx.path_graph(n)
    elif graph_type == "grid":
        side = int(math.ceil(math.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
        nodes = list(G.nodes())[:n]
        G = G.subgraph(nodes).copy()
        return nx.convert_node_labels_to_integers(G)
    elif graph_type == "random":
        G = nx.erdos_renyi_graph(n, 0.3, seed=seed)
        if not nx.is_connected(G):
            for i in range(n - 1):
                G.add_edge(i, i + 1)
        return G
    raise ValueError(f"Unknown: {graph_type}")


# =============================================================================
# JAX-accelerated functions
# =============================================================================

@partial(jit, static_argnums=(1, 2))
def apply_local_op_jax(psi, n_qubits, site, Op):
    """Apply 2x2 operator at site. JIT-compiled."""
    # Reshape to tensor
    shape = tuple([2] * n_qubits)
    tensor = psi.reshape(shape)
    
    # Move target site to front
    perm = [site] + [k for k in range(n_qubits) if k != site]
    tensor = jnp.transpose(tensor, perm)
    
    # Apply operator
    tensor_2d = tensor.reshape(2, -1)
    tensor_2d = Op @ tensor_2d
    tensor = tensor_2d.reshape(tensor.shape)
    
    # Move back
    inv_perm = [0] * n_qubits
    for i, p in enumerate(perm):
        inv_perm[p] = i
    tensor = jnp.transpose(tensor, inv_perm)
    
    return tensor.reshape(-1)


@partial(jit, static_argnums=(1, 2))
def reduced_rho_jax(psi, n_qubits, target):
    """Compute reduced density matrix at target site."""
    shape = tuple([2] * n_qubits)
    tensor = psi.reshape(shape)
    perm = [target] + [k for k in range(n_qubits) if k != target]
    tensor = jnp.transpose(tensor, perm)
    A = tensor.reshape(2, -1)
    return A @ A.conj().T


@jit
def trace_distance_jax(rho, sigma):
    """Trace distance between two 2x2 density matrices."""
    delta = rho - sigma
    # For 2x2, eigenvalues of Hermitian matrix
    s = jnp.linalg.svd(delta, compute_uv=False)
    return 0.5 * jnp.sum(jnp.abs(s))


@jit
def evolve_state(psi, phases, V, Vdag):
    """Evolve state: V @ diag(phases) @ V† @ psi"""
    return V @ (phases * (Vdag @ psi))


def random_product_states_jax(n_qubits, n_states, key):
    """Generate batch of random product states."""
    keys = jax.random.split(key, n_qubits)
    
    # Generate angles for each qubit
    states = []
    for _ in range(n_states):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_qubits)
        
        psi = None
        for k in range(n_qubits):
            theta = jnp.arccos(1 - 2 * jax.random.uniform(keys[k]))
            phi = 2 * jnp.pi * jax.random.uniform(jax.random.split(keys[k])[1])
            qubit = jnp.array([jnp.cos(theta/2), jnp.sin(theta/2) * jnp.exp(1j * phi)])
            if psi is None:
                psi = qubit
            else:
                psi = jnp.kron(psi, qubit)
        states.append(psi)
    
    return jnp.stack(states)


def random_su2_ops(n_ops, key):
    """Generate random SU(2) operators."""
    I2_j = jnp.array(I2)
    X_j = jnp.array(X)
    Y_j = jnp.array(Y)
    Z_j = jnp.array(Z)
    
    ops = [X_j, Y_j, Z_j]  # Always include Paulis
    
    for i in range(max(0, n_ops - 3)):
        key, k1, k2 = jax.random.split(key, 3)
        axis = jax.random.normal(k1, (3,))
        axis = axis / (jnp.linalg.norm(axis) + 1e-12)
        angle = jax.random.uniform(k2) * 2 * jnp.pi
        c, s = jnp.cos(angle/2), jnp.sin(angle/2)
        U = c * I2_j + 1j * s * (axis[0]*X_j + axis[1]*Y_j + axis[2]*Z_j)
        ops.append(U)
    
    return ops


def compute_typicality_jax(V, Vdag, phases, n_qubits, source, target, n_states, n_ops, key):
    """
    Compute typicality using JAX.
    """
    key, k1, k2 = jax.random.split(key, 3)
    
    # Generate random states and operators
    psi_batch = random_product_states_jax(n_qubits, n_states, k1)
    ops = random_su2_ops(n_ops, k2)
    
    total = 0.0
    
    for i in range(n_states):
        psi0 = psi_batch[i]
        
        # Evolve base state
        psi_t = evolve_state(psi0, phases, V, Vdag)
        rho_base = reduced_rho_jax(psi_t, n_qubits, target)
        
        best = 0.0
        for Op in ops:
            # Apply operator at source
            psi0_p = apply_local_op_jax(psi0, n_qubits, source, Op)
            psi_t_p = evolve_state(psi0_p, phases, V, Vdag)
            rho_p = reduced_rho_jax(psi_t_p, n_qubits, target)
            
            D = trace_distance_jax(rho_p, rho_base)
            best = jnp.maximum(best, D)
        
        total += best
    
    return float(total / n_states)


def main():
    parser = argparse.ArgumentParser(description="HIP Typicality (JAX/GPU)")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--graph", default="random", choices=["ring", "line", "grid", "random"])
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.2)
    parser.add_argument("--times", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--ops", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs_typicality")
    
    args = parser.parse_args()
    times = [float(x) for x in args.times.split(",")]
    
    print("=" * 60)
    print("  HIP TYPICALITY (JAX/GPU)")
    print("=" * 60)
    print(f"N = {args.n}, graph = {args.graph}")
    print(f"times = {times}")
    print(f"states = {args.states}, ops = {args.ops}")
    print("=" * 60)
    
    G = make_graph(args.graph, args.n, args.seed)
    edges = list(G.edges())
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Build Hamiltonian (NumPy, one-time cost)
    print("Building Hamiltonian...", end=" ", flush=True)
    t0 = time.time()
    H = build_hamiltonian(G, args.n, args.J, args.h)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # Diagonalize (NumPy)
    print("Diagonalizing...", end=" ", flush=True)
    t0 = time.time()
    w, V_np = np.linalg.eigh(H)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # Transfer to JAX/GPU
    print("Transferring to device...", end=" ", flush=True)
    V = jnp.array(V_np)
    Vdag = V.conj().T
    w_jax = jnp.array(w)
    print("done")
    
    # Warm up JIT
    print("JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    key = jax.random.PRNGKey(args.seed)
    phases_test = jnp.exp(-1j * w_jax * 1.0)
    _ = compute_typicality_jax(V, Vdag, phases_test, args.n, 0, 1, 5, 3, key)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # Storage
    capacity = {f"{u}-{v}": [] for (u, v) in edges}
    typicality = {f"{u}-{v}": [] for (u, v) in edges}
    gap = {f"{u}-{v}": [] for (u, v) in edges}
    
    hip_var_typicality = []
    mean_typicality = []
    
    key = jax.random.PRNGKey(args.seed)
    
    for t_idx, t in enumerate(times):
        print(f"\n[t = {t}]")
        t_start = time.time()
        
        phases = jnp.exp(-1j * w_jax * t)
        T_this_t = {}
        
        for idx, (u, v) in enumerate(edges):
            edge_key = f"{u}-{v}"
            
            key, k1, k2 = jax.random.split(key, 3)
            
            t_uv = compute_typicality_jax(V, Vdag, phases, args.n, u, v, args.states, args.ops, k1)
            t_vu = compute_typicality_jax(V, Vdag, phases, args.n, v, u, args.states, args.ops, k2)
            T = max(t_uv, t_vu)
            
            T_this_t[edge_key] = T
            capacity[edge_key].append(1.0)
            typicality[edge_key].append(T)
            gap[edge_key].append(1.0 - T)
            
            print(f"  {edge_key}: T={T:.4f}", end="")
            if (idx + 1) % 4 == 0:
                print()
        
        if len(edges) % 4 != 0:
            print()
        
        T_vals = list(T_this_t.values())
        
        def variance(x):
            m = sum(x) / len(x)
            return sum((val - m)**2 for val in x) / len(x)
        
        hip_var_typicality.append(variance(T_vals))
        mean_typicality.append(sum(T_vals) / len(T_vals))
        
        elapsed = time.time() - t_start
        print(f"  Time: {elapsed:.1f}s | HIP_T={hip_var_typicality[-1]:.6f} | mean_T={mean_typicality[-1]:.4f}")
    
    # Save results
    os.makedirs(args.out, exist_ok=True)
    
    results = {
        'times': times,
        'capacity': capacity,
        'typicality': typicality,
        'gap': gap,
        'hip_var_capacity': [0.0] * len(times),
        'hip_var_typicality': hip_var_typicality,
        'hip_var_gap': hip_var_typicality,  # Same as typicality when capacity=1
        'mean_capacity': [1.0] * len(times),
        'mean_typicality': mean_typicality,
        'mean_gap': [1.0 - m for m in mean_typicality],
        'config': {
            'n_qubits': args.n,
            'n_edges': len(edges),
            'graph': args.graph,
            'states': args.states,
            'ops': args.ops,
            'seed': args.seed,
            'method': 'jax_sampling'
        }
    }
    
    outpath = os.path.join(args.out, "results.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to {outpath}")
    print("=" * 60)
    print(f"\nNext: python emergent_geometry.py --input {outpath} --metric log_typicality --out geometry_output")


if __name__ == "__main__":
    main()