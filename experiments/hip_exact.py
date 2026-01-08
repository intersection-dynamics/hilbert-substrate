#!/usr/bin/env python3
"""
HIP Exact Permeability via Variational Optimization
====================================================

Computes P_{i->j}(t) as defined in Eq. 2 of the paper:

  P_{i->j}(t) = sup_{ρ, O_i} || ρ_j(t; O_i) - ρ_j(t; I) ||_1

where ρ_j(t; O_i) is the reduced state on subsystem j after evolving
O_i ρ O_i† under U(t).

We restrict to pure states ρ = |ψ⟩⟨ψ| (sufficient for the supremum over
all states for this trace-distance problem) and parameterize O_i as a
general SU(2) element.

Uses JAX for autodiff + GPU acceleration.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

# JAX imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.linalg import expm as jax_expm
import optax

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# -----------------------------
# Pauli matrices (JAX)
# -----------------------------

I2 = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex128)
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
PAULIS = [I2, X, Y, Z]


# -----------------------------
# Tensor product utilities
# -----------------------------

def kron_list(mats):
    """Kronecker product of a list of matrices."""
    out = mats[0]
    for m in mats[1:]:
        out = jnp.kron(out, m)
    return out


def embed_single_site(op2, n_qubits, site):
    """Embed a 2x2 operator at site into 2^n x 2^n space."""
    mats = [I2] * n_qubits
    mats[site] = op2
    return kron_list(mats)


def embed_two_site_via_pauli(op4, n_qubits, site_a, site_b):
    """
    Embed a 4x4 two-qubit operator via Pauli decomposition.
    Works for any site_a, site_b (not just adjacent).
    """
    # Decompose op4 = sum_{m,n} c_{mn} (P_m ⊗ P_n)
    # c_{mn} = (1/4) Tr[(P_m ⊗ P_n)† op4]
    out = jnp.zeros((2**n_qubits, 2**n_qubits), dtype=jnp.complex128)
    for m in range(4):
        for n in range(4):
            basis = jnp.kron(PAULIS[m], PAULIS[n])
            c = 0.25 * jnp.trace(basis.conj().T @ op4)
            if jnp.abs(c) > 1e-14:
                Em = embed_single_site(PAULIS[m], n_qubits, site_a)
                En = embed_single_site(PAULIS[n], n_qubits, site_b)
                out = out + c * (Em @ En)
    return 0.5 * (out + out.conj().T)


# -----------------------------
# State and operator parameterization
# -----------------------------

def params_to_state(params_real, params_imag):
    """
    Convert real parameters to normalized complex state vector.
    params_real, params_imag: arrays of shape (dim,)
    """
    psi = params_real + 1j * params_imag
    norm = jnp.linalg.norm(psi)
    return psi / (norm + 1e-12)


def params_to_su2(theta):
    """
    Convert 3 parameters to SU(2) matrix via exponential map.
    theta: array of shape (3,) representing coefficients of X, Y, Z
    U = exp(i * (theta[0]*X + theta[1]*Y + theta[2]*Z))
    """
    # Generator: H = theta[0]*X + theta[1]*Y + theta[2]*Z
    H = theta[0] * X + theta[1] * Y + theta[2] * Z
    return jax_expm(1j * H)


# -----------------------------
# Reduced density matrix and trace distance
# -----------------------------

def reduced_rho_single_qubit(psi, n_qubits, target_qubit):
    """
    Compute 2x2 reduced density matrix on target_qubit from pure state psi.
    """
    # Reshape to tensor
    tensor = psi.reshape((2,) * n_qubits)
    # Move target qubit to front
    perm = [target_qubit] + [k for k in range(n_qubits) if k != target_qubit]
    tensor = jnp.transpose(tensor, perm)
    # Reshape to (2, 2^{n-1})
    A = tensor.reshape((2, 2**(n_qubits - 1)))
    # rho = A @ A†
    rho = A @ A.conj().T
    return rho


def trace_norm_2x2(A):
    """
    Trace norm of a 2x2 matrix: sum of singular values.
    For Hermitian A with Tr(A)=0: ||A||_1 = 2 * sqrt(|A[0,0]|^2 + |A[0,1]|^2)
    """
    # General SVD approach (works for any 2x2)
    s = jnp.linalg.svd(A, compute_uv=False)
    return jnp.sum(s)


def trace_distance_2x2(rho, sigma):
    """Trace distance: D(ρ,σ) = (1/2)||ρ - σ||_1"""
    return 0.5 * trace_norm_2x2(rho - sigma)


# -----------------------------
# Core permeability computation
# -----------------------------

def permeability_objective(state_params_real, state_params_imag, op_params, 
                           U_t, n_qubits, source, target):
    """
    Compute trace distance for given state and operator parameters.
    This is the quantity we want to MAXIMIZE.
    
    Returns negative trace distance (for minimization).
    """
    # Reconstruct state and operator
    psi0 = params_to_state(state_params_real, state_params_imag)
    O_i = params_to_su2(op_params)
    
    # Embed O_i at source site
    O_i_full = embed_single_site(O_i, n_qubits, source)
    
    # Evolve unperturbed: |φ⟩ = U(t)|ψ⟩
    phi = U_t @ psi0
    rho_j = reduced_rho_single_qubit(phi, n_qubits, target)
    
    # Evolve perturbed: |φ'⟩ = U(t) O_i |ψ⟩
    psi0_pert = O_i_full @ psi0
    phi_pert = U_t @ psi0_pert
    rho_j_pert = reduced_rho_single_qubit(phi_pert, n_qubits, target)
    
    # Trace distance
    D = trace_distance_2x2(rho_j_pert, rho_j)
    
    # Return negative for minimization
    return -D


def optimize_permeability(U_t, n_qubits, source, target, 
                          n_restarts=8, max_iters=200, lr=0.05,
                          rng_key=None):
    """
    Find supremum of trace distance over states and operators.
    Uses multiple random restarts with Adam optimizer.
    
    Returns: best permeability value found
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    dim = 2 ** n_qubits
    
    # JIT compile the objective and its gradient
    @jit
    def loss_fn(params):
        sr, si, op = params['state_real'], params['state_imag'], params['op']
        return permeability_objective(sr, si, op, U_t, n_qubits, source, target)
    
    grad_fn = jit(grad(loss_fn))
    
    best_value = 0.0
    
    for restart in range(n_restarts):
        # Random initialization
        rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
        
        params = {
            'state_real': jax.random.normal(k1, (dim,)),
            'state_imag': jax.random.normal(k2, (dim,)),
            'op': jax.random.uniform(k3, (3,), minval=-math.pi, maxval=math.pi)
        }
        
        # Adam optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        
        for _ in range(max_iters):
            grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        
        # Get final value (negate since we minimized negative)
        final_loss = float(loss_fn(params))
        final_perm = -final_loss
        
        if final_perm > best_value:
            best_value = final_perm
    
    return best_value


# -----------------------------
# Hamiltonian construction
# -----------------------------

def build_heisenberg_hamiltonian(G: nx.Graph, n_qubits: int, J: float = 1.0, h: float = 0.2):
    """Build Heisenberg XXX Hamiltonian on graph G with local Z field."""
    dim = 2 ** n_qubits
    H = jnp.zeros((dim, dim), dtype=jnp.complex128)
    
    # Local fields
    for i in range(n_qubits):
        H = H + h * embed_single_site(Z, n_qubits, i)
    
    # Edge couplings
    XX = jnp.kron(X, X)
    YY = jnp.kron(Y, Y)
    ZZ = jnp.kron(Z, Z)
    HH = J * (XX + YY + ZZ)
    
    for (i, j) in G.edges():
        H = H + embed_two_site_via_pauli(HH, n_qubits, i, j)
    
    return 0.5 * (H + H.conj().T)


# -----------------------------
# Graph generators
# -----------------------------

def make_graph(graph_type: str, n: int, seed: int = 0) -> Tuple[nx.Graph, dict]:
    """Generate interaction graph and layout."""
    rng = np.random.default_rng(seed)
    
    if graph_type == "ring":
        G = nx.cycle_graph(n)
        pos = nx.circular_layout(G)
    elif graph_type == "line":
        G = nx.path_graph(n)
        pos = {i: (i, 0) for i in range(n)}
    elif graph_type == "grid":
        side = int(math.ceil(math.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
        nodes = list(G.nodes())[:n]
        G = G.subgraph(nodes).copy()
        G = nx.convert_node_labels_to_integers(G)
        pos = {i: (i % side, i // side) for i in range(G.number_of_nodes())}
    elif graph_type == "random":
        G = nx.erdos_renyi_graph(n, 0.3, seed=seed)
        # Ensure connected
        if not nx.is_connected(G):
            for i in range(n - 1):
                G.add_edge(i, i + 1)
        pos = nx.spring_layout(G, seed=seed)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    return G, pos


# -----------------------------
# Full HIP computation
# -----------------------------

@dataclass
class HIPResult:
    times: List[float]
    hip_variance: List[float]
    mean_permeability: List[float]
    edge_permeabilities: List[Dict[str, float]]
    node_intensities: List[Dict[int, float]]
    directed_permeabilities: List[Dict[str, float]]
    config: dict


def compute_hip_exact(G: nx.Graph, H: jnp.ndarray, n_qubits: int,
                      times: List[float], n_restarts: int = 8,
                      max_iters: int = 200, lr: float = 0.05,
                      seed: int = 0, verbose: bool = True) -> HIPResult:
    """
    Compute exact HIP diagnostics via variational optimization.
    """
    results = HIPResult(
        times=times,
        hip_variance=[],
        mean_permeability=[],
        edge_permeabilities=[],
        node_intensities=[],
        directed_permeabilities=[],
        config={
            'n_qubits': n_qubits,
            'n_edges': G.number_of_edges(),
            'n_restarts': n_restarts,
            'max_iters': max_iters,
            'seed': seed
        }
    )
    
    edges = list(G.edges())
    nodes = list(G.nodes())
    
    for t_idx, t in enumerate(times):
        if verbose:
            print(f"\n[t={t:.3f}] Computing evolution operator...")
        
        # Time evolution operator
        U_t = jax_expm(-1j * H * t)
        
        # Compute directed permeabilities P_{i->j} for all neighbor pairs
        P_dir = {}
        rng_key = jax.random.PRNGKey(seed + t_idx * 1000)
        
        for source in nodes:
            neighbors = list(G.neighbors(source))
            for target in neighbors:
                if verbose:
                    print(f"  Optimizing P_{source}->{target}...", end=" ", flush=True)
                
                rng_key, subkey = jax.random.split(rng_key)
                
                start = time.time()
                p_val = optimize_permeability(
                    U_t, n_qubits, source, target,
                    n_restarts=n_restarts, max_iters=max_iters, lr=lr,
                    rng_key=subkey
                )
                elapsed = time.time() - start
                
                P_dir[(source, target)] = float(p_val)
                
                if verbose:
                    print(f"{p_val:.4f} ({elapsed:.1f}s)")
        
        # Edge permeabilities (symmetric)
        P_edge = {}
        for (u, v) in edges:
            P_edge[(u, v)] = max(P_dir.get((u, v), 0), P_dir.get((v, u), 0))
        
        edge_vals = list(P_edge.values())
        
        # HIP variance
        if len(edge_vals) > 0:
            mean_p = sum(edge_vals) / len(edge_vals)
            var_p = sum((x - mean_p)**2 for x in edge_vals) / len(edge_vals)
        else:
            mean_p, var_p = 0.0, 0.0
        
        # Node intensities
        I_node = {u: sum(P_dir.get((u, v), 0) for v in G.neighbors(u)) for u in nodes}
        
        results.hip_variance.append(var_p)
        results.mean_permeability.append(mean_p)
        results.edge_permeabilities.append({f"{u}-{v}": p for (u, v), p in P_edge.items()})
        results.node_intensities.append(I_node)
        results.directed_permeabilities.append({f"{u}->{v}": p for (u, v), p in P_dir.items()})
        
        if verbose:
            print(f"  HIP_var = {var_p:.6f}, mean_P = {mean_p:.4f}")
    
    return results


# -----------------------------
# Visualization
# -----------------------------

def plot_results(G: nx.Graph, pos: dict, result: HIPResult, outdir: str):
    """Generate visualization plots."""
    import matplotlib.pyplot as plt
    
    os.makedirs(outdir, exist_ok=True)
    
    # HIP timeseries
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.times, result.hip_variance, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("HIP(t) = Var[P_edge]", fontsize=12)
    ax.set_title("Heterogeneity in Information Propagation", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hip_timeseries.png"), dpi=150)
    plt.close(fig)
    
    # Edge field snapshots
    for t_idx, t in enumerate(result.times):
        P_edge = result.edge_permeabilities[t_idx]
        edges = list(G.edges())
        edge_vals = [P_edge.get(f"{u}-{v}", P_edge.get(f"{v}-{u}", 0)) for (u, v) in edges]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title(f"Edge Permeability (t={t:.3f})", fontsize=14)
        
        vmin = min(edge_vals) if edge_vals else 0
        vmax = max(edge_vals) if edge_vals else 1
        if vmax <= vmin:
            vmax = vmin + 0.01
        
        nx.draw_networkx_nodes(G, pos, node_size=200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        edges_drawn = nx.draw_networkx_edges(
            G, pos, edge_color=edge_vals, width=3,
            edge_cmap=plt.cm.viridis, edge_vmin=vmin, edge_vmax=vmax, ax=ax
        )
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin, vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="P_edge")
        
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"edge_field_t{t_idx:02d}.png"), dpi=150)
        plt.close(fig)
    
    # Node intensity persistence
    fig, ax = plt.subplots(figsize=(9, 5))
    n_nodes = G.number_of_nodes()
    
    # Get top-k nodes by mean intensity
    mean_intens = {}
    for u in G.nodes():
        mean_intens[u] = sum(result.node_intensities[t][u] for t in range(len(result.times))) / len(result.times)
    
    topk = min(5, n_nodes)
    top_nodes = sorted(mean_intens.keys(), key=lambda x: mean_intens[x], reverse=True)[:topk]
    
    for u in top_nodes:
        intensities = [result.node_intensities[t][u] for t in range(len(result.times))]
        ax.plot(result.times, intensities, 'o-', label=f"Node {u}", linewidth=1.5)
    
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("I(u,t) = Σ P_{u→v}(t)", fontsize=12)
    ax.set_title("Node Intensity Persistence (top nodes)", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "node_persistence.png"), dpi=150)
    plt.close(fig)
    
    print(f"Plots saved to {outdir}/")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="HIP Exact Permeability Computation")
    parser.add_argument("--n", type=int, default=6, help="Number of qubits")
    parser.add_argument("--graph", type=str, default="ring", 
                        choices=["ring", "line", "grid", "random"])
    parser.add_argument("--J", type=float, default=1.0, help="Coupling strength")
    parser.add_argument("--h", type=float, default=0.2, help="Local field strength")
    parser.add_argument("--times", type=str, default="0.5,1.0,2.0,4.0",
                        help="Comma-separated time points")
    parser.add_argument("--restarts", type=int, default=8, 
                        help="Optimization restarts per (source,target)")
    parser.add_argument("--iters", type=int, default=200,
                        help="Max iterations per restart")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs_hip_exact")
    parser.add_argument("--no-plots", action="store_true")
    
    args = parser.parse_args()
    
    times = [float(x.strip()) for x in args.times.split(",")]
    
    print("=" * 60)
    print("  HIP EXACT PERMEABILITY (Variational Optimization)")
    print("=" * 60)
    print(f"N = {args.n} qubits, graph = {args.graph}")
    print(f"J = {args.J}, h = {args.h}")
    print(f"times = {times}")
    print(f"restarts = {args.restarts}, iters = {args.iters}, lr = {args.lr}")
    print("=" * 60)
    
    # Build graph and Hamiltonian
    G, pos = make_graph(args.graph, args.n, seed=args.seed)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    H = build_heisenberg_hamiltonian(G, args.n, J=args.J, h=args.h)
    print("Hamiltonian constructed.")
    
    # Compute HIP
    result = compute_hip_exact(
        G, H, args.n, times,
        n_restarts=args.restarts, max_iters=args.iters, lr=args.lr,
        seed=args.seed, verbose=True
    )
    
    # Save results
    os.makedirs(args.out, exist_ok=True)
    
    # Convert to JSON-serializable
    result_dict = asdict(result)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"\nResults saved to {args.out}/results.json")
    
    # Plots
    if not args.no_plots:
        plot_results(G, pos, result, args.out)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, t in enumerate(times):
        print(f"t={t:.2f}: HIP_var={result.hip_variance[i]:.6f}, mean_P={result.mean_permeability[i]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()