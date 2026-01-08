#!/usr/bin/env python3
"""
Joint Optimization of Factorization and Link Topology

Minimizes J(φ, L) = α * LocalityCost(φ, L) + β * DecoherenceCost(φ, L)

where:
- φ is a factorization (parameterized by a unitary U)
- L is a link topology (which subsystems connect, with what strength)

The question: what link structure emerges when we optimize for both
dynamical simplicity AND observational stability?

Author: Ben Bray
"""

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import os

# ============================================================
# PAULI BASIS
# ============================================================

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = [I, X, Y, Z]
PAULI_LABELS = ['I', 'X', 'Y', 'Z']


def tensor_product(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def pauli_basis(n_qubits):
    """Generate full Pauli basis for n qubits."""
    basis = []
    labels = []
    for indices in np.ndindex(*([4] * n_qubits)):
        ops = [PAULIS[i] for i in indices]
        basis.append(tensor_product(ops))
        labels.append(''.join(PAULI_LABELS[i] for i in indices))
    return basis, labels


# ============================================================
# FACTORIZATION PARAMETERIZATION
# ============================================================

def random_hermitian(dim, seed=None):
    """Generate random Hermitian matrix."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    return (A + A.conj().T) / 2


def unitary_from_params(params, dim):
    """
    Parameterize U(dim) via exponential map.
    params: real vector of length dim^2
    Returns: unitary matrix
    """
    # Construct Hermitian generator from params
    H = np.zeros((dim, dim), dtype=complex)
    idx = 0
    
    # Diagonal (real)
    for i in range(dim):
        H[i, i] = params[idx]
        idx += 1
    
    # Off-diagonal (complex)
    for i in range(dim):
        for j in range(i + 1, dim):
            H[i, j] = params[idx] + 1j * params[idx + 1]
            H[j, i] = params[idx] - 1j * params[idx + 1]
            idx += 2
    
    return expm(1j * H)


def params_from_unitary(U):
    """Extract parameters from unitary (inverse of above)."""
    dim = U.shape[0]
    H = -1j * logm(U)
    H = (H + H.conj().T) / 2  # Ensure Hermitian
    
    params = []
    # Diagonal
    for i in range(dim):
        params.append(np.real(H[i, i]))
    
    # Off-diagonal
    for i in range(dim):
        for j in range(i + 1, dim):
            params.append(np.real(H[i, j]))
            params.append(np.imag(H[i, j]))
    
    return np.array(params)


def n_unitary_params(dim):
    """Number of real parameters for U(dim)."""
    return dim + 2 * (dim * (dim - 1) // 2)  # = dim^2


# ============================================================
# LINK TOPOLOGY PARAMETERIZATION  
# ============================================================

def link_strengths_from_params(params, n_sites):
    """
    Convert real parameters to link strengths.
    Uses sigmoid to keep strengths in [0, max_strength].
    """
    max_strength = 2.0
    n_links = n_sites * (n_sites - 1) // 2
    return max_strength / (1 + np.exp(-params[:n_links]))


def params_from_link_strengths(strengths, n_sites):
    """Inverse sigmoid."""
    max_strength = 2.0
    # Avoid numerical issues
    strengths = np.clip(strengths, 1e-6, max_strength - 1e-6)
    return -np.log(max_strength / strengths - 1)


def n_link_params(n_sites):
    """Number of link parameters."""
    return n_sites * (n_sites - 1) // 2


def link_dict_from_strengths(strengths, n_sites):
    """Convert strength array to {(i,j): strength} dict."""
    links = {}
    idx = 0
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            links[(i, j)] = strengths[idx]
            idx += 1
    return links


# ============================================================
# HAMILTONIAN CONSTRUCTION
# ============================================================

def build_heisenberg_hamiltonian(n_sites, links):
    """
    H = h * Σ Z_i + Σ_{i,j} L_ij * (XX + YY + ZZ)
    """
    dim = 2 ** n_sites
    H = np.zeros((dim, dim), dtype=complex)
    h = 0.5  # Local field
    
    # Local terms
    for i in range(n_sites):
        ops = [I] * n_sites
        ops[i] = Z
        H += h * tensor_product(ops)
    
    # Interaction terms
    for (i, j), L_ij in links.items():
        for P in [X, Y, Z]:
            ops = [I] * n_sites
            ops[i] = P
            ops[j] = P
            H += L_ij * tensor_product(ops)
    
    return H


# ============================================================
# LOCALITY COST
# ============================================================

def compute_locality_cost(H, U, n_sites):
    """
    Compute locality cost of H in factorization defined by U.
    
    L[φ; H] = Σ_S |S|^2 * ||H_S||^2 / ||H||^2
    
    where H_S is the component with support S.
    """
    dim = 2 ** n_sites
    
    # Transform H to new factorization
    H_transformed = U @ H @ U.conj().T
    
    # Decompose into Pauli basis
    basis, labels = pauli_basis(n_sites)
    
    H_norm_sq = np.linalg.norm(H_transformed, 'fro') ** 2
    
    locality_cost = 0.0
    
    for op, label in zip(basis, labels):
        # Coefficient in Pauli basis
        coeff = np.trace(op.conj().T @ H_transformed) / dim
        
        if np.abs(coeff) < 1e-12:
            continue
        
        # Support size = number of non-identity Paulis
        support_size = sum(1 for c in label if c != 'I')
        
        # Contribution to locality cost
        contrib = (support_size ** 2) * (np.abs(coeff) ** 2) * dim
        locality_cost += contrib
    
    return locality_cost / H_norm_sq


# ============================================================
# DECOHERENCE COST
# ============================================================

def compute_decoherence_rate(rho, H_int, n_sites):
    """
    Compute initial decoherence rate: d/dt(1 - Tr(ρ²)) at t=0
    
    For H_int = Σ Z_i ⊗ Z_i^env, this measures how fast
    coherence is lost.
    """
    # Simplified: use commutator norm as proxy for decoherence rate
    comm = H_int @ rho - rho @ H_int
    return np.linalg.norm(comm, 'fro') ** 2


def build_interaction_hamiltonian(n_sites, links, env_coupling=0.1):
    """
    System-environment interaction.
    
    H_int = g * Σ_{i,j} L_ij * Z_i ⊗ Z_j
    
    Environment monitors via the same topology as system links.
    Stronger links = more monitoring = more decoherence.
    """
    dim = 2 ** n_sites
    H_int = np.zeros((dim, dim), dtype=complex)
    
    for (i, j), L_ij in links.items():
        ops = [I] * n_sites
        ops[i] = Z
        ops[j] = Z
        H_int += env_coupling * L_ij * tensor_product(ops)
    
    return H_int


def compute_decoherence_cost(U, n_sites, links, n_samples=20, env_coupling=0.1):
    """
    Compute decoherence cost in factorization U with given links.
    
    Average over random initial states in the transformed basis.
    """
    dim = 2 ** n_sites
    
    # Build interaction Hamiltonian
    H_int = build_interaction_hamiltonian(n_sites, links, env_coupling)
    
    # Transform to new factorization
    H_int_transformed = U @ H_int @ U.conj().T
    
    total_rate = 0.0
    
    for _ in range(n_samples):
        # Random pure state
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        
        rate = compute_decoherence_rate(rho, H_int_transformed, n_sites)
        total_rate += rate
    
    return total_rate / n_samples


# ============================================================
# ALTERNATIVE: COHERENCE SURVIVAL
# ============================================================

def simulate_decoherence(rho_init, H_sys, H_int, t_final=0.5, n_steps=50):
    """
    Simulate decoherence by evolving under H_sys + H_int.
    
    Track purity of system.
    """
    dt = t_final / n_steps
    rho = rho_init.copy()
    H_total = H_sys + H_int
    U = expm(-1j * H_total * dt)
    
    purities = [np.real(np.trace(rho @ rho))]
    
    for _ in range(n_steps):
        rho = U @ rho @ U.conj().T
        purities.append(np.real(np.trace(rho @ rho)))
    
    return np.array(purities)


def compute_coherence_survival(U, H, n_sites, links, t_final=0.3, env_coupling=0.2):
    """
    Measure how much coherence survives under decoherence.
    
    Returns average final purity (higher = more robust = lower decoherence cost).
    """
    dim = 2 ** n_sites
    
    H_int = build_interaction_hamiltonian(n_sites, links, env_coupling)
    
    # Transform both to new factorization
    H_transformed = U @ H @ U.conj().T
    H_int_transformed = U @ H_int @ U.conj().T
    
    # Test with computational basis states in transformed frame
    total_survival = 0.0
    n_tests = min(dim, 8)  # Test subset of basis states
    
    for k in range(n_tests):
        # k-th basis state
        psi = np.zeros(dim, dtype=complex)
        psi[k] = 1.0
        rho = np.outer(psi, psi.conj())
        
        purities = simulate_decoherence(rho, H_transformed, H_int_transformed, t_final)
        total_survival += purities[-1]
    
    return total_survival / n_tests


# ============================================================
# COMBINED COST FUNCTION
# ============================================================

def compute_J(params, n_sites, alpha, beta, env_coupling=0.1):
    """
    J(φ, L) = α * LocalityCost + β * DecoherenceCost
    
    params = [unitary_params..., link_params...]
    """
    dim = 2 ** n_sites
    n_U = n_unitary_params(dim)
    n_L = n_link_params(n_sites)
    
    U_params = params[:n_U]
    L_params = params[n_U:n_U + n_L]
    
    # Reconstruct unitary and links
    U = unitary_from_params(U_params, dim)
    link_strengths = link_strengths_from_params(L_params, n_sites)
    links = link_dict_from_strengths(link_strengths, n_sites)
    
    # Build Hamiltonian with current link structure
    H = build_heisenberg_hamiltonian(n_sites, links)
    
    # Compute costs
    L_cost = compute_locality_cost(H, U, n_sites)
    
    # Use coherence survival (higher = better, so we minimize 1 - survival)
    survival = compute_coherence_survival(U, H, n_sites, links, env_coupling=env_coupling)
    D_cost = 1.0 - survival  # Convert to cost (lower survival = higher cost)
    
    J = alpha * L_cost + beta * D_cost
    
    return J


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_factorization_and_links(
    n_sites: int,
    alpha: float,
    beta: float,
    env_coupling: float = 0.1,
    n_restarts: int = 5,
    maxiter: int = 200,
    seed: int = 42
):
    """
    Find optimal (factorization, links) minimizing J = αL + βD.
    """
    np.random.seed(seed)
    
    dim = 2 ** n_sites
    n_U = n_unitary_params(dim)
    n_L = n_link_params(n_sites)
    n_total = n_U + n_L
    
    print(f"Optimizing over {n_U} unitary params + {n_L} link params = {n_total} total")
    print(f"α = {alpha}, β = {beta}")
    
    best_result = None
    best_J = np.inf
    
    for restart in range(n_restarts):
        print(f"\nRestart {restart + 1}/{n_restarts}")
        
        # Random initialization
        U_init = np.random.randn(n_U) * 0.1
        L_init = np.random.randn(n_L) * 0.5  # Sigmoid will center around 1.0
        params_init = np.concatenate([U_init, L_init])
        
        # Callback to track progress
        iteration = [0]
        def callback(params):
            iteration[0] += 1
            if iteration[0] % 20 == 0:
                J = compute_J(params, n_sites, alpha, beta, env_coupling)
                print(f"  Iter {iteration[0]}: J = {J:.4f}")
        
        # Optimize
        result = minimize(
            compute_J,
            params_init,
            args=(n_sites, alpha, beta, env_coupling),
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': maxiter, 'disp': False}
        )
        
        if result.fun < best_J:
            best_J = result.fun
            best_result = result
            print(f"  New best: J = {best_J:.4f}")
    
    return best_result


def analyze_result(result, n_sites, alpha, beta, env_coupling=0.1):
    """Analyze the optimal factorization and link structure."""
    dim = 2 ** n_sites
    n_U = n_unitary_params(dim)
    n_L = n_link_params(n_sites)
    
    params = result.x
    U = unitary_from_params(params[:n_U], dim)
    link_strengths = link_strengths_from_params(params[n_U:n_U + n_L], n_sites)
    links = link_dict_from_strengths(link_strengths, n_sites)
    
    H = build_heisenberg_hamiltonian(n_sites, links)
    
    L_cost = compute_locality_cost(H, U, n_sites)
    survival = compute_coherence_survival(U, H, n_sites, links, env_coupling=env_coupling)
    D_cost = 1.0 - survival
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Total J = {result.fun:.4f}")
    print(f"  Locality cost (L):    {L_cost:.4f}  (weight α = {alpha})")
    print(f"  Decoherence cost (D): {D_cost:.4f}  (weight β = {beta})")
    print(f"  Coherence survival:   {survival:.4f}")
    
    print("\n" + "-" * 40)
    print("LINK STRUCTURE")
    print("-" * 40)
    
    sorted_links = sorted(links.items(), key=lambda x: x[1], reverse=True)
    
    total_strength = sum(links.values())
    
    for (i, j), strength in sorted_links:
        pct = 100 * strength / total_strength
        bar = '█' * int(strength * 10)
        print(f"  ({i},{j}): {strength:.3f} ({pct:5.1f}%)  {bar}")
    
    # Analyze topology
    strong_threshold = 0.5 * max(links.values())
    strong_links = [(i, j) for (i, j), s in links.items() if s > strong_threshold]
    
    print(f"\nStrong links (>{strong_threshold:.2f}): {strong_links}")
    print(f"Number of strong links: {len(strong_links)}")
    
    # Node degrees
    degree = {i: 0 for i in range(n_sites)}
    for (i, j), s in links.items():
        degree[i] += s
        degree[j] += s
    
    print("\nNode degrees (sum of link strengths):")
    for node, d in sorted(degree.items(), key=lambda x: x[1], reverse=True):
        print(f"  Node {node}: {d:.3f}")
    
    # Effective dimension: links per node / 2
    avg_degree = sum(degree.values()) / n_sites
    eff_dim = avg_degree / 2  # Each link contributes to 2 nodes
    print(f"\nEffective dimension (avg_degree/2): {eff_dim:.2f}")
    
    return {
        'U': U,
        'links': links,
        'L_cost': L_cost,
        'D_cost': D_cost,
        'survival': survival,
        'J': result.fun
    }


# ============================================================
# SWEEP OVER ALPHA/BETA
# ============================================================

def sweep_tradeoff(n_sites, alphas, betas, env_coupling=0.1, n_restarts=3, seed=42):
    """
    Sweep over different α/β values to map the tradeoff landscape.
    """
    results = {}
    
    for alpha in alphas:
        for beta in betas:
            print(f"\n{'='*60}")
            print(f"α = {alpha}, β = {beta}")
            print('='*60)
            
            result = optimize_factorization_and_links(
                n_sites, alpha, beta, env_coupling, n_restarts, maxiter=100, seed=seed
            )
            
            analysis = analyze_result(result, n_sites, alpha, beta, env_coupling)
            results[(alpha, beta)] = analysis
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimize factorization and link topology',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--n_sites', type=int, default=3,
                        help='Number of qubits')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for locality cost')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for decoherence cost')
    parser.add_argument('--env_coupling', type=float, default=0.2,
                        help='Environment coupling strength')
    parser.add_argument('--n_restarts', type=int, default=5,
                        help='Number of optimization restarts')
    parser.add_argument('--maxiter', type=int, default=200,
                        help='Max iterations per restart')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep over α/β values')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='tradeoff_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.sweep:
        # Sweep over tradeoff parameter
        alphas = [0.1, 0.5, 1.0, 2.0]
        betas = [0.1, 0.5, 1.0, 2.0]
        
        results = sweep_tradeoff(
            args.n_sites, alphas, betas, args.env_coupling,
            n_restarts=3, seed=args.seed
        )
        
        # Summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data for plotting
        L_costs = np.zeros((len(alphas), len(betas)))
        D_costs = np.zeros((len(alphas), len(betas)))
        n_strong = np.zeros((len(alphas), len(betas)))
        
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                r = results[(alpha, beta)]
                L_costs[i, j] = r['L_cost']
                D_costs[i, j] = r['D_cost']
                
                # Count strong links
                threshold = 0.5 * max(r['links'].values())
                n_strong[i, j] = sum(1 for s in r['links'].values() if s > threshold)
        
        # Plot heatmaps
        ax = axes[0, 0]
        im = ax.imshow(L_costs, cmap='viridis')
        ax.set_xticks(range(len(betas)))
        ax.set_yticks(range(len(alphas)))
        ax.set_xticklabels([f'{b:.1f}' for b in betas])
        ax.set_yticklabels([f'{a:.1f}' for a in alphas])
        ax.set_xlabel('β (decoherence weight)')
        ax.set_ylabel('α (locality weight)')
        ax.set_title('Locality Cost')
        fig.colorbar(im, ax=ax)
        
        ax = axes[0, 1]
        im = ax.imshow(D_costs, cmap='viridis')
        ax.set_xticks(range(len(betas)))
        ax.set_yticks(range(len(alphas)))
        ax.set_xticklabels([f'{b:.1f}' for b in betas])
        ax.set_yticklabels([f'{a:.1f}' for a in alphas])
        ax.set_xlabel('β (decoherence weight)')
        ax.set_ylabel('α (locality weight)')
        ax.set_title('Decoherence Cost')
        fig.colorbar(im, ax=ax)
        
        ax = axes[1, 0]
        im = ax.imshow(n_strong, cmap='viridis')
        ax.set_xticks(range(len(betas)))
        ax.set_yticks(range(len(alphas)))
        ax.set_xticklabels([f'{b:.1f}' for b in betas])
        ax.set_yticklabels([f'{a:.1f}' for a in alphas])
        ax.set_xlabel('β (decoherence weight)')
        ax.set_ylabel('α (locality weight)')
        ax.set_title('Number of Strong Links')
        fig.colorbar(im, ax=ax)
        
        # Pareto frontier
        ax = axes[1, 1]
        for (alpha, beta), r in results.items():
            ax.scatter(r['L_cost'], r['D_cost'], s=100, 
                      label=f'α={alpha},β={beta}')
        ax.set_xlabel('Locality Cost')
        ax.set_ylabel('Decoherence Cost')
        ax.set_title('Tradeoff Frontier')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(f'{args.output_dir}/tradeoff_sweep.png', dpi=150)
        plt.close()
        
    else:
        # Single optimization
        result = optimize_factorization_and_links(
            args.n_sites,
            args.alpha,
            args.beta,
            args.env_coupling,
            args.n_restarts,
            args.maxiter,
            args.seed
        )
        
        analysis = analyze_result(result, args.n_sites, args.alpha, args.beta, args.env_coupling)
        
        # Visualize link structure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        n = args.n_sites
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}
        
        # Draw edges
        max_s = max(analysis['links'].values())
        for (i, j), s in analysis['links'].items():
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            width = 1 + 5 * s / max_s
            alpha_val = 0.2 + 0.8 * s / max_s
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=width, alpha=alpha_val)
        
        # Draw nodes
        for i, (x, y) in pos.items():
            ax.scatter(x, y, s=400, c='lightblue', edgecolors='black', zorder=3)
            ax.text(x, y, str(i), ha='center', va='center', fontsize=12, zorder=4)
        
        ax.set_title(f'Optimal Link Structure\nα={args.alpha}, β={args.beta}, J={analysis["J"]:.3f}')
        ax.set_aspect('equal')
        ax.axis('off')
        
        fig.tight_layout()
        fig.savefig(f'{args.output_dir}/optimal_links.png', dpi=150)
        plt.close()
        
        # Save numerical results
        np.savez(f'{args.output_dir}/results.npz',
                 links=analysis['links'],
                 L_cost=analysis['L_cost'],
                 D_cost=analysis['D_cost'],
                 J=analysis['J'],
                 alpha=args.alpha,
                 beta=args.beta)
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()