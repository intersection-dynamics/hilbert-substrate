"""
pointer_distance_decoherence.py

THE REAL TEST

The previous experiment was flawed: we varied the Hamiltonian basis
but the environment always coupled via ZZ in the computational basis.

The RIGHT question: Does distance from the POINTER BASIS predict decoherence rate?

The pointer basis is determined by H_int (the system-environment coupling).
For H_int = Σ Z_i ⊗ Z_i, the pointer basis is the computational (Z) basis.

So we should test:
- States aligned with pointer basis → slow decoherence
- States orthogonal to pointer basis → fast decoherence
- Measure the angle/distance and correlate with decoherence rate

This is the actual prediction: τ_decoherence = f(distance to pointer basis)
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# =============================================================================
# Basic tools
# =============================================================================

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(*args):
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result

def partial_trace(rho, dims, keep):
    """Partial trace using einsum."""
    n = len(dims)
    keep = sorted(keep)
    trace_out = [i for i in range(n) if i not in keep]
    
    if not trace_out:
        return rho.copy()
    
    shape = list(dims) + list(dims)
    rho_t = rho.reshape(shape)
    
    chars = 'abcdefghijklmnopqrstuvwxyz'
    in_idx = list(range(2*n))
    for t in trace_out:
        in_idx[n + t] = t
    
    out_idx = [k for k in keep] + [n + k for k in keep]
    
    in_str = ''.join(chars[i] for i in in_idx)
    out_str = ''.join(chars[i] for i in out_idx)
    
    result = np.einsum(f"{in_str}->{out_str}", rho_t)
    d_keep = int(np.prod([dims[k] for k in keep]))
    return result.reshape(d_keep, d_keep)

def purity(rho):
    return np.real(np.trace(rho @ rho))

# =============================================================================
# Pointer basis distance measures
# =============================================================================

def coherence_in_basis(state, basis='computational'):
    """
    Measure how much coherence (off-diagonal) the state has in a given basis.
    
    For computational basis: |ρ_ij| for i≠j
    This is exactly what decoherence destroys.
    """
    rho = np.outer(state, state.conj())
    
    if basis == 'computational':
        # Already in computational basis
        pass
    elif basis == 'X':
        # Transform to X eigenbasis
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard
        dim = len(state)
        n = int(np.log2(dim))
        U = H
        for _ in range(n-1):
            U = np.kron(U, H)
        rho = U @ rho @ U.conj().T
    
    # Sum of off-diagonal magnitudes
    coherence = np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))
    return coherence

def pointer_basis_distance(state):
    """
    Distance from computational (pointer) basis.
    
    A state |ψ⟩ = Σ c_i |i⟩ has:
    - Distance 0 if it's a computational basis state (one c_i = 1)
    - Distance max if it's uniform superposition
    
    We use the coherence (off-diagonal magnitude) as the measure.
    Normalized by maximum possible coherence.
    """
    dim = len(state)
    rho = np.outer(state, state.conj())
    
    # Off-diagonal L1 norm
    off_diag = np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))
    
    # Maximum coherence for a pure state is achieved by uniform superposition
    # |+⟩^n has ρ_ij = 1/dim for all i,j
    # Off-diagonal sum = dim^2 - dim terms, each = 1/dim, total = (dim^2-dim)/dim = dim-1
    max_coherence = dim - 1
    
    return off_diag / max_coherence if max_coherence > 0 else 0

def participation_ratio(state):
    """
    Inverse participation ratio: 1/Σ|c_i|^4
    = 1 for localized state, dim for uniform superposition
    """
    probs = np.abs(state)**2
    return 1.0 / np.sum(probs**2)

# =============================================================================
# Decoherence dynamics
# =============================================================================

def measure_decoherence_time(state, n_sys, n_env, J_int=2.0, t_max=3.0, n_steps=100):
    """
    Measure time for purity to decay to (1 + P_min)/2.
    Returns decoherence time τ and the full purity curve.
    """
    n_total = n_sys + n_env
    dims = [2] * n_total
    dim_sys = 2**n_sys
    dim_env = 2**n_env
    
    # Environment ground state
    env = np.zeros(dim_env, dtype=complex)
    env[0] = 1.0
    
    # Total initial state
    psi = np.kron(state, env)
    
    # Interaction Hamiltonian: CONDITIONAL coupling (like before)
    # |1><1| ⊗ X creates real decoherence
    H_int = np.zeros((2**n_total, 2**n_total), dtype=complex)
    
    # System qubit 0 conditionally flips environment qubits
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|
    for i in range(min(n_sys, n_env)):
        ops = [I] * n_total
        ops[i] = P1  # Project system qubit to |1>
        ops[n_sys + i] = X  # Flip environment qubit
        H_int += J_int * tensor(*ops)
    
    # Add environment spreading (XX + YY)
    H_env = np.zeros((2**n_total, 2**n_total), dtype=complex)
    for i in range(n_env - 1):
        for P in [X, Y]:
            ops = [I] * n_total
            ops[n_sys + i] = P
            ops[n_sys + i + 1] = P
            H_env += 0.5 * tensor(*ops)
    
    # Small system Hamiltonian (so pointer basis is truly selected by H_int)
    H_sys = np.zeros((2**n_total, 2**n_total), dtype=complex)
    for i in range(n_sys):
        ops = [I] * n_total
        ops[i] = X
        H_sys += 0.1 * tensor(*ops)
    
    H_total = H_sys + H_env + H_int
    
    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0]
    U = expm(-1j * H_total * dt)
    
    purities = []
    
    for t in times:
        rho_total = np.outer(psi, psi.conj())
        rho_sys = partial_trace(rho_total, dims, list(range(n_sys)))
        purities.append(purity(rho_sys))
        psi = U @ psi
    
    purities = np.array(purities)
    
    # Find decoherence time
    P_init = purities[0]
    P_min = 1.0 / dim_sys  # Maximally mixed
    P_target = (P_init + P_min) / 2
    
    tau = t_max  # Default if doesn't decay enough
    for i, P in enumerate(purities):
        if P < P_target:
            tau = times[i]
            break
    
    return tau, purities, times

# =============================================================================
# Main experiment
# =============================================================================

def run_pointer_distance_experiment(n_sys=2, n_env=4, n_states=60):
    """
    The real test: distance from pointer basis vs decoherence time.
    """
    print("="*60)
    print("POINTER BASIS DISTANCE vs DECOHERENCE TIME")
    print("="*60)
    print(f"System qubits: {n_sys}")
    print(f"Environment qubits: {n_env}")
    print(f"Number of test states: {n_states}")
    print(f"Pointer basis: computational (Z eigenstates)")
    
    dim_sys = 2**n_sys
    np.random.seed(42)
    
    results = []
    
    print("\nGenerating and testing states...")
    
    for i in range(n_states):
        # Generate states with varying distance from pointer basis
        
        if i < n_states // 4:
            # Computational basis states (in pointer basis)
            state = np.zeros(dim_sys, dtype=complex)
            state[i % dim_sys] = 1.0
            state_type = 'pointer'
            
        elif i < n_states // 2:
            # States slightly rotated from computational basis
            angle = np.pi * (i - n_states//4) / (n_states//4) * 0.3  # 0 to 0.3π
            state = np.zeros(dim_sys, dtype=complex)
            state[0] = np.cos(angle)
            state[1] = np.sin(angle)
            state_type = 'near_pointer'
            
        elif i < 3 * n_states // 4:
            # Superposition states (far from pointer basis)
            # |+⟩ type states
            k = i - n_states // 2
            n_terms = 2 + k % (dim_sys - 1)  # 2 to dim_sys terms
            state = np.zeros(dim_sys, dtype=complex)
            indices = np.random.choice(dim_sys, n_terms, replace=False)
            phases = np.exp(2j * np.pi * np.random.rand(n_terms))
            state[indices] = phases / np.sqrt(n_terms)
            state_type = 'superposition'
            
        else:
            # Uniform superposition (maximally far from pointer)
            state = np.ones(dim_sys, dtype=complex) / np.sqrt(dim_sys)
            # Add random phases
            phases = np.exp(2j * np.pi * np.random.rand(dim_sys))
            state = state * phases
            state = state / np.linalg.norm(state)
            state_type = 'uniform'
        
        # Measure distance from pointer basis
        distance = pointer_basis_distance(state)
        pr = participation_ratio(state)
        coherence = coherence_in_basis(state, 'computational')
        
        # Measure decoherence time
        tau, purities, times = measure_decoherence_time(state, n_sys, n_env)
        
        # Decoherence rate = 1/τ
        gamma = 1.0 / tau if tau > 0.01 else 100.0
        
        results.append({
            'state_type': state_type,
            'distance': distance,
            'participation_ratio': pr,
            'coherence': coherence,
            'tau': tau,
            'gamma': gamma,
            'final_purity': purities[-1],
        })
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{n_states}")
    
    print("Done!\n")
    
    return results

def analyze_and_plot(results, save_prefix='pointer_distance'):
    """Analyze results and create plots."""
    
    # Extract data
    distances = np.array([r['distance'] for r in results])
    PRs = np.array([r['participation_ratio'] for r in results])
    coherences = np.array([r['coherence'] for r in results])
    taus = np.array([r['tau'] for r in results])
    gammas = np.array([r['gamma'] for r in results])
    types = [r['state_type'] for r in results]
    
    # Color by type
    type_colors = {
        'pointer': 'green',
        'near_pointer': 'blue', 
        'superposition': 'orange',
        'uniform': 'red'
    }
    colors = [type_colors[t] for t in types]
    
    print("="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    # Correlations
    corr_dist_gamma = np.corrcoef(distances, gammas)[0, 1]
    corr_pr_gamma = np.corrcoef(PRs, gammas)[0, 1]
    corr_coh_gamma = np.corrcoef(coherences, gammas)[0, 1]
    
    print(f"\nCorrelations with decoherence rate γ:")
    print(f"  Pointer distance:     r = {corr_dist_gamma:.4f}")
    print(f"  Participation ratio:  r = {corr_pr_gamma:.4f}")
    print(f"  Coherence:            r = {corr_coh_gamma:.4f}")
    
    # Correlations with decoherence TIME (inverse of rate)
    corr_dist_tau = np.corrcoef(distances, taus)[0, 1]
    corr_coh_tau = np.corrcoef(coherences, taus)[0, 1]
    
    print(f"\nCorrelations with decoherence time τ:")
    print(f"  Pointer distance:     r = {corr_dist_tau:.4f}")
    print(f"  Coherence:            r = {corr_coh_tau:.4f}")
    
    # Summary by type
    print(f"\nBy state type:")
    for t in ['pointer', 'near_pointer', 'superposition', 'uniform']:
        subset = [r for r in results if r['state_type'] == t]
        if subset:
            avg_tau = np.mean([r['tau'] for r in subset])
            avg_dist = np.mean([r['distance'] for r in subset])
            print(f"  {t:15s}: avg τ = {avg_tau:.3f}, avg distance = {avg_dist:.3f}")
    
    # Try fits
    from scipy.optimize import curve_fit
    
    # Linear: τ = a - b*distance
    try:
        def linear(x, a, b):
            return a - b * x
        mask = np.isfinite(taus) & np.isfinite(distances)
        popt, _ = curve_fit(linear, distances[mask], taus[mask], p0=[5, 5], maxfev=5000)
        pred = linear(distances[mask], *popt)
        r2 = 1 - np.sum((taus[mask] - pred)**2) / np.sum((taus[mask] - np.mean(taus[mask]))**2)
        print(f"\nLinear fit: τ = {popt[0]:.3f} - {popt[1]:.3f}*d")
        print(f"  R² = {r2:.4f}")
    except Exception as e:
        print(f"Linear fit failed: {e}")
        popt = None
        r2 = 0
    
    # Exponential: τ = a * exp(-b*distance)
    try:
        def exponential(x, a, b):
            return a * np.exp(-b * x)
        popt_exp, _ = curve_fit(exponential, distances[mask], taus[mask], p0=[5, 2], maxfev=5000)
        pred_exp = exponential(distances[mask], *popt_exp)
        r2_exp = 1 - np.sum((taus[mask] - pred_exp)**2) / np.sum((taus[mask] - np.mean(taus[mask]))**2)
        print(f"\nExponential fit: τ = {popt_exp[0]:.3f} * exp(-{popt_exp[1]:.3f}*d)")
        print(f"  R² = {r2_exp:.4f}")
    except Exception as e:
        print(f"Exponential fit failed: {e}")
        popt_exp = None
        r2_exp = 0
    
    # Inverse: γ = a + b*distance (rate increases with distance)
    try:
        def linear_gamma(x, a, b):
            return a + b * x
        popt_g, _ = curve_fit(linear_gamma, distances[mask], gammas[mask], p0=[0.5, 1], maxfev=5000)
        pred_g = linear_gamma(distances[mask], *popt_g)
        r2_g = 1 - np.sum((gammas[mask] - pred_g)**2) / np.sum((gammas[mask] - np.mean(gammas[mask]))**2)
        print(f"\nLinear γ fit: γ = {popt_g[0]:.3f} + {popt_g[1]:.3f}*d")
        print(f"  R² = {r2_g:.4f}")
    except Exception as e:
        print(f"Linear γ fit failed: {e}")
        popt_g = None
        r2_g = 0
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distance vs decoherence time
    ax = axes[0, 0]
    ax.scatter(distances, taus, c=colors, alpha=0.7, s=60)
    if popt is not None:
        d_sorted = np.sort(distances)
        ax.plot(d_sorted, linear(d_sorted, *popt), 'k--', linewidth=2, 
                label=f'Linear fit (R²={r2:.3f})')
    ax.set_xlabel('Distance from Pointer Basis')
    ax.set_ylabel('Decoherence Time τ')
    ax.set_title(f'Distance vs τ (correlation: {corr_dist_tau:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distance vs decoherence rate
    ax = axes[0, 1]
    ax.scatter(distances, gammas, c=colors, alpha=0.7, s=60)
    if popt_g is not None:
        d_sorted = np.sort(distances)
        ax.plot(d_sorted, linear_gamma(d_sorted, *popt_g), 'k--', linewidth=2,
                label=f'Linear fit (R²={r2_g:.3f})')
    ax.set_xlabel('Distance from Pointer Basis')
    ax.set_ylabel('Decoherence Rate γ = 1/τ')
    ax.set_title(f'Distance vs γ (correlation: {corr_dist_gamma:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Coherence vs decoherence time
    ax = axes[1, 0]
    ax.scatter(coherences, taus, c=colors, alpha=0.7, s=60)
    ax.set_xlabel('Coherence (off-diagonal magnitude)')
    ax.set_ylabel('Decoherence Time τ')
    ax.set_title(f'Coherence vs τ (correlation: {corr_coh_tau:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 4. Legend and summary
    ax = axes[1, 1]
    for t, c in type_colors.items():
        ax.scatter([], [], c=c, s=100, label=t)
    ax.legend(loc='upper left', fontsize=12)
    
    # Add text summary
    summary = f"""
PREDICTION TEST RESULTS

Correlations with decoherence rate γ:
  • Pointer distance: r = {corr_dist_gamma:.3f}
  • Coherence: r = {corr_coh_gamma:.3f}

Best fit (if R² > 0.3):
  γ = {popt_g[0]:.2f} + {popt_g[1]:.2f} × distance
  R² = {r2_g:.3f}

Interpretation:
  {"PREDICTION CONFIRMED" if r2_g > 0.3 else "WEAK RELATIONSHIP"}
  {"States far from pointer basis decohere faster" if corr_dist_gamma > 0.3 else "Need better experiment design"}
"""
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_results.png', dpi=150)
    print(f"\nSaved: {save_prefix}_results.png")
    
    return {
        'corr_dist_gamma': corr_dist_gamma,
        'corr_coh_gamma': corr_coh_gamma,
        'r2_linear': r2_g if popt_g is not None else 0,
        'fit_params': popt_g
    }

# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("THE REAL PREDICTION TEST")
    print("#"*60)
    print("\nHypothesis: States far from the pointer basis decohere faster.")
    print("Pointer basis = eigenstates of system-environment interaction")
    print("For H_int = ZZ, pointer basis = computational basis")
    
    results = run_pointer_distance_experiment(n_sys=2, n_env=4, n_states=60)
    analysis = analyze_and_plot(results)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if analysis['r2_linear'] > 0.5:
        print("\n✓ STRONG PREDICTION: Decoherence rate scales with pointer distance")
        print(f"  γ ≈ {analysis['fit_params'][0]:.2f} + {analysis['fit_params'][1]:.2f} × d")
        print(f"  R² = {analysis['r2_linear']:.3f}")
        print("\nThis is a falsifiable, quantitative prediction!")
    elif analysis['r2_linear'] > 0.3:
        print("\n~ MODERATE PREDICTION: Weak relationship found")
        print(f"  R² = {analysis['r2_linear']:.3f}")
        print("\nNeed larger systems or better controls.")
    else:
        print("\n✗ NO CLEAR PREDICTION from this experiment")
        print(f"  R² = {analysis['r2_linear']:.3f}")
        print("\nPossible issues:")
        print("  - System too small")
        print("  - Environment coupling too weak/strong")
        print("  - Need different distance metric")