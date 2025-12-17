"""
Hilbert Substrate: Lieb-Robinson Light Cone Emergence
======================================================

Starting point: The December 2025 paper.
- Hilbert space (no background)
- Local unitaries (no-signaling)  
- Global unitarity (no-forgetting)

Question: Does spatial structure emerge from these constraints?

Setup:
------
1. N qubits as abstract tensor factors (not "in space")
2. An interaction graph defining which qubits couple
3. Initial state: maximally entangled (Haar random)
4. Dynamics: local unitaries on graph edges
5. Observable: how correlations spread, bounded by Lieb-Robinson

The light cone isn't assumed — it emerges from the locality of interactions.
"Space" is whatever effective geometry the Lieb-Robinson bound implies.

We track:
- Mutual information between all pairs
- Local perturbation spreading
- Emergence of locality in correlation structure
- Pointer state formation

If this works, we'll see:
- Correlations spreading at bounded velocity
- An emergent notion of "near" and "far"
- Classical (pointer) structure respecting that geometry
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from itertools import combinations


# =============================================================================
# State utilities
# =============================================================================

def haar_random_state(n_qubits):
    """
    Generate a Haar-random state on n qubits.
    This is (approximately) maximally entangled across any bipartition.
    """
    dim = 2 ** n_qubits
    # Random complex vector
    real = np.random.randn(dim)
    imag = np.random.randn(dim)
    psi = real + 1j * imag
    psi /= np.linalg.norm(psi)
    return psi


def partial_trace(rho, keep, dims):
    """
    Partial trace of density matrix rho.
    
    keep: list of qubit indices to keep
    dims: list of dimensions for each qubit (all 2 for qubits)
    
    Returns reduced density matrix on the kept subsystem.
    """
    n = len(dims)
    keep = sorted(keep)
    trace_out = [i for i in range(n) if i not in keep]
    
    # Reshape rho into tensor with one index per qubit (twice: row and col)
    rho_tensor = rho.reshape(dims + dims)
    
    # Trace out the unwanted qubits
    # We need to trace pairs of indices (i, i+n) for i in trace_out
    # Do this iteratively from the end to preserve indexing
    for i in sorted(trace_out, reverse=True):
        # Trace index i with index i+n (accounting for current shape)
        n_current = rho_tensor.ndim // 2
        rho_tensor = np.trace(rho_tensor, axis1=i, axis2=i + n_current)
    
    # Reshape back to matrix
    kept_dim = 2 ** len(keep)
    return rho_tensor.reshape(kept_dim, kept_dim)


def reduced_density_matrix(psi, keep, n_qubits):
    """
    Get reduced density matrix for qubits in 'keep' list.
    """
    dims = [2] * n_qubits
    rho = np.outer(psi, psi.conj())
    return partial_trace(rho, keep, dims)


def von_neumann_entropy(rho):
    """Von Neumann entropy of density matrix."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return -np.sum(eigenvalues * np.log2(eigenvalues))


def mutual_information(psi, i, j, n_qubits):
    """
    Mutual information I(i:j) = S(i) + S(j) - S(i,j)
    """
    rho_i = reduced_density_matrix(psi, [i], n_qubits)
    rho_j = reduced_density_matrix(psi, [j], n_qubits)
    rho_ij = reduced_density_matrix(psi, [i, j], n_qubits)
    
    S_i = von_neumann_entropy(rho_i)
    S_j = von_neumann_entropy(rho_j)
    S_ij = von_neumann_entropy(rho_ij)
    
    return S_i + S_j - S_ij


def purity(rho):
    """Purity Tr(rho^2). 1 for pure, 1/d for maximally mixed."""
    return np.real(np.trace(rho @ rho))


# =============================================================================
# Local unitaries
# =============================================================================

def random_su4():
    """Random SU(4) matrix (for 2-qubit gates)."""
    # Generate random Hermitian and exponentiate
    H = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    H = H + H.conj().T  # Hermitian
    U = expm(1j * H)
    return U


def random_su2():
    """Random SU(2) matrix (for 1-qubit gates)."""
    H = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    H = H + H.conj().T
    U = expm(1j * H)
    return U


def apply_two_qubit_gate(psi, U, i, j, n_qubits):
    """
    Apply 2-qubit gate U to qubits i and j.
    """
    dim = 2 ** n_qubits
    psi_tensor = psi.reshape([2] * n_qubits)
    
    # Move qubits i and j to the front
    axes = list(range(n_qubits))
    axes.remove(i)
    axes.remove(j)
    axes = [i, j] + axes
    psi_tensor = np.transpose(psi_tensor, axes)
    
    # Reshape to (4, rest)
    psi_matrix = psi_tensor.reshape(4, -1)
    
    # Apply gate
    psi_matrix = U @ psi_matrix
    
    # Reshape back
    psi_tensor = psi_matrix.reshape([2, 2] + [2] * (n_qubits - 2))
    
    # Transpose back
    inv_axes = [0] * n_qubits
    for new_pos, old_pos in enumerate(axes):
        inv_axes[old_pos] = new_pos
    psi_tensor = np.transpose(psi_tensor, inv_axes)
    
    return psi_tensor.reshape(dim)


def apply_one_qubit_gate(psi, U, i, n_qubits):
    """
    Apply 1-qubit gate U to qubit i.
    """
    dim = 2 ** n_qubits
    psi_tensor = psi.reshape([2] * n_qubits)
    
    # Apply U along axis i
    psi_tensor = np.tensordot(U, psi_tensor, axes=([1], [i]))
    
    # Move the result axis back to position i
    psi_tensor = np.moveaxis(psi_tensor, 0, i)
    
    return psi_tensor.reshape(dim)


# =============================================================================
# Interaction graph
# =============================================================================

def make_1d_chain_graph(n_qubits, periodic=False):
    """
    1D chain: qubit i couples to i+1.
    """
    edges = [(i, i+1) for i in range(n_qubits - 1)]
    if periodic and n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    return edges


def make_2d_lattice_graph(Lx, Ly):
    """
    2D square lattice: qubit (x,y) couples to (x+1,y) and (x,y+1).
    """
    def idx(x, y):
        return x * Ly + y
    
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y)
            if x + 1 < Lx:
                edges.append((i, idx(x+1, y)))
            if y + 1 < Ly:
                edges.append((i, idx(x, y+1)))
    return edges


def graph_distance(edges, n_qubits):
    """
    Compute shortest path distance between all pairs on the graph.
    Returns distance matrix.
    """
    # Build adjacency
    adj = {i: set() for i in range(n_qubits)}
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # BFS from each node
    dist = np.full((n_qubits, n_qubits), np.inf)
    for start in range(n_qubits):
        dist[start, start] = 0
        queue = [start]
        visited = {start}
        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist[start, neighbor] = dist[start, current] + 1
                    queue.append(neighbor)
    
    return dist


# =============================================================================
# Lieb-Robinson diagnostics
# =============================================================================

def local_perturbation(psi, site, n_qubits, strength=0.1):
    """
    Apply a small local perturbation at 'site'.
    Returns the perturbed state.
    """
    # Apply a small rotation (Pauli-X type)
    theta = strength
    U = np.array([
        [np.cos(theta), -1j * np.sin(theta)],
        [-1j * np.sin(theta), np.cos(theta)]
    ])
    return apply_one_qubit_gate(psi, U, site, n_qubits)


def measure_local_observable(psi, site, n_qubits):
    """
    Measure <Z> at a site. Returns expectation value.
    """
    rho = reduced_density_matrix(psi, [site], n_qubits)
    Z = np.array([[1, 0], [0, -1]])
    return np.real(np.trace(rho @ Z))


def track_perturbation_spreading(
    psi_unperturbed,
    psi_perturbed,
    n_qubits,
    edges,
    n_steps,
    gates_per_step=1,
):
    """
    Evolve both states and track how the local difference spreads.
    
    Returns:
        times: step indices
        differences: array of shape (n_steps+1, n_qubits)
                     showing |<Z>_perturbed - <Z>_unperturbed| at each site
    """
    psi_u = psi_unperturbed.copy()
    psi_p = psi_perturbed.copy()
    
    # Initial differences
    differences = []
    diff_0 = np.array([
        abs(measure_local_observable(psi_p, i, n_qubits) - 
            measure_local_observable(psi_u, i, n_qubits))
        for i in range(n_qubits)
    ])
    differences.append(diff_0)
    
    # Evolve
    for step in range(n_steps):
        # Apply random local gates on all edges
        for _ in range(gates_per_step):
            for (i, j) in edges:
                U = random_su4()
                psi_u = apply_two_qubit_gate(psi_u, U, i, j, n_qubits)
                psi_p = apply_two_qubit_gate(psi_p, U, i, j, n_qubits)
        
        # Measure differences
        diff = np.array([
            abs(measure_local_observable(psi_p, i, n_qubits) - 
                measure_local_observable(psi_u, i, n_qubits))
            for i in range(n_qubits)
        ])
        differences.append(diff)
    
    return np.arange(n_steps + 1), np.array(differences)


# =============================================================================
# Main experiment
# =============================================================================

def run_lieb_robinson_experiment(
    n_qubits=8,
    graph_type='1d_chain',
    n_steps=20,
    perturbation_site=None,
    plot=True,
):
    """
    Run the Lieb-Robinson light cone emergence experiment.
    
    1. Create a maximally entangled (Haar random) initial state
    2. Define an interaction graph
    3. Perturb one site
    4. Evolve and watch the perturbation spread
    5. Check if spreading respects Lieb-Robinson bound
    """
    
    print("=" * 60)
    print("Lieb-Robinson Light Cone Emergence")
    print("=" * 60)
    print(f"Qubits: {n_qubits}")
    print(f"Graph: {graph_type}")
    print(f"Steps: {n_steps}")
    
    # Build graph
    if graph_type == '1d_chain':
        edges = make_1d_chain_graph(n_qubits, periodic=False)
    elif graph_type == '1d_periodic':
        edges = make_1d_chain_graph(n_qubits, periodic=True)
    elif graph_type.startswith('2d_'):
        # e.g., '2d_3x3'
        dims = graph_type.split('_')[1].split('x')
        Lx, Ly = int(dims[0]), int(dims[1])
        edges = make_2d_lattice_graph(Lx, Ly)
        n_qubits = Lx * Ly
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    print(f"Edges: {len(edges)}")
    
    # Graph distances
    dist_matrix = graph_distance(edges, n_qubits)
    
    # Perturbation site (default: middle)
    if perturbation_site is None:
        perturbation_site = n_qubits // 2
    print(f"Perturbation site: {perturbation_site}")
    
    # Initial state: Haar random (maximally entangled)
    print("\nGenerating Haar-random initial state...")
    psi_0 = haar_random_state(n_qubits)
    
    # Check entanglement
    print("Initial entanglement (single-qubit entropies):")
    for i in range(min(n_qubits, 4)):
        rho_i = reduced_density_matrix(psi_0, [i], n_qubits)
        S = von_neumann_entropy(rho_i)
        print(f"  S(qubit {i}) = {S:.4f} (max = 1.0)")
    
    # Perturb
    print(f"\nApplying local perturbation at site {perturbation_site}...")
    psi_perturbed = local_perturbation(psi_0, perturbation_site, n_qubits, strength=0.3)
    
    # Track spreading
    print("Evolving and tracking perturbation spread...")
    times, differences = track_perturbation_spreading(
        psi_0, psi_perturbed, n_qubits, edges, n_steps, gates_per_step=1
    )
    
    # Analyze: group by distance from perturbation site
    distances = dist_matrix[perturbation_site, :]
    unique_distances = sorted(set(distances[distances < np.inf]))
    
    print("\n" + "-" * 60)
    print("RESULTS: Perturbation spreading by graph distance")
    print("-" * 60)
    
    distance_vs_time = {}
    for d in unique_distances:
        sites_at_d = [i for i in range(n_qubits) if dist_matrix[perturbation_site, i] == d]
        # Average difference at this distance over time
        avg_diff = np.mean(differences[:, sites_at_d], axis=1)
        distance_vs_time[d] = avg_diff
        print(f"Distance {int(d)}: sites {sites_at_d}")
        print(f"  Initial diff: {avg_diff[0]:.6f}")
        print(f"  Final diff:   {avg_diff[-1]:.6f}")
    
    # Lieb-Robinson check: at time t, only distances <= v*t should show signal
    print("\n" + "-" * 60)
    print("LIEB-ROBINSON CHECK")
    print("-" * 60)
    print("If LR bound holds, perturbation should not reach distance d")
    print("until time t ~ d/v for some velocity v.")
    print()
    
    # Estimate when each distance first sees significant signal
    threshold = 0.01
    arrival_times = {}
    for d in unique_distances:
        if d == 0:
            arrival_times[d] = 0
            continue
        for t in range(len(times)):
            if distance_vs_time[d][t] > threshold:
                arrival_times[d] = t
                break
        else:
            arrival_times[d] = np.inf
    
    print(f"Arrival times (threshold = {threshold}):")
    for d in sorted(arrival_times.keys()):
        t = arrival_times[d]
        if t < np.inf:
            print(f"  Distance {int(d)}: arrives at step {t}")
        else:
            print(f"  Distance {int(d)}: not arrived")
    
    # Check linearity (LR implies t ~ d)
    finite_arrivals = [(d, t) for d, t in arrival_times.items() if t < np.inf and d > 0]
    if len(finite_arrivals) >= 2:
        ds, ts = zip(*finite_arrivals)
        correlation = np.corrcoef(ds, ts)[0, 1]
        print(f"\nCorrelation(distance, arrival_time) = {correlation:.4f}")
        if correlation > 0.8:
            print("✓ Strong positive correlation: consistent with Lieb-Robinson bound")
        else:
            print("? Weak correlation: light cone structure unclear")
    
    print("=" * 60)
    
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Heatmap of differences over time
        ax = axes[0]
        im = ax.imshow(differences.T, aspect='auto', origin='lower',
                       extent=[0, n_steps, -0.5, n_qubits-0.5])
        ax.set_xlabel('Time step')
        ax.set_ylabel('Qubit')
        ax.set_title('Perturbation spreading')
        ax.axhline(perturbation_site, color='red', linestyle='--', alpha=0.5)
        plt.colorbar(im, ax=ax, label='|ΔZ|')
        
        # 2. Difference vs distance at different times
        ax = axes[1]
        for t in [0, n_steps//4, n_steps//2, n_steps]:
            diffs_at_t = []
            for d in unique_distances:
                sites_at_d = [i for i in range(n_qubits) if dist_matrix[perturbation_site, i] == d]
                diffs_at_t.append(np.mean(differences[t, sites_at_d]))
            ax.plot(unique_distances, diffs_at_t, 'o-', label=f't={t}')
        ax.set_xlabel('Graph distance from perturbation')
        ax.set_ylabel('Average |ΔZ|')
        ax.set_title('Signal vs distance at different times')
        ax.legend()
        
        # 3. Arrival time vs distance
        ax = axes[2]
        finite_d = [d for d, t in arrival_times.items() if t < np.inf]
        finite_t = [arrival_times[d] for d in finite_d]
        ax.plot(finite_d, finite_t, 'ko-')
        ax.set_xlabel('Graph distance')
        ax.set_ylabel('Arrival time (steps)')
        ax.set_title('Light cone: arrival time vs distance')
        
        # Linear fit
        if len(finite_d) >= 2:
            z = np.polyfit(finite_d, finite_t, 1)
            p = np.poly1d(z)
            d_range = np.linspace(min(finite_d), max(finite_d), 50)
            ax.plot(d_range, p(d_range), 'r--', alpha=0.5, label=f'v ≈ {1/z[0]:.2f} sites/step')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('lieb_robinson_lightcone.png', dpi=150)
        print("\nSaved: lieb_robinson_lightcone.png")
        plt.show()
    
    return times, differences, distance_vs_time, arrival_times


# =============================================================================
# Additional: Mutual information structure
# =============================================================================

def analyze_mutual_information_structure(
    n_qubits=8,
    graph_type='1d_chain',
    n_steps=50,
    plot=True,
):
    """
    Track how mutual information structure evolves.
    Does MI become localized (respecting the graph) over time?
    """
    
    print("=" * 60)
    print("Mutual Information Structure Evolution")
    print("=" * 60)
    
    if graph_type == '1d_chain':
        edges = make_1d_chain_graph(n_qubits, periodic=False)
    elif graph_type == '1d_periodic':
        edges = make_1d_chain_graph(n_qubits, periodic=True)
    else:
        edges = make_1d_chain_graph(n_qubits)
    
    dist_matrix = graph_distance(edges, n_qubits)
    
    # Initial Haar-random state
    psi = haar_random_state(n_qubits)
    
    # Compute initial MI matrix
    def compute_mi_matrix(psi):
        MI = np.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                MI[i, j] = mutual_information(psi, i, j, n_qubits)
                MI[j, i] = MI[i, j]
        return MI
    
    print("Computing initial mutual information...")
    MI_initial = compute_mi_matrix(psi)
    
    # Evolve
    print(f"Evolving for {n_steps} steps...")
    for step in range(n_steps):
        for (i, j) in edges:
            U = random_su4()
            psi = apply_two_qubit_gate(psi, U, i, j, n_qubits)
    
    print("Computing final mutual information...")
    MI_final = compute_mi_matrix(psi)
    
    # Analyze: correlation between MI and graph distance
    pairs = [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits)]
    distances = [dist_matrix[i, j] for (i, j) in pairs]
    mi_initial = [MI_initial[i, j] for (i, j) in pairs]
    mi_final = [MI_final[i, j] for (i, j) in pairs]
    
    corr_initial = np.corrcoef(distances, mi_initial)[0, 1]
    corr_final = np.corrcoef(distances, mi_final)[0, 1]
    
    print(f"\nCorrelation(distance, MI):")
    print(f"  Initial: {corr_initial:.4f}")
    print(f"  Final:   {corr_final:.4f}")
    
    if corr_final < corr_initial:
        print("✓ MI becoming more negatively correlated with distance")
        print("  -> Locality emerging")
    else:
        print("? MI not becoming more local")
    
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        axes[0].imshow(MI_initial, cmap='hot')
        axes[0].set_title('Initial MI matrix')
        axes[0].set_xlabel('Qubit')
        axes[0].set_ylabel('Qubit')
        
        axes[1].imshow(MI_final, cmap='hot')
        axes[1].set_title('Final MI matrix')
        axes[1].set_xlabel('Qubit')
        axes[1].set_ylabel('Qubit')
        
        axes[2].scatter(distances, mi_initial, alpha=0.5, label='Initial')
        axes[2].scatter(distances, mi_final, alpha=0.5, label='Final')
        axes[2].set_xlabel('Graph distance')
        axes[2].set_ylabel('Mutual information')
        axes[2].set_title('MI vs distance')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('mutual_information_structure.png', dpi=150)
        print("\nSaved: mutual_information_structure.png")
        plt.show()
    
    return MI_initial, MI_final


if __name__ == "__main__":
    # Run both experiments
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Lieb-Robinson Light Cone")
    print("=" * 70 + "\n")
    
    run_lieb_robinson_experiment(
        n_qubits=10,
        graph_type='1d_chain',
        n_steps=15,
        plot=True,
    )
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Mutual Information Locality")
    print("=" * 70 + "\n")
    
    analyze_mutual_information_structure(
        n_qubits=8,
        graph_type='1d_chain',
        n_steps=50,
        plot=True,
    )