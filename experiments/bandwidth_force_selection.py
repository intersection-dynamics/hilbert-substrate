"""
Bandwidth-Constrained Force Selection
======================================

After accessibility collapse fixes spatial structure, test whether
the finite bandwidth constraint selects a preferred operator content
for the weight-2 (force) terms.

Approach:
  1. Fix a spatial lattice
  2. Parameterize weight-2 content: hopping (XX+YY), Ising (ZZ), cross (XY etc)
  3. For each composition, measure Lieb-Robinson velocity
  4. Map the velocity surface over operator space
  5. Find where bandwidth constraint carves out allowed region

Usage:
    python bandwidth_force_selection.py --quick       # Fast test
    python bandwidth_force_selection.py --standard    # Main sweep
    python bandwidth_force_selection.py --fine         # Fine-grained sweep
"""

import numpy as np
from scipy.linalg import expm
from itertools import product as iprod
import json
import time
import argparse


# ============================================================
# PAULI OPERATORS
# ============================================================

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
Sp = (X + 1j*Y) / 2
Sm = (X - 1j*Y) / 2


# ============================================================
# HAMILTONIAN CONSTRUCTION
# ============================================================

def build_parameterized_hamiltonian(N, alpha, beta, gamma, J=1.0, 
                                     topology='chain'):
    """
    Build a nearest-neighbor Hamiltonian with controlled operator content.
    
    Parameters:
        N: number of sites
        alpha: hopping fraction   (XX + YY channel)
        beta:  Ising fraction     (ZZ channel)
        gamma: cross fraction     (XY, XZ, YX, YZ, ZX, ZY channels)
        J: overall coupling strength
        topology: 'chain' (1D periodic) or 'lattice_2d' or 'lattice_3d'
    
    The total interaction strength is normalized so that the 
    Frobenius norm per bond is constant regardless of composition.
    This ensures we're comparing operator *type*, not operator *strength*.
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Get edges based on topology
    edges = get_edges(N, topology)
    
    # Normalize: each channel has a natural number of components
    # Hopping: 2 terms (XX, YY)
    # Ising: 1 term (ZZ)
    # Cross: 6 terms (XY, XZ, YX, YZ, ZX, ZY)
    # We weight so that alpha + beta + gamma = 1 controls the 
    # fraction of total squared coupling in each channel
    
    # Per-component coefficients (equal within each channel)
    # |J_hop|^2 * 2 = alpha, |J_ising|^2 * 1 = beta, |J_cross|^2 * 6 = gamma
    eps = 1e-12
    J_hop   = np.sqrt(max(alpha, 0) / 2 + eps) if alpha > 0 else 0
    J_ising = np.sqrt(max(beta, 0) / 1 + eps) if beta > 0 else 0
    J_cross = np.sqrt(max(gamma, 0) / 6 + eps) if gamma > 0 else 0
    
    for (i, j) in edges:
        # Hopping channel: XX + YY
        if alpha > 0:
            H += J * J_hop * make_2site_term(N, i, j, X, X)
            H += J * J_hop * make_2site_term(N, i, j, Y, Y)
        
        # Ising channel: ZZ
        if beta > 0:
            H += J * J_ising * make_2site_term(N, i, j, Z, Z)
        
        # Cross channel: XY, XZ, YX, YZ, ZX, ZY
        if gamma > 0:
            cross_pairs = [(X,Y), (X,Z), (Y,X), (Y,Z), (Z,X), (Z,Y)]
            for (pa, pb) in cross_pairs:
                H += J * J_cross * make_2site_term(N, i, j, pa, pb)
    
    return H, edges


def make_2site_term(N, i, j, pauli_i, pauli_j):
    """Build a two-site Pauli term acting on sites i and j."""
    term = np.eye(1, dtype=np.complex128)
    for k in range(N):
        if k == i:
            term = np.kron(term, pauli_i)
        elif k == j:
            term = np.kron(term, pauli_j)
        else:
            term = np.kron(term, I)
    return term


def make_1site_term(N, i, pauli):
    """Build a single-site Pauli term."""
    term = np.eye(1, dtype=np.complex128)
    for k in range(N):
        if k == i:
            term = np.kron(term, pauli)
        else:
            term = np.kron(term, I)
    return term


def get_edges(N, topology='chain'):
    """Get edges for given topology."""
    if topology == 'chain':
        return [(i, (i+1) % N) for i in range(N)]
    elif topology == 'lattice_2d':
        side = int(np.sqrt(N))
        if side * side != N:
            return [(i, (i+1) % N) for i in range(N)]
        edges = []
        for r in range(side):
            for c in range(side):
                idx = r * side + c
                right = r * side + (c + 1) % side
                edges.append((idx, right))
                down = ((r + 1) % side) * side + c
                edges.append((idx, down))
        return edges
    else:
        return [(i, (i+1) % N) for i in range(N)]


# ============================================================
# LIEB-ROBINSON VELOCITY MEASUREMENT
# ============================================================

def measure_lr_velocity(H, N, edges, dt=0.05, t_max=5.0,
                        source_site=0, threshold=0.01):
    """
    Measure the Lieb-Robinson velocity by tracking how fast
    a local perturbation propagates across the lattice.
    
    Method:
      1. Prepare initial state |psi_0>
      2. Prepare perturbed state O_source |psi_0>
      3. Evolve both under H
      4. At each time, measure trace distance of reduced states
         at each site
      5. Record when each site first exceeds threshold
      6. Velocity = distance / arrival_time
    """
    dim = 2**N
    
    # Initial state: |000...0>
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0
    
    # Perturbed state: X on source site
    O = make_1site_term(N, source_site, X)
    psi_pert = O @ psi0
    psi_pert /= np.linalg.norm(psi_pert)
    
    # Time evolution operator for one step
    U_dt = expm(-1j * dt * H)
    
    # Precompute graph distances from source
    distances = compute_distances(N, edges, source_site)
    
    # Track propagation
    n_steps = int(t_max / dt)
    times = []
    site_signals = {site: [] for site in range(N)}
    
    # Arrival times: first time signal exceeds threshold at each distance
    arrival_times = {}
    
    state0 = psi0.copy()
    state_p = psi_pert.copy()
    
    for step in range(n_steps + 1):
        t = step * dt
        times.append(t)
        
        for site in range(N):
            signal = site_trace_distance(state0, state_p, site, N)
            site_signals[site].append(signal)
            
            d = distances[site]
            if d > 0 and d not in arrival_times and signal > threshold:
                arrival_times[d] = t
        
        if step < n_steps:
            state0 = U_dt @ state0
            state_p = U_dt @ state_p
            state0 /= np.linalg.norm(state0)
            state_p /= np.linalg.norm(state_p)
    
    # Compute velocity from arrival times
    if len(arrival_times) >= 2:
        dists = sorted(arrival_times.keys())
        ts = [arrival_times[d] for d in dists]
        dists_arr = np.array(dists, dtype=float)
        ts_arr = np.array(ts, dtype=float)
        
        valid = ts_arr > 0
        if np.sum(valid) >= 2:
            v_max = dists_arr[valid][-1] / ts_arr[valid][-1]
            A = np.vstack([ts_arr[valid], np.ones(np.sum(valid))]).T
            result = np.linalg.lstsq(A, dists_arr[valid], rcond=None)
            v_fit = result[0][0]
        else:
            v_max = 0
            v_fit = 0
    elif len(arrival_times) == 1:
        d = list(arrival_times.keys())[0]
        t = list(arrival_times.values())[0]
        v_max = d / t if t > 0 else 0
        v_fit = v_max
    else:
        v_max = 0
        v_fit = 0
    
    # Peak signal at each distance
    peak_by_distance = {}
    for site, sigs in site_signals.items():
        d = distances[site]
        peak = max(sigs)
        if d not in peak_by_distance or peak > peak_by_distance[d]:
            peak_by_distance[d] = peak
    
    # Late-time profile
    late_idx = min(len(times) - 1, int(0.8 * len(times)))
    late_profile = {}
    for site in range(N):
        d = distances[site]
        sig = site_signals[site][late_idx]
        if d not in late_profile or sig > late_profile[d]:
            late_profile[d] = sig
    
    return {
        'velocity_max': v_max,
        'velocity_fit': v_fit,
        'arrival_times': arrival_times,
        'peak_by_distance': peak_by_distance,
        'late_profile': late_profile,
        'max_distance_reached': max(arrival_times.keys()) if arrival_times else 0,
    }


def site_trace_distance(psi1, psi2, site, N):
    """
    Trace distance of reduced states at a single site.
    TD = 0.5 * ||rho1 - rho2||_1
    """
    rho1 = partial_trace_single_site(psi1, site, N)
    rho2 = partial_trace_single_site(psi2, site, N)
    
    diff = rho1 - rho2
    s = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(s)


def partial_trace_single_site(psi, site, N):
    """
    Reduced density matrix of a single site.
    """
    dim = 2**N
    psi = psi.reshape(-1)
    
    # Reshape into (2^a, 2, 2^b) where a = sites before, b = sites after
    a = site
    b = N - site - 1
    dim_a = 2**a
    dim_b = 2**b
    
    psi_tensor = psi.reshape(dim_a, 2, dim_b)
    
    # rho_site = sum over environment indices
    # rho[s1, s2] = sum_{a,b} psi[a,s1,b] * psi[a,s2,b].conj()
    rho = np.einsum('asb,atb->st', psi_tensor, psi_tensor.conj())
    
    return rho


def compute_distances(N, edges, source):
    """BFS distances from source."""
    adj = {i: [] for i in range(N)}
    for (i, j) in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    dist = {source: 0}
    queue = [source]
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    
    return dist


# ============================================================
# INFORMATION TRANSPORT RATE
# ============================================================

def measure_transport_rate(H, N, edges, dt=0.05, t_max=5.0, 
                           source_site=0):
    """
    Measure how much excitation probability is transported.
    
    Create excitation at source_site, evolve, measure distribution.
    """
    dim = 2**N
    
    # Single excitation at source
    psi = np.zeros(dim, dtype=np.complex128)
    idx = 1 << (N - 1 - source_site)
    psi[idx] = 1.0
    
    U_dt = expm(-1j * dt * H)
    distances = compute_distances(N, edges, source_site)
    
    n_steps = int(t_max / dt)
    
    transport_data = []
    
    for step in range(n_steps + 1):
        t = step * dt
        
        site_probs = measure_site_occupations(psi, N)
        
        prob_by_dist = {}
        for site, prob in enumerate(site_probs):
            d = distances[site]
            prob_by_dist[d] = prob_by_dist.get(d, 0) + prob
        
        prob_at_source = site_probs[source_site]
        prob_transported = 1.0 - prob_at_source
        mean_dist = sum(d * p for d, p in prob_by_dist.items())
        
        transport_data.append({
            'time': t,
            'prob_at_source': prob_at_source,
            'prob_transported': prob_transported,
            'mean_distance': mean_dist,
        })
        
        if step < n_steps:
            psi = U_dt @ psi
            psi /= np.linalg.norm(psi)
    
    peak_transport = max(d['prob_transported'] for d in transport_data)
    second_half = transport_data[len(transport_data)//2:]
    mean_transport = np.mean([d['prob_transported'] for d in second_half])
    peak_distance = max(d['mean_distance'] for d in transport_data)
    peak_dist_time = transport_data[
        np.argmax([d['mean_distance'] for d in transport_data])
    ]['time']
    transport_velocity = peak_distance / peak_dist_time if peak_dist_time > 0 else 0
    
    return {
        'peak_transport': peak_transport,
        'mean_transport': mean_transport,
        'peak_distance': peak_distance,
        'transport_velocity': transport_velocity,
    }


def measure_site_occupations(psi, N):
    """Measure <n_i> = <(I-Z_i)/2> for each site."""
    dim = 2**N
    probs = np.abs(psi)**2
    
    occupations = []
    for site in range(N):
        occ = 0.0
        for idx in range(dim):
            if (idx >> (N - 1 - site)) & 1:
                occ += probs[idx]
        occupations.append(occ)
    
    return occupations


# ============================================================
# PARAMETER SWEEP
# ============================================================

def sweep_operator_triangle(N, n_points=10, topology='chain',
                             J=1.0, dt=0.05, t_max=5.0,
                             threshold=0.01, verbose=True):
    """
    Sweep the (alpha, beta, gamma) simplex and measure
    LR velocity and transport rate at each point.
    """
    print("=" * 70)
    print("BANDWIDTH-CONSTRAINED FORCE SELECTION")
    print(f"N={N}, topology={topology}, J={J}")
    print(f"dt={dt}, t_max={t_max}, threshold={threshold}")
    print(f"Sweep resolution: {n_points} points per axis")
    print("=" * 70)
    
    points = []
    for i in range(n_points + 1):
        for j in range(n_points + 1 - i):
            k = n_points - i - j
            alpha = i / n_points
            beta = j / n_points
            gamma = k / n_points
            points.append((alpha, beta, gamma))
    
    print(f"Total configurations: {len(points)}\n")
    
    results = []
    start = time.time()
    
    for idx, (alpha, beta, gamma) in enumerate(points):
        elapsed = time.time() - start
        eta = (elapsed / (idx + 1)) * (len(points) - idx - 1) if idx > 0 else 0
        
        if verbose:
            print(f"[{idx+1}/{len(points)}] "
                  f"hop={alpha:.2f} ising={beta:.2f} cross={gamma:.2f}",
                  end=" ... ", flush=True)
        
        H, edges = build_parameterized_hamiltonian(
            N, alpha, beta, gamma, J, topology
        )
        
        lr = measure_lr_velocity(
            H, N, edges, dt=dt, t_max=t_max, 
            source_site=0, threshold=threshold
        )
        
        transport = measure_transport_rate(
            H, N, edges, dt=dt, t_max=t_max, source_site=0
        )
        
        result = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'lr_velocity_max': lr['velocity_max'],
            'lr_velocity_fit': lr['velocity_fit'],
            'max_distance_reached': lr['max_distance_reached'],
            'peak_transport': transport['peak_transport'],
            'mean_transport': transport['mean_transport'],
            'peak_distance': transport['peak_distance'],
            'transport_velocity': transport['transport_velocity'],
        }
        results.append(result)
        
        if verbose:
            print(f"v_LR={lr['velocity_fit']:.3f} "
                  f"transport={transport['peak_transport']:.3f} "
                  f"[{elapsed/60:.1f}m, ETA {eta/60:.1f}m]")
    
    total_time = time.time() - start
    
    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    velocities = [r['lr_velocity_fit'] for r in results]
    transports = [r['peak_transport'] for r in results]
    
    max_v_idx = np.argmax(velocities)
    max_t_idx = np.argmax(transports)
    
    print(f"\nFastest propagation:")
    r = results[max_v_idx]
    print(f"  hop={r['alpha']:.2f} ising={r['beta']:.2f} cross={r['gamma']:.2f}")
    print(f"  v_LR = {r['lr_velocity_fit']:.4f}, transport = {r['peak_transport']:.4f}")
    
    print(f"\nMaximum transport:")
    r = results[max_t_idx]
    print(f"  hop={r['alpha']:.2f} ising={r['beta']:.2f} cross={r['gamma']:.2f}")
    print(f"  v_LR = {r['lr_velocity_fit']:.4f}, transport = {r['peak_transport']:.4f}")
    
    # Bandwidth analysis
    print("\n" + "=" * 70)
    print("BANDWIDTH CONSTRAINT ANALYSIS")
    print("=" * 70)
    
    v_max = max(v for v in velocities if v > 0) if any(v > 0 for v in velocities) else 1
    
    for B_frac in [0.3, 0.5, 0.7, 0.9]:
        B = B_frac * v_max
        allowed = [r for r in results 
                   if 0 < r['lr_velocity_fit'] <= B]
        
        if allowed:
            best = max(allowed, key=lambda r: r['peak_transport'])
            
            avg_hop = np.mean([r['alpha'] for r in allowed])
            avg_ising = np.mean([r['beta'] for r in allowed])
            avg_cross = np.mean([r['gamma'] for r in allowed])
            
            print(f"\n  Bandwidth B = {B_frac:.0%} of max (v <= {B:.4f}):")
            print(f"    {len(allowed)} allowed configurations")
            print(f"    Average composition: "
                  f"hop={avg_hop:.3f} ising={avg_ising:.3f} cross={avg_cross:.3f}")
            print(f"    Best transport within bandwidth:")
            print(f"      hop={best['alpha']:.2f} ising={best['beta']:.2f} "
                  f"cross={best['gamma']:.2f}")
            print(f"      v_LR={best['lr_velocity_fit']:.4f} "
                  f"transport={best['peak_transport']:.4f}")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    return results


# ============================================================
# PURE CHANNEL COMPARISON
# ============================================================

def compare_pure_channels(N, topology='chain', J=1.0, 
                           dt=0.05, t_max=5.0):
    """
    Detailed comparison of the three pure interaction channels
    plus key mixtures.
    """
    print("=" * 70)
    print("PURE CHANNEL COMPARISON")
    print(f"N={N}, topology={topology}")
    print("=" * 70)
    
    channels = {
        'Pure Hopping (XX+YY)':   (1.0, 0.0, 0.0),
        'Pure Ising (ZZ)':        (0.0, 1.0, 0.0),
        'Pure Cross (XY etc)':    (0.0, 0.0, 1.0),
        'Heisenberg (XX+YY+ZZ)':  (0.4, 0.2, 0.4),
        'Hopping + Ising':        (0.5, 0.5, 0.0),
        'Hopping + Cross':        (0.5, 0.0, 0.5),
        'Ising + Cross':          (0.0, 0.5, 0.5),
        'Democratic':             (0.333, 0.333, 0.334),
    }
    
    results = {}
    
    for name, (alpha, beta, gamma) in channels.items():
        print(f"\n--- {name} (a={alpha:.2f}, b={beta:.2f}, g={gamma:.2f}) ---")
        
        H, edges = build_parameterized_hamiltonian(
            N, alpha, beta, gamma, J, topology
        )
        
        lr = measure_lr_velocity(
            H, N, edges, dt=dt, t_max=t_max, source_site=0
        )
        
        transport = measure_transport_rate(
            H, N, edges, dt=dt, t_max=t_max, source_site=0
        )
        
        print(f"  LR velocity (fit):    {lr['velocity_fit']:.4f}")
        print(f"  LR velocity (max):    {lr['velocity_max']:.4f}")
        print(f"  Peak transport:       {transport['peak_transport']:.4f}")
        print(f"  Mean transport:       {transport['mean_transport']:.4f}")
        print(f"  Peak distance:        {transport['peak_distance']:.4f}")
        print(f"  Transport velocity:   {transport['transport_velocity']:.4f}")
        
        if lr['arrival_times']:
            print(f"  Arrival times: ", end="")
            for d in sorted(lr['arrival_times'].keys()):
                print(f"d={d}->t={lr['arrival_times'][d]:.2f}  ", end="")
            print()
        
        results[name] = {
            'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'lr_velocity_fit': lr['velocity_fit'],
            'lr_velocity_max': lr['velocity_max'],
            'peak_transport': transport['peak_transport'],
            'mean_transport': transport['mean_transport'],
            'transport_velocity': transport['transport_velocity'],
            'arrival_times': lr['arrival_times'],
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("CHANNEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Channel':<25} {'v_LR':<10} {'Transport':<10} {'v_trans':<10}")
    print("-" * 55)
    
    for name in channels:
        r = results[name]
        print(f"{name:<25} {r['lr_velocity_fit']:<10.4f} "
              f"{r['peak_transport']:<10.4f} {r['transport_velocity']:<10.4f}")
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bandwidth-Constrained Force Selection"
    )
    parser.add_argument('--quick', action='store_true', 
                        help='Quick channel comparison (N=8)')
    parser.add_argument('--standard', action='store_true',
                        help='Standard sweep (N=10, 10-point grid)')
    parser.add_argument('--fine', action='store_true',
                        help='Fine sweep (N=10, 20-point grid)')
    parser.add_argument('--channels', action='store_true',
                        help='Pure channel comparison only')
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--points', type=int, default=10)
    parser.add_argument('--topology', type=str, default='chain')
    parser.add_argument('--tmax', type=float, default=5.0)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.quick:
        results = compare_pure_channels(N=8, dt=0.05, t_max=4.0)
    
    elif args.channels:
        N = args.N or 10
        results = compare_pure_channels(
            N=N, topology=args.topology, dt=args.dt, t_max=args.tmax
        )
    
    elif args.standard:
        N = args.N or 10
        chan_results = compare_pure_channels(
            N=N, topology=args.topology, dt=args.dt, t_max=args.tmax
        )
        sweep_results = sweep_operator_triangle(
            N=N, n_points=args.points, topology=args.topology,
            dt=args.dt, t_max=args.tmax, verbose=True
        )
        results = {'channels': chan_results, 'sweep': sweep_results}
    
    elif args.fine:
        N = args.N or 10
        sweep_results = sweep_operator_triangle(
            N=N, n_points=20, topology=args.topology,
            dt=args.dt, t_max=args.tmax, verbose=True
        )
        results = sweep_results
    
    else:
        results = compare_pure_channels(N=8, dt=0.05, t_max=4.0)
    
    # Save
    if args.output:
        def to_serializable(obj):
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            return obj
        
        with open(args.output, 'w') as f:
            json.dump(to_serializable(results), f, indent=2)
        print(f"\nSaved to: {args.output}")
    
    print("\nDone.")