"""
emergent_geometry_demo.py

Emergent geometry and causal structure from a Hilbert-space substrate.

This script uses the hilbert_substrate_core.Substrate class with:

    - N_FACTORS qubit factors (local_dim = 2).
    - A random sparse interaction graph over factor labels.
    - Local XX+YY interactions on each edge of that graph.

We do two things:

1. GEOMETRY FROM GRAPH STRUCTURE
   --------------------------------
   Using only the interaction graph (no reference to an external space),
   we compute:

       - Graph distances d(0, j) from a chosen origin node (here: node 0).
       - Ball volumes B(r) = number of nodes with d <= r for radii r = 0..R_max.

   B(r) encodes the "volume growth" of the interaction graph as a function
   of radius, which is a primitive probe of emergent geometry and effective
   dimension.

2. CAUSAL SPREAD FROM HILBERT-SPACE DYNAMICS
   ------------------------------------------
   We initialize a localized excitation at node 0:

       |Ψ(0)> = |1, 0, 0, ..., 0>

   and evolve under local XX+YY unitaries attached to the random graph edges.

   At each step, we measure local magnetizations:

       m_j(t) = <Ψ(t)| Z_j |Ψ(t)>

   and define an "activity" relative to the +Z background:

       activity_j(t) = 0.5 * |m_j(t) - 1|

   Using that and the graph distances, we define an effective "front radius":

       r_front(t) = [Σ_j d(0,j) * activity_j(t)] / [Σ_j activity_j(t)].

   This r_front(t) should grow approximately linearly with time, reflecting
   a Lieb–Robinson-like finite propagation speed over the abstract interaction
   graph. This is a proto-spacetime structure emerging from Hilbert-space
   dynamics alone.

Outputs:
    emergent_geometry_results.npz with keys:
        "t"           : array (steps+1,)      integer time steps 0..steps
        "mz"          : array (N, steps+1)    local <Z_j>(t)
        "distances"   : array (N,)            graph distance d(0,j)
        "ball_radii"  : array (R_max+1,)      radii 0..R_max
        "ball_volumes": array (R_max+1,)      B(r) = |{ j : d(0,j) <= r }|
        "r_front"     : array (steps+1,)      effective front radius vs time
        "cfg"         : dict                  configuration parameters

This script has no CLI; edit the CONFIG block below to change parameters.

NOTE: Because the substrate engine stores the full state vector, the total
Hilbert-space dimension grows as 2**N_FACTORS. For practical use, keep:

    N_FACTORS ≲ 18

Otherwise you'll run out of memory.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np

from hilbert_substrate_core import Config, Substrate


# ---------------------------------------------------------------------------
# CONFIG (edit these if you want to change the demo)
# ---------------------------------------------------------------------------

# Number of Hilbert-space factors (qubits)
# 2**14 = 16384 complex amplitudes -> very safe.
N_FACTORS: int = 14

# Target average degree of the random interaction graph
AVG_DEGREE: int = 4

# Dynamics parameters
STEPS: int = 80          # number of discrete time steps
DT: float = 0.15         # time step used in exp(-i H_pair * DT)
J_COUPLING: float = 1.0  # coupling strength J for XX+YY interactions

# RNG seed and backend
SEED: int = 123
USE_GPU: bool = False

# Output file
OUT_FILE: str = "emergent_geometry_results.npz"


# ---------------------------------------------------------------------------
# Random sparse interaction graph
# ---------------------------------------------------------------------------

def build_random_sparse_graph(
    n: int,
    avg_degree: int,
    rng: np.random.Generator,
) -> Dict[int, List[int]]:
    """
    Build a simple undirected random sparse graph on n nodes with
    approximate average degree `avg_degree`.

    The procedure:
        - Start with no edges.
        - For each node i, pick avg_degree random neighbors (excluding itself),
          and add undirected edges (i, j).
        - Avoid duplicate neighbors.

    The result is a dictionary mapping node -> sorted list of neighbors.

    This is *not* a geometric graph; it is an abstract interaction graph
    representing which Hilbert-space factors are coupled by local terms.
    """
    neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i in range(n):
        possible = [j for j in range(n) if j != i]
        rng.shuffle(possible)
        chosen = possible[:avg_degree]

        for j in chosen:
            if j not in neighbors[i]:
                neighbors[i].append(j)
            if i not in neighbors[j]:
                neighbors[j].append(i)

    for i in range(n):
        neighbors[i].sort()

    return neighbors


def graph_edges_from_neighbors(neighbors: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """
    Convert an undirected neighbor dictionary to a list of unique edges (i, j)
    with i < j.
    """
    edges: List[Tuple[int, int]] = []
    for i, nbrs in neighbors.items():
        for j in nbrs:
            if i < j:
                edges.append((i, j))
    return edges


# ---------------------------------------------------------------------------
# Attach XX+YY dynamics to the graph
# ---------------------------------------------------------------------------

def build_xx_yy_on_graph(
    sub: Substrate,
    edges: List[Tuple[int, int]],
    dt: float,
    J: float = 1.0,
) -> None:
    """
    For each undirected edge (i, j) in `edges`, register a local XX+YY unitary
    on the corresponding pair of factors in the substrate.

    Local Hamiltonian on (i, j):
        H_pair = J (X ⊗ X + Y ⊗ Y)

    We convert this to a local unitary:
        U_pair = exp(-i H_pair dt)

    and call sub.add_local_unitary(sites=[i, j], unitary=U_pair).
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("build_xx_yy_on_graph is implemented for qubits (local_dim == 2).")

    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    H_pair = J * (np.kron(X, X) + np.kron(Y, Y))
    U_pair = Substrate.hermitian_to_unitary(H_pair, dt, use_gpu=sub.on_gpu)

    for (i, j) in edges:
        sub.add_local_unitary(sites=[i, j], unitary=U_pair)


# ---------------------------------------------------------------------------
# Graph distances and volume growth
# ---------------------------------------------------------------------------

def bfs_distances(
    neighbors: Dict[int, List[int]],
    origin: int,
) -> np.ndarray:
    """
    Compute graph distances d(origin, j) from `origin` to all nodes j via BFS.
    """
    n = len(neighbors)
    dist = np.full(n, np.inf, dtype=float)

    if origin < 0 or origin >= n:
        raise ValueError(f"Origin {origin} out of range 0..{n-1}.")

    from collections import deque

    dist[origin] = 0.0
    q = deque([origin])

    while q:
        i = q.popleft()
        di = dist[i]
        for j in neighbors[i]:
            if np.isinf(dist[j]):
                dist[j] = di + 1.0
                q.append(j)

    return dist


def ball_volumes_from_distances(distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ball volumes B(r) = |{ j : d(0,j) <= r }| for integer radii r.
    """
    finite_dist = distances[np.isfinite(distances)]
    if finite_dist.size == 0:
        return np.array([0], dtype=float), np.array([0], dtype=float)

    max_r = int(finite_dist.max())
    radii = np.arange(max_r + 1, dtype=float)
    volumes = np.zeros_like(radii)

    for r in range(max_r + 1):
        volumes[r] = np.sum(finite_dist <= r)

    return radii, volumes


# ---------------------------------------------------------------------------
# Observable: local Z expectation values and front radius
# ---------------------------------------------------------------------------

def z_expectations(sub: Substrate) -> np.ndarray:
    """
    Compute <Z_j> for each site j of the substrate (assuming qubits).
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("z_expectations is defined only for local_dim == 2.")

    n = sub.cfg.n_factors
    d = sub.cfg.local_dim

    psi = sub.to_numpy(sub.state).reshape((d,) * n)

    mz = np.zeros(n, dtype=float)

    for j in range(n):
        axes = list(range(n))
        axes[j], axes[-1] = axes[-1], axes[j]
        psi_perm = np.transpose(psi, axes=axes)
        psi_flat = psi_perm.reshape(-1, d)

        probs = np.sum(np.abs(psi_flat) ** 2, axis=0)
        mz[j] = probs[0] - probs[1]

    return mz


def front_radius_from_mz(
    distances: np.ndarray,
    mz: np.ndarray,
) -> float:
    """
    Compute an effective "front radius" r_front from graph distances and
    local magnetizations mz[j] = <Z_j>.
    """
    finite_mask = np.isfinite(distances)
    d_finite = distances[finite_mask]
    mz_finite = mz[finite_mask]

    activity = 0.5 * np.abs(mz_finite - 1.0)
    total_activity = np.sum(activity)
    if total_activity == 0.0:
        return 0.0

    r_front = float(np.sum(d_finite * activity) / total_activity)
    return r_front


# ---------------------------------------------------------------------------
# Main simulation function (importable)
# ---------------------------------------------------------------------------

def run_emergent_geometry_demo(
    n_factors: int,
    avg_degree: int,
    steps: int,
    dt: float,
    J: float,
    seed: int,
    use_gpu: bool,
) -> Dict[str, Any]:
    """
    Run the emergent-geometry and causal-spread demonstration.
    """
    rng = np.random.default_rng(seed)

    neighbors = build_random_sparse_graph(n_factors, avg_degree=avg_degree, rng=rng)
    edges = graph_edges_from_neighbors(neighbors)

    print(f"[Geometry] Built random graph with n={n_factors}, "
          f"~avg_degree={avg_degree}, edges={len(edges)}")

    product_state = [0] * n_factors
    product_state[0] = 1  # |1,0,0,...>

    cfg = Config(
        n_factors=n_factors,
        local_dim=2,
        use_gpu=use_gpu,
        seed=seed,
        product_state=product_state,
    )

    # Print total Hilbert dimension so you keep an eye on it
    dim_total = cfg.local_dim ** cfg.n_factors
    print(f"[Geometry] Total Hilbert dimension = {dim_total}")

    sub = Substrate(cfg)
    print("[Geometry] Initial norm:", sub.norm())

    build_xx_yy_on_graph(sub, edges=edges, dt=dt, J=J)

    distances = bfs_distances(neighbors, origin=0)
    ball_radii, ball_volumes = ball_volumes_from_distances(distances)

    max_finite = int(np.nanmax(distances[np.isfinite(distances)]))
    print(f"[Geometry] Max finite graph distance from origin: {max_finite}")
    print("[Geometry] Ball volumes B(r):")
    for r, vol in zip(ball_radii, ball_volumes):
        print(f"  r={int(r)} -> B(r)={int(vol)}")

    t_vals = np.arange(steps + 1, dtype=float)
    mz_ts = np.zeros((n_factors, steps + 1), dtype=float)
    r_front_ts = np.zeros(steps + 1, dtype=float)

    mz_ts[:, 0] = z_expectations(sub)
    r_front_ts[0] = front_radius_from_mz(distances, mz_ts[:, 0])

    for step in range(1, steps + 1):
        sub.step(n_steps=1)
        mz_ts[:, step] = z_expectations(sub)
        r_front_ts[step] = front_radius_from_mz(distances, mz_ts[:, step])

        if step % max(1, steps // 10) == 0:
            print(f"[Dynamics] step {step}/{steps} "
                  f"r_front ≈ {r_front_ts[step]:.3f}")

    results = {
        "t": t_vals,
        "mz": mz_ts,
        "distances": distances,
        "ball_radii": ball_radii,
        "ball_volumes": ball_volumes,
        "r_front": r_front_ts,
        "cfg": {
            "n_factors": n_factors,
            "avg_degree": avg_degree,
            "steps": steps,
            "dt": dt,
            "J": J,
            "seed": seed,
            "use_gpu": use_gpu,
        },
    }
    return results


# ---------------------------------------------------------------------------
# Script entrypoint (no CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Emergent Geometry Demo (no CLI) ===")
    print(f"n_factors={N_FACTORS}, avg_degree={AVG_DEGREE}")
    print(f"steps={STEPS}, dt={DT}, J={J_COUPLING}")
    print(f"seed={SEED}, use_gpu={USE_GPU}")
    print(f"Output file: {OUT_FILE}")

    res = run_emergent_geometry_demo(
        n_factors=N_FACTORS,
        avg_degree=AVG_DEGREE,
        steps=STEPS,
        dt=DT,
        J=J_COUPLING,
        seed=SEED,
        use_gpu=USE_GPU,
    )

    np.savez_compressed(OUT_FILE, **res)
    print(f"Saved results to {OUT_FILE}")
    print("Done.")
