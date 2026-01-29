# constraint_emergence_test.py
# ------------------------------------------------------------
# Operational testbed for HSF-style constraints:
#   (1) No-signaling as finite-speed influence growth
#   (2) No-forgetting as long-time recoverability of a local "message"
#   (3) No-refolding as "don't let coarse-graining win by destroying (1)(2)"
#
# What it does:
#   - Builds an N-qubit 2-local Hamiltonian on a chosen interaction graph
#   - Evolves two initial pure states that differ by a local perturbation
#   - Measures influence over time via trace distance on reduced states
#   - Computes recoverability over time from small regions (blocks)
#   - Computes HIP-like transport heterogeneity on graph edges
#   - Evaluates the same diagnostics under different coarse-grainings
#     (blockings of qubits into "subsystems") to approximate factorization choice
#
# Notes:
#   - This is not a metaphysical "Hilbert space realism" test.
#   - It is a concrete falsification / support test for constraint *sufficiency*
#     in a toy setting.
#
# Requirements:
#   - Python 3.10+
#   - numpy
#
# Optional:
#   - cupy (not required; script auto-detects but defaults to numpy)
#
# Run example (Windows, single line):
#   python constraint_emergence_test.py --N 12 --topology ring --tmax 6 --nt 61 --seed 0 --blockings 1,2,3,4,6,12 --out results_constraints.json --progress
#
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import cupy as cp  # type: ignore
    HAVE_CUPY = True
except Exception:
    HAVE_CUPY = False


# ---------------------------
# Utility: Pauli matrices
# ---------------------------

I2 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PAULI = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


# ---------------------------
# Graph constructors
# ---------------------------

def build_edges(N: int, topology: str, rng: np.random.Generator, rr_deg: int = 3) -> List[Tuple[int, int]]:
    """
    Returns undirected edges (i<j) defining interaction topology.
    """
    topology = topology.lower()
    edges: set[Tuple[int, int]] = set()

    if topology == "ring":
        for i in range(N):
            j = (i + 1) % N
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    elif topology == "line":
        for i in range(N - 1):
            edges.add((i, i + 1))

    elif topology == "complete":
        for i in range(N):
            for j in range(i + 1, N):
                edges.add((i, j))

    elif topology.startswith("rr"):
        # random regular graph of degree rr_deg
        # Simple stub-matching; retries until succeeds.
        deg = rr_deg
        if topology != "rr":
            # allow "rr4" etc
            try:
                deg = int(topology[2:])
            except Exception:
                pass
        if deg >= N:
            raise ValueError("Random regular degree must be < N")

        for attempt in range(2000):
            stubs = []
            for i in range(N):
                stubs.extend([i] * deg)
            rng.shuffle(stubs)
            edges.clear()
            ok = True
            for k in range(0, len(stubs), 2):
                a = stubs[k]
                b = stubs[k + 1]
                if a == b:
                    ok = False
                    break
                e = (a, b) if a < b else (b, a)
                if e in edges:
                    ok = False
                    break
                edges.add(e)
            if ok:
                break
        else:
            raise RuntimeError("Failed to construct random regular graph after many attempts.")

    else:
        raise ValueError(f"Unknown topology: {topology}")

    return sorted(edges)


def neighbors_from_edges(N: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    nbrs = [[] for _ in range(N)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def graph_distance_matrix(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Unweighted shortest-path distances on the interaction graph.
    """
    nbrs = neighbors_from_edges(N, edges)
    dist = np.full((N, N), fill_value=10**9, dtype=np.int32)
    for s in range(N):
        dist[s, s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in nbrs[u]:
                if dist[s, v] > dist[s, u] + 1:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
    return dist


# ---------------------------
# Hamiltonian builder (2-local)
# ---------------------------

def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for k in range(1, len(ops)):
        out = np.kron(out, ops[k])
    return out


def pauli_on_qubit(N: int, q: int, P: str) -> np.ndarray:
    ops = [I2] * N
    ops[q] = PAULI[P]
    return kron_n(ops)


def two_qubit_term(N: int, i: int, j: int, Pi: str, Pj: str) -> np.ndarray:
    ops = [I2] * N
    ops[i] = PAULI[Pi]
    ops[j] = PAULI[Pj]
    return kron_n(ops)


def build_random_2local_hamiltonian(
    N: int,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
    onsite_scale: float = 0.3,
    edge_scale: float = 1.0,
) -> np.ndarray:
    """
    H = sum_i h_i · sigma_i + sum_(i,j in edges) sum_{P,Q in {X,Y,Z}} J_{ij}^{PQ} P_i Q_j

    Real coefficients -> Hermitian.
    """
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)

    # onsite fields
    for i in range(N):
        hx, hy, hz = rng.normal(scale=onsite_scale, size=3)
        H += hx * pauli_on_qubit(N, i, "X")
        H += hy * pauli_on_qubit(N, i, "Y")
        H += hz * pauli_on_qubit(N, i, "Z")

    # edge couplings
    paulis = ["X", "Y", "Z"]
    for (i, j) in edges:
        for P in paulis:
            for Q in paulis:
                J = rng.normal(scale=edge_scale)
                H += J * two_qubit_term(N, i, j, P, Q)

    # Hermitian sanity
    H = 0.5 * (H + H.conj().T)
    return H


# ---------------------------
# State evolution via eigendecomposition
# ---------------------------

@dataclass
class Evolver:
    N: int
    H: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray

    @staticmethod
    def from_hamiltonian(N: int, H: np.ndarray) -> "Evolver":
        evals, evecs = np.linalg.eigh(H)
        return Evolver(N=N, H=H, evals=evals, evecs=evecs)

    def evolve_state(self, psi0: np.ndarray, t: float) -> np.ndarray:
        # psi(t) = V exp(-i E t) V† psi0
        V = self.evecs
        coeff = V.conj().T @ psi0
        phase = np.exp(-1j * self.evals * t)
        coeff_t = phase * coeff
        psi_t = V @ coeff_t
        return psi_t


# ---------------------------
# Reduced density and trace distance
# ---------------------------

def reduced_density_pure_state(psi: np.ndarray, keep: List[int], N: int) -> np.ndarray:
    """
    Compute reduced density matrix on 'keep' qubits for a pure state |psi>.
    Returns rho_keep of shape (2^k, 2^k).
    """
    keep = sorted(keep)
    traced = [i for i in range(N) if i not in keep]
    # reshape into N indices
    psi_nd = psi.reshape([2] * N)
    # move keep to front
    perm = keep + traced
    psi_perm = np.transpose(psi_nd, axes=perm)
    k = len(keep)
    d_keep = 2 ** k
    d_tr = 2 ** (N - k)
    psi_mat = psi_perm.reshape(d_keep, d_tr)
    rho = psi_mat @ psi_mat.conj().T
    return rho


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    0.5 * ||rho - sigma||_1  (trace norm)
    For Hermitian delta, trace norm = sum |eigvals|.
    """
    delta = rho - sigma
    delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))


# ---------------------------
# Coarse-graining / blockings
# ---------------------------

def make_blocking(N: int, block_size: int) -> List[List[int]]:
    """
    Partition qubits into contiguous blocks of size block_size.
    For ring/line, contiguous is meaningful as a blocking; for other graphs it's just a grouping.
    """
    if N % block_size != 0:
        raise ValueError(f"N={N} not divisible by block_size={block_size}")
    blocks = []
    for start in range(0, N, block_size):
        blocks.append(list(range(start, start + block_size)))
    return blocks


def block_graph_from_qubit_graph(blocks: List[List[int]], edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Induce a block-level interaction graph: blocks u,v connected if any qubit edge crosses.
    """
    q2b = {}
    for bi, bq in enumerate(blocks):
        for q in bq:
            q2b[q] = bi

    bedges: set[Tuple[int, int]] = set()
    for a, b in edges:
        ba = q2b[a]
        bb = q2b[b]
        if ba == bb:
            continue
        e = (ba, bb) if ba < bb else (bb, ba)
        bedges.add(e)
    return sorted(bedges)


# ---------------------------
# Metrics
# ---------------------------

def influence_profiles_over_time(
    evolver: Evolver,
    psi_base: np.ndarray,
    psi_pert: np.ndarray,
    times: np.ndarray,
    regions: List[List[int]],
    N: int,
) -> np.ndarray:
    """
    Returns influence[t_idx, region_idx] = trace_distance(rho_base_region, rho_pert_region)
    """
    infl = np.zeros((len(times), len(regions)), dtype=np.float64)
    for ti, t in enumerate(times):
        psi0_t = evolver.evolve_state(psi_base, float(t))
        psi1_t = evolver.evolve_state(psi_pert, float(t))
        for ri, reg in enumerate(regions):
            rho0 = reduced_density_pure_state(psi0_t, reg, N)
            rho1 = reduced_density_pure_state(psi1_t, reg, N)
            infl[ti, ri] = trace_distance(rho0, rho1)
    return infl


def hip_edge_variance(
    evolver: Evolver,
    psi_base: np.ndarray,
    times: np.ndarray,
    edges: List[Tuple[int, int]],
    N: int,
    samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HIP_T-like: sample random local perturbations on node i and measure typical permeability on edge {i,j}
    using trace distance on the neighbor's reduced state.

    Returns:
      hip_t[t] = variance over edges of typical permeability
      edge_mean[t, e] = typical permeability for each edge
    """
    # We'll do: for each time t, for each edge (i,j),
    # estimate T_{i->j}(t) by sampling random Pauli on i applied to random product states (here just psi_base)
    # For tractability, we sample perturbations as {X,Y,Z} and average.
    paulis = ["X", "Y", "Z"]
    edge_mean = np.zeros((len(times), len(edges)), dtype=np.float64)

    # Precompute single-qubit Paulis on each site
    P_ops = {P: [pauli_on_qubit(N, i, P) for i in range(N)] for P in paulis}

    for ti, t in enumerate(times):
        # evolve base once
        psi0_t = evolver.evolve_state(psi_base, float(t))

        vals = []
        for ei, (i, j) in enumerate(edges):
            # Typical permeability estimate: average over random perturbations at i
            acc = 0.0
            for _ in range(samples):
                P = paulis[int(rng.integers(0, 3))]
                psi1 = P_ops[P][i] @ psi_base
                psi1 = psi1 / np.linalg.norm(psi1)
                psi1_t = evolver.evolve_state(psi1, float(t))
                rho0 = reduced_density_pure_state(psi0_t, [j], N)
                rho1 = reduced_density_pure_state(psi1_t, [j], N)
                acc += trace_distance(rho0, rho1)
            acc /= float(samples)
            edge_mean[ti, ei] = acc
            vals.append(acc)

        hip = float(np.var(np.array(vals, dtype=np.float64)))
        # hip_t computed after loop
    hip_t = np.var(edge_mean, axis=1)
    return hip_t, edge_mean


def rank_stability(scores_over_time: np.ndarray) -> float:
    """
    Given scores[t, node], compute average Spearman-ish rank correlation between successive times.
    We'll compute Kendall-like via ranks and Pearson on rank vectors.
    Returns in [-1,1], higher = more stable.
    """
    T, N = scores_over_time.shape
    if T < 2:
        return 1.0
    cors = []
    for t in range(T - 1):
        r1 = np.argsort(np.argsort(scores_over_time[t]))
        r2 = np.argsort(np.argsort(scores_over_time[t + 1]))
        r1 = r1.astype(np.float64)
        r2 = r2.astype(np.float64)
        r1 -= r1.mean()
        r2 -= r2.mean()
        denom = (np.linalg.norm(r1) * np.linalg.norm(r2) + 1e-12)
        cors.append(float((r1 @ r2) / denom))
    return float(np.mean(cors))


# ---------------------------
# Main experiment
# ---------------------------

def make_random_product_state(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random single-qubit product state (Haar on Bloch sphere approx).
    """
    state = np.array([1.0 + 0j])
    for _ in range(N):
        # random point on Bloch sphere: |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
        u = rng.random()
        v = rng.random()
        theta = math.acos(1 - 2 * u)
        phi = 2 * math.pi * v
        a = math.cos(theta / 2.0)
        b = math.sin(theta / 2.0) * complex(math.cos(phi), math.sin(phi))
        q = np.array([a, b], dtype=np.complex128)
        state = np.kron(state, q)
    state = state / np.linalg.norm(state)
    return state


def local_message_flip(N: int, q: int) -> np.ndarray:
    """
    "Message bit": apply Z on source qubit q.
    """
    return pauli_on_qubit(N, q, "Z")


def evaluate_blocking(
    N: int,
    edges: List[Tuple[int, int]],
    evolver: Evolver,
    psi_base: np.ndarray,
    source_qubit: int,
    times: np.ndarray,
    block_size: int,
    recover_threshold: float,
    speed_threshold: float,
    hip_samples: int,
    rng: np.random.Generator,
) -> Dict:
    """
    Evaluate constraints under a blocking (coarse-graining) into blocks of size block_size.
    Regions = blocks (each is a subsystem), so influence and recovery are defined at that resolution.
    """

    blocks = make_blocking(N, block_size)
    bedges = block_graph_from_qubit_graph(blocks, edges)

    # message states
    Zsrc = local_message_flip(N, source_qubit)
    psi_pert = Zsrc @ psi_base
    psi_pert = psi_pert / np.linalg.norm(psi_pert)

    # Influence vs time on each block (trace distance of reduced states)
    infl = influence_profiles_over_time(evolver, psi_base, psi_pert, times, blocks, N)
    # Recovery: best block at each time
    best_rec = infl.max(axis=1)
    # No-forgetting operational: fraction of times with recoverability above threshold
    frac_recover = float(np.mean(best_rec >= recover_threshold))

    # No-signaling operational: "speed": how quickly many blocks become distinguishable.
    # We define "activated" blocks at time t where influence > speed_threshold.
    activated = (infl >= speed_threshold).sum(axis=1)  # count blocks above threshold
    # A crude "causal diameter time": first time when >= 50% blocks activated
    half = max(1, len(blocks) // 2)
    t_half = float(times[np.argmax(activated >= half)]) if np.any(activated >= half) else float("inf")

    # HIP-ish on induced block-graph: compute edge permeabilities by sampling perturbations at one endpoint
    # For tractability we compute HIP on the original qubit graph (single-qubit regions), then aggregate node intensity into blocks.
    hip_t_qubit, edge_mean = hip_edge_variance(evolver, psi_base, times, edges, N, hip_samples, rng)

    # Node intensity at qubit-level: sum of incident edge means
    nbrs = neighbors_from_edges(N, edges)
    # Map edge_mean[t, e] to node outgoing: sum over incident edges
    node_intensity = np.zeros((len(times), N), dtype=np.float64)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    for ti in range(len(times)):
        for i in range(N):
            s = 0.0
            for j in nbrs[i]:
                a, b = (i, j) if i < j else (j, i)
                s += edge_mean[ti, edge_index[(a, b)]]
            node_intensity[ti, i] = s

    # Aggregate node intensity into blocks
    block_intensity = np.zeros((len(times), len(blocks)), dtype=np.float64)
    for bi, bq in enumerate(blocks):
        block_intensity[:, bi] = node_intensity[:, bq].sum(axis=1)

    # Rank stability of blocks (structure persistence)
    stab = rank_stability(block_intensity)

    # "Refolding penalty": coarse blockings are allowed only if they still have:
    # - nontrivial causal diameter time (not instantaneous)
    # - decent recoverability
    # We'll convert to a penalty score; lower is better.
    # Note: this is an operational proxy for "no-refolding preserves structure".
    penalty = 0.0
    if not math.isfinite(t_half):
        penalty += 2.0
    else:
        # Too-fast activation suggests trivial "two boxes" with immediate global access
        # Normalize by time span
        penalty += max(0.0, (1.0 - (t_half / float(times[-1]))))  # smaller t_half => bigger penalty
    penalty += max(0.0, (0.8 - frac_recover))  # require recoverability

    # Multi-constraint score (higher = better)
    # Reward: recoverability, slower-than-trivial activation, persistence stability, HIP variance
    hip_avg = float(np.mean(hip_t_qubit))
    score = (
        1.5 * frac_recover
        + 0.8 * (t_half / float(times[-1]) if math.isfinite(t_half) else 0.0)
        + 0.7 * ((stab + 1.0) / 2.0)
        + 0.3 * math.tanh(5.0 * hip_avg)
        - 1.0 * penalty
    )

    return {
        "block_size": int(block_size),
        "n_blocks": int(len(blocks)),
        "block_edges": int(len(bedges)),
        "recover_threshold": float(recover_threshold),
        "speed_threshold": float(speed_threshold),
        "frac_recover": float(frac_recover),
        "t_half_activation": float(t_half) if math.isfinite(t_half) else None,
        "hip_avg_qubit": float(hip_avg),
        "rank_stability_blocks": float(stab),
        "penalty": float(penalty),
        "score": float(score),
        "times": [float(x) for x in times],
        "best_recoverability_over_time": [float(x) for x in best_rec],
        "activated_blocks_over_time": [int(x) for x in activated],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--topology", type=str, default="ring", help="ring|line|complete|rr|rr4|rr6 ...")
    ap.add_argument("--rr-deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=6.0)
    ap.add_argument("--nt", type=int, default=61)
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--onsite-scale", type=float, default=0.25)
    ap.add_argument("--edge-scale", type=float, default=0.8)

    ap.add_argument("--blockings", type=str, default="1,2,3,4,6,12",
                    help="comma-separated block sizes that divide N. block_size=1 means qubit sites.")
    ap.add_argument("--recover-threshold", type=float, default=0.05,
                    help="trace distance threshold for recoverability (no-forgetting proxy).")
    ap.add_argument("--speed-threshold", type=float, default=0.02,
                    help="trace distance threshold for 'activated' regions (no-signaling proxy).")
    ap.add_argument("--hip-samples", type=int, default=6,
                    help="samples per edge per time for HIP estimate (keep small; cost scales).")
    ap.add_argument("--out", type=str, default="constraint_results.json")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    if args.N < 2 or args.N > 14:
        print("NOTE: This script uses exact diagonalization; N>14 will likely be too slow / too big.")
        print("Proceeding anyway, but you may want to keep N<=12.")
    N = args.N

    rng = np.random.default_rng(args.seed)
    edges = build_edges(N, args.topology, rng, rr_deg=args.rr_deg)
    dist = graph_distance_matrix(N, edges)

    if args.progress:
        print("CONSTRAINT EMERGENCE TEST")
        print("-------------------------")
        print(f"N={N} topology={args.topology} edges={len(edges)} seed={args.seed}")
        print(f"t in [0,{args.tmax}] with nt={args.nt}")
        print(f"blockings={args.blockings}")
        print(f"recover_threshold={args.recover_threshold} speed_threshold={args.speed_threshold}")
        print(f"hip_samples={args.hip_samples}")

    # Build Hamiltonian
    H = build_random_2local_hamiltonian(
        N=N,
        edges=edges,
        rng=rng,
        onsite_scale=args.onsite_scale,
        edge_scale=args.edge_scale,
    )

    if args.progress:
        print("Diagonalizing H...")
    evolver = Evolver.from_hamiltonian(N, H)

    # Base state: random product to avoid embedding special structure
    psi_base = make_random_product_state(N, rng)

    times = np.linspace(0.0, float(args.tmax), int(args.nt), dtype=np.float64)

    # Evaluate blockings
    blocking_sizes = [int(x.strip()) for x in args.blockings.split(",") if x.strip()]
    blocking_sizes = sorted(set(blocking_sizes))
    for b in blocking_sizes:
        if N % b != 0:
            raise ValueError(f"Blocking {b} does not divide N={N}")

    results = {
        "meta": {
            "N": N,
            "topology": args.topology,
            "rr_deg": args.rr_deg,
            "seed": args.seed,
            "tmax": float(args.tmax),
            "nt": int(args.nt),
            "source_qubit": int(args.source),
            "recover_threshold": float(args.recover_threshold),
            "speed_threshold": float(args.speed_threshold),
            "hip_samples": int(args.hip_samples),
            "onsite_scale": float(args.onsite_scale),
            "edge_scale": float(args.edge_scale),
            "edges": edges,
        },
        "blockings": [],
    }

    for b in blocking_sizes:
        if args.progress:
            print(f"Evaluating blocking block_size={b} (n_blocks={N//b}) ...")
        out = evaluate_blocking(
            N=N,
            edges=edges,
            evolver=evolver,
            psi_base=psi_base,
            source_qubit=int(args.source),
            times=times,
            block_size=b,
            recover_threshold=float(args.recover_threshold),
            speed_threshold=float(args.speed_threshold),
            hip_samples=int(args.hip_samples),
            rng=rng,
        )
        results["blockings"].append(out)

    # Sort by score desc
    results["blockings"] = sorted(results["blockings"], key=lambda d: d["score"], reverse=True)
    results["winner"] = results["blockings"][0] if results["blockings"] else None

    # Write
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if args.progress:
        print("\nRESULTS (sorted by score desc)")
        for r in results["blockings"]:
            print(
                f"  b={r['block_size']:>2} blocks={r['n_blocks']:>2} "
                f"score={r['score']:+.3f} "
                f"recover={r['frac_recover']:.3f} "
                f"t_half={r['t_half_activation']} "
                f"stab={r['rank_stability_blocks']:.3f} "
                f"hip={r['hip_avg_qubit']:.4f} "
                f"pen={r['penalty']:.3f}"
            )
        print(f"\nWrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
