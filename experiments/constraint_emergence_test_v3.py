# constraint_emergence_test_v3.py
# ------------------------------------------------------------
# HSF Constraint Test (V3) — adds the FOURTH constraint: finite bandwidth
#
# The four constraints and their tensions:
#
#   (1) No-signaling: influence must propagate at finite speed
#       → rewards factorizations with well-defined causal structure
#
#   (2) No-forgetting: information must remain recoverable somewhere
#       → rewards factorizations where correlations persist
#       → ALONE this favors coarse graining (pool more qubits = more signal)
#
#   (3) No-refolding: can't destroy record-bearing structure
#       → penalizes trivially coarse factorizations (too few blocks)
#
#   (4) Finite bandwidth: each subsystem has LIMITED information capacity
#       → penalizes coarse graining (big blocks exceed capacity)
#       → creates tension with no-forgetting
#
# The hypothesis: these four constraints together create a tension that
# ONLY spatial locality can resolve:
#   - Fine enough to stay within bandwidth
#   - Coarse enough to maintain recoverable correlations
#   - Structured enough to support causal propagation
#
# The bandwidth constraint is implemented as:
#   - Each block has capacity B (in qubit-equivalents)
#   - A block of size k has efficiency min(k, B) / k
#   - Effective recoverability = raw_signal × efficiency
#   - Additional penalty for exceeding bandwidth
#
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# ---------------------------
# Pauli matrices
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
        deg = rr_deg
        if topology != "rr":
            try:
                deg = int(topology[2:])
            except Exception:
                pass
        if deg >= N:
            raise ValueError("Random regular degree must be < N")

        for _attempt in range(2000):
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


def neighbors_from_edges(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    nbrs = [[] for _ in range(n)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def graph_distance_matrix(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    nbrs = neighbors_from_edges(n, edges)
    dist = np.full((n, n), fill_value=10**9, dtype=np.int32)
    for s in range(n):
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
    onsite_scale: float = 0.25,
    edge_scale: float = 0.8,
) -> np.ndarray:
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(N):
        hx, hy, hz = rng.normal(scale=onsite_scale, size=3)
        H += hx * pauli_on_qubit(N, i, "X")
        H += hy * pauli_on_qubit(N, i, "Y")
        H += hz * pauli_on_qubit(N, i, "Z")

    paulis = ["X", "Y", "Z"]
    for (i, j) in edges:
        for P in paulis:
            for Q in paulis:
                J = rng.normal(scale=edge_scale)
                H += J * two_qubit_term(N, i, j, P, Q)

    H = 0.5 * (H + H.conj().T)
    return H


# ---------------------------
# Evolution via eigendecomposition
# ---------------------------

@dataclass
class Evolver:
    N: int
    evals: np.ndarray
    evecs: np.ndarray

    @staticmethod
    def from_hamiltonian(N: int, H: np.ndarray) -> "Evolver":
        evals, evecs = np.linalg.eigh(H)
        return Evolver(N=N, evals=evals, evecs=evecs)

    def evolve_state(self, psi0: np.ndarray, t: float) -> np.ndarray:
        V = self.evecs
        coeff = V.conj().T @ psi0
        phase = np.exp(-1j * self.evals * t)
        return V @ (phase * coeff)


# ---------------------------
# Reduced density and trace distance
# ---------------------------

def reduced_density_pure_state(psi: np.ndarray, keep: List[int], N: int) -> np.ndarray:
    keep = sorted(keep)
    d = 2
    dim_keep = d ** len(keep)
    dim_trace = d ** (N - len(keep))
    
    all_qubits = list(range(N))
    trace_out = [q for q in all_qubits if q not in keep]
    
    perm = keep + trace_out
    
    psi_tensor = psi.reshape([2] * N)
    psi_perm = np.transpose(psi_tensor, perm)
    psi_reshaped = psi_perm.reshape(dim_keep, dim_trace)
    
    rho = psi_reshaped @ psi_reshaped.conj().T
    return rho


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    diff = rho - sigma
    eigs = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigs)))


# ---------------------------
# Initial state
# ---------------------------

def make_random_product_state(N: int, rng: np.random.Generator) -> np.ndarray:
    psi = np.array([1.0], dtype=np.complex128)
    for _ in range(N):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        qubit = np.array([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2)
        ], dtype=np.complex128)
        psi = np.kron(psi, qubit)
    return psi / np.linalg.norm(psi)


# ---------------------------
# Bandwidth efficiency
# ---------------------------

def bandwidth_efficiency(block_size: int, bandwidth_capacity: float) -> float:
    """
    Returns the fraction of information a block can effectively process.
    
    A block of k qubits with bandwidth capacity B can only utilize
    min(k, B) / k of its potential information content.
    
    This implements the finite bandwidth constraint:
    - block_size <= bandwidth: efficiency = 1.0 (full utilization)
    - block_size > bandwidth: efficiency < 1.0 (capacity exceeded)
    """
    if block_size <= 0:
        return 0.0
    if bandwidth_capacity <= 0:
        return 0.0
    return min(float(block_size), bandwidth_capacity) / float(block_size)


def bandwidth_penalty(block_size: int, bandwidth_capacity: float, penalty_scale: float = 1.0) -> float:
    """
    Returns a penalty for exceeding bandwidth capacity.
    
    Penalty grows with how much the block exceeds capacity.
    """
    if block_size <= bandwidth_capacity:
        return 0.0
    excess = (block_size - bandwidth_capacity) / bandwidth_capacity
    return penalty_scale * excess


# ---------------------------
# Blocking utilities
# ---------------------------

def make_contiguous_blocks(N: int, block_size: int) -> List[List[int]]:
    if N % block_size != 0:
        raise ValueError(f"block_size {block_size} does not divide N={N}")
    nB = N // block_size
    return [list(range(b * block_size, (b + 1) * block_size)) for b in range(nB)]


def block_graph_edges(qubit_edges: List[Tuple[int, int]], blocks: List[List[int]]) -> List[Tuple[int, int]]:
    qubit_to_block = {}
    for b_idx, blk in enumerate(blocks):
        for q in blk:
            qubit_to_block[q] = b_idx
    
    block_edges_set: set[Tuple[int, int]] = set()
    for (i, j) in qubit_edges:
        bi = qubit_to_block.get(i)
        bj = qubit_to_block.get(j)
        if bi is None or bj is None:
            continue
        if bi != bj:
            e = (bi, bj) if bi < bj else (bj, bi)
            block_edges_set.add(e)
    return sorted(block_edges_set)


# ---------------------------
# V3 Evaluation with all 4 constraints
# ---------------------------

def evaluate_blocking_v3(
    N: int,
    qubit_edges: List[Tuple[int, int]],
    evolver: Evolver,
    psi_base: np.ndarray,
    source_qubit: int,
    times: np.ndarray,
    block_size: int,
    # Constraint parameters
    speed_threshold: float = 0.02,
    recover_threshold: float = 0.03,
    d_remote_min: int = 2,
    t_late_frac: float = 0.4,
    min_blocks: int = 4,
    # NEW: Bandwidth parameters
    bandwidth_capacity: float = 1.5,
    bandwidth_penalty_scale: float = 1.0,
) -> Dict:
    """
    Evaluate a blocking under all four HSF constraints.
    
    The four constraints create tension:
    1. No-signaling: want structured propagation (finite speed)
    2. No-forgetting: want recoverable information (favors coarse graining)
    3. No-refolding: want enough structure (penalizes too-coarse)
    4. Finite bandwidth: want small blocks (penalizes coarse graining)
    
    The hypothesis: only spatial locality resolves all four tensions.
    """
    blocks = make_contiguous_blocks(N, block_size)
    nB = len(blocks)
    
    bedges = block_graph_edges(qubit_edges, blocks)
    bdist = graph_distance_matrix(nB, bedges) if nB > 1 else np.zeros((1, 1), dtype=np.int32)
    
    source_block = source_qubit // block_size
    dist_from_source = [int(bdist[source_block, b]) for b in range(nB)]
    
    # Compute bandwidth efficiency for this block size
    bw_eff = bandwidth_efficiency(block_size, bandwidth_capacity)
    bw_pen = bandwidth_penalty(block_size, bandwidth_capacity, bandwidth_penalty_scale)
    
    # Apply random Pauli perturbation at source
    rng_pert = np.random.default_rng(12345)
    paulis = [X2, Y2, Z2]
    P_choice = paulis[rng_pert.integers(0, 3)]
    
    ops = [I2] * N
    ops[source_qubit] = P_choice
    U_pert = kron_n(ops)
    
    psi_pert = U_pert @ psi_base
    
    # Evolve and measure block-level influence over time
    nt = len(times)
    infl = np.zeros((nt, nB), dtype=np.float64)
    
    for ti, t in enumerate(times):
        psi_a = evolver.evolve_state(psi_base, t)
        psi_b = evolver.evolve_state(psi_pert, t)
        
        for b in range(nB):
            rho_a = reduced_density_pure_state(psi_a, blocks[b], N)
            rho_b = reduced_density_pure_state(psi_b, blocks[b], N)
            infl[ti, b] = trace_distance(rho_a, rho_b)
    
    # ---- No-signaling proxy: finite-speed propagation
    t_cross = [None] * nB
    for b in range(nB):
        for ti, t in enumerate(times):
            if infl[ti, b] >= speed_threshold:
                t_cross[b] = float(t)
                break
    
    v_candidates = []
    instant_reach = 0
    reached = 0
    for b in range(nB):
        d = dist_from_source[b]
        if d <= 0:
            continue
        tc = t_cross[b]
        if tc is None:
            continue
        reached += 1
        if tc <= 1e-12:
            instant_reach += 1
            continue
        v_candidates.append(d / tc)
    
    v_eff = float(max(v_candidates)) if v_candidates else 0.0
    frac_reached = float(reached / max(1, (nB - 1)))
    
    # ---- No-forgetting proxy: persistent REMOTE recoverability
    # Now scaled by bandwidth efficiency!
    remote_blocks = [b for b in range(nB) if b != source_block and dist_from_source[b] >= d_remote_min]
    
    if not remote_blocks:
        remote_best_raw = np.zeros(len(times), dtype=np.float64)
    else:
        remote_best_raw = infl[:, remote_blocks].max(axis=1)
    
    # Apply bandwidth efficiency: effective signal is raw × efficiency
    remote_best_effective = remote_best_raw * bw_eff
    
    t_late_start = float(times[0] + t_late_frac * (times[-1] - times[0]))
    late_mask = times >= t_late_start
    
    if np.any(late_mask):
        # Raw metrics (without bandwidth)
        frac_remote_recover_late_raw = float(np.mean(remote_best_raw[late_mask] >= recover_threshold))
        remote_recover_mean_late_raw = float(np.mean(remote_best_raw[late_mask]))
        
        # Effective metrics (with bandwidth)
        frac_remote_recover_late_eff = float(np.mean(remote_best_effective[late_mask] >= recover_threshold))
        remote_recover_mean_late_eff = float(np.mean(remote_best_effective[late_mask]))
    else:
        frac_remote_recover_late_raw = 0.0
        remote_recover_mean_late_raw = 0.0
        frac_remote_recover_late_eff = 0.0
        remote_recover_mean_late_eff = 0.0
    
    # ---- No-refolding proxy: must maintain structure
    hard_penalty = 0.0
    if nB < min_blocks:
        hard_penalty += 2.0
    if nB > 1 and len(bedges) == 0:
        hard_penalty += 2.0
    
    # ---- Soft penalties
    soft_penalty = 0.0
    soft_penalty += 0.75 * float(instant_reach)
    soft_penalty += 0.50 * max(0.0, (v_eff - 20.0) / 20.0)
    
    # ---- Bandwidth penalty (NEW in V3)
    # Penalizes blocks that exceed their information capacity
    soft_penalty += bw_pen
    
    # ---- Score using EFFECTIVE (bandwidth-limited) recoverability
    # This is the key change: coarse graining is penalized both by
    # reduced efficiency AND explicit bandwidth penalty
    score = (
        2.0 * frac_remote_recover_late_eff  # Use effective, not raw
        + 0.8 * math.tanh(5.0 * remote_recover_mean_late_eff)  # Use effective
        + 0.5 * frac_reached
        - 1.0 * hard_penalty
        - 1.0 * soft_penalty
    )
    
    return {
        "block_size": int(block_size),
        "n_blocks": int(nB),
        "block_edges": int(len(bedges)),
        "source_block": int(source_block),
        
        # Bandwidth metrics (NEW)
        "bandwidth_capacity": float(bandwidth_capacity),
        "bandwidth_efficiency": float(bw_eff),
        "bandwidth_penalty": float(bw_pen),
        
        # Raw metrics (without bandwidth constraint)
        "frac_remote_recover_late_RAW": float(frac_remote_recover_late_raw),
        "remote_recover_mean_late_RAW": float(remote_recover_mean_late_raw),
        
        # Effective metrics (with bandwidth constraint)
        "frac_remote_recover_late_EFF": float(frac_remote_recover_late_eff),
        "remote_recover_mean_late_EFF": float(remote_recover_mean_late_eff),
        
        # Other metrics
        "d_remote_min": int(d_remote_min),
        "t_late_start": float(t_late_start),
        "speed_threshold": float(speed_threshold),
        "recover_threshold": float(recover_threshold),
        "frac_blocks_reached": float(frac_reached),
        "v_eff": float(v_eff),
        "instant_reach_blocks": int(instant_reach),
        "hard_penalty": float(hard_penalty),
        "soft_penalty": float(soft_penalty),
        "score": float(score),
        
        # Time series (for debugging)
        "times": [float(x) for x in times],
        "remote_best_raw": [float(x) for x in remote_best_raw],
        "remote_best_effective": [float(x) for x in remote_best_effective],
        "t_cross": [None if x is None else float(x) for x in t_cross],
        "dist_from_source": [int(x) for x in dist_from_source],
    }


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="HSF Constraint Test V3 with Finite Bandwidth")
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--topology", type=str, default="ring", help="ring|line|complete|rr|rr4|rr6...")
    ap.add_argument("--rr-deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--tmax", type=float, default=6.0)
    ap.add_argument("--nt", type=int, default=121)

    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--onsite-scale", type=float, default=0.25)
    ap.add_argument("--edge-scale", type=float, default=0.8)

    ap.add_argument("--blockings", type=str, default="1,2,3,4,6,12")
    ap.add_argument("--speed-threshold", type=float, default=0.02)
    ap.add_argument("--recover-threshold", type=float, default=0.03)

    ap.add_argument("--d-remote-min", type=int, default=2)
    ap.add_argument("--t-late-frac", type=float, default=0.4)
    ap.add_argument("--min-blocks", type=int, default=4)
    
    # NEW: Bandwidth parameters
    ap.add_argument("--bandwidth", type=float, default=1.5,
                    help="Bandwidth capacity per block (in qubit-equivalents). "
                         "Blocks larger than this have reduced efficiency.")
    ap.add_argument("--bandwidth-penalty-scale", type=float, default=1.0,
                    help="Scale factor for bandwidth excess penalty")
    ap.add_argument("--bandwidth-sweep", type=str, default=None,
                    help="Sweep bandwidth values, e.g. '0.5,1.0,1.5,2.0,3.0'")

    ap.add_argument("--out", type=str, default="results_constraints_v3.json")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    N = args.N
    if N < 2 or N > 14:
        print("NOTE: exact diagonalization; N>14 will likely be too slow/large.")

    rng = np.random.default_rng(args.seed)
    edges = build_edges(N, args.topology, rng, rr_deg=args.rr_deg)

    if args.progress:
        print("=" * 70)
        print("CONSTRAINT EMERGENCE TEST (V3) — WITH FINITE BANDWIDTH")
        print("=" * 70)
        print(f"N={N} topology={args.topology} edges={len(edges)} seed={args.seed}")
        print(f"t in [0,{args.tmax}] nt={args.nt}")
        print(f"blockings={args.blockings}")
        print(f"bandwidth_capacity={args.bandwidth} (key parameter!)")
        print(f"speed_threshold={args.speed_threshold} recover_threshold={args.recover_threshold}")
        print()

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

    psi_base = make_random_product_state(N, rng)
    times = np.linspace(0.0, float(args.tmax), int(args.nt), dtype=np.float64)

    blocking_sizes = [int(x.strip()) for x in args.blockings.split(",") if x.strip()]
    blocking_sizes = sorted(set(blocking_sizes))
    for b in blocking_sizes:
        if N % b != 0:
            raise ValueError(f"Blocking {b} does not divide N={N}")

    # Handle bandwidth sweep if requested
    if args.bandwidth_sweep:
        bandwidth_values = [float(x.strip()) for x in args.bandwidth_sweep.split(",")]
    else:
        bandwidth_values = [args.bandwidth]

    all_results = []
    
    for bw in bandwidth_values:
        if args.progress:
            print(f"\n--- Bandwidth = {bw} ---")
        
        results_this_bw = {
            "meta": {
                "N": int(N),
                "topology": args.topology,
                "rr_deg": int(args.rr_deg),
                "seed": int(args.seed),
                "tmax": float(args.tmax),
                "nt": int(args.nt),
                "source_qubit": int(args.source),
                "onsite_scale": float(args.onsite_scale),
                "edge_scale": float(args.edge_scale),
                "speed_threshold": float(args.speed_threshold),
                "recover_threshold": float(args.recover_threshold),
                "d_remote_min": int(args.d_remote_min),
                "t_late_frac": float(args.t_late_frac),
                "min_blocks": int(args.min_blocks),
                "bandwidth_capacity": float(bw),
                "bandwidth_penalty_scale": float(args.bandwidth_penalty_scale),
                "edges": edges,
            },
            "blockings": []
        }

        for b in blocking_sizes:
            if args.progress:
                eff = bandwidth_efficiency(b, bw)
                print(f"  block_size={b:>2} (n_blocks={N//b:>2}, efficiency={eff:.2f}) ...", end=" ", flush=True)
            
            out = evaluate_blocking_v3(
                N=N,
                qubit_edges=edges,
                evolver=evolver,
                psi_base=psi_base,
                source_qubit=int(args.source),
                times=times,
                block_size=b,
                speed_threshold=float(args.speed_threshold),
                recover_threshold=float(args.recover_threshold),
                d_remote_min=int(args.d_remote_min),
                t_late_frac=float(args.t_late_frac),
                min_blocks=int(args.min_blocks),
                bandwidth_capacity=float(bw),
                bandwidth_penalty_scale=float(args.bandwidth_penalty_scale),
            )
            results_this_bw["blockings"].append(out)
            
            if args.progress:
                print(f"score={out['score']:+.3f}")

        results_this_bw["blockings"] = sorted(
            results_this_bw["blockings"], key=lambda d: d["score"], reverse=True
        )
        results_this_bw["winner"] = results_this_bw["blockings"][0] if results_this_bw["blockings"] else None
        
        all_results.append(results_this_bw)

    # If single bandwidth, use simple format; otherwise use sweep format
    if len(bandwidth_values) == 1:
        final_results = all_results[0]
    else:
        final_results = {
            "sweep_type": "bandwidth",
            "bandwidth_values": bandwidth_values,
            "results_by_bandwidth": all_results,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    if args.progress:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        for res in all_results:
            bw = res["meta"]["bandwidth_capacity"]
            print(f"\nBandwidth = {bw}:")
            print(f"  {'Block':>6} {'#Blks':>6} {'Effic':>6} {'Score':>8} {'RecovRAW':>9} {'RecovEFF':>9} {'BWpen':>6}")
            print("  " + "-" * 60)
            
            for r in res["blockings"]:
                print(f"  {r['block_size']:>6} {r['n_blocks']:>6} {r['bandwidth_efficiency']:>6.2f} "
                      f"{r['score']:>+8.3f} {r['frac_remote_recover_late_RAW']:>9.3f} "
                      f"{r['frac_remote_recover_late_EFF']:>9.3f} {r['bandwidth_penalty']:>6.2f}")
            
            winner = res["winner"]
            if winner:
                if winner["block_size"] == 1:
                    verdict = "✓ SPATIAL WINS"
                elif winner["block_size"] == N:
                    verdict = "✗ TRIVIAL WINS"
                else:
                    verdict = f"? INTERMEDIATE (b={winner['block_size']})"
                print(f"\n  WINNER: block_size={winner['block_size']} → {verdict}")

        print(f"\nWrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())