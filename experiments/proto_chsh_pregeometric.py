#!/usr/bin/env python3
"""
proto_chsh_pregeometric.py

Numerical "proto-CHSH" / pre-geometric entanglement experiment.

Core idea:
  1) Choose a Hamiltonian (no background geometry assumed).
  2) Infer an interaction graph from the Hamiltonian's 2-body couplings.
  3) Define emergent distance as shortest-path distance on that graph.
  4) Evolve an initial state |psi(0)> under U(t)=exp(-i H t).
  5) For any chosen qubit pair (i,j), compute the *maximal* CHSH correlator value
     S_max(i,j,t) using the Horodecki criterion from the reduced 2-qubit density matrix.

IMPORTANT about "S_max":
  We use the standard CHSH correlator:
      S = E00 + E01 + E10 - E11
  with ±1 outcomes. For an uncorrelated maximally mixed 2-qubit state, E=0 so S=0.
  The classical (local hidden variable) bound is |S| <= 2.
  Violation means S_max > 2.

This script provides:
  - "best over time" summary by inferred distance (like before),
  - plus time-series tracking of specific pairs,
  - optional CSV output,
  - optional matplotlib plot.

Example (Windows):
  python proto_chsh_pregeometric.py --n 8 --topology ring --seed 1 --init distant_bell --tmax 2 --nt 81 --track-pairs 2,6 0,4 1,5 3,7 --plot

"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# ----------------------------
# Basic linear algebra helpers
# ----------------------------

def paulis() -> Dict[str, np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def kron_all(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def op_on_qubit(n: int, q: int, op: np.ndarray) -> np.ndarray:
    P = paulis()
    ops = [P["I"]] * n
    ops[q] = op
    return kron_all(ops)


def op_on_pair(n: int, i: int, j: int, op_i: np.ndarray, op_j: np.ndarray) -> np.ndarray:
    if i == j:
        raise ValueError("i and j must be distinct.")
    P = paulis()
    ops = [P["I"]] * n
    ops[i] = op_i
    ops[j] = op_j
    return kron_all(ops)


def ket0(n: int) -> np.ndarray:
    psi = np.zeros((2**n,), dtype=complex)
    psi[0] = 1.0 + 0j
    return psi


def ket_plus(n: int) -> np.ndarray:
    psi = np.ones((2**n,), dtype=complex)
    psi /= math.sqrt(2**n)
    return psi


def bell_phi_plus() -> np.ndarray:
    # |Φ+> = (|00>+|11>)/sqrt(2)
    v = np.zeros((4,), dtype=complex)
    v[0] = 1 / math.sqrt(2)
    v[3] = 1 / math.sqrt(2)
    return v


def build_bell_matching_state(n: int, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build |Φ+> Bell pairs on a perfect matching over qubits {0..n-1}.
    pairs must cover all qubits exactly once.
    """
    if n % 2 != 0:
        raise ValueError("Bell matching requires even n.")

    used = set()
    for a, b in pairs:
        if a == b:
            raise ValueError("Pair indices must differ.")
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError("Pair index out of range.")
        if a in used or b in used:
            raise ValueError("Pairs must be disjoint (perfect matching).")
        used.add(a); used.add(b)
    if len(used) != n:
        raise ValueError("Pairs must cover all qubits exactly once.")

    # Build tensor product of Bell states in the given pair order as adjacent logical qubits:
    # logical order: (p0a,p0b,p1a,p1b,...)
    phi = bell_phi_plus()
    psi = phi
    for _ in range((n // 2) - 1):
        psi = np.kron(psi, phi)

    # Permute logical axes into physical indices.
    logical_to_phys: List[int] = []
    for a, b in pairs:
        logical_to_phys.extend([a, b])

    # Construct inverse permutation: physical index -> logical axis
    inv = [None] * n
    for logical_axis, phys_index in enumerate(logical_to_phys):
        inv[phys_index] = logical_axis
    if any(v is None for v in inv):
        raise RuntimeError("Permutation construction failed.")

    psi_nd = psi.reshape([2] * n)
    psi_phys = np.transpose(psi_nd, axes=inv).reshape((2**n,))
    return psi_phys


# ----------------------------
# Reduced density matrices
# ----------------------------

def reduced_rho_two_qubits(psi: np.ndarray, n: int, a: int, b: int) -> np.ndarray:
    """
    2-qubit reduced density matrix ρ_ab from pure state |psi>.
    Output basis: |00>,|01>,|10>,|11> for (a,b).
    """
    if a == b:
        raise ValueError("a and b must be distinct.")

    swap = False
    if a > b:
        a, b = b, a
        swap = True

    psi_nd = psi.reshape([2] * n)
    axes = [a, b] + [k for k in range(n) if k not in (a, b)]
    psi_perm = np.transpose(psi_nd, axes=axes)
    env_dim = 2 ** (n - 2)
    psi_ab_env = psi_perm.reshape(4, env_dim)
    rho = psi_ab_env @ psi_ab_env.conj().T

    if swap:
        # swap qubit order back
        swap_perm = np.array([0, 2, 1, 3], dtype=int)
        rho = rho[np.ix_(swap_perm, swap_perm)]
    return rho


# ----------------------------
# Horodecki CHSH maximum
# ----------------------------

def chsh_max_horodecki(rho_2q: np.ndarray) -> float:
    """
    Maximal CHSH correlator value S_max for a 2-qubit state rho (Horodecki).
    Violation iff S_max > 2.
    """
    P = paulis()
    sigmas = [P["X"], P["Y"], P["Z"]]

    T = np.zeros((3, 3), dtype=float)
    for m in range(3):
        for n in range(3):
            op = np.kron(sigmas[m], sigmas[n])
            val = np.trace(rho_2q @ op)
            T[m, n] = float(np.real(val))

    M = T.T @ T
    evals = np.linalg.eigvalsh(M)  # ascending
    u1, u2 = float(evals[-1]), float(evals[-2])
    return 2.0 * math.sqrt(max(0.0, u1 + u2))


# ----------------------------
# Hamiltonian construction
# ----------------------------

@dataclass
class EdgeCoupling:
    i: int
    j: int
    Jx: float
    Jy: float
    Jz: float


def build_graph_edges(n: int, topology: str, rng: random.Random) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []

    if topology == "ring":
        for i in range(n):
            edges.append((i, (i + 1) % n))
    elif topology == "chain":
        for i in range(n - 1):
            edges.append((i, i + 1))
    elif topology == "rr3":
        # random 3-regular graph (simple rejection sampling)
        if n % 2 == 1:
            raise ValueError("rr3 requires even n.")
        for _attempt in range(2000):
            stubs = []
            for i in range(n):
                stubs += [i] * 3
            rng.shuffle(stubs)
            cand = []
            ok = True
            seen = set()
            for k in range(0, len(stubs), 2):
                a, b = stubs[k], stubs[k + 1]
                if a == b:
                    ok = False
                    break
                e = (a, b) if a < b else (b, a)
                if e in seen:
                    ok = False
                    break
                seen.add(e)
                cand.append(e)
            if ok:
                edges = cand
                break
        if not edges:
            raise RuntimeError("Failed to sample rr3 after many attempts.")
    elif topology == "random_sparse":
        # m ~ n edges
        m = n
        seen = set()
        while len(seen) < m:
            a, b = rng.randrange(n), rng.randrange(n)
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            seen.add(e)
        edges = sorted(seen)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # normalize, unique
    norm = []
    for a, b in edges:
        norm.append((a, b) if a < b else (b, a))
    return sorted(set(norm))


def build_hamiltonian(
    n: int,
    edges: List[Tuple[int, int]],
    seed: int,
    J_scale: float,
    h_scale: float,
) -> Tuple[np.ndarray, List[EdgeCoupling]]:
    rng = random.Random(seed)
    P = paulis()
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)

    edge_couplings: List[EdgeCoupling] = []
    for (i, j) in edges:
        Jx = rng.uniform(-J_scale, J_scale)
        Jy = rng.uniform(-J_scale, J_scale)
        Jz = rng.uniform(-J_scale, J_scale)
        edge_couplings.append(EdgeCoupling(i, j, Jx, Jy, Jz))

        H += Jx * op_on_pair(n, i, j, P["X"], P["X"])
        H += Jy * op_on_pair(n, i, j, P["Y"], P["Y"])
        H += Jz * op_on_pair(n, i, j, P["Z"], P["Z"])

    for i in range(n):
        hx = rng.uniform(-h_scale, h_scale)
        hy = rng.uniform(-h_scale, h_scale)
        hz = rng.uniform(-h_scale, h_scale)
        H += hx * op_on_qubit(n, i, P["X"])
        H += hy * op_on_qubit(n, i, P["Y"])
        H += hz * op_on_qubit(n, i, P["Z"])

    H = 0.5 * (H + H.conj().T)
    return H, edge_couplings


# ----------------------------
# Infer graph & distances
# ----------------------------

def infer_pair_weights(n: int, edge_couplings: List[EdgeCoupling]) -> np.ndarray:
    """
    "Inferred" interaction weights (toy model; in real life you'd learn these).
    W[i,j] = sqrt(Jx^2 + Jy^2 + Jz^2) for edges, else 0.
    """
    W = np.zeros((n, n), dtype=float)
    for ec in edge_couplings:
        w = math.sqrt(ec.Jx**2 + ec.Jy**2 + ec.Jz**2)
        W[ec.i, ec.j] = w
        W[ec.j, ec.i] = w
    return W


def build_unweighted_adj(W: np.ndarray, threshold: float) -> List[List[int]]:
    n = W.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > threshold:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def all_pairs_shortest_paths(adj: List[List[int]]) -> np.ndarray:
    n = len(adj)
    dist = np.full((n, n), fill_value=np.inf, dtype=float)
    for s in range(n):
        dist[s, s] = 0
        q = [s]
        head = 0
        while head < len(q):
            v = q[head]; head += 1
            for u in adj[v]:
                if dist[s, u] == np.inf:
                    dist[s, u] = dist[s, v] + 1
                    q.append(u)
    return dist


# ----------------------------
# Parsing helpers
# ----------------------------

def parse_pair_token(tok: str) -> Tuple[int, int]:
    tok = tok.strip()
    if "," not in tok:
        raise ValueError(f"Bad pair token '{tok}', expected i,j")
    a, b = tok.split(",", 1)
    return int(a.strip()), int(b.strip())


def parse_track_pairs(tokens: List[str]) -> List[Tuple[int, int]]:
    pairs = []
    for t in tokens:
        pairs.append(parse_pair_token(t))
    return pairs


# ----------------------------
# Experiment
# ----------------------------

def run_experiment(
    n: int,
    topology: str,
    seed: int,
    J_scale: float,
    h_scale: float,
    threshold: float,
    init: str,
    tmax: float,
    nt: int,
    max_pairs_per_dist: int,
    track_pairs: List[Tuple[int, int]],
    out_csv: Optional[str],
    do_plot: bool,
) -> None:
    rng = random.Random(seed)

    edges = build_graph_edges(n, topology, rng)
    H, edge_couplings = build_hamiltonian(n, edges, seed, J_scale, h_scale)

    # infer emergent interaction graph
    W = infer_pair_weights(n, edge_couplings)
    adj = build_unweighted_adj(W, threshold)
    dist = all_pairs_shortest_paths(adj)

    # choose initial state
    if init == "zero":
        psi0 = ket0(n)
    elif init == "plus":
        psi0 = ket_plus(n)
    elif init == "distant_bell":
        if n % 2 != 0:
            raise ValueError("distant_bell requires even n.")
        pairs = [(i, i + n // 2) for i in range(n // 2)]
        psi0 = build_bell_matching_state(n, pairs)
    else:
        raise ValueError("init must be one of: plus, zero, distant_bell")

    # diagonalize once for evolution
    w, V = np.linalg.eigh(H)

    def evolve_state(t: float) -> np.ndarray:
        ew = np.exp(-1j * w * t)
        U = (V * ew) @ V.conj().T
        return U @ psi0

    # time grid
    ts = np.linspace(0.0, tmax, nt)

    # pairs grouped by inferred distance (for best-over-time summary)
    pairs_by_d: Dict[int, List[Tuple[int, int]]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if math.isfinite(dist[i, j]):
                d = int(dist[i, j])
                pairs_by_d.setdefault(d, []).append((i, j))

    # limit samples for speed
    for d in pairs_by_d:
        rng.shuffle(pairs_by_d[d])
        pairs_by_d[d] = pairs_by_d[d][:max_pairs_per_dist]

    best: Dict[int, Tuple[float, float, Tuple[int, int]]] = {}  # d -> (Smax, t, pair)

    # tracking arrays
    track_pairs_norm = []
    for (a, b) in track_pairs:
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError(f"Bad track pair ({a},{b}) for n={n}")
        track_pairs_norm.append((a, b))

    track_S = np.zeros((len(ts), len(track_pairs_norm)), dtype=float)

    # main loop
    for ti, t in enumerate(ts):
        psi_t = evolve_state(float(t))

        # track time series
        for pi, (a, b) in enumerate(track_pairs_norm):
            rho = reduced_rho_two_qubits(psi_t, n, a, b)
            track_S[ti, pi] = chsh_max_horodecki(rho)

        # update best-by-distance
        for d, pairs in pairs_by_d.items():
            for (i, j) in pairs:
                rho = reduced_rho_two_qubits(psi_t, n, i, j)
                Smax = chsh_max_horodecki(rho)
                cur = best.get(d)
                if cur is None or Smax > cur[0]:
                    best[d] = (Smax, float(t), (i, j))

    # summary print
    print("\nPROTO-CHSH (pre-geometric) SIMULATION")
    print("------------------------------------")
    print(f"n={n} topology={topology} seed={seed}")
    print(f"J_scale={J_scale} h_scale={h_scale} threshold={threshold}")
    print(f"init_state={init}   t in [0,{tmax}] with nt={nt}")

    degs = [len(adj[i]) for i in range(n)]
    disconnected = any(not math.isfinite(dist[0, j]) for j in range(n))
    print("\nInferred interaction graph:")
    print(f"  degrees: min={min(degs)} max={max(degs)} mean={sum(degs)/len(degs):.2f}")
    print(f"  connected: {'NO (some distances are inf)' if disconnected else 'yes'}")

    print("\nBest CHSH by inferred distance d (Horodecki S_max):")
    print("  (CHSH correlator: local bound 2.0000, Tsirelson 2.8284; uncorrelated can be ~0)\n")
    for d in sorted(best.keys()):
        Smax, t_best, (i, j) = best[d]
        flag = "VIOLATION" if Smax > 2.0 + 1e-6 else ""
        print(f"  d={d:2d}  best S_max={Smax:0.4f}  at t={t_best:0.3f}  pair=({i},{j})  {flag}")

    if track_pairs_norm:
        print("\nTracked pair time series (S_max):")
        header = "t" + "".join([f", S({a},{b})" for (a, b) in track_pairs_norm])
        print(header)
        # Print a compact sample: first, middle, last, unless nt is small
        if nt <= 15:
            rows = range(nt)
        else:
            rows = sorted(set([0, 1, 2, nt//2, nt-3, nt-2, nt-1]))
        for k in rows:
            row = f"{ts[k]:0.4f}" + "".join([f", {track_S[k, pi]:0.4f}" for pi in range(len(track_pairs_norm))])
            print(row)

    # CSV output (full time series)
    if out_csv:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("t")
            for (a, b) in track_pairs_norm:
                f.write(f",S_{a}_{b}")
            f.write("\n")
            for ti, t in enumerate(ts):
                f.write(f"{t:.10f}")
                for pi in range(len(track_pairs_norm)):
                    f.write(f",{track_S[ti, pi]:.10f}")
                f.write("\n")
        print(f"\nWrote CSV: {out_csv}")

    # plot
    if do_plot and track_pairs_norm:
        import matplotlib.pyplot as plt  # allowed
        plt.figure()
        for pi, (a, b) in enumerate(track_pairs_norm):
            plt.plot(ts, track_S[:, pi], label=f"S_max({a},{b})")
        plt.axhline(2.0, linestyle="--")
        plt.xlabel("t")
        plt.ylabel("S_max (CHSH correlator)")
        plt.legend()
        plt.title("Tracked CHSH S_max vs time")
        plt.show()
    elif do_plot and not track_pairs_norm:
        print("\n--plot was set, but no --track-pairs were provided; nothing to plot.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="Number of qubits (recommend <=10 for speed)")
    ap.add_argument("--topology", type=str, default="ring",
                    choices=["ring", "chain", "rr3", "random_sparse"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--J-scale", type=float, default=1.0)
    ap.add_argument("--h-scale", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.0, help="Edge threshold for inferred graph")
    ap.add_argument("--init", type=str, default="plus", choices=["plus", "zero", "distant_bell"])
    ap.add_argument("--tmax", type=float, default=2.0)
    ap.add_argument("--nt", type=int, default=41)
    ap.add_argument("--max-pairs-per-dist", type=int, default=64,
                    help="Speed knob: sample up to this many pairs per distance for best-by-d summary")
    ap.add_argument("--track-pairs", nargs="*", default=[],
                    help="Pairs to track as time series, e.g. --track-pairs 2,6 0,4 1,5")
    ap.add_argument("--out-csv", type=str, default="",
                    help="Write tracked time series to CSV (requires --track-pairs).")
    ap.add_argument("--plot", action="store_true", help="Plot tracked time series (requires --track-pairs).")

    args = ap.parse_args()

    if args.n > 10:
        print("WARNING: n>10 will be slow (2^n state + 2^n x 2^n Hamiltonian). Consider n<=10.")

    track_pairs = parse_track_pairs(args.track_pairs) if args.track_pairs else []
    out_csv = args.out_csv.strip() or None

    run_experiment(
        n=args.n,
        topology=args.topology,
        seed=args.seed,
        J_scale=args.J_scale,
        h_scale=args.h_scale,
        threshold=args.threshold,
        init=args.init,
        tmax=args.tmax,
        nt=args.nt,
        max_pairs_per_dist=args.max_pairs_per_dist,
        track_pairs=track_pairs,
        out_csv=out_csv,
        do_plot=args.plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
