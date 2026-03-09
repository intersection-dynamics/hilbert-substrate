# hsf_big_bang_refinement.py
# ------------------------------------------------------------
# HSF Big Bang Refinement (pressure-driven splitting)
#
# Goals:
#   - Start with ONE subsystem (Hilbert lump) with d0 internal basis states.
#   - Apply the four constraints from the beginning:
#       1) no-signaling: only local (node-internal) or edge-local updates
#       2) finite bandwidth: caps on internal update magnitude and edge transport per tick
#       3) no-forgetting: unperformed update demand (residual) is recorded as memory
#       4) no-refolding: when memory is high, links "lock" and stop updating gauge registers
#
# Big Bang mechanism:
#   - Each subsystem attempts an internal "unitary-like" mixing step each tick.
#   - Bandwidth caps how much change can be applied in one tick.
#   - Residual demand (what couldn't be applied) is recorded by no-forgetting.
#   - When a subsystem's internal memory exceeds a split threshold,
#     the subsystem splits into two subsystems.
#   - The split creates a new link (gauge register) between the two children.
#
# After multiple subsystems exist:
#   - Edge-local matter exchange happens (bandwidth limited), writing edge memory.
#   - Gauge phase theta on each edge updates from a local current proxy unless locked.
#   - Edges lock (no-refolding) when edge memory exceeds lock_threshold.
#
# Outputs (non-overwriting):
#   out_dir/<timestamp>_<tag>/
#     log.csv
#     results.npz
#     plots/
#       timeseries.png
#       graph_summary.png
#       node_dim_hist_end.png
#
# Example (Windows one-liner):
#   python hsf_big_bang_refinement.py --d0 64 --T 6000 --B_int 0.06 --B_edge 0.02 --mem_couple_int 2.0 --split_threshold 6.0 --N_max 512
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

TAU = 2.0 * np.pi


# -------------------------
# utils
# -------------------------

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_tag(s: str) -> str:
    return s.replace(" ", "_").replace(":", "_").replace("/", "_").replace(".", "p")


def make_run_dir(base_out: str, tag: str) -> str:
    run_dir = os.path.join(base_out, f"{now_stamp()}_{safe_tag(tag)}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    return run_dir


def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    cum = np.cumsum(xs)
    g = (n + 1.0 - 2.0 * float(np.sum(cum) / cum[-1])) / n
    return float(max(0.0, min(1.0, g)))


def top_fraction(x: np.ndarray, k: int = 10) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    s = float(x.sum())
    if s <= 0 or x.size == 0:
        return 0.0
    k = min(int(k), x.size)
    return float(np.sort(x)[-k:].sum() / s)


# -------------------------
# data structures
# -------------------------

@dataclass
class Node:
    psi: np.ndarray          # complex state vector (internal basis states)
    mem_int: float           # internal no-forgetting ledger (residual demand)
    age: int                 # ticks since creation


@dataclass
class Edge:
    a: int
    b: int
    theta: float             # gauge register (U(1) phase)
    mem_e: float             # no-forgetting on the link
    locked: bool             # no-refolding lock state


@dataclass
class Params:
    d0: int
    T: int
    seed: int

    # Internal evolution (big bang pressure)
    eps_int: float           # attempted internal mixing strength (per tick)
    B_int: float             # internal bandwidth cap on ||delta psi|| per tick
    mem_decay_int: float
    mem_couple_int: float
    split_threshold: float   # split when mem_int exceeds this
    min_dim_to_split: int    # do not split below this dimension

    # Edge dynamics (after multiple nodes exist)
    B_edge: float            # bandwidth cap on edge transport (matter)
    alpha_edge: float        # edge matter exchange strength
    kappa_edge: float        # edge memory throttles flow
    g_coup: float            # gauge response to current
    B_theta: float           # bandwidth cap on |dtheta| per tick

    mem_decay_edge: float
    mem_couple_edge: float
    lock_threshold: float

    # Graph growth
    N_max: int
    attach_k: int            # when a new node is created, attempt to attach up to k edges to existing nodes

    # Output
    out_dir: str
    run_name: str


# -------------------------
# core mechanics
# -------------------------

def attempted_internal_mix(rng: np.random.Generator, psi: np.ndarray, eps: float) -> Tuple[np.ndarray, float]:
    """
    Produce a small unitary-like update direction and return (psi_new_unscaled, delta_norm).
    We keep it simple: propose a random orthogonal-to-psi direction and step along it.
    """
    d = psi.size
    z = (rng.normal(size=d) + 1j * rng.normal(size=d)).astype(np.complex128)
    # project out component along psi to keep it "rotation-like"
    proj = (np.vdot(psi, z) / (np.vdot(psi, psi) + 1e-18)) * psi
    z = z - proj
    nz = np.sqrt(np.vdot(z, z).real + 1e-18)
    z = z / nz
    psi_prop = psi + eps * z
    psi_prop /= np.sqrt(np.vdot(psi_prop, psi_prop).real + 1e-18)
    delta = psi_prop - psi
    delta_norm = float(np.sqrt(np.vdot(delta, delta).real + 1e-18))
    return psi_prop, delta_norm


def apply_bandwidth_to_internal_update(psi: np.ndarray, psi_prop: np.ndarray, delta_norm: float, B_int: float) -> Tuple[np.ndarray, float]:
    """
    Apply at most B_int of change; record residual demand.
    Returns (psi_new, residual).
    """
    if delta_norm <= B_int:
        return psi_prop, 0.0
    # scale delta to B_int
    delta = psi_prop - psi
    psi_new = psi + (B_int / (delta_norm + 1e-18)) * delta
    psi_new /= np.sqrt(np.vdot(psi_new, psi_new).real + 1e-18)
    residual = float(delta_norm - B_int)
    return psi_new, residual


def split_node(nodes: List[Node], edges: List[Edge], idx_node: int, rng: np.random.Generator, p: Params) -> int:
    """
    Split node into two nodes by partitioning its internal basis.
    Creates a new edge (gauge register) between children.
    Returns index of newly created node (second child).
    """
    parent = nodes[idx_node]
    d = parent.psi.size
    # choose a split point near half
    d1 = d // 2
    d2 = d - d1
    if d1 < 1 or d2 < 1:
        return -1

    psi1 = parent.psi[:d1].copy()
    psi2 = parent.psi[d1:].copy()

    # If either child is ~zero, add tiny noise (still no "particle seeding", just numerical hygiene)
    if np.vdot(psi1, psi1).real < 1e-12:
        psi1 = (rng.normal(size=d1) + 1j * rng.normal(size=d1)).astype(np.complex128) * 1e-6
    if np.vdot(psi2, psi2).real < 1e-12:
        psi2 = (rng.normal(size=d2) + 1j * rng.normal(size=d2)).astype(np.complex128) * 1e-6

    psi1 /= np.sqrt(np.vdot(psi1, psi1).real + 1e-18)
    psi2 /= np.sqrt(np.vdot(psi2, psi2).real + 1e-18)

    # Replace parent with child1, append child2
    nodes[idx_node] = Node(psi=psi1, mem_int=parent.mem_int * 0.5, age=0)
    nodes.append(Node(psi=psi2, mem_int=parent.mem_int * 0.5, age=0))
    j = len(nodes) - 1

    # Create a link (gauge register) between children
    edges.append(Edge(a=idx_node, b=j, theta=float(rng.uniform(-np.pi, np.pi)), mem_e=0.0, locked=False))

    # Optionally attach new node to a few existing nodes (pressure-driven "branching")
    # This is not seeding excitations; it's creating mediation capacity for no-signaling/bandwidth.
    if p.attach_k > 0:
        existing = list(range(len(nodes)))
        existing.remove(j)
        rng.shuffle(existing)
        attach = existing[: min(p.attach_k, len(existing))]
        for k in attach:
            if k == idx_node:
                continue
            edges.append(Edge(a=j, b=k, theta=float(rng.uniform(-np.pi, np.pi)), mem_e=0.0, locked=False))

    return j


def edge_step(nodes: List[Node], edges: List[Edge], rng: np.random.Generator, p: Params) -> Tuple[float, float]:
    """
    One edge-local update pass over all edges (no-signaling, bandwidth-limited).
    Uses only the first component of each node (a proxy "addressable mode"),
    and internal mixing will spread influence through the node over time.

    Returns:
      total_flow, total_dtheta
    """
    total_flow = 0.0
    total_dtheta = 0.0

    for e in edges:
        a = e.a
        b = e.b
        if a >= len(nodes) or b >= len(nodes):
            continue
        na = nodes[a]
        nb = nodes[b]
        if na.psi.size == 0 or nb.psi.size == 0:
            continue

        # Select an "interface mode" in each node: component 0
        xa = na.psi[0]
        xb = nb.psi[0]

        phase = np.exp(1j * e.theta)

        # gauge-covariant gradient on the interface
        grad = xa - phase * xb

        # memory throttling on flow (finite processing / congestion)
        throttle = np.exp(-p.kappa_edge * e.mem_e) if p.kappa_edge > 0 else 1.0

        flow = p.alpha_edge * throttle * grad
        mag = float(np.abs(flow))
        if mag > p.B_edge:
            flow *= (p.B_edge / (mag + 1e-18))
            mag = p.B_edge

        # apply flow to interface modes
        na.psi[0] = xa - flow
        nb.psi[0] = xb + np.conjugate(phase) * flow

        # renormalize each node (proxy for local unitarity / norm preservation)
        na.psi /= np.sqrt(np.vdot(na.psi, na.psi).real + 1e-18)
        nb.psi /= np.sqrt(np.vdot(nb.psi, nb.psi).real + 1e-18)

        total_flow += mag

        # theta update from local current proxy unless locked
        dtheta = 0.0
        if not e.locked:
            xa2 = na.psi[0]
            xb2 = nb.psi[0]
            phase2 = np.exp(1j * e.theta)
            j = float(np.imag(np.conjugate(xa2) * phase2 * xb2))
            dtheta = p.g_coup * j
            dabs = abs(dtheta)
            if dabs > p.B_theta:
                dtheta *= (p.B_theta / (dabs + 1e-18))
                dabs = p.B_theta
            e.theta = float(wrap_pi(np.array([e.theta + dtheta]))[0])
            total_dtheta += dabs

        # no-forgetting on the edge
        transported = mag + 1.0 * abs(dtheta)
        e.mem_e = (1.0 - p.mem_decay_edge) * e.mem_e + p.mem_couple_edge * transported

        # no-refolding lock
        if (not e.locked) and (e.mem_e >= p.lock_threshold):
            e.locked = True

    return total_flow, total_dtheta


# -------------------------
# simulation driver
# -------------------------

def run_sim(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)

    # Start with ONE subsystem
    psi0 = (rng.normal(size=p.d0) + 1j * rng.normal(size=p.d0)).astype(np.complex128)
    psi0 /= np.sqrt(np.vdot(psi0, psi0).real + 1e-18)

    nodes: List[Node] = [Node(psi=psi0, mem_int=0.0, age=0)]
    edges: List[Edge] = []

    # Logs
    T = p.T
    N_t = np.zeros(T, dtype=np.int32)
    E_t = np.zeros(T, dtype=np.int32)
    avg_deg_t = np.zeros(T, dtype=np.float64)

    splits_t = np.zeros(T, dtype=np.int32)
    max_mem_int_t = np.zeros(T, dtype=np.float64)
    mean_mem_int_t = np.zeros(T, dtype=np.float64)

    total_edge_mem_t = np.zeros(T, dtype=np.float64)
    locked_frac_t = np.zeros(T, dtype=np.float64)

    flow_t = np.zeros(T, dtype=np.float64)
    dtheta_t = np.zeros(T, dtype=np.float64)

    g_edge_mem_t = np.zeros(T, dtype=np.float64)
    top_edge_mem_t = np.zeros(T, dtype=np.float64)

    # main loop
    for t in range(T):
        # internal mixing + pressure ledger
        split_count = 0

        # age nodes
        for n in nodes:
            n.age += 1

        # internal update for each node
        # (iterate over a snapshot of indices because list can grow via splits)
        i = 0
        while i < len(nodes):
            n = nodes[i]
            d = n.psi.size

            # attempt internal mixing (unitary-like)
            psi_prop, delta_norm = attempted_internal_mix(rng, n.psi, p.eps_int)
            psi_new, residual = apply_bandwidth_to_internal_update(n.psi, psi_prop, delta_norm, p.B_int)
            n.psi = psi_new

            # no-forgetting internal ledger records residual demand
            n.mem_int = (1.0 - p.mem_decay_int) * n.mem_int + p.mem_couple_int * residual

            # pressure-driven split criterion (bandwidth + no-forgetting)
            if (len(nodes) < p.N_max) and (d >= p.min_dim_to_split) and (n.mem_int >= p.split_threshold):
                # split creates link register automatically
                new_idx = split_node(nodes, edges, i, rng, p)
                if new_idx >= 0:
                    split_count += 1
                    # after splitting, keep scanning (children may also be high-pressure)
                    # but avoid infinite splitting cascades in a single tick by damping ledger
                    nodes[i].mem_int *= 0.25
                    nodes[new_idx].mem_int *= 0.25
            i += 1

        # edge-local interactions once we have at least 2 nodes and some edges
        total_flow, total_dtheta = (0.0, 0.0)
        if len(edges) > 0 and len(nodes) > 1:
            total_flow, total_dtheta = edge_step(nodes, edges, rng, p)

        # Metrics
        N = len(nodes)
        E = len(edges)
        N_t[t] = N
        E_t[t] = E
        avg_deg_t[t] = float(2.0 * E / max(N, 1))
        splits_t[t] = split_count
        flow_t[t] = total_flow
        dtheta_t[t] = total_dtheta

        mem_ints = np.array([n.mem_int for n in nodes], dtype=np.float64)
        max_mem_int_t[t] = float(mem_ints.max()) if mem_ints.size else 0.0
        mean_mem_int_t[t] = float(mem_ints.mean()) if mem_ints.size else 0.0

        if E > 0:
            mem_e = np.array([e.mem_e for e in edges], dtype=np.float64)
            total_edge_mem_t[t] = float(mem_e.sum())
            locked_frac_t[t] = float(np.mean([1.0 if e.locked else 0.0 for e in edges]))
            g_edge_mem_t[t] = float(gini(mem_e))
            top_edge_mem_t[t] = float(top_fraction(mem_e, 10))
        else:
            total_edge_mem_t[t] = 0.0
            locked_frac_t[t] = 0.0
            g_edge_mem_t[t] = 0.0
            top_edge_mem_t[t] = 0.0

        # stop early if we've saturated N_max and internal pressure is low
        if (N >= p.N_max) and (max_mem_int_t[t] < 0.1 * p.split_threshold) and (t > 200):
            # fill remaining logs with last values
            for k in range(t + 1, T):
                N_t[k] = N_t[t]
                E_t[k] = E_t[t]
                avg_deg_t[k] = avg_deg_t[t]
                splits_t[k] = 0
                max_mem_int_t[k] = max_mem_int_t[t]
                mean_mem_int_t[k] = mean_mem_int_t[t]
                total_edge_mem_t[k] = total_edge_mem_t[t]
                locked_frac_t[k] = locked_frac_t[t]
                flow_t[k] = 0.0
                dtheta_t[k] = 0.0
                g_edge_mem_t[k] = g_edge_mem_t[t]
                top_edge_mem_t[k] = top_edge_mem_t[t]
            break

    # pack results
    dims = np.array([n.psi.size for n in nodes], dtype=np.int32)
    mem_ints_end = np.array([n.mem_int for n in nodes], dtype=np.float64)

    u = np.array([e.a for e in edges], dtype=np.int32) if edges else np.empty(0, dtype=np.int32)
    v = np.array([e.b for e in edges], dtype=np.int32) if edges else np.empty(0, dtype=np.int32)
    theta = np.array([e.theta for e in edges], dtype=np.float64) if edges else np.empty(0, dtype=np.float64)
    mem_e = np.array([e.mem_e for e in edges], dtype=np.float64) if edges else np.empty(0, dtype=np.float64)
    locked = np.array([1 if e.locked else 0 for e in edges], dtype=np.int8) if edges else np.empty(0, dtype=np.int8)

    return {
        "N_t": N_t,
        "E_t": E_t,
        "avg_deg_t": avg_deg_t,
        "splits_t": splits_t,
        "max_mem_int_t": max_mem_int_t,
        "mean_mem_int_t": mean_mem_int_t,
        "total_edge_mem_t": total_edge_mem_t,
        "locked_frac_t": locked_frac_t,
        "flow_t": flow_t,
        "dtheta_t": dtheta_t,
        "g_edge_mem_t": g_edge_mem_t,
        "top_edge_mem_t": top_edge_mem_t,

        "dims_end": dims,
        "mem_int_end": mem_ints_end,

        "u": u, "v": v,
        "theta_end": theta,
        "mem_e_end": mem_e,
        "locked_end": locked,
    }


# -------------------------
# IO + plots
# -------------------------

def write_csv(path: str, data: dict) -> None:
    T = len(data["N_t"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "N",
            "E",
            "avg_deg",
            "splits",
            "max_mem_int",
            "mean_mem_int",
            "total_edge_mem",
            "locked_frac",
            "flow_total",
            "dtheta_total",
            "gini_edge_mem",
            "top10_edge_mem",
        ])
        for t in range(T):
            w.writerow([
                t,
                int(data["N_t"][t]),
                int(data["E_t"][t]),
                float(data["avg_deg_t"][t]),
                int(data["splits_t"][t]),
                float(data["max_mem_int_t"][t]),
                float(data["mean_mem_int_t"][t]),
                float(data["total_edge_mem_t"][t]),
                float(data["locked_frac_t"][t]),
                float(data["flow_t"][t]),
                float(data["dtheta_t"][t]),
                float(data["g_edge_mem_t"][t]),
                float(data["top_edge_mem_t"][t]),
            ])


def plot_outputs(run_dir: str, data: dict) -> None:
    plots_dir = os.path.join(run_dir, "plots")
    t = np.arange(len(data["N_t"]))

    # timeseries
    plt.figure()
    plt.plot(t, data["N_t"], label="N(t) subsystems")
    plt.plot(t, data["E_t"], label="E(t) links")
    plt.plot(t, data["splits_t"], label="splits/tick")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "timeseries.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # graph + memory summary
    plt.figure()
    plt.plot(t, data["avg_deg_t"], label="avg degree")
    plt.plot(t, data["locked_frac_t"], label="locked fraction")
    plt.plot(t, data["max_mem_int_t"], label="max internal memory")
    plt.plot(t, data["total_edge_mem_t"], label="total edge memory")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "graph_summary.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # end dimension histogram
    dims = data["dims_end"]
    if dims.size > 0:
        plt.figure()
        plt.hist(dims, bins=min(30, max(5, int(np.sqrt(dims.size)))))
        plt.xlabel("node internal dimension (d_i)")
        plt.ylabel("count")
        plt.savefig(os.path.join(plots_dir, "node_dim_hist_end.png"), dpi=160, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()

    # Big bang initial condition
    ap.add_argument("--d0", type=int, default=64, help="initial single-subsystem Hilbert dimension")
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)

    # Internal (pressure-driven) evolution
    ap.add_argument("--eps_int", type=float, default=0.25, help="attempted internal mixing strength")
    ap.add_argument("--B_int", type=float, default=0.06, help="internal bandwidth cap on ||delta psi|| per tick")
    ap.add_argument("--mem_decay_int", type=float, default=0.0005)
    ap.add_argument("--mem_couple_int", type=float, default=2.0)
    ap.add_argument("--split_threshold", type=float, default=6.0)
    ap.add_argument("--min_dim_to_split", type=int, default=8)

    # Edge dynamics (links are gauge registers)
    ap.add_argument("--B_edge", type=float, default=0.02)
    ap.add_argument("--alpha_edge", type=float, default=0.14)
    ap.add_argument("--kappa_edge", type=float, default=6.0)
    ap.add_argument("--g_coup", type=float, default=0.20)
    ap.add_argument("--B_theta", type=float, default=0.12)

    ap.add_argument("--mem_decay_edge", type=float, default=0.0002)
    ap.add_argument("--mem_couple_edge", type=float, default=3.0)
    ap.add_argument("--lock_threshold", type=float, default=0.25)

    # Growth / branching
    ap.add_argument("--N_max", type=int, default=512)
    ap.add_argument("--attach_k", type=int, default=2, help="edges added from newborn node to existing nodes")

    # output
    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="bigbang")

    args = ap.parse_args()

    tag = (
        f"d0{args.d0}_T{args.T}"
        f"_Bint{args.B_int}_eps{args.eps_int}"
        f"_mcI{args.mem_couple_int}_split{args.split_threshold}_mind{args.min_dim_to_split}"
        f"_Bedge{args.B_edge}_aE{args.alpha_edge}_kE{args.kappa_edge}"
        f"_mcE{args.mem_couple_edge}_lock{args.lock_threshold}"
        f"_Nmax{args.N_max}_attach{args.attach_k}_seed{args.seed}_{args.run_name}"
    )
    run_dir = make_run_dir(args.out_dir, tag)

    p = Params(
        d0=args.d0,
        T=args.T,
        seed=args.seed,

        eps_int=args.eps_int,
        B_int=args.B_int,
        mem_decay_int=args.mem_decay_int,
        mem_couple_int=args.mem_couple_int,
        split_threshold=args.split_threshold,
        min_dim_to_split=args.min_dim_to_split,

        B_edge=args.B_edge,
        alpha_edge=args.alpha_edge,
        kappa_edge=args.kappa_edge,
        g_coup=args.g_coup,
        B_theta=args.B_theta,

        mem_decay_edge=args.mem_decay_edge,
        mem_couple_edge=args.mem_couple_edge,
        lock_threshold=args.lock_threshold,

        N_max=args.N_max,
        attach_k=args.attach_k,

        out_dir=args.out_dir,
        run_name=args.run_name,
    )

    data = run_sim(p)

    npz_path = os.path.join(run_dir, "results.npz")
    np.savez_compressed(npz_path, **data)

    csv_path = os.path.join(run_dir, "log.csv")
    write_csv(csv_path, data)

    plot_outputs(run_dir, data)

    print("Run output:", run_dir)
    print("Saved:", csv_path)
    print("Saved:", npz_path)
    print("Plots:", os.path.join(run_dir, "plots"))


if __name__ == "__main__":
    main()
