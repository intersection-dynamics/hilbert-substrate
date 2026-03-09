# hsf_big_bang_refinement_v2.py
# ------------------------------------------------------------
# HSF Big Bang Refinement v2 (pressure-driven splitting + pressure venting + lock budget)
#
# What v2 fixes vs v1:
#   1) Prevent total freeze: lock budget (target locked fraction)
#      - no-refolding still holds: locked edges never unlock
#      - but we stop locking NEW edges once locked_frac >= lock_target
#
#   2) Prevent runaway internal pressure at min factor size:
#      - When a node cannot split further (dim <= min_dim_to_split),
#        and internal memory continues to exceed split_threshold,
#        we "vent" pressure by nucleating new mediation channels (links).
#      - This creates new gauge registers (edge phase theta) as a CONSEQUENCE,
#        and transfers part of internal ledger into the link memory (no-forgetting becomes structure).
#
# Model summary:
#   - Start with ONE subsystem (Hilbert lump) dimension d0.
#   - Each tick, each node attempts a unitary-like internal mix step.
#   - Bandwidth B_int caps applied change; residual is recorded as mem_int (no-forgetting).
#   - If mem_int > split_threshold and dim >= 2*min_dim_to_split: split node into two factors.
#   - Else if mem_int > split_threshold and dim <= 2*min_dim_to_split: vent by adding new links.
#   - Edge-local transport happens on links (bandwidth-limited), writing edge memory.
#   - Edge gauge phase theta updates from local current proxy unless edge locked.
#   - Edge locks when edge memory exceeds lock_threshold AND lock budget not exhausted.
#
# Outputs (non-overwriting):
#   out_dir/<timestamp>_<tag>/
#     log.csv
#     results.npz
#     plots/
#       timeseries.png
#       pressure_summary.png
#       graph_summary.png
#       node_dim_hist_end.png
#
# Example (Windows one-liner):
#   python hsf_big_bang_refinement_v2.py --d0 64 --T 8000 --B_int 0.06 --split_threshold 6.0 --lock_target 0.25 --N_max 512 --run_name bang_v2
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


def wrap_pi_scalar(x: float) -> float:
    # map to (-pi, pi]
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


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
    psi: np.ndarray          # complex state vector (internal basis)
    mem_int: float           # internal ledger (unmet change demand)
    age: int                 # ticks since creation


@dataclass
class Edge:
    a: int
    b: int
    theta: float             # gauge register (U(1) phase)
    mem_e: float             # edge ledger (no-forgetting)
    locked: bool             # no-refolding lock (irreversible)


@dataclass
class Params:
    d0: int
    T: int
    seed: int

    # internal evolution
    eps_int: float
    B_int: float
    mem_decay_int: float
    mem_couple_int: float
    split_threshold: float
    min_dim_to_split: int

    # venting (pressure -> new mediation)
    vent_share: float           # fraction of mem_int transferred into new edge memory when venting
    vent_edges_per_event: int   # how many edges to add per vent event
    vent_cooldown: int          # minimum ticks between vent events per node
    vent_attach_mode: str       # "random" or "low_degree"

    # edge dynamics
    B_edge: float
    alpha_edge: float
    kappa_edge: float
    g_coup: float
    B_theta: float

    mem_decay_edge: float
    mem_couple_edge: float

    lock_threshold: float
    lock_target: float      # stop locking new edges once locked_frac reaches this
    lock_start: int         # don't lock before this tick

    # growth
    N_max: int
    attach_k_split: int     # edges added from newborn split-child to existing nodes

    # output
    out_dir: str
    run_name: str


# -------------------------
# core mechanics
# -------------------------

def attempted_internal_mix(rng: np.random.Generator, psi: np.ndarray, eps: float) -> Tuple[np.ndarray, float]:
    """
    Produce a small unitary-like update direction.
    """
    d = psi.size
    z = (rng.normal(size=d) + 1j * rng.normal(size=d)).astype(np.complex128)

    # project out component along psi
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
    Apply at most B_int of change; residual is unmet demand.
    """
    if delta_norm <= B_int:
        return psi_prop, 0.0
    delta = psi_prop - psi
    psi_new = psi + (B_int / (delta_norm + 1e-18)) * delta
    psi_new /= np.sqrt(np.vdot(psi_new, psi_new).real + 1e-18)
    residual = float(delta_norm - B_int)
    return psi_new, residual


def node_degree(n: int, edges: List[Edge]) -> int:
    deg = 0
    for e in edges:
        if e.a == n or e.b == n:
            deg += 1
    return deg


def has_edge(a: int, b: int, edges: List[Edge]) -> bool:
    if a == b:
        return True
    lo, hi = (a, b) if a < b else (b, a)
    for e in edges:
        lo2, hi2 = (e.a, e.b) if e.a < e.b else (e.b, e.a)
        if lo == lo2 and hi == hi2:
            return True
    return False


def add_edge(edges: List[Edge], a: int, b: int, rng: np.random.Generator, init_mem: float = 0.0) -> None:
    if a == b:
        return
    if has_edge(a, b, edges):
        return
    edges.append(
        Edge(
            a=a,
            b=b,
            theta=float(rng.uniform(-np.pi, np.pi)),
            mem_e=float(max(0.0, init_mem)),
            locked=False,
        )
    )


def split_node(nodes: List[Node], edges: List[Edge], idx_node: int, rng: np.random.Generator, p: Params) -> int:
    """
    Split a node by partitioning its internal basis.
    Creates a new link (gauge register) between children.
    Returns index of the newly created node.
    """
    parent = nodes[idx_node]
    d = parent.psi.size
    d1 = d // 2
    d2 = d - d1
    if d1 < 1 or d2 < 1:
        return -1

    psi1 = parent.psi[:d1].copy()
    psi2 = parent.psi[d1:].copy()

    # numerical hygiene only
    if np.vdot(psi1, psi1).real < 1e-12:
        psi1 = (rng.normal(size=d1) + 1j * rng.normal(size=d1)).astype(np.complex128) * 1e-6
    if np.vdot(psi2, psi2).real < 1e-12:
        psi2 = (rng.normal(size=d2) + 1j * rng.normal(size=d2)).astype(np.complex128) * 1e-6

    psi1 /= np.sqrt(np.vdot(psi1, psi1).real + 1e-18)
    psi2 /= np.sqrt(np.vdot(psi2, psi2).real + 1e-18)

    # replace parent, append child
    nodes[idx_node] = Node(psi=psi1, mem_int=parent.mem_int * 0.5, age=0)
    nodes.append(Node(psi=psi2, mem_int=parent.mem_int * 0.5, age=0))
    j = len(nodes) - 1

    # link between children
    add_edge(edges, idx_node, j, rng, init_mem=0.0)

    # attach child to a few existing nodes to increase mediation capacity
    if p.attach_k_split > 0:
        existing = list(range(len(nodes)))
        existing.remove(j)
        rng.shuffle(existing)
        for k in existing[: min(p.attach_k_split, len(existing))]:
            if k == idx_node:
                continue
            add_edge(edges, j, k, rng, init_mem=0.0)

    # damp ledgers (avoid same-tick cascades)
    nodes[idx_node].mem_int *= 0.25
    nodes[j].mem_int *= 0.25

    return j


def choose_vent_targets(nodes: List[Node], edges: List[Edge], src: int, rng: np.random.Generator, k: int, mode: str) -> List[int]:
    """
    Choose k distinct target nodes to connect to src.
    mode:
      - "random": uniform from other nodes
      - "low_degree": prefer lowest degree nodes (helps spread)
    """
    N = len(nodes)
    if N <= 1:
        return []
    candidates = [i for i in range(N) if i != src]
    if not candidates:
        return []

    if mode == "low_degree":
        degs = [(node_degree(i, edges), i) for i in candidates]
        degs.sort(key=lambda x: x[0])
        # take a pool of low-degree nodes then shuffle
        pool = [i for _, i in degs[: max(k * 4, k)]]
        rng.shuffle(pool)
        out = []
        for i in pool:
            if len(out) >= k:
                break
            if not has_edge(src, i, edges):
                out.append(i)
        return out

    # random
    rng.shuffle(candidates)
    out = []
    for i in candidates:
        if len(out) >= k:
            break
        if not has_edge(src, i, edges):
            out.append(i)
    return out


def vent_pressure(nodes: List[Node], edges: List[Edge], src: int, rng: np.random.Generator, p: Params) -> int:
    """
    Convert internal pressure into new mediation channels:
      - add edges from src to chosen targets
      - transfer a fraction of src.mem_int into edge.mem_e on those new edges
    Returns how many edges were added.
    """
    n = nodes[src]
    if p.vent_edges_per_event <= 0:
        return 0

    targets = choose_vent_targets(nodes, edges, src, rng, p.vent_edges_per_event, p.vent_attach_mode)
    if not targets:
        return 0

    transfer = p.vent_share * n.mem_int
    per_edge = transfer / max(len(targets), 1)

    added = 0
    for t in targets:
        if not has_edge(src, t, edges):
            add_edge(edges, src, t, rng, init_mem=per_edge)
            added += 1

    # reduce internal ledger (vented into structure)
    n.mem_int = max(0.0, n.mem_int - transfer)
    return added


def edge_step(nodes: List[Node], edges: List[Edge], rng: np.random.Generator, p: Params, tick: int) -> Tuple[float, float, int]:
    """
    Edge-local update pass. Returns:
      total_flow, total_dtheta, new_locks
    """
    total_flow = 0.0
    total_dtheta = 0.0
    new_locks = 0

    # lock budget state at start of tick
    locked_frac = float(np.mean([1.0 if e.locked else 0.0 for e in edges])) if edges else 0.0
    can_lock_more = (tick >= p.lock_start) and (locked_frac < p.lock_target)

    for e in edges:
        a = e.a
        b = e.b
        if a >= len(nodes) or b >= len(nodes):
            continue
        na = nodes[a]
        nb = nodes[b]
        if na.psi.size == 0 or nb.psi.size == 0:
            continue

        # interface mode = component 0
        xa = na.psi[0]
        xb = nb.psi[0]
        phase = np.exp(1j * e.theta)

        grad = xa - phase * xb

        throttle = np.exp(-p.kappa_edge * e.mem_e) if p.kappa_edge > 0 else 1.0
        flow = p.alpha_edge * throttle * grad

        mag = float(np.abs(flow))
        if mag > p.B_edge:
            flow *= (p.B_edge / (mag + 1e-18))
            mag = p.B_edge

        na.psi[0] = xa - flow
        nb.psi[0] = xb + np.conjugate(phase) * flow

        na.psi /= np.sqrt(np.vdot(na.psi, na.psi).real + 1e-18)
        nb.psi /= np.sqrt(np.vdot(nb.psi, nb.psi).real + 1e-18)

        total_flow += mag

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
            e.theta = wrap_pi_scalar(e.theta + dtheta)
            total_dtheta += dabs

        transported = mag + abs(dtheta)
        e.mem_e = (1.0 - p.mem_decay_edge) * e.mem_e + p.mem_couple_edge * transported

        if (not e.locked) and can_lock_more and (e.mem_e >= p.lock_threshold):
            e.locked = True
            new_locks += 1

    return total_flow, total_dtheta, new_locks


# -------------------------
# simulation
# -------------------------

def run_sim(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)

    # one subsystem
    psi0 = (rng.normal(size=p.d0) + 1j * rng.normal(size=p.d0)).astype(np.complex128)
    psi0 /= np.sqrt(np.vdot(psi0, psi0).real + 1e-18)

    nodes: List[Node] = [Node(psi=psi0, mem_int=0.0, age=0)]
    edges: List[Edge] = []

    # track vent cooldown per node (ticks since last vent)
    vent_cd = [10**9]  # large initial

    T = p.T

    # logs
    N_t = np.zeros(T, dtype=np.int32)
    E_t = np.zeros(T, dtype=np.int32)
    avg_deg_t = np.zeros(T, dtype=np.float64)

    splits_t = np.zeros(T, dtype=np.int32)
    vents_t = np.zeros(T, dtype=np.int32)
    vent_edges_added_t = np.zeros(T, dtype=np.int32)

    max_mem_int_t = np.zeros(T, dtype=np.float64)
    mean_mem_int_t = np.zeros(T, dtype=np.float64)

    total_edge_mem_t = np.zeros(T, dtype=np.float64)
    locked_frac_t = np.zeros(T, dtype=np.float64)
    new_locks_t = np.zeros(T, dtype=np.int32)

    flow_t = np.zeros(T, dtype=np.float64)
    dtheta_t = np.zeros(T, dtype=np.float64)

    g_edge_mem_t = np.zeros(T, dtype=np.float64)
    top_edge_mem_t = np.zeros(T, dtype=np.float64)

    # main loop
    for t in range(T):
        # age + cooldown
        for i, n in enumerate(nodes):
            n.age += 1
            vent_cd[i] += 1

        split_count = 0
        vent_events = 0
        vent_added = 0

        # internal updates (node list can grow)
        i = 0
        while i < len(nodes):
            n = nodes[i]
            d = n.psi.size

            psi_prop, delta_norm = attempted_internal_mix(rng, n.psi, p.eps_int)
            psi_new, residual = apply_bandwidth_to_internal_update(n.psi, psi_prop, delta_norm, p.B_int)
            n.psi = psi_new

            n.mem_int = (1.0 - p.mem_decay_int) * n.mem_int + p.mem_couple_int * residual

            # pressure decisions
            if n.mem_int >= p.split_threshold and len(nodes) < p.N_max:
                # can we split?
                if d >= 2 * p.min_dim_to_split:
                    new_idx = split_node(nodes, edges, i, rng, p)
                    if new_idx >= 0:
                        split_count += 1
                        # extend cooldown list
                        vent_cd.append(10**9)
                else:
                    # can't split further: vent pressure into new mediation links
                    if vent_cd[i] >= p.vent_cooldown:
                        added = vent_pressure(nodes, edges, i, rng, p)
                        if added > 0:
                            vent_events += 1
                            vent_added += added
                            vent_cd[i] = 0
            i += 1

        # edge-local interactions
        total_flow, total_dtheta, new_locks = (0.0, 0.0, 0)
        if len(edges) > 0 and len(nodes) > 1:
            total_flow, total_dtheta, new_locks = edge_step(nodes, edges, rng, p, t)

        # metrics
        N = len(nodes)
        E = len(edges)

        N_t[t] = N
        E_t[t] = E
        avg_deg_t[t] = float(2.0 * E / max(N, 1))
        splits_t[t] = split_count
        vents_t[t] = vent_events
        vent_edges_added_t[t] = vent_added

        flow_t[t] = total_flow
        dtheta_t[t] = total_dtheta
        new_locks_t[t] = new_locks

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

    dims = np.array([n.psi.size for n in nodes], dtype=np.int32)
    mem_int_end = np.array([n.mem_int for n in nodes], dtype=np.float64)

    u = np.array([e.a for e in edges], dtype=np.int32) if edges else np.empty(0, dtype=np.int32)
    v = np.array([e.b for e in edges], dtype=np.int32) if edges else np.empty(0, dtype=np.int32)
    theta_end = np.array([e.theta for e in edges], dtype=np.float64) if edges else np.empty(0, dtype=np.float64)
    mem_e_end = np.array([e.mem_e for e in edges], dtype=np.float64) if edges else np.empty(0, dtype=np.float64)
    locked_end = np.array([1 if e.locked else 0 for e in edges], dtype=np.int8) if edges else np.empty(0, dtype=np.int8)

    return {
        "N_t": N_t,
        "E_t": E_t,
        "avg_deg_t": avg_deg_t,
        "splits_t": splits_t,
        "vents_t": vents_t,
        "vent_edges_added_t": vent_edges_added_t,

        "max_mem_int_t": max_mem_int_t,
        "mean_mem_int_t": mean_mem_int_t,

        "total_edge_mem_t": total_edge_mem_t,
        "locked_frac_t": locked_frac_t,
        "new_locks_t": new_locks_t,

        "flow_t": flow_t,
        "dtheta_t": dtheta_t,
        "g_edge_mem_t": g_edge_mem_t,
        "top_edge_mem_t": top_edge_mem_t,

        "dims_end": dims,
        "mem_int_end": mem_int_end,

        "u": u,
        "v": v,
        "theta_end": theta_end,
        "mem_e_end": mem_e_end,
        "locked_end": locked_end,
    }


# -------------------------
# outputs
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
            "vents",
            "vent_edges_added",
            "max_mem_int",
            "mean_mem_int",
            "total_edge_mem",
            "locked_frac",
            "new_locks",
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
                int(data["vents_t"][t]),
                int(data["vent_edges_added_t"][t]),
                float(data["max_mem_int_t"][t]),
                float(data["mean_mem_int_t"][t]),
                float(data["total_edge_mem_t"][t]),
                float(data["locked_frac_t"][t]),
                int(data["new_locks_t"][t]),
                float(data["flow_t"][t]),
                float(data["dtheta_t"][t]),
                float(data["g_edge_mem_t"][t]),
                float(data["top_edge_mem_t"][t]),
            ])


def plot_outputs(run_dir: str, data: dict) -> None:
    plots_dir = os.path.join(run_dir, "plots")
    t = np.arange(len(data["N_t"]))

    # timeseries: growth
    plt.figure()
    plt.plot(t, data["N_t"], label="N(t) subsystems")
    plt.plot(t, data["E_t"], label="E(t) links")
    plt.plot(t, data["splits_t"], label="splits/tick")
    plt.plot(t, data["vents_t"], label="vents/tick")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "timeseries.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # pressure summary
    plt.figure()
    plt.plot(t, data["max_mem_int_t"], label="max internal memory")
    plt.plot(t, data["mean_mem_int_t"], label="mean internal memory")
    plt.plot(t, data["total_edge_mem_t"], label="total edge memory")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "pressure_summary.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # graph summary
    plt.figure()
    plt.plot(t, data["avg_deg_t"], label="avg degree")
    plt.plot(t, data["locked_frac_t"], label="locked fraction")
    plt.plot(t, data["flow_t"], label="total edge flow")
    plt.plot(t, data["dtheta_t"], label="total |dtheta|")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "graph_summary.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # end dimension histogram
    dims = data["dims_end"]
    if dims.size > 0:
        plt.figure()
        plt.hist(dims, bins=min(30, max(5, int(np.sqrt(dims.size)))))
        plt.xlabel("node internal dimension d_i")
        plt.ylabel("count")
        plt.savefig(os.path.join(plots_dir, "node_dim_hist_end.png"), dpi=160, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()

    # initial condition
    ap.add_argument("--d0", type=int, default=64)
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)

    # internal pressure
    ap.add_argument("--eps_int", type=float, default=0.25)
    ap.add_argument("--B_int", type=float, default=0.06)
    ap.add_argument("--mem_decay_int", type=float, default=0.0005)
    ap.add_argument("--mem_couple_int", type=float, default=2.0)
    ap.add_argument("--split_threshold", type=float, default=6.0)
    ap.add_argument("--min_dim_to_split", type=int, default=8)

    # venting: pressure -> new mediation
    ap.add_argument("--vent_share", type=float, default=0.60)
    ap.add_argument("--vent_edges_per_event", type=int, default=2)
    ap.add_argument("--vent_cooldown", type=int, default=35)
    ap.add_argument("--vent_attach_mode", type=str, default="low_degree", choices=["random", "low_degree"])

    # edge dynamics
    ap.add_argument("--B_edge", type=float, default=0.02)
    ap.add_argument("--alpha_edge", type=float, default=0.14)
    ap.add_argument("--kappa_edge", type=float, default=6.0)
    ap.add_argument("--g_coup", type=float, default=0.20)
    ap.add_argument("--B_theta", type=float, default=0.12)

    ap.add_argument("--mem_decay_edge", type=float, default=0.0002)
    ap.add_argument("--mem_couple_edge", type=float, default=3.0)

    # no-refolding locks (budgeted)
    ap.add_argument("--lock_threshold", type=float, default=0.25)
    ap.add_argument("--lock_target", type=float, default=0.25)
    ap.add_argument("--lock_start", type=int, default=80)

    # growth
    ap.add_argument("--N_max", type=int, default=512)
    ap.add_argument("--attach_k_split", type=int, default=2)

    # output
    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="bigbang_v2")

    args = ap.parse_args()

    tag = (
        f"d0{args.d0}_T{args.T}_seed{args.seed}"
        f"_Bint{args.B_int}_eps{args.eps_int}_mcI{args.mem_couple_int}"
        f"_split{args.split_threshold}_mind{args.min_dim_to_split}"
        f"_ventS{args.vent_share}_ventK{args.vent_edges_per_event}_ventCD{args.vent_cooldown}_{args.vent_attach_mode}"
        f"_Bedge{args.B_edge}_aE{args.alpha_edge}_kE{args.kappa_edge}"
        f"_mcE{args.mem_couple_edge}_lockth{args.lock_threshold}_lockt{args.lock_target}_lockstart{args.lock_start}"
        f"_Nmax{args.N_max}_attachSplit{args.attach_k_split}_{args.run_name}"
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

        vent_share=args.vent_share,
        vent_edges_per_event=args.vent_edges_per_event,
        vent_cooldown=args.vent_cooldown,
        vent_attach_mode=args.vent_attach_mode,

        B_edge=args.B_edge,
        alpha_edge=args.alpha_edge,
        kappa_edge=args.kappa_edge,
        g_coup=args.g_coup,
        B_theta=args.B_theta,

        mem_decay_edge=args.mem_decay_edge,
        mem_couple_edge=args.mem_couple_edge,

        lock_threshold=args.lock_threshold,
        lock_target=args.lock_target,
        lock_start=args.lock_start,

        N_max=args.N_max,
        attach_k_split=args.attach_k_split,

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
