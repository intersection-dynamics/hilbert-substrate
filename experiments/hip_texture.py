#!/usr/bin/env python3
"""
HIP Texture Visualizer + Persistence Tracking (FAST)
====================================================

Adds persistence diagnostics:
  - I(u,t) = sum_{v~u} P_{u->v}(t)
  - timeseries CSV
  - top-K node curves
  - rank stability (Spearman) vs time
  - persistence scores (mean/std/CV, top-K occupancy)

Outputs added:
  - node_intensity_timeseries.csv
  - topk_persistence.png
  - rank_stability.png
  - persistence_score.json
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------
# Quantum utils
# ----------------------------

def paulis():
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return I, X, Y, Z


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def random_product_state(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    states = []
    for _ in range(n_qubits):
        u = rng.random()
        v = rng.random()
        theta = 2.0 * math.acos(math.sqrt(1.0 - u))
        phi = 2.0 * math.pi * v
        a = math.cos(theta / 2.0)
        b = math.sin(theta / 2.0) * np.exp(1j * phi)
        states.append(np.array([a, b], dtype=np.complex128))
    psi = states[0]
    for s in states[1:]:
        psi = np.kron(psi, s)
    psi /= np.linalg.norm(psi)
    return psi


def basis_state(n_qubits: int, bits: List[int]) -> np.ndarray:
    idx = 0
    for b in bits:
        idx = (idx << 1) | (b & 1)
    v = np.zeros((1 << n_qubits,), dtype=np.complex128)
    v[idx] = 1.0 + 0.0j
    return v


def single_excitation_state(n_qubits: int, excited_qubit: int) -> np.ndarray:
    bits = [0] * n_qubits
    bits[excited_qubit] = 1
    return basis_state(n_qubits, bits)


def trace_norm_2x2(A: np.ndarray) -> float:
    s = np.linalg.svd(A, compute_uv=False)
    return float(np.real(s[0] + s[1]))


def reduced_rho_single_qubit_from_state(psi: np.ndarray, q: int, n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    tensor = psi.reshape([2] * n_qubits)
    axes = [q] + [i for i in range(n_qubits) if i != q]
    tensor = np.transpose(tensor, axes)
    M = tensor.reshape(2, -1)
    rho = M @ M.conj().T
    rho = (rho + rho.conj().T) * 0.5
    tr = np.trace(rho)
    if abs(tr) > 0:
        rho /= tr
    return rho


# ----------------------------
# Graph helpers / generators
# ----------------------------

def bfs_compact_subgraph(G0: nx.Graph, start, n_keep: int) -> nx.Graph:
    if n_keep >= G0.number_of_nodes():
        return G0.copy()
    seen = set([start])
    q = [start]
    order = [start]
    while q and len(order) < n_keep:
        u = q.pop(0)
        for v in G0.neighbors(u):
            if v not in seen:
                seen.add(v)
                q.append(v)
                order.append(v)
                if len(order) >= n_keep:
                    break
    if len(order) < n_keep:
        for v in G0.nodes():
            if v not in seen:
                order.append(v)
                if len(order) >= n_keep:
                    break
    return G0.subgraph(order[:n_keep]).copy()


def pick_center_node_by_coords(G0: nx.Graph, coords: Dict) -> object:
    xs = np.array([coords[n][0] for n in G0.nodes()], dtype=np.float64)
    ys = np.array([coords[n][1] for n in G0.nodes()], dtype=np.float64)
    cx = float(xs.mean())
    cy = float(ys.mean())
    best, best_d2 = None, 1e300
    for n in G0.nodes():
        dx = coords[n][0] - cx
        dy = coords[n][1] - cy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = n
    return best


def relabel_compact(G0: nx.Graph, pos0: Dict, coord0: Optional[Dict] = None) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    mapping = {node: i for i, node in enumerate(G0.nodes())}
    G = nx.relabel_nodes(G0, mapping)
    pos = {mapping[node]: (float(pos0[node][0]), float(pos0[node][1])) for node in pos0 if node in mapping}
    if coord0 is not None:
        for old_node, new_node in mapping.items():
            if old_node in coord0:
                G.nodes[new_node]["coord"] = (int(coord0[old_node][0]), int(coord0[old_node][1]))
    return G, pos


def graph_grid(n: int) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    side = int(math.ceil(math.sqrt(n)))
    G0 = nx.grid_2d_graph(side, side)
    coords = {node: (node[0], node[1]) for node in G0.nodes()}
    center = pick_center_node_by_coords(G0, coords)
    G1 = bfs_compact_subgraph(G0, center, n)
    pos0 = {node: (float(node[0]), float(node[1])) for node in G1.nodes()}
    coord0 = {node: (int(node[0]), int(node[1])) for node in G1.nodes()}
    return relabel_compact(G1, pos0, coord0)


def graph_hex(n: int) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    m = int(max(2, round(math.sqrt(n))))
    k = m
    while True:
        H0 = nx.hexagonal_lattice_graph(m, k)
        if H0.number_of_nodes() >= n:
            break
        k += 1
    coords = {node: (node[0], node[1]) for node in H0.nodes()}
    center = pick_center_node_by_coords(H0, coords)
    H1 = bfs_compact_subgraph(H0, center, n)
    pos0 = {node: (float(node[0]), float(node[1])) for node in H1.nodes()}
    coord0 = {node: (int(node[0]), int(node[1])) for node in H1.nodes()}
    return relabel_compact(H1, pos0, coord0)


def graph_ring(n: int) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    G = nx.cycle_graph(n)
    pos0 = nx.circular_layout(G)
    pos = {i: (float(pos0[i][0]), float(pos0[i][1])) for i in G.nodes()}
    return G, pos


def graph_random(n: int, p: float, seed: int) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    if not nx.is_connected(G):
        for i in range(n):
            G.add_edge(i, (i + 1) % n)
    pos0 = nx.spring_layout(G, seed=seed)
    pos = {i: (float(pos0[i][0]), float(pos0[i][1])) for i in G.nodes()}
    return G, pos


# ----------------------------
# Model params + Hamiltonian
# ----------------------------

@dataclass
class ModelParams:
    graph_type: str
    n_sys: int
    n_env: int
    env_lambda: float
    coupling_J: float
    field_h: float
    t_max: float
    n_t: int
    n_states: int
    init_mode: str
    seed: int
    random_edge_p: float
    out_dir: str
    make_asymmetry: bool
    max_total: int
    verbose: bool
    topk: int


def build_operator_cache(n_qubits: int) -> Dict[Tuple[str, int], np.ndarray]:
    I, X, Y, Z = paulis()
    cache: Dict[Tuple[str, int], np.ndarray] = {}
    for q in range(n_qubits):
        for name, op in [("X", X), ("Y", Y), ("Z", Z)]:
            ops = [I] * n_qubits
            ops[q] = op
            cache[(name, q)] = kron_n(ops)
        ops = [I] * n_qubits
        ops[q] = Z
        cache[("Z_only", q)] = kron_n(ops)
    return cache


def build_hamiltonian(G: nx.Graph, n_sys: int, n_env: int, env_lambda: float,
                      coupling_J: float, field_h: float) -> np.ndarray:
    n_total = n_sys + n_env
    dim = 1 << n_total
    op_cache = build_operator_cache(n_total)
    H = np.zeros((dim, dim), dtype=np.complex128)

    # local fields
    if abs(field_h) > 0:
        for q in range(n_sys):
            H += field_h * op_cache[("Z_only", q)]
        for q in range(n_sys, n_total):
            H += (0.25 * field_h) * op_cache[("Z_only", q)]

    # system couplings (Heisenberg-ish)
    for (i, j) in G.edges():
        Xi = op_cache[("X", i)]
        Xj = op_cache[("X", j)]
        Yi = op_cache[("Y", i)]
        Yj = op_cache[("Y", j)]
        Zi = op_cache[("Z_only", i)]
        Zj = op_cache[("Z_only", j)]
        H += coupling_J * (Xi @ Xj + Yi @ Yj + Zi @ Zj)

    # system-env ZZ couplings
    if n_env > 0 and abs(env_lambda) > 0:
        for e in range(n_env):
            env_q = n_sys + e
            sys_q = e % n_sys
            Zs = op_cache[("Z_only", sys_q)]
            Ze = op_cache[("Z_only", env_q)]
            H += env_lambda * (Zs @ Ze)

    return (H + H.conj().T) * 0.5


def prediag(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(H)
    return w, V


def evolve_batch_from_prediag(w: np.ndarray, V: np.ndarray, states: np.ndarray, t: float) -> np.ndarray:
    phases = np.exp(-1j * w * t)
    tmp = (V.conj().T @ states.T).T
    tmp = tmp * phases[np.newaxis, :]
    out = (V @ tmp.T).T
    return out


def build_initial_states(params: ModelParams, n_total: int, rng: np.random.Generator) -> np.ndarray:
    mode = params.init_mode.lower().strip()
    if mode == "zero":
        psi0 = basis_state(n_total, [0] * n_total)
        return np.stack([psi0.copy() for _ in range(params.n_states)], axis=0)
    if mode == "single_exc":
        states = []
        for k in range(params.n_states):
            q = k % params.n_sys
            states.append(single_excitation_state(n_total, q))
        return np.stack(states, axis=0)
    # random_product
    return np.stack([random_product_state(n_total, rng) for _ in range(params.n_states)], axis=0)


def estimate_permeability_field_fast(w: np.ndarray, V: np.ndarray,
                                     G: nx.Graph,
                                     n_sys: int,
                                     n_env: int,
                                     t: float,
                                     init_states: np.ndarray,
                                     op_cache: Dict[Tuple[str, int], np.ndarray],
                                     verbose: bool = False) -> Tuple[Dict[Tuple[int, int], float], float]:
    n_total = n_sys + n_env
    psi_t_base = evolve_batch_from_prediag(w, V, init_states, t)

    rho_base: Dict[int, Dict[int, np.ndarray]] = {}
    for k in range(init_states.shape[0]):
        rho_base[k] = {}
        for j in range(n_sys):
            rho_base[k][j] = reduced_rho_single_qubit_from_state(psi_t_base[k], j, n_total)

    P_dir: Dict[Tuple[int, int], float] = {}

    for i in range(n_sys):
        neighs = list(G.neighbors(i))
        if not neighs:
            continue

        for op_name in ["X", "Y", "Z"]:
            Oi = op_cache[(op_name, i)]
            psi0_pert = (Oi @ init_states.T).T
            psi_t_pert = evolve_batch_from_prediag(w, V, psi0_pert, t)

            for j in neighs:
                best = P_dir.get((i, j), 0.0)
                for k in range(init_states.shape[0]):
                    rho_p = reduced_rho_single_qubit_from_state(psi_t_pert[k], j, n_total)
                    val = trace_norm_2x2(rho_p - rho_base[k][j])
                    if val > best:
                        best = val
                P_dir[(i, j)] = best

        if verbose:
            print(f"    source i={i+1}/{n_sys} done")

    edge_vals = []
    for (u, v) in G.edges():
        edge_vals.append(max(P_dir.get((u, v), 0.0), P_dir.get((v, u), 0.0)))
    edge_vals = np.array(edge_vals, dtype=np.float64)
    hip_var = float(np.var(edge_vals)) if edge_vals.size > 0 else 0.0
    return P_dir, hip_var


# ----------------------------
# Visualization + persistence
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_graph_edge_field(G: nx.Graph, pos: Dict[int, Tuple[float, float]],
                          P_dir: Dict[Tuple[int, int], float], outpath: str,
                          title: str, eps_zero: float = 1e-12) -> None:
    edges = list(G.edges())
    edge_vals = [max(P_dir.get((u, v), 0.0), P_dir.get((v, u), 0.0)) for (u, v) in edges]
    edge_vals = [0.0 if abs(x) < eps_zero else x for x in edge_vals]
    vmin = float(min(edge_vals)) if edge_vals else 0.0
    vmax = float(max(edge_vals)) if edge_vals else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-12

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(title)
    ax.set_axis_off()
    nx.draw_networkx_nodes(G, pos, node_size=140, linewidths=0.0, ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_vals,
        width=3.0,
        edge_cmap=plt.cm.viridis,
        edge_vmin=vmin,
        edge_vmax=vmax,
        ax=ax
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Edge permeability P_edge(t)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_node_intensity(G: nx.Graph, pos: Dict[int, Tuple[float, float]],
                        P_dir: Dict[Tuple[int, int], float], outpath: str,
                        title: str, eps_zero: float = 1e-12) -> np.ndarray:
    intens = np.zeros((G.number_of_nodes(),), dtype=np.float64)
    for u in G.nodes():
        intens[u] = sum(P_dir.get((u, v), 0.0) for v in G.neighbors(u))
    intens[np.abs(intens) < eps_zero] = 0.0

    vmin = float(np.min(intens)) if intens.size else 0.0
    vmax = float(np.max(intens)) if intens.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-12

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(title)
    ax.set_axis_off()
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.6, ax=ax)
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=intens,
        cmap=plt.cm.magma,
        vmin=vmin,
        vmax=vmax,
        node_size=220,
        linewidths=0.0,
        ax=ax
    )
    fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04, label="Node outgoing intensity Σ P_{u→v}(t)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return intens


def save_timeseries(ts: np.ndarray, ys: np.ndarray, outpath: str, title: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, ys, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("time t")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def ranks_with_ties(values: np.ndarray) -> np.ndarray:
    """Dense ranks for Spearman-like correlation; ties get average rank."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(x)
    i = 0
    r = 1.0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        # average rank for tie block [i..j]
        avg = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        r += (j - i + 1)
        i = j + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = ranks_with_ties(a)
    rb = ranks_with_ties(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom == 0.0:
        return 0.0
    return float((ra @ rb) / denom)


def write_intensity_csv(ts: np.ndarray, I: np.ndarray, outpath: str) -> None:
    # I shape: (T, N)
    header = "t," + ",".join([f"node_{i}" for i in range(I.shape[1])])
    lines = [header]
    for k in range(I.shape[0]):
        row = [f"{ts[k]:.12g}"] + [f"{I[k, i]:.12g}" for i in range(I.shape[1])]
        lines.append(",".join(row))
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_topk_persistence(ts: np.ndarray, I: np.ndarray, topk: int, outpath: str) -> List[int]:
    # pick topk by mean intensity over time
    means = I.mean(axis=0)
    idx = np.argsort(-means)[:max(1, min(topk, I.shape[1]))].tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    for u in idx:
        ax.plot(ts, I[:, u], marker="o", linewidth=1.4, label=f"node {u}")
    ax.set_title(f"Top-{len(idx)} node intensity persistence")
    ax.set_xlabel("time t")
    ax.set_ylabel("I(u,t) = Σ_v P_{u→v}(t)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return idx


def plot_rank_stability(ts: np.ndarray, I: np.ndarray, outpath: str) -> None:
    base = I[0, :]
    rhos = np.array([spearman_corr(base, I[k, :]) for k in range(I.shape[0])], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, rhos, marker="o", linewidth=1.8)
    ax.set_title("Rank stability of node intensity vs initial time (Spearman)")
    ax.set_xlabel("time t")
    ax.set_ylabel("Spearman ρ(rank(I(t)), rank(I(t0)))")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def persistence_scores(ts: np.ndarray, I: np.ndarray, topk: int) -> Dict:
    eps = 1e-12
    means = I.mean(axis=0)
    stds = I.std(axis=0)
    cvs = stds / (means + eps)

    # top-k occupancy: how often each node appears in top-k per time slice
    T, N = I.shape
    occ = np.zeros((N,), dtype=np.int64)
    k = max(1, min(topk, N))
    for t_idx in range(T):
        top = np.argsort(-I[t_idx, :])[:k]
        occ[top] += 1
    occ_frac = occ / float(T)

    # identify "persistent hubs": high mean AND high occupancy
    hub_idx = np.argsort(-(0.7 * (means / (means.max() + eps)) + 0.3 * occ_frac))[:k].tolist()

    return {
        "topk": int(k),
        "time_points": int(T),
        "nodes": int(N),
        "mean_intensity": {str(i): float(means[i]) for i in range(N)},
        "std_intensity": {str(i): float(stds[i]) for i in range(N)},
        "cv_intensity": {str(i): float(cvs[i]) for i in range(N)},
        "topk_occupancy_fraction": {str(i): float(occ_frac[i]) for i in range(N)},
        "persistent_hub_candidates": hub_idx,
    }


# ----------------------------
# Main
# ----------------------------

def make_graph(params: ModelParams) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    gt = params.graph_type.lower()
    if gt == "grid":
        return graph_grid(params.n_sys)
    if gt == "hex":
        return graph_hex(params.n_sys)
    if gt == "ring":
        return graph_ring(params.n_sys)
    if gt == "random":
        return graph_random(params.n_sys, params.random_edge_p, params.seed)
    raise ValueError(f"Unknown graph type: {params.graph_type}")


def parse_args() -> ModelParams:
    ap = argparse.ArgumentParser(description="HIP Texture Visualizer (fast) + persistence.")
    ap.add_argument("--graph", type=str, default="hex", choices=["grid", "hex", "ring", "random"])
    ap.add_argument("--n-sys", type=int, default=12)
    ap.add_argument("--n-env", type=int, default=0)
    ap.add_argument("--env-lambda", type=float, default=0.0)
    ap.add_argument("--J", type=float, default=0.6)
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--t-max", type=float, default=3.0)
    ap.add_argument("--n-t", type=int, default=12)
    ap.add_argument("--n-states", type=int, default=12)
    ap.add_argument("--init", type=str, default="single_exc", choices=["random_product", "zero", "single_exc"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-edge-p", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="outputs_hip_texture")
    ap.add_argument("--asymmetry", action="store_true")
    ap.add_argument("--max-total", type=int, default=14)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--topk", type=int, default=4, help="Top-K nodes to plot for persistence.")
    a = ap.parse_args()

    return ModelParams(
        graph_type=a.graph,
        n_sys=a.n_sys,
        n_env=a.n_env,
        env_lambda=a.env_lambda,
        coupling_J=a.J,
        field_h=a.h,
        t_max=a.t_max,
        n_t=a.n_t,
        n_states=a.n_states,
        init_mode=a.init,
        seed=a.seed,
        random_edge_p=a.random_edge_p,
        out_dir=a.out,
        make_asymmetry=bool(a.asymmetry),
        max_total=a.max_total,
        verbose=bool(a.verbose),
        topk=a.topk,
    )


def main():
    params = parse_args()
    ensure_dir(params.out_dir)

    n_total = params.n_sys + params.n_env
    if n_total > params.max_total:
        raise RuntimeError(
            f"Refusing to run: n_total={n_total} exceeds --max-total={params.max_total}. "
            f"Reduce --n-sys/--n-env or increase --max-total (exponential cost)."
        )

    with open(os.path.join(params.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)

    print(f"[HIP] building graph: {params.graph_type}  n_sys={params.n_sys}")
    G, pos = make_graph(params)

    print(f"[HIP] building Hamiltonian: n_total={n_total}  dim=2^{n_total}={1<<n_total}")
    H = build_hamiltonian(G, params.n_sys, params.n_env, params.env_lambda, params.coupling_J, params.field_h)

    print("[HIP] diagonalizing H once...")
    w, V = prediag(H)
    print("[HIP] diagonalization complete.")

    rng = np.random.default_rng(params.seed)
    init_states = build_initial_states(params, n_total, rng)
    op_cache = build_operator_cache(n_total)

    ts = np.linspace(0.0, params.t_max, params.n_t)
    hip_vals: List[float] = []

    # persistence storage: I[t_idx, node]
    I_all = np.zeros((len(ts), G.number_of_nodes()), dtype=np.float64)

    for idx, t in enumerate(ts):
        print(f"[HIP] time {idx+1}/{len(ts)}  t={t:.4f}")
        P_dir, hip_var = estimate_permeability_field_fast(
            w, V, G, params.n_sys, params.n_env, float(t), init_states, op_cache, verbose=params.verbose
        )
        hip_vals.append(hip_var)

        out_edge = os.path.join(params.out_dir, f"edge_field_t{idx:03d}.png")
        save_graph_edge_field(G, pos, P_dir, out_edge,
                             f"HIP edge permeability field (t={t:.3f}) | HIP_var={hip_var:.6g}")

        out_node = os.path.join(params.out_dir, f"node_intensity_t{idx:03d}.png")
        intens = save_node_intensity(G, pos, P_dir, out_node,
                                     f"Node outgoing permeability intensity (t={t:.3f})")
        I_all[idx, :] = intens

    hip_vals_arr = np.array(hip_vals, dtype=np.float64)
    save_timeseries(ts, hip_vals_arr,
                    os.path.join(params.out_dir, "hip_timeseries.png"),
                    "HIP(t): variance of edge permeability across interaction graph",
                    "HIP_var(t) = Var_e(P_edge(t))")

    # persistence outputs
    write_intensity_csv(ts, I_all, os.path.join(params.out_dir, "node_intensity_timeseries.csv"))
    plot_topk_persistence(ts, I_all, params.topk, os.path.join(params.out_dir, "topk_persistence.png"))
    plot_rank_stability(ts, I_all, os.path.join(params.out_dir, "rank_stability.png"))
    scores = persistence_scores(ts, I_all, params.topk)
    with open(os.path.join(params.out_dir, "persistence_score.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print("[HIP] done.")
    print(f"Outputs: {os.path.abspath(params.out_dir)}")


if __name__ == "__main__":
    main()
