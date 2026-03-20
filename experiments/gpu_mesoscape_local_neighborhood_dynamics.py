#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

BASIS0 = np.array([1.0, 0.0, 0.0], dtype=complex)


def get_xp(device: str):
    if device == "gpu":
        try:
            import cupy as cp
            return cp, True
        except Exception:
            pass
    return np, False


def gell_mann(xp):
    i = 1j
    out = []
    out.append(xp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, -i, 0], [i, 0, 0], [0, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, -i], [0, 0, 0], [i, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, 0], [0, 0, -i], [0, i, 0]], dtype=xp.complex128))
    out.append((1.0 / xp.sqrt(3.0)) * xp.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=xp.complex128))
    return out


def normalize_state(psi, xp):
    n = xp.linalg.norm(psi.reshape(-1))
    if float(n) <= 1e-15:
        raise ValueError("State norm vanished.")
    return psi / n


def apply_one_body(psi, op, site, xp):
    y = xp.moveaxis(psi, site, 0)
    y = xp.tensordot(op, y, axes=([1], [0]))
    y = xp.moveaxis(y, 0, site)
    return y


def apply_two_body_samegen(psi, op, i, j, xp):
    return apply_one_body(apply_one_body(psi, op, i, xp), op, j, xp)


def partial_trace_keep(psi, keep, n_sites, xp):
    keep = sorted(keep)
    trace_out = [i for i in range(n_sites) if i not in keep]
    perm = keep + trace_out
    psi_perm = xp.transpose(psi, perm)
    d_keep = 3 ** len(keep)
    d_tr = 3 ** len(trace_out)
    mat = psi_perm.reshape(d_keep, d_tr)
    return mat @ xp.conjugate(mat.T)


def von_neumann_entropy(rho, xp):
    vals = xp.linalg.eigvalsh(0.5 * (rho + xp.conjugate(rho.T)))
    vals = xp.real(vals)
    vals = xp.maximum(vals, 0.0)
    s = vals.sum()
    if float(s) <= 1e-15:
        return 0.0
    vals = vals / s
    nz = vals[vals > 1e-15]
    return float((-nz * xp.log(nz)).sum())


def leading_pure_state_from_rho(rho, xp):
    vals, vecs = xp.linalg.eigh(0.5 * (rho + xp.conjugate(rho.T)))
    idx = int(xp.argmax(xp.real(vals)))
    v = vecs[:, idx]
    return normalize_state(v, xp)


def build_product_state(local_nodes: List[int], seed_nodes: Optional[Set[int]], xp):
    psi = xp.asarray(BASIS0, dtype=xp.complex128)
    for idx in range(1, len(local_nodes)):
        psi = xp.kron(psi, xp.asarray(BASIS0, dtype=xp.complex128))
    return psi.reshape((3,) * len(local_nodes))


def active_triangles(active_nodes, active_edges):
    edge_set = set((min(i, j), max(i, j)) for i, j in active_edges)
    out = []
    for a, b, c in combinations(sorted(active_nodes), 3):
        if (min(a, b), max(a, b)) in edge_set and (min(b, c), max(b, c)) in edge_set and (min(a, c), max(a, c)) in edge_set:
            out.append((a, b, c))
    return out


def spectral_1d_embedding(active_nodes, active_edges, edge_strengths):
    active_nodes = sorted(active_nodes)
    if len(active_nodes) <= 1:
        return {active_nodes[0]: 0.0} if active_nodes else {}
    idx_of = {node: k for k, node in enumerate(active_nodes)}
    W = np.zeros((len(active_nodes), len(active_nodes)), dtype=float)
    for i, j in active_edges:
        if i in idx_of and j in idx_of:
            a, b = idx_of[i], idx_of[j]
            w = float(edge_strengths.get((min(i, j), max(i, j)), 0.0))
            W[a, b] = w
            W[b, a] = w
    deg = np.sum(W, axis=1)
    L = np.diag(deg) - W
    vals, vecs = np.linalg.eigh(L)
    xs = np.real(vecs[:, 1]) if len(vals) >= 2 else np.zeros(len(active_nodes))
    xs = xs - np.mean(xs)
    s = np.std(xs)
    if s > 1e-12:
        xs = xs / s
    return {node: float(xs[idx_of[node]]) for node in active_nodes}


def metric_snapshot(active_nodes, active_edges, edge_strengths):
    coords = spectral_1d_embedding(active_nodes, active_edges, edge_strengths)
    vals = list(coords.values())
    extent = float(max(vals) - min(vals)) if vals else 0.0
    edge_lengths = [abs(coords[i] - coords[j]) for i, j in active_edges if i in coords and j in coords]
    return {
        "coords": {str(k): float(v) for k, v in coords.items()},
        "metric_extent": extent,
        "total_edge_length": float(sum(edge_lengths)) if edge_lengths else 0.0,
        "mean_edge_length": float(np.mean(edge_lengths)) if edge_lengths else 0.0,
    }


def graph_adjacency(active_nodes, active_edges):
    adj = {i: [] for i in active_nodes}
    for i, j in active_edges:
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)
    return adj


def bfs_distances(seeds: Set[int], active_nodes: List[int], active_edges: List[Tuple[int, int]]):
    adj = graph_adjacency(active_nodes, active_edges)
    dist = {i: 10**9 for i in active_nodes}
    q = []
    for s in seeds:
        if s in dist:
            dist[s] = 0
            q.append(s)
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj.get(u, []):
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist, adj


@dataclass
class SimConfig:
    n_max: int
    n_init: int
    q_max: int
    seed: int
    local_scale: float
    pair_scale: float
    spawn_pair_scale: float
    total_steps: int
    dt: float
    eval_every: int
    settling_windows: int
    lookahead_windows: int
    candidate_fraction: float
    fission_fraction: float
    birth_score_floor: float
    decay_mi_threshold: float
    decay_corr_threshold: float
    mi_survival_floor: float
    corr_survival_floor: float
    persist_windows_required: int
    persist_mean_mi_threshold: float
    persist_entropy_threshold: float
    recenter_every: int
    progress_every: int
    device: str
    json_out: str


class LocalNeighborhoodState:
    def __init__(self, local_nodes: List[int], psi, xp):
        self.local_nodes = list(local_nodes)
        self.psi = psi
        self.xp = xp
        self.index_of = {node: idx for idx, node in enumerate(self.local_nodes)}

    @property
    def n_sites(self):
        return len(self.local_nodes)

    def contains(self, node: int) -> bool:
        return node in self.index_of

    def local_edges(self, active_edges: List[Tuple[int, int]]):
        s = set(self.local_nodes)
        return [e for e in active_edges if e[0] in s and e[1] in s]

    def rebuild(self, new_local_nodes: List[int]):
        new_local_nodes = list(new_local_nodes)
        if new_local_nodes == self.local_nodes:
            return
        old_nodes = self.local_nodes
        overlap = [node for node in new_local_nodes if node in self.index_of]
        xp = self.xp
        if not overlap:
            psi_new = build_product_state(new_local_nodes, None, xp)
        else:
            keep_old = [self.index_of[node] for node in overlap]
            rho_overlap = partial_trace_keep(self.psi, keep_old, len(old_nodes), xp)
            vec = leading_pure_state_from_rho(rho_overlap, xp)
            psi_new = vec.reshape((3,) * len(overlap))
            for _ in range(len(new_local_nodes) - len(overlap)):
                psi_new = xp.kron(psi_new.reshape(-1), xp.asarray(BASIS0, dtype=xp.complex128)).reshape((3,) * (psi_new.ndim + 1))
            if overlap != new_local_nodes[:len(overlap)]:
                # reorder from overlap-first to new_local_nodes ordering
                tmp_nodes = overlap + [node for node in new_local_nodes if node not in overlap]
                perm = [tmp_nodes.index(node) for node in new_local_nodes]
                psi_new = xp.transpose(psi_new, perm)
        self.local_nodes = new_local_nodes
        self.psi = normalize_state(psi_new, xp)
        self.index_of = {node: idx for idx, node in enumerate(self.local_nodes)}


def apply_hamiltonian_local(psi, local_nodes, local_edges, local_coeffs, edge_strengths, GM, xp):
    out = xp.zeros_like(psi)
    idx_of = {node: idx for idx, node in enumerate(local_nodes)}
    for node in local_nodes:
        coeffs = local_coeffs[node]
        site = idx_of[node]
        for a in range(8):
            c = float(coeffs[a])
            if c != 0.0:
                out = out + c * apply_one_body(psi, GM[a], site, xp)
    for i, j in local_edges:
        g = float(edge_strengths[(min(i, j), max(i, j))])
        if g == 0.0:
            continue
        term = xp.zeros_like(psi)
        ii = idx_of[i]
        jj = idx_of[j]
        for a in range(8):
            term = term + apply_two_body_samegen(psi, GM[a], ii, jj, xp)
        out = out + g * term
    return out


def rk4_step_local(psi, local_nodes, local_edges, local_coeffs, edge_strengths, GM, dt, xp):
    def f(state):
        return -1j * apply_hamiltonian_local(state, local_nodes, local_edges, local_coeffs, edge_strengths, GM, xp)
    k1 = f(psi)
    k2 = f(psi + 0.5 * dt * k1)
    k3 = f(psi + 0.5 * dt * k2)
    k4 = f(psi + dt * k3)
    return normalize_state(psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), xp)


def pair_mi_local(state: LocalNeighborhoodState, i: int, j: int):
    xp = state.xp
    ii = state.index_of[i]
    jj = state.index_of[j]
    rho_ab = partial_trace_keep(state.psi, [ii, jj], state.n_sites, xp)
    rho_a = partial_trace_keep(state.psi, [ii], state.n_sites, xp)
    rho_b = partial_trace_keep(state.psi, [jj], state.n_sites, xp)
    return float(von_neumann_entropy(rho_a, xp) + von_neumann_entropy(rho_b, xp) - von_neumann_entropy(rho_ab, xp))


def pair_corr_local(state: LocalNeighborhoodState, GM, i: int, j: int):
    xp = state.xp
    ii = state.index_of[i]
    jj = state.index_of[j]
    vals = []
    for a in range(8):
        tmp = apply_two_body_samegen(state.psi, GM[a], ii, jj, xp)
        vals.append(float(xp.real(xp.vdot(state.psi.reshape(-1), tmp.reshape(-1)))))
    return float(xp.linalg.norm(xp.asarray(vals)))


def one_site_entropy_local(state: LocalNeighborhoodState, i: int):
    xp = state.xp
    ii = state.index_of[i]
    rho_i = partial_trace_keep(state.psi, [ii], state.n_sites, xp)
    return float(von_neumann_entropy(rho_i, xp))


def candidate_features_local(state: LocalNeighborhoodState, active_edges, edge_strengths, GM):
    rows = []
    local_edges = state.local_edges(active_edges)
    adj = graph_adjacency(state.local_nodes, local_edges)
    triangles = active_triangles(state.local_nodes, local_edges)
    for i, j in local_edges:
        mi = pair_mi_local(state, i, j)
        corr = pair_corr_local(state, GM, i, j)
        daughter_count = len(set(adj.get(i, [])).intersection(adj.get(j, [])))
        shell_triangle_count = sum(1 for tri in triangles if i in tri and j in tri)
        score = mi * corr * (1.0 + 0.20 * daughter_count) * (1.0 + 0.10 * shell_triangle_count)
        rows.append({
            "pair": [i, j],
            "mi": mi,
            "corr": corr,
            "daughter_count": daughter_count,
            "shell_triangle_count": shell_triangle_count,
            "score": float(score),
        })
    rows.sort(key=lambda d: d["score"], reverse=True)
    return rows


def choose_candidate_births(rows, dormant_nodes, candidate_fraction, fission_fraction, birth_score_floor):
    if not rows or not dormant_nodes:
        return []
    n_considered = max(1, int(np.ceil(candidate_fraction * len(rows))))
    considered = [r for r in rows[:n_considered] if r["score"] >= birth_score_floor]
    if not considered:
        return []
    n_births = max(1, int(np.floor(fission_fraction * len(considered))))
    n_births = min(n_births, len(considered), len(dormant_nodes))
    return [(considered[idx], dormant_nodes[idx]) for idx in range(n_births)]


def spawn_births(chosen, active_nodes, dormant_nodes, active_edges, edge_strengths, local_coeffs, spawn_pair_scale, rng):
    events = []
    for row, new_node in chosen:
        i, j = row["pair"]
        if new_node not in dormant_nodes:
            continue
        dormant_nodes.remove(new_node)
        active_nodes.append(new_node)
        active_nodes.sort()
        e1 = (min(i, new_node), max(i, new_node))
        e2 = (min(j, new_node), max(j, new_node))
        if e1 not in active_edges:
            active_edges.append(e1)
        if e2 not in active_edges:
            active_edges.append(e2)
        active_edges.sort()
        edge_strengths[e1] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
        edge_strengths[e2] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
        local_coeffs[new_node] = rng.uniform(-0.5, 0.5, size=8)
        events.append({"parents": [i, j], "new_node": new_node, "seed_features": row, "spawn_links": [list(e1), list(e2)]})
    return events


def classify_births_local(events, state: LocalNeighborhoodState, active_nodes, active_edges, edge_strengths, local_coeffs, GM, cfg):
    out = []
    local_edges = state.local_edges(active_edges)
    for evt in events:
        i, j = evt["parents"]
        n = evt["new_node"]
        # Only do exact local classification if the new node and parents are in the current local neighborhood.
        if not (state.contains(i) and state.contains(j) and state.contains(n)):
            out.append({
                "parents": [i, j],
                "new_node": n,
                "windows_with_two_links": 0,
                "mean_new_node_entropy": 0.0,
                "mean_birth_mi": 0.0,
                "latest_corr_i": 0.0,
                "latest_corr_j": 0.0,
                "label": "unresolved_outside_local_patch",
            })
            continue
        mi_i = pair_mi_local(state, i, n) if (min(i, n), max(i, n)) in set(local_edges) else 0.0
        mi_j = pair_mi_local(state, j, n) if (min(j, n), max(j, n)) in set(local_edges) else 0.0
        corr_i = pair_corr_local(state, GM, i, n) if (min(i, n), max(i, n)) in set(local_edges) else 0.0
        corr_j = pair_corr_local(state, GM, j, n) if (min(j, n), max(j, n)) in set(local_edges) else 0.0
        sn = one_site_entropy_local(state, n)
        mean_mi = 0.5 * (mi_i + mi_j)
        windows_with_two_links = int(((min(i, n), max(i, n)) in set(local_edges)) and ((min(j, n), max(j, n)) in set(local_edges)))
        label = "persistent" if (
            windows_with_two_links >= 1 and
            mean_mi >= cfg.mi_survival_floor and
            corr_i >= cfg.corr_survival_floor and
            corr_j >= cfg.corr_survival_floor and
            sn >= cfg.persist_entropy_threshold
        ) else "remerge_prone"
        out.append({
            "parents": [i, j],
            "new_node": n,
            "windows_with_two_links": windows_with_two_links,
            "mean_new_node_entropy": sn,
            "mean_birth_mi": mean_mi,
            "latest_corr_i": corr_i,
            "latest_corr_j": corr_j,
            "label": label,
        })
    return out


def dominant_core_snapshot_local(state: LocalNeighborhoodState, active_edges, edge_strengths, GM):
    rows = candidate_features_local(state, active_edges, edge_strengths, GM)
    if not rows:
        return None
    best = rows[0]
    i, j = best["pair"]
    local_edges = state.local_edges(active_edges)
    adj = graph_adjacency(state.local_nodes, local_edges)
    triangles = active_triangles(state.local_nodes, local_edges)
    shell_nodes = set([i, j])
    shell_edges = []
    for a, b in local_edges:
        if a in (i, j) or b in (i, j):
            shell_nodes.add(a)
            shell_nodes.add(b)
            shell_edges.append((a, b))
    shell_nodes = sorted(shell_nodes)
    shell_triangles = [list(tri) for tri in triangles if i in tri or j in tri]
    shell_mis = []
    shell_corrs = []
    for a, b in shell_edges:
        shell_mis.append(pair_mi_local(state, a, b))
        shell_corrs.append(pair_corr_local(state, GM, a, b))
    return {
        "core_pair": [i, j],
        "core_score": float(best["score"]),
        "pair_mi": float(best["mi"]),
        "pair_corr": float(best["corr"]),
        "daughter_count": int(best["daughter_count"]),
        "shell_nodes": shell_nodes,
        "shell_size": len(shell_nodes),
        "shell_edges": [list(e) for e in shell_edges],
        "shell_triangle_count": len(shell_triangles),
        "shell_triangles": shell_triangles,
        "core_mean_pair_mi": float(np.mean(shell_mis)) if shell_mis else 0.0,
        "core_mean_pair_corr": float(np.mean(shell_corrs)) if shell_corrs else 0.0,
        "local_patch_size": state.n_sites,
    }


def summarize_epochs(snapshots):
    if not snapshots:
        return {
            "core_switch_count": 0,
            "longest_lived_core": None,
            "dominant_pair_counts": {},
            "epochs": [],
        }
    epochs = []
    cur_pair = tuple(snapshots[0]["core"]["core_pair"]) if snapshots[0].get("core") else None
    start_step = snapshots[0]["step"]
    prev_step = snapshots[0]["step"]
    counts = {}
    for snap in snapshots:
        if not snap.get("core"):
            continue
        pair = tuple(snap["core"]["core_pair"])
        counts[str(list(pair))] = counts.get(str(list(pair)), 0) + 1
        if pair != cur_pair:
            epochs.append({"core_pair": list(cur_pair), "step_start": start_step, "step_end": prev_step, "length": (prev_step - start_step) // max(1, snapshots[0].get("eval_every", 1)) + 1})
            cur_pair = pair
            start_step = snap["step"]
        prev_step = snap["step"]
    epochs.append({"core_pair": list(cur_pair), "step_start": start_step, "step_end": prev_step, "length": (prev_step - start_step) // max(1, snapshots[0].get("eval_every", 1)) + 1})
    longest = max(epochs, key=lambda e: e["length"]) if epochs else None
    return {
        "core_switch_count": max(0, len(epochs) - 1),
        "longest_lived_core": longest,
        "dominant_pair_counts": counts,
        "epochs": epochs,
    }


def select_local_patch(anchor_nodes: Set[int], active_nodes: List[int], active_edges: List[Tuple[int, int]], edge_strengths, q_max: int):
    if not active_nodes:
        return []
    dists, adj = bfs_distances(set(anchor_nodes), active_nodes, active_edges)
    tri_counts = {n: 0 for n in active_nodes}
    for tri in active_triangles(active_nodes, active_edges):
        for n in tri:
            tri_counts[n] += 1
    strength = {n: 0.0 for n in active_nodes}
    for i, j in active_edges:
        w = float(edge_strengths.get((min(i, j), max(i, j)), 0.0))
        strength[i] += w
        strength[j] += w
    ranked = sorted(active_nodes, key=lambda n: (dists.get(n, 10**9), -strength.get(n, 0.0), -tri_counts.get(n, 0), n))
    chosen = ranked[: min(q_max, len(ranked))]
    return sorted(chosen)


def run_sim(cfg: SimConfig):
    xp, using_gpu = get_xp(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    GM = gell_mann(xp)

    active_nodes = list(range(cfg.n_init))
    dormant_nodes = list(range(cfg.n_init, cfg.n_max))
    active_edges = [(i, i + 1) for i in range(cfg.n_init - 1)]
    edge_strengths = {(min(i, j), max(i, j)): float(rng.uniform(0.7 * cfg.pair_scale, 1.3 * cfg.pair_scale)) for i, j in active_edges}
    local_coeffs = {i: rng.uniform(-cfg.local_scale, cfg.local_scale, size=8) for i in range(cfg.n_max)}
    for i in dormant_nodes:
        local_coeffs[i] = np.zeros(8, dtype=float)

    local_nodes = select_local_patch(set(active_nodes), active_nodes, active_edges, edge_strengths, cfg.q_max)
    psi = build_product_state(local_nodes, set(active_nodes), xp)
    state = LocalNeighborhoodState(local_nodes, psi, xp)

    birth_events = []
    snapshots = []
    t0 = time.time()

    for step in range(1, cfg.total_steps + 1):
        state.psi = rk4_step_local(state.psi, state.local_nodes, state.local_edges(active_edges), local_coeffs, edge_strengths, GM, cfg.dt, xp)

        if step % cfg.eval_every != 0:
            continue

        core = dominant_core_snapshot_local(state, active_edges, edge_strengths, GM)
        if core is None:
            continue

        rows = candidate_features_local(state, active_edges, edge_strengths, GM)
        chosen = choose_candidate_births(rows, dormant_nodes, cfg.candidate_fraction, cfg.fission_fraction, cfg.birth_score_floor)
        new_events = spawn_births(chosen, active_nodes, dormant_nodes, active_edges, edge_strengths, local_coeffs, cfg.spawn_pair_scale, rng)
        if new_events:
            birth_events.extend(new_events)

        # recentre local patch around organizer and recent births
        anchors = set(core["shell_nodes"])
        for evt in new_events:
            anchors.add(evt["new_node"])
        if step % cfg.recenter_every == 0 or new_events:
            new_local_nodes = select_local_patch(anchors, active_nodes, active_edges, edge_strengths, cfg.q_max)
            state.rebuild(new_local_nodes)

        birth_labels = classify_births_local(new_events, state, active_nodes, active_edges, edge_strengths, local_coeffs, GM, cfg) if new_events else []
        metric = metric_snapshot(active_nodes, active_edges, edge_strengths)
        snapshots.append({
            "step": step,
            "eval_every": cfg.eval_every,
            "active_node_count": len(active_nodes),
            "active_edge_count": len(active_edges),
            "dormant_node_count": len(dormant_nodes),
            "local_patch_nodes": list(state.local_nodes),
            "local_patch_size": state.n_sites,
            "core": core,
            "metric": metric,
            "births": birth_labels,
        })

        if cfg.progress_every and len(snapshots) % cfg.progress_every == 0:
            elapsed = time.time() - t0
            pair = core["core_pair"]
            print(f"[eval {len(snapshots):03d}] step={step:4d} active={len(active_nodes):2d} local={state.n_sites:2d} core={pair} births={len(new_events)} elapsed={elapsed:7.1f}s")

    birth_labels_all = [b for snap in snapshots for b in snap["births"]]
    persistent = sum(1 for b in birth_labels_all if b["label"] == "persistent")
    remerge = sum(1 for b in birth_labels_all if b["label"] == "remerge_prone")
    unresolved = sum(1 for b in birth_labels_all if b["label"] == "unresolved_outside_local_patch")

    epoch_summary = summarize_epochs(snapshots)
    extent0 = snapshots[0]["metric"]["metric_extent"] if snapshots else 0.0
    extent1 = snapshots[-1]["metric"]["metric_extent"] if snapshots else 0.0

    result = {
        "config": vars(cfg),
        "runtime": {
            "device_used": "gpu" if using_gpu else "cpu",
            "seconds": time.time() - t0,
        },
        "summary": {
            "birth_events_total": len(birth_labels_all),
            "persistent_births": persistent,
            "remerge_prone_births": remerge,
            "unresolved_outside_local_patch": unresolved,
            "final_active_nodes": len(active_nodes),
            "final_active_edges": len(active_edges),
            "final_metric_extent": extent1,
            "metric_extent_growth": extent1 - extent0,
            "core_switch_count": epoch_summary["core_switch_count"],
            "longest_lived_core": epoch_summary["longest_lived_core"],
            "dominant_pair_counts": epoch_summary["dominant_pair_counts"],
            "max_local_patch_size": max((snap["local_patch_size"] for snap in snapshots), default=0),
        },
        "snapshots": snapshots,
        "epochs": epoch_summary["epochs"],
        "global_graph": {
            "active_nodes": active_nodes,
            "active_edges": [list(e) for e in active_edges],
            "triangles": [list(t) for t in active_triangles(active_nodes, active_edges)],
        },
        "notes": {
            "model": "Global graph growth with bounded exact quantum neighborhood around the current organizer.",
            "approximation": "When the local quantum patch recenters, overlap is preserved approximately via the leading pure component of the overlap reduced state; new nodes are added in BASIS0.",
        },
    }
    return result


def parse_args():
    p = argparse.ArgumentParser(description="HSF mesoscape dynamics with global graph growth and bounded local quantum neighborhood.")
    p.add_argument("--n-max", type=int, default=27)
    p.add_argument("--n-init", type=int, default=2)
    p.add_argument("--q-max", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local-scale", type=float, default=0.25)
    p.add_argument("--pair-scale", type=float, default=0.18)
    p.add_argument("--spawn-pair-scale", type=float, default=0.22)
    p.add_argument("--total-steps", type=int, default=120)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=4)
    p.add_argument("--settling-windows", type=int, default=2)
    p.add_argument("--lookahead-windows", type=int, default=3)
    p.add_argument("--candidate-fraction", type=float, default=0.5)
    p.add_argument("--fission-fraction", type=float, default=0.35)
    p.add_argument("--birth-score-floor", type=float, default=0.02)
    p.add_argument("--decay-mi-threshold", type=float, default=0.03)
    p.add_argument("--decay-corr-threshold", type=float, default=0.03)
    p.add_argument("--mi-survival-floor", type=float, default=0.076)
    p.add_argument("--corr-survival-floor", type=float, default=0.086)
    p.add_argument("--persist-windows-required", type=int, default=1)
    p.add_argument("--persist-mean-mi-threshold", type=float, default=0.05)
    p.add_argument("--persist-entropy-threshold", type=float, default=0.10)
    p.add_argument("--recenter-every", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--json-out", type=str, default="gpu_mesoscape_local_neighborhood_dynamics.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SimConfig(**vars(args))
    result = run_sim(cfg)
    with open(cfg.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote local-neighborhood mesoscape analysis to {cfg.json_out}")
    print(f"Device used: {result['runtime']['device_used']}")
    print(f"Births: {result['summary']['birth_events_total']} (persistent={result['summary']['persistent_births']} remerge={result['summary']['remerge_prone_births']} unresolved={result['summary']['unresolved_outside_local_patch']})")
    print(f"Final active nodes/edges: {result['summary']['final_active_nodes']} / {result['summary']['final_active_edges']}")
    print(f"Longest-lived core: {result['summary']['longest_lived_core']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
