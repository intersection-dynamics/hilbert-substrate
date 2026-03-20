#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

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


def to_cpu_array(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


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


def apply_hamiltonian(psi, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp):
    out = xp.zeros_like(psi)
    for i in active_nodes:
        coeffs = local_coeffs[i]
        for a in range(8):
            c = float(coeffs[a])
            if c != 0.0:
                out = out + c * apply_one_body(psi, GM[a], i, xp)
    for i, j in active_edges:
        g = float(edge_strengths[(min(i, j), max(i, j))])
        if g == 0.0:
            continue
        term = xp.zeros_like(psi)
        for a in range(8):
            term = term + apply_two_body_samegen(psi, GM[a], i, j, xp)
        out = out + g * term
    return out


def rk4_step(psi, dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp):
    def f(state):
        return -1j * apply_hamiltonian(state, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)

    k1 = f(psi)
    k2 = f(psi + 0.5 * dt * k1)
    k3 = f(psi + 0.5 * dt * k2)
    k4 = f(psi + dt * k3)
    psi2 = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return normalize_state(psi2, xp)


def evolve_windows(psi, n_windows, cfg, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp):
    out = psi
    for _ in range(n_windows):
        out = rk4_step(out, cfg.dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
    return out


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


def mutual_information_from_state(psi, i, j, n_sites, xp):
    rho_ab = partial_trace_keep(psi, [i, j], n_sites, xp)
    rho_a = partial_trace_keep(psi, [i], n_sites, xp)
    rho_b = partial_trace_keep(psi, [j], n_sites, xp)
    return float(von_neumann_entropy(rho_a, xp) + von_neumann_entropy(rho_b, xp) - von_neumann_entropy(rho_ab, xp))


def conditional_mutual_information_from_state(psi, i, k, j, n_sites, xp):
    rho_ikj = partial_trace_keep(psi, [i, k, j], n_sites, xp)
    rho_ik = partial_trace_keep(psi, [i, k], n_sites, xp)
    rho_kj = partial_trace_keep(psi, [k, j], n_sites, xp)
    rho_k = partial_trace_keep(psi, [k], n_sites, xp)
    return float(von_neumann_entropy(rho_ik, xp) + von_neumann_entropy(rho_kj, xp) - von_neumann_entropy(rho_k, xp) - von_neumann_entropy(rho_ikj, xp))


def pair_su3_correlator_strength(psi, GM, i, j, xp):
    vals = []
    for a in range(8):
        tmp = apply_two_body_samegen(psi, GM[a], i, j, xp)
        vals.append(float(xp.real(xp.vdot(psi.reshape(-1), tmp.reshape(-1)))))
    return float(xp.linalg.norm(xp.asarray(vals)))


def active_triangles(active_nodes, active_edges):
    edge_set = set((min(i, j), max(i, j)) for i, j in active_edges)
    out = []
    for a, b, c in combinations(active_nodes, 3):
        if (min(a, b), max(a, b)) in edge_set and (min(b, c), max(b, c)) in edge_set and (min(a, c), max(a, c)) in edge_set:
            out.append((a, b, c))
    return out


def weighted_adjacency(active_nodes, active_edges, edge_strengths):
    idx_of = {node: k for k, node in enumerate(active_nodes)}
    W = np.zeros((len(active_nodes), len(active_nodes)), dtype=float)
    for i, j in active_edges:
        a, b = idx_of[i], idx_of[j]
        w = float(edge_strengths[(min(i, j), max(i, j))])
        W[a, b] = w
        W[b, a] = w
    return W, idx_of


def spectral_1d_embedding(active_nodes, active_edges, edge_strengths):
    if len(active_nodes) <= 1:
        return {active_nodes[0]: 0.0} if active_nodes else {}
    W, idx_of = weighted_adjacency(active_nodes, active_edges, edge_strengths)
    deg = np.sum(W, axis=1)
    L = np.diag(deg) - W
    vals, vecs = np.linalg.eigh(L)
    xs = np.real(vecs[:, 1]) if len(vals) >= 2 else np.zeros(len(active_nodes))
    xs = xs - np.mean(xs)
    s = np.std(xs)
    if s > 1e-12:
        xs = xs / s
    return {node: float(xs[idx_of[node]]) for node in active_nodes}


def matrix_sqrt_psd(rho):
    vals, vecs = np.linalg.eigh(0.5 * (rho + np.conjugate(rho.T)))
    vals = np.maximum(np.real(vals), 0.0)
    return vecs @ np.diag(np.sqrt(vals)) @ np.conjugate(vecs.T)


def fidelity_uhlmann(rho, sigma):
    sr = matrix_sqrt_psd(rho)
    mid = sr @ sigma @ sr
    smid = matrix_sqrt_psd(mid)
    return float(np.real(np.trace(smid)) ** 2)


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


@dataclass
class SimConfig:
    n_max: int
    n_init: int
    seed: int
    local_scale: float
    pair_scale: float
    spawn_pair_scale: float
    total_steps: int
    dt: float
    eval_every: int
    lookahead_windows: int
    settling_windows: int
    candidate_fraction: float
    fission_fraction: float
    birth_score_floor: float
    decay_mi_threshold: float
    decay_corr_threshold: float
    neighborhood_bonus_weight: float
    shell_bonus_weight: float
    mi_survival_floor: float
    corr_survival_floor: float
    persist_windows_required: int
    persist_entropy_threshold: float
    persist_mean_mi_threshold: float
    persist_triangle_threshold: int
    probe_every: int
    branch_windows: int
    max_birth_candidates: int
    max_reabsorb_candidates: int
    cooldown_evals: int
    device: str
    progress_every: int
    json_out: str


def candidate_features(psi, active_nodes, active_edges, edge_strengths, n_sites, GM, xp, cfg):
    existing_edges = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    coords = spectral_1d_embedding(active_nodes, active_edges, edge_strengths)
    adj = {i: [] for i in active_nodes}
    for i, j in active_edges:
        adj[i].append(j)
        adj[j].append(i)
    triangles = active_triangles(active_nodes, active_edges)

    rows = []
    for i, j in combinations(active_nodes, 2):
        if (min(i, j), max(i, j)) not in existing_edges:
            continue
        rho_ab = partial_trace_keep(psi, [i, j], n_sites, xp)
        rho_a = partial_trace_keep(psi, [i], n_sites, xp)
        rho_b = partial_trace_keep(psi, [j], n_sites, xp)
        mi = float(von_neumann_entropy(rho_a, xp) + von_neumann_entropy(rho_b, xp) - von_neumann_entropy(rho_ab, xp))
        corr = pair_su3_correlator_strength(psi, GM, i, j, xp)
        coord_gap = abs(coords.get(i, 0.0) - coords.get(j, 0.0))
        common_nbrs = sorted(set(adj[i]).intersection(adj[j]))
        cmi_mean = 0.0
        if common_nbrs:
            cmis = [conditional_mutual_information_from_state(psi, i, k, j, n_sites, xp) for k in common_nbrs]
            cmi_mean = float(sum(cmis) / len(cmis))
        daughter_count = 0
        shell_triangle_count = 0
        for node in active_nodes:
            if node in (i, j):
                continue
            if (min(i, node), max(i, node)) in existing_edges and (min(j, node), max(j, node)) in existing_edges:
                daughter_count += 1
        for tri in triangles:
            if i in tri and j in tri:
                shell_triangle_count += 1
        score = float(
            mi
            * corr
            * (1.0 + cmi_mean)
            * (1.0 + float(cfg.neighborhood_bonus_weight) * daughter_count)
            * (1.0 + float(cfg.shell_bonus_weight) * shell_triangle_count)
        )
        rows.append({
            "pair": [i, j],
            "mi": mi,
            "corr": corr,
            "coord_gap": coord_gap,
            "common_nbr_count": int(len(common_nbrs)),
            "cmi_mean": cmi_mean,
            "daughter_count": int(daughter_count),
            "shell_triangle_count": int(shell_triangle_count),
            "score": score,
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


def spawn_births(chosen, active_nodes, dormant_nodes, active_edges, edge_strengths, local_coeffs, spawn_pair_scale, rng, eval_index):
    events = []
    birth_eval_map = {}
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
        birth_eval_map[new_node] = eval_index
        events.append({"parents": [i, j], "new_node": new_node})
    return events, birth_eval_map


def clone_local_coeffs(local_coeffs):
    return {k: np.array(v, dtype=float).copy() for k, v in local_coeffs.items()}


def clone_branch_state(psi, active_nodes, active_edges, edge_strengths, local_coeffs):
    return (
        psi.copy(),
        list(active_nodes),
        list(active_edges),
        dict(edge_strengths),
        clone_local_coeffs(local_coeffs),
    )


def classify_births_multiwindow(events, active_nodes, active_edges, edge_strengths, local_coeffs, psi, GM, xp, cfg):
    out = []

    for evt in events:
        i, j = evt["parents"]
        n = evt["new_node"]

        psi_work, active_nodes_w, active_edges_w, edge_strengths_w, local_coeffs_w = clone_branch_state(
            psi, active_nodes, active_edges, edge_strengths, local_coeffs
        )

        links_alive_hist = []
        mean_birth_mi_hist = []
        shell_hist = []
        common_hist = []
        entropy_hist = []

        for w in range(cfg.lookahead_windows):
            psi_work = evolve_windows(
                psi_work, 1, cfg, active_nodes_w, active_edges_w, local_coeffs_w, edge_strengths_w, GM, xp
            )

            def link_stats(parent):
                e = (min(parent, n), max(parent, n))
                if e not in active_edges_w:
                    return 0.0, 0.0, 0
                mi = mutual_information_from_state(psi_work, parent, n, cfg.n_max, xp)
                corr = pair_su3_correlator_strength(psi_work, GM, parent, n, xp)
                alive = 1
                if w >= cfg.settling_windows and mi < cfg.decay_mi_threshold and corr < cfg.decay_corr_threshold:
                    active_edges_w.remove(e)
                    edge_strengths_w.pop(e, None)
                    alive = 0
                return float(mi), float(corr), int(alive)

            if n not in active_nodes_w:
                break

            mi_i, corr_i, alive_i = link_stats(i)
            mi_j, corr_j, alive_j = link_stats(j)
            links_alive = alive_i + alive_j
            if links_alive == 0 and n in active_nodes_w:
                active_nodes_w.remove(n)
                local_coeffs_w[n] = np.zeros(8, dtype=float)

            if n in active_nodes_w:
                rho_n = partial_trace_keep(psi_work, [n], cfg.n_max, xp)
                sn = float(von_neumann_entropy(rho_n, xp))
            else:
                sn = 0.0

            mean_birth_mi = float((mi_i + mi_j) / 2.0)

            adj = {a: [] for a in active_nodes_w}
            for a, b in active_edges_w:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)

            common_support = len(set(adj.get(i, [])).intersection(adj.get(j, [])))
            shell_triangles = 0
            for tri in active_triangles(active_nodes_w, active_edges_w):
                if n in tri and (i in tri or j in tri):
                    shell_triangles += 1

            links_alive_hist.append(int(links_alive))
            mean_birth_mi_hist.append(mean_birth_mi)
            shell_hist.append(int(shell_triangles))
            common_hist.append(int(common_support))
            entropy_hist.append(float(sn))

        windows_with_two_links = sum(1 for x in links_alive_hist if x == 2)
        mean_entropy = float(np.mean(entropy_hist)) if entropy_hist else 0.0
        mean_mi = float(np.mean(mean_birth_mi_hist)) if mean_birth_mi_hist else 0.0
        mean_shell = float(np.mean(shell_hist)) if shell_hist else 0.0
        mean_common = float(np.mean(common_hist)) if common_hist else 0.0
        strong_support = (mean_common >= 1.0) or (mean_shell >= cfg.persist_triangle_threshold)

        edge_i = (min(i, n), max(i, n))
        edge_j = (min(j, n), max(j, n))
        latest_corr_i = (
            pair_su3_correlator_strength(psi_work, GM, i, n, xp)
            if n in active_nodes_w and edge_i in active_edges_w
            else 0.0
        )
        latest_corr_j = (
            pair_su3_correlator_strength(psi_work, GM, j, n, xp)
            if n in active_nodes_w and edge_j in active_edges_w
            else 0.0
        )

        label = "persistent" if (
            windows_with_two_links >= cfg.persist_windows_required
            and mean_mi >= cfg.mi_survival_floor
            and mean_mi >= cfg.persist_mean_mi_threshold
            and latest_corr_i >= cfg.corr_survival_floor
            and latest_corr_j >= cfg.corr_survival_floor
            and mean_entropy >= cfg.persist_entropy_threshold
            and strong_support
        ) else "remerge_prone"

        out.append({
            "parents": [i, j],
            "new_node": n,
            "windows_with_two_links": int(windows_with_two_links),
            "mean_new_node_entropy": mean_entropy,
            "mean_birth_mi": mean_mi,
            "mean_common_support": mean_common,
            "mean_shell_triangles": mean_shell,
            "latest_corr_i": float(latest_corr_i),
            "latest_corr_j": float(latest_corr_j),
            "label": label,
        })

    return out


def dominant_core_snapshot(psi, active_nodes, active_edges, edge_strengths, GM, xp, n_sites):
    if not active_edges:
        return None

    pair_rows = []
    edge_set = set((min(i, j), max(i, j)) for i, j in active_edges)
    for i, j in active_edges:
        mi = mutual_information_from_state(psi, i, j, n_sites, xp)
        corr = pair_su3_correlator_strength(psi, GM, i, j, xp)
        daughter_count = 0
        for node in active_nodes:
            if node in (i, j):
                continue
            if (min(i, node), max(i, node)) in edge_set and (min(j, node), max(j, node)) in edge_set:
                daughter_count += 1
        pair_rows.append({
            "pair": [i, j],
            "mi": mi,
            "corr": corr,
            "daughter_count": daughter_count,
            "core_score": float(mi * corr * (1.0 + daughter_count)),
        })

    pair_rows.sort(key=lambda d: d["core_score"], reverse=True)
    best = pair_rows[0]
    i, j = best["pair"]

    shell_nodes = set([i, j])
    shell_edges = []
    for a, b in active_edges:
        if a in (i, j) or b in (i, j):
            shell_nodes.add(a)
            shell_nodes.add(b)
            shell_edges.append((a, b))

    shell_nodes = sorted(shell_nodes)
    shell_triangles = []
    for tri in active_triangles(active_nodes, active_edges):
        if i in tri and j in tri:
            shell_triangles.append(list(tri))

    return {
        "core_pair": [i, j],
        "core_score": float(best["core_score"]),
        "pair_mi": float(best["mi"]),
        "pair_corr": float(best["corr"]),
        "daughter_count": int(best["daughter_count"]),
        "shell_nodes": shell_nodes,
        "shell_edges": [list(e) for e in shell_edges],
        "shell_size": len(shell_nodes),
        "shell_triangle_count": len(shell_triangles),
        "shell_triangles": shell_triangles,
    }


def organizer_nodes_from_core(core):
    if core is None:
        return []
    return list(core["shell_nodes"])


def active_degree_map(active_nodes, active_edges):
    deg = {n: 0 for n in active_nodes}
    for a, b in active_edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def bandwidth_pressure(active_nodes, active_edges, core):
    if not active_nodes:
        return 0.0
    shell_size = core["shell_size"] if core else 0
    return float((len(active_nodes) + 0.5 * len(active_edges)) / max(1, shell_size))


def no_refolding_pressure(active_nodes, active_edges, core):
    if core is None:
        return 0.0
    shell_size = core["shell_size"]
    tri = core["shell_triangle_count"]
    return float((len(active_edges) + tri) / max(1, shell_size))


def no_signaling_pressure(active_nodes, active_edges, edge_strengths, core):
    if core is None:
        return 0.0
    coords = spectral_1d_embedding(active_nodes, active_edges, edge_strengths)
    cp = core["core_pair"]
    org = set(core["shell_nodes"])
    if not coords or not org:
        return 0.0
    c0 = 0.5 * (coords.get(cp[0], 0.0) + coords.get(cp[1], 0.0))
    dists = [abs(coords[n] - c0) for n in org if n in coords]
    return float(np.mean(dists)) if dists else 0.0


def no_forgetting_pressure(psi, core, n_sites, xp):
    if core is None:
        return 0.0
    org = organizer_nodes_from_core(core)
    if not org:
        return 0.0

    shell_edges = [tuple(e) for e in core.get("shell_edges", [])]
    mi_vals = []
    for a, b in shell_edges:
        mi_vals.append(mutual_information_from_state(psi, a, b, n_sites, xp))
    if mi_vals:
        return float(np.mean(mi_vals))

    ent_vals = []
    for n in org:
        rho_n = partial_trace_keep(psi, [n], n_sites, xp)
        ent_vals.append(von_neumann_entropy(rho_n, xp))
    return float(np.mean(ent_vals)) if ent_vals else 0.0


def system_pressure_vector(psi, active_nodes, active_edges, edge_strengths, core, n_sites, xp):
    return {
        "bandwidth_pressure": bandwidth_pressure(active_nodes, active_edges, core),
        "no_refolding_pressure": no_refolding_pressure(active_nodes, active_edges, core),
        "no_signaling_pressure": no_signaling_pressure(active_nodes, active_edges, edge_strengths, core),
        "no_forgetting_pressure": no_forgetting_pressure(psi, core, n_sites, xp),
    }


def birth_branch_compare(
    psi,
    row,
    new_node,
    active_nodes,
    dormant_nodes,
    active_edges,
    edge_strengths,
    local_coeffs,
    birth_eval_map,
    eval_index,
    cfg,
    GM,
    rng,
    xp,
):
    i, j = row["pair"]

    psi_sup = evolve_windows(
        psi.copy(), cfg.branch_windows, cfg,
        list(active_nodes), list(active_edges),
        {k: np.array(v, dtype=float).copy() for k, v in local_coeffs.items()},
        dict(edge_strengths), GM, xp
    )
    core_sup = dominant_core_snapshot(psi_sup, active_nodes, active_edges, edge_strengths, GM, xp, cfg.n_max)
    p_sup = system_pressure_vector(psi_sup, active_nodes, active_edges, edge_strengths, core_sup, cfg.n_max, xp)

    active_nodes_b = list(active_nodes)
    dormant_nodes_b = list(dormant_nodes)
    active_edges_b = list(active_edges)
    edge_strengths_b = dict(edge_strengths)
    local_coeffs_b = {k: np.array(v, dtype=float).copy() for k, v in local_coeffs.items()}
    _, new_birth_map = spawn_births(
        [(row, new_node)], active_nodes_b, dormant_nodes_b, active_edges_b,
        edge_strengths_b, local_coeffs_b, cfg.spawn_pair_scale, rng, eval_index
    )
    birth_eval_map_b = dict(birth_eval_map)
    birth_eval_map_b.update(new_birth_map)

    psi_b = evolve_windows(
        psi.copy(), cfg.branch_windows, cfg,
        active_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, GM, xp
    )
    core_b = dominant_core_snapshot(psi_b, active_nodes_b, active_edges_b, edge_strengths_b, GM, xp, cfg.n_max)
    p_b = system_pressure_vector(psi_b, active_nodes_b, active_edges_b, edge_strengths_b, core_b, cfg.n_max, xp)

    org_sup = organizer_nodes_from_core(core_sup)
    org_b = organizer_nodes_from_core(core_b)
    common_org = sorted(set(org_sup).intersection(org_b))
    if common_org:
        rho_sup = to_cpu_array(partial_trace_keep(psi_sup, common_org, cfg.n_max, xp))
        rho_b = to_cpu_array(partial_trace_keep(psi_b, common_org, cfg.n_max, xp))
        organizer_fidelity = fidelity_uhlmann(rho_sup, rho_b)
    else:
        organizer_fidelity = 1.0

    return {
        "event_type": "birth_probe",
        "step": int(eval_index * cfg.eval_every),
        "pair": [int(i), int(j)],
        "candidate_new_node": int(new_node),
        "seed_score": float(row["score"]),
        "parent_mi": float(row["mi"]),
        "parent_corr": float(row["corr"]),
        "suppressed": p_sup,
        "allowed": p_b,
        "delta_allowed_minus_suppressed": {k: float(p_b[k] - p_sup[k]) for k in p_sup.keys()},
        "organizer_fidelity_between_branches": float(organizer_fidelity),
    }


def reabsorption_branch_compare(
    psi,
    node,
    active_nodes,
    active_edges,
    edge_strengths,
    local_coeffs,
    core,
    cfg,
    GM,
    xp,
):
    psi_keep = evolve_windows(
        psi.copy(), cfg.branch_windows, cfg,
        list(active_nodes), list(active_edges),
        {k: np.array(v, dtype=float).copy() for k, v in local_coeffs.items()},
        dict(edge_strengths), GM, xp
    )
    core_keep = dominant_core_snapshot(psi_keep, active_nodes, active_edges, edge_strengths, GM, xp, cfg.n_max)
    p_keep = system_pressure_vector(psi_keep, active_nodes, active_edges, edge_strengths, core_keep, cfg.n_max, xp)

    active_nodes_d = [x for x in active_nodes if x != node]
    active_edges_d = [e for e in active_edges if node not in e]
    edge_strengths_d = dict(edge_strengths)
    for e in list(edge_strengths_d.keys()):
        if node in e:
            edge_strengths_d.pop(e, None)
    local_coeffs_d = {k: np.array(v, dtype=float).copy() for k, v in local_coeffs.items()}
    local_coeffs_d[node] = np.zeros(8, dtype=float)

    psi_demote = evolve_windows(
        psi.copy(), cfg.branch_windows, cfg,
        active_nodes_d, active_edges_d, local_coeffs_d, edge_strengths_d, GM, xp
    )
    core_demote = dominant_core_snapshot(psi_demote, active_nodes_d, active_edges_d, edge_strengths_d, GM, xp, cfg.n_max)
    p_demote = system_pressure_vector(psi_demote, active_nodes_d, active_edges_d, edge_strengths_d, core_demote, cfg.n_max, xp)

    org_keep = organizer_nodes_from_core(core_keep)
    org_dem = organizer_nodes_from_core(core_demote)
    common_org = sorted(set(org_keep).intersection(org_dem))
    if common_org:
        rho_keep = to_cpu_array(partial_trace_keep(psi_keep, common_org, cfg.n_max, xp))
        rho_dem = to_cpu_array(partial_trace_keep(psi_demote, common_org, cfg.n_max, xp))
        organizer_fidelity = fidelity_uhlmann(rho_keep, rho_dem)
    else:
        organizer_fidelity = 1.0

    rho_n_dem = to_cpu_array(partial_trace_keep(psi_demote, [node], cfg.n_max, xp))
    basis0_return = float(np.real(rho_n_dem[0, 0]))

    return {
        "event_type": "reabsorb_probe",
        "step": None,
        "candidate_node": int(node),
        "role": "core" if core and node in set(core["core_pair"]) else ("shell" if core and node in set(core["shell_nodes"]) else "outside"),
        "kept": p_keep,
        "demoted": p_demote,
        "delta_demoted_minus_kept": {k: float(p_demote[k] - p_keep[k]) for k in p_keep.keys()},
        "organizer_fidelity_between_branches": float(organizer_fidelity),
        "basis0_return_after_demotion_branch": float(basis0_return),
    }


def run_sim(cfg: SimConfig):
    xp, using_gpu = get_xp(cfg.device)
    GM = gell_mann(xp)
    rng = np.random.default_rng(cfg.seed)

    active_nodes = list(range(cfg.n_init))
    dormant_nodes = list(range(cfg.n_init, cfg.n_max))
    active_edges = [(i, i + 1) for i in range(cfg.n_init - 1)]
    local_coeffs = {
        i: (rng.uniform(-cfg.local_scale, cfg.local_scale, size=8) if i < cfg.n_init else np.zeros(8, dtype=float))
        for i in range(cfg.n_max)
    }
    edge_strengths = {e: float(rng.uniform(0.6 * cfg.pair_scale, 1.4 * cfg.pair_scale)) for e in active_edges}

    local_states = []
    for i in range(cfg.n_max):
        if i < cfg.n_init:
            z = rng.normal(size=3) + 1j * rng.normal(size=3)
            z = z / np.linalg.norm(z)
            local_states.append(xp.asarray(z, dtype=xp.complex128))
        else:
            local_states.append(xp.asarray(BASIS0, dtype=xp.complex128))

    psi = local_states[0]
    for v in local_states[1:]:
        psi = xp.kron(psi, v)
    psi = psi.reshape((3,) * cfg.n_max)
    psi = normalize_state(psi, xp)

    birth_events = []
    snapshots = []
    pressure_events = []
    birth_eval_map = {i: 0 for i in range(cfg.n_init)}

    step = 0
    total_evals = 0
    while step < cfg.total_steps:
        psi = rk4_step(psi, cfg.dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
        step += 1
        if (step % cfg.eval_every) != 0:
            continue

        total_evals += 1
        rows = candidate_features(psi, active_nodes, active_edges, edge_strengths, cfg.n_max, GM, xp, cfg)
        chosen = choose_candidate_births(rows, dormant_nodes, cfg.candidate_fraction, cfg.fission_fraction, cfg.birth_score_floor)
        spawned, new_birth_map = spawn_births(
            chosen, active_nodes, dormant_nodes, active_edges, edge_strengths, local_coeffs, cfg.spawn_pair_scale, rng, total_evals
        )
        birth_eval_map.update(new_birth_map)

        psi = evolve_windows(psi, cfg.settling_windows, cfg, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
        labeled = classify_births_multiwindow(spawned, active_nodes, active_edges, edge_strengths, local_coeffs, psi, GM, xp, cfg)
        birth_events.extend(labeled)

        core = dominant_core_snapshot(psi, active_nodes, active_edges, edge_strengths, GM, xp, cfg.n_max)
        metric = metric_snapshot(active_nodes, active_edges, edge_strengths)

        if total_evals % cfg.probe_every == 0:
            n_birth_probe = min(cfg.max_birth_candidates, len(dormant_nodes), len(rows))
            for idx in range(n_birth_probe):
                evt = birth_branch_compare(
                    psi=psi,
                    row=rows[idx],
                    new_node=dormant_nodes[idx],
                    active_nodes=active_nodes,
                    dormant_nodes=dormant_nodes,
                    active_edges=active_edges,
                    edge_strengths=edge_strengths,
                    local_coeffs=local_coeffs,
                    birth_eval_map=birth_eval_map,
                    eval_index=total_evals,
                    cfg=cfg,
                    GM=GM,
                    rng=rng,
                    xp=xp,
                )
                pressure_events.append(evt)

            deg = active_degree_map(active_nodes, active_edges)
            candidates = []
            for n in active_nodes:
                if core and n in set(core["core_pair"]):
                    continue
                age = total_evals - birth_eval_map.get(n, -10**9)
                if age < cfg.cooldown_evals:
                    continue
                rho_n = to_cpu_array(partial_trace_keep(psi, [n], cfg.n_max, xp))
                fid0 = float(np.real(rho_n[0, 0]))
                candidates.append((-(fid0), deg.get(n, 0), n))
            candidates.sort()

            for _, _, n in candidates[: cfg.max_reabsorb_candidates]:
                evt = reabsorption_branch_compare(
                    psi=psi,
                    node=n,
                    active_nodes=active_nodes,
                    active_edges=active_edges,
                    edge_strengths=edge_strengths,
                    local_coeffs=local_coeffs,
                    core=core,
                    cfg=cfg,
                    GM=GM,
                    xp=xp,
                )
                evt["step"] = step
                pressure_events.append(evt)

        snapshots.append({
            "step": step,
            "active_nodes": list(active_nodes),
            "active_edges": [list(e) for e in active_edges],
            "dominant_core": core,
            "metric": metric,
            "births_this_window": len(labeled),
            "persistent_this_window": int(sum(1 for e in labeled if e["label"] == "persistent")),
            "remerge_this_window": int(sum(1 for e in labeled if e["label"] == "remerge_prone")),
        })

        if cfg.progress_every > 0 and total_evals % cfg.progress_every == 0:
            pair = core["core_pair"] if core else None
            print(
                f"[eval {total_evals:03d}] step={step:4d} active={len(active_nodes):2d} "
                f"edges={len(active_edges):3d} core={pair} births={len(labeled)} "
                f"pressure_events={len(pressure_events)}"
            )

    birth_probe_events = [e for e in pressure_events if e["event_type"] == "birth_probe"]
    reabsorb_probe_events = [e for e in pressure_events if e["event_type"] == "reabsorb_probe"]

    def summarize_delta(events, key):
        if not events:
            return {}
        vals = [e[key] for e in events]
        out = {}
        for name in vals[0].keys():
            arr = np.asarray([v[name] for v in vals], dtype=float)
            out[name] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "q10": float(np.quantile(arr, 0.10)),
                "q90": float(np.quantile(arr, 0.90)),
            }
        return out

    summary = {
        "backend": "cupy" if using_gpu else "numpy",
        "n_evals": total_evals,
        "n_birth_events": len(birth_events),
        "n_persistent_births": int(sum(1 for e in birth_events if e["label"] == "persistent")),
        "n_remerge_prone_births": int(sum(1 for e in birth_events if e["label"] == "remerge_prone")),
        "active_nodes_final": len(active_nodes),
        "active_edges_final": len(active_edges),
        "n_pressure_events": len(pressure_events),
        "n_birth_probe_events": len(birth_probe_events),
        "n_reabsorb_probe_events": len(reabsorb_probe_events),
        "birth_probe_delta_summary": summarize_delta(birth_probe_events, "delta_allowed_minus_suppressed"),
        "reabsorb_probe_delta_summary": summarize_delta(reabsorb_probe_events, "delta_demoted_minus_kept"),
    }

    return {
        "config": cfg.__dict__,
        "birth_events": birth_events,
        "snapshots": snapshots,
        "pressure_events": pressure_events,
        "summary": summary,
        "active_nodes_final": active_nodes,
        "active_edges_final": [list(e) for e in active_edges],
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Constraint-pressure diagnostic for HSF subsystem support dynamics.")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-scale", type=float, default=0.10)
    ap.add_argument("--pair-scale", type=float, default=0.16)
    ap.add_argument("--spawn-pair-scale", type=float, default=0.12)
    ap.add_argument("--total-steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--lookahead-windows", type=int, default=3)
    ap.add_argument("--settling-windows", type=int, default=2)
    ap.add_argument("--candidate-fraction", type=float, default=0.45)
    ap.add_argument("--fission-fraction", type=float, default=0.30)
    ap.add_argument("--birth-score-floor", type=float, default=0.015)
    ap.add_argument("--decay-mi-threshold", type=float, default=0.05)
    ap.add_argument("--decay-corr-threshold", type=float, default=0.07)
    ap.add_argument("--neighborhood-bonus-weight", type=float, default=0.18)
    ap.add_argument("--shell-bonus-weight", type=float, default=0.20)
    ap.add_argument("--mi-survival-floor", type=float, default=0.076)
    ap.add_argument("--corr-survival-floor", type=float, default=0.086)
    ap.add_argument("--persist-windows-required", type=int, default=2)
    ap.add_argument("--persist-entropy-threshold", type=float, default=0.06)
    ap.add_argument("--persist-mean-mi-threshold", type=float, default=0.07)
    ap.add_argument("--persist-triangle-threshold", type=int, default=1)
    ap.add_argument("--probe-every", type=int, default=2)
    ap.add_argument("--branch-windows", type=int, default=1)
    ap.add_argument("--max-birth-candidates", type=int, default=2)
    ap.add_argument("--max-reabsorb-candidates", type=int, default=2)
    ap.add_argument("--cooldown-evals", type=int, default=2)
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--json-out", type=str, default="gpu_mesoscape_constraint_pressure_diagnostic.json")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = SimConfig(
        n_max=args.n_max,
        n_init=args.n_init,
        seed=args.seed,
        local_scale=args.local_scale,
        pair_scale=args.pair_scale,
        spawn_pair_scale=args.spawn_pair_scale,
        total_steps=args.total_steps,
        dt=args.dt,
        eval_every=args.eval_every,
        lookahead_windows=args.lookahead_windows,
        settling_windows=args.settling_windows,
        candidate_fraction=args.candidate_fraction,
        fission_fraction=args.fission_fraction,
        birth_score_floor=args.birth_score_floor,
        decay_mi_threshold=args.decay_mi_threshold,
        decay_corr_threshold=args.decay_corr_threshold,
        neighborhood_bonus_weight=args.neighborhood_bonus_weight,
        shell_bonus_weight=args.shell_bonus_weight,
        mi_survival_floor=args.mi_survival_floor,
        corr_survival_floor=args.corr_survival_floor,
        persist_windows_required=args.persist_windows_required,
        persist_entropy_threshold=args.persist_entropy_threshold,
        persist_mean_mi_threshold=args.persist_mean_mi_threshold,
        persist_triangle_threshold=args.persist_triangle_threshold,
        probe_every=args.probe_every,
        branch_windows=args.branch_windows,
        max_birth_candidates=args.max_birth_candidates,
        max_reabsorb_candidates=args.max_reabsorb_candidates,
        cooldown_evals=args.cooldown_evals,
        device=args.device,
        progress_every=args.progress_every,
        json_out=args.json_out,
    )

    result = run_sim(cfg)
    print("=" * 96)
    print("CONSTRAINT PRESSURE DIAGNOSTIC SUMMARY")
    print("-" * 96)
    print(json.dumps(result["summary"], indent=2))
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON: {args.json_out}")


if __name__ == "__main__":
    main()