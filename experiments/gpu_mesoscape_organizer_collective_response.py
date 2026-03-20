#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

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
    out.append(xp.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=xp.complex128))
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
    if len(active_nodes) == 1:
        return {active_nodes[0]: 0.0}
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


def active_triangles(active_nodes, active_edges):
    edge_set = set((min(i, j), max(i, j)) for i, j in active_edges)
    out = []
    for a, b, c in combinations(active_nodes, 3):
        if (min(a, b), max(a, b)) in edge_set and (min(b, c), max(b, c)) in edge_set and (min(a, c), max(a, c)) in edge_set:
            out.append((a, b, c))
    return out


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
    fission_fraction: float
    candidate_fraction: float
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
    device: str = "gpu"


def candidate_features(psi, active_nodes, active_edges, edge_strengths, n_sites, GM, xp):
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
        pair_entropy = float(von_neumann_entropy(rho_ab, xp))
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
        score = float(mi * corr * (1.0 + cmi_mean) * (1.0 + 0.20 * daughter_count) * (1.0 + 0.10 * shell_triangle_count))
        rows.append({
            "pair": [i, j],
            "mi": mi,
            "corr": corr,
            "pair_entropy": pair_entropy,
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
        events.append({"parents": [i, j], "new_node": new_node})
    return events


def evolve_windows(psi, n_windows, cfg, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp):
    for _ in range(n_windows):
        psi = rk4_step(psi, cfg.dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
    return psi


def classify_births_multiwindow(events, active_nodes, active_edges, edge_strengths, local_coeffs, psi, GM, xp, cfg):
    out = []
    psi_work = psi.copy()
    for evt in events:
        i, j = evt["parents"]
        n = evt["new_node"]
        links_alive_hist = []
        mean_birth_mi_hist = []
        shell_hist = []
        common_hist = []
        entropy_hist = []

        for w in range(cfg.lookahead_windows):
            psi_work = evolve_windows(psi_work, 1, cfg, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)

            def link_stats(parent):
                e = (min(parent, n), max(parent, n))
                if e not in active_edges:
                    return 0.0, 0.0, 0
                mi = mutual_information_from_state(psi_work, parent, n, cfg.n_max, xp)
                corr = pair_su3_correlator_strength(psi_work, GM, parent, n, xp)
                alive = 1
                if w >= cfg.settling_windows and mi < cfg.decay_mi_threshold and corr < cfg.decay_corr_threshold:
                    if e in active_edges:
                        active_edges.remove(e)
                    edge_strengths.pop(e, None)
                    alive = 0
                return float(mi), float(corr), int(alive)

            if n not in active_nodes:
                break

            mi_i, corr_i, alive_i = link_stats(i)
            mi_j, corr_j, alive_j = link_stats(j)
            links_alive = alive_i + alive_j
            if links_alive == 0 and n in active_nodes:
                active_nodes.remove(n)
                local_coeffs[n] = np.zeros(8, dtype=float)

            if n in active_nodes:
                rho_n = partial_trace_keep(psi_work, [n], cfg.n_max, xp)
                sn = float(von_neumann_entropy(rho_n, xp))
            else:
                sn = 0.0

            mean_birth_mi = float((mi_i + mi_j) / 2.0)

            adj = {a: [] for a in active_nodes}
            for a, b in active_edges:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            common_support = len(set(adj.get(i, [])).intersection(adj.get(j, [])))
            shell_triangles = 0
            for tri in active_triangles(active_nodes, active_edges):
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
        latest_corr_i = pair_su3_correlator_strength(psi_work, GM, i, n, xp) if n in active_nodes else 0.0
        latest_corr_j = pair_su3_correlator_strength(psi_work, GM, j, n, xp) if n in active_nodes else 0.0

        label = "persistent" if (
            windows_with_two_links >= cfg.persist_windows_required and
            mean_mi >= cfg.mi_survival_floor and
            latest_corr_i >= cfg.corr_survival_floor and
            latest_corr_j >= cfg.corr_survival_floor and
            mean_entropy >= cfg.persist_entropy_threshold and
            strong_support
        ) else "remerge_prone"

        out.append({"parents": [i, j], "new_node": n, "label": label})
    return out, psi_work


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
        "shell_size": len(shell_nodes),
        "shell_edges": [list(e) for e in shell_edges],
        "shell_triangle_count": len(shell_triangles),
        "shell_triangles": shell_triangles,
    }


def summarize_epochs(snapshots):
    if not snapshots:
        return {"metastable_epochs": []}
    epochs = []
    current = snapshots[0]["dominant_core"]["core_pair"] if snapshots[0]["dominant_core"] else None
    start_idx = 0
    for idx in range(1, len(snapshots)):
        pair = snapshots[idx]["dominant_core"]["core_pair"] if snapshots[idx]["dominant_core"] else None
        if pair != current:
            epochs.append({
                "core_pair": current,
                "start_step": snapshots[start_idx]["step"],
                "end_step": snapshots[idx - 1]["step"],
                "n_snapshots": idx - start_idx,
                "snapshot_start_idx": start_idx,
                "snapshot_end_idx": idx - 1,
            })
            current = pair
            start_idx = idx
    epochs.append({
        "core_pair": current,
        "start_step": snapshots[start_idx]["step"],
        "end_step": snapshots[-1]["step"],
        "n_snapshots": len(snapshots) - start_idx,
        "snapshot_start_idx": start_idx,
        "snapshot_end_idx": len(snapshots) - 1,
    })
    return {"metastable_epochs": [e for e in epochs if e["core_pair"] is not None]}


def unitary_from_generator(gen, epsilon, xp):
    vals, vecs = xp.linalg.eigh(0.5 * (gen + xp.conjugate(gen.T)))
    phases = xp.exp(-1j * epsilon * vals)
    return vecs @ xp.diag(phases) @ xp.conjugate(vecs.T)


def trace_distance(rho, sigma):
    delta = 0.5 * ((rho - sigma) + np.conjugate((rho - sigma).T))
    vals = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(np.real(vals))))


def matrix_sqrt_psd(rho):
    vals, vecs = np.linalg.eigh(0.5 * (rho + np.conjugate(rho.T)))
    vals = np.maximum(np.real(vals), 0.0)
    return vecs @ np.diag(np.sqrt(vals)) @ np.conjugate(vecs.T)


def fidelity_uhlmann(rho, sigma):
    sr = matrix_sqrt_psd(rho)
    mid = sr @ sigma @ sr
    smid = matrix_sqrt_psd(mid)
    return float(np.real(np.trace(smid)) ** 2)


def organizer_partition(core_snapshot, active_edges, edge_strengths):
    organizer_nodes = sorted(core_snapshot["shell_nodes"])
    org_set = set(organizer_nodes)
    organizer_edges = sorted([
        (min(i, j), max(i, j))
        for i, j in active_edges
        if i in org_set and j in org_set
    ])
    coords = spectral_1d_embedding(organizer_nodes, organizer_edges, edge_strengths)
    ordered = sorted(organizer_nodes, key=lambda n: coords.get(n, 0.0))

    core_pair = list(core_snapshot["core_pair"])
    shell_only = [n for n in organizer_nodes if n not in core_pair]
    mid = max(1, len(ordered) // 2)
    left = ordered[:mid]
    right = ordered[mid:]
    if not right:
        right = [ordered[-1]]
        left = ordered[:-1]

    return {
        "organizer_nodes": organizer_nodes,
        "organizer_edges": [list(e) for e in organizer_edges],
        "coords": {str(k): float(v) for k, v in coords.items()},
        "ordered_nodes": ordered,
        "core_nodes": core_pair,
        "shell_nodes": shell_only,
        "left_nodes": left,
        "right_nodes": right,
    }


def choose_target_epoch(epochs, target_step=None):
    if not epochs:
        return None, None
    if target_step is not None:
        best = None
        best_dist = None
        for ep in epochs:
            mid = 0.5 * (ep["start_step"] + ep["end_step"])
            d = abs(mid - target_step)
            if best is None or d < best_dist:
                best = ep
                best_dist = d
        ep = best
    else:
        ep = max(epochs, key=lambda e: e["n_snapshots"])
    idx = (ep["snapshot_start_idx"] + ep["snapshot_end_idx"]) // 2
    return ep, idx


def simulate_mesoscape(cfg, progress=False):
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

    snapshots = []
    states_at_snapshots = []

    step = 0
    while step < cfg.total_steps:
        psi = rk4_step(psi, cfg.dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
        step += 1
        if (step % cfg.eval_every) == 0:
            rows = candidate_features(psi, active_nodes, active_edges, edge_strengths, cfg.n_max, GM, xp)
            chosen = choose_candidate_births(rows, dormant_nodes, cfg.candidate_fraction, cfg.fission_fraction, cfg.birth_score_floor)
            spawned = spawn_births(chosen, active_nodes, dormant_nodes, active_edges, edge_strengths, local_coeffs, cfg.spawn_pair_scale, rng)
            psi = evolve_windows(psi, cfg.settling_windows, cfg, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
            labeled, psi = classify_births_multiwindow(spawned, active_nodes, active_edges, edge_strengths, local_coeffs, psi, GM, xp, cfg)
            core = dominant_core_snapshot(psi, active_nodes, active_edges, edge_strengths, GM, xp, cfg.n_max)
            snapshots.append({
                "step": step,
                "active_nodes": list(active_nodes),
                "active_edges": [list(e) for e in active_edges],
                "dominant_core": core,
                "births_this_window": len(labeled),
            })
            states_at_snapshots.append({
                "psi": psi.copy(),
                "active_nodes": list(active_nodes),
                "active_edges": list(active_edges),
                "edge_strengths": dict(edge_strengths),
                "local_coeffs": {k: np.array(v, dtype=float).copy() for k, v in local_coeffs.items()},
            })
            if progress:
                pair = core["core_pair"] if core else None
                print(f"[eval] step={step:4d} active={len(active_nodes):2d} edges={len(active_edges):3d} core={pair}")
    return xp, using_gpu, GM, snapshots, states_at_snapshots


def apply_collective_poke(
    psi,
    partition,
    active_nodes,
    GM,
    xp,
    generator_index,
    epsilon,
    poke_mode,
    rng,
):
    gen = GM[generator_index]
    psi2 = psi.copy()

    organizer_nodes = partition["organizer_nodes"]
    core_nodes = partition["core_nodes"]
    shell_nodes = partition["shell_nodes"]
    left_nodes = partition["left_nodes"]
    right_nodes = partition["right_nodes"]

    def apply_signed(nodes, sign):
        nonlocal psi2
        if not nodes:
            return
        U = unitary_from_generator(gen, sign * epsilon, xp)
        for node in nodes:
            psi2 = apply_one_body(psi2, U, node, xp)

    if poke_mode == "uniform":
        apply_signed(organizer_nodes, +1.0)

    elif poke_mode == "dipole":
        apply_signed(left_nodes, +1.0)
        apply_signed(right_nodes, -1.0)

    elif poke_mode == "breathing":
        apply_signed(core_nodes, +1.0)
        apply_signed(shell_nodes, -1.0)

    elif poke_mode == "circulation":
        ordered = partition["ordered_nodes"]
        for k, node in enumerate(ordered):
            sign = +1.0 if (k % 2 == 0) else -1.0
            U = unitary_from_generator(gen, sign * epsilon, xp)
            psi2 = apply_one_body(psi2, U, node, xp)

    elif poke_mode == "random_control":
        outside = [n for n in active_nodes if n not in set(organizer_nodes)]
        if len(outside) < len(organizer_nodes):
            raise ValueError("Not enough outside nodes for random_control.")
        chosen = rng.choice(outside, size=len(organizer_nodes), replace=False).tolist()
        for node in chosen:
            U = unitary_from_generator(gen, epsilon, xp)
            psi2 = apply_one_body(psi2, U, int(node), xp)
        return normalize_state(psi2, xp), {"random_control_nodes": [int(x) for x in chosen]}

    else:
        raise ValueError(f"Unknown poke_mode: {poke_mode}")

    return normalize_state(psi2, xp), {}


def organizer_observables(psi, partition, n_sites, xp):
    org_nodes = partition["organizer_nodes"]
    core_nodes = partition["core_nodes"]
    shell_nodes = partition["shell_nodes"]

    rho_org = to_cpu_array(partial_trace_keep(psi, org_nodes, n_sites, xp))
    rho_core = to_cpu_array(partial_trace_keep(psi, core_nodes, n_sites, xp))
    rho_shell = to_cpu_array(partial_trace_keep(psi, shell_nodes, n_sites, xp)) if shell_nodes else None

    obs = {
        "rho_org": rho_org,
        "rho_core": rho_core,
        "entropy_org": von_neumann_entropy(np.asarray(rho_org), np),
        "entropy_core": von_neumann_entropy(np.asarray(rho_core), np),
        "entropy_shell": von_neumann_entropy(np.asarray(rho_shell), np) if rho_shell is not None else 0.0,
    }

    if shell_nodes:
        obs["core_shell_mutual_information"] = (
            von_neumann_entropy(np.asarray(rho_core), np)
            + von_neumann_entropy(np.asarray(rho_shell), np)
            - von_neumann_entropy(np.asarray(rho_org), np)
        )
    else:
        obs["core_shell_mutual_information"] = 0.0
    return obs


def probe_collective_response(
    psi0,
    partition,
    active_nodes,
    active_edges,
    local_coeffs,
    edge_strengths,
    GM,
    xp,
    cfg,
    poke_mode,
    generator_index,
    epsilon,
    probe_steps,
):
    rng = np.random.default_rng(cfg.seed + 999)
    psi_ctrl = psi0.copy()
    psi_poke, poke_meta = apply_collective_poke(
        psi0, partition, active_nodes, GM, xp, generator_index, epsilon, poke_mode, rng
    )

    curves = []
    for t in range(probe_steps + 1):
        ctrl = organizer_observables(psi_ctrl, partition, cfg.n_max, xp)
        poked = organizer_observables(psi_poke, partition, cfg.n_max, xp)

        row = {
            "probe_step": t,
            "org_trace_distance": trace_distance(poked["rho_org"], ctrl["rho_org"]),
            "org_fidelity": fidelity_uhlmann(poked["rho_org"], ctrl["rho_org"]),
            "core_trace_distance": trace_distance(poked["rho_core"], ctrl["rho_core"]),
            "core_fidelity": fidelity_uhlmann(poked["rho_core"], ctrl["rho_core"]),
            "entropy_org_control": ctrl["entropy_org"],
            "entropy_org_poked": poked["entropy_org"],
            "entropy_org_shift": float(poked["entropy_org"] - ctrl["entropy_org"]),
            "entropy_core_shift": float(poked["entropy_core"] - ctrl["entropy_core"]),
            "entropy_shell_shift": float(poked["entropy_shell"] - ctrl["entropy_shell"]),
            "core_shell_mi_control": ctrl["core_shell_mutual_information"],
            "core_shell_mi_poked": poked["core_shell_mutual_information"],
            "core_shell_mi_shift": float(poked["core_shell_mutual_information"] - ctrl["core_shell_mutual_information"]),
        }
        curves.append(row)

        if t < probe_steps:
            psi_ctrl = rk4_step(psi_ctrl, cfg.dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
            psi_poke = rk4_step(psi_poke, cfg.dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)

    org_td = [r["org_trace_distance"] for r in curves]
    core_td = [r["core_trace_distance"] for r in curves]
    mi_shift = [r["core_shell_mi_shift"] for r in curves]
    summary = {
        "max_org_trace_distance": float(max(org_td)),
        "mean_org_trace_distance": float(np.mean(org_td)),
        "argmax_org_trace_distance_step": int(np.argmax(org_td)),
        "max_core_trace_distance": float(max(core_td)),
        "mean_core_trace_distance": float(np.mean(core_td)),
        "max_abs_core_shell_mi_shift": float(max(abs(x) for x in mi_shift)),
        "argmax_abs_core_shell_mi_shift_step": int(np.argmax([abs(x) for x in mi_shift])),
    }
    summary.update(poke_meta)
    return curves, summary


def parse_args():
    ap = argparse.ArgumentParser(description="Collective organizer response witness at organizer scale.")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-scale", type=float, default=0.10)
    ap.add_argument("--pair-scale", type=float, default=0.16)
    ap.add_argument("--spawn-pair-scale", type=float, default=0.12)
    ap.add_argument("--total-steps", type=int, default=800)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=12)
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
    ap.add_argument("--target-step", type=int, default=None)
    ap.add_argument("--generator-index", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=0.10)
    ap.add_argument("--probe-steps", type=int, default=24)
    ap.add_argument("--poke-mode", choices=["uniform", "dipole", "breathing", "circulation", "random_control"], default="breathing")
    ap.add_argument("--json-out", type=str, default="gpu_mesoscape_organizer_collective_response.json")
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
        fission_fraction=args.fission_fraction,
        candidate_fraction=args.candidate_fraction,
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
        device=args.device,
    )

    xp, using_gpu, GM, snapshots, states_at_snapshots = simulate_mesoscape(cfg, progress=True)
    epochs = summarize_epochs(snapshots)["metastable_epochs"]
    if not epochs:
        raise RuntimeError("No valid organizer epochs found.")
    target_epoch, target_idx = choose_target_epoch(epochs, target_step=args.target_step)
    snap = snapshots[target_idx]
    state = states_at_snapshots[target_idx]

    partition = organizer_partition(
        snap["dominant_core"],
        [tuple(e) for e in snap["active_edges"]],
        state["edge_strengths"],
    )

    curves, summary = probe_collective_response(
        psi0=state["psi"],
        partition=partition,
        active_nodes=state["active_nodes"],
        active_edges=[tuple(e) for e in snap["active_edges"]],
        local_coeffs=state["local_coeffs"],
        edge_strengths=state["edge_strengths"],
        GM=GM,
        xp=xp,
        cfg=cfg,
        poke_mode=args.poke_mode,
        generator_index=args.generator_index,
        epsilon=args.epsilon,
        probe_steps=args.probe_steps,
    )

    result = {
        "config": vars(args),
        "backend": "cupy" if using_gpu else "numpy",
        "target_epoch": target_epoch,
        "target_snapshot": {
            "step": snap["step"],
            "dominant_core": snap["dominant_core"],
            "active_nodes": snap["active_nodes"],
            "active_edges": snap["active_edges"],
        },
        "partition": partition,
        "poke_mode": args.poke_mode,
        "response_summary": summary,
        "response_curves": curves,
    }

    print("=" * 96)
    print("MESOSCOPIC ORGANIZER COLLECTIVE RESPONSE WITNESS")
    print("-" * 96)
    print(f"backend={result['backend']}  poke_mode={args.poke_mode}  n_max={args.n_max}")
    print(f"target_epoch={target_epoch['core_pair']}  steps={target_epoch['start_step']}->{target_epoch['end_step']}  n_snapshots={target_epoch['n_snapshots']}")
    print(f"target_snapshot_step={snap['step']}")
    print(f"organizer_nodes={partition['organizer_nodes']}")
    print(f"core_nodes={partition['core_nodes']}  shell_nodes={partition['shell_nodes']}")
    print(f"left_nodes={partition['left_nodes']}  right_nodes={partition['right_nodes']}")
    print(f"generator_index={args.generator_index}  epsilon={args.epsilon}  probe_steps={args.probe_steps}")
    print("-" * 96)
    print(f"max_org_trace_distance         = {summary['max_org_trace_distance']:.6f}")
    print(f"mean_org_trace_distance        = {summary['mean_org_trace_distance']:.6f}")
    print(f"argmax_org_trace_step          = {summary['argmax_org_trace_distance_step']}")
    print(f"max_core_trace_distance        = {summary['max_core_trace_distance']:.6f}")
    print(f"mean_core_trace_distance       = {summary['mean_core_trace_distance']:.6f}")
    print(f"max_abs_core_shell_mi_shift    = {summary['max_abs_core_shell_mi_shift']:.6f}")
    print(f"argmax_abs_core_shell_mi_step  = {summary['argmax_abs_core_shell_mi_shift_step']}")
    if "random_control_nodes" in summary:
        print(f"random_control_nodes           = {summary['random_control_nodes']}")
    print("=" * 96)

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON: {args.json_out}")


if __name__ == "__main__":
    main()