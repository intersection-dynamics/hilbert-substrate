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


def conjugate_generators(GM, xp):
    return [-(g.T).copy() for g in GM]


def normalize_state(psi, xp):
    n = float(to_cpu_array(xp.linalg.norm(psi.reshape(-1))))
    if n <= 1e-15:
        raise ValueError("State norm vanished.")
    return psi / n


def apply_one_body(psi, op, site, xp):
    y = xp.moveaxis(psi, site, 0)
    y = xp.tensordot(op, y, axes=([1], [0]))
    y = xp.moveaxis(y, 0, site)
    return y


def apply_two_body_samegen(psi, op_a, op_b, i, j, xp):
    return apply_one_body(apply_one_body(psi, op_a, i, xp), op_b, j, xp)


def partial_trace_keep(psi, keep, n_sites, xp):
    keep = sorted(keep)
    trace_out = [i for i in range(n_sites) if i not in keep]
    perm = keep + trace_out
    psi_arr = psi
    if getattr(psi_arr, "ndim", None) != n_sites:
        psi_arr = psi_arr.reshape((3,) * n_sites)
    psi_perm = xp.transpose(psi_arr, perm)
    d_keep = 3 ** len(keep)
    d_tr = 3 ** len(trace_out)
    mat = psi_perm.reshape(d_keep, d_tr)
    return mat @ xp.conjugate(mat.T)


def von_neumann_entropy(rho, xp):
    vals = xp.linalg.eigvalsh(0.5 * (rho + xp.conjugate(rho.T)))
    vals = xp.real(vals)
    vals = xp.maximum(vals, 0.0)
    s = float(to_cpu_array(vals.sum()))
    if s <= 1e-15:
        return 0.0
    vals = vals / s
    nz = vals[vals > 1e-15]
    return float(to_cpu_array((-nz * xp.log(nz)).sum()))


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


def pair_su3_correlator_strength(psi, GM, GMbar, i, j, xp):
    vals = []
    for a in range(8):
        tmp = apply_two_body_samegen(psi, GM[a], GMbar[a], i, j, xp)
        vals.append(float(to_cpu_array(xp.real(xp.vdot(psi.reshape(-1), tmp.reshape(-1))))))
    return float(np.linalg.norm(np.asarray(vals, dtype=float)))


def active_triangles(active_nodes, active_edges):
    edge_set = set((min(i, j), max(i, j)) for i, j in active_edges)
    out = []
    for a, b, c in combinations(active_nodes, 3):
        if (min(a, b), max(a, b)) in edge_set and (min(b, c), max(b, c)) in edge_set and (min(a, c), max(a, c)) in edge_set:
            out.append((a, b, c))
    return out


def weighted_adjacency(active_nodes, active_edges, link_states):
    idx_of = {node: k for k, node in enumerate(active_nodes)}
    active_set = set(active_nodes)
    W = np.zeros((len(active_nodes), len(active_nodes)), dtype=float)
    for i, j in active_edges:
        e = (min(i, j), max(i, j))
        if i not in active_set or j not in active_set or e not in link_states:
            continue
        a, b = idx_of[i], idx_of[j]
        W[a, b] = float(link_states[e]["strength"])
        W[b, a] = float(link_states[e]["strength"])
    return W, idx_of


def spectral_1d_embedding(active_nodes, active_edges, link_states):
    if len(active_nodes) <= 1:
        return {active_nodes[0]: 0.0} if active_nodes else {}
    W, idx_of = weighted_adjacency(active_nodes, active_edges, link_states)
    deg = np.sum(W, axis=1)
    if np.allclose(deg, 0.0):
        return {node: float(k) for k, node in enumerate(active_nodes)}
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


def xp_eye(dim, xp):
    return xp.eye(dim, dtype=xp.complex128)


def kron(a, b, xp):
    return xp.kron(a, b)


def hermitize_psd_trace1(rho, xp):
    rho = 0.5 * (rho + xp.conjugate(rho.T))
    vals, vecs = xp.linalg.eigh(rho)
    vals = xp.maximum(xp.real(vals), 0.0)
    s = float(to_cpu_array(vals.sum()))
    if s <= 1e-15:
        return xp_eye(rho.shape[0], xp) / rho.shape[0]
    vals = vals / s
    return vecs @ xp.diag(vals.astype(xp.complex128)) @ xp.conjugate(vecs.T)


def single_site_generator_expectations(rho_site, GM, xp):
    out = []
    for g in GM:
        out.append(float(to_cpu_array(xp.real(xp.trace(rho_site @ g)))))
    return np.asarray(out, dtype=float)


def link_expectation_vector(rho_link, ops, left: bool, xp):
    I3 = xp_eye(3, xp)
    out = []
    for op in ops:
        full = kron(op, I3, xp) if left else kron(I3, op, xp)
        out.append(float(to_cpu_array(xp.real(xp.trace(rho_link @ full)))))
    return np.asarray(out, dtype=float)


def link_influence_matrix(rho_link, GM, GMbar, xp):
    T = np.zeros((8, 8), dtype=float)
    for a in range(8):
        for b in range(8):
            op = kron(GM[a], GMbar[b], xp)
            T[a, b] = float(to_cpu_array(xp.real(xp.trace(rho_link @ op))))
    return T


def link_bandwidth_metrics(rho_link, GM, GMbar, xp, sv_thresh=0.03):
    T = link_influence_matrix(rho_link, GM, GMbar, xp)
    s = np.linalg.svd(T, compute_uv=False)
    rank = int(np.sum(s > sv_thresh))
    score = float(np.sum(s))
    return {
        "sv": [float(x) for x in s.tolist()],
        "rank": rank,
        "score": score,
    }


def link_commitment(rho_link, xp):
    purity = float(to_cpu_array(xp.real(xp.trace(rho_link @ rho_link))))
    entropy = von_neumann_entropy(rho_link, xp)
    return purity, entropy


def init_link_state_from_endpoints(rho_i, rho_j, GM, GMbar, cfg, xp):
    vi = single_site_generator_expectations(rho_i, GM, xp)
    vj = single_site_generator_expectations(rho_j, GM, xp)
    H = xp.zeros((9, 9), dtype=xp.complex128)
    I3 = xp_eye(3, xp)
    for a in range(8):
        H = H + float(vi[a]) * kron(GM[a], I3, xp)
        H = H + float(vj[a]) * kron(I3, GMbar[a], xp)
        H = H + cfg.link_pair_scale * kron(GM[a], GMbar[a], xp)
    vals, vecs = xp.linalg.eigh(0.5 * (H + xp.conjugate(H.T)))
    k = min(3, vals.shape[0])
    weights = xp.exp(-cfg.link_init_beta * (vals - vals.min()))
    weights = weights / xp.sum(weights)
    rho = vecs @ xp.diag(weights.astype(xp.complex128)) @ xp.conjugate(vecs.T)
    return hermitize_psd_trace1(rho, xp)


def update_link_state(rho_link, rho_i, rho_j, GM, GMbar, cfg, xp):
    vi = single_site_generator_expectations(rho_i, GM, xp)
    vj = single_site_generator_expectations(rho_j, GM, xp)
    I3 = xp_eye(3, xp)
    H = xp.zeros((9, 9), dtype=xp.complex128)
    for a in range(8):
        H = H + float(cfg.link_endpoint_coupling * vi[a]) * kron(GM[a], I3, xp)
        H = H + float(cfg.link_endpoint_coupling * vj[a]) * kron(I3, GMbar[a], xp)
        H = H + float(cfg.link_pair_scale) * kron(GM[a], GMbar[a], xp)
    drho = -1j * (H @ rho_link - rho_link @ H)
    rho2 = rho_link + cfg.link_dt * drho
    if cfg.link_mix > 0.0:
        thermal = xp_eye(9, xp) / 9.0
        rho2 = (1.0 - cfg.link_mix) * rho2 + cfg.link_mix * thermal
    return hermitize_psd_trace1(rho2, xp)


def apply_hamiltonian(psi, active_nodes, active_edges, base_local_coeffs, link_states, GM, xp, cfg):
    out = xp.zeros_like(psi)
    effective = {n: np.array(base_local_coeffs[n], copy=True) for n in active_nodes}
    for i, j in active_edges:
        e = (min(i, j), max(i, j))
        if e not in link_states:
            continue
        ls = link_states[e]
        left_vec = np.asarray(ls["left_vec"], dtype=float)
        right_vec = np.asarray(ls["right_vec"], dtype=float)
        strength = float(ls["strength"])
        bw = float(ls["bandwidth_score"])
        if i == e[0]:
            effective[i] += cfg.node_link_backreaction * strength * left_vec
            effective[j] += cfg.node_link_backreaction * strength * right_vec
        else:
            effective[i] += cfg.node_link_backreaction * strength * right_vec
            effective[j] += cfg.node_link_backreaction * strength * left_vec
    for i in active_nodes:
        coeffs = effective[i]
        for a in range(8):
            c = float(coeffs[a])
            if c != 0.0:
                out = out + c * apply_one_body(psi, GM[a], i, xp)
    if cfg.direct_pair_echo > 0.0:
        for i, j in active_edges:
            e = (min(i, j), max(i, j))
            if e not in link_states:
                continue
            g = cfg.direct_pair_echo * float(link_states[e]["strength"])
            for a in range(8):
                out = out + g * apply_two_body_samegen(psi, GM[a], GM[a], i, j, xp)
    return out


def rk4_step(psi, dt, active_nodes, active_edges, base_local_coeffs, link_states, GM, xp, cfg):
    def f(state):
        return -1j * apply_hamiltonian(state, active_nodes, active_edges, base_local_coeffs, link_states, GM, xp, cfg)

    k1 = f(psi)
    k2 = f(psi + 0.5 * dt * k1)
    k3 = f(psi + 0.5 * dt * k2)
    k4 = f(psi + dt * k3)
    psi2 = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return normalize_state(psi2, xp)


def update_all_links(psi, active_edges, link_states, GM, GMbar, cfg, xp):
    one_site_rhos = {}
    touched = sorted(set([n for e in active_edges for n in e]))
    for n in touched:
        one_site_rhos[n] = partial_trace_keep(psi, [n], cfg.n_max, xp)
    for i, j in list(active_edges):
        e = (min(i, j), max(i, j))
        if e not in link_states:
            continue
        rho_link = link_states[e]["rho"]
        rho_link = update_link_state(rho_link, one_site_rhos[i], one_site_rhos[j], GM, GMbar, cfg, xp)
        metrics = link_bandwidth_metrics(rho_link, GM, GMbar, xp, cfg.link_sv_thresh)
        purity, entropy = link_commitment(rho_link, xp)
        left_vec = link_expectation_vector(rho_link, GM, True, xp)
        right_vec = link_expectation_vector(rho_link, GMbar, False, xp)
        strength = float(cfg.link_strength_base + cfg.link_strength_bw_weight * metrics["score"] + cfg.link_strength_purity_weight * purity)
        link_states[e].update({
            "rho": rho_link,
            "bandwidth_rank": int(metrics["rank"]),
            "bandwidth_score": float(metrics["score"]),
            "bandwidth_sv": metrics["sv"],
            "purity": float(purity),
            "entropy": float(entropy),
            "left_vec": [float(x) for x in left_vec.tolist()],
            "right_vec": [float(x) for x in right_vec.tolist()],
            "strength": strength,
        })


def evolve_windows(psi, n_windows, cfg, active_nodes, active_edges, base_local_coeffs, link_states, GM, GMbar, xp):
    out = psi
    for _ in range(n_windows):
        out = rk4_step(out, cfg.dt, active_nodes, active_edges, base_local_coeffs, link_states, GM, xp, cfg)
        update_all_links(out, active_edges, link_states, GM, GMbar, cfg, xp)
    return out


def sanitize_graph_state(active_nodes, dormant_nodes, active_edges, link_states):
    active_nodes[:] = sorted(dict.fromkeys(active_nodes))
    active_set = set(active_nodes)
    if dormant_nodes is not None:
        dormant_nodes[:] = sorted(n for n in dict.fromkeys(dormant_nodes) if n not in active_set)
    kept_edges = []
    seen = set()
    for i, j in active_edges:
        e = (min(i, j), max(i, j))
        if i not in active_set or j not in active_set:
            link_states.pop(e, None)
            continue
        if e not in link_states or e in seen:
            continue
        kept_edges.append(e)
        seen.add(e)
    active_edges[:] = kept_edges


@dataclass
class SimConfig:
    n_max: int
    n_init: int
    seed: int
    local_scale: float
    total_steps: int
    dt: float
    eval_every: int
    settling_windows: int
    lookahead_windows: int
    candidate_fraction: float
    birth_score_floor: float
    max_birth_candidates: int
    max_births_per_eval: int
    birth_exclusion_radius: float
    max_live_reabsorbs: int
    live_reabsorb_every: int
    reabs_fidelity_floor: float
    reabs_basis0_floor: float
    min_active_edges_floor: int
    min_shell_size_floor: int
    shell_edge_protection: float
    edge_bandwidth_prune_thresh: float
    edge_commitment_prune_thresh: float
    direct_pair_echo: float
    node_link_backreaction: float
    link_dt: float
    link_pair_scale: float
    link_endpoint_coupling: float
    link_mix: float
    link_init_beta: float
    link_sv_thresh: float
    link_strength_base: float
    link_strength_bw_weight: float
    link_strength_purity_weight: float
    birth_structural_bias: float
    birth_edge_restore_bonus: float
    progress_every: int
    device: str
    json_out: str


def candidate_features(psi, active_nodes, active_edges, link_states, n_sites, GM, GMbar, cfg, xp):
    # Births should nucleate off committed interfaces. In the link-faithful toy,
    # that means active edges are the parent interfaces, not arbitrary non-edge pairs.
    coords = spectral_1d_embedding(active_nodes, active_edges, link_states)
    adj = {i: [] for i in active_nodes}
    for i, j in active_edges:
        adj[i].append(j)
        adj[j].append(i)
    triangles = active_triangles(active_nodes, active_edges)
    edge_set = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    rows = []
    bootstrap = len(active_nodes) <= 3

    for i, j in active_edges:
        e = (min(i, j), max(i, j))
        if e not in link_states:
            continue

        rho_ab = partial_trace_keep(psi, [i, j], n_sites, xp)
        rho_a = partial_trace_keep(psi, [i], n_sites, xp)
        rho_b = partial_trace_keep(psi, [j], n_sites, xp)
        mi = float(von_neumann_entropy(rho_a, xp) + von_neumann_entropy(rho_b, xp) - von_neumann_entropy(rho_ab, xp))
        corr = pair_su3_correlator_strength(psi, GM, GMbar, i, j, xp)
        coord_gap = abs(coords.get(i, 0.0) - coords.get(j, 0.0))
        common_nbrs = sorted(set(adj[i]).intersection(adj[j]))
        cmi_mean = 0.0
        if common_nbrs:
            cmis = [conditional_mutual_information_from_state(psi, i, k, j, n_sites, xp) for k in common_nbrs]
            cmi_mean = float(sum(cmis) / len(cmis))

        daughter_count = 0
        for node in active_nodes:
            if node in (i, j):
                continue
            if (min(i, node), max(i, node)) in edge_set and (min(j, node), max(j, node)) in edge_set:
                daughter_count += 1

        shell_triangle_count = 0
        for tri in triangles:
            if i in tri and j in tri:
                shell_triangle_count += 1

        ls = link_states[e]
        bw = float(ls.get("bandwidth_score", 0.0))
        rank = int(ls.get("bandwidth_rank", 0))
        purity = float(ls.get("purity", 1.0))
        strength = float(ls.get("strength", 0.0))

        # Bootstrap rule: early on, strong parent-pair MI/corr should be enough to ignite
        # growth. As the graph gets richer, explicit link bandwidth/commitment carries more weight.
        if bootstrap:
            link_factor = 1.0 + 0.35 * bw + 0.10 * rank + 0.15 * purity + 0.10 * strength
        else:
            link_factor = 1.0 + 0.90 * bw + 0.20 * rank + 0.25 * purity + 0.15 * strength

        score = float(
            max(mi, 0.0) * max(corr, 0.0)
            * (1.0 + cmi_mean)
            * (1.0 + cfg.birth_structural_bias * daughter_count)
            * (1.0 + 0.5 * cfg.birth_structural_bias * shell_triangle_count)
            * link_factor
            * (1.0 + cfg.birth_edge_restore_bonus / max(1.0, 1.0 + coord_gap))
        )

        rows.append({
            "pair": [i, j],
            "mi": mi,
            "corr": corr,
            "coord_gap": coord_gap,
            "common_nbr_count": int(len(common_nbrs)),
            "common_nbrs": common_nbrs,
            "cmi_mean": cmi_mean,
            "daughter_count": int(daughter_count),
            "shell_triangle_count": int(shell_triangle_count),
            "pair_center": float(0.5 * (coords.get(i, 0.0) + coords.get(j, 0.0))),
            "pair_bandwidth_score": bw,
            "pair_bandwidth_rank": rank,
            "pair_link_purity": purity,
            "pair_link_strength": strength,
            "bootstrap_mode": bool(bootstrap),
            "score": score,
        })

    rows.sort(key=lambda d: d["score"], reverse=True)
    return rows


def choose_candidate_births(rows, dormant_nodes, cfg):
    if not rows or not dormant_nodes:
        return []
    n_considered = max(1, int(np.ceil(cfg.candidate_fraction * len(rows))))
    considered = [r for r in rows[:n_considered] if r["score"] >= cfg.birth_score_floor]
    if not considered:
        return []
    considered = considered[: cfg.max_birth_candidates]
    chosen = []
    used_nodes = set()
    used_common = set()
    used_centers: List[float] = []
    dormant_iter = iter(dormant_nodes)
    for row in considered:
        if len(chosen) >= min(cfg.max_births_per_eval, len(dormant_nodes)):
            break
        i, j = row["pair"]
        if i in used_nodes or j in used_nodes:
            continue
        common = set(row["common_nbrs"])
        if common & used_common:
            continue
        center = float(row["pair_center"])
        if any(abs(center - c) < cfg.birth_exclusion_radius for c in used_centers):
            continue
        try:
            new_node = next(dormant_iter)
        except StopIteration:
            break
        chosen.append((row, new_node))
        used_nodes.update([i, j])
        used_common |= common
        used_centers.append(center)
    return chosen


def spawn_births(chosen, psi, active_nodes, dormant_nodes, active_edges, link_states, base_local_coeffs, GM, GMbar, cfg, rng, eval_index, xp):
    events = []
    for row, new_node in chosen:
        i, j = row["pair"]
        if new_node not in dormant_nodes:
            continue
        dormant_nodes.remove(new_node)
        active_nodes.append(new_node)
        active_nodes.sort()
        base_local_coeffs[new_node] = rng.uniform(-cfg.local_scale, cfg.local_scale, size=8)
        rho_i = partial_trace_keep(psi, [i], cfg.n_max, xp)
        rho_j = partial_trace_keep(psi, [j], cfg.n_max, xp)
        rho_new = partial_trace_keep(psi, [new_node], cfg.n_max, xp)
        for parent, rho_parent in ((i, rho_i), (j, rho_j)):
            e = (min(parent, new_node), max(parent, new_node))
            if e not in active_edges:
                active_edges.append(e)
            rho_link = init_link_state_from_endpoints(rho_parent, rho_new, GM, GMbar, cfg, xp)
            metrics = link_bandwidth_metrics(rho_link, GM, GMbar, xp, cfg.link_sv_thresh)
            purity, entropy = link_commitment(rho_link, xp)
            left_vec = link_expectation_vector(rho_link, GM, True, xp)
            right_vec = link_expectation_vector(rho_link, GMbar, False, xp)
            link_states[e] = {
                "rho": rho_link,
                "bandwidth_rank": int(metrics["rank"]),
                "bandwidth_score": float(metrics["score"]),
                "bandwidth_sv": metrics["sv"],
                "purity": float(purity),
                "entropy": float(entropy),
                "left_vec": [float(x) for x in left_vec.tolist()],
                "right_vec": [float(x) for x in right_vec.tolist()],
                "strength": float(cfg.link_strength_base + cfg.link_strength_bw_weight * metrics["score"] + cfg.link_strength_purity_weight * purity),
                "birth_eval": int(eval_index),
            }
        active_edges[:] = sorted(set((min(a, b), max(a, b)) for a, b in active_edges))
        events.append({"parents": [i, j], "new_node": int(new_node), "score": float(row["score"])})
    sanitize_graph_state(active_nodes, dormant_nodes, active_edges, link_states)
    return events


def classify_births_multiwindow(events, active_nodes, active_edges, link_states, base_local_coeffs, psi, GM, GMbar, cfg, xp):
    out = []
    for evt in events:
        i, j = evt["parents"]
        n = evt["new_node"]
        psi_w = psi.copy()
        active_nodes_w = list(active_nodes)
        active_edges_w = list(active_edges)
        link_states_w = {e: {k: (v.copy() if hasattr(v, "copy") else list(v) if isinstance(v, list) else v) for k, v in d.items()} for e, d in link_states.items()}
        sanitize_graph_state(active_nodes_w, None, active_edges_w, link_states_w)
        mi_hist, shell_hist, common_hist, entropy_hist, rank_hist = [], [], [], [], []
        for _ in range(cfg.lookahead_windows):
            psi_w = evolve_windows(psi_w, 1, cfg, active_nodes_w, active_edges_w, base_local_coeffs=base_local_coeffs, link_states=link_states_w, GM=GM, GMbar=GMbar, xp=xp)
            if n not in active_nodes_w:
                break
            mi_i = mutual_information_from_state(psi_w, i, n, cfg.n_max, xp) if (min(i, n), max(i, n)) in link_states_w else 0.0
            mi_j = mutual_information_from_state(psi_w, j, n, cfg.n_max, xp) if (min(j, n), max(j, n)) in link_states_w else 0.0
            mi_hist.append(0.5 * (mi_i + mi_j))
            rho_n = partial_trace_keep(psi_w, [n], cfg.n_max, xp)
            entropy_hist.append(von_neumann_entropy(rho_n, xp))
            adj = {a: [] for a in active_nodes_w}
            for a, b in active_edges_w:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            common_hist.append(len(set(adj.get(i, [])).intersection(adj.get(j, []))))
            shell_t = 0
            for tri in active_triangles(active_nodes_w, active_edges_w):
                if n in tri and (i in tri or j in tri):
                    shell_t += 1
            shell_hist.append(shell_t)
            ranks = []
            for e in [(min(i, n), max(i, n)), (min(j, n), max(j, n))]:
                if e in link_states_w:
                    ranks.append(int(link_states_w[e].get("bandwidth_rank", 0)))
            rank_hist.append(float(np.mean(ranks)) if ranks else 0.0)
        label = "persistent" if (np.mean(mi_hist) if mi_hist else 0.0) >= 0.07 and (np.mean(rank_hist) if rank_hist else 0.0) >= 1.0 and (np.mean(shell_hist) if shell_hist else 0.0) >= 0.5 else "remerge_prone"
        out.append({
            "parents": [i, j],
            "new_node": int(n),
            "mean_birth_mi": float(np.mean(mi_hist)) if mi_hist else 0.0,
            "mean_new_node_entropy": float(np.mean(entropy_hist)) if entropy_hist else 0.0,
            "mean_common_support": float(np.mean(common_hist)) if common_hist else 0.0,
            "mean_shell_triangles": float(np.mean(shell_hist)) if shell_hist else 0.0,
            "mean_link_rank": float(np.mean(rank_hist)) if rank_hist else 0.0,
            "label": label,
        })
    return out


def dominant_core_snapshot(psi, active_nodes, active_edges, link_states, GM, GMbar, xp, n_sites):
    if not active_edges:
        return None
    rows = []
    edge_set = set((min(i, j), max(i, j)) for i, j in active_edges)
    for i, j in active_edges:
        e = (min(i, j), max(i, j))
        if e not in link_states:
            continue
        mi = mutual_information_from_state(psi, i, j, n_sites, xp)
        corr = pair_su3_correlator_strength(psi, GM, GMbar, i, j, xp)
        bw = float(link_states[e].get("bandwidth_score", 0.0))
        rank = int(link_states[e].get("bandwidth_rank", 0))
        daughter_count = 0
        for node in active_nodes:
            if node in (i, j):
                continue
            if (min(i, node), max(i, node)) in edge_set and (min(j, node), max(j, node)) in edge_set:
                daughter_count += 1
        core_score = float(mi * corr * max(1.0, bw) * (1.0 + 0.5 * rank) * (1.0 + daughter_count))
        rows.append({"pair": [i, j], "mi": mi, "corr": corr, "bw": bw, "rank": rank, "daughter_count": daughter_count, "core_score": core_score})
    if not rows:
        return None
    rows.sort(key=lambda d: d["core_score"], reverse=True)
    best = rows[0]
    i, j = best["pair"]
    shell_nodes = set([i, j])
    shell_edges = []
    for a, b in active_edges:
        if a in (i, j) or b in (i, j):
            shell_nodes.add(a)
            shell_nodes.add(b)
            shell_edges.append((a, b))
    shell_nodes = sorted(shell_nodes)
    shell_triangles = [list(tri) for tri in active_triangles(active_nodes, active_edges) if i in tri and j in tri]
    return {
        "core_pair": [i, j],
        "core_score": float(best["core_score"]),
        "pair_mi": float(best["mi"]),
        "pair_corr": float(best["corr"]),
        "pair_bandwidth": float(best["bw"]),
        "pair_bandwidth_rank": int(best["rank"]),
        "daughter_count": int(best["daughter_count"]),
        "shell_nodes": shell_nodes,
        "shell_edges": [list(e) for e in shell_edges],
        "shell_size": len(shell_nodes),
        "shell_triangle_count": len(shell_triangles),
        "shell_triangles": shell_triangles,
    }


def organizer_nodes_from_core(core):
    return list(core["shell_nodes"]) if core else []


def graph_stats(active_nodes, active_edges):
    deg = {n: 0 for n in active_nodes}
    for a, b in active_edges:
        if a in deg: deg[a] += 1
        if b in deg: deg[b] += 1
    tri = len(active_triangles(active_nodes, active_edges))
    if active_nodes:
        mean_deg = float(np.mean(list(deg.values())))
    else:
        mean_deg = 0.0
    return {"mean_degree": mean_deg, "triangle_count": tri, "degree_map": {str(k): int(v) for k, v in deg.items()}}


def metric_snapshot(active_nodes, active_edges, link_states):
    coords = spectral_1d_embedding(active_nodes, active_edges, link_states)
    vals = list(coords.values())
    extent = float(max(vals) - min(vals)) if vals else 0.0
    edge_lengths = [abs(coords[i] - coords[j]) for i, j in active_edges if i in coords and j in coords]
    return {
        "coords": {str(k): float(v) for k, v in coords.items()},
        "metric_extent": extent,
        "total_edge_length": float(sum(edge_lengths)) if edge_lengths else 0.0,
        "mean_edge_length": float(np.mean(edge_lengths)) if edge_lengths else 0.0,
    }


def live_edge_prune(active_nodes, active_edges, link_states, core, cfg):
    if not active_edges:
        return []
    shell_edges = set(tuple(sorted(e)) for e in core.get("shell_edges", [])) if core else set()
    pruned = []
    kept = []
    for i, j in active_edges:
        e = (min(i, j), max(i, j))
        if e not in link_states:
            continue
        ls = link_states[e]
        bw = float(ls.get("bandwidth_score", 0.0))
        pur = float(ls.get("purity", 0.0))
        score = bw + cfg.shell_edge_protection * (1.0 if e in shell_edges else 0.0)
        if score < cfg.edge_bandwidth_prune_thresh or pur < cfg.edge_commitment_prune_thresh:
            if len(active_edges) - len(pruned) - 1 < cfg.min_active_edges_floor:
                kept.append(e)
                continue
            pruned.append({"edge": [i, j], "bandwidth_score": bw, "purity": pur})
            link_states.pop(e, None)
        else:
            kept.append(e)
    active_edges[:] = kept
    return pruned


def commit_reabsorptions(psi, active_nodes, dormant_nodes, active_edges, link_states, core, GM, GMbar, cfg, xp):
    if not active_nodes or cfg.max_live_reabsorbs <= 0:
        return []
    shell = set(organizer_nodes_from_core(core))
    event_list = []
    org_nodes = organizer_nodes_from_core(core)
    rho_org_before = partial_trace_keep(psi, org_nodes, cfg.n_max, xp) if org_nodes else None
    adj = {n: [] for n in active_nodes}
    for a, b in active_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    candidates = []
    for n in active_nodes:
        if n in shell:
            continue
        rho_n = partial_trace_keep(psi, [n], cfg.n_max, xp)
        basis0 = float(np.real(BASIS0.conj() @ to_cpu_array(rho_n) @ BASIS0))
        deg = len(adj.get(n, []))
        if basis0 < cfg.reabs_basis0_floor:
            continue
        if deg == 0:
            candidates.append((2.0 + basis0, n, basis0))
            continue
        incident_edges = [(min(n, m), max(n, m)) for m in adj.get(n, [])]
        removable = True
        bw_relief = 0.0
        for e in incident_edges:
            if e not in link_states:
                continue
            bw_relief += float(link_states[e].get("bandwidth_score", 0.0))
        candidates.append((basis0 + 0.2 * bw_relief - 0.1 * deg, n, basis0))
    candidates.sort(reverse=True)
    for _, n, basis0 in candidates[: cfg.max_live_reabsorbs]:
        if n not in active_nodes:
            continue
        incident = [e for e in active_edges if n in e]
        if len(active_edges) - len(incident) < cfg.min_active_edges_floor:
            continue
        active_nodes2 = [x for x in active_nodes if x != n]
        active_edges2 = [e for e in active_edges if n not in e]
        link_states2 = {e: d for e, d in link_states.items() if n not in e}
        core2 = dominant_core_snapshot(psi, active_nodes2, active_edges2, link_states2, GM, GMbar, xp, cfg.n_max)
        shell_size2 = core2["shell_size"] if core2 else 0
        if shell_size2 < cfg.min_shell_size_floor:
            continue
        if rho_org_before is not None and org_nodes:
            keep_nodes = [x for x in org_nodes if x != n]
            if not keep_nodes:
                fidelity = 1.0
            else:
                rho_after = partial_trace_keep(psi, keep_nodes, cfg.n_max, xp)
                rho_before_small = partial_trace_keep(psi, keep_nodes, cfg.n_max, xp)
                fidelity = fidelity_uhlmann(to_cpu_array(rho_before_small), to_cpu_array(rho_after))
        else:
            fidelity = 1.0
        if fidelity < cfg.reabs_fidelity_floor:
            continue
        active_nodes[:] = active_nodes2
        active_edges[:] = active_edges2
        for e in list(link_states.keys()):
            if n in e:
                link_states.pop(e, None)
        dormant_nodes.append(n)
        dormant_nodes[:] = sorted(set(dormant_nodes))
        event_list.append({"node": int(n), "basis0_return": float(basis0), "organizer_fidelity": float(fidelity), "kind": "committed_reabsorption"})
    sanitize_graph_state(active_nodes, dormant_nodes, active_edges, link_states)
    return event_list


def parse_args():
    p = argparse.ArgumentParser(description="HSF mesoscape churn v3.1 with explicit V x Vbar link registers")
    p.add_argument("--n-max", type=int, default=10)
    p.add_argument("--n-init", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local-scale", type=float, default=0.10)
    p.add_argument("--total-steps", type=int, default=160)
    p.add_argument("--dt", type=float, default=0.03)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--settling-windows", type=int, default=2)
    p.add_argument("--lookahead-windows", type=int, default=4)
    p.add_argument("--candidate-fraction", type=float, default=0.4)
    p.add_argument("--birth-score-floor", type=float, default=0.02)
    p.add_argument("--max-birth-candidates", type=int, default=6)
    p.add_argument("--max-births-per-eval", type=int, default=3)
    p.add_argument("--birth-exclusion-radius", type=float, default=0.75)
    p.add_argument("--max-live-reabsorbs", type=int, default=1)
    p.add_argument("--live-reabsorb-every", type=int, default=2)
    p.add_argument("--reabs-fidelity-floor", type=float, default=0.995)
    p.add_argument("--reabs-basis0-floor", type=float, default=0.40)
    p.add_argument("--min-active-edges-floor", type=int, default=4)
    p.add_argument("--min-shell-size-floor", type=int, default=3)
    p.add_argument("--shell-edge-protection", type=float, default=2.0)
    p.add_argument("--edge-bandwidth-prune-thresh", type=float, default=0.25)
    p.add_argument("--edge-commitment-prune-thresh", type=float, default=0.14)
    p.add_argument("--direct-pair-echo", type=float, default=0.02)
    p.add_argument("--node-link-backreaction", type=float, default=0.10)
    p.add_argument("--link-dt", type=float, default=0.04)
    p.add_argument("--link-pair-scale", type=float, default=0.08)
    p.add_argument("--link-endpoint-coupling", type=float, default=0.18)
    p.add_argument("--link-mix", type=float, default=0.02)
    p.add_argument("--link-init-beta", type=float, default=1.2)
    p.add_argument("--link-sv-thresh", type=float, default=0.03)
    p.add_argument("--link-strength-base", type=float, default=0.20)
    p.add_argument("--link-strength-bw-weight", type=float, default=0.15)
    p.add_argument("--link-strength-purity-weight", type=float, default=0.20)
    p.add_argument("--birth-structural-bias", type=float, default=0.20)
    p.add_argument("--birth-edge-restore-bonus", type=float, default=0.30)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--json-out", type=str, default="gpu_mesoscape_churn_extinction_rebirth_v3_1.json")
    a = p.parse_args()
    return SimConfig(**vars(a))


def run_sim(cfg: SimConfig):
    xp, _ = get_xp(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    GM = gell_mann(xp)
    GMbar = conjugate_generators(GM, xp)
    psi = xp.zeros((3,) * cfg.n_max, dtype=xp.complex128)
    psi[(0,) * cfg.n_max] = 1.0
    psi = normalize_state(psi, xp)

    active_nodes = list(range(cfg.n_init))
    dormant_nodes = list(range(cfg.n_init, cfg.n_max))
    active_edges = []
    link_states: Dict[Tuple[int, int], Dict] = {}
    base_local_coeffs = {n: rng.uniform(-cfg.local_scale, cfg.local_scale, size=8) for n in range(cfg.n_max)}

    if cfg.n_init >= 2:
        e = (0, 1)
        active_edges.append(e)
        rho0 = partial_trace_keep(psi, [0], cfg.n_max, xp)
        rho1 = partial_trace_keep(psi, [1], cfg.n_max, xp)
        rho_link = init_link_state_from_endpoints(rho0, rho1, GM, GMbar, cfg, xp)
        metrics = link_bandwidth_metrics(rho_link, GM, GMbar, xp, cfg.link_sv_thresh)
        purity, entropy = link_commitment(rho_link, xp)
        link_states[e] = {
            "rho": rho_link,
            "bandwidth_rank": int(metrics["rank"]),
            "bandwidth_score": float(metrics["score"]),
            "bandwidth_sv": metrics["sv"],
            "purity": float(purity),
            "entropy": float(entropy),
            "left_vec": [0.0] * 8,
            "right_vec": [0.0] * 8,
            "strength": float(cfg.link_strength_base + cfg.link_strength_bw_weight * metrics["score"] + cfg.link_strength_purity_weight * purity),
            "birth_eval": 0,
        }
    sanitize_graph_state(active_nodes, dormant_nodes, active_edges, link_states)
    update_all_links(psi, active_edges, link_states, GM, GMbar, cfg, xp)

    snapshots = []
    birth_events, birth_classifications, edge_prunes, extinction_events = [], [], [], []
    active_count_trace, edge_count_trace = [], []
    n_evals = cfg.total_steps // cfg.eval_every

    for eval_idx in range(1, n_evals + 1):
        step = eval_idx * cfg.eval_every
        psi = evolve_windows(psi, cfg.eval_every, cfg, active_nodes, active_edges, base_local_coeffs, link_states, GM, GMbar, xp)
        core = dominant_core_snapshot(psi, active_nodes, active_edges, link_states, GM, GMbar, xp, cfg.n_max)
        pruned = live_edge_prune(active_nodes, active_edges, link_states, core, cfg)
        if pruned:
            edge_prunes.extend([{**p, "step": step} for p in pruned])
        sanitize_graph_state(active_nodes, dormant_nodes, active_edges, link_states)
        rows = candidate_features(psi, active_nodes, active_edges, link_states, cfg.n_max, GM, GMbar, cfg, xp)
        chosen = choose_candidate_births(rows, dormant_nodes, cfg)
        births = spawn_births(chosen, psi, active_nodes, dormant_nodes, active_edges, link_states, base_local_coeffs, GM, GMbar, cfg, rng, eval_idx, xp)
        if births:
            birth_events.extend([{**b, "step": step} for b in births])
            birth_classifications.extend([{**x, "step": step} for x in classify_births_multiwindow(births, active_nodes, active_edges, link_states, base_local_coeffs, psi, GM, GMbar, cfg, xp)])
        if cfg.live_reabsorb_every > 0 and (eval_idx % cfg.live_reabsorb_every == 0):
            core = dominant_core_snapshot(psi, active_nodes, active_edges, link_states, GM, GMbar, xp, cfg.n_max)
            exts = commit_reabsorptions(psi, active_nodes, dormant_nodes, active_edges, link_states, core, GM, GMbar, cfg, xp)
            extinction_events.extend([{**e, "step": step} for e in exts])
        sanitize_graph_state(active_nodes, dormant_nodes, active_edges, link_states)
        core = dominant_core_snapshot(psi, active_nodes, active_edges, link_states, GM, GMbar, xp, cfg.n_max)
        m = metric_snapshot(active_nodes, active_edges, link_states)
        gs = graph_stats(active_nodes, active_edges)
        link_ranks = [int(link_states[e]["bandwidth_rank"]) for e in active_edges if e in link_states]
        link_scores = [float(link_states[e]["bandwidth_score"]) for e in active_edges if e in link_states]
        snapshots.append({
            "step": step,
            "active_nodes": list(active_nodes),
            "dormant_nodes": list(dormant_nodes),
            "active_edges": [list(e) for e in active_edges],
            "active_edge_count": len(active_edges),
            "births_this_window": len(births),
            "birth_candidates_this_window": len(rows),
            "extinctions_this_window": sum(1 for e in extinction_events if e["step"] == step),
            "mean_link_rank": float(np.mean(link_ranks)) if link_ranks else 0.0,
            "mean_link_score": float(np.mean(link_scores)) if link_scores else 0.0,
            "dominant_core": core,
            "metric": m,
            "graph_stats": gs,
        })
        active_count_trace.append(len(active_nodes))
        edge_count_trace.append(len(active_edges))
        if cfg.progress_every > 0 and (eval_idx % cfg.progress_every == 0):
            cp = core["core_pair"] if core else None
            print(f"[eval {eval_idx:03d}] step={step:4d} active={len(active_nodes):2d} dormant={len(dormant_nodes):2d} edges={len(active_edges):3d} core={cp} births={len(births)} edge_prunes={len(pruned)} extinctions={sum(1 for e in extinction_events if e['step']==step)} cand={len(rows)}")

    summary = {
        "n_birth_events": len(birth_events),
        "n_extinction_events": len(extinction_events),
        "n_edge_prunes": len(edge_prunes),
        "final_active": len(active_nodes),
        "final_dormant": len(dormant_nodes),
        "final_edge_count": len(active_edges),
        "active_count_trace": active_count_trace,
        "edge_count_trace": edge_count_trace,
        "mean_shell_size": float(np.mean([s["dominant_core"]["shell_size"] for s in snapshots if s["dominant_core"] is not None])) if snapshots else 0.0,
    }
    return {
        "config": vars(cfg),
        "summary": summary,
        "birth_events": birth_events,
        "birth_classifications": birth_classifications,
        "edge_prunes": edge_prunes,
        "extinction_events": extinction_events,
        "snapshots": snapshots,
    }


def main():
    cfg = parse_args()
    result = run_sim(cfg)
    with open(cfg.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {cfg.json_out}")


if __name__ == "__main__":
    main()
