#!/usr/bin/env python3
from __future__ import annotations

"""
derive_subsystem_birth_rule_gpu_v1.py

GPU-first redesign:
- CuPy on GPU when available, NumPy fallback otherwise
- No scipy sparse matrices
- No expm_multiply
- Hamiltonian action applied directly to the many-body state tensor
- RK4 time evolution
- Mixed-regime birth/death logic carried over from the CPU line
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression

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


def apply_hamiltonian(
    psi,
    active_nodes: List[int],
    active_edges: List[Tuple[int, int]],
    local_coeffs: Dict[int, np.ndarray],
    edge_strengths: Dict[Tuple[int, int], float],
    GM,
    xp,
):
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


def rk4_step(
    psi,
    dt: float,
    active_nodes: List[int],
    active_edges: List[Tuple[int, int]],
    local_coeffs: Dict[int, np.ndarray],
    edge_strengths: Dict[Tuple[int, int], float],
    GM,
    xp,
):
    def f(state):
        return -1j * apply_hamiltonian(state, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)

    k1 = f(psi)
    k2 = f(psi + 0.5 * dt * k1)
    k3 = f(psi + 0.5 * dt * k2)
    k4 = f(psi + dt * k3)
    psi2 = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return normalize_state(psi2, xp)


def partial_trace_keep(psi, keep: List[int], n_sites: int, xp):
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


def mutual_information_from_state(psi, i: int, j: int, n_sites: int, xp):
    rho_ab = partial_trace_keep(psi, [i, j], n_sites, xp)
    rho_a = partial_trace_keep(psi, [i], n_sites, xp)
    rho_b = partial_trace_keep(psi, [j], n_sites, xp)
    return float(von_neumann_entropy(rho_a, xp) + von_neumann_entropy(rho_b, xp) - von_neumann_entropy(rho_ab, xp))


def conditional_mutual_information_from_state(psi, i: int, k: int, j: int, n_sites: int, xp):
    rho_ikj = partial_trace_keep(psi, [i, k, j], n_sites, xp)
    rho_ik = partial_trace_keep(psi, [i, k], n_sites, xp)
    rho_kj = partial_trace_keep(psi, [k, j], n_sites, xp)
    rho_k = partial_trace_keep(psi, [k], n_sites, xp)
    return float(von_neumann_entropy(rho_ik, xp) + von_neumann_entropy(rho_kj, xp) - von_neumann_entropy(rho_k, xp) - von_neumann_entropy(rho_ikj, xp))


def pair_su3_correlator_strength(psi, GM, i: int, j: int, xp):
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
    json_out: str = ""


def candidate_features(psi, active_nodes, active_edges, edge_strengths, n_sites: int, GM, xp):
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
        sa = float(von_neumann_entropy(rho_a, xp))
        sb = float(von_neumann_entropy(rho_b, xp))
        pair_entropy = float(von_neumann_entropy(rho_ab, xp))
        coord_gap = abs(coords.get(i, 0.0) - coords.get(j, 0.0))
        common_nbrs = sorted(set(adj[i]).intersection(adj[j]))
        cmi_mean = 0.0
        cmi_max = 0.0
        if common_nbrs:
            cmis = [conditional_mutual_information_from_state(psi, i, k, j, n_sites, xp) for k in common_nbrs]
            cmi_mean = float(sum(cmis) / len(cmis))
            cmi_max = float(max(cmis))
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
            "sa": sa,
            "sb": sb,
            "pair_entropy": pair_entropy,
            "coord_gap": coord_gap,
            "common_nbr_count": int(len(common_nbrs)),
            "cmi_mean": cmi_mean,
            "cmi_max": cmi_max,
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
        events.append({
            "parents": [i, j],
            "new_node": new_node,
            "seed_features": row,
            "spawn_links": [list(e1), list(e2)],
        })
    return events


def evolve_windows(psi, n_windows, cfg: SimConfig, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp):
    for _ in range(n_windows):
        psi = rk4_step(psi, cfg.dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
    return psi


def classify_births_multiwindow(events, active_nodes, active_edges, edge_strengths, local_coeffs, psi, GM, xp, cfg: SimConfig):
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

        persistence_score = float(
            0.35 * min(1.0, windows_with_two_links / max(1, cfg.lookahead_windows)) +
            0.20 * (1.0 if mean_entropy >= cfg.persist_entropy_threshold else 0.0) +
            0.20 * (1.0 if mean_mi >= cfg.persist_mean_mi_threshold else 0.0) +
            0.15 * min(1.0, cfg.neighborhood_bonus_weight * mean_common) +
            0.10 * min(1.0, cfg.shell_bonus_weight * mean_shell)
        )

        label = "persistent" if (
            windows_with_two_links >= cfg.persist_windows_required and
            mean_mi >= cfg.mi_survival_floor and
            latest_corr_i >= cfg.corr_survival_floor and
            latest_corr_j >= cfg.corr_survival_floor and
            mean_entropy >= cfg.persist_entropy_threshold and
            strong_support
        ) else "remerge_prone"

        out.append({
            "parents": [i, j],
            "new_node": n,
            "windows_with_two_links": int(windows_with_two_links),
            "mean_new_node_entropy": mean_entropy,
            "mean_birth_mi": mean_mi,
            "mean_common_support": mean_common,
            "mean_shell_triangles": mean_shell,
            "latest_corr_i": latest_corr_i,
            "latest_corr_j": latest_corr_j,
            "persistence_score": persistence_score,
            "label": label,
        })

    return out, psi_work


def run_sim(cfg: SimConfig) -> Dict[str, object]:
    xp, using_gpu = get_xp(cfg.device)
    GM = gell_mann(xp)
    rng = np.random.default_rng(cfg.seed)

    active_nodes = list(range(cfg.n_init))
    dormant_nodes = list(range(cfg.n_init, cfg.n_max))
    active_edges = [(i, i + 1) for i in range(cfg.n_init - 1)]

    local_coeffs = {i: (rng.uniform(-cfg.local_scale, cfg.local_scale, size=8) if i < cfg.n_init else np.zeros(8, dtype=float)) for i in range(cfg.n_max)}
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

    candidate_dataset = []
    birth_events = []
    total_evals = 0

    step = 0
    while step < cfg.total_steps:
        psi = rk4_step(psi, cfg.dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
        step += 1

        if (step % cfg.eval_every) == 0:
            total_evals += 1
            rows = candidate_features(psi, active_nodes, active_edges, edge_strengths, cfg.n_max, GM, xp)
            chosen = choose_candidate_births(rows, dormant_nodes, cfg.candidate_fraction, cfg.fission_fraction, cfg.birth_score_floor)
            spawned = spawn_births(chosen, active_nodes, dormant_nodes, active_edges, edge_strengths, local_coeffs, cfg.spawn_pair_scale, rng)

            psi = evolve_windows(psi, cfg.settling_windows, cfg, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
            labeled, psi = classify_births_multiwindow(spawned, active_nodes, active_edges, edge_strengths, local_coeffs, psi, GM, xp, cfg)

            birth_by_pair = {tuple(evt["parents"]): evt for evt in labeled}
            for row in rows:
                pair = tuple(row["pair"])
                label = birth_by_pair[pair]["label"] if pair in birth_by_pair else "no_birth"
                candidate_dataset.append({**row, "step": step, "outcome": label})
            birth_events.extend(labeled)

    X = []
    y = []
    feature_names = ["mi", "corr", "sa", "sb", "pair_entropy", "coord_gap", "common_nbr_count", "cmi_mean", "cmi_max", "daughter_count", "shell_triangle_count", "score"]
    for row in candidate_dataset:
        if row["outcome"] == "no_birth":
            continue
        X.append([row[k] for k in feature_names])
        y.append(1 if row["outcome"] == "persistent" else 0)

    if len(set(y)) >= 2 and len(y) >= 8:
        clf = LogisticRegression(max_iter=4000)
        clf.fit(np.array(X, dtype=float), np.array(y, dtype=int))
        derived_rule = {
            "feature_names": feature_names,
            "intercept": float(clf.intercept_[0]),
            "coefficients": {feature_names[i]: float(clf.coef_[0][i]) for i in range(len(feature_names))},
            "n_labeled_births": len(y),
            "n_persistent": int(sum(y)),
            "n_remerge_prone": int(len(y) - sum(y)),
        }
    else:
        derived_rule = {
            "feature_names": feature_names,
            "intercept": None,
            "coefficients": {},
            "n_labeled_births": len(y),
            "n_persistent": int(sum(y)) if y else 0,
            "n_remerge_prone": int(len(y) - sum(y)) if y else 0,
            "note": "Insufficient class diversity to fit logistic rule robustly.",
        }

    summary = {
        "backend": "cupy" if using_gpu else "numpy",
        "n_evals": total_evals,
        "n_candidates": len(candidate_dataset),
        "n_birth_events": len(birth_events),
        "n_persistent_births": int(sum(1 for e in birth_events if e["label"] == "persistent")),
        "n_remerge_prone_births": int(sum(1 for e in birth_events if e["label"] == "remerge_prone")),
        "active_nodes_final": len(active_nodes),
        "active_edges_final": len(active_edges),
    }

    return {
        "config": cfg.__dict__,
        "candidate_dataset": candidate_dataset,
        "birth_events": birth_events,
        "derived_rule": derived_rule,
        "summary": summary,
        "active_nodes_final": active_nodes,
        "active_edges_final": [list(e) for e in active_edges],
    }


def pretty_report(result: Dict[str, object]) -> str:
    cfg = result["config"]
    summ = result["summary"]
    rule = result["derived_rule"]
    lines = []
    lines.append("=" * 112)
    lines.append("DERIVE SUBSYSTEM BIRTH RULE GPU (v1)")
    lines.append("-" * 112)
    lines.append(
        f"backend={summ['backend']}  n_max={cfg['n_max']}  n_init={cfg['n_init']}  seed={cfg['seed']}  "
        f"total_steps={cfg['total_steps']}  dt={cfg['dt']}  eval_every={cfg['eval_every']}"
    )
    lines.append(
        f"settling_windows={cfg['settling_windows']}  lookahead_windows={cfg['lookahead_windows']}  "
        f"fission_fraction={cfg['fission_fraction']}  candidate_fraction={cfg['candidate_fraction']}"
    )
    lines.append("-" * 112)
    lines.append(
        f"Counts: evals={summ['n_evals']}  candidates={summ['n_candidates']}  births={summ['n_birth_events']}  "
        f"persistent={summ['n_persistent_births']}  remerge_prone={summ['n_remerge_prone_births']}  "
        f"final_nodes={summ['active_nodes_final']}  final_edges={summ['active_edges_final']}"
    )
    lines.append("-" * 112)
    lines.append("Derived empirical rule:")
    if rule["intercept"] is None:
        lines.append(f"  {rule.get('note', 'No fitted rule.')}")
    else:
        lines.append(f"  logit(P[persistent]) = {rule['intercept']:.4f} + Σ c_k x_k")
        for k, v in rule["coefficients"].items():
            lines.append(f"    {k}: {v:+.4f}")
    lines.append("-" * 112)
    lines.append("Recent labeled births:")
    for evt in result["birth_events"][:12]:
        lines.append(
            f"  parents={evt['parents']}  new_node={evt['new_node']}  label={evt['label']}  "
            f"windows_with_two_links={evt['windows_with_two_links']}  mean_entropy={evt['mean_new_node_entropy']:.4f}  "
            f"mean_birth_mi={evt['mean_birth_mi']:.4f}  mean_common={evt['mean_common_support']:.2f}  "
            f"mean_shell={evt['mean_shell_triangles']:.2f}  persist_score={evt['persistence_score']:.4f}"
        )
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser(description="GPU-first total redesign of the subsystem birth-rule derivation script.")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-scale", type=float, default=0.10)
    ap.add_argument("--pair-scale", type=float, default=0.16)
    ap.add_argument("--spawn-pair-scale", type=float, default=0.12)
    ap.add_argument("--total-steps", type=int, default=120)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--lookahead-windows", type=int, default=3)
    ap.add_argument("--settling-windows", type=int, default=2)
    ap.add_argument("--candidate-fraction", type=float, default=0.45)
    ap.add_argument("--fission-fraction", type=float, default=0.30)
    ap.add_argument("--birth-score-floor", type=float, default=0.015)
    ap.add_argument("--decay-mi-threshold", type=float, default=0.05)
    ap.add_argument("--decay-corr-threshold", type=float, default=0.07)
    ap.add_argument("--neighborhood-bonus-weight", type=float, default=0.18)
    ap.add_argument("--shell-bonus-weight", type=float, default=0.20)
    ap.add_argument("--mi-survival-floor", type=float, default=0.075)
    ap.add_argument("--corr-survival-floor", type=float, default=0.085)
    ap.add_argument("--persist-windows-required", type=int, default=2)
    ap.add_argument("--persist-entropy-threshold", type=float, default=0.06)
    ap.add_argument("--persist-mean-mi-threshold", type=float, default=0.07)
    ap.add_argument("--persist-triangle-threshold", type=int, default=1)
    ap.add_argument("--json-out", type=str, default="")
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
        json_out=args.json_out,
    )
    result = run_sim(cfg)
    print(pretty_report(result))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())