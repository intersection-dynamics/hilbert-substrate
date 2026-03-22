#!/usr/bin/env python3
# filename: hsf_mesoscale_bookkeeping.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from hsf_mesoscale_physics_core import Edge, canonical_edge, to_numpy


@dataclass
class ScoreConfig:
    lambda_B: float = 0.18
    lambda_S: float = 0.12
    lambda_F: float = 0.20
    lambda_R: float = 0.35
    w_mi: float = 1.0
    w_corr: float = 0.5
    w_link: float = 0.5

    retirement_edge_weight: float = 0.30
    retirement_function_weight: float = 0.25
    retirement_bookkeeping_weight: float = 0.20
    retirement_sub_weight: float = 0.15
    retirement_core_penalty: float = 0.15
    retirement_shell_penalty: float = 0.10

    # retained only for compatibility/logging
    retirement_threshold: float = 0.66

    weaken_protected_core_penalty: float = 1.5
    weaken_shell_penalty: float = 0.0

    organizer_large_region_cutoff: int = 6
    exact_mi_cutoff: int = 12
    faithful_expr_large_n_cutoff: int = 12
    shell_reexpression_pm_min: float = 0.90
    shell_reexpression_pk_min: float = 0.90
    shell_reexpression_ps_min: float = 0.75
    birth_redundancy_penalty: float = 0.70
    birth_novelty_weight: float = 0.55
    birth_parent_relief_weight: float = 0.25
    birth_distinctness_weight: float = 0.20


@dataclass
class LocalODiffAudit:
    active_count_before: int
    active_count_after: int
    odiff_before: float
    odiff_after: float
    delta_odiff: float
    mean_pair_overlap_before: float
    mean_pair_overlap_after: float
    role_fingerprints_before: List[Dict[str, object]]
    role_fingerprints_after: List[Dict[str, object]]


def _canonicalize_psi_tensor(psi, n_sites: int, xp):
    arr = psi
    if getattr(arr, "ndim", 0) == n_sites:
        return arr
    size = int(np.prod(arr.shape))
    total_dim = 3 ** int(n_sites)
    if size != total_dim:
        raise ValueError(f"psi has size {size}, expected 3**n_sites = {total_dim} for n_sites={n_sites}")
    return arr.reshape((3,) * int(n_sites))


def partial_trace_keep(psi, keep: Sequence[int], n_sites: int, xp):
    keep = sorted(set(int(k) for k in keep))
    env = [i for i in range(n_sites) if i not in keep]
    perm = keep + env
    psi_t = _canonicalize_psi_tensor(psi, n_sites, xp)
    psi_perm = xp.transpose(psi_t, perm)
    d_keep = 3 ** len(keep)
    d_env = 3 ** len(env)
    mat = psi_perm.reshape(d_keep, d_env)
    return mat @ xp.conjugate(mat.T)


def von_neumann_entropy(rho, xp):
    vals = xp.linalg.eigvalsh(rho)
    vals = xp.real(vals)
    vals = vals[vals > 1e-12]
    if vals.size == 0:
        return 0.0
    return float(-xp.sum(vals * xp.log(vals)))


def mutual_information_from_state(psi, i: int, j: int, n_sites: int, xp):
    rho_i = partial_trace_keep(psi, [i], n_sites, xp)
    rho_j = partial_trace_keep(psi, [j], n_sites, xp)
    rho_ij = partial_trace_keep(psi, [i, j], n_sites, xp)
    return float(von_neumann_entropy(rho_i, xp) + von_neumann_entropy(rho_j, xp) - von_neumann_entropy(rho_ij, xp))


def pair_su3_correlator_strength(psi, i: int, j: int, GM, xp):
    from hsf_mesoscale_physics_core import apply_one_body

    vals = []
    for gm in GM[:3]:
        oi = apply_one_body(psi, xp.asarray(gm, dtype=xp.complex128), i, xp)
        oj = apply_one_body(psi, xp.asarray(gm, dtype=xp.complex128), j, xp)
        ei = xp.real(xp.vdot(psi, oi))
        ej = xp.real(xp.vdot(psi, oj))
        vals.append(float(abs(ei * ej)))
    return float(np.mean(vals) if vals else 0.0)


def bounded_activity_and_rank(link_reg: np.ndarray):
    s = np.linalg.svd(np.asarray(link_reg), compute_uv=False)
    if s.size == 0:
        return 0.0, 0.0
    activity = float(np.sum(s))
    p = s / (np.sum(s) + 1e-12)
    effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    return activity, effective_rank


def influence_map_from_linkreg(link_reg: np.ndarray):
    return np.linalg.svd(np.asarray(link_reg), compute_uv=False)


def _safe_pair_mi(psi, i: int, j: int, n_sites: int, xp, cfg: ScoreConfig):
    if n_sites <= cfg.exact_mi_cutoff:
        try:
            return mutual_information_from_state(psi, i, j, n_sites, xp)
        except Exception:
            return 0.0
    return 0.0


def local_mi_sum(psi, nodes: Iterable[int], active_nodes: Set[int], n_sites: int, xp, cfg: Optional[ScoreConfig] = None):
    cfg = cfg or ScoreConfig()
    nodes = sorted(set(int(n) for n in nodes if n in active_nodes))
    total = 0.0
    for idx, i in enumerate(nodes):
        for j in nodes[idx + 1 :]:
            total += _safe_pair_mi(psi, i, j, n_sites, xp, cfg)
    return float(total)


def organizer_region(core_pair: Optional[Edge], active_edges: Set[Edge], active_nodes: Set[int]):
    if core_pair is None:
        return sorted(active_nodes)
    i, j = core_pair
    region = {i, j}
    for a, b in active_edges:
        if i in (a, b) or j in (a, b):
            region.add(a)
            region.add(b)
    return sorted(region)


def _neighbor_map(active_edges: Set[Edge]):
    nbr = {}
    for a, b in active_edges:
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    return nbr


def _edge_expression_component(psi, e: Edge, link_regs: Dict[Edge, np.ndarray], GM, xp, n_sites: int, cfg: ScoreConfig):
    i, j = e
    mi = _safe_pair_mi(psi, i, j, n_sites, xp, cfg)
    corr = pair_su3_correlator_strength(psi, i, j, GM, xp)
    act, _ = bounded_activity_and_rank(link_regs[e]) if e in link_regs else (0.0, 0.0)
    return {"mi": float(mi), "corr": float(corr), "act": float(act)}


def _edge_role_structure_terms(edge: Edge, active_edges: Set[Edge]):
    i, j = edge
    nbr = _neighbor_map(active_edges)
    shared = sorted((nbr.get(i, set()) & nbr.get(j, set())) - {i, j})
    tri_redundancy = min(1.0, len(shared) / 3.0)

    deg_i = len(nbr.get(i, set()))
    deg_j = len(nbr.get(j, set()))
    route_redundancy = min(1.0, max(0, min(deg_i, deg_j) - 1) / 3.0)

    redundancy = float(0.65 * tri_redundancy + 0.35 * route_redundancy)
    novelty = float(max(0.0, 1.0 - redundancy))
    distinctness = float(max(0.0, 1.0 - tri_redundancy))
    overlap_relief = float(max(0.0, 1.0 - 0.5 * (tri_redundancy + route_redundancy)))

    return {
        "shared_neighbors": shared,
        "tri_redundancy": float(tri_redundancy),
        "route_redundancy": float(route_redundancy),
        "redundancy": redundancy,
        "novelty": novelty,
        "distinctness": distinctness,
        "overlap_relief": overlap_relief,
    }


def _site_role_fingerprint(site_id: int, active_edges: Set[Edge], edge_strengths: Dict[Edge, float]):
    nbr = _neighbor_map(active_edges)
    neighbors = sorted(nbr.get(site_id, set()))
    incident = [canonical_edge(site_id, j) for j in neighbors]
    incident_strengths = [float(edge_strengths.get(e, 0.0)) for e in incident]

    mean_strength = float(np.mean(incident_strengths)) if incident_strengths else 0.0
    activity_sum = float(np.sum(incident_strengths)) if incident_strengths else 0.0
    incident_count = int(len(incident))

    existing = 0
    possible = 0
    for idx, a in enumerate(neighbors):
        for b in neighbors[idx + 1 :]:
            possible += 1
            if canonical_edge(a, b) in active_edges:
                existing += 1
    local_cluster = float(existing / possible) if possible > 0 else 0.0
    sibling_count_norm = float(min(1.0, max(0, incident_count - 1) / 4.0))

    novelty = float(max(0.0, 1.0 - 0.55 * sibling_count_norm - 0.45 * local_cluster))
    relief = float(min(1.0, 0.5 * mean_strength + 0.3 * local_cluster + 0.2 * min(1.0, incident_count / 3.0)))
    distinctness = float(max(0.0, 1.0 - local_cluster))
    weight = float(0.40 * novelty + 0.35 * relief + 0.25 * distinctness)

    parent_anchor = neighbors[:2] if len(neighbors) >= 2 else neighbors

    return {
        "site_id": int(site_id),
        "role_id": f"site_{int(site_id)}",
        "parent_anchor": [int(x) for x in parent_anchor],
        "novelty": float(novelty),
        "relief": float(relief),
        "distinctness": float(distinctness),
        "weight": float(weight),
        "raw_metrics": {
            "incident_count": float(incident_count),
            "mean_strength": float(mean_strength),
            "activity_sum": float(activity_sum),
            "rank_mean": float(min(1.0, incident_count / 3.0)),
            "local_cluster": float(local_cluster),
            "sibling_count_norm": float(sibling_count_norm),
        },
    }


def _local_region_nodes(move_kind: str, obj, active_nodes_before: Set[int], active_edges_before: Set[Edge], active_nodes_after: Set[int], active_edges_after: Set[Edge], core_before):
    cp = tuple(core_before["core_pair"]) if core_before else None
    region = set()
    if cp is not None:
        region.update(cp)

    if move_kind == "birth":
        parents = tuple(obj["parents"])
        child = int(obj["child"])
        region.update(parents)
        region.add(child)
        nbr = _neighbor_map(active_edges_before | active_edges_after)
        for p in parents:
            region.update(nbr.get(int(p), set()))
        region.update(nbr.get(child, set()))

    elif move_kind in ("weaken", "transfer"):
        if isinstance(obj, dict) and "edge" in obj:
            e = canonical_edge(*obj["edge"])
        else:
            e = canonical_edge(*obj)
        region.update(e)
        nbr = _neighbor_map(active_edges_before | active_edges_after)
        region.update(nbr.get(e[0], set()))
        region.update(nbr.get(e[1], set()))

    elif move_kind == "retire":
        node = int(obj)
        region.add(node)
        nbr = _neighbor_map(active_edges_before | active_edges_after)
        region.update(nbr.get(node, set()))

    return sorted(int(n) for n in region if n in (active_nodes_before | active_nodes_after))


def _mean_pair_overlap(region_nodes: Sequence[int], active_edges: Set[Edge]):
    if len(region_nodes) < 2:
        return 0.0
    nbr = _neighbor_map(active_edges)
    vals = []
    nodes = sorted(set(int(n) for n in region_nodes))
    for idx, a in enumerate(nodes):
        na = set(nbr.get(a, set()))
        for b in nodes[idx + 1 :]:
            nb = set(nbr.get(b, set()))
            union = na | nb
            inter = na & nb
            vals.append(len(inter) / max(1, len(union)))
    return float(np.mean(vals)) if vals else 0.0


def _odiff_from_fingerprints(fps: List[Dict[str, object]]) -> float:
    return float(sum(float(fp.get("weight", 0.0)) for fp in fps))


def compute_move_local_odiff_audit(
    move_kind: str,
    obj,
    before_state,
    after_state,
    core_before,
    cfg: ScoreConfig,
):
    psi_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = before_state
    psi_a, active_nodes_a, dormant_nodes_a, active_edges_a, local_coeffs_a, edge_strengths_a, link_regs_a = after_state

    region_nodes = _local_region_nodes(move_kind, obj, active_nodes_b, active_edges_b, active_nodes_a, active_edges_a, core_before)

    fps_before = [_site_role_fingerprint(n, active_edges_b, edge_strengths_b) for n in region_nodes if n in active_nodes_b]
    fps_after = [_site_role_fingerprint(n, active_edges_a, edge_strengths_a) for n in region_nodes if n in active_nodes_a]

    odiff_before = _odiff_from_fingerprints(fps_before)
    odiff_after = _odiff_from_fingerprints(fps_after)

    return LocalODiffAudit(
        active_count_before=int(len(active_nodes_b)),
        active_count_after=int(len(active_nodes_a)),
        odiff_before=float(odiff_before),
        odiff_after=float(odiff_after),
        delta_odiff=float(odiff_after - odiff_before),
        mean_pair_overlap_before=float(_mean_pair_overlap(region_nodes, active_edges_b)),
        mean_pair_overlap_after=float(_mean_pair_overlap(region_nodes, active_edges_a)),
        role_fingerprints_before=fps_before,
        role_fingerprints_after=fps_after,
    )


def dominant_core_snapshot(
    psi,
    active_nodes: Set[int],
    active_edges: Set[Edge],
    edge_strengths: Dict[Edge, float],
    GM,
    xp,
    n_sites: int,
    cfg: Optional[ScoreConfig] = None,
    link_regs: Optional[Dict[Edge, np.ndarray]] = None,
):
    cfg = cfg or ScoreConfig()
    best = None
    for e in sorted(active_edges):
        i, j = e
        strength = float(edge_strengths.get(e, 0.0))
        corr = pair_su3_correlator_strength(psi, i, j, GM, xp)
        mi = _safe_pair_mi(psi, i, j, n_sites, xp, cfg)
        act, rank = bounded_activity_and_rank(link_regs[e]) if link_regs and e in link_regs else (0.0, 0.0)
        score = 1.2 * corr + 0.8 * strength + 0.3 * act + 0.15 * rank + (1.0 * mi if n_sites <= cfg.exact_mi_cutoff else 0.0)
        if best is None or score > best["score"]:
            best = {
                "core_pair": [int(i), int(j)],
                "score": float(score),
                "mi": float(mi),
                "corr": float(corr),
                "strength": float(strength),
                "activity": float(act),
                "rank": float(rank),
                "proxy_core": bool(n_sites > cfg.exact_mi_cutoff),
            }
    return best


def fast_organizer_identity(psi_before, psi_after, nodes_R: Sequence[int], n_sites: int, xp, cutoff: int):
    nodes_R = sorted(set(int(n) for n in nodes_R))
    if len(nodes_R) <= cutoff:
        rho_a = partial_trace_keep(psi_before, nodes_R, n_sites, xp)
        rho_b = partial_trace_keep(psi_after, nodes_R, n_sites, xp)
        num = float(np.real(np.sum(to_numpy(rho_a) * np.conjugate(to_numpy(rho_b)))))
        den_a = float(np.real(np.sum(np.abs(to_numpy(rho_a)) ** 2)))
        den_b = float(np.real(np.sum(np.abs(to_numpy(rho_b)) ** 2)))
        return num / (np.sqrt(den_a * den_b) + 1e-12)
    feats_a = []
    feats_b = []
    for n in nodes_R:
        rho1_a = partial_trace_keep(psi_before, [n], n_sites, xp)
        rho1_b = partial_trace_keep(psi_after, [n], n_sites, xp)
        feats_a.append(np.real(np.diag(to_numpy(rho1_a))))
        feats_b.append(np.real(np.diag(to_numpy(rho1_b))))
    va = np.concatenate(feats_a) if feats_a else np.zeros(1)
    vb = np.concatenate(feats_b) if feats_b else np.zeros(1)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def birth_justification_witness(obj, before_state, after_state, cfg: ScoreConfig, GM, xp):
    parents, child = tuple(obj["parents"]), int(obj["child"])
    psi_b, active_nodes_b, _, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = before_state
    p_edge = canonical_edge(*parents)
    nbr = _neighbor_map(active_edges_b)
    siblings = sorted((nbr.get(parents[0], set()) & nbr.get(parents[1], set())) - set(parents))
    redundant_siblings = len(siblings)
    corr_parent = pair_su3_correlator_strength(psi_b, parents[0], parents[1], GM, xp)
    act_parent, rank_parent = bounded_activity_and_rank(link_regs_b[p_edge]) if p_edge in link_regs_b else (0.0, 0.0)
    channel_capacity = 9.0
    novelty = max(0.0, 1.0 - redundant_siblings / channel_capacity)
    redundancy_similarity = 0.0 if redundant_siblings == 0 else min(1.0, redundant_siblings / 4.0)
    distinctness = max(0.0, 1.0 - redundancy_similarity)
    parent_relief = min(1.0, 0.5 * corr_parent + 0.25 * act_parent + 0.25 * (rank_parent / 3.0))
    birth_justification = max(
        0.0,
        cfg.birth_novelty_weight * novelty
        + cfg.birth_parent_relief_weight * parent_relief
        + cfg.birth_distinctness_weight * distinctness,
    )
    return {
        "birth_parent_pair": [int(parents[0]), int(parents[1])],
        "birth_redundant_siblings": int(redundant_siblings),
        "birth_redundancy_similarity": float(redundancy_similarity),
        "birth_novelty": float(novelty),
        "birth_parent_relief": float(parent_relief),
        "birth_justification": float(birth_justification),
        "birth_distinctness": float(distinctness),
    }


def local_expression(
    psi,
    active_nodes: Set[int],
    active_edges: Set[Edge],
    edge_strengths: Dict[Edge, float],
    link_regs: Dict[Edge, np.ndarray],
    cfg: ScoreConfig,
    GM,
    xp,
    n_sites: int,
):
    if not active_edges:
        return 0.0

    total = 0.0
    for e in sorted(active_edges):
        comp = _edge_expression_component(psi, e, link_regs, GM, xp, n_sites, cfg)
        role = _edge_role_structure_terms(e, active_edges)

        strength = float(edge_strengths.get(e, 0.0))
        corr = float(comp["corr"])
        act = float(comp["act"])
        mi = float(comp["mi"])

        base_signal = 0.45 * corr + 0.20 * act + 0.20 * mi + 0.15 * strength
        role_credit = base_signal * (0.40 + 0.35 * role["novelty"] + 0.25 * role["overlap_relief"])
        distinct_bonus = (0.10 * base_signal + 0.05 * strength) * role["distinctness"]
        redundancy_penalty = cfg.birth_redundancy_penalty * (role["redundancy"] ** 2) * max(
            0.0,
            0.75 * corr + 0.25 * act - 0.35 * mi,
        )

        total += role_credit + distinct_bonus - redundancy_penalty

    if active_nodes:
        deg = {i: 0 for i in active_nodes}
        for a, b in active_edges:
            if a in deg:
                deg[a] += 1
            if b in deg:
                deg[b] += 1
        coverage = np.array([min(1.0, d / 2.0) for d in deg.values()], dtype=float)
        total += 0.03 * float(np.sum(coverage))

    return float(total)


def bandwidth_burden(active_edges: Set[Edge], link_regs: Dict[Edge, np.ndarray]):
    ranks = []
    for e in active_edges:
        _, r = bounded_activity_and_rank(link_regs[e])
        ranks.append(r)
    return float(np.mean(ranks) if ranks else 0.0)


def spread_burden(active_nodes: Set[int], active_edges: Set[Edge]):
    if not active_nodes:
        return 0.0
    deg = {i: 0 for i in active_nodes}
    for i, j in active_edges:
        if i in deg:
            deg[i] += 1
        if j in deg:
            deg[j] += 1
    vals = np.array(list(deg.values()), dtype=float)
    return float(np.std(vals) / (np.mean(vals) + 1e-12) if vals.size else 0.0)


def shell_nodes(core_pair: Optional[Edge], active_edges: Set[Edge]):
    if core_pair is None:
        return set()
    i, j = core_pair
    out = set()
    for a, b in active_edges:
        if i in (a, b) or j in (a, b):
            out.add(a)
            out.add(b)
    out.discard(i)
    out.discard(j)
    return out


def _shell_indispensability(
    node: int,
    active_edges: Set[Edge],
    edge_strengths: Dict[Edge, float],
    core_before,
):
    """
    Function-sensitive shell protection based on counterfactual replaceability.

    High only when the shell node is actually hard to remove without disrupting:
    - core-to-noncore connectivity,
    - neighbor-neighbor reroutability,
    - or a uniquely nonredundant local role.

    Lower when the neighbors are already well-connected without the node.
    """
    cp = tuple(core_before["core_pair"]) if core_before else None
    if cp is None:
        return 0.0

    shell = shell_nodes(cp, active_edges)
    if node not in shell:
        return 0.0

    nbr = _neighbor_map(active_edges)
    neighbors = sorted(nbr.get(node, set()))
    if not neighbors:
        return 0.0

    incident = [canonical_edge(node, j) for j in neighbors]
    incident_strength = float(np.mean([edge_strengths.get(e, 0.0) for e in incident])) if incident else 0.0

    core_neighbors = sorted([n for n in neighbors if n in cp])
    noncore_neighbors = sorted([n for n in neighbors if n not in cp])

    existing = 0
    possible = 0
    for idx, a in enumerate(neighbors):
        for b in neighbors[idx + 1 :]:
            possible += 1
            if canonical_edge(a, b) in active_edges:
                existing += 1
    neighbor_cluster = float(existing / possible) if possible > 0 else 1.0

    cross_existing = 0
    cross_possible = 0
    for a in core_neighbors:
        for b in noncore_neighbors:
            cross_possible += 1
            if canonical_edge(a, b) in active_edges:
                cross_existing += 1
    cross_replaceability = float(cross_existing / cross_possible) if cross_possible > 0 else 1.0

    bridge_unique = 0.0
    if core_neighbors and noncore_neighbors:
        bridge_unique = max(0.0, 1.0 - cross_replaceability)
    elif core_neighbors and len(neighbors) == 1:
        bridge_unique = 0.6

    degree_term = min(1.0, len(neighbors) / 3.0)
    nonredundancy = max(0.0, 1.0 - 0.5 * neighbor_cluster - 0.5 * cross_replaceability)

    indispensability = (
        0.40 * bridge_unique
        + 0.30 * nonredundancy
        + 0.15 * min(1.0, incident_strength / 0.12)
        + 0.15 * degree_term
    )

    return float(np.clip(indispensability, 0.0, 1.0))


def no_refolding_witness(
    kind: str,
    obj,
    psi_before,
    psi_after,
    active_nodes_before: Set[int],
    active_edges_before: Set[Edge],
    edge_strengths_before: Dict[Edge, float],
    active_nodes_after: Set[int],
    active_edges_after: Set[Edge],
    edge_strengths_after: Dict[Edge, float],
    core_before,
    cfg: ScoreConfig,
    GM,
    xp,
    n_sites: int,
):
    cp = tuple(core_before["core_pair"]) if core_before else None
    nodes_R = organizer_region(cp, active_edges_before, active_nodes_before)
    region_set = set(nodes_R)
    region_edges_before = {e for e in active_edges_before if e[0] in region_set and e[1] in region_set}
    region_edges_after = {e for e in active_edges_after if e[0] in region_set and e[1] in region_set}
    changed = len(region_edges_before.symmetric_difference(region_edges_after))
    denom = max(1, len(region_edges_before) + len(region_edges_after))
    w_ep = changed / denom

    slack_before = max(0, len(region_edges_before) - max(0, len(region_set) - 1))
    slack_after = max(0, len(region_edges_after) - max(0, len(region_set.intersection(active_nodes_after)) - 1))
    w_slack = max(0.0, float(slack_after - slack_before)) / max(1, len(region_set))

    f_org = fast_organizer_identity(
        psi_before,
        psi_after,
        nodes_R,
        n_sites,
        xp,
        min(cfg.organizer_large_region_cutoff, 3 if n_sites > cfg.exact_mi_cutoff else cfg.organizer_large_region_cutoff),
    )
    mi_before = local_mi_sum(psi_before, nodes_R, active_nodes_before, n_sites, xp, cfg)
    mi_after = local_mi_sum(psi_after, nodes_R, active_nodes_after, n_sites, xp, cfg)
    mi_loss = max(0.0, mi_before - mi_after) / (abs(mi_before) + 1e-12) if mi_before > 0 else 0.0
    w_func = 0.5 * max(0.0, 1.0 - f_org) + 0.5 * mi_loss

    w_sector = 0.0
    hits_core_pair = False
    shell_weaken = False
    core_weaken = False
    destructive_weaken = False
    lawful_shell_reexpression = False

    if kind == "weaken" and cp is not None:
        e = canonical_edge(*obj)
        hits_core_pair = e == canonical_edge(*cp)
        core_weaken = hits_core_pair
        shell_weaken = (not hits_core_pair) and (cp[0] in e or cp[1] in e)
        P_M = max(0.0, 1.0 - w_func)
        P_K = max(0.0, 1.0 - w_ep)
        shell_before = shell_nodes(cp, active_edges_before)
        shell_after = shell_nodes(cp, active_edges_after)
        overlap = len(shell_before & shell_after) / max(1, len(shell_before | shell_after)) if (shell_before or shell_after) else 1.0
        P_S = overlap
        if hits_core_pair:
            w_sector += cfg.weaken_protected_core_penalty
        elif shell_weaken:
            lawful_shell_reexpression = (
                P_M >= cfg.shell_reexpression_pm_min
                and P_K >= cfg.shell_reexpression_pk_min
                and P_S >= cfg.shell_reexpression_ps_min
            )
            if not lawful_shell_reexpression:
                w_sector += cfg.weaken_shell_penalty
        if e in active_edges_before and e not in active_edges_after:
            destructive_weaken = True
            w_sector += 1.0

    if kind == "retire" and cp is not None and int(obj) in cp:
        w_sector += cfg.weaken_protected_core_penalty

    w_total = float(w_ep + w_slack + w_func + w_sector)
    return {
        "W_ep": float(w_ep),
        "W_slack": float(w_slack),
        "W_func": float(w_func),
        "W_sector": float(w_sector),
        "W_NR": float(w_total),
        "F_org": float(f_org),
        "organizer_region_size": int(len(nodes_R)),
        "nodes_R": [int(n) for n in nodes_R],
        "hits_core_pair": bool(hits_core_pair),
        "core_weaken": bool(core_weaken),
        "shell_weaken": bool(shell_weaken),
        "destructive_weaken": bool(destructive_weaken),
        "lawful_shell_reexpression": bool(lawful_shell_reexpression),
    }


def retirement_readiness(
    node: int,
    psi,
    active_nodes: Set[int],
    active_edges: Set[Edge],
    edge_strengths: Dict[Edge, float],
    link_regs: Dict[Edge, np.ndarray],
    core_before,
    cfg: ScoreConfig,
    GM,
    xp,
    n_sites: int,
):
    incident = [e for e in active_edges if node in e]
    if not incident:
        return {
            "edge_ready": 1.0,
            "functional_ready": 1.0,
            "bookkeeping_safety": 1.0,
            "substitutability": 1.0,
            "core_penalty": 0.0,
            "shell_penalty": 0.0,
            "shell_indispensability": 0.0,
            "retirement_readiness": 1.0,
            "eligible": True,
        }

    strengths = np.array([edge_strengths[e] for e in incident], dtype=float)
    edge_ready = float(max(0.0, 1.0 - np.mean(strengths)))

    corr_sum = 0.0
    act_sum = 0.0
    nbrs = []
    for e in incident:
        other = e[1] if e[0] == node else e[0]
        nbrs.append(other)
        corr_sum += pair_su3_correlator_strength(psi, node, other, GM, xp)
        act, _ = bounded_activity_and_rank(link_regs[e]) if e in link_regs else (0.0, 0.0)
        act_sum += act
    functional_ready = float(1.0 / (1.0 + corr_sum + 0.1 * act_sum))

    nbrs = sorted(set(nbrs))
    existing = 0
    possible = 0
    for idx, a in enumerate(nbrs):
        for b in nbrs[idx + 1 :]:
            possible += 1
            if canonical_edge(a, b) in active_edges:
                existing += 1
    bookkeeping_safety = float(existing / possible) if possible > 0 else 1.0

    deg = {i: 0 for i in active_nodes}
    for a, b in active_edges:
        if a in deg:
            deg[a] += 1
        if b in deg:
            deg[b] += 1
    sub = np.mean([max(0, deg[n] - 1) for n in nbrs]) if nbrs else 1.0
    substitutability = float(sub / (sub + 1.0))

    cp = tuple(core_before["core_pair"]) if core_before else None
    core_penalty = 1.0 if cp is not None and node in cp else 0.0

    shell_indispensability = _shell_indispensability(node, active_edges, edge_strengths, core_before)
    shell_penalty = float(shell_indispensability)

    readiness = (
        cfg.retirement_edge_weight * edge_ready
        + cfg.retirement_function_weight * functional_ready
        + cfg.retirement_bookkeeping_weight * bookkeeping_safety
        + cfg.retirement_sub_weight * substitutability
        - cfg.retirement_core_penalty * core_penalty
        - cfg.retirement_shell_penalty * shell_penalty
    )
    return {
        "edge_ready": float(edge_ready),
        "functional_ready": float(functional_ready),
        "bookkeeping_safety": float(bookkeeping_safety),
        "substitutability": float(substitutability),
        "core_penalty": float(core_penalty),
        "shell_penalty": float(shell_penalty),
        "shell_indispensability": float(shell_indispensability),
        "retirement_readiness": float(readiness),
        "eligible": bool(readiness >= cfg.retirement_threshold),
    }


def candidate_births(active_nodes: Set[int], dormant_nodes: Set[int], active_edges: Set[Edge]):
    if not dormant_nodes:
        return []
    child = min(dormant_nodes)
    return [(e, child) for e in sorted(active_edges)]


def candidate_weakens(active_edges: Set[Edge]):
    return list(sorted(active_edges))


def candidate_retirements(
    psi,
    active_nodes: Set[int],
    active_edges: Set[Edge],
    edge_strengths: Dict[Edge, float],
    link_regs: Dict[Edge, np.ndarray],
    core_before,
    cfg: ScoreConfig,
    GM,
    xp,
    n_sites: int,
):
    out = []
    info = {}
    for node in sorted(active_nodes):
        rr = retirement_readiness(node, psi, active_nodes, active_edges, edge_strengths, link_regs, core_before, cfg, GM, xp, n_sites)
        info[node] = rr
        out.append(node)
    return out, info


def _edge_redundancy_score(edge: Edge, active_edges: Set[Edge]):
    return float(_edge_role_structure_terms(edge, active_edges)["redundancy"])


def _node_redundancy_score(node: int, active_edges: Set[Edge], edge_strengths: Dict[Edge, float], core_before):
    incident = [e for e in active_edges if node in e]
    if not incident:
        return 1.0
    nbr = _neighbor_map(active_edges)
    neighbors = sorted({e[1] if e[0] == node else e[0] for e in incident})
    existing = 0
    possible = 0
    for idx, a in enumerate(neighbors):
        for b in neighbors[idx + 1 :]:
            possible += 1
            if canonical_edge(a, b) in active_edges:
                existing += 1
    cluster = existing / possible if possible > 0 else 1.0
    mean_strength = float(np.mean([edge_strengths[e] for e in incident])) if incident else 0.0
    weak_ready = max(0.0, 1.0 - mean_strength)
    cp = tuple(core_before["core_pair"]) if core_before else None
    core_pen = 1.0 if cp is not None and node in cp else 0.0
    return float(max(0.0, 0.55 * cluster + 0.35 * weak_ready + 0.10 * (1.0 if len(neighbors) <= 2 else 0.0) - 0.75 * core_pen))


def _expression_adjustment_for_weaken_or_retire(kind: str, obj, before_state, after_state, core_before, nr, retire_info, cfg: ScoreConfig):
    psi_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = before_state
    psi_a, active_nodes_a, dormant_nodes_a, active_edges_a, local_coeffs_a, edge_strengths_a, link_regs_a = after_state

    if kind == "weaken":
        e = canonical_edge(*obj)
        redundancy = _edge_redundancy_score(e, active_edges_b)
        if nr.get("core_weaken", False):
            return 0.0, {
                "expr_redundancy_removed": 0.0,
                "expr_distinctness_gain": 0.0,
                "expr_adjustment": 0.0,
                "dE_expr_structural_adjustment": 0.0,
            }

        lawful = nr.get("lawful_shell_reexpression", False) or (not nr.get("hits_core_pair", False) and redundancy > 0.35)
        if not lawful:
            return 0.0, {
                "expr_redundancy_removed": float(redundancy),
                "expr_distinctness_gain": 0.0,
                "expr_adjustment": 0.0,
                "dE_expr_structural_adjustment": 0.0,
            }

        func_pres = max(0.0, 1.0 - nr.get("W_func", 0.0))
        distinctness_gain = redundancy * func_pres
        adjustment = 0.0025 * distinctness_gain
        return float(adjustment), {
            "expr_redundancy_removed": float(redundancy),
            "expr_distinctness_gain": float(distinctness_gain),
            "expr_adjustment": float(adjustment),
            "dE_expr_structural_adjustment": float(adjustment),
        }

    if kind == "retire":
        node = int(obj)
        rr = retire_info or {}
        redundancy = _node_redundancy_score(node, active_edges_b, edge_strengths_b, core_before)
        readiness = float(rr.get("retirement_readiness", 0.0))
        func_pres = max(0.0, 1.0 - nr.get("W_func", 0.0))
        readiness_pos = max(0.0, readiness)
        distinctness_gain = redundancy * readiness_pos * func_pres
        adjustment = 0.9 * distinctness_gain

        return float(adjustment), {
            "expr_redundancy_removed": float(redundancy),
            "expr_distinctness_gain": float(distinctness_gain),
            "expr_adjustment": float(adjustment),
            "dE_expr_structural_adjustment": float(adjustment),
        }

    return 0.0, {
        "expr_redundancy_removed": 0.0,
        "expr_distinctness_gain": 0.0,
        "expr_adjustment": 0.0,
        "dE_expr_structural_adjustment": 0.0,
    }


def score_move(kind: str, obj, before_state, after_state, cfg: ScoreConfig, GM, xp):
    psi_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = before_state
    psi_a, active_nodes_a, dormant_nodes_a, active_edges_a, local_coeffs_a, edge_strengths_a, link_regs_a = after_state
    n_sites = int(local_coeffs_b.shape[0])

    core_before = dominant_core_snapshot(psi_b, active_nodes_b, active_edges_b, edge_strengths_b, GM, xp, n_sites, cfg, link_regs_b)

    expr_b = local_expression(psi_b, active_nodes_b, active_edges_b, edge_strengths_b, link_regs_b, cfg, GM, xp, n_sites)
    expr_a = local_expression(psi_a, active_nodes_a, active_edges_a, edge_strengths_a, link_regs_a, cfg, GM, xp, n_sites)
    dE_raw = float(expr_a - expr_b)

    birth_info = None
    retire_info = None
    dE = dE_raw

    if kind == "birth":
        birth_info = birth_justification_witness(obj, before_state, after_state, cfg, GM, xp)
        dE = dE_raw * birth_info["birth_justification"]

    cb_b = bandwidth_burden(active_edges_b, link_regs_b)
    cb_a = bandwidth_burden(active_edges_a, link_regs_a)
    dCB = float(cb_a - cb_b)

    cs_b = spread_burden(active_nodes_b, active_edges_b)
    cs_a = spread_burden(active_nodes_a, active_edges_a)
    dCS = float(cs_a - cs_b)

    nr = no_refolding_witness(
        kind,
        obj,
        psi_b,
        psi_a,
        active_nodes_b,
        active_edges_b,
        edge_strengths_b,
        active_nodes_a,
        active_edges_a,
        edge_strengths_a,
        core_before,
        cfg,
        GM,
        xp,
        n_sites,
    )
    dCF = float(max(0.0, 1.0 - nr["F_org"]) + max(0.0, nr["W_func"]))

    if kind == "retire":
        retire_info = retirement_readiness(
            int(obj),
            psi_b,
            active_nodes_b,
            active_edges_b,
            edge_strengths_b,
            link_regs_b,
            core_before,
            cfg,
            GM,
            xp,
            n_sites,
        )

    expr_adj, expr_adj_info = _expression_adjustment_for_weaken_or_retire(kind, obj, before_state, after_state, core_before, nr, retire_info, cfg)
    dE = float(dE + expr_adj)

    if kind == "weaken" and nr["core_weaken"]:
        dCB_eff = max(0.0, dCB)
        dCS_eff = max(0.0, dCS)
    else:
        dCB_eff = dCB
        dCS_eff = dCS

    deltaF = dE - cfg.lambda_B * dCB_eff - cfg.lambda_S * dCS_eff - cfg.lambda_F * dCF - cfg.lambda_R * nr["W_NR"]

    if isinstance(obj, tuple):
        move_object = list(obj)
    elif isinstance(obj, dict):
        move_object = {k: list(v) if isinstance(v, tuple) else int(v) for k, v in obj.items()}
    else:
        move_object = int(obj)

    diag = {
        "move_type": str(kind),
        "move_object": move_object,
        "deltaF": float(deltaF),
        "dE_expr": float(dE),
        "dE_expr_raw": float(dE_raw),
        "dCB": float(dCB),
        "dCS": float(dCS),
        "dCF": float(dCF),
        "core_before": core_before,
        **nr,
        **expr_adj_info,
    }
    if birth_info is not None:
        diag.update(birth_info)
    if retire_info is not None:
        diag["retirement_info"] = retire_info
    return float(deltaF), diag