#!/usr/bin/env python3
# filename: hsf_mesoscape_retirement_counterfactual_local_role_audit.py

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def _late_start(n: int) -> int:
    return max(0, (2 * n) // 3)


def _sorted_edge(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _winner_from_snapshot(snap: Dict[str, Any]) -> Optional[str]:
    if _safe_int(snap.get("n_raise_support_this_eval")) > 0:
        return "raise_support"
    if _safe_int(snap.get("n_lower_support_this_eval")) > 0:
        return "lower_support"
    if _safe_int(snap.get("n_edge_up_this_eval")) > 0:
        return "edge_up"
    if _safe_int(snap.get("n_edge_down_this_eval")) > 0:
        return "edge_down"
    return None


def _winner_diag(snap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    winner = _winner_from_snapshot(snap)
    if winner is None:
        return None
    cands = snap.get("candidate_move_diagnostics", []) or []
    chosen = None
    for d in cands:
        if str(d.get("move_type")) == winner:
            chosen = d
            break
    if chosen is None and cands:
        chosen = max(cands, key=lambda d: _safe_float(d.get("deltaF")))
    return chosen


def _neighbor_map(edges: Sequence[Sequence[int]]) -> Dict[int, set]:
    nbr: Dict[int, set] = defaultdict(set)
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            a, b = int(e[0]), int(e[1])
            nbr[a].add(b)
            nbr[b].add(a)
    return nbr


def _edge_set(edges: Sequence[Sequence[int]]) -> set[Tuple[int, int]]:
    out = set()
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.add(_sorted_edge(int(e[0]), int(e[1])))
    return out


def _extract_core_pair_from_snap(snap: Dict[str, Any]) -> List[int]:
    dom = snap.get("dominant_core") or {}
    cp = dom.get("core_pair")
    if isinstance(cp, list) and len(cp) == 2:
        return [int(cp[0]), int(cp[1])]
    return []


def _shell_nodes(core_pair: Sequence[int], active_edges: Sequence[Sequence[int]]) -> List[int]:
    if not isinstance(core_pair, (list, tuple)) or len(core_pair) != 2:
        return []
    i, j = int(core_pair[0]), int(core_pair[1])
    out = set()
    for e in active_edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            a, b = int(e[0]), int(e[1])
            if i in (a, b) or j in (a, b):
                out.add(a)
                out.add(b)
    out.discard(i)
    out.discard(j)
    return sorted(out)


def _local_cluster(node: int, nbr: Dict[int, set]) -> float:
    neighbors = sorted(nbr.get(node, set()))
    if len(neighbors) < 2:
        return 0.0
    existing = 0
    possible = 0
    for idx, a in enumerate(neighbors):
        for b in neighbors[idx + 1 :]:
            possible += 1
            if b in nbr.get(a, set()):
                existing += 1
    return float(existing / possible) if possible > 0 else 0.0


def _site_role_weight(node: int, nbr: Dict[int, set], edge_strength_map: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    neighbors = sorted(nbr.get(node, set()))
    incident = [_sorted_edge(node, j) for j in neighbors]
    incident_strengths = [float(edge_strength_map.get(e, 0.0)) for e in incident]

    mean_strength = float(sum(incident_strengths) / len(incident_strengths)) if incident_strengths else 0.0
    activity_sum = float(sum(incident_strengths)) if incident_strengths else 0.0
    incident_count = int(len(incident))

    cluster = _local_cluster(node, nbr)
    sibling_count_norm = float(min(1.0, max(0, incident_count - 1) / 4.0))

    novelty = float(max(0.0, 1.0 - 0.55 * sibling_count_norm - 0.45 * cluster))
    relief = float(min(1.0, 0.5 * mean_strength + 0.3 * cluster + 0.2 * min(1.0, incident_count / 3.0)))
    distinctness = float(max(0.0, 1.0 - cluster))
    weight = float(0.40 * novelty + 0.35 * relief + 0.25 * distinctness)

    return {
        "incident_count": float(incident_count),
        "mean_strength": float(mean_strength),
        "activity_sum": float(activity_sum),
        "local_cluster": float(cluster),
        "sibling_count_norm": float(sibling_count_norm),
        "novelty": float(novelty),
        "relief": float(relief),
        "distinctness": float(distinctness),
        "weight": float(weight),
    }


def _counterfactual_remove_node(
    node: int,
    sigma: Sequence[float],
    top_interfaces: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    sigma2 = list(float(x) for x in sigma)
    if 0 <= node < len(sigma2):
        sigma2[node] = 0.0

    edge_strength_map: Dict[Tuple[int, int], float] = {}
    kept_edges: List[List[int]] = []
    for row in top_interfaces:
        edge = row.get("edge")
        if not (isinstance(edge, list) and len(edge) == 2):
            continue
        a, b = int(edge[0]), int(edge[1])
        if node in (a, b):
            continue
        e = _sorted_edge(a, b)
        c = _safe_float(row.get("commitment"))
        edge_strength_map[e] = c
        if c > 0.05:
            kept_edges.append([e[0], e[1]])

    return {
        "sigma": sigma2,
        "edge_strength_map": edge_strength_map,
        "active_edges": kept_edges,
    }


def _analyze_retirement_candidate(step: int, snap: Dict[str, Any], diag: Dict[str, Any], winner_type: str) -> Dict[str, Any]:
    node = _safe_int(diag.get("node", diag.get("move_object")))
    core_pair = _extract_core_pair_from_snap(snap)
    active_edges = snap.get("active_edges") or []
    sigma = ((snap.get("sigma_summary") or {}).get("sigma")) or []
    top_interfaces = ((snap.get("commitment_summary") or {}).get("top_interfaces")) or []

    nbr_before = _neighbor_map(active_edges)
    edge_strength_before: Dict[Tuple[int, int], float] = {}
    for row in top_interfaces:
        edge = row.get("edge")
        if isinstance(edge, list) and len(edge) == 2:
            edge_strength_before[_sorted_edge(int(edge[0]), int(edge[1]))] = _safe_float(row.get("commitment"))

    neighbors_before = sorted(nbr_before.get(node, set()))
    core_neighbors_before = [n for n in neighbors_before if n in set(core_pair)]
    noncore_neighbors_before = [n for n in neighbors_before if n not in set(core_pair)]

    before_neighbor_weights = {str(n): _site_role_weight(n, nbr_before, edge_strength_before) for n in neighbors_before}

    cf = _counterfactual_remove_node(node, sigma, top_interfaces)
    nbr_after = _neighbor_map(cf["active_edges"])
    edge_strength_after = cf["edge_strength_map"]

    after_neighbor_weights = {str(n): _site_role_weight(n, nbr_after, edge_strength_after) for n in neighbors_before}

    neighbor_weight_before_vals = [before_neighbor_weights[str(n)]["weight"] for n in neighbors_before]
    neighbor_weight_after_vals = [after_neighbor_weights[str(n)]["weight"] for n in neighbors_before]

    neighbor_cluster_before_vals = [before_neighbor_weights[str(n)]["local_cluster"] for n in neighbors_before]
    neighbor_cluster_after_vals = [after_neighbor_weights[str(n)]["local_cluster"] for n in neighbors_before]

    incident_before_vals = [before_neighbor_weights[str(n)]["incident_count"] for n in neighbors_before]
    incident_after_vals = [after_neighbor_weights[str(n)]["incident_count"] for n in neighbors_before]

    neighbor_weight_delta = float(sum(neighbor_weight_after_vals) - sum(neighbor_weight_before_vals))
    mean_neighbor_weight_delta = float(_mean(neighbor_weight_after_vals) - _mean(neighbor_weight_before_vals))
    neighbor_cluster_delta = float(_mean(neighbor_cluster_after_vals) - _mean(neighbor_cluster_before_vals))
    neighbor_incident_delta = float(_mean(incident_after_vals) - _mean(incident_before_vals))

    edge_set_after = _edge_set(cf["active_edges"])
    pair_links_after = 0
    pair_possible = 0
    for idx, a in enumerate(neighbors_before):
        for b in neighbors_before[idx + 1 :]:
            pair_possible += 1
            if _sorted_edge(a, b) in edge_set_after:
                pair_links_after += 1
    neighbor_reroutability_after = float(pair_links_after / pair_possible) if pair_possible > 0 else 1.0

    case = "retire_winner" if winner_type == "lower_support" else "retire_loser"

    return {
        "step": int(step),
        "node": int(node),
        "winner_move_type": str(winner_type),
        "case": case,
        "core_pair": core_pair,
        "touches_core": bool(node in set(core_pair)),
        "shell_nodes": _shell_nodes(core_pair, active_edges),
        "neighbors_before": neighbors_before,
        "core_neighbors_before": core_neighbors_before,
        "noncore_neighbors_before": noncore_neighbors_before,
        "neighbor_count_before": len(neighbors_before),
        "neighbor_weight_sum_before": float(sum(neighbor_weight_before_vals)),
        "neighbor_weight_sum_after": float(sum(neighbor_weight_after_vals)),
        "neighbor_weight_delta": float(neighbor_weight_delta),
        "mean_neighbor_weight_before": float(_mean(neighbor_weight_before_vals)),
        "mean_neighbor_weight_after": float(_mean(neighbor_weight_after_vals)),
        "mean_neighbor_weight_delta": float(mean_neighbor_weight_delta),
        "mean_neighbor_cluster_before": float(_mean(neighbor_cluster_before_vals)),
        "mean_neighbor_cluster_after": float(_mean(neighbor_cluster_after_vals)),
        "mean_neighbor_cluster_delta": float(neighbor_cluster_delta),
        "mean_neighbor_incident_before": float(_mean(incident_before_vals)),
        "mean_neighbor_incident_after": float(_mean(incident_after_vals)),
        "mean_neighbor_incident_delta": float(neighbor_incident_delta),
        "neighbor_reroutability_after": float(neighbor_reroutability_after),
        "lower_deltaF": _safe_float(diag.get("deltaF")),
        "dE_expr_raw": _safe_float(diag.get("dE_expr_raw")),
        "dE_expr": _safe_float(diag.get("dE_expr")),
        "dE_expr_structural_adjustment": _safe_float(
            diag.get("dE_expr_structural_adjustment", diag.get("expr_adjustment"))
        ),
        "delta_Odiff_R": _safe_float(diag.get("delta_Odiff_R")),
        "expr_redundancy_removed": _safe_float(diag.get("expr_redundancy_removed")),
        "expr_distinctness_gain": _safe_float(diag.get("expr_distinctness_gain")),
        "W_NR": _safe_float(diag.get("W_NR")),
        "dCF": _safe_float(diag.get("dCF")),
        "retirement_readiness": _safe_float((diag.get("retirement_info") or {}).get("retirement_readiness")),
        "shell_penalty": _safe_float((diag.get("retirement_info") or {}).get("shell_penalty")),
        "shell_indispensability": _safe_float((diag.get("retirement_info") or {}).get("shell_indispensability")),
        "before_neighbor_weights": before_neighbor_weights,
        "after_neighbor_weights": after_neighbor_weights,
    }


def _rows_from_run(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snap in data.get("snapshots", []) or []:
        step = _safe_int(snap.get("step"))
        winner = _winner_diag(snap)
        if winner is None:
            continue
        winner_type = str(winner.get("move_type"))
        cands = snap.get("candidate_move_diagnostics", []) or []
        lowers = [d for d in cands if str(d.get("move_type")) == "lower_support"]
        if not lowers:
            continue

        lowers_sorted = sorted(lowers, key=lambda d: _safe_float(d.get("deltaF")), reverse=True)
        best_lower = lowers_sorted[0]
        rows.append(_analyze_retirement_candidate(step, snap, best_lower, winner_type))
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "case_counts": dict(Counter(r["case"] for r in rows)),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in rows)),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in rows]),
        "dE_expr_raw_mean": _mean([r["dE_expr_raw"] for r in rows]),
        "dE_expr_mean": _mean([r["dE_expr"] for r in rows]),
        "neighbor_weight_delta_mean": _mean([r["neighbor_weight_delta"] for r in rows]),
        "mean_neighbor_weight_delta_mean": _mean([r["mean_neighbor_weight_delta"] for r in rows]),
        "neighbor_cluster_delta_mean": _mean([r["mean_neighbor_cluster_delta"] for r in rows]),
        "neighbor_incident_delta_mean": _mean([r["mean_neighbor_incident_delta"] for r in rows]),
        "neighbor_reroutability_after_mean": _mean([r["neighbor_reroutability_after"] for r in rows]),
        "shell_penalty_mean": _mean([r["shell_penalty"] for r in rows]),
    }


def _per_node_summary(rows: List[Dict[str, Any]], node_id: int) -> Dict[str, Any]:
    group = [r for r in rows if r["node"] == node_id]
    return {
        "n": len(group),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in group)),
        "case_counts": dict(Counter(r["case"] for r in group)),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in group]),
        "dE_expr_raw_mean": _mean([r["dE_expr_raw"] for r in group]),
        "dE_expr_mean": _mean([r["dE_expr"] for r in group]),
        "structural_adjustment_mean": _mean([r["dE_expr_structural_adjustment"] for r in group]),
        "neighbor_weight_delta_mean": _mean([r["neighbor_weight_delta"] for r in group]),
        "mean_neighbor_weight_delta_mean": _mean([r["mean_neighbor_weight_delta"] for r in group]),
        "neighbor_cluster_delta_mean": _mean([r["mean_neighbor_cluster_delta"] for r in group]),
        "neighbor_incident_delta_mean": _mean([r["mean_neighbor_incident_delta"] for r in group]),
        "neighbor_reroutability_after_mean": _mean([r["neighbor_reroutability_after"] for r in group]),
        "shell_penalty_mean": _mean([r["shell_penalty"] for r in group]),
        "W_NR_mean": _mean([r["W_NR"] for r in group]),
        "dCF_mean": _mean([r["dCF"] for r in group]),
        "redundancy_removed_mean": _mean([r["expr_redundancy_removed"] for r in group]),
        "distinctness_gain_mean": _mean([r["expr_distinctness_gain"] for r in group]),
    }


def _top_late_nodes(rows: List[Dict[str, Any]], k: int = 5) -> List[int]:
    ctr = Counter(r["node"] for r in rows)
    return [node for node, _ in ctr.most_common(k)]


def _readout(all_rows: List[Dict[str, Any]], late_rows: List[Dict[str, Any]], node_summaries: Dict[str, Any]) -> List[str]:
    reads: List[str] = []
    late_cases = Counter(r["case"] for r in late_rows)

    if late_cases.get("retire_winner", 0) > 0:
        reads.append("Some late retirements preserve enough local role structure to remain lawful winners.")
    if late_cases.get("retire_loser", 0) > 0:
        reads.append("Some late retirements still damage local role structure enough to lose.")

    if "2" in node_summaries and "6" in node_summaries:
        n2 = node_summaries["2"]
        n6 = node_summaries["6"]
        if n2["neighbor_weight_delta_mean"] > n6["neighbor_weight_delta_mean"]:
            reads.append("Retiring node 2 preserves neighbor local-role weight better than retiring node 6.")
        if n6["neighbor_cluster_delta_mean"] < n2["neighbor_cluster_delta_mean"]:
            reads.append("Retiring node 6 appears to reduce neighbor-side local coherence more sharply than retiring node 2.")
        if n6["dE_expr_raw_mean"] < n2["dE_expr_raw_mean"]:
            reads.append("Node 6 remains more raw-expression-destructive than node 2 even before shell terms are considered.")

    if not reads:
        reads.append("No single counterfactual local-role pattern dominates the retirement outcomes.")
    return reads


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 30) -> List[Dict[str, Any]]:
    flagged = sorted(
        rows,
        key=lambda r: (
            r["lower_deltaF"],
            r["neighbor_weight_delta"],
            -r["step"],
        ),
        reverse=True,
    )
    return flagged[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retirement counterfactual local-role audit for node-level demotion. "
            "Compares best lower_support candidates by inspecting neighbor local-role changes "
            "under counterfactual node removal."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_retirement_counterfactual_local_role_audit.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = _load_json(in_path)
    rows = _rows_from_run(data)
    late = rows[_late_start(len(rows)):] if rows else []

    top_nodes = _top_late_nodes(late, k=5)
    node_summaries = {str(node): _per_node_summary(late, node) for node in top_nodes}

    report = {
        "script": "hsf_mesoscape_retirement_counterfactual_local_role_audit.py",
        "input_json": str(in_path),
        "overall_summary": _summary(rows),
        "late_summary": _summary(late),
        "late_top_nodes": top_nodes,
        "late_node_summaries": node_summaries,
        "flagged_rows": _flagged_rows(rows),
        "readout": _readout(rows, late, node_summaries),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_retirement_counterfactual_local_role_audit.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Retirement Counterfactual Local-Role Audit ===")
    print(f"rows:                           {report['overall_summary']['n_rows']}")
    print(f"late rows:                      {report['late_summary']['n_rows']}")
    print(f"overall cases:                  {report['overall_summary']['case_counts']}")
    print(f"late cases:                     {report['late_summary']['case_counts']}")
    print(f"late top nodes:                 {report['late_top_nodes']}")
    print(f"late neighbor weight delta:     {report['late_summary']['neighbor_weight_delta_mean']:.6f}")
    print(f"late neighbor cluster delta:    {report['late_summary']['neighbor_cluster_delta_mean']:.6f}")
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()