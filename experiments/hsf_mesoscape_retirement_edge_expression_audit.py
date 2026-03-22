#!/usr/bin/env python3
# filename: hsf_mesoscape_retirement_edge_expression_audit.py

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


def _extract_core_pair_from_snap(snap: Dict[str, Any]) -> List[int]:
    dom = snap.get("dominant_core") or {}
    cp = dom.get("core_pair")
    if isinstance(cp, list) and len(cp) == 2:
        return [int(cp[0]), int(cp[1])]
    return []


def _edge_strength_map_from_snapshot(snap: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    top_interfaces = ((snap.get("commitment_summary") or {}).get("top_interfaces")) or []
    for row in top_interfaces:
        edge = row.get("edge")
        if isinstance(edge, list) and len(edge) == 2:
            out[_sorted_edge(int(edge[0]), int(edge[1]))] = _safe_float(row.get("commitment"))
    return out


def _edge_role_terms(edge: Tuple[int, int], active_edge_set: set[Tuple[int, int]]) -> Dict[str, float]:
    i, j = edge
    nbr = _neighbor_map([list(e) for e in active_edge_set])
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
        "tri_redundancy": tri_redundancy,
        "route_redundancy": route_redundancy,
        "redundancy": redundancy,
        "novelty": novelty,
        "distinctness": distinctness,
        "overlap_relief": overlap_relief,
        "shared_neighbor_count": float(len(shared)),
    }


def _edge_component_score(edge: Tuple[int, int], strength_map: Dict[Tuple[int, int], float], active_edge_set: set[Tuple[int, int]]) -> Dict[str, float]:
    """
    Reconstructs the bookkeeping-side raw edge expression surrogate from topology + commitment.
    This will not reproduce hidden physics terms, but it will expose which edge classes dominate
    the bookkeeping raw-expression change under retirement.
    """
    strength = float(strength_map.get(edge, 0.0))
    role = _edge_role_terms(edge, active_edge_set)

    # Topology-visible proxy for raw expression component.
    base_signal = 0.15 * strength
    role_credit = base_signal * (0.40 + 0.35 * role["novelty"] + 0.25 * role["overlap_relief"])
    distinct_bonus = (0.10 * base_signal + 0.05 * strength) * role["distinctness"]
    redundancy_penalty = 0.70 * (role["redundancy"] ** 2) * max(0.0, 0.40 * strength)

    expr_proxy = role_credit + distinct_bonus - redundancy_penalty

    return {
        "strength": strength,
        "base_signal": base_signal,
        "role_credit": role_credit,
        "distinct_bonus": distinct_bonus,
        "redundancy_penalty": redundancy_penalty,
        "expr_proxy": expr_proxy,
        **role,
    }


def _counterfactual_after_retirement(node: int, snap: Dict[str, Any]) -> Tuple[set[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    active_edges = snap.get("active_edges") or []
    edge_strength_map = _edge_strength_map_from_snapshot(snap)

    kept_edges: set[Tuple[int, int]] = set()
    kept_strengths: Dict[Tuple[int, int], float] = {}
    for e in active_edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            a, b = int(e[0]), int(e[1])
            if node in (a, b):
                continue
            se = _sorted_edge(a, b)
            kept_edges.add(se)
            kept_strengths[se] = edge_strength_map.get(se, 0.0)
    return kept_edges, kept_strengths


def _incident_and_neighbor_edges(node: int, active_edge_set: set[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    nbr = _neighbor_map([list(e) for e in active_edge_set])
    neighbors = sorted(nbr.get(node, set()))
    incident = sorted([e for e in active_edge_set if node in e])

    neighbor_edges: List[Tuple[int, int]] = []
    for idx, a in enumerate(neighbors):
        for b in neighbors[idx + 1:]:
            e = _sorted_edge(a, b)
            if e in active_edge_set:
                neighbor_edges.append(e)
    return incident, sorted(neighbor_edges)


def _analyze_step(step: int, snap: Dict[str, Any], lower_diag: Dict[str, Any], winner_type: str) -> Dict[str, Any]:
    node = _safe_int(lower_diag.get("node", lower_diag.get("move_object")))
    active_edges = snap.get("active_edges") or []
    before_edge_set = set(_sorted_edge(int(e[0]), int(e[1])) for e in active_edges if isinstance(e, (list, tuple)) and len(e) == 2)
    before_strengths = _edge_strength_map_from_snapshot(snap)

    after_edge_set, after_strengths = _counterfactual_after_retirement(node, snap)

    incident_before, neighbor_edges_before = _incident_and_neighbor_edges(node, before_edge_set)

    # Union of edges relevant to local retirement effect:
    relevant_edges = sorted(set(incident_before) | set(neighbor_edges_before) | set(after_edge_set))

    before_rows: List[Dict[str, Any]] = []
    after_rows: List[Dict[str, Any]] = []

    for e in relevant_edges:
        if e in before_edge_set:
            comp_b = _edge_component_score(e, before_strengths, before_edge_set)
            before_rows.append({"edge": [e[0], e[1]], **comp_b, "status": "before"})
        if e in after_edge_set:
            comp_a = _edge_component_score(e, after_strengths, after_edge_set)
            after_rows.append({"edge": [e[0], e[1]], **comp_a, "status": "after"})

    before_by_edge = {tuple(r["edge"]): r for r in before_rows}
    after_by_edge = {tuple(r["edge"]): r for r in after_rows}

    deltas: List[Dict[str, Any]] = []
    for e in sorted(set(before_by_edge) | set(after_by_edge)):
        b = before_by_edge.get(e)
        a = after_by_edge.get(e)
        deltas.append({
            "edge": [e[0], e[1]],
            "expr_proxy_before": _safe_float(b.get("expr_proxy")) if b else 0.0,
            "expr_proxy_after": _safe_float(a.get("expr_proxy")) if a else 0.0,
            "expr_proxy_delta": (_safe_float(a.get("expr_proxy")) if a else 0.0) - (_safe_float(b.get("expr_proxy")) if b else 0.0),
            "strength_before": _safe_float(b.get("strength")) if b else 0.0,
            "strength_after": _safe_float(a.get("strength")) if a else 0.0,
            "redundancy_before": _safe_float(b.get("redundancy")) if b else 0.0,
            "redundancy_after": _safe_float(a.get("redundancy")) if a else 0.0,
            "novelty_before": _safe_float(b.get("novelty")) if b else 0.0,
            "novelty_after": _safe_float(a.get("novelty")) if a else 0.0,
            "distinctness_before": _safe_float(b.get("distinctness")) if b else 0.0,
            "distinctness_after": _safe_float(a.get("distinctness")) if a else 0.0,
            "edge_class": (
                "incident_lost" if tuple(e) in incident_before and tuple(e) not in after_edge_set else
                "neighbor_survives" if tuple(e) in neighbor_edges_before and tuple(e) in after_edge_set else
                "neighbor_lost" if tuple(e) in neighbor_edges_before and tuple(e) not in after_edge_set else
                "other_survives" if tuple(e) in after_edge_set else
                "other_lost"
            ),
        })

    deltas_sorted = sorted(deltas, key=lambda r: r["expr_proxy_delta"])

    return {
        "step": int(step),
        "node": int(node),
        "winner_move_type": str(winner_type),
        "lower_deltaF": _safe_float(lower_diag.get("deltaF")),
        "dE_expr_raw": _safe_float(lower_diag.get("dE_expr_raw")),
        "dE_expr": _safe_float(lower_diag.get("dE_expr")),
        "dE_expr_structural_adjustment": _safe_float(lower_diag.get("dE_expr_structural_adjustment", lower_diag.get("expr_adjustment"))),
        "delta_Odiff_R": _safe_float(lower_diag.get("delta_Odiff_R")),
        "expr_redundancy_removed": _safe_float(lower_diag.get("expr_redundancy_removed")),
        "expr_distinctness_gain": _safe_float(lower_diag.get("expr_distinctness_gain")),
        "W_NR": _safe_float(lower_diag.get("W_NR")),
        "dCF": _safe_float(lower_diag.get("dCF")),
        "core_pair": _extract_core_pair_from_snap(snap),
        "incident_edges_before": [[a, b] for (a, b) in incident_before],
        "neighbor_edges_before": [[a, b] for (a, b) in neighbor_edges_before],
        "edge_rows_before": before_rows,
        "edge_rows_after": after_rows,
        "edge_deltas": deltas_sorted,
        "sum_expr_proxy_before": float(sum(r["expr_proxy"] for r in before_rows)),
        "sum_expr_proxy_after": float(sum(r["expr_proxy"] for r in after_rows)),
        "sum_expr_proxy_delta": float(sum(r["expr_proxy_delta"] for r in deltas)),
        "largest_negative_edge_deltas": deltas_sorted[:10],
        "largest_positive_edge_deltas": list(reversed(sorted(deltas, key=lambda r: r["expr_proxy_delta"])))[:10],
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
        best_lower = max(lowers, key=lambda d: _safe_float(d.get("deltaF")))
        rows.append(_analyze_step(step, snap, best_lower, winner_type))
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in rows)),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in rows]),
        "dE_expr_raw_mean": _mean([r["dE_expr_raw"] for r in rows]),
        "dE_expr_mean": _mean([r["dE_expr"] for r in rows]),
        "sum_expr_proxy_delta_mean": _mean([r["sum_expr_proxy_delta"] for r in rows]),
    }


def _per_node_summary(rows: List[Dict[str, Any]], node_id: int) -> Dict[str, Any]:
    group = [r for r in rows if r["node"] == node_id]
    edge_class_means: Dict[str, float] = {}
    by_class: Dict[str, List[float]] = defaultdict(list)
    for r in group:
        for ed in r["edge_deltas"]:
            by_class[str(ed["edge_class"])].append(_safe_float(ed["expr_proxy_delta"]))
    for k, vals in by_class.items():
        edge_class_means[k] = _mean(vals)

    return {
        "n": len(group),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in group)),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in group]),
        "dE_expr_raw_mean": _mean([r["dE_expr_raw"] for r in group]),
        "dE_expr_mean": _mean([r["dE_expr"] for r in group]),
        "structural_adjustment_mean": _mean([r["dE_expr_structural_adjustment"] for r in group]),
        "sum_expr_proxy_delta_mean": _mean([r["sum_expr_proxy_delta"] for r in group]),
        "edge_class_delta_means": edge_class_means,
    }


def _top_late_nodes(rows: List[Dict[str, Any]], k: int = 5) -> List[int]:
    ctr = Counter(r["node"] for r in rows)
    return [node for node, _ in ctr.most_common(k)]


def _readout(late_rows: List[Dict[str, Any]], node_summaries: Dict[str, Any]) -> List[str]:
    reads: List[str] = []
    if "2" in node_summaries and "6" in node_summaries:
        n2 = node_summaries["2"]
        n6 = node_summaries["6"]
        if n6["dE_expr_raw_mean"] < n2["dE_expr_raw_mean"]:
            reads.append("Node 6 remains more raw-expression-destructive than node 2 at retirement.")
        ic2 = n2["edge_class_delta_means"].get("incident_lost", 0.0)
        ic6 = n6["edge_class_delta_means"].get("incident_lost", 0.0)
        if ic6 < ic2:
            reads.append("The largest node-6 disadvantage appears in the lost incident-edge contribution.")
        ns2 = n2["edge_class_delta_means"].get("neighbor_survives", 0.0)
        ns6 = n6["edge_class_delta_means"].get("neighbor_survives", 0.0)
        if ns6 < ns2:
            reads.append("Node 6 gets less compensating gain from surviving neighbor-neighbor edges than node 2.")
    if not reads:
        reads.append("No single edge class dominates the raw-expression difference in late retirements.")
    return reads


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["lower_deltaF"], r["dE_expr_raw"]), reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Edge-resolved raw-expression audit for retirement candidates. "
            "Breaks retirement raw-expression change down by lost incident edges and surviving neighbor edges."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_retirement_edge_expression_audit.json",
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
        "script": "hsf_mesoscape_retirement_edge_expression_audit.py",
        "input_json": str(in_path),
        "overall_summary": _summary(rows),
        "late_summary": _summary(late),
        "late_top_nodes": top_nodes,
        "late_node_summaries": node_summaries,
        "flagged_rows": _flagged_rows(rows),
        "readout": _readout(late, node_summaries),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_retirement_edge_expression_audit.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Retirement Edge Expression Audit ===")
    print(f"rows:                    {report['overall_summary']['n_rows']}")
    print(f"late rows:               {report['late_summary']['n_rows']}")
    print(f"late top nodes:          {report['late_top_nodes']}")
    print(f"late raw mean:           {report['late_summary']['dE_expr_raw_mean']:.6f}")
    print(f"late expr-proxy delta:   {report['late_summary']['sum_expr_proxy_delta_mean']:.6f}")
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()