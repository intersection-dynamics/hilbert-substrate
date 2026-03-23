#!/usr/bin/env python3
# filename: hsf_mesoscape_retirement_incident_edge_weighting_sandbox.py

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


def _edge_component_score(
    edge: Tuple[int, int],
    strength_map: Dict[Tuple[int, int], float],
    active_edge_set: set[Tuple[int, int]],
) -> Dict[str, float]:
    strength = float(strength_map.get(edge, 0.0))
    role = _edge_role_terms(edge, active_edge_set)

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
        for b in neighbors[idx + 1 :]:
            e = _sorted_edge(a, b)
            if e in active_edge_set:
                neighbor_edges.append(e)
    return incident, sorted(neighbor_edges)


def _classify_incident_edge(
    edge: Tuple[int, int],
    retired_node: int,
    core_pair: Sequence[int],
    snap: Dict[str, Any],
) -> str:
    cp = set(int(x) for x in core_pair) if len(core_pair) == 2 else set()
    a, b = edge
    other = b if a == retired_node else a

    active_edges = snap.get("active_edges") or []
    nbr = _neighbor_map(active_edges)
    neighbors = sorted(nbr.get(retired_node, set()))
    noncore_neighbors = [n for n in neighbors if n not in cp]
    core_neighbors = [n for n in neighbors if n in cp]

    # Unique core-adjacent support
    if other in cp:
        if len(core_neighbors) == 1:
            return "core_adjacent_unique"
        return "core_adjacent_shared"

    # Peripheral shell-side edges.
    # If the other node has alternative ties to core or to peer shell structure, discount more.
    other_neighbors = set(nbr.get(other, set()))
    alt_core = len(other_neighbors & cp)
    alt_shell = len((other_neighbors - cp) - {retired_node})

    if alt_core == 0 and alt_shell == 0:
        return "peripheral_fragile"
    if alt_core > 0:
        return "peripheral_core_redundant"
    return "peripheral_shell_redundant"


def _incident_discount(
    edge_class: str,
    args: argparse.Namespace,
) -> float:
    if edge_class == "core_adjacent_unique":
        return float(args.discount_core_adjacent_unique)
    if edge_class == "core_adjacent_shared":
        return float(args.discount_core_adjacent_shared)
    if edge_class == "peripheral_fragile":
        return float(args.discount_peripheral_fragile)
    if edge_class == "peripheral_core_redundant":
        return float(args.discount_peripheral_core_redundant)
    if edge_class == "peripheral_shell_redundant":
        return float(args.discount_peripheral_shell_redundant)
    return 1.0


def _build_edge_delta_rows(node: int, snap: Dict[str, Any], core_pair: Sequence[int]) -> Dict[str, Any]:
    active_edges = snap.get("active_edges") or []
    before_edge_set = set(
        _sorted_edge(int(e[0]), int(e[1]))
        for e in active_edges
        if isinstance(e, (list, tuple)) and len(e) == 2
    )
    before_strengths = _edge_strength_map_from_snapshot(snap)
    after_edge_set, after_strengths = _counterfactual_after_retirement(node, snap)

    incident_before, neighbor_edges_before = _incident_and_neighbor_edges(node, before_edge_set)
    relevant_edges = sorted(set(incident_before) | set(neighbor_edges_before) | set(after_edge_set))

    before_by_edge: Dict[Tuple[int, int], Dict[str, Any]] = {}
    after_by_edge: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for e in relevant_edges:
        if e in before_edge_set:
            before_by_edge[e] = _edge_component_score(e, before_strengths, before_edge_set)
        if e in after_edge_set:
            after_by_edge[e] = _edge_component_score(e, after_strengths, after_edge_set)

    deltas: List[Dict[str, Any]] = []
    for e in sorted(set(before_by_edge) | set(after_by_edge)):
        b = before_by_edge.get(e)
        a = after_by_edge.get(e)
        delta = (_safe_float(a.get("expr_proxy")) if a else 0.0) - (_safe_float(b.get("expr_proxy")) if b else 0.0)
        edge_class = (
            "incident_lost" if e in incident_before and e not in after_edge_set else
            "neighbor_survives" if e in neighbor_edges_before and e in after_edge_set else
            "neighbor_lost" if e in neighbor_edges_before and e not in after_edge_set else
            "other_survives" if e in after_edge_set else
            "other_lost"
        )

        incident_subclass = None
        if edge_class == "incident_lost":
            incident_subclass = _classify_incident_edge(e, node, core_pair, snap)

        deltas.append({
            "edge": [e[0], e[1]],
            "edge_class": edge_class,
            "incident_subclass": incident_subclass,
            "expr_proxy_before": _safe_float(b.get("expr_proxy")) if b else 0.0,
            "expr_proxy_after": _safe_float(a.get("expr_proxy")) if a else 0.0,
            "expr_proxy_delta": float(delta),
            "strength_before": _safe_float(b.get("strength")) if b else 0.0,
            "strength_after": _safe_float(a.get("strength")) if a else 0.0,
            "redundancy_before": _safe_float(b.get("redundancy")) if b else 0.0,
            "redundancy_after": _safe_float(a.get("redundancy")) if a else 0.0,
            "novelty_before": _safe_float(b.get("novelty")) if b else 0.0,
            "novelty_after": _safe_float(a.get("novelty")) if a else 0.0,
            "distinctness_before": _safe_float(b.get("distinctness")) if b else 0.0,
            "distinctness_after": _safe_float(a.get("distinctness")) if a else 0.0,
        })

    deltas_sorted = sorted(deltas, key=lambda r: r["expr_proxy_delta"])
    return {
        "incident_edges_before": [[a, b] for (a, b) in incident_before],
        "neighbor_edges_before": [[a, b] for (a, b) in neighbor_edges_before],
        "edge_deltas": deltas_sorted,
        "sum_expr_proxy_before": float(sum(_safe_float(before_by_edge[e].get("expr_proxy")) for e in before_by_edge)),
        "sum_expr_proxy_after": float(sum(_safe_float(after_by_edge[e].get("expr_proxy")) for e in after_by_edge)),
        "sum_expr_proxy_delta": float(sum(r["expr_proxy_delta"] for r in deltas_sorted)),
    }


def _organizer_relief_credit(
    core_pair: Sequence[int],
    edge_deltas: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    core_credit = 0.0
    neighbor_relief = 0.0
    incident_loss_raw = 0.0
    incident_loss_weighted = 0.0

    core_edge = None
    if isinstance(core_pair, (list, tuple)) and len(core_pair) == 2:
        core_edge = list(_sorted_edge(int(core_pair[0]), int(core_pair[1])))

    subclass_loss_buckets: Dict[str, float] = defaultdict(float)

    for row in edge_deltas:
        edge = row.get("edge")
        delta = _safe_float(row.get("expr_proxy_delta"))
        edge_class = str(row.get("edge_class"))
        subclass = row.get("incident_subclass")

        if core_edge is not None and edge == core_edge and delta > 0.0:
            core_credit += delta

        if edge_class == "neighbor_survives" and delta > 0.0:
            neighbor_relief += delta

        if edge_class == "incident_lost" and delta < 0.0:
            burden = abs(delta)
            incident_loss_raw += burden
            discount = _incident_discount(str(subclass), args)
            weighted = burden * discount
            incident_loss_weighted += weighted
            subclass_loss_buckets[str(subclass)] += weighted

    gross = float(args.organizer_relief_weight) * core_credit + float(args.neighbor_relief_weight) * neighbor_relief
    net = gross - float(args.incident_loss_weight) * incident_loss_weighted
    credit = max(0.0, min(float(args.credit_cap), net))

    return {
        "organizer_core_relief_credit": float(core_credit),
        "neighbor_relief_credit": float(neighbor_relief),
        "incident_loss_burden_raw": float(incident_loss_raw),
        "incident_loss_burden_weighted": float(incident_loss_weighted),
        "incident_loss_weighted_by_subclass": dict(subclass_loss_buckets),
        "organizer_relief_credit_gross": float(gross),
        "organizer_relief_credit_net": float(net),
        "organizer_relief_credit_applied": float(credit),
    }


def _rows_from_run(data: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for snap in data.get("snapshots", []) or []:
        step = _safe_int(snap.get("step"))
        winner = _winner_diag(snap)
        if winner is None:
            continue

        cands = snap.get("candidate_move_diagnostics", []) or []
        lowers = [d for d in cands if str(d.get("move_type")) == "lower_support"]
        if not lowers:
            continue

        best_lower = max(lowers, key=lambda d: _safe_float(d.get("deltaF")))
        node = _safe_int(best_lower.get("node", best_lower.get("move_object")))
        winner_type = str(winner.get("move_type"))
        core_pair = _extract_core_pair_from_snap(snap)

        edge_info = _build_edge_delta_rows(node, snap, core_pair)
        credit_info = _organizer_relief_credit(core_pair, edge_info["edge_deltas"], args)

        lower_deltaF = _safe_float(best_lower.get("deltaF"))
        dE_expr = _safe_float(best_lower.get("dE_expr"))
        sandbox_dE_expr = dE_expr + credit_info["organizer_relief_credit_applied"]
        sandbox_deltaF = lower_deltaF + credit_info["organizer_relief_credit_applied"]

        row = {
            "step": int(step),
            "node": int(node),
            "winner_move_type": winner_type,
            "core_pair": core_pair,
            "lower_deltaF": float(lower_deltaF),
            "sandbox_lower_deltaF": float(sandbox_deltaF),
            "deltaF_improvement": float(sandbox_deltaF - lower_deltaF),
            "dE_expr_raw": _safe_float(best_lower.get("dE_expr_raw")),
            "dE_expr": float(dE_expr),
            "sandbox_dE_expr": float(sandbox_dE_expr),
            "dE_expr_structural_adjustment": _safe_float(
                best_lower.get("dE_expr_structural_adjustment", best_lower.get("expr_adjustment"))
            ),
            "delta_Odiff_R": _safe_float(best_lower.get("delta_Odiff_R")),
            "expr_redundancy_removed": _safe_float(best_lower.get("expr_redundancy_removed")),
            "expr_distinctness_gain": _safe_float(best_lower.get("expr_distinctness_gain")),
            "W_NR": _safe_float(best_lower.get("W_NR")),
            "dCF": _safe_float(best_lower.get("dCF")),
            "shell_penalty": _safe_float((best_lower.get("retirement_info") or {}).get("shell_penalty")),
            **edge_info,
            **credit_info,
        }
        row["would_flip_positive"] = bool(lower_deltaF <= 0.0 < sandbox_deltaF)
        row["would_beat_winner"] = bool(sandbox_deltaF > _safe_float(winner.get("deltaF")))
        rows.append(row)

    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in rows)),
        "positive_deltaF_fraction": _mean([1.0 if r["lower_deltaF"] > 0.0 else 0.0 for r in rows]),
        "sandbox_positive_deltaF_fraction": _mean([1.0 if r["sandbox_lower_deltaF"] > 0.0 else 0.0 for r in rows]),
        "would_flip_positive_count": int(sum(1 for r in rows if r["would_flip_positive"])),
        "would_beat_winner_count": int(sum(1 for r in rows if r["would_beat_winner"])),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in rows]),
        "sandbox_lower_deltaF_mean": _mean([r["sandbox_lower_deltaF"] for r in rows]),
        "credit_applied_mean": _mean([r["organizer_relief_credit_applied"] for r in rows]),
        "core_relief_mean": _mean([r["organizer_core_relief_credit"] for r in rows]),
        "neighbor_relief_mean": _mean([r["neighbor_relief_credit"] for r in rows]),
        "incident_loss_raw_mean": _mean([r["incident_loss_burden_raw"] for r in rows]),
        "incident_loss_weighted_mean": _mean([r["incident_loss_burden_weighted"] for r in rows]),
    }


def _per_node_summary(rows: List[Dict[str, Any]], node_id: int) -> Dict[str, Any]:
    group = [r for r in rows if r["node"] == node_id]
    subclass_means: Dict[str, float] = defaultdict(float)
    subclass_vals: Dict[str, List[float]] = defaultdict(list)
    for r in group:
        for k, v in (r.get("incident_loss_weighted_by_subclass") or {}).items():
            subclass_vals[str(k)].append(_safe_float(v))
    for k, vals in subclass_vals.items():
        subclass_means[k] = _mean(vals)

    return {
        "n": len(group),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in group)),
        "positive_deltaF_fraction": _mean([1.0 if r["lower_deltaF"] > 0.0 else 0.0 for r in group]),
        "sandbox_positive_deltaF_fraction": _mean([1.0 if r["sandbox_lower_deltaF"] > 0.0 else 0.0 for r in group]),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in group]),
        "sandbox_lower_deltaF_mean": _mean([r["sandbox_lower_deltaF"] for r in group]),
        "credit_applied_mean": _mean([r["organizer_relief_credit_applied"] for r in group]),
        "core_relief_mean": _mean([r["organizer_core_relief_credit"] for r in group]),
        "neighbor_relief_mean": _mean([r["neighbor_relief_credit"] for r in group]),
        "incident_loss_raw_mean": _mean([r["incident_loss_burden_raw"] for r in group]),
        "incident_loss_weighted_mean": _mean([r["incident_loss_burden_weighted"] for r in group]),
        "incident_loss_weighted_by_subclass_mean": dict(subclass_means),
    }


def _top_late_nodes(rows: List[Dict[str, Any]], k: int = 5) -> List[int]:
    ctr = Counter(r["node"] for r in rows)
    return [node for node, _ in ctr.most_common(k)]


def _readout(late_rows: List[Dict[str, Any]], node_summaries: Dict[str, Any]) -> List[str]:
    reads: List[str] = []
    if any(r["would_flip_positive"] for r in late_rows):
        reads.append("Weighted incident-edge discounting can flip some late losing retirements to positive.")
    if any(r["would_beat_winner"] for r in late_rows):
        reads.append("Weighted incident-edge discounting can make some late retirements beat the accepted winner.")
    if "2" in node_summaries and "6" in node_summaries:
        n2 = node_summaries["2"]
        n6 = node_summaries["6"]
        if n6["incident_loss_weighted_mean"] > n2["incident_loss_weighted_mean"]:
            reads.append("Node 6 still carries a larger weighted incident-loss burden than node 2.")
        if n6["sandbox_lower_deltaF_mean"] > n6["lower_deltaF_mean"]:
            reads.append("The sandbox partially rescues node 6 by discounting some incident-edge losses.")
    if not reads:
        reads.append("Weighted incident-edge discounting does not materially change the late retirement ranking.")
    return reads


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["deltaF_improvement"], r["sandbox_lower_deltaF"]), reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sandbox for weighted incident-edge retirement burden. "
            "Decomposes lost incident edges into subclasses and discounts redundant shell-side loss."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")

    parser.add_argument("--organizer-relief-weight", type=float, default=1.0)
    parser.add_argument("--neighbor-relief-weight", type=float, default=0.5)
    parser.add_argument("--incident-loss-weight", type=float, default=0.5)
    parser.add_argument("--credit-cap", type=float, default=0.35)

    parser.add_argument("--discount-core-adjacent-unique", type=float, default=1.00)
    parser.add_argument("--discount-core-adjacent-shared", type=float, default=0.80)
    parser.add_argument("--discount-peripheral-fragile", type=float, default=0.85)
    parser.add_argument("--discount-peripheral-core-redundant", type=float, default=0.45)
    parser.add_argument("--discount-peripheral-shell-redundant", type=float, default=0.30)

    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_retirement_incident_edge_weighting_sandbox.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = _load_json(in_path)
    rows = _rows_from_run(data, args)
    late = rows[_late_start(len(rows)):] if rows else []

    top_nodes = _top_late_nodes(late, k=5)
    node_summaries = {str(node): _per_node_summary(late, node) for node in top_nodes}

    report = {
        "script": "hsf_mesoscape_retirement_incident_edge_weighting_sandbox.py",
        "input_json": str(in_path),
        "sandbox_config": {
            "organizer_relief_weight": float(args.organizer_relief_weight),
            "neighbor_relief_weight": float(args.neighbor_relief_weight),
            "incident_loss_weight": float(args.incident_loss_weight),
            "credit_cap": float(args.credit_cap),
            "discount_core_adjacent_unique": float(args.discount_core_adjacent_unique),
            "discount_core_adjacent_shared": float(args.discount_core_adjacent_shared),
            "discount_peripheral_fragile": float(args.discount_peripheral_fragile),
            "discount_peripheral_core_redundant": float(args.discount_peripheral_core_redundant),
            "discount_peripheral_shell_redundant": float(args.discount_peripheral_shell_redundant),
        },
        "overall_summary": _summary(rows),
        "late_summary": _summary(late),
        "late_top_nodes": top_nodes,
        "late_node_summaries": node_summaries,
        "flagged_rows": _flagged_rows(rows),
        "readout": _readout(late, node_summaries),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_retirement_incident_edge_weighting_sandbox.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Retirement Incident-Edge Weighting Sandbox ===")
    print(f"rows:                            {report['overall_summary']['n_rows']}")
    print(f"late rows:                       {report['late_summary']['n_rows']}")
    print(f"late top nodes:                  {report['late_top_nodes']}")
    print(f"late positive fraction:          {report['late_summary']['positive_deltaF_fraction']:.6f}")
    print(f"late sandbox positive fraction:  {report['late_summary']['sandbox_positive_deltaF_fraction']:.6f}")
    print(f"late flips to positive:          {report['late_summary']['would_flip_positive_count']}")
    print(f"late beats winner:               {report['late_summary']['would_beat_winner_count']}")
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()