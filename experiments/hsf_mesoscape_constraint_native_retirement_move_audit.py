#!/usr/bin/env python3
# filename: hsf_mesoscape_constraint_native_retirement_move_audit.py

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


def _neighbor_map(edges: Sequence[Sequence[int]]) -> Dict[int, set]:
    nbr: Dict[int, set] = defaultdict(set)
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            a, b = int(e[0]), int(e[1])
            nbr[a].add(b)
            nbr[b].add(a)
    return nbr


def _edge_set(edges: Sequence[Sequence[int]]) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.add(_sorted_edge(int(e[0]), int(e[1])))
    return out


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


def _extract_fp_metric(diag: Dict[str, Any], key: str, metric: str) -> List[float]:
    fps = diag.get(key) or []
    out: List[float] = []
    if isinstance(fps, list):
        for fp in fps:
            if isinstance(fp, dict):
                rm = fp.get("raw_metrics") or {}
                if isinstance(rm, dict):
                    out.append(_safe_float(rm.get(metric)))
    return out


def _extract_fp_weights(diag: Dict[str, Any], key: str) -> List[float]:
    fps = diag.get(key) or []
    out: List[float] = []
    if isinstance(fps, list):
        for fp in fps:
            if isinstance(fp, dict):
                out.append(_safe_float(fp.get("weight")))
    return out


def _counterfactual_after_retirement(node: int, snap: Dict[str, Any]) -> Tuple[set[Tuple[int, int]], Dict[Tuple[int, int], float], List[float]]:
    sigma = ((snap.get("sigma_summary") or {}).get("sigma")) or []
    sigma_after = [float(x) for x in sigma]
    if 0 <= node < len(sigma_after):
        sigma_after[node] = 0.0

    active_edges = snap.get("active_edges") or []
    before_strengths = _edge_strength_map_from_snapshot(snap)

    kept_edges: set[Tuple[int, int]] = set()
    kept_strengths: Dict[Tuple[int, int], float] = {}
    for e in active_edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            a, b = int(e[0]), int(e[1])
            if node in (a, b):
                continue
            se = _sorted_edge(a, b)
            kept_edges.add(se)
            kept_strengths[se] = before_strengths.get(se, 0.0)
    return kept_edges, kept_strengths, sigma_after


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


def _edge_expr_proxy(edge: Tuple[int, int], strength_map: Dict[Tuple[int, int], float], active_edge_set: set[Tuple[int, int]]) -> float:
    strength = float(strength_map.get(edge, 0.0))
    role = _edge_role_terms(edge, active_edge_set)
    base_signal = 0.15 * strength
    role_credit = base_signal * (0.40 + 0.35 * role["novelty"] + 0.25 * role["overlap_relief"])
    distinct_bonus = (0.10 * base_signal + 0.05 * strength) * role["distinctness"]
    redundancy_penalty = 0.70 * (role["redundancy"] ** 2) * max(0.0, 0.40 * strength)
    return float(role_credit + distinct_bonus - redundancy_penalty)


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


def _core_region_edges(core_pair: Sequence[int], active_edge_set: set[Tuple[int, int]]) -> set[Tuple[int, int]]:
    if len(core_pair) != 2:
        return set()
    cp = set(int(x) for x in core_pair)
    nbr = _neighbor_map([list(e) for e in active_edge_set])
    region_nodes = set(cp)
    for c in cp:
        region_nodes.update(nbr.get(c, set()))
    return {e for e in active_edge_set if e[0] in region_nodes and e[1] in region_nodes}


def _witness_sector_audit(step: int, snap: Dict[str, Any], lower_diag: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    node = _safe_int(lower_diag.get("node", lower_diag.get("move_object")))
    core_pair = _extract_core_pair_from_snap(snap)
    core_set = set(core_pair)

    active_edges = snap.get("active_edges") or []
    before_edge_set = _edge_set(active_edges)
    before_strengths = _edge_strength_map_from_snapshot(snap)
    after_edge_set, after_strengths, sigma_after = _counterfactual_after_retirement(node, snap)

    sigma_before = ((snap.get("sigma_summary") or {}).get("sigma")) or []
    sigma_before = [float(x) for x in sigma_before]

    incident_edges, neighbor_edges = _incident_and_neighbor_edges(node, before_edge_set)

    # CR: core continuity / organizer anchor continuity
    core_edge = _sorted_edge(core_pair[0], core_pair[1]) if len(core_pair) == 2 else None
    core_strength_before = float(before_strengths.get(core_edge, 0.0)) if core_edge else 0.0
    core_strength_after = float(after_strengths.get(core_edge, 0.0)) if core_edge else 0.0
    core_expr_before = _edge_expr_proxy(core_edge, before_strengths, before_edge_set) if core_edge and core_edge in before_edge_set else 0.0
    core_expr_after = _edge_expr_proxy(core_edge, after_strengths, after_edge_set) if core_edge and core_edge in after_edge_set else 0.0
    P_C = 1.0 if core_edge and core_edge in after_edge_set else 0.0

    # SR: support continuity / shell lawful re-expression
    shell_before = set(_shell_nodes(core_pair, active_edges))
    shell_after = set(_shell_nodes(core_pair, [list(e) for e in after_edge_set]))
    shell_union = shell_before | shell_after
    shell_overlap = len(shell_before & shell_after) / max(1, len(shell_union)) if shell_union else 1.0
    retired_support_mass = float(sigma_before[node]) if 0 <= node < len(sigma_before) else 0.0
    remaining_shell_mass_before = float(sum(sigma_before[s] for s in shell_before if 0 <= s < len(sigma_before)))
    remaining_shell_mass_after = float(sum(sigma_after[s] for s in shell_after if 0 <= s < len(sigma_after)))
    shell_mass_retention = remaining_shell_mass_after / max(1e-12, remaining_shell_mass_before) if remaining_shell_mass_before > 0 else 1.0
    P_S = float(0.5 * shell_overlap + 0.5 * min(1.0, shell_mass_retention))

    # MR: function/correlation proxy from local role + retained activity proxies
    odiff_before = _safe_float(lower_diag.get("Odiff_before_R"))
    odiff_after = _safe_float(lower_diag.get("Odiff_after_R"))
    delta_odiff = _safe_float(lower_diag.get("delta_Odiff_R"))
    role_weight_before = _mean(_extract_fp_weights(lower_diag, "role_fingerprints_before"))
    role_weight_after = _mean(_extract_fp_weights(lower_diag, "role_fingerprints_after"))
    activity_before = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_before", "activity_sum"))
    activity_after = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_after", "activity_sum"))
    retained_activity_ratio = activity_after / max(1e-12, activity_before) if activity_before > 0 else 1.0
    P_M = float(
        max(0.0, min(1.0, 0.40 * (odiff_after / max(1e-12, odiff_before)) if odiff_before > 0 else 1.0))
        + 0.30 * max(0.0, min(1.0, retained_activity_ratio))
        + 0.30 * max(0.0, min(1.0, role_weight_after / max(1e-12, role_weight_before) if role_weight_before > 0 else 1.0))
    )
    P_M = float(min(1.0, P_M))

    # KR: committed interface bookkeeping continuity in local organizer region
    core_region_before = _core_region_edges(core_pair, before_edge_set)
    core_region_after = _core_region_edges(core_pair, after_edge_set)
    common_core_region = core_region_before & core_region_after
    bookkeeping_retained = len(common_core_region) / max(1, len(core_region_before)) if core_region_before else 1.0
    incident_core_adjacent = [e for e in incident_edges if (e[0] in core_set or e[1] in core_set)]
    core_adjacent_loss = len(incident_core_adjacent) / max(1, len(incident_edges)) if incident_edges else 0.0
    P_K = float(max(0.0, min(1.0, bookkeeping_retained * (1.0 - 0.5 * core_adjacent_loss))))

    # BR: bandwidth / spread continuity proxy
    active_count_before = _safe_int(snap.get("active_edge_count"), len(before_edge_set))
    active_count_after = len(after_edge_set)
    spread_pen_before = _safe_float((lower_diag.get("dCS") or 0.0))
    # For sandbox we only know delta, so use active-edge contraction and no increase in core region slack as mild positives.
    edge_contraction = max(0.0, (active_count_before - active_count_after) / max(1, active_count_before))
    P_B = float(max(0.0, min(1.0, 0.5 + 0.5 * edge_contraction - 0.25 * max(0.0, spread_pen_before))))

    # Expression-side local branch score from the move note's F_R style audit.
    # Use existing move diagnostics plus local sector-preservation summary.
    dE_expr = _safe_float(lower_diag.get("dE_expr"))
    dCB = _safe_float(lower_diag.get("dCB"))
    dCS = _safe_float(lower_diag.get("dCS"))
    dCF = _safe_float(lower_diag.get("dCF"))
    W_NR = _safe_float(lower_diag.get("W_NR"))
    F_R = (
        dE_expr
        - float(args.lambda_B) * dCB
        - float(args.lambda_R) * W_NR
        - float(args.lambda_S) * dCS
        - float(args.lambda_F) * dCF
    )

    # Organizer evacuation indicator from organizer note:
    # retired support is evacuated if MR stays high enough, KR stays high enough,
    # and surviving shell/core carries the role.
    evacuated = bool(
        P_M >= float(args.evacuate_PM_min)
        and P_K >= float(args.evacuate_PK_min)
        and P_S >= float(args.evacuate_PS_min)
        and P_C >= float(args.evacuate_PC_min)
    )

    return {
        "step": int(step),
        "node": int(node),
        "winner_move_type": str(_winner_from_snapshot(snap) or "none"),
        "core_pair": core_pair,
        "touches_core": bool(node in core_set),
        "F_R_retire": float(F_R),
        "retire_would_win_by_F_R": bool(F_R > float(args.win_threshold)),
        "organizer_evacuated": bool(evacuated),
        "P_C": float(P_C),
        "P_S": float(P_S),
        "P_M": float(P_M),
        "P_K": float(P_K),
        "P_B": float(P_B),
        "core_strength_before": float(core_strength_before),
        "core_strength_after": float(core_strength_after),
        "core_expr_before": float(core_expr_before),
        "core_expr_after": float(core_expr_after),
        "shell_overlap": float(shell_overlap),
        "shell_mass_retention": float(shell_mass_retention),
        "retained_activity_ratio": float(retained_activity_ratio),
        "bookkeeping_retained": float(bookkeeping_retained),
        "core_adjacent_loss_fraction": float(core_adjacent_loss),
        "edge_contraction": float(edge_contraction),
        "dE_expr": float(dE_expr),
        "dE_expr_raw": _safe_float(lower_diag.get("dE_expr_raw")),
        "delta_Odiff_R": float(delta_odiff),
        "Odiff_before_R": float(odiff_before),
        "Odiff_after_R": float(odiff_after),
        "dCB": float(dCB),
        "dCS": float(dCS),
        "dCF": float(dCF),
        "W_NR": float(W_NR),
        "expr_redundancy_removed": _safe_float(lower_diag.get("expr_redundancy_removed")),
        "expr_distinctness_gain": _safe_float(lower_diag.get("expr_distinctness_gain")),
        "incident_edges": [[a, b] for a, b in incident_edges],
        "neighbor_edges": [[a, b] for a, b in neighbor_edges],
    }


def _rows_from_run(data: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snap in data.get("snapshots", []) or []:
        step = _safe_int(snap.get("step"))
        cands = snap.get("candidate_move_diagnostics", []) or []
        lowers = [d for d in cands if str(d.get("move_type")) == "lower_support"]
        if not lowers:
            continue
        best_lower = max(lowers, key=lambda d: _safe_float(d.get("deltaF")))
        rows.append(_witness_sector_audit(step, snap, best_lower, args))
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in rows)),
        "retire_positive_FR_fraction": _mean([1.0 if r["F_R_retire"] > 0.0 else 0.0 for r in rows]),
        "retire_evacuated_fraction": _mean([1.0 if r["organizer_evacuated"] else 0.0 for r in rows]),
        "FR_mean": _mean([r["F_R_retire"] for r in rows]),
        "P_C_mean": _mean([r["P_C"] for r in rows]),
        "P_S_mean": _mean([r["P_S"] for r in rows]),
        "P_M_mean": _mean([r["P_M"] for r in rows]),
        "P_K_mean": _mean([r["P_K"] for r in rows]),
        "P_B_mean": _mean([r["P_B"] for r in rows]),
    }


def _per_node_summary(rows: List[Dict[str, Any]], node_id: int) -> Dict[str, Any]:
    group = [r for r in rows if r["node"] == node_id]
    return {
        "n": len(group),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in group)),
        "retire_positive_FR_fraction": _mean([1.0 if r["F_R_retire"] > 0.0 else 0.0 for r in group]),
        "retire_evacuated_fraction": _mean([1.0 if r["organizer_evacuated"] else 0.0 for r in group]),
        "FR_mean": _mean([r["F_R_retire"] for r in group]),
        "P_C_mean": _mean([r["P_C"] for r in group]),
        "P_S_mean": _mean([r["P_S"] for r in group]),
        "P_M_mean": _mean([r["P_M"] for r in group]),
        "P_K_mean": _mean([r["P_K"] for r in group]),
        "P_B_mean": _mean([r["P_B"] for r in group]),
        "dE_expr_mean": _mean([r["dE_expr"] for r in group]),
        "W_NR_mean": _mean([r["W_NR"] for r in group]),
        "dCF_mean": _mean([r["dCF"] for r in group]),
    }


def _top_late_nodes(rows: List[Dict[str, Any]], k: int = 5) -> List[int]:
    ctr = Counter(r["node"] for r in rows)
    return [node for node, _ in ctr.most_common(k)]


def _readout(late_rows: List[Dict[str, Any]], node_summaries: Dict[str, Any]) -> List[str]:
    reads: List[str] = []
    if any(r["organizer_evacuated"] for r in late_rows):
        reads.append("Some late demotion candidates satisfy organizer-evacuation conditions in the witness sectors.")
    if any(r["F_R_retire"] > 0.0 for r in late_rows):
        reads.append("Some late demotion candidates are positive under the constraint-native local F_R audit.")
    if "2" in node_summaries and "6" in node_summaries:
        n2 = node_summaries["2"]
        n6 = node_summaries["6"]
        if n2["P_M_mean"] > n6["P_M_mean"]:
            reads.append("Node 2 preserves the function-bearing MR sector better than node 6.")
        if n2["P_K_mean"] >= n6["P_K_mean"]:
            reads.append("Node 6 is not mainly losing on KR relative to node 2.")
        if n6["FR_mean"] < n2["FR_mean"]:
            reads.append("The node-2 / node-6 split remains concentrated in the local constraint-native retirement score.")
    if not reads:
        reads.append("No single witness sector dominates the late retirement split under this audit.")
    return reads


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["F_R_retire"], r["P_M"], r["P_K"]), reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Constraint-native retirement move audit. "
            "Scores best lower_support candidates with a local F_R functional and organizer-witness sectors "
            "instead of subclass-tuned rescue credits."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")
    parser.add_argument("--lambda-B", type=float, default=0.18)
    parser.add_argument("--lambda-R", type=float, default=0.35)
    parser.add_argument("--lambda-S", type=float, default=0.12)
    parser.add_argument("--lambda-F", type=float, default=0.20)
    parser.add_argument("--win-threshold", type=float, default=0.0)
    parser.add_argument("--evacuate-PC-min", type=float, default=0.50)
    parser.add_argument("--evacuate-PS-min", type=float, default=0.55)
    parser.add_argument("--evacuate-PM-min", type=float, default=0.60)
    parser.add_argument("--evacuate-PK-min", type=float, default=0.60)
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_constraint_native_retirement_move_audit.json",
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
        "script": "hsf_mesoscape_constraint_native_retirement_move_audit.py",
        "input_json": str(in_path),
        "audit_config": {
            "lambda_B": float(args.lambda_B),
            "lambda_R": float(args.lambda_R),
            "lambda_S": float(args.lambda_S),
            "lambda_F": float(args.lambda_F),
            "win_threshold": float(args.win_threshold),
            "evacuate_PC_min": float(args.evacuate_PC_min),
            "evacuate_PS_min": float(args.evacuate_PS_min),
            "evacuate_PM_min": float(args.evacuate_PM_min),
            "evacuate_PK_min": float(args.evacuate_PK_min),
        },
        "overall_summary": _summary(rows),
        "late_summary": _summary(late),
        "late_top_nodes": top_nodes,
        "late_node_summaries": node_summaries,
        "flagged_rows": _flagged_rows(rows),
        "readout": _readout(late, node_summaries),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_constraint_native_retirement_move_audit.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Constraint-Native Retirement Move Audit ===")
    print(f"rows:                         {report['overall_summary']['n_rows']}")
    print(f"late rows:                    {report['late_summary']['n_rows']}")
    print(f"late top nodes:               {report['late_top_nodes']}")
    print(f"late positive F_R fraction:   {report['late_summary']['retire_positive_FR_fraction']:.6f}")
    print(f"late evacuated fraction:      {report['late_summary']['retire_evacuated_fraction']:.6f}")
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()