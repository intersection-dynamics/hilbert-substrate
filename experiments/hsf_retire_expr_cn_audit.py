#!/usr/bin/env python3
# filename: hsf_retire_expr_cn_audit.py

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


def _extract_fp_weights(diag: Dict[str, Any], key: str) -> List[float]:
    fps = diag.get(key) or []
    out: List[float] = []
    if isinstance(fps, list):
        for fp in fps:
            if isinstance(fp, dict):
                out.append(_safe_float(fp.get("weight")))
    return out


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


def _neighbor_map(edges: Sequence[Sequence[int]]) -> Dict[int, set]:
    nbr: Dict[int, set] = defaultdict(set)
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            a, b = int(e[0]), int(e[1])
            nbr[a].add(b)
            nbr[b].add(a)
    return nbr


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


def _core_region_edges(core_pair: Sequence[int], active_edges: Sequence[Sequence[int]]) -> set[Tuple[int, int]]:
    if len(core_pair) != 2:
        return set()
    cp = set(int(x) for x in core_pair)
    nbr = _neighbor_map(active_edges)
    region_nodes = set(cp)
    for c in cp:
        region_nodes.update(nbr.get(c, set()))
    out = set()
    for e in active_edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            a, b = int(e[0]), int(e[1])
            if a in region_nodes and b in region_nodes:
                out.add(_sorted_edge(a, b))
    return out


def _continuous_evac_score(P_C: float, P_S: float, P_M: float, P_K: float, P_B: float, wC: float, wS: float, wM: float, wK: float, wB: float) -> float:
    return float(wC * P_C + wS * P_S + wM * P_M + wK * P_K + wB * P_B)


def _local_sector_scores(snap: Dict[str, Any], lower_diag: Dict[str, Any]) -> Dict[str, float]:
    node = _safe_int(lower_diag.get("node", lower_diag.get("move_object")))
    core_pair = _extract_core_pair_from_snap(snap)
    active_edges = snap.get("active_edges") or []

    sigma_before = ((snap.get("sigma_summary") or {}).get("sigma")) or []
    sigma_before = [float(x) for x in sigma_before]
    sigma_after = list(sigma_before)
    if 0 <= node < len(sigma_after):
        sigma_after[node] = 0.0

    # core continuity
    core_edge = _sorted_edge(core_pair[0], core_pair[1]) if len(core_pair) == 2 else None
    active_edge_set = {_sorted_edge(int(e[0]), int(e[1])) for e in active_edges if isinstance(e, (list, tuple)) and len(e) == 2}
    P_C = 1.0 if core_edge and core_edge in active_edge_set else 0.0

    # shell continuity
    shell_before = set(_shell_nodes(core_pair, active_edges))
    shell_after = set(shell_before)
    if node in shell_after:
        shell_after.remove(node)
    shell_union = shell_before | shell_after
    shell_overlap = len(shell_before & shell_after) / max(1, len(shell_union)) if shell_union else 1.0

    shell_mass_before = float(sum(sigma_before[s] for s in shell_before if 0 <= s < len(sigma_before)))
    shell_mass_after = float(sum(sigma_after[s] for s in shell_after if 0 <= s < len(sigma_after)))
    shell_mass_retention = shell_mass_after / max(1e-12, shell_mass_before) if shell_mass_before > 0 else 1.0
    P_S = float(0.5 * shell_overlap + 0.5 * min(1.0, shell_mass_retention))

    # function continuity from local role data
    odiff_before = _safe_float(lower_diag.get("Odiff_before_R"))
    odiff_after = _safe_float(lower_diag.get("Odiff_after_R"))
    role_weight_before = _mean(_extract_fp_weights(lower_diag, "role_fingerprints_before"))
    role_weight_after = _mean(_extract_fp_weights(lower_diag, "role_fingerprints_after"))
    activity_before = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_before", "activity_sum"))
    activity_after = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_after", "activity_sum"))

    odiff_ratio = odiff_after / max(1e-12, odiff_before) if odiff_before > 0 else 1.0
    role_ratio = role_weight_after / max(1e-12, role_weight_before) if role_weight_before > 0 else 1.0
    activity_ratio = activity_after / max(1e-12, activity_before) if activity_before > 0 else 1.0

    P_M = float(
        min(
            1.0,
            max(0.0, 0.40 * min(1.0, odiff_ratio))
            + 0.30 * max(0.0, min(1.0, role_ratio))
            + 0.30 * max(0.0, min(1.0, activity_ratio)),
        )
    )

    # bookkeeping continuity
    region_before = _core_region_edges(core_pair, active_edges)
    active_edges_after = [list(e) for e in active_edge_set if node not in e]
    region_after = _core_region_edges(core_pair, active_edges_after)
    bookkeeping_retained = len(region_before & region_after) / max(1, len(region_before)) if region_before else 1.0

    nbr = _neighbor_map(active_edges)
    neighbors = sorted(nbr.get(node, set()))
    core_set = set(core_pair)
    incident_edges = [_sorted_edge(node, n) for n in neighbors]
    incident_core_adj = [e for e in incident_edges if e[0] in core_set or e[1] in core_set]
    core_adj_loss_fraction = len(incident_core_adj) / max(1, len(incident_edges)) if incident_edges else 0.0
    P_K = float(max(0.0, min(1.0, bookkeeping_retained * (1.0 - 0.5 * core_adj_loss_fraction))))

    # bandwidth/spread continuity
    active_count_before = len(active_edge_set)
    active_count_after = len(active_edge_set - set(incident_edges))
    edge_contraction = max(0.0, (active_count_before - active_count_after) / max(1, active_count_before))
    dCS = _safe_float(lower_diag.get("dCS"))
    P_B = float(max(0.0, min(1.0, 0.5 + 0.5 * edge_contraction - 0.25 * max(0.0, dCS))))

    return {
        "P_C": float(P_C),
        "P_S": float(P_S),
        "P_M": float(P_M),
        "P_K": float(P_K),
        "P_B": float(P_B),
        "shell_overlap": float(shell_overlap),
        "shell_mass_retention": float(shell_mass_retention),
        "odiff_ratio": float(odiff_ratio),
        "role_ratio": float(role_ratio),
        "activity_ratio": float(activity_ratio),
        "bookkeeping_retained": float(bookkeeping_retained),
        "core_adj_loss_fraction": float(core_adj_loss_fraction),
        "edge_contraction": float(edge_contraction),
    }


def _row_from_diag(step: int, snap: Dict[str, Any], lower_diag: Dict[str, Any], winner_type: str, args: argparse.Namespace) -> Dict[str, Any]:
    node = _safe_int(lower_diag.get("node", lower_diag.get("move_object")))
    core_pair = _extract_core_pair_from_snap(snap)
    sectors = _local_sector_scores(snap, lower_diag)

    dE_expr = _safe_float(lower_diag.get("dE_expr"))
    dE_expr_raw = _safe_float(lower_diag.get("dE_expr_raw"))
    dCB = _safe_float(lower_diag.get("dCB"))
    dCS = _safe_float(lower_diag.get("dCS"))
    dCF = _safe_float(lower_diag.get("dCF"))
    W_NR = _safe_float(lower_diag.get("W_NR"))

    F_R = dE_expr - float(args.lambda_B) * dCB - float(args.lambda_R) * W_NR - float(args.lambda_S) * dCS - float(args.lambda_F) * dCF
    evac_score = _continuous_evac_score(
        sectors["P_C"], sectors["P_S"], sectors["P_M"], sectors["P_K"], sectors["P_B"],
        float(args.wC), float(args.wS), float(args.wM), float(args.wK), float(args.wB)
    )

    if F_R > float(args.win_threshold) and evac_score >= float(args.evac_score_min):
        case = "lawful_retire"
    elif F_R > float(args.win_threshold) and evac_score < float(args.evac_score_min):
        case = "score_positive_evac_weak"
    elif F_R <= float(args.win_threshold) and evac_score >= float(args.evac_score_min):
        case = "evac_ok_expr_bad"
    else:
        case = "retire_unlawful"

    return {
        "step": int(step),
        "node": int(node),
        "winner_move_type": str(winner_type),
        "core_pair": core_pair,
        "touches_core": bool(node in set(core_pair)),
        "case": case,
        "F_R_retire": float(F_R),
        "evac_score": float(evac_score),
        "retire_positive_FR": bool(F_R > float(args.win_threshold)),
        "retire_evac_ok": bool(evac_score >= float(args.evac_score_min)),
        **sectors,
        "dE_expr": float(dE_expr),
        "dE_expr_raw": float(dE_expr_raw),
        "delta_Odiff_R": _safe_float(lower_diag.get("delta_Odiff_R")),
        "Odiff_before_R": _safe_float(lower_diag.get("Odiff_before_R")),
        "Odiff_after_R": _safe_float(lower_diag.get("Odiff_after_R")),
        "dCB": float(dCB),
        "dCS": float(dCS),
        "dCF": float(dCF),
        "W_NR": float(W_NR),
        "expr_redundancy_removed": _safe_float(lower_diag.get("expr_redundancy_removed")),
        "expr_distinctness_gain": _safe_float(lower_diag.get("expr_distinctness_gain")),
        "retirement_readiness": _safe_float((lower_diag.get("retirement_info") or {}).get("retirement_readiness")),
        "shell_penalty": _safe_float((lower_diag.get("retirement_info") or {}).get("shell_penalty")),
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
        rows.append(_row_from_diag(step, snap, best_lower, str(winner.get("move_type")), args))
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in rows)),
        "case_counts": dict(Counter(r["case"] for r in rows)),
        "positive_FR_fraction": _mean([1.0 if r["retire_positive_FR"] else 0.0 for r in rows]),
        "evac_ok_fraction": _mean([1.0 if r["retire_evac_ok"] else 0.0 for r in rows]),
        "FR_mean": _mean([r["F_R_retire"] for r in rows]),
        "evac_score_mean": _mean([r["evac_score"] for r in rows]),
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
        "case_counts": dict(Counter(r["case"] for r in group)),
        "positive_FR_fraction": _mean([1.0 if r["retire_positive_FR"] else 0.0 for r in group]),
        "evac_ok_fraction": _mean([1.0 if r["retire_evac_ok"] else 0.0 for r in group]),
        "FR_mean": _mean([r["F_R_retire"] for r in group]),
        "evac_score_mean": _mean([r["evac_score"] for r in group]),
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
    if any(r["retire_evac_ok"] for r in late_rows):
        reads.append("Some late demotion candidates satisfy the graded organizer-evacuation score.")
    if any(r["retire_positive_FR"] for r in late_rows):
        reads.append("Some late demotion candidates are positive under the local constraint-native retirement score.")
    if "2" in node_summaries and "6" in node_summaries:
        n2 = node_summaries["2"]
        n6 = node_summaries["6"]
        if n2["P_M_mean"] > n6["P_M_mean"]:
            reads.append("Node 2 preserves the function-bearing M-sector better than node 6.")
        if abs(n2["P_K_mean"] - n6["P_K_mean"]) < 0.05:
            reads.append("Node 2 and node 6 are not strongly separated by K-sector continuity.")
        if n6["FR_mean"] < n2["FR_mean"]:
            reads.append("The node-2 / node-6 split remains concentrated in the local retirement score rather than shell status alone.")
    if not reads:
        reads.append("No single graded witness sector dominates the late retirement split.")
    return reads


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["F_R_retire"], r["evac_score"], r["P_M"]), reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Constraint-native retirement expression audit with graded organizer-evacuation score. "
            "Short filename for GitHub-friendly workflows."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")
    parser.add_argument("--lambda-B", type=float, default=0.18)
    parser.add_argument("--lambda-R", type=float, default=0.35)
    parser.add_argument("--lambda-S", type=float, default=0.12)
    parser.add_argument("--lambda-F", type=float, default=0.20)
    parser.add_argument("--win-threshold", type=float, default=0.0)

    parser.add_argument("--wC", type=float, default=0.20)
    parser.add_argument("--wS", type=float, default=0.20)
    parser.add_argument("--wM", type=float, default=0.30)
    parser.add_argument("--wK", type=float, default=0.20)
    parser.add_argument("--wB", type=float, default=0.10)
    parser.add_argument("--evac-score-min", type=float, default=0.60)

    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_rexpr_cn.json",
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
        "script": "hsf_retire_expr_cn_audit.py",
        "input_json": str(in_path),
        "audit_config": {
            "lambda_B": float(args.lambda_B),
            "lambda_R": float(args.lambda_R),
            "lambda_S": float(args.lambda_S),
            "lambda_F": float(args.lambda_F),
            "win_threshold": float(args.win_threshold),
            "wC": float(args.wC),
            "wS": float(args.wS),
            "wM": float(args.wM),
            "wK": float(args.wK),
            "wB": float(args.wB),
            "evac_score_min": float(args.evac_score_min),
        },
        "overall_summary": _summary(rows),
        "late_summary": _summary(late),
        "late_top_nodes": top_nodes,
        "late_node_summaries": node_summaries,
        "flagged_rows": _flagged_rows(rows),
        "readout": _readout(late, node_summaries),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_rexpr_cn.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Retire Expr Constraint-Native Audit ===")
    print(f"rows:                    {report['overall_summary']['n_rows']}")
    print(f"late rows:               {report['late_summary']['n_rows']}")
    print(f"late top nodes:          {report['late_top_nodes']}")
    print(f"late positive F_R frac:  {report['late_summary']['positive_FR_fraction']:.6f}")
    print(f"late evac-ok frac:       {report['late_summary']['evac_ok_fraction']:.6f}")
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()