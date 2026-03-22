#!/usr/bin/env python3
# filename: hsf_mesoscape_retirement_expression_audit.py

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


def _late_start(n: int) -> int:
    return max(0, (2 * n) // 3)


def _extract_core_pair(diag: Dict[str, Any]) -> List[int]:
    core_before = diag.get("core_before", {}) or {}
    cp = core_before.get("core_pair")
    if isinstance(cp, list) and len(cp) == 2:
        return [int(cp[0]), int(cp[1])]
    return []


def _extract_fp_sites(diag: Dict[str, Any], key: str) -> List[int]:
    fps = diag.get(key) or []
    out: List[int] = []
    if isinstance(fps, list):
        for fp in fps:
            if isinstance(fp, dict):
                out.append(_safe_int(fp.get("site_id")))
    return out


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


def _retire_case(diag: Dict[str, Any]) -> str:
    dF = _safe_float(diag.get("deltaF"))
    dE_raw = _safe_float(diag.get("dE_expr_raw"))
    dE = _safe_float(diag.get("dE_expr"))
    dO = _safe_float(diag.get("delta_Odiff_R"))
    expr_adj = _safe_float(diag.get("dE_expr_structural_adjustment", diag.get("expr_adjustment")))
    red_removed = _safe_float(diag.get("expr_redundancy_removed"))
    dist_gain = _safe_float(diag.get("expr_distinctness_gain"))
    WNR = _safe_float(diag.get("W_NR"))
    dCF = _safe_float(diag.get("dCF"))

    if dF > 0.0 and dE > 0.0 and dO > 0.0:
        return "lawful_expression_positive_retirement"
    if dF > 0.0 and expr_adj > 0.0 and dE_raw <= 0.0:
        return "wins_via_structural_expression_adjustment"
    if dF <= 0.0 and dE <= 0.0 and dO <= 0.0:
        return "expression_destructive_retirement"
    if dF <= 0.0 and dE_raw < 0.0 and expr_adj > 0.0:
        return "adjustment_insufficient_to_rescue"
    if dF <= 0.0 and WNR > 1.0:
        return "no_refolding_dominant_loss"
    if dF <= 0.0 and dCF > 0.25:
        return "functional_cost_dominant_loss"
    if red_removed > 0.0 and dist_gain <= 0.0:
        return "redundancy_removed_but_no_distinctness_gain"
    return "mixed"


def _build_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
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

        lowers_sorted = sorted(lowers, key=lambda d: _safe_float(d.get("deltaF")), reverse=True)
        best_lower = lowers_sorted[0]
        node = _safe_int(best_lower.get("node", best_lower.get("move_object")))
        rr = best_lower.get("retirement_info") or {}
        winner_type = str(winner.get("move_type"))
        core_pair = _extract_core_pair(best_lower)

        row = {
            "step": int(step),
            "winner_move_type": winner_type,
            "winner_deltaF": _safe_float(winner.get("deltaF")),
            "lower_node": int(node),
            "lower_deltaF": _safe_float(best_lower.get("deltaF")),
            "deltaF_gap_to_winner": _safe_float(winner.get("deltaF")) - _safe_float(best_lower.get("deltaF")),
            "dE_expr_raw": _safe_float(best_lower.get("dE_expr_raw")),
            "dE_expr": _safe_float(best_lower.get("dE_expr")),
            "dE_expr_structural_adjustment": _safe_float(
                best_lower.get("dE_expr_structural_adjustment", best_lower.get("expr_adjustment"))
            ),
            "delta_Odiff_R": _safe_float(best_lower.get("delta_Odiff_R")),
            "Odiff_before_R": _safe_float(best_lower.get("Odiff_before_R")),
            "Odiff_after_R": _safe_float(best_lower.get("Odiff_after_R")),
            "expr_redundancy_removed": _safe_float(best_lower.get("expr_redundancy_removed")),
            "expr_distinctness_gain": _safe_float(best_lower.get("expr_distinctness_gain")),
            "W_NR": _safe_float(best_lower.get("W_NR")),
            "dCF": _safe_float(best_lower.get("dCF")),
            "dCB": _safe_float(best_lower.get("dCB")),
            "dCS": _safe_float(best_lower.get("dCS")),
            "retirement_readiness": _safe_float(rr.get("retirement_readiness")),
            "bookkeeping_safety": _safe_float(rr.get("bookkeeping_safety")),
            "functional_ready": _safe_float(rr.get("functional_ready")),
            "substitutability": _safe_float(rr.get("substitutability")),
            "shell_penalty": _safe_float(rr.get("shell_penalty")),
            "shell_indispensability": _safe_float(rr.get("shell_indispensability")),
            "core_penalty": _safe_float(rr.get("core_penalty")),
            "core_pair": core_pair,
            "touches_core": bool(node in set(core_pair)),
            "role_sites_before": _extract_fp_sites(best_lower, "role_fingerprints_before"),
            "role_sites_after": _extract_fp_sites(best_lower, "role_fingerprints_after"),
            "role_weight_before_mean": _mean(_extract_fp_weights(best_lower, "role_fingerprints_before")),
            "role_weight_after_mean": _mean(_extract_fp_weights(best_lower, "role_fingerprints_after")),
            "role_cluster_before_mean": _mean(_extract_fp_metric(best_lower, "role_fingerprints_before", "local_cluster")),
            "role_cluster_after_mean": _mean(_extract_fp_metric(best_lower, "role_fingerprints_after", "local_cluster")),
            "role_incident_before_mean": _mean(_extract_fp_metric(best_lower, "role_fingerprints_before", "incident_count")),
            "role_incident_after_mean": _mean(_extract_fp_metric(best_lower, "role_fingerprints_after", "incident_count")),
        }
        row["case"] = _retire_case(best_lower)
        rows.append(row)
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in rows)),
        "case_counts": dict(Counter(r["case"] for r in rows)),
        "positive_deltaF_fraction": _mean([1.0 if r["lower_deltaF"] > 0.0 else 0.0 for r in rows]),
        "dE_expr_raw_mean": _mean([r["dE_expr_raw"] for r in rows]),
        "dE_expr_mean": _mean([r["dE_expr"] for r in rows]),
        "dE_expr_structural_adjustment_mean": _mean([r["dE_expr_structural_adjustment"] for r in rows]),
        "delta_Odiff_mean": _mean([r["delta_Odiff_R"] for r in rows]),
        "redundancy_removed_mean": _mean([r["expr_redundancy_removed"] for r in rows]),
        "distinctness_gain_mean": _mean([r["expr_distinctness_gain"] for r in rows]),
        "W_NR_mean": _mean([r["W_NR"] for r in rows]),
        "dCF_mean": _mean([r["dCF"] for r in rows]),
        "gap_to_winner_mean": _mean([r["deltaF_gap_to_winner"] for r in rows]),
    }


def _per_node_summary(rows: List[Dict[str, Any]], node_id: int) -> Dict[str, Any]:
    group = [r for r in rows if r["lower_node"] == node_id]
    return {
        "n": len(group),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in group)),
        "case_counts": dict(Counter(r["case"] for r in group)),
        "positive_deltaF_fraction": _mean([1.0 if r["lower_deltaF"] > 0.0 else 0.0 for r in group]),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in group]),
        "dE_expr_raw_mean": _mean([r["dE_expr_raw"] for r in group]),
        "dE_expr_mean": _mean([r["dE_expr"] for r in group]),
        "structural_adjustment_mean": _mean([r["dE_expr_structural_adjustment"] for r in group]),
        "delta_Odiff_mean": _mean([r["delta_Odiff_R"] for r in group]),
        "redundancy_removed_mean": _mean([r["expr_redundancy_removed"] for r in group]),
        "distinctness_gain_mean": _mean([r["expr_distinctness_gain"] for r in group]),
        "role_weight_before_mean": _mean([r["role_weight_before_mean"] for r in group]),
        "role_weight_after_mean": _mean([r["role_weight_after_mean"] for r in group]),
        "role_cluster_before_mean": _mean([r["role_cluster_before_mean"] for r in group]),
        "role_cluster_after_mean": _mean([r["role_cluster_after_mean"] for r in group]),
        "role_incident_before_mean": _mean([r["role_incident_before_mean"] for r in group]),
        "role_incident_after_mean": _mean([r["role_incident_after_mean"] for r in group]),
        "shell_penalty_mean": _mean([r["shell_penalty"] for r in group]),
        "W_NR_mean": _mean([r["W_NR"] for r in group]),
        "dCF_mean": _mean([r["dCF"] for r in group]),
    }


def _top_late_nodes(rows: List[Dict[str, Any]], k: int = 5) -> List[int]:
    ctr = Counter(r["lower_node"] for r in rows)
    return [node for node, _ in ctr.most_common(k)]


def _readout(all_rows: List[Dict[str, Any]], late_rows: List[Dict[str, Any]], node_summaries: Dict[str, Any]) -> List[str]:
    reads: List[str] = []
    all_cases = Counter(r["case"] for r in all_rows)
    late_cases = Counter(r["case"] for r in late_rows)

    if late_cases.get("lawful_expression_positive_retirement", 0) > 0:
        reads.append("Some late retirements are genuinely expression-positive and lawful on total score.")
    if late_cases.get("expression_destructive_retirement", 0) > 0:
        reads.append("Some late retirements still look destructive at the expression level.")
    if late_cases.get("adjustment_insufficient_to_rescue", 0) > 0:
        reads.append("For some late nodes, structural retirement credit is present but too small to overcome raw expression loss.")
    if "2" in node_summaries and "6" in node_summaries:
        n2 = node_summaries["2"]
        n6 = node_summaries["6"]
        if n2["dE_expr_mean"] > n6["dE_expr_mean"]:
            reads.append("Node 2 is retirement-favored relative to node 6 mainly on expression, not shell metadata.")
        if n6["dE_expr_raw_mean"] < n2["dE_expr_raw_mean"]:
            reads.append("Node 6 carries a systematically more negative raw retirement expression than node 2.")
    if not reads:
        reads.append("No single retirement-expression pattern dominates the node-level demotion outcomes.")
    return reads


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 30) -> List[Dict[str, Any]]:
    flagged = sorted(
        rows,
        key=lambda r: (
            r["lower_deltaF"],
            r["dE_expr"],
            -r["step"],
        ),
        reverse=True,
    )
    return flagged[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Focused retirement expression audit for node-level demotion. "
            "Compares winning and losing lower_support nodes, especially late-time recurrent cases."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_retirement_expression_audit.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = _load_json(in_path)
    rows = _build_rows(data)
    late = rows[_late_start(len(rows)):] if rows else []

    top_nodes = _top_late_nodes(late, k=5)
    node_summaries = {str(node): _per_node_summary(late, node) for node in top_nodes}

    report = {
        "script": "hsf_mesoscape_retirement_expression_audit.py",
        "input_json": str(in_path),
        "overall_summary": _summary(rows),
        "late_summary": _summary(late),
        "late_top_nodes": top_nodes,
        "late_node_summaries": node_summaries,
        "flagged_rows": _flagged_rows(rows),
        "readout": _readout(rows, late, node_summaries),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_retirement_expression_audit.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Retirement Expression Audit ===")
    print(f"rows:                    {report['overall_summary']['n_rows']}")
    print(f"late rows:               {report['late_summary']['n_rows']}")
    print(f"overall cases:           {report['overall_summary']['case_counts']}")
    print(f"late cases:              {report['late_summary']['case_counts']}")
    print(f"late top nodes:          {report['late_top_nodes']}")
    print(f"late dE_expr_raw mean:   {report['late_summary']['dE_expr_raw_mean']:.6f}")
    print(f"late dE_expr mean:       {report['late_summary']['dE_expr_mean']:.6f}")
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()