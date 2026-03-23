#!/usr/bin/env python3
# filename: hsf_retire_msect_audit.py

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


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


def _m_sector_parts(lower_diag: Dict[str, Any]) -> Dict[str, float]:
    odiff_before = _safe_float(lower_diag.get("Odiff_before_R"))
    odiff_after = _safe_float(lower_diag.get("Odiff_after_R"))

    role_weight_before = _mean(_extract_fp_weights(lower_diag, "role_fingerprints_before"))
    role_weight_after = _mean(_extract_fp_weights(lower_diag, "role_fingerprints_after"))

    act_before = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_before", "activity_sum"))
    act_after = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_after", "activity_sum"))

    cluster_before = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_before", "local_cluster"))
    cluster_after = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_after", "local_cluster"))

    incident_before = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_before", "incident_count"))
    incident_after = _mean(_extract_fp_metric(lower_diag, "role_fingerprints_after", "incident_count"))

    odiff_ratio = odiff_after / max(1e-12, odiff_before) if odiff_before > 0 else 1.0
    role_ratio = role_weight_after / max(1e-12, role_weight_before) if role_weight_before > 0 else 1.0
    activity_ratio = act_after / max(1e-12, act_before) if act_before > 0 else 1.0

    # Coherence proxy: if clustering drops less and incident support drops less, function is more preserved.
    cluster_retention = 1.0 - max(0.0, cluster_before - cluster_after)
    cluster_retention = max(0.0, min(1.0, cluster_retention))

    incident_retention = incident_after / max(1e-12, incident_before) if incident_before > 0 else 1.0
    incident_retention = max(0.0, min(1.0, incident_retention))

    # Weighted decomposition of the previously blended M-sector
    PM_odiff = 0.40 * max(0.0, min(1.0, odiff_ratio))
    PM_role = 0.20 * max(0.0, min(1.0, role_ratio))
    PM_act = 0.20 * max(0.0, min(1.0, activity_ratio))
    PM_clust = 0.10 * cluster_retention
    PM_inc = 0.10 * incident_retention
    P_M = float(min(1.0, PM_odiff + PM_role + PM_act + PM_clust + PM_inc))

    return {
        "odiff_before": float(odiff_before),
        "odiff_after": float(odiff_after),
        "odiff_ratio": float(odiff_ratio),
        "role_weight_before": float(role_weight_before),
        "role_weight_after": float(role_weight_after),
        "role_ratio": float(role_ratio),
        "activity_before": float(act_before),
        "activity_after": float(act_after),
        "activity_ratio": float(activity_ratio),
        "cluster_before": float(cluster_before),
        "cluster_after": float(cluster_after),
        "cluster_retention": float(cluster_retention),
        "incident_before": float(incident_before),
        "incident_after": float(incident_after),
        "incident_retention": float(incident_retention),
        "PM_odiff": float(PM_odiff),
        "PM_role": float(PM_role),
        "PM_act": float(PM_act),
        "PM_clust": float(PM_clust),
        "PM_inc": float(PM_inc),
        "P_M": float(P_M),
    }


def _row_from_snapshot(step: int, snap: Dict[str, Any], lower_diag: Dict[str, Any], winner_type: str, args: argparse.Namespace) -> Dict[str, Any]:
    node = _safe_int(lower_diag.get("node", lower_diag.get("move_object")))
    parts = _m_sector_parts(lower_diag)

    dE_expr = _safe_float(lower_diag.get("dE_expr"))
    dE_expr_raw = _safe_float(lower_diag.get("dE_expr_raw"))
    dCB = _safe_float(lower_diag.get("dCB"))
    dCS = _safe_float(lower_diag.get("dCS"))
    dCF = _safe_float(lower_diag.get("dCF"))
    W_NR = _safe_float(lower_diag.get("W_NR"))

    F_R = dE_expr - float(args.lambda_B) * dCB - float(args.lambda_R) * W_NR - float(args.lambda_S) * dCS - float(args.lambda_F) * dCF

    if F_R > float(args.win_threshold) and parts["P_M"] >= float(args.pm_min):
        case = "lawful_retire"
    elif F_R > float(args.win_threshold) and parts["P_M"] < float(args.pm_min):
        case = "FR_pos_PM_weak"
    elif F_R <= float(args.win_threshold) and parts["P_M"] >= float(args.pm_min):
        case = "PM_ok_expr_bad"
    else:
        case = "retire_unlawful"

    return {
        "step": int(step),
        "node": int(node),
        "winner_move_type": str(winner_type),
        "case": case,
        "F_R_retire": float(F_R),
        "retire_positive_FR": bool(F_R > float(args.win_threshold)),
        "PM_ok": bool(parts["P_M"] >= float(args.pm_min)),
        **parts,
        "dE_expr": float(dE_expr),
        "dE_expr_raw": float(dE_expr_raw),
        "delta_Odiff_R": _safe_float(lower_diag.get("delta_Odiff_R")),
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
        rows.append(_row_from_snapshot(step, snap, best_lower, str(winner.get("move_type")), args))
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in rows)),
        "case_counts": dict(Counter(r["case"] for r in rows)),
        "positive_FR_fraction": _mean([1.0 if r["retire_positive_FR"] else 0.0 for r in rows]),
        "PM_ok_fraction": _mean([1.0 if r["PM_ok"] else 0.0 for r in rows]),
        "FR_mean": _mean([r["F_R_retire"] for r in rows]),
        "P_M_mean": _mean([r["P_M"] for r in rows]),
        "PM_odiff_mean": _mean([r["PM_odiff"] for r in rows]),
        "PM_role_mean": _mean([r["PM_role"] for r in rows]),
        "PM_act_mean": _mean([r["PM_act"] for r in rows]),
        "PM_clust_mean": _mean([r["PM_clust"] for r in rows]),
        "PM_inc_mean": _mean([r["PM_inc"] for r in rows]),
    }


def _per_node_summary(rows: List[Dict[str, Any]], node_id: int) -> Dict[str, Any]:
    group = [r for r in rows if r["node"] == node_id]
    return {
        "n": len(group),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in group)),
        "case_counts": dict(Counter(r["case"] for r in group)),
        "positive_FR_fraction": _mean([1.0 if r["retire_positive_FR"] else 0.0 for r in group]),
        "PM_ok_fraction": _mean([1.0 if r["PM_ok"] else 0.0 for r in group]),
        "FR_mean": _mean([r["F_R_retire"] for r in group]),
        "P_M_mean": _mean([r["P_M"] for r in group]),
        "PM_odiff_mean": _mean([r["PM_odiff"] for r in group]),
        "PM_role_mean": _mean([r["PM_role"] for r in group]),
        "PM_act_mean": _mean([r["PM_act"] for r in group]),
        "PM_clust_mean": _mean([r["PM_clust"] for r in group]),
        "PM_inc_mean": _mean([r["PM_inc"] for r in group]),
        "dE_expr_mean": _mean([r["dE_expr"] for r in group]),
        "dE_expr_raw_mean": _mean([r["dE_expr_raw"] for r in group]),
    }


def _top_late_nodes(rows: List[Dict[str, Any]], k: int = 5) -> List[int]:
    ctr = Counter(r["node"] for r in rows)
    return [node for node, _ in ctr.most_common(k)]


def _readout(late_rows: List[Dict[str, Any]], node_summaries: Dict[str, Any]) -> List[str]:
    reads: List[str] = []
    if any(r["PM_ok"] for r in late_rows):
        reads.append("Some late demotion candidates retain enough M-sector support under the decomposed witness.")
    if any(r["retire_positive_FR"] for r in late_rows):
        reads.append("Some late demotion candidates remain positive under the local retirement score.")
    if "2" in node_summaries and "6" in node_summaries:
        n2 = node_summaries["2"]
        n6 = node_summaries["6"]
        if n2["PM_odiff_mean"] > n6["PM_odiff_mean"]:
            reads.append("Node 2 is stronger than node 6 on Odiff retention inside the M-sector.")
        if n2["PM_act_mean"] > n6["PM_act_mean"]:
            reads.append("Node 2 is stronger than node 6 on retained committed activity.")
        if abs(n2["PM_clust_mean"] - n6["PM_clust_mean"]) < 0.03:
            reads.append("Local cluster-retention is not strongly separating node 2 from node 6.")
        if abs(n2["PM_inc_mean"] - n6["PM_inc_mean"]) < 0.03:
            reads.append("Incident-count retention is not strongly separating node 2 from node 6.")
        if n6["FR_mean"] < n2["FR_mean"]:
            reads.append("The late node-2 / node-6 split remains concentrated in the decomposed M-sector plus expression.")
    if not reads:
        reads.append("No single decomposed M-sector component dominates the late retirement split.")
    return reads


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["F_R_retire"], r["P_M"], r["PM_odiff"], r["PM_act"]), reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decomposed M-sector audit for retirement moves. "
            "Short filename for GitHub-friendly workflows."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")
    parser.add_argument("--lambda-B", type=float, default=0.18)
    parser.add_argument("--lambda-R", type=float, default=0.35)
    parser.add_argument("--lambda-S", type=float, default=0.12)
    parser.add_argument("--lambda-F", type=float, default=0.20)
    parser.add_argument("--win-threshold", type=float, default=0.0)
    parser.add_argument("--pm-min", type=float, default=0.60)
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_msect.json",
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
        "script": "hsf_retire_msect_audit.py",
        "input_json": str(in_path),
        "audit_config": {
            "lambda_B": float(args.lambda_B),
            "lambda_R": float(args.lambda_R),
            "lambda_S": float(args.lambda_S),
            "lambda_F": float(args.lambda_F),
            "win_threshold": float(args.win_threshold),
            "pm_min": float(args.pm_min),
        },
        "overall_summary": _summary(rows),
        "late_summary": _summary(late),
        "late_top_nodes": top_nodes,
        "late_node_summaries": node_summaries,
        "flagged_rows": _flagged_rows(rows),
        "readout": _readout(late, node_summaries),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_msect.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Retire M-Sector Audit ===")
    print(f"rows:                    {report['overall_summary']['n_rows']}")
    print(f"late rows:               {report['late_summary']['n_rows']}")
    print(f"late top nodes:          {report['late_top_nodes']}")
    print(f"late positive F_R frac:  {report['late_summary']['positive_FR_fraction']:.6f}")
    print(f"late P_M-ok frac:        {report['late_summary']['PM_ok_fraction']:.6f}")
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()