#!/usr/bin/env python3
# filename: hsf_retire_odiff_audit.py

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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


def _extract_fp(diag: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    fps = diag.get(key) or []
    if isinstance(fps, list):
        return [fp for fp in fps if isinstance(fp, dict)]
    return []


def _index_fp_by_site(fps: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for fp in fps:
        site = _safe_int(fp.get("site_id"))
        if site >= 0:
            out[site] = fp
    return out


def _metric_from_fp(fp: Dict[str, Any], metric: str) -> float:
    if metric == "weight":
        return _safe_float(fp.get("weight"))
    if metric == "novelty":
        return _safe_float(fp.get("novelty"))
    if metric == "relief":
        return _safe_float(fp.get("relief"))
    if metric == "distinctness":
        return _safe_float(fp.get("distinctness"))
    rm = fp.get("raw_metrics") or {}
    if isinstance(rm, dict):
        return _safe_float(rm.get(metric))
    return 0.0


def _site_delta(before_fp: Optional[Dict[str, Any]], after_fp: Optional[Dict[str, Any]]) -> Dict[str, float]:
    metrics = [
        "weight",
        "novelty",
        "relief",
        "distinctness",
        "incident_count",
        "mean_strength",
        "activity_sum",
        "local_cluster",
        "sibling_count_norm",
    ]
    out: Dict[str, float] = {}
    b = before_fp or {}
    a = after_fp or {}
    for m in metrics:
        out[f"{m}_before"] = _metric_from_fp(b, m)
        out[f"{m}_after"] = _metric_from_fp(a, m)
        out[f"{m}_delta"] = out[f"{m}_after"] - out[f"{m}_before"]
    return out


def _classify_row(row: Dict[str, Any]) -> str:
    if row["lower_deltaF"] > 0.0 and row["odiff_ratio"] >= 0.98:
        return "lawful_high_retention"
    if row["lower_deltaF"] > 0.0 and row["odiff_ratio"] < 0.98:
        return "lawful_partial_retention"
    if row["lower_deltaF"] <= 0.0 and row["odiff_ratio"] >= 0.98:
        return "odiff_retained_expr_bad"
    return "odiff_loss"


def _build_row(step: int, lower_diag: Dict[str, Any], winner_type: str) -> Dict[str, Any]:
    node = _safe_int(lower_diag.get("node", lower_diag.get("move_object")))

    fps_before = _extract_fp(lower_diag, "role_fingerprints_before")
    fps_after = _extract_fp(lower_diag, "role_fingerprints_after")

    idx_before = _index_fp_by_site(fps_before)
    idx_after = _index_fp_by_site(fps_after)

    all_sites = sorted(set(idx_before) | set(idx_after))
    site_rows: List[Dict[str, Any]] = []
    for site in all_sites:
        delta = _site_delta(idx_before.get(site), idx_after.get(site))
        site_rows.append({
            "site_id": int(site),
            **delta,
        })

    site_rows_sorted = sorted(site_rows, key=lambda r: r["weight_delta"])
    most_negative = site_rows_sorted[:10]
    most_positive = list(reversed(sorted(site_rows, key=lambda r: r["weight_delta"])))[:10]

    odiff_before = _safe_float(lower_diag.get("Odiff_before_R"))
    odiff_after = _safe_float(lower_diag.get("Odiff_after_R"))
    odiff_ratio = odiff_after / max(1e-12, odiff_before) if odiff_before > 0 else 1.0

    row = {
        "step": int(step),
        "node": int(node),
        "winner_move_type": str(winner_type),
        "lower_deltaF": _safe_float(lower_diag.get("deltaF")),
        "dE_expr_raw": _safe_float(lower_diag.get("dE_expr_raw")),
        "dE_expr": _safe_float(lower_diag.get("dE_expr")),
        "dE_expr_structural_adjustment": _safe_float(
            lower_diag.get("dE_expr_structural_adjustment", lower_diag.get("expr_adjustment"))
        ),
        "delta_Odiff_R": _safe_float(lower_diag.get("delta_Odiff_R")),
        "Odiff_before_R": float(odiff_before),
        "Odiff_after_R": float(odiff_after),
        "odiff_ratio": float(odiff_ratio),
        "weight_before_mean": _mean([_metric_from_fp(fp, "weight") for fp in fps_before]),
        "weight_after_mean": _mean([_metric_from_fp(fp, "weight") for fp in fps_after]),
        "novelty_before_mean": _mean([_metric_from_fp(fp, "novelty") for fp in fps_before]),
        "novelty_after_mean": _mean([_metric_from_fp(fp, "novelty") for fp in fps_after]),
        "relief_before_mean": _mean([_metric_from_fp(fp, "relief") for fp in fps_before]),
        "relief_after_mean": _mean([_metric_from_fp(fp, "relief") for fp in fps_after]),
        "distinct_before_mean": _mean([_metric_from_fp(fp, "distinctness") for fp in fps_before]),
        "distinct_after_mean": _mean([_metric_from_fp(fp, "distinctness") for fp in fps_after]),
        "incident_before_mean": _mean([_metric_from_fp(fp, "incident_count") for fp in fps_before]),
        "incident_after_mean": _mean([_metric_from_fp(fp, "incident_count") for fp in fps_after]),
        "activity_before_mean": _mean([_metric_from_fp(fp, "activity_sum") for fp in fps_before]),
        "activity_after_mean": _mean([_metric_from_fp(fp, "activity_sum") for fp in fps_after]),
        "cluster_before_mean": _mean([_metric_from_fp(fp, "local_cluster") for fp in fps_before]),
        "cluster_after_mean": _mean([_metric_from_fp(fp, "local_cluster") for fp in fps_after]),
        "site_rows": site_rows,
        "most_negative_sites": most_negative,
        "most_positive_sites": most_positive,
    }
    row["case"] = _classify_row(row)
    return row


def _rows_from_run(data: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        rows.append(_build_row(step, best_lower, str(winner.get("move_type"))))
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in rows)),
        "case_counts": dict(Counter(r["case"] for r in rows)),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in rows]),
        "dE_expr_raw_mean": _mean([r["dE_expr_raw"] for r in rows]),
        "dE_expr_mean": _mean([r["dE_expr"] for r in rows]),
        "odiff_ratio_mean": _mean([r["odiff_ratio"] for r in rows]),
        "delta_Odiff_mean": _mean([r["delta_Odiff_R"] for r in rows]),
        "weight_before_mean": _mean([r["weight_before_mean"] for r in rows]),
        "weight_after_mean": _mean([r["weight_after_mean"] for r in rows]),
        "activity_before_mean": _mean([r["activity_before_mean"] for r in rows]),
        "activity_after_mean": _mean([r["activity_after_mean"] for r in rows]),
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
        "odiff_ratio_mean": _mean([r["odiff_ratio"] for r in group]),
        "delta_Odiff_mean": _mean([r["delta_Odiff_R"] for r in group]),
        "weight_before_mean": _mean([r["weight_before_mean"] for r in group]),
        "weight_after_mean": _mean([r["weight_after_mean"] for r in group]),
        "activity_before_mean": _mean([r["activity_before_mean"] for r in group]),
        "activity_after_mean": _mean([r["activity_after_mean"] for r in group]),
        "cluster_before_mean": _mean([r["cluster_before_mean"] for r in group]),
        "cluster_after_mean": _mean([r["cluster_after_mean"] for r in group]),
    }


def _top_late_nodes(rows: List[Dict[str, Any]], k: int = 5) -> List[int]:
    ctr = Counter(r["node"] for r in rows)
    return [node for node, _ in ctr.most_common(k)]


def _site_aggregate_for_node(rows: List[Dict[str, Any]], node_id: int) -> Dict[str, Dict[str, float]]:
    group = [r for r in rows if r["node"] == node_id]
    buckets: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for r in group:
        for sr in r["site_rows"]:
            buckets[_safe_int(sr.get("site_id"))].append(sr)

    out: Dict[str, Dict[str, float]] = {}
    for site, items in sorted(buckets.items()):
        out[str(site)] = {
            "n": len(items),
            "weight_delta_mean": _mean([_safe_float(x.get("weight_delta")) for x in items]),
            "novelty_delta_mean": _mean([_safe_float(x.get("novelty_delta")) for x in items]),
            "relief_delta_mean": _mean([_safe_float(x.get("relief_delta")) for x in items]),
            "distinctness_delta_mean": _mean([_safe_float(x.get("distinctness_delta")) for x in items]),
            "incident_delta_mean": _mean([_safe_float(x.get("incident_count_delta")) for x in items]),
            "activity_delta_mean": _mean([_safe_float(x.get("activity_sum_delta")) for x in items]),
            "cluster_delta_mean": _mean([_safe_float(x.get("local_cluster_delta")) for x in items]),
        }
    return out


def _readout(late_rows: List[Dict[str, Any]], node_summaries: Dict[str, Any]) -> List[str]:
    reads: List[str] = []
    if "2" in node_summaries and "6" in node_summaries:
        n2 = node_summaries["2"]
        n6 = node_summaries["6"]
        if n2["odiff_ratio_mean"] > n6["odiff_ratio_mean"]:
            reads.append("Node 2 retains more Odiff than node 6 after retirement.")
        if abs(n2["weight_after_mean"] - n6["weight_after_mean"]) < 0.05:
            reads.append("Mean role weight after retirement is not strongly separating node 2 from node 6.")
        if n2["activity_after_mean"] > n6["activity_after_mean"]:
            reads.append("Node 2 retains more activity-bearing role structure than node 6.")
        if abs(n2["cluster_after_mean"] - n6["cluster_after_mean"]) < 0.05:
            reads.append("Local clustering after retirement is not strongly separating node 2 from node 6.")
    if not reads:
        reads.append("No single Odiff-retention pattern dominates the late retirement split.")
    return reads


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["odiff_ratio"], r["lower_deltaF"], r["dE_expr"]), reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Odiff-retention audit for retirement moves. "
            "Short filename for GitHub-friendly workflows."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_odiff.json",
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
    late_site_aggregates = {str(node): _site_aggregate_for_node(late, node) for node in top_nodes}

    report = {
        "script": "hsf_retire_odiff_audit.py",
        "input_json": str(in_path),
        "overall_summary": _summary(rows),
        "late_summary": _summary(late),
        "late_top_nodes": top_nodes,
        "late_node_summaries": node_summaries,
        "late_site_aggregates": late_site_aggregates,
        "flagged_rows": _flagged_rows(rows),
        "readout": _readout(late, node_summaries),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_odiff.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Retire Odiff Audit ===")
    print(f"rows:                    {report['overall_summary']['n_rows']}")
    print(f"late rows:               {report['late_summary']['n_rows']}")
    print(f"late top nodes:          {report['late_top_nodes']}")
    print(f"late odiff ratio mean:   {report['late_summary']['odiff_ratio_mean']:.6f}")
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()