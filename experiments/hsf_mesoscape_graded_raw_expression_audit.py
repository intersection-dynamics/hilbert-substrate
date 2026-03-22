#!/usr/bin/env python3
# filename: hsf_mesoscape_graded_raw_expression_audit.py

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
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def _sorted_edge(edge: Sequence[int]) -> Tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
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


def _accepted_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snap in data.get("snapshots", []) or []:
        winner = _winner_from_snapshot(snap)
        if winner is None:
            continue
        step = _safe_int(snap.get("step"))
        cands = snap.get("candidate_move_diagnostics", []) or []
        chosen = None
        for d in cands:
            if str(d.get("move_type")) == winner:
                chosen = d
                break
        if chosen is None and cands:
            chosen = max(cands, key=lambda d: _safe_float(d.get("deltaF")))
        if chosen is None:
            continue
        row = dict(chosen)
        row["step_hint"] = step
        rows.append(row)
    return rows


def _classify_case(row: Dict[str, Any]) -> str:
    raw = _safe_float(row.get("dE_expr_raw"))
    odiff = _safe_float(row.get("delta_Odiff_R"))
    relief = _safe_float(row.get("edge_up_relief_gain", row.get("birth_parent_relief", 0.0)))
    distinct = _safe_float(row.get("edge_up_distinct_gain", row.get("birth_distinctness", 0.0)))
    red_pen = _safe_float(row.get("graded_redundancy_penalty"))
    mtype = str(row.get("move_type"))

    if raw > 0.0 and odiff <= 0.0 and relief <= 0.0 and distinct <= 0.0:
        return "positive_raw_without_role_gain"
    if raw <= 0.0 and odiff > 0.0:
        return "negative_raw_with_positive_role_gain"
    if mtype == "edge_up" and raw > 0.0 and odiff <= 0.0:
        return "edge_up_raw_positive_role_weak"
    if red_pen < 0.0 and raw > 0.0:
        return "raw_overcomes_redundancy_penalty"
    return "ordinary"


def _late_start_index(n: int) -> int:
    return max(0, (2 * n) // 3)


def _summarize_rows(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    type_counts = Counter(str(r.get("move_type")) for r in rows)
    case_counts = Counter(_classify_case(r) for r in rows)

    return {
        "label": label,
        "n_rows": len(rows),
        "move_type_counts": dict(type_counts),
        "case_counts": dict(case_counts),
        "dE_expr_raw_mean": _mean([_safe_float(r.get("dE_expr_raw")) for r in rows]),
        "dE_expr_mean": _mean([_safe_float(r.get("dE_expr")) for r in rows]),
        "delta_Odiff_mean": _mean([_safe_float(r.get("delta_Odiff_R")) for r in rows]),
        "edge_up_relief_gain_mean": _mean([_safe_float(r.get("edge_up_relief_gain")) for r in rows if str(r.get("move_type")) == "edge_up"]),
        "edge_up_distinct_gain_mean": _mean([_safe_float(r.get("edge_up_distinct_gain")) for r in rows if str(r.get("move_type")) == "edge_up"]),
        "graded_witness_adjustment_mean": _mean([_safe_float(r.get("graded_witness_adjustment")) for r in rows]),
        "graded_redundancy_penalty_mean": _mean([_safe_float(r.get("graded_redundancy_penalty")) for r in rows]),
    }


def _build_component_attribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    component_keys = [
        "dE_expr_raw",
        "expr_odiff_adjustment",
        "graded_relief_term",
        "graded_distinct_term",
        "graded_persistence_term",
        "graded_structural_adjustment",
        "graded_redundancy_penalty",
        "graded_witness_adjustment",
        "dE_expr",
    ]

    for r in rows:
        mtype = str(r.get("move_type"))
        for key in component_keys:
            if key in r:
                by_type[mtype][key].append(_safe_float(r.get(key)))

    out: Dict[str, Any] = {}
    for mtype, comps in by_type.items():
        out[mtype] = {}
        for key, vals in comps.items():
            out[mtype][key] = {
                "mean": _mean(vals),
                "stdev": _stdev(vals),
                "min": min(vals) if vals else 0.0,
                "max": max(vals) if vals else 0.0,
            }
    return out


def _flag_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flagged: List[Dict[str, Any]] = []
    for r in rows:
        case = _classify_case(r)
        if case == "ordinary":
            continue
        flagged.append(
            {
                "step": _safe_int(r.get("step_hint")),
                "move_type": str(r.get("move_type")),
                "case": case,
                "deltaF": _safe_float(r.get("deltaF")),
                "dE_expr_raw": _safe_float(r.get("dE_expr_raw")),
                "dE_expr": _safe_float(r.get("dE_expr")),
                "delta_Odiff_R": _safe_float(r.get("delta_Odiff_R")),
                "edge_up_relief_gain": _safe_float(r.get("edge_up_relief_gain")),
                "edge_up_distinct_gain": _safe_float(r.get("edge_up_distinct_gain")),
                "graded_redundancy_penalty": _safe_float(r.get("graded_redundancy_penalty")),
                "graded_witness_adjustment": _safe_float(r.get("graded_witness_adjustment")),
                "expr_odiff_adjustment": _safe_float(r.get("expr_odiff_adjustment")),
                "move_object": r.get("edge", r.get("node", {"parents": r.get("parents"), "child": r.get("child")})),
            }
        )
    flagged.sort(key=lambda x: (x["step"], x["move_type"]))
    return flagged


def _cross_checks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    edge_up_rows = [r for r in rows if str(r.get("move_type")) == "edge_up"]
    raise_rows = [r for r in rows if str(r.get("move_type")) == "raise_support"]

    def count(pred_rows, pred):
        return sum(1 for r in pred_rows if pred(r))

    return {
        "edge_up_positive_raw_nonpositive_odiff": count(
            edge_up_rows,
            lambda r: _safe_float(r.get("dE_expr_raw")) > 0.0 and _safe_float(r.get("delta_Odiff_R")) <= 0.0,
        ),
        "edge_up_negative_raw_positive_total": count(
            edge_up_rows,
            lambda r: _safe_float(r.get("dE_expr_raw")) < 0.0 and _safe_float(r.get("dE_expr")) > 0.0,
        ),
        "edge_up_positive_raw_zero_relief_zero_distinct": count(
            edge_up_rows,
            lambda r: _safe_float(r.get("dE_expr_raw")) > 0.0
            and _safe_float(r.get("edge_up_relief_gain")) <= 0.0
            and _safe_float(r.get("edge_up_distinct_gain")) <= 0.0,
        ),
        "raise_support_positive_raw_nonpositive_odiff": count(
            raise_rows,
            lambda r: _safe_float(r.get("dE_expr_raw")) > 0.0 and _safe_float(r.get("delta_Odiff_R")) <= 0.0,
        ),
    }


def _readout(full_summary: Dict[str, Any], late_summary: Dict[str, Any], cross: Dict[str, Any]) -> List[str]:
    reads: List[str] = []

    if _safe_int(cross.get("edge_up_positive_raw_nonpositive_odiff")) > 0:
        reads.append(
            "Some accepted edge_up moves still have positive raw expression despite nonpositive differentiated-role gain."
        )

    if _safe_int(cross.get("edge_up_positive_raw_zero_relief_zero_distinct")) > 0:
        reads.append(
            "Some accepted edge_up moves still look positive at the raw layer even without measurable relief or distinctness gain."
        )

    if _safe_int(cross.get("edge_up_negative_raw_positive_total")) == 0:
        reads.append(
            "The hardening appears to have mostly removed witness-side rescue of raw-negative edge_up moves."
        )

    late_cases = late_summary.get("case_counts", {})
    if _safe_int(late_cases.get("positive_raw_without_role_gain")) > 0:
        reads.append(
            "The remaining late failure appears to live in raw local expression itself rather than in witness correction."
        )

    if not reads:
        reads.append("No obvious raw-expression anomaly dominates the accepted-move set.")
    return reads


def _build_report(data: Dict[str, Any]) -> Dict[str, Any]:
    rows = _accepted_rows(data)
    late_rows = rows[_late_start_index(len(rows)):] if rows else []

    full_summary = _summarize_rows(rows, "all_accepted_moves")
    late_summary = _summarize_rows(late_rows, "late_accepted_moves")
    component_attr = _build_component_attribution(rows)
    late_component_attr = _build_component_attribution(late_rows)
    flagged = _flag_rows(rows)
    cross = _cross_checks(rows)
    readout = _readout(full_summary, late_summary, cross)

    return {
        "accepted_move_summary": full_summary,
        "late_move_summary": late_summary,
        "component_attribution": component_attr,
        "late_component_attribution": late_component_attr,
        "cross_checks": cross,
        "flagged_cases": flagged[:50],
        "readout": readout,
    }


def print_summary(report: Dict[str, Any]) -> None:
    a = report["accepted_move_summary"]
    l = report["late_move_summary"]
    c = report["cross_checks"]

    print("=== Graded Raw Expression Audit ===")
    print(f"accepted moves:                            {a['n_rows']}")
    print(f"late accepted moves:                       {l['n_rows']}")
    print(f"accepted move types:                       {a['move_type_counts']}")
    print(f"late move types:                           {l['move_type_counts']}")
    print()
    print(f"mean dE_expr_raw (all):                    {a['dE_expr_raw_mean']:.6f}")
    print(f"mean dE_expr_raw (late):                   {l['dE_expr_raw_mean']:.6f}")
    print(f"mean delta_Odiff (all):                    {a['delta_Odiff_mean']:.6f}")
    print(f"mean delta_Odiff (late):                   {l['delta_Odiff_mean']:.6f}")
    print()
    print(f"edge_up positive raw / nonpositive odiff:  {c['edge_up_positive_raw_nonpositive_odiff']}")
    print(f"edge_up negative raw / positive total:     {c['edge_up_negative_raw_positive_total']}")
    print(f"edge_up positive raw / zero relief-dist:   {c['edge_up_positive_raw_zero_relief_zero_distinct']}")
    print(f"raise positive raw / nonpositive odiff:    {c['raise_support_positive_raw_nonpositive_odiff']}")
    print()
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit raw local expression in the graded-support sandbox, especially accepted cases "
            "where raw expression is positive despite weak or nonpositive differentiated-role gain."
        )
    )
    parser.add_argument("json_path", help="Path to hsf_mesoscape_*_graded_support*.json")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_raw_expression_audit.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = _load_json(in_path)
    audit = _build_report(data)

    report = {
        "script": "hsf_mesoscape_graded_raw_expression_audit.py",
        "input_json": str(in_path),
        **audit,
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_raw_expression_audit.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()