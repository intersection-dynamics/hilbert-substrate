#!/usr/bin/env python3
# filename: hsf_mesoscape_lower_support_competitiveness_audit.py

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


def _extract_core_pair(diag: Dict[str, Any]) -> List[int]:
    core_before = diag.get("core_before", {}) or {}
    cp = core_before.get("core_pair")
    if isinstance(cp, list) and len(cp) == 2:
        return [int(cp[0]), int(cp[1])]
    return []


def _classify_lower_support_case(diag: Dict[str, Any]) -> str:
    rr = ((diag.get("retirement_info") or {}) if isinstance(diag.get("retirement_info"), dict) else {})
    readiness = _safe_float(rr.get("retirement_readiness"))
    dF = _safe_float(diag.get("deltaF"))
    dE_raw = _safe_float(diag.get("dE_expr_raw"))
    dE = _safe_float(diag.get("dE_expr"))
    odiff = _safe_float(diag.get("delta_Odiff_R"))
    dCF = _safe_float(diag.get("dCF"))
    WNR = _safe_float(diag.get("W_NR"))

    core_before = set(_extract_core_pair(diag))
    node = _safe_int(diag.get("node", diag.get("move_object")))
    core_touch = node in core_before
    shell_pen = _safe_float(rr.get("shell_penalty"))
    core_pen = _safe_float(rr.get("core_penalty"))

    if dF > 0.0:
        return "positive_deltaF"
    if core_touch and core_pen > 0.0 and dF <= 0.0:
        return "loses_to_core_protection"
    if shell_pen > 0.0 and dF <= 0.0:
        return "loses_to_shell_protection"
    if dE <= 0.0 and dE_raw <= 0.0 and dF <= 0.0:
        return "loses_on_expression"
    if WNR > 1.0 and dF <= 0.0:
        return "loses_on_no_refolding"
    if dCF > 0.25 and dF <= 0.0:
        return "loses_on_functional_cost"
    if readiness < 0.0 and dF <= 0.0:
        return "negative_readiness_signal"
    if odiff <= 0.0 and dE <= 0.0 and dF <= 0.0:
        return "loses_without_role_gain"
    return "mixed_loss"


def _late_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    start = max(0, (2 * len(rows)) // 3)
    return rows[start:]


def _summarize_lower_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cases = Counter(_classify_lower_support_case(r) for r in rows)
    return {
        "n_rows": len(rows),
        "case_counts": dict(cases),
        "retirement_readiness_mean": _mean([_safe_float((r.get("retirement_info") or {}).get("retirement_readiness")) for r in rows]),
        "retirement_readiness_stdev": _stdev([_safe_float((r.get("retirement_info") or {}).get("retirement_readiness")) for r in rows]),
        "deltaF_mean": _mean([_safe_float(r.get("deltaF")) for r in rows]),
        "deltaF_stdev": _stdev([_safe_float(r.get("deltaF")) for r in rows]),
        "dE_expr_raw_mean": _mean([_safe_float(r.get("dE_expr_raw")) for r in rows]),
        "dE_expr_mean": _mean([_safe_float(r.get("dE_expr")) for r in rows]),
        "delta_Odiff_mean": _mean([_safe_float(r.get("delta_Odiff_R")) for r in rows]),
        "W_NR_mean": _mean([_safe_float(r.get("W_NR")) for r in rows]),
        "dCF_mean": _mean([_safe_float(r.get("dCF")) for r in rows]),
        "core_touch_fraction": _mean([
            1.0 if _safe_int(r.get("node", r.get("move_object"))) in set(_extract_core_pair(r)) else 0.0
            for r in rows
        ]),
        "positive_deltaF_fraction": _mean([
            1.0 if _safe_float(r.get("deltaF")) > 0.0 else 0.0
            for r in rows
        ]),
    }


def _build_report(data: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = data.get("snapshots", []) or []

    step_rows: List[Dict[str, Any]] = []
    lower_rows: List[Dict[str, Any]] = []
    losing_gap_rows: List[Dict[str, Any]] = []

    for snap in snapshots:
        step = _safe_int(snap.get("step"))
        winner = _winner_diag(snap)
        if winner is None:
            continue

        winner_type = str(winner.get("move_type"))
        winner_deltaF = _safe_float(winner.get("deltaF"))

        cands = snap.get("candidate_move_diagnostics", []) or []
        lowers = [d for d in cands if str(d.get("move_type")) == "lower_support"]
        if not lowers:
            continue

        lowers_sorted = sorted(lowers, key=lambda d: _safe_float(d.get("deltaF")), reverse=True)
        best_lower = lowers_sorted[0]

        rr = best_lower.get("retirement_info") or {}
        node = _safe_int(best_lower.get("node", best_lower.get("move_object")))

        row = {
            "step": int(step),
            "winner_move_type": winner_type,
            "winner_deltaF": float(winner_deltaF),
            "lower_node": int(node),
            "lower_deltaF": _safe_float(best_lower.get("deltaF")),
            "deltaF_gap_to_winner": float(winner_deltaF - _safe_float(best_lower.get("deltaF"))),
            "lower_dE_expr_raw": _safe_float(best_lower.get("dE_expr_raw")),
            "lower_dE_expr": _safe_float(best_lower.get("dE_expr")),
            "lower_delta_Odiff_R": _safe_float(best_lower.get("delta_Odiff_R")),
            "retirement_readiness": _safe_float(rr.get("retirement_readiness")),
            "edge_ready": _safe_float(rr.get("edge_ready")),
            "functional_ready": _safe_float(rr.get("functional_ready")),
            "bookkeeping_safety": _safe_float(rr.get("bookkeeping_safety")),
            "substitutability": _safe_float(rr.get("substitutability")),
            "core_penalty": _safe_float(rr.get("core_penalty")),
            "shell_penalty": _safe_float(rr.get("shell_penalty")),
            "W_NR": _safe_float(best_lower.get("W_NR")),
            "dCF": _safe_float(best_lower.get("dCF")),
            "dCB": _safe_float(best_lower.get("dCB")),
            "dCS": _safe_float(best_lower.get("dCS")),
            "case": _classify_lower_support_case(best_lower),
            "core_pair_before": _extract_core_pair(best_lower),
        }
        step_rows.append(row)

        lower_copy = dict(best_lower)
        lower_copy["step_hint"] = int(step)
        lower_rows.append(lower_copy)

        losing_gap_rows.append(
            {
                "step": int(step),
                "winner_move_type": winner_type,
                "winner_deltaF": float(winner_deltaF),
                "lower_node": int(node),
                "lower_deltaF": _safe_float(best_lower.get("deltaF")),
                "gap": float(winner_deltaF - _safe_float(best_lower.get("deltaF"))),
                "case": _classify_lower_support_case(best_lower),
            }
        )

    late_lower_rows = _late_rows(lower_rows)
    late_step_rows = _late_rows(step_rows)

    overall = _summarize_lower_rows(lower_rows)
    late = _summarize_lower_rows(late_lower_rows)

    winner_counts = Counter(r["winner_move_type"] for r in step_rows)
    case_counts = Counter(r["case"] for r in step_rows)
    late_winner_counts = Counter(r["winner_move_type"] for r in late_step_rows)
    late_case_counts = Counter(r["case"] for r in late_step_rows)

    report = {
        "summary": {
            "n_steps_with_lower_support_candidates": len(step_rows),
            "winner_move_type_counts_against_lower": dict(winner_counts),
            "lower_support_case_counts": dict(case_counts),
            "late_winner_move_type_counts_against_lower": dict(late_winner_counts),
            "late_lower_support_case_counts": dict(late_case_counts),
            "mean_deltaF_gap_to_winner": _mean([r["deltaF_gap_to_winner"] for r in step_rows]),
            "late_mean_deltaF_gap_to_winner": _mean([r["deltaF_gap_to_winner"] for r in late_step_rows]),
            "overall_lower_summary": overall,
            "late_lower_summary": late,
            "readout": _readout(step_rows, overall, late, case_counts, late_case_counts),
        },
        "step_rows": step_rows,
        "losing_gap_rows": losing_gap_rows,
        "flagged_rows": _flagged_rows(step_rows),
    }
    return report


def _readout(
    step_rows: List[Dict[str, Any]],
    overall: Dict[str, Any],
    late: Dict[str, Any],
    case_counts: Counter,
    late_case_counts: Counter,
) -> List[str]:
    reads: List[str] = []

    if overall["positive_deltaF_fraction"] > 0.0:
        reads.append("Some lower_support candidates are now genuinely lawful on total score, even before comparing to the winning move.")
    if case_counts.get("loses_on_expression", 0) > 0:
        reads.append("Some lower_support candidates still lose mainly because their expression contribution stays negative.")
    if case_counts.get("loses_on_no_refolding", 0) > 0:
        reads.append("Some lower_support candidates still lose mainly on no-refolding burden.")
    if case_counts.get("loses_on_core_protection", 0) > 0 or case_counts.get("loses_to_core_protection", 0) > 0:
        reads.append("Core protection remains a real blocker for some demotion candidates.")
    if case_counts.get("loses_to_shell_protection", 0) > 0:
        reads.append("Shell protection remains a real blocker for some demotion candidates.")
    if late["deltaF_mean"] > 0.0:
        reads.append("In the late regime, lower_support is positive on average and is genuinely competitive.")
    elif late["deltaF_mean"] <= 0.0:
        reads.append("In the late regime, lower_support is still negative on average and only wins selectively.")
    if late_case_counts.get("positive_deltaF", 0) > 0:
        reads.append("Late-time demotion is a real part of the move economy now.")
    if not reads:
        reads.append("No single lawful term dominates; lower_support outcomes are mixed.")
    return reads


def _flagged_rows(step_rows: List[Dict[str, Any]], limit: int = 40) -> List[Dict[str, Any]]:
    scored = []
    for r in step_rows:
        score = (
            2.0 * max(0.0, r["lower_deltaF"])
            - max(0.0, r["deltaF_gap_to_winner"])
            - 0.5 * max(0.0, r["W_NR"] - 1.0)
        )
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def print_summary(report: Dict[str, Any]) -> None:
    s = report["summary"]
    o = s["overall_lower_summary"]
    l = s["late_lower_summary"]

    print("=== Lower-Support Competitiveness Audit (score-driven) ===")
    print(f"steps with lower_support candidates:   {s['n_steps_with_lower_support_candidates']}")
    print(f"winner counts vs lower_support:        {s['winner_move_type_counts_against_lower']}")
    print(f"lower_support cases:                   {s['lower_support_case_counts']}")
    print(f"late lower_support cases:              {s['late_lower_support_case_counts']}")
    print()

    print(f"mean gap to winner:                    {s['mean_deltaF_gap_to_winner']:.6f}")
    print(f"late mean gap to winner:               {s['late_mean_deltaF_gap_to_winner']:.6f}")
    print()

    print(f"overall readiness mean:                {o['retirement_readiness_mean']:.6f}")
    print(f"overall positive deltaF fraction:      {o['positive_deltaF_fraction']:.6f}")
    print(f"overall deltaF mean:                   {o['deltaF_mean']:.6f}")
    print(f"overall dE_expr_raw mean:              {o['dE_expr_raw_mean']:.6f}")
    print(f"overall dE_expr mean:                  {o['dE_expr_mean']:.6f}")
    print(f"overall delta_Odiff mean:              {o['delta_Odiff_mean']:.6f}")
    print(f"overall W_NR mean:                     {o['W_NR_mean']:.6f}")
    print()

    print(f"late readiness mean:                   {l['retirement_readiness_mean']:.6f}")
    print(f"late positive deltaF fraction:         {l['positive_deltaF_fraction']:.6f}")
    print(f"late deltaF mean:                      {l['deltaF_mean']:.6f}")
    print(f"late dE_expr_raw mean:                 {l['dE_expr_raw_mean']:.6f}")
    print(f"late dE_expr mean:                     {l['dE_expr_mean']:.6f}")
    print(f"late delta_Odiff mean:                 {l['delta_Odiff_mean']:.6f}")
    print(f"late W_NR mean:                        {l['W_NR_mean']:.6f}")
    print()

    print("Readout")
    for line in s["readout"]:
        print(f"  - {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Focused diagnostic for lower_support under the score-driven retirement regime. "
            "Compares the best lower_support candidate each step against the winning move without "
            "treating retirement eligibility as an admissibility gate."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_lower_support_competitiveness_audit.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = _load_json(in_path)
    report = {
        "script": "hsf_mesoscape_lower_support_competitiveness_audit.py",
        "input_json": str(in_path),
        **_build_report(data),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_lower_support_competitiveness_audit.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()