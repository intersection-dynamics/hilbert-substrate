#!/usr/bin/env python3
# filename: hsf_mesoscape_local_expression_alignment_audit.py

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
        row["step_hint"] = _safe_int(snap.get("step"))
        rows.append(row)
    return rows


def _extract_core_set(row: Dict[str, Any]) -> set[int]:
    core_before = row.get("core_before", {}) or {}
    cp = core_before.get("core_pair")
    if isinstance(cp, list) and len(cp) == 2:
        return {int(cp[0]), int(cp[1])}
    return set()


def _touches_core(row: Dict[str, Any]) -> bool:
    core = _extract_core_set(row)
    if not core:
        return False
    mtype = str(row.get("move_type"))
    if mtype == "raise_support":
        parents = row.get("parents")
        child = _safe_int(row.get("child"))
        touched = set()
        if isinstance(parents, list):
            touched.update(int(x) for x in parents)
        if child >= 0:
            touched.add(child)
        return bool(touched & core)
    if mtype in ("edge_up", "edge_down"):
        edge = row.get("edge")
        if isinstance(edge, list) and len(edge) == 2:
            return int(edge[0]) in core or int(edge[1]) in core
    if mtype == "lower_support":
        node = _safe_int(row.get("node"))
        return node in core
    return False


def _support_mass_delta(row: Dict[str, Any]) -> float:
    sigma_before = row.get("sigma_before")
    sigma_after = row.get("sigma_after")
    if not isinstance(sigma_before, list) or not isinstance(sigma_after, list):
        return 0.0
    n = min(len(sigma_before), len(sigma_after))
    return float(sum(_safe_float(sigma_after[i]) - _safe_float(sigma_before[i]) for i in range(n)))


def _interface_count_delta(row: Dict[str, Any]) -> float:
    mtype = str(row.get("move_type"))
    if mtype == "edge_up":
        wb = _safe_float(row.get("w_before"))
        wa = _safe_float(row.get("w_after"))
        return 1.0 if wb <= 1e-12 and wa > 1e-12 else 0.0
    if mtype == "edge_down":
        wb = _safe_float(row.get("w_before"))
        wa = _safe_float(row.get("w_after"))
        return -1.0 if wb > 1e-12 and wa <= 1e-12 else 0.0
    if mtype == "raise_support":
        # two half-edges may be raised, but not necessarily from zero to active
        return 0.0
    return 0.0


def _role_gain_proxy(row: Dict[str, Any]) -> float:
    mtype = str(row.get("move_type"))
    if mtype == "raise_support":
        return max(
            0.0,
            _safe_float(row.get("delta_Odiff_R")),
            _safe_float(row.get("birth_parent_relief")),
            _safe_float(row.get("birth_distinctness")),
        )
    if mtype == "edge_up":
        return max(
            0.0,
            _safe_float(row.get("delta_Odiff_R")),
            _safe_float(row.get("edge_up_relief_gain")),
            _safe_float(row.get("edge_up_distinct_gain")),
        )
    if mtype == "edge_down":
        return max(
            0.0,
            _safe_float(row.get("delta_Odiff_R")),
            _safe_float(row.get("graded_structural_adjustment")),
            _safe_float(row.get("graded_distinct_term")),
        )
    if mtype == "lower_support":
        return max(
            0.0,
            _safe_float(row.get("delta_Odiff_R")),
            _safe_float(row.get("graded_structural_adjustment")),
        )
    return 0.0


def _misalignment_case(row: Dict[str, Any]) -> str:
    raw = _safe_float(row.get("dE_expr_raw"))
    role = _role_gain_proxy(row)
    support_delta = _support_mass_delta(row)
    interface_delta = _interface_count_delta(row)
    core_touch = _touches_core(row)

    if raw < 0.0 and role > 0.0 and support_delta > 0.0:
        return "support_growth_penalized_despite_role_gain"
    if raw < 0.0 and role > 0.0 and interface_delta > 0.0:
        return "interface_growth_penalized_despite_role_gain"
    if raw < 0.0 and role > 0.0 and core_touch:
        return "core_touch_penalized_despite_role_gain"
    if raw < 0.0 and role <= 0.0 and support_delta > 0.0:
        return "support_growth_penalized_without_role_gain"
    if raw < 0.0 and role <= 0.0 and interface_delta > 0.0:
        return "interface_growth_penalized_without_role_gain"
    if raw >= 0.0 and role <= 0.0 and interface_delta > 0.0:
        return "raw_positive_interface_growth_without_role_gain"
    return "ordinary"


def _build_type_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for mtype in sorted(set(str(r.get("move_type")) for r in rows)):
        group = [r for r in rows if str(r.get("move_type")) == mtype]
        out[mtype] = {
            "n": len(group),
            "dE_expr_raw_mean": _mean([_safe_float(r.get("dE_expr_raw")) for r in group]),
            "dE_expr_mean": _mean([_safe_float(r.get("dE_expr")) for r in group]),
            "delta_Odiff_mean": _mean([_safe_float(r.get("delta_Odiff_R")) for r in group]),
            "support_mass_delta_mean": _mean([_support_mass_delta(r) for r in group]),
            "interface_count_delta_mean": _mean([_interface_count_delta(r) for r in group]),
            "role_gain_proxy_mean": _mean([_role_gain_proxy(r) for r in group]),
            "touches_core_fraction": _mean([1.0 if _touches_core(r) else 0.0 for r in group]),
        }
    return out


def _build_case_summary(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    ctr = Counter(_misalignment_case(r) for r in rows)
    return dict(ctr)


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 50) -> List[Dict[str, Any]]:
    flagged: List[Dict[str, Any]] = []
    for r in rows:
        case = _misalignment_case(r)
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
                "role_gain_proxy": _role_gain_proxy(r),
                "support_mass_delta": _support_mass_delta(r),
                "interface_count_delta": _interface_count_delta(r),
                "touches_core": _touches_core(r),
                "move_descriptor": {
                    "parents": r.get("parents"),
                    "child": r.get("child"),
                    "edge": r.get("edge"),
                    "node": r.get("node"),
                },
            }
        )
    flagged.sort(key=lambda x: (x["step"], x["move_type"]))
    return flagged[:limit]


def _correlation_summary(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    # Lightweight signed co-movements via mean products; avoids scipy dependency.
    raws = [_safe_float(r.get("dE_expr_raw")) for r in rows]
    odiffs = [_safe_float(r.get("delta_Odiff_R")) for r in rows]
    role = [_role_gain_proxy(r) for r in rows]
    support = [_support_mass_delta(r) for r in rows]
    iface = [_interface_count_delta(r) for r in rows]

    def cov_like(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        ma, mb = _mean(a), _mean(b)
        return _mean([(x - ma) * (y - mb) for x, y in zip(a, b)])

    return {
        "cov_raw_vs_odiff": cov_like(raws, odiffs),
        "cov_raw_vs_role_gain_proxy": cov_like(raws, role),
        "cov_raw_vs_support_mass_delta": cov_like(raws, support),
        "cov_raw_vs_interface_count_delta": cov_like(raws, iface),
    }


def _late_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    start = max(0, (2 * len(rows)) // 3)
    return rows[start:]


def _readout(all_rows: List[Dict[str, Any]], late_rows: List[Dict[str, Any]], case_counts: Dict[str, int], late_case_counts: Dict[str, int]) -> List[str]:
    reads: List[str] = []

    if case_counts.get("support_growth_penalized_despite_role_gain", 0) > 0:
        reads.append("Raw expression is still penalizing some support growth even when move-local role evidence is positive.")

    if case_counts.get("interface_growth_penalized_despite_role_gain", 0) > 0:
        reads.append("Raw expression is still penalizing some interface growth even when move-local role evidence is positive.")

    if late_case_counts.get("interface_growth_penalized_despite_role_gain", 0) > 0:
        reads.append("The late bottleneck still includes edge/interface additions that look role-positive locally but raw-negative globally.")

    if late_case_counts.get("raw_positive_interface_growth_without_role_gain", 0) == 0:
        reads.append("The old false-positive edge_up pattern appears to remain suppressed in the late regime.")

    if not reads:
        reads.append("No single raw-expression misalignment pattern dominates this accepted-move set.")
    return reads


def _build_report(data: Dict[str, Any]) -> Dict[str, Any]:
    rows = _accepted_rows(data)
    late = _late_rows(rows)

    all_type_summary = _build_type_summary(rows)
    late_type_summary = _build_type_summary(late)

    all_case_summary = _build_case_summary(rows)
    late_case_summary = _build_case_summary(late)

    report = {
        "accepted_summary": {
            "n_rows": len(rows),
            "move_type_summary": all_type_summary,
            "case_summary": all_case_summary,
            "correlation_summary": _correlation_summary(rows),
        },
        "late_summary": {
            "n_rows": len(late),
            "move_type_summary": late_type_summary,
            "case_summary": late_case_summary,
            "correlation_summary": _correlation_summary(late),
        },
        "flagged_cases": _flagged_rows(rows),
        "readout": _readout(rows, late, all_case_summary, late_case_summary),
    }
    return report


def print_summary(report: Dict[str, Any]) -> None:
    a = report["accepted_summary"]
    l = report["late_summary"]

    print("=== Local Expression Alignment Audit ===")
    print(f"accepted rows:          {a['n_rows']}")
    print(f"late rows:              {l['n_rows']}")
    print(f"accepted case summary:  {a['case_summary']}")
    print(f"late case summary:      {l['case_summary']}")
    print()
    print("Accepted move type summary")
    for mtype, stats in a["move_type_summary"].items():
        print(
            f"  {mtype}: n={stats['n']} "
            f"raw={stats['dE_expr_raw_mean']:.4f} "
            f"expr={stats['dE_expr_mean']:.4f} "
            f"odiff={stats['delta_Odiff_mean']:.4f} "
            f"role={stats['role_gain_proxy_mean']:.4f} "
            f"support_d={stats['support_mass_delta_mean']:.4f} "
            f"iface_d={stats['interface_count_delta_mean']:.4f} "
            f"core_frac={stats['touches_core_fraction']:.3f}"
        )
    print()
    print("Late move type summary")
    for mtype, stats in l["move_type_summary"].items():
        print(
            f"  {mtype}: n={stats['n']} "
            f"raw={stats['dE_expr_raw_mean']:.4f} "
            f"expr={stats['dE_expr_mean']:.4f} "
            f"odiff={stats['delta_Odiff_mean']:.4f} "
            f"role={stats['role_gain_proxy_mean']:.4f}"
        )
    print()
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the base local_expression layer is misaligned with graded move-local role evidence."
        )
    )
    parser.add_argument("json_path", help="Path to hsf_mesoscape_*_graded_support*.json")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_local_expression_alignment_audit.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = _load_json(in_path)
    audit = _build_report(data)

    report = {
        "script": "hsf_mesoscape_local_expression_alignment_audit.py",
        "input_json": str(in_path),
        **audit,
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_local_expression_alignment_audit.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()