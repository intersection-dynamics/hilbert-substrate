#!/usr/bin/env python3
# filename: hsf_mesoscape_partial_mode_probe.py

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


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_delta(x: float, scale: float) -> float:
    if scale <= 1e-12:
        return 0.0
    return float(x / scale)


def _sorted_pair(x: Sequence[int]) -> Tuple[int, int]:
    a, b = int(x[0]), int(x[1])
    return (a, b) if a <= b else (b, a)


def _get_target_niche(niche_compare: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int]], int]:
    target = niche_compare.get("target_niche", {}) or {}
    pp = target.get("parent_pair")
    child = _safe_int(target.get("child"))
    if isinstance(pp, list) and len(pp) == 2:
        return _sorted_pair(pp), child
    return None, child


def _extract_cycle_roles(pair: Dict[str, Any], which: str) -> List[Dict[str, Any]]:
    if which not in ("birth_before", "birth_after", "retire_before", "retire_after"):
        raise ValueError(f"Unknown role block: {which}")

    if which.startswith("birth"):
        audit = (pair.get("birth", {}) or {}).get("odiff_audit", {}) or {}
    else:
        audit = (pair.get("retire", {}) or {}).get("odiff_audit", {}) or {}

    key = {
        "birth_before": "role_fingerprints_before",
        "birth_after": "role_fingerprints_after",
        "retire_before": "role_fingerprints_before",
        "retire_after": "role_fingerprints_after",
    }[which]

    fps = audit.get(key, []) or []
    out = []
    for fp in fps:
        out.append(
            {
                "site_id": _safe_int(fp.get("site_id")),
                "role_id": fp.get("role_id"),
                "parent_anchor": tuple(fp.get("parent_anchor", []) or []),
                "novelty": _safe_float(fp.get("novelty")),
                "relief": _safe_float(fp.get("relief")),
                "distinctness": _safe_float(fp.get("distinctness")),
                "weight": _safe_float(fp.get("weight")),
                "raw_metrics": fp.get("raw_metrics", {}) or {},
            }
        )
    return out


def _role_summary(fps: List[Dict[str, Any]]) -> Dict[str, float]:
    if not fps:
        return {
            "count": 0.0,
            "novelty_mean": 0.0,
            "relief_mean": 0.0,
            "distinctness_mean": 0.0,
            "weight_mean": 0.0,
            "weight_sum": 0.0,
            "incident_count_mean": 0.0,
            "mean_strength_mean": 0.0,
            "activity_sum_mean": 0.0,
            "rank_mean_mean": 0.0,
            "local_cluster_mean": 0.0,
            "sibling_count_norm_mean": 0.0,
            "anchor_diversity": 0.0,
        }

    incident = []
    mean_strength = []
    activity_sum = []
    rank_mean = []
    local_cluster = []
    sibling_norm = []
    anchors = Counter()

    for fp in fps:
        raw = fp.get("raw_metrics", {}) or {}
        incident.append(_safe_float(raw.get("incident_count")))
        mean_strength.append(_safe_float(raw.get("mean_strength")))
        activity_sum.append(_safe_float(raw.get("activity_sum")))
        rank_mean.append(_safe_float(raw.get("rank_mean")))
        local_cluster.append(_safe_float(raw.get("local_cluster")))
        sibling_norm.append(_safe_float(raw.get("sibling_count_norm")))
        anchors[tuple(fp.get("parent_anchor", ()))] += 1

    anchor_diversity = float(len(anchors) / max(1, len(fps)))

    return {
        "count": float(len(fps)),
        "novelty_mean": _mean([_safe_float(fp.get("novelty")) for fp in fps]),
        "relief_mean": _mean([_safe_float(fp.get("relief")) for fp in fps]),
        "distinctness_mean": _mean([_safe_float(fp.get("distinctness")) for fp in fps]),
        "weight_mean": _mean([_safe_float(fp.get("weight")) for fp in fps]),
        "weight_sum": float(sum(_safe_float(fp.get("weight")) for fp in fps)),
        "incident_count_mean": _mean(incident),
        "mean_strength_mean": _mean(mean_strength),
        "activity_sum_mean": _mean(activity_sum),
        "rank_mean_mean": _mean(rank_mean),
        "local_cluster_mean": _mean(local_cluster),
        "sibling_count_norm_mean": _mean(sibling_norm),
        "anchor_diversity": anchor_diversity,
    }


def _extract_cycle_metrics(pair: Dict[str, Any]) -> Dict[str, Any]:
    birth = pair.get("birth", {}) or {}
    retire = pair.get("retire", {}) or {}

    birth_before_roles = _extract_cycle_roles(pair, "birth_before")
    birth_after_roles = _extract_cycle_roles(pair, "birth_after")
    retire_before_roles = _extract_cycle_roles(pair, "retire_before")
    retire_after_roles = _extract_cycle_roles(pair, "retire_after")

    birth_before = _role_summary(birth_before_roles)
    birth_after = _role_summary(birth_after_roles)
    retire_before = _role_summary(retire_before_roles)
    retire_after = _role_summary(retire_after_roles)

    return {
        "birth_step": _safe_int(pair.get("birth_step")),
        "retire_step": _safe_int(pair.get("retire_step")),
        "duration_events": _safe_int(pair.get("duration_events")),
        "duration_steps": _safe_int(pair.get("duration_steps")),

        "birth_delta_Odiff": _safe_float(birth.get("delta_Odiff_R")),
        "birth_parent_relief": _safe_float(birth.get("birth_parent_relief")),
        "birth_novelty": _safe_float(birth.get("birth_novelty")),
        "birth_justification": _safe_float(birth.get("birth_justification")),
        "birth_dE_expr": _safe_float(birth.get("dE_expr")),

        "retire_delta_Odiff": _safe_float(retire.get("delta_Odiff_R")),
        "retire_redundancy_removed": _safe_float(retire.get("expr_redundancy_removed")),
        "retire_distinctness_gain": _safe_float(retire.get("expr_distinctness_gain")),
        "retire_adjustment": _safe_float(retire.get("expr_adjustment")),
        "retire_dE_expr": _safe_float(retire.get("dE_expr")),

        "birth_before_summary": birth_before,
        "birth_after_summary": birth_after,
        "retire_before_summary": retire_before,
        "retire_after_summary": retire_after,

        "deltas_absent_to_present": {
            "role_relief_mean": birth_after["relief_mean"] - birth_before["relief_mean"],
            "distinctness_mean": birth_after["distinctness_mean"] - birth_before["distinctness_mean"],
            "local_cluster_mean": birth_after["local_cluster_mean"] - birth_before["local_cluster_mean"],
            "activity_sum_mean": birth_after["activity_sum_mean"] - birth_before["activity_sum_mean"],
            "weight_sum": birth_after["weight_sum"] - birth_before["weight_sum"],
            "anchor_diversity": birth_after["anchor_diversity"] - birth_before["anchor_diversity"],
            "count": birth_after["count"] - birth_before["count"],
        },
    }


def _score_partial_relief(c: Dict[str, Any]) -> Tuple[float, List[str]]:
    ev = []

    s = 0.0
    if c["birth_parent_relief"] >= 0.75:
        s += 2.0
        ev.append("birth parent relief is high")
    if c["birth_delta_Odiff"] > 0.0:
        s += 1.0
        ev.append("birth raises local differentiated-role occupancy")
    if c["retire_redundancy_removed"] >= 0.50:
        s += 1.0
        ev.append("retirement removes substantial redundancy")
    if c["retire_distinctness_gain"] >= 0.25:
        s += 1.0
        ev.append("retirement restores distinctness")
    if c["deltas_absent_to_present"]["activity_sum_mean"] > 0.0:
        s += 0.75
        ev.append("present state raises local activity")
    if c["deltas_absent_to_present"]["role_relief_mean"] > 0.0:
        s += 0.75
        ev.append("present state raises role relief")
    if c["duration_events"] <= 3:
        s += 0.5
        ev.append("cycle is short, suggesting overcorrection around a real niche")
    return s, ev


def _score_shell_redistribution(c: Dict[str, Any]) -> Tuple[float, List[str]]:
    ev = []

    s = 0.0
    if c["deltas_absent_to_present"]["local_cluster_mean"] > 0.05:
        s += 1.5
        ev.append("present state increases local clustering")
    if c["retire_redundancy_removed"] >= 0.50:
        s += 1.0
        ev.append("retirement sees meaningful redundancy removal")
    if c["retire_distinctness_gain"] >= 0.25:
        s += 1.0
        ev.append("retirement sees distinctness recovery")
    if c["deltas_absent_to_present"]["count"] >= 1.0:
        s += 0.75
        ev.append("present state adds a full extra local role carrier")
    if c["birth_parent_relief"] >= 0.50 and c["retire_redundancy_removed"] >= 0.50:
        s += 0.75
        ev.append("birth relieves while retirement cleans structure")
    if c["deltas_absent_to_present"]["anchor_diversity"] <= 0.0:
        s += 0.5
        ev.append("added support does not diversify anchors much, consistent with shell overpacking")
    return s, ev


def _score_shared_support(c: Dict[str, Any]) -> Tuple[float, List[str]]:
    ev = []

    s = 0.0
    if c["birth_delta_Odiff"] > 0.0:
        s += 1.0
        ev.append("birth increases local role occupancy")
    if c["retire_delta_Odiff"] < 0.0:
        s += 1.0
        ev.append("retirement removes an overcomplete role pattern")
    if c["deltas_absent_to_present"]["anchor_diversity"] > 0.0:
        s += 1.0
        ev.append("present state broadens anchor diversity")
    if c["deltas_absent_to_present"]["distinctness_mean"] >= 0.0:
        s += 0.5
        ev.append("present state does not simply collapse distinctness immediately")
    if c["deltas_absent_to_present"]["activity_sum_mean"] > 0.0 and c["retire_redundancy_removed"] > 0.0:
        s += 0.75
        ev.append("extra activity arrives but later looks partially redistributable")
    if c["birth_parent_relief"] >= 0.50 and c["deltas_absent_to_present"]["count"] >= 1.0:
        s += 0.5
        ev.append("a new role carrier helps, suggesting demand may want distribution rather than absence")
    return s, ev


def _classify_cycle(c: Dict[str, Any]) -> Dict[str, Any]:
    pr_score, pr_ev = _score_partial_relief(c)
    sh_score, sh_ev = _score_shell_redistribution(c)
    ss_score, ss_ev = _score_shared_support(c)

    score_map = {
        "partial_relief_mode": pr_score,
        "shell_redistribution_mode": sh_score,
        "shared_support_mode": ss_score,
    }
    best_label = max(score_map.items(), key=lambda kv: kv[1])[0]
    evidence_map = {
        "partial_relief_mode": pr_ev,
        "shell_redistribution_mode": sh_ev,
        "shared_support_mode": ss_ev,
    }

    return {
        "label": best_label,
        "scores": score_map,
        "evidence": evidence_map[best_label],
    }


def _summarize_cycles(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counter = Counter()
    score_buckets = defaultdict(list)
    examples = []

    relief_delta = []
    activity_delta = []
    cluster_delta = []
    distinct_delta = []
    anchor_delta = []

    for c in cycles:
        cls = _classify_cycle(c)
        c["classification"] = cls
        label_counter[cls["label"]] += 1
        for k, v in cls["scores"].items():
            score_buckets[k].append(float(v))

        d = c["deltas_absent_to_present"]
        relief_delta.append(_safe_float(d["role_relief_mean"]))
        activity_delta.append(_safe_float(d["activity_sum_mean"]))
        cluster_delta.append(_safe_float(d["local_cluster_mean"]))
        distinct_delta.append(_safe_float(d["distinctness_mean"]))
        anchor_delta.append(_safe_float(d["anchor_diversity"]))

        if len(examples) < 5:
            examples.append(
                {
                    "birth_step": c["birth_step"],
                    "retire_step": c["retire_step"],
                    "duration_events": c["duration_events"],
                    "duration_steps": c["duration_steps"],
                    "label": cls["label"],
                    "scores": cls["scores"],
                    "evidence": cls["evidence"],
                    "birth_parent_relief": c["birth_parent_relief"],
                    "birth_delta_Odiff": c["birth_delta_Odiff"],
                    "retire_redundancy_removed": c["retire_redundancy_removed"],
                    "retire_distinctness_gain": c["retire_distinctness_gain"],
                    "deltas_absent_to_present": c["deltas_absent_to_present"],
                }
            )

    dominant = label_counter.most_common(1)[0][0] if label_counter else "none"

    return {
        "n_cycles": len(cycles),
        "dominant_mode_hypothesis": dominant,
        "mode_hypothesis_counts": dict(label_counter),
        "mode_score_means": {k: _mean(v) for k, v in score_buckets.items()},
        "mean_absent_to_present_deltas": {
            "role_relief_mean": _mean(relief_delta),
            "activity_sum_mean": _mean(activity_delta),
            "local_cluster_mean": _mean(cluster_delta),
            "distinctness_mean": _mean(distinct_delta),
            "anchor_diversity": _mean(anchor_delta),
        },
        "example_cycles": examples,
    }


def _extract_cycles_from_niche_compare(niche_compare: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs = niche_compare.get("paired_cycles", []) or []
    out = []
    for pair in pairs:
        out.append(_extract_cycle_metrics(pair))
    return out


def _build_report(run_json: Dict[str, Any], niche_compare: Dict[str, Any]) -> Dict[str, Any]:
    target_parent_pair, target_child = _get_target_niche(niche_compare)
    cycles = _extract_cycles_from_niche_compare(niche_compare)
    summary = _summarize_cycles(cycles)

    recommendations = []
    dominant = summary["dominant_mode_hypothesis"]

    if dominant == "partial_relief_mode":
        recommendations.append(
            "The niche most likely wants sub-full burden sharing: more than zero support, less than a full stable patch."
        )
        recommendations.append(
            "The next dynamics-side experiment should approximate partial occupancy or partial local support amplitude before inventing a global recurrence penalty."
        )
    elif dominant == "shell_redistribution_mode":
        recommendations.append(
            "The niche most likely wants support redistributed around the shell rather than added as a whole new stable patch."
        )
        recommendations.append(
            "The next dynamics-side experiment should test a local shell-redistribution or shell-relief move."
        )
    elif dominant == "shared_support_mode":
        recommendations.append(
            "The niche most likely wants its missing role carried in a distributed way across nearby supports."
        )
        recommendations.append(
            "The next dynamics-side experiment should test a shared-support or role-sharing move rather than a full birth."
        )
    else:
        recommendations.append(
            "The current evidence is mixed; keep the result interpretive and gather more niche-state probes before touching the dynamics."
        )

    return {
        "script": "hsf_mesoscape_partial_mode_probe.py",
        "target_niche": {
            "parent_pair": list(target_parent_pair) if target_parent_pair is not None else None,
            "child": int(target_child),
        },
        "physics_config": run_json.get("physics_config", {}),
        "bookkeeping_config": run_json.get("bookkeeping_config", {}),
        "summary": summary,
        "recommendations": recommendations,
    }


def print_human_summary(report: Dict[str, Any]) -> None:
    tgt = report["target_niche"]
    summ = report["summary"]

    print("=== Partial Mode Probe ===")
    print(f"parent pair: {tgt['parent_pair']}")
    print(f"child:       {tgt['child']}")
    print(f"cycles:      {summ['n_cycles']}")
    print(f"dominant:    {summ['dominant_mode_hypothesis']}")
    print()

    print("Mode counts")
    for k, v in sorted((summ.get("mode_hypothesis_counts") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
    print()

    print("Mode score means")
    for k, v in sorted((summ.get("mode_score_means") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v:.3f}")
    print()

    d = summ.get("mean_absent_to_present_deltas", {}) or {}
    print("Mean absent -> present deltas")
    print(f"  role relief mean:  {_safe_float(d.get('role_relief_mean')):.4f}")
    print(f"  activity sum mean: {_safe_float(d.get('activity_sum_mean')):.4f}")
    print(f"  local cluster:     {_safe_float(d.get('local_cluster_mean')):.4f}")
    print(f"  distinctness mean: {_safe_float(d.get('distinctness_mean')):.4f}")
    print(f"  anchor diversity:  {_safe_float(d.get('anchor_diversity')):.4f}")
    print()

    print("Recommendations")
    for rec in report.get("recommendations", []):
        print(f"  - {rec}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score whether a recurrent niche looks like partial relief, shell redistribution, or shared support."
    )
    parser.add_argument("run_json", help="Path to hsf_mesoscape_*.json")
    parser.add_argument("niche_compare_json", help="Path to *_niche_compare_probe.json")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <run stem>_partial_mode_probe.json",
    )
    args = parser.parse_args()

    run_path = Path(args.run_json)
    niche_path = Path(args.niche_compare_json)
    if not run_path.exists():
        raise FileNotFoundError(f"Run JSON not found: {run_path}")
    if not niche_path.exists():
        raise FileNotFoundError(f"Niche compare JSON not found: {niche_path}")

    run_json = _load_json(run_path)
    niche_compare = _load_json(niche_path)

    report = _build_report(run_json, niche_compare)

    out_path = (
        Path(args.json_out)
        if args.json_out
        else run_path.with_name(run_path.stem + "_partial_mode_probe.json")
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_human_summary(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()