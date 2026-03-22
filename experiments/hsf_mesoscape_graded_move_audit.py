#!/usr/bin/env python3
# filename: hsf_mesoscape_graded_move_audit.py

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


def _absmax_term_name(terms: Dict[str, float]) -> str:
    if not terms:
        return "none"
    return max(terms.items(), key=lambda kv: abs(kv[1]))[0]


def _max_term_name(terms: Dict[str, float]) -> str:
    if not terms:
        return "none"
    return max(terms.items(), key=lambda kv: kv[1])[0]


def _min_term_name(terms: Dict[str, float]) -> str:
    if not terms:
        return "none"
    return min(terms.items(), key=lambda kv: kv[1])[0]


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


def _score_terms(diag: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    lamB = _safe_float(cfg.get("lambda_B"), 0.0)
    lamS = _safe_float(cfg.get("lambda_S"), 0.0)
    lamF = _safe_float(cfg.get("lambda_F"), 0.0)
    lamR = _safe_float(cfg.get("lambda_R"), 0.0)

    dE = _safe_float(diag.get("dE_expr"), 0.0)
    b_term = -lamB * _safe_float(diag.get("dCB"), 0.0)
    s_term = -lamS * _safe_float(diag.get("dCS"), 0.0)
    f_term = -lamF * _safe_float(diag.get("dCF"), 0.0)
    r_term = -lamR * _safe_float(diag.get("W_NR"), 0.0)

    return {
        "expr_term": float(dE),
        "finite_bandwidth_term": float(b_term),
        "no_signalling_term": float(s_term),
        "no_refolding_func_term": float(f_term),
        "no_refolding_total_term": float(r_term),
    }


def _constraint_bucket(diag: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Interprets the four allowed pressures in a graded run.

    - finite bandwidth -> lambda_B * dCB term
    - no-signalling -> lambda_S * dCS term
    - no-refolding -> lambda_F * dCF plus lambda_R * W_NR
    - no-forgetting -> approximated by the persistence-preserving contribution inside dE_expr
      and by low retirement / low lowering pressure; here we expose only a proxy indicator
      rather than inventing a fifth term.
    """
    terms = _score_terms(diag, cfg)
    no_refolding_total = terms["no_refolding_func_term"] + terms["no_refolding_total_term"]

    # proxy only; not a new pressure
    noforget_proxy = 0.0
    mt = str(diag.get("move_type"))
    if mt in ("raise_support", "edge_up"):
        noforget_proxy = max(0.0, _safe_float(diag.get("dE_expr"), 0.0))
    elif mt in ("lower_support", "edge_down"):
        noforget_proxy = max(0.0, -_safe_float(diag.get("dE_expr_structural_adjustment"), 0.0))

    return {
        "finite_bandwidth": float(terms["finite_bandwidth_term"]),
        "no_signalling": float(terms["no_signalling_term"]),
        "no_refolding": float(no_refolding_total),
        "no_forgetting_proxy": float(noforget_proxy),
        "expression_total": float(terms["expr_term"]),
    }


def _audit_diag(diag: Dict[str, Any], cfg: Dict[str, Any], step: int, accepted: bool) -> Dict[str, Any]:
    terms = _score_terms(diag, cfg)
    buckets = _constraint_bucket(diag, cfg)

    penalty_terms = {
        "finite_bandwidth_term": terms["finite_bandwidth_term"],
        "no_signalling_term": terms["no_signalling_term"],
        "no_refolding_func_term": terms["no_refolding_func_term"],
        "no_refolding_total_term": terms["no_refolding_total_term"],
    }

    dominant_penalty = _min_term_name(penalty_terms)
    strongest_effect = _absmax_term_name({**terms})
    constraint_driver = _absmax_term_name(
        {
            "finite_bandwidth": buckets["finite_bandwidth"],
            "no_signalling": buckets["no_signalling"],
            "no_refolding": buckets["no_refolding"],
        }
    )

    return {
        "step": int(step),
        "accepted": bool(accepted),
        "move_type": str(diag.get("move_type")),
        "deltaF": _safe_float(diag.get("deltaF")),
        "dE_expr": _safe_float(diag.get("dE_expr")),
        "dE_expr_raw": _safe_float(diag.get("dE_expr_raw")),
        "terms": terms,
        "constraint_buckets": buckets,
        "dominant_penalty_term": dominant_penalty,
        "strongest_effect_term": strongest_effect,
        "constraint_driver": constraint_driver,
        "move_object": diag.get("move_object"),
    }


def _build_audit(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = data.get("bookkeeping_config", {}) or {}
    snaps = data.get("snapshots", []) or []

    accepted_rows: List[Dict[str, Any]] = []
    losing_rows: List[Dict[str, Any]] = []
    winner_vs_runnerup: List[Dict[str, Any]] = []

    accepted_type_counts = Counter()
    accepted_constraint_counts = Counter()
    late_constraint_counts = Counter()
    late_move_counts = Counter()

    if not snaps:
        return {
            "accepted_rows": [],
            "losing_rows": [],
            "winner_vs_runnerup": [],
            "summary": {},
        }

    late_start = max(0, len(snaps) * 2 // 3)

    for idx, snap in enumerate(snaps):
        step = _safe_int(snap.get("step"))
        cands = snap.get("candidate_move_diagnostics", []) or []
        if not cands:
            continue

        winner_type = _winner_from_snapshot(snap)
        scored = sorted(cands, key=lambda d: _safe_float(d.get("deltaF")), reverse=True)

        winner_diag = None
        for d in scored:
            if str(d.get("move_type")) == str(winner_type):
                winner_diag = d
                break
        if winner_diag is None:
            winner_diag = scored[0]

        accepted = _audit_diag(winner_diag, cfg, step, True)
        accepted_rows.append(accepted)
        accepted_type_counts[accepted["move_type"]] += 1
        accepted_constraint_counts[accepted["constraint_driver"]] += 1

        if idx >= late_start:
            late_constraint_counts[accepted["constraint_driver"]] += 1
            late_move_counts[accepted["move_type"]] += 1

        for d in scored:
            if d is winner_diag:
                continue
            losing_rows.append(_audit_diag(d, cfg, step, False))

        if len(scored) >= 2:
            runner = scored[1]
            runner_a = _audit_diag(runner, cfg, step, False)
            winner_vs_runnerup.append(
                {
                    "step": int(step),
                    "winner_move_type": accepted["move_type"],
                    "winner_deltaF": accepted["deltaF"],
                    "winner_constraint_driver": accepted["constraint_driver"],
                    "runnerup_move_type": runner_a["move_type"],
                    "runnerup_deltaF": runner_a["deltaF"],
                    "runnerup_constraint_driver": runner_a["constraint_driver"],
                    "deltaF_gap": float(accepted["deltaF"] - runner_a["deltaF"]),
                    "winner_terms": accepted["terms"],
                    "runnerup_terms": runner_a["terms"],
                }
            )

    # Aggregate accepted move stats by type
    type_term_stats: Dict[str, Dict[str, float]] = {}
    for mtype in sorted(set(r["move_type"] for r in accepted_rows)):
        rows = [r for r in accepted_rows if r["move_type"] == mtype]
        type_term_stats[mtype] = {
            "n": len(rows),
            "deltaF_mean": _mean([r["deltaF"] for r in rows]),
            "dE_expr_mean": _mean([r["dE_expr"] for r in rows]),
            "finite_bandwidth_mean": _mean([r["constraint_buckets"]["finite_bandwidth"] for r in rows]),
            "no_signalling_mean": _mean([r["constraint_buckets"]["no_signalling"] for r in rows]),
            "no_refolding_mean": _mean([r["constraint_buckets"]["no_refolding"] for r in rows]),
            "no_forgetting_proxy_mean": _mean([r["constraint_buckets"]["no_forgetting_proxy"] for r in rows]),
        }

    summary = {
        "n_snapshots_with_moves": len(accepted_rows),
        "accepted_move_type_counts": dict(accepted_type_counts),
        "accepted_constraint_driver_counts": dict(accepted_constraint_counts),
        "late_move_type_counts": dict(late_move_counts),
        "late_constraint_driver_counts": dict(late_constraint_counts),
        "accepted_type_term_stats": type_term_stats,
        "mean_winner_runnerup_gap": _mean([r["deltaF_gap"] for r in winner_vs_runnerup]),
        "late_regime_read": _late_regime_read(late_move_counts, late_constraint_counts),
    }

    return {
        "accepted_rows": accepted_rows,
        "losing_rows": losing_rows,
        "winner_vs_runnerup": winner_vs_runnerup,
        "summary": summary,
    }


def _late_regime_read(late_move_counts: Counter, late_constraint_counts: Counter) -> List[str]:
    reads: List[str] = []
    total_late = sum(late_move_counts.values()) or 1

    edge_frac = (late_move_counts.get("edge_up", 0) + late_move_counts.get("edge_down", 0)) / total_late
    raise_frac = late_move_counts.get("raise_support", 0) / total_late
    lower_frac = late_move_counts.get("lower_support", 0) / total_late

    if edge_frac > 0.60:
        reads.append("Late regime is interface-dominated rather than support-raising dominated.")
    if raise_frac > 0.40:
        reads.append("Late regime still depends strongly on further support seeding.")
    if lower_frac > 0.20:
        reads.append("Late regime includes substantial support-demotion pressure.")

    driver = late_constraint_counts.most_common(1)[0][0] if late_constraint_counts else "none"
    if driver == "finite_bandwidth":
        reads.append("Finite bandwidth appears to be the strongest late-time lawful selector.")
    elif driver == "no_refolding":
        reads.append("No-refolding appears to be the strongest late-time lawful selector.")
    elif driver == "no_signalling":
        reads.append("No-signalling appears to be the strongest late-time lawful selector.")

    if not reads:
        reads.append("Late regime has no single dominant move or constraint signature.")
    return reads


def print_summary(report: Dict[str, Any]) -> None:
    s = report["summary"]
    print("=== Graded Move Audit ===")
    print(f"snapshots with accepted moves:   {s['n_snapshots_with_moves']}")
    print(f"accepted move counts:           {s['accepted_move_type_counts']}")
    print(f"accepted constraint drivers:    {s['accepted_constraint_driver_counts']}")
    print(f"late move counts:               {s['late_move_type_counts']}")
    print(f"late constraint drivers:        {s['late_constraint_driver_counts']}")
    print(f"mean winner-runnerup gap:       {s['mean_winner_runnerup_gap']:.6f}")
    print()

    print("Accepted move term means by type")
    for mtype, stats in s["accepted_type_term_stats"].items():
        print(
            f"  {mtype}: n={stats['n']} "
            f"deltaF={stats['deltaF_mean']:.4f} "
            f"dE={stats['dE_expr_mean']:.4f} "
            f"BW={stats['finite_bandwidth_mean']:.4f} "
            f"NS={stats['no_signalling_mean']:.4f} "
            f"NR={stats['no_refolding_mean']:.4f} "
            f"NFproxy={stats['no_forgetting_proxy_mean']:.4f}"
        )
    print()

    print("Late regime read")
    for line in s["late_regime_read"]:
        print(f"  - {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit accepted and losing moves in a graded-support sandbox run and show which lawful "
            "constraint terms are actually driving selection."
        )
    )
    parser.add_argument("json_path", help="Path to hsf_mesoscape_*_graded_support.json")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_move_audit.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = _load_json(in_path)
    audit = _build_audit(data)

    report = {
        "script": "hsf_mesoscape_graded_move_audit.py",
        "input_json": str(in_path),
        **audit,
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_move_audit.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()