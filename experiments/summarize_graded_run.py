#!/usr/bin/env python3
"""Extract a compact summary from a graded-support sandbox JSON output.

Usage:
    python summarize_graded_run.py graded_v4_N12_safe.json
    python summarize_graded_run.py graded_v4_N12_safe.json --out summary.json

Produces a small JSON (~10-50 KB) with:
  - run config (physics, bookkeeping, v4 settings)
  - move counts and acceptance rate
  - per-eval-window traces (active nodes, edges, sigma, bandwidth, core, moves)
  - accepted move log (type, deltaF, dE, constraint costs — no site dumps)
  - final state summary
  - phase detection (growth / plateau / churn indicators)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_traces(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the lightweight time-series traces."""
    return {
        "active_count_trace": data.get("active_count_trace", []),
        "edge_count_trace": data.get("edge_count_trace", []),
        "sigma_mean_trace": data.get("sigma_mean_trace", []),
        "bandwidth_activity_trace": data.get("bandwidth_activity_trace", []),
    }


def extract_snapshot_timeline(snapshots: List[Dict]) -> List[Dict[str, Any]]:
    """One compact row per eval window."""
    timeline = []
    for s in snapshots:
        core = s.get("dominant_core") or {}
        metrics = s.get("metric") or {}
        gstats = s.get("graph_stats") or {}
        lr = s.get("link_reg_summary") or {}
        sig = s.get("sigma_summary") or {}

        timeline.append({
            "step": s.get("step"),
            "n_active": len(s.get("active_nodes", [])),
            "n_edges": s.get("active_edge_count", 0),
            "core_pair": core.get("core_pair"),
            "core_score": round(core.get("score", 0.0), 4),
            "core_mi": round(core.get("mi", 0.0), 4),
            "mean_edge_strength": round(metrics.get("mean_edge_strength", 0.0), 4),
            "mean_degree": round(gstats.get("mean_degree", 0.0), 2),
            "max_degree": gstats.get("max_degree", 0),
            "mean_sigma": round(sig.get("mean_sigma", 0.0), 4),
            "n_full": sig.get("n_full", 0),
            "n_partial": sig.get("n_partial", 0),
            "bw_activity": round(lr.get("mean_activity", 0.0), 4),
            "bw_eff_rank": round(lr.get("mean_effective_rank", 0.0), 4),
            "n_moves": s.get("n_moves_this_eval", 0),
            "raise": s.get("n_raise_support_this_eval", 0),
            "lower": s.get("n_lower_support_this_eval", 0),
            "edge_up": s.get("n_edge_up_this_eval", 0),
            "edge_down": s.get("n_edge_down_this_eval", 0),
        })
    return timeline


_MOVE_KEEP_KEYS = [
    "move_type", "step_hint", "move_round", "deltaF",
    "dE_expr", "dE_expr_raw", "dCB", "dCS", "dCF",
    "W_NR", "F_org", "W_func", "W_ep",
    "n_active_before", "n_active_after",
    "graded_witness_adjustment",
    # birth-specific
    "parents", "child", "sigma_before", "sigma_after",
    "birth_novelty", "birth_parent_relief", "birth_justification", "birth_distinctness",
    # edge-specific
    "edge", "w_before", "w_after",
    "edge_up_hard_gate_triggered", "edge_up_relief_gain", "edge_up_distinct_gain",
    # lower-specific
    "node",
    # odiff
    "delta_Odiff_R", "odiff_ratio_R",
]


def slim_accepted_moves(moves: List[Dict]) -> List[Dict[str, Any]]:
    """Keep only the diagnostically important fields from each accepted move."""
    out = []
    for m in moves:
        row = {}
        for k in _MOVE_KEEP_KEYS:
            if k in m:
                v = m[k]
                if isinstance(v, float):
                    row[k] = round(v, 6)
                else:
                    row[k] = v
        out.append(row)
    return out


def detect_phases(timeline: List[Dict]) -> Dict[str, Any]:
    """Simple phase detection from the timeline."""
    if not timeline:
        return {"phases": []}

    n_active = [t["n_active"] for t in timeline]
    n_edges = [t["n_edges"] for t in timeline]
    steps = [t["step"] for t in timeline]

    # Find growth phase end (last step where n_active increased)
    growth_end_idx = 0
    for i in range(1, len(n_active)):
        if n_active[i] > n_active[i - 1]:
            growth_end_idx = i

    # Find plateau (5+ consecutive windows with no change in active count)
    plateau_start = None
    run_length = 0
    for i in range(1, len(n_active)):
        if n_active[i] == n_active[i - 1]:
            run_length += 1
            if run_length >= 5 and plateau_start is None:
                plateau_start = i - run_length
        else:
            run_length = 0

    # Check for churn: any lower_support or edge_down moves accepted
    has_lower = any(t["lower"] > 0 for t in timeline)
    has_edge_down = any(t["edge_down"] > 0 for t in timeline)

    # Bandwidth drift: did link_reg activity change meaningfully?
    bw = [t["bw_activity"] for t in timeline if t["bw_activity"] > 0]
    bw_drift = round(bw[-1] - bw[0], 4) if len(bw) >= 2 else 0.0

    # Core stability: how many distinct core pairs appeared?
    cores = [tuple(t["core_pair"]) if t["core_pair"] else None for t in timeline]
    unique_cores = len(set(c for c in cores if c is not None))
    # Last core switch
    last_core_switch = 0
    for i in range(1, len(cores)):
        if cores[i] != cores[i - 1]:
            last_core_switch = steps[i]

    return {
        "growth_ended_at_step": steps[growth_end_idx] if growth_end_idx > 0 else None,
        "peak_active": max(n_active),
        "peak_edges": max(n_edges),
        "final_active": n_active[-1],
        "final_edges": n_edges[-1],
        "plateau_start_step": steps[plateau_start] if plateau_start is not None else None,
        "has_reabsorption_moves": has_lower,
        "has_edge_down_moves": has_edge_down,
        "bandwidth_drift": bw_drift,
        "unique_core_pairs": unique_cores,
        "last_core_switch_step": last_core_switch if last_core_switch > 0 else None,
        "healthy_churn_indicators": int(has_lower) + int(has_edge_down) + int(bw_drift < -0.05),
    }


def summarize(data: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = data.get("snapshots", [])
    accepted = data.get("accepted_moves", [])
    timeline = extract_snapshot_timeline(snapshots)

    return {
        "script": data.get("script", ""),
        "version": data.get("version", ""),
        "physics_config": data.get("physics_config", {}),
        "bookkeeping_config": data.get("bookkeeping_config", {}),
        "graded_support_config": data.get("graded_support_config", {}),
        "v4_config": data.get("v4_config", {}),
        "move_counts": data.get("move_counts", {}),
        "total_moves_accepted": data.get("total_moves_accepted", 0),
        "traces": extract_traces(data),
        "timeline": timeline,
        "accepted_moves": slim_accepted_moves(accepted),
        "final_sigma": data.get("final_sigma", []),
        "final_interface_commitment": data.get("final_interface_commitment", []),
        "final_active_nodes": data.get("final_active_nodes", []),
        "final_active_edges": data.get("final_active_edges", []),
        "phase_detection": detect_phases(timeline),
        "gpu_enabled": data.get("gpu_enabled", False),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize a graded-support sandbox JSON output.")
    parser.add_argument("input", help="Path to the full JSON output file.")
    parser.add_argument("--out", default=None, help="Output path. Default: <input>_summary.json")
    args = parser.parse_args()

    inpath = Path(args.input)
    if not inpath.exists():
        print(f"Error: {inpath} not found", file=sys.stderr)
        sys.exit(1)

    outpath = Path(args.out) if args.out else inpath.with_name(inpath.stem + "_summary.json")

    print(f"Loading {inpath} ...", end=" ", flush=True)
    data = load_json(str(inpath))
    print(f"({inpath.stat().st_size / 1e6:.1f} MB)")

    summary = summarize(data)

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {outpath} ({outpath.stat().st_size / 1e3:.1f} KB)")
    print()

    # Quick console report
    pd = summary["phase_detection"]
    mc = summary["move_counts"]
    total = summary["total_moves_accepted"]
    n_evals = len(summary["timeline"])

    print(f"  Evals: {n_evals}")
    print(f"  Total moves accepted: {total}  ({total/max(1,n_evals):.1f}/eval)")
    print(f"    raise_support: {mc.get('raise_support',0)}")
    print(f"    lower_support: {mc.get('lower_support',0)}")
    print(f"    edge_up:       {mc.get('edge_up',0)}")
    print(f"    edge_down:     {mc.get('edge_down',0)}")
    print(f"  Peak: {pd['peak_active']} nodes, {pd['peak_edges']} edges")
    print(f"  Final: {pd['final_active']} nodes, {pd['final_edges']} edges")
    print(f"  Growth ended at step: {pd['growth_ended_at_step']}")
    print(f"  Plateau start: {pd['plateau_start_step']}")
    print(f"  Unique core pairs: {pd['unique_core_pairs']}")
    print(f"  Last core switch: step {pd['last_core_switch_step']}")
    print(f"  Bandwidth drift: {pd['bandwidth_drift']}")
    print(f"  Healthy churn indicators: {pd['healthy_churn_indicators']}/3")


if __name__ == "__main__":
    main()