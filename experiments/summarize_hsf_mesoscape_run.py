# filename: summarize_hsf_mesoscape_run.py

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarize a large HSF mesoscape graded-support sandbox JSON into a compact report."
        )
    )
    p.add_argument(
        "input_json",
        type=str,
        help="Path to the large JSON output from hsf_mesoscape_graded_support_sandbox.py",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="hsf_mesoscape_graded_support_summary.json",
        help="Path to write the compact summary JSON",
    )
    p.add_argument(
        "--top-moves",
        type=int,
        default=20,
        help="How many accepted moves to keep in the compact summary",
    )
    p.add_argument(
        "--top-snapshots",
        type=int,
        default=12,
        help="How many snapshots to keep in the compact summary",
    )
    return p.parse_args()


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


def summarize_trace(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "min": None,
            "max": None,
            "mean": None,
            "delta": None,
        }
    n = len(values)
    mean_val = sum(values) / float(n)
    return {
        "count": int(n),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": mean_val,
        "delta": values[-1] - values[0],
    }


def pick_snapshot_fields(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    sigma_summary = safe_get(snapshot, "sigma_summary", {})
    commitment_summary = safe_get(snapshot, "commitment_summary", {})
    dominant_core = safe_get(snapshot, "dominant_core", {})
    metric = safe_get(snapshot, "metric", {})
    graph_stats = safe_get(snapshot, "graph_stats", {})

    return {
        "step": safe_get(snapshot, "step"),
        "active_nodes": safe_get(snapshot, "active_nodes", []),
        "active_edge_count": safe_get(snapshot, "active_edge_count"),
        "dominant_core_pair": safe_get(dominant_core, "core_pair"),
        "dominant_core_score": safe_get(dominant_core, "score"),
        "mean_sigma": safe_get(sigma_summary, "mean_sigma"),
        "n_full": safe_get(sigma_summary, "n_full"),
        "n_partial": safe_get(sigma_summary, "n_partial"),
        "n_zero": safe_get(sigma_summary, "n_zero"),
        "n_interfaces_tracked": safe_get(commitment_summary, "n_interfaces_tracked"),
        "mean_commitment": safe_get(commitment_summary, "mean_commitment"),
        "top_interfaces": safe_get(commitment_summary, "top_interfaces", [])[:6],
        "mean_edge_strength": safe_get(metric, "mean_edge_strength"),
        "edge_strength_cv": safe_get(metric, "edge_strength_cv"),
        "n_components": safe_get(graph_stats, "n_components"),
        "largest_component": safe_get(graph_stats, "largest_component"),
        "n_raise_support_this_eval": safe_get(snapshot, "n_raise_support_this_eval"),
        "n_lower_support_this_eval": safe_get(snapshot, "n_lower_support_this_eval"),
        "n_edge_up_this_eval": safe_get(snapshot, "n_edge_up_this_eval"),
        "n_edge_down_this_eval": safe_get(snapshot, "n_edge_down_this_eval"),
        "candidate_count": len(safe_get(snapshot, "candidate_move_diagnostics", [])),
    }


def pick_move_fields(move: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "move_type": safe_get(move, "move_type"),
        "deltaF": safe_get(move, "deltaF"),
        "dE_expr": safe_get(move, "dE_expr"),
        "dE_expr_raw": safe_get(move, "dE_expr_raw"),
        "dCB": safe_get(move, "dCB"),
        "dCS": safe_get(move, "dCS"),
        "dCF": safe_get(move, "dCF"),
        "W_NR": safe_get(move, "W_NR"),
        "F_org": safe_get(move, "F_org"),
        "n_active_before": safe_get(move, "n_active_before"),
        "n_active_after": safe_get(move, "n_active_after"),
    }

    if "parents" in move:
        out["parents"] = safe_get(move, "parents")
    if "child" in move:
        out["child"] = safe_get(move, "child")
    if "node" in move:
        out["node"] = safe_get(move, "node")
    if "edge" in move:
        out["edge"] = safe_get(move, "edge")
    if "sigma_before" in move:
        out["sigma_before"] = safe_get(move, "sigma_before")
    if "sigma_after" in move:
        out["sigma_after"] = safe_get(move, "sigma_after")
    if "w_before" in move:
        out["w_before"] = safe_get(move, "w_before")
    if "w_after" in move:
        out["w_after"] = safe_get(move, "w_after")

    if "delta_Odiff_R" in move:
        out["delta_Odiff_R"] = safe_get(move, "delta_Odiff_R")
    if "expr_odiff_adjustment" in move:
        out["expr_odiff_adjustment"] = safe_get(move, "expr_odiff_adjustment")
    if "graded_witness_adjustment" in move:
        out["graded_witness_adjustment"] = safe_get(move, "graded_witness_adjustment")
    if "edge_up_relief_gain" in move:
        out["edge_up_relief_gain"] = safe_get(move, "edge_up_relief_gain")
    if "edge_up_distinct_gain" in move:
        out["edge_up_distinct_gain"] = safe_get(move, "edge_up_distinct_gain")
    if "edge_up_hard_gate_triggered" in move:
        out["edge_up_hard_gate_triggered"] = safe_get(move, "edge_up_hard_gate_triggered")

    return out


def top_entries_by_abs(values: List[Dict[str, Any]], key: str, n: int) -> List[Dict[str, Any]]:
    return sorted(
        values,
        key=lambda x: abs(float(x.get(key, 0.0) or 0.0)),
        reverse=True,
    )[:n]


def top_entries(values: List[Dict[str, Any]], key: str, n: int) -> List[Dict[str, Any]]:
    return sorted(
        values,
        key=lambda x: float(x.get(key, 0.0) or 0.0),
        reverse=True,
    )[:n]


def sample_snapshots(snapshots: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if not snapshots:
        return []
    if len(snapshots) <= n:
        return [pick_snapshot_fields(s) for s in snapshots]

    idxs = sorted(set([
        0,
        len(snapshots) - 1,
        *[round(i * (len(snapshots) - 1) / max(1, n - 1)) for i in range(n)],
    ]))
    return [pick_snapshot_fields(snapshots[i]) for i in idxs[:n]]


def build_summary(data: Dict[str, Any], top_moves: int, top_snapshots: int) -> Dict[str, Any]:
    accepted_moves = safe_get(data, "accepted_moves", [])
    snapshots = safe_get(data, "snapshots", [])
    move_counts = safe_get(data, "move_counts", {})
    final_sigma = safe_get(data, "final_sigma", [])
    final_interfaces = safe_get(data, "final_interface_commitment", [])
    active_count_trace = safe_get(data, "active_count_trace", [])
    edge_count_trace = safe_get(data, "edge_count_trace", [])
    sigma_mean_trace = safe_get(data, "sigma_mean_trace", [])

    accepted_compact = [pick_move_fields(m) for m in accepted_moves]

    move_type_counter = Counter(m.get("move_type", "unknown") for m in accepted_moves)
    hard_gate_count = sum(
        1 for m in accepted_moves if bool(m.get("edge_up_hard_gate_triggered", False))
    )

    strongest_moves = top_entries(accepted_compact, "deltaF", top_moves)
    biggest_odiff = top_entries_by_abs(
        [m for m in accepted_compact if "delta_Odiff_R" in m],
        "delta_Odiff_R",
        min(top_moves, 12),
    )

    final_snapshot = pick_snapshot_fields(snapshots[-1]) if snapshots else {}
    first_snapshot = pick_snapshot_fields(snapshots[0]) if snapshots else {}

    compact_interfaces = sorted(
        final_interfaces,
        key=lambda x: float(x.get("commitment", 0.0)),
        reverse=True,
    )[:12]

    sigma_nonzero = [float(x) for x in final_sigma if float(x) > 1e-12]
    sigma_full = sum(1 for x in final_sigma if float(x) >= 0.999)
    sigma_partial = sum(1 for x in final_sigma if 1e-12 < float(x) < 0.999)

    summary = {
        "script": safe_get(data, "script"),
        "gpu_enabled": safe_get(data, "gpu_enabled"),
        "physics_config": safe_get(data, "physics_config", {}),
        "bookkeeping_config": safe_get(data, "bookkeeping_config", {}),
        "graded_support_config": safe_get(data, "graded_support_config", {}),
        "run_overview": {
            "n_snapshots": len(snapshots),
            "n_accepted_moves": len(accepted_moves),
            "move_counts_declared": move_counts,
            "move_counts_observed": dict(move_type_counter),
            "edge_up_hard_gate_trigger_count": int(hard_gate_count),
        },
        "trace_summary": {
            "active_count_trace": summarize_trace([float(x) for x in active_count_trace]),
            "edge_count_trace": summarize_trace([float(x) for x in edge_count_trace]),
            "sigma_mean_trace": summarize_trace([float(x) for x in sigma_mean_trace]),
        },
        "initial_state": first_snapshot,
        "final_state": {
            **final_snapshot,
            "final_sigma": [float(x) for x in final_sigma],
            "final_sigma_nonzero": sigma_nonzero,
            "final_sigma_full_count": int(sigma_full),
            "final_sigma_partial_count": int(sigma_partial),
            "top_final_interfaces": compact_interfaces,
        },
        "accepted_moves_top_by_deltaF": strongest_moves,
        "accepted_moves_top_by_abs_delta_Odiff_R": biggest_odiff,
        "snapshot_samples": sample_snapshots(snapshots, top_snapshots),
    }
    return summary


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = build_summary(
        data=data,
        top_moves=int(args.top_moves),
        top_snapshots=int(args.top_snapshots),
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    in_size = input_path.stat().st_size if input_path.exists() else 0
    out_size = output_path.stat().st_size if output_path.exists() else 0

    print(f"read:  {input_path}")
    print(f"wrote: {output_path}")
    print(f"input bytes:  {in_size}")
    print(f"output bytes: {out_size}")
    if in_size > 0:
        print(f"compression ratio: {out_size / in_size:.4f}")


if __name__ == "__main__":
    main()