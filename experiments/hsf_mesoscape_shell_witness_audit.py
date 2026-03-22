#!/usr/bin/env python3
# filename: hsf_mesoscape_shell_witness_audit.py

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


def _sorted_edge(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _neighbor_map(edges: Sequence[Sequence[int]]) -> Dict[int, set]:
    nbr: Dict[int, set] = defaultdict(set)
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        a, b = int(e[0]), int(e[1])
        nbr[a].add(b)
        nbr[b].add(a)
    return nbr


def _extract_core_pair_from_diag(diag: Dict[str, Any]) -> List[int]:
    core_before = diag.get("core_before", {}) or {}
    cp = core_before.get("core_pair")
    if isinstance(cp, list) and len(cp) == 2:
        return [int(cp[0]), int(cp[1])]
    return []


def _extract_snapshot_core_pair(snap: Dict[str, Any]) -> List[int]:
    core = snap.get("dominant_core") or {}
    cp = core.get("core_pair")
    if isinstance(cp, list) and len(cp) == 2:
        return [int(cp[0]), int(cp[1])]
    return []


def _shell_nodes(core_pair: Sequence[int], active_edges: Sequence[Sequence[int]]) -> List[int]:
    if not isinstance(core_pair, (list, tuple)) or len(core_pair) != 2:
        return []
    i, j = int(core_pair[0]), int(core_pair[1])
    out = set()
    for e in active_edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        a, b = int(e[0]), int(e[1])
        if i in (a, b) or j in (a, b):
            out.add(a)
            out.add(b)
    out.discard(i)
    out.discard(j)
    return sorted(out)


def _snapshot_by_step(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for snap in data.get("snapshots", []) or []:
        step = _safe_int(snap.get("step"))
        if step >= 0:
            out[step] = snap
    return out


def _late_start(n: int) -> int:
    return max(0, (2 * n) // 3)


def _winner_gap(winner: Dict[str, Any], lower: Dict[str, Any]) -> float:
    return _safe_float(winner.get("deltaF")) - _safe_float(lower.get("deltaF"))


def _shell_case(row: Dict[str, Any]) -> str:
    if row["winner_move_type"] == "lower_support" and row["lower_deltaF"] > 0.0:
        return "shell_node_wins"
    if row["lower_deltaF"] > 0.0 and row["deltaF_gap_to_winner"] > 0.0:
        return "lawful_but_outcompeted"
    if row["lower_deltaF"] <= 0.0 and row["shell_penalty"] > 0.5:
        return "high_shell_penalty_loss"
    if row["lower_deltaF"] <= 0.0 and row["shell_penalty"] <= 0.5:
        return "non_shell_dominant_loss"
    return "mixed"


def _compare_rows_for_node(rows: List[Dict[str, Any]], node_id: int) -> Dict[str, Any]:
    group = [r for r in rows if r["lower_node"] == node_id]
    return {
        "n": len(group),
        "winner_types": dict(Counter(r["winner_move_type"] for r in group)),
        "case_counts": dict(Counter(_shell_case(r) for r in group)),
        "shell_penalty_mean": _mean([r["shell_penalty"] for r in group]),
        "shell_penalty_stdev": _stdev([r["shell_penalty"] for r in group]),
        "retirement_readiness_mean": _mean([r["retirement_readiness"] for r in group]),
        "bookkeeping_safety_mean": _mean([r["bookkeeping_safety"] for r in group]),
        "substitutability_mean": _mean([r["substitutability"] for r in group]),
        "functional_ready_mean": _mean([r["functional_ready"] for r in group]),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in group]),
        "gap_to_winner_mean": _mean([r["deltaF_gap_to_winner"] for r in group]),
        "core_neighbors_mean": _mean([r["core_neighbor_count"] for r in group]),
        "noncore_neighbors_mean": _mean([r["noncore_neighbor_count"] for r in group]),
        "neighbor_cluster_mean": _mean([r["neighbor_cluster"] for r in group]),
        "cross_replaceability_mean": _mean([r["cross_replaceability"] for r in group]),
        "bridge_unique_mean": _mean([r["bridge_unique"] for r in group]),
    }


def _build_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    snaps = data.get("snapshots", []) or []
    step_to_snap = _snapshot_by_step(data)
    rows: List[Dict[str, Any]] = []

    for snap in snaps:
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

        lower_node = _safe_int(best_lower.get("node", best_lower.get("move_object")))
        rr = best_lower.get("retirement_info") or {}

        snap_edges = snap.get("active_edges") or []
        nbr = _neighbor_map(snap_edges)
        neighbors = sorted(nbr.get(lower_node, set()))

        core_pair = _extract_snapshot_core_pair(snap)
        core_set = set(core_pair)
        core_neighbors = [n for n in neighbors if n in core_set]
        noncore_neighbors = [n for n in neighbors if n not in core_set]

        existing = 0
        possible = 0
        for idx, a in enumerate(neighbors):
            for b in neighbors[idx + 1 :]:
                possible += 1
                e = list(_sorted_edge(a, b))
                if e in [list(_sorted_edge(int(x[0]), int(x[1]))) for x in snap_edges]:
                    existing += 1
        neighbor_cluster = float(existing / possible) if possible > 0 else 1.0

        cross_existing = 0
        cross_possible = 0
        snap_edge_set = {tuple(_sorted_edge(int(e[0]), int(e[1]))) for e in snap_edges if isinstance(e, (list, tuple)) and len(e) == 2}
        for a in core_neighbors:
            for b in noncore_neighbors:
                cross_possible += 1
                if _sorted_edge(a, b) in snap_edge_set:
                    cross_existing += 1
        cross_replaceability = float(cross_existing / cross_possible) if cross_possible > 0 else 1.0

        bridge_unique = 0.0
        if core_neighbors and noncore_neighbors:
            bridge_unique = max(0.0, 1.0 - cross_replaceability)
        elif core_neighbors and len(neighbors) == 1:
            bridge_unique = 0.6

        row = {
            "step": int(step),
            "winner_move_type": str(winner.get("move_type")),
            "winner_deltaF": _safe_float(winner.get("deltaF")),
            "lower_node": int(lower_node),
            "lower_deltaF": _safe_float(best_lower.get("deltaF")),
            "deltaF_gap_to_winner": _winner_gap(winner, best_lower),
            "shell_penalty": _safe_float(rr.get("shell_penalty")),
            "shell_indispensability": _safe_float(rr.get("shell_indispensability")),
            "retirement_readiness": _safe_float(rr.get("retirement_readiness")),
            "edge_ready": _safe_float(rr.get("edge_ready")),
            "functional_ready": _safe_float(rr.get("functional_ready")),
            "bookkeeping_safety": _safe_float(rr.get("bookkeeping_safety")),
            "substitutability": _safe_float(rr.get("substitutability")),
            "core_penalty": _safe_float(rr.get("core_penalty")),
            "W_NR": _safe_float(best_lower.get("W_NR")),
            "dCF": _safe_float(best_lower.get("dCF")),
            "dE_expr_raw": _safe_float(best_lower.get("dE_expr_raw")),
            "dE_expr": _safe_float(best_lower.get("dE_expr")),
            "delta_Odiff_R": _safe_float(best_lower.get("delta_Odiff_R")),
            "core_pair": core_pair,
            "shell_nodes": _shell_nodes(core_pair, snap_edges),
            "neighbors": neighbors,
            "core_neighbor_count": len(core_neighbors),
            "noncore_neighbor_count": len(noncore_neighbors),
            "neighbor_cluster": float(neighbor_cluster),
            "cross_replaceability": float(cross_replaceability),
            "bridge_unique": float(bridge_unique),
        }
        row["case"] = _shell_case(row)
        rows.append(row)

    return rows


def _readout(all_rows: List[Dict[str, Any]], late_rows: List[Dict[str, Any]]) -> List[str]:
    reads: List[str] = []
    all_cases = Counter(r["case"] for r in all_rows)
    late_cases = Counter(r["case"] for r in late_rows)

    if all_cases.get("high_shell_penalty_loss", 0) > 0:
        reads.append("High shell-penalty losses remain common among the best lower_support candidates.")
    if late_cases.get("high_shell_penalty_loss", 0) > 0:
        reads.append("In the late regime, shell protection is still the dominant demotion blocker for some recurrent nodes.")
    if late_cases.get("shell_node_wins", 0) > 0:
        reads.append("Late-time shell demotion does sometimes win, so shell protection is no longer absolute.")
    if not reads:
        reads.append("No single shell-witness pattern dominates the lower-support outcomes.")
    return reads


def _top_late_nodes(rows: List[Dict[str, Any]], k: int = 5) -> List[int]:
    ctr = Counter(r["lower_node"] for r in rows)
    return [node for node, _ in ctr.most_common(k)]


def _flagged_rows(rows: List[Dict[str, Any]], limit: int = 30) -> List[Dict[str, Any]]:
    flagged = sorted(
        rows,
        key=lambda r: (
            -r["shell_penalty"],
            -r["deltaF_gap_to_winner"],
            r["step"],
        ),
    )
    return flagged[:limit]


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "winner_counts": dict(Counter(r["winner_move_type"] for r in rows)),
        "case_counts": dict(Counter(r["case"] for r in rows)),
        "shell_penalty_mean": _mean([r["shell_penalty"] for r in rows]),
        "shell_penalty_stdev": _stdev([r["shell_penalty"] for r in rows]),
        "retirement_readiness_mean": _mean([r["retirement_readiness"] for r in rows]),
        "lower_deltaF_mean": _mean([r["lower_deltaF"] for r in rows]),
        "gap_to_winner_mean": _mean([r["deltaF_gap_to_winner"] for r in rows]),
        "neighbor_cluster_mean": _mean([r["neighbor_cluster"] for r in rows]),
        "cross_replaceability_mean": _mean([r["cross_replaceability"] for r in rows]),
        "bridge_unique_mean": _mean([r["bridge_unique"] for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Focused shell witness audit for lower_support candidates. "
            "Compares winning and losing shell demotion cases and exposes the shell witness ingredients."
        )
    )
    parser.add_argument("json_path", help="Path to graded-support run JSON")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_shell_witness_audit.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = _load_json(in_path)
    rows = _build_rows(data)
    late = rows[_late_start(len(rows)):] if rows else []

    top_nodes = _top_late_nodes(late, k=5)
    node_comparisons = {str(node): _compare_rows_for_node(late, node) for node in top_nodes}

    report = {
        "script": "hsf_mesoscape_shell_witness_audit.py",
        "input_json": str(in_path),
        "overall_summary": _summary(rows),
        "late_summary": _summary(late),
        "late_top_nodes": top_nodes,
        "late_node_comparisons": node_comparisons,
        "flagged_rows": _flagged_rows(rows),
        "readout": _readout(rows, late),
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_shell_witness_audit.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Shell Witness Audit ===")
    print(f"rows:                     {report['overall_summary']['n_rows']}")
    print(f"late rows:                {report['late_summary']['n_rows']}")
    print(f"overall cases:            {report['overall_summary']['case_counts']}")
    print(f"late cases:               {report['late_summary']['case_counts']}")
    print(f"late top nodes:           {report['late_top_nodes']}")
    print(f"late shell penalty mean:  {report['late_summary']['shell_penalty_mean']:.6f}")
    print(f"late replaceability mean: {report['late_summary']['cross_replaceability_mean']:.6f}")
    print("Readout")
    for line in report["readout"]:
        print(f"  - {line}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()