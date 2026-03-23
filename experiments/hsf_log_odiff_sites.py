#!/usr/bin/env python3
# filename: hsf_log_odiff_sites.py

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


Edge = Tuple[int, int]


def sorted_edge(a: int, b: int) -> Edge:
    return (a, b) if a <= b else (b, a)


def neighbor_map(active_edges: Iterable[Edge]) -> Dict[int, Set[int]]:
    nbr: Dict[int, Set[int]] = {}
    for a, b in active_edges:
        nbr.setdefault(int(a), set()).add(int(b))
        nbr.setdefault(int(b), set()).add(int(a))
    return nbr


def shell_nodes(core_pair: Optional[Edge], active_edges: Set[Edge]) -> Set[int]:
    if core_pair is None:
        return set()
    i, j = core_pair
    out: Set[int] = set()
    for a, b in active_edges:
        if i in (a, b) or j in (a, b):
            out.add(a)
            out.add(b)
    out.discard(i)
    out.discard(j)
    return out


def site_role_fingerprint(site_id: int, active_edges: Set[Edge], edge_strengths: Dict[Edge, float]) -> Dict[str, Any]:
    nbr = neighbor_map(active_edges)
    neighbors = sorted(nbr.get(int(site_id), set()))
    incident = [sorted_edge(site_id, j) for j in neighbors]
    incident_strengths = [float(edge_strengths.get(e, 0.0)) for e in incident]

    mean_strength = float(np.mean(incident_strengths)) if incident_strengths else 0.0
    activity_sum = float(np.sum(incident_strengths)) if incident_strengths else 0.0
    incident_count = int(len(incident))

    existing = 0
    possible = 0
    for idx, a in enumerate(neighbors):
        for b in neighbors[idx + 1:]:
            possible += 1
            if sorted_edge(a, b) in active_edges:
                existing += 1
    local_cluster = float(existing / possible) if possible > 0 else 0.0
    sibling_count_norm = float(min(1.0, max(0, incident_count - 1) / 4.0))

    novelty = float(max(0.0, 1.0 - 0.55 * sibling_count_norm - 0.45 * local_cluster))
    relief = float(min(1.0, 0.5 * mean_strength + 0.3 * local_cluster + 0.2 * min(1.0, incident_count / 3.0)))
    distinctness = float(max(0.0, 1.0 - local_cluster))
    weight = float(0.40 * novelty + 0.35 * relief + 0.25 * distinctness)

    return {
        "site_id": int(site_id),
        "role_id": f"site_{int(site_id)}",
        "neighbors": [int(x) for x in neighbors],
        "novelty": float(novelty),
        "relief": float(relief),
        "distinctness": float(distinctness),
        "weight": float(weight),
        "raw_metrics": {
            "incident_count": float(incident_count),
            "mean_strength": float(mean_strength),
            "activity_sum": float(activity_sum),
            "local_cluster": float(local_cluster),
            "sibling_count_norm": float(sibling_count_norm),
        },
    }


def odiff_from_fingerprints(fps: Sequence[Dict[str, Any]]) -> float:
    return float(sum(float(fp.get("weight", 0.0)) for fp in fps))


def local_region_nodes(
    move_kind: str,
    obj: Any,
    active_nodes_before: Set[int],
    active_edges_before: Set[Edge],
    active_nodes_after: Set[int],
    active_edges_after: Set[Edge],
    core_pair: Optional[Edge],
) -> List[int]:
    region: Set[int] = set()
    if core_pair is not None:
        region.update(core_pair)

    union_edges = set(active_edges_before) | set(active_edges_after)
    nbr = neighbor_map(union_edges)

    if move_kind == "birth":
        parents = tuple(int(x) for x in obj["parents"])
        child = int(obj["child"])
        region.update(parents)
        region.add(child)
        for p in parents:
            region.update(nbr.get(p, set()))
        region.update(nbr.get(child, set()))

    elif move_kind in ("weaken", "transfer"):
        if isinstance(obj, dict) and "edge" in obj:
            e = sorted_edge(int(obj["edge"][0]), int(obj["edge"][1]))
        else:
            e = sorted_edge(int(obj[0]), int(obj[1]))
        region.update(e)
        region.update(nbr.get(e[0], set()))
        region.update(nbr.get(e[1], set()))

    elif move_kind in ("retire", "lower_support"):
        node = int(obj if not isinstance(obj, dict) else obj.get("node", obj.get("move_object", -1)))
        region.add(node)
        region.update(nbr.get(node, set()))

    else:
        region.update(active_nodes_before)
        region.update(active_nodes_after)

    return sorted(int(n) for n in region if n in (set(active_nodes_before) | set(active_nodes_after)))


def compute_odiff_site_dump(
    move_kind: str,
    obj: Any,
    active_nodes_before: Set[int],
    active_edges_before: Set[Edge],
    edge_strengths_before: Dict[Edge, float],
    active_nodes_after: Set[int],
    active_edges_after: Set[Edge],
    edge_strengths_after: Dict[Edge, float],
    core_pair: Optional[Edge],
) -> Dict[str, Any]:
    region_nodes = local_region_nodes(
        move_kind=move_kind,
        obj=obj,
        active_nodes_before=active_nodes_before,
        active_edges_before=active_edges_before,
        active_nodes_after=active_nodes_after,
        active_edges_after=active_edges_after,
        core_pair=core_pair,
    )

    fps_before = [
        site_role_fingerprint(n, active_edges_before, edge_strengths_before)
        for n in region_nodes
        if n in active_nodes_before
    ]
    fps_after = [
        site_role_fingerprint(n, active_edges_after, edge_strengths_after)
        for n in region_nodes
        if n in active_nodes_after
    ]

    idx_before = {int(fp["site_id"]): fp for fp in fps_before}
    idx_after = {int(fp["site_id"]): fp for fp in fps_after}

    site_rows: List[Dict[str, Any]] = []
    all_sites = sorted(set(idx_before) | set(idx_after))
    for site in all_sites:
        b = idx_before.get(site, {})
        a = idx_after.get(site, {})
        brm = b.get("raw_metrics", {}) if isinstance(b, dict) else {}
        arm = a.get("raw_metrics", {}) if isinstance(a, dict) else {}

        row = {
            "site_id": int(site),
            "neighbors_before": list(b.get("neighbors", [])) if isinstance(b, dict) else [],
            "neighbors_after": list(a.get("neighbors", [])) if isinstance(a, dict) else [],
            "weight_before": float(b.get("weight", 0.0)) if isinstance(b, dict) else 0.0,
            "weight_after": float(a.get("weight", 0.0)) if isinstance(a, dict) else 0.0,
            "weight_delta": (float(a.get("weight", 0.0)) if isinstance(a, dict) else 0.0)
            - (float(b.get("weight", 0.0)) if isinstance(b, dict) else 0.0),
            "novelty_before": float(b.get("novelty", 0.0)) if isinstance(b, dict) else 0.0,
            "novelty_after": float(a.get("novelty", 0.0)) if isinstance(a, dict) else 0.0,
            "novelty_delta": (float(a.get("novelty", 0.0)) if isinstance(a, dict) else 0.0)
            - (float(b.get("novelty", 0.0)) if isinstance(b, dict) else 0.0),
            "relief_before": float(b.get("relief", 0.0)) if isinstance(b, dict) else 0.0,
            "relief_after": float(a.get("relief", 0.0)) if isinstance(a, dict) else 0.0,
            "relief_delta": (float(a.get("relief", 0.0)) if isinstance(a, dict) else 0.0)
            - (float(b.get("relief", 0.0)) if isinstance(b, dict) else 0.0),
            "distinctness_before": float(b.get("distinctness", 0.0)) if isinstance(b, dict) else 0.0,
            "distinctness_after": float(a.get("distinctness", 0.0)) if isinstance(a, dict) else 0.0,
            "distinctness_delta": (float(a.get("distinctness", 0.0)) if isinstance(a, dict) else 0.0)
            - (float(b.get("distinctness", 0.0)) if isinstance(b, dict) else 0.0),
            "incident_count_before": float(brm.get("incident_count", 0.0)),
            "incident_count_after": float(arm.get("incident_count", 0.0)),
            "incident_count_delta": float(arm.get("incident_count", 0.0)) - float(brm.get("incident_count", 0.0)),
            "mean_strength_before": float(brm.get("mean_strength", 0.0)),
            "mean_strength_after": float(arm.get("mean_strength", 0.0)),
            "mean_strength_delta": float(arm.get("mean_strength", 0.0)) - float(brm.get("mean_strength", 0.0)),
            "activity_sum_before": float(brm.get("activity_sum", 0.0)),
            "activity_sum_after": float(arm.get("activity_sum", 0.0)),
            "activity_sum_delta": float(arm.get("activity_sum", 0.0)) - float(brm.get("activity_sum", 0.0)),
            "local_cluster_before": float(brm.get("local_cluster", 0.0)),
            "local_cluster_after": float(arm.get("local_cluster", 0.0)),
            "local_cluster_delta": float(arm.get("local_cluster", 0.0)) - float(brm.get("local_cluster", 0.0)),
            "sibling_count_norm_before": float(brm.get("sibling_count_norm", 0.0)),
            "sibling_count_norm_after": float(arm.get("sibling_count_norm", 0.0)),
            "sibling_count_norm_delta": float(arm.get("sibling_count_norm", 0.0)) - float(brm.get("sibling_count_norm", 0.0)),
        }
        site_rows.append(row)

    site_rows_sorted = sorted(site_rows, key=lambda r: r["weight_delta"])

    odiff_before = odiff_from_fingerprints(fps_before)
    odiff_after = odiff_from_fingerprints(fps_after)

    return {
        "region_nodes": [int(x) for x in region_nodes],
        "odiff_before": float(odiff_before),
        "odiff_after": float(odiff_after),
        "delta_odiff": float(odiff_after - odiff_before),
        "odiff_ratio": float(odiff_after / max(1e-12, odiff_before)) if odiff_before > 0 else 1.0,
        "site_rows": site_rows,
        "most_negative_sites": site_rows_sorted[:10],
        "most_positive_sites": list(reversed(sorted(site_rows, key=lambda r: r["weight_delta"])))[:10],
        "role_fingerprints_before": fps_before,
        "role_fingerprints_after": fps_after,
    }


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "case_counts": dict(Counter(r["case"] for r in rows)),
        "odiff_ratio_mean": _mean([r["odiff_ratio"] for r in rows]),
        "odiff_ratio_stdev": _stdev([r["odiff_ratio"] for r in rows]),
        "weight_after_mean": _mean([r["weight_after_mean"] for r in rows]),
        "activity_after_mean": _mean([r["activity_after_mean"] for r in rows]),
    }


def _classify(row: Dict[str, Any]) -> str:
    if row["lower_deltaF"] > 0.0 and row["odiff_ratio"] >= 0.98:
        return "lawful_high_retention"
    if row["lower_deltaF"] > 0.0:
        return "lawful_partial_retention"
    if row["odiff_ratio"] >= 0.98:
        return "odiff_retained_expr_bad"
    return "odiff_loss"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc Odiff site audit over a run JSON that already contains nonempty "
            "role_fingerprint data in move diagnostics."
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
        row = _build_row(step, best_lower, str(winner.get("move_type")))
        rows.append(row)

    late = rows[_late_start(len(rows)):] if rows else []
    top_nodes = [node for node, _ in Counter(r["node"] for r in late).most_common(5)]
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
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()