#!/usr/bin/env python3
# filename: hsf_mesoscape_graded_constraint_probe.py

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


def _sorted_edge(x: Sequence[int]) -> Tuple[int, int]:
    a, b = int(x[0]), int(x[1])
    return (a, b) if a <= b else (b, a)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _neighbor_map(edges: Sequence[Tuple[int, int]]) -> Dict[int, set]:
    nbr: Dict[int, set] = defaultdict(set)
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    return nbr


def _find_snapshot_for_step(snapshots: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    for snap in snapshots:
        if _safe_int(snap.get("step")) == int(step):
            return snap
    return None


def _extract_sigma(snap: Dict[str, Any]) -> List[float]:
    return [float(x) for x in (((snap.get("sigma_summary") or {}).get("sigma")) or [])]


def _extract_edges(snap: Dict[str, Any]) -> List[Tuple[int, int]]:
    edges = (snap.get("active_edges") or [])
    out: List[Tuple[int, int]] = []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.append(_sorted_edge(e))
    return out


def _extract_commitments(snap: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
    top = ((snap.get("commitment_summary") or {}).get("top_interfaces")) or []
    out: Dict[Tuple[int, int], float] = {}
    for row in top:
        edge = row.get("edge")
        if isinstance(edge, list) and len(edge) == 2:
            out[_sorted_edge(edge)] = _safe_float(row.get("commitment"))
    return out


def _find_first_full_occupancy_step(snapshots: List[Dict[str, Any]], n_max: int) -> Optional[int]:
    for snap in snapshots:
        sigma = _extract_sigma(snap)
        active_count = sum(1 for s in sigma if s > 0.0)
        if active_count >= int(n_max):
            return _safe_int(snap.get("step"))
    return None


def _classify_sigma(sigma: Sequence[float], full_thresh: float = 0.999, partial_thresh: float = 1e-9) -> Dict[str, Any]:
    full = [i for i, s in enumerate(sigma) if float(s) >= full_thresh]
    partial = [i for i, s in enumerate(sigma) if partial_thresh < float(s) < full_thresh]
    zero = [i for i, s in enumerate(sigma) if float(s) <= partial_thresh]
    return {
        "full_nodes": full,
        "partial_nodes": partial,
        "zero_nodes": zero,
        "n_full": len(full),
        "n_partial": len(partial),
        "n_zero": len(zero),
        "mean_sigma": _mean([float(s) for s in sigma]),
        "partial_sigma_mean": _mean([float(sigma[i]) for i in partial]),
        "partial_sigma_stdev": _stdev([float(sigma[i]) for i in partial]),
    }


def _compute_bandwidth_probe(snap: Dict[str, Any]) -> Dict[str, Any]:
    sigma = _extract_sigma(snap)
    sigma_info = _classify_sigma(sigma)
    edges = _extract_edges(snap)
    nbr = _neighbor_map(edges)

    partial_nodes = sigma_info["partial_nodes"]
    degrees = [len(nbr.get(i, set())) for i in partial_nodes]
    degree_mean = _mean([float(d) for d in degrees])
    degree_stdev = _stdev([float(d) for d in degrees])

    # crowding proxy: many partial nodes plus moderately high degree
    crowding = float(sigma_info["n_partial"]) * max(0.0, degree_mean)
    # concentration proxy: if support mass is spread across many partial nodes, concentration is low
    support_mass = sum(float(s) for s in sigma)
    partial_mass = sum(float(sigma[i]) for i in partial_nodes)
    diffuse_fraction = float(partial_mass / support_mass) if support_mass > 1e-12 else 0.0

    return {
        "n_partial_nodes": int(sigma_info["n_partial"]),
        "partial_nodes": [int(i) for i in partial_nodes],
        "partial_degree_mean": float(degree_mean),
        "partial_degree_stdev": float(degree_stdev),
        "partial_support_mass": float(partial_mass),
        "total_support_mass": float(support_mass),
        "diffuse_partial_fraction": float(diffuse_fraction),
        "finite_bandwidth_crowding_proxy": float(crowding),
    }


def _compute_norefolding_probe(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not snapshots:
        return {
            "late_edge_jaccard_mean": 0.0,
            "late_edge_jaccard_min": 0.0,
            "late_sigma_l1_mean": 0.0,
            "late_sigma_l1_min": 0.0,
        }

    # analyze last third of run
    start = max(0, len(snapshots) * 2 // 3)
    late = snapshots[start:]
    edge_jaccards = []
    sigma_l1s = []

    for a, b in zip(late[:-1], late[1:]):
        ea = set(_extract_edges(a))
        eb = set(_extract_edges(b))
        union = ea | eb
        inter = ea & eb
        jacc = float(len(inter) / len(union)) if union else 1.0
        edge_jaccards.append(jacc)

        sa = _extract_sigma(a)
        sb = _extract_sigma(b)
        m = min(len(sa), len(sb))
        l1 = sum(abs(float(sa[i]) - float(sb[i])) for i in range(m))
        sigma_l1s.append(float(l1))

    return {
        "late_edge_jaccard_mean": _mean(edge_jaccards),
        "late_edge_jaccard_min": min(edge_jaccards) if edge_jaccards else 0.0,
        "late_sigma_l1_mean": _mean(sigma_l1s),
        "late_sigma_l1_max": max(sigma_l1s) if sigma_l1s else 0.0,
    }


def _compute_noforgetting_probe(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not snapshots:
        return {
            "first_full_step": None,
            "post_full_partial_persistence_fraction": 0.0,
            "post_full_full_node_stability_fraction": 0.0,
        }

    n_max = len(_extract_sigma(snapshots[0]))
    first_full_step = _find_first_full_occupancy_step(snapshots, n_max)
    if first_full_step is None:
        return {
            "first_full_step": None,
            "post_full_partial_persistence_fraction": 0.0,
            "post_full_full_node_stability_fraction": 0.0,
        }

    snap0 = _find_snapshot_for_step(snapshots, first_full_step)
    if snap0 is None:
        return {
            "first_full_step": first_full_step,
            "post_full_partial_persistence_fraction": 0.0,
            "post_full_full_node_stability_fraction": 0.0,
        }

    sigma0 = _extract_sigma(snap0)
    cls0 = _classify_sigma(sigma0)
    partial0 = set(cls0["partial_nodes"])
    full0 = set(cls0["full_nodes"])

    relevant = [snap for snap in snapshots if _safe_int(snap.get("step")) >= int(first_full_step)]
    partial_presence_scores = []
    full_presence_scores = []

    for snap in relevant:
        sigma = _extract_sigma(snap)
        current_partial_or_full = {i for i, s in enumerate(sigma) if float(s) > 1e-9}
        current_full = {i for i, s in enumerate(sigma) if float(s) >= 0.999}

        if partial0:
            partial_presence_scores.append(len(partial0 & current_partial_or_full) / len(partial0))
        if full0:
            full_presence_scores.append(len(full0 & current_full) / len(full0))

    return {
        "first_full_step": int(first_full_step),
        "initial_partial_nodes_at_full": sorted(int(i) for i in partial0),
        "initial_full_nodes_at_full": sorted(int(i) for i in full0),
        "post_full_partial_persistence_fraction": _mean(partial_presence_scores),
        "post_full_full_node_stability_fraction": _mean(full_presence_scores),
    }


def _compute_nosignalling_probe(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(snapshots) < 2:
        return {
            "late_move_type_counts": {},
            "late_edge_up_fraction": 0.0,
            "late_edge_down_fraction": 0.0,
            "late_raise_support_fraction": 0.0,
            "late_lower_support_fraction": 0.0,
        }

    # derive accepted winner per snapshot from counters
    start = max(0, len(snapshots) * 2 // 3)
    late = snapshots[start:]
    counter = Counter()

    for snap in late:
        if _safe_int(snap.get("n_edge_up_this_eval")) > 0:
            counter["edge_up"] += 1
        elif _safe_int(snap.get("n_edge_down_this_eval")) > 0:
            counter["edge_down"] += 1
        elif _safe_int(snap.get("n_raise_support_this_eval")) > 0:
            counter["raise_support"] += 1
        elif _safe_int(snap.get("n_lower_support_this_eval")) > 0:
            counter["lower_support"] += 1
        else:
            counter["none"] += 1

    total = sum(counter.values()) or 1
    return {
        "late_move_type_counts": dict(counter),
        "late_edge_up_fraction": float(counter.get("edge_up", 0) / total),
        "late_edge_down_fraction": float(counter.get("edge_down", 0) / total),
        "late_raise_support_fraction": float(counter.get("raise_support", 0) / total),
        "late_lower_support_fraction": float(counter.get("lower_support", 0) / total),
    }


def _constraint_readout(
    bandwidth: Dict[str, Any],
    noref: Dict[str, Any],
    noforget: Dict[str, Any],
    nosig: Dict[str, Any],
) -> Dict[str, Any]:
    # These are interpretive summaries, not new penalties.
    reads = []

    if _safe_float(bandwidth.get("diffuse_partial_fraction")) > 0.45:
        reads.append("finite_bandwidth is the strongest live candidate for why support mass stays broadly distributed across many partial carriers")

    if _safe_float(noref.get("late_edge_jaccard_mean")) > 0.70 and _safe_float(noref.get("late_sigma_l1_mean")) < 1.0:
        reads.append("no_refolding likely contributes to persistent late-time support geometry once a partial network has formed")

    if _safe_float(noforget.get("post_full_partial_persistence_fraction")) > 0.85:
        reads.append("no_forgetting-like persistence is visible in the survival of partial carriers after full-site occupancy is reached")

    if _safe_float(nosig.get("late_edge_up_fraction")) + _safe_float(nosig.get("late_edge_down_fraction")) > 0.60:
        reads.append("no_signalling-compatible retuning is most visible in late-time dynamics shifting into local interface adjustments instead of further support creation")

    if not reads:
        reads.append("no single constraint signature dominates strongly in this run")

    return {
        "dominant_constraint_reads": reads
    }


def print_summary(report: Dict[str, Any]) -> None:
    bw = report["finite_bandwidth_probe"]
    nr = report["no_refolding_probe"]
    nf = report["no_forgetting_probe"]
    ns = report["no_signalling_probe"]

    print("=== Graded Constraint Probe ===")
    print(f"final partial nodes:               {bw['partial_nodes']}")
    print(f"diffuse partial fraction:          {bw['diffuse_partial_fraction']:.4f}")
    print(f"finite-bandwidth crowding proxy:   {bw['finite_bandwidth_crowding_proxy']:.4f}")
    print()

    print(f"late edge jaccard mean:            {nr['late_edge_jaccard_mean']:.4f}")
    print(f"late sigma L1 mean:                {nr['late_sigma_l1_mean']:.4f}")
    print()

    print(f"first full occupancy step:         {nf['first_full_step']}")
    print(f"post-full partial persistence:     {nf['post_full_partial_persistence_fraction']:.4f}")
    print(f"post-full full-node stability:     {nf['post_full_full_node_stability_fraction']:.4f}")
    print()

    print(f"late move counts:                  {ns['late_move_type_counts']}")
    print(f"late edge-up fraction:             {ns['late_edge_up_fraction']:.4f}")
    print(f"late edge-down fraction:           {ns['late_edge_down_fraction']:.4f}")
    print(f"late raise-support fraction:       {ns['late_raise_support_fraction']:.4f}")
    print(f"late lower-support fraction:       {ns['late_lower_support_fraction']:.4f}")
    print()

    print("Interpretive read:")
    for line in report["constraint_readout"]["dominant_constraint_reads"]:
        print(f"  - {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a graded-support sandbox run in terms of the four allowed constraint pressures: "
            "no-signalling, no-forgetting, no-refolding, and finite bandwidth."
        )
    )
    parser.add_argument("json_path", help="Path to hsf_mesoscape_*_graded_support.json")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_constraint_probe.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = _load_json(in_path)
    snapshots = data.get("snapshots", []) or []

    bandwidth = _compute_bandwidth_probe(snapshots[-1] if snapshots else {})
    noref = _compute_norefolding_probe(snapshots)
    noforget = _compute_noforgetting_probe(snapshots)
    nosig = _compute_nosignalling_probe(snapshots)
    readout = _constraint_readout(bandwidth, noref, noforget, nosig)

    report = {
        "script": "hsf_mesoscape_graded_constraint_probe.py",
        "input_json": str(in_path),
        "finite_bandwidth_probe": bandwidth,
        "no_refolding_probe": noref,
        "no_forgetting_probe": noforget,
        "no_signalling_probe": nosig,
        "constraint_readout": readout,
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_constraint_probe.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()