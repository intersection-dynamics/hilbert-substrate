#!/usr/bin/env python3
# filename: hsf_mesoscape_proto_stabilization_probe.py

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _sorted_pair(xs: Sequence[int]) -> Tuple[int, int]:
    a, b = int(xs[0]), int(xs[1])
    return (a, b) if a <= b else (b, a)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def step_for_index(idx: int, snapshots: List[Dict[str, Any]], eval_every: int) -> int:
    if 0 <= idx < len(snapshots):
        step = snapshots[idx].get("step")
        if step is not None:
            return _safe_int(step, (idx + 1) * eval_every)
    return (idx + 1) * eval_every


def parse_edges(edge_lists: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for e in edge_lists:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.append(_sorted_pair(e))
    return out


def neighbor_map(edges: Sequence[Tuple[int, int]]) -> Dict[int, set]:
    nbr: Dict[int, set] = defaultdict(set)
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    return nbr


def extract_target_from_run(data: Dict[str, Any]) -> Dict[str, Any]:
    proto_cfg = data.get("proto_config", {}) or {}
    pp = proto_cfg.get("proto_parent_pair")
    child = proto_cfg.get("proto_child_node")
    primary = proto_cfg.get("proto_primary_parent")

    out = {
        "parent_pair": list(pp) if isinstance(pp, list) and len(pp) == 2 else None,
        "child": _safe_int(child, -1),
        "primary_parent": _safe_int(primary, -1) if primary is not None else None,
    }

    if out["parent_pair"] is None or out["child"] < 0:
        # fallback: infer from first proto event
        for move in data.get("accepted_moves", []) or []:
            if move.get("move_type") == "proto":
                mo = move.get("move_object", {}) or {}
                pp2 = mo.get("parent_pair")
                child2 = mo.get("child")
                primary2 = mo.get("primary_parent")
                out["parent_pair"] = list(pp2) if isinstance(pp2, list) and len(pp2) == 2 else None
                out["child"] = _safe_int(child2, -1)
                out["primary_parent"] = _safe_int(primary2, -1) if primary2 is not None else None
                break
    return out


def collect_events(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    moves = data.get("accepted_moves", []) or []
    snapshots = data.get("snapshots", []) or []
    eval_every = _safe_int(data.get("physics_config", {}).get("eval_every"), 10)

    events: List[Dict[str, Any]] = []
    for idx, move in enumerate(moves):
        step = step_for_index(idx, snapshots, eval_every)
        mtype = move.get("move_type")
        rec = {
            "accepted_index": idx,
            "step": step,
            "move_type": mtype,
            "move_object": move.get("move_object"),
            "deltaF": _safe_float(move.get("deltaF")),
            "delta_Odiff_R": _safe_float(move.get("delta_Odiff_R")),
            "A_before_R": _safe_int(move.get("A_before_R")),
            "A_after_R": _safe_int(move.get("A_after_R")),
            "raw_move": move,
        }
        if mtype == "proto":
            mo = move.get("move_object", {}) or {}
            rec["parent_pair"] = list(mo.get("parent_pair", [])) if isinstance(mo.get("parent_pair"), list) else None
            rec["child"] = _safe_int(mo.get("child"))
            rec["primary_parent"] = _safe_int(mo.get("primary_parent"), -1)
            rec["proto_edge"] = list(mo.get("proto_edge", [])) if isinstance(mo.get("proto_edge"), list) else None
            rec["proto_readiness"] = _safe_float(move.get("proto_readiness"))
            rec["birth_parent_relief"] = _safe_float(move.get("birth_parent_relief"))
            rec["birth_novelty"] = _safe_float(move.get("birth_novelty"))
        elif mtype == "birth":
            mo = move.get("move_object", {}) or {}
            rec["parents"] = list(mo.get("parents", [])) if isinstance(mo.get("parents"), list) else None
            rec["child"] = _safe_int(mo.get("child"))
            rec["birth_variant"] = move.get("birth_variant")
        elif mtype == "weaken":
            mo = move.get("move_object")
            rec["edge"] = list(mo) if isinstance(mo, list) and len(mo) == 2 else None
            rec["lawful_shell_reexpression"] = bool(move.get("lawful_shell_reexpression", False))
            rec["core_weaken"] = bool(move.get("core_weaken", False))
            rec["shell_weaken"] = bool(move.get("shell_weaken", False))
            rec["W_NR"] = _safe_float(move.get("W_NR"))
        elif mtype == "retire":
            rec["node"] = _safe_int(move.get("move_object"))
        events.append(rec)
    return events


def find_proto_event(events: List[Dict[str, Any]], target_pp: Optional[Tuple[int, int]], target_child: int) -> Optional[Dict[str, Any]]:
    for ev in events:
        if ev.get("move_type") != "proto":
            continue
        pp = ev.get("parent_pair")
        child = _safe_int(ev.get("child"), -1)
        if target_pp is not None:
            if not isinstance(pp, list) or len(pp) != 2 or _sorted_pair(pp) != target_pp:
                continue
        if target_child >= 0 and child != target_child:
            continue
        return ev
    return None


def snapshot_occupancy_timeline(
    snapshots: List[Dict[str, Any]],
    proto_step: int,
    child: int,
) -> Dict[str, Any]:
    timeline = []
    first_present_after_proto = None
    first_absent_after_proto = None
    last_present_step = None
    n_present_post_proto = 0
    n_post_proto = 0

    for snap in snapshots:
        step = _safe_int(snap.get("step"))
        active_nodes = set(_safe_int(x) for x in (snap.get("active_nodes", []) or []))
        present = int(child) in active_nodes
        rec = {
            "step": step,
            "child_present": bool(present),
            "active_count": len(active_nodes),
        }
        timeline.append(rec)
        if step >= proto_step:
            n_post_proto += 1
            if present:
                n_present_post_proto += 1
                last_present_step = step
                if first_present_after_proto is None:
                    first_present_after_proto = step
            else:
                if first_absent_after_proto is None:
                    first_absent_after_proto = step

    occupancy_fraction = (n_present_post_proto / n_post_proto) if n_post_proto > 0 else 0.0
    return {
        "proto_step": int(proto_step),
        "child": int(child),
        "first_present_after_proto": first_present_after_proto,
        "first_absent_after_proto": first_absent_after_proto,
        "last_present_step": last_present_step,
        "post_proto_presence_fraction": float(occupancy_fraction),
        "timeline": timeline,
    }


def classify_edge_relation(edge: Tuple[int, int], target_pp: Optional[Tuple[int, int]], child: int) -> str:
    a, b = edge
    if child in edge:
        return "child_incident"
    if target_pp is not None and (target_pp[0] in edge or target_pp[1] in edge):
        return "parent_incident"
    return "other"


def summarize_post_proto_weakens(
    events: List[Dict[str, Any]],
    proto_step: int,
    target_pp: Optional[Tuple[int, int]],
    child: int,
) -> Dict[str, Any]:
    weakens = [ev for ev in events if ev.get("move_type") == "weaken" and _safe_int(ev.get("step")) > proto_step]

    relation_counter = Counter()
    edge_counter = Counter()
    lawful_shell_reexpression_count = 0
    shell_weaken_count = 0
    core_weaken_count = 0

    weaken_rows = []
    for ev in weakens:
        edge_list = ev.get("edge")
        if not isinstance(edge_list, list) or len(edge_list) != 2:
            continue
        edge = _sorted_pair(edge_list)
        rel = classify_edge_relation(edge, target_pp, child)
        relation_counter[rel] += 1
        edge_counter[edge] += 1
        if ev.get("lawful_shell_reexpression", False):
            lawful_shell_reexpression_count += 1
        if ev.get("shell_weaken", False):
            shell_weaken_count += 1
        if ev.get("core_weaken", False):
            core_weaken_count += 1

        weaken_rows.append(
            {
                "step": _safe_int(ev.get("step")),
                "edge": list(edge),
                "relation_to_proto_niche": rel,
                "deltaF": _safe_float(ev.get("deltaF")),
                "delta_Odiff_R": _safe_float(ev.get("delta_Odiff_R")),
                "lawful_shell_reexpression": bool(ev.get("lawful_shell_reexpression", False)),
                "shell_weaken": bool(ev.get("shell_weaken", False)),
                "core_weaken": bool(ev.get("core_weaken", False)),
            }
        )

    top_edges = [
        {"edge": list(edge), "count": int(count)}
        for edge, count in edge_counter.most_common(10)
    ]

    return {
        "n_post_proto_weakens": int(len(weaken_rows)),
        "relation_counts": dict(relation_counter),
        "lawful_shell_reexpression_count": int(lawful_shell_reexpression_count),
        "shell_weaken_count": int(shell_weaken_count),
        "core_weaken_count": int(core_weaken_count),
        "top_weaken_edges": top_edges,
        "weaken_events": weaken_rows,
    }


def summarize_post_proto_births(
    events: List[Dict[str, Any]],
    proto_step: int,
    target_pp: Optional[Tuple[int, int]],
    child: int,
) -> Dict[str, Any]:
    births = [ev for ev in events if ev.get("move_type") == "birth" and _safe_int(ev.get("step")) > proto_step]

    same_child_count = 0
    same_niche_count = 0
    same_child_rows = []

    for ev in births:
        c = _safe_int(ev.get("child"), -1)
        pp = ev.get("parents")
        if c == child:
            same_child_count += 1
            row = {
                "step": _safe_int(ev.get("step")),
                "parents": pp,
                "child": c,
                "birth_variant": ev.get("birth_variant"),
                "deltaF": _safe_float(ev.get("deltaF")),
                "delta_Odiff_R": _safe_float(ev.get("delta_Odiff_R")),
            }
            same_child_rows.append(row)
            if target_pp is not None and isinstance(pp, list) and len(pp) == 2 and _sorted_pair(pp) == target_pp:
                same_niche_count += 1

    return {
        "n_post_proto_births": int(len(births)),
        "n_post_proto_births_same_child": int(same_child_count),
        "n_post_proto_births_same_niche": int(same_niche_count),
        "same_child_birth_events": same_child_rows,
    }


def summarize_post_proto_retires(
    events: List[Dict[str, Any]],
    proto_step: int,
    child: int,
) -> Dict[str, Any]:
    retires = [ev for ev in events if ev.get("move_type") == "retire" and _safe_int(ev.get("step")) > proto_step]
    same_child = []
    for ev in retires:
        node = _safe_int(ev.get("node"), -1)
        if node == child:
            same_child.append(
                {
                    "step": _safe_int(ev.get("step")),
                    "node": node,
                    "deltaF": _safe_float(ev.get("deltaF")),
                    "delta_Odiff_R": _safe_float(ev.get("delta_Odiff_R")),
                }
            )
    return {
        "n_post_proto_retires": int(len(retires)),
        "n_post_proto_retires_same_child": int(len(same_child)),
        "same_child_retire_events": same_child,
    }


def summarize_final_niche_state(
    snapshots: List[Dict[str, Any]],
    target_pp: Optional[Tuple[int, int]],
    child: int,
) -> Dict[str, Any]:
    if not snapshots:
        return {}

    final_snap = snapshots[-1]
    final_active_nodes = set(_safe_int(x) for x in (final_snap.get("active_nodes", []) or []))
    final_edges = parse_edges(final_snap.get("active_edges", []) or [])
    final_nbr = neighbor_map(final_edges)

    child_present = int(child) in final_active_nodes
    child_neighbors = sorted(final_nbr.get(int(child), set())) if child_present else []

    parent_pair_edge_present = False
    parent_to_child_edges = []
    if target_pp is not None:
        parent_pair_edge_present = target_pp in set(final_edges)
        for p in target_pp:
            e = _sorted_pair((p, int(child)))
            if e in set(final_edges):
                parent_to_child_edges.append(list(e))

    return {
        "final_active_count": int(len(final_active_nodes)),
        "child_present_final": bool(child_present),
        "child_neighbors_final": [int(x) for x in child_neighbors],
        "parent_pair_edge_present_final": bool(parent_pair_edge_present),
        "parent_to_child_edges_final": parent_to_child_edges,
    }


def build_readout(
    target: Dict[str, Any],
    proto_event: Optional[Dict[str, Any]],
    occupancy: Dict[str, Any],
    weakens: Dict[str, Any],
    births: Dict[str, Any],
    retires: Dict[str, Any],
    final_state: Dict[str, Any],
) -> Dict[str, Any]:
    child = _safe_int(target.get("child"), -1)
    parent_pair = target.get("parent_pair")

    if proto_event is None:
        return {
            "dominant_read": "no_target_proto_event_found",
            "interpretation": "The targeted proto niche was not instantiated in this run.",
        }

    presence_fraction = _safe_float(occupancy.get("post_proto_presence_fraction"))
    n_same_child_births = _safe_int(births.get("n_post_proto_births_same_child"))
    n_same_child_retires = _safe_int(retires.get("n_post_proto_retires_same_child"))
    n_post_weakens = _safe_int(weakens.get("n_post_proto_weakens"))
    parent_incident_weakens = _safe_int((weakens.get("relation_counts", {}) or {}).get("parent_incident"))
    child_incident_weakens = _safe_int((weakens.get("relation_counts", {}) or {}).get("child_incident"))
    lawful_shell_reexpression_count = _safe_int(weakens.get("lawful_shell_reexpression_count"))

    if presence_fraction >= 0.9 and n_same_child_births == 0 and n_same_child_retires == 0 and n_post_weakens > 0:
        dominant_read = "stable_proto_child_with_post_proto_weaken_flow"
    elif presence_fraction >= 0.75 and n_same_child_births <= 1 and n_same_child_retires <= 1:
        dominant_read = "mostly_stable_proto_child"
    elif n_same_child_births > 0 or n_same_child_retires > 0:
        dominant_read = "proto_child_not_fully_stabilized"
    else:
        dominant_read = "ambiguous_proto_outcome"

    interpretation_lines = []
    interpretation_lines.append(
        f"Target niche {parent_pair} -> {child} accepted a proto move at step {_safe_int(proto_event.get('step'))}."
    )
    interpretation_lines.append(
        f"Child {child} was present for {presence_fraction:.3f} of post-proto snapshots."
    )
    interpretation_lines.append(
        f"Post-proto same-child births: {n_same_child_births}; post-proto same-child retires: {n_same_child_retires}."
    )
    interpretation_lines.append(
        f"Post-proto weakens: {n_post_weakens}, with parent-incident={parent_incident_weakens}, child-incident={child_incident_weakens}, lawful shell re-expression={lawful_shell_reexpression_count}."
    )

    if dominant_read == "stable_proto_child_with_post_proto_weaken_flow":
        interpretation_lines.append(
            "The old birth/retire niche appears to have regularized into a persistent proto-child carrier, with later adjustment handed off to weaken-mediated shell/interface flow."
        )
    elif dominant_read == "mostly_stable_proto_child":
        interpretation_lines.append(
            "The proto-child mostly stabilized the niche, though there may still be minor residual retuning."
        )
    elif dominant_read == "proto_child_not_fully_stabilized":
        interpretation_lines.append(
            "The proto-child changed the niche, but the same child still re-entered churn after proto instantiation."
        )
    else:
        interpretation_lines.append(
            "The proto-child effect is real but not yet cleanly separable from the rest of the dynamics."
        )

    return {
        "dominant_read": dominant_read,
        "interpretation": " ".join(interpretation_lines),
    }


def print_human_summary(report: Dict[str, Any]) -> None:
    target = report["target_niche"]
    proto = report["proto_event"]
    occ = report["proto_child_occupancy"]
    weak = report["post_proto_weaken_flow"]
    births = report["post_proto_birth_flow"]
    retires = report["post_proto_retire_flow"]
    final_state = report["final_niche_state"]
    readout = report["readout"]

    print("=== Proto Stabilization Probe ===")
    print(f"target parent pair: {target.get('parent_pair')}")
    print(f"target child:       {target.get('child')}")
    print(f"dominant read:      {readout.get('dominant_read')}")
    print()

    if proto is None:
        print("No target proto event found.")
        return

    print(f"proto accepted step:          {proto.get('step')}")
    print(f"proto deltaF:                 {_safe_float(proto.get('deltaF')):.6f}")
    print(f"proto delta_Odiff_R:          {_safe_float(proto.get('delta_Odiff_R')):.6f}")
    print(f"proto readiness:              {_safe_float(proto.get('proto_readiness')):.6f}")
    print()

    print(f"post-proto presence fraction: {_safe_float(occ.get('post_proto_presence_fraction')):.3f}")
    print(f"first absent after proto:     {occ.get('first_absent_after_proto')}")
    print(f"last present step:            {occ.get('last_present_step')}")
    print()

    print(f"post-proto same-child births: {births.get('n_post_proto_births_same_child')}")
    print(f"post-proto same-niche births: {births.get('n_post_proto_births_same_niche')}")
    print(f"post-proto same-child retires:{retires.get('n_post_proto_retires_same_child')}")
    print()

    print(f"post-proto weakens:           {weak.get('n_post_proto_weakens')}")
    print(f"weakens by relation:          {weak.get('relation_counts')}")
    print(f"lawful shell reexpression:    {weak.get('lawful_shell_reexpression_count')}")
    print(f"top weaken edges:             {weak.get('top_weaken_edges')}")
    print()

    print(f"child present final:          {final_state.get('child_present_final')}")
    print(f"child neighbors final:        {final_state.get('child_neighbors_final')}")
    print(f"parent->child edges final:    {final_state.get('parent_to_child_edges_final')}")
    print()
    print(readout.get("interpretation"))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether a proto-child move stabilized a niche and redirected later dynamics into weaken flow."
        )
    )
    parser.add_argument("json_path", help="Path to hsf_mesoscape_proto_*.json")
    parser.add_argument(
        "--parent-pair",
        nargs=2,
        type=int,
        default=None,
        help="Optional explicit target parent pair, e.g. --parent-pair 2 3",
    )
    parser.add_argument(
        "--child",
        type=int,
        default=None,
        help="Optional explicit target child, e.g. --child 5",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path. Default: <input>_proto_probe.json",
    )
    args = parser.parse_args()

    in_path = Path(args.json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = load_json(in_path)
    target = extract_target_from_run(data)

    if args.parent_pair is not None:
        target["parent_pair"] = list(_sorted_pair(args.parent_pair))
    if args.child is not None:
        target["child"] = int(args.child)

    target_pp = _sorted_pair(target["parent_pair"]) if isinstance(target.get("parent_pair"), list) and len(target["parent_pair"]) == 2 else None
    target_child = _safe_int(target.get("child"), -1)

    events = collect_events(data)
    snapshots = data.get("snapshots", []) or []

    proto_event = find_proto_event(events, target_pp, target_child)
    proto_step = _safe_int(proto_event.get("step")) if proto_event is not None else -1

    occupancy = snapshot_occupancy_timeline(snapshots, proto_step, target_child) if proto_event is not None else {}
    weakens = summarize_post_proto_weakens(events, proto_step, target_pp, target_child) if proto_event is not None else {}
    births = summarize_post_proto_births(events, proto_step, target_pp, target_child) if proto_event is not None else {}
    retires = summarize_post_proto_retires(events, proto_step, target_child) if proto_event is not None else {}
    final_state = summarize_final_niche_state(snapshots, target_pp, target_child)
    readout = build_readout(target, proto_event, occupancy, weakens, births, retires, final_state)

    report = {
        "script": "hsf_mesoscape_proto_stabilization_probe.py",
        "input_json": str(in_path),
        "target_niche": target,
        "proto_event": proto_event,
        "proto_child_occupancy": occupancy,
        "post_proto_weaken_flow": weakens,
        "post_proto_birth_flow": births,
        "post_proto_retire_flow": retires,
        "final_niche_state": final_state,
        "readout": readout,
    }

    out_path = Path(args.json_out) if args.json_out else in_path.with_name(in_path.stem + "_proto_probe.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_human_summary(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()