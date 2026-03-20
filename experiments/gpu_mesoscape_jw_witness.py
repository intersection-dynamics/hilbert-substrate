#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Node = int
Edge = Tuple[int, int]


def canon_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def build_adj(nodes: Iterable[Node], edges: Iterable[Edge]) -> Dict[Node, Set[Node]]:
    adj = {int(n): set() for n in nodes}
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def connected_components(nodes: Sequence[Node], edges: Sequence[Edge]) -> List[List[Node]]:
    adj = build_adj(nodes, edges)
    seen: Set[Node] = set()
    comps: List[List[Node]] = []
    for n in nodes:
        if n in seen:
            continue
        comp = []
        q = [n]
        seen.add(n)
        while q:
            x = q.pop()
            comp.append(x)
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        comps.append(sorted(comp))
    return comps


def shortest_path(adj: Dict[Node, Set[Node]], src: Node, dst: Node) -> Optional[List[Node]]:
    if src == dst:
        return [src]
    q = deque([src])
    parent = {src: None}
    while q:
        u = q.popleft()
        for v in sorted(adj.get(u, ())):
            if v in parent:
                continue
            parent[v] = u
            if v == dst:
                out = [dst]
                cur = dst
                while parent[cur] is not None:
                    cur = parent[cur]
                    out.append(cur)
                return list(reversed(out))
            q.append(v)
    return None


def bfs_farthest(adj: Dict[Node, Set[Node]], start: Node, allowed: Optional[Set[Node]] = None) -> Tuple[Node, int]:
    q = deque([(start, 0)])
    seen = {start}
    best = (start, 0)
    while q:
        u, d = q.popleft()
        if d > best[1] or (d == best[1] and u < best[0]):
            best = (u, d)
        for v in sorted(adj.get(u, ())):
            if allowed is not None and v not in allowed:
                continue
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return best


def graph_diameter_path(nodes: Sequence[Node], edges: Sequence[Edge]) -> List[Node]:
    if not nodes:
        return []
    if len(nodes) == 1:
        return [nodes[0]]
    adj = build_adj(nodes, edges)
    comps = connected_components(list(nodes), list(edges))
    best_path: List[Node] = [nodes[0]]
    best_len = -1
    for comp in comps:
        allowed = set(comp)
        start = min(comp)
        u, _ = bfs_farthest(adj, start, allowed)
        v, dist = bfs_farthest(adj, u, allowed)
        path = shortest_path(adj, u, v) or [u]
        if dist > best_len or (dist == best_len and path < best_path):
            best_len = dist
            best_path = path
    return best_path


def longest_core_through_path(nodes: Sequence[Node], edges: Sequence[Edge], core_pair: Optional[Sequence[int]]) -> List[Node]:
    if not nodes:
        return []
    if not core_pair or len(core_pair) != 2:
        return graph_diameter_path(nodes, edges)
    i, j = int(core_pair[0]), int(core_pair[1])
    adj = build_adj(nodes, edges)
    if i not in adj or j not in adj:
        return graph_diameter_path(nodes, edges)
    base = shortest_path(adj, i, j)
    if base is None:
        return graph_diameter_path(nodes, edges)
    allowed = set(nodes)
    left, _ = bfs_farthest(adj, i, allowed)
    right, _ = bfs_farthest(adj, j, allowed)
    left_path = shortest_path(adj, left, i) or [i]
    right_path = shortest_path(adj, j, right) or [j]
    merged = left_path + right_path[1:]
    # De-duplicate accidental cycles while preserving order.
    out: List[Node] = []
    seen: Set[Node] = set()
    for n in merged:
        if n not in seen:
            out.append(n)
            seen.add(n)
    if len(out) < len(base):
        return base
    return out


def induced_edges(nodes: Iterable[Node], edges: Iterable[Edge]) -> List[Edge]:
    node_set = set(int(n) for n in nodes)
    out = []
    for a, b in edges:
        e = canon_edge(int(a), int(b))
        if e[0] in node_set and e[1] in node_set:
            out.append(e)
    return sorted(set(out))


def line_graph_stats(path_nodes: Sequence[Node], shell_nodes: Sequence[Node], shell_edges: Sequence[Edge]) -> Dict[str, float]:
    path_nodes = [int(x) for x in path_nodes]
    shell_nodes = [int(x) for x in shell_nodes]
    shell_edges = [canon_edge(*e) for e in shell_edges]
    shell_adj = build_adj(shell_nodes, shell_edges)
    path_set = set(path_nodes)
    path_edges = {canon_edge(path_nodes[k], path_nodes[k + 1]) for k in range(len(path_nodes) - 1)}

    degrees = {n: len(shell_adj.get(n, ())) for n in shell_nodes}
    endpoint_count = sum(1 for n in path_nodes if len([m for m in shell_adj.get(n, ()) if m in path_set]) == 1)
    internal_good = 0
    internal_total = max(0, len(path_nodes) - 2)
    for n in path_nodes[1:-1]:
        along = sum(1 for m in shell_adj.get(n, ()) if m in path_set)
        if along == 2:
            internal_good += 1

    branch_edges = 0
    for n in path_nodes:
        for m in shell_adj.get(n, ()):
            if m not in path_set:
                branch_edges += 1
    branch_edges = branch_edges / 2.0

    shell_triangle_count = 0
    for i_idx, a in enumerate(shell_nodes):
        nbrs = shell_adj.get(a, set())
        for b in [x for x in nbrs if x > a]:
            common = [c for c in shell_adj.get(b, set()) if c > b and c in nbrs]
            shell_triangle_count += len(common)

    path_coverage = len(path_set) / max(1, len(shell_nodes))
    edge_concentration = len(path_edges) / max(1, len(shell_edges))
    endpoint_score = 1.0 if len(path_nodes) == 1 else max(0.0, 1.0 - abs(endpoint_count - 2) / 2.0)
    internal_score = (internal_good / internal_total) if internal_total > 0 else 1.0
    branch_penalty = branch_edges / max(1, len(path_edges)) if path_edges else 0.0
    triangle_penalty = shell_triangle_count / max(1, len(shell_edges)) if shell_edges else 0.0
    line_likeness = path_coverage * edge_concentration * endpoint_score * internal_score
    jw_string_score = line_likeness / (1.0 + branch_penalty + triangle_penalty)

    return {
        "path_node_count": float(len(path_nodes)),
        "path_edge_count": float(max(0, len(path_nodes) - 1)),
        "path_coverage": float(path_coverage),
        "path_edge_concentration": float(edge_concentration),
        "endpoint_score": float(endpoint_score),
        "internal_degree2_score": float(internal_score),
        "branch_penalty": float(branch_penalty),
        "triangle_penalty": float(triangle_penalty),
        "line_likeness": float(line_likeness),
        "jw_string_score": float(jw_string_score),
    }


def path_handoff(prev_path: Sequence[Node], curr_path: Sequence[Node]) -> Dict[str, float]:
    prev_set = set(int(x) for x in prev_path)
    curr_set = set(int(x) for x in curr_path)
    inter = len(prev_set.intersection(curr_set))
    union = len(prev_set.union(curr_set))
    jacc = inter / union if union else 1.0
    return {
        "path_overlap_count": float(inter),
        "path_jaccard": float(jacc),
        "core_on_prev_path": float(0.0),
    }


def path_distance_to_births(path_nodes: Sequence[Node], births_this_window: int, persistent_this_window: int) -> Dict[str, float]:
    # Placeholder for when topology-only JSON is all we have. Keep the structure explicit.
    return {
        "births_this_window": float(births_this_window),
        "persistent_this_window": float(persistent_this_window),
        "persistent_birth_rate": float(persistent_this_window / births_this_window) if births_this_window > 0 else 0.0,
        "birth_path_nucleation_proxy": float(persistent_this_window * max(1, len(path_nodes))),
    }


def summarize_series(rows: Sequence[Dict[str, float]], key: str) -> Dict[str, float]:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "mean_abs_step_change": 0.0}
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    diffs = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean_abs_step_change": float(sum(diffs) / len(diffs)) if diffs else 0.0,
    }


def analyze_snapshot(snap: Dict, prev: Optional[Dict] = None) -> Dict:
    active_nodes = [int(x) for x in snap.get("active_nodes", [])]
    active_edges = [canon_edge(int(a), int(b)) for a, b in snap.get("active_edges", [])]
    core = snap.get("dominant_core") or {}
    core_pair = [int(x) for x in core.get("core_pair", [])] if core.get("core_pair") else None
    shell_nodes = [int(x) for x in core.get("shell_nodes", [])] if core.get("shell_nodes") else list(active_nodes)
    shell_edges_raw = core.get("shell_edges")
    shell_edges = [canon_edge(int(a), int(b)) for a, b in shell_edges_raw] if shell_edges_raw else induced_edges(shell_nodes, active_edges)

    path_nodes = longest_core_through_path(shell_nodes, shell_edges, core_pair)
    path_edges = [list(canon_edge(path_nodes[k], path_nodes[k + 1])) for k in range(len(path_nodes) - 1)]
    line = line_graph_stats(path_nodes, shell_nodes, shell_edges)

    row = {
        "step": int(snap.get("step", 0)),
        "core_pair": list(core_pair) if core_pair else None,
        "shell_nodes": shell_nodes,
        "shell_edges": [list(e) for e in shell_edges],
        "ordered_backbone_nodes": path_nodes,
        "ordered_backbone_edges": path_edges,
        **line,
        **path_distance_to_births(path_nodes, int(snap.get("births_this_window", 0)), int(snap.get("persistent_this_window", 0))),
    }

    if prev is not None:
        hand = path_handoff(prev.get("ordered_backbone_nodes", []), path_nodes)
        row.update(hand)
        prev_path = set(int(x) for x in prev.get("ordered_backbone_nodes", []))
        core_on_prev = 0.0
        if core_pair:
            core_on_prev = float(sum(1 for x in core_pair if x in prev_path) / len(core_pair))
        row["core_on_prev_path"] = core_on_prev
    else:
        row.update({"path_overlap_count": 0.0, "path_jaccard": 0.0, "core_on_prev_path": 0.0})
    return row


def find_best_epoch(snapshots: Sequence[Dict], jw_rows: Sequence[Dict]) -> Dict:
    if not snapshots or not jw_rows:
        return {"epoch": None, "epoch_rows": [], "epoch_summary": {}}
    # Choose the contiguous epoch with the largest cumulative jw_string_score.
    best = None
    current = None
    for snap, row in zip(snapshots, jw_rows):
        core_pair = tuple(snap.get("dominant_core", {}).get("core_pair", [])) if snap.get("dominant_core") else None
        if current is None or current["core_pair"] != core_pair:
            if current is not None:
                score_sum = sum(r["jw_string_score"] for r in current["rows"])
                current["score_sum"] = score_sum
                if best is None or score_sum > best["score_sum"]:
                    best = current
            current = {"core_pair": core_pair, "start_step": row["step"], "rows": [row]}
        else:
            current["rows"].append(row)
    if current is not None:
        current["score_sum"] = sum(r["jw_string_score"] for r in current["rows"])
        if best is None or current["score_sum"] > best["score_sum"]:
            best = current
    if best is None:
        return {"epoch": None, "epoch_rows": [], "epoch_summary": {}}
    epoch_rows = best["rows"]
    epoch = {
        "core_pair": list(best["core_pair"]) if best["core_pair"] is not None else None,
        "start_step": int(best["start_step"]),
        "end_step": int(epoch_rows[-1]["step"]),
        "n_snapshots": int(len(epoch_rows)),
        "jw_string_score_sum": float(best["score_sum"]),
    }
    epoch_summary = {
        "jw_string_score": summarize_series(epoch_rows, "jw_string_score"),
        "line_likeness": summarize_series(epoch_rows, "line_likeness"),
        "path_coverage": summarize_series(epoch_rows, "path_coverage"),
        "path_edge_concentration": summarize_series(epoch_rows, "path_edge_concentration"),
        "path_jaccard": summarize_series(epoch_rows[1:], "path_jaccard") if len(epoch_rows) > 1 else summarize_series([], "path_jaccard"),
        "core_on_prev_path": summarize_series(epoch_rows[1:], "core_on_prev_path") if len(epoch_rows) > 1 else summarize_series([], "core_on_prev_path"),
    }
    return {"epoch": epoch, "epoch_rows": epoch_rows, "epoch_summary": epoch_summary}


def analyze_result(result: Dict) -> Dict:
    snapshots = result.get("snapshots", [])
    jw_rows: List[Dict] = []
    prev = None
    for snap in snapshots:
        row = analyze_snapshot(snap, prev)
        jw_rows.append(row)
        prev = row

    summary = {
        "jw_string_score": summarize_series(jw_rows, "jw_string_score"),
        "line_likeness": summarize_series(jw_rows, "line_likeness"),
        "path_coverage": summarize_series(jw_rows, "path_coverage"),
        "path_edge_concentration": summarize_series(jw_rows, "path_edge_concentration"),
        "branch_penalty": summarize_series(jw_rows, "branch_penalty"),
        "triangle_penalty": summarize_series(jw_rows, "triangle_penalty"),
        "path_jaccard": summarize_series(jw_rows[1:], "path_jaccard") if len(jw_rows) > 1 else summarize_series([], "path_jaccard"),
        "core_on_prev_path": summarize_series(jw_rows[1:], "core_on_prev_path") if len(jw_rows) > 1 else summarize_series([], "core_on_prev_path"),
    }

    best_epoch = find_best_epoch(snapshots, jw_rows)
    out = {
        "input_summary": result.get("summary", {}),
        "jw_witness_rows": jw_rows,
        "jw_witness_summary": summary,
        "best_jw_epoch": best_epoch["epoch"],
        "best_jw_epoch_summary": best_epoch["epoch_summary"],
        "best_jw_epoch_rows": best_epoch["epoch_rows"],
    }
    return out


def pretty_report(result: Dict, analysis: Dict, input_path: Path) -> str:
    base = result.get("summary", {})
    jw = analysis.get("jw_witness_summary", {})
    epoch = analysis.get("best_jw_epoch") or {}
    lines = []
    lines.append("=" * 100)
    lines.append("GPU MESOSCAPE JW WITNESS ANALYSIS")
    lines.append("-" * 100)
    lines.append(f"input_json={input_path}")
    lines.append(
        f"n_births={base.get('n_birth_events', 0)}  persistent={base.get('n_persistent_births', 0)}  "
        f"remerge={base.get('n_remerge_prone_births', 0)}  active_nodes_final={base.get('active_nodes_final', 0)}  "
        f"active_edges_final={base.get('active_edges_final', 0)}"
    )
    lines.append(
        f"longest_core={base.get('longest_lived_core')}  longest_life={base.get('longest_core_lifetime', 0)}  "
        f"core_switches={base.get('core_switch_count', 0)}"
    )
    lines.append("-" * 100)
    lines.append(
        f"jw_string_score: mean={jw['jw_string_score']['mean']:.4f}  std={jw['jw_string_score']['std']:.4f}  "
        f"max={jw['jw_string_score']['max']:.4f}  mean_abs_step_change={jw['jw_string_score']['mean_abs_step_change']:.4f}"
    )
    lines.append(
        f"line_likeness: mean={jw['line_likeness']['mean']:.4f}  path_coverage_mean={jw['path_coverage']['mean']:.4f}  "
        f"path_edge_concentration_mean={jw['path_edge_concentration']['mean']:.4f}"
    )
    lines.append(
        f"penalties: branch_mean={jw['branch_penalty']['mean']:.4f}  triangle_mean={jw['triangle_penalty']['mean']:.4f}  "
        f"path_jaccard_mean={jw['path_jaccard']['mean']:.4f}  core_on_prev_mean={jw['core_on_prev_path']['mean']:.4f}"
    )
    lines.append("-" * 100)
    if epoch:
        lines.append(
            f"best_jw_epoch: core={epoch.get('core_pair')}  start={epoch.get('start_step')}  end={epoch.get('end_step')}  "
            f"n_snapshots={epoch.get('n_snapshots')}  jw_score_sum={epoch.get('jw_string_score_sum', 0.0):.4f}"
        )
        e = analysis.get("best_jw_epoch_summary", {})
        if e:
            lines.append(
                f"epoch jw_string_score mean={e['jw_string_score']['mean']:.4f}  std={e['jw_string_score']['std']:.4f}  "
                f"line_likeness mean={e['line_likeness']['mean']:.4f}"
            )
            lines.append(
                f"epoch path_coverage mean={e['path_coverage']['mean']:.4f}  edge_concentration mean={e['path_edge_concentration']['mean']:.4f}  "
                f"epoch path_jaccard mean={e['path_jaccard']['mean']:.4f}"
            )
    lines.append("-" * 100)
    lines.append("Recent witness rows:")
    for row in analysis.get("jw_witness_rows", [])[:10]:
        lines.append(
            f"  step={row['step']:>3d}  core={row['core_pair']}  backbone={row['ordered_backbone_nodes']}  "
            f"jw={row['jw_string_score']:.4f}  line={row['line_likeness']:.4f}  cover={row['path_coverage']:.4f}  "
            f"edge_conc={row['path_edge_concentration']:.4f}  jacc={row['path_jaccard']:.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Passive JW/backbone witness analyzer for gpu_mesoscape_metric_analysis_v1 JSON outputs.")
    ap.add_argument("--input-json", required=True, help="Path to gpu_mesoscape_metric_analysis_v1 JSON output.")
    ap.add_argument("--json-out", default="", help="Optional output path for witness JSON.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json)
    with input_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    analysis = analyze_result(result)
    print(pretty_report(result, analysis, input_path))
    if args.json_out:
        payload = {
            "input_json": str(input_path),
            "analysis": analysis,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
