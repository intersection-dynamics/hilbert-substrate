#!/usr/bin/env python3
"""
gpu_mesoscape_collective_mode_analysis.py

Passive post-processing for gpu_mesoscape_metric_analysis_v1.py JSON outputs.

Goal:
    Analyze whether the dominant mesoscopic organizer behaves more like a
    collective particle-like mode (persistent relational pattern sustained by
    multiple subsystems) than a single fixed microscopic object.

What it measures:
    - Epochs of dominant-core stability
    - Support persistence vs member turnover
    - Organizer topology class: line / star / triangle-rich / mesh-like
    - Bridge, articulation, branch, and simpliciality diagnostics
    - Birth localization near the organizer
    - Collective identity score: pattern persistence despite microscopic churn

This script does NOT rerun dynamics. It only reads an existing mesoscape JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Node = int
Edge = Tuple[int, int]


def canon_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def to_edge_set(edges: Iterable[Sequence[int]]) -> Set[Edge]:
    out: Set[Edge] = set()
    for e in edges:
        if len(e) != 2:
            continue
        out.add(canon_edge(int(e[0]), int(e[1])))
    return out


def to_node_set(nodes: Iterable[int]) -> Set[Node]:
    return {int(x) for x in nodes}


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def safe_mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def safe_std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = safe_mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for v in counter.values():
        p = v / total
        if p > 0:
            out -= p * math.log(p + 1e-15)
    return out


class SimpleGraph:
    def __init__(self, nodes: Iterable[Node], edges: Iterable[Edge]):
        self.nodes: Set[Node] = set(nodes)
        self.edges: Set[Edge] = set()
        self.adj: Dict[Node, Set[Node]] = {n: set() for n in self.nodes}
        for a, b in edges:
            if a == b:
                continue
            if a not in self.nodes or b not in self.nodes:
                continue
            e = canon_edge(a, b)
            self.edges.add(e)
            self.adj[a].add(b)
            self.adj[b].add(a)

    def degree(self, n: Node) -> int:
        return len(self.adj.get(n, ()))

    def degrees(self) -> Dict[Node, int]:
        return {n: len(self.adj.get(n, ())) for n in self.nodes}

    def components(self) -> List[Set[Node]]:
        seen: Set[Node] = set()
        comps: List[Set[Node]] = []
        for n in self.nodes:
            if n in seen:
                continue
            stack = [n]
            comp: Set[Node] = set()
            seen.add(n)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in self.adj.get(u, ()): 
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
        return comps

    def induced_subgraph(self, nodes: Set[Node]) -> "SimpleGraph":
        return SimpleGraph(nodes, [e for e in self.edges if e[0] in nodes and e[1] in nodes])

    def bridges(self) -> Set[Edge]:
        timer = 0
        tin: Dict[Node, int] = {}
        low: Dict[Node, int] = {}
        visited: Set[Node] = set()
        out: Set[Edge] = set()

        def dfs(v: Node, p: Optional[Node]) -> None:
            nonlocal timer
            visited.add(v)
            tin[v] = timer
            low[v] = timer
            timer += 1
            for to in self.adj.get(v, ()): 
                if to == p:
                    continue
                if to in visited:
                    low[v] = min(low[v], tin[to])
                else:
                    dfs(to, v)
                    low[v] = min(low[v], low[to])
                    if low[to] > tin[v]:
                        out.add(canon_edge(v, to))

        for n in self.nodes:
            if n not in visited:
                dfs(n, None)
        return out

    def articulations(self) -> Set[Node]:
        timer = 0
        tin: Dict[Node, int] = {}
        low: Dict[Node, int] = {}
        visited: Set[Node] = set()
        out: Set[Node] = set()

        def dfs(v: Node, p: Optional[Node]) -> None:
            nonlocal timer
            visited.add(v)
            tin[v] = timer
            low[v] = timer
            timer += 1
            children = 0
            for to in self.adj.get(v, ()): 
                if to == p:
                    continue
                if to in visited:
                    low[v] = min(low[v], tin[to])
                else:
                    dfs(to, v)
                    low[v] = min(low[v], low[to])
                    if p is not None and low[to] >= tin[v]:
                        out.add(v)
                    children += 1
            if p is None and children > 1:
                out.add(v)

        for n in self.nodes:
            if n not in visited:
                dfs(n, None)
        return out

    def triangles(self) -> List[Tuple[Node, Node, Node]]:
        nodes = sorted(self.nodes)
        tri: List[Tuple[Node, Node, Node]] = []
        for i, a in enumerate(nodes):
            na = self.adj.get(a, set())
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                if b not in na:
                    continue
                common = na & self.adj.get(b, set())
                for c in common:
                    if c > b:
                        tri.append((a, b, c))
        return tri

    def shortest_path(self, start: Node, goal: Node) -> List[Node]:
        if start == goal:
            return [start]
        q = [start]
        prev = {start: None}
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in sorted(self.adj.get(u, ())):
                if v in prev:
                    continue
                prev[v] = u
                if v == goal:
                    path = [goal]
                    cur = goal
                    while prev[cur] is not None:
                        cur = prev[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                q.append(v)
        return []

    def diameter_path(self) -> List[Node]:
        if not self.nodes:
            return []
        comps = self.components()
        comp = max(comps, key=len)
        g = self.induced_subgraph(comp)
        best: List[Node] = []
        comp_nodes = sorted(comp)
        for i, a in enumerate(comp_nodes):
            for b in comp_nodes[i + 1 :]:
                p = g.shortest_path(a, b)
                if len(p) > len(best):
                    best = p
        if not best and comp_nodes:
            return [comp_nodes[0]]
        return best


@dataclass
class SnapshotOrganizer:
    step: int
    core_pair: Tuple[int, int]
    nodes: Set[Node]
    edges: Set[Edge]
    core_nodes: Set[Node]
    shell_only_nodes: Set[Node]
    graph: SimpleGraph
    metric_coords: Dict[Node, float]
    births_this_window: int
    persistent_this_window: int
    remerge_this_window: int


def organizer_from_snapshot(snap: dict) -> Optional[SnapshotOrganizer]:
    dom = snap.get("dominant_core") or {}
    core_pair = dom.get("core_pair")
    if not core_pair or len(core_pair) != 2:
        return None
    nodes = to_node_set(dom.get("shell_nodes", []))
    core_nodes = to_node_set(core_pair)
    nodes |= core_nodes
    edges = to_edge_set(dom.get("shell_edges", []))
    cp = canon_edge(core_pair[0], core_pair[1])
    if cp[0] in nodes and cp[1] in nodes:
        edges.add(cp)
    coords_raw = (snap.get("metric") or {}).get("coords", {})
    coords = {int(k): float(v) for k, v in coords_raw.items()}
    graph = SimpleGraph(nodes, edges)
    return SnapshotOrganizer(
        step=int(snap.get("step", 0)),
        core_pair=cp,
        nodes=nodes,
        edges=edges,
        core_nodes=core_nodes,
        shell_only_nodes=nodes - core_nodes,
        graph=graph,
        metric_coords=coords,
        births_this_window=int(snap.get("births_this_window", 0)),
        persistent_this_window=int(snap.get("persistent_this_window", 0)),
        remerge_this_window=int(snap.get("remerge_this_window", 0)),
    )


def nearest_distance_to_set(x: int, support: Set[int], graph: SimpleGraph) -> Optional[int]:
    if x in support:
        return 0
    q = [x]
    dist = {x: 0}
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in graph.adj.get(u, ()): 
            if v in dist:
                continue
            dist[v] = dist[u] + 1
            if v in support:
                return dist[v]
            q.append(v)
    return None


def classify_topology(g: SimpleGraph) -> str:
    n = len(g.nodes)
    m = len(g.edges)
    if n <= 1:
        return "degenerate"
    deg = g.degrees()
    max_deg = max(deg.values(), default=0)
    triangles = len(g.triangles())
    branch_nodes = sum(1 for d in deg.values() if d >= 3)
    leaves = sum(1 for d in deg.values() if d == 1)
    density = 2.0 * m / max(1, n * (n - 1))
    diameter_path = g.diameter_path()
    path_edges = max(0, len(diameter_path) - 1)
    line_likeness = path_edges / max(1, m)
    star_likeness = max_deg / max(1, n - 1)
    triangle_density = triangles / max(1, m)

    if n >= 3 and line_likeness > 0.8 and branch_nodes <= 1 and triangle_density < 0.2:
        return "corridor"
    if n >= 4 and star_likeness > 0.7 and branch_nodes >= 1:
        return "hub"
    if triangles > 0 and triangle_density >= 0.2:
        return "triangulated"
    if density > 0.5:
        return "mesh"
    if leaves >= 2 and branch_nodes == 0:
        return "path_fragment"
    return "mixed"


def snapshot_metrics(org: SnapshotOrganizer) -> dict:
    g = org.graph
    n = len(g.nodes)
    m = len(g.edges)
    deg = g.degrees()
    triangles = g.triangles()
    bridges = g.bridges()
    arts = g.articulations()
    comps = g.components()
    cycle_rank = m - n + len(comps)
    diameter_path = g.diameter_path()
    path_nodes = set(diameter_path)
    path_edges = {canon_edge(diameter_path[i], diameter_path[i + 1]) for i in range(len(diameter_path) - 1)}
    line_likeness = len(path_edges) / max(1, m)
    star_likeness = max(deg.values(), default=0) / max(1, n - 1)
    triangle_density = len(triangles) / max(1, m)
    branch_fraction = sum(1 for d in deg.values() if d >= 3) / max(1, n)
    leaf_fraction = sum(1 for d in deg.values() if d == 1) / max(1, n)
    bridge_fraction = len(bridges) / max(1, m)
    articulation_fraction = len(arts) / max(1, n)
    support_span = 0.0
    if org.metric_coords:
        vals = [org.metric_coords.get(n, 0.0) for n in g.nodes if n in org.metric_coords]
        if vals:
            support_span = float(max(vals) - min(vals))
    hub = None
    if deg:
        hub = max(sorted(deg), key=lambda k: deg[k])
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": 2.0 * m / max(1, n * (n - 1)),
        "mean_degree": safe_mean(list(deg.values())),
        "max_degree": max(deg.values(), default=0),
        "hub_node": hub,
        "cycle_rank": cycle_rank,
        "n_components": len(comps),
        "triangle_count": len(triangles),
        "triangle_density": triangle_density,
        "bridge_fraction": bridge_fraction,
        "articulation_fraction": articulation_fraction,
        "branch_fraction": branch_fraction,
        "leaf_fraction": leaf_fraction,
        "diameter_path": diameter_path,
        "path_edge_count": len(path_edges),
        "line_likeness": line_likeness,
        "star_likeness": star_likeness,
        "support_span": support_span,
        "topology_class": classify_topology(g),
        "path_nodes": sorted(path_nodes),
        "path_edges": [list(e) for e in sorted(path_edges)],
    }


def build_epochs(orgs: List[SnapshotOrganizer]) -> List[dict]:
    if not orgs:
        return []
    epochs: List[List[SnapshotOrganizer]] = []
    cur = [orgs[0]]
    for org in orgs[1:]:
        if org.core_pair == cur[-1].core_pair:
            cur.append(org)
        else:
            epochs.append(cur)
            cur = [org]
    epochs.append(cur)

    out: List[dict] = []
    for idx, group in enumerate(epochs):
        node_sets = [g.nodes for g in group]
        edge_sets = [g.edges for g in group]
        core = group[0].core_pair
        union_nodes = set().union(*node_sets)
        union_edges = set().union(*edge_sets)
        inter_nodes = set(node_sets[0])
        inter_edges = set(edge_sets[0])
        for s in node_sets[1:]:
            inter_nodes &= s
        for s in edge_sets[1:]:
            inter_edges &= s
        node_j = [jaccard(node_sets[i - 1], node_sets[i]) for i in range(1, len(node_sets))]
        edge_j = [jaccard(edge_sets[i - 1], edge_sets[i]) for i in range(1, len(edge_sets))]
        topo_counts = Counter(snapshot_metrics(g)["topology_class"] for g in group)
        births = sum(g.births_this_window for g in group)
        persistent = sum(g.persistent_this_window for g in group)
        remerge = sum(g.remerge_this_window for g in group)
        member_turnover = 1.0 - (len(inter_nodes) / max(1, len(union_nodes)))
        edge_turnover = 1.0 - (len(inter_edges) / max(1, len(union_edges)))
        support_persistence = safe_mean(edge_j) if edge_j else 1.0
        collective_identity = support_persistence * member_turnover
        out.append({
            "epoch_index": idx,
            "core_pair": list(core),
            "start_step": group[0].step,
            "end_step": group[-1].step,
            "n_snapshots": len(group),
            "births_in_epoch": births,
            "persistent_births_in_epoch": persistent,
            "remerge_births_in_epoch": remerge,
            "node_union": sorted(union_nodes),
            "node_intersection": sorted(inter_nodes),
            "edge_union": [list(e) for e in sorted(union_edges)],
            "edge_intersection": [list(e) for e in sorted(inter_edges)],
            "node_union_size": len(union_nodes),
            "node_intersection_size": len(inter_nodes),
            "edge_union_size": len(union_edges),
            "edge_intersection_size": len(inter_edges),
            "mean_node_jaccard": safe_mean(node_j) if node_j else 1.0,
            "mean_edge_jaccard": support_persistence,
            "member_turnover": member_turnover,
            "edge_turnover": edge_turnover,
            "collective_identity_score": collective_identity,
            "dominant_topology_class": topo_counts.most_common(1)[0][0] if topo_counts else "unknown",
            "topology_class_counts": dict(topo_counts),
            "core_is_fixed_but_members_turn_over": bool(len(group) >= 2 and member_turnover > 0.0 and support_persistence > 0.5),
        })
    return out


def birth_localization(orgs: List[SnapshotOrganizer], birth_events: List[dict], active_edges_final: Iterable[Sequence[int]]) -> dict:
    # Approximate localization: each birth is attached to its parent pair and compared
    # to the nearest organizer seen at/after its likely formation window.
    by_step = {o.step: o for o in orgs}
    sorted_steps = sorted(by_step)
    final_graph_nodes = set().union(*(o.nodes for o in orgs)) if orgs else set()
    final_graph = SimpleGraph(final_graph_nodes, to_edge_set(active_edges_final))

    near_counts = Counter()
    dists: List[int] = []
    records = []
    birth_steps = sorted_steps[: len(birth_events)] if sorted_steps else []
    for idx, ev in enumerate(birth_events):
        parents = to_node_set(ev.get("parents", []))
        new_node = int(ev.get("new_node", -1))
        step = birth_steps[idx] if idx < len(birth_steps) else None
        org = by_step.get(step) if step is not None else None
        if org is None and sorted_steps:
            org = by_step[sorted_steps[min(idx, len(sorted_steps) - 1)]]
        organizer_nodes = org.nodes if org else set()
        dist_parent = None
        for p in parents:
            d = nearest_distance_to_set(p, organizer_nodes, final_graph)
            if d is not None:
                dist_parent = d if dist_parent is None else min(dist_parent, d)
        if dist_parent is None:
            near_counts["unresolved"] += 1
        elif dist_parent == 0:
            near_counts["on_organizer"] += 1
            dists.append(0)
        elif dist_parent == 1:
            near_counts["adjacent_to_organizer"] += 1
            dists.append(1)
        else:
            near_counts["far_from_organizer"] += 1
            dists.append(dist_parent)
        records.append({
            "birth_index": idx,
            "step_proxy": step,
            "parents": sorted(parents),
            "new_node": new_node,
            "label": ev.get("label", "unknown"),
            "distance_to_organizer": dist_parent,
            "organizer_core_pair": list(org.core_pair) if org else None,
        })
    return {
        "counts": dict(near_counts),
        "mean_distance": safe_mean([float(x) for x in dists]) if dists else None,
        "records": records,
    }


def run_analysis(data: dict) -> dict:
    raw_snaps = data.get("snapshots", [])
    orgs = [o for s in raw_snaps if (o := organizer_from_snapshot(s)) is not None]
    snap_out: List[dict] = []
    prev_nodes: Optional[Set[Node]] = None
    prev_edges: Optional[Set[Edge]] = None
    prev_path_nodes: Optional[Set[Node]] = None
    for org in orgs:
        met = snapshot_metrics(org)
        node_j = 1.0 if prev_nodes is None else jaccard(prev_nodes, org.nodes)
        edge_j = 1.0 if prev_edges is None else jaccard(prev_edges, org.edges)
        path_nodes = set(met["path_nodes"])
        path_j = 1.0 if prev_path_nodes is None else jaccard(prev_path_nodes, path_nodes)
        # Pattern persistence despite member change: high support continuity and some turnover.
        turnover = 0.0 if prev_nodes is None else 1.0 - jaccard(prev_nodes, org.nodes)
        collective_identity = edge_j * turnover
        snap_out.append({
            "step": org.step,
            "core_pair": list(org.core_pair),
            "shell_nodes": sorted(org.nodes),
            "shell_edges": [list(e) for e in sorted(org.edges)],
            **met,
            "node_jaccard_prev": node_j,
            "edge_jaccard_prev": edge_j,
            "path_jaccard_prev": path_j,
            "member_turnover_prev": turnover,
            "collective_identity_score": collective_identity,
            "births_this_window": org.births_this_window,
            "persistent_this_window": org.persistent_this_window,
            "remerge_this_window": org.remerge_this_window,
        })
        prev_nodes = set(org.nodes)
        prev_edges = set(org.edges)
        prev_path_nodes = path_nodes

    epochs = build_epochs(orgs)
    births = birth_localization(orgs, data.get("birth_events", []), data.get("active_edges_final", []))

    topo_counts = Counter(s["topology_class"] for s in snap_out)
    dominant_topo = topo_counts.most_common(1)[0][0] if topo_counts else "unknown"
    coll_scores = [s["collective_identity_score"] for s in snap_out]
    line_scores = [s["line_likeness"] for s in snap_out]
    star_scores = [s["star_likeness"] for s in snap_out]
    tri_scores = [s["triangle_density"] for s in snap_out]
    branch_scores = [s["branch_fraction"] for s in snap_out]
    path_j_scores = [s["path_jaccard_prev"] for s in snap_out[1:]]

    strongest_epoch = max(epochs, key=lambda e: (e["collective_identity_score"], e["n_snapshots"])) if epochs else None
    longest_epoch = max(epochs, key=lambda e: e["n_snapshots"]) if epochs else None

    interpretation = {
        "supports_collective_particle_picture": False,
        "reason": [],
    }
    if strongest_epoch and strongest_epoch["collective_identity_score"] > 0.15:
        interpretation["supports_collective_particle_picture"] = True
        interpretation["reason"].append("An epoch shows strong pattern persistence alongside nonzero member turnover.")
    if safe_mean(path_j_scores) > 0.6:
        interpretation["reason"].append("Backbone/path support is continuous from snapshot to snapshot.")
    if safe_mean(tri_scores) > safe_mean(line_scores):
        interpretation["reason"].append("Organizer is more triangulated/shared-edge than line-like.")
    if safe_mean(star_scores) > safe_mean(line_scores):
        interpretation["reason"].append("Hub/corridor support dominates over simple 1D path structure.")

    return {
        "input_summary": {
            "n_snapshots": len(raw_snaps),
            "n_birth_events": len(data.get("birth_events", [])),
            "longest_lived_core_from_input": (data.get("summary") or {}).get("longest_lived_core"),
            "longest_core_lifetime_from_input": (data.get("summary") or {}).get("longest_core_lifetime"),
        },
        "snapshot_analysis": snap_out,
        "epoch_analysis": epochs,
        "birth_localization": births,
        "summary": {
            "dominant_topology_class": dominant_topo,
            "topology_class_counts": dict(topo_counts),
            "mean_collective_identity_score": safe_mean(coll_scores),
            "max_collective_identity_score": max(coll_scores) if coll_scores else 0.0,
            "mean_line_likeness": safe_mean(line_scores),
            "mean_star_likeness": safe_mean(star_scores),
            "mean_triangle_density": safe_mean(tri_scores),
            "mean_branch_fraction": safe_mean(branch_scores),
            "mean_path_jaccard_prev": safe_mean(path_j_scores),
            "strongest_collective_epoch": strongest_epoch,
            "longest_epoch": longest_epoch,
            "interpretation": interpretation,
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze collective mesoscopic mode structure from a mesoscape JSON.")
    ap.add_argument("--input-json", required=True, help="Input gpu_mesoscape_metric_analysis JSON")
    ap.add_argument("--json-out", default="gpu_mesoscape_collective_mode_analysis.json", help="Output analysis JSON")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input_json)
    out_path = Path(args.json_out)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    result = run_analysis(data)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote collective mode analysis to {out_path}")
    summ = result["summary"]
    print("Dominant topology:", summ["dominant_topology_class"])
    print("Mean line_likeness:", f"{summ['mean_line_likeness']:.4f}")
    print("Mean triangle_density:", f"{summ['mean_triangle_density']:.4f}")
    print("Mean collective_identity_score:", f"{summ['mean_collective_identity_score']:.4f}")
    strongest = summ.get("strongest_collective_epoch")
    if strongest:
        print(
            "Strongest epoch:",
            strongest["core_pair"],
            f"steps {strongest['start_step']}->{strongest['end_step']}",
            f"score={strongest['collective_identity_score']:.4f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
