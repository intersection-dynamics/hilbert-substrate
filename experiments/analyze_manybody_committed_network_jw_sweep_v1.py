#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Set, Tuple

def load_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def edge_to_tuple(edge: List[int]) -> Tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)

def pair_to_tuple(pair: List[int]) -> Tuple[int, int]:
    return edge_to_tuple(pair)

def top_edges(summary: Dict[str, Any], k: int) -> List[Tuple[int, int]]:
    return [edge_to_tuple(row["edge"]) for row in summary["edge_summary_top5"][:k]]

def top_jw_pairs(summary: Dict[str, Any], k: int) -> List[Tuple[int, int]]:
    return [pair_to_tuple(row["pair"]) for row in summary["jw_summary_top5"][:k]]

def nodes_from_edges(edges: List[Tuple[int, int]]) -> Set[int]:
    out: Set[int] = set()
    for a, b in edges:
        out.add(a)
        out.add(b)
    return out

def overlap_metrics(summary: Dict[str, Any], top_edge_k: int, top_jw_k: int) -> Dict[str, Any]:
    e = top_edges(summary, top_edge_k)
    j = top_jw_pairs(summary, top_jw_k)
    e_nodes = nodes_from_edges(e)
    j_nodes = nodes_from_edges(j)
    exact_pair_overlap = len(set(e).intersection(set(j)))
    jw_pairs_touching_edge_nodes = sum(1 for pair in j if pair[0] in e_nodes or pair[1] in e_nodes)
    jw_pairs_inside_edge_neighborhood = sum(1 for pair in j if pair[0] in e_nodes and pair[1] in e_nodes)
    return {
        "top_edge_k": top_edge_k,
        "top_jw_k": top_jw_k,
        "edge_nodes": sorted(e_nodes),
        "jw_nodes": sorted(j_nodes),
        "exact_pair_overlap_count": exact_pair_overlap,
        "jw_touch_edge_neighborhood_frac": 0.0 if not j else jw_pairs_touching_edge_nodes / len(j),
        "jw_inside_edge_neighborhood_frac": 0.0 if not j else jw_pairs_inside_edge_neighborhood / len(j),
    }

def dominant_occupied_cluster(summary: Dict[str, Any], frac_of_max: float) -> Dict[str, Any]:
    occ = [float(x) for x in summary["node_mean_occupations"]]
    if not occ:
        return {"cluster_nodes": [], "cluster_size": 0, "threshold": 0.0, "mass_in_cluster": 0.0, "cluster_mass_fraction": 0.0}
    max_occ = max(occ)
    thresh = frac_of_max * max_occ
    cluster_nodes = [i for i, x in enumerate(occ) if x >= thresh]
    mass_in_cluster = sum(occ[i] for i in cluster_nodes)
    total_mass = sum(occ)
    return {
        "cluster_nodes": cluster_nodes,
        "cluster_size": len(cluster_nodes),
        "threshold": thresh,
        "mass_in_cluster": mass_in_cluster,
        "cluster_mass_fraction": 0.0 if total_mass <= 1e-15 else mass_in_cluster / total_mass,
    }

def backbone_concentration(summary: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    rows = summary["edge_summary_top5"]
    if not rows:
        return {"top_k": top_k, "top_mean_J_sum": 0.0, "all_mean_J_sum": 0.0, "backbone_concentration": 0.0}
    vals = [float(r["mean_J"]) for r in rows]
    top_sum = sum(vals[:top_k])
    total_sum = sum(vals)
    return {
        "top_k": top_k,
        "top_mean_J_sum": top_sum,
        "all_mean_J_sum": total_sum,
        "backbone_concentration": 0.0 if total_sum <= 1e-15 else top_sum / total_sum,
    }

def enrich_record(rec: Dict[str, Any], top_edge_k: int, top_jw_k: int, occ_frac: float, backbone_top_k: int) -> Dict[str, Any]:
    summ = rec["summary"]
    return {
        "job": rec["job"],
        "summary": summ,
        "diagnostics": {
            "overlap": overlap_metrics(summ, top_edge_k=top_edge_k, top_jw_k=top_jw_k),
            "dominant_cluster": dominant_occupied_cluster(summ, frac_of_max=occ_frac),
            "backbone": backbone_concentration(summ, top_k=backbone_top_k),
        },
    }

def aggregate(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = tuple(r["job"][f] for f in fields)
        buckets[key].append(r)
    out = []
    for key, rows in buckets.items():
        d = {fields[i]: key[i] for i in range(len(fields))}
        d["count"] = len(rows)
        edge_p = [float(r["summary"]["edge_persistence_proxy"]) for r in rows]
        occ_p = [float(r["summary"]["occupation_persistence_proxy"]) for r in rows]
        jw_p = [float(r["summary"]["jw_persistence_proxy"]) for r in rows]
        overlap_touch = [float(r["diagnostics"]["overlap"]["jw_touch_edge_neighborhood_frac"]) for r in rows]
        overlap_inside = [float(r["diagnostics"]["overlap"]["jw_inside_edge_neighborhood_frac"]) for r in rows]
        cluster_size = [int(r["diagnostics"]["dominant_cluster"]["cluster_size"]) for r in rows]
        cluster_mass = [float(r["diagnostics"]["dominant_cluster"]["cluster_mass_fraction"]) for r in rows]
        backbone = [float(r["diagnostics"]["backbone"]["backbone_concentration"]) for r in rows]
        d.update({
            "mean_edge_persistence": mean(edge_p),
            "mean_occupation_persistence": mean(occ_p),
            "mean_jw_persistence": mean(jw_p),
            "mean_jw_touch_edge_neighborhood_frac": mean(overlap_touch),
            "mean_jw_inside_edge_neighborhood_frac": mean(overlap_inside),
            "mean_dominant_cluster_size": mean(cluster_size),
            "mean_cluster_mass_fraction": mean(cluster_mass),
            "mean_backbone_concentration": mean(backbone),
        })
        out.append(d)
    out.sort(key=lambda x: tuple(x[f] for f in fields))
    return out

def print_summary_table(rows: List[Dict[str, Any]], fields: List[str]) -> None:
    for row in rows:
        prefix = " ".join(f"{f}={row[f]}" for f in fields)
        print(
            f"{prefix} count={row['count']} "
            f"edge={row['mean_edge_persistence']:.4f} "
            f"occ={row['mean_occupation_persistence']:.4f} "
            f"jw={row['mean_jw_persistence']:.4f} "
            f"jw_touch={row['mean_jw_touch_edge_neighborhood_frac']:.4f} "
            f"jw_inside={row['mean_jw_inside_edge_neighborhood_frac']:.4f} "
            f"cluster_size={row['mean_dominant_cluster_size']:.2f} "
            f"cluster_mass={row['mean_cluster_mass_fraction']:.4f} "
            f"backbone={row['mean_backbone_concentration']:.4f}"
        )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze manybody committed-network JW sweep results.")
    ap.add_argument("--input", type=str, default="manybody_committed_network_jw_sweep_results.json")
    ap.add_argument("--top-edge-k", type=int, default=3)
    ap.add_argument("--top-jw-k", type=int, default=3)
    ap.add_argument("--occ-frac-of-max", type=float, default=0.25)
    ap.add_argument("--backbone-top-k", type=int, default=3)
    ap.add_argument("--json-out", type=str, default="manybody_committed_network_jw_sweep_analysis.json")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    payload = load_payload(args.input)
    results = payload["results"]
    enriched = [
        enrich_record(rec, top_edge_k=int(args.top_edge_k), top_jw_k=int(args.top_jw_k),
                      occ_frac=float(args.occ_frac_of_max), backbone_top_k=int(args.backbone_top_k))
        for rec in results
    ]
    by_n_graph_target = aggregate(enriched, ["n", "graph_type", "target_commit"])
    by_n = aggregate(enriched, ["n"])
    by_graph = aggregate(enriched, ["graph_type"])
    out = {
        "meta": {
            "input": str(Path(args.input).resolve()),
            "top_edge_k": int(args.top_edge_k),
            "top_jw_k": int(args.top_jw_k),
            "occ_frac_of_max": float(args.occ_frac_of_max),
            "backbone_top_k": int(args.backbone_top_k),
        },
        "enriched_results": enriched,
        "aggregate_by_n_graph_targetcommit": by_n_graph_target,
        "aggregate_by_n": by_n,
        "aggregate_by_graph": by_graph,
    }
    out_path = Path(args.json_out).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print("ANALYSIS OF MANY-BODY COMMITTED NETWORK JW SWEEP")
    print()
    print(f"Saved: {out_path}")
    print()
    print("Aggregate by n / graph / target_commit:")
    print_summary_table(by_n_graph_target, ["n", "graph_type", "target_commit"])
    print()
    print("Aggregate by n:")
    print_summary_table(by_n, ["n"])
    print()
    print("Aggregate by graph:")
    print_summary_table(by_graph, ["graph_type"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
