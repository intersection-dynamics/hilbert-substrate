#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Dict, List, Set, Tuple

def load_target_module(target_path: str):
    spec = importlib.util.spec_from_file_location("mbjw_target_phase2d", target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load target module from {target_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def build_cfg(mod, kwargs: Dict[str, Any]):
    if hasattr(mod, "SimConfig"):
        return mod.SimConfig(**kwargs)
    return SimpleNamespace(**kwargs)

def edge_to_tuple(edge: List[int]) -> Tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)

def pair_to_tuple(pair: List[int]) -> Tuple[int, int]:
    return edge_to_tuple(pair)

def top_edges(summary: Dict[str, Any], k: int) -> List[Tuple[int, int]]:
    rows = summary.get("edge_summary", summary.get("edge_summary_top5", []))
    return [edge_to_tuple(row["edge"]) for row in rows[:k]]

def top_jw_pairs(summary: Dict[str, Any], k: int) -> List[Tuple[int, int]]:
    rows = summary.get("jw_summary_top20", summary.get("jw_summary_top5", []))
    return [pair_to_tuple(row["pair"]) for row in rows[:k]]

def nodes_from_edges(edges: List[Tuple[int, int]]) -> Set[int]:
    out: Set[int] = set()
    for a, b in edges:
        out.add(a); out.add(b)
    return out

def overlap_metrics(summary: Dict[str, Any], top_edge_k: int, top_jw_k: int) -> Dict[str, Any]:
    e = top_edges(summary, top_edge_k)
    j = top_jw_pairs(summary, top_jw_k)
    e_nodes = nodes_from_edges(e)
    exact_pair_overlap = len(set(e).intersection(set(j)))
    jw_touch = sum(1 for pair in j if pair[0] in e_nodes or pair[1] in e_nodes)
    jw_inside = sum(1 for pair in j if pair[0] in e_nodes and pair[1] in e_nodes)
    return {
        "exact_pair_overlap_count": exact_pair_overlap,
        "jw_touch_edge_neighborhood_frac": 0.0 if not j else jw_touch / len(j),
        "jw_inside_edge_neighborhood_frac": 0.0 if not j else jw_inside / len(j),
    }

def dominant_occupied_cluster(summary: Dict[str, Any], frac_of_max: float, n: int) -> Dict[str, Any]:
    occ = [float(x) for x in summary.get("node_mean_occupations", [])]
    if not occ:
        return {"cluster_nodes": [], "cluster_size": 0, "cluster_fraction": 0.0, "threshold": 0.0, "mass_in_cluster": 0.0, "cluster_mass_fraction": 0.0}
    max_occ = max(occ)
    thresh = frac_of_max * max_occ
    cluster_nodes = [i for i, x in enumerate(occ) if x >= thresh]
    mass_in_cluster = sum(occ[i] for i in cluster_nodes)
    total_mass = sum(occ)
    return {
        "cluster_nodes": cluster_nodes,
        "cluster_size": len(cluster_nodes),
        "cluster_fraction": 0.0 if n <= 0 else len(cluster_nodes) / float(n),
        "threshold": thresh,
        "mass_in_cluster": mass_in_cluster,
        "cluster_mass_fraction": 0.0 if total_mass <= 1e-15 else mass_in_cluster / total_mass,
    }

def backbone_concentration(summary: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    rows = summary.get("edge_summary", summary.get("edge_summary_top5", []))
    if not rows:
        return {"backbone_concentration": 0.0, "top_mean_J_sum": 0.0, "all_mean_J_sum": 0.0}
    vals = [float(r["mean_J"]) for r in rows]
    top_sum = sum(vals[:top_k]); total_sum = sum(vals)
    return {
        "backbone_concentration": 0.0 if total_sum <= 1e-15 else top_sum / total_sum,
        "top_mean_J_sum": top_sum,
        "all_mean_J_sum": total_sum,
    }

def classify_regime(cluster_fraction: float, cluster_mass_fraction: float, backbone_concentration_val: float, jw_inside_frac: float) -> str:
    if cluster_mass_fraction >= 0.80 and cluster_fraction <= 0.40 and backbone_concentration_val >= 0.72 and jw_inside_frac >= 0.85:
        return "backbone_dominated"
    if cluster_mass_fraction >= 0.80 and cluster_fraction <= 0.45 and backbone_concentration_val >= 0.60 and jw_inside_frac >= 0.85:
        return "corridor"
    if cluster_mass_fraction >= 0.80 and cluster_fraction <= 0.55 and backbone_concentration_val >= 0.50 and jw_inside_frac >= 0.75:
        return "proto_corridor"
    if cluster_mass_fraction >= 0.70 and cluster_fraction <= 0.65:
        return "clustered"
    return "diffuse"

def run_one(mod, *, target_commit: float, eta_up: float, seed: int, n: int, steps: int, dt: float, graph_type: str,
            hz_scale: float, J0: float, J_min: float, J_max: float, eta_down: float, init_state: str,
            top_edge_k: int, top_jw_k: int, occ_frac_of_max: float, backbone_top_k: int) -> Dict[str, Any]:
    cfg_kwargs = {
        "n": int(n), "steps": int(steps), "dt": float(dt), "graph_type": str(graph_type), "seed": int(seed),
        "hz_scale": float(hz_scale), "J0": float(J0), "J_min": float(J_min), "J_max": float(J_max),
        "eta_up": float(eta_up), "eta_down": float(eta_down), "target_commit": float(target_commit),
        "init_state": str(init_state),
    }
    cfg = build_cfg(mod, cfg_kwargs)
    result = mod.run_sim(cfg)
    summary = result["summary"]
    overlap = overlap_metrics(summary, top_edge_k=top_edge_k, top_jw_k=top_jw_k)
    cluster = dominant_occupied_cluster(summary, frac_of_max=occ_frac_of_max, n=n)
    backbone = backbone_concentration(summary, top_k=backbone_top_k)
    regime = classify_regime(
        cluster_fraction=float(cluster["cluster_fraction"]),
        cluster_mass_fraction=float(cluster["cluster_mass_fraction"]),
        backbone_concentration_val=float(backbone["backbone_concentration"]),
        jw_inside_frac=float(overlap["jw_inside_edge_neighborhood_frac"]),
    )
    return {
        "job": {**cfg_kwargs},
        "order_parameters": {
            "edge_persistence_proxy": float(summary["edge_persistence_proxy"]),
            "occupation_persistence_proxy": float(summary["occupation_persistence_proxy"]),
            "jw_persistence_proxy": float(summary["jw_persistence_proxy"]),
            "jw_touch_edge_neighborhood_frac": float(overlap["jw_touch_edge_neighborhood_frac"]),
            "jw_inside_edge_neighborhood_frac": float(overlap["jw_inside_edge_neighborhood_frac"]),
            "dominant_cluster_size": int(cluster["cluster_size"]),
            "dominant_cluster_fraction": float(cluster["cluster_fraction"]),
            "cluster_mass_fraction": float(cluster["cluster_mass_fraction"]),
            "backbone_concentration": float(backbone["backbone_concentration"]),
        },
        "regime": regime,
        "summary_excerpt": {
            "edge_summary_top5": summary.get("edge_summary", [])[:5],
            "jw_summary_top5": summary.get("jw_summary_top20", [])[:5],
            "node_mean_occupations": summary.get("node_mean_occupations", []),
        },
    }

def aggregate_by_cell(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[float, float], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (float(r["job"]["target_commit"]), float(r["job"]["eta_up"]))
        buckets[key].append(r)
    out = []
    for (tc, eu), rows in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        edge_p = [r["order_parameters"]["edge_persistence_proxy"] for r in rows]
        occ_p = [r["order_parameters"]["occupation_persistence_proxy"] for r in rows]
        jw_p = [r["order_parameters"]["jw_persistence_proxy"] for r in rows]
        jw_touch = [r["order_parameters"]["jw_touch_edge_neighborhood_frac"] for r in rows]
        jw_inside = [r["order_parameters"]["jw_inside_edge_neighborhood_frac"] for r in rows]
        cl_size = [r["order_parameters"]["dominant_cluster_size"] for r in rows]
        cl_frac = [r["order_parameters"]["dominant_cluster_fraction"] for r in rows]
        cl_mass = [r["order_parameters"]["cluster_mass_fraction"] for r in rows]
        backbone = [r["order_parameters"]["backbone_concentration"] for r in rows]
        regimes = Counter(r["regime"] for r in rows)
        out.append({
            "target_commit": tc,
            "eta_up": eu,
            "count": len(rows),
            "mean_edge_persistence": mean(edge_p),
            "mean_occupation_persistence": mean(occ_p),
            "mean_jw_persistence": mean(jw_p),
            "mean_jw_touch_edge_neighborhood_frac": mean(jw_touch),
            "mean_jw_inside_edge_neighborhood_frac": mean(jw_inside),
            "mean_dominant_cluster_size": mean(cl_size),
            "mean_dominant_cluster_fraction": mean(cl_frac),
            "mean_cluster_mass_fraction": mean(cl_mass),
            "mean_backbone_concentration": mean(backbone),
            "regime_counts": dict(regimes),
            "dominant_regime": regimes.most_common(1)[0][0] if regimes else "none",
        })
    return out

def estimate_turn_on(cells: List[Dict[str, Any]], regime_target: str) -> Dict[str, Any]:
    order = ["diffuse", "clustered", "proto_corridor", "corridor", "backbone_dominated"]
    want_idx = order.index(regime_target)
    candidates = [row for row in cells if order.index(row["dominant_regime"]) >= want_idx]
    if not candidates:
        return {"regime_target": regime_target, "first_target_commit": None, "best_eta_up_at_first_target_commit": None}
    first_tc = min(r["target_commit"] for r in candidates)
    same_tc = [r for r in candidates if r["target_commit"] == first_tc]
    best = max(same_tc, key=lambda r: (r["mean_backbone_concentration"], r["mean_jw_inside_edge_neighborhood_frac"]))
    return {
        "regime_target": regime_target,
        "first_target_commit": first_tc,
        "best_eta_up_at_first_target_commit": best["eta_up"],
        "dominant_regime": best["dominant_regime"],
        "mean_backbone_concentration": best["mean_backbone_concentration"],
        "mean_jw_inside_edge_neighborhood_frac": best["mean_jw_inside_edge_neighborhood_frac"],
    }

def frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    x = start
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals

def print_rows(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"target_commit={row['target_commit']:.4f} eta_up={row['eta_up']:.4f} count={row['count']} "
            f"edge={row['mean_edge_persistence']:.4f} occ={row['mean_occupation_persistence']:.4f} "
            f"jw={row['mean_jw_persistence']:.4f} jw_inside={row['mean_jw_inside_edge_neighborhood_frac']:.4f} "
            f"cluster_frac={row['mean_dominant_cluster_fraction']:.4f} cluster_mass={row['mean_cluster_mass_fraction']:.4f} "
            f"backbone={row['mean_backbone_concentration']:.4f} regime={row['dominant_regime']}"
        )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="2D phase sweep over target_commit and eta_up.")
    ap.add_argument("--target-script", type=str, default="manybody_committed_network_jw_v1_fixed.py")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--graph-type", choices=["chain", "ring", "ring_plus_chords", "erdos"], default="ring_plus_chords")
    ap.add_argument("--target-commit-start", type=float, default=0.14)
    ap.add_argument("--target-commit-stop", type=float, default=0.30)
    ap.add_argument("--target-commit-step", type=float, default=0.02)
    ap.add_argument("--eta-up-start", type=float, default=0.02)
    ap.add_argument("--eta-up-stop", type=float, default=0.06)
    ap.add_argument("--eta-up-step", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5,6,7])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--hz-scale", type=float, default=0.18)
    ap.add_argument("--J0", type=float, default=0.14)
    ap.add_argument("--J-min", type=float, default=0.02)
    ap.add_argument("--J-max", type=float, default=0.35)
    ap.add_argument("--eta-down", type=float, default=0.020)
    ap.add_argument("--init-state", choices=["single_excitation_center", "single_excitation_random", "random_pure"], default="single_excitation_center")
    ap.add_argument("--top-edge-k", type=int, default=3)
    ap.add_argument("--top-jw-k", type=int, default=3)
    ap.add_argument("--occ-frac-of-max", type=float, default=0.25)
    ap.add_argument("--backbone-top-k", type=int, default=3)
    ap.add_argument("--json-out", type=str, default="phase_sweep_target_commit_eta_up_v1.json")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    target_path = str(Path(args.target_script).resolve())
    if not Path(target_path).exists():
        raise FileNotFoundError(f"Target script not found: {target_path}")
    mod = load_target_module(target_path)
    tc_values = frange(float(args.target_commit_start), float(args.target_commit_stop), float(args.target_commit_step))
    eu_values = frange(float(args.eta_up_start), float(args.eta_up_stop), float(args.eta_up_step))
    records = []

    print()
    print("2D PHASE SWEEP: target_commit x eta_up")
    print()
    print(f"Target script: {target_path}")
    print(f"N={args.n} graph={args.graph_type} steps={args.steps} dt={args.dt}")
    print(f"target_commit values: {tc_values}")
    print(f"eta_up values: {eu_values}")
    print(f"seeds: {list(args.seeds)}")
    print()

    total_jobs = len(tc_values) * len(eu_values) * len(args.seeds)
    done = 0
    for tc in tc_values:
        for eu in eu_values:
            for seed in args.seeds:
                rec = run_one(
                    mod=mod, target_commit=float(tc), eta_up=float(eu), seed=int(seed), n=int(args.n),
                    steps=int(args.steps), dt=float(args.dt), graph_type=str(args.graph_type),
                    hz_scale=float(args.hz_scale), J0=float(args.J0), J_min=float(args.J_min), J_max=float(args.J_max),
                    eta_down=float(args.eta_down), init_state=str(args.init_state),
                    top_edge_k=int(args.top_edge_k), top_jw_k=int(args.top_jw_k),
                    occ_frac_of_max=float(args.occ_frac_of_max), backbone_top_k=int(args.backbone_top_k),
                )
                records.append(rec)
                done += 1
                if done % max(1, min(16, total_jobs)) == 0 or done == total_jobs:
                    print(f"Completed {done}/{total_jobs} jobs")

    cells = aggregate_by_cell(records)
    out = {
        "meta": {
            "target_script": target_path,
            "n": int(args.n),
            "graph_type": str(args.graph_type),
            "steps": int(args.steps),
            "dt": float(args.dt),
            "seeds": list(args.seeds),
            "target_commit_values": tc_values,
            "eta_up_values": eu_values,
            "hz_scale": float(args.hz_scale),
            "J0": float(args.J0),
            "J_min": float(args.J_min),
            "J_max": float(args.J_max),
            "eta_down": float(args.eta_down),
            "init_state": str(args.init_state),
        },
        "records": records,
        "aggregate_by_cell": cells,
        "turn_on_estimates": {
            "proto_corridor": estimate_turn_on(cells, "proto_corridor"),
            "corridor": estimate_turn_on(cells, "corridor"),
            "backbone_dominated": estimate_turn_on(cells, "backbone_dominated"),
        },
    }

    out_path = Path(args.json_out).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"Saved: {out_path}")
    print()
    print("Aggregate by (target_commit, eta_up):")
    print_rows(cells)
    print()
    print(f"Turn-on estimates: {out['turn_on_estimates']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
