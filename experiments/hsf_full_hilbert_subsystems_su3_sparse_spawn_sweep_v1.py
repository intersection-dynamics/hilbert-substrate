#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def load_target_module(target_path: str):
    spec = importlib.util.spec_from_file_location("hsf_spawn_target", target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load target module from {target_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_one(mod, job: Dict[str, Any]) -> Dict[str, Any]:
    cfg = mod.SimConfig(
        n_max=int(job["n_max"]),
        n_init=int(job["n_init"]),
        seed=int(job["seed"]),
        local_scale=float(job["local_scale"]),
        pair_scale=float(job["pair_scale"]),
        spawn_pair_scale=float(job["spawn_pair_scale"]),
        total_steps=int(job["total_steps"]),
        dt=float(job["dt"]),
        spawn_every=int(job["spawn_every"]),
        spawn_mi_threshold=float(job["spawn_mi_threshold"]),
        spawn_corr_threshold=float(job["spawn_corr_threshold"]),
        max_spawns=int(job["max_spawns"]),
        json_out="",
    )
    result = mod.run_sim(cfg)
    summary = result["summary"]
    active_nodes = result["active_nodes_final"]
    active_edges = result["active_edges_final"]
    spawn_events = result["spawn_events"]

    # simple motif counts from final graph
    edge_set = {tuple(sorted(e)) for e in active_edges}
    triangles = 0
    for i in range(len(active_nodes)):
        for j in range(i + 1, len(active_nodes)):
            for k in range(j + 1, len(active_nodes)):
                a, b, c = active_nodes[i], active_nodes[j], active_nodes[k]
                if tuple(sorted((a, b))) in edge_set and tuple(sorted((b, c))) in edge_set and tuple(sorted((a, c))) in edge_set:
                    triangles += 1

    degree = {v: 0 for v in active_nodes}
    for i, j in edge_set:
        degree[i] += 1
        degree[j] += 1
    max_degree = max(degree.values()) if degree else 0

    dominant_parent_pair = None
    if spawn_events:
        parent_counts = Counter(tuple(evt["parents"]) for evt in spawn_events)
        dominant_parent_pair = list(parent_counts.most_common(1)[0][0])

    late_birth_survival = 0.0
    if spawn_events:
        # crude proxy: fraction of spawned nodes whose final single-subsystem entropy is > 0.05
        ent = summary["single_subsystem_entropies"]
        survived = 0
        for evt in spawn_events:
            node = evt["new_node"]
            val = ent.get(str(node), 0.0) if isinstance(ent, dict) else 0.0
            if float(val) > 0.05:
                survived += 1
        late_birth_survival = survived / len(spawn_events)

    return {
        "job": job,
        "summary": {
            "n_active_final": int(summary["n_active_final"]),
            "n_edges_final": int(summary["n_edges_final"]),
            "n_spawn_events": int(summary["n_spawn_events"]),
            "mean_single_subsystem_entropy": float(summary["mean_single_subsystem_entropy"]),
            "mean_pair_mutual_information": float(summary["mean_pair_mutual_information"]),
            "mean_chain_cmi": float(summary["mean_chain_cmi"]),
            "mean_triangle_pair_mi": float(summary["mean_triangle_pair_mi"]),
            "mean_triangle_frustration": float(summary["mean_triangle_frustration"]),
            "triangle_count_final": int(triangles),
            "max_degree_final": int(max_degree),
            "late_birth_survival_frac": float(late_birth_survival),
            "dominant_parent_pair": dominant_parent_pair,
            "top_pairs": summary["top_pairs"][:5],
            "top_chains": summary["top_chains"][:5],
            "top_triangles": summary["top_triangles"][:5],
        },
        "spawn_events": spawn_events,
        "active_nodes_final": active_nodes,
        "active_edges_final": active_edges,
    }


def aggregate(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = tuple(r["job"][f] for f in fields)
        buckets[key].append(r)

    out = []
    for key, rows in buckets.items():
        row = {fields[i]: key[i] for i in range(len(fields))}
        row["count"] = len(rows)
        row["mean_n_active_final"] = mean(r["summary"]["n_active_final"] for r in rows)
        row["mean_n_edges_final"] = mean(r["summary"]["n_edges_final"] for r in rows)
        row["mean_spawn_events"] = mean(r["summary"]["n_spawn_events"] for r in rows)
        row["mean_pair_mi"] = mean(r["summary"]["mean_pair_mutual_information"] for r in rows)
        row["mean_chain_cmi"] = mean(r["summary"]["mean_chain_cmi"] for r in rows)
        row["mean_triangle_frustration"] = mean(r["summary"]["mean_triangle_frustration"] for r in rows)
        row["mean_triangle_count_final"] = mean(r["summary"]["triangle_count_final"] for r in rows)
        row["mean_max_degree_final"] = mean(r["summary"]["max_degree_final"] for r in rows)
        row["mean_late_birth_survival_frac"] = mean(r["summary"]["late_birth_survival_frac"] for r in rows)

        parent_counter = Counter()
        for r in rows:
            dpp = r["summary"]["dominant_parent_pair"]
            if dpp is not None:
                parent_counter[tuple(dpp)] += 1
        row["most_common_dominant_parent_pair"] = list(parent_counter.most_common(1)[0][0]) if parent_counter else None
        out.append(row)

    out.sort(key=lambda d: tuple(d[f] for f in fields))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep the HSF sparse full-subsystem relational spawning toy.")
    ap.add_argument("--target-script", type=str, default="hsf_full_hilbert_subsystems_su3_sparse_spawn_v3.py")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--spawn-mi-threshold-values", type=float, nargs="+", default=[0.15, 0.20, 0.25, 0.30])
    ap.add_argument("--spawn-corr-threshold-values", type=float, nargs="+", default=[0.35, 0.50, 0.65])
    ap.add_argument("--local-scale", type=float, default=0.10)
    ap.add_argument("--pair-scale", type=float, default=0.16)
    ap.add_argument("--spawn-pair-scale", type=float, default=0.12)
    ap.add_argument("--total-steps", type=int, default=24)
    ap.add_argument("--dt", type=float, default=0.20)
    ap.add_argument("--spawn-every", type=int, default=4)
    ap.add_argument("--max-spawns", type=int, default=4)
    ap.add_argument("--json-out", type=str, default="hsf_full_hilbert_subsystems_su3_sparse_spawn_sweep_v1.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    target_path = str(Path(args.target_script).resolve())
    if not Path(target_path).exists():
        raise FileNotFoundError(f"Target script not found: {target_path}")

    mod = load_target_module(target_path)

    jobs = []
    for mi_th in args.spawn_mi_threshold_values:
        for corr_th in args.spawn_corr_threshold_values:
            for seed in args.seeds:
                jobs.append({
                    "n_max": int(args.n_max),
                    "n_init": int(args.n_init),
                    "seed": int(seed),
                    "local_scale": float(args.local_scale),
                    "pair_scale": float(args.pair_scale),
                    "spawn_pair_scale": float(args.spawn_pair_scale),
                    "total_steps": int(args.total_steps),
                    "dt": float(args.dt),
                    "spawn_every": int(args.spawn_every),
                    "spawn_mi_threshold": float(mi_th),
                    "spawn_corr_threshold": float(corr_th),
                    "max_spawns": int(args.max_spawns),
                })

    print()
    print("HSF FULL-HILBERT SUBSYSTEM SPAWNING SWEEP (v1)")
    print()
    print(f"Target script: {target_path}")
    print(f"Jobs: {len(jobs)}")
    print()

    records = []
    for idx, job in enumerate(jobs, start=1):
        records.append(run_one(mod, job))
        if idx % max(1, min(12, len(jobs))) == 0 or idx == len(jobs):
            print(f"Completed {idx}/{len(jobs)} jobs")

    agg = aggregate(records, ["spawn_mi_threshold", "spawn_corr_threshold"])

    payload = {
        "meta": {
            "target_script": target_path,
            "jobs": len(jobs),
            "n_max": int(args.n_max),
            "n_init": int(args.n_init),
            "seeds": list(args.seeds),
            "spawn_mi_threshold_values": [float(x) for x in args.spawn_mi_threshold_values],
            "spawn_corr_threshold_values": [float(x) for x in args.spawn_corr_threshold_values],
        },
        "records": records,
        "aggregate_by_threshold_cell": agg,
    }

    out_path = Path(args.json_out).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"Saved: {out_path}")
    print()
    print("Aggregate by (spawn_mi_threshold, spawn_corr_threshold):")
    for row in agg:
        print(
            f"  mi={row['spawn_mi_threshold']:.2f} corr={row['spawn_corr_threshold']:.2f} "
            f"count={row['count']} active={row['mean_n_active_final']:.2f} "
            f"edges={row['mean_n_edges_final']:.2f} spawns={row['mean_spawn_events']:.2f} "
            f"pair_MI={row['mean_pair_mi']:.4f} chain_CMI={row['mean_chain_cmi']:.4f} "
            f"tri_frustr={row['mean_triangle_frustration']:.4f} triangles={row['mean_triangle_count_final']:.2f} "
            f"late_survival={row['mean_late_birth_survival_frac']:.4f} "
            f"hub_pair={row['most_common_dominant_parent_pair']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
