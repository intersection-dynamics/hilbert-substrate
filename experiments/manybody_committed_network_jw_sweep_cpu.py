#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def load_target_module(target_path: str):
    spec = importlib.util.spec_from_file_location("mbjw_target", target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load target module from {target_path}")
    mod = importlib.util.module_from_spec(spec)
    # Important on Windows multiprocessing / dataclasses:
    # register the module before executing it.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def estimate_bytes_per_job(n: int) -> int:
    dim = 2 ** n
    one_dense = dim * dim * 16
    multiplier = 5.0
    overhead = 256 * 1024 * 1024
    return int(multiplier * one_dense + overhead)


def choose_worker_count(
    n_values: List[int],
    ram_gb: float,
    cpu_workers: int,
    reserve_gb: float,
    hard_cap: int | None,
):
    max_n = max(n_values)
    bytes_per = estimate_bytes_per_job(max_n)
    avail_bytes = max(1, int((ram_gb - reserve_gb) * (1024 ** 3)))
    by_ram = max(1, avail_bytes // max(1, bytes_per))
    workers = min(cpu_workers, by_ram)
    if hard_cap is not None:
        workers = min(workers, hard_cap)
    workers = max(1, workers)
    return workers, {
        "max_n": max_n,
        "estimated_bytes_per_job": bytes_per,
        "estimated_gb_per_job": bytes_per / (1024 ** 3),
        "available_gb_for_jobs": max(0.0, ram_gb - reserve_gb),
        "workers_by_ram": int(by_ram),
        "workers_final": int(workers),
    }


def run_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    mod = load_target_module(job["target_path"])
    cfg = mod.SimConfig(
        n=int(job["n"]),
        steps=int(job["steps"]),
        dt=float(job["dt"]),
        graph_type=str(job["graph_type"]),
        seed=int(job["seed"]),
        hz_scale=float(job["hz_scale"]),
        J0=float(job["J0"]),
        J_min=float(job["J_min"]),
        J_max=float(job["J_max"]),
        eta_up=float(job["eta_up"]),
        eta_down=float(job["eta_down"]),
        target_commit=float(job["target_commit"]),
        init_state=str(job["init_state"]),
    )
    result = mod.run_sim(cfg)
    summ = result["summary"]
    return {
        "job": {
            "n": int(job["n"]),
            "steps": int(job["steps"]),
            "dt": float(job["dt"]),
            "graph_type": str(job["graph_type"]),
            "seed": int(job["seed"]),
            "target_commit": float(job["target_commit"]),
            "init_state": str(job["init_state"]),
        },
        "summary": {
            "edge_persistence_proxy": float(summ["edge_persistence_proxy"]),
            "occupation_persistence_proxy": float(summ["occupation_persistence_proxy"]),
            "jw_persistence_proxy": float(summ["jw_persistence_proxy"]),
            "node_mean_occupations": summ["node_mean_occupations"],
            "node_std_occupations": summ["node_std_occupations"],
            "edge_summary_top5": summ["edge_summary"][: min(5, len(summ["edge_summary"]))],
            "jw_summary_top5": summ["jw_summary_top20"][: min(5, len(summ["jw_summary_top20"]))],
        },
    }


def group_key(rec: Dict[str, Any], fields: List[str]) -> Tuple[Any, ...]:
    return tuple(rec["job"][f] for f in fields)


def aggregate_results(results: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in results:
        buckets.setdefault(group_key(r, fields), []).append(r)

    out = []
    for k, rows in buckets.items():
        edge_p = [x["summary"]["edge_persistence_proxy"] for x in rows]
        occ_p = [x["summary"]["occupation_persistence_proxy"] for x in rows]
        jw_p = [x["summary"]["jw_persistence_proxy"] for x in rows]

        row = {fields[i]: k[i] for i in range(len(fields))}
        row.update({
            "count": len(rows),
            "mean_edge_persistence": mean(edge_p),
            "mean_occupation_persistence": mean(occ_p),
            "mean_jw_persistence": mean(jw_p),
            "max_edge_persistence": max(edge_p),
            "max_occupation_persistence": max(occ_p),
            "max_jw_persistence": max(jw_p),
        })
        out.append(row)

    out.sort(key=lambda d: tuple(d[f] for f in fields))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CPU sweep runner for manybody_committed_network_jw_v1_fixed.py")
    ap.add_argument("--target-script", type=str, default="manybody_committed_network_jw_v1_fixed.py")
    ap.add_argument("--n-values", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--graph-types", nargs="+", default=["ring_plus_chords", "erdos"])
    ap.add_argument("--target-commit-values", type=float, nargs="+", default=[0.18, 0.22, 0.26])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--hz-scale", type=float, default=0.18)
    ap.add_argument("--J0", type=float, default=0.14)
    ap.add_argument("--J-min", type=float, default=0.02)
    ap.add_argument("--J-max", type=float, default=0.35)
    ap.add_argument("--eta-up", type=float, default=0.035)
    ap.add_argument("--eta-down", type=float, default=0.020)
    ap.add_argument(
        "--init-state",
        choices=["single_excitation_center", "single_excitation_random", "random_pure"],
        default="single_excitation_center",
    )
    ap.add_argument("--ram-gb", type=float, default=32.0)
    ap.add_argument("--reserve-gb", type=float, default=8.0)
    ap.add_argument("--cpu-workers", type=int, default=max(1, (os.cpu_count() or 12) - 1))
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--json-out", type=str, default="manybody_committed_network_jw_sweep_results.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    target_path = str(Path(args.target_script).resolve())
    if not Path(target_path).exists():
        raise FileNotFoundError(f"Target script not found: {target_path}")

    jobs = []
    for n in args.n_values:
        for graph_type in args.graph_types:
            for tc in args.target_commit_values:
                for seed in args.seeds:
                    jobs.append({
                        "target_path": target_path,
                        "n": int(n),
                        "steps": int(args.steps),
                        "dt": float(args.dt),
                        "graph_type": str(graph_type),
                        "seed": int(seed),
                        "hz_scale": float(args.hz_scale),
                        "J0": float(args.J0),
                        "J_min": float(args.J_min),
                        "J_max": float(args.J_max),
                        "eta_up": float(args.eta_up),
                        "eta_down": float(args.eta_down),
                        "target_commit": float(tc),
                        "init_state": str(args.init_state),
                    })

    workers, mem_info = choose_worker_count(
        n_values=[int(x) for x in args.n_values],
        ram_gb=float(args.ram_gb),
        cpu_workers=int(args.cpu_workers),
        reserve_gb=float(args.reserve_gb),
        hard_cap=args.max_workers,
    )

    print()
    print("MANY-BODY COMMITTED NETWORK JW CPU SWEEP")
    print()
    print(f"Target script: {target_path}")
    print(f"Jobs: {len(jobs)}")
    print(f"Using workers: {workers}")
    print(f"Memory planning: {mem_info}")
    print()

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one_job, job) for job in jobs]
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % max(1, min(8, len(jobs))) == 0 or done == len(jobs):
                print(f"Completed {done}/{len(jobs)} jobs")

    results.sort(
        key=lambda r: (
            r["job"]["n"],
            r["job"]["graph_type"],
            r["job"]["target_commit"],
            r["job"]["seed"],
        )
    )

    payload = {
        "meta": {
            "target_script": target_path,
            "jobs": len(jobs),
            "workers": workers,
            "memory_planning": mem_info,
        },
        "results": results,
        "aggregate_by_n_graph_targetcommit": aggregate_results(results, ["n", "graph_type", "target_commit"]),
        "aggregate_by_n": aggregate_results(results, ["n"]),
        "aggregate_by_graph": aggregate_results(results, ["graph_type"]),
    }

    out_path = Path(args.json_out).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"Saved: {out_path}")
    print()
    print("Aggregate by n / graph / target_commit:")
    for row in payload["aggregate_by_n_graph_targetcommit"]:
        print(
            f"  n={row['n']} graph={row['graph_type']} target_commit={row['target_commit']:.2f} "
            f"count={row['count']} edge={row['mean_edge_persistence']:.4f} "
            f"occ={row['mean_occupation_persistence']:.4f} jw={row['mean_jw_persistence']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
