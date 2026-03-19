#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

def load_target_module(target_path: str):
    module_name = f"hsf_birth_rule_target_{abs(hash(target_path))}"
    spec = importlib.util.spec_from_file_location(module_name, target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load target module from {target_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def estimate_job_memory_gb(n_max: int) -> float:
    dim = 3 ** n_max
    bytes_est = dim * 16 * 18
    gb = bytes_est / (1024 ** 3)
    return max(0.35, gb)

def run_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    mod = load_target_module(job["target_script"])
    cfg = mod.SimConfig(
        n_max=int(job["n_max"]),
        n_init=int(job["n_init"]),
        seed=int(job["seed"]),
        local_scale=float(job["local_scale"]),
        pair_scale=float(job["pair_scale"]),
        spawn_pair_scale=float(job["spawn_pair_scale"]),
        total_steps=int(job["total_steps"]),
        dt=float(job["dt"]),
        eval_every=int(job["eval_every"]),
        lookahead_windows=int(job["lookahead_windows"]),
        settling_windows=int(job["settling_windows"]),
        fission_fraction=float(job["fission_fraction"]),
        candidate_fraction=float(job["candidate_fraction"]),
        birth_score_floor=float(job["birth_score_floor"]),
        decay_mi_threshold=float(job["decay_mi_threshold"]),
        decay_corr_threshold=float(job["decay_corr_threshold"]),
        neighborhood_bonus_weight=float(job["neighborhood_bonus_weight"]),
        shell_bonus_weight=float(job["shell_bonus_weight"]),
        mi_survival_floor=float(job["mi_survival_floor"]),
        corr_survival_floor=float(job["corr_survival_floor"]),
        persist_windows_required=int(job["persist_windows_required"]),
        persist_entropy_threshold=float(job["persist_entropy_threshold"]),
        persist_mean_mi_threshold=float(job["persist_mean_mi_threshold"]),
        persist_triangle_threshold=int(job["persist_triangle_threshold"]),
        json_out="",
    )
    result = mod.run_sim(cfg)
    summary = result["summary"]
    births = result["birth_events"]
    derived_rule = result["derived_rule"]
    persistent_mis = [float(b["mean_birth_mi"]) for b in births if b["label"] == "persistent"]
    failing_mis = [float(b["mean_birth_mi"]) for b in births if b["label"] == "remerge_prone"]
    return {
        "job": {k: v for k, v in job.items() if k != "target_script"},
        "summary": summary,
        "birth_events": births,
        "derived_rule": derived_rule,
        "diagnostics": {
            "persistent_mean_mi_mean": mean(persistent_mis) if persistent_mis else None,
            "remerge_mean_mi_mean": mean(failing_mis) if failing_mis else None,
            "fit_available": derived_rule.get("intercept") is not None,
            "n_labeled_births": int(derived_rule.get("n_labeled_births", 0)),
        },
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
        row["mean_births"] = mean(r["summary"]["n_birth_events"] for r in rows)
        row["mean_persistent"] = mean(r["summary"]["n_persistent_births"] for r in rows)
        row["mean_remerge"] = mean(r["summary"]["n_remerge_prone_births"] for r in rows)
        row["mean_final_nodes"] = mean(r["summary"]["active_nodes_final"] for r in rows)
        row["mean_final_edges"] = mean(r["summary"]["active_edges_final"] for r in rows)
        diffs = []
        fit_count = 0
        labeled = []
        for r in rows:
            pm = r["diagnostics"]["persistent_mean_mi_mean"]
            rm = r["diagnostics"]["remerge_mean_mi_mean"]
            if pm is not None and rm is not None:
                diffs.append(pm - rm)
            if r["diagnostics"]["fit_available"]:
                fit_count += 1
            labeled.append(r["diagnostics"]["n_labeled_births"])
        row["mean_persistent_minus_remerge_mi"] = mean(diffs) if diffs else None
        row["fit_available_fraction"] = fit_count / len(rows) if rows else 0.0
        row["mean_labeled_births"] = mean(labeled) if labeled else 0.0
        out.append(row)
    out.sort(key=lambda d: tuple(d[f] for f in fields))
    return out

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parallel seed and parameter sweep for derive_subsystem_birth_rule_v4.py")
    ap.add_argument("--target-script", type=str, default="derive_subsystem_birth_rule_v4.py")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5,6,7])
    ap.add_argument("--total-steps", type=int, default=120)
    ap.add_argument("--dt", type=float, default=0.20)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--lookahead-windows-values", type=int, nargs="+", default=[3,4])
    ap.add_argument("--settling-windows-values", type=int, nargs="+", default=[2,3])
    ap.add_argument("--candidate-fraction-values", type=float, nargs="+", default=[0.45])
    ap.add_argument("--fission-fraction-values", type=float, nargs="+", default=[0.25,0.30])
    ap.add_argument("--mi-survival-floor-values", type=float, nargs="+", default=[0.07,0.075,0.08])
    ap.add_argument("--corr-survival-floor-values", type=float, nargs="+", default=[0.08,0.085,0.09])
    ap.add_argument("--local-scale", type=float, default=0.10)
    ap.add_argument("--pair-scale", type=float, default=0.16)
    ap.add_argument("--spawn-pair-scale", type=float, default=0.12)
    ap.add_argument("--birth-score-floor", type=float, default=0.015)
    ap.add_argument("--decay-mi-threshold", type=float, default=0.05)
    ap.add_argument("--decay-corr-threshold", type=float, default=0.07)
    ap.add_argument("--neighborhood-bonus-weight", type=float, default=0.18)
    ap.add_argument("--shell-bonus-weight", type=float, default=0.20)
    ap.add_argument("--persist-windows-required", type=int, default=2)
    ap.add_argument("--persist-entropy-threshold", type=float, default=0.06)
    ap.add_argument("--persist-mean-mi-threshold", type=float, default=0.07)
    ap.add_argument("--persist-triangle-threshold", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=0)
    ap.add_argument("--reserve-cores", type=int, default=1)
    ap.add_argument("--ram-gb-for-jobs", type=float, default=24.0)
    ap.add_argument("--json-out", type=str, default="derive_subsystem_birth_rule_v4_sweep_parallel.json")
    return ap.parse_args()

def choose_worker_count(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    cpu_total = os.cpu_count() or 12
    by_cpu = max(1, cpu_total - int(args.reserve_cores))
    if int(args.max_workers) > 0:
        by_cpu = min(by_cpu, int(args.max_workers))
    est_gb = estimate_job_memory_gb(int(args.n_max))
    by_ram = max(1, int(math.floor(float(args.ram_gb_for_jobs) / est_gb)))
    workers = max(1, min(by_cpu, by_ram))
    return workers, {
        "cpu_total": cpu_total,
        "reserve_cores": int(args.reserve_cores),
        "workers_by_cpu": by_cpu,
        "estimated_gb_per_job": round(est_gb, 3),
        "ram_gb_for_jobs": float(args.ram_gb_for_jobs),
        "workers_by_ram": by_ram,
        "workers_final": workers,
    }

def main() -> int:
    args = parse_args()
    target_path = str(Path(args.target_script).resolve())
    if not Path(target_path).exists():
        raise FileNotFoundError(f"Target script not found: {target_path}")
    jobs = []
    for seed in args.seeds:
        for lookahead in args.lookahead_windows_values:
            for settling in args.settling_windows_values:
                for cand_frac in args.candidate_fraction_values:
                    for fiss_frac in args.fission_fraction_values:
                        for mi_floor in args.mi_survival_floor_values:
                            for corr_floor in args.corr_survival_floor_values:
                                jobs.append({
                                    "target_script": target_path,
                                    "n_max": int(args.n_max),
                                    "n_init": int(args.n_init),
                                    "seed": int(seed),
                                    "local_scale": float(args.local_scale),
                                    "pair_scale": float(args.pair_scale),
                                    "spawn_pair_scale": float(args.spawn_pair_scale),
                                    "total_steps": int(args.total_steps),
                                    "dt": float(args.dt),
                                    "eval_every": int(args.eval_every),
                                    "lookahead_windows": int(lookahead),
                                    "settling_windows": int(settling),
                                    "fission_fraction": float(fiss_frac),
                                    "candidate_fraction": float(cand_frac),
                                    "birth_score_floor": float(args.birth_score_floor),
                                    "decay_mi_threshold": float(args.decay_mi_threshold),
                                    "decay_corr_threshold": float(args.decay_corr_threshold),
                                    "neighborhood_bonus_weight": float(args.neighborhood_bonus_weight),
                                    "shell_bonus_weight": float(args.shell_bonus_weight),
                                    "mi_survival_floor": float(mi_floor),
                                    "corr_survival_floor": float(corr_floor),
                                    "persist_windows_required": int(args.persist_windows_required),
                                    "persist_entropy_threshold": float(args.persist_entropy_threshold),
                                    "persist_mean_mi_threshold": float(args.persist_mean_mi_threshold),
                                    "persist_triangle_threshold": int(args.persist_triangle_threshold),
                                })
    workers, planning = choose_worker_count(args)
    print()
    print("BIRTH RULE V4 SEED + PARAMETER SWEEP (PARALLEL)")
    print()
    print(f"Target script: {target_path}")
    print(f"Jobs: {len(jobs)}")
    print(f"Using workers: {workers}")
    print(f"Planning: {planning}")
    print()
    records = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one_job, job) for job in jobs]
        for fut in as_completed(futures):
            records.append(fut.result())
            completed += 1
            if completed % max(1, min(20, len(jobs))) == 0 or completed == len(jobs):
                print(f"Completed {completed}/{len(jobs)} jobs")
    agg = aggregate(records, ["lookahead_windows", "settling_windows", "fission_fraction", "mi_survival_floor", "corr_survival_floor"])
    payload = {
        "meta": {
            "target_script": target_path,
            "jobs": len(jobs),
            "planning": planning,
            "seeds": list(args.seeds),
            "lookahead_windows_values": list(args.lookahead_windows_values),
            "settling_windows_values": list(args.settling_windows_values),
            "candidate_fraction_values": list(args.candidate_fraction_values),
            "fission_fraction_values": list(args.fission_fraction_values),
            "mi_survival_floor_values": list(args.mi_survival_floor_values),
            "corr_survival_floor_values": list(args.corr_survival_floor_values),
        },
        "records": records,
        "aggregate": agg,
    }
    out_path = Path(args.json_out).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Saved: {out_path}")
    print()
    print("Top aggregate cells by mixed-regime quality:")
    scored = []
    for row in agg:
        mixedness = min(row["mean_persistent"], row["mean_remerge"])
        fitiness = row["fit_available_fraction"]
        birth_volume = row["mean_labeled_births"]
        scored.append(((mixedness, fitiness, birth_volume), row))
    scored.sort(reverse=True, key=lambda x: x[0])
    for _, row in scored[:12]:
        print(
            f"  lookahead={row['lookahead_windows']} settling={row['settling_windows']} "
            f"fission={row['fission_fraction']:.2f} mi_floor={row['mi_survival_floor']:.3f} "
            f"corr_floor={row['corr_survival_floor']:.3f} count={row['count']} "
            f"births={row['mean_births']:.2f} persistent={row['mean_persistent']:.2f} "
            f"remerge={row['mean_remerge']:.2f} labeled={row['mean_labeled_births']:.2f} "
            f"fit_frac={row['fit_available_fraction']:.2f} "
            f"delta_mi={row['mean_persistent_minus_remerge_mi'] if row['mean_persistent_minus_remerge_mi'] is not None else 'NA'}"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
