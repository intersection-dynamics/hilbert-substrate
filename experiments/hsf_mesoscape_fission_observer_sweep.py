#!/usr/bin/env python3
# filename: hsf_mesoscape_fission_observer_sweep.py

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SweepJob:
    seed: int
    pair_scale: float
    perturb_eps: float


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def safe_get(d: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def strongest_episode_length(run_json: Dict[str, Any]) -> int:
    ep = run_json.get("strongest_fission_episode", None)
    if not isinstance(ep, dict):
        return 0
    return int(ep.get("length", 0))


def candidate_activation_flags(candidate_summaries: Dict[str, Any], candidate: Optional[int]) -> Tuple[Optional[bool], Optional[bool]]:
    if candidate is None:
        return None, None
    row = candidate_summaries.get(str(candidate), {})
    mi_flag = safe_get(row, ["delta_mi_trace_summary", "activation_detected"], None)
    cc_flag = safe_get(row, ["delta_ccorr_trace_summary", "activation_detected"], None)
    return mi_flag, cc_flag


def summarize_run(run_json: Dict[str, Any], job: SweepJob, elapsed_sec: float, status: str, stderr_tail: str = "") -> Dict[str, Any]:
    candidate_summaries = run_json.get("candidate_summaries", {}) if isinstance(run_json, dict) else {}
    winner_candidate = run_json.get("winner_candidate", None) if isinstance(run_json, dict) else None
    winner_count = int(run_json.get("winner_count", 0)) if isinstance(run_json, dict) else 0
    strict_eps = run_json.get("strict_fission_episodes", []) if isinstance(run_json, dict) else []
    mi_flag, cc_flag = candidate_activation_flags(candidate_summaries, winner_candidate)

    return {
        "status": status,
        "seed": int(job.seed),
        "pair_scale": float(job.pair_scale),
        "perturb_eps": float(job.perturb_eps),
        "elapsed_sec": float(elapsed_sec),
        "winner_candidate": winner_candidate,
        "winner_count": winner_count,
        "strict_episode_count": len(strict_eps) if isinstance(strict_eps, list) else 0,
        "strongest_episode_length": strongest_episode_length(run_json) if isinstance(run_json, dict) else 0,
        "winner_mi_activation": mi_flag,
        "winner_ccorr_activation": cc_flag,
        "stderr_tail": stderr_tail,
    }


def build_command(
    python_exe: str,
    observer_script: Path,
    out_json: Path,
    args: argparse.Namespace,
    job: SweepJob,
) -> List[str]:
    cmd = [
        python_exe,
        str(observer_script),
        "--device", args.device,
        "--initial-state", args.initial_state,
        "--n-max", str(args.n_max),
        "--n-init", str(args.n_init),
        "--total-steps", str(args.total_steps),
        "--seed", str(job.seed),
        "--pair-scale", str(job.pair_scale),
        "--perturb-eps", str(job.perturb_eps),
        "--progress-every", str(args.progress_every),
        "--snapshot-every", str(args.snapshot_every),
        "--candidate-threshold-mi", str(args.candidate_threshold_mi),
        "--candidate-threshold-ccorr", str(args.candidate_threshold_ccorr),
        "--candidate-window", str(args.candidate_window),
        "--margin-threshold", str(args.margin_threshold),
        "--consensus-window", str(args.consensus_window),
        "--top-candidates", str(args.top_candidates),
        "--json-out", str(out_json),
    ]
    return cmd


def run_one(job: SweepJob, args: argparse.Namespace, python_exe: str, observer_script: Path, run_dir: Path) -> Dict[str, Any]:
    tag = f"seed{job.seed}_pair{job.pair_scale:g}_eps{job.perturb_eps:g}".replace(".", "p")
    out_json = run_dir / f"{tag}.json"
    cmd = build_command(python_exe, observer_script, out_json, args, job)

    t0 = time.perf_counter()
    stderr_tail = ""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout_sec if args.timeout_sec > 0 else None,
            check=False,
        )
        elapsed = time.perf_counter() - t0

        if proc.stderr:
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-10:])

        if proc.returncode != 0:
            return summarize_run({}, job, elapsed, status=f"error:returncode_{proc.returncode}", stderr_tail=stderr_tail)

        if not out_json.exists():
            return summarize_run({}, job, elapsed, status="error:no_output_json", stderr_tail=stderr_tail)

        with open(out_json, "r", encoding="utf-8") as f:
            run_json = json.load(f)

        summary = summarize_run(run_json, job, elapsed, status="ok", stderr_tail=stderr_tail)

        if not args.keep_run_json:
            try:
                out_json.unlink()
            except OSError:
                pass

        return summary

    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        stderr_tail = (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else ""
        try:
            if out_json.exists() and not args.keep_run_json:
                out_json.unlink()
        except OSError:
            pass
        return summarize_run({}, job, elapsed, status="error:timeout", stderr_tail=stderr_tail)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        try:
            if out_json.exists() and not args.keep_run_json:
                out_json.unlink()
        except OSError:
            pass
        return summarize_run({}, job, elapsed, status=f"error:{type(e).__name__}", stderr_tail=str(e))


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_runs(rows: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(
        key=lambda r: (
            int(r.get("strict_episode_count", 0)),
            int(r.get("strongest_episode_length", 0)),
            int(r.get("winner_count", 0)),
            1 if r.get("winner_mi_activation") else 0,
            1 if r.get("winner_ccorr_activation") else 0,
            -float(r.get("elapsed_sec", 0.0)),
        ),
        reverse=True,
    )
    return ok[:top_k]


def make_jobs(seeds: List[int], pair_scales: List[float], perturb_epses: List[float]) -> List[SweepJob]:
    return [SweepJob(seed=s, pair_scale=p, perturb_eps=e) for s, p, e in itertools.product(seeds, pair_scales, perturb_epses)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parallel seed × parameter sweep runner for hsf_mesoscape_fission_observer.py. "
            "Keeps outputs small by default by extracting compact summaries and deleting per-run JSON files."
        )
    )
    p.add_argument("--observer-script", type=str, default="hsf_mesoscape_fission_observer.py")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--initial-state", choices=["basis_zero", "random", "perturbed_zero"], default="perturbed_zero")

    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=300)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--snapshot-every", type=int, default=50)

    p.add_argument("--candidate-threshold-mi", type=float, default=1e-7)
    p.add_argument("--candidate-threshold-ccorr", type=float, default=1e-7)
    p.add_argument("--candidate-window", type=int, default=5)
    p.add_argument("--margin-threshold", type=float, default=1e-8)
    p.add_argument("--consensus-window", type=int, default=10)
    p.add_argument("--top-candidates", type=int, default=5)

    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--pair-scales", type=str, default="0.12,0.25,0.5,1.0")
    p.add_argument("--perturb-epses", type=str, default="0.01,0.02,0.05")

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument("--keep-run-json", action="store_true", default=False)
    p.add_argument("--outdir", type=str, default="hsf_fission_sweep_out")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    observer_script = Path(args.observer_script).resolve()
    if not observer_script.exists():
        raise FileNotFoundError(f"Observer script not found: {observer_script}")

    outdir = Path(args.outdir).resolve()
    run_dir = outdir / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_csv_ints(args.seeds)
    pair_scales = parse_csv_floats(args.pair_scales)
    perturb_epses = parse_csv_floats(args.perturb_epses)
    jobs = make_jobs(seeds, pair_scales, perturb_epses)

    # On a single GPU, too many concurrent processes often hurts more than it helps.
    # Keep the option, but clamp to 1 by default for GPU unless the user explicitly asks otherwise.
    workers = int(args.workers)
    if args.device == "gpu" and args.workers == 4:
        workers = 1

    summaries: List[Dict[str, Any]] = []

    if workers <= 1:
        for job in jobs:
            summaries.append(run_one(job, args, args.python_exe, observer_script, run_dir))
    else:
        # Use starmap-friendly closure data.
        ctx = mp.get_context("spawn")
        payloads = [(job, args, args.python_exe, observer_script, run_dir) for job in jobs]
        with ctx.Pool(processes=workers) as pool:
            summaries = pool.starmap(run_one, payloads)

    summary_csv = outdir / "summary.csv"
    summary_json = outdir / "summary.json"

    write_csv(summaries, summary_csv)

    payload = {
        "script": "hsf_mesoscape_fission_observer_sweep.py",
        "observer_script": str(observer_script),
        "device": args.device,
        "workers_used": workers,
        "grid": {
            "seeds": seeds,
            "pair_scales": pair_scales,
            "perturb_epses": perturb_epses,
        },
        "settings": {
            "initial_state": args.initial_state,
            "n_max": args.n_max,
            "n_init": args.n_init,
            "total_steps": args.total_steps,
            "candidate_threshold_mi": args.candidate_threshold_mi,
            "candidate_threshold_ccorr": args.candidate_threshold_ccorr,
            "candidate_window": args.candidate_window,
            "margin_threshold": args.margin_threshold,
            "consensus_window": args.consensus_window,
            "top_candidates": args.top_candidates,
            "keep_run_json": bool(args.keep_run_json),
        },
        "results": summaries,
        "best_runs": best_runs(summaries, top_k=20),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote {summary_csv}")
    print(f"wrote {summary_json}")


if __name__ == "__main__":
    main()