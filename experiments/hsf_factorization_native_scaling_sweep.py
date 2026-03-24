# filename: hsf_factorization_native_scaling_sweep.py
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def default_factorizations_by_dimension(total_dimension: int) -> str:
    presets: Dict[int, str] = {
        81: "3x3x3x3;3x3x9;3x27;9x9",
        243: "3x3x3x3x3;3x3x3x9;3x3x27;3x9x9;9x27;3x81",
        729: "3x3x3x3x3x3;3x3x3x3x9;3x3x3x27;3x3x9x9;3x9x27;9x9x9;3x243;9x81;27x27",
    }
    if total_dimension not in presets:
        raise ValueError(f"No default candidate factorization set for total dimension {total_dimension}")
    return presets[total_dimension]


def run_one(
    python_exe: str,
    observer_script: Path,
    outdir: Path,
    total_dimension: int,
    seed: int,
    perturb_eps: float,
    pair_scale: float,
    total_steps: int,
    snapshot_every: int,
    progress_every: int,
    initial_state: str,
    candidate_factorizations: str,
) -> Dict[str, Any]:
    tag = f"d{total_dimension}_seed{seed}_eps{perturb_eps:g}_pair{pair_scale:g}".replace(".", "p")
    json_out = outdir / f"{tag}.json"

    cmd = [
        python_exe,
        str(observer_script),
        "--total-dimension", str(total_dimension),
        "--seed", str(seed),
        "--perturb-eps", str(perturb_eps),
        "--pair-scale", str(pair_scale),
        "--initial-state", initial_state,
        "--candidate-factorizations", candidate_factorizations,
        "--total-steps", str(total_steps),
        "--snapshot-every", str(snapshot_every),
        "--progress-every", str(progress_every),
        "--json-out", str(json_out),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0

    row: Dict[str, Any] = {
        "total_dimension": total_dimension,
        "seed": seed,
        "perturb_eps": perturb_eps,
        "pair_scale": pair_scale,
        "elapsed_sec": elapsed,
        "status": "ok" if proc.returncode == 0 and json_out.exists() else f"error_{proc.returncode}",
        "best_factorization": None,
        "best_accessibility_score": None,
        "runner_up_factorization": None,
        "runner_up_accessibility_score": None,
        "top_gap": None,
        "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-10:]) if proc.stderr else "",
    }

    if row["status"] != "ok":
        return row

    with open(json_out, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    if results:
        best = results[0]
        row["best_factorization"] = "x".join(str(x) for x in best["factorization"])
        row["best_accessibility_score"] = float(best["summary"]["accessibility_score"])

        if len(results) > 1:
            second = results[1]
            row["runner_up_factorization"] = "x".join(str(x) for x in second["factorization"])
            row["runner_up_accessibility_score"] = float(second["summary"]["accessibility_score"])
            row["top_gap"] = float(best["summary"]["accessibility_score"] - second["summary"]["accessibility_score"])

        for item in results:
            fac = "x".join(str(x) for x in item["factorization"])
            prefix = f"fac_{fac}_"
            summary = item["summary"]
            row[prefix + "score"] = float(summary["accessibility_score"])
            row[prefix + "mean_mi"] = float(summary["mean_mean_pair_mi_score"])
            row[prefix + "entropy"] = float(summary["mean_pair_entropy_score"])
            row[prefix + "anti_dom"] = float(summary["mean_anti_dominance_score"])
            row[prefix + "pair_count"] = float(summary["mean_pair_count_score"])
            row[prefix + "core_persist"] = float(summary["mean_core_persistence_score"])
            row[prefix + "entropy_balance"] = float(summary["mean_single_entropy_balance_score"])
            row[prefix + "stability"] = float(summary["temporal_stability_score"])

    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Scaling sweep for hsf_factorization_native_observer.py. "
            "Tests whether maximal qutrit refinement remains the winner as total dimension grows."
        )
    )
    p.add_argument("--observer-script", type=str, default="hsf_factorization_native_observer.py")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--dimensions", type=str, default="81,243,729")
    p.add_argument("--initial-state", choices=["basis_zero", "random", "perturbed_zero"], default="perturbed_zero")
    p.add_argument("--total-steps", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=0)

    p.add_argument("--seeds", type=str, default="1,2,3,4")
    p.add_argument("--perturb-epses", type=str, default="0.01,0.02")
    p.add_argument("--pair-scales", type=str, default="0.08,0.12,0.25")
    p.add_argument("--outdir", type=str, default="hsf_factorization_native_scaling_sweep_out")
    args = p.parse_args()

    observer_script = Path(args.observer_script).resolve()
    outdir = Path(args.outdir).resolve()
    run_dir = outdir / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    dimensions = parse_csv_ints(args.dimensions)
    seeds = parse_csv_ints(args.seeds)
    perturb_epses = parse_csv_floats(args.perturb_epses)
    pair_scales = parse_csv_floats(args.pair_scales)

    rows: List[Dict[str, Any]] = []
    for total_dimension, seed, eps, ps in itertools.product(dimensions, seeds, perturb_epses, pair_scales):
        factorization_set = default_factorizations_by_dimension(total_dimension)
        row = run_one(
            python_exe=args.python_exe,
            observer_script=observer_script,
            outdir=run_dir,
            total_dimension=total_dimension,
            seed=seed,
            perturb_eps=eps,
            pair_scale=ps,
            total_steps=args.total_steps,
            snapshot_every=args.snapshot_every,
            progress_every=args.progress_every,
            initial_state=args.initial_state,
            candidate_factorizations=factorization_set,
        )
        rows.append(row)
        print(
            f"dim={total_dimension} seed={seed} eps={eps:g} pair={ps:g} "
            f"status={row['status']} best={row['best_factorization']}"
        )

    csv_path = outdir / "summary.csv"
    if rows:
        fieldnames: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    winner_counts_by_dim: Dict[str, Dict[str, int]] = {}
    runner_up_counts_by_dim: Dict[str, Dict[str, int]] = {}
    top_gap_summary_by_dim: Dict[str, Dict[str, float]] = {}

    for dim in dimensions:
        dim_rows = [r for r in rows if int(r["total_dimension"]) == dim and r["status"] == "ok"]
        win_counts: Dict[str, int] = {}
        ru_counts: Dict[str, int] = {}
        gaps: List[float] = []

        for row in dim_rows:
            bf = row.get("best_factorization")
            rf = row.get("runner_up_factorization")
            gap = row.get("top_gap")
            if bf:
                win_counts[str(bf)] = win_counts.get(str(bf), 0) + 1
            if rf:
                ru_counts[str(rf)] = ru_counts.get(str(rf), 0) + 1
            if gap is not None:
                gaps.append(float(gap))

        winner_counts_by_dim[str(dim)] = win_counts
        runner_up_counts_by_dim[str(dim)] = ru_counts
        if gaps:
            top_gap_summary_by_dim[str(dim)] = {
                "mean_top_gap": float(sum(gaps) / len(gaps)),
                "min_top_gap": float(min(gaps)),
                "max_top_gap": float(max(gaps)),
            }
        else:
            top_gap_summary_by_dim[str(dim)] = {}

    json_path = outdir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "hsf_factorization_native_scaling_sweep.py",
                "observer_script": str(observer_script),
                "grid": {
                    "dimensions": dimensions,
                    "seeds": seeds,
                    "perturb_epses": perturb_epses,
                    "pair_scales": pair_scales,
                },
                "winner_counts_by_dimension": winner_counts_by_dim,
                "runner_up_counts_by_dimension": runner_up_counts_by_dim,
                "top_gap_summary_by_dimension": top_gap_summary_by_dim,
                "rows": rows,
            },
            f,
            indent=2,
        )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()