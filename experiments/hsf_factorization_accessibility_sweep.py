# filename: hsf_factorization_accessibility_sweep.py
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


def run_one(
    python_exe: str,
    observer_script: Path,
    outdir: Path,
    seed: int,
    perturb_eps: float,
    pair_scale: float,
    device: str,
    total_steps: int,
    snapshot_every: int,
    progress_every: int,
    n_sites: int,
    site_dim: int,
    initial_state: str,
) -> Dict[str, Any]:
    tag = f"seed{seed}_eps{perturb_eps:g}_pair{pair_scale:g}".replace(".", "p")
    json_out = outdir / f"{tag}.json"

    cmd = [
        python_exe,
        str(observer_script),
        "--device", device,
        "--seed", str(seed),
        "--perturb-eps", str(perturb_eps),
        "--pair-scale", str(pair_scale),
        "--n-sites", str(n_sites),
        "--site-dim", str(site_dim),
        "--initial-state", initial_state,
        "--total-steps", str(total_steps),
        "--snapshot-every", str(snapshot_every),
        "--progress-every", str(progress_every),
        "--json-out", str(json_out),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0

    row: Dict[str, Any] = {
        "seed": seed,
        "perturb_eps": perturb_eps,
        "pair_scale": pair_scale,
        "elapsed_sec": elapsed,
        "status": "ok" if proc.returncode == 0 and json_out.exists() else f"error_{proc.returncode}",
        "best_factorization": None,
        "best_accessibility_score": None,
        "runner_up_factorization": None,
        "runner_up_accessibility_score": None,
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

        # copy the most important per-factorization summaries into compact columns
        for item in results:
            fac = "x".join(str(x) for x in item["factorization"])
            prefix = f"fac_{fac}_"
            summary = item["summary"]
            row[prefix + "score"] = float(summary["accessibility_score"])
            row[prefix + "locality"] = float(summary["mean_locality_like_score"])
            row[prefix + "leakage"] = float(summary["mean_signal_leakage_score"])
            row[prefix + "entropy"] = float(summary["mean_pair_entropy_score"])
            row[prefix + "anti_dom"] = float(summary["mean_anti_dominance_score"])
            row[prefix + "pair_count"] = float(summary["mean_pair_count_score"])
            row[prefix + "core_persist"] = float(summary["mean_core_persistence_score"])
            row[prefix + "ccorr"] = float(summary["mean_connected_corr"])

    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Robustness sweep for hsf_factorization_accessibility_observer.py. "
            "Runs a compact grid over seeds / perturbation / pair scale and writes summary CSV + JSON."
        )
    )
    p.add_argument("--observer-script", type=str, default="hsf_factorization_accessibility_observer.py")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--n-sites", type=int, default=4)
    p.add_argument("--site-dim", type=int, default=3)
    p.add_argument("--initial-state", choices=["basis_zero", "random", "perturbed_zero"], default="perturbed_zero")
    p.add_argument("--total-steps", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=0)

    p.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8")
    p.add_argument("--perturb-epses", type=str, default="0.01,0.02,0.05")
    p.add_argument("--pair-scales", type=str, default="0.08,0.12,0.25,0.5")
    p.add_argument("--outdir", type=str, default="hsf_factorization_accessibility_sweep_out")
    args = p.parse_args()

    observer_script = Path(args.observer_script).resolve()
    outdir = Path(args.outdir).resolve()
    run_dir = outdir / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_csv_ints(args.seeds)
    perturb_epses = parse_csv_floats(args.perturb_epses)
    pair_scales = parse_csv_floats(args.pair_scales)

    rows: List[Dict[str, Any]] = []
    for seed, eps, ps in itertools.product(seeds, perturb_epses, pair_scales):
        row = run_one(
            python_exe=args.python_exe,
            observer_script=observer_script,
            outdir=run_dir,
            seed=seed,
            perturb_eps=eps,
            pair_scale=ps,
            device=args.device,
            total_steps=args.total_steps,
            snapshot_every=args.snapshot_every,
            progress_every=args.progress_every,
            n_sites=args.n_sites,
            site_dim=args.site_dim,
            initial_state=args.initial_state,
        )
        rows.append(row)
        print(
            f"seed={seed} eps={eps:g} pair={ps:g} "
            f"status={row['status']} best={row['best_factorization']}"
        )

    # write CSV
    if rows:
        fieldnames: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        csv_path = outdir / "summary.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path = outdir / "summary.csv"

    # aggregate winner counts
    winner_counts: Dict[str, int] = {}
    for row in rows:
        bf = row.get("best_factorization")
        if bf:
            winner_counts[str(bf)] = winner_counts.get(str(bf), 0) + 1

    json_path = outdir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "hsf_factorization_accessibility_sweep.py",
                "observer_script": str(observer_script),
                "device": args.device,
                "grid": {
                    "seeds": seeds,
                    "perturb_epses": perturb_epses,
                    "pair_scales": pair_scales,
                },
                "winner_counts": winner_counts,
                "rows": rows,
            },
            f,
            indent=2,
        )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()