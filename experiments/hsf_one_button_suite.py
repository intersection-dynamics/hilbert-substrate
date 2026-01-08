#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSF One-Button Suite (single script)
====================================

Runs a curated set of HSF experiments (Paper II + Paper III) and packages everything
into ONE output folder automatically.

Key design goals (per your request):
- ONE SCRIPT runs the experiments AND packages outputs.
- No "pick these filenames" games.
- Robust to runner-CLI drift: we inspect each runner's --help and only pass supported flags.
- If an experiment isn't supported by your current runners (missing flag/choice), we SKIP it
  but still produce a complete suite report with "skipped reasons".

What it runs (when supported by your runners):
A) Paper II Endpoints
   1) LOCAL scramble  + recovery (default: both)  -> expects success
   2) GLOBAL scramble + recovery (default: both)  -> expects failure / barrier

B) (Optional) Paper II Transition Curve (if runner supports a depth-like knob)
   - local circuit depth sweep (you can supply depths)

C) Paper III Matter Robustness (if runner supports --fermion-audit)
   - runs fermion audit during A/B conditions

D) Optional Multichain stress test (if multichain runner is found and flags match)
   - Intended to show best-effort recovery within accessible moves.

Outputs (in ONE folder):
- runs/                     (all JSONL + captured stdout logs)
- inputs/                   (copies of JSONL inputs used for packaging)
- combined_flat.csv
- summary_by_condition.json
- REPORT.md
- manifest.json
- optional ZIP of entire folder

Windows one-liner example:
  python hsf_one_button_suite.py --repo "C:\GitHub\hilbert_substrate" --out "outputs\SUITE_latest" --N 8 --model xx --seeds 32 --zip

Tip:
- If the multichain command is skipped due to CLI mismatch, run:
    python experiments\scramble_recover_numba.py --help
  and paste the help here; I’ll lock it in perfectly.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

def run_subprocess(cmd: List[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(">>> " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        return int(p.returncode)

def try_help(py: str, script: Path, cwd: Path) -> str:
    try:
        p = subprocess.run([py, str(script), "--help"], cwd=str(cwd),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="ignore")
        return p.stdout or ""
    except Exception:
        return ""

def flag_supported(help_text: str, flag: str) -> bool:
    return flag in help_text

def choice_supported(help_text: str, flag: str, value: str) -> bool:
    # crude but works: look for either "{a,b,c}" or "choices:" lines
    # We just ensure the value appears somewhere near the flag.
    idx = help_text.find(flag)
    if idx == -1:
        return False
    window = help_text[idx: idx + 300]
    return (value in window) or (value in help_text)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


# ----------------------------
# Packager (embedded)
# ----------------------------

def is_finite(x: float) -> bool:
    return (not math.isnan(x)) and (not math.isinf(x))

def stats(arr: List[float]) -> Dict[str, float]:
    xs = [float(x) for x in arr if is_finite(float(x))]
    if not xs:
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    a = np.array(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def safe_get(obj: Any, path: List[str], default: Any = None) -> Any:
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def first_present(obj: Dict[str, Any], paths: List[List[str]], default: Any = None) -> Any:
    for p in paths:
        v = safe_get(obj, p, None)
        if v is not None:
            return v
    return default

def to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def to_int(x: Any) -> int:
    try:
        if x is None:
            return -1
        return int(x)
    except Exception:
        return -1

def extract_flat_row(r: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    seed = first_present(r, [["seed"], ["run","seed"]], default=None)
    N = first_present(r, [["N"], ["run","N"], ["meta","N"]], default=None)
    model = first_present(r, [["model"], ["run","model"], ["meta","model"]], default=None)
    scramble = first_present(r, [["scramble"], ["run","scramble"], ["meta","scramble"]], default=None)

    strobe_obj = first_present(r, [["strobe","objective"], ["strobe_objective"], ["config","strobe_objective"]], default=None)
    strobe_edges = first_present(r, [["strobe","edges"], ["strobe_edges"], ["config","strobe_edges"]], default=None)
    chains = first_present(r, [["strobe","chains"], ["chains"], ["config","chains"]], default=None)
    cores = first_present(r, [["strobe","cores"], ["cores"], ["config","cores"]], default=None)
    cycles = first_present(r, [["strobe","cycles"], ["cycles"], ["config","cycles"]], default=None)

    flow_steps = first_present(r, [["flow","steps"], ["flow_steps"], ["config","flow_steps"]], default=None)
    dt = first_present(r, [["flow","dt"], ["dt"], ["config","dt"]], default=None)
    p = first_present(r, [["flow","p"], ["p"], ["config","p"]], default=None)
    max_weight = first_present(r, [["flow","max_weight"], ["max_weight"], ["config","max_weight"]], default=None)

    sparse_red = first_present(r, [["metrics","sparse_reduction"], ["sparse_reduction"]], default=None)
    signal_red = first_present(r, [["metrics","signal_entropy_reduction"], ["signal_entropy_reduction"]], default=None)
    top_share = first_present(r, [["metrics","topN_share_final"], ["metrics","topN_share"], ["topN_share_final"]], default=None)
    V2_red = first_present(r, [["metrics","V2_ring_reduction"], ["V2_ring_reduction"]], default=None)
    V2_over_V1_post = first_present(r, [["metrics","V2_over_V1_final"], ["metrics","V2_over_V1_post"], ["V2_over_V1_post"]], default=None)

    sparse_ok = first_present(r, [["success","sparse_ok"], ["locality_recovered_sparse"], ["success_sparse"]], default=None)
    signal_ok = first_present(r, [["success","signal_ok"], ["locality_recovered_signal"], ["success_signal"]], default=None)

    fa = first_present(r, [["fermion_audit_results"], ["fermion_audit"]], default={})
    jw_max = add_err = kappa = None
    if isinstance(fa, dict):
        jw_max = first_present(fa, [["jw_max"], ["jw_anticommutators","max_abs"]], default=None)
        add_err = first_present(fa, [["additivity_error"], ["sector_additivity","error"]], default=None)
        kappa = first_present(fa, [["kappa_proxy"], ["pauli_pressure","kappa_proxy"]], default=None)

    return {
        "source_file": source_file,
        "seed": to_int(seed),
        "N": to_int(N),
        "model": model,
        "scramble": scramble,
        "strobe_objective": strobe_obj,
        "strobe_edges": strobe_edges,
        "chains": to_int(chains),
        "cores": to_int(cores),
        "cycles": to_int(cycles),
        "flow_steps": to_int(flow_steps),
        "dt": to_float(dt),
        "p": to_int(p),
        "max_weight": to_int(max_weight),
        "sparse_reduction": to_float(sparse_red),
        "signal_entropy_reduction": to_float(signal_red),
        "topN_share_final": to_float(top_share),
        "V2_ring_reduction": to_float(V2_red),
        "V2_over_V1_post": to_float(V2_over_V1_post),
        "sparse_ok": bool(sparse_ok) if sparse_ok is not None else None,
        "signal_ok": bool(signal_ok) if signal_ok is not None else None,
        "jw_max": to_float(jw_max),
        "additivity_error": to_float(add_err),
        "kappa_proxy": to_float(kappa),
    }

def condition_key(row: Dict[str, Any]) -> Tuple:
    return (
        row.get("N"),
        row.get("model"),
        row.get("scramble"),
        row.get("strobe_objective"),
        row.get("strobe_edges"),
        row.get("flow_steps"),
        row.get("max_weight"),
        row.get("chains"),
    )

def package_outputs(jsonl_files: List[Path], outdir: Path) -> None:
    ensure_dir(outdir)
    ensure_dir(outdir / "inputs")

    # Copy inputs
    input_meta: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []

    for f in jsonl_files:
        rows = read_jsonl(f)
        input_meta.append({"path": str(f), "rows": len(rows), "bytes": f.stat().st_size})
        shutil.copy2(f, outdir / "inputs" / f.name)
        for r in rows:
            flat_rows.append(extract_flat_row(r, source_file=f.name))

    # CSV
    csv_path = outdir / "combined_flat.csv"
    if flat_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(flat_rows[0].keys()))
            w.writeheader()
            for r in flat_rows:
                w.writerow(r)
    else:
        write_text(csv_path, "")

    # Condition summary
    buckets: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in flat_rows:
        buckets.setdefault(condition_key(r), []).append(r)

    summary: Dict[str, Any] = {}
    for k, rows in buckets.items():
        key_str = "|".join([str(x) for x in k])
        summary[key_str] = {
            "condition": {
                "N": k[0], "model": k[1], "scramble": k[2],
                "strobe_objective": k[3], "strobe_edges": k[4],
                "flow_steps": k[5], "max_weight": k[6], "chains": k[7],
            },
            "runs": len(rows),
            "success_rate_sparse": float(sum(1 for r in rows if r.get("sparse_ok") is True) / max(1, len(rows))),
            "success_rate_signal": float(sum(1 for r in rows if r.get("signal_ok") is True) / max(1, len(rows))),
            "metrics": {
                "sparse_reduction": stats([r.get("sparse_reduction", float("nan")) for r in rows]),
                "signal_entropy_reduction": stats([r.get("signal_entropy_reduction", float("nan")) for r in rows]),
                "topN_share_final": stats([r.get("topN_share_final", float("nan")) for r in rows]),
                "V2_ring_reduction": stats([r.get("V2_ring_reduction", float("nan")) for r in rows]),
                "V2_over_V1_post": stats([r.get("V2_over_V1_post", float("nan")) for r in rows]),
                "jw_max": stats([r.get("jw_max", float("nan")) for r in rows]),
                "additivity_error": stats([r.get("additivity_error", float("nan")) for r in rows]),
                "kappa_proxy": stats([r.get("kappa_proxy", float("nan")) for r in rows]),
            },
        }

    write_text(outdir / "summary_by_condition.json", json.dumps(summary, indent=2))

    # Markdown report
    lines = ["# HSF Suite Report", f"- Created: `{now_utc_iso()}`", f"- JSONL files packaged: {len(jsonl_files)}", ""]
    lines.append("## Conditions\n")
    for _, block in summary.items():
        c = block["condition"]
        m = block["metrics"]
        lines.append(f"### N={c['N']} model={c['model']} scramble={c['scramble']} objective={c['strobe_objective']} edges={c['strobe_edges']}")
        lines.append(f"- runs: {block['runs']}")
        lines.append(f"- success_rate_sparse: {block['success_rate_sparse']:.3f}")
        lines.append(f"- success_rate_signal: {block['success_rate_signal']:.3f}")
        lines.append(f"- sparse_reduction median: {m['sparse_reduction']['median']:.3f}")
        lines.append(f"- signal_entropy_reduction median: {m['signal_entropy_reduction']['median']:.3f}")
        lines.append(f"- V2_over_V1_post median: {m['V2_over_V1_post']['median']:.4f}")
        lines.append(f"- jw_max median: {m['jw_max']['median']:.4g}")
        lines.append("")
    write_text(outdir / "REPORT.md", "\n".join(lines))

    manifest = {
        "created_utc": now_utc_iso(),
        "tool": "hsf_one_button_suite.py",
        "jsonl_inputs": input_meta,
        "outputs": ["combined_flat.csv", "summary_by_condition.json", "REPORT.md", "manifest.json", "inputs/"],
    }
    write_text(outdir / "manifest.json", json.dumps(manifest, indent=2))


# ----------------------------
# Experiment suite logic
# ----------------------------

@dataclass
class SuiteResult:
    name: str
    status: str              # "ok" | "skipped" | "failed"
    jsonl: Optional[str] = None
    log: Optional[str] = None
    reason: Optional[str] = None
    returncode: Optional[int] = None

def find_runner(repo: Path, candidates: List[str]) -> Optional[Path]:
    for rel in candidates:
        p = repo / rel
        if p.exists():
            return p
    return None

def add_arg(cmd: List[str], help_text: str, flag: str, value: Optional[str] = None) -> None:
    if not flag_supported(help_text, flag):
        return
    cmd.append(flag)
    if value is not None:
        cmd.append(value)

def run_single_chain(py: str, repo: Path, script: Path, help_text: str, out_runs: Path,
                    name: str, N: int, model: str, scramble: str, recover: str,
                    strobe_objective: str, cycles: int, flow_steps: int, dt: float, p: int,
                    max_weight: int, seed_start: int, seed_count: int,
                    blas_threads: int, jobs: int, fermion_audit: bool,
                    extra: Dict[str, str]) -> SuiteResult:

    jsonl_path = out_runs / f"{name}.jsonl"
    log_path = out_runs / f"{name}.log.txt"

    # Validate choices if help exposes them
    if flag_supported(help_text, "--scramble") and not choice_supported(help_text, "--scramble", scramble):
        return SuiteResult(name=name, status="skipped", reason=f"Runner does not support --scramble {scramble}")

    cmd = [py, str(script)]

    # Core flags (only pass if supported)
    add_arg(cmd, help_text, "--N", str(N))
    add_arg(cmd, help_text, "--model", str(model))
    add_arg(cmd, help_text, "--scramble", scramble)

    # Recovery selection
    if flag_supported(help_text, "--recover"):
        # if recover choice isn't supported, fall back to "flow" then "strobe" then "none"
        if choice_supported(help_text, "--recover", recover):
            add_arg(cmd, help_text, "--recover", recover)
        elif choice_supported(help_text, "--recover", "flow"):
            add_arg(cmd, help_text, "--recover", "flow")
        elif choice_supported(help_text, "--recover", "strobe"):
            add_arg(cmd, help_text, "--recover", "strobe")
        elif choice_supported(help_text, "--recover", "none"):
            add_arg(cmd, help_text, "--recover", "none")

    add_arg(cmd, help_text, "--strobe-objective", strobe_objective)
    add_arg(cmd, help_text, "--cycles", str(cycles))
    add_arg(cmd, help_text, "--flow-steps", str(flow_steps))
    add_arg(cmd, help_text, "--dt", str(dt))
    add_arg(cmd, help_text, "--p", str(p))
    add_arg(cmd, help_text, "--max-weight", str(max_weight))
    add_arg(cmd, help_text, "--seed-start", str(seed_start))
    add_arg(cmd, help_text, "--seed-count", str(seed_count))
    add_arg(cmd, help_text, "--jobs", str(jobs))
    add_arg(cmd, help_text, "--blas-threads", str(blas_threads))
    if flag_supported(help_text, "--progress"):
        cmd.append("--progress")

    # Output
    # Prefer --partial-output if present; else --output
    if flag_supported(help_text, "--partial-output"):
        cmd += ["--partial-output", str(jsonl_path)]
    elif flag_supported(help_text, "--output"):
        cmd += ["--output", str(jsonl_path)]
    else:
        return SuiteResult(name=name, status="skipped", reason="Runner has no --output or --partial-output")

    # Fermion audit (optional)
    if fermion_audit and flag_supported(help_text, "--fermion-audit"):
        cmd.append("--fermion-audit")

    # Extra flags (e.g., depth knobs) if present
    for k, v in (extra or {}).items():
        if flag_supported(help_text, k):
            cmd += [k, str(v)]

    rc = run_subprocess(cmd, cwd=repo, log_path=log_path)
    if rc == 0 and jsonl_path.exists():
        return SuiteResult(name=name, status="ok", jsonl=str(jsonl_path), log=str(log_path), returncode=rc)
    return SuiteResult(name=name, status="failed", jsonl=str(jsonl_path), log=str(log_path), reason="nonzero return or missing output", returncode=rc)

def run_multichain_best_effort(py: str, repo: Path, script: Path, out_runs: Path,
                               name: str, N: int, model: str, scramble: str,
                               chains: int, cores: int, cycles: int, seed: int) -> SuiteResult:
    # Because your multichain scripts have varied CLIs, we do a best-effort:
    # - call --help and only pass supported flags
    help_text = try_help(py, script, repo)
    jsonl_path = out_runs / f"{name}.jsonl"
    log_path = out_runs / f"{name}.log.txt"

    cmd = [py, str(script)]
    add_arg(cmd, help_text, "--N", str(N))
    add_arg(cmd, help_text, "--model", str(model))
    add_arg(cmd, help_text, "--scramble", scramble)
    add_arg(cmd, help_text, "--chains", str(chains))
    add_arg(cmd, help_text, "--cores", str(cores))
    add_arg(cmd, help_text, "--cycles", str(cycles))
    add_arg(cmd, help_text, "--seed", str(seed))
    if flag_supported(help_text, "--progress"):
        cmd.append("--progress")
    if flag_supported(help_text, "--partial-output"):
        cmd += ["--partial-output", str(jsonl_path)]
    elif flag_supported(help_text, "--output"):
        cmd += ["--output", str(jsonl_path)]
    else:
        # If it doesn't support output, still run and log, but can't package
        rc = run_subprocess(cmd, cwd=repo, log_path=log_path)
        return SuiteResult(name=name, status="skipped", log=str(log_path), reason="multichain runner lacks --output/--partial-output", returncode=rc)

    rc = run_subprocess(cmd, cwd=repo, log_path=log_path)
    if rc == 0 and jsonl_path.exists():
        return SuiteResult(name=name, status="ok", jsonl=str(jsonl_path), log=str(log_path), returncode=rc)
    return SuiteResult(name=name, status="failed", jsonl=str(jsonl_path), log=str(log_path), reason="nonzero return or missing output", returncode=rc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to hilbert_substrate repo root")
    ap.add_argument("--out", required=True, help="Output folder (relative to repo or absolute)")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use (default: current)")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--model", default="xx", choices=["xx", "xxz", "xxx"])
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=8000)
    ap.add_argument("--flow-steps", type=int, default=30)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--max-weight", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--strobe-objective", default="sparse", choices=["sparse", "signal", "range"])
    ap.add_argument("--recover", default="both", help="Preferred recover mode, if supported by runner (e.g. both/flow/strobe/none)")
    ap.add_argument("--fermion-audit", action="store_true")
    ap.add_argument("--transition-depths", default="", help="Comma list like 0,1,2,4,8. Only runs if a depth knob is detected.")
    ap.add_argument("--chains", type=int, default=12)
    ap.add_argument("--cores", type=int, default=12)
    ap.add_argument("--skip-multichain", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.exists():
        print("Repo not found:", repo)
        return 2

    outdir = (Path(args.out) if Path(args.out).is_absolute() else (repo / args.out)).resolve()
    runs_dir = outdir / "runs"
    ensure_dir(runs_dir)

    # Locate runners
    single_runner = find_runner(repo, [
        "experiments/scramble_recover_test_patched_signal_cache.py",
        "experiments/scramble_recover_test_patched_signal_cache_gpu.py",
        "consolidation/scripts/scramble_recover_test_patched_signal_cache.py",
        "consolidation/scripts/scramble_recover_test_patched_signal_cache_gpu.py",
        "scramble_recover_test_patched_signal_cache.py",
    ])
    if single_runner is None:
        print("Could not find single-chain runner. Expected one of:")
        print("  experiments/scramble_recover_test_patched_signal_cache*.py")
        return 2

    multi_runner = find_runner(repo, [
        "experiments/scramble_recover_numba.py",
        "consolidation/scripts/scramble_recover_numba.py",
        "scramble_recover_numba.py",
    ])

    help_text = try_help(args.python, single_runner, repo)

    suite: List[SuiteResult] = []
    produced: List[Path] = []
    skipped: List[SuiteResult] = []

    # Paper II endpoints
    suite.append(run_single_chain(
        args.python, repo, single_runner, help_text, runs_dir,
        name=f"paper2_endpoint_LOCAL_N{args.N}_{args.model}",
        N=args.N, model=args.model, scramble="local",
        recover=args.recover, strobe_objective=args.strobe_objective,
        cycles=args.cycles, flow_steps=args.flow_steps, dt=args.dt, p=args.p,
        max_weight=args.max_weight, seed_start=args.seed_start, seed_count=args.seeds,
        blas_threads=args.blas_threads, jobs=args.jobs, fermion_audit=args.fermion_audit,
        extra={}
    ))
    suite.append(run_single_chain(
        args.python, repo, single_runner, help_text, runs_dir,
        name=f"paper2_endpoint_GLOBAL_N{args.N}_{args.model}",
        N=args.N, model=args.model, scramble="global",
        recover=args.recover, strobe_objective=args.strobe_objective,
        cycles=args.cycles, flow_steps=args.flow_steps, dt=args.dt, p=args.p,
        max_weight=args.max_weight, seed_start=args.seed_start, seed_count=args.seeds,
        blas_threads=args.blas_threads, jobs=args.jobs, fermion_audit=args.fermion_audit,
        extra={}
    ))

    # Transition curve (only if we can detect a depth knob)
    depths = []
    if args.transition_depths.strip():
        depths = [int(x.strip()) for x in args.transition_depths.split(",") if x.strip()]
    depth_flags = [ "--scramble-depth", "--depth", "--circuit-depth", "--scramble_layers" ]
    depth_flag = None
    for df in depth_flags:
        if flag_supported(help_text, df):
            depth_flag = df
            break

    if depths and depth_flag and choice_supported(help_text, "--scramble", "local"):
        for d in depths:
            suite.append(run_single_chain(
                args.python, repo, single_runner, help_text, runs_dir,
                name=f"paper2_transition_localDepth{d}_N{args.N}_{args.model}",
                N=args.N, model=args.model, scramble="local",
                recover=args.recover, strobe_objective=args.strobe_objective,
                cycles=args.cycles, flow_steps=args.flow_steps, dt=args.dt, p=args.p,
                max_weight=args.max_weight, seed_start=args.seed_start, seed_count=args.seeds,
                blas_threads=args.blas_threads, jobs=args.jobs, fermion_audit=args.fermion_audit,
                extra={ depth_flag: str(d) }
            ))
    elif depths and not depth_flag:
        suite.append(SuiteResult(name="paper2_transition_curve", status="skipped",
                                 reason="No depth knob flag detected in runner --help (expected one of --scramble-depth/--depth/--circuit-depth/--scramble_layers)"))

    # Optional multichain global stress test
    if not args.skip_multichain:
        if multi_runner is None:
            suite.append(SuiteResult(name="paper2_multichain_global", status="skipped", reason="No multichain runner found (scramble_recover_numba.py)"))
        else:
            suite.append(run_multichain_best_effort(
                args.python, repo, multi_runner, runs_dir,
                name=f"paper2_multichain_GLOBAL_N{args.N}_{args.model}",
                N=args.N, model=args.model, scramble="global",
                chains=args.chains, cores=args.cores, cycles=args.cycles,
                seed=args.seed_start
            ))

    # Collect produced JSONL
    for r in suite:
        if r.status == "ok" and r.jsonl:
            produced.append(Path(r.jsonl))
        elif r.status == "skipped":
            skipped.append(r)

    # Write suite status summary
    status = {
        "created_utc": now_utc_iso(),
        "repo": str(repo),
        "single_runner": str(single_runner),
        "multi_runner": str(multi_runner) if multi_runner else None,
        "params": {
            "N": args.N, "model": args.model, "seeds": args.seeds, "seed_start": args.seed_start,
            "cycles": args.cycles, "flow_steps": args.flow_steps, "dt": args.dt, "p": args.p,
            "max_weight": args.max_weight, "recover": args.recover,
            "strobe_objective": args.strobe_objective,
            "fermion_audit": bool(args.fermion_audit),
            "transition_depths": depths,
        },
        "runs": [r.__dict__ for r in suite],
    }
    write_text(outdir / "suite_status.json", json.dumps(status, indent=2))

    if not produced:
        # still write a top-level REPORT.md explaining what happened
        lines = [
            "# HSF Suite Report (no packaged outputs)",
            f"- Created: `{now_utc_iso()}`",
            "",
            "No JSONL outputs were produced. This usually means runner CLI flags changed.",
            "Open `suite_status.json` in this folder to see which commands ran and why they failed/skipped.",
            "",
            f"Single runner used: `{single_runner}`",
        ]
        write_text(outdir / "REPORT.md", "\n".join(lines))
        print("No JSONL produced. See:", outdir / "suite_status.json")
        return 1

    # Package
    package_outputs(produced, outdir)

    # Append skipped reasons to REPORT
    rep_path = outdir / "REPORT.md"
    rep = rep_path.read_text(encoding="utf-8") if rep_path.exists() else ""
    if skipped:
        rep += "\n\n## Skipped\n"
        for s in skipped:
            rep += f"- **{s.name}**: {s.reason}\n"
        write_text(rep_path, rep)

    # Zip if requested
    if args.zip:
        zip_path = Path(str(outdir) + ".zip")
        import zipfile
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(outdir):
                for name in files:
                    full = Path(root) / name
                    rel = full.relative_to(outdir)
                    z.write(full, arcname=str(rel))
        print("Wrote ZIP:", zip_path)

    print("DONE. Open:", outdir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
