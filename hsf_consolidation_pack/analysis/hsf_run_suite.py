#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSF One-Button Experiment Suite
===============================

This is the "single script" runner that:
1) runs the core experiments (Paper II + Paper III) using your existing runner scripts
2) writes ALL outputs into ONE run folder (no filename games)
3) packages results into:
   - combined_flat.csv
   - summary_by_condition.json
   - REPORT.md
   - manifest.json
   - optional ZIP

It uses subprocess to call your existing experiment runners so you don't have to merge codebases.

Windows one-liner (from repo root):
  python consolidation\analysis\hsf_run_suite.py --repo "C:\GitHub\hilbert_substrate" --out "outputs\SUITE_latest" --N 8 --model xx --seeds 32 --chains 12 --cores 12 --cycles 8000 --flow-steps 30 --dt 0.001 --p 4 --max-weight 4 --zip

Notes:
- This suite expects these files to exist in your repo (default paths below):
  - experiments/scramble_recover_test_patched_signal_cache.py
  - experiments/scramble_recover_numba.py   (your multi-chain baseline)
  - consolidation/analysis/hsf_package_experiments.py (optional; suite has an internal packager too)
- If a runner doesn't support a flag (e.g. --fermion-audit), the suite will auto-skip that experiment and keep going.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import csv
import numpy as np


# ----------------------------
# Small packager (embedded)
# ----------------------------

def _now_utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

def _is_finite(x: float) -> bool:
    return (not math.isnan(x)) and (not math.isinf(x))

def _stats(arr: List[float]) -> Dict[str, float]:
    xs = [float(x) for x in arr if _is_finite(float(x))]
    if not xs:
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    a = np.array(xs, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(np.mean(a)), "min": float(np.min(a)), "max": float(np.max(a))}

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
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

def _safe_get(obj: Any, path: List[str], default: Any = None) -> Any:
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _first_present(obj: Dict[str, Any], paths: List[List[str]], default: Any = None) -> Any:
    for p in paths:
        v = _safe_get(obj, p, None)
        if v is not None:
            return v
    return default

def _to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def _to_int(x: Any) -> int:
    try:
        if x is None:
            return -1
        return int(x)
    except Exception:
        return -1

def _extract_flat_row(r: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    seed = _first_present(r, [["seed"], ["run","seed"]], default=None)
    N = _first_present(r, [["N"], ["run","N"], ["meta","N"]], default=None)
    model = _first_present(r, [["model"], ["run","model"], ["meta","model"]], default=None)
    scramble = _first_present(r, [["scramble"], ["run","scramble"], ["meta","scramble"]], default=None)

    strobe_obj = _first_present(r, [["strobe","objective"], ["strobe_objective"], ["config","strobe_objective"]], default=None)
    strobe_edges = _first_present(r, [["strobe","edges"], ["strobe_edges"], ["config","strobe_edges"]], default=None)
    chains = _first_present(r, [["strobe","chains"], ["chains"], ["config","chains"]], default=None)
    cores = _first_present(r, [["strobe","cores"], ["cores"], ["config","cores"]], default=None)
    cycles = _first_present(r, [["strobe","cycles"], ["cycles"], ["config","cycles"]], default=None)

    flow_steps = _first_present(r, [["flow","steps"], ["flow_steps"], ["config","flow_steps"]], default=None)
    dt = _first_present(r, [["flow","dt"], ["dt"], ["config","dt"]], default=None)
    p = _first_present(r, [["flow","p"], ["p"], ["config","p"]], default=None)
    max_weight = _first_present(r, [["flow","max_weight"], ["max_weight"], ["config","max_weight"]], default=None)

    m = r.get("metrics", {}) if isinstance(r.get("metrics", {}), dict) else {}
    sparse_red = _first_present(r, [["metrics","sparse_reduction"], ["sparse_reduction"]], default=None)
    signal_red = _first_present(r, [["metrics","signal_entropy_reduction"], ["signal_entropy_reduction"]], default=None)
    top_share = _first_present(r, [["metrics","topN_share_final"], ["metrics","topN_share"], ["topN_share_final"]], default=None)
    V2_red = _first_present(r, [["metrics","V2_ring_reduction"], ["V2_ring_reduction"]], default=None)
    V2_over_V1_post = _first_present(r, [["metrics","V2_over_V1_final"], ["metrics","V2_over_V1_post"], ["V2_over_V1_post"]], default=None)

    sparse_ok = _first_present(r, [["success","sparse_ok"], ["locality_recovered_sparse"], ["success_sparse"]], default=None)
    signal_ok = _first_present(r, [["success","signal_ok"], ["locality_recovered_signal"], ["success_signal"]], default=None)

    fa = _first_present(r, [["fermion_audit_results"], ["fermion_audit"]], default={})
    jw_max = add_err = kappa = None
    if isinstance(fa, dict):
        jw_max = _first_present(fa, [["jw_max"], ["jw_anticommutators","max_abs"]], default=None)
        add_err = _first_present(fa, [["additivity_error"], ["sector_additivity","error"]], default=None)
        kappa = _first_present(fa, [["kappa_proxy"], ["pauli_pressure","kappa_proxy"]], default=None)

    return {
        "source_file": source_file,
        "seed": _to_int(seed),
        "N": _to_int(N),
        "model": model,
        "scramble": scramble,
        "strobe_objective": strobe_obj,
        "strobe_edges": strobe_edges,
        "chains": _to_int(chains),
        "cores": _to_int(cores),
        "cycles": _to_int(cycles),
        "flow_steps": _to_int(flow_steps),
        "dt": _to_float(dt),
        "p": _to_int(p),
        "max_weight": _to_int(max_weight),
        "sparse_reduction": _to_float(sparse_red),
        "signal_entropy_reduction": _to_float(signal_red),
        "topN_share_final": _to_float(top_share),
        "V2_ring_reduction": _to_float(V2_red),
        "V2_over_V1_post": _to_float(V2_over_V1_post),
        "sparse_ok": bool(sparse_ok) if sparse_ok is not None else None,
        "signal_ok": bool(signal_ok) if signal_ok is not None else None,
        "jw_max": _to_float(jw_max),
        "additivity_error": _to_float(add_err),
        "kappa_proxy": _to_float(kappa),
    }

def _condition_key(row: Dict[str, Any]) -> Tuple:
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

def package_jsonl(inputs: List[Path], outdir: Path) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "inputs").mkdir(parents=True, exist_ok=True)

    all_flat: List[Dict[str, Any]] = []
    input_meta: List[Dict[str, Any]] = []
    for f in inputs:
        rows = _read_jsonl(f)
        input_meta.append({"path": str(f), "rows": len(rows), "bytes": f.stat().st_size})
        # copy inputs for archival
        shutil.copy2(f, outdir / "inputs" / f.name)
        for r in rows:
            all_flat.append(_extract_flat_row(r, source_file=f.name))

    # write CSV
    combined_csv = outdir / "combined_flat.csv"
    if all_flat:
        with combined_csv.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(all_flat[0].keys()))
            w.writeheader()
            for r in all_flat:
                w.writerow(r)
    else:
        combined_csv.write_text("", encoding="utf-8")

    # condition summaries
    buckets: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in all_flat:
        buckets.setdefault(_condition_key(r), []).append(r)

    cond_summary: Dict[str, Any] = {}
    for k, rows in buckets.items():
        key_str = "|".join([str(x) for x in k])
        sparse_ok = [1.0 for r in rows if r.get("sparse_ok") is True]
        signal_ok = [1.0 for r in rows if r.get("signal_ok") is True]
        cond_summary[key_str] = {
            "condition": {
                "N": k[0], "model": k[1], "scramble": k[2], "strobe_objective": k[3],
                "strobe_edges": k[4], "flow_steps": k[5], "max_weight": k[6], "chains": k[7],
            },
            "runs": len(rows),
            "success_rate_sparse": float(len(sparse_ok) / max(1, len(rows))),
            "success_rate_signal": float(len(signal_ok) / max(1, len(rows))),
            "metrics": {
                "sparse_reduction": _stats([r.get("sparse_reduction", float("nan")) for r in rows]),
                "signal_entropy_reduction": _stats([r.get("signal_entropy_reduction", float("nan")) for r in rows]),
                "topN_share_final": _stats([r.get("topN_share_final", float("nan")) for r in rows]),
                "V2_ring_reduction": _stats([r.get("V2_ring_reduction", float("nan")) for r in rows]),
                "V2_over_V1_post": _stats([r.get("V2_over_V1_post", float("nan")) for r in rows]),
                "jw_max": _stats([r.get("jw_max", float("nan")) for r in rows]),
                "additivity_error": _stats([r.get("additivity_error", float("nan")) for r in rows]),
                "kappa_proxy": _stats([r.get("kappa_proxy", float("nan")) for r in rows]),
            }
        }

    (outdir / "summary_by_condition.json").write_text(json.dumps(cond_summary, indent=2), encoding="utf-8")

    # report
    report = outdir / "REPORT.md"
    lines = ["# HSF Suite Report", f"- Created: `{_now_utc_iso()}`", f"- JSONL files packaged: {len(inputs)}", ""]
    lines.append("## Conditions\n")
    for _, block in cond_summary.items():
        c = block["condition"]
        lines.append(f"### N={c['N']} model={c['model']} scramble={c['scramble']} objective={c['strobe_objective']} edges={c['strobe_edges']} flow_steps={c['flow_steps']} chains={c['chains']}")
        lines.append(f"- runs: {block['runs']}")
        lines.append(f"- success_rate_sparse: {block['success_rate_sparse']:.3f}")
        lines.append(f"- success_rate_signal: {block['success_rate_signal']:.3f}")
        m = block["metrics"]
        lines.append(f"- sparse_reduction median: {m['sparse_reduction']['median']:.3f}")
        lines.append(f"- signal_entropy_reduction median: {m['signal_entropy_reduction']['median']:.3f}")
        lines.append(f"- V2_over_V1_post median: {m['V2_over_V1_post']['median']:.4f}")
        lines.append(f"- jw_max median: {m['jw_max']['median']:.4g}")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "created_utc": _now_utc_iso(),
        "tool": "hsf_run_suite.py",
        "inputs": input_meta,
        "outputs": {
            "combined_flat_csv": "combined_flat.csv",
            "summary_by_condition_json": "summary_by_condition.json",
            "report_md": "REPORT.md",
            "inputs_copied_to": "inputs/",
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ----------------------------
# Running experiments
# ----------------------------

@dataclass
class RunnerPaths:
    single_chain: Path
    multichain: Path

def run_cmd(cmd: List[str], cwd: Path) -> int:
    print("\n>>> " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd))
    return int(p.returncode)

def supports_flag(script: Path, flag: str) -> bool:
    try:
        txt = script.read_text(encoding="utf-8", errors="ignore")
        return flag in txt
    except Exception:
        return False

def resolve_default_paths(repo: Path) -> RunnerPaths:
    # You can change these defaults to match where you placed the scripts.
    single = repo / "experiments" / "scramble_recover_test_patched_signal_cache.py"
    if not single.exists():
        # fallback: consolidation/scripts
        single = repo / "consolidation" / "scripts" / "scramble_recover_test_patched_signal_cache.py"

    multi = repo / "experiments" / "scramble_recover_numba.py"
    if not multi.exists():
        multi = repo / "consolidation" / "scripts" / "scramble_recover_numba.py"

    return RunnerPaths(single_chain=single, multichain=multi)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to hilbert_substrate repo root")
    ap.add_argument("--out", required=True, help="Output folder (relative to repo or absolute)")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use (default: current)")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--model", default="xx", choices=["xx","xxz","xxx"])
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=8000)
    ap.add_argument("--flow-steps", type=int, default=30)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--max-weight", type=int, default=4)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--chains", type=int, default=12, help="For multichain runner")
    ap.add_argument("--cores", type=int, default=12, help="For multichain runner")
    ap.add_argument("--zip", action="store_true", help="Zip the whole suite output folder at end")
    ap.add_argument("--skip-multichain", action="store_true")
    ap.add_argument("--skip-singlechain", action="store_true")
    ap.add_argument("--fermion-audit", action="store_true", help="Ask runner to emit fermion audits when supported")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.exists():
        print(f"Repo not found: {repo}")
        return 2

    outdir = (Path(args.out) if Path(args.out).is_absolute() else (repo / args.out)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_default_paths(repo)
    if not args.skip_singlechain and not paths.single_chain.exists():
        print(f"Single-chain runner not found: {paths.single_chain}")
        print("Pass --skip-singlechain or place it at experiments/scramble_recover_test_patched_signal_cache.py")
        return 2
    if not args.skip_multichain and not paths.multichain.exists():
        print(f"Multichain runner not found: {paths.multichain}")
        print("Pass --skip-multichain or place it at experiments/scramble_recover_numba.py")
        return 2

    produced_jsonl: List[Path] = []

    # ----------------------------
    # Paper II endpoints
    # ----------------------------
    if not args.skip_singlechain:
        # LOCAL endpoint (expect recoverable)
        local_jsonl = runs_dir / f"paper2_endpoint_local_N{args.N}_{args.model}.jsonl"
        cmd = [
            args.python, str(paths.single_chain),
            "--N", str(args.N),
            "--model", str(args.model),
            "--scramble", "local",
            "--recover", "both",
            "--strobe-objective", "sparse",
            "--cycles", str(args.cycles),
            "--flow-steps", str(args.flow_steps),
            "--dt", str(args.dt),
            "--p", str(args.p),
            "--max-weight", str(args.max_weight),
            "--seed-start", str(args.seed_start),
            "--seed-count", str(args.seeds),
            "--jobs", "1",
            "--blas-threads", str(args.blas_threads),
            "--progress",
            "--partial-output", str(local_jsonl),
        ]
        if args.fermion_audit and supports_flag(paths.single_chain, "--fermion-audit"):
            cmd.append("--fermion-audit")
        rc = run_cmd(cmd, cwd=repo)
        if rc == 0 and local_jsonl.exists():
            produced_jsonl.append(local_jsonl)

        # GLOBAL endpoint (expect not recoverable)
        global_jsonl = runs_dir / f"paper2_endpoint_global_N{args.N}_{args.model}.jsonl"
        cmd = [
            args.python, str(paths.single_chain),
            "--N", str(args.N),
            "--model", str(args.model),
            "--scramble", "global",
            "--recover", "both",
            "--strobe-objective", "sparse",
            "--cycles", str(args.cycles),
            "--flow-steps", str(args.flow_steps),
            "--dt", str(args.dt),
            "--p", str(args.p),
            "--max-weight", str(args.max_weight),
            "--seed-start", str(args.seed_start),
            "--seed-count", str(args.seeds),
            "--jobs", "1",
            "--blas-threads", str(args.blas_threads),
            "--progress",
            "--partial-output", str(global_jsonl),
        ]
        if args.fermion_audit and supports_flag(paths.single_chain, "--fermion-audit"):
            cmd.append("--fermion-audit")
        rc = run_cmd(cmd, cwd=repo)
        if rc == 0 and global_jsonl.exists():
            produced_jsonl.append(global_jsonl)

    # ----------------------------
    # Multichain stress test (Paper II "barrier hardness")
    # ----------------------------
    if not args.skip_multichain:
        # This script's interface is your own; we keep args minimal and robust.
        # If your multichain file expects different flags, edit cmd below once.
        multi_jsonl = runs_dir / f"paper2_multichain_global_N{args.N}_{args.model}.jsonl"
        # Try to call it in a compatible way (common in your output snippet).
        cmd = [
            args.python, str(paths.multichain),
            "--N", str(args.N),
            "--model", str(args.model),
            "--scramble", "global",
            "--chains", str(args.chains),
            "--cores", str(args.cores),
            "--cycles", str(args.cycles),
            "--strobe-objective", "sparse",
            "--seed-start", str(args.seed_start),
            "--seed-count", str(args.seeds),
            "--progress",
            "--partial-output", str(multi_jsonl),
        ]
        # If your multichain script does not have these flags, it will error;
        # we catch and print, but the suite keeps going.
        rc = run_cmd(cmd, cwd=repo)
        if rc == 0 and multi_jsonl.exists():
            produced_jsonl.append(multi_jsonl)

    # ----------------------------
    # Package everything produced
    # ----------------------------
    if not produced_jsonl:
        print("\nNo JSONL outputs were produced. (Runner flags may not match your current scripts.)")
        print("Open hsf_run_suite.py and adjust the multichain cmd block to your script's CLI.")
        return 1

    print("\nPackaging JSONL outputs into:", outdir)
    manifest = package_jsonl(produced_jsonl, outdir)

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

    print("\nDONE.")
    print("Report:", outdir / "REPORT.md")
    print("Summary:", outdir / "summary_by_condition.json")
    print("Flat CSV:", outdir / "combined_flat.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
