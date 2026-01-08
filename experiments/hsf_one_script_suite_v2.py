\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
HSF ONE SCRIPT SUITE (v2 - safer)
=================================

Why you saw "No JSONL produced":
- The suite DID run, but the underlying runner likely errored before writing JSONL
  (common causes: passing an unsupported --strobe-objective value like "signal",
   or the runner not handling --help / failing imports, so the suite couldn't
   detect supported flags).

v2 changes:
- If runner --help fails (empty or nonzero), we enter SAFE MODE:
  - do NOT pass --strobe-objective (unless user forces --force-objective)
  - do NOT pass --fermion-audit
  - do NOT pass --recover
  This avoids "invalid choice" failures when help couldn't be parsed.
- Always writes per-run logs under out/runs/*.log.txt.
- Adds --runner REQUIRED if auto-discovery can't find a runner, but still prints
  what it tried in suite_status.json.

You can run v2 immediately with your known runner path:
  python hsf_one_script_suite_v2.py --repo "C:\GitHub\hilbert_substrate" --out "outputs\SUITE_latest" --runner "C:\Lab\scramble_recover_test_patched_signal_cache_gpu.py" --N 8 --model xx --seeds 32 --recover flow --objective-list sparse,signal --fermion-audit --zip --progress
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def run_capture(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="ignore")
    return int(p.returncode), (p.stdout or "")

def run_logged(cmd: List[str], cwd: Path, log_path: Path) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(">>> " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        return int(p.returncode)

def flag_in_help(help_text: str, flag: str) -> bool:
    return flag in help_text

def find_default_runner(repo: Path) -> Optional[Path]:
    candidates = [
        repo / "experiments" / "scramble_recover_test_patched_signal_cache_gpu.py",
        repo / "experiments" / "scramble_recover_test_patched_signal_cache.py",
        repo / "experiments" / "scramble_recover_test_patched_signal_cache_gpu.py".replace("/", "\\"),
        repo / "experiments" / "scramble_recover_test_patched_signal_cache.py".replace("/", "\\"),
        repo / "consolidation" / "scripts" / "scramble_recover_test_patched_signal_cache_gpu.py",
        repo / "consolidation" / "scripts" / "scramble_recover_test_patched_signal_cache.py",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


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
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(np.mean(a)), "min": float(np.min(a)), "max": float(np.max(a))}

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

def package_all(jsonl_files: List[Path], outdir: Path) -> None:
    ensure_dir(outdir)
    ensure_dir(outdir / "inputs")

    input_meta: List[Dict[str, Any]] = []
    flat: List[Dict[str, Any]] = []
    for f in jsonl_files:
        rows = read_jsonl(f)
        input_meta.append({"path": str(f), "rows": len(rows), "bytes": f.stat().st_size})
        shutil.copy2(f, outdir / "inputs" / f.name)
        for r in rows:
            flat.append(extract_flat_row(r, source_file=f.name))

    csv_path = outdir / "combined_flat.csv"
    if flat:
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(flat[0].keys()))
            w.writeheader()
            for r in flat:
                w.writerow(r)
    else:
        write_text(csv_path, "")

    buckets: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in flat:
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
            }
        }
    write_text(outdir / "summary_by_condition.json", json.dumps(summary, indent=2))

    lines = ["# HSF Suite Report", f"- Created: `{now_utc_iso()}`", f"- JSONL files packaged: {len(jsonl_files)}", ""]
    lines.append("## Conditions\n")
    for _, block in summary.items():
        c = block["condition"]; m = block["metrics"]
        lines.append(f"### N={c['N']} model={c['model']} scramble={c['scramble']} objective={c['strobe_objective']}")
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
        "tool": "hsf_one_script_suite_v2.py",
        "inputs": input_meta,
        "outputs": ["combined_flat.csv", "summary_by_condition.json", "REPORT.md", "manifest.json", "suite_status.json", "inputs/"],
    }
    write_text(outdir / "manifest.json", json.dumps(manifest, indent=2))


# ----------------------------
# Suite runner
# ----------------------------

@dataclass
class SuiteRun:
    name: str
    status: str
    jsonl: Optional[str] = None
    log: Optional[str] = None
    reason: Optional[str] = None
    returncode: Optional[int] = None
    cmd: Optional[List[str]] = None

def maybe_zip_folder(folder: Path) -> Path:
    import zipfile
    zip_path = Path(str(folder) + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(folder)
                z.write(full, arcname=str(rel))
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runner", default="")
    ap.add_argument("--python", default=sys.executable)

    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--model", default="xx", choices=["xx", "xxz", "xxx"])
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed-start", type=int, default=0)

    ap.add_argument("--recover", default="flow")
    ap.add_argument("--objective-list", default="sparse")
    ap.add_argument("--force-objective", action="store_true", help="Pass strobe-objective even if runner --help failed.")
    ap.add_argument("--fermion-audit", action="store_true")

    ap.add_argument("--cycles", type=int, default=8000)
    ap.add_argument("--flow-steps", type=int, default=30)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--max-weight", type=int, default=4)

    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    outdir = (Path(args.out) if Path(args.out).is_absolute() else (repo / args.out)).resolve()
    runs_dir = outdir / "runs"
    ensure_dir(runs_dir)

    # Resolve runner
    if args.runner.strip():
        runner = Path(args.runner).expanduser()
        if not runner.is_absolute():
            runner = (repo / runner).resolve()
    else:
        runner = find_default_runner(repo) or Path("")

    if not runner or not Path(runner).exists():
        status = {
            "created_utc": now_utc_iso(),
            "repo": str(repo),
            "outdir": str(outdir),
            "runner": str(runner) if runner else None,
            "error": "Runner not found. Pass --runner.",
        }
        write_text(outdir / "suite_status.json", json.dumps(status, indent=2))
        print("Runner not found. Pass --runner \"C:\\path\\to\\your\\runner.py\"")
        return 2

    # Try help
    rc_help, help_text = run_capture([args.python, str(runner), "--help"], cwd=repo)
    safe_mode = (rc_help != 0) or (not help_text.strip())

    objectives_raw = [x.strip() for x in args.objective_list.split(",") if x.strip()]
    objectives = objectives_raw[:] if objectives_raw else ["sparse"]

    suite: List[SuiteRun] = []
    produced: List[Path] = []

    for obj in objectives:
        for scramble in ("local", "global"):
            name = f"endpoint_{scramble.upper()}_N{args.N}_{args.model}_{obj}"
            out_jsonl = runs_dir / f"{name}.jsonl"
            log_path = runs_dir / f"{name}.log.txt"

            cmd = [args.python, str(runner)]

            def add(flag: str, value: Optional[str] = None):
                if safe_mode:
                    # in safe mode, only pass the essentials, never risky choice flags
                    if flag not in ("--N","--model","--scramble","--seed-start","--seed-count","--jobs","--blas-threads","--cycles","--flow-steps","--dt","--p","--max-weight","--partial-output","--output","--progress"):
                        return
                else:
                    if not flag_in_help(help_text, flag):
                        return
                cmd.append(flag)
                if value is not None:
                    cmd.append(value)

            add("--N", str(args.N))
            add("--model", args.model)
            add("--scramble", scramble)

            if not safe_mode and flag_in_help(help_text, "--recover"):
                add("--recover", args.recover)

            # objective: only if not safe_mode OR user forces it
            if (not safe_mode and flag_in_help(help_text, "--strobe-objective")) or args.force_objective:
                add("--strobe-objective", obj)

            add("--cycles", str(args.cycles))
            add("--flow-steps", str(args.flow_steps))
            add("--dt", str(args.dt))
            add("--p", str(args.p))
            add("--max-weight", str(args.max_weight))

            add("--seed-start", str(args.seed_start))
            add("--seed-count", str(args.seeds))
            add("--jobs", str(args.jobs))
            add("--blas-threads", str(args.blas_threads))
            if args.progress:
                add("--progress")

            # output
            if (not safe_mode and flag_in_help(help_text, "--partial-output")) or safe_mode:
                cmd += ["--partial-output", str(out_jsonl)]
            elif flag_in_help(help_text, "--output"):
                cmd += ["--output", str(out_jsonl)]
            else:
                suite.append(SuiteRun(name=name, status="skipped", reason="runner lacks --output/--partial-output", log=str(log_path), cmd=cmd))
                continue

            # fermion audit only when help says so (never in safe_mode)
            if args.fermion_audit and (not safe_mode) and flag_in_help(help_text, "--fermion-audit"):
                cmd.append("--fermion-audit")

            rc = run_logged(cmd, cwd=repo, log_path=log_path)
            if rc == 0 and out_jsonl.exists() and out_jsonl.stat().st_size > 0:
                suite.append(SuiteRun(name=name, status="ok", jsonl=str(out_jsonl), log=str(log_path), returncode=rc, cmd=cmd))
                produced.append(out_jsonl)
            else:
                suite.append(SuiteRun(name=name, status="failed", jsonl=str(out_jsonl), log=str(log_path), returncode=rc, reason="nonzero return or missing/empty output", cmd=cmd))

    status = {
        "created_utc": now_utc_iso(),
        "repo": str(repo),
        "outdir": str(outdir),
        "runner": str(runner),
        "python": args.python,
        "help_returncode": rc_help,
        "safe_mode": safe_mode,
        "runs": [r.__dict__ for r in suite],
    }
    write_text(outdir / "suite_status.json", json.dumps(status, indent=2))

    if not produced:
        write_text(outdir / "REPORT.md",
                   "# HSF Suite Report (NO OUTPUT PRODUCED)\n\n"
                   f"- Created: `{now_utc_iso()}`\n"
                   f"- Runner: `{runner}`\n"
                   f"- Safe mode: `{safe_mode}`\n\n"
                   "No JSONL files were produced.\n"
                   "Check:\n"
                   "- `suite_status.json`\n"
                   "- `runs/*.log.txt` (runner output)\n")
        print("No JSONL produced. Open:", outdir / "suite_status.json")
        return 1

    package_all(produced, outdir)

    if args.zip:
        z = maybe_zip_folder(outdir)
        print("Wrote ZIP:", z)

    print("DONE. Open:", outdir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
