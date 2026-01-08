\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
HSF ONE SCRIPT SUITE
===================

This is the single script you asked for:

- Runs the HSF experiments (Paper II endpoints + optional fermion audit)
- Writes ALL artifacts into ONE output folder automatically
- Packages everything into:
    combined_flat.csv
    summary_by_condition.json
    REPORT.md
    manifest.json
    suite_status.json
- Optional ZIP of the whole output folder

Key properties:
- No glob/filename games required.
- Works no matter what directory you run it from.
- Robust to runner CLI drift: we query runner --help and only pass supported flags.
- Runner path is auto-discovered from your repo, BUT you can override with --runner.

What it runs (minimum, always):
  1) LOCAL scramble  (recover = flow/both depending on runner support)
  2) GLOBAL scramble (recover = flow/both depending on runner support)

Optional:
  - If your runner supports --fermion-audit and you pass --fermion-audit, it will emit fermion metrics.
  - If your runner supports --strobe-objective choices and you pass --objective-list, it will run each objective.

USAGE (Windows, single line)
----------------------------
Run from anywhere:

  python hsf_one_script_suite.py --repo "C:\GitHub\hilbert_substrate" --out "outputs\SUITE_latest" --N 8 --model xx --seeds 32 --runner "C:\Lab\scramble_recover_test_patched_signal_cache_gpu.py" --recover flow --objective-list sparse,signal --fermion-audit --zip

If your runner is already inside the repo at experiments/, you can omit --runner:

  python hsf_one_script_suite.py --repo "C:\GitHub\hilbert_substrate" --out "outputs\SUITE_latest" --N 8 --model xx --seeds 32 --recover flow --objective-list sparse,signal --zip

Open:
  <repo>\outputs\SUITE_latest\REPORT.md

NOTE ABOUT YOUR PAST FAILURES
-----------------------------
- You ran tools from repo\experiments, so relative patterns resolved under experiments\.
- This suite resolves EVERYTHING relative to --repo (repo root). No surprises.

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

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def run_capture(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="ignore")
        return int(p.returncode), (p.stdout or "")
    except Exception as e:
        return 1, f"EXCEPTION running {cmd}: {e}"

def run_logged(cmd: List[str], cwd: Path, log_path: Path) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(">>> " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        return int(p.returncode)

def flag_in_help(help_text: str, flag: str) -> bool:
    return flag in help_text

def select_recover(help_text: str, preferred: str) -> Optional[str]:
    """
    Choose a recover mode that the runner supports.
    preferred in {"flow","strobe","both","none"} typically.
    """
    if not flag_in_help(help_text, "--recover"):
        return None

    # If preferred isn't in help, try fallbacks.
    order = [preferred, "flow", "both", "strobe", "none"]
    for r in order:
        # crude: if "choices" list exists it will contain the token.
        if r in help_text:
            return r
    # still pass preferred if --recover exists; runner may accept it even if help format differs
    return preferred

def parse_objectives(help_text: str, objective_list: List[str]) -> List[str]:
    if not flag_in_help(help_text, "--strobe-objective"):
        return ["(none)"]  # marker meaning "don't pass"
    # Filter to those that appear in help text; if none match, keep first.
    keep = [o for o in objective_list if o in help_text]
    return keep if keep else [objective_list[0]]

def find_default_runner(repo: Path) -> Optional[Path]:
    candidates = [
        repo / "experiments" / "scramble_recover_test_patched_signal_cache_gpu.py",
        repo / "experiments" / "scramble_recover_test_patched_signal_cache.py",
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

    # Copy JSONL inputs and flatten rows
    input_meta: List[Dict[str, Any]] = []
    flat: List[Dict[str, Any]] = []
    for f in jsonl_files:
        rows = read_jsonl(f)
        input_meta.append({"path": str(f), "rows": len(rows), "bytes": f.stat().st_size})
        shutil.copy2(f, outdir / "inputs" / f.name)
        for r in rows:
            flat.append(extract_flat_row(r, source_file=f.name))

    # combined_flat.csv
    csv_path = outdir / "combined_flat.csv"
    if flat:
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(flat[0].keys()))
            w.writeheader()
            for r in flat:
                w.writerow(r)
    else:
        write_text(csv_path, "")

    # summary_by_condition.json
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

    # REPORT.md
    lines = ["# HSF Suite Report", f"- Created: `{now_utc_iso()}`", f"- JSONL files packaged: {len(jsonl_files)}", ""]
    lines.append("## Conditions\n")
    for _, block in summary.items():
        c = block["condition"]; m = block["metrics"]
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

    # manifest.json
    manifest = {
        "created_utc": now_utc_iso(),
        "tool": "hsf_one_script_suite.py",
        "inputs": input_meta,
        "outputs": ["combined_flat.csv", "summary_by_condition.json", "REPORT.md", "manifest.json", "suite_status.json", "inputs/"],
    }
    write_text(outdir / "manifest.json", json.dumps(manifest, indent=2))


# ----------------------------
# Suite definition
# ----------------------------

@dataclass
class SuiteRun:
    name: str
    status: str  # ok / failed / skipped
    jsonl: Optional[str] = None
    log: Optional[str] = None
    reason: Optional[str] = None
    returncode: Optional[int] = None
    cmd: Optional[List[str]] = None

def build_cmd(py: str, runner: Path, help_text: str, *,
              N: int, model: str, scramble: str, recover: str,
              objective: Optional[str],
              cycles: int, flow_steps: int, dt: float, p: int, max_weight: int,
              seed_start: int, seed_count: int,
              jobs: int, blas_threads: int,
              fermion_audit: bool,
              out_jsonl: Path,
              progress: bool) -> Tuple[List[str], Optional[str]]:
    cmd = [py, str(runner)]

    def add(flag: str, value: Optional[str] = None):
        if not flag_in_help(help_text, flag):
            return
        cmd.append(flag)
        if value is not None:
            cmd.append(value)

    add("--N", str(N))
    add("--model", model)
    add("--scramble", scramble)

    chosen_recover = select_recover(help_text, recover)
    if chosen_recover is not None:
        add("--recover", chosen_recover)

    if objective is not None and objective != "(none)":
        add("--strobe-objective", objective)

    add("--cycles", str(cycles))
    add("--flow-steps", str(flow_steps))
    add("--dt", str(dt))
    add("--p", str(p))
    add("--max-weight", str(max_weight))

    add("--seed-start", str(seed_start))
    add("--seed-count", str(seed_count))

    add("--jobs", str(jobs))
    add("--blas-threads", str(blas_threads))

    if progress and flag_in_help(help_text, "--progress"):
        cmd.append("--progress")

    # output selection
    if flag_in_help(help_text, "--partial-output"):
        cmd += ["--partial-output", str(out_jsonl)]
    elif flag_in_help(help_text, "--output"):
        cmd += ["--output", str(out_jsonl)]
    else:
        return cmd, "Runner has no --output/--partial-output."

    if fermion_audit and flag_in_help(help_text, "--fermion-audit"):
        cmd.append("--fermion-audit")

    return cmd, None

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
    ap.add_argument("--repo", required=True, help="Repo root (hilbert_substrate). Everything resolves relative to this.")
    ap.add_argument("--out", required=True, help="Output folder (relative to --repo if relative).")
    ap.add_argument("--runner", default="", help="Path to the runner script. If omitted, auto-discover in repo/experiments.")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use.")

    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--model", default="xx", choices=["xx", "xxz", "xxx"])
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed-start", type=int, default=0)

    ap.add_argument("--recover", default="flow", help="Preferred recover mode (flow/both/strobe/none).")
    ap.add_argument("--objective-list", default="sparse", help="Comma list of strobe objectives to run (e.g. sparse,signal).")
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
    if not repo.exists():
        print("Repo not found:", repo)
        return 2

    outdir = (Path(args.out) if Path(args.out).is_absolute() else (repo / args.out)).resolve()
    runs_dir = outdir / "runs"
    ensure_dir(runs_dir)

    # Resolve runner
    runner: Optional[Path]
    if args.runner.strip():
        runner = Path(args.runner).expanduser()
        if not runner.is_absolute():
            runner = (repo / runner).resolve()
        if not runner.exists():
            print("Runner not found at:", runner)
            return 2
    else:
        runner = find_default_runner(repo)
        if runner is None:
            print("Runner not found. Either:")
            print("  - place it at repo\\experiments\\scramble_recover_test_patched_signal_cache*.py")
            print("  - OR pass --runner \"C:\\path\\to\\scramble_recover_test_patched_signal_cache_gpu.py\"")
            return 2

    # Get runner help
    rc_help, help_text = run_capture([args.python, str(runner), "--help"], cwd=repo)
    if rc_help != 0 or not help_text.strip():
        # still proceed, but with fewer safety checks
        help_text = ""

    objectives_raw = [x.strip() for x in args.objective_list.split(",") if x.strip()]
    objectives = parse_objectives(help_text, objectives_raw) if help_text else objectives_raw

    suite: List[SuiteRun] = []
    produced: List[Path] = []

    # Define experiments: endpoints (LOCAL/GLOBAL) for each objective
    for obj in objectives:
        suffix = f"_{obj}" if obj not in ("(none)", None) else ""
        for scramble in ("local", "global"):
            name = f"endpoint_{scramble.upper()}_N{args.N}_{args.model}{suffix}"
            out_jsonl = runs_dir / f"{name}.jsonl"
            log_path = runs_dir / f"{name}.log.txt"

            cmd, err = build_cmd(
                args.python, runner, help_text,
                N=args.N, model=args.model, scramble=scramble, recover=args.recover,
                objective=(obj if obj != "(none)" else None),
                cycles=args.cycles, flow_steps=args.flow_steps, dt=args.dt, p=args.p, max_weight=args.max_weight,
                seed_start=args.seed_start, seed_count=args.seeds,
                jobs=args.jobs, blas_threads=args.blas_threads,
                fermion_audit=args.fermion_audit,
                out_jsonl=out_jsonl,
                progress=args.progress,
            )
            if err is not None:
                suite.append(SuiteRun(name=name, status="skipped", reason=err, cmd=cmd, log=str(log_path)))
                continue

            rc = run_logged(cmd, cwd=repo, log_path=log_path)
            if rc == 0 and out_jsonl.exists() and out_jsonl.stat().st_size > 0:
                suite.append(SuiteRun(name=name, status="ok", jsonl=str(out_jsonl), log=str(log_path), returncode=rc, cmd=cmd))
                produced.append(out_jsonl)
            else:
                suite.append(SuiteRun(name=name, status="failed", jsonl=str(out_jsonl), log=str(log_path), returncode=rc, reason="nonzero return or missing/empty output", cmd=cmd))

    # Write suite_status.json always
    status = {
        "created_utc": now_utc_iso(),
        "repo": str(repo),
        "outdir": str(outdir),
        "runner": str(runner),
        "python": args.python,
        "params": {
            "N": args.N, "model": args.model, "seeds": args.seeds, "seed_start": args.seed_start,
            "recover": args.recover, "objective_list": objectives_raw, "fermion_audit": bool(args.fermion_audit),
            "cycles": args.cycles, "flow_steps": args.flow_steps, "dt": args.dt, "p": args.p, "max_weight": args.max_weight,
            "jobs": args.jobs, "blas_threads": args.blas_threads,
        },
        "runs": [r.__dict__ for r in suite],
    }
    ensure_dir(outdir)
    write_text(outdir / "suite_status.json", json.dumps(status, indent=2))

    if not produced:
        # make a minimal report pointing to logs/status
        lines = [
            "# HSF Suite Report (NO OUTPUT PRODUCED)",
            f"- Created: `{now_utc_iso()}`",
            "",
            "No JSONL files were produced by the runner.",
            "Open `suite_status.json` and the per-run logs under `runs/` to see the exact command lines and outputs.",
            "",
            f"- runner: `{runner}`",
            f"- outdir: `{outdir}`",
        ]
        write_text(outdir / "REPORT.md", "\n".join(lines))
        print("No JSONL produced. Open:", outdir / "suite_status.json")
        return 1

    # Package everything into ONE output
    package_all(produced, outdir)

    # Zip if requested
    if args.zip:
        z = maybe_zip_folder(outdir)
        print("Wrote ZIP:", z)

    print("DONE. Open:", outdir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
