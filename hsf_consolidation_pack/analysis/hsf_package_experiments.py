#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSF Experiment Packager
=======================

Packages multiple HSF experiment outputs (JSONL) into a single consolidated bundle:
- combined flat CSV
- per-condition summary JSON
- quick-look Markdown report
- manifest (inputs, timestamps, schema hints)
- optional ZIP bundle

Windows-friendly usage examples (single-line):

  python hsf_package_experiments.py --in "outputs\*.jsonl" --outdir "outputs\PACK_2026-01-08" --zip

  python hsf_package_experiments.py --in "outputs\sweep_N8_xx_signal.jsonl" --in "outputs\multichain_sparse_N8_xx.jsonl" --outdir "outputs\PACK_latest" --zip

Notes:
- The tool is schema-tolerant: it tries multiple key paths for each metric.
- It does NOT assume your exact JSON layout (single-chain vs multichain scripts differ).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


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


def extract_flat_row(r: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    seed = first_present(r, [["seed"], ["run", "seed"]], default=None)
    N = first_present(r, [["N"], ["run", "N"], ["meta", "N"]], default=None)
    model = first_present(r, [["model"], ["run", "model"], ["meta", "model"]], default=None)
    scramble = first_present(r, [["scramble"], ["run", "scramble"], ["meta", "scramble"]], default=None)

    strobe_obj = first_present(r, [["strobe", "objective"], ["strobe_objective"], ["config", "strobe_objective"]], default=None)
    strobe_edges = first_present(r, [["strobe", "edges"], ["strobe_edges"], ["config", "strobe_edges"]], default=None)
    chains = first_present(r, [["strobe", "chains"], ["chains"], ["config", "chains"]], default=None)
    cores = first_present(r, [["strobe", "cores"], ["cores"], ["config", "cores"]], default=None)
    cycles = first_present(r, [["strobe", "cycles"], ["cycles"], ["config", "cycles"]], default=None)

    flow_steps = first_present(r, [["flow", "steps"], ["flow_steps"], ["config", "flow_steps"]], default=None)
    dt = first_present(r, [["flow", "dt"], ["dt"], ["config", "dt"]], default=None)
    p = first_present(r, [["flow", "p"], ["p"], ["config", "p"]], default=None)
    max_weight = first_present(r, [["flow", "max_weight"], ["max_weight"], ["config", "max_weight"]], default=None)

    sparse_red = first_present(r, [["metrics", "sparse_reduction"], ["sparse_reduction"]], default=None)
    signal_red = first_present(r, [["metrics", "signal_entropy_reduction"], ["signal_entropy_reduction"]], default=None)
    top_share = first_present(r, [["metrics", "topN_share_final"], ["metrics", "topN_share"], ["topN_share_final"]], default=None)
    V2_red = first_present(r, [["metrics", "V2_ring_reduction"], ["V2_ring_reduction"]], default=None)

    V2_over_V1_pre = first_present(r, [["metrics", "V2_over_V1_initial"], ["metrics", "V2_over_V1_pre"], ["V2_over_V1_pre"]], default=None)
    V2_over_V1_post = first_present(r, [["metrics", "V2_over_V1_final"], ["metrics", "V2_over_V1_post"], ["V2_over_V1_post"]], default=None)

    sparse_ok = first_present(r, [["success", "sparse_ok"], ["locality_recovered_sparse"], ["success_sparse"]], default=None)
    signal_ok = first_present(r, [["success", "signal_ok"], ["locality_recovered_signal"], ["success_signal"]], default=None)

    fa = first_present(r, [["fermion_audit_results"], ["fermion_audit"]], default={})
    jw_max = None
    add_err = None
    kappa = None
    if isinstance(fa, dict):
        jw_max = first_present(fa, [["jw_max"], ["jw_anticommutators", "max_abs"]], default=None)
        add_err = first_present(fa, [["additivity_error"], ["sector_additivity", "error"]], default=None)
        kappa = first_present(fa, [["kappa_proxy"], ["pauli_pressure", "kappa_proxy"]], default=None)

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
        "V2_over_V1_pre": to_float(V2_over_V1_pre),
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


def collect_inputs(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            files.extend([Path(m) for m in matches])
        else:
            p = Path(pat)
            if p.exists():
                files.append(p)

    uniq: List[Path] = []
    seen = set()
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_by_condition(flat_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in flat_rows:
        buckets.setdefault(condition_key(r), []).append(r)

    out: Dict[str, Any] = {}
    for k, rows in buckets.items():
        key_str = "|".join([str(x) for x in k])
        sparse_ok = [1.0 for r in rows if r.get("sparse_ok") is True]
        signal_ok = [1.0 for r in rows if r.get("signal_ok") is True]
        out[key_str] = {
            "condition": {
                "N": k[0],
                "model": k[1],
                "scramble": k[2],
                "strobe_objective": k[3],
                "strobe_edges": k[4],
                "flow_steps": k[5],
                "max_weight": k[6],
                "chains": k[7],
            },
            "runs": len(rows),
            "success_rate_sparse": float(len(sparse_ok) / max(1, len(rows))),
            "success_rate_signal": float(len(signal_ok) / max(1, len(rows))),
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
    return out


def write_markdown_report(path: Path, manifest: Dict[str, Any], cond_summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# HSF Packaged Experiment Report\n")
    lines.append(f"- Created: `{manifest['created_utc']}`")
    lines.append(f"- Input files: {len(manifest['inputs'])}\n")

    lines.append("## Conditions\n")
    for _, block in cond_summary.items():
        c = block["condition"]
        lines.append(
            f"### N={c['N']} model={c['model']} scramble={c['scramble']} "
            f"objective={c['strobe_objective']} edges={c['strobe_edges']} "
            f"flow_steps={c['flow_steps']} chains={c['chains']}"
        )
        lines.append(f"- runs: {block['runs']}")
        lines.append(f"- success_rate_sparse: {block['success_rate_sparse']:.3f}")
        lines.append(f"- success_rate_signal: {block['success_rate_signal']:.3f}")
        m = block["metrics"]
        lines.append(f"- sparse_reduction median: {m['sparse_reduction']['median']:.3f}")
        lines.append(f"- signal_entropy_reduction median: {m['signal_entropy_reduction']['median']:.3f}")
        lines.append(f"- V2_over_V1_post median: {m['V2_over_V1_post']['median']:.4f}")
        lines.append(f"- jw_max median: {m['jw_max']['median']:.4g}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_zip(zip_path: Path, folder: Path) -> None:
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(folder)
                z.write(full, arcname=str(rel))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="Input JSONL file path OR glob pattern. Repeatable.")
    ap.add_argument("--outdir", required=True, help="Output directory (created if missing).")
    ap.add_argument("--zip", action="store_true", help="Also produce a ZIP bundle of outdir.")
    ap.add_argument("--copy-inputs", action="store_true", help="Copy input JSONL files into outdir/inputs/")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "inputs").mkdir(parents=True, exist_ok=True)

    files = collect_inputs(args.inputs)
    if not files:
        print("No input files matched.")
        return 2

    all_flat: List[Dict[str, Any]] = []
    input_meta: List[Dict[str, Any]] = []

    for f in files:
        rows = read_jsonl(f)
        input_meta.append({
            "path": str(f),
            "rows": len(rows),
            "bytes": f.stat().st_size,
        })
        if args.copy_inputs:
            dst = outdir / "inputs" / f.name
            if dst.resolve() != f.resolve():
                dst.write_bytes(f.read_bytes())

        for r in rows:
            all_flat.append(extract_flat_row(r, source_file=f.name))

    combined_csv = outdir / "combined_flat.csv"
    write_csv(combined_csv, all_flat)

    cond_summary = summarize_by_condition(all_flat)
    summary_json = outdir / "summary_by_condition.json"
    summary_json.write_text(json.dumps(cond_summary, indent=2), encoding="utf-8")

    manifest = {
        "created_utc": now_utc_iso(),
        "tool": "hsf_package_experiments.py",
        "inputs": input_meta,
        "outputs": {
            "combined_flat_csv": "combined_flat.csv",
            "summary_by_condition_json": "summary_by_condition.json",
            "report_md": "REPORT.md",
        },
        "schema_notes": "Schema-tolerant. Adjust extract_flat_row() if you add new metrics.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_md = outdir / "REPORT.md"
    write_markdown_report(report_md, manifest, cond_summary)

    if args.zip:
        zip_file = Path(str(outdir) + ".zip")
        make_zip(zip_file, outdir)
        print(f"Wrote ZIP: {zip_file}")

    print(f"Wrote: {combined_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {report_md}")
    print(f"Wrote: {outdir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
