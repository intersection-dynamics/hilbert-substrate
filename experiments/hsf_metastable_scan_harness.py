# hsf_metastable_scan_harness.py
# ------------------------------------------------------------
# Harness to scan drive PHASE (phi) and drive FREQUENCY (omega) for the
# SAFE HSF toy model in:
#   hsf_metastable_branch_toy_v2.py
#
# What it does:
#   - runs the v2 toy script repeatedly (subprocess) for:
#       (A) a phase sweep: phi in [phi_min, phi_max]
#       (B) an omega sweep: w in [w_min, w_max]
#   - repeats each sweep point over multiple seeds for error bars
#   - reads each run's results.npz and computes:
#       delta = p_in - p_out (cumulative)
#       SE(delta) and z-score (cumulative)
#       end-state memory/rigidity proxies (mean_edge_mem_end, locked_like_end)
#   - writes:
#       scan_phase.csv, scan_omega.csv
#       plots/phase_scan.png, plots/omega_scan.png
#       plots/regime_memory_vs_delta.png
#
# Windows one-liner example:
#   python hsf_metastable_scan_harness.py --mode both --seeds 8 --phi_steps 25 --w_steps 25 --N 250 --p_edge 0.03 --T 20000 --lambda_capture 0.05 --drive_frac 0.2 --drive_amp 0.35 --drive_gain 2.2
#
# Notes:
#   - This is a SAFE parameter scan of a toy model. No nuclear specifics.
#   - It assumes python can run the v2 script and that it writes results.npz.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

RUN_DIR_RE = re.compile(r"Run output:\s*(.+)\s*$")
CLEAN_PLOT_RE = re.compile(r"Clean plot:\s*(.+)\s*$")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def linspace_inclusive(a: float, b: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([a], dtype=np.float64)
    return np.linspace(a, b, n, dtype=np.float64)


def compute_final_stats_from_npz(npz: Dict) -> Tuple[float, float, float, int, int]:
    """
    Final cumulative delta, standard error (difference of proportions), z-score.
    Derived from captures_in/out and M_in/out totals.
    """
    cap_in = int(np.sum(npz["captures_in"]))
    cap_out = int(np.sum(npz["captures_out"]))
    M_in = int(np.sum(npz["M_in"]))
    M_out = int(np.sum(npz["M_out"]))

    eps = 1e-18
    p_in = M_in / (cap_in + eps)
    p_out = M_out / (cap_out + eps)
    delta = p_in - p_out

    se = float(np.sqrt((p_in * (1.0 - p_in) / (cap_in + eps)) + (p_out * (1.0 - p_out) / (cap_out + eps))))
    z = float(delta / (se + eps))
    return float(delta), float(se), float(z), cap_in, cap_out


def load_run_metrics(run_dir: str) -> Dict[str, float]:
    """
    Reads results.npz from run_dir and returns key metrics.
    """
    npz_path = os.path.join(run_dir, "results.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing results.npz at {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    delta, se, z, cap_in, cap_out = compute_final_stats_from_npz(data)

    mean_edge_mem = data["mean_edge_mem"]
    locked_like = data["locked_like"]
    mean_theta_step = data["mean_theta_step"]

    out = {
        "delta": delta,
        "se": se,
        "z": z,
        "cap_in": float(cap_in),
        "cap_out": float(cap_out),
        "mean_edge_mem_end": float(mean_edge_mem[-1]) if mean_edge_mem.size else 0.0,
        "locked_like_end": float(locked_like[-1]) if locked_like.size else 0.0,
        "mean_theta_step_end": float(mean_theta_step[-1]) if mean_theta_step.size else 0.0,
    }
    return out


def run_v2_once(
    python_exe: str,
    v2_script_path: str,
    out_dir: str,
    base_args: List[str],
    override_args: List[str],
) -> str:
    """
    Runs the v2 script once and returns the created run_dir path (printed by v2).
    """
    cmd = [python_exe, v2_script_path] + base_args + override_args + ["--out_dir", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"v2 script failed with exit code {proc.returncode}")

    run_dir = None
    for line in proc.stdout.splitlines():
        m = RUN_DIR_RE.search(line.strip())
        if m:
            run_dir = m.group(1).strip()

    if not run_dir:
        # Fallback: try to infer newest directory in out_dir
        # (still safe, but less ideal)
        candidates = [os.path.join(out_dir, d) for d in os.listdir(out_dir)]
        candidates = [d for d in candidates if os.path.isdir(d)]
        if not candidates:
            raise RuntimeError("Could not parse run directory from output and no directories found in out_dir.")
        run_dir = max(candidates, key=lambda d: os.path.getmtime(d))

    return run_dir


@dataclass
class PointStats:
    x: float
    delta_mean: float
    delta_se_mean: float      # mean of per-run SE (not used for error bars)
    delta_sem: float          # SEM across seeds (for error bars)
    z_mean: float
    cap_in_mean: float
    cap_out_mean: float
    mean_edge_mem_end: float
    locked_like_end: float
    mean_theta_step_end: float


def aggregate_points(x_vals: np.ndarray, runs: List[List[Dict[str, float]]]) -> List[PointStats]:
    """
    For each x, we have a list of run metric dicts (one per seed).
    Compute mean, SEM, etc.
    """
    out: List[PointStats] = []
    for x, metrics_list in zip(x_vals, runs):
        deltas = np.array([m["delta"] for m in metrics_list], dtype=np.float64)
        ses = np.array([m["se"] for m in metrics_list], dtype=np.float64)
        zs = np.array([m["z"] for m in metrics_list], dtype=np.float64)
        cap_in = np.array([m["cap_in"] for m in metrics_list], dtype=np.float64)
        cap_out = np.array([m["cap_out"] for m in metrics_list], dtype=np.float64)
        mem_end = np.array([m["mean_edge_mem_end"] for m in metrics_list], dtype=np.float64)
        lock_end = np.array([m["locked_like_end"] for m in metrics_list], dtype=np.float64)
        th_end = np.array([m["mean_theta_step_end"] for m in metrics_list], dtype=np.float64)

        # SEM across seeds for delta
        n = max(1, deltas.size)
        delta_sem = float(deltas.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

        out.append(
            PointStats(
                x=float(x),
                delta_mean=float(deltas.mean()) if n else 0.0,
                delta_se_mean=float(ses.mean()) if n else 0.0,
                delta_sem=delta_sem,
                z_mean=float(zs.mean()) if n else 0.0,
                cap_in_mean=float(cap_in.mean()) if n else 0.0,
                cap_out_mean=float(cap_out.mean()) if n else 0.0,
                mean_edge_mem_end=float(mem_end.mean()) if n else 0.0,
                locked_like_end=float(lock_end.mean()) if n else 0.0,
                mean_theta_step_end=float(th_end.mean()) if n else 0.0,
            )
        )
    return out


def write_scan_csv(path: str, header_x: str, pts: List[PointStats]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            header_x,
            "delta_mean",
            "delta_sem",
            "z_mean",
            "cap_in_mean",
            "cap_out_mean",
            "mean_edge_mem_end",
            "locked_like_end",
            "mean_theta_step_end",
        ])
        for p in pts:
            w.writerow([
                p.x,
                p.delta_mean,
                p.delta_sem,
                p.z_mean,
                p.cap_in_mean,
                p.cap_out_mean,
                p.mean_edge_mem_end,
                p.locked_like_end,
                p.mean_theta_step_end,
            ])


def plot_scan(
    path: str,
    title: str,
    xlabel: str,
    pts: List[PointStats],
) -> None:
    x = np.array([p.x for p in pts], dtype=np.float64)
    y = np.array([p.delta_mean for p in pts], dtype=np.float64)
    yerr = np.array([p.delta_sem for p in pts], dtype=np.float64)

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
    plt.axhline(0.0, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("delta = p_in - p_out (cumulative)")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_regime(path: str, label: str, pts: List[PointStats]) -> None:
    """
    Scatter: delta vs end memory, color by locked_like_end.
    """
    mem = np.array([p.mean_edge_mem_end for p in pts], dtype=np.float64)
    delta = np.array([p.delta_mean for p in pts], dtype=np.float64)
    lock = np.array([p.locked_like_end for p in pts], dtype=np.float64)

    plt.figure()
    sc = plt.scatter(mem, delta, c=lock)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("mean_edge_mem_end")
    plt.ylabel("delta_mean")
    plt.title(f"Regime map: memory vs delta ({label})")
    plt.colorbar(sc, label="locked_like_end")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


# -------------------------
# Main harness
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["phase", "omega", "both"], default="both")
    ap.add_argument("--python", dest="python_exe", default=sys.executable)
    ap.add_argument("--v2_script", default="hsf_metastable_branch_toy_v2.py")
    ap.add_argument("--out_dir", default="hsf_scan_out")
    ap.add_argument("--tag", default="scan")

    # Sweep controls
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds per sweep point")

    ap.add_argument("--phi_min", type=float, default=-np.pi)
    ap.add_argument("--phi_max", type=float, default=np.pi)
    ap.add_argument("--phi_steps", type=int, default=25)

    ap.add_argument("--w_min", type=float, default=0.005)
    ap.add_argument("--w_max", type=float, default=0.08)
    ap.add_argument("--w_steps", type=int, default=25)

    # Base model args (pass through to v2)
    ap.add_argument("--N", type=int, default=250)
    ap.add_argument("--p_edge", type=float, default=0.03)
    ap.add_argument("--T", type=int, default=20000)

    ap.add_argument("--lambda_capture", type=float, default=0.05)
    ap.add_argument("--barrier_M", type=float, default=4.0)
    ap.add_argument("--tau_G", type=float, default=45.0)
    ap.add_argument("--tau_M", type=float, default=900.0)

    ap.add_argument("--drive_frac", type=float, default=0.20)
    ap.add_argument("--drive_amp", type=float, default=0.35)
    ap.add_argument("--drive_gain", type=float, default=2.2)

    # Use same rolling settings for the v2 output (doesn't affect delta computation)
    ap.add_argument("--rolling_window", type=int, default=1000)
    ap.add_argument("--min_captures_plot", type=int, default=20)

    # Keep these constant unless you explicitly want to scan them too
    ap.add_argument("--mem_couple", type=float, default=1.8)
    ap.add_argument("--kappa", type=float, default=2.5)
    ap.add_argument("--bandwidth", type=float, default=0.08)
    ap.add_argument("--g_coup", type=float, default=0.20)
    ap.add_argument("--theta_bw", type=float, default=0.10)

    ap.add_argument("--region_bandwidth_mult", type=float, default=1.10)
    ap.add_argument("--region_mem_mult", type=float, default=0.95)

    # For omega scan, keep phase fixed
    ap.add_argument("--phi_fixed", type=float, default=0.0)
    # For phase scan, keep omega fixed
    ap.add_argument("--w_fixed", type=float, default=0.03)

    args = ap.parse_args()

    v2_path = args.v2_script
    if not os.path.isfile(v2_path):
        raise FileNotFoundError(f"Could not find v2 script: {v2_path}")

    ensure_dir(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    ensure_dir(plots_dir)

    base_args = [
        "--N", str(args.N),
        "--p_edge", str(args.p_edge),
        "--T", str(args.T),

        "--lambda_capture", str(args.lambda_capture),
        "--barrier_M", str(args.barrier_M),
        "--tau_G", str(args.tau_G),
        "--tau_M", str(args.tau_M),

        "--drive_frac", str(args.drive_frac),
        "--drive_amp", str(args.drive_amp),
        "--drive_gain", str(args.drive_gain),

        "--rolling_window", str(args.rolling_window),
        "--min_captures_plot", str(args.min_captures_plot),

        "--mem_couple", str(args.mem_couple),
        "--kappa", str(args.kappa),
        "--bandwidth", str(args.bandwidth),
        "--g_coup", str(args.g_coup),
        "--theta_bw", str(args.theta_bw),

        "--region_bandwidth_mult", str(args.region_bandwidth_mult),
        "--region_mem_mult", str(args.region_mem_mult),
    ]

    seed_list = list(range(args.seeds))

    # -------------------------
    # PHASE SWEEP
    # -------------------------
    if args.mode in ("phase", "both"):
        phi_vals = linspace_inclusive(args.phi_min, args.phi_max, args.phi_steps)

        phase_runs: List[List[Dict[str, float]]] = []
        for i, phi in enumerate(phi_vals):
            metrics_this_phi: List[Dict[str, float]] = []
            for s in seed_list:
                # Keep omega fixed, sweep phase
                override = [
                    "--seed", str(s),
                    "--drive_w", str(args.w_fixed),
                    "--drive_phase", str(float(phi)),
                    "--run_name", f"{args.tag}_phase_i{i:03d}_seed{s}",
                ]
                run_dir = run_v2_once(args.python_exe, v2_path, args.out_dir, base_args, override)
                metrics_this_phi.append(load_run_metrics(run_dir))
            phase_runs.append(metrics_this_phi)

        phase_pts = aggregate_points(phi_vals, phase_runs)
        phase_csv = os.path.join(args.out_dir, "scan_phase.csv")
        write_scan_csv(phase_csv, "phi", phase_pts)

        plot_scan(
            os.path.join(plots_dir, "phase_scan.png"),
            title="Phase scan: delta vs drive_phase",
            xlabel="drive_phase (rad)",
            pts=phase_pts,
        )

        plot_regime(
            os.path.join(plots_dir, "regime_memory_vs_delta_phase.png"),
            label="phase scan",
            pts=phase_pts,
        )

        print("Saved:", phase_csv)
        print("Saved:", os.path.join(plots_dir, "phase_scan.png"))

    # -------------------------
    # OMEGA SWEEP
    # -------------------------
    if args.mode in ("omega", "both"):
        w_vals = linspace_inclusive(args.w_min, args.w_max, args.w_steps)

        omega_runs: List[List[Dict[str, float]]] = []
        for i, w in enumerate(w_vals):
            metrics_this_w: List[Dict[str, float]] = []
            for s in seed_list:
                # Keep phase fixed, sweep omega
                override = [
                    "--seed", str(s),
                    "--drive_w", str(float(w)),
                    "--drive_phase", str(args.phi_fixed),
                    "--run_name", f"{args.tag}_omega_i{i:03d}_seed{s}",
                ]
                run_dir = run_v2_once(args.python_exe, v2_path, args.out_dir, base_args, override)
                metrics_this_w.append(load_run_metrics(run_dir))
            omega_runs.append(metrics_this_w)

        omega_pts = aggregate_points(w_vals, omega_runs)
        omega_csv = os.path.join(args.out_dir, "scan_omega.csv")
        write_scan_csv(omega_csv, "omega", omega_pts)

        plot_scan(
            os.path.join(plots_dir, "omega_scan.png"),
            title="Omega scan: delta vs drive_w",
            xlabel="drive_w (rad/tick)",
            pts=omega_pts,
        )

        plot_regime(
            os.path.join(plots_dir, "regime_memory_vs_delta_omega.png"),
            label="omega scan",
            pts=omega_pts,
        )

        print("Saved:", omega_csv)
        print("Saved:", os.path.join(plots_dir, "omega_scan.png"))

    print("Done. Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
