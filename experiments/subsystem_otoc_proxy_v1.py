#!/usr/bin/env python3
"""
subsystem_otoc_proxy_v1_1.py
===========================

Fix: robust k-regular graph construction.
The v1 stub-matching generator can fail for some seeds. Here we use:

  - deterministic k-regular ring lattice (always succeeds)
  - degree-preserving double-edge swaps (randomizes without breaking simplicity)

Everything else is the same: numerical OTOC proxy via damage spreading.

Run:
  python subsystem_otoc_proxy_v1_1.py --mode single
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# --------------------------
# Utilities
# --------------------------

def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _try_git_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="ignore").strip()
        return s if s else "nogit"
    except Exception:
        return "nogit"


def _r2(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    ss = float(np.sum((y - yp) ** 2))
    st = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss / max(st, 1e-12)


def _bootstrap_ci_mean(values: np.ndarray, n_boot: int = 4000, ci: float = 0.95, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return lo, hi


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------
# Robust k-regular graph
# --------------------------

def build_k_regular_ring(N: int, deg: int) -> List[set]:
    """
    Deterministic k-regular ring lattice:
      connect i to i +/- 1..deg/2 (mod N)
    Requires deg even and deg < N.
    Always succeeds and is simple.
    """
    if deg <= 0:
        return [set() for _ in range(N)]
    if deg >= N:
        raise ValueError("deg must be < N")
    if deg % 2 != 0:
        raise ValueError("ring builder requires even deg; choose even deg (e.g. 8).")
    adj = [set() for _ in range(N)]
    half = deg // 2
    for i in range(N):
        for k in range(1, half + 1):
            j1 = (i + k) % N
            j2 = (i - k) % N
            adj[i].add(j1); adj[j1].add(i)
            adj[i].add(j2); adj[j2].add(i)
    return adj


def _pick_random_edge(rng: np.random.Generator, adj: List[set]) -> Tuple[int, int]:
    N = len(adj)
    while True:
        a = int(rng.integers(0, N))
        if not adj[a]:
            continue
        b = int(list(adj[a])[int(rng.integers(0, len(adj[a])))])
        if a != b:
            return (a, b) if a < b else (b, a)


def randomize_by_double_edge_swaps(
    rng: np.random.Generator,
    adj: List[set],
    n_swaps: int,
    max_tries: int = 50_000
) -> None:
    """
    Degree-preserving randomization.
    Repeatedly pick edges (a,b) and (c,d) and swap to (a,d),(c,b) if that keeps graph simple.

    This never destroys regularity and is very robust.
    """
    tries = 0
    swaps = 0
    N = len(adj)
    while swaps < n_swaps and tries < max_tries:
        tries += 1

        a, b = _pick_random_edge(rng, adj)
        c, d = _pick_random_edge(rng, adj)

        # ensure distinct endpoints
        if len({a, b, c, d}) < 4:
            continue

        # propose swap
        # edges (a,b) and (c,d) -> (a,d) and (c,b)
        if (d in adj[a]) or (b in adj[c]):
            continue
        if a == d or c == b:
            continue

        # perform swap
        adj[a].remove(b); adj[b].remove(a)
        adj[c].remove(d); adj[d].remove(c)

        adj[a].add(d); adj[d].add(a)
        adj[c].add(b); adj[b].add(c)

        swaps += 1

    # It's okay if swaps < n_swaps; graph is still valid.
    # (For small graphs or large deg, many proposals collide.)
    # We do NOT fail.

def build_random_regular_graph_robust(rng: np.random.Generator, N: int, deg: int, swaps_per_edge: float = 6.0) -> List[List[int]]:
    """
    Guaranteed construction:
      - ring lattice (requires deg even)
      - randomize with swaps

    For OTOC proxy we care about:
      - bounded degree
      - short-ish path lengths
      - "random enough" mixing

    This provides that without failures.
    """
    adj = build_k_regular_ring(N, deg)
    # approximate edges
    E = (N * deg) // 2
    n_swaps = int(max(0, round(swaps_per_edge * E)))
    randomize_by_double_edge_swaps(rng, adj, n_swaps=n_swaps)
    return [sorted(list(s)) for s in adj]


def bfs_distances(adj: List[List[int]], src: int) -> np.ndarray:
    N = len(adj)
    dist = np.full(N, -1, dtype=int)
    q = [src]
    dist[src] = 0
    qi = 0
    while qi < len(q):
        u = q[qi]; qi += 1
        du = dist[u]
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = du + 1
                q.append(v)
    return dist


# --------------------------
# Local mixing dynamics
# --------------------------

def random_rotation_2x2(rng: np.random.Generator) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def mix_edge_update(xA: np.ndarray, xB: np.ndarray, u: int, v: int, R: np.ndarray) -> None:
    a = xA[u].copy()
    b = xA[v].copy()
    xA[u] = R[0, 0] * a + R[0, 1] * b
    xA[v] = R[1, 0] * a + R[1, 1] * b

    a2 = xB[u].copy()
    b2 = xB[v].copy()
    xB[u] = R[0, 0] * a2 + R[0, 1] * b2
    xB[v] = R[1, 0] * a2 + R[1, 1] * b2


# --------------------------
# OTOC proxy measurement
# --------------------------

@dataclass
class FrontFit:
    vB: float
    intercept: float
    r2: float
    t0: int
    t1: int
    valid: int
    reason: str


def fit_front_velocity(t: np.ndarray, rfront: np.ndarray, start_frac: float, end_frac: float, min_points: int) -> FrontFit:
    t = np.asarray(t, dtype=float)
    rfront = np.asarray(rfront, dtype=float)
    n = len(t)
    i0 = int(max(0, math.floor(n * start_frac)))
    i1 = int(max(i0 + min_points, math.floor(n * end_frac)))
    i1 = min(i1, n)
    tf = t[i0:i1]
    rf = rfront[i0:i1]
    if len(tf) < min_points:
        return FrontFit(float("nan"), float("nan"), float("nan"), int(tf[0]) if len(tf) else 0, int(tf[-1]) if len(tf) else 0, 0, "too_few_points")
    if float(np.std(rf)) < 1e-6:
        return FrontFit(float("nan"), float("nan"), float("nan"), int(tf[0]), int(tf[-1]), 0, "front_constant")
    c = np.polyfit(tf, rf, 1)
    yp = np.polyval(c, tf)
    return FrontFit(float(c[0]), float(c[1]), float(_r2(rf, yp)), int(tf[0]), int(tf[-1]), 1, "")


def compute_C_r(dist: np.ndarray, D: np.ndarray, rmax: int) -> np.ndarray:
    C = np.zeros(rmax + 1, dtype=float)
    counts = np.zeros(rmax + 1, dtype=float)
    for j, r in enumerate(dist):
        if r < 0 or r > rmax:
            continue
        C[r] += float(D[j])
        counts[r] += 1.0
    return C / np.maximum(counts, 1.0)


def extract_front(Cr: np.ndarray, threshold: float) -> int:
    idx = np.where(Cr >= threshold)[0]
    return int(idx.max()) if idx.size else 0


# --------------------------
# Runner
# --------------------------

def run_single_trial(
    rng: np.random.Generator,
    N: int,
    deg: int,
    nsteps: int,
    mix_per_step: int,
    state_dim: int,
    eps: float,
    threshold_frac: float,
    fit_start_frac: float,
    fit_end_frac: float,
    fit_min_points: int,
) -> dict:
    if deg % 2 != 0:
        raise ValueError("deg must be even in v1_1 (ring+swap regular graph). Use deg=8,10,12,...")
    adj = build_random_regular_graph_robust(rng, N, deg, swaps_per_edge=6.0)

    poke = int(rng.integers(0, N))
    dist = bfs_distances(adj, poke)
    rmax = int(dist.max())

    xA = rng.normal(size=(N, state_dim)).astype(float)
    xB = xA.copy()

    dvec = rng.normal(size=(state_dim,)).astype(float)
    dvec /= max(float(np.linalg.norm(dvec)), 1e-12)
    xB[poke] = xB[poke] + eps * dvec

    # edge list
    edges = []
    for u in range(N):
        for v in adj[u]:
            if v > u:
                edges.append((u, v))
    edges = np.array(edges, dtype=int)
    E = len(edges)

    Cmat = np.zeros((nsteps + 1, rmax + 1), dtype=float)
    rfront = np.zeros(nsteps + 1, dtype=float)
    global_diff = np.zeros(nsteps + 1, dtype=float)

    D = np.sum((xA - xB) ** 2, axis=1)
    C0 = compute_C_r(dist, D, rmax)
    Cmat[0] = C0
    th0 = float(threshold_frac * max(float(C0.max()), 1e-30))
    rfront[0] = extract_front(C0, th0)
    global_diff[0] = float(D.mean())

    for t in range(1, nsteps + 1):
        for _ in range(mix_per_step):
            ei = int(rng.integers(0, E))
            u, v = int(edges[ei, 0]), int(edges[ei, 1])
            R = random_rotation_2x2(rng)
            mix_edge_update(xA, xB, u, v, R)

        D = np.sum((xA - xB) ** 2, axis=1)
        Ct = compute_C_r(dist, D, rmax)
        Cmat[t] = Ct
        th = float(threshold_frac * max(float(Ct.max()), 1e-30))
        rfront[t] = extract_front(Ct, th)
        global_diff[t] = float(D.mean())

    tt = np.arange(nsteps + 1, dtype=float)
    ff = fit_front_velocity(tt, rfront, fit_start_frac, fit_end_frac, fit_min_points)

    return {
        "deg": deg,
        "poke": poke,
        "rmax": rmax,
        "Cmat": Cmat,
        "rfront": rfront,
        "global_diff": global_diff,
        "fit": ff,
    }


def plot_trial(outdir: str, label: str, Cmat: np.ndarray, rfront: np.ndarray, fit: FrontFit) -> None:
    plt = _import_matplotlib()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    img = np.log10(np.maximum(Cmat.T, 1e-30))
    ax.imshow(img, origin="lower", aspect="auto")
    ax.set_xlabel("t")
    ax.set_ylabel("r (graph distance)")
    ax.set_title(f"log10 C(r,t) heatmap — {label}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"heatmap_{label}.png"), dpi=170)
    plt.close(fig)

    t = np.arange(len(rfront), dtype=float)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, rfront, label="r_front(t)")
    if fit.valid == 1:
        tf = t[(t >= fit.t0) & (t <= fit.t1)]
        ax.plot(tf, fit.vB * tf + fit.intercept, label=f"fit vB={fit.vB:.3f}, R2={fit.r2:.3f}")
    ax.set_xlabel("t")
    ax.set_ylabel("r_front")
    ax.set_title(f"Butterfly front — {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"rfront_{label}.png"), dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Numerical OTOC proxy via damage spreading on a robust random-regular graph (v1.1)")

    p.add_argument("--mode", type=str, default="single", choices=["single", "sweep_degree"])
    p.add_argument("--N", type=int, default=1600)
    p.add_argument("--deg", type=int, default=8)
    p.add_argument("--deg_list", type=str, default="4,6,8,10")

    p.add_argument("--nsteps", type=int, default=600)
    p.add_argument("--mix_per_step", type=int, default=1200)
    p.add_argument("--state_dim", type=int, default=2)

    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--threshold_frac", type=float, default=0.1)

    p.add_argument("--ntrials", type=int, default=20)
    p.add_argument("--seed0", type=int, default=0)

    p.add_argument("--fit_start_frac", type=float, default=0.15)
    p.add_argument("--fit_end_frac", type=float, default=0.85)
    p.add_argument("--fit_min_points", type=int, default=80)

    p.add_argument("--out_root", type=str, default="hsf_out")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--no_plots", action="store_true")

    args = p.parse_args()

    stamp = _now_stamp()
    git_hash = _try_git_hash()
    suffix = f"__{args.run_name}" if args.run_name else ""
    run_dir = os.path.join(args.out_root, f"{stamp}__otoc_proxy_v1_1__git{git_hash}{suffix}")
    _safe_makedirs(run_dir)
    _safe_makedirs(os.path.join(run_dir, "plots"))
    _safe_makedirs(os.path.join(run_dir, "npz"))
    _safe_makedirs(os.path.join(run_dir, "timeseries"))

    env = {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
        "argv": sys.argv,
        "git_hash": git_hash,
        "timestamp_local": stamp,
    }
    _write_json(os.path.join(run_dir, "env.json"), env)
    _write_json(os.path.join(run_dir, "config.json"), vars(args))

    if args.mode == "single":
        degs = [int(args.deg)]
    else:
        degs = [int(x.strip()) for x in args.deg_list.split(",") if x.strip()]

    rng_ci = np.random.default_rng(12345)
    agg_rows = []

    t0 = time.time()
    for deg in degs:
        print(f"\n{'='*78}\nRunning degree: {deg}\n{'='*78}")
        seeds = [int(args.seed0 + i) for i in range(int(args.ntrials))]

        seed_rows = []
        vBs = []
        r2s = []

        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            out = run_single_trial(
                rng=rng,
                N=int(args.N),
                deg=int(deg),
                nsteps=int(args.nsteps),
                mix_per_step=int(args.mix_per_step),
                state_dim=int(args.state_dim),
                eps=float(args.eps),
                threshold_frac=float(args.threshold_frac),
                fit_start_frac=float(args.fit_start_frac),
                fit_end_frac=float(args.fit_end_frac),
                fit_min_points=int(args.fit_min_points),
            )

            Cmat = out["Cmat"]
            rfront = out["rfront"]
            gd = out["global_diff"]
            fit: FrontFit = out["fit"]

            np.savez_compressed(
                os.path.join(run_dir, "npz", f"Cmat_deg{deg}_seed{seed:04d}.npz"),
                Cmat=Cmat, rfront=rfront, global_diff=gd
            )

            ts_path = os.path.join(run_dir, "timeseries", f"timeseries_deg{deg}_seed{seed:04d}.csv")
            with open(ts_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t", "rfront", "global_diff"])
                for t, rf, g in zip(range(len(rfront)), rfront, gd):
                    w.writerow([t, float(rf), float(g)])

            if (not args.no_plots) and si == 0:
                label = f"deg{deg}_seed{seed:04d}"
                plot_trial(os.path.join(run_dir, "plots"), label, Cmat, rfront, fit)

            row = {
                "deg": deg,
                "seed": seed,
                "vB": fit.vB,
                "fit_r2": fit.r2,
                "fit_valid": fit.valid,
                "fit_reason": fit.reason,
                "fit_t0": fit.t0,
                "fit_t1": fit.t1,
                "rmax": int(out["rmax"]),
                "global_diff_final": float(gd[-1]),
                "rfront_final": float(rfront[-1]),
            }
            seed_rows.append(row)

            if fit.valid == 1 and np.isfinite(fit.vB):
                vBs.append(float(fit.vB))
                r2s.append(float(fit.r2))

        seed_csv = os.path.join(run_dir, f"summary_by_seed_deg{deg}.csv")
        cols = list(seed_rows[0].keys())
        with open(seed_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for row in seed_rows:
                w.writerow([row[c] for c in cols])

        vBs_arr = np.array(vBs, dtype=float)
        lo, hi = _bootstrap_ci_mean(vBs_arr, rng=rng_ci) if len(vBs_arr) else (float("nan"), float("nan"))

        agg = {
            "deg": deg,
            "valid_vB_fraction": float(len(vBs_arr) / max(1, len(seed_rows))),
            "vB_mean": float(np.mean(vBs_arr)) if len(vBs_arr) else float("nan"),
            "vB_std": float(np.std(vBs_arr)) if len(vBs_arr) else float("nan"),
            "vB_ci_lo": float(lo),
            "vB_ci_hi": float(hi),
            "fit_r2_mean": float(np.mean(np.array(r2s, dtype=float))) if len(r2s) else float("nan"),
        }
        agg_rows.append(agg)

        print(f"  valid_vB_fraction = {agg['valid_vB_fraction']:.2f}")
        print(f"  vB_mean = {agg['vB_mean']:.3f}  std = {agg['vB_std']:.3f}  CI95 = [{agg['vB_ci_lo']:.3f}, {agg['vB_ci_hi']:.3f}]")
        print(f"  fit_r2_mean = {agg['fit_r2_mean']:.3f}")

    agg_csv = os.path.join(run_dir, "aggregate_summary.csv")
    if agg_rows:
        cols = list(agg_rows[0].keys())
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for row in agg_rows:
                w.writerow([row.get(c, "") for c in cols])

    if (not args.no_plots) and len(agg_rows) > 1:
        plt = _import_matplotlib()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        deg_x = [r["deg"] for r in agg_rows]
        vB_y = [r["vB_mean"] for r in agg_rows]
        ax.plot(deg_x, vB_y, marker="o")
        ax.set_xlabel("degree (bandwidth proxy)")
        ax.set_ylabel("vB (butterfly velocity)")
        ax.set_title("vB vs degree")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "plots", "vB_vs_degree.png"), dpi=170)
        plt.close(fig)

    elapsed = time.time() - t0
    print(f"\nRun folder: {run_dir}")
    print(f"Aggregate summary: {agg_csv}")
    print(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()