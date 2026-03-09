#!/usr/bin/env python3
"""
subsystem_otoc_proxy_v1_2.py
===========================

Numerical OTOC analogue via "damage spreading" between two copies of the same dynamics.

Key fix vs v1/v1.1:
- Default graph is a *grid torus* (1D ring or 2D torus), so graph distance behaves like space.
- Butterfly front threshold is *absolute* relative to initial local damage:
    thr = threshold_frac * C(r=0, t=0)
  This avoids threshold artifacts on fast-mixing graphs.

Dynamics:
- Each micro-update picks an edge (u,v) and applies a random 2x2 rotation to the pair
  (for each component of a local state vector). Same randomness applied to both copies.
- Copy B receives a small poke at one node at t=0.

Measurements:
- D_j(t) = ||xA_j - xB_j||^2
- C(r,t) = mean D_j(t) for nodes at graph distance r from poke node
- r_front(t) = max r with C(r,t) >= thr
- Fit r_front(t) ~ vB * t + b on a time window.
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
# Graphs with real distance
# --------------------------

def build_grid1d_ring(N: int) -> List[List[int]]:
    # degree 2 ring
    adj = [[] for _ in range(N)]
    for i in range(N):
        adj[i].append((i - 1) % N)
        adj[i].append((i + 1) % N)
    return [sorted(set(v)) for v in adj]


def idx2d(x: int, y: int, L: int) -> int:
    return y * L + x


def build_grid2d_torus(L: int) -> List[List[int]]:
    # degree 4 torus
    N = L * L
    adj = [[] for _ in range(N)]
    for y in range(L):
        for x in range(L):
            u = idx2d(x, y, L)
            adj[u].append(idx2d((x - 1) % L, y, L))
            adj[u].append(idx2d((x + 1) % L, y, L))
            adj[u].append(idx2d(x, (y - 1) % L, L))
            adj[u].append(idx2d(x, (y + 1) % L, L))
    return [sorted(set(v)) for v in adj]


def build_random_regular_ring_swaps(rng: np.random.Generator, N: int, deg: int, swaps_per_edge: float = 6.0) -> List[List[int]]:
    # ring + swaps (kept for optional use)
    if deg % 2 != 0:
        raise ValueError("random_regular requires even deg (ring base).")
    half = deg // 2
    adj = [set() for _ in range(N)]
    for i in range(N):
        for k in range(1, half + 1):
            j1 = (i + k) % N
            j2 = (i - k) % N
            adj[i].add(j1); adj[j1].add(i)
            adj[i].add(j2); adj[j2].add(i)

    def pick_edge():
        while True:
            a = int(rng.integers(0, N))
            if not adj[a]:
                continue
            b = int(list(adj[a])[int(rng.integers(0, len(adj[a])))])
            if a != b:
                return (a, b) if a < b else (b, a)

    E = (N * deg) // 2
    n_swaps = int(max(0, round(swaps_per_edge * E)))
    tries = 0
    swaps = 0
    while swaps < n_swaps and tries < 100_000:
        tries += 1
        a, b = pick_edge()
        c, d = pick_edge()
        if len({a, b, c, d}) < 4:
            continue
        if (d in adj[a]) or (b in adj[c]) or a == d or c == b:
            continue
        adj[a].remove(b); adj[b].remove(a)
        adj[c].remove(d); adj[d].remove(c)
        adj[a].add(d); adj[d].add(a)
        adj[c].add(b); adj[b].add(c)
        swaps += 1

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


def extract_front(Cr: np.ndarray, thr: float) -> int:
    idx = np.where(Cr >= thr)[0]
    return int(idx.max()) if idx.size else 0


# --------------------------
# Trial
# --------------------------

def run_single_trial(
    rng: np.random.Generator,
    graph: str,
    N: int,
    L: int,
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
    if graph == "grid1d":
        adj = build_grid1d_ring(N)
        N_eff = N
    elif graph == "grid2d":
        adj = build_grid2d_torus(L)
        N_eff = L * L
    elif graph == "random_regular":
        adj = build_random_regular_ring_swaps(rng, N, deg, swaps_per_edge=6.0)
        N_eff = N
    else:
        raise ValueError("unknown graph")

    poke = int(rng.integers(0, N_eff))
    dist = bfs_distances(adj, poke)
    rmax = int(dist.max())

    xA = rng.normal(size=(N_eff, state_dim)).astype(float)
    xB = xA.copy()

    dvec = rng.normal(size=(state_dim,)).astype(float)
    dvec /= max(float(np.linalg.norm(dvec)), 1e-12)
    xB[poke] = xB[poke] + eps * dvec

    # edge list
    edges = []
    for u in range(N_eff):
        for v in adj[u]:
            if v > u:
                edges.append((u, v))
    edges = np.array(edges, dtype=int)
    E = len(edges)

    Cmat = np.zeros((nsteps + 1, rmax + 1), dtype=float)
    rfront = np.zeros(nsteps + 1, dtype=float)
    global_diff = np.zeros(nsteps + 1, dtype=float)

    # t=0
    D = np.sum((xA - xB) ** 2, axis=1)
    C0 = compute_C_r(dist, D, rmax)
    Cmat[0] = C0
    thr = float(threshold_frac * max(float(C0[0]), 1e-30))  # FIXED absolute threshold relative to initial local damage
    rfront[0] = extract_front(C0, thr)
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
        rfront[t] = extract_front(Ct, thr)
        global_diff[t] = float(D.mean())

    tt = np.arange(nsteps + 1, dtype=float)
    ff = fit_front_velocity(tt, rfront, fit_start_frac, fit_end_frac, fit_min_points)

    return {
        "graph": graph,
        "N": N_eff,
        "deg": deg if graph == "random_regular" else (2 if graph == "grid1d" else 4),
        "poke": poke,
        "rmax": rmax,
        "thr": thr,
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


# --------------------------
# Main
# --------------------------

def main():
    p = argparse.ArgumentParser(description="Numerical OTOC proxy via damage spreading (grid graphs) v1.2")

    p.add_argument("--graph", type=str, default="grid2d", choices=["grid1d", "grid2d", "random_regular"])

    # grid params
    p.add_argument("--L", type=int, default=64, help="grid2d side length (N=L*L)")
    p.add_argument("--N", type=int, default=1600, help="node count for grid1d or random_regular")
    p.add_argument("--deg", type=int, default=8, help="degree for random_regular (even)")

    p.add_argument("--mode", type=str, default="single", choices=["single"])
    p.add_argument("--nsteps", type=int, default=500)
    p.add_argument("--mix_per_step", type=int, default=800)
    p.add_argument("--state_dim", type=int, default=2)

    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--threshold_frac", type=float, default=0.05)

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
    run_dir = os.path.join(args.out_root, f"{stamp}__otoc_proxy_v1_2__git{git_hash}{suffix}")
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

    seeds = [int(args.seed0 + i) for i in range(int(args.ntrials))]
    vBs = []
    r2s = []
    seed_rows = []

    rng_ci = np.random.default_rng(12345)

    t0 = time.time()
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        out = run_single_trial(
            rng=rng,
            graph=args.graph,
            N=int(args.N),
            L=int(args.L),
            deg=int(args.deg),
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
            os.path.join(run_dir, "npz", f"Cmat_seed{seed:04d}.npz"),
            Cmat=Cmat, rfront=rfront, global_diff=gd, thr=float(out["thr"])
        )

        ts_path = os.path.join(run_dir, "timeseries", f"timeseries_seed{seed:04d}.csv")
        with open(ts_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "rfront", "global_diff"])
            for t, rf, g in zip(range(len(rfront)), rfront, gd):
                w.writerow([t, float(rf), float(g)])

        if (not args.no_plots) and si == 0:
            label = f"{args.graph}_seed{seed:04d}"
            plot_trial(os.path.join(run_dir, "plots"), label, Cmat, rfront, fit)

        seed_rows.append({
            "seed": seed,
            "graph": args.graph,
            "N": int(out["N"]),
            "deg": int(out["deg"]),
            "rmax": int(out["rmax"]),
            "thr": float(out["thr"]),
            "vB": fit.vB,
            "fit_r2": fit.r2,
            "fit_valid": fit.valid,
            "fit_reason": fit.reason,
            "fit_t0": fit.t0,
            "fit_t1": fit.t1,
            "global_diff_final": float(gd[-1]),
            "rfront_final": float(rfront[-1]),
        })

        if fit.valid == 1 and np.isfinite(fit.vB):
            vBs.append(float(fit.vB))
            r2s.append(float(fit.r2))

    # summary_by_seed.csv
    seed_csv = os.path.join(run_dir, "summary_by_seed.csv")
    cols = list(seed_rows[0].keys())
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in seed_rows:
            w.writerow([row[c] for c in cols])

    vBs_arr = np.array(vBs, dtype=float)
    lo, hi = _bootstrap_ci_mean(vBs_arr, rng=rng_ci) if len(vBs_arr) else (float("nan"), float("nan"))
    agg = {
        "graph": args.graph,
        "N": int(seed_rows[0]["N"]),
        "deg": int(seed_rows[0]["deg"]),
        "valid_vB_fraction": float(len(vBs_arr) / max(1, len(seed_rows))),
        "vB_mean": float(np.mean(vBs_arr)) if len(vBs_arr) else float("nan"),
        "vB_std": float(np.std(vBs_arr)) if len(vBs_arr) else float("nan"),
        "vB_ci_lo": float(lo),
        "vB_ci_hi": float(hi),
        "fit_r2_mean": float(np.mean(np.array(r2s, dtype=float))) if len(r2s) else float("nan"),
    }

    agg_csv = os.path.join(run_dir, "aggregate_summary.csv")
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(agg.keys()))
        w.writerow([agg[k] for k in agg.keys()])

    elapsed = time.time() - t0
    print(f"\n{'='*78}\nOTOC proxy summary\n{'='*78}")
    print(f"  graph = {agg['graph']}  N={agg['N']}  deg={agg['deg']}")
    print(f"  valid_vB_fraction = {agg['valid_vB_fraction']:.2f}")
    print(f"  vB_mean = {agg['vB_mean']:.4f}  std={agg['vB_std']:.4f}  CI95=[{agg['vB_ci_lo']:.4f},{agg['vB_ci_hi']:.4f}]")
    print(f"  fit_r2_mean = {agg['fit_r2_mean']:.3f}")
    print(f"\nRun folder: {run_dir}")
    print(f"Aggregate summary: {agg_csv}")
    print(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()