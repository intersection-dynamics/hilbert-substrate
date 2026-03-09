#!/usr/bin/env python3
"""
subsystem_otoc_proxy.py
======================

OTOC-proxy via damage spreading (stable filename).
Version: 1.3

Key change vs v1.2:
- Default velocity estimator is now ARRIVAL-TIME based, not moving-front based.
  This avoids negative/zero slopes caused by dilution + threshold flicker.

Definitions:
- Two copies A and B with identical randomness.
- Poke node i at t=0 in copy B: xB[i] += eps * unit_vector.
- Damage per node: D_j(t) = ||xA_j(t) - xB_j(t)||^2
- Radial profile: C(r,t) = mean D_j(t) for nodes with graph distance r from poke i.

Threshold:
- thr = threshold_frac * C(0,0) (absolute reference to initial local damage)

Estimators:
1) arrival (default):
   t*(r) = first t where C(r,t) >= thr
   Fit r ≈ vB * t*(r) + b on r in [r_lo_frac*rmax, r_hi_frac*rmax]
2) front (legacy):
   r_front(t) = max r where C(r,t) >= thr
   Fit r_front(t) ≈ vB * t + b (can be noisy on this proxy)

Outputs:
- env.json, config.json (includes version)
- per-seed timeseries CSV (global_diff; plus r_front if estimator=front)
- per-seed NPZ (Cmat, derived arrays)
- plots for seed0: heatmap + (arrival plot or r_front plot)
- summary_by_seed.csv and aggregate_summary.csv
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

SCRIPT_VERSION = "1.3"


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


def smooth_time_axis(Cmat: np.ndarray, win: int) -> np.ndarray:
    """
    Simple moving average over time axis.
    Cmat shape: (T, R)
    """
    if win <= 1:
        return Cmat
    T, R = Cmat.shape
    out = np.zeros_like(Cmat)
    half = win // 2
    for t in range(T):
        a = max(0, t - half)
        b = min(T, t + half + 1)
        out[t] = np.mean(Cmat[a:b], axis=0)
    return out


# --------------------------
# Graphs with real distance
# --------------------------

def build_grid1d_ring(N: int) -> List[List[int]]:
    adj = [[] for _ in range(N)]
    for i in range(N):
        adj[i].append((i - 1) % N)
        adj[i].append((i + 1) % N)
    return [sorted(set(v)) for v in adj]


def _idx2d(x: int, y: int, L: int) -> int:
    return y * L + x


def build_grid2d_torus(L: int) -> List[List[int]]:
    N = L * L
    adj = [[] for _ in range(N)]
    for y in range(L):
        for x in range(L):
            u = _idx2d(x, y, L)
            adj[u].append(_idx2d((x - 1) % L, y, L))
            adj[u].append(_idx2d((x + 1) % L, y, L))
            adj[u].append(_idx2d(x, (y - 1) % L, L))
            adj[u].append(_idx2d(x, (y + 1) % L, L))
    return [sorted(set(v)) for v in adj]


def build_random_regular_ring_swaps(rng: np.random.Generator, N: int, deg: int, swaps_per_edge: float = 6.0) -> List[List[int]]:
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
# Measurement / fits
# --------------------------

@dataclass
class FitLine:
    slope: float
    intercept: float
    r2: float
    valid: int
    reason: str


def fit_line(x: np.ndarray, y: np.ndarray, min_points: int) -> FitLine:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < min_points:
        return FitLine(float("nan"), float("nan"), float("nan"), 0, "too_few_points")
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return FitLine(float("nan"), float("nan"), float("nan"), 0, "degenerate")
    c = np.polyfit(x, y, 1)
    yp = np.polyval(c, x)
    return FitLine(float(c[0]), float(c[1]), float(_r2(y, yp)), 1, "")


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


def arrival_times(Cmat: np.ndarray, thr: float) -> np.ndarray:
    """
    For each r, return first t where C(t,r) >= thr, else -1.
    Cmat shape (T, R)
    """
    T, R = Cmat.shape
    tstar = np.full(R, -1, dtype=int)
    for r in range(R):
        hits = np.where(Cmat[:, r] >= thr)[0]
        if hits.size:
            tstar[r] = int(hits[0])
    return tstar


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
    estimator: str,
    smooth_win: int,
    rfit_lo_frac: float,
    rfit_hi_frac: float,
    fit_min_points: int,
) -> dict:
    if graph == "grid1d":
        adj = build_grid1d_ring(N)
        N_eff = N
        deg_eff = 2
    elif graph == "grid2d":
        adj = build_grid2d_torus(L)
        N_eff = L * L
        deg_eff = 4
    elif graph == "random_regular":
        adj = build_random_regular_ring_swaps(rng, N, deg, swaps_per_edge=6.0)
        N_eff = N
        deg_eff = deg
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

    edges = []
    for u in range(N_eff):
        for v in adj[u]:
            if v > u:
                edges.append((u, v))
    edges = np.array(edges, dtype=int)
    E = len(edges)

    Cmat = np.zeros((nsteps + 1, rmax + 1), dtype=float)
    global_diff = np.zeros(nsteps + 1, dtype=float)

    # t=0
    D = np.sum((xA - xB) ** 2, axis=1)
    C0 = compute_C_r(dist, D, rmax)
    Cmat[0] = C0
    global_diff[0] = float(D.mean())

    thr = float(threshold_frac * max(float(C0[0]), 1e-30))

    # evolve
    for t in range(1, nsteps + 1):
        for _ in range(mix_per_step):
            ei = int(rng.integers(0, E))
            u, v = int(edges[ei, 0]), int(edges[ei, 1])
            R = random_rotation_2x2(rng)
            mix_edge_update(xA, xB, u, v, R)

        D = np.sum((xA - xB) ** 2, axis=1)
        Ct = compute_C_r(dist, D, rmax)
        Cmat[t] = Ct
        global_diff[t] = float(D.mean())

    Cuse = smooth_time_axis(Cmat, smooth_win) if smooth_win > 1 else Cmat

    if estimator == "arrival":
        tstar = arrival_times(Cuse, thr)  # length rmax+1

        # choose mid-range radii to fit
        r_lo = int(max(1, math.floor(rfit_lo_frac * rmax)))
        r_hi = int(max(r_lo + 1, math.floor(rfit_hi_frac * rmax)))
        r_hi = min(r_hi, rmax)

        rr = np.arange(r_lo, r_hi + 1, dtype=int)
        tt = tstar[rr]

        mask = (tt >= 0)
        rr2 = rr[mask].astype(float)
        tt2 = tt[mask].astype(float)

        # Fit r vs t*: r = vB * t + b
        fit = fit_line(tt2, rr2, min_points=fit_min_points)
        vB = fit.slope

        return {
            "graph": graph,
            "N": N_eff,
            "deg": deg_eff,
            "poke": poke,
            "rmax": rmax,
            "thr": thr,
            "Cmat": Cmat,
            "Cmat_smooth": Cuse,
            "global_diff": global_diff,
            "estimator": "arrival",
            "tstar": tstar,
            "fit": fit,
            "vB": vB,
            "fit_domain": {"r_lo": r_lo, "r_hi": r_hi, "n_used": int(len(rr2))},
        }

    elif estimator == "front":
        rfront = np.zeros(nsteps + 1, dtype=float)
        for t in range(nsteps + 1):
            rfront[t] = extract_front(Cuse[t], thr)
        # optional: enforce monotone front (prevents negative slope from flicker)
        rfront_mono = np.maximum.accumulate(rfront)

        t = np.arange(nsteps + 1, dtype=float)
        # fit on central window
        i0 = int(max(0, math.floor(0.15 * len(t))))
        i1 = int(min(len(t), math.floor(0.85 * len(t))))
        fit = fit_line(t[i0:i1], rfront_mono[i0:i1], min_points=fit_min_points)
        vB = fit.slope

        return {
            "graph": graph,
            "N": N_eff,
            "deg": deg_eff,
            "poke": poke,
            "rmax": rmax,
            "thr": thr,
            "Cmat": Cmat,
            "Cmat_smooth": Cuse,
            "global_diff": global_diff,
            "estimator": "front",
            "rfront": rfront,
            "rfront_mono": rfront_mono,
            "fit": fit,
            "vB": vB,
            "fit_domain": {"t0": i0, "t1": i1},
        }
    else:
        raise ValueError("unknown estimator")


def plot_seed0(outdir: str, label: str, result: dict) -> None:
    plt = _import_matplotlib()
    Cuse = result["Cmat_smooth"]

    # heatmap
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    img = np.log10(np.maximum(Cuse.T, 1e-30))
    ax.imshow(img, origin="lower", aspect="auto")
    ax.set_xlabel("t")
    ax.set_ylabel("r (graph distance)")
    ax.set_title(f"log10 C(r,t) heatmap — {label}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"heatmap_{label}.png"), dpi=170)
    plt.close(fig)

    if result["estimator"] == "arrival":
        tstar = result["tstar"]
        rr = np.where(tstar >= 0)[0]
        tt = tstar[rr].astype(float)
        rr = rr.astype(float)
        fit: FitLine = result["fit"]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(tt, rr, s=8, label="arrival points (t*, r)")
        if fit.valid == 1:
            xs = np.linspace(tt.min(), tt.max(), 200)
            ax.plot(xs, fit.slope * xs + fit.intercept, label=f"fit vB={fit.slope:.3f}, R2={fit.r2:.3f}")
        ax.set_xlabel("t* (arrival time)")
        ax.set_ylabel("r")
        ax.set_title(f"Arrival-time fit — {label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"arrival_fit_{label}.png"), dpi=170)
        plt.close(fig)

    else:
        rfront = result["rfront_mono"]
        fit: FitLine = result["fit"]
        t = np.arange(len(rfront), dtype=float)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, rfront, label="r_front_mono(t)")
        if fit.valid == 1:
            xs = np.linspace(t.min(), t.max(), 200)
            ax.plot(xs, fit.slope * xs + fit.intercept, label=f"fit vB={fit.slope:.3f}, R2={fit.r2:.3f}")
        ax.set_xlabel("t")
        ax.set_ylabel("r_front")
        ax.set_title(f"Front fit — {label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"front_fit_{label}.png"), dpi=170)
        plt.close(fig)


# --------------------------
# Main
# --------------------------

def main():
    p = argparse.ArgumentParser(description="Numerical OTOC proxy via damage spreading (stable filename)")

    p.add_argument("--graph", type=str, default="grid2d", choices=["grid1d", "grid2d", "random_regular"])
    p.add_argument("--L", type=int, default=64, help="grid2d side length (N=L*L)")
    p.add_argument("--N", type=int, default=1600, help="node count for grid1d or random_regular")
    p.add_argument("--deg", type=int, default=8, help="degree for random_regular (even)")

    p.add_argument("--nsteps", type=int, default=600)
    p.add_argument("--mix_per_step", type=int, default=1200)
    p.add_argument("--state_dim", type=int, default=2)

    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--threshold_frac", type=float, default=0.03)

    p.add_argument("--estimator", type=str, default="arrival", choices=["arrival", "front"])
    p.add_argument("--smooth_win", type=int, default=5, help="time smoothing window for C(r,t) (odd recommended).")

    p.add_argument("--rfit_lo_frac", type=float, default=0.15, help="arrival-fit lower radius fraction of rmax")
    p.add_argument("--rfit_hi_frac", type=float, default=0.85, help="arrival-fit upper radius fraction of rmax")
    p.add_argument("--fit_min_points", type=int, default=30)

    p.add_argument("--ntrials", type=int, default=20)
    p.add_argument("--seed0", type=int, default=0)

    p.add_argument("--out_root", type=str, default="hsf_out")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--no_plots", action="store_true")

    args = p.parse_args()

    stamp = _now_stamp()
    git_hash = _try_git_hash()
    suffix = f"__{args.run_name}" if args.run_name else ""
    run_dir = os.path.join(args.out_root, f"{stamp}__otoc_proxy_v{SCRIPT_VERSION}__git{git_hash}{suffix}")
    _safe_makedirs(run_dir)
    _safe_makedirs(os.path.join(run_dir, "plots"))
    _safe_makedirs(os.path.join(run_dir, "npz"))
    _safe_makedirs(os.path.join(run_dir, "timeseries"))

    env = {
        "script_version": SCRIPT_VERSION,
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
        "argv": sys.argv,
        "git_hash": git_hash,
        "timestamp_local": stamp,
    }
    _write_json(os.path.join(run_dir, "env.json"), env)
    cfg = vars(args).copy()
    cfg["script_version"] = SCRIPT_VERSION
    _write_json(os.path.join(run_dir, "config.json"), cfg)

    seeds = [int(args.seed0 + i) for i in range(int(args.ntrials))]
    vBs = []
    r2s = []
    seed_rows = []

    rng_ci = np.random.default_rng(12345)

    t0 = time.time()
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        res = run_single_trial(
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
            estimator=args.estimator,
            smooth_win=int(args.smooth_win),
            rfit_lo_frac=float(args.rfit_lo_frac),
            rfit_hi_frac=float(args.rfit_hi_frac),
            fit_min_points=int(args.fit_min_points),
        )

        # Save NPZ
        npz_path = os.path.join(run_dir, "npz", f"seed{seed:04d}.npz")
        save_dict = {
            "Cmat": res["Cmat"],
            "Cmat_smooth": res["Cmat_smooth"],
            "global_diff": res["global_diff"],
            "thr": float(res["thr"]),
        }
        if res["estimator"] == "arrival":
            save_dict["tstar"] = res["tstar"]
        else:
            save_dict["rfront"] = res["rfront"]
            save_dict["rfront_mono"] = res["rfront_mono"]
        np.savez_compressed(npz_path, **save_dict)

        # Save timeseries
        ts_path = os.path.join(run_dir, "timeseries", f"timeseries_seed{seed:04d}.csv")
        with open(ts_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if res["estimator"] == "front":
                w.writerow(["t", "global_diff", "rfront_mono"])
                for t in range(len(res["global_diff"])):
                    w.writerow([t, float(res["global_diff"][t]), float(res["rfront_mono"][t])])
            else:
                w.writerow(["t", "global_diff"])
                for t in range(len(res["global_diff"])):
                    w.writerow([t, float(res["global_diff"][t])])

        fit: FitLine = res["fit"]
        vB = float(res["vB"])

        row = {
            "seed": seed,
            "graph": res["graph"],
            "N": int(res["N"]),
            "deg": int(res["deg"]),
            "rmax": int(res["rmax"]),
            "thr": float(res["thr"]),
            "estimator": res["estimator"],
            "vB": vB,
            "fit_r2": float(fit.r2),
            "fit_valid": int(fit.valid),
            "fit_reason": str(fit.reason),
        }
        # add domain info
        for k, v in res.get("fit_domain", {}).items():
            row[f"fit_{k}"] = v

        seed_rows.append(row)

        if fit.valid == 1 and np.isfinite(vB):
            vBs.append(vB)
            r2s.append(float(fit.r2))

        if (not args.no_plots) and si == 0:
            label = f"{args.graph}_{args.estimator}_seed{seed:04d}"
            plot_seed0(os.path.join(run_dir, "plots"), label, res)

    # summary_by_seed.csv
    seed_csv = os.path.join(run_dir, "summary_by_seed.csv")
    cols = list(seed_rows[0].keys())
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in seed_rows:
            w.writerow([row.get(c, "") for c in cols])

    vBs_arr = np.array(vBs, dtype=float)
    lo, hi = _bootstrap_ci_mean(vBs_arr, rng=rng_ci) if len(vBs_arr) else (float("nan"), float("nan"))
    agg = {
        "script_version": SCRIPT_VERSION,
        "graph": args.graph,
        "estimator": args.estimator,
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
    print(f"  script_version = {SCRIPT_VERSION}")
    print(f"  graph = {agg['graph']}  estimator={agg['estimator']}  N={agg['N']}  deg={agg['deg']}")
    print(f"  valid_vB_fraction = {agg['valid_vB_fraction']:.2f}")
    print(f"  vB_mean = {agg['vB_mean']:.4f}  std={agg['vB_std']:.4f}  CI95=[{agg['vB_ci_lo']:.4f},{agg['vB_ci_hi']:.4f}]")
    print(f"  fit_r2_mean = {agg['fit_r2_mean']:.3f}")
    print(f"\nRun folder: {run_dir}")
    print(f"Aggregate summary: {agg_csv}")
    print(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()