#!/usr/bin/env python3
"""
run_link_phase_sweep.py

Sweep-and-report driver for link-weight optimization across
lambda = log(alpha/beta), with standardized artifact bundle.

Implements the exact spec from Michael R. Calder:
  - CLI-driven lambda grid (explicit or min/max/steps)
  - seeds × sweep-direction × lambda
  - R restarts per point, optional warm-start
  - saves W_best + metrics.json per point
  - writes summary.csv + optional summary_plots.png
  - resume/caching (skip computed points unless --force)
  - per-point timing logs

CRITICAL NOTE (adapter requirement):
-----------------------------------
This driver intentionally does NOT implement your optimizer. It calls into your
existing optimizer code as "source of truth".

You must provide a Python module (in your repo) that exposes:

  build_problem(*, n:int, seed:int, alpha:float, beta:float) -> Any
  optimize_links(problem, *, restarts:int, iters:int, init_W:np.ndarray|None, seed:int) -> dict

Where optimize_links(...) returns a dict with at least:
  {
    "W_best": np.ndarray (n,n) symmetric (or will be symmetrized),
    "obj_best": float,
    "obj_components": dict (optional),
    "W_restarts": list[np.ndarray] (optional but strongly recommended),
    "obj_restarts": list[float] (optional but strongly recommended)
  }

By default, this script will try to import:
  optimizer module: "link_optimizer"
  functions: build_problem, optimize_links

You can override via optional CLI:
  --optimizer-module your.module.name
  --build-fn build_problem
  --optimize-fn optimize_links

If your function signatures differ, write a tiny adapter module that matches
the expected interface (recommended).

Author: Ben Bray (driver implementation)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

# Optional Spearman
try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =========================
# Utilities
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(tok.strip()) for tok in s.split(",") if tok.strip()]


def parse_lambda_grid(grid_str: str) -> List[float]:
    """
    Supports either:
      "a,b,c" explicit list
      or pattern "start,step,...,end" where step inferred from 2nd token:
         "-3,-2.5,...,3" -> start=-3, second=-2.5 => step=0.5 => up to end (inclusive if close)
    Also supports "start,...,end" (no step) -> defaults to 9 points linear.
    """
    s = grid_str.strip()
    if "..." not in s:
        vals = [float(tok.strip()) for tok in s.split(",") if tok.strip()]
        if not vals:
            raise ValueError("Empty --lambda-grid.")
        return vals

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if "..." not in parts:
        raise ValueError("Invalid ... pattern in --lambda-grid.")

    i_ellipsis = parts.index("...")
    if i_ellipsis < 1 or i_ellipsis >= len(parts) - 1:
        raise ValueError("Use pattern like start,step,...,end or start,...,end")

    start = float(parts[0])
    end = float(parts[-1])

    if i_ellipsis >= 2:
        second = float(parts[1])
        step = second - start
        if step == 0:
            raise ValueError("Step inferred from first two tokens is 0.")
        # Generate inclusive-ish sequence
        vals = []
        x = start
        # Determine direction
        if step > 0:
            while x <= end + 1e-12:
                vals.append(float(x))
                x += step
        else:
            while x >= end - 1e-12:
                vals.append(float(x))
                x += step
        # Snap last if close
        if abs(vals[-1] - end) > 1e-9:
            vals.append(float(end))
        return vals

    # start,...,end only
    steps = 9
    vals = np.linspace(start, end, steps).tolist()
    return [float(v) for v in vals]


def build_lambda_grid_from_minmax(lmin: float, lmax: float, steps: int) -> List[float]:
    if steps < 2:
        return [float(lmin)]
    vals = np.linspace(lmin, lmax, steps).tolist()
    return [float(v) for v in vals]


def symmetrize(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def upper_tri_flat(W: np.ndarray) -> np.ndarray:
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    return W[iu].astype(np.float64, copy=False)


def topk_edge_set(W: np.ndarray, topk_frac: float) -> set:
    """
    Top-k edges by weight among all possible undirected edges (upper triangle).
    Returns a set of integer indices into the flattened upper triangle for easy overlap.
    """
    v = upper_tri_flat(W)
    m = v.size
    k = max(1, int(round(topk_frac * m)))
    # If there are NaNs, treat them as -inf
    v2 = np.where(np.isfinite(v), v, -np.inf)
    idx = np.argpartition(v2, -k)[-k:]
    return set(int(i) for i in idx.tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    uni = len(a.union(b))
    return inter / uni if uni else 0.0


def weighted_degree(W: np.ndarray) -> np.ndarray:
    return np.sum(W, axis=1)


def coeff_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    if not np.isfinite(mu) or mu == 0:
        return float("nan")
    sig = float(np.std(x))
    return sig / abs(mu)


def gini_coefficient(x: np.ndarray) -> float:
    """
    Gini of nonnegative array.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = np.maximum(x, 0.0)
    s = np.sum(x)
    if s <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    # Gini = (n+1 - 2*sum_i cumx_i / cumx_n) / n
    g = (n + 1.0 - 2.0 * np.sum(cumx) / cumx[-1]) / n
    return float(g)


def effective_rank_from_eigs(W: np.ndarray, eps: float = 1e-12) -> float:
    """
    Effective rank = exp(H), H = entropy of normalized eigenvalues.
    Uses absolute eigenvalues (since W may not be PSD).
    """
    W = np.asarray(W, dtype=np.float64)
    # Symmetric eigs
    vals = np.linalg.eigvalsh(W)
    vals = np.abs(vals)
    vals = vals[np.isfinite(vals)]
    total = float(np.sum(vals))
    if total <= eps:
        return 0.0
    p = vals / total
    p = np.maximum(p, eps)
    H = -float(np.sum(p * np.log(p)))
    return float(np.exp(H))


def frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    D = A - B
    return float(np.linalg.norm(D, ord="fro"))


def avg_pairwise_frobenius(mats: List[np.ndarray]) -> float:
    if len(mats) < 2:
        return 0.0
    s = 0.0
    c = 0
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            s += frobenius_distance(mats[i], mats[j])
            c += 1
    return float(s / c) if c else 0.0


def spearman_corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman on flattened upper triangle. Requires SciPy. If absent, returns NaN.
    """
    if not HAVE_SCIPY:
        return float("nan")
    r, _p = spearmanr(a, b)
    return safe_float(r)


# =========================
# Volume growth dimension
# =========================

def volume_growth_dimension(
    W: np.ndarray,
    sources: Optional[List[int]] = None,
    eps: float = 1e-8,
    n_radii: int = 16,
    quantile_lo: float = 0.15,
    quantile_hi: float = 0.85,
) -> Tuple[float, float]:
    """
    Volume growth estimate:
      - cost c_ij = 1/(W_ij + eps)
      - compute shortest-path distances from several sources
      - N(r) = number of nodes within distance <= r
      - fit slope of log N(r) vs log r over mid-range radii

    Returns mean slope and std over sources.
    """
    W = symmetrize(W)
    n = W.shape[0]
    if sources is None:
        # deterministic-ish: pick up to 5 sources spread out
        k = min(5, n)
        sources = np.linspace(0, n - 1, k).round().astype(int).tolist()

    # Build cost matrix (inf on missing edges)
    C = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(C, 0.0)
    mask = W > 0
    C[mask] = 1.0 / (W[mask] + eps)

    # Floyd-Warshall is fine for n<=14; keep simple and deterministic
    dist = C.copy()
    for k in range(n):
        # dist = min(dist, dist[:,k]+dist[k,:])
        dk = dist[:, [k]] + dist[[k], :]
        dist = np.minimum(dist, dk)

    slopes = []
    for s in sources:
        d = dist[s, :]
        d = d[np.isfinite(d)]
        d = d[d > 0]
        if d.size < 3:
            continue

        # choose radii from quantiles to avoid extremes
        r_lo = float(np.quantile(d, quantile_lo))
        r_hi = float(np.quantile(d, quantile_hi))
        if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
            continue

        radii = np.geomspace(max(r_lo, 1e-12), r_hi, n_radii)
        Ns = np.array([np.sum(dist[s, :] <= r) for r in radii], dtype=np.float64)

        # need mid-range where N between 2 and n-1
        mask_mid = (Ns >= 2) & (Ns <= (n - 1))
        if np.sum(mask_mid) < 4:
            continue

        x = np.log(radii[mask_mid])
        y = np.log(Ns[mask_mid])

        # linear fit slope
        A = np.vstack([x, np.ones_like(x)]).T
        slope, _b = np.linalg.lstsq(A, y, rcond=None)[0]
        if np.isfinite(slope):
            slopes.append(float(slope))

    if len(slopes) == 0:
        return float("nan"), float("nan")
    return float(np.mean(slopes)), float(np.std(slopes))


# =========================
# Optimizer adapter
# =========================

@dataclass
class OptimizerAPI:
    module_name: str
    build_fn_name: str
    optimize_fn_name: str
    module: Any
    build_fn: Any
    optimize_fn: Any


def load_optimizer_api(module_name: str, build_fn: str, optimize_fn: str) -> OptimizerAPI:
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to import optimizer module '{module_name}'.\n"
            f"Add it to PYTHONPATH or run from repo root.\n"
            f"Error: {e}"
        )

    if not hasattr(mod, build_fn):
        raise RuntimeError(
            f"Optimizer module '{module_name}' missing required function '{build_fn}'.\n"
            f"Expected: def {build_fn}(*, n:int, seed:int, alpha:float, beta:float) -> problem"
        )
    if not hasattr(mod, optimize_fn):
        raise RuntimeError(
            f"Optimizer module '{module_name}' missing required function '{optimize_fn}'.\n"
            f"Expected: def {optimize_fn}(problem, *, restarts:int, iters:int, init_W:np.ndarray|None, seed:int) -> dict"
        )

    return OptimizerAPI(
        module_name=module_name,
        build_fn_name=build_fn,
        optimize_fn_name=optimize_fn,
        module=mod,
        build_fn=getattr(mod, build_fn),
        optimize_fn=getattr(mod, optimize_fn),
    )


# =========================
# Metrics bundle
# =========================

def compute_metrics_bundle(
    W_best: np.ndarray,
    *,
    W_prev: Optional[np.ndarray],
    topk_frac: float,
    W_restarts: Optional[List[np.ndarray]] = None,
    obj_restarts: Optional[List[float]] = None,
    W_other_direction_same_lambda: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    W_best = symmetrize(W_best)
    n = W_best.shape[0]
    m = n * (n - 1) // 2

    v_best = upper_tri_flat(W_best)

    # (1) Topology-change metrics vs previous lambda
    if W_prev is not None:
        W_prev = symmetrize(W_prev)
        v_prev = upper_tri_flat(W_prev)
        rank_corr = spearman_corr_safe(v_best, v_prev)
        j_top = jaccard(topk_edge_set(W_best, topk_frac), topk_edge_set(W_prev, topk_frac))
    else:
        rank_corr = float("nan")
        j_top = float("nan")

    # (2) Equal-weight structure diagnostics (still computed everywhere; you can "read at lambda≈0")
    deg = weighted_degree(W_best)
    degree_cv = coeff_var(deg)
    weight_gini = gini_coefficient(v_best)
    eff_rank = effective_rank_from_eigs(W_best)

    # (3) Phase transition indicators
    basin_spread = float("nan")
    basin_obj_std = float("nan")
    if obj_restarts is not None and len(obj_restarts) >= 2:
        basin_obj_std = float(np.std(np.asarray(obj_restarts, dtype=np.float64)))
    if W_restarts is not None and len(W_restarts) >= 2:
        Ws = [symmetrize(W) for W in W_restarts]
        basin_spread = avg_pairwise_frobenius(Ws)

    hysteresis_gap_frob = float("nan")
    hysteresis_gap_topk = float("nan")
    if W_other_direction_same_lambda is not None:
        W_od = symmetrize(W_other_direction_same_lambda)
        hysteresis_gap_frob = frobenius_distance(W_best, W_od)
        hysteresis_gap_topk = jaccard(topk_edge_set(W_best, topk_frac), topk_edge_set(W_od, topk_frac))

    # (4) Effective dimension estimate (volume growth)
    dim_mean, dim_std = volume_growth_dimension(W_best)

    return {
        # Topology-change
        "rank_corr_to_prev": safe_float(rank_corr),
        "jaccard_topk_to_prev": safe_float(j_top),

        # Structure
        "degree_cv": safe_float(degree_cv),
        "weight_gini": safe_float(weight_gini),
        "effective_rank": safe_float(eff_rank),

        # Transition indicators
        "basin_spread": safe_float(basin_spread),
        "basin_obj_std": safe_float(basin_obj_std),
        "hysteresis_gap_frob": safe_float(hysteresis_gap_frob),
        "hysteresis_gap_topk": safe_float(hysteresis_gap_topk),

        # Dimension
        "dim_vgrowth_mean": safe_float(dim_mean),
        "dim_vgrowth_std": safe_float(dim_std),

        # Meta
        "n_edges_possible": int(m),
    }


# =========================
# IO layout
# =========================

def point_dir(outdir: str, seed: int, direction: str, lam: float) -> str:
    # Stable formatting so resume works (avoid floating issues)
    lam_tag = f"{lam:+.6f}".replace("+", "p").replace("-", "m").replace(".", "d")
    return os.path.join(outdir, f"seed_{seed}", f"dir_{direction}", f"lambda_{lam_tag}")


def save_point(out_point_dir: str, W_best: np.ndarray, metrics: Dict[str, Any]) -> None:
    ensure_dir(out_point_dir)
    np.save(os.path.join(out_point_dir, "W_best.npy"), symmetrize(W_best))
    with open(os.path.join(out_point_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def load_point_metrics(out_point_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(out_point_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_point_W(out_point_dir: str) -> Optional[np.ndarray]:
    path = os.path.join(out_point_dir, "W_best.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


# =========================
# Summary outputs
# =========================

SUMMARY_FIELDS = [
    "timestamp",
    "n",
    "seed",
    "direction",
    "lambda",
    "alpha",
    "beta",
    "alpha_base",
    "restarts",
    "iters",
    "warm_start",
    "topk_frac",
    "obj_best",
    "time_sec",

    # metrics bundle
    "rank_corr_to_prev",
    "jaccard_topk_to_prev",
    "degree_cv",
    "weight_gini",
    "effective_rank",
    "basin_spread",
    "basin_obj_std",
    "hysteresis_gap_frob",
    "hysteresis_gap_topk",
    "dim_vgrowth_mean",
    "dim_vgrowth_std",
]


def write_summary_csv(rows: List[Dict[str, Any]], out_csv_path: str) -> None:
    ensure_dir(os.path.dirname(out_csv_path))
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})


def try_make_summary_plots(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not HAVE_PLT:
        return
    if not rows:
        return

    # Plot only for direction "up" by default if multiple directions exist
    # but we include both in separate panels if present.
    dirs = sorted(set(r["direction"] for r in rows))
    seeds = sorted(set(int(r["seed"]) for r in rows))

    def group(dir_name: str):
        sub = [r for r in rows if r["direction"] == dir_name]
        # aggregate across seeds: mean +/- std at each lambda
        lams = sorted(set(float(r["lambda"]) for r in sub))
        out = {lam: [] for lam in lams}
        for r in sub:
            out[float(r["lambda"])].append(r)
        return lams, out

    metrics_to_plot = [
        ("rank_corr_to_prev", "Rank corr to prev"),
        ("jaccard_topk_to_prev", "Top-k Jaccard to prev"),
        ("weight_gini", "Weight Gini"),
        ("degree_cv", "Degree CV"),
        ("dim_vgrowth_mean", "Dim (vgrowth)"),
    ]

    nrows = len(metrics_to_plot)
    ncols = len(dirs)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 2.6 * nrows), squeeze=False)

    for c, dname in enumerate(dirs):
        lams, bylam = group(dname)
        for r_i, (key, title) in enumerate(metrics_to_plot):
            ax = axes[r_i][c]
            means = []
            stds = []
            for lam in lams:
                vals = []
                for rr in bylam[lam]:
                    v = rr.get(key, None)
                    if v is None:
                        continue
                    v = safe_float(v)
                    if np.isfinite(v):
                        vals.append(v)
                if len(vals) == 0:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
            ax.errorbar(lams, means, yerr=stds, fmt="o-", linewidth=2, markersize=4)
            ax.set_title(f"{title} ({dname})")
            ax.set_xlabel("lambda")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# =========================
# Main sweep logic
# =========================

def run_point(
    api: OptimizerAPI,
    *,
    n: int,
    seed: int,
    direction: str,
    lam: float,
    alpha_base: float,
    restarts: int,
    iters: int,
    warm_start: bool,
    init_W: Optional[np.ndarray],
) -> Dict[str, Any]:
    alpha = float(alpha_base * math.exp(lam))
    beta = float(alpha_base)

    problem = api.build_fn(n=n, seed=seed, alpha=alpha, beta=beta)

    # Optimize
    t0 = time.time()
    out = api.optimize_fn(problem, restarts=restarts, iters=iters, init_W=init_W, seed=seed)
    t1 = time.time()

    if not isinstance(out, dict):
        raise RuntimeError("optimize_links must return a dict.")

    if "W_best" not in out:
        raise RuntimeError("optimize_links output missing 'W_best'.")

    W_best = symmetrize(np.asarray(out["W_best"], dtype=np.float64))
    obj_best = safe_float(out.get("obj_best", np.nan))
    obj_components = out.get("obj_components", None)

    W_restarts = out.get("W_restarts", None)
    obj_restarts = out.get("obj_restarts", None)

    # Normalize optional restarts lists
    if isinstance(W_restarts, list):
        W_restarts = [symmetrize(np.asarray(W, dtype=np.float64)) for W in W_restarts]
    else:
        W_restarts = None

    if isinstance(obj_restarts, list):
        obj_restarts = [safe_float(v) for v in obj_restarts]
    else:
        obj_restarts = None

    return {
        "W_best": W_best,
        "obj_best": obj_best,
        "obj_components": obj_components,
        "W_restarts": W_restarts,
        "obj_restarts": obj_restarts,
        "time_sec": float(t1 - t0),
        "alpha": alpha,
        "beta": beta,
    }


def main():
    p = argparse.ArgumentParser(
        description="Sweep-and-report driver for link phase sweep over lambda=log(alpha/beta).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required inputs per spec
    p.add_argument("--n", type=int, required=True, help="Number of nodes")
    p.add_argument("--lambda-grid", type=str, default=None,
                   help='Explicit grid like "-3,-2.5,...,3" or "-3,-2,-1,0,1"')
    p.add_argument("--lambda-min", type=float, default=None)
    p.add_argument("--lambda-max", type=float, default=None)
    p.add_argument("--lambda-steps", type=int, default=None)
    p.add_argument("--alpha-base", type=float, required=True, help="A0 where alpha=A0*exp(lambda), beta=A0")
    p.add_argument("--seeds", type=str, required=True, help='Comma-separated seeds like "0,1,2,3"')
    p.add_argument("--restarts", type=int, required=True, help="Optimizer restarts per seed per lambda")
    p.add_argument("--iters", type=int, required=True, help="Max iterations per restart")
    p.add_argument("--topk-frac", type=float, required=True, help="Top-k fraction for overlap metrics (e.g., 0.2)")

    # Optional but recommended
    p.add_argument("--warm-start", type=str, default="off", choices=["on", "off"])
    p.add_argument("--sweep-direction", type=str, default="both", choices=["up", "down", "both"])
    p.add_argument("--out", type=str, default="link_phase_sweep", help="Output directory")

    # Runtime discipline
    p.add_argument("--force", action="store_true", help="Recompute even if point already exists")
    p.add_argument("--no-plots", action="store_true", help="Skip summary plots generation")

    # Adapter options (optional)
    p.add_argument("--optimizer-module", type=str, default="link_optimizer",
                   help="Module providing build_problem + optimize_links")
    p.add_argument("--build-fn", type=str, default="build_problem")
    p.add_argument("--optimize-fn", type=str, default="optimize_links")

    args = p.parse_args()

    n = int(args.n)
    alpha_base = float(args.alpha_base)
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("No seeds parsed from --seeds.")

    restarts = int(args.restarts)
    iters = int(args.iters)
    topk_frac = float(args.topk_frac)
    warm_start = (args.warm_start == "on")
    outdir = args.out

    # Lambda grid
    if args.lambda_grid is not None:
        lambdas = parse_lambda_grid(args.lambda_grid)
    else:
        if args.lambda_min is None or args.lambda_max is None or args.lambda_steps is None:
            raise SystemExit("Provide either --lambda-grid or (--lambda-min --lambda-max --lambda-steps).")
        lambdas = build_lambda_grid_from_minmax(float(args.lambda_min), float(args.lambda_max), int(args.lambda_steps))

    # Sweep directions list
    if args.sweep_direction == "both":
        directions = ["up", "down"]
    else:
        directions = [args.sweep_direction]

    # Load optimizer API
    api = load_optimizer_api(args.optimizer_module, args.build_fn, args.optimize_fn)

    ensure_dir(outdir)

    # For hysteresis, we need to compare up/down at same lambda once both exist.
    # We'll record per-point outputs and do a second pass to fill hysteresis metrics if both present.
    summary_rows: List[Dict[str, Any]] = []

    # Sweep
    for seed in seeds:
        for direction in directions:
            lam_seq = sorted(lambdas) if direction == "up" else sorted(lambdas, reverse=True)

            prev_W_best: Optional[np.ndarray] = None
            for lam in lam_seq:
                pt_dir = point_dir(outdir, seed, direction, lam)
                metrics_path = os.path.join(pt_dir, "metrics.json")

                if (not args.force) and os.path.exists(metrics_path):
                    # Load cached
                    metrics = load_point_metrics(pt_dir) or {}
                    # Also load W_best for warm-start continuity / next-point metrics
                    prev_W_best = load_point_W(pt_dir) or prev_W_best
                    # Ensure summary row exists from cached metrics
                    if metrics:
                        summary_rows.append(metrics.get("summary_row", metrics))
                    continue

                init_W = prev_W_best if warm_start else None

                # Run point
                print(f"\n[{now_iso()}] seed={seed} dir={direction} lambda={lam:+.4f} warm_start={warm_start}")
                try:
                    out = run_point(
                        api,
                        n=n,
                        seed=seed,
                        direction=direction,
                        lam=float(lam),
                        alpha_base=alpha_base,
                        restarts=restarts,
                        iters=iters,
                        warm_start=warm_start,
                        init_W=init_W,
                    )
                except Exception as e:
                    print(f"ERROR at seed={seed} dir={direction} lambda={lam}: {e}")
                    raise

                W_best = out["W_best"]
                obj_best = out["obj_best"]
                W_restarts = out.get("W_restarts", None)
                obj_restarts = out.get("obj_restarts", None)

                # Metrics vs previous lambda point (same seed+direction)
                bundle = compute_metrics_bundle(
                    W_best,
                    W_prev=prev_W_best,
                    topk_frac=topk_frac,
                    W_restarts=W_restarts,
                    obj_restarts=obj_restarts,
                    W_other_direction_same_lambda=None,  # filled later if both sweeps
                )

                # Build summary row with metadata + objective components
                row = {
                    "timestamp": now_iso(),
                    "n": n,
                    "seed": seed,
                    "direction": direction,
                    "lambda": float(lam),
                    "alpha": float(out["alpha"]),
                    "beta": float(out["beta"]),
                    "alpha_base": float(alpha_base),
                    "restarts": restarts,
                    "iters": iters,
                    "warm_start": int(warm_start),
                    "topk_frac": float(topk_frac),
                    "obj_best": float(obj_best),
                    "time_sec": float(out["time_sec"]),
                }
                row.update(bundle)

                # Attach objective components (optional) into metrics.json (not into summary.csv columns)
                metrics = {
                    "meta": {
                        "n": n,
                        "seed": seed,
                        "direction": direction,
                        "lambda": float(lam),
                        "alpha": float(out["alpha"]),
                        "beta": float(out["beta"]),
                        "alpha_base": float(alpha_base),
                        "restarts": restarts,
                        "iters": iters,
                        "warm_start": bool(warm_start),
                        "topk_frac": float(topk_frac),
                        "optimizer_module": api.module_name,
                        "build_fn": api.build_fn_name,
                        "optimize_fn": api.optimize_fn_name,
                        "timestamp": now_iso(),
                        "time_sec": float(out["time_sec"]),
                    },
                    "objective": {
                        "best": float(obj_best),
                        "components": out.get("obj_components", None),
                        "restarts": obj_restarts,
                    },
                    "metrics": bundle,
                    # convenience: also store a flat row for easy CSV rebuilds
                    "summary_row": row,
                }

                save_point(pt_dir, W_best, metrics)
                summary_rows.append(row)

                # Update prev for next lambda
                prev_W_best = W_best

    # Second pass: compute hysteresis gaps if both directions were requested
    if args.sweep_direction == "both":
        # Build lookup: (seed, lambda) -> (W_up, W_down)
        for seed in seeds:
            for lam in sorted(lambdas):
                pt_up = point_dir(outdir, seed, "up", lam)
                pt_dn = point_dir(outdir, seed, "down", lam)
                mu = load_point_metrics(pt_up)
                md = load_point_metrics(pt_dn)
                Wu = load_point_W(pt_up)
                Wd = load_point_W(pt_dn)
                if mu is None or md is None or Wu is None or Wd is None:
                    continue

                # Update each point's hysteresis metrics if missing or force
                for direction, pt_dir, m, W_this, W_other in [
                    ("up", pt_up, mu, Wu, Wd),
                    ("down", pt_dn, md, Wd, Wu),
                ]:
                    # Compute gaps
                    gaps = compute_metrics_bundle(
                        W_this,
                        W_prev=None,
                        topk_frac=topk_frac,
                        W_restarts=None,
                        obj_restarts=None,
                        W_other_direction_same_lambda=W_other,
                    )
                    # Only take hysteresis fields
                    m["metrics"]["hysteresis_gap_frob"] = gaps["hysteresis_gap_frob"]
                    m["metrics"]["hysteresis_gap_topk"] = gaps["hysteresis_gap_topk"]
                    # Update summary_row too
                    if "summary_row" in m:
                        m["summary_row"]["hysteresis_gap_frob"] = gaps["hysteresis_gap_frob"]
                        m["summary_row"]["hysteresis_gap_topk"] = gaps["hysteresis_gap_topk"]
                    # Save updated metrics.json (W unchanged)
                    with open(os.path.join(pt_dir, "metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(m, f, indent=2, sort_keys=True)

        # Rebuild summary_rows from disk to ensure hysteresis is reflected
        summary_rows = []
        for seed in seeds:
            for direction in ["up", "down"]:
                for lam in lambdas:
                    pt = point_dir(outdir, seed, direction, lam)
                    m = load_point_metrics(pt)
                    if m and "summary_row" in m:
                        summary_rows.append(m["summary_row"])

    # Sort rows for stable CSV
    summary_rows = sorted(summary_rows, key=lambda r: (int(r["seed"]), str(r["direction"]), float(r["lambda"])))

    # Write summary.csv
    summary_csv_path = os.path.join(outdir, "summary.csv")
    write_summary_csv(summary_rows, summary_csv_path)
    print(f"\nWrote: {summary_csv_path}")

    # Summary plots
    if (not args.no_plots):
        plots_path = os.path.join(outdir, "summary_plots.png")
        try_make_summary_plots(summary_rows, plots_path)
        if HAVE_PLT:
            print(f"Wrote: {plots_path}")
        else:
            print("Skipping plots (matplotlib not available).")

    print("\nDone.")


if __name__ == "__main__":
    main()
