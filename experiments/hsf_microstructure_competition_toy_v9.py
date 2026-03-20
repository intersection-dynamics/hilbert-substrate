# hsf_microstructure_competition_toy_v9.py
# ------------------------------------------------------------
# v9: Rare-phase / microstructure toy
#
# Goal: turn the old "branch fraction bias" into something that *looks like alloys*:
#   - Rare nucleation (good phase) vs common nucleation (bad phase)
#   - Runaway growth (autocatalysis / neighbor-assisted growth)
#   - Competition (bad is kinetically favored; good is harder but can be "selected")
#   - Metastability / loss (good can be eaten by bad or relax back to matrix unless supported)
#   - Spatial output (domain maps + domain size distributions)
#
# This is a SAFE, generic, abstract model: phases are just labels:
#   0 = matrix
#   1 = GOOD phase (rare, desirable microstructure)
#   2 = BAD phase (easy, fast, undesired microstructure)
#
# "HSF-clean" knobs:
#   - Positive drive helps the GOOD phase nucleate/grow (selection pressure)
#   - Negative drive adds bookkeeping load to links (edge memory), increasing constraint pressure
#   - Boundary shell (optional) modifies bandwidth/memory at region boundary
#
# Multicore features:
#   - Multi-seed pooling: --seeds 0:19 --n_jobs 12
#   - Optional baseline compare: baseline = no shell + no negative bookkeeping
#
# Sweep features:
#   - 1D or 2D sweeps with pooled stats per point (writes pooled_summary.csv)
#
# Outputs per run:
#   - pooled_summary.csv (always)
#   - summary.csv (per-seed stats)
#   - If --save_seed_artifacts: per-seed plots + snapshots
#
# Example (single run with maps):
#   python hsf_microstructure_competition_toy_v9.py --L 80 --T 1200 --seeds 0:9 --n_jobs 8 --save_seed_artifacts --compare_baseline
#
# Example (sweep neg_mem_gain):
#   python hsf_microstructure_competition_toy_v9.py --L 80 --T 1200 --seeds 0:19 --n_jobs 12 --compare_baseline --sweep_param neg_mem_gain --sweep_values 0,0.05,0.1,0.2
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import hashlib
from dataclasses import dataclass, replace
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# Force a non-interactive backend (important for Windows multiprocessing)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import multiprocessing as mp

TAU = 2.0 * np.pi


# ----------------------------
# Utilities
# ----------------------------

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_tag(s: str) -> str:
    return s.replace(" ", "_").replace(":", "_").replace("/", "_").replace(".", "p")


def short_hash(s: str, n: int = 10) -> str:
    """Short stable hash for path-shortening (Windows MAX_PATH safety)."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def make_run_dir(base_out: str, tag: str) -> str:
    # Windows can still hit MAX_PATH in deep directories; shorten aggressively.
    st = safe_tag(tag)
    h = short_hash(st, 10)
    st = st[:48]  # keep some readability
    run_dir = os.path.join(base_out, f"{now_stamp()}_{st}_{h}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "seed_runs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    return run_dir


def parse_seeds(spec: str) -> List[int]:
    s = spec.strip()
    if "," in s:
        out = []
        for part in s.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        return sorted(list(dict.fromkeys(out)))
    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) == 2:
            a, b = int(parts[0]), int(parts[1])
            step = 1
        elif len(parts) == 3:
            a, b, step = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"Bad --seeds spec: {spec}")
        if step <= 0:
            raise ValueError("--seeds step must be positive")
        if b < a:
            raise ValueError("--seeds end must be >= start")
        return list(range(a, b + 1, step))
    if s:
        return [int(s)]
    raise ValueError("Empty --seeds spec")


def parse_values_csv(spec: str) -> List[float]:
    s = spec.strip()
    if not s:
        return []
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


# ----------------------------
# Model parameters
# ----------------------------

@dataclass(frozen=True)
class Params:
    # lattice / time
    L: int = 80
    T: int = 1200
    snap_every: int = 150

    # region / shell
    region_radius_frac: float = 0.35
    shell_on: bool = True
    shell_width_frac: float = 0.03

    # positive drive (selection support)
    drive_F: float = 0.2
    drive_gain_nuc_good: float = 0.9
    drive_gain_nuc_bad: float = 0.25
    grow_drive_gain_good: float = 0.60
    grow_drive_gain_bad: float = 0.20

    # nucleation base rates + barriers
    nuc_attempt_good: float = 0.0008
    nuc_attempt_bad: float = 0.0035
    barrier_good: float = 2.2
    barrier_bad: float = 1.2

    # neighbor-assisted growth
    neighbor_gain_good: float = 0.35
    neighbor_gain_bad: float = 0.50

    # competition / decay
    eat_bad_over_good: float = 0.25
    relax_good_to_matrix: float = 0.0016

    # link memory / bandwidth bookkeeping (negative load)
    neg_mode: str = "mem_bw"  # off / mem / bw / mem_bw
    neg_mem_gain: float = 0.05
    neg_bw_gain: float = 0.00

    # constraint pressure
    bandwidth_base: float = 0.70
    mem_base: float = 1.35
    kappa: float = 0.65
    mem_couple: float = 0.45
    bw_couple: float = 0.35

    # RNG
    rng_kind: str = "pcg64"


# ----------------------------
# Region helpers
# ----------------------------

def build_region_mask(L: int, r_frac: float) -> np.ndarray:
    cx = (L - 1) / 2.0
    cy = (L - 1) / 2.0
    rr = (r_frac * L)
    rr2 = rr * rr
    y, x = np.ogrid[0:L, 0:L]
    dx = x - cx
    dy = y - cy
    mask = (dx * dx + dy * dy) <= rr2
    return mask.astype(np.int8)


def build_shell_mask(region: np.ndarray, width_frac: float) -> np.ndarray:
    L = region.shape[0]
    shell = np.zeros_like(region, dtype=np.int8)
    w = max(1, int(round(width_frac * L)))
    # shell: within a small band near region boundary (approx via dilation-erosion)
    # dilation:
    dil = region.copy()
    for _ in range(w):
        dil = dilate4(dil)
    ero = region.copy()
    for _ in range(w):
        ero = erode4(ero)
    shell[(dil == 1) & (ero == 0)] = 1
    return shell


def dilate4(mask: np.ndarray) -> np.ndarray:
    L = mask.shape[0]
    out = mask.copy()
    # 4-neighborhood dilation
    out |= np.roll(mask, 1, axis=0)
    out |= np.roll(mask, -1, axis=0)
    out |= np.roll(mask, 1, axis=1)
    out |= np.roll(mask, -1, axis=1)
    return out.astype(np.int8)


def erode4(mask: np.ndarray) -> np.ndarray:
    # erosion = keep cells whose 4-neighbors all 1
    m = mask.astype(bool)
    out = m.copy()
    out &= np.roll(m, 1, axis=0)
    out &= np.roll(m, -1, axis=0)
    out &= np.roll(m, 1, axis=1)
    out &= np.roll(m, -1, axis=1)
    return out.astype(np.int8)


# ----------------------------
# Domain size stats
# ----------------------------

def component_sizes(grid: np.ndarray, phase: int, region: Optional[np.ndarray] = None) -> List[int]:
    L = grid.shape[0]
    seen = np.zeros((L, L), dtype=np.uint8)
    sizes: List[int] = []

    if region is None:
        region = np.ones((L, L), dtype=np.int8)

    for y in range(L):
        for x in range(L):
            if region[y, x] == 0:
                continue
            if seen[y, x]:
                continue
            if grid[y, x] != phase:
                continue
            # BFS flood fill 4-neighborhood
            q = [(y, x)]
            seen[y, x] = 1
            cnt = 0
            while q:
                cy, cx = q.pop()
                cnt += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = (cy + dy) % L
                    nx = (cx + dx) % L
                    if region[ny, nx] == 0:
                        continue
                    if seen[ny, nx]:
                        continue
                    if grid[ny, nx] != phase:
                        continue
                    seen[ny, nx] = 1
                    q.append((ny, nx))
            sizes.append(cnt)

    return sizes


# ----------------------------
# Core dynamics
# ----------------------------

def rng_from_seed(seed: int, kind: str = "pcg64") -> np.random.Generator:
    if kind.lower() == "pcg64":
        bitgen = np.random.PCG64(seed)
    else:
        bitgen = np.random.PCG64(seed)
    return np.random.Generator(bitgen)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def logistic(x: float) -> float:
    # stable-ish logistic; for moderate x this is fine.
    return 1.0 / (1.0 + math.exp(-x))


def compute_drive_field(p: Params, region: np.ndarray, shell: np.ndarray) -> np.ndarray:
    # Base: inside region gets positive drive support; outside gets none.
    F = np.zeros_like(region, dtype=np.float32)
    F[region == 1] = p.drive_F
    # Shell can optionally weaken or alter effective bandwidth/memory; we keep drive same here.
    return F


def compute_capacity_fields(p: Params, region: np.ndarray, shell: np.ndarray, shell_on: bool) -> Tuple[np.ndarray, np.ndarray]:
    # Effective local bandwidth and memory capacity, optionally modified on shell.
    bw = np.zeros_like(region, dtype=np.float32)
    mem = np.zeros_like(region, dtype=np.float32)

    bw[:, :] = p.bandwidth_base
    mem[:, :] = p.mem_base

    if shell_on:
        # Shell slightly reduces bw/mem (acts like a bottleneck boundary)
        bw[shell == 1] *= 0.92
        mem[shell == 1] *= 0.92

    # Outside region: tighter capacities
    bw[region == 0] *= 0.85
    mem[region == 0] *= 0.85
    return bw, mem


def neighbor_counts(grid: np.ndarray, phase: int) -> np.ndarray:
    g = (grid == phase).astype(np.int8)
    # 4-neighbor count
    c = np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0) + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1)
    return c.astype(np.int8)


def run_one_seed(p: Params, seed: int, shell: bool = True) -> Dict[str, Any]:
    rng = rng_from_seed(seed, p.rng_kind)

    L = p.L
    T = p.T

    region = build_region_mask(L, p.region_radius_frac)
    shell_mask = build_shell_mask(region, p.shell_width_frac) if p.shell_on else np.zeros_like(region, dtype=np.int8)

    driveF = compute_drive_field(p, region, shell_mask)
    bw_cap, mem_cap = compute_capacity_fields(p, region, shell_mask, shell_on=(shell and p.shell_on))

    # grid states: 0=matrix, 1=good, 2=bad
    grid = np.zeros((L, L), dtype=np.int8)

    # "edge memory" proxy and "phase change" proxy
    edge_mem = np.zeros((L, L), dtype=np.float32)
    dtheta = np.zeros((L, L), dtype=np.float32)

    frac_good = np.zeros(T, dtype=np.float32)
    frac_bad = np.zeros(T, dtype=np.float32)
    mean_mem = np.zeros(T, dtype=np.float32)
    mean_dtheta = np.zeros(T, dtype=np.float32)
    locked_like = np.zeros(T, dtype=np.float32)

    first_good_t = -1
    first_bad_t = -1

    snaps: List[np.ndarray] = []
    snap_every = max(1, int(p.snap_every))

    for t in range(T):
        # neighbor influence
        ng = neighbor_counts(grid, 1).astype(np.float32)
        nb = neighbor_counts(grid, 2).astype(np.float32)

        # local "support" fields
        # Positive drive helps GOOD more than BAD.
        drive_good = p.drive_gain_nuc_good * driveF + p.grow_drive_gain_good * driveF
        drive_bad = p.drive_gain_nuc_bad * driveF + p.grow_drive_gain_bad * driveF

        # constraint pressure from capacity vs used load
        # We treat edge_mem as "used memory" proxy. Add negative bookkeeping load when enabled.
        neg_load_mem = 0.0
        neg_load_bw = 0.0
        if p.neg_mode in ("mem", "mem_bw"):
            neg_load_mem = p.neg_mem_gain
        if p.neg_mode in ("bw", "mem_bw"):
            neg_load_bw = p.neg_bw_gain

        # Increase effective used load where activity is high (neighbor counts) and where phase boundaries exist.
        # This is the "finite bandwidth" + "no-refolding-ish" pressure: microstructure changes cost.
        activity = (ng + nb) / 4.0
        boundary = ((ng + nb) > 0).astype(np.float32) * 0.5

        used_mem = edge_mem + neg_load_mem * (0.25 + activity + boundary)
        used_bw = (0.15 + activity + boundary) + neg_load_bw * (0.25 + activity)

        # compute "pressure" as overuse fraction
        mem_over = np.maximum(0.0, used_mem - mem_cap) / (mem_cap + 1e-6)
        bw_over = np.maximum(0.0, used_bw - bw_cap) / (bw_cap + 1e-6)
        pressure = p.kappa * (p.mem_couple * mem_over + p.bw_couple * bw_over)

        # ----------------
        # Nucleation attempts (rare GOOD vs common BAD)
        # ----------------
        # Attempt rates reduced by pressure. Barrier reduced by drive and neighbor support.
        # More neighbors -> lower barrier -> autocatalysis.
        empty = (grid == 0)

        # GOOD nucleation
        barrier_eff_good = p.barrier_good - drive_good - p.neighbor_gain_good * (ng / 4.0)
        rate_good = p.nuc_attempt_good * np.exp(-barrier_eff_good)
        rate_good *= np.exp(-pressure)
        # BAD nucleation
        barrier_eff_bad = p.barrier_bad - drive_bad - p.neighbor_gain_bad * (nb / 4.0)
        rate_bad = p.nuc_attempt_bad * np.exp(-barrier_eff_bad)
        rate_bad *= np.exp(-pressure)

        # stochastic nucleation on empty sites
        u = rng.random((L, L), dtype=np.float32)
        new_good = empty & (u < rate_good)
        u2 = rng.random((L, L), dtype=np.float32)
        new_bad = empty & (u2 < rate_bad)

        # If both trigger, BAD wins by default unless drive is strong:
        both = new_good & new_bad
        if np.any(both):
            # probability GOOD survives tie rises with driveF
            tie_u = rng.random((L, L), dtype=np.float32)
            keep_good = tie_u < clamp01(float(np.mean(driveF[both])) * 3.0) if np.any(both) else 0.0
            # vectorize:
            # default: both -> BAD; but where tie_u < f(driveF) -> GOOD
            f = np.clip(3.0 * driveF, 0.0, 1.0)
            good_wins = both & (tie_u < f)
            bad_wins = both & (~good_wins)
            new_good[both] = False
            new_bad[both] = False
            new_good[good_wins] = True
            new_bad[bad_wins] = True

        # apply nucleation
        if first_good_t < 0 and np.any(new_good):
            first_good_t = t
        if first_bad_t < 0 and np.any(new_bad):
            first_bad_t = t
        grid[new_good] = 1
        grid[new_bad] = 2

        # ----------------
        # Growth: empty sites convert based on neighbor counts and drive support
        # ----------------
        empty = (grid == 0)
        # Growth propensity
        g_prop = (ng / 4.0) + 0.55 * driveF
        b_prop = (nb / 4.0) + 0.20 * driveF
        # Pressure suppresses growth too
        g_rate = 0.15 * g_prop * np.exp(-pressure)
        b_rate = 0.18 * b_prop * np.exp(-pressure)

        ug = rng.random((L, L), dtype=np.float32)
        ub = rng.random((L, L), dtype=np.float32)
        grow_good = empty & (ng > 0) & (ug < g_rate)
        grow_bad = empty & (nb > 0) & (ub < b_rate)

        # resolve collision: if both want to grow into empty, BAD slightly favored unless drive is large
        both2 = grow_good & grow_bad
        if np.any(both2):
            tie_u = rng.random((L, L), dtype=np.float32)
            f = np.clip(0.35 + 1.75 * driveF, 0.0, 1.0)
            good_wins = both2 & (tie_u < f)
            bad_wins = both2 & (~good_wins)
            grow_good[both2] = False
            grow_bad[both2] = False
            grow_good[good_wins] = True
            grow_bad[bad_wins] = True

        grid[grow_good] = 1
        grid[grow_bad] = 2

        # ----------------
        # Competition: BAD eats GOOD at boundaries (kinetic advantage)
        # ----------------
        # If a GOOD cell has BAD neighbor(s), it may flip to BAD with prob depending on pressure and drive.
        is_good = (grid == 1)
        bad_adj = (neighbor_counts(grid, 2) > 0)
        eat_mask = is_good & bad_adj
        if np.any(eat_mask):
            # BAD eats more when pressure is high; drive protects GOOD.
            eat_p = p.eat_bad_over_good * (1.0 + 0.6 * pressure) * np.exp(-1.2 * driveF)
            ue = rng.random((L, L), dtype=np.float32)
            eaten = eat_mask & (ue < eat_p)
            grid[eaten] = 2

        # ----------------
        # Metastability / loss: unsupported GOOD relaxes back to matrix
        # ----------------
        is_good = (grid == 1)
        good_support = (neighbor_counts(grid, 1).astype(np.float32) / 4.0) + 0.8 * driveF
        weak = is_good & (good_support < 0.35)
        if np.any(weak):
            # decay higher outside region and under pressure
            decay = p.relax_good_to_matrix * (1.0 + 1.2 * pressure)
            decay *= (1.35 - 0.9 * driveF)
            decay = np.clip(decay, 0.0, 0.25)
            ud = rng.random((L, L), dtype=np.float32)
            decayed = weak & (ud < decay)
            grid[decayed] = 0

        # ----------------
        # Update bookkeeping proxies
        # ----------------
        # edge_mem rises with phase changes and boundary complexity; relaxes slowly.
        # dtheta measures "local churn" proxy.
        phase_is = grid.astype(np.float32)
        local_mix = (neighbor_counts(grid, 1) + neighbor_counts(grid, 2)).astype(np.float32) / 4.0
        churn = local_mix + activity
        dtheta = 0.85 * dtheta + 0.15 * churn

        # edge memory integrates negative bookkeeping and boundary complexity; relaxes with capacity.
        edge_mem = 0.94 * edge_mem + 0.06 * (0.35 * churn + 0.55 * (pressure > 0.0).astype(np.float32))
        if p.neg_mode in ("mem", "mem_bw"):
            edge_mem += 0.02 * p.neg_mem_gain

        # stats
        frac_good[t] = float(np.mean(grid == 1))
        frac_bad[t] = float(np.mean(grid == 2))
        mean_mem[t] = float(np.mean(edge_mem))
        mean_dtheta[t] = float(np.mean(dtheta))
        locked_like[t] = float(np.mean(pressure > 0.0))

        # snapshots
        if (t % snap_every) == 0 or (t == T - 1):
            snaps.append(grid.copy())

    good_sizes = component_sizes(grid, 1, region=region)
    bad_sizes = component_sizes(grid, 2, region=region)

    out = dict(
        seed=seed,
        frac_good=frac_good,
        frac_bad=frac_bad,
        mean_mem=mean_mem,
        mean_dtheta=mean_dtheta,
        locked_like=locked_like,
        first_good_t=first_good_t if first_good_t >= 0 else int(1e9),
        first_bad_t=first_bad_t if first_bad_t >= 0 else int(1e9),
        final_frac_good=float(frac_good[-1]),
        final_frac_bad=float(frac_bad[-1]),
        good_sizes=np.array(good_sizes, dtype=np.int32),
        bad_sizes=np.array(bad_sizes, dtype=np.int32),
        final_grid=grid,
        region=region,
        snapshots=np.array(snaps, dtype=np.int8) if snaps else np.zeros((0, p.L, p.L), dtype=np.int8),
    )
    return out


# ----------------------------
# Plotting
# ----------------------------

def plot_seed_artifacts(seed_dir: str, data: Dict[str, Any], p: Params, mode_name: str) -> None:
    os.makedirs(seed_dir, exist_ok=True)
    plots = os.path.join(seed_dir, "plots")
    os.makedirs(plots, exist_ok=True)

    # time series
    t = np.arange(p.T)
    plt.figure()
    plt.plot(t, data["frac_good"], label="frac_good")
    plt.plot(t, data["frac_bad"], label="frac_bad")
    plt.xlabel("tick")
    plt.ylabel("fraction")
    plt.legend()
    plt.savefig(os.path.join(plots, f"{mode_name}_fractions.png"), dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t, data["mean_mem"], label="mean_edge_mem")
    plt.plot(t, data["locked_like"], label="locked_like")
    plt.plot(t, data["mean_dtheta"], label="mean|dtheta|")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots, f"{mode_name}_link_dynamics.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # final grid map
    grid = data["final_grid"]
    plt.figure()
    plt.imshow(grid, interpolation="nearest")
    plt.title(f"{mode_name} final grid (0=matrix,1=good,2=bad)")
    plt.axis("off")
    plt.savefig(os.path.join(plots, f"{mode_name}_final_grid.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # region mask
    plt.figure()
    plt.imshow(data["region"], interpolation="nearest")
    plt.title("tuned region mask")
    plt.axis("off")
    plt.savefig(os.path.join(plots, f"{mode_name}_region.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # snapshot montage (up to 9)
    snaps = data["snapshots"]
    if snaps.shape[0] > 0:
        k = min(9, snaps.shape[0])
        idx = np.linspace(0, snaps.shape[0] - 1, k).astype(int)
        plt.figure(figsize=(9, 9))
        for i, j in enumerate(idx):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(snaps[j], interpolation="nearest")
            ax.set_title(f"t~{j * p.snap_every}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plots, f"{mode_name}_snapshots.png"), dpi=220, bbox_inches="tight")
        plt.close()

    # domain size histograms
    gs = data["good_sizes"]
    bs = data["bad_sizes"]
    plt.figure()
    if gs.size > 0:
        plt.hist(gs, bins=20, alpha=0.7, label="good sizes")
    if bs.size > 0:
        plt.hist(bs, bins=20, alpha=0.7, label="bad sizes")
    plt.xlabel("domain size (cells)")
    plt.ylabel("count")
    plt.legend()
    plt.savefig(os.path.join(plots, f"{mode_name}_domain_sizes.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------
# Aggregation + CSV
# ----------------------------

@dataclass
class SeedRow:
    sweep_k: str
    sweep_v: str
    seed: int
    mode: str
    final_frac_good: float
    final_frac_bad: float
    first_good_t: int
    first_bad_t: int
    mean_good_domain: float
    mean_bad_domain: float
    n_good_domains: int
    n_bad_domains: int
    out_dir: str


def pooled_row(rows: List[SeedRow], mode: str, sweep_k: str, sweep_v: str) -> Dict[str, Any]:
    sub = [r for r in rows if r.mode == mode and r.sweep_k == sweep_k and r.sweep_v == sweep_v]
    if not sub:
        return {}

    fg = np.array([r.final_frac_good for r in sub], dtype=np.float64)
    fb = np.array([r.final_frac_bad for r in sub], dtype=np.float64)
    tG = np.array([r.first_good_t for r in sub], dtype=np.float64)
    tB = np.array([r.first_bad_t for r in sub], dtype=np.float64)
    mg = np.array([r.mean_good_domain for r in sub], dtype=np.float64)
    mb = np.array([r.mean_bad_domain for r in sub], dtype=np.float64)

    delta = float(np.mean(fg - fb))
    se = float(np.std(fg - fb, ddof=1) / math.sqrt(len(sub))) if len(sub) > 1 else 0.0
    z = float(delta / se) if se > 0 else 0.0

    return dict(
        sweep_k=sweep_k,
        sweep_v=sweep_v,
        mode=mode,
        n=len(sub),
        final_good_mean=float(np.mean(fg)),
        final_good_std=float(np.std(fg, ddof=1)) if len(sub) > 1 else 0.0,
        final_bad_mean=float(np.mean(fb)),
        final_bad_std=float(np.std(fb, ddof=1)) if len(sub) > 1 else 0.0,
        first_good_med=float(np.median(tG)),
        first_bad_med=float(np.median(tB)),
        mean_good_domain=float(np.nanmean(mg)),
        mean_bad_domain=float(np.nanmean(mb)),
        delta_good_minus_bad=delta,
        delta_se=se,
        delta_z=z,
    )


def apply_sweep(p: Params, name: str, value: float) -> Params:
    if not hasattr(p, name):
        raise ValueError(f"Unknown sweep param: {name}")
    return replace(p, **{name: value})


def _worker(args: Tuple[int, str, Params, bool, str, str, bool, str, str]) -> SeedRow:
    seed, mode, p, shell, run_root, point_tag, save_artifacts, sweep_k, sweep_v = args
    data = run_one_seed(p, seed, shell=shell)

    # domain stats
    gs = data["good_sizes"]
    bs = data["bad_sizes"]
    mean_g = float(np.mean(gs)) if gs.size > 0 else float("nan")
    mean_b = float(np.mean(bs)) if bs.size > 0 else float("nan")

    out_dir = ""
    if save_artifacts:
        seed_dir = os.path.join(run_root, point_tag, mode, f"seed_{seed:05d}")
        os.makedirs(seed_dir, exist_ok=True)
        # save npz
        np.savez_compressed(
            os.path.join(seed_dir, "results.npz"),
            seed=np.int32(seed),
            final_grid=data["final_grid"],
            region=data["region"],
            frac_good=data["frac_good"],
            frac_bad=data["frac_bad"],
            mean_mem=data["mean_mem"],
            mean_dtheta=data["mean_dtheta"],
            locked_like=data["locked_like"],
            first_good_t=np.int32(data["first_good_t"]),
            first_bad_t=np.int32(data["first_bad_t"]),
            good_sizes=data["good_sizes"],
            bad_sizes=data["bad_sizes"],
            snapshots=data["snapshots"],
        )
        plot_seed_artifacts(seed_dir, data, p, mode_name=mode)
        out_dir = seed_dir

    return SeedRow(
        sweep_k=sweep_k,
        sweep_v=sweep_v,
        seed=seed,
        mode=mode,
        final_frac_good=data["final_frac_good"],
        final_frac_bad=data["final_frac_bad"],
        first_good_t=int(data["first_good_t"]),
        first_bad_t=int(data["first_bad_t"]),
        mean_good_domain=mean_g,
        mean_bad_domain=mean_b,
        n_good_domains=int(data["good_sizes"].size),
        n_bad_domains=int(data["bad_sizes"].size),
        out_dir=out_dir,
    )


def write_seed_csv(path: str, rows: List[SeedRow]) -> None:
    cols = [
        "sweep_k", "sweep_v", "mode", "seed",
        "final_frac_good", "final_frac_bad",
        "first_good_t", "first_bad_t",
        "mean_good_domain", "mean_bad_domain",
        "n_good_domains", "n_bad_domains",
        "out_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: getattr(r, c) for c in cols})


def write_pooled_csv(path: str, pooled: List[Dict[str, Any]]) -> None:
    if not pooled:
        return
    # union of keys
    keys = []
    seen = set()
    for row in pooled:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in pooled:
            w.writerow(row)


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="HSF microstructure competition toy v9")

    ap.add_argument("--L", type=int, default=80)
    ap.add_argument("--T", type=int, default=1200)
    ap.add_argument("--snap_every", type=int, default=150)

    ap.add_argument("--region_radius_frac", type=float, default=0.35)
    ap.add_argument("--shell_on", action="store_true")
    ap.add_argument("--no_shell_on", action="store_true")
    ap.add_argument("--shell_width_frac", type=float, default=0.03)

    ap.add_argument("--drive_F", type=float, default=0.2)
    ap.add_argument("--nuc_attempt_good", type=float, default=0.0008)
    ap.add_argument("--nuc_attempt_bad", type=float, default=0.0035)
    ap.add_argument("--barrier_good", type=float, default=2.2)
    ap.add_argument("--barrier_bad", type=float, default=1.2)

    ap.add_argument("--neighbor_gain_good", type=float, default=0.35)
    ap.add_argument("--neighbor_gain_bad", type=float, default=0.50)

    ap.add_argument("--eat_bad_over_good", type=float, default=0.25)
    ap.add_argument("--relax_good_to_matrix", type=float, default=0.0016)

    ap.add_argument("--neg_mode", type=str, default="mem_bw", choices=["off", "mem", "bw", "mem_bw"])
    ap.add_argument("--neg_mem_gain", type=float, default=0.05)
    ap.add_argument("--neg_bw_gain", type=float, default=0.00)

    ap.add_argument("--bandwidth_base", type=float, default=0.70)
    ap.add_argument("--mem_base", type=float, default=1.35)
    ap.add_argument("--kappa", type=float, default=0.65)
    ap.add_argument("--mem_couple", type=float, default=0.45)
    ap.add_argument("--bw_couple", type=float, default=0.35)

    ap.add_argument("--seeds", type=str, default="0:9")
    ap.add_argument("--n_jobs", type=int, default=0)

    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--save_seed_artifacts", action="store_true")
    ap.add_argument("--compare_baseline", action="store_true")

    ap.add_argument("--sweep_param", type=str, default="")
    ap.add_argument("--sweep_values", type=str, default="")
    ap.add_argument("--sweep_param2", type=str, default="")
    ap.add_argument("--sweep_values2", type=str, default="")

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    shell_on = True
    if args.no_shell_on:
        shell_on = False
    elif args.shell_on:
        shell_on = True

    p0 = Params(
        L=args.L,
        T=args.T,
        snap_every=args.snap_every,
        region_radius_frac=args.region_radius_frac,
        shell_on=shell_on,
        shell_width_frac=args.shell_width_frac,
        drive_F=args.drive_F,
        nuc_attempt_good=args.nuc_attempt_good,
        nuc_attempt_bad=args.nuc_attempt_bad,
        barrier_good=args.barrier_good,
        barrier_bad=args.barrier_bad,
        neighbor_gain_good=args.neighbor_gain_good,
        neighbor_gain_bad=args.neighbor_gain_bad,
        eat_bad_over_good=args.eat_bad_over_good,
        relax_good_to_matrix=args.relax_good_to_matrix,
        neg_mode=args.neg_mode,
        neg_mem_gain=args.neg_mem_gain,
        neg_bw_gain=args.neg_bw_gain,
        bandwidth_base=args.bandwidth_base,
        mem_base=args.mem_base,
        kappa=args.kappa,
        mem_couple=args.mem_couple,
        bw_couple=args.bw_couple,
    )

    seeds = parse_seeds(args.seeds)
    ncpu = os.cpu_count() or 1
    n_jobs = args.n_jobs if args.n_jobs > 0 else min(len(seeds), max(1, ncpu // 2))

    do_sweep = bool(args.sweep_param.strip()) and bool(args.sweep_values.strip())
    do_sweep2 = bool(args.sweep_param2.strip()) and bool(args.sweep_values2.strip())

    sweep_vals1 = parse_values_csv(args.sweep_values) if do_sweep else []
    sweep_vals2 = parse_values_csv(args.sweep_values2) if do_sweep2 else []

    # tag for directory naming (will be shortened + hashed in make_run_dir)
    tag = f"L{p0.L}_T{p0.T}_drvF{p0.drive_F}_A{p0.region_radius_frac}_w{p0.shell_width_frac}_negmem_nm{p0.neg_mem_gain}_nb{p0.neg_bw_gain}_bwb{p0.bandwidth_base}_memb{p0.mem_base}_seeds{seeds[0]}_{seeds[-1]}_jobs{n_jobs}"
    tag += ("_SWEEP_microstructure_v9" if do_sweep else "_microstructure_v9")

    run_dir = make_run_dir(args.out_dir, tag)
    seed_root = os.path.join(run_dir, "seed_runs")

    print("============================================================")
    print("HSF microstructure competition toy v9" + (" — SWEEP" if do_sweep else ""))
    print("------------------------------------------------------------")
    print(f"seeds   = {seeds}")
    print(f"n_jobs  = {n_jobs} (cpu_count={ncpu})")
    print(f"run_dir = {run_dir}")
    print("------------------------------------------------------------")
    print("Phases: 0=matrix, 1=GOOD (rare), 2=BAD (fast)")
    print("v9: rare nucleation + runaway growth + competition + metastability")
    print("HSF knobs: pos-drive selection; neg-drive bookkeeping; optional boundary shell")
    if args.compare_baseline:
        print("baseline: shell OFF + negative bookkeeping OFF")
    if do_sweep:
        print("------------------------------------------------------------")
        print(f"sweep_param  = {args.sweep_param} values={sweep_vals1}")
        if do_sweep2:
            print(f"sweep_param2 = {args.sweep_param2} values={sweep_vals2}")
    print("============================================================")

    ctx = mp.get_context("spawn")

    all_rows: List[SeedRow] = []
    pooled_rows: List[Dict[str, Any]] = []

    def run_point(p_point: Params, sweep_k: str, sweep_v: str) -> None:
        nonlocal all_rows, pooled_rows

        # Short per-point tag to avoid Windows MAX_PATH (deep seed artifact dirs)
        # Keep human hint + stable hash.
        point_tag = f"{safe_tag(sweep_k)[:10]}_{safe_tag(sweep_v)[:14]}_{short_hash(sweep_k + '=' + sweep_v, 6)}"

        # v9 mode (shell ON; neg bookkeeping per p_point.neg_mode)
        v9_work = []
        for s in seeds:
            v9_work.append((s, "v9", p_point, True, seed_root, point_tag, args.save_seed_artifacts, sweep_k, sweep_v))

        base_work = []
        if args.compare_baseline:
            # baseline: disable bookkeeping + shell off
            p_base = replace(p_point, neg_mem_gain=0.0, neg_bw_gain=0.0, neg_mode="off")
            for s in seeds:
                base_work.append((s, "baseline", p_base, False, seed_root, point_tag, args.save_seed_artifacts, sweep_k, sweep_v))

        with ctx.Pool(processes=n_jobs) as pool:
            v9_rows = pool.map(_worker, v9_work)
            base_rows = pool.map(_worker, base_work) if base_work else []

        all_rows.extend(v9_rows)
        all_rows.extend(base_rows)

        # pooled records (keep sweep_v tagged for CSV readability)
        pr_v9 = pooled_row(all_rows, "v9", sweep_k, safe_tag(sweep_v))
        pr_base = pooled_row(all_rows, "baseline", sweep_k, safe_tag(sweep_v)) if base_rows else {}

        if pr_v9:
            if pr_base:
                pr_v9["baseline_final_good_mean"] = pr_base.get("final_good_mean", float("nan"))
                pr_v9["baseline_final_bad_mean"] = pr_base.get("final_bad_mean", float("nan"))
                pr_v9["baseline_delta"] = pr_base.get("delta_good_minus_bad", float("nan"))
                pr_v9["delta_improve"] = pr_v9["delta_good_minus_bad"] - pr_v9["baseline_delta"]
            pooled_rows.append(pr_v9)

    if do_sweep and not do_sweep2:
        for v1 in sweep_vals1:
            p1 = apply_sweep(p0, args.sweep_param, v1)
            name1 = args.sweep_param.strip()
            run_point(p1, name1, f"{name1}={v1}")
    elif do_sweep and do_sweep2:
        for v1 in sweep_vals1:
            for v2 in sweep_vals2:
                p1 = apply_sweep(p0, args.sweep_param, v1)
                p2 = apply_sweep(p1, args.sweep_param2, v2)
                name1 = args.sweep_param.strip()
                name2 = args.sweep_param2.strip()
                run_point(p2, f"{name1}+{name2}", f"{name1}={v1},{name2}={v2}")
    else:
        run_point(p0, "nosweep", "default")

    seed_csv = os.path.join(run_dir, "summary.csv")
    pooled_csv = os.path.join(run_dir, "pooled_summary.csv")
    write_seed_csv(seed_csv, all_rows)
    write_pooled_csv(pooled_csv, pooled_rows)

    print("------------------------------------------------------------")
    print("Wrote:")
    print("  summary.csv        :", seed_csv)
    print("  pooled_summary.csv :", pooled_csv)
    print("run_dir           :", run_dir)
    print("seed_runs dir     :", seed_root)
    print("------------------------------------------------------------")
    print("Done.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
