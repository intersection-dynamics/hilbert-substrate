#!/usr/bin/env python3
"""
subsystem_branching_v4_1.py
==========================
Subsystem Growth Model v4.1 (scaling-lab instrument, corrected observable)

WHAT CHANGED vs v4
------------------
Your v4 analysis fit alpha using total_subsystems, which includes "interface subsystems"
that grow with link formation even after site creation saturates. That contaminates the
geometric claim N(t) ~ t^d.

v4.1 FIX:
  - All growth-law analysis (alpha fits, alpha_local, primary N plots) uses SITE subsystems:
        N_site(t) = site_subsystems
  - total_subsystems is still recorded and saved, but treated as a secondary bookkeeping series.
  - Fit window auto-truncates BEFORE saturation of site_subsystems, so caps don't distort alpha.

OUTPUT CONTRACT
---------------
Creates: out_root/<timestamp>__subsystem_growth_v4_1__gitXXXX__<run_name>/
  env.json
  config.json
  aggregate_summary.csv
  plots/ (sweep-level plots)
  <condition>/aggregate.json
  <condition>/summary_by_seed.csv
  <condition>/timeseries/timeseries_seedXXXX.csv
  <condition>/plots/*.png

MODES
-----
  --mode single
  --mode v2_dim_sweep
  --mode v3_gamma_sweep

DEPENDENCIES
------------
  numpy
  matplotlib (optional, unless --no_plots)

WINDOWS ONE-LINERS
------------------
v2 sweep (alpha ~ d):
  python subsystem_branching_v4_1.py --mode v2_dim_sweep --dims 1,2,3 --nsteps 500 --ntrials 30 --seed0 0 --no_signaling --spatial_exclusion --bandwidth 4 --exclusion_radius 0.15 --interaction_rate 0.08 --spawn_rate 0.03 --light_speed 1.0 --max_subsystems 20000 --out_root hsf_out --run_name v2_alpha_eq_d_clean

v3 gamma sweep (alpha vs gamma):
  python subsystem_branching_v4_1.py --mode v3_gamma_sweep --spatial_dim 3 --gammas 0,0.05,0.1,0.2,0.3,0.4 --nsteps 600 --ntrials 30 --seed0 0 --energy_total 50000 --energy_cost 1.0 --no_signaling --spatial_exclusion --bandwidth 4 --exclusion_radius 0.15 --interaction_rate 0.08 --spawn_rate 0.03 --light_speed 1.0 --max_subsystems 20000 --out_root hsf_out --run_name v3_alpha_vs_gamma

"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _try_git_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="ignore").strip()
        return s if s else "nogit"
    except Exception:
        return "nogit"


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def _bootstrap_ci_mean(values: np.ndarray, n_boot: int = 4000, ci: float = 0.95, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
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


def _r2(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    ss = float(np.sum((y - yp) ** 2))
    st = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss / max(st, 1e-12)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

@dataclass
class StepCounters:
    births_attempted: int = 0
    births_accepted: int = 0
    births_reject_lightcone: int = 0
    births_reject_exclusion: int = 0
    births_reject_energy: int = 0
    links_added_from_births: int = 0
    random_links_attempts: int = 0
    random_links_added: int = 0


class SubstrateGraph:
    """
    Growing graph with:
      - hard lightcone for births (optional)
      - spatial exclusion for births (optional)
      - energy cost + density suppression for births (optional)
      - bandwidth cap for links (optional)
    """

    def __init__(
        self,
        rng: np.random.Generator,
        n_initial: int = 2,
        d_hilbert: int = 3,
        bandwidth: Optional[int] = None,
        spatial_dim: int = 3,
        light_speed: float = 1.0,
        no_signaling: bool = False,
        spatial_exclusion: bool = False,
        exclusion_radius: float = 0.2,
        energy_total: Optional[float] = None,   # None => infinite (off)
        energy_cost: float = 1.0,
        energy_gamma: float = 0.0,
        volume_floor_r: float = 0.1,
        link_distance_factor: float = 1.5,
        birth_distance_min: float = 0.1,
        birth_distance_max_factor: float = 0.8,
    ):
        self.rng = rng
        self.d_hilbert = int(d_hilbert)
        self.bandwidth = bandwidth
        self.spatial_dim = int(spatial_dim)
        self.light_speed = float(light_speed)
        self.no_signaling = bool(no_signaling)
        self.spatial_exclusion = bool(spatial_exclusion)
        self.exclusion_radius = float(exclusion_radius)

        self.energy_total = energy_total
        self.energy_remaining = float(energy_total) if energy_total is not None else float("inf")
        self.energy_cost = float(energy_cost)
        self.energy_gamma = float(energy_gamma)
        self.initial_energy_density = None  # set lazily
        self.volume_floor_r = float(volume_floor_r)

        self.link_distance_factor = float(link_distance_factor)
        self.birth_distance_min = float(birth_distance_min)
        self.birth_distance_max_factor = float(birth_distance_max_factor)

        # Graph state
        self.timestep = 0
        self.n_subsystems = int(n_initial)  # SITE subsystems
        self.n_links = 0
        self.n_interface_subsystems = 0  # bookkeeping: 2 per link
        self.links = set()
        self.degree: Dict[int, int] = {}

        # Positions, activation
        self.positions: Dict[int, np.ndarray] = {}
        self.activation_time: Dict[int, int] = {}

        for i in range(n_initial):
            self.positions[i] = self.rng.standard_normal(self.spatial_dim) * 0.1
            self.activation_time[i] = 0
            self.degree[i] = 0

        for i in range(n_initial - 1):
            self._add_link(i, i + 1)

        # Caches
        self._active_cache_step = -1
        self._active_pos_array = np.empty((0, self.spatial_dim), dtype=float)

        self._all_pos_dirty = True
        self._all_pos_array = np.empty((0, self.spatial_dim), dtype=float)

        # History: stored per-step
        self.history = {
            "t": [0],
            "site_subsystems": [int(self.n_subsystems)],
            "total_subsystems": [int(self.total_subsystems())],
            "interface_subsystems": [int(self.n_interface_subsystems)],
            "n_links": [int(self.n_links)],
            "mean_degree": [float(self._mean_degree())],
            "hilbert_log2_dim": [float(self._log2_hilbert_dim())],
            "frontier_radius": [0.0],
            "volume_proxy": [float(self._volume_proxy(self.volume_floor_r))],
            "density": [float(self._density())],
            "energy_remaining": [0.0 if self.energy_total is None else float(self.energy_remaining)],
            "energy_density": [0.0 if self.energy_total is None else float(self._energy_density())],
            "creation_rate_modifier": [1.0],
            "births_attempted": [0],
            "births_accepted": [0],
            "births_reject_lightcone": [0],
            "births_reject_exclusion": [0],
            "births_reject_energy": [0],
            "links_added_from_births": [0],
            "random_links_attempts": [0],
            "random_links_added": [0],
        }

    def total_subsystems(self) -> int:
        return int(self.n_subsystems + self.n_interface_subsystems)

    def _mean_degree(self) -> float:
        if self.n_subsystems <= 0:
            return 0.0
        return 2.0 * self.n_links / max(1, self.n_subsystems)

    def _log2_hilbert_dim(self) -> float:
        return (
            self.n_subsystems * math.log2(self.d_hilbert)
            + self.n_links * 2.0 * math.log2(self.d_hilbert)
        )

    def _frontier_radius(self) -> float:
        if not self.positions:
            return 0.0
        origin = np.zeros(self.spatial_dim, dtype=float)
        return float(max(np.linalg.norm(p - origin) for p in self.positions.values()))

    def _volume_proxy(self, r: float) -> float:
        d = self.spatial_dim
        r = float(max(r, 0.0))
        coeff = math.pi ** (0.5 * d) / math.gamma(0.5 * d + 1.0)
        return float(coeff * (r ** d))

    def _density(self) -> float:
        r = self._frontier_radius()
        vol = self._volume_proxy(max(r, self.volume_floor_r))
        return float(self.n_subsystems / max(vol, 1e-12))

    def _energy_density(self) -> float:
        if self.energy_total is None:
            return 0.0
        r = self._frontier_radius()
        vol = self._volume_proxy(max(r, self.volume_floor_r))
        return float(self.energy_remaining / max(vol, 1e-12))

    def _creation_rate_modifier(self) -> float:
        if self.energy_total is None or self.energy_gamma == 0.0:
            return 1.0
        if self.energy_remaining <= 0.0:
            return 0.0

        rho = self._energy_density()
        if self.initial_energy_density is None or self.initial_energy_density <= 0.0:
            self.initial_energy_density = rho
            return 1.0

        ratio = rho / max(self.initial_energy_density, 1e-12)
        mod = ratio ** self.energy_gamma
        return float(min(1.0, max(0.0, mod)))

    def _is_active(self, node_id: int) -> bool:
        return self.activation_time.get(node_id, 10**18) < self.timestep

    def _rebuild_active_positions(self) -> None:
        active_ids = [nid for nid, ta in self.activation_time.items() if ta < self.timestep]
        if active_ids:
            self._active_pos_array = np.array([self.positions[nid] for nid in active_ids], dtype=float)
        else:
            self._active_pos_array = np.empty((0, self.spatial_dim), dtype=float)
        self._active_cache_step = self.timestep

    def _in_light_cone(self, pos: np.ndarray) -> bool:
        if not self.no_signaling:
            return True
        if self._active_cache_step != self.timestep:
            self._rebuild_active_positions()
        if self._active_pos_array.shape[0] == 0:
            return False
        dists = np.linalg.norm(self._active_pos_array - pos, axis=1)
        return bool(np.min(dists) <= self.light_speed)

    def _check_exclusion(self, pos: np.ndarray) -> bool:
        if not self.spatial_exclusion:
            return True
        if self._all_pos_dirty:
            ids = sorted(self.positions.keys())
            if ids:
                self._all_pos_array = np.array([self.positions[i] for i in ids], dtype=float)
            else:
                self._all_pos_array = np.empty((0, self.spatial_dim), dtype=float)
            self._all_pos_dirty = False
        if self._all_pos_array.shape[0] == 0:
            return True
        dists = np.linalg.norm(self._all_pos_array - pos, axis=1)
        return bool(np.min(dists) >= self.exclusion_radius)

    def _can_link(self, i: int, j: int) -> bool:
        if i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in self.links:
            return False
        if self.bandwidth is not None:
            if self.degree.get(i, 0) >= self.bandwidth or self.degree.get(j, 0) >= self.bandwidth:
                return False
        if i in self.positions and j in self.positions:
            dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
            if dist > self.light_speed * self.link_distance_factor:
                return False
        return True

    def _add_link(self, i: int, j: int) -> None:
        a, b = (i, j) if i < j else (j, i)
        self.links.add((a, b))
        self.n_links += 1
        self.degree[i] = self.degree.get(i, 0) + 1
        self.degree[j] = self.degree.get(j, 0) + 1
        self.n_interface_subsystems += 2

    def _propose_position(self, parent: int) -> np.ndarray:
        parent_pos = self.positions[parent]
        direction = self.rng.standard_normal(self.spatial_dim)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        dmin = self.birth_distance_min
        dmax = self.light_speed * self.birth_distance_max_factor
        dist = float(self.rng.uniform(dmin, max(dmin, dmax)))
        return parent_pos + direction * dist

    def _add_subsystem(self, pos: np.ndarray) -> int:
        new_id = int(self.n_subsystems)
        self.n_subsystems += 1
        self.positions[new_id] = pos
        self.activation_time[new_id] = self.timestep
        self.degree[new_id] = 0
        self._all_pos_dirty = True
        return new_id

    def _record(self, counters: StepCounters) -> None:
        t = self.timestep
        r = self._frontier_radius()
        vol = self._volume_proxy(max(r, self.volume_floor_r))
        rhoE = self._energy_density()

        self.history["t"].append(int(t))
        self.history["site_subsystems"].append(int(self.n_subsystems))
        self.history["total_subsystems"].append(int(self.total_subsystems()))
        self.history["interface_subsystems"].append(int(self.n_interface_subsystems))
        self.history["n_links"].append(int(self.n_links))
        self.history["mean_degree"].append(float(self._mean_degree()))
        self.history["hilbert_log2_dim"].append(float(self._log2_hilbert_dim()))
        self.history["frontier_radius"].append(float(r))
        self.history["volume_proxy"].append(float(vol))
        self.history["density"].append(float(self._density()))
        self.history["energy_remaining"].append(0.0 if self.energy_total is None else float(self.energy_remaining))
        self.history["energy_density"].append(0.0 if self.energy_total is None else float(rhoE))
        self.history["creation_rate_modifier"].append(float(self._creation_rate_modifier()))

        self.history["births_attempted"].append(int(counters.births_attempted))
        self.history["births_accepted"].append(int(counters.births_accepted))
        self.history["births_reject_lightcone"].append(int(counters.births_reject_lightcone))
        self.history["births_reject_exclusion"].append(int(counters.births_reject_exclusion))
        self.history["births_reject_energy"].append(int(counters.births_reject_energy))
        self.history["links_added_from_births"].append(int(counters.links_added_from_births))
        self.history["random_links_attempts"].append(int(counters.random_links_attempts))
        self.history["random_links_added"].append(int(counters.random_links_added))

    def step(
        self,
        interaction_rate: float = 0.1,
        spawn_rate: float = 0.05,
        max_subsystems: int = 10000,
        random_link_samples_cap: int = 500,
    ) -> StepCounters:
        self.timestep += 1
        counters = StepCounters()

        if self.n_subsystems >= max_subsystems:
            # even if site growth is capped, we still allow random links (keeps graph alive)
            # but births from Phase 1 will stop.
            pass

        rate_mod = self._creation_rate_modifier()
        effective_rate = float(interaction_rate * rate_mod)

        # Rebuild caches for this timestep
        self._rebuild_active_positions()
        self._all_pos_dirty = True

        new_subsystems: List[Tuple[int, int]] = []

        # Phase 1: Active links attempt births (SITE subsystems only)
        for (i, j) in list(self.links):
            if self.n_subsystems >= max_subsystems:
                break
            if self.energy_total is not None and self.energy_remaining < self.energy_cost:
                break
            if not (self._is_active(i) and self._is_active(j)):
                continue

            if float(self.rng.random()) < effective_rate:
                counters.births_attempted += 1

                parent = i if float(self.rng.random()) < 0.5 else j
                pos = self._propose_position(parent)

                if not self._in_light_cone(pos):
                    counters.births_reject_lightcone += 1
                    continue
                if not self._check_exclusion(pos):
                    counters.births_reject_exclusion += 1
                    continue

                if self.energy_total is not None:
                    if self.energy_remaining < self.energy_cost:
                        counters.births_reject_energy += 1
                        continue
                    self.energy_remaining -= self.energy_cost

                new_id = self._add_subsystem(pos)
                new_subsystems.append((new_id, parent))
                counters.births_accepted += 1

        # Phase 2: Link new subsystems to parents
        for new_id, parent in new_subsystems:
            if self._can_link(new_id, parent):
                self._add_link(new_id, parent)
                counters.links_added_from_births += 1

        # Phase 3: Random link formation among active sites
        n_samples = int(min(self.n_subsystems * 2, random_link_samples_cap))
        for _ in range(n_samples):
            counters.random_links_attempts += 1
            i = int(self.rng.integers(0, self.n_subsystems))
            j = int(self.rng.integers(0, self.n_subsystems))
            if i != j and self._can_link(i, j):
                if self._is_active(i) and self._is_active(j):
                    if float(self.rng.random()) < spawn_rate:
                        self._add_link(i, j)
                        counters.random_links_added += 1

        self._record(counters)
        return counters


# -----------------------------------------------------------------------------
# Analysis: fit window control + fits + alpha_local
# -----------------------------------------------------------------------------

@dataclass
class FitResult:
    fit_type: str
    exp_rate: float
    exp_r2: float
    power_exp: float
    power_r2: float
    lin_rate: float
    lin_r2: float
    t_start: int
    t_end: int
    saturated: int  # 0/1
    sat_t: int      # timestep when saturation detected (or -1)


def _detect_saturation_time(site_series: np.ndarray, max_subsystems: int, sat_frac: float = 0.98) -> int:
    """Return first index t where site_series >= sat_frac * max_subsystems, else -1."""
    thresh = sat_frac * float(max_subsystems)
    idx = np.where(site_series >= thresh)[0]
    return int(idx[0]) if idx.size > 0 else -1


def _fit_slice_indices(
    t: np.ndarray,
    site_series: np.ndarray,
    max_subsystems: int,
    start_frac: float,
    end_frac: float,
    sat_frac: float = 0.98,
    min_points: int = 12,
) -> Tuple[int, int, int, int]:
    """
    Compute [i0, i1) indices for fitting, truncating end before saturation.
    Returns: i0, i1, saturated(0/1), sat_t (t index, not physical time)
    """
    t = np.asarray(t, dtype=float)
    site_series = np.asarray(site_series, dtype=float)
    n = len(t)
    i0 = int(max(2, math.floor(n * start_frac)))

    # Requested end
    req_i1 = int(max(i0 + min_points, math.floor(n * end_frac)))
    req_i1 = min(req_i1, n)

    sat_i = _detect_saturation_time(site_series, max_subsystems, sat_frac=sat_frac)
    saturated = 0
    sat_t = -1
    i1 = req_i1

    if sat_i >= 0:
        saturated = 1
        sat_t = sat_i
        # truncate to before saturation, but keep at least min_points if possible
        i1 = min(i1, sat_i)
        if i1 - i0 < min_points:
            # not enough points before saturation; fall back to requested window (but mark saturated)
            i1 = req_i1

    # final guard
    if i1 - i0 < min_points:
        i0 = max(2, i1 - min_points)

    return i0, i1, saturated, sat_t


def fit_growth_site(
    t: np.ndarray,
    siteN: np.ndarray,
    max_subsystems: int,
    start_frac: float = 1 / 7,
    end_frac: float = 1.0,
    sat_frac: float = 0.98,
) -> FitResult:
    t = np.asarray(t, dtype=float)
    N = np.asarray(siteN, dtype=float)

    i0, i1, saturated, sat_t = _fit_slice_indices(t, N, max_subsystems, start_frac, end_frac, sat_frac=sat_frac)

    tf = t[i0:i1]
    Nf = N[i0:i1]

    if len(tf) < 8 or Nf[-1] <= Nf[0] + 1:
        return FitResult(
            fit_type="stalled",
            exp_rate=0.0, exp_r2=0.0,
            power_exp=0.0, power_r2=0.0,
            lin_rate=0.0, lin_r2=0.0,
            t_start=int(tf[0]) if len(tf) else int(t[0]),
            t_end=int(tf[-1]) if len(tf) else int(t[-1]),
            saturated=saturated,
            sat_t=sat_t,
        )

    logN = np.log(np.maximum(Nf, 1.0))
    logt = np.log(np.maximum(tf, 1.0))

    # Exponential: logN ~ a t + b
    ce = np.polyfit(tf, logN, 1)
    yp_e = np.polyval(ce, tf)
    r2e = _r2(logN, yp_e)

    # Power: logN ~ a logt + b
    cp = np.polyfit(logt, logN, 1)
    yp_p = np.polyval(cp, logt)
    r2p = _r2(logN, yp_p)

    # Linear: N ~ a t + b
    cl = np.polyfit(tf, Nf, 1)
    yp_l = np.polyval(cl, tf)
    r2l = _r2(Nf, yp_l)

    if r2p > r2e + 0.005:
        ftype = "power_law"
    elif r2e > r2p + 0.005:
        ftype = "exponential"
    else:
        ftype = "power_law" if r2p > 0.95 else "exponential"

    return FitResult(
        fit_type=ftype,
        exp_rate=float(ce[0]), exp_r2=float(r2e),
        power_exp=float(cp[0]), power_r2=float(r2p),
        lin_rate=float(cl[0]), lin_r2=float(r2l),
        t_start=int(tf[0]), t_end=int(tf[-1]),
        saturated=saturated,
        sat_t=sat_t,
    )


@dataclass
class FrontierFit:
    speed: float
    r2: float
    t_start: int
    t_end: int


def fit_frontier(
    t: np.ndarray,
    R: np.ndarray,
    start_frac: float = 1 / 7,
    end_frac: float = 1.0,
) -> FrontierFit:
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    n = len(t)
    i0 = int(max(2, math.floor(n * start_frac)))
    i1 = int(max(i0 + 12, math.floor(n * end_frac)))
    i1 = min(i1, n)

    tf = t[i0:i1]
    rf = R[i0:i1]

    if len(tf) < 8 or rf[-1] <= rf[0] + 1e-6:
        return FrontierFit(speed=0.0, r2=0.0, t_start=int(tf[0]) if len(tf) else int(t[0]), t_end=int(tf[-1]) if len(tf) else int(t[-1]))

    c = np.polyfit(tf, rf, 1)
    yp = np.polyval(c, tf)
    return FrontierFit(speed=float(c[0]), r2=float(_r2(rf, yp)), t_start=int(tf[0]), t_end=int(tf[-1]))


def alpha_local(
    t: np.ndarray,
    siteN: np.ndarray,
    window: int = 35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding window slope of log(siteN) vs log(t)."""
    t = np.asarray(t, dtype=float)
    N = np.asarray(siteN, dtype=float)
    if len(t) < max(window + 5, 12):
        return np.asarray([]), np.asarray([])

    tt = np.maximum(t, 1.0)
    NN = np.maximum(N, 1.0)
    logt = np.log(tt)
    logN = np.log(NN)

    slopes = []
    centers = []
    for i in range(window, len(t) + 1):
        xs = logt[i - window:i]
        ys = logN[i - window:i]
        m = np.polyfit(xs, ys, 1)[0]
        slopes.append(float(m))
        centers.append(float(t[i - 1]))
    return np.asarray(centers), np.asarray(slopes)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_timeseries_mean_with_band(outpath: str, t: np.ndarray, series_by_seed: np.ndarray, y_label: str, title: str, logx: bool = False, logy: bool = False) -> None:
    plt = _import_matplotlib()
    mean = series_by_seed.mean(axis=0)
    std = series_by_seed.std(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, mean)
    ax.fill_between(t, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("t")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_loglog_growth_with_fit(outpath: str, t: np.ndarray, N_mean: np.ndarray, fit: FitResult, title: str) -> None:
    plt = _import_matplotlib()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(t, N_mean, label="site N(t) mean")

    # fit segment
    i0 = np.searchsorted(t, fit.t_start)
    i1 = np.searchsorted(t, fit.t_end) + 1
    i0 = max(i0, 1)
    i1 = min(i1, len(t))

    tf = t[i0:i1]
    if len(tf) > 3:
        a = fit.power_exp
        logt = np.log(np.maximum(tf, 1.0))
        logN = np.log(np.maximum(N_mean[i0:i1], 1.0))
        b = float(np.polyfit(logt, logN, 1)[1])
        yfit = np.exp(b) * (np.maximum(tf, 1.0) ** a)
        ax.plot(tf, yfit, linestyle="--", label=f"fit: α={a:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("site N")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_alpha_local(outpath: str, centers_ref: np.ndarray, slopes_by_seed: List[np.ndarray], title: str) -> None:
    plt = _import_matplotlib()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if len(slopes_by_seed) == 0 or centers_ref.size == 0:
        ax.set_title(title + " (insufficient data)")
        fig.tight_layout()
        fig.savefig(outpath, dpi=160)
        plt.close(fig)
        return

    L = min(len(s) for s in slopes_by_seed)
    S = np.stack([s[:L] for s in slopes_by_seed], axis=0)
    c = centers_ref[:L]

    mean = S.mean(axis=0)
    std = S.std(axis=0)

    ax.plot(c, mean)
    ax.fill_between(c, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("t")
    ax.set_ylabel("alpha_local")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_scatter_with_errorbars(outpath: str, x: np.ndarray, y: np.ndarray, yerr: np.ndarray, xlabel: str, ylabel: str, title: str, line_y_eq_x: bool = False) -> None:
    plt = _import_matplotlib()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    if line_y_eq_x:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ax.plot([xmin, xmax], [xmin, xmax], linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Experiment engine
# -----------------------------------------------------------------------------

@dataclass
class Condition:
    label: str
    spatial_dim: int
    d_hilbert: int
    bandwidth: Optional[int]
    light_speed: float
    no_signaling: bool
    spatial_exclusion: bool
    exclusion_radius: float
    energy_total: Optional[float]
    energy_cost: float
    energy_gamma: float

    nsteps: int
    n_initial: int
    interaction_rate: float
    spawn_rate: float
    max_subsystems: int
    random_link_samples_cap: int

    fit_start_frac: float
    fit_end_frac: float
    alpha_local_window: int
    sat_frac: float

    volume_floor_r: float


def build_condition_from_args(label: str, args: argparse.Namespace, spatial_dim: int, energy_gamma: float) -> Condition:
    return Condition(
        label=label,
        spatial_dim=int(spatial_dim),
        d_hilbert=int(args.d_hilbert),
        bandwidth=None if args.bandwidth is None else int(args.bandwidth),
        light_speed=float(args.light_speed),
        no_signaling=bool(args.no_signaling),
        spatial_exclusion=bool(args.spatial_exclusion),
        exclusion_radius=float(args.exclusion_radius),
        energy_total=None if args.energy_total is None else float(args.energy_total),
        energy_cost=float(args.energy_cost),
        energy_gamma=float(energy_gamma),
        nsteps=int(args.nsteps),
        n_initial=int(args.n_initial),
        interaction_rate=float(args.interaction_rate),
        spawn_rate=float(args.spawn_rate),
        max_subsystems=int(args.max_subsystems),
        random_link_samples_cap=int(args.random_link_samples_cap),
        fit_start_frac=float(args.fit_start_frac),
        fit_end_frac=float(args.fit_end_frac),
        alpha_local_window=int(args.alpha_local_window),
        sat_frac=float(args.sat_frac),
        volume_floor_r=float(args.volume_floor_r),
    )


def run_condition(cond: Condition, seeds: List[int]) -> Tuple[List[Dict[str, np.ndarray]], List[FitResult], List[FrontierFit], List[Tuple[np.ndarray, np.ndarray]]]:
    histories = []
    growth_fits = []
    frontier_fits = []
    alpha_locals = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        g = SubstrateGraph(
            rng=rng,
            n_initial=cond.n_initial,
            d_hilbert=cond.d_hilbert,
            bandwidth=cond.bandwidth,
            spatial_dim=cond.spatial_dim,
            light_speed=cond.light_speed,
            no_signaling=cond.no_signaling,
            spatial_exclusion=cond.spatial_exclusion,
            exclusion_radius=cond.exclusion_radius,
            energy_total=cond.energy_total,
            energy_cost=cond.energy_cost,
            energy_gamma=cond.energy_gamma,
            volume_floor_r=cond.volume_floor_r,
        )

        for _ in range(cond.nsteps):
            g.step(
                interaction_rate=cond.interaction_rate,
                spawn_rate=cond.spawn_rate,
                max_subsystems=cond.max_subsystems,
                random_link_samples_cap=cond.random_link_samples_cap,
            )

        hist = {k: np.asarray(v) for k, v in g.history.items()}
        histories.append(hist)

        t = hist["t"]
        siteN = hist["site_subsystems"]
        R = hist["frontier_radius"]

        gf = fit_growth_site(
            t=t,
            siteN=siteN,
            max_subsystems=cond.max_subsystems,
            start_frac=cond.fit_start_frac,
            end_frac=cond.fit_end_frac,
            sat_frac=cond.sat_frac,
        )
        ff = fit_frontier(t, R, start_frac=cond.fit_start_frac, end_frac=cond.fit_end_frac)
        growth_fits.append(gf)
        frontier_fits.append(ff)

        centers, slopes = alpha_local(t, siteN, window=cond.alpha_local_window)
        alpha_locals.append((centers, slopes))

    return histories, growth_fits, frontier_fits, alpha_locals


def histories_to_matrix(histories: List[Dict[str, np.ndarray]], key: str) -> np.ndarray:
    if len(histories) == 0:
        return np.empty((0, 0))
    L = min(len(h[key]) for h in histories)
    return np.stack([h[key][:L] for h in histories], axis=0)


def write_timeseries_csv(path: str, hist: Dict[str, np.ndarray]) -> None:
    keys = list(hist.keys())
    keys = ["t"] + [k for k in keys if k != "t"]
    L = len(hist["t"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(L):
            w.writerow([hist[k][i] for k in keys])


def write_seed_summary_csv(path: str, cond: Condition, seeds: List[int], growth_fits: List[FitResult], frontier_fits: List[FrontierFit], histories: List[Dict[str, np.ndarray]]) -> None:
    cols = [
        "seed",
        "label",
        "spatial_dim",
        "energy_gamma",
        "energy_total",
        "energy_cost",
        "bandwidth",
        "no_signaling",
        "spatial_exclusion",
        "exclusion_radius",
        "interaction_rate",
        "spawn_rate",
        "nsteps",
        "max_subsystems",
        "siteN_final",
        "totalN_final",
        "R_final",
        "E_final",
        "modifier_final",
        "fit_type",
        "alpha_site",
        "power_r2",
        "exp_rate",
        "exp_r2",
        "frontier_speed",
        "frontier_r2",
        "fit_t_start",
        "fit_t_end",
        "saturated",
        "sat_t_index",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for seed, gf, ff, hist in zip(seeds, growth_fits, frontier_fits, histories):
            siteN_final = float(hist["site_subsystems"][-1])
            totalN_final = float(hist["total_subsystems"][-1])
            R_final = float(hist["frontier_radius"][-1])
            E_final = float(hist["energy_remaining"][-1])
            mod_final = float(hist["creation_rate_modifier"][-1])
            w.writerow([
                seed,
                cond.label,
                cond.spatial_dim,
                cond.energy_gamma,
                "" if cond.energy_total is None else cond.energy_total,
                cond.energy_cost,
                "" if cond.bandwidth is None else cond.bandwidth,
                int(cond.no_signaling),
                int(cond.spatial_exclusion),
                cond.exclusion_radius,
                cond.interaction_rate,
                cond.spawn_rate,
                cond.nsteps,
                cond.max_subsystems,
                siteN_final,
                totalN_final,
                R_final,
                E_final,
                mod_final,
                gf.fit_type,
                gf.power_exp,
                gf.power_r2,
                gf.exp_rate,
                gf.exp_r2,
                ff.speed,
                ff.r2,
                gf.t_start,
                gf.t_end,
                gf.saturated,
                gf.sat_t,
            ])


def aggregate_condition_summary(cond: Condition, seeds: List[int], growth_fits: List[FitResult], frontier_fits: List[FrontierFit], histories: List[Dict[str, np.ndarray]], rng: np.random.Generator) -> Dict[str, float]:
    alphas = np.array([gf.power_exp for gf in growth_fits], dtype=float)
    power_r2s = np.array([gf.power_r2 for gf in growth_fits], dtype=float)
    speeds = np.array([ff.speed for ff in frontier_fits], dtype=float)

    siteNf = np.array([h["site_subsystems"][-1] for h in histories], dtype=float)
    totalNf = np.array([h["total_subsystems"][-1] for h in histories], dtype=float)
    Rf = np.array([h["frontier_radius"][-1] for h in histories], dtype=float)

    sat_flags = np.array([gf.saturated for gf in growth_fits], dtype=float)

    lo, hi = _bootstrap_ci_mean(alphas, rng=rng)

    return {
        "n_seeds": float(len(seeds)),
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
        "alpha_ci_lo": float(lo),
        "alpha_ci_hi": float(hi),
        "power_r2_mean": float(np.mean(power_r2s)),
        "frontier_speed_mean": float(np.mean(speeds)),
        "siteN_final_mean": float(np.mean(siteNf)),
        "totalN_final_mean": float(np.mean(totalNf)),
        "R_final_mean": float(np.mean(Rf)),
        "saturation_rate": float(np.mean(sat_flags)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Subsystem Growth Model v4.1 (correct observable: site_subsystems)")

    p.add_argument("--mode", type=str, default="single", choices=["single", "v2_dim_sweep", "v3_gamma_sweep"])

    p.add_argument("--nsteps", type=int, default=500)
    p.add_argument("--ntrials", type=int, default=30, help="Seeds per condition.")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--n_initial", type=int, default=2)
    p.add_argument("--max_subsystems", type=int, default=20000)

    p.add_argument("--spatial_dim", type=int, default=3)
    p.add_argument("--d_hilbert", type=int, default=3)
    p.add_argument("--light_speed", type=float, default=1.0)
    p.add_argument("--bandwidth", type=int, default=4)
    p.add_argument("--no_signaling", action="store_true")
    p.add_argument("--spatial_exclusion", action="store_true")
    p.add_argument("--exclusion_radius", type=float, default=0.15)

    p.add_argument("--interaction_rate", type=float, default=0.08)
    p.add_argument("--spawn_rate", type=float, default=0.03)
    p.add_argument("--random_link_samples_cap", type=int, default=500)

    p.add_argument("--energy_total", type=float, default=None, help="None disables energy (v2).")
    p.add_argument("--energy_cost", type=float, default=1.0)
    p.add_argument("--energy_gamma", type=float, default=0.0, help="Used in --mode single.")

    p.add_argument("--volume_floor_r", type=float, default=0.1)

    # analysis
    p.add_argument("--fit_start_frac", type=float, default=1/7)
    p.add_argument("--fit_end_frac", type=float, default=1.0)
    p.add_argument("--alpha_local_window", type=int, default=35)
    p.add_argument("--sat_frac", type=float, default=0.98, help="Saturation threshold fraction of max_subsystems for truncating fit window.")

    # sweeps
    p.add_argument("--dims", type=str, default="1,2,3")
    p.add_argument("--gammas", type=str, default="0,0.05,0.1,0.2,0.3,0.4")

    # output
    p.add_argument("--out_root", type=str, default="hsf_out")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--no_plots", action="store_true")

    args = p.parse_args()

    run_stamp = _now_stamp()
    git_hash = _try_git_hash()
    suffix = f"__{args.run_name}" if args.run_name else ""
    run_dir = os.path.join(args.out_root, f"{run_stamp}__subsystem_growth_v4_1__git{git_hash}{suffix}")

    _safe_makedirs(run_dir)
    _safe_makedirs(os.path.join(run_dir, "plots"))

    env = {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
        "argv": sys.argv,
        "git_hash": git_hash,
        "timestamp_local": run_stamp,
    }
    _write_json(os.path.join(run_dir, "env.json"), env)

    seeds = [int(args.seed0 + i) for i in range(int(args.ntrials))]
    cfg = dataclasses.asdict(args) if dataclasses.is_dataclass(args) else vars(args)
    cfg["seeds"] = seeds
    _write_json(os.path.join(run_dir, "config.json"), cfg)

    # Build conditions
    conditions: List[Condition] = []
    if args.mode == "single":
        conditions.append(build_condition_from_args(
            label=f"single_dim{args.spatial_dim}_g{args.energy_gamma:.3f}",
            args=args,
            spatial_dim=args.spatial_dim,
            energy_gamma=args.energy_gamma,
        ))
    elif args.mode == "v2_dim_sweep":
        dims = _parse_int_list(args.dims)
        # force energy off for v2 sweep
        args_energy_total = args.energy_total
        args_energy_gamma = args.energy_gamma
        args.energy_total = None
        args.energy_gamma = 0.0
        for d in dims:
            conditions.append(build_condition_from_args(
                label=f"v2_dim{d}",
                args=args,
                spatial_dim=d,
                energy_gamma=0.0,
            ))
        args.energy_total = args_energy_total
        args.energy_gamma = args_energy_gamma
    elif args.mode == "v3_gamma_sweep":
        gammas = _parse_float_list(args.gammas)
        if args.energy_total is None:
            raise SystemExit("v3_gamma_sweep requires --energy_total (finite pool).")
        for g in gammas:
            conditions.append(build_condition_from_args(
                label=f"v3_dim{args.spatial_dim}_g{g:.3f}",
                args=args,
                spatial_dim=args.spatial_dim,
                energy_gamma=g,
            ))
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    t0 = time.time()
    aggregate_rows = []
    rng_ci = np.random.default_rng(12345)

    for cond in conditions:
        cond_dir = os.path.join(run_dir, cond.label)
        _safe_makedirs(cond_dir)
        _safe_makedirs(os.path.join(cond_dir, "plots"))
        _safe_makedirs(os.path.join(cond_dir, "timeseries"))

        print(f"\n{'='*78}\nRunning condition: {cond.label}\n{'='*78}")

        histories, growth_fits, frontier_fits, alpha_locals = run_condition(cond, seeds)

        # per-seed timeseries
        for seed, hist in zip(seeds, histories):
            ts_path = os.path.join(cond_dir, "timeseries", f"timeseries_seed{seed:04d}.csv")
            write_timeseries_csv(ts_path, hist)

        # per-seed summary
        seed_summary_path = os.path.join(cond_dir, "summary_by_seed.csv")
        write_seed_summary_csv(seed_summary_path, cond, seeds, growth_fits, frontier_fits, histories)

        # aggregate
        agg = aggregate_condition_summary(cond, seeds, growth_fits, frontier_fits, histories, rng=rng_ci)
        agg_row = {
            "label": cond.label,
            "spatial_dim": cond.spatial_dim,
            "energy_gamma": cond.energy_gamma,
            "energy_total": "" if cond.energy_total is None else cond.energy_total,
            "energy_cost": cond.energy_cost,
            "bandwidth": "" if cond.bandwidth is None else cond.bandwidth,
            "no_signaling": int(cond.no_signaling),
            "spatial_exclusion": int(cond.spatial_exclusion),
            "exclusion_radius": cond.exclusion_radius,
            "interaction_rate": cond.interaction_rate,
            "spawn_rate": cond.spawn_rate,
            "nsteps": cond.nsteps,
            "max_subsystems": cond.max_subsystems,
            **agg,
        }
        aggregate_rows.append(agg_row)
        _write_json(os.path.join(cond_dir, "aggregate.json"), agg_row)

        # plots
        if not args.no_plots:
            t_mat = histories_to_matrix(histories, "t")
            L = t_mat.shape[1]
            t_axis = t_mat[0, :L]

            siteN_mat = histories_to_matrix(histories, "site_subsystems")
            totalN_mat = histories_to_matrix(histories, "total_subsystems")
            R_mat = histories_to_matrix(histories, "frontier_radius")

            plot_timeseries_mean_with_band(
                outpath=os.path.join(cond_dir, "plots", "siteN_vs_t.png"),
                t=t_axis,
                series_by_seed=siteN_mat,
                y_label="site N(t)",
                title=f"site N(t): {cond.label}",
            )
            plot_timeseries_mean_with_band(
                outpath=os.path.join(cond_dir, "plots", "totalN_vs_t.png"),
                t=t_axis,
                series_by_seed=totalN_mat,
                y_label="total N(t)",
                title=f"total N(t): {cond.label} (secondary)",
            )
            plot_timeseries_mean_with_band(
                outpath=os.path.join(cond_dir, "plots", "R_vs_t.png"),
                t=t_axis,
                series_by_seed=R_mat,
                y_label="R(t)",
                title=f"Frontier radius R(t): {cond.label}",
            )

            # mean fit overlay (analysis still per-seed in summary)
            siteN_mean = siteN_mat.mean(axis=0)
            gf_mean = fit_growth_site(
                t=t_axis,
                siteN=siteN_mean,
                max_subsystems=cond.max_subsystems,
                start_frac=cond.fit_start_frac,
                end_frac=cond.fit_end_frac,
                sat_frac=cond.sat_frac,
            )
            plot_loglog_growth_with_fit(
                outpath=os.path.join(cond_dir, "plots", "siteN_loglog_mean_fit.png"),
                t=t_axis,
                N_mean=siteN_mean,
                fit=gf_mean,
                title=f"log-log site N(t) mean + fit: {cond.label}",
            )

            centers_ref = alpha_locals[0][0] if len(alpha_locals) else np.asarray([])
            slopes_list = [sl for (_, sl) in alpha_locals]
            plot_alpha_local(
                outpath=os.path.join(cond_dir, "plots", "alpha_local.png"),
                centers_ref=centers_ref,
                slopes_by_seed=slopes_list,
                title=f"alpha_local(t) from site N: {cond.label}",
            )

        print(f"  alpha_mean = {agg['alpha_mean']:.3f}   std = {agg['alpha_std']:.3f}   CI95 = [{agg['alpha_ci_lo']:.3f}, {agg['alpha_ci_hi']:.3f}]")
        print(f"  power_r2_mean = {agg['power_r2_mean']:.3f}   frontier_speed_mean = {agg['frontier_speed_mean']:.3f}")
        print(f"  siteN_final_mean = {agg['siteN_final_mean']:.1f}   totalN_final_mean = {agg['totalN_final_mean']:.1f}   saturation_rate = {agg['saturation_rate']:.2f}")

    # Write aggregate_summary.csv
    agg_csv = os.path.join(run_dir, "aggregate_summary.csv")
    if aggregate_rows:
        cols = list(aggregate_rows[0].keys())
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for row in aggregate_rows:
                w.writerow([row.get(c, "") for c in cols])

    # Sweep-level plots
    if not args.no_plots and len(aggregate_rows) >= 2:
        plots_dir = os.path.join(run_dir, "plots")
        _safe_makedirs(plots_dir)

        xs = []
        ys = []
        yerrs = []

        if args.mode == "v2_dim_sweep":
            for row in aggregate_rows:
                xs.append(float(row["spatial_dim"]))
                ys.append(float(row["alpha_mean"]))
                yerrs.append(float(row["alpha_std"]))
            plot_scatter_with_errorbars(
                outpath=os.path.join(plots_dir, "alpha_vs_dim.png"),
                x=np.asarray(xs),
                y=np.asarray(ys),
                yerr=np.asarray(yerrs),
                xlabel="spatial dimension d",
                ylabel="alpha (site growth exponent)",
                title="v2 sweep: alpha(site) vs dimension (target: alpha = d)",
                line_y_eq_x=True,
            )

        if args.mode == "v3_gamma_sweep":
            for row in aggregate_rows:
                xs.append(float(row["energy_gamma"]))
                ys.append(float(row["alpha_mean"]))
                yerrs.append(float(row["alpha_std"]))
            plot_scatter_with_errorbars(
                outpath=os.path.join(plots_dir, "alpha_vs_gamma.png"),
                x=np.asarray(xs),
                y=np.asarray(ys),
                yerr=np.asarray(yerrs),
                xlabel="gamma",
                ylabel="alpha (site growth exponent)",
                title=f"v3 sweep: alpha(site) vs gamma (dim={args.spatial_dim})",
                line_y_eq_x=False,
            )

    elapsed = time.time() - t0
    print(f"\nRun folder: {run_dir}")
    print(f"Aggregate summary: {agg_csv}")
    print(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()