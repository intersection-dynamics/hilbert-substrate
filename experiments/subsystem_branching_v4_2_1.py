#!/usr/bin/env python3
"""
subsystem_branching_v4_2_1.py
============================
v4.2.1 = v4.2 + critical performance fix

FIX:
- Frontier radius R(t) is tracked incrementally (Rmax) instead of O(N) scan each step.
- When death_rate == 0 (v2/v3 default), Rmax is strictly monotone => never rescans.
- When deaths are enabled, we rescan ONLY if a death might have removed the current frontier.

Everything else matches v4.2 output contract and semantics.
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


class SpatialHash:
    def __init__(self, dim: int, cell_size: float):
        self.dim = int(dim)
        self.cell_size = float(max(cell_size, 1e-12))
        self.cells: Dict[Tuple[int, ...], List[int]] = {}

    def _cell_of(self, pos: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(math.floor(float(x) / self.cell_size)) for x in pos.tolist())

    def insert(self, node_id: int, pos: np.ndarray) -> None:
        c = self._cell_of(pos)
        self.cells.setdefault(c, []).append(int(node_id))

    def remove(self, node_id: int, pos: np.ndarray) -> None:
        c = self._cell_of(pos)
        lst = self.cells.get(c)
        if not lst:
            return
        try:
            lst.remove(int(node_id))
        except ValueError:
            pass
        if not lst:
            self.cells.pop(c, None)

    def neighbors_in_radius(self, pos: np.ndarray, radius: float, positions: Dict[int, np.ndarray]) -> List[int]:
        r = float(radius)
        cs = self.cell_size
        k = int(math.ceil(r / cs))
        base = self._cell_of(pos)
        out: List[int] = []

        if self.dim == 1:
            for dx in range(-k, k + 1):
                out.extend(self.cells.get((base[0] + dx,), []))
        elif self.dim == 2:
            for dx in range(-k, k + 1):
                for dy in range(-k, k + 1):
                    out.extend(self.cells.get((base[0] + dx, base[1] + dy), []))
        elif self.dim == 3:
            for dx in range(-k, k + 1):
                for dy in range(-k, k + 1):
                    for dz in range(-k, k + 1):
                        out.extend(self.cells.get((base[0] + dx, base[1] + dy, base[2] + dz), []))
        else:
            def rec(axis: int, prefix: List[int]):
                if axis == self.dim:
                    out.extend(self.cells.get(tuple(prefix), []))
                    return
                for d in range(-k, k + 1):
                    prefix.append(base[axis] + d)
                    rec(axis + 1, prefix)
                    prefix.pop()
            rec(0, [])

        if not out:
            return []
        out2 = []
        for nid in out:
            p = positions.get(nid)
            if p is None:
                continue
            if float(np.linalg.norm(p - pos)) < r:
                out2.append(nid)
        return out2


@dataclass
class StepCounters:
    birth_attempts: int = 0
    births_accepted: int = 0
    births_reject_exclusion: int = 0
    births_reject_memory: int = 0
    births_reject_energy: int = 0
    deaths_attempted: int = 0
    deaths_executed: int = 0
    links_added_random: int = 0
    links_attempted_random: int = 0


class SubstrateBoilingModel:
    def __init__(
        self,
        rng: np.random.Generator,
        spatial_dim: int,
        light_speed: float,
        no_signaling: bool,
        spatial_exclusion: bool,
        exclusion_radius: float,
        bandwidth: Optional[int],
        link_distance_factor: float,
        birth_attempts_per_step: int,
        frontier_bias: float,
        step_frac_min: float,
        step_frac_max: float,
        birth_distance_min: float,
        random_link_attempts_per_step: int,
        random_link_prob: float,
        energy_total: Optional[float],
        energy_cost: float,
        energy_gamma: float,
        volume_floor_r: float,
        death_rate: float,
        memory_total: Optional[float],
        memory_per_site: float,
        memory_per_linkend: float,
        memory_refund_frac: float,
        max_subsystems: int,
        n_initial: int = 2,
        d_hilbert: int = 3,
    ):
        self.rng = rng
        self.spatial_dim = int(spatial_dim)
        self.light_speed = float(light_speed)
        self.no_signaling = bool(no_signaling)
        self.spatial_exclusion = bool(spatial_exclusion)
        self.exclusion_radius = float(exclusion_radius)
        self.bandwidth = bandwidth
        self.link_distance_factor = float(link_distance_factor)

        self.birth_attempts_per_step = int(max(0, birth_attempts_per_step))
        self.frontier_bias = float(np.clip(frontier_bias, 0.0, 1.0))
        self.step_frac_min = float(step_frac_min)
        self.step_frac_max = float(step_frac_max)
        self.birth_distance_min = float(birth_distance_min)

        self.random_link_attempts_per_step = int(max(0, random_link_attempts_per_step))
        self.random_link_prob = float(np.clip(random_link_prob, 0.0, 1.0))

        self.energy_total = None if energy_total is None else float(energy_total)
        self.energy_remaining = float("inf") if energy_total is None else float(energy_total)
        self.energy_cost = float(energy_cost)
        self.energy_gamma = float(energy_gamma)
        self.initial_energy_density = None
        self.volume_floor_r = float(volume_floor_r)

        self.death_rate = float(np.clip(death_rate, 0.0, 1.0))

        self.memory_total = None if memory_total is None else float(memory_total)
        self.memory_per_site = float(max(0.0, memory_per_site))
        self.memory_per_linkend = float(max(0.0, memory_per_linkend))
        self.memory_refund_frac = float(np.clip(memory_refund_frac, 0.0, 1.0))

        self.max_subsystems = int(max_subsystems)
        self.d_hilbert = int(d_hilbert)

        self.timestep = 0
        self.positions: Dict[int, np.ndarray] = {}
        self.birth_time: Dict[int, int] = {}
        self.alive: List[int] = []
        self.alive_index: Dict[int, int] = {}
        self.links: set[Tuple[int, int]] = set()
        self.neighbors: Dict[int, set[int]] = {}
        self.next_node_id = 0

        self.hash = SpatialHash(self.spatial_dim, self.exclusion_radius) if self.spatial_exclusion else None

        # NEW: frontier tracking
        self.Rmax = 0.0
        self._needs_Rmax_rescan = False  # only used when deaths enabled

        for _ in range(int(n_initial)):
            pos = self.rng.standard_normal(self.spatial_dim) * 0.1
            nid = self._create_node(pos)
            self.birth_time[nid] = 0

        for i in range(len(self.alive) - 1):
            self._try_add_link(self.alive[i], self.alive[i + 1], force=True)

        self.history: Dict[str, List[float]] = {}
        self._init_history()

    def _init_history(self) -> None:
        for k in [
            "t","site_subsystems","total_subsystems","interface_subsystems","n_links","mean_degree",
            "frontier_radius","volume_proxy","density",
            "energy_remaining","energy_density","creation_rate_modifier",
            "memory_total","memory_bound","memory_free",
            "birth_attempts","births_accepted","births_reject_exclusion","births_reject_memory","births_reject_energy",
            "deaths_attempted","deaths_executed",
            "links_attempted_random","links_added_random",
        ]:
            self.history[k] = []
        self._record(StepCounters())

    def n_sites(self) -> int:
        return int(len(self.alive))

    def n_links(self) -> int:
        return int(len(self.links))

    def n_interface_subsystems(self) -> int:
        return 2 * self.n_links()

    def total_subsystems(self) -> int:
        return self.n_sites() + self.n_interface_subsystems()

    def mean_degree(self) -> float:
        n = self.n_sites()
        return 0.0 if n <= 0 else 2.0 * self.n_links() / max(1, n)

    def frontier_radius(self) -> float:
        if not self.alive:
            self.Rmax = 0.0
            return 0.0
        if self.death_rate > 0.0 and self._needs_Rmax_rescan:
            # rescan only when deaths could have removed frontier
            self.Rmax = float(max(np.linalg.norm(self.positions[n]) for n in self.alive))
            self._needs_Rmax_rescan = False
        return float(self.Rmax)

    def volume_proxy(self, r: float) -> float:
        d = self.spatial_dim
        r = float(max(r, 0.0))
        coeff = math.pi ** (0.5 * d) / math.gamma(0.5 * d + 1.0)
        return float(coeff * (r ** d))

    def density(self) -> float:
        r = self.frontier_radius()
        vol = self.volume_proxy(max(r, self.volume_floor_r))
        return float(self.n_sites() / max(vol, 1e-12))

    def energy_density(self) -> float:
        if self.energy_total is None:
            return 0.0
        r = self.frontier_radius()
        vol = self.volume_proxy(max(r, self.volume_floor_r))
        return float(self.energy_remaining / max(vol, 1e-12))

    def creation_rate_modifier(self) -> float:
        if self.energy_total is None or self.energy_gamma == 0.0:
            return 1.0
        if self.energy_remaining <= 0.0:
            return 0.0
        rho = self.energy_density()
        if self.initial_energy_density is None or self.initial_energy_density <= 0.0:
            self.initial_energy_density = rho
            return 1.0
        ratio = rho / max(self.initial_energy_density, 1e-12)
        return float(min(1.0, max(0.0, ratio ** self.energy_gamma)))

    def memory_bound(self) -> float:
        return self.memory_per_site * float(self.n_sites()) + self.memory_per_linkend * float(2 * self.n_links())

    def memory_free(self) -> float:
        if self.memory_total is None:
            return float("inf")
        return float(self.memory_total - self.memory_bound())

    def _can_afford_birth_memory(self) -> bool:
        if self.memory_total is None:
            return True
        return self.memory_free() >= self.memory_per_site - 1e-12

    def _create_node(self, pos: np.ndarray) -> int:
        nid = int(self.next_node_id)
        self.next_node_id += 1
        pos = np.asarray(pos, dtype=float)
        self.positions[nid] = pos
        self.birth_time[nid] = int(self.timestep)
        self.neighbors[nid] = set()
        self.alive_index[nid] = len(self.alive)
        self.alive.append(nid)
        if self.hash is not None:
            self.hash.insert(nid, pos)

        # NEW: update frontier incrementally
        r = float(np.linalg.norm(pos))
        if r > self.Rmax:
            self.Rmax = r
        return nid

    def _remove_link(self, i: int, j: int) -> None:
        a, b = (i, j) if i < j else (j, i)
        self.links.discard((a, b))
        if i in self.neighbors:
            self.neighbors[i].discard(j)
        if j in self.neighbors:
            self.neighbors[j].discard(i)

    def _remove_node(self, nid: int) -> None:
        # if this node might be on frontier, mark rescan
        if self.death_rate > 0.0:
            p = self.positions.get(nid)
            if p is not None:
                r = float(np.linalg.norm(p))
                if r >= self.Rmax - 1e-9:
                    self._needs_Rmax_rescan = True

        nbrs = list(self.neighbors.get(nid, []))
        for j in nbrs:
            self._remove_link(nid, j)

        pos = self.positions.get(nid)
        if pos is not None and self.hash is not None:
            self.hash.remove(nid, pos)

        idx = self.alive_index.get(nid)
        if idx is not None:
            last = self.alive[-1]
            self.alive[idx] = last
            self.alive_index[last] = idx
            self.alive.pop()
            self.alive_index.pop(nid, None)

        self.positions.pop(nid, None)
        self.neighbors.pop(nid, None)
        self.birth_time.pop(nid, None)

    def _can_link(self, i: int, j: int) -> bool:
        if i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in self.links:
            return False
        if i not in self.positions or j not in self.positions:
            return False
        if self.bandwidth is not None:
            if len(self.neighbors.get(i, set())) >= self.bandwidth:
                return False
            if len(self.neighbors.get(j, set())) >= self.bandwidth:
                return False
        dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
        return dist <= self.light_speed * self.link_distance_factor

    def _try_add_link(self, i: int, j: int, force: bool = False) -> bool:
        if not force and not self._can_link(i, j):
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in self.links:
            return False
        self.links.add((a, b))
        self.neighbors[i].add(j)
        self.neighbors[j].add(i)
        return True

    def _passes_exclusion(self, pos: np.ndarray) -> bool:
        if not self.spatial_exclusion:
            return True
        if self.hash is None:
            for nid in self.alive:
                if float(np.linalg.norm(self.positions[nid] - pos)) < self.exclusion_radius:
                    return False
            return True
        neigh = self.hash.neighbors_in_radius(pos, self.exclusion_radius, self.positions)
        return len(neigh) == 0

    def _choose_parent(self) -> Optional[int]:
        if not self.alive:
            return None
        if self.frontier_bias <= 0.0:
            return int(self.alive[int(self.rng.integers(0, len(self.alive)))])
        Rmax = self.frontier_radius()
        if Rmax <= 1e-12:
            return int(self.alive[int(self.rng.integers(0, len(self.alive)))])
        thresh = (1.0 - self.frontier_bias) * Rmax
        for _ in range(32):
            nid = int(self.alive[int(self.rng.integers(0, len(self.alive)))])
            if float(np.linalg.norm(self.positions[nid])) >= thresh:
                return nid
        frontier = [nid for nid in self.alive if float(np.linalg.norm(self.positions[nid])) >= thresh]
        if frontier:
            return int(frontier[int(self.rng.integers(0, len(frontier)))])
        return int(self.alive[int(self.rng.integers(0, len(self.alive)))])

    def _propose_birth_position(self, parent: int) -> np.ndarray:
        p = self.positions[parent]
        direction = self.rng.standard_normal(self.spatial_dim)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        fmin = float(min(self.step_frac_min, self.step_frac_max))
        fmax = float(max(self.step_frac_min, self.step_frac_max))
        dist = float(self.light_speed * self.rng.uniform(fmin, fmax))
        dist = max(dist, self.birth_distance_min)
        return p + direction * dist

    def step(self) -> StepCounters:
        self.timestep += 1
        ctr = StepCounters()

        # deaths
        if self.death_rate > 0.0 and self.alive:
            n = len(self.alive)
            k = int(self.rng.binomial(n, self.death_rate))
            ctr.deaths_attempted = k
            if k > 0:
                victims_idx = self.rng.choice(n, size=min(k, n), replace=False)
                victims = [self.alive[int(i)] for i in victims_idx]
                for nid in victims:
                    if len(self.alive) <= 1:
                        break
                    self._remove_node(nid)
                    ctr.deaths_executed += 1

        mod = self.creation_rate_modifier()

        for _ in range(self.birth_attempts_per_step):
            if self.n_sites() >= self.max_subsystems:
                break
            if self.energy_total is not None and self.energy_remaining < self.energy_cost:
                ctr.births_reject_energy += 1
                continue
            if not self._can_afford_birth_memory():
                ctr.births_reject_memory += 1
                continue
            if mod < 1.0 and float(self.rng.random()) > mod:
                continue

            parent = self._choose_parent()
            if parent is None:
                continue
            if self.no_signaling:
                bt = self.birth_time.get(parent, 0)
                if bt == self.timestep - 1:
                    continue

            ctr.birth_attempts += 1
            pos = self._propose_birth_position(parent)
            if self.no_signaling:
                if float(np.linalg.norm(pos - self.positions[parent])) > self.light_speed + 1e-9:
                    continue
            if not self._passes_exclusion(pos):
                ctr.births_reject_exclusion += 1
                continue
            if self.energy_total is not None:
                if self.energy_remaining < self.energy_cost:
                    ctr.births_reject_energy += 1
                    continue
                self.energy_remaining -= self.energy_cost

            nid = self._create_node(pos)
            ctr.births_accepted += 1
            self._try_add_link(nid, parent, force=False)

        if self.random_link_attempts_per_step > 0 and len(self.alive) >= 2:
            n = len(self.alive)
            for _ in range(self.random_link_attempts_per_step):
                ctr.links_attempted_random += 1
                if float(self.rng.random()) > self.random_link_prob:
                    continue
                i = int(self.alive[int(self.rng.integers(0, n))])
                j = int(self.alive[int(self.rng.integers(0, n))])
                if i == j:
                    continue
                if self._try_add_link(i, j, force=False):
                    ctr.links_added_random += 1

        self._record(ctr)
        return ctr

    def _record(self, ctr: StepCounters) -> None:
        t = int(self.timestep)
        R = self.frontier_radius()
        V = self.volume_proxy(max(R, self.volume_floor_r))
        dens = self.density()
        self.history["t"].append(t)
        self.history["site_subsystems"].append(float(self.n_sites()))
        self.history["interface_subsystems"].append(float(self.n_interface_subsystems()))
        self.history["total_subsystems"].append(float(self.total_subsystems()))
        self.history["n_links"].append(float(self.n_links()))
        self.history["mean_degree"].append(float(self.mean_degree()))
        self.history["frontier_radius"].append(float(R))
        self.history["volume_proxy"].append(float(V))
        self.history["density"].append(float(dens))
        self.history["energy_remaining"].append(0.0 if self.energy_total is None else float(self.energy_remaining))
        self.history["energy_density"].append(0.0 if self.energy_total is None else float(self.energy_density()))
        self.history["creation_rate_modifier"].append(float(self.creation_rate_modifier()))
        self.history["memory_total"].append(float("inf") if self.memory_total is None else float(self.memory_total))
        self.history["memory_bound"].append(float(self.memory_bound()))
        self.history["memory_free"].append(float(self.memory_free()))
        self.history["birth_attempts"].append(float(ctr.birth_attempts))
        self.history["births_accepted"].append(float(ctr.births_accepted))
        self.history["births_reject_exclusion"].append(float(ctr.births_reject_exclusion))
        self.history["births_reject_memory"].append(float(ctr.births_reject_memory))
        self.history["births_reject_energy"].append(float(ctr.births_reject_energy))
        self.history["deaths_attempted"].append(float(ctr.deaths_attempted))
        self.history["deaths_executed"].append(float(ctr.deaths_executed))
        self.history["links_attempted_random"].append(float(ctr.links_attempted_random))
        self.history["links_added_random"].append(float(ctr.links_added_random))


@dataclass
class FitResult:
    fit_type: str
    alpha: float
    power_r2: float
    exp_rate: float
    exp_r2: float
    lin_rate: float
    lin_r2: float
    t_start: int
    t_end: int
    saturated: int
    sat_t_index: int
    valid_alpha: int
    invalid_reason: str


def _detect_saturation_time(site_series: np.ndarray, max_subsystems: int, sat_frac: float) -> int:
    thresh = sat_frac * float(max_subsystems)
    idx = np.where(site_series >= thresh)[0]
    return int(idx[0]) if idx.size > 0 else -1


def _fit_slice_indices(t: np.ndarray, site_series: np.ndarray, max_subsystems: int, start_frac: float, end_frac: float, sat_frac: float, min_points: int) -> Tuple[int, int, int, int]:
    n = len(t)
    i0 = int(max(2, math.floor(n * start_frac)))
    req_i1 = int(max(i0 + min_points, math.floor(n * end_frac)))
    req_i1 = min(req_i1, n)
    sat_i = _detect_saturation_time(site_series, max_subsystems, sat_frac=sat_frac)
    saturated, sat_t = 0, -1
    i1 = req_i1
    if sat_i >= 0:
        saturated, sat_t = 1, sat_i
        i1 = min(i1, sat_i)
    if i1 - i0 < min_points:
        i1 = req_i1
    i1 = min(i1, n)
    i0 = min(i0, i1 - 1)
    return i0, i1, saturated, sat_t


def fit_growth_site(t: np.ndarray, siteN: np.ndarray, max_subsystems: int, start_frac: float, end_frac: float, sat_frac: float, min_points: int) -> FitResult:
    t = np.asarray(t, dtype=float)
    N = np.asarray(siteN, dtype=float)
    i0, i1, saturated, sat_t = _fit_slice_indices(t, N, max_subsystems, start_frac, end_frac, sat_frac, min_points)
    tf = t[i0:i1]
    Nf = N[i0:i1]
    valid, reason = 1, ""
    if len(tf) < min_points:
        valid, reason = 0, "too_few_points"
    elif Nf[-1] <= Nf[0] + 2:
        valid, reason = 0, "no_growth"
    if valid == 0:
        return FitResult("invalid", float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                         int(tf[0]) if len(tf) else int(t[0]), int(tf[-1]) if len(tf) else int(t[-1]),
                         saturated, sat_t, 0, reason)

    logN = np.log(np.maximum(Nf, 1.0))
    logt = np.log(np.maximum(tf, 1.0))
    ce = np.polyfit(tf, logN, 1)
    r2e = _r2(logN, np.polyval(ce, tf))
    cp = np.polyfit(logt, logN, 1)
    r2p = _r2(logN, np.polyval(cp, logt))
    cl = np.polyfit(tf, Nf, 1)
    r2l = _r2(Nf, np.polyval(cl, tf))
    ftype = "power_law" if r2p >= r2e else "exponential"
    return FitResult(ftype, float(cp[0]), float(r2p), float(ce[0]), float(r2e), float(cl[0]), float(r2l),
                     int(tf[0]), int(tf[-1]), saturated, sat_t, 1, "")


@dataclass
class FrontierFit:
    speed: float
    r2: float
    t_start: int
    t_end: int


def fit_frontier(t: np.ndarray, R: np.ndarray, start_frac: float, end_frac: float, min_points: int) -> FrontierFit:
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    n = len(t)
    i0 = int(max(2, math.floor(n * start_frac)))
    i1 = int(max(i0 + min_points, math.floor(n * end_frac)))
    i1 = min(i1, n)
    tf = t[i0:i1]
    rf = R[i0:i1]
    if len(tf) < min_points or rf[-1] <= rf[0] + 1e-6:
        return FrontierFit(0.0, 0.0, int(tf[0]) if len(tf) else int(t[0]), int(tf[-1]) if len(tf) else int(t[-1]))
    c = np.polyfit(tf, rf, 1)
    return FrontierFit(float(c[0]), float(_r2(rf, np.polyval(c, tf))), int(tf[0]), int(tf[-1]))


def alpha_local(t: np.ndarray, siteN: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    N = np.asarray(siteN, dtype=float)
    if len(t) < max(window + 5, 20):
        return np.asarray([]), np.asarray([])
    logt = np.log(np.maximum(t, 1.0))
    logN = np.log(np.maximum(N, 1.0))
    slopes, centers = [], []
    for i in range(window, len(t) + 1):
        m = np.polyfit(logt[i - window:i], logN[i - window:i], 1)[0]
        slopes.append(float(m))
        centers.append(float(t[i - 1]))
    return np.asarray(centers), np.asarray(slopes)


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_timeseries_mean_with_band(outpath: str, t: np.ndarray, mat: np.ndarray, y_label: str, title: str) -> None:
    plt = _import_matplotlib()
    mean, std = mat.mean(axis=0), mat.std(axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, mean)
    ax.fill_between(t, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel(y_label); ax.set_title(title)
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)


def plot_loglog_growth_with_fit(outpath: str, t: np.ndarray, N_mean: np.ndarray, alpha: float, t0: int, t1: int, title: str) -> None:
    plt = _import_matplotlib()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, N_mean, label="site N(t) mean")
    i0 = int(np.searchsorted(t, t0)); i1 = int(np.searchsorted(t, t1) + 1)
    i0 = max(i0, 1); i1 = min(i1, len(t))
    tf = t[i0:i1]
    if len(tf) > 3 and np.isfinite(alpha):
        b = float(np.polyfit(np.log(np.maximum(tf, 1.0)), np.log(np.maximum(N_mean[i0:i1], 1.0)), 1)[1])
        yfit = np.exp(b) * (np.maximum(tf, 1.0) ** alpha)
        ax.plot(tf, yfit, linestyle="--", label=f"fit α={alpha:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("t"); ax.set_ylabel("site N"); ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)


def plot_alpha_local(outpath: str, centers_ref: np.ndarray, slopes_by_seed: List[np.ndarray], title: str) -> None:
    plt = _import_matplotlib()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if not slopes_by_seed or centers_ref.size == 0:
        ax.set_title(title + " (insufficient data)")
        fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig); return
    L = min(len(s) for s in slopes_by_seed)
    S = np.stack([s[:L] for s in slopes_by_seed], axis=0)
    c = centers_ref[:L]
    mean, std = S.mean(axis=0), S.std(axis=0)
    ax.plot(c, mean)
    ax.fill_between(c, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel("alpha_local"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)


def plot_scatter_with_errorbars(outpath: str, x: np.ndarray, y: np.ndarray, yerr: np.ndarray, xlabel: str, ylabel: str, title: str, line_y_eq_x: bool = False) -> None:
    plt = _import_matplotlib()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    if line_y_eq_x:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ax.plot([xmin, xmax], [xmin, xmax], linestyle="--")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)


@dataclass
class Condition:
    label: str
    spatial_dim: int
    light_speed: float
    no_signaling: bool
    spatial_exclusion: bool
    exclusion_radius: float
    bandwidth: Optional[int]
    link_distance_factor: float
    birth_attempts_per_step: int
    frontier_bias: float
    step_frac_min: float
    step_frac_max: float
    birth_distance_min: float
    random_link_attempts_per_step: int
    random_link_prob: float
    energy_total: Optional[float]
    energy_cost: float
    energy_gamma: float
    volume_floor_r: float
    death_rate: float
    memory_total: Optional[float]
    memory_per_site: float
    memory_per_linkend: float
    memory_refund_frac: float
    nsteps: int
    max_subsystems: int
    n_initial: int
    d_hilbert: int
    fit_start_frac: float
    fit_end_frac: float
    sat_frac: float
    fit_min_points: int
    alpha_local_window: int


def build_condition_from_args(label: str, args: argparse.Namespace, spatial_dim: int, energy_gamma: float) -> Condition:
    return Condition(
        label=label,
        spatial_dim=int(spatial_dim),
        light_speed=float(args.light_speed),
        no_signaling=bool(args.no_signaling),
        spatial_exclusion=bool(args.spatial_exclusion),
        exclusion_radius=float(args.exclusion_radius),
        bandwidth=None if args.bandwidth is None else int(args.bandwidth),
        link_distance_factor=float(args.link_distance_factor),
        birth_attempts_per_step=int(args.birth_attempts),
        frontier_bias=float(args.frontier_bias),
        step_frac_min=float(args.step_frac_min),
        step_frac_max=float(args.step_frac_max),
        birth_distance_min=float(args.birth_distance_min),
        random_link_attempts_per_step=int(args.random_link_attempts),
        random_link_prob=float(args.random_link_prob),
        energy_total=None if args.energy_total is None else float(args.energy_total),
        energy_cost=float(args.energy_cost),
        energy_gamma=float(energy_gamma),
        volume_floor_r=float(args.volume_floor_r),
        death_rate=float(args.death_rate),
        memory_total=None if args.memory_total is None else float(args.memory_total),
        memory_per_site=float(args.memory_per_site),
        memory_per_linkend=float(args.memory_per_linkend),
        memory_refund_frac=float(args.memory_refund_frac),
        nsteps=int(args.nsteps),
        max_subsystems=int(args.max_subsystems),
        n_initial=int(args.n_initial),
        d_hilbert=int(args.d_hilbert),
        fit_start_frac=float(args.fit_start_frac),
        fit_end_frac=float(args.fit_end_frac),
        sat_frac=float(args.sat_frac),
        fit_min_points=int(args.fit_min_points),
        alpha_local_window=int(args.alpha_local_window),
    )


def run_condition(cond: Condition, seeds: List[int]):
    histories, fits, ffits, alpha_locals = [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        m = SubstrateBoilingModel(
            rng=rng,
            spatial_dim=cond.spatial_dim,
            light_speed=cond.light_speed,
            no_signaling=cond.no_signaling,
            spatial_exclusion=cond.spatial_exclusion,
            exclusion_radius=cond.exclusion_radius,
            bandwidth=cond.bandwidth,
            link_distance_factor=cond.link_distance_factor,
            birth_attempts_per_step=cond.birth_attempts_per_step,
            frontier_bias=cond.frontier_bias,
            step_frac_min=cond.step_frac_min,
            step_frac_max=cond.step_frac_max,
            birth_distance_min=cond.birth_distance_min,
            random_link_attempts_per_step=cond.random_link_attempts_per_step,
            random_link_prob=cond.random_link_prob,
            energy_total=cond.energy_total,
            energy_cost=cond.energy_cost,
            energy_gamma=cond.energy_gamma,
            volume_floor_r=cond.volume_floor_r,
            death_rate=cond.death_rate,
            memory_total=cond.memory_total,
            memory_per_site=cond.memory_per_site,
            memory_per_linkend=cond.memory_per_linkend,
            memory_refund_frac=cond.memory_refund_frac,
            max_subsystems=cond.max_subsystems,
            n_initial=cond.n_initial,
            d_hilbert=cond.d_hilbert,
        )
        for _ in range(cond.nsteps):
            m.step()
        hist = {k: np.asarray(v, dtype=float) for k, v in m.history.items()}
        histories.append(hist)

        t = hist["t"]; siteN = hist["site_subsystems"]; R = hist["frontier_radius"]
        fr = fit_growth_site(t, siteN, cond.max_subsystems, cond.fit_start_frac, cond.fit_end_frac, cond.sat_frac, cond.fit_min_points)
        ff = fit_frontier(t, R, cond.fit_start_frac, cond.fit_end_frac, cond.fit_min_points)
        fits.append(fr); ffits.append(ff)
        alpha_locals.append(alpha_local(t, siteN, cond.alpha_local_window))
    return histories, fits, ffits, alpha_locals


def histories_to_matrix(histories: List[Dict[str, np.ndarray]], key: str) -> np.ndarray:
    L = min(len(h[key]) for h in histories)
    return np.stack([h[key][:L] for h in histories], axis=0)


def write_timeseries_csv(path: str, hist: Dict[str, np.ndarray]) -> None:
    keys = ["t"] + [k for k in hist.keys() if k != "t"]
    L = len(hist["t"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(L):
            w.writerow([hist[k][i] for k in keys])


def write_seed_summary_csv(path: str, cond: Condition, seeds: List[int], fits: List[FitResult], ffits: List[FrontierFit], histories: List[Dict[str, np.ndarray]]) -> None:
    cols = [
        "seed","label","spatial_dim","energy_gamma","energy_total","death_rate","memory_total",
        "bandwidth","birth_attempts","frontier_bias","step_frac_min","step_frac_max",
        "no_signaling","spatial_exclusion","exclusion_radius","nsteps","max_subsystems",
        "siteN_final","totalN_final","R_final","fit_type","alpha","power_r2",
        "frontier_speed","frontier_r2","valid_alpha","invalid_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for seed, fr, ff, hist in zip(seeds, fits, ffits, histories):
            w.writerow([
                seed, cond.label, cond.spatial_dim, cond.energy_gamma,
                "" if cond.energy_total is None else cond.energy_total,
                cond.death_rate,
                "" if cond.memory_total is None else cond.memory_total,
                "" if cond.bandwidth is None else cond.bandwidth,
                cond.birth_attempts_per_step, cond.frontier_bias, cond.step_frac_min, cond.step_frac_max,
                int(cond.no_signaling), int(cond.spatial_exclusion), cond.exclusion_radius,
                cond.nsteps, cond.max_subsystems,
                float(hist["site_subsystems"][-1]), float(hist["total_subsystems"][-1]), float(hist["frontier_radius"][-1]),
                fr.fit_type, fr.alpha, fr.power_r2,
                ff.speed, ff.r2, fr.valid_alpha, fr.invalid_reason
            ])


def aggregate_condition_summary(cond: Condition, fits: List[FitResult], ffits: List[FrontierFit], histories: List[Dict[str, np.ndarray]], rng: np.random.Generator) -> Dict[str, float]:
    valid = [fr for fr in fits if fr.valid_alpha == 1 and np.isfinite(fr.alpha)]
    alphas = np.array([fr.alpha for fr in valid], dtype=float)
    power_r2 = np.array([fr.power_r2 for fr in valid], dtype=float)
    valid_frac = float(len(valid) / max(1, len(fits)))
    speeds = np.array([ff.speed for ff in ffits], dtype=float)
    r2s = np.array([ff.r2 for ff in ffits], dtype=float)
    siteNf = np.array([h["site_subsystems"][-1] for h in histories], dtype=float)
    Rf = np.array([h["frontier_radius"][-1] for h in histories], dtype=float)
    sat_flags = np.array([fr.saturated for fr in fits], dtype=float)
    lo, hi = _bootstrap_ci_mean(alphas, rng=rng) if len(alphas) else (float("nan"), float("nan"))
    return {
        "valid_alpha_fraction": valid_frac,
        "alpha_mean": float(np.mean(alphas)) if len(alphas) else float("nan"),
        "alpha_std": float(np.std(alphas)) if len(alphas) else float("nan"),
        "alpha_ci_lo": float(lo), "alpha_ci_hi": float(hi),
        "power_r2_mean": float(np.mean(power_r2)) if len(power_r2) else float("nan"),
        "frontier_speed_mean": float(np.mean(speeds)),
        "frontier_r2_mean": float(np.mean(r2s)),
        "siteN_final_mean": float(np.mean(siteNf)),
        "R_final_mean": float(np.mean(Rf)),
        "saturation_rate": float(np.mean(sat_flags)),
    }


def main():
    p = argparse.ArgumentParser(description="Subsystem Growth Model v4.2.1 (fast frontier tracking)")
    p.add_argument("--mode", type=str, default="single", choices=["single","v2_dim_sweep","v3_gamma_sweep"])
    p.add_argument("--nsteps", type=int, default=700)
    p.add_argument("--ntrials", type=int, default=30)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--n_initial", type=int, default=2)
    p.add_argument("--max_subsystems", type=int, default=250000)
    p.add_argument("--spatial_dim", type=int, default=3)
    p.add_argument("--light_speed", type=float, default=1.0)
    p.add_argument("--no_signaling", action="store_true")
    p.add_argument("--spatial_exclusion", action="store_true")
    p.add_argument("--exclusion_radius", type=float, default=0.15)
    p.add_argument("--bandwidth", type=int, default=4)
    p.add_argument("--link_distance_factor", type=float, default=1.5)
    p.add_argument("--random_link_attempts", type=int, default=0)
    p.add_argument("--random_link_prob", type=float, default=0.05)
    p.add_argument("--birth_attempts", type=int, default=220)
    p.add_argument("--frontier_bias", type=float, default=0.15)
    p.add_argument("--step_frac_min", type=float, default=0.85)
    p.add_argument("--step_frac_max", type=float, default=1.0)
    p.add_argument("--birth_distance_min", type=float, default=0.05)
    p.add_argument("--energy_total", type=float, default=None)
    p.add_argument("--energy_cost", type=float, default=1.0)
    p.add_argument("--energy_gamma", type=float, default=0.0)
    p.add_argument("--volume_floor_r", type=float, default=0.1)
    p.add_argument("--death_rate", type=float, default=0.0)
    p.add_argument("--memory_total", type=float, default=None)
    p.add_argument("--memory_per_site", type=float, default=1.0)
    p.add_argument("--memory_per_linkend", type=float, default=0.0)
    p.add_argument("--memory_refund_frac", type=float, default=1.0)
    p.add_argument("--d_hilbert", type=int, default=3)
    p.add_argument("--fit_start_frac", type=float, default=1/7)
    p.add_argument("--fit_end_frac", type=float, default=1.0)
    p.add_argument("--sat_frac", type=float, default=0.995)
    p.add_argument("--fit_min_points", type=int, default=40)
    p.add_argument("--alpha_local_window", type=int, default=45)
    p.add_argument("--dims", type=str, default="1,2,3")
    p.add_argument("--gammas", type=str, default="0,0.05,0.1,0.2,0.3,0.4")
    p.add_argument("--out_root", type=str, default="hsf_out")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    run_stamp = _now_stamp()
    git_hash = _try_git_hash()
    suffix = f"__{args.run_name}" if args.run_name else ""
    run_dir = os.path.join(args.out_root, f"{run_stamp}__subsystem_growth_v4_2_1__git{git_hash}{suffix}")
    _safe_makedirs(run_dir); _safe_makedirs(os.path.join(run_dir, "plots"))

    env = {"python": sys.version.replace("\n"," "), "numpy": np.__version__, "platform": platform.platform(),
           "executable": sys.executable, "argv": sys.argv, "git_hash": git_hash, "timestamp_local": run_stamp}
    _write_json(os.path.join(run_dir, "env.json"), env)
    seeds = [int(args.seed0 + i) for i in range(int(args.ntrials))]
    cfg = dataclasses.asdict(args) if dataclasses.is_dataclass(args) else vars(args)
    cfg["seeds"] = seeds
    _write_json(os.path.join(run_dir, "config.json"), cfg)

    conditions: List[Condition] = []
    if args.mode == "single":
        conditions.append(build_condition_from_args(f"single_dim{args.spatial_dim}_g{args.energy_gamma:.3f}", args, args.spatial_dim, float(args.energy_gamma)))
    elif args.mode == "v2_dim_sweep":
        dims = _parse_int_list(args.dims)
        saved_total, saved_gamma = args.energy_total, args.energy_gamma
        args.energy_total = None; args.energy_gamma = 0.0
        for d in dims:
            conditions.append(build_condition_from_args(f"v2_dim{d}", args, int(d), 0.0))
        args.energy_total, args.energy_gamma = saved_total, saved_gamma
    elif args.mode == "v3_gamma_sweep":
        if args.energy_total is None:
            raise SystemExit("v3_gamma_sweep requires --energy_total")
        for g in _parse_float_list(args.gammas):
            conditions.append(build_condition_from_args(f"v3_dim{args.spatial_dim}_g{g:.3f}", args, int(args.spatial_dim), float(g)))

    t0 = time.time()
    rng_ci = np.random.default_rng(12345)
    aggregate_rows = []

    for cond in conditions:
        cond_dir = os.path.join(run_dir, cond.label)
        _safe_makedirs(cond_dir); _safe_makedirs(os.path.join(cond_dir, "plots")); _safe_makedirs(os.path.join(cond_dir, "timeseries"))

        print(f"\n{'='*78}\nRunning condition: {cond.label}\n{'='*78}")
        histories, fits, ffits, alpha_locals = run_condition(cond, seeds)

        for seed, hist in zip(seeds, histories):
            write_timeseries_csv(os.path.join(cond_dir, "timeseries", f"timeseries_seed{seed:04d}.csv"), hist)

        write_seed_summary_csv(os.path.join(cond_dir, "summary_by_seed.csv"), cond, seeds, fits, ffits, histories)
        agg = aggregate_condition_summary(cond, fits, ffits, histories, rng_ci)
        agg_row = {"label": cond.label, "spatial_dim": cond.spatial_dim, "energy_gamma": cond.energy_gamma, **agg}
        _write_json(os.path.join(cond_dir, "aggregate.json"), agg_row)
        aggregate_rows.append(agg_row)

        print(f"  valid_alpha_fraction = {agg['valid_alpha_fraction']:.2f}")
        print(f"  alpha_mean = {agg['alpha_mean']:.3f}   std = {agg['alpha_std']:.3f}   CI95 = [{agg['alpha_ci_lo']:.3f}, {agg['alpha_ci_hi']:.3f}]")
        print(f"  frontier_speed_mean = {agg['frontier_speed_mean']:.3f}   frontier_r2_mean = {agg['frontier_r2_mean']:.3f}")
        print(f"  siteN_final_mean = {agg['siteN_final_mean']:.1f}   R_final_mean = {agg['R_final_mean']:.3f}   saturation_rate = {agg['saturation_rate']:.2f}")

        if not args.no_plots:
            t_axis = histories_to_matrix(histories, "t")[0]
            siteN_mat = histories_to_matrix(histories, "site_subsystems")
            R_mat = histories_to_matrix(histories, "frontier_radius")
            plot_timeseries_mean_with_band(os.path.join(cond_dir, "plots", "siteN_vs_t.png"), t_axis, siteN_mat, "site N(t)", f"site N(t): {cond.label}")
            plot_timeseries_mean_with_band(os.path.join(cond_dir, "plots", "R_vs_t.png"), t_axis, R_mat, "R(t)", f"R(t): {cond.label}")
            siteN_mean = siteN_mat.mean(axis=0)
            fr_mean = fit_growth_site(t_axis, siteN_mean, cond.max_subsystems, cond.fit_start_frac, cond.fit_end_frac, cond.sat_frac, cond.fit_min_points)
            plot_loglog_growth_with_fit(os.path.join(cond_dir, "plots", "siteN_loglog_mean_fit.png"), t_axis, siteN_mean, fr_mean.alpha, fr_mean.t_start, fr_mean.t_end, f"log-log site N mean: {cond.label}")
            centers_ref = alpha_locals[0][0] if alpha_locals else np.asarray([])
            slopes_list = [sl for (_, sl) in alpha_locals]
            plot_alpha_local(os.path.join(cond_dir, "plots", "alpha_local.png"), centers_ref, slopes_list, f"alpha_local: {cond.label}")

    agg_csv = os.path.join(run_dir, "aggregate_summary.csv")
    if aggregate_rows:
        cols = list(aggregate_rows[0].keys())
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols)
            for row in aggregate_rows:
                w.writerow([row.get(c, "") for c in cols])

    elapsed = time.time() - t0
    print(f"\nRun folder: {run_dir}")
    print(f"Aggregate summary: {agg_csv}")
    print(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()