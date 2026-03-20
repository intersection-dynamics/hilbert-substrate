# hsf_metastable_branch_toy_v8.py
# ------------------------------------------------------------
# v8: "HSF-clean" promotion of the *working* mechanism + SWEEP MODE + MULTI-SEED MULTICORE
#
# Empirical lesson from your pooled v7 run:
#   - Baseline (unsigned pos-only barrier coupling) is REAL (pooled z ~ 3+).
#   - Signed barrier coupling (v7 diode) was NOT stable (pooled z ~ 0).
#
# v8 makes baseline "HSF-clean" by giving negative half-cycle a job:
#   - Positive drive lowers the branch barrier (selection pressure).
#   - Negative drive does NOT directly anti-select; it loads bookkeeping (edge memory) and/or tightens bandwidth,
#     which is exactly the "constraint cost" picture (no-forgetting + finite bandwidth).
#
# Mechanisms:
#   (A) Barrier coupling (pos-only; preserves the strong pooled effect):
#       drive_term_barrier = gain_plus * max(0, drive) / (1 + local_mem)
#
#   (B) Negative drive -> bookkeeping load (HSF-clean):
#       For inside edges each tick:
#           m_e += neg_mem_gain * max(0, -drive)
#       Optionally also:
#           bandwidth inside edges *= (1 - neg_bw_gain * max(0, -drive_norm))
#
#   (C) Boundary shell (optional; same multipliers as before)
#
# Multicore features:
#   - Multi-seed pooling: --seeds 0:9 --n_jobs 8
#   - Baseline compare on the same seeds (optional): --compare_baseline
#
# Sweep mode:
#   - Sweep ONE parameter across values while pooling across seeds.
#   - Example:
#       --sweep_param neg_mem_gain --sweep_values 0,0.05,0.1,0.2
#   - Or sweep TWO parameters as a grid (small grids recommended):
#       --sweep_param neg_mem_gain --sweep_values 0,0.05,0.1
#       --sweep_param2 boundary_mem_mult --sweep_values2 1.0,1.2,1.35
#
# Outputs:
#   - summary.csv (per-seed per-mode)
#   - pooled_summary.csv (pooled stats per sweep point)
#   - For non-sweep: per-seed run dirs with plots + pooled printout.
#   - For sweep: saves only pooled CSV by default (to keep disk sane),
#                optionally save per-seed artifacts with --save_seed_artifacts.
#
# SAFE MODEL (generic branching, non-nuclear).
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
from dataclasses import dataclass, replace
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


TAU = 2.0 * np.pi


# ----------------------------
# Utilities
# ----------------------------

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_tag(s: str) -> str:
    return s.replace(" ", "_").replace(":", "_").replace("/", "_").replace(".", "p")


def make_run_dir(base_out: str, tag: str) -> str:
    run_dir = os.path.join(base_out, f"{now_stamp()}_{safe_tag(tag)}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "seed_runs"), exist_ok=True)
    return run_dir


def wrap_pi_scalar(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def parse_seeds(spec: str) -> List[int]:
    s = spec.strip()
    if "," in s:
        out = []
        for part in s.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        # unique, sorted
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
        if step == 0:
            raise ValueError("step cannot be 0")
        # inclusive range
        if a <= b and step > 0:
            return list(range(a, b + 1, step))
        if a > b and step > 0:
            return list(range(a, b - 1, -step))
        if a >= b and step < 0:
            return list(range(a, b - 1, step))
        return list(range(a, b + 1, step))
    return [int(s)]


def parse_float_list(spec: str) -> List[float]:
    spec = spec.strip()
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if w <= 1:
        return x.copy()
    c = np.cumsum(x)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


# ----------------------------
# Model core
# ----------------------------

@dataclass(frozen=True)
class Params:
    # graph
    N: int
    p_edge: float

    # time
    T: int
    dt: float

    # constraints / dynamics
    bandwidth: float
    mem_decay: float
    mem_couple: float
    kappa: float

    # gauge
    g_coup: float
    theta_bw: float

    # events
    lambda_capture: float
    barrier_M: float
    base_bias: float

    # lifetimes
    tau_G: float
    tau_M: float

    # tuned region + drive
    drive_frac: float
    drive_amp: float
    drive_w: float
    drive_phase: float

    # barrier coupling (pos only)
    drive_gain_plus: float

    # negative-drive bookkeeping
    neg_mem_gain: float        # adds to edge memory on inside edges on negative half-cycle
    neg_bw_gain: float         # optional bandwidth tightening inside edges on negative half-cycle (0 disables)

    # boundary shell multipliers
    region_bandwidth_mult: float
    region_mem_mult: float
    boundary_bandwidth_mult: float
    boundary_mem_mult: float

    # diagnostics
    rolling_window: int
    min_captures_plot: int
    phase_bins: int
    min_captures_phase: int


def build_erdos_renyi(rng: np.random.Generator, N: int, p_edge: float) -> Tuple[np.ndarray, np.ndarray]:
    u_list, v_list = [], []
    for i in range(N):
        r = rng.random(N - i - 1)
        js = np.where(r < p_edge)[0] + (i + 1)
        for j in js:
            u_list.append(i)
            v_list.append(int(j))
    if not u_list:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.array(u_list, dtype=np.int32), np.array(v_list, dtype=np.int32)


def choose_tuned_region(rng: np.random.Generator, N: int, frac: float) -> np.ndarray:
    frac = float(np.clip(frac, 0.0, 1.0))
    k = int(round(frac * N))
    mask = np.zeros(N, dtype=np.int8)
    if k <= 0:
        return mask
    idx = np.arange(N)
    rng.shuffle(idx)
    mask[idx[:k]] = 1
    return mask


def branch_prob_metastable(barrier_M: float, drive_term: float, base_bias: float) -> float:
    eff = barrier_M - drive_term - base_bias
    if eff > 60:
        return 0.0
    if eff < -60:
        return 1.0
    return float(1.0 / (1.0 + np.exp(eff)))


def final_stats_counts(cap_in: int, cap_out: int, M_in: int, M_out: int) -> Tuple[float, float, float]:
    eps = 1e-18
    p_in = M_in / (cap_in + eps)
    p_out = M_out / (cap_out + eps)
    delta = p_in - p_out
    se = math.sqrt((p_in * (1.0 - p_in) / (cap_in + eps)) + (p_out * (1.0 - p_out) / (cap_out + eps)))
    z = delta / (se + eps)
    return float(delta), float(se), float(z)


def compute_series(data: dict, w: int) -> dict:
    eps = 1e-12
    cap_in = data["captures_in"].astype(np.float64)
    cap_out = data["captures_out"].astype(np.float64)
    cap_tot = data["captures_total"].astype(np.float64)
    M_in = data["M_in"].astype(np.float64)
    M_out = data["M_out"].astype(np.float64)
    M_tot = data["M_total"].astype(np.float64)

    CapInCum = np.cumsum(cap_in)
    CapOutCum = np.cumsum(cap_out)
    CapTotCum = np.cumsum(cap_tot)
    MinCum = np.cumsum(M_in)
    MoutCum = np.cumsum(M_out)
    MtotCum = np.cumsum(M_tot)

    frac_in_cum = MinCum / (CapInCum + eps)
    frac_out_cum = MoutCum / (CapOutCum + eps)
    frac_tot_cum = MtotCum / (CapTotCum + eps)
    delta_cum = frac_in_cum - frac_out_cum

    cap_in_roll = rolling_sum(cap_in, w)
    cap_out_roll = rolling_sum(cap_out, w)
    cap_tot_roll = rolling_sum(cap_tot, w)
    M_in_roll = rolling_sum(M_in, w)
    M_out_roll = rolling_sum(M_out, w)
    M_tot_roll = rolling_sum(M_tot, w)

    frac_in_roll = M_in_roll / (cap_in_roll + eps)
    frac_out_roll = M_out_roll / (cap_out_roll + eps)
    frac_tot_roll = M_tot_roll / (cap_tot_roll + eps)
    delta_roll = frac_in_roll - frac_out_roll

    return dict(
        frac_in_cum=frac_in_cum,
        frac_out_cum=frac_out_cum,
        frac_tot_cum=frac_tot_cum,
        delta_cum=delta_cum,
        cap_in_roll=cap_in_roll,
        cap_out_roll=cap_out_roll,
        cap_tot_roll=cap_tot_roll,
        frac_in_roll=frac_in_roll,
        frac_out_roll=frac_out_roll,
        frac_tot_roll=frac_tot_roll,
        delta_roll=delta_roll,
    )


def phase_stats(data: dict) -> dict:
    B = int(data["phi_bins"])
    eps = 1e-12
    caps_in = data["phi_caps_in"].astype(np.float64)
    caps_out = data["phi_caps_out"].astype(np.float64)
    M_in = data["phi_M_in"].astype(np.float64)
    M_out = data["phi_M_out"].astype(np.float64)
    p_in = M_in / (caps_in + eps)
    p_out = M_out / (caps_out + eps)
    delta = p_in - p_out
    phi_centers = (np.arange(B) + 0.5) * (TAU / B)
    return dict(phi_centers=phi_centers, caps_in=caps_in, caps_out=caps_out, p_in=p_in, p_out=p_out, delta=delta)


def write_csv(path: str, data: dict, series: dict) -> None:
    T = len(data["captures_total"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "captures_total", "captures_in", "captures_out",
            "M_total", "M_in", "M_out",
            "frac_M_in_cum", "frac_M_out_cum", "delta_cum",
            "frac_M_in_roll", "frac_M_out_roll", "delta_roll",
            "locked_like", "mean_edge_mem", "mean_theta_step",
        ])
        for t in range(T):
            w.writerow([
                t,
                int(data["captures_total"][t]),
                int(data["captures_in"][t]),
                int(data["captures_out"][t]),
                int(data["M_total"][t]),
                int(data["M_in"][t]),
                int(data["M_out"][t]),
                float(series["frac_in_cum"][t]),
                float(series["frac_out_cum"][t]),
                float(series["delta_cum"][t]),
                float(series["frac_in_roll"][t]),
                float(series["frac_out_roll"][t]),
                float(series["delta_roll"][t]),
                float(data["locked_like"][t]),
                float(data["mean_edge_mem"][t]),
                float(data["mean_theta_step"][t]),
            ])


def write_phase_csv(path: str, ph: dict, min_caps: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin", "phi_center", "caps_in", "caps_out", "pM_in", "pM_out", "delta"])
        for i in range(len(ph["phi_centers"])):
            ci = float(ph["caps_in"][i])
            co = float(ph["caps_out"][i])
            if (ci < min_caps) or (co < min_caps):
                w.writerow([i, float(ph["phi_centers"][i]), ci, co, "nan", "nan", "nan"])
            else:
                w.writerow([i, float(ph["phi_centers"][i]), ci, co, float(ph["p_in"][i]), float(ph["p_out"][i]), float(ph["delta"][i])])


def plot_outputs(out_dir: str, data: dict, series: dict, ph: dict, p: Params, prefix: str) -> None:
    plots = os.path.join(out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    t = np.arange(len(data["captures_total"]))

    plt.figure()
    plt.plot(t, data["mean_edge_mem"], label="mean edge memory")
    plt.plot(t, data["locked_like"], label="locked_like (p85 mem fraction)")
    plt.plot(t, data["mean_theta_step"], label="mean |dtheta|")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots, f"{prefix}_timeseries.png"), dpi=160, bbox_inches="tight")
    plt.close()

    mask_roll = (series["cap_in_roll"] >= p.min_captures_plot) & (series["cap_out_roll"] >= p.min_captures_plot)
    frac_in_roll = series["frac_in_roll"].copy()
    frac_out_roll = series["frac_out_roll"].copy()
    delta_roll = series["delta_roll"].copy()
    frac_in_roll[~mask_roll] = np.nan
    frac_out_roll[~mask_roll] = np.nan
    delta_roll[~mask_roll] = np.nan

    plt.figure()
    plt.plot(t, series["frac_in_cum"], label="M tuned (cum)")
    plt.plot(t, series["frac_out_cum"], label="M out (cum)")
    plt.plot(t, series["delta_cum"], label="delta cum")
    plt.plot(t, frac_in_roll, label=f"M tuned (roll w={p.rolling_window})", linewidth=1.0)
    plt.plot(t, frac_out_roll, label=f"M out (roll w={p.rolling_window})", linewidth=1.0)
    plt.plot(t, delta_roll, label=f"delta roll (w={p.rolling_window})", linewidth=1.0)
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots, f"{prefix}_branch_clean.png"), dpi=180, bbox_inches="tight")
    plt.close()

    phi = ph["phi_centers"]
    caps_in = ph["caps_in"]
    caps_out = ph["caps_out"]
    p_in = ph["p_in"].copy()
    p_out = ph["p_out"].copy()
    delta = ph["delta"].copy()
    mask_phi = (caps_in >= p.min_captures_phase) & (caps_out >= p.min_captures_phase)
    p_in[~mask_phi] = np.nan
    p_out[~mask_phi] = np.nan
    delta[~mask_phi] = np.nan

    plt.figure()
    plt.plot(phi, p_in, marker="o", linewidth=1.0, label="pM tuned vs phase")
    plt.plot(phi, p_out, marker="o", linewidth=1.0, label="pM out vs phase")
    plt.plot(phi, delta, marker="o", linewidth=1.0, label="delta vs phase")
    plt.xlabel("drive phase (rad)")
    plt.legend()
    plt.savefig(os.path.join(plots, f"{prefix}_phase_response.png"), dpi=180, bbox_inches="tight")
    plt.close()


def run_sim(
    p: Params,
    seed: int,
    *,
    shell: bool,
    neg_mode: str,
) -> dict:
    """
    neg_mode:
      - "off": negative half-cycle does nothing extra (pure baseline mechanics, but shell can still be on)
      - "mem": negative half-cycle adds extra bookkeeping to inside edges via neg_mem_gain
      - "bw":  negative half-cycle tightens bandwidth on inside edges via neg_bw_gain
      - "mem+bw": both
    """
    neg_mode = neg_mode.lower().strip()
    neg_mem = ("mem" in neg_mode) and (p.neg_mem_gain > 0.0)
    neg_bw = ("bw" in neg_mode) and (p.neg_bw_gain > 0.0)

    rng = np.random.default_rng(seed)

    u, v = build_erdos_renyi(rng, p.N, p.p_edge)
    E = u.size
    theta = rng.uniform(-np.pi, np.pi, size=E).astype(np.float64)
    m_e = np.zeros(E, dtype=np.float64)

    occ_G = np.zeros(p.N, dtype=np.int32)
    occ_M = np.zeros(p.N, dtype=np.int32)

    region = choose_tuned_region(rng, p.N, p.drive_frac)

    if E > 0:
        ru = region[u].astype(np.int8)
        rv = region[v].astype(np.int8)
        edge_inside = (ru == 1) & (rv == 1)
        edge_boundary = (ru ^ rv) == 1
    else:
        edge_inside = np.zeros(0, dtype=bool)
        edge_boundary = np.zeros(0, dtype=bool)

    neigh_edges = [[] for _ in range(p.N)]
    for ei in range(E):
        a = int(u[ei]); b = int(v[ei])
        neigh_edges[a].append(ei)
        neigh_edges[b].append(ei)

    T = p.T
    captures_total = np.zeros(T, dtype=np.int32)
    captures_in = np.zeros(T, dtype=np.int32)
    captures_out = np.zeros(T, dtype=np.int32)

    M_total = np.zeros(T, dtype=np.int32)
    M_in = np.zeros(T, dtype=np.int32)
    M_out = np.zeros(T, dtype=np.int32)

    locked_like = np.zeros(T, dtype=np.float64)
    mean_edge_mem = np.zeros(T, dtype=np.float64)
    mean_theta_step = np.zeros(T, dtype=np.float64)

    B = int(max(8, p.phase_bins))
    phi_caps_in = np.zeros(B, dtype=np.int64)
    phi_caps_out = np.zeros(B, dtype=np.int64)
    phi_M_in = np.zeros(B, dtype=np.int64)
    phi_M_out = np.zeros(B, dtype=np.int64)

    for t in range(T):
        phase_arg = p.drive_w * t + p.drive_phase
        s = float(np.sin(phase_arg))
        drive = p.drive_amp * s
        pos = max(0.0, drive)
        neg = max(0.0, -drive)
        neg_norm = neg / (abs(p.drive_amp) + 1e-12)  # in [0,1]

        phi = float(phase_arg % TAU)
        b = int(np.floor((phi / TAU) * B))
        b = min(b, B - 1)

        # Edge updates (gauge register + edge memory)
        if E > 0:
            dtheta_abs_sum = 0.0
            for ei in range(E):
                inside = bool(edge_inside[ei])
                boundary = bool(edge_boundary[ei]) and shell

                bw_mult = 1.0
                mem_mult = 1.0
                if inside:
                    bw_mult *= p.region_bandwidth_mult
                    mem_mult *= p.region_mem_mult
                if boundary:
                    bw_mult *= p.boundary_bandwidth_mult
                    mem_mult *= p.boundary_mem_mult

                # Negative-drive bandwidth tightening INSIDE only (HSF "pressure")
                if inside and neg_bw:
                    bw_mult *= max(0.05, 1.0 - p.neg_bw_gain * neg_norm)

                bw = p.bandwidth * bw_mult
                mc = p.mem_couple * mem_mult

                a = int(u[ei]); bnode = int(v[ei])
                da = (occ_G[a] + occ_M[a]) - (occ_G[bnode] + occ_M[bnode])

                throttle = np.exp(-p.kappa * m_e[ei])
                flow = throttle * np.tanh(0.25 * da)
                if abs(flow) > bw:
                    flow = np.sign(flow) * bw

                dth = p.g_coup * flow
                if inside:
                    dth += (drive * 0.15)
                if abs(dth) > p.theta_bw:
                    dth = np.sign(dth) * p.theta_bw

                theta[ei] = wrap_pi_scalar(theta[ei] + dth)
                dtheta_abs_sum += abs(dth)

                transported = abs(flow) + abs(dth)
                m_e[ei] = (1.0 - p.mem_decay) * m_e[ei] + mc * transported

                # Negative-drive bookkeeping load INSIDE only (HSF "ledger cost")
                if inside and neg_mem:
                    m_e[ei] += p.neg_mem_gain * neg_norm

            mean_theta_step[t] = dtheta_abs_sum / max(E, 1)
            mean_edge_mem[t] = float(m_e.mean())
            thr = np.quantile(m_e, 0.85) if E > 5 else (m_e.mean() + 1e-9)
            locked_like[t] = float(np.mean(m_e >= thr))

        # Events (uniform Poisson; this is what behaved best in your ablations)
        events = rng.poisson(lam=p.lambda_capture, size=p.N).astype(np.int32)
        captures_total[t] = int(events.sum())

        for i in range(p.N):
            k = int(events[i])
            if k <= 0:
                continue

            in_reg = (region[i] == 1)
            if in_reg:
                captures_in[t] += k
                phi_caps_in[b] += k
            else:
                captures_out[t] += k
                phi_caps_out[b] += k

            local_mem = float(np.mean(m_e[np.array(neigh_edges[i], dtype=np.int32)])) if neigh_edges[i] else 0.0

            drive_term = 0.0
            if in_reg:
                # THE WORKING MECHANISM: positive drive lowers barrier; negative does not directly anti-select
                drive_term = (p.drive_gain_plus * pos) / (1.0 + local_mem)

            pM = branch_prob_metastable(p.barrier_M, drive_term, p.base_bias)
            m_count = rng.binomial(n=k, p=pM)

            M_total[t] += int(m_count)
            if in_reg:
                M_in[t] += int(m_count)
                phi_M_in[b] += int(m_count)
            else:
                M_out[t] += int(m_count)
                phi_M_out[b] += int(m_count)

            # Occupancy bookkeeping for flow dynamics (G is the complement)
            g_count = k - m_count
            occ_M[i] += int(m_count)
            occ_G[i] += int(g_count)

        # Decays
        if p.tau_G > 0:
            decG = rng.binomial(n=occ_G, p=min(1.0, 1.0 / p.tau_G))
            occ_G -= decG
        if p.tau_M > 0:
            decM = rng.binomial(n=occ_M, p=min(1.0, 1.0 / p.tau_M))
            occ_M -= decM

    return dict(
        u=u, v=v, theta_end=theta, mem_e_end=m_e, region=region,
        captures_total=captures_total, captures_in=captures_in, captures_out=captures_out,
        M_total=M_total, M_in=M_in, M_out=M_out,
        locked_like=locked_like, mean_edge_mem=mean_edge_mem, mean_theta_step=mean_theta_step,
        phi_bins=np.int32(B), phi_caps_in=phi_caps_in, phi_caps_out=phi_caps_out, phi_M_in=phi_M_in, phi_M_out=phi_M_out,
    )


# ----------------------------
# Multi-seed + sweep machinery
# ----------------------------

@dataclass
class SeedResult:
    seed: int
    mode: str
    delta: float
    se: float
    z: float
    cap_in: int
    cap_out: int
    M_in: int
    M_out: int
    out_dir: str
    sweep_k: str
    sweep_v: str


def stats_from_data(data: dict) -> Tuple[float, float, float, int, int, int, int]:
    cap_in = int(np.sum(data["captures_in"]))
    cap_out = int(np.sum(data["captures_out"]))
    M_in = int(np.sum(data["M_in"]))
    M_out = int(np.sum(data["M_out"]))
    delta, se, z = final_stats_counts(cap_in, cap_out, M_in, M_out)
    return delta, se, z, cap_in, cap_out, M_in, M_out


def pooled_stats(results: List[SeedResult]) -> Tuple[float, float, float, int, int, int, int]:
    cap_in = sum(r.cap_in for r in results)
    cap_out = sum(r.cap_out for r in results)
    M_in = sum(r.M_in for r in results)
    M_out = sum(r.M_out for r in results)
    delta, se, z = final_stats_counts(cap_in, cap_out, M_in, M_out)
    return delta, se, z, cap_in, cap_out, M_in, M_out


def write_summary_csv(path: str, rows: List[SeedResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sweep_k", "sweep_v", "seed", "mode", "delta", "se", "z", "cap_in", "cap_out", "M_in", "M_out", "seed_out_dir"])
        for r in rows:
            w.writerow([r.sweep_k, r.sweep_v, r.seed, r.mode, f"{r.delta:.8f}", f"{r.se:.8f}", f"{r.z:.4f}", r.cap_in, r.cap_out, r.M_in, r.M_out, r.out_dir])


def write_pooled_csv(path: str, pooled_rows: List[Dict[str, Any]]) -> None:
    if not pooled_rows:
        return
    keys = list(pooled_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in pooled_rows:
            w.writerow([r.get(k, "") for k in keys])


def maybe_save_seed_artifacts(seed_dir: str, data: dict, p: Params) -> None:
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(os.path.join(seed_dir, "plots"), exist_ok=True)
    series = compute_series(data, p.rolling_window)
    ph = phase_stats(data)
    np.savez_compressed(os.path.join(seed_dir, "results.npz"), **data, **series, **ph)
    write_csv(os.path.join(seed_dir, "log.csv"), data, series)
    write_phase_csv(os.path.join(seed_dir, "phase_bins.csv"), ph, p.min_captures_phase)
    plot_outputs(seed_dir, data, series, ph, p, prefix="run")


def _worker_seed(args: Tuple[int, str, Params, bool, str, str, str, bool]) -> SeedResult:
    seed, mode, p, shell, neg_mode, out_root, sweep_k, save_artifacts = args

    data = run_sim(p, seed, shell=shell, neg_mode=neg_mode)
    delta, se, z, cap_in, cap_out, M_in, M_out = stats_from_data(data)

    seed_dir = os.path.join(out_root, f"{sweep_k}_{safe_tag(mode)}", f"seed_{seed:05d}")
    if save_artifacts:
        maybe_save_seed_artifacts(seed_dir, data, p)

    return SeedResult(
        seed=seed,
        mode=mode,
        delta=delta,
        se=se,
        z=z,
        cap_in=cap_in,
        cap_out=cap_out,
        M_in=M_in,
        M_out=M_out,
        out_dir=seed_dir if save_artifacts else "",
        sweep_k=sweep_k,
        sweep_v="",  # filled by wrapper
    )


def apply_sweep(p: Params, name: str, value: float) -> Params:
    if not hasattr(p, name):
        raise ValueError(f"Unknown sweep param: {name}")
    # dataclass is frozen; use replace
    return replace(p, **{name: type(getattr(p, name))(value) if isinstance(getattr(p, name), int) else value})


def main():
    ap = argparse.ArgumentParser()

    # graph
    ap.add_argument("--N", type=int, default=250)
    ap.add_argument("--p_edge", type=float, default=0.03)

    # time
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=1.0)

    # dynamics
    ap.add_argument("--bandwidth", type=float, default=0.08)
    ap.add_argument("--mem_decay", type=float, default=0.0005)
    ap.add_argument("--mem_couple", type=float, default=1.8)
    ap.add_argument("--kappa", type=float, default=2.5)

    # gauge
    ap.add_argument("--g_coup", type=float, default=0.20)
    ap.add_argument("--theta_bw", type=float, default=0.10)

    # events
    ap.add_argument("--lambda_capture", type=float, default=0.09)
    ap.add_argument("--barrier_M", type=float, default=4.0)
    ap.add_argument("--base_bias", type=float, default=0.0)

    # lifetimes
    ap.add_argument("--tau_G", type=float, default=45.0)
    ap.add_argument("--tau_M", type=float, default=900.0)

    # drive
    ap.add_argument("--drive_frac", type=float, default=0.20)
    ap.add_argument("--drive_amp", type=float, default=0.35)
    ap.add_argument("--drive_w", type=float, default=0.03)
    ap.add_argument("--drive_phase", type=float, default=0.0)
    ap.add_argument("--drive_gain_plus", type=float, default=2.2)

    # negative-drive bookkeeping (HSF-clean knobs)
    ap.add_argument("--neg_mem_gain", type=float, default=0.10)
    ap.add_argument("--neg_bw_gain", type=float, default=0.0)
    ap.add_argument("--neg_mode", type=str, default="mem", choices=["off", "mem", "bw", "mem+bw"])

    # shell multipliers
    ap.add_argument("--region_bandwidth_mult", type=float, default=1.10)
    ap.add_argument("--region_mem_mult", type=float, default=0.95)
    ap.add_argument("--boundary_bandwidth_mult", type=float, default=0.70)
    ap.add_argument("--boundary_mem_mult", type=float, default=1.35)

    # diagnostics
    ap.add_argument("--rolling_window", type=int, default=1000)
    ap.add_argument("--min_captures_plot", type=int, default=50)
    ap.add_argument("--phase_bins", type=int, default=24)
    ap.add_argument("--min_captures_phase", type=int, default=400)

    # multicore seeds
    ap.add_argument("--seeds", type=str, default="0:9")
    ap.add_argument("--n_jobs", type=int, default=0)

    # compare + storage
    ap.add_argument("--compare_baseline", action="store_true")
    ap.add_argument("--save_seed_artifacts", action="store_true", help="for sweeps, saves per-seed plots/files (large disk use)")

    # sweep
    ap.add_argument("--sweep_param", type=str, default="")
    ap.add_argument("--sweep_values", type=str, default="")
    ap.add_argument("--sweep_param2", type=str, default="")
    ap.add_argument("--sweep_values2", type=str, default="")

    # output
    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="metastable_toy_v8")

    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds parsed.")
    ncpu = os.cpu_count() or 1
    if args.n_jobs <= 0:
        n_jobs = max(1, ncpu - 1)
    else:
        n_jobs = max(1, int(args.n_jobs))

    base_p = Params(
        N=args.N, p_edge=args.p_edge,
        T=args.T, dt=args.dt,
        bandwidth=args.bandwidth, mem_decay=args.mem_decay, mem_couple=args.mem_couple, kappa=args.kappa,
        g_coup=args.g_coup, theta_bw=args.theta_bw,
        lambda_capture=args.lambda_capture, barrier_M=args.barrier_M, base_bias=args.base_bias,
        tau_G=args.tau_G, tau_M=args.tau_M,
        drive_frac=args.drive_frac, drive_amp=args.drive_amp, drive_w=args.drive_w, drive_phase=args.drive_phase,
        drive_gain_plus=args.drive_gain_plus,
        neg_mem_gain=args.neg_mem_gain, neg_bw_gain=args.neg_bw_gain,
        region_bandwidth_mult=args.region_bandwidth_mult, region_mem_mult=args.region_mem_mult,
        boundary_bandwidth_mult=args.boundary_bandwidth_mult, boundary_mem_mult=args.boundary_mem_mult,
        rolling_window=max(50, int(args.rolling_window)),
        min_captures_plot=max(1, int(args.min_captures_plot)),
        phase_bins=max(8, int(args.phase_bins)),
        min_captures_phase=max(1, int(args.min_captures_phase)),
    )

    do_sweep = bool(args.sweep_param.strip()) and bool(args.sweep_values.strip())
    do_sweep2 = do_sweep and bool(args.sweep_param2.strip()) and bool(args.sweep_values2.strip())

    sweep_vals1 = parse_float_list(args.sweep_values) if do_sweep else []
    sweep_vals2 = parse_float_list(args.sweep_values2) if do_sweep2 else []

    tag = (
        f"N{base_p.N}_p{base_p.p_edge}_T{base_p.T}"
        f"_lam{base_p.lambda_capture}_bar{base_p.barrier_M}"
        f"_drvF{base_p.drive_frac}_A{base_p.drive_amp}_w{base_p.drive_w}_gP{base_p.drive_gain_plus}"
        f"_neg{args.neg_mode}_nm{base_p.neg_mem_gain}_nb{base_p.neg_bw_gain}"
        f"_bwb{base_p.boundary_bandwidth_mult}_memb{base_p.boundary_mem_mult}"
        f"_seeds{safe_tag(args.seeds)}_jobs{n_jobs}"
        f"{'_SWEEP' if do_sweep else ''}_{args.run_name}"
    )
    run_dir = make_run_dir(args.out_dir, tag)
    seed_root = os.path.join(run_dir, "seed_runs")

    print("============================================================")
    print("HSF metastable toy v8 — MULTI-SEED" + (" — SWEEP" if do_sweep else ""))
    print("------------------------------------------------------------")
    print(f"seeds   = {seeds}")
    print(f"n_jobs  = {n_jobs} (cpu_count={ncpu})")
    print(f"run_dir = {run_dir}")
    print("------------------------------------------------------------")
    print("v8 mechanism:")
    print("  barrier: pos-drive only (working bias)")
    print(f"  negative-drive bookkeeping: mode={args.neg_mode}  neg_mem_gain={base_p.neg_mem_gain}  neg_bw_gain={base_p.neg_bw_gain}")
    print("  shell: ON for v8, OFF for baseline compare")
    if args.compare_baseline:
        print("  baseline: pos-drive only, shell OFF, negative bookkeeping OFF")
    if do_sweep:
        print("------------------------------------------------------------")
        print(f"sweep_param  = {args.sweep_param} values={sweep_vals1}")
        if do_sweep2:
            print(f"sweep_param2 = {args.sweep_param2} values={sweep_vals2}")
    print("============================================================")

    ctx = mp.get_context("spawn")

    all_seed_rows: List[SeedResult] = []
    pooled_rows: List[Dict[str, Any]] = []

    def run_point(p_point: Params, sweep_label: str, sweep_v1: str, sweep_v2: str) -> None:
        nonlocal all_seed_rows, pooled_rows

        # v8 mode: shell ON, neg_mode per args
        work_v8 = []
        for s in seeds:
            work_v8.append((s, "v8", p_point, True, args.neg_mode, seed_root, sweep_label, args.save_seed_artifacts))

        # baseline mode (optional): shell OFF and neg_mode off (strict baseline)
        work_base = []
        if args.compare_baseline:
            p_base = replace(p_point, neg_mem_gain=0.0, neg_bw_gain=0.0)  # ensure baseline isn't using bookkeeping knobs
            for s in seeds:
                work_base.append((s, "baseline", p_base, False, "off", seed_root, sweep_label, args.save_seed_artifacts))

        with ctx.Pool(processes=n_jobs) as pool:
            v8_res = pool.map(_worker_seed, work_v8)
            base_res = pool.map(_worker_seed, work_base) if work_base else []

        # fill sweep values
        for r in v8_res:
            r.sweep_v = f"{sweep_v1}{(' | ' + sweep_v2) if sweep_v2 else ''}"
        for r in base_res:
            r.sweep_v = f"{sweep_v1}{(' | ' + sweep_v2) if sweep_v2 else ''}"

        all_seed_rows.extend(v8_res)
        all_seed_rows.extend(base_res)

        dv8, sev8, zv8, cap_in_v8, cap_out_v8, Min_v8, Mout_v8 = pooled_stats(v8_res)

        row = dict(
            sweep_k=sweep_label,
            sweep_v=f"{sweep_v1}{(' | ' + sweep_v2) if sweep_v2 else ''}",
            mode="v8",
            delta=dv8,
            se=sev8,
            z=zv8,
            cap_in=cap_in_v8,
            cap_out=cap_out_v8,
            M_in=Min_v8,
            M_out=Mout_v8,
        )

        if base_res:
            db, seb, zb, cap_in_b, cap_out_b, Min_b, Mout_b = pooled_stats(base_res)
            row.update(dict(
                delta_baseline=db,
                se_baseline=seb,
                z_baseline=zb,
                delta_improve=dv8 - db,
                z_improve=zv8 - zb,
            ))
        pooled_rows.append(row)

        # print pooled for this point
        print("------------------------------------------------------------")
        print(f"POOLED — {sweep_label} = {row['sweep_v']}")
        print(f"  v8:    delta={dv8:+.6f}  se={sev8:.6f}  z={zv8:+.3f}")
        if base_res:
            print(f"  base:  delta={row['delta_baseline']:+.6f}  se={row['se_baseline']:.6f}  z={row['z_baseline']:+.3f}")
            print(f"  Δimpr: {row['delta_improve']:+.6f}   Δz: {row['z_improve']:+.3f}")

    if not do_sweep:
        # Single point run
        run_point(base_p, "nosweep", "default", "")
    else:
        # Sweep 1D or 2D
        name1 = args.sweep_param.strip()
        name2 = args.sweep_param2.strip() if do_sweep2 else ""

        if not do_sweep2:
            for v1 in sweep_vals1:
                p1 = apply_sweep(base_p, name1, v1)
                run_point(p1, name1, f"{name1}={v1}", "")
        else:
            for v1 in sweep_vals1:
                for v2 in sweep_vals2:
                    p1 = apply_sweep(base_p, name1, v1)
                    p2 = apply_sweep(p1, name2, v2)
                    run_point(p2, f"{name1}&{name2}", f"{name1}={v1}", f"{name2}={v2}")

    # Write CSV outputs
    summary_csv = os.path.join(run_dir, "summary.csv")
    pooled_csv = os.path.join(run_dir, "pooled_summary.csv")
    write_summary_csv(summary_csv, all_seed_rows)
    write_pooled_csv(pooled_csv, pooled_rows)

    print("============================================================")
    print("DONE")
    print("------------------------------------------------------------")
    print("summary.csv       :", summary_csv)
    print("pooled_summary.csv:", pooled_csv)
    if not do_sweep or args.save_seed_artifacts:
        print("seed_runs dir     :", seed_root)
    else:
        print("seed_runs dir     : (not saved; use --save_seed_artifacts to write per-seed plots/files)")
    print("run_dir           :", run_dir)
    print("============================================================")


if __name__ == "__main__":
    main()
