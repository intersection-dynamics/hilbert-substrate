# hsf_metastable_branch_toy_v7.py
# ------------------------------------------------------------
# v7: HSF toy model with LINK-LEVEL RECTIFICATION + MULTI-SEED MULTICORE RUNNER
#
# What changed vs v6:
#   - STOP trying to rectify via event timing (eps_lambda defaults to 0).
#   - Add "substrate diode" at the BRANCH COUPLING level:
#       drive_term = (gain_plus * max(0, drive) - gain_minus * max(0, -drive)) / (1 + local_mem)
#     If gain_minus < gain_plus, negative half-cycle hurts less than positive helps -> net bias possible
#     while still being "signed physics".
#
# Still included:
#   - boundary shell (bandwidth bottleneck + stickier memory)
#   - gauge phase register theta_e, edge memory m_e
#   - cumulative + rolling fractions, phase plots
#
# New major feature:
#   - Run many seeds in parallel on CPU using multiprocessing:
#       --seeds 0:9
#       --n_jobs 8
#     Produces:
#       - per-seed console stats
#       - pooled stats (weighted by captures)
#       - per-seed results in run_dir/seed_runs/
#       - a single summary CSV and pooled plots
#
# SAFE MODEL (generic branching, non-nuclear).
#
# Example:
#   python hsf_metastable_branch_toy_v7.py --N 250 --p_edge 0.03 --T 20000 --lambda_capture 0.09 \
#     --drive_frac 0.20 --drive_amp 0.35 --drive_w 0.03 --drive_gain_plus 2.2 --drive_gain_minus 0.6 \
#     --boundary_bandwidth_mult 0.70 --boundary_mem_mult 1.35 \
#     --seeds 0:9 --n_jobs 8 --compare_baseline --run_name v7_multiseed
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import os
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

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
    """
    Accepts:
      "0:9"  -> [0..9]
      "0:18:2" -> [0,2,4,...,18]
      "0,1,2,5,8"
      "7" -> [7]
    """
    s = spec.strip()
    if "," in s:
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
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
        if step == 0:
            raise ValueError("step cannot be 0")
        if a <= b and step > 0:
            return list(range(a, b + 1, step))
        if a >= b and step < 0:
            return list(range(a, b - 1, step))
        # if user gives descending without negative step, we still handle
        if a > b and step > 0:
            return list(range(a, b - 1, -step))
        return list(range(a, b + 1, step))
    return [int(s)]


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

@dataclass
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

    # diode coupling to barrier
    drive_gain_plus: float
    drive_gain_minus: float

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

    # output
    out_dir: str
    run_name: str


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

    return {
        "frac_in_cum": frac_in_cum,
        "frac_out_cum": frac_out_cum,
        "frac_tot_cum": frac_tot_cum,
        "delta_cum": delta_cum,

        "cap_in_roll": cap_in_roll,
        "cap_out_roll": cap_out_roll,
        "cap_tot_roll": cap_tot_roll,
        "frac_in_roll": frac_in_roll,
        "frac_out_roll": frac_out_roll,
        "frac_tot_roll": frac_tot_roll,
        "delta_roll": delta_roll,
    }


def final_stats_counts(cap_in: int, cap_out: int, M_in: int, M_out: int) -> Tuple[float, float, float]:
    eps = 1e-18
    p_in = M_in / (cap_in + eps)
    p_out = M_out / (cap_out + eps)
    delta = p_in - p_out
    se = math.sqrt((p_in * (1.0 - p_in) / (cap_in + eps)) + (p_out * (1.0 - p_out) / (cap_out + eps)))
    z = delta / (se + eps)
    return float(delta), float(se), float(z)


def final_stats(data: dict) -> Tuple[float, float, float, int, int, int, int]:
    cap_in = int(np.sum(data["captures_in"]))
    cap_out = int(np.sum(data["captures_out"]))
    M_in = int(np.sum(data["M_in"]))
    M_out = int(np.sum(data["M_out"]))
    delta, se, z = final_stats_counts(cap_in, cap_out, M_in, M_out)
    return delta, se, z, cap_in, cap_out, M_in, M_out


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


def plot_outputs(run_dir: str, data: dict, series: dict, ph: dict, p: Params, prefix: str) -> None:
    plots = os.path.join(run_dir, "plots")
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


def run_sim(p: Params, seed: int, *, signed_diode: bool, shell: bool) -> dict:
    rng = np.random.default_rng(seed)

    # graph + region fixed per seed for comparability
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
    G_total = np.zeros(T, dtype=np.int32)
    G_in = np.zeros(T, dtype=np.int32)
    G_out = np.zeros(T, dtype=np.int32)

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
        phi = float(phase_arg % TAU)
        b = int(np.floor((phi / TAU) * B))
        b = min(b, B - 1)

        # edge updates
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

            mean_theta_step[t] = dtheta_abs_sum / max(E, 1)
            mean_edge_mem[t] = float(m_e.mean())
            thr = np.quantile(m_e, 0.85) if E > 5 else (m_e.mean() + 1e-9)
            locked_like[t] = float(np.mean(m_e >= thr))

        # events (uniform, as per what worked in your ablations)
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
                if signed_diode:
                    # NEW: barrier coupling diode (rectified, but still "signed")
                    pos = max(0.0, drive)
                    neg = max(0.0, -drive)
                    drive_term = (p.drive_gain_plus * pos - p.drive_gain_minus * neg) / (1.0 + local_mem)
                else:
                    # baseline from earlier work: unsigned drive (only helps)
                    drive_term = (p.drive_gain_plus * max(0.0, drive)) / (1.0 + local_mem)

            pM = branch_prob_metastable(p.barrier_M, drive_term, p.base_bias)
            m_count = rng.binomial(n=k, p=pM)
            g_count = k - m_count

            occ_M[i] += int(m_count)
            occ_G[i] += int(g_count)

            M_total[t] += int(m_count)
            G_total[t] += int(g_count)

            if in_reg:
                M_in[t] += int(m_count)
                G_in[t] += int(g_count)
                phi_M_in[b] += int(m_count)
            else:
                M_out[t] += int(m_count)
                G_out[t] += int(g_count)
                phi_M_out[b] += int(m_count)

        # decays
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
        G_total=G_total, G_in=G_in, G_out=G_out,
        locked_like=locked_like, mean_edge_mem=mean_edge_mem, mean_theta_step=mean_theta_step,
        phi_bins=np.int32(B), phi_caps_in=phi_caps_in, phi_caps_out=phi_caps_out, phi_M_in=phi_M_in, phi_M_out=phi_M_out,
    )


# ----------------------------
# Multiseed runner + pooling
# ----------------------------

@dataclass
class SeedResult:
    seed: int
    mode: str  # "v7" or "baseline"
    delta: float
    se: float
    z: float
    cap_in: int
    cap_out: int
    M_in: int
    M_out: int
    run_dir: str


def _worker_one(args: Tuple[int, str, Params, bool, bool, str]) -> SeedResult:
    seed, mode, p, signed_diode, shell, out_subdir = args
    data = run_sim(p, seed, signed_diode=signed_diode, shell=shell)
    delta, se, z, cap_in, cap_out, M_in, M_out = final_stats(data)

    seed_dir = os.path.join(out_subdir, f"seed_{seed:05d}_{mode}")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(os.path.join(seed_dir, "plots"), exist_ok=True)

    series = compute_series(data, p.rolling_window)
    ph = phase_stats(data)

    np.savez_compressed(os.path.join(seed_dir, "results.npz"), **data, **series, **ph)
    write_csv(os.path.join(seed_dir, "log.csv"), data, series)
    write_phase_csv(os.path.join(seed_dir, "phase_bins.csv"), ph, p.min_captures_phase)
    plot_outputs(seed_dir, data, series, ph, p, prefix="run")

    return SeedResult(seed, mode, delta, se, z, cap_in, cap_out, M_in, M_out, seed_dir)


def pooled_stats(results: List[SeedResult]) -> Tuple[float, float, float, int, int, int, int]:
    cap_in = sum(r.cap_in for r in results)
    cap_out = sum(r.cap_out for r in results)
    M_in = sum(r.M_in for r in results)
    M_out = sum(r.M_out for r in results)
    delta, se, z = final_stats_counts(cap_in, cap_out, M_in, M_out)
    return delta, se, z, cap_in, cap_out, M_in, M_out


def write_summary_csv(path: str, results_v7: List[SeedResult], results_base: Optional[List[SeedResult]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "mode", "delta", "se", "z", "cap_in", "cap_out", "M_in", "M_out", "seed_run_dir"])
        for r in results_v7:
            w.writerow([r.seed, r.mode, f"{r.delta:.8f}", f"{r.se:.8f}", f"{r.z:.4f}", r.cap_in, r.cap_out, r.M_in, r.M_out, r.run_dir])
        if results_base is not None:
            for r in results_base:
                w.writerow([r.seed, r.mode, f"{r.delta:.8f}", f"{r.se:.8f}", f"{r.z:.4f}", r.cap_in, r.cap_out, r.M_in, r.M_out, r.run_dir])


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

    # tuned region / drive
    ap.add_argument("--drive_frac", type=float, default=0.20)
    ap.add_argument("--drive_amp", type=float, default=0.35)
    ap.add_argument("--drive_w", type=float, default=0.03)
    ap.add_argument("--drive_phase", type=float, default=0.0)

    # diode barrier coupling
    ap.add_argument("--drive_gain_plus", type=float, default=2.2)
    ap.add_argument("--drive_gain_minus", type=float, default=0.6)

    # boundary shell
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
    ap.add_argument("--seeds", type=str, default="0", help='e.g. "0:9" or "0:18:2" or "0,1,2,5"')
    ap.add_argument("--n_jobs", type=int, default=0, help="0 means use os.cpu_count()-1 (min 1)")

    # baseline compare
    ap.add_argument("--compare_baseline", action="store_true")

    # output
    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="metastable_toy_v7")

    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds parsed.")

    ncpu = os.cpu_count() or 1
    if args.n_jobs <= 0:
        n_jobs = max(1, ncpu - 1)
    else:
        n_jobs = max(1, int(args.n_jobs))

    p = Params(
        N=args.N,
        p_edge=args.p_edge,
        T=args.T,
        dt=args.dt,

        bandwidth=args.bandwidth,
        mem_decay=args.mem_decay,
        mem_couple=args.mem_couple,
        kappa=args.kappa,

        g_coup=args.g_coup,
        theta_bw=args.theta_bw,

        lambda_capture=args.lambda_capture,
        barrier_M=args.barrier_M,
        base_bias=args.base_bias,

        tau_G=args.tau_G,
        tau_M=args.tau_M,

        drive_frac=args.drive_frac,
        drive_amp=args.drive_amp,
        drive_w=args.drive_w,
        drive_phase=args.drive_phase,

        drive_gain_plus=args.drive_gain_plus,
        drive_gain_minus=args.drive_gain_minus,

        region_bandwidth_mult=args.region_bandwidth_mult,
        region_mem_mult=args.region_mem_mult,
        boundary_bandwidth_mult=args.boundary_bandwidth_mult,
        boundary_mem_mult=args.boundary_mem_mult,

        rolling_window=max(50, int(args.rolling_window)),
        min_captures_plot=max(1, int(args.min_captures_plot)),
        phase_bins=max(8, int(args.phase_bins)),
        min_captures_phase=max(1, int(args.min_captures_phase)),

        out_dir=args.out_dir,
        run_name=args.run_name,
    )

    tag = (
        f"N{p.N}_p{p.p_edge}_T{p.T}"
        f"_lam{p.lambda_capture}_bar{p.barrier_M}"
        f"_drvF{p.drive_frac}_A{p.drive_amp}_w{p.drive_w}"
        f"_gP{p.drive_gain_plus}_gM{p.drive_gain_minus}"
        f"_bwb{p.boundary_bandwidth_mult}_memb{p.boundary_mem_mult}"
        f"_seeds{safe_tag(args.seeds)}_jobs{n_jobs}_{p.run_name}"
    )
    run_dir = make_run_dir(p.out_dir, tag)
    seed_root = os.path.join(run_dir, "seed_runs")

    print("============================================================")
    print("HSF metastable toy v7 — MULTI-SEED RUN")
    print("------------------------------------------------------------")
    print(f"seeds     = {seeds}")
    print(f"n_jobs    = {n_jobs} (cpu_count={ncpu})")
    print(f"run_dir   = {run_dir}")
    print("------------------------------------------------------------")
    print("Model:")
    print("  v7: diode barrier coupling (gain_plus*pos - gain_minus*neg), shell ON")
    if args.compare_baseline:
        print("  baseline: unsigned drive (pos only), shell OFF")
    print("============================================================")

    # Build work list
    work_v7 = [(s, "v7", p, True, True, seed_root) for s in seeds]
    work_base = [(s, "baseline", p, False, False, seed_root) for s in seeds] if args.compare_baseline else []

    # Run multiprocess (spawn-safe on Windows)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_jobs) as pool:
        results_v7 = pool.map(_worker_one, work_v7)
        results_base = pool.map(_worker_one, work_base) if work_base else None

    # Sort by seed
    results_v7.sort(key=lambda r: r.seed)
    if results_base is not None:
        results_base.sort(key=lambda r: r.seed)

    # Per-seed print
    print("============================================================")
    print("PER-SEED RESULTS — v7")
    print("------------------------------------------------------------")
    for r in results_v7:
        print(f"seed={r.seed:4d}  delta={r.delta:+.6f}  se={r.se:.6f}  z={r.z:+.3f}  cap_in={r.cap_in} cap_out={r.cap_out}")
    print("============================================================")

    if results_base is not None:
        print("============================================================")
        print("PER-SEED RESULTS — baseline")
        print("------------------------------------------------------------")
        for r in results_base:
            print(f"seed={r.seed:4d}  delta={r.delta:+.6f}  se={r.se:.6f}  z={r.z:+.3f}  cap_in={r.cap_in} cap_out={r.cap_out}")
        print("============================================================")

    # Pooled
    dv7, sev7, zv7, cap_in_v7, cap_out_v7, Min_v7, Mout_v7 = pooled_stats(results_v7)
    print("============================================================")
    print("POOLED (ALL SEEDS) — v7")
    print("------------------------------------------------------------")
    print(f"captures_in  = {cap_in_v7}")
    print(f"captures_out = {cap_out_v7}")
    print(f"M_in         = {Min_v7}")
    print(f"M_out        = {Mout_v7}")
    print(f"delta (p_in - p_out) = {dv7:+.6f}")
    print(f"SE(delta)           = {sev7:.6f}")
    print(f"z-score             = {zv7:+.3f}")
    print("============================================================")

    if results_base is not None:
        db, seb, zb, cap_in_b, cap_out_b, Min_b, Mout_b = pooled_stats(results_base)
        print("============================================================")
        print("POOLED (ALL SEEDS) — baseline")
        print("------------------------------------------------------------")
        print(f"captures_in  = {cap_in_b}")
        print(f"captures_out = {cap_out_b}")
        print(f"M_in         = {Min_b}")
        print(f"M_out        = {Mout_b}")
        print(f"delta (p_in - p_out) = {db:+.6f}")
        print(f"SE(delta)           = {seb:.6f}")
        print(f"z-score             = {zb:+.3f}")
        print("============================================================")

        print("============================================================")
        print("POOLED Δ IMPROVEMENT (v7 - baseline)")
        print("------------------------------------------------------------")
        print(f"delta_v7      = {dv7:+.6f}")
        print(f"delta_baseline= {db:+.6f}")
        print(f"Δ increase    = {dv7 - db:+.6f}")
        print(f"z_v7          = {zv7:+.3f}")
        print(f"z_baseline    = {zb:+.3f}")
        print(f"Δz            = {zv7 - zb:+.3f}")
        print("============================================================")

    # Save summary CSV
    summary_csv = os.path.join(run_dir, "summary.csv")
    write_summary_csv(summary_csv, results_v7, results_base)
    print("Summary CSV:", summary_csv)

    print("Run output:", run_dir)
    print("Seed runs in:", seed_root)


if __name__ == "__main__":
    main()
