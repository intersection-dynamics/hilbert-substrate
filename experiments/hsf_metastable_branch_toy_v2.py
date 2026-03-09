# hsf_metastable_branch_toy_v2.py
# ------------------------------------------------------------
# HSF-inspired toy model: bias selection into a metastable branch (v2)
#
# v2 improvements (requested):
#   - Plots CUMULATIVE + ROLLING branch fractions (trusted, low-noise views)
#   - Hides ticks where captures < min_captures_plot (for per-tick/rolling display)
#   - Prints final delta, standard error, and z-score (cumulative)
#   - Saves a clean branch plot you can trust
#
# SAFE MODEL (generic branching, non-nuclear).
#
# Typical run (Windows one-liner):
#   python hsf_metastable_branch_toy_v2.py --N 250 --p_edge 0.03 --T 20000 --lambda_capture 0.02 --drive_amp 0.35 --drive_w 0.03 --drive_frac 0.20 --seed 0
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_tag(s: str) -> str:
    return s.replace(" ", "_").replace(":", "_").replace("/", "_").replace(".", "p")


def make_run_dir(base_out: str, tag: str) -> str:
    run_dir = os.path.join(base_out, f"{now_stamp()}_{safe_tag(tag)}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    return run_dir


def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


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

    # gauge register
    g_coup: float
    theta_bw: float

    # event process
    lambda_capture: float
    barrier_M: float
    base_bias: float

    # lifetimes
    tau_G: float
    tau_M: float

    # tuned region
    drive_frac: float
    drive_amp: float
    drive_w: float
    drive_phase: float
    drive_gain: float
    region_bandwidth_mult: float
    region_mem_mult: float

    # plotting / stats
    rolling_window: int
    min_captures_plot: int

    # output
    out_dir: str
    run_name: str
    seed: int


def build_erdos_renyi(rng: np.random.Generator, N: int, p_edge: float) -> Tuple[np.ndarray, np.ndarray]:
    u_list = []
    v_list = []
    for i in range(N):
        r = rng.random(N - i - 1)
        js = np.where(r < p_edge)[0] + (i + 1)
        for j in js:
            u_list.append(i)
            v_list.append(int(j))
    if len(u_list) == 0:
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


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """
    Rolling window sum with window w. Output length == len(x).
    Values are sum over [t-w+1, t], truncated near start.
    """
    x = np.asarray(x, dtype=np.float64)
    if w <= 1:
        return x.copy()
    c = np.cumsum(x)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def run_sim(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)

    # graph
    u, v = build_erdos_renyi(rng, p.N, p.p_edge)
    E = u.size

    # edge registers
    theta = rng.uniform(-np.pi, np.pi, size=E).astype(np.float64)
    m_e = np.zeros(E, dtype=np.float64)

    # per-node occupancy
    occ_G = np.zeros(p.N, dtype=np.int32)
    occ_M = np.zeros(p.N, dtype=np.int32)

    # region mask
    region = choose_tuned_region(rng, p.N, p.drive_frac)

    # adjacency: edges incident to node
    neigh_edges = [[] for _ in range(p.N)]
    for ei in range(E):
        neigh_edges[int(u[ei])].append(ei)
        neigh_edges[int(v[ei])].append(ei)

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

    for t in range(T):
        drive = p.drive_amp * np.sin(p.drive_w * t + p.drive_phase)

        # edge-local bookkeeping + gauge update
        if E > 0:
            dtheta_abs_sum = 0.0
            for ei in range(E):
                a = int(u[ei])
                b = int(v[ei])
                in_reg = (region[a] == 1) or (region[b] == 1)

                bw = p.bandwidth * (p.region_bandwidth_mult if in_reg else 1.0)
                mc = p.mem_couple * (p.region_mem_mult if in_reg else 1.0)

                da = (occ_G[a] + occ_M[a]) - (occ_G[b] + occ_M[b])
                throttle = np.exp(-p.kappa * m_e[ei])
                flow = throttle * np.tanh(0.25 * da)
                if abs(flow) > bw:
                    flow = np.sign(flow) * bw

                j = flow
                dth = p.g_coup * j + (drive * 0.15 if in_reg else 0.0)
                if abs(dth) > p.theta_bw:
                    dth = np.sign(dth) * p.theta_bw

                theta[ei] = wrap_pi(np.array([theta[ei] + dth], dtype=np.float64))[0]
                dtheta_abs_sum += abs(dth)

                transported = abs(flow) + abs(dth)
                m_e[ei] = (1.0 - p.mem_decay) * m_e[ei] + mc * transported

            mean_theta_step[t] = float(dtheta_abs_sum / max(E, 1))
            mean_edge_mem[t] = float(m_e.mean())
            thr = np.quantile(m_e, 0.85) if E > 5 else (m_e.mean() + 1e-9)
            locked_like[t] = float(np.mean(m_e >= thr))
        else:
            mean_theta_step[t] = 0.0
            mean_edge_mem[t] = 0.0
            locked_like[t] = 0.0

        # capture-like events
        events = rng.poisson(lam=p.lambda_capture, size=p.N).astype(np.int32)
        captures_total[t] = int(events.sum())

        for i in range(p.N):
            k = int(events[i])
            if k <= 0:
                continue

            in_reg = (region[i] == 1)
            if in_reg:
                captures_in[t] += k
            else:
                captures_out[t] += k

            local_mem = float(np.mean(m_e[np.array(neigh_edges[i], dtype=np.int32)])) if neigh_edges[i] else 0.0

            drive_term = 0.0
            if in_reg:
                drive_term = p.drive_gain * max(0.0, drive) / (1.0 + local_mem)

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
            else:
                M_out[t] += int(m_count)
                G_out[t] += int(g_count)

        # decays
        if p.tau_G > 0:
            p_decay_G = min(1.0, 1.0 / p.tau_G)
            decG = rng.binomial(n=occ_G, p=p_decay_G)
            occ_G = occ_G - decG

        if p.tau_M > 0:
            p_decay_M = min(1.0, 1.0 / p.tau_M)
            decM = rng.binomial(n=occ_M, p=p_decay_M)
            occ_M = occ_M - decM

    # end maps
    occ_total_end = (occ_G + occ_M).astype(np.int32)
    occ_M_end = occ_M.astype(np.int32)

    return {
        "u": u, "v": v,
        "theta_end": theta,
        "mem_e_end": m_e,
        "region": region,

        "captures_total": captures_total,
        "captures_in": captures_in,
        "captures_out": captures_out,

        "M_total": M_total,
        "M_in": M_in,
        "M_out": M_out,

        "G_total": G_total,
        "G_in": G_in,
        "G_out": G_out,

        "locked_like": locked_like,
        "mean_edge_mem": mean_edge_mem,
        "mean_theta_step": mean_theta_step,

        "occ_total_end": occ_total_end,
        "occ_M_end": occ_M_end,
    }


def compute_series(data: dict, w: int) -> dict:
    """
    Compute per-tick, cumulative, and rolling branch fractions for M.
    """
    eps = 1e-12

    cap_in = data["captures_in"].astype(np.float64)
    cap_out = data["captures_out"].astype(np.float64)
    cap_tot = data["captures_total"].astype(np.float64)

    M_in = data["M_in"].astype(np.float64)
    M_out = data["M_out"].astype(np.float64)
    M_tot = data["M_total"].astype(np.float64)

    # per-tick fractions
    frac_in_tick = M_in / (cap_in + eps)
    frac_out_tick = M_out / (cap_out + eps)
    frac_tot_tick = M_tot / (cap_tot + eps)

    # cumulative fractions
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

    # rolling fractions
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
        "frac_in_tick": frac_in_tick,
        "frac_out_tick": frac_out_tick,
        "frac_tot_tick": frac_tot_tick,

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


def final_stats(data: dict) -> Tuple[float, float, float, int, int]:
    """
    Final cumulative delta, standard error (difference of proportions), z-score.
    """
    cap_in = int(np.sum(data["captures_in"]))
    cap_out = int(np.sum(data["captures_out"]))
    M_in = int(np.sum(data["M_in"]))
    M_out = int(np.sum(data["M_out"]))

    eps = 1e-18
    p_in = M_in / (cap_in + eps)
    p_out = M_out / (cap_out + eps)
    delta = p_in - p_out

    # standard error for difference in independent proportions
    se = np.sqrt((p_in * (1.0 - p_in) / (cap_in + eps)) + (p_out * (1.0 - p_out) / (cap_out + eps)))
    z = delta / (se + eps)

    return float(delta), float(se), float(z), cap_in, cap_out


def write_csv(path: str, data: dict, series: dict) -> None:
    T = len(data["captures_total"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "captures_total", "captures_in", "captures_out",
            "M_total", "M_in", "M_out",
            "G_total", "G_in", "G_out",
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
                int(data["G_total"][t]),
                int(data["G_in"][t]),
                int(data["G_out"][t]),
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


def plot_outputs(run_dir: str, data: dict, series: dict, p: Params) -> None:
    plots = os.path.join(run_dir, "plots")
    t = np.arange(len(data["captures_total"]))

    # substrate-ish stats
    plt.figure()
    plt.plot(t, data["mean_edge_mem"], label="mean edge memory")
    plt.plot(t, data["locked_like"], label="locked_like (p85 mem fraction)")
    plt.plot(t, data["mean_theta_step"], label="mean |dtheta|")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots, "timeseries.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # clean branch plot: cumulative + rolling (trusted)
    cap_in = data["captures_in"].astype(np.float64)
    cap_out = data["captures_out"].astype(np.float64)

    # "hide" ticks for rolling where window captures too small
    mask_roll = (series["cap_in_roll"] >= p.min_captures_plot) & (series["cap_out_roll"] >= p.min_captures_plot)
    # also hide early transients if desired; here we only use capture threshold

    frac_in_roll = series["frac_in_roll"].copy()
    frac_out_roll = series["frac_out_roll"].copy()
    delta_roll = series["delta_roll"].copy()

    frac_in_roll[~mask_roll] = np.nan
    frac_out_roll[~mask_roll] = np.nan
    delta_roll[~mask_roll] = np.nan

    plt.figure()
    plt.plot(t, series["frac_in_cum"], label="M fraction tuned (cumulative)")
    plt.plot(t, series["frac_out_cum"], label="M fraction outside (cumulative)")
    plt.plot(t, series["delta_cum"], label="delta (in-out) cumulative")

    plt.plot(t, frac_in_roll, label=f"M tuned (rolling w={p.rolling_window})", linewidth=1.0)
    plt.plot(t, frac_out_roll, label=f"M outside (rolling w={p.rolling_window})", linewidth=1.0)
    plt.plot(t, delta_roll, label=f"delta rolling (w={p.rolling_window})", linewidth=1.0)

    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots, "branch_clean.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # metastable occupancy histogram end
    region = data["region"]
    occM = data["occ_M_end"].astype(np.float64)
    in_vals = occM[region == 1]
    out_vals = occM[region == 0]

    plt.figure()
    bins = min(30, max(8, int(np.sqrt(len(occM) + 1))))
    plt.hist(in_vals, bins=bins, alpha=0.7, label="tuned region")
    plt.hist(out_vals, bins=bins, alpha=0.7, label="outside")
    plt.xlabel("metastable occupancy per node at end")
    plt.ylabel("count of nodes")
    plt.legend()
    plt.savefig(os.path.join(plots, "metastable_map_end.png"), dpi=160, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()

    # graph
    ap.add_argument("--N", type=int, default=250)
    ap.add_argument("--p_edge", type=float, default=0.03)

    # time
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=1.0)

    # constraints / dynamics
    ap.add_argument("--bandwidth", type=float, default=0.08)
    ap.add_argument("--mem_decay", type=float, default=0.0005)
    ap.add_argument("--mem_couple", type=float, default=1.8)
    ap.add_argument("--kappa", type=float, default=2.5)

    # gauge
    ap.add_argument("--g_coup", type=float, default=0.20)
    ap.add_argument("--theta_bw", type=float, default=0.10)

    # events
    ap.add_argument("--lambda_capture", type=float, default=0.02)
    ap.add_argument("--barrier_M", type=float, default=4.0)
    ap.add_argument("--base_bias", type=float, default=0.0)
    ap.add_argument("--tau_G", type=float, default=45.0)
    ap.add_argument("--tau_M", type=float, default=900.0)

    # tuned region / drive
    ap.add_argument("--drive_frac", type=float, default=0.20)
    ap.add_argument("--drive_amp", type=float, default=0.35)
    ap.add_argument("--drive_w", type=float, default=0.03)
    ap.add_argument("--drive_phase", type=float, default=0.0)
    ap.add_argument("--drive_gain", type=float, default=2.2)
    ap.add_argument("--region_bandwidth_mult", type=float, default=1.10)
    ap.add_argument("--region_mem_mult", type=float, default=0.95)

    # plotting / stats
    ap.add_argument("--rolling_window", type=int, default=1000, help="rolling window in ticks")
    ap.add_argument("--min_captures_plot", type=int, default=20, help="hide ticks where rolling captures in/out < this")

    # output
    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="metastable_toy_v2")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

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
        drive_gain=args.drive_gain,
        region_bandwidth_mult=args.region_bandwidth_mult,
        region_mem_mult=args.region_mem_mult,

        rolling_window=max(50, int(args.rolling_window)),
        min_captures_plot=max(1, int(args.min_captures_plot)),

        out_dir=args.out_dir,
        run_name=args.run_name,
        seed=args.seed,
    )

    tag = (
        f"N{p.N}_p{p.p_edge}_T{p.T}_seed{p.seed}"
        f"_lam{p.lambda_capture}_bar{p.barrier_M}_tauG{p.tau_G}_tauM{p.tau_M}"
        f"_drvF{p.drive_frac}_A{p.drive_amp}_w{p.drive_w}_gain{p.drive_gain}"
        f"_wroll{p.rolling_window}_mincap{p.min_captures_plot}"
        f"_{p.run_name}"
    )
    run_dir = make_run_dir(p.out_dir, tag)

    data = run_sim(p)
    series = compute_series(data, p.rolling_window)

    delta, se, z, cap_in, cap_out = final_stats(data)

    print("============================================================")
    print("FINAL (CUMULATIVE) BRANCH STATS")
    print("------------------------------------------------------------")
    print(f"captures_in  = {cap_in}")
    print(f"captures_out = {cap_out}")
    print(f"delta (p_in - p_out) = {delta:.6f}")
    print(f"SE(delta)           = {se:.6f}")
    print(f"z-score             = {z:.3f}")
    print("============================================================")

    np.savez_compressed(os.path.join(run_dir, "results.npz"), **data, **series)
    write_csv(os.path.join(run_dir, "log.csv"), data, series)
    plot_outputs(run_dir, data, series, p)

    print("Run output:", run_dir)
    print("Clean plot:", os.path.join(run_dir, "plots", "branch_clean.png"))


if __name__ == "__main__":
    main()
