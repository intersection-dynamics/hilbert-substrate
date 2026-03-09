# hsf_metastable_branch_toy.py
# ------------------------------------------------------------
# HSF-inspired toy model: bias selection into a metastable branch
#
# SAFE MODEL (non-nuclear, generic):
# - Nodes represent subsystems in an emergent substrate.
# - Edges carry a gauge-like phase register (theta) and a memory ledger (m_e).
# - A "capture-like" event occurs at random nodes with rate lambda_capture.
# - Each event chooses one of two branches:
#     G: fast-decaying (ground-like)
#     M: slow-decaying (metastable-like), normally rare due to a barrier
# - A "tuned region" modulates local link properties over time:
#     - periodic drive affects edge theta updates and effective "barrier relief"
#     - local memory/bandwidth changes can make M more/less likely in that region
#
# Outputs:
# - Non-overwriting run folder with:
#     results.npz
#     log.csv
#     plots/
#       timeseries.png
#       branch_fraction.png
#       metastable_map_end.png
#
# Typical run (Windows one-liner):
#   python hsf_metastable_branch_toy.py --N 250 --p_edge 0.03 --T 20000 --lambda_capture 0.02 --drive_amp 0.35 --drive_w 0.03 --drive_frac 0.20 --seed 0
#
# What to look for:
# - Compare "M branch fraction" inside driven region vs outside.
# - If drive is doing something, you should see a stable offset in:
#     M_frac_in - M_frac_out
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
    bandwidth: float          # cap on per-edge "influence"
    mem_decay: float
    mem_couple: float
    kappa: float              # memory throttling strength

    # gauge register
    g_coup: float
    theta_bw: float

    # event process
    lambda_capture: float     # rate per node per tick of "capture-like" event
    barrier_M: float          # metastable barrier (higher => rarer M)
    base_bias: float          # baseline bias (can be 0)

    # lifetimes (in ticks)
    tau_G: float              # mean lifetime for G before disappearing
    tau_M: float              # mean lifetime for M before disappearing

    # tuned region
    drive_frac: float         # fraction of nodes in tuned region
    drive_amp: float          # drive amplitude (dimensionless)
    drive_w: float            # angular frequency (rad/tick)
    drive_phase: float        # phase offset
    drive_gain: float         # how strongly drive reduces effective barrier (toy mechanism)
    region_bandwidth_mult: float  # bandwidth multiplier inside region
    region_mem_mult: float        # memory coupling multiplier inside region

    # output
    out_dir: str
    run_name: str
    seed: int


def build_erdos_renyi(rng: np.random.Generator, N: int, p_edge: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build undirected ER graph as edge lists u,v (no self edges, no duplicates).
    """
    u_list = []
    v_list = []
    for i in range(N):
        # sample edges to j>i
        r = rng.random(N - i - 1)
        js = np.where(r < p_edge)[0] + (i + 1)
        for j in js:
            u_list.append(i)
            v_list.append(int(j))
    if len(u_list) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.array(u_list, dtype=np.int32), np.array(v_list, dtype=np.int32)


def choose_tuned_region(rng: np.random.Generator, N: int, frac: float) -> np.ndarray:
    """
    Mark a subset of nodes as "tuned region".
    """
    frac = float(np.clip(frac, 0.0, 1.0))
    k = int(round(frac * N))
    mask = np.zeros(N, dtype=np.int8)
    if k <= 0:
        return mask
    idx = np.arange(N)
    rng.shuffle(idx)
    mask[idx[:k]] = 1
    return mask


def softplus(x: float) -> float:
    # numerically stable-ish for our small ranges
    if x > 40:
        return x
    return float(np.log1p(np.exp(x)))


def branch_prob_metastable(barrier_M: float, drive_term: float, base_bias: float) -> float:
    """
    Map (barrier - drive) to a probability via logistic.
    Higher barrier => lower p(M). Positive drive_term reduces effective barrier.
    base_bias lets you shift baseline.
    """
    eff = barrier_M - drive_term - base_bias
    # logistic on -eff
    # p = 1/(1+exp(eff))
    if eff > 60:
        return 0.0
    if eff < -60:
        return 1.0
    return float(1.0 / (1.0 + np.exp(eff)))


def run_sim(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)

    # graph
    u, v = build_erdos_renyi(rng, p.N, p.p_edge)
    E = u.size

    # edge registers
    theta = rng.uniform(-np.pi, np.pi, size=E).astype(np.float64)
    m_e = np.zeros(E, dtype=np.float64)

    # per-node occupancy of "excitation counts"
    occ_G = np.zeros(p.N, dtype=np.int32)
    occ_M = np.zeros(p.N, dtype=np.int32)

    # tuned region mask
    region = choose_tuned_region(rng, p.N, p.drive_frac)

    # stats logs
    T = p.T
    t_arr = np.arange(T, dtype=np.int32)

    captures_total = np.zeros(T, dtype=np.int32)
    captures_in = np.zeros(T, dtype=np.int32)
    captures_out = np.zeros(T, dtype=np.int32)

    M_total = np.zeros(T, dtype=np.int32)
    M_in = np.zeros(T, dtype=np.int32)
    M_out = np.zeros(T, dtype=np.int32)

    G_total = np.zeros(T, dtype=np.int32)
    G_in = np.zeros(T, dtype=np.int32)
    G_out = np.zeros(T, dtype=np.int32)

    locked_like = np.zeros(T, dtype=np.float64)  # proxy "rigidity": fraction of edges with high memory
    mean_edge_mem = np.zeros(T, dtype=np.float64)
    mean_theta_step = np.zeros(T, dtype=np.float64)

    # derived per-node adjacency list for efficiency
    # edge i connects u[i] <-> v[i]
    # build neighbor edge indices
    neigh_edges = [[] for _ in range(p.N)]
    for ei in range(E):
        neigh_edges[int(u[ei])].append(ei)
        neigh_edges[int(v[ei])].append(ei)

    # simulation
    for t in range(T):
        # drive term at time t
        drive = p.drive_amp * np.sin(p.drive_w * t + p.drive_phase)

        # --- edge-local gauge & memory dynamics (HSF-ish bookkeeping) ---
        # We use a toy "current" proxy based on occupancy imbalance across the edge.
        # (No wavefunctions here; we’re testing selection bias mechanisms.)
        if E > 0:
            dtheta_abs_sum = 0.0

            for ei in range(E):
                a = int(u[ei])
                b = int(v[ei])

                # local region multipliers based on whether either endpoint is in tuned region
                in_reg = (region[a] == 1) or (region[b] == 1)
                bw = p.bandwidth * (p.region_bandwidth_mult if in_reg else 1.0)
                mc = p.mem_couple * (p.region_mem_mult if in_reg else 1.0)

                # "flow demand" proxy: difference in total occupancy across edge
                da = (occ_G[a] + occ_M[a]) - (occ_G[b] + occ_M[b])
                # convert to bounded influence; memory throttles it
                throttle = np.exp(-p.kappa * m_e[ei])
                flow = throttle * np.tanh(0.25 * da)

                # finite bandwidth
                if abs(flow) > bw:
                    flow = np.sign(flow) * bw

                # theta update from a local current proxy
                # include drive as external bias on theta dynamics inside region
                j = flow
                dth = p.g_coup * j + (drive * 0.15 if in_reg else 0.0)

                # cap theta update bandwidth
                if abs(dth) > p.theta_bw:
                    dth = np.sign(dth) * p.theta_bw

                theta[ei] = wrap_pi(np.array([theta[ei] + dth], dtype=np.float64))[0]
                dtheta_abs_sum += abs(dth)

                # no-forgetting: edge memory records activity + unmet demand proxy
                transported = abs(flow) + abs(dth)
                m_e[ei] = (1.0 - p.mem_decay) * m_e[ei] + mc * transported

            mean_theta_step[t] = float(dtheta_abs_sum / max(E, 1))
            mean_edge_mem[t] = float(m_e.mean())

            # "locked_like": edges with memory above a percentile threshold
            thr = np.quantile(m_e, 0.85) if E > 5 else (m_e.mean() + 1e-9)
            locked_like[t] = float(np.mean(m_e >= thr))
        else:
            mean_theta_step[t] = 0.0
            mean_edge_mem[t] = 0.0
            locked_like[t] = 0.0

        # --- events: capture-like branching into G or M ---
        # For each node, generate number of events ~ Poisson(lambda_capture)
        # Branch probability for M is lower outside tuned region.
        lam = p.lambda_capture
        events = rng.poisson(lam=lam, size=p.N).astype(np.int32)

        captures_total[t] = int(events.sum())

        # branch each event batch per node
        for i in range(p.N):
            k = int(events[i])
            if k <= 0:
                continue

            in_reg = (region[i] == 1)
            captures_in[t] += int(k) if in_reg else 0
            captures_out[t] += int(k) if (not in_reg) else 0

            # drive term only for tuned region, using local neighborhood "gauge agitation"
            # local gauge agitation proxy: mean edge memory around node i (higher memory = harder to change)
            if neigh_edges[i]:
                local_mem = float(np.mean(m_e[np.array(neigh_edges[i], dtype=np.int32)]))
            else:
                local_mem = 0.0

            # In this toy model, the drive can reduce effective barrier,
            # but local congestion/memory fights it.
            # This encodes your intuition: bandwidth+no-forgetting can trap or channel outcomes.
            drive_term = 0.0
            if in_reg:
                drive_term = p.drive_gain * max(0.0, drive) / (1.0 + local_mem)

            pM = branch_prob_metastable(p.barrier_M, drive_term, p.base_bias)

            # draw how many become M
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

        # --- decay (metastable vs fast) ---
        # Each excitation has per-tick decay probability ~ 1/tau.
        # Use binomial thinning.
        if p.tau_G > 0:
            p_decay_G = min(1.0, 1.0 / p.tau_G)
            decG = rng.binomial(n=occ_G, p=p_decay_G)
            occ_G = occ_G - decG

        if p.tau_M > 0:
            p_decay_M = min(1.0, 1.0 / p.tau_M)
            decM = rng.binomial(n=occ_M, p=p_decay_M)
            occ_M = occ_M - decM

    # summary maps at end
    occ_total_end = (occ_G + occ_M).astype(np.int32)
    occ_M_end = occ_M.astype(np.int32)

    # compute branch fractions (avoid div by zero)
    eps = 1e-12
    frac_M_total = (M_total.astype(np.float64) / (captures_total.astype(np.float64) + eps))
    frac_M_in = (M_in.astype(np.float64) / (captures_in.astype(np.float64) + eps))
    frac_M_out = (M_out.astype(np.float64) / (captures_out.astype(np.float64) + eps))
    delta_in_out = frac_M_in - frac_M_out

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

        "frac_M_total": frac_M_total,
        "frac_M_in": frac_M_in,
        "frac_M_out": frac_M_out,
        "delta_in_out": delta_in_out,

        "locked_like": locked_like,
        "mean_edge_mem": mean_edge_mem,
        "mean_theta_step": mean_theta_step,

        "occ_total_end": occ_total_end,
        "occ_M_end": occ_M_end,
    }


def write_csv(path: str, data: dict) -> None:
    T = len(data["captures_total"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "captures_total", "captures_in", "captures_out",
            "M_total", "M_in", "M_out",
            "G_total", "G_in", "G_out",
            "frac_M_total", "frac_M_in", "frac_M_out", "delta_in_out",
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
                float(data["frac_M_total"][t]),
                float(data["frac_M_in"][t]),
                float(data["frac_M_out"][t]),
                float(data["delta_in_out"][t]),
                float(data["locked_like"][t]),
                float(data["mean_edge_mem"][t]),
                float(data["mean_theta_step"][t]),
            ])


def plot_outputs(run_dir: str, data: dict) -> None:
    plots = os.path.join(run_dir, "plots")
    t = np.arange(len(data["captures_total"]))

    # timeseries: substrate-ish stats
    plt.figure()
    plt.plot(t, data["mean_edge_mem"], label="mean edge memory")
    plt.plot(t, data["locked_like"], label="locked_like (p85 mem fraction)")
    plt.plot(t, data["mean_theta_step"], label="mean |dtheta|")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots, "timeseries.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # branch fractions
    plt.figure()
    plt.plot(t, data["frac_M_total"], label="M fraction total")
    plt.plot(t, data["frac_M_in"], label="M fraction tuned region")
    plt.plot(t, data["frac_M_out"], label="M fraction outside")
    plt.plot(t, data["delta_in_out"], label="delta (in - out)")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots, "branch_fraction.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # metastable occupancy map at end (as a bar plot by region membership)
    region = data["region"]
    occM = data["occ_M_end"].astype(np.float64)

    in_vals = occM[region == 1]
    out_vals = occM[region == 0]

    plt.figure()
    bins = min(30, max(8, int(np.sqrt(len(occM) + 1))))
    plt.hist(in_vals, bins=bins, alpha=0.7, label="tuned region", density=False)
    plt.hist(out_vals, bins=bins, alpha=0.7, label="outside", density=False)
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

    # gauge register
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

    # output
    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="metastable_toy")
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

        out_dir=args.out_dir,
        run_name=args.run_name,
        seed=args.seed,
    )

    tag = (
        f"N{p.N}_p{p.p_edge}_T{p.T}"
        f"_lam{p.lambda_capture}_bar{p.barrier_M}_tauG{p.tau_G}_tauM{p.tau_M}"
        f"_drvF{p.drive_frac}_A{p.drive_amp}_w{p.drive_w}_gain{p.drive_gain}"
        f"_bw{p.bandwidth}_mc{p.mem_couple}_k{p.kappa}"
        f"_seed{p.seed}_{p.run_name}"
    )
    run_dir = make_run_dir(p.out_dir, tag)

    data = run_sim(p)

    np.savez_compressed(os.path.join(run_dir, "results.npz"), **data)
    write_csv(os.path.join(run_dir, "log.csv"), data)
    plot_outputs(run_dir, data)

    print("Run output:", run_dir)
    print("Plots:", os.path.join(run_dir, "plots"))


if __name__ == "__main__":
    main()
