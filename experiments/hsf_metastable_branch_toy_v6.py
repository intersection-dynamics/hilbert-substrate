# hsf_metastable_branch_toy_v6.py
# ------------------------------------------------------------
# v6: Fixes rectification.
# - signed drive stays (help vs hurt)
# - boundary shell stays
# - event-rate coupling becomes RECTIFIED (positive-half only by default),
#   so signed drive no longer cancels.
# ------------------------------------------------------------

from __future__ import annotations
import argparse, csv, datetime, os
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

TAU = 2.0 * np.pi

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_tag(s: str) -> str:
    return s.replace(" ", "_").replace(":", "_").replace("/", "_").replace(".", "p")

def make_run_dir(base_out: str, tag: str) -> str:
    run_dir = os.path.join(base_out, f"{now_stamp()}_{safe_tag(tag)}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    return run_dir

def wrap_pi_scalar(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)

def build_erdos_renyi(rng: np.random.Generator, N: int, p_edge: float) -> Tuple[np.ndarray, np.ndarray]:
    u_list, v_list = [], []
    for i in range(N):
        r = rng.random(N - i - 1)
        js = np.where(r < p_edge)[0] + (i + 1)
        for j in js:
            u_list.append(i); v_list.append(int(j))
    if not u_list:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.array(u_list, dtype=np.int32), np.array(v_list, dtype=np.int32)

def choose_tuned_region(rng: np.random.Generator, N: int, frac: float) -> np.ndarray:
    frac = float(np.clip(frac, 0.0, 1.0))
    k = int(round(frac * N))
    mask = np.zeros(N, dtype=np.int8)
    if k <= 0:
        return mask
    idx = np.arange(N); rng.shuffle(idx)
    mask[idx[:k]] = 1
    return mask

def branch_prob_metastable(barrier_M: float, drive_term: float, base_bias: float) -> float:
    eff = barrier_M - drive_term - base_bias
    if eff > 60: return 0.0
    if eff < -60: return 1.0
    return float(1.0 / (1.0 + np.exp(eff)))

def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if w <= 1: return x.copy()
    c = np.cumsum(x)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out

def compute_series(data: dict, w: int) -> dict:
    eps = 1e-12
    cap_in = data["captures_in"].astype(np.float64)
    cap_out = data["captures_out"].astype(np.float64)
    cap_tot = data["captures_total"].astype(np.float64)
    M_in = data["M_in"].astype(np.float64)
    M_out = data["M_out"].astype(np.float64)
    M_tot = data["M_total"].astype(np.float64)

    CapInCum = np.cumsum(cap_in); CapOutCum = np.cumsum(cap_out); CapTotCum = np.cumsum(cap_tot)
    MinCum = np.cumsum(M_in); MoutCum = np.cumsum(M_out); MtotCum = np.cumsum(M_tot)

    frac_in_cum = MinCum/(CapInCum+eps)
    frac_out_cum = MoutCum/(CapOutCum+eps)
    frac_tot_cum = MtotCum/(CapTotCum+eps)
    delta_cum = frac_in_cum - frac_out_cum

    cap_in_roll = rolling_sum(cap_in, w); cap_out_roll = rolling_sum(cap_out, w); cap_tot_roll = rolling_sum(cap_tot, w)
    M_in_roll = rolling_sum(M_in, w); M_out_roll = rolling_sum(M_out, w); M_tot_roll = rolling_sum(M_tot, w)
    frac_in_roll = M_in_roll/(cap_in_roll+eps)
    frac_out_roll = M_out_roll/(cap_out_roll+eps)
    frac_tot_roll = M_tot_roll/(cap_tot_roll+eps)
    delta_roll = frac_in_roll - frac_out_roll

    return dict(
        frac_in_cum=frac_in_cum, frac_out_cum=frac_out_cum, frac_tot_cum=frac_tot_cum, delta_cum=delta_cum,
        cap_in_roll=cap_in_roll, cap_out_roll=cap_out_roll, cap_tot_roll=cap_tot_roll,
        frac_in_roll=frac_in_roll, frac_out_roll=frac_out_roll, frac_tot_roll=frac_tot_roll, delta_roll=delta_roll
    )

def final_stats(data: dict) -> Tuple[float,float,float,int,int]:
    cap_in = int(np.sum(data["captures_in"]))
    cap_out = int(np.sum(data["captures_out"]))
    M_in = int(np.sum(data["M_in"]))
    M_out = int(np.sum(data["M_out"]))
    eps = 1e-18
    p_in = M_in/(cap_in+eps); p_out = M_out/(cap_out+eps)
    delta = p_in - p_out
    se = np.sqrt((p_in*(1-p_in)/(cap_in+eps)) + (p_out*(1-p_out)/(cap_out+eps)))
    z = delta/(se+eps)
    return float(delta), float(se), float(z), cap_in, cap_out

def phase_stats(data: dict) -> dict:
    B = int(data["phi_bins"]); eps = 1e-12
    caps_in = data["phi_caps_in"].astype(np.float64)
    caps_out = data["phi_caps_out"].astype(np.float64)
    M_in = data["phi_M_in"].astype(np.float64)
    M_out = data["phi_M_out"].astype(np.float64)
    p_in = M_in/(caps_in+eps); p_out = M_out/(caps_out+eps); delta = p_in - p_out
    phi_centers = (np.arange(B)+0.5)*(TAU/B)
    return dict(phi_centers=phi_centers, caps_in=caps_in, caps_out=caps_out, p_in=p_in, p_out=p_out, delta=delta)

def write_csv(path: str, data: dict, series: dict) -> None:
    T = len(data["captures_total"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t","captures_total","captures_in","captures_out",
            "M_total","M_in","M_out",
            "frac_M_in_cum","frac_M_out_cum","delta_cum",
            "frac_M_in_roll","frac_M_out_roll","delta_roll",
            "locked_like","mean_edge_mem","mean_theta_step",
            "lambda_in_factor"
        ])
        for t in range(T):
            w.writerow([
                t,
                int(data["captures_total"][t]), int(data["captures_in"][t]), int(data["captures_out"][t]),
                int(data["M_total"][t]), int(data["M_in"][t]), int(data["M_out"][t]),
                float(series["frac_in_cum"][t]), float(series["frac_out_cum"][t]), float(series["delta_cum"][t]),
                float(series["frac_in_roll"][t]), float(series["frac_out_roll"][t]), float(series["delta_roll"][t]),
                float(data["locked_like"][t]), float(data["mean_edge_mem"][t]), float(data["mean_theta_step"][t]),
                float(data["lambda_in_factor"][t]),
            ])

def write_phase_csv(path: str, ph: dict, min_caps: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin","phi_center","caps_in","caps_out","pM_in","pM_out","delta"])
        for i in range(len(ph["phi_centers"])):
            ci = float(ph["caps_in"][i]); co = float(ph["caps_out"][i])
            if (ci < min_caps) or (co < min_caps):
                w.writerow([i, float(ph["phi_centers"][i]), ci, co, "nan","nan","nan"])
            else:
                w.writerow([i, float(ph["phi_centers"][i]), ci, co, float(ph["p_in"][i]), float(ph["p_out"][i]), float(ph["delta"][i])])

def plot_outputs(run_dir: str, data: dict, series: dict, ph: dict, p, prefix: str) -> None:
    plots = os.path.join(run_dir, "plots")
    t = np.arange(len(data["captures_total"]))

    plt.figure()
    plt.plot(t, data["mean_edge_mem"], label="mean edge memory")
    plt.plot(t, data["locked_like"], label="locked_like (p85 mem fraction)")
    plt.plot(t, data["mean_theta_step"], label="mean |dtheta|")
    plt.xlabel("tick"); plt.legend()
    plt.savefig(os.path.join(plots, f"{prefix}_timeseries.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t, data["lambda_in_factor"], label="lambda_in_factor (rectified)")
    plt.xlabel("tick"); plt.legend()
    plt.savefig(os.path.join(plots, f"{prefix}_lambda_in_factor.png"), dpi=160, bbox_inches="tight")
    plt.close()

    mask_roll = (series["cap_in_roll"] >= p.min_captures_plot) & (series["cap_out_roll"] >= p.min_captures_plot)
    frac_in_roll = series["frac_in_roll"].copy(); frac_out_roll = series["frac_out_roll"].copy(); delta_roll = series["delta_roll"].copy()
    frac_in_roll[~mask_roll] = np.nan; frac_out_roll[~mask_roll] = np.nan; delta_roll[~mask_roll] = np.nan

    plt.figure()
    plt.plot(t, series["frac_in_cum"], label="M tuned (cum)")
    plt.plot(t, series["frac_out_cum"], label="M out (cum)")
    plt.plot(t, series["delta_cum"], label="delta cum")
    plt.plot(t, frac_in_roll, label=f"M tuned (roll w={p.rolling_window})", linewidth=1.0)
    plt.plot(t, frac_out_roll, label=f"M out (roll w={p.rolling_window})", linewidth=1.0)
    plt.plot(t, delta_roll, label=f"delta roll (w={p.rolling_window})", linewidth=1.0)
    plt.xlabel("tick"); plt.legend()
    plt.savefig(os.path.join(plots, f"{prefix}_branch_clean.png"), dpi=180, bbox_inches="tight")
    plt.close()

    phi = ph["phi_centers"]
    caps_in = ph["caps_in"]; caps_out = ph["caps_out"]
    p_in = ph["p_in"].copy(); p_out = ph["p_out"].copy(); delta = ph["delta"].copy()
    mask_phi = (caps_in >= p.min_captures_phase) & (caps_out >= p.min_captures_phase)
    p_in[~mask_phi] = np.nan; p_out[~mask_phi] = np.nan; delta[~mask_phi] = np.nan

    plt.figure()
    plt.plot(phi, p_in, marker="o", linewidth=1.0, label="pM tuned vs phase")
    plt.plot(phi, p_out, marker="o", linewidth=1.0, label="pM out vs phase")
    plt.plot(phi, delta, marker="o", linewidth=1.0, label="delta vs phase")
    plt.xlabel("drive phase (rad)"); plt.legend()
    plt.savefig(os.path.join(plots, f"{prefix}_phase_response.png"), dpi=180, bbox_inches="tight")
    plt.close()

def lambda_rectifier_factor(rectifier: str, eps: float, s: float, thresh: float) -> float:
    # s = sin(phase)
    if eps == 0.0:
        return 1.0
    rectifier = rectifier.lower().strip()
    if rectifier == "sin":
        # legacy (bad): symmetric modulation
        return max(0.0, 1.0 + eps * s)
    if rectifier == "poshalf":
        # NEW DEFAULT: only boosts during positive half-cycle
        return 1.0 + eps * max(0.0, s)
    if rectifier == "threshold":
        # only boosts when s exceeds a threshold
        return 1.0 + eps * (1.0 if s > thresh else 0.0)
    # fallback
    return 1.0 + eps * max(0.0, s)

def run_sim(p, rng_seed: int, *, signed_drive: bool, shell: bool, rectifier: str, thresh: float) -> dict:
    rng = np.random.default_rng(rng_seed)

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
        neigh_edges[a].append(ei); neigh_edges[b].append(ei)

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
    lambda_in_factor = np.ones(T, dtype=np.float64)

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
        b = int(np.floor((phi / TAU) * B)); b = min(b, B-1)

        # rectified event rate multiplier in region
        lam_factor = lambda_rectifier_factor(p.lambda_rectifier, p.eps_lambda, s, thresh)
        lambda_in_factor[t] = lam_factor

        # edges
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

        # events
        lam0 = p.lambda_capture
        events = np.zeros(p.N, dtype=np.int32)
        out_mask = (region == 0)
        in_mask = ~out_mask
        if np.any(out_mask):
            events[out_mask] = rng.poisson(lam=lam0, size=int(np.sum(out_mask))).astype(np.int32)
        if np.any(in_mask):
            events[in_mask] = rng.poisson(lam=lam0 * lam_factor, size=int(np.sum(in_mask))).astype(np.int32)

        captures_total[t] = int(events.sum())

        for i in range(p.N):
            k = int(events[i])
            if k <= 0: continue
            in_reg = (region[i] == 1)
            if in_reg:
                captures_in[t] += k; phi_caps_in[b] += k
            else:
                captures_out[t] += k; phi_caps_out[b] += k

            local_mem = float(np.mean(m_e[np.array(neigh_edges[i], dtype=np.int32)])) if neigh_edges[i] else 0.0

            drive_term = 0.0
            if in_reg:
                if signed_drive:
                    drive_term = p.drive_gain * drive / (1.0 + local_mem)
                else:
                    drive_term = p.drive_gain * max(0.0, drive) / (1.0 + local_mem)

            pM = branch_prob_metastable(p.barrier_M, drive_term, p.base_bias)
            m_count = rng.binomial(n=k, p=pM)
            g_count = k - m_count
            occ_M[i] += m_count; occ_G[i] += g_count
            M_total[t] += m_count; G_total[t] += g_count
            if in_reg:
                M_in[t] += m_count; G_in[t] += g_count; phi_M_in[b] += m_count
            else:
                M_out[t] += m_count; G_out[t] += g_count; phi_M_out[b] += m_count

        # decays
        if p.tau_G > 0:
            decG = rng.binomial(n=occ_G, p=min(1.0, 1.0/p.tau_G))
            occ_G -= decG
        if p.tau_M > 0:
            decM = rng.binomial(n=occ_M, p=min(1.0, 1.0/p.tau_M))
            occ_M -= decM

    return dict(
        u=u, v=v, theta_end=theta, mem_e_end=m_e, region=region,
        captures_total=captures_total, captures_in=captures_in, captures_out=captures_out,
        M_total=M_total, M_in=M_in, M_out=M_out, G_total=G_total, G_in=G_in, G_out=G_out,
        locked_like=locked_like, mean_edge_mem=mean_edge_mem, mean_theta_step=mean_theta_step,
        lambda_in_factor=lambda_in_factor,
        phi_bins=np.int32(B), phi_caps_in=phi_caps_in, phi_caps_out=phi_caps_out, phi_M_in=phi_M_in, phi_M_out=phi_M_out
    )

@dataclass
class RunParams:
    N:int; p_edge:float; T:int; dt:float
    bandwidth:float; mem_decay:float; mem_couple:float; kappa:float
    g_coup:float; theta_bw:float
    lambda_capture:float; barrier_M:float; base_bias:float
    tau_G:float; tau_M:float
    drive_frac:float; drive_amp:float; drive_w:float; drive_phase:float; drive_gain:float
    eps_lambda:float; lambda_rectifier:str; lambda_thresh:float
    region_bandwidth_mult:float; region_mem_mult:float
    boundary_bandwidth_mult:float; boundary_mem_mult:float
    rolling_window:int; min_captures_plot:int
    phase_bins:int; min_captures_phase:int
    compare_baseline:bool
    out_dir:str; run_name:str; seed:int

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=250)
    ap.add_argument("--p_edge", type=float, default=0.03)
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=1.0)

    ap.add_argument("--bandwidth", type=float, default=0.08)
    ap.add_argument("--mem_decay", type=float, default=0.0005)
    ap.add_argument("--mem_couple", type=float, default=1.8)
    ap.add_argument("--kappa", type=float, default=2.5)

    ap.add_argument("--g_coup", type=float, default=0.20)
    ap.add_argument("--theta_bw", type=float, default=0.10)

    ap.add_argument("--lambda_capture", type=float, default=0.09)
    ap.add_argument("--barrier_M", type=float, default=4.0)
    ap.add_argument("--base_bias", type=float, default=0.0)
    ap.add_argument("--tau_G", type=float, default=45.0)
    ap.add_argument("--tau_M", type=float, default=900.0)

    ap.add_argument("--drive_frac", type=float, default=0.20)
    ap.add_argument("--drive_amp", type=float, default=0.35)
    ap.add_argument("--drive_w", type=float, default=0.03)
    ap.add_argument("--drive_phase", type=float, default=0.0)
    ap.add_argument("--drive_gain", type=float, default=2.2)

    ap.add_argument("--eps_lambda", type=float, default=0.30)
    ap.add_argument("--lambda_rectifier", type=str, default="poshalf", choices=["poshalf","sin","threshold"])
    ap.add_argument("--lambda_thresh", type=float, default=0.25)

    ap.add_argument("--region_bandwidth_mult", type=float, default=1.10)
    ap.add_argument("--region_mem_mult", type=float, default=0.95)
    ap.add_argument("--boundary_bandwidth_mult", type=float, default=0.70)
    ap.add_argument("--boundary_mem_mult", type=float, default=1.35)

    ap.add_argument("--rolling_window", type=int, default=1000)
    ap.add_argument("--min_captures_plot", type=int, default=50)
    ap.add_argument("--phase_bins", type=int, default=24)
    ap.add_argument("--min_captures_phase", type=int, default=400)

    ap.add_argument("--compare_baseline", action="store_true")
    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="metastable_toy_v6")
    ap.add_argument("--seed", type=int, default=0)

    a = ap.parse_args()
    p = RunParams(
        N=a.N, p_edge=a.p_edge, T=a.T, dt=a.dt,
        bandwidth=a.bandwidth, mem_decay=a.mem_decay, mem_couple=a.mem_couple, kappa=a.kappa,
        g_coup=a.g_coup, theta_bw=a.theta_bw,
        lambda_capture=a.lambda_capture, barrier_M=a.barrier_M, base_bias=a.base_bias,
        tau_G=a.tau_G, tau_M=a.tau_M,
        drive_frac=a.drive_frac, drive_amp=a.drive_amp, drive_w=a.drive_w, drive_phase=a.drive_phase, drive_gain=a.drive_gain,
        eps_lambda=float(a.eps_lambda), lambda_rectifier=a.lambda_rectifier, lambda_thresh=float(a.lambda_thresh),
        region_bandwidth_mult=a.region_bandwidth_mult, region_mem_mult=a.region_mem_mult,
        boundary_bandwidth_mult=a.boundary_bandwidth_mult, boundary_mem_mult=a.boundary_mem_mult,
        rolling_window=max(50,int(a.rolling_window)), min_captures_plot=max(1,int(a.min_captures_plot)),
        phase_bins=max(8,int(a.phase_bins)), min_captures_phase=max(1,int(a.min_captures_phase)),
        compare_baseline=bool(a.compare_baseline),
        out_dir=a.out_dir, run_name=a.run_name, seed=int(a.seed),
    )

    tag = (
        f"N{p.N}_p{p.p_edge}_T{p.T}_seed{p.seed}"
        f"_lam{p.lambda_capture}_bar{p.barrier_M}"
        f"_drvF{p.drive_frac}_A{p.drive_amp}_w{p.drive_w}_gain{p.drive_gain}"
        f"_epsLam{p.eps_lambda}_{p.lambda_rectifier}"
        f"_bwb{p.boundary_bandwidth_mult}_memb{p.boundary_mem_mult}"
        f"_bins{p.phase_bins}_wroll{p.rolling_window}_{p.run_name}"
    )
    run_dir = make_run_dir(p.out_dir, tag)

    # v6
    data_v6 = run_sim(p, p.seed, signed_drive=True, shell=True, rectifier=p.lambda_rectifier, thresh=p.lambda_thresh)
    dv6, sev6, zv6, cap_in, cap_out = final_stats(data_v6)
    print("============================================================")
    print("FINAL (CUMULATIVE) BRANCH STATS — v6 (signed + shell + RECTIFIED lambda)")
    print("------------------------------------------------------------")
    print(f"captures_in  = {cap_in}")
    print(f"captures_out = {cap_out}")
    print(f"delta (p_in - p_out) = {dv6:.6f}")
    print(f"SE(delta)           = {sev6:.6f}")
    print(f"z-score             = {zv6:.3f}")
    print("============================================================")

    series_v6 = compute_series(data_v6, p.rolling_window)
    ph_v6 = phase_stats(data_v6)
    np.savez_compressed(os.path.join(run_dir, "results_v6.npz"), **data_v6, **series_v6, **ph_v6)
    write_csv(os.path.join(run_dir, "log_v6.csv"), data_v6, series_v6)
    write_phase_csv(os.path.join(run_dir, "phase_bins_v6.csv"), ph_v6, p.min_captures_phase)
    plot_outputs(run_dir, data_v6, series_v6, ph_v6, p, prefix="v6")

    # baseline compare
    if p.compare_baseline:
        data_base = run_sim(p, p.seed, signed_drive=False, shell=False, rectifier="poshalf", thresh=p.lambda_thresh)
        db, seb, zb, cap_in_b, cap_out_b = final_stats(data_base)
        print("============================================================")
        print("FINAL (CUMULATIVE) BRANCH STATS — baseline (unsigned drive, no shell, no rect)")
        print("------------------------------------------------------------")
        print(f"captures_in  = {cap_in_b}")
        print(f"captures_out = {cap_out_b}")
        print(f"delta (p_in - p_out) = {db:.6f}")
        print(f"SE(delta)           = {seb:.6f}")
        print(f"z-score             = {zb:.3f}")
        print("============================================================")

        print("============================================================")
        print("Δ IMPROVEMENT REPORT (v6 - baseline)")
        print("------------------------------------------------------------")
        print(f"delta_v6      = {dv6:.6f}")
        print(f"delta_baseline= {db:.6f}")
        print(f"Δ increase    = {dv6 - db:+.6f}")
        print(f"z_v6          = {zv6:.3f}")
        print(f"z_baseline    = {zb:.3f}")
        print(f"Δz            = {zv6 - zb:+.3f}")
        print("============================================================")

        series_b = compute_series(data_base, p.rolling_window)
        ph_b = phase_stats(data_base)
        np.savez_compressed(os.path.join(run_dir, "results_baseline.npz"), **data_base, **series_b, **ph_b)
        write_csv(os.path.join(run_dir, "log_baseline.csv"), data_base, series_b)
        write_phase_csv(os.path.join(run_dir, "phase_bins_baseline.csv"), ph_b, p.min_captures_phase)
        plot_outputs(run_dir, data_base, series_b, ph_b, p, prefix="baseline")

    print("Run output:", run_dir)
    print("v6 clean plot :", os.path.join(run_dir, "plots", "v6_branch_clean.png"))
    print("v6 phase plot :", os.path.join(run_dir, "plots", "v6_phase_response.png"))
    print("v6 lambda plot:", os.path.join(run_dir, "plots", "v6_lambda_in_factor.png"))

if __name__ == "__main__":
    main()
