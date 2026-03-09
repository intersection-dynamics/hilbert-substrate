# hsf_full_dynamics_sim_v2.py
# ------------------------------------------------------------
# HSF full dynamics, v2:
# - Same 4 constraints + rewiring/locking
# - Fixes single-site collapse via SATURATING self-focusing:
#       psi *= exp( eta * (amp2 - mean) / (1 + beta*amp2) )
#   where beta > 0 provides a soft-core (prevents delta collapse).
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import os
from dataclasses import dataclass
from typing import List, Tuple

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


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    cum = np.cumsum(xs)
    g = (n + 1.0 - 2.0 * float(np.sum(cum) / cum[-1])) / n
    return float(max(0.0, min(1.0, g)))


def top_fraction(x: np.ndarray, k: int = 10) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    s = float(x.sum())
    if s <= 0 or x.size == 0:
        return 0.0
    k = min(int(k), x.size)
    return float(np.sort(x)[-k:].sum() / s)


def support_size(weights: np.ndarray, frac: float = 0.95) -> int:
    w = np.asarray(weights, dtype=float)
    tot = float(w.sum())
    if tot <= 0:
        return 0
    idx = np.argsort(w)[::-1]
    c = np.cumsum(w[idx]) / tot
    return int(np.searchsorted(c, frac) + 1)


def edges_to_adj(N: int, u: np.ndarray, v: np.ndarray) -> List[np.ndarray]:
    deg = np.zeros(N, dtype=np.int32)
    for a, b in zip(u, v):
        deg[a] += 1
        deg[b] += 1
    tmp = [np.empty(deg[i], dtype=np.int32) for i in range(N)]
    pos = np.zeros(N, dtype=np.int32)
    for a, b in zip(u, v):
        tmp[a][pos[a]] = b
        pos[a] += 1
        tmp[b][pos[b]] = a
        pos[b] += 1
    return tmp


def count_excitation_peaks(weights: np.ndarray, adj: List[np.ndarray], thresh: float) -> int:
    w = weights
    peaks = 0
    for i in range(w.size):
        if w[i] < thresh:
            continue
        nb = adj[i]
        if nb.size == 0:
            peaks += 1
        else:
            if np.all(w[i] > w[nb]):
                peaks += 1
    return peaks


def build_random_k_regular_graph(N: int, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if k >= N:
        raise ValueError("k must be < N")
    stubs = np.repeat(np.arange(N), k)
    rng.shuffle(stubs)
    edges = set()
    attempts = 0
    max_attempts = 30 * len(stubs)

    i = 0
    while i + 1 < len(stubs) and attempts < max_attempts:
        a = int(stubs[i])
        b = int(stubs[i + 1])
        attempts += 1
        if a == b:
            rng.shuffle(stubs[i:])
            continue
        u, v = (a, b) if a < b else (b, a)
        if (u, v) in edges:
            rng.shuffle(stubs[i:])
            continue
        edges.add((u, v))
        i += 2

    target_m = (N * k) // 2
    while len(edges) < target_m:
        a = int(rng.integers(0, N))
        b = int(rng.integers(0, N))
        if a == b:
            continue
        u, v = (a, b) if a < b else (b, a)
        edges.add((u, v))

    u = np.fromiter((e[0] for e in edges), dtype=np.int32)
    v = np.fromiter((e[1] for e in edges), dtype=np.int32)
    return u, v


def bfs_ball_sizes(adj: List[np.ndarray], start: int, max_r: int) -> np.ndarray:
    N = len(adj)
    seen = np.zeros(N, dtype=np.int8)
    frontier = np.array([start], dtype=np.int32)
    seen[start] = 1
    counts = np.zeros(max_r, dtype=np.int32)
    total = 1
    for r in range(max_r):
        nxt = []
        for node in frontier:
            for nb in adj[node]:
                if not seen[nb]:
                    seen[nb] = 1
                    nxt.append(nb)
        frontier = np.array(nxt, dtype=np.int32) if nxt else np.empty(0, dtype=np.int32)
        total += frontier.size
        counts[r] = total
        if frontier.size == 0:
            counts[r:] = total
            break
    return counts


def estimate_effective_dimension(adj: List[np.ndarray], rng: np.random.Generator, samples: int = 12, max_r: int = 6) -> float:
    N = len(adj)
    rs = np.arange(1, max_r + 1, dtype=float)
    logs = []
    for _ in range(samples):
        s = int(rng.integers(0, N))
        ball = bfs_ball_sizes(adj, s, max_r).astype(float)
        ball = np.maximum(ball, 1.0)
        logs.append(np.log(ball))
    med_logV = np.median(np.vstack(logs), axis=0)
    x = np.log(rs[1:])
    y = med_logV[1:]
    if np.allclose(x.var(), 0):
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


@dataclass
class SimParams:
    N: int
    T: int
    k0: int
    target_deg: int
    bandwidth: float
    alpha: float
    kappa: float
    mem_decay: float
    mem_coupling: float
    lock_threshold: float
    eta: float
    focus_softcap: float
    rewire_every: int
    rewire_budget: int
    warmup_steps: int
    kick_every: int
    kick_strength: float
    seed: int


def run_sim(p: SimParams) -> dict:
    rng = np.random.default_rng(p.seed)

    u, v = build_random_k_regular_graph(p.N, p.k0, rng)
    m = np.zeros(u.size, dtype=np.float64)
    locked = np.zeros(u.size, dtype=np.int8)

    psi = (rng.normal(size=p.N) + 1j * rng.normal(size=p.N)).astype(np.complex128)
    psi /= np.sqrt(np.vdot(psi, psi).real + 1e-12)

    steps = p.T
    mean_deg = np.zeros(steps, dtype=np.float64)
    locked_frac = np.zeros(steps, dtype=np.float64)
    S95 = np.zeros(steps, dtype=np.int32)
    peaks = np.zeros(steps, dtype=np.int32)
    L_flow = np.zeros(steps, dtype=np.float64)
    P_press = np.zeros(steps, dtype=np.float64)
    g_flow = np.zeros(steps, dtype=np.float64)
    top_flow = np.zeros(steps, dtype=np.float64)
    g_mem = np.zeros(steps, dtype=np.float64)
    top_mem = np.zeros(steps, dtype=np.float64)
    dim_est = np.zeros(steps, dtype=np.float64)

    for t in range(steps):
        adj = edges_to_adj(p.N, u, v)

        dpsi = np.zeros_like(psi)
        flow_mag = np.zeros(u.size, dtype=np.float64)

        # kicks seed excitations
        if p.kick_every > 0 and (t % p.kick_every == 0) and (t > 0):
            idx = int(rng.integers(0, p.N))
            psi[idx] += p.kick_strength * np.exp(1j * rng.uniform(0, 2 * np.pi))

        for ei, (a, b) in enumerate(zip(u, v)):
            grad = psi[a] - psi[b]
            throttle = np.exp(-p.kappa * m[ei]) if p.kappa > 0 else 1.0
            flow = p.alpha * throttle * grad

            mag = np.abs(flow)
            if mag > p.bandwidth:
                flow *= (p.bandwidth / (mag + 1e-12))
                mag = p.bandwidth

            dpsi[a] -= flow
            dpsi[b] += flow
            flow_mag[ei] = float(mag)

            transported = float(mag)
            m[ei] = (1.0 - p.mem_decay) * m[ei] + p.mem_coupling * transported
            if locked[ei] == 0 and m[ei] >= p.lock_threshold:
                locked[ei] = 1

        # SATURATING self-focusing: prevents single-site collapse
        if p.eta != 0.0:
            amp2 = (np.abs(psi) ** 2)
            mu = float(amp2.mean())
            beta = max(0.0, p.focus_softcap)
            drive = (amp2 - mu) / (1.0 + beta * amp2)
            psi = psi * np.exp(p.eta * drive)

        psi = psi + dpsi
        psi /= np.sqrt(np.vdot(psi, psi).real + 1e-12)

        w = (np.abs(psi) ** 2).astype(np.float64)
        S = support_size(w, 0.95)

        mean_deg[t] = float(2.0 * u.size / p.N)
        locked_frac[t] = float(locked.mean())
        S95[t] = int(S)

        thresh = float(w.mean() * 8.0)
        peaks[t] = int(count_excitation_peaks(w, adj, thresh))

        L = float(flow_mag.sum())
        L_flow[t] = L
        P_press[t] = float(L / max(S, 1))
        g_flow[t] = float(gini(flow_mag))
        top_flow[t] = float(top_fraction(flow_mag, 10))
        g_mem[t] = float(gini(np.abs(m)))
        top_mem[t] = float(top_fraction(np.abs(m), 10))

        if t % max(10, p.rewire_every) == 0:
            dim_est[t] = estimate_effective_dimension(adj, rng, samples=10, max_r=6)
        else:
            dim_est[t] = dim_est[t - 1] if t > 0 else float("nan")

        # rewiring during warmup only; locked edges cannot be removed
        if (p.rewire_every > 0) and (t % p.rewire_every == 0) and (t < p.warmup_steps):
            removable = np.where(locked == 0)[0]
            if removable.size > 0:
                scores = flow_mag[removable] + 0.15 * np.abs(m[removable])
                rm_count = min(p.rewire_budget, removable.size // 8 + 1)
                rm_idx = removable[np.argsort(scores)[:rm_count]]

                keep_mask = np.ones(u.size, dtype=bool)
                keep_mask[rm_idx] = False
                u = u[keep_mask]
                v = v[keep_mask]
                m = m[keep_mask]
                locked = locked[keep_mask]

            # add edges to approach target degree, using triadic closure + correlation preference
            adj = edges_to_adj(p.N, u, v)
            deg = np.array([a.size for a in adj], dtype=np.int32)
            need = np.where(deg < p.target_deg)[0]
            add_budget = p.rewire_budget

            edge_set = set()
            for a, b in zip(u, v):
                edge_set.add((int(a), int(b)))

            rng.shuffle(need)
            for i in need[: min(need.size, add_budget)]:
                if adj[i].size > 0:
                    j = int(rng.choice(adj[i]))
                    k = int(rng.choice(adj[j])) if adj[j].size > 0 else int(rng.integers(0, p.N))
                else:
                    k = int(rng.integers(0, p.N))

                if k == i:
                    continue
                a, b = (i, k) if i < k else (k, i)
                if (a, b) in edge_set:
                    continue

                accept = (abs(psi[a]) * abs(psi[b])) > np.median(np.abs(psi)) ** 2
                if not accept and rng.random() > 0.2:
                    continue

                edge_set.add((a, b))
                u = np.append(u, np.int32(a))
                v = np.append(v, np.int32(b))
                m = np.append(m, 0.0)
                locked = np.append(locked, np.int8(0))

    return {
        "u": u.astype(np.int32),
        "v": v.astype(np.int32),
        "mem": m.astype(np.float64),
        "locked": locked.astype(np.int8),
        "mean_deg": mean_deg,
        "locked_frac": locked_frac,
        "S95": S95,
        "peaks": peaks,
        "L_flow": L_flow,
        "P_press": P_press,
        "g_flow": g_flow,
        "top_flow": top_flow,
        "g_mem": g_mem,
        "top_mem": top_mem,
        "dim_est": dim_est,
    }


def write_csv(path: str, data: dict) -> None:
    T = len(data["mean_deg"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "mean_deg", "locked_frac", "S95", "peaks",
            "L_flow", "P_press",
            "g_flow", "top10_flow",
            "g_mem", "top10_mem",
            "dim_est"
        ])
        for t in range(T):
            w.writerow([
                t,
                float(data["mean_deg"][t]),
                float(data["locked_frac"][t]),
                int(data["S95"][t]),
                int(data["peaks"][t]),
                float(data["L_flow"][t]),
                float(data["P_press"][t]),
                float(data["g_flow"][t]),
                float(data["top_flow"][t]),
                float(data["g_mem"][t]),
                float(data["top_mem"][t]),
                float(data["dim_est"][t]),
            ])


def plot_all(run_dir: str, data: dict) -> None:
    plots_dir = os.path.join(run_dir, "plots")
    T = len(data["mean_deg"])
    t = np.arange(T)

    plt.figure()
    plt.plot(t, data["mean_deg"], label="mean degree")
    plt.plot(t, data["locked_frac"] * np.max(data["mean_deg"]), label="locked fraction (scaled)")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "graph_metrics.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t, data["S95"], label="S95 support")
    plt.plot(t, data["peaks"], label="excitation peak count")
    plt.plot(t, data["P_press"] * np.max(data["S95"]), label="pressure (scaled)")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "support_excitation.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t, data["g_mem"], label="Gini(memory)")
    plt.plot(t, data["top_mem"], label="Top-10 memory fraction")
    plt.plot(t, data["g_flow"], label="Gini(flow)")
    plt.plot(t, data["top_flow"], label="Top-10 flow fraction")
    plt.xlabel("tick")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "memory_concentration.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t, data["dim_est"], label="effective dimension estimate")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "dimension_estimate.png"), dpi=160, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=384)
    ap.add_argument("--T", type=int, default=1200)

    ap.add_argument("--k0", type=int, default=10)
    ap.add_argument("--target_deg", type=int, default=6)

    ap.add_argument("--bandwidth", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.14)
    ap.add_argument("--kappa", type=float, default=8.0)

    ap.add_argument("--mem_decay", type=float, default=0.0002)
    ap.add_argument("--mem_coupling", type=float, default=3.0)
    ap.add_argument("--lock_threshold", type=float, default=0.25)

    ap.add_argument("--eta", type=float, default=0.20)
    ap.add_argument("--focus_softcap", type=float, default=4.0,
                    help="beta soft-core for focusing; larger = stronger saturation")

    ap.add_argument("--rewire_every", type=int, default=6)
    ap.add_argument("--rewire_budget", type=int, default=20)
    ap.add_argument("--warmup_steps", type=int, default=700)

    ap.add_argument("--kick_every", type=int, default=25)
    ap.add_argument("--kick_strength", type=float, default=0.35)

    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = (
        f"N{args.N}_T{args.T}_k0{args.k0}_deg{args.target_deg}"
        f"_bw{args.bandwidth}_a{args.alpha}_k{args.kappa}"
        f"_eta{args.eta}_beta{args.focus_softcap}"
        f"_lock{args.lock_threshold}"
        f"_rw{args.rewire_every}_rb{args.rewire_budget}_warm{args.warmup_steps}"
        f"_kick{args.kick_every}_seed{args.seed}"
    )
    if args.run_name.strip():
        tag = f"{tag}_{args.run_name.strip()}"

    run_dir = make_run_dir(args.out_dir, tag)

    params = SimParams(
        N=args.N,
        T=args.T,
        k0=args.k0,
        target_deg=args.target_deg,
        bandwidth=args.bandwidth,
        alpha=args.alpha,
        kappa=args.kappa,
        mem_decay=args.mem_decay,
        mem_coupling=args.mem_coupling,
        lock_threshold=args.lock_threshold,
        eta=args.eta,
        focus_softcap=args.focus_softcap,
        rewire_every=args.rewire_every,
        rewire_budget=args.rewire_budget,
        warmup_steps=args.warmup_steps,
        kick_every=args.kick_every,
        kick_strength=args.kick_strength,
        seed=args.seed,
    )

    data = run_sim(params)

    npz_path = os.path.join(run_dir, "results.npz")
    np.savez_compressed(npz_path, **data)

    csv_path = os.path.join(run_dir, "log.csv")
    write_csv(csv_path, data)

    plot_all(run_dir, data)

    print("Run output:", run_dir)
    print("Saved:", npz_path)
    print("Saved:", csv_path)
    print("Plots:", os.path.join(run_dir, "plots"))


if __name__ == "__main__":
    main()
