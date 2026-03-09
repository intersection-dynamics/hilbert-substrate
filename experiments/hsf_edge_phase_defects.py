# hsf_edge_phase_defects.py
# ------------------------------------------------------------
# HSF: Explicit Edge Phase Register (Twist/Holonomy) + Defects
#
# Core idea:
#   - Sites are emergent locality nodes (here: fixed LxL torus for clean holonomy).
#   - Matter-like state lives on sites: psi[i] (complex).
#   - Gauge-like bookkeeping lives on edges: theta[e] (real phase).
#   - Holonomy/flux is *real*, computed on plaquettes:
#         Phi_p = wrap_to_pi( sum_{edges in plaquette} theta_dir )
#   - Defects ("charge") are vortices: q_p = round(Phi_p / 2π).
#
# Four constraints operationalized:
#   1) No-signaling: only edge-local updates.
#   2) Finite bandwidth: cap |flow| per edge per tick AND cap |dtheta| per edge per tick.
#   3) No-forgetting: edge memory m[e] records |flow| and |dtheta|.
#   4) No-refolding: edges lock once m[e] crosses lock_threshold; locked edges stop changing theta.
#
# Output:
#   out_dir/<timestamp>_<tag>/
#     results.npz
#     log.csv
#     plots/
#       timeseries.png
#       flux_snapshots.png
#       memory_snapshots.png
#
# Example (Windows one-liner):
#   python hsf_edge_phase_defects.py --L 28 --T 1200 --bandwidth 0.02 --theta_bw 0.12 --alpha 0.14 --g_coup 0.20 --kappa 8 --mem_coupling 3.0 --lock_threshold 0.25 --seed_defect -1 --seed_xy 14 14
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def wrap_pi(x: np.ndarray) -> np.ndarray:
    # map to (-pi, pi]
    return (x + np.pi) % (2 * np.pi) - np.pi


def support_size(weights: np.ndarray, frac: float = 0.95) -> int:
    w = np.asarray(weights, dtype=float)
    tot = float(w.sum())
    if tot <= 0:
        return 0
    idx = np.argsort(w)[::-1]
    c = np.cumsum(w[idx]) / tot
    return int(np.searchsorted(c, frac) + 1)


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


# -------------------------
# Grid + Edge/plaquette maps
# -------------------------

def idx(L: int, x: int, y: int) -> int:
    return (y % L) * L + (x % L)


@dataclass
class GridMaps:
    L: int
    N: int
    # undirected edges: base orientation is (u < v)
    u: np.ndarray
    v: np.ndarray
    # for each directed neighbor move (right, up), store edge index and direction sign (+1 means theta_dir = +theta[e], -1 means theta_dir = -theta[e])
    e_right: np.ndarray  # shape (N,)
    s_right: np.ndarray  # shape (N,)
    e_up: np.ndarray     # shape (N,)
    s_up: np.ndarray     # shape (N,)
    # Plaquette edge lists (each plaquette is at (x,y) with corners (x,y)->(x+1,y)->(x+1,y+1)->(x,y+1)->back)
    # We'll store per-plaquette 4 edge indices and 4 signs for directed orientation.
    p_edges: np.ndarray  # shape (P,4)
    p_signs: np.ndarray  # shape (P,4)


def build_grid_maps(L: int) -> GridMaps:
    N = L * L

    # Build undirected edge list: right and up neighbors (periodic)
    edges = []
    for y in range(L):
        for x in range(L):
            a = idx(L, x, y)
            b = idx(L, x + 1, y)   # right
            c = idx(L, x, y + 1)   # up
            u1, v1 = (a, b) if a < b else (b, a)
            u2, v2 = (a, c) if a < c else (c, a)
            edges.append((u1, v1))
            edges.append((u2, v2))

    edges = sorted(set(edges))
    u = np.array([e[0] for e in edges], dtype=np.int32)
    v = np.array([e[1] for e in edges], dtype=np.int32)

    # Map undirected edge to index
    edge_index: Dict[Tuple[int, int], int] = {}
    for i, (a, b) in enumerate(zip(u, v)):
        edge_index[(int(a), int(b))] = i

    # For each site, determine right/up directed edge index and sign
    e_right = np.zeros(N, dtype=np.int32)
    s_right = np.zeros(N, dtype=np.int8)
    e_up = np.zeros(N, dtype=np.int32)
    s_up = np.zeros(N, dtype=np.int8)

    for y in range(L):
        for x in range(L):
            a = idx(L, x, y)
            b = idx(L, x + 1, y)
            c = idx(L, x, y + 1)

            # right: a -> b
            uu, vv = (a, b) if a < b else (b, a)
            ei = edge_index[(uu, vv)]
            e_right[a] = ei
            s_right[a] = +1 if a == uu else -1

            # up: a -> c
            uu, vv = (a, c) if a < c else (c, a)
            ei = edge_index[(uu, vv)]
            e_up[a] = ei
            s_up[a] = +1 if a == uu else -1

    # Plaquettes: one per cell (x,y)
    P = N
    p_edges = np.zeros((P, 4), dtype=np.int32)
    p_signs = np.zeros((P, 4), dtype=np.int8)

    # Loop orientation: (x,y)->(x+1,y)->(x+1,y+1)->(x,y+1)->back
    for y in range(L):
        for x in range(L):
            p = idx(L, x, y)
            a = idx(L, x, y)
            b = idx(L, x + 1, y)
            d = idx(L, x + 1, y + 1)
            c = idx(L, x, y + 1)

            # edges: a->b (right at a), b->d (up at b), d->c (left, i.e. reverse of right at c), c->a (down, i.e. reverse of up at a)
            # a->b
            p_edges[p, 0] = e_right[a]
            p_signs[p, 0] = s_right[a]
            # b->d (up at b)
            p_edges[p, 1] = e_up[b]
            p_signs[p, 1] = s_up[b]
            # d->c is reverse of c->d (right at c)
            p_edges[p, 2] = e_right[c]
            p_signs[p, 2] = -s_right[c]
            # c->a is reverse of a->c (up at a)
            p_edges[p, 3] = e_up[a]
            p_signs[p, 3] = -s_up[a]

    return GridMaps(
        L=L, N=N, u=u, v=v,
        e_right=e_right, s_right=s_right,
        e_up=e_up, s_up=s_up,
        p_edges=p_edges, p_signs=p_signs
    )


def plaquette_flux(theta: np.ndarray, maps: GridMaps) -> np.ndarray:
    # Phi_p = wrap_pi( sum_k sign[p,k]*theta[edge[p,k]] )
    raw = np.sum(maps.p_signs * theta[maps.p_edges], axis=1)
    return wrap_pi(raw)


def seed_single_defect(theta: np.ndarray, maps: GridMaps, q: int, x: int, y: int) -> None:
    """
    Seed a flux defect of charge q at plaquette (x,y).
    Achieved by distributing flux q*2π around that plaquette's 4 edges.
    This is a concrete "twist/holonomy is real" injection.
    """
    p = idx(maps.L, x, y)
    dphi = float(q) * TAU / 4.0
    for k in range(4):
        ei = int(maps.p_edges[p, k])
        sgn = int(maps.p_signs[p, k])
        theta[ei] += sgn * dphi
    theta[:] = wrap_pi(theta)


# -------------------------
# Dynamics
# -------------------------

@dataclass
class Params:
    L: int
    T: int
    bandwidth: float        # cap on |flow| per edge per tick
    theta_bw: float         # cap on |dtheta| per edge per tick
    alpha: float            # matter transport strength
    g_coup: float           # gauge/twist update coupling to current
    kappa: float            # memory throttling strength on flow
    mem_decay: float
    mem_coupling: float
    theta_mem_weight: float # how much |dtheta| contributes to memory record
    lock_threshold: float
    seed_defect: int
    seed_x: int
    seed_y: int
    kick_every: int
    kick_strength: float
    seed: int
    out_dir: str
    run_name: str


def run_sim(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)

    maps = build_grid_maps(p.L)
    N = maps.N
    E = maps.u.size

    # Matter state on sites
    psi = (rng.normal(size=N) + 1j * rng.normal(size=N)).astype(np.complex128)
    psi /= np.sqrt(np.vdot(psi, psi).real + 1e-18)

    # Edge phase register (twist/connection) + memory + lock
    theta = np.zeros(E, dtype=np.float64)
    m = np.zeros(E, dtype=np.float64)
    locked = np.zeros(E, dtype=np.int8)

    # Optional: seed a real holonomy defect
    if p.seed_defect != 0:
        seed_single_defect(theta, maps, p.seed_defect, p.seed_x, p.seed_y)

    # Logs
    T = p.T
    S95 = np.zeros(T, dtype=np.int32)
    L_flow = np.zeros(T, dtype=np.float64)
    L_theta = np.zeros(T, dtype=np.float64)
    P_press = np.zeros(T, dtype=np.float64)

    locked_frac = np.zeros(T, dtype=np.float64)
    g_mem = np.zeros(T, dtype=np.float64)
    top_mem = np.zeros(T, dtype=np.float64)
    g_flow = np.zeros(T, dtype=np.float64)
    top_flow = np.zeros(T, dtype=np.float64)

    # Defect metrics
    q_total = np.zeros(T, dtype=np.int32)
    q_abs = np.zeros(T, dtype=np.int32)
    flux_rms = np.zeros(T, dtype=np.float64)

    # Snapshots
    snap_times = {0, T // 4, T // 2, (3 * T) // 4, T - 1}
    snap_t = []
    snap_flux = []
    snap_mem = []

    for t in range(T):
        # optional local matter kick (doesn't directly set theta)
        if p.kick_every > 0 and (t % p.kick_every == 0) and (t > 0):
            i = int(rng.integers(0, N))
            psi[i] += p.kick_strength * np.exp(1j * rng.uniform(0, TAU))
            psi /= np.sqrt(np.vdot(psi, psi).real + 1e-18)

        dpsi = np.zeros_like(psi)
        dtheta = np.zeros_like(theta)

        flow_mag = np.zeros(E, dtype=np.float64)
        dtheta_mag = np.zeros(E, dtype=np.float64)

        # Edge-local updates only (no-signaling)
        for ei, (a, b) in enumerate(zip(maps.u, maps.v)):
            a = int(a)
            b = int(b)

            # Use the base orientation a<b with stored theta[ei] as theta_dir for a->b.
            # Covariant neighbor term: exp(i theta) psi[b]
            phase = np.exp(1j * theta[ei])

            # Covariant gradient (gauge-covariant difference)
            grad = psi[a] - phase * psi[b]

            # memory throttling on matter flow
            throttle = np.exp(-p.kappa * m[ei]) if p.kappa > 0 else 1.0

            # Matter flow
            flow = p.alpha * throttle * grad
            mag = float(np.abs(flow))
            if mag > p.bandwidth:
                flow *= (p.bandwidth / (mag + 1e-18))
                mag = p.bandwidth

            # Apply flow in a gauge-covariant way:
            #   psi[a] -= flow
            #   psi[b] += phase_conj * flow
            dpsi[a] -= flow
            dpsi[b] += np.conjugate(phase) * flow

            flow_mag[ei] = mag

            # Update theta from local current (gauge field responds to matter transport)
            # A standard gauge-invariant current proxy on edge:
            #   j = Im( conj(psi[a]) * phase * psi[b] )
            # We use it to rotate theta (twist changes due to current),
            # but we respect no-refolding lock.
            if locked[ei] == 0:
                j = float(np.imag(np.conjugate(psi[a]) * phase * psi[b]))
                dth = p.g_coup * j

                # bandwidth cap on twist change
                dth_abs = abs(dth)
                if dth_abs > p.theta_bw:
                    dth *= (p.theta_bw / (dth_abs + 1e-18))
                    dth_abs = p.theta_bw

                dtheta[ei] += dth
                dtheta_mag[ei] = dth_abs

            # no-forgetting: record transport *and* twist change into memory
            transported = mag + p.theta_mem_weight * dtheta_mag[ei]
            m[ei] = (1.0 - p.mem_decay) * m[ei] + p.mem_coupling * transported

            # no-refolding: lock once memory high
            if locked[ei] == 0 and m[ei] >= p.lock_threshold:
                locked[ei] = 1

        # Apply updates + renormalize (unitary proxy)
        psi = psi + dpsi
        psi /= np.sqrt(np.vdot(psi, psi).real + 1e-18)

        theta = wrap_pi(theta + dtheta)

        # Flux/defects
        flux = plaquette_flux(theta, maps)
        q = np.rint(flux / TAU).astype(np.int32)  # vortex charge per plaquette
        q_total[t] = int(q.sum())
        q_abs[t] = int(np.abs(q).sum())
        flux_rms[t] = float(np.sqrt(np.mean(flux * flux)))

        # Support + pressure
        w = (np.abs(psi) ** 2).astype(np.float64)
        S = support_size(w, 0.95)
        S95[t] = int(S)

        Lf = float(flow_mag.sum())
        Lt = float(dtheta_mag.sum())
        L_flow[t] = Lf
        L_theta[t] = Lt
        P_press[t] = float(Lf / max(S, 1))

        locked_frac[t] = float(locked.mean())
        g_mem[t] = float(gini(m))
        top_mem[t] = float(top_fraction(m, 10))
        g_flow[t] = float(gini(flow_mag))
        top_flow[t] = float(top_fraction(flow_mag, 10))

        if t in snap_times:
            snap_t.append(t)
            snap_flux.append(flux.reshape(p.L, p.L).copy())
            snap_mem.append(m.reshape(-1).copy())

    return {
        "L": p.L,
        "u": maps.u,
        "v": maps.v,
        "psi_final": psi,
        "theta_final": theta,
        "mem_final": m,
        "locked_final": locked,

        "S95": S95,
        "L_flow": L_flow,
        "L_theta": L_theta,
        "P_press": P_press,
        "locked_frac": locked_frac,
        "g_mem": g_mem,
        "top_mem": top_mem,
        "g_flow": g_flow,
        "top_flow": top_flow,

        "q_total": q_total,
        "q_abs": q_abs,
        "flux_rms": flux_rms,

        "snap_t": np.array(snap_t, dtype=np.int32),
        "snap_flux": np.array(snap_flux, dtype=np.float64),
        "snap_mem": np.array(snap_mem, dtype=np.float64),
    }


def write_csv(path: str, data: dict) -> None:
    T = len(data["S95"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "S95",
            "L_flow",
            "L_theta",
            "P_press",
            "locked_frac",
            "g_mem",
            "top10_mem",
            "g_flow",
            "top10_flow",
            "q_total",
            "q_abs",
            "flux_rms",
        ])
        for t in range(T):
            w.writerow([
                t,
                int(data["S95"][t]),
                float(data["L_flow"][t]),
                float(data["L_theta"][t]),
                float(data["P_press"][t]),
                float(data["locked_frac"][t]),
                float(data["g_mem"][t]),
                float(data["top_mem"][t]),
                float(data["g_flow"][t]),
                float(data["top_flow"][t]),
                int(data["q_total"][t]),
                int(data["q_abs"][t]),
                float(data["flux_rms"][t]),
            ])


def plot_outputs(run_dir: str, data: dict) -> None:
    plots_dir = os.path.join(run_dir, "plots")
    T = len(data["S95"])
    t = np.arange(T)

    # Time series summary
    plt.figure()
    plt.plot(t, data["S95"], label="S95 support (matter extent)")
    plt.plot(t, data["q_abs"], label="sum |q_p| (defect content)")
    plt.plot(t, data["flux_rms"], label="flux RMS")
    plt.plot(t, data["locked_frac"] * max(1.0, float(np.max(data["S95"]))), label="locked_frac (scaled)")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "timeseries.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Flux snapshots (holonomy is real)
    snap_t = data["snap_t"]
    snap_flux = data["snap_flux"]
    if snap_flux.size > 0:
        cols = snap_flux.shape[0]
        plt.figure(figsize=(3.2 * cols, 3.2))
        for i in range(cols):
            ax = plt.subplot(1, cols, i + 1)
            im = ax.imshow(snap_flux[i], origin="lower")
            ax.set_title(f"flux t={int(snap_t[i])}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(plots_dir, "flux_snapshots.png"), dpi=160, bbox_inches="tight")
        plt.close()

    # Memory snapshot distribution (edge memory is the bookkeeping support)
    snap_mem = data["snap_mem"]
    if snap_mem.size > 0:
        plt.figure()
        for i in range(snap_mem.shape[0]):
            ms = np.sort(snap_mem[i])
            plt.plot(ms, label=f"t={int(snap_t[i])}")
        plt.xlabel("edge rank")
        plt.ylabel("memory m[e] (sorted)")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "memory_snapshots.png"), dpi=160, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=28, help="grid size LxL (periodic)")
    ap.add_argument("--T", type=int, default=1200)

    # Constraints / dynamics
    ap.add_argument("--bandwidth", type=float, default=0.02, help="cap on |flow| per edge per tick")
    ap.add_argument("--theta_bw", type=float, default=0.12, help="cap on |dtheta| per edge per tick")
    ap.add_argument("--alpha", type=float, default=0.14, help="matter transport strength")
    ap.add_argument("--g_coup", type=float, default=0.20, help="theta update coupling to current")
    ap.add_argument("--kappa", type=float, default=8.0, help="memory throttling on flow")

    ap.add_argument("--mem_decay", type=float, default=0.0002)
    ap.add_argument("--mem_coupling", type=float, default=3.0)
    ap.add_argument("--theta_mem_weight", type=float, default=1.0, help="how much |dtheta| contributes to memory record")
    ap.add_argument("--lock_threshold", type=float, default=0.25)

    # Defect seeding (holonomy injection)
    ap.add_argument("--seed_defect", type=int, default=-1, help="integer vortex charge to seed at (x,y); 0 disables")
    ap.add_argument("--seed_xy", nargs=2, type=int, default=None, help="seed plaquette x y (default center)")

    # Matter kicks (optional)
    ap.add_argument("--kick_every", type=int, default=25)
    ap.add_argument("--kick_strength", type=float, default=0.35)

    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.seed_xy is None:
        sx = args.L // 2
        sy = args.L // 2
    else:
        sx, sy = int(args.seed_xy[0]), int(args.seed_xy[1])

    tag = (
        f"L{args.L}_T{args.T}"
        f"_bw{args.bandwidth}_thbw{args.theta_bw}"
        f"_a{args.alpha}_g{args.g_coup}_k{args.kappa}"
        f"_mdec{args.mem_decay}_mc{args.mem_coupling}_mth{args.theta_mem_weight}_lock{args.lock_threshold}"
        f"_seedq{args.seed_defect}_seedxy{sx}_{sy}"
        f"_kick{args.kick_every}_seed{args.seed}"
    )
    if args.run_name.strip():
        tag = f"{tag}_{args.run_name.strip()}"

    run_dir = make_run_dir(args.out_dir, tag)

    params = Params(
        L=args.L,
        T=args.T,
        bandwidth=args.bandwidth,
        theta_bw=args.theta_bw,
        alpha=args.alpha,
        g_coup=args.g_coup,
        kappa=args.kappa,
        mem_decay=args.mem_decay,
        mem_coupling=args.mem_coupling,
        theta_mem_weight=args.theta_mem_weight,
        lock_threshold=args.lock_threshold,
        seed_defect=args.seed_defect,
        seed_x=sx,
        seed_y=sy,
        kick_every=args.kick_every,
        kick_strength=args.kick_strength,
        seed=args.seed,
        out_dir=args.out_dir,
        run_name=args.run_name,
    )

    data = run_sim(params)

    npz_path = os.path.join(run_dir, "results.npz")
    np.savez_compressed(npz_path, **data)

    csv_path = os.path.join(run_dir, "log.csv")
    write_csv(csv_path, data)

    plot_outputs(run_dir, data)

    print("Run output:", run_dir)
    print("Saved:", npz_path)
    print("Saved:", csv_path)
    print("Plots:", os.path.join(run_dir, "plots"))


if __name__ == "__main__":
    main()
