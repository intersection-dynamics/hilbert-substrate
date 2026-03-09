# hsf_edge_phase_defects_v2.py
# ------------------------------------------------------------
# HSF: Explicit edge phase register + REAL holonomy + defect charge map
#
# Fix vs v1:
#   - Do NOT wrap the plaquette sum before extracting charge.
#   - Compute:
#       raw_p = sum(sign * theta_edge)          (can be ~[-4π, +4π])
#       q_p   = round(raw_p / 2π)              (integer defect/charge)
#       phi_p = wrap_to_pi(raw_p - 2π*q_p)     (remainder flux in (-π, π])
#
# This makes a seeded -1 defect show up as q_p = -1 on the target plaquette.
# ------------------------------------------------------------

from __future__ import annotations

import argparse, csv, datetime, os
from dataclasses import dataclass
from typing import Dict, Tuple

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
    return (x + np.pi) % (2 * np.pi) - np.pi


def idx(L: int, x: int, y: int) -> int:
    return (y % L) * L + (x % L)


@dataclass
class GridMaps:
    L: int
    N: int
    u: np.ndarray
    v: np.ndarray
    e_right: np.ndarray
    s_right: np.ndarray
    e_up: np.ndarray
    s_up: np.ndarray
    p_edges: np.ndarray  # (P,4)
    p_signs: np.ndarray  # (P,4)


def build_grid_maps(L: int) -> GridMaps:
    N = L * L

    # Undirected edges: right + up on a torus
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

    edge_index: Dict[Tuple[int, int], int] = {}
    for i, (a, b) in enumerate(zip(u, v)):
        edge_index[(int(a), int(b))] = i

    e_right = np.zeros(N, dtype=np.int32)
    s_right = np.zeros(N, dtype=np.int8)
    e_up = np.zeros(N, dtype=np.int32)
    s_up = np.zeros(N, dtype=np.int8)

    for y in range(L):
        for x in range(L):
            a = idx(L, x, y)
            b = idx(L, x + 1, y)
            c = idx(L, x, y + 1)

            uu, vv = (a, b) if a < b else (b, a)
            ei = edge_index[(uu, vv)]
            e_right[a] = ei
            s_right[a] = +1 if a == uu else -1

            uu, vv = (a, c) if a < c else (c, a)
            ei = edge_index[(uu, vv)]
            e_up[a] = ei
            s_up[a] = +1 if a == uu else -1

    P = N
    p_edges = np.zeros((P, 4), dtype=np.int32)
    p_signs = np.zeros((P, 4), dtype=np.int8)

    for y in range(L):
        for x in range(L):
            p = idx(L, x, y)
            a = idx(L, x, y)
            b = idx(L, x + 1, y)
            c = idx(L, x, y + 1)

            # loop: a->(x+1,y)->(x+1,y+1)->(x,y+1)->a
            p_edges[p, 0] = e_right[a]
            p_signs[p, 0] = s_right[a]

            p_edges[p, 1] = e_up[b]
            p_signs[p, 1] = s_up[b]

            p_edges[p, 2] = e_right[c]
            p_signs[p, 2] = -s_right[c]

            p_edges[p, 3] = e_up[a]
            p_signs[p, 3] = -s_up[a]

    return GridMaps(L, N, u, v, e_right, s_right, e_up, s_up, p_edges, p_signs)


def plaquette_raw(theta: np.ndarray, maps: GridMaps) -> np.ndarray:
    return np.sum(maps.p_signs * theta[maps.p_edges], axis=1)


def plaquette_charge_and_flux(theta: np.ndarray, maps: GridMaps) -> tuple[np.ndarray, np.ndarray]:
    raw = plaquette_raw(theta, maps)  # NOT wrapped
    q = np.rint(raw / TAU).astype(np.int32)
    phi = wrap_pi(raw - TAU * q)
    return q, phi


def seed_single_defect(theta: np.ndarray, maps: GridMaps, q: int, x: int, y: int) -> None:
    # Add +/-2π around one plaquette (distributed evenly)
    p = idx(maps.L, x, y)
    dphi = float(q) * TAU / 4.0
    for k in range(4):
        ei = int(maps.p_edges[p, k])
        sgn = int(maps.p_signs[p, k])
        theta[ei] += sgn * dphi
    theta[:] = wrap_pi(theta)  # keep each edge phase bounded


@dataclass
class Params:
    L: int
    T: int
    bandwidth: float
    theta_bw: float
    alpha: float
    g_coup: float
    kappa: float
    mem_decay: float
    mem_coupling: float
    theta_mem_weight: float
    lock_threshold: float
    seed_defect: int
    seed_x: int
    seed_y: int
    kick_every: int
    kick_strength: float
    seed: int


def run_sim(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)
    maps = build_grid_maps(p.L)
    N = maps.N
    E = maps.u.size

    psi = (rng.normal(size=N) + 1j * rng.normal(size=N)).astype(np.complex128)
    psi /= np.sqrt(np.vdot(psi, psi).real + 1e-18)

    theta = np.zeros(E, dtype=np.float64)
    m = np.zeros(E, dtype=np.float64)
    locked = np.zeros(E, dtype=np.int8)

    if p.seed_defect != 0:
        seed_single_defect(theta, maps, p.seed_defect, p.seed_x, p.seed_y)

    # logs
    q_abs = np.zeros(p.T, dtype=np.int32)
    q_total = np.zeros(p.T, dtype=np.int32)
    flux_rms = np.zeros(p.T, dtype=np.float64)
    locked_frac = np.zeros(p.T, dtype=np.float64)
    L_flow = np.zeros(p.T, dtype=np.float64)
    L_theta = np.zeros(p.T, dtype=np.float64)

    snap_times = {0, p.T // 4, p.T // 2, (3 * p.T) // 4, p.T - 1}
    snap_t, snap_q, snap_phi = [], [], []

    for t in range(p.T):
        if p.kick_every > 0 and (t % p.kick_every == 0) and (t > 0):
            i = int(rng.integers(0, N))
            psi[i] += p.kick_strength * np.exp(1j * rng.uniform(0, TAU))
            psi /= np.sqrt(np.vdot(psi, psi).real + 1e-18)

        dpsi = np.zeros_like(psi)
        dtheta = np.zeros_like(theta)
        flow_mag = np.zeros(E, dtype=np.float64)
        dtheta_mag = np.zeros(E, dtype=np.float64)

        for ei, (a, b) in enumerate(zip(maps.u, maps.v)):
            a = int(a); b = int(b)
            phase = np.exp(1j * theta[ei])

            grad = psi[a] - phase * psi[b]
            throttle = np.exp(-p.kappa * m[ei]) if p.kappa > 0 else 1.0
            flow = p.alpha * throttle * grad

            mag = float(np.abs(flow))
            if mag > p.bandwidth:
                flow *= (p.bandwidth / (mag + 1e-18))
                mag = p.bandwidth

            dpsi[a] -= flow
            dpsi[b] += np.conjugate(phase) * flow
            flow_mag[ei] = mag

            if locked[ei] == 0:
                j = float(np.imag(np.conjugate(psi[a]) * phase * psi[b]))
                dth = p.g_coup * j
                dth_abs = abs(dth)
                if dth_abs > p.theta_bw:
                    dth *= (p.theta_bw / (dth_abs + 1e-18))
                    dth_abs = p.theta_bw
                dtheta[ei] += dth
                dtheta_mag[ei] = dth_abs

            transported = mag + p.theta_mem_weight * dtheta_mag[ei]
            m[ei] = (1.0 - p.mem_decay) * m[ei] + p.mem_coupling * transported
            if locked[ei] == 0 and m[ei] >= p.lock_threshold:
                locked[ei] = 1

        psi = psi + dpsi
        psi /= np.sqrt(np.vdot(psi, psi).real + 1e-18)

        theta = wrap_pi(theta + dtheta)

        q, phi = plaquette_charge_and_flux(theta, maps)

        q_total[t] = int(q.sum())
        q_abs[t] = int(np.abs(q).sum())
        flux_rms[t] = float(np.sqrt(np.mean(phi * phi)))
        locked_frac[t] = float(locked.mean())
        L_flow[t] = float(flow_mag.sum())
        L_theta[t] = float(dtheta_mag.sum())

        if t in snap_times:
            snap_t.append(t)
            snap_q.append(q.reshape(p.L, p.L).copy())
            snap_phi.append(phi.reshape(p.L, p.L).copy())

    return {
        "L": p.L,
        "u": maps.u, "v": maps.v,
        "psi_final": psi,
        "theta_final": theta,
        "mem_final": m,
        "locked_final": locked,
        "q_total": q_total,
        "q_abs": q_abs,
        "flux_rms": flux_rms,
        "locked_frac": locked_frac,
        "L_flow": L_flow,
        "L_theta": L_theta,
        "snap_t": np.array(snap_t, dtype=np.int32),
        "snap_q": np.array(snap_q, dtype=np.int32),
        "snap_phi": np.array(snap_phi, dtype=np.float64),
    }


def write_csv(path: str, data: dict) -> None:
    T = len(data["q_abs"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t","q_total","q_abs","flux_rms","locked_frac","L_flow","L_theta"])
        for t in range(T):
            w.writerow([
                t,
                int(data["q_total"][t]),
                int(data["q_abs"][t]),
                float(data["flux_rms"][t]),
                float(data["locked_frac"][t]),
                float(data["L_flow"][t]),
                float(data["L_theta"][t]),
            ])


def plot_outputs(run_dir: str, data: dict) -> None:
    plots_dir = os.path.join(run_dir, "plots")
    T = len(data["q_abs"])
    t = np.arange(T)

    plt.figure()
    plt.plot(t, data["q_abs"], label="sum |q_p| (defect content)")
    plt.plot(t, data["flux_rms"], label="flux RMS (remainder phi)")
    plt.plot(t, data["locked_frac"], label="locked_frac")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "timeseries.png"), dpi=160, bbox_inches="tight")
    plt.close()

    snap_t = data["snap_t"]
    snap_q = data["snap_q"]
    snap_phi = data["snap_phi"]
    cols = snap_q.shape[0] if snap_q.size else 0

    if cols:
        plt.figure(figsize=(3.2 * cols, 3.2))
        for i in range(cols):
            ax = plt.subplot(1, cols, i + 1)
            im = ax.imshow(snap_q[i], origin="lower", vmin=-2, vmax=2)
            ax.set_title(f"q t={int(snap_t[i])}")
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(plots_dir, "charge_snapshots.png"), dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(3.2 * cols, 3.2))
        for i in range(cols):
            ax = plt.subplot(1, cols, i + 1)
            im = ax.imshow(snap_phi[i], origin="lower")
            ax.set_title(f"phi t={int(snap_t[i])}")
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(plots_dir, "flux_snapshots.png"), dpi=160, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=28)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--bandwidth", type=float, default=0.02)
    ap.add_argument("--theta_bw", type=float, default=0.12)
    ap.add_argument("--alpha", type=float, default=0.14)
    ap.add_argument("--g_coup", type=float, default=0.20)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--mem_decay", type=float, default=0.0002)
    ap.add_argument("--mem_coupling", type=float, default=3.0)
    ap.add_argument("--theta_mem_weight", type=float, default=1.0)
    ap.add_argument("--lock_threshold", type=float, default=0.25)
    ap.add_argument("--seed_defect", type=int, default=-1)
    ap.add_argument("--seed_xy", nargs=2, type=int, default=None)
    ap.add_argument("--kick_every", type=int, default=25)
    ap.add_argument("--kick_strength", type=float, default=0.35)
    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="v2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sx, sy = (args.L // 2, args.L // 2) if args.seed_xy is None else (int(args.seed_xy[0]), int(args.seed_xy[1]))

    tag = (
        f"L{args.L}_T{args.T}"
        f"_bw{args.bandwidth}_thbw{args.theta_bw}"
        f"_a{args.alpha}_g{args.g_coup}_k{args.kappa}"
        f"_mc{args.mem_coupling}_lock{args.lock_threshold}"
        f"_seedq{args.seed_defect}_xy{sx}_{sy}_kick{args.kick_every}_seed{args.seed}"
        f"_{args.run_name}"
    )
    run_dir = make_run_dir(args.out_dir, tag)

    params = Params(
        L=args.L, T=args.T,
        bandwidth=args.bandwidth, theta_bw=args.theta_bw,
        alpha=args.alpha, g_coup=args.g_coup, kappa=args.kappa,
        mem_decay=args.mem_decay, mem_coupling=args.mem_coupling,
        theta_mem_weight=args.theta_mem_weight, lock_threshold=args.lock_threshold,
        seed_defect=args.seed_defect, seed_x=sx, seed_y=sy,
        kick_every=args.kick_every, kick_strength=args.kick_strength,
        seed=args.seed
    )

    data = run_sim(params)
    np.savez_compressed(os.path.join(run_dir, "results.npz"), **data)
    write_csv(os.path.join(run_dir, "log.csv"), data)
    plot_outputs(run_dir, data)

    print("Run output:", run_dir)
    print("Plots:", os.path.join(run_dir, "plots"))


if __name__ == "__main__":
    main()
