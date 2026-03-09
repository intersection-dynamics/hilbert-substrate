# hsf_edge_phase_defects_v3.py
# ------------------------------------------------------------
# HSF: Edge phase register (U(1)) + defects + drive field + trajectory
#
# New in v3:
#   (1) Lock fraction targeting (0.1–0.4) via a LOCK BUDGET:
#       - no-refolding still holds: once an edge locks, it never unlocks
#       - but we stop locking additional edges once locked_frac >= lock_target
#       This keeps the substrate rigid-but-not-frozen.
#
#   (2) External drive field ("EM-like potential"):
#       - Adds a uniform bias to dtheta on horizontal/vertical edges each tick:
#           dtheta_ext = +A_x on horizontal edges, +A_y on vertical edges
#       - Applied only on UNLOCKED edges (otherwise refolding is frozen there).
#       - Capped by theta_bw like everything else (finite bandwidth).
#
#   (3) Defect tracking:
#       - Computes plaquette charge q(x,y) = round(raw / 2π) (integer)
#       - Tracks the (x,y) of the primary defect (default: most-negative q)
#       - Saves trajectory plots:
#           plots/trajectory_xy.png
#           plots/trajectory_time.png
#
# Output:
#   out_dir/<timestamp>_<tag>/
#     results.npz
#     log.csv
#     plots/
#       timeseries.png
#       charge_snapshots.png
#       flux_snapshots.png
#       trajectory_xy.png
#       trajectory_time.png
#
# Example (Windows one-liner):
#   python hsf_edge_phase_defects_v3.py --L 28 --T 3000 --seed_defect -1 --seed_xy 14 14 --drive_x 0.015 --drive_y 0.000 --lock_target 0.25
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
    # map to (-pi, pi]
    return (x + np.pi) % (2 * np.pi) - np.pi


def idx(L: int, x: int, y: int) -> int:
    return (y % L) * L + (x % L)


@dataclass
class GridMaps:
    L: int
    N: int
    # undirected edges with base orientation u< v
    u: np.ndarray
    v: np.ndarray
    # edge type: 1 if horizontal, 0 if vertical
    is_horiz: np.ndarray  # shape (E,), uint8
    # plaquette edges/signs for raw sum
    p_edges: np.ndarray   # (P,4)
    p_signs: np.ndarray   # (P,4)


def build_grid_maps(L: int) -> GridMaps:
    N = L * L

    # Undirected edges: right + up on a torus, tagged horiz/vert
    edges = []
    for y in range(L):
        for x in range(L):
            a = idx(L, x, y)
            b = idx(L, x + 1, y)   # right (horizontal)
            c = idx(L, x, y + 1)   # up (vertical)

            u1, v1 = (a, b) if a < b else (b, a)
            u2, v2 = (a, c) if a < c else (c, a)

            edges.append((u1, v1, 1))  # horiz
            edges.append((u2, v2, 0))  # vert

    # unique by (u,v); keep type by re-deriving later
    edges_uv = sorted(set((e[0], e[1]) for e in edges))
    u = np.array([e[0] for e in edges_uv], dtype=np.int32)
    v = np.array([e[1] for e in edges_uv], dtype=np.int32)

    # map edge (u,v) -> index
    edge_index: Dict[Tuple[int, int], int] = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(u, v))}

    # Determine horiz/vert by comparing coordinates
    # If nodes share y coordinate -> horiz; else vert (periodic)
    is_horiz = np.zeros(u.size, dtype=np.uint8)
    for i, (a, b) in enumerate(zip(u, v)):
        a = int(a); b = int(b)
        ax, ay = a % L, a // L
        bx, by = b % L, b // L
        # On torus, horizontal neighbors differ in x by 1 mod L and same y
        if ay == by:
            is_horiz[i] = 1
        else:
            is_horiz[i] = 0

    # Helper: directed edge (i->j) gives (edge index, sign) where sign means theta_dir = sign * theta[e]
    def directed_edge(i: int, j: int) -> Tuple[int, int]:
        uu, vv = (i, j) if i < j else (j, i)
        ei = edge_index[(uu, vv)]
        sign = +1 if i == uu else -1
        return ei, sign

    # Plaquette construction: one per cell (x,y)
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

            # a->b
            ei, sg = directed_edge(a, b)
            p_edges[p, 0], p_signs[p, 0] = ei, sg
            # b->d
            ei, sg = directed_edge(b, d)
            p_edges[p, 1], p_signs[p, 1] = ei, sg
            # d->c
            ei, sg = directed_edge(d, c)
            p_edges[p, 2], p_signs[p, 2] = ei, sg
            # c->a
            ei, sg = directed_edge(c, a)
            p_edges[p, 3], p_signs[p, 3] = ei, sg

    return GridMaps(L=L, N=N, u=u, v=v, is_horiz=is_horiz, p_edges=p_edges, p_signs=p_signs)


def plaquette_raw(theta: np.ndarray, maps: GridMaps) -> np.ndarray:
    # NOT wrapped
    return np.sum(maps.p_signs * theta[maps.p_edges], axis=1)


def plaquette_charge_and_flux(theta: np.ndarray, maps: GridMaps) -> tuple[np.ndarray, np.ndarray]:
    raw = plaquette_raw(theta, maps)
    q = np.rint(raw / TAU).astype(np.int32)
    phi = wrap_pi(raw - TAU * q)  # remainder in (-pi,pi]
    return q, phi


def seed_single_defect(theta: np.ndarray, maps: GridMaps, q: int, x: int, y: int) -> None:
    # Add +/-2π around one plaquette distributed on its 4 directed edges
    p = idx(maps.L, x, y)
    dphi = float(q) * TAU / 4.0
    for k in range(4):
        ei = int(maps.p_edges[p, k])
        sgn = int(maps.p_signs[p, k])
        theta[ei] += sgn * dphi
    theta[:] = wrap_pi(theta)


def minimal_image_step(L: int, dx: int) -> int:
    # choose step in [-L/2, +L/2] (integer)
    if dx > L // 2:
        dx -= L
    elif dx < -L // 2:
        dx += L
    return dx


def track_defect_xy(q_map: np.ndarray, L: int, prefer_negative: bool = True) -> Tuple[int, int, int]:
    """
    Return (x,y,qval) for the "primary" defect.
    Strategy:
      - If prefer_negative: pick most-negative q; else pick largest |q|.
      - If no defects: return (-1,-1,0).
    """
    q_flat = q_map.reshape(-1)
    if prefer_negative:
        min_q = int(q_flat.min())
        if min_q >= 0:
            # no negative defect; fallback to any nonzero
            nz = np.nonzero(q_flat)[0]
            if nz.size == 0:
                return -1, -1, 0
            i = int(nz[0])
            return i % L, i // L, int(q_flat[i])
        # choose one of the most-negative cells (if multiple)
        candidates = np.where(q_flat == min_q)[0]
        i = int(candidates[0])
        return i % L, i // L, int(min_q)
    else:
        absq = np.abs(q_flat)
        if absq.max() == 0:
            return -1, -1, 0
        i = int(np.argmax(absq))
        return i % L, i // L, int(q_flat[i])


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
    lock_target: float   # <= 1.0, target locked fraction
    lock_start: int      # only start locking after this tick (optional)

    seed_defect: int
    seed_x: int
    seed_y: int

    kick_every: int
    kick_strength: float

    drive_x: float
    drive_y: float

    seed: int


def run_sim(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)
    maps = build_grid_maps(p.L)
    N = maps.N
    E = maps.u.size

    # Matter on sites
    psi = (rng.normal(size=N) + 1j * rng.normal(size=N)).astype(np.complex128)
    psi /= np.sqrt(np.vdot(psi, psi).real + 1e-18)

    # Edge phase register + memory + locks
    theta = np.zeros(E, dtype=np.float64)
    m = np.zeros(E, dtype=np.float64)
    locked = np.zeros(E, dtype=np.int8)

    # Seed a defect
    if p.seed_defect != 0:
        seed_single_defect(theta, maps, p.seed_defect, p.seed_x, p.seed_y)

    # Logs
    T = p.T
    q_total = np.zeros(T, dtype=np.int32)
    q_abs = np.zeros(T, dtype=np.int32)
    flux_rms = np.zeros(T, dtype=np.float64)

    locked_frac = np.zeros(T, dtype=np.float64)
    L_flow = np.zeros(T, dtype=np.float64)
    L_theta = np.zeros(T, dtype=np.float64)

    # Trajectory
    x_t = np.full(T, -1, dtype=np.int32)
    y_t = np.full(T, -1, dtype=np.int32)
    q_t = np.zeros(T, dtype=np.int32)

    # Unwrapped trajectory (so it doesn't jump across periodic boundary)
    x_unwrap = np.full(T, np.nan, dtype=np.float64)
    y_unwrap = np.full(T, np.nan, dtype=np.float64)

    # Snapshots
    snap_times = {0, T // 4, T // 2, (3 * T) // 4, T - 1}
    snap_t, snap_q, snap_phi = [], [], []

    lock_target = float(np.clip(p.lock_target, 0.0, 1.0))

    for t in range(T):
        # Optional local kick to keep some dynamics alive
        if p.kick_every > 0 and (t % p.kick_every == 0) and (t > 0):
            i = int(rng.integers(0, N))
            psi[i] += p.kick_strength * np.exp(1j * rng.uniform(0, TAU))
            psi /= np.sqrt(np.vdot(psi, psi).real + 1e-18)

        dpsi = np.zeros_like(psi)
        dtheta = np.zeros_like(theta)

        flow_mag = np.zeros(E, dtype=np.float64)
        dtheta_mag = np.zeros(E, dtype=np.float64)

        # Lock budget state at start of tick
        lf = float(locked.mean())
        can_lock_more = (t >= p.lock_start) and (lf < lock_target)

        # Edge-local updates only (no-signaling)
        for ei, (a, b) in enumerate(zip(maps.u, maps.v)):
            a = int(a); b = int(b)
            phase = np.exp(1j * theta[ei])

            # Gauge-covariant gradient
            grad = psi[a] - phase * psi[b]

            # Memory throttles matter flow
            throttle = np.exp(-p.kappa * m[ei]) if p.kappa > 0 else 1.0

            flow = p.alpha * throttle * grad
            mag = float(np.abs(flow))
            if mag > p.bandwidth:
                flow *= (p.bandwidth / (mag + 1e-18))
                mag = p.bandwidth

            dpsi[a] -= flow
            dpsi[b] += np.conjugate(phase) * flow

            flow_mag[ei] = mag

            # Theta update from matter current + external drive (only if unlocked)
            if locked[ei] == 0:
                # gauge-invariant current proxy
                j = float(np.imag(np.conjugate(psi[a]) * phase * psi[b]))
                dth = p.g_coup * j

                # external drive field (uniform)
                if maps.is_horiz[ei]:
                    dth += p.drive_x
                else:
                    dth += p.drive_y

                # finite bandwidth on theta update
                dth_abs = abs(dth)
                if dth_abs > p.theta_bw:
                    dth *= (p.theta_bw / (dth_abs + 1e-18))
                    dth_abs = p.theta_bw

                dtheta[ei] += dth
                dtheta_mag[ei] = dth_abs

            # no-forgetting: memory records both matter and twist activity
            transported = mag + p.theta_mem_weight * dtheta_mag[ei]
            m[ei] = (1.0 - p.mem_decay) * m[ei] + p.mem_coupling * transported

            # no-refolding: lock when memory high, but obey lock budget
            if locked[ei] == 0 and can_lock_more and m[ei] >= p.lock_threshold:
                locked[ei] = 1

        # Apply updates
        psi = psi + dpsi
        psi /= np.sqrt(np.vdot(psi, psi).real + 1e-18)

        theta = wrap_pi(theta + dtheta)

        # Defects / holonomy
        q, phi = plaquette_charge_and_flux(theta, maps)
        q_map = q.reshape(p.L, p.L)

        q_total[t] = int(q.sum())
        q_abs[t] = int(np.abs(q).sum())
        flux_rms[t] = float(np.sqrt(np.mean(phi * phi)))

        locked_frac[t] = float(locked.mean())
        L_flow[t] = float(flow_mag.sum())
        L_theta[t] = float(dtheta_mag.sum())

        # Track primary defect
        x, y, qv = track_defect_xy(q_map, p.L, prefer_negative=True)
        x_t[t], y_t[t], q_t[t] = x, y, qv

        # Unwrap trajectory (only if a defect exists)
        if t == 0:
            if x >= 0:
                x_unwrap[t] = float(x)
                y_unwrap[t] = float(y)
        else:
            if x >= 0 and x_t[t - 1] >= 0:
                dx = minimal_image_step(p.L, int(x) - int(x_t[t - 1]))
                dy = minimal_image_step(p.L, int(y) - int(y_t[t - 1]))
                x_unwrap[t] = x_unwrap[t - 1] + dx
                y_unwrap[t] = y_unwrap[t - 1] + dy
            elif x >= 0:
                # reappeared after missing: start a new unwrap branch
                x_unwrap[t] = float(x)
                y_unwrap[t] = float(y)

        if t in snap_times:
            snap_t.append(t)
            snap_q.append(q_map.copy())
            snap_phi.append(phi.reshape(p.L, p.L).copy())

    return {
        "L": p.L,
        "u": maps.u, "v": maps.v,
        "is_horiz": maps.is_horiz,
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

        "x_t": x_t,
        "y_t": y_t,
        "q_t": q_t,
        "x_unwrap": x_unwrap,
        "y_unwrap": y_unwrap,

        "snap_t": np.array(snap_t, dtype=np.int32),
        "snap_q": np.array(snap_q, dtype=np.int32),
        "snap_phi": np.array(snap_phi, dtype=np.float64),
    }


def write_csv(path: str, data: dict) -> None:
    T = len(data["q_abs"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t","q_total","q_abs","flux_rms","locked_frac","L_flow","L_theta","x","y","q_primary","x_unwrap","y_unwrap"])
        for t in range(T):
            w.writerow([
                t,
                int(data["q_total"][t]),
                int(data["q_abs"][t]),
                float(data["flux_rms"][t]),
                float(data["locked_frac"][t]),
                float(data["L_flow"][t]),
                float(data["L_theta"][t]),
                int(data["x_t"][t]),
                int(data["y_t"][t]),
                int(data["q_t"][t]),
                float(data["x_unwrap"][t]) if np.isfinite(data["x_unwrap"][t]) else "",
                float(data["y_unwrap"][t]) if np.isfinite(data["y_unwrap"][t]) else "",
            ])


def plot_outputs(run_dir: str, data: dict) -> None:
    plots_dir = os.path.join(run_dir, "plots")
    T = len(data["q_abs"])
    t = np.arange(T)

    # timeseries
    plt.figure()
    plt.plot(t, data["q_abs"], label="sum |q_p|")
    plt.plot(t, data["flux_rms"], label="flux RMS (remainder)")
    plt.plot(t, data["locked_frac"], label="locked_frac")
    plt.plot(t, data["L_theta"], label="L_theta")
    plt.xlabel("tick")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "timeseries.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # charge snapshots
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

    # trajectory in xy (unwrapped)
    xuw = data["x_unwrap"]
    yuw = data["y_unwrap"]
    mask = np.isfinite(xuw) & np.isfinite(yuw)
    if np.any(mask):
        plt.figure()
        plt.plot(xuw[mask], yuw[mask])
        plt.scatter([xuw[mask][0]], [yuw[mask][0]], marker="o", label="start")
        plt.scatter([xuw[mask][-1]], [yuw[mask][-1]], marker="x", label="end")
        plt.xlabel("x (unwrapped)")
        plt.ylabel("y (unwrapped)")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "trajectory_xy.png"), dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(t[mask], xuw[mask], label="x_unwrap")
        plt.plot(t[mask], yuw[mask], label="y_unwrap")
        plt.xlabel("tick")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "trajectory_time.png"), dpi=160, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--L", type=int, default=28)
    ap.add_argument("--T", type=int, default=3000)

    # Constraints/dynamics
    ap.add_argument("--bandwidth", type=float, default=0.02, help="cap on |flow| per edge per tick")
    ap.add_argument("--theta_bw", type=float, default=0.12, help="cap on |dtheta| per edge per tick")
    ap.add_argument("--alpha", type=float, default=0.14, help="matter transport strength")
    ap.add_argument("--g_coup", type=float, default=0.20, help="theta response to current")
    ap.add_argument("--kappa", type=float, default=8.0, help="memory throttling on flow")

    ap.add_argument("--mem_decay", type=float, default=0.0002)
    ap.add_argument("--mem_coupling", type=float, default=3.0)
    ap.add_argument("--theta_mem_weight", type=float, default=1.0)

    # Locking: target fraction ~0.1–0.4 (budget)
    ap.add_argument("--lock_threshold", type=float, default=0.25)
    ap.add_argument("--lock_target", type=float, default=0.25, help="stop locking new edges once locked_frac reaches this")
    ap.add_argument("--lock_start", type=int, default=50, help="don’t start locking until this tick (lets motion happen first)")

    # Defect
    ap.add_argument("--seed_defect", type=int, default=-1)
    ap.add_argument("--seed_xy", nargs=2, type=int, default=None)

    # Kicks
    ap.add_argument("--kick_every", type=int, default=35)
    ap.add_argument("--kick_strength", type=float, default=0.25)

    # Drive field (external EM-like potential ramp)
    ap.add_argument("--drive_x", type=float, default=0.015, help="uniform theta bias on horizontal edges per tick")
    ap.add_argument("--drive_y", type=float, default=0.000, help="uniform theta bias on vertical edges per tick")

    ap.add_argument("--out_dir", type=str, default="hsf_out")
    ap.add_argument("--run_name", type=str, default="v3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sx, sy = (args.L // 2, args.L // 2) if args.seed_xy is None else (int(args.seed_xy[0]), int(args.seed_xy[1]))

    tag = (
        f"L{args.L}_T{args.T}"
        f"_bw{args.bandwidth}_thbw{args.theta_bw}"
        f"_a{args.alpha}_g{args.g_coup}_k{args.kappa}"
        f"_mc{args.mem_coupling}_lockth{args.lock_threshold}_lockt{args.lock_target}_lockstart{args.lock_start}"
        f"_seedq{args.seed_defect}_xy{sx}_{sy}"
        f"_kick{args.kick_every}_dx{args.drive_x}_dy{args.drive_y}_seed{args.seed}"
        f"_{args.run_name}"
    )
    run_dir = make_run_dir(args.out_dir, tag)

    params = Params(
        L=args.L, T=args.T,
        bandwidth=args.bandwidth, theta_bw=args.theta_bw,
        alpha=args.alpha, g_coup=args.g_coup, kappa=args.kappa,
        mem_decay=args.mem_decay, mem_coupling=args.mem_coupling, theta_mem_weight=args.theta_mem_weight,
        lock_threshold=args.lock_threshold, lock_target=args.lock_target, lock_start=args.lock_start,
        seed_defect=args.seed_defect, seed_x=sx, seed_y=sy,
        kick_every=args.kick_every, kick_strength=args.kick_strength,
        drive_x=args.drive_x, drive_y=args.drive_y,
        seed=args.seed,
    )

    data = run_sim(params)

    np.savez_compressed(os.path.join(run_dir, "results.npz"), **data)
    write_csv(os.path.join(run_dir, "log.csv"), data)
    plot_outputs(run_dir, data)

    print("Run output:", run_dir)
    print("Saved:", os.path.join(run_dir, "log.csv"))
    print("Plots:", os.path.join(run_dir, "plots"))


if __name__ == "__main__":
    main()
