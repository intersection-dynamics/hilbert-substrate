# poke_excitation_evolution.py
# ------------------------------------------------------------
# HSF: Poke / Excitation Evolution with Edge Memory + Bandwidth
# (Revised)
#
# Key revision:
#   - Edge stress tau_e(t) := |flow_e(t)|   (tracks motion/maintenance work)
#   - Edge memory m_e(t) records cumulative transport (no-forgetting)
#   - Memory throttles future flow via exp(-kappa*m_e) (no-refolding flavor)
#
# Outputs:
#   out_dir/
#     results.npz
#     stress_log.csv
#     support_pressure.png
#     concentration.png
#     flow_concentration.png
#     snapshots.png
#
# Windows one-liner example:
#   python poke_excitation_evolution.py --graph grid --L 28 --T 350 --bandwidth 0.02 --alpha 0.18 --kappa 2.0
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


Edge = Tuple[int, int]


def build_ring(n: int) -> Tuple[int, np.ndarray]:
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, np.array(edges, dtype=np.int32)


def build_grid(L: int) -> Tuple[int, np.ndarray]:
    def idx(x: int, y: int) -> int:
        return (y % L) * L + (x % L)

    edges = []
    for y in range(L):
        for x in range(L):
            a = idx(x, y)
            b = idx(x + 1, y)
            c = idx(x, y + 1)

            u, v = (a, b) if a < b else (b, a)
            edges.append((u, v))
            u, v = (a, c) if a < c else (c, a)
            edges.append((u, v))

    edges = sorted(set(edges))
    return L * L, np.array(edges, dtype=np.int32)


def support_size(weights: np.ndarray, frac: float = 0.95) -> int:
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return 0
    order = np.argsort(w)[::-1]
    csum = np.cumsum(w[order]) / total
    k = int(np.searchsorted(csum, frac) + 1)
    return k


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


@dataclass
class MetricsRow:
    t: int
    S: int
    L_flow: float
    P: float
    gini_flow: float
    top10_flow: float
    gini_mem: float
    top10_mem: float
    norm: float


def run_sim(
    n: int,
    edges: np.ndarray,
    T: int,
    bandwidth: float,
    alpha: float,
    mem_decay: float,
    mem_coupling: float,
    kappa: float,
    support_frac: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    m_edges = edges.shape[0]

    # Localized excitation: delta + small random-phase halo
    psi = np.zeros(n, dtype=np.complex128)
    center = n // 2
    psi[center] = 1.0 + 0.0j
    for k in range(1, min(4, n)):
        psi[(center + k) % n] += 0.2 * np.exp(1j * rng.uniform(0, 2 * np.pi))
        psi[(center - k) % n] += 0.2 * np.exp(1j * rng.uniform(0, 2 * np.pi))
    psi /= np.sqrt(np.vdot(psi, psi).real)

    # Edge memory register (no-forgetting bookkeeping)
    m = np.zeros(m_edges, dtype=np.float64)

    # Logs
    S_log = np.zeros(T, dtype=np.int32)
    L_flow_log = np.zeros(T, dtype=np.float64)
    P_log = np.zeros(T, dtype=np.float64)
    g_flow_log = np.zeros(T, dtype=np.float64)
    top_flow_log = np.zeros(T, dtype=np.float64)
    g_mem_log = np.zeros(T, dtype=np.float64)
    top_mem_log = np.zeros(T, dtype=np.float64)
    norm_log = np.zeros(T, dtype=np.float64)

    # Snapshots
    snap_t = []
    snap_w = []

    for t in range(T):
        w = (np.abs(psi) ** 2).astype(np.float64)
        S = support_size(w, support_frac)

        dpsi = np.zeros_like(psi)
        flow_mag = np.zeros(m_edges, dtype=np.float64)

        # Edge-wise local transport
        for ei, (u, v) in enumerate(edges):
            grad = psi[u] - psi[v]

            # Memory feedback throttles flow: edges with high record are "stiff"
            # kappa=0 => no feedback (pure diffusion-like)
            throttle = np.exp(-kappa * m[ei]) if kappa > 0 else 1.0
            flow = alpha * throttle * grad

            # Bandwidth cap per edge per tick
            mag = np.abs(flow)
            if mag > bandwidth:
                flow *= (bandwidth / (mag + 1e-12))
                mag = bandwidth

            # Apply antisymmetric transport
            dpsi[u] -= flow
            dpsi[v] += flow

            # Record transported magnitude into memory (no-forgetting)
            transported = float(mag)
            m[ei] = (1.0 - mem_decay) * m[ei] + mem_coupling * transported

            flow_mag[ei] = transported

        # Update state; renormalize as unitary proxy
        psi = psi + dpsi
        norm = np.sqrt(np.vdot(psi, psi).real)
        if norm > 0:
            psi /= norm

        # Metrics
        L_flow = float(flow_mag.sum())
        P = float(L_flow / max(S, 1))

        g_flow = float(gini(flow_mag))
        top_flow = float(top_fraction(flow_mag, 10))
        g_mem = float(gini(np.abs(m)))
        top_mem = float(top_fraction(np.abs(m), 10))
        norm_val = float(np.vdot(psi, psi).real)

        S_log[t] = S
        L_flow_log[t] = L_flow
        P_log[t] = P
        g_flow_log[t] = g_flow
        top_flow_log[t] = top_flow
        g_mem_log[t] = g_mem
        top_mem_log[t] = top_mem
        norm_log[t] = norm_val

        if t in {0, T // 4, T // 2, (3 * T) // 4, T - 1}:
            snap_t.append(t)
            snap_w.append(w.copy())

    return {
        "S": S_log,
        "L_flow": L_flow_log,
        "P": P_log,
        "gini_flow": g_flow_log,
        "top10_flow": top_flow_log,
        "gini_mem": g_mem_log,
        "top10_mem": top_mem_log,
        "norm": norm_log,
        "snap_t": np.array(snap_t, dtype=np.int32),
        "snap_w": np.array(snap_w, dtype=np.float64),
        "edges": edges.astype(np.int32),
        "final_mem": m.astype(np.float64),
    }


def write_csv(out_dir: str, data: Dict[str, np.ndarray]) -> str:
    path = os.path.join(out_dir, "stress_log.csv")
    T = len(data["S"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "S", "L_flow", "P", "gini_flow", "top10_flow", "gini_mem", "top10_mem", "norm"])
        for t in range(T):
            w.writerow([
                t,
                int(data["S"][t]),
                float(data["L_flow"][t]),
                float(data["P"][t]),
                float(data["gini_flow"][t]),
                float(data["top10_flow"][t]),
                float(data["gini_mem"][t]),
                float(data["top10_mem"][t]),
                float(data["norm"][t]),
            ])
    return path


def plot(out_dir: str, data: Dict[str, np.ndarray]) -> List[str]:
    paths: List[str] = []
    T = len(data["S"])
    t = np.arange(T)

    # Support + Pressure
    plt.figure()
    plt.plot(t, data["S"], label="S(t) support (95%)")
    plt.plot(t, data["P"], label="P(t)=L_flow/S pressure")
    plt.xlabel("tick")
    plt.legend()
    p = os.path.join(out_dir, "support_pressure.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # Concentration of MEMORY (tube-ness in stored structure)
    plt.figure()
    plt.plot(t, data["gini_mem"], label="Gini(t) memory concentration")
    plt.plot(t, data["top10_mem"], label="Top-10 edge memory fraction")
    plt.xlabel("tick")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    p = os.path.join(out_dir, "concentration.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # Concentration of FLOW (tube-ness in actual transport work)
    plt.figure()
    plt.plot(t, data["gini_flow"], label="Gini(t) flow concentration")
    plt.plot(t, data["top10_flow"], label="Top-10 edge flow fraction")
    plt.xlabel("tick")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    p = os.path.join(out_dir, "flow_concentration.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # Snapshots
    snap_t = data["snap_t"]
    snap_w = data["snap_w"]
    plt.figure()
    for i in range(snap_w.shape[0]):
        plt.plot(np.arange(snap_w.shape[1]), snap_w[i], label=f"t={int(snap_t[i])}")
    plt.xlabel("site")
    plt.ylabel("|psi|^2")
    plt.legend()
    p = os.path.join(out_dir, "snapshots.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    paths.append(p)

    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["ring", "grid"], default="grid")
    ap.add_argument("--n", type=int, default=128, help="sites for ring")
    ap.add_argument("--L", type=int, default=28, help="grid size LxL")
    ap.add_argument("--T", type=int, default=350)
    ap.add_argument("--bandwidth", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.18)

    ap.add_argument("--mem_decay", type=float, default=0.002)
    ap.add_argument("--mem_coupling", type=float, default=1.0)

    # NEW: memory feedback strength (self-trapping knob)
    ap.add_argument("--kappa", type=float, default=2.0,
                    help="Memory throttling strength; 0 disables feedback")

    ap.add_argument("--support_frac", type=float, default=0.95)
    ap.add_argument("--out_dir", type=str, default="poke_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.graph == "ring":
        n, edges = build_ring(args.n)
    else:
        n, edges = build_grid(args.L)

    os.makedirs(args.out_dir, exist_ok=True)

    data = run_sim(
        n=n,
        edges=edges,
        T=args.T,
        bandwidth=args.bandwidth,
        alpha=args.alpha,
        mem_decay=args.mem_decay,
        mem_coupling=args.mem_coupling,
        kappa=args.kappa,
        support_frac=args.support_frac,
        seed=args.seed,
    )

    npz_path = os.path.join(args.out_dir, "results.npz")
    np.savez_compressed(npz_path, **data)

    csv_path = write_csv(args.out_dir, data)
    plots = plot(args.out_dir, data)

    print("Wrote:")
    print(" ", npz_path)
    print(" ", csv_path)
    for p in plots:
        print(" ", p)


if __name__ == "__main__":
    main()
