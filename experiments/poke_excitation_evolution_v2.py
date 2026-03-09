# poke_excitation_evolution_v2.py
# ------------------------------------------------------------
# HSF: Poke / Excitation Evolution with Edge Memory + Bandwidth
# v2: Unique output directory per run (no overwrites)
#
# Key features:
#   - Edge stress tau_e(t) := |flow_e(t)|  (tracks motion/maintenance work)
#   - Edge memory m_e(t) records cumulative transport (no-forgetting bookkeeping)
#   - Memory throttles future flow via exp(-kappa*m_e) (no-refolding flavor)
#   - Unique run folder: out_dir/<timestamp>_<tag>/
#
# Example Windows command:
#   python poke_excitation_evolution_v2.py --graph grid --L 28 --T 350 --bandwidth 0.02 --alpha 0.18 --kappa 2.0
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


Edge = Tuple[int, int]


def build_ring(n: int) -> Tuple[int, np.ndarray]:
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, np.array(edges, dtype=np.int32)


def build_grid(L: int) -> Tuple[int, np.ndarray]:
    # 2D LxL grid with periodic boundary conditions (torus)
    def idx(x: int, y: int) -> int:
        return (y % L) * L + (x % L)

    edges: List[Edge] = []
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


def make_run_dir(base_out_dir: str, tag: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = tag.replace(" ", "_").replace(":", "_").replace("/", "_")
    # Avoid dots in folder names for readability (0.02 -> 0p02)
    safe_tag = safe_tag.replace(".", "p")
    run_dir = os.path.join(base_out_dir, f"{stamp}_{safe_tag}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


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
    """
    State:
      psi[i] complex amplitude at site i
      m[e]   real edge memory register

    Update rule per tick:
      flow_e = alpha * exp(-kappa*m_e) * (psi[u] - psi[v])
      |flow_e| capped by bandwidth
      psi updated by antisymmetric edge flows
      m_e updated by transported magnitude (with slight decay)

    Logged:
      S(t)    support size (95% of |psi|^2)
      L_flow  total flow magnitude sum_e |flow_e|
      P(t)    pressure = L_flow / S
      concentration metrics for flow and for memory
    """
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

    m = np.zeros(m_edges, dtype=np.float64)

    S_log = np.zeros(T, dtype=np.int32)
    L_flow_log = np.zeros(T, dtype=np.float64)
    P_log = np.zeros(T, dtype=np.float64)

    gini_flow_log = np.zeros(T, dtype=np.float64)
    top10_flow_log = np.zeros(T, dtype=np.float64)
    gini_mem_log = np.zeros(T, dtype=np.float64)
    top10_mem_log = np.zeros(T, dtype=np.float64)

    norm_log = np.zeros(T, dtype=np.float64)

    # Snapshots of |psi|^2 at a few times
    snap_times = {0, T // 4, T // 2, (3 * T) // 4, T - 1}
    snap_t: List[int] = []
    snap_w: List[np.ndarray] = []

    for t in range(T):
        w = (np.abs(psi) ** 2).astype(np.float64)
        S = support_size(w, support_frac)

        dpsi = np.zeros_like(psi)
        flow_mag = np.zeros(m_edges, dtype=np.float64)

        for ei, (u, v) in enumerate(edges):
            grad = psi[u] - psi[v]
            throttle = np.exp(-kappa * m[ei]) if kappa > 0 else 1.0
            flow = alpha * throttle * grad

            mag = np.abs(flow)
            if mag > bandwidth:
                flow *= (bandwidth / (mag + 1e-12))
                mag = bandwidth

            dpsi[u] -= flow
            dpsi[v] += flow

            transported = float(mag)
            m[ei] = (1.0 - mem_decay) * m[ei] + mem_coupling * transported
            flow_mag[ei] = transported

        # Update + renormalize (unitary proxy)
        psi = psi + dpsi
        norm = np.sqrt(np.vdot(psi, psi).real)
        if norm > 0:
            psi /= norm

        # Metrics
        L_flow = float(flow_mag.sum())
        P = float(L_flow / max(S, 1))

        S_log[t] = S
        L_flow_log[t] = L_flow
        P_log[t] = P

        gini_flow_log[t] = float(gini(flow_mag))
        top10_flow_log[t] = float(top_fraction(flow_mag, 10))
        gini_mem_log[t] = float(gini(np.abs(m)))
        top10_mem_log[t] = float(top_fraction(np.abs(m), 10))

        norm_log[t] = float(np.vdot(psi, psi).real)

        if t in snap_times:
            snap_t.append(t)
            snap_w.append(w.copy())

    return {
        "S": S_log,
        "L_flow": L_flow_log,
        "P": P_log,
        "gini_flow": gini_flow_log,
        "top10_flow": top10_flow_log,
        "gini_mem": gini_mem_log,
        "top10_mem": top10_mem_log,
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

    # Memory concentration
    plt.figure()
    plt.plot(t, data["gini_mem"], label="Gini(t) memory concentration")
    plt.plot(t, data["top10_mem"], label="Top-10 edge memory fraction")
    plt.xlabel("tick")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    p = os.path.join(out_dir, "concentration_memory.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # Flow concentration
    plt.figure()
    plt.plot(t, data["gini_flow"], label="Gini(t) flow concentration")
    plt.plot(t, data["top10_flow"], label="Top-10 edge flow fraction")
    plt.xlabel("tick")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    p = os.path.join(out_dir, "concentration_flow.png")
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

    ap.add_argument("--kappa", type=float, default=2.0,
                    help="Memory throttling strength; 0 disables feedback")

    ap.add_argument("--support_frac", type=float, default=0.95)

    ap.add_argument("--out_dir", type=str, default="poke_out",
                    help="Base output directory. A unique subfolder is created each run.")
    ap.add_argument("--run_name", type=str, default="",
                    help="Optional label appended to the run folder name.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.graph == "ring":
        n, edges = build_ring(args.n)
        graph_tag = f"ring_n{args.n}"
    else:
        n, edges = build_grid(args.L)
        graph_tag = f"grid_L{args.L}"

    # Build a descriptive tag for the run folder
    tag = f"{graph_tag}_T{args.T}_bw{args.bandwidth}_a{args.alpha}_k{args.kappa}_seed{args.seed}"
    if args.run_name.strip():
        tag = f"{tag}_{args.run_name.strip()}"

    run_dir = make_run_dir(args.out_dir, tag)

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

    npz_path = os.path.join(run_dir, "results.npz")
    np.savez_compressed(npz_path, **data)

    csv_path = write_csv(run_dir, data)
    plot_paths = plot(run_dir, data)

    print("Run output:")
    print(" ", run_dir)
    print("Files:")
    print(" ", npz_path)
    print(" ", csv_path)
    for p in plot_paths:
        print(" ", p)


if __name__ == "__main__":
    main()
