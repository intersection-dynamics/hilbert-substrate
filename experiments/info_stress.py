# info_stress.py
# ------------------------------------------------------------
# Hilbert Substrate Framework: Informational Stress + Pressure Meter
#
# Purpose:
#   Turn "bookkeeping cost" into measured quantities:
#     - edge stress tau_ij(t)
#     - total load L(t)
#     - support size S(t) (95% mass)
#     - pressure P(t) = L/S
#     - concentration metrics (is stress tube-like?)
#
# You provide two functions each tick:
#   site_weights() -> list/np.ndarray of length N_sites (nonnegative)
#   edge_stress()  -> dict mapping (i,j) -> nonnegative float
#
# Outputs:
#   - CSV log
#   - PNG plots
# ------------------------------------------------------------

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


Edge = Tuple[int, int]


def support_size(weights: np.ndarray, frac: float = 0.95) -> int:
    """Return the smallest k such that top-k weights sum to >= frac of total."""
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return 0
    idx = np.argsort(w)[::-1]
    csum = np.cumsum(w[idx]) / total
    k = int(np.searchsorted(csum, frac) + 1)
    return k


def gini(values: np.ndarray) -> float:
    """Gini coefficient (0 = uniform, 1 = maximally concentrated)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    # Gini = (n+1 - 2 * sum_i (cum_i)/cum_n) / n
    g = (n + 1.0 - 2.0 * float(np.sum(cum) / cum[-1])) / n
    return float(max(0.0, min(1.0, g)))


def top_fraction(values: np.ndarray, top_k: int = 10) -> float:
    """Fraction of total in the top_k entries."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    tot = float(x.sum())
    if tot <= 0 or x.size == 0:
        return 0.0
    k = min(int(top_k), x.size)
    return float(np.sort(x)[-k:].sum() / tot)


@dataclass
class StressSample:
    t: int
    L: float
    S: int
    P: float
    tau_max: float
    tau_mean: float
    tau_gini: float
    tau_top10_frac: float


@dataclass
class InfoStressMeter:
    n_sites: int
    out_dir: str = "stress_out"
    support_frac: float = 0.95
    topk_edges: int = 10

    # Internal state
    samples: List[StressSample] = field(default_factory=list)

    def update(
        self,
        t: int,
        site_weights_fn: Callable[[], np.ndarray],
        edge_stress_fn: Callable[[], Dict[Edge, float]],
    ) -> StressSample:
        """Call once per tick."""
        w = np.asarray(site_weights_fn(), dtype=float)
        if w.shape[0] != self.n_sites:
            raise ValueError(f"site_weights length {w.shape[0]} != n_sites {self.n_sites}")

        edge_tau = edge_stress_fn()
        if not isinstance(edge_tau, dict):
            raise TypeError("edge_stress_fn must return dict {(i,j): tau}")

        tau_vals = np.array([float(v) for v in edge_tau.values() if v is not None], dtype=float)
        tau_vals = tau_vals[np.isfinite(tau_vals)]
        tau_vals[tau_vals < 0] = 0.0

        L = float(tau_vals.sum()) if tau_vals.size else 0.0
        S = int(support_size(w, self.support_frac))
        P = float(L / max(S, 1))

        tau_max = float(tau_vals.max()) if tau_vals.size else 0.0
        tau_mean = float(tau_vals.mean()) if tau_vals.size else 0.0
        tau_g = float(gini(tau_vals)) if tau_vals.size else 0.0
        tau_top = float(top_fraction(tau_vals, self.topk_edges)) if tau_vals.size else 0.0

        s = StressSample(
            t=t,
            L=L,
            S=S,
            P=P,
            tau_max=tau_max,
            tau_mean=tau_mean,
            tau_gini=tau_g,
            tau_top10_frac=tau_top,
        )
        self.samples.append(s)
        return s

    def write_csv(self, filename: str = "stress_log.csv") -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "L", "S", "P", "tau_max", "tau_mean", "tau_gini", "tau_topk_frac"])
            for s in self.samples:
                w.writerow([s.t, s.L, s.S, s.P, s.tau_max, s.tau_mean, s.tau_gini, s.tau_top10_frac])
        return path

    def plot(self, filename_prefix: str = "stress") -> List[str]:
        """Save a few core plots as PNGs. Returns list of filepaths."""
        os.makedirs(self.out_dir, exist_ok=True)
        if not self.samples:
            return []

        t = np.array([s.t for s in self.samples], dtype=int)
        L = np.array([s.L for s in self.samples], dtype=float)
        S = np.array([s.S for s in self.samples], dtype=float)
        P = np.array([s.P for s in self.samples], dtype=float)
        g = np.array([s.tau_gini for s in self.samples], dtype=float)
        top = np.array([s.tau_top10_frac for s in self.samples], dtype=float)

        paths: List[str] = []

        # Load & Support
        plt.figure()
        plt.plot(t, L, label="L(t) total edge load")
        plt.plot(t, S, label="S(t) support (95%)")
        plt.xlabel("tick")
        plt.legend()
        p1 = os.path.join(self.out_dir, f"{filename_prefix}_load_support.png")
        plt.savefig(p1, dpi=160, bbox_inches="tight")
        plt.close()
        paths.append(p1)

        # Pressure
        plt.figure()
        plt.plot(t, P, label="P(t)=L/S")
        plt.xlabel("tick")
        plt.legend()
        p2 = os.path.join(self.out_dir, f"{filename_prefix}_pressure.png")
        plt.savefig(p2, dpi=160, bbox_inches="tight")
        plt.close()
        paths.append(p2)

        # Concentration (tube-ness)
        plt.figure()
        plt.plot(t, g, label="Gini(t) stress concentration")
        plt.plot(t, top, label=f"Top-{self.topk_edges} edge fraction")
        plt.xlabel("tick")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        p3 = os.path.join(self.out_dir, f"{filename_prefix}_concentration.png")
        plt.savefig(p3, dpi=160, bbox_inches="tight")
        plt.close()
        paths.append(p3)

        # Phase plot: Pressure vs Support
        plt.figure()
        plt.plot(S, P, marker="o", linestyle="-")
        plt.xlabel("S(t) support")
        plt.ylabel("P(t) pressure")
        p4 = os.path.join(self.out_dir, f"{filename_prefix}_phase_PS.png")
        plt.savefig(p4, dpi=160, bbox_inches="tight")
        plt.close()
        paths.append(p4)

        return paths


# ------------------------------------------------------------
# Example adapter (replace with your sim hooks)
# ------------------------------------------------------------
def _example_usage():
    """
    This is a demonstration scaffold.
    Replace `site_weights()` and `edge_stress()` with your sim's actual accessors.
    """
    N = 32
    rng = np.random.default_rng(0)

    # Fake evolving weights
    weights = np.abs(rng.normal(size=N))
    weights /= weights.sum()

    # Fake edges
    edges = [(i, i + 1) for i in range(N - 1)]

    meter = InfoStressMeter(n_sites=N)

    for t in range(200):
        # pretend excitation localizes over time
        center = int(10 + 8 * math.sin(t / 30))
        weights = np.exp(-0.5 * ((np.arange(N) - center) / max(1.5, 6 - t / 60)) ** 2)
        weights = weights / weights.sum()

        # pretend stress concentrates as it localizes
        tau = {}
        for (i, j) in edges:
            base = 0.01 * rng.random()
            focus = math.exp(-0.5 * ((i - center) / 2.0) ** 2)
            tau[(i, j)] = base + 0.2 * focus

        meter.update(
            t=t,
            site_weights_fn=lambda w=weights: w,
            edge_stress_fn=lambda tau=tau: tau,
        )

    csv_path = meter.write_csv()
    pngs = meter.plot()
    print("Wrote:", csv_path)
    print("Plots:", *pngs, sep="\n  ")


if __name__ == "__main__":
    _example_usage()
