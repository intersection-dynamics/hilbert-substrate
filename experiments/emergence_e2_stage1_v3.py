#!/usr/bin/env python3
r"""
emergence_e2_stage1_v3.py — defense sweep driver (rev4)
======================================================

Goal
----
Produce a defense-ready occupancy-vs-N plot (with seed counts + thresholds printed)
for the E2 Stage-1 accessibility experiment.

Key fixes vs your rev3 output (where final_cost ~ N^p)
------------------------------------------------------
Your output showed final_cost ~ N^p (near the MAX possible cost), meaning the flow
was not descending. This rewrite fixes that with two hard guards:

1) Centered proxy M for the ratio objective C_p (Rayleigh-quotient style):
      M = Σ (w^p - C_p(H)) * c * P
   rather than M = Σ w^p c P (which can push weight to the top sector).

2) Always-step-down line search at every iteration:
   - Try both +dt and -dt steps
   - Pick the one that decreases cost
   - If neither decreases cost, shrink dt (and stop if dt underflows)

This makes "it climbs to the maximum" basically impossible unless something
is fundamentally inconsistent.

Output
------
hsf_out/emergence_e2_stage1_defense/<timestamp>/
  - occupancy_vs_N.png
  - occupancy_vs_N.csv
  - defense_sweep.json

Practical scaling guard
-----------------------
Exact Pauli decomposition is O(4^N). This script SKIPS N > --max-feasible-N (default 8).

Usage
-----
Check you’re running the right file:
  python emergence_e2_stage1_v3.py --version

Run sweep (your usual command):
  python emergence_e2_stage1_v3.py --defense-sweep --Ns 3,4,5,6,7,8,10,12,16,20,27 --seed0 0 --seed1 19 --p 4 --max-iter 400 --eps-spatial-rel 0.25

Tighten/loosen flow if needed:
  --dt 0.05  (default)
  --dt-min 1e-6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product as iprod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

VERSION = "rev4-centered-M-linesearch"

np.set_printoptions(precision=6, suppress=True, linewidth=140)

# ============================================================
# Utilities
# ============================================================

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(base_dir: str, run_name: str) -> Path:
    d = Path(base_dir) / run_name / _now_tag()
    d.mkdir(parents=True, exist_ok=True)
    return d


def json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [json_safe(x) for x in o]
    return o


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================
# Pauli basis
# ============================================================

I2 = np.eye(2, dtype=complex)
PAULIS = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_tensor(label: str) -> np.ndarray:
    out = PAULIS[label[0]]
    for ch in label[1:]:
        out = np.kron(out, PAULIS[ch])
    return out


def all_pauli_labels(N: int) -> List[str]:
    return ["".join(s) for s in iprod("IXYZ", repeat=N)]


def hamming_weight(label: str) -> int:
    return sum(1 for ch in label if ch != "I")


def decompose_pauli(H: np.ndarray, N: int) -> Dict[str, float]:
    """
    Exact Pauli decomposition (O(4^N)).
    """
    dim = 2**N
    coeffs: Dict[str, float] = {}
    for label in all_pauli_labels(N):
        P = pauli_tensor(label)
        c = (np.trace(H @ P).real) / dim
        if abs(c) > 1e-15:
            coeffs[label] = float(c)
    return coeffs


def locality_cost_fast(coeffs: Dict[str, float], p: float) -> float:
    """
    C_p = (Σ w^p c^2) / (Σ c^2)
    """
    num = 0.0
    den = 0.0
    for label, c in coeffs.items():
        w = hamming_weight(label)
        c2 = c * c
        den += c2
        num += (float(w) ** float(p)) * c2
    return float(num / den) if den > 0 else 0.0


def locality_cost(H: np.ndarray, N: int, p: float) -> float:
    return locality_cost_fast(decompose_pauli(H, N), p)


def weight_distribution(coeffs: Dict[str, float], N: int) -> Dict[int, float]:
    by_w = {w: 0.0 for w in range(N + 1)}
    den = 0.0
    for label, c in coeffs.items():
        w = hamming_weight(label)
        c2 = c * c
        by_w[w] += c2
        den += c2
    if den <= 0:
        return {0: 1.0}
    return {w: float(by_w[w] / den) for w in range(N + 1)}


def lw2_fraction(wdist: Dict[int, float]) -> float:
    return float(wdist.get(2, 0.0))


# ============================================================
# Reference Hamiltonians
# ============================================================

def heisenberg_ring(N: int, Jx=1.0, Jy=1.0, Jz=1.0) -> np.ndarray:
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        for P, J in (("X", Jx), ("Y", Jy), ("Z", Jz)):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            H += J * pauli_tensor("".join(lab))
    return 0.5 * (H + H.conj().T)


def diag_spectrum_baseline_cost(H: np.ndarray, N: int, p: float) -> float:
    evals = np.linalg.eigvalsh(H)
    H_diag = np.diag(evals.astype(complex))
    return locality_cost(H_diag, N, p)


# ============================================================
# Scrambling
# ============================================================

def random_hermitian(dim: int) -> np.ndarray:
    A = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / math.sqrt(2.0)
    return A + A.conj().T


def scramble(H: np.ndarray, N: int, depth: int) -> np.ndarray:
    dim = 2**N
    U = np.eye(dim, dtype=complex)
    for _ in range(max(1, depth)):
        G = random_hermitian(dim)
        U = expm(-1j * 0.03 * G) @ U
    Hs = U @ H @ U.conj().T
    return 0.5 * (Hs + Hs.conj().T)


# ============================================================
# Stabilized double-bracket flow with line search
# ============================================================

@dataclass
class FlowResult:
    trace: List[float]
    steps: int
    elapsed_sec: float
    stall_reason: str
    diverged: bool
    final_weight_dist: Dict[int, float]
    dt_final: float


def _comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def _fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def compute_centered_M(coeffs: Dict[str, float], N: int, p: float, cost: float) -> np.ndarray:
    """
    Centered proxy for ratio objective:
      M = Σ (w^p - cost) * c * P
    """
    dim = 2**N
    M = np.zeros((dim, dim), dtype=complex)
    for label, c in coeffs.items():
        w = hamming_weight(label)
        if w == 0:
            continue
        M += (float(w) ** float(p) - float(cost)) * float(c) * pauli_tensor(label)
    return 0.5 * (M + M.conj().T)


def run_flow(
    H0: np.ndarray,
    N: int,
    p: float,
    *,
    max_iter: int,
    dt: float,
    dt_min: float,
    stall_tol: float,
    trace_every: int = 1,
) -> FlowResult:
    """
    Descent guard:
      - centered M
      - try +/- step, choose best
      - shrink dt if neither improves
      - renormalize Frobenius norm each accepted step
    """
    t0 = time.time()
    H = np.array(H0, copy=True)
    H = 0.5 * (H + H.conj().T)

    target_norm = _fro(H) + 1e-18
    trace: List[float] = []
    last_cost: Optional[float] = None
    diverged = False
    stall_reason = "max_iter"

    cur_dt = float(dt)

    for it in range(int(max_iter)):
        coeffs = decompose_pauli(H, N)
        cost = float(locality_cost_fast(coeffs, p))

        if it % max(1, trace_every) == 0:
            trace.append(cost)

        if last_cost is not None and abs(last_cost - cost) < stall_tol:
            stall_reason = "stall_tol"
            break
        last_cost = cost

        M = compute_centered_M(coeffs, N, p, cost)

        # canonical double-commutator direction
        dH = -_comm(_comm(H, M), H)
        dnorm = _fro(dH)

        if not math.isfinite(dnorm) or dnorm > 1e200:
            diverged = True
            stall_reason = "diverged_dnorm"
            break

        # scale step by dH norm to avoid blow-up
        dt_eff = cur_dt / (1.0 + dnorm)

        # propose forward/backward
        H_fwd = H + dt_eff * dH
        H_bwd = H - dt_eff * dH

        # symmetrize
        H_fwd = 0.5 * (H_fwd + H_fwd.conj().T)
        H_bwd = 0.5 * (H_bwd + H_bwd.conj().T)

        # renormalize scale
        H_fwd *= (target_norm / (_fro(H_fwd) + 1e-18))
        H_bwd *= (target_norm / (_fro(H_bwd) + 1e-18))

        if not np.isfinite(H_fwd).all() or not np.isfinite(H_bwd).all():
            diverged = True
            stall_reason = "diverged_nonfinite_trial"
            break

        # evaluate costs
        cost_fwd = float(locality_cost_fast(decompose_pauli(H_fwd, N), p))
        cost_bwd = float(locality_cost_fast(decompose_pauli(H_bwd, N), p))

        # accept the best improving step
        if cost_fwd <= cost and cost_fwd <= cost_bwd:
            H = H_fwd
            # keep dt as-is
        elif cost_bwd <= cost:
            H = H_bwd
        else:
            # neither improved: shrink dt and try again next iter
            cur_dt *= 0.5
            if cur_dt < dt_min:
                stall_reason = "dt_underflow"
                break
            continue

    elapsed = time.time() - t0
    final_coeffs = decompose_pauli(H, N) if not diverged else decompose_pauli(H0, N)
    wdist = weight_distribution(final_coeffs, N)

    return FlowResult(
        trace=trace,
        steps=int(it + 1),
        elapsed_sec=float(elapsed),
        stall_reason=str(stall_reason),
        diverged=bool(diverged),
        final_weight_dist=wdist,
        dt_final=float(cur_dt),
    )


# ============================================================
# Defense sweep
# ============================================================

def defense_sweep(
    *,
    base_dir: str,
    run_name: str,
    Ns: List[int],
    seed0: int,
    seed1: int,
    p: float,
    max_iter: int,
    dt: float,
    dt_min: float,
    stall_tol: float,
    scramble_depth_factor: int,
    eps_spatial_rel: float,
    lw2_min: float,
    max_feasible_N: int,
) -> None:
    import matplotlib.pyplot as plt

    run_dir = make_run_dir(base_dir, run_name)
    out_csv = run_dir / "occupancy_vs_N.csv"
    out_png = run_dir / "occupancy_vs_N.png"
    out_json = run_dir / "defense_sweep.json"

    seeds = list(range(seed0, seed1 + 1))
    rows: List[dict] = []
    perN: Dict[int, dict] = {}

    print("=" * 70)
    print("DEFENSE SWEEP (occupancy vs N)")
    print(f"version: {VERSION}")
    print(f"run_dir: {run_dir}")
    print(f"Ns: {Ns}")
    print(f"seeds: {seed0}..{seed1} (count={len(seeds)})")
    print(f"p={p} max_iter={max_iter} dt={dt} dt_min={dt_min} stall_tol={stall_tol}")
    print(f"scramble_depth={scramble_depth_factor}*N")
    print(f"eps_spatial_rel={eps_spatial_rel} lw2_min={lw2_min}")
    print(f"max_feasible_N={max_feasible_N}")
    print("=" * 70)

    for N in Ns:
        counts = {
            "harm_nearest": 0,
            "spatial_nearest": 0,
            "spatial_thr": 0,
            "diverged": 0,
            "skipped": 0,
            "total": 0,
        }

        if N > max_feasible_N:
            print(f"\n--- N={N} --- SKIPPED (N>{max_feasible_N})")
            counts["skipped"] = len(seeds)
            counts["total"] = len(seeds)
            perN[N] = counts
            for seed in seeds:
                rows.append({"N": N, "seed": seed, "label": "skipped", "reason": f"N>{max_feasible_N}"})
            continue

        H_local = heisenberg_ring(N)
        c_spat = float(locality_cost(H_local, N, p))
        c_base = float(diag_spectrum_baseline_cost(H_local, N, p))

        spat_lo = c_spat * (1.0 - eps_spatial_rel)
        spat_hi = c_spat * (1.0 + eps_spatial_rel)

        finals: List[float] = []
        inits: List[float] = []

        print(f"\n--- N={N} ---")
        print(f"  spatial_target=C_p(H_local)={c_spat:.6f}")
        print(f"  diag_baseline              ={c_base:.6f}")
        print(f"  spatial band=[{spat_lo:.3f},{spat_hi:.3f}]")

        for seed in seeds:
            np.random.seed(seed)
            H_scr = scramble(H_local, N, depth=scramble_depth_factor * N)

            c_init = float(locality_cost(H_scr, N, p))
            inits.append(c_init)

            flow = run_flow(
                H_scr, N, p,
                max_iter=max_iter,
                dt=dt,
                dt_min=dt_min,
                stall_tol=stall_tol,
                trace_every=1,
            )

            c_final = float(flow.trace[-1]) if flow.trace else float("nan")
            finals.append(c_final)

            lw2 = lw2_fraction(flow.final_weight_dist)

            if flow.diverged or (not math.isfinite(c_final)):
                counts["diverged"] += 1
                label = "diverged"
            else:
                # forced-choice occupancy (always assigns)
                dh = abs(c_final - c_base)
                ds = abs(c_final - c_spat)
                if dh < ds:
                    counts["harm_nearest"] += 1
                    label = "harm_nearest"
                else:
                    counts["spatial_nearest"] += 1
                    label = "spatial_nearest"

                # optional threshold bookkeeping for spatial
                if (spat_lo <= c_final <= spat_hi) and (lw2 >= lw2_min):
                    counts["spatial_thr"] += 1

            counts["total"] += 1

            rows.append({
                "N": N,
                "seed": seed,
                "p": p,
                "scramble_depth": scramble_depth_factor * N,
                "init_cost": c_init,
                "final_cost": c_final if math.isfinite(c_final) else None,
                "diag_baseline_cost": c_base,
                "spatial_target_cost": c_spat,
                "spatial_lo": spat_lo,
                "spatial_hi": spat_hi,
                "eps_spatial_rel": eps_spatial_rel,
                "lw2": float(lw2),
                "lw2_min": lw2_min,
                "label": label,
                "diverged": bool(flow.diverged),
                "stall_reason": flow.stall_reason,
                "steps": flow.steps,
                "elapsed_sec": flow.elapsed_sec,
                "dt_final": flow.dt_final,
            })

        perN[N] = counts

        inits_sorted = sorted([x for x in inits if math.isfinite(x)])
        finals_sorted = sorted([x for x in finals if math.isfinite(x)])
        if inits_sorted and finals_sorted:
            init_med = inits_sorted[len(inits_sorted)//2]
            fin_med = finals_sorted[len(finals_sorted)//2]
            print(f"  init_cost  min/med/max = {inits_sorted[0]:.4f} / {init_med:.4f} / {inits_sorted[-1]:.4f}")
            print(f"  final_cost min/med/max = {finals_sorted[0]:.4f} / {fin_med:.4f} / {finals_sorted[-1]:.4f}")

        print(f"  counts: harm_nearest={counts['harm_nearest']} spatial_nearest={counts['spatial_nearest']} spatial_thr={counts['spatial_thr']} diverged={counts['diverged']} total={counts['total']}")

    # CSV output
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot forced-choice occupancy P_harm_nearest(N)
    import matplotlib.pyplot as plt
    Ns_sorted = sorted(perN.keys())
    P_harm = []
    SE = []
    n_per = []
    for N in Ns_sorted:
        c = perN[N]
        n = max(1, c["total"])
        ph = c["harm_nearest"] / n
        se = math.sqrt(ph * (1 - ph) / n) if n else 0.0
        P_harm.append(ph)
        SE.append(se)
        n_per.append(n)

    plt.figure(figsize=(10, 5.6))
    plt.errorbar(Ns_sorted, P_harm, yerr=SE, fmt="o-", capsize=4)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("System size N")
    plt.ylabel("P_harm_nearest(N)")
    plt.title("HSF accessibility sweep: forced-choice basin occupancy vs N")

    for N, ph, n in zip(Ns_sorted, P_harm, n_per):
        plt.annotate(f"n={n}", (N, ph), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    txt = (
        "Order parameter (forced-choice):\n"
        "  harm_win := |final_cost - diag_baseline| < |final_cost - spatial_target|\n\n"
        f"p={p}\nmax_iter={max_iter}\ndt={dt}\ndt_min={dt_min}\nstall_tol={stall_tol}\n"
        f"scramble_depth={scramble_depth_factor}*N\n"
        f"eps_spatial_rel={eps_spatial_rel}\nlw2_min={lw2_min}\n"
        f"max_feasible_N={max_feasible_N}\n"
        f"version={VERSION}\n"
    )
    plt.gcf().text(0.02, 0.02, txt, fontsize=9, family="monospace", va="bottom")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    payload = {
        "version": VERSION,
        "run_info": {
            "timestamp_local": datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cwd": str(Path.cwd()),
            "run_dir": str(run_dir),
        },
        "config": {
            "Ns": Ns,
            "seed0": seed0,
            "seed1": seed1,
            "p": p,
            "max_iter": max_iter,
            "dt": dt,
            "dt_min": dt_min,
            "stall_tol": stall_tol,
            "scramble_depth_factor": scramble_depth_factor,
            "eps_spatial_rel": eps_spatial_rel,
            "lw2_min": lw2_min,
            "max_feasible_N": max_feasible_N,
        },
        "perN_counts": perN,
        "files": {"csv": str(out_csv), "png": str(out_png)},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)

    print("\n=== DEFENSE SWEEP COMPLETE ===")
    print(f"CSV: {out_csv}")
    print(f"FIG: {out_png}")
    print(f"JSON:{out_json}")


# ============================================================
# CLI
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--defense-sweep", action="store_true")
    ap.add_argument("--base-dir", type=str, default="hsf_out")
    ap.add_argument("--run-name", type=str, default="emergence_e2_stage1_defense")
    ap.add_argument("--Ns", type=str, default="3,4,5,6,7,8")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seed1", type=int, default=19)
    ap.add_argument("--p", type=float, default=4.0)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--dt-min", type=float, default=1e-6)
    ap.add_argument("--stall-tol", type=float, default=1e-12)
    ap.add_argument("--scramble-depth-factor", type=int, default=2)
    ap.add_argument("--eps-spatial-rel", type=float, default=0.25)
    ap.add_argument("--lw2-min", type=float, default=0.0)
    ap.add_argument("--max-feasible-N", type=int, default=8)
    args = ap.parse_args()

    if args.version:
        print(VERSION)
        return 0

    if not args.defense_sweep:
        print("Nothing to do (use --defense-sweep).")
        return 0

    defense_sweep(
        base_dir=args.base_dir,
        run_name=args.run_name,
        Ns=_parse_int_list(args.Ns),
        seed0=args.seed0,
        seed1=args.seed1,
        p=float(args.p),
        max_iter=int(args.max_iter),
        dt=float(args.dt),
        dt_min=float(args.dt_min),
        stall_tol=float(args.stall_tol),
        scramble_depth_factor=int(args.scramble_depth_factor),
        eps_spatial_rel=float(args.eps_spatial_rel),
        lw2_min=float(args.lw2_min),
        max_feasible_N=int(args.max_feasible_N),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())