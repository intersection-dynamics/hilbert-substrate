#!/usr/bin/env python3
# filename: hsf_mesoscape_observer_v2.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import hsf_mesoscale_bookkeeping as bk
import hsf_mesoscale_physics_core as phys
from hsf_mesoscale_physics_core import PhysicsConfig, GM_MATRICES, apply_one_body

Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HSF mesoscale observer v2. Evolves psi on a fixed lawful substrate setup and "
            "reports MI-first, connected-correlator diagnostics, with an initial-state switch."
        )
    )
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--n-init", type=int, default=2)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--local-scale", type=float, default=0.15)
    p.add_argument("--pair-scale", type=float, default=0.12)
    p.add_argument("--spawn-pair-scale", type=float, default=0.11)
    p.add_argument("--total-steps", type=int, default=300)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--snapshot-every", type=int, default=10)

    p.add_argument("--initial-state", choices=["basis_zero", "random", "perturbed_zero"], default="basis_zero")
    p.add_argument("--perturb-eps", type=float, default=0.02)

    p.add_argument("--w-mi", type=float, default=1.0)
    p.add_argument("--w-ccorr", type=float, default=0.25)
    p.add_argument("--ccorr-scale", type=float, default=4.0)
    p.add_argument("--top-pairs", type=int, default=8)
    p.add_argument("--mi-warning-threshold", type=float, default=1e-10)
    p.add_argument("--json-out", type=str, default="hsf_mesoscape_observer_v2.json")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> PhysicsConfig:
    return PhysicsConfig(
        n_max=args.n_max,
        n_init=args.n_init,
        seed=args.seed,
        local_scale=args.local_scale,
        pair_scale=args.pair_scale,
        spawn_pair_scale=args.spawn_pair_scale,
        total_steps=1,
        dt=args.dt,
        eval_every=1,
        lookahead_windows=1,
        weaken_factor=0.55,
        progress_every=args.progress_every,
        device=args.device,
    )


def random_state(n_sites: int, xp, seed: int):
    rng = np.random.default_rng(seed)
    shape = (3,) * n_sites
    real = rng.normal(size=shape)
    imag = rng.normal(size=shape)
    arr = real + 1j * imag
    arr = arr / np.linalg.norm(arr.ravel())
    return xp.asarray(arr, dtype=xp.complex128)


def perturbed_zero_state(n_sites: int, xp, seed: int, eps: float):
    psi0 = phys.basis_state_zero(n_sites, xp)
    psi_rand = random_state(n_sites, xp, seed)
    psi = (1.0 - eps) * psi0 + eps * psi_rand
    norm = xp.sqrt(xp.real(xp.vdot(psi, psi)))
    return psi / norm


def make_initial_state(initial_state: str, n_sites: int, xp, seed: int, perturb_eps: float):
    if initial_state == "basis_zero":
        return phys.basis_state_zero(n_sites, xp)
    if initial_state == "random":
        return random_state(n_sites, xp, seed)
    if initial_state == "perturbed_zero":
        return perturbed_zero_state(n_sites, xp, seed, perturb_eps)
    raise ValueError(initial_state)


def init_prepared_state(cfg: PhysicsConfig, xp, initial_state: str, perturb_eps: float):
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs, _rng = phys.init_state(cfg, xp)
    psi = make_initial_state(initial_state, cfg.n_max, xp, cfg.seed, perturb_eps)
    return psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs


def connected_pair_correlator_strength(psi, i: int, j: int, GM, xp) -> float:
    vals: List[float] = []
    for gm in GM:
        op = xp.asarray(gm, dtype=xp.complex128)
        oi = apply_one_body(psi, op, i, xp)
        oj = apply_one_body(psi, op, j, xp)
        e_i = xp.real(xp.vdot(psi, oi))
        e_j = xp.real(xp.vdot(psi, oj))
        oij = apply_one_body(oj, op, i, xp)
        e_ij = xp.real(xp.vdot(psi, oij))
        connected = e_ij - e_i * e_j
        vals.append(float(abs(connected)))
    return float(np.mean(vals) if vals else 0.0)


def compute_pair_matrices(psi, n_sites: int, n_max: int, xp, args: argparse.Namespace):
    mi = np.zeros((n_max, n_max), dtype=np.float64)
    ccorr = np.zeros((n_max, n_max), dtype=np.float64)
    signal = np.zeros((n_max, n_max), dtype=np.float64)

    for i in range(n_max):
        for j in range(i + 1, n_max):
            mij = float(bk.mutual_information_from_state(psi, i, j, n_sites, xp))
            ccij = float(connected_pair_correlator_strength(psi, i, j, GM_MATRICES, xp))
            sij = float(
                args.w_mi * np.tanh(max(0.0, mij))
                + args.w_ccorr * np.tanh(args.ccorr_scale * max(0.0, ccij))
            )
            mi[i, j] = mi[j, i] = mij
            ccorr[i, j] = ccorr[j, i] = ccij
            signal[i, j] = signal[j, i] = sij
    return mi, ccorr, signal


def mean_upper_triangle(mat: np.ndarray) -> float:
    n = mat.shape[0]
    count = max(1, n * (n - 1) // 2)
    return float(np.sum(np.triu(mat, 1)) / count)


def top_pairs(signal: np.ndarray, mi: np.ndarray, ccorr: np.ndarray, k: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = signal.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "pair": [int(i), int(j)],
                    "signal": float(signal[i, j]),
                    "mutual_information": float(mi[i, j]),
                    "connected_correlator": float(ccorr[i, j]),
                }
            )
    rows.sort(key=lambda r: r["signal"], reverse=True)
    return rows[:k]


def find_turning_points(trace: List[float]) -> Dict[str, Any]:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.size < 3:
        return {"maxima": [], "minima": [], "reversal_steps": [], "half_period_estimates": []}

    maxima: List[Dict[str, Any]] = []
    minima: List[Dict[str, Any]] = []
    reversal_steps: List[int] = []

    for i in range(1, len(arr) - 1):
        left = arr[i] - arr[i - 1]
        right = arr[i + 1] - arr[i]
        if left > 0.0 and right < 0.0:
            maxima.append({"step_index": int(i + 1), "value": float(arr[i])})
            reversal_steps.append(int(i + 1))
        elif left < 0.0 and right > 0.0:
            minima.append({"step_index": int(i + 1), "value": float(arr[i])})
            reversal_steps.append(int(i + 1))

    half_period_estimates: List[int] = []
    for a, b in zip(reversal_steps[:-1], reversal_steps[1:]):
        half_period_estimates.append(int(b - a))

    return {
        "maxima": maxima,
        "minima": minima,
        "reversal_steps": reversal_steps,
        "half_period_estimates": half_period_estimates,
    }


def dominant_fft_component(trace: List[float], dt: float) -> Dict[str, Any]:
    arr = np.asarray(trace, dtype=np.float64)
    n = arr.size
    if n < 4:
        return {"frequency": None, "period_steps": None, "period_time": None, "power": None}

    centered = arr - np.mean(arr)
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)
    if power.size <= 1:
        return {"frequency": None, "period_steps": None, "period_time": None, "power": None}
    power[0] = 0.0
    idx = int(np.argmax(power))
    freq = float(freqs[idx])
    if freq <= 0.0:
        return {"frequency": None, "period_steps": None, "period_time": None, "power": None}
    period_time = float(1.0 / freq)
    period_steps = float(period_time / dt)
    return {
        "frequency": freq,
        "period_steps": period_steps,
        "period_time": period_time,
        "power": float(power[idx]),
    }


def phase_lock_score(trace_a: List[float], trace_b: List[float]) -> float:
    a = np.asarray(trace_a, dtype=np.float64)
    b = np.asarray(trace_b, dtype=np.float64)
    if a.size < 3 or b.size < 3 or a.size != b.size:
        return 0.0
    da = np.sign(np.diff(a))
    db = np.sign(np.diff(b))
    return float(np.mean(da == db))


def run_observer(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = build_config(args)
    xp, is_gpu = phys.get_array_module(args.device)
    prepared = init_prepared_state(cfg, xp, args.initial_state, args.perturb_eps)

    mean_mi_trace: List[float] = []
    mean_ccorr_trace: List[float] = []
    mean_signal_trace: List[float] = []
    snapshots: List[Dict[str, Any]] = []

    for step in range(1, args.total_steps + 1):
        prepared = phys.evolve_prepared_state(prepared, cfg, xp)
        psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = prepared
        n_sites = int(psi.ndim)

        mi, ccorr, signal = compute_pair_matrices(psi, n_sites, cfg.n_max, xp, args)
        mean_mi_trace.append(mean_upper_triangle(mi))
        mean_ccorr_trace.append(mean_upper_triangle(ccorr))
        mean_signal_trace.append(mean_upper_triangle(signal))

        core = bk.dominant_core_snapshot(
            psi,
            active_nodes,
            active_edges,
            edge_strengths,
            GM_MATRICES,
            xp,
            n_sites,
            None,
            link_regs,
        )

        if step == 1 or step % args.snapshot_every == 0 or step == args.total_steps:
            metrics = phys.metric_snapshot(active_nodes, active_edges, edge_strengths)
            gstats = phys.graph_stats(active_nodes, active_edges)
            rows = top_pairs(signal, mi, ccorr, args.top_pairs)
            snapshots.append(
                {
                    "step": int(step),
                    "active_nodes": sorted(int(i) for i in active_nodes),
                    "dormant_nodes": sorted(int(i) for i in dormant_nodes),
                    "active_edges": [list(e) for e in sorted(active_edges)],
                    "mean_mutual_information": mean_mi_trace[-1],
                    "mean_connected_correlator": mean_ccorr_trace[-1],
                    "mean_signal": mean_signal_trace[-1],
                    "dominant_core": core,
                    "top_pairs": rows,
                    "metric": metrics,
                    "graph_stats": gstats,
                }
            )

        if args.progress_every > 0 and step % args.progress_every == 0:
            cp = None
            if core is not None and core.get("core_pair") is not None:
                cp = core.get("core_pair")
            print(
                f"[step {step:04d}] mean_mi={mean_mi_trace[-1]:.6e} "
                f"mean_ccorr={mean_ccorr_trace[-1]:.6e} mean_signal={mean_signal_trace[-1]:.6e} core={cp}"
            )

    max_abs_mi = float(np.max(np.abs(np.asarray(mean_mi_trace, dtype=np.float64)))) if mean_mi_trace else 0.0
    mi_warning = None
    if max_abs_mi < float(args.mi_warning_threshold):
        mi_warning = (
            "Mutual information stayed below the configured warning threshold for the full run; "
            "any composite pair signal is therefore dominated by the connected-correlator channel, not MI."
        )

    summary = {
        "mean_mutual_information_turning_points": find_turning_points(mean_mi_trace),
        "mean_connected_correlator_turning_points": find_turning_points(mean_ccorr_trace),
        "mean_signal_turning_points": find_turning_points(mean_signal_trace),
        "mean_mutual_information_fft": dominant_fft_component(mean_mi_trace, args.dt),
        "mean_connected_correlator_fft": dominant_fft_component(mean_ccorr_trace, args.dt),
        "mean_signal_fft": dominant_fft_component(mean_signal_trace, args.dt),
        "phase_lock_mi_vs_ccorr": phase_lock_score(mean_mi_trace, mean_ccorr_trace),
        "phase_lock_ccorr_vs_signal": phase_lock_score(mean_ccorr_trace, mean_signal_trace),
        "max_abs_mean_mutual_information": max_abs_mi,
        "mi_warning": mi_warning,
    }

    return {
        "script": "hsf_mesoscape_observer_v2.py",
        "observer_mode": True,
        "physics_config": asdict(cfg),
        "observer_config": {
            "initial_state": args.initial_state,
            "perturb_eps": float(args.perturb_eps),
            "w_mi": float(args.w_mi),
            "w_ccorr": float(args.w_ccorr),
            "ccorr_scale": float(args.ccorr_scale),
            "top_pairs": int(args.top_pairs),
            "mi_warning_threshold": float(args.mi_warning_threshold),
            "principle": "observe the substrate without choosing births or pruning; use connected correlators and warn when MI remains numerically zero",
        },
        "mean_mutual_information_trace": mean_mi_trace,
        "mean_connected_correlator_trace": mean_ccorr_trace,
        "mean_signal_trace": mean_signal_trace,
        "summary": summary,
        "snapshots": snapshots,
        "gpu_enabled": bool(is_gpu),
    }


def main() -> None:
    args = parse_args()
    result = run_observer(args)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()