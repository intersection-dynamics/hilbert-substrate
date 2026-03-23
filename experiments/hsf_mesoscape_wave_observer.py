#!/usr/bin/env python3
# filename: hsf_mesoscape_wave_observer.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import hsf_mesoscale_bookkeeping as bk
import hsf_mesoscale_physics_core as phys
from hsf_mesoscale_physics_core import PhysicsConfig, GM_MATRICES


Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HSF mesoscale wave observer. "
            "Evolves psi on the fixed lawful substrate setup and measures whether mean "
            "entanglement / correlator traces show oscillatory or beat-like structure."
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

    p.add_argument("--w-mi", type=float, default=1.0)
    p.add_argument("--w-corr", type=float, default=0.25)
    p.add_argument("--corr-scale", type=float, default=4.0)
    p.add_argument("--top-pairs", type=int, default=8)
    p.add_argument("--json-out", type=str, default="hsf_mesoscape_wave_observer.json")
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


def compute_pair_matrices(psi, n_sites: int, n_max: int, xp, args: argparse.Namespace):
    mi = np.zeros((n_max, n_max), dtype=np.float64)
    corr = np.zeros((n_max, n_max), dtype=np.float64)
    ent = np.zeros((n_max, n_max), dtype=np.float64)

    for i in range(n_max):
        for j in range(i + 1, n_max):
            mij = float(bk.mutual_information_from_state(psi, i, j, n_sites, xp))
            cij = float(bk.pair_su3_correlator_strength(psi, i, j, GM_MATRICES, xp))
            eij = float(
                args.w_mi * np.tanh(max(0.0, mij))
                + args.w_corr * np.tanh(args.corr_scale * max(0.0, cij))
            )
            mi[i, j] = mi[j, i] = mij
            corr[i, j] = corr[j, i] = cij
            ent[i, j] = ent[j, i] = eij

    return mi, corr, ent


def top_entangled_pairs(ent: np.ndarray, mi: np.ndarray, corr: np.ndarray, k: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = ent.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "pair": [int(i), int(j)],
                    "entanglement": float(ent[i, j]),
                    "mutual_information": float(mi[i, j]),
                    "correlator": float(corr[i, j]),
                }
            )
    rows.sort(key=lambda r: r["entanglement"], reverse=True)
    return rows[:k]


def mean_upper_triangle(mat: np.ndarray) -> float:
    n = mat.shape[0]
    count = max(1, n * (n - 1) // 2)
    return float(np.sum(np.triu(mat, 1)) / count)


def find_turning_points(trace: List[float]) -> Dict[str, Any]:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.size < 3:
        return {"maxima": [], "minima": [], "reversal_steps": [], "half_period_estimates": []}

    diffs = np.diff(arr)
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


def pick_reference_pair(initial_top_pairs: List[Dict[str, Any]]) -> Optional[Edge]:
    if not initial_top_pairs:
        return None
    pair = initial_top_pairs[0]["pair"]
    return (int(pair[0]), int(pair[1]))


def run_observer(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = build_config(args)
    xp, is_gpu = phys.get_array_module(args.device)

    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs, _rng = phys.init_state(cfg, xp)
    prepared = (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs)

    mean_mi_trace: List[float] = []
    mean_corr_trace: List[float] = []
    mean_ent_trace: List[float] = []

    reference_pair: Optional[Edge] = None
    ref_pair_ent_trace: List[float] = []
    ref_pair_corr_trace: List[float] = []
    ref_pair_mi_trace: List[float] = []

    core_pair_trace: List[Optional[List[int]]] = []
    core_pair_ent_trace: List[Optional[float]] = []
    core_pair_corr_trace: List[Optional[float]] = []
    core_pair_mi_trace: List[Optional[float]] = []

    snapshots: List[Dict[str, Any]] = []

    for step in range(1, args.total_steps + 1):
        prepared = phys.evolve_prepared_state(prepared, cfg, xp)
        psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = prepared
        n_sites = int(psi.ndim)

        mi, corr, ent = compute_pair_matrices(psi, n_sites, cfg.n_max, xp, args)

        mean_mi_trace.append(mean_upper_triangle(mi))
        mean_corr_trace.append(mean_upper_triangle(corr))
        mean_ent_trace.append(mean_upper_triangle(ent))

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

        top_pairs = top_entangled_pairs(ent, mi, corr, args.top_pairs)
        if reference_pair is None:
            reference_pair = pick_reference_pair(top_pairs)

        if reference_pair is not None:
            i, j = reference_pair
            ref_pair_ent_trace.append(float(ent[i, j]))
            ref_pair_corr_trace.append(float(corr[i, j]))
            ref_pair_mi_trace.append(float(mi[i, j]))
        else:
            ref_pair_ent_trace.append(0.0)
            ref_pair_corr_trace.append(0.0)
            ref_pair_mi_trace.append(0.0)

        core_pair: Optional[Edge] = None
        if core is not None and core.get("core_pair") is not None:
            cp = core["core_pair"]
            if isinstance(cp, (list, tuple)) and len(cp) == 2:
                core_pair = (int(cp[0]), int(cp[1]))

        if core_pair is not None:
            i, j = core_pair
            core_pair_trace.append([int(i), int(j)])
            core_pair_ent_trace.append(float(ent[i, j]))
            core_pair_corr_trace.append(float(corr[i, j]))
            core_pair_mi_trace.append(float(mi[i, j]))
        else:
            core_pair_trace.append(None)
            core_pair_ent_trace.append(None)
            core_pair_corr_trace.append(None)
            core_pair_mi_trace.append(None)

        if step == 1 or step % args.snapshot_every == 0 or step == args.total_steps:
            metrics = phys.metric_snapshot(active_nodes, active_edges, edge_strengths)
            gstats = phys.graph_stats(active_nodes, active_edges)

            snapshots.append(
                {
                    "step": int(step),
                    "active_nodes": sorted(int(i) for i in active_nodes),
                    "dormant_nodes": sorted(int(i) for i in dormant_nodes),
                    "active_edges": [list(e) for e in sorted(active_edges)],
                    "mean_mutual_information": mean_mi_trace[-1],
                    "mean_correlator": mean_corr_trace[-1],
                    "mean_entanglement": mean_ent_trace[-1],
                    "reference_pair": [int(reference_pair[0]), int(reference_pair[1])] if reference_pair is not None else None,
                    "reference_pair_entanglement": ref_pair_ent_trace[-1],
                    "reference_pair_correlator": ref_pair_corr_trace[-1],
                    "reference_pair_mutual_information": ref_pair_mi_trace[-1],
                    "dominant_core": core,
                    "dominant_core_pair_entanglement": core_pair_ent_trace[-1],
                    "dominant_core_pair_correlator": core_pair_corr_trace[-1],
                    "dominant_core_pair_mutual_information": core_pair_mi_trace[-1],
                    "top_entangled_pairs": top_pairs,
                    "metric": metrics,
                    "graph_stats": gstats,
                }
            )

        if args.progress_every > 0 and step % args.progress_every == 0:
            cp = None
            if core is not None and core.get("core_pair") is not None:
                cp = core.get("core_pair")
            print(
                f"[step {step:04d}] mean_ent={mean_ent_trace[-1]:.6f} "
                f"mean_corr={mean_corr_trace[-1]:.6f} mean_mi={mean_mi_trace[-1]:.6f} "
                f"ref_ent={ref_pair_ent_trace[-1]:.6f} core={cp}"
            )

    wave_summary = {
        "mean_entanglement_turning_points": find_turning_points(mean_ent_trace),
        "mean_correlator_turning_points": find_turning_points(mean_corr_trace),
        "reference_pair_entanglement_turning_points": find_turning_points(ref_pair_ent_trace),
        "reference_pair_correlator_turning_points": find_turning_points(ref_pair_corr_trace),
        "mean_entanglement_fft": dominant_fft_component(mean_ent_trace, args.dt),
        "mean_correlator_fft": dominant_fft_component(mean_corr_trace, args.dt),
        "reference_pair_entanglement_fft": dominant_fft_component(ref_pair_ent_trace, args.dt),
        "reference_pair_correlator_fft": dominant_fft_component(ref_pair_corr_trace, args.dt),
        "phase_lock_mean_ent_vs_mean_corr": phase_lock_score(mean_ent_trace, mean_corr_trace),
        "phase_lock_ref_ent_vs_ref_corr": phase_lock_score(ref_pair_ent_trace, ref_pair_corr_trace),
    }

    return {
        "script": "hsf_mesoscape_wave_observer.py",
        "observer_mode": True,
        "physics_config": asdict(cfg),
        "observer_config": {
            "w_mi": float(args.w_mi),
            "w_corr": float(args.w_corr),
            "corr_scale": float(args.corr_scale),
            "top_pairs": int(args.top_pairs),
            "principle": "observe oscillatory / beat-like structure in entanglement and correlator traces without choosing subsystem moves",
        },
        "reference_pair": [int(reference_pair[0]), int(reference_pair[1])] if reference_pair is not None else None,
        "mean_mutual_information_trace": mean_mi_trace,
        "mean_correlator_trace": mean_corr_trace,
        "mean_entanglement_trace": mean_ent_trace,
        "reference_pair_entanglement_trace": ref_pair_ent_trace,
        "reference_pair_correlator_trace": ref_pair_corr_trace,
        "reference_pair_mutual_information_trace": ref_pair_mi_trace,
        "dominant_core_pair_trace": core_pair_trace,
        "dominant_core_pair_entanglement_trace": core_pair_ent_trace,
        "dominant_core_pair_correlator_trace": core_pair_corr_trace,
        "dominant_core_pair_mutual_information_trace": core_pair_mi_trace,
        "wave_summary": wave_summary,
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