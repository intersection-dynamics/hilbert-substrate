#!/usr/bin/env python3
# filename: hsf_mesoscape_fission_observer.py

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
            "HSF one-to-two fission observer. Starts with one active subsystem and observes whether "
            "a second subsystem candidate becomes persistently distinguished from the dormant background, "
            "without applying any birth/pruning moves."
        )
    )
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--local-scale", type=float, default=0.15)
    p.add_argument("--pair-scale", type=float, default=0.5)
    p.add_argument("--spawn-pair-scale", type=float, default=0.11)
    p.add_argument("--total-steps", type=int, default=300)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--snapshot-every", type=int, default=10)

    p.add_argument("--initial-state", choices=["basis_zero", "random", "perturbed_zero"], default="perturbed_zero")
    p.add_argument("--perturb-eps", type=float, default=0.02)

    p.add_argument("--w-mi", type=float, default=1.0)
    p.add_argument("--w-ccorr", type=float, default=0.25)
    p.add_argument("--ccorr-scale", type=float, default=4.0)

    p.add_argument("--candidate-threshold-mi", type=float, default=1e-7)
    p.add_argument("--candidate-threshold-ccorr", type=float, default=1e-7)
    p.add_argument("--candidate-window", type=int, default=5)
    p.add_argument("--margin-threshold", type=float, default=1e-8)
    p.add_argument("--consensus-window", type=int, default=10)
    p.add_argument("--top-candidates", type=int, default=5)

    p.add_argument("--json-out", type=str, default="hsf_mesoscape_fission_observer.json")
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


def find_active_reference(active_nodes: Set[int]) -> int:
    if not active_nodes:
        raise ValueError("No active nodes present for fission observer.")
    return int(sorted(active_nodes)[0])


def longest_positive_run(trace: List[float], threshold: float) -> Dict[str, Any]:
    best_len = 0
    best_start = None
    best_end = None
    cur_len = 0
    cur_start = None

    for idx, x in enumerate(trace, start=1):
        val = float(x)
        if val > threshold:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = idx - 1
            cur_len = 0
            cur_start = None

    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start
        best_end = len(trace)

    return {
        "length": int(best_len),
        "start_step": None if best_start is None else int(best_start),
        "end_step": None if best_end is None else int(best_end),
        "threshold": float(threshold),
    }


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


def strict_fission_runs(
    top_trace: List[Optional[int]],
    margin_trace: List[Optional[float]],
    delta_mi_by_candidate: Dict[int, List[float]],
    delta_ccorr_by_candidate: Dict[int, List[float]],
    mi_threshold: float,
    cc_threshold: float,
    margin_threshold: float,
    window: int,
) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    if not top_trace:
        return episodes

    current_candidate = None
    current_start = None
    current_len = 0

    for idx, cand in enumerate(top_trace, start=1):
        if cand is None:
            if current_candidate is not None and current_len >= window:
                episodes.append(
                    {
                        "candidate": int(current_candidate),
                        "start_step": int(current_start),
                        "end_step": int(idx - 1),
                        "length": int(current_len),
                    }
                )
            current_candidate = None
            current_start = None
            current_len = 0
            continue

        margin = margin_trace[idx - 1]
        mi_val = delta_mi_by_candidate.get(int(cand), [0.0] * len(top_trace))[idx - 1]
        cc_val = delta_ccorr_by_candidate.get(int(cand), [0.0] * len(top_trace))[idx - 1]
        ok = (
            margin is not None
            and float(margin) > margin_threshold
            and float(mi_val) > mi_threshold
            and float(cc_val) > cc_threshold
        )

        if ok:
            if current_candidate == int(cand):
                current_len += 1
            else:
                if current_candidate is not None and current_len >= window:
                    episodes.append(
                        {
                            "candidate": int(current_candidate),
                            "start_step": int(current_start),
                            "end_step": int(idx - 1),
                            "length": int(current_len),
                        }
                    )
                current_candidate = int(cand)
                current_start = idx
                current_len = 1
        else:
            if current_candidate is not None and current_len >= window:
                episodes.append(
                    {
                        "candidate": int(current_candidate),
                        "start_step": int(current_start),
                        "end_step": int(idx - 1),
                        "length": int(current_len),
                    }
                )
            current_candidate = None
            current_start = None
            current_len = 0

    if current_candidate is not None and current_len >= window:
        episodes.append(
            {
                "candidate": int(current_candidate),
                "start_step": int(current_start),
                "end_step": int(len(top_trace)),
                "length": int(current_len),
            }
        )

    return episodes


def run_observer(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = build_config(args)
    xp, is_gpu = phys.get_array_module(args.device)
    prepared = init_prepared_state(cfg, xp, args.initial_state, args.perturb_eps)

    pair_mi_traces: Dict[int, List[float]] = {i: [] for i in range(cfg.n_max)}
    pair_ccorr_traces: Dict[int, List[float]] = {i: [] for i in range(cfg.n_max)}
    delta_pair_mi_traces: Dict[int, List[float]] = {i: [] for i in range(cfg.n_max)}
    delta_pair_ccorr_traces: Dict[int, List[float]] = {i: [] for i in range(cfg.n_max)}

    top_candidate_trace: List[Optional[int]] = []
    top_candidate_margin_trace: List[Optional[float]] = []
    top_candidate_delta_mi_trace: List[Optional[float]] = []
    top_candidate_delta_ccorr_trace: List[Optional[float]] = []

    snapshots: List[Dict[str, Any]] = []

    reference_active: Optional[int] = None
    baseline_mi: Optional[np.ndarray] = None
    baseline_ccorr: Optional[np.ndarray] = None

    for step in range(1, args.total_steps + 1):
        prepared = phys.evolve_prepared_state(prepared, cfg, xp)
        psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = prepared
        n_sites = int(psi.ndim)

        if reference_active is None:
            reference_active = find_active_reference(active_nodes)

        mi, ccorr, signal = compute_pair_matrices(psi, n_sites, cfg.n_max, xp, args)

        if baseline_mi is None:
            baseline_mi = np.array(mi, copy=True)
            baseline_ccorr = np.array(ccorr, copy=True)

        delta_mi = mi - baseline_mi
        delta_ccorr = ccorr - baseline_ccorr

        dormant_rows: List[Dict[str, Any]] = []
        for j in sorted(int(x) for x in dormant_nodes):
            pair_mi_traces[j].append(float(mi[reference_active, j]))
            pair_ccorr_traces[j].append(float(ccorr[reference_active, j]))
            delta_pair_mi_traces[j].append(float(delta_mi[reference_active, j]))
            delta_pair_ccorr_traces[j].append(float(delta_ccorr[reference_active, j]))

            dormant_rows.append(
                {
                    "candidate": int(j),
                    "raw_mi": float(mi[reference_active, j]),
                    "raw_ccorr": float(ccorr[reference_active, j]),
                    "delta_mi": float(delta_mi[reference_active, j]),
                    "delta_ccorr": float(delta_ccorr[reference_active, j]),
                    "score": float(
                        args.w_mi * max(0.0, float(delta_mi[reference_active, j]))
                        + args.w_ccorr * max(0.0, float(delta_ccorr[reference_active, j]))
                    ),
                }
            )

        dormant_rows.sort(key=lambda r: r["score"], reverse=True)
        top_row = dormant_rows[0] if dormant_rows else None
        second_row = dormant_rows[1] if len(dormant_rows) > 1 else None

        top_candidate_trace.append(None if top_row is None else int(top_row["candidate"]))
        top_candidate_delta_mi_trace.append(None if top_row is None else float(top_row["delta_mi"]))
        top_candidate_delta_ccorr_trace.append(None if top_row is None else float(top_row["delta_ccorr"]))
        if top_row is None or second_row is None:
            top_candidate_margin_trace.append(None)
        else:
            top_candidate_margin_trace.append(float(top_row["score"] - second_row["score"]))

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
            snapshots.append(
                {
                    "step": int(step),
                    "reference_active": int(reference_active),
                    "active_nodes": sorted(int(i) for i in active_nodes),
                    "dormant_nodes": sorted(int(i) for i in dormant_nodes),
                    "active_edges": [list(e) for e in sorted(active_edges)],
                    "top_fission_candidate": top_row,
                    "second_fission_candidate": second_row,
                    "top_candidate_margin": top_candidate_margin_trace[-1],
                    "top_candidates": dormant_rows[: max(1, int(args.top_candidates))],
                    "dominant_core": core,
                    "metric": metrics,
                    "graph_stats": gstats,
                }
            )

        if args.progress_every > 0 and step % args.progress_every == 0:
            if top_row is None:
                print(f"[step {step:04d}] active_ref={reference_active} top_cand=None")
            else:
                print(
                    f"[step {step:04d}] active_ref={reference_active} "
                    f"top_cand={top_row['candidate']} "
                    f"dMI={top_row['delta_mi']:.3e} "
                    f"dCC={top_row['delta_ccorr']:.3e} "
                    f"margin={(top_candidate_margin_trace[-1] if top_candidate_margin_trace[-1] is not None else None)}"
                )

    candidate_summaries: Dict[str, Any] = {}
    for j in range(cfg.n_max):
        if j == reference_active:
            continue
        if not delta_pair_mi_traces[j] and not delta_pair_ccorr_traces[j]:
            continue
        mi_run = longest_positive_run(delta_pair_mi_traces[j], args.candidate_threshold_mi)
        cc_run = longest_positive_run(delta_pair_ccorr_traces[j], args.candidate_threshold_ccorr)
        candidate_summaries[str(j)] = {
            "delta_mi_trace_summary": {
                "turning_points": find_turning_points(delta_pair_mi_traces[j]),
                "fft": dominant_fft_component(delta_pair_mi_traces[j], args.dt),
                "positive_run": mi_run,
                "activation_detected": bool(mi_run["length"] >= args.candidate_window),
            },
            "delta_ccorr_trace_summary": {
                "turning_points": find_turning_points(delta_pair_ccorr_traces[j]),
                "fft": dominant_fft_component(delta_pair_ccorr_traces[j], args.dt),
                "positive_run": cc_run,
                "activation_detected": bool(cc_run["length"] >= args.candidate_window),
            },
        }

    stable_counts: Dict[int, int] = {}
    for x in top_candidate_trace:
        if x is None:
            continue
        stable_counts[int(x)] = stable_counts.get(int(x), 0) + 1

    winner_candidate = None
    winner_count = 0
    if stable_counts:
        winner_candidate, winner_count = max(stable_counts.items(), key=lambda kv: kv[1])

    episodes = strict_fission_runs(
        top_trace=top_candidate_trace,
        margin_trace=top_candidate_margin_trace,
        delta_mi_by_candidate=delta_pair_mi_traces,
        delta_ccorr_by_candidate=delta_pair_ccorr_traces,
        mi_threshold=args.candidate_threshold_mi,
        cc_threshold=args.candidate_threshold_ccorr,
        margin_threshold=args.margin_threshold,
        window=args.consensus_window,
    )

    strongest_episode = None
    if episodes:
        strongest_episode = max(episodes, key=lambda e: e["length"])

    return {
        "script": "hsf_mesoscape_fission_observer.py",
        "observer_mode": True,
        "physics_config": asdict(cfg),
        "observer_config": {
            "initial_state": args.initial_state,
            "perturb_eps": float(args.perturb_eps),
            "w_mi": float(args.w_mi),
            "w_ccorr": float(args.w_ccorr),
            "ccorr_scale": float(args.ccorr_scale),
            "candidate_threshold_mi": float(args.candidate_threshold_mi),
            "candidate_threshold_ccorr": float(args.candidate_threshold_ccorr),
            "candidate_window": int(args.candidate_window),
            "margin-threshold": float(args.margin_threshold),
            "consensus_window": int(args.consensus_window),
            "top_candidates": int(args.top_candidates),
            "principle": "observe whether one active subsystem develops a persistently distinguished second subsystem candidate without applying birth moves; require dual-channel support, margin over runner-up, and persistence before calling the signal fission-like",
        },
        "reference_active": None if reference_active is None else int(reference_active),
        "top_candidate_trace": top_candidate_trace,
        "top_candidate_margin_trace": top_candidate_margin_trace,
        "top_candidate_delta_mi_trace": top_candidate_delta_mi_trace,
        "top_candidate_delta_ccorr_trace": top_candidate_delta_ccorr_trace,
        "pair_delta_mi_traces": {str(k): v for k, v in delta_pair_mi_traces.items() if v},
        "pair_delta_ccorr_traces": {str(k): v for k, v in delta_pair_ccorr_traces.items() if v},
        "candidate_summaries": candidate_summaries,
        "stable_candidate_counts": {str(k): int(v) for k, v in stable_counts.items()},
        "winner_candidate": None if winner_candidate is None else int(winner_candidate),
        "winner_count": int(winner_count),
        "strict_fission_episodes": episodes,
        "strongest_fission_episode": strongest_episode,
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