#!/usr/bin/env python3
# filename: hsf_mesoscape_observer_v3.py

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
            "HSF mesoscale observer v3. Evolves psi on a fixed lawful substrate setup and "
            "reports MI-first and connected-correlator diagnostics with baseline subtraction "
            "and sector decomposition (active-active, active-dormant, dormant-dormant)."
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
    p.add_argument("--json-out", type=str, default="hsf_mesoscape_observer_v3.json")
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


def sector_pair_lists(active_nodes: Set[int], dormant_nodes: Set[int]) -> Dict[str, List[Edge]]:
    active = sorted(int(i) for i in active_nodes)
    dormant = sorted(int(i) for i in dormant_nodes)
    aa: List[Edge] = []
    ad: List[Edge] = []
    dd: List[Edge] = []
    for idx, i in enumerate(active):
        for j in active[idx + 1:]:
            aa.append((i, j))
    for i in active:
        for j in dormant:
            ad.append((min(i, j), max(i, j)))
    for idx, i in enumerate(dormant):
        for j in dormant[idx + 1:]:
            dd.append((i, j))
    return {
        "active_active": aa,
        "active_dormant": ad,
        "dormant_dormant": dd,
    }


def mean_over_pairs(mat: np.ndarray, pairs: List[Edge]) -> Optional[float]:
    if not pairs:
        return None
    vals = [float(mat[i, j]) for i, j in pairs]
    return float(np.mean(vals))


def top_pairs_from_matrix(
    score_mat: np.ndarray,
    mi_mat: np.ndarray,
    ccorr_mat: np.ndarray,
    k: int,
    allowed_pairs: Optional[Set[Edge]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = score_mat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            e = (i, j)
            if allowed_pairs is not None and e not in allowed_pairs:
                continue
            rows.append(
                {
                    "pair": [int(i), int(j)],
                    "score": float(score_mat[i, j]),
                    "mutual_information": float(mi_mat[i, j]),
                    "connected_correlator": float(ccorr_mat[i, j]),
                }
            )
    rows.sort(key=lambda r: r["score"], reverse=True)
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


def phase_lock_score(trace_a: List[float], trace_b: List[float]) -> Optional[float]:
    if len(trace_a) != len(trace_b) or len(trace_a) < 3:
        return None
    a = np.asarray(trace_a, dtype=np.float64)
    b = np.asarray(trace_b, dtype=np.float64)
    da = np.sign(np.diff(a))
    db = np.sign(np.diff(b))
    return float(np.mean(da == db))


def safe_trace_summary(trace: List[Optional[float]], dt: float) -> Dict[str, Any]:
    vals = [x for x in trace if x is not None]
    if not vals:
        return {
            "turning_points": {"maxima": [], "minima": [], "reversal_steps": [], "half_period_estimates": []},
            "fft": {"frequency": None, "period_steps": None, "period_time": None, "power": None},
        }
    clean = [float(x) for x in vals]
    return {
        "turning_points": find_turning_points(clean),
        "fft": dominant_fft_component(clean, dt),
    }


def run_observer(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = build_config(args)
    xp, is_gpu = phys.get_array_module(args.device)
    prepared = init_prepared_state(cfg, xp, args.initial_state, args.perturb_eps)

    mean_mi_trace: List[float] = []
    mean_ccorr_trace: List[float] = []
    mean_signal_trace: List[float] = []

    sector_mean_traces: Dict[str, Dict[str, List[Optional[float]]]] = {
        "raw_mi": {"active_active": [], "active_dormant": [], "dormant_dormant": []},
        "raw_ccorr": {"active_active": [], "active_dormant": [], "dormant_dormant": []},
        "delta_mi": {"active_active": [], "active_dormant": [], "dormant_dormant": []},
        "delta_ccorr": {"active_active": [], "active_dormant": [], "dormant_dormant": []},
    }

    baseline_mi: Optional[np.ndarray] = None
    baseline_ccorr: Optional[np.ndarray] = None

    snapshots: List[Dict[str, Any]] = []

    for step in range(1, args.total_steps + 1):
        prepared = phys.evolve_prepared_state(prepared, cfg, xp)
        psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = prepared
        n_sites = int(psi.ndim)

        mi, ccorr, signal = compute_pair_matrices(psi, n_sites, cfg.n_max, xp, args)

        if baseline_mi is None:
            baseline_mi = np.array(mi, copy=True)
            baseline_ccorr = np.array(ccorr, copy=True)

        delta_mi = mi - baseline_mi
        delta_ccorr = ccorr - baseline_ccorr

        mean_mi_trace.append(mean_upper_triangle(mi))
        mean_ccorr_trace.append(mean_upper_triangle(ccorr))
        mean_signal_trace.append(mean_upper_triangle(signal))

        sectors = sector_pair_lists(active_nodes, dormant_nodes)
        for sector_name, pairs in sectors.items():
            sector_mean_traces["raw_mi"][sector_name].append(mean_over_pairs(mi, pairs))
            sector_mean_traces["raw_ccorr"][sector_name].append(mean_over_pairs(ccorr, pairs))
            sector_mean_traces["delta_mi"][sector_name].append(mean_over_pairs(delta_mi, pairs))
            sector_mean_traces["delta_ccorr"][sector_name].append(mean_over_pairs(delta_ccorr, pairs))

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
            top_global = top_pairs_from_matrix(signal, mi, ccorr, args.top_pairs)
            top_delta_global = top_pairs_from_matrix(np.abs(delta_ccorr), delta_mi, delta_ccorr, args.top_pairs)

            sector_tops: Dict[str, Any] = {}
            for sector_name, pairs in sectors.items():
                allowed = set(pairs)
                sector_tops[sector_name] = {
                    "top_raw_signal_pairs": top_pairs_from_matrix(signal, mi, ccorr, args.top_pairs, allowed_pairs=allowed),
                    "top_delta_ccorr_pairs": top_pairs_from_matrix(np.abs(delta_ccorr), delta_mi, delta_ccorr, args.top_pairs, allowed_pairs=allowed),
                }

            snapshots.append(
                {
                    "step": int(step),
                    "active_nodes": sorted(int(i) for i in active_nodes),
                    "dormant_nodes": sorted(int(i) for i in dormant_nodes),
                    "active_edges": [list(e) for e in sorted(active_edges)],
                    "mean_mutual_information": mean_mi_trace[-1],
                    "mean_connected_correlator": mean_ccorr_trace[-1],
                    "mean_signal": mean_signal_trace[-1],
                    "sector_means": {
                        "raw_mi": {k: sector_mean_traces["raw_mi"][k][-1] for k in sector_mean_traces["raw_mi"]},
                        "raw_ccorr": {k: sector_mean_traces["raw_ccorr"][k][-1] for k in sector_mean_traces["raw_ccorr"]},
                        "delta_mi": {k: sector_mean_traces["delta_mi"][k][-1] for k in sector_mean_traces["delta_mi"]},
                        "delta_ccorr": {k: sector_mean_traces["delta_ccorr"][k][-1] for k in sector_mean_traces["delta_ccorr"]},
                    },
                    "dominant_core": core,
                    "top_global_pairs": top_global,
                    "top_delta_ccorr_global_pairs": top_delta_global,
                    "top_pairs_by_sector": sector_tops,
                    "metric": metrics,
                    "graph_stats": gstats,
                }
            )

        if args.progress_every > 0 and step % args.progress_every == 0:
            cp = None
            if core is not None and core.get("core_pair") is not None:
                cp = core.get("core_pair")
            aa_dmi = sector_mean_traces["delta_mi"]["active_active"][-1]
            ad_dmi = sector_mean_traces["delta_mi"]["active_dormant"][-1]
            dd_dmi = sector_mean_traces["delta_mi"]["dormant_dormant"][-1]
            aa_dcc = sector_mean_traces["delta_ccorr"]["active_active"][-1]
            ad_dcc = sector_mean_traces["delta_ccorr"]["active_dormant"][-1]
            dd_dcc = sector_mean_traces["delta_ccorr"]["dormant_dormant"][-1]
            print(
                f"[step {step:04d}] mean_mi={mean_mi_trace[-1]:.6e} "
                f"mean_ccorr={mean_ccorr_trace[-1]:.6e} "
                f"dMI(AA/AD/DD)=({aa_dmi:.3e},{ad_dmi:.3e},{dd_dmi:.3e}) "
                f"dCC(AA/AD/DD)=({aa_dcc:.3e},{ad_dcc:.3e},{dd_dcc:.3e}) "
                f"core={cp}"
            )

    max_abs_mi = float(np.max(np.abs(np.asarray(mean_mi_trace, dtype=np.float64)))) if mean_mi_trace else 0.0
    mi_warning = None
    if max_abs_mi < float(args.mi_warning_threshold):
        mi_warning = (
            "Mutual information stayed below the configured warning threshold for the full run; "
            "any composite pair signal is therefore dominated by the connected-correlator channel, not MI."
        )

    summary = {
        "global": {
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
        },
        "sector_delta_mi": {
            sec: safe_trace_summary(sector_mean_traces["delta_mi"][sec], args.dt)
            for sec in ("active_active", "active_dormant", "dormant_dormant")
        },
        "sector_delta_ccorr": {
            sec: safe_trace_summary(sector_mean_traces["delta_ccorr"][sec], args.dt)
            for sec in ("active_active", "active_dormant", "dormant_dormant")
        },
    }

    return {
        "script": "hsf_mesoscape_observer_v3.py",
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
            "principle": "observe the substrate without choosing births or pruning; use connected correlators, baseline subtraction, and sector decomposition to separate initial-condition residue from dynamics",
        },
        "mean_mutual_information_trace": mean_mi_trace,
        "mean_connected_correlator_trace": mean_ccorr_trace,
        "mean_signal_trace": mean_signal_trace,
        "sector_mean_traces": sector_mean_traces,
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