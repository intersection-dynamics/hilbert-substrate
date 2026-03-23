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
from hsf_mesoscale_physics_core import PhysicsConfig, GM_MATRICES, apply_one_body, canonical_edge

Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HSF one-to-two fission susceptibility observer. Starts with one active subsystem and "
            "measures which dormant candidate is most favored by a minimal split-probe branch. "
            "Unlike the earlier observer, pair_scale matters here because each candidate is tested "
            "through a weak active-dormant probe interface before comparing the probed branch to baseline."
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

    p.add_argument("--probe-sigma", type=float, default=0.12)
    p.add_argument("--probe-edge", type=float, default=0.12)
    p.add_argument("--probe-steps", type=int, default=1)
    p.add_argument("--probe-weight-mi", type=float, default=1.0)
    p.add_argument("--probe-weight-ccorr", type=float, default=0.25)
    p.add_argument("--probe-weight-expr", type=float, default=0.25)
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


def init_graded_state(cfg: PhysicsConfig, xp, initial_state: str, perturb_eps: float):
    psi, _active_nodes, _dormant_nodes, _active_edges, _local_coeffs, _edge_strengths, _link_regs, _rng = phys.init_state(cfg, xp)
    psi = make_initial_state(initial_state, cfg.n_max, xp, cfg.seed, perturb_eps)
    sigma = np.zeros(cfg.n_max, dtype=np.float64)
    sigma[: cfg.n_init] = 1.0
    interface_commitment: Dict[Edge, float] = {}
    for i in range(cfg.n_max):
        for j in range(i + 1, cfg.n_max):
            interface_commitment[canonical_edge(i, j)] = 0.0
    link_memory = {canonical_edge(i, j): phys.default_linkreg().copy() for i in range(cfg.n_max) for j in range(i + 1, cfg.n_max)}
    return psi, sigma, interface_commitment, link_memory


def clone_graded_state(psi, sigma: np.ndarray, interface_commitment: Dict[Edge, float], link_memory: Dict[Edge, np.ndarray]):
    return (
        psi.copy(),
        np.array(sigma, copy=True),
        dict(interface_commitment),
        {canonical_edge(*e): np.array(reg, copy=True) for e, reg in link_memory.items()},
    )


def materialize_state(
    psi,
    sigma: np.ndarray,
    interface_commitment: Dict[Edge, float],
    link_memory: Dict[Edge, np.ndarray],
    cfg: PhysicsConfig,
    sigma_on_threshold: float = 1e-12,
    edge_on_threshold: float = 1e-12,
):
    active_nodes: Set[int] = {i for i in range(len(sigma)) if float(sigma[i]) > float(sigma_on_threshold)}
    dormant_nodes: Set[int] = set(range(len(sigma))) - active_nodes

    local_coeffs = np.zeros(len(sigma), dtype=np.float64)
    for i in active_nodes:
        local_coeffs[i] = float(cfg.local_scale) * float(np.clip(sigma[i], 0.0, 1.0))

    active_edges: Set[Edge] = set()
    edge_strengths: Dict[Edge, float] = {}
    link_regs: Dict[Edge, np.ndarray] = {}

    for e, w in interface_commitment.items():
        i, j = e
        if i not in active_nodes or j not in active_nodes:
            continue
        if float(w) <= float(edge_on_threshold):
            continue
        active_edges.add(e)
        strength = float(cfg.pair_scale) * float(np.clip(w, 0.0, 1.0)) * float(np.clip(sigma[i], 0.0, 1.0)) * float(np.clip(sigma[j], 0.0, 1.0))
        edge_strengths[e] = strength
        link_regs[e] = np.array(link_memory.get(e, phys.default_linkreg()), copy=True)

    phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
    return psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs


def evolve_graded_state(graded_state, cfg: PhysicsConfig, xp, steps: int = 1):
    state = graded_state
    for _ in range(max(1, int(steps))):
        prepared = materialize_state(state[0], state[1], state[2], state[3], cfg)
        prepared = phys.evolve_prepared_state(prepared, cfg, xp)
        psi, active_nodes, dormant_nodes, active_edges, _local_coeffs, _edge_strengths, link_regs = prepared
        link_memory = dict(state[3])
        for e, reg in link_regs.items():
            link_memory[canonical_edge(*e)] = np.array(reg, copy=True)
        state = (psi, np.array(state[1], copy=True), dict(state[2]), link_memory)
    return state


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


def evaluate_pair(psi, i: int, j: int, n_sites: int, xp, args: argparse.Namespace) -> Dict[str, float]:
    mi = float(bk.mutual_information_from_state(psi, i, j, n_sites, xp))
    cc = float(connected_pair_correlator_strength(psi, i, j, GM_MATRICES, xp))
    signal = float(args.w_mi * np.tanh(max(0.0, mi)) + args.w_ccorr * np.tanh(args.ccorr_scale * max(0.0, cc)))
    return {"mi": mi, "ccorr": cc, "signal": signal}


def expression_value(prepared_state, xp) -> float:
    psi, active_nodes, _dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = prepared_state
    n_sites = int(local_coeffs.shape[0])

    class SimpleScoreCfg:
        # expression weights
        w_mi = 1.0
        w_corr = 0.5
        w_link = 0.5

        # MI path
        exact_mi_cutoff = 64

        # local_expression() may touch these even in observer mode
        birth_parent_relief_weight = 0.0
        birth_novelty_weight = 0.0
        birth_distinctness_weight = 0.0
        birth_redundancy_penalty = 0.0

        retirement_threshold = 1.0
        retirement_bookkeeping_weight = 0.0
        retirement_function_weight = 0.0
        retirement_core_penalty = 0.0
        retirement_shell_penalty = 0.0
        retirement_edge_weight = 0.0
        retirement_sub_weight = 0.0

        weaken_shell_penalty = 0.0
        weaken_protected_core_penalty = 0.0

        shell_reexpression_pk_min = 0.0
        shell_reexpression_pm_min = 0.0
        shell_reexpression_ps_min = 0.0

        # harmless placeholders
        lambda_B = 0.0
        lambda_S = 0.0
        lambda_F = 0.0
        lambda_R = 0.0
        organizer_large_region_cutoff = 64

    return float(
        bk.local_expression(
            psi,
            active_nodes,
            active_edges,
            edge_strengths,
            link_regs,
            SimpleScoreCfg(),
            GM_MATRICES,
            xp,
            n_sites,
        )
    )
def candidate_probe_rows(
    baseline_state,
    cfg: PhysicsConfig,
    xp,
    args: argparse.Namespace,
    reference_active: int,
) -> List[Dict[str, Any]]:
    psi_b, sigma_b, interface_b, link_mem_b = baseline_state
    n_sites = int(psi_b.ndim)

    baseline_prepared = materialize_state(psi_b, sigma_b, interface_b, link_mem_b, cfg)
    baseline_expr = expression_value(baseline_prepared, xp)

    rows: List[Dict[str, Any]] = []
    for cand in range(cfg.n_max):
        if cand == reference_active:
            continue

        probed = clone_graded_state(psi_b, sigma_b, interface_b, link_mem_b)
        psi_p, sigma_p, interface_p, link_p = probed
        sigma_p[cand] = max(float(sigma_p[cand]), float(args.probe_sigma))
        interface_p[canonical_edge(reference_active, cand)] = max(
            float(interface_p.get(canonical_edge(reference_active, cand), 0.0)),
            float(args.probe_edge),
        )

        evolved = evolve_graded_state((psi_p, sigma_p, interface_p, link_p), cfg, xp, steps=args.probe_steps)
        psi_e, sigma_e, interface_e, link_e = evolved
        prepared_e = materialize_state(psi_e, sigma_e, interface_e, link_e, cfg)
        expr_e = expression_value(prepared_e, xp)

        base_pair = evaluate_pair(psi_b, reference_active, cand, n_sites, xp, args)
        probed_pair = evaluate_pair(psi_e, reference_active, cand, n_sites, xp, args)

        dmi = float(probed_pair["mi"] - base_pair["mi"])
        dcc = float(probed_pair["ccorr"] - base_pair["ccorr"])
        dexpr = float(expr_e - baseline_expr)
        score = float(
            args.probe_weight_mi * max(0.0, dmi)
            + args.probe_weight_ccorr * max(0.0, dcc)
            + args.probe_weight_expr * max(0.0, dexpr)
        )

        rows.append(
            {
                "candidate": int(cand),
                "baseline_mi": float(base_pair["mi"]),
                "baseline_ccorr": float(base_pair["ccorr"]),
                "probed_mi": float(probed_pair["mi"]),
                "probed_ccorr": float(probed_pair["ccorr"]),
                "delta_mi": dmi,
                "delta_ccorr": dcc,
                "delta_expression": dexpr,
                "score": score,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


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
    delta_expr_by_candidate: Dict[int, List[float]],
    mi_threshold: float,
    cc_threshold: float,
    expr_threshold: float,
    margin_threshold: float,
    window: int,
) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
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
        ex_val = delta_expr_by_candidate.get(int(cand), [0.0] * len(top_trace))[idx - 1]
        ok = (
            margin is not None
            and float(margin) > margin_threshold
            and float(mi_val) > mi_threshold
            and float(cc_val) > cc_threshold
            and float(ex_val) > expr_threshold
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
    state = init_graded_state(cfg, xp, args.initial_state, args.perturb_eps)

    reference_active = 0
    top_candidate_trace: List[Optional[int]] = []
    top_candidate_margin_trace: List[Optional[float]] = []
    top_candidate_delta_mi_trace: List[Optional[float]] = []
    top_candidate_delta_ccorr_trace: List[Optional[float]] = []
    top_candidate_delta_expression_trace: List[Optional[float]] = []

    delta_mi_by_candidate: Dict[int, List[float]] = {i: [] for i in range(cfg.n_max) if i != reference_active}
    delta_ccorr_by_candidate: Dict[int, List[float]] = {i: [] for i in range(cfg.n_max) if i != reference_active}
    delta_expr_by_candidate: Dict[int, List[float]] = {i: [] for i in range(cfg.n_max) if i != reference_active}

    snapshots: List[Dict[str, Any]] = []

    for step in range(1, args.total_steps + 1):
        state = evolve_graded_state(state, cfg, xp, steps=1)
        psi, sigma, interface_commitment, link_memory = state
        prepared = materialize_state(psi, sigma, interface_commitment, link_memory, cfg)
        _psi_p, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = prepared
        n_sites = int(psi.ndim)

        rows = candidate_probe_rows(state, cfg, xp, args, reference_active)

        for row in rows:
            c = int(row["candidate"])
            delta_mi_by_candidate[c].append(float(row["delta_mi"]))
            delta_ccorr_by_candidate[c].append(float(row["delta_ccorr"]))
            delta_expr_by_candidate[c].append(float(row["delta_expression"]))

        top_row = rows[0] if rows else None
        second_row = rows[1] if len(rows) > 1 else None

        top_candidate_trace.append(None if top_row is None else int(top_row["candidate"]))
        top_candidate_delta_mi_trace.append(None if top_row is None else float(top_row["delta_mi"]))
        top_candidate_delta_ccorr_trace.append(None if top_row is None else float(top_row["delta_ccorr"]))
        top_candidate_delta_expression_trace.append(None if top_row is None else float(top_row["delta_expression"]))
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
                    "top_candidates": rows[: max(1, int(args.top_candidates))],
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
                    f"dExpr={top_row['delta_expression']:.3e} "
                    f"margin={(top_candidate_margin_trace[-1] if top_candidate_margin_trace[-1] is not None else None)}"
                )

    stable_counts: Dict[int, int] = {}
    for x in top_candidate_trace:
        if x is None:
            continue
        stable_counts[int(x)] = stable_counts.get(int(x), 0) + 1

    winner_candidate = None
    winner_count = 0
    if stable_counts:
        winner_candidate, winner_count = max(stable_counts.items(), key=lambda kv: kv[1])

    candidate_summaries: Dict[str, Any] = {}
    for c in sorted(delta_mi_by_candidate.keys()):
        candidate_summaries[str(c)] = {
            "delta_mi_positive_run": longest_positive_run(delta_mi_by_candidate[c], args.candidate_threshold_mi),
            "delta_ccorr_positive_run": longest_positive_run(delta_ccorr_by_candidate[c], args.candidate_threshold_ccorr),
            "delta_expression_positive_run": longest_positive_run(delta_expr_by_candidate[c], 0.0),
            "delta_mi_fft": dominant_fft_component(delta_mi_by_candidate[c], args.dt),
            "delta_ccorr_fft": dominant_fft_component(delta_ccorr_by_candidate[c], args.dt),
            "delta_expression_fft": dominant_fft_component(delta_expr_by_candidate[c], args.dt),
        }

    episodes = strict_fission_runs(
        top_trace=top_candidate_trace,
        margin_trace=top_candidate_margin_trace,
        delta_mi_by_candidate=delta_mi_by_candidate,
        delta_ccorr_by_candidate=delta_ccorr_by_candidate,
        delta_expr_by_candidate=delta_expr_by_candidate,
        mi_threshold=args.candidate_threshold_mi,
        cc_threshold=args.candidate_threshold_ccorr,
        expr_threshold=0.0,
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
            "probe_sigma": float(args.probe_sigma),
            "probe_edge": float(args.probe_edge),
            "probe_steps": int(args.probe_steps),
            "probe_weight_mi": float(args.probe_weight_mi),
            "probe_weight_ccorr": float(args.probe_weight_ccorr),
            "probe_weight_expr": float(args.probe_weight_expr),
            "w_mi": float(args.w_mi),
            "w_ccorr": float(args.w_ccorr),
            "ccorr_scale": float(args.ccorr_scale),
            "candidate_threshold_mi": float(args.candidate_threshold_mi),
            "candidate_threshold_ccorr": float(args.candidate_threshold_ccorr),
            "candidate_window": int(args.candidate_window),
            "margin_threshold": float(args.margin_threshold),
            "consensus_window": int(args.consensus_window),
            "top_candidates": int(args.top_candidates),
            "principle": "observe whether one active subsystem develops a persistently distinguished second subsystem candidate through a minimal split-probe branch; require dual-channel support, expression gain, margin over runner-up, and persistence before calling the signal fission-like",
        },
        "reference_active": int(reference_active),
        "top_candidate_trace": top_candidate_trace,
        "top_candidate_margin_trace": top_candidate_margin_trace,
        "top_candidate_delta_mi_trace": top_candidate_delta_mi_trace,
        "top_candidate_delta_ccorr_trace": top_candidate_delta_ccorr_trace,
        "top_candidate_delta_expression_trace": top_candidate_delta_expression_trace,
        "pair_delta_mi_traces": {str(k): v for k, v in delta_mi_by_candidate.items()},
        "pair_delta_ccorr_traces": {str(k): v for k, v in delta_ccorr_by_candidate.items()},
        "pair_delta_expression_traces": {str(k): v for k, v in delta_expr_by_candidate.items()},
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