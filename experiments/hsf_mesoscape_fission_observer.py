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
            "HSF observer for persistent secondary loci attached to a single active subsystem. "
            "Characterizes parent anchoring versus local-neighborhood emergence."
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

    p.add_argument("--w-mi", type=float, default=1.0)
    p.add_argument("--w-ccorr", type=float, default=0.25)
    p.add_argument("--ccorr-scale", type=float, default=4.0)

    p.add_argument("--top-candidates", type=int, default=5)
    p.add_argument("--margin-threshold", type=float, default=1e-8)
    p.add_argument("--attachment-threshold", type=float, default=1e-7)
    p.add_argument("--isolation-threshold", type=float, default=1e-8)
    p.add_argument("--persistence-window", type=int, default=10)

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
    link_memory = {
        canonical_edge(i, j): phys.default_linkreg().copy()
        for i in range(cfg.n_max) for j in range(i + 1, cfg.n_max)
    }
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
        strength = (
            float(cfg.pair_scale)
            * float(np.clip(w, 0.0, 1.0))
            * float(np.clip(sigma[i], 0.0, 1.0))
            * float(np.clip(sigma[j], 0.0, 1.0))
        )
        edge_strengths[e] = strength
        link_regs[e] = np.array(link_memory.get(e, phys.default_linkreg()), copy=True)

    phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
    return psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs


def evolve_graded_state(graded_state, cfg: PhysicsConfig, xp, steps: int = 1):
    state = graded_state
    for _ in range(max(1, int(steps))):
        prepared = materialize_state(state[0], state[1], state[2], state[3], cfg)
        prepared = phys.evolve_prepared_state(prepared, cfg, xp)
        psi, _active_nodes, _dormant_nodes, _active_edges, _local_coeffs, _edge_strengths, link_regs = prepared
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
    mi_contrib = float(args.w_mi * np.tanh(max(0.0, mi)))
    cc_contrib = float(args.w_ccorr * np.tanh(args.ccorr_scale * max(0.0, cc)))
    signal = float(mi_contrib + cc_contrib)
    return {
        "mi": mi,
        "ccorr": cc,
        "signal": signal,
        "mi_contrib": mi_contrib,
        "ccorr_contrib": cc_contrib,
    }


def characterize_locus(
    psi,
    reference_active: int,
    cand: int,
    n_sites: int,
    xp,
    args: argparse.Namespace,
    n_max: int,
) -> Dict[str, Any]:
    parent_pair = evaluate_pair(psi, reference_active, cand, n_sites, xp, args)

    other_rows = []
    for other in range(n_max):
        if other in (reference_active, cand):
            continue
        rel = evaluate_pair(psi, cand, other, n_sites, xp, args)
        rel["other"] = int(other)
        other_rows.append(rel)

    mean_other_mi = float(np.mean([r["mi"] for r in other_rows])) if other_rows else 0.0
    mean_other_cc = float(np.mean([r["ccorr"] for r in other_rows])) if other_rows else 0.0
    mean_other_signal = float(np.mean([r["signal"] for r in other_rows])) if other_rows else 0.0
    max_other_signal = float(np.max([r["signal"] for r in other_rows])) if other_rows else 0.0

    attachment_mi = float(parent_pair["mi"] - mean_other_mi)
    attachment_ccorr = float(parent_pair["ccorr"] - mean_other_cc)
    attachment_signal = float(parent_pair["signal"] - max_other_signal)

    denom = abs(parent_pair["mi_contrib"]) + abs(parent_pair["ccorr_contrib"])
    mi_frac = float(abs(parent_pair["mi_contrib"]) / denom) if denom > 0.0 else 0.0
    cc_frac = 1.0 - mi_frac if denom > 0.0 else 0.0

    isolation_ratio = float(parent_pair["signal"] / max(1e-12, mean_other_signal)) if mean_other_signal > 0.0 else float("inf")

    strongest_other = None
    if other_rows:
        strongest_other_row = max(other_rows, key=lambda r: r["signal"])
        strongest_other = {
            "other": int(strongest_other_row["other"]),
            "mi": float(strongest_other_row["mi"]),
            "ccorr": float(strongest_other_row["ccorr"]),
            "signal": float(strongest_other_row["signal"]),
            "mi_contrib": float(strongest_other_row["mi_contrib"]),
            "ccorr_contrib": float(strongest_other_row["ccorr_contrib"]),
        }

    parent_anchor_advantage = float(parent_pair["signal"] - (0.0 if strongest_other is None else strongest_other["signal"]))
    neighborhood_pressure = float(0.0 if strongest_other is None else strongest_other["signal"] / max(1e-12, parent_pair["signal"]))

    return {
        "parent_candidate_mi": float(parent_pair["mi"]),
        "parent_candidate_ccorr": float(parent_pair["ccorr"]),
        "parent_candidate_signal": float(parent_pair["signal"]),
        "parent_candidate_mi_contrib": float(parent_pair["mi_contrib"]),
        "parent_candidate_ccorr_contrib": float(parent_pair["ccorr_contrib"]),
        "mi_fraction": float(mi_frac),
        "ccorr_fraction": float(cc_frac),
        "mean_other_mi": mean_other_mi,
        "mean_other_ccorr": mean_other_cc,
        "mean_other_signal": mean_other_signal,
        "max_other_signal": max_other_signal,
        "attachment_mi": attachment_mi,
        "attachment_ccorr": attachment_ccorr,
        "attachment_signal": attachment_signal,
        "attachment_dominance": float(parent_pair["signal"] - mean_other_signal),
        "isolation_ratio": isolation_ratio,
        "strongest_other_relation": strongest_other,
        "parent_anchor_advantage": parent_anchor_advantage,
        "neighborhood_pressure": neighborhood_pressure,
    }


def candidate_probe_rows(baseline_state, cfg: PhysicsConfig, xp, args: argparse.Namespace, reference_active: int) -> List[Dict[str, Any]]:
    psi_b, sigma_b, interface_b, link_mem_b = baseline_state
    n_sites = int(psi_b.ndim)

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
        psi_e, _sigma_e, _interface_e, _link_e = evolved

        content = characterize_locus(
            psi=psi_e,
            reference_active=reference_active,
            cand=cand,
            n_sites=n_sites,
            xp=xp,
            args=args,
            n_max=cfg.n_max,
        )

        row = {
            "candidate": int(cand),
            **content,
            "score": float(
                max(0.0, content["attachment_signal"])
                + 0.5 * max(0.0, content["attachment_mi"])
                + 0.5 * max(0.0, content["attachment_ccorr"])
            ),
        }
        rows.append(row)

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
    return {"frequency": freq, "period_steps": period_steps, "period_time": period_time, "power": float(power[idx])}


def migration_episodes(top_trace: List[Optional[int]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not top_trace:
        return out
    cur = top_trace[0]
    start = 1
    for idx in range(2, len(top_trace) + 1):
        nxt = top_trace[idx - 1]
        if nxt != cur:
            out.append({"candidate": None if cur is None else int(cur), "start_step": int(start), "end_step": int(idx - 1), "length": int(idx - start)})
            cur = nxt
            start = idx
    out.append({"candidate": None if cur is None else int(cur), "start_step": int(start), "end_step": int(len(top_trace)), "length": int(len(top_trace) + 1 - start)})
    return out


def persistent_secondary_locus_runs(
    top_trace: List[Optional[int]],
    margin_trace: List[Optional[float]],
    attach_mi_by_candidate: Dict[int, List[float]],
    attach_ccorr_by_candidate: Dict[int, List[float]],
    attach_signal_by_candidate: Dict[int, List[float]],
    attach_threshold: float,
    isolation_threshold: float,
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
                episodes.append({"candidate": int(current_candidate), "start_step": int(current_start), "end_step": int(idx - 1), "length": int(current_len)})
            current_candidate = None
            current_start = None
            current_len = 0
            continue

        margin = margin_trace[idx - 1]
        ami = attach_mi_by_candidate.get(int(cand), [0.0] * len(top_trace))[idx - 1]
        acc = attach_ccorr_by_candidate.get(int(cand), [0.0] * len(top_trace))[idx - 1]
        asg = attach_signal_by_candidate.get(int(cand), [0.0] * len(top_trace))[idx - 1]

        ok = (
            margin is not None
            and float(margin) > margin_threshold
            and float(ami) > attach_threshold
            and float(acc) > attach_threshold
            and float(asg) > isolation_threshold
        )

        if ok:
            if current_candidate == int(cand):
                current_len += 1
            else:
                if current_candidate is not None and current_len >= window:
                    episodes.append({"candidate": int(current_candidate), "start_step": int(current_start), "end_step": int(idx - 1), "length": int(current_len)})
                current_candidate = int(cand)
                current_start = idx
                current_len = 1
        else:
            if current_candidate is not None and current_len >= window:
                episodes.append({"candidate": int(current_candidate), "start_step": int(current_start), "end_step": int(idx - 1), "length": int(current_len)})
            current_candidate = None
            current_start = None
            current_len = 0

    if current_candidate is not None and current_len >= window:
        episodes.append({"candidate": int(current_candidate), "start_step": int(current_start), "end_step": int(len(top_trace)), "length": int(current_len)})
    return episodes


def episode_content_summaries(
    episodes: List[Dict[str, Any]],
    content_by_candidate: Dict[int, Dict[str, List[float]]],
    strongest_other_by_candidate: Dict[int, List[Optional[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    summary_keys = [
        "attachment_mi",
        "attachment_ccorr",
        "attachment_signal",
        "parent_candidate_mi",
        "parent_candidate_ccorr",
        "parent_candidate_signal",
        "parent_candidate_mi_contrib",
        "parent_candidate_ccorr_contrib",
        "mi_fraction",
        "ccorr_fraction",
        "mean_other_mi",
        "mean_other_ccorr",
        "mean_other_signal",
        "max_other_signal",
        "attachment_dominance",
        "isolation_ratio",
        "parent_anchor_advantage",
        "neighborhood_pressure",
    ]

    for ep in episodes:
        c = int(ep["candidate"])
        s0 = int(ep["start_step"]) - 1
        s1 = int(ep["end_step"])
        entry = {
            "candidate": c,
            "start_step": int(ep["start_step"]),
            "end_step": int(ep["end_step"]),
            "length": int(ep["length"]),
        }
        for key in summary_keys:
            seg = content_by_candidate[c][key][s0:s1]
            entry[f"mean_{key}"] = None if not seg else float(np.mean(seg))

        strongest_others = [x for x in strongest_other_by_candidate[c][s0:s1] if x is not None]
        if strongest_others:
            ids = [int(x["other"]) for x in strongest_others]
            vals, counts = np.unique(np.asarray(ids, dtype=int), return_counts=True)
            winner_idx = int(np.argmax(counts))
            modal_other = int(vals[winner_idx])
            modal_share = float(counts[winner_idx] / len(ids))
        else:
            modal_other = None
            modal_share = None

        entry["modal_strongest_other"] = modal_other
        entry["modal_strongest_other_share"] = modal_share
        out.append(entry)
    return out


def run_observer(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = build_config(args)
    xp, is_gpu = phys.get_array_module(args.device)
    state = init_graded_state(cfg, xp, args.initial_state, args.perturb_eps)

    reference_active = 0
    top_candidate_trace: List[Optional[int]] = []
    top_candidate_margin_trace: List[Optional[float]] = []

    content_keys = [
        "attachment_mi",
        "attachment_ccorr",
        "attachment_signal",
        "parent_candidate_mi",
        "parent_candidate_ccorr",
        "parent_candidate_signal",
        "parent_candidate_mi_contrib",
        "parent_candidate_ccorr_contrib",
        "mi_fraction",
        "ccorr_fraction",
        "mean_other_mi",
        "mean_other_ccorr",
        "mean_other_signal",
        "max_other_signal",
        "attachment_dominance",
        "isolation_ratio",
        "parent_anchor_advantage",
        "neighborhood_pressure",
    ]
    content_by_candidate: Dict[int, Dict[str, List[float]]] = {
        i: {k: [] for k in content_keys} for i in range(cfg.n_max) if i != reference_active
    }
    strongest_other_by_candidate: Dict[int, List[Optional[Dict[str, Any]]]] = {
        i: [] for i in range(cfg.n_max) if i != reference_active
    }

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
            for key in content_keys:
                content_by_candidate[c][key].append(float(row[key]))
            strongest_other_by_candidate[c].append(row["strongest_other_relation"])

        top_row = rows[0] if rows else None
        second_row = rows[1] if len(rows) > 1 else None

        top_candidate_trace.append(None if top_row is None else int(top_row["candidate"]))
        if top_row is None or second_row is None:
            top_candidate_margin_trace.append(None)
        else:
            top_candidate_margin_trace.append(float(top_row["score"] - second_row["score"]))

        core = bk.dominant_core_snapshot(
            psi, active_nodes, active_edges, edge_strengths, GM_MATRICES, xp, n_sites, None, link_regs
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
                    "top_secondary_locus_candidate": top_row,
                    "second_secondary_locus_candidate": second_row,
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
                    f"MIfrac={top_row['mi_fraction']:.3f} "
                    f"anchorAdv={top_row['parent_anchor_advantage']:.3e} "
                    f"nbrPress={top_row['neighborhood_pressure']:.3e} "
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
    for c in sorted(content_by_candidate.keys()):
        candidate_summaries[str(c)] = {
            "attachment_signal_positive_run": longest_positive_run(content_by_candidate[c]["attachment_signal"], args.isolation_threshold),
            "attachment_mi_positive_run": longest_positive_run(content_by_candidate[c]["attachment_mi"], args.attachment_threshold),
            "attachment_ccorr_positive_run": longest_positive_run(content_by_candidate[c]["attachment_ccorr"], args.attachment_threshold),
            "parent_anchor_advantage_positive_run": longest_positive_run(content_by_candidate[c]["parent_anchor_advantage"], 0.0),
            "mi_fraction_fft": dominant_fft_component(content_by_candidate[c]["mi_fraction"], args.dt),
            "isolation_ratio_fft": dominant_fft_component(content_by_candidate[c]["isolation_ratio"], args.dt),
            "neighborhood_pressure_fft": dominant_fft_component(content_by_candidate[c]["neighborhood_pressure"], args.dt),
        }

    episodes = persistent_secondary_locus_runs(
        top_trace=top_candidate_trace,
        margin_trace=top_candidate_margin_trace,
        attach_mi_by_candidate={c: content_by_candidate[c]["attachment_mi"] for c in content_by_candidate},
        attach_ccorr_by_candidate={c: content_by_candidate[c]["attachment_ccorr"] for c in content_by_candidate},
        attach_signal_by_candidate={c: content_by_candidate[c]["attachment_signal"] for c in content_by_candidate},
        attach_threshold=args.attachment_threshold,
        isolation_threshold=args.isolation_threshold,
        margin_threshold=args.margin_threshold,
        window=args.persistence_window,
    )

    strongest_episode = None
    if episodes:
        strongest_episode = max(episodes, key=lambda e: e["length"])

    migrations = migration_episodes(top_candidate_trace)
    episode_contents = episode_content_summaries(episodes, content_by_candidate, strongest_other_by_candidate)

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
            "w_mi": float(args.w_mi),
            "w_ccorr": float(args.w_ccorr),
            "ccorr_scale": float(args.ccorr_scale),
            "top_candidates": int(args.top_candidates),
            "margin_threshold": float(args.margin_threshold),
            "attachment_threshold": float(args.attachment_threshold),
            "isolation_threshold": float(args.isolation_threshold),
            "persistence_window": int(args.persistence_window),
            "principle": "characterize parent anchoring versus local-neighborhood emergence in persistent secondary loci",
        },
        "reference_active": int(reference_active),
        "top_candidate_trace": top_candidate_trace,
        "top_candidate_margin_trace": top_candidate_margin_trace,
        "content_traces_by_candidate": {
            str(c): {k: v for k, v in content_by_candidate[c].items()} for c in content_by_candidate
        },
        "strongest_other_traces_by_candidate": {
            str(c): strongest_other_by_candidate[c] for c in strongest_other_by_candidate
        },
        "candidate_summaries": candidate_summaries,
        "stable_candidate_counts": {str(k): int(v) for k, v in stable_counts.items()},
        "winner_candidate": None if winner_candidate is None else int(winner_candidate),
        "winner_count": int(winner_count),
        "persistent_secondary_locus_episodes": episodes,
        "strongest_secondary_locus_episode": strongest_episode,
        "episode_content_summaries": episode_contents,
        "migration_episodes": migrations,
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