#!/usr/bin/env python3
# filename: hsf_mesoscape_graded_support_sandbox.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import hsf_mesoscale_bookkeeping as bk
import hsf_mesoscale_physics_core as phys
from hsf_mesoscale_physics_core import PhysicsConfig, GM_MATRICES, canonical_edge

Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HSF mesoscale graded-support sandbox (scalar-only selector, no hard vetoes). "
            "No heuristic selector shaping, no EMA link memory, no admissibility gate. "
            "All candidate moves compete only by "
            "DeltaF_phys = dE_raw - lambda_B dCB - lambda_S dCS - lambda_F dCF - lambda_R W_NR."
        )
    )

    # Physics
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--n-init", type=int, default=2)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--local-scale", type=float, default=0.15)
    p.add_argument("--pair-scale", type=float, default=0.12)
    p.add_argument("--spawn-pair-scale", type=float, default=0.11)
    p.add_argument("--total-steps", type=int, default=300)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lookahead-windows", type=int, default=1)
    p.add_argument("--weaken-factor", type=float, default=0.55)
    p.add_argument("--progress-every", type=int, default=1)

    # Constraint / bookkeeping parameters
    p.add_argument("--lambda-B", type=float, default=0.18)
    p.add_argument("--lambda-S", type=float, default=0.12)
    p.add_argument("--lambda-F", type=float, default=0.20)
    p.add_argument("--lambda-R", type=float, default=0.35)
    p.add_argument("--w-mi", type=float, default=1.0)
    p.add_argument("--w-corr", type=float, default=0.5)
    p.add_argument("--w-link", type=float, default=0.5)
    p.add_argument("--retirement-threshold", type=float, default=0.66)
    p.add_argument("--organizer-large-region-cutoff", type=int, default=6)

    # Graded support controls
    p.add_argument("--sigma-on-threshold", type=float, default=0.08)
    p.add_argument("--edge-on-threshold", type=float, default=0.05)
    p.add_argument("--sigma-step", type=float, default=0.35)
    p.add_argument("--edge-step", type=float, default=0.30)
    p.add_argument("--max-raise-candidates-per-child", type=int, default=4)

    # Exact witness controls
    p.add_argument(
        "--organizer-exact-cutoff",
        type=int,
        default=8,
        help="Use exact organizer witness whenever n_sites <= this cutoff.",
    )

    # Acceptance rule
    p.add_argument(
        "--require-positive-deltaf",
        action="store_true",
        default=False,
        help="If enabled, commit a move only when the best candidate has DeltaF_phys > 0.",
    )

    # Output
    p.add_argument("--compact-json", action="store_true", default=True)
    p.add_argument("--full-json", dest="compact_json", action="store_false")
    p.add_argument("--candidate-summary-topk", type=int, default=8)
    p.add_argument("--json-out", type=str, default="hsf_mesoscape_graded_support_sandbox.json")
    return p.parse_args()


def build_configs(args: argparse.Namespace):
    pcfg = PhysicsConfig(
        n_max=args.n_max,
        n_init=args.n_init,
        seed=args.seed,
        local_scale=args.local_scale,
        pair_scale=args.pair_scale,
        spawn_pair_scale=args.spawn_pair_scale,
        total_steps=args.total_steps,
        dt=args.dt,
        eval_every=args.eval_every,
        lookahead_windows=args.lookahead_windows,
        weaken_factor=args.weaken_factor,
        progress_every=args.progress_every,
        device=args.device,
    )
    scfg = bk.ScoreConfig(
        lambda_B=args.lambda_B,
        lambda_S=args.lambda_S,
        lambda_F=args.lambda_F,
        lambda_R=args.lambda_R,
        w_mi=args.w_mi,
        w_corr=args.w_corr,
        w_link=args.w_link,
        retirement_threshold=args.retirement_threshold,
        organizer_large_region_cutoff=args.organizer_large_region_cutoff,
    )
    return pcfg, scfg


def exact_score_config(cfg: bk.ScoreConfig, n_sites: int, exact_cutoff: int) -> bk.ScoreConfig:
    if n_sites <= int(exact_cutoff):
        return replace(cfg, organizer_large_region_cutoff=n_sites)
    return cfg


def init_graded_state(pcfg: PhysicsConfig, xp):
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs, rng = phys.init_state(pcfg, xp)
    sigma = np.zeros(pcfg.n_max, dtype=np.float64)
    sigma[: pcfg.n_init] = 1.0

    interface_commitment: Dict[Edge, float] = {}
    for e in active_edges:
        interface_commitment[canonical_edge(*e)] = 1.0

    return psi, sigma, interface_commitment, rng


def materialize_state(
    psi,
    sigma: np.ndarray,
    interface_commitment: Dict[Edge, float],
    pcfg: PhysicsConfig,
    sigma_on_threshold: float,
    edge_on_threshold: float,
):
    active_nodes: Set[int] = {i for i in range(len(sigma)) if float(sigma[i]) > float(sigma_on_threshold)}
    dormant_nodes: Set[int] = set(range(len(sigma))) - active_nodes

    local_coeffs = np.zeros(len(sigma), dtype=np.float64)
    for i in active_nodes:
        local_coeffs[i] = float(pcfg.local_scale) * float(np.clip(sigma[i], 0.0, 1.0))

    active_edges: Set[Edge] = set()
    edge_strengths: Dict[Edge, float] = {}
    link_regs: Dict[Edge, np.ndarray] = {}

    for e, w in interface_commitment.items():
        ce = canonical_edge(*e)
        i, j = ce
        if i not in active_nodes or j not in active_nodes:
            continue
        if float(w) <= float(edge_on_threshold):
            continue
        active_edges.add(ce)
        edge_strengths[ce] = float(pcfg.pair_scale) * float(np.clip(w, 0.0, 1.0))
        link_regs[ce] = phys.default_linkreg().copy()

    phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
    return psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs


def clone_graded_state(psi, sigma: np.ndarray, interface_commitment: Dict[Edge, float]):
    return psi.copy(), np.array(sigma, copy=True), dict(interface_commitment)


def active_edge_set(
    sigma: np.ndarray,
    interface_commitment: Dict[Edge, float],
    sigma_on_threshold: float,
    edge_on_threshold: float,
) -> Set[Edge]:
    active_nodes = {i for i in range(len(sigma)) if float(sigma[i]) > float(sigma_on_threshold)}
    out: Set[Edge] = set()
    for e, w in interface_commitment.items():
        ce = canonical_edge(*e)
        if ce[0] in active_nodes and ce[1] in active_nodes and float(w) > float(edge_on_threshold):
            out.add(ce)
    return out


def candidate_raise_supports(
    sigma: np.ndarray,
    interface_commitment: Dict[Edge, float],
    args: argparse.Namespace,
) -> List[Tuple[Edge, int]]:
    partial_or_dormant = [i for i in range(len(sigma)) if float(sigma[i]) < 1.0 - 1e-12]
    existing_edges = sorted(active_edge_set(sigma, interface_commitment, args.sigma_on_threshold, args.edge_on_threshold))
    candidates: List[Tuple[Edge, int]] = []
    counts: Dict[int, int] = {}

    for parents in existing_edges:
        i, j = canonical_edge(*parents)
        for child in partial_or_dormant:
            if child in (i, j):
                continue
            if counts.get(child, 0) >= int(args.max_raise_candidates_per_child):
                continue
            candidates.append((parents, int(child)))
            counts[child] = counts.get(child, 0) + 1
    return candidates


def candidate_sigma_downs(sigma: np.ndarray, args: argparse.Namespace) -> List[int]:
    return [i for i in range(len(sigma)) if float(sigma[i]) > float(args.sigma_on_threshold)]


def candidate_edge_ups(
    sigma: np.ndarray,
    interface_commitment: Dict[Edge, float],
    args: argparse.Namespace,
) -> List[Edge]:
    active_nodes = [i for i in range(len(sigma)) if float(sigma[i]) > float(args.sigma_on_threshold)]
    out: List[Edge] = []
    for idx, i in enumerate(active_nodes):
        for j in active_nodes[idx + 1:]:
            e = canonical_edge(i, j)
            if float(interface_commitment.get(e, 0.0)) < 1.0 - 1e-12:
                out.append(e)
    return out


def candidate_edge_downs(
    sigma: np.ndarray,
    interface_commitment: Dict[Edge, float],
    args: argparse.Namespace,
) -> List[Edge]:
    return sorted(
        e for e, w in interface_commitment.items()
        if float(w) > float(args.edge_on_threshold)
        and float(sigma[e[0]]) > float(args.sigma_on_threshold)
        and float(sigma[e[1]]) > float(args.sigma_on_threshold)
    )


def prepare_raise_support_move(graded_state, parents: Edge, child: int, args: argparse.Namespace):
    psi, sigma, interface_commitment = clone_graded_state(*graded_state)
    i, j = canonical_edge(*parents)
    child = int(child)

    sigma_before = float(sigma[child])
    sigma[child] = float(np.clip(sigma[child] + float(args.sigma_step), 0.0, 1.0))

    for p in (i, j):
        e = canonical_edge(p, child)
        interface_commitment[e] = float(np.clip(interface_commitment.get(e, 0.0) + 0.5 * float(args.edge_step), 0.0, 1.0))

    meta = {
        "move_type": "raise_support",
        "parents": [int(i), int(j)],
        "child": int(child),
        "sigma_before": float(sigma_before),
        "sigma_after": float(sigma[child]),
    }
    return (psi, sigma, interface_commitment), meta


def prepare_sigma_down_move(graded_state, node: int, args: argparse.Namespace):
    psi, sigma, interface_commitment = clone_graded_state(*graded_state)
    node = int(node)
    sigma_before = float(sigma[node])
    sigma[node] = float(max(0.0, sigma[node] - float(args.sigma_step)))
    ratio = float(sigma[node] / max(1e-12, sigma_before)) if sigma_before > 0 else 0.0

    for e in list(interface_commitment.keys()):
        if node in e:
            interface_commitment[e] = float(np.clip(interface_commitment[e] * ratio, 0.0, 1.0))
            if interface_commitment[e] < 1e-12:
                interface_commitment[e] = 0.0

    meta = {
        "move_type": "lower_support",
        "node": int(node),
        "sigma_before": float(sigma_before),
        "sigma_after": float(sigma[node]),
    }
    return (psi, sigma, interface_commitment), meta


def prepare_edge_up_move(graded_state, edge: Edge, args: argparse.Namespace):
    psi, sigma, interface_commitment = clone_graded_state(*graded_state)
    e = canonical_edge(*edge)
    w_before = float(interface_commitment.get(e, 0.0))
    interface_commitment[e] = float(np.clip(w_before + float(args.edge_step), 0.0, 1.0))
    meta = {
        "move_type": "edge_up",
        "edge": [int(e[0]), int(e[1])],
        "w_before": float(w_before),
        "w_after": float(interface_commitment[e]),
    }
    return (psi, sigma, interface_commitment), meta


def prepare_edge_down_move(graded_state, edge: Edge, args: argparse.Namespace):
    psi, sigma, interface_commitment = clone_graded_state(*graded_state)
    e = canonical_edge(*edge)
    w_before = float(interface_commitment.get(e, 0.0))
    interface_commitment[e] = float(max(0.0, w_before - float(args.edge_step)))
    meta = {
        "move_type": "edge_down",
        "edge": [int(e[0]), int(e[1])],
        "w_before": float(w_before),
        "w_after": float(interface_commitment[e]),
    }
    return (psi, sigma, interface_commitment), meta


def evolve_materialized_state(graded_state, pcfg: PhysicsConfig, args: argparse.Namespace, xp):
    prepared = materialize_state(
        graded_state[0],
        graded_state[1],
        graded_state[2],
        pcfg,
        args.sigma_on_threshold,
        args.edge_on_threshold,
    )
    return phys.evolve_prepared_state(prepared, pcfg, xp)


def move_object_for_kind(meta: Dict[str, object]) -> Tuple[str, Any]:
    move_type = str(meta["move_type"])
    if move_type == "raise_support":
        return "birth", {"parents": tuple(meta["parents"]), "child": int(meta["child"])}
    if move_type == "lower_support":
        return "retire", int(meta["node"])
    if move_type == "edge_up":
        return "transfer", {"edge": canonical_edge(*meta["edge"])}
    if move_type == "edge_down":
        return "weaken", canonical_edge(*meta["edge"])
    raise ValueError(f"Unknown move_type: {move_type}")


def score_move(
    meta: Dict[str, object],
    graded_before,
    graded_after,
    pcfg: PhysicsConfig,
    scfg: bk.ScoreConfig,
    args: argparse.Namespace,
    xp,
):
    state_b = evolve_materialized_state(graded_before, pcfg, args, xp)
    state_a = evolve_materialized_state(graded_after, pcfg, args, xp)

    psi_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = state_b
    psi_a, active_nodes_a, dormant_nodes_a, active_edges_a, local_coeffs_a, edge_strengths_a, link_regs_a = state_a
    n_sites = int(local_coeffs_b.shape[0])

    scfg_exact = exact_score_config(scfg, n_sites, args.organizer_exact_cutoff)

    expr_b = bk.local_expression(
        psi_b, active_nodes_b, active_edges_b, edge_strengths_b, link_regs_b, scfg_exact, GM_MATRICES, xp, n_sites
    )
    expr_a = bk.local_expression(
        psi_a, active_nodes_a, active_edges_a, edge_strengths_a, link_regs_a, scfg_exact, GM_MATRICES, xp, n_sites
    )
    dE_raw = float(expr_a - expr_b)

    cb_b = bk.bandwidth_burden(active_edges_b, link_regs_b)
    cb_a = bk.bandwidth_burden(active_edges_a, link_regs_a)
    dCB = float(cb_a - cb_b)

    cs_b = bk.spread_burden(active_nodes_b, active_edges_b)
    cs_a = bk.spread_burden(active_nodes_a, active_edges_a)
    dCS = float(cs_a - cs_b)

    core_before = bk.dominant_core_snapshot(
        psi_b, active_nodes_b, active_edges_b, edge_strengths_b, GM_MATRICES, xp, n_sites, scfg_exact, link_regs_b
    )

    nr_kind, nr_obj = move_object_for_kind(meta)
    nr = bk.no_refolding_witness(
        nr_kind,
        nr_obj,
        psi_b,
        psi_a,
        active_nodes_b,
        active_edges_b,
        edge_strengths_b,
        active_nodes_a,
        active_edges_a,
        edge_strengths_a,
        core_before,
        scfg_exact,
        GM_MATRICES,
        xp,
        n_sites,
    )
    dCF = float(max(0.0, 1.0 - nr["F_org"]) + max(0.0, nr["W_func"]))

    deltaF_phys = float(
        dE_raw
        - scfg.lambda_B * dCB
        - scfg.lambda_S * dCS
        - scfg.lambda_F * dCF
        - scfg.lambda_R * nr["W_NR"]
    )

    diag = {
        "move_type": str(meta["move_type"]),
        "dE_raw": float(dE_raw),
        "dCB": float(dCB),
        "dCS": float(dCS),
        "dCF": float(dCF),
        "deltaF_phys": float(deltaF_phys),
        "F_org": float(nr["F_org"]),
        "W_func": float(nr["W_func"]),
        "W_NR": float(nr["W_NR"]),
        "W_ep": float(nr["W_ep"]),
        "W_slack": float(nr["W_slack"]),
        "W_sector": float(nr["W_sector"]),
        "hits_core_pair": bool(nr.get("hits_core_pair", False)),
        "core_weaken": bool(nr.get("core_weaken", False)),
        "destructive_weaken": bool(nr.get("destructive_weaken", False)),
        "lawful_shell_reexpression": bool(nr.get("lawful_shell_reexpression", False)),
        "n_active_before": int(len(active_nodes_b)),
        "n_active_after": int(len(active_nodes_a)),
        "active_edges_before": int(len(active_edges_b)),
        "active_edges_after": int(len(active_edges_a)),
        "core_before": core_before,
        **meta,
    }
    return diag, state_a


def compact_move_diag(diag: Dict[str, object]) -> Dict[str, object]:
    out = {
        "move_type": diag["move_type"],
        "dE_raw": float(diag["dE_raw"]),
        "dCB": float(diag["dCB"]),
        "dCS": float(diag["dCS"]),
        "dCF": float(diag["dCF"]),
        "deltaF_phys": float(diag["deltaF_phys"]),
        "F_org": float(diag["F_org"]),
        "W_func": float(diag["W_func"]),
        "W_NR": float(diag["W_NR"]),
        "W_ep": float(diag["W_ep"]),
        "W_slack": float(diag["W_slack"]),
        "W_sector": float(diag["W_sector"]),
        "hits_core_pair": bool(diag["hits_core_pair"]),
        "core_weaken": bool(diag["core_weaken"]),
        "destructive_weaken": bool(diag["destructive_weaken"]),
        "lawful_shell_reexpression": bool(diag["lawful_shell_reexpression"]),
        "n_active_before": int(diag["n_active_before"]),
        "n_active_after": int(diag["n_active_after"]),
    }
    for key in ("parents", "child", "node", "edge", "sigma_before", "sigma_after", "w_before", "w_after"):
        if key in diag:
            out[key] = diag[key]
    return out


def sigma_summary(sigma: np.ndarray) -> Dict[str, object]:
    return {
        "sigma": [float(x) for x in sigma.tolist()],
        "mean_sigma": float(np.mean(sigma)),
        "n_full": int(np.sum(sigma >= 0.999)),
        "n_partial": int(np.sum((sigma > 1e-9) & (sigma < 0.999))),
        "n_zero": int(np.sum(sigma <= 1e-9)),
    }


def commitment_summary(interface_commitment: Dict[Edge, float]) -> Dict[str, object]:
    vals = list(float(v) for v in interface_commitment.values())
    return {
        "n_interfaces_tracked": int(len(vals)),
        "mean_commitment": float(np.mean(vals)) if vals else 0.0,
        "max_commitment": float(np.max(vals)) if vals else 0.0,
        "min_commitment": float(np.min(vals)) if vals else 0.0,
        "top_interfaces": [
            {"edge": [int(e[0]), int(e[1])], "commitment": float(v)}
            for e, v in sorted(interface_commitment.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        ],
    }


def candidate_summary_block(move_diags: List[Dict[str, object]], topk: int) -> Dict[str, object]:
    compact = [compact_move_diag(d) for d in move_diags]
    top_by_deltaF_phys = sorted(compact, key=lambda d: float(d["deltaF_phys"]), reverse=True)[:topk]
    top_by_dE_raw = sorted(compact, key=lambda d: float(d["dE_raw"]), reverse=True)[:topk]
    return {
        "candidate_count": int(len(move_diags)),
        "top_by_deltaF_phys": top_by_deltaF_phys,
        "top_by_dE_raw": top_by_dE_raw,
    }


def tie_key(diag: Dict[str, object]) -> Tuple[float, ...]:
    return (
        -float(diag["dE_raw"]),
        float(diag["W_NR"]),
        -float(diag["F_org"]),
        float(diag["W_func"]),
        float(diag["dCB"]),
        float(diag["dCS"]),
        float(diag["dCF"]),
        float(diag["n_active_after"]),
        0.0 if str(diag["move_type"]) == "raise_support" else 1.0
        if str(diag["move_type"]) == "edge_up" else 2.0
        if str(diag["move_type"]) == "edge_down" else 3.0,
    )


def run_sim(pcfg: PhysicsConfig, scfg: bk.ScoreConfig, args: argparse.Namespace):
    xp, is_gpu = phys.get_array_module(pcfg.device)
    psi, sigma, interface_commitment, rng = init_graded_state(pcfg, xp)

    accepted_moves: List[Dict[str, object]] = []
    snapshots: List[Dict[str, object]] = []
    active_trace: List[int] = []
    edge_trace: List[int] = []
    sigma_mean_trace: List[float] = []

    move_counts = {
        "raise_support": 0,
        "lower_support": 0,
        "edge_up": 0,
        "edge_down": 0,
        "no_move": 0,
    }

    for step in range(pcfg.eval_every, pcfg.total_steps + 1, pcfg.eval_every):
        graded_before = (psi, sigma, interface_commitment)
        baseline_state = evolve_materialized_state(graded_before, pcfg, args, xp)

        raise_candidates = candidate_raise_supports(sigma, interface_commitment, args)
        sigma_down_candidates = candidate_sigma_downs(sigma, args)
        edge_up_candidates = candidate_edge_ups(sigma, interface_commitment, args)
        edge_down_candidates = candidate_edge_downs(sigma, interface_commitment, args)

        candidate_specs: List[Tuple[Dict[str, object], Tuple[Any, np.ndarray, Dict[Edge, float]]]] = []

        for parents, child in raise_candidates:
            graded_after, meta = prepare_raise_support_move(graded_before, parents, child, args)
            candidate_specs.append((meta, graded_after))

        for node in sigma_down_candidates:
            graded_after, meta = prepare_sigma_down_move(graded_before, node, args)
            candidate_specs.append((meta, graded_after))

        for edge in edge_up_candidates:
            graded_after, meta = prepare_edge_up_move(graded_before, edge, args)
            candidate_specs.append((meta, graded_after))

        for edge in edge_down_candidates:
            graded_after, meta = prepare_edge_down_move(graded_before, edge, args)
            candidate_specs.append((meta, graded_after))

        move_diags: List[Dict[str, object]] = []
        scored_candidates: List[Tuple[float, Tuple[float, ...], Dict[str, object], Tuple[Any, np.ndarray, Dict[Edge, float]]]] = []

        for meta, graded_after in candidate_specs:
            diag, _ = score_move(meta, graded_before, graded_after, pcfg, scfg, args, xp)
            diag["step_hint"] = int(step)
            move_diags.append(diag)
            scored_candidates.append((float(diag["deltaF_phys"]), tie_key(diag), diag, graded_after))

        best_diag: Optional[Dict[str, object]] = None
        best_graded_after = None
        accepted_move_record = None
        n_this_eval = {k: 0 for k in move_counts.keys()}

        if scored_candidates:
            scored_candidates.sort(key=lambda item: (-item[0], item[1]))
            best_deltaF, best_diag, best_graded_after = scored_candidates[0][0], scored_candidates[0][2], scored_candidates[0][3]

            should_accept = True
            if bool(args.require_positive_deltaf) and float(best_deltaF) <= 0.0:
                should_accept = False

            if should_accept and best_diag is not None and best_graded_after is not None:
                evolved_best_state = evolve_materialized_state(best_graded_after, pcfg, args, xp)
                psi_eff, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = evolved_best_state
                psi, sigma, interface_commitment = best_graded_after
                psi = psi_eff

                accepted_move_record = compact_move_diag(best_diag)
                accepted_moves.append(accepted_move_record)
                move_counts[str(best_diag["move_type"])] += 1
                n_this_eval[str(best_diag["move_type"])] += 1
            else:
                psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = baseline_state
                move_counts["no_move"] += 1
                n_this_eval["no_move"] += 1
        else:
            psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = baseline_state
            move_counts["no_move"] += 1
            n_this_eval["no_move"] += 1

        state_now = materialize_state(psi, sigma, interface_commitment, pcfg, args.sigma_on_threshold, args.edge_on_threshold)
        psi_now, active_nodes_now, dormant_nodes_now, active_edges_now, local_coeffs_now, edge_strengths_now, link_regs_now = state_now
        n_sites = int(local_coeffs_now.shape[0])
        scfg_exact = exact_score_config(scfg, n_sites, args.organizer_exact_cutoff)

        core_now = bk.dominant_core_snapshot(
            psi_now, active_nodes_now, active_edges_now, edge_strengths_now, GM_MATRICES, xp, n_sites, scfg_exact, link_regs_now
        )
        metrics = phys.metric_snapshot(active_nodes_now, active_edges_now, edge_strengths_now)
        gstats = phys.graph_stats(active_nodes_now, active_edges_now)

        snap = {
            "step": int(step),
            "active_nodes": list(sorted(active_nodes_now)),
            "dormant_nodes": list(sorted(dormant_nodes_now)),
            "active_edges": [list(e) for e in sorted(active_edges_now)],
            "active_edge_count": int(len(active_edges_now)),
            "dominant_core": core_now,
            "metric": metrics,
            "graph_stats": gstats,
            "sigma_summary": sigma_summary(sigma),
            "commitment_summary": commitment_summary(interface_commitment),
            "n_raise_support_this_eval": int(n_this_eval["raise_support"]),
            "n_lower_support_this_eval": int(n_this_eval["lower_support"]),
            "n_edge_up_this_eval": int(n_this_eval["edge_up"]),
            "n_edge_down_this_eval": int(n_this_eval["edge_down"]),
            "n_no_move_this_eval": int(n_this_eval["no_move"]),
            "accepted_move": accepted_move_record,
        }

        if args.compact_json:
            snap["candidate_summary"] = candidate_summary_block(move_diags, int(args.candidate_summary_topk))
        else:
            snap["candidate_move_diagnostics"] = move_diags

        snapshots.append(snap)

        active_trace.append(len(active_nodes_now))
        edge_trace.append(len(active_edges_now))
        sigma_mean_trace.append(float(np.mean(sigma)))

        if pcfg.progress_every > 0 and len(snapshots) % pcfg.progress_every == 0:
            cp = core_now["core_pair"] if core_now else None
            winner = accepted_move_record["move_type"] if accepted_move_record is not None else None
            best_df = accepted_move_record["deltaF_phys"] if accepted_move_record is not None else None
            print(
                f"[eval {len(snapshots):03d}] step={step:4d} active={len(active_nodes_now):2d} "
                f"edges={len(active_edges_now):3d} mean_sigma={np.mean(sigma):.3f} "
                f"core={cp} raise={n_this_eval['raise_support']} lower={n_this_eval['lower_support']} "
                f"edge_up={n_this_eval['edge_up']} edge_down={n_this_eval['edge_down']} "
                f"no_move={n_this_eval['no_move']} winner={winner} deltaF={best_df} cand={len(move_diags)}"
            )

    return {
        "script": "hsf_mesoscape_graded_support_sandbox.py",
        "physics_config": asdict(pcfg),
        "bookkeeping_config": asdict(scfg),
        "selector_config": {
            "require_positive_deltaf": bool(args.require_positive_deltaf),
            "organizer_exact_cutoff": int(args.organizer_exact_cutoff),
            "compact_json": bool(args.compact_json),
            "candidate_summary_topk": int(args.candidate_summary_topk),
            "selector_rule": "best move by deltaF_phys only; no hard vetoes",
        },
        "move_counts": {k: int(v) for k, v in move_counts.items()},
        "final_sigma": [float(x) for x in sigma.tolist()],
        "final_interface_commitment": [
            {"edge": [int(e[0]), int(e[1])], "commitment": float(v)}
            for e, v in sorted(interface_commitment.items())
        ],
        "final_active_nodes": snapshots[-1]["active_nodes"] if snapshots else [],
        "final_active_edges": snapshots[-1]["active_edges"] if snapshots else [],
        "active_count_trace": active_trace,
        "edge_count_trace": edge_trace,
        "sigma_mean_trace": sigma_mean_trace,
        "accepted_moves": accepted_moves,
        "snapshots": snapshots,
        "gpu_enabled": bool(is_gpu),
    }


def main() -> None:
    args = parse_args()
    pcfg, scfg = build_configs(args)
    result = run_sim(pcfg, scfg, args)

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()