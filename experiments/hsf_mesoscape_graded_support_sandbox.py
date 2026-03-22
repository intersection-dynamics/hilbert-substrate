#!/usr/bin/env python3
# filename: hsf_mesoscape_graded_support_sandbox.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

import hsf_mesoscale_bookkeeping as bk
import hsf_mesoscale_physics_core as phys
from hsf_mesoscale_physics_core import PhysicsConfig, GM_MATRICES, canonical_edge

Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HSF mesoscale graded-support sandbox v3. "
            "Subsystems carry graded commitment sigma_i in [0,1], interfaces carry graded "
            "commitment w_ij in [0,1], and edge_up is hardened so it cannot win on persistence alone."
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

    # Bookkeeping / lawful move base
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

    # Witness-based graded expression weights
    p.add_argument("--w-graded-odiff", type=float, default=0.35)
    p.add_argument("--w-graded-relief", type=float, default=0.30)
    p.add_argument("--w-graded-distinct", type=float, default=0.20)
    p.add_argument("--w-graded-redundancy", type=float, default=0.25)
    p.add_argument("--w-graded-persistence", type=float, default=0.15)
    p.add_argument("--graded-odiff-clip", type=float, default=0.25)

    # Hardened edge_up gates
    p.add_argument(
        "--edge-up-odiff-min",
        type=float,
        default=1e-6,
        help="Minimum positive delta_Odiff required for edge_up to receive positive witness credit.",
    )
    p.add_argument(
        "--edge-up-relief-min",
        type=float,
        default=0.02,
        help="Minimum real interface-relief proxy required for edge_up to receive positive witness credit.",
    )
    p.add_argument(
        "--edge-up-distinct-min",
        type=float,
        default=0.02,
        help="Minimum distinctness improvement required for edge_up to receive positive witness credit.",
    )
    p.add_argument(
        "--edge-up-hard-gate-negative-raw",
        action="store_true",
        help=(
            "If enabled, edge_up gets no positive witness correction whenever dE_expr_raw < 0 "
            "and there is no positive odiff or relief gain."
        ),
    )

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


def neighbor_map_from_commitment(
    sigma: np.ndarray,
    interface_commitment: Dict[Edge, float],
    sigma_on_threshold: float,
    edge_on_threshold: float,
) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for e in active_edge_set(sigma, interface_commitment, sigma_on_threshold, edge_on_threshold):
        i, j = e
        out.setdefault(i, set()).add(j)
        out.setdefault(j, set()).add(i)
    return out


def candidate_raise_supports(
    sigma: np.ndarray,
    interface_commitment: Dict[Edge, float],
    args: argparse.Namespace,
) -> List[Tuple[Edge, int]]:
    partial_or_dormant = [i for i in range(len(sigma)) if float(sigma[i]) < 1.0 - 1e-12]
    existing_edges = sorted(
        active_edge_set(sigma, interface_commitment, args.sigma_on_threshold, args.edge_on_threshold)
    )
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


def _odiff_adjustment_from_audit(odiff_audit, args: argparse.Namespace) -> Tuple[float, Dict[str, float]]:
    if odiff_audit is None:
        return 0.0, {
            "expr_odiff_delta_raw": 0.0,
            "expr_odiff_delta_clipped": 0.0,
            "expr_odiff_adjustment": 0.0,
        }
    raw = float(odiff_audit.delta_odiff)
    clipped = float(np.clip(raw, -float(args.graded_odiff_clip), float(args.graded_odiff_clip)))
    adj = float(args.w_graded_odiff) * clipped
    return adj, {
        "expr_odiff_delta_raw": raw,
        "expr_odiff_delta_clipped": clipped,
        "expr_odiff_adjustment": adj,
    }


def _graded_raise_witness(meta: Dict[str, object], state_b, state_a, core_before, scfg: bk.ScoreConfig, args: argparse.Namespace, xp):
    parents = tuple(meta["parents"])  # type: ignore[arg-type]
    child = int(meta["child"])
    sigma_before = float(meta["sigma_before"])
    sigma_after = float(meta["sigma_after"])
    sigma_increment = max(0.0, sigma_after - sigma_before)

    birth_info = bk.birth_justification_witness(
        {"parents": tuple(parents), "child": child},
        state_b,
        state_a,
        scfg,
        GM_MATRICES,
        xp,
    )
    odiff_audit = bk.compute_move_local_odiff_audit(
        "birth",
        {"parents": tuple(parents), "child": child},
        state_b,
        state_a,
        core_before,
        scfg,
    )

    odiff_pos = max(0.0, float(odiff_audit.delta_odiff))
    odiff_neg = max(0.0, -float(odiff_audit.delta_odiff))
    relief = float(birth_info.get("birth_parent_relief", 0.0))
    distinct = float(birth_info.get("birth_distinctness", 0.0) if "birth_distinctness" in birth_info else 0.0)
    novelty = float(birth_info.get("birth_novelty", 0.0))
    support_factor = float(np.clip(sigma_increment, 0.0, 1.0))

    witness_adj = (
        float(args.w_graded_relief) * support_factor * relief
        + float(args.w_graded_distinct) * support_factor * distinct
        + float(args.w_graded_persistence) * support_factor * novelty
        - float(args.w_graded_redundancy) * odiff_neg
    )
    odiff_adj, odiff_info = _odiff_adjustment_from_audit(odiff_audit, args)

    info = {
        "sigma_increment": float(sigma_increment),
        "graded_relief_term": float(float(args.w_graded_relief) * support_factor * relief),
        "graded_distinct_term": float(float(args.w_graded_distinct) * support_factor * distinct),
        "graded_persistence_term": float(float(args.w_graded_persistence) * support_factor * novelty),
        "graded_redundancy_penalty": float(-float(args.w_graded_redundancy) * odiff_neg),
        "graded_witness_adjustment": float(witness_adj),
        **birth_info,
        **odiff_info,
    }
    return float(witness_adj + odiff_adj), info, odiff_audit


def _graded_edge_up_witness(
    meta: Dict[str, object],
    state_b,
    state_a,
    core_before,
    scfg: bk.ScoreConfig,
    args: argparse.Namespace,
    xp,
    dE_raw: float,
):
    edge = tuple(meta["edge"])  # type: ignore[arg-type]
    i, j = canonical_edge(*edge)
    w_before = float(meta["w_before"])
    w_after = float(meta["w_after"])
    w_increment = max(0.0, w_after - w_before)

    odiff_audit = bk.compute_move_local_odiff_audit(
        "transfer",
        {"edge": canonical_edge(i, j)},
        state_b,
        state_a,
        core_before,
        scfg,
    )

    # Real local differentiated-role gain only.
    odiff_pos = max(0.0, float(odiff_audit.delta_odiff))
    odiff_neg = max(0.0, -float(odiff_audit.delta_odiff))

    # Relief proxy from reduced overlap. This is a real local structural change, not persistence.
    overlap_before = float(odiff_audit.mean_pair_overlap_before)
    overlap_after = float(odiff_audit.mean_pair_overlap_after)
    relief_gain = max(0.0, overlap_before - overlap_after)

    # Distinctness gain from overlap reduction.
    distinct_gain = max(0.0, (1.0 - overlap_after) - (1.0 - overlap_before))

    support_factor = float(np.clip(w_increment, 0.0, 1.0))

    # Hard gate:
    # no positive witness correction if raw expression is already negative AND
    # there is no real odiff gain AND no real relief gain.
    hard_gate_triggered = bool(
        args.edge_up_hard_gate_negative_raw
        and float(dE_raw) < 0.0
        and float(odiff_pos) <= float(args.edge_up_odiff_min)
        and float(relief_gain) <= float(args.edge_up_relief_min)
        and float(distinct_gain) <= float(args.edge_up_distinct_min)
    )

    if hard_gate_triggered:
        witness_adj = -float(args.w_graded_redundancy) * odiff_neg
        odiff_adj = 0.0
        odiff_info = {
            "expr_odiff_delta_raw": float(odiff_audit.delta_odiff),
            "expr_odiff_delta_clipped": 0.0,
            "expr_odiff_adjustment": 0.0,
        }
        relief_term = 0.0
        distinct_term = 0.0
        redundancy_penalty = -float(args.w_graded_redundancy) * odiff_neg
    else:
        # Positive witness credit only from real differentiated-role gain or real relief/distinctness gain.
        relief_term = (
            float(args.w_graded_relief) * support_factor * relief_gain
            if relief_gain > float(args.edge_up_relief_min) else 0.0
        )
        distinct_term = (
            float(args.w_graded_distinct) * support_factor * distinct_gain
            if distinct_gain > float(args.edge_up_distinct_min) else 0.0
        )
        redundancy_penalty = -float(args.w_graded_redundancy) * odiff_neg

        witness_adj = float(relief_term + distinct_term + redundancy_penalty)

        if odiff_pos > float(args.edge_up_odiff_min):
            odiff_adj, odiff_info = _odiff_adjustment_from_audit(odiff_audit, args)
        else:
            odiff_adj = 0.0
            odiff_info = {
                "expr_odiff_delta_raw": float(odiff_audit.delta_odiff),
                "expr_odiff_delta_clipped": 0.0,
                "expr_odiff_adjustment": 0.0,
            }

    info = {
        "w_increment": float(w_increment),
        "edge_up_hard_gate_triggered": bool(hard_gate_triggered),
        "edge_up_relief_gain": float(relief_gain),
        "edge_up_distinct_gain": float(distinct_gain),
        "graded_relief_term": float(relief_term),
        "graded_distinct_term": float(distinct_term),
        "graded_redundancy_penalty": float(redundancy_penalty),
        "graded_witness_adjustment": float(witness_adj),
        **odiff_info,
    }
    return float(witness_adj + odiff_adj), info, odiff_audit


def _graded_lower_support_witness(meta: Dict[str, object], state_b, state_a, core_before, scfg: bk.ScoreConfig, args: argparse.Namespace, xp):
    node = int(meta["node"])
    psi_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = state_b
    n_sites = int(local_coeffs_b.shape[0])

    retire_info = bk.retirement_readiness(
        node, psi_b, active_nodes_b, active_edges_b, edge_strengths_b, link_regs_b, core_before, scfg, GM_MATRICES, xp, n_sites
    )
    nr_tmp = {"W_func": 0.0}
    expr_adj, expr_info = bk._expression_adjustment_for_weaken_or_retire(
        "retire", node, state_b, state_a, core_before, nr_tmp, retire_info, scfg
    )

    odiff_audit = bk.compute_move_local_odiff_audit(
        "retire",
        node,
        state_b,
        state_a,
        core_before,
        scfg,
    )
    odiff_pos = max(0.0, float(odiff_audit.delta_odiff))
    odiff_neg = max(0.0, -float(odiff_audit.delta_odiff))

    witness_adj = (
        float(expr_adj)
        + float(args.w_graded_persistence) * odiff_pos
        - float(args.w_graded_redundancy) * max(0.0, odiff_neg - 0.05)
    )
    odiff_adj, odiff_info = _odiff_adjustment_from_audit(odiff_audit, args)

    info = {
        "graded_structural_adjustment": float(expr_adj),
        "graded_persistence_term": float(float(args.w_graded_persistence) * odiff_pos),
        "graded_redundancy_penalty": float(-float(args.w_graded_redundancy) * max(0.0, odiff_neg - 0.05)),
        "graded_witness_adjustment": float(witness_adj),
        "retirement_info": retire_info,
        **expr_info,
        **odiff_info,
    }
    return float(witness_adj + odiff_adj), info, odiff_audit


def _graded_edge_down_witness(meta: Dict[str, object], state_b, state_a, core_before, scfg: bk.ScoreConfig, args: argparse.Namespace, xp):
    edge = tuple(meta["edge"])  # type: ignore[arg-type]
    e = canonical_edge(*edge)
    nr_tmp = bk.no_refolding_witness(
        "weaken",
        e,
        state_b[0],
        state_a[0],
        state_b[1],
        state_b[3],
        state_b[5],
        state_a[1],
        state_a[3],
        state_a[5],
        core_before,
        scfg,
        GM_MATRICES,
        xp,
        int(state_b[4].shape[0]),
    )
    expr_adj, expr_info = bk._expression_adjustment_for_weaken_or_retire(
        "weaken", e, state_b, state_a, core_before, nr_tmp, None, scfg
    )
    odiff_audit = bk.compute_move_local_odiff_audit(
        "weaken",
        e,
        state_b,
        state_a,
        core_before,
        scfg,
    )
    odiff_pos = max(0.0, float(odiff_audit.delta_odiff))
    odiff_neg = max(0.0, -float(odiff_audit.delta_odiff))

    witness_adj = (
        float(expr_adj)
        + float(args.w_graded_distinct) * odiff_pos
        - float(args.w_graded_redundancy) * max(0.0, odiff_neg - 0.05)
    )
    odiff_adj, odiff_info = _odiff_adjustment_from_audit(odiff_audit, args)

    info = {
        "graded_structural_adjustment": float(expr_adj),
        "graded_distinct_term": float(float(args.w_graded_distinct) * odiff_pos),
        "graded_redundancy_penalty": float(-float(args.w_graded_redundancy) * max(0.0, odiff_neg - 0.05)),
        "graded_witness_adjustment": float(witness_adj),
        **expr_info,
        **odiff_info,
    }
    return float(witness_adj + odiff_adj), info, odiff_audit


def score_graded_move(
    meta: Dict[str, object],
    graded_before,
    graded_after,
    pcfg: PhysicsConfig,
    scfg: bk.ScoreConfig,
    args: argparse.Namespace,
    xp,
):
    psi_b, sigma_b, commit_b = graded_before
    psi_a, sigma_a, commit_a = graded_after

    state_b = materialize_state(psi_b, sigma_b, commit_b, pcfg, args.sigma_on_threshold, args.edge_on_threshold)
    state_a = materialize_state(psi_a, sigma_a, commit_a, pcfg, args.sigma_on_threshold, args.edge_on_threshold)
    state_a = phys.evolve_prepared_state(state_a, pcfg, xp)

    psi0_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = state_b
    psi1_a, active_nodes_a, dormant_nodes_a, active_edges_a, local_coeffs_a, edge_strengths_a, link_regs_a = state_a
    n_sites = int(local_coeffs_b.shape[0])

    expr_b = bk.local_expression(psi0_b, active_nodes_b, active_edges_b, edge_strengths_b, link_regs_b, scfg, GM_MATRICES, xp, n_sites)
    expr_a = bk.local_expression(psi1_a, active_nodes_a, active_edges_a, edge_strengths_a, link_regs_a, scfg, GM_MATRICES, xp, n_sites)
    dE_raw = float(expr_a - expr_b)
    dE = float(dE_raw)

    core_before = bk.dominant_core_snapshot(
        psi0_b, active_nodes_b, active_edges_b, edge_strengths_b, GM_MATRICES, xp, n_sites, scfg, link_regs_b
    )

    move_type = str(meta["move_type"])
    move_object_for_nr = None
    nr_kind = "transfer"
    extra_diag: Dict[str, object] = {}
    odiff_audit = None

    if move_type == "raise_support":
        witness_adj, witness_info, odiff_audit = _graded_raise_witness(meta, state_b, state_a, core_before, scfg, args, xp)
        dE = float(dE_raw + witness_adj)
        nr_kind = "birth"
        move_object_for_nr = {"parents": tuple(meta["parents"]), "child": int(meta["child"])}  # type: ignore[arg-type]
        extra_diag.update(witness_info)

    elif move_type == "lower_support":
        witness_adj, witness_info, odiff_audit = _graded_lower_support_witness(meta, state_b, state_a, core_before, scfg, args, xp)
        dE = float(dE_raw + witness_adj)
        nr_kind = "retire"
        move_object_for_nr = int(meta["node"])
        extra_diag.update(witness_info)

    elif move_type == "edge_up":
        witness_adj, witness_info, odiff_audit = _graded_edge_up_witness(meta, state_b, state_a, core_before, scfg, args, xp, dE_raw)
        dE = float(dE_raw + witness_adj)
        nr_kind = "transfer"
        move_object_for_nr = {"edge": canonical_edge(*meta["edge"])}  # type: ignore[arg-type]
        extra_diag.update(witness_info)

    elif move_type == "edge_down":
        witness_adj, witness_info, odiff_audit = _graded_edge_down_witness(meta, state_b, state_a, core_before, scfg, args, xp)
        dE = float(dE_raw + witness_adj)
        nr_kind = "weaken"
        move_object_for_nr = canonical_edge(*meta["edge"])  # type: ignore[arg-type]
        extra_diag.update(witness_info)

    cb_b = bk.bandwidth_burden(active_edges_b, link_regs_b)
    cb_a = bk.bandwidth_burden(active_edges_a, link_regs_a)
    dCB = float(cb_a - cb_b)

    cs_b = bk.spread_burden(active_nodes_b, active_edges_b)
    cs_a = bk.spread_burden(active_nodes_a, active_edges_a)
    dCS = float(cs_a - cs_b)

    nr = bk.no_refolding_witness(
        nr_kind,
        move_object_for_nr,
        psi0_b,
        psi1_a,
        active_nodes_b,
        active_edges_b,
        edge_strengths_b,
        active_nodes_a,
        active_edges_a,
        edge_strengths_a,
        core_before,
        scfg,
        GM_MATRICES,
        xp,
        n_sites,
    )
    dCF = float(max(0.0, 1.0 - nr["F_org"]) + max(0.0, nr["W_func"]))

    deltaF = float(dE - scfg.lambda_B * dCB - scfg.lambda_S * dCS - scfg.lambda_F * dCF - scfg.lambda_R * nr["W_NR"])

    diag = {
        "move_type": move_type,
        "deltaF": float(deltaF),
        "dE_expr": float(dE),
        "dE_expr_raw": float(dE_raw),
        "dCB": float(dCB),
        "dCS": float(dCS),
        "dCF": float(dCF),
        "core_before": core_before,
        "sigma_before": [float(x) for x in sigma_b.tolist()],
        "sigma_after": [float(x) for x in sigma_a.tolist()],
        "n_active_before": int(len(active_nodes_b)),
        "n_active_after": int(len(active_nodes_a)),
        **extra_diag,
        **nr,
    }
    if odiff_audit is not None:
        diag["A_before_R"] = int(odiff_audit.active_count_before)
        diag["A_after_R"] = int(odiff_audit.active_count_after)
        diag["Odiff_before_R"] = float(odiff_audit.odiff_before)
        diag["Odiff_after_R"] = float(odiff_audit.odiff_after)
        diag["delta_Odiff_R"] = float(odiff_audit.delta_odiff)
    diag.update(meta)
    return deltaF, diag, state_a


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
    }

    for step in range(pcfg.eval_every, pcfg.total_steps + 1, pcfg.eval_every):
        graded_before = (psi, sigma, interface_commitment)

        raise_candidates = candidate_raise_supports(sigma, interface_commitment, args)
        sigma_down_candidates = candidate_sigma_downs(sigma, args)
        edge_up_candidates = candidate_edge_ups(sigma, interface_commitment, args)
        edge_down_candidates = candidate_edge_downs(sigma, interface_commitment, args)

        move_diags: List[Dict[str, object]] = []
        best_diag: Optional[Dict[str, object]] = None
        best_delta = 0.0
        best_state = None
        best_graded_after = None

        for parents, child in raise_candidates:
            graded_after, meta = prepare_raise_support_move(graded_before, parents, child, args)
            dF, diag, state_after = score_graded_move(meta, graded_before, graded_after, pcfg, scfg, args, xp)
            diag["step_hint"] = int(step)
            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_diag = diag
                best_state = state_after
                best_graded_after = graded_after

        for node in sigma_down_candidates:
            graded_after, meta = prepare_sigma_down_move(graded_before, node, args)
            dF, diag, state_after = score_graded_move(meta, graded_before, graded_after, pcfg, scfg, args, xp)
            diag["step_hint"] = int(step)
            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_diag = diag
                best_state = state_after
                best_graded_after = graded_after

        for edge in edge_up_candidates:
            graded_after, meta = prepare_edge_up_move(graded_before, edge, args)
            dF, diag, state_after = score_graded_move(meta, graded_before, graded_after, pcfg, scfg, args, xp)
            diag["step_hint"] = int(step)
            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_diag = diag
                best_state = state_after
                best_graded_after = graded_after

        for edge in edge_down_candidates:
            graded_after, meta = prepare_edge_down_move(graded_before, edge, args)
            dF, diag, state_after = score_graded_move(meta, graded_before, graded_after, pcfg, scfg, args, xp)
            diag["step_hint"] = int(step)
            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_diag = diag
                best_state = state_after
                best_graded_after = graded_after

        n_this_eval = {k: 0 for k in move_counts.keys()}

        if best_diag is not None and best_delta > 0.0 and best_graded_after is not None:
            psi, sigma, interface_commitment = best_graded_after
            psi_eff, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = best_state  # type: ignore[misc]
            psi = psi_eff
            accepted_moves.append(best_diag)
            move_counts[best_diag["move_type"]] += 1
            n_this_eval[best_diag["move_type"]] += 1

        state_now = materialize_state(psi, sigma, interface_commitment, pcfg, args.sigma_on_threshold, args.edge_on_threshold)
        psi_now, active_nodes_now, dormant_nodes_now, active_edges_now, local_coeffs_now, edge_strengths_now, link_regs_now = state_now
        core_now = bk.dominant_core_snapshot(
            psi_now, active_nodes_now, active_edges_now, edge_strengths_now, GM_MATRICES, xp, pcfg.n_max, scfg, link_regs_now
        )
        metrics = phys.metric_snapshot(active_nodes_now, active_edges_now, edge_strengths_now)
        gstats = phys.graph_stats(active_nodes_now, active_edges_now)

        snapshots.append(
            {
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
                "candidate_move_diagnostics": move_diags,
            }
        )

        active_trace.append(len(active_nodes_now))
        edge_trace.append(len(active_edges_now))
        sigma_mean_trace.append(float(np.mean(sigma)))

        if pcfg.progress_every > 0 and len(snapshots) % pcfg.progress_every == 0:
            cp = core_now["core_pair"] if core_now else None
            winner = best_diag["move_type"] if best_diag is not None and best_delta > 0.0 else None
            print(
                f"[eval {len(snapshots):03d}] step={step:4d} active={len(active_nodes_now):2d} "
                f"edges={len(active_edges_now):3d} mean_sigma={np.mean(sigma):.3f} core={cp} "
                f"raise={n_this_eval['raise_support']} lower={n_this_eval['lower_support']} "
                f"edge_up={n_this_eval['edge_up']} edge_down={n_this_eval['edge_down']} "
                f"winner={winner} cand={len(move_diags)}"
            )

    return {
        "script": "hsf_mesoscape_graded_support_sandbox.py",
        "physics_config": asdict(pcfg),
        "bookkeeping_config": asdict(scfg),
        "graded_support_config": {
            "sigma_on_threshold": float(args.sigma_on_threshold),
            "edge_on_threshold": float(args.edge_on_threshold),
            "sigma_step": float(args.sigma_step),
            "edge_step": float(args.edge_step),
            "max_raise_candidates_per_child": int(args.max_raise_candidates_per_child),
            "w_graded_odiff": float(args.w_graded_odiff),
            "w_graded_relief": float(args.w_graded_relief),
            "w_graded_distinct": float(args.w_graded_distinct),
            "w_graded_redundancy": float(args.w_graded_redundancy),
            "w_graded_persistence": float(args.w_graded_persistence),
            "graded_odiff_clip": float(args.graded_odiff_clip),
            "edge_up_odiff_min": float(args.edge_up_odiff_min),
            "edge_up_relief_min": float(args.edge_up_relief_min),
            "edge_up_distinct_min": float(args.edge_up_distinct_min),
            "edge_up_hard_gate_negative_raw": bool(args.edge_up_hard_gate_negative_raw),
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