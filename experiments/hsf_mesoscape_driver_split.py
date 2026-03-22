#!/usr/bin/env python3
# filename: hsf_mesoscape_driver_split.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import hsf_mesoscale_bookkeeping as bk
import hsf_mesoscale_physics_core as phys
from hsf_mesoscale_physics_core import PhysicsConfig, GM_MATRICES, canonical_edge


Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HSF mesoscape mainline driver with proto-child integration. "
            "Adds proto as an explicit candidate move class alongside birth, weaken, and retire."
        )
    )

    # Physics core args
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

    # Existing bookkeeping / lawful move args
    p.add_argument("--lambda-B", type=float, default=0.18)
    p.add_argument("--lambda-S", type=float, default=0.12)
    p.add_argument("--lambda-F", type=float, default=0.20)
    p.add_argument("--lambda-R", type=float, default=0.35)
    p.add_argument("--w-mi", type=float, default=1.0)
    p.add_argument("--w-corr", type=float, default=0.5)
    p.add_argument("--w-link", type=float, default=0.5)
    p.add_argument("--retirement-threshold", type=float, default=0.66)
    p.add_argument("--organizer-large-region-cutoff", type=int, default=6)

    # Proto integration args
    p.add_argument(
        "--enable-proto",
        action="store_true",
        help="Enable proto-child candidate generation and scoring.",
    )
    p.add_argument(
        "--proto-local-alpha",
        type=float,
        default=0.45,
        help="Initial local coefficient scale for proto-child.",
    )
    p.add_argument(
        "--proto-edge-alpha",
        type=float,
        default=0.65,
        help="Initial one-sided proto edge scale.",
    )
    p.add_argument(
        "--proto-bonus",
        type=float,
        default=0.0,
        help="Optional additive deltaF bonus for accepted proto candidates.",
    )
    p.add_argument(
        "--proto-admission-threshold",
        type=float,
        default=0.50,
        help="Minimum proto admission witness score required to consider a proto move.",
    )
    p.add_argument(
        "--proto-readiness-weight",
        type=float,
        default=0.10,
        help="Additive contribution of proto-readiness to dE_expr.",
    )
    p.add_argument(
        "--proto-promotion-bonus",
        type=float,
        default=0.0,
        help="Optional bonus when a full birth acts as promotion of an existing proto-child.",
    )
    p.add_argument(
        "--proto-primary-parent-policy",
        choices=["lower_index", "higher_relief_bias"],
        default="lower_index",
        help="How to pick the one-sided parent for proto-child attachment.",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="hsf_mesoscape_mainline_proto.json",
    )
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


def choose_proto_primary_parent(
    parents: Edge,
    before_state,
    cfg: bk.ScoreConfig,
    xp,
    policy: str,
) -> int:
    i, j = canonical_edge(*parents)
    if policy == "lower_index":
        return i

    # best-effort asymmetry proxy: compare single-sided proto readiness from each parent
    psi_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = before_state
    try:
        n_sites = int(local_coeffs_b.shape[0])
        core_before = bk.dominant_core_snapshot(
            psi_b, active_nodes_b, active_edges_b, edge_strengths_b, GM_MATRICES, xp, n_sites, cfg, link_regs_b
        )
        # If the dominant core contains one of the parents, bias to the non-core side only if
        # both are not equally represented. Otherwise fall back to lower index.
        cp = core_before.get("core_pair")
        if isinstance(cp, list) and len(cp) == 2:
            cp_set = {int(cp[0]), int(cp[1])}
            if i in cp_set and j not in cp_set:
                return j
            if j in cp_set and i not in cp_set:
                return i
    except Exception:
        pass

    return i


def prepare_proto_move(
    state,
    parents: Edge,
    child: int,
    pcfg: PhysicsConfig,
    args: argparse.Namespace,
    primary_parent: int,
):
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = phys.clone_graph_state(*state)
    i, j = canonical_edge(*parents)
    secondary_parent = j if int(primary_parent) == i else i

    local_alpha = max(0.0, float(args.proto_local_alpha))
    edge_alpha = max(0.0, float(args.proto_edge_alpha))

    active_nodes.add(int(child))
    dormant_nodes.discard(int(child))
    local_coeffs[int(child)] = float(pcfg.spawn_pair_scale) * local_alpha

    proto_edge = canonical_edge(int(primary_parent), int(child))
    active_edges.add(proto_edge)

    base_strength = max(float(pcfg.spawn_pair_scale), float(pcfg.pair_scale))
    edge_strengths[proto_edge] = float(base_strength * edge_alpha)
    link_regs[proto_edge] = phys.default_linkreg().copy()

    phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)

    meta = {
        "parent_pair": [int(i), int(j)],
        "child": int(child),
        "primary_parent": int(primary_parent),
        "secondary_parent": int(secondary_parent),
        "proto_edge": [int(proto_edge[0]), int(proto_edge[1])],
        "proto_local_alpha": float(local_alpha),
        "proto_edge_alpha": float(edge_alpha),
    }
    return (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs), meta


def proto_admission_witness(
    obj: Dict[str, Any],
    before_state,
    after_state,
    cfg: bk.ScoreConfig,
    xp,
) -> Dict[str, float]:
    """
    First-pass proto admission witness:
    J_proto = vR*R_parent + vO*max(0, delta_Odiff) + vA*A_asym + vP*P_partial

    We do not yet have a fully derived asymmetry/partial-demand theorem, so:
    - R_parent comes from birth justification
    - delta_Odiff comes from local differentiated-role occupancy audit
    - A_asym is a bounded heuristic favoring one-sided support if full birth tends to overshoot
    - P_partial is a bounded heuristic from low initial child support + positive local gain
    """
    psi_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = before_state
    n_sites = int(local_coeffs_b.shape[0])

    core_before = bk.dominant_core_snapshot(
        psi_b, active_nodes_b, active_edges_b, edge_strengths_b, GM_MATRICES, xp, n_sites, cfg, link_regs_b
    )

    birth_info = bk.birth_justification_witness(
        {"parents": tuple(obj["parent_pair"]), "child": int(obj["child"])},
        before_state,
        after_state,
        cfg,
        GM_MATRICES,
        xp,
    )

    delta_odiff = 0.0
    asym_bias = 0.0
    partial_demand = 0.0

    if cfg.odiff_enabled:
        odiff_audit = bk.compute_move_local_odiff_audit(
            "birth",
            {"parents": tuple(obj["parent_pair"]), "child": int(obj["child"])},
            before_state,
            after_state,
            core_before,
            cfg,
        )
        delta_odiff = max(0.0, float(odiff_audit.delta_odiff))

    # Heuristic asymmetry proxy:
    # one-sided support is favored when only one new edge is added and the child local amplitude is reduced.
    asym_bias = 1.0

    # Heuristic partial-demand proxy:
    # if parent relief is real and the proto child is intentionally sub-full, count this as partial support demand.
    partial_demand = float(
        np.clip(
            0.5 * float(birth_info.get("birth_parent_relief", 0.0))
            + 0.5 * float(birth_info.get("birth_novelty", 0.0)),
            0.0,
            1.0,
        )
    )

    vR, vO, vA, vP = 0.35, 0.30, 0.15, 0.20
    j_proto = (
        vR * float(birth_info.get("birth_parent_relief", 0.0))
        + vO * float(delta_odiff)
        + vA * float(asym_bias)
        + vP * float(partial_demand)
    )

    return {
        "J_proto": float(j_proto),
        "R_parent": float(birth_info.get("birth_parent_relief", 0.0)),
        "delta_Odiff_proto": float(delta_odiff),
        "A_asym": float(asym_bias),
        "P_partial": float(partial_demand),
        "birth_novelty": float(birth_info.get("birth_novelty", 0.0)),
        "birth_justification": float(birth_info.get("birth_justification", 0.0)),
    }


def score_proto_move(
    obj: Dict[str, Any],
    before_state,
    after_state,
    cfg: bk.ScoreConfig,
    xp,
    args: argparse.Namespace,
):
    """
    Proto scoring:
    - birth-like local differentiated-role occupancy audit remains active
    - full-birth justification is NOT multiplied in as a full-birth event
    - instead use bounded proto-readiness bonus
    """
    psi_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = before_state
    psi_a, active_nodes_a, dormant_nodes_a, active_edges_a, local_coeffs_a, edge_strengths_a, link_regs_a = after_state
    n_sites = int(local_coeffs_b.shape[0])

    core_before = bk.dominant_core_snapshot(
        psi_b, active_nodes_b, active_edges_b, edge_strengths_b, GM_MATRICES, xp, n_sites, cfg, link_regs_b
    )

    expr_b = bk.local_expression(psi_b, active_nodes_b, active_edges_b, edge_strengths_b, link_regs_b, cfg, GM_MATRICES, xp, n_sites)
    expr_a = bk.local_expression(psi_a, active_nodes_a, active_edges_a, edge_strengths_a, link_regs_a, cfg, GM_MATRICES, xp, n_sites)
    dE_raw = float(expr_a - expr_b)
    dE = float(dE_raw)

    birth_info = bk.birth_justification_witness(
        {"parents": tuple(obj["parent_pair"]), "child": int(obj["child"])},
        before_state,
        after_state,
        cfg,
        GM_MATRICES,
        xp,
    )

    proto_readiness = (
        0.45 * float(birth_info.get("birth_novelty", 0.0))
        + 0.40 * float(birth_info.get("birth_parent_relief", 0.0))
        + 0.15 * float(birth_info.get("birth_distinctness", 0.0) if "birth_distinctness" in birth_info else 0.0)
    )
    proto_readiness = float(np.clip(proto_readiness, 0.0, 1.0))
    dE += float(args.proto_readiness_weight) * proto_readiness

    cb_b = bk.bandwidth_burden(active_edges_b, link_regs_b)
    cb_a = bk.bandwidth_burden(active_edges_a, link_regs_a)
    dCB = float(cb_a - cb_b)

    cs_b = bk.spread_burden(active_nodes_b, active_edges_b)
    cs_a = bk.spread_burden(active_nodes_a, active_edges_a)
    dCS = float(cs_a - cs_b)

    nr = bk.no_refolding_witness(
        "proto",
        obj,
        psi_b,
        psi_a,
        active_nodes_b,
        active_edges_b,
        edge_strengths_b,
        active_nodes_a,
        active_edges_a,
        edge_strengths_a,
        core_before,
        cfg,
        GM_MATRICES,
        xp,
        n_sites,
    )
    dCF = float(max(0.0, 1.0 - nr["F_org"]) + max(0.0, nr["W_func"]))

    odiff_expr_adj = 0.0
    odiff_info = {
        "expr_odiff_delta_raw": 0.0,
        "expr_odiff_delta_clipped": 0.0,
        "expr_odiff_adjustment": 0.0,
        "expr_odiff_applied": False,
    }
    odiff_audit = None
    if cfg.odiff_enabled:
        odiff_audit = bk.compute_move_local_odiff_audit(
            "birth",
            {"parents": tuple(obj["parent_pair"]), "child": int(obj["child"])},
            before_state,
            after_state,
            core_before,
            cfg,
        )
        odiff_expr_adj, base_info = bk._bounded_odiff_expr_adjustment(odiff_audit, cfg)
        dE = float(dE + odiff_expr_adj)
        odiff_info.update(base_info)
        odiff_info["expr_odiff_applied"] = True

    deltaF = dE - cfg.lambda_B * dCB - cfg.lambda_S * dCS - cfg.lambda_F * dCF - cfg.lambda_R * nr["W_NR"]

    diag = {
        "move_type": "proto",
        "move_object": {
            "parent_pair": [int(x) for x in obj["parent_pair"]],
            "child": int(obj["child"]),
            "primary_parent": int(obj["primary_parent"]),
            "secondary_parent": int(obj["secondary_parent"]),
            "proto_edge": [int(x) for x in obj["proto_edge"]],
        },
        "deltaF": float(deltaF),
        "dE_expr": float(dE),
        "dE_expr_raw": float(dE_raw),
        "dE_expr_structural_adjustment": 0.0,
        "dE_expr_odiff_adjustment": float(odiff_expr_adj),
        "dE_expr_proto_readiness": float(args.proto_readiness_weight) * float(proto_readiness),
        "proto_readiness": float(proto_readiness),
        "dCB": float(dCB),
        "dCS": float(dCS),
        "dCF": float(dCF),
        "core_before": core_before,
        **nr,
        **birth_info,
        **odiff_info,
    }

    if odiff_audit is not None:
        diag["odiff_audit"] = asdict(odiff_audit)
        diag["A_before_R"] = int(odiff_audit.active_count_before)
        diag["A_after_R"] = int(odiff_audit.active_count_after)
        diag["Odiff_before_R"] = float(odiff_audit.odiff_before)
        diag["Odiff_after_R"] = float(odiff_audit.odiff_after)
        diag["delta_Odiff_R"] = float(odiff_audit.delta_odiff)

    return float(deltaF), diag


def proto_identity_witness(proto_record: Dict[str, Any], state, cfg: bk.ScoreConfig, xp) -> Dict[str, float]:
    """
    First-pass persistent-local-mode witness.
    Function-first and light:
    - child still active
    - proto edge still active
    - local coefficient still nonzero
    """
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = state
    child = int(proto_record["child"])
    proto_edge = canonical_edge(*proto_record["proto_edge"])
    child_active = float(child in active_nodes)
    edge_active = float(proto_edge in active_edges)
    local_amp = float(abs(local_coeffs[child])) if 0 <= child < len(local_coeffs) else 0.0
    amp_score = float(np.clip(local_amp / max(1e-12, float(proto_record["proto_local_amp_ref"])), 0.0, 1.0))
    persistence = 0.40 * child_active + 0.35 * edge_active + 0.25 * amp_score
    return {
        "proto_persistence": float(np.clip(persistence, 0.0, 1.0)),
        "child_active": float(child_active),
        "proto_edge_active": float(edge_active),
        "proto_amp_score": float(amp_score),
    }


def proto_promotion_witness(
    proto_record: Dict[str, Any],
    candidate_birth_parents: Edge,
    candidate_birth_child: int,
    before_state,
    after_state_full_birth,
    cfg: bk.ScoreConfig,
    xp,
) -> Dict[str, float]:
    """
    Promotion asks whether adding the missing second edge now looks lawful and not overshooting.
    """
    target_child = int(proto_record["child"])
    if int(candidate_birth_child) != target_child:
        return {
            "promotion_match": 0.0,
            "promotion_eligible": 0.0,
            "promotion_second_edge_gain": 0.0,
            "promotion_overshoot_risk": 1.0,
        }

    pp_target = canonical_edge(*proto_record["parent_pair"])
    pp_birth = canonical_edge(*candidate_birth_parents)
    if pp_birth != pp_target:
        return {
            "promotion_match": 0.0,
            "promotion_eligible": 0.0,
            "promotion_second_edge_gain": 0.0,
            "promotion_overshoot_risk": 1.0,
        }

    psi_b, active_nodes_b, dormant_nodes_b, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = before_state
    n_sites = int(local_coeffs_b.shape[0])
    core_before = bk.dominant_core_snapshot(
        psi_b, active_nodes_b, active_edges_b, edge_strengths_b, GM_MATRICES, xp, n_sites, cfg, link_regs_b
    )
    odiff_audit = bk.compute_move_local_odiff_audit(
        "birth",
        {"parents": tuple(candidate_birth_parents), "child": int(candidate_birth_child)},
        before_state,
        after_state_full_birth,
        core_before,
        cfg,
    )
    delta_gain = float(max(0.0, odiff_audit.delta_odiff))
    overshoot_risk = float(np.clip(-min(0.0, odiff_audit.delta_odiff), 0.0, 1.0))
    eligible = float(delta_gain > 0.02 and overshoot_risk < 0.20)
    return {
        "promotion_match": 1.0,
        "promotion_eligible": eligible,
        "promotion_second_edge_gain": delta_gain,
        "promotion_overshoot_risk": overshoot_risk,
    }


def proto_reexpression_witness(proto_record: Dict[str, Any], weaken_diag: Dict[str, Any]) -> Dict[str, float]:
    """
    If weaken is winning while proto persists, that is exactly the re-expression pathway
    the probe identified. Treat lawful shell re-expression as a positive sign.
    """
    lawful = float(bool(weaken_diag.get("lawful_shell_reexpression", False)))
    shell = float(bool(weaken_diag.get("shell_weaken", False)))
    edge = weaken_diag.get("move_object")
    proto_edge = proto_record.get("proto_edge")
    child = int(proto_record["child"])

    edge_rel = 0.0
    if isinstance(edge, list) and len(edge) == 2:
        e = canonical_edge(int(edge[0]), int(edge[1]))
        if child in e:
            edge_rel = 1.0
        elif proto_edge is not None:
            pe = canonical_edge(int(proto_edge[0]), int(proto_edge[1]))
            if pe[0] in e or pe[1] in e:
                edge_rel = 0.5

    score = 0.50 * lawful + 0.30 * shell + 0.20 * edge_rel
    return {
        "proto_reexpression_score": float(np.clip(score, 0.0, 1.0)),
        "proto_reexpression_eligible": float(score >= 0.50),
    }


def maybe_register_proto(proto_registry: Dict[int, Dict[str, Any]], best_diag: Dict[str, Any], state) -> None:
    if best_diag.get("move_type") != "proto":
        return
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = state
    mo = best_diag.get("move_object", {}) or {}
    child = int(mo["child"])
    proto_registry[child] = {
        "parent_pair": list(mo["parent_pair"]),
        "child": child,
        "primary_parent": int(mo["primary_parent"]),
        "secondary_parent": int(mo["secondary_parent"]),
        "proto_edge": list(mo["proto_edge"]),
        "proto_state": "proto",
        "proto_birth_step": int(best_diag.get("step_hint", -1)),
        "proto_local_amp_ref": float(abs(local_coeffs[child])) if 0 <= child < len(local_coeffs) else 1.0,
        "proto_last_event_step": int(best_diag.get("step_hint", -1)),
        "proto_last_deltaF": float(best_diag.get("deltaF", 0.0)),
    }


def update_proto_registry_after_move(
    proto_registry: Dict[int, Dict[str, Any]],
    best_diag: Optional[Dict[str, Any]],
    state,
) -> None:
    if best_diag is None:
        return

    move_type = best_diag.get("move_type")
    if move_type == "proto":
        maybe_register_proto(proto_registry, best_diag, state)
        return

    if move_type == "retire":
        child = int(best_diag.get("move_object"))
        if child in proto_registry:
            proto_registry[child]["proto_state"] = "retired"
            del proto_registry[child]
        return

    if move_type == "birth" and bool(best_diag.get("proto_promotion_used", False)):
        mo = best_diag.get("move_object", {}) or {}
        child = int(mo["child"])
        if child in proto_registry:
            proto_registry[child]["proto_state"] = "full"
            proto_registry[child]["proto_last_event_step"] = int(best_diag.get("step_hint", -1))
            proto_registry[child]["proto_last_deltaF"] = float(best_diag.get("deltaF", 0.0))
        return

    if move_type == "weaken":
        for child, rec in proto_registry.items():
            rew = proto_reexpression_witness(rec, best_diag)
            if rew["proto_reexpression_eligible"] > 0.5:
                rec["proto_state"] = "reexpressed"
                rec["proto_last_event_step"] = int(best_diag.get("step_hint", -1))
                rec["proto_last_deltaF"] = float(best_diag.get("deltaF", 0.0))


def serialize_proto_registry(proto_registry: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for child in sorted(proto_registry.keys()):
        out.append(dict(proto_registry[child]))
    return out


def run_sim(pcfg: PhysicsConfig, scfg: bk.ScoreConfig, args: argparse.Namespace):
    xp, is_gpu = phys.get_array_module(pcfg.device)
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs, rng = phys.init_state(pcfg, xp)

    accepted_moves: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []
    active_trace: List[int] = []
    edge_trace: List[int] = []

    proto_registry: Dict[int, Dict[str, Any]] = {}

    n_proto_events = 0
    n_birth_events = 0
    n_promotion_events = 0
    n_weaken_events = 0
    n_extinction_events = 0

    for step in range(pcfg.eval_every, pcfg.total_steps + 1, pcfg.eval_every):
        base_state = (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs)
        core_before = bk.dominant_core_snapshot(
            psi, active_nodes, active_edges, edge_strengths, GM_MATRICES, xp, pcfg.n_max
        )

        birth_moves = bk.candidate_births(active_nodes, dormant_nodes, active_edges)
        weaken_moves = bk.candidate_weakens(active_edges)
        retire_moves, retire_info = bk.candidate_retirements(
            psi, active_nodes, active_edges, edge_strengths, link_regs, core_before, scfg, GM_MATRICES, xp, pcfg.n_max
        )

        move_diags: List[Dict[str, Any]] = []
        best_kind: Optional[str] = None
        best_state = None
        best_diag: Optional[Dict[str, Any]] = None
        best_delta = 0.0

        # Proto candidates
        if args.enable_proto:
            for parents, child in birth_moves:
                primary_parent = choose_proto_primary_parent(
                    parents=parents,
                    before_state=base_state,
                    cfg=scfg,
                    xp=xp,
                    policy=args.proto_primary_parent_policy,
                )
                moved, meta = prepare_proto_move(base_state, parents, child, pcfg, args, primary_parent)
                moved = phys.evolve_prepared_state(moved, pcfg, xp)

                obj = {
                    "parent_pair": canonical_edge(*parents),
                    "child": int(child),
                    "primary_parent": int(meta["primary_parent"]),
                    "secondary_parent": int(meta["secondary_parent"]),
                    "proto_edge": tuple(meta["proto_edge"]),
                }
                adm = proto_admission_witness(obj, base_state, moved, scfg, xp)
                if float(adm["J_proto"]) < float(args.proto_admission_threshold):
                    continue

                dF, diag = score_proto_move(obj, base_state, moved, scfg, xp, args)
                if float(args.proto_bonus) != 0.0:
                    dF = float(dF + float(args.proto_bonus))

                diag["proto_variant"] = "mainline_proto"
                diag["proto_used"] = True
                diag["step_hint"] = int(step)
                diag["deltaF_after_proto_bonus"] = float(dF)
                diag.update(meta)
                diag.update(adm)

                move_diags.append(diag)
                if dF > best_delta:
                    best_delta = float(dF)
                    best_kind = "proto"
                    best_state = moved
                    best_diag = diag

        # Full birth candidates, with proto-promotion awareness
        for parents, child in birth_moves:
            moved = phys.prepare_birth_move(base_state, parents, child, pcfg)
            moved = phys.evolve_prepared_state(moved, pcfg, xp)

            dF, diag = bk.score_move("birth", {"parents": parents, "child": child}, base_state, moved, scfg, GM_MATRICES, xp)
            diag["move_object"] = {"parents": list(parents), "child": int(child)}
            diag["birth_variant"] = "full_birth"
            diag["step_hint"] = int(step)

            proto_promotion_used = False
            promotion_info = {
                "promotion_match": 0.0,
                "promotion_eligible": 0.0,
                "promotion_second_edge_gain": 0.0,
                "promotion_overshoot_risk": 1.0,
            }
            if int(child) in proto_registry:
                promotion_info = proto_promotion_witness(
                    proto_record=proto_registry[int(child)],
                    candidate_birth_parents=parents,
                    candidate_birth_child=int(child),
                    before_state=base_state,
                    after_state_full_birth=moved,
                    cfg=scfg,
                    xp=xp,
                )
                if promotion_info["promotion_eligible"] > 0.5:
                    proto_promotion_used = True
                    diag["birth_variant"] = "proto_promotion"
                    if float(args.proto_promotion_bonus) != 0.0:
                        dF = float(dF + float(args.proto_promotion_bonus))

            diag["proto_promotion_used"] = bool(proto_promotion_used)
            diag["deltaF_after_proto_promotion_bonus"] = float(dF)
            diag.update(promotion_info)

            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_kind = "birth"
                best_state = moved
                best_diag = diag

        # Weaken
        for edge in weaken_moves:
            moved = phys.prepare_weaken_move(base_state, edge, pcfg)
            if moved is None:
                continue
            moved = phys.evolve_prepared_state(moved, pcfg, xp)
            dF, diag = bk.score_move("weaken", edge, base_state, moved, scfg, GM_MATRICES, xp)
            diag["move_object"] = list(edge)
            diag["step_hint"] = int(step)

            # add current proto reexpression hints for audit
            proto_reexpression_rows = []
            for child, rec in proto_registry.items():
                rew = proto_reexpression_witness(rec, diag)
                proto_reexpression_rows.append(
                    {
                        "child": int(child),
                        "proto_state": rec.get("proto_state"),
                        **rew,
                    }
                )
            diag["proto_reexpression_audit"] = proto_reexpression_rows

            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_kind = "weaken"
                best_state = moved
                best_diag = diag

        # Retire
        for node in retire_moves:
            moved = phys.prepare_retire_move(base_state, node, pcfg)
            if moved is None:
                continue
            moved = phys.evolve_prepared_state(moved, pcfg, xp)
            dF, diag = bk.score_move("retire", node, base_state, moved, scfg, GM_MATRICES, xp)
            diag["move_object"] = int(node)
            diag["retirement_info"] = retire_info[node]
            diag["step_hint"] = int(step)

            if int(node) in proto_registry:
                diag["retire_variant"] = "proto_retirement"
            else:
                diag["retire_variant"] = "ordinary_retirement"

            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_kind = "retire"
                best_state = moved
                best_diag = diag

        n_proto_this_eval = 0
        n_births_this_eval = 0
        n_promotions_this_eval = 0
        n_weakens_this_eval = 0
        n_extinctions_this_eval = 0

        if best_kind is not None and best_delta > 0.0 and best_state is not None and best_diag is not None:
            psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = best_state
            accepted_moves.append(best_diag)

            if best_kind == "proto":
                n_proto_events += 1
                n_proto_this_eval = 1
            elif best_kind == "birth":
                n_birth_events += 1
                n_births_this_eval = 1
                if bool(best_diag.get("proto_promotion_used", False)):
                    n_promotion_events += 1
                    n_promotions_this_eval = 1
            elif best_kind == "weaken":
                n_weaken_events += 1
                n_weakens_this_eval = 1
            elif best_kind == "retire":
                n_extinction_events += 1
                n_extinctions_this_eval = 1

            update_proto_registry_after_move(proto_registry, best_diag, (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs))

        phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
        core_after = bk.dominant_core_snapshot(
            psi, active_nodes, active_edges, edge_strengths, GM_MATRICES, xp, pcfg.n_max
        )
        metrics = phys.metric_snapshot(active_nodes, active_edges, edge_strengths)
        gstats = phys.graph_stats(active_nodes, active_edges)
        mean_link_rank = 0.0
        if active_edges:
            mean_link_rank = float(np.mean([bk.bounded_activity_and_rank(link_regs[e])[1] for e in active_edges]))

        proto_status_rows = []
        current_state = (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs)
        for child, rec in proto_registry.items():
            iw = proto_identity_witness(rec, current_state, scfg, xp)
            rec["proto_persistence"] = float(iw["proto_persistence"])
            proto_status_rows.append(
                {
                    "child": int(child),
                    "parent_pair": list(rec["parent_pair"]),
                    "proto_state": rec.get("proto_state"),
                    **iw,
                }
            )

        snapshots.append(
            {
                "step": int(step),
                "active_nodes": list(sorted(active_nodes)),
                "dormant_nodes": list(sorted(dormant_nodes)),
                "active_edges": [list(e) for e in sorted(active_edges)],
                "active_edge_count": int(len(active_edges)),
                "dominant_core": core_after,
                "metric": metrics,
                "graph_stats": gstats,
                "birth_candidates_this_window": int(len(birth_moves)),
                "retirement_candidate_count": int(len(retire_moves)),
                "retirement_candidates": [int(n) for n in retire_moves],
                "mean_link_rank": float(mean_link_rank),
                "n_proto_this_eval": int(n_proto_this_eval),
                "n_births_this_eval": int(n_births_this_eval),
                "n_promotions_this_eval": int(n_promotions_this_eval),
                "n_weakens_this_eval": int(n_weakens_this_eval),
                "n_extinctions_this_eval": int(n_extinctions_this_eval),
                "proto_status": proto_status_rows,
                "candidate_move_diagnostics": move_diags,
            }
        )

        active_trace.append(len(active_nodes))
        edge_trace.append(len(active_edges))

        if pcfg.progress_every > 0 and len(snapshots) % pcfg.progress_every == 0:
            cp = core_after["core_pair"] if core_after else None
            variant = None
            if best_diag is not None:
                if best_diag.get("move_type") == "birth":
                    variant = best_diag.get("birth_variant")
                elif best_diag.get("move_type") == "proto":
                    variant = best_diag.get("proto_variant")
                elif best_diag.get("move_type") == "retire":
                    variant = best_diag.get("retire_variant")
            print(
                f"[eval {len(snapshots):03d}] step={step:4d} active={len(active_nodes):2d} "
                f"dormant={len(dormant_nodes):2d} edges={len(active_edges):3d} core={cp} "
                f"proto={n_proto_this_eval} births={n_births_this_eval} promo={n_promotions_this_eval} "
                f"weakens={n_weakens_this_eval} extinctions={n_extinctions_this_eval} "
                f"winner={best_diag.get('move_type') if best_diag else None} variant={variant} "
                f"cand={len(move_diags)}"
            )

    return {
        "script": "hsf_mesoscape_driver_split.py",
        "physics_config": asdict(pcfg),
        "bookkeeping_config": asdict(scfg),
        "proto_config": {
            "enabled": bool(args.enable_proto),
            "proto_local_alpha": float(args.proto_local_alpha),
            "proto_edge_alpha": float(args.proto_edge_alpha),
            "proto_bonus": float(args.proto_bonus),
            "proto_admission_threshold": float(args.proto_admission_threshold),
            "proto_readiness_weight": float(args.proto_readiness_weight),
            "proto_promotion_bonus": float(args.proto_promotion_bonus),
            "proto_primary_parent_policy": str(args.proto_primary_parent_policy),
        },
        "n_proto_events": int(n_proto_events),
        "n_birth_events": int(n_birth_events),
        "n_promotion_events": int(n_promotion_events),
        "n_weaken_events": int(n_weaken_events),
        "n_extinction_events": int(n_extinction_events),
        "final_active_nodes": list(sorted(active_nodes)),
        "final_dormant_nodes": list(sorted(dormant_nodes)),
        "final_active_edges": [list(e) for e in sorted(active_edges)],
        "final_proto_registry": serialize_proto_registry(proto_registry),
        "active_count_trace": active_trace,
        "edge_count_trace": edge_trace,
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