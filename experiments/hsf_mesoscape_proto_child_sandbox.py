#!/usr/bin/env python3
# filename: hsf_mesoscape_proto_child_sandbox.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np

import hsf_mesoscale_bookkeeping as bk
import hsf_mesoscale_physics_core as phys
from hsf_mesoscale_physics_core import PhysicsConfig, GM_MATRICES, canonical_edge


Edge = Tuple[int, int]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "HSF mesoscale proto-child sandbox. "
            "Tests a semi-instantiated, one-sided child carrier: a new child is created with "
            "reduced local amplitude and attached to only one parent initially."
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
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lookahead-windows", type=int, default=1)
    p.add_argument("--weaken-factor", type=float, default=0.55)
    p.add_argument("--progress-every", type=int, default=1)

    p.add_argument("--lambda-B", type=float, default=0.18)
    p.add_argument("--lambda-S", type=float, default=0.12)
    p.add_argument("--lambda-F", type=float, default=0.20)
    p.add_argument("--lambda-R", type=float, default=0.35)
    p.add_argument("--w-mi", type=float, default=1.0)
    p.add_argument("--w-corr", type=float, default=0.5)
    p.add_argument("--w-link", type=float, default=0.5)
    p.add_argument("--retirement-threshold", type=float, default=0.66)
    p.add_argument("--organizer-large-region-cutoff", type=int, default=6)

    # Proto-child controls.
    p.add_argument(
        "--proto-child",
        action="store_true",
        help="Enable the proto-child candidate move for the targeted niche.",
    )
    p.add_argument(
        "--proto-parent-pair",
        nargs=2,
        type=int,
        default=None,
        help="Target parent pair for proto-child creation, e.g. --proto-parent-pair 2 3",
    )
    p.add_argument(
        "--proto-child-node",
        type=int,
        default=None,
        help="Target child node for proto-child creation, e.g. --proto-child-node 5",
    )
    p.add_argument(
        "--proto-primary-parent",
        type=int,
        default=None,
        help="Parent that gets the initial one-sided attachment. Default: lower-index parent.",
    )
    p.add_argument(
        "--proto-local-alpha",
        type=float,
        default=0.45,
        help="Scale factor for newborn local coefficient amplitude.",
    )
    p.add_argument(
        "--proto-edge-alpha",
        type=float,
        default=0.65,
        help="Scale factor for the single initial parent-child edge.",
    )
    p.add_argument(
        "--proto-bonus",
        type=float,
        default=0.0,
        help="Optional additive bonus to deltaF for proto-child moves.",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="hsf_mesoscape_proto_child_sandbox.json",
    )
    return p.parse_args()


def build_configs(args):
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


def proto_matches(args, parents: Edge, child: int) -> bool:
    if not bool(args.proto_child):
        return False

    pp = canonical_edge(*parents)
    target_pp = None
    if args.proto_parent_pair is not None:
        target_pp = canonical_edge(int(args.proto_parent_pair[0]), int(args.proto_parent_pair[1]))

    target_child = None if args.proto_child_node is None else int(args.proto_child_node)

    if target_pp is None and target_child is None:
        return True
    if target_pp is not None and pp != target_pp:
        return False
    if target_child is not None and int(child) != target_child:
        return False
    return True


def choose_primary_parent(parents: Edge, requested_primary: Optional[int]) -> int:
    i, j = canonical_edge(*parents)
    if requested_primary is None:
        return i
    rp = int(requested_primary)
    return rp if rp in (i, j) else i


def prepare_proto_child_move(
    state,
    parents: Edge,
    child: int,
    pcfg: PhysicsConfig,
    args,
):
    """
    Semi-instantiated child carrier:
    - child becomes active
    - child gets reduced local coefficient amplitude
    - child attaches to only one parent initially
    - second parent edge is NOT created on this move
    """
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = phys.clone_graph_state(*state)
    i, j = canonical_edge(*parents)

    primary_parent = choose_primary_parent((i, j), args.proto_primary_parent)
    secondary_parent = j if primary_parent == i else i

    local_alpha = max(0.0, float(args.proto_local_alpha))
    edge_alpha = max(0.0, float(args.proto_edge_alpha))

    active_nodes.add(int(child))
    dormant_nodes.discard(int(child))

    local_coeffs[int(child)] = float(pcfg.spawn_pair_scale) * local_alpha

    e_primary = canonical_edge(primary_parent, int(child))
    active_edges.add(e_primary)

    base_strength = max(float(pcfg.spawn_pair_scale), float(pcfg.pair_scale))
    edge_strengths[e_primary] = float(base_strength * edge_alpha)
    link_regs[e_primary] = phys.default_linkreg().copy()

    phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)

    meta = {
        "proto_parent_pair": [int(i), int(j)],
        "primary_parent": int(primary_parent),
        "secondary_parent": int(secondary_parent),
        "proto_edge": list(e_primary),
        "proto_local_alpha": float(local_alpha),
        "proto_edge_alpha": float(edge_alpha),
    }
    return (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs), meta


def score_proto_child_move(
    obj,
    before_state,
    after_state,
    cfg: bk.ScoreConfig,
    xp,
):
    """
    Custom sandbox scoring for proto-child moves.

    Treated like a birth-like local event for local-region auditing, but without
    multiplying dE_raw by the normal full-birth justification witness. The point
    is to see whether a semi-committed carrier can win on its own merits.
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

    # Proto-child compromise:
    # keep birth-side relief/novelty visibility, but do not force the full-birth
    # multiplier onto dE. Instead add a modest bounded proto-readiness bonus.
    proto_readiness = (
        0.45 * float(birth_info.get("birth_novelty", 0.0))
        + 0.40 * float(birth_info.get("birth_parent_relief", 0.0))
        + 0.15 * float(birth_info.get("birth_distinctness", 0.0) if "birth_distinctness" in birth_info else 0.0)
    )
    proto_readiness = float(np.clip(proto_readiness, 0.0, 1.0))
    dE = float(dE + 0.10 * proto_readiness)

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

    odiff_audit = None
    odiff_expr_adj = 0.0
    odiff_expr_info = {
        "expr_odiff_delta_raw": 0.0,
        "expr_odiff_delta_clipped": 0.0,
        "expr_odiff_adjustment": 0.0,
        "expr_odiff_applied": False,
    }

    if cfg.odiff_enabled:
        odiff_audit = bk.compute_move_local_odiff_audit(
            "birth",
            {"parents": tuple(obj["parent_pair"]), "child": int(obj["child"])},
            before_state,
            after_state,
            core_before,
            cfg,
        )
        # Birth-like odiff bonus stays on for proto-child.
        odiff_expr_adj, base_info = bk._bounded_odiff_expr_adjustment(odiff_audit, cfg)
        dE = float(dE + odiff_expr_adj)
        odiff_expr_info.update(base_info)
        odiff_expr_info["expr_odiff_applied"] = True

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
        "dE_expr_proto_readiness": float(0.10 * proto_readiness),
        "proto_readiness": float(proto_readiness),
        "dCB": float(dCB),
        "dCS": float(dCS),
        "dCF": float(dCF),
        "core_before": core_before,
        **nr,
        **birth_info,
        **odiff_expr_info,
    }

    if odiff_audit is not None:
        diag["odiff_audit"] = asdict(odiff_audit)
        diag["A_before_R"] = int(odiff_audit.active_count_before)
        diag["A_after_R"] = int(odiff_audit.active_count_after)
        diag["Odiff_before_R"] = float(odiff_audit.odiff_before)
        diag["Odiff_after_R"] = float(odiff_audit.odiff_after)
        diag["delta_Odiff_R"] = float(odiff_audit.delta_odiff)

    return float(deltaF), diag


def winner_variant_label(diag: Optional[Dict]) -> Optional[str]:
    if diag is None:
        return None
    if diag.get("move_type") == "proto":
        return "proto_child"
    if diag.get("move_type") == "birth":
        return diag.get("birth_variant", "birth")
    return None


def run_sim(pcfg: PhysicsConfig, scfg: bk.ScoreConfig, args):
    xp, is_gpu = phys.get_array_module(pcfg.device)
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs, rng = phys.init_state(pcfg, xp)

    accepted_moves = []
    snapshots = []
    active_trace = []
    edge_trace = []

    n_proto_events = 0
    n_birth_events = 0
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

        move_diags = []
        best_kind = None
        best_state = None
        best_diag = None
        best_delta = 0.0

        # Proto-child candidates over the same birth candidate list, but only for the targeted niche.
        for parents, child in birth_moves:
            if not proto_matches(args, parents, child):
                continue
            moved, meta = prepare_proto_child_move(base_state, parents, child, pcfg, args)
            moved = phys.evolve_prepared_state(moved, pcfg, xp)

            obj = {
                "parent_pair": canonical_edge(*parents),
                "child": int(child),
                "primary_parent": int(meta["primary_parent"]),
                "secondary_parent": int(meta["secondary_parent"]),
                "proto_edge": tuple(meta["proto_edge"]),
            }
            dF, diag = score_proto_child_move(obj, base_state, moved, scfg, xp)
            if float(args.proto_bonus) != 0.0:
                dF = float(dF + float(args.proto_bonus))

            diag["proto_variant"] = "one_sided_proto_child"
            diag["proto_used"] = True
            diag["deltaF_after_proto_bonus"] = float(dF)
            diag.update(meta)

            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_kind = "proto"
                best_state = moved
                best_diag = diag

        # Normal full births remain available.
        for parents, child in birth_moves:
            moved = phys.prepare_birth_move(base_state, parents, child, pcfg)
            moved = phys.evolve_prepared_state(moved, pcfg, xp)
            dF, diag = bk.score_move("birth", {"parents": parents, "child": child}, base_state, moved, scfg, GM_MATRICES, xp)
            diag["move_object"] = {"parents": list(parents), "child": int(child)}
            diag["birth_variant"] = "full_birth"
            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_kind = "birth"
                best_state = moved
                best_diag = diag

        for edge in weaken_moves:
            moved = phys.prepare_weaken_move(base_state, edge, pcfg)
            if moved is None:
                continue
            moved = phys.evolve_prepared_state(moved, pcfg, xp)
            dF, diag = bk.score_move("weaken", edge, base_state, moved, scfg, GM_MATRICES, xp)
            diag["move_object"] = list(edge)
            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_kind = "weaken"
                best_state = moved
                best_diag = diag

        for node in retire_moves:
            moved = phys.prepare_retire_move(base_state, node, pcfg)
            if moved is None:
                continue
            moved = phys.evolve_prepared_state(moved, pcfg, xp)
            dF, diag = bk.score_move("retire", node, base_state, moved, scfg, GM_MATRICES, xp)
            diag["move_object"] = int(node)
            diag["retirement_info"] = retire_info[node]
            move_diags.append(diag)
            if dF > best_delta:
                best_delta = float(dF)
                best_kind = "retire"
                best_state = moved
                best_diag = diag

        n_proto_this_eval = 0
        n_births_this_eval = 0
        n_weakens_this_eval = 0
        n_extinctions_this_eval = 0

        if best_kind is not None and best_delta > 0.0:
            psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = best_state
            accepted_moves.append(best_diag)

            if best_kind == "proto":
                n_proto_events += 1
                n_proto_this_eval = 1
            elif best_kind == "birth":
                n_birth_events += 1
                n_births_this_eval = 1
            elif best_kind == "weaken":
                n_weaken_events += 1
                n_weakens_this_eval = 1
            elif best_kind == "retire":
                n_extinction_events += 1
                n_extinctions_this_eval = 1

        phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
        core_after = bk.dominant_core_snapshot(
            psi, active_nodes, active_edges, edge_strengths, GM_MATRICES, xp, pcfg.n_max
        )
        metrics = phys.metric_snapshot(active_nodes, active_edges, edge_strengths)
        gstats = phys.graph_stats(active_nodes, active_edges)
        mean_link_rank = 0.0
        if active_edges:
            mean_link_rank = float(np.mean([bk.bounded_activity_and_rank(link_regs[e])[1] for e in active_edges]))

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
                "n_weakens_this_eval": int(n_weakens_this_eval),
                "n_extinctions_this_eval": int(n_extinctions_this_eval),
                "candidate_move_diagnostics": move_diags,
            }
        )

        active_trace.append(len(active_nodes))
        edge_trace.append(len(active_edges))

        if pcfg.progress_every > 0 and len(snapshots) % pcfg.progress_every == 0:
            cp = core_after["core_pair"] if core_after else None
            print(
                f"[eval {len(snapshots):03d}] step={step:4d} active={len(active_nodes):2d} "
                f"dormant={len(dormant_nodes):2d} edges={len(active_edges):3d} core={cp} "
                f"proto={n_proto_this_eval} births={n_births_this_eval} weakens={n_weakens_this_eval} "
                f"extinctions={n_extinctions_this_eval} winner={best_diag.get('move_type') if best_diag else None} "
                f"variant={winner_variant_label(best_diag)} "
                f"cand={len(birth_moves) + len(weaken_moves) + len(retire_moves) + sum(1 for parents, child in birth_moves if proto_matches(args, parents, child))}"
            )

    return {
        "script": "hsf_mesoscape_proto_child_sandbox.py",
        "physics_config": asdict(pcfg),
        "bookkeeping_config": asdict(scfg),
        "proto_config": {
            "enabled": bool(args.proto_child),
            "proto_parent_pair": (
                list(canonical_edge(*args.proto_parent_pair)) if args.proto_parent_pair is not None else None
            ),
            "proto_child_node": (int(args.proto_child_node) if args.proto_child_node is not None else None),
            "proto_primary_parent": (int(args.proto_primary_parent) if args.proto_primary_parent is not None else None),
            "proto_local_alpha": float(args.proto_local_alpha),
            "proto_edge_alpha": float(args.proto_edge_alpha),
            "proto_bonus": float(args.proto_bonus),
        },
        "n_proto_events": int(n_proto_events),
        "n_birth_events": int(n_birth_events),
        "n_weaken_events": int(n_weaken_events),
        "n_extinction_events": int(n_extinction_events),
        "final_active_nodes": list(sorted(active_nodes)),
        "final_dormant_nodes": list(sorted(dormant_nodes)),
        "final_active_edges": [list(e) for e in sorted(active_edges)],
        "active_count_trace": active_trace,
        "edge_count_trace": edge_trace,
        "accepted_moves": accepted_moves,
        "snapshots": snapshots,
        "gpu_enabled": bool(is_gpu),
    }


def main():
    args = parse_args()
    pcfg, scfg = build_configs(args)
    result = run_sim(pcfg, scfg, args)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()