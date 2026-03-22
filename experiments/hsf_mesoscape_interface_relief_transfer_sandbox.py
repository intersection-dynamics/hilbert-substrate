#!/usr/bin/env python3
# filename: hsf_mesoscape_interface_relief_transfer_sandbox.py

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


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "HSF mesoscale interface-relief-transfer sandbox. "
            "Tests a sub-child relief mode that reweights existing parent-interface structure "
            "without instantiating a new support patch."
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

    # Targeted interface-relief transfer controls.
    p.add_argument(
        "--interface-transfer",
        action="store_true",
        help="Enable the interface-relief-transfer candidate move for the targeted niche.",
    )
    p.add_argument(
        "--transfer-parent-pair",
        nargs=2,
        type=int,
        default=None,
        help="Target parent pair, e.g. --transfer-parent-pair 2 3",
    )
    p.add_argument(
        "--transfer-primary-parent",
        type=int,
        default=None,
        help="If set, favor relief transfer from this parent side. Otherwise the lower-index parent is primary.",
    )
    p.add_argument(
        "--transfer-support-node",
        type=int,
        default=None,
        help="Optional specific active support node to receive transferred relief.",
    )
    p.add_argument(
        "--pair-retain",
        type=float,
        default=0.88,
        help="Retain factor for the parent-pair edge strength during transfer.",
    )
    p.add_argument(
        "--primary-boost",
        type=float,
        default=1.30,
        help="Multiplicative boost for edges from the primary parent to selected support nodes.",
    )
    p.add_argument(
        "--secondary-boost",
        type=float,
        default=1.05,
        help="Multiplicative boost for edges from the secondary parent to selected support nodes.",
    )
    p.add_argument(
        "--max-boost-edges",
        type=int,
        default=2,
        help="Maximum number of support nodes to receive redistributed relief.",
    )
    p.add_argument(
        "--support-selection",
        choices=["common_first", "union_first", "common_only"],
        default="common_first",
        help="How to choose existing support nodes around the parent pair.",
    )
    p.add_argument(
        "--transfer-bonus",
        type=float,
        default=0.0,
        help="Optional additive bonus to deltaF for transfer moves.",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="hsf_mesoscape_interface_relief_transfer_sandbox.json",
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


def neighbor_map(active_edges: Set[Edge]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for a, b in active_edges:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def transfer_matches(args, parents: Edge) -> bool:
    if not bool(args.interface_transfer):
        return False
    if args.transfer_parent_pair is None:
        return True
    return canonical_edge(*parents) == canonical_edge(int(args.transfer_parent_pair[0]), int(args.transfer_parent_pair[1]))


def choose_primary_parent(parents: Edge, requested_primary: Optional[int]) -> int:
    i, j = canonical_edge(*parents)
    if requested_primary is None:
        return i
    rp = int(requested_primary)
    return rp if rp in (i, j) else i


def select_support_nodes(
    parents: Edge,
    active_nodes: Set[int],
    active_edges: Set[Edge],
    args,
) -> List[int]:
    i, j = canonical_edge(*parents)
    nbr = neighbor_map(active_edges)
    common = sorted((nbr.get(i, set()) & nbr.get(j, set())) - {i, j})
    union = sorted((nbr.get(i, set()) | nbr.get(j, set())) - {i, j})

    if args.transfer_support_node is not None:
        node = int(args.transfer_support_node)
        if node in active_nodes and node in union:
            return [node]
        return []

    if args.support_selection == "common_only":
        chosen = common
    elif args.support_selection == "union_first":
        chosen = union
    else:
        chosen = common if common else union

    return [int(n) for n in chosen[: max(1, int(args.max_boost_edges))]]


def prepare_interface_relief_transfer_move(
    state,
    parents: Edge,
    pcfg: PhysicsConfig,
    args,
):
    """
    Reweight existing parent-interface structure without creating a new node:
    - slightly relax the parent-pair edge
    - strengthen existing edges from the parent pair into already-active nearby support
    This is a sandbox stand-in for a partial local relief mode.
    """
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = phys.clone_graph_state(*state)
    i, j = canonical_edge(*parents)
    pp = canonical_edge(i, j)
    if pp not in active_edges:
        return None, None

    support_nodes = select_support_nodes((i, j), active_nodes, active_edges, args)
    if not support_nodes:
        return None, None

    primary_parent = choose_primary_parent((i, j), args.transfer_primary_parent)
    secondary_parent = j if primary_parent == i else i

    pair_retain = max(0.0, float(args.pair_retain))
    primary_boost = max(0.0, float(args.primary_boost))
    secondary_boost = max(0.0, float(args.secondary_boost))

    adjusted_edges = []

    # Relax the parent pair slightly.
    old_pp_strength = float(edge_strengths.get(pp, 0.0))
    edge_strengths[pp] = float(old_pp_strength * pair_retain)
    adjusted_edges.append(
        {
            "edge": list(pp),
            "old_strength": old_pp_strength,
            "new_strength": float(edge_strengths[pp]),
            "kind": "pair_relax",
        }
    )

    n_adjusted = 0
    for node in support_nodes:
        did_any = False

        e1 = canonical_edge(primary_parent, node)
        if e1 in active_edges:
            old_s = float(edge_strengths.get(e1, 0.0))
            edge_strengths[e1] = float(old_s * primary_boost)
            adjusted_edges.append(
                {
                    "edge": list(e1),
                    "old_strength": old_s,
                    "new_strength": float(edge_strengths[e1]),
                    "kind": "primary_boost",
                }
            )
            did_any = True

        e2 = canonical_edge(secondary_parent, node)
        if e2 in active_edges:
            old_s = float(edge_strengths.get(e2, 0.0))
            edge_strengths[e2] = float(old_s * secondary_boost)
            adjusted_edges.append(
                {
                    "edge": list(e2),
                    "old_strength": old_s,
                    "new_strength": float(edge_strengths[e2]),
                    "kind": "secondary_boost",
                }
            )
            did_any = True

        if did_any:
            n_adjusted += 1

    if n_adjusted == 0:
        return None, None

    phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)

    meta = {
        "transfer_parent_pair": [int(i), int(j)],
        "primary_parent": int(primary_parent),
        "secondary_parent": int(secondary_parent),
        "support_nodes": [int(n) for n in support_nodes],
        "pair_retain": float(pair_retain),
        "primary_boost": float(primary_boost),
        "secondary_boost": float(secondary_boost),
        "adjusted_edges": adjusted_edges,
    }
    return (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs), meta


def score_transfer_move(
    obj,
    before_state,
    after_state,
    cfg: bk.ScoreConfig,
    xp,
):
    """
    Sandbox scoring for interface transfer.
    Uses the same burden and no-refolding machinery as normal scoring,
    but treats this as a distinct move kind instead of forcing it into birth/weaken/retire.
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

    cb_b = bk.bandwidth_burden(active_edges_b, link_regs_b)
    cb_a = bk.bandwidth_burden(active_edges_a, link_regs_a)
    dCB = float(cb_a - cb_b)

    cs_b = bk.spread_burden(active_nodes_b, active_edges_b)
    cs_a = bk.spread_burden(active_nodes_a, active_edges_a)
    dCS = float(cs_a - cs_b)

    nr = bk.no_refolding_witness(
        "transfer",
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
        odiff_audit = bk.compute_move_local_odiff_audit("transfer", obj, before_state, after_state, core_before, cfg)

    deltaF = dE - cfg.lambda_B * dCB - cfg.lambda_S * dCS - cfg.lambda_F * dCF - cfg.lambda_R * nr["W_NR"]

    diag = {
        "move_type": "transfer",
        "move_object": {
            "parent_pair": [int(x) for x in obj["parent_pair"]],
            "support_nodes": [int(x) for x in obj["support_nodes"]],
            "primary_parent": int(obj["primary_parent"]),
            "secondary_parent": int(obj["secondary_parent"]),
        },
        "deltaF": float(deltaF),
        "dE_expr": float(dE),
        "dE_expr_raw": float(dE_raw),
        "dE_expr_structural_adjustment": 0.0,
        "dE_expr_odiff_adjustment": float(odiff_expr_adj),
        "dCB": float(dCB),
        "dCS": float(dCS),
        "dCF": float(dCF),
        "core_before": core_before,
        **nr,
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
    if diag.get("move_type") == "transfer":
        return "interface_transfer"
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

    n_transfer_events = 0
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

        # Targeted interface-transfer candidate.
        if args.interface_transfer and args.transfer_parent_pair is not None:
            tp = canonical_edge(int(args.transfer_parent_pair[0]), int(args.transfer_parent_pair[1]))
            if transfer_matches(args, tp):
                moved, meta = prepare_interface_relief_transfer_move(base_state, tp, pcfg, args)
                if moved is not None and meta is not None:
                    moved = phys.evolve_prepared_state(moved, pcfg, xp)
                    obj = {
                        "parent_pair": tp,
                        "support_nodes": tuple(meta["support_nodes"]),
                        "primary_parent": int(meta["primary_parent"]),
                        "secondary_parent": int(meta["secondary_parent"]),
                    }
                    dF, diag = score_transfer_move(obj, base_state, moved, scfg, xp)
                    if float(args.transfer_bonus) != 0.0:
                        dF = float(dF + float(args.transfer_bonus))
                    diag["transfer_variant"] = "interface_relief_transfer"
                    diag["transfer_used"] = True
                    diag["deltaF_after_transfer_bonus"] = float(dF)
                    diag.update(meta)
                    move_diags.append(diag)
                    if dF > best_delta:
                        best_delta = float(dF)
                        best_kind = "transfer"
                        best_state = moved
                        best_diag = diag

        # Normal birth candidates.
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

        # Normal weaken candidates.
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

        # Normal retire candidates.
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

        n_births_this_eval = 0
        n_transfers_this_eval = 0
        n_weakens_this_eval = 0
        n_extinctions_this_eval = 0

        if best_kind is not None and best_delta > 0.0:
            psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = best_state
            accepted_moves.append(best_diag)

            if best_kind == "transfer":
                n_transfer_events += 1
                n_transfers_this_eval = 1
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
                "n_births_this_eval": int(n_births_this_eval),
                "n_transfers_this_eval": int(n_transfers_this_eval),
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
                f"births={n_births_this_eval} transfers={n_transfers_this_eval} "
                f"weakens={n_weakens_this_eval} extinctions={n_extinctions_this_eval} "
                f"winner={best_diag.get('move_type') if best_diag else None} "
                f"variant={winner_variant_label(best_diag)} "
                f"cand={len(birth_moves) + len(weaken_moves) + len(retire_moves) + (1 if args.interface_transfer and args.transfer_parent_pair is not None else 0)}"
            )

    return {
        "script": "hsf_mesoscape_interface_relief_transfer_sandbox.py",
        "physics_config": asdict(pcfg),
        "bookkeeping_config": asdict(scfg),
        "transfer_config": {
            "enabled": bool(args.interface_transfer),
            "transfer_parent_pair": (
                list(canonical_edge(*args.transfer_parent_pair)) if args.transfer_parent_pair is not None else None
            ),
            "transfer_primary_parent": (
                int(args.transfer_primary_parent) if args.transfer_primary_parent is not None else None
            ),
            "transfer_support_node": (
                int(args.transfer_support_node) if args.transfer_support_node is not None else None
            ),
            "pair_retain": float(args.pair_retain),
            "primary_boost": float(args.primary_boost),
            "secondary_boost": float(args.secondary_boost),
            "max_boost_edges": int(args.max_boost_edges),
            "support_selection": str(args.support_selection),
            "transfer_bonus": float(args.transfer_bonus),
        },
        "n_transfer_events": int(n_transfer_events),
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