#!/usr/bin/env python3
# filename: hsf_mesoscape_asymmetric_soft_birth_sandbox.py

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
            "HSF mesoscale asymmetric soft-birth sandbox. "
            "Tests whether partial parent-interface relief is better represented by "
            "an asymmetrically attached newborn than by a symmetric soft birth."
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

    # Asymmetric soft-birth controls.
    p.add_argument(
        "--asym-soft-birth",
        action="store_true",
        help="Enable asymmetric soft birth for the targeted niche. If omitted, all births are full births.",
    )
    p.add_argument(
        "--soft-birth-parent-pair",
        nargs=2,
        type=int,
        default=None,
        help="Target parent pair for asymmetric soft birth, e.g. --soft-birth-parent-pair 2 3",
    )
    p.add_argument(
        "--soft-birth-child",
        type=int,
        default=None,
        help="Target child for asymmetric soft birth, e.g. --soft-birth-child 5",
    )
    p.add_argument(
        "--soft-birth-local-alpha",
        type=float,
        default=0.60,
        help="Scale factor for newborn local coefficient amplitude.",
    )
    p.add_argument(
        "--soft-birth-primary-edge-alpha",
        type=float,
        default=0.80,
        help="Scale factor for the stronger of the two newborn parent-child interfaces.",
    )
    p.add_argument(
        "--soft-birth-secondary-edge-alpha",
        type=float,
        default=0.20,
        help="Scale factor for the weaker of the two newborn parent-child interfaces.",
    )
    p.add_argument(
        "--soft-birth-primary-parent",
        type=int,
        default=None,
        help=(
            "If set, use this parent as the stronger newborn interface. "
            "Otherwise the lower-index parent gets the stronger edge."
        ),
    )
    p.add_argument(
        "--soft-birth-bonus",
        type=float,
        default=0.0,
        help="Optional additive bonus to deltaF for asymmetric soft births.",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="hsf_mesoscape_asymmetric_soft_birth_sandbox.json",
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


def _asym_birth_matches(args, parents: Edge, child: int) -> bool:
    if not bool(args.asym_soft_birth):
        return False

    pp = canonical_edge(*parents)
    target_pp = None
    if args.soft_birth_parent_pair is not None:
        target_pp = canonical_edge(int(args.soft_birth_parent_pair[0]), int(args.soft_birth_parent_pair[1]))
    target_child = None if args.soft_birth_child is None else int(args.soft_birth_child)

    if target_pp is None and target_child is None:
        return True
    if target_pp is not None and pp != target_pp:
        return False
    if target_child is not None and int(child) != target_child:
        return False
    return True


def _choose_primary_parent(parents: Edge, requested_primary: Optional[int]) -> int:
    i, j = canonical_edge(*parents)
    if requested_primary is None:
        return i
    rp = int(requested_primary)
    if rp in (i, j):
        return rp
    return i


def prepare_asymmetric_soft_birth_move(
    state,
    parents: Edge,
    child: int,
    pcfg: PhysicsConfig,
    args,
):
    """
    Sandbox move:
    same topology as a birth, but the newborn is attached asymmetrically:
    one parent-child edge is stronger, the other weaker.
    This approximates partial parent-interface relief rather than just a dim child.
    """
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = phys.clone_graph_state(*state)
    i, j = canonical_edge(*parents)
    primary_parent = _choose_primary_parent((i, j), args.soft_birth_primary_parent)
    secondary_parent = j if primary_parent == i else i

    local_alpha = max(0.0, float(args.soft_birth_local_alpha))
    primary_alpha = max(0.0, float(args.soft_birth_primary_edge_alpha))
    secondary_alpha = max(0.0, float(args.soft_birth_secondary_edge_alpha))

    active_nodes.add(int(child))
    dormant_nodes.discard(int(child))

    local_coeffs[int(child)] = float(pcfg.spawn_pair_scale) * local_alpha

    base_strength = max(float(pcfg.spawn_pair_scale), float(pcfg.pair_scale))

    e_primary = canonical_edge(primary_parent, int(child))
    e_secondary = canonical_edge(secondary_parent, int(child))

    active_edges.add(e_primary)
    active_edges.add(e_secondary)

    edge_strengths[e_primary] = float(base_strength * primary_alpha)
    edge_strengths[e_secondary] = float(base_strength * secondary_alpha)

    link_regs[e_primary] = phys.default_linkreg().copy()
    link_regs[e_secondary] = phys.default_linkreg().copy()

    phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
    meta = {
        "primary_parent": int(primary_parent),
        "secondary_parent": int(secondary_parent),
        "primary_edge": list(e_primary),
        "secondary_edge": list(e_secondary),
        "soft_birth_local_alpha": float(local_alpha),
        "soft_birth_primary_edge_alpha": float(primary_alpha),
        "soft_birth_secondary_edge_alpha": float(secondary_alpha),
    }
    return (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs), meta


def _winner_variant_label(diag: Optional[Dict]) -> Optional[str]:
    if diag is None:
        return None
    if diag.get("move_type") != "birth":
        return None
    if bool(diag.get("asym_soft_birth_used", False)):
        return "asym_soft_birth"
    return "full_birth"


def run_sim(pcfg: PhysicsConfig, scfg: bk.ScoreConfig, args):
    xp, is_gpu = phys.get_array_module(pcfg.device)
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs, rng = phys.init_state(pcfg, xp)

    accepted_moves = []
    snapshots = []
    active_trace = []
    edge_trace = []

    n_asym_soft_birth_events = 0
    n_full_birth_events = 0

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

        for parents, child in birth_moves:
            use_asym = _asym_birth_matches(args, parents, child)

            if use_asym:
                moved, asym_meta = prepare_asymmetric_soft_birth_move(base_state, parents, child, pcfg, args)
                birth_variant = "asym_soft_birth"
            else:
                moved = phys.prepare_birth_move(base_state, parents, child, pcfg)
                asym_meta = None
                birth_variant = "full_birth"

            moved = phys.evolve_prepared_state(moved, pcfg, xp)
            dF, diag = bk.score_move("birth", {"parents": parents, "child": child}, base_state, moved, scfg, GM_MATRICES, xp)

            if use_asym and float(args.soft_birth_bonus) != 0.0:
                dF = float(dF + float(args.soft_birth_bonus))

            diag["move_object"] = {"parents": list(parents), "child": int(child)}
            diag["birth_variant"] = birth_variant
            diag["asym_soft_birth_used"] = bool(use_asym)
            diag["deltaF_after_soft_birth_bonus"] = float(dF)
            if asym_meta is not None:
                diag.update(asym_meta)

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

        n_births = 0
        n_asym_soft_births = 0
        n_full_births = 0
        n_weakens = 0
        n_extinctions = 0

        if best_kind is not None and best_delta > 0.0:
            psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = best_state
            accepted_moves.append(best_diag)

            if best_kind == "birth":
                n_births = 1
                if bool(best_diag.get("asym_soft_birth_used", False)):
                    n_asym_soft_births = 1
                    n_asym_soft_birth_events += 1
                else:
                    n_full_births = 1
                    n_full_birth_events += 1
            elif best_kind == "weaken":
                n_weakens = 1
            elif best_kind == "retire":
                n_extinctions = 1

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
                "n_births_this_eval": int(n_births),
                "n_asym_soft_births_this_eval": int(n_asym_soft_births),
                "n_full_births_this_eval": int(n_full_births),
                "n_weakens_this_eval": int(n_weakens),
                "n_extinctions_this_eval": int(n_extinctions),
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
                f"births={n_births} asym={n_asym_soft_births} weakens={n_weakens} extinctions={n_extinctions} "
                f"winner={best_diag.get('move_type') if best_diag else None} "
                f"variant={_winner_variant_label(best_diag)} "
                f"cand={len(birth_moves) + len(weaken_moves) + len(retire_moves)}"
            )

    return {
        "script": "hsf_mesoscape_asymmetric_soft_birth_sandbox.py",
        "physics_config": asdict(pcfg),
        "bookkeeping_config": asdict(scfg),
        "asymmetric_soft_birth_config": {
            "enabled": bool(args.asym_soft_birth),
            "soft_birth_parent_pair": (
                list(canonical_edge(*args.soft_birth_parent_pair)) if args.soft_birth_parent_pair is not None else None
            ),
            "soft_birth_child": (int(args.soft_birth_child) if args.soft_birth_child is not None else None),
            "soft_birth_local_alpha": float(args.soft_birth_local_alpha),
            "soft_birth_primary_edge_alpha": float(args.soft_birth_primary_edge_alpha),
            "soft_birth_secondary_edge_alpha": float(args.soft_birth_secondary_edge_alpha),
            "soft_birth_primary_parent": (
                int(args.soft_birth_primary_parent) if args.soft_birth_primary_parent is not None else None
            ),
            "soft_birth_bonus": float(args.soft_birth_bonus),
        },
        "n_birth_events": int(sum(1 for m in accepted_moves if m["move_type"] == "birth")),
        "n_asym_soft_birth_events": int(n_asym_soft_birth_events),
        "n_full_birth_events": int(n_full_birth_events),
        "n_weaken_events": int(sum(1 for m in accepted_moves if m["move_type"] == "weaken")),
        "n_extinction_events": int(sum(1 for m in accepted_moves if m["move_type"] == "retire")),
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