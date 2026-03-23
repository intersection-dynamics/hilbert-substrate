#!/usr/bin/env python3
# filename: hsf_mesoscape_entanglement_local.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import hsf_mesoscale_bookkeeping as bk
import hsf_mesoscale_physics_core as phys
from hsf_mesoscale_physics_core import PhysicsConfig, GM_MATRICES, canonical_edge

Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HSF mesoscale entanglement-local sandbox. "
            "Birth candidates are generated only from entangled parent pairs, using the current "
            "project note language for lawful support-management while keeping entanglement critical to birth."
        )
    )
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--n-init", type=int, default=2)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--local-scale", type=float, default=0.15)
    p.add_argument("--pair-scale", type=float, default=0.12)
    p.add_argument("--spawn-pair-scale", type=float, default=0.11)
    p.add_argument("--total-steps", type=int, default=60)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--progress-every", type=int, default=1)

    p.add_argument("--lambda-B", type=float, default=0.18)
    p.add_argument("--lambda-S", type=float, default=0.12)
    p.add_argument("--lambda-F", type=float, default=0.20)
    p.add_argument("--lambda-R", type=float, default=0.35)
    p.add_argument("--w-mi", type=float, default=1.0)
    p.add_argument("--w-corr", type=float, default=0.5)
    p.add_argument("--w-link", type=float, default=0.5)
    p.add_argument("--retirement-threshold", type=float, default=0.66)
    p.add_argument("--organizer-large-region-cutoff", type=int, default=8)

    p.add_argument("--sigma-on-threshold", type=float, default=0.08)
    p.add_argument("--edge-on-threshold", type=float, default=0.05)
    p.add_argument("--sigma-step", type=float, default=0.25)
    p.add_argument("--edge-step", type=float, default=0.25)

    p.add_argument("--max-parent-pairs", type=int, default=4)
    p.add_argument("--max-children-per-pair", type=int, default=2)
    p.add_argument("--max-lower-cands", type=int, default=2)
    p.add_argument("--max-edge-up-cands", type=int, default=4)
    p.add_argument("--max-edge-down-cands", type=int, default=4)

    p.add_argument("--ent-threshold", type=float, default=0.01)
    p.add_argument("--json-out", type=str, default="hsf_mesoscape_entanglement_local.json")
    return p.parse_args()


def build_configs(args: argparse.Namespace):
    pcfg = PhysicsConfig(
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
    scfg = bk.ScoreConfig(
        lambda_B=args.lambda_B,
        lambda_S=args.lambda_S,
        lambda_F=args.lambda_F,
        lambda_R=args.lambda_R,
        w_mi=args.w-mi if False else args.w_mi,  # keep exact runtime behavior below
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
    for i in range(pcfg.n_max):
        for j in range(i + 1, pcfg.n_max):
            e = canonical_edge(i, j)
            interface_commitment[e] = 1.0 if e in active_edges else 0.0
    link_memory = {canonical_edge(*e): np.array(reg, copy=True) for e, reg in link_regs.items()}
    return psi, sigma, interface_commitment, link_memory, rng


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
        i, j = e
        if i not in active_nodes or j not in active_nodes:
            continue
        if float(w) <= float(edge_on_threshold):
            continue
        active_edges.add(e)
        edge_strengths[e] = float(pcfg.pair_scale) * float(np.clip(w, 0.0, 1.0))
        link_regs[e] = np.array(link_memory.get(e, phys.default_linkreg()), copy=True)

    phys.sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
    return psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs


def evolve_prepared(graded_state, pcfg: PhysicsConfig, args: argparse.Namespace, xp):
    prepared = materialize_state(
        graded_state[0], graded_state[1], graded_state[2], graded_state[3], pcfg,
        args.sigma_on_threshold, args.edge_on_threshold,
    )
    return phys.evolve_prepared_state(prepared, pcfg, xp)


def entanglement_sources(psi, sigma: np.ndarray, active_nodes: Set[int], xp, scfg: bk.ScoreConfig) -> List[Dict[str, Any]]:
    n_sites = int(psi.ndim)
    rows: List[Dict[str, Any]] = []
    nodes = sorted(active_nodes)
    for idx, a in enumerate(nodes):
        for b in nodes[idx + 1:]:
            mi = float(bk.mutual_information_from_state(psi, a, b, n_sites, xp))
            corr = float(bk.pair_su3_correlator_strength(psi, a, b, GM_MATRICES, xp))
            ent = float(np.tanh(max(0.0, mi)) + 0.25 * np.tanh(4.0 * max(0.0, corr)))
            source = float(ent * np.clip(sigma[a], 0.0, 1.0) * np.clip(sigma[b], 0.0, 1.0))
            rows.append({"parents": [int(a), int(b)], "mi": mi, "corr": corr, "ent": ent, "source": source})
    rows.sort(key=lambda r: r["source"], reverse=True)
    return rows


def candidate_births(psi, sigma: np.ndarray, interface_commitment: Dict[Edge, float], active_nodes: Set[int], args: argparse.Namespace, xp, scfg: bk.ScoreConfig):
    parent_rows = entanglement_sources(psi, sigma, active_nodes, xp, scfg)
    out: List[Tuple[Edge, int, Dict[str, Any]]] = []
    for row in parent_rows[: int(args.max_parent_pairs)]:
        if float(row["source"]) <= float(args.ent_threshold):
            continue
        a, b = int(row["parents"][0]), int(row["parents"][1])
        candidates: List[Tuple[float, int]] = []
        for child in range(len(sigma)):
            if child in (a, b):
                continue
            attach = float(interface_commitment.get(canonical_edge(a, child), 0.0) + interface_commitment.get(canonical_edge(b, child), 0.0))
            child_priority = float(1.0 - np.clip(sigma[child], 0.0, 1.0)) * float(1.0 - 0.5 * np.clip(attach, 0.0, 1.0))
            candidates.append((child_priority, child))
        candidates.sort(reverse=True)
        for _, child in candidates[: int(args.max_children_per_pair)]:
            out.append((canonical_edge(a, b), int(child), dict(row)))
    return out


def candidate_lower_supports(sigma: np.ndarray, active_nodes: Set[int], max_cands: int) -> List[int]:
    rows = sorted(((float(sigma[i]), int(i)) for i in active_nodes), key=lambda x: x[0])
    return [i for _, i in rows[: max_cands]]


def candidate_edge_ups(interface_commitment: Dict[Edge, float], births: List[Tuple[Edge, int, Dict[str, Any]]], max_cands: int) -> List[Edge]:
    seen: List[Edge] = []
    for parents, child, _ in births:
        a, b = parents
        for e in (canonical_edge(a, child), canonical_edge(b, child)):
            if e not in seen and float(interface_commitment.get(e, 0.0)) < 1.0 - 1e-12:
                seen.append(e)
            if len(seen) >= max_cands:
                return seen
    return seen


def candidate_edge_downs(interface_commitment: Dict[Edge, float], active_edges: Set[Edge], max_cands: int) -> List[Edge]:
    rows = sorted(((float(interface_commitment.get(e, 0.0)), e) for e in active_edges), key=lambda x: x[0])
    return [e for _, e in rows[: max_cands]]


def prepare_birth_move(graded_state, parents: Edge, child: int, args: argparse.Namespace):
    psi, sigma, interface_commitment, link_memory = clone_graded_state(*graded_state)
    a, b = parents
    sigma_before = float(sigma[child])
    sigma[child] = float(np.clip(sigma[child] + float(args.sigma_step), 0.0, 1.0))
    for p in (a, b):
        e = canonical_edge(p, child)
        interface_commitment[e] = float(np.clip(interface_commitment.get(e, 0.0) + float(args.edge_step), 0.0, 1.0))
    return (psi, sigma, interface_commitment, link_memory), {
        "move_type": "birth",
        "parents": [int(a), int(b)],
        "child": int(child),
        "sigma_before": sigma_before,
        "sigma_after": float(sigma[child]),
    }


def prepare_lower_support_move(graded_state, node: int, args: argparse.Namespace):
    psi, sigma, interface_commitment, link_memory = clone_graded_state(*graded_state)
    sigma_before = float(sigma[node])
    sigma[node] = float(max(0.0, sigma[node] - float(args.sigma_step)))
    ratio = float(sigma[node] / max(1e-12, sigma_before)) if sigma_before > 0 else 0.0
    for e in list(interface_commitment.keys()):
        if node in e:
            interface_commitment[e] = float(np.clip(interface_commitment[e] * ratio, 0.0, 1.0))
    return (psi, sigma, interface_commitment, link_memory), {
        "move_type": "lower_support",
        "node": int(node),
        "sigma_before": sigma_before,
        "sigma_after": float(sigma[node]),
    }


def prepare_edge_up_move(graded_state, edge: Edge, args: argparse.Namespace):
    psi, sigma, interface_commitment, link_memory = clone_graded_state(*graded_state)
    e = canonical_edge(*edge)
    before = float(interface_commitment.get(e, 0.0))
    interface_commitment[e] = float(np.clip(before + float(args.edge_step), 0.0, 1.0))
    return (psi, sigma, interface_commitment, link_memory), {
        "move_type": "edge_up",
        "edge": [int(e[0]), int(e[1])],
        "w_before": before,
        "w_after": float(interface_commitment[e]),
    }


def prepare_edge_down_move(graded_state, edge: Edge, args: argparse.Namespace):
    psi, sigma, interface_commitment, link_memory = clone_graded_state(*graded_state)
    e = canonical_edge(*edge)
    before = float(interface_commitment.get(e, 0.0))
    interface_commitment[e] = float(max(0.0, before - float(args.edge_step)))
    return (psi, sigma, interface_commitment, link_memory), {
        "move_type": "edge_down",
        "edge": [int(e[0]), int(e[1])],
        "w_before": before,
        "w_after": float(interface_commitment[e]),
    }


def move_kind_and_obj(meta: Dict[str, Any]):
    mt = str(meta["move_type"])
    if mt == "birth":
        return "birth", {"parents": tuple(meta["parents"]), "child": int(meta["child"])}
    if mt == "lower_support":
        return "retire", int(meta["node"])
    if mt == "edge_up":
        return "transfer", {"edge": canonical_edge(*meta["edge"])}
    if mt == "edge_down":
        return "weaken", canonical_edge(*meta["edge"])
    raise ValueError(mt)


def score_candidate(meta: Dict[str, Any], graded_before, graded_after, baseline_state, pcfg: PhysicsConfig, scfg: bk.ScoreConfig, args: argparse.Namespace, xp):
    state_b = baseline_state
    state_a = evolve_prepared(graded_after, pcfg, args, xp)

    psi_b, active_nodes_b, _, active_edges_b, local_coeffs_b, edge_strengths_b, link_regs_b = state_b
    psi_a, active_nodes_a, _, active_edges_a, local_coeffs_a, edge_strengths_a, link_regs_a = state_a
    n_sites = int(local_coeffs_b.shape[0])

    expr_b = bk.local_expression(psi_b, active_nodes_b, active_edges_b, edge_strengths_b, link_regs_b, scfg, GM_MATRICES, xp, n_sites)
    expr_a = bk.local_expression(psi_a, active_nodes_a, active_edges_a, edge_strengths_a, link_regs_a, scfg, GM_MATRICES, xp, n_sites)
    dE_raw = float(expr_a - expr_b)

    dCB = float(bk.bandwidth_burden(active_edges_a, link_regs_a) - bk.bandwidth_burden(active_edges_b, link_regs_b))
    dCS = float(bk.spread_burden(active_nodes_a, active_edges_a) - bk.spread_burden(active_nodes_b, active_edges_b))

    core_before = bk.dominant_core_snapshot(psi_b, active_nodes_b, active_edges_b, edge_strengths_b, GM_MATRICES, xp, n_sites, scfg, link_regs_b)
    move_kind, move_obj = move_kind_and_obj(meta)
    nr = bk.no_refolding_witness(
        move_kind,
        move_obj,
        psi_b,
        psi_a,
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
    deltaF = float(dE_raw - scfg.lambda_B * dCB - scfg.lambda_S * dCS - scfg.lambda_F * dCF - scfg.lambda_R * nr["W_NR"])

    out = {
        "move_type": meta["move_type"],
        "deltaF": deltaF,
        "dE_raw": dE_raw,
        "dCB": dCB,
        "dCS": dCS,
        "dCF": dCF,
        "F_org": float(nr["F_org"]),
        "W_func": float(nr["W_func"]),
        "W_NR": float(nr["W_NR"]),
        **meta,
    }
    if meta["move_type"] == "birth":
        bw = bk.birth_justification_witness(move_obj, state_b, state_a, scfg, GM_MATRICES, xp)
        out.update(bw)
    return out, state_a


def compact(diag: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "move_type", "deltaF", "dE_raw", "dCB", "dCS", "dCF", "F_org", "W_func", "W_NR",
        "parents", "child", "node", "edge", "sigma_before", "sigma_after", "w_before", "w_after",
        "birth_parent_relief", "birth_novelty", "birth_justification", "birth_distinctness"
    ]
    return {k: diag[k] for k in keep if k in diag}


def run_sim(pcfg: PhysicsConfig, scfg: bk.ScoreConfig, args: argparse.Namespace):
    xp, is_gpu = phys.get_array_module(args.device)
    psi, sigma, interface_commitment, link_memory, rng = init_graded_state(pcfg, xp)

    accepted_moves: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []
    move_counts = {"birth": 0, "lower_support": 0, "edge_up": 0, "edge_down": 0, "no_move": 0}

    for step in range(1, args.total_steps + 1):
        graded_before = (psi, sigma, interface_commitment, link_memory)
        baseline_state = evolve_prepared(graded_before, pcfg, args, xp)
        psi = baseline_state[0]
        baseline_state = evolve_prepared((psi, sigma, interface_commitment, link_memory), pcfg, args, xp)
        psi_b, active_nodes_b, _, active_edges_b, _, _, link_regs_b = baseline_state

        births = candidate_births(psi_b, sigma, interface_commitment, active_nodes_b, args, xp, scfg)
        lowers = candidate_lower_supports(sigma, active_nodes_b, int(args.max_lower_cands))
        edge_ups = candidate_edge_ups(interface_commitment, births, int(args.max_edge_up_cands))
        edge_downs = candidate_edge_downs(interface_commitment, active_edges_b, int(args.max_edge_down_cands))

        diags: List[Dict[str, Any]] = []
        best_diag: Optional[Dict[str, Any]] = None
        best_state = None
        best_graded_after = None
        best_delta = 0.0

        for parents, child, src in births:
            graded_after, meta = prepare_birth_move(graded_before, parents, child, args)
            meta["ent_source"] = float(src["source"])
            meta["ent_mi"] = float(src["mi"])
            meta["ent_corr"] = float(src["corr"])
            diag, state_after = score_candidate(meta, graded_before, graded_after, baseline_state, pcfg, scfg, args, xp)
            diags.append(diag)
            if float(diag["deltaF"]) > best_delta:
                best_delta = float(diag["deltaF"])
                best_diag = diag
                best_state = state_after
                best_graded_after = graded_after

        for node in lowers:
            graded_after, meta = prepare_lower_support_move(graded_before, node, args)
            diag, state_after = score_candidate(meta, graded_before, graded_after, baseline_state, pcfg, scfg, args, xp)
            diags.append(diag)
            if float(diag["deltaF"]) > best_delta:
                best_delta = float(diag["deltaF"])
                best_diag = diag
                best_state = state_after
                best_graded_after = graded_after

        for e in edge_ups:
            graded_after, meta = prepare_edge_up_move(graded_before, e, args)
            diag, state_after = score_candidate(meta, graded_before, graded_after, baseline_state, pcfg, scfg, args, xp)
            diags.append(diag)
            if float(diag["deltaF"]) > best_delta:
                best_delta = float(diag["deltaF"])
                best_diag = diag
                best_state = state_after
                best_graded_after = graded_after

        for e in edge_downs:
            graded_after, meta = prepare_edge_down_move(graded_before, e, args)
            diag, state_after = score_candidate(meta, graded_before, graded_after, baseline_state, pcfg, scfg, args, xp)
            diags.append(diag)
            if float(diag["deltaF"]) > best_delta:
                best_delta = float(diag["deltaF"])
                best_diag = diag
                best_state = state_after
                best_graded_after = graded_after

        accepted = None
        if best_diag is not None and best_graded_after is not None and best_state is not None:
            psi, sigma, interface_commitment, _ = best_graded_after
            psi = best_state[0]
            link_memory = {canonical_edge(*e): np.array(reg, copy=True) for e, reg in best_state[6].items()}
            accepted = compact(best_diag)
            accepted_moves.append(accepted)
            move_counts[str(best_diag["move_type"])] += 1
        else:
            move_counts["no_move"] += 1

        vis_nodes = [int(i) for i in range(len(sigma)) if float(sigma[i]) > float(args.sigma_on_threshold)]
        vis_edges = [list(e) for e, w in sorted(interface_commitment.items()) if float(w) > float(args.edge_on_threshold) and e[0] in vis_nodes and e[1] in vis_nodes]
        ent_rows = entanglement_sources(psi, sigma, set(vis_nodes), xp, scfg) if vis_nodes else []
        snapshots.append({
            "step": int(step),
            "visible_nodes": vis_nodes,
            "visible_edges": vis_edges,
            "mean_sigma": float(np.mean(sigma)),
            "top_entangled_pairs": ent_rows[:8],
            "candidate_count": int(len(diags)),
            "accepted_move": accepted,
        })

        if int(args.progress_every) > 0 and step % int(args.progress_every) == 0:
            print(
                f"[step {step:04d}] visible_nodes={len(vis_nodes):2d} visible_edges={len(vis_edges):2d} "
                f"mean_sigma={np.mean(sigma):.3f} winner={(accepted['move_type'] if accepted else None)} "
                f"deltaF={(accepted['deltaF'] if accepted else None)} cand={len(diags)}"
            )

    return {
        "script": "hsf_mesoscape_entanglement_local.py",
        "physics_config": asdict(pcfg),
        "bookkeeping_config": asdict(scfg),
        "run_config": {
            "sigma_on_threshold": float(args.sigma_on_threshold),
            "edge_on_threshold": float(args.edge_on_threshold),
            "sigma_step": float(args.sigma_step),
            "edge_step": float(args.edge_step),
            "max_parent_pairs": int(args.max_parent_pairs),
            "max_children_per_pair": int(args.max_children_per_pair),
            "max_lower_cands": int(args.max_lower_cands),
            "max_edge_up_cands": int(args.max_edge_up_cands),
            "max_edge_down_cands": int(args.max_edge_down_cands),
            "ent_threshold": float(args.ent_threshold),
            "principle": "birth candidates are generated only from entangled parent pairs; current note language governs local birth justification and differentiated-role bookkeeping",
        },
        "move_counts": move_counts,
        "final_sigma": [float(x) for x in sigma.tolist()],
        "final_interface_commitment": [
            {"edge": [int(e[0]), int(e[1])], "commitment": float(v)}
            for e, v in sorted(interface_commitment.items())
        ],
        "accepted_moves": accepted_moves,
        "snapshots": snapshots,
        "gpu_enabled": bool(is_gpu),
    }


def main() -> None:
    args = parse_args()
    pcfg, scfg = build_configs(args)
    # Fix accidental typo in build_configs fallback path without changing runtime behavior.
    scfg.w_mi = args.w_mi
    result = run_sim(pcfg, scfg, args)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()