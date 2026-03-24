# filename: hsf_factorization_accessibility_observer.py
#!/usr/bin/env python3
# filename: hsf_factorization_accessibility_observer.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import hsf_mesoscale_bookkeeping as bk
import hsf_mesoscale_physics_core as phys
from hsf_mesoscale_physics_core import PhysicsConfig, GM_MATRICES, apply_one_body


FactorDims = Tuple[int, ...]


@dataclass
class AccessibilityConfig:
    w_locality: float = 1.0
    w_signal_leakage: float = 1.0
    w_pair_entropy: float = 1.0
    w_anti_dominance: float = 1.0
    w_pair_count: float = 1.0
    w_core_persistence: float = 0.5
    w_connected_corr: float = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HSF factorization accessibility observer. "
            "Compares candidate factorizations of a fixed Hilbert space and reports which "
            "decompositions are most accessible / lawful under the present dynamics. "
            "This revision penalizes low-pair-count degeneracy and uses non-saturating "
            "pair-distribution metrics."
        )
    )
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--n-sites", type=int, default=4)
    p.add_argument("--site-dim", type=int, default=3)
    p.add_argument("--local-scale", type=float, default=0.15)
    p.add_argument("--pair-scale", type=float, default=0.12)
    p.add_argument("--spawn-pair-scale", type=float, default=0.11)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--total-steps", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=20)

    p.add_argument("--initial-state", choices=["basis_zero", "random", "perturbed_zero"], default="perturbed_zero")
    p.add_argument("--perturb-eps", type=float, default=0.02)

    p.add_argument(
        "--candidate-factorizations",
        type=str,
        default="",
        help=(
            "Optional explicit candidate factorizations, e.g. "
            "\"3x3x3x3;9x9;3x9x3\". "
            "If omitted, the script generates all nontrivial factorizations of total dimension."
        ),
    )

    p.add_argument("--w-locality", type=float, default=1.0)
    p.add_argument("--w-signal-leakage", type=float, default=1.0)
    p.add_argument("--w-pair-entropy", type=float, default=1.0)
    p.add_argument("--w-anti-dominance", type=float, default=1.0)
    p.add_argument("--w-pair-count", type=float, default=1.0)
    p.add_argument("--w-core-persistence", type=float, default=0.5)
    p.add_argument("--w-connected-corr", type=float, default=1.0)

    p.add_argument("--json-out", type=str, default="hsf_factorization_accessibility_observer.json")
    return p.parse_args()


def build_phys_config(args: argparse.Namespace) -> PhysicsConfig:
    return PhysicsConfig(
        n_max=args.n_sites,
        n_init=max(2, min(args.n_sites, 2)),
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


def build_accessibility_config(args: argparse.Namespace) -> AccessibilityConfig:
    return AccessibilityConfig(
        w_locality=float(args.w_locality),
        w_signal_leakage=float(args.w_signal_leakage),
        w_pair_entropy=float(args.w_pair_entropy),
        w_anti_dominance=float(args.w_anti_dominance),
        w_pair_count=float(args.w_pair_count),
        w_core_persistence=float(args.w_core_persistence),
        w_connected_corr=float(args.w_connected_corr),
    )


def random_state(shape: Sequence[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.normal(size=shape)
    imag = rng.normal(size=shape)
    arr = real + 1j * imag
    arr = arr / np.linalg.norm(arr.ravel())
    return arr.astype(np.complex128)


def perturbed_zero_state(shape: Sequence[int], seed: int, eps: float) -> np.ndarray:
    psi0 = np.zeros(shape, dtype=np.complex128)
    psi0[(0,) * len(shape)] = 1.0
    psi_rand = random_state(shape, seed)
    psi = (1.0 - eps) * psi0 + eps * psi_rand
    psi = psi / np.linalg.norm(psi.ravel())
    return psi.astype(np.complex128)


def make_initial_state(shape: Sequence[int], initial_state: str, seed: int, perturb_eps: float) -> np.ndarray:
    if initial_state == "basis_zero":
        psi = np.zeros(shape, dtype=np.complex128)
        psi[(0,) * len(shape)] = 1.0
        return psi
    if initial_state == "random":
        return random_state(shape, seed)
    if initial_state == "perturbed_zero":
        return perturbed_zero_state(shape, seed, perturb_eps)
    raise ValueError(initial_state)


def all_factorizations_of_dim(total_dim: int, min_parts: int = 2, max_parts: int = 6) -> List[FactorDims]:
    out: set[FactorDims] = set()

    def rec(remaining: int, start: int, prefix: List[int]) -> None:
        if remaining == 1:
            if min_parts <= len(prefix) <= max_parts:
                out.add(tuple(prefix))
            return
        for f in range(start, remaining + 1):
            if f > 1 and remaining % f == 0:
                prefix.append(f)
                rec(remaining // f, f, prefix)
                prefix.pop()

    rec(total_dim, 2, [])
    return sorted(out, key=lambda x: (len(x), x))


def parse_factorizations(text: str) -> List[FactorDims]:
    out: List[FactorDims] = []
    if not text.strip():
        return out
    for block in text.split(";"):
        dims = tuple(int(x.strip()) for x in block.split("x") if x.strip())
        if len(dims) >= 2:
            out.append(dims)
    return out


def reshape_state_to_factorization(flat_psi: np.ndarray, dims: FactorDims) -> np.ndarray:
    total = int(np.prod(dims))
    if flat_psi.size != total:
        raise ValueError(f"State size {flat_psi.size} incompatible with factorization {dims}")
    return flat_psi.reshape(dims)


def reduced_density_matrix(psi: np.ndarray, keep_axes: Sequence[int]) -> np.ndarray:
    keep_axes = tuple(sorted(int(ax) for ax in keep_axes))
    all_axes = tuple(range(psi.ndim))
    trace_axes = tuple(ax for ax in all_axes if ax not in keep_axes)

    perm = keep_axes + trace_axes
    psi_perm = np.transpose(psi, perm)

    keep_dim = int(np.prod([psi.shape[i] for i in keep_axes], dtype=int))
    trace_dim = int(np.prod([psi.shape[i] for i in trace_axes], dtype=int)) if trace_axes else 1

    psi_mat = psi_perm.reshape(keep_dim, trace_dim)
    rho = psi_mat @ np.conjugate(psi_mat.T)
    return rho


def von_neumann_entropy_from_rho(rho: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(np.real(vals), 0.0, 1.0)
    nz = vals[vals > 1e-15]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log(nz)))


def pair_mi_for_factorization(psi: np.ndarray) -> np.ndarray:
    n = psi.ndim
    mi = np.zeros((n, n), dtype=np.float64)
    single_ent = []
    for i in range(n):
        rho_i = reduced_density_matrix(psi, [i])
        single_ent.append(von_neumann_entropy_from_rho(rho_i))
    for i in range(n):
        for j in range(i + 1, n):
            rho_ij = reduced_density_matrix(psi, [i, j])
            sij = von_neumann_entropy_from_rho(rho_ij)
            mij = single_ent[i] + single_ent[j] - sij
            mi[i, j] = mi[j, i] = float(mij)
    return mi


def connected_pair_correlator_for_factorization(psi: np.ndarray) -> np.ndarray:
    n = psi.ndim
    corr = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            vals: List[float] = []
            for gm in GM_MATRICES:
                gm_arr = np.asarray(gm, dtype=np.complex128)
                oi = apply_one_body(psi, gm_arr, i, np)
                oj = apply_one_body(psi, gm_arr, j, np)
                e_i = np.real(np.vdot(psi, oi))
                e_j = np.real(np.vdot(psi, oj))
                oij = apply_one_body(oj, gm_arr, i, np)
                e_ij = np.real(np.vdot(psi, oij))
                vals.append(float(abs(e_ij - e_i * e_j)))
            corr[i, j] = corr[j, i] = float(np.mean(vals) if vals else 0.0)
    return corr


def pair_weights(combo: np.ndarray) -> np.ndarray:
    vals = combo[np.triu_indices_from(combo, k=1)]
    vals = np.clip(vals.astype(np.float64), 0.0, None)
    return vals


def pair_count_score(num_pairs: int, max_pairs: int) -> float:
    if max_pairs <= 1:
        return 0.0
    if num_pairs <= 1:
        return 0.0
    return float(np.log(num_pairs) / np.log(max_pairs))


def pair_entropy_score(weights: np.ndarray) -> float:
    num_pairs = int(weights.size)
    if num_pairs <= 1:
        return 0.0
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    p = weights / total
    nz = p[p > 1e-15]
    h = float(-np.sum(nz * np.log(nz)))
    return float(h / np.log(num_pairs))


def anti_dominance_score(weights: np.ndarray) -> float:
    num_pairs = int(weights.size)
    if num_pairs <= 1:
        return 0.0
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    max_share = float(np.max(weights) / total)
    uniform_share = 1.0 / num_pairs
    if max_share <= uniform_share:
        return 1.0
    return float(1.0 - (max_share - uniform_share) / (1.0 - uniform_share))


def locality_like_score(mi: np.ndarray, corr: np.ndarray) -> float:
    combo = mi + corr
    vals = pair_weights(combo)
    if vals.size == 0:
        return 0.0
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if mean <= 0.0:
        return 0.0
    return float(mean / (mean + std))


def signaling_leakage_score(mi: np.ndarray) -> float:
    vals = pair_weights(mi)
    if vals.size <= 1:
        return 0.0
    total = float(np.sum(vals))
    if total <= 0.0:
        return 1.0
    max_val = float(np.max(vals))
    uniform_share = 1.0 / vals.size
    max_share = max_val / total
    if max_share <= uniform_share:
        return 1.0
    return float(1.0 - (max_share - uniform_share) / (1.0 - uniform_share))


def dominant_pair(mat: np.ndarray) -> Optional[Tuple[int, int]]:
    n = mat.shape[0]
    best = None
    best_val = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] > best_val:
                best_val = float(mat[i, j])
                best = (i, j)
    return best


def core_persistence_score(core_pairs: List[Optional[Tuple[int, int]]], num_pairs: int) -> float:
    if num_pairs <= 1 or not core_pairs:
        return 0.0
    counts: Dict[Tuple[int, int], int] = {}
    total = 0
    for cp in core_pairs:
        if cp is None:
            continue
        counts[cp] = counts.get(cp, 0) + 1
        total += 1
    if total == 0:
        return 0.0
    raw = float(max(counts.values()) / total)
    # Discount trivial persistence when pair count is small.
    return raw * float(np.log(num_pairs) / np.log(max(num_pairs, 2)))


def score_factorization(
    snapshots: List[Dict[str, Any]],
    acc_cfg: AccessibilityConfig,
    max_pairs_in_grid: int,
) -> Dict[str, Any]:
    locality_vals = [float(s["locality_like_score"]) for s in snapshots]
    leakage_vals = [float(s["signal_leakage_score"]) for s in snapshots]
    entropy_vals = [float(s["pair_entropy_score"]) for s in snapshots]
    anti_dom_vals = [float(s["anti_dominance_score"]) for s in snapshots]
    count_vals = [float(s["pair_count_score"]) for s in snapshots]
    cc_vals = [float(s["mean_connected_corr"]) for s in snapshots]
    core_persist_vals = [float(s["core_persistence_score"]) for s in snapshots]

    locality = float(np.mean(locality_vals)) if locality_vals else 0.0
    leakage = float(np.mean(leakage_vals)) if leakage_vals else 0.0
    entropy = float(np.mean(entropy_vals)) if entropy_vals else 0.0
    anti_dom = float(np.mean(anti_dom_vals)) if anti_dom_vals else 0.0
    count_score = float(np.mean(count_vals)) if count_vals else 0.0
    cc_mean = float(np.mean(cc_vals)) if cc_vals else 0.0
    persistence = float(np.mean(core_persist_vals)) if core_persist_vals else 0.0

    total = (
        acc_cfg.w_locality * locality
        + acc_cfg.w_signal_leakage * leakage
        + acc_cfg.w_pair_entropy * entropy
        + acc_cfg.w_anti_dominance * anti_dom
        + acc_cfg.w_pair_count * count_score
        + acc_cfg.w_core_persistence * persistence
        + acc_cfg.w_connected_corr * cc_mean
    )

    return {
        "mean_locality_like_score": locality,
        "mean_signal_leakage_score": leakage,
        "mean_pair_entropy_score": entropy,
        "mean_anti_dominance_score": anti_dom,
        "mean_pair_count_score": count_score,
        "mean_connected_corr": cc_mean,
        "mean_core_persistence_score": persistence,
        "accessibility_score": float(total),
    }


def snapshot_for_factorization(
    flat: np.ndarray,
    dims: FactorDims,
    step: int,
    max_pairs_in_grid: int,
) -> Dict[str, Any]:
    psi_fac = reshape_state_to_factorization(flat, dims)
    mi = pair_mi_for_factorization(psi_fac)
    cc = connected_pair_correlator_for_factorization(psi_fac)
    combo = mi + cc
    weights = pair_weights(combo)
    num_pairs = int(weights.size)

    return {
        "step": int(step),
        "factorization": list(dims),
        "num_factors": int(len(dims)),
        "num_pairs": num_pairs,
        "mean_mi": float(np.mean(pair_weights(mi))) if num_pairs > 0 else 0.0,
        "mean_connected_corr": float(np.mean(pair_weights(cc))) if num_pairs > 0 else 0.0,
        "pair_entropy_score": pair_entropy_score(weights),
        "anti_dominance_score": anti_dominance_score(weights),
        "pair_count_score": pair_count_score(num_pairs, max_pairs_in_grid),
        "locality_like_score": locality_like_score(mi, cc),
        "signal_leakage_score": signaling_leakage_score(mi),
        "dominant_pair": list(dominant_pair(combo)) if dominant_pair(combo) is not None else None,
        "core_persistence_score": 0.0,  # filled later per factorization summary
    }


def main() -> None:
    args = parse_args()
    phys_cfg = build_phys_config(args)
    acc_cfg = build_accessibility_config(args)

    total_dim = int(args.site_dim ** args.n_sites)
    explicit = parse_factorizations(args.candidate_factorizations)
    factorizations = explicit if explicit else all_factorizations_of_dim(total_dim)

    if not factorizations:
        raise ValueError("No candidate factorizations available.")

    max_pairs_in_grid = max((len(f) * (len(f) - 1)) // 2 for f in factorizations)

    xp, is_gpu = phys.get_array_module(args.device)

    base_shape = (args.site_dim,) * args.n_sites
    psi0 = make_initial_state(base_shape, args.initial_state, args.seed, args.perturb_eps)
    psi0_xp = xp.asarray(psi0, dtype=xp.complex128)

    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs, _rng = phys.init_state(phys_cfg, xp)
    psi = psi0_xp
    prepared = (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs)

    flat_snapshots: List[np.ndarray] = []
    snapshot_steps: List[int] = []

    for step in range(1, args.total_steps + 1):
        prepared = phys.evolve_prepared_state(prepared, phys_cfg, xp)
        psi_cur = np.asarray(prepared[0]).reshape(-1).copy()

        if step % args.snapshot_every == 0 or step == 1 or step == args.total_steps:
            flat_snapshots.append(psi_cur)
            snapshot_steps.append(step)

        if args.progress_every > 0 and step % args.progress_every == 0:
            print(f"[step {step:04d}] stored_snapshots={len(flat_snapshots)}")

    factorization_results: List[Dict[str, Any]] = []

    for dims in factorizations:
        factor_snaps: List[Dict[str, Any]] = []
        core_pairs: List[Optional[Tuple[int, int]]] = []

        for flat, step in zip(flat_snapshots, snapshot_steps):
            snap = snapshot_for_factorization(flat, dims, step, max_pairs_in_grid)
            factor_snaps.append(snap)

            dp = snap["dominant_pair"]
            core_pairs.append(None if dp is None else (int(dp[0]), int(dp[1])))

        num_pairs = int(factor_snaps[0]["num_pairs"]) if factor_snaps else 0
        persistence = core_persistence_score(core_pairs, num_pairs)
        for snap in factor_snaps:
            snap["core_persistence_score"] = persistence

        summary = score_factorization(factor_snaps, acc_cfg, max_pairs_in_grid)

        factorization_results.append(
            {
                "factorization": list(dims),
                "summary": summary,
                "snapshots": factor_snaps,
            }
        )

    factorization_results.sort(
        key=lambda r: float(r["summary"]["accessibility_score"]),
        reverse=True,
    )

    result = {
        "script": "hsf_factorization_accessibility_observer.py",
        "physics_config": asdict(phys_cfg),
        "accessibility_config": asdict(acc_cfg),
        "initial_state": args.initial_state,
        "perturb_eps": float(args.perturb_eps),
        "total_dimension": total_dim,
        "candidate_factorizations": [list(f) for f in factorizations],
        "max_pairs_in_grid": int(max_pairs_in_grid),
        "results": factorization_results,
        "best_factorization": factorization_results[0]["factorization"] if factorization_results else None,
        "gpu_enabled": bool(is_gpu),
    }

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()