#!/usr/bin/env python3
"""
Experiment 4: Factorization Selection Under Constraints (No Assumed Structure)
=============================================================================

This script tests whether three operational constraints can discriminate among
factorizations when the Hamiltonian is structureless (random Hermitian).

Constraints (operational proxies):
- No-signaling-ish: influence does not become globally detectable instantly.
- No-forgetting-ish (NONTRIVIAL): a *localized record* forms (stronger at some site
  than elsewhere) and persists at late times.
- No-refolding-ish: measured indirectly via persistence/heterogeneity metrics
  (rank stability + HIP-like variance).

IMPORTANT CHANGE (record-like no-forgetting)
--------------------------------------------
Old metric (trivial under scrambling):
    frac_recover = mean_t [ max_site influence(t, site) >= threshold ]

New metric (nontrivial):
    signal_r(t) = influence(t,r) - mean_{j!=r} influence(t,j)
    frac_record = mean_{t in late} [ signal_best(t) >= threshold ]
where "best" is the site r!=source with the highest mean signal over late times.

Bugfix (2026-01-13)
-------------------
Fixed partial_trace() tracing axes accounting for tensor rank reduction.
Also made rank_stability robust (avoid NaNs / warnings when ranks are constant or
n_sites < 2).
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Tuple

import numpy as np


# =============================================================================
# FACTORIZATIONS
# =============================================================================

def get_factorizations(D: int) -> List[Tuple[int, ...]]:
    """All unordered factorizations of D into integers >= 2."""
    out = set()

    def factor(n: int, min_f: int, acc: Tuple[int, ...]):
        if n == 1:
            out.add(tuple(sorted(acc)))
            return
        for f in range(min_f, n + 1):
            if f < 2:
                continue
            if n % f == 0:
                factor(n // f, f, acc + (f,))

    factor(D, 2, tuple())
    return sorted(out, key=lambda x: (len(x), x))


def factorization_label(f: Tuple[int, ...]) -> str:
    return "⊗".join(str(d) for d in f)


# =============================================================================
# RANDOM HAMILTONIAN + EVOLUTION
# =============================================================================

def random_hermitian(D: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Random dense Hermitian matrix."""
    A = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    H = (A + A.conj().T) / 2.0
    return scale * H


def diagonalize(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose H = V diag(E) V†."""
    E, V = np.linalg.eigh(H)
    return E, V


def evolve(psi0: np.ndarray, E: np.ndarray, V: np.ndarray, t: float) -> np.ndarray:
    """psi(t) = V exp(-iEt) V† psi0."""
    coeff = V.conj().T @ psi0
    phase = np.exp(-1j * E * t)
    psi_t = V @ (phase * coeff)
    return psi_t


# =============================================================================
# STATES + LOCAL PERTURBATION UNDER A FACTORIZATION
# =============================================================================

def random_product_state(factorization: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """
    Random product state with respect to the given factorization.
    For each site of dimension d: sample a random complex vector and normalize,
    then take the Kronecker product.
    """
    psi = np.array([1.0 + 0j])
    for d in factorization:
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        v = v / np.linalg.norm(v)
        psi = np.kron(psi, v)
    psi = psi / np.linalg.norm(psi)
    return psi


def local_perturbation(factorization: Tuple[int, ...], site: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Random unitary on one site, identity elsewhere."""
    d = factorization[site]

    # Random U(d) via QR
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    # Fix phases to make Q unitary “nice”
    Q = Q * np.exp(-1j * np.angle(np.diag(R)))

    dims_before = int(np.prod(factorization[:site])) if site > 0 else 1
    dims_after = int(np.prod(factorization[site + 1:])) if site < len(factorization) - 1 else 1

    U = np.kron(np.kron(np.eye(dims_before), Q), np.eye(dims_after))
    return U


# =============================================================================
# PARTIAL TRACE + TRACE DISTANCE
# =============================================================================

def partial_trace(rho: np.ndarray, factorization: Tuple[int, ...], keep_sites: List[int]) -> np.ndarray:
    """
    Partial trace over all sites not in keep_sites.
    Returns reduced density matrix on keep_sites.

    BUGFIX: after each trace, tensor rank shrinks; we must update n accordingly.
    """
    keep_sites = sorted(keep_sites)
    n_sites0 = len(factorization)
    dims0 = list(factorization)

    # reshape into 2n tensor indices: (s1..sn, s1'..sn')
    rho_t = rho.reshape(dims0 + dims0)

    trace_sites = [i for i in range(n_sites0) if i not in keep_sites]

    dims = dims0[:]          # mutable current dims
    n = n_sites0             # mutable current site count

    # Trace in descending index order so earlier pops don't shift later indices
    for s in sorted(trace_sites, reverse=True):
        # axes are (s) in ket indices and (s+n) in bra indices for the CURRENT n
        rho_t = np.trace(rho_t, axis1=s, axis2=s + n)
        dims.pop(s)
        n -= 1

    keep_dims = [dims0[i] for i in keep_sites]
    d_keep = int(np.prod(keep_dims)) if keep_dims else 1
    return rho_t.reshape(d_keep, d_keep)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """0.5 * ||rho - sigma||_1 using eigenvalues for Hermitian delta."""
    delta = rho - sigma
    delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))


# =============================================================================
# METRICS
# =============================================================================

def measure_influence_spread(
    psi_base: np.ndarray,
    psi_pert: np.ndarray,
    E: np.ndarray,
    V: np.ndarray,
    factorization: Tuple[int, ...],
    times: np.ndarray
) -> np.ndarray:
    """
    influence[t, site] = trace_distance( rho_base_site(t), rho_pert_site(t) )
    where rho_site is the reduced state for that site alone.
    """
    n_sites = len(factorization)
    influence = np.zeros((len(times), n_sites), dtype=np.float64)

    for ti, t in enumerate(times):
        psi0 = evolve(psi_base, E, V, float(t))
        psi1 = evolve(psi_pert, E, V, float(t))

        rho0 = np.outer(psi0, psi0.conj())
        rho1 = np.outer(psi1, psi1.conj())

        for s in range(n_sites):
            red0 = partial_trace(rho0, factorization, [s])
            red1 = partial_trace(rho1, factorization, [s])
            influence[ti, s] = trace_distance(red0, red1)

    return influence


def safe_rank_stability_from_influence(influence_avg: np.ndarray) -> float:
    """
    Compute rank stability of sites over time robustly.
    Returns ~[0,1] where higher means more stable ordering.
    """
    T, n_sites = influence_avg.shape
    if T < 2 or n_sites < 2:
        return 0.0

    # ranks[t, i]
    ranks = np.argsort(np.argsort(influence_avg, axis=1), axis=1).astype(np.float64)

    cors = []
    for t in range(T - 1):
        x = ranks[t]
        y = ranks[t + 1]
        # If either is constant, correlation is undefined; treat as no information (skip)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        # Pearson correlation
        x0 = x - x.mean()
        y0 = y - y.mean()
        denom = (np.linalg.norm(x0) * np.linalg.norm(y0))
        if denom < 1e-12:
            continue
        cors.append(float((x0 @ y0) / denom))

    if not cors:
        return 0.0
    # Map [-1,1] -> [0,1] later if desired by caller; here we keep [-1,1]
    return float(np.mean(cors))


def evaluate_factorization(
    H: np.ndarray,
    E: np.ndarray,
    V: np.ndarray,
    factorization: Tuple[int, ...],
    rng: np.random.Generator,
    times: np.ndarray,
    n_trials: int = 3,
    recover_threshold: float = 0.05,
    speed_threshold: float = 0.02
) -> Dict:
    """
    Evaluate factorization under constraints.

    Returns dict:
      - mean_recover: record-like localized recoverability (0..1)
      - mean_t_half: activation time proxy (None if never reaches threshold)
      - hip: average variance across sites of influence(t,:)
      - rank_stability: mean rank correlation across time ([-1,1] typically)
      - score: combined objective
    """
    n_sites = len(factorization)

    all_influence = []
    all_recover = []
    all_t_half = []

    for _trial in range(n_trials):
        psi_base = random_product_state(factorization, rng)

        # Perturb source site 0
        U_pert = local_perturbation(factorization, 0, rng)
        psi_pert = U_pert @ psi_base
        psi_pert = psi_pert / np.linalg.norm(psi_pert)

        influence = measure_influence_spread(psi_base, psi_pert, E, V, factorization, times)
        all_influence.append(influence)

        # Record-like no-forgetting
        source_site = 0
        if n_sites < 2:
            frac_record = 1.0
        else:
            t_late0 = 0.60 * float(times[-1])
            late_idx = np.where(times >= t_late0)[0]
            if late_idx.size == 0:
                late_idx = np.arange(len(times))

            infl_sum = influence.sum(axis=1)

            best_mean_signal = -1e9
            best_frac_record = 0.0

            for r in range(n_sites):
                if r == source_site:
                    continue
                mean_other = (infl_sum - influence[:, r]) / float(n_sites - 1)
                signal = influence[:, r] - mean_other
                mean_signal = float(signal[late_idx].mean())
                frac = float(np.mean(signal[late_idx] >= recover_threshold))
                if mean_signal > best_mean_signal:
                    best_mean_signal = mean_signal
                    best_frac_record = frac

            frac_record = best_frac_record

        all_recover.append(frac_record)

        # No-signaling-ish activation time (>= half sites above speed_threshold)
        activated = (influence >= speed_threshold).sum(axis=1)
        half = max(1, n_sites // 2)
        if np.any(activated >= half):
            t_half = float(times[np.argmax(activated >= half)])
        else:
            t_half = float("inf")
        all_t_half.append(t_half)

    mean_recover = float(np.mean(all_recover))

    finite_ts = [t for t in all_t_half if np.isfinite(t)]
    mean_t_half = float(np.mean(finite_ts)) if finite_ts else float("inf")

    influence_avg = np.mean(np.array(all_influence), axis=0)  # (T, n_sites)

    hip = float(np.mean(np.var(influence_avg, axis=1)))

    rank_stability = safe_rank_stability_from_influence(influence_avg)  # [-1,1] or 0 if undefined

    # Normalize t_half to [0,1]
    if not np.isfinite(mean_t_half) or float(times[-1]) <= 0:
        t_half_norm = 0.0
    else:
        t_half_norm = max(0.0, min(1.0, mean_t_half / float(times[-1])))

    # mild coarseness penalty so n_sites=1 doesn't always dominate when trivial
    coarseness_penalty = 0.02 * (1.0 / max(1, n_sites))

    score = (
        1.5 * mean_recover +
        0.7 * t_half_norm +
        0.6 * ((rank_stability + 1.0) / 2.0) +   # map [-1,1] -> [0,1]
        0.3 * math.tanh(5.0 * hip) -
        coarseness_penalty
    )

    return {
        "factorization": list(factorization),
        "label": factorization_label(factorization),
        "n_sites": n_sites,
        "mean_recover": float(mean_recover),
        "mean_t_half": None if not np.isfinite(mean_t_half) else float(mean_t_half),
        "hip": float(hip),
        "rank_stability": float(rank_stability),
        "score": float(score),
        "is_qubit": all(d == 2 for d in factorization),
        "is_coarsest": n_sites == 2,
        "max_local_dim": int(max(factorization)),
        "min_local_dim": int(min(factorization)),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test factorization selection under constraints (no assumed structure)"
    )
    parser.add_argument("--D", type=int, default=16, help="Hilbert space dimension (e.g., 16)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--tmax", type=float, default=1.0, help="Max evolution time")
    parser.add_argument("--nt", type=int, default=11, help="Number of time samples")
    parser.add_argument("--trials", type=int, default=3, help="Trials per factorization")
    parser.add_argument("--recover-threshold", type=float, default=0.05,
                        help="Threshold on record-signal for 'no-forgetting'")
    parser.add_argument("--speed-threshold", type=float, default=0.02,
                        help="Threshold for activation in 'no-signaling'")
    parser.add_argument("--H-scale", type=float, default=1.0, help="Scale of random Hermitian")
    parser.add_argument("--out", type=str, default="factorization_constraints_results_v2.json",
                        help="Output JSON path")
    parser.add_argument("--progress", action="store_true", help="Print progress + winners")
    args = parser.parse_args()

    D = int(args.D)
    factorizations = get_factorizations(D)
    times = np.linspace(0.0, float(args.tmax), int(args.nt), dtype=np.float64)

    all_runs = []
    winners = []

    for seed in range(int(args.seeds)):
        rng = np.random.default_rng(seed)
        H = random_hermitian(D, rng, scale=float(args.H_scale))
        E, V = diagonalize(H)

        results = []
        for f in factorizations:
            res = evaluate_factorization(
                H=H, E=E, V=V,
                factorization=f,
                rng=rng,
                times=times,
                n_trials=int(args.trials),
                recover_threshold=float(args.recover_threshold),
                speed_threshold=float(args.speed_threshold),
            )
            results.append(res)

        results_sorted = sorted(results, key=lambda d: d["score"], reverse=True)
        best = results_sorted[0]
        winners.append(best)

        all_runs.append({
            "seed": seed,
            "best_factorization": best["label"],
            "best_is_qubit": bool(best["is_qubit"]),
            "best_is_coarsest": bool(best["is_coarsest"]),
            "results": results_sorted,
        })

        if args.progress:
            print(f"[seed {seed}] winner = {best['label']}  score={best['score']:+.3f}  "
                  f"record={best['mean_recover']:.3f}  t_half={best['mean_t_half']}  "
                  f"stab={best['rank_stability']:.3f} hip={best['hip']:.4f}")

    n_sites_wins: Dict[int, int] = {}
    for w in winners:
        n_sites_wins[w["n_sites"]] = n_sites_wins.get(w["n_sites"], 0) + 1

    summary = {
        "meta": {
            "D": D,
            "seeds": int(args.seeds),
            "tmax": float(args.tmax),
            "nt": int(args.nt),
            "trials": int(args.trials),
            "recover_threshold": float(args.recover_threshold),
            "speed_threshold": float(args.speed_threshold),
            "H_scale": float(args.H_scale),
            "note": "mean_recover is NONTRIVIAL record-like metric (site signal above others over late times).",
        },
        "best_factorizations": [w["label"] for w in winners],
        "n_sites_wins": n_sites_wins,
        "runs": all_runs,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.progress:
        print("\nSUMMARY")
        print("best_factorizations:", summary["best_factorizations"])
        print("n_sites_wins:", summary["n_sites_wins"])
        print(f"Wrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
