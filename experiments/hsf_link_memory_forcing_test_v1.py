#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hsf_link_memory_forcing_test_v1.py
==================================

Goal
----
Test (in a controlled toy model) whether a *bidirectional link Hilbert space* is dynamically
forced by the HSF-style requirements that a link:

  - is an OBJECT (not just a coupling),
  - transmits influence locally (no-signaling),
  - preserves information globally (no-forgetting / unitarity),
  - has limited per-step capacity (finite bandwidth),
  - and accumulates persistent "echo" state (here: link memory state changes and persists).

This script implements a minimal, falsifiable proxy:

  We build a 2-endpoint system (Left site L, Right site R) with a link-memory subsystem E
  that mediates influence via LOCAL sequential unitaries:

      U_step = exp(-i dt H_LE) exp(-i dt H_ER)

  (no-signaling / locality is enforced by only touching (L,E) and then (E,R) each step)

  Global evolution is unitary (no-forgetting is built-in).

  Finite bandwidth is modeled by limiting the number of coupled generator channels per end.

Then we run an "echo algebra probe" similar in spirit to your step-1 extraction:
  - Fix a baseline initial state |0>_L |0>_E |0>_R
  - For each su(d) generator on L (and separately on R), apply a small poke unitary
    at that endpoint, then run one step U_step, then look at the induced change on E.
  - From those induced changes, build an 8D (for d=3) response-operator set on E and
    check:
      * basis_dim (rank) of the span
      * singular values of the response map
      * whether the extracted LEFT and RIGHT spans commute (cross commutator stats)

Interpretation
--------------
If E truly has a bidirectional structure capable of independent endpoint actions (a bimodule),
we expect:
  - LEFT span ~ su(3) (dim 8)
  - RIGHT span ~ su(3) (dim 8)
  - and importantly: [LEFT, RIGHT] ~ 0 (commuting actions), up to numerical noise.

In a minimal faithful representation, this typically requires link memory dimension >= d^2.

In this script we demonstrate the "forced" direction by comparing:
  - env_dim = 3 (single qutrit): cannot host two commuting su(3) actions
  - env_dim = 6 (intermediate): usually still fails robust commuting structure
  - env_dim = 9 (d^2): we can hide a true left/right factorization inside topology
    by scrambling the basis (random unitary), and the probe should recover commuting spans.

Crucially: for env_dim=9 we DO NOT expose factorization to the probe. We scramble it
to imitate "topology basis" coordinates.

Run (Windows one-liners)
------------------------
python hsf_link_memory_forcing_test_v1.py --d 3 --env_dims 3,6,9 --dt 0.10 --bandwidth 8 --poke_eps 1e-3 --seed 0 --trials 8

Outputs
-------
Writes JSON into ./hsf_out/hsf_link_memory_forcing_test_v1_<timestamp>.json

Dependencies
------------
numpy, scipy
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.linalg import svd

from scipy.linalg import expm


# -------------------------
# Utilities
# -------------------------

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_out_dir() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def hermitize(A: np.ndarray) -> np.ndarray:
    return (A + A.conj().T) / 2.0

def traceless(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d, dtype=complex)

def hs_inner(A: np.ndarray, B: np.ndarray) -> complex:
    return np.trace(A.conj().T @ B)

def hs_norm(A: np.ndarray) -> float:
    return float(np.sqrt(max(hs_inner(A, A).real, 0.0)))

def normalize_hs(A: np.ndarray, tol: float = 1e-30) -> np.ndarray:
    n = hs_norm(A)
    if n < tol:
        return A.copy()
    return A / n

def gram_schmidt_hs(basis, tol=1e-12):
    out = []
    for A in basis:
        B = A.copy()
        for Q in out:
            B -= hs_inner(Q, B) * Q
        n = hs_norm(B)
        if n > tol:
            out.append(B / n)
    return out

def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(X)
    diag = np.diag(R)
    ph = diag / np.where(np.abs(diag) > 0, np.abs(diag), 1.0)
    Q = Q * ph
    return Q

def kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out


# -------------------------
# su(d) Hermitian generator basis (HS-orthonormal)
# -------------------------

def su_generators(d: int):
    """
    HS-orthonormal Hermitian traceless basis for su(d). For d=3 -> 8 generators.
    """
    gens = []

    # symmetric/antisymmetric off-diagonals
    for i in range(d):
        for j in range(i + 1, d):
            S = np.zeros((d, d), dtype=complex)
            S[i, j] = 1.0
            S[j, i] = 1.0

            A = np.zeros((d, d), dtype=complex)
            A[i, j] = -1j
            A[j, i] = 1j

            gens.append(S)
            gens.append(A)

    # diagonal traceless (d-1)
    for k in range(1, d):
        D = np.zeros((d, d), dtype=complex)
        for i in range(k):
            D[i, i] = 1.0
        D[k, k] = -float(k)
        D = D * math.sqrt(2.0 / (k * (k + 1.0)))
        gens.append(D)

    gens = [normalize_hs(traceless(hermitize(G))) for G in gens]
    gens = gram_schmidt_hs(gens, tol=1e-12)
    return gens


# -------------------------
# Model: sequential local unitaries L-E then E-R
# -------------------------

@dataclass
class ModelSpec:
    d: int = 3
    env_dim: int = 3
    dt: float = 0.10
    bandwidth: int = 8       # number of coupled channels per end (<= d^2-1 for d=3 -> 8)
    seed: int = 0
    trial: int = 0
    topology_scramble: bool = True  # for env_dim==d^2, hide factorization by scrambling


def build_hidden_bidirectional_env_ops(d: int, rng: np.random.Generator):
    """
    For env_dim = d^2, build a hidden factorization E ≅ C^d ⊗ C^d (operator-space link).
    Then provide two commuting su(d) action sets on E:
      L^a = T^a ⊗ I
      R^a = I ⊗ (T^a)^T
    Finally scramble the basis by a random unitary S so the probe doesn't "see" factorization.
    """
    T = su_generators(d)  # list of (d^2-1) Hermitian
    I = np.eye(d, dtype=complex)

    L_ops = [kron(Ta, I) for Ta in T]
    R_ops = [kron(I, Ta.T) for Ta in T]  # transpose is the natural right action in vec convention

    # Scramble: O -> S O S^\dagger
    S = random_unitary(d * d, rng)
    L_ops = [S @ O @ S.conj().T for O in L_ops]
    R_ops = [S @ O @ S.conj().T for O in R_ops]

    # normalize HS again (harmless)
    L_ops = [normalize_hs(traceless(hermitize(O))) for O in L_ops]
    R_ops = [normalize_hs(traceless(hermitize(O))) for O in R_ops]
    return L_ops, R_ops, S


def build_generic_env_ops(env_dim: int, n_ops: int, rng: np.random.Generator):
    """
    Build random Hermitian traceless operators on E, HS-orthonormalized.
    Used when env_dim != d^2 (no hidden bimodule structure).
    """
    ops = []
    for _ in range(max(n_ops, 1) * 3):
        X = rng.normal(size=(env_dim, env_dim)) + 1j * rng.normal(size=(env_dim, env_dim))
        H = hermitize(X)
        H = traceless(H)
        ops.append(H)
        if len(ops) >= n_ops * 2:
            break
    ops = [normalize_hs(O) for O in ops]
    ops = gram_schmidt_hs(ops, tol=1e-12)
    return ops[:n_ops]


def build_local_hamiltonians(spec: ModelSpec, rng: np.random.Generator):
    """
    Construct:
      H_LE = sum_{a=1..k}  T_L^a ⊗ E_L^a
      H_ER = sum_{a=1..k}  E_R^a ⊗ T_R^a

    where k = bandwidth (channels).
    For env_dim == d^2, we use hidden commuting L/R operator sets on E.
    For other env_dim, we use generic random E operators for both ends (which generally overlap).
    """
    d = spec.d
    env_dim = spec.env_dim
    k = int(spec.bandwidth)

    T = su_generators(d)
    k = min(k, len(T))

    if env_dim == d * d and spec.topology_scramble:
        E_left_ops, E_right_ops, _S = build_hidden_bidirectional_env_ops(d, rng)
        E_left = E_left_ops[:k]
        E_right = E_right_ops[:k]
        env_mode = "hidden_bimodule_scrambled"
    else:
        # Generic env operators: left and right draw from same operator pool => usually noncommuting overlap
        pool = build_generic_env_ops(env_dim, n_ops=min(env_dim * env_dim - 1, max(k, 1)), rng=rng)
        if len(pool) < k:
            # pad if needed
            pool = pool + build_generic_env_ops(env_dim, n_ops=k - len(pool), rng=rng)
        # left picks first k; right picks a rotated mix (still living in same space)
        E_left = pool[:k]
        # mix for right
        mix = rng.normal(size=(k, k))
        Q, _ = np.linalg.qr(mix)
        E_right = []
        for a in range(k):
            H = np.zeros((env_dim, env_dim), dtype=complex)
            for b in range(k):
                H += Q[b, a] * E_left[b]
            E_right.append(normalize_hs(traceless(hermitize(H))))
        env_mode = "generic_overlap"

    # Build H_LE on L⊗E and H_ER on E⊗R
    H_LE = np.zeros((d * env_dim, d * env_dim), dtype=complex)
    H_ER = np.zeros((env_dim * d, env_dim * d), dtype=complex)

    for a in range(k):
        H_LE += kron(T[a], E_left[a])
        H_ER += kron(E_right[a], T[a])

    # Scale so dt is the real knob
    H_LE = hermitize(H_LE)
    H_ER = hermitize(H_ER)

    return H_LE, H_ER, env_mode, k


def step_unitary(spec: ModelSpec, H_LE: np.ndarray, H_ER: np.ndarray):
    """
    Build U_step on L⊗E⊗R:
      U = (exp(-i dt H_LE) ⊗ I_R)  then  (I_L ⊗ exp(-i dt H_ER))
    """
    d = spec.d
    e = spec.env_dim
    U_LE = expm(-1j * spec.dt * H_LE)               # on L⊗E
    U_ER = expm(-1j * spec.dt * H_ER)               # on E⊗R

    I_L = np.eye(d, dtype=complex)
    I_R = np.eye(d, dtype=complex)

    U1 = kron(U_LE, I_R)                            # (L⊗E)⊗R
    U2 = kron(I_L, U_ER)                            # L⊗(E⊗R)

    return U2 @ U1


# -------------------------
# Echo probe: poke endpoint, run one step, measure induced operator on E
# -------------------------

def pure0(d: int) -> np.ndarray:
    v = np.zeros((d,), dtype=complex)
    v[0] = 1.0
    return v

def density_from_state(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())

def partial_trace_rho(rho: np.ndarray, dims: tuple[int, int, int], keep: str) -> np.ndarray:
    """
    Partial trace for 3-part system (L,E,R).
    keep in {"L","E","R"} returns the reduced density.
    """
    dL, dE, dR = dims
    rho = rho.reshape((dL, dE, dR, dL, dE, dR))

    if keep == "E":
        # trace L and R
        out = np.zeros((dE, dE), dtype=complex)
        for iL in range(dL):
            for iR in range(dR):
                out += rho[iL, :, iR, iL, :, iR]
        return out
    if keep == "L":
        out = np.zeros((dL, dL), dtype=complex)
        for iE in range(dE):
            for iR in range(dR):
                out += rho[:, iE, iR, :, iE, iR]
        return out
    if keep == "R":
        out = np.zeros((dR, dR), dtype=complex)
        for iL in range(dL):
            for iE in range(dE):
                out += rho[iL, iE, :, iL, iE, :]
        return out
    raise ValueError("keep must be L,E,R")

def response_operators_on_E(spec: ModelSpec, U: np.ndarray, poke_eps: float):
    """
    For each su(d) generator on L and on R:
      apply poke exp(-i eps T^a) to that endpoint
      run one step U
      compute delta rho_E = rho_E(poked) - rho_E(base)

    Convert each delta rho_E into a Hermitian traceless operator basis candidate.
    Return left_ops, right_ops (lists of matrices on E).
    """
    d = spec.d
    e = spec.env_dim
    dims = (d, e, d)

    T = su_generators(d)
    k = len(T)

    # base initial state
    psi0 = kron(pure0(d), pure0(e), pure0(d))
    rho0 = density_from_state(psi0)

    # evolve base
    psi_base = U @ psi0
    rho_base = density_from_state(psi_base)
    rhoE_base = partial_trace_rho(rho_base, dims, keep="E")

    left_ops = []
    right_ops = []

    # poke left
    for a in range(k):
        UL = expm(-1j * poke_eps * T[a])
        poke = kron(UL, np.eye(e, dtype=complex), np.eye(d, dtype=complex))
        psi = U @ (poke @ psi0)
        rho = density_from_state(psi)
        rhoE = partial_trace_rho(rho, dims, keep="E")
        dE = hermitize(rhoE - rhoE_base)
        dE = traceless(dE)
        left_ops.append(dE)

    # poke right
    for a in range(k):
        UR = expm(-1j * poke_eps * T[a])
        poke = kron(np.eye(d, dtype=complex), np.eye(e, dtype=complex), UR)
        psi = U @ (poke @ psi0)
        rho = density_from_state(psi)
        rhoE = partial_trace_rho(rho, dims, keep="E")
        dE = hermitize(rhoE - rhoE_base)
        dE = traceless(dE)
        right_ops.append(dE)

    return left_ops, right_ops


# -------------------------
# Linear span diagnostics
# -------------------------

def span_rank_and_svs(ops: list[np.ndarray], sv_eps: float = 1e-10):
    """
    Take ops on E, vectorize them, and compute SVD of the stack.
    Return (rank, svs).
    """
    if len(ops) == 0:
        return 0, []

    e = ops[0].shape[0]
    M = np.stack([op.reshape(-1) for op in ops], axis=1)  # (e^2) x n_ops
    # Complex SVD
    U, S, Vh = svd(M, full_matrices=False)
    rank = int(np.sum(S > sv_eps))
    return rank, [float(x) for x in S.tolist()]

def orthonormalize_ops(ops: list[np.ndarray], tol=1e-12):
    basis = [normalize_hs(traceless(hermitize(O))) for O in ops]
    basis = gram_schmidt_hs(basis, tol=tol)
    return basis

def cross_comm_stats(left_basis: list[np.ndarray], right_basis: list[np.ndarray]):
    """
    Compute Frobenius norms of commutators [L_i, R_j] over all pairs.
    Return summary stats.
    """
    if not left_basis or not right_basis:
        return {"n_pairs": 0, "median": None, "mean": None, "max": None}

    vals = []
    for A in left_basis:
        for B in right_basis:
            C = comm(A, B)
            vals.append(hs_norm(C))
    arr = np.array(vals, dtype=float)
    return {
        "n_pairs": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


# -------------------------
# Run one trial for one env_dim
# -------------------------

def run_trial(d: int, env_dim: int, dt: float, bandwidth: int, poke_eps: float, seed: int, trial: int,
              topology_scramble: bool = True):
    rng = np.random.default_rng(seed + 1000 * trial + 17 * env_dim)

    spec = ModelSpec(
        d=d,
        env_dim=env_dim,
        dt=dt,
        bandwidth=bandwidth,
        seed=seed,
        trial=trial,
        topology_scramble=topology_scramble
    )

    H_LE, H_ER, env_mode, k_used = build_local_hamiltonians(spec, rng)
    U = step_unitary(spec, H_LE, H_ER)

    left_ops, right_ops = response_operators_on_E(spec, U, poke_eps=poke_eps)

    # Span diagnostics
    rank_L, svs_L = span_rank_and_svs(left_ops, sv_eps=1e-10)
    rank_R, svs_R = span_rank_and_svs(right_ops, sv_eps=1e-10)

    # Orthonormal bases for commutator stats (take up to 8 if d=3)
    left_basis = orthonormalize_ops(left_ops, tol=1e-12)
    right_basis = orthonormalize_ops(right_ops, tol=1e-12)

    # Clip to expected su(d) size for reporting consistency
    target_dim = d * d - 1
    left_basis = left_basis[:target_dim]
    right_basis = right_basis[:target_dim]

    cross = cross_comm_stats(left_basis, right_basis)

    # Also check internal closure-ish magnitude: commutators among left basis itself (optional)
    def internal_comm_summary(basis):
        if len(basis) < 2:
            return {"median": None, "mean": None, "max": None}
        vals = []
        for i in range(len(basis)):
            for j in range(i + 1, len(basis)):
                vals.append(hs_norm(comm(basis[i], basis[j])))
        arr = np.array(vals, dtype=float)
        return {"median": float(np.median(arr)), "mean": float(np.mean(arr)), "max": float(np.max(arr))}

    internal_L = internal_comm_summary(left_basis)
    internal_R = internal_comm_summary(right_basis)

    return {
        "env_dim": env_dim,
        "env_mode": env_mode,
        "k_used": int(k_used),
        "rank_left": int(rank_L),
        "rank_right": int(rank_R),
        "svs_left": svs_L,
        "svs_right": svs_R,
        "cross_comm": cross,
        "internal_comm_left": internal_L,
        "internal_comm_right": internal_R,
    }


def aggregate_trials(trials: list[dict]):
    """
    Aggregate ranks and comm stats across trials for a given env_dim.
    """
    env_dim = trials[0]["env_dim"] if trials else None
    ranks_L = np.array([t["rank_left"] for t in trials], dtype=float)
    ranks_R = np.array([t["rank_right"] for t in trials], dtype=float)

    cross_median = np.array([t["cross_comm"]["median"] for t in trials if t["cross_comm"]["median"] is not None], dtype=float)
    cross_max = np.array([t["cross_comm"]["max"] for t in trials if t["cross_comm"]["max"] is not None], dtype=float)

    out = {
        "env_dim": env_dim,
        "n_trials": int(len(trials)),
        "rank_left_mean": float(np.mean(ranks_L)) if len(ranks_L) else None,
        "rank_right_mean": float(np.mean(ranks_R)) if len(ranks_R) else None,
        "rank_left_min": int(np.min(ranks_L)) if len(ranks_L) else None,
        "rank_left_max": int(np.max(ranks_L)) if len(ranks_L) else None,
        "rank_right_min": int(np.min(ranks_R)) if len(ranks_R) else None,
        "rank_right_max": int(np.max(ranks_R)) if len(ranks_R) else None,
        "cross_comm_median_mean": float(np.mean(cross_median)) if len(cross_median) else None,
        "cross_comm_median_min": float(np.min(cross_median)) if len(cross_median) else None,
        "cross_comm_median_max": float(np.max(cross_median)) if len(cross_median) else None,
        "cross_comm_max_mean": float(np.mean(cross_max)) if len(cross_max) else None,
        "cross_comm_max_min": float(np.min(cross_max)) if len(cross_max) else None,
        "cross_comm_max_max": float(np.max(cross_max)) if len(cross_max) else None,
    }
    return out


# -------------------------
# CLI
# -------------------------

def parse_csv_ints(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--env_dims", type=str, default="3,6,9", help="comma list, e.g. 3,6,9")
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--bandwidth", type=int, default=8, help="number of coupled generator channels per end (<= d^2-1)")
    ap.add_argument("--poke_eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--no_scramble", action="store_true", help="disable topology basis scrambling (env_dim=d^2 only)")
    args = ap.parse_args()

    d = args.d
    env_dims = parse_csv_ints(args.env_dims)
    if not env_dims:
        env_dims = [d, d * d]
    topology_scramble = (not args.no_scramble)

    meta = {
        "script": os.path.basename(__file__),
        "timestamp": now_tag(),
        "d": d,
        "env_dims": env_dims,
        "dt": args.dt,
        "bandwidth": args.bandwidth,
        "poke_eps": args.poke_eps,
        "seed": args.seed,
        "trials": args.trials,
        "topology_scramble": topology_scramble,
        "note": "Echo-probe test: does link memory E support two commuting su(d) endpoint actions? Compare env_dim vs d^2.",
    }

    results = {"meta": meta, "by_env_dim": []}

    for env_dim in env_dims:
        trials = []
        for t in range(args.trials):
            rec = run_trial(
                d=d,
                env_dim=env_dim,
                dt=args.dt,
                bandwidth=args.bandwidth,
                poke_eps=args.poke_eps,
                seed=args.seed,
                trial=t,
                topology_scramble=topology_scramble
            )
            trials.append(rec)

        agg = aggregate_trials(trials)

        print("------------------------------------------------------------")
        print(f"env_dim={env_dim}  (d={d}, d^2={d*d})")
        print(f"  rank_left  mean={agg['rank_left_mean']:.3f}  min={agg['rank_left_min']}  max={agg['rank_left_max']}")
        print(f"  rank_right mean={agg['rank_right_mean']:.3f}  min={agg['rank_right_min']}  max={agg['rank_right_max']}")
        print(f"  cross_comm median(mean)={agg['cross_comm_median_mean']}  max(mean)={agg['cross_comm_max_mean']}")
        # show first trial mode
        print(f"  example env_mode={trials[0]['env_mode']}  k_used={trials[0]['k_used']}")

        results["by_env_dim"].append({
            "env_dim": env_dim,
            "aggregate": agg,
            "trials": trials
        })

    out_dir = ensure_out_dir()
    out_path = os.path.join(out_dir, f"hsf_link_memory_forcing_test_v1_{meta['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("============================================================")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()