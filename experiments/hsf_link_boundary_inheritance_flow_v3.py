#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hsf_link_boundary_inheritance_flow_v3.py
=======================================

Purpose
-------
Measure a "mixing flow" under repeated time steps in the boundary-inheritance toy model.

We iterate a stroboscopic evolution with endpoint reset each step:

  rho_E^{(n+1)} = Tr_{LR}[ U (rho_L ⊗ rho_E^{(n)} ⊗ rho_R) U^† ]

This makes "flow" meaningful: E accumulates echo memory while endpoints act like fresh baths
(or fresh probes) each step.

We then estimate a scalar mixing measure m_n at each step n by extracting, from the current rho_E^{(n)},
the endpoint-response operator spans on E using a poke-and-evolve probe, and computing:

  m_n := median_{i,j} || [L_i, R_j] ||_HS

where {L_i} and {R_j} are HS-orthonormal bases (up to su(d) size) extracted from the
left- and right-poke response operators on E.

We sweep mix_strength values (internal coupling between boundary modes), and optionally compare
boundary vs blended.

Outputs
-------
Writes JSON to ./hsf_out/hsf_link_boundary_inheritance_flow_v3_<timestamp>.json
and prints a compact table.

Run (Windows one-liners)
------------------------
python hsf_link_boundary_inheritance_flow_v3.py --d 3 --core_dim 1 --dt 0.10 --bandwidth 8 --poke_eps 1e-3 --seed 0 --steps 30 --trials 6 --mix_values 0,0.01,0.05,0.1,0.2 --model boundary

Compare both models:
python hsf_link_boundary_inheritance_flow_v3.py --model both --mix_values 0,0.05,0.1,0.2

Notes
-----
- This is a *measurement* script, not an enforcement script.
- For stability, we use the same U each run (same seed+trial+mix) and reset endpoints each step.
- The probe uses the current rho_E^{(n)} as the initial E state (not |0>), consistent with echo accumulation.
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
# su(d) basis
# -------------------------

def su_generators(d: int):
    gens = []
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
# Spec
# -------------------------

@dataclass
class Spec:
    d: int = 3
    core_dim: int = 1
    dt: float = 0.10
    bandwidth: int = 8
    poke_eps: float = 1e-3
    seed: int = 0
    trial: int = 0
    mix_strength: float = 0.0
    model: str = "boundary"   # "boundary" or "blended"
    scramble_topology: bool = True
    reset_mode: str = "pure0" # "pure0" or "maxmix"


# -------------------------
# Partial trace helpers (L,E,R)
# -------------------------

def density_from_state(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())

def pure0(n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=complex)
    v[0] = 1.0
    return v

def partial_trace_E(rho_LER: np.ndarray, dL: int, dE: int, dR: int) -> np.ndarray:
    rho = rho_LER.reshape((dL, dE, dR, dL, dE, dR))
    out = np.zeros((dE, dE), dtype=complex)
    for iL in range(dL):
        for iR in range(dR):
            out += rho[iL, :, iR, iL, :, iR]
    return out

def partial_trace_LR_to_E_of_op(O: np.ndarray, dL: int, dE: int, dR: int) -> np.ndarray:
    # Same as above but for an operator that already lives in LER
    return partial_trace_E(O, dL, dE, dR)


# -------------------------
# Topology operator construction
# -------------------------

def build_topology_ops_boundary(d: int, core_dim: int, bandwidth: int, mix_strength: float,
                               scramble: bool, rng: np.random.Generator):
    """
    E = E_L ⊗ E_R ⊗ core
    - left couplers act on E_L
    - right couplers act on E_R
    - internal mixing Hmix couples E_L and E_R
    """
    c = core_dim
    dimE = d * d * c

    T = su_generators(d)
    k = min(bandwidth, len(T))
    I_d = np.eye(d, dtype=complex)
    I_c = np.eye(c, dtype=complex)

    E_left = [kron(T[a], I_d, I_c) for a in range(k)]
    E_right = [kron(I_d, T[a].T, I_c) for a in range(k)]

    if mix_strength > 0:
        n_mix = min(3, len(T))
        mix_terms = [kron(T[a], T[a], I_c) for a in range(n_mix)]
        Hmix = sum(mix_terms)
        Hmix = normalize_hs(traceless(hermitize(Hmix)))
        Hmix = mix_strength * Hmix
    else:
        Hmix = np.zeros((dimE, dimE), dtype=complex)

    if scramble:
        S = random_unitary(dimE, rng)
        E_left = [S @ O @ S.conj().T for O in E_left]
        E_right = [S @ O @ S.conj().T for O in E_right]
        Hmix = S @ Hmix @ S.conj().T
    else:
        S = np.eye(dimE, dtype=complex)

    E_left = [normalize_hs(traceless(hermitize(O))) for O in E_left]
    E_right = [normalize_hs(traceless(hermitize(O))) for O in E_right]
    Hmix = hermitize(Hmix)

    info = {"dimE": dimE, "k_used": k, "scrambled": scramble}
    return E_left, E_right, Hmix, info

def build_topology_ops_blended(d: int, core_dim: int, bandwidth: int, mix_strength: float,
                               rng: np.random.Generator):
    """
    Control: overlapping operator pool for both ends.
    """
    c = core_dim
    dimE = d * d * c
    k = min(bandwidth, d * d - 1)

    pool = []
    for _ in range(max(3 * k, 10)):
        X = rng.normal(size=(dimE, dimE)) + 1j * rng.normal(size=(dimE, dimE))
        H = traceless(hermitize(X))
        pool.append(H)
    pool = [normalize_hs(O) for O in pool]
    pool = gram_schmidt_hs(pool, tol=1e-12)
    if len(pool) < k:
        while len(pool) < k:
            X = rng.normal(size=(dimE, dimE)) + 1j * rng.normal(size=(dimE, dimE))
            pool.append(normalize_hs(traceless(hermitize(X))))
        pool = gram_schmidt_hs(pool, tol=1e-12)

    E_left = pool[:k]
    mix = rng.normal(size=(k, k))
    Q, _ = np.linalg.qr(mix)
    E_right = []
    for a in range(k):
        H = np.zeros((dimE, dimE), dtype=complex)
        for b in range(k):
            H += Q[b, a] * E_left[b]
        E_right.append(normalize_hs(traceless(hermitize(H))))

    if mix_strength > 0:
        X = rng.normal(size=(dimE, dimE)) + 1j * rng.normal(size=(dimE, dimE))
        Hmix = mix_strength * normalize_hs(traceless(hermitize(X)))
    else:
        Hmix = np.zeros((dimE, dimE), dtype=complex)

    info = {"dimE": dimE, "k_used": k, "scrambled": False}
    return E_left, E_right, hermitize(Hmix), info


# -------------------------
# Build one-step unitary U for a given trial+mix
# -------------------------

def build_U_step(spec: Spec, rng: np.random.Generator):
    d = spec.d
    T = su_generators(d)
    k = min(spec.bandwidth, len(T))

    if spec.model == "boundary":
        E_left, E_right, Hmix_E, infoE = build_topology_ops_boundary(
            d=d, core_dim=spec.core_dim, bandwidth=spec.bandwidth,
            mix_strength=spec.mix_strength, scramble=spec.scramble_topology, rng=rng
        )
    elif spec.model == "blended":
        E_left, E_right, Hmix_E, infoE = build_topology_ops_blended(
            d=d, core_dim=spec.core_dim, bandwidth=spec.bandwidth,
            mix_strength=spec.mix_strength, rng=rng
        )
    else:
        raise ValueError("model must be boundary or blended")

    dimE = infoE["dimE"]
    I_L = np.eye(d, dtype=complex)
    I_R = np.eye(d, dtype=complex)

    # H_LE on L⊗E
    H_LE = np.zeros((d * dimE, d * dimE), dtype=complex)
    for a in range(k):
        H_LE += kron(T[a], E_left[a])
    H_LE = hermitize(H_LE)

    # H_ER on E⊗R
    H_ER = np.zeros((dimE * d, dimE * d), dtype=complex)
    for a in range(k):
        H_ER += kron(E_right[a], T[a])
    H_ER = hermitize(H_ER)

    # full H_total on L⊗E⊗R
    H_total = kron(H_LE, I_R) + kron(I_L, H_ER) + kron(I_L, Hmix_E, I_R)
    H_total = hermitize(H_total)

    U = expm(-1j * spec.dt * H_total)
    return U, infoE


# -------------------------
# Endpoint reset states
# -------------------------

def endpoint_state(d: int, mode: str) -> np.ndarray:
    if mode == "pure0":
        v = pure0(d)
        return density_from_state(v)
    if mode == "maxmix":
        return np.eye(d, dtype=complex) / d
    raise ValueError("reset_mode must be pure0 or maxmix")


# -------------------------
# One-step map rho_E -> rho_E'
# -------------------------

def step_rhoE(U: np.ndarray, rhoE: np.ndarray, rhoL: np.ndarray, rhoR: np.ndarray, d: int, dimE: int) -> np.ndarray:
    # Build rho_LER = rhoL ⊗ rhoE ⊗ rhoR
    rho_LER = kron(rhoL, rhoE, rhoR)
    rho2 = U @ rho_LER @ U.conj().T
    rhoE2 = partial_trace_E(rho2, d, dimE, d)
    rhoE2 = hermitize(rhoE2)
    # Numerical cleanup: keep trace=1
    tr = np.trace(rhoE2).real
    if tr <= 0:
        return rhoE2
    return rhoE2 / tr


# -------------------------
# Probe at a given rhoE: extract left/right response operators on E
# -------------------------

def probe_response_ops(U: np.ndarray, rhoE: np.ndarray, spec: Spec, dimE: int):
    """
    Use current rhoE as the E part of the initial state, endpoints reset to rhoL,rhoR.
    Then poke L or R and measure delta rhoE after one step.
    """
    d = spec.d
    T = su_generators(d)

    rhoL0 = endpoint_state(d, spec.reset_mode)
    rhoR0 = endpoint_state(d, spec.reset_mode)

    # base
    rho_base = step_rhoE(U, rhoE, rhoL0, rhoR0, d, dimE)

    left_ops = []
    right_ops = []

    for a in range(len(T)):
        UL = expm(-1j * spec.poke_eps * T[a])
        rhoL = UL @ rhoL0 @ UL.conj().T
        rho_p = step_rhoE(U, rhoE, rhoL, rhoR0, d, dimE)
        dE = traceless(hermitize(rho_p - rho_base))
        left_ops.append(dE)

    for a in range(len(T)):
        UR = expm(-1j * spec.poke_eps * T[a])
        rhoR = UR @ rhoR0 @ UR.conj().T
        rho_p = step_rhoE(U, rhoE, rhoL0, rhoR, d, dimE)
        dE = traceless(hermitize(rho_p - rho_base))
        right_ops.append(dE)

    return left_ops, right_ops


def span_rank_and_svs(ops: list[np.ndarray], sv_eps: float = 1e-10):
    if not ops:
        return 0, []
    M = np.stack([op.reshape(-1) for op in ops], axis=1)  # (dimE^2) x n
    _, S, _ = svd(M, full_matrices=False)
    rank = int(np.sum(S > sv_eps))
    return rank, [float(x) for x in S.tolist()]


def orthonormalize_ops(ops: list[np.ndarray], tol=1e-12):
    basis = [normalize_hs(traceless(hermitize(O))) for O in ops]
    basis = gram_schmidt_hs(basis, tol=tol)
    return basis

def cross_comm_median(left_basis: list[np.ndarray], right_basis: list[np.ndarray]):
    if not left_basis or not right_basis:
        return None, None, None
    vals = []
    for A in left_basis:
        for B in right_basis:
            vals.append(hs_norm(comm(A, B)))
    arr = np.array(vals, dtype=float)
    return float(np.median(arr)), float(np.mean(arr)), float(np.max(arr))


# -------------------------
# Run one flow trajectory
# -------------------------

def run_flow(spec: Spec, steps: int):
    # deterministic per trial/mix/model
    rng = np.random.default_rng(spec.seed + 1000 * spec.trial + int(100000 * spec.mix_strength) + (7 if spec.model == "blended" else 0))
    U, infoE = build_U_step(spec, rng)
    dimE = infoE["dimE"]

    # initial rhoE: pure |0><0|
    vE0 = pure0(dimE)
    rhoE = density_from_state(vE0)

    rhoL0 = endpoint_state(spec.d, spec.reset_mode)
    rhoR0 = endpoint_state(spec.d, spec.reset_mode)

    # record time series
    series = []
    for n in range(steps + 1):
        # measure mixing at current rhoE (before stepping)
        left_ops, right_ops = probe_response_ops(U, rhoE, spec, dimE)
        rankL, _ = span_rank_and_svs(left_ops)
        rankR, _ = span_rank_and_svs(right_ops)

        left_basis = orthonormalize_ops(left_ops, tol=1e-12)
        right_basis = orthonormalize_ops(right_ops, tol=1e-12)

        target_dim = spec.d * spec.d - 1
        left_basis = left_basis[:target_dim]
        right_basis = right_basis[:target_dim]

        med, mean, mx = cross_comm_median(left_basis, right_basis)

        series.append({
            "step": int(n),
            "rank_left": int(rankL),
            "rank_right": int(rankR),
            "mix_median": med,
            "mix_mean": mean,
            "mix_max": mx,
        })

        # update rhoE (skip after final measurement)
        if n < steps:
            rhoE = step_rhoE(U, rhoE, rhoL0, rhoR0, spec.d, dimE)

    return {"infoE": infoE, "series": series}


def aggregate_series(trajs: list[dict], steps: int):
    """
    Aggregate mixing median across trials at each step.
    """
    med = np.full((len(trajs), steps + 1), np.nan, dtype=float)
    rankL = np.full((len(trajs), steps + 1), np.nan, dtype=float)
    rankR = np.full((len(trajs), steps + 1), np.nan, dtype=float)

    for i, tr in enumerate(trajs):
        for row in tr["series"]:
            n = row["step"]
            if row["mix_median"] is not None:
                med[i, n] = float(row["mix_median"])
            rankL[i, n] = float(row["rank_left"])
            rankR[i, n] = float(row["rank_right"])

    def stats(arr2d):
        out = []
        for n in range(steps + 1):
            col = arr2d[:, n]
            col = col[np.isfinite(col)]
            if col.size == 0:
                out.append({"step": n, "mean": None, "median": None, "min": None, "max": None})
            else:
                out.append({
                    "step": int(n),
                    "mean": float(np.mean(col)),
                    "median": float(np.median(col)),
                    "min": float(np.min(col)),
                    "max": float(np.max(col)),
                })
        return out

    return {
        "mix_median_stats": stats(med),
        "rank_left_stats": stats(rankL),
        "rank_right_stats": stats(rankR),
    }


# -------------------------
# CLI
# -------------------------

def parse_csv_floats(s: str):
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--core_dim", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--bandwidth", type=int, default=8)
    ap.add_argument("--poke_eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--mix_values", type=str, default="0,0.01,0.05,0.1,0.2")
    ap.add_argument("--model", type=str, default="boundary", help="boundary | blended | both")
    ap.add_argument("--reset_mode", type=str, default="pure0", help="pure0 | maxmix")
    ap.add_argument("--no_scramble", action="store_true")
    args = ap.parse_args()

    mix_values = parse_csv_floats(args.mix_values)
    if not mix_values:
        mix_values = [0.0, 0.05, 0.1, 0.2]

    models = ["boundary", "blended"] if args.model == "both" else [args.model]
    ts = now_tag()

    meta = {
        "script": os.path.basename(__file__),
        "timestamp": ts,
        "d": args.d,
        "core_dim": args.core_dim,
        "dt": args.dt,
        "bandwidth": args.bandwidth,
        "poke_eps": args.poke_eps,
        "seed": args.seed,
        "trials": args.trials,
        "steps": args.steps,
        "mix_values": mix_values,
        "models": models,
        "reset_mode": args.reset_mode,
        "scramble_topology": (not args.no_scramble),
        "note": "v3 measures mixing flow m_n by iterating rhoE with endpoint reset and probing commutator stats each step.",
    }

    out = {"meta": meta, "runs": []}

    for model in models:
        for mix in mix_values:
            trajs = []
            for t in range(args.trials):
                spec = Spec(
                    d=args.d,
                    core_dim=args.core_dim,
                    dt=args.dt,
                    bandwidth=args.bandwidth,
                    poke_eps=args.poke_eps,
                    seed=args.seed,
                    trial=t,
                    mix_strength=float(mix),
                    model=model,
                    scramble_topology=(not args.no_scramble),
                    reset_mode=args.reset_mode,
                )
                trajs.append(run_flow(spec, steps=args.steps))

            agg = aggregate_series(trajs, steps=args.steps)

            # Print a compact snapshot: m at step 0, mid, final
            m0 = agg["mix_median_stats"][0]["median"]
            mm = agg["mix_median_stats"][args.steps // 2]["median"]
            mf = agg["mix_median_stats"][args.steps]["median"]

            print("------------------------------------------------------------")
            print(f"model={model:8s} mix={mix:.5f} reset={args.reset_mode} steps={args.steps} trials={args.trials}")
            print(f"  mix_median: step0={m0}  step{args.steps//2}={mm}  step{args.steps}={mf}")

            out["runs"].append({
                "model": model,
                "mix_strength": float(mix),
                "aggregate": agg,
                "trials": trajs
            })

    out_dir = ensure_out_dir()
    path = os.path.join(out_dir, f"hsf_link_boundary_inheritance_flow_v3_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("============================================================")
    print("Wrote:", path)


if __name__ == "__main__":
    main()