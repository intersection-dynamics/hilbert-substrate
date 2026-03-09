#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hsf_link_boundary_inheritance_test_v2.py
======================================

What this tests
---------------
Your hypothesis:

  "Links inherit/absorb something from each subsystem that makes each side different."
  i.e. a link is ONE object, but it develops TWO distinguishable boundary imprints:
      - a left-end (A-facing) boundary mode
      - a right-end (B-facing) boundary mode

In gauge-language: the link should support two independent endpoint actions that
(approximately) commute:
      [L^a, R^b] ~ 0

In HSF-language: the "echo" is in topology; boundary echoes should be separable.

This script tests that idea in a minimal, falsifiable toy model.

Core setup (one step)
---------------------
We simulate two endpoint subsystems L and R (each dimension d, default d=3),
and a link-topology subsystem E. But E is implemented as:

    E  ≅  E_L ⊗ E_R ⊗ E_core

where:
    dim(E_L)=d, dim(E_R)=d, dim(E_core)=core_dim (default 1)
so dim(E)=d*d*core_dim.

Then we *scramble the basis* of E with a random unitary S so that externally it
looks like a single opaque "topology object":

    O_E (physical) = S (O_{E_L E_R core}) S^†

Two competing dynamics
----------------------
A) Boundary-separable inheritance (the thing you want):
   - L couples only to E_L
   - R couples only to E_R
   - Optional internal mixing between E_L and E_R controlled by mix_strength

B) Blended inheritance (control):
   - Both ends couple into the same shared E operator pool (overlapping memory traces)
   - This tends to destroy clean left/right commutation

Probe (echo algebra style)
--------------------------
Starting from |0>_L |0>_E |0>_R:
  - poke L by exp(-i eps T^a) for each su(d) generator
  - run one step U
  - measure delta rho_E
  - build span statistics (rank, singular values)

Same for pokes on R.

Then compute cross-commutator norms between orthonormalized left-basis and right-basis
operators extracted on E:

    cross_comm_median, cross_comm_mean, cross_comm_max

Interpretation
--------------
If boundary inheritance really yields two different sides, then in the boundary-separable
model (with low mixing) we should see:
  - rank_left ~ d^2-1 (for d=3 -> 8)  [or at least > the blended model]
  - rank_right ~ d^2-1
  - cross_comm stats drop significantly (toward numerical noise) as mix_strength -> 0

If cross_comm stays O(1) even at mix_strength=0, then the probe cannot detect
two commuting endpoint actions in this model (meaning we still don't have the right structure).

Run (Windows one-liners)
------------------------
python hsf_link_boundary_inheritance_test_v2.py --d 3 --core_dim 1 --dt 0.10 --bandwidth 8 --poke_eps 1e-3 --seed 0 --trials 8 --mix_values 0,0.01,0.05,0.1,0.2

You can also compare "boundary" vs "blended" explicitly:
python hsf_link_boundary_inheritance_test_v2.py --model both --mix_values 0,0.05,0.1,0.2

Outputs
-------
Writes JSON to ./hsf_out/hsf_link_boundary_inheritance_test_v2_<timestamp>.json

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

def parse_csv_floats(s: str):
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_csv_ints(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# -------------------------
# su(d) Hermitian generator basis (HS-orthonormal)
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
# Model
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
    model: str = "boundary"  # "boundary" or "blended"
    scramble_topology: bool = True


def pure0(n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=complex)
    v[0] = 1.0
    return v

def density_from_state(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())

def partial_trace_rho(rho: np.ndarray, dims: tuple[int, int, int], keep: str) -> np.ndarray:
    dL, dE, dR = dims
    rho = rho.reshape((dL, dE, dR, dL, dE, dR))

    if keep == "E":
        out = np.zeros((dE, dE), dtype=complex)
        for iL in range(dL):
            for iR in range(dR):
                out += rho[iL, :, iR, iL, :, iR]
        return out
    raise ValueError("keep must be 'E' in this script")


def build_topology_ops_boundary(spec: Spec, rng: np.random.Generator):
    """
    Boundary-separable link-topology:
      E = E_L ⊗ E_R ⊗ core
      - left-end couplers act only on E_L
      - right-end couplers act only on E_R
      - optional mixing between E_L and E_R controlled by mix_strength

    We scramble the full E basis by S to imitate "opaque topology coordinates".
    """
    d = spec.d
    c = spec.core_dim
    dimE = d * d * c

    T = su_generators(d)
    k = min(spec.bandwidth, len(T))

    I_d = np.eye(d, dtype=complex)
    I_c = np.eye(c, dtype=complex)

    # Coupling operators living on E (pre-scramble)
    E_left = [kron(T[a], I_d, I_c) for a in range(k)]   # acts on E_L only
    E_right = [kron(I_d, T[a].T, I_c) for a in range(k)]  # acts on E_R only (transpose for right action style)

    # Mixing operator (pre-scramble): couple E_L and E_R to blend boundary records
    # Build a small set of mix terms and scale by mix_strength.
    mix_terms = []
    if spec.mix_strength > 0:
        # use first few generators to construct mixing
        n_mix = min(3, len(T))
        for a in range(n_mix):
            mix_terms.append(kron(T[a], T[a], I_c))
        # normalize mix_terms collectively
        # (not critical; we just want stable scaling)
        mixH = sum(mix_terms)
        mixH = normalize_hs(traceless(hermitize(mixH)))
    else:
        mixH = np.zeros((dimE, dimE), dtype=complex)

    # Scramble
    if spec.scramble_topology:
        S = random_unitary(dimE, rng)
        E_left = [S @ O @ S.conj().T for O in E_left]
        E_right = [S @ O @ S.conj().T for O in E_right]
        mixH = S @ mixH @ S.conj().T
    else:
        S = np.eye(dimE, dtype=complex)

    # Normalize / clean
    E_left = [normalize_hs(traceless(hermitize(O))) for O in E_left]
    E_right = [normalize_hs(traceless(hermitize(O))) for O in E_right]
    mixH = spec.mix_strength * normalize_hs(traceless(hermitize(mixH))) if spec.mix_strength > 0 else mixH

    return E_left, E_right, mixH, {"dimE": dimE, "k_used": k, "scrambled": spec.scramble_topology}


def build_topology_ops_blended(spec: Spec, rng: np.random.Generator):
    """
    Blended control:
      E is one object; both ends couple into an overlapping operator pool.
      This simulates 'echo traces blend into a single topological memory channel'.

    We still allow a mixing Hamiltonian (here redundant, but kept for symmetry).
    """
    d = spec.d
    c = spec.core_dim
    dimE = d * d * c
    k = min(spec.bandwidth, d * d - 1)

    # build a pool of random Hermitian traceless ops on E
    pool = []
    for _ in range(max(3 * k, 10)):
        X = rng.normal(size=(dimE, dimE)) + 1j * rng.normal(size=(dimE, dimE))
        H = traceless(hermitize(X))
        pool.append(H)
    pool = [normalize_hs(O) for O in pool]
    pool = gram_schmidt_hs(pool, tol=1e-12)
    if len(pool) < k:
        # pad lightly if numerical issues
        while len(pool) < k:
            X = rng.normal(size=(dimE, dimE)) + 1j * rng.normal(size=(dimE, dimE))
            pool.append(normalize_hs(traceless(hermitize(X))))
        pool = gram_schmidt_hs(pool, tol=1e-12)

    # both ends draw from same pool but different mixes
    E_left = pool[:k]
    mix = rng.normal(size=(k, k))
    Q, _ = np.linalg.qr(mix)
    E_right = []
    for a in range(k):
        H = np.zeros((dimE, dimE), dtype=complex)
        for b in range(k):
            H += Q[b, a] * E_left[b]
        E_right.append(normalize_hs(traceless(hermitize(H))))

    # optional extra blend mixing inside E (random)
    if spec.mix_strength > 0:
        X = rng.normal(size=(dimE, dimE)) + 1j * rng.normal(size=(dimE, dimE))
        mixH = spec.mix_strength * normalize_hs(traceless(hermitize(X)))
    else:
        mixH = np.zeros((dimE, dimE), dtype=complex)

    return E_left, E_right, mixH, {"dimE": dimE, "k_used": k, "scrambled": False}


def build_step_unitary(spec: Spec, rng: np.random.Generator):
    """
    Build U_step on L ⊗ E ⊗ R:
        U = exp(-i dt H_total)
    where:
        H_total = H_LE + H_ER + H_mix(E)

    Locality is enforced by taking:
        H_LE = sum_a T_L^a ⊗ E_left^a  (acts on L and E)
        H_ER = sum_a E_right^a ⊗ T_R^a (acts on E and R)
    """
    d = spec.d
    T = su_generators(d)
    k = min(spec.bandwidth, len(T))

    if spec.model == "boundary":
        E_left, E_right, Hmix_E, infoE = build_topology_ops_boundary(spec, rng)
    elif spec.model == "blended":
        E_left, E_right, Hmix_E, infoE = build_topology_ops_blended(spec, rng)
    else:
        raise ValueError("model must be 'boundary' or 'blended'")

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

    # Lift to full L⊗E⊗R
    H_total = kron(H_LE, I_R) + kron(I_L, H_ER) + kron(I_L, Hmix_E, I_R)
    H_total = hermitize(H_total)

    U = expm(-1j * spec.dt * H_total)

    return U, dimE, infoE


def response_ops_on_E(spec: Spec, U: np.ndarray, dimE: int):
    """
    Echo-probe:
      - base evolve
      - poke L / R separately by su(d) generators
      - record delta rho_E
    """
    d = spec.d
    dims = (d, dimE, d)
    T = su_generators(d)
    nT = len(T)

    psi0 = kron(pure0(d), pure0(dimE), pure0(d))
    rho0 = density_from_state(psi0)

    psi_base = U @ psi0
    rho_base = density_from_state(psi_base)
    rhoE_base = partial_trace_rho(rho_base, dims, keep="E")

    left_ops = []
    right_ops = []

    # poke left
    for a in range(nT):
        UL = expm(-1j * spec.poke_eps * T[a])
        poke = kron(UL, np.eye(dimE, dtype=complex), np.eye(d, dtype=complex))
        psi = U @ (poke @ psi0)
        rho = density_from_state(psi)
        rhoE = partial_trace_rho(rho, dims, keep="E")
        dE = traceless(hermitize(rhoE - rhoE_base))
        left_ops.append(dE)

    # poke right
    for a in range(nT):
        UR = expm(-1j * spec.poke_eps * T[a])
        poke = kron(np.eye(d, dtype=complex), np.eye(dimE, dtype=complex), UR)
        psi = U @ (poke @ psi0)
        rho = density_from_state(psi)
        rhoE = partial_trace_rho(rho, dims, keep="E")
        dE = traceless(hermitize(rhoE - rhoE_base))
        right_ops.append(dE)

    return left_ops, right_ops


def span_rank_and_svs(ops: list[np.ndarray], sv_eps: float = 1e-10):
    if not ops:
        return 0, []
    dimE = ops[0].shape[0]
    M = np.stack([op.reshape(-1) for op in ops], axis=1)  # (dimE^2) x n
    _, S, _ = svd(M, full_matrices=False)
    rank = int(np.sum(S > sv_eps))
    return rank, [float(x) for x in S.tolist()]


def orthonormalize_ops(ops: list[np.ndarray], tol=1e-12):
    basis = [normalize_hs(traceless(hermitize(O))) for O in ops]
    basis = gram_schmidt_hs(basis, tol=tol)
    return basis


def cross_comm_stats(left_basis: list[np.ndarray], right_basis: list[np.ndarray]):
    if not left_basis or not right_basis:
        return {"n_pairs": 0, "median": None, "mean": None, "max": None}
    vals = []
    for A in left_basis:
        for B in right_basis:
            vals.append(hs_norm(comm(A, B)))
    arr = np.array(vals, dtype=float)
    return {
        "n_pairs": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def run_one(spec: Spec):
    rng = np.random.default_rng(spec.seed + 1000 * spec.trial + int(10000 * spec.mix_strength) + (1 if spec.model == "blended" else 0))

    U, dimE, infoE = build_step_unitary(spec, rng)
    left_ops, right_ops = response_ops_on_E(spec, U, dimE)

    rankL, svsL = span_rank_and_svs(left_ops, sv_eps=1e-10)
    rankR, svsR = span_rank_and_svs(right_ops, sv_eps=1e-10)

    left_basis = orthonormalize_ops(left_ops, tol=1e-12)
    right_basis = orthonormalize_ops(right_ops, tol=1e-12)

    target_dim = spec.d * spec.d - 1
    left_basis = left_basis[:target_dim]
    right_basis = right_basis[:target_dim]

    cross = cross_comm_stats(left_basis, right_basis)

    return {
        "model": spec.model,
        "mix_strength": float(spec.mix_strength),
        "trial": int(spec.trial),
        "dimE": int(dimE),
        "infoE": infoE,
        "rank_left": int(rankL),
        "rank_right": int(rankR),
        "svs_left": svsL,
        "svs_right": svsR,
        "cross_comm": cross,
    }


def aggregate(records: list[dict]):
    ranksL = np.array([r["rank_left"] for r in records], dtype=float)
    ranksR = np.array([r["rank_right"] for r in records], dtype=float)
    med = np.array([r["cross_comm"]["median"] for r in records if r["cross_comm"]["median"] is not None], dtype=float)
    mx = np.array([r["cross_comm"]["max"] for r in records if r["cross_comm"]["max"] is not None], dtype=float)

    return {
        "n_trials": int(len(records)),
        "rank_left_mean": float(np.mean(ranksL)) if len(ranksL) else None,
        "rank_right_mean": float(np.mean(ranksR)) if len(ranksR) else None,
        "rank_left_min": int(np.min(ranksL)) if len(ranksL) else None,
        "rank_left_max": int(np.max(ranksL)) if len(ranksL) else None,
        "rank_right_min": int(np.min(ranksR)) if len(ranksR) else None,
        "rank_right_max": int(np.max(ranksR)) if len(ranksR) else None,
        "cross_comm_median_mean": float(np.mean(med)) if len(med) else None,
        "cross_comm_median_min": float(np.min(med)) if len(med) else None,
        "cross_comm_median_max": float(np.max(med)) if len(med) else None,
        "cross_comm_max_mean": float(np.mean(mx)) if len(mx) else None,
        "cross_comm_max_min": float(np.min(mx)) if len(mx) else None,
        "cross_comm_max_max": float(np.max(mx)) if len(mx) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--core_dim", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--bandwidth", type=int, default=8)
    ap.add_argument("--poke_eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--mix_values", type=str, default="0,0.01,0.05,0.1,0.2")
    ap.add_argument("--model", type=str, default="both", help="boundary | blended | both")
    ap.add_argument("--no_scramble", action="store_true", help="disable topology scrambling (boundary model only)")
    args = ap.parse_args()

    mix_values = parse_csv_floats(args.mix_values)
    if not mix_values:
        mix_values = [0.0, 0.05, 0.1, 0.2]

    models = ["boundary", "blended"] if args.model == "both" else [args.model]

    meta = {
        "script": os.path.basename(__file__),
        "timestamp": now_tag(),
        "d": args.d,
        "core_dim": args.core_dim,
        "dt": args.dt,
        "bandwidth": args.bandwidth,
        "poke_eps": args.poke_eps,
        "seed": args.seed,
        "trials": args.trials,
        "mix_values": mix_values,
        "models": models,
        "scramble_topology": (not args.no_scramble),
        "note": "v2 tests boundary-inheritance vs blended echo; sweeps internal mixing between left/right boundary modes.",
    }

    out = {"meta": meta, "runs": []}

    for model in models:
        for mix in mix_values:
            recs = []
            for t in range(args.trials):
                spec = Spec(
                    d=args.d,
                    core_dim=args.core_dim,
                    dt=args.dt,
                    bandwidth=args.bandwidth,
                    poke_eps=args.poke_eps,
                    seed=args.seed,
                    trial=t,
                    mix_strength=mix,
                    model=model,
                    scramble_topology=(not args.no_scramble),
                )
                recs.append(run_one(spec))

            agg = aggregate(recs)

            print("------------------------------------------------------------")
            print(f"model={model:8s} mix={mix:.5f}  dimE={recs[0]['dimE']}  k_used={recs[0]['infoE'].get('k_used')}")
            print(f"  rank_left  mean={agg['rank_left_mean']:.3f}  min={agg['rank_left_min']}  max={agg['rank_left_max']}")
            print(f"  rank_right mean={agg['rank_right_mean']:.3f}  min={agg['rank_right_min']}  max={agg['rank_right_max']}")
            print(f"  cross_comm median(mean)={agg['cross_comm_median_mean']}  max(mean)={agg['cross_comm_max_mean']}")

            out["runs"].append({
                "model": model,
                "mix_strength": float(mix),
                "aggregate": agg,
                "trials": recs
            })

    out_dir = ensure_out_dir()
    path = os.path.join(out_dir, f"hsf_link_boundary_inheritance_test_v2_{meta['timestamp']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("============================================================")
    print("Wrote:", path)


if __name__ == "__main__":
    main()