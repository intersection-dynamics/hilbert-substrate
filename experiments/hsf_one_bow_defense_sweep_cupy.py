#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hsf_one_bow_defense_sweep_cupy.py

GPU (CuPy) defense sweep for occupancy-vs-N using a monotone Cp descent, with a
TRUNCATED Pauli basis (weight <= wmax) so N~7–12 is feasible.

What you get (in --outdir):
  - occupancy_vs_N.png
  - occupancy_vs_N.csv
  - sweep.json

Core idea
---------
We measure an accessibility-style order parameter as a forced-choice basin label:

  harm_win := |Cp_final - Cp_harm_base| < |Cp_final - Cp_spatial|

Where:
  - Cp_spatial   = Cp(H_spatial) for the Heisenberg nearest-neighbor chain/ring reference
  - Cp_harm_base = Cp(diag-spectrum baseline) (eigenvalues on computational diagonal)
  - Cp_final     = Cp(H_final) after monotone descent

Important performance design choices
------------------------------------
1) We DO NOT form the full 4^N Pauli basis. We build only weight <= wmax strings:
     count ~= Σ_{k=1..wmax} C(N,k) * 3^k
2) Coefficients are computed as elementwise trace trick (O(dim^2) per basis element),
   not matrix multiply (O(dim^3)):
       trace(H @ P) = sum_{ij} H_ij * P_ji = sum(H * P.T)
3) Scrambling uses ONLY adjacent 2-qubit gates (i, i+1), so embedding is simple
   via Kronecker products without expensive permutations.

Dependencies
------------
- cupy
- cupyx.scipy.linalg (for expm on small 4x4 blocks)
- matplotlib (plot)
If CuPy is not available, the script will exit with a clear error.

Windows example (single line)
-----------------------------
python hsf_one_bow_defense_sweep_cupy.py --outdir hsf_defense_sweep_gpu --Ns 3,4,5,6,7,8,9,10,12 --seed0 0 --seed1 19 --p 4 --steps 800 --dt 0.03 --wmax 3 --scramble-depth-factor 2

Tips
----
- Start with: --wmax 2, then try 3.
- For speed: reduce --steps, or reduce seed count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from itertools import combinations, product
from typing import Dict, List, Tuple

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.linalg as cpx_linalg
except Exception as e:
    raise SystemExit(
        "CuPy (and cupyx.scipy.linalg) is required for this script.\n"
        f"Import error: {e}\n"
        "Install matching CuPy for your CUDA version, e.g. cupy-cuda12x."
    )

# Matplotlib for plotting (CPU-side)
import matplotlib.pyplot as plt


# ----------------------------
# GPU linear algebra helpers
# ----------------------------

def herm(A: cp.ndarray) -> cp.ndarray:
    return (A + A.conj().T) * 0.5

def comm(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    return A @ B - B @ A

def fro_norm(A: cp.ndarray) -> float:
    return float(cp.linalg.norm(A, ord="fro").get())

def is_finite_matrix(A: cp.ndarray) -> bool:
    return bool(cp.isfinite(A).all().get())

def trace_elem(H: cp.ndarray, P: cp.ndarray) -> complex:
    # trace(H @ P) = sum_{ij} H_ij * P_ji = sum(H * P.T)
    return complex(cp.sum(H * P.T).get())


# ----------------------------
# Pauli ops / qubit utilities
# ----------------------------

I2 = cp.eye(2, dtype=cp.complex128)
X  = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
Y  = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
Z  = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)

PAULI_SINGLE = [("X", X), ("Y", Y), ("Z", Z)]  # no I in the truncated basis elements


def kron_all(ops: List[cp.ndarray]) -> cp.ndarray:
    out = cp.array([[1]], dtype=cp.complex128)
    for op in ops:
        out = cp.kron(out, op)
    return out


def pauli_strings_truncated(N: int, wmax: int) -> Tuple[List[str], List[cp.ndarray], cp.ndarray]:
    """
    Build truncated Pauli basis of weight 1..wmax (excluding identity).
    Returns (labels, matrices, weights_float_array).
    """
    labels: List[str] = []
    mats: List[cp.ndarray] = []
    ws: List[int] = []

    sites = list(range(N))
    for w in range(1, wmax + 1):
        for support in combinations(sites, w):
            for ops_choice in product(PAULI_SINGLE, repeat=w):
                label_chars = ["I"] * N
                ops = [I2] * N
                for s, (ch, mat) in zip(support, ops_choice):
                    label_chars[s] = ch
                    ops[s] = mat
                labels.append("".join(label_chars))
                mats.append(kron_all(ops))
                ws.append(w)

    weights = cp.asarray(np.array(ws, dtype=np.float64))
    return labels, mats, weights


def pauli_coeffs(H: cp.ndarray, mats: List[cp.ndarray], dim: int) -> cp.ndarray:
    """
    Compute coefficients ck = trace(H @ Pk)/dim using elementwise trace trick.
    O(len(mats) * dim^2). Keeps everything on GPU except the final scalar get(),
    but that get() is only for a complex number per basis element.
    """
    ck = cp.empty((len(mats),), dtype=cp.complex128)
    inv_dim = 1.0 / float(dim)
    for i, Pk in enumerate(mats):
        # use GPU sum then pull scalar
        ck[i] = cp.sum(H * Pk.T) * inv_dim
    return ck


def locality_cost_cp(ck: cp.ndarray, weights: cp.ndarray, p: float) -> float:
    """
    Cp = (Σ w^p |ck|^2) / (Σ |ck|^2)
    """
    amp2 = cp.abs(ck) ** 2
    den = cp.sum(amp2) + 1e-30
    num = cp.sum((weights ** float(p)) * amp2)
    return float((num / den).get())


def build_M_from_coeffs(mats: List[cp.ndarray], ck: cp.ndarray, weights: cp.ndarray, p: float) -> cp.ndarray:
    """
    M = Σ w^p ck Pk   (scaled by 2/||ck||^2 like your earlier monotone variant)
    """
    amp2 = cp.abs(ck) ** 2
    den = cp.sum(amp2) + 1e-30
    scale = (2.0 / den).astype(cp.complex128)

    dim = mats[0].shape[0]
    M = cp.zeros((dim, dim), dtype=cp.complex128)
    wp = (weights ** float(p)).astype(cp.complex128)
    for Pk, c, wpi in zip(mats, ck, wp):
        M += (wpi * c) * Pk
    M *= scale
    return herm(M)


# ----------------------------
# Spatial model + scrambling (adjacent 2-qubit gates)
# ----------------------------

def heisenberg_chain(N: int, J: float = 1.0) -> cp.ndarray:
    """
    Nearest-neighbor Heisenberg chain on a LINE (adjacent tensor order), not periodic.
    That makes adjacent-gate scrambling embedding simple.
    """
    dim = 2 ** N
    H = cp.zeros((dim, dim), dtype=cp.complex128)
    for i in range(N - 1):
        j = i + 1
        # build XX + YY + ZZ using kron lists
        for A, B in ((X, X), (Y, Y), (Z, Z)):
            ops = [I2] * N
            ops[i] = A
            ops[j] = B
            H += J * kron_all(ops)
    return herm(H)


def random_su4(rng: np.random.Generator) -> cp.ndarray:
    """
    Generate a random SU(4) matrix on CPU, then move to GPU.
    QR-based Haar-ish unitary, then determinant fix.
    """
    A = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / math.sqrt(2.0)
    Q, R = np.linalg.qr(A)
    ph = np.diag(R) / np.abs(np.diag(R))
    Q = Q @ np.diag(np.conj(ph))
    det = np.linalg.det(Q)
    Q = Q / det ** (1 / 4)
    return cp.asarray(Q, dtype=cp.complex128)


def embed_adjacent_two_qubit(U4: cp.ndarray, i: int, N: int) -> cp.ndarray:
    """
    Embed a 4x4 gate on adjacent qubits (i, i+1) in N qubits, assuming tensor order.
    Returns full dim x dim unitary.
    """
    assert 0 <= i < N - 1
    ops: List[cp.ndarray] = []
    k = 0
    while k < N:
        if k == i:
            ops.append(U4)
            k += 2
        else:
            ops.append(I2)
            k += 1

    U = ops[0]
    for op in ops[1:]:
        U = cp.kron(U, op)
    return U


def scramble_hamiltonian_adjacent(H: cp.ndarray, N: int, depth: int, rng: np.random.Generator) -> cp.ndarray:
    """
    Scramble H via conjugation by a product of random adjacent SU(4) gates:
      H <- U H U^\dagger
    """
    dim = 2 ** N
    U = cp.eye(dim, dtype=cp.complex128)
    for _ in range(max(1, depth)):
        i = int(rng.integers(0, N - 1))
        U4 = random_su4(rng)
        Uij = embed_adjacent_two_qubit(U4, i, N)
        U = Uij @ U
    return herm(U @ H @ U.conj().T)


# ----------------------------
# Monotone backtracking flow on Cp (GPU)
# ----------------------------

@dataclass
class SweepRow:
    N: int
    seed: int
    Cp_init: float
    Cp_final: float
    Cp_spatial: float
    Cp_harm_base: float
    harm_nearest: int
    spatial_nearest: int
    accepted_steps: int
    final_iter: int
    dt_last: float


def diag_spectrum_baseline_cost(H: cp.ndarray, N: int, p: float,
                                mats: List[cp.ndarray], weights: cp.ndarray) -> float:
    """
    Compute Cp for diag-spectrum baseline:
      eigenvalues of H on diagonal in computational basis.
    """
    evals = cp.linalg.eigvalsh(H).astype(cp.complex128)
    H_diag = cp.diag(evals)
    ck = pauli_coeffs(H_diag, mats, 2**N)
    return locality_cost_cp(ck, weights, p)


def run_double_bracket_flow_monotone(
    H0: cp.ndarray,
    N: int,
    mats: List[cp.ndarray],
    weights: cp.ndarray,
    p: float,
    steps: int,
    dt0: float,
    dt_min: float = 1e-10,
    backtrack: float = 0.5,
    eps_accept: float = 1e-14,
) -> Tuple[cp.ndarray, Dict[str, object]]:
    """
    Monotone backtracking descent on Cp, using:
      dH = [H, [H, M]]  (then H <- H - dt dH)
    where M is built from truncated Pauli coefficients.
    """
    H = herm(H0.copy())
    dim = 2 ** N
    if not is_finite_matrix(H):
        raise ValueError("Initial H0 not finite.")

    ck = pauli_coeffs(H, mats, dim)
    Cp_old = locality_cost_cp(ck, weights, p)
    if not np.isfinite(Cp_old):
        raise ValueError("Initial Cp not finite.")

    accepted = 0
    dt_last = float(dt0)
    s = 0

    for s in range(steps):
        ck = pauli_coeffs(H, mats, dim)
        M = build_M_from_coeffs(mats, ck, weights, p)
        dH = herm(comm(H, comm(H, M)))

        dt = float(dt0)
        ok = False
        while dt >= dt_min:
            H_new = herm(H - dt * dH)
            if not is_finite_matrix(H_new):
                dt *= backtrack
                continue

            ck_new = pauli_coeffs(H_new, mats, dim)
            Cp_new = locality_cost_cp(ck_new, weights, p)
            if np.isfinite(Cp_new) and (Cp_new <= Cp_old + eps_accept):
                H = H_new
                Cp_old = Cp_new
                accepted += 1
                dt_last = dt
                ok = True
                break

            dt *= backtrack

        if not ok:
            break

    stats = {
        "accepted_steps": int(accepted),
        "final_iter": int(s),
        "best_Cp": float(Cp_old),
        "dt_last": float(dt_last),
        "stopped_early": bool(accepted < steps),
    }
    return H, stats


# ----------------------------
# Driver
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="hsf_defense_sweep_gpu")
    ap.add_argument("--Ns", type=str, default="3,4,5,6,7,8")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seed1", type=int, default=19)
    ap.add_argument("--p", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--dt_min", type=float, default=1e-10)
    ap.add_argument("--scramble-depth-factor", type=int, default=2)
    ap.add_argument("--wmax", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    cp.cuda.Device(int(args.device)).use()

    os.makedirs(args.outdir, exist_ok=True)
    Ns = [int(x.strip()) for x in args.Ns.split(",") if x.strip()]
    seeds = list(range(int(args.seed0), int(args.seed1) + 1))

    rows: List[SweepRow] = []
    perN: Dict[int, Dict[str, int]] = {}

    for N in Ns:
        print(f"\n=== N={N} ===")
        dim = 2 ** N

        # Build truncated basis once per N
        labels, mats, weights = pauli_strings_truncated(N, int(args.wmax))
        print(f"  truncated basis size: {len(mats)}  (wmax={args.wmax})  dim={dim}")

        # Reference spatial Hamiltonian on a line (adjacent)
        H_spatial = heisenberg_chain(N, J=1.0)

        # Cp_spatial in truncated basis
        ck_spat = pauli_coeffs(H_spatial, mats, dim)
        Cp_spatial = locality_cost_cp(ck_spat, weights, float(args.p))

        # Cp_harm_base from diag-spectrum baseline
        Cp_harm_base = diag_spectrum_baseline_cost(H_spatial, N, float(args.p), mats, weights)

        print(f"  Cp_spatial(target)={Cp_spatial:.6g}   Cp_harm_base(diag-spectrum)={Cp_harm_base:.6g}")

        counts = {"harm_nearest": 0, "spatial_nearest": 0, "total": 0}

        for seed in seeds:
            rng = np.random.default_rng(int(seed))

            # scramble
            depth = int(args.scramble_depth_factor) * int(N)
            H_scr = scramble_hamiltonian_adjacent(H_spatial, N, depth=depth, rng=rng)

            # init Cp
            ck_scr = pauli_coeffs(H_scr, mats, dim)
            Cp_init = locality_cost_cp(ck_scr, weights, float(args.p))

            # descent
            H_fin, stats = run_double_bracket_flow_monotone(
                H0=H_scr,
                N=N,
                mats=mats,
                weights=weights,
                p=float(args.p),
                steps=int(args.steps),
                dt0=float(args.dt),
                dt_min=float(args.dt_min),
            )

            ck_fin = pauli_coeffs(H_fin, mats, dim)
            Cp_final = locality_cost_cp(ck_fin, weights, float(args.p))

            # forced-choice basin
            dh = abs(Cp_final - Cp_harm_base)
            ds = abs(Cp_final - Cp_spatial)
            harm_nearest = int(dh < ds)
            spatial_nearest = int(not harm_nearest)

            counts["harm_nearest"] += harm_nearest
            counts["spatial_nearest"] += spatial_nearest
            counts["total"] += 1

            rows.append(SweepRow(
                N=N,
                seed=int(seed),
                Cp_init=float(Cp_init),
                Cp_final=float(Cp_final),
                Cp_spatial=float(Cp_spatial),
                Cp_harm_base=float(Cp_harm_base),
                harm_nearest=harm_nearest,
                spatial_nearest=spatial_nearest,
                accepted_steps=int(stats["accepted_steps"]),
                final_iter=int(stats["final_iter"]),
                dt_last=float(stats["dt_last"]),
            ))

        perN[N] = counts
        print(f"  counts: harm_nearest={counts['harm_nearest']} spatial_nearest={counts['spatial_nearest']} total={counts['total']}")

        # free some GPU memory before next N
        cp.get_default_memory_pool().free_all_blocks()

    # Write CSV
    csv_path = os.path.join(args.outdir, "occupancy_vs_N.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([field for field in SweepRow.__dataclass_fields__.keys()])
        for r in rows:
            w.writerow(list(asdict(r).values()))

    # Plot
    png_path = ""
    Ns_sorted = sorted(perN.keys())
    P_harm = []
    SE = []
    n_per = []
    for N in Ns_sorted:
        n = perN[N]["total"]
        ph = perN[N]["harm_nearest"] / max(1, n)
        se = math.sqrt(ph * (1 - ph) / max(1, n))
        P_harm.append(ph)
        SE.append(se)
        n_per.append(n)

    plt.figure(figsize=(10, 5.6))
    plt.errorbar(Ns_sorted, P_harm, yerr=SE, fmt="o-", capsize=4)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("System size N")
    plt.ylabel("P_harm_nearest(N)  (GPU, truncated basis)")
    plt.title("HSF defense sweep: occupancy vs N (forced-choice)")

    for N, ph, n in zip(Ns_sorted, P_harm, n_per):
        plt.annotate(f"n={n}", (N, ph), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    txt = (
        "Order parameter:\n"
        "  harm_win := |Cp_final - Cp_harm_base| < |Cp_final - Cp_spatial|\n\n"
        f"p={args.p}\nsteps={args.steps}\ndt0={args.dt}\ndt_min={args.dt_min}\n"
        f"scramble_depth={args.scramble_depth_factor}*N (adjacent only)\n"
        f"wmax={args.wmax} (Pauli weight truncation)\n"
        f"seeds={args.seed0}..{args.seed1} (count={len(seeds)})\n"
        f"device={args.device}\n"
    )
    plt.gcf().text(0.02, 0.02, txt, fontsize=9, family="monospace", va="bottom")
    plt.tight_layout()
    png_path = os.path.join(args.outdir, "occupancy_vs_N.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    # Write JSON summary
    jpath = os.path.join(args.outdir, "sweep.json")
    payload = {
        "config": {
            "Ns": Ns,
            "seed0": args.seed0,
            "seed1": args.seed1,
            "p": args.p,
            "steps": args.steps,
            "dt0": args.dt,
            "dt_min": args.dt_min,
            "scramble_depth_factor": args.scramble_depth_factor,
            "wmax": args.wmax,
            "device": args.device,
            "note": "Scramble uses adjacent (i,i+1) SU(4) gates only; spatial H is chain (not periodic).",
        },
        "perN_counts": perN,
        "files": {"csv": csv_path, "png": png_path, "json": jpath},
    }
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== DONE ===")
    print(f"CSV: {csv_path}")
    print(f"FIG: {png_path}")
    print(f"JSON:{jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())