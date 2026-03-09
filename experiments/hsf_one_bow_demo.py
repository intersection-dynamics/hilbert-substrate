#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hsf_one_bow_demo.py  (rewrite v2: bound state fixed via single-excitation sector)

Pillars:
  (1) Spatial locality emergence: scramble local Heisenberg ring and run monotone
      double-bracket descent on C_p.

  (2) Jordan–Wigner strings: verify fermionic anti-commutation; naive fails.

  (3) Bound state (REAL): tight-binding on the emergent ring in the 1-excitation
      sector (dimension N), with a local potential well. Auto-tunes V so the IPR
      is in a nice band (localized but not delta).

  (4) Gauge emergence: SU(2) minimal link demo, dB=2 fails, dB=4 works,
      [H,G]=0 and singlet sector exists.

Outputs:
  report.json + PNGs:
    cost_cp.png, weight_profile.png, dt_used.png,
    bound_state_sites.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.linalg import eigh as scipy_eigh
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ----------------------------
# Linear algebra helpers
# ----------------------------

def herm(A: np.ndarray) -> np.ndarray:
    return (A + A.conj().T) / 2.0

def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def fro_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))

def tr(A: np.ndarray) -> complex:
    return np.trace(A)

def eigh(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if HAVE_SCIPY:
        return scipy_eigh(H)
    return np.linalg.eigh(H)

def is_finite_matrix(A: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(A)))

def is_finite_scalar(x: float) -> bool:
    return bool(np.isfinite(x))


# ----------------------------
# Pauli ops / qubit utilities
# ----------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_all(ops: List[np.ndarray]) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out

def op_on_site(single: np.ndarray, site: int, N: int) -> np.ndarray:
    ops = [I2] * N
    ops[site] = single
    return kron_all(ops)

def two_site_op(A: np.ndarray, i: int, B: np.ndarray, j: int, N: int) -> np.ndarray:
    ops = [I2] * N
    ops[i] = A
    ops[j] = B
    return kron_all(ops)


# ----------------------------
# (1) Spatial locality: Heisenberg ring + scramble + flow
# ----------------------------

def heisenberg_ring(N: int, J: float = 1.0) -> np.ndarray:
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        H += J * (
            two_site_op(X, i, X, j, N)
            + two_site_op(Y, i, Y, j, N)
            + two_site_op(Z, i, Z, j, N)
        )
    return herm(H)

def random_su4(rng: np.random.Generator) -> np.ndarray:
    A = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / math.sqrt(2)
    Q, R = np.linalg.qr(A)
    ph = np.diag(R) / np.abs(np.diag(R))
    Q = Q @ np.diag(np.conj(ph))
    det = np.linalg.det(Q)
    Q = Q / det ** (1 / 4)
    return Q

def embed_two_qubit(U4: np.ndarray, i: int, j: int, N: int) -> np.ndarray:
    if i > j:
        i, j = j, i
    dim = 2**N
    U = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        bits = [(b >> k) & 1 for k in range(N)]
        bi, bj = bits[i], bits[j]
        sub_in = (bi << 1) | bj
        for sub_out in range(4):
            oi = (sub_out >> 1) & 1
            oj = sub_out & 1
            bits2 = bits.copy()
            bits2[i] = oi
            bits2[j] = oj
            b2 = 0
            for k in range(N):
                b2 |= (bits2[k] << k)
            U[b2, b] += U4[sub_out, sub_in]
    return U

def scramble_hamiltonian(H: np.ndarray, N: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    U = np.eye(2**N, dtype=complex)
    for _ in range(depth):
        i = int(rng.integers(0, N))
        j = int((i + int(rng.integers(1, N))) % N)
        U4 = random_su4(rng)
        Uij = embed_two_qubit(U4, i, j, N)
        U = Uij @ U
    return herm(U @ H @ U.conj().T)


# ---- Pauli basis, coefficients, and C_p ----

def pauli_strings(N: int) -> List[Tuple[str, np.ndarray, int]]:
    single = [("I", I2), ("X", X), ("Y", Y), ("Z", Z)]
    out: List[Tuple[str, np.ndarray, int]] = []

    def rec(pos: int, label: List[str], ops: List[np.ndarray]):
        if pos == N:
            if all(ch == "I" for ch in label):
                return
            w = sum(1 for ch in label if ch != "I")
            out.append(("".join(label), kron_all(ops), w))
            return
        for ch, op in single:
            rec(pos + 1, label + [ch], ops + [op])

    rec(0, [], [])
    return out

def pauli_coeffs(H: np.ndarray, basis: List[Tuple[str, np.ndarray, int]], N: int) -> np.ndarray:
    dim = 2**N
    ck = np.zeros((len(basis),), dtype=complex)
    for idx, (_, Pk, _) in enumerate(basis):
        ck[idx] = tr(H @ Pk) / dim
    return ck

def locality_cost_cp(ck: np.ndarray, weights: np.ndarray, p: float) -> float:
    if not np.all(np.isfinite(ck)):
        return float("nan")
    num = np.sum((weights**p) * (np.abs(ck) ** 2))
    den = np.sum((np.abs(ck) ** 2)) + 1e-30
    return float(num / den)

def weight_distribution(ck: np.ndarray, weights: np.ndarray) -> Dict[int, float]:
    if not np.all(np.isfinite(ck)):
        return {int(w): float("nan") for w in np.unique(weights)}
    pow2 = np.abs(ck) ** 2
    total = float(np.sum(pow2) + 1e-30)
    out: Dict[int, float] = {}
    for w in np.unique(weights):
        out[int(w)] = float(np.sum(pow2[weights == w]) / total)
    return out

def build_M_from_coeffs(basis: List[Tuple[str, np.ndarray, int]], ck: np.ndarray, p: float, N: int) -> np.ndarray:
    den = np.sum(np.abs(ck) ** 2) + 1e-30
    M = np.zeros((2**N, 2**N), dtype=complex)
    for (_, Pk, w), c in zip(basis, ck):
        M += (float(w) ** p) * c * Pk
    M *= (2.0 / den)
    return herm(M)

@dataclass
class FlowTrace:
    t: List[float]
    Cp: List[float]
    w1: List[float]
    w2: List[float]
    w3: List[float]
    w4plus: List[float]
    dt_used: List[float]

def run_double_bracket_flow_monotone(
    H0: np.ndarray,
    basis: List[Tuple[str, np.ndarray, int]],
    weights: np.ndarray,
    p: float,
    steps: int,
    dt0: float,
    dt_min: float = 1e-10,
    backtrack: float = 0.5,
    eps_accept: float = 1e-14,
) -> Tuple[np.ndarray, FlowTrace, Dict[str, object]]:
    N = int(round(math.log2(H0.shape[0])))
    H = herm(H0.copy())
    if not is_finite_matrix(H):
        raise ValueError("Initial H0 is not finite.")

    ck = pauli_coeffs(H, basis, N)
    Cp_old = locality_cost_cp(ck, weights, p)
    if not is_finite_scalar(Cp_old):
        raise ValueError("Initial Cp is not finite.")

    trace = FlowTrace(t=[], Cp=[], w1=[], w2=[], w3=[], w4plus=[], dt_used=[])

    for s in range(steps + 1):
        ck = pauli_coeffs(H, basis, N)
        Cp_now = locality_cost_cp(ck, weights, p)
        wd = weight_distribution(ck, weights)

        trace.t.append(s)  # will convert to accumulated dt
        trace.Cp.append(Cp_now)
        trace.w1.append(wd.get(1, 0.0))
        trace.w2.append(wd.get(2, 0.0))
        trace.w3.append(wd.get(3, 0.0))
        trace.w4plus.append(sum(v for k, v in wd.items() if k >= 4))
        trace.dt_used.append(0.0)

        if s == steps:
            break

        M = build_M_from_coeffs(basis, ck, p, N)
        dH = herm(comm(H, comm(H, M)))

        dt = dt0
        accepted = False
        while dt >= dt_min:
            H_new = herm(H - dt * dH)
            if not is_finite_matrix(H_new):
                dt *= backtrack
                continue
            ck_new = pauli_coeffs(H_new, basis, N)
            Cp_new = locality_cost_cp(ck_new, weights, p)
            if is_finite_scalar(Cp_new) and (Cp_new <= Cp_old + eps_accept):
                H = H_new
                Cp_old = Cp_new
                trace.dt_used[-1] = dt
                accepted = True
                break
            dt *= backtrack

        if not accepted:
            break

    # Convert step index to flow time by accumulated accepted dt
    t_accum = []
    t = 0.0
    for k in range(len(trace.dt_used)):
        t_accum.append(t)
        if k < len(trace.dt_used) - 1:
            t += trace.dt_used[k]
    trace.t = t_accum

    stats = {
        "accepted_steps": int(np.sum(np.array(trace.dt_used[:-1]) > 0)),
        "final_iter": len(trace.Cp) - 1,
        "stopped_early": (len(trace.Cp) < steps + 1),
        "best_Cp": float(np.nanmin(np.array(trace.Cp))),
    }
    return H, trace, stats


# ----------------------------
# (2) Jordan–Wigner witness
# ----------------------------

def sigma_plus() -> np.ndarray:
    return np.array([[0, 1], [0, 0]], dtype=complex)

def sigma_minus() -> np.ndarray:
    return np.array([[0, 0], [1, 0]], dtype=complex)

def jw_creation(j: int, N: int) -> np.ndarray:
    ops = []
    for k in range(N):
        if k < j:
            ops.append(Z)
        elif k == j:
            ops.append(sigma_plus())
        else:
            ops.append(I2)
    return kron_all(ops)

def jw_annihilation(j: int, N: int) -> np.ndarray:
    ops = []
    for k in range(N):
        if k < j:
            ops.append(Z)
        elif k == j:
            ops.append(sigma_minus())
        else:
            ops.append(I2)
    return kron_all(ops)

def naive_creation(j: int, N: int) -> np.ndarray:
    return op_on_site(sigma_plus(), j, N)

def max_anticomm_violation(cre_ops: List[np.ndarray], ann_ops: List[np.ndarray]) -> float:
    N = len(cre_ops)
    dim = cre_ops[0].shape[0]
    I = np.eye(dim, dtype=complex)
    worst = 0.0
    for i in range(N):
        for j in range(N):
            a = ann_ops[i]
            adag = cre_ops[j]
            A = a @ adag + adag @ a
            target = I if i == j else np.zeros_like(I)
            worst = max(worst, fro_norm(A - target))
            a2 = ann_ops[j]
            B = a @ a2 + a2 @ a
            worst = max(worst, fro_norm(B))
    return worst


# ----------------------------
# (3) Bound state: 1-excitation tight-binding on ring
# ----------------------------

def tb_ring_1exc(N: int, t_hop: float = 1.0) -> np.ndarray:
    """
    Tight-binding Hamiltonian on ring in single-excitation sector.
    Basis |j> (excitation at site j), j=0..N-1.
    H_ij = -t if i,j neighbors, else 0.
    """
    H = np.zeros((N, N), dtype=float)
    for j in range(N):
        jp = (j + 1) % N
        jm = (j - 1) % N
        H[j, jp] += -t_hop
        H[j, jm] += -t_hop
    return H

def add_local_well_1exc(H: np.ndarray, V: float, sites: Tuple[int, int] = (0, 1)) -> np.ndarray:
    H2 = H.copy()
    for s in sites:
        H2[s, s] += -V  # attractive
    return H2

def ipr(vec: np.ndarray) -> float:
    p = np.abs(vec) ** 2
    return float(np.sum(p**2))

def best_bound_state_1exc(H: np.ndarray) -> Dict[str, object]:
    evals, evecs = np.linalg.eigh(H)
    iprs = np.array([ipr(evecs[:, k]) for k in range(evecs.shape[1])], dtype=float)
    kmax = int(np.argmax(iprs))
    v = evecs[:, kmax]
    return {
        "eig_index": kmax,
        "E": float(evals[kmax]),
        "IPR": float(iprs[kmax]),
        "psi": v,
        "prob_sites": (np.abs(v) ** 2),
        "E_min": float(np.min(evals)),
        "E_max": float(np.max(evals)),
    }

def autotune_bound_state_1exc(
    N: int,
    t_hop: float,
    ipr_low: float,
    ipr_high: float,
    V_lo: float,
    V_hi: float,
    max_iters: int = 40,
) -> Dict[str, object]:
    H0 = tb_ring_1exc(N, t_hop=t_hop)
    lo, hi = V_lo, V_hi
    best = None

    for _ in range(max_iters):
        V = 0.5 * (lo + hi)
        H = add_local_well_1exc(H0, V=V, sites=(0, 1))
        info = best_bound_state_1exc(H)
        info["V"] = V
        best = info
        if ipr_low <= info["IPR"] <= ipr_high:
            return {"status": "hit", "result": info, "lo": lo, "hi": hi}
        if info["IPR"] > ipr_high:
            hi = V
        else:
            lo = V

    return {"status": "best_effort", "result": best, "lo": lo, "hi": hi}


# ----------------------------
# (4) Gauge emergence (SU(2) minimal link)
# ----------------------------

def su2_generators_half() -> List[np.ndarray]:
    return [0.5 * X, 0.5 * Y, 0.5 * Z]

def link_endpoints_su2(dB: int) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    T = su2_generators_half()
    if dB == 2:
        return T, T
    if dB == 4:
        I = I2
        L = [np.kron(t, I) for t in T]
        R = [np.kron(I, t) for t in T]
        return L, R
    return None, None

def max_comm_lr(L: List[np.ndarray], R: List[np.ndarray]) -> float:
    worst = 0.0
    for La in L:
        for Rb in R:
            worst = max(worst, fro_norm(comm(La, Rb)))
    return worst

def build_gauge_invariant_toy(J: float = 1.0) -> Dict[str, object]:
    T = su2_generators_half()
    SA = T
    SC = T
    L, R = link_endpoints_su2(4)
    assert L is not None and R is not None

    IA = I2
    IC = I2
    IB = np.eye(4, dtype=complex)

    def embed_A(opA: np.ndarray) -> np.ndarray:
        return np.kron(np.kron(opA, IB), IC)

    def embed_B(opB: np.ndarray) -> np.ndarray:
        return np.kron(np.kron(IA, opB), IC)

    def embed_C(opC: np.ndarray) -> np.ndarray:
        return np.kron(np.kron(IA, IB), opC)

    GL = [embed_A(SA[a]) + embed_B(L[a]) for a in range(3)]
    GR = [embed_B(R[a]) + embed_C(SC[a]) for a in range(3)]

    H = np.zeros((16, 16), dtype=complex)
    for a in range(3):
        H += -J * (embed_A(SA[a]) @ embed_B(L[a]) + embed_B(R[a]) @ embed_C(SC[a]))
    H = herm(H)

    comm_GL = [fro_norm(comm(H, G)) for G in GL]
    comm_GR = [fro_norm(comm(H, G)) for G in GR]
    max_commG = float(max(comm_GL + comm_GR))

    G2 = np.zeros_like(H)
    for a in range(3):
        G2 += GL[a] @ GL[a] + GR[a] @ GR[a]
    G2 = herm(G2)
    evals, _ = eigh(G2)
    tol = 1e-9
    dim_singlet = int(np.sum(np.abs(evals) < tol))

    return {
        "max_norm_comm_H_G": max_commG,
        "comm_H_GL": comm_GL,
        "comm_H_GR": comm_GR,
        "gauss_singlet_dim_tol1e-9": dim_singlet,
        "G2_min_eig": float(np.min(evals)),
    }


# ----------------------------
# Plotting
# ----------------------------

def maybe_plot_flow(outdir: str, trace: FlowTrace) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return paths

    fig = plt.figure()
    plt.plot(trace.t, trace.Cp)
    plt.xlabel("flow time")
    plt.ylabel("locality cost C_p")
    plt.title("Spatial locality emergence: C_p descent")
    p1 = os.path.join(outdir, "cost_cp.png")
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["cost_cp.png"] = p1

    fig = plt.figure()
    plt.plot(trace.t, trace.w1, label="w=1")
    plt.plot(trace.t, trace.w2, label="w=2")
    plt.plot(trace.t, trace.w3, label="w=3")
    plt.plot(trace.t, trace.w4plus, label="w>=4")
    plt.xlabel("flow time")
    plt.ylabel("fraction of ||H||^2 in weight sector")
    plt.title("Locality weight profile")
    plt.legend()
    p2 = os.path.join(outdir, "weight_profile.png")
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["weight_profile.png"] = p2

    fig = plt.figure()
    plt.plot(trace.t[:-1], trace.dt_used[:-1])
    plt.xlabel("flow time")
    plt.ylabel("accepted dt")
    plt.title("Backtracking step sizes")
    p3 = os.path.join(outdir, "dt_used.png")
    fig.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["dt_used.png"] = p3

    return paths

def plot_bound_sites(outdir: str, prob_sites: np.ndarray, V: float, ipr_val: float) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return paths

    N = prob_sites.size
    fig = plt.figure()
    plt.bar(np.arange(N), prob_sites)
    plt.xlabel("site")
    plt.ylabel("|psi(site)|^2")
    plt.title(f"1-excitation bound state on ring: V={V:.4g}, IPR={ipr_val:.3f}")
    p = os.path.join(outdir, "bound_state_sites.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["bound_state_sites.png"] = p
    return paths


# ----------------------------
# Report
# ----------------------------

@dataclass
class Report:
    config: Dict[str, object]
    spatial_locality: Dict[str, object]
    jordan_wigner: Dict[str, object]
    bound_states: Dict[str, object]
    gauge_emergence: Dict[str, object]
    files: Dict[str, str]


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="hsf_one_bow_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--p", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--dt_min", type=float, default=1e-10)
    ap.add_argument("--scramble_depth", type=int, default=-1)

    # bound state targets (meaningful now)
    ap.add_argument("--ipr_low", type=float, default=0.15)
    ap.add_argument("--ipr_high", type=float, default=0.35)
    ap.add_argument("--V_lo", type=float, default=0.0)
    ap.add_argument("--V_hi", type=float, default=6.0)
    ap.add_argument("--t_hop", type=float, default=1.0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # (1) Spatial locality
    basis = pauli_strings(args.N)
    weights = np.array([w for (_, _, w) in basis], dtype=float)

    H_spatial = heisenberg_ring(args.N, J=1.0)
    depth = args.scramble_depth if args.scramble_depth >= 0 else args.N
    H_scrambled = scramble_hamiltonian(H_spatial, args.N, depth=depth, rng=rng)

    ck_spatial = pauli_coeffs(H_spatial, basis, args.N)
    ck_scr = pauli_coeffs(H_scrambled, basis, args.N)
    Cp_spatial = locality_cost_cp(ck_spatial, weights, args.p)
    Cp_scr = locality_cost_cp(ck_scr, weights, args.p)

    evals, _ = eigh(H_spatial)
    H_diag = np.diag(evals)
    ck_diag = pauli_coeffs(H_diag, basis, args.N)
    Cp_harmonion_ref = locality_cost_cp(ck_diag, weights, args.p)

    H_final, trace, flow_stats = run_double_bracket_flow_monotone(
        H0=H_scrambled,
        basis=basis,
        weights=weights,
        p=args.p,
        steps=args.steps,
        dt0=args.dt,
        dt_min=args.dt_min,
        backtrack=0.5,
        eps_accept=1e-14,
    )

    ck_final = pauli_coeffs(H_final, basis, args.N)
    Cp_final = locality_cost_cp(ck_final, weights, args.p)

    spatial_block = {
        "N": args.N,
        "p": args.p,
        "scramble_depth": depth,
        "Cp_spatial_target": Cp_spatial,
        "Cp_scrambled_start": Cp_scr,
        "Cp_final": Cp_final,
        "Cp_harmonion_reference_diag": Cp_harmonion_ref,
        "final_weight_distribution": weight_distribution(ck_final, weights),
        "start_weight_distribution": weight_distribution(ck_scr, weights),
        **flow_stats,
    }

    # (2) JW
    jw_cre = [jw_creation(j, args.N) for j in range(args.N)]
    jw_ann = [jw_annihilation(j, args.N) for j in range(args.N)]
    naive_cre = [naive_creation(j, args.N) for j in range(args.N)]
    naive_ann = [c.conj().T for c in naive_cre]
    jw_violation = max_anticomm_violation(jw_cre, jw_ann)
    naive_violation = max_anticomm_violation(naive_cre, naive_ann)
    jw_block = {
        "N": args.N,
        "max_anticommutator_violation_JW": jw_violation,
        "max_anticommutator_violation_naive_no_strings": naive_violation,
    }

    # (3) Bound state in 1-excitation sector
    tuned = autotune_bound_state_1exc(
        N=args.N,
        t_hop=args.t_hop,
        ipr_low=args.ipr_low,
        ipr_high=args.ipr_high,
        V_lo=args.V_lo,
        V_hi=args.V_hi,
        max_iters=40,
    )
    res = tuned["result"]
    bound_block = {
        "model": "1-excitation tight-binding ring + 2-site attractive well",
        "tuning_status": tuned["status"],
        "ipr_target_band": [args.ipr_low, args.ipr_high],
        "V": float(res["V"]),
        "metrics": {
            "E": float(res["E"]),
            "IPR": float(res["IPR"]),
            "eig_index": int(res["eig_index"]),
            "E_min": float(res["E_min"]),
            "E_max": float(res["E_max"]),
        },
        "prob_sites": res["prob_sites"].tolist(),
    }

    # (4) Gauge
    L2, R2 = link_endpoints_su2(2)
    comm_d2 = max_comm_lr(L2, R2) if (L2 is not None and R2 is not None) else None
    L4, R4 = link_endpoints_su2(4)
    comm_d4 = max_comm_lr(L4, R4) if (L4 is not None and R4 is not None) else None
    gauge_toy = build_gauge_invariant_toy(J=1.0)
    gauge_block = {
        "endpoint_commutativity": {
            "dB_2_single_qubit_link_max||[L,R]||F": float(comm_d2) if comm_d2 is not None else None,
            "dB_4_composite_link_max||[L,R]||F": float(comm_d4) if comm_d4 is not None else None,
        },
        "gauge_invariant_hamiltonian_demo": gauge_toy,
    }

    # Plots
    files: Dict[str, str] = {}
    files.update(maybe_plot_flow(args.outdir, trace))
    files.update(plot_bound_sites(args.outdir, np.array(res["prob_sites"]), float(res["V"]), float(res["IPR"])))

    # Write report
    report = Report(
        config={
            "outdir": args.outdir,
            "seed": args.seed,
            "N": args.N,
            "p": args.p,
            "steps": args.steps,
            "dt0": args.dt,
            "dt_min": args.dt_min,
            "scramble_depth": depth,
            "scipy": HAVE_SCIPY,
            "ipr_low": args.ipr_low,
            "ipr_high": args.ipr_high,
            "V_lo": args.V_lo,
            "V_hi": args.V_hi,
            "t_hop": args.t_hop,
        },
        spatial_locality=spatial_block,
        jordan_wigner=jw_block,
        bound_states=bound_block,
        gauge_emergence=gauge_block,
        files=files,
    )

    jpath = os.path.join(args.outdir, "report.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print("============================================================")
    print("HSF ONE-BOW DEMO (bound state fixed: 1-excitation sector)")
    print("============================================================")
    print(f"[Spatial] Cp start={Cp_scr:.6g}  Cp final={Cp_final:.6g}  accepted_steps={spatial_block['accepted_steps']}")
    print(f"[JW]      JW_violation={jw_violation:.3e}  naive_violation={naive_violation:.3e}")
    print(f"[Bound]   status={tuned['status']}  V={res['V']:.4g}  IPR={res['IPR']:.3f}  E={res['E']:.4g}")
    print(f"[Gauge]   dB2_comm={comm_d2:.3e}  dB4_comm={comm_d4:.3e}  max||[H,G]||={gauge_toy['max_norm_comm_H_G']:.3e}")
    print(f"Wrote: {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())