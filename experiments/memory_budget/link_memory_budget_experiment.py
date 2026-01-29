#!/usr/bin/env python3
"""
Link-Memory + Soft Capacity Budget Experiment
=============================================

Goal
----
Study whether a sparse "mesoscopic fabric" and locality-like transport can emerge when:

  (1) Links J_ij update via a rule (learn/random/frozen)
  (2) Links are constrained by a soft per-node capacity budget (row-L1 ~ Lambda)

Key diagnostics
---------------
A) Structure: sparsity, degrees, gini(|J|), row-L1 budget stats
B) Thresholded BFS distance vs influence (legacy)
C) Weighted-distance (Dijkstra, no thresholds) vs influence (preferred)

Patch v5 (final)
----------------
1) Seed sweep: --seeds, --seed-start
   - Writes per-seed JSON outputs + one summary JSON.
2) Clean output:
   - One line per seed with the key stats.
   - Compact "SUMMARY" table at the end.
3) Output cleanup:
   - --history-every controls snapshot density (0 disables history)
   - --save-J toggles saving full J_final (default off)

Probe decoupling (important)
----------------------------
--probe-mode directJ:
    Influence probe evolution uses true coupling strengths J_ij (old behavior).
    Caveat: distance derived from |J| and influence both depend on |J|.

--probe-mode masked_uniform:
    Influence probe evolution uses UNIFORM strength gates, but only on edges allowed by J.
    J sets "who can talk", not "how hard" — reduces trivial metric coupling.

Also supports --k-steps K for transport-style influence (multi-step).

Typical decisive run (seed sweep)
---------------------------------
python link_memory_budget_experiment.py --control learn --probe-mode masked_uniform --k-steps 3 --mask-thr 0.05 --uniform-strength 1.0 --N 10 --steps 200 --dt 0.05 --pairs-per-step 40 --eta 0.20 --decay 0.01 --budget 3.5 --budget-iters 6 --seeds 10 --seed-start 0 --out results_learn_masked.json --progress

Then repeat for --control frozen/random with same args.
"""

from __future__ import annotations

import argparse
import json
import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def normalize_state(psi: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(psi)
    if n <= 0:
        raise ValueError("State norm is zero.")
    return psi / n


def nearest_unitary(U: np.ndarray) -> np.ndarray:
    X, _, Yh = np.linalg.svd(U)
    return X @ Yh


def paulis() -> Dict[str, np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


# =============================================================================
# State ops
# =============================================================================

def single_qubit_rho(psi: np.ndarray, N: int, q: int) -> np.ndarray:
    psi_t = psi.reshape([2] * N)
    axes = [q] + [i for i in range(N) if i != q]
    psi_perm = np.transpose(psi_t, axes).reshape(2, -1)
    rho = psi_perm @ psi_perm.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    return rho


def trace_distance_2x2(rho: np.ndarray, sigma: np.ndarray) -> float:
    delta = rho - sigma
    delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))


def apply_two_qubit_gate_statevector(psi: np.ndarray, N: int, a: int, b: int, U4: np.ndarray) -> np.ndarray:
    if a == b:
        return psi
    if a > b:
        a, b = b, a

    psi_t = psi.reshape([2] * N)
    axes = [i for i in range(N) if i not in (a, b)] + [a, b]
    inv_axes = np.argsort(axes)
    psi_perm = np.transpose(psi_t, axes)
    rest_dim = 2 ** (N - 2)
    psi_mat = psi_perm.reshape(rest_dim, 4)

    psi_mat2 = psi_mat @ U4.T
    psi_perm2 = psi_mat2.reshape([2] * (N - 2) + [2, 2])
    psi_t2 = np.transpose(psi_perm2, inv_axes).reshape(-1)
    return psi_t2


def apply_single_qubit_unitary(psi: np.ndarray, N: int, q: int, U2: np.ndarray) -> np.ndarray:
    psi_t = psi.reshape([2] * N)
    axes = [q] + [i for i in range(N) if i != q]
    inv_axes = np.argsort(axes)
    psi_perm = np.transpose(psi_t, axes).reshape(2, -1)
    psi_perm2 = (U2 @ psi_perm).reshape([2] + [2] * (N - 1))
    psi_t2 = np.transpose(psi_perm2, inv_axes).reshape(-1)
    return psi_t2


# =============================================================================
# Gates / interactions
# =============================================================================

def two_qubit_unitary_xx_yy_zz(dt: float, J: float, Delta: float) -> np.ndarray:
    P = paulis()
    XX = np.kron(P["X"], P["X"])
    YY = np.kron(P["Y"], P["Y"])
    ZZ = np.kron(P["Z"], P["Z"])
    H = J * (XX + YY + Delta * ZZ)
    w, V = np.linalg.eigh(H)
    U = V @ np.diag(np.exp(-1j * dt * w)) @ V.conj().T
    return U


def random_product_state(N: int, rng: np.random.Generator) -> np.ndarray:
    psi = np.array([1.0 + 0j])
    for _ in range(N):
        v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        v = v / (np.linalg.norm(v) + 1e-12)
        psi = np.kron(psi, v)
    return normalize_state(psi)


def random_single_qubit_unitary(rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    Q, R = np.linalg.qr(A)
    Q = Q * np.exp(-1j * np.angle(np.diag(R)))
    return Q


# =============================================================================
# Budget ops
# =============================================================================

def initialize_J(N: int, rng: np.random.Generator, scale: float, clip: float) -> np.ndarray:
    A = rng.standard_normal((N, N))
    J = 0.5 * (A + A.T)
    np.fill_diagonal(J, 0.0)
    J = scale * J / (np.std(J) + 1e-12)
    J = np.clip(J, -clip, clip)
    return J


def row_l1_offdiag(J: np.ndarray) -> np.ndarray:
    A = np.abs(J)
    np.fill_diagonal(A, 0.0)
    return A.sum(axis=1)


def apply_soft_budget_symmetric(J: np.ndarray, budget: float, iters: int = 6) -> np.ndarray:
    """
    Symmetric multiplicative scaling to approximately enforce per-node row-L1(|J|)=budget
      J_ij <- J_ij * sqrt(a_i a_j)
    """
    if budget <= 0:
        return J

    N = J.shape[0]
    a = np.ones(N, dtype=np.float64)

    J_work = J.copy()
    np.fill_diagonal(J_work, 0.0)

    eps = 1e-12
    for _ in range(max(1, iters)):
        S = row_l1_offdiag(J_work) + eps
        f = budget / S
        a *= f
        g = np.sqrt(np.outer(a, a))
        J_work = J * g
        np.fill_diagonal(J_work, 0.0)
        J_work = 0.5 * (J_work + J_work.T)

    return J_work


# =============================================================================
# Diagnostics
# =============================================================================

def summarize_structure(J: np.ndarray, thr: float = 0.5) -> Dict:
    N = J.shape[0]
    A = (np.abs(J) >= thr).astype(np.int32)
    np.fill_diagonal(A, 0)
    degrees = A.sum(axis=1).tolist()
    m = int(A.sum() // 2)

    vals = np.abs(J[np.triu_indices(N, 1)])
    vals_sorted = np.sort(vals)

    if np.all(vals_sorted < 1e-12):
        gini = 0.0
    else:
        n = len(vals_sorted)
        cum = np.cumsum(vals_sorted)
        gini = float((n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-12)) / n)

    rl1 = row_l1_offdiag(J)
    return {
        "thr": float(thr),
        "edges_ge_thr": m,
        "deg_min": int(np.min(degrees)),
        "deg_max": int(np.max(degrees)),
        "deg_mean": float(np.mean(degrees)),
        "deg_list": degrees,
        "gini_absJ": float(gini),
        "absJ_mean": float(np.mean(vals)),
        "absJ_std": float(np.std(vals)),
        "rowL1_min": float(np.min(rl1)),
        "rowL1_max": float(np.max(rl1)),
        "rowL1_mean": float(np.mean(rl1)),
        "rowL1_std": float(np.std(rl1)),
    }


def influence_vs_distance(J: np.ndarray, infl_samples: List[Tuple[int, int, float]], thr: float = 0.5) -> Dict:
    N = J.shape[0]
    A = (np.abs(J) >= thr).astype(np.int32)
    np.fill_diagonal(A, 0)

    def bfs(src: int) -> List[float]:
        dist = [math.inf] * N
        dist[src] = 0
        q = [src]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in np.where(A[u] > 0)[0]:
                if dist[v] == math.inf:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    srcs = sorted(set(i for (i, _, _) in infl_samples))
    dist_map = {s: bfs(s) for s in srcs}

    pairs = []
    for (i, j, val) in infl_samples:
        d = dist_map[i][j]
        pairs.append((d, val))

    finite = [(d, v) for (d, v) in pairs if math.isfinite(d)]
    if not finite:
        return {"thr": float(thr), "n": len(pairs), "n_finite": 0, "corr_dist_influence": None, "mean_influence_by_dist": {}}

    ds = np.array([d for (d, _) in finite], dtype=np.float64)
    vs = np.array([v for (_, v) in finite], dtype=np.float64)

    corr = None
    if np.std(ds) > 1e-12 and np.std(vs) > 1e-12:
        corr = float(np.corrcoef(ds, vs)[0, 1])

    out_bins = {}
    for d in sorted(set(ds.tolist())):
        mask = ds == d
        out_bins[int(d)] = float(np.mean(vs[mask]))

    return {
        "thr": float(thr),
        "n": len(pairs),
        "n_finite": int(len(finite)),
        "corr_dist_influence": corr,
        "mean_influence_by_dist": out_bins,
    }


def weighted_influence_vs_distance(
    J: np.ndarray,
    infl_samples: List[Tuple[int, int, float]],
    eps: float = 1e-6,
    bins: int = 6,
) -> Dict:
    """
    Weighted shortest-path distance using edge length = 1/(|J_ij|+eps).
    Threshold-free.
    """
    N = J.shape[0]
    absJ = np.abs(J).astype(np.float64)
    np.fill_diagonal(absJ, 0.0)

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = absJ[i, j]
            if w <= 0.0:
                continue
            length = 1.0 / (w + float(eps))
            adj[i].append((j, length))

    def dijkstra(src: int) -> List[float]:
        dist = [math.inf] * N
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, wlen in adj[u]:
                nd = d + wlen
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    srcs = sorted(set(i for (i, _, _) in infl_samples))
    dist_map = {s: dijkstra(s) for s in srcs}

    ds: List[float] = []
    vs: List[float] = []
    for (i, j, val) in infl_samples:
        d = dist_map[i][j]
        if math.isfinite(d):
            ds.append(float(d))
            vs.append(float(val))

    if len(ds) == 0:
        return {
            "eps": float(eps), "bins": int(bins),
            "n": int(len(infl_samples)), "n_finite": 0,
            "corr_wdist_influence": None,
            "slope_influence_vs_wdist": None,
            "intercept_influence_vs_wdist": None,
            "mean_influence_by_wdist_bin": []
        }

    ds_arr = np.array(ds, dtype=np.float64)
    vs_arr = np.array(vs, dtype=np.float64)

    corr = None
    if np.std(ds_arr) > 1e-12 and np.std(vs_arr) > 1e-12:
        corr = float(np.corrcoef(ds_arr, vs_arr)[0, 1])

    slope = None
    intercept = None
    if np.std(ds_arr) > 1e-12:
        A = np.vstack([np.ones_like(ds_arr), ds_arr]).T
        coeff, *_ = np.linalg.lstsq(A, vs_arr, rcond=None)
        intercept = float(coeff[0])
        slope = float(coeff[1])

    order = np.argsort(ds_arr)
    ds_s = ds_arr[order]
    vs_s = vs_arr[order]
    nb = max(1, int(bins))
    splits = np.array_split(np.arange(len(ds_s)), nb)
    bin_means = []
    for idx in splits:
        if len(idx) == 0:
            continue
        d_lo = float(ds_s[idx[0]])
        d_hi = float(ds_s[idx[-1]])
        v_mean = float(np.mean(vs_s[idx]))
        bin_means.append({"d_lo": d_lo, "d_hi": d_hi, "mean_influence": v_mean, "n": int(len(idx))})

    return {
        "eps": float(eps),
        "bins": int(bins),
        "n": int(len(infl_samples)),
        "n_finite": int(len(ds_arr)),
        "corr_wdist_influence": corr,
        "slope_influence_vs_wdist": slope,
        "intercept_influence_vs_wdist": intercept,
        "mean_influence_by_wdist_bin": bin_means,
    }


# =============================================================================
# Core simulation
# =============================================================================

@dataclass
class Params:
    N: int
    steps: int
    dt: float
    Delta: float
    pairs_per_step: int
    eta: float
    decay: float
    J_init_scale: float
    J_clip: float
    influence_eps: float
    budget: float
    budget_iters: int
    control: str
    seed: int
    wd_eps: float
    wd_bins: int
    probe_mode: str
    k_steps: int
    mask_thr: float
    uniform_strength: float
    history_every: int
    save_J: bool


def sample_edges_for_step(N: int, rng: np.random.Generator, sample: bool) -> List[Tuple[int, int]]:
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if not sample:
        edges = all_edges
    else:
        m = min(len(all_edges), max(10, (N * (N - 1)) // 4))
        idx = rng.choice(len(all_edges), size=m, replace=False)
        edges = [all_edges[k] for k in idx]
    rng.shuffle(edges)
    return edges


def evolve_one_step_with_edges(
    psi: np.ndarray,
    J: np.ndarray,
    params: Params,
    edges: List[Tuple[int, int]],
    *,
    mode: str,
) -> np.ndarray:
    """
    mode:
      - directJ: use Jij for gate strength
      - masked_uniform: use uniform_strength on edges where |J| > mask_thr
    """
    N = params.N
    dt = params.dt
    Delta = params.Delta

    psi2 = psi

    if mode == "directJ":
        for (i, j) in edges:
            Jij = float(J[i, j])
            if abs(Jij) < 1e-12:
                continue
            U4 = two_qubit_unitary_xx_yy_zz(dt, Jij, Delta)
            psi2 = apply_two_qubit_gate_statevector(psi2, N, i, j, U4)

    elif mode == "masked_uniform":
        thr = float(params.mask_thr)
        mag = float(params.uniform_strength)
        for (i, j) in edges:
            Jij = float(J[i, j])
            if abs(Jij) <= thr:
                continue
            # preserve sign to avoid bias, but fixed magnitude
            J_eff = float(np.sign(Jij) * mag)
            if abs(J_eff) < 1e-12:
                continue
            U4 = two_qubit_unitary_xx_yy_zz(dt, J_eff, Delta)
            psi2 = apply_two_qubit_gate_statevector(psi2, N, i, j, U4)
    else:
        raise ValueError(f"Unknown evolve mode: {mode}")

    return normalize_state(psi2)


def evolve_k_steps(
    psi: np.ndarray,
    J: np.ndarray,
    params: Params,
    rng: np.random.Generator,
    *,
    mode: str,
    k: int,
) -> np.ndarray:
    psi2 = psi
    for _ in range(max(1, int(k))):
        edges = sample_edges_for_step(params.N, rng, sample=(params.N > 10))
        psi2 = evolve_one_step_with_edges(psi2, J, params, edges, mode=mode)
    return psi2


def estimate_influence_pair(
    psi: np.ndarray,
    J: np.ndarray,
    params: Params,
    rng: np.random.Generator,
    src: int,
    dst: int,
) -> float:
    """
    Influence src->dst:
      - perturb src slightly
      - evolve base and perturbed k steps with identical schedules
      - compare reduced state at dst
    """
    N = params.N

    U_rand = random_single_qubit_unitary(rng)
    eps = float(params.influence_eps)
    U_mix = (1.0 - eps) * np.eye(2, dtype=np.complex128) + eps * U_rand
    U2 = nearest_unitary(U_mix)
    psi_pert = apply_single_qubit_unitary(psi, N, src, U2)

    # Clone RNG state to guarantee identical schedules
    bitgen_state = rng.bit_generator.state

    rng_base = np.random.default_rng()
    rng_base.bit_generator.state = bitgen_state
    psi_a = evolve_k_steps(psi, J, params, rng_base, mode=params.probe_mode, k=params.k_steps)

    rng_pert = np.random.default_rng()
    rng_pert.bit_generator.state = bitgen_state
    psi_b = evolve_k_steps(psi_pert, J, params, rng_pert, mode=params.probe_mode, k=params.k_steps)

    # Advance original RNG to match having consumed schedules once
    rng.bit_generator.state = rng_base.bit_generator.state

    rho_a = single_qubit_rho(psi_a, N, dst)
    rho_b = single_qubit_rho(psi_b, N, dst)
    return trace_distance_2x2(rho_a, rho_b)


def update_links(J: np.ndarray, influences: List[Tuple[int, int, float]], params: Params, rng: np.random.Generator) -> np.ndarray:
    """
    control:
      - frozen: no update (still budget-projected)
      - learn  : increments proportional to measured influence
      - random : random increments on sampled pairs
    """
    if params.control == "frozen":
        J2 = J.copy()
        J2 = apply_soft_budget_symmetric(J2, params.budget, params.budget_iters)
        J2 = np.clip(J2, -params.J_clip, params.J_clip)
        np.fill_diagonal(J2, 0.0)
        return J2

    eta = float(params.eta)
    decay = float(params.decay)

    J2 = (1.0 - decay) * J
    inc = np.zeros_like(J2)

    if params.control == "learn":
        for (i, j, val) in influences:
            inc[i, j] += val
            inc[j, i] += val
    elif params.control == "random":
        for (i, j, _val) in influences:
            r = float(rng.standard_normal())
            inc[i, j] += r
            inc[j, i] += r
    else:
        raise ValueError(f"Unknown control mode: {params.control}")

    max_abs = float(np.max(np.abs(inc)))
    if max_abs > 1e-12:
        inc = inc / max_abs

    J2 = J2 + eta * inc
    J2 = 0.5 * (J2 + J2.T)
    np.fill_diagonal(J2, 0.0)
    J2 = np.clip(J2, -params.J_clip, params.J_clip)

    J2 = apply_soft_budget_symmetric(J2, params.budget, params.budget_iters)
    J2 = np.clip(J2, -params.J_clip, params.J_clip)
    np.fill_diagonal(J2, 0.0)
    return J2


def run_one(params: Params) -> Dict:
    rng = np.random.default_rng(params.seed)

    psi = random_product_state(params.N, rng)
    J = initialize_J(params.N, rng, params.J_init_scale, params.J_clip)
    J = apply_soft_budget_symmetric(J, params.budget, params.budget_iters)
    J = np.clip(J, -params.J_clip, params.J_clip)
    np.fill_diagonal(J, 0.0)

    history = []
    infl_buffer_last: List[Tuple[int, int, float]] = []

    he = int(params.history_every)
    if he < 0:
        he = 0

    for step in range(params.steps):
        infl_samples: List[Tuple[int, int, float]] = []
        for _ in range(params.pairs_per_step):
            i = int(rng.integers(0, params.N))
            j = int(rng.integers(0, params.N - 1))
            if j >= i:
                j += 1
            val = estimate_influence_pair(psi, J, params, rng, i, j)
            infl_samples.append((i, j, float(val)))

        infl_buffer_last = infl_samples
        J = update_links(J, infl_samples, params, rng)

        # "World evolution" always uses directJ
        edges = sample_edges_for_step(params.N, rng, sample=(params.N > 10))
        psi = evolve_one_step_with_edges(psi, J, params, edges, mode="directJ")

        if he > 0 and ((step % he) == 0 or step == params.steps - 1):
            struct = summarize_structure(J, thr=0.5)
            dist_cmp = influence_vs_distance(J, infl_samples, thr=0.5)
            wdist_cmp = weighted_influence_vs_distance(J, infl_samples, eps=params.wd_eps, bins=params.wd_bins)
            history.append({
                "step": int(step),
                "structure": struct,
                "dist_cmp": dist_cmp,
                "wdist_cmp": wdist_cmp,
                "mean_influence": float(np.mean([v for (_, _, v) in infl_samples])),
                "max_influence": float(np.max([v for (_, _, v) in infl_samples])),
            })

    final_struct = summarize_structure(J, thr=0.5)
    final_dist = influence_vs_distance(J, infl_buffer_last, thr=0.5)
    final_wdist = weighted_influence_vs_distance(J, infl_buffer_last, eps=params.wd_eps, bins=params.wd_bins)

    out = {
        "meta": {
            "N": params.N,
            "steps": params.steps,
            "dt": params.dt,
            "Delta": params.Delta,
            "pairs_per_step": params.pairs_per_step,
            "eta": params.eta,
            "decay": params.decay,
            "J_init_scale": params.J_init_scale,
            "J_clip": params.J_clip,
            "influence_eps": params.influence_eps,
            "budget": params.budget,
            "budget_iters": params.budget_iters,
            "control": params.control,
            "seed": params.seed,
            "wd_eps": params.wd_eps,
            "wd_bins": params.wd_bins,
            "probe_mode": params.probe_mode,
            "k_steps": params.k_steps,
            "mask_thr": params.mask_thr,
            "uniform_strength": params.uniform_strength,
            "history_every": params.history_every,
            "save_J": params.save_J,
        },
        "final": {
            "structure": final_struct,
            "dist_cmp": final_dist,
            "wdist_cmp": final_wdist,
        },
    }

    if he > 0:
        out["history"] = history
    if params.save_J:
        out["J_final"] = J.tolist()

    return out


# =============================================================================
# Output + sweep
# =============================================================================

def fmt_float(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "None"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NaN"
    return f"{x:+.{nd}f}"


def write_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def seed_out_path(base_out: str, seed: int) -> str:
    # results_x.json -> results_x_seed000.json
    if base_out.lower().endswith(".json"):
        stem = base_out[:-5]
        return f"{stem}_seed{seed:03d}.json"
    return f"{base_out}_seed{seed:03d}.json"


def summary_out_path(base_out: str) -> str:
    if base_out.lower().endswith(".json"):
        stem = base_out[:-5]
        return f"{stem}_SUMMARY.json"
    return f"{base_out}_SUMMARY.json"


def print_seed_line(res: Dict) -> None:
    fin = res["final"]
    st = fin["structure"]
    wd = fin["wdist_cmp"]

    corr = wd.get("corr_wdist_influence", None)
    slope = wd.get("slope_influence_vs_wdist", None)

    print(
        f"[seed {res['meta']['seed']:3d}] "
        f"edges>=0.5={st['edges_ge_thr']:2d} "
        f"deg_mean={st['deg_mean']:.2f} "
        f"gini={st['gini_absJ']:.3f} "
        f"corr_wd={fmt_float(corr,3)} "
        f"slope_wd={fmt_float(slope,6)}"
    )


def print_summary_table(rows: List[Dict]) -> None:
    print("\nSUMMARY (sorted by corr_wd asc; more negative = more local)")
    rows2 = sorted(rows, key=lambda r: (1e9 if r["corr_wd"] is None else r["corr_wd"]))
    print(" seed | edges | degμ  | gini  | corr_wd  | slope_wd")
    print("------+-------+-------+-------+----------+-----------")
    for r in rows2:
        print(
            f"{r['seed']:5d} |"
            f"{r['edges']:6d} |"
            f"{r['deg_mean']:6.2f} |"
            f"{r['gini']:.3f} |"
            f"{fmt_float(r['corr_wd'],3):>9} |"
            f"{fmt_float(r['slope_wd'],6):>10}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=str, default="learn", choices=["learn", "random", "frozen"])

    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--Delta", type=float, default=0.0)

    ap.add_argument("--pairs-per-step", type=int, default=40)
    ap.add_argument("--eta", type=float, default=0.20)
    ap.add_argument("--decay", type=float, default=0.01)

    ap.add_argument("--J-init-scale", type=float, default=1.0)
    ap.add_argument("--J-clip", type=float, default=2.5)

    ap.add_argument("--influence-eps", type=float, default=0.08)

    ap.add_argument("--budget", type=float, default=3.5)
    ap.add_argument("--budget-iters", type=int, default=6)

    ap.add_argument("--wd-eps", type=float, default=1e-6)
    ap.add_argument("--wd-bins", type=int, default=6)

    ap.add_argument("--probe-mode", type=str, default="directJ", choices=["directJ", "masked_uniform"])
    ap.add_argument("--k-steps", type=int, default=1)
    ap.add_argument("--mask-thr", type=float, default=0.05)
    ap.add_argument("--uniform-strength", type=float, default=1.0)

    ap.add_argument("--history-every", type=int, default=20,
                    help="Save a history snapshot every K steps (0 disables history).")

    ap.add_argument("--save-J", action="store_true", help="Save full J_final in JSON (large).")

    ap.add_argument("--seed", type=int, default=0, help="Single-seed mode (ignored if --seeds>1).")
    ap.add_argument("--seed-start", type=int, default=0, help="Seed sweep start.")
    ap.add_argument("--seeds", type=int, default=1, help="Number of seeds (1 means single run).")

    ap.add_argument("--out", type=str, default="results_link_budget.json",
                    help="Base output name (per-seed + summary files will be generated for sweeps).")
    ap.add_argument("--progress", action="store_true",
                    help="Print per-seed lines and final summary table.")
    args = ap.parse_args()

    n_seeds = int(args.seeds)
    if n_seeds < 1:
        n_seeds = 1

    # Build shared params template
    def make_params(seed: int) -> Params:
        return Params(
            N=int(args.N),
            steps=int(args.steps),
            dt=float(args.dt),
            Delta=float(args.Delta),
            pairs_per_step=int(args.pairs_per_step),
            eta=float(args.eta),
            decay=float(args.decay),
            J_init_scale=float(args.J_init_scale),
            J_clip=float(args.J_clip),
            influence_eps=float(args.influence_eps),
            budget=float(args.budget),
            budget_iters=int(args.budget_iters),
            control=str(args.control),
            seed=int(seed),
            wd_eps=float(args.wd_eps),
            wd_bins=int(args.wd_bins),
            probe_mode=str(args.probe_mode),
            k_steps=int(args.k_steps),
            mask_thr=float(args.mask_thr),
            uniform_strength=float(args.uniform_strength),
            history_every=int(args.history_every),
            save_J=bool(args.save_J),
        )

    if n_seeds == 1:
        params = make_params(int(args.seed))
        res = run_one(params)
        write_json(args.out, res)
        if args.progress:
            print_seed_line(res)
            print("Wrote:", args.out)
        return 0

    # Sweep
    seed0 = int(args.seed_start)
    rows = []
    out_files = []

    for s in range(seed0, seed0 + n_seeds):
        params = make_params(s)
        res = run_one(params)

        out_path = seed_out_path(args.out, s)
        write_json(out_path, res)
        out_files.append(out_path)

        fin = res["final"]
        st = fin["structure"]
        wd = fin["wdist_cmp"]
        rows.append({
            "seed": s,
            "edges": int(st["edges_ge_thr"]),
            "deg_mean": float(st["deg_mean"]),
            "gini": float(st["gini_absJ"]),
            "corr_wd": wd.get("corr_wdist_influence", None),
            "slope_wd": wd.get("slope_influence_vs_wdist", None),
        })

        if args.progress:
            print_seed_line(res)

    summary = {
        "meta": {
            "control": str(args.control),
            "N": int(args.N),
            "steps": int(args.steps),
            "dt": float(args.dt),
            "Delta": float(args.Delta),
            "pairs_per_step": int(args.pairs_per_step),
            "eta": float(args.eta),
            "decay": float(args.decay),
            "budget": float(args.budget),
            "budget_iters": int(args.budget_iters),
            "probe_mode": str(args.probe_mode),
            "k_steps": int(args.k_steps),
            "mask_thr": float(args.mask_thr),
            "uniform_strength": float(args.uniform_strength),
            "wd_eps": float(args.wd_eps),
            "wd_bins": int(args.wd_bins),
            "history_every": int(args.history_every),
            "save_J": bool(args.save_J),
            "seed_start": seed0,
            "seeds": n_seeds,
        },
        "rows": rows,
        "files": out_files,
    }

    sum_path = summary_out_path(args.out)
    write_json(sum_path, summary)

    if args.progress:
        print_summary_table(rows)
        print("\nWrote summary:", sum_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
