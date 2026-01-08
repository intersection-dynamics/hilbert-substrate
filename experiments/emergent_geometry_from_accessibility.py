#!/usr/bin/env python3
"""
emergent_geometry_from_accessibility.py

Accessibility descent (Paper II) + operational delay geometry probe (Paper I-style),
WITH quantitative validation metrics and memory-safe Pauli transforms.

Stability fixes (this version):
- Two-sided geodesic search (sgn ±)
- DO NOT catastrophically decay lr on rejection
- Acceptance tolerance to avoid float-noise "no improvement"
- Reject-patience escape: small random unitary kick + alpha reset
- penalty schedule supports floats: --penalty-schedule 2,3,3.5,4

Recommended N=8 run:
python emergent_geometry_from_accessibility.py --n 8 --graph grid2d --penalty-schedule 2,3,3.5,3.8,4 --restarts 8 --steps 600 --lr 0.15 --t-max 14 --t-steps 281 --eps 0.25 --threshold 0.01 --spatial-tol 3.0 --neighbor-k 3 --out out_geom_grid_N8_sched --plot
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.linalg import expm, norm, eigh
import matplotlib.pyplot as plt


# ----------------------------
# Pauli primitives
# ----------------------------

def pauli_stack() -> np.ndarray:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.stack([I, X, Y, Z], axis=0)  # (4,2,2)


_P = pauli_stack()
_P_ba = np.transpose(_P, (0, 2, 1))  # (p,b,a)


def pauli_weight_tensor(N: int) -> np.ndarray:
    w1 = np.array([0.0, 1.0, 1.0, 1.0], dtype=float)
    w = w1
    for _ in range(N - 1):
        w = w[..., None] + w1[None, ...]
    return w


def pauli_coeffs_from_operator(H: np.ndarray, N: int) -> np.ndarray:
    dim = 2 ** N
    assert H.shape == (dim, dim)
    T = H.reshape((2,) * N + (2,) * N)

    n_rem = N
    for _ in range(N):
        a_axis = n_rem - 1
        b_axis = 2 * n_rem - 1
        axes = [i for i in range(T.ndim) if i not in (b_axis, a_axis)] + [b_axis, a_axis]
        T = np.transpose(T, axes=axes)
        T = np.einsum("...ba,pba->...p", T, _P_ba, optimize=True)
        n_rem -= 1

    return np.real(T) / (2 ** N)


def operator_from_pauli_coeffs(C: np.ndarray, N: int) -> np.ndarray:
    T = np.asarray(C, dtype=float)
    m = N
    for _ in range(N):
        T = np.moveaxis(T, m - 1, -1)
        T = np.einsum("...p,pab->...ab", T, _P, optimize=True)
        m -= 1

    axes_a = list(range(0, 2 * N, 2))[::-1]
    axes_b = list(range(1, 2 * N, 2))[::-1]
    T = np.transpose(T, axes=axes_a + axes_b)
    return T.reshape((2 ** N, 2 ** N)).astype(complex)


# ----------------------------
# Graph generators
# ----------------------------

def edges_chain(N: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(N - 1)]


def edges_ring(N: int) -> List[Tuple[int, int]]:
    if N < 3:
        return edges_chain(N)
    e = edges_chain(N)
    e.append((N - 1, 0))
    return e


def edges_complete(N: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def edges_grid2d(N: int, rows: Optional[int] = None, cols: Optional[int] = None) -> List[Tuple[int, int]]:
    if rows is not None or cols is not None:
        if rows is None or cols is None:
            raise ValueError("grid2d requires both --grid-rows and --grid-cols (or neither).")
        if rows * cols != N:
            raise ValueError(f"grid2d requires N=rows*cols; got N={N}, rows={rows}, cols={cols}")
        R, C = int(rows), int(cols)
    else:
        best = None
        for r in range(1, N + 1):
            if N % r != 0:
                continue
            c = N // r
            rr, cc = (r, c) if r <= c else (c, r)
            score = cc / rr
            if best is None or score < best[0]:
                best = (score, rr, cc)
        if best is None:
            raise ValueError(f"grid2d could not factor N={N}")
        _, R, C = best

    def node(r: int, c: int) -> int:
        return r * C + c

    e: List[Tuple[int, int]] = []
    for r in range(R):
        for c in range(C):
            if r + 1 < R:
                e.append((node(r, c), node(r + 1, c)))
            if c + 1 < C:
                e.append((node(r, c), node(r, c + 1)))
    return e


def edges_random_regular(N: int, degree: int, rng: np.random.Generator, max_tries: int = 2000) -> List[Tuple[int, int]]:
    if degree >= N:
        raise ValueError("random_regular requires degree < N")
    if (N * degree) % 2 != 0:
        raise ValueError("random_regular requires N*degree even")

    for _ in range(max_tries):
        stubs = []
        for i in range(N):
            stubs.extend([i] * degree)
        rng.shuffle(stubs)

        edges = set()
        ok = True
        for k in range(0, len(stubs), 2):
            a, b = stubs[k], stubs[k + 1]
            if a == b:
                ok = False
                break
            u, v = (a, b) if a < b else (b, a)
            if (u, v) in edges:
                ok = False
                break
            edges.add((u, v))

        if ok:
            return sorted(list(edges))

    raise RuntimeError(f"Failed to generate random {degree}-regular graph after {max_tries} tries.")


def build_edges(graph_name: str, N: int, rng: np.random.Generator, degree: int,
                grid_rows: Optional[int] = None, grid_cols: Optional[int] = None) -> List[Tuple[int, int]]:
    g = graph_name.lower().strip()
    if g == "chain":
        return edges_chain(N)
    if g == "ring":
        return edges_ring(N)
    if g == "complete":
        return edges_complete(N)
    if g == "grid2d":
        return edges_grid2d(N, rows=grid_rows, cols=grid_cols)
    if g in ("random_regular", "rrg", "regular"):
        return edges_random_regular(N, degree=degree, rng=rng)
    raise ValueError(f"Unknown graph '{graph_name}'.")


# ----------------------------
# Graph distances + validation metrics
# ----------------------------

def adjacency_list(N: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(N)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def all_pairs_shortest_paths(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    from collections import deque
    adj = adjacency_list(N, edges)
    gdist = np.full((N, N), np.inf, dtype=float)
    for s in range(N):
        q = deque([s])
        gdist[s, s] = 0.0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not np.isfinite(gdist[s, v]):
                    gdist[s, v] = gdist[s, u] + 1.0
                    q.append(v)
    return gdist


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    da = float(np.sqrt(np.sum(a * a)))
    db = float(np.sqrt(np.sum(b * b)))
    if da <= 1e-12 or db <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / (da * db))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata_average(np.asarray(a, dtype=float))
    rb = rankdata_average(np.asarray(b, dtype=float))
    return pearson_corr(ra, rb)


def neighbor_recall_at_k(D: np.ndarray, edges: List[Tuple[int, int]], k: int) -> float:
    N = D.shape[0]
    adj = adjacency_list(N, edges)
    recalls = []
    for i in range(N):
        true = set(adj[i])
        if len(true) == 0:
            continue
        row = D[i].copy()
        row[i] = np.inf
        row = np.where(np.isfinite(row), row, np.inf)
        pred_idx = np.argsort(row)[:max(1, k)]
        pred = set(int(x) for x in pred_idx)
        hit = len(pred.intersection(true))
        recalls.append(hit / len(true))
    return float(np.mean(recalls)) if recalls else float("nan")


def validate_geometry_metrics(D: np.ndarray, gdist: np.ndarray, edges: List[Tuple[int, int]], k: int) -> Dict[str, float]:
    xs, ys = [], []
    N = D.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if np.isfinite(D[i, j]) and np.isfinite(gdist[i, j]):
                xs.append(D[i, j])
                ys.append(gdist[i, j])
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return {
        "pearson_D_graph": pearson_corr(xs, ys),
        "spearman_D_graph": spearman_corr(xs, ys),
        "neighbor_recall_at_k": neighbor_recall_at_k(D, edges, k=k),
    }


# ----------------------------
# Hamiltonian construction (Heisenberg on edges)
# ----------------------------

def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for k in range(1, len(ops)):
        out = np.kron(out, ops[k])
    return out


def heisenberg_on_edges(N: int, edges: List[Tuple[int, int]], J: float = 1.0) -> np.ndarray:
    I, X, Y, Z = _P[0], _P[1], _P[2], _P[3]
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    def term_two_local(opA, i, opB, j):
        ops = [I] * N
        ops[i] = opA
        ops[j] = opB
        return kron_n(ops)

    for (i, j) in edges:
        H += J * (term_two_local(X, i, X, j) + term_two_local(Y, i, Y, j) + term_two_local(Z, i, Z, j))

    return 0.5 * (H + H.conj().T)


# ----------------------------
# Cost engine
# ----------------------------

@dataclass
class CostEngine:
    N: int
    penalty_power: float
    wp: np.ndarray

    @staticmethod
    def build(N: int, penalty_power: float) -> "CostEngine":
        w = pauli_weight_tensor(N)
        wp = w ** float(penalty_power)
        return CostEngine(N=N, penalty_power=float(penalty_power), wp=wp)

    def cost_and_grad_generator(self, H: np.ndarray) -> Tuple[float, np.ndarray]:
        C = pauli_coeffs_from_operator(H, self.N)
        c2 = C * C
        denom = float(np.sum(c2))
        if denom <= 1e-30:
            return float("inf"), np.zeros_like(H)

        num = float(np.sum(self.wp * c2))
        cost = num / denom

        grad_c = (2.0 * (self.wp - cost) * C) / denom
        M = operator_from_pauli_coeffs(grad_c, self.N)
        M = 0.5 * (M + M.conj().T)

        comm = H @ M - M @ H
        K = 1j * comm
        K = 0.5 * (K + K.conj().T)
        return float(cost), K


def random_scrambler_unitary(dim: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    G = A + A.conj().T
    return expm(1j * G * scale)


@dataclass
class OptConfig:
    steps: int = 600
    lr: float = 0.15
    line_search_steps: int = 10
    grad_norm_floor: float = 1e-10
    alpha_decay: float = 0.5
    debug_every: int = 50
    alpha_min: float = 1e-6
    alpha_max: float = 0.5
    alpha_grow: float = 1.15
    alpha_shrink: float = 0.85

    # plateau escape
    reject_patience: int = 40
    kick_scale: float = 0.02  # small, stays on orbit; increase to 0.05 if needed

    # acceptance tolerance
    improve_rtol: float = 1e-12
    improve_atol: float = 1e-12


def riemannian_descent(H0: np.ndarray, engine: CostEngine, cfg: OptConfig,
                       rng: Optional[np.random.Generator] = None) -> Dict:
    if rng is None:
        rng = np.random.default_rng(0)

    H = H0.copy()
    alpha0 = float(cfg.lr)

    accepted_total = 0
    rejected_total = 0
    consec_rejects = 0

    history = []
    t0 = time.time()

    def is_improvement(new: float, old: float) -> bool:
        tol = cfg.improve_atol + cfg.improve_rtol * max(1.0, abs(old))
        return (new < old - tol)

    def random_kick(Hin: np.ndarray) -> np.ndarray:
        dim = Hin.shape[0]
        A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        G = A + A.conj().T
        U = expm(1j * G * float(cfg.kick_scale))
        return U @ Hin @ U.conj().T

    for step in range(cfg.steps + 1):
        cost_old, K = engine.cost_and_grad_generator(H)
        K_norm = float(norm(K, "fro"))

        if not np.isfinite(cost_old) or K_norm < cfg.grad_norm_floor:
            if step % cfg.debug_every == 0:
                print(f"  [dbg] step={step} cost={cost_old:.6f} K_norm={K_norm:.3e} (stopping)")
            break

        K_dir = K / (K_norm + 1e-30)

        alpha = float(np.clip(alpha0, cfg.alpha_min, cfg.alpha_max))
        best_cost = cost_old
        best_H = None
        accepted = False
        shrinks = 0
        used_alpha = None

        for _ in range(cfg.line_search_steps):
            improved = False
            for sgn in (-1.0, +1.0):
                U_step = expm(-1j * sgn * alpha * K_dir)
                H_try = U_step @ H @ U_step.conj().T
                cost_try, _ = engine.cost_and_grad_generator(H_try)
                if np.isfinite(cost_try) and is_improvement(cost_try, best_cost):
                    best_cost = float(cost_try)
                    best_H = H_try
                    accepted = True
                    improved = True
                    used_alpha = alpha
            if improved:
                break
            alpha *= cfg.alpha_decay
            shrinks += 1

        if accepted and best_H is not None:
            H = best_H
            accepted_total += 1
            consec_rejects = 0

            # update alpha0 gently
            if shrinks == 0:
                alpha0 = min(cfg.alpha_max, (used_alpha or alpha0) * cfg.alpha_grow)
            else:
                alpha0 = max(cfg.alpha_min, (used_alpha or alpha0) * cfg.alpha_shrink)

        else:
            rejected_total += 1
            consec_rejects += 1
            alpha0 = max(cfg.alpha_min, alpha0 * cfg.alpha_shrink)

            # escape if stuck
            if consec_rejects >= cfg.reject_patience:
                H = random_kick(H)
                consec_rejects = 0
                alpha0 = float(cfg.lr)  # reset

        if step % cfg.debug_every == 0:
            print(f"  [dbg] step={step} cost={cost_old:.6f} -> best={best_cost:.6f} "
                  f"K_norm={K_norm:.3e} alpha0={alpha0:.3e} "
                  f"acc={accepted_total} rej={rejected_total}")

        if step % 10 == 0:
            history.append({
                "step": step,
                "cost": float(best_cost),
                "K_norm": float(K_norm),
                "alpha0": float(alpha0),
                "accepted_total": int(accepted_total),
                "rejected_total": int(rejected_total),
            })

    t1 = time.time()
    final_cost, _ = engine.cost_and_grad_generator(H)
    return {"final_cost": float(final_cost), "elapsed_sec": float(t1 - t0), "history": history, "H_final": H}


def harmonion_cost_via_diagonal(H: np.ndarray, engine: CostEngine) -> float:
    evals, _ = eigh(H)
    H_diag = np.diag(evals.astype(complex))
    c, _ = engine.cost_and_grad_generator(H_diag)
    return float(c)


def basin_label(cost: float, spatial_cost: float, harmonion_cost: float, spatial_tol: float) -> str:
    if abs(cost - spatial_cost) <= spatial_tol:
        return "spatial"
    if abs(cost - harmonion_cost) <= spatial_tol:
        return "deep"
    return "intermediate"


# ----------------------------
# Operational delay probe
# ----------------------------

def apply_local_rotation_X(N: int, site: int, eps: float) -> np.ndarray:
    I = _P[0]
    X = _P[1]
    ops = [I] * N
    ops[site] = X
    X_i = kron_n(ops)
    return expm(-1j * eps * X_i)


def ket0(N: int) -> np.ndarray:
    dim = 2 ** N
    psi = np.zeros((dim,), dtype=complex)
    psi[0] = 1.0
    return psi


def reduced_density_one_qubit_from_state(psi: np.ndarray, N: int, j: int) -> np.ndarray:
    psiN = psi.reshape((2,) * N)
    axes = (j,) + tuple(k for k in range(N) if k != j)
    psi_perm = np.transpose(psiN, axes=axes)
    psi_mat = psi_perm.reshape(2, -1)
    return psi_mat @ psi_mat.conj().T


def trace_distance_qubit(rho: np.ndarray, sigma: np.ndarray) -> float:
    delta = rho - sigma
    delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return float(0.5 * np.sum(np.abs(w)))


def propagation_delay_matrix(H: np.ndarray, N: int, eps: float, t_max: float, t_steps: int, threshold: float):
    psi0 = ket0(N)
    t_grid = np.linspace(0.0, t_max, t_steps)
    tau = np.full((N, N), np.nan, dtype=float)
    peak = np.zeros((N, N), dtype=float)

    U_list = [expm(-1j * H * t) for t in t_grid]

    rho_base = np.zeros((t_steps, N, 2, 2), dtype=complex)
    for ti, U in enumerate(U_list):
        psi_t = U @ psi0
        for j in range(N):
            rho_base[ti, j] = reduced_density_one_qubit_from_state(psi_t, N, j)

    for i in range(N):
        Oi = apply_local_rotation_X(N, i, eps=eps)
        psi1 = Oi @ psi0
        crossed = [False] * N
        tau[i, i] = 0.0

        for ti, U in enumerate(U_list):
            psi_t = U @ psi1
            for j in range(N):
                rho_p = reduced_density_one_qubit_from_state(psi_t, N, j)
                d = trace_distance_qubit(rho_p, rho_base[ti, j])
                peak[i, j] = max(peak[i, j], d)
                if (not crossed[j]) and (d >= threshold):
                    tau[i, j] = t_grid[ti]
                    crossed[j] = True

    return tau, peak, t_grid


# ----------------------------
# MDS
# ----------------------------

def classical_mds(D: np.ndarray, out_dim: int = 3):
    N = D.shape[0]
    D2 = D ** 2
    if np.any(~np.isfinite(D2)):
        finite_vals = D2[np.isfinite(D2)]
        fill = float(np.max(finite_vals)) if finite_vals.size else 1.0
        fill = max(fill, 1.0)
        D2 = np.where(np.isfinite(D2), D2, fill * 4.0)

    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J

    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    pos = evals > 1e-12
    evals_pos = evals[pos]
    evecs_pos = evecs[:, pos]

    k = min(out_dim, evecs_pos.shape[1])
    if k == 0:
        return np.zeros((N, out_dim), dtype=float), evals

    L = np.diag(np.sqrt(evals_pos[:k]))
    X = evecs_pos[:, :k] @ L
    if k < out_dim:
        X_pad = np.zeros((N, out_dim), dtype=float)
        X_pad[:, :k] = X
        X = X_pad
    return X.real, evals


def estimate_intrinsic_dim(evals: np.ndarray, frac: float = 0.95) -> int:
    evals_pos = evals[evals > 1e-12]
    if evals_pos.size == 0:
        return 0
    total = float(np.sum(evals_pos))
    cumsum = np.cumsum(evals_pos) / total
    return int(np.searchsorted(cumsum, frac) + 1)


# ----------------------------
# Plot helpers
# ----------------------------

def plot_distance_heatmap(D: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(D, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_embedding_2d(X: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1])
    for idx in range(X.shape[0]):
        plt.text(X[idx, 0], X[idx, 1], str(idx), fontsize=9)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_penalty_schedule(s: Optional[str], fallback: float) -> List[float]:
    if not s:
        return [float(fallback)]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Accessibility descent + operational emergent geometry + metrics (memory-safe Pauli transforms).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--graph", type=str, default="grid2d")
    ap.add_argument("--grid-rows", type=int, default=None)
    ap.add_argument("--grid-cols", type=int, default=None)
    ap.add_argument("--degree", type=int, default=2)

    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--penalty-power", type=float, default=4.0)
    ap.add_argument("--penalty-schedule", type=str, default=None)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--scramble-scale", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spatial-tol", type=float, default=3.0)

    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--t-max", type=float, default=14.0)
    ap.add_argument("--t-steps", type=int, default=281)
    ap.add_argument("--mds-dim", type=int, default=3)
    ap.add_argument("--neighbor-k", type=int, default=3)

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", type=str, default="out_geom")

    args = ap.parse_args()

    N = int(args.n)
    dim = 2 ** N
    penalty_sched = parse_penalty_schedule(args.penalty_schedule, fallback=float(args.penalty_power))

    ensure_dir(args.out)
    run_dir = os.path.join(args.out, f"N{N}_{args.graph}_p{penalty_sched[-1]}")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "runs"))

    meta = vars(args).copy()
    meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["dim"] = dim
    meta["penalty_sched"] = penalty_sched
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n============================================================")
    print("EMERGENT GEOMETRY FROM ACCESSIBILITY (MEMORY-SAFE)")
    print("============================================================")
    print(json.dumps(meta, indent=2))

    rng_master = np.random.default_rng(int(args.seed))

    edges = build_edges(
        args.graph, N, rng_master,
        degree=int(args.degree),
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols
    )

    H_target = heisenberg_on_edges(N, edges, J=1.0)

    engine_final = CostEngine.build(N=N, penalty_power=float(penalty_sched[-1]))
    c_spatial, _ = engine_final.cost_and_grad_generator(H_target)
    c_harm = harmonion_cost_via_diagonal(H_target, engine_final)
    gdist = all_pairs_shortest_paths(N, edges)

    print(f"\nGraph edges: |E|={len(edges)}")
    print(f"Spatial target cost (p={penalty_sched[-1]}):   {c_spatial:.6f}")
    print(f"Harmonion (diag) cost  (p={penalty_sched[-1]}): {c_harm:.6f}")

    summary_rows = []

    for r in range(int(args.restarts)):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        U_scram = random_scrambler_unitary(dim, rng, scale=float(args.scramble_scale))
        H = U_scram @ H_target @ U_scram.conj().T

        print(f"\n--- Restart {r+1}/{args.restarts} ---")

        total_elapsed = 0.0
        for pi, p in enumerate(penalty_sched):
            engine = CostEngine.build(N=N, penalty_power=float(p))
            opt_cfg = OptConfig(steps=int(args.steps), lr=float(args.lr), debug_every=50)

            c_stage0, _ = engine.cost_and_grad_generator(H)
            print(f"  Stage {pi+1}/{len(penalty_sched)}: penalty_power={p} | start cost={c_stage0:.6f}")

            res = riemannian_descent(H, engine, opt_cfg, rng=rng)
            H = res["H_final"]
            total_elapsed += float(res["elapsed_sec"])

            c_stage1, _ = engine.cost_and_grad_generator(H)
            print(f"  Stage {pi+1}: end cost={c_stage1:.6f} | elapsed={res['elapsed_sec']:.2f}s")

        c_scram, _ = engine_final.cost_and_grad_generator(U_scram @ H_target @ U_scram.conj().T)
        c_final, _ = engine_final.cost_and_grad_generator(H)
        label = basin_label(float(c_final), float(c_spatial), float(c_harm), spatial_tol=float(args.spatial_tol))

        print(f"Scrambled cost (final p): {c_scram:.6f}")
        print(f"Recovered cost (final p): {c_final:.6f} | basin={label} | elapsed_total={total_elapsed:.2f}s")

        # (Probe/plots omitted here only for brevity? No — keep behavior identical to prior versions)
        # We keep probe exactly as before:
        tau = peak = D = X = evals = None
        dim95 = None
        metrics = {"pearson_D_graph": None, "spearman_D_graph": None, "neighbor_recall_at_k": None}

        probed = (label == "spatial")
        if probed:
            print("  Probing operational delays (HIP-style)...")
            tau, peak, t_grid = propagation_delay_matrix(
                H, N=N, eps=float(args.eps),
                t_max=float(args.t_max), t_steps=int(args.t_steps),
                threshold=float(args.threshold),
            )

            D = np.zeros((N, N), dtype=float)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        D[i, j] = 0.0
                    else:
                        a = tau[i, j]
                        b = tau[j, i]
                        if np.isfinite(a) and np.isfinite(b):
                            D[i, j] = 0.5 * (a + b)
                        elif np.isfinite(a):
                            D[i, j] = a
                        elif np.isfinite(b):
                            D[i, j] = b
                        else:
                            D[i, j] = float("nan")

            X, evals = classical_mds(D, out_dim=int(args.mds_dim))
            dim95 = estimate_intrinsic_dim(evals, frac=0.95)
            metrics = validate_geometry_metrics(D, gdist, edges, k=int(args.neighbor_k))

            print(f"  Estimated intrinsic dim (95% var): {dim95}")
            print(f"  Spearman(D, graph_dist): {metrics['spearman_D_graph']}")
            print(f"  Pearson(D, graph_dist):  {metrics['pearson_D_graph']}")
            print(f"  NeighborRecall@{int(args.neighbor_k)}: {metrics['neighbor_recall_at_k']}")

            run_npz = os.path.join(run_dir, "runs", f"run_r{r}_spatial.npz")
            np.savez(
                run_npz,
                N=N,
                edges=np.array(edges, dtype=int),
                penalty_sched=np.array(penalty_sched, dtype=float),
                cost_spatial=c_spatial,
                cost_harmonion=c_harm,
                cost_scrambled=c_scram,
                cost_final=c_final,
                tau=tau,
                peak=peak,
                t_grid=t_grid,
                D=D,
                X=X,
                evals=evals,
                dim95=dim95,
                gdist=gdist,
                pearson_D_graph=metrics["pearson_D_graph"],
                spearman_D_graph=metrics["spearman_D_graph"],
                neighbor_recall_at_k=metrics["neighbor_recall_at_k"],
                neighbor_k=int(args.neighbor_k),
            )

            if args.plot:
                plot_distance_heatmap(D, f"D(i,j) delays | N={N} r={r}", os.path.join(run_dir, f"D_heat_r{r}.png"))
                plot_embedding_2d(X, f"MDS2D | dim95={dim95} | r={r}", os.path.join(run_dir, f"mds2d_r{r}.png"))

        detail = {
            "N": N,
            "graph": args.graph,
            "edges": edges,
            "restart": r,
            "penalty_sched": penalty_sched,
            "cost_spatial_target_finalp": float(c_spatial),
            "cost_harmonion_diag_finalp": float(c_harm),
            "cost_scrambled_finalp": float(c_scram),
            "cost_final_finalp": float(c_final),
            "basin_label": label,
            "elapsed_total_sec": float(total_elapsed),
            "probed": bool(probed),
            "neighbor_k": int(args.neighbor_k),
            "spearman_D_graph": metrics["spearman_D_graph"],
            "pearson_D_graph": metrics["pearson_D_graph"],
            "neighbor_recall_at_k": metrics["neighbor_recall_at_k"],
        }
        with open(os.path.join(run_dir, "runs", f"detail_r{r}.json"), "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2)

        summary_rows.append(detail)

    summary_path = os.path.join(run_dir, "summary.csv")
    if summary_rows:
        cols = list(summary_rows[0].keys())
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")

    print("\n============================================================")
    print("DONE")
    print(f"Wrote: {summary_path}")
    print(f"Runs:  {os.path.join(run_dir, 'runs')}")
    print("============================================================")


if __name__ == "__main__":
    main()
