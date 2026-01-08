#!/usr/bin/env python3
# emergent_spacetime_from_hip_cupy.py
#
# GPU-accelerated emergent spacetime diagnostics from HIP-style influence.
#
# Key improvements vs naive density-matrix approach:
# - Uses pure statevectors |psi(t)> and computes single-qubit reduced density rho_j(t) (2x2) directly:
#     reshape -> (2, 2^(n-1)) then rho = M M^\dagger
# - Evolves statevectors via Hamiltonian eigendecomposition:
#     psi(t) = V [exp(-i λ t) ⊙ (V† psi(0)) ]
# - Avoids building U(t) explicitly.
#
# Supports one-shot sweeps:
#   --sweep-topk "2,3,4"
#   --sweep-delta "0.03,0.05,0.08"   (or "0.02:0.14:7")
#
# Outputs:
#   <out>/sweep_results.csv
#   <out>/sweep_results.json
#   <out>/best/  (plots + npz + best_summary.json)
#
# Windows example (one command):
#   python emergent_spacetime_from_hip_cupy.py --n 8 --graph ring --samples 128 --t-max 8 --t-steps 81 --window-frac 0.5 --sweep-topk "2,3,4" --sweep-delta "0.03,0.05,0.08,0.12" --tau-mode rel --eta 0.5 --seed 0 --out outputs\spacetime_sweep_gpu
#
import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Backend selection: CuPy if available, else NumPy
# -----------------------------

def get_xp(prefer_cupy: bool = True):
    if not prefer_cupy:
        return np, False
    try:
        import cupy as cp
        # quick sanity: can we allocate on GPU?
        _ = cp.zeros((1,), dtype=cp.float32)
        return cp, True
    except Exception:
        return np, False


# -----------------------------
# Pauli / small gates (backend-agnostic)
# -----------------------------

def paulis(xp):
    I = xp.array([[1, 0], [0, 1]], dtype=xp.complex128)
    X = xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
    Y = xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128)
    Z = xp.array([[1, 0], [0, -1]], dtype=xp.complex128)
    H = (1.0 / math.sqrt(2.0)) * xp.array([[1, 1], [1, -1]], dtype=xp.complex128)
    return {"I": I, "X": X, "Y": Y, "Z": Z, "H": H}


def kron_n(xp, ops: List):
    out = ops[0]
    for a in ops[1:]:
        out = xp.kron(out, a)
    return out


def embed_one_qubit_op_dense(xp, n: int, i: int, op2: "xp.ndarray") -> "xp.ndarray":
    P = paulis(xp)
    ops = []
    for k in range(n):
        ops.append(op2 if k == i else P["I"])
    return kron_n(xp, ops)


def embed_two_qubit_op_dense(xp, n: int, i: int, j: int, op_i: "xp.ndarray", op_j: "xp.ndarray") -> "xp.ndarray":
    P = paulis(xp)
    ops = []
    for k in range(n):
        if k == i:
            ops.append(op_i)
        elif k == j:
            ops.append(op_j)
        else:
            ops.append(P["I"])
    return kron_n(xp, ops)


# -----------------------------
# Graph helpers
# -----------------------------

def edges_ring(n: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def edges_grid_2d(nx: int, ny: int) -> Tuple[int, List[Tuple[int, int]]]:
    n = nx * ny
    E = []
    def idx(x, y): return y * nx + x
    for y in range(ny):
        for x in range(nx):
            u = idx(x, y)
            if x + 1 < nx:
                E.append((u, idx(x + 1, y)))
            if y + 1 < ny:
                E.append((u, idx(x, y + 1)))
    return n, E


def edges_erdos_renyi(n: int, p: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    E = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                E.append((i, j))
    if len(E) == 0:
        E = edges_ring(n)
    return E


# -----------------------------
# Random product states (GPU-friendly)
# -----------------------------

def random_single_qubit_state(xp, rng, use_gpu: bool) -> "xp.ndarray":
    # Haar-ish via normalized complex Gaussian
    if use_gpu:
        # CuPy random: use xp.random
        v = xp.random.normal(size=(2,)) + 1j * xp.random.normal(size=(2,))
    else:
        v = rng.normal(size=(2,)) + 1j * rng.normal(size=(2,))
        v = xp.asarray(v)
    v = v / xp.linalg.norm(v)
    return v.astype(xp.complex128)


def random_product_state(xp, n: int, rng, use_gpu: bool) -> "xp.ndarray":
    psi = random_single_qubit_state(xp, rng, use_gpu)
    for _ in range(n - 1):
        psi = xp.kron(psi, random_single_qubit_state(xp, rng, use_gpu))
    psi = psi / xp.linalg.norm(psi)
    return psi.astype(xp.complex128)


# -----------------------------
# Apply 1-qubit gate to statevector without building full operator
# -----------------------------

def apply_one_qubit_gate_to_state(xp, psi: "xp.ndarray", gate2: "xp.ndarray", qubit: int, n: int) -> "xp.ndarray":
    """
    Apply gate2 (2x2) to qubit 'qubit' of an n-qubit statevector psi (shape [2^n]).
    Implemented via reshape/transposes (GPU-friendly).
    """
    # reshape to tensor
    tens = psi.reshape([2] * n)
    # bring target qubit to axis 0
    axes = [qubit] + [k for k in range(n) if k != qubit]
    inv_axes = [0] * n
    for new_pos, old_axis in enumerate(axes):
        inv_axes[old_axis] = new_pos
    tens = xp.transpose(tens, axes)
    mat = tens.reshape(2, -1)  # (2, 2^(n-1))
    mat2 = gate2 @ mat
    tens2 = mat2.reshape([2] + [2] * (n - 1))
    tens2 = xp.transpose(tens2, inv_axes)
    return tens2.reshape(-1)


# -----------------------------
# Reduced single-qubit density from statevector
# -----------------------------

def reduced_rho_qubit_from_state(xp, psi: "xp.ndarray", qubit: int, n: int) -> "xp.ndarray":
    """
    rho_j = Tr_{~j} |psi><psi|, returns 2x2 complex matrix.
    """
    tens = psi.reshape([2] * n)
    axes = [qubit] + [k for k in range(n) if k != qubit]
    tens = xp.transpose(tens, axes)
    mat = tens.reshape(2, -1)
    rho = mat @ mat.conj().T
    # numerical hermitize
    rho = 0.5 * (rho + rho.conj().T)
    return rho


# -----------------------------
# Fast trace distance between 2x2 density matrices (Hermitian)
# -----------------------------

def trace_distance_2x2(xp, rho: "xp.ndarray", sigma: "xp.ndarray") -> "xp.ndarray":
    """
    D = 0.5 ||rho-sigma||_1 for 2x2 Hermitian.
    Eigenvalues analytic: (tr/2) ± sqrt(((a-d)/2)^2 + |b|^2)
    """
    A = rho - sigma
    A = 0.5 * (A + A.conj().T)
    a = A[0, 0]
    d = A[1, 1]
    b = A[0, 1]
    tr = a + d
    half = 0.5
    disc = ((a - d) * half) ** 2 + (b * b.conj())
    # disc should be real nonnegative (tiny imag from numerics)
    disc = xp.real(disc)
    s = xp.sqrt(xp.maximum(disc, 0.0))
    e1 = tr * half + s
    e2 = tr * half - s
    tn = xp.abs(e1) + xp.abs(e2)
    return 0.5 * tn


# -----------------------------
# Hamiltonian construction (dense)
# -----------------------------

def build_hamiltonian_dense(xp, n: int, E: List[Tuple[int, int]], rng: np.random.Generator,
                            J: float = 1.0, h: float = 0.3) -> "xp.ndarray":
    P = paulis(xp)
    dim = 2 ** n
    H = xp.zeros((dim, dim), dtype=xp.complex128)

    # Local random fields (CPU RNG for reproducibility; then transfer)
    for i in range(n):
        hx = h * float(rng.normal())
        hz = h * float(rng.normal())
        H = H + hx * embed_one_qubit_op_dense(xp, n, i, P["X"])
        H = H + hz * embed_one_qubit_op_dense(xp, n, i, P["Z"])

    # Edge Heisenberg-like couplings
    for (i, j) in E:
        H = H + J * embed_two_qubit_op_dense(xp, n, i, j, P["X"], P["X"])
        H = H + J * embed_two_qubit_op_dense(xp, n, i, j, P["Y"], P["Y"])
        H = H + J * embed_two_qubit_op_dense(xp, n, i, j, P["Z"], P["Z"])

    H = 0.5 * (H + H.conj().T)
    return H


# -----------------------------
# Compute T[t,i,j] on GPU (or CPU fallback)
# -----------------------------

def compute_T_tensor_statevector(xp,
                                 use_gpu: bool,
                                 n: int,
                                 H: "xp.ndarray",
                                 times: np.ndarray,
                                 samples: int,
                                 rng: np.random.Generator,
                                 op_labels: List[str]) -> "xp.ndarray":
    """
    Returns T[t, i, j] typical directed permeability, using pure-state reduced densities.
    """
    P = paulis(xp)
    ops = [(lab, P[lab]) for lab in op_labels if lab != "I"]
    dim = 2 ** n
    t_cp = xp.asarray(times, dtype=xp.float64) if use_gpu else times

    # Eigh once (dominant cost)
    evals, evecs = xp.linalg.eigh(H)  # evals: (dim,), evecs: (dim,dim)
    Vh = evecs.conj().T

    T = xp.zeros((len(times), n, n), dtype=xp.float64)

    for s in range(samples):
        psi0 = random_product_state(xp, n, rng, use_gpu)  # (dim,)
        # coefficients c = V† psi0
        c0 = Vh @ psi0

        # Precompute coefficients for perturbed initial states: c_{i,op} = V† (Oi psi0)
        c_pert = {}  # (i, lab) -> (dim,)
        for i in range(n):
            for lab, gate2 in ops:
                psi_p = apply_one_qubit_gate_to_state(xp, psi0, gate2, i, n)
                c_pert[(i, lab)] = Vh @ psi_p

        for ti in range(len(times)):
            t = t_cp[ti]
            phase = xp.exp(-1j * evals * t)

            # baseline psi(t)
            psi_t = evecs @ (phase * c0)

            # baseline reduced rhos for all j
            rho_base = [reduced_rho_qubit_from_state(xp, psi_t, j, n) for j in range(n)]

            for i in range(n):
                best = xp.zeros((n,), dtype=xp.float64)
                for lab, _gate2 in ops:
                    psi_p_t = evecs @ (phase * c_pert[(i, lab)])
                    for j in range(n):
                        if j == i:
                            continue
                        rho_p = reduced_rho_qubit_from_state(xp, psi_p_t, j, n)
                        d = trace_distance_2x2(xp, rho_p, rho_base[j])
                        # d is scalar xp type
                        if use_gpu:
                            best_j = best[j]
                            best[j] = xp.maximum(best_j, d)
                        else:
                            best[j] = max(float(best[j]), float(d))
                T[ti, i, :] = T[ti, i, :] + best

    T = T / float(samples)
    # zero diagonal
    for ti in range(len(times)):
        if use_gpu:
            idx = xp.arange(n)
            T[ti, idx, idx] = 0.0
        else:
            np.fill_diagonal(T[ti], 0.0)
    return T


# -----------------------------
# Emergent adjacency, lengths, distances (CPU-side; small n)
# -----------------------------

def persistence_weighted_A_np(T_np: np.ndarray, window_start_idx: int, mode: str) -> np.ndarray:
    late = T_np[window_start_idx:, :, :]
    if mode == "median":
        A = np.median(late, axis=0)
    elif mode == "mean":
        A = np.mean(late, axis=0)
    elif mode == "mean_cv":
        mu = np.mean(late, axis=0)
        sigma = np.std(late, axis=0)
        cv = sigma / (mu + 1e-12)
        A = mu * (1.0 / (1.0 + cv))
    else:
        raise ValueError(f"Unknown influence-mode: {mode}")
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float64)


def symmetrize_np(A: np.ndarray, how: str) -> np.ndarray:
    if how == "max":
        return np.maximum(A, A.T)
    if how == "mean":
        return 0.5 * (A + A.T)
    if how == "min":
        return np.minimum(A, A.T)
    raise ValueError(f"Unknown sym: {how}")


def lengths_from_influence_np(W: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    L = -np.log(eps + np.clip(W, 0.0, 1.0))
    np.fill_diagonal(L, 0.0)
    return L


def neighbor_mask_topk_np(W: np.ndarray, k: int) -> np.ndarray:
    n = W.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        idx = np.argsort(W[i])[::-1]
        cnt = 0
        for j in idx:
            if j == i:
                continue
            if W[i, j] <= 0:
                continue
            mask[i, j] = True
            cnt += 1
            if cnt >= k:
                break
    mask = np.logical_or(mask, mask.T)
    np.fill_diagonal(mask, False)
    return mask


def all_pairs_shortest_paths_np(L: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    D = np.full((n, n), np.inf, dtype=np.float64)
    for s in range(n):
        dist = np.full((n,), np.inf, dtype=np.float64)
        dist[s] = 0.0
        used = np.zeros((n,), dtype=bool)
        for _ in range(n):
            u = int(np.argmin(np.where(used, np.inf, dist)))
            if used[u] or not np.isfinite(dist[u]):
                break
            used[u] = True
            for v in range(n):
                if v == u:
                    continue
                if not mask[u, v]:
                    continue
                w = L[u, v]
                if not np.isfinite(w) or w <= 0.0:
                    continue
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
        D[s, :] = dist
    np.fill_diagonal(D, 0.0)
    return D


# -----------------------------
# Diagnostics (CPU)
# -----------------------------

def triangle_violation_stats_np(D: np.ndarray, triples: int, rng: np.random.Generator) -> Dict[str, float]:
    n = D.shape[0]
    tol = 1e-9
    viol = 0
    max_ratio = 0.0
    mean_excess = 0.0
    checked = 0
    for _ in range(triples):
        a, b, c = rng.integers(0, n, size=3)
        if len({int(a), int(b), int(c)}) < 3:
            continue
        dab = D[a, b]
        dbc = D[b, c]
        dac = D[a, c]
        if not (np.isfinite(dab) and np.isfinite(dbc) and np.isfinite(dac)):
            continue
        checked += 1
        rhs = dab + dbc
        if dac > rhs + tol:
            viol += 1
            excess = float(dac - rhs)
            mean_excess += excess
            ratio = float(dac / (rhs + 1e-12))
            if ratio > max_ratio:
                max_ratio = ratio
    if checked == 0:
        return {"checked": 0, "violation_rate": 1.0, "mean_excess": float("nan"), "max_ratio": float("nan")}
    mean_excess = mean_excess / max(1, viol)
    return {
        "checked": checked,
        "violation_rate": viol / float(checked),
        "mean_excess": mean_excess if viol > 0 else 0.0,
        "max_ratio": max_ratio if viol > 0 else 1.0,
    }


def classical_mds_np(D: np.ndarray, dim: int) -> np.ndarray:
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / float(n)
    B = -0.5 * J @ D2 @ J
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    pos = np.maximum(evals[:dim], 0.0)
    X = evecs[:, :dim] * np.sqrt(pos + 1e-18)
    return X.astype(np.float64)


def stress_np(D: np.ndarray, X: np.ndarray) -> float:
    n = D.shape[0]
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dij = D[i, j]
            if not np.isfinite(dij):
                continue
            dh = float(np.linalg.norm(X[i] - X[j]))
            num += (dij - dh) ** 2
            den += dij ** 2
    return float(np.sqrt(num / (den + 1e-18)))


def first_arrival_abs_np(T: np.ndarray, times: np.ndarray, delta: float) -> np.ndarray:
    n = T.shape[1]
    tau = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n):
        tau[i, i] = 0.0
        for j in range(n):
            if i == j:
                continue
            hits = np.where(T[:, i, j] >= delta)[0]
            if hits.size > 0:
                tau[i, j] = float(times[int(hits[0])])
    return tau


def first_arrival_rel_np(T: np.ndarray, times: np.ndarray, eta: float) -> np.ndarray:
    n = T.shape[1]
    tau = np.full((n, n), np.inf, dtype=np.float64)
    Tmax = np.max(T, axis=0)
    for i in range(n):
        tau[i, i] = 0.0
        for j in range(n):
            if i == j:
                continue
            m = float(Tmax[i, j])
            if m <= 0.0:
                continue
            thr = eta * m
            hits = np.where(T[:, i, j] >= thr)[0]
            if hits.size > 0:
                tau[i, j] = float(times[int(hits[0])])
    return tau


def fit_line_np(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3:
        return {"n": int(x.size), "a": float("nan"), "b": float("nan"), "r2": float("nan"), "rmse": float("nan")}
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-18)
    rmse = float(np.sqrt(ss_res / float(x.size)))
    return {"n": int(x.size), "a": a, "b": b, "r2": float(r2), "rmse": rmse}


# -----------------------------
# Sweep parsing
# -----------------------------

def parse_list_or_linspace(spec: str) -> List[float]:
    spec = (spec or "").strip()
    if not spec:
        return []
    if ":" in spec:
        a, b, c = spec.split(":")
        a = float(a); b = float(b); c = int(float(c))
        if c < 2:
            return [a]
        return [float(x) for x in np.linspace(a, b, c)]
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


def parse_int_list(spec: str) -> List[int]:
    spec = (spec or "").strip()
    if not spec:
        return []
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


# -----------------------------
# Plot / output helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_heatmap(mat: np.ndarray, title: str, path: str) -> None:
    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xlabel("j"); plt.ylabel("i")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_curve(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, path: str) -> None:
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_scatter(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, path: str) -> None:
    plt.figure()
    plt.scatter(x, y, s=12)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class SweepRow:
    topk: int
    delta: float
    tau_mode: str
    eta: float
    window_start_time: float
    triangle_checked: int
    triangle_violation_rate: float
    mds_stress_1: float
    mds_stress_2: float
    mds_stress_3: float
    cone_n: int
    cone_a: float
    cone_b: float
    cone_r2: float
    cone_rmse: float


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="GPU HIP->spacetime diagnostics (CuPy).")
    ap.add_argument("--prefer-cupy", action="store_true", help="Prefer CuPy GPU backend if available.")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--graph", type=str, default="ring", choices=["ring", "grid", "er"])
    ap.add_argument("--grid-nx", type=int, default=3)
    ap.add_argument("--grid-ny", type=int, default=3)
    ap.add_argument("--er-p", type=float, default=0.35)

    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--t-max", type=float, default=8.0)
    ap.add_argument("--t-steps", type=int, default=81)
    ap.add_argument("--window-frac", type=float, default=0.5)

    ap.add_argument("--influence-mode", type=str, default="median", choices=["median", "mean", "mean_cv"])
    ap.add_argument("--sym", type=str, default="max", choices=["max", "mean", "min"])

    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--sweep-topk", type=str, default="")
    ap.add_argument("--sweep-delta", type=str, default="")

    ap.add_argument("--tau-mode", type=str, default="rel", choices=["abs", "rel"])
    ap.add_argument("--eta", type=float, default=0.5)

    ap.add_argument("--triangle-triples", type=int, default=4000)

    ap.add_argument("--ops", type=str, default="X,Y,Z,H")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs_spacetime_sweep_gpu")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    xp, use_gpu = get_xp(prefer_cupy=args.prefer_cupy)

    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "runs"))

    # Build graph
    if args.graph == "ring":
        n = args.n
        E = edges_ring(n)
        graph_desc = f"ring(n={n})"
    elif args.graph == "grid":
        n, E = edges_grid_2d(args.grid_nx, args.grid_ny)
        graph_desc = f"grid({args.grid_nx}x{args.grid_ny})"
    else:
        n = args.n
        E = edges_erdos_renyi(n, args.er_p, rng)
        graph_desc = f"er(n={n},p={args.er_p})"

    dim = 2 ** n
    print(f"[backend] {'cupy(GPU)' if use_gpu else 'numpy(CPU)'}")
    print(f"[info] n={n} dim={dim} graph={graph_desc} |E|={len(E)}")
    print(f"[info] samples={args.samples} t_steps={args.t_steps} t_max={args.t_max}")

    # Build H (dense) on backend
    H = build_hamiltonian_dense(xp, n, E, rng, J=1.0, h=0.3)

    # Times (CPU list; we pass numpy times into compute and convert as needed)
    times = np.linspace(0.0, float(args.t_max), int(args.t_steps)).astype(np.float64)

    op_labels = [s.strip() for s in args.ops.split(",") if s.strip()]
    print(f"[info] ops={op_labels}")

    print("[phase] computing T[t,i,j] (this is the expensive step)...")
    T_xp = compute_T_tensor_statevector(
        xp=xp,
        use_gpu=use_gpu,
        n=n,
        H=H,
        times=times,
        samples=int(args.samples),
        rng=rng,
        op_labels=op_labels
    )

    # Bring T back to CPU for sweeps / plotting (small)
    if use_gpu:
        import cupy as cp
        T = cp.asnumpy(T_xp)
    else:
        T = np.asarray(T_xp)

    # Window index
    widx = int(np.floor(args.window_frac * (len(times) - 1)))
    widx = max(0, min(widx, len(times) - 2))
    window_start_time = float(times[widx])
    print(f"[info] late-time window starts at index {widx} (t={window_start_time:.3f})")

    # Aggregate A once
    A_dir = persistence_weighted_A_np(T, window_start_idx=widx, mode=args.influence_mode)
    A_sym = symmetrize_np(A_dir, how=args.sym)

    # Save global plots
    save_heatmap(T[-1], f"T(t_max) (t={times[-1]:.2f})", os.path.join(args.out, "T_tmax.png"))
    save_heatmap(A_dir, f"A_dir ({args.influence_mode})", os.path.join(args.out, "A_dir.png"))
    save_heatmap(A_sym, f"A_sym (sym={args.sym})", os.path.join(args.out, "A_sym.png"))

    # Sweep lists
    topk_list = parse_int_list(args.sweep_topk) if args.sweep_topk.strip() else [int(args.topk)]
    delta_list = parse_list_or_linspace(args.sweep_delta) if args.sweep_delta.strip() else [float(args.delta)]

    if args.tau_mode == "rel" and args.sweep_delta.strip():
        print("[warn] tau-mode rel ignores delta numerically (kept only as a sweep label). "
              "Use tau-mode abs if you want delta to matter.")

    print(f"[sweep] topk_list={topk_list}")
    print(f"[sweep] delta_list={delta_list}")
    print(f"[sweep] tau_mode={args.tau_mode} eta={args.eta}")

    rows: List[SweepRow] = []
    best_score = -np.inf
    best_tag = None

    for topk in topk_list:
        mask = neighbor_mask_topk_np(A_sym, k=int(topk))
        L = lengths_from_influence_np(A_sym, eps=1e-6)
        Lm = L.copy()
        Lm[~mask] = np.inf
        np.fill_diagonal(Lm, 0.0)
        D = all_pairs_shortest_paths_np(Lm, mask=mask)

        tri = triangle_violation_stats_np(D, triples=int(args.triangle_triples), rng=rng)

        max_dim = min(6, n - 1)
        mds = {}
        for d in range(1, max_dim + 1):
            X = classical_mds_np(D, dim=d)
            mds[d] = stress_np(D, X)

        for delta in delta_list:
            tag = f"topk{topk}_delta{delta:.6g}_{args.tau_mode}"
            run_dir = os.path.join(args.out, "runs", tag)
            ensure_dir(run_dir)

            if args.tau_mode == "abs":
                tau = first_arrival_abs_np(T, times, float(delta))
            else:
                tau = first_arrival_rel_np(T, times, float(args.eta))

            xs = []
            ys = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if not np.isfinite(D[i, j]):
                        continue
                    if not np.isfinite(tau[i, j]):
                        continue
                    if D[i, j] <= 0.0:
                        continue
                    xs.append(D[i, j])
                    ys.append(tau[i, j])
            xs = np.array(xs, dtype=np.float64)
            ys = np.array(ys, dtype=np.float64)

            cone = fit_line_np(xs, ys)

            # Save run artifacts
            save_heatmap(mask.astype(np.float64), f"mask(topk={topk})", os.path.join(run_dir, "mask.png"))
            save_heatmap(D, f"D_emergent(topk={topk})", os.path.join(run_dir, "D.png"))
            dims = list(range(1, max_dim + 1))
            stresses = [float(mds[d]) for d in dims]
            save_curve(dims, stresses, f"MDS stress (topk={topk})", "dim", "stress-1", os.path.join(run_dir, "mds_stress.png"))
            if xs.size >= 3:
                save_scatter(xs, ys, f"tau vs d (topk={topk}, mode={args.tau_mode})", "d(i,j)", "tau(i->j)", os.path.join(run_dir, "cone.png"))

            np.savez_compressed(
                os.path.join(run_dir, "run.npz"),
                times=times,
                T=T,
                A_dir=A_dir,
                A_sym=A_sym,
                mask=mask,
                L=L,
                D=D,
                tau=tau,
                edges=np.array(E, dtype=np.int64),
                topk=int(topk),
                delta=float(delta),
                tau_mode=args.tau_mode,
                eta=float(args.eta),
            )

            row = SweepRow(
                topk=int(topk),
                delta=float(delta),
                tau_mode=args.tau_mode,
                eta=float(args.eta),
                window_start_time=window_start_time,
                triangle_checked=int(tri["checked"]),
                triangle_violation_rate=float(tri["violation_rate"]),
                mds_stress_1=float(mds.get(1, float("nan"))),
                mds_stress_2=float(mds.get(2, float("nan"))),
                mds_stress_3=float(mds.get(3, float("nan"))),
                cone_n=int(cone["n"]),
                cone_a=float(cone["a"]),
                cone_b=float(cone["b"]),
                cone_r2=float(cone["r2"]),
                cone_rmse=float(cone["rmse"]),
            )
            rows.append(row)

            # Best score: prioritize cone_r2, penalize triangle violations, reward more pairs
            n_pairs = max(1, int(cone["n"]))
            score = float(cone["r2"]) - 0.25 * float(tri["violation_rate"]) + 0.02 * math.log(float(n_pairs))

            if np.isfinite(score) and score > best_score:
                best_score = score
                best_tag = tag

            print(f"[run] {tag} cone_r2={cone['r2']:.4f} n={cone['n']} tri_v={tri['violation_rate']:.4f} "
                  f"mds2={float(mds.get(2, float('nan'))):.3f} score={score:.4f}")

    # Write sweep tables
    csv_path = os.path.join(args.out, "sweep_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if rows:
            w.writerow(list(asdict(rows[0]).keys()))
            for r in rows:
                w.writerow(list(asdict(r).values()))

    json_path = os.path.join(args.out, "sweep_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

    # Copy best run
    if best_tag is not None:
        best_src = os.path.join(args.out, "runs", best_tag)
        best_dst = os.path.join(args.out, "best")
        if os.path.isdir(best_dst):
            shutil.rmtree(best_dst)
        ensure_dir(best_dst)

        for fn in ["mask.png", "D.png", "mds_stress.png", "cone.png", "run.npz"]:
            src = os.path.join(best_src, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(best_dst, fn))

        best_summary = {
            "best_tag": best_tag,
            "best_score": best_score,
            "backend": "cupy" if use_gpu else "numpy",
            "graph": graph_desc,
            "n": n,
            "dim": dim,
            "edges": len(E),
            "samples": int(args.samples),
            "t_max": float(args.t_max),
            "t_steps": int(args.t_steps),
            "window_start_time": window_start_time,
            "influence_mode": args.influence_mode,
            "sym": args.sym,
            "tau_mode": args.tau_mode,
            "eta": float(args.eta),
            "topk_list": topk_list,
            "delta_list": delta_list,
        }
        with open(os.path.join(best_dst, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

        print(f"[best] {best_tag} score={best_score:.4f}")
        print(f"[best] artifacts in: {best_dst}")

    print(f"[done] wrote: {csv_path}")
    print(f"[done] wrote: {json_path}")


if __name__ == "__main__":
    main()
