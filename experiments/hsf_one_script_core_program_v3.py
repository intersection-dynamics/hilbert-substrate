#!/usr/bin/env python3
"""
hsf_one_script_core_program_v3.py

Single-file demonstrator of the *core* HSF claims (distilled, reproducible):
  A) Emergence of spatial locality from constrained influence/accessibility
  B) Emergence of a preferred Hilbert-space factorization (bidirectional link L/R structure)

v3 changes (based on your v2 output):
  - Locality: add finite coordination (top-z neighbors per node) to prevent hub-collapse
  - Factorization: add balance penalty + kron-error tie-break so 4x4 beats 2x8 when appropriate

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def now_iso() -> str:
    import datetime as dt
    return dt.datetime.now().isoformat(timespec="seconds")

def frob(x: np.ndarray) -> float:
    return float(np.sqrt(np.vdot(x, x).real))

def symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def unitary_random(n: int) -> np.ndarray:
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / math.sqrt(2.0)
    q, r = la.qr(z)
    d = np.diag(r)
    ph = d / (np.abs(d) + 1e-12)
    return q * ph

def hermitian_random(n: int, scale: float = 1.0) -> np.ndarray:
    x = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / math.sqrt(2.0)
    h = x + x.conj().T
    w = la.eigvalsh(h)
    h = h / (np.max(np.abs(w)) + 1e-12) * scale
    return 0.5 * (h + h.conj().T)

def expm_unitary_from_antiherm(K: np.ndarray) -> np.ndarray:
    return la.expm(K)


# ----------------------------
# Part A: Locality from constrained influence flow
# ----------------------------

@dataclass
class ConstraintWeights:
    finite_bandwidth: float = 1.0
    no_signaling: float = 1.0
    no_refolding: float = 1.0
    no_forgetting: float = 1.0

@dataclass
class LocalityMetrics:
    knn_reciprocity: float
    edge_length_cv: float
    lattice_score: float

def initial_influence_kernel(n_sites: int, seed_scale: float = 1.0) -> np.ndarray:
    x = np.random.randn(n_sites, n_sites)
    a = symmetrize(x)
    np.fill_diagonal(a, 0.0)
    a = np.maximum(0.0, a)
    a = a / (np.max(a) + 1e-12) * seed_scale
    return a.astype(np.float64)

def spectral_bandwidth_truncate_pos(a: np.ndarray, k: int) -> np.ndarray:
    w, v = la.eigh(a)
    idx = np.argsort(np.abs(w))[::-1]
    keep = idx[:k]
    w2 = np.zeros_like(w)
    w2[keep] = w[keep]
    a2 = (v * w2) @ v.T
    a2 = symmetrize(a2)
    np.fill_diagonal(a2, 0.0)
    a2 = np.maximum(0.0, a2)
    # rescale to match original scale
    m = np.max(a)
    if np.max(a2) > 0:
        a2 = a2 / (np.max(a2) + 1e-12) * (m + 1e-12)
    return a2

def embed_from_kernel(a: np.ndarray, dim: int = 3) -> np.ndarray:
    w = np.maximum(0.0, a.copy())
    np.fill_diagonal(w, 0.0)
    d = np.sum(w, axis=1) + 1e-12
    Dinv_sqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(w.shape[0]) - Dinv_sqrt @ w @ Dinv_sqrt
    evals, evecs = la.eigh(L)
    coords = evecs[:, 1:1 + dim].real
    coords = coords / (np.std(coords) + 1e-12)
    return coords

def knn_graph(coords: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    d2 = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argsort(d2, axis=1)[:, :k]
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, nbrs[i]] = 1.0
    return A

def locality_metrics_from_embedding(coords: np.ndarray, k: int = 4) -> LocalityMetrics:
    A = knn_graph(coords, k=k)
    mutual = A * A.T
    recip = float(mutual.sum() / (A.sum() + 1e-12))

    d2 = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argsort(d2, axis=1)[:, :k]
    lengths = []
    for i in range(coords.shape[0]):
        for j in nbrs[i]:
            lengths.append(math.sqrt(float(d2[i, j])))
    lengths = np.array(lengths, dtype=np.float64)
    mean = float(np.mean(lengths) + 1e-12)
    cv = float(np.std(lengths) / mean)

    lattice = clip01(0.65 * recip + 0.35 * (1.0 - clip01(cv)))
    return LocalityMetrics(recip, cv, lattice)

def sparsify_topz(A: np.ndarray, z: int) -> np.ndarray:
    """
    Finite coordination constraint:
      keep only top-z neighbors per node (by weight), then symmetrize.
    Prevents hub-collapse and enforces a local-graph flavor.
    """
    n = A.shape[0]
    if z >= n - 1:
        return A
    B = np.zeros_like(A)
    for i in range(n):
        row = A[i].copy()
        row[i] = 0.0
        # indices of top z weights
        idx = np.argsort(row)[::-1][:z]
        B[i, idx] = row[idx]
    B = symmetrize(B)
    np.fill_diagonal(B, 0.0)
    B = np.maximum(0.0, B)
    return B

def constrained_flow(
    A0: np.ndarray,
    weights: ConstraintWeights,
    steps: int,
    dt: float,
    bandwidth_k: int,
    memory_alpha: float,
    refold_beta: float,
    signal_mix: float,
    coord_z: int,
) -> np.ndarray:
    """
    v3: keep A nonnegative + add finite coordination (top-z sparsification) each step.
    """
    A = A0.copy()
    A_hist = A0.copy()
    scale0 = float(np.max(A0) + 1e-12)

    for _ in range(steps):
        # finite bandwidth (spectral)
        if weights.finite_bandwidth > 0:
            Abw = spectral_bandwidth_truncate_pos(A, k=bandwidth_k)
            lam = 0.25 * weights.finite_bandwidth
            A = (1 - lam) * A + lam * Abw

        # no-forgetting (history)
        if weights.no_forgetting > 0:
            A_hist = (1 - memory_alpha) * A_hist + memory_alpha * A
            lam = 0.20 * weights.no_forgetting
            A = (1 - lam) * A + lam * A_hist

        # signaling proxy: align toward distance kernel softly
        coords = embed_from_kernel(A, dim=3)
        if weights.no_signaling > 0:
            d2 = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=2)
            sig2 = float(np.median(d2[np.isfinite(d2)]) + 1e-12)
            K = np.exp(-d2 / (sig2 + 1e-12))
            np.fill_diagonal(K, 0.0)
            K = K / (np.max(K) + 1e-12) * scale0

            lam = float(max(0.0, min(0.6, signal_mix * weights.no_signaling)))
            A = (1 - lam) * A + lam * K

        # no-refolding proxy: resist rapid rewiring
        if weights.no_refolding > 0:
            delta = A - A_hist
            A = A - dt * weights.no_refolding * refold_beta * delta

        # enforce nonnegativity + finite coordination
        A = symmetrize(A)
        np.fill_diagonal(A, 0.0)
        A = np.maximum(0.0, A)
        A = sparsify_topz(A, z=coord_z)

        # rescale
        m = float(np.max(A) + 1e-12)
        A = A / m * scale0

    return A


# ----------------------------
# Part B: Factorization via basis search + balanced tie-break
# ----------------------------

@dataclass
class FactorizationDiagnostics:
    inferred_dims: Tuple[int, int]
    objective_tail: float
    rel_kron_error_rankR: float
    score: float
    top_singular_values: List[float]

def operator_schmidt_singular_values(M: np.ndarray, dL: int, dR: int) -> np.ndarray:
    T = M.reshape(dL, dR, dL, dR).transpose(0, 2, 1, 3).reshape(dL*dL, dR*dR)
    return la.svd(T, compute_uv=False)

def schmidt_tail_objective(M: np.ndarray, dL: int, dR: int, rank_keep: int) -> Tuple[float, List[float]]:
    s = operator_schmidt_singular_values(M, dL, dR)
    s = np.maximum(0.0, s.real)
    tot = float(np.sum(s) + 1e-12)
    tail = float(np.sum(s[rank_keep:]))
    obj = tail / tot
    top = [float(x) for x in s[:10]]
    return obj, top

def kron_rankR_approx_error(M: np.ndarray, dL: int, dR: int, R: int) -> float:
    T = M.reshape(dL, dR, dL, dR).transpose(0, 2, 1, 3).reshape(dL*dL, dR*dR)
    U, s, Vh = la.svd(T, full_matrices=False)
    R = int(min(R, len(s)))
    approx = np.zeros_like(M)
    for r in range(R):
        u = U[:, r] * math.sqrt(float(s[r]))
        v = Vh[r, :] * math.sqrt(float(s[r]))
        A = u.reshape(dL, dL)
        B = v.reshape(dR, dR)
        approx = approx + np.kron(A, B)
    return float(frob(M - approx) / (frob(M) + 1e-12))

def scramble_factorization_operator(dL: int, dR: int) -> Tuple[np.ndarray, np.ndarray]:
    A = hermitian_random(dL, scale=1.0)
    B = hermitian_random(dR, scale=1.0)
    I_L = np.eye(dL, dtype=np.complex128)
    I_R = np.eye(dR, dtype=np.complex128)
    eps = 0.35
    H_true = np.kron(A, I_R) + np.kron(I_L, B) + eps * np.kron(A, B)
    H_true = 0.5 * (H_true + H_true.conj().T)

    D = dL * dR
    U = unitary_random(D)
    H_scr = U @ H_true @ U.conj().T
    H_scr = 0.5 * (H_scr + H_scr.conj().T)
    return H_true, H_scr

def basis_search_low_schmidt(
    H: np.ndarray,
    dL: int,
    dR: int,
    rank_keep: int,
    iters: int,
    step_scale: float,
    restarts: int,
) -> Tuple[np.ndarray, float, List[float]]:
    D = dL * dR
    I = np.eye(D, dtype=np.complex128)

    best_obj = None
    best_W = None
    best_top = None

    for r in range(restarts):
        W = I.copy() if r == 0 else unitary_random(D)
        M = W.conj().T @ H @ W
        obj, top = schmidt_tail_objective(M, dL, dR, rank_keep)

        for _ in range(iters):
            X = (np.random.randn(D, D) + 1j*np.random.randn(D, D)) / math.sqrt(2.0)
            X = X - X.conj().T  # anti-Hermitian
            X = X / (frob(X) + 1e-12)
            K = step_scale * X
            U = expm_unitary_from_antiherm(K)
            W_try = W @ U
            M_try = W_try.conj().T @ H @ W_try
            obj_try, top_try = schmidt_tail_objective(M_try, dL, dR, rank_keep)
            if obj_try < obj:
                W, obj, top = W_try, obj_try, top_try

        if best_obj is None or obj < best_obj:
            best_obj, best_W, best_top = obj, W, top

    return best_W, float(best_obj), best_top

def recover_factorization(
    H_scr: np.ndarray,
    candidates: List[Tuple[int, int]],
    rank_keep: int,
    iters: int,
    step_scale: float,
    restarts: int,
    kron_rank: int,
    lambda_balance: float,
    lambda_kron: float,
) -> FactorizationDiagnostics:
    best: FactorizationDiagnostics | None = None

    for dL, dR in candidates:
        if dL * dR != H_scr.shape[0]:
            continue

        W, obj, top = basis_search_low_schmidt(
            H_scr, dL, dR, rank_keep=rank_keep, iters=iters,
            step_scale=step_scale, restarts=restarts
        )
        M = W.conj().T @ H_scr @ W
        rel = kron_rankR_approx_error(M, dL, dR, R=kron_rank)

        # Balanced bidirectional link preference:
        bal = abs(math.log((dL + 1e-12) / (dR + 1e-12)))
        score = obj + lambda_balance * bal + lambda_kron * rel

        diag = FactorizationDiagnostics(
            inferred_dims=(dL, dR),
            objective_tail=float(obj),
            rel_kron_error_rankR=float(rel),
            score=float(score),
            top_singular_values=top,
        )

        if best is None or diag.score < best.score:
            best = diag

    assert best is not None
    return best


# ----------------------------
# Plotting
# ----------------------------

def plot_influence_matrix(A: np.ndarray, outpath: Path, title: str) -> None:
    ensure_dir(outpath.parent)
    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(A, aspect="auto")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_embedding(coords: np.ndarray, outpath: Path, title: str) -> None:
    ensure_dir(outpath.parent)
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=32)
    for i, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(i), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_singular_values(sv: List[float], outpath: Path, title: str) -> None:
    ensure_dir(outpath.parent)
    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(111)
    x = np.arange(1, len(sv) + 1)
    ax.plot(x, sv, marker="o")
    ax.set_title(title)
    ax.set_xlabel("index"); ax.set_ylabel("singular value")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="hsf_one_script_out_v3")
    ap.add_argument("--seed", type=int, default=0)

    # Locality
    ap.add_argument("--n_sites", type=int, default=16)
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--bandwidth_k", type=int, default=8)
    ap.add_argument("--memory_alpha", type=float, default=0.08)
    ap.add_argument("--refold_beta", type=float, default=0.85)
    ap.add_argument("--signal_mix", type=float, default=0.12)
    ap.add_argument("--coord_z", type=int, default=4, help="Finite coordination: neighbors kept per node.")

    ap.add_argument("--w_bandwidth", type=float, default=1.0)
    ap.add_argument("--w_signaling", type=float, default=1.0)
    ap.add_argument("--w_refolding", type=float, default=1.0)
    ap.add_argument("--w_forgetting", type=float, default=1.0)

    # Factorization
    ap.add_argument("--dL", type=int, default=4)
    ap.add_argument("--dR", type=int, default=4)
    ap.add_argument("--try_dims", type=str, default="2x8,4x4,8x2")

    ap.add_argument("--schmidt_rank", type=int, default=3)
    ap.add_argument("--schmidt_iters", type=int, default=450)
    ap.add_argument("--schmidt_step", type=float, default=0.14)
    ap.add_argument("--schmidt_restarts", type=int, default=4)
    ap.add_argument("--kron_rank", type=int, default=3)

    ap.add_argument("--lambda_balance", type=float, default=0.20)
    ap.add_argument("--lambda_kron", type=float, default=0.08)

    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)

    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir / "figures")
    ensure_dir(outdir / "data")

    # ---- A) Locality
    weights = ConstraintWeights(
        finite_bandwidth=float(args.w_bandwidth),
        no_signaling=float(args.w_signaling),
        no_refolding=float(args.w_refolding),
        no_forgetting=float(args.w_forgetting),
    )

    A0 = initial_influence_kernel(args.n_sites, seed_scale=1.0)
    A = constrained_flow(
        A0=A0, weights=weights, steps=int(args.steps), dt=float(args.dt),
        bandwidth_k=int(args.bandwidth_k), memory_alpha=float(args.memory_alpha),
        refold_beta=float(args.refold_beta), signal_mix=float(args.signal_mix),
        coord_z=int(args.coord_z),
    )

    coords0 = embed_from_kernel(A0, dim=3)
    coords = embed_from_kernel(A, dim=3)

    k_nn = max(3, min(6, args.n_sites // 4))
    m0 = locality_metrics_from_embedding(coords0, k=k_nn)
    m = locality_metrics_from_embedding(coords, k=k_nn)

    plot_influence_matrix(A0, outdir / "figures" / "locality_A0_influence.png",
                          "Initial influence (pre-geometry, nonnegative)")
    plot_influence_matrix(A,  outdir / "figures" / "locality_A_constrained.png",
                          f"Constrained influence (post-flow, top-z={args.coord_z})")
    plot_embedding(coords0, outdir / "figures" / "locality_embedding_initial.png",
                   f"Initial embedding (lattice_score={m0.lattice_score:.3f})")
    plot_embedding(coords,  outdir / "figures" / "locality_embedding_constrained.png",
                   f"Constrained embedding (lattice_score={m.lattice_score:.3f})")

    # ---- B) Factorization
    _, H_scr = scramble_factorization_operator(args.dL, args.dR)
    D = H_scr.shape[0]

    cand: List[Tuple[int, int]] = []
    for token in args.try_dims.split(","):
        token = token.strip().lower()
        if "x" not in token:
            continue
        a, b = token.split("x", 1)
        try:
            cand.append((int(a), int(b)))
        except ValueError:
            pass

    diag = recover_factorization(
        H_scr=H_scr, candidates=cand,
        rank_keep=int(args.schmidt_rank),
        iters=int(args.schmidt_iters),
        step_scale=float(args.schmidt_step),
        restarts=int(args.schmidt_restarts),
        kron_rank=int(args.kron_rank),
        lambda_balance=float(args.lambda_balance),
        lambda_kron=float(args.lambda_kron),
    )

    bandwidth_profile = " ".join([f"{x:.6g}" for x in diag.top_singular_values])
    plot_singular_values(
        diag.top_singular_values,
        outdir / "figures" / "factorization_top_singular_values.png",
        f"Top Schmidt SVs (best dims={diag.inferred_dims[0]}x{diag.inferred_dims[1]}, score={diag.score:.3f})"
    )

    summary: Dict[str, Any] = {
        "timestamp": now_iso(),
        "seed": int(args.seed),
        "locality": {
            "n_sites": int(args.n_sites),
            "k_nn": int(k_nn),
            "coord_z": int(args.coord_z),
            "weights": vars(weights),
            "flow": {
                "steps": int(args.steps),
                "dt": float(args.dt),
                "bandwidth_k": int(args.bandwidth_k),
                "memory_alpha": float(args.memory_alpha),
                "refold_beta": float(args.refold_beta),
                "signal_mix": float(args.signal_mix),
            },
            "metrics_initial": vars(m0),
            "metrics_constrained": vars(m),
        },
        "factorization": {
            "true_dims": [int(args.dL), int(args.dR)],
            "candidate_dims": [[int(a), int(b)] for a, b in cand],
            "inferred_dims": [int(diag.inferred_dims[0]), int(diag.inferred_dims[1])],
            "schmidt_rank_keep": int(args.schmidt_rank),
            "objective_tail_fraction": float(diag.objective_tail),
            "kron_rank": int(args.kron_rank),
            "relative_kron_fit_error_rankR": float(diag.rel_kron_error_rankR),
            "lambda_balance": float(args.lambda_balance),
            "lambda_kron": float(args.lambda_kron),
            "score": float(diag.score),
            "top_singular_values": diag.top_singular_values,
            "bandwidth_profile_line": bandwidth_profile,
        },
        "figures": {
            "locality_A0_influence": "figures/locality_A0_influence.png",
            "locality_A_constrained": "figures/locality_A_constrained.png",
            "locality_embedding_initial": "figures/locality_embedding_initial.png",
            "locality_embedding_constrained": "figures/locality_embedding_constrained.png",
            "factorization_top_singular_values": "figures/factorization_top_singular_values.png",
        },
    }
    (outdir / "data" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    elapsed = time.time() - t0

    print("\n=== HSF one-script core program (v3) ===")
    print(f"outdir: {outdir}")

    print("\n[Locality]")
    print(f"  lattice_score: initial={m0.lattice_score:.4f}  constrained={m.lattice_score:.4f}")
    print(f"  kNN reciprocity: initial={m0.knn_reciprocity:.4f}  constrained={m.knn_reciprocity:.4f}")
    print(f"  edge length CV: initial={m0.edge_length_cv:.4f}  constrained={m.edge_length_cv:.4f}")

    print("\n[Factorization]")
    print(f"  true dims: {args.dL}x{args.dR}  (D={D})")
    print(f"  inferred dims: {diag.inferred_dims[0]}x{diag.inferred_dims[1]}")
    print(f"  schmidt-tail objective: {diag.objective_tail:.6f}")
    print(f"  rel_kron_fit_error_rank{args.kron_rank}: {diag.rel_kron_error_rankR:.6f}")
    print(f"  score (tail + balance + kron): {diag.score:.6f}")
    print(f"  bandwidth_profile: {bandwidth_profile}")

    print(f"\nTotal runtime: {elapsed:.2f}s")
    print("=======================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())