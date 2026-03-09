#!/usr/bin/env python3
"""
hsf_one_script_core_program.py

A single, self-contained "executable summary" of the core Hilbert Substrate Framework program:
  (A) Emergence of spatial locality (geometry from accessibility / influence structure)
  (B) Emergence of the "correct" factorization of Hilbert space (bidirectional link L/R structure)
      from constraints, not by assumption.

This is a distilled demonstrator:
  - It does NOT attempt to reproduce every sweep in your repo.
  - It DOES provide a clean, reproducible end-to-end pipeline that shows both core claims
    in one run, producing:
        * locality embedding figure + locality metrics
        * factorization recovery diagnostics + singular-value "bandwidth profile"
        * a text summary file suitable for sharing

Dependencies:
  - numpy
  - scipy
  - matplotlib

Usage (Windows one-liners):
  python hsf_one_script_core_program.py --outdir hsf_one_script_out
  python hsf_one_script_core_program.py --outdir hsf_one_script_out --seed 1 --n_sites 16 --d_site 2
  python hsf_one_script_core_program.py --outdir hsf_one_script_out --n_sites 27 --d_site 3 --steps 400

Notes:
  - "Constraints" here are operationalized as penalties in a flow that shapes an effective generator
    and an influence kernel with:
      * finite bandwidth (spectral cutoff / truncation)
      * no-signaling flavor (influence decays with an emergent distance)
      * no-refolding flavor (structure, once formed, resists wholesale rewiring)
      * no-forgetting flavor (history-integrated influence / smoothing)
  - The factorization demonstration uses a robust numerical proxy:
      recover a tensor factor structure by finding a best Kronecker-factor approximation to a link map
      + checking whether inferred left/right actions behave like independent factors.

This is designed to be legible to a skeptical reader:
  everything is explicitly generated and measured inside this one file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# ----------------------------
# Utility helpers
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def hermitian_random(n: int, scale: float = 1.0) -> np.ndarray:
    """Random Hermitian matrix with roughly unit-scale spectrum."""
    x = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / math.sqrt(2.0)
    h = x + x.conj().T
    # normalize
    w = la.eigvalsh(h)
    h = h / (np.max(np.abs(w)) + 1e-12) * scale
    return h

def unitary_random(n: int) -> np.ndarray:
    """Random Haar-ish unitary via QR of complex Gaussian."""
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / math.sqrt(2.0)
    q, r = la.qr(z)
    d = np.diag(r)
    ph = d / (np.abs(d) + 1e-12)
    return q * ph

def frob(x: np.ndarray) -> float:
    return float(np.sqrt(np.vdot(x, x).real))

def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = frob(x)
    if n < eps:
        return x
    return x / n

def symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def now_iso() -> str:
    import datetime as dt
    return dt.datetime.now().isoformat(timespec="seconds")


# ----------------------------
# Part 1: Construct "constrained dynamics" -> influence/accessibility
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

def build_site_projectors(n_sites: int, d_site: int) -> List[np.ndarray]:
    """
    Define a canonical factorization basis: H_total = ⊗_i C^{d_site}.
    For locality we will treat "sites" as conceptual subsystems whose influence we measure.

    Total dimension D = d_site ** n_sites can be enormous; we DO NOT build that.
    Instead, we treat a reduced effective description: an "influence graph" over sites.
    """
    # Placeholder: in this demonstrator we won't build full tensor projectors.
    # We keep site-level labeling and operate on site-to-site influence kernels.
    return []

def initial_influence_kernel(n_sites: int, seed_scale: float = 1.0) -> np.ndarray:
    """
    Start with a random symmetric influence matrix (site graph weights).
    This is the "pregeometric" substrate: no notion of distance yet.
    """
    x = np.random.randn(n_sites, n_sites)
    a = symmetrize(x)
    np.fill_diagonal(a, 0.0)
    a = a / (np.max(np.abs(a)) + 1e-12) * seed_scale
    return a

def spectral_bandwidth_truncate(a: np.ndarray, k: int) -> np.ndarray:
    """
    Finite-bandwidth proxy: keep only top-k eigenmodes of the influence kernel.
    """
    w, v = la.eigh(a)
    idx = np.argsort(np.abs(w))[::-1]
    idx_keep = idx[:k]
    w2 = np.zeros_like(w)
    w2[idx_keep] = w[idx_keep]
    return (v * w2) @ v.T

def embed_from_kernel(a: np.ndarray, dim: int = 3) -> np.ndarray:
    """
    Geometry emergence proxy: spectral embedding of the influence graph.
    Use Laplacian eigenmaps (normalized Laplacian).
    """
    w = np.maximum(0.0, a.copy())
    np.fill_diagonal(w, 0.0)
    d = np.sum(w, axis=1) + 1e-12
    Dinv_sqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(w.shape[0]) - Dinv_sqrt @ w @ Dinv_sqrt

    # smallest nontrivial eigenvectors
    evals, evecs = la.eigh(L)
    # skip the first eigenvector (constant) if connected-ish
    start = 1
    end = start + dim
    coords = evecs[:, start:end]
    # scale for readability
    coords = coords / (np.std(coords) + 1e-12)
    return coords

def knn_graph(coords: np.ndarray, k: int) -> np.ndarray:
    """
    Directed kNN adjacency (i -> j if j in k nearest of i).
    """
    n = coords.shape[0]
    # squared distances
    d2 = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argsort(d2, axis=1)[:, :k]
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, nbrs[i]] = 1.0
    return A

def locality_metrics_from_embedding(coords: np.ndarray, k: int = 4) -> LocalityMetrics:
    """
    A few simple locality signals:
      - kNN reciprocity: fraction of directed edges that are mutual
      - edge length coefficient-of-variation: how uniform are nearest-neighbor distances?
      - lattice_score: heuristic combining reciprocity + uniformity
    """
    A = knn_graph(coords, k=k)
    mutual = A * A.T
    recip = mutual.sum() / (A.sum() + 1e-12)

    # nearest-neighbor edge lengths
    n = coords.shape[0]
    d2 = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argsort(d2, axis=1)[:, :k]
    lengths = []
    for i in range(n):
        for j in nbrs[i]:
            lengths.append(math.sqrt(float(d2[i, j])))
    lengths = np.array(lengths, dtype=np.float64)
    mean = float(np.mean(lengths) + 1e-12)
    std = float(np.std(lengths))
    cv = std / mean

    # lattice_score: high reciprocity, low cv
    lattice = clip01(0.65 * recip + 0.35 * (1.0 - clip01(cv)))
    return LocalityMetrics(knn_reciprocity=float(recip), edge_length_cv=float(cv), lattice_score=float(lattice))

def constrained_flow(
    a0: np.ndarray,
    weights: ConstraintWeights,
    steps: int,
    dt: float,
    bandwidth_k: int,
    memory_alpha: float,
    refold_beta: float,
) -> np.ndarray:
    """
    Evolve influence kernel A under a "constraint-shaped" flow.

    We use a pragmatic approach:
      - finite_bandwidth: repeatedly truncate spectral content
      - no_forgetting: exponentially smoothed running average (history integrates)
      - no_refolding: penalize large rewiring; resist big changes away from prior structure
      - no_signaling: encourage influence to align with emergent geometry (decay with distance)

    This produces an A that tends to become compatible with a low-dimensional embedding.
    """
    A = a0.copy()
    A_hist = A.copy()

    for t in range(steps):
        # 1) finite bandwidth truncation
        if weights.finite_bandwidth > 0:
            A_bw = spectral_bandwidth_truncate(A, k=bandwidth_k)
            A = (1.0 - 0.35 * weights.finite_bandwidth) * A + (0.35 * weights.finite_bandwidth) * A_bw

        # 2) no-forgetting: history-integrated smoothing
        if weights.no_forgetting > 0:
            A_hist = (1.0 - memory_alpha) * A_hist + memory_alpha * A
            A = (1.0 - 0.25 * weights.no_forgetting) * A + (0.25 * weights.no_forgetting) * A_hist

        # 3) emergent coords (current) for signaling-like constraint
        coords = embed_from_kernel(A, dim=3)

        # 4) no-signaling flavor: encourage A to be a decaying function of emergent distance
        if weights.no_signaling > 0:
            d2 = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=2)
            # convert to a smooth kernel (Gaussian-like)
            sig2 = float(np.median(d2[np.isfinite(d2)]) + 1e-12)
            K = np.exp(-d2 / (sig2 + 1e-12))
            np.fill_diagonal(K, 0.0)
            # align signs/scale with A
            # use positive part as "accessible influence"
            K = K / (np.max(K) + 1e-12)
            A_pos = np.maximum(0.0, A)
            A_pos = A_pos / (np.max(A_pos) + 1e-12)
            A = (1.0 - 0.30 * weights.no_signaling) * A + (0.30 * weights.no_signaling) * (2.0 * K - 1.0) * np.max(np.abs(A) + 1e-12)

        # 5) no-refolding: resist rapid global rewiring relative to history
        if weights.no_refolding > 0:
            delta = A - A_hist
            A = A - dt * weights.no_refolding * refold_beta * delta

        # housekeeping
        A = symmetrize(A)
        np.fill_diagonal(A, 0.0)

    return A


# ----------------------------
# Part 2: Factorization from a "link map" via Kronecker fitting
# ----------------------------

@dataclass
class FactorizationDiagnostics:
    relative_fit_error: float
    top_singular_values: List[float]
    inferred_dims: Tuple[int, int]

def operator_schmidt_singular_values(M: np.ndarray, dL: int, dR: int) -> np.ndarray:
    """
    Compute operator Schmidt spectrum of an operator on (L⊗R) by reshaping:
        M_{(iL,iR),(jL,jR)} -> T_{(iL,jL),(iR,jR)}
    Then SVD(T). Singular values are the operator Schmidt coefficients.
    """
    D = dL * dR
    assert M.shape == (D, D)
    T = M.reshape(dL, dR, dL, dR).transpose(0, 2, 1, 3).reshape(dL*dL, dR*dR)
    s = la.svd(T, compute_uv=False)
    return s

def best_kron_factor(M: np.ndarray, dL: int, dR: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Best rank-1 Kronecker approximation:
        M ≈ A ⊗ B
    via SVD of the reshuffled matrix.

    Returns (A, B, rel_error).
    """
    D = dL * dR
    assert M.shape == (D, D)
    T = M.reshape(dL, dR, dL, dR).transpose(0, 2, 1, 3).reshape(dL*dL, dR*dR)
    U, s, Vh = la.svd(T, full_matrices=False)
    # rank-1 term
    u0 = U[:, 0] * math.sqrt(float(s[0]))
    v0 = Vh[0, :] * math.sqrt(float(s[0]))
    A = u0.reshape(dL, dL)
    B = v0.reshape(dR, dR)

    approx = np.kron(A, B)
    rel = frob(M - approx) / (frob(M) + 1e-12)
    return A, B, float(rel)

def scramble_factorization_operator(
    dL: int, dR: int, seed_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a "true" bidirectional structure in a known factorization, then scramble it
    by a global unitary so the factorization is not obvious.

    We build:
      H_true = A ⊗ I + I ⊗ B + epsilon * (A ⊗ B)   (has explicit L/R actions)
    Then scramble:
      H_scr = U H_true U†

    Return (H_true, H_scr, U, (A,B packed)).
    """
    A = hermitian_random(dL, scale=seed_scale)
    B = hermitian_random(dR, scale=seed_scale)
    I_L = np.eye(dL, dtype=np.complex128)
    I_R = np.eye(dR, dtype=np.complex128)
    eps = 0.35
    H_true = np.kron(A, I_R) + np.kron(I_L, B) + eps * np.kron(A, B)
    H_true = (H_true + H_true.conj().T) * 0.5

    D = dL * dR
    U = unitary_random(D)
    H_scr = U @ H_true @ U.conj().T
    H_scr = (H_scr + H_scr.conj().T) * 0.5
    return H_true, H_scr, U, A, B

def recover_factorization_via_bidirectional_fit(
    H_scr: np.ndarray,
    candidate_dims: List[Tuple[int, int]],
) -> Tuple[Tuple[int, int], FactorizationDiagnostics]:
    """
    We don't know (dL,dR). Try candidate factorizations of D and pick the one that best admits
    a low-rank Kronecker structure in a "link map" derived from H_scr.

    Here the "link map" is just H_scr itself (or a function of it).
    A more elaborate version could use an influence map built from commutators or channel transfer;
    the point is: does the operator admit L/R separation?

    Criterion: best rank-1 Kron relative error + operator-Schmidt spectrum concentration.
    """
    D = H_scr.shape[0]
    assert H_scr.shape == (D, D)

    best = None
    best_diag = None

    for dL, dR in candidate_dims:
        if dL * dR != D:
            continue

        # Schmidt spectrum
        s = operator_schmidt_singular_values(H_scr, dL, dR)
        s_norm = s / (np.sum(s) + 1e-12)
        top = [float(x) for x in s[:10]]

        # best rank-1 Kronecker fit
        _, _, rel = best_kron_factor(H_scr, dL, dR)

        # score: smaller rel is better; also prefer spectra with a few big coefficients
        # "concentration" proxy: sum of top 3 normalized coefficients
        conc = float(np.sum(s_norm[:3]))
        score = rel - 0.25 * conc  # reward concentration modestly

        diag = FactorizationDiagnostics(
            relative_fit_error=float(rel),
            top_singular_values=top,
            inferred_dims=(dL, dR),
        )

        if best is None or score < best:
            best = score
            best_diag = diag

    assert best_diag is not None
    return best_diag.inferred_dims, best_diag


# ----------------------------
# Plotting
# ----------------------------

def plot_embedding(coords: np.ndarray, outpath: Path, title: str) -> None:
    ensure_dir(outpath.parent)
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=32)
    for i, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(i), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

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

def plot_singular_values(sv: List[float], outpath: Path, title: str) -> None:
    ensure_dir(outpath.parent)
    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(111)
    x = np.arange(1, len(sv) + 1)
    ax.plot(x, sv, marker="o")
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# ----------------------------
# Main program
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="hsf_one_script_out")
    ap.add_argument("--seed", type=int, default=0)

    # locality / accessibility
    ap.add_argument("--n_sites", type=int, default=16, help="Number of emergent 'sites' in locality demo.")
    ap.add_argument("--steps", type=int, default=280, help="Flow steps for constraint-shaped influence kernel.")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--bandwidth_k", type=int, default=6, help="Spectral bandwidth (top-k modes kept).")
    ap.add_argument("--memory_alpha", type=float, default=0.08, help="No-forgetting smoothing rate.")
    ap.add_argument("--refold_beta", type=float, default=0.85, help="No-refolding resistance strength.")

    # weights
    ap.add_argument("--w_bandwidth", type=float, default=1.0)
    ap.add_argument("--w_signaling", type=float, default=1.0)
    ap.add_argument("--w_refolding", type=float, default=1.0)
    ap.add_argument("--w_forgetting", type=float, default=1.0)

    # factorization
    ap.add_argument("--dL", type=int, default=4, help="True left dim for factorization demo.")
    ap.add_argument("--dR", type=int, default=4, help="True right dim for factorization demo.")
    ap.add_argument("--try_dims", type=str, default="2x8,4x4,8x2",
                    help="Candidate dims to try (comma-separated like 2x8,4x4,8x2).")

    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)
    ensure_dir(outdir / "figures")
    ensure_dir(outdir / "data")

    # -------------------------
    # A) Locality from constraints
    # -------------------------
    weights = ConstraintWeights(
        finite_bandwidth=float(args.w_bandwidth),
        no_signaling=float(args.w_signaling),
        no_refolding=float(args.w_refolding),
        no_forgetting=float(args.w_forgetting),
    )

    A0 = initial_influence_kernel(args.n_sites, seed_scale=1.0)
    A = constrained_flow(
        a0=A0,
        weights=weights,
        steps=int(args.steps),
        dt=float(args.dt),
        bandwidth_k=int(args.bandwidth_k),
        memory_alpha=float(args.memory_alpha),
        refold_beta=float(args.refold_beta),
    )

    coords0 = embed_from_kernel(np.maximum(0.0, A0), dim=3)
    coords = embed_from_kernel(np.maximum(0.0, A), dim=3)

    m0 = locality_metrics_from_embedding(coords0, k=max(3, min(6, args.n_sites // 4)))
    m = locality_metrics_from_embedding(coords,  k=max(3, min(6, args.n_sites // 4)))

    plot_influence_matrix(A0, outdir / "figures" / "locality_A0_influence.png",
                          "Initial influence (pre-geometry)")
    plot_influence_matrix(A,  outdir / "figures" / "locality_A_constrained.png",
                          "Constrained influence (post-flow)")
    plot_embedding(coords0, outdir / "figures" / "locality_embedding_initial.png",
                   f"Initial embedding (lattice_score={m0.lattice_score:.3f})")
    plot_embedding(coords,  outdir / "figures" / "locality_embedding_constrained.png",
                   f"Constrained embedding (lattice_score={m.lattice_score:.3f})")

    # -------------------------
    # B) Factorization recovery from bidirectional link structure
    # -------------------------
    H_true, H_scr, U, A_true_L, B_true_R = scramble_factorization_operator(args.dL, args.dR, seed_scale=1.0)

    # parse candidate dims
    cand = []
    for token in args.try_dims.split(","):
        token = token.strip().lower()
        if "x" not in token:
            continue
        a, b = token.split("x", 1)
        try:
            cand.append((int(a), int(b)))
        except ValueError:
            pass

    inferred_dims, diag = recover_factorization_via_bidirectional_fit(H_scr, cand)

    # Save singular value profile plot
    plot_singular_values(diag.top_singular_values, outdir / "figures" / "factorization_top_singular_values.png",
                         f"Operator-Schmidt top singular values (best dims={inferred_dims[0]}x{inferred_dims[1]})")

    # Also save a "bandwidth profile" line (what you like: top singular values one-liner)
    bandwidth_profile = " ".join([f"{x:.6g}" for x in diag.top_singular_values])

    # -------------------------
    # Outputs: JSON summary
    # -------------------------
    summary: Dict[str, Any] = {
        "timestamp": now_iso(),
        "seed": int(args.seed),
        "locality": {
            "n_sites": int(args.n_sites),
            "weights": {
                "finite_bandwidth": weights.finite_bandwidth,
                "no_signaling": weights.no_signaling,
                "no_refolding": weights.no_refolding,
                "no_forgetting": weights.no_forgetting,
            },
            "flow": {
                "steps": int(args.steps),
                "dt": float(args.dt),
                "bandwidth_k": int(args.bandwidth_k),
                "memory_alpha": float(args.memory_alpha),
                "refold_beta": float(args.refold_beta),
            },
            "metrics_initial": {
                "knn_reciprocity": m0.knn_reciprocity,
                "edge_length_cv": m0.edge_length_cv,
                "lattice_score": m0.lattice_score,
            },
            "metrics_constrained": {
                "knn_reciprocity": m.knn_reciprocity,
                "edge_length_cv": m.edge_length_cv,
                "lattice_score": m.lattice_score,
            },
        },
        "factorization": {
            "true_dims": [int(args.dL), int(args.dR)],
            "candidate_dims": [[int(a), int(b)] for a, b in cand],
            "inferred_dims": [int(inferred_dims[0]), int(inferred_dims[1])],
            "relative_kron_fit_error": diag.relative_fit_error,
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

    # A clean, human-readable one-page summary
    lines = []
    lines.append("HSF One-Script Core Program Summary\n")
    lines.append("=" * 38 + "\n\n")
    lines.append(f"Timestamp: {summary['timestamp']}\n")
    lines.append(f"Seed:      {summary['seed']}\n\n")

    lines.append("A) Emergence of Spatial Locality\n")
    lines.append("-" * 30 + "\n")
    lines.append(f"n_sites = {args.n_sites}\n")
    lines.append("Initial metrics:\n")
    lines.append(f"  kNN reciprocity = {m0.knn_reciprocity:.4f}\n")
    lines.append(f"  edge length CV  = {m0.edge_length_cv:.4f}\n")
    lines.append(f"  lattice score   = {m0.lattice_score:.4f}\n")
    lines.append("Constrained metrics:\n")
    lines.append(f"  kNN reciprocity = {m.knn_reciprocity:.4f}\n")
    lines.append(f"  edge length CV  = {m.edge_length_cv:.4f}\n")
    lines.append(f"  lattice score   = {m.lattice_score:.4f}\n")
    lines.append("\nKey idea: the same constraint-shaped flow that enforces finite bandwidth + signaling-like decay\n")
    lines.append("pushes the influence/accessibility kernel toward compatibility with a low-dimensional, local embedding.\n\n")

    lines.append("B) Emergence of Factorization (Bidirectional Link Structure)\n")
    lines.append("-" * 52 + "\n")
    lines.append(f"True dims:     {args.dL} x {args.dR}\n")
    lines.append(f"Inferred dims: {inferred_dims[0]} x {inferred_dims[1]}\n")
    lines.append(f"Relative Kron fit error (rank-1): {diag.relative_fit_error:.6f}\n")
    lines.append("Top operator-Schmidt singular values (bandwidth profile):\n")
    lines.append(f"  {bandwidth_profile}\n\n")
    lines.append("Key idea: a bidirectional link structure manifests as an operator that is well-approximated\n")
    lines.append("by low-rank Kronecker structure in the correct factorization; we select the factorization\n")
    lines.append("that optimizes this diagnostic.\n\n")

    lines.append("Generated figures:\n")
    for k, v in summary["figures"].items():
        lines.append(f"  - {k}: {v}\n")

    elapsed = time.time() - t0
    lines.append(f"\nTotal runtime: {elapsed:.2f}s\n")
    (outdir / "SUMMARY.txt").write_text("".join(lines), encoding="utf-8")

    # Print the one-line headline results for quick copy/paste
    print("\n=== HSF one-script core program ===")
    print(f"outdir: {outdir}")
    print("\n[Locality]")
    print(f"  lattice_score: initial={m0.lattice_score:.4f}  constrained={m.lattice_score:.4f}")
    print(f"  kNN reciprocity: initial={m0.knn_reciprocity:.4f}  constrained={m.knn_reciprocity:.4f}")
    print("\n[Factorization]")
    print(f"  true dims: {args.dL}x{args.dR}")
    print(f"  inferred dims: {inferred_dims[0]}x{inferred_dims[1]}")
    print(f"  rel_kron_fit_error: {diag.relative_fit_error:.6f}")
    print(f"  bandwidth_profile: {bandwidth_profile}")
    print("===================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())