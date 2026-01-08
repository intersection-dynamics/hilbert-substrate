#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
HSF Constraint-Separation Refolding Suite (random sparse target)
===============================================================

What you asked for:
- NO required output path argument.
- By default, it writes output into the SAME FOLDER this script lives in.
  (Specifically: ./REFOLD_SEP_SPARSE_<timestamp>/ next to this .py file)

Goal: Constraint separation test between:
  - no-signaling (local gate dynamics; locality-only objective), and
  - no-refolding (preserve existing matter structure while attempting to refold geometry).

We attempt to "refold" a Hamiltonian that is local on a SOURCE geometry A (1D ring)
into being local on a TARGET geometry B (random sparse graph),
using ONLY adjacent 2-qubit local gates (so no-signaling is always respected).

Two regimes per seed:
  1) free:          minimize leak_B(H)                        (no-refolding OFF)
  2) no_refolding:  minimize leak_B(H) w/ anchor constraint   (no-refolding ON)

If (1) succeeds but (2) stalls, no-refolding is independent of no-signaling.

Outputs (streaming):
  <script_dir>/REFOLD_SEP_SPARSE_<timestamp>/
    manifest.json
    baseline.json
    targets.jsonl
    runs/
      runs.jsonl
    summary.json
    REPORT.md

Windows examples (single-line):
  python hsf_refolding_separation_sparse.py --N 8 --seeds 12 --steps 2500 --cost-every 5 --anchor-min 0.85 --progress
  python hsf_refolding_separation_sparse.py --out ".\my_output_folder" --N 8 --seeds 20 --hard-anchor --anchor source_hop --anchor-min 0.90 --progress

Notes:
- "Matter" model: free-fermion hopping Hamiltonian on the ring, built via Jordan–Wigner.
- "Anchor" measures (matter preservation):
    quadratic_fraction: how quadratic (hopping-like) the Hamiltonian remains
    source_hopping_ratio: how much hopping amplitude stays on SOURCE edges
    min_both: min(quadratic_fraction, source_hopping_ratio)

Speed:
- Uses a fast locality proxy (projection onto on-site + 2-body Pauli terms on a given edge set):
    leak ≈ 1 - ||P_local(H)||^2 / ||H||^2
- Avoids full 4^N Pauli enumeration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# -------------------------
# IO helpers (streaming)
# -------------------------

def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def now_stamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()

def zip_folder(folder: Path) -> Path:
    zip_path = Path(str(folder) + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(folder)
                z.write(full, arcname=str(rel))
    return zip_path


# -------------------------
# Pauli + tensor utilities
# -------------------------

I2 = np.array([[1, 0],[0, 1]], dtype=np.complex128)
X  = np.array([[0, 1],[1, 0]], dtype=np.complex128)
Y  = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
Z  = np.array([[1, 0],[0, -1]], dtype=np.complex128)
SIG = {"X": X, "Y": Y, "Z": Z}

def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def op_on_qubit(op: np.ndarray, N: int, q: int) -> np.ndarray:
    mats = [I2] * N
    mats[q] = op
    return kron_all(mats)

def two_site_term(opA: np.ndarray, opB: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    mats = [I2] * N
    mats[i] = opA
    mats[j] = opB
    return kron_all(mats)

def embed_two_qubit_gate_adjacent(U2: np.ndarray, N: int, i: int) -> np.ndarray:
    """
    Embed a 4x4 gate U2 onto adjacent qubits (i, i+1) in an N-qubit Hilbert space.
    """
    assert 0 <= i < N - 1
    out = None
    q = 0
    while q < N:
        if q == i:
            out = U2 if out is None else np.kron(out, U2)
            q += 2
        else:
            out = I2 if out is None else np.kron(out, I2)
            q += 1
    return out


# -------------------------
# Random unitaries (local gates)
# -------------------------

def hermitian_rand(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (a + a.conj().T) / 2.0

def unitary_from_hermitian(h: np.ndarray, t: float) -> np.ndarray:
    w, v = np.linalg.eigh(h)
    ph = np.exp(-1j * t * w)
    return (v * ph) @ v.conj().T


# -------------------------
# Graphs / geometries
# -------------------------

def ring_edges(N: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % N) for i in range(N)]

def edge_set_undirected(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    s = set()
    for a, b in edges:
        if a == b:
            continue
        i, j = (a, b) if a < b else (b, a)
        s.add((i, j))
    return sorted(s)

def random_sparse_edges(N: int, M: int, rng: np.random.Generator, force_connected: bool = True) -> List[Tuple[int,int]]:
    """
    Random undirected graph with M edges. Optionally force connected by seeding a random spanning tree first.
    """
    if force_connected and N > 1 and M < (N - 1):
        raise ValueError(f"M={M} too small to force connectivity on N={N} (need at least N-1).")

    edges = set()

    if force_connected and N > 1:
        perm = rng.permutation(N).tolist()
        for i in range(1, N):
            a = perm[i]
            b = perm[int(rng.integers(0, i))]
            u, v = (a, b) if a < b else (b, a)
            edges.add((u, v))

    attempts = 0
    max_attempts = 50 * max(1, M)
    while len(edges) < M and attempts < max_attempts:
        a = int(rng.integers(0, N))
        b = int(rng.integers(0, N))
        attempts += 1
        if a == b:
            continue
        u, v = (a, b) if a < b else (b, a)
        edges.add((u, v))

    return sorted(edges)

def degree_stats(N: int, edges: List[Tuple[int,int]]) -> Dict[str, float]:
    deg = np.zeros(N, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return {
        "min": float(np.min(deg)) if N else 0.0,
        "max": float(np.max(deg)) if N else 0.0,
        "mean": float(np.mean(deg)) if N else 0.0,
        "std": float(np.std(deg)) if N else 0.0,
    }


# -------------------------
# Locality proxy (fast)
# -------------------------

@dataclass
class LocalBasis:
    ops: List[np.ndarray]
    tags: List[str]

def build_local_basis_from_edges(N: int, edges: List[Tuple[int,int]]) -> LocalBasis:
    """
    Local basis = onsite {X,Y,Z} on each qubit + 2-body Pauli products on each edge.
    Basis size ~ 3N + 9|E|.
    """
    ops: List[np.ndarray] = []
    tags: List[str] = []

    for i in range(N):
        for lab, op in SIG.items():
            ops.append(op_on_qubit(op, N, i))
            tags.append(f"{lab}_{i}")

    elist = edge_set_undirected(edges)
    for (i, j) in elist:
        for a in ("X", "Y", "Z"):
            for b in ("X", "Y", "Z"):
                ops.append(two_site_term(SIG[a], SIG[b], N, i, j))
                tags.append(f"{a}{b}_{i}-{j}")

    return LocalBasis(ops=ops, tags=tags)

def frob2(H: np.ndarray) -> float:
    return float(np.vdot(H, H).real)

def proj_local_norm2(H: np.ndarray, basis: LocalBasis, N: int) -> float:
    """
    Uses orthonormality up to scale: sum |Tr(P^† H)|^2 / d
    """
    d = 2 ** N
    acc = 0.0
    for P in basis.ops:
        t = np.trace(P.conj().T @ H)
        acc += float((t.conjugate() * t).real)
    return acc / float(d)

def locality_proxy(H: np.ndarray, basis: LocalBasis, N: int) -> Tuple[float, float, float]:
    total = frob2(H)
    local = proj_local_norm2(H, basis, N)
    local_frac = float(local / (total + 1e-18))
    leak_frac = float(1.0 - local_frac)
    return leak_frac, local_frac, total


# -------------------------
# Jordan–Wigner fermions (matter model)
# -------------------------

def jordan_wigner_ops(N: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns (c_create, c_destroy) for N sites under standard ordering 0..N-1.
    """
    b_create  = np.array([[0,0],[1,0]], dtype=np.complex128)   # |1><0|
    b_destroy = np.array([[0,1],[0,0]], dtype=np.complex128)   # |0><1|

    c_create: List[np.ndarray] = []
    c_destroy: List[np.ndarray] = []

    for j in range(N):
        ops_c: List[np.ndarray] = []
        ops_d: List[np.ndarray] = []
        for m in range(N):
            if m < j:
                ops_c.append(Z); ops_d.append(Z)
            elif m == j:
                ops_c.append(b_create); ops_d.append(b_destroy)
            else:
                ops_c.append(I2); ops_d.append(I2)
        c_create.append(kron_all(ops_c))
        c_destroy.append(kron_all(ops_d))

    return c_create, c_destroy

def free_fermion_hopping_H(N: int, edges: List[Tuple[int,int]], t_hop: float) -> np.ndarray:
    """
    H = -t Σ_{<i,j>} (c†_i c_j + c†_j c_i)
    """
    c_create, c_destroy = jordan_wigner_ops(N)
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)
    for (i, j) in edge_set_undirected(edges):
        H -= float(t_hop) * (c_create[i] @ c_destroy[j] + c_create[j] @ c_destroy[i])
    H = (H + H.conj().T) / 2.0
    return H

def hopping_matrix_from_H(H: np.ndarray, N: int) -> np.ndarray:
    """
    t_ij = Tr(H c†_i c_j)/d
    """
    c_create, c_destroy = jordan_wigner_ops(N)
    d = 2 ** N
    tmat = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            op = c_create[i] @ c_destroy[j]
            tmat[i, j] = np.trace(H @ op) / d
    return tmat

def matter_anchor_metrics(H: np.ndarray, N: int, source_edges: List[Tuple[int,int]]) -> Dict[str, float]:
    """
    Matter anchor (cheap, operational):
      - quadratic_fraction: how well H is approximated by quadratic hopping terms c†_i c_j
      - source_hopping_ratio: fraction of hopping amplitude on SOURCE edge set
    """
    tmat = hopping_matrix_from_H(H, N)

    # reconstruct quadratic Hamiltonian
    c_create, c_destroy = jordan_wigner_ops(N)
    Hq = np.zeros_like(H)
    for i in range(N):
        for j in range(N):
            tij = tmat[i, j]
            if abs(tij) > 0:
                Hq += tij * (c_create[i] @ c_destroy[j])
    Hq = (Hq + Hq.conj().T) / 2.0

    nH = float(np.linalg.norm(H))
    res = H - Hq
    quadratic_fraction = float(1.0 - (float(np.linalg.norm(res)) / (nH + 1e-12)))

    src = set(edge_set_undirected(source_edges))
    on_src = 0.0
    off_src = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            amp = float(abs(tmat[i, j]))
            if (i, j) in src:
                on_src += amp
            else:
                off_src += amp
    source_hopping_ratio = float(on_src / (on_src + off_src + 1e-12))

    return {
        "quadratic_fraction": quadratic_fraction,
        "source_hopping_ratio": source_hopping_ratio,
    }

def anchor_value(anchor_kind: str, a: Dict[str, float]) -> float:
    if anchor_kind == "quadratic":
        return float(a["quadratic_fraction"])
    if anchor_kind == "source_hop":
        return float(a["source_hopping_ratio"])
    if anchor_kind == "min_both":
        return float(min(a["quadratic_fraction"], a["source_hopping_ratio"]))
    raise ValueError(f"Unknown anchor kind: {anchor_kind}")


# -------------------------
# Refolding optimizer (Metropolis local conjugations)
# -------------------------

@dataclass
class FlowParams:
    steps: int
    eps: float
    temp0: float
    temp_decay: float

@dataclass
class RefoldDiag:
    best_cost: float
    best_leakB: float
    best_anchor: float
    final_cost: float
    final_leakB: float
    final_anchor: float
    accepted: int
    evaluated: int
    rejected_by_anchor: int

def refold_flow(
    H_start: np.ndarray,
    N: int,
    rng: np.random.Generator,
    params: FlowParams,
    basis_B: LocalBasis,
    source_edges: List[Tuple[int,int]],
    anchor_kind: str,
    anchor_min: float,
    mode: str,            # "free" or "no_refolding"
    hard_anchor: bool,
    lam: float,
    cost_every: int,
    anchor_every: int,
) -> Tuple[np.ndarray, RefoldDiag]:
    """
    Minimize leak_B(H) by local adjacent 2-qubit conjugations.
    If mode == "no_refolding", enforce an anchor constraint to preserve "matter".

    hard_anchor=True: reject steps that drop anchor below anchor_min
    hard_anchor=False: add penalty lam * max(0, anchor_min - anchor)^2
    """
    H = H_start.copy()
    bestH = H.copy()

    leakB, _, _ = locality_proxy(H, basis_B, N)
    anch = anchor_value(anchor_kind, matter_anchor_metrics(H, N, source_edges))
    temp = float(params.temp0)

    def cost_fn(leak: float, anchor: float) -> float:
        if mode != "no_refolding":
            return float(leak)
        if hard_anchor:
            return float(leak)  # rejection handles constraint
        deficit = max(0.0, float(anchor_min) - float(anchor))
        return float(leak + float(lam) * deficit * deficit)

    cost = cost_fn(leakB, anch)
    best_cost = float(cost)
    best_leakB = float(leakB)
    best_anchor = float(anch)

    accepted = 0
    evaluated = 0
    rejected_by_anchor = 0

    # cached anchor (update on schedule)
    anchor_cached = float(anch)

    for step in range(int(params.steps)):
        i = int(rng.integers(0, N - 1))
        U2 = unitary_from_hermitian(hermitian_rand(4, rng), t=float(params.eps))
        G = embed_two_qubit_gate_adjacent(U2, N, i)
        Hn = G @ H @ G.conj().T
        Hn = (Hn + Hn.conj().T) / 2.0

        # leak evaluation schedule
        if (step % max(1, int(cost_every))) == 0:
            evaluated += 1
            leakB_n, _, _ = locality_proxy(Hn, basis_B, N)
        else:
            leakB_n = leakB

        # anchor evaluation schedule
        anchor_n = anchor_cached
        if mode == "no_refolding":
            need_anchor = hard_anchor or ((step % max(1, int(anchor_every))) == 0) or ((step % max(1, int(cost_every))) == 0)
            if need_anchor:
                anchor_n = anchor_value(anchor_kind, matter_anchor_metrics(Hn, N, source_edges))

        # hard anchor rejection
        if mode == "no_refolding" and hard_anchor and (anchor_n < float(anchor_min)):
            rejected_by_anchor += 1
            temp *= float(params.temp_decay)
            continue

        cn = cost_fn(float(leakB_n), float(anchor_n))

        # Metropolis accept
        accept = False
        if cn <= cost:
            accept = True
        else:
            if temp > 0:
                p = math.exp(-(cn - cost) / max(1e-12, temp))
                if float(rng.random()) < p:
                    accept = True

        if accept:
            accepted += 1
            H = Hn
            leakB = float(leakB_n)
            anchor_cached = float(anchor_n)
            anch = float(anchor_n)
            cost = float(cn)

            if cost < best_cost:
                best_cost = float(cost)
                bestH = H.copy()
                best_leakB = float(leakB)
                best_anchor = float(anch)

        temp *= float(params.temp_decay)

    # final metrics
    leakB_f, _, _ = locality_proxy(H, basis_B, N)
    anch_f = anchor_value(anchor_kind, matter_anchor_metrics(H, N, source_edges))
    cost_f = cost_fn(float(leakB_f), float(anch_f))

    diag = RefoldDiag(
        best_cost=float(best_cost),
        best_leakB=float(best_leakB),
        best_anchor=float(best_anchor),
        final_cost=float(cost_f),
        final_leakB=float(leakB_f),
        final_anchor=float(anch_f),
        accepted=int(accepted),
        evaluated=int(evaluated),
        rejected_by_anchor=int(rejected_by_anchor),
    )
    return bestH, diag


# -------------------------
# Summary aggregation
# -------------------------

def checkpoint_summary(outdir: Path, rows: List[dict]) -> None:
    leakB_best_free = []
    leakB_best_con = []
    barrier = []
    for r in rows:
        try:
            bfree = float(r["results"]["free"]["best_leakB"])
            bcon  = float(r["results"]["no_refolding"]["best_leakB"])
            leakB_best_free.append(bfree)
            leakB_best_con.append(bcon)
            barrier.append(bcon - bfree)
        except Exception:
            pass

    summ = {
        "created_utc": now_utc_iso(),
        "runs": len(rows),
        "best_leakB_free": {
            "mean": float(np.mean(leakB_best_free)) if leakB_best_free else None,
            "median": float(np.median(leakB_best_free)) if leakB_best_free else None,
        },
        "best_leakB_no_refolding": {
            "mean": float(np.mean(leakB_best_con)) if leakB_best_con else None,
            "median": float(np.median(leakB_best_con)) if leakB_best_con else None,
        },
        "barrier_no_refolding_minus_free": {
            "mean": float(np.mean(barrier)) if barrier else None,
            "median": float(np.median(barrier)) if barrier else None,
        },
    }
    write_text(outdir / "summary.json", json.dumps(summ, indent=2))


# -------------------------
# Output directory selection
# -------------------------

def default_outdir() -> Path:
    """
    Default output directory: sibling of this script file.
    <script_dir>/REFOLD_SEP_SPARSE_<timestamp>/
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir / f"REFOLD_SEP_SPARSE_{now_stamp()}"


# -------------------------
# Main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Optional output folder. Default: next to this script.")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=0)

    # Source (A) geometry is a ring on N
    ap.add_argument("--t-hop", type=float, default=1.0, help="hopping strength for source matter Hamiltonian")

    # Target (B) random sparse
    ap.add_argument("--M", type=int, default=12, help="number of edges in target random sparse graph")
    ap.add_argument("--force-connected", action="store_true", help="seed a spanning tree first")
    ap.set_defaults(force_connected=True)

    # Flow knobs
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--temp0", type=float, default=0.02)
    ap.add_argument("--temp-decay", type=float, default=0.9995)
    ap.add_argument("--cost-every", type=int, default=5)
    ap.add_argument("--anchor-every", type=int, default=5)

    # No-refolding constraint knobs
    ap.add_argument("--anchor", choices=["quadratic", "source_hop", "min_both"], default="min_both")
    ap.add_argument("--anchor-min", type=float, default=0.85, help="minimum anchor value enforced under no-refolding")
    ap.add_argument("--hard-anchor", action="store_true", help="hard-reject violating steps instead of soft penalty")
    ap.add_argument("--lambda", dest="lam", type=float, default=10.0, help="soft-penalty weight (ignored if --hard-anchor)")

    # Misc
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    N = int(args.N)
    if N < 2:
        raise SystemExit("N must be >= 2.")

    outdir = Path(args.out).resolve() if args.out else default_outdir()
    ensure_dir(outdir)
    ensure_dir(outdir / "runs")

    # Write manifest
    write_text(outdir / "manifest.json", json.dumps({
        "created_utc": now_utc_iso(),
        "tool": Path(__file__).name,
        "outdir": str(outdir),
        "args": vars(args),
    }, indent=2))

    # Source geometry A
    edges_A = ring_edges(N)
    basis_A = build_local_basis_from_edges(N, edges_A)

    # Baseline source matter Hamiltonian (pre-geometric "matter" anchored to A)
    H0 = free_fermion_hopping_H(N, edges_A, t_hop=float(args.t_hop))

    leakA0, localA0, norm0 = locality_proxy(H0, basis_A, N)
    anch0 = matter_anchor_metrics(H0, N, edges_A)

    write_text(outdir / "baseline.json", json.dumps({
        "created_utc": now_utc_iso(),
        "N": N,
        "source_geometry": "ring",
        "source_edges": edge_set_undirected(edges_A),
        "baseline_locality_A": {"leakA": leakA0, "local_fracA": localA0, "norm2": norm0},
        "baseline_matter_anchor": anch0,
    }, indent=2))

    # Streaming files
    targets_jsonl = outdir / "targets.jsonl"
    runs_jsonl = outdir / "runs" / "runs.jsonl"
    if targets_jsonl.exists():
        targets_jsonl.unlink()
    if runs_jsonl.exists():
        runs_jsonl.unlink()

    rows: List[dict] = []
    t_start = time.time()

    for s in range(int(args.seed_start), int(args.seed_start) + int(args.seeds)):
        rng = np.random.default_rng(int(s))

        # Build target geometry B (random sparse)
        edges_B = random_sparse_edges(N, int(args.M), rng, force_connected=bool(args.force_connected))
        basis_B = build_local_basis_from_edges(N, edges_B)

        append_jsonl(targets_jsonl, {
            "created_utc": now_utc_iso(),
            "seed": int(s),
            "N": N,
            "M": int(args.M),
            "force_connected": bool(args.force_connected),
            "edges_B": edges_B,
            "deg_stats": degree_stats(N, edges_B),
        })

        # How nonlocal H0 looks on target geometry B
        leakB0, localB0, _ = locality_proxy(H0, basis_B, N)

        if args.progress:
            print(
                f"[seed {s}] out={outdir.name}  M={len(edges_B)}  "
                f"leakB0={leakB0:.4f} localB0={localB0:.4f}  "
                f"anchor0={anchor_value(args.anchor, anch0):.4f}"
            )

        fp = FlowParams(
            steps=int(args.steps),
            eps=float(args.eps),
            temp0=float(args.temp0),
            temp_decay=float(args.temp_decay),
        )

        # Run 1: free refolding (no-refolding OFF)
        _, diag_free = refold_flow(
            H_start=H0,
            N=N,
            rng=rng,
            params=fp,
            basis_B=basis_B,
            source_edges=edges_A,
            anchor_kind=str(args.anchor),
            anchor_min=float(args.anchor_min),
            mode="free",
            hard_anchor=bool(args.hard_anchor),
            lam=float(args.lam),
            cost_every=int(args.cost_every),
            anchor_every=int(args.anchor_every),
        )

        # Run 2: constrained refolding (no-refolding ON)
        rng2 = np.random.default_rng(int(s) + 10_000_000)
        _, diag_con = refold_flow(
            H_start=H0,
            N=N,
            rng=rng2,
            params=fp,
            basis_B=basis_B,
            source_edges=edges_A,
            anchor_kind=str(args.anchor),
            anchor_min=float(args.anchor_min),
            mode="no_refolding",
            hard_anchor=bool(args.hard_anchor),
            lam=float(args.lam),
            cost_every=int(args.cost_every),
            anchor_every=int(args.anchor_every),
        )

        row = {
            "meta": {
                "created_utc": now_utc_iso(),
                "seed": int(s),
                "N": N,
                "source_geometry": "ring",
                "target_geometry": "random_sparse",
                "M": int(args.M),
                "force_connected": bool(args.force_connected),
                "flow": {
                    "steps": int(args.steps),
                    "eps": float(args.eps),
                    "temp0": float(args.temp0),
                    "temp_decay": float(args.temp_decay),
                    "cost_every": int(args.cost_every),
                    "anchor_every": int(args.anchor_every),
                },
                "no_refolding": {
                    "anchor": str(args.anchor),
                    "anchor_min": float(args.anchor_min),
                    "hard_anchor": bool(args.hard_anchor),
                    "lambda": float(args.lam),
                },
            },
            "target": {
                "edges_B": edges_B,
                "deg_stats": degree_stats(N, edges_B),
            },
            "initial": {
                "leakA0": float(leakA0),
                "localA0": float(localA0),
                "leakB0": float(leakB0),
                "localB0": float(localB0),
                "anchor0": {k: float(v) for k, v in anch0.items()},
                "anchor0_value": float(anchor_value(args.anchor, anch0)),
            },
            "results": {
                "free": {
                    "best_cost": float(diag_free.best_cost),
                    "best_leakB": float(diag_free.best_leakB),
                    "best_anchor": float(diag_free.best_anchor),
                    "final_cost": float(diag_free.final_cost),
                    "final_leakB": float(diag_free.final_leakB),
                    "final_anchor": float(diag_free.final_anchor),
                    "accepted": int(diag_free.accepted),
                    "evaluated": int(diag_free.evaluated),
                    "rejected_by_anchor": int(diag_free.rejected_by_anchor),
                },
                "no_refolding": {
                    "best_cost": float(diag_con.best_cost),
                    "best_leakB": float(diag_con.best_leakB),
                    "best_anchor": float(diag_con.best_anchor),
                    "final_cost": float(diag_con.final_cost),
                    "final_leakB": float(diag_con.final_leakB),
                    "final_anchor": float(diag_con.final_anchor),
                    "accepted": int(diag_con.accepted),
                    "evaluated": int(diag_con.evaluated),
                    "rejected_by_anchor": int(diag_con.rejected_by_anchor),
                },
                "barrier": {
                    "best_leakB_no_refolding_minus_free": float(diag_con.best_leakB - diag_free.best_leakB),
                    "final_leakB_no_refolding_minus_free": float(diag_con.final_leakB - diag_free.final_leakB),
                }
            },
        }

        rows.append(row)
        append_jsonl(runs_jsonl, row)

        if (len(rows) % max(1, int(args.checkpoint_every))) == 0:
            checkpoint_summary(outdir, rows)

    checkpoint_summary(outdir, rows)
    runtime = float(time.time() - t_start)

    report_lines = [
        "# HSF Refolding Constraint-Separation Report (random sparse target)",
        f"- Created: `{now_utc_iso()}`",
        f"- Outdir: `{outdir}`",
        f"- N={N}  seeds={int(args.seeds)}  target_edges_M={int(args.M)}  force_connected={bool(args.force_connected)}",
        f"- Flow: steps={int(args.steps)} eps={float(args.eps)} temp0={float(args.temp0)} temp_decay={float(args.temp_decay)} cost_every={int(args.cost_every)} anchor_every={int(args.anchor_every)}",
        f"- No-refolding: anchor={str(args.anchor)} anchor_min={float(args.anchor_min)} hard_anchor={bool(args.hard_anchor)} lambda={float(args.lam)}",
        f"- runtime_sec: {runtime:.2f}",
        "",
        "## What to check",
        "Open `runs/runs.jsonl` and compare per seed:",
        "- initial.leakB0 (how nonlocal H0 looks on the sparse target geometry)",
        "- results.free.best_leakB (how far refolding goes with locality-only)",
        "- results.no_refolding.best_leakB (how far refolding goes while preserving matter)",
        "- results.barrier.best_leakB_no_refolding_minus_free (positive => no-refolding blocks refolding)",
        "",
        "## Key separation criterion",
        "If free refolding reduces leakB substantially but constrained refolding cannot,",
        "then no-refolding is separated from no-signaling (both runs use only local adjacent gates).",
        "",
        "## Files",
        "- baseline.json",
        "- targets.jsonl",
        "- runs/runs.jsonl",
        "- summary.json",
        "- manifest.json",
    ]
    write_text(outdir / "REPORT.md", "\n".join(report_lines))

    if args.zip:
        z = zip_folder(outdir)
        print("Wrote ZIP:", z)

    print("DONE. Open:", outdir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
