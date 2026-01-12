#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hsf_spectral_dimension_study.py
==============================

Graph-only spectral dimension study script.

Fix in this revision:
- Random-regular generator is now *guaranteed* to succeed for even degrees (like 4,6)
  using a circulant d-regular construction + randomized edge swaps.
  This avoids the rejection-sampling failure you hit at N=12.

Outputs (in --out folder):
- manifest.json
- spectral_runs.jsonl
- spectral_summary.json
- REPORT.md
- optionally: out.zip if --zip
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# =============================================================================
# UTILITIES
# =============================================================================

def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


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


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-18)


# =============================================================================
# GRAPH
# =============================================================================

@dataclass
class GraphTopology:
    name: str
    dimension: Optional[int]  # None for non-geometric graphs
    edges: List[Tuple[int, int]]
    N: int
    metadata: Dict = field(default_factory=dict)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def degree_mean(self) -> float:
        return 2.0 * self.num_edges / self.N

    def adjacency_dict(self) -> Dict[int, Set[int]]:
        adj = {i: set() for i in range(self.N)}
        for (i, j) in self.edges:
            adj[i].add(j)
            adj[j].add(i)
        return adj


def make_1d_ring(N: int) -> GraphTopology:
    edges = [(i, (i + 1) % N) for i in range(N)]
    return GraphTopology(
        name=f"1D_ring_{N}",
        dimension=1,
        edges=edges,
        N=N,
        metadata={"boundary": "periodic", "L": N},
    )


def make_1d_chain(N: int) -> GraphTopology:
    edges = [(i, i + 1) for i in range(N - 1)]
    return GraphTopology(
        name=f"1D_chain_{N}",
        dimension=1,
        edges=edges,
        N=N,
        metadata={"boundary": "open", "L": N},
    )


def make_2d_lattice(Lx: int, Ly: int, periodic: bool = True) -> GraphTopology:
    N = Lx * Ly
    edges: List[Tuple[int, int]] = []

    def idx(x, y): return x * Ly + y

    for x in range(Lx):
        for y in range(Ly):
            if periodic or x < Lx - 1:
                edges.append((idx(x, y), idx((x + 1) % Lx, y)))
            if periodic or y < Ly - 1:
                edges.append((idx(x, y), idx(x, (y + 1) % Ly)))

    return GraphTopology(
        name=f"2D_lattice_{Lx}x{Ly}",
        dimension=2,
        edges=edges,
        N=N,
        metadata={"boundary": "periodic" if periodic else "open", "Lx": Lx, "Ly": Ly},
    )


def make_3d_lattice(Lx: int, Ly: int, Lz: int, periodic: bool = True) -> GraphTopology:
    N = Lx * Ly * Lz
    edges: List[Tuple[int, int]] = []

    def idx(x, y, z): return x * Ly * Lz + y * Lz + z

    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                if periodic or x < Lx - 1:
                    edges.append((idx(x, y, z), idx((x + 1) % Lx, y, z)))
                if periodic or y < Ly - 1:
                    edges.append((idx(x, y, z), idx(x, (y + 1) % Ly, z)))
                if periodic or z < Lz - 1:
                    edges.append((idx(x, y, z), idx(x, y, (z + 1) % Lz)))

    return GraphTopology(
        name=f"3D_lattice_{Lx}x{Ly}x{Lz}",
        dimension=3,
        edges=edges,
        N=N,
        metadata={"boundary": "periodic" if periodic else "open", "Lx": Lx, "Ly": Ly, "Lz": Lz},
    )


def make_4d_lattice(L: int, periodic: bool = True) -> GraphTopology:
    N = L ** 4
    edges: List[Tuple[int, int]] = []

    def idx(x, y, z, w): return x * L**3 + y * L**2 + z * L + w

    for x in range(L):
        for y in range(L):
            for z in range(L):
                for w in range(L):
                    if periodic or x < L - 1:
                        edges.append((idx(x, y, z, w), idx((x + 1) % L, y, z, w)))
                    if periodic or y < L - 1:
                        edges.append((idx(x, y, z, w), idx(x, (y + 1) % L, z, w)))
                    if periodic or z < L - 1:
                        edges.append((idx(x, y, z, w), idx(x, y, (z + 1) % L, w)))
                    if periodic or w < L - 1:
                        edges.append((idx(x, y, z, w), idx(x, y, z, (w + 1) % L)))

    return GraphTopology(
        name=f"4D_lattice_{L}x{L}x{L}x{L}",
        dimension=4,
        edges=edges,
        N=N,
        metadata={"boundary": "periodic" if periodic else "open", "L": L},
    )


# =============================================================================
# RANDOM REGULAR (ROBUST)
# =============================================================================

def _edge_set_to_list(edges: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return list(edges)


def _circulant_regular_edges(N: int, degree: int) -> Set[Tuple[int, int]]:
    """
    Guaranteed simple d-regular construction for even degree:
    connect i to i±1..±(d/2) modulo N.
    Requires: degree even, N > degree, N >= 3.
    """
    if degree % 2 != 0:
        raise ValueError("Circulant construction requires even degree.")
    if N <= degree:
        raise ValueError("Need N > degree for a simple d-regular graph.")
    kmax = degree // 2
    edges: Set[Tuple[int, int]] = set()
    for i in range(N):
        for k in range(1, kmax + 1):
            j = (i + k) % N
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
        for k in range(1, kmax + 1):
            j = (i - k) % N
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return edges


def _has_self_or_multi(edges: Set[Tuple[int, int]]) -> bool:
    for (u, v) in edges:
        if u == v:
            return True
    # set implies no multiedges
    return False


def _random_edge_swap(edges: Set[Tuple[int, int]], rng: np.random.Generator, tries: int = 50) -> bool:
    """
    Do one double-edge swap: (a,b),(c,d) -> (a,d),(c,b) with all endpoints distinct and no collisions.
    Returns True if swap executed.
    """
    if len(edges) < 2:
        return False
    edge_list = list(edges)
    m = len(edge_list)
    for _ in range(tries):
        e1 = edge_list[int(rng.integers(0, m))]
        e2 = edge_list[int(rng.integers(0, m))]
        if e1 == e2:
            continue
        a, b = e1
        c, d = e2
        # ensure all distinct endpoints
        if len({a, b, c, d}) < 4:
            continue
        # propose swap
        new1 = (a, d) if a < d else (d, a)
        new2 = (c, b) if c < b else (b, c)
        # avoid loops
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        # avoid creating existing edges
        if new1 in edges or new2 in edges:
            continue
        # execute: remove old, add new
        edges.remove(e1)
        edges.remove(e2)
        edges.add(new1)
        edges.add(new2)
        return True
    return False


def make_random_regular(N: int, degree: int, rng: np.random.Generator, swaps_per_edge: int = 20) -> GraphTopology:
    """
    Robust random d-regular graph generator.

    For even degree (e.g. 4,6), we:
      1) build a guaranteed simple circulant d-regular graph
      2) randomize via many double-edge swaps (preserves degrees, keeps graph simple)

    For odd degree, we fall back to a rejection pairing method (not used in your N=12 rr4/rr6 case).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if degree == 0:
        return GraphTopology(name=f"random_regular_d{degree}", dimension=None, edges=[], N=N,
                             metadata={"degree": degree, "method": "empty"})

    if N <= degree:
        raise ValueError(f"Need N > degree for simple regular graph (N={N}, degree={degree}).")
    if (N * degree) % 2 != 0:
        raise ValueError("N*degree must be even")

    # Even-degree fast path (guaranteed)
    if degree % 2 == 0 and N >= 3:
        edges = _circulant_regular_edges(N, degree)
        # circulant should be simple already
        assert not _has_self_or_multi(edges)
        # randomize
        target_swaps = swaps_per_edge * len(edges)
        swaps = 0
        for _ in range(target_swaps * 3):  # extra attempts for failed swaps
            if swaps >= target_swaps:
                break
            if _random_edge_swap(edges, rng):
                swaps += 1

        return GraphTopology(
            name=f"random_regular_d{degree}",
            dimension=None,
            edges=_edge_set_to_list(edges),
            N=N,
            metadata={"degree": degree, "method": "circulant+swaps", "swaps": swaps, "target_swaps": target_swaps},
        )

    # Odd-degree fallback (rare for your use)
    max_attempts = 20000
    for attempt in range(max_attempts):
        stubs: List[int] = []
        for node in range(N):
            stubs.extend([node] * degree)
        rng.shuffle(stubs)

        edges: Set[Tuple[int, int]] = set()
        ok = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                ok = False
                break
            e = (u, v) if u < v else (v, u)
            if e in edges:
                ok = False
                break
            edges.add(e)

        if ok:
            return GraphTopology(
                name=f"random_regular_d{degree}",
                dimension=None,
                edges=_edge_set_to_list(edges),
                N=N,
                metadata={"degree": degree, "method": "rejection", "attempt": attempt},
            )

    raise RuntimeError(f"Failed to generate random regular graph (N={N}, degree={degree}).")


# =============================================================================
# SHAPE ENUMERATION
# =============================================================================

def factor_pairs(N: int) -> List[Tuple[int, int]]:
    out = []
    for a in range(2, N + 1):
        if N % a == 0:
            b = N // a
            if b >= 2:
                out.append((a, b))
    return out


def factor_triples(N: int) -> List[Tuple[int, int, int]]:
    out = []
    for a in range(2, N + 1):
        if N % a != 0:
            continue
        rem = N // a
        for b in range(2, rem + 1):
            if rem % b == 0:
                c = rem // b
                if c >= 2:
                    out.append((a, b, c))
    uniq = []
    seen = set()
    for (a, b, c) in out:
        key = tuple(sorted((a, b, c)))
        if key not in seen:
            seen.add(key)
            uniq.append((a, b, c))
    return uniq


def fourth_root_int(N: int) -> Optional[int]:
    L = int(round(N ** 0.25))
    if L >= 2 and L**4 == N:
        return L
    return None


def parse_shape(shape: str) -> Tuple[int, ...]:
    parts = shape.lower().replace("×", "x").split("x")
    dims = tuple(int(p.strip()) for p in parts if p.strip())
    if len(dims) < 1:
        raise ValueError(f"Bad shape: {shape}")
    return dims


# =============================================================================
# SPECTRAL DIMENSION CORE
# =============================================================================

def laplacian_eigs(graph: GraphTopology) -> np.ndarray:
    N = graph.N
    adj = graph.adjacency_dict()
    L = np.zeros((N, N), dtype=float)
    for i in range(N):
        di = len(adj[i])
        L[i, i] = di
        for j in adj[i]:
            L[i, j] = -1.0
    return np.linalg.eigvalsh(L)


def heat_return_prob(eigs: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.mean(np.exp(-np.outer(t, eigs)), axis=1)


def spectral_dimension_local_slope(t: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    logt = np.log(t)
    logP = np.log(P + 1e-300)
    dlogP = logP[2:] - logP[:-2]
    dlogt = logt[2:] - logt[:-2]
    slope = dlogP / (dlogt + 1e-300)
    Ds = -2.0 * slope
    return t[1:-1], Ds


def fit_windows(logt: np.ndarray, logP: np.ndarray, windows: List[Tuple[float, float]]) -> List[Dict]:
    n = len(logt)
    fits = []
    for (q0, q1) in windows:
        i0 = int(np.floor(q0 * (n - 1)))
        i1 = int(np.ceil(q1 * (n - 1)))
        i0 = max(0, min(n - 2, i0))
        i1 = max(i0 + 2, min(n, i1))

        x = logt[i0:i1]
        y = logP[i0:i1]
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        r2 = r2_score(y, yhat)
        Ds = -2.0 * float(slope)

        fits.append({
            "q_window": [float(q0), float(q1)],
            "idx_window": [int(i0), int(i1 - 1)],
            "t_window": [float(np.exp(x[0])), float(np.exp(x[-1]))],
            "slope": float(slope),
            "intercept": float(intercept),
            "Ds_fit": float(Ds),
            "r2": float(r2),
            "n_points": int(len(x)),
        })
    return fits


def bootstrap_window_fits(logt: np.ndarray,
                          logP: np.ndarray,
                          base_window: Tuple[float, float],
                          rng: np.random.Generator,
                          B: int = 200) -> Dict:
    n = len(logt)
    q0, q1 = base_window
    i0 = int(np.floor(q0 * (n - 1)))
    i1 = int(np.ceil(q1 * (n - 1)))
    i0 = max(0, min(n - 2, i0))
    i1 = max(i0 + 2, min(n, i1))

    x = logt[i0:i1]
    y = logP[i0:i1]
    m = len(x)

    Ds_samples = []
    for _ in range(B):
        idx = rng.integers(0, m, size=m)
        xb = x[idx]
        yb = y[idx]
        order = np.argsort(xb)
        xb = xb[order]
        yb = yb[order]
        slope, _ = np.polyfit(xb, yb, 1)
        Ds_samples.append(-2.0 * float(slope))

    Ds_samples = np.array(Ds_samples, dtype=float)
    return {
        "window": [float(q0), float(q1)],
        "B": int(B),
        "Ds_mean": float(np.mean(Ds_samples)),
        "Ds_std": float(np.std(Ds_samples)),
        "Ds_p16": float(np.percentile(Ds_samples, 16)),
        "Ds_p50": float(np.percentile(Ds_samples, 50)),
        "Ds_p84": float(np.percentile(Ds_samples, 84)),
    }


def run_one_graph(graph: GraphTopology,
                  t_grid: np.ndarray,
                  fit_windows_q: List[Tuple[float, float]],
                  bootstrap_window_q: Optional[Tuple[float, float]],
                  bootstrap_B: int,
                  rng: np.random.Generator) -> Dict:
    eigs = laplacian_eigs(graph)
    P = heat_return_prob(eigs, t_grid)

    t_mid, Ds_curve = spectral_dimension_local_slope(t_grid, P)
    logt = np.log(t_grid)
    logP = np.log(P + 1e-300)

    fits = fit_windows(logt, logP, fit_windows_q)

    Ds_vals = np.array([f["Ds_fit"] for f in fits], dtype=float)
    r2_vals = np.array([f["r2"] for f in fits], dtype=float)

    boot = None
    if bootstrap_window_q is not None and bootstrap_B > 0:
        boot = bootstrap_window_fits(logt, logP, bootstrap_window_q, rng=rng, B=bootstrap_B)

    return {
        "graph": {
            "name": graph.name,
            "dimension": graph.dimension,
            "N": graph.N,
            "edges": graph.num_edges,
            "degree_mean": graph.degree_mean,
            "metadata": graph.metadata,
        },
        "laplacian": {
            "eigenvalues": eigs.tolist(),
            "lambda_min": float(np.min(eigs)),
            "lambda_max": float(np.max(eigs)),
            "spectral_gap": float(np.sort(eigs)[1] if len(eigs) > 1 else 0.0),
        },
        "heat": {
            "t": t_grid.tolist(),
            "P": P.tolist(),
        },
        "Ds_curve": {
            "t_mid": t_mid.tolist(),
            "Ds": Ds_curve.tolist(),
            "Ds_mid_median": float(np.median(Ds_curve)),
            "Ds_mid_mean": float(np.mean(Ds_curve)),
        },
        "fits": fits,
        "fit_stability": {
            "Ds_fit_mean": float(np.mean(Ds_vals)),
            "Ds_fit_std": float(np.std(Ds_vals)),
            "Ds_fit_min": float(np.min(Ds_vals)),
            "Ds_fit_max": float(np.max(Ds_vals)),
            "r2_mean": float(np.mean(r2_vals)),
            "r2_min": float(np.min(r2_vals)),
        },
        "bootstrap": boot,
    }


# =============================================================================
# BUILD GRAPH SET
# =============================================================================

def build_graphs(N: int,
                 include: Set[str],
                 shapes_2d: Optional[List[Tuple[int, int]]],
                 shapes_3d: Optional[List[Tuple[int, int, int]]],
                 shape_4d: Optional[int],
                 rr_degrees: List[int],
                 rr_repeats: int,
                 rr_seed: int) -> List[GraphTopology]:
    graphs: List[GraphTopology] = []

    if "1d" in include or "1d_ring" in include:
        graphs.append(make_1d_ring(N))
    if "1d_chain" in include:
        graphs.append(make_1d_chain(N))

    if "2d" in include and shapes_2d:
        for (Lx, Ly) in shapes_2d:
            graphs.append(make_2d_lattice(Lx, Ly, periodic=True))

    if "3d" in include and shapes_3d:
        for (Lx, Ly, Lz) in shapes_3d:
            graphs.append(make_3d_lattice(Lx, Ly, Lz, periodic=True))

    if "4d" in include and shape_4d is not None:
        graphs.append(make_4d_lattice(shape_4d, periodic=True))

    if "rr" in include or any(f"rr{d}" in include for d in rr_degrees):
        rng = np.random.default_rng(rr_seed)
        for deg in rr_degrees:
            if "rr" in include or f"rr{deg}" in include:
                for k in range(rr_repeats):
                    g = make_random_regular(N, deg, rng)
                    g.metadata["rr_repeat"] = k
                    graphs.append(g)

    return graphs


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="HSF Spectral Dimension Study (graph-only)")

    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--N", type=int, required=True, help="Number of nodes")

    ap.add_argument(
        "--include",
        default="1d,2d,3d",
        help="Comma-separated graph families: 1d,1d_chain,2d,3d,4d,rr,rr4,rr6",
    )

    ap.add_argument(
        "--shapes",
        default="auto",
        help="Optional explicit lattice shapes. Comma-separated like: 2x6 (2D), 2x2x3 (3D). "
             "Use 'auto' to enumerate all valid shapes for N (>=2 in each axis). "
             "Use 'first' to pick the first valid shape per dimension.",
    )

    ap.add_argument("--t-min", type=float, default=1e-3)
    ap.add_argument("--t-max", type=float, default=1e3)
    ap.add_argument("--t-points", type=int, default=120)

    ap.add_argument(
        "--fit-windows",
        default="0.20-0.50,0.33-0.66,0.50-0.80",
        help="Quantile windows on t-grid for log-log fits, e.g. '0.2-0.5,0.33-0.66'",
    )

    ap.add_argument(
        "--bootstrap-window",
        default="0.33-0.66",
        help="Quantile window for bootstrap uncertainty (or 'none')",
    )
    ap.add_argument("--bootstrap-B", type=int, default=200)

    ap.add_argument("--rr-repeats", type=int, default=10, help="How many random-regular samples per degree")
    ap.add_argument("--rr-seed", type=int, default=0, help="RNG seed for RR generation")

    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--zip", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.out).resolve()
    ensure_dir(outdir)

    include = {s.strip().lower() for s in args.include.split(",") if s.strip()}

    # t-grid
    t_grid = np.logspace(np.log10(args.t_min), np.log10(args.t_max), args.t_points).astype(float)

    # fit windows
    fit_windows_q: List[Tuple[float, float]] = []
    for w in args.fit_windows.split(","):
        w = w.strip()
        if not w:
            continue
        a, b = w.split("-")
        fit_windows_q.append((float(a), float(b)))

    # bootstrap window
    bootstrap_window_q = None
    if args.bootstrap_window.strip().lower() != "none":
        a, b = args.bootstrap_window.strip().split("-")
        bootstrap_window_q = (float(a), float(b))

    # shape selection
    shapes_2d: List[Tuple[int, int]] = []
    shapes_3d: List[Tuple[int, int, int]] = []
    shape_4d: Optional[int] = None

    if args.shapes.strip().lower() == "auto":
        shapes_2d = factor_pairs(args.N) if "2d" in include else []
        shapes_3d = factor_triples(args.N) if "3d" in include else []
        shape_4d = fourth_root_int(args.N) if "4d" in include else None

    elif args.shapes.strip().lower() == "first":
        if "2d" in include:
            pairs = factor_pairs(args.N)
            shapes_2d = [pairs[0]] if pairs else []
        if "3d" in include:
            triples = factor_triples(args.N)
            shapes_3d = [triples[0]] if triples else []
        if "4d" in include:
            shape_4d = fourth_root_int(args.N)

    else:
        for shp in args.shapes.split(","):
            shp = shp.strip()
            if not shp:
                continue
            dims = parse_shape(shp)
            if len(dims) == 2:
                shapes_2d.append((dims[0], dims[1]))
            elif len(dims) == 3:
                shapes_3d.append((dims[0], dims[1], dims[2]))
            elif len(dims) == 4 and len(set(dims)) == 1:
                shape_4d = dims[0]
            else:
                raise ValueError(f"Unsupported shape '{shp}'. Use 2D (axb) or 3D (axbxc) or 4D (LxLxLxL).")

    rr_degrees = []
    if "rr" in include or "rr4" in include:
        rr_degrees.append(4)
    if "rr" in include or "rr6" in include:
        rr_degrees.append(6)

    graphs = build_graphs(
        N=args.N,
        include=include,
        shapes_2d=shapes_2d,
        shapes_3d=shapes_3d,
        shape_4d=shape_4d,
        rr_degrees=rr_degrees,
        rr_repeats=args.rr_repeats,
        rr_seed=args.rr_seed,
    )

    # Write manifest
    manifest = {
        "created_utc": now_utc_iso(),
        "tool": "hsf_spectral_dimension_study.py",
        "args": vars(args),
        "graphs_planned": [
            {"name": g.name, "dimension": g.dimension, "N": g.N, "edges": g.num_edges, "metadata": g.metadata}
            for g in graphs
        ],
    }
    write_text(outdir / "manifest.json", json.dumps(manifest, indent=2))

    runs_path = outdir / "spectral_runs.jsonl"
    if runs_path.exists():
        runs_path.unlink()

    print("=" * 72)
    print("HSF Spectral Dimension Study (graph-only)")
    print("=" * 72)
    print(f"N = {args.N} nodes")
    print(f"t-grid: [{args.t_min:g}, {args.t_max:g}] with {args.t_points} points")
    print(f"Fit windows (quantiles): {fit_windows_q}")
    print(f"Bootstrap window: {bootstrap_window_q} | B={args.bootstrap_B if bootstrap_window_q else 0}")
    print("Graphs:")
    for g in graphs:
        dstr = f"d={g.dimension}" if g.dimension is not None else "non-geometric"
        print(f"  - {g.name:28s} | edges={g.num_edges:4d} | {dstr}")
    print("=" * 72)

    t0 = time.time()
    rng = np.random.default_rng(12345)

    results: List[Dict] = []
    for idx, g in enumerate(graphs, start=1):
        if args.progress:
            print(f"[{idx}/{len(graphs)}] {g.name}")

        rec = {
            "meta": {
                "created_utc": now_utc_iso(),
                "N": args.N,
                "graph_name": g.name,
                "graph_dimension": g.dimension,
                "graph_edges": g.num_edges,
                "graph_degree_mean": g.degree_mean,
                "graph_metadata": g.metadata,
            },
            "spectral": run_one_graph(
                graph=g,
                t_grid=t_grid,
                fit_windows_q=fit_windows_q,
                bootstrap_window_q=bootstrap_window_q,
                bootstrap_B=args.bootstrap_B,
                rng=rng,
            ),
        }
        append_jsonl(runs_path, rec)
        results.append(rec)

    runtime = time.time() - t0

    # Aggregate summary by dimension/name
    by_dim: Dict[str, List[float]] = {}
    by_name: Dict[str, List[float]] = {}

    for rec in results:
        d = rec["meta"]["graph_dimension"]
        key_dim = str(d) if d is not None else "non-geometric"
        Ds_mean = rec["spectral"]["fit_stability"]["Ds_fit_mean"]
        by_dim.setdefault(key_dim, []).append(Ds_mean)
        by_name.setdefault(rec["meta"]["graph_name"], []).append(Ds_mean)

    summary = {
        "created_utc": now_utc_iso(),
        "runtime_sec": float(runtime),
        "total_graphs": int(len(results)),
        "by_dimension": {
            k: {
                "count": int(len(v)),
                "Ds_fit_mean": float(np.mean(v)),
                "Ds_fit_std": float(np.std(v)),
                "Ds_fit_min": float(np.min(v)),
                "Ds_fit_max": float(np.max(v)),
            }
            for k, v in by_dim.items()
        },
        "by_graph_name": {
            k: {
                "count": int(len(v)),
                "Ds_fit_mean": float(np.mean(v)),
                "Ds_fit_std": float(np.std(v)),
            }
            for k, v in by_name.items()
        },
    }
    write_text(outdir / "spectral_summary.json", json.dumps(summary, indent=2))

    # REPORT.md
    lines = []
    lines.append("# HSF Spectral Dimension Study (graph-only)")
    lines.append(f"- Created: `{now_utc_iso()}`")
    lines.append(f"- N = {args.N}")
    lines.append(f"- Total graphs: {len(results)}")
    lines.append(f"- Runtime: {runtime:.2f}s")
    lines.append("")
    lines.append("## Dimension aggregates (mean over fit-window means)")
    lines.append("")
    for dim_key, stats in summary["by_dimension"].items():
        lines.append(f"- **{dim_key}**: n={stats['count']} | "
                     f"Ds={stats['Ds_fit_mean']:.3f} ± {stats['Ds_fit_std']:.3f} "
                     f"(min={stats['Ds_fit_min']:.3f}, max={stats['Ds_fit_max']:.3f})")
    lines.append("")
    lines.append("## Per-graph quick view")
    lines.append("")
    for rec in results:
        gname = rec["meta"]["graph_name"]
        d = rec["meta"]["graph_dimension"]
        Ds_mean = rec["spectral"]["fit_stability"]["Ds_fit_mean"]
        Ds_std = rec["spectral"]["fit_stability"]["Ds_fit_std"]
        r2_mean = rec["spectral"]["fit_stability"]["r2_mean"]
        lines.append(f"- **{gname}** (d={d}): Ds_fit_mean={Ds_mean:.3f}, Ds_fit_std={Ds_std:.3f}, r2_mean={r2_mean:.3f}")
    lines.append("")
    lines.append("## Interpretation tip")
    lines.append("")
    lines.append("For small graphs, spectral dimension is typically *window-sensitive*. "
                 "A good sign of a meaningful estimate is: (i) high R^2 across windows, "
                 "(ii) low Ds_fit_std across windows, and (iii) a local-slope Ds(t) curve "
                 "that plateaus over a decade or more in t. This script makes those failure "
                 "modes visible rather than hiding them.")
    write_text(outdir / "REPORT.md", "\n".join(lines))

    if args.zip:
        z = zip_folder(outdir)
        print(f"Wrote ZIP: {z}")

    print(f"\nDONE. Results in: {outdir}")
    print(f"Open: {outdir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
