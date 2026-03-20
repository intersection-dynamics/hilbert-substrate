#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import expm_multiply

def gell_mann() -> List[np.ndarray]:
    i = 1j
    out = []
    out.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex))
    out.append(np.array([[0, -i, 0], [i, 0, 0], [0, 0, 0]], dtype=complex))
    out.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex))
    out.append(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex))
    out.append(np.array([[0, 0, -i], [0, 0, 0], [i, 0, 0]], dtype=complex))
    out.append(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex))
    out.append(np.array([[0, 0, 0], [0, 0, -i], [0, i, 0]], dtype=complex))
    out.append((1.0 / np.sqrt(3.0)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex))
    return out

GM = [csr_matrix(x) for x in gell_mann()]
I3 = identity(3, dtype=complex, format="csr")
BASIS0 = np.array([1.0, 0.0, 0.0], dtype=complex)

def kron_all_dense(vs: List[np.ndarray]) -> np.ndarray:
    out = vs[0]
    for v in vs[1:]:
        out = np.kron(out, v)
    return out

def kron_all_sparse(ops: List[csr_matrix]) -> csr_matrix:
    out = ops[0]
    for op in ops[1:]:
        out = kron(out, op, format="csr")
    return out

def normalize_state(psi: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(psi)
    if n <= 1e-15:
        raise ValueError("State norm vanished.")
    return psi / n

def pure_density(psi: np.ndarray) -> np.ndarray:
    v = psi.reshape(-1, 1)
    return v @ v.conj().T

def expect_sparse(psi: np.ndarray, op: csr_matrix) -> complex:
    return np.vdot(psi, op @ psi)

def partial_trace_keep(rho: np.ndarray, dims: List[int], keep: List[int]) -> np.ndarray:
    n = len(dims)
    keep = sorted(keep)
    trace_out = [i for i in range(n) if i not in keep]
    reshaped = rho.reshape(dims + dims)
    current_n = n
    for ax in sorted(trace_out, reverse=True):
        reshaped = np.trace(reshaped, axis1=ax, axis2=ax + current_n)
        current_n -= 1
    kept_dims = [dims[i] for i in keep]
    out_dim = int(np.prod(kept_dims)) if kept_dims else 1
    return reshaped.reshape(out_dim, out_dim)

def von_neumann_entropy(rho: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    vals = np.real(vals)
    vals[vals < 0.0] = 0.0
    s = vals.sum()
    if s <= 1e-15:
        return 0.0
    vals = vals / s
    nz = vals[vals > 1e-15]
    return float(-np.sum(nz * np.log(nz)))

def mutual_information(rho_ab: np.ndarray, rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    return float(von_neumann_entropy(rho_a) + von_neumann_entropy(rho_b) - von_neumann_entropy(rho_ab))

def build_site_generator_cache(n_max: int) -> Dict[Tuple[int, int], csr_matrix]:
    cache: Dict[Tuple[int, int], csr_matrix] = {}
    for site in range(n_max):
        for a, lam in enumerate(GM):
            ops = [I3] * n_max
            ops[site] = lam
            cache[(site, a)] = kron_all_sparse(ops)
    return cache

def build_hamiltonian(n_max: int, active_nodes: List[int], active_edges: List[Tuple[int, int]], cache: Dict[Tuple[int, int], csr_matrix], local_coeffs: Dict[int, np.ndarray], edge_strengths: Dict[Tuple[int, int], float]) -> csr_matrix:
    dim = 3 ** n_max
    H = csr_matrix((dim, dim), dtype=complex)
    active_set = set(active_nodes)
    for i in active_nodes:
        coeffs = local_coeffs[i]
        for a in range(8):
            H = H + float(coeffs[a]) * cache[(i, a)]
    for (i, j) in active_edges:
        if i in active_set and j in active_set:
            g = float(edge_strengths[(min(i, j), max(i, j))])
            for a in range(8):
                H = H + g * (cache[(i, a)] @ cache[(j, a)])
    return H.tocsr()

def initial_state_qutrits(n_max: int, n_init: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    local_states = []
    for i in range(n_max):
        if i < n_init:
            z = rng.normal(size=3) + 1j * rng.normal(size=3)
            z = z / np.linalg.norm(z)
            local_states.append(z)
        else:
            local_states.append(BASIS0.copy())
    return normalize_state(kron_all_dense(local_states))

def pair_su3_correlator_strength(psi: np.ndarray, cache: Dict[Tuple[int, int], csr_matrix], i: int, j: int) -> float:
    vals = []
    for a in range(8):
        vals.append(float(np.real(expect_sparse(psi, cache[(i, a)] @ cache[(j, a)]))))
    return float(np.linalg.norm(vals))

def maybe_spawn_subsystem(psi: np.ndarray, rho: np.ndarray, active_nodes: List[int], active_edges: List[Tuple[int, int]], dormant_nodes: List[int], dims: List[int], cache: Dict[Tuple[int, int], csr_matrix], edge_strengths: Dict[Tuple[int, int], float], local_coeffs: Dict[int, np.ndarray], spawn_pair_scale: float, mi_threshold: float, corr_threshold: float, rng: np.random.Generator) -> Optional[Dict[str, object]]:
    if not dormant_nodes:
        return None
    existing_edges = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    candidates = []
    for i, j in combinations(active_nodes, 2):
        if (min(i, j), max(i, j)) not in existing_edges:
            continue
        rho_ab = partial_trace_keep(rho, dims, [i, j])
        rho_a = partial_trace_keep(rho, dims, [i])
        rho_b = partial_trace_keep(rho, dims, [j])
        mi = mutual_information(rho_ab, rho_a, rho_b)
        corr = pair_su3_correlator_strength(psi, cache, i, j)
        candidates.append((mi * corr, i, j, mi, corr))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, i, j, mi, corr = candidates[0]
    if mi < mi_threshold or corr < corr_threshold:
        return None
    new_node = dormant_nodes.pop(0)
    active_nodes.append(new_node)
    active_nodes.sort()
    e1 = (min(i, new_node), max(i, new_node))
    e2 = (min(j, new_node), max(j, new_node))
    active_edges.append(e1)
    active_edges.append(e2)
    active_edges.sort()
    edge_strengths[e1] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
    edge_strengths[e2] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
    local_coeffs[new_node] = rng.uniform(-0.5, 0.5, size=8)
    return {"parents": [i, j], "new_node": new_node, "trigger_mi": float(mi), "trigger_corr": float(corr), "spawn_links": [list(e1), list(e2)]}

def weighted_adjacency(active_nodes: List[int], active_edges: List[Tuple[int, int]], edge_strengths: Dict[Tuple[int, int], float]) -> Tuple[np.ndarray, Dict[int, int]]:
    idx_of = {node: k for k, node in enumerate(active_nodes)}
    n = len(active_nodes)
    W = np.zeros((n, n), dtype=float)
    for i, j in active_edges:
        if i in idx_of and j in idx_of:
            a, b = idx_of[i], idx_of[j]
            w = float(edge_strengths[(min(i, j), max(i, j))])
            W[a, b] = w
            W[b, a] = w
    return W, idx_of

def spectral_1d_embedding(active_nodes: List[int], active_edges: List[Tuple[int, int]], edge_strengths: Dict[Tuple[int, int], float]) -> Dict[int, float]:
    if len(active_nodes) == 1:
        return {active_nodes[0]: 0.0}
    W, idx_of = weighted_adjacency(active_nodes, active_edges, edge_strengths)
    deg = np.sum(W, axis=1)
    L = np.diag(deg) - W
    vals, vecs = np.linalg.eigh(L)
    xs = np.real(vecs[:, 1]) if len(vals) >= 2 else np.zeros(len(active_nodes), dtype=float)
    xs = xs - np.mean(xs)
    std = np.std(xs)
    if std > 1e-12:
        xs = xs / std
    return {node: float(xs[idx_of[node]]) for node in active_nodes}

def align_embedding(new_coords: Dict[int, float], old_coords: Optional[Dict[int, float]]) -> Dict[int, float]:
    if not old_coords:
        return new_coords
    common = [k for k in new_coords.keys() if k in old_coords]
    if len(common) < 2:
        return new_coords
    new_v = np.array([new_coords[k] for k in common], dtype=float)
    old_v = np.array([old_coords[k] for k in common], dtype=float)
    s1 = np.sum((new_v - old_v) ** 2)
    s2 = np.sum((-new_v - old_v) ** 2)
    if s2 < s1:
        return {k: -v for k, v in new_coords.items()}
    return new_coords

def metric_observables(active_nodes: List[int], active_edges: List[Tuple[int, int]], edge_strengths: Dict[Tuple[int, int], float], prev_coords: Optional[Dict[int, float]]) -> Dict[str, object]:
    coords = spectral_1d_embedding(active_nodes, active_edges, edge_strengths)
    coords = align_embedding(coords, prev_coords)
    edge_lengths = [abs(coords[i] - coords[j]) for (i, j) in active_edges if i in coords and j in coords]
    vals = list(coords.values())
    extent = float(max(vals) - min(vals)) if vals else 0.0
    insertion_strain = 0.0
    if prev_coords:
        common = [k for k in coords.keys() if k in prev_coords]
        insertion_strain = float(sum(abs(coords[k] - prev_coords[k]) for k in common))
    return {
        "coords": {str(k): float(v) for k, v in coords.items()},
        "mean_edge_length": float(np.mean(edge_lengths)) if edge_lengths else 0.0,
        "total_edge_length": float(np.sum(edge_lengths)) if edge_lengths else 0.0,
        "metric_extent": extent,
        "insertion_strain": insertion_strain,
    }

@dataclass
class SimConfig:
    n_max: int
    n_init: int
    seed: int
    local_scale: float
    pair_scale: float
    spawn_pair_scale: float
    total_steps: int
    dt: float
    spawn_every: int
    snapshot_every: int
    spawn_mi_threshold: float
    spawn_corr_threshold: float
    max_spawns: int
    json_out: str = ""

def run_sim(cfg: SimConfig) -> Dict[str, object]:
    if cfg.n_init >= cfg.n_max:
        raise ValueError("Require n_init < n_max.")
    if cfg.n_max > 9:
        raise ValueError("Keep n_max <= 9 for this sparse qutrit metric-growth test.")
    rng = np.random.default_rng(cfg.seed)
    dims = [3] * cfg.n_max
    cache = build_site_generator_cache(cfg.n_max)
    active_nodes = list(range(cfg.n_init))
    dormant_nodes = list(range(cfg.n_init, cfg.n_max))
    active_edges = [(i, i + 1) for i in range(cfg.n_init - 1)]
    local_coeffs: Dict[int, np.ndarray] = {}
    for i in range(cfg.n_max):
        local_coeffs[i] = rng.uniform(-cfg.local_scale, cfg.local_scale, size=8) if i < cfg.n_init else np.zeros(8, dtype=float)
    edge_strengths: Dict[Tuple[int, int], float] = {}
    for e in active_edges:
        edge_strengths[e] = float(rng.uniform(0.6 * cfg.pair_scale, 1.4 * cfg.pair_scale))
    psi = initial_state_qutrits(cfg.n_max, cfg.n_init, cfg.seed + 1000)
    spawn_events: List[Dict[str, object]] = []
    snapshots: List[Dict[str, object]] = []
    prev_coords: Optional[Dict[int, float]] = None

    for step in range(cfg.total_steps):
        H = build_hamiltonian(cfg.n_max, active_nodes, active_edges, cache, local_coeffs, edge_strengths)
        psi = expm_multiply((-1j * cfg.dt) * H, psi)
        psi = normalize_state(np.asarray(psi))
        rho = pure_density(psi)
        spawn_evt = None
        if ((step + 1) % cfg.spawn_every == 0) and (len(spawn_events) < cfg.max_spawns):
            spawn_evt = maybe_spawn_subsystem(psi, rho, active_nodes, active_edges, dormant_nodes, dims, cache, edge_strengths, local_coeffs, cfg.spawn_pair_scale, cfg.spawn_mi_threshold, cfg.spawn_corr_threshold, rng)
            if spawn_evt is not None:
                spawn_evt["step"] = step + 1
                spawn_events.append(spawn_evt)
        if ((step + 1) % cfg.snapshot_every == 0) or (spawn_evt is not None) or (step == cfg.total_steps - 1):
            metric = metric_observables(active_nodes, active_edges, edge_strengths, prev_coords)
            prev_coords = {int(k): float(v) for k, v in metric["coords"].items()}
            snapshots.append({
                "step": step + 1,
                "n_active": len(active_nodes),
                "n_edges": len(active_edges),
                "spawn_event": spawn_evt,
                "mean_edge_length": metric["mean_edge_length"],
                "total_edge_length": metric["total_edge_length"],
                "metric_extent": metric["metric_extent"],
                "insertion_strain": metric["insertion_strain"],
                "coords": metric["coords"],
            })

    spawn_snapshots = [s for s in snapshots if s["spawn_event"] is not None]
    summary = {
        "n_active_final": len(active_nodes),
        "n_edges_final": len(active_edges),
        "n_spawn_events": len(spawn_events),
        "mean_metric_extent": float(np.mean([s["metric_extent"] for s in snapshots])) if snapshots else 0.0,
        "max_metric_extent": float(np.max([s["metric_extent"] for s in snapshots])) if snapshots else 0.0,
        "mean_total_edge_length": float(np.mean([s["total_edge_length"] for s in snapshots])) if snapshots else 0.0,
        "mean_spawn_insertion_strain": float(np.mean([s["insertion_strain"] for s in spawn_snapshots])) if spawn_snapshots else 0.0,
        "max_spawn_insertion_strain": float(np.max([s["insertion_strain"] for s in spawn_snapshots])) if spawn_snapshots else 0.0,
        "extent_growth": float(snapshots[-1]["metric_extent"] - snapshots[0]["metric_extent"]) if len(snapshots) >= 2 else 0.0,
        "total_edge_length_growth": float(snapshots[-1]["total_edge_length"] - snapshots[0]["total_edge_length"]) if len(snapshots) >= 2 else 0.0,
    }
    return {"config": cfg.__dict__, "active_nodes_final": active_nodes, "active_edges_final": [list(e) for e in active_edges], "spawn_events": spawn_events, "snapshots": snapshots, "summary": summary}

def pretty_report(result: Dict[str, object]) -> str:
    cfg = result["config"]; summ = result["summary"]
    lines = []
    lines.append("=" * 108)
    lines.append("METRIC GROWTH FROM SPAWNING (v1)")
    lines.append("-" * 108)
    lines.append(f"n_max={cfg['n_max']}  n_init={cfg['n_init']}  seed={cfg['seed']}  steps={cfg['total_steps']}  dt={cfg['dt']}  spawn_every={cfg['spawn_every']}  snapshot_every={cfg['snapshot_every']}")
    lines.append(f"spawn thresholds: MI>={cfg['spawn_mi_threshold']}  corr>={cfg['spawn_corr_threshold']}  max_spawns={cfg['max_spawns']}")
    lines.append("Metric = 1D spectral embedding of the active weighted graph; spawning strain is displacement of old nodes after insertion.")
    lines.append("-" * 108)
    lines.append(f"Final counts: active_nodes={summ['n_active_final']}  active_edges={summ['n_edges_final']}  spawn_events={summ['n_spawn_events']}")
    lines.append(f"Metric summary: mean_extent={summ['mean_metric_extent']:.4f}  max_extent={summ['max_metric_extent']:.4f}  extent_growth={summ['extent_growth']:.4f}")
    lines.append(f"Edge-length summary: mean_total_edge_length={summ['mean_total_edge_length']:.4f}  total_edge_length_growth={summ['total_edge_length_growth']:.4f}")
    lines.append(f"Spawn strain: mean_spawn_insertion_strain={summ['mean_spawn_insertion_strain']:.4f}  max_spawn_insertion_strain={summ['max_spawn_insertion_strain']:.4f}")
    lines.append("-" * 108)
    lines.append("Spawn-linked snapshots:")
    found = False
    for snap in result["snapshots"]:
        evt = snap["spawn_event"]
        if evt is not None:
            found = True
            lines.append(f"  step={snap['step']}  parents={evt['parents']}  new_node={evt['new_node']}  MI={evt['trigger_mi']:.4f}  corr={evt['trigger_corr']:.4f}  extent={snap['metric_extent']:.4f}  total_edge_len={snap['total_edge_length']:.4f}  strain={snap['insertion_strain']:.4f}")
    if not found:
        lines.append("  none")
    return "\\n".join(lines)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Track emergent metric growth driven by relational subsystem spawning.")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-scale", type=float, default=0.10)
    ap.add_argument("--pair-scale", type=float, default=0.16)
    ap.add_argument("--spawn-pair-scale", type=float, default=0.12)
    ap.add_argument("--total-steps", type=int, default=60)
    ap.add_argument("--dt", type=float, default=0.20)
    ap.add_argument("--spawn-every", type=int, default=4)
    ap.add_argument("--snapshot-every", type=int, default=4)
    ap.add_argument("--spawn-mi-threshold", type=float, default=0.25)
    ap.add_argument("--spawn-corr-threshold", type=float, default=0.50)
    ap.add_argument("--max-spawns", type=int, default=6)
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    cfg = SimConfig(n_max=int(args.n_max), n_init=int(args.n_init), seed=int(args.seed), local_scale=float(args.local_scale), pair_scale=float(args.pair_scale), spawn_pair_scale=float(args.spawn_pair_scale), total_steps=int(args.total_steps), dt=float(args.dt), spawn_every=int(args.spawn_every), snapshot_every=int(args.snapshot_every), spawn_mi_threshold=float(args.spawn_mi_threshold), spawn_corr_threshold=float(args.spawn_corr_threshold), max_spawns=int(args.max_spawns), json_out=str(args.json_out))
    result = run_sim(cfg)
    print(pretty_report(result))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\\nSaved JSON: {args.json_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
