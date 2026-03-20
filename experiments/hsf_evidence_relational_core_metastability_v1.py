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

def active_pair_stats(psi: np.ndarray, rho: np.ndarray, active_edges: List[Tuple[int, int]], dims: List[int], cache: Dict[Tuple[int, int], csr_matrix]) -> List[Dict[str, object]]:
    out = []
    for (i, j) in active_edges:
        rho_ab = partial_trace_keep(rho, dims, [i, j])
        rho_a = partial_trace_keep(rho, dims, [i])
        rho_b = partial_trace_keep(rho, dims, [j])
        out.append({
            "edge": [i, j],
            "mutual_information": mutual_information(rho_ab, rho_a, rho_b),
            "su3_pair_correlator_strength": pair_su3_correlator_strength(psi, cache, i, j),
        })
    out.sort(key=lambda d: (d["mutual_information"], d["su3_pair_correlator_strength"]), reverse=True)
    return out

def active_triangles(rho: np.ndarray, active_nodes: List[int], active_edges: List[Tuple[int, int]], dims: List[int]) -> List[Dict[str, object]]:
    edge_set = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    out = []
    for a, b, c in combinations(active_nodes, 3):
        eab = (min(a, b), max(a, b)); ebc = (min(b, c), max(b, c)); eca = (min(c, a), max(c, a))
        if eab in edge_set and ebc in edge_set and eca in edge_set:
            rho_abc = partial_trace_keep(rho, dims, [a, b, c])
            rho_ab = partial_trace_keep(rho, dims, [a, b])
            rho_bc = partial_trace_keep(rho, dims, [b, c])
            rho_ca = partial_trace_keep(rho, dims, [c, a])
            rho_a = partial_trace_keep(rho, dims, [a])
            rho_b = partial_trace_keep(rho, dims, [b])
            rho_c = partial_trace_keep(rho, dims, [c])
            mi_ab = mutual_information(rho_ab, rho_a, rho_b)
            mi_bc = mutual_information(rho_bc, rho_b, rho_c)
            mi_ca = mutual_information(rho_ca, rho_c, rho_a)
            mi_vals = np.array([mi_ab, mi_bc, mi_ca], dtype=float)
            imbalance = float(np.std(mi_vals))
            tri_entropy = von_neumann_entropy(rho_abc)
            pair_mean = float(np.mean(mi_vals))
            frustration = float(min(1.0, imbalance / (1.0 + pair_mean) + tri_entropy / (1.0 + tri_entropy)))
            out.append({
                "triangle": [a, b, c],
                "pair_mi_mean": pair_mean,
                "pair_mi_std": imbalance,
                "triangle_frustration_proxy": frustration,
            })
    out.sort(key=lambda d: (d["pair_mi_mean"], -d["triangle_frustration_proxy"]), reverse=True)
    return out

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
    active_edges.append(e1); active_edges.append(e2); active_edges.sort()
    edge_strengths[e1] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
    edge_strengths[e2] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
    local_coeffs[new_node] = rng.uniform(-0.5, 0.5, size=8)
    return {"parents": [i, j], "new_node": new_node, "trigger_mi": float(mi), "trigger_corr": float(corr), "spawn_links": [list(e1), list(e2)]}

def daughter_count_for_pair(pair: Tuple[int, int], spawn_events: List[Dict[str, object]], active_edges: List[Tuple[int, int]]) -> int:
    p = tuple(pair)
    e_set = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    count = 0
    for evt in spawn_events:
        if tuple(evt["parents"]) == p:
            new_node = evt["new_node"]
            if (min(p[0], new_node), max(p[0], new_node)) in e_set and (min(p[1], new_node), max(p[1], new_node)) in e_set:
                count += 1
    return count

def dominant_core_snapshot(psi: np.ndarray, rho: np.ndarray, active_nodes: List[int], active_edges: List[Tuple[int, int]], dims: List[int], cache: Dict[Tuple[int, int], csr_matrix], spawn_events: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not active_edges:
        return None
    pair_rows = active_pair_stats(psi, rho, active_edges, dims, cache)
    triangles = active_triangles(rho, active_nodes, active_edges, dims)
    best = None; best_score = -1.0
    for row in pair_rows:
        i, j = row["edge"]
        daughters = daughter_count_for_pair((i, j), spawn_events, active_edges)
        score = float(row["mutual_information"]) * float(row["su3_pair_correlator_strength"]) * (1.0 + daughters)
        if score > best_score:
            best_score = score
            best = {"core_pair": [i, j], "core_score": score, "pair_mi": float(row["mutual_information"]), "pair_corr": float(row["su3_pair_correlator_strength"]), "attached_daughter_count": daughters}
    if best is None:
        return None
    a, b = best["core_pair"]
    attached_daughters = []
    e_set = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    for node in active_nodes:
        if node in (a, b):
            continue
        if (min(a, node), max(a, node)) in e_set and (min(b, node), max(b, node)) in e_set:
            attached_daughters.append(node)
    fr_vals = []
    core_triangles = []
    for tri in triangles:
        t = tri["triangle"]
        if a in t and b in t:
            core_triangles.append(t)
            fr_vals.append(float(tri["triangle_frustration_proxy"]))
    best["attached_daughters"] = attached_daughters
    best["n_core_triangles"] = len(core_triangles)
    best["core_triangles"] = core_triangles
    best["mean_core_triangle_frustration"] = float(np.mean(fr_vals)) if fr_vals else 0.0
    return best

def summarize_core_lifetimes(snapshots: List[Dict[str, object]]) -> Dict[str, object]:
    if not snapshots:
        return {"core_switch_count": 0, "longest_lived_core": None, "longest_lifetime_snapshots": 0, "mean_core_lifetime_snapshots": 0.0, "phase_label": "none"}
    runs = []
    current_pair = tuple(snapshots[0]["dominant_core"]["core_pair"]) if snapshots[0]["dominant_core"] else None
    start = 0
    for idx in range(1, len(snapshots)):
        pair = tuple(snapshots[idx]["dominant_core"]["core_pair"]) if snapshots[idx]["dominant_core"] else None
        if pair != current_pair:
            runs.append((current_pair, start, idx - 1))
            current_pair = pair
            start = idx
    runs.append((current_pair, start, len(snapshots) - 1))
    valid_runs = [r for r in runs if r[0] is not None]
    lifetimes = [end - start + 1 for (_, start, end) in valid_runs]
    switch_count = max(0, len(valid_runs) - 1)
    if lifetimes:
        best_idx = int(np.argmax(lifetimes))
        best_pair = valid_runs[best_idx][0]
        best_life = lifetimes[best_idx]
        mean_life = float(np.mean(lifetimes))
    else:
        best_pair = None; best_life = 0; mean_life = 0.0
    if switch_count == 0:
        phase = "stable_core"
        pair0 = tuple(snapshots[0]["dominant_core"]["core_pair"]) if snapshots[0]["dominant_core"] else None
        if pair0 is not None:
            daughter_sets = [tuple(sorted(s["dominant_core"].get("attached_daughters", []))) for s in snapshots if s["dominant_core"]]
            if len(set(daughter_sets)) > 1:
                phase = "stable_core_shell_churn"
    elif switch_count <= 2:
        phase = "metastable_core_switching"
    else:
        phase = "boiling_core_switching"
    return {"core_switch_count": switch_count, "longest_lived_core": list(best_pair) if best_pair is not None else None, "longest_lifetime_snapshots": int(best_life), "mean_core_lifetime_snapshots": float(mean_life), "phase_label": phase}

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
        raise ValueError("Keep n_max <= 9 for this sparse qutrit metastability test.")
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
    history: List[Dict[str, object]] = []
    prev_core: Optional[Tuple[int, int]] = None
    for step in range(cfg.total_steps):
        H = build_hamiltonian(cfg.n_max, active_nodes, active_edges, cache, local_coeffs, edge_strengths)
        psi = expm_multiply((-1j * cfg.dt) * H, psi)
        psi = normalize_state(np.asarray(psi))
        if ((step + 1) % cfg.spawn_every == 0) and (len(spawn_events) < cfg.max_spawns):
            rho = pure_density(psi)
            evt = maybe_spawn_subsystem(psi, rho, active_nodes, active_edges, dormant_nodes, dims, cache, edge_strengths, local_coeffs, cfg.spawn_pair_scale, cfg.spawn_mi_threshold, cfg.spawn_corr_threshold, rng)
            if evt is not None:
                evt["step"] = step + 1
                spawn_events.append(evt)
        if ((step + 1) % cfg.snapshot_every == 0) or (step == cfg.total_steps - 1):
            rho = pure_density(psi)
            core = dominant_core_snapshot(psi, rho, active_nodes, active_edges, dims, cache, spawn_events)
            core_pair = tuple(core["core_pair"]) if core is not None else None
            snapshots.append({"step": step + 1, "n_active": len(active_nodes), "n_edges": len(active_edges), "dominant_core": core, "core_changed": (prev_core is not None and core_pair != prev_core)})
            prev_core = core_pair
        history.append({"step": step + 1, "n_active": len(active_nodes), "n_edges": len(active_edges)})
    rho = pure_density(psi)
    pairs = active_pair_stats(psi, rho, active_edges, dims, cache)
    triangles = active_triangles(rho, active_nodes, active_edges, dims)
    single_entropies = {}
    for i in active_nodes:
        rho_i = partial_trace_keep(rho, dims, [i])
        single_entropies[str(i)] = float(von_neumann_entropy(rho_i))
    core_summary = summarize_core_lifetimes(snapshots)
    summary = {
        "n_active_final": len(active_nodes),
        "n_edges_final": len(active_edges),
        "n_spawn_events": len(spawn_events),
        "mean_single_subsystem_entropy": float(np.mean(list(single_entropies.values()))) if single_entropies else 0.0,
        "mean_pair_mutual_information": float(np.mean([x["mutual_information"] for x in pairs])) if pairs else 0.0,
        "mean_triangle_pair_mi": float(np.mean([x["pair_mi_mean"] for x in triangles])) if triangles else 0.0,
        "mean_triangle_frustration": float(np.mean([x["triangle_frustration_proxy"] for x in triangles])) if triangles else 0.0,
        "single_subsystem_entropies": single_entropies,
        "top_pairs": pairs[:min(10, len(pairs))],
        "top_triangles": triangles[:min(10, len(triangles))],
        **core_summary,
    }
    return {"config": cfg.__dict__, "active_nodes_final": active_nodes, "active_edges_final": [list(e) for e in active_edges], "spawn_events": spawn_events, "snapshots": snapshots, "history": history, "summary": summary}

def pretty_report(result: Dict[str, object]) -> str:
    cfg = result["config"]; summ = result["summary"]
    lines = []
    lines.append("=" * 112)
    lines.append("HSF EVIDENCE TEST: RELATIONAL CORE METASTABILITY (v1)")
    lines.append("-" * 112)
    lines.append(f"n_max={cfg['n_max']}  n_init={cfg['n_init']}  seed={cfg['seed']}  steps={cfg['total_steps']}  dt={cfg['dt']}  spawn_every={cfg['spawn_every']}  snapshot_every={cfg['snapshot_every']}")
    lines.append(f"spawn thresholds: MI>={cfg['spawn_mi_threshold']}  corr>={cfg['spawn_corr_threshold']}  max_spawns={cfg['max_spawns']}")
    lines.append("Dominant core = strongest parent pair by MI * SU3_corr * (1 + attached_daughter_count).")
    lines.append("-" * 112)
    lines.append(f"Final counts: active_nodes={summ['n_active_final']}  active_edges={summ['n_edges_final']}  spawn_events={summ['n_spawn_events']}")
    lines.append(f"Means: single_entropy={summ['mean_single_subsystem_entropy']:.4f}  pair_MI={summ['mean_pair_mutual_information']:.4f}  triangle_pair_MI={summ['mean_triangle_pair_mi']:.4f}  triangle_frustr={summ['mean_triangle_frustration']:.4f}")
    lines.append(f"Core summary: phase={summ['phase_label']}  switches={summ['core_switch_count']}  longest_core={summ['longest_lived_core']}  longest_life_snapshots={summ['longest_lifetime_snapshots']}  mean_life_snapshots={summ['mean_core_lifetime_snapshots']:.2f}")
    lines.append("-" * 112)
    lines.append("Spawn events:")
    if result["spawn_events"]:
        for evt in result["spawn_events"]:
            lines.append(f"  step={evt['step']}  parents={evt['parents']}  new_node={evt['new_node']}  MI={evt['trigger_mi']:.4f}  corr={evt['trigger_corr']:.4f}  links={evt['spawn_links']}")
    else:
        lines.append("  none")
    lines.append("-" * 112)
    lines.append("Core snapshots:")
    for snap in result["snapshots"][:min(15, len(result["snapshots"]))]:
        core = snap["dominant_core"]
        if core is None:
            lines.append(f"  step={snap['step']}  no_core")
        else:
            lines.append(f"  step={snap['step']}  core={core['core_pair']}  score={core['core_score']:.4f}  MI={core['pair_mi']:.4f}  corr={core['pair_corr']:.4f}  daughters={core['attached_daughters']}  triangles={core['n_core_triangles']}  mean_frustr={core['mean_core_triangle_frustration']:.4f}  changed={snap['core_changed']}")
    return "\n".join(lines)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HSF evidence test for relational core metastability.")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-scale", type=float, default=0.10)
    ap.add_argument("--pair-scale", type=float, default=0.16)
    ap.add_argument("--spawn-pair-scale", type=float, default=0.12)
    ap.add_argument("--total-steps", type=int, default=80)
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
        print(f"\nSaved JSON: {args.json_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
