#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

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


def conditional_mutual_information(rho_abc: np.ndarray, rho_ab: np.ndarray, rho_bc: np.ndarray, rho_b: np.ndarray) -> float:
    return float(von_neumann_entropy(rho_ab) + von_neumann_entropy(rho_bc) - von_neumann_entropy(rho_b) - von_neumann_entropy(rho_abc))


def build_site_generator_cache(n_max: int) -> Dict[Tuple[int, int], csr_matrix]:
    cache: Dict[Tuple[int, int], csr_matrix] = {}
    for site in range(n_max):
        for a, lam in enumerate(GM):
            ops = [I3] * n_max
            ops[site] = lam
            cache[(site, a)] = kron_all_sparse(ops)
    return cache


def pair_su3_correlator_strength(psi: np.ndarray, cache: Dict[Tuple[int, int], csr_matrix], i: int, j: int) -> float:
    vals = []
    for a in range(8):
        vals.append(float(np.real(expect_sparse(psi, cache[(i, a)] @ cache[(j, a)]))))
    return float(np.linalg.norm(vals))


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
    spawn_mi_threshold: float
    spawn_corr_threshold: float
    max_spawns: int
    json_out: str = ""


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


def build_hamiltonian(
    n_max: int,
    active_nodes: List[int],
    active_edges: List[Tuple[int, int]],
    cache: Dict[Tuple[int, int], csr_matrix],
    local_coeffs: Dict[int, np.ndarray],
    edge_strengths: Dict[Tuple[int, int], float],
) -> csr_matrix:
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


def analyze_active_pairs(
    psi: np.ndarray,
    rho: np.ndarray,
    active_nodes: List[int],
    active_edges: List[Tuple[int, int]],
    dims: List[int],
    cache: Dict[Tuple[int, int], csr_matrix],
) -> List[Dict[str, object]]:
    out = []
    for (i, j) in active_edges:
        rho_ab = partial_trace_keep(rho, dims, [i, j])
        rho_a = partial_trace_keep(rho, dims, [i])
        rho_b = partial_trace_keep(rho, dims, [j])
        out.append({
            "edge": [i, j],
            "mutual_information": mutual_information(rho_ab, rho_a, rho_b),
            "su3_pair_correlator_strength": pair_su3_correlator_strength(psi, cache, i, j),
            "entropy_i": von_neumann_entropy(rho_a),
            "entropy_j": von_neumann_entropy(rho_b),
        })
    out.sort(key=lambda d: (d["mutual_information"], d["su3_pair_correlator_strength"]), reverse=True)
    return out


def analyze_active_chains(
    rho: np.ndarray,
    active_nodes: List[int],
    active_edges: List[Tuple[int, int]],
    dims: List[int],
) -> List[Dict[str, object]]:
    edge_set = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    adj = {i: [] for i in active_nodes}
    for i, j in active_edges:
        adj[i].append(j)
        adj[j].append(i)
    out = []
    for center in active_nodes:
        nbrs = sorted(adj.get(center, []))
        for a, c in combinations(nbrs, 2):
            rho_abc = partial_trace_keep(rho, dims, [a, center, c])
            rho_ab = partial_trace_keep(rho, dims, [a, center])
            rho_bc = partial_trace_keep(rho, dims, [center, c])
            rho_b = partial_trace_keep(rho, dims, [center])
            out.append({
                "chain": [a, center, c],
                "conditional_mutual_information": conditional_mutual_information(rho_abc, rho_ab, rho_bc, rho_b),
                "has_direct_ac_edge": (min(a, c), max(a, c)) in edge_set,
                "three_body_entropy": von_neumann_entropy(rho_abc),
            })
    out.sort(key=lambda d: d["conditional_mutual_information"], reverse=True)
    return out


def analyze_active_triangles(
    rho: np.ndarray,
    active_nodes: List[int],
    active_edges: List[Tuple[int, int]],
    dims: List[int],
) -> List[Dict[str, object]]:
    edge_set = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    out = []
    for a, b, c in combinations(active_nodes, 3):
        eab = (min(a, b), max(a, b))
        ebc = (min(b, c), max(b, c))
        eca = (min(c, a), max(c, a))
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
                "three_body_entropy": tri_entropy,
                "triangle_frustration_proxy": frustration,
            })
    out.sort(key=lambda d: (d["pair_mi_mean"], -d["triangle_frustration_proxy"]), reverse=True)
    return out


def maybe_spawn_subsystem(
    psi: np.ndarray,
    rho: np.ndarray,
    active_nodes: List[int],
    active_edges: List[Tuple[int, int]],
    dormant_nodes: List[int],
    dims: List[int],
    cache: Dict[Tuple[int, int], csr_matrix],
    edge_strengths: Dict[Tuple[int, int], float],
    local_coeffs: Dict[int, np.ndarray],
    spawn_pair_scale: float,
    mi_threshold: float,
    corr_threshold: float,
    rng: np.random.Generator,
) -> Dict[str, object] | None:
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
        score = mi * corr
        candidates.append((score, i, j, mi, corr))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    score, i, j, mi, corr = candidates[0]
    if mi < mi_threshold or corr < corr_threshold:
        return None

    new_node = dormant_nodes.pop(0)
    active_nodes.append(new_node)
    active_nodes.sort()

    # relational spawning rule:
    # new subsystem inherits links to both parents only, representing
    # a new factor added to accommodate persistent relational structure.
    e1 = (min(i, new_node), max(i, new_node))
    e2 = (min(j, new_node), max(j, new_node))
    active_edges.append(e1)
    active_edges.append(e2)
    active_edges.sort()

    edge_strengths[e1] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
    edge_strengths[e2] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
    local_coeffs[new_node] = rng.uniform(-0.5, 0.5, size=8)

    return {
        "parents": [i, j],
        "new_node": new_node,
        "trigger_mi": float(mi),
        "trigger_corr": float(corr),
        "spawn_links": [list(e1), list(e2)],
    }


def run_sim(cfg: SimConfig) -> Dict[str, object]:
    if cfg.n_init >= cfg.n_max:
        raise ValueError("Require n_init < n_max so there are dormant subsystem slots available.")
    if cfg.n_max > 9:
        raise ValueError("Keep n_max <= 9 for this sparse qutrit spawning toy.")

    rng = np.random.default_rng(cfg.seed)
    dims = [3] * cfg.n_max
    cache = build_site_generator_cache(cfg.n_max)

    active_nodes = list(range(cfg.n_init))
    dormant_nodes = list(range(cfg.n_init, cfg.n_max))
    active_edges = [(i, i + 1) for i in range(cfg.n_init - 1)]  # start from a minimal chain among initially active subsystems

    local_coeffs: Dict[int, np.ndarray] = {}
    for i in range(cfg.n_max):
        if i < cfg.n_init:
            local_coeffs[i] = rng.uniform(-cfg.local_scale, cfg.local_scale, size=8)
        else:
            local_coeffs[i] = np.zeros(8, dtype=float)

    edge_strengths: Dict[Tuple[int, int], float] = {}
    for e in active_edges:
        edge_strengths[e] = float(rng.uniform(0.6 * cfg.pair_scale, 1.4 * cfg.pair_scale))

    psi = initial_state_qutrits(cfg.n_max, cfg.n_init, cfg.seed + 1000)

    spawn_events = []
    history = []

    for step in range(cfg.total_steps):
        H = build_hamiltonian(cfg.n_max, active_nodes, active_edges, cache, local_coeffs, edge_strengths)
        psi = expm_multiply((-1j * cfg.dt) * H, psi)
        psi = normalize_state(np.asarray(psi))

        if ((step + 1) % cfg.spawn_every == 0) and (len(spawn_events) < cfg.max_spawns):
            rho = pure_density(psi)
            evt = maybe_spawn_subsystem(
                psi=psi,
                rho=rho,
                active_nodes=active_nodes,
                active_edges=active_edges,
                dormant_nodes=dormant_nodes,
                dims=dims,
                cache=cache,
                edge_strengths=edge_strengths,
                local_coeffs=local_coeffs,
                spawn_pair_scale=cfg.spawn_pair_scale,
                mi_threshold=cfg.spawn_mi_threshold,
                corr_threshold=cfg.spawn_corr_threshold,
                rng=rng,
            )
            if evt is not None:
                evt["step"] = step + 1
                spawn_events.append(evt)

        history.append({
            "step": step + 1,
            "n_active": len(active_nodes),
            "n_edges": len(active_edges),
        })

    rho = pure_density(psi)
    pairs = analyze_active_pairs(psi, rho, active_nodes, active_edges, dims, cache)
    chains = analyze_active_chains(rho, active_nodes, active_edges, dims)
    triangles = analyze_active_triangles(rho, active_nodes, active_edges, dims)

    single_entropies = {}
    for i in active_nodes:
        rho_i = partial_trace_keep(rho, dims, [i])
        single_entropies[str(i)] = float(von_neumann_entropy(rho_i))

    summary = {
        "n_active_final": len(active_nodes),
        "n_edges_final": len(active_edges),
        "n_spawn_events": len(spawn_events),
        "mean_single_subsystem_entropy": float(np.mean(list(single_entropies.values()))) if single_entropies else 0.0,
        "mean_pair_mutual_information": float(np.mean([x["mutual_information"] for x in pairs])) if pairs else 0.0,
        "mean_chain_cmi": float(np.mean([x["conditional_mutual_information"] for x in chains])) if chains else 0.0,
        "mean_triangle_pair_mi": float(np.mean([x["pair_mi_mean"] for x in triangles])) if triangles else 0.0,
        "mean_triangle_frustration": float(np.mean([x["triangle_frustration_proxy"] for x in triangles])) if triangles else 0.0,
        "top_pairs": pairs[:min(10, len(pairs))],
        "top_chains": chains[:min(10, len(chains))],
        "top_triangles": triangles[:min(10, len(triangles))],
        "single_subsystem_entropies": single_entropies,
    }

    return {
        "config": cfg.__dict__,
        "active_nodes_final": active_nodes,
        "active_edges_final": [list(e) for e in active_edges],
        "spawn_events": spawn_events,
        "history": history,
        "summary": summary,
    }


def pretty_report(result: Dict[str, object]) -> str:
    cfg = result["config"]
    summ = result["summary"]
    lines = []
    lines.append("=" * 108)
    lines.append("HSF FULL-HILBERT SUBSYSTEMS SU(3) SPARSE + RELATIONAL SPAWNING (v3)")
    lines.append("-" * 108)
    lines.append(
        f"n_max={cfg['n_max']}  n_init={cfg['n_init']}  seed={cfg['seed']}  "
        f"steps={cfg['total_steps']}  dt={cfg['dt']}"
    )
    lines.append(
        f"local_scale={cfg['local_scale']}  pair_scale={cfg['pair_scale']}  spawn_pair_scale={cfg['spawn_pair_scale']}"
    )
    lines.append(
        f"spawn_every={cfg['spawn_every']}  spawn_mi_threshold={cfg['spawn_mi_threshold']}  "
        f"spawn_corr_threshold={cfg['spawn_corr_threshold']}  max_spawns={cfg['max_spawns']}"
    )
    lines.append(
        "Subsystems are full qutrit Hilbert factors. New effective subsystems are activated when a parent pair's "
        "persistent relational structure passes MI + SU(3)-correlator thresholds."
    )
    lines.append("-" * 108)
    lines.append(
        f"Final counts: active_nodes={summ['n_active_final']}  active_edges={summ['n_edges_final']}  "
        f"spawn_events={summ['n_spawn_events']}"
    )
    lines.append(
        f"Means: single_entropy={summ['mean_single_subsystem_entropy']:.4f}  "
        f"pair_MI={summ['mean_pair_mutual_information']:.4f}  "
        f"chain_CMI={summ['mean_chain_cmi']:.4f}  "
        f"triangle_pair_MI={summ['mean_triangle_pair_mi']:.4f}  "
        f"triangle_frustr={summ['mean_triangle_frustration']:.4f}"
    )
    lines.append("-" * 108)
    lines.append("Spawn events:")
    if result["spawn_events"]:
        for evt in result["spawn_events"]:
            lines.append(
                f"  step={evt['step']}  parents={evt['parents']}  new_node={evt['new_node']}  "
                f"MI={evt['trigger_mi']:.4f}  corr={evt['trigger_corr']:.4f}  links={evt['spawn_links']}"
            )
    else:
        lines.append("  none")
    lines.append("-" * 108)
    lines.append("Top pairs:")
    for row in summ["top_pairs"]:
        lines.append(
            f"  edge={row['edge']}  MI={row['mutual_information']:.4f}  "
            f"SU3_corr={row['su3_pair_correlator_strength']:.4f}"
        )
    lines.append("-" * 108)
    lines.append("Top 2nd-order chains:")
    for row in summ["top_chains"]:
        lines.append(
            f"  chain={row['chain']}  CMI={row['conditional_mutual_information']:.4f}  "
            f"direct_ac={row['has_direct_ac_edge']}  S_ABC={row['three_body_entropy']:.4f}"
        )
    if summ["top_triangles"]:
        lines.append("-" * 108)
        lines.append("Top 3rd-order triangles:")
        for row in summ["top_triangles"]:
            lines.append(
                f"  tri={row['triangle']}  MImean={row['pair_mi_mean']:.4f}  "
                f"MIstd={row['pair_mi_std']:.4f}  frustr={row['triangle_frustration_proxy']:.4f}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sparse full-qutrit subsystem evolution with relational subsystem spawning.")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-scale", type=float, default=0.10)
    ap.add_argument("--pair-scale", type=float, default=0.16)
    ap.add_argument("--spawn-pair-scale", type=float, default=0.12)
    ap.add_argument("--total-steps", type=int, default=24)
    ap.add_argument("--dt", type=float, default=0.20)
    ap.add_argument("--spawn-every", type=int, default=4)
    ap.add_argument("--spawn-mi-threshold", type=float, default=0.25)
    ap.add_argument("--spawn-corr-threshold", type=float, default=0.50)
    ap.add_argument("--max-spawns", type=int, default=4)
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SimConfig(
        n_max=int(args.n_max),
        n_init=int(args.n_init),
        seed=int(args.seed),
        local_scale=float(args.local_scale),
        pair_scale=float(args.pair_scale),
        spawn_pair_scale=float(args.spawn_pair_scale),
        total_steps=int(args.total_steps),
        dt=float(args.dt),
        spawn_every=int(args.spawn_every),
        spawn_mi_threshold=float(args.spawn_mi_threshold),
        spawn_corr_threshold=float(args.spawn_corr_threshold),
        max_spawns=int(args.max_spawns),
        json_out=str(args.json_out),
    )
    result = run_sim(cfg)
    print(pretty_report(result))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
