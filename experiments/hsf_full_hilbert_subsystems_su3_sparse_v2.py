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


def make_graph_edges(n: int, graph_type: str, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = set()
    if graph_type == "chain":
        for i in range(n - 1):
            edges.add((i, i + 1))
    elif graph_type == "ring":
        for i in range(n):
            edges.add(tuple(sorted((i, (i + 1) % n))))
    elif graph_type == "ring_plus_chords":
        for i in range(n):
            edges.add(tuple(sorted((i, (i + 1) % n))))
        chords = max(1, n // 3)
        tries = 0
        while len(edges) < n + chords and tries < 12 * n:
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            if abs(a - b) not in (1, n - 1):
                edges.add((a, b))
            tries += 1
    elif graph_type == "triangle_tail":
        if n < 4:
            raise ValueError("triangle_tail requires n >= 4")
        edges.update({(0, 1), (1, 2), (0, 2), (2, 3)})
        for i in range(4, n):
            edges.add((i - 1, i))
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    return sorted(edges)


def build_site_generator_cache(n: int) -> Dict[Tuple[int, int], csr_matrix]:
    cache: Dict[Tuple[int, int], csr_matrix] = {}
    for site in range(n):
        for a, lam in enumerate(GM):
            ops = [I3] * n
            ops[site] = lam
            cache[(site, a)] = kron_all_sparse(ops)
    return cache


def build_hamiltonian(
    n: int,
    edges: List[Tuple[int, int]],
    cache: Dict[Tuple[int, int], csr_matrix],
    rng: np.random.Generator,
    local_scale: float,
    pair_scale: float,
) -> Tuple[csr_matrix, Dict[Tuple[int, int], float]]:
    dim = 3 ** n
    H = csr_matrix((dim, dim), dtype=complex)

    for site in range(n):
        coeffs = rng.uniform(-local_scale, local_scale, size=8)
        for a in range(8):
            H = H + float(coeffs[a]) * cache[(site, a)]

    edge_strengths: Dict[Tuple[int, int], float] = {}
    for e in edges:
        i, j = e
        g = float(rng.uniform(0.6 * pair_scale, 1.4 * pair_scale))
        edge_strengths[e] = g
        for a in range(8):
            H = H + g * (cache[(i, a)] @ cache[(j, a)])

    return H.tocsr(), edge_strengths


def random_product_qutrit_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    local_states = []
    for _ in range(n):
        z = rng.normal(size=3) + 1j * rng.normal(size=3)
        z = z / np.linalg.norm(z)
        local_states.append(z)
    psi = local_states[0]
    for v in local_states[1:]:
        psi = np.kron(psi, v)
    return normalize_state(psi)


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


def pair_su3_correlator_strength(psi: np.ndarray, cache: Dict[Tuple[int, int], csr_matrix], i: int, j: int) -> float:
    vals = []
    for a in range(8):
        vals.append(float(np.real(expect_sparse(psi, cache[(i, a)] @ cache[(j, a)]))))
    return float(np.linalg.norm(vals))


def analyze_pairs(psi: np.ndarray, rho: np.ndarray, n: int, dims: List[int], cache: Dict[Tuple[int, int], csr_matrix], edges: List[Tuple[int, int]]) -> List[Dict[str, object]]:
    out = []
    for (i, j) in edges:
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


def analyze_chains(rho: np.ndarray, n: int, dims: List[int], edges: List[Tuple[int, int]]) -> List[Dict[str, object]]:
    edge_set = set(edges)
    adj = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    out = []
    for center in range(n):
        nbrs = sorted(adj[center])
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


def analyze_triangles(rho: np.ndarray, n: int, dims: List[int], edges: List[Tuple[int, int]]) -> List[Dict[str, object]]:
    edge_set = set(edges)
    out = []
    for a, b, c in combinations(range(n), 3):
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
                "mi_ab": mi_ab,
                "mi_bc": mi_bc,
                "mi_ca": mi_ca,
                "pair_mi_mean": pair_mean,
                "pair_mi_std": imbalance,
                "three_body_entropy": tri_entropy,
                "triangle_frustration_proxy": frustration,
            })
    out.sort(key=lambda d: (d["pair_mi_mean"], -d["triangle_frustration_proxy"]), reverse=True)
    return out


@dataclass
class SimConfig:
    n: int
    graph_type: str
    seed: int
    local_scale: float
    pair_scale: float
    time: float


def run_sim(cfg: SimConfig) -> Dict[str, object]:
    if cfg.n > 9:
        raise ValueError("Keep N <= 9 for this sparse full-qutrit version unless you reduce observables or runtime.")
    rng = np.random.default_rng(cfg.seed)
    edges = make_graph_edges(cfg.n, cfg.graph_type, rng)
    cache = build_site_generator_cache(cfg.n)
    H, edge_strengths = build_hamiltonian(cfg.n, edges, cache, rng, local_scale=cfg.local_scale, pair_scale=cfg.pair_scale)
    psi0 = random_product_qutrit_state(cfg.n, cfg.seed + 1000)
    psi = expm_multiply((-1j * cfg.time) * H, psi0)
    psi = normalize_state(np.asarray(psi))
    rho = pure_density(psi)
    dims = [3] * cfg.n
    pairs = analyze_pairs(psi, rho, cfg.n, dims, cache, edges)
    chains = analyze_chains(rho, cfg.n, dims, edges)
    triangles = analyze_triangles(rho, cfg.n, dims, edges)
    single_entropies = [float(von_neumann_entropy(partial_trace_keep(rho, dims, [i]))) for i in range(cfg.n)]
    summary = {
        "n_edges": len(edges),
        "n_pairs": len(pairs),
        "n_chains": len(chains),
        "n_triangles": len(triangles),
        "mean_single_subsystem_entropy": float(np.mean(single_entropies)) if single_entropies else 0.0,
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
        "edges": [list(e) for e in edges],
        "edge_strengths": {f"{i}-{j}": float(g) for (i, j), g in edge_strengths.items()},
        "summary": summary,
    }

def pretty_report(result: Dict[str, object]) -> str:
    cfg = result["config"]; summ = result["summary"]
    lines = []
    lines.append("=" * 108)
    lines.append("HSF FULL-HILBERT SUBSYSTEMS SU(3) SPARSE ANALYSIS (v2)")
    lines.append("-" * 108)
    lines.append(f"N={cfg['n']}  graph={cfg['graph_type']}  seed={cfg['seed']}  local_scale={cfg['local_scale']}  pair_scale={cfg['pair_scale']}  time={cfg['time']}")
    lines.append("Subsystems are full qutrit Hilbert factors. Evolution uses sparse Krylov propagation instead of dense diagonalization.")
    lines.append("-" * 108)
    lines.append(f"Counts: edges={summ['n_edges']}  pairs={summ['n_pairs']}  chains={summ['n_chains']}  triangles={summ['n_triangles']}")
    lines.append(f"Means: single_entropy={summ['mean_single_subsystem_entropy']:.4f}  pair_MI={summ['mean_pair_mutual_information']:.4f}  chain_CMI={summ['mean_chain_cmi']:.4f}  triangle_pair_MI={summ['mean_triangle_pair_mi']:.4f}  triangle_frustr={summ['mean_triangle_frustration']:.4f}")
    lines.append("-" * 108)
    lines.append("Top pairs:")
    for row in summ["top_pairs"]:
        lines.append(f"  edge={row['edge']}  MI={row['mutual_information']:.4f}  SU3_corr={row['su3_pair_correlator_strength']:.4f}  S_i={row['entropy_i']:.4f}  S_j={row['entropy_j']:.4f}")
    lines.append("-" * 108)
    lines.append("Top 2nd-order chains:")
    for row in summ["top_chains"]:
        lines.append(f"  chain={row['chain']}  CMI={row['conditional_mutual_information']:.4f}  direct_ac={row['has_direct_ac_edge']}  S_ABC={row['three_body_entropy']:.4f}")
    if summ["top_triangles"]:
        lines.append("-" * 108)
        lines.append("Top 3rd-order triangles:")
        for row in summ["top_triangles"]:
            lines.append(f"  tri={row['triangle']}  MImean={row['pair_mi_mean']:.4f}  MIstd={row['pair_mi_std']:.4f}  S_ABC={row['three_body_entropy']:.4f}  frustr={row['triangle_frustration_proxy']:.4f}")
    return "\n".join(lines)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sparse exact-ish full-Hilbert qutrit subsystem SU(3) analyzer.")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--graph-type", choices=["chain","ring","ring_plus_chords","triangle_tail"], default="triangle_tail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-scale", type=float, default=0.12)
    ap.add_argument("--pair-scale", type=float, default=0.18)
    ap.add_argument("--time", type=float, default=1.20)
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    cfg = SimConfig(n=int(args.n), graph_type=str(args.graph_type), seed=int(args.seed), local_scale=float(args.local_scale), pair_scale=float(args.pair_scale), time=float(args.time))
    result = run_sim(cfg)
    print(pretty_report(result))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved JSON: {args.json_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
