#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

def kron_all(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def normalize_state(psi: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(psi)
    if n <= 1e-15:
        raise ValueError("State norm vanished.")
    return psi / n

def expect(psi: np.ndarray, op: np.ndarray) -> complex:
    return np.vdot(psi, op @ psi)

def matrix_exp_hermitian(H: np.ndarray, dt: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(H)
    return vecs @ np.diag(np.exp(-1j * dt * vals)) @ vecs.conj().T

def build_operator_cache(n: int) -> Dict[str, List[np.ndarray]]:
    I = np.eye(2, dtype=complex)
    X1 = np.array([[0, 1], [1, 0]], dtype=complex)
    Y1 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z1 = np.array([[1, 0], [0, -1]], dtype=complex)
    X, Y, Z = [], [], []
    for i in range(n):
        ops = [I] * n
        ops[i] = X1
        X.append(kron_all(ops))
        ops = [I] * n
        ops[i] = Y1
        Y.append(kron_all(ops))
        ops = [I] * n
        ops[i] = Z1
        Z.append(kron_all(ops))
    return {"X": X, "Y": Y, "Z": Z}

def jw_string_ops(n: int, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0 <= i < j < n):
        raise ValueError("Require 0 <= i < j < n.")
    I = np.eye(2, dtype=complex)
    X1 = np.array([[0, 1], [1, 0]], dtype=complex)
    Y1 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z1 = np.array([[1, 0], [0, -1]], dtype=complex)
    ops_xx = [I] * n
    ops_yy = [I] * n
    ops_xx[i] = X1
    ops_yy[i] = Y1
    for k in range(i + 1, j):
        ops_xx[k] = Z1
        ops_yy[k] = Z1
    ops_xx[j] = X1
    ops_yy[j] = Y1
    return kron_all(ops_xx), kron_all(ops_yy)

def make_graph_edges(n: int, graph_type: str, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = set()
    if graph_type == "ring":
        for i in range(n):
            edges.add(tuple(sorted((i, (i + 1) % n))))
    elif graph_type == "chain":
        for i in range(n - 1):
            edges.add((i, i + 1))
    elif graph_type == "ring_plus_chords":
        for i in range(n):
            edges.add(tuple(sorted((i, (i + 1) % n))))
        chords = max(1, n // 3)
        tries = 0
        while len(edges) < n + chords and tries < 10 * n:
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            if abs(a - b) not in (1, n - 1):
                edges.add((a, b))
            tries += 1
    elif graph_type == "erdos":
        p = min(0.55, max(0.25, 2.5 / max(2, n)))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.uniform() < p:
                    edges.add((i, j))
        for i in range(n - 1):
            edges.add((i, i + 1))
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    return sorted(edges)

def build_hamiltonian(n: int, cache: Dict[str, List[np.ndarray]], hz: np.ndarray, J: Dict[Tuple[int, int], float]) -> np.ndarray:
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        H += float(hz[i]) * cache["Z"][i]
    for (i, j), Jij in J.items():
        H += float(Jij) * (cache["X"][i] @ cache["X"][j] + cache["Y"][i] @ cache["Y"][j])
    return 0.5 * (H + H.conj().T)

def node_occupations(psi: np.ndarray, cache: Dict[str, List[np.ndarray]]) -> np.ndarray:
    n = len(cache["Z"])
    occ = np.zeros(n, dtype=float)
    for i in range(n):
        z = float(np.real(expect(psi, cache["Z"][i])))
        occ[i] = 0.5 * (1.0 - z)
    return occ

def edge_coherence(psi: np.ndarray, cache: Dict[str, List[np.ndarray]], i: int, j: int) -> float:
    val = expect(psi, cache["X"][i] @ cache["X"][j] + cache["Y"][i] @ cache["Y"][j])
    return 0.5 * abs(float(np.real(val)))

def jw_amplitude(psi: np.ndarray, jw_ops: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]], i: int, j: int) -> float:
    Oxx, Oyy = jw_ops[(i, j)]
    val = expect(psi, Oxx + Oyy)
    return 0.5 * abs(float(np.real(val)))

def commitment_scores(edges: List[Tuple[int, int]], coh: Dict[Tuple[int, int], float], n: int) -> Dict[Tuple[int, int], float]:
    node_sum = np.zeros(n, dtype=float)
    for (i, j), c in coh.items():
        node_sum[i] += c
        node_sum[j] += c
    out = {}
    for (i, j) in edges:
        c = coh[(i, j)]
        excl_i = 0.0 if node_sum[i] <= 1e-15 else c / node_sum[i]
        excl_j = 0.0 if node_sum[j] <= 1e-15 else c / node_sum[j]
        out[(i, j)] = float(np.sqrt(max(0.0, excl_i * excl_j)))
    return out

def update_couplings(J: Dict[Tuple[int, int], float], commit: Dict[Tuple[int, int], float], J_min: float, J_max: float, eta_up: float, eta_down: float, target: float) -> Dict[Tuple[int, int], float]:
    newJ = {}
    for e, old in J.items():
        c = commit[e]
        delta = eta_up * max(0.0, c - target) - eta_down * max(0.0, target - c)
        newJ[e] = float(min(J_max, max(J_min, old + delta)))
    return newJ

def make_initial_state(n: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 2 ** n
    if kind == "single_excitation_center":
        k = n // 2
        basis_index = 1 << (n - 1 - k)
        psi = np.zeros(dim, dtype=complex)
        psi[basis_index] = 1.0
        return psi
    if kind == "single_excitation_random":
        k = int(rng.integers(0, n))
        basis_index = 1 << (n - 1 - k)
        psi = np.zeros(dim, dtype=complex)
        psi[basis_index] = 1.0
        return psi
    if kind == "random_pure":
        z = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        return normalize_state(z)
    raise ValueError(f"Unknown init_state: {kind}")

@dataclass
class SimConfig:
    n: int
    steps: int
    dt: float
    graph_type: str
    seed: int
    hz_scale: float
    J0: float
    J_min: float
    J_max: float
    eta_up: float
    eta_down: float
    target_commit: float
    init_state: str

def run_sim(cfg: SimConfig) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    edges = make_graph_edges(cfg.n, cfg.graph_type, rng)
    cache = build_operator_cache(cfg.n)
    hz = rng.uniform(-cfg.hz_scale, cfg.hz_scale, size=cfg.n)
    J = {e: float(cfg.J0 * rng.uniform(0.85, 1.15)) for e in edges}
    psi = make_initial_state(cfg.n, cfg.init_state, cfg.seed)
    jw_ops = {(i, j): jw_string_ops(cfg.n, i, j) for i in range(cfg.n) for j in range(i + 1, cfg.n)}

    edge_order = edges[:]
    jw_pairs = list(jw_ops.keys())

    J_series, commit_series, coh_series, occ_series, jw_series = [], [], [], [], []

    for _ in range(cfg.steps):
        H = build_hamiltonian(cfg.n, cache, hz, J)
        U = matrix_exp_hermitian(H, cfg.dt)
        psi = normalize_state(U @ psi)

        occ = node_occupations(psi, cache)
        coh = {e: edge_coherence(psi, cache, e[0], e[1]) for e in edge_order}
        commit = commitment_scores(edge_order, coh, cfg.n)

        J_series.append([J[e] for e in edge_order])
        coh_series.append([coh[e] for e in edge_order])
        commit_series.append([commit[e] for e in edge_order])
        occ_series.append(occ.tolist())
        jw_series.append([jw_amplitude(psi, jw_ops, i, j) for (i, j) in jw_pairs])

        J = update_couplings(J, commit, cfg.J_min, cfg.J_max, cfg.eta_up, cfg.eta_down, cfg.target_commit)

    J_arr = np.array(J_series, dtype=float)
    coh_arr = np.array(coh_series, dtype=float)
    commit_arr = np.array(commit_series, dtype=float)
    occ_arr = np.array(occ_series, dtype=float)
    jw_arr = np.array(jw_series, dtype=float)

    late = max(0, cfg.steps // 2)
    J_late = J_arr[late:]
    coh_late = coh_arr[late:]
    commit_late = commit_arr[late:]
    occ_late = occ_arr[late:]
    jw_late = jw_arr[late:]

    edge_summary = []
    for k, e in enumerate(edge_order):
        edge_summary.append({
            "edge": list(e),
            "mean_J": float(J_late[:, k].mean() if len(J_late) else J_arr[:, k].mean()),
            "std_J": float(J_late[:, k].std() if len(J_late) else J_arr[:, k].std()),
            "mean_commit": float(commit_late[:, k].mean() if len(commit_late) else commit_arr[:, k].mean()),
            "mean_coherence": float(coh_late[:, k].mean() if len(coh_late) else coh_arr[:, k].mean()),
        })
    edge_summary.sort(key=lambda d: d["mean_J"], reverse=True)

    jw_summary = []
    for k, (i, j) in enumerate(jw_pairs):
        arr = jw_late[:, k] if len(jw_late) else jw_arr[:, k]
        jw_summary.append({
            "pair": [i, j],
            "length": int(j - i),
            "mean_jw_amp": float(arr.mean()),
            "std_jw_amp": float(arr.std()),
        })
    jw_summary.sort(key=lambda d: d["mean_jw_amp"], reverse=True)

    occ_mean = occ_late.mean(axis=0) if len(occ_late) else occ_arr.mean(axis=0)
    occ_std = occ_late.std(axis=0) if len(occ_late) else occ_arr.std(axis=0)
    edge_std = np.array([r["std_J"] for r in edge_summary], dtype=float)
    jw_std = np.array([r["std_jw_amp"] for r in jw_summary], dtype=float)

    return {
        "config": cfg.__dict__,
        "edges": [list(e) for e in edge_order],
        "jw_pairs": [list(p) for p in jw_pairs],
        "time_series": {
            "J": J_series,
            "commitment": commit_series,
            "coherence": coh_series,
            "occupations": occ_series,
            "jw_amplitudes": jw_series,
        },
        "summary": {
            "edge_summary": edge_summary,
            "node_mean_occupations": [float(x) for x in occ_mean.tolist()],
            "node_std_occupations": [float(x) for x in occ_std.tolist()],
            "jw_summary_top20": jw_summary[:20],
            "edge_persistence_proxy": float(np.mean(1.0 / (1.0 + edge_std))) if len(edge_std) else 0.0,
            "occupation_persistence_proxy": float(np.mean(1.0 / (1.0 + occ_std))) if len(occ_std) else 0.0,
            "jw_persistence_proxy": float(np.mean(1.0 / (1.0 + jw_std))) if len(jw_std) else 0.0,
        },
    }

def pretty_report(result: Dict[str, object]) -> str:
    cfg = result["config"]
    summ = result["summary"]
    edge_top = summ["edge_summary"][: min(10, len(summ["edge_summary"]))]
    jw_top = summ["jw_summary_top20"][: min(10, len(summ["jw_summary_top20"]))]
    lines = []
    lines.append("=" * 100)
    lines.append("MANY-BODY COMMITTED NETWORK + JW TEST (v1)")
    lines.append("-" * 100)
    lines.append(f"N={cfg['n']}  steps={cfg['steps']}  dt={cfg['dt']}  graph={cfg['graph_type']}  init={cfg['init_state']}")
    lines.append(f"J0={cfg['J0']}  J_range=[{cfg['J_min']}, {cfg['J_max']}]  eta_up={cfg['eta_up']}  eta_down={cfg['eta_down']}  target_commit={cfg['target_commit']}")
    lines.append(f"Persistence proxies: edge={summ['edge_persistence_proxy']:.4f}  occupation={summ['occupation_persistence_proxy']:.4f}  JW={summ['jw_persistence_proxy']:.4f}")
    lines.append("-" * 100)
    lines.append("Top late-time edges by mean J:")
    for row in edge_top:
        lines.append(f"  edge={row['edge']}  mean_J={row['mean_J']:.4f}  std_J={row['std_J']:.4f}  mean_commit={row['mean_commit']:.4f}  mean_coh={row['mean_coherence']:.4f}")
    lines.append("-" * 100)
    lines.append("Top late-time JW strings:")
    for row in jw_top:
        lines.append(f"  pair={row['pair']}  len={row['length']}  mean_amp={row['mean_jw_amp']:.4f}  std_amp={row['std_jw_amp']:.4f}")
    lines.append("-" * 100)
    lines.append(f"Late-time node mean occupations: {[round(x, 4) for x in summ['node_mean_occupations']]}")
    lines.append(f"Late-time node std occupations:  {[round(x, 4) for x in summ['node_std_occupations']]}")
    return "\n".join(lines)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Many-body committed-network dynamics with Jordan-Wigner analysis.")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--graph-type", choices=["chain", "ring", "ring_plus_chords", "erdos"], default="ring_plus_chords")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hz-scale", type=float, default=0.18)
    ap.add_argument("--J0", type=float, default=0.14)
    ap.add_argument("--J-min", type=float, default=0.02)
    ap.add_argument("--J-max", type=float, default=0.35)
    ap.add_argument("--eta-up", type=float, default=0.035)
    ap.add_argument("--eta-down", type=float, default=0.020)
    ap.add_argument("--target-commit", type=float, default=0.22)
    ap.add_argument("--init-state", choices=["single_excitation_center", "single_excitation_random", "random_pure"], default="single_excitation_center")
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    cfg = SimConfig(
        n=int(args.n), steps=int(args.steps), dt=float(args.dt), graph_type=str(args.graph_type), seed=int(args.seed),
        hz_scale=float(args.hz_scale), J0=float(args.J0), J_min=float(args.J_min), J_max=float(args.J_max),
        eta_up=float(args.eta_up), eta_down=float(args.eta_down), target_commit=float(args.target_commit),
        init_state=str(args.init_state),
    )
    result = run_sim(cfg)
    print(pretty_report(result))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON: {args.json_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
