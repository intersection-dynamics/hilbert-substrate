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
from sklearn.linear_model import LogisticRegression


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


def active_triangles(active_nodes: List[int], active_edges: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    edge_set = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    out = []
    for a, b, c in combinations(active_nodes, 3):
        if (min(a, b), max(a, b)) in edge_set and (min(b, c), max(b, c)) in edge_set and (min(a, c), max(a, c)) in edge_set:
            out.append((a, b, c))
    return out


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
    eval_every: int
    lookahead_steps: int
    fission_fraction: float
    persist_fraction: float
    json_out: str = ""


def candidate_features(
    psi: np.ndarray,
    rho: np.ndarray,
    active_nodes: List[int],
    active_edges: List[Tuple[int, int]],
    edge_strengths: Dict[Tuple[int, int], float],
    dims: List[int],
    cache: Dict[Tuple[int, int], csr_matrix],
) -> List[Dict[str, object]]:
    existing_edges = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    coords = spectral_1d_embedding(active_nodes, active_edges, edge_strengths)
    adj = {i: [] for i in active_nodes}
    for i, j in active_edges:
        adj[i].append(j); adj[j].append(i)

    rows = []
    for i, j in combinations(active_nodes, 2):
        if (min(i, j), max(i, j)) not in existing_edges:
            continue
        rho_ab = partial_trace_keep(rho, dims, [i, j])
        rho_a = partial_trace_keep(rho, dims, [i])
        rho_b = partial_trace_keep(rho, dims, [j])
        mi = mutual_information(rho_ab, rho_a, rho_b)
        corr = pair_su3_correlator_strength(psi, cache, i, j)
        sa = von_neumann_entropy(rho_a)
        sb = von_neumann_entropy(rho_b)
        pair_entropy = von_neumann_entropy(rho_ab)
        coord_gap = abs(coords.get(i, 0.0) - coords.get(j, 0.0))
        common_nbrs = sorted(set(adj[i]).intersection(adj[j]))
        cmi_mean = 0.0
        cmi_max = 0.0
        if common_nbrs:
            cmis = []
            for k in common_nbrs:
                rho_ijk = partial_trace_keep(rho, dims, [i, k, j])
                rho_ik = partial_trace_keep(rho, dims, [i, k])
                rho_kj = partial_trace_keep(rho, dims, [k, j])
                rho_k = partial_trace_keep(rho, dims, [k])
                cmis.append(conditional_mutual_information(rho_ijk, rho_ik, rho_kj, rho_k))
            cmi_mean = float(np.mean(cmis))
            cmi_max = float(np.max(cmis))
        daughter_count = 0
        for node in active_nodes:
            if node in (i, j):
                continue
            if (min(i, node), max(i, node)) in existing_edges and (min(j, node), max(j, node)) in existing_edges:
                daughter_count += 1
        rows.append({
            "pair": [i, j],
            "mi": float(mi),
            "corr": float(corr),
            "sa": float(sa),
            "sb": float(sb),
            "pair_entropy": float(pair_entropy),
            "coord_gap": float(coord_gap),
            "common_nbr_count": int(len(common_nbrs)),
            "cmi_mean": float(cmi_mean),
            "cmi_max": float(cmi_max),
            "daughter_count": int(daughter_count),
            "score": float(mi * corr * (1.0 + cmi_mean) * (1.0 + 0.25 * daughter_count)),
        })
    rows.sort(key=lambda d: d["score"], reverse=True)
    return rows


def choose_births(rows: List[Dict[str, object]], dormant_nodes: List[int], fission_fraction: float) -> List[Tuple[List[int], int]]:
    if not rows or not dormant_nodes:
        return []
    n_births = max(0, min(len(dormant_nodes), int(np.floor(fission_fraction * len(rows)))))
    if n_births <= 0:
        n_births = 1 if rows else 0
    chosen = []
    used_pairs = set()
    dormant_iter = list(dormant_nodes[:n_births])
    idx = 0
    for row in rows:
        p = tuple(row["pair"])
        if p in used_pairs:
            continue
        chosen.append((row["pair"], dormant_iter[idx]))
        used_pairs.add(p)
        idx += 1
        if idx >= len(dormant_iter):
            break
    return chosen


def spawn_births(
    births: List[Tuple[List[int], int]],
    active_nodes: List[int],
    dormant_nodes: List[int],
    active_edges: List[Tuple[int, int]],
    edge_strengths: Dict[Tuple[int, int], float],
    local_coeffs: Dict[int, np.ndarray],
    spawn_pair_scale: float,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    events = []
    for parents, new_node in births:
        i, j = parents
        if new_node not in dormant_nodes:
            continue
        dormant_nodes.remove(new_node)
        active_nodes.append(new_node)
        active_nodes.sort()
        e1 = (min(i, new_node), max(i, new_node))
        e2 = (min(j, new_node), max(j, new_node))
        if e1 not in active_edges:
            active_edges.append(e1)
        if e2 not in active_edges:
            active_edges.append(e2)
        active_edges.sort()
        edge_strengths[e1] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
        edge_strengths[e2] = float(rng.uniform(0.7 * spawn_pair_scale, 1.3 * spawn_pair_scale))
        local_coeffs[new_node] = rng.uniform(-0.5, 0.5, size=8)
        events.append({"parents": [i, j], "new_node": new_node, "spawn_links": [list(e1), list(e2)]})
    return events


def classify_persistence(events: List[Dict[str, object]], active_nodes: List[int], active_edges: List[Tuple[int, int]], rho: np.ndarray, dims: List[int], persist_fraction: float) -> List[Dict[str, object]]:
    existing_edges = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    out = []
    for evt in events:
        i, j = evt["parents"]
        n = evt["new_node"]
        survived_links = int((min(i, n), max(i, n)) in existing_edges) + int((min(j, n), max(j, n)) in existing_edges)
        rho_n = partial_trace_keep(rho, dims, [n])
        sn = von_neumann_entropy(rho_n)
        persistence_score = float(0.5 * survived_links + sn)
        label = "persistent" if persistence_score >= persist_fraction else "remerge_prone"
        out.append({
            **evt,
            "survived_links": survived_links,
            "new_node_entropy": float(sn),
            "persistence_score": persistence_score,
            "label": label,
        })
    return out


def run_sim(cfg: SimConfig) -> Dict[str, object]:
    if cfg.n_init >= cfg.n_max:
        raise ValueError("Require n_init < n_max.")
    if cfg.n_max > 9:
        raise ValueError("Keep n_max <= 9 for this sparse qutrit birth-rule derivation toy.")

    rng = np.random.default_rng(cfg.seed)
    dims = [3] * cfg.n_max
    cache = build_site_generator_cache(cfg.n_max)

    active_nodes = list(range(cfg.n_init))
    dormant_nodes = list(range(cfg.n_init, cfg.n_max))
    active_edges = [(i, i + 1) for i in range(cfg.n_init - 1)]
    local_coeffs = {i: (rng.uniform(-cfg.local_scale, cfg.local_scale, size=8) if i < cfg.n_init else np.zeros(8, dtype=float)) for i in range(cfg.n_max)}
    edge_strengths = {e: float(rng.uniform(0.6 * cfg.pair_scale, 1.4 * cfg.pair_scale)) for e in active_edges}

    psi = initial_state_qutrits(cfg.n_max, cfg.n_init, cfg.seed + 1000)

    candidate_dataset = []
    birth_events = []
    derived_rule = None

    for step in range(cfg.total_steps):
        H = build_hamiltonian(cfg.n_max, active_nodes, active_edges, cache, local_coeffs, edge_strengths)
        psi = expm_multiply((-1j * cfg.dt) * H, psi)
        psi = normalize_state(np.asarray(psi))

        if ((step + 1) % cfg.eval_every) == 0:
            rho = pure_density(psi)
            rows = candidate_features(psi, rho, active_nodes, active_edges, edge_strengths, dims, cache)
            births = choose_births(rows, dormant_nodes, cfg.fission_fraction)
            spawn_events = spawn_births(births, active_nodes, dormant_nodes, active_edges, edge_strengths, local_coeffs, cfg.spawn_pair_scale, rng)

            # advance lookahead window before classifying whether births persist
            for _ in range(cfg.lookahead_steps):
                H2 = build_hamiltonian(cfg.n_max, active_nodes, active_edges, cache, local_coeffs, edge_strengths)
                psi = expm_multiply((-1j * cfg.dt) * H2, psi)
                psi = normalize_state(np.asarray(psi))

            rho2 = pure_density(psi)
            labeled_births = classify_persistence(spawn_events, active_nodes, active_edges, rho2, dims, cfg.persist_fraction)
            birth_by_pair = {tuple(evt["parents"]): evt for evt in labeled_births}

            for row in rows:
                pair = tuple(row["pair"])
                label = birth_by_pair[pair]["label"] if pair in birth_by_pair else "no_birth"
                candidate_dataset.append({**row, "step": step + 1, "outcome": label})

            birth_events.extend(labeled_births)

    # derive empirical rule
    X = []
    y = []
    feature_names = ["mi", "corr", "sa", "sb", "pair_entropy", "coord_gap", "common_nbr_count", "cmi_mean", "cmi_max", "daughter_count"]
    for row in candidate_dataset:
        if row["outcome"] == "no_birth":
            continue
        X.append([row[k] for k in feature_names])
        y.append(1 if row["outcome"] == "persistent" else 0)

    if len(set(y)) >= 2 and len(y) >= 6:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(np.array(X, dtype=float), np.array(y, dtype=int))
        derived_rule = {
            "feature_names": feature_names,
            "intercept": float(clf.intercept_[0]),
            "coefficients": {feature_names[i]: float(clf.coef_[0][i]) for i in range(len(feature_names))},
            "n_labeled_births": len(y),
            "n_persistent": int(sum(y)),
            "n_remerge_prone": int(len(y) - sum(y)),
        }
    else:
        derived_rule = {
            "feature_names": feature_names,
            "intercept": None,
            "coefficients": {},
            "n_labeled_births": len(y),
            "n_persistent": int(sum(y)) if y else 0,
            "n_remerge_prone": int(len(y) - sum(y)) if y else 0,
            "note": "Insufficient class diversity to fit logistic rule robustly."
        }

    summary = {
        "n_candidates": len(candidate_dataset),
        "n_birth_events": len(birth_events),
        "n_persistent_births": int(sum(1 for e in birth_events if e["label"] == "persistent")),
        "n_remerge_prone_births": int(sum(1 for e in birth_events if e["label"] == "remerge_prone")),
        "active_nodes_final": len(active_nodes),
        "active_edges_final": len(active_edges),
    }

    return {
        "config": cfg.__dict__,
        "candidate_dataset": candidate_dataset,
        "birth_events": birth_events,
        "derived_rule": derived_rule,
        "summary": summary,
        "active_nodes_final": active_nodes,
        "active_edges_final": [list(e) for e in active_edges],
    }


def pretty_report(result: Dict[str, object]) -> str:
    cfg = result["config"]
    summ = result["summary"]
    rule = result["derived_rule"]
    lines = []
    lines.append("=" * 112)
    lines.append("DERIVE SUBSYSTEM BIRTH RULE (v1)")
    lines.append("-" * 112)
    lines.append(
        f"n_max={cfg['n_max']}  n_init={cfg['n_init']}  seed={cfg['seed']}  total_steps={cfg['total_steps']}  "
        f"dt={cfg['dt']}  eval_every={cfg['eval_every']}  lookahead_steps={cfg['lookahead_steps']}"
    )
    lines.append(
        f"fission_fraction={cfg['fission_fraction']}  persist_fraction={cfg['persist_fraction']}  "
        f"local_scale={cfg['local_scale']}  pair_scale={cfg['pair_scale']}"
    )
    lines.append("-" * 112)
    lines.append(
        f"Counts: candidates={summ['n_candidates']}  births={summ['n_birth_events']}  "
        f"persistent={summ['n_persistent_births']}  remerge_prone={summ['n_remerge_prone_births']}  "
        f"final_nodes={summ['active_nodes_final']}  final_edges={summ['active_edges_final']}"
    )
    lines.append("-" * 112)
    lines.append("Derived empirical rule:")
    if rule["intercept"] is None:
        lines.append(f"  {rule.get('note', 'No fitted rule.')}")
    else:
        lines.append(f"  logit(P[persistent]) = {rule['intercept']:.4f} + Σ c_k x_k")
        for k, v in rule["coefficients"].items():
            lines.append(f"    {k}: {v:+.4f}")
    lines.append("-" * 112)
    lines.append("Recent labeled births:")
    for evt in result["birth_events"][:10]:
        lines.append(
            f"  parents={evt['parents']}  new_node={evt['new_node']}  label={evt['label']}  "
            f"survived_links={evt['survived_links']}  new_entropy={evt['new_node_entropy']:.4f}  "
            f"persist_score={evt['persistence_score']:.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Infer an empirical subsystem birth/death rule from observed churn.")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-scale", type=float, default=0.10)
    ap.add_argument("--pair-scale", type=float, default=0.16)
    ap.add_argument("--spawn-pair-scale", type=float, default=0.12)
    ap.add_argument("--total-steps", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.20)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--lookahead-steps", type=int, default=4)
    ap.add_argument("--fission-fraction", type=float, default=0.20)
    ap.add_argument("--persist-fraction", type=float, default=0.55)
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
        eval_every=int(args.eval_every),
        lookahead_steps=int(args.lookahead_steps),
        fission_fraction=float(args.fission_fraction),
        persist_fraction=float(args.persist_fraction),
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
