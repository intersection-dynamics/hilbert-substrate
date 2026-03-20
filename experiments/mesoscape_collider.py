#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import networkx as nx
from dataclasses import dataclass
from itertools import combinations

BASIS0 = np.array([1.0, 0.0, 0.0], dtype=complex)

def get_xp(device: str):
    if device == "gpu":
        try:
            import cupy as cp
            return cp, True
        except Exception:
            pass
    return np, False

def gell_mann(xp):
    i = 1j
    out = []
    out.append(xp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, -i, 0], [i, 0, 0], [0, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, -i], [0, 0, 0], [i, 0, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=xp.complex128))
    out.append(xp.array([[0, 0, 0], [0, 0, -i], [0, i, 0]], dtype=xp.complex128))
    out.append((1.0 / xp.sqrt(3.0)) * xp.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=xp.complex128))
    return out

def normalize_state(psi, xp):
    n = xp.linalg.norm(psi.reshape(-1))
    if float(n) <= 1e-15:
        raise ValueError("State norm vanished.")
    return psi / n

def apply_one_body(psi, op, site, xp):
    y = xp.moveaxis(psi, site, 0)
    y = xp.tensordot(op, y, axes=([1], [0]))
    y = xp.moveaxis(y, 0, site)
    return y

def apply_two_body_samegen(psi, op, i, j, xp):
    return apply_one_body(apply_one_body(psi, op, i, xp), op, j, xp)

def apply_hamiltonian(psi, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp):
    out = xp.zeros_like(psi)
    for i in active_nodes:
        coeffs = local_coeffs[i]
        for a in range(8):
            c = float(coeffs[a])
            if c != 0.0:
                out = out + c * apply_one_body(psi, GM[a], i, xp)
    for i, j in active_edges:
        g = float(edge_strengths[(min(i, j), max(i, j))])
        if g == 0.0:
            continue
        term = xp.zeros_like(psi)
        for a in range(8):
            term = term + apply_two_body_samegen(psi, GM[a], i, j, xp)
        out = out + g * term
    return out

def rk4_step(psi, dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp):
    def f(state):
        return -1j * apply_hamiltonian(state, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
    k1 = f(psi)
    k2 = f(psi + 0.5 * dt * k1)
    k3 = f(psi + 0.5 * dt * k2)
    k4 = f(psi + dt * k3)
    psi2 = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return normalize_state(psi2, xp)

def partial_trace_keep(psi, keep, n_sites, xp):
    keep = sorted(keep)
    trace_out = [i for i in range(n_sites) if i not in keep]
    perm = keep + trace_out
    psi_perm = xp.transpose(psi, perm)
    d_keep = 3 ** len(keep)
    d_tr = 3 ** len(trace_out)
    mat = psi_perm.reshape(d_keep, d_tr)
    return mat @ xp.conjugate(mat.T)

def von_neumann_entropy(rho, xp):
    vals = xp.linalg.eigvalsh(0.5 * (rho + xp.conjugate(rho.T)))
    vals = xp.real(vals)
    vals = xp.maximum(vals, 0.0)
    s = vals.sum()
    if float(s) <= 1e-15:
        return 0.0
    vals = vals / s
    nz = vals[vals > 1e-15]
    return float((-nz * xp.log(nz)).sum())

def mutual_information_from_state(psi, i, j, n_sites, xp):
    rho_ab = partial_trace_keep(psi, [i, j], n_sites, xp)
    rho_a = partial_trace_keep(psi, [i], n_sites, xp)
    rho_b = partial_trace_keep(psi, [j], n_sites, xp)
    return float(von_neumann_entropy(rho_a, xp) + von_neumann_entropy(rho_b, xp) - von_neumann_entropy(rho_ab, xp))

def conditional_mutual_information_from_state(psi, i, k, j, n_sites, xp):
    rho_ikj = partial_trace_keep(psi, [i, k, j], n_sites, xp)
    rho_ik = partial_trace_keep(psi, [i, k], n_sites, xp)
    rho_kj = partial_trace_keep(psi, [k, j], n_sites, xp)
    rho_k = partial_trace_keep(psi, [k], n_sites, xp)
    return float(von_neumann_entropy(rho_ik, xp) + von_neumann_entropy(rho_kj, xp) - von_neumann_entropy(rho_k, xp) - von_neumann_entropy(rho_ikj, xp))

def pair_su3_correlator_strength(psi, GM, i, j, xp):
    vals = []
    for a in range(8):
        tmp = apply_two_body_samegen(psi, GM[a], i, j, xp)
        vals.append(float(xp.real(xp.vdot(psi.reshape(-1), tmp.reshape(-1)))))
    return float(xp.linalg.norm(xp.asarray(vals)))

def weighted_adjacency(active_nodes, active_edges, edge_strengths):
    idx_of = {node: k for k, node in enumerate(active_nodes)}
    W = np.zeros((len(active_nodes), len(active_nodes)), dtype=float)
    for i, j in active_edges:
        a, b = idx_of[i], idx_of[j]
        w = float(edge_strengths[(min(i, j), max(i, j))])
        W[a, b] = w
        W[b, a] = w
    return W, idx_of

def spectral_1d_embedding(active_nodes, active_edges, edge_strengths):
    if len(active_nodes) == 1:
        return {active_nodes[0]: 0.0}
    W, idx_of = weighted_adjacency(active_nodes, active_edges, edge_strengths)
    deg = np.sum(W, axis=1)
    L = np.diag(deg) - W
    vals, vecs = np.linalg.eigh(L)
    xs = np.real(vecs[:, 1]) if len(vals) >= 2 else np.zeros(len(active_nodes))
    xs = xs - np.mean(xs)
    s = np.std(xs)
    if s > 1e-12:
        xs = xs / s
    return {node: float(xs[idx_of[node]]) for node in active_nodes}

def active_triangles(active_nodes, active_edges):
    edge_set = set((min(i, j), max(i, j)) for i, j in active_edges)
    out = []
    for a, b, c in combinations(active_nodes, 3):
        if (min(a, b), max(a, b)) in edge_set and (min(b, c), max(b, c)) in edge_set and (min(a, c), max(a, c)) in edge_set:
            out.append((a, b, c))
    return out

@dataclass
class SimConfig:
    n_max: int
    seed: int
    local_scale: float
    pair_scale: float
    spawn_pair_scale: float
    total_steps: int
    collision_step: int
    collision_strength: float
    dt: float
    eval_every: int
    lookahead_windows: int
    settling_windows: int
    fission_fraction: float
    candidate_fraction: float
    birth_score_floor: float
    decay_mi_threshold: float
    decay_corr_threshold: float
    neighborhood_bonus_weight: float
    shell_bonus_weight: float
    mi_survival_floor: float
    corr_survival_floor: float
    persist_windows_required: int
    persist_entropy_threshold: float
    persist_mean_mi_threshold: float
    persist_triangle_threshold: int
    device: str = "gpu"
    json_out: str = "gpu_mesoscape_collider_results.json"

def candidate_features(psi, active_nodes, active_edges, edge_strengths, n_sites, GM, xp):
    existing_edges = set((min(i, j), max(i, j)) for (i, j) in active_edges)
    coords = spectral_1d_embedding(active_nodes, active_edges, edge_strengths)
    adj = {i: [] for i in active_nodes}
    for i, j in active_edges:
        adj[i].append(j)
        adj[j].append(i)
    triangles = active_triangles(active_nodes, active_edges)

    rows = []
    for i, j in combinations(active_nodes, 2):
        if (min(i, j), max(i, j)) not in existing_edges:
            continue
        rho_ab = partial_trace_keep(psi, [i, j], n_sites, xp)
        rho_a = partial_trace_keep(psi, [i], n_sites, xp)
        rho_b = partial_trace_keep(psi, [j], n_sites, xp)
        mi = float(von_neumann_entropy(rho_a, xp) + von_neumann_entropy(rho_b, xp) - von_neumann_entropy(rho_ab, xp))
        corr = pair_su3_correlator_strength(psi, GM, i, j, xp)
        
        # If MI is functionally zero (unconnected components), skip scoring
        if mi < 1e-10:
            continue
            
        common_nbrs = sorted(set(adj[i]).intersection(adj[j]))
        cmi_mean = 0.0
        if common_nbrs:
            cmis = [conditional_mutual_information_from_state(psi, i, k, j, n_sites, xp) for k in common_nbrs]
            cmi_mean = float(sum(cmis) / len(cmis))
        daughter_count = 0
        shell_triangle_count = 0
        for node in active_nodes:
            if node in (i, j):
                continue
            if (min(i, node), max(i, node)) in existing_edges and (min(j, node), max(j, node)) in existing_edges:
                daughter_count += 1
        for tri in triangles:
            if i in tri and j in tri:
                shell_triangle_count += 1
        score = float(mi * corr * (1.0 + cmi_mean) * (1.0 + 0.20 * daughter_count) * (1.0 + 0.10 * shell_triangle_count))
        rows.append({
            "pair": [i, j],
            "score": score,
        })
    rows.sort(key=lambda d: d["score"], reverse=True)
    return rows

def choose_candidate_births(rows, dormant_nodes, candidate_fraction, fission_fraction, birth_score_floor):
    if not rows or not dormant_nodes:
        return []
    n_considered = max(1, int(np.ceil(candidate_fraction * len(rows))))
    considered = [r for r in rows[:n_considered] if r["score"] >= birth_score_floor]
    if not considered:
        return []
    n_births = max(1, int(np.floor(fission_fraction * len(considered))))
    n_births = min(n_births, len(considered), len(dormant_nodes))
    return [(considered[idx], dormant_nodes[idx]) for idx in range(n_births)]

def spawn_births(chosen, active_nodes, dormant_nodes, active_edges, edge_strengths, local_coeffs, spawn_pair_scale, rng):
    events = []
    for row, new_node in chosen:
        i, j = row["pair"]
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
        events.append({"parents": [i, j], "new_node": new_node})
    return events

def run_collider_sim(cfg):
    xp, using_gpu = get_xp(cfg.device)
    GM = gell_mann(xp)
    rng = np.random.default_rng(cfg.seed)

    # ---------------------------------------------------------
    # DUAL SEED INITIALIZATION
    # ---------------------------------------------------------
    # We initialize TWO disjoint pairs to act as Particle A and Particle B
    active_nodes = [0, 1, 2, 3]
    dormant_nodes = list(range(4, cfg.n_max))
    active_edges = [(0, 1), (2, 3)]
    
    local_coeffs = {i: (rng.uniform(-cfg.local_scale, cfg.local_scale, size=8) if i < 4 else np.zeros(8, dtype=float)) for i in range(cfg.n_max)}
    edge_strengths = {(0, 1): float(rng.uniform(0.6 * cfg.pair_scale, 1.4 * cfg.pair_scale)), 
                      (2, 3): float(rng.uniform(0.6 * cfg.pair_scale, 1.4 * cfg.pair_scale))}

    local_states = []
    for i in range(cfg.n_max):
        if i < 4:
            z = rng.normal(size=3) + 1j * rng.normal(size=3)
            z = z / np.linalg.norm(z)
            local_states.append(xp.asarray(z, dtype=xp.complex128))
        else:
            local_states.append(xp.asarray(BASIS0, dtype=xp.complex128))

    psi = local_states[0]
    for v in local_states[1:]:
        psi = xp.kron(psi, v)
    psi = psi.reshape((3,) * cfg.n_max)
    psi = normalize_state(psi, xp)

    snapshots = []
    collision_occurred = False
    collision_edge = None
    
    print(f"Starting Collider Simulation. Device: {'GPU' if using_gpu else 'CPU'}")
    print(f"Phase 1: Incubation. Particles A [0,1] and B [2,3] are growing independently...")

    step = 0
    while step < cfg.total_steps:
        psi = rk4_step(psi, cfg.dt, active_nodes, active_edges, local_coeffs, edge_strengths, GM, xp)
        step += 1
        
        # ---------------------------------------------------------
        # COLLISION EVENT
        # ---------------------------------------------------------
        if step == cfg.collision_step and not collision_occurred:
            print(f"\n>>> TARGET STEP {step} REACHED: INITIATING COLLISION <<<")
            
            # Find the distinct clusters
            G = nx.Graph()
            G.add_nodes_from(active_nodes)
            G.add_edges_from(active_edges)
            components = list(nx.connected_components(G))
            
            if len(components) >= 2:
                # Sort components by size
                components.sort(key=len, reverse=True)
                cluster_A = list(components[0])
                cluster_B = list(components[1])
                
                # Find the highest degree "shell" node in each cluster to act as the point of impact
                degrees = dict(G.degree())
                impact_node_A = max(cluster_A, key=lambda n: degrees[n])
                impact_node_B = max(cluster_B, key=lambda n: degrees[n])
                
                collision_edge = (min(impact_node_A, impact_node_B), max(impact_node_A, impact_node_B))
                active_edges.append(collision_edge)
                active_edges.sort()
                
                # Slam them together with massive energy
                edge_strengths[collision_edge] = cfg.collision_strength
                collision_occurred = True
                print(f"IMPACT: Connected Cluster A (Size {len(cluster_A)}) to Cluster B (Size {len(cluster_B)}) via edge {collision_edge} with strength {cfg.collision_strength}")
                print(f"Phase 2: Post-Collision Observation...\n")
            else:
                print("Warning: Substrate already fused before collision step!")

        # ---------------------------------------------------------
        # BIRTH / DECAY CYCLE
        # ---------------------------------------------------------
        if (step % cfg.eval_every) == 0:
            rows = candidate_features(psi, active_nodes, active_edges, edge_strengths, cfg.n_max, GM, xp)
            chosen = choose_candidate_births(rows, dormant_nodes, cfg.candidate_fraction, cfg.fission_fraction, cfg.birth_score_floor)
            spawn_births(chosen, active_nodes, dormant_nodes, active_edges, edge_strengths, local_coeffs, cfg.spawn_pair_scale, rng)

            snapshots.append({
                "step": step,
                "active_nodes": list(active_nodes),
                "active_edges": [list(e) for e in active_edges],
                "n_components": nx.number_connected_components(nx.Graph(active_edges)),
                "collision_active": collision_occurred
            })
            
            print(f"[Step {step:4d}] Nodes: {len(active_nodes):2d} | Edges: {len(active_edges):2d} | Components: {snapshots[-1]['n_components']}")

    result = {
        "config": cfg.__dict__,
        "collision_edge": list(collision_edge) if collision_edge else None,
        "snapshots": snapshots,
        "active_nodes_final": active_nodes,
        "active_edges_final": [list(e) for e in active_edges],
    }
    
    return result

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--n-max", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=800)
    ap.add_argument("--collision-step", type=int, default=400, help="Step to slam the two clusters together")
    ap.add_argument("--collision-strength", type=float, default=2.5, help="Interaction strength of the collision (Standard pair scale is ~0.16)")
    ap.add_argument("--json-out", type=str, default="mesoscape_collider_results.json")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = SimConfig(
        n_max=args.n_max,
        n_init=4,  # Forces 2 pairs: [0,1] and [2,3]
        seed=args.seed,
        local_scale=0.10,
        pair_scale=0.16,
        spawn_pair_scale=0.12,
        total_steps=args.total_steps,
        collision_step=args.collision_step,
        collision_strength=args.collision_strength,
        dt=0.2,
        eval_every=12,
        lookahead_windows=3,
        settling_windows=2,
        fission_fraction=0.30,
        candidate_fraction=0.45,
        birth_score_floor=0.015,
        decay_mi_threshold=0.05,
        decay_corr_threshold=0.07,
        neighborhood_bonus_weight=0.18,
        shell_bonus_weight=0.20,
        mi_survival_floor=0.076,
        corr_survival_floor=0.086,
        persist_windows_required=2,
        persist_entropy_threshold=0.06,
        persist_mean_mi_threshold=0.07,
        persist_triangle_threshold=1,
        device=args.device,
        json_out=args.json_out,
    )
    
    result = run_collider_sim(cfg)
    
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved JSON: {args.json_out}")

if __name__ == "__main__":
    raise SystemExit(main())