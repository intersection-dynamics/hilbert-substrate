# relational_ecology_v1.py

r"""
Large-N relational ecology toy for HSF-style exploration.

What it does
------------
- Builds a sparse graph of many subsystems.
- Each node has a small local state vector.
- Each edge carries a small "link memory" vector.
- Repeated local interactions:
    node <-> node via edge
- Tracks:
    - activity per node
    - active link rank / bandwidth proxy
    - cluster formation
    - persistence of hot regions
    - path length / geometry proxies

This is NOT exact many-body quantum evolution.
It is a scalable ecology toy designed to let many subsystems "run loose"
and show emergent structure patterns on a normal computer.

Outputs
-------
summary.json
timeseries.csv
activity_hist.png
link_rank_hist.png
largest_cluster.png
path_length.png

Example
-------
python relational_ecology_v1.py --outdir ecology_out --nodes 128 --steps 2000 --seed 0
"""

from typing import Dict
import argparse
import csv
import json
import math
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows, path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ============================================================
# Graph construction
# ============================================================

def build_sparse_graph(n: int, degree: int, rng: np.random.Generator):
    """
    Start from a ring, then add random edges until target average degree.
    """
    adj = [set() for _ in range(n)]

    # ring backbone
    for i in range(n):
        j = (i + 1) % n
        adj[i].add(j)
        adj[j].add(i)

    target_edges = max(n, int(round(n * degree / 2)))
    current_edges = n

    attempts = 0
    while current_edges < target_edges and attempts < 20 * target_edges:
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        attempts += 1
        if a == b or b in adj[a]:
            continue
        adj[a].add(b)
        adj[b].add(a)
        current_edges += 1

    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))

    return adj, edges


# ============================================================
# Core model
# ============================================================

class RelationalEcology:
    def __init__(
        self,
        n_nodes: int,
        state_dim: int,
        link_dim: int,
        degree: int,
        seed: int,
        node_mix: float,
        link_mix: float,
        memory_decay: float,
        activation_bias: float,
    ):
        self.rng = np.random.default_rng(seed)
        self.n = n_nodes
        self.d = state_dim
        self.k = link_dim
        self.node_mix = node_mix
        self.link_mix = link_mix
        self.memory_decay = memory_decay
        self.activation_bias = activation_bias

        self.adj, self.edges = build_sparse_graph(n_nodes, degree, self.rng)

        # Node states
        self.node = self.rng.normal(size=(self.n, self.d))
        self.node = np.array([normalize(v) for v in self.node], dtype=float)

        # Link memories and stats
        self.edge_index = {}
        self.link = np.zeros((len(self.edges), self.k), dtype=float)
        self.link_energy = np.zeros(len(self.edges), dtype=float)
        self.link_samples = [[] for _ in range(len(self.edges))]

        for ei, (a, b) in enumerate(self.edges):
            self.edge_index[(a, b)] = ei
            self.edge_index[(b, a)] = ei
            self.link[ei] = normalize(self.rng.normal(size=self.k))

        # Node stats
        self.node_activity = np.zeros(self.n, dtype=float)
        self.node_persistence = np.zeros(self.n, dtype=float)

        # Cached diagnostics
        self.last_active_edge_mask = np.zeros(len(self.edges), dtype=bool)

    def edge_id(self, a: int, b: int) -> int:
        return self.edge_index[(a, b)]

    def interact_edge(self, ei: int):
        a, b = self.edges[ei]

        va = self.node[a]
        vb = self.node[b]
        lm = self.link[ei]

        # Project endpoint relation into link space via fixed random map-like slicing
        # Cheap proxy: use first k components from pair features
        pair_features = np.concatenate([
            va[:self.k],
            vb[:self.k],
            0.5 * (va[:self.k] + vb[:self.k]),
        ])
        rel_signal = pair_features[:self.k].copy()
        rel_signal += 0.5 * (vb[:self.k] - va[:self.k])

        # Activation probability based on mismatch + stored memory
        mismatch = 1.0 - cosine(va, vb)
        memory_align = 0.5 * (cosine(lm, rel_signal) + 1.0)
        p_fire = sigmoid(3.0 * mismatch + 1.5 * memory_align - self.activation_bias)

        fired = self.rng.random() < p_fire
        if not fired:
            # quiet decay
            self.link[ei] *= (1.0 - self.memory_decay)
            return False

        # Update link memory
        new_link = (1.0 - self.link_mix) * lm + self.link_mix * rel_signal
        new_link = normalize(new_link)
        self.link[ei] = new_link

        # Update nodes through the link
        link_back = np.zeros(self.d, dtype=float)
        link_back[:self.k] = new_link

        va_new = normalize((1.0 - self.node_mix) * va + self.node_mix * (0.65 * vb + 0.35 * link_back))
        vb_new = normalize((1.0 - self.node_mix) * vb + self.node_mix * (0.65 * va + 0.35 * link_back))

        self.node[a] = va_new
        self.node[b] = vb_new

        # Stats
        strength = float(np.linalg.norm(rel_signal))
        self.link_energy[ei] = 0.95 * self.link_energy[ei] + 0.05 * strength
        self.node_activity[a] = 0.98 * self.node_activity[a] + 0.02 * 1.0
        self.node_activity[b] = 0.98 * self.node_activity[b] + 0.02 * 1.0

        # Store link samples for rank proxy
        self.link_samples[ei].append(rel_signal.copy())
        if len(self.link_samples[ei]) > 12:
            self.link_samples[ei].pop(0)

        return True

    def step(self, updates_per_step: int):
        fired_mask = np.zeros(len(self.edges), dtype=bool)
        chosen = self.rng.integers(0, len(self.edges), size=updates_per_step)

        for ei in chosen:
            fired = self.interact_edge(int(ei))
            if fired:
                fired_mask[int(ei)] = True

        # Persistence: whether node stays locally "hot"
        hot = self.node_activity > np.quantile(self.node_activity, 0.8)
        self.node_persistence = 0.97 * self.node_persistence + 0.03 * hot.astype(float)

        self.last_active_edge_mask = fired_mask

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def link_rank_proxy(self, ei: int, sv_thresh: float = 0.15) -> int:
        samples = self.link_samples[ei]
        if len(samples) < 2:
            return 0
        X = np.array(samples, dtype=float)
        X = X - X.mean(axis=0, keepdims=True)
        if np.linalg.norm(X) < 1e-12:
            return 0
        s = np.linalg.svd(X, compute_uv=False)
        if len(s) == 0 or s[0] < 1e-12:
            return 0
        rel = s / s[0]
        return int(np.sum(rel > sv_thresh))

    def all_link_ranks(self) -> np.ndarray:
        out = np.zeros(len(self.edges), dtype=int)
        for ei in range(len(self.edges)):
            out[ei] = self.link_rank_proxy(ei)
        return out

    def active_subgraph_components(self, edge_energy_thresh: float):
        active_nodes = set()
        active_adj = {i: set() for i in range(self.n)}

        for ei, (a, b) in enumerate(self.edges):
            if self.link_energy[ei] >= edge_energy_thresh:
                active_nodes.add(a)
                active_nodes.add(b)
                active_adj[a].add(b)
                active_adj[b].add(a)

        comps = []
        unseen = set(active_nodes)
        while unseen:
            root = next(iter(unseen))
            q = deque([root])
            unseen.remove(root)
            comp = [root]
            while q:
                u = q.popleft()
                for v in active_adj[u]:
                    if v in unseen:
                        unseen.remove(v)
                        q.append(v)
                        comp.append(v)
            comps.append(comp)

        comps.sort(key=len, reverse=True)
        return comps

    def mean_active_shortest_path(self, edge_energy_thresh: float) -> float:
        active_adj = {i: set() for i in range(self.n)}
        active_nodes = set()

        for ei, (a, b) in enumerate(self.edges):
            if self.link_energy[ei] >= edge_energy_thresh:
                active_adj[a].add(b)
                active_adj[b].add(a)
                active_nodes.add(a)
                active_nodes.add(b)

        nodes = list(active_nodes)
        if len(nodes) < 2:
            return 0.0

        # sample a subset for speed
        sample_nodes = nodes[: min(24, len(nodes))]
        dists = []

        for src in sample_nodes:
            dist = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in active_adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for dst in sample_nodes:
                if dst > src and dst in dist:
                    dists.append(dist[dst])

        if not dists:
            return 0.0
        return float(np.mean(dists))

    def snapshot_metrics(self, edge_energy_quantile: float = 0.75) -> Dict[str, float]:
        ranks = self.all_link_ranks()
        thresh = float(np.quantile(self.link_energy, edge_energy_quantile))
        comps = self.active_subgraph_components(thresh)
        largest_cluster = len(comps[0]) if comps else 0
        n_clusters = len(comps)
        mean_path = self.mean_active_shortest_path(thresh)

        hot_nodes = int(np.sum(self.node_activity > np.quantile(self.node_activity, 0.8)))
        persistent_nodes = int(np.sum(self.node_persistence > 0.5))

        return {
            "mean_node_activity": float(np.mean(self.node_activity)),
            "max_node_activity": float(np.max(self.node_activity)),
            "mean_link_energy": float(np.mean(self.link_energy)),
            "max_link_energy": float(np.max(self.link_energy)),
            "mean_link_rank": float(np.mean(ranks)),
            "max_link_rank": int(np.max(ranks)) if len(ranks) else 0,
            "largest_cluster": int(largest_cluster),
            "n_clusters": int(n_clusters),
            "mean_active_shortest_path": float(mean_path),
            "hot_nodes": int(hot_nodes),
            "persistent_nodes": int(persistent_nodes),
            "edge_energy_threshold": float(thresh),
        }


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Large-N relational ecology toy")
    p.add_argument("--outdir", type=str, default="ecology_out")
    p.add_argument("--nodes", type=int, default=128)
    p.add_argument("--state_dim", type=int, default=8)
    p.add_argument("--link_dim", type=int, default=4)
    p.add_argument("--degree", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--updates_per_step", type=int, default=64)
    p.add_argument("--snapshot_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--node_mix", type=float, default=0.14)
    p.add_argument("--link_mix", type=float, default=0.22)
    p.add_argument("--memory_decay", type=float, default=0.01)
    p.add_argument("--activation_bias", type=float, default=1.25)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    model = RelationalEcology(
        n_nodes=args.nodes,
        state_dim=args.state_dim,
        link_dim=args.link_dim,
        degree=args.degree,
        seed=args.seed,
        node_mix=args.node_mix,
        link_mix=args.link_mix,
        memory_decay=args.memory_decay,
        activation_bias=args.activation_bias,
    )

    print("=" * 72)
    print("RELATIONAL ECOLOGY (v1)")
    print("=" * 72)
    print(f"outdir: {args.outdir}")
    print(f"nodes={args.nodes}, state_dim={args.state_dim}, link_dim={args.link_dim}, degree={args.degree}")
    print(f"steps={args.steps}, updates_per_step={args.updates_per_step}, snapshot_every={args.snapshot_every}")
    print()

    rows = []
    for step in range(args.steps + 1):
        if step > 0:
            model.step(args.updates_per_step)

        if step % args.snapshot_every == 0 or step == args.steps:
            m = model.snapshot_metrics()
            row = {"step": step, **m}
            rows.append(row)
            print(
                f"step={step:>5}  "
                f"mean_rank={m['mean_link_rank']:.3f}  "
                f"largest_cluster={m['largest_cluster']:>4}  "
                f"mean_path={m['mean_active_shortest_path']:.3f}  "
                f"persistent_nodes={m['persistent_nodes']:>4}"
            )

    # Final detailed diagnostics
    final_ranks = model.all_link_ranks().tolist()
    final_metrics = model.snapshot_metrics()

    summary = {
        "params": vars(args),
        "final_metrics": final_metrics,
        "n_edges": len(model.edges),
        "final_link_ranks": final_ranks,
        "mean_node_persistence": float(np.mean(model.node_persistence)),
        "max_node_persistence": float(np.max(model.node_persistence)),
    }

    save_json(summary, os.path.join(args.outdir, "summary.json"))
    save_csv(rows, os.path.join(args.outdir, "timeseries.csv"))

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    steps = [r["step"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.hist(model.node_activity, bins=20)
    plt.xlabel("node activity")
    plt.ylabel("count")
    plt.title("Final node activity histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "activity_hist.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(final_ranks, bins=np.arange(-0.5, max(final_ranks + [1]) + 1.5, 1))
    plt.xlabel("link rank proxy")
    plt.ylabel("count")
    plt.title("Final link rank histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "link_rank_hist.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["largest_cluster"] for r in rows])
    plt.xlabel("step")
    plt.ylabel("largest active cluster")
    plt.title("Largest active cluster over time")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "largest_cluster.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["mean_active_shortest_path"] for r in rows])
    plt.xlabel("step")
    plt.ylabel("mean active shortest path")
    plt.title("Active-path geometry over time")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "path_length.png"), dpi=160)
    plt.close()

    print()
    print("Saved:")
    print(f"  {os.path.join(args.outdir, 'summary.json')}")
    print(f"  {os.path.join(args.outdir, 'timeseries.csv')}")
    print(f"  {os.path.join(args.outdir, 'activity_hist.png')}")
    print(f"  {os.path.join(args.outdir, 'link_rank_hist.png')}")
    print(f"  {os.path.join(args.outdir, 'largest_cluster.png')}")
    print(f"  {os.path.join(args.outdir, 'path_length.png')}")


if __name__ == "__main__":
    main()