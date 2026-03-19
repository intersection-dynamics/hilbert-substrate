"""
hsf_relational_dark_energy_probe_v1.py

Falsification-first toy test for the HSF idea:

    entangling / local interaction -> irreducible relation
    irreducible relation -> new subsystem
    no-refolding prevents cheap recoarse-graining
    interior interleaving of these relation-subsystems increases
    effective separation between the SAME old landmark subsystems

This script does NOT claim cosmology.
It tests whether your proposed mechanism beats two hard nulls:

  1) INTERLEAVE + NO-REFOLDING      (hypothesis)
  2) INTERLEAVE + FREE-REFOLDING    (null A)
  3) EDGE GROWTH ONLY               (null B)

Key observable:
  shortest-path distances between the ORIGINAL primitive subsystems
  as the relation graph grows.

If your idea is right, then:
  - model (1) should show sustained scale-factor growth,
  - model (2) should suppress that growth,
  - model (3) should add nodes without strongly increasing old-old distance,
  - and model (1) should place many new relation nodes ON interior shortest paths.

Outputs:
  - JSON summary
  - CSV summary
  - PNG plots

Example:
  python hsf_relational_dark_energy_probe_v1.py --outdir hsf_dark_energy_probe_out --n0 12 --steps 160 --p_interact 0.08 --seeds 12

Windows example:
  python hsf_relational_dark_energy_probe_v1.py --outdir hsf_dark_energy_probe_out --n0 12 --steps 160 --p_interact 0.08 --seeds 12
"""

import argparse
import csv
import json
import math
import os
from collections import deque, defaultdict
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  Graph utilities
# ============================================================

class RelationalGraph:
    def __init__(self):
        self.adj = {}       # node -> set(neighbors)
        self.meta = {}      # node -> dict(kind, birth, persistent, parents)
        self.next_id = 0

    def add_node(self, kind, birth, persistent=True, parents=None):
        nid = self.next_id
        self.next_id += 1
        self.adj[nid] = set()
        self.meta[nid] = {
            "kind": kind,                 # "primitive" or "relation"
            "birth": int(birth),
            "persistent": bool(persistent),
            "parents": list(parents) if parents is not None else [],
        }
        return nid

    def add_edge(self, u, v):
        if u == v:
            return
        self.adj[u].add(v)
        self.adj[v].add(u)

    def remove_edge(self, u, v):
        self.adj[u].discard(v)
        self.adj[v].discard(u)

    def has_edge(self, u, v):
        return v in self.adj.get(u, set())

    def remove_node(self, u):
        nbrs = list(self.adj[u])
        for v in nbrs:
            self.adj[v].discard(u)
        del self.adj[u]
        del self.meta[u]

    def degree(self, u):
        return len(self.adj[u])

    def nodes(self):
        return list(self.adj.keys())

    def edges(self):
        out = []
        for u in self.adj:
            for v in self.adj[u]:
                if u < v:
                    out.append((u, v))
        return out

    def n_nodes(self):
        return len(self.adj)

    def n_edges(self):
        return sum(len(v) for v in self.adj.values()) // 2

    def copy(self):
        g = RelationalGraph()
        g.adj = {k: set(v) for k, v in self.adj.items()}
        g.meta = {k: dict(v) for k, v in self.meta.items()}
        g.next_id = self.next_id
        return g


def make_ring_graph(n0):
    """
    Primitive landmark graph: cycle of n0 original subsystems.
    This avoids a privileged center and gives a clean initial geometry.
    """
    g = RelationalGraph()
    landmarks = []
    for _ in range(n0):
        landmarks.append(g.add_node(kind="primitive", birth=0, persistent=True, parents=[]))
    for i in range(n0):
        g.add_edge(landmarks[i], landmarks[(i + 1) % n0])
    return g, landmarks


def bfs_distances(graph, src):
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in graph.adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def pairwise_landmark_distances(graph, landmarks):
    pairs = []
    for i, j in combinations(landmarks, 2):
        dist = bfs_distances(graph, i).get(j, math.inf)
        pairs.append((i, j, dist))
    return pairs


def mean_pair_distance(graph, landmarks):
    pairs = pairwise_landmark_distances(graph, landmarks)
    vals = [d for _, _, d in pairs if np.isfinite(d)]
    if not vals:
        return math.inf
    return float(np.mean(vals))


def median_pair_distance(graph, landmarks):
    pairs = pairwise_landmark_distances(graph, landmarks)
    vals = [d for _, _, d in pairs if np.isfinite(d)]
    if not vals:
        return math.inf
    return float(np.median(vals))


def cv_pair_distance(graph, landmarks):
    pairs = pairwise_landmark_distances(graph, landmarks)
    vals = np.array([d for _, _, d in pairs if np.isfinite(d)], dtype=float)
    if len(vals) == 0:
        return math.inf
    mu = np.mean(vals)
    if mu <= 1e-12:
        return 0.0
    return float(np.std(vals) / mu)


def shortest_path_nodes_union(graph, landmarks):
    """
    Union of nodes that lie on at least one shortest path between
    distinct landmark pairs.

    A node x lies on a shortest path i->j iff dist(i,x)+dist(x,j)=dist(i,j).
    """
    dist_cache = {i: bfs_distances(graph, i) for i in landmarks}
    union_nodes = set()
    for i, j in combinations(landmarks, 2):
        dij = dist_cache[i].get(j, math.inf)
        if not np.isfinite(dij):
            continue
        for x in graph.nodes():
            dix = dist_cache[i].get(x, math.inf)
            dxj = dist_cache[j].get(x, math.inf)
            if np.isfinite(dix) and np.isfinite(dxj) and dix + dxj == dij:
                union_nodes.add(x)
    return union_nodes


def count_components(graph):
    unseen = set(graph.nodes())
    comps = 0
    while unseen:
        comps += 1
        root = next(iter(unseen))
        q = deque([root])
        unseen.remove(root)
        while q:
            u = q.popleft()
            for v in graph.adj[u]:
                if v in unseen:
                    unseen.remove(v)
                    q.append(v)
    return comps


# ============================================================
#  Model dynamics
# ============================================================

def canonical_edge_list(graph):
    edges = graph.edges()
    edges.sort()
    return edges


def relation_birth_probability(graph, u, v, args):
    """
    Local interaction -> candidate relation birth probability.

    Suppress heavily oversubscribed nodes to mimic finite bandwidth.
    """
    du = graph.degree(u)
    dv = graph.degree(v)

    overload_u = max(0, du - args.bandwidth)
    overload_v = max(0, dv - args.bandwidth)

    penalty = math.exp(-args.bandwidth_penalty * (overload_u + overload_v))
    p = args.p_interact * penalty
    return max(0.0, min(1.0, p))


def can_use_budget(per_step_budget, u, v, args):
    return per_step_budget[u] < args.max_interactions_per_node and per_step_budget[v] < args.max_interactions_per_node


def spawn_interleaved_relation(graph, u, v, t, args):
    """
    Replace edge (u,v) by u-r-v.
    This is the key 'interleaving in the interior' move.
    """
    persistent = True
    r = graph.add_node(kind="relation", birth=t, persistent=persistent, parents=[u, v])
    graph.remove_edge(u, v)
    graph.add_edge(u, r)
    graph.add_edge(r, v)
    return r


def spawn_edge_growth_relation(graph, u, v, t, args):
    """
    Add a relation node attached to u and v but KEEP the direct u-v edge.
    This creates new structure without forcing interior metric growth.
    """
    persistent = True
    r = graph.add_node(kind="relation", birth=t, persistent=persistent, parents=[u, v])
    graph.add_edge(u, r)
    graph.add_edge(v, r)
    return r


def refold_one_degree2_relation(graph, node):
    """
    Collapse a degree-2 relation node back into a direct edge between its neighbors.
    This is the null-model stand-in for free recoarse-graining / free refolding.
    """
    if node not in graph.adj:
        return False
    if graph.meta[node]["kind"] != "relation":
        return False
    nbrs = list(graph.adj[node])
    if len(nbrs) != 2:
        return False
    a, b = nbrs
    graph.remove_node(node)
    graph.add_edge(a, b)
    return True


def perform_free_refolding(graph, t, args, rng):
    """
    Null model: freely collapse degree-2 relation nodes after a short age.
    """
    nodes = graph.nodes()
    rng.shuffle(nodes)
    collapsed = 0
    for node in nodes:
        if node not in graph.adj:
            continue
        meta = graph.meta[node]
        if meta["kind"] != "relation":
            continue
        age = t - meta["birth"]
        if age < args.refold_min_age:
            continue
        if graph.degree(node) != 2:
            continue
        if rng.random() < args.refold_prob:
            ok = refold_one_degree2_relation(graph, node)
            if ok:
                collapsed += 1
    return collapsed


def step_model(graph, model_name, t, args, rng):
    per_step_budget = defaultdict(int)
    births = 0

    edges = canonical_edge_list(graph)
    rng.shuffle(edges)

    for u, v in edges:
        if u not in graph.adj or v not in graph.adj:
            continue
        if not graph.has_edge(u, v):
            continue
        if not can_use_budget(per_step_budget, u, v, args):
            continue

        p = relation_birth_probability(graph, u, v, args)
        if rng.random() >= p:
            continue

        # Local interaction happened on edge (u,v)
        if model_name in ("interleave_noref", "interleave_refold"):
            spawn_interleaved_relation(graph, u, v, t, args)
        elif model_name == "edge_growth":
            spawn_edge_growth_relation(graph, u, v, t, args)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        per_step_budget[u] += 1
        per_step_budget[v] += 1
        births += 1

    collapsed = 0
    if model_name == "interleave_refold":
        collapsed = perform_free_refolding(graph, t, args, rng)

    return births, collapsed


# ============================================================
#  Metrics
# ============================================================

def graph_metrics(graph, landmarks, initial_mean_dist):
    relation_nodes = [u for u in graph.nodes() if graph.meta[u]["kind"] == "relation"]
    primitive_nodes = [u for u in graph.nodes() if graph.meta[u]["kind"] == "primitive"]

    mean_dist = mean_pair_distance(graph, landmarks)
    med_dist = median_pair_distance(graph, landmarks)
    cv_dist = cv_pair_distance(graph, landmarks)

    if initial_mean_dist <= 0:
        scale_factor = math.nan
    else:
        scale_factor = mean_dist / initial_mean_dist

    on_path_union = shortest_path_nodes_union(graph, landmarks)

    if len(relation_nodes) > 0:
        relation_on_shortest_path = sum(1 for u in relation_nodes if u in on_path_union)
        on_path_fraction = relation_on_shortest_path / len(relation_nodes)
        degree2_fraction = sum(1 for u in relation_nodes if graph.degree(u) == 2) / len(relation_nodes)
        high_degree_fraction = sum(1 for u in relation_nodes if graph.degree(u) >= 3) / len(relation_nodes)
    else:
        on_path_fraction = 0.0
        degree2_fraction = 0.0
        high_degree_fraction = 0.0

    return {
        "n_nodes": graph.n_nodes(),
        "n_edges": graph.n_edges(),
        "n_primitive": len(primitive_nodes),
        "n_relation": len(relation_nodes),
        "components": count_components(graph),
        "mean_landmark_distance": mean_dist,
        "median_landmark_distance": med_dist,
        "cv_landmark_distance": cv_dist,
        "scale_factor": scale_factor,
        "relation_on_shortest_path_fraction": on_path_fraction,
        "relation_degree2_fraction": degree2_fraction,
        "relation_high_degree_fraction": high_degree_fraction,
    }


# ============================================================
#  Running one seed
# ============================================================

def run_single_seed(model_name, seed, args):
    rng = np.random.default_rng(seed)
    graph, landmarks = make_ring_graph(args.n0)

    initial_mean_dist = mean_pair_distance(graph, landmarks)
    initial_metrics = graph_metrics(graph, landmarks, initial_mean_dist)

    history = {
        "step": [],
        "births": [],
        "collapsed": [],
        "n_nodes": [],
        "n_edges": [],
        "n_relation": [],
        "mean_landmark_distance": [],
        "median_landmark_distance": [],
        "cv_landmark_distance": [],
        "scale_factor": [],
        "relation_on_shortest_path_fraction": [],
        "relation_degree2_fraction": [],
        "relation_high_degree_fraction": [],
    }

    total_births = 0
    total_collapsed = 0

    for t in range(1, args.steps + 1):
        births, collapsed = step_model(graph, model_name, t, args, rng)
        total_births += births
        total_collapsed += collapsed
        m = graph_metrics(graph, landmarks, initial_mean_dist)

        history["step"].append(t)
        history["births"].append(int(births))
        history["collapsed"].append(int(collapsed))
        history["n_nodes"].append(int(m["n_nodes"]))
        history["n_edges"].append(int(m["n_edges"]))
        history["n_relation"].append(int(m["n_relation"]))
        history["mean_landmark_distance"].append(float(m["mean_landmark_distance"]))
        history["median_landmark_distance"].append(float(m["median_landmark_distance"]))
        history["cv_landmark_distance"].append(float(m["cv_landmark_distance"]))
        history["scale_factor"].append(float(m["scale_factor"]))
        history["relation_on_shortest_path_fraction"].append(float(m["relation_on_shortest_path_fraction"]))
        history["relation_degree2_fraction"].append(float(m["relation_degree2_fraction"]))
        history["relation_high_degree_fraction"].append(float(m["relation_high_degree_fraction"]))

    final_metrics = graph_metrics(graph, landmarks, initial_mean_dist)

    # Simple falsification-facing score:
    # We want scale growth, interiority, and not-crazy inhomogeneity.
    score = (
        2.0 * final_metrics["scale_factor"]
        + 1.5 * final_metrics["relation_on_shortest_path_fraction"]
        - 0.75 * final_metrics["cv_landmark_distance"]
    )

    return {
        "model": model_name,
        "seed": int(seed),
        "params": vars(args),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "summary": {
            "total_births": int(total_births),
            "total_collapsed": int(total_collapsed),
            "net_relations": int(final_metrics["n_relation"]),
            "score": float(score),
        },
        "history": history,
    }


# ============================================================
#  Aggregation
# ============================================================

def aggregate_runs(runs):
    if not runs:
        return {}

    keys_final = [
        "n_nodes",
        "n_edges",
        "n_relation",
        "mean_landmark_distance",
        "median_landmark_distance",
        "cv_landmark_distance",
        "scale_factor",
        "relation_on_shortest_path_fraction",
        "relation_degree2_fraction",
        "relation_high_degree_fraction",
    ]

    out = {"n_runs": len(runs), "final_mean": {}, "final_std": {}, "seed_scores": []}

    for k in keys_final:
        vals = np.array([r["final_metrics"][k] for r in runs], dtype=float)
        out["final_mean"][k] = float(np.mean(vals))
        out["final_std"][k] = float(np.std(vals))

    out["seed_scores"] = [float(r["summary"]["score"]) for r in runs]
    out["score_mean"] = float(np.mean(out["seed_scores"]))
    out["score_std"] = float(np.std(out["seed_scores"]))

    # Mean timeseries
    hist_keys = list(runs[0]["history"].keys())
    ts = {}
    for k in hist_keys:
        arr = np.array([r["history"][k] for r in runs], dtype=float)
        ts[k] = {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
        }
    out["timeseries"] = ts

    return out


def compare_models(model_aggs):
    """
    Hard comparison logic.
    """
    need = ["interleave_noref", "interleave_refold", "edge_growth"]
    if not all(k in model_aggs for k in need):
        return {}

    H = model_aggs["interleave_noref"]["final_mean"]
    R = model_aggs["interleave_refold"]["final_mean"]
    E = model_aggs["edge_growth"]["final_mean"]

    report = {}

    report["hypothesis_beats_refolding_on_scale"] = bool(
        H["scale_factor"] > R["scale_factor"] * 1.15
    )
    report["hypothesis_beats_edge_growth_on_scale"] = bool(
        H["scale_factor"] > E["scale_factor"] * 1.15
    )
    report["hypothesis_is_more_interior_than_edge_growth"] = bool(
        H["relation_on_shortest_path_fraction"] > E["relation_on_shortest_path_fraction"] * 1.50 + 1e-12
    )
    report["hypothesis_is_not_wildly_more_lumpy_than_nulls"] = bool(
        H["cv_landmark_distance"] < max(R["cv_landmark_distance"], E["cv_landmark_distance"]) * 1.35 + 1e-12
    )

    n_true = sum(int(v) for v in report.values())
    report["n_tests_passed"] = n_true
    report["overall_support"] = "YES" if n_true >= 3 else "NO"

    return report


# ============================================================
#  Saving
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv_summary(model_aggs, comparison, path):
    rows = []
    for model, agg in model_aggs.items():
        fm = agg["final_mean"]
        rows.append({
            "model": model,
            "n_runs": agg["n_runs"],
            "score_mean": agg["score_mean"],
            "scale_factor_final_mean": fm["scale_factor"],
            "mean_landmark_distance_final_mean": fm["mean_landmark_distance"],
            "cv_landmark_distance_final_mean": fm["cv_landmark_distance"],
            "relation_on_shortest_path_fraction_final_mean": fm["relation_on_shortest_path_fraction"],
            "n_relation_final_mean": fm["n_relation"],
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)

        # Spacer + comparison lines
        w.writerow({})
        w.writerow({"model": "comparison"})
        for k, v in comparison.items():
            w.writerow({"model": k, "n_runs": v})


def plot_timeseries(model_aggs, outdir):
    steps = np.array(model_aggs[next(iter(model_aggs))]["timeseries"]["step"]["mean"], dtype=float)

    # 1) scale factor
    plt.figure(figsize=(8, 5))
    for model, agg in model_aggs.items():
        y = np.array(agg["timeseries"]["scale_factor"]["mean"], dtype=float)
        s = np.array(agg["timeseries"]["scale_factor"]["std"], dtype=float)
        plt.plot(steps, y, label=model)
        plt.fill_between(steps, y - s, y + s, alpha=0.18)
    plt.xlabel("step")
    plt.ylabel("effective scale factor a_eff")
    plt.title("Landmark separation growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scale_factor.png"), dpi=160)
    plt.close()

    # 2) relation count
    plt.figure(figsize=(8, 5))
    for model, agg in model_aggs.items():
        y = np.array(agg["timeseries"]["n_relation"]["mean"], dtype=float)
        s = np.array(agg["timeseries"]["n_relation"]["std"], dtype=float)
        plt.plot(steps, y, label=model)
        plt.fill_between(steps, y - s, y + s, alpha=0.18)
    plt.xlabel("step")
    plt.ylabel("relation-node count")
    plt.title("Relational proliferation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "relation_count.png"), dpi=160)
    plt.close()

    # 3) interiority
    plt.figure(figsize=(8, 5))
    for model, agg in model_aggs.items():
        y = np.array(agg["timeseries"]["relation_on_shortest_path_fraction"]["mean"], dtype=float)
        s = np.array(agg["timeseries"]["relation_on_shortest_path_fraction"]["std"], dtype=float)
        plt.plot(steps, y, label=model)
        plt.fill_between(steps, y - s, y + s, alpha=0.18)
    plt.xlabel("step")
    plt.ylabel("fraction of relation nodes on shortest landmark paths")
    plt.title("Interior interleaving witness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "interiority.png"), dpi=160)
    plt.close()

    # 4) homogeneity / lumpiness
    plt.figure(figsize=(8, 5))
    for model, agg in model_aggs.items():
        y = np.array(agg["timeseries"]["cv_landmark_distance"]["mean"], dtype=float)
        s = np.array(agg["timeseries"]["cv_landmark_distance"]["std"], dtype=float)
        plt.plot(steps, y, label=model)
        plt.fill_between(steps, y - s, y + s, alpha=0.18)
    plt.xlabel("step")
    plt.ylabel("CV of landmark-pair distances")
    plt.title("Expansion homogeneity proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "homogeneity_cv.png"), dpi=160)
    plt.close()


# ============================================================
#  Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="HSF relational proliferation / dark-energy toy probe"
    )
    p.add_argument("--outdir", type=str, default="hsf_dark_energy_probe_out")
    p.add_argument("--n0", type=int, default=12,
                   help="number of original primitive landmarks on the initial ring")
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--p_interact", type=float, default=0.08,
                   help="base probability that a local edge interaction births a relation")
    p.add_argument("--bandwidth", type=int, default=2,
                   help="soft local capacity target")
    p.add_argument("--bandwidth_penalty", type=float, default=0.75,
                   help="penalty strength for interactions beyond bandwidth")
    p.add_argument("--max_interactions_per_node", type=int, default=1,
                   help="per-step local interaction budget")
    p.add_argument("--refold_prob", type=float, default=0.45,
                   help="probability of collapsing an eligible relation in free-refolding null")
    p.add_argument("--refold_min_age", type=int, default=1,
                   help="minimum age before a relation may be refolded")
    p.add_argument("--seeds", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    print("=" * 72)
    print("HSF RELATIONAL DARK ENERGY PROBE (v1)")
    print("=" * 72)
    print(f"outdir: {args.outdir}")
    print(f"n0={args.n0}, steps={args.steps}, p_interact={args.p_interact}")
    print(f"bandwidth={args.bandwidth}, max_interactions_per_node={args.max_interactions_per_node}")
    print(f"free-refold null: refold_prob={args.refold_prob}, refold_min_age={args.refold_min_age}")
    print(f"seeds={args.seeds}")
    print()
    print("Models:")
    print("  interleave_noref   : hypothesis")
    print("  interleave_refold  : null A")
    print("  edge_growth        : null B")
    print()

    model_names = ["interleave_noref", "interleave_refold", "edge_growth"]
    all_runs = {m: [] for m in model_names}

    seed_list = list(range(args.seeds))
    for model in model_names:
        print("-" * 72)
        print(f"RUNNING MODEL: {model}")
        print("-" * 72)
        for seed in seed_list:
            run = run_single_seed(model, seed, args)
            all_runs[model].append(run)
            fm = run["final_metrics"]
            print(
                f"  seed={seed:>2}  "
                f"scale={fm['scale_factor']:.3f}  "
                f"mean_d={fm['mean_landmark_distance']:.3f}  "
                f"on_path={fm['relation_on_shortest_path_fraction']:.3f}  "
                f"cv={fm['cv_landmark_distance']:.3f}  "
                f"n_rel={fm['n_relation']:>5}"
            )
        print()

    model_aggs = {m: aggregate_runs(all_runs[m]) for m in model_names}
    comparison = compare_models(model_aggs)

    print("=" * 72)
    print("FINAL MODEL MEANS")
    print("=" * 72)
    for model in model_names:
        fm = model_aggs[model]["final_mean"]
        print(f"{model:>18}: "
              f"scale={fm['scale_factor']:.3f}   "
              f"mean_d={fm['mean_landmark_distance']:.3f}   "
              f"on_path={fm['relation_on_shortest_path_fraction']:.3f}   "
              f"cv={fm['cv_landmark_distance']:.3f}   "
              f"n_rel={fm['n_relation']:.1f}")

    print()
    print("=" * 72)
    print("HARD TESTS")
    print("=" * 72)
    for k, v in comparison.items():
        print(f"{k}: {v}")

    payload = {
        "params": vars(args),
        "per_seed_runs": all_runs,
        "model_aggregates": model_aggs,
        "comparison": comparison,
    }

    save_json(payload, os.path.join(args.outdir, "summary.json"))
    save_csv_summary(model_aggs, comparison, os.path.join(args.outdir, "summary.csv"))
    plot_timeseries(model_aggs, args.outdir)

    print()
    print(f"Saved:")
    print(f"  {os.path.join(args.outdir, 'summary.json')}")
    print(f"  {os.path.join(args.outdir, 'summary.csv')}")
    print(f"  {os.path.join(args.outdir, 'scale_factor.png')}")
    print(f"  {os.path.join(args.outdir, 'relation_count.png')}")
    print(f"  {os.path.join(args.outdir, 'interiority.png')}")
    print(f"  {os.path.join(args.outdir, 'homogeneity_cv.png')}")


if __name__ == "__main__":
    main()