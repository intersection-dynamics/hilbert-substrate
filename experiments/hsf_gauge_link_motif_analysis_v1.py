#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple
import numpy as np

def gell_mann() -> List[np.ndarray]:
    i = 1j
    out = []
    out.append(np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex))
    out.append(np.array([[0,-i,0],[i,0,0],[0,0,0]], dtype=complex))
    out.append(np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=complex))
    out.append(np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=complex))
    out.append(np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex))
    out.append(np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex))
    out.append(np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex))
    out.append((1.0/np.sqrt(3.0))*np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=complex))
    return out

GM = gell_mann()

def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)

def op_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord=2))

def singular_stats(M: np.ndarray) -> Dict[str, object]:
    s = np.linalg.svd(M, compute_uv=False)
    s2 = np.abs(s)**2
    total = float(np.sum(s2))
    top = float(np.max(np.abs(s))) if len(s) else 0.0
    rank = int(np.sum(np.abs(s) > 1e-10))
    stable_rank = 0.0 if top <= 1e-15 else float(total / (top**2))
    if total <= 1e-15:
        entropy_rank = 0.0
        top_frac = 0.0
    else:
        p = s2 / total
        nz = p[p > 1e-15]
        entropy_rank = float(np.exp(-np.sum(nz * np.log(nz))))
        top_frac = float(np.max(s2) / total)
    return {"singular_values":[float(x) for x in s.tolist()], "rank":rank, "stable_rank":stable_rank, "entropy_rank":entropy_rank, "top_mode_energy_fraction":top_frac}

def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a

def random_orthogonal_8(rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(8,8))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R)); signs[signs == 0] = 1.0
    return Q @ np.diag(signs)

def su3_link_endpoint_ops() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    I3 = np.eye(3, dtype=complex)
    left, right = [], []
    for T in GM:
        left.append(kron(T, I3))
        right.append(kron(I3, -T.conj()))
    return left, right

LEFT_OPS, RIGHT_OPS = su3_link_endpoint_ops()

def endpoint_commutativity_witness() -> float:
    mx = 0.0
    for La in LEFT_OPS:
        for Rb in RIGHT_OPS:
            mx = max(mx, op_norm(commutator(La, Rb)))
    return mx

ENDPOINT_COMM_MAX = endpoint_commutativity_witness()

@dataclass
class GaugeLink:
    i: int
    j: int
    left_map: np.ndarray
    right_map: np.ndarray
    slack_dims: int
    endpoint_commutator_max: float
    bandwidth_rank: int
    stable_rank: float
    entropy_rank: float
    commitment_score: float
    spectrum_left: Dict[str, object]
    spectrum_right: Dict[str, object]

def make_bandlimited_map(rng: np.random.Generator, active_rank: int, strength_lo: float, strength_hi: float, random_mix: bool) -> np.ndarray:
    active_rank = max(1, min(8, int(active_rank)))
    vals = np.zeros(8, dtype=float)
    vals[:active_rank] = rng.uniform(strength_lo, strength_hi, size=active_rank)
    rng.shuffle(vals)
    D = np.diag(vals)
    if random_mix:
        return random_orthogonal_8(rng) @ D @ random_orthogonal_8(rng).T
    return D

def commitment_from_maps(L: np.ndarray, R: np.ndarray, slack_dims: int) -> float:
    sL = singular_stats(L); sR = singular_stats(R)
    rank_pen = 0.5 * (sL["stable_rank"] + sR["stable_rank"]) / 8.0
    mismatch = float(np.linalg.norm(np.abs(L) - np.abs(R), ord="fro")) / (1.0 + np.linalg.norm(np.abs(L), ord="fro") + np.linalg.norm(np.abs(R), ord="fro"))
    slack_pen = float(slack_dims) / (float(slack_dims) + 4.0)
    raw = 1.0 - (0.45 * rank_pen + 0.35 * mismatch + 0.20 * slack_pen)
    return float(max(0.0, min(1.0, raw)))

def make_link(rng: np.random.Generator, i: int, j: int, active_rank: int, slack_dims: int, random_mix: bool) -> GaugeLink:
    L = make_bandlimited_map(rng, active_rank, 0.65, 1.15, random_mix)
    R = make_bandlimited_map(rng, active_rank, 0.65, 1.15, random_mix)
    sL = singular_stats(L); sR = singular_stats(R)
    return GaugeLink(
        i=i, j=j, left_map=L, right_map=R, slack_dims=int(slack_dims),
        endpoint_commutator_max=float(ENDPOINT_COMM_MAX),
        bandwidth_rank=min(int(sL["rank"]), int(sR["rank"])),
        stable_rank=0.5*(float(sL["stable_rank"]) + float(sR["stable_rank"])),
        entropy_rank=0.5*(float(sL["entropy_rank"]) + float(sR["entropy_rank"])),
        commitment_score=commitment_from_maps(L, R, slack_dims),
        spectrum_left=sL, spectrum_right=sR
    )

def make_graph_edges(n: int, graph_type: str, rng: np.random.Generator) -> List[Tuple[int,int]]:
    edges = set()
    if graph_type == "chain":
        for i in range(n - 1): edges.add((i, i+1))
    elif graph_type == "ring":
        for i in range(n): edges.add(tuple(sorted((i, (i+1)%n))))
    elif graph_type == "ring_plus_chords":
        for i in range(n): edges.add(tuple(sorted((i, (i+1)%n))))
        chords = max(1, n//3); tries = 0
        while len(edges) < n + chords and tries < 12*n:
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            if abs(a - b) not in (1, n-1): edges.add((a,b))
            tries += 1
    elif graph_type == "erdos":
        p = min(0.55, max(0.25, 2.5/max(2,n)))
        for i in range(n):
            for j in range(i+1, n):
                if rng.uniform() < p: edges.add((i,j))
        for i in range(n-1): edges.add((i, i+1))
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    return sorted(edges)

def build_network(n: int, graph_type: str, seed: int, rank_lo: int, rank_hi: int, slack_prob: float, random_mix: bool) -> Dict[Tuple[int,int], GaugeLink]:
    rng = np.random.default_rng(seed)
    out = {}
    for (i,j) in make_graph_edges(n, graph_type, rng):
        active_rank = int(rng.integers(rank_lo, rank_hi+1))
        slack_dims = int(rng.integers(1,4)) if rng.uniform() < slack_prob else 0
        out[(i,j)] = make_link(rng, i, j, active_rank, slack_dims, random_mix)
    return out

def compatibility_at_shared_node(left_link: GaugeLink, shared_node: int, right_link: GaugeLink) -> float:
    A = left_link.left_map if shared_node == left_link.i else left_link.right_map
    B = right_link.left_map if shared_node == right_link.i else right_link.right_map
    denom = 1.0 + float(np.linalg.norm(A, ord="fro")) + float(np.linalg.norm(B, ord="fro"))
    mismatch = float(np.linalg.norm(A - B, ord="fro")) / denom
    return float(max(0.0, 1.0 - mismatch))

def chain_transmission(left_link: GaugeLink, shared_node: int, right_link: GaugeLink) -> Dict[str, object]:
    A_out = left_link.left_map if shared_node == left_link.i else left_link.right_map
    B_in = right_link.left_map if shared_node == right_link.i else right_link.right_map
    T = B_in @ A_out
    stats = singular_stats(T)
    return {"matrix_fro": float(np.linalg.norm(T, ord="fro")), "rank": int(stats["rank"]), "stable_rank": float(stats["stable_rank"]), "entropy_rank": float(stats["entropy_rank"]), "top_mode_energy_fraction": float(stats["top_mode_energy_fraction"]), "singular_values": stats["singular_values"]}

def analyze_chains(n: int, links: Dict[Tuple[int,int], GaugeLink]) -> List[Dict[str, object]]:
    adj = {i: [] for i in range(n)}
    for (i,j) in links:
        adj[i].append(j); adj[j].append(i)
    out = []
    for center in range(n):
        nbrs = sorted(adj[center])
        for a, c in combinations(nbrs, 2):
            e1 = (min(a, center), max(a, center)); e2 = (min(center, c), max(center, c))
            L1, L2 = links[e1], links[e2]
            comp = compatibility_at_shared_node(L1, center, L2)
            trans = chain_transmission(L1, center, L2)
            out.append({
                "chain": [a, center, c],
                "compatibility": comp,
                "mean_link_commitment": 0.5*(L1.commitment_score + L2.commitment_score),
                "mean_link_bandwidth_rank": 0.5*(L1.bandwidth_rank + L2.bandwidth_rank),
                "two_step_rank": trans["rank"],
                "two_step_stable_rank": trans["stable_rank"],
                "two_step_fro": trans["matrix_fro"],
                "two_step_top_frac": trans["top_mode_energy_fraction"],
            })
    out.sort(key=lambda d: (d["compatibility"], d["two_step_fro"], d["mean_link_commitment"]), reverse=True)
    return out

def analyze_triangles(n: int, links: Dict[Tuple[int,int], GaugeLink]) -> List[Dict[str, object]]:
    edge_set = set(links.keys())
    out = []
    for a, b, c in combinations(range(n), 3):
        eab = (min(a,b), max(a,b)); ebc = (min(b,c), max(b,c)); eca = (min(c,a), max(c,a))
        if eab in edge_set and ebc in edge_set and eca in edge_set:
            Lab, Lbc, Lca = links[eab], links[ebc], links[eca]
            comp_ab_bc = compatibility_at_shared_node(Lab, b, Lbc)
            comp_bc_ca = compatibility_at_shared_node(Lbc, c, Lca)
            comp_ca_ab = compatibility_at_shared_node(Lca, a, Lab)
            Mab = Lab.right_map if Lab.j == b else Lab.left_map
            Mbc = Lbc.right_map if Lbc.j == c else Lbc.left_map
            Mca = Lca.right_map if Lca.j == a else Lca.left_map
            H = Mca @ Mbc @ Mab
            hol_defect = float(np.linalg.norm(H - np.eye(8), ord="fro")) / (1.0 + float(np.linalg.norm(H, ord="fro")))
            out.append({
                "triangle": [a,b,c],
                "compatibility_mean": float((comp_ab_bc + comp_bc_ca + comp_ca_ab)/3.0),
                "loop_holonomy_defect": hol_defect,
                "frustration_score": float(min(1.0, hol_defect)),
                "triangle_commitment_mean": float((Lab.commitment_score + Lbc.commitment_score + Lca.commitment_score)/3.0),
            })
    out.sort(key=lambda d: (d["compatibility_mean"], -d["frustration_score"], d["triangle_commitment_mean"]), reverse=True)
    return out

def analyze_branches(n: int, links: Dict[Tuple[int,int], GaugeLink]) -> List[Dict[str, object]]:
    adj = {i: [] for i in range(n)}
    for (i,j) in links:
        adj[i].append(j); adj[j].append(i)
    out = []
    for center in range(n):
        nbrs = sorted(adj[center])
        if len(nbrs) < 3: continue
        endpoint_strengths = []
        for nb in nbrs:
            e = (min(center, nb), max(center, nb)); L = links[e]
            M = L.left_map if center == L.i else L.right_map
            endpoint_strengths.append((nb, float(np.linalg.norm(M, ord="fro")), L.commitment_score, L.bandwidth_rank))
        endpoint_strengths.sort(key=lambda x: x[1], reverse=True)
        vals = [x[1] for x in endpoint_strengths]
        total = sum(vals); top = vals[0] if vals else 0.0
        out.append({
            "center": center,
            "degree": len(nbrs),
            "branch_asymmetry": 0.0 if total <= 1e-15 else top / total,
            "ordered_neighbors": [{"neighbor": nb, "endpoint_strength": s, "commitment": c, "bandwidth_rank": br} for (nb, s, c, br) in endpoint_strengths],
        })
    out.sort(key=lambda d: d["branch_asymmetry"], reverse=True)
    return out

def aggregate_network(n: int, links: Dict[Tuple[int,int], GaugeLink]) -> Dict[str, object]:
    link_rows = []
    for e, L in links.items():
        link_rows.append({
            "edge": list(e), "bandwidth_rank": int(L.bandwidth_rank), "stable_rank": float(L.stable_rank),
            "entropy_rank": float(L.entropy_rank), "commitment_score": float(L.commitment_score),
            "slack_dims": int(L.slack_dims), "endpoint_commutator_max": float(L.endpoint_commutator_max),
        })
    link_rows.sort(key=lambda d: (d["commitment_score"], -d["stable_rank"]), reverse=True)
    chains = analyze_chains(n, links); triangles = analyze_triangles(n, links); branches = analyze_branches(n, links)
    link_commit = [r["commitment_score"] for r in link_rows]
    link_band = [r["bandwidth_rank"] for r in link_rows]
    chain_comp = [r["compatibility"] for r in chains]
    chain_rank = [r["two_step_rank"] for r in chains]
    tri_coh = [r["compatibility_mean"] for r in triangles]
    tri_frust = [r["frustration_score"] for r in triangles]
    branch_asym = [r["branch_asymmetry"] for r in branches]
    return {
        "n_links": len(link_rows), "n_chains": len(chains), "n_triangles": len(triangles), "n_branches": len(branches),
        "mean_link_commitment": float(np.mean(link_commit)) if link_commit else 0.0,
        "mean_link_bandwidth_rank": float(np.mean(link_band)) if link_band else 0.0,
        "mean_chain_compatibility": float(np.mean(chain_comp)) if chain_comp else 0.0,
        "mean_chain_two_step_rank": float(np.mean(chain_rank)) if chain_rank else 0.0,
        "mean_triangle_compatibility": float(np.mean(tri_coh)) if tri_coh else 0.0,
        "mean_triangle_frustration": float(np.mean(tri_frust)) if tri_frust else 0.0,
        "mean_branch_asymmetry": float(np.mean(branch_asym)) if branch_asym else 0.0,
        "top_links": link_rows[:min(10, len(link_rows))],
        "top_chains": chains[:min(10, len(chains))],
        "top_triangles": triangles[:min(10, len(triangles))],
        "top_branches": branches[:min(10, len(branches))],
    }

def pretty_report(summary: Dict[str, object], cfg: Dict[str, object]) -> str:
    lines = []
    lines.append("="*108)
    lines.append("HSF GAUGE-LINK MOTIF ANALYSIS (v1)")
    lines.append("-"*108)
    lines.append(f"N={cfg['n']}  graph={cfg['graph_type']}  seed={cfg['seed']}  rank_range=[{cfg['rank_lo']},{cfg['rank_hi']}]  slack_prob={cfg['slack_prob']}  random_mix={cfg['random_mix']}")
    lines.append("HSF link model: H_link ~= V ⊗ Vbar, dim=9, commuting SU(3) endpoints, bandwidth from influence-map spectrum, commitment vs slack.")
    lines.append("-"*108)
    lines.append(f"Counts: links={summary['n_links']}  chains={summary['n_chains']}  triangles={summary['n_triangles']}  branches={summary['n_branches']}")
    lines.append(f"Means: link_commit={summary['mean_link_commitment']:.4f}  link_band_rank={summary['mean_link_bandwidth_rank']:.4f}  chain_comp={summary['mean_chain_compatibility']:.4f}  chain_rank={summary['mean_chain_two_step_rank']:.4f}  tri_comp={summary['mean_triangle_compatibility']:.4f}  tri_frust={summary['mean_triangle_frustration']:.4f}  branch_asym={summary['mean_branch_asymmetry']:.4f}")
    lines.append("-"*108)
    lines.append("Top links:")
    for row in summary["top_links"]:
        lines.append(f"  edge={row['edge']}  commit={row['commitment_score']:.4f}  band_rank={row['bandwidth_rank']}  stable_rank={row['stable_rank']:.3f}  slack={row['slack_dims']}  endpoint_comm_max={row['endpoint_commutator_max']:.2e}")
    lines.append("-"*108)
    lines.append("Top 2nd-order chains:")
    for row in summary["top_chains"]:
        lines.append(f"  chain={row['chain']}  compat={row['compatibility']:.4f}  mean_commit={row['mean_link_commitment']:.4f}  2step_rank={row['two_step_rank']}  2step_fro={row['two_step_fro']:.4f}")
    lines.append("-"*108)
    lines.append("Top 3rd-order triangles:")
    for row in summary["top_triangles"]:
        lines.append(f"  tri={row['triangle']}  compat_mean={row['compatibility_mean']:.4f}  hol_defect={row['loop_holonomy_defect']:.4f}  frustr={row['frustration_score']:.4f}  tri_commit={row['triangle_commitment_mean']:.4f}")
    if summary["top_branches"]:
        lines.append("-"*108)
        lines.append("Top branch motifs:")
        for row in summary["top_branches"]:
            lines.append(f"  center={row['center']} degree={row['degree']} branch_asym={row['branch_asymmetry']:.4f}")
    return "\n".join(lines)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze 2nd- and 3rd-order effects in an HSF-correct gauge-link network.")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--graph-type", choices=["chain","ring","ring_plus_chords","erdos"], default="ring_plus_chords")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank-lo", type=int, default=2)
    ap.add_argument("--rank-hi", type=int, default=5)
    ap.add_argument("--slack-prob", type=float, default=0.25)
    ap.add_argument("--random-mix", action="store_true")
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    cfg = {"n": int(args.n), "graph_type": str(args.graph_type), "seed": int(args.seed), "rank_lo": int(args.rank_lo), "rank_hi": int(args.rank_hi), "slack_prob": float(args.slack_prob), "random_mix": bool(args.random_mix)}
    links = build_network(cfg["n"], cfg["graph_type"], cfg["seed"], cfg["rank_lo"], cfg["rank_hi"], cfg["slack_prob"], cfg["random_mix"])
    summary = aggregate_network(cfg["n"], links)
    print(pretty_report(summary, cfg))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "summary": summary}, f, indent=2)
        print(f"\nSaved JSON: {args.json_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
