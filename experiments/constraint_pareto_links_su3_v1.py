#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List
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
SQRT2 = np.sqrt(2.0)

def hs_inner(x: np.ndarray, y: np.ndarray) -> complex:
    return np.trace(x.conj().T @ y)

def traceless_part(x: np.ndarray) -> np.ndarray:
    d = x.shape[0]
    return x - np.trace(x) * np.eye(d, dtype=complex) / d

def gm_coeffs(x: np.ndarray) -> np.ndarray:
    xt = traceless_part(x)
    coeffs = []
    for lam in GM:
        e = lam / SQRT2
        coeffs.append(float(np.real(hs_inner(e, xt))))
    return np.array(coeffs, dtype=float)

def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)

def matrix_exp_hermitian(h: np.ndarray, t: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(h)
    return vecs @ np.diag(np.exp(-1j*t*vals)) @ vecs.conj().T

def pure_density(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(-1,1)
    return psi @ psi.conj().T

def partial_trace_ab_to_b(rho_ab: np.ndarray, da: int, db: int) -> np.ndarray:
    resh = rho_ab.reshape(da, db, da, db)
    out = np.zeros((db,db), dtype=complex)
    for a in range(da):
        out += resh[a,:,a,:]
    return out

def random_qutrit_state(rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=3) + 1j * rng.normal(size=3)
    return z / np.linalg.norm(z)

def build_ab_hamiltonian(coupling_weights: np.ndarray, local_scale_a: float, local_scale_b: float) -> np.ndarray:
    d = 3
    I = np.eye(d, dtype=complex)
    hA = local_scale_a * (0.8*GM[2] + 0.3*GM[7] + 0.2*GM[0])
    hB = local_scale_b * (0.7*GM[2] - 0.25*GM[7] + 0.15*GM[5])
    H = kron(hA, I) + kron(I, hB)
    for a in range(8):
        H += float(coupling_weights[a]) * kron(GM[a], GM[a])
    return 0.5*(H + H.conj().T)

def induced_delta_B(delta: float, time: float, poke_idx: int, coupling_weights: np.ndarray,
                    local_scale_a: float, local_scale_b: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    da = db = 3
    H = build_ab_hamiltonian(coupling_weights, local_scale_a, local_scale_b)
    U = matrix_exp_hermitian(H, time)
    psiA = random_qutrit_state(rng)
    psiB = random_qutrit_state(rng)
    psi0 = np.kron(psiA, psiB)

    Kp = np.eye(da*db, dtype=complex) - 1j*0.035*kron(GM[poke_idx], np.eye(db, dtype=complex))
    Km = np.eye(da*db, dtype=complex) + 1j*0.035*kron(GM[poke_idx], np.eye(db, dtype=complex))

    psi_p = Kp @ psi0; psi_p = psi_p / np.linalg.norm(psi_p)
    psi_m = Km @ psi0; psi_m = psi_m / np.linalg.norm(psi_m)

    rho_p = pure_density(U @ psi_p)
    rho_m = pure_density(U @ psi_m)
    rhoBp = partial_trace_ab_to_b(rho_p, da, db)
    rhoBm = partial_trace_ab_to_b(rho_m, da, db)
    dB = traceless_part((rhoBp - rhoBm)/(2.0*0.035))
    return 0.5*(dB + dB.conj().T)

@dataclass
class Candidate:
    name: str
    strengths: np.ndarray
    slack_dims: int
    family: str

    def map_matrix(self, priority_weighting: np.ndarray | None = None) -> np.ndarray:
        mats = []
        for i in range(self.strengths.shape[0]):
            blk = np.diag(self.strengths[i].astype(float))
            if priority_weighting is not None:
                blk = np.sqrt(float(priority_weighting[i])) * blk
            mats.append(blk)
        return np.vstack(mats)

def candidate_partitioned(rng: np.random.Generator, jitter: float = 0.0) -> Candidate:
    strengths = np.zeros((3,8), dtype=float)
    groups = [[0,1,2],[3,4,5],[6,7]]
    for i, grp in enumerate(groups):
        for j in grp:
            strengths[i,j] = 1.0 + jitter * rng.uniform(-0.15, 0.15)
    return Candidate("partitioned", np.clip(strengths,0.0,None), 0, "partitioned")

def candidate_shared(rng: np.random.Generator) -> Candidate:
    strengths = np.zeros((3,8), dtype=float)
    strengths[0,[0,1,2,7]] = rng.uniform(0.85,1.10,size=4)
    strengths[1,[3,4,7]] = rng.uniform(0.85,1.10,size=3)
    strengths[2,[5,6,7]] = rng.uniform(0.85,1.10,size=3)
    return Candidate("shared", strengths, 0, "shared")

def candidate_dark(rng: np.random.Generator) -> Candidate:
    strengths = np.zeros((3,8), dtype=float)
    strengths[0,[0,2]] = rng.uniform(0.45,0.8,size=2)
    strengths[1,[3,7]] = rng.uniform(0.45,0.8,size=2)
    strengths[2,[5]] = rng.uniform(0.45,0.8,size=1)
    return Candidate("dark", strengths, int(rng.integers(0,2)), "dark")

def candidate_slacky(rng: np.random.Generator) -> Candidate:
    c = candidate_shared(rng)
    return Candidate("slacky", c.strengths, int(rng.integers(2,7)), "slacky")

def candidate_random(rng: np.random.Generator) -> Candidate:
    strengths = np.zeros((3,8), dtype=float)
    for i in range(3):
        mask = rng.uniform(size=8) < rng.uniform(0.25, 0.65)
        vals = rng.uniform(0.35, 1.15, size=8)
        strengths[i] = np.where(mask, vals, 0.0)
    return Candidate("random", strengths, int(rng.integers(0,5)), "random")

def generate_candidates(n_random: int, seed: int) -> List[Candidate]:
    rng = np.random.default_rng(seed)
    cands = [candidate_partitioned(rng), candidate_partitioned(rng, jitter=0.08), candidate_shared(rng), candidate_dark(rng), candidate_slacky(rng)]
    for _ in range(n_random):
        fam = rng.choice(["partitioned","shared","dark","slacky","random"], p=[0.15,0.20,0.15,0.15,0.35])
        if fam == "partitioned":
            cands.append(candidate_partitioned(rng, jitter=float(rng.uniform(0.0, 0.12))))
        elif fam == "shared":
            cands.append(candidate_shared(rng))
        elif fam == "dark":
            cands.append(candidate_dark(rng))
        elif fam == "slacky":
            cands.append(candidate_slacky(rng))
        else:
            cands.append(candidate_random(rng))
    out = []
    for k, c in enumerate(cands):
        out.append(Candidate(f"{c.name}_{k:03d}", c.strengths, c.slack_dims, c.family))
    return out

def singular_metrics(M: np.ndarray) -> Dict[str, object]:
    s = np.linalg.svd(M, compute_uv=False)
    s2 = np.abs(s)**2
    total = float(np.sum(s2))
    op = float(np.max(np.abs(s))) if len(s) else 0.0
    rank = int(np.sum(np.abs(s) > 1e-12))
    stable_rank = 0.0 if op <= 1e-15 else float(total / (op**2))
    if total <= 1e-15:
        entropy_rank = 0.0
        top_frac = 0.0
    else:
        p = s2 / total
        nz = p[p > 1e-15]
        entropy_rank = float(np.exp(-np.sum(nz*np.log(nz))))
        top_frac = float(np.max(s2) / total)
    return {"operator_gain": op, "rank": rank, "stable_rank": stable_rank, "entropy_rank": entropy_rank,
            "top_mode_energy_fraction": top_frac, "singular_values": [float(x) for x in s.tolist()]}

def visibility_stats(strengths: np.ndarray) -> Dict[str, object]:
    seen_count = np.sum(strengths > 1e-15, axis=0)
    dark_idx = [int(j) for j in np.where(seen_count == 0)[0]]
    exclusive_idx = [int(j) for j in np.where(seen_count == 1)[0]]
    shared_idx = [int(j) for j in np.where(seen_count > 1)[0]]
    overlaps = []
    for i in range(strengths.shape[0]):
        vi = strengths[i]; ni = float(np.linalg.norm(vi))
        for j in range(i+1, strengths.shape[0]):
            vj = strengths[j]; nj = float(np.linalg.norm(vj))
            overlaps.append(0.0 if ni <= 1e-15 or nj <= 1e-15 else float(np.dot(vi, vj)/(ni*nj)))
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    duplicate_overclaim = float(np.sum(np.maximum(0.0, np.sum(strengths, axis=0) - np.max(strengths, axis=0))))
    return {"dark_idx": dark_idx, "exclusive_idx": exclusive_idx, "shared_idx": shared_idx,
            "dark_frac": float(len(dark_idx)/8.0), "exclusive_frac": float(len(exclusive_idx)/8.0),
            "shared_frac": float(len(shared_idx)/8.0), "mean_link_overlap": mean_overlap,
            "duplicate_overclaim": duplicate_overclaim}

def no_signaling_proxy(vs): return 0.70*float(vs["shared_frac"]) + 0.30*float(vs["mean_link_overlap"])
def no_forgetting_proxy(sm, vs):
    target_visible_rank = 8 - len(vs["dark_idx"])
    if target_visible_rank <= 0: return 1.0
    rank_loss = max(0.0, (target_visible_rank - sm["rank"]) / max(1.0, target_visible_rank))
    return 0.65*float(vs["dark_frac"]) + 0.35*rank_loss
def no_refolding_proxy(vs, slack_dims):
    dup = float(vs["duplicate_overclaim"])
    dup_scaled = dup / (dup + 4.0) if dup > 0 else 0.0
    slack_scaled = float(slack_dims) / (float(slack_dims) + 4.0)
    return 0.40*float(vs["dark_frac"]) + 0.30*dup_scaled + 0.30*slack_scaled
def redundancy_proxy(vs):
    dup = float(vs["duplicate_overclaim"])
    dup_scaled = dup / (dup + 4.0) if dup > 0 else 0.0
    return 0.55*dup_scaled + 0.45*float(vs["mean_link_overlap"])
def dark_proxy(vs): return float(vs["dark_frac"])
def bandwidth_spread_proxy(sm):
    stable_scaled = float(sm["stable_rank"]) / 8.0
    entropy_scaled = float(sm["entropy_rank"]) / 8.0
    concentration_pen = 1.0 - float(sm["top_mode_energy_fraction"])
    return 0.45*stable_scaled + 0.35*entropy_scaled + 0.20*concentration_pen

def delta_probe_metrics(M, seed, poke_idx, time):
    coupling_weights = np.array([1.00,0.82,0.74,0.58,0.47,0.40,0.33,0.66], dtype=float)
    dB = induced_delta_B(0.035, time, poke_idx, coupling_weights, 0.17, 0.11, seed)
    cB = gm_coeffs(dB)
    y = M @ cB
    B_norm = float(np.linalg.norm(cB))
    Y_norm = float(np.linalg.norm(y))
    frac = 0.0 if B_norm <= 1e-15 else float(min(1.0, Y_norm / B_norm))
    return {"B_norm": B_norm, "stacked_out_norm": Y_norm, "visible_fraction": frac}

def evaluate_candidate(c: Candidate, priorities, seed, poke_idx, time):
    M = c.map_matrix(priority_weighting=priorities)
    sm = singular_metrics(M)
    vs = visibility_stats(c.strengths)
    dp = delta_probe_metrics(M, seed=seed, poke_idx=poke_idx, time=time)
    objs = {"no_signaling": float(no_signaling_proxy(vs)),
            "no_forgetting": float(no_forgetting_proxy(sm, vs)),
            "no_refolding": float(no_refolding_proxy(vs, c.slack_dims)),
            "redundancy": float(redundancy_proxy(vs)),
            "darkness": float(dark_proxy(vs)),
            "bandwidth_spread": float(bandwidth_spread_proxy(sm)),
            "outward_visibility_reward": float(dp["visible_fraction"])}
    return {"name": c.name, "family": c.family, "slack_dims": int(c.slack_dims),
            "objectives": objs, "visibility": vs, "singular_metrics": sm, "delta_probe": dp,
            "strengths": c.strengths.tolist()}

MIN_KEYS = ["no_signaling","no_forgetting","no_refolding","redundancy","darkness","bandwidth_spread"]
MAX_KEYS = ["outward_visibility_reward"]

def dominates(a, b, tol: float = 1e-12) -> bool:
    ao = a["objectives"]; bo = b["objectives"]
    no_worse = True; strictly_better = False
    for k in MIN_KEYS:
        if ao[k] > bo[k] + tol:
            no_worse = False; break
        if ao[k] < bo[k] - tol:
            strictly_better = True
    if no_worse:
        for k in MAX_KEYS:
            if ao[k] < bo[k] - tol:
                no_worse = False; break
            if ao[k] > bo[k] + tol:
                strictly_better = True
    return bool(no_worse and strictly_better)

def pareto_front(results):
    idxs = []
    for i, ri in enumerate(results):
        dominated = False
        for j, rj in enumerate(results):
            if i != j and dominates(rj, ri):
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs

def normalized_objective_matrix(front):
    cols = []
    for k in MIN_KEYS:
        cols.append(np.array([r["objectives"][k] for r in front], dtype=float))
    for k in MAX_KEYS:
        cols.append(-np.array([r["objectives"][k] for r in front], dtype=float))
    X = np.stack(cols, axis=1) if cols else np.zeros((len(front),0))
    if X.shape[0] == 0:
        return X
    mins = X.min(axis=0); maxs = X.max(axis=0)
    den = np.where(maxs - mins > 1e-15, maxs - mins, 1.0)
    return (X - mins) / den

def crowding_proxy(front):
    X = normalized_objective_matrix(front)
    n = X.shape[0]
    if n == 0: return []
    if n == 1: return [0.0]
    out = []
    for i in range(n):
        d = np.linalg.norm(X[i] - X, axis=1)
        d = np.delete(d, i)
        out.append(float(np.mean(d)) if len(d) else 0.0)
    return out

def display_order(front):
    scores = []
    for i, r in enumerate(front):
        o = r["objectives"]
        key = (-o["outward_visibility_reward"], o["darkness"], o["redundancy"], o["no_refolding"], o["bandwidth_spread"])
        scores.append((key, i))
    scores.sort()
    return [i for _, i in scores]

def pretty_result(r, crowd=None):
    o = r["objectives"]; vs = r["visibility"]; sm = r["singular_metrics"]; dp = r["delta_probe"]
    lines = []
    head = f"{r['name']} | family={r['family']} | slack_dims={r['slack_dims']}"
    if crowd is not None: head += f" | crowding={crowd:.3f}"
    lines.append("="*112); lines.append(head); lines.append("-"*112)
    lines.append(f"objectives: NS={o['no_signaling']:.3f}  NF={o['no_forgetting']:.3f}  NR={o['no_refolding']:.3f}  RED={o['redundancy']:.3f}  DARK={o['darkness']:.3f}  BW={o['bandwidth_spread']:.3f}  VIS_REWARD={o['outward_visibility_reward']:.3f}")
    lines.append(f"visibility: dark={vs['dark_idx']}  exclusive={vs['exclusive_idx']}  shared={vs['shared_idx']}  overlap={vs['mean_link_overlap']:.3f}  dup={vs['duplicate_overclaim']:.3f}")
    lines.append(f"spectrum: s={[round(x,4) for x in sm['singular_values']]}  stable_rank={sm['stable_rank']:.3f}  entropy_rank={sm['entropy_rank']:.3f}  top_frac={sm['top_mode_energy_fraction']:.3f}")
    lines.append(f"probe: B_norm={dp['B_norm']:.6e}  stacked_out={dp['stacked_out_norm']:.6e}  visible_fraction={dp['visible_fraction']:.6f}")
    return "\\n".join(lines)

def parse_args():
    ap = argparse.ArgumentParser(description="Pareto-front link/interface selection for SU(3) HSF scaffolding.")
    ap.add_argument("--n-random", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poke-idx", type=int, default=2, choices=list(range(8)))
    ap.add_argument("--time", type=float, default=1.20)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--priority-weighting", action="store_true")
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()

def main():
    args = parse_args()
    priorities = np.array([1.8, 1.0, 1.0], dtype=float) if args.priority_weighting else None
    candidates = generate_candidates(n_random=args.n_random, seed=args.seed)
    results = [evaluate_candidate(c, priorities=priorities, seed=args.seed, poke_idx=args.poke_idx, time=args.time) for c in candidates]
    pidx = pareto_front(results)
    front = [results[i] for i in pidx]
    crowd = crowding_proxy(front)
    order = display_order(front)
    fam_counts_all = {}
    for r in results: fam_counts_all[r["family"]] = fam_counts_all.get(r["family"], 0) + 1
    fam_counts_front = {}
    for r in front: fam_counts_front[r["family"]] = fam_counts_front.get(r["family"], 0) + 1
    print(); print("HSF PARETO LINK SELECTION TEST (SU3 v1)"); print()
    print(f"Candidates evaluated: {len(results)}")
    print(f"Pareto frontier size: {len(front)}")
    print(f"Priority weighting: {bool(args.priority_weighting)}"); print()
    print(f"All families: {fam_counts_all}")
    print(f"Frontier families: {fam_counts_front}"); print()
    for rank, j in enumerate(order[:max(1, int(args.top_k))], start=1):
        print(f"[Frontier display rank {rank}]")
        print(pretty_result(front[j], crowd=crowd[j])); print()
    if args.json_out:
        payload = {"all_results": results, "pareto_indices": pidx, "pareto_front": front, "crowding_proxy": crowd,
                   "display_order": order, "family_counts_all": fam_counts_all, "family_counts_front": fam_counts_front}
        with open(args.json_out, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
        print(f"Saved JSON: {args.json_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
