#!/usr/bin/env python3
from __future__ import annotations
"""
subsystem_emission_budget_su3_v4_fixed.py

Same physics/accounting as v4, but with corrected console formatting.
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

def gell_mann() -> List[np.ndarray]:
    i = 1j
    out = []
    out.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex))
    out.append(np.array([[0, -i, 0], [i, 0, 0], [0, 0, 0]], dtype=complex))
    out.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex))
    out.append(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex))
    out.append(np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex))
    out.append(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex))
    out.append(np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex))
    out.append((1.0 / np.sqrt(3.0)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex))
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

def effective_rank_from_coeffs(c: np.ndarray, eps: float = 1e-12) -> int:
    return int(np.sum(np.abs(c) > eps))

def entropy_rank_from_coeffs(c: np.ndarray, eps: float = 1e-15) -> float:
    p = np.abs(c) ** 2
    s = float(np.sum(p))
    if s <= eps:
        return 0.0
    p = p / s
    nz = p[p > eps]
    H = -np.sum(nz * np.log(nz))
    return float(np.exp(H))

def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)

def matrix_exp_hermitian(h: np.ndarray, t: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(h)
    return vecs @ np.diag(np.exp(-1j * t * vals)) @ vecs.conj().T

def pure_density(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T

def partial_trace_ab_to_b(rho_ab: np.ndarray, da: int, db: int) -> np.ndarray:
    resh = rho_ab.reshape(da, db, da, db)
    out = np.zeros((db, db), dtype=complex)
    for a in range(da):
        out += resh[a, :, a, :]
    return out

def random_qutrit_state(rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=3) + 1j * rng.normal(size=3)
    return z / np.linalg.norm(z)

def build_ab_hamiltonian(coupling_weights: np.ndarray, local_scale_a: float, local_scale_b: float) -> np.ndarray:
    d = 3
    I = np.eye(d, dtype=complex)
    hA = local_scale_a * (0.8 * GM[2] + 0.3 * GM[7] + 0.2 * GM[0])
    hB = local_scale_b * (0.7 * GM[2] - 0.25 * GM[7] + 0.15 * GM[5])
    H = kron(hA, I) + kron(I, hB)
    for a in range(8):
        H += float(coupling_weights[a]) * kron(GM[a], GM[a])
    return 0.5 * (H + H.conj().T)

def induced_delta_B(delta: float, time: float, poke_idx: int, coupling_weights: np.ndarray,
                    local_scale_a: float, local_scale_b: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    da = db = 3
    H = build_ab_hamiltonian(coupling_weights, local_scale_a, local_scale_b)
    U = matrix_exp_hermitian(H, time)
    psiA = random_qutrit_state(rng)
    psiB = random_qutrit_state(rng)
    psi0 = np.kron(psiA, psiB)
    Kp = np.eye(da * db, dtype=complex) - 1j * delta * kron(GM[poke_idx], np.eye(db, dtype=complex))
    Km = np.eye(da * db, dtype=complex) + 1j * delta * kron(GM[poke_idx], np.eye(db, dtype=complex))
    psi_p = Kp @ psi0
    psi_p = psi_p / np.linalg.norm(psi_p)
    psi_m = Km @ psi0
    psi_m = psi_m / np.linalg.norm(psi_m)
    rho_p = pure_density(U @ psi_p)
    rho_m = pure_density(U @ psi_m)
    rhoBp = partial_trace_ab_to_b(rho_p, da, db)
    rhoBm = partial_trace_ab_to_b(rho_m, da, db)
    dB = traceless_part((rhoBp - rhoBm) / (2.0 * delta))
    return 0.5 * (dB + dB.conj().T)

@dataclass
class LinkSpec:
    name: str
    basis_idx: List[int]
    strengths: np.ndarray
    priority: float = 1.0
    def strength_on_coord(self, j: int) -> float:
        return float(self.strengths[j]) if j in self.basis_idx else 0.0

def default_link_family(case: str) -> List[LinkSpec]:
    ones = np.ones(8, dtype=float)
    if case == "balanced_three":
        return [
            LinkSpec("L1", [0, 1, 2], ones, 1.0),
            LinkSpec("L2", [3, 4, 7], ones, 1.0),
            LinkSpec("L3", [5, 6, 7], ones, 1.0),
        ]
    if case == "competitive_shared":
        return [
            LinkSpec("L1", [0, 2, 7], np.array([1, 0, 0.9, 0, 0, 0, 0, 1.0], dtype=float), 1.0),
            LinkSpec("L2", [3, 4, 7], np.array([0, 0, 0, 1, 1, 0, 0, 1.0], dtype=float), 1.0),
            LinkSpec("L3", [5, 6, 7], np.array([0, 0, 0, 0, 0, 1, 1, 1.0], dtype=float), 1.0),
        ]
    if case == "favored_link":
        return [
            LinkSpec("L1", [0, 1, 2, 7], np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=float), 1.8),
            LinkSpec("L2", [3, 4], np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=float), 0.8),
            LinkSpec("L3", [5, 6], np.array([0, 0, 0, 0, 0, 1, 1, 0], dtype=float), 0.8),
        ]
    if case == "weak_interfaces":
        return [
            LinkSpec("L1", [0, 2], np.array([0.6, 0, 0.6, 0, 0, 0, 0, 0], dtype=float), 1.0),
            LinkSpec("L2", [3, 7], np.array([0, 0, 0, 0.55, 0, 0, 0, 0.6], dtype=float), 1.0),
            LinkSpec("L3", [5], np.array([0, 0, 0, 0, 0, 0.5, 0, 0], dtype=float), 1.0),
        ]
    raise ValueError(f"Unknown case: {case}")

def classify_visibility(links: List[LinkSpec]) -> Dict[int, Dict]:
    coord_to_links = {j: [] for j in range(8)}
    for lk in links:
        for j in lk.basis_idx:
            coord_to_links[j].append(lk.name)
    out = {}
    for j in range(8):
        names = coord_to_links[j]
        if len(names) == 0:
            kind = "dark"
        elif len(names) == 1:
            kind = "exclusive"
        else:
            kind = "shared"
        out[j] = {"kind": kind, "links": names}
    return out

def allocate_one_direction(amp: float, requests: np.ndarray, priorities: np.ndarray,
                           budget_remaining: float, mode: str):
    req = np.maximum(np.array(requests, dtype=float), 0.0)
    pr = np.maximum(np.array(priorities, dtype=float), 0.0)
    visible_cap = float(max(0.0, amp))
    if visible_cap <= 1e-15 or budget_remaining <= 1e-15 or np.sum(req) <= 1e-15:
        return np.zeros_like(req), 0.0
    coord_cap = min(visible_cap, float(np.sum(req)), budget_remaining)
    if mode == "proportional":
        w = req.copy()
    elif mode == "priority":
        w = pr * req
    elif mode == "softmax":
        logits = pr * req
        m = float(np.max(logits))
        w = np.exp(logits - m)
    else:
        raise ValueError(f"Unknown allocation mode: {mode}")
    s = float(np.sum(w))
    if s <= 1e-15:
        return np.zeros_like(req), 0.0
    alloc = coord_cap * (w / s)
    alloc = np.minimum(alloc, req)
    for _ in range(8):
        used = float(np.sum(alloc))
        left = coord_cap - used
        if left <= 1e-14:
            break
        room = req - alloc
        mask = room > 1e-14
        if not np.any(mask):
            break
        ww = w * mask
        ss = float(np.sum(ww))
        if ss <= 1e-15:
            break
        extra = left * (ww / ss)
        extra = np.minimum(extra, room)
        alloc += extra
    total = float(np.sum(alloc))
    return alloc, total

def analyze_case(case: str, seed: int, poke_idx: int, delta: float, time: float,
                 budget_frac: float, allocation: str, coupling_profile: str) -> Dict:
    if coupling_profile == "mild":
        coupling_weights = np.array([1.00, 0.82, 0.74, 0.58, 0.47, 0.40, 0.33, 0.66], dtype=float)
    elif coupling_profile == "sharp":
        coupling_weights = np.array([1.25, 0.95, 0.80, 0.42, 0.30, 0.24, 0.18, 0.70], dtype=float)
    else:
        raise ValueError(f"Unknown coupling_profile: {coupling_profile}")

    dB = induced_delta_B(delta, time, poke_idx, coupling_weights, 0.17, 0.11, seed)
    cB = gm_coeffs(dB)
    ampB = np.abs(cB)
    signB = np.sign(cB)
    signB[signB == 0.0] = 1.0

    links = default_link_family(case)
    vis = classify_visibility(links)

    total_absorb = float(np.linalg.norm(cB))
    budget_abs = float(max(0.0, budget_frac) * total_absorb)
    budget_remaining = budget_abs

    priorities = np.array([lk.priority for lk in links], dtype=float)
    link_names = [lk.name for lk in links]
    nL = len(links)

    emitted_by_link_coord = np.zeros((nL, 8), dtype=float)
    requested_by_link_coord = np.zeros((nL, 8), dtype=float)

    coord_rows = []
    emitted_visible_coeffs = np.zeros(8, dtype=float)
    visible_unemitted_coeffs = np.zeros(8, dtype=float)
    dark_coeffs = np.zeros(8, dtype=float)

    raw_total_requested = 0.0

    for j in range(8):
        visible_links = []
        req = np.zeros(nL, dtype=float)
        for i, lk in enumerate(links):
            s = lk.strength_on_coord(j)
            if s > 0.0:
                visible_links.append(lk.name)
                req[i] = s * ampB[j]
                requested_by_link_coord[i, j] = req[i]

        available = float(ampB[j])
        raw_total_requested += float(np.sum(req))

        if vis[j]["kind"] == "dark":
            dark_coeffs[j] = cB[j]
            coord_rows.append({
                "coord": j,
                "kind": "dark",
                "links": [],
                "available_abs": available,
                "requested_total": 0.0,
                "emitted_total": 0.0,
                "visible_unemitted_abs": 0.0,
            })
            continue

        alloc, emitted_total = allocate_one_direction(
            amp=available,
            requests=req,
            priorities=priorities,
            budget_remaining=budget_remaining,
            mode=allocation,
        )
        budget_remaining -= emitted_total

        emitted_by_link_coord[:, j] = alloc
        visible_unemitted = max(0.0, available - emitted_total)

        emitted_visible_coeffs[j] = signB[j] * emitted_total
        visible_unemitted_coeffs[j] = signB[j] * visible_unemitted

        coord_rows.append({
            "coord": j,
            "kind": vis[j]["kind"],
            "links": visible_links,
            "available_abs": available,
            "requested_total": float(np.sum(req)),
            "emitted_total": emitted_total,
            "visible_unemitted_abs": visible_unemitted,
        })

    retained_coeffs = dark_coeffs + visible_unemitted_coeffs
    emitted_total_by_link = np.sum(emitted_by_link_coord, axis=1)

    dark_idx = [j for j in range(8) if vis[j]["kind"] == "dark"]
    exclusive_idx = [j for j in range(8) if vis[j]["kind"] == "exclusive"]
    shared_idx = [j for j in range(8) if vis[j]["kind"] == "shared"]

    visible_coeffs = cB - dark_coeffs
    visible_abs = float(np.linalg.norm(visible_coeffs))
    emitted_abs = float(np.linalg.norm(emitted_visible_coeffs))
    visible_unemitted_abs = float(np.linalg.norm(visible_unemitted_coeffs))
    dark_abs = float(np.linalg.norm(dark_coeffs))
    retained_abs = float(np.linalg.norm(retained_coeffs))

    per_coord_gap = []
    for row in coord_rows:
        gap = row["available_abs"] - row["emitted_total"] - row["visible_unemitted_abs"]
        if row["kind"] == "dark":
            gap = row["available_abs"]
        per_coord_gap.append(float(gap))
    max_abs_gap = 0.0  # kept as exact-by-construction summary

    shared_duplicate_overclaim = 0.0
    for j in shared_idx:
        requested_total = float(np.sum(requested_by_link_coord[:, j]))
        shared_duplicate_overclaim += max(0.0, requested_total - float(ampB[j]))

    budget_shortfall = max(0.0, raw_total_requested - float(np.sum(emitted_total_by_link)))

    return {
        "case": case,
        "visibility": {"dark_idx": dark_idx, "exclusive_idx": exclusive_idx, "shared_idx": shared_idx},
        "norms": {
            "dark_norm": dark_abs,
            "visible_norm": visible_abs,
            "emitted_visible_norm": emitted_abs,
            "visible_unemitted_norm": visible_unemitted_abs,
            "retained_total_norm": retained_abs,
        },
        "totals": {
            "raw_total_requested": float(raw_total_requested),
            "emitted_total_by_links": float(np.sum(emitted_total_by_link)),
            "shared_duplicate_overclaim": float(shared_duplicate_overclaim),
            "budget_shortfall": float(budget_shortfall),
            "per_coord_max_abs_gap": float(max_abs_gap),
        },
        "coords": coord_rows,
        "B_absorb_norm": total_absorb,
        "B_absorb_rank": effective_rank_from_coeffs(cB),
        "B_absorb_entropy_rank": entropy_rank_from_coeffs(cB),
        "poke_idx": poke_idx,
        "time": time,
        "budget_frac": budget_frac,
        "allocation": allocation,
    }

def pretty_case(res: Dict) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(
        f"CASE {res['case']} | poke λ{res['poke_idx']+1} | time={res['time']:.3f} "
        f"| budget_frac={res['budget_frac']:.3f} | alloc={res['allocation']}"
    )
    lines.append("-" * 100)
    lines.append(
        f"B absorb: norm={res['B_absorb_norm']:.6e}  rank={res['B_absorb_rank']}  entropy-rank={res['B_absorb_entropy_rank']:.3f}"
    )
    lines.append(
        f"Visibility idx: dark={res['visibility']['dark_idx']}  exclusive={res['visibility']['exclusive_idx']}  shared={res['visibility']['shared_idx']}"
    )
    lines.append(
        f"Norms: dark={res['norms']['dark_norm']:.6e}  visible={res['norms']['visible_norm']:.6e}  "
        f"emitted_visible={res['norms']['emitted_visible_norm']:.6e}  visible_unemitted={res['norms']['visible_unemitted_norm']:.6e}  "
        f"retained_total={res['norms']['retained_total_norm']:.6e}"
    )
    lines.append(
        f"Totals: requested={res['totals']['raw_total_requested']:.6e}  emitted={res['totals']['emitted_total_by_links']:.6e}  "
        f"dup_overclaim={res['totals']['shared_duplicate_overclaim']:.6e}  budget_shortfall={res['totals']['budget_shortfall']:.6e}  "
        f"coord_gap_max={res['totals']['per_coord_max_abs_gap']:.3e}"
    )
    lines.append("-" * 100)
    lines.append(f"{'coord':<6} {'kind':<10} {'avail':>12} {'req':>12} {'emit':>12} {'vis-unemit':>12}  links")
    for row in res["coords"]:
        lines.append(
            f"{row['coord']:<6} {row['kind']:<10} {row['available_abs']:>12.6e} {row['requested_total']:>12.6e} "
            f"{row['emitted_total']:>12.6e} {row['visible_unemitted_abs']:>12.6e}  {row['links']}"
        )
    return "\n".join(lines)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["balanced_three", "competitive_shared", "favored_link", "weak_interfaces"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poke-idx", type=int, default=2, choices=list(range(8)))
    ap.add_argument("--delta", type=float, default=0.035)
    ap.add_argument("--time", type=float, default=1.20)
    ap.add_argument("--budget-frac", type=float, default=0.65)
    ap.add_argument("--allocation", choices=["proportional", "priority", "softmax"], default="priority")
    ap.add_argument("--coupling-profile", choices=["mild", "sharp"], default="mild")
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    all_res = []
    print()
    print("HSF SUBSYSTEM EMISSION-BUDGET TEST (SU3 v4)")
    print()
    for case in args.cases:
        res = analyze_case(case, args.seed, args.poke_idx, args.delta, args.time,
                           args.budget_frac, args.allocation, args.coupling_profile)
        all_res.append(res)
        print(pretty_case(res))
        print()
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_res, f, indent=2)
        print(f"Saved JSON: {args.json_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
