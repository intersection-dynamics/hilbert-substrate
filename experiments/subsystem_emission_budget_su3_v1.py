#!/usr/bin/env python3
from __future__ import annotations
"""
subsystem_emission_budget_su3_v1.py

HSF-oriented experiment:
A poke on subsystem A perturbs subsystem B through an A-B interaction.
B does NOT broadcast its whole state to every attached link.
Instead, each attached link samples only the component of B's changed
operator content that overlaps its inherited endpoint/interface sector.

Main stages
-----------
1. Build an induced traceless operator response Δ_B on subsystem B from an A-side poke.
2. Define multiple attached links L_i, each with its own inherited interface sector on B.
3. Project Δ_B into each link's interface sector.
4. Enforce a finite total emission budget across all attached links.
5. Report total induced change on B, per-link requests/emissions, and retained remainder.
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


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
    out.append(
        (1.0 / np.sqrt(3.0))
        * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex)
    )
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


def coeffs_to_op(c: np.ndarray) -> np.ndarray:
    out = np.zeros((3, 3), dtype=complex)
    for a, val in enumerate(c):
        out += float(val) * (GM[a] / SQRT2)
    return out


def project_coeffs(
    c: np.ndarray, basis_idx: List[int], weights: np.ndarray | None = None
) -> np.ndarray:
    out = np.zeros_like(c)
    for j in basis_idx:
        out[j] = (1.0 if weights is None else float(weights[j])) * c[j]
    return out


def effective_rank_from_coeffs(c: np.ndarray, eps: float = 1e-12) -> int:
    return int(np.sum(np.abs(c) > eps))


def entropy_rank_from_coeffs(c: np.ndarray, eps: float = 1e-15) -> float:
    p = np.abs(c) ** 2
    s = p.sum()
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


def build_ab_hamiltonian(
    coupling_weights: np.ndarray, local_scale_a: float, local_scale_b: float
) -> np.ndarray:
    d = 3
    I = np.eye(d, dtype=complex)
    hA = local_scale_a * (0.8 * GM[2] + 0.3 * GM[7] + 0.2 * GM[0])
    hB = local_scale_b * (0.7 * GM[2] - 0.25 * GM[7] + 0.15 * GM[5])
    H = kron(hA, I) + kron(I, hB)
    for a in range(8):
        H += float(coupling_weights[a]) * kron(GM[a], GM[a])
    return 0.5 * (H + H.conj().T)


def induced_delta_B(
    delta: float,
    time: float,
    poke_idx: int,
    coupling_weights: np.ndarray,
    local_scale_a: float,
    local_scale_b: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    da = db = 3
    H = build_ab_hamiltonian(coupling_weights, local_scale_a, local_scale_b)
    U = matrix_exp_hermitian(H, time)

    psiA = random_qutrit_state(rng)
    psiB = random_qutrit_state(rng)
    psi0 = np.kron(psiA, psiB)

    Kp = np.eye(da * db, dtype=complex) - 1j * delta * kron(
        GM[poke_idx], np.eye(db, dtype=complex)
    )
    Km = np.eye(da * db, dtype=complex) + 1j * delta * kron(
        GM[poke_idx], np.eye(db, dtype=complex)
    )

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

    def projection_coeffs(self, cB: np.ndarray) -> np.ndarray:
        return project_coeffs(cB, self.basis_idx, self.strengths)

    def overlap_fraction(self, cB: np.ndarray) -> float:
        denom = float(np.dot(cB, cB))
        if denom <= 1e-15:
            return 0.0
        proj = self.projection_coeffs(cB)
        return float(np.dot(proj, proj)) / denom


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
            LinkSpec(
                "L1",
                [0, 2, 7],
                np.array([1, 0, 0.9, 0, 0, 0, 0, 1.0], dtype=float),
                1.0,
            ),
            LinkSpec(
                "L2",
                [3, 4, 7],
                np.array([0, 0, 0, 1, 1, 0, 0, 1.0], dtype=float),
                1.0,
            ),
            LinkSpec(
                "L3",
                [5, 6, 7],
                np.array([0, 0, 0, 0, 0, 1, 1, 1.0], dtype=float),
                1.0,
            ),
        ]

    if case == "favored_link":
        return [
            LinkSpec(
                "L1",
                [0, 1, 2, 7],
                np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=float),
                1.8,
            ),
            LinkSpec(
                "L2",
                [3, 4],
                np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=float),
                0.8,
            ),
            LinkSpec(
                "L3",
                [5, 6],
                np.array([0, 0, 0, 0, 0, 1, 1, 0], dtype=float),
                0.8,
            ),
        ]

    if case == "weak_interfaces":
        return [
            LinkSpec(
                "L1",
                [0, 2],
                np.array([0.6, 0, 0.6, 0, 0, 0, 0, 0], dtype=float),
                1.0,
            ),
            LinkSpec(
                "L2",
                [3, 7],
                np.array([0, 0, 0, 0.55, 0, 0, 0, 0.6], dtype=float),
                1.0,
            ),
            LinkSpec(
                "L3",
                [5],
                np.array([0, 0, 0, 0, 0, 0.5, 0, 0], dtype=float),
                1.0,
            ),
        ]

    raise ValueError(f"Unknown case: {case}")


def allocate_budget(
    raw_strengths: np.ndarray, priorities: np.ndarray, budget: float, mode: str
) -> np.ndarray:
    raw = np.maximum(np.array(raw_strengths, dtype=float), 0.0)
    pr = np.maximum(np.array(priorities, dtype=float), 0.0)
    total_raw = float(np.sum(raw))

    if total_raw <= 1e-15 or budget <= 0.0:
        return np.zeros_like(raw)

    if total_raw <= budget:
        return raw.copy()

    if mode == "proportional":
        return budget * raw / total_raw

    if mode == "priority":
        weights = pr * raw
        s = float(np.sum(weights))
        return np.zeros_like(raw) if s <= 1e-15 else budget * weights / s

    if mode == "softmax":
        logits = pr * raw
        m = float(np.max(logits))
        w = np.exp(logits - m)
        s = float(np.sum(w))
        alloc = np.zeros_like(raw) if s <= 1e-15 else budget * (w / s)
        return np.minimum(alloc, raw)

    raise ValueError(f"Unknown allocation mode: {mode}")


def analyze_case(
    case: str,
    seed: int,
    poke_idx: int,
    delta: float,
    time: float,
    budget_frac: float,
    allocation: str,
    coupling_profile: str,
) -> Dict:
    if coupling_profile == "mild":
        coupling_weights = np.array(
            [1.00, 0.82, 0.74, 0.58, 0.47, 0.40, 0.33, 0.66], dtype=float
        )
    elif coupling_profile == "sharp":
        coupling_weights = np.array(
            [1.25, 0.95, 0.80, 0.42, 0.30, 0.24, 0.18, 0.70], dtype=float
        )
    else:
        raise ValueError(f"Unknown coupling_profile: {coupling_profile}")

    dB = induced_delta_B(
        delta, time, poke_idx, coupling_weights, 0.17, 0.11, seed
    )
    cB = gm_coeffs(dB)
    total_absorb = float(np.linalg.norm(cB))

    links = default_link_family(case)

    raw_coeffs = []
    raw_strengths = []
    overlap_fracs = []
    priorities = []

    for lk in links:
        proj = lk.projection_coeffs(cB)
        raw_coeffs.append(proj)
        raw_strengths.append(float(np.linalg.norm(proj)))
        overlap_fracs.append(lk.overlap_fraction(cB))
        priorities.append(lk.priority)

    raw_strengths = np.array(raw_strengths, dtype=float)
    priorities = np.array(priorities, dtype=float)

    budget = float(max(0.0, budget_frac) * total_absorb)
    emitted_strengths = allocate_budget(raw_strengths, priorities, budget, allocation)

    emitted_coeffs = []
    retained_coeffs = cB.copy()
    for i, proj in enumerate(raw_coeffs):
        r = raw_strengths[i]
        emitted = np.zeros_like(proj) if r <= 1e-15 else proj * (emitted_strengths[i] / r)
        emitted_coeffs.append(emitted)
        retained_coeffs = retained_coeffs - emitted

    retained_op = coeffs_to_op(retained_coeffs)

    stacked = np.stack(raw_coeffs, axis=0) if raw_coeffs else np.zeros((0, 8), dtype=float)
    union_mask = (
        np.sum(np.abs(stacked), axis=0) > 1e-15
        if len(raw_coeffs)
        else np.zeros(8, dtype=bool)
    )
    union_coeffs = np.where(union_mask, cB, 0.0)
    union_norm = float(np.linalg.norm(union_coeffs))
    raw_sum = float(np.sum(raw_strengths))

    link_rows = []
    for i, lk in enumerate(links):
        proj = raw_coeffs[i]
        em = emitted_coeffs[i]
        link_rows.append(
            {
                "name": lk.name,
                "priority": float(lk.priority),
                "basis_idx": list(lk.basis_idx),
                "raw_norm": float(np.linalg.norm(proj)),
                "emitted_norm": float(np.linalg.norm(em)),
                "overlap_fraction_of_B": float(overlap_fracs[i]),
                "raw_rank": effective_rank_from_coeffs(proj),
                "emitted_rank": effective_rank_from_coeffs(em),
                "raw_coeffs": [float(x) for x in proj.tolist()],
                "emitted_coeffs": [float(x) for x in em.tolist()],
            }
        )

    return {
        "case": case,
        "seed": int(seed),
        "poke_idx": int(poke_idx),
        "time": float(time),
        "delta": float(delta),
        "budget_frac": float(budget_frac),
        "budget_abs": float(budget),
        "allocation": allocation,
        "coupling_profile": coupling_profile,
        "B_absorb_norm": total_absorb,
        "B_absorb_rank": effective_rank_from_coeffs(cB),
        "B_absorb_entropy_rank": entropy_rank_from_coeffs(cB),
        "B_coeffs": [float(x) for x in cB.tolist()],
        "raw_total_requested": raw_sum,
        "emitted_total": float(np.sum(emitted_strengths)),
        "retained_norm": float(np.linalg.norm(retained_coeffs)),
        "competition_excess": float(np.sum(raw_strengths) - np.sum(emitted_strengths)),
        "competition_ratio": 0.0
        if raw_sum <= 1e-15
        else float(np.sum(raw_strengths) - np.sum(emitted_strengths)) / raw_sum,
        "overlap_pressure": 0.0
        if raw_sum <= 1e-15
        else max(0.0, raw_sum - union_norm) / raw_sum,
        "retained_rank": effective_rank_from_coeffs(retained_coeffs),
        "retained_entropy_rank": entropy_rank_from_coeffs(retained_coeffs),
        "retained_trace_zero_check": float(abs(np.trace(retained_op))),
        "links": link_rows,
    }


def pretty_case(res: Dict) -> str:
    lines = []
    lines.append("=" * 84)
    lines.append(
        f"CASE {res['case']} | poke λ{res['poke_idx']+1} | time={res['time']:.3f} "
        f"| budget_frac={res['budget_frac']:.3f} | alloc={res['allocation']}"
    )
    lines.append("-" * 84)
    lines.append(
        f"B absorb: norm={res['B_absorb_norm']:.6e}  "
        f"rank={res['B_absorb_rank']}  "
        f"entropy-rank={res['B_absorb_entropy_rank']:.3f}"
    )
    lines.append(
        f"Requested total={res['raw_total_requested']:.6e}  "
        f"Budgeted total={res['emitted_total']:.6e}  "
        f"Retained/internal={res['retained_norm']:.6e}"
    )
    lines.append(
        f"Competition ratio={res['competition_ratio']:.3f}  "
        f"Overlap pressure={res['overlap_pressure']:.3f}  "
        f"Retained rank={res['retained_rank']}"
    )
    lines.append("-" * 84)
    lines.append(
        f"{'Link':<8} {'prio':>5} {'raw':>12} {'emit':>12} "
        f"{'frac(B)':>10} {'r_rank':>7} {'e_rank':>7}  basis"
    )
    for lk in res["links"]:
        lines.append(
            f"{lk['name']:<8} {lk['priority']:>5.2f} "
            f"{lk['raw_norm']:>12.6e} {lk['emitted_norm']:>12.6e} "
            f"{lk['overlap_fraction_of_B']:>10.3f} "
            f"{lk['raw_rank']:>7d} {lk['emitted_rank']:>7d}  {lk['basis_idx']}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="HSF subsystem emission-budget witness on SU(3) operator content."
    )
    ap.add_argument(
        "--cases",
        nargs="+",
        default=["balanced_three", "competitive_shared", "favored_link", "weak_interfaces"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poke-idx", type=int, default=2, choices=list(range(8)))
    ap.add_argument("--delta", type=float, default=0.035)
    ap.add_argument("--time", type=float, default=1.20)
    ap.add_argument("--budget-frac", type=float, default=0.65)
    ap.add_argument(
        "--allocation",
        choices=["proportional", "priority", "softmax"],
        default="priority",
    )
    ap.add_argument("--coupling-profile", choices=["mild", "sharp"], default="mild")
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    all_res = []

    print("\nHSF SUBSYSTEM EMISSION-BUDGET TEST (SU3 v1)\n")

    for case in args.cases:
        res = analyze_case(
            case,
            args.seed,
            args.poke_idx,
            args.delta,
            args.time,
            args.budget_frac,
            args.allocation,
            args.coupling_profile,
        )
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