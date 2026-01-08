\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
HSF single-script simulation suite (standalone)
==============================================

This is ONE self-contained Python script (no repo runner dependency) that:

1) Builds an N-qubit XX spin-chain Hamiltonian (nearest-neighbor, open or ring)
2) Applies either:
   - LOCAL scramble: random single-qubit unitaries (product)
   - GLOBAL scramble: random Haar unitary on 2^N (full scramble)
3) Attempts "FLOW" recovery using ONLY local 2-qubit neighbor gates via stochastic descent
4) Measures a locality diagnostic before/after:
   - V1 = Pauli-mass at range==1
   - V2 = Pauli-mass at range==2
   - reports V2/V1 (ideal XX => ~0)
5) Optionally runs a fermion audit (toy but concrete):
   - JW anticommutator max error for c_j in the current basis
   - Free-fermion spectrum consistency error for XX ring/open chain
   - Pauli-pressure proxy: mass(weight=2)/mass(weight=1) in Hamiltonian Pauli expansion

Outputs:
- ONE folder containing:
  runs.jsonl
  summary.json
  REPORT.md
  manifest.json
  optional ZIP

Install requirements: python + numpy (pip install numpy)
Optional: scipy is NOT required.

Windows run example:
  python hsf_single_script_sim_suite.py --out "outputs/SUITE_latest" --N 8 --J 1.0 --ring --seeds 16 --steps 2500 --eps 0.06 --scrambles local,global --fermion-audit --zip

Notes:
- This is designed to be *robust* and *reproducible* rather than fastest.
- For N=8 (d=256) it’s feasible on a normal laptop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


# ----------------------------
# Basic linear algebra helpers
# ----------------------------

def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def hermitian_rand(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = (a + a.conj().T) / 2.0
    return h

def unitary_from_hermitian(h: np.ndarray, t: float) -> np.ndarray:
    # U = exp(-i t H) via eigendecomposition (stable for small dims)
    w, v = np.linalg.eigh(h)
    ph = np.exp(-1j * t * w)
    return (v * ph) @ v.conj().T

def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    # Fix phases
    d = np.diag(r)
    ph = d / np.abs(d)
    q = q * ph
    return q

def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def embed_two_qubit_gate(U2: np.ndarray, N: int, i: int) -> np.ndarray:
    """
    Embed a 4x4 gate acting on qubits (i, i+1) into 2^N space (0-indexed).
    Uses big-endian convention: qubit 0 is the leftmost in kron order.
    """
    I2 = np.eye(2, dtype=complex)
    mats = []
    for q in range(N):
        if q == i:
            mats.append(None)  # placeholder for 2-qubit block
        elif q == i + 1:
            continue
        else:
            mats.append(I2)
    # Build kron with insertion
    out = None
    idx = 0
    for q in range(N):
        if q == i:
            block = U2
            if out is None:
                out = block
            else:
                out = np.kron(out, block)
            idx += 1
        elif q == i + 1:
            continue
        else:
            if out is None:
                out = I2
            else:
                out = np.kron(out, I2)
            idx += 1
    return out


# ----------------------------
# Pauli basis utilities
# ----------------------------

I = np.array([[1, 0],[0, 1]], dtype=complex)
X = np.array([[0, 1],[1, 0]], dtype=complex)
Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
Z = np.array([[1, 0],[0, -1]], dtype=complex)
PAULIS = [I, X, Y, Z]
PAULI_LABELS = ["I", "X", "Y", "Z"]

def pauli_mass_by_range_and_weight(H: np.ndarray, N: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Expand H in N-qubit Pauli basis, return:
      - mass_by_range[r] = sum |c_P|^2 for Pauli strings with range==r
      - mass_by_weight[w] = sum |c_P|^2 for strings with Pauli weight==w (count non-identity)
    Uses orthonormal basis: P / sqrt(2^N) with coefficient c = Tr(P H)/2^N
    Complexity: O(4^N * d^2) if done naively; here we compute via kron products and traces, which is still
    heavy but okay for N<=8 in small counts.
    """
    d = 2**N
    norm = float(d)
    mass_range: Dict[int, float] = {}
    mass_w: Dict[int, float] = {}

    # Precompute single-qubit paulis
    single = PAULIS

    # Iterate all Pauli strings in base-4
    for idx in range(4**N):
        tmp = idx
        ops = []
        support = []
        w = 0
        for q in range(N-1, -1, -1):
            p = tmp & 3
            tmp >>= 2
            ops.append(single[p])
            if p != 0:
                support.append(q)
                w += 1
        ops = ops[::-1]  # restore q=0 leftmost
        P = kron_all(ops)
        c = np.trace(P.conj().T @ H) / norm
        m = float((c.conjugate() * c).real)
        if w == 0:
            r = 0
        else:
            r = int(max(support) - min(support))
        mass_range[r] = mass_range.get(r, 0.0) + m
        mass_w[w] = mass_w.get(w, 0.0) + m

    return mass_range, mass_w

def locality_v2_over_v1(H: np.ndarray, N: int) -> Tuple[float, Dict[int, float], Dict[int, float]]:
    mass_range, mass_w = pauli_mass_by_range_and_weight(H, N)
    V1 = mass_range.get(1, 0.0)
    V2 = mass_range.get(2, 0.0)
    ratio = float(V2 / (V1 + 1e-18))
    return ratio, mass_range, mass_w


# ----------------------------
# Hamiltonian: XX chain
# ----------------------------

def op_on_qubit(op: np.ndarray, N: int, q: int) -> np.ndarray:
    mats = []
    for i in range(N):
        mats.append(op if i == q else I)
    return kron_all(mats)

def two_site_term(opA: np.ndarray, opB: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    mats = []
    for q in range(N):
        if q == i:
            mats.append(opA)
        elif q == j:
            mats.append(opB)
        else:
            mats.append(I)
    return kron_all(mats)

def build_xx_hamiltonian(N: int, J: float = 1.0, ring: bool = True) -> np.ndarray:
    H = np.zeros((2**N, 2**N), dtype=complex)
    edges = [(i, i+1) for i in range(N-1)]
    if ring and N > 2:
        edges.append((N-1, 0))
    for (i, j) in edges:
        H += (J/2.0) * (two_site_term(X, X, N, i, j) + two_site_term(Y, Y, N, i, j))
    return H


# ----------------------------
# Scrambles
# ----------------------------

def scramble_local(H: np.ndarray, N: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # product of random single-qubit unitaries
    mats = []
    for _ in range(N):
        h = hermitian_rand(2, rng)
        u = unitary_from_hermitian(h, t=1.0)  # random-ish
        mats.append(u)
    U = kron_all(mats)
    return U @ H @ U.conj().T, U

def scramble_global(H: np.ndarray, N: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    d = 2**N
    U = haar_unitary(d, rng)
    return U @ H @ U.conj().T, U


# ----------------------------
# FLOW recovery (stochastic local descent)
# ----------------------------

@dataclass
class FlowParams:
    steps: int
    eps: float           # gate strength
    temp0: float         # acceptance temp
    temp_decay: float    # multiply each step
    ring: bool

def flow_recover(Hs: np.ndarray, N: int, rng: np.random.Generator, params: FlowParams) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Attempt to reduce long-range Pauli mass using only local 2-qubit gates on neighbors.
    Maintains current Hamiltonian in current basis (conjugated by recovery U).
    """
    H = Hs.copy()
    bestH = H.copy()
    best_cost = None

    def cost_fn(Hm: np.ndarray) -> float:
        ratio, mass_range, _ = locality_v2_over_v1(Hm, N)
        # penalize all ranges >=2, normalized by V1
        V1 = mass_range.get(1, 0.0)
        leak = sum(v for r, v in mass_range.items() if r >= 2)
        return float(leak / (V1 + 1e-18))

    cost = cost_fn(H)
    best_cost = cost
    temp = params.temp0

    for step in range(params.steps):
        # pick neighbor pair
        i = int(rng.integers(0, N-1))
        # random small 2-qubit gate
        h2 = hermitian_rand(4, rng)
        U2 = unitary_from_hermitian(h2, t=params.eps)
        G = embed_two_qubit_gate(U2, N, i)

        Hn = G @ H @ G.conj().T
        cn = cost_fn(Hn)

        accept = False
        if cn <= cost:
            accept = True
        else:
            # Metropolis
            if temp > 0:
                p = math.exp(-(cn - cost) / max(1e-12, temp))
                if float(rng.random()) < p:
                    accept = True

        if accept:
            H = Hn
            cost = cn
            if cost < best_cost:
                best_cost = cost
                bestH = H.copy()

        temp *= params.temp_decay

    diagnostics = {"best_cost": float(best_cost), "final_cost": float(cost)}
    return bestH, diagnostics


# ----------------------------
# Fermion audit (toy but concrete)
# ----------------------------

def jw_c_ops(N: int) -> List[np.ndarray]:
    """
    Jordan–Wigner fermion annihilation operators c_j for qubit chain.
    c_j = (prod_{k<j} Z_k) * (X_j + i Y_j)/2
    """
    ops = []
    for j in range(N):
        zstring = np.eye(2**N, dtype=complex)
        for k in range(j):
            zstring = zstring @ op_on_qubit(Z, N, k)
        sigma_minus = (op_on_qubit(X, N, j) + 1j * op_on_qubit(Y, N, j)) / 2.0
        ops.append(zstring @ sigma_minus)
    return ops

def jw_anticommutator_max(N: int) -> float:
    cs = jw_c_ops(N)
    d = 2**N
    I_d = np.eye(d, dtype=complex)
    max_abs = 0.0
    for i in range(N):
        for j in range(N):
            A = cs[i] @ cs[j] + cs[j] @ cs[i]
            max_abs = max(max_abs, float(np.max(np.abs(A))))
            B = cs[i] @ cs[j].conj().T + cs[j].conj().T @ cs[i]
            target = (1.0 if i == j else 0.0) * I_d
            max_abs = max(max_abs, float(np.max(np.abs(B - target))))
    return max_abs

def free_fermion_spectrum_error_xx(N: int, J: float, ring: bool, H: np.ndarray) -> float:
    """
    Compare many-body spectrum of H to free-fermion predicted spectrum for XX.
    For a clean XX chain, many-body eigenvalues are sums of single-particle energies.
    We compute single-particle energies from adjacency (tight-binding).
    Then build all 2^N many-body energies by occupancy and compare sorted lists.
    Error = mean absolute difference / (std + eps).
    """
    # Many-body eigvals
    evals = np.linalg.eigvalsh(H).real
    evals_sorted = np.sort(evals)

    # Single-particle energies for XX chain (tight-binding):
    # open: eps_m = 2J cos(m*pi/(N+1)), m=1..N
    # ring: eps_k = 2J cos(2*pi*k/N), k=0..N-1
    if ring:
        ks = np.arange(N)
        eps = 2.0 * J * np.cos(2.0*np.pi*ks/N)
    else:
        ms = np.arange(1, N+1)
        eps = 2.0 * J * np.cos(ms*np.pi/(N+1))

    # Build many-body energies from eps with occupation n_k in {0,1}
    mb = np.zeros(2**N, dtype=float)
    for state in range(2**N):
        occ = [(state >> b) & 1 for b in range(N)]
        mb[state] = float(np.sum(eps * np.array(occ)))
    mb_sorted = np.sort(mb)

    # Align (overall energy shift can exist due to conventions; remove best-fit shift)
    shift = float(np.mean(evals_sorted - mb_sorted))
    mb_sorted2 = mb_sorted + shift

    err = float(np.mean(np.abs(evals_sorted - mb_sorted2)))
    denom = float(np.std(evals_sorted) + 1e-12)
    return err / denom

def pauli_pressure_proxy(H: np.ndarray, N: int) -> float:
    _, mass_w = pauli_mass_by_range_and_weight(H, N)
    w1 = mass_w.get(1, 0.0)
    w2 = mass_w.get(2, 0.0)
    return float(w2 / (w1 + 1e-18))


# ----------------------------
# Suite
# ----------------------------

def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def zip_folder(folder: Path) -> Path:
    zip_path = Path(str(folder) + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(folder)
                z.write(full, arcname=str(rel))
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output folder (created if missing).")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--ring", action="store_true", help="Use ring boundary (default).")
    ap.add_argument("--open", dest="ring", action="store_false", help="Use open boundary.")
    ap.set_defaults(ring=True)

    ap.add_argument("--seeds", type=int, default=8, help="Number of random seeds.")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--scrambles", default="local,global", help="Comma list: local,global")
    ap.add_argument("--steps", type=int, default=2500, help="FLOW steps per run.")
    ap.add_argument("--eps", type=float, default=0.06, help="2-qubit gate strength.")
    ap.add_argument("--temp0", type=float, default=0.02)
    ap.add_argument("--temp-decay", type=float, default=0.9995)

    ap.add_argument("--fermion-audit", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out).resolve()
    ensure_dir(outdir)
    ensure_dir(outdir / "runs")

    N = args.N
    J = args.J
    ring = bool(args.ring)
    scrambles = [s.strip().lower() for s in args.scrambles.split(",") if s.strip()]
    scrambles = [s for s in scrambles if s in ("local", "global")]
    if not scrambles:
        scrambles = ["local", "global"]

    # Build base Hamiltonian and baseline locality
    H0 = build_xx_hamiltonian(N, J=J, ring=ring)
    base_ratio, base_mass_range, base_mass_w = locality_v2_over_v1(H0, N)

    suite_rows: List[dict] = []
    t0 = time.time()

    # Shared fermion baseline (should be tiny for clean JW algebra)
    jw_max_baseline = jw_anticommutator_max(N) if args.fermion_audit else None

    for sidx in range(args.seed_start, args.seed_start + args.seeds):
        rng = np.random.default_rng(sidx)

        for scramble in scrambles:
            # Scramble
            if scramble == "local":
                Hs, U = scramble_local(H0, N, rng)
            else:
                Hs, U = scramble_global(H0, N, rng)

            ratio_before, mass_range_before, mass_w_before = locality_v2_over_v1(Hs, N)

            # Recover via FLOW
            fp = FlowParams(steps=args.steps, eps=args.eps, temp0=args.temp0, temp_decay=args.temp_decay, ring=ring)
            Hr, flow_diag = flow_recover(Hs, N, rng, fp)
            ratio_after, mass_range_after, mass_w_after = locality_v2_over_v1(Hr, N)

            row: Dict[str, object] = {
                "meta": {
                    "created_utc": now_utc_iso(),
                    "N": N,
                    "J": J,
                    "ring": ring,
                    "seed": int(sidx),
                    "scramble": scramble,
                    "flow_steps": int(args.steps),
                    "eps": float(args.eps),
                    "temp0": float(args.temp0),
                    "temp_decay": float(args.temp_decay),
                },
                "metrics": {
                    "V2_over_V1_before": float(ratio_before),
                    "V2_over_V1_after": float(ratio_after),
                    "flow_best_cost": float(flow_diag["best_cost"]),
                    "flow_final_cost": float(flow_diag["final_cost"]),
                    "mass_range_before": {str(k): float(v) for k, v in sorted(mass_range_before.items())},
                    "mass_range_after": {str(k): float(v) for k, v in sorted(mass_range_after.items())},
                    "pauli_weight_mass_before": {str(k): float(v) for k, v in sorted(mass_w_before.items())},
                    "pauli_weight_mass_after": {str(k): float(v) for k, v in sorted(mass_w_after.items())},
                },
                "success": {
                    # success heuristic: drive ratio down by at least 20% and keep it < 0.2 (tune if you want)
                    "recovered": bool((ratio_after < 0.2) and (ratio_after < 0.8 * ratio_before)),
                }
            }

            if args.fermion_audit:
                row["fermion_audit"] = {
                    "jw_max_baseline": float(jw_max_baseline),
                    "jw_max_current_basis": float(jw_anticommutator_max(N)),  # algebra itself is basis-independent here
                    "free_fermion_spectrum_error": float(free_fermion_spectrum_error_xx(N, J, ring, Hr)),
                    "pauli_pressure_proxy": float(pauli_pressure_proxy(Hr, N)),
                }

            suite_rows.append(row)

    # Write runs.jsonl
    jsonl_path = outdir / "runs" / "runs.jsonl"
    write_jsonl(jsonl_path, suite_rows)

    # Summarize
    def collect(key: str) -> List[float]:
        out = []
        for r in suite_rows:
            v = r
            for part in key.split("."):
                v = v.get(part, None) if isinstance(v, dict) else None
            if v is not None:
                out.append(float(v))
        return out

    before = collect("metrics.V2_over_V1_before")
    after = collect("metrics.V2_over_V1_after")
    rec = [1.0 if r["success"]["recovered"] else 0.0 for r in suite_rows]  # type: ignore

    summary = {
        "created_utc": now_utc_iso(),
        "N": N,
        "J": J,
        "ring": ring,
        "scrambles": scrambles,
        "runs": len(suite_rows),
        "baseline": {
            "V2_over_V1": float(base_ratio),
        },
        "V2_over_V1_before": {
            "median": float(np.median(before)),
            "mean": float(np.mean(before)),
            "min": float(np.min(before)),
            "max": float(np.max(before)),
        },
        "V2_over_V1_after": {
            "median": float(np.median(after)),
            "mean": float(np.mean(after)),
            "min": float(np.min(after)),
            "max": float(np.max(after)),
        },
        "recovery_rate": float(np.mean(rec)),
        "runtime_sec": float(time.time() - t0),
        "files": {
            "runs_jsonl": str(jsonl_path),
        }
    }

    write_text(outdir / "summary.json", json.dumps(summary, indent=2))

    # Report
    lines = []
    lines.append("# HSF Simulation Suite Report")
    lines.append(f"- Created: `{summary['created_utc']}`")
    lines.append(f"- N={N}  J={J}  boundary={'ring' if ring else 'open'}")
    lines.append(f"- Runs: {summary['runs']}")
    lines.append("")
    lines.append("## Baseline")
    lines.append(f"- V2/V1 (XX unscrumbled): {summary['baseline']['V2_over_V1']:.6g}")
    lines.append("")
    lines.append("## Aggregate locality diagnostic (V2/V1)")
    lines.append(f"- Before: median={summary['V2_over_V1_before']['median']:.6g}  mean={summary['V2_over_V1_before']['mean']:.6g}")
    lines.append(f"- After:  median={summary['V2_over_V1_after']['median']:.6g}  mean={summary['V2_over_V1_after']['mean']:.6g}")
    lines.append(f"- Recovery rate (heuristic): {summary['recovery_rate']:.3f}")
    lines.append("")
    lines.append("## Interpretation guidance")
    lines.append("- LOCAL scramble should remain in (or near) the accessibility basin: recovery should usually improve V2/V1.")
    lines.append("- GLOBAL scramble typically leaves the basin: recovery often fails to reduce V2/V1 meaningfully.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- runs: `{(outdir / 'runs' / 'runs.jsonl').as_posix()}`")
    lines.append(f"- summary: `{(outdir / 'summary.json').as_posix()}`")
    write_text(outdir / "REPORT.md", "\n".join(lines))

    # Manifest
    manifest = {
        "created_utc": now_utc_iso(),
        "tool": "hsf_single_script_sim_suite.py",
        "args": vars(args),
        "outputs": ["runs/runs.jsonl", "summary.json", "REPORT.md", "manifest.json"],
    }
    write_text(outdir / "manifest.json", json.dumps(manifest, indent=2))

    if args.zip:
        z = zip_folder(outdir)
        print("Wrote ZIP:", z)

    print("DONE. Open:", outdir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
