\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
HSF single-script simulation suite (v2: streaming + faster)
==========================================================

If your previous run "ran for hours and produced no output":
- v1 only wrote output at the END.
- v1 also used a FULL Pauli enumeration (4^N strings). At N=8 that's 65,536 strings,
  and it was being recomputed many times -> can take a very long time.

v2 fixes BOTH:
1) STREAMING OUTPUT:
   - Creates the output folder immediately
   - Writes runs/runs.jsonl incrementally (flushes each run)
   - Writes a checkpoint summary.json every --checkpoint-every runs
2) FASTER LOCALITY METRIC (default):
   - Uses a projection-to-local-subspace proxy that evaluates only a limited local basis
     (on-site + nearest-neighbor 2-body) and estimates leak:
         leak ≈ 1 - ||P_local(H)||^2 / ||H||^2

REQUIREMENTS: python + numpy (pip install numpy)

Windows example:
  python hsf_single_script_sim_suite_v2_streaming_fast.py --out "C:\GitHub\hilbert_substrate\outputs\SUITE_latest" --N 8 --seeds 8 --steps 2000 --scrambles local,global --zip --progress

Speed knob:
  --cost-every 5   (evaluate cost every 5 steps; faster)
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


def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()

def zip_folder(folder: Path) -> Path:
    zip_path = Path(str(folder) + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(folder)
                z.write(full, arcname=str(rel))
    return zip_path


def hermitian_rand(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (a + a.conj().T) / 2.0

def unitary_from_hermitian(h: np.ndarray, t: float) -> np.ndarray:
    w, v = np.linalg.eigh(h)
    ph = np.exp(-1j * t * w)
    return (v * ph) @ v.conj().T

def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.maximum(1e-12, np.abs(d))
    return q * ph

def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def embed_two_qubit_gate(U2: np.ndarray, N: int, i: int) -> np.ndarray:
    I2 = np.eye(2, dtype=complex)
    out = None
    q = 0
    while q < N:
        if q == i:
            out = U2 if out is None else np.kron(out, U2)
            q += 2
        else:
            out = I2 if out is None else np.kron(out, I2)
            q += 1
    return out


I = np.array([[1, 0],[0, 1]], dtype=complex)
X = np.array([[0, 1],[1, 0]], dtype=complex)
Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
Z = np.array([[1, 0],[0, -1]], dtype=complex)
SIG = {"X": X, "Y": Y, "Z": Z}

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

def op_on_qubit(op: np.ndarray, N: int, q: int) -> np.ndarray:
    mats = []
    for i in range(N):
        mats.append(op if i == q else I)
    return kron_all(mats)

def build_xx_hamiltonian(N: int, J: float = 1.0, ring: bool = True) -> np.ndarray:
    H = np.zeros((2**N, 2**N), dtype=complex)
    edges = [(i, i+1) for i in range(N-1)]
    if ring and N > 2:
        edges.append((N-1, 0))
    for (i, j) in edges:
        H += (J/2.0) * (two_site_term(X, X, N, i, j) + two_site_term(Y, Y, N, i, j))
    return H


def scramble_local(H: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    mats = []
    for _ in range(N):
        u = unitary_from_hermitian(hermitian_rand(2, rng), t=1.0)
        mats.append(u)
    U = kron_all(mats)
    return U @ H @ U.conj().T

def scramble_global(H: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    d = 2**N
    U = haar_unitary(d, rng)
    return U @ H @ U.conj().T


@dataclass
class LocalBasis:
    ops: List[np.ndarray]
    tags: List[str]

def build_local_basis(N: int, ring: bool) -> LocalBasis:
    ops: List[np.ndarray] = []
    tags: List[str] = []
    for i in range(N):
        for lab, op in SIG.items():
            ops.append(op_on_qubit(op, N, i))
            tags.append(f"{lab}_{i}")
    edges = [(i, i+1) for i in range(N-1)]
    if ring and N > 2:
        edges.append((N-1, 0))
    pa = ["X","Y","Z"]
    for (i, j) in edges:
        for a in pa:
            for b in pa:
                ops.append(two_site_term(SIG[a], SIG[b], N, i, j))
                tags.append(f"{a}{b}_{i}-{j}")
    return LocalBasis(ops=ops, tags=tags)

def frob2(H: np.ndarray) -> float:
    return float(np.vdot(H, H).real)

def proj_local_norm2(H: np.ndarray, basis: LocalBasis, N: int) -> float:
    d = 2**N
    acc = 0.0
    for P in basis.ops:
        t = np.trace(P.conj().T @ H)
        acc += float((t.conjugate()*t).real)
    return acc / float(d)

def locality_proxy(H: np.ndarray, basis: LocalBasis, N: int) -> Tuple[float, float, float]:
    total = frob2(H)
    local = proj_local_norm2(H, basis, N)
    local_frac = float(local / (total + 1e-18))
    leak_frac = float(1.0 - local_frac)
    return leak_frac, local_frac, total


@dataclass
class FlowParams:
    steps: int
    eps: float
    temp0: float
    temp_decay: float

def flow_recover(Hs: np.ndarray, N: int, rng: np.random.Generator, params: FlowParams, basis: LocalBasis,
                 cost_every: int = 1) -> Tuple[np.ndarray, Dict[str, float]]:
    H = Hs.copy()
    bestH = H.copy()

    leak, local_frac, _ = locality_proxy(H, basis, N)
    cost = leak
    best_cost = cost
    temp = params.temp0

    accepted = 0
    evaluated = 0

    for step in range(params.steps):
        i = int(rng.integers(0, N-1))
        U2 = unitary_from_hermitian(hermitian_rand(4, rng), t=params.eps)
        G = embed_two_qubit_gate(U2, N, i)
        Hn = G @ H @ G.conj().T

        if (step % cost_every) == 0:
            evaluated += 1
            leak_n, _, _ = locality_proxy(Hn, basis, N)
            cn = leak_n
        else:
            cn = cost

        accept = False
        if cn <= cost:
            accept = True
        else:
            if temp > 0:
                p = math.exp(-(cn - cost) / max(1e-12, temp))
                if float(rng.random()) < p:
                    accept = True

        if accept:
            accepted += 1
            H = Hn
            cost = cn
            if cost < best_cost:
                best_cost = cost
                bestH = H.copy()

        temp *= params.temp_decay

    leak_f, local_f, _ = locality_proxy(H, basis, N)
    diag = {
        "best_leak_frac": float(best_cost),
        "final_leak_frac": float(leak_f),
        "final_local_frac": float(local_f),
        "accepted": float(accepted),
        "evaluated": float(evaluated),
    }
    return bestH, diag


def jw_c_ops(N: int) -> List[np.ndarray]:
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


def checkpoint_summary(outdir: Path, rows: List[dict]) -> None:
    leak_before = [r["metrics"]["leak_frac_before"] for r in rows]
    leak_after = [r["metrics"]["leak_frac_after"] for r in rows]
    rec = [1.0 if r["success"]["recovered"] else 0.0 for r in rows]
    summary = {
        "created_utc": now_utc_iso(),
        "runs": len(rows),
        "leak_before": {"median": float(np.median(leak_before)), "mean": float(np.mean(leak_before))} if leak_before else {},
        "leak_after": {"median": float(np.median(leak_after)), "mean": float(np.mean(leak_after))} if leak_after else {},
        "recovery_rate": float(np.mean(rec)) if rec else 0.0,
    }
    write_text(outdir / "summary.json", json.dumps(summary, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--ring", action="store_true")
    ap.add_argument("--open", dest="ring", action="store_false")
    ap.set_defaults(ring=True)

    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--scrambles", default="local,global")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--temp0", type=float, default=0.02)
    ap.add_argument("--temp-decay", type=float, default=0.9995)
    ap.add_argument("--cost-every", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--progress", action="store_true")

    ap.add_argument("--fermion-audit", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out).resolve()
    ensure_dir(outdir)
    ensure_dir(outdir / "runs")

    # Immediate outputs
    write_text(outdir / "manifest.json", json.dumps({
        "created_utc": now_utc_iso(),
        "tool": "hsf_single_script_sim_suite_v2_streaming_fast.py",
        "args": vars(args),
    }, indent=2))

    N = args.N
    ring = bool(args.ring)
    scrambles = [s.strip().lower() for s in args.scrambles.split(",") if s.strip()]
    scrambles = [s for s in scrambles if s in ("local", "global")]
    if not scrambles:
        scrambles = ["local", "global"]

    H0 = build_xx_hamiltonian(N, J=args.J, ring=ring)
    basis = build_local_basis(N, ring=ring)

    leak0, local0, total0 = locality_proxy(H0, basis, N)
    write_text(outdir / "baseline.json", json.dumps({
        "created_utc": now_utc_iso(),
        "baseline": {"leak_frac": leak0, "local_frac": local0, "total_norm2": total0}
    }, indent=2))

    runs_jsonl = outdir / "runs" / "runs.jsonl"
    if runs_jsonl.exists():
        runs_jsonl.unlink()

    rows: List[dict] = []
    t0 = time.time()
    jw0 = jw_anticommutator_max(N) if args.fermion_audit else None

    total_runs = args.seeds * len(scrambles)
    run_idx = 0

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        rng = np.random.default_rng(seed)
        for scramble in scrambles:
            run_idx += 1
            if args.progress:
                print(f"[{run_idx}/{total_runs}] seed={seed} scramble={scramble}")

            Hs = scramble_local(H0, N, rng) if scramble == "local" else scramble_global(H0, N, rng)
            leak_b, local_b, _ = locality_proxy(Hs, basis, N)

            fp = FlowParams(steps=args.steps, eps=args.eps, temp0=args.temp0, temp_decay=args.temp_decay)
            Hr, diag = flow_recover(Hs, N, rng, fp, basis, cost_every=max(1, int(args.cost_every)))

            leak_a, local_a, _ = locality_proxy(Hr, basis, N)

            row = {
                "meta": {"created_utc": now_utc_iso(), "N": N, "J": float(args.J), "ring": ring,
                         "seed": int(seed), "scramble": scramble, "flow_steps": int(args.steps),
                         "eps": float(args.eps), "temp0": float(args.temp0), "temp_decay": float(args.temp_decay),
                         "cost_every": int(args.cost_every)},
                "metrics": {"leak_frac_before": float(leak_b), "local_frac_before": float(local_b),
                            "leak_frac_after": float(leak_a), "local_frac_after": float(local_a),
                            **{k: float(v) for k, v in diag.items()}},
                "success": {"recovered": bool((leak_a < leak_b) and (local_a > local_b))}
            }

            if args.fermion_audit:
                row["fermion_audit"] = {"jw_max_baseline": float(jw0), "jw_max": float(jw_anticommutator_max(N))}

            rows.append(row)
            append_jsonl(runs_jsonl, row)

            if (len(rows) % max(1, int(args.checkpoint_every))) == 0:
                checkpoint_summary(outdir, rows)

    checkpoint_summary(outdir, rows)
    runtime = float(time.time() - t0)

    write_text(outdir / "REPORT.md", "\n".join([
        "# HSF Simulation Suite Report (v2 streaming+fast)",
        f"- Created: `{now_utc_iso()}`",
        f"- N={N} boundary={'ring' if ring else 'open'}",
        f"- Runs: {len(rows)}  runtime_sec: {runtime:.2f}",
        "",
        "## Files",
        "- baseline.json",
        "- runs/runs.jsonl",
        "- summary.json",
        "- manifest.json",
    ]))

    if args.zip:
        z = zip_folder(outdir)
        print("Wrote ZIP:", z)

    print("DONE. Open:", outdir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
