#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
fermionic_dimension_test_fast.py
================================

Surgery rewrite of fermionic_dimension_test.py to make it *fast and streaming*.

Key changes vs original:
- Removes full Pauli enumeration (4^N) locality cost/gradient.
- Uses a fast locality proxy: projection onto a *local operator subspace*
  consisting of on-site Paulis + 2-body Paulis on lattice edges:
      leak ≈ 1 - ||P_local(H)||^2 / ||H||^2
- Replaces double-bracket flow with a stochastic 2-qubit conjugation recovery
  loop (Metropolis + annealing), adapted from the v2 streaming suite.

Outputs:
- out/manifest.json
- out/baseline_<label>.json for each lattice
- out/runs/runs.jsonl streaming (one row per lattice)
- out/summary.json checkpoint updates
- out/REPORT.md

Windows example:
  python fermionic_dimension_test_fast.py --out "C:\GitHub\hilbert_substrate\outputs\FERM_DIM_FAST" --seed 42 --steps 1200 --cost-every 5 --orderings 32 --progress

Notes:
- This still uses exact dense matrices. Keep N <= 9 for sanity.
- The "ordering sensitivity" test remains a JW-ordering proxy (fast).
  If you want a real braid/permutation witness, the next step is explicit
  loop transport (Berry phase from two distinct exchange loops).
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


# -------------------------
# IO helpers (streaming)
# -------------------------

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


# -------------------------
# Pauli + tensor utilities
# -------------------------

I2 = np.array([[1, 0],[0, 1]], dtype=np.complex128)
X  = np.array([[0, 1],[1, 0]], dtype=np.complex128)
Y  = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
Z  = np.array([[1, 0],[0, -1]], dtype=np.complex128)
SIG = {"X": X, "Y": Y, "Z": Z}

def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def op_on_qubit(op: np.ndarray, N: int, q: int) -> np.ndarray:
    mats = [I2] * N
    mats[q] = op
    return kron_all(mats)

def two_site_term(opA: np.ndarray, opB: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    mats = [I2] * N
    mats[i] = opA
    mats[j] = opB
    return kron_all(mats)

def embed_two_qubit_gate_adjacent(U2: np.ndarray, N: int, i: int) -> np.ndarray:
    """
    Embed a 4x4 gate U2 onto adjacent qubits (i, i+1) in an N-qubit Hilbert space.
    """
    assert 0 <= i < N-1
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


# -------------------------
# Lattice generators
# -------------------------

def make_1d_lattice(N: int) -> Tuple[List[Tuple[int,int]], str]:
    edges = [(i, (i+1) % N) for i in range(N)]
    return edges, "1D Ring"

def make_2d_lattice(Lx: int, Ly: int) -> Tuple[List[Tuple[int,int]], str]:
    N = Lx * Ly
    edges: List[Tuple[int,int]] = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            j = ((x+1) % Lx) * Ly + y
            k = x * Ly + ((y+1) % Ly)
            edges.append((i, j))
            edges.append((i, k))
    return edges, f"2D Torus ({Lx}x{Ly})"

def make_3d_lattice(Lx: int, Ly: int, Lz: int) -> Tuple[List[Tuple[int,int]], str]:
    N = Lx * Ly * Lz
    edges: List[Tuple[int,int]] = []
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                i = x * Ly * Lz + y * Lz + z
                j = ((x+1) % Lx) * Ly * Lz + y * Lz + z
                k = x * Ly * Lz + ((y+1) % Ly) * Lz + z
                l = x * Ly * Lz + y * Lz + ((z+1) % Lz)
                edges.append((i, j))
                edges.append((i, k))
                edges.append((i, l))
    return edges, f"3D Torus ({Lx}x{Ly}x{Lz})"


# -------------------------
# Random unitaries + scrambling
# -------------------------

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


# -------------------------
# Jordan–Wigner fermions
# -------------------------

def jordan_wigner_operators(N: int, ordering: Optional[List[int]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build JW creation and annihilation operators for N sites.
    ordering: permutation of sites (optional); if None uses natural ordering.
    """
    b_create  = np.array([[0,0],[1,0]], dtype=np.complex128)   # |1><0|
    b_destroy = np.array([[0,1],[0,0]], dtype=np.complex128)   # |0><1|

    if ordering is None:
        ordering = list(range(N))

    inv = [0]*N
    for pos, site in enumerate(ordering):
        inv[site] = pos

    c_create: List[np.ndarray] = []
    c_destroy: List[np.ndarray] = []

    for j in range(N):
        pos = inv[j]
        ops_c: List[np.ndarray] = []
        ops_d: List[np.ndarray] = []
        for m in range(N):
            mpos = inv[m]
            if mpos < pos:
                ops_c.append(Z); ops_d.append(Z)
            elif mpos == pos:
                ops_c.append(b_create); ops_d.append(b_destroy)
            else:
                ops_c.append(I2); ops_d.append(I2)
        c_create.append(kron_all(ops_c))
        c_destroy.append(kron_all(ops_d))
    return c_create, c_destroy

def jw_anticommutator_max(N: int) -> float:
    """
    Max absolute entry error across {c_i,c_j} and {c_i,c_j^†}-δ_ij.
    This is a baseline sanity check; it does not depend on H.
    """
    c_create, c_destroy = jordan_wigner_operators(N)
    d = 2**N
    I_d = np.eye(d, dtype=np.complex128)
    max_abs = 0.0
    for i in range(N):
        for j in range(N):
            A = c_destroy[i] @ c_destroy[j] + c_destroy[j] @ c_destroy[i]
            max_abs = max(max_abs, float(np.max(np.abs(A))))
            B = c_destroy[i] @ c_create[j] + c_create[j] @ c_destroy[i]
            target = (1.0 if i == j else 0.0) * I_d
            max_abs = max(max_abs, float(np.max(np.abs(B - target))))
    return max_abs


# -------------------------
# Fermionic Hamiltonian (free hopping on graph)
# -------------------------

def free_fermion_hamiltonian(edges: List[Tuple[int,int]], N: int, t: float = 1.0) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    H = -t Σ_{<i,j>} (c†_i c_j + h.c.)
    """
    c_create, c_destroy = jordan_wigner_operators(N)
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    for (i, j) in edges:
        H -= t * (c_create[i] @ c_destroy[j] + c_create[j] @ c_destroy[i])
    H = (H + H.conj().T)/2.0
    return H, c_create, c_destroy


# -------------------------
# Fast locality proxy (project onto local operator subspace)
# -------------------------

@dataclass
class LocalBasis:
    ops: List[np.ndarray]
    tags: List[str]

def build_local_basis_from_edges(N: int, edges: List[Tuple[int,int]]) -> LocalBasis:
    """
    Local basis = onsite {X,Y,Z} on each site + 2-body Pauli products on each lattice edge.
    Basis size ~ 3N + 9|E|.
    """
    ops: List[np.ndarray] = []
    tags: List[str] = []

    for i in range(N):
        for lab, op in SIG.items():
            ops.append(op_on_qubit(op, N, i))
            tags.append(f"{lab}_{i}")

    # de-duplicate edges (undirected)
    edge_set = set()
    for (a,b) in edges:
        if a == b: 
            continue
        i, j = (a,b) if a < b else (b,a)
        edge_set.add((i,j))
    edge_list = sorted(edge_set)

    for (i,j) in edge_list:
        for a in ("X","Y","Z"):
            for b in ("X","Y","Z"):
                ops.append(two_site_term(SIG[a], SIG[b], N, i, j))
                tags.append(f"{a}{b}_{i}-{j}")

    return LocalBasis(ops=ops, tags=tags)

def frob2(H: np.ndarray) -> float:
    return float(np.vdot(H, H).real)

def proj_local_norm2(H: np.ndarray, basis: LocalBasis, N: int) -> float:
    """
    Using orthonormality up to scale: sum |Tr(P^† H)|^2 / d
    """
    d = 2**N
    acc = 0.0
    for P in basis.ops:
        t = np.trace(P.conj().T @ H)
        acc += float((t.conjugate() * t).real)
    return acc / float(d)

def locality_proxy(H: np.ndarray, basis: LocalBasis, N: int) -> Tuple[float, float, float]:
    total = frob2(H)
    local = proj_local_norm2(H, basis, N)
    local_frac = float(local / (total + 1e-18))
    leak_frac = float(1.0 - local_frac)
    return leak_frac, local_frac, total


# -------------------------
# Fast recovery flow (Metropolis local conjugations)
# -------------------------

@dataclass
class FlowParams:
    steps: int
    eps: float
    temp0: float
    temp_decay: float

def flow_recover(Hs: np.ndarray, N: int, rng: np.random.Generator, params: FlowParams,
                 basis: LocalBasis, cost_every: int = 1) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Stochastic recovery by conjugating with random adjacent 2-qubit unitaries and
    accepting moves that reduce leak (or occasionally accepting worse moves).
    """
    H = Hs.copy()
    bestH = H.copy()

    leak, local_frac, _ = locality_proxy(H, basis, N)
    cost = leak
    best_cost = cost
    temp = float(params.temp0)

    accepted = 0
    evaluated = 0

    for step in range(params.steps):
        i = int(rng.integers(0, N-1))  # adjacent gate index
        U2 = unitary_from_hermitian(hermitian_rand(4, rng), t=float(params.eps))
        G = embed_two_qubit_gate_adjacent(U2, N, i)
        Hn = G @ H @ G.conj().T
        Hn = (Hn + Hn.conj().T)/2.0

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

        temp *= float(params.temp_decay)

    leak_f, local_f, _ = locality_proxy(H, basis, N)
    diag = {
        "best_leak_frac": float(best_cost),
        "final_leak_frac": float(leak_f),
        "final_local_frac": float(local_f),
        "accepted": float(accepted),
        "evaluated": float(evaluated),
    }
    return bestH, diag


# -------------------------
# Fermion structure measures (kept, but made efficient-ish)
# -------------------------

def measure_fermionic_structure(H: np.ndarray, N: int, edges: List[Tuple[int,int]]) -> Dict[str, object]:
    """
    Project H onto the quadratic sector spanned by c†_i c_j and measure:
    - quadratic_fraction: 1 - ||H - H_quad||/||H||
    - locality_ratio: fraction of hopping amplitude on lattice edges vs off-lattice
    """
    c_create, c_destroy = jordan_wigner_operators(N)

    # hopping matrix t_ij = Tr(H c†_i c_j)/d
    d = 2**N
    tmat = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            op = c_create[i] @ c_destroy[j]
            tmat[i, j] = np.trace(H @ op) / d

    # reconstruct quadratic Hamiltonian
    Hq = np.zeros_like(H)
    for i in range(N):
        for j in range(N):
            if abs(tmat[i, j]) > 0:
                Hq += tmat[i, j] * (c_create[i] @ c_destroy[j])
    Hq = (Hq + Hq.conj().T)/2.0

    nH = np.linalg.norm(H)
    res = H - Hq
    quad_fraction = float(1.0 - (np.linalg.norm(res) / (nH + 1e-12)))

    edge_set = set()
    for (a,b) in edges:
        i, j = (a,b) if a < b else (b,a)
        edge_set.add((i,j))

    on_lattice = 0.0
    off_lattice = 0.0
    for i in range(N):
        for j in range(i+1, N):
            amp = float(abs(tmat[i, j]))
            if (i,j) in edge_set:
                on_lattice += amp
            else:
                off_lattice += amp
    locality_ratio = float(on_lattice / (on_lattice + off_lattice + 1e-12))

    return {
        "quadratic_fraction": quad_fraction,
        "locality_ratio": locality_ratio,
        "hopping_matrix": tmat,  # optional; can be large in JSON if dumped raw
    }

def ordering_sensitivity_test(N: int, n_orderings: int, rng: np.random.Generator) -> Dict[str, object]:
    """
    JW ordering sensitivity proxy:
    Create two-particle states using different JW orderings and measure the spread in
    phase(angle(<psi_01|psi_10>)). This is fast but is still a proxy.
    """
    phases: List[float] = []
    d = 2**N
    vacuum = np.zeros(d, dtype=np.complex128)
    vacuum[0] = 1.0

    for _ in range(n_orderings):
        ordering = rng.permutation(N).tolist()
        c_create, _ = jordan_wigner_operators(N, ordering=ordering)

        psi_01 = c_create[0] @ c_create[1] @ vacuum
        psi_10 = c_create[1] @ c_create[0] @ vacuum
        n1 = np.linalg.norm(psi_01)
        n2 = np.linalg.norm(psi_10)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        psi_01 /= n1
        psi_10 /= n2
        ov = np.vdot(psi_01, psi_10)
        phases.append(float(np.angle(ov)))

    if not phases:
        return {"n": 0, "phase_std": None, "mean_abs_phase_over_pi": None}

    ph = np.array(phases, dtype=float)
    # compare |phase| to pi (fermionic)
    phase_std = float(np.std(np.abs(ph) - math.pi))
    mean_abs_phase_over_pi = float(np.mean(np.abs(ph)) / math.pi)

    return {
        "n": int(len(phases)),
        "phase_std": phase_std,
        "mean_abs_phase_over_pi": mean_abs_phase_over_pi,
        "ordering_independent": bool(phase_std < 0.1),
    }


# -------------------------
# Main experiment
# -------------------------

def checkpoint_summary(outdir: Path, rows: List[dict]) -> None:
    leak_b = [r["metrics"]["leak_before"] for r in rows]
    leak_a = [r["metrics"]["leak_after"] for r in rows]
    summary = {
        "created_utc": now_utc_iso(),
        "runs": len(rows),
        "leak_before": {"median": float(np.median(leak_b)), "mean": float(np.mean(leak_b))} if leak_b else {},
        "leak_after": {"median": float(np.median(leak_a)), "mean": float(np.mean(leak_a))} if leak_a else {},
    }
    write_text(outdir / "summary.json", json.dumps(summary, indent=2))

def run_one_lattice(label: str, edges: List[Tuple[int,int]], N: int,
                    args, rng: np.random.Generator, outdir: Path) -> dict:
    if N > args.max_N:
        return {
            "meta": {"label": label, "N": N, "skipped": True, "reason": f"N>{args.max_N}"},
        }

    # Build H0 (fermion hopping)
    H0, _, _ = free_fermion_hamiltonian(edges, N, t=float(args.t))

    # Local basis depends on lattice edges
    basis = build_local_basis_from_edges(N, edges)

    # Baseline locality of H0
    leak0, loc0, norm0 = locality_proxy(H0, basis, N)
    write_text(outdir / f"baseline_{label.replace(' ','_').replace('/','_')}.json",
               json.dumps({"created_utc": now_utc_iso(),
                           "label": label, "N": N,
                           "baseline": {"leak": leak0, "local_frac": loc0, "norm2": norm0}}, indent=2))

    # Scramble
    if args.scramble == "local":
        Hs = scramble_local(H0, N, rng)
    else:
        Hs = scramble_global(H0, N, rng)

    leak_b, loc_b, _ = locality_proxy(Hs, basis, N)

    # Recover
    fp = FlowParams(steps=int(args.steps), eps=float(args.eps),
                    temp0=float(args.temp0), temp_decay=float(args.temp_decay))
    Hr, diag = flow_recover(Hs, N, rng, fp, basis, cost_every=max(1, int(args.cost_every)))

    leak_a, loc_a, _ = locality_proxy(Hr, basis, N)

    # Fermionic structure measures
    ferm = measure_fermionic_structure(Hr, N, edges)
    # Avoid dumping full hopping matrix unless requested
    if not args.dump_hopping:
        ferm = {k: v for k, v in ferm.items() if k != "hopping_matrix"}

    # Ordering sensitivity proxy
    order_test = ordering_sensitivity_test(N, int(args.orderings), rng)

    row = {
        "meta": {
            "created_utc": now_utc_iso(),
            "label": label,
            "N": int(N),
            "scramble": args.scramble,
            "steps": int(args.steps),
            "eps": float(args.eps),
            "temp0": float(args.temp0),
            "temp_decay": float(args.temp_decay),
            "cost_every": int(args.cost_every),
        },
        "metrics": {
            "leak_before": float(leak_b),
            "local_frac_before": float(loc_b),
            "leak_after": float(leak_a),
            "local_frac_after": float(loc_a),
            **{k: float(v) for k, v in diag.items()},
        },
        "fermion_structure": ferm,
        "ordering_proxy": order_test,
        "success": {
            "recovered_locality": bool((leak_a < leak_b) and (loc_a > loc_b))
        }
    }

    if args.fermion_audit:
        row["fermion_audit"] = {"jw_anticommutator_max": float(jw_anticommutator_max(N))}

    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scramble", choices=["local", "global"], default="global")

    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--temp0", type=float, default=0.02)
    ap.add_argument("--temp-decay", type=float, default=0.9995)
    ap.add_argument("--cost-every", type=int, default=5)

    ap.add_argument("--orderings", type=int, default=32, help="JW ordering trials for ordering sensitivity proxy")
    ap.add_argument("--t", type=float, default=1.0)

    ap.add_argument("--max-N", dest="max_N", type=int, default=9)
    ap.add_argument("--dump-hopping", action="store_true")

    ap.add_argument("--fermion-audit", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out).resolve()
    ensure_dir(outdir)
    ensure_dir(outdir / "runs")

    write_text(outdir / "manifest.json", json.dumps({
        "created_utc": now_utc_iso(),
        "tool": "fermionic_dimension_test_fast.py",
        "args": vars(args),
    }, indent=2))

    rng = np.random.default_rng(int(args.seed))

    # Define lattices (match your original intent)
    lattices: List[Tuple[str, List[Tuple[int,int]], int]] = []
    edges1, label1 = make_1d_lattice(8)
    lattices.append((label1, edges1, 8))

    edges2, label2 = make_2d_lattice(3, 3)
    lattices.append((label2, edges2, 9))

    edges3, label3 = make_3d_lattice(2, 2, 2)
    lattices.append((label3, edges3, 8))

    runs_jsonl = outdir / "runs" / "runs.jsonl"
    if runs_jsonl.exists():
        runs_jsonl.unlink()

    rows: List[dict] = []
    t0 = time.time()

    for idx, (label, edges, N) in enumerate(lattices, start=1):
        if args.progress:
            print(f"[{idx}/{len(lattices)}] lattice={label} N={N} scramble={args.scramble}")
        row = run_one_lattice(label, edges, N, args, rng, outdir)
        rows.append(row)
        append_jsonl(runs_jsonl, row)

        if (len(rows) % max(1, int(args.checkpoint_every))) == 0:
            checkpoint_summary(outdir, rows)

    checkpoint_summary(outdir, rows)
    runtime = float(time.time() - t0)

    # Write a human report
    lines = [
        "# Fermionic Dimension Test (FAST)",
        f"- Created: `{now_utc_iso()}`",
        f"- scramble: `{args.scramble}`  seed: `{args.seed}`",
        f"- runtime_sec: {runtime:.2f}",
        "",
        "## Files",
        "- manifest.json",
        "- runs/runs.jsonl (streaming)",
        "- summary.json",
        "- baseline_*.json",
        "",
        "## Quick read",
        "Look inside runs/runs.jsonl for:",
        "- metrics.leak_before vs metrics.leak_after (locality recovery proxy)",
        "- fermion_structure.quadratic_fraction (quadratic sector retention)",
        "- fermion_structure.locality_ratio (hopping on-lattice vs off-lattice)",
        "- ordering_proxy.phase_std (JW ordering sensitivity proxy)",
    ]
    write_text(outdir / "REPORT.md", "\n".join(lines))

    if args.zip:
        z = zip_folder(outdir)
        print("Wrote ZIP:", z)

    print("DONE. Open:", outdir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
