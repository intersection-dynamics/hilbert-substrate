#!/usr/bin/env python3
"""
Geometry Robustness Sweep (Accessibility-Respecting Protocol) — seed-parallel

Adds:
  --jobs K      Run seeds in parallel across K processes (Windows-safe).
  --blas-threads T   Force BLAS threading inside each worker (default 1).

Protocol:
  For each seed:
    - Build local XX Hamiltonian on SOURCE geometry
    - Apply LOCAL scramble on SOURCE edges for depth D (reachable)
    - For each TARGET geometry, run local recovery constrained to target edges
    - Measure k-local "leak" (MC estimate) and leak reduction
  Aggregate across seeds.

Outputs:
  geometry_robustness_curves.csv
  geometry_robustness_summary.json
  geometry_robustness_details.json

Note:
  Dense operator sims -> keep N small (<=10 recommended).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Pauli utilities
# -----------------------------

PAULI_SINGLE = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}
PAULI_LABELS = np.array(["I", "X", "Y", "Z"], dtype="<U1")


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def sample_pauli_strings(N: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(low=0, high=4, size=(m, N), dtype=np.int8)
    weights = np.sum(labels != 0, axis=1).astype(np.int32)
    return labels, weights


def pauli_labels_to_matrix(labels_row: np.ndarray) -> np.ndarray:
    ops = [PAULI_SINGLE[PAULI_LABELS[int(v)]] for v in labels_row]
    return kron_n(ops)


def estimate_klocal_leak(H: np.ndarray, pauli_samples: Tuple[np.ndarray, np.ndarray], k_max: int) -> float:
    labels, weights = pauli_samples
    N = labels.shape[1]
    dim = 2 ** N

    num = 0.0
    den = 0.0
    for row, k in zip(labels, weights):
        P = pauli_labels_to_matrix(row)
        a = np.trace(P @ H) / dim
        w = (np.abs(a) ** 2).real
        den += w
        if k > k_max:
            num += w

    if den <= 0:
        return 1.0
    return float(num / den)


# -----------------------------
# Random unitaries
# -----------------------------

def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex128)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.where(np.abs(d) > 0, np.abs(d), 1.0)
    q = q * np.conj(ph)
    return q


def random_two_qubit_unitary(rng: np.random.Generator) -> np.ndarray:
    return random_unitary(4, rng)


# -----------------------------
# Graph builders
# -----------------------------

def edges_1d_ring(N: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % N) for i in range(N)]


def edges_1d_chain(N: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(N - 1)]


def edges_2d_lattice(N: int) -> Optional[List[Tuple[int, int]]]:
    L = int(round(math.sqrt(N)))
    if L * L != N:
        return None
    edges = []
    def idx(r, c): return r * L + c
    for r in range(L):
        for c in range(L):
            if c + 1 < L:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < L:
                edges.append((idx(r, c), idx(r + 1, c)))
    return edges


def edges_3d_lattice(N: int) -> Optional[List[Tuple[int, int]]]:
    L = int(round(N ** (1 / 3)))
    if L * L * L != N:
        return None
    edges = []
    def idx(x, y, z): return (x * L + y) * L + z
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if z + 1 < L:
                    edges.append((idx(x, y, z), idx(x, y, z + 1)))
                if y + 1 < L:
                    edges.append((idx(x, y, z), idx(x, y + 1, z)))
                if x + 1 < L:
                    edges.append((idx(x, y, z), idx(x + 1, y, z)))
    return edges


def make_random_regular_edges(N: int, d: int, rng: np.random.Generator, max_tries: int = 50000) -> Optional[List[Tuple[int, int]]]:
    if d >= N or (N * d) % 2 != 0:
        return None
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(N, dtype=np.int32), d)
        rng.shuffle(stubs)
        edges = []
        ok = True
        for i in range(0, len(stubs), 2):
            a = int(stubs[i]); b = int(stubs[i + 1])
            if a == b:
                ok = False
                break
            e = (a, b) if a < b else (b, a)
            edges.append(e)
        if not ok:
            continue
        if len(set(edges)) != len(edges):
            continue
        return edges
    return None


def build_edges(name: str, N: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    name = name.strip()
    if name == "1d_ring":
        return edges_1d_ring(N)
    if name == "1d_chain":
        return edges_1d_chain(N)
    if name == "2d":
        e = edges_2d_lattice(N)
        if e is None:
            raise ValueError(f"2d requires N=L^2; got N={N}")
        return e
    if name == "3d":
        e = edges_3d_lattice(N)
        if e is None:
            raise ValueError(f"3d requires N=L^3; got N={N}")
        return e
    if name == "rr4":
        e = make_random_regular_edges(N, 4, rng)
        if e is None:
            raise ValueError(f"Failed to generate rr4 for N={N}")
        return e
    if name == "rr6":
        e = make_random_regular_edges(N, 6, rng)
        if e is None:
            raise ValueError(f"Failed to generate rr6 for N={N}")
        return e
    raise ValueError(f"Unknown topology: {name}")


# -----------------------------
# Hamiltonian construction (XX)
# -----------------------------

def embed_two_qubit_op(N: int, i: int, j: int, op2: np.ndarray) -> np.ndarray:
    if i == j:
        raise ValueError("i == j")
    if i > j:
        i, j = j, i

    paulis = ["I", "X", "Y", "Z"]
    basis = []
    for a in paulis:
        for b in paulis:
            basis.append(np.kron(PAULI_SINGLE[a], PAULI_SINGLE[b]))
    B = np.stack([b.reshape(-1) for b in basis], axis=1)  # 16x16
    coeffs = np.linalg.solve(B, op2.reshape(-1))

    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)

    idx = 0
    for a in paulis:
        for b in paulis:
            c = coeffs[idx]; idx += 1
            if np.abs(c) < 1e-14:
                continue
            ops_full = []
            for q in range(N):
                if q == i:
                    ops_full.append(PAULI_SINGLE[a])
                elif q == j:
                    ops_full.append(PAULI_SINGLE[b])
                else:
                    ops_full.append(PAULI_SINGLE["I"])
            H += c * kron_n(ops_full)

    return H


def build_xx_hamiltonian(N: int, edges: List[Tuple[int, int]], J: float = 1.0) -> np.ndarray:
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)
    XX = np.kron(PAULI_SINGLE["X"], PAULI_SINGLE["X"])
    YY = np.kron(PAULI_SINGLE["Y"], PAULI_SINGLE["Y"])
    op2 = 0.5 * (XX + YY)
    for (i, j) in edges:
        H += J * embed_two_qubit_op(N, i, j, op2)
    H = 0.5 * (H + H.conj().T)
    return H


def apply_two_qubit_conjugation(H: np.ndarray, N: int, i: int, j: int, U2: np.ndarray) -> np.ndarray:
    if i == j:
        raise ValueError("i == j")
    if i > j:
        i, j = j, i

    dim = 2 ** N
    shp = (2,) * N
    Ht = H.reshape(shp + shp)

    qubits = list(range(N))
    rest = [q for q in qubits if q not in (i, j)]

    ket_order = [i, j] + rest
    bra_order = [i + N, j + N] + [q + N for q in rest]
    perm = ket_order + bra_order

    Hperm = np.transpose(Ht, axes=perm)
    Hperm = Hperm.reshape(4, 2 ** (N - 2), 4, 2 ** (N - 2))

    U = U2
    Udag = U2.conj().T
    Hnew = np.einsum("am,mxny,bn->axby", U, Hperm, Udag, optimize=True)

    Hnew = Hnew.reshape((2, 2) + (2,) * (N - 2) + (2, 2) + (2,) * (N - 2))
    inv = np.argsort(np.array(perm))
    Hback = np.transpose(Hnew, axes=inv).reshape(dim, dim)
    Hback = 0.5 * (Hback + Hback.conj().T)
    return Hback


def local_scramble_on_edges(H: np.ndarray, N: int, edges: List[Tuple[int, int]], rng: np.random.Generator, steps: int) -> np.ndarray:
    Hs = H.copy()
    for _ in range(steps):
        (i, j) = edges[int(rng.integers(0, len(edges)))]
        U2 = random_two_qubit_unitary(rng)
        Hs = apply_two_qubit_conjugation(Hs, N, i, j, U2)
    return Hs


# -----------------------------
# Recovery
# -----------------------------

@dataclass
class RecoveryResult:
    leak_initial: float
    leak_final: float
    leak_reduction: float
    accepted_moves: int
    steps: int


def annealed_recovery(
    H_start: np.ndarray,
    N: int,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
    steps: int,
    temp0: float,
    temp_decay: float,
    pauli_samples: Tuple[np.ndarray, np.ndarray],
    k_max: int,
) -> RecoveryResult:
    H = H_start.copy()
    leak0 = estimate_klocal_leak(H, pauli_samples, k_max=k_max)
    best = leak0
    accepted = 0
    T = temp0

    for _ in range(steps):
        (i, j) = edges[int(rng.integers(0, len(edges)))]
        U2 = random_two_qubit_unitary(rng)
        Hcand = apply_two_qubit_conjugation(H, N, i, j, U2)
        leak = estimate_klocal_leak(Hcand, pauli_samples, k_max=k_max)

        d = leak - best
        if d <= 0:
            accept = True
        else:
            accept = (rng.random() < math.exp(-d / max(T, 1e-12)))

        if accept:
            H = Hcand
            if leak < best:
                best = leak
            accepted += 1

        T *= temp_decay

    leakF = best
    return RecoveryResult(
        leak_initial=float(leak0),
        leak_final=float(leakF),
        leak_reduction=float(leak0 - leakF),
        accepted_moves=int(accepted),
        steps=int(steps),
    )


# -----------------------------
# Helpers
# -----------------------------

def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            if b < a:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(token))
    return out


def compute_D_star(depths: List[int], values: List[float], min_mean_improvement: float) -> Optional[int]:
    D_star = None
    for D, v in zip(depths, values):
        if v >= min_mean_improvement:
            D_star = D
    return D_star


# -----------------------------
# Parallel worker
# -----------------------------

def _set_blas_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def run_one_seed(args_dict: dict, seed: int) -> Tuple[int, List[dict]]:
    """
    Compute all (depth, target) runs for a single seed.
    Returns (seed, details_rows).
    """
    _set_blas_threads(int(args_dict["blas_threads"]))

    N = int(args_dict["N"])
    source = args_dict["source_topology"]
    targets = list(args_dict["targets"])
    depths = list(args_dict["depths"])
    recovery_steps = int(args_dict["recovery_steps"])
    temp0 = float(args_dict["temp0"])
    temp_decay = float(args_dict["temp_decay"])
    mc_samples = int(args_dict["mc_samples"])
    kmax = int(args_dict["kmax"])

    rng = np.random.default_rng(seed)

    src_edges = build_edges(source, N, rng)
    H0 = build_xx_hamiltonian(N, src_edges, J=1.0)

    pauli_samples = sample_pauli_strings(N, mc_samples, rng)

    # Important: build target edges per seed for rr graphs deterministically
    tgt_edges_map = {t: build_edges(t, N, rng) for t in targets}

    details: List[dict] = []
    for D in depths:
        Hscr = local_scramble_on_edges(H0, N, src_edges, rng, steps=int(D))

        for tname in targets:
            rr = annealed_recovery(
                H_start=Hscr,
                N=N,
                edges=tgt_edges_map[tname],
                rng=rng,
                steps=recovery_steps,
                temp0=temp0,
                temp_decay=temp_decay,
                pauli_samples=pauli_samples,
                k_max=kmax,
            )
            details.append({
                "seed": int(seed),
                "depth": int(D),
                "source": source,
                "target": tname,
                "kmax": int(kmax),
                "mc_samples": int(mc_samples),
                "leak_initial": float(rr.leak_initial),
                "leak_final": float(rr.leak_final),
                "leak_reduction": float(rr.leak_reduction),
                "accepted_moves": int(rr.accepted_moves),
                "recovery_steps": int(rr.steps),
            })
    return seed, details


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Geometry robustness sweep (accessibility-respecting), seed-parallel.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--N", type=int, default=8, help="Number of qubits (dense matrices).")
    ap.add_argument("--seeds", type=int, default=8, help="Number of RNG seeds to run.")
    ap.add_argument("--seed-start", type=int, default=0, help="Starting seed index.")

    ap.add_argument("--jobs", type=int, default=1, help="Parallel worker processes (seeds in parallel).")
    ap.add_argument("--blas-threads", type=int, default=1, help="BLAS threads per worker process.")

    ap.add_argument("--source-topology", default="1d_ring",
                    choices=["1d_ring", "1d_chain", "2d", "3d", "rr4", "rr6"])
    ap.add_argument("--targets", default="1d_ring,3d,rr4",
                    help="Comma-separated targets, e.g. 1d_ring,2d,3d,rr4,rr6")
    ap.add_argument("--depths", default="0,5,10,20,40,80,120",
                    help="Comma-separated depths. Supports ranges like 0-50.")

    ap.add_argument("--recovery-steps", type=int, default=600, help="Recovery steps per run.")
    ap.add_argument("--temp0", type=float, default=0.02, help="Initial temperature.")
    ap.add_argument("--temp-decay", type=float, default=0.9995, help="Temperature decay per step.")

    ap.add_argument("--mc-samples", type=int, default=8192, help="Pauli MC samples for leak estimator.")
    ap.add_argument("--kmax", type=int, default=1, help="Leak counts k-body support > kmax.")

    ap.add_argument("--min-mean-improvement", type=float, default=1e-3,
                    help="Success threshold for D*: mean leak_reduction >= this value.")
    ap.add_argument("--progress", action="store_true", help="Print progress as results arrive.")

    args = ap.parse_args()

    if args.N > 10:
        raise SystemExit("N>10 will be very slow for dense matrices. Use N<=10.")

    targets = parse_list(args.targets)
    depths = sorted(set(parse_int_list(args.depths)))

    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()

    # Prepare args payload for workers
    args_dict = {
        "N": int(args.N),
        "source_topology": args.source_topology,
        "targets": targets,
        "depths": depths,
        "recovery_steps": int(args.recovery_steps),
        "temp0": float(args.temp0),
        "temp_decay": float(args.temp_decay),
        "mc_samples": int(args.mc_samples),
        "kmax": int(args.kmax),
        "blas_threads": int(args.blas_threads),
    }

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # Run seeds (parallel if jobs>1)
    all_details: List[dict] = []

    if args.jobs <= 1:
        for s in seeds:
            _, det = run_one_seed(args_dict, s)
            all_details.extend(det)
            if args.progress:
                # print last few entries for heartbeat
                tail = det[-len(targets):]
                for row in tail:
                    print(f"[seed {row['seed']:3d}] D={row['depth']:4d} target={row['target']:8s} "
                          f"leak0={row['leak_initial']:.6f} leakF={row['leak_final']:.6f} leak_red={row['leak_reduction']:.6f}")
    else:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            futs = [ex.submit(run_one_seed, args_dict, s) for s in seeds]
            for fut in cf.as_completed(futs):
                s, det = fut.result()
                all_details.extend(det)
                if args.progress:
                    # Summarize per-seed quickly
                    # Print D=0 rows for that seed (first len(targets))
                    det_sorted = sorted(det, key=lambda r: (r["depth"], r["target"]))
                    for row in det_sorted[:len(targets)]:
                        print(f"[seed {row['seed']:3d}] D={row['depth']:4d} target={row['target']:8s} "
                              f"leak0={row['leak_initial']:.6f} leakF={row['leak_final']:.6f} leak_red={row['leak_reduction']:.6f}")

    # Aggregate curves
    curves: Dict[str, Dict[int, Dict[str, float]]] = {t: {} for t in targets}
    for row in all_details:
        tname = row["target"]
        D = int(row["depth"])
        rr0 = float(row["leak_initial"])
        rrF = float(row["leak_final"])
        rrR = float(row["leak_reduction"])
        acc = float(row["accepted_moves"]) / max(1, int(row["recovery_steps"]))

        slot = curves[tname].setdefault(D, {
            "n": 0,
            "sum_leak0": 0.0,
            "sum_leakF": 0.0,
            "sum_leak_red": 0.0,
            "sum_leak_red2": 0.0,
            "sum_accept": 0.0,
            "sum_accept2": 0.0,
        })
        slot["n"] += 1
        slot["sum_leak0"] += rr0
        slot["sum_leakF"] += rrF
        slot["sum_leak_red"] += rrR
        slot["sum_leak_red2"] += rrR ** 2
        slot["sum_accept"] += acc
        slot["sum_accept2"] += acc ** 2

    # Write CSV
    csv_path = os.path.join(args.out, "geometry_robustness_curves.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "target", "depth", "n",
            "mean_leak0", "mean_leakF", "mean_leak_reduction",
            "std_leak_reduction", "mean_accept_rate", "std_accept_rate",
        ])
        for tname in targets:
            for D in sorted(curves[tname].keys()):
                slot = curves[tname][D]
                n = slot["n"]
                mean_leak0 = slot["sum_leak0"] / n
                mean_leakF = slot["sum_leakF"] / n
                mean_red = slot["sum_leak_red"] / n
                var_red = max(0.0, slot["sum_leak_red2"] / n - mean_red ** 2)
                std_red = math.sqrt(var_red)
                mean_acc = slot["sum_accept"] / n
                var_acc = max(0.0, slot["sum_accept2"] / n - mean_acc ** 2)
                std_acc = math.sqrt(var_acc)
                w.writerow([tname, D, n, mean_leak0, mean_leakF, mean_red, std_red, mean_acc, std_acc])

    # Summary JSON
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "N": int(args.N),
        "source_topology": args.source_topology,
        "targets": targets,
        "depths": depths,
        "seeds": int(args.seeds),
        "seed_start": int(args.seed_start),
        "jobs": int(args.jobs),
        "blas_threads": int(args.blas_threads),
        "recovery": {
            "steps": int(args.recovery_steps),
            "temp0": float(args.temp0),
            "temp_decay": float(args.temp_decay),
        },
        "leak_estimator": {
            "mc_samples": int(args.mc_samples),
            "kmax": int(args.kmax),
        },
        "min_mean_improvement": float(args.min_mean_improvement),
        "by_target": {},
        "runtime_sec": None,
    }

    for tname in targets:
        depths_sorted = sorted(curves[tname].keys())
        mean_reds = []
        curve_rows = []
        for D in depths_sorted:
            slot = curves[tname][D]
            n = slot["n"]
            mean_leak0 = slot["sum_leak0"] / n
            mean_leakF = slot["sum_leakF"] / n
            mean_red = slot["sum_leak_red"] / n
            mean_reds.append(mean_red)
            curve_rows.append({
                "depth": int(D),
                "n": int(n),
                "mean_leak0": float(mean_leak0),
                "mean_leakF": float(mean_leakF),
                "mean_leak_reduction": float(mean_red),
            })
        D_star = compute_D_star(depths_sorted, mean_reds, args.min_mean_improvement)
        summary["by_target"][tname] = {"curve": curve_rows, "D_star": D_star}

    summary["runtime_sec"] = float(time.time() - t0)

    json_summary_path = os.path.join(args.out, "geometry_robustness_summary.json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    json_details_path = os.path.join(args.out, "geometry_robustness_details.json")
    with open(json_details_path, "w", encoding="utf-8") as f:
        json.dump({"details": all_details}, f, indent=2)

    if args.progress:
        print("\nWrote:")
        print(" -", csv_path)
        print(" -", json_summary_path)
        print(" -", json_details_path)
        print("\nD* estimates:")
        for tname in targets:
            print(f"  {tname:8s}  D* = {summary['by_target'][tname]['D_star']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
