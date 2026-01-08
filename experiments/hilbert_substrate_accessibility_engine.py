"""
run_graph_topology_stress_test.py

Graph/topology stress test for the "Humpty Dumpty" locality-attractor experiment.

What it does
------------
For each graph topology, and for each N:
  1) Build a target Hamiltonian H_target as a Heisenberg sum over the graph edges.
  2) Compute:
       - "Spatial target cost"  = cost(H_target)
       - "Harmonion (eigen) cost" = cost(diag(eigs(H_target)))   [theoretical optimum if fully diagonalized]
  3) For each restart:
       - Scramble H_target by a random unitary conjugation -> H_scrambled
       - Run Riemannian descent on the unitary orbit (H -> U H U†) to minimize locality cost
       - Record recovered cost and runtime
  4) Save per-run JSON + per-sweep CSV summary.

This is the direct, apples-to-apples topology robustness test for your observed N=4→5 accessibility transition.

Dependencies
------------
numpy, scipy

Example (Windows one-liner)
---------------------------
python run_graph_topology_stress_test.py --n-list 3,4,5 --graphs chain,ring,grid2d,random_regular --restarts 5 --penalty-power 4 --steps 120 --out out_topology_test
"""

import argparse
import json
import math
import os
import time
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.linalg import expm, norm, eigh


# ----------------------------
# Basic Pauli utilities
# ----------------------------

def pauli_matrices() -> List[np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I, X, Y, Z]


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for k in range(1, len(ops)):
        out = np.kron(out, ops[k])
    return out


@dataclass
class PauliBasis:
    mats: np.ndarray        # shape (4^N, dim, dim)
    weights: np.ndarray     # shape (4^N,)
    dim: int
    N: int


def build_pauli_basis(N: int) -> PauliBasis:
    """
    Builds full N-qubit Pauli product basis and weights (support size).
    Note: 4^N matrices of size (2^N,2^N) can get heavy beyond N=6.
    """
    P = pauli_matrices()
    dim = 2 ** N

    mats = []
    weights = []

    # product over {I,X,Y,Z}^N
    for idxs in itertools.product(range(4), repeat=N):
        ops = [P[i] for i in idxs]
        mats.append(kron_n(ops))
        weights.append(sum(1 for i in idxs if i != 0))

    mats = np.array(mats)
    weights = np.array(weights, dtype=float)
    return PauliBasis(mats=mats, weights=weights, dim=dim, N=N)


# ----------------------------
# Graph generators (no networkx)
# ----------------------------

def edges_chain(N: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(N - 1)]


def edges_ring(N: int) -> List[Tuple[int, int]]:
    if N < 3:
        return edges_chain(N)
    e = edges_chain(N)
    e.append((N - 1, 0))
    return e


def edges_complete(N: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def edges_grid2d(N: int) -> List[Tuple[int, int]]:
    """
    2D square lattice with nearest neighbors.
    Requires N = L*L
    """
    L = int(round(math.sqrt(N)))
    if L * L != N:
        raise ValueError(f"grid2d requires N to be a perfect square; got N={N}")
    def node(r, c): return r * L + c

    e = []
    for r in range(L):
        for c in range(L):
            if r + 1 < L:
                e.append((node(r, c), node(r + 1, c)))
            if c + 1 < L:
                e.append((node(r, c), node(r, c + 1)))
    return e


def edges_random_regular(N: int, degree: int, rng: np.random.Generator, max_tries: int = 2000) -> List[Tuple[int, int]]:
    """
    Simple retry-based random d-regular graph generator for small N.
    degree must satisfy N*degree even and degree < N.
    """
    if degree >= N:
        raise ValueError("random_regular requires degree < N")
    if (N * degree) % 2 != 0:
        raise ValueError("random_regular requires N*degree even")

    for _ in range(max_tries):
        stubs = []
        for i in range(N):
            stubs.extend([i] * degree)
        rng.shuffle(stubs)

        edges = set()
        ok = True
        for k in range(0, len(stubs), 2):
            a, b = stubs[k], stubs[k + 1]
            if a == b:
                ok = False
                break
            u, v = (a, b) if a < b else (b, a)
            if (u, v) in edges:
                ok = False
                break
            edges.add((u, v))

        if ok:
            return sorted(list(edges))

    raise RuntimeError(f"Failed to generate random {degree}-regular graph after {max_tries} tries.")


def build_edges(graph_name: str, N: int, rng: np.random.Generator, degree: int) -> List[Tuple[int, int]]:
    g = graph_name.lower().strip()
    if g == "chain":
        return edges_chain(N)
    if g == "ring":
        return edges_ring(N)
    if g == "complete":
        return edges_complete(N)
    if g == "grid2d":
        return edges_grid2d(N)
    if g in ("random_regular", "rrg", "regular"):
        return edges_random_regular(N, degree=degree, rng=rng)
    raise ValueError(f"Unknown graph '{graph_name}'. Supported: chain, ring, complete, grid2d, random_regular")


# ----------------------------
# Hamiltonian construction
# ----------------------------

def heisenberg_on_edges(N: int, edges: List[Tuple[int, int]], J: float = 1.0) -> np.ndarray:
    """
    H = sum_{(i,j) in edges} J (X_i X_j + Y_i Y_j + Z_i Z_j)
    """
    P = pauli_matrices()
    I, X, Y, Z = P

    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    def term_two_local(opA, i, opB, j):
        ops = [I] * N
        ops[i] = opA
        ops[j] = opB
        return kron_n(ops)

    for (i, j) in edges:
        H += J * (term_two_local(X, i, X, j) + term_two_local(Y, i, Y, j) + term_two_local(Z, i, Z, j))

    # Ensure Hermitian
    H = 0.5 * (H + H.conj().T)
    return H


# ----------------------------
# Cost + Riemannian gradient on unitary orbit
# ----------------------------

@dataclass
class CostEngine:
    basis: PauliBasis
    penalty_power: int

    def cost_and_grad_generator(self, H: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Returns:
          cost: scalar
          K: tangent generator for unitary-orbit descent (Hermitian)
             implemented as K = i[H, M], where M is Euclidean gradient operator.
        """
        B = self.basis
        dim = B.dim

        # Coeffs c_k = Tr(H P_k) / dim (real for Hermitian H and Pauli basis)
        coeffs = np.real(np.einsum("ij,kji->k", H, B.mats)) / dim

        norm_sq = float(np.sum(coeffs ** 2))
        if norm_sq <= 1e-30:
            # degenerate; return zero gradient
            return float("inf"), np.zeros_like(H)

        w = B.weights.astype(float)
        wp = w ** self.penalty_power

        cost = float(np.sum(wp * (coeffs ** 2)) / norm_sq)

        # Euclidean gradient w.r.t. coeffs; then lift back to operator
        grad_coeffs = (2.0 * wp * coeffs) / norm_sq
        M = np.tensordot(grad_coeffs, B.mats, axes=([0], [0]))

        comm = H @ M - M @ H
        K = 1j * comm  # Hermitian if H,M Hermitian
        K = 0.5 * (K + K.conj().T)
        return cost, K


def random_scrambler_unitary(dim: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    """
    U = exp(i * G * scale), with G Hermitian from complex Gaussian.
    """
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    G = A + A.conj().T
    return expm(1j * G * scale)


@dataclass
class OptConfig:
    steps: int = 120
    lr: float = 0.1
    line_search_steps: int = 6
    grad_norm_floor: float = 1e-7
    lr_decay_if_fail: float = 0.5
    alpha_decay: float = 0.5


def riemannian_descent(H0: np.ndarray, engine: CostEngine, cfg: OptConfig) -> Dict:
    """
    Minimize cost(H) over the unitary orbit H = U H0 U† via Riemannian gradient flow:
      K = i[H, M]
      U_step = exp(-i * alpha * K_dir)
    """
    dim = H0.shape[0]
    H = H0.copy()
    U_total = np.eye(dim, dtype=complex)

    lr = float(cfg.lr)
    history = []
    t0 = time.time()

    for step in range(cfg.steps + 1):
        cost_old, K = engine.cost_and_grad_generator(H)

        K_norm = float(norm(K, "fro"))
        if K_norm < cfg.grad_norm_floor or not np.isfinite(cost_old):
            break

        K_dir = K / (K_norm + 1e-30)

        alpha = lr
        accepted = False
        best_try_cost = None

        for _ in range(cfg.line_search_steps):
            U_step = expm(-1j * alpha * K_dir)
            H_try = U_step @ H @ U_step.conj().T
            cost_try, _ = engine.cost_and_grad_generator(H_try)

            best_try_cost = cost_try if best_try_cost is None else min(best_try_cost, cost_try)

            if np.isfinite(cost_try) and (cost_try < cost_old):
                H = H_try
                U_total = U_step @ U_total
                cost_old = cost_try
                accepted = True
                break

            alpha *= cfg.alpha_decay

        if not accepted:
            lr *= cfg.lr_decay_if_fail

        if step % 10 == 0:
            history.append({"step": step, "cost": float(cost_old), "lr": float(lr), "K_norm": float(K_norm)})

    t1 = time.time()
    final_cost, _ = engine.cost_and_grad_generator(H)

    return {
        "final_cost": float(final_cost),
        "elapsed_sec": float(t1 - t0),
        "history": history,
        "U_total": U_total,
        "H_final": H,
    }


# ----------------------------
# Harmonion (eigen) theoretical optimum proxy
# ----------------------------

def harmonion_cost_via_diagonal(H: np.ndarray, engine: CostEngine) -> float:
    """
    Diagonalize H = V diag(e) V†, then compute cost(diag(e)).
    This matches your operational definition: "cost of stationary/eigen basis"
    expressed as a diagonal matrix in the computational basis.

    Note: This is a *proxy* for "best cost in eigenbasis coordinates", not a proof of global optimum.
    """
    evals, _ = eigh(H)
    H_diag = np.diag(evals.astype(complex))
    c, _ = engine.cost_and_grad_generator(H_diag)
    return float(c)


# ----------------------------
# Experiment runner
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def basin_label(cost: float, spatial_cost: float, deep_cost: float, tol: float = 1.5) -> str:
    """
    Simple classifier for reporting:
      - "spatial" if near spatial_cost
      - "deep" if near deep_cost
      - otherwise "intermediate"
    """
    if abs(cost - spatial_cost) <= tol:
        return "spatial"
    if abs(cost - deep_cost) <= tol:
        return "deep"
    return "intermediate"


def main():
    ap = argparse.ArgumentParser(
        description="Graph/topology stress test for the Humpty Dumpty locality-attractor experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--n-list", type=str, default="3,4,5",
                    help="Comma-separated N values to test (qubits / nodes).")
    ap.add_argument("--graphs", type=str, default="chain,ring,random_regular,grid2d",
                    help="Comma-separated graphs: chain, ring, complete, grid2d, random_regular.")
    ap.add_argument("--degree", type=int, default=2,
                    help="Degree for random_regular graphs.")
    ap.add_argument("--restarts", type=int, default=5,
                    help="Number of independent scrambles+optimizations per (N,graph).")
    ap.add_argument("--penalty-power", type=int, default=4,
                    help="Locality penalty exponent p in w^p.")
    ap.add_argument("--steps", type=int, default=120,
                    help="Optimization steps per restart.")
    ap.add_argument("--lr", type=float, default=0.1,
                    help="Initial learning rate.")
    ap.add_argument("--scramble-scale", type=float, default=0.2,
                    help="Scale for random scrambler unitary exponent.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base RNG seed.")
    ap.add_argument("--out", type=str, default="out_topology_test",
                    help="Output directory.")

    args = ap.parse_args()

    N_list = parse_int_list(args.n_list)
    graphs = parse_csv_list(args.graphs)

    ensure_dir(args.out)

    # Summary CSV header
    summary_rows = []
    summary_path = os.path.join(args.out, "summary.csv")
    detail_dir = os.path.join(args.out, "details")
    ensure_dir(detail_dir)

    run_meta = {
        "n_list": N_list,
        "graphs": graphs,
        "degree": args.degree,
        "restarts": args.restarts,
        "penalty_power": args.penalty_power,
        "steps": args.steps,
        "lr": args.lr,
        "scramble_scale": args.scramble_scale,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("\n============================================================")
    print("GRAPH / TOPOLOGY STRESS TEST")
    print("============================================================")
    print(json.dumps(run_meta, indent=2))

    for N in N_list:
        print(f"\n----------------------------")
        print(f"Building Pauli basis for N={N} (4^N={4**N} matrices, dim={2**N})")
        print(f"----------------------------")
        t_basis0 = time.time()
        basis = build_pauli_basis(N)
        t_basis1 = time.time()
        print(f"Pauli basis built in {t_basis1 - t_basis0:.2f}s")

        engine = CostEngine(basis=basis, penalty_power=args.penalty_power)
        opt_cfg = OptConfig(steps=args.steps, lr=args.lr)

        rng_master = np.random.default_rng(args.seed + 1000 * N)

        for gname in graphs:
            rng_graph = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

            try:
                edges = build_edges(gname, N, rng_graph, degree=args.degree)
            except Exception as e:
                print(f"[skip] N={N} graph={gname}: {e}")
                continue

            # Target H for this graph
            H_target = heisenberg_on_edges(N, edges, J=1.0)

            spatial_cost, _ = engine.cost_and_grad_generator(H_target)
            harmonion_cost = harmonion_cost_via_diagonal(H_target, engine)

            print(f"\n=== N={N} graph={gname} |E|={len(edges)} ===")
            print(f"Spatial target cost:   {spatial_cost:.6f}")
            print(f"Harmonion (diag) cost: {harmonion_cost:.6f}")

            recovered_costs = []
            labels = []
            elapsed = []

            # Restarts: independent scrambles
            for r in range(args.restarts):
                rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

                U_scram = random_scrambler_unitary(basis.dim, rng, scale=args.scramble_scale)
                H_scrambled = U_scram @ H_target @ U_scram.conj().T
                c_scrambled, _ = engine.cost_and_grad_generator(H_scrambled)

                result = riemannian_descent(H_scrambled, engine, opt_cfg)

                c_final = float(result["final_cost"])
                recovered_costs.append(c_final)
                elapsed.append(float(result["elapsed_sec"]))
                labels.append(basin_label(c_final, spatial_cost, harmonion_cost))

                # Save a small detail JSON per restart (no huge arrays by default)
                detail = {
                    "N": N,
                    "graph": gname,
                    "edges": edges,
                    "restart": r,
                    "penalty_power": args.penalty_power,
                    "scramble_scale": args.scramble_scale,
                    "cost_spatial_target": float(spatial_cost),
                    "cost_harmonion_diag": float(harmonion_cost),
                    "cost_scrambled": float(c_scrambled),
                    "cost_final": float(c_final),
                    "basin_label": labels[-1],
                    "elapsed_sec": float(result["elapsed_sec"]),
                    "history": result["history"],
                }

                detail_path = os.path.join(detail_dir, f"detail_N{N}_{gname}_r{r}.json")
                with open(detail_path, "w", encoding="utf-8") as f:
                    json.dump(detail, f, indent=2)

                print(f"  restart {r+1}/{args.restarts}: scrambled={c_scrambled:.4f} -> final={c_final:.4f} [{labels[-1]}] ({elapsed[-1]:.2f}s)")

            recovered = np.array(recovered_costs, dtype=float)
            mean_c = float(np.mean(recovered))
            std_c = float(np.std(recovered))
            min_c = float(np.min(recovered))
            max_c = float(np.max(recovered))

            # Basin counts
            counts = {k: labels.count(k) for k in sorted(set(labels))}
            counts_all = {"deep": labels.count("deep"), "intermediate": labels.count("intermediate"), "spatial": labels.count("spatial")}

            row = {
                "N": N,
                "graph": gname,
                "num_edges": len(edges),
                "degree_param": args.degree if gname.lower() in ("random_regular", "rrg", "regular") else "",
                "penalty_power": args.penalty_power,
                "restarts": args.restarts,
                "spatial_target_cost": float(spatial_cost),
                "harmonion_diag_cost": float(harmonion_cost),
                "recovered_mean": mean_c,
                "recovered_std": std_c,
                "recovered_min": min_c,
                "recovered_max": max_c,
                "basin_deep": counts_all["deep"],
                "basin_intermediate": counts_all["intermediate"],
                "basin_spatial": counts_all["spatial"],
                "elapsed_mean_sec": float(np.mean(elapsed)) if len(elapsed) else float("nan"),
            }
            summary_rows.append(row)

    # Write summary.csv
    if summary_rows:
        cols = list(summary_rows[0].keys())
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row[c]) for c in cols) + "\n")

        print("\n============================================================")
        print(f"Done. Wrote: {summary_path}")
        print(f"Details in:  {detail_dir}")
        print("============================================================")
    else:
        print("\nNo results produced (check N/graph compatibility).")


if __name__ == "__main__":
    main()
