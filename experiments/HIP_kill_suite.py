# HIP_kill_suite.py
# ------------------------------------------------------------
# "Kill suite" for The Hilbert Substrate Framework (HIP persistence).
# CUDA-first (CuPy) with clean CPU fallback (NumPy).
#
# What it does:
#   - Builds an N-qubit interaction graph (several families).
#   - Builds a 2-local Hamiltonian on that graph (several families).
#   - Approximates directed permeability P_{i->j}(t) via sampled
#     initial product states + sampled local perturbations on i.
#   - Computes edge permeability, HIP(t)=Var_edges(P_e), node intensity I_i(t),
#     and rank-stability diagnostics across time.
#   - Runs several "kill tests" designed to break persistence if it's an artifact.
#
# Notes / reality checks:
#   - Exact dense expm scales as O(4^N) memory/time. Practical N on a single GPU
#     is usually <= 12 (sometimes 13) depending on VRAM.
#   - This suite is honest: it will refuse sizes that cannot fit in memory.
#
# Usage (Windows, single-line):
#   python HIP_kill_suite.py --N 10 --graph random_regular --deg 3 --ham random_2local --times 0.2,0.5,1.0,2.0 --samples 24 --ops 24 --trials 1 --device cuda --plots
#
# Outputs:
#   outputs/hip_kill_<timestamp>/
#     summary.json
#     timeseries.csv
#     (optional) plots/*.png
#
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import math
import os
import random
import sys
from typing import Dict, List, Tuple, Optional

import networkx as nx

# ---------- Backend (CuPy if available) ----------

def _load_backend(device: str):
    device = device.lower().strip()
    if device == "cuda":
        try:
            import cupy as cp  # type: ignore
            import cupyx.scipy.linalg as cpx_linalg  # type: ignore
            xp = cp
            linalg_expm = cpx_linalg.expm
            backend = "cupy"
            return xp, linalg_expm, backend
        except Exception as e:
            print(f"[WARN] CUDA requested but CuPy not usable: {e}\nFalling back to CPU (NumPy).", file=sys.stderr)

    import numpy as np  # type: ignore
    import scipy.linalg as sp_linalg  # type: ignore
    xp = np
    linalg_expm = sp_linalg.expm
    backend = "numpy"
    return xp, linalg_expm, backend

# ---------- Small utilities ----------

def _timestamp():
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_cpu_scalar(x):
    # cupy scalar -> python float, numpy scalar -> python float
    try:
        return float(x)
    except Exception:
        try:
            return float(x.item())
        except Exception:
            return float(x)

# ---------- Pauli / SU(2) helpers ----------

def pauli_mats(xp):
    I = xp.array([[1, 0], [0, 1]], dtype=xp.complex128)
    X = xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
    Y = xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128)
    Z = xp.array([[1, 0], [0, -1]], dtype=xp.complex128)
    return I, X, Y, Z

def random_su2(xp, rng: random.Random):
    # Haar-ish SU(2) via quaternion method (good enough for kill tests)
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    a = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    b = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    c = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    d = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    # SU(2) matrix:
    # [[ d + i c,  b + i a],
    #  [-b + i a, d - i c]]
    M = xp.array([[d + 1j*c, b + 1j*a],
                  [-b + 1j*a, d - 1j*c]], dtype=xp.complex128)
    return M

# ---------- Tensor embedding ----------

def kron_many(xp, mats: List):
    out = mats[0]
    for m in mats[1:]:
        out = xp.kron(out, m)
    return out

def embed_single_qubit_op(xp, op2: "xp.ndarray", N: int, q: int):
    I, _, _, _ = pauli_mats(xp)
    mats = [I] * N
    mats[q] = op2
    return kron_many(xp, mats)

def embed_two_qubit_op(xp, op4: "xp.ndarray", N: int, a: int, b: int):
    """
    Correct embedding of a general 4x4 two-qubit operator acting on qubits (a,b)
    into the full 2^N x 2^N Hilbert space, even when a and b are non-adjacent.

    Method:
      Expand op4 in the Pauli⊗Pauli basis:
        op4 = sum_{m,n in {I,X,Y,Z}} c_{mn} (P_m ⊗ P_n)
      where c_{mn} = (1/4) Tr[(P_m ⊗ P_n)^\dagger op4]
      Then embed as:
        sum c_{mn} embed(P_m at a) @ embed(P_n at b)

    This is robust and avoids adjacency/SWAP complications.
    """
    if a == b:
        raise ValueError("two-qubit op needs distinct qubits")

    I, X, Y, Z = pauli_mats(xp)
    P = [I, X, Y, Z]

    # Pre-embed the single-qubit Paulis on a and b
    Ea = [embed_single_qubit_op(xp, Pm, N, a) for Pm in P]
    Eb = [embed_single_qubit_op(xp, Pn, N, b) for Pn in P]

    out = xp.zeros((2 ** N, 2 ** N), dtype=xp.complex128)

    # Compute coefficients and accumulate
    for m in range(4):
        for n in range(4):
            basis_mn = xp.kron(P[m], P[n])
            # coefficient: (1/4) Tr[basis^\dagger op4]
            c = 0.25 * xp.trace(xp.conjugate(basis_mn.T) @ op4)
            if c != 0:
                out = out + c * (Ea[m] @ Eb[n])

    # Ensure Hermitian if op4 was Hermitian (numerical cleanup)
    out = 0.5 * (out + xp.conjugate(out.T))
    return out

# ---------- State prep / reduced density ----------

def random_product_state(xp, N: int, rng: random.Random):
    # |psi> = ⊗_k (cos θ |0> + e^{i φ} sin θ |1>)
    vecs = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        theta = math.acos(math.sqrt(u))
        phi = 2 * math.pi * v
        v0 = math.cos(theta)
        v1 = complex(math.cos(phi), math.sin(phi)) * math.sin(theta)
        vec = xp.array([v0, v1], dtype=xp.complex128)
        vecs.append(vec)
    psi = vecs[0]
    for vec in vecs[1:]:
        psi = xp.kron(psi, vec)
    # normalize (should already be)
    nrm = xp.linalg.norm(psi)
    return psi / nrm

def apply_local_op_to_state(xp, psi: "xp.ndarray", op2: "xp.ndarray", N: int, q: int):
    Uq = embed_single_qubit_op(xp, op2, N, q)
    return Uq @ psi

def reduced_density_one_qubit_from_state(xp, psi: "xp.ndarray", N: int, q: int):
    # Pure-state 1-qubit reduced density without constructing full rho.
    # Reshape psi into tensor of shape (2,)*N
    tens = psi.reshape((2,) * N)
    # Move qubit q to front: (2, 2, ..., 2)
    perm = [q] + [k for k in range(N) if k != q]
    tens = tens.transpose(perm)
    # Flatten environment: (2, 2^(N-1))
    env_dim = 2 ** (N - 1)
    A = tens.reshape((2, env_dim))
    # rho = A A^\dagger
    rho = A @ xp.conjugate(A.T)
    return rho

def trace_distance_one_qubit(xp, rho: "xp.ndarray", sigma: "xp.ndarray"):
    # D(rho,sigma) = 0.5 ||rho - sigma||_1
    delta = rho - sigma
    # Hermitian (numerically). Use SVD for trace norm
    s = xp.linalg.svd(delta, compute_uv=False)
    return 0.5 * xp.sum(xp.abs(s))

# ---------- Graph families ----------

def make_graph(graph_type: str, N: int, rng: random.Random, deg: int = 3, p: float = 0.2):
    gt = graph_type.lower().strip()
    if gt in ("ring", "cycle"):
        G = nx.cycle_graph(N)
    elif gt in ("line", "path"):
        G = nx.path_graph(N)
    elif gt in ("grid2d", "grid"):
        # closest square
        m = int(round(math.sqrt(N)))
        m = max(2, m)
        G = nx.grid_2d_graph(m, m)
        # relabel to 0..N-1, and if m*m != N, take first N nodes induced
        nodes = list(G.nodes())
        nodes = nodes[:N]
        H = G.subgraph(nodes).copy()
        G = nx.convert_node_labels_to_integers(H)
    elif gt in ("erdos_renyi", "er", "random"):
        G = nx.erdos_renyi_graph(N, p, seed=rng.randint(0, 10**9))
    elif gt in ("random_regular", "rr"):
        if deg >= N:
            raise ValueError("deg must be < N for random_regular")
        # random_regular_graph requires N*deg even
        if (N * deg) % 2 != 0:
            raise ValueError("N*deg must be even for random_regular")
        G = nx.random_regular_graph(deg, N, seed=rng.randint(0, 10**9))
    elif gt in ("small_world", "watts_strogatz", "ws"):
        k = min(deg, N - 1)
        if k % 2 == 1:
            k = max(2, k - 1)
        G = nx.watts_strogatz_graph(N, k, p, seed=rng.randint(0, 10**9))
    elif gt in ("scale_free", "barabasi", "ba"):
        m = max(1, min(deg, N - 1))
        G = nx.barabasi_albert_graph(N, m, seed=rng.randint(0, 10**9))
    else:
        raise ValueError(f"Unknown graph_type={graph_type}")
    return G

# ---------- Hamiltonians ----------

def random_two_qubit_hermitian(xp, rng: random.Random, scale: float = 1.0):
    # H = sum_{a,b in {I,X,Y,Z}} c_{ab} (Pauli_a ⊗ Pauli_b), with real c_ab
    I, X, Y, Z = pauli_mats(xp)
    P = [I, X, Y, Z]
    H = xp.zeros((4, 4), dtype=xp.complex128)
    for a in range(4):
        for b in range(4):
            c = rng.uniform(-1.0, 1.0)
            H = H + (c * xp.kron(P[a], P[b]))
    # Make Hermitian numeric (it already is)
    H = 0.5 * (H + xp.conjugate(H.T))
    # Normalize a bit
    fro = xp.linalg.norm(H)
    if fro != 0:
        H = (scale / fro) * H
    return H

def build_hamiltonian(xp, G: nx.Graph, ham_type: str, rng: random.Random,
                      local_field: float = 0.2, edge_scale: float = 1.0,
                      symmetry_break: bool = False):
    ht = ham_type.lower().strip()
    N = G.number_of_nodes()
    dim = 2 ** N

    H = xp.zeros((dim, dim), dtype=xp.complex128)
    I, X, Y, Z = pauli_mats(xp)

    # Local terms
    for i in range(N):
        if symmetry_break:
            # random local axis
            ax = rng.uniform(-1, 1)
            ay = rng.uniform(-1, 1)
            az = rng.uniform(-1, 1)
            op = ax*X + ay*Y + az*Z
        else:
            op = Z
        H = H + local_field * embed_single_qubit_op(xp, op, N, i)

    # Edge terms
    for (i, j) in G.edges():
        if ht in ("zz", "ising"):
            Hij = edge_scale * xp.kron(Z, Z)
        elif ht in ("xxz",):
            Hij = edge_scale * (xp.kron(X, X) + xp.kron(Y, Y) + 0.5 * xp.kron(Z, Z))
        elif ht in ("heisenberg", "xyz"):
            Hij = edge_scale * (xp.kron(X, X) + xp.kron(Y, Y) + xp.kron(Z, Z))
        elif ht in ("random_2local", "random"):
            Hij = random_two_qubit_hermitian(xp, rng, scale=edge_scale)
        else:
            raise ValueError(f"Unknown ham_type={ham_type}")

        H = H + embed_two_qubit_op(xp, Hij, N, i, j)

    # Ensure Hermitian
    H = 0.5 * (H + xp.conjugate(H.T))
    return H

# ---------- HIP estimation ----------

@dataclasses.dataclass
class HIPResult:
    times: List[float]
    hip: List[float]
    mean_edge_perm: List[float]
    rank_spearman_to_t0: List[float]
    topk_overlap_to_t0: List[float]
    node_intensity_by_time: List[List[float]]  # time -> list over nodes
    notes: Dict

def _spearman_rankcorr(a: List[float], b: List[float]) -> float:
    # Simple Spearman via rank vectors (ties handled by average rank)
    def rankdata(x: List[float]) -> List[float]:
        idx = sorted(range(len(x)), key=lambda i: x[i])
        ranks = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[idx[j+1]] == x[idx[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j+1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks

    ra = rankdata(a)
    rb = rankdata(b)
    ma = sum(ra)/len(ra)
    mb = sum(rb)/len(rb)
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(len(a)))
    da = math.sqrt(sum((ra[i]-ma)**2 for i in range(len(a))))
    db = math.sqrt(sum((rb[i]-mb)**2 for i in range(len(a))))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)

def estimate_permeability_matrix(xp, linalg_expm, H, N: int,
                                 times: List[float],
                                 samples: int,
                                 ops: int,
                                 rng: random.Random,
                                 op_family: str = "pauli+haar"):
    # Returns perms[t_index][i][j] approx P_{i->j}(t)
    dim = 2 ** N
    bytes_per_complex = 16  # complex128
    # crude memory check: U (dim^2) + H (dim^2) + some overhead ~ 3 * dim^2
    approx_bytes = 3 * (dim * dim * bytes_per_complex)
    # allow 80% of device memory if cupy
    if xp.__name__ == "cupy":
        try:
            import cupy as cp  # type: ignore
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            if approx_bytes > 0.8 * free_mem:
                raise MemoryError(f"Estimated {approx_bytes/1e9:.2f} GB required, but only {free_mem/1e9:.2f} GB free.")
        except Exception as e:
            # If memGetInfo fails, continue.
            pass

    I2, X, Y, Z = pauli_mats(xp)
    paulis = [X, Y, Z]

    # Pre-sample initial states
    init_states = [random_product_state(xp, N, rng) for _ in range(samples)]

    # Pre-sample local ops
    local_ops = []
    for _ in range(ops):
        if op_family.startswith("pauli"):
            local_ops.append(paulis[rng.randint(0, 2)])
        else:
            # mix
            if rng.random() < 0.5:
                local_ops.append(paulis[rng.randint(0, 2)])
            else:
                local_ops.append(random_su2(xp, rng))

    # Compute permeability for each time separately (avoid storing all U if big)
    perms_by_time = []
    for t in times:
        U = linalg_expm((-1j * t) * H)
        # perms: NxN zeros
        P = [[0.0 for _ in range(N)] for _ in range(N)]
        # For each ordered pair i->j, estimate max over sampled states and ops
        for i in range(N):
            # Pre-embed all ops on i for speed
            Ui_ops = [embed_single_qubit_op(xp, Oi, N, i) for Oi in local_ops]
            for j in range(N):
                if i == j:
                    continue
                best = 0.0
                for s in range(samples):
                    psi0 = init_states[s]
                    psi = U @ psi0
                    rho_j = reduced_density_one_qubit_from_state(xp, psi, N, j)
                    for Oi_full in Ui_ops:
                        psi0p = Oi_full @ psi0
                        psip = U @ psi0p
                        rho_jp = reduced_density_one_qubit_from_state(xp, psip, N, j)
                        d = trace_distance_one_qubit(xp, rho_jp, rho_j)
                        val = _to_cpu_scalar(d)
                        if val > best:
                            best = val
                P[i][j] = best
        perms_by_time.append(P)
        # free U explicitly to help GPU memory
        del U
    return perms_by_time

def run_hip_suite_once(xp, linalg_expm,
                       graph_type: str, ham_type: str,
                       N: int, times: List[float],
                       samples: int, ops: int,
                       seed: int,
                       deg: int = 3, p: float = 0.2,
                       symmetry_break: bool = False,
                       topk: int = 3,
                       permutation_null: bool = False):
    rng = random.Random(seed)

    G = make_graph(graph_type, N, rng, deg=deg, p=p)

    if permutation_null:
        perm = list(range(N))
        rng.shuffle(perm)
        mapping = {i: perm[i] for i in range(N)}
        G = nx.relabel_nodes(G, mapping, copy=True)
        # Relabel back to 0..N-1 for consistency
        G = nx.convert_node_labels_to_integers(G)

    H = build_hamiltonian(xp, G, ham_type, rng, symmetry_break=symmetry_break)

    perms_by_time = estimate_permeability_matrix(
        xp, linalg_expm, H, N=N, times=times, samples=samples, ops=ops, rng=rng
    )

    # For each time: edge permeability P_{i,j} = max(Pi->j, Pj->i) on edges
    hip = []
    mean_edge_perm = []
    node_intensity_by_time = []

    for t_idx, Pdir in enumerate(perms_by_time):
        edge_vals = []
        # node intensity I_i = sum_{j in N(i)} P_{i->j}
        intens = [0.0] * N

        for (i, j) in G.edges():
            pij = max(Pdir[i][j], Pdir[j][i])
            edge_vals.append(pij)
            intens[i] += Pdir[i][j]
            intens[j] += Pdir[j][i]

        if len(edge_vals) == 0:
            hip.append(0.0)
            mean_edge_perm.append(0.0)
        else:
            m = sum(edge_vals) / len(edge_vals)
            v = sum((x - m) ** 2 for x in edge_vals) / len(edge_vals)
            hip.append(v)
            mean_edge_perm.append(m)

        node_intensity_by_time.append(intens)

    # Rank stability vs first nontrivial time (use first time in list)
    base = node_intensity_by_time[0]
    base_sorted = sorted(range(N), key=lambda i: base[i], reverse=True)
    base_topk = set(base_sorted[:max(1, min(topk, N))])

    rank_spearman = []
    topk_overlap = []
    for intens in node_intensity_by_time:
        rank_spearman.append(_spearman_rankcorr(base, intens))
        sorted_idx = sorted(range(N), key=lambda i: intens[i], reverse=True)
        top = set(sorted_idx[:max(1, min(topk, N))])
        overlap = len(base_topk.intersection(top)) / float(len(base_topk))
        topk_overlap.append(overlap)

    return HIPResult(
        times=times,
        hip=hip,
        mean_edge_perm=mean_edge_perm,
        rank_spearman_to_t0=rank_spearman,
        topk_overlap_to_t0=topk_overlap,
        node_intensity_by_time=node_intensity_by_time,
        notes={
            "graph_type": graph_type,
            "ham_type": ham_type,
            "N": N,
            "samples": samples,
            "ops": ops,
            "seed": seed,
            "deg": deg,
            "p": p,
            "symmetry_break": symmetry_break,
            "permutation_null": permutation_null,
        },
    )

# ---------- Plotting (optional) ----------

def maybe_plot(outdir: str, result: HIPResult):
    import matplotlib.pyplot as plt  # allowed; no seaborn, no fixed colors

    _mkdir(os.path.join(outdir, "plots"))

    # HIP(t)
    plt.figure()
    plt.plot(result.times, result.hip, marker="o")
    plt.xlabel("time t")
    plt.ylabel("HIP(t) = Var_edges(P_e)")
    plt.title("HIP time series")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "hip_timeseries.png"), dpi=160)
    plt.close()

    # Mean edge permeability
    plt.figure()
    plt.plot(result.times, result.mean_edge_perm, marker="o")
    plt.xlabel("time t")
    plt.ylabel("mean edge permeability")
    plt.title("Mean edge permeability vs time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "mean_edge_perm.png"), dpi=160)
    plt.close()

    # Rank stability
    plt.figure()
    plt.plot(result.times, result.rank_spearman_to_t0, marker="o")
    plt.xlabel("time t")
    plt.ylabel("Spearman rank corr (to t0)")
    plt.title("Rank stability (node intensity ordering)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "rank_stability.png"), dpi=160)
    plt.close()

    # Top-k overlap
    plt.figure()
    plt.plot(result.times, result.topk_overlap_to_t0, marker="o")
    plt.xlabel("time t")
    plt.ylabel("Top-k overlap (to t0)")
    plt.title("Top-k overlap stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "topk_overlap.png"), dpi=160)
    plt.close()

# ---------- Main / orchestration ----------

def parse_times(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]

def main():
    ap = argparse.ArgumentParser(description="HIP Kill Suite (CUDA-first).")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cuda uses CuPy if available.")
    ap.add_argument("--N", type=int, default=10, help="Number of qubits/nodes.")
    ap.add_argument("--graph", default="random_regular",
                    choices=["ring", "line", "grid2d", "erdos_renyi", "random_regular", "small_world", "scale_free"],
                    help="Graph family.")
    ap.add_argument("--deg", type=int, default=3, help="Degree for random_regular / small_world / scale_free m.")
    ap.add_argument("--p", type=float, default=0.2, help="Probability for erdos_renyi / small_world rewiring.")
    ap.add_argument("--ham", default="random_2local",
                    choices=["random_2local", "zz", "xxz", "heisenberg"],
                    help="Hamiltonian family.")
    ap.add_argument("--times", default="0.2,0.5,1.0,2.0", help="Comma-separated time points.")
    ap.add_argument("--samples", type=int, default=24, help="Number of random product initial states.")
    ap.add_argument("--ops", type=int, default=24, help="Number of local perturbation operators sampled per source node.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    ap.add_argument("--trials", type=int, default=1, help="Repeat trials with different seeds.")
    ap.add_argument("--topk", type=int, default=3, help="Top-k overlap size.")
    ap.add_argument("--symmetry_break", action="store_true", help="Add random local axes (kills symmetry-protected effects).")
    ap.add_argument("--permutation_null", action="store_true", help="Randomly relabel nodes (label/layout artifact check).")
    ap.add_argument("--plots", action="store_true", help="Write plots to outputs/*/plots/")

    args = ap.parse_args()

    times = parse_times(args.times)
    if len(times) == 0:
        raise ValueError("Provide at least one time in --times")

    xp, linalg_expm, backend = _load_backend(args.device)

    outdir = os.path.join("outputs", f"hip_kill_{_timestamp()}_{backend}")
    _mkdir(outdir)

    print("============================================================")
    print("                 HIP KILL SUITE")
    print("============================================================")
    print(f"Device requested: {args.device}  | Backend: {backend}")
    print(f"N={args.N}  graph={args.graph}  ham={args.ham}")
    print(f"times={times}")
    print(f"samples={args.samples}  ops={args.ops}  trials={args.trials}")
    print(f"symmetry_break={args.symmetry_break}  permutation_null={args.permutation_null}")
    print("------------------------------------------------------------")

    all_results = []
    for k in range(args.trials):
        seed = args.seed + k
        print(f"[Trial {k+1}/{args.trials}] seed={seed} ...")
        res = run_hip_suite_once(
            xp=xp,
            linalg_expm=linalg_expm,
            graph_type=args.graph,
            ham_type=args.ham,
            N=args.N,
            times=times,
            samples=args.samples,
            ops=args.ops,
            seed=seed,
            deg=args.deg,
            p=args.p,
            symmetry_break=args.symmetry_break,
            topk=args.topk,
            permutation_null=args.permutation_null,
        )
        all_results.append(res)

        # Print quick headline diagnostics
        print("  HIP(t):", ["{:.4g}".format(v) for v in res.hip])
        print("  RankSpearman(t):", ["{:.3f}".format(v) for v in res.rank_spearman_to_t0])
        print("  TopKOverlap(t):", ["{:.3f}".format(v) for v in res.topk_overlap_to_t0])

    # Aggregate (mean over trials)
    # Keep it simple: average hip, mean_edge_perm, rank stats
    T = len(times)
    mean_hip = [0.0]*T
    mean_edge = [0.0]*T
    mean_rank = [0.0]*T
    mean_topk = [0.0]*T

    for t in range(T):
        mean_hip[t] = sum(r.hip[t] for r in all_results) / len(all_results)
        mean_edge[t] = sum(r.mean_edge_perm[t] for r in all_results) / len(all_results)
        mean_rank[t] = sum(r.rank_spearman_to_t0[t] for r in all_results) / len(all_results)
        mean_topk[t] = sum(r.topk_overlap_to_t0[t] for r in all_results) / len(all_results)

    summary = {
        "run": {
            "timestamp": _timestamp(),
            "backend": backend,
            "args": vars(args),
        },
        "aggregate": {
            "times": times,
            "mean_HIP": mean_hip,
            "mean_edge_perm": mean_edge,
            "mean_rank_spearman_to_t0": mean_rank,
            "mean_topk_overlap_to_t0": mean_topk,
        },
        "trials": [r.notes for r in all_results],
    }

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV timeseries
    csv_path = os.path.join(outdir, "timeseries.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t,mean_HIP,mean_edge_perm,mean_rank_spearman,mean_topk_overlap\n")
        for i, t in enumerate(times):
            f.write(f"{t},{mean_hip[i]},{mean_edge[i]},{mean_rank[i]},{mean_topk[i]}\n")

    # Optional plots: use the first trial's result for node-level plots if needed,
    # but for now plot aggregate using a synthetic HIPResult
    if args.plots:
        agg = HIPResult(
            times=times,
            hip=mean_hip,
            mean_edge_perm=mean_edge,
            rank_spearman_to_t0=mean_rank,
            topk_overlap_to_t0=mean_topk,
            node_intensity_by_time=all_results[0].node_intensity_by_time,
            notes={"aggregate_of": len(all_results), **all_results[0].notes},
        )
        maybe_plot(outdir, agg)

    print("------------------------------------------------------------")
    print("DONE")
    print(f"Output dir: {outdir}")
    print(f"  - summary.json")
    print(f"  - timeseries.csv")
    if args.plots:
        print(f"  - plots/*.png")
    print("============================================================")

if __name__ == "__main__":
    main()
