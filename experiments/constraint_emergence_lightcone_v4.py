# constraint_emergence_lightcone_v4.py
# ------------------------------------------------------------
# HSF Constraint Test V4 — Light-Cone / Metric Emergence Probe
#
# Built directly on constraint_emergence_test_v3_scalable.py, but adds:
#   • Per-block first-crossing times t_cross(b)
#   • Light-cone fits: ballistic (t ~ a + d/v) vs diffusive (t ~ a + α d^2)
#   • Fit quality (R^2), estimated v (if ballistic), and a simple verdict
#   • Optional plot output (PNG)
#
# The four constraints tested remain:
#   (1) No-signaling: finite-speed propagation
#   (2) No-forgetting: persistent recoverability
#   (3) No-refolding: maintain structural complexity (via blocking penalties)
#   (4) Finite bandwidth: limited information capacity per block
#
# Usage examples (Windows one-liners):
#   python constraint_emergence_lightcone_v4.py --N 24 --topology ring --bandwidth 1.5 --nt 9 --tmax 3.0 --progress
#   python constraint_emergence_lightcone_v4.py --N 24 --bandwidth-sweep "1.0,1.5,2.0" --nt 9 --tmax 3.0 --plot --progress
#
# Notes:
#   • For N>=20 default is Trotter evolution (memory friendlier).
#   • Partial traces scale with nt * nBlocks; keep nt modest for big N.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Protocol, Any

import numpy as np

# SciPy (sparse Krylov path)
from scipy import sparse
from scipy.sparse import csr_matrix, kron as sparse_kron, eye as sparse_eye
from scipy.sparse.linalg import expm_multiply

# Matplotlib for optional plotting
import matplotlib.pyplot as plt


# ============================================================
# Evolution interface
# ============================================================

class Evolver(Protocol):
    def evolve_state_times(self, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
        ...


# ============================================================
# Sparse Pauli utilities
# ============================================================

def sparse_pauli(p: str) -> csr_matrix:
    if p == "I":
        return sparse_eye(2, dtype=np.complex128, format="csr")
    if p == "X":
        return csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    if p == "Y":
        return csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
    if p == "Z":
        return csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    raise ValueError(f"Unknown Pauli: {p}")


_PAULI_CACHE = {p: sparse_pauli(p) for p in ["I", "X", "Y", "Z"]}


def sparse_kron_n(ops: List[csr_matrix]) -> csr_matrix:
    result = ops[0]
    for op in ops[1:]:
        result = sparse_kron(result, op, format="csr")
    return result


def sparse_pauli_string(N: int, paulis: Dict[int, str]) -> csr_matrix:
    ops: List[csr_matrix] = []
    for i in range(N):
        ops.append(_PAULI_CACHE[paulis.get(i, "I")])
    return sparse_kron_n(ops)


# ============================================================
# Graph constructors
# ============================================================

def build_edges(N: int, topology: str, rng: np.random.Generator, rr_deg: int = 3) -> List[Tuple[int, int]]:
    topology = topology.lower()
    edges: set[Tuple[int, int]] = set()

    if topology == "ring":
        for i in range(N):
            j = (i + 1) % N
            edges.add((min(i, j), max(i, j)))

    elif topology == "line":
        for i in range(N - 1):
            edges.add((i, i + 1))

    elif topology == "complete":
        for i in range(N):
            for j in range(i + 1, N):
                edges.add((i, j))

    elif topology.startswith("rr"):
        deg = rr_deg
        if topology != "rr":
            try:
                deg = int(topology[2:])
            except Exception:
                pass
        if deg >= N or (N * deg) % 2 != 0:
            raise ValueError(f"Invalid random regular: N={N}, deg={deg}")

        for _attempt in range(5000):
            stubs: List[int] = []
            for i in range(N):
                stubs.extend([i] * deg)
            rng.shuffle(stubs)

            edges.clear()
            ok = True
            for k in range(0, len(stubs), 2):
                a, b = stubs[k], stubs[k + 1]
                if a == b:
                    ok = False
                    break
                e = (min(a, b), max(a, b))
                if e in edges:
                    ok = False
                    break
                edges.add(e)

            if ok:
                break
        else:
            raise RuntimeError("Failed to build random regular graph")

    elif topology == "ladder":
        if N % 2 != 0:
            raise ValueError("Ladder requires even N")
        L = N // 2
        for i in range(L - 1):
            edges.add((i, i + 1))
            edges.add((L + i, L + i + 1))
        for i in range(L):
            edges.add((i, L + i))

    elif topology == "grid2d":
        side = int(np.sqrt(N))
        if side * side != N:
            raise ValueError(f"grid2d requires perfect square N, got {N}")
        for r in range(side):
            for c in range(side):
                i = r * side + c
                if c < side - 1:
                    edges.add((i, i + 1))
                if r < side - 1:
                    edges.add((i, i + side))
    else:
        raise ValueError(f"Unknown topology: {topology}")

    return sorted(edges)


def neighbors_from_edges(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    nbrs = [[] for _ in range(n)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def graph_distance_matrix(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    nbrs = neighbors_from_edges(n, edges)
    dist = np.full((n, n), fill_value=10**9, dtype=np.int32)
    for s in range(n):
        dist[s, s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in nbrs[u]:
                if dist[s, v] > dist[s, u] + 1:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
    return dist


# ============================================================
# Hamiltonian builders / Evolvers
# ============================================================

def build_sparse_2local_hamiltonian(
    N: int,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
    onsite_scale: float = 0.25,
    edge_scale: float = 0.8,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> csr_matrix:
    dim = 2**N
    H = csr_matrix((dim, dim), dtype=np.complex128)

    if progress_callback:
        progress_callback(f"Building onsite terms (3×{N})...")
    for i in range(N):
        hx, hy, hz = rng.normal(scale=onsite_scale, size=3)
        if abs(hx) > 1e-12:
            H = H + hx * sparse_pauli_string(N, {i: "X"})
        if abs(hy) > 1e-12:
            H = H + hy * sparse_pauli_string(N, {i: "Y"})
        if abs(hz) > 1e-12:
            H = H + hz * sparse_pauli_string(N, {i: "Z"})

    if progress_callback:
        progress_callback(f"Building edge terms (9×{len(edges)})...")
    paulis = ["X", "Y", "Z"]
    for idx, (i, j) in enumerate(edges):
        if progress_callback and idx % max(1, len(edges) // 10) == 0:
            progress_callback(f"  Edge {idx+1}/{len(edges)}...")
        for Pi in paulis:
            for Pj in paulis:
                J = rng.normal(scale=edge_scale)
                if abs(J) > 1e-12:
                    H = H + J * sparse_pauli_string(N, {i: Pi, j: Pj})

    H = 0.5 * (H + H.conj().T)
    H.eliminate_zeros()

    if progress_callback:
        nnz = H.nnz
        density = nnz / (dim * dim)
        progress_callback(f"H built: {nnz:,} non-zeros ({density:.2e} density)")
    return H


@dataclass
class SparseEvolver:
    N: int
    H: csr_matrix

    def evolve_state_times(self, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
        if len(times) == 0:
            return np.zeros((0, len(psi0)), dtype=np.complex128)

        t_max = float(times[-1])
        if t_max < 1e-14:
            return np.tile(psi0, (len(times), 1))

        # expm_multiply can generate intermediate steps in one shot
        out = expm_multiply(-1j * self.H, psi0, start=0, stop=t_max, num=len(times), endpoint=True)
        return out


class TrotterEvolver:
    def __init__(
        self,
        N: int,
        edges: List[Tuple[int, int]],
        onsite_coeffs: Dict[int, Tuple[float, float, float]],
        edge_coeffs: Dict[Tuple[int, int], np.ndarray],
        dt: float = 0.05,
    ):
        self.N = N
        self.edges = edges
        self.onsite_coeffs = onsite_coeffs
        self.edge_coeffs = edge_coeffs
        self.dt = float(dt)

        self._precompute_gates()

    def _precompute_gates(self) -> None:
        paulis_2 = {
            "I": np.eye(2, dtype=np.complex128),
            "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        labels = ["X", "Y", "Z"]

        # Onsite exact single-qubit rotations
        self.onsite_gates: Dict[int, np.ndarray] = {}
        for q, (hx, hy, hz) in self.onsite_coeffs.items():
            norm = float(np.sqrt(hx * hx + hy * hy + hz * hz))
            if norm <= 1e-12:
                self.onsite_gates[q] = paulis_2["I"]
                continue
            n = np.array([hx, hy, hz], dtype=np.float64) / norm
            angle = norm * self.dt
            self.onsite_gates[q] = (
                np.cos(angle) * paulis_2["I"]
                - 1j * np.sin(angle) * (n[0] * paulis_2["X"] + n[1] * paulis_2["Y"] + n[2] * paulis_2["Z"])
            )

        # Edge gates: exact 4x4 exponentials
        self.edge_gates: Dict[Tuple[int, int], np.ndarray] = {}
        for (i, j), J in self.edge_coeffs.items():
            H_edge = np.zeros((4, 4), dtype=np.complex128)
            for a, Pa in enumerate(labels):
                for b, Pb in enumerate(labels):
                    H_edge += J[a, b] * np.kron(paulis_2[Pa], paulis_2[Pb])

            evals, evecs = np.linalg.eigh(H_edge)
            self.edge_gates[(i, j)] = evecs @ np.diag(np.exp(-1j * evals * self.dt)) @ evecs.conj().T

    def _apply_single_qubit_gate(self, psi: np.ndarray, qubit: int, gate: np.ndarray) -> np.ndarray:
        psi_tensor = psi.reshape([2] * self.N)
        psi_tensor = np.moveaxis(psi_tensor, qubit, -1)
        shape = psi_tensor.shape
        flat = psi_tensor.reshape(-1, 2)
        flat = flat @ gate.T
        psi_tensor = flat.reshape(shape)
        psi_tensor = np.moveaxis(psi_tensor, -1, qubit)
        return psi_tensor.reshape(-1)

    def _apply_two_qubit_gate(self, psi: np.ndarray, q1: int, q2: int, gate: np.ndarray) -> np.ndarray:
        psi_tensor = psi.reshape([2] * self.N)
        axes = list(range(self.N))
        axes.remove(q1)
        axes.remove(q2)
        axes.extend([q1, q2])
        psi_tensor = np.transpose(psi_tensor, axes)
        shape = psi_tensor.shape
        flat = psi_tensor.reshape(-1, 4)
        flat = flat @ gate.T
        psi_tensor = flat.reshape(shape)
        inv = [0] * self.N
        for new_pos, old_pos in enumerate(axes):
            inv[old_pos] = new_pos
        psi_tensor = np.transpose(psi_tensor, inv)
        return psi_tensor.reshape(-1)

    def trotter_step(self, psi: np.ndarray) -> np.ndarray:
        for q, gate in self.onsite_gates.items():
            psi = self._apply_single_qubit_gate(psi, q, gate)
        for (i, j), gate in self.edge_gates.items():
            psi = self._apply_two_qubit_gate(psi, i, j, gate)
        return psi

    def evolve_state_times(self, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
        if len(times) == 0:
            return np.zeros((0, len(psi0)), dtype=np.complex128)
        if float(times[-1]) < 1e-14:
            return np.tile(psi0, (len(times), 1))

        out = [psi0.copy()]
        psi = psi0.copy()
        for k in range(1, len(times)):
            dt_needed = float(times[k] - times[k - 1])
            n_steps = max(1, int(np.ceil(dt_needed / self.dt)))
            for _ in range(n_steps):
                psi = self.trotter_step(psi)
            psi = psi / np.linalg.norm(psi)
            out.append(psi.copy())
        return np.array(out)


def build_trotter_evolver(
    N: int,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
    onsite_scale: float = 0.25,
    edge_scale: float = 0.8,
    dt: float = 0.05,
) -> TrotterEvolver:
    onsite: Dict[int, Tuple[float, float, float]] = {}
    for i in range(N):
        hx, hy, hz = rng.normal(scale=onsite_scale, size=3)
        onsite[i] = (float(hx), float(hy), float(hz))

    edge_coeffs: Dict[Tuple[int, int], np.ndarray] = {}
    for (i, j) in edges:
        edge_coeffs[(i, j)] = rng.normal(scale=edge_scale, size=(3, 3)).astype(np.float64)

    return TrotterEvolver(N, edges, onsite, edge_coeffs, dt=float(dt))


# ============================================================
# State prep / local perturbation
# ============================================================

def random_product_state(N: int, rng: np.random.Generator) -> np.ndarray:
    psi = np.array([1.0], dtype=np.complex128)
    for _ in range(N):
        theta = rng.uniform(0.0, np.pi)
        phi = rng.uniform(0.0, 2 * np.pi)
        q = np.array([np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)], dtype=np.complex128)
        psi = np.kron(psi, q)
    return psi


def apply_local_pauli(psi: np.ndarray, qubit: int, pauli: str, N: int) -> np.ndarray:
    psi_tensor = psi.reshape([2] * N)

    if pauli == "I":
        return psi.copy()

    if pauli == "X":
        psi_tensor = np.swapaxes(psi_tensor, qubit, -1)
        out = psi_tensor.copy()
        out[..., 0], out[..., 1] = psi_tensor[..., 1].copy(), psi_tensor[..., 0].copy()
        out = np.swapaxes(out, qubit, -1)
        return out.reshape(-1)

    if pauli == "Y":
        psi_tensor = np.swapaxes(psi_tensor, qubit, -1)
        out = np.zeros_like(psi_tensor)
        out[..., 0] = -1j * psi_tensor[..., 1]
        out[..., 1] = 1j * psi_tensor[..., 0]
        out = np.swapaxes(out, qubit, -1)
        return out.reshape(-1)

    if pauli == "Z":
        psi_tensor = np.swapaxes(psi_tensor, qubit, -1)
        out = psi_tensor.copy()
        out[..., 1] *= -1
        out = np.swapaxes(out, qubit, -1)
        return out.reshape(-1)

    raise ValueError(f"Unknown Pauli: {pauli}")


# ============================================================
# Partial trace + trace distance (pure-state reduced density)
# ============================================================

def partial_trace_pure_state(psi: np.ndarray, keep: List[int], N: int) -> Optional[np.ndarray]:
    keep = sorted(keep)
    n_keep = len(keep)

    if n_keep == 0:
        return np.array([[np.abs(np.vdot(psi, psi))]], dtype=np.complex128)

    if n_keep == N:
        # sentinel: "full system" (we avoid building 2^N x 2^N density matrices)
        return None

    psi_tensor = psi.reshape([2] * N)
    trace_out = [i for i in range(N) if i not in keep]
    perm = keep + trace_out
    psi_perm = np.transpose(psi_tensor, perm)

    dim_keep = 2**n_keep
    dim_trace = 2 ** (N - n_keep)
    psi_mat = psi_perm.reshape(dim_keep, dim_trace)
    rho = psi_mat @ psi_mat.conj().T
    return rho


def trace_distance_from_rhos(rho: Optional[np.ndarray], sigma: Optional[np.ndarray]) -> float:
    if rho is None or sigma is None:
        return 0.0
    diff = rho - sigma
    eigs = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigs)))


# ============================================================
# Bandwidth + blocking utilities
# ============================================================

def bandwidth_efficiency(block_size: int, bandwidth_capacity: float) -> float:
    if block_size <= 0 or bandwidth_capacity <= 0:
        return 0.0
    return min(float(block_size), float(bandwidth_capacity)) / float(block_size)


def bandwidth_penalty(block_size: int, bandwidth_capacity: float, scale: float = 1.0) -> float:
    if float(block_size) <= float(bandwidth_capacity):
        return 0.0
    return float(scale) * (float(block_size) - float(bandwidth_capacity)) / float(bandwidth_capacity)


def make_contiguous_blocks(N: int, block_size: int) -> List[List[int]]:
    if N % block_size != 0:
        raise ValueError(f"block_size {block_size} does not divide N={N}")
    nB = N // block_size
    return [list(range(b * block_size, (b + 1) * block_size)) for b in range(nB)]


def block_graph_edges(qubit_edges: List[Tuple[int, int]], blocks: List[List[int]]) -> List[Tuple[int, int]]:
    qubit_to_block: Dict[int, int] = {}
    for b_idx, blk in enumerate(blocks):
        for q in blk:
            qubit_to_block[q] = b_idx

    edges: set[Tuple[int, int]] = set()
    for (i, j) in qubit_edges:
        bi = qubit_to_block[i]
        bj = qubit_to_block[j]
        if bi != bj:
            edges.add((min(bi, bj), max(bi, bj)))
    return sorted(edges)


# ============================================================
# Light-cone fitting
# ============================================================

def _r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_lightcone(dist: List[int], t_cross: List[Optional[float]]) -> Dict[str, Any]:
    # Use reached blocks with d>0 and finite t_cross
    xs: List[float] = []
    ys: List[float] = []
    for d, tc in zip(dist, t_cross):
        if d <= 0:
            continue
        if tc is None:
            continue
        if tc <= 0:
            # treat instant as ~0 for fit, but keep it; can harm linearity
            tc = 0.0
        xs.append(float(d))
        ys.append(float(tc))

    if len(xs) < 3:
        return {
            "n_fit": len(xs),
            "ballistic": None,
            "diffusive": None,
            "verdict": "insufficient_data",
        }

    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)

    # Ballistic: y = a + b x  (v = 1/b)
    A = np.vstack([np.ones_like(x), x]).T
    coef_b, *_ = np.linalg.lstsq(A, y, rcond=None)
    a_lin, b_lin = float(coef_b[0]), float(coef_b[1])
    yhat_lin = a_lin + b_lin * x
    r2_lin = _r2_score(y, yhat_lin)

    v_est = None
    if b_lin > 1e-12:
        v_est = float(1.0 / b_lin)

    # Diffusive: y = a + c x^2
    x2 = x * x
    A2 = np.vstack([np.ones_like(x2), x2]).T
    coef_d, *_ = np.linalg.lstsq(A2, y, rcond=None)
    a_dif, c_dif = float(coef_d[0]), float(coef_d[1])
    yhat_dif = a_dif + c_dif * x2
    r2_dif = _r2_score(y, yhat_dif)

    verdict = "ballistic" if r2_lin >= r2_dif else "diffusive"

    return {
        "n_fit": int(len(x)),
        "ballistic": {
            "a": a_lin,
            "b": b_lin,
            "v": v_est,
            "r2": r2_lin,
        },
        "diffusive": {
            "a": a_dif,
            "c": c_dif,
            "r2": r2_dif,
        },
        "verdict": verdict,
        "data": {
            "d": xs,
            "t_cross": ys,
        },
    }


def save_lightcone_plot(
    out_png: str,
    fit: Dict[str, Any],
    title: str,
) -> None:
    if fit.get("n_fit", 0) < 2 or fit.get("data") is None:
        return

    d = np.array(fit["data"]["d"], dtype=np.float64)
    t = np.array(fit["data"]["t_cross"], dtype=np.float64)

    plt.figure()
    plt.scatter(d, t)
    d_grid = np.linspace(float(np.min(d)), float(np.max(d)), 200)

    bal = fit.get("ballistic")
    dif = fit.get("diffusive")

    if bal is not None:
        a = float(bal["a"])
        b = float(bal["b"])
        plt.plot(d_grid, a + b * d_grid, linewidth=1)

    if dif is not None:
        a = float(dif["a"])
        c = float(dif["c"])
        plt.plot(d_grid, a + c * (d_grid ** 2), linewidth=1)

    plt.xlabel("graph distance d")
    plt.ylabel("first crossing time t_cross")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ============================================================
# Core evaluation (constraints + light-cone extraction)
# ============================================================

def evaluate_blocking_v4(
    N: int,
    qubit_edges: List[Tuple[int, int]],
    evolver: Evolver,
    psi_base: np.ndarray,
    source_qubit: int,
    times: np.ndarray,
    block_size: int,
    speed_threshold: float = 0.02,
    recover_threshold: float = 0.03,
    d_remote_min: int = 2,
    t_late_frac: float = 0.4,
    min_blocks: int = 4,
    bandwidth_capacity: float = 1.5,
    bandwidth_penalty_scale: float = 1.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    blocks = make_contiguous_blocks(N, block_size)
    nB = len(blocks)

    bedges = block_graph_edges(qubit_edges, blocks)
    bdist = graph_distance_matrix(nB, bedges) if nB > 1 else np.zeros((1, 1), dtype=np.int32)

    source_block = source_qubit // block_size
    dist_from_source = [int(bdist[source_block, b]) for b in range(nB)]

    bw_eff = bandwidth_efficiency(block_size, bandwidth_capacity)
    bw_pen = bandwidth_penalty(block_size, bandwidth_capacity, bandwidth_penalty_scale)

    # Perturbation
    rng_pert = np.random.default_rng(12345)
    pauli_choice = ["X", "Y", "Z"][int(rng_pert.integers(0, 3))]
    psi_pert = apply_local_pauli(psi_base, source_qubit, pauli_choice, N)

    if progress_callback:
        progress_callback(f"    Evolving base state to {len(times)} times...")
    psi_base_t = evolver.evolve_state_times(psi_base, times)

    if progress_callback:
        progress_callback(f"    Evolving perturbed state...")
    psi_pert_t = evolver.evolve_state_times(psi_pert, times)

    if progress_callback:
        progress_callback(f"    Computing {len(times)}×{nB} partial traces...")

    nt = len(times)
    infl = np.zeros((nt, nB), dtype=np.float64)
    for ti in range(nt):
        for b in range(nB):
            rho_a = partial_trace_pure_state(psi_base_t[ti], blocks[b], N)
            rho_b = partial_trace_pure_state(psi_pert_t[ti], blocks[b], N)
            infl[ti, b] = trace_distance_from_rhos(rho_a, rho_b)

    # Free big arrays
    del psi_base_t, psi_pert_t
    gc.collect()

    # No-signaling: first crossing times per block
    t_cross: List[Optional[float]] = [None] * nB
    for b in range(nB):
        for ti, t in enumerate(times):
            if infl[ti, b] >= speed_threshold:
                t_cross[b] = float(t)
                break

    # v_eff estimate
    v_candidates: List[float] = []
    instant_reach = 0
    reached = 0
    for b in range(nB):
        d = dist_from_source[b]
        if d <= 0:
            continue
        tc = t_cross[b]
        if tc is None:
            continue
        reached += 1
        if tc <= 1e-12:
            instant_reach += 1
        else:
            v_candidates.append(d / tc)

    v_eff = float(max(v_candidates)) if v_candidates else 0.0
    frac_reached = float(reached / max(1, nB - 1))

    # No-forgetting: remote recoverability
    remote_blocks = [b for b in range(nB) if b != source_block and dist_from_source[b] >= d_remote_min]
    if not remote_blocks:
        remote_best_raw = np.zeros(nt, dtype=np.float64)
    else:
        remote_best_raw = infl[:, remote_blocks].max(axis=1)

    remote_best_eff = remote_best_raw * bw_eff

    t_late_start = float(times[0] + t_late_frac * (times[-1] - times[0]))
    late_mask = times >= t_late_start

    if np.any(late_mask):
        frac_recover_raw = float(np.mean(remote_best_raw[late_mask] >= recover_threshold))
        mean_recover_raw = float(np.mean(remote_best_raw[late_mask]))
        frac_recover_eff = float(np.mean(remote_best_eff[late_mask] >= recover_threshold))
        mean_recover_eff = float(np.mean(remote_best_eff[late_mask]))
    else:
        frac_recover_raw = mean_recover_raw = 0.0
        frac_recover_eff = mean_recover_eff = 0.0

    # Penalties
    hard_penalty = 0.0
    if nB < min_blocks:
        hard_penalty += 2.0
    if nB > 1 and len(bedges) == 0:
        hard_penalty += 2.0

    soft_penalty = 0.0
    soft_penalty += 0.75 * float(instant_reach)
    soft_penalty += 0.50 * max(0.0, (v_eff - 20.0) / 20.0)
    soft_penalty += bw_pen

    # Score (same structure as v3 scalable)
    score = (
        2.0 * frac_recover_eff
        + 0.8 * math.tanh(5.0 * mean_recover_eff)
        + 0.5 * frac_reached
        - 1.0 * hard_penalty
        - 1.0 * soft_penalty
    )

    # Light-cone fit on block graph distances
    lc_fit = fit_lightcone(dist_from_source, t_cross)

    return {
        "block_size": int(block_size),
        "n_blocks": int(nB),
        "block_edges": int(len(bedges)),
        "bandwidth_capacity": float(bandwidth_capacity),
        "bandwidth_efficiency": float(bw_eff),
        "bandwidth_penalty": float(bw_pen),
        "frac_remote_recover_late_RAW": float(frac_recover_raw),
        "remote_recover_mean_late_RAW": float(mean_recover_raw),
        "frac_remote_recover_late_EFF": float(frac_recover_eff),
        "remote_recover_mean_late_EFF": float(mean_recover_eff),
        "frac_blocks_reached": float(frac_reached),
        "v_eff": float(v_eff),
        "instant_reach_blocks": int(instant_reach),
        "hard_penalty": float(hard_penalty),
        "soft_penalty": float(soft_penalty),
        "score": float(score),
        "lightcone": lc_fit,
        # Optional raw diagnostic arrays (compact)
        "t_cross_by_block": [None if tc is None else float(tc) for tc in t_cross],
        "dist_from_source": [int(d) for d in dist_from_source],
    }


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="HSF V4: Constraints + Light-Cone Fit")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--topology", type=str, default="ring", help="ring|line|ladder|grid2d|complete|rr|rr4...")
    ap.add_argument("--rr-deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--tmax", type=float, default=6.0)
    ap.add_argument("--nt", type=int, default=61)

    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--onsite-scale", type=float, default=0.25)
    ap.add_argument("--edge-scale", type=float, default=0.8)

    ap.add_argument("--blockings", type=str, default=None, help="Comma-separated block sizes, e.g. '1,2,4,8'. Default: factors of N")
    ap.add_argument("--speed-threshold", type=float, default=0.02)
    ap.add_argument("--recover-threshold", type=float, default=0.03)
    ap.add_argument("--d-remote-min", type=int, default=2)
    ap.add_argument("--t-late-frac", type=float, default=0.4)
    ap.add_argument("--min-blocks", type=int, default=4)

    ap.add_argument("--bandwidth", type=float, default=1.5)
    ap.add_argument("--bandwidth-penalty-scale", type=float, default=1.0)
    ap.add_argument("--bandwidth-sweep", type=str, default=None)

    ap.add_argument("--method", type=str, default="auto", choices=["auto", "krylov", "trotter"])
    ap.add_argument("--trotter-dt", type=float, default=0.05)

    ap.add_argument("--out", type=str, default="results_v4_lightcone.json")
    ap.add_argument("--plot", action="store_true", help="Write light-cone PNG(s) alongside JSON")
    ap.add_argument("--plot-dir", type=str, default=None, help="Directory for plots (default: alongside --out)")
    ap.add_argument("--progress", action="store_true")

    args = ap.parse_args()
    N = int(args.N)

    def log(msg: str) -> None:
        if args.progress:
            print(msg, flush=True)

    log("=" * 70)
    log(f"HSF CONSTRAINT TEST V4 — LIGHTCONE (N={N}, dim=2^{N}={2**N:,})")
    log("=" * 70)

    mem_state_mb = (2**N * 16) / (1024**2)
    log(f"State vector memory: {mem_state_mb:.1f} MB")
    if N > 26:
        log("WARNING: N>26 may exceed available memory!")

    rng = np.random.default_rng(int(args.seed))

    log(f"\nBuilding {args.topology} graph...")
    t0 = time.time()
    edges = build_edges(N, args.topology, rng, rr_deg=int(args.rr_deg))
    log(f"  {len(edges)} edges, built in {time.time()-t0:.2f}s")

    # Decide method
    use_trotter = (args.method == "trotter") or (args.method == "auto" and N >= 20)

    evolver: Evolver
    if use_trotter:
        log(f"\nUsing TROTTER evolution (dt={args.trotter_dt})")
        t0 = time.time()
        rng_evolver = np.random.default_rng(int(args.seed) + 1000)
        evolver = build_trotter_evolver(
            N=N,
            edges=edges,
            rng=rng_evolver,
            onsite_scale=float(args.onsite_scale),
            edge_scale=float(args.edge_scale),
            dt=float(args.trotter_dt),
        )
        log(f"  Trotter evolver built in {time.time()-t0:.2f}s")
        log(f"  {len(evolver.onsite_gates)} onsite gates, {len(evolver.edge_gates)} edge gates")
    else:
        log(f"\nUsing KRYLOV evolution (expm_multiply)")
        log("Building sparse Hamiltonian...")
        t0 = time.time()
        H = build_sparse_2local_hamiltonian(
            N=N,
            edges=edges,
            rng=rng,
            onsite_scale=float(args.onsite_scale),
            edge_scale=float(args.edge_scale),
            progress_callback=log if args.progress else None,
        )
        log(f"  Built in {time.time()-t0:.1f}s")
        log(f"  H shape: {H.shape}, nnz: {H.nnz:,}")
        mem_H_mb = (H.data.nbytes + H.indices.nbytes + H.indptr.nbytes) / (1024**2)
        log(f"  H memory: {mem_H_mb:.1f} MB")
        evolver = SparseEvolver(N=N, H=H)

    log("\nGenerating initial state...")
    psi_base = random_product_state(N, rng)

    times = np.linspace(0.0, float(args.tmax), int(args.nt))

    # Block sizes
    if args.blockings:
        blocking_sizes = [int(x.strip()) for x in args.blockings.split(",") if x.strip()]
    else:
        blocking_sizes = [b for b in range(1, N + 1) if N % b == 0]
    blocking_sizes = sorted(set(blocking_sizes))

    log(f"\nBlockings to test: {blocking_sizes}")
    log(f"Times: {len(times)} points in [0, {args.tmax}]")

    # Bandwidth values
    if args.bandwidth_sweep:
        bandwidth_values = [float(x.strip()) for x in args.bandwidth_sweep.split(",") if x.strip()]
    else:
        bandwidth_values = [float(args.bandwidth)]

    all_results: List[Dict[str, Any]] = []

    # Plot directory setup
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path) if os.path.dirname(out_path) else os.getcwd()
    plot_dir = args.plot_dir if args.plot_dir else out_dir
    if args.plot and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    for bw in bandwidth_values:
        log(f"\n{'='*50}")
        log(f"BANDWIDTH = {bw}")
        log(f"{'='*50}")

        res_bw: Dict[str, Any] = {
            "meta": {
                "N": N,
                "dim": 2**N,
                "topology": args.topology,
                "n_edges": len(edges),
                "seed": int(args.seed),
                "tmax": float(args.tmax),
                "nt": int(args.nt),
                "bandwidth_capacity": float(bw),
                "speed_threshold": float(args.speed_threshold),
                "recover_threshold": float(args.recover_threshold),
                "lightcone_models": ["ballistic", "diffusive"],
            },
            "blockings": [],
        }

        for b in blocking_sizes:
            eff = bandwidth_efficiency(int(b), float(bw))
            log(f"\n  block_size={b} (n_blocks={N//b}, efficiency={eff:.2f})")

            t0 = time.time()
            out = evaluate_blocking_v4(
                N=N,
                qubit_edges=edges,
                evolver=evolver,
                psi_base=psi_base,
                source_qubit=int(args.source),
                times=times,
                block_size=int(b),
                speed_threshold=float(args.speed_threshold),
                recover_threshold=float(args.recover_threshold),
                d_remote_min=int(args.d_remote_min),
                t_late_frac=float(args.t_late_frac),
                min_blocks=int(args.min_blocks),
                bandwidth_capacity=float(bw),
                bandwidth_penalty_scale=float(args.bandwidth_penalty_scale),
                progress_callback=log if args.progress else None,
            )
            elapsed = time.time() - t0

            lc = out.get("lightcone", {})
            verdict = lc.get("verdict", "n/a")
            bal = lc.get("ballistic")
            v_str = "n/a"
            r2_str = "n/a"
            if bal is not None:
                v_str = f"{bal.get('v', None):.3g}" if bal.get("v", None) is not None else "n/a"
                r2_str = f"{bal.get('r2', 0.0):.3f}"

            log(f"    score={out['score']:+.3f} (took {elapsed:.1f}s)")
            log(f"    RAW recover={out['frac_remote_recover_late_RAW']:.3f} | EFF recover={out['frac_remote_recover_late_EFF']:.3f}")
            log(f"    Lightcone verdict={verdict} | ballistic v~{v_str} | R2={r2_str}")

            # Optional plot per blocking
            if args.plot:
                safe_topo = str(args.topology).replace("/", "_")
                png_name = f"lightcone_N{N}_{safe_topo}_bw{bw:g}_b{b}.png"
                png_path = os.path.join(plot_dir, png_name)
                title = f"N={N} topo={args.topology} bw={bw:g} block={b} verdict={verdict}"
                save_lightcone_plot(png_path, lc, title=title)
                out["lightcone_plot"] = png_path

            res_bw["blockings"].append(out)

        # Winner
        res_bw["blockings"] = sorted(res_bw["blockings"], key=lambda x: x["score"], reverse=True)
        res_bw["winner"] = res_bw["blockings"][0]
        all_results.append(res_bw)

    # Final JSON object
    if len(all_results) == 1:
        final_obj = all_results[0]
    else:
        final_obj = {
            "sweep_type": "bandwidth",
            "bandwidth_values": bandwidth_values,
            "results_by_bandwidth": all_results,
        }

    with open(args.out, "w") as f:
        json.dump(final_obj, f, indent=2)

    # Summary print
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    for res in all_results:
        bw = res["meta"]["bandwidth_capacity"]
        winner = res["winner"]
        lc = winner.get("lightcone", {})
        verdict = lc.get("verdict", "n/a")
        bal = lc.get("ballistic")
        vtxt = "n/a"
        r2txt = "n/a"
        if bal is not None:
            if bal.get("v", None) is not None:
                vtxt = f"{bal['v']:.4g}"
            r2txt = f"{bal.get('r2', 0.0):.3f}"

        log(f"\nBandwidth={bw}: winner block_size={winner['block_size']} score={winner['score']:+.3f}")
        log(f"  Lightcone: verdict={verdict} | ballistic v~{vtxt} | R2={r2txt}")
        log(f"  Remote recover (EFF late): frac={winner['frac_remote_recover_late_EFF']:.3f} mean={winner['remote_recover_mean_late_EFF']:.4f}")

    log(f"\nWrote: {args.out}")
    if args.plot:
        log(f"Plots in: {plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
