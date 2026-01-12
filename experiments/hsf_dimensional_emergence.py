#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
HSF Dimensional Emergence Study (Drop-in, Memory-Safe Upgrade)
=============================================================

Drop-in replacement for `hsf_dimensional_emergence.py` with:
- legacy dense backend preserved
- tensor backend (memory-safe) avoids U_full
- Monte Carlo locality metric (memory-safe) avoids dense basis

Fix in this revision:
- MC trace routine uses einsum (rank-stable) to avoid axis shrink bugs.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np


# =============================================================================
# UTILITIES
# =============================================================================

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


# =============================================================================
# GRAPH TOPOLOGIES
# =============================================================================

@dataclass
class GraphTopology:
    name: str
    dimension: Optional[int]  # None for non-geometric graphs
    edges: List[Tuple[int, int]]
    N: int
    metadata: Dict = field(default_factory=dict)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def degree(self) -> float:
        return 2 * self.num_edges / self.N

    def adjacency_dict(self) -> Dict[int, Set[int]]:
        adj = {i: set() for i in range(self.N)}
        for (i, j) in self.edges:
            adj[i].add(j)
            adj[j].add(i)
        return adj


def make_1d_ring(N: int) -> GraphTopology:
    edges = [(i, (i + 1) % N) for i in range(N)]
    return GraphTopology(name="1D_ring", dimension=1, edges=edges, N=N,
                         metadata={"boundary": "periodic", "L": N})


def make_1d_chain(N: int) -> GraphTopology:
    edges = [(i, i + 1) for i in range(N - 1)]
    return GraphTopology(name="1D_chain", dimension=1, edges=edges, N=N,
                         metadata={"boundary": "open", "L": N})


def make_2d_lattice(Lx: int, Ly: int, periodic: bool = True) -> GraphTopology:
    N = Lx * Ly
    edges: List[Tuple[int, int]] = []

    def idx(x, y): return x * Ly + y

    for x in range(Lx):
        for y in range(Ly):
            if periodic or x < Lx - 1:
                nx = (x + 1) % Lx
                edges.append((idx(x, y), idx(nx, y)))
            if periodic or y < Ly - 1:
                ny = (y + 1) % Ly
                edges.append((idx(x, y), idx(x, ny)))

    return GraphTopology(name=f"2D_lattice_{Lx}x{Ly}", dimension=2, edges=edges, N=N,
                         metadata={"boundary": "periodic" if periodic else "open", "Lx": Lx, "Ly": Ly})


def make_3d_lattice(Lx: int, Ly: int, Lz: int, periodic: bool = True) -> GraphTopology:
    N = Lx * Ly * Lz
    edges: List[Tuple[int, int]] = []

    def idx(x, y, z): return x * Ly * Lz + y * Lz + z

    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                if periodic or x < Lx - 1:
                    nx = (x + 1) % Lx
                    edges.append((idx(x, y, z), idx(nx, y, z)))
                if periodic or y < Ly - 1:
                    ny = (y + 1) % Ly
                    edges.append((idx(x, y, z), idx(x, ny, z)))
                if periodic or z < Lz - 1:
                    nz = (z + 1) % Lz
                    edges.append((idx(x, y, z), idx(x, y, nz)))

    return GraphTopology(name=f"3D_lattice_{Lx}x{Ly}x{Lz}", dimension=3, edges=edges, N=N,
                         metadata={"boundary": "periodic" if periodic else "open", "Lx": Lx, "Ly": Ly, "Lz": Lz})


def make_4d_lattice(L: int, periodic: bool = True) -> GraphTopology:
    N = L ** 4
    edges: List[Tuple[int, int]] = []

    def idx(x, y, z, w): return x * L**3 + y * L**2 + z * L + w

    for x in range(L):
        for y in range(L):
            for z in range(L):
                for w in range(L):
                    if periodic or x < L - 1:
                        edges.append((idx(x, y, z, w), idx((x + 1) % L, y, z, w)))
                    if periodic or y < L - 1:
                        edges.append((idx(x, y, z, w), idx(x, (y + 1) % L, z, w)))
                    if periodic or z < L - 1:
                        edges.append((idx(x, y, z, w), idx(x, y, (z + 1) % L, w)))
                    if periodic or w < L - 1:
                        edges.append((idx(x, y, z, w), idx(x, y, z, (w + 1) % L)))

    return GraphTopology(name=f"4D_lattice_{L}x{L}x{L}x{L}", dimension=4, edges=edges, N=N,
                         metadata={"boundary": "periodic" if periodic else "open", "L": L})


def make_complete_graph(N: int) -> GraphTopology:
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    return GraphTopology(name="complete", dimension=None, edges=edges, N=N, metadata={"type": "complete"})


def make_random_regular(N: int, degree: int, rng: np.random.Generator) -> GraphTopology:
    if (N * degree) % 2 != 0:
        raise ValueError("N*degree must be even")

    max_attempts = 1000
    for attempt in range(max_attempts):
        stubs = []
        for node in range(N):
            stubs.extend([node] * degree)
        rng.shuffle(stubs)

        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            e = (min(u, v), max(u, v))
            if e in edges:
                valid = False
                break
            edges.add(e)

        if valid:
            return GraphTopology(name=f"random_regular_d{degree}", dimension=None,
                                 edges=list(edges), N=N, metadata={"degree": degree, "seed_attempt": attempt})

    raise RuntimeError("Failed to generate random regular graph")


def find_2d_shape(N: int) -> Optional[Tuple[int, int]]:
    for Lx in range(2, N + 1):
        if N % Lx == 0:
            Ly = N // Lx
            if Ly >= 2:
                return (Lx, Ly)
    return None


def find_3d_shape(N: int) -> Optional[Tuple[int, int, int]]:
    for Lx in range(2, N + 1):
        if N % Lx != 0:
            continue
        rem = N // Lx
        for Ly in range(2, rem + 1):
            if rem % Ly == 0:
                Lz = rem // Ly
                if Lz >= 2:
                    return (Lx, Ly, Lz)
    return None


def find_4d_L(N: int) -> Optional[int]:
    L = round(N ** 0.25)
    if L >= 2 and L**4 == N:
        return L
    return None


def generate_topologies_for_N(N: int, rng: np.random.Generator) -> List[GraphTopology]:
    topologies: List[GraphTopology] = [make_1d_ring(N), make_1d_chain(N)]

    shape2 = find_2d_shape(N)
    if shape2 is not None:
        topologies.append(make_2d_lattice(*shape2, periodic=True))

    shape3 = find_3d_shape(N)
    if shape3 is not None:
        topologies.append(make_3d_lattice(*shape3, periodic=True))

    L4 = find_4d_L(N)
    if L4 is not None:
        topologies.append(make_4d_lattice(L4, periodic=True))

    if N >= 4:
        try:
            topologies.append(make_random_regular(N, degree=4, rng=rng))
        except Exception:
            pass
    if N >= 6:
        try:
            topologies.append(make_random_regular(N, degree=6, rng=rng))
        except Exception:
            pass

    return topologies


# =============================================================================
# PAULIS / BASIC OPS
# =============================================================================

I2 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PAULI_MAP = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}
PAULI_NONTRIV = ["X", "Y", "Z"]


def hermitian_rand(dim: int, rng: np.random.Generator, dtype=np.complex128) -> np.ndarray:
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = (a + a.conj().T) / 2.0
    return h.astype(dtype, copy=False)


def unitary_from_hermitian(h: np.ndarray, t: float = 1.0) -> np.ndarray:
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * t * w)) @ v.conj().T


def haar_unitary(dim: int, rng: np.random.Generator, dtype=np.complex128) -> np.ndarray:
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))).astype(dtype, copy=False)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.maximum(1e-12, np.abs(d))
    return q * ph


def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def op_on_site_dense(op: np.ndarray, N: int, site: int) -> np.ndarray:
    mats = [I2 if i != site else op for i in range(N)]
    return kron_all(mats)


def two_site_op_dense(opA: np.ndarray, opB: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    mats = []
    for k in range(N):
        if k == i:
            mats.append(opA)
        elif k == j:
            mats.append(opB)
        else:
            mats.append(I2)
    return kron_all(mats)


# =============================================================================
# HAMILTONIAN CONSTRUCTION (dense)
# =============================================================================

def build_xx_hamiltonian_dense(graph: GraphTopology, J: float = 1.0, dtype=np.complex128) -> np.ndarray:
    N = graph.N
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=dtype)
    X = X2.astype(dtype, copy=False)
    Y = Y2.astype(dtype, copy=False)
    for (i, j) in graph.edges:
        H += (J / 2.0) * (two_site_op_dense(X, X, N, i, j) + two_site_op_dense(Y, Y, N, i, j))
    return H


def build_heisenberg_hamiltonian_dense(graph: GraphTopology, J: float = 1.0, dtype=np.complex128) -> np.ndarray:
    N = graph.N
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=dtype)
    X = X2.astype(dtype, copy=False)
    Y = Y2.astype(dtype, copy=False)
    Z = Z2.astype(dtype, copy=False)
    for (i, j) in graph.edges:
        H += J * (two_site_op_dense(X, X, N, i, j) +
                  two_site_op_dense(Y, Y, N, i, j) +
                  two_site_op_dense(Z, Z, N, i, j))
    return H


# =============================================================================
# LOCALITY METRIC: dense exact basis (legacy)
# =============================================================================

@dataclass
class LocalBasisDense:
    ops: List[np.ndarray]
    tags: List[str]


def build_local_basis_dense(graph: GraphTopology, dtype=np.complex128) -> LocalBasisDense:
    N = graph.N
    ops: List[np.ndarray] = []
    tags: List[str] = []

    for i in range(N):
        for name, P in [("X", X2), ("Y", Y2), ("Z", Z2)]:
            ops.append(op_on_site_dense(P.astype(dtype, copy=False), N, i))
            tags.append(f"{name}_{i}")

    for (i, j) in graph.edges:
        for na, Pa in [("X", X2), ("Y", Y2), ("Z", Z2)]:
            for nb, Pb in [("X", X2), ("Y", Y2), ("Z", Z2)]:
                ops.append(two_site_op_dense(Pa.astype(dtype, copy=False), Pb.astype(dtype, copy=False), N, i, j))
                tags.append(f"{na}{nb}_{i}-{j}")

    return LocalBasisDense(ops=ops, tags=tags)


def frob_norm_sq_dense(H: np.ndarray) -> float:
    return float(np.vdot(H, H).real)


def project_to_local_basis_dense(H: np.ndarray, basis: LocalBasisDense, N: int) -> float:
    d = 2 ** N
    acc = 0.0
    for op in basis.ops:
        coef = np.trace(op.conj().T @ H) / d
        acc += float((coef.conjugate() * coef).real) * d
    return acc


def locality_metric_dense_exact(H: np.ndarray, basis: LocalBasisDense, N: int) -> Dict[str, float]:
    total = frob_norm_sq_dense(H)
    local = project_to_local_basis_dense(H, basis, N)
    local_frac = local / (total + 1e-18)
    leak_frac = 1.0 - local_frac
    return {"total_norm_sq": total, "local_norm_sq": local, "local_frac": local_frac, "leak_frac": leak_frac}


# =============================================================================
# TENSOR BACKEND
# =============================================================================

def dense_to_tensor(H: np.ndarray, N: int) -> np.ndarray:
    return H.reshape([2] * N + [2] * N)


def tensor_to_dense(Ht: np.ndarray, N: int) -> np.ndarray:
    return Ht.reshape((2 ** N, 2 ** N))


def frob_norm_sq_tensor(Ht: np.ndarray) -> float:
    return float(np.vdot(Ht, Ht).real)


def apply_two_qubit_conjugation_tensor(Ht: np.ndarray, U2: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    if i == j:
        return Ht
    if i > j:
        i, j = j, i

    # Left multiply on ket indices (i,j)
    ket_axes = [i, j]
    ket_rest = [k for k in range(N) if k not in ket_axes]
    perm1 = ket_axes + ket_rest + list(range(N, 2 * N))
    H1 = np.transpose(Ht, perm1)
    H1 = H1.reshape(4, 2 ** (N - 2), 2 ** N)
    H1m = (U2 @ H1.reshape(4, -1)).reshape(4, 2 ** (N - 2), 2 ** N)
    H1 = H1m.reshape([2, 2] + [2] * (N - 2) + [2] * N)
    inv1 = np.argsort(perm1)
    H1 = np.transpose(H1, inv1)

    # Right multiply on bra indices (i,j): H <- H * U†
    bra_i, bra_j = N + i, N + j
    bra_rest = [k for k in range(N, 2 * N) if k not in (bra_i, bra_j)]
    perm2 = list(range(N)) + bra_rest + [bra_i, bra_j]
    H2 = np.transpose(H1, perm2)
    H2 = H2.reshape(2 ** N, 2 ** (N - 2), 4)
    Udag = U2.conj().T
    H2m = (H2.reshape(-1, 4) @ Udag).reshape(2 ** N, 2 ** (N - 2), 4)
    H2 = H2m.reshape([2] * N + [2] * (N - 2) + [2, 2])
    inv2 = np.argsort(perm2)
    H2 = np.transpose(H2, inv2)

    return H2


# =============================================================================
# LOCALITY METRIC: Monte Carlo (FIXED)
# =============================================================================

def _einsum_labels(n: int) -> List[str]:
    # Need 2N distinct labels; N<=26 gives 52 with lowercase+uppercase
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if n > 26:
        raise ValueError("N too large for this einsum labeling helper; extend labels.")
    return letters[:2 * n]


def trace_pauli_tensor_einsum(Ht: np.ndarray, pauli_per_site: List[np.ndarray], N: int) -> complex:
    """
    Compute Tr(P H)/d without building P, using a stable einsum:
      Tr(PH) = sum_{a,b} H[a,b] * Π_k Pk[bk, ak]
    where H is tensor with axes (a1..aN, b1..bN).
    """
    labels = _einsum_labels(N)
    a = labels[:N]
    b = labels[N:2 * N]

    H_sub = "".join(a + b)

    # Each Pk uses indices (bk, ak)
    P_subs = [f"{b[k]}{a[k]}" for k in range(N)]

    # Build einsum string: "a1..aN b1..bN, b1a1, b2a2, ... ->"
    eq = H_sub + "," + ",".join(P_subs) + "->"

    # Ensure types are consistent
    # NOTE: Pk are 2x2; Ht is 2^(2N) tensor
    val = np.einsum(eq, Ht, *[pauli_per_site[k] for k in range(N)], optimize=True)
    return val / (2 ** N)


def build_local_term_catalog(graph: GraphTopology) -> Tuple[List[Tuple[str, Tuple[int, ...], Tuple[str, ...]]], int]:
    N = graph.N
    terms: List[Tuple[str, Tuple[int, ...], Tuple[str, ...]]] = []
    for i in range(N):
        for P in PAULI_NONTRIV:
            terms.append(("1", (i,), (P,)))
    for (i, j) in graph.edges:
        for Pa in PAULI_NONTRIV:
            for Pb in PAULI_NONTRIV:
                terms.append(("2", (i, j), (Pa, Pb)))
    return terms, len(terms)


def locality_metric_mc(H: np.ndarray,
                       graph: GraphTopology,
                       N: int,
                       rng: np.random.Generator,
                       samples: int = 2048,
                       tensor_mode: bool = False) -> Dict[str, float]:
    d = 2 ** N

    if tensor_mode:
        Ht = H
        total = frob_norm_sq_tensor(Ht)
    else:
        Hmat = H
        total = frob_norm_sq_dense(Hmat)
        Ht = dense_to_tensor(Hmat, N)

    catalog, M = build_local_term_catalog(graph)
    K = min(samples, M)
    idxs = rng.choice(M, size=K, replace=False)

    sum_sq = 0.0
    for idx in idxs:
        typ, sites, Ps = catalog[idx]
        paulis = [I2] * N
        if typ == "1":
            (i,) = sites
            (Pa,) = Ps
            paulis[i] = PAULI_MAP[Pa]
        else:
            i, j = sites
            Pa, Pb = Ps
            paulis[i] = PAULI_MAP[Pa]
            paulis[j] = PAULI_MAP[Pb]

        c = trace_pauli_tensor_einsum(Ht, paulis, N)
        sum_sq += float((c.conjugate() * c).real)

    est_sum_sq = (M / K) * sum_sq
    local = d * est_sum_sq
    local_frac = local / (total + 1e-18)
    leak_frac = 1.0 - local_frac
    return {
        "total_norm_sq": float(total),
        "local_norm_sq": float(local),
        "local_frac": float(local_frac),
        "leak_frac": float(leak_frac),
        "mc_samples": int(K),
        "mc_terms": int(M),
    }


# =============================================================================
# SCRAMBLING
# =============================================================================

def scramble_local_dense(H: np.ndarray, N: int, rng: np.random.Generator, dtype=np.complex128) -> np.ndarray:
    mats = [unitary_from_hermitian(hermitian_rand(2, rng, dtype=dtype)) for _ in range(N)]
    U = kron_all(mats).astype(dtype, copy=False)
    return U @ H @ U.conj().T


def scramble_global_dense(H: np.ndarray, N: int, rng: np.random.Generator, dtype=np.complex128) -> np.ndarray:
    U = haar_unitary(2 ** N, rng, dtype=dtype)
    return U @ H @ U.conj().T


def embed_two_qubit_gate_general_dense(U2: np.ndarray, N: int, i: int, j: int, dtype=np.complex128) -> np.ndarray:
    if i > j:
        i, j = j, i
    dim = 2 ** N
    U_full = np.zeros((dim, dim), dtype=dtype)
    for basis_idx in range(dim):
        bits = [(basis_idx >> k) & 1 for k in range(N)]
        bi, bj = bits[i], bits[j]
        input_2q = bi * 2 + bj
        for output_2q in range(4):
            coef = U2[output_2q, input_2q]
            if abs(coef) < 1e-15:
                continue
            new_bi = (output_2q >> 1) & 1
            new_bj = output_2q & 1
            new_bits = bits.copy()
            new_bits[i] = new_bi
            new_bits[j] = new_bj
            new_idx = sum(b << k for k, b in enumerate(new_bits))
            U_full[new_idx, basis_idx] += coef
    return U_full


def scramble_layer_dense(H: np.ndarray, graph: GraphTopology, rng: np.random.Generator,
                         depth: int, eps: float, dtype=np.complex128) -> np.ndarray:
    N = graph.N
    H_out = H.copy()
    for _ in range(depth):
        for (i, j) in graph.edges:
            U2 = unitary_from_hermitian(hermitian_rand(4, rng, dtype=dtype), t=eps).astype(dtype, copy=False)
            U_full = embed_two_qubit_gate_general_dense(U2, N, i, j, dtype=dtype)
            H_out = U_full @ H_out @ U_full.conj().T
    return H_out


def scramble_layer_tensor(Ht: np.ndarray, graph: GraphTopology, rng: np.random.Generator,
                          depth: int, eps: float, dtype=np.complex128) -> np.ndarray:
    N = graph.N
    H_out = Ht.copy()
    for _ in range(depth):
        for (i, j) in graph.edges:
            U2 = unitary_from_hermitian(hermitian_rand(4, rng, dtype=dtype), t=eps).astype(dtype, copy=False)
            H_out = apply_two_qubit_conjugation_tensor(H_out, U2, N, i, j)
    return H_out


def scramble_local_tensor(Ht: np.ndarray, N: int, rng: np.random.Generator, eps: float, dtype=np.complex128) -> np.ndarray:
    dummy_graph = GraphTopology(name="local_as_layer", dimension=None,
                                edges=[(i, (i + 1) % N) for i in range(N)], N=N)
    return scramble_layer_tensor(Ht, dummy_graph, rng, depth=1, eps=eps, dtype=dtype)


# =============================================================================
# LOCALITY RECOVERY FLOW
# =============================================================================

@dataclass
class FlowParams:
    steps: int = 2000
    eps: float = 0.06
    temp0: float = 0.02
    temp_decay: float = 0.9995
    cost_every: int = 1


def flow_recover_tensor(Ht: np.ndarray, graph: GraphTopology,
                        rng: np.random.Generator,
                        params: FlowParams,
                        mc_samples: int,
                        dtype=np.complex128) -> Tuple[np.ndarray, Dict]:
    N = graph.N
    H_curr = Ht.copy()
    H_best = Ht.copy()

    metrics = locality_metric_mc(H_curr, graph, N, rng, samples=mc_samples, tensor_mode=True)
    cost = metrics["leak_frac"]
    best_cost = cost
    temp = params.temp0

    accepted = 0
    evaluated = 0
    cost_history = [float(cost)]

    gate_edges = graph.edges

    for step in range(params.steps):
        i, j = gate_edges[int(rng.integers(0, len(gate_edges)))]
        U2 = unitary_from_hermitian(hermitian_rand(4, rng, dtype=dtype), t=params.eps).astype(dtype, copy=False)
        H_new = apply_two_qubit_conjugation_tensor(H_curr, U2, N, i, j)

        if step % params.cost_every == 0:
            evaluated += 1
            metrics_new = locality_metric_mc(H_new, graph, N, rng, samples=mc_samples, tensor_mode=True)
            cost_new = metrics_new["leak_frac"]
        else:
            cost_new = cost

        accept = False
        if cost_new <= cost:
            accept = True
        elif temp > 0:
            p = math.exp(-(cost_new - cost) / max(1e-12, temp))
            if rng.random() < p:
                accept = True

        if accept:
            accepted += 1
            H_curr = H_new
            cost = cost_new
            if cost < best_cost:
                best_cost = cost
                H_best = H_curr.copy()

        temp *= params.temp_decay

        if step % 100 == 0:
            cost_history.append(float(cost))

    final_metrics = locality_metric_mc(H_curr, graph, N, rng, samples=mc_samples, tensor_mode=True)
    best_metrics = locality_metric_mc(H_best, graph, N, rng, samples=mc_samples, tensor_mode=True)

    diagnostics = {
        "accepted": int(accepted),
        "evaluated": int(evaluated),
        "final_leak": float(final_metrics["leak_frac"]),
        "final_local": float(final_metrics["local_frac"]),
        "best_leak": float(best_metrics["leak_frac"]),
        "best_local": float(best_metrics["local_frac"]),
        "cost_history": cost_history,
    }
    return H_best, diagnostics


# =============================================================================
# SPECTRAL DIMENSION (graph-only)
# =============================================================================

def measure_spectral_dimension(graph: GraphTopology, t_values: Optional[List[float]] = None) -> Dict:
    N = graph.N
    adj = graph.adjacency_dict()

    L = np.zeros((N, N), dtype=float)
    for i in range(N):
        L[i, i] = len(adj[i])
        for j in adj[i]:
            L[i, j] = -1.0

    eigvals = np.linalg.eigvalsh(L)

    if t_values is None:
        t_values = np.logspace(-3, 3, 80)

    P_t = np.array([float(np.mean(np.exp(-t * eigvals))) for t in t_values], dtype=float)
    log_t = np.log(np.array(t_values, dtype=float))
    log_P = np.log(P_t + 1e-30)

    mid_start = len(t_values) // 3
    mid_end = 2 * len(t_values) // 3
    coeffs = np.polyfit(log_t[mid_start:mid_end], log_P[mid_start:mid_end], 1)
    d_spectral = -2 * coeffs[0]

    return {
        "spectral_dimension": float(d_spectral),
        "laplacian_eigenvalues": eigvals.tolist(),
        "fit_slope": float(coeffs[0]),
        "fit_intercept": float(coeffs[1]),
        "t_fit_window": [float(t_values[mid_start]), float(t_values[mid_end - 1])],
    }


def measure_effective_dimension_stub() -> Dict:
    return {"num_terms": None, "single_site_frac": None, "two_site_frac": None}


# =============================================================================
# EXPERIMENT RUN (tensor-focused; dense kept minimal here)
# =============================================================================

def run_experiment_tensor(graph: GraphTopology,
                          seed: int,
                          scramble_type: str,
                          hamiltonian_type: str,
                          flow_params: FlowParams,
                          mc_samples: int,
                          dtype_name: str,
                          progress: bool = False) -> Dict:
    rng = np.random.default_rng(seed)
    N = graph.N
    dtype = np.complex64 if dtype_name == "complex64" else np.complex128

    # Build baseline Hamiltonian dense then convert (OK at N=12)
    if hamiltonian_type == "heisenberg":
        H0_dense = build_heisenberg_hamiltonian_dense(graph, dtype=dtype)
    else:
        H0_dense = build_xx_hamiltonian_dense(graph, dtype=dtype)
    H0 = dense_to_tensor(H0_dense, N)

    baseline_metrics = locality_metric_mc(H0, graph, N, rng, samples=mc_samples, tensor_mode=True)

    if scramble_type == "global":
        if progress:
            print("  [tensor backend] scramble=global disabled; using scramble=layer instead.")
        scramble_type = "layer"

    if scramble_type == "local":
        Hs = scramble_local_tensor(H0, N, rng, eps=flow_params.eps, dtype=dtype)
    elif scramble_type == "layer":
        Hs = scramble_layer_tensor(H0, graph, rng, depth=N, eps=flow_params.eps, dtype=dtype)
    else:
        raise ValueError(f"Unknown scramble type (tensor backend): {scramble_type}")

    scrambled_metrics = locality_metric_mc(Hs, graph, N, rng, samples=mc_samples, tensor_mode=True)

    Hr, flow_diag = flow_recover_tensor(Hs, graph, rng, flow_params, mc_samples=mc_samples, dtype=dtype)
    recovered_metrics = locality_metric_mc(Hr, graph, N, rng, samples=mc_samples, tensor_mode=True)

    spectral_info = measure_spectral_dimension(graph)
    effective_info = measure_effective_dimension_stub()

    leak_reduction = float(scrambled_metrics["leak_frac"] - recovered_metrics["leak_frac"])
    local_improvement = float(recovered_metrics["local_frac"] - scrambled_metrics["local_frac"])

    return {
        "meta": {
            "created_utc": now_utc_iso(),
            "seed": int(seed),
            "N": int(N),
            "graph_name": graph.name,
            "graph_dimension": graph.dimension,
            "graph_edges": int(len(graph.edges)),
            "graph_degree": float(graph.degree),
            "graph_metadata": graph.metadata,
            "scramble_type": scramble_type,
            "hamiltonian_type": hamiltonian_type,
            "flow_steps": int(flow_params.steps),
            "backend": "tensor",
            "locality_mode": "mc",
            "mc_samples": int(mc_samples),
            "dtype": dtype_name,
        },
        "baseline": {"local_frac": float(baseline_metrics["local_frac"]),
                     "leak_frac": float(baseline_metrics["leak_frac"])},
        "scrambled": {"local_frac": float(scrambled_metrics["local_frac"]),
                      "leak_frac": float(scrambled_metrics["leak_frac"])},
        "recovered": {"local_frac": float(recovered_metrics["local_frac"]),
                      "leak_frac": float(recovered_metrics["leak_frac"]),
                      "best_local": float(flow_diag["best_local"]),
                      "best_leak": float(flow_diag["best_leak"])},
        "flow": {"accepted": int(flow_diag["accepted"]),
                 "evaluated": int(flow_diag["evaluated"]),
                 "acceptance_rate": float(flow_diag["accepted"] / max(1, flow_params.steps))},
        "recovery": {"leak_reduction": float(leak_reduction),
                     "local_improvement": float(local_improvement),
                     "recovered": bool(recovered_metrics["local_frac"] > scrambled_metrics["local_frac"])},
        "spectral": spectral_info,
        "effective": effective_info,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="HSF Dimensional Emergence Study (fixed MC einsum)")

    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--N", type=int, default=8, help="Number of qubits")
    ap.add_argument("--hamiltonian", choices=["xx", "heisenberg"], default="xx")

    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--scrambles", default="layer", help="Comma-separated: local,global,layer")

    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--temp0", type=float, default=0.02)
    ap.add_argument("--temp-decay", type=float, default=0.9995)
    ap.add_argument("--cost-every", type=int, default=10)

    ap.add_argument("--topologies", default="3d",
                    help="Comma-separated topology names, or 'all'. Supports: 1d_ring,1d_chain,2d,3d,4d,rr4,rr6,complete")

    ap.add_argument("--backend", choices=["tensor"], default="tensor",
                    help="This fixed script focuses on the tensor backend for N>=12 sanity checks.")

    ap.add_argument("--mc-samples", type=int, default=4096)
    ap.add_argument("--dtype", choices=["complex128", "complex64"], default="complex64")

    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--zip", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.out).resolve()
    ensure_dir(outdir)
    ensure_dir(outdir / "runs")

    write_text(outdir / "manifest.json", json.dumps({
        "created_utc": now_utc_iso(),
        "tool": "hsf_dimensional_emergence.py",
        "args": vars(args),
    }, indent=2))

    scrambles = [s.strip().lower() for s in args.scrambles.split(",") if s.strip()]
    topo_rng = np.random.default_rng(999)

    topo_names = [t.strip().lower() for t in args.topologies.split(",") if t.strip()]
    topologies: List[GraphTopology] = []

    for name in topo_names:
        if name == "all":
            topologies = generate_topologies_for_N(args.N, topo_rng)
            break
        if name in ("3d", "3d_lattice", "3d_cubic"):
            shape3 = find_3d_shape(args.N)
            if shape3 is None:
                raise ValueError(f"No 3D factorization exists for N={args.N}")
            topologies.append(make_3d_lattice(*shape3, periodic=True))
        elif name in ("2d", "2d_lattice"):
            shape2 = find_2d_shape(args.N)
            if shape2 is None:
                raise ValueError(f"No 2D factorization exists for N={args.N}")
            topologies.append(make_2d_lattice(*shape2, periodic=True))
        elif name in ("1d_ring", "ring"):
            topologies.append(make_1d_ring(args.N))
        elif name in ("1d_chain", "chain"):
            topologies.append(make_1d_chain(args.N))
        elif name in ("rr4",):
            topologies.append(make_random_regular(args.N, degree=4, rng=topo_rng))
        elif name in ("rr6",):
            topologies.append(make_random_regular(args.N, degree=6, rng=topo_rng))
        elif name in ("complete",):
            topologies.append(make_complete_graph(args.N))
        else:
            raise ValueError(f"Unknown topology name: {name}")

    flow_params = FlowParams(
        steps=args.steps,
        eps=args.eps,
        temp0=args.temp0,
        temp_decay=args.temp_decay,
        cost_every=args.cost_every,
    )

    print("=" * 72)
    print("HSF Dimensional Emergence Study (tensor backend, fixed MC einsum)")
    print("=" * 72)
    print(f"N = {args.N} qubits (dim = {2**args.N})")
    print(f"Backend = tensor | Locality = mc | dtype = {args.dtype}")
    print(f"MC samples = {args.mc_samples}")
    print(f"Seeds = {args.seeds} (start={args.seed_start})")
    print(f"Scrambles = {scrambles}")
    print("\nTopologies:")
    for topo in topologies:
        dim_str = f"d={topo.dimension}" if topo.dimension is not None else "non-geometric"
        print(f"  - {topo.name:24s} | edges={topo.num_edges:4d} | {dim_str}")
    print("=" * 72)

    runs_path = outdir / "runs" / "runs.jsonl"
    if runs_path.exists():
        runs_path.unlink()

    all_results: List[Dict] = []
    total_runs = len(topologies) * len(scrambles) * args.seeds
    run_idx = 0
    t0 = time.time()

    for topo in topologies:
        for scramble in scrambles:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                run_idx += 1
                if args.progress:
                    print(f"[{run_idx}/{total_runs}] {topo.name} | {scramble} | seed={seed}")
                try:
                    result = run_experiment_tensor(
                        graph=topo,
                        seed=seed,
                        scramble_type=scramble,
                        hamiltonian_type=args.hamiltonian,
                        flow_params=flow_params,
                        mc_samples=args.mc_samples,
                        dtype_name=args.dtype,
                        progress=args.progress
                    )
                    all_results.append(result)
                    append_jsonl(runs_path, result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

    runtime = time.time() - t0

    summary = {
        "created_utc": now_utc_iso(),
        "total_runs": len(all_results),
        "runtime_sec": runtime,
        "by_topology": {},
        "by_dimension": {}
    }

    for topo in topologies:
        topo_results = [r for r in all_results if r["meta"]["graph_name"] == topo.name]
        if topo_results:
            leak_reductions = [r["recovery"]["leak_reduction"] for r in topo_results]
            recovered_count = sum(1 for r in topo_results if r["recovery"]["recovered"])
            summary["by_topology"][topo.name] = {
                "dimension": topo.dimension,
                "runs": len(topo_results),
                "recovery_rate": recovered_count / len(topo_results),
                "mean_leak_reduction": float(np.mean(leak_reductions)),
                "spectral_dimension": topo_results[0]["spectral"]["spectral_dimension"],
            }

    write_text(outdir / "summary.json", json.dumps(summary, indent=2))

    report_lines = [
        "# HSF Dimensional Emergence Study Report (tensor backend, fixed MC einsum)",
        f"- Created: `{now_utc_iso()}`",
        f"- N = {args.N} qubits",
        f"- Total runs: {len(all_results)}",
        f"- Runtime: {runtime:.1f}s",
        "",
        "## Results by Topology",
        ""
    ]

    for topo_name, data in summary["by_topology"].items():
        report_lines.append(f"### {topo_name} (d={data['dimension']})")
        report_lines.append(f"- Recovery rate: {data['recovery_rate']:.1%}")
        report_lines.append(f"- Mean leak reduction: {data['mean_leak_reduction']:.6f}")
        report_lines.append(f"- Spectral dimension (graph-only estimate): {data['spectral_dimension']:.3f}")
        report_lines.append("")

    write_text(outdir / "REPORT.md", "\n".join(report_lines))

    if args.zip:
        z = zip_folder(outdir)
        print(f"Wrote ZIP: {z}")

    print(f"\nDONE. Results in: {outdir}")
    print(f"Open: {outdir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
