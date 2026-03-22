
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


Edge = Tuple[int, int]
LinkRegs = Dict[Edge, np.ndarray]


@dataclass
class PhysicsConfig:
    n_max: int = 8
    n_init: int = 2
    seed: int = 0
    local_scale: float = 0.15
    pair_scale: float = 0.12
    spawn_pair_scale: float = 0.11
    total_steps: int = 80
    dt: float = 0.04
    eval_every: int = 10
    lookahead_windows: int = 1
    weaken_factor: float = 0.55
    demote_edge_threshold: float = 0.25
    progress_every: int = 1
    device: str = "cpu"


def get_array_module(device: str):
    if device == "gpu":
        if cp is None:
            raise RuntimeError("CuPy is not available for --device gpu.")
        return cp, True
    return np, False


def to_numpy(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def canonical_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def gell_mann_matrices():
    """Eight SU(3) generators in physics normalization."""
    z = np.zeros((3, 3), dtype=np.complex128)
    mats = []

    m = z.copy()
    m[0, 1] = m[1, 0] = 1.0
    mats.append(m)

    m = z.copy()
    m[0, 1] = -1j
    m[1, 0] = 1j
    mats.append(m)

    m = z.copy()
    m[0, 0] = 1.0
    m[1, 1] = -1.0
    mats.append(m)

    m = z.copy()
    m[0, 2] = m[2, 0] = 1.0
    mats.append(m)

    m = z.copy()
    m[0, 2] = -1j
    m[2, 0] = 1j
    mats.append(m)

    m = z.copy()
    m[1, 2] = m[2, 1] = 1.0
    mats.append(m)

    m = z.copy()
    m[1, 2] = -1j
    m[2, 1] = 1j
    mats.append(m)

    m = z.copy()
    m[0, 0] = 1.0 / math.sqrt(3.0)
    m[1, 1] = 1.0 / math.sqrt(3.0)
    m[2, 2] = -2.0 / math.sqrt(3.0)
    mats.append(m)

    return [m / 2.0 for m in mats]


GM_MATRICES = gell_mann_matrices()


def random_state(n_sites: int, xp, rng: np.random.Generator):
    shape = (3,) * n_sites
    data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    psi = xp.asarray(data, dtype=xp.complex128)
    psi = psi / xp.linalg.norm(psi)
    return psi


def basis_state_zero(n_sites: int, xp):
    shape = (3,) * n_sites
    psi = xp.zeros(shape, dtype=xp.complex128)
    psi[(0,) * n_sites] = 1.0 + 0.0j
    return psi


def apply_one_body(psi, op, site: int, xp):
    axes = list(range(psi.ndim))
    axes[0], axes[site] = axes[site], axes[0]
    psi_perm = xp.transpose(psi, axes)
    flat = psi_perm.reshape(3, -1)
    out = op @ flat
    out = out.reshape(psi_perm.shape)
    inv = np.argsort(axes)
    return xp.transpose(out, tuple(inv))


def apply_two_body_samegen(psi, left_op, i: int, right_op, j: int, xp):
    if i == j:
        raise ValueError("Sites must differ for a two-body term.")
    axes = list(range(psi.ndim))
    axes[0], axes[i] = axes[i], axes[0]
    axes[1], axes[j] = axes[j], axes[1]
    psi_perm = xp.transpose(psi, axes)
    flat = psi_perm.reshape(9, -1)
    op = xp.asarray(np.kron(to_numpy(left_op), to_numpy(right_op)), dtype=xp.complex128)
    out = op @ flat
    out = out.reshape(psi_perm.shape)
    inv = np.argsort(axes)
    return xp.transpose(out, tuple(inv))


def apply_hamiltonian(
    psi,
    active_nodes: Set[int],
    active_edges: Set[Edge],
    local_coeffs: np.ndarray,
    edge_strengths: Dict[Edge, float],
    xp,
):
    out = xp.zeros_like(psi)
    gm_xp = [xp.asarray(m, dtype=xp.complex128) for m in GM_MATRICES]

    for i in sorted(active_nodes):
        coeff = float(local_coeffs[i])
        if coeff == 0.0:
            continue
        for gm in gm_xp[:3]:
            out = out + coeff * apply_one_body(psi, gm, i, xp)

    for i, j in sorted(active_edges):
        strength = float(edge_strengths.get((i, j), 0.0))
        if strength == 0.0:
            continue
        for gm in gm_xp:
            out = out + strength * apply_two_body_samegen(psi, gm, i, gm, j, xp)
    return out


def rk4_step(
    psi,
    active_nodes: Set[int],
    active_edges: Set[Edge],
    local_coeffs: np.ndarray,
    edge_strengths: Dict[Edge, float],
    dt: float,
    xp,
):
    def f(state):
        return -1j * apply_hamiltonian(state, active_nodes, active_edges, local_coeffs, edge_strengths, xp)

    k1 = f(psi)
    k2 = f(psi + 0.5 * dt * k1)
    k3 = f(psi + 0.5 * dt * k2)
    k4 = f(psi + dt * k3)
    out = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out / xp.linalg.norm(out)


def evolve_windows(
    psi,
    active_nodes: Set[int],
    active_edges: Set[Edge],
    local_coeffs: np.ndarray,
    edge_strengths: Dict[Edge, float],
    cfg: PhysicsConfig,
    xp,
):
    out = psi
    for _ in range(int(cfg.lookahead_windows)):
        out = rk4_step(out, active_nodes, active_edges, local_coeffs, edge_strengths, cfg.dt, xp)
    return out


def default_linkreg():
    return np.eye(3, dtype=np.complex128)


def build_link_regs(active_edges: Iterable[Edge]) -> LinkRegs:
    return {canonical_edge(i, j): default_linkreg().copy() for i, j in active_edges}


def sanitize_graph_state(
    active_nodes: Set[int],
    dormant_nodes: Set[int],
    active_edges: Set[Edge],
    edge_strengths: Dict[Edge, float],
    link_regs: LinkRegs,
):
    bad_edges = []
    for e in list(active_edges):
        i, j = e
        if i not in active_nodes or j not in active_nodes:
            bad_edges.append(e)
            continue
        edge_strengths[e] = float(edge_strengths.get(e, 0.0))
        if edge_strengths[e] <= 0.0:
            bad_edges.append(e)
    for e in bad_edges:
        active_edges.discard(e)
        edge_strengths.pop(e, None)
        link_regs.pop(e, None)

    active_nodes.intersection_update(set(range(max(active_nodes | dormant_nodes) + 1 if active_nodes or dormant_nodes else 0)))
    dormant_nodes.difference_update(active_nodes)
    for e in list(link_regs.keys()):
        if e not in active_edges:
            link_regs.pop(e, None)


def init_state(cfg: PhysicsConfig, xp):
    rng = np.random.default_rng(cfg.seed)
    psi = basis_state_zero(cfg.n_max, xp)
    local_coeffs = np.zeros(cfg.n_max, dtype=np.float64)
    local_coeffs[: cfg.n_init] = cfg.local_scale
    active_nodes = set(range(cfg.n_init))
    dormant_nodes = set(range(cfg.n_init, cfg.n_max))
    active_edges: Set[Edge] = set()
    edge_strengths: Dict[Edge, float] = {}
    for i in range(cfg.n_init):
        for j in range(i + 1, cfg.n_init):
            e = (i, j)
            active_edges.add(e)
            edge_strengths[e] = cfg.pair_scale
    link_regs = build_link_regs(active_edges)
    return (
        psi,
        active_nodes,
        dormant_nodes,
        active_edges,
        local_coeffs,
        edge_strengths,
        link_regs,
        rng,
    )


def clone_graph_state(
    psi,
    active_nodes: Set[int],
    dormant_nodes: Set[int],
    active_edges: Set[Edge],
    local_coeffs: np.ndarray,
    edge_strengths: Dict[Edge, float],
    link_regs: LinkRegs,
):
    return (
        psi.copy(),
        set(active_nodes),
        set(dormant_nodes),
        set(active_edges),
        np.array(local_coeffs, copy=True),
        dict(edge_strengths),
        {k: np.array(v, copy=True) for k, v in link_regs.items()},
    )


def prepare_birth_move(
    state,
    parents: Edge,
    child: int,
    cfg: PhysicsConfig,
):
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = clone_graph_state(*state)
    i, j = parents
    active_nodes.add(child)
    dormant_nodes.discard(child)
    local_coeffs[child] = cfg.spawn_pair_scale

    for a in (i, j):
        e = canonical_edge(a, child)
        active_edges.add(e)
        edge_strengths[e] = max(cfg.spawn_pair_scale, cfg.pair_scale)
        link_regs[e] = default_linkreg().copy()

    sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
    return (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs)


def prepare_weaken_move(
    state,
    edge: Edge,
    cfg: PhysicsConfig,
):
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = clone_graph_state(*state)
    e = canonical_edge(*edge)
    if e not in active_edges:
        return None
    edge_strengths[e] = float(edge_strengths[e]) * float(cfg.weaken_factor)
    if edge_strengths[e] < 1e-12:
        edge_strengths[e] = 0.0
    sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
    return (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs)


def prepare_retire_move(
    state,
    node: int,
    cfg: PhysicsConfig,
):
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = clone_graph_state(*state)
    if node not in active_nodes:
        return None
    active_nodes.discard(node)
    dormant_nodes.add(node)
    local_coeffs[node] = 0.0
    for e in list(active_edges):
        if node in e:
            active_edges.discard(e)
            edge_strengths.pop(e, None)
            link_regs.pop(e, None)
    sanitize_graph_state(active_nodes, dormant_nodes, active_edges, edge_strengths, link_regs)
    return (psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs)


def evolve_prepared_state(prepared_state, cfg: PhysicsConfig, xp):
    psi, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs = prepared_state
    psi_evolved = evolve_windows(
        psi,
        active_nodes,
        active_edges,
        local_coeffs,
        edge_strengths,
        cfg,
        xp,
    )
    return (psi_evolved, active_nodes, dormant_nodes, active_edges, local_coeffs, edge_strengths, link_regs)


def graph_stats(active_nodes: Set[int], active_edges: Set[Edge]):
    deg = {i: 0 for i in active_nodes}
    for i, j in active_edges:
        if i in deg:
            deg[i] += 1
        if j in deg:
            deg[j] += 1
    return {
        "degrees": {int(k): int(v) for k, v in deg.items()},
        "max_degree": int(max(deg.values()) if deg else 0),
        "mean_degree": float(np.mean(list(deg.values())) if deg else 0.0),
    }


def metric_snapshot(active_nodes: Set[int], active_edges: Set[Edge], edge_strengths: Dict[Edge, float]):
    strengths = [float(edge_strengths[e]) for e in active_edges] if active_edges else []
    return {
        "n_active_nodes": int(len(active_nodes)),
        "n_active_edges": int(len(active_edges)),
        "mean_edge_strength": float(np.mean(strengths) if strengths else 0.0),
        "min_edge_strength": float(np.min(strengths) if strengths else 0.0),
        "max_edge_strength": float(np.max(strengths) if strengths else 0.0),
    }
