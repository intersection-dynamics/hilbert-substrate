#!/usr/bin/env python3
# filename: hsf_gs_v4.py

"""
HSF graded-support sandbox rewrite (v4)

Purpose
-------
This is a full rewrite of the graded-support sandbox/front-end layer for the
three-layer mesoscape stack:

    sandbox -> bookkeeping -> physics_core

The rewrite is built around the two high-priority fixes identified in the
March 22, 2026 handoff:

1. link-register state must persist across materialization/rematerialization,
2. the wavefunction must evolve continuously between evaluation windows.

Design notes
------------
This file intentionally keeps the sandbox/front-end role narrow.
It owns:
    - graded support state: sigma[i]
    - graded interface state: interface_commitment[(i,j)]
    - persistent link-register memory in graded state
    - candidate generation / branch application at the graded level
    - background evolution scheduling
    - compact JSON logging

It does NOT attempt to replace your physics core or bookkeeping layer.
Instead it adapts to them through a small set of wrapper functions gathered in:

    PhysicsAdapter
    BookkeepingAdapter

Those wrappers are the only spots you should need to tune if your existing
module function names differ.

What this rewrite changes
-------------------------
- materialization no longer recreates link_regs from scratch
- baseline state evolves for eval_every background steps even if no move wins
- candidate moves are scored against that evolved baseline, not a frozen state
- compact JSON is default
- optional richer Odiff logging is available in a size-controlled form

What this rewrite does NOT do yet
---------------------------------
- it still uses single-winner-per-eval-window acceptance
- it does not integrate proto-child into this sandbox
- it does not redesign your expression functional

Those were explicitly lower priority than persistent memory and continuous
background evolution.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("This script requires numpy.") from exc


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


Edge = Tuple[int, int]


def canonical_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def complex_to_jsonable(z: complex) -> List[float]:
    return [float(np.real(z)), float(np.imag(z))]


def vector_to_jsonable(vec: np.ndarray) -> List[List[float]]:
    arr = np.asarray(vec).reshape(-1)
    return [complex_to_jsonable(complex(v)) for v in arr]


def normalize_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128)
    norm = np.linalg.norm(psi)
    if norm <= 0.0:
        return psi.copy()
    return psi / norm


# -----------------------------------------------------------------------------
# Dataclasses for persistent graded state
# -----------------------------------------------------------------------------


@dataclass
class LinkRegisterMemory:
    """
    Persistent per-edge memory carried by the graded sandbox state.

    This is the key structural fix for the handoff diagnosis:
    link-register history must not be discarded every time the current graded
    commitments are materialized into a physics state.

    The payload is intentionally generic so it can carry whichever fields your
    physics/bookkeeping layers actually use.
    """

    amp: complex = 0.0 + 0.0j
    phase: float = 0.0
    age: int = 0
    active_steps: int = 0
    last_on_step: Optional[int] = None
    last_off_step: Optional[int] = None
    cumulative_flux: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "amp": complex_to_jsonable(self.amp),
            "phase": float(self.phase),
            "age": int(self.age),
            "active_steps": int(self.active_steps),
            "last_on_step": self.last_on_step,
            "last_off_step": self.last_off_step,
            "cumulative_flux": float(self.cumulative_flux),
            "metadata": self.metadata,
        }


@dataclass
class MaterializedState:
    """
    Concrete physics-layer state derived from the current graded commitments.

    This object is cached on the sandbox side and updated over time.
    """

    psi: np.ndarray
    active_nodes: List[int]
    active_edges: List[Edge]
    graph_payload: Dict[str, Any]
    physics_payload: Dict[str, Any]


@dataclass
class GradedState:
    n_nodes: int
    sigma: np.ndarray
    interface_commitment: Dict[Edge, float]
    link_memory: Dict[Edge, LinkRegisterMemory]
    materialized: MaterializedState
    current_step: int = 0
    accepted_moves: int = 0
    rng_seed: int = 0
    run_metadata: Dict[str, Any] = field(default_factory=dict)

    def copy_deep(self) -> "GradedState":
        return copy.deepcopy(self)


# -----------------------------------------------------------------------------
# Adapter layer to existing modules
# -----------------------------------------------------------------------------


class PhysicsAdapter:
    """
    Thin compatibility layer for hsf_mesoscale_physics_core.py.

    You should only need to edit this wrapper if your actual core function
    names differ.
    """

    def __init__(self, module_name: str = "hsf_mesoscale_physics_core") -> None:
        self.module = importlib.import_module(module_name)

    def build_graph_payload(
        self,
        *,
        n_nodes: int,
        active_nodes: Sequence[int],
        active_edges: Sequence[Edge],
        sigma: np.ndarray,
        interface_commitment: Mapping[Edge, float],
        link_memory: Mapping[Edge, LinkRegisterMemory],
        args: argparse.Namespace,
    ) -> Dict[str, Any]:
        """
        Build the graph/hamiltonian-facing payload.

        Default behavior is generic and self-contained, but if your physics core
        exposes a graph constructor you can swap it in here.
        """
        return {
            "n_nodes": int(n_nodes),
            "active_nodes": list(map(int, active_nodes)),
            "active_edges": [list(map(int, e)) for e in active_edges],
            "sigma": [float(x) for x in sigma.tolist()],
            "interface_commitment": {
                f"{i}-{j}": float(w) for (i, j), w in interface_commitment.items()
            },
            "link_regs": {
                f"{i}-{j}": mem.to_json() for (i, j), mem in link_memory.items()
            },
        }

    def initialize_psi(
        self,
        *,
        graph_payload: Mapping[str, Any],
        args: argparse.Namespace,
        prior_psi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Initialize or project a state into the current materialized support.

        Priority:
        1. use a physics-core helper if available,
        2. otherwise keep prior_psi when compatible,
        3. otherwise create a uniform normalized state over a fallback dimension.
        """
        if hasattr(self.module, "initialize_state"):
            try:
                psi = self.module.initialize_state(graph_payload=graph_payload, args=args, prior_psi=prior_psi)
                return normalize_state(np.asarray(psi, dtype=np.complex128))
            except TypeError:
                pass

        if prior_psi is not None and len(prior_psi) > 0:
            return normalize_state(np.asarray(prior_psi, dtype=np.complex128))

        dim = int(getattr(args, "fallback_dim", 16))
        psi = np.ones(dim, dtype=np.complex128)
        return normalize_state(psi)

    def build_hamiltonian(
        self,
        *,
        graph_payload: Mapping[str, Any],
        args: argparse.Namespace,
    ) -> Any:
        if hasattr(self.module, "build_hamiltonian"):
            return self.module.build_hamiltonian(graph_payload=graph_payload, args=args)
        return None

    def evolve_steps(
        self,
        *,
        psi: np.ndarray,
        graph_payload: Mapping[str, Any],
        args: argparse.Namespace,
        n_steps: int,
    ) -> np.ndarray:
        """
        Continuous background evolution under the current materialized state.

        If the physics core has a native evolution routine, use it.
        Otherwise use a tiny deterministic phase rotation fallback so the
        sandbox remains runnable.
        """
        if n_steps <= 0:
            return normalize_state(psi)

        if hasattr(self.module, "evolve_state"):
            try:
                out = self.module.evolve_state(
                    psi=psi,
                    graph_payload=graph_payload,
                    args=args,
                    n_steps=n_steps,
                )
                return normalize_state(np.asarray(out, dtype=np.complex128))
            except TypeError:
                pass

        dt = safe_float(getattr(args, "dt", 0.05), 0.05)
        dim = len(psi)
        phase_grid = np.exp(-1j * dt * np.arange(dim, dtype=np.float64))
        out = np.asarray(psi, dtype=np.complex128).copy()
        for _ in range(n_steps):
            out = out * phase_grid
            out = normalize_state(out)
        return out

    def update_link_memory_from_materialized(
        self,
        *,
        link_memory: MutableMapping[Edge, LinkRegisterMemory],
        materialized: MaterializedState,
        active_edges: Sequence[Edge],
        step: int,
    ) -> None:
        """
        Update persistent edge memory from the current evolved materialized state.

        The fallback is deliberately simple. If your physics core can expose richer
        edge-register observables, wire them in here.
        """
        active_set = {canonical_edge(*e) for e in active_edges}
        psi = np.asarray(materialized.psi, dtype=np.complex128)
        mean_amp = complex(np.mean(psi)) if psi.size else 0.0 + 0.0j
        mean_phase = float(np.angle(mean_amp)) if abs(mean_amp) > 0 else 0.0
        mean_flux = float(np.mean(np.abs(psi) ** 2)) if psi.size else 0.0

        for edge, mem in link_memory.items():
            mem.age += 1
            if edge in active_set:
                mem.active_steps += 1
                mem.last_on_step = step
                mem.amp = mean_amp
                mem.phase = mean_phase
                mem.cumulative_flux += mean_flux
            else:
                mem.last_off_step = step


class BookkeepingAdapter:
    """
    Thin compatibility layer for hsf_mesoscale_bookkeeping.py.

    The rewrite assumes bookkeeping owns candidate scoring and diagnostics.
    If your function names differ, tune the wrappers here.
    """

    def __init__(self, module_name: str = "hsf_mesoscale_bookkeeping") -> None:
        self.module = importlib.import_module(module_name)

    def score_branch(
        self,
        *,
        baseline_state: GradedState,
        trial_state: GradedState,
        move: Mapping[str, Any],
        args: argparse.Namespace,
    ) -> Dict[str, Any]:
        """
        Return a score dict with at least:
            total_score
            dE_expr_raw
            dCB
            dCR
            dCS
            dCF

        If the bookkeeping module exposes a richer scorer, use it.
        Otherwise provide a sandbox-compatible fallback.
        """
        if hasattr(self.module, "score_graded_move"):
            try:
                out = self.module.score_graded_move(
                    baseline_state=baseline_state,
                    trial_state=trial_state,
                    move=move,
                    args=args,
                )
                return dict(out)
            except TypeError:
                pass

        # Fallback local branch score.
        base_sigma = baseline_state.sigma
        trial_sigma = trial_state.sigma
        expr_gain = float(np.sum(trial_sigma) - np.sum(base_sigma))

        base_iface = sum(baseline_state.interface_commitment.values())
        trial_iface = sum(trial_state.interface_commitment.values())
        iface_gain = float(trial_iface - base_iface)

        bw_penalty = 0.0
        for edge, mem in trial_state.link_memory.items():
            prev = baseline_state.link_memory.get(edge)
            prev_flux = 0.0 if prev is None else float(prev.cumulative_flux)
            bw_penalty += abs(float(mem.cumulative_flux) - prev_flux)

        lam_b = safe_float(getattr(args, "lambda_B", 0.05), 0.05)
        lam_r = safe_float(getattr(args, "lambda_R", 0.05), 0.05)
        lam_s = safe_float(getattr(args, "lambda_S", 0.05), 0.05)
        lam_f = safe_float(getattr(args, "lambda_F", 0.05), 0.05)

        dCB = float(bw_penalty)
        dCR = float(max(0.0, -iface_gain))
        dCS = float(max(0.0, abs(iface_gain) - 0.5))
        dCF = float(max(0.0, -expr_gain))
        total = float(expr_gain + 0.25 * iface_gain - lam_b * dCB - lam_r * dCR - lam_s * dCS - lam_f * dCF)

        return {
            "total_score": total,
            "dE_expr_raw": expr_gain,
            "dCB": dCB,
            "dCR": dCR,
            "dCS": dCS,
            "dCF": dCF,
            "fallback_iface_gain": iface_gain,
        }

    def odiff_diagnostics(
        self,
        *,
        baseline_state: GradedState,
        trial_state: GradedState,
        move: Mapping[str, Any],
        args: argparse.Namespace,
    ) -> Dict[str, Any]:
        """
        Optional size-controlled Odiff diagnostics.
        """
        if hasattr(self.module, "graded_local_odiff_audit"):
            try:
                out = self.module.graded_local_odiff_audit(
                    baseline_state=baseline_state,
                    trial_state=trial_state,
                    move=move,
                    args=args,
                )
                return dict(out)
            except TypeError:
                pass

        region_nodes = []
        if "node" in move:
            region_nodes = [int(move["node"])]
        elif "edge" in move:
            i, j = move["edge"]
            region_nodes = [int(i), int(j)]

        delta_sigma = trial_state.sigma - baseline_state.sigma
        site_rows = []
        for idx, ds in enumerate(delta_sigma.tolist()):
            site_rows.append({
                "site": int(idx),
                "delta_sigma": float(ds),
            })

        site_rows_sorted = sorted(site_rows, key=lambda row: row["delta_sigma"])
        k = int(getattr(args, "odiff_top_k", 3))
        most_negative = site_rows_sorted[:k]
        most_positive = list(reversed(site_rows_sorted[-k:])) if k > 0 else []
        odiff_before = float(np.sum(np.abs(baseline_state.sigma)))
        odiff_after = float(np.sum(np.abs(trial_state.sigma)))
        delta = float(odiff_after - odiff_before)
        ratio = float(odiff_after / odiff_before) if odiff_before > 1e-12 else None

        return {
            "region_nodes": region_nodes,
            "odiff_before": odiff_before,
            "odiff_after": odiff_after,
            "delta": delta,
            "ratio": ratio,
            "most_negative_sites": most_negative,
            "most_positive_sites": most_positive,
            "site_rows": site_rows if bool(getattr(args, "debug_full_site_rows", False)) else None,
        }


# -----------------------------------------------------------------------------
# Sandbox mechanics
# -----------------------------------------------------------------------------


class GradedSupportSandbox:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(int(args.seed))
        self.np_rng = np.random.default_rng(int(args.seed))
        self.physics = PhysicsAdapter(args.physics_module)
        self.bookkeeping = BookkeepingAdapter(args.bookkeeping_module)

    # -----------------------------------------------------------------
    # Initialization / materialization
    # -----------------------------------------------------------------

    def make_initial_state(self) -> GradedState:
        n_nodes = int(self.args.n_nodes)
        sigma = np.zeros(n_nodes, dtype=np.float64)

        # Default initial condition: first n_init nodes fully active.
        for idx in range(min(int(self.args.n_init), n_nodes)):
            sigma[idx] = 1.0

        interface_commitment: Dict[Edge, float] = {}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                interface_commitment[(i, j)] = 0.0

        # Seed interfaces among initially active nodes if requested.
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if sigma[i] >= self.args.sigma_on_threshold and sigma[j] >= self.args.sigma_on_threshold:
                    interface_commitment[(i, j)] = float(self.args.initial_edge_commitment)

        link_memory = {
            canonical_edge(i, j): LinkRegisterMemory()
            for i in range(n_nodes)
            for j in range(i + 1, n_nodes)
        }

        materialized = self.materialize_state(
            sigma=sigma,
            interface_commitment=interface_commitment,
            link_memory=link_memory,
            prior_psi=None,
        )

        return GradedState(
            n_nodes=n_nodes,
            sigma=sigma,
            interface_commitment=interface_commitment,
            link_memory=link_memory,
            materialized=materialized,
            current_step=0,
            accepted_moves=0,
            rng_seed=int(self.args.seed),
            run_metadata={
                "script": "hsf_gs_v4.py",
                "rewrite": "persistent_link_memory_and_continuous_background_evolution",
            },
        )

    def materialize_state(
        self,
        *,
        sigma: np.ndarray,
        interface_commitment: Mapping[Edge, float],
        link_memory: Mapping[Edge, LinkRegisterMemory],
        prior_psi: Optional[np.ndarray],
    ) -> MaterializedState:
        """
        Materialize the currently committed support into a physics state.

        Crucial behavior:
        - link_memory is passed through from graded state
        - it is NOT rebuilt from scratch here
        """
        active_nodes = [i for i, s in enumerate(sigma.tolist()) if s >= self.args.sigma_on_threshold]
        active_edges = [
            edge
            for edge, w in interface_commitment.items()
            if w >= self.args.edge_on_threshold
            and sigma[edge[0]] >= self.args.sigma_on_threshold
            and sigma[edge[1]] >= self.args.sigma_on_threshold
        ]

        graph_payload = self.physics.build_graph_payload(
            n_nodes=len(sigma),
            active_nodes=active_nodes,
            active_edges=active_edges,
            sigma=sigma,
            interface_commitment=interface_commitment,
            link_memory=link_memory,
            args=self.args,
        )

        psi = self.physics.initialize_psi(
            graph_payload=graph_payload,
            args=self.args,
            prior_psi=prior_psi,
        )

        h_payload = {
            "hamiltonian": self.physics.build_hamiltonian(graph_payload=graph_payload, args=self.args)
        }

        return MaterializedState(
            psi=psi,
            active_nodes=active_nodes,
            active_edges=active_edges,
            graph_payload=graph_payload,
            physics_payload=h_payload,
        )

    # -----------------------------------------------------------------
    # Background evolution
    # -----------------------------------------------------------------

    def background_evolve(self, state: GradedState, n_steps: int) -> None:
        if n_steps <= 0:
            return

        evolved_psi = self.physics.evolve_steps(
            psi=state.materialized.psi,
            graph_payload=state.materialized.graph_payload,
            args=self.args,
            n_steps=n_steps,
        )
        state.materialized.psi = normalize_state(evolved_psi)
        state.current_step += int(n_steps)

        self.physics.update_link_memory_from_materialized(
            link_memory=state.link_memory,
            materialized=state.materialized,
            active_edges=state.materialized.active_edges,
            step=state.current_step,
        )

    # -----------------------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------------------

    def generate_candidates(self, state: GradedState) -> List[Dict[str, Any]]:
        moves: List[Dict[str, Any]] = []
        n = state.n_nodes
        ds = float(self.args.support_step)
        dw = float(self.args.edge_step)

        # raise_support / lower_support
        for i in range(n):
            if state.sigma[i] < 1.0 - 1e-12:
                moves.append({
                    "kind": "raise_support",
                    "node": i,
                    "delta": ds,
                })
            if state.sigma[i] > 0.0 + 1e-12:
                moves.append({
                    "kind": "lower_support",
                    "node": i,
                    "delta": ds,
                })

        # edge_up / edge_down
        for edge, w in state.interface_commitment.items():
            i, j = edge
            if state.sigma[i] <= 0.0 or state.sigma[j] <= 0.0:
                continue
            if w < 1.0 - 1e-12:
                moves.append({
                    "kind": "edge_up",
                    "edge": edge,
                    "delta": dw,
                })
            if w > 0.0 + 1e-12:
                moves.append({
                    "kind": "edge_down",
                    "edge": edge,
                    "delta": dw,
                })

        return moves

    # -----------------------------------------------------------------
    # Branch application and scoring
    # -----------------------------------------------------------------

    def apply_move(self, base_state: GradedState, move: Mapping[str, Any]) -> GradedState:
        trial = base_state.copy_deep()
        kind = move["kind"]

        if kind == "raise_support":
            i = int(move["node"])
            trial.sigma[i] = clamp01(trial.sigma[i] + float(move["delta"]))
        elif kind == "lower_support":
            i = int(move["node"])
            trial.sigma[i] = clamp01(trial.sigma[i] - float(move["delta"]))
            if trial.sigma[i] < self.args.sigma_on_threshold:
                # Pull down incident edges when support falls below materialization threshold.
                for edge in list(trial.interface_commitment.keys()):
                    if i in edge:
                        trial.interface_commitment[edge] = min(
                            trial.interface_commitment[edge],
                            float(self.args.edge_off_when_node_off_cap),
                        )
        elif kind == "edge_up":
            edge = canonical_edge(*move["edge"])
            trial.interface_commitment[edge] = clamp01(trial.interface_commitment[edge] + float(move["delta"]))
        elif kind == "edge_down":
            edge = canonical_edge(*move["edge"])
            trial.interface_commitment[edge] = clamp01(trial.interface_commitment[edge] - float(move["delta"]))
        else:
            raise ValueError(f"Unknown move kind: {kind}")

        # Important: rematerialize WITHOUT resetting link_memory.
        trial.materialized = self.materialize_state(
            sigma=trial.sigma,
            interface_commitment=trial.interface_commitment,
            link_memory=trial.link_memory,
            prior_psi=base_state.materialized.psi,
        )

        # Allow a short branch-local settle if requested.
        branch_settle_steps = int(getattr(self.args, "branch_settle_steps", 0))
        if branch_settle_steps > 0:
            self.background_evolve(trial, branch_settle_steps)

        return trial

    def evaluate_candidates(self, baseline_state: GradedState) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        candidates = self.generate_candidates(baseline_state)
        scored: List[Dict[str, Any]] = []

        for move in candidates:
            trial = self.apply_move(baseline_state, move)
            score = self.bookkeeping.score_branch(
                baseline_state=baseline_state,
                trial_state=trial,
                move=move,
                args=self.args,
            )

            entry: Dict[str, Any] = {
                "move": dict(move),
                "score": score,
                "trial_state": trial,
            }

            if move["kind"] == "lower_support" and bool(self.args.enable_odiff_logging):
                entry["odiff"] = self.bookkeeping.odiff_diagnostics(
                    baseline_state=baseline_state,
                    trial_state=trial,
                    move=move,
                    args=self.args,
                )

            scored.append(entry)

        if not scored:
            return scored, None

        scored.sort(key=lambda row: safe_float(row["score"].get("total_score", -1e99)), reverse=True)
        best = scored[0]
        if safe_float(best["score"].get("total_score", -1e99)) <= float(self.args.min_accept_score):
            return scored, None
        return scored, best

    # -----------------------------------------------------------------
    # Logging helpers
    # -----------------------------------------------------------------

    def summarize_state(self, state: GradedState) -> Dict[str, Any]:
        return {
            "step": int(state.current_step),
            "accepted_moves": int(state.accepted_moves),
            "sigma": [float(x) for x in state.sigma.tolist()],
            "active_nodes": list(map(int, state.materialized.active_nodes)),
            "active_edges": [list(map(int, e)) for e in state.materialized.active_edges],
            "mean_sigma": float(np.mean(state.sigma)),
            "sum_sigma": float(np.sum(state.sigma)),
            "mean_interface_commitment": float(np.mean(list(state.interface_commitment.values()))) if state.interface_commitment else 0.0,
        }

    def compact_candidate_summary(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        move = dict(entry["move"])
        score = dict(entry["score"])
        out = {
            "move": move,
            "total_score": safe_float(score.get("total_score", 0.0)),
            "dE_expr_raw": safe_float(score.get("dE_expr_raw", 0.0)),
            "dCB": safe_float(score.get("dCB", 0.0)),
            "dCR": safe_float(score.get("dCR", 0.0)),
            "dCS": safe_float(score.get("dCS", 0.0)),
            "dCF": safe_float(score.get("dCF", 0.0)),
        }
        if "odiff" in entry:
            od = dict(entry["odiff"])
            out["odiff_compact"] = {
                "region_nodes": od.get("region_nodes"),
                "odiff_before": od.get("odiff_before"),
                "odiff_after": od.get("odiff_after"),
                "delta": od.get("delta"),
                "ratio": od.get("ratio"),
                "most_negative_sites": od.get("most_negative_sites"),
                "most_positive_sites": od.get("most_positive_sites"),
            }
        return out

    def accepted_move_detail(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        trial_state: GradedState = entry["trial_state"]
        out = {
            "move": dict(entry["move"]),
            "score": dict(entry["score"]),
            "resulting_state": self.summarize_state(trial_state),
        }
        if "odiff" in entry:
            od = dict(entry["odiff"])
            if bool(self.args.compact_json):
                out["odiff"] = {
                    "region_nodes": od.get("region_nodes"),
                    "odiff_before": od.get("odiff_before"),
                    "odiff_after": od.get("odiff_after"),
                    "delta": od.get("delta"),
                    "ratio": od.get("ratio"),
                    "most_negative_sites": od.get("most_negative_sites"),
                    "most_positive_sites": od.get("most_positive_sites"),
                }
            else:
                out["odiff"] = od
        return out

    # -----------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        state = self.make_initial_state()
        snapshots: List[Dict[str, Any]] = []
        accepted_log: List[Dict[str, Any]] = []
        eval_log: List[Dict[str, Any]] = []
        move_counts = {
            "raise_support": 0,
            "lower_support": 0,
            "edge_up": 0,
            "edge_down": 0,
        }

        total_eval_windows = int(self.args.total_steps // self.args.eval_every)
        for window in range(total_eval_windows):
            # Priority 2 fix: unconditional background evolution of the baseline.
            self.background_evolve(state, int(self.args.eval_every))

            scored, winner = self.evaluate_candidates(state)
            top_summaries = [self.compact_candidate_summary(row) for row in scored[: int(self.args.keep_top_candidates)]]

            window_record: Dict[str, Any] = {
                "window": int(window),
                "baseline_after_background": self.summarize_state(state),
                "top_candidates": top_summaries,
                "winner": None,
            }

            if winner is not None:
                trial_state: GradedState = winner["trial_state"]
                move_kind = str(winner["move"]["kind"])
                move_counts[move_kind] = move_counts.get(move_kind, 0) + 1
                trial_state.accepted_moves = state.accepted_moves + 1
                state = trial_state

                accepted_detail = self.accepted_move_detail(winner)
                accepted_detail["window"] = int(window)
                accepted_log.append(accepted_detail)
                window_record["winner"] = accepted_detail
            else:
                window_record["winner"] = None

            eval_log.append(window_record)

            if window % int(self.args.snapshot_every_windows) == 0:
                snapshots.append(self.summarize_state(state))

        final_payload = {
            "run_metadata": {
                **state.run_metadata,
                "seed": int(self.args.seed),
                "n_nodes": int(self.args.n_nodes),
                "n_init": int(self.args.n_init),
                "total_steps": int(self.args.total_steps),
                "eval_every": int(self.args.eval_every),
                "sigma_on_threshold": float(self.args.sigma_on_threshold),
                "edge_on_threshold": float(self.args.edge_on_threshold),
                "support_step": float(self.args.support_step),
                "edge_step": float(self.args.edge_step),
                "compact_json": bool(self.args.compact_json),
                "enable_odiff_logging": bool(self.args.enable_odiff_logging),
            },
            "move_counts": move_counts,
            "final_state": self.summarize_state(state),
            "snapshots": snapshots,
            "accepted_moves": accepted_log,
            "eval_windows": eval_log,
        }

        if bool(self.args.include_final_link_memory):
            final_payload["final_link_memory"] = {
                f"{i}-{j}": mem.to_json() for (i, j), mem in state.link_memory.items()
            }

        return final_payload


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HSF graded-support sandbox rewrite (v4).")

    p.add_argument("--physics-module", default="hsf_mesoscale_physics_core")
    p.add_argument("--bookkeeping-module", default="hsf_mesoscale_bookkeeping")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-nodes", type=int, default=8)
    p.add_argument("--n-init", type=int, default=2)
    p.add_argument("--total-steps", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--fallback-dim", type=int, default=16)

    p.add_argument("--sigma-on-threshold", type=float, default=0.50)
    p.add_argument("--edge-on-threshold", type=float, default=0.50)
    p.add_argument("--initial-edge-commitment", type=float, default=0.60)
    p.add_argument("--support-step", type=float, default=0.15)
    p.add_argument("--edge-step", type=float, default=0.15)
    p.add_argument("--edge-off-when-node-off-cap", type=float, default=0.35)
    p.add_argument("--branch-settle-steps", type=int, default=0)
    p.add_argument("--min-accept-score", type=float, default=0.0)

    p.add_argument("--lambda-B", dest="lambda_B", type=float, default=0.05)
    p.add_argument("--lambda-R", dest="lambda_R", type=float, default=0.05)
    p.add_argument("--lambda-S", dest="lambda_S", type=float, default=0.05)
    p.add_argument("--lambda-F", dest="lambda_F", type=float, default=0.05)

    p.add_argument("--compact-json", action="store_true", default=True)
    p.add_argument("--no-compact-json", dest="compact_json", action="store_false")
    p.add_argument("--keep-top-candidates", type=int, default=8)
    p.add_argument("--snapshot-every-windows", type=int, default=1)
    p.add_argument("--enable-odiff-logging", action="store_true", default=False)
    p.add_argument("--odiff-top-k", type=int, default=3)
    p.add_argument("--debug-full-site-rows", action="store_true", default=False)
    p.add_argument("--include-final-link-memory", action="store_true", default=False)

    p.add_argument("--json-out", default="n8_t300_s0_v8.json")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    sandbox = GradedSupportSandbox(args)
    out = sandbox.run()

    out_path = os.path.abspath(args.json_out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== hsf_gs_v4 ===")
    print(f"json_out: {out_path}")
    print(f"final step: {out['final_state']['step']}")
    print(f"accepted moves: {sum(out['move_counts'].values())}")
    print("move counts:")
    for k, v in out["move_counts"].items():
        print(f"  {k}: {v}")
    print(f"final sigma: {out['final_state']['sigma']}")
    print(f"active nodes: {out['final_state']['active_nodes']}")
    print(f"active edges: {out['final_state']['active_edges']}")


if __name__ == "__main__":
    main()
