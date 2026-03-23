# filename: hsf_mesoscape_local_move_witness_audit.py
"""
HSF mesoscape local move witness audit
--------------------------------------

Purpose
-------
Audit accepted local moves in a mesoscape run using the frozen minimal
no-refolding witness bundle

    W_NR^min = (W_func, W_book, W_slack)

with a move classifier:

    lawful_reexpression / refolding_like / unresolved

This script is intentionally conservative and proxy-based. It is not a claim
that no-refolding is fully derived. It is an implementation-facing audit tool
for the current organizer / no-refolding program.

What it tries to do
-------------------
1. Load a saved mesoscape JSON run.
2. Recover a snapshot sequence and accepted move sequence from a variety of
   likely schema variants.
3. For each accepted move, build a local comparison region R around the move.
4. Compute provisional sector scores:
      - W_func  : loss of local relief-function proxies
      - W_book  : loss of interface/bookkeeping continuity
      - W_slack : increase in underused / weakly justified support
5. Classify each move.
6. Emit JSON and/or CSV logs.

Design choices
--------------
- Function-first identity:
    W_func is the primary sector.
- Bookkeeping continuity is secondary but real:
    W_book captures committed interface continuity.
- Slack suppression prevents fake preservation:
    W_slack penalizes padded or weakly integrated support.

Accepted temporary proxy families used here
-------------------------------------------
W_func  <- local support/activity retention, local edge-weight retention,
           optional local MI bundle / correlator bundle if present.
W_book  <- endpoint/interface continuity, top committed-interface overlap,
           moved-endpoint retention.
W_slack <- positive increase in support that is weakly integrated into the
           local support/interface pattern.

Important limitation
--------------------
This script can only use what is present in the run JSON. If the JSON does not
contain exact local MI or correlator objects, it falls back to admissible
provisional support/interface proxies. Those fallbacks are explicitly logged.

Usage
-----
Windows one-line examples:

python hsf_mesoscape_local_move_witness_audit.py run.json

python hsf_mesoscape_local_move_witness_audit.py run.json --json-out audit.json --csv-out audit.csv

python hsf_mesoscape_local_move_witness_audit.py run.json --move-types raise_support edge_up proto birth weaken retire

python hsf_mesoscape_local_move_witness_audit.py run.json --sigma-thresh 0.10 --edge-thresh 0.05 --radius 1

Notes
-----
- The script is intentionally schema-tolerant and tries several common keys.
- It assumes snapshots are ordered in time if no explicit step key exists.
- It assumes accepted moves align roughly with successive snapshots.
- If alignment is ambiguous, it still emits rows and marks low-confidence cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires numpy. Install it with: pip install numpy"
    ) from exc


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    step: int
    sigma: np.ndarray
    W: np.ndarray  # weighted adjacency / interface commitment matrix
    active: np.ndarray
    extras: Dict[str, Any]


@dataclass
class Move:
    idx: int
    step: Optional[int]
    move_type: str
    accepted: bool
    raw: Dict[str, Any]
    nodes_hint: List[int]
    confidence: str


@dataclass
class AuditRow:
    move_index: int
    move_step: Optional[int]
    move_type: str
    accepted: bool
    pre_step: Optional[int]
    post_step: Optional[int]
    region_nodes: List[int]
    region_size: int
    seed_nodes: List[int]

    w_func: float
    w_book: float
    w_slack: float

    p_func_retained: float
    p_book_retained: float
    slack_pre: float
    slack_post: float
    delta_slack_pos: float

    classifier: str
    confidence: str
    proxy_notes: List[str]

    local_sigma_l1: float
    local_edge_l1: float
    endpoint_overlap: float
    top_interface_jaccard: float
    active_node_retention: float

    raw_expr_delta: Optional[float]
    total_expr_delta: Optional[float]
    notes: List[str]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return default if abs(b) < 1e-12 else a / b


def flatten_one_level(seq: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    for x in seq:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def maybe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def is_square_matrix_like(x: Any) -> bool:
    if not isinstance(x, list) or not x:
        return False
    if not all(isinstance(r, list) for r in x):
        return False
    n = len(x)
    return all(len(r) == n for r in x)


def argmax_index(seq: Sequence[float]) -> int:
    best_i = 0
    best_v = seq[0]
    for i, v in enumerate(seq):
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


def unique_sorted_ints(xs: Iterable[int]) -> List[int]:
    return sorted(set(int(x) for x in xs if x is not None))


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Snapshot parsing
# ---------------------------------------------------------------------------

def find_sequence_by_keys(root: Any, candidate_keys: Sequence[str]) -> Optional[List[Any]]:
    if isinstance(root, dict):
        for k in candidate_keys:
            v = root.get(k)
            if isinstance(v, list) and v:
                return v
        for v in root.values():
            res = find_sequence_by_keys(v, candidate_keys)
            if res is not None:
                return res
    elif isinstance(root, list):
        for item in root:
            res = find_sequence_by_keys(item, candidate_keys)
            if res is not None:
                return res
    return None


def infer_n_from_snapshot_dict(d: Dict[str, Any]) -> Optional[int]:
    for key in ("sigma", "support", "support_commitment", "supports", "node_support"):
        if key in d and isinstance(d[key], list):
            return len(d[key])
    for key in ("W", "w", "adj", "adjacency", "weights", "edge_weights", "commitment_matrix"):
        if key in d and is_square_matrix_like(d[key]):
            return len(d[key])
    edges = None
    for key in ("edges", "edge_list", "active_edges", "weighted_edges"):
        if key in d and isinstance(d[key], list):
            edges = d[key]
            break
    if edges:
        m = -1
        for e in edges:
            if isinstance(e, dict):
                u = maybe_int(e.get("i", e.get("u", e.get("src"))))
                v = maybe_int(e.get("j", e.get("v", e.get("dst"))))
            elif isinstance(e, list) and len(e) >= 2:
                u = maybe_int(e[0])
                v = maybe_int(e[1])
            else:
                continue
            if u is not None:
                m = max(m, u)
            if v is not None:
                m = max(m, v)
        if m >= 0:
            return m + 1
    return None


def extract_sigma(d: Dict[str, Any], n: int) -> np.ndarray:
    candidates = [
        "sigma", "support", "support_commitment", "supports",
        "node_support", "support_levels", "sigma_vec"
    ]
    for key in candidates:
        if key in d and isinstance(d[key], list):
            vals = np.array([to_float(x) for x in d[key]], dtype=float)
            if len(vals) == n:
                return vals
    # fallback to binary active if present
    for key in ("active", "active_nodes", "alive", "occupied"):
        if key in d:
            v = d[key]
            if isinstance(v, list):
                if len(v) == n and all(isinstance(x, (bool, int, float)) for x in v):
                    return np.array([1.0 if bool(x) else 0.0 for x in v], dtype=float)
                if all(isinstance(x, int) for x in v):
                    sigma = np.zeros(n, dtype=float)
                    for i in v:
                        if 0 <= i < n:
                            sigma[i] = 1.0
                    return sigma
    return np.zeros(n, dtype=float)


def extract_weight_matrix(d: Dict[str, Any], n: int) -> np.ndarray:
    matrix_keys = [
        "W", "w", "adj", "adjacency", "weights",
        "edge_weights", "commitment_matrix", "interface_weights"
    ]
    for key in matrix_keys:
        if key in d and is_square_matrix_like(d[key]):
            arr = np.array(d[key], dtype=float)
            if arr.shape == (n, n):
                return arr

    W = np.zeros((n, n), dtype=float)
    edge_keys = ["edges", "edge_list", "active_edges", "weighted_edges", "links"]
    edge_list = None
    for key in edge_keys:
        if key in d and isinstance(d[key], list):
            edge_list = d[key]
            break

    if edge_list is None:
        return W

    for e in edge_list:
        u = v = None
        w = 1.0
        if isinstance(e, dict):
            u = maybe_int(e.get("i", e.get("u", e.get("src"))))
            v = maybe_int(e.get("j", e.get("v", e.get("dst"))))
            w = to_float(
                e.get("w", e.get("weight", e.get("value", e.get("commitment", 1.0)))),
                default=1.0,
            )
        elif isinstance(e, list) or isinstance(e, tuple):
            if len(e) >= 2:
                u = maybe_int(e[0])
                v = maybe_int(e[1])
            if len(e) >= 3:
                w = to_float(e[2], default=1.0)
        if u is None or v is None or u < 0 or v < 0 or u >= n or v >= n:
            continue
        W[u, v] = max(W[u, v], w)
        W[v, u] = max(W[v, u], w)
    return W


def extract_active(sigma: np.ndarray, W: np.ndarray, sigma_thresh: float, edge_thresh: float) -> np.ndarray:
    support_active = sigma > sigma_thresh
    edge_active = (W > edge_thresh).any(axis=1)
    return np.logical_or(support_active, edge_active)


def coerce_snapshot(item: Any, default_step: int, sigma_thresh: float, edge_thresh: float) -> Optional[Snapshot]:
    if not isinstance(item, dict):
        return None
    n = infer_n_from_snapshot_dict(item)
    if n is None:
        return None

    step = maybe_int(item.get("step", item.get("t", item.get("time", default_step))))
    if step is None:
        step = default_step

    sigma = extract_sigma(item, n)
    W = extract_weight_matrix(item, n)
    active = extract_active(sigma, W, sigma_thresh=sigma_thresh, edge_thresh=edge_thresh)

    extras = {}
    for k in ("local_mi", "mi_bundle", "correlators", "correlator_profile", "activity", "notes"):
        if k in item:
            extras[k] = item[k]

    return Snapshot(step=step, sigma=sigma, W=W, active=active, extras=extras)


def parse_snapshots(root: Any, sigma_thresh: float, edge_thresh: float) -> List[Snapshot]:
    seq = find_sequence_by_keys(
        root,
        candidate_keys=[
            "snapshots", "history", "states", "trajectory", "frames", "steps"
        ],
    )
    if seq is None:
        raise ValueError("Could not find a snapshot/history/state sequence in the JSON.")

    snaps: List[Snapshot] = []
    for i, item in enumerate(seq):
        snap = coerce_snapshot(item, default_step=i, sigma_thresh=sigma_thresh, edge_thresh=edge_thresh)
        if snap is not None:
            snaps.append(snap)

    if len(snaps) < 2:
        raise ValueError("Need at least two usable snapshots for move auditing.")

    # Keep original order, but normalize step monotonicity if impossible.
    return snaps


# ---------------------------------------------------------------------------
# Move parsing
# ---------------------------------------------------------------------------

def find_move_sequence(root: Any) -> Optional[List[Any]]:
    return find_sequence_by_keys(
        root,
        candidate_keys=[
            "moves", "move_log", "events", "accepted_moves", "event_log", "actions"
        ],
    )


def infer_move_type(d: Dict[str, Any]) -> str:
    for key in ("move_type", "type", "kind", "move", "action"):
        if key in d:
            return str(d[key])
    return "unknown"


def infer_move_accepted(d: Dict[str, Any]) -> bool:
    if "accepted" in d:
        return bool(d["accepted"])
    if "accept" in d:
        return bool(d["accept"])
    if "winner" in d:
        return bool(d["winner"])
    # default: if it appears in move log, often it is accepted
    return True


def nodes_from_move_dict(d: Dict[str, Any]) -> Tuple[List[int], str]:
    nodes: Set[int] = set()
    confidence = "low"

    direct_pairs = [
        ("i", "j"), ("u", "v"), ("src", "dst"),
        ("parent", "child"), ("from", "to")
    ]
    for a, b in direct_pairs:
        if a in d:
            x = maybe_int(d.get(a))
            if x is not None:
                nodes.add(x)
                confidence = "medium"
        if b in d:
            y = maybe_int(d.get(b))
            if y is not None:
                nodes.add(y)
                confidence = "medium"

    for key in ("node", "child", "primary_parent", "secondary_parent"):
        if key in d:
            x = maybe_int(d.get(key))
            if x is not None:
                nodes.add(x)
                confidence = "medium"

    for key in ("parent_pair", "edge", "proto_edge", "nodes", "region_nodes", "support_nodes"):
        if key in d and isinstance(d[key], list):
            for x in d[key]:
                xi = maybe_int(x)
                if xi is not None:
                    nodes.add(xi)
                    confidence = "high" if key in ("parent_pair", "edge", "proto_edge", "nodes") else confidence

    # nested move object
    for key in ("move_obj", "candidate", "proposal"):
        if key in d and isinstance(d[key], dict):
            sub_nodes, sub_conf = nodes_from_move_dict(d[key])
            nodes.update(sub_nodes)
            if sub_conf == "high":
                confidence = "high"
            elif sub_conf == "medium" and confidence == "low":
                confidence = "medium"

    return unique_sorted_ints(nodes), confidence


def coerce_move(item: Any, idx: int) -> Optional[Move]:
    if not isinstance(item, dict):
        return None
    move_type = infer_move_type(item)
    accepted = infer_move_accepted(item)
    step = maybe_int(item.get("step", item.get("t", item.get("time"))))
    nodes_hint, confidence = nodes_from_move_dict(item)
    return Move(
        idx=idx,
        step=step,
        move_type=move_type,
        accepted=accepted,
        raw=item,
        nodes_hint=nodes_hint,
        confidence=confidence,
    )


def parse_moves(root: Any) -> List[Move]:
    seq = find_move_sequence(root)
    if seq is None:
        return []
    moves: List[Move] = []
    for i, item in enumerate(seq):
        mv = coerce_move(item, idx=i)
        if mv is not None:
            moves.append(mv)
    return moves


# ---------------------------------------------------------------------------
# Alignment and region building
# ---------------------------------------------------------------------------

def align_moves_to_snapshot_pairs(
    snapshots: List[Snapshot],
    moves: List[Move],
    accepted_only: bool = True,
) -> List[Tuple[Move, Snapshot, Snapshot]]:
    usable_moves = [m for m in moves if (m.accepted or not accepted_only)]
    if not usable_moves:
        # fallback: synthesize unknown moves between every successive snapshot pair
        synth: List[Tuple[Move, Snapshot, Snapshot]] = []
        for i in range(len(snapshots) - 1):
            mv = Move(
                idx=i,
                step=snapshots[i].step,
                move_type="unknown",
                accepted=True,
                raw={},
                nodes_hint=[],
                confidence="low",
            )
            synth.append((mv, snapshots[i], snapshots[i + 1]))
        return synth

    # If move steps are present and can be mapped, do that.
    by_step: Dict[int, int] = {snap.step: i for i, snap in enumerate(snapshots)}
    aligned: List[Tuple[Move, Snapshot, Snapshot]] = []

    all_have_step = all(m.step is not None for m in usable_moves)
    if all_have_step:
        for m in usable_moves:
            if m.step in by_step:
                i = by_step[m.step]
                if i < len(snapshots) - 1:
                    aligned.append((m, snapshots[i], snapshots[i + 1]))
        if aligned:
            return aligned

    # fallback: assume accepted moves correspond in order to snapshot transitions
    n_pairs = min(len(usable_moves), len(snapshots) - 1)
    return [(usable_moves[i], snapshots[i], snapshots[i + 1]) for i in range(n_pairs)]


def neighborhood_nodes(W: np.ndarray, seeds: Iterable[int], radius: int, edge_thresh: float) -> Set[int]:
    n = W.shape[0]
    current: Set[int] = set(int(x) for x in seeds if 0 <= int(x) < n)
    visited: Set[int] = set(current)
    if radius <= 0:
        return visited

    active_adj = W > edge_thresh
    frontier = set(current)
    for _ in range(radius):
        nxt: Set[int] = set()
        for u in frontier:
            nbrs = set(np.where(active_adj[u])[0].tolist())
            nxt.update(nbrs)
        nxt -= visited
        visited.update(nxt)
        frontier = nxt
        if not frontier:
            break
    return visited


def infer_seed_nodes_from_delta(pre: Snapshot, post: Snapshot, sigma_thresh: float, edge_thresh: float) -> List[int]:
    ds = np.abs(post.sigma - pre.sigma)
    dW = np.abs(post.W - pre.W).sum(axis=1)
    touched = set(np.where(ds > 1e-9)[0].tolist())
    touched.update(np.where(dW > 1e-9)[0].tolist())
    if touched:
        return sorted(touched)
    # fallback to changed active support/edge sets
    changed_active = set(np.where(pre.active != post.active)[0].tolist())
    if changed_active:
        return sorted(changed_active)
    # no visible change
    return []


def build_local_region(
    move: Move,
    pre: Snapshot,
    post: Snapshot,
    radius: int,
    sigma_thresh: float,
    edge_thresh: float,
) -> Tuple[List[int], List[int], List[str]]:
    notes: List[str] = []

    seeds = list(move.nodes_hint)
    if not seeds:
        seeds = infer_seed_nodes_from_delta(pre, post, sigma_thresh=sigma_thresh, edge_thresh=edge_thresh)
        if seeds:
            notes.append("seed_nodes inferred from local state delta")
        else:
            # last-resort: pick top-changed node
            ds = np.abs(post.sigma - pre.sigma)
            dW = np.abs(post.W - pre.W).sum(axis=1)
            score = ds + dW
            if score.max() > 0:
                seeds = [int(argmax_index(score.tolist()))]
                notes.append("seed_nodes inferred from argmax local delta")
            else:
                seeds = [0]
                notes.append("seed_nodes defaulted to [0]")

    region = neighborhood_nodes(np.maximum(pre.W, post.W), seeds, radius=radius, edge_thresh=edge_thresh)
    region.update(seeds)

    # Also include nodes with changed support/incident edge in the immediate local neighborhood.
    changed = set(infer_seed_nodes_from_delta(pre, post, sigma_thresh=sigma_thresh, edge_thresh=edge_thresh))
    for c in changed:
        if c in region:
            region.update(neighborhood_nodes(np.maximum(pre.W, post.W), [c], radius=1, edge_thresh=edge_thresh))

    return sorted(region), sorted(set(seeds)), notes


# ---------------------------------------------------------------------------
# Local proxy computation
# ---------------------------------------------------------------------------

def subvector(x: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    return x[np.array(idx, dtype=int)]


def submatrix(W: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    arr = np.array(idx, dtype=int)
    return W[np.ix_(arr, arr)]


def weighted_degree(W: np.ndarray) -> np.ndarray:
    return W.sum(axis=1)


def top_interfaces(W: np.ndarray, idx: Sequence[int], edge_thresh: float, top_k: int = 8) -> Set[Tuple[int, int]]:
    local = submatrix(W, idx)
    edges: List[Tuple[float, int, int]] = []
    for a in range(local.shape[0]):
        for b in range(a + 1, local.shape[1]):
            w = float(local[a, b])
            if w > edge_thresh:
                edges.append((w, idx[a], idx[b]))
    edges.sort(reverse=True)
    picked = {(min(i, j), max(i, j)) for _, i, j in edges[:top_k]}
    return picked


def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    return safe_div(len(a & b), len(a | b), default=1.0)


def edge_weight_l1(preW: np.ndarray, postW: np.ndarray) -> float:
    denom = float(np.abs(preW).sum() + 1e-12)
    return float(np.abs(postW - preW).sum() / denom)


def sigma_l1(pre_sigma: np.ndarray, post_sigma: np.ndarray) -> float:
    denom = float(np.abs(pre_sigma).sum() + 1e-12)
    return float(np.abs(post_sigma - pre_sigma).sum() / denom)


def retained_activity_ratio(pre: Snapshot, post: Snapshot, idx: Sequence[int]) -> float:
    pre_local_W = submatrix(pre.W, idx)
    post_local_W = submatrix(post.W, idx)
    pre_mass = float(pre_local_W.sum())
    if pre_mass < 1e-12:
        return 1.0
    retained = float(np.minimum(pre_local_W, post_local_W).sum())
    return clamp01(retained / pre_mass)


def active_node_retention(pre: Snapshot, post: Snapshot, idx: Sequence[int], sigma_thresh: float, edge_thresh: float) -> float:
    pre_sig = subvector(pre.sigma, idx)
    post_sig = subvector(post.sigma, idx)
    preW = submatrix(pre.W, idx)
    postW = submatrix(post.W, idx)

    pre_active = np.logical_or(pre_sig > sigma_thresh, (preW > edge_thresh).any(axis=1))
    post_active = np.logical_or(post_sig > sigma_thresh, (postW > edge_thresh).any(axis=1))

    denom = int(pre_active.sum())
    if denom == 0:
        return 1.0
    retained = int(np.logical_and(pre_active, post_active).sum())
    return clamp01(retained / denom)


def endpoint_overlap_score(move: Move, pre: Snapshot, post: Snapshot, idx: Sequence[int], edge_thresh: float) -> float:
    hinted = move.nodes_hint
    if not hinted:
        return 1.0

    pre_local = set()
    post_local = set()

    for node in hinted:
        if node < 0 or node >= pre.W.shape[0]:
            continue
        pre_nbrs = set(np.where(pre.W[node] > edge_thresh)[0].tolist())
        post_nbrs = set(np.where(post.W[node] > edge_thresh)[0].tolist())
        pre_local.update((min(node, j), max(node, j)) for j in pre_nbrs if j in idx)
        post_local.update((min(node, j), max(node, j)) for j in post_nbrs if j in idx)

    if not pre_local and not post_local:
        return 1.0
    return jaccard(pre_local, post_local)


def slack_burden(snapshot: Snapshot, idx: Sequence[int], sigma_thresh: float, edge_thresh: float) -> float:
    sig = subvector(snapshot.sigma, idx)
    W = submatrix(snapshot.W, idx)
    deg = weighted_degree(W)

    # Normalize degree against local max for scale tolerance.
    deg_norm = deg / max(float(deg.max()), 1e-12)

    # Underused support: support present, but weak edge integration.
    underused = np.maximum(sig - deg_norm, 0.0)

    # Isolated active carriers are extra suspicious.
    isolated = np.logical_and(sig > sigma_thresh, deg <= edge_thresh).astype(float)

    return float(underused.sum() + 0.5 * isolated.sum())


def local_mi_bundle(snapshot: Snapshot, idx: Sequence[int]) -> Optional[np.ndarray]:
    # Accept several possible extras shapes.
    for key in ("local_mi", "mi_bundle"):
        if key in snapshot.extras:
            raw = snapshot.extras[key]
            if isinstance(raw, list):
                arr = np.array(raw, dtype=float)
                if arr.ndim == 2 and arr.shape[0] == snapshot.W.shape[0] and arr.shape[1] == snapshot.W.shape[1]:
                    return submatrix(arr, idx)
                if arr.ndim == 1 and len(arr) == snapshot.W.shape[0]:
                    return subvector(arr, idx)
    return None


def local_correlator_bundle(snapshot: Snapshot, idx: Sequence[int]) -> Optional[np.ndarray]:
    for key in ("correlators", "correlator_profile"):
        if key in snapshot.extras:
            raw = snapshot.extras[key]
            if isinstance(raw, list):
                arr = np.array(raw, dtype=float)
                if arr.ndim == 2 and arr.shape[0] == snapshot.W.shape[0] and arr.shape[1] == snapshot.W.shape[1]:
                    return submatrix(arr, idx)
                if arr.ndim == 1 and len(arr) == snapshot.W.shape[0]:
                    return subvector(arr, idx)
    return None


def normalized_bundle_loss(pre_arr: np.ndarray, post_arr: np.ndarray) -> float:
    denom = float(np.abs(pre_arr).sum() + 1e-12)
    return clamp01(float(np.abs(post_arr - pre_arr).sum() / denom))


def compute_w_func(
    move: Move,
    pre: Snapshot,
    post: Snapshot,
    idx: Sequence[int],
    sigma_thresh: float,
    edge_thresh: float,
) -> Tuple[float, float, List[str]]:
    notes: List[str] = []

    # Primary provisional pieces
    activity_ret = retained_activity_ratio(pre, post, idx)
    active_ret = active_node_retention(pre, post, idx, sigma_thresh=sigma_thresh, edge_thresh=edge_thresh)

    pre_sig = subvector(pre.sigma, idx)
    post_sig = subvector(post.sigma, idx)
    preW = submatrix(pre.W, idx)
    postW = submatrix(post.W, idx)

    sigma_loss = sigma_l1(pre_sig, post_sig)
    edge_loss = edge_weight_l1(preW, postW)

    pieces: List[float] = [
        1.0 - activity_ret,
        1.0 - active_ret,
        sigma_loss,
        edge_loss,
    ]
    weights: List[float] = [0.35, 0.20, 0.20, 0.25]
    notes.append("W_func uses retained activity + active-node retention + local sigma/edge retention")

    # Optional MI bundle
    pre_mi = local_mi_bundle(pre, idx)
    post_mi = local_mi_bundle(post, idx)
    if pre_mi is not None and post_mi is not None and pre_mi.shape == post_mi.shape:
        mi_loss = normalized_bundle_loss(pre_mi, post_mi)
        pieces.append(mi_loss)
        weights.append(0.20)
        notes.append("W_func includes local MI bundle loss")
    else:
        notes.append("W_func MI bundle unavailable -> fallback proxies only")

    # Optional correlator bundle
    pre_corr = local_correlator_bundle(pre, idx)
    post_corr = local_correlator_bundle(post, idx)
    if pre_corr is not None and post_corr is not None and pre_corr.shape == post_corr.shape:
        corr_loss = normalized_bundle_loss(pre_corr, post_corr)
        pieces.append(corr_loss)
        weights.append(0.15)
        notes.append("W_func includes local correlator bundle loss")
    else:
        notes.append("W_func correlator bundle unavailable -> fallback proxies only")

    # Normalize weights if optional pieces were added.
    wsum = sum(weights)
    w_norm = [w / wsum for w in weights]
    w_func = clamp01(sum(w * p for w, p in zip(w_norm, pieces)))
    p_func_retained = clamp01(1.0 - w_func)
    return w_func, p_func_retained, notes


def compute_w_book(
    move: Move,
    pre: Snapshot,
    post: Snapshot,
    idx: Sequence[int],
    edge_thresh: float,
) -> Tuple[float, float, float, float, List[str]]:
    notes: List[str] = []

    endpoint_overlap = endpoint_overlap_score(move, pre, post, idx, edge_thresh=edge_thresh)
    top_pre = top_interfaces(pre.W, idx, edge_thresh=edge_thresh, top_k=8)
    top_post = top_interfaces(post.W, idx, edge_thresh=edge_thresh, top_k=8)
    top_j = jaccard(top_pre, top_post)

    activity_ret = retained_activity_ratio(pre, post, idx)

    # Bookkeeping continuity is mostly top-interface continuity plus endpoint overlap.
    p_book = clamp01(0.45 * endpoint_overlap + 0.35 * top_j + 0.20 * activity_ret)
    w_book = clamp01(1.0 - p_book)

    notes.append("W_book uses endpoint/interface overlap + top-interface Jaccard + retained interface activity")
    if not move.nodes_hint:
        notes.append("W_book endpoint overlap is weaker because move endpoints were inferred or absent")

    return w_book, p_book, endpoint_overlap, top_j, notes


def compute_w_slack(
    pre: Snapshot,
    post: Snapshot,
    idx: Sequence[int],
    sigma_thresh: float,
    edge_thresh: float,
) -> Tuple[float, float, float, float, List[str]]:
    notes: List[str] = []

    s_pre = slack_burden(pre, idx, sigma_thresh=sigma_thresh, edge_thresh=edge_thresh)
    s_post = slack_burden(post, idx, sigma_thresh=sigma_thresh, edge_thresh=edge_thresh)
    delta_pos = max(0.0, s_post - s_pre)

    # Normalize using local support mass plus local edge mass.
    local_scale = (
        float(subvector(pre.sigma, idx).sum() + subvector(post.sigma, idx).sum()) +
        float(submatrix(pre.W, idx).sum() + submatrix(post.W, idx).sum()) +
        1e-12
    )
    w_slack = clamp01(delta_pos / local_scale)

    notes.append("W_slack uses positive increase in underused support / weak integration burden")
    return w_slack, s_pre, s_post, delta_pos, notes


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_move(
    w_func: float,
    w_book: float,
    w_slack: float,
    eps_func: float,
    eps_book: float,
    eps_slack: float,
    unresolved_margin: float,
) -> str:
    lawful = (
        w_func <= eps_func and
        w_book <= eps_book and
        w_slack <= eps_slack
    )
    if lawful:
        return "lawful_reexpression"

    # Refolding-like if any strong violation, especially function or bookkeeping.
    strong_refolding = (
        w_func > eps_func + unresolved_margin or
        w_book > eps_book + unresolved_margin or
        w_slack > eps_slack + unresolved_margin
    )
    if strong_refolding:
        return "refolding_like"

    return "unresolved"


# ---------------------------------------------------------------------------
# Raw move diagnostics
# ---------------------------------------------------------------------------

def extract_expr_delta(raw: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in raw:
            try:
                return float(raw[key])
            except Exception:
                pass
    # search one nested layer
    for outer in ("candidate", "proposal", "move_obj", "audit"):
        sub = raw.get(outer)
        if isinstance(sub, dict):
            for key in keys:
                if key in sub:
                    try:
                        return float(sub[key])
                    except Exception:
                        pass
    return None


# ---------------------------------------------------------------------------
# Audit runner
# ---------------------------------------------------------------------------

def audit_run(
    root: Any,
    move_type_filter: Optional[Set[str]],
    radius: int,
    sigma_thresh: float,
    edge_thresh: float,
    eps_func: float,
    eps_book: float,
    eps_slack: float,
    unresolved_margin: float,
) -> List[AuditRow]:
    snapshots = parse_snapshots(root, sigma_thresh=sigma_thresh, edge_thresh=edge_thresh)
    moves = parse_moves(root)
    aligned = align_moves_to_snapshot_pairs(snapshots, moves, accepted_only=True)

    rows: List[AuditRow] = []
    for move, pre, post in aligned:
        if move_type_filter and move.move_type not in move_type_filter:
            continue

        region_nodes, seed_nodes, region_notes = build_local_region(
            move=move,
            pre=pre,
            post=post,
            radius=radius,
            sigma_thresh=sigma_thresh,
            edge_thresh=edge_thresh,
        )

        if not region_nodes:
            region_nodes = seed_nodes if seed_nodes else [0]

        w_func, p_func_retained, notes_func = compute_w_func(
            move=move,
            pre=pre,
            post=post,
            idx=region_nodes,
            sigma_thresh=sigma_thresh,
            edge_thresh=edge_thresh,
        )
        w_book, p_book_retained, endpoint_overlap, top_j, notes_book = compute_w_book(
            move=move,
            pre=pre,
            post=post,
            idx=region_nodes,
            edge_thresh=edge_thresh,
        )
        w_slack, s_pre, s_post, delta_slack_pos, notes_slack = compute_w_slack(
            pre=pre,
            post=post,
            idx=region_nodes,
            sigma_thresh=sigma_thresh,
            edge_thresh=edge_thresh,
        )

        classifier = classify_move(
            w_func=w_func,
            w_book=w_book,
            w_slack=w_slack,
            eps_func=eps_func,
            eps_book=eps_book,
            eps_slack=eps_slack,
            unresolved_margin=unresolved_margin,
        )

        pre_sig = subvector(pre.sigma, region_nodes)
        post_sig = subvector(post.sigma, region_nodes)
        preW = submatrix(pre.W, region_nodes)
        postW = submatrix(post.W, region_nodes)

        active_ret = active_node_retention(pre, post, region_nodes, sigma_thresh=sigma_thresh, edge_thresh=edge_thresh)

        raw_expr_delta = extract_expr_delta(
            move.raw,
            keys=("dE_expr_raw", "expr_raw", "delta_expr_raw", "d_expr_raw")
        )
        total_expr_delta = extract_expr_delta(
            move.raw,
            keys=("dE_expr", "expr", "delta_expr", "d_expr")
        )

        proxy_notes = region_notes + notes_func + notes_book + notes_slack
        confidence = move.confidence
        if not move.nodes_hint:
            confidence = "medium" if seed_nodes else "low"

        rows.append(
            AuditRow(
                move_index=move.idx,
                move_step=move.step,
                move_type=move.move_type,
                accepted=move.accepted,
                pre_step=pre.step,
                post_step=post.step,
                region_nodes=region_nodes,
                region_size=len(region_nodes),
                seed_nodes=seed_nodes,

                w_func=w_func,
                w_book=w_book,
                w_slack=w_slack,

                p_func_retained=p_func_retained,
                p_book_retained=p_book_retained,
                slack_pre=s_pre,
                slack_post=s_post,
                delta_slack_pos=delta_slack_pos,

                classifier=classifier,
                confidence=confidence,
                proxy_notes=proxy_notes,

                local_sigma_l1=sigma_l1(pre_sig, post_sig),
                local_edge_l1=edge_weight_l1(preW, postW),
                endpoint_overlap=endpoint_overlap,
                top_interface_jaccard=top_j,
                active_node_retention=active_ret,

                raw_expr_delta=raw_expr_delta,
                total_expr_delta=total_expr_delta,
                notes=[],
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Output and summary
# ---------------------------------------------------------------------------

def write_json_rows(path: str, rows: List[AuditRow], args_dict: Dict[str, Any]) -> None:
    payload = {
        "script": "hsf_mesoscape_local_move_witness_audit.py",
        "args": args_dict,
        "row_count": len(rows),
        "rows": [asdict(r) for r in rows],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def write_csv_rows(path: str, rows: List[AuditRow]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["move_index", "move_type", "classifier"])
        return

    fieldnames = [
        "move_index", "move_step", "move_type", "accepted",
        "pre_step", "post_step", "region_nodes", "region_size", "seed_nodes",
        "w_func", "w_book", "w_slack",
        "p_func_retained", "p_book_retained",
        "slack_pre", "slack_post", "delta_slack_pos",
        "classifier", "confidence",
        "local_sigma_l1", "local_edge_l1",
        "endpoint_overlap", "top_interface_jaccard", "active_node_retention",
        "raw_expr_delta", "total_expr_delta",
        "proxy_notes", "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            d = asdict(r)
            d["region_nodes"] = json.dumps(d["region_nodes"])
            d["seed_nodes"] = json.dumps(d["seed_nodes"])
            d["proxy_notes"] = json.dumps(d["proxy_notes"])
            d["notes"] = json.dumps(d["notes"])
            writer.writerow(d)


def summarize_rows(rows: List[AuditRow]) -> str:
    if not rows:
        return "No audit rows produced."

    by_class = defaultdict(int)
    by_type = defaultdict(int)
    by_type_class = defaultdict(lambda: defaultdict(int))

    for r in rows:
        by_class[r.classifier] += 1
        by_type[r.move_type] += 1
        by_type_class[r.move_type][r.classifier] += 1

    def mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    lines: List[str] = []
    lines.append("=== HSF local move witness audit ===")
    lines.append(f"rows: {len(rows)}")
    lines.append("")
    lines.append("Classifier counts:")
    for k in sorted(by_class.keys()):
        lines.append(f"  {k}: {by_class[k]}")
    lines.append("")
    lines.append("Mean witness scores:")
    lines.append(f"  W_func : {mean([r.w_func for r in rows]):.4f}")
    lines.append(f"  W_book : {mean([r.w_book for r in rows]):.4f}")
    lines.append(f"  W_slack: {mean([r.w_slack for r in rows]):.4f}")
    lines.append("")
    lines.append("By move type:")
    for mt in sorted(by_type.keys()):
        lines.append(f"  {mt}: {by_type[mt]}")
        for cls in sorted(by_type_class[mt].keys()):
            lines.append(f"    {cls}: {by_type_class[mt][cls]}")
    lines.append("")

    # A few useful diagnostic means by class
    for cls in ("lawful_reexpression", "refolding_like", "unresolved"):
        subset = [r for r in rows if r.classifier == cls]
        if subset:
            lines.append(f"{cls} means:")
            lines.append(f"  W_func : {mean([r.w_func for r in subset]):.4f}")
            lines.append(f"  W_book : {mean([r.w_book for r in subset]):.4f}")
            lines.append(f"  W_slack: {mean([r.w_slack for r in subset]):.4f}")
            red = [r.raw_expr_delta for r in subset if r.raw_expr_delta is not None]
            ted = [r.total_expr_delta for r in subset if r.total_expr_delta is not None]
            if red:
                lines.append(f"  dE_expr_raw mean: {mean(red):.4f}")
            if ted:
                lines.append(f"  dE_expr mean    : {mean(ted):.4f}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit mesoscape local moves using the minimal no-refolding witness bundle."
    )
    p.add_argument("json_path", help="Path to the mesoscape run JSON")
    p.add_argument("--json-out", default="", help="Optional output JSON path")
    p.add_argument("--csv-out", default="", help="Optional output CSV path")
    p.add_argument(
        "--move-types",
        nargs="*",
        default=[],
        help="Optional move types to include, e.g. raise_support edge_up proto birth weaken retire",
    )
    p.add_argument("--radius", type=int, default=1, help="Neighborhood radius for local region R")
    p.add_argument("--sigma-thresh", type=float, default=0.10, help="Support-active threshold")
    p.add_argument("--edge-thresh", type=float, default=0.05, help="Committed-edge threshold")
    p.add_argument("--eps-func", type=float, default=0.35, help="Lawful threshold for W_func")
    p.add_argument("--eps-book", type=float, default=0.35, help="Lawful threshold for W_book")
    p.add_argument("--eps-slack", type=float, default=0.20, help="Lawful threshold for W_slack")
    p.add_argument(
        "--unresolved-margin",
        type=float,
        default=0.10,
        help="Margin beyond threshold before classifying as definitely refolding-like",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    root = read_json(args.json_path)
    move_type_filter = set(args.move_types) if args.move_types else None

    rows = audit_run(
        root=root,
        move_type_filter=move_type_filter,
        radius=int(args.radius),
        sigma_thresh=float(args.sigma_thresh),
        edge_thresh=float(args.edge_thresh),
        eps_func=float(args.eps_func),
        eps_book=float(args.eps_book),
        eps_slack=float(args.eps_slack),
        unresolved_margin=float(args.unresolved_margin),
    )

    summary = summarize_rows(rows)
    print(summary)

    if args.json_out:
        write_json_rows(args.json_out, rows, args_dict=vars(args))
        print(f"\nWrote JSON audit: {args.json_out}")

    if args.csv_out:
        write_csv_rows(args.csv_out, rows)
        print(f"Wrote CSV audit:  {args.csv_out}")


if __name__ == "__main__":
    main()