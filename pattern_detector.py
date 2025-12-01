#!/usr/bin/env python3
"""
pattern_detector.py

Pattern detectors AND scan experiment for emergent "lumps" in the Hilbert substrate.

Ontology:
  - There are no fundamental particles.
  - The Substrate is a big Hilbert sea with local DOFs and entanglement.
  - "Proton / electron / photon" are labels for *patterns* in this sea:
      proton-like  : heavy, strongly entangled, localized composite
      electron-like: lighter, localized, less entangled / less redundant
      photon-like  : bosonic, delocalized, *copyable* info:
                     many sites carry very similar pointer-basis distributions.

Updated for new substrate.py API:
  - Coupling matrix J replaces edge_list
  - Node indices are integers; node IDs are strings like "n001"
  - substrate._neighbors is list[list[int]]
  - substrate.d replaces substrate.dim

This file now serves two roles:

  1) Library:
       - compute_node_features(substrate)
       - compute_species_scores(features)
       - pick_species_candidates(scores)
       - analyze_substrate(substrate)

  2) Experiment script:
       - When run as __main__, it builds a Substrate,
         evolves it, and periodically scans for
         proton/electron/photon-like patterns,
         logging results to:

           outputs/pattern_detector_scan/<run_id>/
             params.json
             summary.json
             data/candidates.csv
             data/frames/*.npz  (optional)

Usage example (Windows, from engines directory):

  python pattern_detector.py ^
      --n-nodes 64 ^
      --internal-dim 4 ^
      --steps 4000 ^
      --record-stride 10 ^
      --defrag-rate 0.05 ^
      --connectivity 0.2 ^
      --dt 0.05 ^
      --seed 42 ^
      --output-root outputs ^
      --tag proton_electron_photon_scan ^
      --use-gpu ^
      --save-frames

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np

from substrate import Config, Substrate  # type: ignore


# ---------------------------------------------------------------------
# Helpers for backend conversion
# ---------------------------------------------------------------------

try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False


def _to_numpy(x) -> np.ndarray:
    """Convert xp array (numpy or cupy) to NumPy ndarray."""
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class NodeFeatures:
    """Per-node features extracted from a single Substrate snapshot."""
    degree: np.ndarray             # (N,)
    clustering: np.ndarray         # (N,)
    total_entanglement: np.ndarray # (N,)
    localization: np.ndarray       # (N,)
    coherence: np.ndarray          # (N,)
    excitation: np.ndarray         # (N,)
    redundancy: np.ndarray         # (N,) "copyability" of pointer distributions


@dataclass
class SpeciesScores:
    """Per-node scores for each emergent species."""
    proton_score: np.ndarray    # (N,)
    electron_score: np.ndarray  # (N,)
    photon_score: np.ndarray    # (N,)


@dataclass
class SpeciesCandidates:
    """Best candidate ids and scores for each emergent species at a snapshot."""
    proton_id: int
    proton_score: float
    electron_id: int
    electron_score: float
    photon_id: int
    photon_score: float


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def compute_node_features(substrate: Substrate) -> NodeFeatures:
    """
    Compute per-node features from a Substrate snapshot.

    Features:
      - degree        : number of neighbors
      - clustering    : simple local clustering coefficient
      - total_ent     : sum of |J_ij| for neighbors j
      - localization  : 1 - S(|psi|^2)/log(d)
      - coherence     : sum |rho_ij|, i != j, for pure state rho = |psi><psi|
      - excitation    : 1 - |<psi | psi_avg>|^2   (psi_avg over all nodes)
      - redundancy    : "copyability" of pointer distributions:
                        mean Bhattacharyya similarity of |psi|^2 to others'
    """
    n_nodes = substrate.n_nodes
    d = substrate.d

    if n_nodes == 0:
        raise ValueError("Substrate has no nodes; cannot compute features.")

    # Get coupling matrix as numpy
    J = _to_numpy(substrate.couplings)
    neighbors = substrate._neighbors  # list[list[int]]

    # --- degree ---
    degree = np.array([len(neighbors[i]) for i in range(n_nodes)], dtype=float)

    # --- clustering coefficient (triangle density) ---
    clustering = np.zeros(n_nodes, dtype=float)
    for i in range(n_nodes):
        nbs = neighbors[i]
        k = len(nbs)
        if k < 2:
            clustering[i] = 0.0
            continue
        nb_set = set(nbs)
        edges_between = 0
        for j_idx in range(k):
            j = nbs[j_idx]
            for k_idx in range(j_idx + 1, k):
                m = nbs[k_idx]
                if m in neighbors[j]:
                    edges_between += 1
        clustering[i] = 2.0 * edges_between / (k * (k - 1))

    # --- total entanglement (sum of |J_ij| over neighbors) ---
    total_entanglement = np.sum(np.abs(J), axis=1)

    # --- states (as numpy) ---
    states = _to_numpy(substrate.states).astype(np.complex128)

    # Global average state for excitation calculation
    psi_avg = states.mean(axis=0)
    nrm_avg = np.linalg.norm(psi_avg)
    if nrm_avg > 0:
        psi_avg = psi_avg / nrm_avg

    localization = np.zeros(n_nodes, dtype=float)
    coherence = np.zeros(n_nodes, dtype=float)
    excitation = np.zeros(n_nodes, dtype=float)

    logd = np.log(d) if d > 1 else 1.0

    # Classical probability distributions for redundancy
    probs = np.abs(states) ** 2
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    probs = np.clip(probs, 1e-12, 1.0)

    for i in range(n_nodes):
        psi = states[i]
        p = probs[i]

        # Entropy / localization
        S = float(-np.sum(p * np.log(p)))
        localization[i] = 1.0 - S / logd

        # Coherence: off-diagonal magnitude of rho = |psi><psi|
        rho = np.outer(psi, np.conjugate(psi))
        off = rho - np.diag(np.diag(rho))
        coherence[i] = float(np.sum(np.abs(off)))

        # Excitation relative to global average pattern
        ov = np.vdot(psi, psi_avg)
        excitation[i] = float(1.0 - np.abs(ov) ** 2)

    # --- redundancy / copyability ---
    # Bhattacharyya coefficients: BC(i,j) = sum_k sqrt(p_i[k] p_j[k])
    sqrt_p = np.sqrt(probs)
    bc_matrix = sqrt_p @ sqrt_p.T  # (N, N)
    np.fill_diagonal(bc_matrix, 0.0)
    redundancy = bc_matrix.sum(axis=1) / np.maximum(n_nodes - 1, 1)

    return NodeFeatures(
        degree=degree,
        clustering=clustering,
        total_entanglement=total_entanglement,
        localization=localization,
        coherence=coherence,
        excitation=excitation,
        redundancy=redundancy,
    )


# ---------------------------------------------------------------------
# Species scoring
# ---------------------------------------------------------------------

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma


def compute_species_scores(features: NodeFeatures) -> SpeciesScores:
    """
    Build heuristic scores for proton-like, electron-like, photon-like patterns.

    We normalize features across nodes and then use simple linear combinations:

      proton_score  ~ +deg + ent + loc - small*excitation
      electron_score~ +loc + excitation - 0.5*deg - 0.5*ent - 0.5*redundancy
      photon_score  ~ -loc + coherence + 0.5*ent + 1.5*redundancy

    Intuition:
      - Proton: heavy, strongly entangled, localized, not too exotic.
      - Electron: localized & excited but NOT highly entangled / high-degree /
                  highly redundant.
      - Photon: bosonic, delocalized, coherent, *copyable* (high redundancy).
    """
    z_deg = _zscore(features.degree)
    z_ent = _zscore(features.total_entanglement)
    z_loc = _zscore(features.localization)
    z_coh = _zscore(features.coherence)
    z_exc = _zscore(features.excitation)
    z_red = _zscore(features.redundancy)

    # Proton: heavy, strongly entangled, localized, modest excitation
    proton_score = (
        1.0 * z_deg +
        1.0 * z_ent +
        1.0 * z_loc -
        0.2 * z_exc
    )

    # Electron: localized & excited, but not super-entangled or highly redundant
    electron_score = (
        1.0 * z_loc +
        1.0 * z_exc -
        0.5 * z_deg -
        0.5 * z_ent -
        0.5 * z_red
    )

    # Photon: delocalized, coherent, somewhat entangled, highly redundant
    photon_score = (
        -1.0 * z_loc +
        1.0 * z_coh +
        0.5 * z_ent +
        1.5 * z_red
    )

    return SpeciesScores(
        proton_score=proton_score,
        electron_score=electron_score,
        photon_score=photon_score,
    )


def pick_species_candidates(scores: SpeciesScores) -> SpeciesCandidates:
    """
    Choose best candidate ids and scores for each species from per-node scores.
    Returns integer node indices (not string IDs).
    """
    p_idx = int(np.argmax(scores.proton_score))
    e_idx = int(np.argmax(scores.electron_score))
    ph_idx = int(np.argmax(scores.photon_score))

    return SpeciesCandidates(
        proton_id=p_idx,
        proton_score=float(scores.proton_score[p_idx]),
        electron_id=e_idx,
        electron_score=float(scores.electron_score[e_idx]),
        photon_id=ph_idx,
        photon_score=float(scores.photon_score[ph_idx]),
    )


# ---------------------------------------------------------------------
# High-level analysis entry point (library API)
# ---------------------------------------------------------------------

def analyze_substrate(
    substrate: Substrate,
) -> Tuple[NodeFeatures, SpeciesScores, SpeciesCandidates]:
    """
    Analyze a Substrate snapshot and return:

      - NodeFeatures      : per-node feature arrays
      - SpeciesScores     : per-node proton/electron/photon heuristic scores
      - SpeciesCandidates : best candidate ids & scores (integer indices)

    Use this as a high-level call from experiments or from the CLI driver below.
    """
    feats = compute_node_features(substrate)
    scores = compute_species_scores(feats)
    cands = pick_species_candidates(scores)
    return feats, scores, cands


# ---------------------------------------------------------------------
# Experiment driver (when run as a script)
# ---------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_run_id(tag: str | None) -> str:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if tag:
        safe_tag = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(tag)
        )
        return f"{now}_{safe_tag}"
    return now


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan a Substrate for proton/electron/photon-like patterns "
            "using Hilbert-space pattern detectors."
        )
    )

    # Substrate / dynamics config
    p.add_argument("--n-nodes", type=int, default=64, help="Number of nodes.")
    p.add_argument(
        "--internal-dim", type=int, default=4, help="Local Hilbert dimension d."
    )
    p.add_argument(
        "--monogamy-budget",
        type=float,
        default=1.0,
        help="Target row-sum of |J_ij| for each node.",
    )
    p.add_argument(
        "--defrag-rate",
        type=float,
        default=0.05,
        help="Defrag rate used in substrate defrag_step.",
    )
    p.add_argument(
        "--connectivity",
        type=float,
        default=0.2,
        help="Initial Bernoulli connectivity probability.",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Time step for the substrate integrator.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Total number of evolution steps to run.",
    )
    p.add_argument(
        "--record-stride",
        type=int,
        default=20,
        help="How many steps between particle-scan snapshots.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base random seed for the substrate.",
    )
    p.add_argument(
        "--use-gpu",
        action="store_true",
        help="If set, request CuPy backend when available.",
    )

    # Output config
    p.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for experiment outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="pattern_detector_scan",
        help="Tag to include in the run_id (for easier grouping).",
    )
    p.add_argument(
        "--save-frames",
        action="store_true",
        help="If set, saves per-snapshot feature arrays into data/frames/*.npz.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ------------------- run directory -------------------
    run_id = _make_run_id(args.tag)
    run_root = os.path.join(args.output_root, "pattern_detector_scan", run_id)

    data_dir = os.path.join(run_root, "data")
    frames_dir = os.path.join(data_dir, "frames")
    logs_dir = os.path.join(run_root, "logs")

    _ensure_dir(run_root)
    _ensure_dir(data_dir)
    _ensure_dir(logs_dir)
    if args.save_frames:
        _ensure_dir(frames_dir)

    # ------------------- substrate config -------------------
    cfg = Config(
        n_nodes=args.n_nodes,
        internal_dim=args.internal_dim,
        monogamy_budget=args.monogamy_budget,
        defrag_rate=args.defrag_rate,
        dt=args.dt,
        seed=args.seed,
        connectivity=args.connectivity,
        use_gpu=True if args.use_gpu else False,
    )

    sub = Substrate(cfg)

    # Save params.json
    params = {
        "n_nodes": args.n_nodes,
        "internal_dim": args.internal_dim,
        "monogamy_budget": args.monogamy_budget,
        "defrag_rate": args.defrag_rate,
        "connectivity": args.connectivity,
        "dt": args.dt,
        "steps": args.steps,
        "record_stride": args.record_stride,
        "seed": args.seed,
        "use_gpu": args.use_gpu,
        "run_id": run_id,
    }
    _write_json(os.path.join(run_root, "params.json"), params)

    # Prepare candidates.csv
    cand_path = os.path.join(data_dir, "candidates.csv")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write(
            "step,t,"
            "proton_id,proton_score,"
            "electron_id,electron_score,"
            "photon_id,photon_score\n"
        )

    print("=" * 60)
    print("  Hilbert Pattern Detector Scan")
    print("=" * 60)
    print(f"Run ID:      {run_id}")
    print(f"Output root: {run_root}")
    print(f"Nodes:       {args.n_nodes}, d={args.internal_dim}")
    print(f"Steps:       {args.steps}, record_stride={args.record_stride}")
    print(f"Defrag rate: {args.defrag_rate}, connectivity={args.connectivity}")
    print(f"Seed:        {args.seed}")
    print("=" * 60)
    sys.stdout.flush()

    last_feats = None
    last_scores = None
    last_cands = None

    # ------------------- evolution loop -------------------
    for step in range(args.steps + 1):
        t = step * args.dt

        # Record before evolving at step 0, step=record_stride, ...
        if step % args.record_stride == 0:
            feats, scores, cands = analyze_substrate(sub)

            last_feats = feats
            last_scores = scores
            last_cands = cands

            # Append one row to candidates.csv
            with open(cand_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{step},{t:.8f},"
                    f"{cands.proton_id},{cands.proton_score:.6f},"
                    f"{cands.electron_id},{cands.electron_score:.6f},"
                    f"{cands.photon_id},{cands.photon_score:.6f}\n"
                )

            print(
                f"[step {step:6d} | t={t:8.4f}]  "
                f"p: id={cands.proton_id:3d}, score={cands.proton_score:8.4f}  "
                f"e: id={cands.electron_id:3d}, score={cands.electron_score:8.4f}  "
                f"ph: id={cands.photon_id:3d}, score={cands.photon_score:8.4f}"
            )
            sys.stdout.flush()

            # Optional: save full features/scores snapshot
            if args.save_frames:
                frame_name = f"frame_step_{step:06d}.npz"
                frame_path = os.path.join(frames_dir, frame_name)
                np.savez_compressed(
                    frame_path,
                    step=step,
                    t=t,
                    degree=feats.degree,
                    clustering=feats.clustering,
                    total_entanglement=feats.total_entanglement,
                    localization=feats.localization,
                    coherence=feats.coherence,
                    excitation=feats.excitation,
                    redundancy=feats.redundancy,
                    proton_score=scores.proton_score,
                    electron_score=scores.electron_score,
                    photon_score=scores.photon_score,
                )

        if step >= args.steps:
            break

        # Single evolution step
        sub.evolve(n_steps=1)

    # ------------------- summary.json -------------------
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "steps": args.steps,
        "dt": args.dt,
        "defrag_rate": args.defrag_rate,
        "connectivity": args.connectivity,
    }

    if last_cands is not None:
        summary["final_candidates"] = {
            "proton": {
                "node_id": int(last_cands.proton_id),
                "score": float(last_cands.proton_score),
            },
            "electron": {
                "node_id": int(last_cands.electron_id),
                "score": float(last_cands.electron_score),
            },
            "photon": {
                "node_id": int(last_cands.photon_id),
                "score": float(last_cands.photon_score),
            },
        }

    _write_json(os.path.join(run_root, "summary.json"), summary)

    print("=" * 60)
    print("Scan complete.")
    print(f"Candidate time series: {cand_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
