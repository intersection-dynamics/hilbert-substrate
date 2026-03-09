"""
E2 Stage 1: Spatial Locality Emergence (v2 — JSON + corrected baselines)
======================================================================
Self-contained demonstration that "spatial locality" can emerge as a
dynamical attractor under a double-bracket (Brockett) flow that decreases
a Pauli-weight locality cost C_p (Paper II / HSF).

What this script actually measures
---------------------------------
We minimize a Pauli-weight cost over the *unitary orbit* of a Hamiltonian H:

  C_p(H) = (Σ_k w(P_k)^p |c_k|^2) / (Σ_k |c_k|^2)

where H = Σ_k c_k P_k in the N-qubit Pauli basis and w(P_k) is Hamming weight
(# of non-identity factors in P_k).

Important clarification:
------------------------
The previous version printed a "Harmonion ideal" value computed as:
  - take eigenvalues of H and place them on the diagonal in the *computational basis*
This is only a *reference baseline*, not a proven global minimum of C_p over the
unitary orbit. In v2 we label it correctly as "diag-spectrum baseline".

This script now:
  - Saves all results to JSON in a standard output directory
  - Fixes regime/phase labels to be based on proximity to the spatial target
  - Records per-trial metadata (seed, elapsed time, steps, final weight distribution)
  - Keeps the N=2 validation that the flow descends reliably

Run (Windows):
  python emergence_e2_stage1.py

Output:
  hsf_out\emergence_e2_stage1\YYYYMMDD_HHMMSS\stage1_results.json
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product as iprod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ============================================================
# Output utilities
# ============================================================

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_run_dir(base_dir: str = "hsf_out", run_name: str = "emergence_e2_stage1") -> Path:
    """
    Create a standard run directory:
      <base_dir>/<run_name>/<timestamp>/
    """
    run_dir = Path(base_dir) / run_name / _now_tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def json_safe(o):
    """Convert numpy types to JSON-serializable python types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [json_safe(x) for x in o]
    return o

def decimate_trace(trace: List[float], max_points: int = 300) -> List[float]:
    """Keep at most max_points points from a trace (uniform subsampling)."""
    if len(trace) <= max_points:
        return [float(x) for x in trace]
    idx = np.linspace(0, len(trace) - 1, max_points).astype(int)
    return [float(trace[i]) for i in idx]


# ============================================================
# Pauli infrastructure
# ============================================================

I2 = np.eye(2, dtype=complex)
PAULIS = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def pauli_tensor(label: str) -> np.ndarray:
    """Build N-qubit Pauli tensor product from string like 'XIZI'."""
    result = PAULIS[label[0]]
    for ch in label[1:]:
        result = np.kron(result, PAULIS[ch])
    return result

def all_pauli_labels(N: int) -> List[str]:
    """Generate all 4^N Pauli labels for N qubits."""
    return ["".join(s) for s in iprod("IXYZ", repeat=N)]

def hamming_weight(label: str) -> int:
    """Number of non-identity positions in Pauli label."""
    return sum(1 for ch in label if ch != "I")

def decompose_pauli(H: np.ndarray, N: int) -> Dict[str, float]:
    """
    Decompose Hermitian H into Pauli basis: H = sum_k c_k P_k.
    Returns dict {label: coefficient}. Coeffs are real for Hermitian H.
    """
    dim = 2**N
    coeffs: Dict[str, float] = {}
    for label in all_pauli_labels(N):
        P = pauli_tensor(label)
        c = (np.trace(H @ P).real) / dim
        if abs(c) > 1e-15:
            coeffs[label] = float(c)
    return coeffs


# ============================================================
# Locality cost and gradient
# ============================================================

def locality_cost_fast(coeffs: Dict[str, float], p: float) -> float:
    """Compute C_p from pre-computed Pauli coefficients."""
    num = 0.0
    den = 0.0
    for label, c in coeffs.items():
        w = hamming_weight(label)
        c2 = c * c
        den += c2
        num += (w**p) * c2
    return (num / den) if den > 0 else 0.0

def locality_cost(H: np.ndarray, N: int, p: float) -> float:
    """C_p(H) = sum_k w(P_k)^p |c_k|^2 / sum_k |c_k|^2"""
    return locality_cost_fast(decompose_pauli(H, N), p)

def weight_distribution(coeffs: Dict[str, float], N: int) -> Dict[int, float]:
    """Fraction of Frobenius norm at each Hamming weight."""
    weight_power: Dict[int, float] = {}
    total = 0.0
    for label, c in coeffs.items():
        w = hamming_weight(label)
        weight_power[w] = weight_power.get(w, 0.0) + c * c
        total += c * c
    if total == 0:
        return {}
    return {w: float(weight_power.get(w, 0.0) / total) for w in range(N + 1)}

def gradient_operator_M(coeffs: Dict[str, float], N: int, p: float) -> np.ndarray:
    """
    Build gradient operator:
      M = sum_k w(P_k)^p * c_k * P_k

    The double-bracket flow dH/dt = [[H,M], H] decreases C_p monotonically.
    """
    dim = 2**N
    M = np.zeros((dim, dim), dtype=complex)
    for label, c in coeffs.items():
        w = hamming_weight(label)
        if w > 0:  # identity term doesn't affect gradient direction
            M += (w**p) * c * pauli_tensor(label)
    return M

def spatial_cost(N: int, p: float) -> float:
    """Cost of nearest-neighbor two-body terms (weight 2)."""
    return float(2**p)

def diag_spectrum_baseline_cost(H: np.ndarray, N: int, p: float) -> float:
    """
    Baseline reference (NOT a proven global minimum):
      - take eigenvalues of H
      - place them on the diagonal in the computational basis
      - compute C_p of that diagonal matrix
    """
    evals = np.linalg.eigvalsh(H)
    H_diag = np.diag(evals)
    return locality_cost(H_diag, N, p)


# ============================================================
# Hamiltonian construction and scrambling
# ============================================================

def heisenberg_ring(N: int, J: float = 1.0) -> np.ndarray:
    """Heisenberg XXX chain with periodic boundaries (ring)."""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        for pauli in ["X", "Y", "Z"]:
            label = ["I"] * N
            label[i] = pauli
            label[j] = pauli
            H += J * pauli_tensor("".join(label))
    return H

def random_unitary(dim: int) -> np.ndarray:
    """Haar-random unitary via QR decomposition."""
    Z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    D = np.diag(R)
    Ph = np.diag(D / np.abs(D))
    return Q @ Ph

def scramble(H: np.ndarray, N: int, depth: int) -> np.ndarray:
    """
    Scramble Hamiltonian via random 2-qubit brickwork circuit.
    Applies `depth` layers of non-overlapping 2-qubit Haar gates.
    """
    dim = 2**N
    U_total = np.eye(dim, dtype=complex)

    for layer in range(depth):
        offset = layer % 2
        for i in range(offset, N - 1, 2):
            u2 = random_unitary(4)
            left = np.eye(2**i, dtype=complex) if i > 0 else np.array([[1.0 + 0j]])
            right_dim = N - i - 2
            right = np.eye(2**right_dim, dtype=complex) if right_dim > 0 else np.array([[1.0 + 0j]])
            U_layer = np.kron(np.kron(left, u2), right)
            U_total = U_layer @ U_total

    return U_total @ H @ U_total.conj().T


# ============================================================
# Double-bracket flow with robust line search
# ============================================================

def double_bracket_step(H: np.ndarray, K: np.ndarray, dt: float) -> np.ndarray:
    """
    Isospectral update:
      H' = exp(dt*K) H exp(-dt*K)
    with K = [H, M] anti-Hermitian.
    """
    U = expm(dt * K)
    return U @ H @ U.conj().T

@dataclass
class FlowResult:
    final_H: np.ndarray
    trace: List[float]
    final_coeffs: Dict[str, float]
    final_weight_dist: Dict[int, float]
    steps: int
    elapsed_sec: float
    stall_reason: str

def run_flow(
    H0: np.ndarray,
    N: int,
    p: float,
    max_iter: int = 500,
    tol: float = 1e-8,
    verbose: bool = True,
) -> FlowResult:
    """
    Double-bracket flow to minimize locality cost C_p(H).

    Generator:
      M = Σ w^p c_k P_k
      K = [H, M]
    Step:
      H <- exp(dt*K) H exp(-dt*K)

    Step size selection:
      dt_base ~ 1/||K|| with a small candidate set of scales; pick best decrease.
    """
    H = H0.copy()
    t0 = time.time()

    coeffs = decompose_pauli(H, N)
    cost = locality_cost_fast(coeffs, p)
    trace = [float(cost)]

    if verbose:
        wd = weight_distribution(coeffs, N)
        wd_str = ", ".join(f"w{w}:{f:.3f}" for w, f in sorted(wd.items()) if f > 0.001)
        print(f"    step 0: cost={cost:.4f}  [{wd_str}]")

    stall_count = 0
    stall_reason = "max_iter"

    # Candidate multipliers (ordered): prefer sensible dt, then try larger, then very small
    scales = [1.0, 0.5, 0.25, 0.1, 0.05, 2.0, 4.0, 0.01, 0.005, 0.001]

    for it in range(1, max_iter + 1):
        M = gradient_operator_M(coeffs, N, p)
        K = H @ M - M @ H
        K_norm = np.linalg.norm(K, "fro")

        if K_norm < 1e-14:
            stall_reason = "K_zero_fixed_point"
            if verbose:
                print(f"    step {it}: CONVERGED (K≈0, fixed point)")
            break

        dt_base = 1.0 / K_norm

        best_cost = cost
        best_H = None
        best_dt = 0.0
        best_coeffs = None

        for s in scales:
            dt = dt_base * s
            H_try = double_bracket_step(H, K, dt)
            coeffs_try = decompose_pauli(H_try, N)
            cost_try = locality_cost_fast(coeffs_try, p)
            if cost_try < best_cost:
                best_cost = cost_try
                best_H = H_try
                best_dt = dt
                best_coeffs = coeffs_try

        if best_dt == 0.0:
            stall_count += 1
            if stall_count >= 10:
                stall_reason = "line_search_stalled_10"
                if verbose:
                    print(f"    step {it}: STALLED after 10 failures (||K||={K_norm:.2e})")
                break
            continue
        else:
            stall_count = 0

        improvement = cost - best_cost
        H = best_H
        cost = best_cost
        coeffs = best_coeffs
        trace.append(float(cost))

        if verbose and (it <= 5 or it % 50 == 0):
            print(f"    step {it}: cost={cost:.4f}, ||K||={K_norm:.2e}, dt={best_dt:.2e}, imp={improvement:.2e}")

        if 0 < improvement < tol:
            stall_reason = "tol_reached"
            if verbose:
                print(f"    step {it}: CONVERGED (improvement < {tol:.0e}), cost={cost:.4f}")
            break

    elapsed = time.time() - t0
    wd_final = weight_distribution(coeffs, N)

    if verbose:
        wd_str = ", ".join(f"w{w}:{f:.3f}" for w, f in sorted(wd_final.items()) if f > 0.001)
        print(f"    FINAL: cost={cost:.4f}  [{wd_str}]  ({len(trace)} steps)")

    return FlowResult(
        final_H=H,
        trace=trace,
        final_coeffs=coeffs,
        final_weight_dist=wd_final,
        steps=len(trace),
        elapsed_sec=float(elapsed),
        stall_reason=stall_reason,
    )


# ============================================================
# Classification helpers (fixed)
# ============================================================

def classify_by_spatial(mean_cost: float, spatial_target: float) -> str:
    """
    Classify using proximity to the spatial target.
      - near spatial: emergent geometry basin
      - far above spatial: glassy / stuck high
      - otherwise: fluid / reorganizing (not trapped near spatial)
    """
    if spatial_target <= 0:
        return "Unknown"
    ratio = mean_cost / spatial_target
    if ratio <= 1.25:
        return "Emergent Geometry (near spatial)"
    if ratio >= 2.0:
        return "Glassy / stuck high"
    return "Quantum Fluid / reorganizing"

def detect_bimodal(recovered: List[float]) -> bool:
    """Heuristic: very wide spread suggests multiple attractors."""
    if len(recovered) < 3:
        return False
    r = np.array(recovered, dtype=float)
    spread = (r.max() - r.min()) / (r.mean() + 1e-12)
    return bool(spread > 0.6 and r.std() > 0.25 * r.mean())


# ============================================================
# Main run
# ============================================================

def main() -> int:
    run_dir = make_run_dir()
    out_json = run_dir / "stage1_results.json"

    results = {
        "run_info": {
            "timestamp_local": datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": "unknown",
            "cwd": str(Path.cwd()),
            "run_dir": str(run_dir),
        },
        "config": {
            "validation": {"N": 2, "p": 4, "seed": 42, "scramble_depth": 6, "max_iter": 200},
            "exp1": {"p": 4, "Ns": [3, 4, 5], "n_trials": 3, "scramble_depth": "2*N", "max_iter": 400},
            "exp2": {"N": 5, "ps": [1.0, 2.0, 3.0, 4.0, 5.0], "n_trials": 3, "scramble_depth": "2*N", "max_iter": 400},
            "exp3": {"N": 6, "p": 4, "seed": 3000, "scramble_depth": "2*N", "max_iter": 300},
            "trace_max_points": 300,
        },
        "validation": {},
        "experiments": {"exp1": {}, "exp2": {}, "exp3": {}},
    }

    # Try to capture scipy version (best effort)
    try:
        import scipy
        results["run_info"]["scipy"] = scipy.__version__
    except Exception:
        pass

    # ============================================================
    # VALIDATION
    # ============================================================
    print("=" * 70)
    print("VALIDATION: Double-bracket flow on N=2")
    print("  Should descend reliably to the diag-spectrum baseline")
    print("=" * 70)

    N_v = 2
    p_v = 4
    H_v = heisenberg_ring(N_v)
    c_base_v = diag_spectrum_baseline_cost(H_v, N_v, p_v)
    c_spat_v = spatial_cost(N_v, p_v)
    print(f"  Diag-spectrum baseline: {c_base_v:.4f}")
    print(f"  Spatial target:         {c_spat_v:.1f}")

    np.random.seed(42)
    H_scr_v = scramble(H_v, N_v, depth=6)
    c_init_v = locality_cost(H_scr_v, N_v, p_v)
    print(f"  Scrambled cost:         {c_init_v:.4f}")

    coeffs_v = decompose_pauli(H_scr_v, N_v)
    M_v = gradient_operator_M(coeffs_v, N_v, p_v)
    K_v = H_scr_v @ M_v - M_v @ H_scr_v
    print(f"  ||H|| = {np.linalg.norm(H_scr_v, 'fro'):.4f}")
    print(f"  ||M|| = {np.linalg.norm(M_v, 'fro'):.4f}")
    print(f"  ||[H,M]|| = {np.linalg.norm(K_v, 'fro'):.4f}")
    print(f"  Weight dist: {weight_distribution(coeffs_v, N_v)}")
    print()

    flow_v = run_flow(H_scr_v, N_v, p_v, max_iter=200, verbose=True)
    c_final_v = flow_v.trace[-1]

    print(f"\n  Result: {c_init_v:.4f} -> {c_final_v:.4f} (baseline: {c_base_v:.4f})")
    validation_ok = bool(c_final_v <= c_base_v * 1.5)

    results["validation"] = {
        "N": N_v,
        "p": p_v,
        "seed": 42,
        "baseline_cost": float(c_base_v),
        "spatial_cost": float(c_spat_v),
        "init_cost": float(c_init_v),
        "final_cost": float(c_final_v),
        "passed": validation_ok,
        "stall_reason": flow_v.stall_reason,
        "steps": flow_v.steps,
        "elapsed_sec": flow_v.elapsed_sec,
        "final_weight_dist": json_safe(flow_v.final_weight_dist),
        "trace": decimate_trace(flow_v.trace, max_points=results["config"]["trace_max_points"]),
    }

    if validation_ok:
        print("  *** VALIDATION PASSED ***")
    else:
        print("  *** VALIDATION FAILED ***")
        print("  The flow is not descending properly. Fix gradient/step logic before experiments.")
        results["run_info"]["status"] = "FAILED_VALIDATION"
        out_json.write_text(json.dumps(json_safe(results), indent=2))
        print(f"\nSaved JSON: {out_json}")
        return 1

    # ============================================================
    # EXPERIMENT 1
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Accessibility / trapping vs N (p=4)")
    print("=" * 70)

    p = 4.0
    n_trials = 3

    for N in [3, 4, 5]:
        dim = 2**N
        c_spat = spatial_cost(N, p)
        H_local = heisenberg_ring(N)
        c_base = diag_spectrum_baseline_cost(H_local, N, p)

        print(f"\n--- N={N} (dim={dim}, 4^N={4**N} Paulis) ---")
        print(f"    Spatial target:         {c_spat:.1f}")
        print(f"    Diag-spectrum baseline: {c_base:.4f}")

        recovered: List[float] = []
        trials_out = []

        for trial in range(n_trials):
            seed = 1000 + N * 100 + trial
            np.random.seed(seed)

            H_scr = scramble(H_local, N, depth=2 * N)
            c_init = locality_cost(H_scr, N, p)

            print(f"\n  Trial {trial + 1}: scrambled cost = {c_init:.2f}")
            t0 = time.time()
            flow = run_flow(H_scr, N, p, max_iter=400, verbose=True)
            elapsed = time.time() - t0

            c_final = float(flow.trace[-1])
            recovered.append(c_final)

            print(f"  Trial {trial + 1} done: {c_init:.2f} -> {c_final:.4f} ({elapsed:.1f}s)")

            trials_out.append({
                "trial": trial + 1,
                "seed": seed,
                "init_cost": float(c_init),
                "final_cost": c_final,
                "elapsed_sec": float(elapsed),
                "steps": flow.steps,
                "stall_reason": flow.stall_reason,
                "final_weight_dist": json_safe(flow.final_weight_dist),
                "trace": decimate_trace(flow.trace, max_points=results["config"]["trace_max_points"]),
            })

        mean_rec = float(np.mean(recovered))
        std_rec = float(np.std(recovered))
        regime = classify_by_spatial(mean_rec, c_spat)
        if detect_bimodal(recovered):
            regime += " (bimodal-ish)"

        results["experiments"]["exp1"][str(N)] = {
            "N": N,
            "p": float(p),
            "spatial_cost": float(c_spat),
            "diag_spectrum_baseline_cost": float(c_base),
            "recovered": [float(x) for x in recovered],
            "mean": mean_rec,
            "std": std_rec,
            "mean_over_spatial": float(mean_rec / c_spat) if c_spat > 0 else None,
            "regime": regime,
            "trials": trials_out,
        }

    print("\n" + "=" * 70)
    print("EXPERIMENT 1 SUMMARY: Accessibility / trapping vs N (p=4)")
    print("=" * 70)
    print(f"{'N':>3} {'Spatial':>10} {'DiagBase':>10} {'Recovered':>16} {'Mean/Spat':>10} {'Regime':>28}")
    print("-" * 90)
    for N in [3, 4, 5]:
        r = results["experiments"]["exp1"][str(N)]
        rec_str = f"{r['mean']:.2f} +/- {r['std']:.2f}"
        print(f"{N:>3} {r['spatial_cost']:>10.1f} {r['diag_spectrum_baseline_cost']:>10.2f} {rec_str:>16} {r['mean_over_spatial']:>10.2f} {r['regime']:>28}")

    # ============================================================
    # EXPERIMENT 2
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Phase diagram vs p (N=5)")
    print("=" * 70)

    N = 5
    H_local = heisenberg_ring(N)
    n_trials_pd = 3

    for p_val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        c_spat = spatial_cost(N, p_val)
        c_base = diag_spectrum_baseline_cost(H_local, N, p_val)

        print(f"\n--- p={p_val:.1f} (spatial={c_spat:.1f}, diag-spectrum baseline={c_base:.2f}) ---")

        recovered: List[float] = []
        trials_out = []

        for trial in range(n_trials_pd):
            seed = 2000 + int(p_val * 100) + trial
            np.random.seed(seed)

            H_scr = scramble(H_local, N, depth=2 * N)
            c_init = locality_cost(H_scr, N, p_val)

            print(f"  Trial {trial + 1}:")
            flow = run_flow(H_scr, N, p_val, max_iter=400, verbose=True)
            c_final = float(flow.trace[-1])
            recovered.append(c_final)

            trials_out.append({
                "trial": trial + 1,
                "seed": seed,
                "init_cost": float(c_init),
                "final_cost": c_final,
                "elapsed_sec": flow.elapsed_sec,
                "steps": flow.steps,
                "stall_reason": flow.stall_reason,
                "final_weight_dist": json_safe(flow.final_weight_dist),
                "trace": decimate_trace(flow.trace, max_points=results["config"]["trace_max_points"]),
            })

        mean_rec = float(np.mean(recovered))
        std_rec = float(np.std(recovered))
        phase = classify_by_spatial(mean_rec, c_spat)
        if detect_bimodal(recovered):
            phase += " (bimodal-ish)"

        results["experiments"]["exp2"][str(p_val)] = {
            "N": N,
            "p": float(p_val),
            "spatial_cost": float(c_spat),
            "diag_spectrum_baseline_cost": float(c_base),
            "recovered": [float(x) for x in recovered],
            "mean": mean_rec,
            "std": std_rec,
            "mean_over_spatial": float(mean_rec / c_spat) if c_spat > 0 else None,
            "phase": phase,
            "trials": trials_out,
        }

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 SUMMARY: Phase diagram vs p (N=5)")
    print("=" * 70)
    print(f"{'p':>5} {'Spatial':>10} {'DiagBase':>10} {'Recovered':>16} {'Mean/Spat':>10} {'Phase':>28}")
    print("-" * 90)
    for p_val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        r = results["experiments"]["exp2"][str(p_val)]
        rec_str = f"{r['mean']:.2f} +/- {r['std']:.2f}"
        print(f"{p_val:>5.1f} {r['spatial_cost']:>10.1f} {r['diag_spectrum_baseline_cost']:>10.2f} {rec_str:>16} {r['mean_over_spatial']:>10.2f} {r['phase']:>28}")

    # ============================================================
    # EXPERIMENT 3
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Topology universality (N=6, p=4)")
    print("=" * 70)

    N = 6
    p = 4.0
    c_spat = spatial_cost(N, p)

    def heisenberg_graph(N: int, edges: List[Tuple[int, int]], J: float = 1.0) -> np.ndarray:
        """Heisenberg Hamiltonian on arbitrary graph."""
        dim = 2**N
        H = np.zeros((dim, dim), dtype=complex)
        for (i, j) in edges:
            for pauli in ["X", "Y", "Z"]:
                label = ["I"] * N
                label[i] = pauli
                label[j] = pauli
                H += J * pauli_tensor("".join(label))
        return H

    topologies = {
        "1D Ring": [(i, (i + 1) % 6) for i in range(6)],
        "2D Grid 2x3": [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)],
        "Random d=3": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)],
    }

    for name, edges in topologies.items():
        H_graph = heisenberg_graph(N, edges)
        c_base = diag_spectrum_baseline_cost(H_graph, N, p)

        print(f"\n--- {name} ({len(edges)} edges, diag-spectrum baseline={c_base:.2f}) ---")

        seed = 3000
        np.random.seed(seed)
        H_scr = scramble(H_graph, N, depth=2 * N)
        c_init = locality_cost(H_scr, N, p)

        t0 = time.time()
        flow = run_flow(H_scr, N, p, max_iter=300, verbose=True)
        elapsed = time.time() - t0

        final_cost = float(flow.trace[-1])
        trapped = "YES" if (final_cost / c_spat) <= 1.25 else "NO"

        results["experiments"]["exp3"][name] = {
            "topology": name,
            "N": N,
            "p": float(p),
            "n_edges": int(len(edges)),
            "seed": seed,
            "spatial_cost": float(c_spat),
            "diag_spectrum_baseline_cost": float(c_base),
            "init_cost": float(c_init),
            "final_cost": final_cost,
            "final_over_spatial": float(final_cost / c_spat),
            "classified": classify_by_spatial(final_cost, c_spat),
            "trapped_near_spatial": trapped == "YES",
            "elapsed_sec": float(elapsed),
            "steps": flow.steps,
            "stall_reason": flow.stall_reason,
            "final_weight_dist": json_safe(flow.final_weight_dist),
            "trace": decimate_trace(flow.trace, max_points=results["config"]["trace_max_points"]),
        }

        print(f"  -> {c_init:.2f} -> {final_cost:.2f} ({elapsed:.1f}s)  trapped_near_spatial={trapped}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3 SUMMARY: Topology universality (N=6, p=4)")
    print("=" * 70)
    print(f"{'Topology':>15} {'Edges':>6} {'Spatial':>10} {'Recovered':>10} {'Final/Spat':>11} {'NearSpat?':>10}")
    print("-" * 70)
    for name in topologies.keys():
        r = results["experiments"]["exp3"][name]
        print(f"{name:>15} {r['n_edges']:>6} {r['spatial_cost']:>10.1f} {r['final_cost']:>10.2f} {r['final_over_spatial']:>11.2f} {str(r['trapped_near_spatial']):>10}")

    # ============================================================
    # Save JSON
    # ============================================================
    results["run_info"]["status"] = "OK"
    out_json.write_text(json.dumps(json_safe(results), indent=2))
    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETE — results saved")
    print("=" * 70)
    print(f"Saved JSON: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
