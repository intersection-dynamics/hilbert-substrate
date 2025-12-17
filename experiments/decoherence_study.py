"""
decoherence_study.py (clean version)

Comprehensive study of decoherence from multiple angles:

1. SCALING: How does decoherence depend on environment size?
2. COUPLING: How does coupling strength affect timescales?
3. SPREADING: How does information propagate through environment?
4. REVERSIBILITY: Can we undo decoherence? When is it irreversible?
5. POINTER BASIS: What determines which states survive?
6. REDUNDANCY: Is information copied redundantly (Quantum Darwinism)?
7. STRUCTURE: How does environment topology affect decoherence?
8. TIMESCALES: What sets the decoherence time?

Changes vs. the original:
- Fixes the Loschmidt echo off-by-one in the reversal step count.
- Makes Quantum Darwinism sampling robust (no float equality on time).
- Replaces the misleading "fidelity" helper with a correct density-matrix fidelity
  (and a safe fallback if SciPy isn't available).
- Adds a consistent decoherence-time extractor and writes a small run summary
  (CSV + JSON) alongside the plots.

Run (Windows):
    python decoherence_study.py

Outputs:
    decoherence_studies\study1_env_scaling.png
    ...
    decoherence_studies\study8_timescales.png
    decoherence_studies\decoherence_summary.csv
    decoherence_studies\decoherence_summary.json
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Optional (used for correct Uhlmann fidelity). If unavailable, we fall back safely.
try:
    from scipy.linalg import sqrtm as _sqrtm  # type: ignore
except Exception:  # pragma: no cover
    _sqrtm = None  # type: ignore

# =============================================================================
# Basic quantum utilities
# =============================================================================

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def tensor(*ops: np.ndarray) -> np.ndarray:
    """Kronecker product of a list of operators."""
    out = np.array([[1.0]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def partial_trace(rho: np.ndarray, dims: List[int], keep: List[int]) -> np.ndarray:
    """
    Partial trace using einsum.

    Args:
        rho: density matrix on ⊗_i C^{dims[i]}
        dims: subsystem dimensions
        keep: indices to keep (trace out the rest)

    Returns:
        Reduced density matrix on the kept subsystems.
    """
    n = len(dims)
    keep = sorted(keep)
    trace_out = [i for i in range(n) if i not in keep]

    if not trace_out:
        return rho.copy()

    shape = list(dims) + list(dims)
    rho_t = rho.reshape(shape)

    # Build einsum indices
    in_idx = list(range(2 * n))
    for t in trace_out:
        in_idx[n + t] = t

    out_idx = keep + [n + k for k in keep]

    chars = "abcdefghijklmnopqrstuvwxyz"
    if 2 * n > len(chars):
        raise ValueError("Too many subsystems for this simple einsum index scheme.")

    in_str = "".join(chars[i] for i in in_idx)
    out_str = "".join(chars[i] for i in out_idx)

    result = np.einsum(f"{in_str}->{out_str}", rho_t)
    d = int(np.prod([dims[k] for k in keep]))
    return result.reshape(d, d)


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Von Neumann entropy S(ρ) = -Tr ρ log2 ρ."""
    # Ensure hermitian numerically
    rho_h = 0.5 * (rho + rho.conj().T)
    evals = np.linalg.eigvalsh(rho_h)
    evals = np.clip(np.real(evals), 0.0, 1.0)
    evals = evals[evals > eps]
    if evals.size == 0:
        return 0.0
    return float(-np.sum(evals * np.log2(evals)))


def mutual_information(rho: np.ndarray, dims: List[int], A: List[int], B: List[int]) -> float:
    """I(A:B) = S(A) + S(B) - S(AB)."""
    rho_A = partial_trace(rho, dims, A)
    rho_B = partial_trace(rho, dims, B)
    rho_AB = partial_trace(rho, dims, sorted(set(A + B)))
    return von_neumann_entropy(rho_A) + von_neumann_entropy(rho_B) - von_neumann_entropy(rho_AB)


def coherence_offdiag_01(rho_sys_2x2: np.ndarray) -> float:
    """|ρ01| for a 2x2 system density matrix."""
    return float(np.abs(rho_sys_2x2[0, 1]))


def state_overlap(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """|<psi1|psi2>|^2 for state vectors."""
    return float(np.abs(np.vdot(psi1, psi2)) ** 2)


def fidelity_density(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Uhlmann fidelity F(ρ,σ) = (Tr sqrt(sqrt(ρ) σ sqrt(ρ)))^2

    If SciPy sqrtm isn't available, we fall back:
    - If both states are (numerically) pure, use Tr(ρσ).
    - Otherwise return Tr(ρσ) clipped (a lower-fidelity proxy, clearly marked by name).
    """
    # Hermitize numerically
    rho_h = 0.5 * (rho + rho.conj().T)
    sig_h = 0.5 * (sigma + sigma.conj().T)

    if _sqrtm is None:
        # Pure-state detection by purity ~ 1
        if abs(purity(rho_h) - 1.0) < 1e-6 and abs(purity(sig_h) - 1.0) < 1e-6:
            return float(np.real(np.trace(rho_h @ sig_h)))
        # Proxy (NOT full fidelity, but safe and bounded)
        return float(np.clip(np.real(np.trace(rho_h @ sig_h)), 0.0, 1.0))

    sqrt_rho = _sqrtm(rho_h)
    inside = sqrt_rho @ sig_h @ sqrt_rho
    sqrt_inside = _sqrtm(inside)
    tr = np.real(np.trace(sqrt_inside))
    F = float(np.clip(tr * tr, 0.0, 1.0))
    return F


def decoherence_time_first_crossing(times: np.ndarray, coh: np.ndarray, threshold: float = 0.1) -> float:
    """
    Define t_dec as the first time where coherence <= threshold.
    Linear interpolation between neighboring sample points.
    Returns NaN if it never crosses.
    """
    if coh.size == 0:
        return float("nan")
    if coh[0] <= threshold:
        return float(times[0])

    for i in range(1, len(times)):
        if coh[i] <= threshold:
            t0, t1 = float(times[i - 1]), float(times[i])
            c0, c1 = float(coh[i - 1]), float(coh[i])
            if abs(c1 - c0) < 1e-15:
                return t1
            # linear interpolation for crossing
            alpha = (threshold - c0) / (c1 - c0)
            return t0 + alpha * (t1 - t0)
    return float("nan")


# =============================================================================
# Hamiltonian builders
# =============================================================================

def conditional_decoherence_H(n_env: int, J_int: float = 1.0, J_spread: float = 0.5, J_sys: float = 0.05) -> np.ndarray:
    """
    Standard decoherence Hamiltonian: 1 system + n_env environment.
    Conditional X rotations + environment spreading.
    """
    n = 1 + n_env
    H = np.zeros((2**n, 2**n), dtype=complex)

    # Weak system self-Hamiltonian
    ops = [X] + [I] * n_env
    H += J_sys * tensor(*ops)

    # Conditional interaction: system Z couples to each env X
    # H_int = sum_k Z_sys ⊗ X_k
    for k in range(n_env):
        ops = [Z] + [I] * n_env
        ops[1 + k] = X
        H += J_int * tensor(*ops)

    # Environment spreading: nearest-neighbor XX + YY
    for k in range(n_env - 1):
        # XX term
        ops = [I] * (1 + n_env)
        ops[1 + k] = X
        ops[1 + k + 1] = X
        H += J_spread * tensor(*ops)

        # YY term
        ops = [I] * (1 + n_env)
        ops[1 + k] = Y
        ops[1 + k + 1] = Y
        H += J_spread * tensor(*ops)

    return H


def star_topology_H(n_env: int, J_int: float = 1.0, J_spread: float = 0.5, J_sys: float = 0.05) -> np.ndarray:
    """
    Alternative environment topology: system coupled to each env, env also coupled to a central hub qubit (env[0]).
    """
    n = 1 + n_env
    H = np.zeros((2**n, 2**n), dtype=complex)

    # system self term
    ops = [X] + [I] * n_env
    H += J_sys * tensor(*ops)

    # system-env interaction
    for k in range(n_env):
        ops = [Z] + [I] * n_env
        ops[1 + k] = X
        H += J_int * tensor(*ops)

    # env star coupling around hub env[0]
    if n_env >= 2:
        hub = 0
        for k in range(1, n_env):
            ops = [I] * (1 + n_env)
            ops[1 + hub] = X
            ops[1 + k] = X
            H += J_spread * tensor(*ops)

            ops = [I] * (1 + n_env)
            ops[1 + hub] = Y
            ops[1 + k] = Y
            H += J_spread * tensor(*ops)

    return H


# =============================================================================
# Initial states
# =============================================================================

def initial_state(n_env: int) -> np.ndarray:
    """
    System: (|0> + |1>)/sqrt(2)
    Env: |0...0>
    """
    psi_sys = (np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0))
    psi_env = np.zeros((2**n_env,), dtype=complex)
    psi_env[0] = 1.0
    psi = np.kron(psi_sys, psi_env)
    return psi


def initial_state_pointer_basis(n_env: int, basis: str = "Z") -> np.ndarray:
    """
    Different initial system basis states for pointer studies.
    basis:
      - "Z+": |0>
      - "Z-": |1>
      - "X+": (|0>+|1>)/sqrt2
      - "X-": (|0>-|1>)/sqrt2
      - "Y+": (|0>+i|1>)/sqrt2
      - "Y-": (|0>-i|1>)/sqrt2
    """
    if basis == "Z+":
        psi_sys = np.array([1.0, 0.0], dtype=complex)
    elif basis == "Z-":
        psi_sys = np.array([0.0, 1.0], dtype=complex)
    elif basis == "X+":
        psi_sys = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    elif basis == "X-":
        psi_sys = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0)
    elif basis == "Y+":
        psi_sys = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
    elif basis == "Y-":
        psi_sys = np.array([1.0, -1.0j], dtype=complex) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unknown basis '{basis}'")

    psi_env = np.zeros((2**n_env,), dtype=complex)
    psi_env[0] = 1.0
    return np.kron(psi_sys, psi_env)


# =============================================================================
# Evolution helpers
# =============================================================================

def evolve_collect_system_coherence(
    H: np.ndarray,
    psi0: np.ndarray,
    n_env: int,
    times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve under H from psi0 and collect |rho01| at each time sample.
    Uses a fixed dt derived from times spacing.
    """
    n = 1 + n_env
    dims = [2] * n

    dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
    U = expm(-1j * H * dt)

    psi = psi0.copy()
    coh = np.zeros((len(times),), dtype=float)

    for i in range(len(times)):
        rho = np.outer(psi, psi.conj())
        rho_sys = partial_trace(rho, dims, [0])
        coh[i] = coherence_offdiag_01(rho_sys)
        if i < len(times) - 1:
            psi = U @ psi

    return times, coh


# =============================================================================
# Study 1: Environment scaling
# =============================================================================

def study_environment_scaling(
    env_sizes: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
    t_max: float = 3.0,
    n_steps: int = 60,
    threshold: float = 0.1,
) -> Dict[str, object]:
    """How does decoherence depend on environment size?"""
    print("\n" + "=" * 60)
    print("STUDY 1: ENVIRONMENT SIZE SCALING")
    print("=" * 60)

    times = np.linspace(0.0, t_max, n_steps)

    decoh_times: List[float] = []
    final_coh: List[float] = []

    plt.figure(figsize=(10, 6))
    for n_env in env_sizes:
        H = conditional_decoherence_H(n_env, J_spread=0.5)
        psi0 = initial_state(n_env)
        t, coh = evolve_collect_system_coherence(H, psi0, n_env, times)
        decoh_times.append(decoherence_time_first_crossing(t, coh, threshold=threshold))
        final_coh.append(float(coh[-1]))
        plt.plot(t, coh, label=f"n_env={n_env}")

    plt.xlabel("Time")
    plt.ylabel(r"System coherence $|\rho_{01}|$")
    plt.title("Decoherence vs Environment Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("study1_env_scaling.png", dpi=150)
    print("\nSaved: study1_env_scaling.png")

    return {
        "env_sizes": env_sizes,
        "times": times.tolist(),
        "threshold": threshold,
        "decoherence_times": decoh_times,
        "final_coherence": final_coh,
    }


# =============================================================================
# Study 2: Coupling strength
# =============================================================================

def study_coupling_strength(
    n_env: int = 6,
    couplings: List[float] = [0.2, 0.5, 1.0, 1.5, 2.0],
    t_max: float = 3.0,
    n_steps: int = 60,
    threshold: float = 0.1,
) -> Dict[str, object]:
    """How does coupling strength affect decoherence timescale?"""
    print("\n" + "=" * 60)
    print("STUDY 2: COUPLING STRENGTH EFFECT")
    print("=" * 60)

    times = np.linspace(0.0, t_max, n_steps)

    decoh_times: List[float] = []
    final_coh: List[float] = []

    plt.figure(figsize=(10, 6))
    for J_int in couplings:
        H = conditional_decoherence_H(n_env, J_int=J_int, J_spread=0.5)
        psi0 = initial_state(n_env)
        t, coh = evolve_collect_system_coherence(H, psi0, n_env, times)
        decoh_times.append(decoherence_time_first_crossing(t, coh, threshold=threshold))
        final_coh.append(float(coh[-1]))
        plt.plot(t, coh, label=f"J_int={J_int:.2f}")

    plt.xlabel("Time")
    plt.ylabel(r"System coherence $|\rho_{01}|$")
    plt.title("Decoherence vs Coupling Strength")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("study2_coupling.png", dpi=150)
    print("\nSaved: study2_coupling.png")

    return {
        "n_env": n_env,
        "couplings": couplings,
        "times": times.tolist(),
        "threshold": threshold,
        "decoherence_times": decoh_times,
        "final_coherence": final_coh,
    }


# =============================================================================
# Study 3: Information spreading
# =============================================================================

def study_information_spreading(
    n_env: int = 8,
    t_max: float = 4.0,
    n_steps: int = 80,
) -> Dict[str, object]:
    """How does information propagate through the environment?"""
    print("\n" + "=" * 60)
    print("STUDY 3: INFORMATION SPREADING")
    print("=" * 60)

    H = conditional_decoherence_H(n_env, J_int=1.0, J_spread=0.5)
    psi = initial_state(n_env)
    n = 1 + n_env
    dims = [2] * n

    times = np.linspace(0.0, t_max, n_steps)
    dt = float(times[1] - times[0])
    U = expm(-1j * H * dt)

    # Track MI between system and each environment qubit
    sys_env_mi = np.zeros((len(times), n_env), dtype=float)

    for i, _t in enumerate(times):
        rho = np.outer(psi, psi.conj())
        for k in range(n_env):
            sys_env_mi[i, k] = mutual_information(rho, dims, [0], [1 + k])
        if i < len(times) - 1:
            psi = U @ psi

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for k in range(n_env):
        ax.plot(times, sys_env_mi[:, k], label=f"env[{k}]")

    ax.set_xlabel("Time")
    ax.set_ylabel("I(System : env[k])")
    ax.set_title("Mutual Information: System with Each Environment Qubit")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("study3_spreading.png", dpi=150)
    print("\nSaved: study3_spreading.png")

    return {"times": times.tolist(), "sys_env_mi": sys_env_mi.tolist()}


# =============================================================================
# Study 4: Reversibility (Loschmidt echo)
# =============================================================================

def study_reversibility(
    n_env: int = 5,
    t_forward: float = 3.0,
    n_steps: int = 60,
) -> Dict[str, object]:
    """Can we reverse decoherence? Loschmidt echo."""
    print("\n" + "=" * 60)
    print("STUDY 4: REVERSIBILITY (LOSCHMIDT ECHO)")
    print("=" * 60)

    H = conditional_decoherence_H(n_env)
    psi0 = initial_state(n_env)
    n = 1 + n_env
    dims = [2] * n

    times_forward = np.linspace(0.0, t_forward, n_steps)
    dt = float(times_forward[1] - times_forward[0])
    U_fwd = expm(-1j * H * dt)
    U_bwd = expm(+1j * H * dt)  # exact time reverse for the same H

    # Choose reversal times (fractions of t_forward)
    t_reversals = [t_forward / 4.0, t_forward / 2.0, 3.0 * t_forward / 4.0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    results: Dict[str, object] = {"t_reversals": t_reversals, "final_overlaps": []}

    for ax, t_rev in zip(axes, t_reversals):
        # Forward evolve until closest index to t_rev
        rev_idx = int(np.argmin(np.abs(times_forward - t_rev)))

        psi = psi0.copy()
        forward_coh: List[float] = []

        for i in range(rev_idx + 1):
            rho = np.outer(psi, psi.conj())
            rho_sys = partial_trace(rho, dims, [0])
            forward_coh.append(coherence_offdiag_01(rho_sys))
            if i < rev_idx:
                psi = U_fwd @ psi

        # Reverse evolution for EXACTLY rev_idx steps (fixes off-by-one)
        backward_coh: List[float] = []
        for j in range(rev_idx):
            rho = np.outer(psi, psi.conj())
            rho_sys = partial_trace(rho, dims, [0])
            backward_coh.append(coherence_offdiag_01(rho_sys))
            psi = U_bwd @ psi

        # Record final overlap after reversal steps
        final_overlap = state_overlap(psi, psi0)
        results["final_overlaps"].append(final_overlap)

        # Plot: forward then backward time axis
        t_fwd = times_forward[: rev_idx + 1]
        ax.plot(t_fwd, forward_coh, linewidth=2, label="Forward")
        if len(backward_coh) > 0:
            t_bwd = t_rev - times_forward[: rev_idx]  # reversed axis to show "coming back"
            ax.plot(t_bwd, backward_coh, linewidth=2, label="Backward")

        ax.axvline(x=float(times_forward[rev_idx]), color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_title(f"Reverse at t≈{times_forward[rev_idx]:.2f}\nFinal overlap={final_overlap:.4f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel(r"System coherence $|\rho_{01}|$")
    plt.suptitle("Loschmidt Echo: Reversing Decoherence", fontsize=14)
    plt.tight_layout()
    plt.savefig("study4_reversibility.png", dpi=150)
    print("\nSaved: study4_reversibility.png")

    return results


# =============================================================================
# Study 5: Pointer basis
# =============================================================================

def pointer_H(n_env: int, pointer_op: str = "Z", J_int: float = 1.0, J_spread: float = 0.5, J_sys: float = 0.05) -> np.ndarray:
    """
    Let the interaction define the pointer basis by choosing the system operator
    that conditions the environment coupling.

    pointer_op:
      - "Z": couples via Z_sys ⊗ X_env
      - "X": couples via X_sys ⊗ X_env
      - "Y": couples via Y_sys ⊗ X_env
    """
    if pointer_op == "Z":
        S = Z
    elif pointer_op == "X":
        S = X
    elif pointer_op == "Y":
        S = Y
    else:
        raise ValueError(f"Unknown pointer_op '{pointer_op}'")

    n = 1 + n_env
    H = np.zeros((2**n, 2**n), dtype=complex)

    # Weak system self-H
    H += J_sys * tensor(*([X] + [I] * n_env))

    # Conditional interaction
    for k in range(n_env):
        ops = [S] + [I] * n_env
        ops[1 + k] = X
        H += J_int * tensor(*ops)

    # Environment spreading (chain)
    for k in range(n_env - 1):
        ops = [I] * (1 + n_env)
        ops[1 + k] = X
        ops[1 + k + 1] = X
        H += J_spread * tensor(*ops)

        ops = [I] * (1 + n_env)
        ops[1 + k] = Y
        ops[1 + k + 1] = Y
        H += J_spread * tensor(*ops)

    return H


def study_pointer_basis(
    n_env: int = 6,
    t_max: float = 3.0,
    n_steps: int = 60,
) -> Dict[str, object]:
    """What determines which states survive? Compare different interaction-defined pointer bases."""
    print("\n" + "=" * 60)
    print("STUDY 5: POINTER BASIS")
    print("=" * 60)

    times = np.linspace(0.0, t_max, n_steps)

    pointer_ops = ["Z", "X", "Y"]
    init_states = ["Z+", "Z-", "X+", "X-", "Y+", "Y-"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    results: Dict[str, object] = {"pointer_ops": pointer_ops, "init_states": init_states, "traces": {}}

    for ax, p_op in zip(axes, pointer_ops):
        H = pointer_H(n_env, pointer_op=p_op)
        traces_for_op: Dict[str, List[float]] = {}

        for st in init_states:
            psi0 = initial_state_pointer_basis(n_env, basis=st)
            _t, coh = evolve_collect_system_coherence(H, psi0, n_env, times)
            traces_for_op[st] = coh.tolist()
            ax.plot(times, coh, label=st)

        results["traces"][p_op] = traces_for_op

        ax.set_title(f"Interaction Pointer: {p_op}")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)

    axes[0].set_ylabel(r"System coherence $|\rho_{01}|$")
    plt.suptitle("Pointer Basis Selection by Interaction", fontsize=14)
    plt.tight_layout()
    plt.savefig("study5_pointer_basis.png", dpi=150)
    print("\nSaved: study5_pointer_basis.png")

    return results


# =============================================================================
# Study 6: Quantum Darwinism (redundancy)
# =============================================================================

def _choose_sampling_indices(times: np.ndarray, sample_times: List[float]) -> List[int]:
    """Map desired sample_times to nearest indices (unique, sorted)."""
    idxs = sorted(set(int(np.argmin(np.abs(times - st))) for st in sample_times))
    return idxs


def study_quantum_darwinism(
    n_env: int = 8,
    t_max: float = 5.0,
    n_steps: int = 50,
    seed: int = 123,
) -> Dict[str, object]:
    """Is information about the system redundantly encoded in environment fragments?"""
    print("\n" + "=" * 60)
    print("STUDY 6: QUANTUM DARWINISM (REDUNDANCY)")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    H = conditional_decoherence_H(n_env, J_spread=0.5)
    psi = initial_state(n_env)
    n = 1 + n_env
    dims = [2] * n

    times = np.linspace(0.0, t_max, n_steps)
    dt = float(times[1] - times[0])
    U = expm(-1j * H * dt)

    fragment_sizes = list(range(1, n_env + 1))

    # Sample at fixed fractions of t_max (robust to float equality)
    desired_times = [0.0, t_max / 4.0, t_max / 2.0, 3.0 * t_max / 4.0, t_max]
    sample_idxs = _choose_sampling_indices(times, desired_times)

    mi_vs_frag: Dict[float, List[float]] = {}
    info_profile_samples: Dict[float, List[float]] = {}

    for i, t in enumerate(times):
        if i in sample_idxs:
            rho = np.outer(psi, psi.conj())
            S_sys = von_neumann_entropy(partial_trace(rho, dims, [0]))

            mi_for_size: List[float] = []
            for frag_size in fragment_sizes:
                # sample random fragments of this size
                # cap samples for tractability
                n_samples = min(12, int(_n_choose_k(n_env, frag_size)))
                mis: List[float] = []
                for _ in range(n_samples):
                    env_qubits = np.arange(1, n)
                    frag_env = np.sort(rng.choice(env_qubits, size=frag_size, replace=False))
                    I_SF = mutual_information(rho, dims, [0], frag_env.tolist())
                    mis.append(I_SF)
                mi_for_size.append(float(np.mean(mis)) if mis else 0.0)

            # Profile: I(S:env[k]) per qubit at this time
            profile = [mutual_information(rho, dims, [0], [1 + k]) for k in range(n_env)]

            mi_vs_frag[float(times[i])] = mi_for_size
            info_profile_samples[float(times[i])] = profile

        if i < len(times) - 1:
            psi = U @ psi

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for t_key in sorted(mi_vs_frag.keys()):
        ax.plot(fragment_sizes, mi_vs_frag[t_key], "o-", label=f"t={t_key:.2f}")
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="1 bit")
    ax.set_xlabel("Fragment Size (# of env qubits)")
    ax.set_ylabel("I(System : Fragment)")
    ax.set_title("Mutual Information vs Fragment Size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for t_key in sorted(info_profile_samples.keys()):
        ax.plot(range(n_env), info_profile_samples[t_key], "o-", label=f"t={t_key:.2f}")
    ax.set_xlabel("Environment Qubit Index")
    ax.set_ylabel("I(System : env[k])")
    ax.set_title("Information About System Across Environment")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("study6_darwinism.png", dpi=150)
    print("\nSaved: study6_darwinism.png")

    return {
        "times": times.tolist(),
        "fragment_sizes": fragment_sizes,
        "mi_vs_frag": {f"{k:.6f}": v for k, v in mi_vs_frag.items()},
        "info_profile": {f"{k:.6f}": v for k, v in info_profile_samples.items()},
        "seed": seed,
    }


def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    c = 1
    for i in range(1, k + 1):
        c = c * (n - k + i) // i
    return c


# =============================================================================
# Study 7: Topology
# =============================================================================

def study_topology(
    n_env: int = 8,
    t_max: float = 3.0,
    n_steps: int = 60,
) -> Dict[str, object]:
    """Compare chain vs star environment topology."""
    print("\n" + "=" * 60)
    print("STUDY 7: ENVIRONMENT TOPOLOGY")
    print("=" * 60)

    times = np.linspace(0.0, t_max, n_steps)

    H_chain = conditional_decoherence_H(n_env, J_spread=0.5)
    H_star = star_topology_H(n_env, J_spread=0.5)

    psi0 = initial_state(n_env)

    _, coh_chain = evolve_collect_system_coherence(H_chain, psi0, n_env, times)
    _, coh_star = evolve_collect_system_coherence(H_star, psi0, n_env, times)

    plt.figure(figsize=(10, 6))
    plt.plot(times, coh_chain, label="Chain topology")
    plt.plot(times, coh_star, label="Star topology")
    plt.xlabel("Time")
    plt.ylabel(r"System coherence $|\rho_{01}|$")
    plt.title("Decoherence vs Environment Topology")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("study7_topology.png", dpi=150)
    print("\nSaved: study7_topology.png")

    return {"times": times.tolist(), "coh_chain": coh_chain.tolist(), "coh_star": coh_star.tolist()}


# =============================================================================
# Study 8: Timescales
# =============================================================================

def study_timescales(
    n_env: int = 6,
    couplings: List[float] = [0.3, 0.5, 0.8, 1.0, 1.3, 1.6],
    t_max: float = 4.0,
    n_steps: int = 80,
    threshold: float = 0.1,
) -> Dict[str, object]:
    """Estimate decoherence timescale vs coupling strength and compare simple scalings."""
    print("\n" + "=" * 60)
    print("STUDY 8: DECOHERENCE TIMESCALES")
    print("=" * 60)

    times = np.linspace(0.0, t_max, n_steps)

    tdec: List[float] = []
    for J_int in couplings:
        H = conditional_decoherence_H(n_env, J_int=J_int, J_spread=0.5)
        psi0 = initial_state(n_env)
        _t, coh = evolve_collect_system_coherence(H, psi0, n_env, times)
        tdec.append(decoherence_time_first_crossing(times, coh, threshold=threshold))

    tdec = np.array(tdec, dtype=float)
    couplings_arr = np.array(couplings, dtype=float)

    # Simple reference scalings (avoid divide-by-zero)
    invJ = 1.0 / np.clip(couplings_arr, 1e-12, None)
    invJ2 = 1.0 / np.clip(couplings_arr**2, 1e-12, None)

    plt.figure(figsize=(10, 6))
    plt.plot(couplings_arr, tdec, "o-", label=r"measured $t_{\rm dec}$ (first |ρ01|<thr)")
    plt.plot(couplings_arr, invJ * (tdec[0] / invJ[0]), "--", label=r"$\propto 1/J$")
    plt.plot(couplings_arr, invJ2 * (tdec[0] / invJ2[0]), "--", label=r"$\propto 1/J^2$")
    plt.xlabel("Coupling strength J_int")
    plt.ylabel("Decoherence time t_dec")
    plt.title(f"Decoherence timescale vs coupling (threshold={threshold})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("study8_timescales.png", dpi=150)
    print("\nSaved: study8_timescales.png")

    return {
        "n_env": n_env,
        "couplings": couplings,
        "threshold": threshold,
        "t_dec": tdec.tolist(),
    }


# =============================================================================
# RUN ALL STUDIES + SUMMARY
# =============================================================================

def _write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    import csv

    if not rows:
        return

    # stable column order: union of keys
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def run_all_studies() -> None:
    """Run all decoherence studies."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DECOHERENCE STUDY (CLEAN)")
    print("=" * 60)

    os.makedirs("decoherence_studies", exist_ok=True)
    os.chdir("decoherence_studies")

    # Collect compact metrics for a run summary
    summary_rows: List[Dict[str, object]] = []
    summary_json: Dict[str, object] = {}

    r1 = study_environment_scaling()
    summary_json["study1_environment_scaling"] = r1
    for n_env, tdec, fcoh in zip(r1["env_sizes"], r1["decoherence_times"], r1["final_coherence"]):
        summary_rows.append(
            {
                "study": "study1_env_scaling",
                "n_env": int(n_env),
                "threshold": r1["threshold"],
                "t_dec": tdec,
                "final_coherence": fcoh,
            }
        )

    r2 = study_coupling_strength()
    summary_json["study2_coupling_strength"] = r2
    for J, tdec, fcoh in zip(r2["couplings"], r2["decoherence_times"], r2["final_coherence"]):
        summary_rows.append(
            {
                "study": "study2_coupling_strength",
                "n_env": int(r2["n_env"]),
                "J_int": float(J),
                "threshold": r2["threshold"],
                "t_dec": tdec,
                "final_coherence": fcoh,
            }
        )

    r3 = study_information_spreading()
    summary_json["study3_information_spreading"] = r3

    r4 = study_reversibility()
    summary_json["study4_reversibility"] = r4
    for t_rev, ov in zip(r4["t_reversals"], r4["final_overlaps"]):
        summary_rows.append(
            {
                "study": "study4_reversibility",
                "n_env": 5,
                "t_reverse": float(t_rev),
                "final_overlap": float(ov),
            }
        )

    r5 = study_pointer_basis()
    summary_json["study5_pointer_basis"] = r5

    r6 = study_quantum_darwinism()
    summary_json["study6_quantum_darwinism"] = r6

    r7 = study_topology()
    summary_json["study7_topology"] = r7

    r8 = study_timescales()
    summary_json["study8_timescales"] = r8
    for J, tdec in zip(r8["couplings"], r8["t_dec"]):
        summary_rows.append(
            {
                "study": "study8_timescales",
                "n_env": int(r8["n_env"]),
                "J_int": float(J),
                "threshold": r8["threshold"],
                "t_dec": float(tdec),
            }
        )

    _write_summary_csv("decoherence_summary.csv", summary_rows)
    with open("decoherence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\nSaved: decoherence_summary.csv")
    print("Saved: decoherence_summary.json")

    print("\n" + "=" * 60)
    print("ALL STUDIES COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  study1_env_scaling.png")
    print("  study2_coupling.png")
    print("  study3_spreading.png")
    print("  study4_reversibility.png")
    print("  study5_pointer_basis.png")
    print("  study6_darwinism.png")
    print("  study7_topology.png")
    print("  study8_timescales.png")
    print("  decoherence_summary.csv")
    print("  decoherence_summary.json")


if __name__ == "__main__":
    run_all_studies()
