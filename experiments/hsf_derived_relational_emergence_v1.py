# hsf_derived_relational_emergence_v1.py

r"""
HSF DERIVED RELATIONAL EMERGENCE PROBE (v1, patched witness)

Purpose
-------
Probe whether interaction-generated relational/interface structure can be
identified as an effective subsystem-like factor from the dynamics, without
manually spawning new nodes.

System
------
A small closed A-L-B model:
  - A: endpoint subsystem
  - L: candidate interface / relational register
  - B: endpoint subsystem

What this script tests
----------------------
1. Committed minimal interface:
   dL = 2 active link space.

2. Oversized interface with weak slack:
   dL = 6, but only a 2D active sector is strongly used.

3. Oversized interface with random weak slack mixing:
   dL = 6, again testing whether extra dimensions matter.

Diagnostics
-----------
At each step:
  - I(A:B), I(A:L), I(B:L)
  - I3(A:B:L)
  - W_relational = I(A:L) + I(B:L) - I(A:B)
  - A -> L response matrix singular spectrum on committed active readout
  - rank-1 / rank-2 response compactness errors
  - slack occupancy

Patched no-refolding / refolding-like null
------------------------------------------
The physical evolution is kept FIXED.
We do NOT co-rotate the Hamiltonian with the state.

We use two privileged, incomplete readout dictionaries on L:
  1) active readout basis on the committed 2D active sector
  2) slack readout basis on the remaining dimensions

Then for the null, we scramble the physical link basis relative to those
committed dictionaries.

Headline witnesses
------------------
Instead of relying on the old compactness-damage metric, this version adds:

  - active_power:
      Frobenius norm squared of the active readout response matrix

  - slack_power:
      Frobenius norm squared of the slack readout response matrix

  - damage_active_power:
      active_power(committed) - active_power(scrambled)

  - leakage_ratio:
      slack_power / active_power

  - leakage_gain:
      leakage_ratio(scrambled) - leakage_ratio(committed)

Interpretation
--------------
Support for a committed interface sector is stronger when:
  - W_relational becomes nontrivial
  - active response is concentrated in a few modes
  - oversizing L does not help much
  - scrambled readout reduces active_power
  - scrambled readout increases leakage_ratio into slack observables

Outputs
-------
summary.json
summary.csv
w_relational.png
sv1.png
damage_active_power.png
leakage_gain.png
slack_occupancy.png

Example
-------
python hsf_derived_relational_emergence_v1.py --outdir hsf_derived_relational_out --steps 120 --dt 0.08 --seed 0
"""

import argparse
import csv
import json
import math
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Basic linear algebra
# ============================================================

def dagger(x: np.ndarray) -> np.ndarray:
    return x.conj().T


def kron_list(ops: List[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def matrix_exp_hermitian(H: np.ndarray, dt: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    return evecs @ np.diag(np.exp(-1j * dt * evals)) @ dagger(evecs)


def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    phases = np.ones(n, dtype=complex)
    mask = np.abs(d) > 1e-15
    phases[mask] = d[mask] / np.abs(d[mask])
    return q * phases[np.newaxis, :]


def normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(psi)
    if nrm < 1e-15:
        raise ValueError("State norm is ~0.")
    return psi / nrm


def dims_product(dims: List[int]) -> int:
    out = 1
    for d in dims:
        out *= d
    return out


# ============================================================
# Subsystem bookkeeping
# ============================================================

def reshape_state_for_partition(psi: np.ndarray, dims: List[int], keep: List[int]) -> np.ndarray:
    n = len(dims)
    rest = [i for i in range(n) if i not in keep]
    perm = keep + rest
    arr = psi.reshape(dims)
    arr = np.transpose(arr, axes=perm)
    d_keep = dims_product([dims[i] for i in keep])
    d_rest = dims_product([dims[i] for i in rest])
    return arr.reshape(d_keep, d_rest)


def reduced_density_matrix_pure(psi: np.ndarray, dims: List[int], keep: List[int]) -> np.ndarray:
    mat = reshape_state_for_partition(psi, dims, keep)
    rho = mat @ dagger(mat)
    rho = 0.5 * (rho + dagger(rho))
    return rho


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    evals = np.linalg.eigvalsh(0.5 * (rho + dagger(rho)))
    evals = np.clip(np.real(evals), 0.0, 1.0)
    nz = evals[evals > eps]
    if len(nz) == 0:
        return 0.0
    return float(-np.sum(nz * np.log2(nz)))


def mutual_information_pure(psi: np.ndarray, dims: List[int], A: List[int], B: List[int]) -> float:
    A = sorted(A)
    B = sorted(B)
    AB = sorted(A + B)
    rhoA = reduced_density_matrix_pure(psi, dims, A)
    rhoB = reduced_density_matrix_pure(psi, dims, B)
    rhoAB = reduced_density_matrix_pure(psi, dims, AB)
    return von_neumann_entropy(rhoA) + von_neumann_entropy(rhoB) - von_neumann_entropy(rhoAB)


def interaction_information_pure(psi: np.ndarray, dims: List[int], A: List[int], B: List[int], C: List[int]) -> float:
    A = sorted(A)
    B = sorted(B)
    C = sorted(C)
    AB = sorted(A + B)
    AC = sorted(A + C)
    BC = sorted(B + C)
    ABC = sorted(A + B + C)

    SA = von_neumann_entropy(reduced_density_matrix_pure(psi, dims, A))
    SB = von_neumann_entropy(reduced_density_matrix_pure(psi, dims, B))
    SC = von_neumann_entropy(reduced_density_matrix_pure(psi, dims, C))
    SAB = von_neumann_entropy(reduced_density_matrix_pure(psi, dims, AB))
    SAC = von_neumann_entropy(reduced_density_matrix_pure(psi, dims, AC))
    SBC = von_neumann_entropy(reduced_density_matrix_pure(psi, dims, BC))
    SABC = von_neumann_entropy(reduced_density_matrix_pure(psi, dims, ABC))
    return float(SA + SB + SC - SAB - SAC - SBC + SABC)


# ============================================================
# Operators / bases
# ============================================================

def pauli_basis() -> Dict[str, np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def su2_generators() -> List[np.ndarray]:
    pb = pauli_basis()
    return [0.5 * pb["X"], 0.5 * pb["Y"], 0.5 * pb["Z"]]


def local_operator_on_subsystem(op: np.ndarray, dims: List[int], target: int) -> np.ndarray:
    ops = []
    for i, d in enumerate(dims):
        if i == target:
            if op.shape != (d, d):
                raise ValueError(f"Operator shape {op.shape} incompatible with dim {d} at site {i}.")
            ops.append(op)
        else:
            ops.append(np.eye(d, dtype=complex))
    return kron_list(ops)


def committed_active_link_basis(dL: int) -> List[np.ndarray]:
    """
    Privileged, incomplete readout basis tied to the active 2D sector.
    Intentionally not complete.
    """
    if dL < 2:
        raise ValueError("Need dL >= 2.")

    basis: List[np.ndarray] = []

    P0 = np.zeros((dL, dL), dtype=complex)
    P1 = np.zeros((dL, dL), dtype=complex)
    P0[0, 0] = 1.0
    P1[1, 1] = 1.0

    X = np.zeros((dL, dL), dtype=complex)
    Y = np.zeros((dL, dL), dtype=complex)
    Z = np.zeros((dL, dL), dtype=complex)

    X[:2, :2] = np.array([[0, 1], [1, 0]], dtype=complex)
    Y[:2, :2] = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z[:2, :2] = np.array([[1, 0], [0, -1]], dtype=complex)

    basis.extend([P0, P1, X, Y, Z])
    return basis


def committed_slack_link_basis(dL: int) -> List[np.ndarray]:
    """
    Privileged, incomplete readout basis on the slack subspace.
    For dL <= 2, this is empty.
    """
    if dL <= 2:
        return []

    basis: List[np.ndarray] = []

    for i in range(2, dL):
        P = np.zeros((dL, dL), dtype=complex)
        P[i, i] = 1.0
        basis.append(P)

    for i in range(2, dL):
        for j in range(i + 1, dL):
            X = np.zeros((dL, dL), dtype=complex)
            Y = np.zeros((dL, dL), dtype=complex)
            X[i, j] = X[j, i] = 1.0
            Y[i, j] = -1j
            Y[j, i] = 1j
            basis.append(X / math.sqrt(2.0))
            basis.append(Y / math.sqrt(2.0))

    return basis


# ============================================================
# Model construction
# ============================================================

def build_model_hamiltonian(
    dA: int,
    dL: int,
    dB: int,
    eta_AL: float,
    eta_LB: float,
    omega_site: float,
    rng: np.random.Generator,
    slack_mode: str = "none",
) -> Dict[str, np.ndarray]:
    dims = [dA, dL, dB]

    gens2 = su2_generators()
    XA, YA, ZA = [local_operator_on_subsystem(op, dims, 0) for op in gens2]
    XB, YB, ZB = [local_operator_on_subsystem(op, dims, 2) for op in gens2]

    if dL < 2:
        raise ValueError("Need dL >= 2.")

    X2 = np.array([[0, 1], [1, 0]], dtype=complex)
    Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

    XL = np.zeros((dL, dL), dtype=complex)
    YL = np.zeros((dL, dL), dtype=complex)
    ZL = np.zeros((dL, dL), dtype=complex)
    XL[:2, :2] = X2
    YL[:2, :2] = Y2
    ZL[:2, :2] = Z2

    XL_big = local_operator_on_subsystem(0.5 * XL, dims, 1)
    YL_big = local_operator_on_subsystem(0.5 * YL, dims, 1)
    ZL_big = local_operator_on_subsystem(0.5 * ZL, dims, 1)

    H_local = omega_site * (ZA + ZB)
    H_local += 0.35 * omega_site * ZL_big

    H_couple = eta_AL * (XA @ XL_big + YA @ YL_big + ZA @ ZL_big)
    H_couple += eta_LB * (XL_big @ XB + YL_big @ YB + ZL_big @ ZB)

    H_slack = np.zeros((dims_product(dims), dims_product(dims)), dtype=complex)
    if dL > 2:
        if slack_mode == "none":
            pass
        elif slack_mode == "weak":
            for s in range(2, dL):
                mix = np.zeros((dL, dL), dtype=complex)
                mix[0, s] = mix[s, 0] = 1.0
                H_slack += 0.05 * omega_site * local_operator_on_subsystem(mix, dims, 1)
        elif slack_mode == "random_weak":
            M = rng.normal(size=(dL, dL)) + 1j * rng.normal(size=(dL, dL))
            M = 0.5 * (M + dagger(M))
            H_slack += 0.03 * omega_site * local_operator_on_subsystem(M, dims, 1)
        else:
            raise ValueError(f"Unknown slack_mode: {slack_mode}")

    H = H_local + H_couple + H_slack
    return {
        "dims": dims,
        "H": H,
        "H_local": H_local,
        "H_couple": H_couple,
        "H_slack": H_slack,
    }


def initial_product_state(dA: int, dL: int, dB: int) -> np.ndarray:
    psiA = np.zeros(dA, dtype=complex)
    psiA[0] = 1.0

    psiL = np.zeros(dL, dtype=complex)
    psiL[0] = 1.0

    psiB = np.zeros(dB, dtype=complex)
    psiB[-1] = 1.0

    psi = kron_list([psiA[:, None], psiL[:, None], psiB[:, None]]).reshape(-1)
    return normalize_state(psi)


# ============================================================
# Diagnostics
# ============================================================

def link_occupancy_distribution(psi: np.ndarray, dims: List[int]) -> np.ndarray:
    rhoL = reduced_density_matrix_pure(psi, dims, [1])
    pops = np.real(np.diag(rhoL))
    pops = np.clip(pops, 0.0, 1.0)
    s = pops.sum()
    if s > 1e-15:
        pops /= s
    return pops


def slack_occupancy(psi: np.ndarray, dims: List[int], active_dim: int = 2) -> float:
    pops = link_occupancy_distribution(psi, dims)
    if len(pops) <= active_dim:
        return 0.0
    return float(np.sum(pops[active_dim:]))


def response_matrix_A_to_L_with_basis(
    H: np.ndarray,
    psi: np.ndarray,
    dims: List[int],
    dt: float,
    link_basis_ops_small: List[np.ndarray],
    link_readout_unitary: np.ndarray = None,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Poke A with small local unitaries, evolve one step, measure response on L.

    The readout uses a privileged, incomplete operator dictionary.
    For the scrambled null, we keep the physical dynamics unchanged but rotate
    the physical link basis relative to the committed readout operators.
    """
    U_step = matrix_exp_hermitian(H, dt)
    gensA = su2_generators()
    poke_ops = [local_operator_on_subsystem(G, dims, 0) for G in gensA]

    if link_readout_unitary is not None:
        U = link_readout_unitary
        link_basis_ops_small = [dagger(U) @ B @ U for B in link_basis_ops_small]

    link_ops = [local_operator_on_subsystem(Bi, dims, 1) for Bi in link_basis_ops_small]

    psi0 = U_step @ psi
    base_vals = np.array([np.real(np.vdot(psi0, O @ psi0)) for O in link_ops], dtype=float)

    rows = []
    for P in poke_ops:
        U_poke = matrix_exp_hermitian(P, eps)
        psi_p = U_step @ (U_poke @ psi)
        vals = np.array([np.real(np.vdot(psi_p, O @ psi_p)) for O in link_ops], dtype=float)
        rows.append((vals - base_vals) / eps)

    M = np.array(rows, dtype=float)
    svals = np.linalg.svd(M, compute_uv=False) if M.size > 0 else np.array([], dtype=float)
    return M, svals


def best_rank_k_error(M: np.ndarray, k: int) -> float:
    if M.size == 0:
        return 0.0
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    s2 = s.copy()
    if k < len(s2):
        s2[k:] = 0.0
    Mk = (U * s2[np.newaxis, :]) @ Vh
    denom = np.linalg.norm(M)
    if denom < 1e-15:
        return 0.0
    return float(np.linalg.norm(M - Mk) / denom)


def response_power(M: np.ndarray) -> float:
    if M.size == 0:
        return 0.0
    return float(np.sum(M * M))


def diagnostics_for_state(
    psi: np.ndarray,
    H: np.ndarray,
    dims: List[int],
    dt_resp: float,
    link_readout_unitary: np.ndarray = None,
) -> Dict[str, float]:
    dL = dims[1]

    I_AB = mutual_information_pure(psi, dims, [0], [2])
    I_AL = mutual_information_pure(psi, dims, [0], [1])
    I_BL = mutual_information_pure(psi, dims, [2], [1])
    I3 = interaction_information_pure(psi, dims, [0], [1], [2])
    W = I_AL + I_BL - I_AB

    active_basis = committed_active_link_basis(dL)
    slack_basis = committed_slack_link_basis(dL)

    M_active, svals_active = response_matrix_A_to_L_with_basis(
        H=H,
        psi=psi,
        dims=dims,
        dt=dt_resp,
        link_basis_ops_small=active_basis,
        link_readout_unitary=link_readout_unitary,
    )

    M_slack, _ = response_matrix_A_to_L_with_basis(
        H=H,
        psi=psi,
        dims=dims,
        dt=dt_resp,
        link_basis_ops_small=slack_basis,
        link_readout_unitary=link_readout_unitary,
    ) if len(slack_basis) > 0 else (np.zeros((3, 0), dtype=float), np.array([], dtype=float))

    sv1 = float(svals_active[0]) if len(svals_active) >= 1 else 0.0
    sv2 = float(svals_active[1]) if len(svals_active) >= 2 else 0.0
    sv3 = float(svals_active[2]) if len(svals_active) >= 3 else 0.0

    active_power = response_power(M_active)
    slack_power = response_power(M_slack)
    leakage_ratio = float(slack_power / active_power) if active_power > 1e-15 else 0.0

    return {
        "I_AB": float(I_AB),
        "I_AL": float(I_AL),
        "I_BL": float(I_BL),
        "I3": float(I3),
        "W_relational": float(W),
        "sv1": sv1,
        "sv2": sv2,
        "sv3": sv3,
        "bandwidth_ratio_21": float(sv2 / sv1) if sv1 > 1e-15 else 0.0,
        "rank1_err": best_rank_k_error(M_active, 1),
        "rank2_err": best_rank_k_error(M_active, 2),
        "active_power": active_power,
        "slack_power": slack_power,
        "leakage_ratio": leakage_ratio,
        "slack_occupancy": slack_occupancy(psi, dims, active_dim=2),
    }


# ============================================================
# Experiment runner
# ============================================================

def run_single_case(
    label: str,
    dL: int,
    slack_mode: str,
    steps: int,
    dt: float,
    dt_resp: float,
    eta_AL: float,
    eta_LB: float,
    omega_site: float,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)

    model = build_model_hamiltonian(
        dA=2,
        dL=dL,
        dB=2,
        eta_AL=eta_AL,
        eta_LB=eta_LB,
        omega_site=omega_site,
        rng=rng,
        slack_mode=slack_mode,
    )
    dims = model["dims"]
    H = model["H"]
    psi = initial_product_state(2, dL, 2)
    U_step = matrix_exp_hermitian(H, dt)

    U_readout_scramble = random_unitary(dL, rng)

    history = {
        "step": [],
        "time": [],
        "I_AB": [],
        "I_AL": [],
        "I_BL": [],
        "I3": [],
        "W_relational": [],
        "sv1": [],
        "sv2": [],
        "sv3": [],
        "bandwidth_ratio_21": [],
        "rank1_err": [],
        "rank2_err": [],
        "active_power": [],
        "slack_power": [],
        "leakage_ratio": [],
        "slack_occupancy": [],

        "W_relational_scrambled": [],
        "sv1_scrambled": [],
        "sv2_scrambled": [],
        "sv3_scrambled": [],
        "bandwidth_ratio_21_scrambled": [],
        "rank1_err_scrambled": [],
        "rank2_err_scrambled": [],
        "active_power_scrambled": [],
        "slack_power_scrambled": [],
        "leakage_ratio_scrambled": [],

        "damage_sv1": [],
        "damage_active_power": [],
        "leakage_gain": [],
    }

    best = {
        "step": 0,
        "time": 0.0,
        "W_relational": -1e99,
        "sv1": 0.0,
        "damage_sv1": 0.0,
        "damage_active_power": 0.0,
        "leakage_gain": 0.0,
        "slack_occupancy": 0.0,
    }

    for step in range(steps + 1):
        t = step * dt

        diag = diagnostics_for_state(
            psi=psi,
            H=H,
            dims=dims,
            dt_resp=dt_resp,
            link_readout_unitary=None,
        )
        diag_scrambled = diagnostics_for_state(
            psi=psi,
            H=H,
            dims=dims,
            dt_resp=dt_resp,
            link_readout_unitary=U_readout_scramble,
        )

        damage_sv1 = diag["sv1"] - diag_scrambled["sv1"]
        damage_active_power = diag["active_power"] - diag_scrambled["active_power"]
        leakage_gain = diag_scrambled["leakage_ratio"] - diag["leakage_ratio"]

        history["step"].append(step)
        history["time"].append(float(t))

        for k in [
            "I_AB", "I_AL", "I_BL", "I3", "W_relational",
            "sv1", "sv2", "sv3", "bandwidth_ratio_21",
            "rank1_err", "rank2_err",
            "active_power", "slack_power", "leakage_ratio",
            "slack_occupancy"
        ]:
            history[k].append(float(diag[k]))

        for k in [
            "W_relational", "sv1", "sv2", "sv3", "bandwidth_ratio_21",
            "rank1_err", "rank2_err",
            "active_power", "slack_power", "leakage_ratio"
        ]:
            history[f"{k}_scrambled"].append(float(diag_scrambled[k]))

        history["damage_sv1"].append(float(damage_sv1))
        history["damage_active_power"].append(float(damage_active_power))
        history["leakage_gain"].append(float(leakage_gain))

        if diag["W_relational"] > best["W_relational"]:
            best = {
                "step": int(step),
                "time": float(t),
                "W_relational": float(diag["W_relational"]),
                "sv1": float(diag["sv1"]),
                "damage_sv1": float(damage_sv1),
                "damage_active_power": float(damage_active_power),
                "leakage_gain": float(leakage_gain),
                "slack_occupancy": float(diag["slack_occupancy"]),
            }

        if step < steps:
            psi = U_step @ psi
            psi = normalize_state(psi)

    final = {k: history[k][-1] for k in history if len(history[k]) > 0}

    score = (
        2.0 * max(history["W_relational"])
        + 1.5 * max(history["sv1"])
        + 1.0 * max(history["damage_sv1"])
        + 1.0 * max(history["damage_active_power"])
        + 1.0 * max(history["leakage_gain"])
        - 1.5 * float(np.mean(history["slack_occupancy"]))
    )

    return {
        "label": label,
        "params": {
            "dL": dL,
            "slack_mode": slack_mode,
            "steps": steps,
            "dt": dt,
            "dt_resp": dt_resp,
            "eta_AL": eta_AL,
            "eta_LB": eta_LB,
            "omega_site": omega_site,
            "seed": seed,
        },
        "best": best,
        "final": final,
        "score": float(score),
        "history": history,
    }


# ============================================================
# Output helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv_summary(rows: List[Dict], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_case_bundle(cases: List[Dict], outdir: str):
    times = np.array(cases[0]["history"]["time"], dtype=float)

    plt.figure(figsize=(8, 5))
    for c in cases:
        plt.plot(times, c["history"]["W_relational"], label=c["label"])
    plt.xlabel("time")
    plt.ylabel("W_relational")
    plt.title("Relational witness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "w_relational.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for c in cases:
        plt.plot(times, c["history"]["sv1"], label=c["label"])
    plt.xlabel("time")
    plt.ylabel("leading response singular value")
    plt.title("A -> L response strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sv1.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for c in cases:
        plt.plot(times, c["history"]["damage_active_power"], label=c["label"])
    plt.xlabel("time")
    plt.ylabel("active_power(committed) - active_power(scrambled)")
    plt.title("Committed active-power loss under scrambled readout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "damage_active_power.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for c in cases:
        plt.plot(times, c["history"]["leakage_gain"], label=c["label"])
    plt.xlabel("time")
    plt.ylabel("leakage_ratio(scrambled) - leakage_ratio(committed)")
    plt.title("Slack leakage gain under scrambled readout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "leakage_gain.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for c in cases:
        plt.plot(times, c["history"]["slack_occupancy"], label=c["label"])
    plt.xlabel("time")
    plt.ylabel("link slack occupancy")
    plt.title("Slack usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "slack_occupancy.png"), dpi=160)
    plt.close()


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="HSF derived relational emergence probe v1")
    p.add_argument("--outdir", type=str, default="hsf_derived_relational_out")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--dt", type=float, default=0.08)
    p.add_argument("--dt_resp", type=float, default=0.08)
    p.add_argument("--eta_AL", type=float, default=0.85)
    p.add_argument("--eta_LB", type=float, default=0.85)
    p.add_argument("--omega_site", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    t0 = time.time()

    print("=" * 72)
    print("HSF DERIVED RELATIONAL EMERGENCE PROBE (v1, patched witness)")
    print("=" * 72)
    print(f"outdir: {args.outdir}")
    print(f"steps={args.steps}, dt={args.dt}, dt_resp={args.dt_resp}, seed={args.seed}")
    print(f"eta_AL={args.eta_AL}, eta_LB={args.eta_LB}, omega_site={args.omega_site}")
    print()
    print("Cases:")
    print("  1) committed_minimal_dL2")
    print("  2) committed_oversized_dL6_weakslack")
    print("  3) committed_oversized_dL6_randweak")
    print()

    cases = []

    c1 = run_single_case(
        label="committed_minimal_dL2",
        dL=2,
        slack_mode="none",
        steps=args.steps,
        dt=args.dt,
        dt_resp=args.dt_resp,
        eta_AL=args.eta_AL,
        eta_LB=args.eta_LB,
        omega_site=args.omega_site,
        seed=args.seed,
    )
    cases.append(c1)

    c2 = run_single_case(
        label="committed_oversized_dL6_weakslack",
        dL=6,
        slack_mode="weak",
        steps=args.steps,
        dt=args.dt,
        dt_resp=args.dt_resp,
        eta_AL=args.eta_AL,
        eta_LB=args.eta_LB,
        omega_site=args.omega_site,
        seed=args.seed + 1,
    )
    cases.append(c2)

    c3 = run_single_case(
        label="committed_oversized_dL6_randweak",
        dL=6,
        slack_mode="random_weak",
        steps=args.steps,
        dt=args.dt,
        dt_resp=args.dt_resp,
        eta_AL=args.eta_AL,
        eta_LB=args.eta_LB,
        omega_site=args.omega_site,
        seed=args.seed + 2,
    )
    cases.append(c3)

    print("-" * 72)
    print("SUMMARY")
    print("-" * 72)

    rows = []
    for c in cases:
        row = {
            "label": c["label"],
            "score": c["score"],
            "best_time": c["best"]["time"],
            "best_W_relational": c["best"]["W_relational"],
            "best_sv1": c["best"]["sv1"],
            "best_damage_sv1": c["best"]["damage_sv1"],
            "best_damage_active_power": c["best"]["damage_active_power"],
            "best_leakage_gain": c["best"]["leakage_gain"],
            "mean_slack_occupancy": float(np.mean(c["history"]["slack_occupancy"])),
            "final_W_relational": c["final"]["W_relational"],
            "final_sv1": c["final"]["sv1"],
            "final_damage_sv1": c["final"]["damage_sv1"],
            "final_damage_active_power": c["final"]["damage_active_power"],
            "final_leakage_gain": c["final"]["leakage_gain"],
            "final_slack_occupancy": c["final"]["slack_occupancy"],
        }
        rows.append(row)

        print(
            f"{row['label']}: "
            f"score={row['score']:.4f}  "
            f"best_W={row['best_W_relational']:.4f}  "
            f"best_sv1={row['best_sv1']:.4e}  "
            f"best_dmg_sv1={row['best_damage_sv1']:.4e}  "
            f"best_dmg_active={row['best_damage_active_power']:.4e}  "
            f"best_leak_gain={row['best_leakage_gain']:.4e}  "
            f"mean_slack={row['mean_slack_occupancy']:.4f}"
        )

    payload = {
        "params": vars(args),
        "cases": cases,
        "summary_rows": rows,
        "runtime_sec": time.time() - t0,
    }

    save_json(payload, os.path.join(args.outdir, "summary.json"))
    save_csv_summary(rows, os.path.join(args.outdir, "summary.csv"))
    plot_case_bundle(cases, args.outdir)

    print()
    print("Saved:")
    print(f"  {os.path.join(args.outdir, 'summary.json')}")
    print(f"  {os.path.join(args.outdir, 'summary.csv')}")
    print(f"  {os.path.join(args.outdir, 'w_relational.png')}")
    print(f"  {os.path.join(args.outdir, 'sv1.png')}")
    print(f"  {os.path.join(args.outdir, 'damage_active_power.png')}")
    print(f"  {os.path.join(args.outdir, 'leakage_gain.png')}")
    print(f"  {os.path.join(args.outdir, 'slack_occupancy.png')}")
    print()
    print(f"Runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()