# hsf_inferred_interface_sweep_v1.py

r"""
HSF INFERRED INTERFACE SWEEP (self-contained, v1)

Purpose
-------
Self-contained sweep script for inferred interface emergence in a fixed
four-site chain:

    A - X - Y - B

No external probe script required.

What it does
------------
For each parameter setting, it:

1. Builds a fixed 4-site qubit chain with nearest-neighbor couplings only.
2. Evolves a pure state under unitary dynamics.
3. Treats the middle block M = X⊗Y as a candidate interface region.
4. Builds the full A -> M response matrix.
5. Infers a low-dimensional active subspace inside M from the response.
6. Measures:
     - W_relational
     - active capture ratio
     - active-power loss under misaligned readout
     - leakage gain into the complement under misaligned readout
7. Sweeps:
     - active_dim
     - coupling strength
     - omega_site

Outputs
-------
aggregate_summary.json
aggregate_summary.csv
aggregate_case_rows.csv
dim_sweep.csv
coupling_sweep.csv
omega_sweep.csv

Run
---
python hsf_inferred_interface_sweep_v1.py --outdir inferred_sweep_out --steps 120 --dt 0.08 --seed 0
"""

import argparse
import csv
import json
import math
import os
import statistics
import time
from typing import Any, Dict, List, Tuple

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


def hermitian_basis(dim: int) -> List[np.ndarray]:
    basis = []
    for i in range(dim):
        e = np.zeros((dim, dim), dtype=complex)
        e[i, i] = 1.0
        basis.append(e)
    for i in range(dim):
        for j in range(i + 1, dim):
            s = np.zeros((dim, dim), dtype=complex)
            a = np.zeros((dim, dim), dtype=complex)
            s[i, j] = s[j, i] = 1.0
            a[i, j] = -1j
            a[j, i] = 1j
            basis.append(s / math.sqrt(2.0))
            basis.append(a / math.sqrt(2.0))
    return basis


def privileged_subspace_basis(dimM: int, U_sub: np.ndarray, keep_dim: int, complement: bool) -> List[np.ndarray]:
    if complement:
        idx = list(range(keep_dim, dimM))
    else:
        idx = list(range(keep_dim))

    if len(idx) == 0:
        return []

    basis = []

    for i in idx:
        P = np.zeros((dimM, dimM), dtype=complex)
        P[i, i] = 1.0
        basis.append(U_sub @ P @ dagger(U_sub))

    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i = idx[a]
            j = idx[b]
            X = np.zeros((dimM, dimM), dtype=complex)
            Y = np.zeros((dimM, dimM), dtype=complex)
            X[i, j] = X[j, i] = 1.0
            Y[i, j] = -1j
            Y[j, i] = 1j
            basis.append(U_sub @ (X / math.sqrt(2.0)) @ dagger(U_sub))
            basis.append(U_sub @ (Y / math.sqrt(2.0)) @ dagger(U_sub))

    return basis


# ============================================================
# Model construction
# ============================================================

def build_chain_hamiltonian(
    eta_AX: float,
    eta_XY: float,
    eta_YB: float,
    omega_site: float,
) -> Dict[str, np.ndarray]:
    dims = [2, 2, 2, 2]
    gens = su2_generators()

    locals_ops = {}
    for site in range(4):
        locals_ops[site] = [local_operator_on_subsystem(g, dims, site) for g in gens]

    ZA = locals_ops[0][2]
    ZX = locals_ops[1][2]
    ZY = locals_ops[2][2]
    ZB = locals_ops[3][2]

    H_local = omega_site * (ZA + 0.8 * ZX + 0.8 * ZY + ZB)

    def heisenberg_pair(i: int, j: int, eta: float) -> np.ndarray:
        Xi, Yi, Zi = locals_ops[i]
        Xj, Yj, Zj = locals_ops[j]
        return eta * (Xi @ Xj + Yi @ Yj + Zi @ Zj)

    H_couple = (
        heisenberg_pair(0, 1, eta_AX) +
        heisenberg_pair(1, 2, eta_XY) +
        heisenberg_pair(2, 3, eta_YB)
    )

    H = H_local + H_couple
    return {"dims": dims, "H": H}


def initial_product_state() -> np.ndarray:
    up = np.array([1.0, 0.0], dtype=complex)
    dn = np.array([0.0, 1.0], dtype=complex)
    psi = kron_list([up[:, None], dn[:, None], dn[:, None], dn[:, None]]).reshape(-1)
    return normalize_state(psi)


# ============================================================
# Response / inferred subspace
# ============================================================

def response_matrix_A_to_M(
    H: np.ndarray,
    psi: np.ndarray,
    dims: List[int],
    dt: float,
    basis_M: List[np.ndarray],
    basis_transform_M: np.ndarray = None,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    U_step = matrix_exp_hermitian(H, dt)
    poke_ops = [local_operator_on_subsystem(G, dims, 0) for G in su2_generators()]

    readout_basis = basis_M
    if basis_transform_M is not None:
        U = basis_transform_M
        readout_basis = [dagger(U) @ B @ U for B in basis_M]

    full_ops = []
    for B in readout_basis:
        full_ops.append(kron_list([
            np.eye(2, dtype=complex),
            B,
            np.eye(2, dtype=complex),
        ]))

    psi0 = U_step @ psi
    base_vals = np.array([np.real(np.vdot(psi0, O @ psi0)) for O in full_ops], dtype=float)

    rows = []
    for P in poke_ops:
        U_poke = matrix_exp_hermitian(P, eps)
        psi_p = U_step @ (U_poke @ psi)
        vals = np.array([np.real(np.vdot(psi_p, O @ psi_p)) for O in full_ops], dtype=float)
        rows.append((vals - base_vals) / eps)

    M = np.array(rows, dtype=float)
    svals = np.linalg.svd(M, compute_uv=False) if M.size > 0 else np.array([], dtype=float)
    return M, svals


def response_power(M: np.ndarray) -> float:
    if M.size == 0:
        return 0.0
    return float(np.sum(M * M))


def infer_active_subspace_from_response(
    M_full: np.ndarray,
    full_basis_M: List[np.ndarray],
    active_dim: int,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    dimM = full_basis_M[0].shape[0]
    if dimM != 4:
        raise ValueError("This sweep assumes middle block dim = 4.")

    if M_full.size == 0:
        U_active = np.eye(dimM, dtype=complex)
        return U_active, [], []

    U, s, Vh = np.linalg.svd(M_full, full_matrices=False)
    coeff_power = np.sum((s[:, None] * Vh) ** 2, axis=0)

    K = np.zeros((dimM, dimM), dtype=complex)
    for w, O in zip(coeff_power, full_basis_M):
        K += float(w) * (O @ O)

    evals, evecs = np.linalg.eigh(0.5 * (K + dagger(K)))
    order = np.argsort(evals)[::-1]
    U_active = evecs[:, order]

    active_basis = privileged_subspace_basis(dimM=dimM, U_sub=U_active, keep_dim=active_dim, complement=False)
    complement_basis = privileged_subspace_basis(dimM=dimM, U_sub=U_active, keep_dim=active_dim, complement=True)
    return U_active, active_basis, complement_basis


# ============================================================
# Diagnostics
# ============================================================

def diagnostics_for_state(
    psi: np.ndarray,
    H: np.ndarray,
    dims: List[int],
    dt_resp: float,
    active_dim: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    I_AB = mutual_information_pure(psi, dims, [0], [3])
    I_AM = mutual_information_pure(psi, dims, [0], [1, 2])
    I_MB = mutual_information_pure(psi, dims, [1, 2], [3])
    I3 = interaction_information_pure(psi, dims, [0], [1, 2], [3])
    W = I_AM + I_MB - I_AB

    full_basis_M = hermitian_basis(4)
    M_full, s_full = response_matrix_A_to_M(
        H=H,
        psi=psi,
        dims=dims,
        dt=dt_resp,
        basis_M=full_basis_M,
        basis_transform_M=None,
    )
    full_power = response_power(M_full)

    _, active_basis, complement_basis = infer_active_subspace_from_response(
        M_full=M_full,
        full_basis_M=full_basis_M,
        active_dim=active_dim,
    )

    M_active, s_active = response_matrix_A_to_M(
        H=H,
        psi=psi,
        dims=dims,
        dt=dt_resp,
        basis_M=active_basis,
        basis_transform_M=None,
    )
    active_power = response_power(M_active)

    if len(complement_basis) > 0:
        M_comp, _ = response_matrix_A_to_M(
            H=H,
            psi=psi,
            dims=dims,
            dt=dt_resp,
            basis_M=complement_basis,
            basis_transform_M=None,
        )
    else:
        M_comp = np.zeros((3, 0), dtype=float)
    comp_power = response_power(M_comp)

    capture_ratio = float(active_power / full_power) if full_power > 1e-15 else 0.0
    leakage_ratio = float(comp_power / active_power) if active_power > 1e-15 else 0.0

    U_scramble = random_unitary(4, rng)

    M_active_scr, _ = response_matrix_A_to_M(
        H=H,
        psi=psi,
        dims=dims,
        dt=dt_resp,
        basis_M=active_basis,
        basis_transform_M=U_scramble,
    )
    active_power_scr = response_power(M_active_scr)

    if len(complement_basis) > 0:
        M_comp_scr, _ = response_matrix_A_to_M(
            H=H,
            psi=psi,
            dims=dims,
            dt=dt_resp,
            basis_M=complement_basis,
            basis_transform_M=U_scramble,
        )
    else:
        M_comp_scr = np.zeros((3, 0), dtype=float)
    comp_power_scr = response_power(M_comp_scr)

    leakage_ratio_scr = float(comp_power_scr / active_power_scr) if active_power_scr > 1e-15 else 0.0

    full_sv1 = float(s_full[0]) if len(s_full) >= 1 else 0.0
    full_sv2 = float(s_full[1]) if len(s_full) >= 2 else 0.0
    full_sv3 = float(s_full[2]) if len(s_full) >= 3 else 0.0

    return {
        "I_AB": float(I_AB),
        "I_AM": float(I_AM),
        "I_MB": float(I_MB),
        "I3": float(I3),
        "W_relational": float(W),
        "full_sv1": full_sv1,
        "full_sv2": full_sv2,
        "full_sv3": full_sv3,
        "full_power": full_power,
        "active_power": active_power,
        "comp_power": comp_power,
        "capture_ratio": capture_ratio,
        "leakage_ratio": leakage_ratio,
        "active_power_scrambled": active_power_scr,
        "comp_power_scrambled": comp_power_scr,
        "leakage_ratio_scrambled": leakage_ratio_scr,
        "damage_active_power": float(active_power - active_power_scr),
        "leakage_gain": float(leakage_ratio_scr - leakage_ratio),
    }


# ============================================================
# Per-case runner
# ============================================================

def run_single_case(
    label: str,
    eta_AX: float,
    eta_XY: float,
    eta_YB: float,
    omega_site: float,
    steps: int,
    dt: float,
    dt_resp: float,
    active_dim: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    model = build_chain_hamiltonian(
        eta_AX=eta_AX,
        eta_XY=eta_XY,
        eta_YB=eta_YB,
        omega_site=omega_site,
    )
    dims = model["dims"]
    H = model["H"]
    psi = initial_product_state()
    U_step = matrix_exp_hermitian(H, dt)

    best = {
        "time": 0.0,
        "W_relational": -1e99,
        "capture_ratio": 0.0,
        "damage_active_power": 0.0,
        "leakage_gain": 0.0,
    }

    for step in range(steps + 1):
        t = step * dt
        diag = diagnostics_for_state(
            psi=psi,
            H=H,
            dims=dims,
            dt_resp=dt_resp,
            active_dim=active_dim,
            rng=rng,
        )

        if diag["W_relational"] > best["W_relational"]:
            best = {
                "time": float(t),
                "W_relational": float(diag["W_relational"]),
                "capture_ratio": float(diag["capture_ratio"]),
                "damage_active_power": float(diag["damage_active_power"]),
                "leakage_gain": float(diag["leakage_gain"]),
            }

        if step < steps:
            psi = U_step @ psi
            psi = normalize_state(psi)

    score = (
        2.0 * best["W_relational"] +
        2.0 * best["capture_ratio"] +
        1.0 * best["damage_active_power"] +
        1.0 * best["leakage_gain"]
    )

    return {
        "label": label,
        "score": float(score),
        "best_time": best["time"],
        "best_W_relational": best["W_relational"],
        "best_capture_ratio": best["capture_ratio"],
        "best_damage_active_power": best["damage_active_power"],
        "best_leakage_gain": best["leakage_gain"],
    }


# ============================================================
# Sweep helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def mean_or_nan(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    return float(statistics.mean(vals))


def stdev_or_zero(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(statistics.pstdev(vals))


def summarize_group(
    run_tag: str,
    active_dim: int,
    eta_ax: float,
    eta_xy: float,
    eta_yb: float,
    omega_site: float,
    case_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    score_vals = [r["score"] for r in case_rows]
    cap_vals = [r["best_capture_ratio"] for r in case_rows]
    dmg_vals = [r["best_damage_active_power"] for r in case_rows]
    leak_vals = [r["best_leakage_gain"] for r in case_rows]
    w_vals = [r["best_W_relational"] for r in case_rows]

    return {
        "run_tag": run_tag,
        "active_dim": active_dim,
        "eta_AX": eta_ax,
        "eta_XY": eta_xy,
        "eta_YB": eta_yb,
        "omega_site": omega_site,
        "n_cases": len(case_rows),
        "score_mean": mean_or_nan(score_vals),
        "score_std": stdev_or_zero(score_vals),
        "best_capture_ratio_mean": mean_or_nan(cap_vals),
        "best_capture_ratio_std": stdev_or_zero(cap_vals),
        "best_damage_active_power_mean": mean_or_nan(dmg_vals),
        "best_damage_active_power_std": stdev_or_zero(dmg_vals),
        "best_leakage_gain_mean": mean_or_nan(leak_vals),
        "best_leakage_gain_std": stdev_or_zero(leak_vals),
        "best_W_relational_mean": mean_or_nan(w_vals),
        "best_W_relational_std": stdev_or_zero(w_vals),
    }


def choose_best_dimension(rows: List[Dict[str, Any]]) -> int:
    best_dim = None
    best_score = None
    by_dim: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_dim.setdefault(int(r["active_dim"]), []).append(r)

    for dim, grp in by_dim.items():
        cap = mean_or_nan([g["best_capture_ratio_mean"] for g in grp])
        leak = max(0.0, mean_or_nan([g["best_leakage_gain_mean"] for g in grp]))
        dmg = max(0.0, mean_or_nan([g["best_damage_active_power_mean"] for g in grp]))
        score = 2.0 * cap + math.log1p(leak) + dmg
        if best_score is None or score > best_score:
            best_score = score
            best_dim = dim

    return 2 if best_dim is None else int(best_dim)


def run_group(
    run_tag: str,
    active_dim: int,
    eta_ax: float,
    eta_xy: float,
    eta_yb: float,
    omega_site: float,
    steps: int,
    dt: float,
    dt_resp: float,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    case_rows = [
        run_single_case(
            label="balanced_chain",
            eta_AX=eta_ax,
            eta_XY=eta_xy,
            eta_YB=eta_yb,
            omega_site=omega_site,
            steps=steps,
            dt=dt,
            dt_resp=dt_resp,
            active_dim=active_dim,
            seed=seed,
        ),
        run_single_case(
            label="middle_heavier",
            eta_AX=eta_ax,
            eta_XY=1.25 * eta_xy,
            eta_YB=eta_yb,
            omega_site=omega_site,
            steps=steps,
            dt=dt,
            dt_resp=dt_resp,
            active_dim=active_dim,
            seed=seed + 1,
        ),
        run_single_case(
            label="endpoint_heavier",
            eta_AX=1.15 * eta_ax,
            eta_XY=eta_xy,
            eta_YB=1.15 * eta_yb,
            omega_site=omega_site,
            steps=steps,
            dt=dt,
            dt_resp=dt_resp,
            active_dim=active_dim,
            seed=seed + 2,
        ),
    ]

    group = summarize_group(
        run_tag=run_tag,
        active_dim=active_dim,
        eta_ax=eta_ax,
        eta_xy=eta_xy,
        eta_yb=eta_yb,
        omega_site=omega_site,
        case_rows=case_rows,
    )

    enriched_rows = []
    for row in case_rows:
        row2 = dict(row)
        row2.update({
            "run_tag": run_tag,
            "active_dim": active_dim,
            "eta_AX": eta_ax,
            "eta_XY": eta_xy,
            "eta_YB": eta_yb,
            "omega_site": omega_site,
        })
        enriched_rows.append(row2)

    return group, enriched_rows


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Self-contained inferred interface sweep")
    p.add_argument("--outdir", type=str, default="inferred_sweep_out")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--dt", type=float, default=0.08)
    p.add_argument("--dt_resp", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--base_eta", type=float, default=0.85)
    p.add_argument("--base_omega", type=float, default=0.35)

    p.add_argument("--dim_list", type=str, default="1,2,3")
    p.add_argument("--eta_list", type=str, default="0.6,0.85,1.0")
    p.add_argument("--omega_list", type=str, default="0.2,0.35,0.5")

    p.add_argument("--skip_dim_sweep", action="store_true")
    p.add_argument("--skip_coupling_sweep", action="store_true")
    p.add_argument("--skip_omega_sweep", action="store_true")
    return p.parse_args()


def parse_num_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    dim_list = parse_num_list(args.dim_list, int)
    eta_list = parse_num_list(args.eta_list, float)
    omega_list = parse_num_list(args.omega_list, float)

    print("=" * 72)
    print("HSF INFERRED INTERFACE SWEEP (self-contained, v1)")
    print("=" * 72)
    print(f"outdir: {args.outdir}")
    print(f"steps={args.steps}, dt={args.dt}, dt_resp={args.dt_resp}, seed={args.seed}")
    print(f"base_eta={args.base_eta}, base_omega={args.base_omega}")
    print()

    all_group_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []

    dim_summary_rows: List[Dict[str, Any]] = []
    if not args.skip_dim_sweep:
        print("-" * 72)
        print("DIMENSION SWEEP")
        print("-" * 72)
        for d in dim_list:
            tag = f"dim{d}"
            group, case_rows = run_group(
                run_tag=tag,
                active_dim=d,
                eta_ax=args.base_eta,
                eta_xy=args.base_eta,
                eta_yb=args.base_eta,
                omega_site=args.base_omega,
                steps=args.steps,
                dt=args.dt,
                dt_resp=args.dt_resp,
                seed=args.seed,
            )
            dim_summary_rows.append(group)
            all_group_rows.append(group)
            all_case_rows.extend(case_rows)

            print(
                f"{tag}: "
                f"capture_mean={group['best_capture_ratio_mean']:.4f}  "
                f"dmg_active_mean={group['best_damage_active_power_mean']:.4e}  "
                f"leak_gain_mean={group['best_leakage_gain_mean']:.4f}  "
                f"score_mean={group['score_mean']:.4f}"
            )

        save_csv(dim_summary_rows, os.path.join(args.outdir, "dim_sweep.csv"))
    else:
        print("Skipping dimension sweep.")

    best_dim = choose_best_dimension(dim_summary_rows) if dim_summary_rows else 2
    print()
    print(f"Chosen active_dim for subsequent sweeps: {best_dim}")
    print()

    coupling_summary_rows: List[Dict[str, Any]] = []
    if not args.skip_coupling_sweep:
        print("-" * 72)
        print("COUPLING SWEEP")
        print("-" * 72)
        for eta in eta_list:
            tag = f"eta{str(eta).replace('.', 'p')}"
            group, case_rows = run_group(
                run_tag=tag,
                active_dim=best_dim,
                eta_ax=eta,
                eta_xy=eta,
                eta_yb=eta,
                omega_site=args.base_omega,
                steps=args.steps,
                dt=args.dt,
                dt_resp=args.dt_resp,
                seed=args.seed,
            )
            coupling_summary_rows.append(group)
            all_group_rows.append(group)
            all_case_rows.extend(case_rows)

            print(
                f"{tag}: "
                f"capture_mean={group['best_capture_ratio_mean']:.4f}  "
                f"dmg_active_mean={group['best_damage_active_power_mean']:.4e}  "
                f"leak_gain_mean={group['best_leakage_gain_mean']:.4f}  "
                f"score_mean={group['score_mean']:.4f}"
            )

        save_csv(coupling_summary_rows, os.path.join(args.outdir, "coupling_sweep.csv"))
    else:
        print("Skipping coupling sweep.")

    omega_summary_rows: List[Dict[str, Any]] = []
    if not args.skip_omega_sweep:
        print()
        print("-" * 72)
        print("OMEGA SWEEP")
        print("-" * 72)
        for om in omega_list:
            tag = f"om{str(om).replace('.', 'p')}"
            group, case_rows = run_group(
                run_tag=tag,
                active_dim=best_dim,
                eta_ax=args.base_eta,
                eta_xy=args.base_eta,
                eta_yb=args.base_eta,
                omega_site=om,
                steps=args.steps,
                dt=args.dt,
                dt_resp=args.dt_resp,
                seed=args.seed,
            )
            omega_summary_rows.append(group)
            all_group_rows.append(group)
            all_case_rows.extend(case_rows)

            print(
                f"{tag}: "
                f"capture_mean={group['best_capture_ratio_mean']:.4f}  "
                f"dmg_active_mean={group['best_damage_active_power_mean']:.4e}  "
                f"leak_gain_mean={group['best_leakage_gain_mean']:.4f}  "
                f"score_mean={group['score_mean']:.4f}"
            )

        save_csv(omega_summary_rows, os.path.join(args.outdir, "omega_sweep.csv"))
    else:
        print("Skipping omega sweep.")

    aggregate = {
        "config": {
            "steps": args.steps,
            "dt": args.dt,
            "dt_resp": args.dt_resp,
            "seed": args.seed,
            "base_eta": args.base_eta,
            "base_omega": args.base_omega,
            "dim_list": dim_list,
            "eta_list": eta_list,
            "omega_list": omega_list,
            "chosen_active_dim": best_dim,
        },
        "dim_summary_rows": dim_summary_rows,
        "coupling_summary_rows": coupling_summary_rows,
        "omega_summary_rows": omega_summary_rows,
        "all_group_rows": all_group_rows,
        "all_case_rows": all_case_rows,
    }

    save_json(aggregate, os.path.join(args.outdir, "aggregate_summary.json"))
    save_csv(all_group_rows, os.path.join(args.outdir, "aggregate_summary.csv"))
    save_csv(all_case_rows, os.path.join(args.outdir, "aggregate_case_rows.csv"))

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Saved:")
    print(f"  {os.path.join(args.outdir, 'aggregate_summary.json')}")
    print(f"  {os.path.join(args.outdir, 'aggregate_summary.csv')}")
    print(f"  {os.path.join(args.outdir, 'aggregate_case_rows.csv')}")
    if dim_summary_rows:
        print(f"  {os.path.join(args.outdir, 'dim_sweep.csv')}")
    if coupling_summary_rows:
        print(f"  {os.path.join(args.outdir, 'coupling_sweep.csv')}")
    if omega_summary_rows:
        print(f"  {os.path.join(args.outdir, 'omega_sweep.csv')}")


if __name__ == "__main__":
    main()