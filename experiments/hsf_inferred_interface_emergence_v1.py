# hsf_inferred_interface_emergence_v1.py

r"""
HSF INFERRED INTERFACE EMERGENCE PROBE (v1)

Goal
----
Move beyond the explicit A-L-B toy by removing the hand-declared interface
register. Instead, build a fixed larger system

    A - X - Y - B

with local couplings only, and ask whether the dynamics themselves pick out
a low-dimensional effective interface sector inside the middle block

    M = X \otimes Y

that carries the dominant endpoint-to-middle response.

What this script does
---------------------
1. Build a fixed 4-site chain:
      A (qubit) - X (qubit) - Y (qubit) - B (qubit)

2. Evolve a pure state under a local Hamiltonian with nearest-neighbor
   couplings only.

3. Treat the middle block M = X⊗Y as the candidate "interface region".

4. At each time step:
   - compute endpoint/middle mutual informations
   - build an A -> M response matrix using a full Hermitian basis on M
   - do an SVD of that response matrix
   - infer the best low-dimensional active subspace of M from the leading
     right singular vectors
   - construct a privileged active readout dictionary on that inferred
     subspace
   - compare active vs complement response power
   - scramble the M basis relative to that inferred dictionary
   - measure active-power loss and leakage gain

Interpretation
--------------
This supports inferred interface emergence when:
  - the A -> M response is dominated by a small number of singular directions
  - a low-dimensional active subspace captures most response power
  - the complementary directions are weak
  - basis misalignment reduces active power and increases leakage

Outputs
-------
summary.json
summary.csv
w_relational.png
active_capture_ratio.png
damage_active_power.png
leakage_gain.png
sv_spectrum.png

Example
-------
python hsf_inferred_interface_emergence_v1.py --outdir hsf_inferred_interface_out --steps 120 --dt 0.08 --seed 0
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


# ============================================================
# Model construction
# ============================================================

def build_chain_hamiltonian(
    eta_AX: float,
    eta_XY: float,
    eta_YB: float,
    omega_site: float,
) -> Dict[str, np.ndarray]:
    """
    4-site qubit chain: A - X - Y - B
    """
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
    return {"dims": dims, "H": H, "H_local": H_local, "H_couple": H_couple}


def initial_product_state() -> np.ndarray:
    """
    A up, X down, Y down, B down
    """
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
    """
    Poke A, evolve one step, read out on middle block M = X⊗Y.

    basis_M: operators on the 4D middle block.
    basis_transform_M: optional unitary applied as O -> U^\dagger O U
                       to simulate readout misalignment.
    """
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


def infer_active_subspace_from_response(
    M_full: np.ndarray,
    full_basis_M: List[np.ndarray],
    active_dim: int,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], Dict[str, float]]:
    """
    Infer an active subspace inside M from the right singular vectors of the
    A->M response matrix.

    We treat each operator in full_basis_M as a coordinate axis in readout space.
    The leading right singular directions define an operator subspace.
    We use that to score / choose a state-space active subspace by diagonalizing
    a weighted operator covariance on M.

    Returns:
      U_active        : 4x4 unitary whose first active_dim columns define
                        the inferred active state subspace
      active_basis    : privileged active readout dictionary on that subspace
      complement_basis: privileged complement/slack readout dictionary
      info            : capture metrics
    """
    dimM = full_basis_M[0].shape[0]
    if dimM != 4:
        raise ValueError("This v1 assumes middle block dim = 4.")

    if M_full.size == 0:
        U_active = np.eye(dimM, dtype=complex)
        return U_active, [], [], {
            "capture_ratio": 0.0,
            "full_power": 0.0,
            "active_dim": active_dim,
        }

    U, s, Vh = np.linalg.svd(M_full, full_matrices=False)
    coeff_power = np.sum((s[:, None] * Vh) ** 2, axis=0)

    # Build an operator covariance / salience matrix on M
    K = np.zeros((dimM, dimM), dtype=complex)
    for w, O in zip(coeff_power, full_basis_M):
        K += float(w) * (O @ O)

    evals, evecs = np.linalg.eigh(0.5 * (K + dagger(K)))
    order = np.argsort(evals)[::-1]
    U_active = evecs[:, order]

    active_basis = privileged_subspace_basis(dimM=dimM, U_sub=U_active, keep_dim=active_dim, complement=False)
    complement_basis = privileged_subspace_basis(dimM=dimM, U_sub=U_active, keep_dim=active_dim, complement=True)

    # Capture ratio
    M_active, _ = response_matrix_A_to_M(
        H=np.zeros((16, 16), dtype=complex),  # dummy, not used for power below
        psi=np.zeros(16, dtype=complex),       # dummy
        dims=[2, 2, 2, 2],                     # dummy
        dt=0.0,                                # dummy
        basis_M=active_basis,
    )
    # above dummy call shape not needed; overwrite below
    full_power = response_power(M_full)

    return U_active, active_basis, complement_basis, {
        "capture_ratio": math.nan,  # will be filled by caller with actual active_power/full_power
        "full_power": full_power,
        "active_dim": active_dim,
    }


def privileged_subspace_basis(dimM: int, U_sub: np.ndarray, keep_dim: int, complement: bool) -> List[np.ndarray]:
    """
    Build a privileged incomplete basis on either the active subspace or its complement.
    """
    if complement:
        idx = list(range(keep_dim, dimM))
    else:
        idx = list(range(keep_dim))

    if len(idx) == 0:
        return []

    basis = []

    # projectors
    for i in idx:
        P = np.zeros((dimM, dimM), dtype=complex)
        P[i, i] = 1.0
        basis.append(U_sub @ P @ dagger(U_sub))

    # pair coherences within chosen sector
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
    # A = 0, M = [1,2], B = 3
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

    U_inferred, active_basis, complement_basis, info = infer_active_subspace_from_response(
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

    M_comp, _ = response_matrix_A_to_M(
        H=H,
        psi=psi,
        dims=dims,
        dt=dt_resp,
        basis_M=complement_basis,
        basis_transform_M=None,
    ) if len(complement_basis) > 0 else (np.zeros((3, 0), dtype=float), np.array([], dtype=float))
    comp_power = response_power(M_comp)

    capture_ratio = float(active_power / full_power) if full_power > 1e-15 else 0.0
    leakage_ratio = float(comp_power / active_power) if active_power > 1e-15 else 0.0

    # Scrambled readout relative to inferred privileged dictionary
    U_scramble = random_unitary(4, rng)

    M_active_scr, s_active_scr = response_matrix_A_to_M(
        H=H,
        psi=psi,
        dims=dims,
        dt=dt_resp,
        basis_M=active_basis,
        basis_transform_M=U_scramble,
    )
    active_power_scr = response_power(M_active_scr)

    M_comp_scr, _ = response_matrix_A_to_M(
        H=H,
        psi=psi,
        dims=dims,
        dt=dt_resp,
        basis_M=complement_basis,
        basis_transform_M=U_scramble,
    ) if len(complement_basis) > 0 else (np.zeros((3, 0), dtype=float), np.array([], dtype=float))
    comp_power_scr = response_power(M_comp_scr)

    leakage_ratio_scr = float(comp_power_scr / active_power_scr) if active_power_scr > 1e-15 else 0.0

    damage_active_power = active_power - active_power_scr
    leakage_gain = leakage_ratio_scr - leakage_ratio

    sv1 = float(s_active[0]) if len(s_active) >= 1 else 0.0
    sv2 = float(s_active[1]) if len(s_active) >= 2 else 0.0
    sv3 = float(s_active[2]) if len(s_active) >= 3 else 0.0

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

        "sv1": sv1,
        "sv2": sv2,
        "sv3": sv3,
        "rank1_err_active": best_rank_k_error(M_active, 1),
        "rank2_err_active": best_rank_k_error(M_active, 2),

        "active_power": active_power,
        "comp_power": comp_power,
        "capture_ratio": capture_ratio,
        "leakage_ratio": leakage_ratio,

        "active_power_scrambled": active_power_scr,
        "comp_power_scrambled": comp_power_scr,
        "leakage_ratio_scrambled": leakage_ratio_scr,

        "damage_active_power": float(damage_active_power),
        "leakage_gain": float(leakage_gain),
    }


# ============================================================
# Runner
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
) -> Dict:
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

    history = {
        "step": [],
        "time": [],
        "I_AB": [],
        "I_AM": [],
        "I_MB": [],
        "I3": [],
        "W_relational": [],
        "full_sv1": [],
        "full_sv2": [],
        "full_sv3": [],
        "sv1": [],
        "sv2": [],
        "sv3": [],
        "active_power": [],
        "comp_power": [],
        "capture_ratio": [],
        "leakage_ratio": [],
        "damage_active_power": [],
        "leakage_gain": [],
    }

    best = {
        "step": 0,
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

        history["step"].append(step)
        history["time"].append(float(t))
        for k in [
            "I_AB", "I_AM", "I_MB", "I3", "W_relational",
            "full_sv1", "full_sv2", "full_sv3",
            "sv1", "sv2", "sv3",
            "active_power", "comp_power", "capture_ratio",
            "leakage_ratio", "damage_active_power", "leakage_gain"
        ]:
            history[k].append(float(diag[k]))

        if diag["W_relational"] > best["W_relational"]:
            best = {
                "step": int(step),
                "time": float(t),
                "W_relational": float(diag["W_relational"]),
                "capture_ratio": float(diag["capture_ratio"]),
                "damage_active_power": float(diag["damage_active_power"]),
                "leakage_gain": float(diag["leakage_gain"]),
            }

        if step < steps:
            psi = U_step @ psi
            psi = normalize_state(psi)

    final = {k: history[k][-1] for k in history if len(history[k]) > 0}

    score = (
        2.0 * max(history["W_relational"])
        + 2.0 * max(history["capture_ratio"])
        + 1.0 * max(history["damage_active_power"])
        + 1.0 * max(history["leakage_gain"])
    )

    return {
        "label": label,
        "params": {
            "eta_AX": eta_AX,
            "eta_XY": eta_XY,
            "eta_YB": eta_YB,
            "omega_site": omega_site,
            "steps": steps,
            "dt": dt,
            "dt_resp": dt_resp,
            "active_dim": active_dim,
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
        plt.plot(times, c["history"]["capture_ratio"], label=c["label"])
    plt.xlabel("time")
    plt.ylabel("active_power / full_power")
    plt.title("Inferred active-subspace capture ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "active_capture_ratio.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for c in cases:
        plt.plot(times, c["history"]["damage_active_power"], label=c["label"])
    plt.xlabel("time")
    plt.ylabel("active_power - active_power_scrambled")
    plt.title("Active-power loss under misaligned readout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "damage_active_power.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for c in cases:
        plt.plot(times, c["history"]["leakage_gain"], label=c["label"])
    plt.xlabel("time")
    plt.ylabel("leakage_ratio_scrambled - leakage_ratio")
    plt.title("Leakage gain under misaligned readout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "leakage_gain.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for c in cases:
        plt.plot(times, c["history"]["full_sv1"], label=f"{c['label']}: sv1")
        plt.plot(times, c["history"]["full_sv2"], linestyle="--", label=f"{c['label']}: sv2")
    plt.xlabel("time")
    plt.ylabel("singular values")
    plt.title("Full A -> M response spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sv_spectrum.png"), dpi=160)
    plt.close()


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="HSF inferred interface emergence probe v1")
    p.add_argument("--outdir", type=str, default="hsf_inferred_interface_out")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--dt", type=float, default=0.08)
    p.add_argument("--dt_resp", type=float, default=0.08)
    p.add_argument("--eta_AX", type=float, default=0.85)
    p.add_argument("--eta_XY", type=float, default=0.85)
    p.add_argument("--eta_YB", type=float, default=0.85)
    p.add_argument("--omega_site", type=float, default=0.35)
    p.add_argument("--active_dim", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)
    t0 = time.time()

    print("=" * 72)
    print("HSF INFERRED INTERFACE EMERGENCE PROBE (v1)")
    print("=" * 72)
    print(f"outdir: {args.outdir}")
    print(f"steps={args.steps}, dt={args.dt}, dt_resp={args.dt_resp}, seed={args.seed}")
    print(f"eta_AX={args.eta_AX}, eta_XY={args.eta_XY}, eta_YB={args.eta_YB}, omega_site={args.omega_site}")
    print(f"active_dim={args.active_dim}")
    print()

    cases = [
        run_single_case(
            label="balanced_chain",
            eta_AX=args.eta_AX,
            eta_XY=args.eta_XY,
            eta_YB=args.eta_YB,
            omega_site=args.omega_site,
            steps=args.steps,
            dt=args.dt,
            dt_resp=args.dt_resp,
            active_dim=args.active_dim,
            seed=args.seed,
        ),
        run_single_case(
            label="middle_heavier",
            eta_AX=args.eta_AX,
            eta_XY=1.25 * args.eta_XY,
            eta_YB=args.eta_YB,
            omega_site=args.omega_site,
            steps=args.steps,
            dt=args.dt,
            dt_resp=args.dt_resp,
            active_dim=args.active_dim,
            seed=args.seed + 1,
        ),
        run_single_case(
            label="endpoint_heavier",
            eta_AX=1.15 * args.eta_AX,
            eta_XY=args.eta_XY,
            eta_YB=1.15 * args.eta_YB,
            omega_site=args.omega_site,
            steps=args.steps,
            dt=args.dt,
            dt_resp=args.dt_resp,
            active_dim=args.active_dim,
            seed=args.seed + 2,
        ),
    ]

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
            "best_capture_ratio": c["best"]["capture_ratio"],
            "best_damage_active_power": c["best"]["damage_active_power"],
            "best_leakage_gain": c["best"]["leakage_gain"],
            "final_W_relational": c["final"]["W_relational"],
            "final_capture_ratio": c["final"]["capture_ratio"],
            "final_damage_active_power": c["final"]["damage_active_power"],
            "final_leakage_gain": c["final"]["leakage_gain"],
        }
        rows.append(row)
        print(
            f"{row['label']}: "
            f"score={row['score']:.4f}  "
            f"best_W={row['best_W_relational']:.4f}  "
            f"best_capture={row['best_capture_ratio']:.4f}  "
            f"best_dmg_active={row['best_damage_active_power']:.4e}  "
            f"best_leak_gain={row['best_leakage_gain']:.4e}"
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
    print(f"  {os.path.join(args.outdir, 'active_capture_ratio.png')}")
    print(f"  {os.path.join(args.outdir, 'damage_active_power.png')}")
    print(f"  {os.path.join(args.outdir, 'leakage_gain.png')}")
    print(f"  {os.path.join(args.outdir, 'sv_spectrum.png')}")
    print()
    print(f"Runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()