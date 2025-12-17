"""
pointer_basis_demo.py

Numerical demonstration of pointer-basis formation in a finite Hilbert substrate.

We model:
    - N_SYS "system" qubits
    - N_ENV "environment" qubits
with total Hilbert space H_total = (C^2)^(⊗(N_SYS + N_ENV)).

Initial state:
    |Ψ(0)> = |+>^(⊗ N_SYS) ⊗ |0>^(⊗ N_ENV)

Dynamics (per timestep):
    1. Local nearest-neighbor unitaries among system qubits (XX + YY interaction).
    2. Local Ising-type interactions between each system qubit and its corresponding
       environment qubit (Z ⊗ Z).

We track the reduced density matrices ρ_i(t) for each system qubit i and record the
off-diagonal coherence |ρ_01| as a function of time.

Running this file directly will:
    - run the simulation with the parameters in the CONFIG block
    - save an .npz file with:
        t : array of shape (T+1,)             time points
        coherences : array of shape (N_SYS, T+1)
        cfg : dict with basic simulation parameters
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from hilbert_substrate_core import Config, Substrate


# ---------------------------------------------------------------------------
# Simple config block (no CLI)
# ---------------------------------------------------------------------------

# System / environment sizes
N_SYS: int = 4
N_ENV: int = 4

# Number of discrete steps
STEPS: int = 50

# Time steps for the two interaction types
DT_SYS: float = 0.25   # system-system
DT_ENV: float = 0.30   # system-environment

# RNG seed and backend
SEED: int = 1
USE_GPU: bool = False

# Output file name (relative to current working directory)
OUT_FILE: str = "pointer_basis_results.npz"


# ---------------------------------------------------------------------------
# Helper: single-qubit unitary application for state preparation
# ---------------------------------------------------------------------------

def apply_single_qubit_unitary(sub: Substrate, site: int, U: np.ndarray) -> None:
    """
    Apply a single-qubit unitary U to factor `site` of the substrate's state |Ψ>.

    This is used for state preparation (e.g., applying Hadamards to prepare |+> states)
    before we register the dynamical local terms.

    Parameters
    ----------
    sub : Substrate
        The Hilbert substrate instance.
    site : int
        Index of the factor to which U is applied.
    U : np.ndarray
        2x2 unitary matrix, assumed to act on local_dim == 2.
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("apply_single_qubit_unitary assumes local_dim == 2.")

    xp = sub.xp
    n = sub.cfg.n_factors
    d = sub.cfg.local_dim

    if not (0 <= site < n):
        raise ValueError(f"Site index {site} out of range [0, {n-1}] for n_factors={n}.")

    U_backend = xp.asarray(np.asarray(U, dtype=np.complex128))

    # Reshape |Ψ> to rank-n tensor
    psi = sub.state.reshape((d,) * n)

    # Permutation: all non-target axes first, then the target axis
    non_sites = [i for i in range(n) if i != site]
    perm = non_sites + [site]
    psi_perm = xp.transpose(psi, axes=perm)

    # Collapse to (rest_dim, d)
    rest_dim = d ** (n - 1)
    psi_flat = psi_perm.reshape(rest_dim, d)

    # Apply U on the last index: psi'_r,a = Σ_b psi_r,b U_{a,b}
    psi_flat = psi_flat @ U_backend.T

    # Reshape back and invert permutation
    psi_perm = psi_flat.reshape((d,) * n)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    psi_new = xp.transpose(psi_perm, axes=inv_perm)

    sub.state = psi_new.reshape(-1)


# ---------------------------------------------------------------------------
# Building the dynamics
# ---------------------------------------------------------------------------

def build_dynamics(
    sub: Substrate,
    n_sys: int,
    dt_sys: float,
    dt_env: float,
    J_sys: float = 1.0,
    g_env: float = 1.0,
) -> None:
    """
    Register local-unitary dynamics on the substrate:

    - System-system nearest-neighbor interactions (XX + YY).
    - System-environment Ising-type interactions (Z ⊗ Z).

    Parameters
    ----------
    sub : Substrate
        Substrate instance whose local_terms list will be populated.
    n_sys : int
        Number of system qubits; these occupy factor indices [0, n_sys-1].
        Environment qubits occupy [n_sys, n_sys + n_env - 1].
    dt_sys : float
        Timestep used in exp(-i H_sys dt_sys) for system-system interactions.
    dt_env : float
        Timestep used in exp(-i H_env dt_env) for system-env interactions.
    J_sys : float
        Coupling strength for system-system XX+YY term.
    g_env : float
        Coupling strength for system-env Z⊗Z term.
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("build_dynamics is currently implemented for qubits (local_dim == 2).")

    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # System-system local Hamiltonian: H_sys = J_sys (X⊗X + Y⊗Y)
    H_sys = J_sys * (np.kron(X, X) + np.kron(Y, Y))

    # System-env local Hamiltonian: H_env = g_env (Z⊗Z)
    H_env = g_env * np.kron(Z, Z)

    # Convert to unitaries via small-time evolution
    U_sys = Substrate.hermitian_to_unitary(H_sys, dt_sys, use_gpu=sub.on_gpu)
    U_env = Substrate.hermitian_to_unitary(H_env, dt_env, use_gpu=sub.on_gpu)

    # Register nearest-neighbor unitaries on system subspace
    for i in range(n_sys - 1):
        sub.add_local_unitary(sites=[i, i + 1], unitary=U_sys)

    # Register system-env unitaries: pair system i with environment j = n_sys + i
    n_total = sub.cfg.n_factors
    n_env = n_total - n_sys
    if n_env < n_sys:
        raise ValueError(
            f"Not enough environment qubits (n_env={n_env}) to pair with n_sys={n_sys}."
        )

    for i in range(n_sys):
        env_site = n_sys + i
        sub.add_local_unitary(sites=[i, env_site], unitary=U_env)


# ---------------------------------------------------------------------------
# Main simulation function (importable)
# ---------------------------------------------------------------------------

def run_simulation(
    n_sys: int,
    n_env: int,
    steps: int,
    dt_sys: float,
    dt_env: float,
    seed: int,
    use_gpu: bool,
) -> Dict[str, Any]:
    """
    Run the pointer-basis decoherence simulation.

    Parameters
    ----------
    n_sys : int
        Number of system qubits.
    n_env : int
        Number of environment qubits.
    steps : int
        Number of discrete dynamical steps to evolve.
    dt_sys : float
        Time step for system-system unitaries.
    dt_env : float
        Time step for system-env unitaries.
    seed : int
        RNG seed used for reproducibility of the initial state.
    use_gpu : bool
        Whether to use GPU backend (CuPy) if available.

    Returns
    -------
    results : Dict[str, Any]
        Dictionary with keys:
            "t"           : array of times (T+1,)
            "coherences"  : array (n_sys, T+1) of |ρ_01| per system qubit
            "cfg"         : dict of run configuration
    """
    # Total factors = system + environment
    n_factors = n_sys + n_env

    # Start in |0>^(N_sys + N_env) and then apply Hadamards to system qubits
    product_state = [0] * n_factors

    cfg = Config(
        n_factors=n_factors,
        local_dim=2,
        use_gpu=use_gpu,
        seed=seed,
        product_state=product_state,
    )
    sub = Substrate(cfg)

    # Prepare |+>^⊗N_sys on the system qubits via Hadamard
    H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    for i in range(n_sys):
        apply_single_qubit_unitary(sub, site=i, U=H)

    # Sanity: check initial coherence ~ 0.5 for each system qubit
    init_coherences = [sub.single_factor_coherence(i) for i in range(n_sys)]
    print("Initial single-qubit coherences (system qubits):", init_coherences)

    # Register the dynamical local unitaries
    build_dynamics(sub, n_sys=n_sys, dt_sys=dt_sys, dt_env=dt_env)

    # Allocate arrays to store coherence vs time
    t_vals = np.arange(steps + 1, dtype=float)
    coherences = np.zeros((n_sys, steps + 1), dtype=float)

    # Record initial coherences
    for i in range(n_sys):
        coherences[i, 0] = sub.single_factor_coherence(i)

    # Time evolution
    for step in range(1, steps + 1):
        sub.step(n_steps=1)
        for i in range(n_sys):
            coherences[i, step] = sub.single_factor_coherence(i)

        if step % max(1, steps // 10) == 0:
            print(f"[step {step}/{steps}] coherences (sys 0..{n_sys-1}):",
                  [f"{coherences[i, step]:.4f}" for i in range(n_sys)])

    results = {
        "t": t_vals,
        "coherences": coherences,
        "cfg": {
            "n_sys": n_sys,
            "n_env": n_env,
            "steps": steps,
            "dt_sys": dt_sys,
            "dt_env": dt_env,
            "seed": seed,
            "use_gpu": use_gpu,
        },
    }
    return results


# ---------------------------------------------------------------------------
# Run when executed as a script (no CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Pointer-Basis Decoherence Demo (no CLI) ===")
    print(f"n_sys={N_SYS}, n_env={N_ENV}, steps={STEPS}")
    print(f"dt_sys={DT_SYS}, dt_env={DT_ENV}, seed={SEED}, use_gpu={USE_GPU}")
    print(f"Output file: {OUT_FILE}")

    res = run_simulation(
        n_sys=N_SYS,
        n_env=N_ENV,
        steps=STEPS,
        dt_sys=DT_SYS,
        dt_env=DT_ENV,
        seed=SEED,
        use_gpu=USE_GPU,
    )

    np.savez_compressed(OUT_FILE, **res)
    print(f"Saved results to {OUT_FILE}")
    print("Done.")
