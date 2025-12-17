"""
lieb_robinson_demo.py

Demonstration of an emergent Lieb–Robinson "light cone" in a Hilbert substrate.

We consider:
    - N qubits (local_dim = 2).
    - A local Hamiltonian that is a sum of nearest-neighbor terms:
          H = sum_j J (X_j X_{j+1} + Y_j Y_{j+1})
    - Time evolution via a discrete Trotter step:
          U_step = prod_j exp(-i h_{j,j+1} dt)

Initial state:
    |Ψ(0)> = |1, 0, 0, ..., 0>

We track the local expectation values:
    m_j(t) = <Ψ(t)| Z_j |Ψ(t)>

and store them as a function of "site index" j and time step t.

Outputs:
    - An .npz file containing:
        t       : array of shape (T+1,) of integer time steps: t = 0,1,...,T
        mz      : array of shape (N, T+1), where mz[j, t] = <Z_j>(t)
        cfg     : dict with basic simulation parameters

This script has no CLI; parameters are defined in the CONFIG block below.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from hilbert_substrate_core import Config, Substrate


# ---------------------------------------------------------------------------
# CONFIG (edit these directly)
# ---------------------------------------------------------------------------

N_FACTORS: int = 16      # number of qubits
STEPS: int = 100         # number of Trotter steps
DT: float = 0.15         # time step for exp(-i H_local dt)
J_COUPLING: float = 1.0  # strength of XX+YY coupling

SEED: int = 0            # RNG seed (mostly irrelevant here)
USE_GPU: bool = False    # set True if you want GPU (and CuPy is installed)

OUT_FILE: str = "lieb_robinson_results.npz"


# ---------------------------------------------------------------------------
# Build local dynamics: nearest-neighbor XX + YY
# ---------------------------------------------------------------------------

def build_xx_yy_chain(sub: Substrate, dt: float, J: float = 1.0) -> None:
    """
    Register nearest-neighbor XX+YY local unitaries on the substrate.

    Hamiltonian on each pair (j, j+1):
        h_{j,j+1} = J (X ⊗ X + Y ⊗ Y)

    A single Trotter step is:
        U_step = prod_j exp(-i h_{j,j+1} dt)
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("build_xx_yy_chain is implemented for qubits (local_dim == 2).")

    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    # Local Hamiltonian H_pair = J (X⊗X + Y⊗Y)
    H_pair = J * (np.kron(X, X) + np.kron(Y, Y))

    # Convert to a unitary for a time step dt
    U_pair = Substrate.hermitian_to_unitary(H_pair, dt, use_gpu=sub.on_gpu)

    # Register the same local unitary on each neighbor pair (j, j+1)
    for j in range(sub.cfg.n_factors - 1):
        sub.add_local_unitary(sites=[j, j + 1], unitary=U_pair)


# ---------------------------------------------------------------------------
# Observable: local Z expectation values
# ---------------------------------------------------------------------------

def z_expectations(sub: Substrate) -> np.ndarray:
    """
    Compute <Z_j> for each site j of the substrate (assuming qubits).

    Returns
    -------
    mz : np.ndarray
        Array of shape (n_factors,) with mz[j] = <Z_j>.
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("z_expectations is defined only for local_dim == 2.")

    n = sub.cfg.n_factors
    d = sub.cfg.local_dim
    xp = sub.xp

    # Pauli Z
    Z = xp.asarray(np.array([[1, 0], [0, -1]], dtype=np.complex128))

    # Reshape |Ψ> into rank-n tensor with dimension (2,2,...,2)
    psi = sub.state.reshape((d,) * n)

    # We'll compute <Z_j> by explicitly contracting the tensor network form:
    # <Z_j> = sum_{σ,σ'} ψ*_{σ} ψ_{σ'} <σ|Z_j|σ'>, but since Z is diagonal,
    # this reduces to sum_{σ} |ψ_{σ}|^2 z(σ_j).
    # For efficiency, we do something a bit more direct.
    psi_np = sub.to_numpy(psi)
    mz = np.zeros(n, dtype=float)

    # Iterate over sites and compute expectation by summing over all configs
    # This is not optimal for large n, but fine for moderate chain sizes.
    for j in range(n):
        # Move site j to the last axis
        axes = list(range(n))
        axes[j], axes[-1] = axes[-1], axes[j]
        psi_perm = np.transpose(psi_np, axes=axes)
        psi_flat = psi_perm.reshape(-1, d)  # shape: (2^{n-1}, 2)

        # Probability weights for |0>, |1> at site j
        probs = np.sum(np.abs(psi_flat) ** 2, axis=0)  # shape (2,)
        # <Z> = P(0)*1 + P(1)*(-1) = P(0) - P(1)
        mz[j] = probs[0] - probs[1]

    return mz


# ---------------------------------------------------------------------------
# Main simulation function (importable)
# ---------------------------------------------------------------------------

def run_lieb_robinson_demo(
    n_factors: int,
    steps: int,
    dt: float,
    J: float,
    seed: int,
    use_gpu: bool,
) -> Dict[str, Any]:
    """
    Run the Lieb–Robinson light-cone demonstration.

    Parameters
    ----------
    n_factors : int
        Number of qubits in the chain.
    steps : int
        Number of discrete Trotter steps to evolve.
    dt : float
        Time step used for each local exp(-i H_pair dt).
    J : float
        Coupling strength for XX+YY nearest-neighbor interactions.
    seed : int
        RNG seed (not very important here, but included for consistency).
    use_gpu : bool
        Whether to use GPU backend (CuPy) if available.

    Returns
    -------
    results : Dict[str, Any]
        Dictionary with keys:
            "t"   : array of shape (steps+1,) of integer time steps
            "mz"  : array (n_factors, steps+1), <Z_j>(t) for each site j and time t
            "cfg" : dict of run configuration
    """
    # Prepare a product state with a single excitation at site 0:
    # |1,0,0,...,0>
    product_state = [0] * n_factors
    product_state[0] = 1

    cfg = Config(
        n_factors=n_factors,
        local_dim=2,
        use_gpu=use_gpu,
        seed=seed,
        product_state=product_state,
    )
    sub = Substrate(cfg)

    print("Initial norm:", sub.norm())

    # Register local XX+YY unitaries
    build_xx_yy_chain(sub, dt=dt, J=J)

    # Allocate array for <Z_j>(t)
    t_vals = np.arange(steps + 1, dtype=float)
    mz = np.zeros((n_factors, steps + 1), dtype=float)

    # Record t=0
    mz[:, 0] = z_expectations(sub)

    # Time evolution
    for step in range(1, steps + 1):
        sub.step(n_steps=1)
        mz[:, step] = z_expectations(sub)

        if step % max(1, steps // 10) == 0:
            # print a quick snapshot of <Z_j> near the "front"
            print(f"[step {step}/{steps}] "
                  f"mz center ~ {mz[n_factors//2, step]:.3f} | "
                  f"mz edge ~ {mz[-1, step]:.3f}")

    results = {
        "t": t_vals,
        "mz": mz,
        "cfg": {
            "n_factors": n_factors,
            "steps": steps,
            "dt": dt,
            "J": J,
            "seed": seed,
            "use_gpu": use_gpu,
        },
    }
    return results


# ---------------------------------------------------------------------------
# Run on import as a script (no CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Lieb–Robinson Light-Cone Demo (no CLI) ===")
    print(f"n_factors={N_FACTORS}, steps={STEPS}, dt={DT}, J={J_COUPLING}")
    print(f"seed={SEED}, use_gpu={USE_GPU}")
    print(f"Output file: {OUT_FILE}")

    res = run_lieb_robinson_demo(
        n_factors=N_FACTORS,
        steps=STEPS,
        dt=DT,
        J=J_COUPLING,
        seed=SEED,
        use_gpu=USE_GPU,
    )

    np.savez_compressed(OUT_FILE, **res)
    print(f"Saved results to {OUT_FILE}")
    print("Done.")
