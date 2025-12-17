"""
substrate_decoherence_sim.py

Hilbert Substrate Simulation: No Signaling + No Forgetting
=========================================================

This script implements a minimal toy of your core framework:

  • The Substrate IS Hilbert space: H_total = H_sys ⊗ H_env.
  • Global state |Ψ(t)> evolves unitarily at all times (no forgetting).
  • Dynamics is built from local (few-qubit) unitaries (no signaling).
  • Decoherence and pointer states emerge because system qubits
    become entangled with environment qubits (environment-as-records).

What it does:

  - N_sys system qubits (the "classical world" candidates).
  - N_env environment qubits (the record / history bath).
  - Global pure state |Ψ> ∈ C^(2^(N_sys + N_env)).
  - Each timestep:
      (1) Apply local 2-qubit unitaries on system qubits (random "physics").
      (2) Apply unitary system–environment couplings that entangle
          system computational basis states with environment (defrag).
  - After each step, compute:
      * reduced density matrix ρ_sys (trace over env),
      * single-qubit ρ_i and the magnitude of off-diagonal elements
        in the computational (Z) basis (candidate pointer basis).

The key constraints:

  - No signaling: only local 2-qubit gates between neighboring system qubits.
  - No forgetting: all evolution is unitary on H_sys ⊗ H_env; we never
    trace out in the dynamical rules, only in diagnostics.

Run with:
    python experiments/substrate_decoherence_sim.py
"""

import math
import cmath
from typing import Dict, List

try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # drop-in for this script
    xp = cp
    GPU_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

CONFIG: Dict = {
    # Number of system qubits (the ones we watch decohere)
    "N_sys": 4,

    # Number of environment qubits (the "record" bath)
    "N_env": 4,

    # Number of timesteps
    "n_steps": 50,

    # Coupling strengths
    "sys_coupling_strength": 0.3,   # random 2-qubit system-system interaction
    "env_coupling_strength": 1.0,   # system-env decoherence coupling

    # Random seed (for reproducibility)
    "seed": 1234,
}


# ============================================================
# BASIC LINEAR ALGEBRA HELPERS
# ============================================================

def complex_dtype():
    return xp.complex128


def zeros(shape):
    return xp.zeros(shape, dtype=complex_dtype())


def normalize(psi: xp.ndarray) -> xp.ndarray:
    """Normalize a state vector |psi>."""
    norm = xp.linalg.norm(psi)
    if norm == 0:
        return psi
    return psi / norm


def kron(*ops):
    """Kronecker product of a sequence of matrices."""
    out = ops[0]
    for op in ops[1:]:
        out = xp.kron(out, op)
    return out


# Pauli matrices
SIGMA_X = xp.array([[0, 1],
                    [1, 0]], dtype=complex_dtype())
SIGMA_Y = xp.array([[0, -1j],
                    [1j,  0]], dtype=complex_dtype())
SIGMA_Z = xp.array([[1,  0],
                    [0, -1]], dtype=complex_dtype())
IDENT_2 = xp.eye(2, dtype=complex_dtype())


# ============================================================
# HILBERT SUBSTRATE MODEL
# ============================================================

class HilbertSubstrate:
    """
    Global Hilbert substrate: H_total = H_sys ⊗ H_env, with:

      - N_sys system qubits
      - N_env environment qubits

    State is |Ψ> ∈ C^(2^(N_sys + N_env)), always normalized.

    Dynamics per timestep:

      1. Local system-system unitary dynamics (no signaling beyond
         nearest neighbors in an abstract 1D chain).

      2. System-env decoherence unitaries, one env qubit per system qubit
         (records / history), no forgetting (unitary on full H_total).

    Diagnostics:

      - Reduced density matrix ρ_sys (trace over env).
      - Single-qubit reduced density matrices ρ_i.
      - Off-diagonal magnitude in Z basis: measure of coherence.

    The computational (Z) basis is treated as the pointer-basis candidate.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.N_sys = cfg["N_sys"]
        self.N_env = cfg["N_env"]
        self.N_total = self.N_sys + self.N_env

        self.D_sys = 2 ** self.N_sys
        self.D_env = 2 ** self.N_env
        self.D_total = 2 ** self.N_total

        # Seed RNG (CuPy or NumPy)
        if GPU_AVAILABLE:
            cp.random.seed(cfg["seed"])
        else:
            xp.random.seed(cfg["seed"])

        # Initialize |Ψ(0)> = |+>^{⊗ N_sys} ⊗ |0>^{⊗ N_env}
        psi_sys = self._plus_state(self.N_sys)   # (2^N_sys,)
        psi_env = self._zero_state(self.N_env)   # (2^N_env,)
        self.psi = xp.kron(psi_sys, psi_env)     # (2^(N_sys+N_env),)
        self.psi = normalize(self.psi)

    # --------------------------------------------------------
    # Initial states
    # --------------------------------------------------------
    def _zero_state(self, N: int) -> xp.ndarray:
        """|0...0> for N qubits."""
        psi = zeros(2 ** N)
        psi[0] = 1.0
        return psi

    def _plus_state(self, N: int) -> xp.ndarray:
        """|+>^{⊗ N} = (|0>+|1>)^{⊗ N} / sqrt(2^N)."""
        v = xp.array([1.0, 1.0], dtype=complex_dtype()) / xp.sqrt(2.0)
        psi = v
        for _ in range(N - 1):
            psi = xp.kron(psi, v)
        return psi

    # --------------------------------------------------------
    # Operator construction helpers
    # --------------------------------------------------------
    def _single_qubit_op_on_total(self, op: xp.ndarray, q: int) -> xp.ndarray:
        """
        Embed a single-qubit operator 'op' on qubit index q
        (0-based, ordering: system qubits first, then env qubits)
        into the full N_total-qubit space.
        """
        ops = []
        for i in range(self.N_total):
            if i == q:
                ops.append(op)
            else:
                ops.append(IDENT_2)
        return kron(*ops)

    def _two_qubit_op_on_total(self, op_4x4: xp.ndarray, q1: int, q2: int) -> xp.ndarray:
        """
        Embed a 4x4 two-qubit operator on qubits q1, q2 into the full space.

        We construct by inserting op_4x4 in the right spots via a simple
        kron-based construction. For N_total ≤ 8 this is fine.
        """
        if q1 == q2:
            raise ValueError("q1 and q2 must be distinct")
        if q1 > q2:
            q1, q2 = q2, q1

        ops = []
        for i in range(self.N_total):
            if i == q1:
                ops.append(None)  # placeholder for the 2-qubit block
            elif i == q2:
                ops.append(None)  # second placeholder
            else:
                ops.append(IDENT_2)

        # Merge placeholders into op_4x4
        result = None
        used_block = False
        for block in ops:
            if block is None and not used_block:
                block = op_4x4
                used_block = True
            elif block is None and used_block:
                # skip second placeholder
                continue
            if result is None:
                result = block
            else:
                result = xp.kron(result, block)

        return result

    # --------------------------------------------------------
    # Dynamics: system-system unitaries
    # --------------------------------------------------------
    def _random_two_qubit_unitary(self, strength: float) -> xp.ndarray:
        """
        Build a simple entangling 2-qubit unitary:

            U = exp(-i * strength * H),

        where H = (Z⊗Z + X⊗X) / 2, as a toy Hamiltonian.
        """
        H = 0.5 * (xp.kron(SIGMA_Z, SIGMA_Z) + xp.kron(SIGMA_X, SIGMA_X))
        eigvals, eigvecs = xp.linalg.eigh(H)
        phase = xp.exp(-1j * strength * eigvals)
        U = eigvecs @ (phase[:, None] * eigvecs.conj().T)
        return U

    def apply_system_dynamics(self):
        """
        Apply local two-qubit unitaries on neighboring system qubits
        in a 1D chain (no-signaling beyond nearest neighbors).
        """
        N_sys = self.N_sys
        strength = self.cfg["sys_coupling_strength"]

        # Neighbor pairs: (0,1), (1,2), ..., (N_sys-2, N_sys-1)
        for i in range(N_sys - 1):
            q1 = i
            q2 = i + 1
            U_2 = self._random_two_qubit_unitary(strength)
            U_total = self._two_qubit_op_on_total(U_2, q1, q2)
            self.psi = U_total @ self.psi
            self.psi = normalize(self.psi)

    # --------------------------------------------------------
    # Dynamics: system-environment decoherence (no forgetting)
    # --------------------------------------------------------
    def apply_env_coupling(self):
        """
        Entangle each system qubit with its corresponding environment qubit
        via a controlled-phase gate:

            U = |00><00| + |01><01| + |10><10| + e^{-iπ*strength} |11><11|.

        This is purely unitary on H_sys ⊗ H_env; no information is destroyed.
        It effectively performs "which-basis-state" marking in the Z basis.
        """
        N_sys = self.N_sys
        strength = self.cfg["env_coupling_strength"]

        # Compute the phase as a plain Python complex (to avoid CuPy/NumPy
        # scalar mixing issues inside xp.array).
        phase_11 = cmath.exp(-1j * math.pi * strength)

        CZ = xp.diag(
            xp.array([1.0, 1.0, 1.0, phase_11], dtype=complex_dtype())
        )

        for i in range(N_sys):
            # Pair system qubit i with environment qubit i (offset by N_sys)
            q_sys = i
            q_env = N_sys + i
            U_total = self._two_qubit_op_on_total(CZ, q_sys, q_env)
            self.psi = U_total @ self.psi
            self.psi = normalize(self.psi)

    # --------------------------------------------------------
    # Reduced density matrices and pointer diagnostics
    # --------------------------------------------------------
    def reduced_rho_sys(self) -> xp.ndarray:
        """
        ρ_sys = Tr_env |Ψ><Ψ|, shape (D_sys, D_sys).
        Implemented by reshaping |Ψ> into (D_sys, D_env) and contracting.
        """
        psi = self.psi.reshape(self.D_sys, self.D_env)
        rho_sys = psi @ psi.conj().T
        return rho_sys

    def single_qubit_rho(self, rho_sys: xp.ndarray, qubit: int) -> xp.ndarray:
        """
        Reduced density matrix for a single system qubit.

        rho_sys has shape (2^N_sys, 2^N_sys). We treat the system qubits
        as ordered [0,1,...,N_sys-1] in binary, and partial trace out all
        but 'qubit'.
        """
        N = self.N_sys

        # Reshape rho_sys to (2, 2, ..., 2; 2, 2, ..., 2) with N factors each side
        dims = [2] * N
        rho_reshaped = rho_sys.reshape(dims + dims)

        # Bring qubit 'qubit' to the front via permutation of axes.
        axes_s = [qubit] + [i for i in range(N) if i != qubit]
        axes_t = [qubit + N] + [i + N for i in range(N) if i != qubit]
        perm = axes_s + axes_t
        rho_perm = xp.transpose(rho_reshaped, axes=perm)

        # Now shape: (2, 2^(N-1), 2, 2^(N-1))
        rest_dim = 2 ** (N - 1)
        rho_perm = rho_perm.reshape(2, rest_dim, 2, rest_dim)

        # Partial trace over the "rest_dim" index:
        #   ρ1[i,j] = sum_k rho_perm[i,k,j,k]
        rho1 = xp.einsum("ikjk->ij", rho_perm)
        return rho1

    def pointer_coherence(self, rho1: xp.ndarray) -> float:
        """
        Measure off-diagonal magnitude in the computational basis
        for a single-qubit density matrix ρ1.

        Pointer basis candidate = {|0>, |1>} in Z.
        Coherence = |ρ01|.
        """
        return float(abs(rho1[0, 1]))

    # --------------------------------------------------------
    # One full timestep: system dynamics + decoherence
    # --------------------------------------------------------
    def step(self):
        self.apply_system_dynamics()
        self.apply_env_coupling()
        self.psi = normalize(self.psi)  # numerical safety


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation(cfg: Dict):
    print("============================================================")
    print("Hilbert Substrate Simulation: No Signaling + No Forgetting")
    print("============================================================")
    print(f"N_sys = {cfg['N_sys']} system qubits")
    print(f"N_env = {cfg['N_env']} environment qubits")
    print(f"GPU available: {GPU_AVAILABLE}")
    print("------------------------------------------------------------")

    hs = HilbertSubstrate(cfg)

    print("Initial state: |Ψ(0)> = |+>^{⊗ N_sys} ⊗ |0>^{⊗ N_env}")
    rho_sys = hs.reduced_rho_sys()

    # Initial diagnostics
    coherences0: List[float] = []
    for q in range(cfg["N_sys"]):
        rho1 = hs.single_qubit_rho(rho_sys, q)
        coh = hs.pointer_coherence(rho1)
        coherences0.append(coh)

    print("Initial single-qubit coherences in Z basis (|ρ01|):")
    for q, coh in enumerate(coherences0):
        print(f"  qubit {q}: {coh:.6f}")
    print("------------------------------------------------------------")

    # Time evolution
    n_steps = cfg["n_steps"]
    for step in range(1, n_steps + 1):
        hs.step()

        if step % 5 == 0 or step == n_steps:
            rho_sys = hs.reduced_rho_sys()
            coherences: List[float] = []
            for q in range(cfg["N_sys"]):
                rho1 = hs.single_qubit_rho(rho_sys, q)
                coh = hs.pointer_coherence(rho1)
                coherences.append(coh)

            avg_coh = sum(coherences) / len(coherences)
            print(f"Step {step:3d}: avg |ρ01| = {avg_coh:.6f}")
            for q, coh in enumerate(coherences):
                print(f"    qubit {q}: |ρ01| = {coh:.6f}")
            print("------------------------------------------------------------")

    print("Final diagnostics:")
    rho_sys = hs.reduced_rho_sys()
    for q in range(cfg["N_sys"]):
        rho1 = hs.single_qubit_rho(rho_sys, q)
        coh = hs.pointer_coherence(rho1)
        print(f"  qubit {q}: final |ρ01| = {coh:.6f}")
        print(f"           ρ = [[{rho1[0,0]:+.3f}, {rho1[0,1]:+.3f}],")
        print(f"                [{rho1[1,0]:+.3f}, {rho1[1,1]:+.3f}]]")
    print("============================================================")
    print("Interpretation:")
    print("  • Global |Ψ(t)> evolves unitarily on H_sys ⊗ H_env (no forgetting).")
    print("  • System dynamics is local (2-qubit nearest-neighbor on system).")
    print("  • System–env coupling marks Z-basis states in the environment,")
    print("    suppressing off-diagonals in ρ_sys and ρ_i (decoherence).")
    print("  • The computational basis becomes an approximate pointer basis.")
    print("============================================================")


if __name__ == "__main__":
    run_simulation(CONFIG)
