"""
Decoherence Demo: Watching the Vortex Form

The key physics:
- System starts in superposition |+⟩ = (|0⟩ + |1⟩)/√2
- Environment starts in definite state |0⟩
- Interaction: conditioned on system being |1⟩, environment rotates
- Result: environment becomes correlated with system → entanglement
- From system's perspective: coherence is lost, it looks mixed

This is how pointer states form:
- States that DON'T trigger environment change stay coherent
- States that trigger DIFFERENT environment responses decohere
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(*args):
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result

def partial_trace(rho, dims, keep):
    """Partial trace using einsum"""
    n = len(dims)
    keep = sorted(keep)
    trace_out = [i for i in range(n) if i not in keep]
    
    if not trace_out:
        return rho.copy()
    
    shape = list(dims) + list(dims)
    rho_t = rho.reshape(shape)
    
    # Build einsum
    in_idx = list(range(2*n))
    for t in trace_out:
        in_idx[n + t] = t
    
    out_idx = keep + [n + k for k in keep]
    
    chars = 'abcdefghijklmnop'
    in_str = ''.join(chars[i] for i in in_idx)
    out_str = ''.join(chars[i] for i in out_idx)
    
    result = np.einsum(f"{in_str}->{out_str}", rho_t)
    d = int(np.prod([dims[k] for k in keep]))
    return result.reshape(d, d)

def purity(rho):
    return np.real(np.trace(rho @ rho))

def von_neumann_entropy(rho):
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-12]
    return -np.sum(eigs * np.log2(eigs))

#==============================================================================
# SETUP: 1 system qubit + multiple environment qubits
#==============================================================================

n_env = 5  # More environment qubits = more thorough decoherence
n_total = 1 + n_env

print("="*60)
print("WATCHING DECOHERENCE HAPPEN")
print("="*60)
print(f"\nSystem: 1 qubit")
print(f"Environment: {n_env} qubits")

# Hamiltonian: System-conditioned rotations of environment
# H = sum_i (|1><1|_sys ⊗ X_i) 
# When system is |1>, each environment qubit rotates

H = np.zeros((2**n_total, 2**n_total), dtype=complex)

# Projector onto |1⟩ for system
P1 = np.array([[0, 0], [0, 1]], dtype=complex)

coupling_strengths = [1.0, 0.8, 0.6, 0.4, 0.2]  # Different strengths

for i in range(n_env):
    ops = [P1] + [I]*n_env
    ops[1 + i] = X
    J = coupling_strengths[i] if i < len(coupling_strengths) else 0.5
    H += J * tensor(*ops)

# Also add some environment self-dynamics (makes it more realistic)
for i in range(n_env):
    ops = [I] + [I]*n_env
    ops[1 + i] = Z
    H += 0.1 * tensor(*ops)

print("\nHamiltonian: conditional X rotations + environment Z terms")

#==============================================================================
# INITIAL STATE: |+⟩ ⊗ |00...0⟩
#==============================================================================

plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
zero = np.array([1, 0], dtype=complex)

psi0 = plus
for _ in range(n_env):
    psi0 = np.kron(psi0, zero)

print(f"\nInitial state: |+⟩ ⊗ |00...0⟩")
print("  System: coherent superposition")
print("  Environment: definite ground state")

#==============================================================================
# EVOLUTION
#==============================================================================

times = np.linspace(0, 5, 100)
dt = times[1] - times[0]
U_dt = expm(-1j * H * dt)

# Storage
purities = []
entropies = []
coherences = []
env_entropies = []

psi = psi0.copy()
dims = [2] * n_total

print("\n" + "-"*60)
print("EVOLUTION")
print("-"*60)

for i, t in enumerate(times):
    rho = np.outer(psi, psi.conj())
    
    # System reduced density matrix
    rho_sys = partial_trace(rho, dims, [0])
    
    # Environment reduced density matrix
    rho_env = partial_trace(rho, dims, list(range(1, n_total)))
    
    purities.append(purity(rho_sys))
    entropies.append(von_neumann_entropy(rho_sys))
    coherences.append(np.abs(rho_sys[0, 1]))
    env_entropies.append(von_neumann_entropy(rho_env))
    
    if i % 20 == 0:
        print(f"t={t:.2f}: purity={purities[-1]:.4f}, S={entropies[-1]:.4f}, |coh|={coherences[-1]:.4f}")
    
    psi = U_dt @ psi

#==============================================================================
# RESULTS
#==============================================================================

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nInitial system purity: {purities[0]:.4f}")
print(f"Final system purity:   {purities[-1]:.4f}")
print(f"\nInitial system entropy: {entropies[0]:.4f} bits")
print(f"Final system entropy:   {entropies[-1]:.4f} bits")
print(f"  (Maximum possible: 1.0 bits)")
print(f"\nInitial coherence: {coherences[0]:.4f}")
print(f"Final coherence:   {coherences[-1]:.4f}")

# What are the pointer states?
print("\n" + "-"*60)
print("POINTER STATE ANALYSIS")
print("-"*60)

rho_final = np.outer(psi, psi.conj())
rho_sys_final = partial_trace(rho_final, dims, [0])

print("\nFinal system density matrix:")
print(f"  ρ_00 = {rho_sys_final[0,0]:.4f}")
print(f"  ρ_01 = {rho_sys_final[0,1]:.4f}")
print(f"  ρ_10 = {rho_sys_final[1,0]:.4f}")
print(f"  ρ_11 = {rho_sys_final[1,1]:.4f}")

eigs, vecs = np.linalg.eigh(rho_sys_final)
print(f"\nEigenvalues: {eigs}")
print("Pointer states are eigenstates of reduced density matrix:")
for i, (e, v) in enumerate(zip(eigs[::-1], vecs.T[::-1])):
    if e > 0.01:
        print(f"  λ={e:.4f}: |ψ⟩ = {v[0]:.3f}|0⟩ + {v[1]:.3f}|1⟩")

#==============================================================================
# PLOT
#==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.plot(times, purities, 'b-', linewidth=2)
ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='maximally mixed')
ax.set_xlabel('Time')
ax.set_ylabel('System Purity')
ax.set_title('Decoherence: Purity Decay')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(times, entropies, 'g-', linewidth=2)
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='maximum (1 bit)')
ax.set_xlabel('Time')
ax.set_ylabel('System Entropy (bits)')
ax.set_title('Decoherence: Entropy Growth')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(times, coherences, 'm-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('|ρ_01| (off-diagonal)')
ax.set_title('Decoherence: Coherence Decay')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(times, env_entropies, 'c-', linewidth=2, label='Environment')
ax.plot(times, entropies, 'g--', linewidth=2, label='System')
ax.set_xlabel('Time')
ax.set_ylabel('Entropy (bits)')
ax.set_title('Entanglement: Entropy Growth')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Decoherence: The Vortex of Locality', fontsize=14)
plt.tight_layout()
plt.savefig('decoherence_vortex.png', dpi=150)
print("\nSaved: decoherence_vortex.png")

#==============================================================================
# THE PICTURE
#==============================================================================

print("\n" + "="*60)
print("WHAT WE'RE SEEING")
print("="*60)
print("""
The system starts in superposition: |0⟩ + |1⟩

The interaction is CONDITIONAL:
  - If system is |0⟩: environment does nothing
  - If system is |1⟩: environment rotates

After evolution:
  |0⟩|env_A⟩ + |1⟩|env_B⟩

where env_A ≠ env_B.

From the SYSTEM's perspective:
  - It can't see the environment
  - It only sees its reduced density matrix
  - The coherence (|0⟩⟨1| term) is suppressed
  - It looks MIXED even though the global state is pure

This is decoherence. The "vortex" that pulls superposition
into definiteness. Not by collapse - by entanglement.

The POINTER STATES are |0⟩ and |1⟩ because:
  - These are the states that trigger DIFFERENT environment responses
  - Superpositions of them become entangled with environment
  - Only they remain as "valid" classical outcomes
""")

plt.show()