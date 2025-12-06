# Resonance Engine: The Substrate Framework

**Deriving the Laws of Physics from Information-Theoretic Constraints.**

![Build Status](https://img.shields.io/badge/physics-emergent-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Python](https://img.shields.io/badge/python-3.8+-yellow.svg)

## 🌌 Overview

The **Resonance Engine** is a physics simulation suite that does not presuppose the existence of space, time, particles, or forces. Instead, it starts from a minimal set of information-theoretic axioms and demonstrates how classical reality emerges from the dynamics of a quantum graph.

This project serves as the computational proof-of-concept for the **Substrate Framework**, demonstrating that fundamental physical laws—from the Pauli Exclusion Principle to the Strong Nuclear Force—can be derived as emergent properties of a unitary, gauge-invariant network.

## 📐 The Axioms

The engine is built on three non-negotiable rules:

1.  **Hilbert Space Realism:** The fundamental object is the state vector $|\Psi\rangle$. Space is not a container; it is a network of relations between basis states.
2.  **Unitary Evolution:** Time evolution is strictly unitary ($U = e^{-iHt}$). There is no "wavefunction collapse" in the fundamental ontology.
3.  **Geometry is Memory:** Forces are not external laws. They are the energy costs associated with deforming the gauge field (history) on the graph links.

## 🚀 Installation

The engine requires Python 3.8+ and standard scientific computing libraries.

```bash
# Clone the repository
git clone https://github.com/resonance-laboratory/resonance-engine.git
cd resonance-engine

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** `numpy`, `scipy`, `matplotlib` (Optional: `cupy` for GPU acceleration).

## 🧪 Experiments & Derivations

The `experiments/` directory contains the rigorous proofs that standard physical laws emerge from the Substrate.

### 1. Emergent Space-Time
* **Script:** `experiments/01_emergent_space.py`
* **The Physics:** Simulates information propagation on a raw graph.
* **The Result:** Demonstrates the **Lieb-Robinson Bound**, proving that a finite "Speed of Light" ($c$) and a causal metric emerge naturally from local connectivity, independent of background geometry.

### 2. Derivation of Fermions
* **Script:** `experiments/02_derive_fermions.py`
* **The Physics:** Simulates the exchange of two excitations on a graph with Quaternionic ($SU(2)$) connections.
* **The Result:** Derives the **Pauli Exclusion Principle**. We observe an exact $-1$ phase shift upon particle exchange due to the non-Abelian topology of the rotation group, explaining the origin of matter statistics.

### 3. Derivation of Inertia ($F=ma$)
* **Script:** `experiments/03_derive_inertia.py`
* **The Physics:** Simulates a particle moving through a gauge field with varying stiffness ($g$).
* **The Result:** Shows that **Inertial Mass** is an effective parameter describing the coupling to the vacuum history. "Mass" is simply the resistance of the substrate to geometric deformation.

### 4. The Fundamental Forces
* **Script:** `experiments/04_derive_forces.py`
* **The Physics:** Derives interaction potentials from geometric constraints.
* **The Result:**
    * **Strong Force:** Emerges from Flux Tube conservation (Linear Confinement $V \propto r$).
    * **Weak Force:** Emerges from massive gauge links (Yukawa Screening $V \propto e^{-mr}/r$).
    * **EM Force:** Emerges from geometric spreading ($V \propto 1/r$).

### 5. The Topological Atom
* **Script:** `experiments/05_topological_atom.py`
* **The Physics:** Solves for eigenstates of a particle orbiting a topological defect (Monopole).
* **The Result:** Reproduces the **Periodic Table of Orbitals**. We visualize the emergence of $1s$ (spheres), $2p$ (dumbbells), and $3d$ (cloverleaves) orbitals from graph spherical harmonics.

### 6. Light-Matter Interaction (QED)
* **Script:** `experiments/06_qed_laser.py`
* **The Physics:** Couples the Topological Atom to a quantized photon field.
* **The Result:** Demonstrates **Vacuum Rabi Oscillations**. We observe the spontaneous absorption and re-emission of a photon, with the wavefunction morphing between $s$ and $p$ orbitals in real-time.

### 7. Nuclear Dynamics
* **Script:** `experiments/07_nuclear_fusion.py`
* **The Physics:** Simulates scattering and decay processes.
* **The Result:** Demonstrates **Quantum Tunneling** (Fusion) through a Coulomb barrier and **Weak Decay** (Flavor Changing) via topological transitions.

## 💻 Computational Universality

Can this framework simulate *everything*?

* **Script:** `demos/universality_test.py`
* **The Proof:** We implement a **Controlled-Z (CZ) Gate** using only gauge-strain interactions between two particles.
* **Significance:** Combined with single-qubit rotations (verified in Experiment 06), this proves the Substrate is a **Universal Quantum Computer** (Turing Complete). It is theoretically capable of simulating any local quantum field theory.

## 🛠 Usage Example

You can use the `UnifiedSubstrate` class to build your own universes.

```python
from engine.substrate import UnifiedSubstrate

# 1. Initialize a Universe (11x11x11 Grid)
uni = UnifiedSubstrate(L_size=11)

# 2. Inject a "Proton" (Topological Defect)
uni.inject_defect(strength=4.0)

# 3. Compile the Physics
uni.build_hamiltonian()

# 4. Discover Matter (Solve for Orbitals)
energies, wavefunctions = uni.solve_eigenstates(k=5)

# 5. Visualize the Ground State
uni.psi = wavefunctions[:, 0]
uni.plot_density(title="Emergent Ground State")
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Resonance Laboratory** | *Deriving Reality from First Principles*