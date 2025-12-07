# Resonance Engine: The Substrate Framework

**A Computational Framework Investigating Emergent Physics from Information-Theoretic Constraints.**

![Build Status](https://img.shields.io/badge/simulation-active-green.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8+-yellow.svg)

## 🌌 Overview

The **Resonance Engine** is a research codebase designed to simulate physics not by discretizing differential equations, but by modeling the dynamics of a unitary, graph-based substrate. The project explores the hypothesis that fundamental physical laws (spacetime, matter, and forces) can emerge as effective field theories from a minimal set of information-theoretic constraints.

Rather than presupposing particles or background geometry, this framework starts with a raw Hilbert space and tests whether standard physical phenomena appear as emergent behaviors of the system's unitary evolution.

## 📐 Theoretical Postulates

The simulation is strictly constrained by three core assumptions:

1.  **Hilbert Space Realism:** The system is defined by a state vector $|\Psi\rangle$ on a graph. Space is modeled as a relational network rather than a pre-existing manifold.
2.  **Unitary Evolution:** Dynamics are governed solely by unitary rotation ($U = e^{-iHt}$), preserving information.
3.  **Geometric Interactions:** "Forces" are modeled as energy costs associated with the deformation of local gauge fields (Berry phases) on the graph links.

## 🚀 Installation

The engine requires Python 3.8+ and standard scientific computing libraries.

    # Clone the repository
    git clone https://github.com/resonance-laboratory/resonance-engine.git
    cd resonance-engine

    # Install dependencies
    pip install -r requirements.txt

**Requirements:** `numpy`, `scipy`, `matplotlib` (Optional: `cupy` for GPU acceleration).

## 🧪 Experiments & Validations

The `experiments/` directory contains scripts that test the framework's ability to reproduce known physical phenomena.

### 1. Causal Structure (Emergent Spacetime)
* **Script:** `experiments/01_emergent_space.py`
* **Objective:** Test if a causal metric emerges from local graph connectivity.
* **Observation:** The simulation exhibits the **Lieb-Robinson Bound**, demonstrating that information propagation is limited by a finite maximum velocity ($c$), consistent with locality in relativistic physics.

### 2. Fermionic Statistics
* **Script:** `experiments/02_derive_fermions.py`
* **Objective:** Investigate the exchange statistics of excitations on a graph with $SU(2)$ gauge connections.
* **Observation:** Exchanging two excitations yields an exact $-1$ phase shift. This suggests that the **Pauli Exclusion Principle** can be modeled as a topological consequence of non-Abelian geometry.

### 3. Inertial Dynamics ($F=ma$)
* **Script:** `experiments/03_derive_inertia.py`
* **Objective:** Simulate the response of a localized excitation to an external potential under varying gauge field stiffness ($g$).
* **Observation:** The acceleration is inversely proportional to $g$, consistent with **Newton’s Second Law**. This supports the model's interpretation of inertial mass as a coupling constant to the vacuum history.

### 4. Fundamental Interaction Potentials
* **Script:** `experiments/04_derive_forces.py`
* **Objective:** Derive effective potentials $V(r)$ based on geometric constraints.
* **Observation:**
    * **Strong-like:** Flux conservation yields linear confinement ($V \propto r$).
    * **Weak-like:** Massive gauge links yield Yukawa screening ($V \propto e^{-mr}/r$).
    * **EM-like:** Geometric spreading yields Coulomb-like behavior ($V \propto 1/r$).

### 5. Atomic Spectra
* **Script:** `experiments/05_topological_atom.py`
* **Objective:** Solve for the eigenstates of a particle interacting with a topological defect (Monopole).
* **Observation:** The system yields eigenstates with nodal structures matching standard **Hydrogen orbitals** ($1s$, $2p$, $3d$), arising from the spherical harmonics of the graph Laplacian.

### 6. Light-Matter Interaction (Cavity QED)
* **Script:** `experiments/06_qed_laser.py`
* **Objective:** Couple the emergent atomic states to a quantized photon mode.
* **Observation:** The system exhibits **Vacuum Rabi Oscillations**, demonstrating unitary population transfer (absorption/emission) between the "atom" and the "field" without ad-hoc transition rules.

### 7. Nuclear Dynamics Models
* **Script:** `experiments/07_nuclear_fusion.py`
* **Objective:** Simulate high-energy scattering and flavor-changing processes.
* **Observation:** The framework successfully models **Quantum Tunneling** through a repulsive barrier and spontaneous state transitions analogous to **Beta Decay**.

## 💻 Computational Universality

To verify the computational capacity of the substrate:

* **Script:** `demos/universality_test.py`
* **Method:** We construct a **Controlled-Z (CZ) Gate** utilizing gauge-strain interactions.
* **Significance:** The successful implementation of entangling gates implies that the Substrate is a **Universal Quantum Computer**, satisfying the Lloyd-Deutsch criteria for a system capable of simulating local quantum field theories.

## 🛠 Usage Example

You can use the `UnifiedSubstrate` class to construct and evolve custom graph topologies.

    from engine.substrate import UnifiedSubstrate

    # 1. Initialize a Universe (11x11x11 Grid)
    uni = UnifiedSubstrate(L_size=11)

    # 2. Inject a "Proton" (Topological Defect)
    uni.inject_defect(strength=4.0)

    # 3. Compile the Hamiltonian
    uni.build_hamiltonian()

    # 4. Solve for Bound States
    energies, wavefunctions = uni.solve_eigenstates(k=5)

    # 5. Visualize
    uni.psi = wavefunctions[:, 0]
    uni.plot_density(title="Ground State Density")

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Resonance Laboratory** | *Investigating the Computational Foundations of Physics*