# The Hilbert Substrate Framework

**A Computational Framework Investigating Emergent Physics from Information-Theoretic Constraints.**

![Build Status](https://img.shields.io/badge/simulation-active-green.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8+-yellow.svg)

## Overview

This research codebase simulates physics not by discretizing differential equations, but by modeling the dynamics of a unitary, graph-based substrate. The project explores the hypothesis that fundamental physical laws (spacetime, matter, and forces) can emerge as effective field theories from a minimal set of information-theoretic constraints.

Rather than presupposing particles or background geometry, this framework starts with a raw Hilbert space and tests whether standard physical phenomena appear as emergent behaviors of the system's unitary evolution.

## Theoretical Postulates

The simulation is strictly constrained by three core assumptions:

1.  **Hilbert Space Realism:** The system is defined by a state vector |Ψ⟩ on a graph. Space is modeled as a relational network rather than a pre-existing manifold.
2.  **Unitary Evolution:** Dynamics are governed solely by unitary rotation (U = e^{-iHt}), preserving information.
3.  **Geometric Memory:** "Forces" are modeled as energy costs associated with the deformation of local gauge fields (Berry phases) on the graph links.

## Installation

The framework requires Python 3.8+ and standard scientific computing libraries.

```bash
# Clone the repository
git clone https://github.com/intersection-dynamics/hilbert-substrate.git
cd hilbert-substrate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** `numpy`, `scipy`, `matplotlib` (Optional: `cupy` for GPU acceleration).

## Experiments & Validations

The `experiments/` directory contains scripts that test the framework's ability to reproduce known physical phenomena.

| Experiment | Script | Observation |
|------------|--------|-------------|
| 01: Causal Structure | `01_emergent_space.py` | Lieb-Robinson bound emerges; finite speed of light from local connectivity |
| 02: Fermionic Statistics | `02_derive_fermions.py` | Exact −1 phase shift under particle exchange from SU(2) holonomy |
| 03: Inertial Dynamics | `03_derive_inertia.py` | F=ma recovered; mass as gauge stiffness |
| 04: Force Potentials | `04_derive_forces.py` | 1/r (EM), Yukawa (Weak), linear confinement (Strong) |
| 05: Atomic Spectra | `05_topological_atom.py` | Hydrogen-like orbitals (1s, 2p, 3d) from topological defects |
| 06: Light-Matter Coupling | `06_qed_laser.py` | Vacuum Rabi oscillations without ad-hoc transition rules |
| 07: Nuclear Dynamics | `07_nuclear_fusion.py` | Quantum tunneling; beta decay analogues |
| 08: Memory Commit | `08_memory_commit.py` | Classical trajectories emerge from finite information resolution |
| 09: Bell Test | `09_chsh_violation.py` | CHSH parameter S = 2.8284 (Tsirelson bound) |

### Quantitative Results

| Phenomenon | Substrate Result | Theoretical Target | Status |
|------------|------------------|-------------------|--------|
| CHSH Bell Parameter | 2.8284 | 2√2 | **Exact** |
| Fermion Exchange Phase | −1.0000 + 0.0000i | −1 | **Exact** |
| Lieb-Robinson Velocity | 0.94 t | ≤ 2t | **Consistent** |
| Bound State Energy (1s) | −2.78 t | Discrete well | **Confirmed** |
| Rabi Frequency | 0.047 t | g√(n+1) | **Confirmed** |

## Computational Universality

To verify the computational capacity of the substrate:

* **Script:** `demos/universality_test.py`
* **Method:** Construction of a **Controlled-Z (CZ) Gate** via gauge-strain interactions.
* **Significance:** Successful implementation of entangling gates confirms the substrate is a **Universal Quantum Computer**, satisfying the Lloyd-Deutsch criteria for simulating local quantum field theories.

## Usage Example

```python
from engine.substrate import UnifiedSubstrate

# 1. Initialize a lattice (11x11x11 grid)
uni = UnifiedSubstrate(L_size=11)

# 2. Inject a topological defect (monopole)
uni.inject_defect(strength=4.0)

# 3. Compile the Hamiltonian
uni.build_hamiltonian()

# 4. Solve for bound states
energies, wavefunctions = uni.solve_eigenstates(k=5)

# 5. Visualize ground state
uni.psi = wavefunctions[:, 0]
uni.plot_density(title="Ground State Density")
```

## Related Work

This framework relates to several active research programs:

- **Loop Quantum Gravity:** Shares the SU(2) spin-network structure; differs in computational tractability
- **Tensor Networks (MERA/PEPS):** Shares geometry-entanglement connection; differs in focus on dynamics vs. ground states
- **Wolfram Physics Project:** Shares discrete graph approach; differs in using unitary (quantum) vs. classical rewriting rules

The novel contribution is the **Memory Commit hypothesis**: wavefunction collapse as an information-theoretic consequence of finite geometric bandwidth.

## Project Structure

```
hilbert-substrate/
├── engine/
│   └── substrate.py       # Core simulation class
├── experiments/
│   ├── 01_emergent_space.py
│   ├── 02_derive_fermions.py
│   ├── ...
│   └── 09_chsh_violation.py
├── demos/
│   └── universality_test.py
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this framework in research, please cite:

```bibtex
@software{bray2025substrate,
  author = {Bray, Ben},
  title = {The Hilbert Substrate Framework},
  year = {2025},
  url = {https://github.com/intersection-dynamics/hilbert-substrate}
}
```

---

*Independent Research Project*
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
