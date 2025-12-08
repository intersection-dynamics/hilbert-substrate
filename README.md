# Finite-Memory Unitary Lattice Simulations

**Investigating emergent decoherence in unitary quantum systems under information-theoretic constraints.**

![Build Status](https://img.shields.io/badge/status-experimental-orange.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8+-yellow.svg)

## The Core Result: Parameter-Free Decoherence

This repository contains numerical experiments testing the hypothesis that **wavefunction collapse** can be modeled as a "memory commit" event in a finite-resource unitary system.

Our primary finding (Experiment 08) demonstrates that imposing a *derived* information density limit on a unitary lattice induces a **sawtooth entropy cycle**, effectively reproducing classical trajectories from quantum dispersion without external noise or arbitrary collapse parameters.

The key insight: the vacuum noise floor of a finite lattice scales as

$$\varepsilon = \sqrt{\frac{2}{N}}$$

where $N$ is the number of lattice sites. Probability amplitudes below this threshold are thermodynamically indistinguishable from vacuum fluctuations. This is not a tunable parameter—it follows from system size.

![Sawtooth Entropy](experiments/08_derived_memory.png)

*Fig 1: Shannon entropy of a wavepacket on a 60³ lattice. The sawtooth pattern represents cycles of unitary dispersion (entropy rise) followed by memory commit events (entropy drop) when low-density tails fall below the vacuum noise floor.*

## Key Experiments

| Experiment | Script | Result |
|------------|--------|--------|
| Memory Commit | `08_memory_commit.py` | Classical trajectories emerge from derived ε = √(2/N) |
| Fermionic Statistics | `02_derive_fermions.py` | Exact −1 phase shift from SU(2) holonomy |
| Causal Bounds | `01_emergent_space.py` | Lieb-Robinson bound yields effective c |
| Topological Atom | `05_topological_atom.py` | Hydrogen-like orbitals from monopole defects |
| Light-Matter Coupling | `06_qed_laser.py` | Vacuum Rabi oscillations |
| Bell Test | `09_chsh_violation.py` | S = 2.8284 (Tsirelson bound, exact) |
| Stability Basin | `10_stability_basin.py` | Phase diagram shows narrow window for stable atoms |

### Quantitative Results

| Phenomenon | Result | Target | Status |
|------------|--------|--------|--------|
| CHSH Bell Parameter | 2.8284 | 2√2 | **Exact** |
| Fermion Exchange Phase | −1.0000 | −1 | **Exact** |
| Memory Grain ε | 0.00304 | √(2/N) | **Derived** |
| Stability Window | Q ∈ [5, 7] | — | **Discovered** |

## Quick Start

Requires Python 3.8+ and standard scientific libraries.

```bash
# Clone the repository
git clone https://github.com/intersection-dynamics/hilbert-substrate.git
cd hilbert-substrate

# Install dependencies
pip install -r requirements.txt

# Run the core memory experiment
python experiments/08_memory_commit.py
```

**Requirements:** `numpy`, `scipy`, `matplotlib`

## The Memory Commit Mechanism

The central experiment (`08_memory_commit.py`) works as follows:

1. Initialize a Gaussian wavepacket on a 3D lattice
2. Evolve unitarily via sparse Hamiltonian exponentiation
3. At each timestep, prune amplitudes where |ψ|² < ε
4. Renormalize and continue

The system self-regulates. Instead of spreading indefinitely (pure unitary dispersion), the wavepacket maintains a coherent, localized trajectory. The entropy oscillates in a characteristic sawtooth pattern:

- **Rise:** Unitary evolution disperses the wavepacket, increasing spatial entropy
- **Drop:** Memory commit prunes sub-threshold tails, decreasing entropy

This cycle continues without external intervention. Classical behavior emerges from quantum dynamics plus finite information capacity.

## Project Structure

```
hilbert-substrate/
├── engine/
│   └── substrate.py          # Core lattice simulation class
├── experiments/
│   ├── 01_emergent_space.py  # Lieb-Robinson bound
│   ├── 02_derive_fermions.py # Exchange statistics
│   ├── 05_topological_atom.py# Atomic orbitals
│   ├── 06_qed_laser.py       # Rabi oscillations
│   ├── 08_memory_commit.py   # Core decoherence result
│   ├── 09_chsh_violation.py  # Bell test
│   └── 10_stability_basin.py # Phase diagram
├── demos/
│   └── universality_test.py  # CZ gate implementation
├── requirements.txt
└── README.md
```

## Theoretical Context

This work relates to several active research programs:

- **Objective Collapse Models (GRW, Penrose-Diósi):** These postulate collapse parameters. We derive ε from system size.
- **Loop Quantum Gravity:** Shares SU(2) spin-network structure. We prioritize computational tractability.
- **Tensor Networks:** Shares geometry-entanglement connection. We focus on dynamics rather than ground states.

The novel contribution is demonstrating that a *derived* (not postulated) information limit can induce decoherence-like behavior in a purely unitary system.

## Limitations

This is exploratory numerical work, not a complete theory. Known limitations:

- Lattice artifacts at high momentum (Lorentz violation at k → π/a)
- No continuum limit proof
- Collapse mechanism is phenomenological, not derived from first principles
- Stability basin boundaries not yet mapped to physical constants

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{bray2025finitemem,
  author = {Bray, Ben},
  title = {Finite-Memory Unitary Lattice Simulations},
  year = {2025},
  url = {https://github.com/intersection-dynamics/hilbert-substrate}
}
```

---

*Independent research project. Contact: [your email if desired]*