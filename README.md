# The Hilbert Substrate Framework

**Emergent Spacetime from Quantum Information Geometry**

Benjamin August Bray  
Independent Researcher  
December 2025

---

## Overview

This repository contains the numerical experiments supporting the paper *The Hilbert Substrate Framework: Emergent Spacetime from Quantum Information Geometry*.

We demonstrate that classical spacetime structure emerges from purely quantum-mechanical principles through two competing selection mechanisms:

1. **Dynamical Simplicity** — The Hamiltonian selects a factorization minimizing locality cost, yielding collective modes (harmonions)
2. **Observational Stability** — Environmental decoherence selects a pointer basis, yielding spatial structure

The central result: **Space is not fundamental. Space is what survives when the substrate observes itself.**

---

## Requirements

```
numpy>=1.20.0
scipy>=1.7.0
```

---

## Experiments

### Experiment 1: The Humpty Dumpty Test

**Script:** `experiments/reconstruction_and_analysis.py`

Scramble a local Hamiltonian with a random unitary, then recover the optimal factorization by minimizing locality cost.

```
python experiments/reconstruction_and_analysis.py
```

**Output:**
```
--- TARGET (Original Chain) Interaction Matrix ---
   Q0  Q1  Q2
Q0  -  **3.0 .
Q1 **3.0 -  **3.0
Q2  .  **3.0 -

--- SCRAMBLED (Hidden Geometry) Interaction Matrix ---
   Q0  Q1  Q2
Q0  -  **9.9**8.2
Q1 **9.9 -  **8.8
Q2 **8.2**8.8 -

--- RECOVERED (Emergent Geometry) Interaction Matrix ---
   Q0  Q1  Q2
Q0  -  **1.1**1.1
Q1 **1.1 -  **2.8
Q2 **1.1**2.8 -

Original Cost:  16.0000
Scrambled Cost: 37.9110
Recovered Cost: 4.7500
```

**Result:** The optimizer finds a basis with *lower* locality cost than the original spatial representation. It discovered the quasiparticle (harmonion) basis.

---

### Experiment 2: Basis Structure Analysis

**Script:** `experiments/analyze_recovered_unitary.py`

Examine what the recovered basis vectors look like in the original spatial frame.

```
python experiments/analyze_recovered_unitary.py
```

**Output:**
```
--- BASIS ANALYSIS ---
Mapping: Basis State (Rec) -> State Vector (Target/Site Frame)

|110_rec>  =>  0.79*|111>
|001_rec>  =>  0.70*|000> + tails
Others     =>  Delocalized magnon superpositions
```

**Result:** The vacuum state |111⟩ (ferromagnetic eigenstate) is preserved with 79% fidelity. Other basis vectors correspond to collective excitations (magnons). The optimizer found the quasiparticle basis.

---

### Experiment 3: The Battle of the Bases

**Script:** `experiments/decoherence_basis_test.py`

Couple both spatial and harmonion states to an environment and track decoherence.

```
python experiments/decoherence_basis_test.py
```

**Output:**
```
    Results: Purity vs Time
    Time   | Purity (Spatial)   | Purity (Harmonion)
    --------------------------------------------------
    0.00   | 1.0000             | 1.0000
    0.10   | 0.9748             | 0.3063
    0.20   | 0.8745             | 0.2352
    0.50   | 0.6260             | 0.2098

    CONCLUSION: CONFIRMED.
    The Spatial basis survived. The Harmonion basis was destroyed.
    Space is the 'Pointer Basis' selected by the environment.
```

**Result:** The harmonion basis, despite being dynamically optimal (lower locality cost), is fragile under decoherence. The spatial basis survives. Environmental monitoring selects for spatial structure.

---

## The Core Result

| Principle | Selection Criterion | Produces |
|-----------|---------------------|----------|
| Dynamical Simplicity | Minimize locality cost | Harmonion basis |
| Observational Stability | Survive decoherence | Spatial basis |

The substrate wants to be harmonions.  
The environment forces it to be spatial.  
**Reality is the compromise.**

---

## Repository Structure

```
hilbert_substrate/
├── README.md
├── requirements.txt
├── The_Hilbert_Substrate_Framework.pdf
├── experiments/
│   ├── humpty_dumpty.py              # Basic locality cost test
│   ├── reconstruction_and_analysis.py # Geometry recovery + analysis
│   ├── analyze_recovered_unitary.py   # Basis structure examination
│   └── decoherence_basis_test.py      # Battle of the Bases
```

---

## Citation

```bibtex
@article{bray2025substrate,
  title={The Hilbert Substrate Framework: Emergent Spacetime 
         from Quantum Information Geometry},
  author={Bray, Benjamin August},
  year={2025}
}
```

---

## Contact

Benjamin August Bray  
sjbbray@gmail.com