# Hilbert Substrate Framework (HSF) — Consolidation Pack

This folder is a **clean, reproducible “spine”** for consolidating the Hilbert Substrate Framework work:
- **Paper II:** Accessibility / Harmonion phase transition (locality as a kinetic phase)
- **Paper III:** Emergent fermionic statistics (robust even when geometry recovery is hard)
- **Shared tooling:** common metrics (V(d) ruler, TopN share, signal-entropy), JSONL logging, correlation analysis

It is designed to sit **next to** your existing repo (e.g. `c:\GitHub\hilbert_substrate\`) and give you:
1) a canonical folder layout,
2) a single place to drop/rename “blessed” scripts,
3) a repeatable way to analyze JSONL sweeps into tables/figures.

---

## 1) Canonical directory layout

```
hsf_consolidation_pack/
  docs/
    PAPER2_accessibility_notes.md
    PAPER3_fermions_notes.md
    RESULTS_SCHEMA.md
  scripts/
    scramble_recover_test_patched_signal_cache.py
    scramble_recover_test_patched_signal_cache_gpu.py
    scramble_recover_multichain.py
  analysis/
    parse_jsonl.py
    summarize_jsonl.py
    correlations.py
  outputs/
    (your runs go here; JSONL + summary JSON)
```

**Rule of thumb:** scripts produce JSONL; analysis consumes JSONL and produces CSV/plots.

---

## 2) “Blessed” scripts and what they are for

### A) `scramble_recover_test_patched_signal_cache.py`
Single-chain STROBE + FLOW pipeline with:
- objectives: sparse / signal / range
- metrics: sparse reduction, signal entropy reduction, V(d) ring ruler, TopN share
- JSONL output per seed

Use it for quick sweeps and algorithm iteration.

### B) `scramble_recover_test_patched_signal_cache_gpu.py`
Same pipeline, but with optional `--backend cupy` to accelerate FLOW-only on an NVIDIA GPU.

Use it for “hero runs” (longer FLOW, higher N if you go there) and profiling.

### C) `scramble_recover_multichain.py`
Best-of-K **multi-chain STROBE** (CPU-parallel) + optional FLOW-once on best chain.

Use it to test Paper II’s central claim:
> global scrambles exit the accessible basin and are exponentially hard to undo with local moves.

---

## 3) The two core empirical claims you now have

### Paper II (Accessibility / Harmonion transition)
**Observation:**  
- 1-local (product SU(2)^N) scrambles preserve the interaction graph; recovery is trivial.
- global SU(2^N) scrambles destroy the ring ruler hierarchy (V(2)/V(1) ~ 1) and local recovery fails.

**Interpretation:**  
Locality is a *kinetically protected phase*; global scrambles move you outside the locally-accessible basin.

**Next figure to lock this in:**  
Sweep **scramble circuit depth** (neighbor 2-qubit layers) and measure success probability vs depth.

### Paper III (Emergent fermions)
**Observation:**  
Fermionic fingerprints (JW anticommutation, sector additivity, Pauli-pressure curvature) remain robust even when geometry-blind locality recovery is inconsistent at N=8.

**Interpretation:**  
Exchange statistics / matter structure can be a more stable invariant than geometry at small N, and may serve as a scaffold for later locality.

---

## 4) Output schema (what every run should log)

Each seed/run should log:
- run header: N, model, scramble type, strobe objective, move set, flow parameters
- metrics:
  - `sparse_cost_initial/final`, `sparse_reduction`
  - `signal_entropy_initial/final`, `signal_entropy_reduction`
  - `topN_share_final`
  - `V_ring` dictionary OR at least `V2_ring_reduction` and (optionally) `V2_over_V1` both before/after
- success flags:
  - `locality_recovered_sparse`
  - `locality_recovered_signal`
- optional `fermion_audit_results` for Paper III

See `docs/RESULTS_SCHEMA.md`.

---

## 5) How to run (Windows one-liners)

### Single-chain sweep (8 seeds)
```bat
python scripts\scramble_recover_test_patched_signal_cache.py --N 8 --model xx --scramble global --recover both --strobe-objective signal --cycles 12000 --flow-steps 30 --dt 0.001 --p 4 --max-weight 4 --seed-start 0 --seed-count 8 --jobs 1 --blas-threads 1 --progress --partial-output outputs\sweep_N8_xx_signal.jsonl
```

### Multi-chain (best-of-12) per seed, 32 seeds, log JSONL
```bat
python scripts\scramble_recover_multichain.py --N 8 --model xx --scramble global --chains 12 --cores 12 --cycles 8000 --strobe-objective sparse --seed-start 0 --seed-count 32 --partial-output outputs\multichain_sparse_N8_xx.jsonl --progress
```

### Summarize a JSONL sweep
```bat
python analysis\summarize_jsonl.py --in outputs\sweep_N8_xx_signal.jsonl
```

### Correlate metrics (e.g. fermion vs locality)
```bat
python analysis\correlations.py --in outputs\yourfile.jsonl --out outputs\corr.csv
```

---

## 6) What to do next (clean, paper-ready)

1) **Freeze “blessed” script versions** (hash + tag in manifest)
2) Run the decisive Paper II sweep:
   - neighbor-circuit scramble depth d ∈ {0,2,4,6,8,12,16}
   - recovery with local move constraints
   - plot success probability vs depth (“accessibility transition curve”)
3) Run Paper III robustness sweeps:
   - models: xx vs xxz vs xxx
   - scramble: local-circuit depth (moderate) vs global
   - show fermion audits remain sharp when locality fails
