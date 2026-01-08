# HSF Refolding Constraint-Separation Report (random sparse target)
- Created: `2026-01-08T19:10:57.086491Z`
- Outdir: `D:\Github\hilbert-substrate\experiments\REFOLD_SEP_SPARSE_20260108_123748`
- N=8  seeds=12  target_edges_M=12  force_connected=True
- Flow: steps=2500 eps=0.06 temp0=0.02 temp_decay=0.9995 cost_every=5 anchor_every=5
- No-refolding: anchor=quadratic anchor_min=0.98 hard_anchor=True lambda=10.0
- runtime_sec: 5588.66

## Separation criterion
If free refolding reduces leakB substantially but constrained refolding cannot,
then no-refolding is separated from no-signaling (both use only local adjacent gates).

## Files
- baseline.json
- targets.jsonl
- runs/runs.jsonl
- summary.json
- manifest.json