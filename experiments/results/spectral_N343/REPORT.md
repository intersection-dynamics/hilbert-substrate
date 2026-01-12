# HSF Spectral Dimension Study (graph-only)
- Created: `2026-01-10T16:31:42.157877Z`
- N = 343
- Total graphs: 31
- Runtime: 1.35s

## Dimension aggregates (mean over fit-window means)

- **3**: n=1 | Ds=1.743 ± 0.000 (min=1.743, max=1.743)
- **non-geometric**: n=30 | Ds=1.746 ± 0.036 (min=1.708, max=1.783)

## Per-graph quick view

- **3D_lattice_7x7x7** (d=3): Ds_fit_mean=1.743, Ds_fit_std=0.774, r2_mean=0.846
- **random_regular_d4** (d=None): Ds_fit_mean=1.782, Ds_fit_std=0.705, r2_mean=0.864
- **random_regular_d4** (d=None): Ds_fit_mean=1.782, Ds_fit_std=0.702, r2_mean=0.865
- **random_regular_d4** (d=None): Ds_fit_mean=1.781, Ds_fit_std=0.700, r2_mean=0.866
- **random_regular_d4** (d=None): Ds_fit_mean=1.783, Ds_fit_std=0.708, r2_mean=0.863
- **random_regular_d4** (d=None): Ds_fit_mean=1.783, Ds_fit_std=0.705, r2_mean=0.864
- **random_regular_d4** (d=None): Ds_fit_mean=1.783, Ds_fit_std=0.709, r2_mean=0.863
- **random_regular_d4** (d=None): Ds_fit_mean=1.781, Ds_fit_std=0.701, r2_mean=0.866
- **random_regular_d4** (d=None): Ds_fit_mean=1.783, Ds_fit_std=0.708, r2_mean=0.863
- **random_regular_d4** (d=None): Ds_fit_mean=1.783, Ds_fit_std=0.706, r2_mean=0.864
- **random_regular_d4** (d=None): Ds_fit_mean=1.782, Ds_fit_std=0.706, r2_mean=0.864
- **random_regular_d4** (d=None): Ds_fit_mean=1.782, Ds_fit_std=0.707, r2_mean=0.864
- **random_regular_d4** (d=None): Ds_fit_mean=1.782, Ds_fit_std=0.706, r2_mean=0.864
- **random_regular_d4** (d=None): Ds_fit_mean=1.782, Ds_fit_std=0.705, r2_mean=0.864
- **random_regular_d4** (d=None): Ds_fit_mean=1.782, Ds_fit_std=0.708, r2_mean=0.863
- **random_regular_d4** (d=None): Ds_fit_mean=1.782, Ds_fit_std=0.709, r2_mean=0.863
- **random_regular_d6** (d=None): Ds_fit_mean=1.709, Ds_fit_std=0.963, r2_mean=0.765
- **random_regular_d6** (d=None): Ds_fit_mean=1.710, Ds_fit_std=0.960, r2_mean=0.766
- **random_regular_d6** (d=None): Ds_fit_mean=1.709, Ds_fit_std=0.965, r2_mean=0.764
- **random_regular_d6** (d=None): Ds_fit_mean=1.709, Ds_fit_std=0.963, r2_mean=0.765
- **random_regular_d6** (d=None): Ds_fit_mean=1.710, Ds_fit_std=0.959, r2_mean=0.767
- **random_regular_d6** (d=None): Ds_fit_mean=1.710, Ds_fit_std=0.960, r2_mean=0.766
- **random_regular_d6** (d=None): Ds_fit_mean=1.709, Ds_fit_std=0.962, r2_mean=0.765
- **random_regular_d6** (d=None): Ds_fit_mean=1.709, Ds_fit_std=0.962, r2_mean=0.765
- **random_regular_d6** (d=None): Ds_fit_mean=1.709, Ds_fit_std=0.965, r2_mean=0.764
- **random_regular_d6** (d=None): Ds_fit_mean=1.709, Ds_fit_std=0.963, r2_mean=0.765
- **random_regular_d6** (d=None): Ds_fit_mean=1.709, Ds_fit_std=0.964, r2_mean=0.765
- **random_regular_d6** (d=None): Ds_fit_mean=1.710, Ds_fit_std=0.961, r2_mean=0.766
- **random_regular_d6** (d=None): Ds_fit_mean=1.708, Ds_fit_std=0.966, r2_mean=0.763
- **random_regular_d6** (d=None): Ds_fit_mean=1.710, Ds_fit_std=0.961, r2_mean=0.766
- **random_regular_d6** (d=None): Ds_fit_mean=1.710, Ds_fit_std=0.961, r2_mean=0.766

## Interpretation tip

For small graphs, spectral dimension is typically *window-sensitive*. A good sign of a meaningful estimate is: (i) high R^2 across windows, (ii) low Ds_fit_std across windows, and (iii) a local-slope Ds(t) curve that plateaus over a decade or more in t. This script makes those failure modes visible rather than hiding them.