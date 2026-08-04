# P0-B SI geometry-reference validation

Tier: **B build, A artifact validation**.  Date: 2026-08-03.

This record concerns the MADI I SI §S.IV Eq. 5 calibration table only.  It
does **not** estimate `k_io` from residence times: the SI explicitly assigns
that observable to a different average, `1/<tau_i>`, rather than `⟨k_io⟩`.

## Completed reference artifact

`data/geometry_reference_si_kappa_0p9.npz`

- SHA-256: `6a2957eb3d6f89fdebc61d83e7cabb65c67812415879e00b5c07614680785e47`
- schema: `madi-si-single-cell-reference-v1`
- source: 8 contiguous shards, numbered 0 through 7
- cells per shard: 625,000; total: 5,000,000 independent central cells
- `kappa`: 0.90
- contraction: `all_shifted_voronoi_facets_S2`
- estimator: `untrimmed_arithmetic_mean_A_over_V`
- alpha grid: 26 values, linearly spaced in dimensionless
  `x = rho^(1/3) alpha*`
- supported production range: `v_i=0.40` through 1.00

The file has arrays `vi`, `alpha_x`, `mean_A_over_V_norm`,
`se_A_over_V_norm`, and `metadata_json`, all of the expected shape and dtype.
`v_i` increases strictly, while `alpha_x` and normalized `⟨A/V⟩` decrease
strictly.  The normalized mean ranges from 9.59314 at `v_i=0.40` to 6.30045 at
`v_i=1.00`; its relative Monte-Carlo SE is 0.0092--0.0157% across the table.
The production loader accepted the artifact with
`allow_uncertified_geometry_reference=False`.

## Scale and interpolation checks

`madi.ensemble.governing_mean_A_over_V()` linearly interpolates the reference
in `v_i` and multiplies it by `(rho / 1e9)^(1/3)`, where `rho` is in
cells/µL.  At `rho=5e5 cells/µL` this produced:

| `v_i` | `⟨A/V⟩` (`µm^-1`) |
|---:|---:|
| 0.40 | 0.761408 |
| 0.50 | 0.685608 |
| 0.78 | 0.558196 |
| 0.80 | 0.551862 |
| 0.90 | 0.523677 |
| 0.99 | 0.502292 |

The SI Fig. S6b low-volume sanity point, `V=0.75 pL` at this density, has
`v_i=0.375`, just below the production table's deliberately enforced 0.40
floor.  A **diagnostic-only** linear extrapolation of its first interval gives
0.783 `µm^-1`, consistent with the figure's approximately 0.80 value.  The
runtime refuses this extrapolation.  The high-volume SI point is represented:
`V=1.98 pL` at `v_i=0.99` gives 0.502 `µm^-1`, consistent with the plotted
approximately 0.50 value.

## Entry-level smoke check

A small Tier-A diagnostic entry at nominal `(k_io, rho, V) =
(20 s^-1, 5e5 cells/µL, 1.6 pL)` used the certified table and stored:

- governing `⟨A/V⟩ = 0.5518623725 µm^-1`;
- `p_p = 0.001620743206`; and
- `k_io_analytic_eq5 = 20.000000000000004 s^-1`.

Its finite realization was deliberately labelled from direct geometry
measurement, not from the request: `rho=513,499.80 cells/µL`, `V=1.55540 pL`,
and `v_i=0.79870`, satisfying `rho * V * 1e-6 = v_i` exactly to printed
precision.  It had zero escapes.  The diagnostic uses only 64 walkers / one
ensemble and is evidence of metadata/calibration routing, not signal
precision.

## Reproduction

The unit tests are Tier A:

```bash
PYTHONPATH=. pytest -q tests/physics_audit/test_si_reference_builder.py \
  tests/physics_audit/test_geometry_diagnostics.py -k governing_reference
```

The artifact validation command is Tier A once the Tier-B build exists:

```bash
PYTHONPATH=. python -m scripts.validate_full_facet_geometry \
  --geometry-reference data/geometry_reference_si_kappa_0p9.npz \
  --output logs/p0a_full_facet_certified_validation.json \
  --volume-samples 50000 --discrepancy-samples 100000 --brute-force-samples 64
```

The full reference-cloud calculation is Tier B and is implemented by
`scripts/build_si_geometry_reference.py`; no realized-`Omega_sim` A/V
measurement, percentile trimming, or residence-time calibration is part of
the production path.
