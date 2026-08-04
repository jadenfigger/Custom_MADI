# P0-A full-facet geometry validation

Tier: **A** (local CPU).  Date: 2026-08-03.

This report validates the production geometry implemented in
`madi/ensemble.py`: SI Eq. S2's conjunction of all shifted Voronoi facets,
with `alpha_i=min(alpha*,0.9*d_nn/2)`, and SI Eq. S7a inversion for
`alpha*`.  It does not build a library.  The run used the certified
five-million-cell reference artifact
`data/geometry_reference_si_kappa_0p9.npz` (SHA-256
`6a2957eb3d6f89fdebc61d83e7cabb65c67812415879e00b5c07614680785e47`).
The reference is not used to calibrate the measured `v_i`, but loading it here
also proves that the production runtime accepts the completed artifact without
an uncertified-reference override.

All entries use the 128-ms SI Eq. S8 domain (`W=554.25626 um`) and 50,000
independent uniform volume samples.  The acceptance bound is
`max(0.005, 4*MC SE)`.  Every row passes.

| Target `v_i` | `rho=1e5`: measured ± SE | `rho=5e5`: measured ± SE | `rho=1e6`: measured ± SE |
|---:|---:|---:|---:|
| 0.40 | 0.39768 ± 0.00219 | 0.39822 ± 0.00219 | 0.39954 ± 0.00219 |
| 0.50 | 0.50188 ± 0.00224 | 0.49808 ± 0.00224 | 0.49812 ± 0.00224 |
| 0.60 | 0.59770 ± 0.00219 | 0.59902 ± 0.00219 | 0.59706 ± 0.00219 |
| 0.70 | 0.69890 ± 0.00205 | 0.69982 ± 0.00205 | 0.69710 ± 0.00206 |
| 0.80 | 0.79970 ± 0.00179 | 0.80002 ± 0.00179 | 0.79972 ± 0.00179 |
| 0.85 | 0.85098 ± 0.00159 | 0.85192 ± 0.00159 | 0.85240 ± 0.00159 |
| 0.90 | 0.90182 ± 0.00133 | 0.90010 ± 0.00134 | 0.89952 ± 0.00134 |
| 0.95 | 0.95120 ± 0.00096 | 0.95034 ± 0.00097 | 0.94890 ± 0.00098 |
| 0.99 | 0.99094 ± 0.00042 | 0.99042 ± 0.00044 | 0.99068 ± 0.00043 |

## SI §S.IV.a shortcut discrepancy

At `rho=5e5 cells/uL`, 100,000 samples per target compare the diagnostic-only
two-nearest shortcut with the full Eq. S2 rule.  The full rule is a conjunction
including the second-nearest facet, so the shortcut's error is provably
one-sided: it changes gap points to cells and never the reverse.

| Target `v_i` | Disagreement | Shortcut excess `v_i` ± MC SE |
|---:|---:|---:|
| 0.40 | 2.870% | 0.02870 ± 0.00053 |
| 0.50 | 2.103% | 0.02103 ± 0.00045 |
| 0.60 | 1.348% | 0.01348 ± 0.00036 |
| 0.70 | 0.770% | 0.00770 ± 0.00028 |
| 0.80 | 0.333% | 0.00333 ± 0.00018 |
| 0.85 | 0.190% | 0.00190 ± 0.00014 |
| 0.90 | 0.113% | 0.00113 ± 0.00011 |
| 0.95 | 0.021% | 0.00021 ± 0.00005 |
| 0.99 | 0.001% | 0.00001 ± 0.00001 |

For each target, a separate 64-point test compared the adaptive-radius
classifier against a brute-force all-seed SI Eq. S2 calculation.  There were
zero disagreements at every target.  The limit test at `alpha=0` also gives
identical full-facet and two-nearest ordinary-Voronoi labels.

The structured reproduction record is
`logs/p0a_full_facet_certified_validation.json` (an ignored runtime artifact).
Rerun with `scripts/validate_full_facet_geometry.py`, using the certified
reference and omitting `--allow-uncertified-geometry-reference`.
