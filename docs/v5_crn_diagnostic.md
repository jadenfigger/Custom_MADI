# v5 pilot CRN diagnostic

Analysis completed 2026-08-12 from the four local
`libraries/madi_v5_remediation_pilot.shard00*.npz` artifacts. Reproducible
outputs, including the machine-readable summary and full per-column table, are
in [`figures/v5_crn_diagnostic/`](figures/v5_crn_diagnostic/).

## Verdict

**INVESTIGATE — retain the current 40 ensembles × 100,000 walkers
configuration for now; do not use this pilot to approve a rho- or V-axis
Fisher/CRLB derivative-noise claim.**

The artifact passes the v5 diagnostic-array and ensemble-index provenance
checks. However, its realized grid has no rho- or V-only neighbour pair at
all, so it cannot estimate the two correlations central to the planned
rho--V analysis, nor any requested beta value. In addition, the available
`k_io` correlations are much lower than the expected near-one cancellation:
median `r = 0.313` (IQR `-0.002` to `0.571`), rather than about 0.99. That is
an important contradiction to investigate before treating CRN cancellation as
an established production property.

This record does not launch or prepare any build, W4 replicate, simulator, or
fitter change.

## Artifact and contract checks

The four shards contain 25 entries: 24 cellular and one free-water atom. All
have the required float32 diagnostic arrays, `8` ensemble means, and the
declared 200-column pair-major/b-major subset.

The embedded metadata explicitly records the required pairing contract:

- `ensemble_means_subset` axis 1 is `e=0..7`;
- the order is the same across entries and independent of `rho`, `V`, and
  `k_io`;
- the walk seed is `(build_seed + 104729 * ensemble_index) mod 2^31`;
- the geometry seed is `(build_seed + 97003 * ensemble_index) mod 2^31`.

The variance cross-check passed before any correlation was calculated. Across
all 25 × 200 subset values, recomputing `var_e(m, ddof=1)` from the stored
ensemble means matched the corresponding stored `signal_variance` with maximum
absolute error `1.091e-09` and maximum relative error `3.660e-06`, within the
float32 check tolerance (`atol=2e-07`, `rtol=5e-05`). The companion subset
mean versus main-signal check had maximum absolute error `1.898e-08`.

As a further provenance check, all eight fixed-`(rho,V)` `k_io` groups carry
identical `per_ensemble_geometry` metadata across their three `k_io` entries.
This confirms geometry reuse in the artifact metadata; it does not by itself
prove that the retained signals enjoy near-complete random-walk cancellation.

## Realized-label adjacency

This analysis intentionally used the stored `kios`, `rhos`, and `Vs`, without
using nominal labels, rounding, or reconstructing a lattice. The eight
cellular `(rho,V)` groups are:

| Realized rho (cells/uL) | Realized V (pL) |
|---:|---:|
| 10,055.500 | 80.791424 |
| 26,604.457 | 33.328978 |
| 71,535.222 | 13.506083 |
| 193,170.280 | 2.218153 |
| 517,515.752 | 0.902371 |
| 1,390,103.623 | 0.365451 |
| 3,727,848.332 | 0.149076 |
| 9,999,135.938 | 0.060522 |

Each row has `k_io={0,20,130}` s^-1, but the rows trace a diagonal through
`(rho,V)` rather than an axis-aligned grid. The actual immediate-neighbour
structure is therefore:

| Axis | Fixed-coordinate groups | Group sizes | Immediate pairs | Result |
|---|---:|---:|---:|---|
| `k_io` | 8 | 8 × 3 | 16 | 8 pairs at 0--20 and 8 at 20--130 s^-1 |
| `rho` | 24 | 24 × 1 | 0 | not estimable |
| `V` | 24 | 24 × 1 | 0 | not estimable |

Any comparison between two of the eight rows would change both rho and V. It
would be a diagonal derivative, not an rho or V derivative, so it was not used.

## CRN correlations

Eight `b=0` columns per pair have exactly zero between-ensemble variance and
an undefined `0/0` correlation. They are reported as undefined, not filtered
for a cleaner result. The `k_io` histogram therefore contains 3,072 valid
values: 16 neighbour pairs × 192 nonzero-variance columns.

| `k_io` transition (s^-1) | Neighbour pairs | Median r | IQR |
|---|---:|---:|---:|
| 0 → 20 | 8 | 0.468 | 0.205 to 0.670 |
| 20 → 130 | 8 | 0.163 | -0.140 to 0.401 |
| all immediate `k_io` pairs | 16 | 0.313 | -0.002 to 0.571 |

The full `k_io` distribution ranges from `-0.840` to `0.968`. A
fixed-`(rho,V)`-group block bootstrap (eight blocks, preserving the correlated
columns and two transitions within a group) gives a descriptive 95% interval
of `0.204` to `0.402` for the aggregate median. It is deliberately not treated
as a precise inferential interval: every individual r uses only eight
ensembles, hence six degrees of freedom and an approximate standard error of
`1/sqrt(5) ≈ 0.45` near zero. The 200 columns are also correlated with each
other.

The observed pattern materially contradicts the anticipated `k_io r ≈ 0.99`.
The source code and metadata both establish the intended shared seed/index
contract, and the artifact shows matching geometry records, so this result is
not evidence to silently dismiss as an index mismatch. It does mean the
signal-level cancellation is weak in this pilot. Different accept/reject paths
after a permeability change could be one explanation, but this artifact cannot
distinguish that from a failure to share some part of the random stream. That
lineage needs direct investigation before the expected cancellation is used in
Fisher planning.

![Per-axis CRN-correlation histograms.](figures/v5_crn_diagnostic/crn_correlation_histograms.png)

## Observed noise versus the nominal walker level

For the pilot, the specified nominal independent walker level is

`1 / sqrt(3 × 8 × 512) = 0.0090211`.

For the 24 cellular entries and 192 `b>0` diagnostic columns, the observed
SE-to-nominal ratio has median `0.699` (IQR `0.578` to `0.842`; 5th--95th
percentile `0.421` to `1.071`; maximum `1.462`). The median rises from 0.592
at `b=500` to about 0.78 around `b=1,500--2,000`, then is about 0.70 at
`b=12,000` s/mm². The exact `b=0` signal is deterministic and has ratio zero.

Only 7.94% of the `b>0` entry-columns lie above the nominal level. Under the
requested two-component planning decomposition, the residual component is
zero at the median and third quartile; its 95th-percentile share of observed
variance is 12.8%, and its maximum is 53.2%.

A ratio substantially above one is compatible with an additional correlated
component, but it is **not** a clean geometry-noise measurement. It can also
arise because three Cartesian components harvested from one 3-D walk are not
independent. This pilot cannot separate those causes; the independent-seed W4
replicate is the post-production experiment that would do so.

![Observed-to-nominal SE ratio by b value and timing pair.](figures/v5_crn_diagnostic/observed_vs_nominal_se_ratio.png)

## Production projection and beta

For transparency, let `s_obs²` be the pilot variance of the entry mean,
`s_nom² = 1/(3×8×512)`, and define

`s_walk,pilot² = min(s_obs², s_nom²)` and
`s_resid,pilot² = max(s_obs² - s_nom², 0)`.

The cap prevents a physically meaningless negative residual at columns whose
signal variance is naturally below the signal-independent nominal level,
including `b=0`. For a configuration `(E,Nw)`, the script projects

`SE²(E,Nw) = s_walk,pilot² × (8×512)/(E×Nw) + s_resid,pilot² × 8/E`.

Thus the first component follows `1/sqrt(3 E Nw)` and the residual follows
`1/sqrt(E)`, exactly as specified. At 40 × 100,000, the projected cellular
`b>0` SE has median `2.019e-04`, IQR `1.668e-04` to `2.430e-04`, and a broad
5th--95th range `1.216e-04` to `1.570e-03`. This is an order-of-magnitude
extrapolation from eight ensembles and 512 walkers, not a precision forecast.

No beta is estimable from this artifact. `rho` and `V` have no three-node
realized-label central stencil even for `k=1`, and therefore none for
`k=2,3,4`; `k_io={0,20,130}` is asymmetric, includes zero, and cannot be put
on the specified log-grid central stencil. The table below is intentionally
all N/A rather than deriving numbers from diagonal or one-sided comparisons.

| Axis | beta(k=1) | beta(k=2) | beta(k=3) | beta(k=4) |
|---|---:|---:|---:|---:|
| `k_io` | N/A | N/A | N/A | N/A |
| `V` | N/A | N/A | N/A | N/A |
| `rho` | N/A | N/A | N/A | N/A |

If a qualifying artifact supplies these stencils, the analysis uses the
specified central-difference variance and reports
`beta = Var(J_hat)/J²`. An uncorrected Fisher diagonal would then be inflated
by `1 + beta`, and an uncorrected CRLB would be deflated by
`sqrt(1 + beta)`. No such bias statement is justified here because neither the
rho derivative magnitude nor its endpoint covariance exists in this pilot.

![Projected beta versus stencil width.](figures/v5_crn_diagnostic/projected_beta_vs_stencil.png)

## Fixed-cost ensemble/walker reallocations

At fixed `3×E×Nw`, the stipulated independent-walker term is invariant for
the entry mean: fewer walkers per ensemble are exactly offset by more
ensembles. The modeled residual term is the only term that improves with more
ensembles, so this simple model has no interior optimum. In this pilot it is
zero for most columns, hence the median projected SE is unchanged; only the
upper tail improves.

| Configuration | Axis-walks | Median projected SE, b>0 | P05--P95 projected SE, b>0 | rho beta(k=1) |
|---|---:|---:|---:|---:|
| 20 × 200,000 | 12,000,000 | 2.019e-04 | 1.216e-04--2.202e-03 | N/A |
| 40 × 100,000 (current) | 12,000,000 | 2.019e-04 | 1.216e-04--1.570e-03 | N/A |
| 60 × 66,667 | 12,000,060 | 2.019e-04 | 1.216e-04--1.293e-03 | N/A |
| 80 × 50,000 | 12,000,000 | 2.019e-04 | 1.216e-04--1.129e-03 | N/A |
| 100 × 40,000 | 12,000,000 | 2.019e-04 | 1.216e-04--1.018e-03 | N/A |

The 60 × 66,667 row differs by 60 axis-walks because walkers are integral.
The 80-ensemble option reduces the modeled 95th-percentile SE by about 28%,
but this applies only to the small upper-tail subset and cannot be translated
into the decision-relevant rho beta. Given the missing rho stencils, the
eight-ensemble uncertainty in the decomposition, and the validation cost of a
configuration change, this is not evidence for a reallocation.

The detailed 200-column results for every configuration are in
[`projected_reallocation_by_column.csv`](figures/v5_crn_diagnostic/projected_reallocation_by_column.csv).

![Fixed-cost reallocation table.](figures/v5_crn_diagnostic/ensemble_walker_reallocation_table.png)

## What must be resolved before a GO decision

1. Investigate why a fixed-geometry, same-index `k_io` sweep has only modest
   signal-level correlations despite the recorded shared seed contract. The
   problem this solves is direct: without knowing whether CRN cancellation
   actually survives the walker dynamics, a projected derivative covariance is
   not trustworthy.
2. Obtain an approved diagnostic artifact with genuine one-axis rho and V
   lines under the coordinate identity that the Fisher consumer will use. A
   central `k=1` estimate needs at least three such nodes; evaluating through
   `k=4` needs at least nine. The problem this solves is that diagonal rho/V
   changes cannot distinguish their two derivatives or supply endpoint
   covariance. This report does not prescribe a schema or simulator change.

Until those two issues are resolved, the evidence supports neither a
production reallocation nor a GO for the planned rho--V finite-difference
precision claim. Keeping 40 × 100,000 unchanged avoids an unvalidated build
change while the diagnostic gap is closed.

## Reproduction

Run the standalone reader in an environment with NumPy and Matplotlib:

```bash
python analysis/v5_crn_diagnostic.py \
  libraries/madi_v5_remediation_pilot.shard000.npz \
  libraries/madi_v5_remediation_pilot.shard001.npz \
  libraries/madi_v5_remediation_pilot.shard002.npz \
  libraries/madi_v5_remediation_pilot.shard003.npz \
  --output-dir docs/figures/v5_crn_diagnostic
```

The script is an artifact-only analysis: it imports no `madi` module and does
not alter a production build path.
