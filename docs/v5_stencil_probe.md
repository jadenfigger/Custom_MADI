# v5 production-grid stencil probe

## Verdict

**GO — retain 40 ensembles × 100,000 walkers.** The production-step k_io result is decisively above the pilot correlation range, while the single-configuration probe does not identify evidence strong enough to justify an allocation change.

This report was generated from the restricted production-Monte-Carlo cross. It neither launches nor modifies the production build, W4, simulator, geometry code, library schema, or fitters.

## Probe geometry and contract checks

The declaration contains 13 retained `(rho, V)` pairs crossed with `[19.0, 20.0, 21.0]` s^-1: exactly 39 cellular entries and no free-water atom. This analysis received 13 complete `(rho,V)` groups / 39 triplets.

Its centre is canonical indices `(21, 41)`, `rho=100000` cells/uL, `V=6.29626521` pL, and `v_i=0.629626521`. The declared rho line supports `k=1,2,3,4`; the declared V line supports `k=1,2`; and `k_io=(19, 20, 21)` supports its linear `k=1` central difference.

The canonical code-derived spacings are `ln(rho[i+1]/rho[i])=0.10964690919` (`0.047619047619` decades) and `ln(V[j+1]/V[j])=0.15719821512` (`0.0682703173915` decades). `madi/config.py` supplies the timing/physics defaults, while the rho/V production generator is `madi.library.make_remediation_log_grid()`. These values differ slightly from the prompt's quoted decade values `0.047648` and `0.068259`; the canonical 64-node `geomspace` formulas were used rather than silently rounding either value.

All v5 checks passed before correlations were calculated: metadata records the ensemble-index CRN contract, the 40 x 100,000 production settings and full storage grid, variance reconstruction had maximum absolute error `2.83e-11`, and the subset mean reconstructed the main signal with maximum absolute error `6.6e-09`.

All 13 available fixed-nominal `(rho,V)` k_io groups have identical `per_ensemble_geometry` metadata. Stored `rhos`/`Vs` are retained as realized finite-geometry provenance, not used as literal adjacency keys: their realized/nominal ratio summaries are rho median `0.998285` and V median `1.00172`.

## CRN correlations at production spacing

Every listed pair preserves the shared ensemble index. The bootstrap resamples those aligned indices across all pair-column values, so it describes finite-ensemble uncertainty conditional on this one declared cross; it does not treat the 200 columns as 200 independent tissue realizations.

| Axis | Immediate pairs | Median r | IQR | Paired-ensemble bootstrap 95% CI | Undefined 0/0 values |
|---|---:|---:|---:|---:|---:|
| `rho` | 24 | 0.361 | 0.237 to 0.493 | 0.324 to 0.421 | 192 |
| `V` | 12 | 0.506 | 0.361 to 0.637 | 0.477 to 0.547 | 96 |
| `k_io` | 26 | 0.961 | 0.939 to 0.976 | 0.960 to 0.965 | 208 |

![Per-axis CRN correlation histograms.](figures/v5_stencil_probe/crn_correlation_histograms.png)

For the 1 s^-1 `k_io` transitions, the path-divergence heuristic gives `N_div=T Δk_io=0.128` and trajectory survival `exp(-N_div)=0.880`. The result is evaluated against the pilot aggregate median 0.313 and pilot IQR upper endpoint 0.571; it is not a claim that the signal correlation itself must equal 0.880.

| Transition (s^-1) | Neighbor pairs | N_div | exp(-N_div) | Median r | IQR |
|---|---:|---:|---:|---:|---:|
| 19 → 20 | 13 | 0.128 | 0.880 | 0.961 | 0.938 to 0.977 |
| 20 → 21 | 13 | 0.128 | 0.880 | 0.96 | 0.94 to 0.976 |

The path-divergence prediction status is **supported** under the predeclared bootstrap rule: The production-step result is called support only when its paired-bootstrap 95% lower bound exceeds the pilot IQR upper endpoint 0.571; it is called not supported when its upper bound is at or below the pilot median 0.313. Intermediate results are inconclusive. This is not a test that signal-level r must equal exp(-N_div); exp(-N_div) is a trajectory-survival heuristic.

## Directly observed noise

The primary noise result is the directly observed per-entry SE `sqrt(signal_variance / 40)`, shown by timing and b below. A signal-independent `1/sqrt(3 E Nw)` line is intentionally not used as a decomposition reference: it assumes `Var(cos phi)=1`, which fails at low b.

Across all cellular entry-columns with `b>0`, observed SE has median `0.00022`, IQR `0.000198` to `0.000248`, and is `1.09×` the pilot's projected median `0.000202`. This comparison is descriptive, not a clean geometry/walker split.

![Observed production-probe SE by b and timing.](figures/v5_stencil_probe/observed_se_by_b_and_timing.png)

The v5 stored arrays do not contain per-walker second moments or independently re-used geometries, so they cannot separate geometry-realization noise from within-walker three-axis correlation. The post-production W4 independent-seed replicate is the experiment that would distinguish those causes; it is not launched or assumed here.

## Derivative magnitudes and beta

For each declared central stencil, `J_hat=(S(+k)-S(-k))/(2 k h1)` and `Var(J_hat)=[Var(S+)+Var(S-)-2 Cov(S+,S-)]/(2 k h1)^2`, with endpoint covariance calculated from matched ensemble indices and divided by 40 for covariance of entry means. `h1` is the exact canonical natural-log step for rho/V and 1 s^-1 for k_io. `beta=Var(J_hat)/J_hat^2` is retained as undefined for deterministic `0/0` columns and infinite if a zero derivative retains variance; no columns are filtered away.

| Axis | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| `rho` | 0.00681 [0.000603, 0.09] (finite 576/600; ∞ 0; undefined 24) | 0.00201 [0.000241, 0.0157] (finite 576/600; ∞ 0; undefined 24) | 0.000998 [9.71e-05, 0.00747] (finite 576/600; ∞ 0; undefined 24) | 0.000355 [4.51e-05, 0.00209] (finite 576/600; ∞ 0; undefined 24) |
| `V` | 0.148 [0.0209, 0.716] (finite 576/600; ∞ 0; undefined 24) | 0.0724 [0.0039, 0.524] (finite 576/600; ∞ 0; undefined 24) | N/A | N/A |
| `k_io` | 0.00694 [0.000774, 0.0307] (finite 2496/2600; ∞ 0; undefined 104) | N/A | N/A | N/A |

Fisher information is quadratic in the derivative: `E[J_hat^2]=J^2+Var(J_hat)`. Thus leaving this uncorrected inflates a Fisher diagonal by `1+beta` and deflates the corresponding CRLB by `sqrt(1+beta)`.

![Finite beta distributions versus stencil width.](figures/v5_stencil_probe/beta_vs_stencil_width.png)

The largest finite beta values and their absolute-derivative percentile are listed below. A low derivative percentile means the column is weakly informative for that axis under comparable noise; this report cannot know a downstream Fisher weighting or exclusion policy, so it does not claim which columns a fitter will ultimately use.

| Axis / k | δ, Δ, b | beta | |J| percentile | Lowest derivative quartile? |
|---|---|---:|---:|---|
| `rho` / 1 | (7, 25, 10500) | 1.65e+03 | 0.042 | yes |
| `rho` / 1 | (5, 15, 5000) | 703 | 0.045 | yes |
| `rho` / 1 | (7, 25, 9500) | 690 | 0.043 | yes |
| `rho` / 2 | (5, 15, 9000) | 5.47e+03 | 0.042 | yes |
| `rho` / 2 | (5, 15, 9500) | 182 | 0.043 | yes |
| `rho` / 2 | (5, 15, 10000) | 115 | 0.045 | yes |
| `rho` / 3 | (5, 15, 12000) | 0.631 | 0.042 | yes |
| `rho` / 3 | (5, 15, 12000) | 0.493 | 0.043 | yes |
| `rho` / 3 | (5, 15, 12000) | 0.458 | 0.045 | yes |
| `rho` / 4 | (5, 15, 12000) | 0.0881 | 0.042 | yes |
| `rho` / 4 | (5, 15, 11500) | 0.0706 | 0.047 | yes |
| `rho` / 4 | (5, 15, 12000) | 0.0668 | 0.043 | yes |
| `V` / 1 | (15, 40, 11000) | 6.99e+03 | 0.042 | yes |
| `V` / 1 | (25, 60, 11000) | 6.67e+03 | 0.043 | yes |
| `V` / 1 | (25, 60, 11500) | 3.61e+03 | 0.045 | yes |
| `V` / 2 | (20, 50, 6500) | 2.55e+05 | 0.042 | yes |
| `V` / 2 | (20, 50, 7000) | 3.25e+03 | 0.043 | yes |
| `V` / 2 | (15, 40, 7000) | 1.63e+03 | 0.045 | yes |
| `k_io` / 1 | (5, 15, 10500) | 8.13e+03 | 0.040 | yes |
| `k_io` / 1 | (5, 15, 11000) | 59.6 | 0.041 | yes |
| `k_io` / 1 | (5, 15, 10000) | 39.3 | 0.041 | yes |

## Fixed-cost ensemble/walker reallocation

A single 40 × 100,000 configuration cannot identify the independent-walker versus ensemble-correlated share of the paired endpoint-difference variance. At a new configuration, those extrema scale as `(40×100,000)/(E×Nw)` and `40/E`, respectively. The table therefore gives a range, not a fitted point prediction. At fixed total cost, the independent extreme is invariant and the correlated extreme improves monotonically with more ensembles; no defensible interior optimum can be identified from this artifact alone.

| Configuration | Axis-walks | Variance-scale envelope | Finite rho beta median envelope | Finite rho beta P05–P95 envelope |
|---|---:|---:|---:|---:|
| 20 x 200,000 | 12,000,000 | 1–2 | 0.00681–0.0136 | [4.55e-05–2.23] to [9.11e-05–4.45] |
| 40 x 100,000 (current) | 12,000,000 | 1–1 | 0.00681–0.00681 | [4.55e-05–2.23] to [4.55e-05–2.23] |
| 60 x 66,667 | 12,000,060 | 0.667–1 | 0.00454–0.00681 | [3.04e-05–1.48] to [4.55e-05–2.23] |
| 80 x 50,000 | 12,000,000 | 0.5–1 | 0.00341–0.00681 | [2.28e-05–1.11] to [4.55e-05–2.23] |
| 100 x 40,000 | 12,000,000 | 0.4–1 | 0.00272–0.00681 | [1.82e-05–0.89] to [4.55e-05–2.23] |

![Fixed-cost rho-beta reallocation bounds.](figures/v5_stencil_probe/ensemble_walker_reallocation_table.png)

The literal 20 × 600,000 through 100 × 120,000 request is not a fixed 12-million-axis-walk allocation: every row costs 36 million axis-walks. It is retained in `rho_beta_reallocation_bounds.csv` as a separately labelled three-times-budget sensitivity sweep, not used to recommend a production configuration.

No `GO-WITH-CHANGE` recommendation is made from a one-configuration envelope. A configuration change would require a new validation cycle and an identifiable component decomposition; absent that evidence, the current 40 × 100,000 specification is retained.

## Limitations

This is one deliberately chosen cross, not a measurement over the full masked grid. The CRN bootstrap is conditional on its 40 ensembles and correlated diagnostic columns. Beta is strongly structured and can diverge wherever the finite-difference derivative approaches zero. Reallocation bounds assume stationary derivative magnitude and component correlations while changing E/Nw; they are planning bounds, not a precision forecast. W4 remains out of scope and is required to separate geometry-realization noise from within-walker axis correlation.

## Reproduction

```bash
python analysis/v5_stencil_probe.py \
  libraries/madi_v5_stencil_probe.shard*.npz \
  --declaration data/madi_v5_stencil_probe_entry_subset.json \
  --expected-shards 13 \
  --declared-shards 13 \
  --output-dir docs/figures/v5_stencil_probe \
  --report docs/v5_stencil_probe.md
```
