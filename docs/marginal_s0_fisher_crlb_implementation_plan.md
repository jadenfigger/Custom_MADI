# Marginal S0 and Fisher/CRLB implementation handoff

## Purpose and status

This document is the implementation handoff for two deferred MADI features:

1. nuisance-amplitude (`S0`) fitting by normalization, profiling, or Gaussian
   marginalization, with an optional marginalized-noise-scale Bayes model;
2. Fisher-information / Cramer-Rao lower-bound (CRLB) analysis of the current
   universal signal library.

The work was intentionally stopped after a read-only Phase 0 audit. **No
production fitting, library, or analysis code has been changed for this
feature.** Use this document as the starting point when implementation resumes.

Audit snapshot: commit `ce71729` (`running edema cohort`) with pre-existing
uncommitted user changes. Re-check file references if the surrounding code has
changed materially.

## Settled design decisions

| Topic | Decision |
|---|---|
| AMICO | Keep existing normalized/free-linear-amplitude AMICO behavior. Marginal-S0 and Student-t likelihoods are MAP/Bayes features. `--method amico --s0-mode marginal` should fail clearly, not silently run a different algorithm. |
| Marginal amplitude default | Fit one latent amplitude per input `(delta, Delta)` group. This agrees with the current default of averaging b=0 images within each Delta scan. |
| Cross-Delta averaging | In marginal mode, `--avg-s0` means one shared latent amplitude across Delta groups. It is the principled pooled-amplitude version of legacy S0 averaging. |
| Raw noise | Reuse existing raw `--noise-sigma`, supplied by the user or estimated by the existing background procedure when `--rician-correct` is on. Do not add an independently tuned `marginal_sigma`. |
| `sigma_m` | Keep it as the legacy normalized-Bayes temperature only. Marginal Bayes uses raw Gaussian noise and does not reinterpret `sigma_m`. |
| Amplitude uncertainty | Write amplitude standard-deviation maps. Do not add an S0 scalar uncertainty directly to kio/rho/V standard deviations; candidate weights carry the tissue-uncertainty effect. |
| Caching | Cache norms/log-norms per selected protocol and candidate filter, not globally at library load. The selected rows and b=0 weighting change those norms. |
| Precision | Use float64 for norms, dot products, residuals, likelihoods, and GPU accumulators. |

## Phase 0 architecture map

### Library format

`madi.library.LibraryEntry` stores `(kio, rho, V, vector)` at
`madi/library.py:61-67`. A v2 vector is a flat, row-major
`(delta, Delta, b)` block: Delta-pair-major then b-major; see
`madi/signal.py:46-57` and `madi/signal.py:179-182`.

V2 `.npz` files persist parameter arrays, `vectors`, `pair_deltas`,
`pair_Deltas`, `b_values`, `n_b`, and `h_ms` (`madi/library.py:250-280`). The
loader returns a Python list of entries (`madi/library.py:283-303`). Every
fitter currently converts the filtered list into a dense selected matrix through
`_build_candidate_lib_matrix()` (`madi/library.py:451-487`).

The inspected universal library is `(40645, 31125)` float64, with 1,245 stored
`(delta, Delta)` pairs and 25 b-values including b=0. It has 55 kio values and
739 surviving `(rho,V)` pairs after volume-fraction filtering, so it is not a
regular rectangular 3-D tensor. The library builder also supports explicit
triplets (`madi/library.py:95-104`) and full products (`:224-243`); analysis
must support irregular/scattered libraries.

The default v2 b grid includes zero (`madi/config.py:96-107`). As
`G_from_b(0)` is zero (`madi/signal.py:32-39`), b=0 library signal entries are
exactly one. Legacy/slim libraries can omit b=0 and must be rejected for a
marginal acquisition requiring it.

### Current S0 path

`scripts/fit_data.py:430-747` loads DWI, detects b=0 volumes, averages them
inside each input Delta group (`:688-699`), and may average the S0 images across
groups (`:701-717`). It then forms `fit_triples` from positive-b shells only
(`:645-653`) and emits normalized positive-b `measured` plus unnormalized
positive-b `raw_signal` (`:720-745`). Thus b=0 information is consumed and
dropped before fitting.

Fixed-S0 MAP is at `madi/library.py:494-541`; free-S0 VARPRO MAP at
`:548-612`; Bayes fixed/free-S0 at `madi/fitters.py:110-236`; AMICO at
`:392-508`. They share candidate filtering/subsetting only. AMICO is a mixture
regression, not a candidate-wise likelihood, so it cannot receive a
candidate-specific marginal log-norm penalty.

GPU MAP/Bayes stream candidates per voxel without a voxel-by-library matrix
(`madi/fitters_gpu.py:8-16`). CPU MAP/Bayes materialize the matrix. GPU AMICO
alone chunks voxels (`:550-581`). Bayes already has stable row-wise max
subtraction on CPU (`madi/fitters.py:203-223`) and an equivalent minimum-score
shift on GPU (`madi/fitters_gpu.py:173-207`); there is no shared logsumexp
utility.

### Configuration and testing

Fitting options are argparse flags in `scripts/fit_data.py`; `SimConfig` is
for simulation and library construction only. Backward compatibility therefore
means additive flags and preserved legacy defaults.

There is no automated fitting test suite. `analysis/test_drms_curves.py` is a
pytest-capable physics test. `analysis/verify_gpu_fitters.py` and
`scripts/_sanity_fitters.py` are manual fitter checks but are stale after the
universal-library migration, as documented in
`docs/universal_library.md:171-183`. The controlled comparison prototype in
`scripts/edema_summary/controlled_s0_comparison.py` may be extended for
experiments, but is not production code.

## CLI and compatibility contract

### S0 selector

Add:

```
--s0-mode {normalize,fit,marginal}
```

Do not give the parser a naive `normalize` default, because existing configs
may pass `--fit-s0`. Resolve the mode as follows:

1. if `--s0-mode` is omitted, preserve legacy behavior: `fit` when `--fit-s0`
   is present, otherwise `normalize`;
2. if `--s0-mode` is given, use it and reject contradictory `--fit-s0`;
3. retain `--fit-s0` as a documented compatibility alias for `--s0-mode fit`.

Legacy normalize and fit code paths must remain numerically unchanged. Reject
`--log-space` in marginal mode because the linear amplitude model is required.

### Coupling and b=0 precision

Add marginal-only controls:

```
--s0-coupling {per-delta,shared}       # default: per-delta
--n0-eff DELTA,DELTA=value [...]       # optional per-group override
```

The timing key must contain `(delta_ms, Delta_ms)` because Delta alone is not
unique in the universal library. Omitted groups use their observed b=0 count.
The effective precision is:

```
n0_eff = sigma_raw**2 / Var(mean_b0).
```

`per-delta` gives each acquisition its own amplitude. `shared` pools all
groups. In marginal mode `--avg-s0` is an alias for `--s0-coupling shared`; an
explicit contradictory coupling should error. Legacy `--avg-s0` semantics are
untouched outside marginal mode.

### Raw sigma versus `sigma_m`

The marginal Gaussian likelihood needs a raw per-volume noise standard
deviation `sigma_raw`. Resolve it as:

1. explicit `--noise-sigma`;
2. the background estimate made when `--rician-correct` is enabled;
3. otherwise reject marginal MAP/Bayes with a clear request for raw sigma.

Do not fall back to `DEFAULT_SIGMA_M`. The current conversion

```
sigma_m ~= sigma_raw / (S0_median * sqrt(mean_n_dir))
```

in `madi/fitters.py:90-103` shows that `sigma_m` is a normalized residual
temperature and cannot serve as a raw likelihood standard deviation.

Rician correction is the expected workflow because it already estimates raw
sigma. Document that the Gaussian likelihood after second-moment correction is
a high-SNR approximation: clipping/correction makes noise heteroscedastic. A
raw Rician/noncentral-chi likelihood is not part of this implementation.

## Marginal-S0 mathematics

### One acquisition group

For positive-b raw vector `m`, normalized library curve `s_i`, mean b=0
measurement `m0`, raw sigma `sigma`, and effective b=0 precision `n0`:

```
A_i = ||s_i||^2_(b>0) + n0
B_i = <m, s_i>_(b>0) + n0 * m0
C   = ||m||^2_(b>0) + n0 * m0^2

a_tilde_i = B_i / A_i
R_i^2     = C - B_i^2 / A_i

-2 log p(data | i) = R_i^2 / sigma^2 + log(A_i) + constant independent of i.
```

This is Gaussian amplitude marginalization with
`a ~ Normal(m0, sigma^2/n0)`. The log(A) term is required; profile fitting
alone omits it.

### Full-vector shortcut and fractional precision

Because b=0 library signal is one, ordinary VARPRO on a full vector gives the
same `A`, `B`, and `R^2`. Integer n0 can be represented by repeated b=0 rows.
For fractional `n0_eff`, use one equivalent weighted pseudo-row:

```
m_b0 = sqrt(n0_eff) * mean_b0
s_b0 = sqrt(n0_eff).
```

For MAP compare the dimensionless score `R_i^2/sigma^2 + log(A_i)`. Do not add
an unscaled log norm to raw residual units.

### Per-Delta default and shared mode

For acquisition group `g`, compute `A_ig`, `B_ig`, and `R_ig^2` separately.
The default candidate score is:

```
score_i = sum_g [ R_ig^2 / sigma_g^2 + log(A_ig) ].
```

The conditional group amplitude is `B_ig/A_ig`, with conditional variance
`sigma_g^2/A_ig`. Bayes output must include both conditional and
candidate-selection uncertainty:

```
E[a_g]   = sum_i w_i * a_tilde_ig
Var[a_g] = sum_i w_i * (sigma_g^2/A_ig + a_tilde_ig^2) - E[a_g]^2.
```

Write timing-qualified maps such as `s0_mean_delta6_D15.nii.gz` and
`s0_std_delta6_D15.nii.gz`; preserve legacy `s0_fit_map.nii.gz` for fit mode.

Shared mode pools groups before marginalizing:

```
A_i = sum_g A_ig
B_i = sum_g B_ig
C   = sum_g C_g
R_i^2 = C - B_i^2/A_i.
```

Then use one amplitude and one log(A). If scans have different raw sigma,
whiten their rows before applying this shorthand.

### Shell-mean precision choice

Current positive-b rows are direction-averaged shell means. Before coding,
choose and document one convention:

1. **Initial/simple:** treat all shell means as equal-variance rows, while
   n0_eff controls only b=0 precision. This most directly follows the requested
   formula.
2. **Replicate-aware:** scale each shell mean and its library row by
   `sqrt(n_directions)` and use the `sqrt(n0_eff)` b=0 pseudo-row. Then every
   transformed row has raw per-volume sigma.

The second option is statistically preferable if direction counts differ. Do
not silently combine the two models. If replicate-aware precision is selected,
report both unweighted q and likelihood-weighted q diagnostics.

## Optional marginalized noise scale (Bayes only)

Add:

```
--noise-model {fixed_sigma,marginal_sigma}
```

with `fixed_sigma` as default. This is Bayes-only; AMICO remains unchanged and
MAP should reject or clearly ignore the option according to final CLI policy.
For N rows under the selected row/weight convention:

```
log_w_i = -0.5 * log(A_i)
          -0.5 * (N - 1) * log(max(R_i^2, residual_floor)).
```

Subtract a per-voxel maximum before exponentiation. Clamp cancellation-level
negative residuals to a positive floor, count clamps, and warn if frequent.
The required test is scale invariance: scaling all raw measurements/residuals
must not alter Student-t weight ratios.

## Phase 1 q diagnostic

For each selected protocol/group compute:

```
q_i  = sum_(b>0 rows) s_i^2
w0_i = n0_eff / (n0_eff + q_i).
```

Report q min/median/max plus histogram, and w0 min/median/max plus median.
It is acquisition-specific, so `library_summary()` cannot compute it alone:
implement the reusable calculation by `_build_candidate_lib_matrix()` in
`madi/library.py`, then invoke it in `scripts/fit_data.py` after b=0 layout and
`fit_triples` are known. Report per Delta in default coupling and pooled
precision in shared mode. Save the summary in fit metadata.

## Implementation sequence

1. **Re-audit:** preserve unrelated dirty-worktree changes; verify active
   library metadata and b=0 availability; update stale manual helpers only as
   needed to establish a baseline.
2. **Observation representation:** extend `load_dwi_and_average()` only on the
   opt-in marginal path. Preserve legacy returns exactly. Add a narrow internal
   group representation containing raw shell means/triples, b=0 mean/count,
   n0_eff, direction counts, and raw sigma. Do not create a second loader.
3. **Diagnostic:** add/report q and w0 before modifying the fitter; record
   actual values when a real dataset is supplied.
4. **Protocol cache:** create a small selected-candidate container or helper
   holding signals, parameter arrays, group row layout, positive/full norms,
   and log norms. Cache it once per protocol/filter and transfer it with the
   selected matrix to GPU. Do not introduce a global persistent GPU library
   object without a separately approved lifetime refactor.
5. **CPU reference:** add dedicated marginal MAP and Bayes paths in float64;
   do not modify existing normalize/fit arithmetic. Implement group amplitudes,
   amplitude moments, stable weights, invalid-row handling, and Student-t
   Bayes after fixed-sigma marginal Bayes is verified.
6. **GPU parity:** add float64 streaming MAP/Bayes kernels only after CPU
   tests pass. Use the established two-pass pattern: best/max-score pass,
   then selected/weighted accumulation pass. Never form a GPU voxel-by-library
   matrix.
7. **CLI/docs/output:** wire settings, warnings, map names, and metadata;
   document formulas and assumptions in `docs/fitting_methods.md`.

## Fisher/CRLB implementation plan

### Migrate the existing module

`madi/identifiability.py` already computes irregular-axis finite differences
and a 3x3 FIM, but is on the pre-v2 `(Delta,b)` interface. It imports removed
`_pair_indices`; `scripts/analyze_identifiability.py` also reads obsolete
`meta["deltas"]`. Repair and extend these existing files rather than creating a
parallel analyzer. Use v2 `(delta,Delta,b)` selection with the same semantics as
the fitter (`madi.library._grid_columns()` or one shared helper).

### Derivatives and coordinates

Use:

```
theta = (kio, v_in, V)
v_in  = (rho/1e9) * (V*1e3) = rho*V*1e-6.
```

This is the codebase's volume-fraction convention (`madi/library.py:85-88`),
not an unscaled raw `rho*V`. The universal library is irregular, so estimate
derivatives with local linear regression over k nearest neighbours in
standardized `(kio,v_in,V)` space. The existing axis finite differences should
remain available as a derivative-noise comparison, but do not pretend they are
a regular-grid solution.

Report derivative reliability, at minimum by changing local-neighbour count or
comparing against available central/one-sided derivatives. Shared random-number
seeding was deliberately added for low-noise rho/V differences
(`madi/library.py:109-114`, `madi/walker_gpu.py:35-40`).

### Fisher outputs

With b>0 tissue derivative matrix `S`, amplitude `a`, noise sigma consistent
with the selected convention, positive-b `q=||s||^2`, and b=0 precision n0:

```
J_full = (1/sigma^2) * [ a^2 S^T S    a S^T s ]
                         [ a s^T S      q+n0  ].
```

Report per library entry:

1. CRLBs for unknown amplitude (`lambda=0`), known amplitude (invert tissue
   block), and finite b=0 precision (`lambda=n0_eff`);
2. leverage `eta_p = <s,S_p>^2 / (||s||^2 ||S_p||^2)`;
3. small-eigenvalue vector and eigenvalue ratio of the `(v_in,V)` block;
4. 3x3 tissue-block condition number;
5. derivative-reliability metrics.

Transform covariance back to `(kio,rho,V)` through the Jacobian of
`rho = 1e6*v_in/V`; document that covariance transforms as `J Cov J^T`.
Handle singular matrices explicitly and retain/extend current PSD checks.

### Multi-Delta and map projection

The v2 library stores many Delta values. The analysis CLI should accept explicit
`(delta,Delta,b)` selections and compare an additional stored Delta without new
simulation. Reject unavailable rows; do not interpolate or fabricate them.

Reuse the existing `save_map()` behavior from `scripts/fit_data.py:754-759`
for CRLB projection, moving/exposing it as a shared helper only if necessary.
Do not add another NIfTI writer. Prefer fitter assignment-index projection when
available; clearly label a nearest-parameter fallback.

## Required tests

Add automated, small in-memory fitting fixtures; do not require a GPU, external
NIfTI data, or the 10 GB library. Extend the controlled comparison script for
larger experiments.

### Marginal fitting

1. n0 -> 0 reproduces free-S0 VARPRO on positive-b rows.
2. n0 -> large approaches fixed-amplitude/normalization behavior.
3. Literal formula and full-vector shortcut give identical MAP winners and
   Bayes weights.
4. Per-Delta amplitudes recover independent synthetic gains; shared mode pools
   them.
5. Fractional n0 pseudo-row equals the direct formula.
6. Student-t weights are scale invariant.
7. Cancellation clamps remain finite and are counted; ordinary cases do not
   clamp often.
8. CPU/GPU marginal MAP and fixed-sigma Bayes agree to float64 tolerance.
9. Synthetic round trips compare normalize, fit, and marginal recovery at
   multiple SNRs; report unexpected ordering honestly rather than forcing it.

### CRLB

1. Analytic linear fixtures verify local derivatives, FIM, and Jacobian
   covariance transform.
2. Known/unknown/finite-amplitude CRLB regimes have expected ordering.
3. Leverage matches direct scalar calculations.
4. Additional selected rows do not reduce Fisher information under one fixed
   noise model.
5. Simulated fitting covariance agrees with CRLB in order of magnitude and,
   especially, degeneracy direction.
6. A v2 metadata/column-selection test prevents regression to the current
   stale `(Delta,b)` interface.

## Expected files

| File | Work |
|---|---|
| `scripts/fit_data.py` | Marginal observations, CLI resolution, diagnostics, maps, metadata. |
| `madi/library.py` | Selected candidate layout/norm/cache and q diagnostics. |
| `madi/fitters.py` | CPU marginal MAP/Bayes, amplitudes, Student-t weights. |
| `madi/fitters_gpu.py` | Float64 streaming marginal MAP/Bayes parity kernels. |
| `madi/identifiability.py` | V2 migration, local derivatives, v_in Fisher/CRLB. |
| `scripts/analyze_identifiability.py` | V2 CLI, output tables/plots, multi-Delta comparison, projection. |
| `docs/fitting_methods.md` | S0/noise assumptions and derivations. |
| `docs/identifiability.md` | Fisher parameterization, derivative reliability, output interpretation. |
| fitting tests and controlled comparison | Regression, invariant, round-trip, and empirical checks. |

## Guardrails

* Preserve legacy normalize and fit behavior exactly unless a new mode is
  selected.
* Do not apply marginal candidate penalties to AMICO.
* Do not use log-space fitting with linear amplitude marginalization.
* Do not assume a regular `(kio,rho,V)` grid.
* Keep b=0 row counting/weighting consistent in likelihoods, Student-t N,
  diagnostics, and Fisher matrices.
* Keep raw `noise_sigma`, normalized `sigma_m`, and n0_eff visibly distinct in
  code, CLI help, metadata, and documentation.
* Ask before broad library API or persistent-GPU-cache refactors.
* Report contrary empirical results honestly, including leverage, derivative
  reliability, residual clamping, and synthetic recovery behavior.
