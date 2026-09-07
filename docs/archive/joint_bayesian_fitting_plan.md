# Joint Bayesian MADI fitting plan

## Decision

The intended production Bayesian method should retain the library as a
discrete approximation to the parameter space, but fit the unnormalised
measurements jointly with a positive, voxel-specific amplitude S0.  Every
measured b=0 observation must enter the likelihood.  This is referred to
below as `bayes-joint`.

This is not merely another form of `--fit-s0`.  It is a different statistical
model.  It removes the noisy-ratio construction, gives S0 a posterior rather
than treating a candidate-specific least-squares estimate as known, and
accounts for the number of averages in each b=0 and diffusion shell.

The current `bayes` and `bayes --fit-s0` modes should remain available and
clearly labelled as legacy comparison methods until `bayes-joint` has passed
the validation plan below.

## Current methods and their limitations

Let `r_i = (r_i1, ..., r_ip)` be the library decay curve for candidate `i` at
the p fitted `(delta, Delta, b)` features.  `r_ij` is a dimensionless
prediction for `S(b_j) / S0`.

### Fixed-S0 Bayes

The current fixed-S0 code constructs shell means `m_j` and the mean of n0
b=0 measurements `s0_bar`, then fits ratios

```
x_j = m_j / s0_bar.
w_i proportional to exp(-sum_j (x_j - r_ij)^2 / (2 sigma_m^2)).
```

It treats `s0_bar` as exact and uses one independent residual variance for
all shells.  This is convenient, but the shared noisy denominator induces
correlated errors.  To first order, assuming independent additive raw-signal
noise with variance `sigma^2`,

```
Var(x_j) approximately sigma^2/(n_j S0^2)
                      + r_ij^2 sigma^2/(n0 S0^2)
Cov(x_j, x_k) approximately r_ij r_ik sigma^2/(n0 S0^2), j != k.
```

The current likelihood ignores both the unequal `n_j` and the off-diagonal
covariance.  Its `sigma_m` is therefore a posterior-temperature parameter,
not a fully specified acquisition-noise likelihood.

### `bayes --fit-s0`

The current free-S0 mode uses only positive-b shell means `m` and profiles
an amplitude separately for each candidate:

```
S0_i* = (m dot r_i) / (r_i dot r_i)
RSS_i = ||m||^2 - (m dot r_i)^2 / (r_i dot r_i).
```

It then weights candidates using `RSS_i / (S0_i*)^2` so that the numerical
scale resembles the ratio fit.  This makes the code stable, but it is not the
likelihood of the acquired data:

* the b=0 observations are excluded after preprocessing;
* S0 is optimized, not integrated over or constrained by b=0 data;
* dividing by a candidate-specific fitted S0 changes the error model and
  omits the corresponding normalization term;
* an additional free parameter necessarily reduces residuals, so a common
  `sigma_m` has no comparable meaning between this method and fixed-S0.

Matching `n_eff` is useful for controlled sensitivity figures, but it only
matches posterior concentration.  It cannot establish that either likelihood
is physically calibrated.

## Proposed likelihood

For one voxel and one acquisition group, retain the raw b=0 measurements
`y_0k`, k = 1,...,n0, and the mean positive-b measurement `y_j` for each
shell j = 1,...,p.  The library candidate is `i`, and its unknown positive
amplitude is `a` (the latent S0).

The initial implementation will use the high-SNR Gaussian approximation on
the *raw magnitude data after no second-moment correction*:

```
y_0k | a                 ~ Normal(a,                    sigma_0k^2)
y_j  | a, i              ~ Normal(a r_ij, sigma_j^2/n_j + tau_j^2).
```

`sigma_j` is the raw per-volume noise standard deviation for the acquisition
group.  `n_j` is the number of directions averaged into that shell.  The
optional `tau_j` is a declared model-discrepancy/overdispersion term.  It is
zero in the first validation implementation.  It must not be silently folded
into `sigma_j`: thermal noise and mismatch between a powder-average library
curve and an in-vivo shell mean are distinct uncertainties.

For data with several DWI input groups (for example multiple Delta scans),
use a separate amplitude `a_g` per group:

```
y_0gk | a_g              ~ Normal(a_g, sigma_g0k^2)
y_gj  | a_g, i           ~ Normal(a_g r_igj, sigma_gj^2/n_gj + tau_gj^2).
```

This avoids the current ad hoc `--avg-s0` decision.  A later hierarchical
version can partially pool `a_g` across groups after scanner gain, motion,
and true T1/TE effects have been assessed.  It should not force a common S0
by default.

### Conjugate Gaussian form for the first production version

For a single group, stack all observations in `y` and define

```
g_i = (1, ..., 1, r_i1, ..., r_ip)^T
Sigma = diag(sigma_01^2, ..., sigma_0n0^2,
             sigma_1^2/n_1 + tau_1^2, ..., sigma_p^2/n_p + tau_p^2).
```

The default amplitude prior should be the common improper positive-flat prior,
`p(a) proportional to 1(a > 0)`.  It is valid for comparing candidates
because the same multiplicative constant applies to every candidate and the
conditional posterior is proper (`A_i > 0`).  Do not derive a voxel-specific
amplitude prior from the b=0 values and then include those same values in the
likelihood; that would count the b=0 evidence twice.

A proper sensitivity prior, such as a positive log-normal derived from an
independent calibration scan or a cross-voxel empirical-Bayes fit, can be
offered later.  Its source and hyperparameters must be recorded.  It is not
the default.

For candidate i, define

```
A_i = g_i^T Sigma^-1 g_i
B_i = g_i^T Sigma^-1 y
C   = y^T Sigma^-1 y.
```

Then

```
a | y, i ~ Normal(B_i/A_i, 1/A_i), truncated to a > 0
log p(y | i) = constant - 0.5 log(A_i) - 0.5 (C - B_i^2/A_i)
               + log Phi(B_i / sqrt(A_i)).
```

The final `log Phi` term is the positive-amplitude correction.  At normal
brain SNR it is nearly zero, but including it makes the model well-defined at
the boundary.  A proper Gaussian prior `Normal(mu_a, s_a^2)` is an optional
extension obtained by adding `1/s_a^2`, `mu_a/s_a^2`, and `mu_a^2/s_a^2` to
`A_i`, `B_i`, and `C`, respectively.

The discrete posterior is

```
log p(i | y) = log p(y | i) + log pi_i - logsumexp_l[log p(y | l) + log pi_l].
```

Posterior summaries follow directly:

```
E[theta | y] = sum_i w_i theta_i
Var(theta | y) = sum_i w_i theta_i^2 - E[theta | y]^2
E[a | y] = sum_i w_i E[a | y, i]
Var(a | y) = sum_i w_i [Var(a | y, i) + E[a | y, i]^2] - E[a | y]^2.
```

The last equation is important: the S0 variance includes uncertainty within
each candidate and uncertainty caused by not knowing the tissue candidate.

### Relationship to existing methods

* Fixed-S0 Bayes is approximately the limit `s_a^2 -> 0`, `mu_a = s0_bar`,
  followed by dividing the data by `s0_bar`.  The proposed method does not
  need that division and preserves the b=0 uncertainty.
* Current `--fit-s0` resembles a positive-b-only, flat-prior *profile*
  likelihood.  It is not the marginal likelihood above because it omits b=0
  data and the `-0.5 log(A_i)` integration term.
* The preliminary `joint_s0_bayes` in
  `scripts/edema_summary/controlled_s0_comparison.py` already implements the
  weighted least-squares profile part using b=0 plus shell means.  It was
  intentionally a controlled-comparison prototype.  It needs the amplitude
  prior, marginalization term, production data interface, diagnostics,
  chunking, GPU kernel, and validation before it can be called a production
  fitter.

## Magnitude-noise model: staged, not ignored

The Gaussian model above is an appropriate first production implementation
only when the b=0 and fitted-shell SNR are sufficiently high.  It should use
raw magnitude data, not `sqrt(M^2 - 2 sigma^2)`, because that correction
creates clipped, heteroscedastic values that no longer obey a Gaussian
measurement model.

The more principled eventual observation model is Rician magnitude noise:

```
p(M | nu, sigma) = (M/sigma^2) exp(-(M^2 + nu^2)/(2 sigma^2)) I0(M nu/sigma^2)
nu_0k = a
nu_jd = a r_ij.
```

Here `I0` must be evaluated through a numerically stable scaled Bessel
function or a stable `log I0` implementation.  This needs one-dimensional
positive integration or Laplace approximation over `a` for every candidate;
the Gaussian conjugacy is lost.  It is feasible because the latent dimension
is one, but it should follow, not precede, validation of the Gaussian joint
method.

There is an important caveat: a MADI library predicts a direction-averaged
signal, whereas individual DWI directions in white matter may differ because
of anisotropy.  Applying an independent Rician likelihood to every direction
as if they had identical predicted signal would overstate evidence.  Until an
orientation-aware forward model exists, retain shell means and their declared
variance inflation, or estimate the likelihood of the shell mean by Monte
Carlo.  Do not call a raw-direction Rician model more exact if it violates
the forward model.

If reconstruction uses multiple coils with sum-of-squares magnitude rather
than a Rician noise distribution, replace Rician with the appropriate
noncentral-chi likelihood.  Coil/reconstruction metadata are required to
choose this correctly.

## Candidate prior is part of the model

Equal weights per library row are an implicit prior over the *library
sampling scheme*, not necessarily over tissue parameters.  The universal
library may be nonuniform in `(k_io, rho, V)` or in derived volume fraction.
Increasing candidate density in one region should not increase its posterior
probability without an explicit prior decision.

The initial `bayes-joint` prior should therefore support:

1. `grid-cell` (default): each candidate receives a quadrature weight
   proportional to its local cell volume in the chosen transformed parameter
   coordinates; this approximates a declared continuous prior.
2. `uniform-linear`: uniform on the retained physical grid volume.
3. `uniform-log`: uniform in log(k_io), log(rho), and log(V), with documented
   finite bounds.
4. `legacy-row`: equal mass per retained library row, only for reproduction
   of existing experiments.

The final prior and every filtering bound (`vi_min`, `vi_max`, `rho_max`) must
be written to `fit_metadata.json`.  Any tissue-specific empirical prior is a
later research model and must be trained on independent data or validated
with subject-level cross-validation.

## Noise, model discrepancy, and calibration

`n_eff` remains useful as a descriptive posterior-concentration diagnostic.
It must not select the likelihood scale for a primary scientific analysis.
The initial joint model should estimate or accept raw `sigma_g` from
background data per acquisition group and use the known replicate counts.

Before a nonzero `tau_gj` is enabled by default, estimate it from an
independent repeat-scan or test-retest dataset.  A possible relative
discrepancy model is

```
Var(y_gj | a, i) = sigma_g^2/n_gj + (eta_gj a r_igj)^2,
```

where `eta_gj` is fixed from validation or given a conservative prior.  This
is no longer conjugate because the variance depends on `a`, so it belongs in
the Rician/numerical-integration stage.

Validation should assess posterior calibration directly: simulate known
library candidates under measured noise and b=0 drift, then check 50%, 80%,
and 95% credible-interval coverage.  `n_eff` matching alone is not a
calibration criterion.

## Implementation plan

### 1. Define the production interface and preserve all observations

Modify `load_dwi_and_average()` in `scripts/fit_data.py` to return a
structured `BayesObservations` object only when `--method bayes-joint` is
selected.  It must contain, per input group:

* b=0 observations before ratio normalization: `(n_vox, n_b0)`;
* retained shell means: `(n_vox, n_shell)`;
* `n_b0`, `n_dir` per shell, `(delta, Delta, b)` feature mapping;
* raw noise sigma per input group, acquisition/reconstruction provenance,
  and whether any preprocessing has altered the raw magnitude values.

Keep the existing `measured`, `raw`, and legacy modes byte-for-byte
unchanged.  `bayes-joint` should reject `--fit-s0`, `--log-space`,
`--target-n-eff`, and `--rician-correct` in its first version, with clear
messages.  It uses raw magnitude values and handles the chosen likelihood
internally.

### 2. Implement a CPU reference fitter first

Add `joint_bayes_fit()` in `madi/fitters.py`.  It should:

* reuse `_build_candidate_lib_matrix()` and all present candidate filters;
* calculate `A_i`, `B_i`, and marginal log weights in float64;
* process voxels in configurable chunks so CPU memory is
  `O(chunk_voxels * n_library)` rather than whole-brain by library;
* use `logsumexp` stabilization and explicit invalid-voxel handling;
* return posterior means, standard deviations, quantiles, S0 mean/std,
  log evidence up to common constants, posterior predictive residuals, and
  `n_eff`.

Implement the truncated-normal formulas using `scipy.special.log_ndtr` on
CPU.  Test the untruncated version first against a direct numerical integral
for a small candidate set, then add positivity correction.

### 3. Make the command-line contract explicit

Extend `--method` to include `bayes-joint`.  Add, at minimum:

```
--joint-s0-prior {flat-positive,lognormal}  # default: flat-positive
--joint-s0-prior-config PATH                # required for lognormal
--joint-prior {grid-cell,uniform-linear,uniform-log,legacy-row}
--joint-model-discrepancy VALUE_OR_CONFIG
--bayes-chunk-voxels N
--joint-likelihood {gaussian,rician}       # rician initially unavailable
```

Save the resolved variance per shell, amplitude-prior settings, candidate
prior, feature replicate counts, invalid-voxel count, and software version
in `fit_metadata.json`.  Name output maps so there is no ambiguity with
legacy Bayes: for example `s0_joint_mean.nii.gz`, `s0_joint_std.nii.gz`,
`log_evidence.nii.gz`, and `ppc_residual.nii.gz`.

### 4. Add tests before GPU work

Add unit tests that verify:

* a one-candidate analytic posterior agrees with closed-form calculations;
* `s_a -> 0` approaches a fixed-amplitude weighted likelihood;
* adding b=0 replicates reduces S0 posterior variance;
* candidate posterior weights agree with brute-force numerical integration;
* CPU results are invariant to chunk size;
* nonuniform duplicate library rows do not change a `grid-cell` posterior;
* invalid/noise-only voxels produce defined zero-support outputs, not NaNs.

Extend `controlled_s0_comparison.py` rather than treating it as the
production test suite.  Simulations must include nominal noise, b=0 bias,
b=0 drift, shell-dependent noise, library misspecification, and a
non-library ground truth.  Report bias, RMSE, coverage, interval width,
posterior-predictive checks, and sensitivity to priors.

### 5. Validate on available measured data

Use only validated reference ROIs (grey matter, white matter, contralateral
reference), not the incomplete tumour/edema masks.  Run fixed Bayes,
legacy fits0, and joint Bayes at the same library/filter settings.  Compare:

* S0 posterior versus observed b=0 means and leave-one-b0-out predictions;
* parameter shifts versus b=0 replicate variability;
* posterior predictive residuals by b shell and ROI;
* posterior interval width and empirical repeat/within-scan coverage where
  independent repeats exist;
* sensitivity to noise sigma, amplitude prior width, and candidate prior.

Pre-register the primary comparison: joint Bayes is favored only if it has
better simulated coverage and no material degradation in held-out b=0/shell
prediction.  Visual plausibility alone is not sufficient.

### 6. Add the GPU implementation after CPU equivalence

The existing CUDA Bayes architecture already streams candidates per voxel,
which is appropriate for the 40,645-entry universal library.  Add a separate
kernel that accumulates `A_i`, `B_i`, and the marginal log weight for each
candidate.  Two passes per voxel are enough: find the maximum log weight,
then accumulate weighted parameter and S0 moments.  It must use float64 and
be tested against the CPU reference on small crops before whole-brain use.

Do not materialize an `(n_voxel, n_library)` matrix on GPU or CPU.  Add a
small-crop benchmark and a full-brain memory/time benchmark to the release
criteria.

### 7. Add the Rician/noncentral-chi extension only after stage 1 passes

Implement a one-dimensional positive quadrature or Laplace approximation for
each `(voxel, candidate)` amplitude.  Use stable log-Bessel arithmetic and
the acquisition's actual noise distribution.  Validate it against simulated
low-SNR magnitude data, including the current second-moment correction as a
comparator.  This extension should be gated behind an explicit likelihood
flag until it is validated on the actual reconstruction type.

## Data that would materially improve the model

* Noise-only scans or reconstruction/g-factor information per acquisition,
  plus coil-combination metadata to establish Rician versus noncentral-chi.
* Repeat DWI acquisitions and repeated b=0 images distributed through each
  scan to quantify drift and validate uncertainty coverage.
* Gradient-direction information and an orientation-aware forward model, or
  an acquisition designed for reliable powder averaging, to separate thermal
  noise from directional biological variation.
* Phantom data and independent tissue/parameter references for checking
  library model discrepancy and parameter identifiability.
* Validated tissue segmentations for biological ROI analyses; incomplete
  tumor/edema masks should remain excluded.
