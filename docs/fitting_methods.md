# MADI fitting methods

`scripts/fit_data.py --fit` estimates the three MADI parameters per voxel by
comparing the measured multi-Δ, multi-b signal against a precomputed library
of simulated signals. The parameters are:

- **k_io** — intracellular exchange / membrane permeability rate [s⁻¹]
- **ρ** — cell density [cells/µL]
- **V** — mean cell volume [pL]

There are now three ways to turn the measured signal into parameter maps,
selected with `--method`:

| method  | flag           | output style                          |
|---------|----------------|---------------------------------------|
| `map`   | `--method map`   | single best library entry (default)   |
| `bayes` | `--method bayes` | posterior mean ± std over the library |
| `amico` | `--method amico` | elastic-net NNLS mixture, mean ± std  |

All three share the same candidate filtering (`--vi-min`, `--vi-max`,
`--rho-max`), the same `(Δ, b)` subsetting, the same optional `--log_space`
transform (except AMICO, see below), and the same free-S₀ option (`--fit-s0`).

**`--log_space` applies only to fixed-S₀ MAP and Bayes.** It transforms both
the measured signal and the library curves to `log(clip(S/S₀, 1e-3, 1.0))`
before the residual is computed, rebalancing the least-squares metric so the
low-signal high-b tail (where the tissue-discriminating information lives) is
not drowned out by the high-signal low-b points. It is silently inapplicable
to three cases: (1) **AMICO** — the regression `||Dx − m||²` is linear in the
signal, so a log transform is meaningless (`log` of a mixture ≠ mixture of
`log`s); this is warned and ignored. (2) **MAP with `--fit-s0`** — the free-S₀
matcher (`match_voxels_batch_fits0`) has no `log_space` parameter, because its
analytic `S₀* = (m·sᵢ)/(sᵢ·sᵢ)` projection assumes a linear signal model that
the log transform breaks. (3) **Bayes with `--fit-s0`** — `bayes_fit` applies
the log transform only in its fixed-S₀ branch; the free-S₀ branch operates on
raw signal and ignores `--log_space`.

---

## Method 1 — MAP (`--method map`, default)

For each voxel with measured signal **m** and library entries with signals
**sᵢ**, pick the entry with the smallest residual:

```
i* = argmin_i ||m − sᵢ||²          (fixed S₀)
```

and report its (k_io, ρ, V). With `--fit-s0`, S₀ is a free per-voxel scalar
solved analytically per entry (`S0* = (m·sᵢ)/(sᵢ·sᵢ)`), and the residual is
`||m||² − (m·sᵢ)²/(sᵢ·sᵢ)`.

**Intuition.** This is a maximum-a-posteriori estimate under a flat prior: the
answer is whichever simulated tissue best reproduces the data. It is fast,
deterministic, and gives no uncertainty — a single grid point, so the maps
inherit the discreteness of the library grid and are sensitive to noise near
decision boundaries. This is the original, unchanged behaviour.

**Outputs:** `kio_map.nii.gz`, `rho_map.nii.gz`, `V_map.nii.gz`,
`residual_map.nii.gz` (+ `s0_fit_map.nii.gz`, `s0_fit_over_measured.nii.gz`
with `--fit-s0`).

---

## Method 2 — Bayesian posterior mean (`--method bayes`)

Instead of choosing one entry, treat every library entry as a hypothesis and
weight it by a Gaussian likelihood:

```
wᵢ ∝ exp( −||m − sᵢ||² / (2 σ_m²) ),     Σ wᵢ = 1
```

then report, for each parameter θ ∈ {k_io, ρ, V},

```
⟨θ⟩  = Σ wᵢ θᵢ
σ_θ  = sqrt( Σ wᵢ θᵢ² − ⟨θ⟩² )
```

Weights are computed in log-space with a per-voxel `max(log w)` subtraction
before exponentiating, for numerical stability.

**With `--fit-s0`** each candidate gets its own amplitude `a`, and the weight
is the **profile likelihood** over that amplitude, with the noise scaled by the
voxel's *measured* S₀:

```
rᵢ = min_a ||m − a·sᵢ||²  =  ( ||M||² − (M·sᵢ)²/(sᵢ·sᵢ) ) / S0_measured²
wᵢ ∝ exp( −rᵢ / (2 σ_m²) ),     m = M / S0_measured
```

Here `M` is the raw (un-normalized) shell signal. Candidates whose best
amplitude is negative get zero weight. The reported `s0_fit` is the posterior
mean of `a·S0_measured`, in raw units. Voxels with no usable measured S₀ (≤ 0
or non-finite) get the zero-weight "no support" result. The noise is fixed in
raw units, so every candidate is judged against the same scale, and σ_m means
the same thing as in the fixed-S₀ branch.

*Earlier versions* divided each candidate's residual by **its own** fitted
S₀ᵢ² instead of by the measured S₀². That is `rᵢ/aᵢ²`, so each candidate got
a noise level proportional to the amplitude it needed, and low-signal
candidates (large cells, low ρ, which need a large `a`) were favoured. On
CDMD sub-001 (arm A8) this pushed both cortical GM and WM to ρ ≈ 0.03–0.06·10⁶ cells/µL and
V ≈ 33–39 pL, with posterior-mean S₀ 13–33× the measured b=0. Profile
weighting on the same voxels gives GM k_io ≈ 60 s⁻¹, ρ ≈ 1.0·10⁶, V ≈ 0.8 pL,
a ≈ 1.12. `bayes --fit-s0` outputs made before this fix should be
regenerated. The MAP free-S₀ matcher was never affected, because the argmin
of the raw residual does not depend on the noise scale.

The profile likelihood replaces the amplitude by its best value; it does not
integrate it out. With a flat prior on `a`, integrating would add a factor
`1/||sᵢ||` per candidate. That factor is not applied.

**σ_m** (residual noise std on the *normalized* signal, **regardless of
whether `--fit-s0` is set**) is chosen as:

1. `--target-n-eff FLOAT` if given — see **σ_m calibration** below (source
   logged as `target-n-eff`; overrides `--sigma-m` with a warning if both are
   given);
2. else `--sigma-m FLOAT` if given (source logged as `user`);
3. else, if `--rician-correct` is on, auto-estimated by propagating the Rician
   σ through shell averaging and S₀ normalization,
   `σ_m ≈ σ_rician / (S0_median · sqrt(mean_n_dir_per_shell))` (logged as
   `auto-rician`);
4. else a placeholder `0.02` (2 % of the normalized signal) with a warning.

**Intuition.** σ_m sets how sharply the posterior concentrates. As σ_m → 0 the
weight collapses onto the single best entry and the posterior mean reproduces
the MAP estimate; as σ_m grows, more entries contribute and the estimate
becomes a smooth, noise-averaged blend that can fall *between* grid points.
The posterior **std** is a genuine uncertainty map: it is small where one
tissue clearly explains the data (e.g. white matter) and large where many
tissues fit comparably well (e.g. CSF / partial-volume voxels).

`n_eff = 1 / Σ wᵢ²` (same "effective number of atoms" definition AMICO uses,
see Method 3) is the diagnostic for how sharply concentrated a given σ_m made
the posterior: `n_eff → 1` means one entry dominates (MAP-equivalent,
possibly overconfident); `n_eff → n_lib` (the vi-filtered candidate count)
means the weights are nearly uniform and the posterior mean is dominated by
the *candidate library's* distribution rather than the data. A useful target
is a small fraction of `n_lib` — enough to smooth over per-entry MC/grid
noise without washing out the fit; `fit_metadata.json` records `n_lib` so
this can be judged after the fact.

### σ_m calibration (`--target-n-eff`)

`--target-n-eff FLOAT` calibrates σ_m against `n_eff` instead of against
noise. `n_eff(σ_m)` increases monotonically with σ_m under either S₀
convention, because a larger σ_m always flattens the posterior. So
`calibrate_sigma_m()` (`madi/fitters.py`) bisects σ_m on a random subsample of
the voxels being fit (`sample_size`, default 1500) until the sample's *median*
`n_eff` matches the target. It then runs the full fit with that σ_m. Use it to
reason in terms of posterior sharpness instead of a noise value.

σ_m has the same meaning with and without `--fit-s0`: noise std on
`S/S0_measured`. So a noise-derived σ_m (`--sigma-m` or auto-Rician) can be
used directly for `--fit-s0`, and no conversion between the modes is needed.
The two posteriors still differ at the same σ_m. Profiling out `a` can only
shrink a residual, so `--fit-s0` posteriors are as broad as the fixed-S₀
ones or broader. That reflects the extra free parameter, not a unit
mismatch. Matching `n_eff` across the two modes with `--target-n-eff`
therefore compares them at equal sharpness, not at equal noise.

Calibration progress (`σ_m → median n_eff` per bisection step) is printed.
The final calibrated σ_m is logged the same way a user-supplied one would be
(`method_meta['sigma_m']`, `method_meta['sigma_m_source'] = "target-n-eff"`).

**Outputs:** `kio_mean.nii.gz`, `rho_mean.nii.gz`, `V_mean.nii.gz`,
`kio_std.nii.gz`, `rho_std.nii.gz`, `V_std.nii.gz`, `residual.nii.gz`
(posterior-weighted mean residual) (+ `s0_fit_map.nii.gz` with `--fit-s0`),
plus `fit_metadata.json`.

---

## Method 3 — AMICO elastic-net NNLS (`--method amico`)

Treat the library as a dictionary **D** (columns = entry signals) and solve a
non-negative, regularized regression per voxel:

```
x* = argmin_{x ≥ 0}  ||D x − m||²  +  λ₁ ||x||₁  +  λ₂ ||x||₂²
```

Then normalize `w = x / Σx` and report the same weighted means/stds as Bayes,
plus an **effective number of atoms** `n_eff = 1 / Σ wᵢ²` (1 = one entry
explains the voxel; larger = a broad mixture).

Solver:

- `λ₁ = 0`: augmented `scipy.optimize.nnls` (ridge folded in as `sqrt(λ₂)·I`).
- `λ₁ > 0`, `λ₂ > 0`: NNLS with the L1 term folded in by completing the square.
- `λ₁ > 0`, `λ₂ = 0`: `sklearn.linear_model.ElasticNet(positive=True)`.

Defaults are `--lambda1 0.0 --lambda2 0.01` (pure light ridge). With
`--fit-s0` the un-normalized signal is used as **m**; because **x** is
unconstrained in magnitude it absorbs the amplitude, so `Σx` is the fitted S₀
(`S₀ as a free linear parameter`, same semantics as the MAP free-S₀ matcher).
`--log_space` is ignored here — the regression is linear in the signal.

**Intuition.** Rather than committing to one entry (MAP) or a
likelihood-weighted average (Bayes), AMICO reconstructs the voxel as a *sparse
non-negative combination* of library signals. λ₁ pushes toward using few
entries (sparsity); λ₂ shrinks and stabilizes the weights (ridge), which
regularizes ill-conditioned fits and yields smoother maps. `n_eff` is a handy
diagnostic for how many library atoms a voxel actually needed.

**Cost note.** AMICO solves an NNLS per voxel over the whole candidate set, so
it is much slower than MAP/Bayes and scales with the number of candidates.
Tighten `--vi-min` / `--rho-max` to shrink the dictionary on large volumes.

**Outputs:** `kio_mean.nii.gz`, `rho_mean.nii.gz`, `V_mean.nii.gz`,
`kio_std.nii.gz`, `rho_std.nii.gz`, `V_std.nii.gz`, `residual.nii.gz`
(`||D x − m||²`), `n_eff.nii.gz` (+ `s0_fit_map.nii.gz` with `--fit-s0`),
plus `fit_metadata.json`.

---

## GPU acceleration (`--device`)

All three methods can run on GPU via `numba.cuda` (no extra dependency —
`numba` is already required, same as the random-walk simulator in
`madi/walker_gpu.py`):

```
--device auto   # default: use CUDA if available, else CPU
--device gpu    # force GPU; errors out if CUDA isn't available
--device cpu    # force CPU
```

**MAP and Bayes GPU kernels are exact reorderings of the CPU math** — one
CUDA thread per voxel streams over the library instead of NumPy building the
full `(n_voxels, n_library_entries)` matrix, but the arithmetic is identical
and output matches the CPU path to float64 precision.

**AMICO's GPU path is an approximation, not a reimplementation.** The CPU
solves each voxel's `argmin_{x≥0} ||Dx-m||² + λ₁||x||₁ + λ₂||x||²` exactly
via `scipy.optimize.nnls` (active-set). On GPU, every voxel has to be solved
in parallel, so exact active-set NNLS isn't an option — instead each voxel
runs a fixed/adaptive-iteration **FISTA** (accelerated projected-gradient)
solve of the same objective, with positivity enforced by clamping to zero
each step. In practice this lands within a percent or so of the exact
optimum (see `analysis/verify_gpu_fitters.py`), but AMICO-GPU and AMICO-CPU
maps will be *close, not bit-identical* — unlike MAP/Bayes. Three flags tune
the AMICO GPU solver (ignored otherwise):

- `--gpu-chunk-voxels` (default 20000) — voxels processed per GPU launch.
  Only AMICO needs this (its per-voxel workspace is `O(n_library)`); MAP/Bayes
  process the whole volume in one launch since their device-memory use
  scales with `n_voxels + n_library`, not the product.
- `--amico-gpu-iters` (default 2000) — max FISTA iterations per voxel.
- `--amico-gpu-tol` (default 1e-8) — early-exit once the per-voxel
  elastic-net objective's relative change has stayed below this for 20
  consecutive iterations.

**Convergence depends strongly on `--lambda2`.** The ridge term isn't just
a regularizer for AMICO-GPU — it's what conditions the problem for a
first-order (FISTA) solver, and MADI library entries are highly correlated
(similar decay-curve shapes), so the unregularized problem is genuinely
ill-conditioned. At the library's own default (`--lambda2 0.01`) a couple
thousand iterations converge to match CPU almost exactly. As `--lambda2`
drops toward `0`, convergence gets much slower — `n_eff` can look stuck
high (a diffuse, non-sparse solution) well past the default iteration
budget, and *this is a real slow-convergence effect, not a stuck/broken
solver* — raising `--amico-gpu-iters` into the tens of thousands recovers
CPU-like sparsity, though exact agreement at `--lambda2 0` isn't
guaranteed even then. If you need exact low-`lambda2` AMICO behavior, use
`--device cpu`.

The resolved device (`gpu`/`cpu`) is printed at the start of the run and
recorded in `fit_metadata.json` as `"device"`.

---

## Which method should I use?

- **`map`** — fast default, reproducible, no uncertainty. Use for quick looks
  and when you need the original behaviour exactly.
- **`bayes`** — when you want uncertainty maps and noise-averaged, off-grid
  estimates. Tune `σ_m` to your noise level (auto with `--rician-correct`),
  or set `--target-n-eff` directly if you want to reason in terms of
  discrimination strength instead of a noise value (see **σ_m calibration**
  above). The same σ_m applies with or without `--fit-s0`.
- **`amico`** — when you want a sparse mixture interpretation, regularized /
  smoother maps, and the `n_eff` diagnostic, and can afford the per-voxel
  solve. Start from the ridge default and raise `--lambda2` for more smoothing
  or `--lambda1` for more sparsity. (No `--target-n-eff` equivalent yet —
  AMICO's `n_eff` is governed by `--lambda2`, not a `σ_m`-style noise
  parameter, so it would need its own calibration knob; not implemented.)

Every non-`map` run writes `fit_metadata.json` in the output directory
recording the method, all parameters used (including the resolved σ_m and its
source — `user` / `auto-rician` / `target-n-eff` / `default-placeholder` —
plus `n_lib`, the vi/rho_max-filtered candidate-library size, for judging
`n_eff` against), the inputs, and a timestamp, so experiments are
self-documenting.

---

## Fit-time trust floor (`--trust-floor`)

*Added 2026-09-10 for Fisher/CRLB Phase 4, hypothesis H3; see the amendment log below.*

The pre-registered `S/S0` trust floor (`trust_floor` in
`madi/fisher_crlb_preregistration.json`, 0.015) says library signal that low is
Monte-Carlo noise rather than tissue contrast. It is an analysis-time and fit-time
mask, never a builder property, and it is **off by default**. `--trust-floor`
switches it on as an experimental condition. A bare `--trust-floor` uses the
pre-registered value, and `--trust-floor 0.02` overrides it. It applies to every
method and both S₀ conventions, because it acts on the columns and candidates
before any matcher runs.

| mode | flag | what is removed | what it costs |
|---|---|---|---|
| `column` (default) | `--trust-floor-mode column` | any acquisition column at which **any** candidate falls below the floor | trustworthy high-b measurements, discarded because some other tissue decays below the floor there |
| `candidate` | `--trust-floor-mode candidate` | any candidate **entry** below the floor at any used column | nothing measured; the untrustworthy library entries simply cannot win |

A per-`(entry, column)` residual mask is deliberately **not** offered. A fit is a
comparison between candidates, and masking cells gives each candidate a different
number of residual terms. That makes residuals incomparable, and it favours the
very entries the floor indicts, since an entry that drops its own worst-fitting
high-b columns is scored on an easier problem. The candidate filter is
`madi.library.candidate_selection_mask`, the same filter every matcher applies,
and the mask is `madi.fisher_crlb.fit_trust_floor_masks`. The run sidecar records
the floor, its source, the mode, the cell counts, and every dropped column, under
`fit_configuration.trust_floor_mask`.

With `column` mode on a five-shell acquisition, the fit can drop to three columns
for three parameters. The fitter prints the retained feature vector; read it.

---

## Amendment log

This reference is amended in place, per the convention in `INDEX.md`.

### 2026-10-17-bayes-fits0-profile-likelihood — free-S₀ Bayes weighting fixed

- **Previously:** `bayes --fit-s0` divided each candidate's raw residual by
  that candidate's own fitted S₀², and the σ_m-calibration section said the
  fixed- and free-S₀ residuals were on incomparable scales.
- **Now:** the weight is the profile likelihood, raw residual / measured S₀²,
  and `bayes_fit(fit_s0=True)` requires `s0_measured`. The σ_m-calibration
  section and the `--target-n-eff` help text now say that σ_m has one meaning
  in both modes.
- **Why:** the old scaling gave each candidate a noise level proportional to
  its own amplitude, which favoured low-signal candidates and drove cortical-GM
  fits to ρ ≈ 0.05·10⁶, V ≈ 35 pL with S₀ 13–33× the measured b=0. See
  `analysis/cdmd_cohort/springer_discrepancy/`.

### 2026-09-10-trust-floor — the fit-time trust floor is documented

- **Previously:** this reference described no trust-floor option, because the
  fitter had none; the floor existed only as an analysis-time Fisher mask.
- **Now:** the section above documents `--trust-floor` and `--trust-floor-mode`.
- **Why:** Fisher/CRLB Phase 4 tests hypothesis H3 — that high-b library values
  below the floor drive the unrealistic-volume artifact — by switching the floor on
  in a fit. See `fisher_crlb_analysis_plan.md` §7 and its amendment
  `2026-09-10-phase4-trust-floor-fit-forms`. Nothing else in this document changed.
