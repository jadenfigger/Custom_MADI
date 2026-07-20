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
before exponentiating, for numerical stability. With `--fit-s0` the residual
in the exponent is the free-S₀ residual **divided by that candidate's own
fitted S₀²** before being compared against `σ_m`, and negative-S₀ candidates
get zero weight. This keeps `σ_m` on the same normalized S/S₀ scale in both
modes — the raw free-S₀ residual scales like `S₀²` (thousands, for typical
DWI intensities), so applying `σ_m` to it directly (without this
normalization) made the posterior collapse to a one-hot MAP-equivalent pick
regardless of `σ_m` (`n_eff=1`, `std=0` everywhere) — a bug fixed in both
the CPU and GPU paths.

**σ_m** (residual noise std on the *normalized* signal, **regardless of
whether `--fit-s0` is set**) is chosen as:

1. `--sigma-m FLOAT` if given (source logged as `user`);
2. else, if `--rician-correct` is on, auto-estimated by propagating the Rician
   σ through shell averaging and S₀ normalization,
   `σ_m ≈ σ_rician / (S0_median · sqrt(mean_n_dir_per_shell))` (logged as
   `auto-rician`);
3. else a placeholder `0.02` (2 % of the normalized signal) with a warning.

**Intuition.** σ_m sets how sharply the posterior concentrates. As σ_m → 0 the
weight collapses onto the single best entry and the posterior mean reproduces
the MAP estimate; as σ_m grows, more entries contribute and the estimate
becomes a smooth, noise-averaged blend that can fall *between* grid points.
The posterior **std** is a genuine uncertainty map: it is small where one
tissue clearly explains the data (e.g. white matter) and large where many
tissues fit comparably well (e.g. CSF / partial-volume voxels).

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
  estimates. Tune `σ_m` to your noise level (auto with `--rician-correct`).
- **`amico`** — when you want a sparse mixture interpretation, regularized /
  smoother maps, and the `n_eff` diagnostic, and can afford the per-voxel
  solve. Start from the ridge default and raise `--lambda2` for more smoothing
  or `--lambda1` for more sparsity.

Every non-`map` run writes `fit_metadata.json` in the output directory
recording the method, all parameters used (including the auto-estimated σ_m),
the inputs, and a timestamp, so experiments are self-documenting.
