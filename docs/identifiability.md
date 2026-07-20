# MADI acquisition identifiability (Cramer-Rao / Fisher Information)

`scripts/analyze_identifiability.py` (core math in `madi/identifiability.py`)
quantifies how well a given (Delta, b) acquisition can separate the three
MADI parameters — k_io (membrane permeability), rho (cell density), V (mean
cell volume) — using the Cramer-Rao Lower Bound (CRLB). Its main use so far:
diagnosing why rho and V maps come out noisy/salt-and-pepper at the current
single-Delta acquisition, and checking whether adding a second diffusion
time or extending the b-range would help.

## The CRLB in plain terms

For a signal model S(theta) observed with i.i.d. Gaussian noise of std
sigma on each measurement, the Fisher Information Matrix is

    F_jk = (1/sigma^2) * sum_over_measurements (dS/dtheta_j)(dS/dtheta_k)

The Cramer-Rao bound says: no unbiased estimator can have a covariance
smaller than F^-1. So `diag(F^-1)` is a *best-case* variance for each
parameter, and `trace(F^-1)` summarizes total uncertainty. Lower is better.

The **off-diagonal structure of F is the interesting part here**. If the
derivative vectors dS/drho and dS/dV point in nearly the same direction in
measurement space, the model literally cannot tell a change in rho from a
compensating change in V — F becomes near-singular in that 2x2 block, and
CRLB(rho), CRLB(V) both blow up together. This is exactly the degeneracy
this tool measures, via:

- **rho-V correlation**: `corr = (dS/drho . dS/dV) / (||dS/drho|| ||dS/dV||)`,
  computed directly from the (acquisition-subset) derivative vectors,
  independent of sigma_m. Near ±1 means rho and V are nearly degenerate at
  that point in parameter space.
- **rho-V condition number**: condition number of the 2x2 rho-V block of F.
  Large means the block is nearly singular (consistent with |corr| near 1).
- **Full 3x3 eigendecomposition of F**: the eigenvector of the *smallest*
  eigenvalue points along the least-identifiable direction in
  (k_io, rho, V) space; whichever parameter dominates that eigenvector's
  components is the "most confounded" one at that grid point.

## Finite differences on an irregular grid — and its limits

The MADI forward model is a Monte Carlo random-walk simulator, not a
differentiable function — there's no autodiff path. So `dS/dtheta` is
estimated by **finite differences across neighbouring library entries**.

The library grid is **not regular**:

- kio always spans the full grid at every valid (rho, V) (kio doesn't
  affect the `rho*V <= vi_max` validity filter, so it's regular along that
  one axis).
- rho and V are **irregular**: spacing is denser at small V, and the
  `rho*V <= vi_max` cutoff removes high-rho/high-V corners, so many
  (kio, rho) columns have a different set of valid V's than their
  neighbours, and vice versa.

For each library entry, the code finds its immediate neighbours along each
axis (same other two parameters, next value up/down in the irregular grid)
and applies:

- **Central difference** (neighbours on both sides, spacings h_minus,
  h_plus, generally unequal):
  `dS/dtheta ~= (S(theta+h_plus) - S(theta-h_minus)) / (h_minus + h_plus)`.
  This is the simple non-uniform-spacing central difference, not a
  higher-order weighted-stencil version — deliberately, given the Monte
  Carlo noise floor already in each library entry's simulated signal.
- **One-sided difference** at grid edges (only one neighbour exists, e.g.
  the extreme V for a given (kio, rho) once the vi filter has trimmed the
  far side): `dS/dtheta ~= (S(neighbour) - S(theta)) / (neighbour - theta)`.
- Entries with **no neighbour at all** along some axis (isolated grid
  point) are skipped entirely and counted separately
  (`n_skipped_isolated` in the summary JSON).

The output reports how many entries used central vs. one-sided stencils
per axis (`derivative_stencil_counts`), so you know how much of the library
is affected by edge effects — a large one-sided fraction along an axis
means CRLBs near that axis's grid boundary should be trusted less.

## The one big caveat

**CRLB assumes a continuous, differentiable model and an unbiased
estimator.** MADI's matcher (nearest-library-entry MAP, or the
soft-nearest-neighbour Bayes/AMICO fitters) is neither: it's discrete (the
answer is always a library grid point, or a mixture of them) and biased
(finite library resolution, discretization, and regularization all
introduce bias). On top of that, the derivatives feeding the FIM are
themselves finite-difference approximations across a Monte-Carlo-noisy,
irregularly spaced grid.

Consequently: **treat absolute CRLB numbers as approximate, and only trust
relative comparisons** — one acquisition vs. another, or one region of
(kio, rho, V) space vs. another, computed with the same library and
sigma_m. Don't quote a CRLB as "the variance you will get from the
matcher"; quote it as "acquisition A's CRLB for rho is N times
acquisition B's, at this part of parameter space."

A future "universal"/spectral library — one that stores the underlying
diffusion propagator/spectrum per (kio, rho, V) entry instead of a fixed
(Delta, b) signal vector — would let S(Delta, b) be reconstructed
analytically for arbitrary acquisitions and differentiated exactly,
removing the finite-difference step entirely. This tool is the practical
stand-in until that library format exists.

## Noise sigma (sigma_m)

`--sigma-m` (default 0.02) is the assumed noise std on the *normalized*
S/S0 signal — the same convention and same default as
`madi.fitters.DEFAULT_SIGMA_M` used by the Bayesian fitter, so identifiability
numbers are on a comparable footing with a real Bayesian fit's sigma_m
setting. `--sigma-m-pershell` accepts a per-(Delta,b)-column value instead
(heteroscedastic noise, e.g. because different shells average over
different numbers of directions) — only supported for a single
(non-`--compare`) acquisition run.

## Interpreting a run

A single run (`--acquisition current`, say) produces, per library entry:
CRLB(kio), CRLB(rho), CRLB(V), trace(F^-1), the rho-V correlation and
condition number, the smallest FIM eigenvalue, and which parameter
dominates its eigenvector — plus library-wide summary stats (median/IQR of
each CRLB, median rho-V correlation, and the "degenerate fraction": the
fraction of entries with |rho-V correlation| above `--degenerate-threshold`
(default 0.9)).

`--compare` runs several acquisitions in one invocation (e.g.
`--compare current "trunc=subset_b:500,1000,1500"`) and additionally
produces a bar-chart comparison of median CRLB(rho), CRLB(V), and
degenerate fraction across them, plus an automatic monotonicity sanity
check (fewer (Delta,b) columns should never lower a CRLB — a FAIL here
would point to a finite-difference bug, not real physics).

## Validation built into the tool

1. **PSD check**: each 3x3 FIM should be positive semi-definite (it's a
   sum of outer products of real vectors). A significantly negative
   eigenvalue triggers a warning and is counted in `n_nonpsd` — this
   indicates a finite-difference artifact (e.g. from Monte Carlo noise in
   neighbouring library entries), not real physics.
2. **Gaussian mono-exponential sanity check**: independent of the MADI
   library entirely, the script also evaluates CRLB(ADC) for a
   hypothetical S(b) = exp(-b*ADC) signal on the library's own b-grid, and
   checks that the grid's best b matches the textbook optimal-b ~ 1/ADC
   result (Farooq et al. 2026, Mukherjee et al.). This validates the CRLB
   machinery itself, independent of the finite-difference/irregular-grid
   machinery.
3. **Monotonicity check** (in `--compare` mode): restricting the b-range
   should only ever increase (or leave unchanged) the CRLBs, never
   decrease them. Automatically checked between the acquisition with the
   fewest and the most (Delta,b) columns.
4. **Derivative stencil counts**: reported so you know what fraction of
   the library relies on one-sided (grid-edge) differences per axis.

## Optional: projecting onto voxel space

If you pass `--kio-map --rho-map --V-map` (fitted parameter maps, e.g. from
`scripts/fit_data.py --method map`) and optionally `--seg` (a mask/segmentation),
each voxel's fitted (kio, rho, V) is matched to its nearest library entry
(Euclidean nearest-neighbour in kio/rho/V space, each axis standardized by
the library's own std so the very different native scales don't dominate
the distance), and that entry's CRLB is written back out as
`CRLB_kio.nii.gz`, `CRLB_rho.nii.gz`, `CRLB_V.nii.gz`,
`rhoV_correlation.nii.gz`. This shows spatially where the *acquisition* (not
the data) limits identifiability. It's optional — the core library-only
analysis needs no NIfTI inputs.

## CLI reference

```
python scripts/analyze_identifiability.py \
    --library data/libraries/madi_dense_human.npz \
    --sigma-m 0.02 \
    --acquisition current \
    --out data/outputs/identifiability_current/
```

```
python scripts/analyze_identifiability.py \
    --library data/libraries/madi_dense_human.npz \
    --sigma-m 0.02 \
    --compare current "trunc=subset_b:500,1000,1500" \
    --out data/outputs/identifiability_compare/
```

Key flags:

- `--acquisition {current,subset_b,custom}` — which (Delta,b) columns to
  analyze in a single run. `current` = every column in the library.
  `subset_b` (with `--b-subset 500,1000,1500`) = every library Delta, only
  the given b-values. `custom` (with `--pairs 50:500,50:1000,...`) =
  explicit (Delta,b) pairs — the extension point for a future multi-Delta
  library (adding a second Delta's entries to a rebuilt library and
  including its columns here is the intended one-line change).
- `--compare LABEL=TYPE:PARAMS ...` — run several acquisitions at once and
  get a comparison plot/summary in addition to each one's own outputs.
- `--sigma-m` / `--sigma-m-pershell` — noise model (see above).
- `--degenerate-threshold` — |rho-V corr| cutoff for the "degenerate
  fraction" summary stat (default 0.9).
- `--kio-map --rho-map --V-map [--seg]` — optional voxel-space projection.

## Outputs

- `identifiability_table.csv` — per-entry table (see column list in
  `analyze_library`'s docstring / the CSV header).
- `identifiability_summary.json` — library-wide summary stats.
- `rhoV_correlation_hist.png` — histogram of the rho-V correlation across
  the library (the headline "how degenerate is this acquisition overall"
  plot).
- `crlb_vs_paramspace.png` — CRLB(rho), CRLB(V) vs. V and vs. rho, showing
  where in parameter space estimation is hardest.
- (in `--compare` mode) `acquisition_comparison.png`,
  `comparison_summary.json`.
- (with `--kio-map` etc.) `CRLB_kio.nii.gz`, `CRLB_rho.nii.gz`,
  `CRLB_V.nii.gz`, `rhoV_correlation.nii.gz`.
