# MADI manifold slicing explorer

An interactive view of the universal `(delta, Delta, b)` library as a sampled
model manifold: pick some acquisition columns, pretend you measured them, and
see which parameter triples a measurement there could not tell apart.

Everything here is read-only with respect to the library. The builder, the
fitters, the library file and existing results are untouched.

---

## What it does

Each library row is one point in a 31,125-dimensional prediction space, labelled
by `(rho, V, k_io)`. Each column is one acquisition coordinate `(delta, Delta, b)`.

**Slicing** = choose MEASURED columns and a reference point, then keep the entries
whose predictions on those columns agree with the reference to within the noise:

```
chi2(entry) = sum over measured columns of ((s_entry - s_ref) / sigma)^2   <=  threshold
```

and read off two things:

* how wide the survivors are in `(rho, V, k_io)` â€” a long smear means those
  parameters are **not identifiable** from that measurement;
* how wide they are on unmeasured DISPLAY columns â€” narrow means that column is
  already implied by what you measured, so acquiring it adds nothing.

The intended experiment: slice at one `Delta`, watch the survivors stretch along
a constant `rho*V` curve, then add a second `Delta` and watch the curve collapse.

The slice is the **exact** region. The **Fisher / CRLB** view puts its *local
quadratic approximation* on the same axes — `chi2 ~ dtheta^T F dtheta` — so you
can see where the linearisation holds and where it does not. Measured at
`rho=3.3e5, V=2.1, k_io=20`, sigma 0.01:

| measured | CRLB log rho | rho width the Fisher bound predicts | rho width the slice measures |
|---|---|---|---|
| `Delta=50`, 13 columns | 0.381 | 15.6 | **80.3** |
| `+ Delta=20`, 26 columns | 0.063 | 1.90 | **1.73** |

At one diffusion time the bound is optimistic by 5x, because the true
degeneracy is a curved hyperbola and an ellipse can only be its tangent. Once a
second `Delta` shrinks the region, the two agree to 10%. That comparison is the
reason both are drawn together.

## Setup (once)

```bash
pip install dash
```

Then build the column-major cache. The library is stored row-major, so reading
one column from it costs a full 4.7 GB scan (~3 s); transposed, a column is
150 KB of contiguous bytes (~10 ms). Without the cache the app still works, it
is just slow.

```bash
python -m tools.manifold_explorer.build_column_cache
```

Writes into `data/cache/` (a separate folder from `data/libraries/`, so it can
never be mistaken for a library artifact):

| file | shape | dtype | size |
|---|---|---|---|
| `madi_dense_universal_remediated.columns_f64.npy` | (31125, 18820) | float64 | 4.69 GB |
| `madi_dense_universal_remediated.variance_columns_f32.npy` | (31125, 18820) | float32 | 2.34 GB |
| `madi_dense_universal_remediated.columns.json` | â€” | â€” | provenance sidecar |

**7.03 GB total, 328.8 s to build** as measured on this machine. float64 keeps
the signal cache bit-identical to the source; `--dtype float32` halves it if
disk is short. `--no-variance` skips the second file (and disables the
"add library MC variance" option). The sidecar records the source file and its
size, and is only marked `complete` at the end, so an interrupted build is
never used.

## Run

```bash
python -m tools.manifold_explorer.app
```

Opens `http://127.0.0.1:8050` in your browser. Useful flags:

| flag | meaning |
|---|---|
| `--library PATH` | a different library `.npz` |
| `--cache-dir PATH` | a different cache folder |
| `--no-cache` | read columns straight from the `.npz` (no cache needed, ~3 s per new `(delta, Delta)`) |
| `--port N` | serve on another port |
| `--no-browser` | do not open a browser |
| `--debug` | Dash debug mode with the error overlay |

It opens on a working slice: measured columns at `delta = 20 ms`,
`Delta = 80 ms`, `b <= 6000`; display axes at `delta = 4 ms`,
`Delta = 20 / 30 / 40 ms`, `b = 1000-4000`, collapsed by Mean.

## Layout

**Left** — controls in the order you use them, each section foldable:

| section | what it decides |
|---|---|
| 1. Measured columns | which `(delta, Delta, b)` the slice is computed on |
| 2. Display axes | the plot coordinates only; never the slice |
| 3. Reference point | the point everything is measured from |
| 4. Slice | sigma, threshold, S0 convention, candidate filter |
| 5. Colour | which quantity the points are coloured by |
| 6. Fisher / CRLB | stencil, ellipse convention, debias, diagnostics |

**Right** — one summary line that is always visible, then three views:
**Slice** (prediction space and the two parameter planes), **Fisher / CRLB**
(the matrix, its spectrum, the ellipse over the survivors, CRLB vs column
count) and **Widths**. All three stay mounted, so a click in any plot
re-centres every other one.

## Using it

1. **Measured columns.** Pick `delta` and `Delta`; leave the b-list empty for all
   25 b-values or select specific ones. *add these* appends them; *add one b
   across all Delta* appends one b-value at every stored `Delta` for that
   `delta`; *clear* empties the set. Add a second `(delta, Delta)` by changing
   the dropdowns and pressing *add these* again â€” that is the second-`Delta`
   experiment.
2. **Display axes.** Each axis is a *group* of columns collapsed to one number.
   Three slots; set one to *off* for a 2-D plot. A slot is either
   **(delta, Delta) + b** (one diffusion time, any set of b-values) or
   **any columns** (an arbitrary set, Mean only). Two collapse methods:

   | method | definition | units | when |
   |---|---|---|---|
   | **Mean signal** (default) | mean of S/S0 over the group's columns | S/S0 | any group |
   | **ADC** | `-` OLS slope of `ln S` on `b`, in closed form | um^2/ms | one (delta, Delta), >= 2 distinct b |

   A group holding a single b-value collapses to exactly that column's value,
   so the older one-column-per-axis behaviour is the special case. ADC is
   disabled with a reason shown when any axis cannot support it. Entries with
   `S <= 0` anywhere in an ADC group have no logarithm; they are **excluded
   from the plot, never clipped**, the count appears in the title and readout,
   and the library rows are logged to the console.

   Collapsing is **display only** - the slice always runs on the individual
   measured columns. *use display groups as measured columns* is a convenience
   toggle that points the measured set at the union of the group columns; the
   chi2 still runs per column.
3. **Reference point.** Either click any point in any plot (faint background
   points are clickable too), or type `rho`/`V`/`k_io` and press *snap to
   nearest entry* (nearest in `log rho`, `log V`, `k_io`), or switch to
   *pasted signal values* and paste one S/S0 number per measured column.
4. **Slice.** `sigma` is the measurement noise in S/S0 units (log slider,
   default 0.02 â€” the fitters' placeholder). The threshold slider is *reduced*
   chi2, so `1.0` means "agrees to about one sigma per column". `S0 handling`
   switches between the fixed-amplitude and marginalised-amplitude conventions.
   `add library MC variance in quadrature` folds in the library's own
   Monte-Carlo standard error per column, `sqrt(signal_variance / 40)`.
5. **Colour.** Four hues read off the entry's labels (rho, V, k_io, rho*V) and
   two measured from the reference point over the current measured columns:

   - **RMSE to reference** - `sqrt(mean((s - r)^2))` in S/S0 units, with no
     sigma in it, so it does not move when the noise slider does;
   - **chi distance** - `sqrt(chi2)` using the slice's own sigma, MC-variance
     term and S0 convention, so `chi <= sqrt(threshold)` is exactly the
     surviving set. That edge is marked on the colorbar.

   Both are 0 at the reference entry and are disabled until a reference
   exists. They colour the faint background entries as well, so you see the
   whole distance field and where the slice cuts it; parameter hues leave the
   background grey. The linear/log toggle helps because both span orders of
   magnitude; on a log scale values are floored at `COLOUR_LOG_FLOOR`
   (1e-6, named in `app.py` and shown in the colorbar title) because the
   reference itself is exactly 0.
6. **Read the table.** Survivor count, and min / max / std / (maxÃ·min) for each
   parameter and each display column. `max/min` is the natural width on a
   log-spaced axis: `1.0` = a single grid value, `10` = a decade.
7. **Fisher / CRLB** (section 6, and the view of the same name). At the
   reference node the app builds the Jacobian by central differences across
   neighbouring library entries — the same canonical nominal grid, log-coordinate
   denominators and pre-registered stencil widths `scripts/run_fisher_phase1.py`
   uses — and then calls `madi.fisher_crlb` for everything else. You get:

   - the 3x3 matrix in `(log rho, log V, k_io)`, its CRLB, `kappa`, and the
     eigenvalues of `D F D`;
   - the sloppy direction's angle from constant-`v_i`, in 3-D and in the
     `k_io`-profiled plane (Phase 3's quantity);
   - the confidence ellipse at the slice's own chi2 threshold, drawn over the
     survivors, either with `k_io` profiled out (marginal, the default and the
     honest one when `k_io` is unknown) or with `k_io` known (conditional,
     always smaller);
   - CRLB against the number of measured columns.

   Options: stencil half-width `k=1` (pre-registered), `k=2`, or Richardson
   `(4J1-J2)/3`; Monte-Carlo debiasing of the derivatives (off by default, see
   below); and an overlay of the ellipse on the Slice view's `(rho, V)` plot.
8. **Widths vs column count** (checkbox in section 6) adds the measured columns
   one at a time in selection order and plots how the spreads shrink, with the
   Fisher-predicted width beside the measured one.

### What the Fisher view will not do

* **No central stencil, no matrix.** Both neighbours must exist. At the `v_i`
  band edges and at `k_io = 0 / 130` they do not, so only **11,417 of 18,819
  entries (60.7%)** carry one at `k=1` (233 of 369 `(rho, V)` nodes x 49 of 51
  `k_io` values) and **4,982** at `k=2`. Those entries report no Fisher matrix
  rather than a one-sided substitute, and a Fisher hue draws them in pale grey
  with a count rather than dropping them.
* **Derivatives are w.r.t. the nominal grid coordinate.** The realised
  finite-geometry `rho`/`V` differ by up to 0.69% / 0.93%.
* **Monte-Carlo debiasing is off by default.** Common random numbers make a
  difference of neighbouring entries far less noisy than `var_minus + var_plus`
  suggests, but the covariance correction needs `ensemble_means_subset`, stored
  for 8 `(delta, Delta)` pairs x 25 b = 200 columns only. The app uses the
  correction on whatever columns it covers and says how many; the rest get the
  conservative form, which subtracts too much.
* **An indefinite matrix is reported, not repaired.** `fisher_crlb` refuses to
  invert one, and the panel says so.
* **The ellipse knows nothing about the `v_i` band.** It can extend past
  `v_i in [0.40, 0.99]`, where the slice cannot follow it, because a CRLB has
  no prior. Compare shapes near the reference, not extents.

## Checks

```bash
python -m pytest tests/manifold_explorer -q              # needs pytest
python -m tools.manifold_explorer.selftest               # no pytest needed
python -m tools.manifold_explorer.selftest --slow        # also reads the 15 GB artifact
python -m tools.manifold_explorer.sanity_checks          # prints timings + the degeneracy result
```

`sanity_checks` is deliberately print-only: it reports the rho-V degeneracy
result as measured, with nothing asserted or tuned.

## Files

| file | role |
|---|---|
| `axes.py` | axis groups and the Mean / ADC collapse |
| `fisher.py` | node grid, finite-difference Jacobian, ellipse; everything else is `madi.fisher_crlb` |
| `figures.py` | every Plotly figure and the shared colour scale |
| `columns.py` | label loading and the two column readers (`.npz` memmap, column cache) |
| `build_column_cache.py` | one-time banded transpose into `data/cache/` |
| `slice.py` | candidate filter, chi2 conventions, slice + width statistics |
| `app.py` | the Dash application |
| `sanity_checks.py` | timings and the three sanity checks against the real library |
| `selftest.py` | runs the test suite without pytest |
| `KNOWN_ISSUES.md` | two defects in existing code this tool routes around |

## Conventions reused from the project

* Library vectors are **already S/S0** â€” the `b = 0` column of every
  `(delta, Delta)` pair is exactly 1.0 by construction â€” which is what
  `madi.library.match_voxels_batch` expects. No renormalization is applied.
* The candidate filter calls `madi.library.candidate_selection_mask` itself, on
  label-only `LibraryEntry` objects, so the `v_i` band / `rho_max` / free-water
  rules cannot drift from the fitters'.
* Fixed-S0 chi2 is the exponent of the `bayes` fitter's Gaussian weight, so a
  slice at threshold `T` is exactly the set of entries whose Bayes weight
  relative to the reference exceeds `exp(-T/2)` â€” shown in the readout.
* Free-S0 chi2 mirrors `madi.library.match_voxels_batch_fits0`, including the
  division by the fitted amplitude squared that the `--fit-s0` bayes path
  applies to keep `sigma` on the S/S0 scale. A test checks the two agree.
* ADC uses the closed-form OLS slope of `ln S` on `b`, negated, with `b` in
  s/mm^2 converted to um^2/ms via `B_S_MM2_TO_MS_UM2` (1 s/mm^2 = 1e-3 ms/um^2).
  Checked against `np.polyfit` entry by entry and against a synthetic
  `exp(-b D)` curve.
* The Jacobian is built exactly as `scripts/run_fisher_phase1.py` builds it,
  and every quantity downstream of it -- `fisher_matrix`, `fisher_spectrum`,
  `fisher_diagnostics`, `rho_V_profiled_block`, `degeneracy_geometry`,
  `directional_crlb`, `amplitude_marginal_fisher`, `derivative_variance` -- is
  imported from `madi.fisher_crlb`, never reimplemented. Independent check:
  this module finds central stencils at 233 `(rho, V)` nodes, which is exactly
  the row count of Phase 3's committed
  `table3_1_rho_V_medians_full_stored_domain.csv`.
* Row labels come only from the small arrays (`kios`, `rhos`, `Vs`, `vis`,
  `nominal_*`, `weights`, `is_free_water`). `entry_metadata_json` (5.27 GB) is
  never read, and neither is `load_library_meta` â€” see `KNOWN_ISSUES.md`.
