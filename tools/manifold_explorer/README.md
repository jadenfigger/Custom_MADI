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

* how wide the survivors are in `(rho, V, k_io)` — a long smear means those
  parameters are **not identifiable** from that measurement;
* how wide they are on unmeasured DISPLAY columns — narrow means that column is
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
pip install -r tools/manifold_explorer/requirements.txt
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
| `madi_dense_universal_remediated.columns.json` | — | — | provenance sidecar |

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
| `--memory-cache-mb N` | shared array-cache budget in MiB (default 256; 0 disables retention) |
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
| 7. Workspace / export | session files, undo/redo, data exports, plot density |

**Right** — one summary line that is always visible, then four views:
**Slice** (prediction space and the two parameter planes), **Fisher / CRLB**
(the matrix, its spectrum, the ellipse over the survivors, CRLB vs column
count), **Widths**, and **Inspect / next acquisition**. All views stay mounted, so a click in any plot
re-centres every other one.

## Workspace, inspection and acquisition planning

* **Save / load sessions.** Section 7 downloads versioned JSON with ordered measured
  columns, reference signals or row, display groups, noise/filter/Fisher settings,
  colours, plot budget, active view and explicit inspected row. Loading restores
  these controls together. Library fingerprints use labels and stored array
  checksums; mismatched libraries are rejected before changing controls. Zoom and
  camera are retained during normal redraws, but are not saved in session JSON.
* **Undo / redo.** Up to 50 states are kept in this browser tab. Rapid changes are
  coalesced over 350 ms. Loading JSON or signal CSV is undoable. Reloading the page
  clears history; save JSON for durable workspaces.
* **Reference CSV.** Import `column_id,signal` or `delta,Delta,b,signal` headers,
  with one acquisition per row and finite S/S0 values. File order becomes measured
  order, including in column-count curves. Import replaces the measured set,
  disables group-following and selects pasted reference mode. Duplicate or unknown
  acquisitions, malformed values and uploads over 2 MiB are rejected. Export
  **Reference signals** for an import template.
* **Inspector.** Hover a scatter point, then open **Inspect / next acquisition**.
  The last hovered row is retained; entering a row pins the inspection. It reports
  nominal/realised parameters, transformed coordinates, survival, chi-square,
  raw RMSE and fitted amplitude, with per-acquisition residuals and the 20 largest
  discrepancies. Free-S0 residuals are `(a*S - reference)/(a*sigma)`, matching the
  slice. Nonpositive amplitudes are rejected with undefined residuals.
* **Acquisition ranking.** Enable ranking to scan unmeasured columns at the
  measured-column picker's `(delta, Delta)`. Score = population standard deviation
  of surviving predictions / measurement noise. With MC variance enabled, the
  noise denominator also includes mean survivor MC variance / 40. Prediction
  quantiles are shown too. This equally weighted, raw-S/S0 heuristic is descriptive,
  not posterior information gain or an amplitude-marginal design bound. Choose a
  ranked acquisition and add it directly in entry-reference mode. Pasted references
  instead need a CSV containing the additional observed signal.
* **Candidate cap.** Section 4 exposes maximum realised rho, using the fitter's
  selection rules. Leave it empty for no cap.

| Shortcut (outside text inputs) | Action |
|---|---|
| Ctrl/Command+S | Save session JSON |
| Alt+Z / Alt+Shift+Z | Undo / redo |
| Alt+1 / 2 / 3 / 4 | Slice / Fisher / Widths / Inspect |
| Alt+R | Snap reference to entered parameters |

## Exporting results

Use section 7's **Export current result** after computation finishes. Numeric
exports use all eligible entries, irrespective of plot sampling.

| Format | Contents |
|---|---|
| Survivor / all-candidate CSV | Row ids, nominal/realised parameters, chi-square, survival and transformed coordinates/validity |
| NumPy NPZ | Eligible signals, ordered acquisition ids/triples, reference, optional MC variance, parameters, chi-square, survival mask, transformed coordinates/validity, session JSON, available reference Fisher matrix/CRLB/spectrum and computed acquisition ranking |
| Reference CSV | Acquisition ids/triples and S/S0; suitable for reimport |
| HTML report | Currently computed interactive plots and session provenance; Plotly embedded for offline viewing |
| Plot camera button | Current plot as a 3600 × 2400 PNG, rendered in the browser |

NPZ files contain numeric, boolean and Unicode arrays only: use
`np.load(path, allow_pickle=False)`. `session_json` is a Unicode scalar containing
JSON. Undefined values (e.g. free-water k_io and invalid ADC) remain NaN in numeric
arrays/CSV, accompanied by masks. Source libraries and caches stay read-only.
The tool imports MADI reference measurements, not arbitrary point clouds or
volumetric images: acquisition and row identities belong to the chosen library.

## Responsiveness and resource limits

Control changes schedule background work. Previous plots remain usable while
status says queued/running; only the newest request in a browser can publish or
export results. Obsolete queued jobs are cancelled and running calculations
check cancellation between stages. An active library read cannot be interrupted.
Errors show an actionable message; **Recompute** retries without losing controls.
Idle pages stop polling.

The default preview samples at most 5,000 distinct entries deterministically from
survivors and background, retaining the reference. Colour ranges, counts, widths,
ranking and exports use all eligible entries. Raise the plot budget for the whole
cloud. Column-count curves are deferred until their view opens. Slice widths
accumulate prefix statistics in linear rather than quadratic column time.

Signal, variance and batch-Jacobian arrays share a byte-bounded LRU cache. The
default 256 MiB limits cached arrays, not total process memory: running jobs,
completed results and figures also use memory. Each input block is limited to
128 MiB and measured selections/imports to 2,048 columns. Two workers serve up to
eight retained browser results. Completed idle results are evicted first when
full and expire after 30 minutes without access; recompute an evicted result.
This local application uses one server process. Its jobs are not shared between
multiple WSGI processes.

Richardson derivatives are used consistently in reference, colour and column-count
views. Richardson MC debias is rejected because cross-stencil covariance is
unavailable; use k=1/k=2 or disable debias. Fisher colours state their fixed-S0,
measurement-noise-only, non-debiased convention. The reference panel follows the
selected S0/debias convention and uses measurement noise only; the exact slice
can additionally include candidate-specific library MC variance.

## Using it

1. **Measured columns.** Pick `delta` and `Delta`; leave the b-list empty for all
   25 b-values or select specific ones. *add these* appends them; *add one b
   across all Delta* appends one b-value at every stored `Delta` for that
   `delta`; *clear* empties the set. Add a second `(delta, Delta)` by changing
   the dropdowns and pressing *add these* again — that is the second-`Delta`
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
   default 0.02 — the fitters' placeholder). The threshold slider is *reduced*
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
6. **Read the table.** Survivor count, and min / max / std / (max÷min) for each
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
node --test tests/manifold_explorer/workspace_history.test.cjs  # optional browser-history unit checks
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
| `data.py` | read-only library access and cached candidate selection |
| `analysis.py` | UI-independent analysis, inspection, ranking and scientific exports |
| `runtime.py` | bounded array cache and background jobs |
| `session.py` | session schema, library identity and validated JSON/CSV imports |
| `workspace.py` | Dash adapters for jobs, downloads and inspection |
| `assets/workspace.js` | browser-local history, debouncing and shortcuts |
| `app.py` | Dash layout and scientific view controller |
| `sanity_checks.py` | timings and the three sanity checks against the real library |
| `selftest.py` | runs the test suite without pytest |
| `KNOWN_ISSUES.md` | two defects in existing code this tool routes around |

## Conventions reused from the project

* Library vectors are **already S/S0** — the `b = 0` column of every
  `(delta, Delta)` pair is exactly 1.0 by construction — which is what
  `madi.library.match_voxels_batch` expects. No renormalization is applied.
* The candidate filter calls `madi.library.candidate_selection_mask` itself, on
  label-only `LibraryEntry` objects, so the `v_i` band / `rho_max` / free-water
  rules cannot drift from the fitters'.
* Fixed-S0 chi2 is the exponent of the `bayes` fitter's Gaussian weight, so a
  slice at threshold `T` is exactly the set of entries whose Bayes weight
  relative to the reference exceeds `exp(-T/2)` — shown in the readout.
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
  never read, and neither is `load_library_meta` — see `KNOWN_ISSUES.md`.
