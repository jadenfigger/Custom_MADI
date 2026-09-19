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
| `--debug` | Dash debug mode with the error overlay |

It opens on a working slice: `delta = 20 ms`, `Delta = 80 ms`, `b <= 6000`, with
three of those columns as the plot axes.

## Using it

1. **Measured columns.** Pick `delta` and `Delta`; leave the b-list empty for all
   25 b-values or select specific ones. *add these* appends them; *add one b
   across all Delta* appends one b-value at every stored `Delta` for that
   `delta`; *clear* empties the set. Add a second `(delta, Delta)` by changing
   the dropdowns and pressing *add these* again — that is the second-`Delta`
   experiment.
2. **Display columns.** Two columns give a WebGL 2-D scatter, three give a 3-D
   one. The menu offers the current pair's columns plus everything already
   measured or displayed.
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
5. **Read the table.** Survivor count, and min / max / std / (max÷min) for each
   parameter and each display column. `max/min` is the natural width on a
   log-spaced axis: `1.0` = a single grid value, `10` = a decade.
6. **Widths vs column count** (checkbox at the bottom) adds the measured columns
   one at a time in selection order and plots how the spreads shrink.

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
| `columns.py` | label loading and the two column readers (`.npz` memmap, column cache) |
| `build_column_cache.py` | one-time banded transpose into `data/cache/` |
| `slice.py` | candidate filter, chi2 conventions, slice + width statistics |
| `app.py` | the Dash application |
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
* Row labels come only from the small arrays (`kios`, `rhos`, `Vs`, `vis`,
  `nominal_*`, `weights`, `is_free_water`). `entry_metadata_json` (5.27 GB) is
  never read, and neither is `load_library_meta` — see `KNOWN_ISSUES.md`.
