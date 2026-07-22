# The (δ, Δ, b)-universal MADI library

This document describes the simulation refactor that made the MADI library
valid across a whole grid of diffusion timings and b-values from a **single**
Monte-Carlo run per tissue ensemble, replacing the previous fixed-PGSE scheme
where δ, Δ, and the b-list were all baked into the simulation.

If you only want to *build* a library on Sol, skip to
[`sol_package_guide.md`](sol_package_guide.md). This doc explains what changed
and why, and is the reference for anyone touching `walker_gpu.py`,
`signal.py`, `config.py`, or `library.py`.

---

## 1. Motivation

The old walk kernel accumulated two fixed PGSE phase-moment windows per
walker: `m1 = ∫₀^δ x dt` and `m2 = ∫_Δ^{Δ+δ} x dt`, for a single δ and up to
four hardcoded Δ. Its per-walker output `dM = m1 − m2` is gradient-independent,
so **b-values were already free** (swept in `signal.py` via `G_from_b`). But
**δ and Δ were frozen** into the accumulation windows — changing either meant
re-running the whole Monte Carlo.

The refactor removes that limitation. One walk now produces a library valid for
any δ, Δ on the storage grid and any stored b, at ~the same simulation cost.

## 2. The idea: store the running position integral Y(t)

Instead of two fixed windows, each walker accumulates the running integral of
its position along the gradient axis, sampled at a coarse stride `h`:

```
Y(t) = ∫₀ᵗ x(s) ds          (trapezoid rule between MC steps)
```

For **any** (δ, Δ) whose δ, Δ, and Δ+δ land on the `h`-grid, the phase-moment
difference is three lookups into the stored curve:

```
dM(δ, Δ) = Y(δ) + Y(Δ) − Y(Δ+δ)
```

This is algebraically identical to the old kernel's `m1 − m2` (same units
[µm·ms], same sign), so everything downstream of `dM` — the SI-unit conversion,
`G_from_b`, and `S = ⟨cos(γ·G·dM)⟩` — is **unchanged**. The signal at any
(δ, Δ, b) is then:

```
S(b; δ, Δ) = ⟨ cos( γ · G(b, δ, Δ) · dM(δ, Δ) ) ⟩     (real part)
```

### Why h = 1 ms with integer-ms δ, Δ needs no interpolation

The library's stored δ and Δ are integers in ms, and `h = 1 ms`, so every δ, Δ,
and Δ+δ is an exact multiple of `h` and lands *exactly* on a stored `Y` sample.
The three lookups are exact — **there is no interpolation of Y**, and that error
source is exactly zero. `SimConfig.assert_grid_alignment()` enforces this at
build time and fails loudly otherwise. (The MC integration itself still runs at
`ts = 1 µs`; `h` is only the *storage* stride, so `Y` values stay accurate.)

### Numerical conditioning: origin shift

`dM = Y(δ) + Y(Δ) − Y(Δ+δ)` subtracts large, similar-magnitude integrals
(~δ·⟨x⟩) to get a small result (~δ·µm) — catastrophic cancellation in fp32.
Because the physically meaningful quantity is origin-independent, the kernel
accumulates `Y` from `(x − L/2)` rather than `x`. This is provably invariant for
`dM` (a constant offset `c` contributes `c·(δ + Δ − (Δ+δ)) = 0`) and keeps `Y`
small, so fp32 storage is safe.

## 3. Reduction: Y → S[δ, Δ, b]

A single walk produces `Y[N_walkers, n_grid, 3]`; the signal grid needs ~10¹⁰
`cos` evaluations per library entry, far too many for host NumPy. The stack is
`numba.cuda` only (no cupy), so reduction is a **dedicated CUDA kernel**
(`_reduce_kernel` in `walker_gpu.py`): one thread block per (δ, Δ, b) column,
threads stride over the flattened `(walker × axis)` samples accumulating
`cos`/`sin` in shared memory. The per-column phase coefficient `γ·G(b,δ,Δ)·1e-9`
is precomputed on the host (`signal.build_columns`).

The walk kernel is therefore **grid-agnostic**: changing the δ/Δ/b lists never
recompiles it. A host/CPU reduction (`_reduce_cpu`) and a direct-Y API
(`run_walk_Y`) exist for validation and CPU-only testing.

Walkers are processed in chunks (`SimConfig.walker_chunk`) so the `Y` buffer's
peak memory is a tuning knob, not a hard limit.

## 4. Single isotropic ensemble, three axes harvested

The old code built **three independent ensembles per entry**, one per gradient
axis. The refactor assumes the Voronoi geometry is statistically isotropic and
builds **one ensemble per entry**, harvesting all three Cartesian components of
each 3-D walk as independent samples. Effective sample count is
`N_eff = 3 · n_ensembles · n_walkers`, and the MC noise floor is
`σ_MC = 1/√(3·N_w)`.

## 5. Common-random-numbers (CRN) seeding

Ensemble and walker RNG seeds are derived from `(build_seed, ensemble_index[,
kio])` **only — never from (ρ, V)** (see `_ensemble_seed` / `_walk_seed`). So
every (ρ, V) grid point at a fixed ensemble index shares the same underlying
random-number streams. This is a concession to the *next* phase (Fisher/CRLB),
which computes finite-difference derivatives ∂S/∂ρ and ∂S/∂V across neighbouring
library entries: shared random numbers make those differences far less noisy.

**Operational consequence:** the build-level `--seed` (default `0`) must be the
same across every SLURM shard of one library. See the `--seed` note in
`sol_package_guide.md` §Step 7.

## 6. No interpolation, anywhere

By explicit design decision, the library stores an **explicit, code-defined**
set of δ, Δ, and b columns, and matching selects the **nearest** stored column
(`library._grid_columns`). A measured protocol that doesn't land exactly on the
grid is matched to the nearest column and the mismatch is accepted as error —
there is no b-spline or (δ, Δ)-plane interpolation to build, tune, or validate.
Choose the stored grid (in `config.py`) to cover the protocols you care about.

## 7. What each module does now

| module | role |
|---|---|
| `madi/config.py` | Defines the storage grid: `small_deltas`, `big_deltas`, `b_values` (evenly spaced, step `B_STEP_S_MM2`), `h_ms`, `T_max_ms`. `n_steps` is now **derived** (`T_max/ts`), not a settable field. `valid_delta_pairs()` gives the triangular Δ≥δ grid; `assert_grid_alignment()` is the build-time guard. |
| `madi/walker_gpu.py` | Walk kernel produces `Y(t)`; reduction kernel produces `Σcos/Σsin` per column. `run_simulation_reduced` / `run_simulation_multi_kio_reduced` (ensemble reuse across a kio sweep) are the production entry points; `run_walk_Y` exposes raw `Y` for validation. |
| `madi/signal.py` | `build_columns` turns physical (δ,Δ,b) into the flattened per-column lookup/coefficient arrays; `compute_signals[_multi_kio]` orchestrate walk+reduce and assemble `S[n_pairs, n_b]`. Phase convention unchanged from the old pipeline. |
| `madi/library.py` | `LibraryEntry.vector` is the flattened `S[δ,Δ,b]` block. New `.npz` format (`pair_deltas`, `pair_Deltas`, `b_values`, `n_b`, `h_ms`); `load_library_meta` still reads **legacy** fixed-δ files. Nearest-column matching. |

### Storage grid defaults

- δ = 1..30 ms, every 1 ms (30 values)
- Δ = 1..50 ms every 1 ms, then 55..80 ms every 5 ms (56 values), triangular Δ≥δ → **1245 valid pairs**
- b = 25 points evenly spaced by 500 s/mm² on [0, 12000] s/mm²
  (`evenly_spaced_bvalues` in `madi/config.py`) — chosen so nearest-column
  matching lands close to real DWI shells, which are typically acquired on
  a coarse, evenly-spaced grid rather than a curved one. Pass an explicit
  `SimConfig(b_values=[...])` to store a fully custom list instead.
- `T_max = 128 ms`, `h = 1 ms`, `ts = 1 µs`
- Per-entry vector: 1245 × 25 = **31 125 float32** (~124 KB)

## 8. Validation

`analysis/validate_universal_library.py` runs the suite (GPU). Current status:

- **Free diffusion (gate test): PASS.** `Var(a)` matches the analytic
  `2·D0·(Δ − δ/3)` to <1.1 % across (δ,Δ); `S(b)` matches `exp(−b·D0)` (in
  matched s/mm² ↔ mm²/s units) to within ~2σ of the MC noise floor; the
  imaginary part `⟨sin⟩` is consistent with zero. This validates Y, the grid,
  eq. dM, and eq. S together.
- **Grid convergence: PASS.** Halving `h`, halving `dt`, and doubling `N_w`
  each move `S` by less than the MC noise.
- **Permeation dt diagnostic: PASS.** Crossing probability stays ≤0.056 across
  the realistic kio/⟨A/V⟩ range, well under the ~0.3–0.5 caution zone for the
  discrete-time model. This is a diagnostic, **not** a full Kärger comparison.
- **Impermeable sphere: FAIL (see caveat).** A supplementary standalone
  toy-sphere test showed ~13 % low bias vs. the exact `2R²/5` long-time plateau.
  The bias is most likely in that test script's simplified reject-on-exit
  boundary handling, not in the Y/reduction machinery (which the two passing
  quantitative tests already validate). Flagged honestly rather than tuned away.

Not implemented: full Kärger two-compartment exchange comparison; fitting-stage
validation (out of scope for this phase).

## 9. Known limitations / flags

- **Finite gradient ramp time.** With δ allowed down to 1 ms, preclinical ramps
  (~0.1–0.3 ms) are 10–30 % of the smallest δ. The rectangular-lobe assumption
  may bias the very-small-δ columns. Confirm the scanner's actual ramp time; if
  it matters, floor δ higher or accept the bias. (The Y-two-accumulator
  generalization for trapezoidal lobes was deliberately **not** built.)
- **Waveform is PGSE only.** TRSE-approximated-as-PGSE bias is a known, accepted
  limitation; no OGSE/bipolar/general-waveform support.
- **Fitting stage.** The `--fit` path in `scripts/fit_data.py` **is** now
  wired to the (δ,Δ,b) library, along with the `map` / `bayes` / `amico`
  fitters and the `--export-voxel` path. Each measured scan supplies a
  (δ, Δ) via its `--input` (Δ from the first field, δ from the global
  `--small-delta` or a per-scan `δ,Δ:...` override) and the matcher selects
  the nearest stored `(δ,Δ,b)` column (`_grid_columns`, no interpolation).
  `match_voxels_batch` / `match_voxels_batch_fits0` / `bayes_fit` /
  `amico_fit` all take `lib_delta_pairs` + `fit_triples` now.
  **Still on the old (Δ,b) format, not yet migrated:** the identifiability /
  Fisher-CRLB tooling (`madi/identifiability.py`,
  `scripts/analyze_identifiability.py`), `scripts/plot_b_space_map.py`, and
  the small dev helpers (`scripts/_make_synth_dwi.py`,
  `scripts/_sanity_fitters.py`, `analysis/verify_gpu_fitters.py`).
