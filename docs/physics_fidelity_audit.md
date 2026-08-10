# Physics-fidelity audit of the MADI reimplementation

Audit date: 2026-08-02.  Scope: the checked-in implementation and the actual
production artifact `data/libraries/madi_dense_universal.npz`, not an intended
build recipe.  No production code was modified for this audit.  The only new
files are rerunnable audit tests under `tests/physics_audit/`.

## Executive result

The implementation has a usable **reference fitter**: fixed-S0, linear-space,
exhaustive least-squares MAP is the CLI default when `--method map`, no
`--fit-s0`, no `--log-space`, and no `--rician-correct` are selected.  The
unit conversions, `b` formula, use of `cos(phi)`, symmetric `p_p`, rejection
on failed membrane transmission, and Poisson seed generation are also
substantially aligned with the stated MADI model.

It is nevertheless **not physics-equivalent to the paper reference for
identical inputs**.  The blockers are material:

1. The forward model uses finite rectangular gradient-lobe phase integrals,
   while the requested paper reference is the endpoint-displacement
   (`q*d_parallel`) model.  On the same reduced Monte-Carlo paths this changed
   the signal by 12.5% at `b=1000` and up to 228% at `b=6000` for
   `(delta, Delta)=(20,50)` ms; for MADI III timing `(7,25)` ms it changed the
   signal by 14.3% and 110%, respectively.  This is not a numerical detail.
2. The production library is a 40,645-entry **masked dense rho-by-V grid**,
   not the paper's discrete `v_i` hyperbolae.  It has 739 distinct `v_i`
   values, no `k_io=0`, and does not cover the paper's low-density, small-cell,
   high-volume, or near-free-water limits.
3. The geometry lookup table silently clamps high requested `v_i`; the
   resulting ensemble is labelled with the requested value rather than its
   measured value.  In a real current-code diagnostic, requested `v_i=0.90`
   produced about `0.782` actual intracellular volume fraction.  The
   production artifact contains labels up to `0.9488` but no geometry
   provenance, so this cannot be ruled out for its high-`v_i` rows.
4. The CUDA classifier uses the two nearest seeds cached at a 1-um voxel
   center, not the true two nearest seeds at each walker position.  At
   `rho=500,000`, `V=1.4`, target `v_i=0.70`, 100,000 random points showed
   1.166% compartment disagreement with exact KD-tree classification and
   0.826% disagreement in a proposed-step membrane count.  That is 35% of the
   2.351% true crossing attempts in that experiment.

The pass/fail detail below distinguishes a demonstrated failure from an item
that cannot be established because the artifact does not retain needed
provenance.  “Indeterminate” is not a pass.

## Method and uncertainty

The production file was inspected directly as a ZIP/NPY container and the
uncompressed `vectors.npy` payload was memory-mapped; it was not inferred from
the writer.  All 128 shards were compared by their exact rounded
`(k_io,rho,V)` keys.  The signal-quality scan visited all 1,265,075,625 stored
values.

The dense preset nominally has `3 * 40 * 100,000 = 12,000,000` axis samples
per entry, so `1/sqrt(N)=2.887e-4` is only a **nominal** independent-sample
Monte-Carlo scale.  It is not a defensible per-entry confidence interval:
three axes share a path and geometry, geometries are reused across a kio sweep,
and the file contains no replicate-level variance.  Reported artifact counts
are exact counts, not MC estimates.  The finite-lobe comparison used only
1,024 walkers (nominal `N_eff=3,072`, rough bound `1/sqrt(N_eff)=0.0180`), so
it is a conservative waveform-difference demonstration rather than a
production-precision decay.

## 1. Equation-to-code map

| Paper specification | Code location and exact expression | Verdict |
|---|---|---|
| `D0=3.0 um2/ms` | `madi/config.py:35`: `D0_UM2_MS = 3.0`; `SimConfig.D0` at line 144 | Pass |
| `t_s=1 us` | `madi/config.py:145`: `ts: float = 1e-3` ms | Pass |
| RMS step length | `madi/config.py:183-190`: `sigma = sqrt(2*D0*ts)`, `ls_rms = sqrt(6*D0*ts)` | Pass for the stated 134 nm RMS value.  The actual step is Gaussian per axis, not fixed-magnitude isotropic. |
| Maximum RW duration 80 ms | `madi/config.py:71,146`: `T_MAX_MS=128.0` and `T_max_ms=T_MAX_MS` | Fail for the production/default path.  A custom 80-ms limited grid can be made, but the production universal grid needs 110 ms. |
| Proton gamma | `madi/config.py:37`: `GAMMA_RAD = 2.675222e8` rad/(s T) | Pass |
| 42 M intracellular water / `0.042*k_io*V` | No forward, derived-map, or metabolic conversion implementation was found. | Missing |
| Representative `p_p=0.005` | No fixed representative value; `p_p` is derived from `k_io`. | Not applicable as a default; correct only when an input `k_io` implies it. |
| 10,000 walkers/batch; 120 ensembles; 12 million walks | `madi/walker_gpu.py:75` states `PAPER_WALKS_PER_ENTRY=12_000_000`; dense preset `scripts/fit_data.py:135-136` uses 100,000 walkers and 40 ensembles, then harvests 3 axes. | Nominal count 12 million; not the paper's 120 independent ensembles and axes are not validated as independent. |
| 400 b values / trusted to 0.015 | Production file has 25 b values; `0.015` is not enforced by simulator or fitter. | Fail / missing guard |
| Poisson Voronoi seeds, no Lloyd relaxation | `madi/ensemble.py:493-497`: `rng.poisson(...)`, `rng.uniform(...)`, `cKDTree`; no relaxation code exists. | Pass |
| `v_i=rho*V*1e-6` | `madi/ensemble.py:89-93,475-479`; `madi/library.py:85-88,461-465` | Pass |
| Contracted-cell rule | `madi/ensemble.py:487,500`: lookup `alpha_star`, then `np.minimum(alpha_star, cfg.kappa * nn_dist / 2.0)` with `kappa=.95` (`config.py:166`) | Choice is documented in code but underdetermined by paper.  See geometry findings. |
| Paper-style geometry statistics | `madi/ensemble.py:285-314` uses one fixed seed cloud, 25 alpha values, 200,000 volume samples, and exact-cell calculation; `215-218` returns a 5--95% trimmed mean of per-cell `A/V`. | Divergence: not 26 alpha values or 5 million shapes, not an untrimmed actual-generated-ensemble mean. |
| Eq. 3--5, `p_p <-> k_io` | `madi/walker_gpu.py:86-94`: `factor = sqrt(D0/(6*ts))*mean_AV`; `return (kio/1000)/factor`; inverse `pp*factor*1000` | Algebra and ms-to-s conversion pass.  `mean_AV` is lookup/trimmed, not measured from the generated ensemble. |
| Side-symmetric `p_p`, `p_p^m`, rejection | CUDA `madi/walker_gpu.py:210-228`; CPU `354-367`; execution clamp `411-412` | Pass in source.  No production occupancy or `m` telemetry is retained. |
| `p_p` bounds | `madi/walker_gpu.py:411-412`: `pp = float(np.clip(pp, 0.0, 1.0))` | Partial pass: it clips in the execution path, but silently; `kio_to_pp` itself does not warn or guard. |
| `b=(gamma*G*delta)^2*(Delta-delta/3)` | `madi/signal.py:32-39`: `b_si=b_s_mm2*1e6`, `d_si=delta_ms*1e-3`, `tD_si=(Delta_ms-delta_ms/3)*1e-3`, then inverse square root | Pass |
| Published narrow-pulse phase `q*(r-r0)` | Current `madi/signal.py:10-13`, `madi/walker_gpu.py:16-20,234-247`, reduction `287-290`: `dM=Y(delta)+Y(Delta)-Y(Delta+delta)`, then `cos(phase_coef*dM)` | Fail: finite-lobe integral, not endpoint displacement. |
| Signal is real ensemble average | `madi/signal.py:108-119`: `S=cos_sum/n_eff`; imaginary part only diagnostic | Pass |
| No compartment-specific D/T2/flow/vascular term | Random proposal uses one `cfg.sigma`; core forward model has no such term. | Pass in reference forward path |
| Trace-averaged data convention | `scripts/fit_data.py:629-635` only warns from b-vectors; `727-733` averages all shell directions. | Fail: no trace/single-direction state is recorded or enforced. |
| Unit conversion for ADC | `madi/signal.py:169-176`: `b_int=b_values[idx]/1e6`; `-log(S)/b_int` | Pass; it divides by b. |
| Vector storage/order | Writer `madi/library.py:267-280`; reader `306-331`; pair-major index `399-448` | Pass; independently verified from the file and slim derivative. |
| Library topology | Generic builder `madi/library.py:224-243` creates a full cross-product then masks by `v_i`; dense preset `scripts/fit_data.py:131-136` | Fail against MADI-II hyperbola topology. |
| Reference inverse problem | `madi/library.py:494-541`: default `log_space=False`, `dists=m2+l2-2*measured@lib_m.T`, `np.argmin`; CLI defaults at `scripts/fit_data.py:844-889,957-965` | Pass for fixed-S0 linear exhaustive MAP. |
| b=0 and library/data reconciliation | Input builder omits b=0 from `fit_triples` (`scripts/fit_data.py:645-651`) and normalizes at `727-733`; reader uses nearest stored b/pair, no interpolation (`library.py:399-448`). | b=0 omission is harmless to unnormalised sum-of-squares but differs from any per-point normalized cost.  Nearest snapping is a documented non-paper choice. |
| Derived maps / voxel-first aggregation | Current MAP output saves only kio, rho, V, residual (`scripts/fit_data.py:1441-1449`). | Missing: no current-code `v_i`, `k_ioV`, `k_ioVrho`, or `0.042*k_ioV` map/aggregation path. |

### Step-distribution caveat

The supplied prompt calls out a fixed-step mapping.  The checked local MADI-I
supporting information instead describes a three-dimensional normal proposal,
which matches the code's `sigma=sqrt(2D0*ts)` convention.  The code also uses
the paper's RMS-length Eq. 3 factor `sqrt(D0/(6*ts))=22.3607 um/ms`.

Those two facts are not automatically physically identical to a planar
continuum-flux derivation.  If a fixed isotropic step of RMS length is used,
the leading planar crossing flux is 1.5 times `D/l_rms`; for a per-axis
Gaussian it is `sqrt(6/pi)=1.382` times `D/l_rms`.  These are leading-order
interface estimates, not measured output biases.  Thus the code matches the
paper's written Eq. 3 and the local SI's Gaussian proposal, but the original
source implementation would be needed to resolve the paper/SI convention
unambiguously.

## 2. `madi_dense_universal.npz` audit

### Direct file structure

The file is 10,121,602,990 bytes and contains exactly these nine uncompressed
NPY members:

| Array | Shape | dtype | Meaning / inferred units |
|---|---:|---|---|
| `kios` | `(40645,)` | float64 | `k_io` [s^-1] |
| `rhos` | `(40645,)` | float64 | cells/uL |
| `Vs` | `(40645,)` | float64 | pL/cell |
| `vectors` | `(40645, 31125)` | float64 | pair-major then b-major `S/S0` |
| `pair_deltas` | `(1245,)` | float64 | delta [ms] |
| `pair_Deltas` | `(1245,)` | float64 | Delta [ms] |
| `b_values` | `(25,)` | float64 | s/mm^2 |
| `n_b` | scalar | int64 | 25 |
| `h_ms` | scalar | float64 | 1 ms |

The timing grid is exactly all valid triangular pairs from delta 1--30 ms and
Delta 1--50 ms plus 55, 60, 65, 70, 75, 80 ms (`Delta >= delta`), for 1,245
pairs.  `b_values` is exactly 0, 500, ..., 12,000 s/mm2.  `(20,50)` is pair
923: its b=1000,1500,2000,2500 columns are 23,077--23,080.  The independent
slim derivative records these same source columns and its first 100 rows match
the memory-mapped production values bit-for-bit.  Writer, reader, and file
therefore agree on axis ordering and b units.

### Parameter topology and coverage

`k_io` has 55 values: 1--50 by one, then 60,70,80,90,100.  There is no zero
entry.  There are 100 rho values from 100,000 to 3,000,000 and 99 surviving V
values from 0.189899 to 9.0 pL.  There are 739 unique `(rho,V)` pairs, each
with all 55 kio values (`739*55=40,645`), and all values obey
`0.400790 <= rho*V*1e-6 <= 0.948817`.

This is a masked independent dense grid, not the reported 20 `v_i` hyperbolae:
it has **739** distinct `v_i` values.  It also cannot represent `rho=0`,
`V=0`, `k_io=0`, the paper's rho range below 100,000 or above 3,000,000, the
paper's V below 0.1899 or above 9 pL, or `v_i` above 0.9488.  Consequently it
cannot supply an identical-input reference output for several published
parameter sets.

V spacing is 0.089899 pL everywhere, versus the paper's quoted approximately
0.05 pL below 1--2 pL and approximately 0.2 pL above 8--9 pL.  rho spacing is
29,292.93 cells/uL.  kio spacing is 1 s^-1 through 50, then 10 s^-1.
The fitter does nearest-entry matching; it does not interpolate this grid.

All 128 expected shards `000` through `127` exist.  Their metadata match the
production file; their parameter-key union equals it exactly; there are no
duplicate triplets or holes.  The shard count pattern is 99 shards of 330 rows
and 29 shards of 275 rows.

The checked-in reproduction launcher has a separate provenance defect:
`scripts/build_lib.sbatch:33` ends `--vi-max 0.95 \ ` (backslash followed by a
space).  In POSIX shell this does not continue the command, so the next
`--verbose true` line is a separate command.  The completed shards are
structurally self-consistent, but the artifact cannot be certified as having
been generated by that launcher exactly as it now stands.

### Signal integrity

Every b=0 value is exactly 1.0, and there are no NaN or infinite values.  There
are 322,340 negative stored samples; the minimum is `-8.2548788e-4`.  There
are 1,037,247 positive adjacent-b increments, maximum `1.2428258e-4`, but
none occurs while the preceding signal is at or above 0.015.  Thus the stated
trust-region monotonicity condition passes, while a strict physical
non-negativity condition fails at the low-signal noise floor.

### Universalization and finite-delta result

At build time each walker accumulates a transient float64
`Y(t)=integral(x(t)dt)` on the 1-ms grid.  That is sufficient to construct the
finite rectangular-lobe PGSE moment for every stored grid pair:

`dM = Y(delta) + Y(Delta) - Y(Delta+delta)`.

The NPZ file stores **only final signal vectors**, not `Y`, a propagator, or
walker moments.  It can therefore not reconstruct arbitrary new delta/Delta
or waveform values after build; the reader snaps to nearest stored values.  On
the stored integer-ms rectangular-lobe grid it is mathematically sufficient
for that finite-lobe model (apart from Monte-Carlo and trapezoid discretization
error).  It is not sufficient for the prompt's endpoint-displacement reference
physics.

Using the same reduced trajectories and geometry, finite-lobe and narrow-pulse
signals were compared directly:

| Timing | b [s/mm2] | finite-lobe S | narrow-pulse S | relative difference |
|---|---:|---:|---:|---:|
| 20/50 ms | 500 | 0.6968 | 0.6599 | 5.6% |
| 20/50 ms | 1,000 | 0.5203 | 0.4624 | 12.5% |
| 20/50 ms | 2,000 | 0.3365 | 0.2591 | 29.9% |
| 20/50 ms | 4,000 | 0.1991 | 0.1112 | 79.0% |
| 20/50 ms | 6,000 | 0.1421 | 0.0433 | 228.0% |
| 7/25 ms | 500 | 0.6701 | 0.6333 | 5.8% |
| 7/25 ms | 1,000 | 0.4790 | 0.4191 | 14.3% |
| 7/25 ms | 2,000 | 0.2809 | 0.2044 | 37.4% |
| 7/25 ms | 4,000 | 0.1267 | 0.0707 | 79.3% |
| 7/25 ms | 6,000 | 0.0735 | 0.0350 | 109.9% |

The test had `rho=700,000`, `V=1.0`, `k_io=20`, 1,024 walkers, and no escapes.
The difference is many nominal standard errors above b=1000.  The universal
refactor does reproduce the previous code's finite-lobe `m1-m2` definition by
algebra, but that definition is itself not the narrow-pulse paper reference
specified for this audit.

## 3. Geometry, boundary, and random-walk findings

### What passes

- Seed count is Poisson and positions are uniform in an enclosing source box.
  There is no relaxation step.
- `Omega_pop=[-pop_margin,L+pop_margin]^3`, strictly larger than `Omega_sim`.
  The default walker source is the interior `[buffer,L-buffer]^3`.
- Failed crossing uses exact positional rejection, not specular reflection.
- Both CUDA and CPU paths use `m=1` for intra/extra change and `m=2` for an
  intra-to-different-intra label; accepted probability is `p_p**m`.
- The quick symmetric-exchange occupancy diagnostic passed for the CPU path:
  target `v_i=.70`, `k_io=20`, `p_p=.0015595`, 512 walkers, 80 ms.  Occupancy
  ranged 0.6641--0.7344, mean 0.6981; the maximum departure 0.0359 was below
  the 5-sigma binomial allowance (`sigma=0.0203`) used by the test.

### Demonstrated geometry failures or gaps

1. **High-v_i lookup clamp and false label.**  The L=180 current lookup cache
   spans only `v_i=0.150..0.784`.  Requests 0.80, 0.85, 0.90, and 0.94 all use
   identical `alpha*=0.519044 um` and `mean_AV=0.544552 um^-1` at rho 500,000,
   while `Ensemble.vi` remains the requested value.  A direct volume sample
   measured about 0.782.  At target 0.90 that is -0.118 absolute / -13.1%
   relative; at target .94 it is -0.158 / -16.8%.  The artifact has no lookup
   table, alpha, realized vi, or build configuration, so its high-v_i physical
   geometry cannot be verified.
2. **`mean_AV` is not actual-ensemble `mean(A_i/V_i)`.**  It comes from a
   25-point, one-realization lookup and is a 5--95% trimmed mean.  The actual
   simulation ensemble is sampled later and does not calculate its own A/V.
   This fails the requested actual-geometry inversion condition, though it is
   conceptually an average of ratios rather than `mean(A)/mean(V)`.
3. **No retained A/V, A, V, or annulus statistics.**  The artifact cannot
   provide the requested functions of v_i.  Current-code L=180 diagnostics at
   rho 500,000 gave `(target v_i, mean_AV, alpha*)` of
   `(0.40,.759434,1.860499)`, `(0.70,.573529,.752757)`, and
   `(>=.80,.544552,.519044)`.  The annulus median equals alpha*; its standard
   deviations were .1993, .0192, and .0054 um, respectively.  No production
   statistic establishes the actual table used for the 300-um dense build.
4. **Cached-grid classifier error.**  At the representative 0.70 geometry,
   CUDA-style reranking of the cached pair has 0.476% cell-label and 1.166%
   compartment disagreement relative to exact KD-tree labels.  Current CPU
   fallback does not rerank, giving 5.579% and 1.169%.  For one 1-us proposal,
   true crossing fraction was 2.351%, grid-emulated fraction 2.555%, and
   membrane count disagreed in 0.826% of all proposals.  Exact KD-tree k=2
   agrees with the first two of k=8; the error is the cache, not the k=2
   Voronoi fact.
5. **Extreme requested regimes are not testable.**  The artifact's smallest V
   is .1899 pL and its apparent physical v_i cannot exceed the lookup limit in
   the diagnostic.  It does not reach the paper's .0113 pL or .994 v_i, so the
   requested multi-membrane / thin-interstitium stress test is not covered.
6. **Escape handling is a production risk.**  A walker that leaves is frozen
   and subsequently dropped.  `run_walk_Y` checks `max_escape_frac`, but the
   production reduction path (`walker_gpu.py:524-556`) silently drops it and
   only returns an in-memory count.  The NPZ has no escape counts.  Since all
   signal columns discard walkers that escape anywhere in the full 128-ms
   walk, an escape after a particular column's encoding time can bias that
   earlier column too.  No periodic unwrapping exists.
7. **CPU/GPU equivalence has not been shown.**  CUDA is unavailable in this
   environment.  Static inspection already proves the classifier discrepancy;
   CPU and GPU use different RNG families as well.  The three Cartesian axes
   are harvested as samples from the same paths/geometry, not separate
   ensembles.

The source-domain margin itself was spot-checked by extending a synthetic
Poisson cloud: at production rho >=100,000 no nearest-two labels changed in
the sampled interior source.  At the paper's low rho=4,400, 0.65--1.71% of
points near the simulation boundary changed, although sampled interior-source
points did not.  This reinforces that the production range is narrower than
the paper range.

### Part-5 bug-class checklist

| Check | Disposition and proving test/evidence |
|---|---|
| Encompassing source domain / displacement wrapping | Source domain is larger and walkers are not wrapped.  Extension spot-check above passes in the production-density range; no unwrapping is needed.  Exit dropping remains an unguarded production bias path. |
| Two-nearest KD-tree correctness | Exact KD-tree k=2 equals first two of k=8.  Cached-grid GPU emulation fails exact classification and membrane-count tests (strict XFAIL). |
| Contraction, A/V, and annulus distributions | Rule is explicit, but required actual-ensemble A/V/A/V statistics are absent.  Direct diagnostic exposes high-v_i clamping. |
| Poisson versus lattice | Pass: `rng.poisson` count and uniform seed positions; no relaxation/lattice code. |
| Step adequacy and m distribution | `p_p**m` is implemented.  Required extreme `.0113 pL/.994` coverage is absent; cached-grid m errors are quantified above. |
| Rejection versus reflection | Pass: failed crossing reverts to old position in both paths. |
| Uniform-volume initial placement | Pass in source: walkers are initialized with `rng.uniform(lo,hi,(N,3))`; no forced compartment split. |
| Compartment-specific diffusivities | Pass in reference forward path: one `cfg.sigma` is used for all walkers/states. |
| Batch/RNG independence and CPU/GPU equivalence | Fixed seeds make runs reproducible, but common random numbers intentionally correlate rho/V entries.  CPU/GPU statistical equivalence was not run because CUDA is unavailable; their classifiers already differ. |
| Float precision of phase/moment | Pass: Y, moments, coefficients, and reductions are float64. |
| Signal monotonicity | Pass only down to .015; strict nonnegative signal XFAIL documents the low-signal floor. |
| S0/reference contaminants | b0 is exactly 1 in the artifact; reference CLI defaults leave Rician and free-S0 off. |
| Simulation/fit b consistency | Both use stored b values in s/mm2, but input data are snapped within 30 s/mm2 and no vendor-b definition audit is recorded. |
| NPZ axes/holes/NaNs/kio=0 | Axes, shard coverage, and finite values pass.  kio=0 is missing. |
| Universal finite-delta sufficiency | Pass only for stored finite rectangular-lobe columns; fails narrow-pulse reference and cannot synthesize new timings after load. |
| Universal versus old hardcoded PGSE | Algebraically the code preserves its old finite-lobe `m1-m2`; an independent archived hardcoded simulation was not available as a production regression oracle. |
| kio/grid granularity | Actual 55-value grid and V/rho spacings are audited; it does not match paper spacing/range and fitter does no interpolation. |
| Exhaustive reference search | Pass: full candidate matrix and `np.argmin`; no PCA, KD-tree, or early exit. |
| Acellular masking | Fail: mask controls spatial inclusion only.  There is no T2-guided acellular exclusion and no free-water row; unmasked acellular voxels edge-fit. |
| Alternative-fitter contamination | Reference defaults do not call Rician, free-S0, Bayes, or NNLS.  Those paths are opt-in flags/methods. |
| Determinism | CPU reference matcher repeated bit-for-bit on a synthetic library.  A full production-map duplicate run is indeterminate because no matching provenance-compatible input map was supplied. |

## 4. Part-2 regression table

| Target | Result | Observed value / uncertainty | Verdict |
|---|---|---|---|
| Free-water `S=exp(-bD0)` | Formula/path is correct for a homogeneous finite-lobe fluid; a previous local validation document reports a GPU pass.  Production has no free-water entry. | `D0=3e-3 mm2/s`; no independent production-MC CI available | Partial pass |
| Einstein `mean(d2)=6D0t` | Follows directly from per-axis `sigma=sqrt(2D0ts)`; fast audit test passes. | RMS=.134164 um at 1 us | Pass |
| `rho,V -> 0` free-water limit | Production min `v_i=.400790`; no zero entry. | Not evaluable from artifact | Fail coverage |
| MADI-I Fig. 3 `rho=781k,V=1,kio=0` | No kio=0 production row.  Existing `analysis/test_drms_curves.py` is not runnable as written because it passes obsolete `n_steps=` to `SimConfig`. | No valid regression result | Indeterminate |
| Well-mixed timing scaling | No implementation/regression computes this timing criterion. | Expected 2 ms / .7 ms not tested | Indeterminate |
| kio monotonicity | At b=1000/2000, 0/39,906 increasing-S violations; at b=4000, 141/39,906.  Kurtosis-fit D has 8/39,906 wrong-sign comparisons. | Artifact exact counts; no CI | Pass through trust range; minor high-b MC failures |
| rho monotonicity | At b=1000/2000, 0/35,200 wrong-sign; at b=4000, 7/35,200.  D has 1/35,200 wrong-sign. | Artifact exact counts | Pass through trust range |
| V monotonicity | Raw signal wrong-sign counts are 23, 153, 1,702 / 35,145 at b=1000,2000,4000; 5,115 additional exact ties are caused by collapsed/clamped geometry. D has 265 wrong-sign and 5,115 tied comparisons. | Artifact exact counts | Partial fail |
| Kurtosis positivity | Fitting b=0..4000 at 20/50 gives D=.1732..2.5502 and K=.1937..5.0346; all 40,645 K are positive. | Deterministic artifact result | Pass |
| K low-kio behavior | Median K: kio 1=1.453, 10=.987, 20=.745, 30=.606, 100=.295. | No kio=0 / free-water limiting test | Partial pass |
| Published parameter decays | None of the required parameter triplets exists exactly.  Nearest-entry D values: brain GM .777, lesion .942, NA prostate 1.624, SW620 .512 um2/ms; pure water absent. | Grid quantization dominates; no paper curve digitization in repo | Fail / non-comparable |
| Table-2 50,000 realization precision | Ran exact count through fixed-S0, linear, Gaussian S/N=50 matcher using bundled 4-shell slim production derivative. Cortex returned `(kio,rho,V)=(31,715k,1.089)` vs paper `(11,160k,4.7)`; WM `(18,686k,1.089)` vs `(22,520k,1.2)`. | Gaussian convention, no Rician correction; non-comparable 4-shell/dense-grid protocol | Fail to reproduce; diagnostic only |
| Whole-map out-of-range / ROI contrasts | Provisional co-located universal-map check: `madi_output_glioma_v3.0/map` has 123,601 nonzero voxels, matching its sibling universal-library metadata.  Its median v_i=.5121 and 49.04% lie below .5; map-fits0 median=.6123 and 36.78% below .5. | Both stay within the *stored* .4--.95 grid; fixed-S0 rho hits lower edge in 46.30% and kio upper edge in 18.20% | Fail published-brain-range diagnostic; tumor ROI contrast remains indeterminate |

For the 50,000-realization diagnostic, cortical V was highly multimodal; its
most common returns were .2798 pL (5,000/50,000) and .3697 pL (4,377/50,000).
WM V most often returned .1899 pL (10,864/50,000).  These are consequences of
the current protocol/grid, not estimates of MADI-II uncertainty.

The following requested numerical targets remain explicitly unvalidated rather
than assumed: the Fig.-3 intra/extracellular displacement curves, `kappa_Dio`
zeros, well-mixed onset, high-q exchange limit, all MADI-III rat controls, and
whole-map tumor ROI contrast directions.  The current production artifact
cannot cover several required inputs, and the checked-in Fig.-3 script is
stale against the present `SimConfig` API.  The provisional v3.0 map folders
do not contain their own MAP metadata or ROI masks; the universal provenance
comes from sibling Bayesian metadata with the same 123,601 voxels.  A result
from a different, reduced-grid simulation would not answer the identical-input
question.

### Derived quantities and aggregation

The identity `v_i=rho*V*1e-6` is used correctly wherever candidates are
filtered.  Current fitting code, however, writes only `kio_map`, `rho_map`,
`V_map`, and residual.  A source search found no implementation of `kioV`,
`kioVrho`, `0.042*kioV`, or a current ROI aggregation function.  The older
output directory contains files named `kio-V_map` and `kio-rho-V_map`, but no
current source path produces them, so their aggregation order cannot be
audited.  This is a missing verification path, not evidence that multiplying
ROI medians is safe.

## 5. Divergence list: differences from the requested original behavior

Ordered by demonstrated or directly bounded magnitude.

| Rank | Divergence | Quantified impact | Classification |
|---:|---|---|---|
| 1 | Finite-lobe phase instead of paper endpoint displacement | 12.5--228% relative S difference at 20/50; 14.3--109.9% at 7/25 over b=1000--6000 in same-path diagnostic | Physics divergence, not merely an addition |
| 2 | High-v_i lookup clamp retained as target label | target .90 measured about .782 (absolute -.118); target .94 also about .782 (absolute -.158) in current diagnostic | Bug / provenance failure |
| 3 | Dense masked grid replaces hyperbolae and published range | 739 rather than 20 v_i values; no kio0, no free water, rho only .1--3M, V .1899--9 | Deliberate library change that prevents identical-input equivalence |
| 4 | Cached two-seed grid misclassifies crossing states | .826% all proposed-step m mismatch at representative case, versus 2.351% true crossing attempts | Physics bug risk |
| 5 | 128-ms production walk and silent escape filtering | 60% longer than paper max; escape effect unquantified because counts are not retained | Deliberate universalization with unguarded bias path |
| 6 | Lookup/trimmed A/V rather than actual ensemble average | 25 alpha points, one reference cloud, 5--95% trimming; no artifact evidence for bias | Physics-calibration divergence, magnitude unknown |
| 7 | Nearest timing/b snapping rather than interpolation/resimulation | input shells snapped within 30 s/mm2; pairs allowed up to 1.5 ms mismatch before CLI error | Deliberate choice; sensitivity unquantified |
| 8 | Input direction convention is not represented | MADI-III’s reported one-vs-three-direction parameter shifts can be as large as ADC +18.1%, kioV +16.9%, vi -8.4% | Pipeline semantics gap |
| 9 | MC output has low-signal negatives / nonmonotonic samples | 322,340 negative values; no increases above .015 | Sampling-quality issue, localized below trust floor |
| 10 | Reference fitter behavior | Linear, fixed-S0 exhaustive argmin with deterministic first tie is available and uncontaminated when defaults are used | No divergence in reference mode |

## 6. Underdetermined choices and code choices

| Underdetermined item | Code choice | Consequence / what SI would resolve it |
|---|---|---|
| Exact contraction rule / alpha_i limit | `min(alpha_star,.95*d_nn/2)` with an alpha lookup interpolated in v_i | Directly changes A/V and hence p-to-kio calibration.  Need original geometry generator or SI pseudocode plus raw calibration table. |
| Geometry averaging | 5--95% trimmed mean of per-cell A/V in one reference realization | Paper calls for an ensemble average of ratios; it does not specify trimming.  Need raw A_i,V_i distribution or source implementation. |
| Step distribution versus Eq. 3 interpretation | Gaussian components, but RMS Eq. 3 permeability factor | Local SI supports Gaussian; fixed-step wording in the prompt conflicts.  Need original crossing-calibration derivation/source. |
| PGSE finite-lobe treatment | True rectangular-lobe integral of position | Prompt specifies narrow-pulse displacement.  Need original phase-loop pseudocode/source to resolve whether paper prose or code controlled publication library. |
| Boundary policy after an exit | Freeze, then exclude from all columns; no wrap/unwrapping | SI says exits should halt/error.  Need original handling and per-column survival policy. |
| Data direction convention | Arithmetic shell average, warning-only tensor-spread check | Need acquisition direction list and an explicit trace/powder-average contract. |
| Noise in MADI-II precision study | Audit used Gaussian S/N=50 and no Rician correction in reference mode | Paper does not say Gaussian/Rician.  Need SI noise generator and whether b0 was noisy/renormalized. |
| b-grid reconciliation | Nearest stored column; no interpolation | Paper does not specify.  Need original matcher source or SI wording. |
| Tie behavior | NumPy `argmin`, first library row | Paper does not specify.  Need source to know whether ordering differed. |

## 7. Known paper inconsistencies checked

- The code uses the correct `rho*V*1e-6` identity, so it is not calibrated to
  the inconsistent MADI-I Figure-1 caption.
- It uses the stated 134-nm RMS value, not the numerically inconsistent
  printed `sqrt(12 D t/pi)` expression.
- The current build minimum `v_i=.4` could represent the NA-prostate .44
  region in principle; it does not enforce the reported .5 floor.
- No ATP-flux conversion is implemented, so neither the pmol/fmol inconsistency
  is inherited.
- ADC divides by b correctly.
- No panel-label citation for the MADI-II kurtosis figure was found in core
  physics code.
- No code treats a median-of-medians table row as an algebraic v_i identity.

## 8. Minimal reproducible tests

Fast checks:

```bash
PYTHONPATH=. pytest -q tests/physics_audit -m 'not slow'
```

Full artifact/shard scan (about four minutes in this environment):

```bash
PYTHONPATH=. MADI_AUDIT_SLOW=1 \
  pytest -q tests/physics_audit/test_production_library.py -m slow
```

Real-geometry checks (uses cached L=180 lookup table; set
`MADI_AUDIT_ALLOW_LOOKUP_BUILD=1` to build it on a clean host):

```bash
PYTHONPATH=. MADI_AUDIT_GEOMETRY=1 \
  pytest -q tests/physics_audit/test_geometry_diagnostics.py -m slow
```

Reduced same-path finite-lobe versus narrow-pulse comparison at both MADI-I
and MADI-III timings:

```bash
PYTHONPATH=. MADI_AUDIT_GEOMETRY=1 \
  pytest -q tests/physics_audit/test_finite_delta_reference.py -m slow
```

The exact 50,000-realization current-library control is opt-in:

```bash
PYTHONPATH=. MADI_AUDIT_SLOW=1 \
  pytest -q tests/physics_audit/test_precision_protocol.py
```

Expected strict XFAILs are intentional audit alarms: low-signal negative
vectors, cached-grid classification, cached-grid membrane counts, high-v_i
label fidelity, and the non-comparable Table-2 precision reproduction.  An
XPASS means the corresponding divergence was removed and the written audit
must be re-evaluated.

## 9. Required evidence to close remaining indeterminate items

To decide exact equivalence rather than infer it, retain or obtain:

1. The original simulation phase loop and the MADI-I/MADI-II supporting source
   specifying finite versus narrow pulse treatment.
2. The production build's cached A/V table, config, alpha/realized-v_i values,
   p_p values, per-entry escape counts, and independent ensemble variance.
3. A canonical hyperbola-structured library with kio=0 and free-water entries,
   at the paper's b grid and acquisition timings.
4. The exact noise generator, b0 handling, and acquisition direction list for
   the Table-2 precision experiment.
5. Input DWI, masks, ROI definitions, and fit metadata produced with this
   universal file for the requested whole-map checks.
