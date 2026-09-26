# Adaptive common-TE protocol analysis

This is an independent Windows workflow. It reads the complete stored Fisher
substrate and never changes the simulator, universal library, historical
notebooks, reports, caches, documentation, or `analysis/protocol_statistics.xlsx`.
The expanded export preserves that workbook's useful protocol, acquisition,
matrix, eigenvector, uncertainty, physical-bound, and definition fields; it
separates aggregate statistics from node-level records.

## Run on Windows

From the repository root:

```powershell
python -m analysis.adaptive_protocol.run
node --max-old-space-size=10000 analysis/adaptive_protocol/export_workbook.mjs analysis/adaptive_protocol/outputs/windows_run
python -m pytest analysis/adaptive_protocol/test_workflow.py -q
```

`run_windows.ps1` runs the first two steps using the existing Windows Python and
the bundled Windows Node runtime. It creates a dependency junction only inside
this analysis directory if needed; it does not install packages or change any
other project/environment. Tested scientific packages: NumPy 2.1.0, SciPy
1.14.1, Numba 0.63.1, pandas 2.2.2, Matplotlib 3.9.2. The existing MADI import
requires SciPy/Numba even though this analysis uses no GPU or simulation.
The Excel author uses the bundled `@oai/artifact-tool` JavaScript package.

The default `runs` path in `config.json` is a Windows UNC path into the existing
Ubuntu filesystem. WSL supplies files only; **all Python and Node execution is
on Windows**. Windows must have permission to read that UNC path. You can later
copy the source files to Windows and point `runs` at their new parent directory.
No copy or migration is performed by this workflow. `library` already points at
the Windows project. All source NumPy maps are opened read-only.

Additional commands:

```powershell
python -m analysis.adaptive_protocol.run --config analysis/adaptive_protocol/config.json
python -m analysis.adaptive_protocol.run --figures-only
```

Close Excel before replacing an open workbook. One process may write an output
directory at a time; a lock guards the analysis. `--skip-validation` is for
development and explicitly marks its result unvalidated. The standard command
always runs the integration checks. The unit tests use only synthetic data.

## Configure the experiment

Edit `config.json`. Defaults are an explicit initial experiment, not a claim
about every possible scanner or tissue domain:

| Setting | Default |
|---|---|
| Total volume budgets | 16 and 128; each includes 2 acquired true-b0 volumes |
| Noise | SNR_ref=50 at TE_ref=0 s, inherited from the combined notebook |
| TE | One common TE = max(delta + Delta) + 0.014 s, including b0 |
| T2 | 0.040 s; alternate TE modes/T2/offsets are rejected |
| Noise SD | exp((TE_A - TE_ref)/T2)/SNR_ref, for each volume |
| Tissue coordinates | (ln rho, ln V, k_io), with named matrix entries |
| Relative k_io SD | SD_kio / max(k_io, 5 s^-1) |
| Scaling | D=diag(1,1,max(k_io,5)); F_scaled=D F_native D |
| Derivative/debias | Existing Phase-1 k=1 J; Phase-2 endpoint-only MC diagonal correction |
| Objective | Marginal-S0 robust minimax; fixed-S0 is always reported alongside |
| Node population | All 233 complete-stencil rho/V pairs at k_io=5,20,40; 699 nodes |
| DW design | 47 stored timing pairs × 24 positive b values; 1,128 columns |
| Timing choices | delta=2,4,8,12,20,30 ms; Delta=10,15,20,25,30,40,50,60,80 ms, where stored |
| b choices | Every 500 s/mm² from 500 through 12,000 |
| Hardware/trust/Rician | No gradient ceiling; trust=0 and Rician=0 (disabled) |
| Repeats | Integer, minimum 1 per selected column; maximum budget per column |
| Diversity limits | At most 12 selected columns and 4 timing pairs, including b0 |
| Minimum coverage | 0.51 of the entire declared node population |
| Search | Seed 20260925 (+ budget), 4 starts × 450 proposals, 2 local passes |

The initial node set includes a high-exchange slice; it is **not the full 11,417
complete-stencil node population**. Set `nodes.k_io_values` to `null` for the
entire population, or provide `nodes.indices` for exact canonical triples.
The requested node `[250000,4,20]` is an annotation used to select the nearest
evaluated node for acquisition-level derivative details, not an interpolation
or a replacement for the declared population. Canonical and realized
coordinates, entry indices, and realized derivative spacings are exported.

Set timing/b lists to `null` to admit all stored values. Optional inclusive
`delta_bounds_ms`, `Delta_bounds_ms`, `b_bounds_s_mm2`, and `G_max_T_m` further
restrict the **evaluation design**, never source extraction. For example,
`G_max_T_m=0.08` and `0.3` are separate clinical/research scenarios. Use separate
config/output directories for such comparisons. The default unlimited-gradient
result does not assert hardware feasibility. Large node/column searches need
more memory and runtime; the source cache is still full-domain and read-only.

`protocols` accepts named explicit protocols:

```json
{"name":"reference", "columns":[[4,15,0,2],[4,15,1000,4],[4,25,2500,5],[4,40,4000,5]]}
```

Each row is `[delta_ms,Delta_ms,b_s_mm2,integer_repeats]`. Duplicate columns
canonicalize by summing repeats; zero counts are omitted. Missing columns,
noninteger/negative counts, hardware violations and inconsistent budgets fail
explicitly. User protocols must fit the declared design. A true b0 reference
is an acquired volume; there is no uncounted amplitude prior. Both models use
the same acquisition and measurement budget.

The optional Rician basis is explicit: `phase2_repeated_mean` preserves the
executed Phase-2 averaging-aware convention; `single_measurement` uses the
combined notebook's convention. Neither implements a Rician likelihood. With
the default threshold zero, this distinction has no numerical effect. Masked
volumes still consume budget. Debiasing is linear in repetitions and does not
pretend repeated acquisitions reduce Monte Carlo derivative uncertainty.

## Data reuse and numerical authority

The source manifest must declare all 31,125 stored columns. `ColumnDomain` and
`require_columns` enforce identities; no nearest-column substitutions occur.
`build_node_table` supplies authoritative shared-stencil nodes and spacings.
Existing `J_rho_k1.npy`, `J_V_k1.npy`, `J_k_io_k1.npy` are read directly, cast to
float64. Signals and endpoint variances come from the existing
`vectors_selected_T.npy` and `signal_variance_selected_T.npy`.

Endpoint-only derivative variance is newly assembled with the authoritative
`derivative_variance` function because it is not already cached across all
columns. The existing diagnostic `VarJ` contains a CRN covariance correction
on only 200 columns and is not interchangeable with the uniform executed
Phase-2 endpoint-only convention. We do not mix those two variance estimators.
This correction can be conservative; finite-difference truncation remains.
No Monte Carlo signals or production derivatives are recomputed.

Complete weighted protocol information is summed first. The authoritative
packed Schur complement and positive-minor/inverse-diagonal functions are
reused. Scaled eigenvalues also gate numerical positive definiteness at
`pd_rtol=1e-12`. Covariance comes from Cholesky inversion only after that gate.
There is no ridge, pseudoinverse, or time-normalized metric.

The default objective is the maximum of the three per-parameter medians across
**all declared nodes**, assigning +infinity to invalid nodes. Identifiable-only
means are labeled secondary diagnostics. Configurable alternatives minimize
median scaled trace (A), median negative scaled log determinant (D), or median
inverse smallest scaled eigenvalue (E), each also including infinite invalid
losses. Objectives are never silently combined. Median binding node(s) and the
worst node are different records. When an A/D/E objective is selected, the
binding-parameter fields still describe the secondary robust score.

## Search and persistence

Seeds cover every allowed single timing and a set of simple two-timing
protocols. The best single-timing seeds receive local allocation/replacement
refinement. Multistart annealing explores whole-column replacements,
within-timing b changes, whole-timing-group moves, and integer repeat transfers
that can add/remove columns. Strict-improvement local passes refine results.
The next configured budget also receives a rounded, budget-matched version of
the previous winner. Search is finite: **best found**, not a certified global
optimum. The single-timing baseline is best found among its reported seeds and
refinements; simple multi is the best declared balanced two-timing baseline.

SQLite caches candidate scores by canonical column/repetition specification,
evaluation/design/noise settings, complete source manifest fingerprints and
authoritative/evaluator code hashes. Source file sizes and modification times
are recorded; manifests have SHA-256 hashes. Every proposal, including invalid
ones and cache hits, has a reproducible history row. Counts and stopping rules
are in `run_report.json`. Rerunning identical code/config preserves IDs and
upserts records rather than duplicating them. Changed settings/code get new
keys. CSV updates and workbook replacement are atomic; the workbook is a
static numerical snapshot, so changing Excel cells does not rerun the optimizer.

## Outputs

All deliverables are in `outputs/windows_run/`:

| Workbook sheet / CSV | Interpretation |
|---|---|
| `protocol_summary` | Selected top-k and baselines; one row per protocol/evaluation/amplitude model, population aggregates, noise/settings/provenance |
| `top_protocols` | Ranked distinct feasible protocols for each budget |
| `node_metrics` | Every declared node of each exported protocol, separate fixed/marginal rows, full matrices/eigensystems, SD and variance bounds, reasons for failure |
| `acquisition_columns` | Normalized integer-repetition specification; signal/J/VarJ annotations at the declared nearest node |
| `optimization_history` | Every optimizer/sweep candidate request, canonical protocol, scores, coverage, feasibility, cache hits and progress |
| `protocol_sweeps` | Controlled one-group timing, one-column b, repeat-transfer sweeps, and explicitly separate complete timing-diversity comparisons |
| `definitions_config` | Definitions, units, source/domain manifests, complete config, validation and run report (long JSON losslessly split into numbered cells) |

Node rows are exported for top-k/baseline/user protocols. The candidate history
contains every tested protocol and its score/coverage; candidates outside the
selected reporting set do not inflate `node_metrics`. Native trace has mixed
coordinate units; scaled trace is the dimensionless comparison. `+inf` is
literal text for invalid uncertainty/loss; blank covariance cells mean invalid,
never zero. Physical SDs use a local first-order coordinate transformation.

Figures are generated **only from these CSV snapshots**, grouped into
`parameter_space`, `protocol_space`, `comparison`, and `optimization`. PNGs are
300 dpi; PDF and SVG versions retain vector content. Each figure has embedded
noise/config metadata and a JSON sidecar. Heatmaps retain all fixed columns and
recompute common TE from the complete candidate. Gray parameter-map cells are
invalid or outside the evaluation grid; the coverage map distinguishes valid
and invalid available nodes. Color limits use declared 2nd/98th percentiles for
positive-log maps, with extended colorbars. Distribution figures show finite
nodes only and explicitly label coverage; ranking still includes failures.

## Verification

`validation.json` records repetition scaling for both S0 models, whole-protocol
adaptive TE (including retained long columns in timing sweeps), singular/non-PD
and masked invalidity, explicit 4×4 nuisance-S0 inverse agreement, an independent
local Phase-2 cached-data calculation under the same adaptive noise, and an
84-allocation exhaustive oracle compared with the same discrete optimizer.
The local Phase-2 check is allowed to call the authoritative endpoint arithmetic
as an independent validation only; production reads existing derivatives.
Expected small differences arise from stored float32 J versus Phase-2 float64
endpoint arithmetic. Historical Phase-2 TE/T2 assumptions are not rerun or
relabelled as this adaptive experiment.

`test_workflow.py` additionally covers malformed repetitions, whole-FIM versus
single-column identifiability, absent amplitude information, scaling/units,
infinite-tail medians, masks/budget, score-cache keys, and idempotent exports.
`workbook_audit.json` and per-sheet previews document spreadsheet verification.
After exporting, `python -m analysis.adaptive_protocol.verify_products
--check-sources` independently checks the CSV scores, budgets, TE, matrix
inverses, workbook row/column counts, figure metadata, and unchanged source
files. Omit `--check-sources` if the source filesystem is not currently available.
