# Fisher / CRLB Phase 0–1 framework

## Verdict

> **Carry-forward, 2026-09-06.** A second defect was found after this record was
> written, in the Phase-1 *column domain* rather than in any Phase-1 number:
> extraction was restricted to the research (300 mT/m) feasibility mask, so the
> derivative fields held 24,111 of the library's 31,125 stored columns. Every
> number in this document was computed correctly on the columns it used and
> **stands as executed**; the fields were re-materialised at full width from the
> same complete library and reproduce bit-for-bit on the overlap. One reported
> statistic is basis-dependent by construction and therefore differs on a
> full-width re-run — the *absolute* truncation-bias RMS, which is pooled over
> every selected column; the *relative* form, which is the one §1.4 asks for,
> reproduces exactly. See the amendment log entry `2026-09-06-substrate-domain`
> at the end of this file, and
> [`fisher_domain_audit.md`](fisher_domain_audit.md).

**COMPLETE PRODUCTION ARTIFACT; PHASE 0–1 EXECUTED AND CLEAN; ONE DEFECT FOUND
AND FIXED.** `data/libraries/madi_dense_universal_remediated.npz` contains all
369 canonical cellular groups plus the free-water atom. Every tool reports
`grid_complete: true` with zero missing, duplicate, or extra groups, and zero
invalidated stencils. The shard-45 diff against the 368-group development
artifact is confined exactly to the predicted footprint; no pre-existing group
changed at all.

Three results carry beyond the framework:

1. **The `k_io` step-size handling is correct; the alarming `k_io` noise was a
   `Var(J_hat)` normalization defect, now fixed.** The derivative denominators
   read the real, non-uniform `k_io` spacing at every node. The apparent
   25–48× excess of `k_io` derivative noise over `rho` was caused by a missing
   `1/n_ensembles` on the endpoint variance terms in
   `scripts/run_fisher_phase1.py`, which inflated `beta` by a factor that
   *grows with common-random-number correlation* and therefore hit `k_io`, the
   best-correlated axis, hardest. Corrected, the full-grid worst `k_io` `beta`
   is `1.51e-4`, **3.7× below** the pre-registered projection of `5.538e-4`.
2. **`k_io` identifiability collapses above roughly 30 s⁻¹, and this is real
   physics, not a grid artifact.** Not one column at any `k_io > 30` node
   reaches the top 5% of `|dS/dk_io|` — zero out of 1,402,200 cells, from a
   region holding 38.8% of all centres.
3. **Truncation, not Monte-Carlo noise, is the binding error term** for the
   `rho` and `V` derivatives at every stencil width. `k = 1` is the right
   choice for both.

The `S0` marginalization amendment is adopted and implemented. `t_epi` is
corrected to 30 ms. Phase 2 is reframed from three named protocols to a
declared acquisition sweep. Phases 2–5 remain out of scope here.

## Implemented framework

| File | Purpose |
|---|---|
| `madi/fisher_crlb.py` | Canonical-grid manifest; incomplete-output guard; memory-bounded v5 validator; split gates; per-entry feasibility; CRN variance; Fisher primitives; **nuisance-amplitude (`S0`) block matrix and Schur complement**; **the analysis-domain contract `ColumnDomain` / `require_columns` (2026-09-06)**. |
| `madi/identifiability.py` | Selects v5 neighbours by nominal grid coordinates, uses realized log-rho/log-V denominators, and exposes debiased Fisher in `(log rho, log V, k_io)` order. |
| `scripts/verify_fisher_shards.py` | Validates a merged artifact or shard directory and writes per-group JSON. |
| `scripts/validate_free_water_fisher.py` | Runs Gate A and Gate B without materializing `vectors`. |
| `scripts/analyze_fisher_feasibility.py` | Streams gradient, per-entry trust/Rician, and TE-noise masks. **`T2` and `t_epi` defaults now come from the pre-registration, not a second local copy.** **Its output is a conditional annotation and no longer selects any extraction domain (2026-09-06).** |
| `scripts/run_fisher_phase1.py` | Streams central derivatives **over every stored column (2026-09-06)**, diagnostic `Var(J)`, and Richardson fields **with relative truncation bias**. |
| `scripts/report_s0_marginal_crlb.py` | **New.** Fixed-`S0` vs `S0`-marginalized CRLBs and the gap between them. |
| `madi/fisher_crlb_preregistration.json` | Stencils, SNRs, kappa threshold, noise/filter defaults, **amplitude model, protocol sweep, and an `amendment_log`**. |
| `tests/physics_audit/test_fisher_crlb.py` | Synthetic coverage, covariance, debiasing, kappa, feasibility, **amplitude marginalization, a `Var(J)` normalization regression guard, and the analysis-domain invariants** (24 tests). |

Every tool reconstructs the canonical grid from `make_remediation_log_grid()`. It must match `nominal_rhos`/`nominal_Vs`: v5 `rhos`/`Vs` are realized geometry provenance and deliberately differ from canonical nodes. Realized labels are used only in finite-difference denominators. Incomplete inputs show an `INCOMPLETE GRID` banner and refuse output names lacking `partial`, `incomplete`, or `smoke`.

## The complete artifact, and the shard-45 diff

The 368-group development artifact was **not** overwritten. Both files exist:

| Artifact | Entries | Groups | Bytes |
|---|---:|---:|---:|
| `madi_dense_universal_remediated_partial368.npz` | 18,769 | 368 | 15,200,860,970 |
| `madi_dense_universal_remediated.npz` | 18,820 | 369 | 15,242,165,309 |

The retained `/tmp` outputs from the original smoke test had been cleared by a
host restart, so **both sides of the diff were regenerated from the two `.npz`
artifacts using identical code and an identical `t_epi = 0` column basis.** The
regenerated partial368 numbers reproduce this document's previously recorded
values, which is itself the check that the regeneration is faithful.

### Everything outside the shard-45 footprint is unchanged

| Quantity | partial368 | complete | Note |
|---|---|---|---|
| Gate A (derivative / Fisher / CRLB abs error) | `0` / `7.105e-15` / `2.776e-17` | identical | free water untouched |
| Gate B (max signal error, nominal σ, observed σ) | `8.796e-4` / `2.155` / `3.078` | identical | identical |
| `independent_sampling_floor` | `4.082e-4` | identical | |
| `minimum_signal_high_b` | `-1.2430158468260469e-3` | identical | bit-identical to 17 digits |
| `subset_mean_max_abs_error` / `max_rel_error` | `1.290e-8` / `3.231e-3` | identical | |
| CRN contract, columns, duplicates, extras | pass / 31,125 / none / none | identical | |
| **Per-group records for all 368 pre-existing groups** | — | **0 differ** | full-record comparison |
| Feasibility derivative column selection | 24,081 columns | **identical index set** | |

The only group present in the complete artifact and absent from partial368 is
exactly `(rho_index=7, V_index=52)`. `negative_signal_count_high_b` rises
5,624,054 → 5,700,311 (+76,257), which is that group's own contribution; the
minimum is unchanged, so it introduced no new extreme.

### The invalidated stencils became populated, exactly as predicted

The previously listed invalid centres are now all valid, and the Phase-1 centre
counts grew by exactly the predicted amount:

| axis / width | previously invalidated centres | predicted new rows | observed |
|---|---|---:|---:|
| rho `k=1` | (6,52), (7,52), (8,52) | 3 × 51 = 153 | **+153** |
| rho `k=2` | (5,52), (7,52) | 2 × 51 = 102 | **+102** |
| rho `k=3`, `k=4` | none | 0 | **+0** |
| V `k=1` | (7,51), (7,52) | 2 × 51 = 102 | **+102** |
| V `k=2` | (7,50) | 1 × 51 = 51 | **+51** |
| k_io `k=1` | (7,52), all its interior k_io | 49 | **+49** |

`stencil_invalidated_nodes` is now empty for every axis and width. The `rho k=3`
and `rho k=4` fields gained no centres and are **bit-identical** across the two
runs in every audited statistic — a clean control confirming no global
perturbation. Elsewhere the audited values move only as far as adding centres to
a pooled distribution requires: `k_io k=1` `beta` max is identical (`0.1498`),
its q95 moves 0.27%, and the largest single move is `rho k=1` `beta` max
(+6.6%), from one new centre entering the top-5% population. Feasibility moves
only in per-entry *distribution* quantiles (clinical median 3,997 → 3,985;
research median 5,464 → 5,449; research q05 2,485 → 2,483), never in a
column-level count.

**No discrepancy outside the shard-45 footprint was found.**

## Finding 1: the `k_io` step size is correct; the noise was a defect

### The step-size code does read the real spacing

The `k_io` grid is non-uniform: 1 s⁻¹ from 0 to 30, then 5 s⁻¹ from 35 to 130.
`scripts/run_fisher_phase1.py` computes every denominator from the realized
labels:

```python
def _denominator(axis: str, minus: int, plus: int, rhos: np.ndarray,
                 volumes: np.ndarray, kios: np.ndarray) -> float:
    if axis == "rho":
        return float(np.log(rhos[plus]) - np.log(rhos[minus]))
    if axis == "V":
        return float(np.log(volumes[plus]) - np.log(volumes[minus]))
    return float(kios[plus] - kios[minus])
```

There is no constant `h` anywhere. The realized total steps that actually occur
are `h = 2` (10,701 centres, `k_io` 1–29), `h = 6` (369 centres, the `k_io = 30`
boundary node whose neighbours are 29 and 35), and `h = 10` (7,011 centres,
`k_io` 35–125). `Var(J)` divides by the same `denom ** 2`, and Richardson
extrapolation is only ever applied to `rho` and `V`, whose grids are uniform in
log — the `k_io` axis has a single pre-registered half-width and no `(4J_h −
J_2h)/3` is formed on it, so no uniform-spacing assumption exists there either.

Verified directly against raw library rows at five nodes spanning all three
step sizes: the stored field equals `(S(+1) − S(−1)) / (kios[+1] − kios[−1])`
with **ratio 1.000000** and differences at float32 rounding (`≤ 2.1e-8`).

| centre `k_io` | true `h` | stored `max|J|` | recomputed | ratio |
|---:|---:|---|---|---:|
| 20 | 2 | 0.00577787 | 0.00577786 | 1.000000 |
| 30 | 6 | 0.00338539 | 0.00338539 | 1.000000 |
| 35 | 10 | 0.00300027 | 0.00300027 | 1.000000 |
| 80 | 10 | 0.00110312 | 0.00110312 | 1.000000 |
| 125 | 10 | 0.000646263 | 0.000646261 | 1.000000 |

Independently, `|J|` is smooth across the spacing change (`k_io` 29 → 30 → 35
gives 0.001462 → 0.001292 → 0.001148). A denominator that ignored the coarse
spacing would make `|J|` jump discontinuously by 5× at that boundary. It does
not.

### The probe cross-check agrees

Recomputing the `k = 1` `k_io` derivative and `beta` at the stencil probe's own
node — canonical `(21, 41)`, `rho = 100,000` cells/µL, `V = 6.29626521` pL,
`k_io = {19, 20, 21}` — from the complete production library, using the correct
variance formula:

| Statistic | Probe (40 × 100,000) | Production (40 × 50,000) | Ratio | Expected |
|---|---|---|---:|---|
| all-column `beta` median | 0.00694 | 0.01214 | 1.75 | ~2× from halving walkers |
| top-5%-\|J\| max `beta` | 2.769e-4 | 2.842e-4 | 1.03 | 1×–2× |

The probe's own reallocation analysis states the bound: halving walkers scales
the independent-noise share by 2× and the ensemble-correlated share by 1×, so
any ratio in `[1, 2]` is consistent. Both observations sit inside it. **The step
size handling is correct.**

### What was actually wrong

`madi.fisher_crlb.derivative_variance` — the audited helper — divides *both*
the endpoint variances and the covariance by `n_ensembles`, because
`signal_variance` is the between-ensemble sample variance of per-ensemble means
and the build metadata states plainly that "consumer SE is
`sqrt(signal_variance / n_ensembles)`". `scripts/run_fisher_phase1.py`
reimplemented this inline for streaming and scaled the two endpoint terms by
`1/h²` alone while dividing only the covariance term by `n_ensembles`:

```python
# defective
variance_actions.setdefault(minus, []).append((work_index, row, 1.0 / denom ** 2))
variance_actions.setdefault(plus,  []).append((work_index, row, 1.0 / denom ** 2))
...
VarJ[row] = np.maximum(VarJ[row] - 2.0 * covariance / (n_ensembles * denom ** 2), 0.0)
```

The resulting inflation is `n_E (1 − r/n_E) / (1 − r)`, which **grows with the
common-random-number correlation `r`**. Because `k_io` has by far the best CRN
correlation (probe median `r = 0.961`, versus 0.361 for `rho` and 0.506 for
`V`), the correct formula has the largest cancellation there and the defect
therefore damaged `k_io` most. That is the entire explanation for `k_io` noise
appearing 25–48× `rho`'s.

| axis | probe `r` | predicted inflation | observed q95 inflation |
|---|---:|---:|---:|
| `rho` | 0.361 | 62× | 38–50× |
| `V` | 0.506 | 80× | 48–59× |
| `k_io` | 0.961 | 1001× | 313× |

(Predictions use the probe's single-location median `r`; the full grid carries a
distribution of `r`, so the ordering rather than the exact factor is the
prediction being tested.)

Fixed by using one consistent endpoint scale, with
`tests/physics_audit/test_fisher_crlb.py::test_streaming_variance_accumulation_matches_the_audited_helper`
pinning the streaming path to the audited helper. **Derivative fields `J` were
never affected** — only `Var(J_hat)`, `SNR_partial`, and `beta`.

## Finding 2: `k_io` identifiability collapses above roughly 30 s⁻¹

MADI II Figure 2g reports that only `k_io < 30 s⁻¹` lifts diffusion kurtosis
meaningfully above zero. The production grid runs to 130, so 20 of its 51 nodes
sit where the signal is expected to stop responding. It does:

| `k_io` | step `h` | median \|dS/dk_io\| | relative to `k_io = 1` | median `beta` |
|---:|---:|---|---:|---|
| 1 | 2 | 6.20e-3 | 1.000 | 9.61e-5 |
| 10 | 2 | 3.57e-3 | 0.576 | 2.19e-4 |
| 20 | 2 | 2.18e-3 | 0.351 | 5.39e-4 |
| 29 | 2 | 1.46e-3 | 0.236 | 1.24e-3 |
| 30 | 6 | 1.29e-3 | 0.208 | 4.78e-4 |
| 35 | 10 | 1.15e-3 | 0.185 | 3.38e-4 |
| 50 | 10 | 6.06e-4 | 0.098 | 1.25e-3 |
| 80 | 10 | 1.98e-4 | 0.032 | 1.16e-2 |
| 100 | 10 | 1.07e-4 | 0.017 | 3.85e-2 |
| 125 | 10 | 5.86e-5 | **0.0095** | 1.30e-1 |

The discriminating statistic, and it is not marginal:

| Region | share of centres | share of top-5%-\|J\| cells | median `beta` | cells with `beta > 1` |
|---|---:|---:|---|---:|
| `k_io ≤ 30` | 61.2% | **100.00%** | 3.70e-4 | 4.4% |
| `k_io > 30` | 38.8% | **0.00%** | 1.09e-2 | **18.0%** |

**Not one of the 1,402,200 cells at a `k_io > 30` node (7,011 centres x 200
diagnostic columns) reaches the top 5% of `|dS/dk_io|`.** The largest `|J|` anywhere in that region is `8.33e-3`, below
the global top-5% threshold of `1.374e-2`. In 18% of those cells the Monte-Carlo
noise in the derivative exceeds the derivative itself.

Stated plainly: **above roughly 30 s⁻¹ the acquisition stops carrying
information about `k_io`, and no amount of estimator cleverness recovers it.**
This is the regime containing the field's commonly cited elevated white-matter
and tumor `k_io` values, so it is a limitation of the measurement, not of this
implementation. It also has a direct methodological consequence, now
pre-registered: a minimax relative-CRLB criterion taken over an unrestricted
`k_io` range is degenerate, because the `k_io` CRLB is effectively unbounded
there for *every* candidate protocol, which would make the criterion identical
across protocols and blind to the differences it exists to detect.

## Executed results on the complete artifact

### Split free-water gates

| Gate | Predeclared criterion | Result |
|---|---|---|
| A — blocking software gate | Synthetic `S=exp(-bD0)` derivative/Fisher/CRLB, absolute tolerance `1e-12`. | PASS: derivative `0`, Fisher `7.105e-15`, CRLB `2.776e-17` absolute error. |
| B — informational artifact check | Maximum standardized stored-signal deviation <4 sigma; nominal SE `1/sqrt(6,000,000)` and observed/nominal ratio 0.7. | PASS: max error `8.796e-4`, 2.155 nominal sigma, 3.078 observed-SE sigma. |

Gate B does not make the library atom analytic and cannot stop the software
phase.

### Full shard verification

| Check | complete-artifact result |
|---|---|
| Required v5 arrays, shapes/dtypes, no non-finites, exact b=0, non-negative variance | PASS for all 369 groups |
| 51-kio coverage; bit-identical realized rho/V per group; CRN contract | PASS |
| Diagnostic subset reconstruction | max absolute `1.290e-8`, max relative `3.231e-3` near tiny values; PASS by float32 `allclose` |
| High-b negatives | 5,700,311 values; minimum `-1.243e-3` versus `4.082e-4` independent reference SE |

Negative values are reported, not clipped or interpreted here.

### Per-entry feasibility, at the corrected `t_epi = 30 ms`

The `S/S0=0.015` floor remains absent from the builder and fitters. It is an
analysis-time hard mask **inside each `(entry,column)` Fisher sum**. A global
signal minimum is retained only as a labelled diagnostic.

| G limit | `t_epi` | Gradient cols | Combined, any entry | Combined global-min | Per-entry survivors min / median / max |
|---|---:|---:|---:|---:|---|
| Clinical 80 mT/m | 0 ms | 9,999 | 9,999 | 2,162 | 2,162 / 3,985 / 9,999 |
| Clinical 80 mT/m | **30 ms** | 9,999 | 9,999 | 2,075 | 2,075 / **3,372** / 9,999 |
| Research 300 mT/m | 0 ms | 24,081 | 24,081 | 2,406 | 2,406 / 5,449 / 24,081 |
| Research 300 mT/m | **30 ms** | 24,081 | 24,081 | 2,319 | 2,319 / **4,391** / 24,081 |

Rician validity, any entry, falls 31,072 → 31,033; the trust floor is unmoved at
31,118. The task's expectation is confirmed precisely: `t_epi` "matters little
for the Phase 1 Rician mask but is load-bearing for protocol optimization." The
*any-entry* column mask does not move at all, because it is gradient-bound —
for any column some low-`v_i` entry still clears the Rician threshold. The
*per-entry* survivor counts, which are what actually enter a Fisher sum and
therefore what a protocol optimizer sees, fall 15–19%.

### Phase-1 derivative fields, complete grid, corrected `Var(J_hat)`

Output: `/home/jaden/madi_fisher_runs/final_varfix/phase1/`, 7,673,408,900 bytes
(7.15 GiB). Column basis: 24,081 research-feasible + 200 diagnostic = 24,111 of
31,125 stored columns.

*That column basis is the defect corrected on 2026-09-06. It is retained here
because it is what this table was computed on. The re-materialised full-width
fields live in `/home/jaden/madi_fisher_runs/full_domain/phase1/` and reproduce
the derivative arrays bit-for-bit on the overlapping 24,111 columns. **Every
number in this table is unchanged** by the wider basis, because `SNR_partial` and
`beta` are measured on the 200 diagnostic columns, which both bases share. The
one statistic that does move is the absolute truncation-bias RMS of the next
section, which is pooled over all selected columns; see the amendment log.*

*The `k = 3` and `k = 4` columns below are retained deliberately. Those stencils
were dropped from the pre-registration on 2026-09-05 (amendment
`2026-09-05-drop-k3-k4`) on the strength of this table and the truncation table
that follows it; the rows are the evidence for that decision and stay here. A
Phase-1 re-run no longer produces them.*

| Field | Centres | `SNR_partial >= 3` | `beta` top-5%-\|J\| q95 / max |
|---|---:|---:|---|
| rho k=1 / 2 / 3 / 4 | 13,821 / 9,078 / 4,641 / 510 | .7304 / .7871 / .8256 / .8870 | `1.426e-5/5.552e-5`; `4.635e-6/1.436e-5`; `2.023e-6/4.332e-6`; `7.782e-7/1.139e-6` |
| V k=1 / 2 | 12,291 / 5,763 | .6761 / .7213 | `8.297e-6/1.504e-5`; `2.448e-6/5.019e-6` |
| kio k=1 | 18,081 | .7572 | `5.726e-5/1.511e-4` |

`SNR_partial >= 3` rises from .35–.71 to .68–.89 once the variance is correctly
normalized. No pooled `beta` median is reported. The pre-registered expectation
— worst top-5% `beta` near `5.538e-4` at 50,000 walkers — is met with 3.7×
margin.

### Noise versus truncation, as comparable quantities

The previously reported absolute truncation-bias RMS (`0.01565` rho, `0.01540`
V) is retained but is **not** comparable with `beta`, which is a squared
*relative* quantity. Reported relative to `|J|` on the same cells (200
diagnostic columns, top 5% of `|J_k1|`):

| axis | k | `beta` median | `|ΔJ|/|J|` median | `(ΔJ/J)²` median | truncation / noise |
|---|---:|---|---|---|---:|
| rho | 1 | 1.09e-5 | 0.0136 | 1.85e-4 | **17×** |
| rho | 2 | 2.40e-6 | 0.0544 | 2.96e-3 | 1,232× |
| V | 1 | 6.05e-6 | 0.0182 | 3.30e-4 | **55×** |
| V | 2 | 1.41e-6 | 0.0727 | 5.29e-3 | 3,743× |

Total relative squared error is approximately `beta + (ΔJ/J)²`. Truncation
already dominates Monte-Carlo noise by 17× (`rho`) and 55× (`V`) at `k = 1`, and
widening to `k = 2` trades a ~3× reduction in `beta` for a 16× rise in squared
truncation bias. **`k = 1` is the correct stencil for both axes, and the noise
term is not the limiting factor** — the opposite of the concern the wider
stencils were pre-registered to address. The `k = 2` bias is exactly 4× the
`k = 1` bias by construction, which is the `O(h²)` doubling made explicit.

### `S0` marginalization: the gap between the two bounds

`scripts/report_s0_marginal_crlb.py`, 24 evaluation nodes spread over the shared
`(rho, V, k_io)` node set, SNR 50, `T2` 80 ms, `t_epi` 30 ms, `n0_eff = 4`.
Values are median (and max) `CRLB_marginal / CRLB_fixed`, which is `>= 1` by the
Loewner ordering:

| Acquisition | cols | regime | log rho | log V | k_io |
|---|---:|---|---|---|---|
| MADI II `(20,50)` | 24 | unknown amplitude | 1.310 (2.470) | 1.223 (2.143) | 1.354 (2.280) |
| | | finite `b0`, `n0_eff=4` | 1.002 (1.048) | 1.001 (1.016) | 1.004 (1.021) |
| MADI III `(7,25)` | 24 | unknown amplitude | 1.377 (2.188) | 1.196 (1.916) | 1.298 (1.810) |
| | | finite `b0`, `n0_eff=4` | 1.002 (1.015) | 1.001 (1.012) | 1.002 (1.011) |
| Jackson-like `b` range | 9 | unknown amplitude | 1.780 (4.347) | 1.620 (5.344) | 1.869 (5.063) |
| | | finite `b0`, `n0_eff=4` | 1.002 (1.009) | 1.002 (1.010) | 1.004 (1.017) |

The known-amplitude regime returns exactly 1.000 by construction, confirming the
`lambda -> infinity` limit.

Reading: **an acquisition with no usable amplitude reference pays 20–87% in
every parameter, and up to 5.3× at individual nodes; four `b ≈ 0` averages buy
almost all of that back** (residual ≤ 0.4%). The penalty grows sharply as the
column count falls — the 9-column truncated-`b`-range case is the worst — which
is what one expects, since a wide `b` range lets the tissue columns
self-calibrate the amplitude. This bears directly on Phase 4 H4: the cost of the
thesis's `b = 50` reference is **not** a variance penalty, which is small, but
the bias from asserting `S(50) = 1`. Phase 5 must divide `--fit-s0` estimator
RMSE by the marginalized bound, not the fixed one.

*Limitation, stated rather than buried:* the stored `b` grid starts at 0 and
steps by 500, so **there is no stored `b = 50` column**. `S(50)` is extrapolated
from the lowest stored positive shell under a local mono-exponential law. It
sets only a scalar prior precision and sits within a few percent of 1, so it
cannot carry the conclusion, but the Jackson row is an *approximate placement*
and is labelled as one throughout. Its `(delta, Delta)` remain
`requires_source_timing_confirmation` and are deliberately not pursued further.

## Decisions recorded in this pass

Each has a written record at the point of change, independent of this document.

| Decision | Where recorded |
|---|---|
| `k_io` step size correct; defect was `Var(J_hat)` normalization | plan §2.5 + amendment `2026-09-05-varj-normalization`; regression test |
| `S0` marginalization adopted | plan §2.7 + amendment `2026-09-05-s0-marginalization`; pre-registration `amplitude_model`; `marginal_s0_fisher_crlb_implementation_plan.md` un-archived with a per-section status header |
| `t_epi = 30 ms` | plan §2.6 + amendment `2026-09-05-t-epi`; pre-registration `noise_model` |
| Relative truncation bias required | plan §1.4 + amendment `2026-09-05-relative-truncation-bias` |
| Protocol sweep replaces three named protocols | plan §5 + amendment `2026-09-05-protocol-sweep`; pre-registration `protocol_sweep` / `reference_protocols` |
| Complete production artifact exists | `deviations_from_paper.md` amendment log; plan amendment `2026-09-05-complete-artifact` |
| Amendment-log convention itself | `INDEX.md` header |

## Reproduction

*Updated 2026-09-06 for the corrected column domain. A fresh agent should follow
this section and not reconstruct the incident from git history; the amendment log
below carries the reasoning.*

Run locally from the repository root. Reserve roughly 10 GiB of output space per
full Phase-1 run plus working cache; a full pass over `vectors` streams 4.7 GB.
The whole sequence below is about 6 minutes of wall clock on the reference host.

```bash
LIB=data/libraries/madi_dense_universal_remediated.npz
OUT=/home/jaden/madi_fisher_runs/full_domain

python -m scripts.verify_fisher_shards "$LIB" --output "$OUT/shard_verification.json"
python -m scripts.validate_free_water_fisher --artifact "$LIB" --output "$OUT/free_water_gates.json" --tolerance 1e-12
# Continue only when Gate A reports pass: true.

# Phase 0.4 is a CONDITIONAL annotation. It does not choose what Phase 1 extracts.
python -m scripts.analyze_fisher_feasibility --artifact "$LIB" --output "$OUT/column_feasibility.json" --snr 50

# Phase 1 is the REUSABLE SUBSTRATE: all 31,125 stored columns, by default.
# --feasibility is optional and is recorded as an annotation only.
python -m scripts.run_fisher_phase1 --artifact "$LIB" --feasibility "$OUT/column_feasibility.json" --output-dir "$OUT/phase1"

python -m scripts.report_s0_marginal_crlb --artifact "$LIB" --phase1 "$OUT/phase1" --output "$OUT/s0_marginal_crlb.json"
```

Healthy output prints `GRID COMPLETE` **and**
`COLUMN DOMAIN COMPLETE — all 31125 stored (delta, Delta, b) columns`, has
`grid_complete: true` with no missing groups, a passing validator and Gate A,
every declared derivative/Richardson field, and no temporary ensemble cache left
behind. `analyze_fisher_feasibility` takes `T2` and `t_epi` from the
pre-registration; pass `--t-epi-ms 0` only to reproduce the historical
`t_epi = 0` basis.

**To reproduce the historical restricted basis on purpose** — the 24,111-column
domain every number in this document was computed on — add
`--restrict-columns-to-feasibility`. The manifest is then stamped
`COLUMN DOMAIN RESTRICTED` and downstream guards refuse to read it as universal,
which is the intent.

**Do not** apply a gradient, trust-floor or Rician mask when building Phase 1.
Those are evaluation-time conditions and belong to Phase 2 and later; see plan
§2.8 and [`fisher_domain_audit.md`](fisher_domain_audit.md).

Do **not** compare a `beta`, `SNR_partial`, or `Var(J)` figure produced before
2026-09-05 against one produced after; the normalization changed. Derivative
fields `J`, Richardson fields, feasibility masks, and gates are unaffected and
remain comparable.

## Plan items still needing a decision

- W4, the independent-seed replicate, remains the only check that establishes
  whether the W2 between-ensemble estimator is calibrated. Every `beta` and
  every debiased Fisher diagonal in this document rests on `signal_variance`
  being correct in absolute terms. W4 must land before the manuscript is frozen.

*(Resolved since the previous revision: the `S0` decision is approved and
implemented; Jackson timing is deprioritized rather than blocking. The
clinical-versus-research question was **deliberately left open** by user
decision on 2026-09-05 — Phase 2 reports both, side by side, and nominates
neither; see* [`fisher_phase2.md`](fisher_phase2.md)*.)*

## Limitations

- Phase 0/1 output is a derivative field and its audit. It is not a CRLB map, a
  degeneracy conclusion, or a protocol recommendation; those are Phases 2–5.
- The `S0` gap table is a 24-node smoke report at three declared acquisitions,
  not a sweep. It demonstrates the implementation and gives the order of
  magnitude of the amplitude penalty.
- `beta` is strongly structured, not uniform: it scales as `1/J²` and diverges
  wherever `J -> 0`, which is precisely along the sloppy direction. It is
  reported on the top-5%-`|J|` cells and by region, never as a pooled median.
- The `k_io > 30` result is a statement about `dS/dk_io` on this library's
  timing grid under the declared feasibility mask. It is consistent with MADI II
  Figure 2g, but this framework measures sensitivity, not kurtosis.
- The regenerated partial368 outputs are a faithful reproduction of the recorded
  originals, not the original files, which the host had cleared.

## Documentation reconciliation

[`INDEX.md`](INDEX.md) is the complete, one-line-per-file classification and
orientation record, and now also carries the amendment-log convention. No
document was deleted. `marginal_s0_fisher_crlb_implementation_plan.md` moved
*out* of `archive/` in this pass, the first un-archiving; its header states
per-section status because the document is genuinely mixed.

| Classification | Reconciled status |
|---|---|
| CURRENT | `deviations_from_paper.md`, Fisher plan/framework, the un-archived marginal-`S0` handoff, `fitting_methods.md`, `sol_package_guide.md`, `tumorsynth_install.md`, `universal_library.md`, the parameter workbook, and primary sources. |
| PROVENANCE | P0/P0A/P0B, SI, finite-geometry, physics-audit, CRN/stencil-probe, launch-readiness, and prior-reorganization records, retained unchanged. |
| STALE PLAN | Pre-v5 identifiability, joint-Bayesian handoff, old P0/schema/runbook/probe plans, and the checklist. |
| SCRATCH | The shard viewer notebook and command history. |
| UNCLEAR | None. |

### Contradictions retained explicitly

- The P0 plan's 4.36 GiB is the vectors matrix, whereas the v5 schema note's
  9.29 GiB is the four-matrix raw floor; its pilot dimensions are also pre-v5.
- `universal_library.md` allows a `k_io` seed term in prose, but current
  `_walk_seed` is independent of `k_io`, which is required for CRN kio
  derivatives.
- The old Fisher plan called stored free water analytic; the library routes it
  through Monte Carlo. The split gate resolves this without a rebuild.
- The pre-SI physics audit and SI/full-facet records describe different
  implementations; neither silently invalidates the other audit trail.
- The Sol guide's 64-shard MADI appendix is historical; the production layout
  is 369 groups. Pre-launch stencil instructions say INVESTIGATE while the
  completed stencil record says GO.
- `provenance/v5_stencil_probe.md` states "retain 40 ensembles × 100,000
  walkers"; production ran 40 × 50,000. The reduction was taken later and is
  justified in `provenance/v5_fast_classifier_launch_readiness.md`. Both records
  stand unedited, and the walker difference is why probe and production `beta`
  values must be compared with the 1×–2× scaling band, not directly.

## Amendment log

This record is updated in place. Entries record what changed, what it replaced,
and why, so a reader can reconstruct the decision without diffing git history.

### 2026-09-06-substrate-domain — the column basis was a scanner mask, and is now the stored grid

- **Previously:** Phase 1 required `--feasibility` and extracted only the columns
  in the research (300 mT/m) combined mask plus the 200 diagnostic columns —
  24,111 of the library's 31,125 stored `(delta, Delta, b)` columns. This
  document recorded that basis as a line of provenance and nothing downstream
  could tell that "absent from the cache" and "not applicable" were different
  things.
- **Now:** Phase 1 extracts over every stored column. `--feasibility` is optional
  and annotation-only; `--restrict-columns-to-feasibility` reproduces the old
  basis on purpose and stamps the manifest RESTRICTED. The manifest is schema
  `v3` and carries a `column_domain` block. Output:
  `/home/jaden/madi_fisher_runs/full_domain/phase1/`, 9,245,110,382 bytes
  (8.61 GiB), 31,125 columns.
- **Why:** the mask is a 300 mT/m scanner ceiling. It removed the entire
  short-`delta`, high-`b` corner from the reusable substrate — 7,014 columns and
  84 timing pairs that lost every diffusion-weighted column — so no later
  analysis at any other gradient setting could reach them. It was never
  pre-registered and appears in no amendment log. Full audit and classification
  of every other restriction in Phases 0-6:
  [`fisher_domain_audit.md`](fisher_domain_audit.md).
- **Verification, on the overlapping 24,111 columns — bit-for-bit, not to a
  tolerance:**

  | Field | rows | cells compared | differing bits |
  |---|---:|---:|---:|
  | `J_rho_k1` | 13,821 | 333,238,131 | **0** |
  | `J_rho_k2` | 9,078 | 218,879,658 | **0** |
  | `J_V_k1` | 12,291 | 296,348,301 | **0** |
  | `J_V_k2` | 5,763 | 138,951,693 | **0** |
  | `J_k_io_k1` | 18,081 | 435,950,991 | **0** |
  | `Richardson_rho_k1_k2` | 9,078 | 218,879,658 | **0** |
  | `Richardson_V_k1_k2` | 5,763 | 138,951,693 | **0** |
  | all five `VarJ_*_diagnostic` | — | 200 columns each | **0** |

  Sample rows are identical for every field, the diagnostic column indices are
  identical, and the Phase-2 caches agree bit-for-bit on all 24,081
  research-feasible columns (453,204,420 cells per member, both members).

- **What did not change:** every `SNR_partial >= 3` fraction and every `beta`
  quantile in the derivative-field table above, to all reported digits. They are
  computed on the 200 diagnostic columns, which are common to both bases. The
  relative truncation-bias table also reproduces exactly (`rho` `k = 1` median
  0.013608, `k = 2` 0.054432; `V` `k = 1` 0.018176, `k = 2` 0.072704; squared
  medians 1.85e-4, 2.96e-3, 3.30e-4, 5.29e-3), and the full-domain run is now
  the manifest that carries it.
- **What did change, and why it is not a correction:** the **absolute**
  truncation-bias RMS, `rho` 0.015650 → 0.014583 and `V` 0.015398 → 0.014311.
  That statistic is pooled over every selected column, so widening the basis
  changes its denominator and its population; the added columns sit at high `b`
  where `|J|` and its absolute bias are small, which pulls the RMS down. The
  absolute **maxima** are unchanged (0.248967 `rho`, 0.193433 `V`). The values in
  the table above are the executed ones on the 24,111-column basis and stay as
  they are. This is also why §1.4 asks for the *relative* bias: it is the
  quantity that means the same thing on both bases.
- **`k = 3` / `k = 4`:** not regenerated (dropped from the pre-registration on
  2026-09-05). The originals in `final_varfix/phase1/` are retained unchanged as
  the evidence for that decision.
- **The `S0` gap table reproduces exactly** on the corrected substrate, to every
  reported digit, at all three acquisitions and all three regimes. One thing it
  now also reports, which it could not before: the MADI III `(7, 25)` row uses
  columns requiring up to **389 mT/m**, above the very ceiling that had shaped
  the substrate. It survived only because `(7, 25)` is a diagnostic timing pair,
  so its columns were re-added by the union. The report's column basis is now
  declared rather than incidental, and `--gradient-scenario research` gives the
  conditioned reading (14 columns instead of 24). See the audit, R18.

### 2026-09-05-phase2-carry-forward

- **Previously:** the derivative-field table presented the `rho` `k = 3` and
  `k = 4` fields as ordinary pre-registered stencils, and "Plan items still
  needing a decision" listed the clinical-versus-research choice as open for
  the analyst to settle.
- **Now:** a note above the table states that `k = 3` / `k = 4` were dropped from
  the pre-registration on 2026-09-05 and that the rows are retained as the
  evidence for that decision. The scenario choice is recorded as deliberately
  deferred by the user rather than pending.
- **Why:** Phase 2 execution (`fisher_phase2.md`). The stencil rows must not be
  deleted because they *are* the evidence; the scenario line must not read as
  an unresolved analyst decision because the user resolved it by declining to
  choose.
- **Not changed:** every number in this document. The `Var(J_hat)` normalization,
  the `k_io` sensitivity result, the shard-45 diff, and the truncation analysis
  stand as executed.
