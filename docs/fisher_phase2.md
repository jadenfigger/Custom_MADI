# Fisher / CRLB Phase 2

> **Carry-forward, 2026-09-06 — scope, not correctness.** Everything in this
> record was computed on a Phase-1 substrate that held 24,111 of the library's
> 31,125 stored `(delta, Delta, b)` columns, because Phase-1 extraction had been
> restricted to a 300 mT/m research-scanner mask. **No number here changes.**
> Every Phase-2 result was already conditioned on a declared gradient scenario,
> and the restricted substrate contained every column either scenario admits
> (clinical ⊂ research ⊂ the old substrate), so applying the same declared
> conditions to the corrected full-width substrate reproduces this record.
>
> What changes is the **scope of the claims**. Nothing below is a statement about
> intrinsic model identifiability over the stored acquisition domain; every
> conclusion is conditional on its declared gradient scenario, `T2`, `t_epi`,
> SNR, budget `N` and averaging. The region that was absent — short `delta` at
> high `b` — is the narrow-pulse, high-`q` regime in which cell volume `V` is
> most strongly encoded, so that distinction is load-bearing rather than
> pedantic. What the model says there is now askable and has **not** been
> answered. See the amendment log at the end of this file and
> [`fisher_domain_audit.md`](fisher_domain_audit.md).

> **Forward pointer, 2026-09-09 — no number here changes.** Phase 3 has since
> run ([`fisher_phase3.md`](fisher_phase3.md)) and answers two things this record
> deliberately left open: the deferred question of §3.5, and the model-layer
> reading the 2026-09-06 carry-forward said was now askable. It also re-forms the
> four optimal arms of §3.2 by an independent accumulation path and reproduces
> **72 of 72 compared quantities at exactly zero relative difference**, which is
> an external confirmation of this record rather than a revision of it. Nothing
> below was edited; see the amendment log entry `2026-09-09-phase3-forward`.

## Verdict

**PHASE 2 EXECUTED. THE HEADLINE PREDICTION HOLDS, AND IS STRONGER THAN
PREDICTED.** The plan expected `k_io` to be weakly determined at a single
diffusion time and substantially improved by multi-`Delta` sampling. What the
sweep actually shows is that **no single-`Delta` acquisition is jointly
identifiable in `(rho, V, k_io)` at the median grid node at all** — not one of
3,820 single-`Delta` arms across both gradient scenarios and all three b-subset
sizes reaches a finite minimax relative CRLB — and that adding a second
diffusion time fixes it. Four results carry beyond this phase:

1. **Single-`Delta` fails, and it fails on `rho`, not on `k_io`.** For every one
   of the 3,820 single-`Delta` arms the binding parameter is `log rho`. The best
   single-`Delta` acquisition identifies 34.4% (clinical) or 36.9% (research) of
   evaluation nodes. Two diffusion times raise that to 67–73% and produce a
   finite criterion. This is the Phase-2 answer to Objective 1, and it is a
   sharper claim than "`k_io` is hard".
2. **The Monte-Carlo debias decides the verdict, and it is a sub-1% correction.**
   Subtracting `Var(J_hat)` moves the Fisher diagonal by only 0.03–0.6%, yet it
   takes the single-`Delta` identifiable fraction from 63–80% down to 32–37% and
   turns a finite minimax into an infinite one. The debiased matrix is that close
   to singular. This is exactly the bias trap section 2.5 of the plan was written
   to catch, now quantified: **an undebiased finite-difference Fisher analysis of
   this library would have reported single-`Delta` MADI as well conditioned.**
3. **Withdrawing the `k_io` restriction did *not* make the minimax degenerate.**
   That was the risk the withdrawn cap existed to prevent, and it did not
   materialise. Across the m=2 sweeps only 0.02–0.3% of arms fall within 5% of
   the best and the median arm is 2.5–3.1x worse than the best, so the criterion
   discriminates strongly. `k_io` is the argmax for 0.04% of finite clinical arms
   and 39% of finite research arms — not the saturation that was feared.
4. **At matched scan time, fewer and better-averaged b-values beat more of
   them,** and the optimum always pairs one short-`Delta` with one long-`Delta`
   timing. Subset size 8 beats 12 beats 16 in every scenario and every arm
   cardinality.

**Neither gradient scenario is nominated, and no `k_io` ceiling was imposed.**
Both were deliberately withheld and both remain withheld. Both scenarios *are*
imposed on the acquisitions being scored, which is what a protocol comparison
means; since 2026-09-06 they are imposed only there, and not on the substrate
they are scored against.

**The largest methodological caveat is stated up front:** the measured greedy
optimality gap at `m = 2` is **35% to 142%**. Greedy forward selection over
timing pairs is a poor search here. `m = 1` and `m = 2` were enumerated
exhaustively and are trustworthy; **`m = 3` and `m = 4` are greedy and are
therefore upper bounds on what those cardinalities can achieve.** Where `m = 4`
already beats `m = 2` (research, subset sizes 8 and 12) the conclusion is safe
because it survives a handicap; where `m = 2` wins (clinical, and research size
16) the result cannot be called final.

---

## 1. Carry-forward corrections

### 1.1 The `k = 3` and `k = 4` stencils are dropped

Removed from `madi/fisher_crlb_preregistration.json` (`stencil_half_widths` is now
`{"rho": [1, 2], "V": [1, 2], "k_io": [1]}`) and from Phase-2 computation and
storage. **No Fisher matrix in Phase 2 is built from any stencil but `k = 1`.**
`k = 2` survives in the pre-registration for one reason only: the Richardson
estimate `(4 J_k1 - J_k2)/3` needs it to quantify truncation bias.

The Phase-1 evidence that decided this is retained unedited in
[`fisher_phase01_framework.md`](fisher_phase01_framework.md), with a note above
the table saying why those rows stay. In short: `k = 3` and `k = 4` exist at only
4,641 and 510 centres, and truncation bias already dominates Monte-Carlo noise by
17x (`rho`) and 55x (`V`) at `k = 1`, while each doubling of `h` quadruples it.
Widening the stencil was the wrong direction, and the two widest stencils were
pre-registered on mask-band geometry alone, before that was known.

A Phase-1 re-run no longer writes `J_rho_k3.npy` or `J_rho_k4.npy`. The existing
files are left in place; they are evidence, not output.

### 1.2 Audit of duplicated `madi/` arithmetic

The Phase-1 `Var(J_hat)` defect arose because a script reimplemented a `madi/`
function inline for streaming and then drifted from it. That pattern usually
appears more than once, so `scripts/` and `analysis/` were swept for any place
where a script re-derives arithmetic that already exists in a `madi/` module.

**Four pre-existing sites found. One was a live drift risk carrying a latent bug,
one is a deliberate second path now pinned by test, one is superseded scratch,
and one is correct by design and must not be "fixed".**

| Site | What it re-derives | Verdict | Action |
|---|---|---|---|
| `analysis/v5_stencil_probe.py::compute_beta` | `madi.fisher_crlb.derivative_variance`, and the central-difference denominator | **Drift risk, with a latent uniform-spacing assumption** | Refactored onto the audited helper; denominator now read from realized labels; 2 regression tests |
| `analysis/v5_crn_diagnostic.py::paired_correlation_matrix` | the CRN covariance of entry means | Deliberate separate path — it also bootstraps over resampled ensemble indices, which the helper does not do | Kept; pinned to the helper by regression test |
| `analysis/prod_inter_fisher_analysis.ipynb` | gradient formula, Fisher matrix, `kappa`, canonical-grid indexing, `Var(J_hat)` with `n_ensembles` as a literal `40.0` | Superseded exploratory scratch; nothing imports it | Retained unedited apart from a header cell naming the authoritative implementation of each piece; classified SCRATCH in `INDEX.md` |
| `scripts/validate_v5_production.py` | `np.var(subset, ddof=1)` against the stored `signal_variance` | **Correct as-is; must NOT be refactored** — it is the only independent check that the stored array is right | None |

**The one real finding.** `analysis/v5_stencil_probe.py` computed the
central-difference variance inline:

```python
variance = artifact.arrays["signal_variance"][:, artifact.subset_indices] / float(artifact.n_ensembles)
...
denominator = 2.0 * int(width) * float(record["h1"])
numerator = variance[left] + variance[right] - 2.0 * covariance
derivative_variance = np.maximum(numerator, 0.0) / denominator**2
```

The *variance* arithmetic here is arithmetically identical to
`madi.fisher_crlb.derivative_variance`. That is worth stating plainly rather than
passing over, because this probe produced the measured `beta` calibration that
the Phase-1 audit was compared against. Had it drifted, the reference would have
moved rather than the measurement. It had not.

The *denominator* is the latent problem. `2.0 * width * h1` assumes uniform
spacing. That holds for `rho` and `V`, whose grids are uniform in log, and it
holds for the probe's declared `k_io` triple — but only because
`scripts/validate_v5_stencil_probe.py` pins `KIO_VALUES = (19.0, 20.0, 21.0)`,
which sits inside the 1 s^-1 region, and `h1` is the literal `1.0`. A `k_io`
stencil straddling the 30 s^-1 boundary, where spacing becomes 5 s^-1, would have
been differenced over a denominator three times too small, with nothing to catch
it. This is the same class of assumption that the previous task went looking for
in the Phase-1 engine and did not find there.

Both are fixed. The value now comes from the audited helper, and the denominator
is computed from realized endpoint labels with an assertion against the declared
uniform step:

```python
denominator = float(record["right_coordinate"]) - float(record["left_coordinate"])
nominal = 2.0 * int(record["width"]) * float(record["h1"])
if not math.isclose(denominator, nominal, rel_tol=1.0e-9, abs_tol=0.0):
    raise DiagnosticError("ABORT: realised stencil spacing disagrees with the declared uniform step ...")
```

**The refactor is a no-op for the recorded probe run**, confirmed directly: for
the declared triple the realized denominator is exactly `2.0`, the value the old
code hardcoded. The probe artifact itself no longer exists on disk — only its
figures, CSVs and summary JSON were retained — so the probe cannot be re-run. The
no-op check plus a synthetic regression test are what establish that
`provenance/v5_stencil_probe.md` still stands unamended. It does.

**Two further sites were created by Phase 2 itself and are pinned the same way.**
`madi.fisher_crlb.packed_inverse_diagonal` and `packed_amplitude_marginal` are
vectorized cofactor forms of `fisher_diagnostics` and `amplitude_marginal_fisher`,
needed because Phase 2 inverts on the order of `10^10` 3x3 matrices. They were
put in `madi/` rather than in the script, and two new tests pin them to the
single-matrix reference. Phase 2 also recomputes `J` on the fly from the cached
signal instead of reading the 7 GiB Phase-1 fields — the same pattern again —
and that one is pinned not by a synthetic test but by comparing against the
stored Phase-1 field on every run, reported in section 3.1 below.

**Clean elsewhere.** `scripts/analyze_identifiability.py` imports its derivative
and Fisher core from `madi.identifiability`.
`scripts/build_si_geometry_reference.py` builds the `<A/V>` table independently,
but that is its purpose — it is the reference `madi/ensemble.py` consumes, and
its docstring says so.

Five regression tests were added, in the style of the existing
`test_streaming_variance_accumulation_matches_the_audited_helper`. The Fisher
suite is now 18 tests; the full suite is 107 passed, 4 skipped, 1 xfailed, and
one pre-existing unrelated failure (`test_gpu_golden.py`, a stale
`cpu_gpu_golden_v1.npz` fixture, confirmed pre-existing in the previous task).

---

## 2. How Phase 2 was run, and the specification gaps closed to run it

The sweep is specified by `madi/fisher_crlb_preregistration.json`
(`protocol_sweep`) and section 5 of
[`fisher_crlb_analysis_plan.md`](fisher_crlb_analysis_plan.md). Five things had
to be settled before it could execute. Each is a dated amendment in both
documents, and each was fixed **before** any ranking was looked at.

Two are user decisions carried into this task and implemented as stated:

- **No `k_io` restriction in the minimax** (`2026-09-05-withdraw-kio-restriction`).
  A previous revision capped the evaluation set at `k_io <= 30 s^-1` to stop the
  criterion saturating. That cap is withdrawn. The criterion runs over the full
  interior grid, 1 to 125 s^-1, and the predicted degeneracy is *measured* rather
  than prevented — see section 4.
- **No choice between the clinical and research gradient scenarios.** Every
  result is computed under both, side by side, never pooled or averaged, and
  neither is nominated. The gradient *mask* stays: the stored grid contains
  columns requiring about 15.9 T/m, and an unmasked optimizer would happily
  recommend acquisitions no scanner can perform.

Three are gaps found at execution, where the pre-registration was silent or
self-contradictory.

**(a) The search strategy, because the declared sweep is not enumerable.**
The pre-registration declared `m` in `{2, 3, 4}` without saying how those sets
are searched. As declared, at research subset size 8 that is 993 single arms,
492,528 pairs, **1.6 x 10^8** triples and **4.0 x 10^10** quadruples, times
735,471 b-subsets per timing pair. `m = 1` and `m = 2` are therefore enumerated
exhaustively; `m = 3` and `m = 4` are greedy forward extensions of the exhaustive
`m = 2` optimum. The greedy optimality gap is **measured, not assumed**, by also
running greedy `m = 2` forward from the best `m = 1` and comparing against the
exhaustive answer. It is large; see section 5.

**(b) Rician validity at the acquisition's own averaging**
(`2026-09-05-phase2-averaging-aware-mask`). The pre-registration says a budget of
`N` images is split across the selected columns, *and* that
`averages_per_column = 1`. Those cannot both hold. Taken literally, the
one-average reading leaves the median grid node with **2 to 3 usable columns out
of 24**, so no three-parameter Fisher matrix exists anywhere on the grid and
Phase 2 cannot run at all. `averages_per_column = 1` belongs to the Phase-1
screen for which columns are *ever* usable, not to the evaluation of a named
acquisition. Rician validity is therefore evaluated at
`S >= 3 * sigma_1 / sqrt(n_c)` with `n_c = N/(m * size)`, which is known before
the b-subset is selected because `N`, `m` and the size are all declared. The
trust floor stays at `S/S0 >= 0.015`, and it is not the binding filter: at
`(20,50)` with `n_c = 16` the Rician threshold is `S/S0 >= 0.052`, well above it.

*Consequence, stated because it corrects an earlier claim of mine.* The Fisher
matrix is linear in `N`, so at first sight the ranking is budget-independent, and
the `2026-09-05-protocol-sweep` amendment asserted exactly that. **That assertion
was wrong.** The mask depends on `n_c`, so `N` changes which columns are usable
and can change the ranking. Results are reported at a declared `N = 128`
diffusion-weighted images, and the `N` sensitivity is an open item in section 8.

**(c) The node aggregate is a median, not a mean**
(`2026-09-05-phase2-node-aggregation`). The pre-registration weights nodes
"uniformly" and takes a "max over parameters" without fixing the order of
operations. Read literally as a uniform mean, the criterion is `+inf` for
**every** candidate acquisition, because once the debias is applied no
acquisition is identifiable at every node. A criterion that is infinite
everywhere ranks nothing.

The median is the uniformly weighted order statistic that survives an infinite
tail: it is finite exactly when a protocol identifies more than half the
evaluation nodes. So, per parameter, take the median over evaluation nodes with
`+inf` at unidentified nodes; then the max over parameters.

The obvious alternative — averaging over only the identifiable nodes — was
**rejected**, because it rewards a protocol for identifying *fewer* nodes, which
is exactly backwards. It is reported alongside but is not the criterion. The
identifiable-node fraction is itself a primary reported quantity, and the count
of arms with a finite minimax is reported for every sweep, so a sweep in which
the criterion cannot rank is visible rather than quietly resolved by an arbitrary
`argmin`. That is not hypothetical: it is what happened to every `m = 1` sweep,
and section 3.2 reports it as a result rather than hiding it behind a winner.

A node counts as identifiable only when the **whole inverse diagonal of `F` is
positive**, which is Sylvester's leading-minor test strengthened to what a CRLB
actually requires. The two are equivalent in exact arithmetic; in floating point
a near-singular matrix can pass the leading-minor test and still return a
non-positive cofactor ratio, surfacing as a NaN CRLB rather than as an
unidentified node. One such node was found during development and is what
prompted the change. The strengthened test is strictly conservative.

### What is computed at each node

At every node carrying all three `k = 1` central stencils, for each acquisition,

    F = (N / n_selected) * sum_c u_c * [ J_c J_c^T - diag Var(J_hat_c) ]

with `u_c = 1/sigma_{1,c}^2`, `sigma_{1,c} = sigma_0 exp((delta_c + Delta_c + t_epi)/T2)`,
and both hard masks applied per `(node, column)` inside the sum. The debias is
applied **to the diagonal only**: central differences on the three axes use four
disjoint library entries, so their Monte-Carlo noises are independent and the
off-diagonals — the elements carrying the degeneracy — are already unbiased.

`b = 0` is excluded from every diffusion-weighted subset. `S(0) = 1` exactly for
every entry, so `J` is identically zero there; the column carries no tissue
information and enters only as the amplitude reference, which the `n0_eff` sweep
already models. Including it would double-count it.

Relative CRLB is `sqrt([F^-1]_jj)` for the two log parameters — a log-parameter
CRLB *is* the fractional precision — and `sqrt([F^-1]_33)/max(k_io, 5)` for the
linear `k_io`, the floor being pre-registered so the smallest-`k_io` node does not
dominate as an artifact of division.

Run parameters: `N = 128` diffusion-weighted images, SNR 50 at `b = 0`,
`T2 = 80 ms`, `t_epi = 30 ms`, `n_ensembles = 40`, 50,000 walkers per ensemble.

### The debias, and the one thing the library cannot supply

`Var(J_hat)` needs the CRN covariance between the two stencil endpoints, and the
library stores `ensemble_means_subset` for only **200 diagnostic columns, which
is 8 timing pairs out of 1,245**. For the other 1,237 pairs that covariance does
not exist and no amount of care recovers it.

The sweep therefore uses the endpoint-only form
`Var(J_hat) = (Var(S-) + Var(S+))/(n_ensembles h^2)` everywhere. Dropping a
positive covariance **overstates** `Var(J_hat)`, so the debiased Fisher matrix is
a lower bound and every reported CRLB is conservative. The size of that
conservatism is measured rather than asserted, at the 8 pairs where both forms
exist:

| axis | endpoint-only / exact, median | q95 | max |
|---|---:|---:|---:|
| `rho` | 1.26 | 2.50 | 18.5 |
| `V` | 1.46 | 3.48 | 34.3 |
| `k_io` | **6.96** | 27.8 | 241 |

The ordering is exactly what the Phase-1 CRN correlations predict — `k_io` is the
best-correlated axis (`r = 0.961`), so it is the axis whose covariance term
matters most and whose endpoint-only variance is most inflated. A direct check at
`(20, 50)`, which *is* a diagnostic pair, puts the practical size of the
difference at 6 percentage points of identifiable fraction (0.325 endpoint-only
versus 0.385 exact, against 0.930 undebiased). **So the endpoint-only rule is
genuinely conservative rather than qualitatively different, and every
identifiability fraction below is understated by roughly that margin.**

---

## 3. Results

### 3.1 The evaluation node set, and the derivative cross-check

**11,417 nodes** = 233 `(rho, V)` mask pairs x 49 interior `k_io` values, being
every node that carries all three `k = 1` central stencils. 38.8% of them sit at
`k_io > 30 s^-1`, and none is excluded: the withdrawn restriction is genuinely
withdrawn.

**136 of the 369 retained `(rho, V)` pairs are lost** to incomplete stencils. They
are not scattered: they are the two edges of the mask band. A node needs
`rho +/- 1` and `V +/- 1` both present, and the band
`0.40 <= rho*V*1e-6 <= 0.99` is only 0.394 decades wide against a `V` node
spacing of 0.0683 decades, so the outermost one-to-two `V` indices at each `rho`
fall outside it. The lost list runs `[0,53] [0,54] ... [63,13] [63,14]` — a
strip along each boundary. Every retained pair loses its two `k_io` end nodes
(indices 0 and 50) for the same reason. **This matters for Phase 4**, because the
unrealistic-volume pathology is a mask-boundary hypothesis and the boundary is
precisely where Phase 2 has no Fisher matrix at all.

Phase 2 recomputes `J` from the cached signal rather than reading the Phase-1
fields, so the two are compared on every run over 2,048 sampled cells per axis:

| axis | max abs difference | max relative difference |
|---|---|---|
| `rho` | 3.72e-7 | 6.66e-5 |
| `V` | 2.71e-7 | 2.10e-4 |
| `k_io` | 5.96e-9 | 5.97e-6 |

Phase 1 accumulated in float32, so agreement is expected at float32 rounding
rather than exactly. This is an independent confirmation of the Phase-1
derivative fields as well as of the Phase-2 path.

### 3.2 Single-`Delta` versus multi-`Delta` at matched measurement count

The core comparison, at `N = 128` diffusion-weighted images for every arm.
`ident` is the identifiable-node fraction; `minimax` is the pre-registered
criterion; `argmax` is the parameter that attains it. **Reported under both
gradient scenarios, never pooled.**

**Research, 300 mT/m**

| size | arm | timing pairs (delta, Delta) ms | cols | avg | ident | minimax | argmax |
|---:|---|---|---:|---:|---:|---:|---|
| 8 | m1 | (10,43) | 8 | 16.0 | 0.369 | **inf** | log_rho |
| 8 | m2 | (4,44) + (10,10) | 16 | 8.0 | 0.732 | 0.4831 | k_io |
| 8 | m3 | (4,44) + (7,21) + (10,10) | 24 | 5.3 | 0.722 | 0.4649 | k_io |
| 8 | **m4** | (3,75) + (4,44) + (7,21) + (10,10) | 32 | 4.0 | **0.765** | **0.4187** | k_io |
| 12 | m1 | (10,42) | 12 | 10.7 | 0.359 | **inf** | log_rho |
| 12 | m2 | (10,13) + (24,37) | 24 | 5.3 | 0.719 | 0.6504 | k_io |
| 12 | m3 | (4,60) + (10,13) + (24,37) | 36 | 3.6 | 0.740 | 0.6890 | k_io |
| 12 | **m4** | (4,60) + (10,13) + (16,24) + (24,37) | 48 | 2.7 | 0.741 | **0.6287** | k_io |
| 16 | m1 | (11,41) | 16 | 8.0 | 0.351 | **inf** | log_rho |
| 16 | **m2** | (12,13) + (27,42) | 32 | 4.0 | 0.732 | **0.7657** | k_io |
| 16 | m3 | (12,13) + (16,24) + (27,42) | 48 | 2.7 | 0.721 | 0.7824 | k_io |
| 16 | m4 | (4,80) + (12,13) + (16,24) + (27,42) | 64 | 2.0 | 0.748 | 0.8199 | k_io |

**Clinical, 80 mT/m**

| size | arm | timing pairs (delta, Delta) ms | cols | avg | ident | minimax | argmax |
|---:|---|---|---:|---:|---:|---:|---|
| 8 | m1 | (29,41) | 8 | 16.0 | 0.344 | **inf** | log_rho |
| 8 | **m2** | (11,80) + (24,24) | 16 | 8.0 | 0.671 | **1.081** | log_rho |
| 8 | m3 | (11,80) + (15,50) + (24,24) | 24 | 5.3 | 0.687 | 1.100 | log_rho |
| 8 | m4 | (11,80) + (12,70) + (15,50) + (24,24) | 32 | 4.0 | 0.690 | 1.221 | log_rho |
| 12 | m1 | (30,40) | 12 | 10.7 | 0.331 | **inf** | log_rho |
| 12 | **m2** | (14,75) + (27,27) | 24 | 5.3 | 0.643 | **1.553** | log_rho |
| 12 | m3 | (14,75) + (18,55) + (27,27) | 36 | 3.6 | 0.650 | 1.727 | log_rho |
| 12 | m4 | (14,75) + (18,55) + (26,80) + (27,27) | 48 | 2.7 | 0.653 | 1.876 | log_rho |
| 16 | m1 | (29,41) | 16 | 8.0 | 0.319 | **inf** | log_rho |
| 16 | **m2** | (16,75) + (30,30) | 32 | 4.0 | 0.620 | **2.571** | log_rho |
| 16 | m3 | (16,75) + (24,70) + (30,30) | 48 | 2.7 | 0.623 | 2.760 | log_rho |
| 16 | m4 | (16,75) + (21,50) + (24,70) + (30,30) | 64 | 2.0 | 0.606 | 3.044 | log_rho |

Five things to read off these tables.

**Single-`Delta` never reaches a finite criterion.** Across all six sweeps — 449,
319, 223 clinical arms and 993, 941, 895 research arms, 3,820 in total — the
count of arms with a finite minimax is **zero**. The best single-`Delta`
acquisition identifies 34.4% of nodes (clinical, `(29,41)`) or 36.9% (research,
`(10,43)`); the *median* single-`Delta` arm identifies 29–36%. So the ranking in
those rows is by identifiable-node fraction, not by the primary criterion, and
the table says so. This is not a near miss: the criterion needs more than 50% and
the whole distribution tops out below 37%.

**It fails on `rho`, not on `k_io`.** For all 3,820 single-`Delta` arms the argmax
parameter is `log rho`. The `rho` axis is the one the plan already flagged as
structurally noisiest — changing `rho` changes both the Poisson seed count and
the domain size, so neighbouring entries are independent tissue realizations
rather than one tissue perturbed, and common random numbers buy little there.
Phase 2 shows the consequence is not merely a noisier derivative but a loss of
identifiability. The expectation going in was that `k_io` would be the weak
parameter; at a single diffusion time it is not, because `rho` is worse.

**Two diffusion times fix it.** Identifiable fraction roughly doubles, 0.35 to
0.62–0.73, and a finite criterion appears. That is the headline the plan
predicted, obtained for a stronger reason than predicted.

**Beyond two, the scenarios diverge.** Under research gradients `m = 4` is best at
sizes 8 and 12 (0.4187 and 0.6287) while `m = 2` is best at 16. Under clinical
gradients `m = 2` wins at every size and `m = 3` and `m = 4` are monotonically
worse. The mechanism is visible in the `avg` column: at matched `N`, each extra
timing pair divides the averaging, and the clinical mask — which has already
deleted most high-`b` columns — cannot afford it. **Given the greedy gap of
section 5, the `m = 2` wins are the ones that should be treated as provisional.**

**Smaller b-subsets win at matched scan time,** monotonically, in both scenarios:
research 0.419 (size 8) < 0.629 (12) < 0.820 (16); clinical 1.081 < 1.553 <
2.571. Eight well-averaged b-values beat sixteen thin ones. The pre-registered
sizes bracketed the optimum from above, and the honest reading is that the true
optimum may be **below 8** and was not searched.

**The winning geometry is always one short-`Delta` plus one long-`Delta` pair**,
which is the physically expected answer: a short diffusion time constrains
restriction and geometry, a long one constrains exchange. What differs between
scenarios is `delta`, not that structure. Research optima use `delta = 3–12 ms`;
clinical optima are pushed to `delta = 11–30 ms` because
`G = sqrt(b/(gamma^2 delta^2 (Delta - delta/3)))` forces long pulses at 80 mT/m.

### 3.3 Published protocols, for context only

Placed in the swept space, not evaluated as targets.

| protocol | scenario | size | ident | minimax |
|---|---|---:|---:|---:|
| MADI II `(20,50)` | research | 8 | 0.349 | inf |
| MADI II `(20,50)` | research | 12 | 0.332 | inf |
| MADI II `(20,50)` | research | 16 | 0.321 | inf |
| MADI II `(20,50)` | clinical | 8 | 0.300 | inf |
| MADI II `(20,50)` | clinical | 12 | 0.278 | inf |
| MADI III `(7,25)` | research | 8 | 0.333 | inf |
| MADI III `(7,25)` | research | 12 | 0.319 | inf |

Both published single-`Delta` timings behave like every other single-`Delta`
arm: no finite minimax, identifiable fraction 0.28–0.35, argmax `log rho`. They
are neither unusually good nor unusually bad choices — they sit close to the best
single-`Delta` arm available (0.369 research, 0.344 clinical), which is a fair
thing to say about them. The limitation is the single diffusion time, not the
particular timing chosen.

MADI III `(7,25)` cannot supply 16 gradient-feasible b-values under research
limits, nor 8 under clinical limits, so those cells are blank: `delta = 7 ms` is
too short to reach high `b` within `G_max`.

### 3.4 The Monte-Carlo debias decides the verdict

The same acquisitions scored with and without subtracting `Var(J_hat)` from the
Fisher diagonal. `debias/diag` is the mean fraction of each diagonal element that
the correction removes.

| scenario | size | arm | ident undebiased | ident debiased | minimax undebiased | minimax debiased | debias/diag (rho, V, k_io) |
|---|---:|---|---:|---:|---:|---:|---|
| research | 8 | m1 | 0.687 | **0.369** | 3.8e5 | **inf** | 3.7e-4, 1.3e-3, 3.4e-3 |
| research | 8 | m2 | 1.000 | 0.732 | 0.4344 | 0.4831 | 7.1e-4, 3.0e-3, 6.2e-3 |
| research | 12 | m1 | 0.654 | **0.359** | 1.1e6 | **inf** | 3.6e-4, 1.3e-3, 3.3e-3 |
| research | 16 | m1 | 0.628 | **0.351** | 1.6e6 | **inf** | 3.6e-4, 1.3e-3, 3.3e-3 |
| clinical | 8 | m1 | 0.800 | **0.344** | 11.9 | **inf** | 3.4e-4, 1.1e-3, 3.0e-3 |
| clinical | 8 | m2 | 1.000 | 0.671 | 0.9009 | 1.081 | 4.1e-4, 1.5e-3, 3.5e-3 |
| clinical | 12 | m1 | 0.759 | **0.331** | 16.01 | **inf** | 3.3e-4, 1.0e-3, 2.9e-3 |
| clinical | 16 | m1 | 0.725 | **0.319** | 19.94 | **inf** | 3.2e-4, 1.0e-3, 2.8e-3 |

**The correction is between 0.03% and 0.6% of the Fisher diagonal, and it removes
identifiability at a third to a half of all nodes.** That is only possible because
the single-`Delta` Fisher matrix is very nearly rank-2: its smallest eigenvalue is
smaller than a 0.3% perturbation of its largest. Measured directly at MADI II
timing, the ratio `lambda_min/lambda_max` of the non-dimensionalized matrix has a
median of about `1.7e-4`, so a `1e-3` diagonal correction is several times the
smallest eigenvalue and flips the determinant's sign.

Two consequences worth carrying into the manuscript:

- **An undebiased analysis would have reached the opposite conclusion.** Every
  `m = 1` row has a finite undebiased minimax and an identifiable fraction of
  63–80%. A naive finite-difference Fisher analysis of this library would have
  reported single-`Delta` MADI as adequately conditioned. Section 2.5 of the plan
  predicted exactly this failure mode; this table is the measurement of it.
- **`m = 2` and above are robust to it.** Their identifiable fraction is 0.97–1.00
  undebiased and 0.61–0.77 debiased, and their minimax moves by only 10–40%. The
  multi-`Delta` conclusion does not rest on the debias being exactly right — which
  matters, because W4 has not run (section 7).

### 3.5 CRLB and degeneracy-inflation maps

`kappa_j = sqrt([F^-1]_jj * F_jj) >= 1` isolates the cost of joint estimation
from the noise level: `kappa = 1` means the parameter is informationally
orthogonal to the other two, `kappa = 10` a tenfold loss purely to parameter
trade-off. Distributions are over identifiable nodes; per-node maps are written
to `maps_*.npy` alongside `evaluation_nodes.npy` and
`evaluation_node_labels.npy`.

**Research, subset size 8.**

| arm | parameter | `kappa` median | q95 | max | fraction `kappa > 10` |
|---|---|---:|---:|---:|---:|
| m1 `(10,43)` | log_rho | 27.2 | 105 | 1.55e3 | 0.369 |
| m1 | log_V | 19.7 | 92.9 | 1.49e3 | 0.298 |
| m1 | k_io | 6.14 | 30.8 | 309 | 0.102 |
| m2 `(4,44)+(10,10)` | log_rho | **9.16** | 41.4 | 1.18e3 | 0.334 |
| m2 | log_V | **5.80** | 26.8 | 655 | 0.185 |
| m2 | k_io | **4.00** | 18.8 | 541 | 0.126 |

**Clinical, subset size 8.**

| arm | parameter | `kappa` median | q95 | max | fraction `kappa > 10` |
|---|---|---:|---:|---:|---:|
| m1 `(29,41)` | log_rho | 35.5 | 190 | 3.05e3 | 0.343 |
| m1 | log_V | 26.2 | 179 | 2.96e3 | 0.302 |
| m1 | k_io | 7.42 | 31.5 | 483 | 0.103 |
| m2 `(11,80)+(24,24)` | log_rho | **17.0** | 138 | 2.40e3 | 0.471 |
| m2 | log_V | **10.9** | 123 | 2.23e3 | 0.352 |
| m2 | k_io | **4.71** | 26.5 | 371 | 0.146 |

Adding a second diffusion time cuts median `kappa` by about 3x on `log rho` and
`log V` and by 1.5x on `k_io`. But `kappa` remains **above the pre-registered
threshold of 10 at a third to a half of identifiable nodes**, so the degeneracy is
reduced, not removed, and `rho` is the worst-affected parameter under every
acquisition tested. Note that the `kappa > 10` *fraction* can rise from m1 to m2
even as the median falls: `m = 2` identifies twice as many nodes, and the nodes it
newly reaches are the badly conditioned ones that `m = 1` could not reach at all.

Relative CRLB over identifiable nodes, research size 8:

| arm | parameter | q05 | median | q95 |
|---|---|---:|---:|---:|
| m1 `(10,43)` | log_rho | 0.172 | 0.459 | 2.58 |
| m1 | log_V | 0.143 | 0.437 | 2.54 |
| m1 | k_io | 0.131 | 0.318 | 3.07 |
| m2 `(4,44)+(10,10)` | log_rho | **0.043** | **0.154** | 1.86 |
| m2 | log_V | **0.053** | **0.162** | 1.77 |
| m2 | k_io | **0.092** | **0.230** | 2.73 |

So the best two-`Delta` research acquisition delivers about **15% fractional
precision on cell density and volume and 23% on `k_io`** at the median
identifiable node, at `N = 128` and SNR 50 — against 44–46% and 32% for the best
single-`Delta` acquisition, over a node set less than half the size.

**Where `kappa` is worst.** The worst `log rho` and `log V` nodes coincide, which
is itself informative — they are the same degenerate direction. Under research
`m = 2` the worst sits at `rho = 4.6e4` cells/uL, `V = 10.1` pL, `k_io = 40`,
`v_i = 0.468`; under clinical `m = 1` at `rho = 7.2e6`, `V = 0.066` pL,
`v_i = 0.475`. Both sit in the middle of the `v_i` band rather than at its edges.
**Whether the worst nodes lie along constant-`v_i` hyperbolae is a Phase-3
question and is deliberately not answered here**; eigendecomposition and the
sloppy-direction analysis are out of scope. *(Answered 2026-09-09 in*
[`fisher_phase3.md`](fisher_phase3.md) *§3.2: in the `(log rho, log V)` plane the
degeneracy direction is the constant-`v_i` hyperbola at a median 2.95 degrees, so
the coincident worst `log rho` and `log V` `kappa` nodes noted above are what a
shared hyperbolic degeneracy predicts. The three-parameter sloppy eigenvector,
however, points mostly along `k_io`.)*

### 3.6 The `k_io > 30` region

Phase 1 established that no column at any `k_io > 30` node reaches the top 5% of
`|dS/dk_io|`. Phase 2 reports what that costs, separately by region, without
folding it into or out of the criterion.

| scenario | arm | region | nodes | identifiable | median relative CRLB (rho, V, k_io) |
|---|---|---|---:|---:|---|
| research | m1 `(10,43)` | `k_io <= 30` | 6,990 | 3,895 | 0.426, 0.402, 0.314 |
| research | m1 | `k_io > 30` | 4,427 | **319** | 1.58, 1.67, 0.484 |
| research | m2 | `k_io <= 30` | 6,990 | 5,593 | 0.094, 0.110, 0.189 |
| research | m2 | `k_io > 30` | 4,427 | **2,760** | 0.369, 0.339, 0.345 |
| clinical | m2 | `k_io <= 30` | 6,990 | 5,974 | 0.306, 0.308, 0.227 |
| clinical | m2 | `k_io > 30` | 4,427 | 1,692 | 1.63, 1.61, 0.801 |

At a single diffusion time the coarse region is essentially unreachable: 319 of
4,427 nodes, 7%. **A second diffusion time recovers it substantially** — 2,760 of
4,427, 62%, at a `k_io` relative CRLB of 0.345 against 0.189 in the fine region,
a factor of 1.8 rather than the collapse Phase 1's sensitivity measurement might
suggest. This is not a contradiction of Phase 1: sensitivity per column really
does fall by two orders of magnitude, but with 32 well-averaged columns spanning
two diffusion times the parameter is still bounded, just poorly. Under clinical
gradients the recovery is much weaker (38% of nodes, relative CRLB 0.801).

The honest summary is that **`k_io` above 30 s^-1 is measurable but imprecise, and
only with multi-`Delta` sampling and research gradients**. That is a more useful
statement than either "it collapses" or "it is fine".

---

## 4. Is the minimax degenerate without the `k_io` restriction?

The withdrawn restriction existed to prevent exactly one failure: that with
`k_io` unbounded above 30 s^-1, the criterion would saturate on `k_io` and become
near-identical across protocols, blind to the differences it exists to detect.

**Measured, it does not happen.** The diagnosis is reported here as required,
whatever way it landed, and it landed against the concern.

| scenario | size | finite-minimax arms | within 1% of best | within 5% of best | median / best | argmax counts (rho / V / k_io) |
|---|---:|---:|---:|---:|---:|---|
| research | 8 | 320,349 / 492,528 | 0.002% | 0.018% | 2.96 | 29,675 / 167,259 / 123,415 |
| research | 12 | 276,290 / 442,270 | 0.011% | 0.316% | 2.86 | 20,043 / 157,660 / 98,587 |
| research | 16 | 240,927 / 400,065 | 0.005% | 0.122% | 2.92 | 38,995 / 100,041 / 101,891 |
| clinical | 8 | 41,057 / 100,576 | 0.010% | 0.034% | 2.88 | 5,141 / 35,899 / **17** |
| clinical | 12 | 17,587 / 50,721 | 0.006% | 0.017% | 3.10 | 2,473 / 15,097 / **17** |
| clinical | 16 | 8,435 / 24,753 | 0.024% | 0.095% | 2.48 | 2,693 / 5,739 / **3** |

Reading, stated plainly:

- **The criterion discriminates strongly.** Two to twenty arms in ten thousand
  fall within 5% of the best, and the median arm is 2.5–3.1x worse than the best.
  A saturated criterion would put a large fraction of arms within a few percent of
  each other. Nothing of the kind occurs.
- **`k_io` is not the universal argmax.** Under clinical gradients it is the
  binding parameter for **17 of 41,057** finite arms — 0.04% — and `log V` binds
  for 87%. Under research gradients `k_io` binds for 39% and `log V` for 52%.
- **The reason the criterion survives is structural, and worth stating because it
  is not luck.** Nodes at `k_io > 30` are 38.8% of the grid — just under half. A
  median over all nodes stays finite as long as more than 50% are identifiable, so
  even an acquisition that fails at every coarse-region node can still be ranked.
  Had the coarse region been a majority of the grid, the criterion would have
  collapsed and the restriction would have been necessary. It is 38.8%, so it is
  not.
- **The `(log rho, log V)`-only contrast agrees, which is the point of computing
  it.** It is a diagnostic, explicitly **not** a criterion and not a substitute for
  the unrestricted result. Under research `m = 2` at size 8 it gives 0.354 against
  the full 0.483, so excluding `k_io` would improve the apparent score by 27% and
  change the argmax from `k_io` to `log rho` — but it would not change which
  acquisition wins. Under clinical it is identical to the full criterion at every
  optimum, because `log rho` binds there anyway.

**So the user decision to withdraw the restriction cost nothing and bought
transparency.** The concern that motivated the cap was reasonable and turned out
to be unfounded, and this is the measurement that establishes it rather than an
assumption in either direction.

---

## 5. The greedy optimality gap, and what it invalidates

`m = 1` and `m = 2` are exhaustive. `m = 3` and `m = 4` are greedy forward from
the exhaustive `m = 2` optimum. To bound that greedy, the same forward step was
run at `m = 2` starting from the best `m = 1` arm, where the exhaustive answer is
known:

| scenario | size | greedy `m = 2` from best `m = 1` | exhaustive `m = 2` | gap |
|---|---:|---|---|---:|
| research | 8 | (10,43) + (11,11) → 0.6831 | (4,44) + (10,10) → 0.4831 | **+41.4%** |
| research | 12 | (10,42) + (11,12) → 0.8777 | (10,13) + (24,37) → 0.6504 | **+34.9%** |
| research | 16 | (11,41) + (12,13) → 1.061 | (12,13) + (27,42) → 0.7657 | **+38.5%** |
| clinical | 8 | (11,80) + (29,41) → 2.614 | (11,80) + (24,24) → 1.081 | **+142%** |
| clinical | 12 | (14,75) + (30,40) → 3.597 | (14,75) + (27,27) → 1.553 | **+132%** |
| clinical | 16 | (16,75) + (29,41) → 4.795 | (16,75) + (30,30) → 2.571 | **+86.6%** |

**This is a large gap and it must not be waved through.** The mechanism is visible
in the timings: the best single-`Delta` pair is a compromise between short and
long diffusion times, and greedy keeps it. The exhaustive optimum discards it
entirely and pairs a *specialised* short-`Delta` with a *specialised*
long-`Delta`. Under research, greedy retains `(10,43)` and adds a near-duplicate
`(11,11)`; the true optimum is `(4,44) + (10,10)`. Under clinical the effect is
worse because the gradient mask leaves less room to recover.

Consequences, stated rather than buried:

- **`m = 3` and `m = 4` are upper bounds, not optima.** Their true optima are
  plausibly tens of percent better. Where `m = 4` already beats `m = 2` — research
  sizes 8 and 12 — the conclusion survives a handicap and is safe. Where `m = 2`
  wins — clinical everywhere, research size 16 — **the comparison is not settled**,
  because the losing side was searched with a method measured to lose 35–142%.
- **The b-subset greedy is a separate, unmeasured greedy.** The pre-registration
  asks for its gap to be bounded against exhaustive enumeration on a reduced
  subproblem; that check was not run and is an open item (section 8).
- This does *not* affect the single- versus multi-`Delta` headline, which rests on
  the two exhaustive cardinalities.

---

## 6. The `n0_eff` amplitude sweep

Every result is reported under the fixed-`S0` bound and the `S0`-marginalized
Schur-complement bound, at `n0_eff` in `{0, 1, 2, 4, 8}` plus the known-amplitude
limit. `n0_eff = 0` is the unknown-amplitude regime: no usable `b ~ 0` reference
at all. The reference shell is modelled at the arm's shortest-TE timing pair,
where `S(0) = 1` exactly in the stored grid, so no extrapolation enters.

Median per-node CRLB ratio against the fixed-`S0` bound, at the research `m = 2`
optimum, subset size 8:

| regime | minimax | ratio log_rho | ratio log_V | ratio k_io |
|---|---:|---:|---:|---:|
| known amplitude | 0.4831 | 1.00 | 1.00 | 1.00 |
| `n0_eff = 0` | 1.616 | 1.78 | 1.14 | 1.59 |
| `n0_eff = 1` | 0.5941 | 1.21 | 1.04 | 1.18 |
| `n0_eff = 2` | 0.5425 | 1.12 | 1.03 | 1.10 |
| `n0_eff = 4` | 0.5140 | 1.06 | 1.01 | 1.06 |
| `n0_eff = 8` | 0.5001 | 1.03 | 1.01 | 1.03 |

The pattern is consistent across all twelve reported optima:

- **`n0_eff = 0` is expensive and sometimes fatal.** Among the 18 multi-`Delta`
  optima the minimax degrades by 1.9x to 16x, and in **four** of them it becomes
  infinite outright — the amplitude absorbs enough information to push the
  protocol below the 50% identifiability threshold. Per-parameter the cost across
  all 24 reported optima is 1.36–1.78x on `log rho`, 1.07–1.38x on `log V`, and
  1.30–1.72x on `k_io`. **`log rho` pays most**, which is consistent: amplitude and
  cell density both act on overall signal scale, so they are the pair that
  competes.
- **One `b ~ 0` average recovers most of it.** `n0_eff = 1` already brings every
  per-parameter ratio into 1.01–1.25.
- **The penalty is negligible by `n0_eff = 4`** (1.00–1.08) **and essentially gone
  by 8** (1.00–1.04). The Phase-0/1 assumption of `n0_eff = 4` was therefore
  sitting just past the interesting range, which is what motivated sweeping it.

**Where the informative range is: 0 to 2.** Everything above `n0_eff = 2` is a
sub-10% effect. A protocol designer should acquire at least two `b ~ 0` averages
and then stop worrying about amplitude variance.

**What this does not establish.** This is a **variance** statement. It says
nothing about an amplitude reference that is *biased* rather than merely noisy,
which is the Jackson-thesis situation: a lowest shell at `b = 50` rather than 0
asserts `S(50) = 1` and so carries a systematic error of roughly
`1 - exp(-50 D)` on every point. **Averaging reduces noise; it cannot reduce
bias.** The Phase-0/1 finding that four `b ~ 0` averages cost under 0.4% is
therefore not reassurance about the thesis acquisition. That distinction is
Phase 4 hypothesis H4, it must reuse
`madi.fisher_crlb.amplitude_prior_precision` rather than duplicate it, and it is
explicitly **not resolved here**.

---

## 7. Limitations

Stated here rather than left implicit.

- **Truncation bias is the dominant systematic error, not Monte-Carlo noise.**
  Phase 1 measured roughly 1.4% relative on `rho` and 1.8% on `V` at `k = 1`,
  which propagates to about 2.7% and 3.7% in the Fisher diagonal and about
  1.4–1.8% on a CRLB. Richardson extrapolation was considered and deliberately
  not adopted, because `k = 2` coverage is only 66% (`rho`) and 47% (`V`) of
  nodes and clusters away from the mask boundary, which would produce a
  heterogeneous derivative field for a sub-2% effect.
  **Which Phase-2 conclusions could a 2–3% shift change?** Not the
  single-versus-multi-`Delta` headline: single-`Delta` misses the identifiability
  threshold by 13 percentage points, not by 2%. Not the `n0_eff` reading, whose
  steps are 6–78%. **But the `m = 2` versus `m = 4` comparisons are within
  reach** — research size 12 separates 0.6287 from 0.6504, a 3.4% difference, and
  research size 8's `m = 3` versus `m = 4` separates 0.4649 from 0.4187, 11%.
  The size-12 `m = 4` preference is flagged specifically as within the systematic
  error, and the greedy gap of section 5 dominates even that.
- **W4, the independent-seed replicate, has not run.** It is the only independent
  check that `signal_variance` is correct in absolute terms, and every debias in
  this document rests on it. Its priority is genuinely low — with `beta` around
  `1e-5` a 2x calibration error moves the Fisher matrix by about 0.002% — but
  Phase 2 has raised the stakes slightly: the debias flips the single-`Delta`
  verdict, so a *gross* error in `signal_variance` would matter even though a
  factor-of-two error would not. It remains a pre-freeze sanity check, not a
  Phase-2 gate.
- **The debias is endpoint-only outside the 8 diagnostic timing pairs**, which
  overstates `Var(J_hat)` and therefore understates identifiability. Calibrated at
  6 percentage points of identifiable fraction at `(20,50)`. Conservative in the
  safe direction, but not exact.
- **`m = 3` and `m = 4` are greedy, with a measured `m = 2` gap of 35–142%**
  (section 5). The b-subset greedy gap is unmeasured.
- **Subset sizes were bracketed from above.** Size 8 wins monotonically, so the
  optimum may lie below the smallest pre-registered size and was not searched.
- **`N = 128` is a declared point, not a swept one**, and because the Rician mask
  depends on `n_c = N/n_selected`, the ranking is not `N`-independent. The
  sensitivity run was not performed.
- **The swept acquisition space was bounded by the gradient scenarios in two
  distinct ways, and only one of them was intended.** Intended: a scenario admits
  only columns it can play, which is why the research sweep at subset size 8
  enumerated 993 of the 1,245 stored timing pairs. Unintended, and corrected on
  2026-09-06: the Phase-1 substrate itself held only the research-feasible
  columns, so an *unrestricted* sweep was not merely unselected but impossible.
  The two coincide for the clinical and research scenarios reported here — which
  is why no number moves — but they are not the same thing, and only the first
  survives.
- **136 of 369 `(rho, V)` pairs have no Fisher matrix at all**, being the
  mask-band edge where a `rho` or `V` neighbour falls outside
  `0.40 <= rho*V*1e-6 <= 0.99`. Phase 4's unrealistic-volume hypothesis is a
  mask-boundary hypothesis, so Phase 2 is blind exactly where Phase 4 will look.
- **The `k_io > 30` sensitivity collapse is unmitigated in the criterion**, by
  user decision, and its cost is reported in section 3.6 rather than corrected for.
- Phase 3 material — eigendecomposition, sloppy directions, the constant-`v_i`
  hyperbola hypothesis — is **out of scope and not pre-empted**, even where
  section 3.5's coincident worst-`kappa` nodes for `log rho` and `log V` invite
  the inference. *(Executed separately on 2026-09-09;*
  [`fisher_phase3.md`](fisher_phase3.md)*. This limitation stands as written for
  this record: nothing here was computed from it.)*
- Phase 5 item 5.1, the continuous averaging-allocation optimization under a
  fixed scan-time budget, is out of scope. Phase 2 evaluates the declared sweep
  at equal allocation across selected columns. That boundary is drawn correctly:
  unequal allocation is a different and larger optimization, and running it here
  would have made the matched-budget comparison incomparable.

---

## 8. Open decisions, for the user

Things Phase 2 could not settle on its own, in descending order of consequence.

1. **Whether to re-search `m = 3` and `m = 4` exhaustively or semi-exhaustively.**
   The measured greedy gap is 35–142%, which is large enough that the
   `m = 2`-versus-`m = 4` verdict is not established under clinical gradients or
   at research size 16. A bounded compromise — exhaustive `m = 3` over the top
   ~200 timing pairs, about `1.3 x 10^6` arms — would cost a few hours and would
   settle it. **This is the single largest open item.**
2. **Whether to extend the b-subset sizes below 8.** Size 8 wins monotonically in
   both scenarios, so the pre-registered set brackets the optimum from one side
   only. Sizes 4 and 6 would say whether the trend continues or turns.
3. **Whether to sweep `N`.** The ranking is not budget-independent because the
   Rician mask depends on `n_c`. `N` in `{64, 256}` would establish whether the
   optimal timings and the optimal `m` are stable, and each re-run is about
   2 hours.
4. **Whether to bound the b-subset greedy gap**, as the pre-registration asks, by
   exhaustive enumeration on the declared reduced subproblem.
5. **Whether the clinical/research choice should stay deferred.** It remains
   deferred as instructed. The two scenarios now disagree about the optimal arm
   cardinality (`m = 2` clinical, `m = 4` research at sizes 8 and 12), so the
   choice has begun to have consequences beyond precision numbers.
6. **Whether to run W4** before Phase 3, given that the debias now carries a
   qualitative verdict rather than only a quantitative correction.

---

## 9. Reproduction

*Updated 2026-09-06 for the corrected column domain. A fresh agent should follow
this section and not reconstruct the incident from git history.*

Roughly 2 hours of compute and about 10 GiB of scratch beyond the Phase-1
fields. The caches are built once and reused.

```bash
LIB=data/libraries/madi_dense_universal_remediated.npz
PH1=/home/jaden/madi_fisher_runs/full_domain/phase1     # the UNRESTRICTED substrate
OUT=/home/jaden/madi_fisher_runs/full_domain

python -m scripts.build_fisher_phase2_cache --artifact "$LIB" --phase1 "$PH1" --cache-dir "$OUT/cache"
python -m scripts.run_fisher_phase2 --artifact "$LIB" --phase1 "$PH1" \
    --cache-dir "$OUT/cache" --output-dir "$OUT/phase2_N128" --budget-images 128 --m2-block 48
python -m scripts.summarize_fisher_phase2 "$OUT/phase2_N128/phase2_report.json"
```

Both the cache builder and the sweep print the column-domain banner and record
`column_domain` in their manifests; a healthy run says
`COLUMN DOMAIN COMPLETE — all 31125 stored (delta, Delta, b) columns`. Each
gradient scenario now asserts that the substrate carries every column it admits
and raises rather than quietly scoring a smaller acquisition, so pointing
`--phase1` at a restricted Phase-1 directory is safe: it either reproduces or it
tells you why it cannot.

The historical run used `PH1=/home/jaden/madi_fisher_runs/final_varfix/phase1`
and `OUT=/home/jaden/madi_fisher_runs/phase2`, whose outputs are retained.

`--node-stride` subsamples nodes for a smoke run and must not be used for a
reported result; the manifest records it. `--skip-m2` likewise.

The recorded `runtime_seconds` of 42,325 s is **wall clock and includes a host
suspension of about 9.7 hours between the `research size=12` and
`research size=16` blocks.** Actual compute was approximately 7,300 s. This is
noted so the figure is not mistaken for a performance characteristic.

Outputs: `phase2_report.json` (the full record, including every per-node summary
and both bounds at every regime), `summary.txt`, per-node `maps_*.npy` for the
`m = 1` through `m = 4` optima at subset size 8, under both the known-amplitude
and the `n0_eff = 4` regimes, in both gradient scenarios, and `evaluation_nodes.npy` /
`evaluation_node_labels.npy` giving the `(rho_index, V_index, k_io_index)` and
`(rho, V, k_io)` of every map row.

---

## 10. Amendment log

This record is amended in place. `madi/fisher_crlb_preregistration.json` and
`fisher_crlb_analysis_plan.md` carry matching entries.

### 2026-09-09-phase3-forward — Phase 3 ran; this record is confirmed, not revised

- **Previously:** section 3.5 deferred the constant-`v_i` question to Phase 3,
  section 7 listed all Phase-3 material as out of scope and not pre-empted, and
  the 2026-09-06 carry-forward said the model-layer identifiability question was
  now askable and had not been answered.
- **Now:** a forward pointer at the top of this file, a pointer beside each of
  those two deferrals, and nothing else. No result, table, figure or conclusion
  in this record was edited.
- **Why:** Phase 3 executed on 2026-09-09
  ([`fisher_phase3.md`](fisher_phase3.md)) and answered both deferrals. It also
  re-formed the four optimal arms of section 3.2 from the same cache by a
  different accumulation path — one streaming pass over all 1,245 timing pairs
  with per-domain re-weighting — and reproduced 72 of 72 compared quantities at a
  maximum relative difference of exactly 0.0 across all six `S0` regimes,
  including the `+inf` minimax of both `m = 1` arms. That is independent
  confirmation of this record.
- **Not changed:** every executed Phase-2 number, and the limitation in section 7
  as written, which is a statement about what this record computed.

### 2026-09-06-substrate-domain — the substrate was restricted; the results are not affected

- **Previously:** the sweep consumed a Phase-1 substrate holding 24,111 of the
  31,125 stored `(delta, Delta, b)` columns, and it intersected each scenario's
  gradient mask with that substrate silently
  (`feasible_column[full] & (position_of[full] >= 0)`), so an absent column and
  an inadmissible one were indistinguishable.
- **Now:** the sweep reads a substrate spanning every stored column, asserts via
  `madi.fisher_crlb.require_columns` that it carries every column each declared
  scenario admits, and records `column_domain` in `phase2_report.json`. The
  gradient mask keeps its role and its place: applied at evaluation, per
  scenario, never pooled, still unchosen between clinical and research.
- **Why:** the restriction was a 300 mT/m ceiling applied at extraction rather
  than at evaluation. See [`fisher_domain_audit.md`](fisher_domain_audit.md).
- **Effect on this record: none, and that is verified rather than argued.** The
  restricted substrate contained every column either scenario admits, because
  clinical ⊂ research ⊂ the old substrate. Concretely, on the corrected
  substrate:
  - every timing pair's candidate column set is **identical** to the one this
    run used, for both scenarios — 0 of 1,245 pairs differ, 449 clinical and 993
    research pairs carry at least 8 diffusion-weighted columns, and the total
    admissible column counts are unchanged at 8,754 and 22,836;
  - the Phase-2 caches are **bit-for-bit identical** on all 24,081
    research-feasible columns, 453,204,420 cells per member for both `vectors`
    and `signal_variance`;
  - the Phase-1 derivative fields the sweep cross-checks against are bit-for-bit
    identical on the overlap (see the Phase-0/1 framework's amendment log).

  Identical inputs over an identical search space give identical output. That was
  **confirmed rather than inferred**: the whole `N = 128` sweep was re-executed
  against the corrected substrate and diffed against this record. **2,911
  reported quantities match exactly, zero mismatches**, across every arm, subset
  size, scenario and `S0` regime; all 32 per-node `maps_*.npy` files are
  identical; and all six exhaustive `m = 2` optima reproduce to every printed
  digit at identical arm counts. The re-run lives at
  `/home/jaden/madi_fisher_runs/full_domain/phase2_N128/` and this record's own
  outputs were not overwritten.
- **What changes is the scope of the claims, not the numbers.** See the
  carry-forward note at the top of this document.

### 2026-09-06-initial — first execution of Phase 2

- **Previously:** no Phase-2 record existed. Phase 2 was specified in
  `fisher_crlb_analysis_plan.md` section 5 and in the pre-registration's
  `protocol_sweep` block, and had not been run.
- **Now:** executed on the complete 369-group artifact at `N = 128`, SNR 50,
  under both gradient scenarios, with the `k_io` restriction withdrawn and the
  gradient-scenario choice deferred, both by user decision.
- **Specification gaps closed at execution**, each recorded as its own dated
  amendment in the plan and the pre-registration: the search strategy
  (`2026-09-05-phase2-specification`), the averaging-aware Rician mask
  (`2026-09-05-phase2-averaging-aware-mask`), and the median node aggregate
  (`2026-09-05-phase2-node-aggregation`).
- **Corrected an earlier claim of my own:** the `2026-09-05-protocol-sweep`
  amendment asserted that protocol ranking at matched `N` is independent of `N`.
  It is not, because the Rician mask depends on the per-column averaging. The
  pre-registration entry is corrected in place and says so.
