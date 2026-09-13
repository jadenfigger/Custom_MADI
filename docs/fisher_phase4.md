# Fisher / CRLB Phase 4

Executed 2026-09-10. Classification: CURRENT (see [`INDEX.md`](INDEX.md)).
Specification: [`fisher_crlb_analysis_plan.md`](fisher_crlb_analysis_plan.md) §7,
as amended `2026-09-10-phase4-dataset-and-h4-scope`,
`2026-09-10-phase4-trust-floor-fit-forms` and
`2026-09-10-phase4-v_i-estimable-bound`. Every declaration was pre-registered in
the `phase4` block of `madi/fisher_crlb_preregistration.json` **before any fit
result was read**. One control was added after the first report was read — the
geometric null of §2.5 — and it is logged as post-hoc, with the original
criterion left in place (amendment `2026-09-10-phase4-geometric-null`).

## Verdict

**PHASE 4 EXECUTED, ON ONE SUBJECT AND ONE ACQUISITION. THE UNREALISTIC
VOLUMES ARE SIGNAL THE CELLULAR LIBRARY CANNOT EXPLAIN, PUSHED INTO ONE CORNER
OF THE LIBRARY. THAT IS H2, NOT THE DEGENERACY RIDGE OF H1.** Seven results
carry beyond this phase.

1. **H2 is supported: the pathological voxels fit badly and diffuse like free
   water.** Blow-up voxels — fitted `V > 20 pL`, 24.0% of MAP voxels and 20.3% of
   Bayes voxels — have a residual **16.6 times** their own noise floor at the
   median, against **0.34** for the rest (MAP; probability of superiority 0.90;
   Bayes 22.8 against 0.81, 0.89). Their ADC at `b <= 1000` has a median of
   **2.42 µm²/ms** against **0.92** (0.99), and **21.9%** of them decay *faster
   than free water itself*, against 0.03% of the rest. This analysis does not say
   *which* out-of-model signal it is — free water, CSF partial volume, IVIM or an
   artefact — only that the cellular library does not describe it.
2. **The blow-up is mostly one library entry.** **79.9%** of blow-up voxels
   (23,685 of 29,642) are assigned the same volume, **41.79 pL**. **85.3%** sit on
   the lowest `rho` node of the grid (10,000 cells/µL), a node **no other voxel
   reaches** (0 of 93,959), and **90%** sit at `v_i < 0.45`, the least-restricted
   edge of the library band. A fitted cell volume above 20 pL is therefore, for
   the most part, the address of the library's `(rho_min, v_i_min)` corner rather
   than a measured volume. Plan §4.4 predicted estimates landing on the mask
   boundary "not as a random fitting failure would". They do — but on the *lower*
   `v_i` edge at about 42 pL, not at the 99 pL reachable maximum §4.4 named; only
   3.0% of blow-ups reach the upper corner (91.6 pL).
3. **H1 is not supported as the cause.** Its residual prediction fails outright
   (result 1). Its second signature, estimates sliding along the constant-`v_i`
   hyperbola when a fit is perturbed, is **not distinguishable from the shape of
   the mask**: against a null that keeps the band and the fitted distribution and
   permutes only which voxel a move belongs to, the probability that the null
   angle exceeds the observed one is **0.43–0.60** for every perturbation. The
   degeneracy is real — Phase 3 measured it, and at this acquisition `kappa` is in
   the hundreds wherever the Fisher matrix inverts at all — but it is not what
   drives the blow-up.
4. **H4 halves the moderate blow-ups and grows the extreme tail.** Fitting `S0`
   per voxel cuts the fraction above 20 pL by **47%** (MAP) and **49%** (Bayes),
   and by 36–49% at every cutoff from 5 to 30 pL, while *raising* the fraction
   above 50 pL by 60% (MAP) and 25% (Bayes). The thesis's own H4 mechanism, a
   biased `b = 50` amplitude reference, cannot occur on this acquisition, which
   has true `b = 0` volumes, and is untested.
5. **H3 removes the extreme tail and leaves the bulk.** Masking library values
   below the trust floor moves the fraction above 20 pL by only **−5% to +3%**
   across both masking forms and both estimators, but cuts the fraction above
   50 pL by **13% to 100%** (Bayes columns to Bayes candidates) and above 90 pL by
   46–88% (MAP). The most extreme volumes come from library entries whose high-`b`
   signal is below the floor; the 20–50 pL bulk does not.
6. **At this acquisition the Fisher matrix is blind exactly where the pathology
   lives, and the §4.5 replacement reports almost nothing.** **98.1%** of blow-up
   voxels map to a node with no interior `(rho, V)` stencil. The debiased Fisher
   matrix is positive definite at 1.57% of nodes. `rho` and `V` are separately
   reportable (`kappa < 10`) at **0%** of voxels. `v_i` carries a bound at
   **0.06%** (strict) to **34.5%** (defect tolerance 0.20) of voxels, but at
   **≤ 0.12%** of blow-up voxels at any tolerance. The Bayes posterior spread
   could not be validated against the CRLB, because a per-parameter CRLB exists
   at only 106 voxels, and there it does not rank-track the posterior (Spearman
   −0.14 for `V`, −0.53 for `rho`).
7. **Two defects in existing outputs, reported and not repaired here.** The
   shipped Bayes `vi_map` multiplies posterior means, and **55.8%** of its voxels
   fall outside the physical band `[0.40, 0.99]` (median 1.46). And
   `data/outputs/madi_output_glioma_v4.0/map` is not a glioma fit: it is edema
   sub-187 fitted with the sub-187 mask, filed beside the correctly masked sub-125
   Bayes run.

**Validation that carries the most weight.** The Phase-4 baseline Bayes fit,
re-run through a fitter now carrying the new `--trust-floor` code (off by
default), reproduces the executed `madi_output_glioma_v4.0/bayes_s-auto` record
**bit for bit on all eight maps**. The trust-floor record matches an independent
direct count (5,174 cells below the floor). The Phase-4 per-column Fisher sums are
pinned to the audited Fisher primitives to 1e-12, and the estimable `v_i` bound
matches the exact contrast CRLB on every invertible node to 1.1e-8.

**Scope, stated up front.** One subject (Mayo_Glioma sub-125) at one
single-`Delta`, five-shell clinical acquisition, SNR 25. Nothing here shows that
the mechanism carries to multi-`Delta` acquisitions, to rodent data, or to the
OHSU / MADI III data the pathology was first reported in.

---

## 1. What was run

### 1.1 The dataset and the fit arms

| | |
|---|---|
| Subject | Mayo_Glioma sub-125, 123,601 brain-mask voxels |
| Acquisition | PGSE `delta = 20 ms`, `Delta = 50 ms`; `b = 0` ×5, 500 ×6, 1000 ×18, 1500 ×24, 2000 ×30, 2500 ×36 s/mm², powder-averaged per shell |
| Noise | Rician σ = 172.09 (48-iteration background dilation, as the fits used), median `S0` = 4289.5, SNR at `b = 0` = 24.9 |
| Library | `data/libraries/madi_dense_universal_remediated.npz`, `v_i` band `[0.40, 0.99]`, free water excluded, no `rho_max` |
| Why this dataset | the one repository dataset with an executed, correctly masked reference fit showing the pathology; sub-059 shares the protocol |

Eight fit arms, one condition varied per arm, each under MAP and Bayes
(`scripts/run_fisher_phase4_fits.sh`):

| Arm | Condition switched on | Hypothesis | Features |
|---|---|---|---:|
| `baseline` | none — the executed v4.0 configuration | — | 5 |
| `fit_s0` | `--fit-s0` | H4, item 4.1 | 5 |
| `trust_floor_column` | `--trust-floor --trust-floor-mode column` | H3, item 4.2 | 3 |
| `trust_floor_candidate` | `--trust-floor --trust-floor-mode candidate` | H3, item 4.2 | 5 |

The Bayes free-`S0` arm matches `sigma_m` on the baseline Bayes run's median
`n_eff` (34.42) via `--target-n-eff`, the procedure
[`fitting_methods.md`](fitting_methods.md) prescribes because `sigma_m` is not
comparable between the fixed- and free-`S0` branches.

### 1.2 Conventions settled before any result was read

All pre-registered under `phase4`.

**(a) The trust floor's two fit-time forms.** Plan §2.6 applies the floor per
`(entry, column)` inside a Fisher sum. A fit instead compares candidates, and a
per-cell residual mask would give each candidate a different residual
dimensionality — incomparable residuals, and a selection bias toward exactly the
entries the floor indicts. `column` mode drops a column at which any candidate
is below the floor (here `b = 2000` and `b = 2500`, leaving three columns for
three parameters); `candidate` mode drops an entry below the floor at any column
(here 3,355 of 18,819 entries). See [`fitting_methods.md`](fitting_methods.md).

**(b) The pathology definition.** Fitted `V > 20 pL`, the Jackson-thesis cutoff
this phase is asked to replace — cited, not chosen — with every contrast also
reported at 5, 10, 15, 30, 50 and 90 pL. A voxel at `rho = V = 0` is carried as
not fitted, never as a volume of zero.

**(c) The H1/H2 discriminator.** Goodness-of-fit ratio = residual ÷ the residual
noise alone would leave, `sum_c sigma² / (S0_voxel² n_c)`, compared between
groups by probability of superiority. It is the zero-fitted-degrees-of-freedom
expectation, so its absolute level is not a calibrated goodness of fit; the
comparison between groups is the claim.

**(d) The voxel-to-node join.** Nearest canonical node, matched in `log rho` and
`log V` and linearly in `k_io`. A voxel whose node carries no Fisher matrix gets
no bound, never a neighbour's. MAP labels land within 0.13 of half a grid step of
their node, as realised labels should.

**(e) The Fisher geometry at the acquisition.** Phase 2's `pair_contributions`
called once per column, each weighted by its own averaging — which Phase 3's
one-noise-level-per-pair `Domain` cannot express. At one timing pair the TE/T2
factor is common to every column, so `kappa`, angles and defects are identical
under the pre-registered TE model; only absolute bounds carry the measured scale.

**(f) The §4.5 bound.** At this acquisition the debiased Fisher matrix inverts at
1.57% of nodes, so a plain CRLB on `log v_i` would exist almost nowhere. The error
bar is `madi.fisher_crlb.estimable_rho_V_contrast_bound` — the bound from the
directions the data inform, on the `k_io`-profiled block — reported with its
**estimability defect**, the share of the `v_i` contrast lying in uninformative
directions. The defect tolerance below which `v_i` is reported is not
pre-registered and **no tolerance is nominated**; every one is reported.

**(g) The guards.** A fit arm is refused unless its recorded mask hash equals the
declared mask's and its status is `completed`. Every signal-shaping loader
argument is taken from the baseline fit's own record — the fitter's CLI default
background dilation is 48 iterations, its function default 16, and the wrong one
would silently change the Rician correction — and the recovered σ is asserted
equal to the recorded one.

### 1.3 Added after the first report was read

Stated separately, because the pre-registration's value is that nothing in §1.2
could have been tuned on the outcome.

| Addition | Why | Does it change a criterion? |
|---|---|---|
| **Geometric null for the ridge test** (§2.5) | the library band is about 23 times longer along the hyperbola than across it, so *any* move inside it makes a small angle; the pre-registered 45° uniform null credited the mask's shape to the degeneracy | **yes — it reverses the ridge reading**, and is logged as amendment `2026-09-10-phase4-geometric-null`; the original criterion is kept and still reported |
| Why a node has no Fisher matrix, split into `k_io` grid end and `(rho, V)` band edge; `rho`, `v_i` and `k_io` railing of the blow-ups; non-physical ADC counts | the first report showed 98% of blow-ups had no bound and did not say why | no — descriptive quantities only |
| `estimable_rho_V_contrast_bound` returns NaN, not 0, where no part of the contrast is informed | the first report showed a median "bound" of 0 on voxels whose defect was 1 | no — the coverage gate never admitted those voxels; the map and one summary were misleading |
| ADC axis display range, zero-safe log axes, de-collided labels in the figures | the first figures were unreadable | no |

---

## 2. Results

### 2.1 The two branches — [`fig4_1`](provenance/figures/fisher_phase4/fig4_1_adc_vs_volume.png)

The characterization the pathology was named from reproduces. Plotted against
ADC, the Bayes fits separate into a branch where volume rises with ADC and a
branch that saturates near 40 pL at high ADC; the MAP fits show the second branch
as a single horizontal line at 41.79 pL running from ADC ≈ 1.5 past free water.

| | blow-up (`V > 20 pL`) | rest | probability of superiority |
|---|---:|---:|---:|
| ADC, `b <= 1000` (µm²/ms), MAP median | **2.42** | 0.92 | 0.99 |
| ADC, all shells, MAP median | 1.69 | 0.74 | 0.98 |
| ADC above free water (3.0) | **21.9%** | 0.03% | — |
| ADC below zero | 0.0% | 0.86% | — |
| `log v_i` position across the band (0 = lower edge), median | **0.043** | 0.501 | 0.27 |
| `V` / reachable ceiling at that `rho`, median | 0.42 | — | — |

Where the blow-ups sit, MAP:

| Location | share of blow-ups |
|---|---:|
| lowest `rho` node, 10,000 cells/µL | **85.3%** |
| fifth-lowest `rho` node, 15,505 cells/µL | 9.1% |
| `v_i < 0.45` | **90.0%** |
| `v_i > 0.90` | 5.3% |
| exactly `V = 41.79 pL` | **79.9%** |
| `30 < V <= 50 pL` / `20 < V <= 30` / `50 < V <= 90` / `90 < V` | 82.5% / 10.8% / 3.6% / 3.0% |
| `k_io` at the 130 s⁻¹ grid end | 40.4% |

**No** non-blow-up voxel (0 of 93,959) sits on any of the six lowest `rho` nodes.

### 2.2 H4 — the amplitude — [`fig4_2`](provenance/figures/fisher_phase4/fig4_2_condition_arms.png)

Relative change in the fraction of voxels above each cutoff when `S0` is fitted
per voxel, on the voxels both fits reached:

| cutoff (pL) | 5 | 10 | 15 | 20 | 30 | 50 | 90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAP | −38% | −36% | −47% | **−47%** | −46% | **+60%** | −70% |
| Bayes | −46% | −46% | −48% | **−49%** | −44% | **+25%** | — (none) |

At 20 pL, MAP: 0.240 → 0.128; 15,276 voxels resolved, 14,366 stayed, 1,432 newly
blew up. Median fitted volume 5.38 → 2.10 pL. The Bayes free-`S0` fit left 884
voxels unfitted, where every candidate's least-squares `S0` was negative.

On the Fisher side, the five `b = 0` volumes pin the amplitude strongly
(prior precision over `F_s0s0`, median 0.94): the identifiable node fraction moves
1.57% → 1.40% and the `k_io`-profiled sloppy angle 3.84° → 3.89°. With no
amplitude reference at all, no node is identifiable and the angle is 6.03°.

**Reading.** Amplitude freedom absorbs part of what the tissue model cannot, so
about half the moderate blow-ups resolve; but it does not remove the tail, and
the 50 pL fraction grows. That is what an out-of-model shape (H2) predicts and
what a pure amplitude error would not: a scale error is fully absorbed by a free
scale. The thesis mechanism — a *biased* reference from a `b = 50` shell — is a
bias, which averaging and a fitted scale cannot remove; it needs a dataset with
that structure, and none exists in the repository.

### 2.3 H3 — the trust floor

Relative change in the fraction above each cutoff:

| arm | 5 | 10 | 15 | 20 | 30 | 50 | 90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| columns, MAP | −5% | −0% | +4% | **+3%** | +3% | **−53%** | −46% |
| candidates, MAP | −0% | +1% | +1% | **+2%** | +13% | **−78%** | −88% |
| columns, Bayes | +9% | +10% | +3% | **−5%** | −3% | −13% | — |
| candidates, Bayes | 0% | 0% | −0% | **+3%** | +22% | **−100%** | — |

At 20 pL the candidate MAP arm resolves 23 voxels and newly blows up 507; the
column MAP arm resolves 1,112 and newly blows up 1,957. With the floor on in
candidate mode, the largest Bayes volume anywhere falls from 60.6 to 41.8 pL.

**Reading.** H3 as stated — the blow-up shrinks when the sub-floor columns are
masked — fails for the blow-up as defined. It holds for the extreme tail: the
volumes beyond 50 pL are largely made of library entries whose high-`b` values are
Monte-Carlo noise.

### 2.4 H1 against H2 — the residuals — [`fig4_3`](provenance/figures/fisher_phase4/fig4_3_residual_discriminator.png)

| | blow-up median | rest median | ratio | probability of superiority |
|---|---:|---:|---:|---:|
| goodness-of-fit ratio, MAP | **16.57** | 0.34 | 48.8 | **0.90** |
| goodness-of-fit ratio, Bayes | **22.79** | 0.81 | 28.2 | **0.89** |
| raw residual, MAP | 0.0034 | 0.00024 | 14.1 | 0.80 |
| raw residual, Bayes | 0.0046 | 0.00050 | 9.2 | 0.80 |

H1 predicted low residuals (many entries along the ridge fit comparably and one
was chosen); H2 predicted high residuals (nothing fits). The distributions barely
overlap, and they sit on the H2 side. The rest of the brain sits below the noise
floor at the median, as a three-parameter fit to five columns should.

### 2.5 H1 — where the pathology sits, and whether estimates slide — [`fig4_4`](provenance/figures/fisher_phase4/fig4_4_ridge.png)

**The Fisher overlay is blind where the pathology lives.** Why a voxel's node
carries no Fisher matrix, at this acquisition:

| node | blow-up | rest |
|---|---:|---:|
| has a Fisher matrix | 1.7% | 47.3% |
| `(rho, V)` grid boundary only — no interior `rho` or `V` stencil | **57.0%** | 39.0% |
| `k_io` grid end only | 0.2% | 7.9% |
| both | **41.1%** | 5.8% |

So 98.1% of blow-up voxels sit on the `(rho, V)` boundary, and none (0 of 29,642)
lands on a node where the Fisher matrix inverts — against 0.08% of the rest. The
overlay of §4.4 on this acquisition's `kappa` map therefore compares 0 blow-up
voxels with 79; the Phase-3 model-layer `kappa_V` reaches 14 blow-up voxels. Plan
§7's standing caveat — the unrealistic-volume hypothesis is a mask-boundary
hypothesis and Phase 2 is blind exactly there — is now measured, and it is total.

**The ridge signature does not survive the geometric null.** Estimates were
perturbed three ways, and the direction each moved in `(log rho, log V)` was
measured against the constant-`v_i` hyperbola:

| perturbation | voxels moved | observed angle, median | geometric null, median | P(null > observed) | `|Δ log v_i| / |Δ log V|`, observed / null |
|---|---:|---:|---:|---:|---:|
| baseline MAP → `S0` fitted | 113,023 | 3.58° | 4.29° | **0.53** | 0.128 / 0.162 |
| baseline MAP → trust floor, columns | 79,373 | 2.08° | 4.30° | **0.60** | 0.070 / 0.162 |
| baseline MAP → trust floor, candidates | 5,873 | 2.03° | 4.51° | **0.43** | 0.068 / 0.163 |
| MAP mode → Bayes posterior mean | 122,710 | 26.85° | 13.96° | 0.24 | 1.36 / 0.56 |

Against the pre-registered uniform null (45°) the first three rows look like
decisive support for H1 — 54–76% of moves within 10°. The geometric null keeps
each voxel's starting estimate and pairs it with a randomly chosen *other*
voxel's perturbed estimate, preserving the band and the fitted distribution; it
gives 75% within 10° on its own. A probability near 0.5 means an observed move is
no more along the hyperbola than a random move of the same fitted population. The
Bayes row is not a ridge test at all: a posterior mean averages `rho` and `V`
separately in linear space and leaves the band (result 7).

### 2.6 The Fisher geometry at this acquisition

Node-level, the 11,417 evaluation nodes of Phases 2 and 3, known amplitude:

| quantity | value |
|---|---|
| positive-definite (identifiable) node fraction | **1.57%** (179 nodes) |
| `kappa` median over those nodes, `rho` / `V` / `k_io` | 298 / 229 / 76 |
| relative CRLB median, `log rho` / `log V` | 2.32 / 2.09 (±230% / ±210%) |
| `k_io`-profiled sloppy angle to the hyperbola, median | 3.84° |
| non-positive eigenvalues of the profiled block: 0 / 1 / 2 | 179 / 10,124 / 1,114 nodes |
| `v_i` defect, median | 0.067 |
| with the trust-floor columns dropped (3 columns) | 0% identifiable; angle 3.44° |

This is Phase 2's single-`Delta` result at the acquisition that produced the
pathology: no three-parameter bound almost anywhere, and a degeneracy that still
runs along the hyperbola (Phase 3) where it can be measured.

### 2.7 The §4.5 replacement — [`fig4_5`](provenance/figures/fisher_phase4/fig4_5_replacement.png)

The replacement was specified "if H1 is confirmed". H1 is not, and the rule is
reported anyway, because what it would deliver at a real acquisition is itself a
finding.

| `v_i` defect tolerance | strict (0) | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---:|---:|---:|---:|---:|---:|
| voxels reporting `v_i` with a bound | 0.06% | 1.1% | 2.4% | 10.7% | 21.4% | **34.5%** |
| blow-up voxels reporting it | 0% | 0.01% | 0.03% | 0.05% | 0.08% | **0.12%** |
| bound where reported, median | 1.23 | 0.070 | 0.068 | 0.057 | 0.063 | 0.067 |

- `rho` and `V` separately (`kappa < 10`): **0%** of voxels. Nothing at this
  acquisition licenses a separate cell density or cell volume.
- Where `v_i` is reported, its bound is about **6–7%** — the stiff combination is
  genuinely measured — but the coverage question is decided by the tolerance, and
  the blow-ups are excluded at every tolerance because their nodes have no Fisher
  matrix.
- **Posterior flag validation fails for lack of a comparator.** A per-parameter
  CRLB exists at 106 of 123,601 voxels. There the Bayes fractional posterior SD is
  about 1/40 of the CRLB on `log V` (median ratio 0.024) — capped by the extent of
  the candidate set, not set by the data — and ranks against it at Spearman −0.14
  (`V`) and −0.53 (`rho`).

### 2.8 Defects found in existing outputs

**The Bayes `vi_map` is a product of means.** `derive_voxelwise_biomarkers` is
called with `rho_mean` and `V_mean`, so `vi_map = <rho><V> 1e-6`, not `<rho V>`.
In 55.8% of voxels that product lies outside `[0.40, 0.99]` — median 1.46, maximum
15.2 — which no library entry can take. `madi/biomarkers.py` already refuses the
same identity for ROI medians; the posterior-mean call site reintroduces it one
level up. The MAP `vi_map` is unaffected. Not changed here.

**`madi_output_glioma_v4.0/map` is edema sub-187.** Its command fits
`edema/.../sub-187_desc-preproc_dwi.nii.gz` with the sub-187 mask (95,655 voxels).
The data are left untouched; the Phase-4 runner refuses any arm whose mask hash
differs from the declared mask, which is the guard this record calls for.

---

## 3. What Phase 4 answers that earlier phases left open

- **[`fisher_phase3.md`](fisher_phase3.md) §3.3** predicted the pathology lives in
  the low-density, large-cell corner, where no `k_io` node is identifiable.
  **Confirmed as a location**: 85.3% of blow-ups sit on the lowest `rho` node. Its
  open decision 4 asked whether that table should be an H1 prior or a confound.
  The answer here is neither in the form it was asked: the corner is where H2
  signal is pushed, and the ridge that would have made it an H1 mechanism does
  not show up above geometry.
- **[`fisher_phase2.md`](fisher_phase2.md) §3.1** warned that Phase 2 is blind
  exactly where Phase 4 would look. **Measured at 98.1%** of blow-up voxels.
- **Plan §4.4's boundary prediction** holds in form and differs in place: the
  lower `v_i` edge near 42 pL, not the reachable maximum near 99 pL.

---

## 4. Limitations

- **One subject, one acquisition, one SNR.** The mechanism is shown for a
  single-`Delta`, five-shell clinical acquisition at SNR 25. An acquisition whose
  Fisher matrix inverts could behave differently, and the rodent data the thesis
  also saw the structure in were not analysed.
- **H2 is supported, not identified.** High residuals, free-water-like ADC and
  super-free-water decay say the cellular library does not describe the signal.
  They do not separate free water from CSF partial volume, IVIM or artefact.
- **The thesis H4 mechanism is untested.** A biased `b = 50` reference needs data
  with that structure; §2.7 of the plan already records that an extrapolated
  `S(50)` cannot carry a conclusion.
- **The goodness-of-fit ratio's absolute scale is not calibrated.** It uses the
  zero-fitted-degrees-of-freedom noise expectation; a MAP fit on a grid removes an
  ill-defined amount. The group comparison is the claim.
- **The blow-up definition is the thesis's 20 pL cutoff.** Every contrast is
  given across 5–90 pL, and the H4 and H3 readings change sign across that range,
  which is why both are stated per cutoff.
- **The geometric null is post-hoc.** It was added after the uniform-null result
  was read. It is the more conservative reference, it removed a claim rather than
  creating one, and the pre-registered uniform-null numbers stay in the report.
- **The debias is endpoint-only**, as in Phases 2 and 3, so every identifiable
  fraction here is a lower bound; at 1.57% the qualitative picture cannot turn on
  it.
- **The Phase-2/3 node set has no Fisher matrix at the `(rho, V)` grid boundary**
  (136 of 369 pairs). That is where 98% of the pathology is, so the Fisher side of
  Phase 4 is structurally unable to speak about the pathological voxels directly.
- **The fits needed about 10 GB of the host's 11 GB.** The Claude Code session's
  memory guard stopped the Bayes arms twice; they were completed outside the
  session with the same resumable runner. No result depends on that, but a
  reproduction should expect it.

---

## 5. Open decisions, for the user

1. **The `v_i` defect tolerance for the §4.5 report.** Still pending, as
   pre-registered. Its effect here is on overall coverage (0.06% → 34.5%), not on
   the pathology, which is excluded at every tolerance.
2. **Whether to test the mechanism where the Fisher matrix inverts.** A
   multi-`Delta` acquisition is the natural test of whether corner-pinning
   persists once `rho` and `V` are identifiable; sub-059 would replicate this one.
3. **Whether H2 should become a model change.** The library carries a free-water
   atom (`--include-free-water`), and the result points at an out-of-model
   compartment. That is a fitting-method decision, and Phase 5 territory.
4. **Whether to repair the Bayes `vi_map`** to a posterior mean of `rho V` rather
   than a product of means. A pipeline change with consequences for every existing
   Bayes output.
5. **Whether to test the thesis H4 mechanism by simulation**, since no repository
   dataset has a `b = 50`-without-`b = 0` structure.

---

## 6. Reproduction

*A fresh agent should follow this section rather than reconstruct the run.* The
fits take about 10 minutes each and need about 10 GB of RAM, so run them from a
normal terminal; the analysis takes about 13 seconds and the figures a few
seconds more.

```bash
conda activate mri
# 1. The eight fit arms, sequential and resumable (a completed arm is skipped).
bash scripts/run_fisher_phase4_fits.sh sub-125

# 2. The analysis.
DS=/mnt/c/miscellaneous/coding_projects/python/mri_processing/data_storage/data/Mayo_Glioma/derivatives/preproc/sub-125/dwi
RUNS=/home/jaden/madi_fisher_runs/full_domain
PYTHONPATH=. python -m scripts.run_fisher_phase4 \
    --fit-root data/outputs/fisher_phase4_sub-125 \
    --dwi $DS/sub-125_desc-madi-input_dwi.nii.gz --bval $DS/sub-125_desc-madi-input_dwi.bval \
    --bvec $DS/sub-125_desc-madi-input_dwi.bvec --mask $DS/sub-125_desc-brain_mask.nii.gz \
    --small-delta 20 --Delta 50 --artifact data/libraries/madi_dense_universal_remediated.npz \
    --phase1 $RUNS/phase1 --cache-dir $RUNS/cache --phase3-run-dir $RUNS/phase3 \
    --output-dir $RUNS/phase4

# 3. The figures.
PYTHONPATH=. python -m scripts.plot_fisher_phase4 --run-dir $RUNS/phase4 \
    --fit-root data/outputs/fisher_phase4_sub-125 --mask $DS/sub-125_desc-brain_mask.nii.gz \
    --output-dir docs/provenance/figures/fisher_phase4
```

A healthy analysis run prints all eight arms present, `sigma=172.09`, an
acquisition Fisher PD fraction of 0.0157, and writes `phase4_report.json`. It
**refuses** a restricted Phase-1 substrate, any arm whose recorded mask differs
from `--mask`, any arm not `completed`, and any loader state whose σ differs from
the baseline fit's record.

**Outputs.** `$RUNS/phase4/phase4_report.json` is the full record: every arm,
every cutoff, every hypothesis block, both nulls, the Fisher geometry under three
amplitude regimes, and the §4.5 coverage. `$RUNS/phase4/maps/` holds the voxel
arrays as NIfTI in the mask's geometry: both ADCs, blow-up masks, goodness-of-fit
ratios, node Fisher coverage, `kappa_V`, condition number, `log v_i` with its bound
and defect, the `rho`/`V` report gates, the per-voxel CRLBs and posterior SDs used
for the flag validation, and a displacement-angle map and its geometric-null map
for every perturbation. The figures and `table4_2_condition_arms_cutoff_sweep.csv`
are committed under [`provenance/figures/fisher_phase4/`](provenance/figures/fisher_phase4/).

**Implementation.**

| File | Role |
|---|---|
| `madi/fisher_crlb.py` | `packed_adjugate`, `directional_crlb`, `LOG_VI_CONTRAST`, `estimable_rho_V_contrast_bound`, `nearest_canonical_node`, `fit_trust_floor_masks` |
| `madi/library.py` | `candidate_selection_mask`, extracted from `_build_candidate_lib_matrix` so the trust floor and every matcher use one filter |
| `madi/volume_pathology.py` | ADC, reachable ceiling, band position, probability of superiority, stratified comparison, cutoff sweep, the kappa gate, flag validation, noise-floor residual, displacement and its geometric null |
| `scripts/fit_data.py` | `--trust-floor` and `--trust-floor-mode`, recorded in the run sidecar |
| `scripts/run_fisher_phase4_fits.sh` | the eight fit arms |
| `scripts/run_fisher_phase4.py` | orchestration; imports Phase 2's node table and per-column helper and the fitter's own loader rather than re-deriving them |
| `scripts/plot_fisher_phase4.py` | read-out only |

**Tests.** 26 added — 8 in `tests/physics_audit/test_fisher_crlb.py`, the new
`tests/physics_audit/test_volume_pathology.py` (12) and
`tests/physics_audit/test_fisher_phase4_runner.py` (6). They pin the contrast
bound to an explicit inverse and its unnormalized contrast; the estimable bound's
exactness on invertible matrices, its analytic value when the null is the
hyperbola, its defect when the null is `rho`, its `k_io`-scale invariance, and
its NaN when nothing is informed; the node join's log-space geometry; both
trust-floor forms and the extracted candidate filter; the per-column Fisher sums
against the audited primitives; the amplitude regime ordering; the mask-hash and
status guards; the ADC, effect size, ceiling, band position, noise-floor
calibration, displacement, kappa gate, flag validation and cutoff sweep; and that
a thin band alone produces small angles.

The full suite is **148 passed, 4 skipped, 1 xfailed**, plus the one
**pre-existing, unrelated** failure
`tests/physics_audit/test_gpu_golden.py::test_cpu_golden_hash_and_deterministic_reference_replay`
(`KeyError: 'realised_vi_spatial_se'`), recorded in
[`fisher_domain_audit.md`](fisher_domain_audit.md) §7b and not touched.

---

## 7. Amendment log

This record is updated in place.

### 2026-09-10-initial — first execution of Phase 4

- **Previously:** Phase 4 was unrun. Plan §7 specified H1–H4 and items 4.1–4.5
  without a dataset, a fit-time trust floor, a residual discriminator or a bound
  for `log v_i` where the Fisher matrix does not invert.
- **Now:** this record; the fit arms under `data/outputs/fisher_phase4_sub-125`;
  the analysis under `/home/jaden/madi_fisher_runs/full_domain/phase4`; the code
  and tests listed in §6; the `phase4` pre-registration block, declared before any
  fit result was read.
- **Not changed:** the substrate, the Phase-1 fields, the Phase-2 cache, every
  executed Phase-0/1, Phase-2 and Phase-3 number, and every existing fit output,
  including the two found defective.

### 2026-09-10-geometric-null — the ridge claim was withdrawn before it was written

- **Previously:** the pre-registered ridge criterion compared move angles with a
  45° uniform null. The first report showed medians of 2.0–3.6° and 54–76% of
  moves within 10°, which read as strong support for H1.
- **Now:** a permutation null that preserves the band geometry and the fitted
  distribution is reported beside it (seed 20260910), and the ridge conclusion is
  read against it: P(null > observed) = 0.43–0.60, so no ridge signature above
  geometry.
- **Why:** the band is about 23 times longer along the hyperbola than across it.
  Any move inside it makes a small angle, and the uniform null credited that to the
  degeneracy.
- **Transparency:** added after the first report was read and before any ridge
  claim was written; logged in the plan and the pre-registration too; the
  uniform-null numbers remain in `phase4_report.json`.

### 2026-09-10-estimable-bound-nan — no bound is reported as no bound

- **Previously:** `estimable_rho_V_contrast_bound` summed an empty informative set
  to 0 where no part of the `v_i` contrast was informed, and the first report
  carried a median "bound" of 0 on blow-up voxels whose defect was 1.
- **Now:** NaN wherever the defect is 1, pinned by
  `test_a_contrast_with_no_informative_projection_has_no_bound_not_a_zero_one`.
- **Effect:** none on coverage, which already required a defect at or below the
  tolerance; the `log_vi_bound` map and the blow-up bound summary were corrected.
