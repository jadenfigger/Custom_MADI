# Fisher/CRLB analysis-domain audit

Date: 2026-09-06. Classification: CURRENT (see [`INDEX.md`](INDEX.md)).

## Verdict

**One material defect, found in Phase 1, with a clean root cause and no
contaminated numbers.** Phase 1 took its extraction domain from Phase 0.4's
`G_max` feasibility report, so the reusable derivative substrate held **24,111 of
the library's 31,125 stored `(delta, Delta, b)` columns**. The 7,014 absent
columns are exactly the ones requiring more than 300 mT/m — the short-`delta`,
high-`b` corner of the stored grid. Because the substrate was built that way, no
later analysis at any other gradient setting could reach them, and relaxing a
notebook-level `G_max` could not recover them.

**Nothing that was computed is numerically wrong.** Every Phase-0, Phase-1 and
Phase-2 number was computed correctly on the columns it actually used, and every
Phase-2 result was already conditioned on a declared gradient scenario whose
columns the restricted substrate happened to contain in full. The damage is to
what could be *asked* later, not to what was answered.

Three things follow, and they are separated deliberately throughout this
document:

1. the **architecture** is corrected — the substrate is now the whole stored
   grid and hardware masks are applied at evaluation;
2. the **executed results** stand as executed, and the Phase-2 conditional
   analyses reproduce exactly on the corrected substrate;
3. the **interpretation** of one class of statement changes: no Phase-2 result
   may be read as a statement about intrinsic model identifiability over the
   stored acquisition domain, because 22.5% of that domain was never in the
   substrate.

---

## 1. The boundary this audit is about

The Fisher work has two layers, and the defect was a leak from the second into
the first.

| Layer | Contents | Depends on | Domain |
|---|---|---|---|
| **Reusable substrate** | stored signal `S`; central-difference Jacobians `J`; `Var(J_hat)` and its CRN covariance; Richardson truncation-bias fields | the validated library only | every stored `(delta, Delta, b)` column |
| **Conditional analysis** | gradient feasibility at a chosen `G_max`; TE/T2 noise; Rician validity at a chosen averaging; the `S/S0` trust floor; budget `N`; protocol optimisation; estimator efficiency | a *declared* scanner and acquisition | whatever that declaration admits |

The governing rule, now pre-registered as `analysis_domain_architecture`:

> Preserve and analyse the full domain supplied by the validated forward
> model/library unless an exclusion is inherent to the validated model,
> mathematically required for the requested analysis, or explicitly approved as
> part of the scientific question. Downstream feasibility quantities may be
> calculated and retained as metadata without censoring the reusable upstream
> analysis substrate.

Its machine-checkable form is `madi.fisher_crlb.ColumnDomain`: every cache and
every report declares the domain it represents, and `require_columns` refuses to
let a restricted cache be read as universal.

---

## 2. Audit table

Every place in Phases 0–6 where a condition reduces the analysis domain.
Classification key: **1** forward-model/domain constraint, **2** mathematical or
statistical requirement of a specific analysis, **3** numerical/computational
implementation decision, **4** downstream acquisition/hardware/engineering
assumption, **5** user-approved scientific restriction.

| # | Restriction | Implementation | Specification | Purpose | Affects | Artifacts depending on it | Class | Disposition | Recompute? |
|---|---|---|---|---|---|---|---|---|---|
| R1 | `(rho, V)` mask band `0.40 <= rho*V*1e-6 <= 0.99` | `madi/library.py::make_remediation_log_grid`; `madi/fisher_crlb.py::canonical_grid` | `deviations_from_paper.md`, "Dense masked log-coordinate grid" | the library contains no entry outside the band; it is what was built | everything | all | **1** | **Keep.** Defines the available domain. | no |
| R2 | Interior nodes only: a centre needs `rho±k`, `V±k`, `k_io±k` | `run_fisher_phase1.py::_centres`; `run_fisher_phase2.py::build_node_table` | plan §1.3 | a central difference has no meaning without both endpoints | extraction and evaluation | all Phase-1/2 fields | **1/3** | **Keep,** already reported (136 of 369 `(rho, V)` pairs, both mask-band edges). | no |
| R3 | `ensemble_means_subset` exists for 200 columns only | v5 build schema | `archive/v5_schema_prebuild_note.md`; prereg `monte_carlo_debias_coverage` | the exact CRN covariance was only ever stored there | evaluation (exact vs endpoint-only debias) | Phase-1 `VarJ_*_diagnostic`; the Phase-2 debias calibration | **1** | **Keep.** A library-storage fact, already declared and calibrated. | no |
| R4 | Stencil half-widths `k = 1, 2`; `k = 3, 4` dropped | prereg `stencil_half_widths` | plan amendment `2026-09-05-drop-k3-k4` | measured: truncation bias already dominates noise 17x/55x at `k = 1` | extraction | the retained `k = 3/4` fields are the evidence | **3/5** | **Keep.** Evidence-based and evidence-retained. | no |
| R5 | Fisher matrices built from `k = 1` only | prereg `stencil_rule` | same | `k = 2` exists solely to feed the Richardson estimate | evaluation | Phase 2 | **3** | **Keep.** | no |
| **R6** | **Phase-1 derivative extraction restricted to the research (300 mT/m) feasibility mask ∪ 200 diagnostic columns** | `run_fisher_phase1.py` (`--feasibility` was required; `selected_columns` came from `derivative_column_selection`); `analyze_fisher_feasibility.py` emitted that key | **plan §1.3 only.** *Not* in the pre-registration, and **no amendment-log entry anywhere records it as a decision** | stated as "avoiding fields for globally unreachable columns" — unreachable *on a 300 mT/m scanner*, not in the model | **extraction and caching** | Phase-1 fields; the Phase-2 cache; `phase2_report.json`; the `S0` gap report; both explorer notebooks | **4, misapplied** | **REMOVED.** The substrate is the full stored grid; the mask survives as a conditional evaluation mask and as an opt-in reproduction flag. | **yes — Phase 1 and the Phase-2 cache** |
| R7 | Trust floor `S/S0 >= 0.015`, hard per-`(entry, column)` exclusion | `madi/fisher_crlb.py::feasibility_masks`; `run_fisher_phase2.py::pair_contributions` | plan §2.6, §12; prereg `trust_floor` | library values that low are below the measurement's usable range | evaluation; **and it fed R6's `combined_any`** | Phase-2 Fisher sums; Phase-0.4 counts | **4** (a measurement-trust assumption) | **Keep at evaluation**, where it is declared per result. Removed from anything that selects the substrate. Phase 4 H3 tests it explicitly, which is the right use. | no |
| R8 | Rician validity `S/sigma_c >= 3` | same two sites | plan §2.6; prereg `rician_magnitude_snr_min`; amendment `2026-09-05-phase2-averaging-aware-mask` | the Gaussian Fisher formula is invalid below magnitude SNR ≈ 3 | evaluation; **and it fed R6's `combined_any`** | as R7 | **2 conditioned on 4** — a real mathematical validity condition, but only relative to an assumed SNR, `T2`, `t_epi` and averaging | **Keep at evaluation.** Removed from substrate selection. | no |
| R9 | `t_epi = 30 ms`, `T2 = 80 ms`, SNR 50 | prereg `noise_model`; `analyze_fisher_feasibility.py` | plan §2.6; amendment `2026-09-05-t-epi` | prices the readout so a protocol optimiser is not biased to long `Delta` | evaluation; **and it fed R6 through R8** | Phase-0.4 counts; Phase 2 | **4** | **Keep as declared acquisition parameters.** | no |
| R10 | `averages_per_column = 1` in the Phase-1 screen | prereg `noise_model` | amendment `2026-09-05-phase2-averaging-aware-mask` | "which columns are *ever* usable" | evaluation | Phase-0.4 counts | **4** | **Keep as an annotation.** Correctly already excluded from Phase-2 scoring. | no |
| R11 | Phase-2 gradient scenarios, clinical 80 / research 300 mT/m, reported separately and never pooled | `run_fisher_phase2.py`, per-scenario loop | prereg `gradient_limits_T_per_m`, `feasibility_basis.gradient_scenario_choice` (**DEFERRED by user decision**) | a protocol recommendation must be playable | evaluation only | `phase2_report.json` | **4/5** — declared, dual-reported, user-deferred | **Keep exactly as is.** This is the model of how a hardware assumption *should* enter. | no |
| R12 | `b = 0` excluded from every diffusion-weighted subset | `run_fisher_phase2.py` | plan §5; prereg | `S(0) = 1` exactly for every entry, so `J ≡ 0`; it carries no tissue information and enters only as the amplitude reference | evaluation | Phase 2 | **2** | **Keep.** Mathematically forced, not an assumption. | no |
| R13 | `k_io` relative-CRLB denominator floored at 5 s⁻¹ | `run_fisher_phase2.py` | prereg `relative_scale_for_k_io` | an unfloored ratio is dominated by the smallest-`k_io` node as an artifact of division | evaluation, scaling only — **no node is removed** | Phase 2 | **2/3** | **Keep.** | no |
| R14 | Node aggregate = median with `+inf` at unidentified nodes | `run_fisher_phase2.py::_score_block` | amendment `2026-09-05-phase2-node-aggregation` | the pre-registered mean is `+inf` for every protocol and ranks nothing | evaluation | Phase 2 | **3** | **Keep.** | no |
| R15 | `k_io <= 30` evaluation cap | — | amendment `2026-09-05-withdraw-kio-restriction` | — | — | — | **withdrawn** | Already withdrawn by user decision and measured not to matter. Precedent for this audit. | n/a |
| R16 | b-subset sizes `{8, 12, 16}`; `m <= 4`; `m = 3, 4` greedy | `run_fisher_phase2.py` | prereg `b_subset_rule`, `search_strategy`; amendment `2026-09-05-phase2-specification` | `C(24,8) x 1245` is not enumerable; `C(1245,3)` is not enumerable | evaluation, search space | Phase 2 | **3**, with a **4** flavour in the `m <= 4` rationale | **Keep, already flagged.** Size 8 wins monotonically so the optimum may lie below 8 (`fisher_phase2.md` §8 item 2); the `m <= 4` justification cites the Rician mask and should be read as conditional. | no |
| R17 | `N = 128` declared, not swept | `run_fisher_phase2.py` | prereg `feasibility_basis.noise_model` | a matched-budget comparison needs a budget | evaluation | Phase 2 | **4** | **Keep, already flagged** as an open item; the ranking is not `N`-independent. | no |
| R18 | `report_s0_marginal_crlb.py` silently dropped requested columns absent from the Phase-1 selection | `report_s0_marginal_crlb.py` | — | none; an unguarded convenience | evaluation | `s0_marginal_crlb.json` | **defect** | **FIXED.** Missing columns now raise; the column basis is declared, and `--gradient-scenario` makes a hardware condition explicit. | **regenerate the report** |
| R19 | `run_fisher_phase2.py` silently intersected its scenario mask with the Phase-1 selection (`position_of[full] >= 0`) | `run_fisher_phase2.py` | — | none; an unguarded convenience | evaluation | `phase2_report.json` | **defect (latent)** | **FIXED.** The scenario now asserts the substrate covers every column it admits. Inert for both declared scenarios, since clinical ⊂ research ⊂ old substrate. | no (results unchanged) |
| R20 | Rectangular-lobe (no ramp) approximation at very small `delta` | `madi/walker_gpu.py`, `madi/signal.py` | `universal_library.md` §9 | ramps of 0.1–0.3 ms are 10–30% of `delta = 1 ms`, so those columns may be biased | **nothing** — it is a stated caveat, not a filter | — | **1, as an annotation** | **Keep as an annotation, never as a censor.** It is a fidelity caveat on a stored column, and this audit's rule is explicit that it must not remove the column from the substrate. | no |
| R21 | Phase 3–6 questions not separated by the assumptions they require | `fisher_crlb_analysis_plan.md` §§6–9 | plan | — | future work | none yet | **spec gap** | **FIXED** in the plan: each remaining phase now names the question it answers and the assumptions it requires. | n/a |

---

## 3. Root cause

The mask that became destructive is the same mask that is entirely legitimate one
layer down. Three steps turned an annotation into a selection:

1. **Plan §1.3 wrote the mask into the extraction step.** Its wording —
   *"compute central differences only for columns surviving the declared research
   feasibility mask at at least one cellular node, plus all 200 diagnostic
   columns. This preserves every physically usable measurement while avoiding
   fields for globally unreachable columns"* — reads as a storage economy. It is
   not: "globally unreachable" means unreachable *by a 300 mT/m scanner*. The
   library's forward model is defined at those timings and stores a signal for
   them; only a scanner cannot play them.
2. **Phase 0.4 emitted an extraction directive.** The feasibility report's key was
   literally named `derivative_column_selection`, so the conditional mask arrived
   downstream already shaped as an instruction about what to extract.
3. **Nothing downstream could tell.** The Phase-1 manifest recorded a column
   count but declared no *basis*, so Phase 2, the `S0` report and both notebooks
   treated "absent from the cache" and "not applicable" as the same thing, with
   a silent `position_of >= 0` test.

The restriction was never pre-registered and never appears in any amendment log,
so it entered as an implementation decision by an earlier coding agent, not as an
approved scientific choice. Contrast R11 and R15: the *same* `G_max` quantity,
used at evaluation, was declared, dual-reported, deliberately left unchosen by
the user, and — in R15's case — withdrawn on request and then measured. That is
the pattern; R6 was the departure from it.

The savings the restriction bought were also small. At full width the Phase-1
fields grow 7.15 → 8.61 GiB and each Phase-2 cache member 1.69 → 2.18 GiB. That
is not a computational limitation; it is a rounding error against a 15.2 GiB
library.

---

## 4. What was actually lost, and where it sits

The 7,014 absent columns are not scattered. They are the **short-pulse,
high-`b`** corner:

| `delta` (ms) | timing pairs | median research-feasible DW columns (of 24) |
|---:|---:|---:|
| 1 | 56 | **0** |
| 2 | 55 | 1 |
| 3 | 54 | 3 |
| 4 | 53 | 5 |
| 5 | 52 | 9 |
| 6 | 51 | 13 |
| 7 | 50 | 18 |
| ≥ 8 | 924 | 24 |

- **84 of the 1,245 stored timing pairs had no research-feasible
  diffusion-weighted column at all.** Only their `b = 0` column survived into
  the derivative field, and `J` is identically zero there, so those timings
  carried no usable derivative information whatsoever.
- 252 pairs could not supply even 8 diffusion-weighted columns, which is why the
  Phase-2 research sweep at subset size 8 enumerated 993 pairs rather than 1,245.
- The worked symptom: at `(delta = 6, Delta = 20) ms` the stored grid holds 24
  diffusion-weighted columns; the substrate held 8. `b = 4500` upward required
  0.31–0.51 T/m and was gone.

This matters scientifically, and the direction is worth stating plainly: short
`delta` at high `b` is the narrow-pulse, high-`q` regime in which restriction —
and therefore cell volume `V` — is most strongly encoded. Phase 2 concluded that
single-`Delta` identifiability fails on `log rho` and `log V`. That conclusion is
correct *for the acquisitions it evaluated*, all of which were gradient-limited
by construction, but it could not have been a statement about the model, because
the region where `V` is best determined was not in the substrate. The corrected
substrate makes that question askable; **this audit does not answer it, and no
new scientific claim is made here.**

The `universal_library.md` §9 ramp-time caveat applies to part of the recovered
region and is carried forward as an annotation: at `delta = 1–3 ms` a real
gradient ramp is a sizeable fraction of the pulse, so the rectangular-lobe
forward model is least accurate there. That is a reason to *label* those
columns, and it is explicitly not a reason to delete them from the substrate.

---

## 5. Phase-by-phase disposition

### Phase 0 — valid, one output reclassified

| Item | Status |
|---|---|
| 0.1 shard verification | **Valid unchanged.** Reads the artifact, not a mask. |
| 0.2 `identifiability.py` migration | **Valid unchanged.** Carries no feasibility mask at all. |
| 0.3 free-water Gates A and B | **Valid unchanged.** Gate A is synthetic; Gate B reads the free-water row. |
| 0.4 feasibility masks and TE noise model | **Calculations valid; one output was at the wrong architectural layer.** Every count (clinical 9,999 / research 24,081 gradient columns, the per-entry survivor distributions at `t_epi` 0 and 30 ms) is correct and is retained. Only `derivative_column_selection` was misused. |
| 0.5 pre-registration | **Amended,** not invalidated: a new `analysis_domain_architecture` block. |

**No Phase-0 calculation needs rerunning.** The feasibility report was
regenerated only so it carries the new schema and the per-scenario index lists;
its numbers are identical to the recorded `column_feasibility_tepi30.json`.

### Phase 1 — numerically valid, domain re-materialised

- The existing derivative fields and the corrected `Var(J_hat)` are **numerically
  valid on the columns that were materialised.** The `2026-09-05-varj-normalization`
  fix, the `k_io` step-size finding, the `k_io > 30` sensitivity collapse, the
  truncation-versus-noise table and the shard-45 diff all stand.
- The omitted columns are exactly `G > 300 mT/m` minus the 30 gradient-infeasible
  diagnostic columns that the union re-admitted: 7,044 − 30 = **7,014**.
- Remediation is a **re-materialisation from the existing complete universal
  library**. No Monte-Carlo rebuild, no new simulation, no new seed. The library
  already stores every column; Phase 1 simply reads all of them now.
- The `k = 3` and `k = 4` fields are **not** regenerated (dropped 2026-09-05) and
  the originals are retained in place as the evidence for that decision.

### Phase 2 — conditional results stand; universal readings do not

- Every reported Phase-2 number was computed under a declared gradient scenario
  whose admissible columns the old substrate contained **in full**
  (clinical ⊂ research = old substrate minus the 30 diagnostic extras, none of
  which is admissible at either ceiling). So the executed results are exactly
  reproducible on the corrected substrate and are **not** re-labelled as wrong.
- What changes is the **scope of the claim**. "No single-`Delta` acquisition is
  jointly identifiable" is established for acquisitions playable at ≤ 300 mT/m.
  It is not established for the stored acquisition domain, and the manuscript must
  not read it that way.
- The `k_io > 30` region split, the `n0_eff` sweep, the debias-decides-the-verdict
  table, the greedy-gap measurement and the `kappa` maps are all conditional on the
  same scenarios and carry the same qualification.
- Masks that are appropriate to a CRLB/noise/protocol question — R7, R8, R11 —
  **remain available and remain applied** in that role. Nothing conditional was
  deleted.

### Phases 3–6 — question/assumption separation made explicit

See `fisher_crlb_analysis_plan.md` §§6–9 as amended. In short: Phase 3 is
model-conditioned on a noise model and must say so; Phase 4 H3/H4 use the trust
floor and the amplitude prior as *experimental conditions*, which is the correct
use; Phase 5.1 is acquisition optimisation and is where hardware belongs; Phase
5.2 must divide by the bound matching the fitter. The load-bearing consequence of
this audit for them is that **the substrate no longer has to be regenerated to
change a downstream engineering choice.**

---

## 6. Code changes

| File | Change |
|---|---|
| `madi/fisher_crlb.py` | New `ColumnDomain`, `stored_column_domain`, `read_column_domain`, `require_columns`, `gradient_feasible_columns`, `STORED_COLUMN_DOMAIN`, `LEGACY_COLUMN_DOMAIN`. `feasibility_masks` gains a docstring stating that it is conditional and is not an extraction filter. |
| `scripts/run_fisher_phase1.py` | `--feasibility` is now optional and annotation-only. Default domain is every stored column. `--restrict-columns-to-feasibility` reproduces the historical basis on purpose and stamps the manifest. Manifest schema `v3` carries `column_domain`. |
| `scripts/analyze_fisher_feasibility.py` | Schema `v2`. Adds per-scenario `gradient_feasible_column_indices` and `combined_any_entry_column_indices` so a conditional analysis is reconstructible; marks `derivative_column_selection` DEPRECATED with the reason. |
| `scripts/build_fisher_phase2_cache.py` | Inherits and records the Phase-1 domain; schema `v2`. |
| `scripts/run_fisher_phase2.py` | Reads the declared domain, `require_columns` on every scenario's admissible set, records the domain in the report. The silent `position_of >= 0` intersection is gone. |
| `scripts/report_s0_marginal_crlb.py` | Missing columns raise instead of being dropped; declares its column basis; new `--gradient-scenario` for an explicit hardware condition; reports `max_gradient_T_per_m_used`. |
| `scripts/summarize_fisher_phase2.py` | Prints the report's column-domain banner, and says so explicitly when a report predates the contract. |
| `tests/physics_audit/test_fisher_crlb.py` | Six new tests, listed in §8. |
| `analysis/phase2_barebones.ipynb` | Repointed at the full-domain run; **asserts** its substrate is complete rather than trusting it, and takes its column mapping from `ColumnDomain`. This is the notebook where turning `G_MAX` off is the point. |
| `analysis/phase2_single_delta_explorer.ipynb` | Repointed at the full-domain run; declares the domain on load and states that `G_MAX` is now the only gradient filter in play. |

---

## 7. Data and artifact changes

**Nothing was deleted or overwritten.**

| Artifact | Old | New | Status |
|---|---|---|---|
| Universal library | `data/libraries/madi_dense_universal_remediated.npz` | unchanged | **not rebuilt** |
| Phase-0.4 report | `final_varfix/column_feasibility_tepi{0,30}.json` | `full_domain/column_feasibility.json` | old retained as provenance; counts identical |
| Phase-1 fields | `final_varfix/phase1/` — 24,111 columns, includes `k = 3/4` | `full_domain/phase1/` — 31,125 columns, `k = 1, 2` | old retained; it is the `k = 3/4` evidence and the overlap reference |
| Phase-2 cache | `phase2/cache/` — 24,111 columns | `full_domain/cache/` — 31,125 columns | old retained |
| `S0` gap report | `final_varfix/s0_marginal_crlb.json` | `full_domain/s0_marginal_crlb.json`, with a declared basis | old retained as the executed record; every ratio identical |
| Phase-2 sweep | `phase2/N128/` | `full_domain/phase2_N128/` — the reproduction run | old retained as the executed record; 2,911 quantities and all 32 maps identical |

Acquisition-domain coverage: **24,111 → 31,125 stored columns (77.5% → 100%)**;
timing pairs with at least one diffusion-weighted column **1,161 → 1,245**.

---

## 7b. Verification

Every claim in §3 and §5 that "nothing changes" is measured, not argued.

**Phase-1 overlap — bit-for-bit, on all 24,111 columns the old substrate held.**
Not a tolerance: zero differing bits in 1,781,200,125 float32 cells.

| Field | rows | cells | differing bits |
|---|---:|---:|---:|
| `J_rho_k1` | 13,821 | 333,238,131 | **0** |
| `J_rho_k2` | 9,078 | 218,879,658 | **0** |
| `J_V_k1` | 12,291 | 296,348,301 | **0** |
| `J_V_k2` | 5,763 | 138,951,693 | **0** |
| `J_k_io_k1` | 18,081 | 435,950,991 | **0** |
| `Richardson_rho_k1_k2` | 9,078 | 218,879,658 | **0** |
| `Richardson_V_k1_k2` | 5,763 | 138,951,693 | **0** |
| `VarJ_*_diagnostic` (all five) | — | 200 columns each | **0** |

Sample-index arrays are identical for every field, and the diagnostic column
indices are identical.

**Phase-2 caches — bit-for-bit** on all 24,081 research-feasible columns:
`vectors` and `signal_variance` each 453,204,420 cells, **0** differing bits.

**Phase-2 search space — identical.** For both declared scenarios, every one of
the 1,245 timing pairs has the identical candidate column set: 0 pairs differ,
449 clinical and 993 research pairs carry at least 8 diffusion-weighted columns,
and the totals are unchanged at 8,754 and 22,836 admissible columns.

**Phase-2 sweep — re-run in full, and identical.** Identical inputs over an
identical search space were confirmed rather than inferred: the whole N = 128
sweep was executed against the corrected substrate
(`/home/jaden/madi_fisher_runs/full_domain/phase2_N128/`) and diffed against the
executed record. **2,911 reported quantities match exactly, zero mismatches** —
every arm at every subset size in both scenarios, and for each: the best timing
pairs, the chosen `b` values, the column count, the arm counts, the finite-minimax
counts, the greedy optimality gap, and, under all six `S0` regimes, the minimax
score, the argmax parameter, the identifiable fraction, the worst node, A- and
D-optimality, the `(log rho, log V)`-only diagnostic, the per-parameter relative
CRLBs, `kappa` means and maxima, the marginal/fixed CRLB ratios, and the four
debias-effect quantities. The reference-protocol rows match, including which
cells are unavailable. All **32 per-node `maps_*.npy` files are identical**, as
are `evaluation_nodes.npy` and `evaluation_node_labels.npy`. All six exhaustive
`m = 2` optima reproduce to every printed digit: clinical 1.0808 / 1.5527 /
2.5695 and research 0.48321 / 0.6504 / 0.76569, at identical arm counts.

**The reported symptom, before and after.** At `(delta = 6, Delta = 20) ms`, the
stored grid holds 24 diffusion-weighted columns:

| substrate, condition | diffusion-weighted columns | `b` range | max `G` |
|---|---:|---|---:|
| old, `G_MAX = inf` | **8** | 500–4000 | 0.294 T/m |
| new, `G_MAX = inf` (model, no condition) | **24** | 500–12000 | 0.509 T/m |
| new, `G_MAX = 0.30` (research) | 8 | 500–4000 | 0.294 T/m |
| new, `G_MAX = 0.08` (clinical) | 0 | — | — |

The first row is the defect: removing the notebook's own gradient filter changed
nothing, because the substrate had already applied one. The third row shows the
conditional analysis reproducing the old behaviour exactly when it is asked for.

**Reported results that reproduce exactly.** The `S0` fixed-versus-marginalized
CRLB gap table, regenerated on the corrected substrate, matches
`fisher_phase01_framework.md` to every reported digit at all three acquisitions
and all three amplitude regimes. The relative truncation-bias table likewise
(`rho` `k = 1` 0.013608, `k = 2` 0.054432; `V` `k = 1` 0.018176, `k = 2`
0.072704).

**The one statistic that moves, and why it is not a correction.** The
*absolute* truncation-bias RMS: `rho` 0.015650 → 0.014583, `V` 0.015398 →
0.014311. It is pooled over every selected column, so a wider basis changes its
denominator and its population; the recovered columns sit at high `b` where
`|J|` and its absolute bias are small. The absolute maxima are unchanged
(0.248967, 0.193433). This is precisely why plan §1.4 asks for the *relative*
bias — that is the form which means the same thing on both bases — and the
executed value stays in the framework's table as executed.

**Tests.** The Fisher suite is 24 tests, all passing. The full suite is
112 passed, 4 skipped, 1 xfailed, plus one **pre-existing, unrelated** failure:
`tests/physics_audit/test_gpu_golden.py`, whose stored `cpu_gpu_golden_v1.npz`
fixture lacks the `realised_vi_spatial_se` geometry field
(`KeyError: 'realised_vi_spatial_se'`). It is a geometry-stats schema drift,
already recorded as pre-existing in `fisher_phase2.md` §1.2, and it was not
touched.

---

## 8. Tests added

In `tests/physics_audit/test_fisher_crlb.py` (the suite is now 24 tests):

1. `test_a_hardware_profile_cannot_decide_what_the_substrate_contains` — the
   architectural invariant, end to end on a synthetic v5 artifact: two
   feasibility reports standing for two different scanner ceilings produce the
   *same* Phase-1 column domain, and it is the complete stored grid.
2. `test_a_restricted_substrate_is_opt_in_stamped_and_refused_downstream` — the
   historical basis stays reproducible, is stamped RESTRICTED, and
   `require_columns` refuses to serve a column it lacks.
3. `test_a_legacy_manifest_is_not_assumed_to_be_universal` — a manifest written
   before the contract declares no basis and is not guessed to be complete.
4. `test_the_conditional_gradient_mask_reconstructs_from_the_full_substrate` —
   the clinical and research masks are pure subset operations on the substrate
   and reproduce 9,999 / 24,081 / 24,111 / 7,014 exactly.
5. `test_the_domain_mapping_preserves_canonical_column_identity` — positions
   round-trip and each keeps its `(delta, Delta, b)`.
6. `test_widening_the_column_domain_leaves_the_overlap_bit_identical` — the
   unit-level form of the remediation's acceptance check: a full-width Phase-1
   run reproduces a restricted one bit-for-bit on the overlapping columns, so a
   future change to the streaming path cannot quietly perturb columns it was not
   supposed to touch.

No existing physics or statistical regression test was weakened.

---

## 9. Amendment log

### 2026-09-06-initial — the audit

- **Previously:** no document separated the reusable Fisher substrate from the
  conditional acquisition analyses layered on it, and plan §1.3 specified an
  extraction domain set by a 300 mT/m scanner ceiling.
- **Now:** this document, the pre-registration block
  `analysis_domain_architecture`, and the code contract in
  `madi.fisher_crlb.ColumnDomain`.
- **Why:** the ceiling had removed 7,014 of 31,125 stored columns from the
  reusable substrate, so a later analysis at any other gradient setting was
  impossible without regenerating it, and a notebook-level filter could not
  recover them.
- **Not changed:** every executed Phase-0, Phase-1 and Phase-2 number. They are
  correct on the columns they used, and the Phase-2 conditional analyses
  reproduce on the corrected substrate.
