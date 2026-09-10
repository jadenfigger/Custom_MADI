# Documentation index

Read this file first.  It separates live specifications from immutable evidence
and from historical working material.  “Current” means the file is still an
active reference, not that every historical statement inside it is timeless.
Dates are the document dates where stated; otherwise they are not inferred.

**Analysis-domain convention (adopted 2026-09-06).** Fisher/CRLB work separates a
**reusable model-derived substrate** from **conditional acquisition analyses**.
A hardware, noise, averaging or trust assumption may annotate, stratify, report
and mask a declared analysis; it may never decide which stored library columns
are extracted or cached. The rule, its rationale and the audit that produced it
are in [`fisher_domain_audit.md`](fisher_domain_audit.md); the machine-checkable
form is `madi.fisher_crlb.ColumnDomain`; the pre-registered form is the
`analysis_domain_architecture` block.

**Amendment-log convention (adopted 2026-09-05).** A CURRENT document is never
edited by silently replacing its content. Each carries an **Amendment log**
section recording, per change: what the document previously said, what it says
now, and why. `madi/fisher_crlb_preregistration.json` carries the same record as
an `amendment_log` array, since JSON cannot hold comments. Retirement has two
distinct forms and they are not interchangeable: a record of something that
actually happened (a validation result, a measured gate, a completed run) is
**provenance** and moves to `provenance/` unchanged; a plan, assumption, or
default that a later decision superseded is a **stale plan** and moves to
`archive/` with a note naming what replaced it. Nothing is deleted.

| Document | Classification | Purpose | Status / superseder | Date |
|---|---|---|---|---|
| `deviations_from_paper.md` | CURRENT | Authoritative ledger of deliberate source-method departures and SI corrections. | Current. Its former claim that no remediated production artifact exists was corrected on 2026-09-05 (see its amendment log); the complete 369-group artifact is now named there. | 2026-09-05 |
| `fisher_crlb_analysis_plan.md` | CURRENT | Phased specification for Fisher/CRLB work. | Current specification. Amended 2026-09-09 at Phase-3 execution: the §6 analysis-domain choice was made by the user (both layers, contrasted; both conditional readings), and new item 3.2a requires the `k_io`-profiled `(log rho, log V)` block beside the pre-registered 3x3 spectrum. Amended 2026-09-06: new section 2.8 draws the reusable-substrate/conditional-analysis boundary, section 1.3 extracts over every stored column instead of the research feasibility mask, section 0.4's outputs are annotations, and Phases 2-6 each name the question they answer and the assumptions it requires (see `fisher_domain_audit.md`). Amended 2026-09-05 in two passes. First: S0 marginalization adopted (new section 2.7), Phase 2 reframed from three named protocols to a declared acquisition sweep, `t_epi` set to 30 ms, relative truncation-bias reporting required. Then at Phase-2 execution: `k = 3` / `k = 4` stencils dropped, the `k_io` restriction on the minimax withdrawn, the `n0_eff` sweep adopted, the Rician mask made averaging-aware, and the node aggregate changed from a mean to a median. See its amendment log; all twelve entries are dated. | 2026-09-05 |
| `fisher_phase2.md` | CURRENT | Executed Phase-2 record: debiased Fisher matrices, CRLB and `kappa` maps, the single-`Delta` versus multi-`Delta` sweep, and the `n0_eff` amplitude sweep. | Current record; every number stands as executed. Carries a 2026-09-06 carry-forward stating that all of it is conditional on a declared gradient scenario and none of it is a statement about intrinsic model identifiability over the stored acquisition domain. Carries the carry-forward corrections of its own section 1 (the `k = 3` / `k = 4` stencil reduction and the streaming-reimplementation audit) and the two user decisions that shaped the run: no `k_io` restriction in the minimax, and no choice between the clinical and research gradient scenarios. | 2026-09-05 |
| `fisher_phase3.md` | CURRENT | Executed Phase-3 record: the degeneracy map, the constant-`v_i` hyperbola test, the nine declared domains, and the structural figures. | Current record; every number stands as executed. The hyperbola hypothesis holds in the `(log rho, log V)` plane — median 2.95° from constant-`v_i`, stable at 2.6–5.1° across all nine domains — while the three-parameter sloppy eigenvector points mostly along `k_io` instead. Records the user's 2026-09-09 domain decision, the four conventions settled at execution, and a bit-exact reproduction of the four executed Phase-2 optima by an independent accumulation path. | 2026-09-09 |
| `fisher_domain_audit.md` | CURRENT | Audit of every domain restriction in Phases 0-6, the substrate/conditional boundary, and the 2026-09-06 remediation. | Current record. Classifies each restriction, names the one that was destructive (the Phase-1 `G_max` extraction domain), and states what was regenerated versus preserved. | 2026-09-06 |
| `fisher_phase01_framework.md` | CURRENT | Executed Phase 0/1 framework, validation results, and rerun instructions. | Current record, updated in place. Its rerun instructions were corrected on 2026-09-06 and its results carry a note that they were computed on a 24,111-column basis since re-materialised at the full 31,125. Reports the complete 369-group artifact, the shard-45 diff, a corrected `Var(J_hat)` normalization, and the `k_io` sensitivity result. Its `k = 3` / `k = 4` results are retained deliberately as the evidence for the Phase-2 stencil reduction, even though those stencils are no longer pre-registered. | 2026-09-05 |
| `marginal_s0_fisher_crlb_implementation_plan.md` | CURRENT | Nuisance-amplitude (`S0`) specification and the deferred marginal-S0 fitting handoff. | Un-archived 2026-09-05 when its nuisance-`S0` decision was approved. **Mixed status, per the table in its own header:** its Fisher half is LIVE and implemented in `madi/fisher_crlb.py`; its `--s0-mode` fitting half is STILL A PLAN; its v2 coordinate and `(Delta, b)` column material is SUPERSEDED. | 2026-09-05 |
| `fitting_methods.md` | CURRENT | Supported MAP, Bayesian, AMICO, and `--fit-s0` method reference. | Current. | — |
| `sol_package_guide.md` | CURRENT | General Sol environment and batch-job rules. | Current general guidance; its 64-shard MADI appendix is historical. | — |
| `tumorsynth_install.md` | CURRENT | TumorSynth installation and edema-workflow guide. | Current operational reference. | — |
| `universal_library.md` | CURRENT | Universal `(delta, Delta, b)` library design and limitations. | Current design reference; its seed wording and old identifiability status need correction in a future documentation pass. | — |
| `Literature_Parameter_Values.xlsx` | CURRENT | Curated literature parameter reference workbook. | Current supporting reference. | — |
| `papers/NMR in Biomedicine - 2022 - Springer - Metabolic activity diffusion imaging  MADI   I  Metabolic  cytometric modeling and.pdf` | CURRENT | Primary MADI I source paper. | Source reference. | 2022 |
| `papers/NMR in Biomedicine - 2022 - Springer - Metabolic activity diffusion imaging  MADI   II  Noninvasive  high‐resolution human (1).pdf` | CURRENT | Primary MADI II source paper. | Source reference. | 2022 |
| `papers/nbm4781-sup-0001-supporting information.docx` | CURRENT | MADI I supporting information source. | Source reference. | 2022 |
| `papers/nbm4782-sup-0001-supporting information.docx` | CURRENT | MADI II supporting information source. | Source reference. | 2022 |
| `provenance/finite_geometry_acceptance.md` | PROVENANCE | Incident, diagnosis, and restart scope for finite-geometry packing rejection. | Retained audit trail; production remediation supersedes it operationally. | — |
| `provenance/p0_pilot_validation.md` | PROVENANCE | Original restricted pilot validation. | Superseded for schema precision by the v5 pilot record. | 2026-08-03 |
| `provenance/p0_sol_execution.md` | PROVENANCE | P0 Sol execution record, including historic v4 commands. | Retained execution audit; use active Sol/Fisher guides for new work. | 2026-08-11 |
| `provenance/p0_v5_pilot_validation.md` | PROVENANCE | v5 schema/storage/provenance pilot result. | Retained as the accepted v5 pilot evidence. | — |
| `provenance/p0a_full_facet_validation.md` | PROVENANCE | Full-facet SI geometry certification. | Retained production-geometry evidence. | — |
| `provenance/p0b_si_geometry_reference_validation.md` | PROVENANCE | Five-million-cell SI geometry-reference validation. | Retained production-geometry evidence. | 2026-08-03 |
| `provenance/physics_fidelity_audit.md` | PROVENANCE | Pre-SI audit of the old universal artifact and implementation. | Superseded operationally by SI remediation and `deviations_from_paper.md`; retain unchanged. | 2026-08-02 |
| `provenance/reorg_plan.md` | PROVENANCE | Record of the earlier executed documentation/code reorganization. | The earlier reorganization executed; this index and tree complete the remaining docs separation. | — |
| `provenance/si_interim_update.md` | PROVENANCE | Interim SI remediation status and decisions. | Superseded as a live status record by later certification and production documents. | — |
| `provenance/v5_crn_diagnostic.md` | PROVENANCE | Pilot CRN correlation measurement and its limitations. | Superseded for production-step evidence by the stencil probe. | 2026-08-12 |
| `provenance/v5_fast_classifier_launch_readiness.md` | PROVENANCE | Exact-cache validation and production launch-readiness result. | Retained launch evidence; production is now in post-build completion. | — |
| `provenance/v5_stencil_probe.md` | PROVENANCE | Restricted production-grid derivative-noise/CRN probe. | Retained source of measured stencil-noise calibration. | — |
| `provenance/figures/fisher_phase3/` | PROVENANCE | Phase-3 structural figures and the per-`(rho, V)` median table. | Belongs to `fisher_phase3.md`. Retained in the repository because the 8.6 GiB substrate they derive from is not, so they are the surviving record of that run. | 2026-09-09 |
| `provenance/figures/v5_crn_diagnostic/` | PROVENANCE | Figures, CSVs, and JSON supporting the CRN diagnostic. | Belongs to `v5_crn_diagnostic.md`. | 2026-08-12 |
| `provenance/figures/v5_stencil_probe/` | PROVENANCE | Figures, CSVs, and JSON supporting the stencil probe. | Belongs to `v5_stencil_probe.md`. | — |
| `archive/identifiability.md` | STALE PLAN | Pre-v5 Fisher/identifiability user guide. | Superseded by v5 framework/code; retained for its parameterization rationale. | — |
| `archive/joint_bayesian_fitting_plan.md` | STALE PLAN | Deferred `bayes-joint` fitter proposal. | Superseded as an immediate implementation plan by the marginal-S0 handoff; not implemented. | — |
| `archive/madi_checklist.txt` | STALE PLAN | Historical project checklist. | Many items were closed or invalidated by SI/v5 work; see framework reconciliation. | — |
| `archive/p0_full_rebuild_plan.md` | STALE PLAN | Pre-submission 369-shard production plan. | Superseded by the executed v5 production build and its records. | — |
| `archive/v5_pilot_runbook.md` | STALE PLAN | Pre-run v5 pilot procedure. | Superseded by the completed v5 pilot validation. | — |
| `archive/v5_schema_prebuild_note.md` | STALE PLAN | Pre-build v5 schema/storage specification. | Superseded by the built schema and v5 pilot validation. | — |
| `archive/v5_stencil_probe-instructions.md` | STALE PLAN | Pre-launch stencil-probe instructions. | Superseded by the completed `v5_stencil_probe.md` record. | — |
| `archive/shard_viewer.ipynb` | SCRATCH | Interactive shard-inspection notebook. | Development aid; no independent validation record. | — |
| `archive/terminal_command_history.txt` | SCRATCH | Personal command log. | Historical scratch only. | — |

No document is currently classified UNCLEAR.  The four primary papers and the
spreadsheet are reference sources, so they remain at the top level rather than
being treated as project-work provenance.

`madi/fisher_crlb_preregistration.json` is not a document but is governed by the
same rules: it is the authoritative pre-registration for the Fisher/CRLB work,
it carries its own `amendment_log`, and superseded entries are demoted within it
(as `reference_protocols`) rather than removed.

**Three artifacts outside `docs/` are classified here** because the Phase-2 audit of
duplicated `madi/` arithmetic and the 2026-09-06 domain audit reached them and
nothing else indexes them.

`analysis/phase2_single_delta_explorer.ipynb` is **CURRENT**. It is an interactive
read-out of the Phase-2 machinery at a single parameter coordinate under a
single-`Delta` (m = 1) acquisition: raw derivative vector `J`, `Var(J_hat)`,
`beta`, `r_trunc`, the debiased and undebiased Fisher matrices, CRLB and `kappa`.
It **imports** `madi.fisher_crlb` and `scripts.run_fisher_phase2` rather than
reimplementing them, and its section 6 asserts cell-by-cell that it reproduces
the identifiable fraction, relative CRLB, `kappa` and stored maps already written
into `phase2_report.json`. If that assertion ever fails, `phase2_report.json` is
the authority. It reads data only; it writes nothing. See `fisher_phase2.md`.

`analysis/phase2_barebones.ipynb` is **CURRENT**. It is a deliberately minimal
single-node, single-`Delta` read-out in which every acquisition limit (`G_MAX`,
`TRUST_FLOOR`, `RICIAN_MIN`, `AVERAGES`) defaults to **off**, so the model-derived
Fisher quantities can be inspected with no downstream condition applied. It is
classified here because it is the artifact at which the premature-domain problem
surfaced: before 2026-09-06 setting `G_MAX = inf` could not recover the
gradient-infeasible columns, because the Phase-1 substrate did not contain them.
It now reads the full-domain run directories. Like the explorer it imports
`madi.fisher_crlb` and `scripts.run_fisher_phase2` and writes nothing. See
`fisher_domain_audit.md`.

`analysis/prod_inter_fisher_analysis.ipynb` is **SCRATCH**. It independently
reimplements the gradient formula, the Fisher matrix, `kappa`, the canonical-grid
index math, and the derivative variance (with `n_ensembles` as a literal `40.0`),
and it predates `madi/fisher_crlb.py`. Nothing imports it. It is retained
unedited apart from a header cell added on 2026-09-05 that names the
authoritative implementation of each duplicated piece and warns that its `beta`,
`Var(J)`, and CRLB figures must not be compared with Phase-1 or Phase-2 output.
See `fisher_phase2.md`, "Audit of duplicated `madi/` arithmetic".
