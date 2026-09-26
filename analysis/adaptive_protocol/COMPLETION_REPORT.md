# Completed adaptive-TE experiment

All scientific execution ran in **Windows Python**. Existing WSL files were
read through Windows UNC paths. The new workflow and outputs are isolated in
`analysis/adaptive_protocol/`; original notebooks, reports, simulator, library,
and caches were not edited.

## Delivered files

- `core.py`, `search.py`, `run.py`: cached-data evaluator, discrete optimizer,
  repeat allocation, baseline comparisons, controlled protocol sweeps.
- `export.py`, `export_workbook.mjs`, `figures.py`: extended protocol-statistics
  tables, workbook, and plots generated from numerical products.
- `config.json`, `run_windows.ps1`, `README.md`: configuration and Windows rerun
  instructions.
- `validate.py`, `test_workflow.py`, `verify_products.py`: integration, unit and
  exported-product verification.
- `outputs/windows_run/protocol_statistics.xlsx`: seven sheets, including all
  five required sheets plus ranked protocols and protocol sweeps.
- Matching CSVs: 20 summary rows, 13,980 node/model rows, 67 acquisition rows,
  6,788 candidate requests, 226 controlled-sweep rows and 6 ranked protocols.
- `outputs/windows_run/figures/`: 40 figures, each in 300-dpi PNG, vector PDF and
  SVG (120 files), with embedded metadata and JSON sidecars.
- `run_report.json`, `validation.json`, `product_validation.json`,
  `candidate_evaluations.sqlite`: settings, evidence, provenance and reusable scores.

## Exact experiment

SNR_ref=50 at TE_ref=0 s; T2=0.040 s; TE offset=0.014 s.
Every acquisition, including b0, uses
`TE_A=max(delta+Delta)+0.014 s` and
`sigma_A=exp((TE_A-TE_ref)/T2)/SNR_ref`.
Budgets are 16 and 128 acquired volumes, including two true-b0 volumes each.
No gradient ceiling, trust filter or Rician filter is applied. The optional
Rician convention is explicitly recorded as the Phase-2 repeated-mean basis.
Endpoint-only MC derivative debiasing is enabled. Both fixed- and marginal-S0
results are exported; marginal S0 uses no independent uncounted prior.

The declared design contains 1,128 DW columns at 47 stored timing pairs, with
b=500:500:12000 s/mm², up to 12 selected columns and four timing pairs. The
objective population contains all 233 complete-stencil rho/V pairs at
k_io=5,20,40 s^-1, totaling 699 nodes. Relative k_io SD uses max(k_io,5 s^-1).
Invalid nodes count as +infinity in the per-parameter medians; the primary
objective is their maximum, subject to at least 51% identifiable coverage.

These are **best found within this declared design and node population**.
They are not a full-grid/global optimum or a hardware-feasibility claim.

## Best protocols

The marginal-S0 objective binds on k_io for both budgets. Both selected
protocols have TE=66 ms and single-volume normalized noise SD=0.1041395965.

| N | Marginal score | Marginal coverage | Fixed-S0 score | Fixed-S0 coverage | Binding node (marginal) |
|---|---:|---:|---:|---:|---:|
| 16 | 1.623809534 | 78.8269% | 1.596214347 | 79.1130% | 5066 |
| 128 | 0.590238852 | 78.5408% | 0.557788052 | 78.6838% | 6423 |

Columns are `(delta ms, Delta ms, b s/mm², repetitions)`:

```text
N=16
(2,50,500,2), (2,50,2500,6), (4,30,0,2),
(8,10,1000,5), (8,30,9500,1)

N=128
(2,50,500,9), (2,50,2500,61), (2,50,6000,8),
(8,15,0,2), (8,15,1000,32), (8,15,7500,2),
(8,20,7500,8), (12,15,4000,6)
```

The best searched single-timing baseline scores were 11.6586 (N=16) and
4.47998 (N=128); the best simple balanced two-timing baseline scores were
2.25012 and 0.793500. The workbook includes their full acquisitions and node
distributions, plus the top three distinct protocols per budget.

Search used random seed 20260925 plus budget, four starts of 450 proposals and
two local refinement passes. The initial complete numerical run took about
42.9 seconds before file export/plot generation. It made 5,476 fresh candidate
evaluations, reused 766 cached evaluations, and logged 6,788 total requests,
including design-invalid proposals. Each candidate's status/reason is retained.

## Reuse and validation

Reused: full-domain Phase-1 stored k=1 Jacobians/sample tables, Phase-2
column-major normalized signals and signal variances, source manifests and
Windows library metadata. The 31,125-column substrate remained intact.

Newly computed: endpoint-only derivative variances (not already stored over
the full domain), weighted complete-protocol FIMs, S0 Schur complements,
diagnostics, optimizer evaluations, tables and figures. Existing diagnostic
VarJ arrays include CRN covariance for only 200 columns and cannot replace the
uniform Phase-2 endpoint-only estimator. No Monte Carlo signals or production
finite-difference derivatives were regenerated.

Validation passed: 11 unit tests; exact FIM doubling and covariance halving
for both S0 models; common adaptive TE and retained-column sweep checks;
singular/non-PD/masked rejection; explicit 4×4 S0 inverse comparison; the same
optimizer matched the best of all 84 reduced-space allocations exactly.
A valid-node local calculation using the authoritative Phase-2 cached-data
helper agreed to 2.065e-7 relative FIM error and about 1.2e-6 relative SD error.
This uses the new common adaptive noise and does not claim reproduction of the
historical Phase-2 noise assumptions.

Independent CSV checks recomputed robust scores, acquisition budgets and TE,
confirmed unique record keys and blank invalid covariance, and found a maximum
sampled scaled inverse residual of 3.64e-12. All 11 recorded source files kept
their original sizes/timestamps and the source manifests kept their hashes.
