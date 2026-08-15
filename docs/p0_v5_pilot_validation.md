# P0 v5 remediation-pilot validation

Tier A validation record received 2026-08-11.  Source:
`logs/madi_v5_remediation_pilot_validation.json`, SHA-256
`cec9a653dc87a76a28bf02175da49e8a44bc6d67f9694a53cc00db2ce2987594`.

## Verdict

**PASS — v5 schema, storage, and provenance pilot.** The four returned
shards satisfy the v5 diagnostic-array contract and clear the schema/storage
gate for a production-launch decision.  This is not a production Monte-Carlo
precision result and must not be used for fitting data.

The returned JSON does not record the Slurm array job ID or shard SHA-256
values, and the shard files are not present in this checkout.  They must be
retained on Sol and hashed using the runbook before the production audit.

## Returned validation results

| Shard | Entries | Main signal shape | Ensemble-subset shape | Max subset-mean error |
|---|---:|---:|---:|---:|
| `000` | 7 | `(7, 1500)` | `(7, 8, 200)` | `1.8980804e-08` |
| `001` | 6 | `(6, 1500)` | `(6, 8, 200)` | `8.2146686e-09` |
| `002` | 6 | `(6, 1500)` | `(6, 8, 200)` | `1.3905702e-08` |
| `003` | 6 | `(6, 1500)` | `(6, 8, 200)` | `1.4260456e-08` |

The combined result contains 25 entries: 24 cellular entries and one
free-water atom.  Shard coverage is exactly `7,6,6,6`, with no reported
errors, no non-finite signals, and no exact cross-label signal ties.  The
minimum pairwise vector distance is `0.2876517`.

## v5 diagnostic acceptance

- Validator schema: `madi-remediation-pilot-validation-v2`; `pass: true`.
- The immutable 200-column diagnostic subset reconstructs the matching main
  signal columns to a maximum absolute error of `1.8980804e-08`, well within
  the float32 cross-check tolerance.
- The artifact carries eight independently constructed ensemble means for
  each declared subset column, preserving the recorded common-random-number
  ensemble-index contract.
- The largest absolute standardized imaginary signal is `10.8005213`, below
  the declared two-sided family-wise threshold `19.1184610` at alpha `0.01`.
  The imaginary symmetry check therefore passes at the pilot sample size.

## Pilot-only limitations

The pilot intentionally uses 8 ensembles of 512 walkers, or 12,288 harvested
axis-walks per entry, rather than the production 12,000,000.  It also uses a
compact 1,500-column timing/acquisition grid rather than the production
31,125-column grid.  It validates the schema path, not signal precision.

The validator reports 4,952 negative low-signal samples, with minimum
`-0.0222133`.  This is an expected low-Monte-Carlo pilot warning, not a schema
failure or production-quality signal result.  It reinforces that the existing
analysis/fit trust-floor policy, rather than this pilot, controls low-signal
use.  The independent-seed W4 replicate remains the post-production check of
the W2 variance estimator's calibration.

## Recommendation

**GO for the v5 schema gate, conditional on scratch-capacity confirmation and
explicit authorization to submit production.** No production or W4 replicate
job is authorized or launched by this record.

Use [`v5_pilot_runbook.md`](v5_pilot_runbook.md) for the reproducible current
pilot configuration and Sol commands.
