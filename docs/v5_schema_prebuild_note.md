# MADI library v5 schema: pre-build note

This note records the additive Monte-Carlo diagnostics required before the
next production library build.  It does not authorize a production or
replicate submission.

## Stored arrays

The existing real `vectors` matrix is unchanged: one collapsed real signal
per `(entry, delta, Delta, b)` column, stored as float64.  v5 adds:

| Array | Shape | dtype | Meaning |
|---|---:|---|---|
| `signal_imag` | `(entry, column)` | float32 | Collapsed `mean(sin(phase))`; a symmetry / physics check. |
| `signal_variance` | `(entry, column)` | float32 | Sample variance (`ddof=1`) of the independent ensemble mean real signals. It is **not** a standard error and is **not** a pooled walker / axis variance. The consumer uses `sqrt(signal_variance / n_ensembles)`. |
| `ensemble_means_subset` | `(entry, ensemble_index, declared_column)` | float32 | Real per-ensemble means for 200 declared columns, retained to estimate CRN covariance between entries. |

The 200 declared columns are pair-major then b-major: `(5,15)`, `(7,25)`,
`(10,30)`, `(12,36)`, `(15,40)`, `(20,50)`, `(25,60)`, `(30,80)` ms, each
crossed with b = 0, 500, ..., 12000 s/mm².  The `ensemble_index` axis is
always `e=0..n_ensembles-1`; the v5 metadata explicitly records that the same
index is the common-random-number partner across every entry, using walk seed
`(build_seed + 104729*e) mod 2^31` and a matching fixed geometry-index order.

The builder also records one per-entry maximum imaginary Student statistic in
metadata, together with its column, mean, and standard error.  This is only a
compact validation aid: per-ensemble imaginary signals are not written, so it
is the way to evaluate the requested zero-symmetry standardization without
adding another full matrix.  The validator cross-checks that recorded mean
against `signal_imag` at the recorded column.

## Expected production artifact size

For 18,820 entries, 31,125 main columns, 40 ensembles, and 200 diagnostic
columns, raw uncompressed numeric array sizes are:

| Array | Bytes | GiB |
|---|---:|---:|
| `vectors` (float64) | 4,686,180,000 | 4.3643 |
| `signal_imag` (float32) | 2,343,090,000 | 2.1822 |
| `signal_variance` (float32) | 2,343,090,000 | 2.1822 |
| `ensemble_means_subset` (float32) | 602,240,000 | 0.5609 |
| **These four matrices** | **9,974,600,000** | **9.2896** |

This is the raw matrix floor.  NPZ members are written uncompressed, and
entry metadata, parameter arrays, ZIP headers, checkpoints, and any
simultaneous merge input/output copies add space beyond it.  Confirm scratch
has materially more than 9.29 GiB available for the final merged artifact and
the merge workflow before launch; this note intentionally does not guess a
quota or silently treat the raw floor as a safe allocation.

## Pilot and replicate scope

The v5 pilot retains the old 25-entry coverage and four-shard layout but uses
8 independent ensembles of 512 walkers.  It uses a compact 60-pair timing
grid that contains every declared diagnostic pair and the unchanged 25-point
b grid.  It tests schema/storage/provenance only, not production Monte-Carlo
precision.

`data/madi_v5_replicate_entry_subset.json` declares the later W4 restricted
replicate: 21 spread `(rho,V)` pairs crossed with nine k_io values plus free
water, for 190 entries.  The launcher rejects coordinates outside the
canonical P0 grid and uses a distinct fixed build seed.  It remains explicitly
unlaunched until after production and separate approval.
