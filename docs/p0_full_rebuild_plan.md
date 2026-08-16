# Contingent full MADI-library rebuild plan

This plan is a specification only. It does not authorize a production launch.
The P0 geometry, boundary, grid, and pilot structural gates have passed. The
exact-classifier cache passed its CPU/GPU equivalence and full-A100 speed
gates; see `v5_fast_classifier_launch_readiness.md` for the measured results
and the final user-controlled launch decision.

## Production configuration

| Item | Value |
|---|---:|
| Coordinate grid | 64 uniform-log `rho` nodes × 64 uniform-log `V` nodes, masked to `0.40 <= rho*V*1e-6 <= 0.99` |
| Retained `(rho,V)` pairs | 369 |
| `k_io` grid | 51 values, `0`, `1..30` by 1, and `35..130` by 5 s^-1 |
| Entries | 18,819 cellular + 1 free-water atom = 18,820 |
| Quadrature | analytic `rho*V*dlogrho*dlogV*dkio`, plus discrete free-water weight |
| Walk | 128 ms, 1 us steps, finite-lobe phase model |
| Monte Carlo | 50,000 walkers × 40 ensembles × 3 axes = 6,000,000 axis-walks/entry |
| Stored columns | 1,245 `(delta,Delta)` pairs × 25 b values = 31,125/entry |
| Geometry / exchange | certified 5,000,000-cell SI reference, `kappa=0.9`, full SI Eq. S2 facets, SI Eq. S8 domain, Eq. 5 untrimmed governing-process `<A/V>` |

The uncompressed float64 signal matrix alone is approximately 4.36 GiB;
per-entry metadata and archive overhead raise the final artifact size. The
pilot cannot predict wall time reliably because it used 1,536 axis-walks and
24 columns per entry, versus 6,000,000 axis-walks and 31,125 columns here.

The current grid count is 51, not the 55-value count mentioned in the earlier
audit prompt. P0-E requires a declared uniform/piecewise-uniform grid and
analytic weights, which this design satisfies; changing the 51-point spacing
to a particular 55-point scheme would be a separate parameter-resolution
decision and is not made silently here.

## Sol launch, only after explicit approval

The revised layout has 369 shards: exactly one cellular `(rho,V)` group per
task, ordered monotonically by rho, with the free-water atom on shard 0.  The
full-A100 cache benchmark (job 61533979) measured cached group maxima of
2.89 h, 5.16 h, and 9.21 h across the three rho bands.  The production plan
therefore uses a full A100, one CPU, Sol's mandatory 24 GiB host-memory
minimum, and independent, unthrottled high/centre/low submissions with 19 h,
11 h, and 6 h limits, respectively. Omitting `%N` lets the scheduler use as
many available A100s as fair-share permits; it does not change total
GPU-hours.

```bash
PROD_HIGH_JOB="$(sbatch --partition=public --qos=public --array=248-368 --time=0-19:00:00 --cpus-per-task=1 --mem=24G -G a100:1 scripts/build_lib.sbatch production 369 | awk '{print $4}')"
printf 'High-rho job: %s\n' "${PROD_HIGH_JOB}"
PROD_CENTRE_JOB="$(sbatch --partition=public --qos=public --array=127-247 --time=0-11:00:00 --cpus-per-task=1 --mem=24G -G a100:1 scripts/build_lib.sbatch production 369 | awk '{print $4}')"
printf 'Centre-rho job: %s\n' "${PROD_CENTRE_JOB}"
PROD_LOW_JOB="$(sbatch --partition=public --qos=public --array=0-126 --time=0-06:00:00 --cpus-per-task=1 --mem=24G -G a100:1 scripts/build_lib.sbatch production 369 | awk '{print $4}')"
printf 'Low-rho job: %s\n' "${PROD_LOW_JOB}"
```

## Before declaring success

1. Slurm must report every array task `COMPLETED`, with no escape error,
   geometry-target assertion, or failed p_p range assertion in any log.
2. Exactly `madi_dense_universal_remediated.shard000.npz` through
   `shard368.npz` must exist. Do not overwrite the current production library.
3. Confirm a representative `seff <jobid_task>` report before committing to a
   larger repeat or changed resource request.
4. On Sol, merge only after all 369 shards exist and the log scan is clean.
   Submit the committed post-processing job; it hashes every shard before
   merging, requires exactly ids 0--368, validates the merged v5 artifact, and
   refuses to overwrite the current production library or prior evidence:

   ```bash
   cd /scratch/tksimmo2/madi/custom_madi
   PROD_POST_JOB="$(sbatch scripts/postprocess_v5_production.sbatch | awk '{print $4}')"
   printf 'Production post-processing job: %s\n' "${PROD_POST_JOB}"
   squeue -j "${PROD_POST_JOB}"
   ```

5. Return `logs/madi_dense_universal_remediated_shards.sha256`,
   `logs/v5_production_validation.json`, representative array logs, and a
   representative `seff` report for the post-build physics audit. Do not
   replace `madi_dense_universal.npz` until the merged artifact passes that
   audit.

## Remaining non-P0 recommendation

The pilot makes P2-L observable rather than theoretical: low Monte Carlo
counts produce negative tail samples and nonmonotonic decay samples. Implement
the explicit `S/S0 = 0.015` trust-floor rule before any production library is
used in a fitter, even if the high-statistics build is expected to reduce the
problem substantially.
