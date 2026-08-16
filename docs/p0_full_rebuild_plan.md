# Contingent full MADI-library rebuild plan

This plan is a specification only. It does not authorize a production launch.
The P0 geometry, boundary, grid, and pilot structural gates have passed.  The
exact-classifier cache still requires its Sol GPU equivalence/speed gate; see
`v5_fast_classifier_launch_readiness.md` before any submission.

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
24 columns per entry, versus 12,000,000 axis-walks and 31,125 columns here.

The current grid count is 51, not the 55-value count mentioned in the earlier
audit prompt. P0-E requires a declared uniform/piecewise-uniform grid and
analytic weights, which this design satisfies; changing the 51-point spacing
to a particular 55-point scheme would be a separate parameter-resolution
decision and is not made silently here.

## Sol launch, only after explicit approval

The revised layout has 369 shards: exactly one cellular `(rho,V)` group per
task, ordered monotonically by rho, with the free-water atom on shard 0.  The
array is split into three independent rho bands (task ranges 0--126,
127--247, and 248--368) so their wall-time requests can be truthful.  The
band limits and submission commands are intentionally not set until the
measured GPU fast-path speed is available.  Resharding changes queue and
wall-time behaviour, not the total allocation consumed.

## Before declaring success

1. Slurm must report every array task `COMPLETED`, with no escape error,
   geometry-target assertion, or failed p_p range assertion in any log.
2. Exactly `madi_dense_universal_remediated.shard000.npz` through
   `shard368.npz` must exist. Do not overwrite the current production library.
3. Confirm a representative `seff <jobid_task>` report before committing to a
   larger repeat or changed resource request.
4. On Sol, merge only after all 369 shards exist:

   ```bash
   module load mamba/latest
   source activate madiEnv
   cd /scratch/jfigger/madi/Custom_MADI
   python -m scripts.merge_shards \
     libraries/madi_dense_universal_remediated.shard*.npz \
     --require-shards 369 \
     --out libraries/madi_dense_universal_remediated.npz
   ```

5. Return the merged artifact, all shard hashes, and representative logs for
   the post-build physics audit. Do not replace `madi_dense_universal.npz`
   until the merged artifact passes that audit.

## Remaining non-P0 recommendation

The pilot makes P2-L observable rather than theoretical: low Monte Carlo
counts produce negative tail samples and nonmonotonic decay samples. Implement
the explicit `S/S0 = 0.015` trust-floor rule before any production library is
used in a fitter, even if the high-statistics build is expected to reduce the
problem substantially.
