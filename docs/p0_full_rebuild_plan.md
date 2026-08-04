# Contingent full MADI-library rebuild plan

This plan is a specification only. It does not authorize a production launch.
The P0 geometry, boundary, CPU/GPU-equivalence, grid, and pilot structural
gates have passed; the remaining decision is whether to spend the shared Sol
allocation on this production-scale artifact.

## Production configuration

| Item | Value |
|---|---:|
| Coordinate grid | 64 uniform-log `rho` nodes × 64 uniform-log `V` nodes, masked to `0.40 <= rho*V*1e-6 <= 0.99` |
| Retained `(rho,V)` pairs | 369 |
| `k_io` grid | 51 values, `0`, `1..30` by 1, and `35..130` by 5 s^-1 |
| Entries | 18,819 cellular + 1 free-water atom = 18,820 |
| Quadrature | analytic `rho*V*dlogrho*dlogV*dkio`, plus discrete free-water weight |
| Walk | 128 ms, 1 us steps, finite-lobe phase model |
| Monte Carlo | 100,000 walkers × 40 ensembles × 3 axes = 12,000,000 axis-walks/entry |
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

This exceeds 240 minutes, so use the `public/public` production partition,
not `htc`. The existing launcher requests one GPU, 16 CPU cores, 64 GiB host
memory, and a one-day wall limit; it retains `--export=NONE` and performs the
required Mamba activation itself.

```bash
cd /scratch/jfigger/madi/Custom_MADI
git status --short
git log -1 --oneline
cd data
sha256sum -c geometry_reference_si_kappa_0p9.npz.sha256
cd ..

sbatch --partition=public --qos=public --array=0-127 \
  --time=1-00:00:00 --cpus-per-task=16 --mem=64G -G 1 \
  scripts/build_lib.sbatch production 128
```

There are 128 shards. Round-robin assignment over 369 geometry groups gives
two or three `(rho,V)` groups per shard, or approximately 102--153 cellular
entries per shard (plus the free-water atom on shard 0).

## Before declaring success

1. Slurm must report every array task `COMPLETED`, with no escape error,
   geometry-target assertion, or failed p_p range assertion in any log.
2. Exactly `madi_dense_universal_remediated.shard000.npz` through
   `shard127.npz` must exist. Do not overwrite the current production library.
3. Confirm a representative `seff <jobid_task>` report before committing to a
   larger repeat or changed resource request.
4. On Sol, merge only after all 128 shards exist:

   ```bash
   module load mamba/latest
   source activate madiEnv
   cd /scratch/jfigger/madi/Custom_MADI
   python -m scripts.merge_shards \
     libraries/madi_dense_universal_remediated.shard*.npz \
     --require-shards 128 \
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
