# P0 Sol execution package

This execution package defines the deterministic CPU/GPU full-facet golden
check and the small pilot library. It follows
`docs/sol_package_guide.md`: the existing `scripts/build_lib.sbatch` is the
only launcher; it has `#SBATCH --export=NONE`, and performs exactly `module
load mamba/latest` followed by `source activate madiEnv`.  Do not pre-activate
an environment before submitting: the job deliberately does not inherit it.
For GPU work expected to finish in under 240 minutes, submit to `htc` with the
`public` QOS and at least 24 GiB host memory.  The production launcher default
remains `public/public` because its wall time is one day.

The Sol repository path below is `/scratch/jfigger/madi/Custom_MADI`.

## Execution record — 2026-08-03

- `gpu_golden_60169767` and `gpu_golden_60222142` are retired cache-era runs:
  schema `v1`, golden hash `226946…`, and obsolete survivor metrics.
- `gpu_golden_60222901` is the accepted full-facet CPU/GPU check. It reports
  schema `madi-cpu-gpu-golden-report-v3-full-facet`, fixture SHA-256
  `9db549929e5b63542b5de335fbc415419d3e61909f82dc0604aa98fb366ec9df`,
  `pass: true`, zero escapes, identical occupancy, and maximum `Y` difference
  `5.684341886080802e-14` under `atol=5e-12`, `rtol=5e-11`.
- Pilot array `60223180` completed all four tasks. Its returned shards pass
  the structural/provenance validator; see `docs/p0_pilot_validation.md`.
  The pilot's deliberately low walker count makes it unsuitable as a fitting
  library and it must not be substituted for a production build.

## Gate before either submission

No environment/module command is required in the submitting login shell.
Before the **pilot** (not required by the golden check), verify that the
version-controlled geometry-reference artifact is present on Sol:

```bash
cd /scratch/jfigger/madi/Custom_MADI
sha256sum data/geometry_reference_si_kappa_0p9.npz
```

Expected SHA-256:

```text
6a2957eb3d6f89fdebc61d83e7cabb65c67812415879e00b5c07614680785e47
```

If the file is absent or its hash differs, stop before the pilot and pull the
commit that carries it (or recover that exact validated artifact).  Do not
regenerate it on Sol as part of this GPU work; its build is a separate Tier-B
CPU artifact.

## 1. Tier C — CPU/GPU full-facet golden check

Submit exactly:

```bash
cd /scratch/jfigger/madi/Custom_MADI
sbatch --partition=htc --qos=public --array=0-0 --time=00:20:00 \
  --cpus-per-task=4 --mem=24G -G 1 \
  scripts/build_lib.sbatch golden
```

Requested resources: one GPU, four CPU cores, 24 GiB host RAM, 20 minutes on
`htc`. Expected runtime is under five minutes; the larger request leaves room
for queue/node startup and CUDA JIT compilation while meeting Sol's GPU-memory
minimum.

Before considering the job a pass, carry back these exact files (replace
`<jobid>` with the submission ID):

- `sol_outputs/gpu_golden_<jobid>/gpu_golden_diff.json`
- `sol_outputs/gpu_golden_<jobid>/gpu_golden_result.npz`
- `logs/madi_dense_<jobid>_0.out`
- `logs/madi_dense_<jobid>_0.err`

Pass means Slurm reports `COMPLETED` with exit status 0, and the JSON reports:

- schema `madi-cpu-gpu-golden-report-v3-full-facet`;
- `golden_sha256` =
  `9db549929e5b63542b5de335fbc415419d3e61909f82dc0604aa98fb366ec9df`;
- `cuda_available: true`, `pass: true`, and `n_escaped_gpu: 0`; and
- `Y` and `occupancy_fraction` each have `allclose: true` under
  `atol=5e-12`, `rtol=5e-11`.

Any nonzero job exit, missing JSON, a false `pass`, an escape, or a hash
mismatch is a failure.  Do not submit the pilot in that case; return the four
files above for diagnosis.

## 2. Tier C — restricted pilot library

Run this only after the golden check passes and the reference SHA check above
matches.  Submit exactly:

```bash
cd /scratch/jfigger/madi/Custom_MADI
sbatch --partition=htc --qos=public --array=0-3 --time=02:00:00 \
  --cpus-per-task=8 --mem=32G -G 1 \
  scripts/build_lib.sbatch pilot 4
```

Requested resources: four independent one-GPU `htc` array tasks, each with
eight CPU cores, 32 GiB host RAM, and a two-hour wall-clock limit.  The
expected runtime is 30--90 minutes per task; this is an estimate because the
new full-facet CUDA traversal has not yet been throughput-benchmarked on Sol.
The pilot is intentionally small enough that a two-hour failure provides a
useful capacity signal rather than an expensive production-scale loss.

The pilot is a 8-by-12 uniform-log `(rho,V)` lattice over
`rho=1e4..1e7 cells/uL` and `V=0.01..200 pL`, masked to `0.40 <= v_i <= 0.99`.
It contains eight retained cellular pairs (`v_i=0.4283..0.9664`), each at
`k_io={0,20,130} s^-1`, plus the separate free-water atom: **25 entries** in
total.  Shard 0 has seven entries; shards 1--3 have six each.  Each entry has
one 512-walker ensemble, a 128-ms finite-lobe walk, four `(delta,Delta)`
pairs `(7,25)`, `(7,50)`, `(20,25)`, `(20,50)` ms, and six b values
`0,500,1000,2000,4000,6000 s/mm²`.  This is a structural/provenance pilot,
not a production precision library.

After all four jobs complete, carry back:

- `libraries/madi_remediation_pilot.shard000.npz`
- `libraries/madi_remediation_pilot.shard001.npz`
- `libraries/madi_remediation_pilot.shard002.npz`
- `libraries/madi_remediation_pilot.shard003.npz`
- `logs/madi_dense_<jobid>_0.out` through `logs/madi_dense_<jobid>_3.out`
- `logs/madi_dense_<jobid>_0.err` through `logs/madi_dense_<jobid>_3.err`

Do not launch the production library or merge the pilot as a substitute for
verification.  After the shard files are returned locally, run the checklist
as one CPU-only command:

```bash
PYTHONPATH=. python -m scripts.validate_remediation_pilot \
  --shards libraries/madi_remediation_pilot.shard*.npz \
  --output logs/remediation_pilot_validation.json
```

It exits zero only on a pass.  The pilot acceptance checklist is:

1. exactly four distinct shards and exactly 25 unique entries—the same shard
   coverage condition that `merge_shards.py --require-shards 4` enforces;
2. free-water atom present; cellular `k_io=0` entries present; all weights
   finite and positive; and no invalid `rho*V*1e-6` labels;
3. stored `(delta,Delta,b)` axes match the specified 4-by-6 grid and vectors
   have 24 columns in pair-major, b-major order;
4. build metadata says `finite_lobe`, `kappa=0.9`, `si_fatal_escape`, full
   Eq. S2 classifier, certified untrimmed `A/V` reference, and common random
   numbers;
5. every entry stores measured geometry and Eq. 5 `p_p`/`k_io_analytic_eq5`,
   with zero escapes and no geometry label outside its tolerance; and
6. no missing/duplicated shard, NaN/negative configuration fields, or signal
   vectors tied across distinct labels.

Only after this checklist passes will a full rebuild command be prepared.
