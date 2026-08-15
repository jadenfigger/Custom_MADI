# v5 production-grid stencil probe

## Current status and verdict

**INVESTIGATE — no production launch and no configuration change is authorized
yet.** This record prepares the pre-launch restricted probe required by the
v5 pilot CRN diagnostic. The probe artifact does not exist in this checkout,
so no CRN, beta, noise, or reallocation result is reported here. The completed
artifact analysis command below regenerates this file with the final result
and a GO / GO-WITH-CHANGE / INVESTIGATE verdict.

This probe is distinct from both the production library and the post-production
W4 independent-seed replicate. It must not overwrite either.

## Declared production-grid cross

[`data/madi_v5_stencil_probe_entry_subset.json`](../data/madi_v5_stencil_probe_entry_subset.json)
is a declared, canonical-grid-restricted entry set. The launcher resolves
every coordinate against `make_remediation_log_grid()` before it builds any GPU
entry, so a typo or out-of-grid coordinate aborts rather than changing the
production quadrature or simulator behavior.

The source grid has 64 nodes in each log coordinate. `madi/config.py` supplies
the timing/physics defaults; the actual rho/V production-grid generator is
`madi.library.make_remediation_log_grid()`. Its exact code-derived steps are

- `ln(rho[i+1]/rho[i]) = 0.10964690919019218`, or
  `0.0476190476190474` decades;
- `ln(V[j+1]/V[j]) = 0.15719821511962115`, or
  `0.0682703173914918` decades.

Those differ slightly from the prompt's quoted `0.047648` and `0.068259`
decades. The declaration and analysis use the canonical `geomspace` values,
rather than silently choosing a rounded alternative.

The node nearest the geometric centre of the mask is `(rho_index, V_index) =
(21, 41)`: `rho = 100,000` cells/uL, `V = 6.2962652107424715` pL, and
`v_i = 0.6296265210742472`. The retained cross is:

| Line | Canonical nodes | Supported central half-widths |
|---|---|---|
| rho at `V_index=41` | `rho_index=17..25` | `k=1,2,3,4` |
| V at `rho_index=21` | `V_index=39..43` | `k=1,2` |
| k_io at every cross pair | `19,20,21` s^-1 | `k=1` |

The two lines share their centre, so there are 13 retained `(rho,V)` pairs.
Crossing them with the three production `k_io` nodes yields exactly 39 cellular
entries and no free-water atom. The resource-efficient Sol run below assigns
one `(rho,V)` group (its three `k_io` entries) to each of 13 shards.

The analysis uses these declared canonical indices to define rho/V adjacency.
This is intentional: v5 `rhos` and `Vs` are finite-geometry realized summary
labels, not requested controls. Requiring literal equality of realized labels
would again erase every intended one-axis rho/V line. Those realized values are
still checked and reported as provenance; all stored `nominal_*` labels must
match the declaration exactly.

## Build settings

The `stencil-probe` launcher mode in
[`scripts/build_lib.sbatch`](../scripts/build_lib.sbatch) uses the unchanged
dense production preset: 100,000 walkers x 40 ensembles x 3 harvested axes,
128 ms at 1 us, finite-lobe phase, kappa 0.90, SI fatal escape, full-facet SI
Eq. S2 classifier, production seed `20260803`, and the certified 5-million-cell
geometry reference. It has no timing or b override, so it stores the full
production 1,245 timing-pair x 25 b-value grid plus the unchanged 200-column
v5 diagnostic subset.

The probe uses a 20-GiB A100 MIG slice on `htc`, not a full A100. The dominant
CUDA buffer is `100,000 x 129 x 3` float64 values, or about 0.29 GiB; the
remaining walker buffers and one active geometry are far below the 20-GiB
device limit. A 10-GiB slice would also fit the memory footprint, but has less
compute capacity and no useful safety advantage for this long-running probe.

MIG slices have less compute capacity than a full A100, so simply swapping the
old four 12-entry tasks to a slice could exceed the `htc` four-hour limit.
Thirteen one-geometry-group tasks preserve the identical 39-entry experiment
while keeping each task short enough for backfill. The `%1` throttle deliberately
allows only one 20-GiB slice at a time: the peak allocation is one GPU, two CPU
cores, and 8 GiB host RAM. This improves scheduling eligibility at the cost of
longer total elapsed time. The pilot's peak host RAM was 1.17 GiB and its CPU
use was approximately one core; 8 GiB and two cores retain a conservative
margin for the 40-ensemble/full-column probe. No production array, W4 array,
merge, or overwrite is included.

## 1. Sol pre-flight commands

Run these exact commands from the Sol login node. Do not activate Mamba on the
login node; the batch launcher has `#SBATCH --export=NONE` and performs the
required `module load mamba/latest` / `source activate madiEnv` sequence inside
the allocated job.

```bash
cd /scratch/jfigger/madi/Custom_MADI
git status --short
git fetch origin
git pull --ff-only origin main
git log -1 --oneline
git rev-parse HEAD
grep -n 'stencil-probe' scripts/build_lib.sbatch
grep -n 'madi-v5-stencil-probe-entry-subset-v1' scripts/fit_data.py
grep -n 'expected_cellular_entries' data/madi_v5_stencil_probe_entry_subset.json
cd data
sha256sum -c geometry_reference_si_kappa_0p9.npz.sha256
cd ..
sha256sum data/madi_v5_stencil_probe_entry_subset.json > logs/madi_v5_stencil_probe_definition.sha256
myquota
```

Stop if the pull is not a fast-forward, the geometry checksum does not print
`OK`, the declaration hash cannot be written, or quota is inadequate for thirteen
full-column shards and their simultaneous diagnostic output. Do not repair a
dirty checkout with reset or checkout commands.

## 2. Submit the MIG-slice restricted probe

```bash
cd /scratch/jfigger/madi/Custom_MADI
sbatch --job-name=madi_v5_stencil_probe --output=logs/madi_v5_stencil_probe_%A_%a.out --error=logs/madi_v5_stencil_probe_%A_%a.err --partition=public --qos=public --array=0-12 --time=0-12:00:00 --cpus-per-task=2 --mem=24G -G a100.20gb:1 scripts/build_lib.sbatch stencil-probe 13
```

The launcher writes these 13 distinct files:

```text
libraries/madi_v5_stencil_probe.shard000.npz
...
libraries/madi_v5_stencil_probe.shard012.npz
```

## 3. Completion, hashes, and artifact validation

Wait until all 13 tasks are complete with zero exit status. These commands
avoid an unknown job-id placeholder by querying the explicit job name.

```bash
cd /scratch/jfigger/madi/Custom_MADI
squeue -u jfigger -n madi_v5_stencil_probe
sacct -u jfigger --name=madi_v5_stencil_probe \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,MaxRSS
sha256sum libraries/madi_v5_stencil_probe.shard0{00..12}.npz \
  > logs/madi_v5_stencil_probe_shards.sha256
cat logs/madi_v5_stencil_probe_definition.sha256
cat logs/madi_v5_stencil_probe_shards.sha256
module load mamba/latest
source activate madiEnv
which python
python -m scripts.validate_v5_stencil_probe \
  --declaration data/madi_v5_stencil_probe_entry_subset.json
python analysis/v5_stencil_probe.py libraries/madi_v5_stencil_probe.shard0{00..12}.npz --declaration data/madi_v5_stencil_probe_entry_subset.json --expected-shards 13 --output-dir docs/figures/v5_stencil_probe --report docs/v5_stencil_probe.md
```

The analysis aborts before calculating a correlation if any of the following
fails: v5 shapes/dtypes; full production storage grid; production metadata;
ensemble-index CRN contract; variance reconstruction; subset mean
reconstruction; exact declared nominal coordinates; or identical
`per_ensemble_geometry` records within all 13 fixed-nominal `(rho,V)` k_io
groups.

On success it writes the per-axis correlation histograms, direct observed SE
versus b/timing, beta-versus-stencil figure, reallocation table, full CSVs, and
a JSON summary under `docs/figures/v5_stencil_probe/`. It then replaces this
pre-launch record with the final validation report. Retain all 13 shards,
hash files, Slurm logs, CSVs, JSON, figures, and generated report before any
GO decision. Do not launch production or W4 merely because this probe
completed.
