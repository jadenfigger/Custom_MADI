# MADI v5 pilot build and validation runbook

This is the current Sol runbook for the v5 schema pilot.  It supersedes the
historic one-ensemble pilot commands in `p0_sol_execution.md`.  It does not
authorize a production or W4 replicate submission.

## Pilot settings

| Setting | Value |
|---|---|
| Artifact base name | `libraries/madi_v5_remediation_pilot.npz` |
| Shards | 4 (`000` through `003`); expected entries `7,6,6,6` |
| Entry grid | 8-by-12 uniform-log `(rho,V)` grid, `rho=1e4..1e7` cells/uL and `V=0.01..200` pL, masked to `0.40 <= v_i <= 0.99` |
| Entries | 8 retained cellular `(rho,V)` pairs x `k_io={0,20,130}` s^-1, plus one free-water atom = 25 |
| Monte Carlo | 8 independent ensembles x 512 walkers x 3 harvested axes = 12,288 axis-walks per entry |
| Build seed | `20260803`; ensemble index is the CRN partner across entries |
| Physics | 128-ms finite-lobe walk; `kappa=0.90`; exact full-facet SI Eq. S2 classifier; SI fatal-escape boundary; certified 5-million-cell geometry reference |
| Timing grid | delta = `5,7,10,12,15,20,25,30` ms; Delta = `15,25,30,36,40,50,60,80` ms; 60 valid triangular pairs |
| b grid | 0 through 12,000 s/mm2 in 500 s/mm2 steps: 25 values |
| Stored columns | 60 timing pairs x 25 b values = 1,500 per entry |
| v5 subset | 8 declared timing pairs x all 25 b values = 200 columns; stored as 8 per-ensemble means per entry |
| Sol resources | `htc` / `public`, one GPU, 16 CPU cores, 24 GiB host memory, 4-hour array limit |

The pilot proves the v5 storage, indexing, and metadata path.  It does not
establish production Monte-Carlo precision, signal monotonicity at the tail,
or calibration of the W2 uncertainty estimator.

## 1. Update and pre-flight check on Sol

Run from the Sol login node.  Do not activate Mamba here; the batch launcher
uses the required clean `--export=NONE` environment and activates `madiEnv`
inside the job.

```bash
cd /scratch/jfigger/madi/Custom_MADI
git status --short
git fetch origin
git pull --ff-only origin main
git log -1 --oneline
grep -nE 'madi-remediation-pilot-validation-v2|PILOT_N_ENSEMBLES' scripts/validate_remediation_pilot.py
grep -n 'madi_v5_remediation_pilot' scripts/build_lib.sbatch
cd data
sha256sum -c geometry_reference_si_kappa_0p9.npz.sha256
cd ..
myquota
```

Stop if the fast-forward pull fails, the checksum does not print `OK`, or
scratch capacity is not comfortably above the raw final-artifact floor and
its simultaneous merge copies.  Do not repair a dirty checkout with reset or
checkout commands; inspect its status first.

## 2. Submit the four-task pilot array

```bash
cd /scratch/jfigger/madi/Custom_MADI
sbatch --partition=htc --qos=public --array=0-3 \
  --time=0-04:00:00 --cpus-per-task=16 --mem=24G -G 1 \
  scripts/build_lib.sbatch pilot 4
```

The smaller `htc`/4-hour request is appropriate for the restricted pilot.
Production uses the materially larger `public`/one-day/64-GiB configuration;
do not substitute this command for a production submission.

## 3. Verify completion, preserve hashes, and validate

Wait for all four array tasks to be `COMPLETED` with exit status zero.  Then
run the following on Sol:

```bash
cd /scratch/jfigger/madi/Custom_MADI
sha256sum \
  libraries/madi_v5_remediation_pilot.shard000.npz \
  libraries/madi_v5_remediation_pilot.shard001.npz \
  libraries/madi_v5_remediation_pilot.shard002.npz \
  libraries/madi_v5_remediation_pilot.shard003.npz \
  > logs/madi_v5_remediation_pilot_shards.sha256
module load mamba/latest
source activate madiEnv
which python
python -m scripts.validate_remediation_pilot \
  --shards \
    libraries/madi_v5_remediation_pilot.shard000.npz \
    libraries/madi_v5_remediation_pilot.shard001.npz \
    libraries/madi_v5_remediation_pilot.shard002.npz \
    libraries/madi_v5_remediation_pilot.shard003.npz \
  --output logs/madi_v5_remediation_pilot_validation.json
```

The validator must exit zero and report `schema:
madi-remediation-pilot-validation-v2` and `pass: true`.  Retain the four
shards, their hash file, the JSON report, and all four stdout/stderr logs.
Do not launch production or the W4 replicate solely because the command was
submitted; require the completed validation result and explicit approval.
