# Mouse Brain DWI Preprocessing — Sol Setup & Run Procedure

This document consolidates the review of the uploaded scripts (`run_cohort.sbatch`, `preprocess.sh`, `submit_cohort.sh`, `check_cohort.sh`, `manifest.tsv`) against ASU Sol's mandatory practices (from the official docs and your `sol_package_guide.md`), and provides the full setup + run procedure.

---

## Part 1 — Review of uploaded scripts

### Critical issues in `run_cohort.sbatch` (must fix before next submission)

Your sbatch script violates three of the five "must never break" rules from `sol_package_guide.md`. All three are silent failure modes — the job will start, produce output, and then die in a confusing way.

| # | Rule violated | What your script has | What it needs |
|---|---|---|---|
| 1 | Always include `#SBATCH --export=NONE` | Missing | Add the line |
| 2 | Never use `mamba activate` / `conda activate` | `mamba activate mrtrix3` | `source activate mrtrix3` |
| 3 | Never use `set -u` (or `set -o pipefail`) around mamba activation | `set -euo pipefail` | `set -e` only |

The combination of #1 and #3 is *exactly* the "bug we spent hours debugging" that your own guide warns about. Without `--export=NONE`, the job inherits a half-configured Mamba state from the submitting shell. `source activate mrtrix3` then silently no-ops, `which python` returns `/packages/apps/mamba/.../bin/python` instead of the env's Python, and `nibabel` / `dwidenoise` / `eddy_cuda` are not on PATH.

### Minor issues in `run_cohort.sbatch`

- The `conda.sh` / `mamba.sh` sourcing block is unnecessary on Sol. `module load mamba/latest` already exposes `source activate`; the explicit hook sourcing is a conda-init-style pattern that conflicts with Sol's custom wrapper. Remove it.
- Partition: `--partition=htc --qos=public` with `--gres=gpu:a100:1` works currently, but the future-proof GPU form per Sol docs is `-p general -q public` (or `-p public -q public` once general is removed). HTC is historically a CPU-oriented partition; if you hit scheduling issues, switch to general/public.

### `preprocess.sh`

Logical issues (should revisit):

1. **Step 4b** extracts the first volume of the reverse-b0 with `fslroi ... 0 1`. If your reverse-b0 file is already a single 3D volume, this is a no-op. If it's 4D with multiple b=0s, you discard information you might want to average. Fine as written for a single-volume reverse-b0; worth a comment.
2. **Step 4c** flips the reverse-b0 with `mrconvert -stride 1,-2,3,4`. This is a **geometric flip of the y-axis**, not a phase-encode direction swap. If your reverse-b0 was acquired with the opposite PE direction (which is the point of topup), you should *not* flip it — topup needs the actual acquired geometry and tells the directions apart via `acqparams.txt`. **Verify** this step against the Bruker acquisition: if the reverse-b0 was reconstructed with the same stride as the forward DWI and only differs in PE direction, remove the flip. If the Bruker scanner writes it with a reversed y-stride for some protocol reason, keep it. Either way, add a comment explaining *why*.
3. **Step 6** auto-generates the brain mask from the topup-corrected mean b0, then extrudes the edge slices into the padding region. This is fine as a first pass but brittle — if the first/last original slices happen to contain non-brain tissue, the padding gets filled with junk. The comment already flags this; consider adding an optional `--mask` argument to the sbatch script so the user can supply a hand-edited mask for prototypes that fail QC.
4. **Step 10b/10c** are labeled as Step 10 even though Step 9 is "Step 9". Cosmetic.

Otherwise the skip/rerun logic (`run_step` / `run_step_multi`) is well-designed and lets you recover from partial failures cleanly.

### `submit_cohort.sh`, `check_cohort.sh`

Both are fine. `set -euo pipefail` at this level is safe — these scripts don't activate mamba. Clean, idempotent, and the `.last_jobid` / `job_history.txt` pattern is good practice.

### `manifest.tsv`

Correct format. CRLF line endings need the `dos2unix` step in your `steps.txt` — keep that step, it's not optional.

### `steps.txt`

- The order is right, but step 4 is a bandaid. It assumes the `mrtrix3` env already exists. If it doesn't, `source activate mrtrix3` fails silently and you waste a submission. See Part 2 below for a proper env bootstrap.
- Duplicate "5" label (one as `5a. convert to unix format`, one as `5. Submit`). Renumber.
- There's no verification step between "activate env" and "submit". Add a `which dwidenoise && which eddy_cuda && python -c "import nibabel"` sanity check.

---

## Part 2 — Full setup procedure (one-time)

Run all of this from a **login node** except where marked interactive/sbatch.

### Step 0 — From laptop

VPN on (`sslvpn.asu.edu/2fa`), then either open <https://sol.asu.edu> → Sol Shell Access, or `ssh jfigger@sol.asu.edu`.

### Step 1 — Hygiene check (one-time)

Make sure nothing in `~/.local/lib` will shadow the env's Python:

```bash
ls ~/.local/lib 2>/dev/null
# If you see any python3.X directories from earlier pip --user mishaps:
# mv ~/.local/lib/python3.11 ~/.local/lib/python3.11.bak
```

Check `~/.bashrc` is clean of conda-init blocks:

```bash
grep -n "conda\|mamba" ~/.bashrc
# If you see anything from `conda init`, run:
# remove_conda_from_bashrc && source ~/.bashrc
```

### Step 2 — Get an interactive GPU session for the env build

```bash
interactive -p public -q public -G a100:1 -c 8 --mem=32G -t 0-02:00
```

Wait until the prompt switches from `login0X` to a compute node like `cg012`. If allocation takes too long, fall back to `interactive -p htc -c 8 -t 0-01:00` (no GPU — you just can't run the CUDA verification on the same node).

### Step 3 — Build the `mrtrix3` env in a single mamba call

Check if mrtrix3 already exists by looking in the list outputed from:

```bash
mamba info --envs
```

Single-command creation is critical — it lets Mamba resolve the whole dependency graph in one pass.

```bash
module load mamba/latest

mamba create -y -n mrtrix3 -c mrtrix3 -c conda-forge \
    python=3.11 \
    mrtrix3 \
    nibabel \
    numpy \
    scipy \
    cudatoolkit=11.8

source activate mrtrix3
```

If the env already exists but is missing `cudatoolkit=11.8` (the common case per your current `steps.txt`):

```bash
mamba list cudatoolkit | grep -q "11.8" || mamba install -y -c conda-forge cudatoolkit=11.8
```

### Step 4 — Verify the env

`which python` must show `/home/jfigger/.conda/envs/mrtrix3/bin/python`. Anything else (especially `/packages/apps/mamba/.../bin/python`) means activation didn't take — stop and fix that before going further.

```bash
which python
which dwidenoise mrdegibbs mrconvert maskfilter mrthreshold
python -c "import nibabel, numpy; print('imports OK')"

# Load FSL on top of the active env (same order as the sbatch will)
module load fsl/6.0.7
which eddy_cuda topup fslroi fslmerge fslval
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
eddy_cuda --help 2>&1 | head -5   # should print eddy usage, not a CUDA error
```

If `eddy_cuda --help` complains about CUDA libraries, `cudatoolkit=11.8` isn't correctly exposed — re-check `LD_LIBRARY_PATH`.

Release the interactive session: `exit`.

---

## Part 3 — Per-cohort run procedure

### Step 1 — Make a new cohort directory

```bash
mkdir -p /scratch/jfigger/dwi_cohorts/2026-04-10_mixed_deltas
cd       /scratch/jfigger/dwi_cohorts/2026-04-10_mixed_deltas
```

### Step 2 — Create `manifest.tsv`

One tab-separated row per subject. Columns:

```
name  dwi  reverseb0  bval  bvec  acqparams  topup_cnf  outdir  [pad]  [nvoxhp]
```

Use absolute paths. Lines starting with `#` and blank lines are skipped.

```bash
nano manifest.tsv
```

### Step 3 — Sanity-check manifest

```bash
grep -cEv '^\s*(#|$)' manifest.tsv          # subject count
column -t -s $'\t' manifest.tsv | less -S    # column alignment
```

### Step 4 — Convert line endings (only if files came from Windows/a Confluence export)

```bash
dos2unix manifest.tsv
# scripts are one-time — only re-run if you edit them on Windows
dos2unix /scratch/jfigger/dwi_cohorts/scripts/submit_cohort.sh
dos2unix /scratch/jfigger/dwi_cohorts/scripts/check_cohort.sh
dos2unix /scratch/jfigger/dwi_cohorts/scripts/run_cohort.sbatch
```

### Step 5 — Test with one subject first

This catches environment/activation issues without burning 7 GPU-hours.

```bash
sbatch --array=1 /scratch/jfigger/dwi_cohorts/scripts/run_cohort.sbatch
squeue --me
```

Once it transitions to RUNNING, tail its output:

```bash
tail -f logs/dwi_<JOBID>_1.out
```

The first ~10 lines should include:

```
Python:    /home/jfigger/.conda/envs/mrtrix3/bin/python
eddy_cuda: /packages/apps/fsl/6.0.7/bin/eddy_cuda
```

If `Python:` shows `/packages/apps/mamba/...`, **stop and debug**. The fix is almost always `--export=NONE` missing or `set -u` still present.

Once `[DONE] Step 1: Denoise` appears, the env is working — you can either let shard 1 finish or `scancel <JOBID>` before launching the full cohort.

### Step 6 — Submit the full cohort

```bash
bash /scratch/jfigger/dwi_cohorts/scripts/submit_cohort.sh
```

`submit_cohort.sh` counts usable rows and submits an array of that size, logs the job ID to `.last_jobid` and to `~/job_history.log`.

### Step 7 — Monitor

```bash
squeue --me
tail -f logs/dwi_<JOBID>_1.out              # any specific subject
bash /scratch/jfigger/dwi_cohorts/scripts/check_cohort.sh
```

### Step 8 — Post-mortem and QC

```bash
seff <JOBID>_<N>                           # resource usage per task
ls <outdir>/eddy_corrected.nii.gz          # final output exists?
```

For each subject, visually inspect in `mrview` or `fsleyes`:

- `eddy_corrected.nii.gz` — should be geometrically aligned to the original space
- `field_hz_cropped.nii.gz` — susceptibility field, should show B0 inhomogeneity smoothly varying across the brain
- `mask_cropped.nii.gz` — does it cover the brain without leaking into background?
- `eddy_corrected_padded.eddy_outlier_report` — any volumes flagged as outliers?
- `eddy_corrected_padded.eddy_movement_rms` — reasonable rigid-body motion (typically <0.5 mm for mouse)

If any mask is bad, replace `mask_padded.nii.gz` in that subject's output directory with a hand-edited version and re-run `preprocess.sh` directly (it will skip completed steps).

---

## Part 4 — Files to update

- **`run_cohort.sbatch`** — replace with the corrected version (attached). Critical.
- **`steps.txt`** — either replace with this document or renumber + add the env-bootstrap and env-verification steps.
- **`preprocess.sh`** — no changes required for correctness, but add a comment to Step 4c explaining the `-stride 1,-2,3,4` flip (why it's there for your Bruker protocol, or remove it if the reverse-b0 doesn't actually need flipping).
- **`submit_cohort.sh`** / **`check_cohort.sh`** — no changes.
