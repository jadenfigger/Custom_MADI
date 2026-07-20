# Installing `mri_TumorSynth` and segmenting the edema cohort

`mri_TumorSynth` (Wu & Prados, UCL) is an nnU-Net (v1) CNN that segments healthy
brain tissue **and** brain tumor from MR scans, in one pass. We use it to derive
tumor / tissue ROIs for the 10 edema subjects so the MADI parameter maps
(`kio`, `rho`, `V`) can be summarized per ROI.

Wiki: https://surfer.nmr.mgh.harvard.edu/fswiki/TumorSynth

> **Architecture note.** This machine is `x86_64` with an NVIDIA RTX 2060 — both
> requirements are met (`mri_TumorSynth` is x86-only; GPU gives ~10 s/scan vs.
> ~3 min on CPU). Verified end-to-end on 2026-07-06 (`--wholetumor` on the
> SRI-24 template itself, ~85 s for the 5-fold ensemble on GPU).

---

## 0. What you end up with

1. **A conda env `nnUNet_v1.7`** with python 3.8.13 + pytorch 2.1.2 (CUDA 11.8)
   + the `nnunet` pip package, plus the two model weight sets (`Task002_Tumor`,
   `Task003_InnerTumor`) placed in nnU-Net's results folder.
2. **FreeSurfer 8.2.0** — already ships `mri_synthstrip`, `mri_robust_register`,
   `mri_binarize`, **and `mri_tumorsynth` itself** at
   `/usr/local/freesurfer/8.2.0/bin/`. No separate launcher download needed.
3. **A patched copy of `mri_tumorsynth`** on `PATH` ahead of FreeSurfer's own
   (see §2d — the vendor script has two real bugs that break every run on
   Linux).
4. **The SRI-24 template**, downloaded separately (§1c) — it does **not** ship
   with FreeSurfer, despite what the wiki implies.

---

## 1. Prerequisites

### 1a. conda
Already installed here at `/home/jaden/miniconda3`:
```bash
CONDA_SH=/home/jaden/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
```

### 1b. FreeSurfer
Already installed at `/usr/local/freesurfer/8.2.0` (satisfies the ≥7.4.1
requirement) and confirmed to include `mri_synthstrip`, `mri_robust_register`,
`mri_binarize`, `mri_tumorsynth`:
```bash
export FREESURFER_HOME=/usr/local/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
```

### 1c. The SRI-24 template (not bundled with FreeSurfer)
Confirmed by searching the entire FreeSurfer 8.2.0 tree — there is no SRI24
file anywhere in it. The real source is NITRC's SRI24 project
(https://www.nitrc.org/projects/sri24/, CC BY-SA 3.0), specifically the
"anatomy" NIfTI package:

```bash
mkdir -p ~/Downloads/sri24 && cd ~/Downloads/sri24
curl -L -o sri24_anatomy_nifti.zip \
  "https://www.nitrc.org/frs/download.php/4499/sri24_anatomy_nifti.zip"
unzip -o sri24_anatomy_nifti.zip
```

This unpacks `spgr.nii` (T1-weighted), `erly.nii`, `late.nii`, and a `LICENSE`.
`spgr.nii` is the T1 registration target `--wholetumor` expects
(confirmed 240×240×155 @ 1mm isotropic — matches the published SRI24 spec):
```bash
mkdir -p ~/sri24
gzip -c ~/Downloads/sri24/sri24/spgr.nii > ~/sri24/T1.nii.gz
cp ~/Downloads/sri24/sri24/LICENSE ~/sri24/LICENSE
```

### 1d. Download the two model weight sets
Links are on the wiki ("Key links"):

- **Whole tumor + healthy tissue:** `TumourSynth_v1.0.zip`
- **Inner tumor substructures:** `Task003_InnerTumor.zip`

Unzip both. Note: NITRC/OneDrive zips sometimes extract flat (without the
wrapping `TaskXXX_Name` folder) — check for a `nnUNetTrainerV2__*` folder with
real `fold_0..4/model_final_checkpoint.model` files and training logs inside;
that's the genuine model content regardless of what directory it lands in.

---

## 2. Build the conda env with `create_nnUNet_v1.7_env.sh`

> **Naming gotcha (found 2026-07-06):** `mri_tumorsynth` computes its
> `RESULTS_FOLDER` by string-splitting the model file path at the **first**
> literal `nnUNet/` substring it finds. If your install root itself is named
> `nnUNet` (as the wiki's own example suggests, e.g. `~/nnUNet`), the path
> `~/nnUNet/nnUNet_v1.7/.../nnUNet/3d_fullres/...` contains `nnUNet/` twice,
> and the script splits at the wrong (outer) one — `nnUNet_predict` then looks
> for the model in the wrong place and fails with
> `AssertionError: model output folder not found`.
>
> **Fix: never name the install root `nnUNet`.** Use something like
> `~/madi_tumorsynth_models` instead.

```bash
NNUNET_ROOT=$HOME/madi_tumorsynth_models   # NOT "nnUNet" — see gotcha above

./create_nnUNet_v1.7_env.sh \
    -e /home/jaden/miniconda3/etc/profile.d/conda.sh \
    -m ~/Downloads/Task002_Tumor \
    -n nnUNet_v1.7 \
    -d "$NNUNET_ROOT" \
    -c
```

If `conda create` fails with "unrecognized arguments" on conda ≥25 — package
specs must be contiguous, before any `-c channel` flags. (This repo's local
copy of the vendor script already has this fixed.)

The `pip install .` step for the `nnunet` package itself is memory-hungry
(pulls in scikit-image, SimpleITK, batchgenerators alongside torch already
loaded). **On a memory-constrained WSL2 VM this can OOM and crash the entire
WSL instance**, not just the pip process. If that happens: raise the WSL
memory ceiling in `%UserProfile%\.wslconfig` on the Windows side, e.g.
```ini
[wsl2]
memory=12GB
processors=8
swap=8GB
```
then `wsl --shutdown` from Windows and reopen — this fixed it here (default
cap was ~7.9GB on a 16GB host).

### 2a. Add the second model (inner tumor)
```bash
mkdir -p "$NNUNET_ROOT"/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres/Task003_InnerTumor
cp -r ~/Downloads/<wherever the nnUNetTrainerV2__* folder actually landed> \
      "$NNUNET_ROOT"/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres/Task003_InnerTumor/
```
Verify both models: each should have
`nnUNetTrainerV2__nnUNetPlansv2.1/fold_{0..4}/model_final_checkpoint.model`
directly under `Task002_Tumor/` and `Task003_InnerTumor/`.

### 2b. Make the nnU-Net env vars auto-load on activate
```bash
source /home/jaden/miniconda3/etc/profile.d/conda.sh
conda activate nnUNet_v1.7
cat > nnUNet_v1.7_path.sh <<EOF
#!/usr/bin/env bash
export nnUNet_raw_data_base=$NNUNET_ROOT/nnUNet_v1.7/nnUNet_raw_data_base
export nnUNet_preprocessed=$NNUNET_ROOT/nnUNet_v1.7/nnUNet_preprocessed
export RESULTS_FOLDER=$NNUNET_ROOT/nnUNet_v1.7/nnUNet_trained_models
EOF
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cp nnUNet_v1.7_path.sh "$CONDA_PREFIX/etc/conda/activate.d/"
```
(Note: `mri_tumorsynth` actually recomputes these three vars itself from
whatever you pass to `--nnUnet`, so this step mostly matters if you ever call
`nnUNet_predict` directly rather than through the wrapper.)

### 2c. `mri_tumorsynth` already ships with FreeSurfer — but is buggy on Linux
No separate launcher download needed (`/usr/local/freesurfer/8.2.0/bin/mri_tumorsynth`
exists already). However it has two bugs that make every invocation fail on
this machine, both found via `bash -x` tracing an actual run:

1. `mktemp -d -t TumorSynth_` — GNU `mktemp` (Linux) requires explicit `X`s in
   a `-t` template; BSD/macOS `mktemp` doesn't. Fails immediately with
   `mktemp: too few X's in template`.
2. `((i++))` at the end of the per-input loop — when `i` is `0` (first/only
   input), bash's `((...))` arithmetic evaluates to the *pre*-increment value,
   which is `0` (falsy), so with `set -e` at the top of the script this
   silently aborts right after inference finishes — no error message, just no
   output file and no "Fusing the results" line.

Fix: copy the script, patch both lines, and put it on `PATH` ahead of
FreeSurfer's copy (don't edit FreeSurfer's own file — it's root-owned):
```bash
mkdir -p ~/.local/bin
cp /usr/local/freesurfer/8.2.0/bin/mri_tumorsynth ~/.local/bin/mri_tumorsynth
sed -i 's/mktemp -d -t TumorSynth_/mktemp -d -t TumorSynth_XXXXXX/' ~/.local/bin/mri_tumorsynth
sed -i 's/((i++))/i=$((i+1))/' ~/.local/bin/mri_tumorsynth
chmod +x ~/.local/bin/mri_tumorsynth
export PATH="$HOME/.local/bin:$PATH"   # put this ahead of $FREESURFER_HOME/bin
```
`scripts/segment_tumorsynth.sh` does this `PATH` prepend automatically.

---

## 3. Known-issue fixes (apply if you hit them)

```bash
source /home/jaden/miniconda3/etc/profile.d/conda.sh
conda activate nnUNet_v1.7

# conda ToS blocks the install:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# inference crash "undefined symbol: iJIT_NotifyEvent":
conda install "mkl<2025.0.0" "numpy<2"
conda clean -a -y
```

---

## 4. Preprocessing each subject (required before `--wholetumor`)

`--wholetumor` inputs must be **skull-stripped** *and* **registered to the
SRI-24 template** (`~/sri24/T1.nii.gz`, §1c). Per subject:

1. **skull-strip** the primary sequence with `mri_synthstrip`;
2. **register** the stripped volume to SRI-24 with `mri_robust_register`
   (`--resample` writes the moved volume so non-FreeSurfer viewers stay aligned).

Primary sequence is **T1CE** (`sub-XXX_ce-Gd_T1w.nii.gz`); T2 and FLAIR are
added as extra comma-separated inputs for accuracy.

`scripts/segment_tumorsynth.sh` does all of steps 4–5 for the whole cohort.

---

## 5. Running the segmentation

**Critical flag:** the wiki's help text says `--nnUNet`, but the actual script
only matches `--nnUnet` (lowercase "net") — bash `case` is case-sensitive, and
there's no environment-variable fallback, so the documented flag silently
fails with "directory to save the nnUNet model files is not set." Always pass
`--nnUnet <root>` in exactly that case, pointing at `$NNUNET_ROOT` (the parent
of `nnUNet_v1.7/`, **not** the `nnUNet_v1.7` subdirectory itself).

```bash
# whole tumor + healthy tissue (multi-sequence, GPU):
mri_tumorsynth \
  --i sub-XXX_ce-Gd_T1w_sri24.nii.gz,sub-XXX_T2w_sri24.nii.gz,sub-XXX_FLAIR_sri24.nii.gz \
  --o sub-XXX_desc-wholetumor_dseg.nii.gz \
  --wholetumor --nnUnet "$NNUNET_ROOT"

# isolate the tumor ROI from the whole-tumor label (label 18) and multiply by T1CE:
mri_binarize --i sub-XXX_desc-wholetumor_dseg.nii.gz --match 18 \
             --o sub-XXX_desc-tumormask.nii.gz
fslmaths sub-XXX_ce-Gd_T1w_sri24.nii.gz -mul sub-XXX_desc-tumormask.nii.gz \
         sub-XXX_desc-tumorroi.nii.gz

# inner tumor substructures (ET / NET / necrosis):
mri_tumorsynth \
  --i sub-XXX_desc-tumorroi.nii.gz \
  --o sub-XXX_desc-innertumor_dseg.nii.gz \
  --innertumor --nnUnet "$NNUNET_ROOT"
```

### Output label maps

`--wholetumor` (FreeSurfer anatomy + tumor):

| label | structure | label | structure |
|------:|-----------|------:|-----------|
| 1 | Cerebral-White-Matter | 10 | Pallidum |
| 2 | Cerebral-Cortex | 11 | 3rd-Ventricle |
| 3 | Lateral-Ventricle | 12 | 4th-Ventricle |
| 4 | Inferior-Lateral-Ventricle | 13 | Brain-Stem |
| 5 | Cerebellum-White-Matter | 14 | Hippocampus |
| 6 | Cerebellum-Cortex | 15 | Amygdala |
| 7 | Thalamus | 16 | Accumbens-Area |
| 8 | Caudate | 17 | Ventral-DC |
| 9 | Putamen | **18** | **Whole tumor** |

`--innertumor` (BraTS-compliant, per `mri_tumorsynth --help` verbatim —
"Outputs BraTS-compliant subclasses: Tumor Core (TC), Non-Enhancing Tumor
(NET), and Edema" — in that order): `1` Tumor Core (TC), `2` Non-Enhancing
Tumor (NET), `3` Edema. (An earlier draft of this doc had this table wrong,
transcribed from memory instead of the actual help text — corrected
2026-07-06 after `scripts/edema_figures/tumorsynth_roi_space.py` produced an
implausible near-zero "edema" region under the wrong mapping.)

Verified 2026-07-06 by running `--wholetumor` directly on the SRI-24 template
itself (a healthy brain): produced all 17 tissue labels, correctly no label 18.

---

## 6. Feeding the ROIs back to the MADI figures

TumorSynth output is in **SRI-24 space**. To use these labels as ROI masks
against the MADI maps (which live in native DWI space), resample each label into
DWI space — mirror what `scripts/edema_figures/roi_space.py` already does for the
edema/contra masks, writing
`derivatives/rois/sub-XXX/sub-XXX_desc-<roi>-dwi_mask.nii.gz`. Once those exist,
`scripts/edema_figures/fig_roi_method_bars.py` auto-discovers them (it globs
every `*-dwi_mask.nii.gz`), so the new tumor/tissue ROIs appear in the bar charts
with no code change.
