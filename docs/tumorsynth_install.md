# Installing `mri_TumorSynth` and segmenting the edema cohort

`mri_TumorSynth` (Wu & Prados, UCL) is an nnU-Net (v1) CNN that segments healthy
brain tissue **and** brain tumor from MR scans, in one pass. We use it to derive
tumor / tissue ROIs for the 10 edema subjects so the MADI parameter maps
(`kio`, `rho`, `V`) can be summarized per ROI.

Wiki: https://surfer.nmr.mgh.harvard.edu/fswiki/TumorSynth

> **Architecture note.** This machine is `x86_64` with an NVIDIA RTX 2060 — both
> requirements are met (`mri_TumorSynth` is x86-only; GPU gives ~10 s/scan vs.
> ~3 min on CPU).

---

## 0. What you end up with

Two things must be on `PATH` / installed before the segmentation script runs:

1. **A conda env `nnUNet_v1.7`** holding the exact python + pytorch + nnU-Net v1
   the models were trained with, plus the two downloaded model weight sets
   (`Task002_Tumor`, `Task003_InnerTumor`) placed in nnU-Net's results folder.
2. **FreeSurfer ≥ 7.4.1** for the required preprocessing (skull-strip +
   registration to the SRI-24 template) and for the `mri_tumorsynth` wrapper
   script itself.

---

## 1. Prerequisites

### 1a. conda
Already installed here at `/home/jaden/miniconda3`. Its activation script is:

```bash
CONDA_SH=/home/jaden/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
```

(The wiki tells you to find this with `cat ~/.bashrc | grep conda.sh`.)

### 1b. FreeSurfer ≥ 7.4.1
Needed for `mri_synthstrip` (skull-strip), `mri_robust_register` /
`mri_easyreg` (SRI-24 registration), the SRI-24 template itself, and the
`mri_tumorsynth` launcher. Install from
https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall, then:

```bash
export FREESURFER_HOME=/usr/local/freesurfer/7.4.1     # adjust to your install
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
```

Verify: `mri_synthstrip --help` and `mri_robust_register --help` both print usage.

### 1c. Download the two model weight sets
Links are on the wiki ("Key links"):

- **Whole tumor + healthy tissue:** `TumourSynth_v1.0.zip`
- **Inner tumor substructures:** `Task003_InnerTumor.zip`

```bash
cd ~/Downloads
unzip TumourSynth_v1.0.zip     &&  mv TumourSynth_v1.0 Task002_Tumor
unzip Task003_InnerTumor.zip   # already named Task003_InnerTumor
```

Note both directory paths — the install script needs `Task002_Tumor`, and
`Task003_InnerTumor` gets copied into the results folder afterwards.

---

## 2. Build the conda env with `create_nnUNet_v1.7_env.sh`

Download `create_nnUNet_v1.7_env.sh` (wiki "Key links"), then:

```bash
NNUNET_ROOT=$HOME/nnUNet                 # where nnU-Net + data live

./create_nnUNet_v1.7_env.sh \
    -e /home/jaden/miniconda3/etc/profile.d/conda.sh \  # conda.sh
    -m ~/Downloads/Task002_Tumor \                      # unpacked whole-tumor model
    -n nnUNet_v1.7 \                                    # conda env name
    -d "$NNUNET_ROOT" \                                 # nnU-Net install root
    -c                                                 # install CUDA (we have a GPU)
```

The script: creates the `nnUNet_v1.7` conda env (tested python/pytorch), clones &
installs nnU-Net v1, builds the data/results dir structure, drops
`Task002_Tumor` into the results folder, and writes `nnUNet_v1.7_path.sh` (the
env vars nnU-Net needs) into `$PWD`.

### 2a. Add the second model (inner tumor)
Copy `Task003_InnerTumor` next to `Task002_Tumor` in the results tree:

```bash
cp -r ~/Downloads/Task003_InnerTumor \
      "$NNUNET_ROOT"/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres/
```

(Equivalently, re-run the install script with `-m ~/Downloads/Task003_InnerTumor`.)

### 2b. Make the nnU-Net env vars auto-load on activate
```bash
source /home/jaden/miniconda3/etc/profile.d/conda.sh
conda activate nnUNet_v1.7
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cp nnUNet_v1.7_path.sh "$CONDA_PREFIX/etc/conda/activate.d/"
```

Now activating `nnUNet_v1.7` sets `NNUNET_ENV_DIR`, `nnUNet_raw_data_base`,
`nnUNet_preprocessed`, and `RESULTS_FOLDER` for you.

### 2c. Install the `mri_tumorsynth` launcher
Download the main script from the GitHub repo linked on the wiki
("mri_TumorSynth main script"), make it executable, and put it on `PATH`
(e.g. into `$FREESURFER_HOME/bin` or `~/.local/bin`):

```bash
chmod +x mri_tumorsynth
install -m 755 mri_tumorsynth ~/.local/bin/     # ensure ~/.local/bin is on PATH
```

---

## 3. Known-issue fixes (apply if you hit them)

All are from the wiki FAQ, run inside the activated env:

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

The wiki is explicit: `--wholetumor` inputs must be **skull-stripped** *and*
**registered to the SRI-24 template**. Our raw anatomicals in
`data/edema/sub-XXX/anat/` are neither. Per subject we therefore:

1. **skull-strip** the primary sequence with `mri_synthstrip`;
2. **register** the stripped volume to SRI-24 with `mri_robust_register`
   (`--resample` writes the moved volume so non-FreeSurfer viewers stay aligned).

The recommended primary sequence is **T1CE** (`sub-XXX_ce-Gd_T1w.nii.gz`). T2 and
FLAIR are added as extra comma-separated inputs for accuracy, so they are
stripped + registered the same way.

The SRI-24 template ships with FreeSurfer; the script points at
`$FREESURFER_HOME/average` / your local SRI-24 file — set `SRI24_TEMPLATE` to it.

`scripts/segment_tumorsynth.sh` does all of steps 4–5 for the whole cohort.

---

## 5. Running the segmentation (what the script does per subject)

```bash
# whole tumor + healthy tissue (multi-sequence, GPU):
mri_tumorsynth \
  --i sub-XXX_ce-Gd_T1w_sri24.nii.gz,sub-XXX_T2w_sri24.nii.gz,sub-XXX_FLAIR_sri24.nii.gz \
  --o sub-XXX_desc-wholetumor_dseg.nii.gz \
  --wholetumor

# isolate the tumor ROI from the whole-tumor label (label 18) and multiply by T1CE:
mri_binarize --i sub-XXX_desc-wholetumor_dseg.nii.gz --match 18 \
             --o sub-XXX_desc-tumormask.nii.gz
fslmaths sub-XXX_ce-Gd_T1w_sri24.nii.gz -mul sub-XXX_desc-tumormask.nii.gz \
         sub-XXX_desc-tumorroi.nii.gz

# inner tumor substructures (ET / NET / necrosis):
mri_tumorsynth \
  --i sub-XXX_desc-tumorroi.nii.gz \
  --o sub-XXX_desc-innertumor_dseg.nii.gz \
  --innertumor
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

`--innertumor` (BraTS): `1` Non-enhancing, `2` Enhancing, `3` Necrosis.

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
