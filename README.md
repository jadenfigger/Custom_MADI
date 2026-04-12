# MADI-GPU — Metabolic Activity Diffusion Imaging (GPU-accelerated)

Monte Carlo simulation of water diffusion in contracted Voronoi cell
ensembles with semipermeable membranes, accelerated with Numba CUDA.

Implements the model from Springer et al. (NMR in Biomedicine 2023;36:e4781),
tailored to a **multi-Δ preclinical 9.4T acquisition**:

- δ = 6 ms,  Δ = 15 / 25 / 30 / 40 ms
- b = 0, 1000, 2500, 4000, 6000 s/mm²
- 24 gradient directions per shell


## Setup

```bash
pip install numpy scipy matplotlib numba nibabel
```

**CUDA requirement:** Nvidia GPU + CUDA toolkit.  Numba detects automatically.
Falls back to CPU if no GPU.

```bash
conda install -c conda-forge cudatoolkit numba
```


## Quick start

```bash
# 1. Smoke test
python run_simulation.py --minimal

# 2. Build library
python fit_data.py --build-library --lib-preset default

# 3. Fit data (all four deltas)
python fit_data.py --fit \
    --input 15:dwi15.nii.gz 25:dwi25.nii.gz 30:dwi30.nii.gz 40:dwi40.nii.gz \
    --mask mask.nii.gz

# 3b. Fit data (only two deltas — works fine)
python fit_data.py --fit \
    --input 15:dwi15.nii.gz 25:dwi25.nii.gz \
    --mask mask.nii.gz
```


## Building & extending the library

The library always simulates all 4 Δ values per entry.  At fitting time,
you choose which subset to use.  All `--build-library` modes support
`--append` to skip already-computed entries.

### Mode 1: Preset grid

Full cross-product of a named preset:

```bash
python fit_data.py --build-library --lib-preset default
python fit_data.py --build-library --lib-preset dense
```

### Mode 2: Preset + custom additions

Merge extra values into the preset grid:

```bash
# Add high-rho values (crossed with ALL preset kio and V)
python fit_data.py --build-library --append \
    --custom-rhos 1500000 2000000 3000000

# Add extra values for all three parameters
python fit_data.py --build-library --append \
    --custom-kios 40 60 --custom-rhos 2000000 --custom-Vs 0.2 7.0
```

### Mode 3: Explicit sub-grid

Cross ONLY the values you specify (ignores the preset grid):

```bash
# Only simulate these specific rho values at these specific kio and V
python fit_data.py --build-library --append --explicit \
    --grid-kios 12 25 50 \
    --grid-rhos 1500000 2000000 3000000 \
    --grid-Vs 0.5 1.0 2.0

# If you omit one axis, the preset values are used for that axis:
# This crosses custom rhos with ALL preset kios and Vs
python fit_data.py --build-library --append --explicit \
    --grid-rhos 1500000 2000000
```

### Mode 4: Exact triplets

Simulate specific (kio, rho, V) points:

```bash
python fit_data.py --build-library --append \
    --triplets 12,1500000,0.5  25,2000000,1.0  50,3000000,0.3
```

### Inspect a library

```bash
python fit_data.py --info --library madi_library.npz
```


## Fitting with flexible Δ inputs

The `--input` flag takes `delta:path` pairs.  Use as many or as few as you
have.  The library stores all 4 Δ values; at fitting time, only the columns
matching your Δ values are compared.

```bash
# All four
python fit_data.py --fit \
    --input 15:dwi15.nii.gz 25:dwi25.nii.gz 30:dwi30.nii.gz 40:dwi40.nii.gz \
    --mask mask.nii.gz --out results_4delta/

# Just two
python fit_data.py --fit \
    --input 15:dwi15.nii.gz 25:dwi25.nii.gz \
    --mask mask.nii.gz --out results_2delta/

# Just one (less identifiability, but works)
python fit_data.py --fit \
    --input 25:dwi25.nii.gz \
    --mask mask.nii.gz --out results_1delta/
```

**Legacy syntax still works** (but --input is preferred):
```bash
python fit_data.py --fit --dwi15 dwi15.nii.gz --dwi25 dwi25.nii.gz --mask mask.nii.gz
```


## Project structure

```
madi_gpu/
├── madi/
│   ├── __init__.py
│   ├── config.py          # Constants, acquisition parameters, SimConfig
│   ├── ensemble.py        # Contracted Voronoi + voxelised lookup grid
│   ├── walker_gpu.py      # Numba CUDA random walk kernel + CPU fallback
│   ├── signal.py          # Multi-delta SDE signal computation
│   ├── library.py         # Library builder (append, triplets, sub-grids)
│   └── plotting.py        # Visualisation
├── run_simulation.py      # Reproduce Figure 4 (parameter sensitivity)
├── fit_data.py            # Build library + fit NIfTI data → maps
├── requirements.txt
└── README.md
```


## Parameter ranges

From Springer et al. MADI I & II (NMR in Biomedicine 2023;36:e4781, e4782):

### Paper's full library (14,000+ entries)

| Parameter | Range | Unique values | Step size |
|-----------|-------|---------------|-----------|
| V (cell volume) | 0.01 – 206 pL | 949 | ~0.05 pL below 2 pL, ~0.2 pL above 8 pL |
| ρ (cell density) | 4,400 – 58×10⁶ cells/μL | 186 | Non-uniform |
| k_io (efflux rate) | 0.0 – 130 s⁻¹ | 65 | Non-uniform |
| v_i (volume fraction) | 0.5 – 0.994 | 20 | Entries on ρ·V hyperbolae |

### Paper's brain map results (MADI-II Table 2)

| Region | ρ (10⁵ cells/μL) | V (pL) | k_io (s⁻¹) |
|--------|------------------|--------|-------------|
| Cortical GM | 1.1 | 6.0 | 6.6 |
| Thalamus | 6.2 | 1.2 | 22 |
| Putamen | 2.9 | 2.6 | 17 |
| White matter | 6.9 | 0.91 | 22 |

### Dense tissue (MADI-II Table 1)

| Tissue | ρ (cells/μL) | V (pL) | k_io (s⁻¹) |
|--------|-------------|--------|-------------|
| Colorectal cancer | 950,000 | 0.74 | 6.6 |
| Prostate lesion | 1,200,000 | 0.54 | 39 |
| Lymphocyte beds | up to 6,000,000 | — | — |

### This implementation's presets

| Preset | k_io | ρ (cells/μL) | V (pL) |
|--------|------|-------------|--------|
| small | 4 vals (5–50) | 5 vals (100k–1.2M) | 4 vals (0.5–3.5) |
| default | 9 vals (2–75) | 9 vals (100k–1.5M) | 9 vals (0.3–5.0) |
| dense | 16 vals (2–100) | 13 vals (100k–3M) | 15 vals (0.2–9.0) |


## How it works

### GPU random walk kernel

Each CUDA thread runs one walker through all time steps:
1. **Propose displacement:** Gaussian with σ = √(2·D₀·t_s) per axis
2. **Classify compartment:** Voxel grid lookup → nearest seeds → bisecting-plane test
3. **Permeation test:** If boundary crossed, accept with probability p_p^m
4. **Accumulate encoding moments:** All 4 Δ windows simultaneously (4× efficiency)

### Library matching

Each library entry stores a 16-element signal vector (4 Δ × 4 b-values).
When fitting with fewer Δ values, the matcher extracts the corresponding
subset of columns from each library vector before computing distances.
Matching is nearest-neighbour in signal space, vectorised with NumPy.


## Troubleshooting

### All voxels hit the ρ upper boundary

```bash
python fit_data.py --build-library --append \
    --custom-rhos 1500000 2000000 3000000 5000000
```

### Want to fill in a specific region of parameter space

```bash
python fit_data.py --build-library --append --explicit \
    --grid-kios 8 12 18 25 \
    --grid-rhos 800000 1000000 1200000 1500000 2000000 \
    --grid-Vs 0.3 0.5 0.8 1.0 1.5
```

### Maps look noisy / blocky

Library is too sparse.  Either densify the whole thing:
```bash
python fit_data.py --build-library --lib-preset dense --append
```
Or target the parameter region your tissue occupies.

### Fitting output says "⚠ 40% of voxels hit rho UPPER bound"

The fitting code now warns you automatically.  Extend the grid for that
parameter and re-fit.


## Estimated run times (RTX 2060 Mobile)

| Task | GPU | CPU fallback |
|------|-----|------|
| run_simulation.py --minimal | ~2 min | ~10 min |
| Library (small) | ~15 min | ~3 hr |
| Library (default) | ~1.5 hr | ~24 hr |
| Library (dense) | ~6 hr | ~5 days |
| Fitting (voxel matching) | ~5 sec | ~5 sec |


## Setting up envirement and installing packages on Sol Supercomputer through shell

interactive -p general -q public -G a100:1 -c 8 --mem=32G -t 0-04:00
module load mamba/latest
mamba create -y -n madi python=3.11 && source activate madi
mamba install -y -c conda-forge numpy scipy matplotlib nibabel numba cudatoolkit=11.8
python -c "import numpy, scipy, matplotlib, nibabel, numba; from numba import cuda; print('CUDA available:', cuda.is_availabl ()); print('GPU:', cuda.get_current_device().name if cuda.is_available() else 'none')"

## License

Research/educational use.  The MADI method is described in US Provisional
Patent Application No. 62/482,520 ("Activity MRI").
