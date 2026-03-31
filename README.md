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

**CUDA requirement:** You need an Nvidia GPU with CUDA support and the CUDA
toolkit installed.  Numba will detect it automatically.  If no GPU is found,
the code falls back to a (much slower) CPU implementation.

For your RTX 2060 laptop, install CUDA via:
```bash
conda install -c conda-forge cudatoolkit numba
```


## Quick start

```bash
# 1. Reproduce Figure 4 (smoke test)
python run_simulation.py --minimal       # ~2 min GPU

# 2. Build a simulation library (do once)
python fit_data.py --build-library --lib-preset small    # ~15 min GPU
python fit_data.py --build-library --lib-preset default  # ~1 hr GPU
python fit_data.py --build-library --lib-preset dense    # ~4 hr GPU

# 3. Fit your mouse data
python fit_data.py --fit \
    --dwi15 /path/to/DWI_15ms/eddy_corrected.nii.gz \
    --dwi25 /path/to/DWI_25ms/eddy_corrected.nii.gz \
    --dwi30 /path/to/DWI_30ms/eddy_corrected.nii.gz \
    --dwi40 /path/to/DWI_40ms/eddy_corrected.nii.gz \
    --mask  /path/to/mask.nii.gz \
    --out   /path/to/madi_output/
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
│   ├── library.py         # Library builder + voxel matcher
│   └── plotting.py        # Visualisation
├── run_simulation.py      # Reproduce Figure 4 (parameter sensitivity)
├── fit_data.py            # Build library + fit NIfTI data → maps
├── requirements.txt
└── README.md
```


## How it works

### GPU random walk kernel

Each CUDA thread runs **one walker** through all 50,000 time steps:

1. **Propose displacement:** Gaussian with σ = √(2·D₀·t_s) per axis
2. **Classify compartment:** Voxel grid lookup (precomputed on CPU) → nearest
   two seeds → bisecting-plane test → intracellular or interstitial
3. **Permeation test:** If compartment changed, draw uniform random; accept
   if u < p_p^m (m = number of membranes crossed)
4. **Accumulate encoding moments:** During PFG-1 (0→δ) and each PFG-2
   (Δ_i → Δ_i+δ), accumulate position × dt

**Key optimisation:** All 4 Δ values share the same PFG-1, so we run one
walk and accumulate 4 different PFG-2 windows simultaneously.  This is 4×
more efficient than separate walks.

### Voxelised spatial grid

The Voronoi ensemble is "rasterised" onto a 3D grid where each voxel stores
the indices of the two nearest seeds.  This converts the expensive KD-tree
query (O(log N) per point) into a simple array index (O(1)).

Grid memory: 250³ × 4 bytes × 2 = ~125 MB (fits in 6 GB VRAM).

### Signal computation

For each Δ and b-value, compute the gradient strength G via b = (γGδ)²·tD,
then calculate the phase for each walker and average cos(φ) over walkers
and gradient directions.

### Library matching

The library contains one simulated signal vector per (k_io, ρ, V) triplet.
Each vector has n_deltas × n_shells = 16 values.  Fitting is nearest-neighbor
in this 16-D signal space (vectorised with NumPy, ~seconds for a whole brain).


## Why not disimpy?

[Disimpy](https://github.com/kerkelae/disimpy) is excellent for impermeable
surfaces but does not support **semipermeable** membranes.  MADI requires
water to permeate cell membranes with probability p_p at each encounter —
a fundamentally different boundary condition.  This codebase implements its
own Numba CUDA kernel with the permeation logic from the paper's SI §IV.b.


## Estimated run times (RTX 2060 Mobile)

| Task | GPU | CPU fallback |
|------|-----|------|
| run_simulation.py --minimal | ~2 min | ~10 min |
| run_simulation.py (default) | ~10 min | ~2 hr |
| Library (small, 36 entries) | ~15 min | ~3 hr |
| Library (default, 378 entries) | ~1 hr | ~20 hr |
| Library (dense, 1280 entries) | ~4 hr | ~3 days |
| Fitting (voxel matching) | ~5 sec | ~5 sec |


## MADI biomarkers

| Symbol | Meaning | Mouse brain range |
|--------|---------|------------------|
| k_io | Water efflux rate constant | 5 – 100 s⁻¹ |
| ρ | Cell number density | 100k – 800k cells/μL |
| V | Mean cell volume | 0.3 – 5 pL |


## Adapting to your data

Edit the top of `madi/config.py`:
- `DELTA_SMALL` — your δ (currently 6 ms)
- `DELTAS_BIG` — your Δ values
- `BVALS_UNIQUE` — your b-value shells

Edit the top of `fit_data.py`:
- `SHELLS` — volume index ranges for each b-shell
- `DELTAS_MS` — must match your DWI file ordering


## Multi-delta advantage

The paper simulates a single tD per parameter set.  Your protocol with
4 Δ values gives 4 independent decay curves per voxel, each with different
sensitivity to restriction and exchange.  Joint fitting across all 4 Δ values
provides much better identifiability of k_io, ρ, and V than a single tD.

This is analogous to how NEXI leverages multiple diffusion times — but
MADI returns cytometric parameters (cell density, cell volume) in addition
to the exchange rate.


## License

Research/educational use.  The MADI method is described in US Provisional
Patent Application No. 62/482,520 ("Activity MRI").
