"""Generate a tiny synthetic DWI + bvals from the human library for
end-to-end CLI testing of fit_data.py (Δ=50 ms, b=500..2500)."""
import os
import numpy as np
import nibabel as nib
from madi.library import load_library, load_library_meta, _build_candidate_lib_matrix

rng = np.random.default_rng(1)
LIB = "data/libraries/madi_dense_human.npz"
meta = load_library_meta(LIB)
lib = load_library(LIB)
deltas = meta["deltas"]; bvals = meta["b_values"]; n_b = meta["n_b"]
fit_pairs = [(float(deltas[0]), float(b)) for b in sorted(bvals)]

lib_mat, kios, rhos, Vs = _build_candidate_lib_matrix(
    lib, None, deltas, n_b, 0.5, 0.95, None,
    fit_pairs=fit_pairs, lib_b_values=bvals)

# 6x6x3 volume, assign each voxel a random library ratio vector (S/S0).
H, W, Z = 6, 6, 3
n_vox = H * W * Z
sel = rng.choice(lib_mat.shape[0], size=n_vox, replace=True)
ratios = lib_mat[sel]                       # (n_vox, 5)

# One b=0 + 4 directions per shell (5 shells) = 21 volumes.
n_dir = 4
S0 = 1000.0
bvals_row = [0.0]
vols = [np.full((H, W, Z), S0)]             # b=0
for bi, b in enumerate(sorted(bvals)):
    for _ in range(n_dir):
        v = (S0 * ratios[:, bi]).reshape(H, W, Z)
        v = v + 5.0 * rng.standard_normal(v.shape)   # mild noise
        vols.append(v)
        bvals_row.append(float(b))
data = np.stack(vols, axis=-1).astype(np.float32)

os.makedirs("data/_synth", exist_ok=True)
nib.save(nib.Nifti1Image(data, np.eye(4)), "data/_synth/dwi50.nii.gz")
mask = np.ones((H, W, Z), dtype=np.uint8)
nib.save(nib.Nifti1Image(mask, np.eye(4)), "data/_synth/mask.nii.gz")
np.savetxt("data/_synth/dwi50.bval", np.array(bvals_row)[None, :], fmt="%g")
print(f"wrote data/_synth/  ({data.shape}, {len(bvals_row)} vols, {n_vox} voxels)")
