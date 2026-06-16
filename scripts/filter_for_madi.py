#!/usr/bin/env python3
"""
Filter an eddy-corrected DWI to match a MADI library's b-value set.

Keeps b=0 volumes plus any volume whose b-value is within TOL of one of
the canonical MADI targets, snaps those bvals to the exact target, and
drops everything else.  Bvecs are taken from eddy's rotated output
(post-motion-correction).
"""
import numpy as np
import nibabel as nib

# ---------------- config ----------------
TARGETS = np.array([500, 1000, 1500, 2000, 2500])  # MADI library shells
TOL     = 25      # ± s/mm² window that snaps to TARGETS
B0_MAX  = 50      # anything below this is b=0

DWI_IN  = "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/059/eddy_unwarped_images.nii.gz"
BVAL_IN = "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/059/dwidata_filtered.bval"
BVEC_IN = "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/059/eddy_unwarped_images.eddy_rotated_bvecs"   # USE POST-EDDY BVECS

DWI_OUT  = "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/059/dwi_madi.nii.gz"
BVAL_OUT = "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/059/dwi_madi.bval"
BVEC_OUT = "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/059/dwi_madi.bvec"

# ---------------- load ----------------
bvals = np.loadtxt(BVAL_IN).ravel()
bvecs = np.loadtxt(BVEC_IN)        # FSL convention: shape (3, N)
img   = nib.load(DWI_IN)
data  = img.get_fdata()
N     = len(bvals)
assert bvecs.shape == (3, N), f"bvec shape {bvecs.shape} mismatched with N={N}"
assert data.shape[-1] == N,   f"image has {data.shape[-1]} vols, bvals has {N}"

# ---------------- classify ----------------
keep      = np.zeros(N, dtype=bool)
new_bvals = np.zeros(N, dtype=int)
for i, b in enumerate(bvals):
    if b < B0_MAX:
        keep[i] = True
        new_bvals[i] = 0
        continue
    j = int(np.argmin(np.abs(TARGETS - b)))
    if abs(b - TARGETS[j]) <= TOL:
        keep[i] = True
        new_bvals[i] = TARGETS[j]

# ---------------- report ----------------
print(f"Input:  {N} volumes")
print(f"Kept:   {keep.sum()}")
print(f"Drop:   {(~keep).sum()}\n")
print(f"{'b':>5} {'n':>4}  raw bval range")
for t in [0, *TARGETS.tolist()]:
    sel = (new_bvals == t) & keep
    if sel.any():
        print(f"{t:>5} {sel.sum():>4}  [{int(bvals[sel].min())}, {int(bvals[sel].max())}]")
if (~keep).any():
    dropped = sorted(set(int(b) for b in bvals[~keep]))
    print(f"\nDropped b-values: {dropped}")

# ---------------- save ----------------
idx      = np.where(keep)[0]
new_data = data[..., idx].astype(np.float32)

nib.save(nib.Nifti1Image(new_data, img.affine, img.header), DWI_OUT)
np.savetxt(BVAL_OUT, new_bvals[idx][None, :], fmt="%d")   # 1×N row
np.savetxt(BVEC_OUT, bvecs[:, idx], fmt="%.6f")            # 3×N

print(f"\nWrote:")
print(f"  {DWI_OUT}   shape={new_data.shape}")
print(f"  {BVAL_OUT}")
print(f"  {BVEC_OUT}")
