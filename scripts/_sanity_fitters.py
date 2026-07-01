"""Ad-hoc sanity checks for the new MADI fitters (not a pytest suite).

Uses a ~150-entry random subset of the library so the per-voxel AMICO NNLS
stays fast; the mathematical properties being checked are independent of the
candidate-set size.
"""
import numpy as np
from madi.library import (load_library, load_library_meta,
                          match_voxels_batch, _build_candidate_lib_matrix)
from madi.fitters import bayes_fit, amico_fit

rng = np.random.default_rng(0)
LIB = "data/libraries/madi_dense.npz"
full = load_library(LIB)
meta = load_library_meta(LIB)
deltas = meta["deltas"]; bvals = meta["b_values"]; n_b = meta["n_b"]

# Subsample library for speed (keep AMICO's per-voxel NNLS small).
sub = rng.choice(len(full), size=150, replace=False)
lib = [full[i] for i in sub]

fit_pairs = [(float(d), float(b)) for d in sorted(deltas) for b in sorted(bvals)]
kw = dict(fit_pairs=fit_pairs, lib_deltas=deltas, lib_b_values=bvals, n_b=n_b,
          vi_min=0.0, vi_max=0.95, rho_max=None)

lib_mat, kios, rhos, Vs = _build_candidate_lib_matrix(
    lib, None, deltas, n_b, 0.0, 0.95, None,
    fit_pairs=fit_pairs, lib_b_values=bvals)
n_cand = lib_mat.shape[0]
print(f"candidates={n_cand}, features={lib_mat.shape[1]}")

gt_idx = rng.choice(n_cand, size=40, replace=False)
measured = lib_mat[gt_idx] + 0.01 * rng.standard_normal((len(gt_idx), lib_mat.shape[1]))

kio_map, rho_map, V_map, res_map = match_voxels_batch(measured, lib, **kw)

# ===== Check 2: Bayes -> MAP as sigma_m -> 0 =====
b = bayes_fit(measured, lib, sigma_m=1e-6, **kw)
d_kio = np.max(np.abs(b["kio_mean"] - kio_map))
d_rho = np.max(np.abs(b["rho_mean"] - rho_map))
d_V   = np.max(np.abs(b["V_mean"]   - V_map))
print("\n[Check 2] Bayes(sigma_m=1e-6) vs MAP  max|Δ|:")
print(f"   kio={d_kio:.3e}  rho={d_rho:.3e}  V={d_V:.3e}")
print("   PASS" if (d_kio < 1e-3 and d_rho < 1e-1 and d_V < 1e-4) else "   FAIL")

# ===== Check 3: AMICO(l1=l2=0) sparse & close to MAP =====
a = amico_fit(measured, lib, lambda1=0.0, lambda2=0.0, verbose=False, **kw)
print("\n[Check 3] AMICO(l1=l2=0):")
print(f"   median n_eff = {np.median(a['n_eff']):.2f} atoms (sparse ~1-3 expected)")
print(f"   kio_mean vs MAP median|Δ| = {np.median(np.abs(a['kio_mean']-kio_map)):.2f}")
print(f"   V_mean   vs MAP median|Δ| = {np.median(np.abs(a['V_mean']-V_map)):.3f}")
print("   PASS" if np.median(a["n_eff"]) < 4 else "   FAIL")

# ===== Check 4: regularization smooths a spatial field =====
def grad_mag(field2d):
    gy, gx = np.gradient(field2d)
    return np.median(np.sqrt(gx**2 + gy**2))

H = W = 16
yy, xx = np.mgrid[0:H, 0:W]
Vtarget = (0.5 + 4.0 * (xx / (W - 1))).ravel()      # smooth ramp
# nearest candidate (by V) to each target
sel = np.argmin(np.abs(Vs[None, :] - Vtarget[:, None]), axis=1)
clean = lib_mat[sel]
noisy = clean + 0.03 * rng.standard_normal(clean.shape)

km, rm, Vm, _ = match_voxels_batch(noisy, lib, **kw)
bb = bayes_fit(noisy, lib, sigma_m=0.05, **kw)
aa = amico_fit(noisy, lib, lambda1=0.0, lambda2=0.5, verbose=False, **kw)
g_map = grad_mag(Vm.reshape(H, W))
g_bay = grad_mag(bb["V_mean"].reshape(H, W))
g_ami = grad_mag(aa["V_mean"].reshape(H, W))
print("\n[Check 4] median spatial |grad V|  (lower = smoother):")
print(f"   MAP={g_map:.4f}   Bayes(sig=0.05)={g_bay:.4f}   AMICO(l2=0.5)={g_ami:.4f}")
print("   PASS" if (g_bay < g_map or g_ami < g_map) else "   FAIL")

# ===== Check 5: uncertainty larger for ambiguous (partial-volume) voxels =====
# WM-like: a single distinctive library entry + tiny noise -> one entry fits,
# posterior is sharp.  CSF/partial-volume-like: a blend of several distant
# entries -> matched by many entries but none exactly -> broad posterior.
wm = lib_mat[gt_idx[:20]] + 0.01 * rng.standard_normal((20, lib_mat.shape[1]))
csf = np.empty((20, lib_mat.shape[1]))
for j in range(20):
    mix = rng.choice(n_cand, size=5, replace=False)
    csf[j] = lib_mat[mix].mean(axis=0)
csf += 0.01 * rng.standard_normal(csf.shape)
bw = bayes_fit(wm, lib, sigma_m=0.05, **kw)
bc = bayes_fit(csf, lib, sigma_m=0.05, **kw)
print("\n[Check 5] posterior std (WM-like vs partial-volume-like):")
print(f"   V_std   WM={np.median(bw['V_std']):.3f}  CSF={np.median(bc['V_std']):.3f}")
print(f"   kio_std WM={np.median(bw['kio_std']):.2f}  CSF={np.median(bc['kio_std']):.2f}")
print("   PASS" if np.median(bc["V_std"]) > np.median(bw["V_std"]) else "   FAIL")

# ===== free-S0 smoke test for both methods =====
raw = 1000.0 * measured
bs = bayes_fit(measured, lib, sigma_m=0.02, fit_s0=True, raw_signal=raw, **kw)
as_ = amico_fit(measured, lib, lambda1=0.0, lambda2=0.01, fit_s0=True,
                raw_signal=raw, verbose=False, **kw)
print("\n[free-S0 smoke] bayes s0_fit median={:.1f}, amico s0_fit median={:.1f}"
      .format(np.median(bs["s0_fit"]), np.median(as_["s0_fit"])))
print("   (expected ~1000)")
print("\nALL CHECKS DONE")
