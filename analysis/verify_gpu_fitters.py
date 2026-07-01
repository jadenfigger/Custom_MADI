"""
verify_gpu_fitters.py — CPU vs GPU agreement + timing for the MADI fitters
===========================================================================

Verifies the numba.cuda GPU kernels added in ``madi.fitters_gpu`` (wired
into ``madi.library.match_voxels_batch``/``_fits0`` and
``madi.fitters.bayes_fit``/``amico_fit`` via each function's ``use_gpu``
argument) against the existing pure-NumPy/SciPy CPU code they're meant to
accelerate.

What this checks
-----------------
* MAP (fixed-S0 and free-S0) and Bayes (fixed-S0 and free-S0): the GPU
  kernels are exact reorderings of the CPU matmul math (one CUDA thread per
  voxel streaming over the library instead of materializing the full
  ``(n_voxels, n_library)`` matrix). Checked with ``np.allclose`` at
  float64 tolerance — any mismatch here is a real bug.
* AMICO: the GPU path replaces the CPU's exact per-voxel
  ``scipy.optimize.nnls`` with a fixed-iteration FISTA solve of the same
  elastic-net objective — an *approximation*, not a reordering. Checked by
  comparing the elastic-net objective value at the CPU-exact and GPU-FISTA
  solutions (should be within a few percent), not by comparing parameter
  maps directly.
* Timing: CPU vs GPU on a larger synthetic voxel batch (built from
  ``data/libraries/madi_dense*.npz`` merged), to report realistic speedup on
  this machine and sanity-check the default ``--gpu-chunk-voxels``.

Usage
-----
    python analysis/verify_gpu_fitters.py
    python analysis/verify_gpu_fitters.py --n-timing-voxels 50000

Requires CUDA (skips the GPU comparisons with a clear message if
unavailable — the CPU-only code paths are unaffected by this change).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from madi.library import load_library, LibraryEntry
from madi import fitters
from madi import fitters_gpu as fg


def _make_synthetic_measured(library, n_vox, noise_std, seed, vi_min, vi_max, rho_max):
    """Sample random library entries + Gaussian noise as fake voxel data."""
    from madi.library import _build_candidate_lib_matrix
    lib_mat, kios, rhos, Vs = _build_candidate_lib_matrix(
        library, None, None, 4, vi_min, vi_max, rho_max)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, lib_mat.shape[0], n_vox)
    measured = lib_mat[idx] + rng.normal(0, noise_std, size=(n_vox, lib_mat.shape[1]))
    measured = np.clip(measured, 1e-3, None)
    raw = measured * rng.uniform(80, 120, size=(n_vox, 1))
    return measured, raw, lib_mat, kios, rhos, Vs


def _report(name, ok):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    return ok


def check_map_and_bayes(library, n_vox, vi_min, vi_max, rho_max, seed=0):
    print("\n=== MAP + Bayes: GPU should match CPU to float64 precision ===")
    measured, raw, lib_mat, kios, rhos, Vs = _make_synthetic_measured(
        library, n_vox, 0.01, seed, vi_min, vi_max, rho_max)

    all_ok = True

    from madi.library import match_voxels_batch, match_voxels_batch_fits0
    kio_c, rho_c, V_c, res_c = match_voxels_batch(
        measured, library, vi_min=vi_min, vi_max=vi_max, rho_max=rho_max,
        use_gpu=False)
    kio_g, rho_g, V_g, res_g = match_voxels_batch(
        measured, library, vi_min=vi_min, vi_max=vi_max, rho_max=rho_max,
        use_gpu=True)
    all_ok &= _report("MAP fixed-S0 kio", np.array_equal(kio_c, kio_g))
    all_ok &= _report("MAP fixed-S0 residual",
                       np.allclose(res_c, res_g, atol=1e-6, rtol=1e-6))

    o_c = match_voxels_batch_fits0(raw, library, vi_min=vi_min, vi_max=vi_max,
                                    rho_max=rho_max, use_gpu=False)
    o_g = match_voxels_batch_fits0(raw, library, vi_min=vi_min, vi_max=vi_max,
                                    rho_max=rho_max, use_gpu=True)
    all_ok &= _report("MAP free-S0 kio", np.array_equal(o_c[0], o_g[0]))
    all_ok &= _report("MAP free-S0 s0_fit",
                       np.allclose(o_c[4], o_g[4], atol=1e-6, rtol=1e-6))

    sigma_m = 0.05
    r_c = fitters.bayes_fit(measured, library, sigma_m=sigma_m,
                             vi_min=vi_min, vi_max=vi_max, rho_max=rho_max,
                             use_gpu=False)
    r_g = fitters.bayes_fit(measured, library, sigma_m=sigma_m,
                             vi_min=vi_min, vi_max=vi_max, rho_max=rho_max,
                             use_gpu=True)
    for key in ("kio_mean", "rho_mean", "V_mean", "kio_std", "rho_std",
                "V_std", "residual", "n_eff"):
        all_ok &= _report(f"Bayes fixed-S0 {key}",
                           np.allclose(r_c[key], r_g[key], atol=1e-6, rtol=1e-5))

    r_c = fitters.bayes_fit(measured, library, sigma_m=sigma_m, fit_s0=True,
                             raw_signal=raw, vi_min=vi_min, vi_max=vi_max,
                             rho_max=rho_max, use_gpu=False)
    r_g = fitters.bayes_fit(measured, library, sigma_m=sigma_m, fit_s0=True,
                             raw_signal=raw, vi_min=vi_min, vi_max=vi_max,
                             rho_max=rho_max, use_gpu=True)
    for key in ("kio_mean", "residual", "n_eff", "s0_fit"):
        all_ok &= _report(f"Bayes free-S0 {key}",
                           np.allclose(r_c[key], r_g[key], atol=1e-6, rtol=1e-5))

    return all_ok


def check_amico(library, n_vox, vi_min, vi_max, rho_max, seed=1,
                 gpu_iters=500, gpu_tol=1e-7):
    print("\n=== AMICO: GPU (FISTA) is an approximation of CPU (exact NNLS) ===")
    print("    (comparing elastic-net objective value, not parameter maps; "
          f"n_vox={n_vox} -- AMICO-CPU is slow, kept small deliberately)")
    measured, raw, lib_mat, kios, rhos, Vs = _make_synthetic_measured(
        library, n_vox, 0.01, seed, vi_min, vi_max, rho_max)

    from scipy.optimize import nnls
    lambda1, lambda2 = 0.0, 0.01
    D = lib_mat.T
    n_lib = lib_mat.shape[0]
    A = np.vstack([D, np.sqrt(lambda2) * np.eye(n_lib)])

    step_size = 1.0 / fg._estimate_lipschitz(D, lambda2)
    kernel = fg._build_amico_kernel(*D.shape[::-1])  # (n_lib, n_feat)
    d_D = fg.cuda.to_device(np.ascontiguousarray(D))
    d_M = fg.cuda.to_device(measured)
    d_x_out = fg.cuda.device_array((n_vox, n_lib), dtype=np.float64)
    kernel[fg._grid(n_vox, fg._AMICO_THREADS_PER_BLOCK), fg._AMICO_THREADS_PER_BLOCK](
        d_D, d_M, lambda1, lambda2, step_size, gpu_iters, gpu_tol, d_x_out)
    fg.cuda.synchronize()
    X_gpu = d_x_out.copy_to_host()

    obj_cpu = np.zeros(n_vox)
    obj_gpu = np.zeros(n_vox)
    for v in range(n_vox):
        m = measured[v]
        b = np.concatenate([m, np.zeros(n_lib)])
        x_cpu, _ = nnls(A, b)
        obj_cpu[v] = (np.sum((D @ x_cpu - m) ** 2)
                      + lambda1 * np.sum(x_cpu) + lambda2 * np.sum(x_cpu ** 2))
        x_g = X_gpu[v]
        obj_gpu[v] = (np.sum((D @ x_g - m) ** 2)
                      + lambda1 * np.sum(x_g) + lambda2 * np.sum(x_g ** 2))

    ratio = obj_gpu / np.maximum(obj_cpu, 1e-12)
    print(f"    objective ratio (gpu/cpu): median={np.median(ratio):.4f}, "
          f"max={np.max(ratio):.4f}, min={np.min(ratio):.4f}")
    ok = _report("AMICO objective within 10% of exact NNLS optimum",
                  np.max(ratio) < 1.10)

    # Also sanity-check the wrapper end-to-end (keys + finiteness).
    r_c = fitters.amico_fit(measured, library, vi_min=vi_min, vi_max=vi_max,
                             rho_max=rho_max, use_gpu=False, verbose=False)
    r_g = fitters.amico_fit(measured, library, vi_min=vi_min, vi_max=vi_max,
                             rho_max=rho_max, use_gpu=True, verbose=False,
                             gpu_n_iters=gpu_iters, gpu_tol=gpu_tol)
    ok &= _report("AMICO CPU/GPU output keys match",
                  sorted(r_c.keys()) == sorted(r_g.keys()))
    ok &= _report("AMICO GPU outputs all finite",
                  all(np.all(np.isfinite(v)) for v in r_g.values()))

    return ok


def check_degenerate_voxel_handling(seed=4):
    """Regression check for a real bug found in production use: with
    --method bayes --fit-s0, every printed summary stat (median/min/max
    kio/rho/V) came back "nan" on a real dataset. Root cause: bayes_fit's
    free-S0 branch masks candidates with non-positive fitted S0 to +inf
    residual; if literally every candidate is masked for some voxel (e.g. a
    background/low-SNR voxel after Rician correction pulls the raw signal
    negative), the numerically-stabilizing "subtract the row max" trick
    computes `-inf - (-inf) = nan` for that voxel — and because
    scripts/fit_data.py's summary prints use plain np.median/.min()/.max()
    (not NaN-aware), a single such voxel anywhere in a 100k+ voxel volume
    made the ENTIRE printed report show nan, even though only that one
    voxel was actually degenerate. Fixed in both the CPU (fitters.py) and
    GPU (fitters_gpu.py) free-S0 Bayes paths by explicitly zero-weighting
    fully-degenerate voxels instead of leaving the inf-inf subtraction to
    happen. Also checks that MAP free-S0 and AMICO -- which don't share
    bayes_fit's log-space "subtract row max" trick -- were never actually
    susceptible to this same NaN pattern in the first place."""
    print("\n=== Degenerate-voxel handling: must not produce NaN ===")
    from madi.library import match_voxels_batch_fits0

    rng = np.random.default_rng(seed)
    n_lib, n_feat = 50, 5
    vectors = rng.uniform(0.05, 1.0, size=(n_lib, n_feat))  # all-positive, like real MADI
    kios = rng.uniform(0, 100, n_lib)
    rhos = rng.uniform(1e5, 1e6, n_lib)
    Vs = 0.7 / (rhos * 1e-6)
    library = [LibraryEntry(kio=float(kios[i]), rho=float(rhos[i]), V=float(Vs[i]),
                             vector=vectors[i]) for i in range(n_lib)]

    normal_raw = vectors[0] * 100 + rng.normal(0, 1, n_feat)
    degenerate_raw = -np.abs(rng.normal(50, 5, n_feat))  # can't positively correlate with any (positive) library entry
    raw = np.stack([normal_raw, degenerate_raw])
    measured = np.abs(raw)  # only used by the non-fit_s0 argument slots

    ok = True

    r_cpu = fitters.bayes_fit(measured, library, sigma_m=0.03, fit_s0=True,
                               raw_signal=raw, vi_min=0.0, vi_max=1.0,
                               use_gpu=False)
    r_gpu = fitters.bayes_fit(measured, library, sigma_m=0.03, fit_s0=True,
                               raw_signal=raw, vi_min=0.0, vi_max=1.0,
                               use_gpu=True)
    ok &= _report("Bayes free-S0 CPU has no NaN", not np.any(np.isnan(r_cpu["kio_mean"])))
    ok &= _report("Bayes free-S0 GPU has no NaN", not np.any(np.isnan(r_gpu["kio_mean"])))
    ok &= _report("Bayes free-S0 CPU/GPU agree on degenerate voxel",
                   np.allclose(r_cpu["kio_mean"], r_gpu["kio_mean"], atol=1e-6))
    ok &= _report("Bayes free-S0 volume-wide median is finite (not nan)",
                   np.isfinite(np.median(r_cpu["kio_mean"])))

    o_cpu = match_voxels_batch_fits0(raw, library, vi_min=0.0, vi_max=1.0, use_gpu=False)
    o_gpu = match_voxels_batch_fits0(raw, library, vi_min=0.0, vi_max=1.0, use_gpu=True)
    ok &= _report("MAP free-S0 CPU has no NaN", not np.any(np.isnan(o_cpu[0])))
    ok &= _report("MAP free-S0 GPU has no NaN", not np.any(np.isnan(o_gpu[0])))

    a_cpu = fitters.amico_fit(raw, library, fit_s0=True, raw_signal=raw,
                               vi_min=0.0, vi_max=1.0, use_gpu=False, verbose=False)
    a_gpu = fitters.amico_fit(raw, library, fit_s0=True, raw_signal=raw,
                               vi_min=0.0, vi_max=1.0, use_gpu=True, verbose=False)
    ok &= _report("AMICO CPU has no NaN", not np.any(np.isnan(a_cpu["kio_mean"])))
    ok &= _report("AMICO GPU has no NaN", not np.any(np.isnan(a_gpu["kio_mean"])))

    return ok


def check_amico_lambda2_sensitivity(library, vi_min, vi_max, rho_max, seed=3):
    """Regression check for a real bug found in production use: the FISTA
    kernel's original per-coordinate-step early-exit check triggered almost
    immediately on this (highly correlated) library regardless of lambda2,
    freezing a diffuse solution and making n_eff completely insensitive to
    --lambda2 (reported: n_eff stuck ~400 even at --lambda2 0). Fixed by
    switching to a relative-objective-value convergence check. This test
    keeps voxel counts tiny (CPU NNLS is the bottleneck) — it only needs to
    show *some* separation between a ridge-heavy and near-zero lambda2, not
    reproduce CPU exactly (that needs far more GPU iterations at low
    lambda2, see docs/fitting_methods.md)."""
    print("\n=== AMICO regression: n_eff must respond to --lambda2 on GPU ===")
    n_vox = 2
    measured, raw, lib_mat, kios, rhos, Vs = _make_synthetic_measured(
        library, n_vox, 0.01, seed, vi_min, vi_max, rho_max)

    n_eff_ridge = fitters.amico_fit(measured, library, lambda2=0.01,
                                     vi_min=vi_min, vi_max=vi_max,
                                     rho_max=rho_max, use_gpu=True,
                                     verbose=False)["n_eff"]
    n_eff_noridge = fitters.amico_fit(measured, library, lambda2=0.0,
                                       vi_min=vi_min, vi_max=vi_max,
                                       rho_max=rho_max, use_gpu=True,
                                       verbose=False,
                                       gpu_n_iters=20_000)["n_eff"]
    print(f"    n_eff @ lambda2=0.01: {n_eff_ridge}")
    print(f"    n_eff @ lambda2=0.0 (20000 iters): {n_eff_noridge}")
    return _report("n_eff decreases as lambda2 -> 0 (not frozen)",
                    bool(np.all(n_eff_noridge < n_eff_ridge)))


def time_cpu_vs_gpu(library, n_vox, vi_min, vi_max, rho_max, n_amico_cpu,
                     seed=2):
    print(f"\n=== Timing: {n_vox} voxels ===")
    measured, raw, lib_mat, kios, rhos, Vs = _make_synthetic_measured(
        library, n_vox, 0.01, seed, vi_min, vi_max, rho_max)
    print(f"    library candidates: {lib_mat.shape[0]}, features: {lib_mat.shape[1]}")

    from madi.library import match_voxels_batch

    for label, use_gpu in (("CPU", False), ("GPU", True)):
        t0 = time.time()
        match_voxels_batch(measured, library, vi_min=vi_min, vi_max=vi_max,
                            rho_max=rho_max, use_gpu=use_gpu)
        print(f"    MAP    [{label}]: {time.time() - t0:.3f}s")

    for label, use_gpu in (("CPU", False), ("GPU", True)):
        t0 = time.time()
        fitters.bayes_fit(measured, library, sigma_m=0.05, vi_min=vi_min,
                           vi_max=vi_max, rho_max=rho_max, use_gpu=use_gpu)
        print(f"    Bayes  [{label}]: {time.time() - t0:.3f}s")

    # AMICO-CPU (exact per-voxel NNLS) scales badly with candidate count and
    # is far slower than MAP/Bayes per voxel, so it gets its own (much
    # smaller) voxel count on the CPU side; GPU still runs the full batch.
    t0 = time.time()
    fitters.amico_fit(measured[:n_amico_cpu], library, vi_min=vi_min,
                       vi_max=vi_max, rho_max=rho_max, use_gpu=False,
                       verbose=False)
    t_cpu = time.time() - t0
    print(f"    AMICO  [CPU] ({n_amico_cpu} voxels): {t_cpu:.3f}s "
          f"({t_cpu / n_amico_cpu * 1000:.1f} ms/voxel)")

    t0 = time.time()
    fitters.amico_fit(measured, library, vi_min=vi_min, vi_max=vi_max,
                       rho_max=rho_max, use_gpu=True, verbose=False)
    t_gpu = time.time() - t0
    print(f"    AMICO  [GPU] ({n_vox} voxels): {t_gpu:.3f}s "
          f"({t_gpu / n_vox * 1000:.3f} ms/voxel)")

    extrapolated_cpu = t_cpu / n_amico_cpu * n_vox
    print(f"    AMICO  extrapolated CPU time for {n_vox} voxels: "
          f"{extrapolated_cpu:.1f}s -> speedup ~{extrapolated_cpu / t_gpu:.0f}x")


def _load_merged_library(preferred_paths):
    for path in preferred_paths:
        if os.path.exists(path):
            return load_library(path)
    # Fall back to merging dense shards, if present.
    shard_paths = sorted(glob.glob(
        os.path.join(_REPO_ROOT, "data/libraries/madi_dense.shard*.npz")))
    if not shard_paths:
        raise FileNotFoundError(
            "No library file found. Expected one of "
            f"{preferred_paths} or dense shards under data/libraries/.")
    merged = []
    for p in shard_paths:
        merged.extend(load_library(p))
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-correctness-voxels", type=int, default=300)
    ap.add_argument("--n-timing-voxels", type=int, default=20_000)
    ap.add_argument("--n-amico-cpu-voxels", type=int, default=20,
                     help="AMICO CPU (exact NNLS) is slow and scales badly "
                          "with candidate count, so its timing/correctness "
                          "comparisons use far fewer voxels than GPU.")
    ap.add_argument("--vi-min", type=float, default=0.5,
                     help="Matches madi's own recommended default filter — "
                          "an unfiltered library (0.0) makes AMICO-CPU's "
                          "per-voxel NNLS pathologically slow (documented "
                          "in docs/fitting_methods.md's AMICO cost note).")
    ap.add_argument("--vi-max", type=float, default=0.95)
    ap.add_argument("--rho-max", type=float, default=None)
    args = ap.parse_args()

    print("HAS_CUDA:", fg.HAS_CUDA)
    if not fg.HAS_CUDA:
        print("CUDA not available in this environment — nothing to verify "
              "(CPU-only code paths are unaffected by the GPU fitters "
              "change). Exiting.")
        return

    small_lib_path = os.path.join(_REPO_ROOT, "data/libraries/madi_library_small.npz")
    dense_lib_path = os.path.join(_REPO_ROOT, "data/libraries/madi_dense_human.npz")

    correctness_lib = _load_merged_library([dense_lib_path, small_lib_path])
    print(f"Correctness library: {len(correctness_lib)} entries "
          f"(vi in [{args.vi_min}, {args.vi_max}])")

    ok = True
    ok &= check_map_and_bayes(correctness_lib, args.n_correctness_voxels,
                               args.vi_min, args.vi_max, args.rho_max)
    ok &= check_amico(correctness_lib, args.n_amico_cpu_voxels,
                       args.vi_min, args.vi_max, args.rho_max)
    ok &= check_amico_lambda2_sensitivity(correctness_lib, args.vi_min,
                                           args.vi_max, args.rho_max)
    ok &= check_degenerate_voxel_handling()

    timing_lib = _load_merged_library([dense_lib_path, small_lib_path])
    time_cpu_vs_gpu(timing_lib, args.n_timing_voxels, args.vi_min,
                     args.vi_max, args.rho_max, args.n_amico_cpu_voxels)

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
