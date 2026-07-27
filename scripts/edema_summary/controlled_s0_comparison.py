#!/usr/bin/env python3
"""Low-memory S0-treatment benchmark for MADI Bayesian fitting.

The benchmark has two complementary components:

1. Synthetic raw DWI with known MADI-library truth. This tests fixed-S0
   Bayes, free-S0 Bayes, and a preliminary joint-S0 model under nominal noise
   and a deliberately biased b=0 condition.
2. A small spatial crop from sub-187's grey-matter mask. This is not an
   accuracy test (the truth is unknown), but it checks method stability on
   real acquired DWI without loading a full-volume fit into memory.

The joint-S0 model is a Gaussian high-SNR approximation after Rician
second-moment correction. It retains all b=0 replicates and treats S0 as a
candidate-specific nuisance parameter:

  b0_k ~ N(S0, sigma^2)
  shell_j ~ N(S0 * r_j(theta), sigma^2 / n_directions_j)

For every candidate curve r(theta), S0 is solved by weighted least squares,
then integrated approximately through candidate weights. It is deliberately
kept separate from production fitting: this script is a validation harness.

Usage
-----
    PYTHONPATH=. python -m scripts.edema_summary.controlled_s0_comparison
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter

from madi import fitters, library
from scripts.edema_figures import config
from scripts.fit_data import estimate_sigma_m, parse_bvals, rician_correct_secondmoment


SUBJECT = "187"
TEST_LIBRARY = Path("data/libraries/madi_dense_human.npz")
FIT_BVALUES = np.array([1000.0, 1500.0, 2000.0, 2500.0])
FIT_TRIPLES = [(20.0, 50.0, float(b)) for b in FIT_BVALUES]
PARAMETERS = ("kio", "rho", "V")
METHOD_LABELS = {
    "fixed": "BAYES (fixed S0)",
    "fits0": "BAYES-fits0",
    "joint": "Joint S0 preliminary",
}
METHOD_COLORS = {"fixed": "#0072b2", "fits0": "#d55e00", "joint": "#009e73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=SUBJECT, help="Subject id without the sub- prefix.")
    parser.add_argument(
        "--library",
        default=str(TEST_LIBRARY),
        help="Protocol-matched benchmark library. Use the slim universal-library derivative for the final run; the small human library is the low-memory test default.",
    )
    parser.add_argument("--max-voxels", type=int, default=192, help="Maximum grey-matter voxels to fit from the crop.")
    parser.add_argument("--crop-size", nargs=3, type=int, default=(24, 24, 6), metavar=("X", "Y", "Z"), help="Spatial DWI crop size.")
    parser.add_argument("--n-synthetic", type=int, default=80, help="Synthetic voxels per acquisition condition.")
    parser.add_argument("--target-n-eff", type=float, default=64.0, help="Common median n_eff target for matched-sharpness runs.")
    parser.add_argument(
        "--manual-sigma-m",
        type=float,
        default=None,
        help="Optional normalized-signal sigma_m for an additional manual-sigma run. The joint model converts it to a scale relative to its auto-noise approximation.",
    )
    parser.add_argument("--seed", type=int, default=20260724, help="Random seed for synthetic data and voxel subsampling.")
    parser.add_argument("--output-dir", default="data/outputs/controlled_s0_comparison", help="Output directory, relative to the repository unless absolute.")
    return parser.parse_args()


def _paths(subject: str) -> Tuple[Path, Path, Path, Path]:
    preproc = Path(config.PREPROC_ROOT) / f"sub-{subject}" / "dwi"
    rois = Path(config.ROIS_ROOT) / f"sub-{subject}"
    metadata = Path(config.MADI_ROOT) / f"sub-{subject}" / "dwi" / "method-BAYES" / "fit_metadata.json"
    return (
        preproc / f"sub-{subject}_desc-preproc_dwi.nii.gz",
        preproc / f"sub-{subject}_desc-preproc_dwi.bval",
        rois / f"sub-{subject}_desc-grey-matter-dwi_mask.nii.gz",
        metadata,
    )


def _crop_bounds(mask: np.ndarray, crop_size: Sequence[int]) -> Tuple[slice, slice, slice]:
    size = np.asarray(crop_size, dtype=int)
    if np.any(size <= 0) or np.any(size > np.asarray(mask.shape)):
        raise ValueError(f"invalid crop size {tuple(size)} for mask shape {mask.shape}")
    # Local density picks a tissue-rich block while keeping a deterministic crop.
    score = uniform_filter(mask.astype(float), size=tuple(size), mode="constant")
    center = np.asarray(np.unravel_index(np.argmax(score), score.shape))
    starts = np.clip(center - size // 2, 0, np.asarray(mask.shape) - size)
    stops = starts + size
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))


def _crop_affine(affine: np.ndarray, bounds: Sequence[slice]) -> np.ndarray:
    shifted = affine.copy()
    start = np.array([item.start for item in bounds], dtype=float)
    shifted[:3, 3] = affine[:3, 3] + affine[:3, :3] @ start
    return shifted


def _load_truncated_real_data(
    subject: str,
    crop_size: Sequence[int],
    max_voxels: int,
    output_dir: Path,
    seed: int,
) -> Dict[str, object]:
    dwi_path, bval_path, gm_path, metadata_path = _paths(subject)
    for path in (dwi_path, bval_path, gm_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    gm_img = nib.load(gm_path)
    gm = np.asarray(gm_img.dataobj).astype(bool)
    bounds = _crop_bounds(gm, crop_size)
    local_gm = gm[bounds]
    coords = np.argwhere(local_gm)
    if len(coords) == 0:
        raise ValueError("selected crop contains no grey-matter voxels")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(coords), size=min(max_voxels, len(coords)), replace=False)
    selected_coords = coords[chosen]
    selected_mask = np.zeros(local_gm.shape, dtype=np.uint8)
    selected_mask[tuple(selected_coords.T)] = 1

    dwi_img = nib.load(dwi_path)
    # Nibabel reads only the small crop here; it never materializes the full DWI.
    crop = np.asarray(dwi_img.dataobj[bounds[0], bounds[1], bounds[2], :], dtype=np.float64)
    bvals = np.loadtxt(bval_path).ravel().astype(float)
    if crop.shape[-1] != len(bvals):
        raise ValueError("b-values do not match the cropped DWI volume count")
    _, b0_idx, parsed_shells, _ = parse_bvals(str(bval_path), FIT_BVALUES, tol=30.0)
    shell_by_b = {float(b): index for b, index in parsed_shells}
    shell_idx = [shell_by_b.get(float(b), np.array([], dtype=int)) for b in FIT_BVALUES]
    if len(b0_idx) == 0 or any(len(index) == 0 for index in shell_idx):
        raise ValueError("required b=0 or fitted b-shell volumes are missing")

    metadata = json.loads(metadata_path.read_text())
    noise_sigma = float(metadata["noise_sigma"])
    corrected = rician_correct_secondmoment(crop, noise_sigma)
    selected = selected_mask.astype(bool)
    b0 = corrected[selected][:, b0_idx]
    shells = np.column_stack([corrected[selected][:, index].mean(axis=1) for index in shell_idx])
    n_dirs = np.array([len(index) for index in shell_idx], dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_affine = _crop_affine(dwi_img.affine, bounds)
    nib.save(nib.Nifti1Image(crop.astype(np.float32), crop_affine), output_dir / f"sub-{subject}_desc-grey-matter-crop_dwi.nii.gz")
    nib.save(nib.Nifti1Image(selected_mask, crop_affine), output_dir / f"sub-{subject}_desc-grey-matter-sample_mask.nii.gz")
    (output_dir / f"sub-{subject}_desc-grey-matter-crop.bval").write_text(" ".join(f"{value:g}" for value in bvals) + "\n")

    global_coords = selected_coords + np.array([item.start for item in bounds], dtype=int)
    return {
        "b0": b0,
        "shells": shells,
        "n_dirs": n_dirs,
        "noise_sigma": noise_sigma,
        "coords": global_coords,
        "crop_shape": crop.shape,
        "crop_bounds": [[item.start, item.stop] for item in bounds],
        "n_b0": int(len(b0_idx)),
    }


def _candidate_matrix(lib: list[library.LibraryEntry], meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix, kios, rhos, volumes = library._build_candidate_lib_matrix(
        lib,
        meta["delta_pairs"],
        meta["b_values"],
        meta["n_b"],
        vi_min=0.0,
        vi_max=0.95,
        rho_max=None,
        fit_triples=FIT_TRIPLES,
    )
    return matrix.astype(np.float64), kios, rhos, volumes


def _weighted_summary(weights: np.ndarray, kios: np.ndarray, rhos: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
    result = {}
    for name, values in (("kio", kios), ("rho", rhos), ("V", volumes)):
        mean = weights @ values
        variance = np.maximum(weights @ (values ** 2) - mean ** 2, 0.0)
        result[name] = mean
        result[f"{name}_std"] = np.sqrt(variance)
    result["n_eff"] = 1.0 / np.maximum(np.sum(weights ** 2, axis=1), 1e-300)
    return result


def joint_s0_bayes(
    b0: np.ndarray,
    shells: np.ndarray,
    candidate_matrix: np.ndarray,
    kios: np.ndarray,
    rhos: np.ndarray,
    volumes: np.ndarray,
    noise_sigma: float,
    n_dirs: np.ndarray,
    noise_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Bayesian candidate weighting with b=0 observations and a latent S0.

    The conditional weighted-least-squares S0 has a closed form for every
    candidate. Gaussian errors are an intentional preliminary approximation
    after Rician correction, not a replacement for a full Rician likelihood.
    """
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")
    b0_var = (noise_sigma * noise_scale) ** 2
    shell_var = (noise_sigma * noise_scale) ** 2 / n_dirs
    inv_b0_var = 1.0 / b0_var
    inv_shell_var = 1.0 / shell_var
    n_b0 = b0.shape[1]

    # For each candidate r_i, solve the weighted linear amplitude S0_i*.
    numerator = b0.sum(axis=1, keepdims=True) * inv_b0_var
    numerator = numerator + (shells * inv_shell_var) @ candidate_matrix.T
    denominator = n_b0 * inv_b0_var + (candidate_matrix ** 2 * inv_shell_var).sum(axis=1)
    s0_candidates = numerator / denominator[None, :]

    b0_rss = (b0 ** 2).sum(axis=1, keepdims=True) * inv_b0_var
    shell_rss = (shells ** 2 * inv_shell_var).sum(axis=1, keepdims=True)
    rss = b0_rss + shell_rss - numerator ** 2 / denominator[None, :]
    valid = s0_candidates > 0
    log_weights = np.where(valid, -0.5 * rss, -np.inf)
    row_max = np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights - row_max)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
    result = _weighted_summary(weights, kios, rhos, volumes)
    result["s0_fit"] = np.sum(weights * s0_candidates, axis=1)
    result["residual"] = np.sum(weights * rss, axis=1)
    return result


def _joint_noise_scale_for_target(
    b0: np.ndarray,
    shells: np.ndarray,
    candidate_matrix: np.ndarray,
    kios: np.ndarray,
    rhos: np.ndarray,
    volumes: np.ndarray,
    noise_sigma: float,
    n_dirs: np.ndarray,
    target: float,
) -> float:
    lo, hi = 0.05, 20.0
    for _ in range(16):
        mid = np.sqrt(lo * hi)
        fitted = joint_s0_bayes(b0, shells, candidate_matrix, kios, rhos, volumes, noise_sigma, n_dirs, mid)
        if np.median(fitted["n_eff"]) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _fit_existing_bayes(
    method: str,
    lib: list[library.LibraryEntry],
    meta: dict,
    ratios: np.ndarray,
    shells: np.ndarray,
    sigma_m: float,
) -> Dict[str, np.ndarray]:
    fitted = fitters.bayes_fit(
        ratios,
        lib,
        sigma_m=sigma_m,
        lib_delta_pairs=meta["delta_pairs"],
        lib_b_values=meta["b_values"],
        n_b=meta["n_b"],
        fit_triples=FIT_TRIPLES,
        vi_min=0.0,
        vi_max=0.95,
        rho_max=None,
        fit_s0=method == "fits0",
        raw_signal=shells if method == "fits0" else None,
        use_gpu=False,
    )
    for parameter in PARAMETERS:
        fitted[parameter] = fitted[f"{parameter}_mean"]
    return fitted


def _calibrate_existing_sigma(
    method: str,
    lib: list[library.LibraryEntry],
    meta: dict,
    ratios: np.ndarray,
    shells: np.ndarray,
    target_n_eff: float,
) -> float:
    return fitters.calibrate_sigma_m(
        ratios,
        lib,
        lib_delta_pairs=meta["delta_pairs"],
        lib_b_values=meta["b_values"],
        n_b=meta["n_b"],
        fit_triples=FIT_TRIPLES,
        target_n_eff=target_n_eff,
        fit_s0=method == "fits0",
        raw_signal=shells if method == "fits0" else None,
        vi_min=0.0,
        vi_max=0.95,
        rho_max=None,
        n_iter=16,
        sample_size=len(ratios),
        use_gpu=False,
        verbose=False,
    )


def fit_three_methods(
    b0: np.ndarray,
    shells: np.ndarray,
    n_dirs: np.ndarray,
    noise_sigma: float,
    lib: list[library.LibraryEntry],
    meta: dict,
    candidate_matrix: np.ndarray,
    kios: np.ndarray,
    rhos: np.ndarray,
    volumes: np.ndarray,
    fit_mode: str,
    target_n_eff: float,
    manual_sigma_m: float | None = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float]]:
    s0_measured = np.mean(b0, axis=1)
    ratios = shells / np.maximum(s0_measured[:, None], 1e-10)
    if fit_mode == "matched_n_eff":
        sigma_fixed = _calibrate_existing_sigma("fixed", lib, meta, ratios, shells, target_n_eff)
        sigma_fits0 = _calibrate_existing_sigma("fits0", lib, meta, ratios, shells, target_n_eff)
        joint_scale = _joint_noise_scale_for_target(
            b0, shells, candidate_matrix, kios, rhos, volumes, noise_sigma, n_dirs, target_n_eff
        )
    elif fit_mode == "auto_sigma":
        median_s0 = float(np.median(s0_measured))
        sigma_fixed = estimate_sigma_m(noise_sigma, median_s0, float(np.mean(n_dirs)))
        sigma_fits0 = sigma_fixed
        joint_scale = 1.0
    elif fit_mode == "manual_sigma":
        if manual_sigma_m is None or manual_sigma_m <= 0:
            raise ValueError("manual_sigma mode requires --manual-sigma-m > 0")
        median_s0 = float(np.median(s0_measured))
        auto_sigma = estimate_sigma_m(noise_sigma, median_s0, float(np.mean(n_dirs)))
        sigma_fixed = manual_sigma_m
        sigma_fits0 = manual_sigma_m
        joint_scale = manual_sigma_m / auto_sigma
    else:
        raise ValueError(f"unknown fit mode {fit_mode}")
    results = {
        "fixed": _fit_existing_bayes("fixed", lib, meta, ratios, shells, sigma_fixed),
        "fits0": _fit_existing_bayes("fits0", lib, meta, ratios, shells, sigma_fits0),
        "joint": joint_s0_bayes(b0, shells, candidate_matrix, kios, rhos, volumes, noise_sigma, n_dirs, joint_scale),
    }
    for method, result in results.items():
        if method == "fixed":
            result["s0_fit"] = s0_measured
    settings = {"sigma_fixed": sigma_fixed, "sigma_fits0": sigma_fits0, "joint_noise_scale": joint_scale}
    return results, settings


def _rician(signal: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.sqrt((signal + rng.normal(0.0, sigma)) ** 2 + rng.normal(0.0, sigma) ** 2)


def make_synthetic_data(
    candidate_matrix: np.ndarray,
    parameter_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    n_dirs: np.ndarray,
    noise_sigma: float,
    n_samples: int,
    condition: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    indices = rng.choice(len(candidate_matrix), size=n_samples, replace=True)
    s0_truth = rng.uniform(3200.0, 5200.0, size=n_samples)
    b0_multiplier = 1.0 if condition == "nominal" else 1.12
    b0_noise_multiplier = 1.0 if condition == "nominal" else 2.0
    # Independent b=0 replicates, then independently noisy direction means per shell.
    b0 = _rician(np.repeat(s0_truth[:, None] * b0_multiplier, 5, axis=1), noise_sigma * b0_noise_multiplier, rng)
    shell_means = []
    for column, n_direction in enumerate(n_dirs.astype(int)):
        true_signal = s0_truth[:, None] * candidate_matrix[indices, column, None]
        repeats = _rician(np.repeat(true_signal, n_direction, axis=1), noise_sigma, rng)
        shell_means.append(repeats.mean(axis=1))
    shells = np.column_stack(shell_means)
    kios, rhos, volumes = parameter_arrays
    truth = pd.DataFrame(
        {
            "sample": np.arange(n_samples),
            "truth_kio": kios[indices],
            "truth_rho": rhos[indices],
            "truth_V": volumes[indices],
            "truth_s0": s0_truth,
            "condition": condition,
        }
    )
    return b0, shells, truth


def _synthetic_results(
    b0: np.ndarray,
    shells: np.ndarray,
    truth: pd.DataFrame,
    n_dirs: np.ndarray,
    noise_sigma: float,
    lib: list[library.LibraryEntry],
    meta: dict,
    candidate_matrix: np.ndarray,
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_n_eff: float,
    fit_modes: Sequence[str],
    manual_sigma_m: float | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    settings_rows = []
    for fit_mode in fit_modes:
        fits, settings = fit_three_methods(
            b0, shells, n_dirs, noise_sigma, lib, meta, candidate_matrix, *arrays, fit_mode, target_n_eff, manual_sigma_m
        )
        settings_rows.append({"condition": truth["condition"].iat[0], "fit_mode": fit_mode, **settings})
        for method, result in fits.items():
            for parameter in PARAMETERS:
                estimate = result[parameter]
                posterior_std = result[f"{parameter}_std"]
                true = truth[f"truth_{parameter}"].to_numpy()
                for sample, estimate_value, std_value, true_value, n_eff in zip(
                    truth["sample"], estimate, posterior_std, true, result["n_eff"]
                ):
                    rows.append(
                        {
                            "condition": truth["condition"].iat[0],
                            "fit_mode": fit_mode,
                            "method": method,
                            "sample": int(sample),
                            "parameter": parameter,
                            "truth": float(true_value),
                            "estimate": float(estimate_value),
                            "posterior_std": float(std_value),
                            "n_eff": float(n_eff),
                            "absolute_relative_error": float(abs(estimate_value - true_value) / max(abs(true_value), 1e-12)),
                            "covered_by_one_posterior_std": bool(abs(estimate_value - true_value) <= std_value),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(settings_rows)


def _real_results(
    real: Dict[str, object],
    lib: list[library.LibraryEntry],
    meta: dict,
    candidate_matrix: np.ndarray,
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_n_eff: float,
    fit_modes: Sequence[str],
    manual_sigma_m: float | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    settings_rows = []
    for fit_mode in fit_modes:
        fits, settings = fit_three_methods(
            real["b0"], real["shells"], real["n_dirs"], real["noise_sigma"],
            lib, meta, candidate_matrix, *arrays, fit_mode, target_n_eff, manual_sigma_m
        )
        settings_rows.append({"fit_mode": fit_mode, **settings})
        for method, result in fits.items():
            for index, coordinate in enumerate(real["coords"]):
                rows.append(
                    {
                        "fit_mode": fit_mode,
                        "method": method,
                        "voxel": index,
                        "x": int(coordinate[0]),
                        "y": int(coordinate[1]),
                        "z": int(coordinate[2]),
                        "measured_s0": float(np.mean(real["b0"][index])),
                        "s0_fit": float(result["s0_fit"][index]),
                        "n_eff": float(result["n_eff"][index]),
                        **{parameter: float(result[parameter][index]) for parameter in PARAMETERS},
                        **{f"{parameter}_std": float(result[f"{parameter}_std"][index]) for parameter in PARAMETERS},
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(settings_rows)


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def plot_synthetic_error(results: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(len(PARAMETERS), 2, figsize=(10.2, 9.2), sharex=True)
    for row, parameter in enumerate(PARAMETERS):
        for column, condition in enumerate(("nominal", "b0_biased")):
            ax = axes[row, column]
            data = results[(results["parameter"] == parameter) & (results["condition"] == condition) & (results["fit_mode"] == "matched_n_eff")]
            sns.boxplot(
                data=data,
                x="method",
                y="absolute_relative_error",
                hue="method",
                order=list(METHOD_LABELS),
                hue_order=list(METHOD_LABELS),
                palette=METHOD_COLORS,
                dodge=False,
                fliersize=0,
                legend=False,
                ax=ax,
            )
            sns.stripplot(data=data, x="method", y="absolute_relative_error", order=list(METHOD_LABELS), color="black", alpha=0.35, size=2.2, ax=ax)
            ax.set_yscale("log")
            ax.set_xlabel("")
            ax.set_ylabel(f"{parameter} absolute relative error")
            ax.set_title("Nominal acquisition" if condition == "nominal" else "b=0 bias +12%, b=0 noise x2")
            ax.set_xticks(range(len(METHOD_LABELS)))
            ax.set_xticklabels(["Fixed", "Fits0", "Joint"])
    fig.suptitle("Synthetic recovery with matched posterior sharpness", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "01_synthetic_parameter_error_matched_neff.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_synthetic_neff(results: pd.DataFrame, output_dir: Path) -> Path:
    data = results[results["parameter"] == "kio"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    sns.boxplot(data=data, x="method", y="n_eff", hue="fit_mode", order=list(METHOD_LABELS), palette="Set2", ax=axes[0])
    axes[0].set_title("Posterior sharpness by calibration mode")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Effective library atoms")
    coverage = data.groupby(["condition", "fit_mode", "method"], as_index=False)["covered_by_one_posterior_std"].mean()
    sns.barplot(data=coverage, x="method", y="covered_by_one_posterior_std", hue="condition", order=list(METHOD_LABELS), palette="Set2", ax=axes[1])
    axes[1].axhline(0.6827, color="0.25", linestyle="--", linewidth=1, label="68.3% nominal")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("One-posterior-SD coverage for kio")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Empirical coverage")
    fig.tight_layout()
    path = output_dir / "02_synthetic_neff_and_coverage.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_real_agreement(real_results: pd.DataFrame, output_dir: Path) -> Path:
    data = real_results[real_results["fit_mode"] == "matched_n_eff"]
    fig, axes = plt.subplots(1, len(PARAMETERS), figsize=(4.2 * len(PARAMETERS), 4.0))
    fixed = data[data["method"] == "fixed"].set_index("voxel")
    fits0 = data[data["method"] == "fits0"].set_index("voxel")
    joint = data[data["method"] == "joint"].set_index("voxel")
    for ax, parameter in zip(np.ravel(axes), PARAMETERS):
        ax.scatter(fixed[parameter], fits0[parameter], s=16, alpha=0.55, color=METHOD_COLORS["fits0"], label="Fits0")
        ax.scatter(fixed[parameter], joint[parameter], s=16, alpha=0.55, color=METHOD_COLORS["joint"], label="Joint")
        values = np.concatenate([fixed[parameter].to_numpy(), fits0[parameter].to_numpy(), joint[parameter].to_numpy()])
        low, high = np.nanmin(values), np.nanmax(values)
        pad = 0.05 * (high - low) if high > low else 1.0
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color="0.35", linewidth=1)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel("Fixed-S0 BAYES")
        ax.set_ylabel("Alternative estimate")
        ax.set_title(parameter)
        ax.grid(color="0.9")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Sub-187 truncated grey-matter fit: matched n_eff", fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    path = output_dir / "03_sub187_truncated_grey_matter_parameter_agreement.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_real_s0(real_results: pd.DataFrame, output_dir: Path) -> Path:
    data = real_results[real_results["fit_mode"] == "matched_n_eff"]
    measured = data[data["method"] == "fixed"].set_index("voxel")["measured_s0"]
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    for method in ("fits0", "joint"):
        fitted = data[data["method"] == method].set_index("voxel")["s0_fit"]
        ax.scatter(measured, fitted, s=18, alpha=0.58, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    values = np.concatenate([measured.to_numpy(), data[data["method"].isin(["fits0", "joint"])]["s0_fit"].to_numpy()])
    low, high = np.nanmin(values), np.nanmax(values)
    pad = 0.05 * (high - low)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="0.35", linewidth=1)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Measured mean b=0 (five replicates)")
    ax.set_ylabel("Fitted S0")
    ax.set_title("Sub-187 truncated grey matter")
    ax.legend(frameon=False)
    ax.grid(color="0.9")
    fig.tight_layout()
    path = output_dir / "04_sub187_truncated_grey_matter_s0_agreement.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    library_path = Path(args.library)
    if not library_path.exists():
        raise SystemExit(f"benchmark library is missing: {library_path}")

    lib = library.load_library(str(library_path))
    meta = library.load_library_meta(str(library_path))
    candidate_matrix, kios, rhos, volumes = _candidate_matrix(lib, meta)
    arrays = (kios, rhos, volumes)
    rng = np.random.default_rng(args.seed)
    fit_modes = ["matched_n_eff", "auto_sigma"]
    if args.manual_sigma_m is not None:
        fit_modes.append("manual_sigma")

    real = _load_truncated_real_data(args.subject, args.crop_size, args.max_voxels, output_dir, args.seed)
    synthetic_rows = []
    setting_rows = []
    for condition in ("nominal", "b0_biased"):
        b0, shells, truth = make_synthetic_data(
            candidate_matrix, arrays, real["n_dirs"], real["noise_sigma"], args.n_synthetic, condition, rng
        )
        result, settings = _synthetic_results(
            b0, shells, truth, real["n_dirs"], real["noise_sigma"], lib, meta,
            candidate_matrix, arrays, args.target_n_eff, fit_modes, args.manual_sigma_m
        )
        synthetic_rows.append(result)
        setting_rows.append(settings)
    synthetic = pd.concat(synthetic_rows, ignore_index=True)
    synthetic_settings = pd.concat(setting_rows, ignore_index=True)
    real_results, real_settings = _real_results(
        real, lib, meta, candidate_matrix, arrays, args.target_n_eff, fit_modes, args.manual_sigma_m
    )

    _save_csv(synthetic, output_dir / "synthetic_results_long.csv")
    _save_csv(synthetic_settings, output_dir / "synthetic_fit_settings.csv")
    _save_csv(real_results, output_dir / f"sub-{args.subject}_truncated_grey_matter_results.csv")
    _save_csv(real_settings, output_dir / f"sub-{args.subject}_truncated_grey_matter_fit_settings.csv")
    figures = [
        plot_synthetic_error(synthetic, output_dir),
        plot_synthetic_neff(synthetic, output_dir),
        plot_real_agreement(real_results, output_dir),
        plot_real_s0(real_results, output_dir),
    ]
    metadata = {
        "subject": args.subject,
        "max_voxels": args.max_voxels,
        "crop_size": args.crop_size,
        "crop_shape": list(real["crop_shape"]),
        "crop_bounds": real["crop_bounds"],
        "selected_grey_matter_voxels": int(len(real["coords"])),
        "n_b0_replicates": real["n_b0"],
        "fit_bvalues": FIT_BVALUES.tolist(),
        "n_directions_per_shell": real["n_dirs"].astype(int).tolist(),
        "noise_sigma": real["noise_sigma"],
        "target_n_eff": args.target_n_eff,
        "manual_sigma_m": args.manual_sigma_m,
        "library": str(library_path),
        "library_candidates": int(len(candidate_matrix)),
        "notes": "Small-library, cropped-DWI validation harness; not a replacement for production universal-library maps.",
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"output directory: {output_dir}")
    print(f"cropped DWI shape: {real['crop_shape']}; fitted grey-matter voxels: {len(real['coords'])}")
    print(f"synthetic rows: {len(synthetic)}; real result rows: {len(real_results)}")
    for figure in figures:
        print(f"  {figure}")


if __name__ == "__main__":
    main()
