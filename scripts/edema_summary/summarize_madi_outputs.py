#!/usr/bin/env python3
"""
Create CSV summaries and overview figures for edema-cohort MADI outputs.

This intentionally lives outside scripts/edema_figures. It imports the old
figure configuration/loaders for paths, subjects, parameter labels, and display
helpers, but does not modify the paper-figure replication code.

Usage
-----
    PYTHONPATH=. python -m scripts.edema_summary.summarize_madi_outputs
    PYTHONPATH=. python -m scripts.edema_summary.summarize_madi_outputs --methods BAYES BAYES-fits0
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.edema_figures import config, loaders, slicing, windowing


DEFAULT_ROI_ORDER = [
    "edema",
    "contra",
    "tumor",
    "tumor-core",
    "tumor-net",
    "tumor-edema",
    "grey-matter",
    "white-matter",
]

ROI_LABELS = {
    "edema": "Edema",
    "contra": "Contralateral",
    "tumor": "Tumor",
    "tumor-core": "Tumor core",
    "tumor-net": "Tumor NET",
    "tumor-edema": "Tumor edema",
    "grey-matter": "Grey matter",
    "white-matter": "White matter",
}

ROI_COLORS = {
    "edema": "#d55e00",
    "contra": "#0072b2",
    "tumor": "#cc79a7",
    "tumor-core": "#e69f00",
    "tumor-net": "#56b4e9",
    "tumor-edema": "#009e73",
    "grey-matter": "#6b7280",
    "white-matter": "#8b5cf6",
}

METHOD_MARKERS = {"MAP": "o", "MAP-fits0": "s", "BAYES": "^", "BAYES-fits0": "D"}

# The fits0 Bayesian solution is the cohort's primary estimand.  Keep the
# other methods in the tables and use them only in explicitly comparative plots.
PRIMARY_METHOD = "BAYES-fits0"
REFERENCE_ROIS = ("grey-matter", "white-matter", "contra")
REFERENCE_BAYES_METHODS = ("BAYES", "BAYES-fits0")


def _roi_mask_files(subject: str) -> Dict[str, str]:
    pattern = os.path.join(config.ROIS_ROOT, f"sub-{subject}", f"sub-{subject}_desc-*-dwi_mask.nii.gz")
    paths = sorted(glob.glob(pattern))
    rois = {}
    for path in paths:
        name = os.path.basename(path)
        roi = name.split("_desc-", 1)[1].split("-dwi_mask", 1)[0]
        rois[roi] = path
    return rois


def discover_subjects() -> List[str]:
    subjects = []
    for path in sorted(glob.glob(os.path.join(config.MADI_ROOT, "sub-*"))):
        subject = os.path.basename(path).replace("sub-", "")
        if _roi_mask_files(subject):
            subjects.append(subject)
    return subjects


def roi_sort_key(roi: str) -> Tuple[int, str]:
    try:
        return DEFAULT_ROI_ORDER.index(roi), roi
    except ValueError:
        return len(DEFAULT_ROI_ORDER), roi


def ordered_rois(rois: Iterable[str]) -> List[str]:
    return sorted(set(rois), key=roi_sort_key)


def load_mask(path: str, shape: Tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(nib.load(path).dataobj).astype(bool)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match map shape {shape}: {path}")
    return mask


def finite_roi_values(data: np.ndarray, mask: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    values = data[mask & brain_mask]
    return values[np.isfinite(values)]


def stats_for_values(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "n_voxels": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
        }
    return {
        "n_voxels": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def build_summary(
    subjects: Sequence[str],
    methods: Sequence[str],
    params: Sequence[str],
    allowed_rois: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    rows = []
    allowed_roi_set = set(allowed_rois) if allowed_rois is not None else None
    for subject in subjects:
        roi_paths = _roi_mask_files(subject)
        if not roi_paths:
            continue
        for method in methods:
            for param in params:
                map_path = loaders.param_map_path(subject, method, param)
                if not os.path.exists(map_path):
                    continue
                data = loaders.load_param_map(subject, method, param)
                brain_mask = loaders.load_brain_mask(subject, data.shape)
                for roi, mask_path in roi_paths.items():
                    if allowed_roi_set is not None and roi not in allowed_roi_set:
                        continue
                    try:
                        mask = load_mask(mask_path, data.shape)
                    except ValueError as e:
                        print(f"[summary] skip sub-{subject} {roi}: {e}")
                        continue
                    stat = stats_for_values(finite_roi_values(data, mask, brain_mask))
                    rows.append(
                        {
                            "subject": subject,
                            "method": method,
                            "parameter": param,
                            "roi": roi,
                            "roi_label": ROI_LABELS.get(roi, roi),
                            "map_path": map_path,
                            "mask_path": mask_path,
                            **stat,
                        }
                    )
    return pd.DataFrame(rows)


def build_overlap(subjects: Sequence[str], allowed_rois: Optional[Sequence[str]] = None) -> pd.DataFrame:
    rows = []
    allowed_roi_set = set(allowed_rois) if allowed_rois is not None else None
    for subject in subjects:
        roi_paths = _roi_mask_files(subject)
        if not roi_paths:
            continue
        ref_shape = None
        masks = {}
        for method in config.ALL_METHODS:
            p = loaders.param_map_path(subject, method, config.PARAMS[0])
            if os.path.exists(p):
                ref_shape = loaders.load_param_map(subject, method, config.PARAMS[0]).shape
                break
        if ref_shape is None:
            continue
        for roi, path in roi_paths.items():
            if allowed_roi_set is not None and roi not in allowed_roi_set:
                continue
            try:
                masks[roi] = load_mask(path, ref_shape)
            except ValueError:
                continue
        for roi_a in ordered_rois(masks):
            for roi_b in ordered_rois(masks):
                a = masks[roi_a]
                b = masks[roi_b]
                rows.append(
                    {
                        "subject": subject,
                        "roi_a": roi_a,
                        "roi_b": roi_b,
                        "overlap_voxels": int(np.count_nonzero(a & b)),
                        "roi_a_voxels": int(np.count_nonzero(a)),
                        "roi_b_voxels": int(np.count_nonzero(b)),
                        "fraction_of_roi_a": float(np.count_nonzero(a & b) / np.count_nonzero(a))
                        if np.count_nonzero(a) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def save_csvs(summary: pd.DataFrame, overlap: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary, tables_dir / "madi_roi_summary_long.csv")
    wide = summary.pivot_table(
        index=["subject", "method", "parameter"],
        columns="roi",
        values=["n_voxels", "mean", "median", "std", "min", "max", "range"],
        aggfunc="first",
    )
    wide.columns = [f"{stat}_{roi}" for stat, roi in wide.columns]
    _write_csv(wide.reset_index(), tables_dir / "madi_roi_summary_wide.csv")
    _write_csv(overlap, tables_dir / "madi_roi_overlap_voxels.csv")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _figure_path(figures_dir: Path, name: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / f"{name}.png"


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)


def _format_roi_xticks(ax, rois: Sequence[str]) -> None:
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels([ROI_LABELS.get(roi, roi) for roi in rois], rotation=35, ha="right")


def plot_mean_bars(summary: pd.DataFrame, figures_dir: Path) -> List[Path]:
    outputs = []
    methods = [m for m in config.ALL_METHODS if m in set(summary["method"])]
    rois = ordered_rois(summary["roi"])
    for param in config.PARAMS:
        sub = summary[(summary["parameter"] == param) & summary["mean"].notna()]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(rois) * len(methods)), 4.8))
        sns.barplot(data=sub, x="roi", y="mean", hue="method", order=rois, hue_order=methods, errorbar="sd", ax=ax)
        sns.stripplot(data=sub, x="roi", y="mean", order=rois, color="black", size=3, alpha=0.45, dodge=False, ax=ax)
        ax.set_title(f"ROI mean by method: {param}")
        ax.set_xlabel("")
        ax.set_ylabel(config.PARAM_LABELS[param])
        _format_roi_xticks(ax, rois)
        _clean_axes(ax)
        fig.tight_layout()
        out = _figure_path(figures_dir, f"01_mean_bar_{param}")
        fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def plot_median_subject_lines(summary: pd.DataFrame, figures_dir: Path) -> List[Path]:
    outputs = []
    rois = ordered_rois(summary["roi"])
    for param in config.PARAMS:
        sub = summary[(summary["parameter"] == param) & (summary["method"] == PRIMARY_METHOD) & summary["median"].notna()]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        sns.pointplot(data=sub, x="roi", y="median", hue="subject", order=rois, errorbar=None, markers="o", ax=ax)
        ax.set_title(f"Subject ROI medians: {param} ({PRIMARY_METHOD})")
        ax.set_xlabel("")
        ax.set_ylabel(config.PARAM_LABELS[param])
        _format_roi_xticks(ax, rois)
        _clean_axes(ax)
        fig.tight_layout()
        out = _figure_path(figures_dir, f"02_subject_median_lines_{param}")
        fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def plot_edema_contra_scatter(summary: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    sub = summary[(summary["method"] == PRIMARY_METHOD) & summary["roi"].isin(["edema", "contra"])]
    if sub.empty:
        return None
    wide = sub.pivot_table(index=["subject", "parameter"], columns="roi", values="mean", aggfunc="first").reset_index()
    wide = wide.dropna(subset=["edema", "contra"])
    if wide.empty:
        return None
    fig, axes = plt.subplots(1, len(config.PARAMS), figsize=(4.2 * len(config.PARAMS), 4.0))
    for ax, param in zip(np.ravel(axes), config.PARAMS):
        psub = wide[wide["parameter"] == param]
        if psub.empty:
            ax.axis("off")
            continue
        ax.scatter(psub["contra"], psub["edema"], s=60, color="#d55e00", edgecolor="white", linewidth=0.8)
        for _, row in psub.iterrows():
            ax.text(row["contra"], row["edema"], f" {row['subject']}", va="center", fontsize=8)
        lims = [np.nanmin(psub[["contra", "edema"]].to_numpy()), np.nanmax(psub[["contra", "edema"]].to_numpy())]
        pad = (lims[1] - lims[0]) * 0.08 if lims[1] > lims[0] else 1.0
        ax.plot([lims[0] - pad, lims[1] + pad], [lims[0] - pad, lims[1] + pad], color="0.5", linewidth=1)
        ax.set_xlim(lims[0] - pad, lims[1] + pad)
        ax.set_ylim(lims[0] - pad, lims[1] + pad)
        ax.set_title(param)
        ax.set_xlabel("Contralateral mean")
        ax.set_ylabel("Edema mean")
        _clean_axes(ax)
    fig.suptitle(f"Edema vs. contralateral ROI means ({PRIMARY_METHOD})")
    fig.tight_layout()
    out = _figure_path(figures_dir, "03_edema_contra_scatter")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_method_correlation(summary: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    wide = summary.pivot_table(index=["subject", "parameter", "roi"], columns="method", values="mean", aggfunc="first")
    cols = [m for m in config.ALL_METHODS if m in wide.columns]
    if len(cols) < 2:
        return None
    corr = wide[cols].corr()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("Method agreement across ROI means")
    fig.tight_layout()
    out = _figure_path(figures_dir, "04_method_correlation_heatmap")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_voxel_count_heatmap(summary: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    sub = summary[(summary["method"] == PRIMARY_METHOD) & (summary["parameter"] == config.PARAMS[0])]
    if sub.empty:
        return None
    rois = ordered_rois(sub["roi"])
    table = sub.pivot_table(index="subject", columns="roi", values="n_voxels", aggfunc="first").reindex(columns=rois)
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(rois)), 3.8))
    sns.heatmap(table, cmap="viridis", annot=True, fmt=".0f", linewidths=0.4, ax=ax)
    ax.set_title("ROI voxel counts")
    ax.set_xlabel("")
    ax.set_ylabel("Subject")
    _format_roi_xticks(ax, rois)
    fig.tight_layout()
    out = _figure_path(figures_dir, "05_roi_voxel_count_heatmap")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def pick_subject_slice(subject: str, brain_mask: np.ndarray) -> int:
    for roi in ("edema", "tumor-edema", "tumor"):
        path = _roi_mask_files(subject).get(roi)
        if path:
            mask = load_mask(path, brain_mask.shape)
            if mask.any():
                return loaders.mask_slice_index(mask)
    counts = brain_mask.sum(axis=(0, 1))
    return int(np.argmax(counts)) if counts.any() else brain_mask.shape[2] // 2


def plot_brain_montage(subjects: Sequence[str], figures_dir: Path, method: str = PRIMARY_METHOD) -> Optional[Path]:
    loaded = {}
    for subject in subjects:
        try:
            maps = {p: loaders.load_param_map(subject, method, p) for p in config.PARAMS}
            brain = loaders.load_brain_mask(subject, maps[config.PARAMS[0]].shape)
            z = pick_subject_slice(subject, brain)
            loaded[subject] = (maps, brain, z)
        except (FileNotFoundError, ValueError):
            continue
    if not loaded:
        return None
    windows = windowing.compute_windows(
        {p: [loaded[s][0][p][loaded[s][1]] for s in loaded] for p in config.PARAMS}
    )
    for param, override in config.FIG1_CONTRAST_OVERRIDE.items():
        if param in windows:
            windows[param] = override

    n_rows, n_cols = len(loaded), len(config.PARAMS)
    fig = plt.figure(figsize=(3.0 * n_cols, 2.5 * n_rows))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig, width_ratios=[1] * n_cols + [0.08], wspace=0.05, hspace=0.04)
    last_im = None
    for r, subject in enumerate(loaded):
        maps, brain, z = loaded[subject]
        bbox = slicing.square_crop_bbox(slicing.axial(brain, z), margin_frac=0.16)
        for c, param in enumerate(config.PARAMS):
            ax = fig.add_subplot(gs[r, c])
            last_im = windowing.render_panel(
                ax,
                slicing.axial(maps[param], z),
                slicing.axial(brain, z),
                *windows[param],
                crop_bbox=bbox,
            )
            if r == 0:
                ax.set_title(config.PARAM_LABELS[param], fontsize=11)
            if c == 0:
                ax.text(-0.10, 0.5, f"sub-{subject}\nz={z}", ha="right", va="center", fontsize=9, transform=ax.transAxes)
    if last_im is not None:
        cax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=4)
        cbar.update_ticks()
    fig.suptitle(f"Cropped MADI parameter maps ({method})", fontsize=13)
    windowing.style_dark_figure(fig)
    out = _figure_path(figures_dir, "06_cropped_brain_parameter_montage")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_roi_overlay_montage(subjects: Sequence[str], figures_dir: Path, method: str = PRIMARY_METHOD, param: str = "kio") -> Optional[Path]:
    loaded = {}
    for subject in subjects:
        try:
            data = loaders.load_param_map(subject, method, param)
            brain = loaders.load_brain_mask(subject, data.shape)
            roi_paths = _roi_mask_files(subject)
            masks = {roi: load_mask(path, data.shape) for roi, path in roi_paths.items() if roi in ROI_COLORS}
            z = pick_subject_slice(subject, brain)
            loaded[subject] = (data, brain, masks, z)
        except (FileNotFoundError, ValueError):
            continue
    if not loaded:
        return None
    vmin, vmax = windowing.compute_windows({param: [loaded[s][0][loaded[s][1]] for s in loaded]})[param]
    fig, axes = plt.subplots(1, len(loaded), figsize=(3.0 * len(loaded), 3.2))
    axes = np.ravel([axes])
    for ax, subject in zip(axes, loaded):
        data, brain, masks, z = loaded[subject]
        brain2d = slicing.axial(brain, z)
        bbox = slicing.square_crop_bbox(brain2d, margin_frac=0.16)
        windowing.render_panel(ax, slicing.axial(data, z), brain2d, vmin, vmax, crop_bbox=bbox)
        for roi in ordered_rois(masks):
            mask2d = slicing.axial(masks[roi], z)
            if mask2d.any():
                ax.contour(mask2d, levels=[0.5], colors=[ROI_COLORS.get(roi, "white")], linewidths=1.1)
        ax.set_title(f"sub-{subject} z={z}", fontsize=10)
    handles = [
        plt.Line2D([], [], color=ROI_COLORS[roi], linewidth=2, label=ROI_LABELS.get(roi, roi))
        for roi in DEFAULT_ROI_ORDER
        if any(roi in loaded[s][2] for s in loaded)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, labelcolor="0.92")
    fig.suptitle(f"Cropped {param} maps with ROI outlines ({method})", fontsize=13)
    windowing.style_dark_figure(fig)
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    out = _figure_path(figures_dir, "07_cropped_brain_roi_overlay")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_roi_overlap(summary: pd.DataFrame, overlap: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    if overlap.empty:
        return None
    rois = ordered_rois(summary["roi"])
    table = (
        overlap.groupby(["roi_a", "roi_b"])["fraction_of_roi_a"]
        .mean()
        .unstack()
        .reindex(index=rois, columns=rois)
    )
    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(rois)), max(5.5, 0.6 * len(rois))))
    sns.heatmap(table, vmin=0, vmax=1, cmap="mako", annot=True, fmt=".2f", linewidths=0.4, ax=ax)
    ax.set_title("Mean ROI overlap fraction")
    ax.set_xlabel("ROI B")
    ax.set_ylabel("ROI A")
    _format_roi_xticks(ax, rois)
    ax.set_yticklabels([ROI_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()], rotation=0)
    fig.tight_layout()
    out = _figure_path(figures_dir, "08_roi_overlap_heatmap")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_std_vs_mean(summary: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    sub = summary[(summary["method"] == PRIMARY_METHOD) & summary["mean"].notna() & summary["std"].notna()]
    if sub.empty:
        return None
    fig, axes = plt.subplots(1, len(config.PARAMS), figsize=(4.2 * len(config.PARAMS), 4.0))
    for ax, param in zip(np.ravel(axes), config.PARAMS):
        psub = sub[sub["parameter"] == param]
        sns.scatterplot(data=psub, x="mean", y="std", hue="roi", style="subject", palette=ROI_COLORS, s=70, ax=ax)
        ax.set_title(param)
        ax.set_xlabel("Mean")
        ax.set_ylabel("Std")
        handles, labels = ax.get_legend_handles_labels()
        _clean_axes(ax)
        if ax.legend_:
            ax.legend_.remove()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"ROI variability vs mean ({PRIMARY_METHOD})")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = _figure_path(figures_dir, "09_std_vs_mean_scatter")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_method_delta(summary: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    required = {"BAYES", "BAYES-fits0"}
    if not required.issubset(set(summary["method"])):
        return None
    sub = summary[summary["method"].isin(required)].pivot_table(
        index=["subject", "parameter", "roi"], columns="method", values="mean", aggfunc="first"
    ).dropna()
    if sub.empty:
        return None
    sub["delta_BAYES_fits0_minus_BAYES"] = sub["BAYES-fits0"] - sub["BAYES"]
    frame = sub.reset_index()
    rois = ordered_rois(frame["roi"])
    fig, axes = plt.subplots(1, len(config.PARAMS), figsize=(4.4 * len(config.PARAMS), 4.2))
    for ax, param in zip(np.ravel(axes), config.PARAMS):
        psub = frame[frame["parameter"] == param]
        sns.boxplot(data=psub, x="roi", y="delta_BAYES_fits0_minus_BAYES", order=rois, color="0.82", ax=ax)
        sns.stripplot(data=psub, x="roi", y="delta_BAYES_fits0_minus_BAYES", order=rois, hue="subject", dodge=False, size=4, ax=ax)
        ax.axhline(0, color="0.35", linewidth=1)
        ax.set_title(param)
        ax.set_xlabel("")
        ax.set_ylabel("BAYES-fits0 - BAYES")
        _format_roi_xticks(ax, rois)
        if ax.legend_:
            ax.legend_.remove()
        _clean_axes(ax)
    fig.suptitle("Mean change from BAYES to BAYES-fits0")
    fig.tight_layout()
    out = _figure_path(figures_dir, "10_bayes_fits0_delta")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _method_dir(subject: str, method: str) -> Path:
    return Path(config.MADI_ROOT) / f"sub-{subject}" / "dwi" / f"method-{method}"


def _auxiliary_map_path(subject: str, method: str, name: str) -> Path:
    filenames = {
        "residual": "residual.nii.gz" if method.startswith("BAYES") else "residual_map.nii.gz",
        "n_eff": "n_eff.nii.gz",
    }
    return _method_dir(subject, method) / filenames[name]


def _load_auxiliary_map(subject: str, method: str, name: str, shape: Tuple[int, ...]) -> np.ndarray:
    path = _auxiliary_map_path(subject, method, name)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.asarray(nib.load(path).dataobj)
    if data.shape != shape:
        raise ValueError(f"{name} shape {data.shape} does not match parameter map shape {shape}: {path}")
    return data


def _reference_roi_order(rois: Iterable[str]) -> List[str]:
    found = set(rois)
    return [roi for roi in REFERENCE_ROIS if roi in found]


def build_bayesian_reference_qc(subjects: Sequence[str]) -> pd.DataFrame:
    """Summarize Bayesian uncertainty and fit diagnostics in reference ROIs only."""
    rows = []
    for subject in subjects:
        roi_paths = _roi_mask_files(subject)
        roi_paths = {roi: path for roi, path in roi_paths.items() if roi in REFERENCE_ROIS}
        if not roi_paths:
            continue
        for method in REFERENCE_BAYES_METHODS:
            for param in config.PARAMS:
                mean_path = loaders.param_map_path(subject, method, param)
                std_path = _method_dir(subject, method) / f"{param}_std.nii.gz"
                if not os.path.exists(mean_path) or not std_path.exists():
                    continue
                mean_map = loaders.load_param_map(subject, method, param)
                std_map = np.asarray(nib.load(std_path).dataobj)
                brain_mask = loaders.load_brain_mask(subject, mean_map.shape)
                if std_map.shape != mean_map.shape:
                    raise ValueError(f"posterior std shape does not match mean map: {std_path}")
                try:
                    residual = _load_auxiliary_map(subject, method, "residual", mean_map.shape)
                    n_eff = _load_auxiliary_map(subject, method, "n_eff", mean_map.shape)
                except (FileNotFoundError, ValueError) as error:
                    print(f"[reference-qc] skip sub-{subject} {method} {param}: {error}")
                    continue
                for roi, mask_path in roi_paths.items():
                    mask = load_mask(mask_path, mean_map.shape)
                    selected = mask & brain_mask
                    values = mean_map[selected]
                    uncertainty = std_map[selected]
                    residual_values = residual[selected]
                    n_eff_values = n_eff[selected]
                    valid = np.isfinite(values) & np.isfinite(uncertainty) & np.isfinite(residual_values) & np.isfinite(n_eff_values)
                    values, uncertainty = values[valid], uncertainty[valid]
                    residual_values, n_eff_values = residual_values[valid], n_eff_values[valid]
                    relative_uncertainty = uncertainty / np.maximum(np.abs(values), np.finfo(float).eps)
                    rows.append(
                        {
                            "subject": subject,
                            "method": method,
                            "fit_s0": method == "BAYES-fits0",
                            "parameter": param,
                            "roi": roi,
                            "roi_label": ROI_LABELS[roi],
                            "n_voxels": int(values.size),
                            "mean_parameter": float(np.mean(values)) if values.size else np.nan,
                            "mean_posterior_std": float(np.mean(uncertainty)) if uncertainty.size else np.nan,
                            "median_posterior_std": float(np.median(uncertainty)) if uncertainty.size else np.nan,
                            "median_relative_posterior_std": float(np.median(relative_uncertainty)) if relative_uncertainty.size else np.nan,
                            "p90_relative_posterior_std": float(np.quantile(relative_uncertainty, 0.90)) if relative_uncertainty.size else np.nan,
                            "median_residual": float(np.median(residual_values)) if residual_values.size else np.nan,
                            "median_n_eff": float(np.median(n_eff_values)) if n_eff_values.size else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def build_reference_method_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    """Subject-level ROI agreement metrics for the requested method pairs."""
    frames = []
    for method_a, method_b in (("BAYES-fits0", "BAYES"), ("BAYES", "MAP")):
        left = summary[summary["method"] == method_a]
        right = summary[summary["method"] == method_b]
        merged = left.merge(
            right,
            on=["subject", "parameter", "roi", "roi_label"],
            suffixes=("_a", "_b"),
        )
        if merged.empty:
            continue
        comparison_mean = (merged["mean_a"] + merged["mean_b"]) / 2.0
        merged["method_a"] = method_a
        merged["method_b"] = method_b
        merged["mean_a_minus_b"] = merged["mean_a"] - merged["mean_b"]
        merged["mean_of_methods"] = comparison_mean
        merged["percent_difference_of_mean"] = 100.0 * merged["mean_a_minus_b"] / comparison_mean.replace(0, np.nan)
        frames.append(
            merged[
                [
                    "subject", "parameter", "roi", "roi_label", "method_a", "method_b",
                    "mean_a", "mean_b", "median_a", "median_b", "std_a", "std_b",
                    "n_voxels_a", "n_voxels_b", "mean_a_minus_b", "mean_of_methods",
                    "percent_difference_of_mean",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _subject_style(subjects: Sequence[str]) -> Tuple[Dict[str, tuple], Dict[str, str]]:
    palette = sns.color_palette("colorblind", n_colors=max(len(subjects), 3))
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    return (
        {subject: palette[index] for index, subject in enumerate(subjects)},
        {subject: markers[index % len(markers)] for index, subject in enumerate(subjects)},
    )


def _paired_roi_plot(ax, frame: pd.DataFrame, value: str, rois: Sequence[str], colors: Dict[str, tuple]) -> None:
    x_positions = np.arange(len(rois))
    for subject, subject_frame in frame.groupby("subject", sort=True):
        ordered = subject_frame.set_index("roi").reindex(rois)
        values = ordered[value].to_numpy(dtype=float)
        valid = np.isfinite(values)
        ax.plot(x_positions[valid], values[valid], color=colors[subject], alpha=0.65, linewidth=1.1, zorder=2)
        ax.scatter(x_positions[valid], values[valid], color=colors[subject], s=34, edgecolor="white", linewidth=0.5, zorder=3, label=f"sub-{subject}")
    medians = frame.groupby("roi")[value].median().reindex(rois)
    for index, median in enumerate(medians):
        if np.isfinite(median):
            ax.hlines(median, index - 0.22, index + 0.22, color="black", linewidth=1.7, zorder=4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([ROI_LABELS[roi] for roi in rois], rotation=22, ha="right")
    _clean_axes(ax)


def plot_reference_roi_means(summary: pd.DataFrame, figures_dir: Path) -> List[Path]:
    rois = _reference_roi_order(summary["roi"])
    primary = summary[(summary["method"] == PRIMARY_METHOD) & summary["roi"].isin(rois)]
    if primary.empty or len(rois) < 2:
        return []
    subjects = sorted(primary["subject"].unique())
    colors, _ = _subject_style(subjects)
    fig, axes = plt.subplots(1, len(config.PARAMS), figsize=(4.2 * len(config.PARAMS), 4.4), sharex=True)
    for ax, param in zip(np.ravel(axes), config.PARAMS):
        panel = primary[primary["parameter"] == param]
        _paired_roi_plot(ax, panel, "mean", rois, colors)
        ax.set_title(config.PARAM_LABELS[param])
        ax.set_ylabel("ROI mean")
    handles, labels = axes[0].get_legend_handles_labels()
    deduplicated = dict(zip(labels, handles))
    fig.legend(deduplicated.values(), deduplicated.keys(), loc="lower center", ncol=len(subjects), frameon=False)
    fig.suptitle(f"Reference ROI means ({PRIMARY_METHOD}); black bars are subject medians", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    out = _figure_path(figures_dir, "01_reference_roi_primary_means")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return [out]


def plot_reference_posterior_uncertainty(qc: pd.DataFrame, figures_dir: Path) -> List[Path]:
    primary = qc[qc["method"] == PRIMARY_METHOD].copy()
    rois = _reference_roi_order(primary["roi"])
    if primary.empty:
        return []
    primary["relative_uncertainty_percent"] = 100.0 * primary["median_relative_posterior_std"]
    subjects = sorted(primary["subject"].unique())
    colors, _ = _subject_style(subjects)
    fig, axes = plt.subplots(1, len(config.PARAMS), figsize=(4.2 * len(config.PARAMS), 4.4), sharex=True)
    for ax, param in zip(np.ravel(axes), config.PARAMS):
        panel = primary[primary["parameter"] == param]
        _paired_roi_plot(ax, panel, "relative_uncertainty_percent", rois, colors)
        ax.set_title(config.PARAM_LABELS[param])
        ax.set_ylabel("Median posterior SD / mean (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    deduplicated = dict(zip(labels, handles))
    fig.legend(deduplicated.values(), deduplicated.keys(), loc="lower center", ncol=len(subjects), frameon=False)
    fig.suptitle(f"Posterior relative uncertainty by reference ROI ({PRIMARY_METHOD})", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    out = _figure_path(figures_dir, "02_reference_roi_posterior_uncertainty")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return [out]


def plot_reference_fit_quality(qc: pd.DataFrame, figures_dir: Path) -> List[Path]:
    primary = qc[(qc["method"] == PRIMARY_METHOD) & (qc["parameter"] == config.PARAMS[0])]
    rois = _reference_roi_order(primary["roi"])
    if primary.empty:
        return []
    subjects = sorted(primary["subject"].unique())
    colors, _ = _subject_style(subjects)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), sharex=True)
    for ax, value, label in zip(
        axes,
        ("median_n_eff", "median_residual"),
        ("Median effective library atoms", "Median residual (within-method QC)"),
    ):
        _paired_roi_plot(ax, primary, value, rois, colors)
        ax.set_yscale("log")
        ax.set_ylabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    deduplicated = dict(zip(labels, handles))
    fig.legend(deduplicated.values(), deduplicated.keys(), loc="lower center", ncol=len(subjects), frameon=False)
    fig.suptitle(f"Fit-quality diagnostics in reference ROIs ({PRIMARY_METHOD})", fontsize=13)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    out = _figure_path(figures_dir, "03_reference_roi_fit_quality")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return [out]


def _agreement_scatter(ax, frame: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str) -> None:
    for roi in _reference_roi_order(frame["roi"]):
        panel = frame[frame["roi"] == roi]
        ax.scatter(panel[x], panel[y], s=58, color=ROI_COLORS[roi], edgecolor="white", linewidth=0.6, label=ROI_LABELS[roi])
    values = frame[[x, y]].to_numpy(dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    pad = (hi - lo) * 0.06 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.35", linewidth=1.0, zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _clean_axes(ax)


def plot_method_agreement(
    comparisons: pd.DataFrame,
    method_a: str,
    method_b: str,
    figures_dir: Path,
    stem: str,
) -> List[Path]:
    frame = comparisons[(comparisons["method_a"] == method_a) & (comparisons["method_b"] == method_b)]
    if frame.empty:
        return []
    fig, axes = plt.subplots(2, len(config.PARAMS), figsize=(4.2 * len(config.PARAMS), 7.4))
    for column, param in enumerate(config.PARAMS):
        panel = frame[frame["parameter"] == param]
        _agreement_scatter(axes[0, column], panel, "mean_b", "mean_a", f"{method_b} ROI mean", f"{method_a} ROI mean")
        axes[0, column].set_title(config.PARAM_LABELS[param])
        axes[1, column].scatter(panel["mean_of_methods"], panel["percent_difference_of_mean"], s=58, c=[ROI_COLORS[roi] for roi in panel["roi"]], edgecolor="white", linewidth=0.6)
        bias = panel["percent_difference_of_mean"].mean()
        spread = panel["percent_difference_of_mean"].std(ddof=1)
        axes[1, column].axhline(0, color="0.35", linewidth=1.0)
        axes[1, column].axhline(bias, color="0.15", linestyle="--", linewidth=1.0)
        if np.isfinite(spread):
            axes[1, column].axhline(bias + 1.96 * spread, color="0.5", linestyle=":", linewidth=0.9)
            axes[1, column].axhline(bias - 1.96 * spread, color="0.5", linestyle=":", linewidth=0.9)
        axes[1, column].set_xlabel("Mean of methods")
        axes[1, column].set_ylabel(f"100 x ({method_a} - {method_b}) / mean")
        _clean_axes(axes[1, column])
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=ROI_COLORS[roi], label=ROI_LABELS[roi]) for roi in _reference_roi_order(frame["roi"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(f"Reference-ROI agreement: {method_a} versus {method_b}\nDashed: mean difference; dotted: descriptive +/-1.96 SD", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    out = _figure_path(figures_dir, stem)
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return [out]


def plot_bayesian_uncertainty_comparison(qc: pd.DataFrame, figures_dir: Path) -> List[Path]:
    frame = qc[qc["method"].isin(REFERENCE_BAYES_METHODS)]
    if frame.empty:
        return []
    fig, axes = plt.subplots(1, len(config.PARAMS) + 1, figsize=(4.1 * (len(config.PARAMS) + 1), 4.0))
    for ax, param in zip(axes[: len(config.PARAMS)], config.PARAMS):
        panel = frame[frame["parameter"] == param].pivot_table(
            index=["subject", "roi"], columns="method", values="median_relative_posterior_std", aggfunc="first"
        ).dropna()
        panel = panel.reset_index()
        _agreement_scatter(ax, panel, "BAYES", "BAYES-fits0", "BAYES relative posterior SD", "BAYES-fits0 relative posterior SD")
        ax.set_title(config.PARAM_LABELS[param])
    n_eff = frame[frame["parameter"] == config.PARAMS[0]].pivot_table(
        index=["subject", "roi"], columns="method", values="median_n_eff", aggfunc="first"
    ).dropna().reset_index()
    _agreement_scatter(axes[-1], n_eff, "BAYES", "BAYES-fits0", "BAYES median n_eff", "BAYES-fits0 median n_eff")
    axes[-1].set_title("Effective library atoms")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=ROI_COLORS[roi], label=ROI_LABELS[roi]) for roi in _reference_roi_order(frame["roi"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle("Bayesian uncertainty and effective-library-size comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    out = _figure_path(figures_dir, "05_bayes_fits0_vs_bayes_uncertainty")
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return [out]


def make_reference_roi_figures(summary: pd.DataFrame, qc: pd.DataFrame, comparisons: pd.DataFrame, figures_dir: Path) -> List[Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    outputs: List[Path] = []
    outputs.extend(plot_reference_roi_means(summary, figures_dir))
    outputs.extend(plot_reference_posterior_uncertainty(qc, figures_dir))
    outputs.extend(plot_reference_fit_quality(qc, figures_dir))
    outputs.extend(plot_method_agreement(comparisons, "BAYES-fits0", "BAYES", figures_dir, "04_bayes_fits0_vs_bayes_agreement"))
    outputs.extend(plot_bayesian_uncertainty_comparison(qc, figures_dir))
    outputs.extend(plot_method_agreement(comparisons, "BAYES", "MAP", figures_dir, "06_bayes_vs_map_agreement"))
    return outputs


def make_figures(summary: pd.DataFrame, overlap: pd.DataFrame, subjects: Sequence[str], figures_dir: Path) -> List[Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    outputs: List[Path] = []
    outputs.extend(plot_mean_bars(summary, figures_dir))
    outputs.extend(plot_median_subject_lines(summary, figures_dir))
    for maybe in [
        plot_edema_contra_scatter(summary, figures_dir),
        plot_method_correlation(summary, figures_dir),
        plot_voxel_count_heatmap(summary, figures_dir),
        plot_brain_montage(subjects, figures_dir),
        plot_roi_overlay_montage(subjects, figures_dir),
        plot_roi_overlap(summary, overlap, figures_dir),
        plot_std_vs_mean(summary, figures_dir),
        plot_method_delta(summary, figures_dir),
    ]:
        if maybe is not None:
            outputs.append(maybe)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject ids without sub- prefix. Default: discover subjects with ROI masks.")
    parser.add_argument("--methods", nargs="*", default=config.ALL_METHODS, help="Fitting methods to summarize.")
    parser.add_argument("--params", nargs="*", default=config.PARAMS, help="MADI parameters to summarize.")
    parser.add_argument("--output-dir", default=os.path.join(config.MADI_ROOT, "summaries"), help="Directory for CSVs and figures.")
    parser.add_argument(
        "--reference-roi-analysis",
        action="store_true",
        help="Analyze only grey matter, white matter, and contralateral reference ROIs; write Bayesian QC and method-comparison figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = args.subjects if args.subjects is not None else discover_subjects()
    if not subjects:
        raise SystemExit(f"No subjects with DWI-space ROI masks found under {config.ROIS_ROOT}")

    output_dir = Path(args.output_dir)
    if args.reference_roi_analysis:
        output_dir = output_dir / "reference_rois"
    tables_dir = output_dir / "csv"
    figures_dir = output_dir / "figures"

    allowed_rois = REFERENCE_ROIS if args.reference_roi_analysis else None
    summary = build_summary(subjects, args.methods, args.params, allowed_rois=allowed_rois)
    if summary.empty:
        raise SystemExit("No parameter/ROI combinations were summarized.")
    overlap = build_overlap(subjects, allowed_rois=allowed_rois)
    save_csvs(summary, overlap, tables_dir)
    if args.reference_roi_analysis:
        qc = build_bayesian_reference_qc(subjects)
        comparisons = build_reference_method_comparisons(summary)
        _write_csv(qc, tables_dir / "madi_reference_roi_bayesian_qc.csv")
        _write_csv(comparisons, tables_dir / "madi_reference_roi_method_comparison.csv")
        figures = make_reference_roi_figures(summary, qc, comparisons, figures_dir)
    else:
        figures = make_figures(summary, overlap, subjects, figures_dir)

    print(f"subjects: {', '.join(subjects)}")
    print(f"summary rows: {len(summary)}")
    print(f"csv directory: {tables_dir}")
    print(f"figures written: {len(figures)}")
    for path in figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
