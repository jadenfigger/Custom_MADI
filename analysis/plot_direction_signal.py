#!/usr/bin/env python3
"""Signal and two-point ADC against b-value, one series per gradient direction.

Takes one diffusion-weighted acquisition per gradient direction (``posx``,
``negx``, ``posy``, ``negy``, ``posz``, ``negz``) and writes two figures:

``signal_by_bvalue.png``
    The absolute (un-normalized) signal against b-value for every direction.
``adc_by_bvalue.png``
    A two-point ADC at every diffusion-weighted b-value, from that direction's
    own ``S0`` and that single b-value only: ``ADC(b) = -ln(S(b)/S0) / b``.

plus ``direction_signal_table.csv`` with every plotted number.

Inputs
------
Each input is a NIfTI (``.nii`` / ``.nii.gz``) with an FSL-style ``.bval``
sidecar of the same stem, or pass ``--bvals`` when every file shares one list.
The direction is read from the filename (``posx``, ``neg_z``, ``x_pos``,
``+y`` ...); a file with no recognisable direction is labelled by its stem.

Signal is the mean over a region of interest: ``--mask`` if given, otherwise
every voxel whose mean ``b = 0`` intensity in the first file exceeds
``--b0-fraction`` of that image's 99th percentile, applied to all files.
Volumes whose b-values agree within ``--b-tol`` are one shell and are averaged;
``b < --b0-max`` is ``S0``.  The ADC is computed from the ROI-mean signal, not
voxelwise, so high-b noise does not put the log of a negative number into the
average.

Usage
-----
    python analysis/plot_direction_signal.py data/posx.nii.gz data/negx.nii.gz \\
        data/posy.nii.gz data/negy.nii.gz data/posz.nii.gz data/negz.nii.gz \\
        --output-dir figures/direction_signal
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from madi.volume_pathology import apparent_diffusion_coefficient  # noqa: E402

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"
# Hue encodes the gradient axis, marker encodes the sign, so identity never
# rests on colour alone.  Slots follow the reference palette's fixed order.
AXIS_COLOUR = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
FALLBACK_COLOURS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIRECTION_PATTERNS = (
    re.compile(r"(?i)(?<![a-z])(pos|neg)[_-]?([xyz])(?![a-z])"),
    re.compile(r"(?i)(?<![a-z])([xyz])[_-]?(pos|neg)(?![a-z])"),
    re.compile(r"(?i)([+-])([xyz])(?![a-z])"),
)
DIRECTION_ORDER = ["posx", "negx", "posy", "negy", "posz", "negz"]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "text.color": INK_PRIMARY, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.edgecolor": AXIS_RULE, "axes.linewidth": 0.6,
        "grid.color": GRIDLINE, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "legend.frameon": False, "figure.dpi": 160,
    })


def _stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_direction(path: Path) -> str | None:
    """`posx` / `negz` ... from a filename, or None if it names no direction."""
    stem = _stem(path)
    for index, pattern in enumerate(DIRECTION_PATTERNS):
        match = pattern.search(stem)
        if not match:
            continue
        if index == 0:
            sign, axis = match.group(1), match.group(2)
        elif index == 1:
            axis, sign = match.group(1), match.group(2)
        else:
            sign, axis = ("pos" if match.group(1) == "+" else "neg"), match.group(2)
        return f"{sign.lower()}{axis.lower()}"
    return None


def load_bvals(path: Path, override: list[float] | None, n_volumes: int) -> np.ndarray:
    if override is not None:
        bvals = np.asarray(override, dtype=float)
    else:
        sidecar = path.with_name(_stem(path) + ".bval")
        if not sidecar.exists():
            raise SystemExit(f"{path.name}: no {sidecar.name} beside it; pass --bvals")
        bvals = np.loadtxt(sidecar).ravel()
    if bvals.size != n_volumes:
        raise SystemExit(f"{path.name}: {n_volumes} volumes but {bvals.size} b-values")
    return bvals


def group_shells(bvals: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Indices of volumes per shell, merging b-values within `tolerance`, ascending."""
    order = np.argsort(bvals, kind="stable")
    shells, current = [], [order[0]]
    for index in order[1:]:
        if bvals[index] - bvals[current[0]] <= tolerance:
            current.append(index)
        else:
            shells.append(np.asarray(current))
            current = [index]
    shells.append(np.asarray(current))
    return shells


def default_mask(image: np.ndarray, bvals: np.ndarray, b0_max: float, fraction: float) -> np.ndarray:
    b0 = bvals < b0_max
    if not b0.any():
        raise SystemExit("the first file has no b = 0 volume to build a default mask from; pass --mask")
    mean_b0 = image[..., b0].mean(axis=-1)
    return mean_b0 > fraction * np.percentile(mean_b0, 99)


def summarise(path: Path, mask: np.ndarray, args) -> dict:
    image = np.asarray(nib.load(path).dataobj, dtype=float)
    if image.ndim != 4:
        raise SystemExit(f"{path.name}: expected a 4-D image, got shape {image.shape}")
    if image.shape[:3] != mask.shape:
        raise SystemExit(f"{path.name}: grid {image.shape[:3]} does not match the mask {mask.shape}")
    bvals = load_bvals(path, args.bvals, image.shape[-1])
    roi = image[mask]                                          # (voxels, volumes)
    shells = []
    for indices in group_shells(bvals, args.b_tol):
        volume_means = roi[:, indices].mean(axis=0)
        shells.append({"b": float(np.mean(bvals[indices])), "n_volumes": int(indices.size),
                       "signal": float(volume_means.mean()),
                       "signal_sd_across_voxels": float(roi[:, indices].mean(axis=1).std())})
    b0_shells = [s for s in shells if s["b"] < args.b0_max]
    if not b0_shells:
        raise SystemExit(f"{path.name}: no b < {args.b0_max:g} volume, so no S0 for the ADC")
    s0 = float(np.average([s["signal"] for s in b0_shells], weights=[s["n_volumes"] for s in b0_shells]))
    for shell in shells:
        if shell["b"] < args.b0_max:
            shell["adc_um2_per_ms"] = float("nan")
            continue
        # The two-point ADC: S0 and this one b-value only.
        shell["adc_um2_per_ms"] = float(apparent_diffusion_coefficient(
            np.array([[shell["signal"] / s0]]), np.array([shell["b"]]))[0])
    direction = parse_direction(path)
    return {"label": direction or _stem(path), "direction": direction, "path": path,
            "s0": s0, "shells": shells}


def _series_style(series: dict, fallback_index: int) -> dict:
    direction = series["direction"]
    if direction is None:
        colour = FALLBACK_COLOURS[fallback_index % len(FALLBACK_COLOURS)]
        return {"color": colour, "marker": "D", "markerfacecolor": colour}
    colour = AXIS_COLOUR[direction[-1]]
    filled = direction.startswith("pos")
    return {"color": colour, "marker": "o" if filled else "s",
            "markerfacecolor": colour if filled else SURFACE}


def _ordered(all_series: list[dict]) -> list[dict]:
    rank = {name: index for index, name in enumerate(DIRECTION_ORDER)}
    return sorted(all_series, key=lambda s: (rank.get(s["direction"], len(rank)), s["label"]))


def plot(all_series: list[dict], key: str, ylabel: str, title: str, path: Path, *,
         skip_b0: bool, b0_max: float, log_y: bool) -> None:
    _style()
    figure, ax = plt.subplots(figsize=(7.6, 4.6))
    fallback = 0
    for series in _ordered(all_series):
        shells = [s for s in series["shells"] if not (skip_b0 and s["b"] < b0_max)]
        b = np.array([s["b"] for s in shells])
        y = np.array([s[key] for s in shells])
        keep = np.isfinite(y) & ((y > 0) if log_y else True)
        style = _series_style(series, fallback)
        fallback += series["direction"] is None
        ax.plot(b[keep], y[keep], linewidth=1.8, markersize=6, markeredgewidth=1.4,
                label=series["label"], zorder=3, **style)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"b-value (s/mm$^2$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11)
    ax.grid(True, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(title="direction", loc="best", fontsize=8, title_fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="one 4-D NIfTI per gradient direction")
    parser.add_argument("--bvals", type=float, nargs="+", default=None,
                        help="one b-value list shared by every file, instead of .bval sidecars")
    parser.add_argument("--mask", type=Path, default=None, help="ROI mask on the images' grid")
    parser.add_argument("--b0-fraction", type=float, default=0.10,
                        help="default mask: mean b=0 above this fraction of its 99th percentile")
    parser.add_argument("--b0-max", type=float, default=50.0, help="b below this is S0 (s/mm^2)")
    parser.add_argument("--b-tol", type=float, default=25.0, help="b-values within this are one shell (s/mm^2)")
    parser.add_argument("--log-y", action="store_true", help="log-scale the signal axis")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to the first file's directory")
    args = parser.parse_args()

    if args.mask is not None:
        mask = np.asarray(nib.load(args.mask).dataobj) > 0
        mask_source = str(args.mask)
    else:
        first = np.asarray(nib.load(args.files[0]).dataobj, dtype=float)
        mask = default_mask(first, load_bvals(args.files[0], args.bvals, first.shape[-1]),
                            args.b0_max, args.b0_fraction)
        mask_source = f"mean b=0 of {args.files[0].name} > {args.b0_fraction:g} x its 99th percentile"
    if not mask.any():
        raise SystemExit("the ROI mask is empty")
    print(f"ROI: {int(mask.sum()):,} voxels ({mask_source})")

    all_series = [summarise(path, mask, args) for path in args.files]
    output = args.output_dir or args.files[0].parent
    output.mkdir(parents=True, exist_ok=True)

    plot(all_series, "signal", "mean ROI signal (a.u.)", "Absolute signal against b-value, by direction",
         output / "signal_by_bvalue.png", skip_b0=False, b0_max=args.b0_max, log_y=args.log_y)
    plot(all_series, "adc_um2_per_ms", r"two-point ADC ($\times 10^{-3}$ mm$^2$/s)",
         r"ADC from S$_0$ and each b-value, by direction",
         output / "adc_by_bvalue.png", skip_b0=True, b0_max=args.b0_max, log_y=False)

    table = output / "direction_signal_table.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["direction", "file", "b_s_mm2", "n_volumes", "mean_signal",
                         "signal_sd_across_voxels", "s0", "adc_1e-3_mm2_per_s"])
        for series in _ordered(all_series):
            for shell in series["shells"]:
                writer.writerow([series["label"], series["path"].name, f"{shell['b']:g}", shell["n_volumes"],
                                 f"{shell['signal']:.6g}", f"{shell['signal_sd_across_voxels']:.6g}",
                                 f"{series['s0']:.6g}",
                                 "" if not np.isfinite(shell["adc_um2_per_ms"]) else f"{shell['adc_um2_per_ms']:.6g}"])
    print(f"wrote {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
