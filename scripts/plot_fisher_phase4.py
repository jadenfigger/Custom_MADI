#!/usr/bin/env python3
"""Phase 4 figures, drawn from the outputs of `scripts/run_fisher_phase4.py`.

A pure read-out layer, like the Phase-3 plotter: it computes no Fisher or
fitting quantity.  Every voxel array it draws was written by the Phase-4 runner
into `maps/`, or by a fit arm into its own directory; every summary number comes
from `phase4_report.json`.

Figures
-------
`fig4_1_adc_vs_volume.png`
    The characterization the pathology was named from: fitted cell volume
    against ADC, voxelwise, with the Jackson-thesis 20 pL cutoff and the
    free-water ADC marked.  MAP and Bayes side by side.  The ADC axis is drawn
    over a declared display range; voxels outside it are counted in the panel,
    and no statistic in the report excludes them.
`fig4_2_condition_arms.png`
    Hypotheses H4 and H3 (plan items 4.1, 4.2): the fraction of voxels above a
    volume cutoff, across the whole cutoff sweep, for the baseline fit and each
    arm that switches one condition on.
`fig4_3_residual_discriminator.png`
    Plan item 4.3, H1 against H2: residual over the voxel's own noise floor, for
    the pathological voxels and the rest.
`fig4_4_ridge.png`
    Plan item 4.4, H1: why the pathological voxels' nodes carry no Fisher matrix
    at this acquisition, and the direction estimates moved in when a fit was
    perturbed, as cumulative distributions against both the uniform null and the
    geometric null that permutes voxels within the same pair of fits.  The band
    is long and thin, so only the geometric null says whether a move is along the
    ridge rather than along the mask.
`fig4_5_replacement.png`
    Plan item 4.5: how many voxels can report `v_i` with a bound, across the
    estimability-defect tolerances (none nominated), and the Bayes posterior
    spread against the CRLB where a per-parameter CRLB exists at all.

Colour follows the data's job: densities are one hue light-to-dark; two-group
comparisons and arm comparisons are categorical, in the reference palette's
fixed slot order, and every series carries a direct label or a legend, never
colour alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
# Categorical slots, in the reference palette's fixed order.  Arms keep their
# colour in every figure; the two-group comparisons use slots 1 and 2.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ARM_COLOUR = {"baseline": SLOT[0], "fit_s0": SLOT[1], "trust_floor_column": SLOT[2],
              "trust_floor_candidate": SLOT[3]}
ARM_LABEL = {"baseline": "baseline", "fit_s0": "S0 fitted (H4)",
             "trust_floor_column": "trust floor, columns (H3)",
             "trust_floor_candidate": "trust floor, candidates (H3)"}
REST, BLOWN = SLOT[0], SLOT[1]
THESIS_CUTOFF = 20.0
FREE_WATER_ADC = 3.0
# Display range for the ADC axis only.  Voxels outside it are counted on the
# panel; the report's statistics use every voxel.
ADC_DISPLAY_UM2_PER_MS = (-0.5, 4.5)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "text.color": INK_PRIMARY, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.edgecolor": AXIS_RULE, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "grid.color": GRIDLINE, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "legend.frameon": False, "figure.dpi": 160,
    })


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


class Phase4Outputs:
    """Read-only access to one Phase-4 run and the fit arms it analysed."""

    def __init__(self, run_dir: Path, fit_root: Path, mask: Path) -> None:
        self.run_dir = Path(run_dir)
        self.fit_root = Path(fit_root)
        self.report = json.loads((self.run_dir / "phase4_report.json").read_text(encoding="utf-8"))
        self.mask = np.asarray(nib.load(mask).dataobj).astype(bool)

    def map(self, stem: str) -> np.ndarray:
        return np.asarray(nib.load(self.run_dir / "maps" / f"{stem}.nii.gz").dataobj, dtype=float)[self.mask]

    def arm(self, arm: str, stem: str) -> np.ndarray | None:
        path = self.fit_root / arm / f"{stem}.nii.gz"
        if not path.exists():
            return None
        return np.asarray(nib.load(path).dataobj, dtype=float)[self.mask]


def figure_adc_vs_volume(out: Phase4Outputs, directory: Path) -> Path:
    _style()
    adc = out.map("adc_b_le_1000")
    low, high = ADC_DISPLAY_UM2_PER_MS
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    for ax, (arm, v_stem, rho_stem, title) in zip(axes, (("baseline_map", "V_map", "rho_map", "MAP"),
                                                         ("baseline_bayes", "V_mean", "rho_mean",
                                                          "Bayes posterior mean"))):
        volume = out.arm(arm, v_stem)
        rho = out.arm(arm, rho_stem)
        keep = np.isfinite(adc) & np.isfinite(volume) & (volume > 0) & (rho > 0)
        shown = keep & (adc >= low) & (adc <= high)
        hidden = int(np.count_nonzero(keep & ~shown))
        density = ax.hexbin(adc[shown], np.log10(volume[shown]), gridsize=(60, 45), mincnt=1, bins="log",
                            cmap=SEQUENTIAL_BLUE, linewidths=0, extent=(low, high, -1.5, 2.1))
        ax.set_xlim(low, high)
        ax.axhline(np.log10(THESIS_CUTOFF), color=INK_PRIMARY, linewidth=0.9)
        ax.annotate("20 pL, Jackson-thesis cutoff", xy=(low, np.log10(THESIS_CUTOFF)), xytext=(4, 4),
                    textcoords="offset points", ha="left", va="bottom", color=INK_SECONDARY, fontsize=7.5)
        ax.axvline(FREE_WATER_ADC, color=INK_MUTED, linewidth=0.8)
        ax.annotate("free water", xy=(FREE_WATER_ADC, -1.5), xytext=(3, 3), textcoords="offset points",
                    color=INK_MUTED, fontsize=7.5)
        ax.annotate(f"{hidden:,} voxels ({100 * hidden / max(int(keep.sum()), 1):.1f}%) have ADC outside "
                    f"[{low:g}, {high:g}] and are not drawn", xy=(0.0, -0.17), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=7.5)
        fraction = out.report["arms"][arm]["fraction_above_thesis_cutoff"]
        ax.set_title(f"{title} · {fraction * 100:.1f}% of fitted voxels above 20 pL",
                     loc="left", color=INK_PRIMARY, fontsize=10)
        ax.set_xlabel(r"ADC, b $\leq$ 1000 s/mm$^2$  ($\mu$m$^2$/ms)")
        ax.grid(True, zorder=0)
        _clean(ax)
        bar = figure.colorbar(density, ax=ax, pad=0.015, fraction=0.04)
        bar.set_label("voxels", color=INK_SECONDARY, fontsize=8)
        bar.outline.set_visible(False)
    axes[0].set_ylabel(r"fitted cell volume, $\log_{10}$ pL")
    figure.suptitle("Fisher/CRLB Phase 4 — fitted cell volume against ADC, voxelwise",
                    x=0.008, ha="left", color=INK_PRIMARY, fontsize=12)
    figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    path = directory / "fig4_1_adc_vs_volume.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _place_end_labels(ax, endpoints, min_gap=0.055):
    """Direct labels at line ends, pushed apart vertically so none collide.

    Positions are resolved in axes-fraction space after the axis limits are
    final, so the separation holds whatever the log-scale range turns out to be.
    """
    if not endpoints:
        return
    to_axes = ax.transData + ax.transAxes.inverted()
    placed = sorted(((to_axes.transform((x, y))[1], x, y, text) for x, y, text in endpoints))
    heights = [item[0] for item in placed]
    for index in range(1, len(heights)):
        heights[index] = max(heights[index], heights[index - 1] + min_gap)
    for height, (_, x, y, text) in zip(heights, placed):
        anchor = to_axes.transform((x, y))
        ax.annotate(text, xy=(x, y), xytext=(anchor[0] + 0.012, height), textcoords="axes fraction",
                    color=INK_SECONDARY, fontsize=7, va="center")


def figure_condition_arms(out: Phase4Outputs, directory: Path) -> tuple[Path, Path]:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharey=True)
    rows = []
    endpoints: list[tuple[float, float, str]] = []
    for ax, method in zip(axes, ("map", "bayes")):
        for key in ("baseline", "fit_s0", "trust_floor_column", "trust_floor_candidate"):
            arm = out.report["arms"].get(f"{key}_{method}")
            if not arm:
                continue
            sweep = arm["cutoff_sweep"]
            cutoffs = np.array([r["cutoff_pL"] for r in sweep])
            fractions = np.array([100.0 * r["fraction_above"] for r in sweep])
            drawn = fractions > 0      # a log axis cannot place zero; the line ends there instead
            ax.plot(cutoffs[drawn], fractions[drawn], color=ARM_COLOUR[key], linewidth=2.0, marker="o",
                    markersize=4, label=ARM_LABEL[key], zorder=3)
            label = ARM_LABEL[key]
            if (~drawn).any():
                label += f"  (none above {cutoffs[~drawn].min():g} pL)"
            if drawn.any():
                endpoints.append((cutoffs[drawn][-1], fractions[drawn][-1], label))
            rows.extend({"method": method, "arm": key, **r} for r in sweep)
        ax.axvline(THESIS_CUTOFF, color=INK_MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(4.0, 400.0)
        ax.relim()
        ax.autoscale_view(scalex=False)
        _place_end_labels(ax, endpoints)
        endpoints.clear()
        ax.set_xlabel("volume cutoff (pL)")
        ax.set_title("MAP" if method == "map" else "Bayes posterior mean", loc="left",
                     color=INK_PRIMARY, fontsize=10)
        ax.grid(True, which="major", zorder=0)
        _clean(ax)
    axes[0].set_ylabel("fitted voxels above the cutoff (%)")
    axes[0].legend(loc="lower left", fontsize=7.5)
    figure.suptitle("Fisher/CRLB Phase 4 — does switching one condition on shrink the large-volume tail? "
                    "(H4, H3)", x=0.008, ha="left", color=INK_PRIMARY, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = directory / "fig4_2_condition_arms.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    table = directory / "table4_2_condition_arms_cutoff_sweep.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "arm", "cutoff_pL", "fraction_above", "count_above"])
        writer.writeheader()
        writer.writerows(rows)
    return path, table


def _step_density(ax, values, bins, colour, label):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.8, color=colour,
            label=f"{label} (n={values.size:,})", zorder=3)


def figure_residual_discriminator(out: Phase4Outputs, directory: Path) -> Path:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharey=True)
    bins = np.linspace(-2.5, 4.5, 90)
    for ax, method in zip(axes, ("map", "bayes")):
        ratio = out.map(f"gof_ratio_baseline_{method}")
        blown = out.map(f"blow_up_baseline_{method}") == 1
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log10(np.where(ratio > 0, ratio, np.nan))
        _step_density(ax, logged[~blown], bins, REST, "V <= 20 pL")
        _step_density(ax, logged[blown], bins, BLOWN, "V > 20 pL")
        ax.axvline(0.0, color=INK_PRIMARY, linewidth=0.9)
        ax.annotate("residual = noise floor", xy=(0.0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-4, -12), textcoords="offset points", ha="right", color=INK_SECONDARY, fontsize=7.5)
        stats = out.report["H1_vs_H2_residuals"][method]["goodness_of_fit_ratio"]
        ax.set_title(("MAP" if method == "map" else "Bayes") +
                     f" · P(blow-up ratio > rest ratio) = "
                     f"{stats['probability_of_superiority_inside_over_outside']:.2f}",
                     loc="left", color=INK_PRIMARY, fontsize=10)
        ax.set_xlabel(r"$\log_{10}$ (residual / expected noise residual)")
        ax.grid(True, zorder=0)
        ax.legend(loc="upper right", fontsize=7.5)
        _clean(ax)
    axes[0].set_ylabel("density")
    figure.suptitle("Fisher/CRLB Phase 4 — are the large-volume voxels fitted worse than noise allows? "
                    "(H1 low, H2 high)", x=0.008, ha="left", color=INK_PRIMARY, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = directory / "fig4_3_residual_discriminator.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_ridge(out: Phase4Outputs, directory: Path) -> Path:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.0, 1.15]})

    reasons = out.report["H1_overlay"]["why_no_fisher_matrix"]
    categories = [("interior_node", "node has a Fisher matrix"),
                  ("rho_V_band_edge_only", "(rho, V) band edge"),
                  ("k_io_end_node_only", "k_io grid end"),
                  ("both", "both")]
    positions = np.arange(len(categories))
    height = 0.38
    for shift, group, colour, label in ((-height / 2, "rest", REST, "V <= 20 pL"),
                                        (height / 2, "blow_up", BLOWN, "V > 20 pL")):
        values = [100.0 * reasons[group].get(key, 0.0) for key, _ in categories]
        axes[0].barh(positions + shift, values, height=height, color=colour,
                     label=f"{label} (n={reasons[group]['voxels']:,})", zorder=3)
        for y, value in zip(positions + shift, values):
            axes[0].annotate(f"{value:.1f}%", xy=(value, y), xytext=(3, 0), textcoords="offset points",
                             va="center", color=INK_SECONDARY, fontsize=7.5)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([label for _, label in categories])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 115)
    axes[0].set_xlabel("share of voxels (%)")
    axes[0].set_title("A   where each voxel's node sits, at this acquisition", loc="left",
                      color=INK_PRIMARY, fontsize=9.5)
    axes[0].legend(loc="lower right", fontsize=7.5)
    axes[0].grid(True, axis="x", zorder=0)
    _clean(axes[0])

    labels = {"map_to_bayes": ("MAP mode -> Bayes mean", SLOT[0]),
              "baseline_map_to_fit_s0_map": ("-> S0 fitted", SLOT[1]),
              "baseline_map_to_trust_floor_column_map": ("-> trust floor, columns", SLOT[2]),
              "baseline_map_to_trust_floor_candidate_map": ("-> trust floor, candidates", SLOT[3])}
    def _ecdf(values):
        values = np.sort(values[np.isfinite(values)])
        return values, np.arange(1, values.size + 1) / max(values.size, 1)

    drew_null = False
    for stem, (label, colour) in labels.items():
        if not (out.run_dir / "maps" / f"displacement_angle_{stem}.nii.gz").exists():
            continue
        angles, cumulative = _ecdf(out.map(f"displacement_angle_{stem}"))
        if angles.size == 0:
            continue
        axes[1].plot(angles, cumulative, color=colour, linewidth=2.0, label=f"{label} (n={angles.size:,})", zorder=3)
        # The geometric null for the same pair of fits, in the same hue but thin and
        # light: the band's shape alone makes small angles, so this, not the
        # diagonal, is what an along-ridge move has to beat.
        null_path = out.run_dir / "maps" / f"displacement_angle_null_{stem}.nii.gz"
        if null_path.exists():
            null_angles, null_cumulative = _ecdf(out.map(f"displacement_angle_null_{stem}"))
            axes[1].plot(null_angles, null_cumulative, color=colour, linewidth=1.0, alpha=0.45, zorder=2)
            drew_null = True
    axes[1].plot([0, 90], [0, 1], color=INK_MUTED, linewidth=0.9, zorder=1)
    axes[1].annotate("uniform direction", xy=(62, 62 / 90), xytext=(4, -10), textcoords="offset points",
                     color=INK_MUTED, fontsize=7.5)
    if drew_null:
        axes[1].plot([], [], color=INK_MUTED, linewidth=1.0, alpha=0.45,
                     label="thin, same colour: voxels permuted (geometric null)")
    axes[1].set_xlim(0, 90)
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    axes[1].set_xlabel(r"angle of the $(\log\rho, \log V)$ move to the constant-$v_i$ hyperbola (deg)")
    axes[1].set_ylabel("cumulative share of moved voxels")
    axes[1].set_title("B   which way estimates moved when the fit was perturbed", loc="left",
                      color=INK_PRIMARY, fontsize=9.5)
    axes[1].legend(loc="lower right", fontsize=7.5)
    axes[1].grid(True, zorder=0)
    _clean(axes[1])
    figure.suptitle("Fisher/CRLB Phase 4 — the degeneracy ridge (H1)", x=0.008, ha="left",
                    color=INK_PRIMARY, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = directory / "fig4_4_ridge.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_replacement(out: Phase4Outputs, directory: Path) -> Path:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    replacement = out.report["item_4_5_replacement"]
    coverage = replacement["coverage_by_defect_tolerance"]
    positions = np.arange(len(coverage))
    ticks = ["strict (0)" if row["defect_tolerance"] == 0 else f"{row['defect_tolerance']:g}" for row in coverage]
    for key, colour, label in (("v_i_reportable_fraction_all", SLOT[0], "all fitted voxels"),
                               ("v_i_reportable_fraction_blow_up", SLOT[1], "voxels with V > 20 pL")):
        values = [100.0 * (row[key] if row[key] is not None else np.nan) for row in coverage]
        axes[0].plot(positions, values, color=colour, linewidth=2.0, marker="o", markersize=4, label=label, zorder=3)
        axes[0].annotate(label, xy=(positions[-1], values[-1]), xytext=(5, 0), textcoords="offset points",
                         color=INK_SECONDARY, fontsize=7.5, va="center")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(ticks)
    axes[0].set_xlim(-0.3, len(coverage) - 1 + 1.4)
    axes[0].set_xlabel("tolerated share of the v_i contrast in uninformative directions (defect)")
    axes[0].set_ylabel(r"voxels reporting $v_i$ with a bound (%)")
    axes[0].set_title("A   coverage of the v_i report per tolerance — none nominated", loc="left",
                      color=INK_PRIMARY, fontsize=9.5)
    axes[0].legend(loc="upper left", fontsize=7.5)
    axes[0].grid(True, zorder=0)
    _clean(axes[0])

    posterior = out.map("posterior_fractional_sd_V")
    crlb = out.map("crlb_log_V_at_bayes_node")
    keep = np.isfinite(posterior) & np.isfinite(crlb) & (posterior > 0) & (crlb > 0)
    x, y = np.log10(crlb[keep]), np.log10(posterior[keep])
    axes[1].scatter(x, y, s=16, color=SLOT[0], linewidths=0, alpha=0.85, zorder=3)
    if keep.any():
        span = [min(x.min(), y.min()) - 0.2, max(x.max(), y.max()) + 0.2]
        axes[1].plot(span, span, color=INK_MUTED, linewidth=0.9, zorder=2)
        axes[1].annotate("posterior SD = CRLB", xy=(span[1], span[1]), xytext=(-4, -12), textcoords="offset points",
                         ha="right", color=INK_MUTED, fontsize=7.5)
    validation = replacement["posterior_flag_validation"]["V"]
    total = int(out.report["arms"]["baseline_bayes"]["fitted_voxels"])
    axes[1].set_title(f"B   Bayes posterior spread vs CRLB, V · {int(keep.sum())} of {total:,} voxels have a "
                      f"CRLB · Spearman {validation.get('spearman_rank_correlation', float('nan')):.2f}",
                      loc="left", color=INK_PRIMARY, fontsize=9.5)
    axes[1].set_xlabel(r"$\log_{10}$ CRLB on $\log V$ at the voxel's node")
    axes[1].set_ylabel(r"$\log_{10}$ posterior SD(V) / mean(V)")
    axes[1].grid(True, zorder=0)
    _clean(axes[1])
    figure.suptitle("Fisher/CRLB Phase 4 — the principled replacement for the 20 pL cutoff (item 4.5)",
                    x=0.008, ha="left", color=INK_PRIMARY, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = directory / "fig4_5_replacement.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="a Phase-4 output directory")
    parser.add_argument("--fit-root", type=Path, required=True, help="the fit arms that run analysed")
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = Phase4Outputs(args.run_dir, args.fit_root, args.mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = [figure_adc_vs_volume(out, args.output_dir)]
    written.extend(figure_condition_arms(out, args.output_dir))
    written.append(figure_residual_discriminator(out, args.output_dir))
    written.append(figure_ridge(out, args.output_dir))
    written.append(figure_replacement(out, args.output_dir))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
