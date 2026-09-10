#!/usr/bin/env python3
"""Phase 3 item 3.3: the structural figures, drawn from the stored Phase-3 maps.

Plan section 3.3 asks for "the `(rho, V)` plane with the sloppy eigenvector drawn
as a short line segment at each node, overlaid on constant-`v_i` hyperbolae. If
the hypothesis holds, the alignment is visible at a glance. The figure is the
argument."

This is a pure read-out layer.  It computes no Fisher quantity: every number it
draws was written by `scripts/run_fisher_phase3.py` into `maps_*.npy`, which in
turn used only `madi.fisher_crlb`.  It also writes the per-`(rho, V)` aggregates
as CSV, so every figure has a table beside it.

The band-aligned frame, and why it is not a distortion
------------------------------------------------------
The library's `(rho, V)` mask band is `0.40 <= rho*V*1e-6 <= 0.99`, which in
`(log rho, log V)` is a diagonal strip roughly 0.28 decades thick and 4.2
decades long.  Drawn on square log-log axes at equal aspect it is a hairline and
nothing is legible; drawn at unequal aspect the segment angles on the page are
no longer the angles in the data, which is the one thing this figure exists to
show.  So the plane is **rigidly rotated** by 45 degrees:

    u = (log10 rho - log10 V) / sqrt(2)      along the constant-v_i hyperbola
    w = (log10 rho + log10 V) / sqrt(2)      across it, i.e. v_i itself

A rotation is orthogonal, so every angle and every length is exactly preserved,
and the panels stay at equal aspect.  The constant-`v_i` direction `(1, -1)`
becomes horizontal, so the hypothesis under test reads off the page directly:
**if the degeneracy runs along constant-`v_i` hyperbolae, the segments are
horizontal.**  Lines of constant `rho` are drawn and labelled as the second
reference family, so the physical coordinates stay readable.  The band is then
cut into equal strips stacked down the page, which is the only way to give a
23:1 strip enough height to read an angle in.

Figures
-------
`fig3_3_sloppy_direction_field_<domain>_kio<k>.png`
    The structural figure.  The `k_io`-profiled `(log rho, log V)` sloppy
    direction as a segment at every node, coloured by its angle to the
    constant-`v_i` direction.
`fig3_1_spectrum_maps_<domain>.png`
    Median over the `k_io` grid: the profiled sloppy angle, and the condition
    number `lambda_1 / lambda_3` of `D F D`.
`fig3_4_out_of_plane_<domain>.png`
    Median `|k_io|` component of the three-parameter sloppy eigenvector: how far
    the full problem's least-determined direction leaves the `(rho, V)` plane.
`fig3_2_angle_distributions.png`
    Small multiples, one panel per declared domain: the distribution of the
    profiled sloppy angle against the uniform null a random direction gives.

Colour follows the data's job.  Every scale here is a magnitude, so every ramp
is one hue light-to-dark, never multi-hue; where two sequential contexts share a
figure the second takes the next categorical slot's hue as its own one-hue ramp.
Palette values are the documented reference instance, used unmodified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Reference palette, light surface.  Sequential blue is the documented ramp,
# steps 100 -> 700.  Orange is the documented categorical slot 2 (#eb6834)
# extended into its own one-hue ramp, which is what the palette prescribes for a
# second sequential context on the same figure.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGE_RAMP = ["#fce4d6", "#f9cbb1", "#f5b18e", "#f19a6c", "#ee8050", "#eb6834", "#d75c2b",
               "#c05023", "#a8441c", "#8f3816", "#752c10", "#5c220b", "#431806"]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SEQUENTIAL_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)

ROOT2 = math.sqrt(2.0)
V_I_TICKS = (0.5, 0.6, 0.7, 0.8)
RHO_DECADES = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)


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


def rotate_point(log_rho: np.ndarray, log_V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(log10 rho, log10 V)` -> the band-aligned frame `(u, w)`."""
    return (log_rho - log_V) / ROOT2, (log_rho + log_V) / ROOT2


def rotate_direction(direction: np.ndarray) -> np.ndarray:
    """A direction in `(log rho, log V)` -> the same direction in `(u, w)`.

    The transform is the same rotation, applied to a difference rather than a
    point, so it preserves the angle between any two directions.  The
    constant-`v_i` direction `(1, -1)/sqrt(2)` maps to `(1, 0)`: horizontal.
    """
    direction = np.asarray(direction, dtype=float)
    return np.stack([(direction[:, 0] - direction[:, 1]) / ROOT2,
                     (direction[:, 0] + direction[:, 1]) / ROOT2], axis=1)


class Phase3Maps:
    """Read-only accessor for one Phase-3 run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.report = json.loads((self.run_dir / "phase3_report.json").read_text(encoding="utf-8"))
        self.nodes = np.load(self.run_dir / "evaluation_nodes.npy")
        labels = np.load(self.run_dir / "evaluation_node_labels.npy")
        self.rho, self.V, self.kio = labels[:, 0], labels[:, 1], labels[:, 2]
        self.v_i = self.rho * self.V * 1e-6
        self.u, self.w = rotate_point(np.log10(self.rho), np.log10(self.V))

    def domains(self) -> list[str]:
        return [name for name, entry in self.report["domains"].items() if "empty" not in entry]

    def map(self, domain: str, name: str) -> np.ndarray:
        return np.load(self.run_dir / f"maps_{domain}.{name}.npy")

    def profiled_valid(self, domain: str) -> np.ndarray:
        """Nodes carrying a `k_io`-profiled block at all, i.e. those with `F_kk > 0`.

        This is deliberately the SAME population `run_fisher_phase3.geometry_report`
        aggregates over, so a figure and the report it accompanies can never quote
        different medians.  It is not narrowed to nodes where the profiled block is
        positive definite: the least-determined direction is defined whenever the
        block exists, and dropping the indefinite nodes would select on the outcome
        -- those are exactly the most degenerate nodes, and their angles are larger.
        """
        return np.isfinite(self.map(domain, "profiled_sloppy_angle_deg"))


def _band_limits(maps: Phase3Maps, pad: float = 0.02) -> tuple[float, float]:
    return float(maps.w.min()) - pad, float(maps.w.max()) + pad


def _dress_strip(ax, maps: Phase3Maps, u_lo: float, u_hi: float, w_lim: tuple[float, float],
                 label_rho: bool) -> None:
    """Constant-`v_i` horizontals and labelled constant-`rho` diagonals."""
    for v in V_I_TICKS:
        ax.axhline(math.log10(v * 1e6) / ROOT2, color=GRIDLINE, linewidth=0.6, zorder=0)
    for decade in RHO_DECADES:
        # log10 rho = (u + w)/sqrt(2) = decade  ->  u = sqrt(2)*decade - w
        w_axis = np.asarray(w_lim)
        u_axis = ROOT2 * decade - w_axis
        if u_axis.max() < u_lo or u_axis.min() > u_hi:
            continue
        ax.plot(u_axis, w_axis, color=GRIDLINE, linewidth=0.6, zorder=0)
        if label_rho:
            u_top = ROOT2 * decade - w_lim[1]
            if u_lo <= u_top <= u_hi:
                ax.annotate(rf"$\rho\!=\!10^{{{decade:g}}}$", xy=(u_top, w_lim[1]),
                            xytext=(0, 3), textcoords="offset points", color=INK_MUTED,
                            fontsize=7, ha="center", va="bottom", annotation_clip=False)
    ax.set_xlim(u_lo, u_hi)
    ax.set_ylim(*w_lim)
    ax.set_aspect("equal")
    ax.set_yticks([math.log10(v * 1e6) / ROOT2 for v in V_I_TICKS])
    ax.set_yticklabels([f"{v:g}" for v in V_I_TICKS])
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)


def figure_direction_field(maps: Phase3Maps, domain: str, k_io: float, out: Path,
                           strips: int = 3) -> Path:
    _style()
    slice_mask = np.isclose(maps.kio, k_io)
    if not slice_mask.any():
        raise SystemExit(f"no evaluation node at k_io = {k_io} s^-1")
    angle = maps.map(domain, "profiled_sloppy_angle_deg")
    vector = maps.map(domain, "profiled_sloppy_vector")
    ok = slice_mask & maps.profiled_valid(domain)
    unbounded = slice_mask & ~ok

    w_lim = _band_limits(maps)
    u_lo, u_hi = float(maps.u.min()) - 0.06, float(maps.u.max()) + 0.06
    edges = np.linspace(u_lo, u_hi, strips + 1)
    rotated = rotate_direction(vector)
    half = 0.030

    # Equal aspect fixes the box shape, so the axes are placed explicitly rather
    # than left to a layout engine that would shrink them to fit.
    span_u = (u_hi - u_lo) / strips
    span_w = w_lim[1] - w_lim[0]
    left, right, top_pad, bottom_pad, gap = 0.055, 0.895, 1.35, 0.85, 0.30
    width_in = 13.0 * (right - left)
    height_in = width_in * span_w / span_u
    figure_height = strips * height_in + (strips - 1) * gap + top_pad + bottom_pad
    figure = plt.figure(figsize=(13.0, figure_height))
    axes = []
    for index in range(strips):
        bottom_in = bottom_pad + (strips - 1 - index) * (height_in + gap)
        axes.append(figure.add_axes([left, bottom_in / figure_height,
                                     right - left, height_in / figure_height]))
    axes = np.asarray(axes, dtype=object)
    collection = None
    for index, ax in enumerate(axes):
        lo, hi = edges[index], edges[index + 1]
        _dress_strip(ax, maps, lo, hi, w_lim, label_rho=True)
        window = (maps.u >= lo - 0.05) & (maps.u <= hi + 0.05)
        blank = unbounded & window
        if blank.any():
            ax.plot(maps.u[blank], maps.w[blank], "o", markersize=2.6, markerfacecolor="none",
                    markeredgecolor=INK_MUTED, markeredgewidth=0.6, zorder=2)
        draw = ok & window
        if not draw.any():
            continue
        unit = rotated[draw] / np.linalg.norm(rotated[draw], axis=1, keepdims=True)
        offset = unit * half
        starts = np.stack([maps.u[draw] - offset[:, 0], maps.w[draw] - offset[:, 1]], axis=1)
        ends = np.stack([maps.u[draw] + offset[:, 0], maps.w[draw] + offset[:, 1]], axis=1)
        collection = LineCollection(np.stack([starts, ends], axis=1), cmap=SEQUENTIAL_BLUE,
                                    norm=Normalize(0, 45), linewidths=1.6, capstyle="round",
                                    zorder=3)
        collection.set_array(angle[draw])
        ax.add_collection(collection)
    axes[-1].set_xlabel("along the constant-$v_i$ hyperbola  "
                        r"($u=\log_{10}(\rho/V)/\sqrt{2}$, band cut into "
                        f"{strips} strips top to bottom)")
    for ax in axes:
        ax.set_ylabel("$v_i$", rotation=0, labelpad=10, va="center")

    bar_bottom = bottom_pad / figure_height
    bar_height = (strips * height_in + (strips - 1) * gap) / figure_height
    bar_axes = figure.add_axes([right + 0.018, bar_bottom, 0.012, bar_height])
    bar = figure.colorbar(collection, cax=bar_axes)
    bar.set_label("angle of the least-determined direction\nto the constant-$v_i$ hyperbola (deg)",
                  color=INK_SECONDARY, fontsize=8)
    bar.outline.set_visible(False)

    median = float(np.nanmedian(angle[ok]))
    below = float(np.mean(angle[ok] < 10.0))
    identifiable = maps.map(domain, "positive_definite").astype(bool)
    figure.text(0.008, 1.0 - 0.42 / figure_height,
                "In the $(\\rho, V)$ plane the degeneracy runs along the constant-$v_i$ "
                "hyperbolae — the segments are horizontal",
                color=INK_PRIMARY, fontsize=12, ha="left", va="top")
    figure.text(0.008, 1.0 - 0.72 / figure_height,
                f"{domain} · $k_{{io}}$ = {k_io:g} s$^{{-1}}$ · least-determined direction of the "
                f"$(\\log\\rho,\\log V)$ block with $k_{{io}}$ profiled out · "
                f"median angle {median:.1f}$\\degree$, {below * 100:.0f}% below 10$\\degree$ "
                f"(a random direction would give 45$\\degree$)",
                color=INK_SECONDARY, fontsize=8.5, ha="left", va="top")
    missing = int(unbounded.sum())
    figure.text(0.008, 1.0 - 0.94 / figure_height,
                f"{int((slice_mask & identifiable).sum())} of {int(slice_mask.sum())} nodes here "
                "are identifiable in all three parameters"
                + (f"; open circles are the {missing} carrying no profiled block at all"
                   if missing else "; every node carries a profiled block")
                + " · the frame is a rigid 45$\\degree$ rotation of "
                "$(\\log_{10}\\rho, \\log_{10}V)$, so every drawn angle is the true angle",
                color=INK_SECONDARY, fontsize=8.5, ha="left", va="top")
    path = out / f"fig3_3_sloppy_direction_field_{domain}_kio{k_io:g}.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _median_over_kio(maps: Phase3Maps, values: np.ndarray, valid: np.ndarray):
    """Collapse the `k_io` axis by median, per `(rho, V)` node."""
    keys = maps.nodes[:, 0].astype(np.int64) * 10_000 + maps.nodes[:, 1]
    order = np.argsort(keys, kind="stable")
    out_u, out_w, out_v, out_n, out_row = [], [], [], [], []
    for start, stop in _runs(keys[order]):
        rows = order[start:stop]
        rows = rows[valid[rows] & np.isfinite(values[rows])]
        if rows.size == 0:
            continue
        out_u.append(maps.u[rows[0]])
        out_w.append(maps.w[rows[0]])
        out_v.append(float(np.median(values[rows])))
        out_n.append(int(rows.size))
        out_row.append(int(rows[0]))
    return (np.asarray(out_u), np.asarray(out_w), np.asarray(out_v),
            np.asarray(out_n), np.asarray(out_row))


def _runs(sorted_keys: np.ndarray):
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(sorted_keys)]])
    return zip(starts, stops)


def _band_scatter(ax, maps: Phase3Maps, u, w, values, cmap, norm, w_lim):
    for v in V_I_TICKS:
        ax.axhline(math.log10(v * 1e6) / ROOT2, color=GRIDLINE, linewidth=0.6, zorder=0)
    for decade in RHO_DECADES:
        w_axis = np.asarray(w_lim)
        ax.plot(ROOT2 * decade - w_axis, w_axis, color=GRIDLINE, linewidth=0.6, zorder=0)
        u_top = ROOT2 * decade - w_lim[1]
        ax.annotate(rf"$\rho\!=\!10^{{{decade:g}}}$", xy=(u_top, w_lim[1]), xytext=(0, 3),
                    textcoords="offset points", color=INK_MUTED, fontsize=7, ha="center",
                    va="bottom", annotation_clip=False)
    points = ax.scatter(u, w, c=values, s=30, cmap=cmap, norm=norm, linewidths=0, zorder=3)
    ax.set_ylim(*w_lim)
    ax.set_yticks([math.log10(v * 1e6) / ROOT2 for v in V_I_TICKS])
    ax.set_yticklabels([f"{v:g}" for v in V_I_TICKS])
    ax.set_xticks([])
    ax.set_ylabel("$v_i$", rotation=0, labelpad=10, va="center")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    return points


def figure_spectrum_maps(maps: Phase3Maps, domain: str, out: Path) -> tuple[Path, Path]:
    _style()
    angle = maps.map(domain, "profiled_sloppy_angle_deg")
    condition = maps.map(domain, "condition_number")
    identifiable = maps.map(domain, "positive_definite").astype(bool)
    w_lim = _band_limits(maps, pad=0.03)

    figure, axes = plt.subplots(2, 1, figsize=(12.6, 5.2))
    u1, w1, angle_median, angle_count, rows = _median_over_kio(maps, angle, maps.profiled_valid(domain))
    points = _band_scatter(axes[0], maps, u1, w1, angle_median, SEQUENTIAL_BLUE,
                           Normalize(0, 45), w_lim)
    bar = figure.colorbar(points, ax=axes[0], pad=0.012, fraction=0.020, aspect=14)
    bar.set_label("median angle to\nconstant-$v_i$ (deg)", color=INK_SECONDARY, fontsize=8)
    bar.outline.set_visible(False)
    axes[0].set_title("A   Where the hyperbola hypothesis holds, and where it does not",
                      loc="left", color=INK_PRIMARY, fontsize=10, pad=18)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_condition = np.log10(condition)
    u2, w2, condition_median, condition_count, _ = _median_over_kio(maps, log_condition, identifiable)
    points = _band_scatter(axes[1], maps, u2, w2, condition_median, SEQUENTIAL_ORANGE,
                           Normalize(float(np.floor(condition_median.min())),
                                     float(np.ceil(condition_median.max()))), w_lim)
    bar = figure.colorbar(points, ax=axes[1], pad=0.012, fraction=0.020, aspect=14)
    bar.set_label(r"median $\log_{10}$" "\n" r"$\lambda_1/\lambda_3$ of $DFD$",
                  color=INK_SECONDARY, fontsize=8)
    bar.outline.set_visible(False)
    axes[1].set_title("B   How badly conditioned the three-parameter problem is",
                      loc="left", color=INK_PRIMARY, fontsize=10, pad=18)
    axes[1].set_xlabel(r"along the constant-$v_i$ hyperbola  ($u=\log_{10}(\rho/V)/\sqrt{2}$)")

    figure.suptitle(f"Fisher/CRLB Phase 3 — {domain}, median over the interior $k_{{io}}$ grid",
                    x=0.008, ha="left", color=INK_PRIMARY, fontsize=12, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / f"fig3_1_spectrum_maps_{domain}.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    condition_lookup = {int(a): (b, c) for a, b, c in
                        zip(_median_over_kio(maps, log_condition, identifiable)[4],
                            condition_median, condition_count)}
    table = out / f"table3_1_rho_V_medians_{domain}.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rho_cells_per_uL", "V_pL", "v_i",
                         "median_profiled_sloppy_angle_deg", "k_io_nodes_with_a_profiled_block",
                         "median_log10_condition_number", "k_io_nodes_identifiable"])
        for row, value, count in zip(rows, angle_median, angle_count):
            other = condition_lookup.get(int(row), ("", ""))
            writer.writerow([f"{maps.rho[row]:.6g}", f"{maps.V[row]:.6g}", f"{maps.v_i[row]:.6f}",
                             f"{value:.4f}", count,
                             f"{other[0]:.4f}" if other[0] != "" else "", other[1]])
    return path, table


def figure_out_of_plane(maps: Phase3Maps, domain: str, out: Path) -> Path:
    _style()
    fraction = maps.map(domain, "sloppy_k_io_fraction")
    identifiable = maps.map(domain, "positive_definite").astype(bool)
    w_lim = _band_limits(maps, pad=0.03)
    figure, ax = plt.subplots(1, 1, figsize=(12.6, 2.9))
    u, w, values, counts, _ = _median_over_kio(maps, fraction, identifiable)
    points = _band_scatter(ax, maps, u, w, values, SEQUENTIAL_BLUE, Normalize(0, 1), w_lim)
    bar = figure.colorbar(points, ax=ax, pad=0.012, fraction=0.020, aspect=12)
    bar.set_label("median $|k_{io}|$ component", color=INK_SECONDARY, fontsize=8)
    bar.outline.set_visible(False)
    ax.set_xlabel(r"along the constant-$v_i$ hyperbola  ($u=\log_{10}(\rho/V)/\sqrt{2}$)")
    figure.suptitle("The full three-parameter problem's least-determined direction points mostly "
                    "out of the $(\\rho, V)$ plane, along $k_{io}$",
                    x=0.008, ha="left", color=INK_PRIMARY, fontsize=12, y=0.995)
    figure.text(0.008, 0.90,
                f"{domain} · $|k_{{io}}|$ component of the unit sloppy eigenvector of $DFD$, median "
                f"over the interior $k_{{io}}$ grid · 1.0 is a direction that is pure $k_{{io}}$, "
                "0.0 one that lies wholly in the plane",
                color=INK_SECONDARY, fontsize=8, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    path = out / f"fig3_4_out_of_plane_{domain}.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_angle_distributions(maps: Phase3Maps, out: Path) -> Path:
    _style()
    domains = maps.domains()
    columns = 3
    rows = int(np.ceil(len(domains) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(11.5, 2.2 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    edges = np.arange(0, 91, 3.0)
    for ax, domain in zip(axes, domains):
        angle = maps.map(domain, "profiled_sloppy_angle_deg")
        values = angle[maps.profiled_valid(domain)]
        ax.hist(values, bins=edges, color=BLUE_RAMP[7], edgecolor=SURFACE, linewidth=0.4,
                density=True, zorder=3)
        ax.axhline(1.0 / 90.0, color=INK_MUTED, linewidth=0.9, zorder=2)
        ax.axvline(float(np.median(values)), color=INK_PRIMARY, linewidth=1.0, zorder=4)
        ax.annotate(f"median {np.median(values):.1f}$\\degree$\n"
                    f"{np.mean(values < 10) * 100:.0f}% below 10$\\degree$",
                    xy=(0.97, 0.92), xycoords="axes fraction", ha="right", va="top",
                    color=INK_SECONDARY, fontsize=7.5)
        ax.set_title(domain, loc="left", color=INK_PRIMARY, fontsize=8, pad=4)
        ax.set_yscale("log")
        ax.set_xlim(0, 90)
        ax.set_xticks([0, 30, 60, 90])
        ax.grid(axis="y", zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[len(domains):]:
        ax.set_visible(False)
    for index in range(len(domains) - columns, len(domains)):
        axes[index].set_xlabel("angle to constant-$v_i$  (deg)")
    for index in range(0, len(axes), columns):
        axes[index].set_ylabel("density (log)")
    figure.suptitle("The $k_{io}$-profiled sloppy direction concentrates on the constant-$v_i$ "
                    "hyperbola under every declared domain",
                    x=0.008, ha="left", color=INK_PRIMARY, fontsize=12, y=0.998)
    figure.text(0.008, 0.955,
                "horizontal rule: the flat density a uniformly random direction would give · "
                "vertical rule: the median",
                color=INK_SECONDARY, fontsize=8, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "fig3_2_angle_distributions.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="a Phase-3 output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", default="full_stored_domain",
                        help="which declared domain the structural figures are drawn for")
    parser.add_argument("--k-io", type=float, default=10.0,
                        help="the k_io slice the (rho, V)-plane field is drawn at, in s^-1")
    parser.add_argument("--strips", type=int, default=3,
                        help="how many strips the band is cut into in the structural figure")
    args = parser.parse_args()

    maps = Phase3Maps(args.run_dir)
    if args.domain not in maps.domains():
        raise SystemExit(f"{args.domain!r} is not a non-empty domain in this run; "
                         f"available: {', '.join(maps.domains())}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = [figure_direction_field(maps, args.domain, args.k_io, args.output_dir, args.strips)]
    written.extend(figure_spectrum_maps(maps, args.domain, args.output_dir))
    written.append(figure_out_of_plane(maps, args.domain, args.output_dir))
    written.append(figure_angle_distributions(maps, args.output_dir))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
