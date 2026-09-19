#!/usr/bin/env python3
"""Presentation figures: the universal MADI library and the Fisher/CRLB Phase 0-4 analyses.

Nine figures, each written to docs/figures/ as PNG (300 dpi) plus PDF:

  fisher_01_library_atlas             the (rho, V) band, its groups, k_io, representative nodes
  fisher_02_acquisition_atlas         the stored (delta, Delta) grid coloured by information
  fisher_03_fisher_anatomy            one node, three domains: F, correlation, spectrum, eigenvectors
  fisher_04_fisher_field              normalized Fisher matrices across the (rho, V) plane
  fisher_05_crlb_landscape            relative CRLB and kappa maps, model layer vs acquisition
  fisher_06_sloppy_direction_field    k_io-profiled sloppy directions on constant-v_i lines
  fisher_07_profiled_vs_full          profiled 2x2 geometry vs the full 3x3 sloppy eigenvector
  fisher_08_acquisition_design        single Delta -> +second Delta -> optimized pair -> full domain
  fisher_09_mc_debias                 size of the Var(J_hat) correction vs what it does to the CRLB

Run from the repository root (conda env `mri`):

    PYTHONPATH=. python -m scripts.plot_fisher_visualizations            # all nine
    PYTHONPATH=. python -m scripts.plot_fisher_visualizations --only 3 8  # a subset

or call one figure from Python: `plot_03_fisher_anatomy(FisherData())`.

Nothing here writes to an analysis output.  Inputs (see DEFAULT_PATHS in
scripts/fisher_viz_common.py): the universal library's small metadata members,
the Phase-1 manifest/stencil samples, the Phase-2 column cache, the Phase-2
N = 128 report and maps, and the Phase-3 report and maps.  The only derived
product that is saved is a plotting cache of figure 2's per-timing-pair pass
(~/.cache/madi_fisher_viz), safe to delete.

Conventions (docs/fisher_crlb_analysis_plan.md sections 2.2-2.4, fisher_phase3.md 1.2):
theta = (log rho, log V, k_io); D = diag(1, 1, max(k_io, 5 s^-1)); the 3x3
spectrum is of D F D; the k_io-profiled block is the (log rho, log V) Schur
complement; angles are acute; v_i = rho[cells/uL] * V[pL] * 1e-6.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

from madi.fisher_crlb import (CONSTANT_VI_DIRECTION, direction_angle_deg, fisher_spectrum,
                              gradient_strength_t_per_m, in_plane_direction_diagnostics,
                              nondimensionalized_fisher, pack_fisher, packed_inverse_diagonal)
from scripts.fisher_viz_common import (AXIS_RULE, CATEGORICAL, DEFAULT_PATHS, DIVERGING, GRIDLINE, INK,
                                       INK_2, INK_MUTED, NEUTRAL, PARAM_LABELS, SEQ_BLUE, SEQ_ORANGE,
                                       SURFACE, UNDEFINED_NO_STENCIL, UNDEFINED_NOT_PD, FisherData,
                                       band_pairs, crlb_kappa, despine, draw_cells, draw_marked_cells,
                                       draw_matrix, draw_profiled_ellipse, fisher_correlation, form_fisher,
                                       full_stored_domain, inset_colorbar, node_text, pair_pass, per_pair,
                                       phase2_arm, phase2_greedy_second_delta, save,
                                       select_lattice, setup_plane, style, title, undefined_handles)

# ===========================================================================
# CONFIGURATION -- edit here
# ===========================================================================

OUTPUT = dict(dir=DEFAULT_PATHS["figures"], dpi=300, vector="pdf")

# Representative nodes.  Rule: fisher_viz_common.select_lattice (geometric
# quantiles of the evaluation nodes; no Fisher result consulted).  k_io = 10 s^-1
# is the slice of the executed Phase-3 structural figure (fisher_phase3.md 3.2,
# plot_fisher_phase3.py --k-io 10), inside the k_io <= 30 region where Phase 1
# found dS/dk_io informative.  The lattice centre is the reference node.
NODES = dict(k_io=10.0, rho_quantiles=(0.10, 0.50, 0.90), vi_quantiles=(0.15, 0.50, 0.85))
NODE_LETTERS = "ABCDEFGHI"          # lattice cells in reading order; E is the reference node

# Gradient scenario for every Phase-2 acquisition drawn (figures 2, 3, 8, 9).
# A DISPLAY choice, not a nomination: the project reports clinical and research
# side by side and nominates neither (prereg gradient_scenario_choice: DEFERRED).
SCENARIO = "research"

FIG1 = dict(stem="fisher_01_library_atlas", figsize=(13.0, 7.4))
FIG2 = dict(stem="fisher_02_acquisition_atlas", figsize=(14.0, 6.6),
            # conditional annotation only: (scenario, b) -> "every b up to this value is playable"
            gradient_contours=(("research", 4000.0), ("research", 12000.0), ("clinical", 4000.0)),
            share_norm=(0.08, 4.0))
FIG3 = dict(stem="fisher_03_fisher_anatomy", figsize=(13.0, 11.5))
FIG4 = dict(stem="fisher_04_fisher_field", figsize=(14.0, 7.6), domain="full_stored_domain",
            normalization="correlation")          # or "dfd_over_lambda1"
FIG5 = dict(stem="fisher_05_crlb_landscape", figsize=(14.0, 10.2),
            # Model-layer CRLBs rescaled to the Phase-2 budget (N images spread over every stored column):
            # exact by Fisher linearity, so both rows can share one colour scale.  kappa is scale-free anyway.
            rescale_model_layer_to_budget=True,
            rows=(("full_stored_domain", "model layer · all stored DW columns"),
                  (f"phase2_optimum_{SCENARIO}_size8_m2", "conditional acquisition · Phase-2 two-Δ optimum")),
            crlb_norm=(0.01, 10.0), kappa_norm=(1.0, 1000.0), kappa_parameter=0)
FIG6 = dict(stem="fisher_06_sloppy_direction_field", figsize=(13.0, 7.4), domain="full_stored_domain",
            stride=(3, 2), segment_length=0.16, condition_norm=(1.0, 1000.0))
FIG7 = dict(stem="fisher_07_profiled_vs_full", figsize=(14.0, 8.4), domain="full_stored_domain")
FIG8 = dict(stem="fisher_08_acquisition_design", figsize=(14.5, 12.5), ellipse_limit=1.0)
FIG9 = dict(stem="fisher_09_mc_debias", figsize=(14.5, 9.6))


def _write(fig, cfg: dict) -> list:
    return save(fig, cfg["stem"], OUTPUT["dir"], dpi=OUTPUT["dpi"], vector=OUTPUT["vector"])


def _lattice_letters(selection: dict) -> dict:
    """node row -> letter, in reading order of the lattice."""
    return {row: NODE_LETTERS[i] for i, (_, row) in enumerate(sorted(selection["lattice"].items()))}


def _mark_nodes(ax, data: FisherData, selection: dict, size: float = 7.0, label: bool = True) -> None:
    letters = _lattice_letters(selection)
    lattice_row = {row: i for (i, _), row in selection["lattice"].items()}
    for row, letter in letters.items():
        x, y = np.log10(data.rho[row]), np.log10(data.V[row])
        reference = row == selection["reference"]
        ax.plot(x, y, "o", markersize=size + (2 if reference else 0), markerfacecolor=INK if reference else "white",
                markeredgecolor=SURFACE if reference else INK, markeredgewidth=1.2, zorder=8)
        if label:
            # label direction by lattice row (high / median / low v_i), so one column's labels never collide
            offset, ha, va = {0: ((-6, 4), "right", "bottom"), 1: ((7, 0), "left", "center"),
                              2: ((-6, -4), "right", "top")}[lattice_row[row]]
            ax.annotate(letter, (x, y), xytext=offset, textcoords="offset points", fontsize=8, ha=ha, va=va,
                        color=INK, fontweight="bold" if reference else "normal", zorder=9)


# ===========================================================================
# 1. Universal-library parameter-space atlas
# ===========================================================================

def plot_01_library_atlas(data: FisherData) -> list:
    """The stored (rho, V) groups coloured by v_i, with k_io as a compact strip.

    Source: canonical grid (madi.fisher_crlb.canonical_grid) and the Phase-2 node
    table (which groups carry a complete k = 1 stencil).  Domain: the library
    itself -- nothing conditional enters.  Representative nodes: select_lattice.
    """
    cfg = FIG1
    style()
    g = data.grid
    selection = select_lattice(data, **NODES)
    fig = plt.figure(figsize=cfg["figsize"])
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0), height_ratios=(0.8, 1.2), wspace=0.18, hspace=0.35)

    ax = fig.add_subplot(grid[:, 0])
    setup_plane(ax, data)
    rho_i, V_i = band_pairs(data, "retained")
    v_i = g["rhos"][rho_i] * g["Vs"][V_i] * 1e-6
    mesh = draw_cells(ax, data, rho_i, V_i, v_i, SEQ_BLUE, Normalize(*g["vi_band"]))
    edge_r, edge_v = band_pairs(data, "edge")
    draw_marked_cells(ax, data, edge_r, edge_v, dict(facecolor="none", edgecolor=SURFACE, hatch="////",
                                                       linewidth=0.0), zorder=3)
    _mark_nodes(ax, data, selection)
    inset_colorbar(ax, mesh, r"intracellular volume fraction $v_i=\rho V$", ticks=[0.4, 0.6, 0.8, 0.99])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=SEQ_BLUE(0.6), edgecolor=SURFACE, hatch="////",
                             label=f"band-edge group: no complete stencil, no Fisher matrix ({len(edge_r)})"),
                       Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK,
                              label=f"representative node (k_io = {selection['k_io']:g} s⁻¹)"),
                       Line2D([], [], marker="o", linestyle="", markerfacecolor=INK, markeredgecolor=SURFACE,
                              markersize=8, label="E: reference node (figures 3, 8, 9)")],
              loc="lower left", fontsize=7.5)
    title(ax, "The universal library is a band of constant-$v_i$ lines in the $(\\rho, V)$ plane",
          f"{len(rho_i)} (ρ, V) groups of a 64×64 log grid with {g['vi_band'][0]:.2f} ≤ vᵢ ≤ {g['vi_band'][1]:.2f}")

    # --- k_io: one compact strip -------------------------------------------
    ax_k = fig.add_subplot(grid[0, 1])
    kios = g["kios"]
    interior = np.isin(np.arange(len(kios)), np.unique(data.nodes[:, 2]))
    ax_k.axvspan(kios.min() - 2, 30, color=NEUTRAL, zorder=0)
    ax_k.vlines(kios[interior], 0.25, 0.75, color=INK_2, linewidth=0.9)
    ax_k.vlines(kios[~interior], 0.25, 0.75, color=AXIS_RULE, linewidth=0.9)
    ax_k.plot(selection["k_io"], 0.9, "v", color=INK, markersize=6)
    ax_k.text(selection["k_io"] + 2, 0.93, "slice used for the maps", fontsize=7.5, color=INK_2, va="center")
    ax_k.text(15, 0.08, "1 s⁻¹ steps", fontsize=7.5, color=INK_2, ha="center")
    ax_k.text(33, 0.08, "5 s⁻¹ steps · |∂S/∂k_io| collapses above ~30 s⁻¹ (Phase 1)", fontsize=7.5,
              color=INK_2, ha="left")
    ax_k.set_xlim(kios.min() - 2, kios.max() + 2)
    ax_k.set_ylim(0, 1.05)
    ax_k.set_yticks([])
    ax_k.set_xlabel(r"exchange rate $k_{io}$ (s$^{-1}$)")
    despine(ax_k, keep=("bottom",))
    title(ax_k, f"Every group is simulated at {len(kios)} $k_{{io}}$ values",
          f"dark ticks: {interior.sum()} interior values carrying a central k_io stencil; light: grid ends")

    # --- counts and the node key --------------------------------------------
    ax_t = fig.add_subplot(grid[1, 1])
    ax_t.axis("off")
    n_columns = len(data.pair_deltas) * data.n_b
    lines = [("library entries", f"{data.n_entries:,}",
              f"{len(rho_i)} groups × {len(kios)} k_io + 1 free-water atom"),
             ("stored columns per entry", f"{n_columns:,}",
              f"{len(data.pair_deltas):,} (δ, Δ) pairs × {data.n_b} b-values"),
             ("Fisher evaluation nodes", f"{len(data.rho):,}",
              f"{len(g['evaluated_pairs'])} interior groups × {interior.sum()} interior k_io")]
    y = 0.98
    for label, value, detail in lines:
        ax_t.text(0.0, y, value, fontsize=15, color=INK, va="top")
        ax_t.text(0.32, y - 0.005, label, fontsize=9, color=INK, va="top")
        ax_t.text(0.32, y - 0.075, detail, fontsize=7.5, color=INK_2, va="top")
        y -= 0.17
    ax_t.text(0.0, y - 0.02, f"Representative nodes at k_io = {selection['k_io']:g} s⁻¹", fontsize=9, color=INK, va="top")
    ax_t.text(0.0, y - 0.075, "columns: log ρ quantiles " + "/".join(f"{q:.0%}" for q in NODES["rho_quantiles"])
              + " · rows: log vᵢ quantiles " + "/".join(f"{q:.0%}" for q in NODES["vi_quantiles"][::-1])
              + " · nearest evaluation node", fontsize=7, color=INK_2, va="top")
    letters = _lattice_letters(selection)
    for (i, j), row in sorted(selection["lattice"].items()):
        ax_t.text(j * 0.34, y - 0.15 - i * 0.115,
                  f"{letters[row]}  ρ = {data.rho[row]:.2g} µL⁻¹\n     V = {data.V[row]:.2g} pL, vᵢ = {data.v_i[row]:.2f}",
                  fontsize=7.5, color=INK, va="top", linespacing=1.3,
                  fontweight="bold" if row == selection["reference"] else "normal")
    return _write(fig, cfg)


# ===========================================================================
# 2. Acquisition-space atlas
# ===========================================================================

def _pair_image(data: FisherData, values: np.ndarray):
    """(delta, Delta) image on the stored grid; Delta cells are 1 ms below 50 ms and 5 ms above."""
    deltas = np.unique(data.pair_deltas)
    Deltas = np.unique(data.pair_Deltas)
    image = np.full((len(Deltas), len(deltas)), np.nan)
    image[np.searchsorted(Deltas, data.pair_Deltas), np.searchsorted(deltas, data.pair_deltas)] = values
    x_edges = np.concatenate([deltas - 0.5, [deltas[-1] + 0.5]])
    mid = (Deltas[1:] + Deltas[:-1]) / 2
    y_edges = np.concatenate([[Deltas[0] - 0.5], mid, [Deltas[-1] + 2.5]])
    return x_edges, y_edges, np.ma.masked_invalid(image)


def plot_02_acquisition_atlas(data: FisherData) -> list:
    """Where informative measurements live in (delta, Delta).

    Source: fisher_viz_common.pair_pass -- one pass over the Phase-2 column cache
    through the audited `pair_contributions`, with the stored Phase-3
    full_stored_domain sloppy eigenpairs; Phase-2 optima from phase2_report.json.
    Aggregation: Fisher information is ADDITIVE over independent columns, so each
    timing pair's matrix is the sum over its 24 stored DW b-values; that is then
    summarized over all evaluation nodes.  Domain: model layer (no gradient,
    trust or Rician mask).  Gradient ceilings are drawn, never applied.
    """
    cfg = FIG2
    style()
    passed = pair_pass(data)
    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"], sharey=True)
    fraction = passed["identifiable_fraction"]
    share = passed["sloppy_share"] * len(data.pair_deltas)
    top = int(np.argmax(fraction))
    panels = [
        (fraction, SEQ_BLUE, Normalize(0.0, math.ceil(fraction.max() * 10) / 10),
         "share of tissue nodes with a CRLB",
         "A   No single timing gives a CRLB at most tissue nodes",
         f"one (δ, Δ) with all 24 stored b-values; best {fraction.max():.0%} at "
         f"({data.pair_deltas[top]:g}, {data.pair_Deltas[top]:g}) ms · free of SNR, T2 and averaging"),
        (share, SEQ_ORANGE, LogNorm(*cfg["share_norm"]),
         "share of sloppy-direction information (1 = average timing)",
         "B   Where the information on the least-determined direction lives",
         "median over identifiable nodes of v₃ᵀ(D F_pair D)v₃ / λ₃ (full stored domain), × 1,245 · TE/T2 weights"),
    ]
    for ax, (values, cmap, norm, bar_label, heading, sub) in zip(axes, panels):
        x_edges, y_edges, image = _pair_image(data, values)
        mesh = ax.pcolormesh(x_edges, y_edges, image, cmap=cmap, norm=norm, shading="flat", rasterized=True)
        bar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.13, fraction=0.045, aspect=40)
        bar.outline.set_visible(False)
        bar.set_label(bar_label, fontsize=8, color=INK_2)
        bar.ax.tick_params(labelsize=7.5)
        if isinstance(norm, LogNorm):
            bar.set_ticks([0.1, 0.3, 1, 3], labels=["0.1", "0.3", "1", "3"])
        # Conditional annotation: the lowest Delta at which a declared ceiling can
        # play every b up to b_max (contour of the audited gradient formula).
        d_fine, D_fine = np.meshgrid(np.linspace(1, 30, 300), np.linspace(1, 82, 400))
        for scenario, b_max in cfg["gradient_contours"]:
            g_max = float(data.prereg["gradient_limits_T_per_m"][scenario])
            G = gradient_strength_t_per_m(d_fine, D_fine, np.full_like(d_fine, b_max))
            G = np.where(D_fine >= d_fine, G, np.nan)
            ls_color = INK if scenario == "research" else INK_MUTED
            contour = ax.contour(d_fine, D_fine, G, levels=[g_max], colors=[ls_color],
                                 linewidths=1.1 if b_max < 10000 else 0.8)
            ax.clabel(contour, fmt={g_max: f"{g_max * 1e3:g} mT/m, b ≤ {b_max:,.0f}"}, fontsize=7, inline=True)
        _mark_phase2_optima(ax, data)
        ax.set_xlim(0.5, 30.5)
        ax.set_ylim(0.5, y_edges[-1])
        ax.set_xlabel("pulse duration δ (ms)")
        despine(ax)
        title(ax, heading, sub)
    axes[0].set_ylabel("diffusion time Δ (ms)")
    handles = [Line2D([], [], color=INK, linewidth=1.1, label="research ceiling: every b up to the label is playable above/right"),
               Line2D([], [], color=INK_MUTED, linewidth=1.1, label="clinical ceiling (same reading)"),
               Line2D([], [], marker="o", linestyle="", color=INK, markerfacecolor="white",
                      label="Phase-2 research optima: R1 one timing; both R2 markers form one acquisition"),
               Line2D([], [], marker="s", linestyle="", color=INK_MUTED, markerfacecolor="white",
                      label="Phase-2 clinical optima C1, C2 (reported separately, never pooled)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Acquisition atlas: 1,245 stored timing pairs, information summed over b and summarized over tissue",
                 x=0.01, ha="left", fontsize=12, color=INK)
    return _write(fig, cfg)


def _mark_phase2_optima(ax, data: FisherData) -> None:
    for scenario, marker, color in (("research", "o", INK), ("clinical", "s", INK_MUTED)):
        arms = data.phase2["scenarios"][scenario]["arms"]["8"]
        for arm in ("m1", "m2"):
            timing = np.asarray(arms[arm]["timing_pairs_ms"], dtype=float)
            ax.plot(timing[:, 0], timing[:, 1], linestyle="none", color=color, marker=marker,
                    markersize=6.5, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.3, zorder=6)
            for d, D in timing:
                ax.annotate(f"{scenario[0].upper()}{arm[1]}", (d, D), xytext=(5, -9), textcoords="offset points",
                            fontsize=7, color=color, zorder=7)


# ===========================================================================
# 3. Anatomy of the Fisher matrix
# ===========================================================================

def _oriented(vectors: np.ndarray) -> np.ndarray:
    """Eigenvector columns with the largest-magnitude component made positive (an eigenvector has no sign)."""
    out = np.array(vectors, dtype=float)
    for i in range(out.shape[1]):
        if out[np.argmax(np.abs(out[:, i])), i] < 0:
            out[:, i] *= -1.0
    return out


def _stage_lines(acquisition) -> str:
    if acquisition.layer == "model layer":
        return f"MODEL LAYER\n{acquisition.note}"
    return (f"CONDITIONAL ACQUISITION\n{acquisition.label}, {acquisition.n_columns} columns, "
            f"{acquisition.averages:g} avg each\n{acquisition.note}")


def _spectrum_panel(ax, eigenvalues: np.ndarray, reference: float, color: str, ylim=(1e-6, 3.0)) -> None:
    """Eigenvalues of D F D over `reference`, log axis; a non-positive one is a hollow marker at the floor."""
    for i, value in enumerate(np.asarray(eigenvalues) / reference):
        if value > 0:
            ax.vlines(i, ylim[0], value, color=color, linewidth=2)
            ax.plot(i, value, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        else:
            ax.plot(i, ylim[0] * 2, "v", markerfacecolor="white", markeredgecolor=color, markersize=8)
            ax.text(i, ylim[0] * 6, f"λ ≤ 0\n({value:+.1e})", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["λ₁ stiff", "λ₂", "λ₃ sloppy"])
    ax.grid(axis="y")
    despine(ax)


def plot_03_fisher_anatomy(data: FisherData) -> list:
    """One node, three domains: what the Fisher matrix holds and how acquisition changes it.

    Node: reference node E (select_lattice).  Domains: the executed Phase-2
    single-Delta and two-Delta optima (CONDITIONAL: scenario gradient mask, trust
    floor, Rician validity at their own averaging, N = 128, SNR 50) and the full
    stored domain (MODEL LAYER).  Source: matrices formed with the audited
    pair_contributions and checked against the stored Phase-3 maps; spectrum of
    D F D via madi.fisher_crlb.fisher_spectrum.  Each D F D is divided by its own
    lambda_1: eigenvalue SCALE is not comparable across domains of 8 and 29,880
    columns (fisher_phase3.md section 5), shape and direction are.
    """
    cfg = FIG3
    style()
    selection = select_lattice(data, **NODES)
    node = selection["reference"]
    kio_ref = data.kio_ref[[node]]
    stages = [("Best single Δ", phase2_arm(data, SCENARIO, "m1")),
              ("Optimized two Δ", phase2_arm(data, SCENARIO, "m2")),
              ("Full stored domain", full_stored_domain(data))]
    fig = plt.figure(figsize=cfg["figsize"])
    grid = fig.add_gridspec(4, 3, height_ratios=(1.0, 1.0, 0.8, 1.0), hspace=0.62, wspace=0.42,
                            top=0.81, bottom=0.13, left=0.1, right=0.97)
    signed = TwoSlopeNorm(0.0, -1.0, 1.0)
    row_names = ["D F D / λ₁", "correlation\nF_jk / √(F_jj F_kk)", "spectrum of D F D\n(÷ λ₁)",
                 "eigenvectors of D F D\n(rows; k_io in units of k_io,ref)"]
    for col, (heading, acquisition) in enumerate(stages):
        packed = form_fisher(data, acquisition, [node])["debiased"]
        spectrum = fisher_spectrum(packed, kio_ref)
        values = spectrum["eigenvalues"][0]
        vectors = _oriented(spectrum["eigenvectors"][0])
        first = col == 0

        ax = fig.add_subplot(grid[0, col])
        image = draw_matrix(ax, nondimensionalized_fisher(packed, kio_ref)[0] / values[0], DIVERGING, signed,
                            fmt="{:+.3f}", row_labels=PARAM_LABELS if first else None)
        ax.text(0.0, 1.08, _stage_lines(acquisition), transform=ax.transAxes, fontsize=7.5, color=INK_2,
                ha="left", va="bottom")
        ax.text(0.0, 1.46, heading, transform=ax.transAxes, fontsize=11, color=INK, ha="left", va="bottom")

        ax = fig.add_subplot(grid[1, col])
        draw_matrix(ax, fisher_correlation(packed)[0], DIVERGING, signed, row_labels=PARAM_LABELS if first else None,
                    blank_diagonal=True)

        ax = fig.add_subplot(grid[2, col])
        _spectrum_panel(ax, values, values[0], CATEGORICAL[0])
        if values[2] > 0:
            status = f"λ₁/λ₃ = {values[0] / values[2]:.3g}   λ₂/λ₃ = {values[1] / values[2]:.3g}"
        else:
            status = "λ₃ ≤ 0: debiased F not positive definite, no CRLB"
        ax.text(1.0, 1.04, status, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK)

        ax = fig.add_subplot(grid[3, col])
        draw_matrix(ax, vectors.T, DIVERGING, signed, row_labels=["stiff", "middle", "sloppy"] if first else None)
        sloppy = spectrum["sloppy_vector"]
        angle = float(direction_angle_deg(sloppy, CONSTANT_VI_DIRECTION)[0])
        k_share = float(in_plane_direction_diagnostics(sloppy)["k_io_fraction"][0])
        ax.text(0.5, -0.3, f"sloppy direction: {angle:.0f}° from constant $v_i$, |$k_{{io}}$| component {k_share:.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK)
        if first:
            for r, name in enumerate(row_names):
                fig.axes[-1 - (3 - r)].text(-0.42 if r != 2 else -0.3, 0.5, name, rotation=90, ha="center",
                                            va="center", fontsize=8, color=INK_2,
                                            transform=fig.axes[-1 - (3 - r)].transAxes)
    cax = fig.add_axes([0.38, 0.025, 0.26, 0.01])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7.5)
    bar.set_label("signed value (matrix rows 1, 2 and 4 share this scale)", fontsize=8, color=INK_2)
    bar.ax.xaxis.set_label_position("top")
    fig.suptitle(f"Anatomy of the Fisher matrix at reference node E   ({node_text(data, node)}, "
                 f"$k_{{io}}$ = {data.kio[node]:g} s⁻¹)", x=0.02, ha="left", y=0.985, fontsize=13, color=INK)
    fig.text(0.02, 0.955, "known S0, Monte-Carlo-debiased; θ = (log ρ, log V, k_io); D = diag(1, 1, k_io,ref), "
             f"k_io,ref = max(k_io, 5 s⁻¹) = {kio_ref[0]:g} s⁻¹; gradient scenario shown: {SCENARIO} (neither is nominated)",
             fontsize=8, color=INK_2, ha="left")
    return _write(fig, cfg)


# ===========================================================================
# 4. The Fisher matrix as a field over parameter space
# ===========================================================================

def plot_04_fisher_field(data: FisherData) -> list:
    """Nine normalized Fisher matrices, laid out by where their nodes sit in the band.

    Source: the STORED Phase-3 per-node eigen-decomposition of D F D for
    FIG4["domain"], reassembled as E diag(lambda) E' -- no recomputation.
    Nodes: select_lattice at NODES["k_io"]; lattice rows = v_i level (top high),
    columns = position along the band (rho rising, V falling).
    One shared normalization for all nine:
      "correlation"       R_jk = F_jk / sqrt(F_jj F_kk), free of D and of the noise
                          level, so only the coupling structure is compared;
      "dfd_over_lambda1"  D F D / lambda_1, which also keeps the diagonal balance.
    """
    cfg = FIG4
    style()
    selection = select_lattice(data, **NODES)
    letters = _lattice_letters(selection)
    eigenvalues = data.map3(cfg["domain"], "eigenvalues").astype(float)
    eigenvectors = data.map3(cfg["domain"], "eigenvectors").astype(float)
    fig = plt.figure(figsize=cfg["figsize"])
    outer = fig.add_gridspec(1, 2, width_ratios=(0.9, 1.5), wspace=0.1, left=0.05, right=0.97, top=0.84, bottom=0.12)
    ax_map = fig.add_subplot(outer[0])
    setup_plane(ax_map, data)
    rho_i, V_i = band_pairs(data, "retained")
    draw_cells(ax_map, data, rho_i, V_i, np.full(len(rho_i), 0.18), SEQ_BLUE, Normalize(0, 1))
    _mark_nodes(ax_map, data, selection)
    title(ax_map, "Node positions", f"k_io = {selection['k_io']:g} s⁻¹ · {cfg['domain']}")

    inner = outer[1].subgridspec(3, 3, hspace=0.62, wspace=0.18)
    norm = TwoSlopeNorm(0.0, -1.0, 1.0)
    corners = {}
    for (i, j), row in sorted(selection["lattice"].items()):
        ax = fig.add_subplot(inner[i, j])
        corners[(i, j)] = ax
        dfd = eigenvectors[row] @ np.diag(eigenvalues[row]) @ eigenvectors[row].T
        if cfg["normalization"] == "correlation":
            matrix, blank = fisher_correlation(pack_fisher(dfd)), True
        else:
            matrix, blank = dfd / eigenvalues[row, 0], False
        image = draw_matrix(ax, matrix, DIVERGING, norm, row_labels=PARAM_LABELS if j == 0 else None,
                            col_labels=PARAM_LABELS if i == 2 else None, blank_diagonal=blank, fontsize=7.5)
        status = (f"λ₃/λ₁ = {eigenvalues[row, 2] / eigenvalues[row, 0]:.1e}" if eigenvalues[row, 2] > 0
                  else "λ₃ ≤ 0: no CRLB")
        ax.set_title(f"{letters[row]}  ρ={data.rho[row]:.2g}, V={data.V[row]:.2g} pL\n{status}",
                     fontsize=7.5, loc="left", color=INK)
    top_left, bottom_right = corners[(0, 0)].get_position(), corners[(2, 2)].get_position()
    fig.text(top_left.x0 - 0.045, (top_left.y1 + bottom_right.y0) / 2, "higher $v_i$  →", rotation=90,
             ha="center", va="center", fontsize=8.5, color=INK_2)
    fig.text((top_left.x0 + bottom_right.x1) / 2, bottom_right.y0 - 0.06,
             "along the band  →  higher ρ, smaller V", ha="center", fontsize=8.5, color=INK_2)
    cax = fig.add_axes([bottom_right.x1 - 0.2, 0.035, 0.2, 0.012])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.outline.set_visible(False)
    bar.set_label("F_jk / √(F_jj F_kk)  (shared by all nine; diagonal ≡ 1)" if cfg["normalization"] == "correlation"
                  else "D F D / λ₁  (shared by all nine)", fontsize=8, color=INK_2)
    fig.suptitle("The Fisher matrix is a field over tissue space, not one global matrix",
                 x=0.02, ha="left", y=0.975, fontsize=13, color=INK)
    fig.text(0.02, 0.925, f"{cfg['domain']} · known S0, Monte-Carlo-debiased · one colour scale for all nine",
             fontsize=8, color=INK_2, ha="left")
    return _write(fig, cfg)


# ===========================================================================
# 5. CRLB / degeneracy landscape
# ===========================================================================

def _map_panel(ax, data: FisherData, rows: np.ndarray, values: np.ndarray, cmap, norm, mark_edge: bool = True):
    """One per-node quantity on its (rho, V) cells; NaN cells hatched as 'no CRLB'."""
    rho_i, V_i = data.nodes[rows, 0], data.nodes[rows, 1]
    finite = np.isfinite(values)
    mesh = draw_cells(ax, data, rho_i[finite], V_i[finite], values[finite], cmap, norm)
    draw_marked_cells(ax, data, rho_i[~finite], V_i[~finite], UNDEFINED_NOT_PD)
    if mark_edge:
        edge_r, edge_v = band_pairs(data, "edge")
        draw_marked_cells(ax, data, edge_r, edge_v, UNDEFINED_NO_STENCIL)
    return mesh, int(np.count_nonzero(~finite))


def _grid_contour(ax, data: FisherData, rows: np.ndarray, values: np.ndarray, level: float, **kw) -> None:
    g = data.grid
    image = np.full((len(g["Vs"]), len(g["rhos"])), np.nan)
    image[data.nodes[rows, 1], data.nodes[rows, 0]] = values
    ax.contour(np.log10(g["rhos"]), np.log10(g["Vs"]), np.ma.masked_invalid(image), levels=[level], **kw)


def _mark_reference(ax, data: FisherData, selection: dict) -> None:
    row = selection["reference"]
    ax.plot(np.log10(data.rho[row]), np.log10(data.V[row]), "o", markersize=6, markerfacecolor=INK,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=8)
    ax.annotate("E", (np.log10(data.rho[row]), np.log10(data.V[row])), xytext=(5, 4),
                textcoords="offset points", fontsize=8, color=INK, zorder=9)


def plot_05_crlb_landscape(data: FisherData) -> list:
    """Relative CRLB and kappa across (log rho, log V) at the representative k_io.

    Source: stored Phase-3 per-node maps (relative_crlb, kappa) -- the Phase-2
    definitions (fisher_phase2.md section 2): sqrt([F^-1]_jj) for the log
    parameters, kappa_j = sqrt([F^-1]_jj F_jj).  NaN in a map means the debiased F
    fails the strengthened positive-definiteness test: drawn hatched, never
    clipped into the colour range.  Rows: the MODEL LAYER (one average per stored
    column, a scale reference rather than a protocol) and a CONDITIONAL Phase-2
    acquisition.  Colour scales are shared down each column.
    """
    cfg = FIG5
    style()
    selection = select_lattice(data, **NODES)
    rows = selection["slice_rows"]
    threshold = float(data.prereg["kappa_unidentified_threshold"])
    budget = float(data.phase2["budget_images_N"])
    p = cfg["kappa_parameter"]
    columns = [(f"relative CRLB of log ρ  (≈ σ_ρ/ρ)", "relative_crlb", 0, SEQ_BLUE, LogNorm(*cfg["crlb_norm"])),
               (f"relative CRLB of log V  (≈ σ_V/V)", "relative_crlb", 1, SEQ_BLUE, LogNorm(*cfg["crlb_norm"])),
               (f"κ of {PARAM_LABELS[p]}  (1 = no trade-off)", "kappa", p, SEQ_ORANGE,
                LogNorm(*cfg["kappa_norm"]))]
    fig, axes = plt.subplots(len(cfg["rows"]), 3, figsize=cfg["figsize"], squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.1, wspace=0.12, hspace=0.28)
    for r, (domain, row_label) in enumerate(cfg["rows"]):
        entry = data.phase3["domains"][domain]
        for c, (heading, name, index, cmap, norm) in enumerate(columns):
            ax = axes[r, c]
            setup_plane(ax, data, xlabel=r == len(cfg["rows"]) - 1, ylabel=c == 0)
            values = data.map3(domain, name)[rows, index].astype(float)
            if (name == "relative_crlb" and cfg["rescale_model_layer_to_budget"]
                    and entry["layer"].startswith("model") and "uniform" not in domain):
                values *= math.sqrt(entry["columns"] / budget)      # CRLB scales as 1/sqrt(images)
            mesh, undefined = _map_panel(ax, data, rows, values, cmap, norm)
            if name == "kappa":
                _grid_contour(ax, data, rows, values, threshold, colors=[INK], linewidths=0.9, zorder=6)
            _mark_reference(ax, data, selection)
            ax.text(0.03, 0.04, f"no CRLB at {undefined} of {len(rows)} nodes", transform=ax.transAxes,
                    fontsize=7.5, color=INK_2)
            if r == 0:
                inset_colorbar(ax, mesh, heading + "\n(scale shared down the column)")
        position = axes[r, 0].get_position()
        columns_text = f"{entry['columns']:,} columns" if "columns" in entry else ""
        fig.text(0.012, (position.y0 + position.y1) / 2, f"{row_label}\n{domain} · {columns_text}", rotation=90,
                 ha="center", va="center", fontsize=8.5, color=INK)
    handles = undefined_handles() + [Line2D([], [], color=INK, linewidth=0.9,
                                            label=f"κ = {threshold:g} (pre-registered 'unidentified' threshold)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Where (ρ, V) is identifiable at $k_{{io}}$ = {selection['k_io']:g} s⁻¹: the model layer and a declared acquisition",
                 x=0.02, ha="left", fontsize=13, color=INK)
    scale_note = (f"model-layer CRLBs rescaled to the same {budget:g} images spread over all its stored columns "
                  "(exact by Fisher linearity; hypothetical: fractional averages, no masks)"
                  if cfg["rescale_model_layer_to_budget"] else "model-layer CRLBs at one average per stored column")
    fig.text(0.02, 0.925, f"known S0, Monte-Carlo-debiased · {scale_note}", fontsize=8, color=INK_2, ha="left")
    return _write(fig, cfg)


# ===========================================================================
# 6. Sloppy-direction field and the constant-v_i hyperbola
# ===========================================================================

def plot_06_sloppy_direction_field(data: FisherData) -> list:
    """The k_io-profiled (log rho, log V) sloppy direction drawn on constant-v_i lines.

    Source: stored Phase-3 maps for FIG6["domain"] -- profiled_sloppy_vector (unit,
    natural-log coordinates), profiled_condition_number (NaN where the profiled
    block is not positive definite) and profiled_sloppy_angle_deg.  Definition
    (fisher_phase3.md 1.1): S = [[F_rr, F_rV], [F_rV, F_VV]] - outer([F_rk, F_Vk])/F_kk,
    its least-determined eigenvector, acute angle to (1, -1)/sqrt(2).  Segments on
    a sparse sub-lattice (every stride[0]-th rho index and stride[1]-th V index);
    a direction is drawn wherever S exists, including where it is not positive
    definite (Phase-3 convention c).  Panel B aggregates every node per k_io value.
    """
    cfg = FIG6
    style()
    selection = select_lattice(data, **NODES)
    rows = selection["slice_rows"]
    domain = cfg["domain"]
    vector = data.map3(domain, "profiled_sloppy_vector").astype(float)
    condition = data.map3(domain, "profiled_condition_number").astype(float)
    angle = data.map3(domain, "profiled_sloppy_angle_deg").astype(float)

    fig = plt.figure(figsize=cfg["figsize"])
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1.0), wspace=0.2, left=0.06, right=0.98, top=0.86, bottom=0.1)
    ax = fig.add_subplot(grid[0])
    setup_plane(ax, data)
    mesh, undefined = _map_panel(ax, data, rows, condition[rows], SEQ_BLUE, LogNorm(*cfg["condition_norm"]))
    inset_colorbar(ax, mesh, "condition number of the profiled block S")
    sparse = rows[(data.nodes[rows, 0] % cfg["stride"][0] == 0) & (data.nodes[rows, 1] % cfg["stride"][1] == 0)
                  & np.all(np.isfinite(vector[rows]), axis=1)]
    centre = np.stack([np.log10(data.rho[sparse]), np.log10(data.V[sparse])], axis=1)
    half = 0.5 * cfg["segment_length"] * vector[sparse] / np.linalg.norm(vector[sparse], axis=1, keepdims=True)
    segments = np.stack([centre - half, centre + half], axis=1)
    ax.add_collection(LineCollection(segments, colors=SURFACE, linewidths=3.4, capstyle="round", zorder=6))
    ax.add_collection(LineCollection(segments, colors=INK, linewidths=1.5, capstyle="round", zorder=7))
    handles = [Line2D([], [], color=INK, linewidth=1.5,
                      label="least-determined (ρ, V) direction, sparse sub-lattice"),
               Line2D([], [], color=AXIS_RULE, linewidth=0.9, label="constant $v_i$ lines (hyperbolae in linear ρ, V)")]
    ax.legend(handles=handles + undefined_handles(), loc="lower left", fontsize=7.5)
    title(ax, "A   Segments run along the constant-$v_i$ lines",
          f"k_io = {selection['k_io']:g} s⁻¹ · {domain} · {len(sparse)} of {len(rows)} nodes drawn · "
          f"median angle on this slice {np.nanmedian(angle[rows]):.1f}°")

    ax_k = fig.add_subplot(grid[1])
    kios = np.unique(data.kio)
    stats = np.array([np.nanpercentile(angle[data.kio == k], (25, 50, 75)) for k in kios])
    ax_k.axvspan(30, kios.max() + 3, color=NEUTRAL, zorder=0)
    ax_k.fill_between(kios, stats[:, 0], stats[:, 2], color=CATEGORICAL[0], alpha=0.12, linewidth=0)
    ax_k.plot(kios, stats[:, 1], color=CATEGORICAL[0], linewidth=2)
    ax_k.axhline(45, color=INK_MUTED, linewidth=0.9)
    ax_k.text(kios.max(), 46, "random direction (median 45°)", ha="right", va="bottom", fontsize=7.5, color=INK_2)
    ax_k.plot(selection["k_io"], stats[np.searchsorted(kios, selection["k_io"]), 1], "o", color=INK, markersize=5)
    top = max(66.0, float(np.nanmax(stats)) + 4.0)
    ax_k.text(33, top - 4, "k_io > 30 s⁻¹", fontsize=7.5, color=INK_2)
    ax_k.set_xlim(0, kios.max() + 3)
    ax_k.set_ylim(0, top)
    ax_k.set_yticks([0, 10, 20, 30, 45, 60])
    ax_k.set_xlabel(r"$k_{io}$ (s$^{-1}$)")
    ax_k.set_ylabel("angle to constant $v_i$ (deg)")
    ax_k.grid(axis="y")
    despine(ax_k)
    valid = np.isfinite(angle)
    title(ax_k, "B   Alignment across the $k_{io}$ grid",
          f"median (line) and interquartile range over the band · all nodes: median {np.median(angle[valid]):.2f}°, "
          f"{np.mean(angle[valid] < 10):.0%} below 10°")
    fig.suptitle("In the (ρ, V) plane the Fisher degeneracy follows the constant-$v_i$ hyperbola",
                 x=0.02, ha="left", fontsize=13, color=INK)
    return _write(fig, cfg)


# ===========================================================================
# 7. Profiled 2D versus full 3D Fisher geometry
# ===========================================================================

def _shape_ellipse(ax, eigenvalues: np.ndarray, sloppy: np.ndarray, color: str) -> str:
    """Shape of the profiled CRLB region with its scale removed (major semi-axis = 1)."""
    ax.axline((0, 0), slope=-1, color=AXIS_RULE, linewidth=1.0, zorder=1)
    ax.axline((0, 0), slope=1, color=GRIDLINE, linewidth=1.0, zorder=1)
    stiff = np.array([-sloppy[1], sloppy[0]])
    t = np.linspace(0, 2 * np.pi, 240)
    if eigenvalues[1] > 0:
        minor = math.sqrt(eigenvalues[1] / eigenvalues[0])
        points = np.outer(np.cos(t), sloppy) + minor * np.outer(np.sin(t), stiff)
        ax.fill(points[:, 0], points[:, 1], color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.8, zorder=3)
        text = f"axis ratio {1 / minor:.0f} : 1"
    else:
        for side in (-1, 1):
            ax.axline(tuple(side * 0.06 * stiff), slope=sloppy[1] / sloppy[0], color=color, linewidth=1.8)
        text = "unbounded along the ridge"
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Δ log ρ")
    ax.set_ylabel("Δ log V")
    despine(ax)
    return text


def plot_07_profiled_vs_full(data: FisherData) -> list:
    """The k_io-profiled (log rho, log V) geometry beside the full 3x3 sloppy eigenvector.

    Two different objects, never conflated (fisher_phase3.md 1.1):
      left  -- the k_io-profiled 2x2 block S: its CRLB-region shape (scale removed)
               and its sloppy angle; D-free by construction;
      right -- the least-determined eigenvector of D F D, D = diag(1, 1, k_io_ref):
               its signed components and its |k_io| share (convention-dependent in
               the k_io component only).
    Nodes: lattice middle row (D, E, F: low, mid, high rho at median v_i).
    Aggregates: all nodes of FIG7["domain"] from the stored Phase-3 maps.
    """
    cfg = FIG7
    style()
    domain = cfg["domain"]
    selection = select_lattice(data, **NODES)
    letters = _lattice_letters(selection)
    nodes = [selection["lattice"][(1, j)] for j in range(3)]
    prof_values = data.map3(domain, "profiled_eigenvalues").astype(float)
    prof_vector = data.map3(domain, "profiled_sloppy_vector").astype(float)
    prof_angle = data.map3(domain, "profiled_sloppy_angle_deg").astype(float)
    vectors = data.map3(domain, "eigenvectors").astype(float)
    angle3 = data.map3(domain, "sloppy_angle_deg").astype(float)
    k_share = data.map3(domain, "sloppy_k_io_fraction").astype(float)

    fig = plt.figure(figsize=cfg["figsize"])
    grid = fig.add_gridspec(2, 7, width_ratios=(1, 1, 1, 0.25, 1, 1, 1), height_ratios=(1, 0.9),
                            hspace=0.55, wspace=0.35, left=0.05, right=0.98, top=0.8, bottom=0.08)
    for j, node in enumerate(nodes):
        ax = fig.add_subplot(grid[0, j])
        text = _shape_ellipse(ax, prof_values[node], prof_vector[node], CATEGORICAL[0])
        ax.set_title(f"{letters[node]}  ρ={data.rho[node]:.2g}, V={data.V[node]:.2g} pL", fontsize=8.5, loc="left")
        ax.text(0.5, -0.28, f"{prof_angle[node]:.1f}° from constant $v_i$\n{text}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color=INK)
        ax = fig.add_subplot(grid[0, 4 + j])
        v3 = _oriented(vectors[node])[:, 2]
        ax.bar(range(3), v3, color=CATEGORICAL[0], width=0.55)
        ax.axhline(0, color=AXIS_RULE, linewidth=0.8)
        for i, value in enumerate(v3):
            ax.text(i, value + (0.06 if value >= 0 else -0.06), f"{value:+.2f}", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=7.5, color=INK)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks(range(3))
        ax.set_xticklabels([r"$\log\rho$", r"$\log V$", r"$k_{io}/k_{io,ref}$"], fontsize=8)
        ax.set_yticks([-1, 0, 1])
        despine(ax)
        ax.set_title(f"{letters[node]}  ρ={data.rho[node]:.2g}, V={data.V[node]:.2g} pL", fontsize=8.5, loc="left")
        ax.text(0.5, -0.2, f"|k_io| component {k_share[node]:.2f}\n{angle3[node]:.0f}° from constant $v_i$",
                transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK)
    left, right = fig.axes[0].get_position(), fig.axes[-1].get_position()
    fig.text(left.x0, 0.87, "k_io-profiled 2×2 block of (log ρ, log V)", fontsize=11, color=INK)
    fig.text(left.x0, 0.845, "shape of the 1σ CRLB region (scale removed); gray line: constant $v_i$", fontsize=8, color=INK_2)
    x_right = fig.axes[1].get_position().x0
    fig.text(x_right, 0.87, "Full 3×3 D F D: least-determined eigenvector", fontsize=11, color=INK)
    fig.text(x_right, 0.845, "signed components; k_io measured in units of k_io,ref = max(k_io, 5 s⁻¹)",
             fontsize=8, color=INK_2)

    ax = fig.add_subplot(grid[1, 0:3])
    values = prof_angle[np.isfinite(prof_angle)]
    ax.hist(values, bins=np.arange(0, 91, 2.0), density=True, color=CATEGORICAL[0], edgecolor=SURFACE, linewidth=0.4)
    ax.axhline(1 / 90, color=INK_MUTED, linewidth=0.9)
    ax.axvline(np.median(values), color=INK, linewidth=1.0)
    ax.text(np.median(values) + 2, 0.2, f"median {np.median(values):.2f}°\n{np.mean(values < 10):.0%} below 10°",
            fontsize=8, color=INK, va="top")
    ax.text(88, 1 / 90 * 1.3, "random direction", ha="right", fontsize=7.5, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlim(0, 90)
    ax.set_xlabel("profiled sloppy angle to constant $v_i$ (deg)")
    ax.set_ylabel("density (log)")
    despine(ax)
    title(ax, "Every node: the profiled direction sits on the hyperbola", f"{domain} · {values.size:,} nodes")

    ax = fig.add_subplot(grid[1, 4:7])
    decades = ((4, 5), (5, 6), (6, 7.01))
    for colour, (lo, hi) in zip(CATEGORICAL, decades):
        members = k_share[(np.log10(data.rho) >= lo) & (np.log10(data.rho) < hi)]
        members = members[np.isfinite(members)]
        ax.hist(members, bins=np.linspace(0, 1, 26), density=True, histtype="step", color=colour, linewidth=2,
                label=rf"$\rho$ = $10^{{{lo}}}$–$10^{{{int(hi)}}}$: median {np.median(members):.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("|k_io| component of the 3×3 sloppy eigenvector (0 = in the plane, 1 = pure k_io)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=7.5)
    despine(ax)
    valid = np.isfinite(angle3)
    title(ax, "Every node: the 3-parameter sloppy direction is mostly exchange",
          f"median |k_io| component {np.median(k_share[np.isfinite(k_share)]):.2f}; 3D angle to constant $v_i$ "
          f"median {np.median(angle3[valid]):.1f}° (random: 60°)")
    fig.suptitle("Two answers to two questions: (ρ, V) trade off along $v_i$; the whole problem is least sure about $k_{io}$",
                 x=0.02, ha="left", fontsize=13, color=INK, y=0.975)
    return _write(fig, cfg)


# ===========================================================================
# 8. Acquisition design reshapes Fisher information
# ===========================================================================

def _identifiable_fraction(data: FisherData, key: str, acquisition) -> float:
    """Share of all evaluation nodes with a CRLB: the recorded value where Phase 2/3 recorded one."""
    if key == "full":
        return float(data.phase3["domains"]["full_stored_domain"]["regimes"]["known_amplitude"]["crlb"]
                     ["positive_definite_fraction"])
    if key in ("m1", "m2"):
        return float(data.phase2["scenarios"][SCENARIO]["arms"]["8"][key]["report"]["regimes"]["known_amplitude"]
                     ["positive_definite_fraction"])
    return float(packed_inverse_diagonal(form_fisher(data, acquisition)["debiased"])[2].mean())


def plot_08_acquisition_design(data: FisherData) -> list:
    """How acquisition design reshapes the Fisher matrix at the reference node.

    Stages, from the executed Phase-2 N = 128 sweep under SCENARIO:
      1  the exhaustive single-Delta optimum;
      2  that timing plus the greedily added second timing (greedy_from_best_m1);
      3  the exhaustive two-Delta optimum;
      context: the full stored domain (MODEL LAYER), rescaled to the same 128
         images spread evenly over its columns -- exact by Fisher linearity, but
         hypothetical (fractional averages, no masks).
    All four carry one budget, so absolute eigenvalues, CRLB regions and CRLBs
    share axes.  Source: form_fisher (stages 1, 3 and the context checked against
    the stored Phase-3 maps; stage 2's b-subsets re-selected and its score checked).
    """
    cfg = FIG8
    style()
    selection = select_lattice(data, **NODES)
    node = selection["reference"]
    kio_ref = data.kio_ref[[node]]
    budget = float(data.phase2["budget_images_N"])
    full = full_stored_domain(data)
    stages = [("1  Best single Δ", "m1", phase2_arm(data, SCENARIO, "m1"), 1.0),
              ("2  + a second Δ (greedy)", "greedy", phase2_greedy_second_delta(data, SCENARIO), 1.0),
              ("3  Both timings optimized", "m2", phase2_arm(data, SCENARIO, "m2"), 1.0),
              ("Context: full stored domain", "full", full, budget / full.n_columns)]
    packed = [form_fisher(data, acquisition, [node])["debiased"] * factor for _, _, acquisition, factor in stages]
    spectra = [fisher_spectrum(p, kio_ref)["eigenvalues"][0] for p in packed]
    positive = np.concatenate([s[s > 0] for s in spectra])
    spectrum_ylim = (10.0 ** math.floor(np.log10(positive.min()) - 0.5), 10.0 ** math.ceil(np.log10(positive.max()) + 0.3))
    bounds = [crlb_kappa(p, kio_ref) for p in packed]
    finite = np.concatenate([b["relative"][0][np.isfinite(b["relative"][0])] for b in bounds])
    crlb_ylim = (10.0 ** math.floor(np.log10(finite.min())), finite.max() * 12)

    fig = plt.figure(figsize=cfg["figsize"])
    grid = fig.add_gridspec(4, 4, height_ratios=(1.0, 1.15, 0.8, 0.85), hspace=0.72, wspace=0.34,
                            top=0.82, bottom=0.06, left=0.09, right=0.98)
    signed = TwoSlopeNorm(0.0, -1.0, 1.0)
    limit = cfg["ellipse_limit"]
    for col, ((heading, key, acquisition, factor), p, values, bound) in enumerate(zip(stages, packed, spectra, bounds)):
        first = col == 0
        ax = fig.add_subplot(grid[0, col])
        image = draw_matrix(ax, fisher_correlation(p)[0], DIVERGING, signed, blank_diagonal=True,
                            row_labels=PARAM_LABELS if first else None)
        detail = (_stage_lines(acquisition) if key != "full" else
                  f"MODEL LAYER, rescaled\n{budget:g} images spread over all {acquisition.n_columns:,} DW columns\n"
                  "(hypothetical: fractional averages, no masks)")
        ax.text(0.0, 1.1, f"{detail}\nCRLB exists at {_identifiable_fraction(data, key, acquisition):.0%} of all nodes",
                transform=ax.transAxes, fontsize=7.5, color=INK_2, va="bottom")
        ax.text(0.0, 1.52, heading, transform=ax.transAxes, fontsize=11, color=INK if key != "full" else INK_2,
                va="bottom")

        ax = fig.add_subplot(grid[1, col])
        ax.axline((0, 0), slope=-1, color=AXIS_RULE, linewidth=0.9, zorder=1)
        info = draw_profiled_ellipse(ax, p[0], CATEGORICAL[0], limit)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_xlabel("Δ log ρ")
        ax.set_ylabel("Δ log V" if first else "")
        ax.grid(True)
        despine(ax)
        if info["kind"] == "ellipse":
            note = f"±{info['semi_major']:.2f} along the ridge\n±{info['semi_minor']:.3f} across it"
            if info["semi_major"] > limit:
                note += "\n(extends past the axes)"
        elif info["kind"] == "band":
            note = f"unbounded along the ridge\n±{info['half_width']:.3f} across it"
        else:
            note = ""
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=7.5, color=INK, va="bottom")

        ax = fig.add_subplot(grid[2, col])
        _spectrum_panel(ax, values, 1.0, CATEGORICAL[0], ylim=spectrum_ylim)
        ax.set_ylabel("eigenvalue of D F D" if first else "")

        ax = fig.add_subplot(grid[3, col])
        relative, kappa = bound["relative"][0], bound["kappa"][0]
        if bool(bound["positive"][0]):
            ax.bar(range(3), relative, color=CATEGORICAL[0], width=0.55)
            for i in range(3):
                ax.text(i, relative[i] * 1.25, f"{relative[i]:.2f}\nκ={kappa[i]:.0f}", ha="center", va="bottom",
                        fontsize=7, color=INK)
        else:
            ax.text(1, math.sqrt(crlb_ylim[0] * crlb_ylim[1]), "no CRLB\n(F not positive definite)", ha="center",
                    va="center", fontsize=8, color=INK_2)
        ax.set_yscale("log")
        ax.set_ylim(*crlb_ylim)
        ax.axhline(1.0, color=INK_MUTED, linewidth=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([r"$\log\rho$", r"$\log V$", r"$k_{io}$"])
        ax.set_ylabel("relative CRLB" if first else "")
        despine(ax)
    names = ["correlation", "k_io-profiled 1σ\nCRLB region", "spectrum\n(same budget)", "relative CRLB\n(κ above bars)"]
    for r, name in enumerate(names):
        position = fig.axes[r].get_position()
        fig.text(0.012, (position.y0 + position.y1) / 2, name, rotation=90, ha="center", va="center", fontsize=8.5, color=INK_2)
    cax = fig.add_axes([0.83, 0.955, 0.14, 0.009])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7)
    bar.set_label("F_jk / √(F_jj F_kk)", fontsize=7.5, color=INK_2)
    fig.suptitle(f"Sampling a second diffusion time reshapes the Fisher matrix at node E ({node_text(data, node)}, "
                 f"$k_{{io}}$ = {data.kio[node]:g} s⁻¹)", x=0.02, ha="left", y=0.985, fontsize=13, color=INK)
    fig.text(0.02, 0.955, f"known S0, debiased · N = {budget:g} images, SNR 50 · gradient scenario shown: {SCENARIO} "
             "(neither nominated) · Δ log is a natural-log step, ≈ fractional", fontsize=8, color=INK_2)
    return _write(fig, cfg)


# ===========================================================================
# 9. Monte-Carlo Fisher debiasing
# ===========================================================================

def plot_09_mc_debias(data: FisherData) -> list:
    """The Var(J_hat) correction: its numerical size, and what it does to the CRLB.

    Debias (plan 2.5; Phase 2's endpoint-only form, conservative): Var(J_hat) is
    subtracted from the Fisher DIAGONAL only.  Example domain: the Phase-2
    single-Delta optimum under SCENARIO (CONDITIONAL).  Example node rule: the
    reference node E if debiasing changes its verdict; otherwise the nearest node
    on the same k_io slice (grid-step distance) whose verdict changes -- the maps
    beside it show how common that is.  Maps: share of each group's interior
    k_io nodes at which debiasing removes the CRLB, for the model layer (pair_pass,
    Phase 3's own accumulation) and the single-Delta acquisition.  Bars: recorded
    Phase-2/3 identifiable fractions and mean diagonal corrections.
    """
    cfg = FIG9
    style()
    selection = select_lattice(data, **NODES)
    rows = selection["slice_rows"]
    g = data.grid
    single = phase2_arm(data, SCENARIO, "m1")
    formed = form_fisher(data, single)
    identifiable_raw = packed_inverse_diagonal(formed["undebiased"])[2]
    identifiable = packed_inverse_diagonal(formed["debiased"])[2]
    flips = identifiable_raw & ~identifiable
    node = selection["reference"]
    example = "E" if flips[node] else "✕ (nearest to E that flips)"
    if not flips[node]:
        cost = (((np.log10(data.rho[rows]) - np.log10(data.rho[node])) / g["h_log10_rho"]) ** 2
                + ((np.log10(data.V[rows]) - np.log10(data.V[node])) / g["h_log10_V"]) ** 2)
        node = next(int(rows[k]) for k in np.argsort(cost, kind="stable") if flips[rows[k]])
    passed = pair_pass(data)
    full_raw = passed["full_debiased"].copy()
    full_raw[:, [0, 3, 5]] += passed["full_debias"]
    full_flips = packed_inverse_diagonal(full_raw)[2] & ~packed_inverse_diagonal(passed["full_debiased"])[2]

    fig = plt.figure(figsize=cfg["figsize"])
    grid = fig.add_gridspec(2, 3, height_ratios=(0.8, 1.0), hspace=0.42, wspace=0.28, left=0.06, right=0.98,
                            top=0.86, bottom=0.07)
    raw, debiased = formed["undebiased"][node], formed["debiased"][node]
    kio_ref = data.kio_ref[[node]]

    ax = fig.add_subplot(grid[0, 0])
    correction = 100.0 * (raw[[0, 3, 5]] - debiased[[0, 3, 5]]) / raw[[0, 3, 5]]
    ax.barh(range(3), correction, color=CATEGORICAL[0], height=0.5)
    for i, value in enumerate(correction):
        ax.text(value, i, f"  {value:.3f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r"$F_{\rho\rho}$", r"$F_{VV}$", r"$F_{kk}$"])
    ax.invert_yaxis()
    ax.set_xlim(0, correction.max() * 1.6)
    ax.set_xlabel("Var(Ĵ) removed, % of the undebiased diagonal entry")
    ax.text(0.98, 0.03, "off-diagonal entries: unchanged", transform=ax.transAxes, ha="right", fontsize=7.5,
            color=INK_2)
    ax.grid(axis="x")
    despine(ax)
    title(ax, "A   The correction is small", f"node {example}: ρ={data.rho[node]:.2g} µL⁻¹, V={data.V[node]:.2g} pL")

    ax = fig.add_subplot(grid[0, 1])
    before = fisher_spectrum(raw[None], kio_ref)["eigenvalues"][0]
    after = fisher_spectrum(debiased[None], kio_ref)["eigenvalues"][0]
    for offset, values, colour, label in ((-0.13, before, CATEGORICAL[0], "undebiased"),
                                          (0.13, after, CATEGORICAL[1], "debiased")):
        ratio = values / before[0]
        ax.vlines(np.arange(3) + offset, 0, ratio, color=colour, linewidth=2)
        ax.plot(np.arange(3) + offset, ratio, "o", color=colour, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
        ax.text(2 + offset, ratio[2], f" {ratio[2]:+.1e}", fontsize=7.5, color=INK,
                ha="left" if offset > 0 else "right", va="bottom" if ratio[2] > 0 else "top")
    linthresh = 10.0 ** math.floor(np.log10(abs(after[2] / before[0])) - 1)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["λ₁ stiff", "λ₂", "λ₃ sloppy"])
    ax.set_ylabel("eigenvalue of D F D ÷ undebiased λ₁ (symlog)")
    ax.legend(loc="upper right", fontsize=8)
    despine(ax)
    title(ax, "B   … but λ₃ changes sign", f"{single.label} · {single.note} · the CRLB disappears")

    ax = fig.add_subplot(grid[0, 2])
    spectrum = fisher_spectrum(formed["undebiased"], data.kio_ref)["eigenvalues"]
    removed = formed["undebiased"][:, [0, 3, 5]] - formed["debiased"][:, [0, 3, 5]]
    removed[:, 2] *= data.kio_ref ** 2
    # Nodes with undebiased lambda_3/lambda_1 below 1e-12 are numerically rank-deficient (too few usable
    # columns after this acquisition's masks): no conditioning statement, so they are left out and counted.
    keep = spectrum[:, 2] > 1e-12 * spectrum[:, 0]
    x = np.log10(removed.max(axis=1)[keep] / spectrum[keep, 0])
    y = np.log10(spectrum[keep, 2] / spectrum[keep, 0])
    hexes = ax.hexbin(x, y, C=flips[keep].astype(float), reduce_C_function=np.mean, gridsize=36, cmap=SEQ_ORANGE,
                      vmin=0, vmax=1, mincnt=1, linewidths=0)
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([low, high], [low, high], color=INK, linewidth=1.0)
    ax.text(0.03, 0.62, "above the line the correction is smaller than λ₃,\nso the CRLB must survive (Weyl's inequality)",
            transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
    ax.set_xlabel("largest diagonal correction ÷ λ₁ (log10)")
    ax.set_ylabel("undebiased λ₃ ÷ λ₁ (log10)")
    despine(ax)
    bar = fig.colorbar(hexes, ax=ax, pad=0.02, fraction=0.05)
    bar.outline.set_visible(False)
    bar.set_label("share of nodes losing their CRLB", fontsize=8, color=INK_2)
    title(ax, "C   It matters where λ₃ is already that small",
          f"{int(keep.sum()):,} of {len(keep):,} nodes (undebiased λ₃/λ₁ > 1e-12) · {single.label}")

    for col, (flags, heading, sub) in enumerate((
            (full_flips, "D   Model layer: where debiasing removes the CRLB",
             f"full stored domain · {full_flips.mean():.0%} of all nodes"),
            (flips, "E   Single-Δ acquisition", f"{single.label} · {flips.mean():.0%} of all nodes"))):
        ax = fig.add_subplot(grid[1, col])
        setup_plane(ax, data, ylabel=col == 0)
        rho_i, V_i, share = per_pair(data, np.arange(len(data.rho)), flags.astype(float), np.mean)
        mesh = draw_cells(ax, data, rho_i, V_i, share, SEQ_ORANGE, Normalize(0, 1))
        edge_r, edge_v = band_pairs(data, "edge")
        draw_marked_cells(ax, data, edge_r, edge_v, UNDEFINED_NO_STENCIL)
        ax.plot(np.log10(data.rho[node]), np.log10(data.V[node]), "X", color=INK, markersize=8,
                markeredgecolor=SURFACE, zorder=8)
        inset_colorbar(ax, mesh, "share of the group's k_io nodes losing the CRLB")
        title(ax, heading, sub)

    ax = fig.add_subplot(grid[1, 2])
    p3 = data.phase3["domains"]["full_stored_domain"]
    records = [("full stored domain", p3["monte_carlo_debias_effect"]["geometry_undebiased"]["positive_definite_fraction"],
                p3["regimes"]["known_amplitude"]["crlb"]["positive_definite_fraction"],
                p3["monte_carlo_debias_effect"]["mean_debias_over_fisher_diagonal"])]
    for arm, name in (("m1", "single-Δ optimum"), ("m2", "two-Δ optimum")):
        effect = data.phase2["scenarios"][SCENARIO]["arms"]["8"][arm]["report"]["monte_carlo_debias_effect"]
        records.append((name, effect["identifiable_fraction_undebiased"], effect["identifiable_fraction_debiased"],
                        effect["mean_debias_over_fisher_diagonal"]))
    for i, (name, before_fraction, after_fraction, mean_correction) in enumerate(records):
        for offset, value, colour in ((-0.2, before_fraction, CATEGORICAL[0]), (0.1, after_fraction, CATEGORICAL[1])):
            ax.barh(i + offset, value, height=0.26, color=colour)
            ax.text(value, i + offset, f" {value:.0%}", va="center", fontsize=8, color=INK)
        ax.text(0.0, i + 0.3, "mean diagonal correction: " + ", ".join(
            f"{label} {100 * c:.2f}%" for label, c in zip(("ρ", "V", "k_io"), mean_correction)),
            fontsize=7.2, color=INK_2, va="top")
    ax.set_yticks(range(len(records)))
    ax.set_yticklabels([r[0] for r in records])
    ax.set_ylim(len(records) - 0.35, -0.85)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("share of evaluation nodes with a CRLB")
    ax.legend(handles=[Line2D([], [], color=CATEGORICAL[0], linewidth=6, label="undebiased"),
                       Line2D([], [], color=CATEGORICAL[1], linewidth=6, label="debiased")],
              loc="upper right", ncol=2, fontsize=8)
    despine(ax)
    title(ax, "F   Recorded consequence (Phase 2 and 3)", f"known S0 · gradient scenario {SCENARIO} for the acquisitions")
    fig.suptitle("A sub-percent Monte-Carlo correction decides whether a CRLB exists",
                 x=0.02, ha="left", fontsize=13, color=INK)
    return _write(fig, cfg)


# ===========================================================================
# Driver
# ===========================================================================

FIGURES = {1: plot_01_library_atlas, 2: plot_02_acquisition_atlas, 3: plot_03_fisher_anatomy,
           4: plot_04_fisher_field, 5: plot_05_crlb_landscape, 6: plot_06_sloppy_direction_field,
           7: plot_07_profiled_vs_full, 8: plot_08_acquisition_design, 9: plot_09_mc_debias}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=int, nargs="*", choices=sorted(FIGURES), help="figure numbers (default: all)")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: docs/figures)")
    args = parser.parse_args(argv)
    if args.out is not None:
        OUTPUT["dir"] = args.out
    data = FisherData()
    for number in args.only or sorted(FIGURES):
        for path in FIGURES[number](data):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
