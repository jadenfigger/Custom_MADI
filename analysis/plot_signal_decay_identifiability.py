#!/usr/bin/env python3
"""
plot_signal_decay_identifiability.py — signal-decay + inverse-problem figures
===============================================================================

Two presentation-quality figures at a single fixed acquisition timing
(δ, Δ) = (4, 20) ms, b = 0-6000 s/mm², drawn from the remediated universal
(δ,Δ,b) library (`madi.library` v5/v4 schema — NOT the old fixed-δ format,
so this script reads `vectors`/`pair_deltas`/`pair_Deltas`/`b_values`
directly rather than assuming any legacy layout):

  1. Reference decay plot: S/S0 vs b for the whole library at this
     acquisition (thin, semi-transparent gray background curves) with a
     handful of representative (k_io, rho, V) curves highlighted, selected
     by spanning the quantiles of curve area-under-decay (a model-free
     summary of overall exchange/geometry behaviour) rather than by
     picking arbitrary or hand-labelled combinations.

  2. Identifiability plot: a small set of (k_io, rho, V) combinations that
     are computationally selected to have (a) nearly-identical decay
     curves over this acquisition's b-range and (b) substantial pairwise
     separation in normalized parameter space — i.e. a concrete example of
     the curve degeneracy that motivates a Fisher/CRLB identifiability
     analysis (Springer et al.'s observation that very different
     parameter sets can produce near-indistinguishable decay curves).

The library file is large (tens of GB, mostly per-column Monte-Carlo
diagnostics this script never touches); only the six small parameter
arrays and the 13 b<=6000 columns of the one matching (delta,Delta) pair
are ever pulled out of the on-disk `vectors` array.

Usage
-----
    python analysis/plot_signal_decay_identifiability.py \\
        --library data/libraries/madi_dense_universal_remediated.npz \\
        --outdir docs/figures/signal_decay_identifiability


    python analysis/plot_signal_decay_identifiability.py --library data/libraries/madi_dense_universal_remediated.npz --outdir docs/figures/signal_decay_identifiability
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIBRARY = os.path.join(REPO_ROOT, "data/libraries/madi_dense_universal_remediated.npz")
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "docs/figures/signal_decay_identifiability")

TARGET_DELTA_MS = 4.0     # small pulse duration delta
TARGET_BIGDELTA_MS = 20.0  # diffusion time Delta
B_MAX_S_MM2 = 6000.0

LOG_FLOOR = 1e-4  # S/S0 plotting floor; only clips MC-noise artifacts (<1% of curves)

STYLE = dict(
    background_color="0.55",
    background_alpha=0.10,
    background_lw=0.5,
    grid_alpha=0.58,
    grid_lw=0.5,
    highlight_lw=2.0,
    marker_size=5,
)

HIGHLIGHT_COLORS_REF = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d",
]
HIGHLIGHT_COLORS_IDENT = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
]


# ---------------------------------------------------------------------------
# Library I/O — schema of madi_dense_universal_remediated.npz
# ---------------------------------------------------------------------------

def load_acquisition_subset(path, delta_ms, Delta_ms, b_max):
    """Pull the small parameter arrays plus one (delta,Delta) pair's b<=b_max
    columns out of the (delta,Delta,b)-universal library, without ever
    materializing the full `vectors` array or the large per-column
    Monte-Carlo diagnostic arrays.
    """
    t0 = time.time()
    data = np.load(path, mmap_mode="r")

    kios = np.asarray(data["kios"])
    rhos = np.asarray(data["rhos"])
    Vs = np.asarray(data["Vs"])
    is_free_water = np.asarray(data["is_free_water"])

    pair_deltas = np.asarray(data["pair_deltas"])
    pair_Deltas = np.asarray(data["pair_Deltas"])
    b_values_full = np.asarray(data["b_values"], dtype=float)
    n_b = int(data["n_b"])

    pair_idx = np.where(
        np.isclose(pair_deltas, delta_ms) & np.isclose(pair_Deltas, Delta_ms)
    )[0]
    if len(pair_idx) != 1:
        raise ValueError(
            f"expected exactly one (delta,Delta)=({delta_ms},{Delta_ms}) ms pair "
            f"in the library grid, found {len(pair_idx)}"
        )
    pair_idx = int(pair_idx[0])

    b_mask = b_values_full <= b_max + 1e-6
    b_values = b_values_full[b_mask]
    cols = pair_idx * n_b + np.where(b_mask)[0]

    vectors = data["vectors"]  # lazy memmap handle — not yet read
    signal = np.asarray(vectors[:, cols])  # only these 13 columns are pulled from disk

    print(f"[library] loaded (delta,Delta)=({delta_ms:g},{Delta_ms:g}) ms, "
          f"b in [0,{b_max:g}] s/mm^2 -> {signal.shape[1]} b-values, "
          f"{signal.shape[0]} library entries, in {time.time()-t0:.1f}s")

    return kios, rhos, Vs, is_free_water, b_values, signal


def filter_cellular_entries(kios, rhos, Vs, is_free_water, signal):
    """Drop the free-water atom (k_io undefined) and any degenerate rows."""
    mask = (~is_free_water) & (rhos > 0) & (Vs > 0) & np.isfinite(kios)
    return kios[mask], rhos[mask], Vs[mask], signal[mask]


# ---------------------------------------------------------------------------
# Figure 1 — reference decay plot
# ---------------------------------------------------------------------------

N_BACKGROUND_CURVES = 500


def select_reference_highlights(b_values, signal, n_highlights=4):
    """Pick real library curves spanning the quantiles of decay area (a
    model-free summary of overall signal loss), so the highlighted curves
    are representative of the library's spread rather than arbitrary.
    """
    auc = np.trapezoid(signal, b_values, axis=1)
    quantiles = np.linspace(0.05, 0.95, n_highlights)
    targets = np.quantile(auc, quantiles)
    idx = []
    for t in targets:
        candidate = int(np.argmin(np.abs(auc - t)))
        if candidate not in idx:
            idx.append(candidate)
    return idx


def make_reference_figure(b_values, signal, kios, rhos, Vs, outpath_base, dpi, seed=0):
    highlight_idx = select_reference_highlights(b_values, signal)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    rng = np.random.default_rng(seed)
    n_bg = min(N_BACKGROUND_CURVES, signal.shape[0])
    bg_idx = rng.choice(signal.shape[0], size=n_bg, replace=False)
    background = np.clip(signal[bg_idx], LOG_FLOOR, None)
    x_row = np.broadcast_to(b_values, background.shape)
    segments = np.stack([x_row, background], axis=-1)
    lc = LineCollection(
        segments, colors=STYLE["background_color"], alpha=STYLE["background_alpha"],
        linewidths=STYLE["background_lw"], zorder=1,
    )
    ax.add_collection(lc)

    for i, idx in enumerate(highlight_idx):
        color = HIGHLIGHT_COLORS_REF[i % len(HIGHLIGHT_COLORS_REF)]
        y = np.clip(signal[idx], LOG_FLOOR, None)
        label = (rf"$k_{{io}}$={kios[idx]:.2f} s$^{{-1}}$, "
                 rf"$\rho$={rhos[idx]:,.0f} $\mu$L$^{{-1}}$, "
                 rf"$V$={Vs[idx]:.2f} pL")
        ax.plot(b_values, y, "o-", color=color, lw=STYLE["highlight_lw"],
                 ms=STYLE["marker_size"], label=label, zorder=3,
                 markeredgecolor="white", markeredgewidth=0.4)

    ax.set_xlim(b_values.min(), b_values.max())
    ax.set_ylim(LOG_FLOOR, 1.3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$b$ (s/mm$^2$)")
    ax.set_ylabel(r"$S/S_0$")
    ax.set_title(rf"MADI signal decay — $\delta$={TARGET_DELTA_MS:g} ms, "
                 rf"$\Delta$={TARGET_BIGDELTA_MS:g} ms")
    # ax.grid(True, which="both", alpha=STYLE["grid_alpha"], lw=STYLE["grid_lw"])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, framealpha=0.92, loc="upper right", title="highlighted (k_io, ρ, V)")
    fig.tight_layout()

    fig.savefig(outpath_base + ".png", dpi=dpi)
    fig.savefig(outpath_base + ".pdf")
    plt.close(fig)
    print(f"[figure 1] saved {outpath_base}.png / .pdf  "
          f"({len(highlight_idx)} highlighted curves; {n_bg} of {signal.shape[0]} "
          f"library entries shown as background)")
    return highlight_idx


# ---------------------------------------------------------------------------
# Figure 2 — identifiability / degeneracy plot
# ---------------------------------------------------------------------------

def _param_coords(kios, rhos, Vs):
    """Normalized log-space parameter coordinates for measuring 'substantial'
    separation between (k_io, rho, V) triples on comparable scales.
    """
    coords = np.stack([np.log10(kios + 1.0), np.log10(rhos), np.log10(Vs)], axis=1)
    mu = coords.mean(axis=0)
    sd = coords.std(axis=0)
    return (coords - mu) / sd


def find_degenerate_group(
    kios, rhos, Vs, signal,
    group_size=4,
    curve_maxabs_thresh=0.01,
    min_pairwise_param_dist=1.5,
    n_candidates=4000,
    seed=0,
):
    """Search the library for `group_size` (k_io, rho, V) combinations whose
    decay curves are nearly identical (max abs S/S0 difference across the
    b-grid below `curve_maxabs_thresh`) while every pair is separated by at
    least `min_pairwise_param_dist` in normalized log-parameter space.

    Method: build a KD-tree on the curve vectors, probe many candidate
    "curve centers", and within each center's tight curve-similarity ball
    greedily farthest-point-sample in parameter space. This is a
    computational search over the whole library, not a hand-picked
    neighboring-grid-point comparison.
    """
    n = signal.shape[0]
    tree = cKDTree(signal)
    # A generous radius in absolute S/S0 units; refined by the exact
    # max-abs-difference check below (Euclidean over b-columns upper-bounds
    # the max-abs difference by at most a factor of sqrt(n_b)).
    radius = curve_maxabs_thresh * np.sqrt(signal.shape[1])

    Z = _param_coords(kios, rhos, Vs)

    rng = np.random.default_rng(seed)
    candidate_centers = rng.choice(n, size=min(n_candidates, n), replace=False)

    best_group = None
    best_score = -np.inf
    best_metrics = None

    for c in candidate_centers:
        neighbors = tree.query_ball_point(signal[c], r=radius)
        if len(neighbors) < group_size:
            continue

        chosen = [c]
        remaining = set(neighbors) - {c}
        while len(chosen) < group_size and remaining:
            best_next, best_next_dist = None, -1.0
            for j in remaining:
                d = min(np.linalg.norm(Z[j] - Z[k]) for k in chosen)
                if d > best_next_dist:
                    best_next_dist, best_next = d, j
            chosen.append(best_next)
            remaining.discard(best_next)

        if len(chosen) < group_size:
            continue

        curves = signal[chosen]
        diffs = curves[:, None, :] - curves[None, :, :]
        maxabs = float(np.max(np.abs(diffs)))
        rmse = float(np.sqrt(np.mean(diffs ** 2)))
        if maxabs > curve_maxabs_thresh:
            continue

        pair_dists = [
            float(np.linalg.norm(Z[chosen[i]] - Z[chosen[j]]))
            for i, j in itertools.combinations(range(group_size), 2)
        ]
        min_pd = min(pair_dists)
        if min_pd < min_pairwise_param_dist:
            continue

        score = min_pd  # maximize the worst-case parameter separation
        if score > best_score:
            best_score = score
            best_group = list(chosen)
            best_metrics = dict(maxabs=maxabs, rmse=rmse, min_param_dist=min_pd,
                                 param_dists=pair_dists)

    return best_group, best_metrics


def make_identifiability_figure(
    b_values, signal, kios, rhos, Vs, group_idx, metrics, outpath_base, dpi,
):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    for i, idx in enumerate(group_idx):
        color = HIGHLIGHT_COLORS_IDENT[i % len(HIGHLIGHT_COLORS_IDENT)]
        y = np.clip(signal[idx], LOG_FLOOR, None)
        label = (rf"$k_{{io}}$={kios[idx]:.2f} s$^{{-1}}$, "
                 rf"$\rho$={rhos[idx]:,.0f} $\mu$L$^{{-1}}$, "
                 rf"$V$={Vs[idx]:.2f} pL")
        ax.plot(b_values, y, "-", color=color, lw=STYLE["highlight_lw"] + 0.3,
                 zorder=3, label=label)
        ax.plot(b_values, y, "o", color=color, ms=STYLE["marker_size"] + 1,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=4)

    ax.set_xlim(b_values.min(), b_values.max())
    ax.set_ylim(LOG_FLOOR, 1.3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$b$ (s/mm$^2$)")
    ax.set_ylabel(r"$S/S_0$")
    ax.set_title(rf"Near-degenerate decay curves — $\delta$={TARGET_DELTA_MS:g} ms, "
                 rf"$\Delta$={TARGET_BIGDELTA_MS:g} ms")
    ax.grid(True, which="both", alpha=STYLE["grid_alpha"], lw=STYLE["grid_lw"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.92, loc="upper right", title="(k_io, ρ, V)")
    fig.tight_layout()

    fig.savefig(outpath_base + ".png", dpi=dpi)
    fig.savefig(outpath_base + ".pdf")
    plt.close(fig)
    print(f"[figure 2] saved {outpath_base}.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=DEFAULT_LIBRARY)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--group-size", type=int, default=4,
                     help="number of near-degenerate curves to find for figure 2 (3 or 4)")
    ap.add_argument("--curve-maxabs-thresh", type=float, default=0.01,
                     help="max allowed |S/S0| difference across the b-grid within the group")
    ap.add_argument("--min-param-dist", type=float, default=1.5,
                     help="min pairwise normalized log-parameter distance within the group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.library):
        raise FileNotFoundError(
            f"library not found: {args.library}\n"
            "(pass --library to point at your copy of "
            "madi_dense_universal_remediated.npz)"
        )
    os.makedirs(args.outdir, exist_ok=True)

    kios, rhos, Vs, is_free_water, b_values, signal = load_acquisition_subset(
        args.library, TARGET_DELTA_MS, TARGET_BIGDELTA_MS, B_MAX_S_MM2,
    )
    kios, rhos, Vs, signal = filter_cellular_entries(kios, rhos, Vs, is_free_water, signal)
    print(f"[library] {signal.shape[0]} cellular (k_io, rho, V) combinations after "
          f"dropping the free-water atom")

    # ---- Figure 1 ----
    make_reference_figure(
        b_values, signal, kios, rhos, Vs,
        os.path.join(args.outdir, "madi_signal_decay_reference"), args.dpi,
    )

    # ---- Figure 2 ----
    group_size = args.group_size
    group_idx, metrics = None, None
    thresh = args.curve_maxabs_thresh
    min_pd = args.min_param_dist
    # Relax thresholds gradually if the strict request finds nothing —
    # keeps the search robust to the exact library realization without
    # ever hand-picking specific points.
    for attempt in range(6):
        group_idx, metrics = find_degenerate_group(
            kios, rhos, Vs, signal,
            group_size=group_size,
            curve_maxabs_thresh=thresh,
            min_pairwise_param_dist=min_pd,
            seed=args.seed,
        )
        if group_idx is not None:
            break
        if group_size > 3:
            group_size -= 1
        else:
            thresh *= 1.5
            min_pd *= 0.85
        print(f"[search] no group found, relaxing to group_size={group_size}, "
              f"curve_maxabs_thresh={thresh:.4g}, min_param_dist={min_pd:.3g}")

    if group_idx is None:
        raise RuntimeError("could not find any sufficiently degenerate/separated group; "
                            "try relaxing --curve-maxabs-thresh or --min-param-dist")

    print(f"\nSelected {len(group_idx)} near-degenerate (k_io, rho, V) combinations "
          f"at (delta,Delta)=({TARGET_DELTA_MS:g},{TARGET_BIGDELTA_MS:g}) ms, "
          f"b in [0,{B_MAX_S_MM2:g}] s/mm^2:\n")
    header = f"{'k_io (1/s)':>12} {'rho (1/uL)':>14} {'V (pL)':>10} {'v_i':>8}"
    print(header)
    for idx in group_idx:
        vi = rhos[idx] * Vs[idx] * 1e-6
        print(f"{kios[idx]:12.3f} {rhos[idx]:14.1f} {Vs[idx]:10.4f} {vi:8.4f}")

    curves = signal[group_idx]
    diffs = curves[:, None, :] - curves[None, :, :]
    print(f"\nCurve-similarity metrics across the {len(group_idx)} selected curves "
          f"(b=0-{B_MAX_S_MM2:g} s/mm^2):")
    print(f"  overall max |S/S0 difference| : {metrics['maxabs']:.5f}")
    print(f"  overall RMSE                  : {metrics['rmse']:.5f}")
    print("  pairwise max |difference| (S/S0):")
    for (i, j) in itertools.combinations(range(len(group_idx)), 2):
        pd_ij = float(np.max(np.abs(diffs[i, j])))
        rmse_ij = float(np.sqrt(np.mean(diffs[i, j] ** 2)))
        print(f"    curve {i} vs curve {j}: max|diff|={pd_ij:.5f}  RMSE={rmse_ij:.5f}")
    print(f"\n  min pairwise normalized-log-parameter distance: {metrics['min_param_dist']:.3f}")

    make_identifiability_figure(
        b_values, signal, kios, rhos, Vs, group_idx, metrics,
        os.path.join(args.outdir, "madi_identifiability_degeneracy"), args.dpi,
    )


if __name__ == "__main__":
    main()
