#!/usr/bin/env python3
"""Plot each parameter's conditional CRLB independently at each (delta, Delta, b).

Run: python analysis/plot_conditional_crlb_vs_b.py
Figures show KNOWN S0 only; the CSV retains uncertain-S0 diagnostics.
Inputs: remediated v5 library metadata and existing full-domain Phase-1/2 caches.
Set MADI_FISHER_RUNS or RUNS below to the cache parent. No library rebuild or
new cache is needed. PNG/PDF figures, numerical CSV and JSON provenance go to
analysis/outputs/conditional_crlb_vs_b by default.

Each point uses REPEATS independent observations of ONE stored acquisition,
with no supporting DW columns and no cumulative b sweep. Each figure is a
different estimation problem: only its named tissue parameter is unknown;
the other two tissue parameters are KNOWN EXACTLY at the selected node.
Known S0: B_j = 1/F_jj. Uncertain S0: B_j = 1/I_j, where
I_j = F_jj - F_jS0**2/(F_S0S0 + lambda), the scalar Schur complement for the
two-parameter (theta_j, S0) problem. This is NOT diag(inv(F_3x3)), and the
singularity of the full three-tissue-parameter Fisher matrix is irrelevant.
An independent true-b0 reference supplies lambda = N0_EFF/sigma_b0**2. With
N0_EFF=0, one repeated acquisition cannot separate a tissue parameter from
unknown amplitude, so the uncertain-S0 bound is unavailable (not inverted).

Native coordinates are (ln rho, ln V, k_io), plotted in (k_io, ln rho, ln V)
order. Ordinates are VARIANCE bounds: k_io in s^-2; ln rho and ln V
dimensionless. Y_SCALE optionally uses a base-10 logarithmic axis (labelled
'log CRLB'); values/tick labels remain native CRLB, not log10-transformed data.
There is no square root, coordinate conversion or relative-parameter scaling.

Uses build_node_table/pair_contributions from scripts.run_fisher_phase2 for
canonical-neighbour derivatives on realized spacings and MC diagonal debias,
and madi.fisher_crlb for columns, noise and amplitude marginalization. See
docs/fisher_crlb_analysis_plan.md sections 2.1-2.8 (especially the conditional
1/F_jj comparison in 2.3), fisher_phase2.md and fisher_domain_audit.md.
Full-domain caches are read-only; optional hardware/signal masks are applied
only to this declared analysis. No interpolation or simulation is performed.

Defaults follow the requested experiment: sigma=1/50, three repeats, MC debias
OFF. Uncorrected finite-difference noise can inflate information and make
bounds optimistic. Optional debias uses Phase-2's endpoint-only variance:
CRN covariance is unavailable over most columns, so this correction can be
conservative. Finite-difference truncation error remains in either mode.
Noise is independent Gaussian on normalized signal at nominal S0=1; these
are local model bounds, not validated low-SNR magnitude-data performance.
Missing/nonpositive/nonfinite information is recorded and drawn as gaps, with
no pseudoinverse, regularization or invented finite values.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullLocator
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from madi.fisher_crlb import (
    PARAMETER_ORDER, amplitude_prior_precision, column_arrays,
    gradient_strength_t_per_m, packed_amplitude_marginal,
    read_column_domain, require_columns, te_noise_sigma,
)
from scripts.run_fisher_phase2 import build_node_table, pair_contributions

# ======================= USER-EDITABLE CONFIGURATION =======================
LIBRARY = REPO / "data/libraries/madi_dense_universal_remediated.npz"
RUNS = Path(os.environ.get("MADI_FISHER_RUNS", str(Path.home() / "madi_fisher_runs/full_domain")))
PHASE1, CACHE = RUNS / "phase1", RUNS / "cache"
# Targets: (rho cells/uL, V pL, k_io s^-1). Nearest complete-stencil node,
# using normalized log-rho/log-V/linear-k_io distance as in the notebooks.
# Actual selected coordinates are printed and saved; targets are not interpolated.
TISSUE_TARGETS = [(1e6, 0.5, 5.0)]
NODE_INDICES = None  # optional exact [(rho_index, V_index, k_io_index), ...]
TIMINGS_MS = [(4.0, 20.0)]  # fixed (delta, Delta) per curve; G varies with b
B_VALUES = list(range(500, 12001, 500))  # exact stored b in s/mm^2
COLUMN_IDS = None  # optional canonical flat IDs override TIMINGS_MS/B_VALUES
ORDER_BY = "b"  # 'b' or 'gradient'; x always b; separate curve per timing/node
REPEATS = 3  # independent observations of the same acquisition at each point
NOISE_MODEL = "constant"  # 'constant' or 'te'
CONSTANT_SIGMA = 1 / 50   # SD of each observation; repeated-mean SD = sigma/sqrt(REPEATS)
SNR_AT_TE_ZERO = 50.0    # used only for NOISE_MODEL='te'
T2_MS, T_EPI_MS = 80.0, 30.0
N0_EFF = 4.0  # CSV uncertain-S0 diagnostic only; does not affect plotted known-S0 bound
B0_TIMING_MS = (20.0, 50.0)  # TE for the b0 reference when NOISE_MODEL='te'
DEBIAS_MC = True  # optional Phase-2 MC diagonal correction
G_MAX_T_M = None   # optional evaluation-time gradient ceiling, e.g. 0.3
TRUST_FLOOR = None # optional minimum normalized signal, e.g. 0.015
RICIAN_MIN = None  # optional SNR threshold on the repeated mean, e.g. 3
INVALID_POLICY = "gap"  # 'gap' or 'raise'; reasons always saved in CSV
# Reject near-cancellation in scalar S0 marginalization, relative to F_jj.
# This does not impose a threshold on absolute tissue sensitivity.
SCHUR_RTOL = 1e-12
FIGSIZE, FONT_SIZE = (10.0, 6.5), 18
COLORS = ["#256abf", "#d75c2b", "#1b8a68", "#8755a3"]
LINEWIDTH, MARKERSIZE = 2.4, 4
# Inclusive plotting range in s/mm^2; None means no bound. Points outside this
# range are excluded BEFORE y-axis autoscaling; full selected data stay in CSV.
B_MIN, B_MAX = None, 8000.0
Y_SCALE = "log"  # 'linear' or 'log' (base-10 axis; stored CRLB unchanged)
# Limits always use native CRLB values, including when Y_SCALE='log'.
YLIMS = {"k_io": None, "log_rho": None, "log_V": None}
TITLES = {"k_io": "", "log_rho": "", "log_V": ""}  # optional titles
OUTPUT = REPO / "analysis/outputs/conditional_crlb_vs_b"
FORMATS, DPI = ("png",), 300
# ========================= END CONFIGURATION ==============================

PLOT_ORDER = ("k_io", "log_rho", "log_V")
DIAGONAL = np.array([0, 3, 5])  # packed order: rho-rho, V-V, k_io-k_io


def exact_column(triple, columns):
    hits = np.flatnonzero(np.all(np.array(columns).T == triple, axis=1))
    if len(hits) != 1:
        raise ValueError(f"No unique stored column for {triple}; use exact stored timings/b-values.")
    return int(hits[0])


def observation_sigma(columns):
    if NOISE_MODEL == "constant":
        if not np.isfinite(CONSTANT_SIGMA) or CONSTANT_SIGMA <= 0:
            raise ValueError("CONSTANT_SIGMA must be finite and positive.")
        return np.full(len(columns[0]), CONSTANT_SIGMA)
    if NOISE_MODEL != "te":
        raise ValueError("NOISE_MODEL must be constant or te.")
    if not np.all(np.isfinite([SNR_AT_TE_ZERO, T2_MS, T_EPI_MS])) or min(SNR_AT_TE_ZERO, T2_MS) <= 0 or T_EPI_MS < 0:
        raise ValueError("TE noise requires positive finite SNR/T2 and nonnegative finite TE overhead.")
    return te_noise_sigma(columns[0], columns[1], sigma0=1/SNR_AT_TE_ZERO,
                          T2_ms=T2_MS, t_epi_ms=T_EPI_MS)


def conditional_bounds(tissue, amplitude, prior, rtol=SCHUR_RTOL):
    """Three separate scalar problems; no joint tissue-matrix rank requirement.

    Input is the accumulated information from repeats of ONE column. Return
    per-regime (variance bounds, scalar information, reasons) in native order.
    Explicitly reject prior=0 marginal bounds: their exact information is zero
    without MC debias, regardless of floating-point cancellation residuals.
    """
    fixed = np.asarray(tissue)[DIAGONAL]
    marginal = packed_amplitude_marginal(tissue, amplitude[:3], amplitude[3], prior)[DIAGONAL]
    output = {}
    for regime, information in (("fixed", fixed), ("marginal", marginal)):
        bounds = np.full(3, np.nan)
        reasons = []
        for j, value in enumerate(information):
            if not np.isfinite(value) or not np.all(np.isfinite(amplitude)):
                reason = "nonfinite information"
            elif regime == "marginal" and prior == 0:
                reason = "unknown S0 and tissue parameter inseparable without b0 reference"
            elif value <= 0:
                reason = "nonpositive scalar information (zero sensitivity or MC correction)"
            elif regime == "marginal" and value <= rtol * abs(fixed[j]):
                reason = "scalar S0 Schur complement numerically unresolved"
            else:
                with np.errstate(over="ignore", divide="ignore"):
                    bound = 1.0 / value
                reason = "ok" if np.isfinite(bound) and bound > 0 else "nonfinite variance bound"
                if reason == "ok":
                    bounds[j] = bound
            reasons.append(reason)
        output[regime] = (bounds, information, reasons)
    return output


def load_inputs():
    paths = [LIBRARY, PHASE1 / "phase1_manifest.json", CACHE / "phase2_cache_manifest.json",
             CACHE / "ensemble_means_subset.npy"]
    paths += [PHASE1 / f"samples_{a}_k1.npy" for a in ("rho", "V", "k_io")]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input:\n" + "\n".join(missing) +
                                "\nSet MADI_FISHER_RUNS to the existing full_domain run (phase1/ and cache/). "
                                "See docs/fisher_domain_audit.md section 7; no library rebuild is needed.")
    domain = read_column_domain(json.loads(paths[1].read_text()))
    cache_meta = json.loads(paths[2].read_text())
    if not domain.is_complete or not cache_meta["column_domain"]["is_complete_stored_grid"]:
        raise ValueError("Use the complete unrestricted Phase-1/2 caches, not a legacy restricted substrate.")
    with np.load(LIBRARY, allow_pickle=False) as lib:
        columns = column_arrays(lib["pair_deltas"], lib["pair_Deltas"], lib["b_values"])
        table = build_node_table(PHASE1, lib)
        n_entries = len(lib["is_free_water"])
    arrays = []
    for member in ("vectors", "signal_variance"):
        path = CACHE / f"{member}_selected_T.npy"
        if path.is_file():
            array = np.load(path, mmap_mode="r")
        else:
            path = CACHE / f"{member}_selected.npy"
            array = np.load(path, mmap_mode="r").T  # read-only fallback, no new cache
        if array.shape != (len(columns[0]), n_entries):
            raise ValueError(f"Cache/library shape mismatch: {path}: {array.shape}")
        arrays.append(array)
    n_ensembles = np.load(paths[3], mmap_mode="r").shape[1]
    return columns, table, arrays, n_ensembles, domain, cache_meta


def select_nodes(table):
    rows = []
    if NODE_INDICES is not None:
        for node in NODE_INDICES:
            hits = np.flatnonzero(np.all(table["nodes"] == node, axis=1))
            if len(hits) != 1:
                raise ValueError(f"Node {node} lacks complete k=1 stencils; choose an interior node.")
            rows.append(int(hits[0]))
    else:
        for rho, V, kio in TISSUE_TARGETS:
            if not np.all(np.isfinite([rho, V, kio])) or min(rho, V) <= 0:
                raise ValueError("Tissue targets must be finite, with positive rho and V.")
            cost = sum(((fn(table[key]) - fn(target)) / np.ptp(fn(table[key]))) ** 2
                       for key, target, fn in (("rho", rho, np.log), ("V", V, np.log), ("kio", kio, np.asarray)))
            rows.append(int(np.argmin(cost)))
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("Select at least one distinct node; targets must not map to the same node.")
    return rows


def main():
    if not isinstance(REPEATS, int) or REPEATS < 1:
        raise ValueError("REPEATS must be a positive integer.")
    if not np.isfinite(N0_EFF) or N0_EFF < 0 or not 0 < SCHUR_RTOL < 1:
        raise ValueError("N0_EFF must be finite/nonnegative, and 0 < SCHUR_RTOL < 1.")
    if ORDER_BY not in ("b", "gradient") or INVALID_POLICY not in ("gap", "raise"):
        raise ValueError("Use ORDER_BY=b/gradient and INVALID_POLICY=gap/raise.")
    for key, val in (("G_MAX_T_M", G_MAX_T_M), ("TRUST_FLOOR", TRUST_FLOOR), ("RICIAN_MIN", RICIAN_MIN)):
        if val is not None and (not np.isfinite(val) or val < 0):
            raise ValueError(f"{key} must be None or finite/nonnegative.")
    assert tuple(PARAMETER_ORDER) == ("log_rho", "log_V", "k_io")
    columns, table, arrays, n_ensembles, domain, cache_meta = load_inputs()
    rows = select_nodes(table)
    selected = ([exact_column((d, D, b), columns) for d, D in TIMINGS_MS for b in B_VALUES]
                if COLUMN_IDS is None else list(COLUMN_IDS))
    if not selected or len(set(selected)) != len(selected) or any(int(c) != c for c in selected):
        raise ValueError("Select nonempty, unique integer column IDs; use REPEATS for repeated measurements.")
    selected = np.asarray(selected, dtype=int)
    if np.any(selected < 0) or np.any(selected >= len(columns[0])):
        raise ValueError("Column ID is outside the stored library.")
    if np.any(columns[2][selected] <= 0):
        raise ValueError("Select b>0 for tissue curves; use N0_EFF for the independent b0 reference.")
    positions = require_columns(domain, selected, "conditional CRLB", *columns)
    sigma = observation_sigma(columns)
    gradient = gradient_strength_t_per_m(*columns)
    prior = amplitude_prior_precision(N0_EFF, float(sigma[exact_column((*B0_TIMING_MS, 0), columns)]))
    subtable = {key: value[rows] for key, value in table.items() if isinstance(value, np.ndarray)}
    records = []
    for col, pos in zip(selected, positions):
        # Audit-backed derivatives/variance; n_ensembles is read, never hard-coded.
        tissue, amplitude, correction = pair_contributions(
            subtable, *arrays, np.array([pos]), REPEATS / sigma[col]**2, n_ensembles,
            -np.inf if TRUST_FLOOR is None else TRUST_FLOOR,
            -np.inf if RICIAN_MIN is None else RICIAN_MIN*sigma[col]/np.sqrt(REPEATS))
        if not DEBIAS_MC:
            tissue[..., DIAGONAL] += correction
        weight = REPEATS / sigma[col]**2
        # Diagnostic only: compare squared sensitivity with the helper's
        # endpoint-only MC derivative variance. Neither rescales plotted bounds.
        derivative_squared = (tissue[0][:, DIAGONAL] + (correction[0] if DEBIAS_MC else 0)) / weight
        derivative_variance = correction[0] / weight
        for n, row in enumerate(rows):
            result = conditional_bounds(tissue[0, n], amplitude[0, n], prior)
            for regime, (bounds, info, reasons) in result.items():
                for j, parameter in enumerate(PARAMETER_ORDER):
                    reason = reasons[j]
                    if G_MAX_T_M is not None and gradient[col] > G_MAX_T_M:
                        reason = "column exceeds declared gradient ceiling"
                    elif amplitude[0, n, 3] == 0:
                        reason = "zero signal or column excluded by signal/SNR mask"
                    if reason != "ok" and INVALID_POLICY == "raise":
                        raise ValueError(f"Node {row}, column {col}, {parameter}, {regime}: {reason}")
                    records.append(dict(node_row=int(row), canonical_node=table["nodes"][row].tolist(),
                                        rho=float(table["rho"][row]), V=float(table["V"][row]), k_io=float(table["kio"][row]),
                                        column=int(col), delta_ms=float(columns[0][col]), Delta_ms=float(columns[1][col]),
                                        b_s_mm2=float(columns[2][col]), gradient_T_m=float(gradient[col]),
                                        sigma_per_observation=float(sigma[col]), repeats=REPEATS, parameter=parameter,
                                        regime=regime, scalar_information=float(info[j]),
                                        derivative_squared=float(derivative_squared[n, j]),
                                        derivative_variance_endpoint_only=float(derivative_variance[n, j]),
                                        variance_bound=float(bounds[j]) if reason == "ok" else float("nan"),
                                        mc_debias=DEBIAS_MC, reason=reason))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "conditional_crlb_values.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    config = {k: v for k, v in globals().items() if k.isupper() and k not in ("PARAMETER_ORDER", "DIAGONAL")}
    nodes = [{k: table[k][row].tolist() for k in ("nodes", "rho", "V", "kio")} for row in rows]
    provenance = dict(configuration=config, interpretation="one unknown tissue parameter; other two known exactly",
                      plot_order=PLOT_ORDER, plotted_regime="fixed (known S0)", native_order=list(PARAMETER_ORDER), selected_nodes=nodes,
                      selected_columns=selected.tolist(), domain=domain.as_dict(), cache_manifest=cache_meta,
                      n_ensembles=int(n_ensembles), amplitude_prior_precision=prior,
                      variance_units={"k_io": "s^-2", "log_rho": "dimensionless", "log_V": "dimensionless"})
    (OUTPUT / "run_config.json").write_text(json.dumps(provenance, indent=2, default=str)+"\n", encoding="utf-8")
    print("Actual tissue nodes:", nodes)
    draw_figures(records)
    print(f"{sum(r['reason'] != 'ok' for r in records)}/{len(records)} parameter/column/regime results undefined; see CSV.")


def readable_log_ticks(ax):
    """Label native values, including intermediate values for sub-decade ranges."""
    low, high = ax.get_ylim()
    locator = LogLocator(base=10, subs=(1, 2, 5), numticks=30)
    ticks = locator.tick_values(low, high)
    ticks = ticks[(ticks >= low) & (ticks <= high)]
    if len(ticks) < 4:
        # A narrow range may contain only one power of ten. Choose pleasant
        # native-value labels; their positions still follow the true log scale.
        ticks = MaxNLocator(nbins=5, min_n_ticks=3).tick_values(low, high)
        ticks = ticks[(ticks >= low) & (ticks <= high) & (ticks > 0)]
    elif len(ticks) > 8:
        ticks = LogLocator(base=10, numticks=8).tick_values(low, high)
        ticks = ticks[(ticks >= low) & (ticks <= high)]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.4g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def draw_figures(records):
    if Y_SCALE not in ("linear", "log"):
        raise ValueError("Y_SCALE must be linear or log.")
    if any(v is not None and not np.isfinite(v) for v in (B_MIN, B_MAX)):
        raise ValueError("B_MIN/B_MAX must be finite or None.")
    if B_MIN is not None and B_MAX is not None and B_MIN >= B_MAX:
        raise ValueError("B_MIN must be smaller than B_MAX.")
    records = [r for r in records if r["regime"] == "fixed"
               and (B_MIN is None or r["b_s_mm2"] >= B_MIN)
               and (B_MAX is None or r["b_s_mm2"] <= B_MAX)]
    if not records:
        raise ValueError("No selected b-values lie within B_MIN/B_MAX; widen the plotting range.")
    for parameter, limits in YLIMS.items():
        if limits is not None and (len(limits) != 2 or not np.all(np.isfinite(limits))
                                   or limits[0] >= limits[1] or (Y_SCALE == "log" and limits[0] <= 0)):
            raise ValueError(f"Invalid YLIMS for {parameter}; log-axis limits must be positive.")
    plt.rcParams.update({"font.size": FONT_SIZE, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42})
    groups = sorted(set((r["node_row"], r["delta_ms"], r["Delta_ms"]) for r in records))
    node_ids = sorted(set(r["node_row"] for r in records))
    symbols = {"k_io": r"k_{io}", "log_rho": r"\ln\rho", "log_V": r"\ln V"}
    for parameter in PLOT_ORDER:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        subset = [r for r in records if r["parameter"] == parameter and r["regime"] == "fixed"]
        for i, (node, d, D) in enumerate(groups):
            color = COLORS[i % len(COLORS)]
            for regime, style, marker in (("fixed", "-", "o"),):
                curve = [r for r in subset if (r["node_row"], r["delta_ms"], r["Delta_ms"], r["regime"]) == (node, d, D, regime)]
                curve.sort(key=lambda r: r["b_s_mm2"] if ORDER_BY == "b" else r["gradient_T_m"])
                label = rf"$\delta$={d:g}, $\Delta$={D:g} ms"
                if len(node_ids) > 1:
                    label += f"; node {node}"
                if not any(np.isfinite(r["variance_bound"]) for r in curve):
                    label += " (undefined)"
                ax.plot([r["b_s_mm2"] for r in curve], [r["variance_bound"] for r in curve],
                        style, marker=marker, color=color, linewidth=LINEWIDTH, markersize=MARKERSIZE,
                        markerfacecolor="white" if regime == "fixed" else color, label=label)
        prefix = "log CRLB" if Y_SCALE == "log" else "CRLB"
        ax.set(xlabel=r"b-value (s/mm$^2$)", ylabel=rf"{prefix} (${symbols[parameter]}$)",
               title=TITLES[parameter], yscale=Y_SCALE)
        if not any(np.isfinite(r["variance_bound"]) for r in subset):
            ax.set_yticks([])
            ax.text(.5, .5, "No finite conditional CRLB\nSee CSV for failure reasons", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FONT_SIZE-2)
        if B_MIN is not None or B_MAX is not None:
            ax.set_xlim(left=B_MIN, right=B_MAX)
        if YLIMS[parameter] is not None:
            ax.set_ylim(YLIMS[parameter])
        ax.grid(axis="y", which="major", alpha=.12, linewidth=.8)
        if Y_SCALE == "linear":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        elif any(np.isfinite(r["variance_bound"]) for r in subset):
            readable_log_ticks(ax)
        if len(groups) > 1:
            ax.legend(fontsize=FONT_SIZE-5, loc="best")
        nodes = []
        for node in node_ids:
            r = next(r for r in records if r["node_row"] == node)
            text = rf"$\rho$ = {r['rho']:.3g} cells/$\mu$L,  $V$ = {r['V']:.3g} pL,  $k_{{io}}$ = {r['k_io']:g} s$^{{-1}}$"
            nodes.append((f"Node {node}: " if len(node_ids) > 1 else "") + text)
        fig.text(.5, .025, "\n".join(nodes), fontsize=13, ha="center", va="bottom", color="#444444")
        if any(r["reason"] != "ok" for r in subset):
            ax.text(.02, .98, "Gaps: undefined CRLB", transform=ax.transAxes,
                    fontsize=10, va="top", color="#777777")
        fig.tight_layout(rect=(0, .065 + .03*len(node_ids), 1, 1))
        for extension in FORMATS:
            path = OUTPUT / f"conditional_crlb_{parameter}.{extension}"
            # Match the project's plotting convention: Windows previewers can
            # hold the old file open, so save a temporary image then replace it.
            temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
            fig.savefig(temporary, dpi=DPI, facecolor="white")
            os.replace(temporary, path)
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"Conditional CRLB plotting failed: {error}") from error
