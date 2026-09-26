#!/usr/bin/env python3
"""Three slide figures: native CRLB versus the b-value of one added measurement.

Run ``python analysis/plot_log_crlb_vs_b.py`` in the project Python environment.
Inputs are the remediated v5 library metadata and existing full-domain Phase-1
and Phase-2 caches; set MADI_FISHER_RUNS or edit CONFIGURATION below to find them.
No simulation, cache generation, interpolation, or protocol optimization occurs.

Each point is a FIXED supporting DW protocol PLUS SWEEP_REPEATS independent
measurements at the displayed b, not a cumulative b sweep. Empty support with
three identical repeats is a rank-deficiency diagnostic: repetition multiplies
information but adds no independent sensitivity directions. The endpoint-only
MC debias can make this singular ideal Fisher matrix indefinite. With
ALLOW_UNDEFINED_FIGURES enabled, all-invalid curves produce explicitly labelled
diagnostic figures and CSV eigenvalues, never finite substitute bounds.
The repository defines Fisher information over column sets, but prescribes no
unique b-axis plot. This illustrative construction follows analysis-plan
sections 2.1, 2.5-2.8 and fisher_phase2.md section 2. A lone scalar measurement
has rank <= 1; even adding b=0 cannot identify three tissue coordinates.

Internal coordinates are (ln rho, ln V, k_io), with k_io in s^-1. Figures are
explicitly reordered to (k_io, ln rho, ln V). The ordinate is the untransformed
VARIANCE bound diag(F^-1), on a linear axis: s^-2 for k_io and dimensionless
for the two natural-log coordinates. No outer logarithm, square root, base-10
coordinate conversion, or relative-k_io rescaling is applied to plotted bounds.
The SD ratios in the CSV remain separate amplitude-uncertainty diagnostics.

Reuse Phase-2's endpoint-only MC diagonal debias (signal_variance/n_ensembles,
realized stencil spacings), unless DEBIAS_MC=False for a labelled diagnostic.
Disabling it restores the helper's subtracted diagonal; finite-difference noise
can then inflate Fisher information and make bounds overly optimistic.
CRN covariance exists only at diagnostic columns;
omitting positive covariance is conservative, as quantified in fisher_phase2.md.
Finite differences still have truncation error; these are local approximate
bounds, not guarantees of estimator performance. Both fixed S0 and one SHARED
unknown S0 with independent true-b0 prior precision are reported, including
their SD ratio. No separate amplitude per timing is introduced.

Defaults use independent Gaussian normalized-signal noise with TE/T2 weighting,
no hardware/trust/Rician exclusions (the barebones-notebook convention). They
are model-conditioned illustrations, not scanner recommendations or validated
low-SNR magnitude bounds. Optional masks act only at evaluation time. Invalid
or non-positive-definite matrices are gaps, never pseudoinverted, regularized,
clipped to a finite CRLB, or connected across. CSV and JSON preserve all points,
selection, noise assumptions, actual nodes, and failure reasons.
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
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from madi.fisher_crlb import (
    PARAMETER_ORDER, amplitude_prior_precision, column_arrays,
    gradient_strength_t_per_m, nondimensionalized_fisher,
    packed_amplitude_marginal, packed_inverse_diagonal, read_column_domain,
    require_columns, te_noise_sigma,
)
from scripts.run_fisher_phase2 import build_node_table, pair_contributions

# ======================= USER-EDITABLE CONFIGURATION =======================
LIBRARY = REPO / "data/libraries/madi_dense_universal_remediated.npz"
RUNS = Path(os.environ.get("MADI_FISHER_RUNS", str(Path.home() / "madi_fisher_runs/full_domain")))
PHASE1, CACHE = RUNS / "phase1", RUNS / "cache"
# Targets (rho cells/uL, V pL, k_io s^-1), nearest complete-stencil node.
# Selection uses the notebooks' normalized log-rho/log-V/linear-k_io distance.
# Use exact canonical triples instead by setting NODE_INDICES (ir, iv, ik).
TISSUE_TARGETS = [(1.0e6, 0.5, 50.0)]
NODE_INDICES = None
# One curve per fixed (delta, Delta) in ms. G varies with b by the PGSE formula.
SWEEP_TIMINGS = [(4.0, 20.0)]
B_VALUES = list(range(500, 12001, 500))  # exact stored b, s/mm^2; no snapping
# Optional canonical flattened column IDs override SWEEP_TIMINGS/B_VALUES.
# IDs are grouped by timing automatically, preserving same-b distinctions.
SWEEP_COLUMN_IDS = None
# Fixed support: (delta ms, Delta ms, list of exact b in s/mm^2).
SUPPORT = [(20.0, 50.0, [1000, 3000, 6000])]
# SUPPORT = []
ORDER_BY = "b"  # 'b' or 'gradient'; x remains b, separate fixed-timing curves
GRADIENT_RANGE_T_M = (0.0, None)  # optional conditional limits, both support/sweep
SWEEP_REPEATS = 1  # independent repeats of the SAME (delta, Delta, b)
AVERAGES = 1.0  # averages per repeat; 3 repeats x 1 average = 3 DW observations
NOISE_MODEL = "constant"  # 'te' or 'constant'; constant ignores T2_MS and T_EPI_MS
CONSTANT_SIGMA = 1 / 50  # normalized-signal SD per independent measurement
# Effective SD after averaging is CONSTANT_SIGMA/sqrt(AVERAGES).
# Set AVERAGES=1 if 1/50 is the desired final SD at each plotted column.
SNR_AT_TE_ZERO = 50.0
T2_MS, T_EPI_MS = 80.0, 30.0
TRUST_FLOOR = None  # optional normalized signal floor, e.g. 0.015
RICIAN_MIN = None   # optional magnitude SNR screen, e.g. 3; averaging-aware
N0_EFF = 4.0        # 0 = wholly unknown S0; independent of the DW budget
B0_TIMING = (20.0, 50.0)  # actual b=0 prior reference, not extrapolated b=50
# Plot diag(F^-1) directly, without logarithms, square roots, or rescaling.
INVALID_POLICY = "gap" # 'gap' (record/warn) or 'raise' on any invalid point
DEBIAS_MC = True  # False: uncorrected J^T W J diagnostic; may inflate information
ALLOW_UNDEFINED_FIGURES = True  # diagnostic output even when no finite line exists
RANK_RTOL = 1e-12       # numerical rank gate on D F D, D=(1,1,max(k_io,5))
FIGSIZE = (10.0, 6.3)
FONT_SIZE = 18
COLORS = ["#256abf", "#d75c2b", "#1b8a68", "#8755a3"]
LINEWIDTH, MARKERSIZE = 2.4, 4
XLIM = None
YLIMS = {"k_io": None, "log_rho": None, "log_V": None}
TITLES = {"k_io": r"Exchange rate $k_{io}$",
          "log_rho": r"Log cell density $\ln\rho$",
          "log_V": r"Log cell volume $\ln V$"}
OUTPUT = REPO / "analysis/outputs/crlb_vs_b"
FORMATS, DPI = ("png",), 300
# ========================= END CONFIGURATION ==============================

PLOT_ORDER = ("k_io", "log_rho", "log_V")


def exact_column(delta, Delta, b, columns):
    matches = np.flatnonzero((columns[0] == delta) & (columns[1] == Delta) & (columns[2] == b))
    if len(matches) != 1:
        raise ValueError(f"No unique stored column for ({delta}, {Delta}, {b}); edit selection to stored values.")
    return int(matches[0])


def checked_bound(packed, kio):
    """Check numerical rank before obtaining the established inverse diagonal."""
    if not np.all(np.isfinite(packed)):
        return np.full(3, np.nan), "nonfinite Fisher matrix"
    eig = np.linalg.eigvalsh(nondimensionalized_fisher(packed, max(kio, 5.0)))
    if eig[-1] <= 0 or eig[0] <= 0:
        return np.full(3, np.nan), "Fisher not positive definite"
    if eig[0] <= RANK_RTOL * eig[-1]:
        return np.full(3, np.nan), "numerically rank deficient"
    bound, _, positive = packed_inverse_diagonal(packed)
    if not positive or not np.all(np.isfinite(bound) & (bound > 0)):
        return np.full(3, np.nan), "invalid inverse diagonal"
    return bound, "ok"


def noise_sigma(columns):
    """One-average normalized SD; the same noise model also prices the b0 prior."""
    if NOISE_MODEL == "constant":
        if not np.isfinite(CONSTANT_SIGMA) or CONSTANT_SIGMA <= 0:
            raise ValueError("CONSTANT_SIGMA must be finite and positive.")
        return np.full(len(columns[0]), CONSTANT_SIGMA, dtype=float)
    if NOISE_MODEL != "te":
        raise ValueError("NOISE_MODEL must be 'te' or 'constant'.")
    return te_noise_sigma(columns[0], columns[1], sigma0=1/SNR_AT_TE_ZERO,
                          T2_ms=T2_MS, t_epi_ms=T_EPI_MS)


def main():
    if not isinstance(SWEEP_REPEATS, int) or SWEEP_REPEATS < 1:
        raise ValueError("SWEEP_REPEATS must be a positive integer.")
    if INVALID_POLICY not in ("gap", "raise"):
        raise ValueError("INVALID_POLICY must be gap/raise.")
    if ORDER_BY not in ("b", "gradient"):
        raise ValueError("Choose ORDER_BY b/gradient.")
    if min(AVERAGES, SNR_AT_TE_ZERO, T2_MS) <= 0 or N0_EFF < 0 or T_EPI_MS < 0:
        raise ValueError("Noise scales/averages must be positive; N0_EFF and TE overhead nonnegative.")
    if not np.all(np.isfinite([AVERAGES, SNR_AT_TE_ZERO, T2_MS, N0_EFF, T_EPI_MS])):
        raise ValueError("Noise settings must be finite (fixed S0 is always reported separately).")
    if not 0 < RANK_RTOL < 1:
        raise ValueError("RANK_RTOL must be between zero and one.")
    assert tuple(PARAMETER_ORDER) == ("log_rho", "log_V", "k_io")
    order = [PARAMETER_ORDER.index(p) for p in PLOT_ORDER]
    required = [LIBRARY, PHASE1 / "phase1_manifest.json", CACHE / "phase2_cache_manifest.json",
                CACHE / "ensemble_means_subset.npy"]
    required += [PHASE1 / f"samples_{axis}_k1.npy" for axis in ("rho", "V", "k_io")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required data:\n" + "\n".join(missing) +
                                "\nSet MADI_FISHER_RUNS (or RUNS) to the existing full_domain run; "
                                "see docs/fisher_domain_audit.md section 7. Do not rebuild the library.")
    manifest = json.loads(required[1].read_text())
    domain = read_column_domain(manifest)
    cache_manifest = json.loads(required[2].read_text())
    if not domain.is_complete or not cache_manifest["column_domain"]["is_complete_stored_grid"]:
        raise ValueError("Use the complete, unrestricted Phase-1/2 substrate; legacy restricted caches are refused.")
    with np.load(LIBRARY, allow_pickle=False) as lib:
        columns = column_arrays(lib["pair_deltas"], lib["pair_Deltas"], lib["b_values"])
        table = build_node_table(PHASE1, lib)
        n_entries = len(lib["is_free_water"])
    arrays = []
    for member in ("vectors", "signal_variance"):
        path = CACHE / f"{member}_selected_T.npy"
        if path.exists():
            array = np.load(path, mmap_mode="r")
        else:
            path = CACHE / f"{member}_selected.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; point CACHE to the existing Phase-2 cache.")
            array = np.load(path, mmap_mode="r").T  # read-only view; never build a second cache
        if array.shape != (len(columns[0]), n_entries):
            raise ValueError(f"Cache/library shape mismatch: {path}: {array.shape}")
        arrays.append(array)
    n_ensembles = np.load(required[3], mmap_mode="r").shape[1]
    if NODE_INDICES is not None:
        rows = []
        for node in NODE_INDICES:
            match = np.flatnonzero(np.all(table["nodes"] == node, axis=1))
            if len(match) != 1:
                raise ValueError(f"Node {node} lacks a complete k=1 stencil. Choose an interior node.")
            rows.append(int(match[0]))
    else:
        rows = []
        for rho, V, kio in TISSUE_TARGETS:
            if rho <= 0 or V <= 0:
                raise ValueError("Target rho and V must be positive.")
            cost = sum(((f(table[key]) - f(target)) / np.ptp(f(table[key]))) ** 2
                       for key, target, f in (("rho", rho, np.log), ("V", V, np.log),
                                              ("kio", kio, np.asarray)))
            rows.append(int(np.argmin(cost)))
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("Select at least one distinct node; multiple targets must not snap to the same node.")
    support = [exact_column(d, D, b, columns) for d, D, bs in SUPPORT for b in bs]
    sweep = ([exact_column(d, D, b, columns) for d, D in SWEEP_TIMINGS for b in B_VALUES]
             if SWEEP_COLUMN_IDS is None else list(SWEEP_COLUMN_IDS))
    if not sweep or len(set(sweep)) != len(sweep) or len(set(support)) != len(support):
        raise ValueError("Use nonempty, unique sweep columns and unique support columns (set AVERAGES for repeats).")
    selected = np.asarray(sorted(set(support + sweep)), dtype=int)
    if any(int(c) != c for c in support + sweep):
        raise ValueError("Canonical column IDs must be integers.")
    if np.any(selected < 0) or np.any(selected >= len(columns[0])):
        raise ValueError("Column ID outside the stored library.")
    if np.any(columns[2][selected] <= 0):
        raise ValueError("DW columns must have b > 0; use N0_EFF for the independent b0 reference.")
    positions = require_columns(domain, selected, "slide sweep", *columns)
    gradient = gradient_strength_t_per_m(*columns)
    sigma = noise_sigma(columns)
    b0 = exact_column(*B0_TIMING, 0, columns)
    prior = amplitude_prior_precision(N0_EFF, float(sigma[b0]))
    # Small node table; pair_contributions still uses the original library entry IDs.
    subtable = {key: value[rows] for key, value in table.items() if isinstance(value, np.ndarray)}
    contributions = {}
    for col, pos in zip(selected, positions):
        t, a, subtracted = pair_contributions(subtable, *arrays, np.array([pos]),
                                    AVERAGES / sigma[col]**2, n_ensembles,
                                    -np.inf if TRUST_FLOOR is None else TRUST_FLOOR,
                                    -np.inf if RICIAN_MIN is None else RICIAN_MIN*sigma[col]/np.sqrt(AVERAGES))
        if not DEBIAS_MC:
            # Phase-2 returns the exact weighted/masked correction it subtracted.
            # Packed diagonal positions correspond to (ln rho, ln V, k_io).
            t[..., [0, 3, 5]] += subtracted
        contributions[int(col)] = (t[0], a[0])
    records, curves = [], []
    timing_groups = sorted(set((columns[0][c], columns[1][c]) for c in sweep))
    for n, row in enumerate(rows):
        node_label = (rf"$\rho$={table['rho'][row]:.3g} cells/$\mu$L, "
                      rf"$V$={table['V'][row]:.3g} pL, $k_{{io}}$={table['kio'][row]:g} s$^{{-1}}$")
        for d, D in timing_groups:
            cols = [c for c in sweep if columns[0][c] == d and columns[1][c] == D]
            cols.sort(key=lambda c: columns[2][c] if ORDER_BY == "b" else gradient[c])
            values = {"fixed": [], "marginal": []}
            for c in cols:
                protocol = support + [c] * SWEEP_REPEATS
                low, high = GRADIENT_RANGE_T_M
                hardware_ok = all((low is None or gradient[q] >= low) and
                                  (high is None or gradient[q] <= high) for q in protocol)
                t = sum(contributions[q][0][n] for q in protocol)
                a = sum(contributions[q][1][n] for q in protocol)
                used = sum(contributions[q][1][n, 3] > 0 for q in protocol)
                added_used = bool(contributions[c][1][n, 3] > 0)
                bounds = {}
                for regime in values:
                    F = t if regime == "fixed" else packed_amplitude_marginal(t, a[:3], a[3], prior)
                    scaled = nondimensionalized_fisher(F, max(float(table["kio"][row]), 5.0))
                    eig = np.linalg.eigvalsh(scaled) if np.all(np.isfinite(scaled)) else np.full(3, np.nan)
                    bound, reason = checked_bound(F, table["kio"][row]) if hardware_ok else (
                        np.full(3, np.nan), "protocol outside declared gradient range")
                    bounds[regime] = bound
                    values[regime].append(bound[order])
                    if reason != "ok" and INVALID_POLICY == "raise":
                        raise ValueError(f"Node {row}, column {c}, {regime}: {reason}")
                    for j, p in enumerate(PARAMETER_ORDER):
                        records.append(dict(node_row=int(row), canonical_node=table["nodes"][row].tolist(),
                                            rho=float(table["rho"][row]), V=float(table["V"][row]),
                                            k_io=float(table["kio"][row]), column=int(c), delta_ms=float(d),
                                            Delta_ms=float(D), b_s_mm2=float(columns[2][c]),
                                            gradient_T_m=float(gradient[c]), regime=regime, parameter=p,
                                            variance_bound=float(bound[j]), reason=reason,
                                            mc_debias=DEBIAS_MC,
                                            scaled_fisher_eigenvalues=eig.tolist(),
                                            fisher_packed=F.tolist(),
                                            used_dw_columns=int(used), added_column_used=added_used))
                ratio = np.sqrt(bounds["marginal"]/bounds["fixed"])
                for rec in records[-6:]:
                    rec["sd_ratio_marginal_fixed"] = float(ratio[PARAMETER_ORDER.index(rec["parameter"])])
            curves.append(dict(b=columns[2][cols], values=values,
                               label=rf"$\delta$={d:g}, $\Delta$={D:g} ms", node=node_label))
    # Normally refuse an empty curve; explicit diagnostic mode permits labelled
    # undefined figures. No y-value is invented for a missing CRLB.
    for curve in curves:
        for regime, values in curve["values"].items():
            valid = np.all(np.isfinite(values), axis=1)
            if not np.any(valid[:-1] & valid[1:]) and not ALLOW_UNDEFINED_FIGURES:
                raise ValueError(f"No adjacent valid CRLB points for {curve['label']}, {regime}, {curve['node']}. "
                                 "Choose an informative multi-timing SUPPORT/node or review declared masks; "
                                 "no pseudoinverse or regularization is used.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "crlb_values.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    config = {key: value for key, value in globals().items() if key.isupper() and key != "PARAMETER_ORDER"}
    provenance = dict(configuration=config, internal_order=list(PARAMETER_ORDER), plot_order=PLOT_ORDER,
                      plotted_quantity="untransformed variance bound diag(F^-1), linear axis",
                      phase1_domain=domain.as_dict(), cache_manifest=cache_manifest,
                      support_columns=support, sweep_columns=sweep, n_ensembles=int(n_ensembles),
                      amplitude_prior_precision=prior,
                      interpretation=f"fixed support plus {SWEEP_REPEATS} independent repeats of one DW column",
                      selected_nodes=[{k: table[k][row].tolist() for k in ("nodes", "rho", "V", "kio")} for row in rows])
    (OUTPUT / "run_config.json").write_text(json.dumps(provenance, indent=2, default=str)+"\n", encoding="utf-8")
    plt.rcParams.update({"font.size": FONT_SIZE, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    invalid = sum(r["reason"] != "ok" for r in records) // 3
    masked = sum(not r["added_column_used"] for r in records) // 6
    for j, parameter in enumerate(PLOT_ORDER):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for i, curve in enumerate(curves):
            label = curve["label"] + ("\n" + curve["node"] if len(rows) > 1 else "")
            # Draw known S0 last with open markers so nearly coincident bounds
            # remain recognizable without artificially offsetting either curve.
            for regime, style in (("marginal", "-"), ("fixed", "--")):
                ax.plot(curve["b"], np.asarray(curve["values"][regime])[:, j],
                        style, marker="o" if regime == "fixed" else "s",
                        markerfacecolor="white" if regime == "fixed" else COLORS[i % len(COLORS)],
                        color=COLORS[i % len(COLORS)], linewidth=LINEWIDTH,
                        markersize=MARKERSIZE, label=label + ("; known $S_0$" if regime == "fixed" else "; uncertain $S_0$"))
        ylabel = (r"CRLB of $k_{io}$ (s$^{-2}$)" if parameter == "k_io" else
                  (r"CRLB of $\ln\rho$ (dimensionless)" if parameter == "log_rho" else
                   r"CRLB of $\ln V$ (dimensionless)"))
        ax.set(xlabel=r"Added measurement b-value (s/mm$^2$)",
               ylabel=ylabel, yscale="linear", title=TITLES[parameter])
        any_finite = any(np.any(np.isfinite(np.asarray(v)[:, j]))
                         for curve in curves for v in curve["values"].values())
        if not any_finite:
            ax.set_yticks([])
            all_b = np.concatenate([curve["b"] for curve in curves])
            ax.set_xlim(float(all_b.min()) - 250, float(all_b.max()) + 250)
            ax.text(0.5, 0.55, "No finite joint CRLB at any selected b-value\n"
                    "Fisher matrix fails the positive-definite / rank check",
                    ha="center", va="center", transform=ax.transAxes, fontsize=FONT_SIZE-2)
        if XLIM is not None:
            ax.set_xlim(XLIM)
        if YLIMS[parameter] is not None:
            ax.set_ylim(YLIMS[parameter])
        ax.grid(axis="y", alpha=0.18)
        ax.legend(fontsize=FONT_SIZE-4, loc="best")
        note = f"Fixed {len(support)}-measurement support + {SWEEP_REPEATS} identical DW repeats; {AVERAGES:g} averages each"
        noise_note = (f"constant single-measurement sigma={CONSTANT_SIGMA:g}" if NOISE_MODEL == "constant"
                      else f"SNR(TE=0)={SNR_AT_TE_ZERO:g}, T2={T2_MS:g} ms")
        note += f"\nGaussian noise: {noise_note}; b0 reference: n0={N0_EFF:g}"
        ax.text(0.0, 1.01, "MC debias ON" if DEBIAS_MC else "MC debias OFF (uncorrected Fisher)",
                transform=ax.transAxes, fontsize=10, color="#444444")
        if len(rows) == 1:
            note += "\n" + curves[0]["node"]
        if invalid:
            note += "\nGaps: invalid bounds (details in CSV)"
        elif masked:
            note += "\nSome added columns are masked: bound equals supporting protocol (see CSV)"
        fig.text(0.12, 0.015, note, fontsize=11, va="bottom", color="#444444")
        fig.tight_layout(rect=(0, 0.19 if invalid or masked else 0.16, 1, 1))
        for extension in FORMATS:
            path = OUTPUT / f"crlb_{parameter}.{extension}"
            fig.savefig(path, dpi=DPI, facecolor="white")
            print(path)
        plt.close(fig)
    print(f"Parameter order: {PLOT_ORDER}; {invalid} invalid node/column/regime points; "
          f"{masked} masked added columns (see CSV).")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"CRLB plotting failed: {error}") from error
