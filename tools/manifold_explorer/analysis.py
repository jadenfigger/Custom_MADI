"""UI-independent slice analysis, inspection, sampling and data export."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass

import numpy as np

from . import axes, slice as slicing
from .session import MAX_COLUMNS, parse_signal

MAX_BLOCK_BYTES = 128 * 1024 * 1024


@dataclass
class Analysis:
    result: slicing.SliceResult
    measured: list[int]
    block: np.ndarray
    variance: np.ndarray | None
    display: np.ndarray
    valid: np.ndarray
    names: list[str]
    reference_row: int | None


def compute(data, measured, groups, method, reference_row, reference_mode, pasted,
            sigma, reduced_threshold, s0_mode, options, vi_min, vi_max,
            rho_max=None, checkpoint=lambda: None):
    if not measured or len(measured) > MAX_COLUMNS:
        raise ValueError(f"Select between 1 and {MAX_COLUMNS} measured columns.")
    if not 2 <= len(groups) <= 3:
        raise ValueError("Pick two or three display axes.")
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(reduced_threshold) or reduced_threshold < 0:
        raise ValueError("Noise must be positive and threshold nonnegative.")
    for columns in [measured] + [g.columns for g in groups]:
        if len(columns) * data.labels.n_entries * 8 > MAX_BLOCK_BYTES:
            raise ValueError("This selection exceeds the 128 MiB block limit. Select fewer columns per group.")
    eligible = data.eligible(vi_min, vi_max, "free_water" in options, rho_max)
    if not len(eligible):
        raise ValueError("No entries pass the candidate filter. Widen the v_i band or maximum rho.")
    checkpoint()
    block = data.block(measured)[eligible]
    variance = data.variance(measured)[eligible] if "variance" in options else None
    checkpoint()
    if reference_mode == "paste":
        reference = parse_signal(pasted, len(measured))
        marked = None
    else:
        where = np.flatnonzero(eligible == reference_row) if reference_row is not None else []
        if not len(where):
            raise ValueError("Reference is outside the candidate filter. Snap to an eligible entry or widen the filter.")
        reference = block[int(where[0])]
        marked = int(reference_row)
    values, names = [], []
    valid = np.ones(len(eligible), dtype=bool)
    for group in groups:
        collapsed, group_valid = axes.collapse(data.block(group.columns)[eligible], group, method)
        values.append(collapsed)
        names.append(group.label(method))
        valid &= group_valid
        checkpoint()
    display = np.column_stack(values)
    result = slicing.slice_manifold(block, reference, data.labels, eligible,
        threshold=reduced_threshold * len(measured), sigma_measurement=sigma,
        variance_block=variance, s0_mode=s0_mode, display_values=display, display_names=names)
    result.measured_columns = np.asarray(measured, dtype=np.int64)
    return Analysis(result, measured, block, variance, display, valid, names, marked)


def sample_indices(rows, survivors, budget, reference_row=None):
    """Deterministic stratified plot sample; exact analysis is never sampled."""
    rows = np.asarray(rows)
    if len(rows) <= budget:
        return np.arange(len(rows))
    budget = max(1, int(budget))
    mandatory = np.flatnonzero(rows == reference_row) if reference_row is not None else np.array([], dtype=int)
    selected = set(mandatory.tolist())
    remaining = budget - len(selected)
    foreground = np.flatnonzero(survivors)
    background = np.flatnonzero(~np.asarray(survivors, dtype=bool))
    foreground = np.asarray([i for i in foreground if i not in selected], dtype=int)
    background = np.asarray([i for i in background if i not in selected], dtype=int)
    n_fore = min(len(foreground), max(remaining // 2, remaining - len(background)))
    n_back = min(len(background), remaining - n_fore)
    for indices, count in ((foreground, n_fore), (background, n_back)):
        if count:
            selected.update(indices[np.linspace(0, len(indices) - 1, count, dtype=int)].tolist())
    return np.asarray(sorted(selected), dtype=int)


def inspect(analysis, row):
    result = analysis.result
    found = np.flatnonzero(result.eligible_rows == row)
    if not len(found):
        raise ValueError("The inspected row is outside the current candidate filter.")
    index = int(found[0])
    signal = analysis.block[index]
    reference = result.reference_signal
    variance = None if analysis.variance is None else analysis.variance[index]
    sigma = np.broadcast_to(slicing.column_sigma(result.sigma_measurement, variance), signal.shape)
    amplitude = 1.0
    if result.s0_mode == "free":
        s, r = signal / sigma, reference / sigma
        amplitude = float(np.dot(s, r) / max(np.dot(s, s), 1e-300))
    residual = signal * amplitude - reference
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = residual / (sigma * amplitude) if amplitude > 0 else np.full_like(signal, np.nan)
    return {"row": int(row), "survives": bool(result.survivors[index]),
            "chi2": float(result.chi2[index]), "reduced_chi2": float(result.chi2[index] / len(signal)),
            "rmse": float(np.sqrt(np.mean((signal - reference) ** 2))),
            "amplitude": amplitude, "signal": signal, "fitted": amplitude * signal,
            "residual": residual, "standardized": standardized, "sigma": sigma,
            "display": analysis.display[index], "display_valid": bool(analysis.valid[index])}


def rank_acquisitions(data, analysis, columns, checkpoint=lambda: None):
    """Rank raw prediction disagreement / noise among equally weighted survivors.

    A descriptive acquisition heuristic, not posterior information gain or a
    free-amplitude design bound. Only the picker pair is scanned (usually 25
    columns), and already measured columns are excluded.
    """
    rows = analysis.result.survivor_rows
    if len(rows) < 2:
        return []
    columns = [int(c) for c in columns if c not in analysis.measured]
    records = []
    for start in range(0, len(columns), 32):
        checkpoint()
        batch = columns[start:start + 32]
        block = data.block(batch)[rows]
        noise2 = np.full(len(batch), analysis.result.sigma_measurement ** 2)
        if analysis.variance is not None:
            noise2 += np.mean(data.variance(batch)[rows], axis=0) / slicing.N_ENSEMBLES
        spread = np.std(block, axis=0)
        q05, median, q95 = np.quantile(block, [0.05, 0.5, 0.95], axis=0)
        for j, column in enumerate(batch):
            records.append({"column_id": column, "acquisition": data.labels.column_label(column),
                            "spread_noise": float(spread[j] / np.sqrt(noise2[j])),
                            "std": float(spread[j]), "q05": float(q05[j]),
                            "median": float(median[j]), "q95": float(q95[j])})
    return sorted(records, key=lambda item: (-item["spread_noise"], item["column_id"]))


def export_csv(data, analysis, survivors_only=True):
    result, labels = analysis.result, data.labels
    indices = np.flatnonzero(result.survivors) if survivors_only else np.arange(len(result.eligible_rows))
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["row", "rho_nominal", "V_nominal", "k_io", "rho_realised", "V_realised",
                     "v_i", "free_water", "chi2", "reduced_chi2", "survives", "display_valid"] + analysis.names)
    for i in indices:
        row = result.eligible_rows[i]
        writer.writerow([row, labels.nominal_rhos[row], labels.nominal_Vs[row], labels.kios[row],
                         labels.rhos[row], labels.Vs[row], labels.vis[row], int(labels.is_free_water[row]),
                         result.chi2[i], result.chi2[i] / len(analysis.measured),
                         int(result.survivors[i]), int(analysis.valid[i]), *analysis.display[i]])
    return output.getvalue()


def export_reference_csv(data, analysis):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["column_id", "delta", "Delta", "b", "signal"])
    for column, signal in zip(analysis.measured, analysis.result.reference_signal):
        writer.writerow([column, *data.labels.column_triple(column), signal])
    return output.getvalue()


def export_npz(data, analysis, session, fisher_report=None, ranking=None):
    """Numeric arrays plus a Unicode JSON manifest; load with allow_pickle=False."""
    result = analysis.result
    rows = result.eligible_rows
    output = io.BytesIO()
    arrays = {"row": rows, "measured_columns": result.measured_columns,
              "acquisition_coordinates": np.asarray([data.labels.column_triple(c) for c in analysis.measured]),
              "signal": analysis.block, "reference": result.reference_signal,
              "chi2": result.chi2, "survives": result.survivors,
              "display": analysis.display, "display_valid": analysis.valid,
              "display_names": np.asarray(analysis.names),
              "session_json": np.asarray(json.dumps(session, allow_nan=False))}
    for name in ("rhos", "Vs", "kios", "nominal_rhos", "nominal_Vs", "vis", "is_free_water"):
        arrays[name] = getattr(data.labels, name)[rows]
    if analysis.variance is not None:
        arrays["signal_variance"] = analysis.variance
    if fisher_report is not None:
        arrays["fisher_matrix"] = fisher_report["F"]
        arrays["fisher_crlb"] = fisher_report["crlb"]
        arrays["fisher_eigenvalues"] = fisher_report["eigenvalues"]
    if ranking:
        arrays["ranked_columns"] = np.asarray([r["column_id"] for r in ranking])
        arrays["ranked_spread_noise"] = np.asarray([r["spread_noise"] for r in ranking])
    np.savez_compressed(output, **arrays)
    return output.getvalue()
