#!/usr/bin/env python3
"""Phase 2: debiased Fisher matrices, CRLB/kappa maps, and the protocol sweep.

Consumes the Phase-1 `k = 1` derivative fields and the Phase-2 column caches.
Everything it reports is computed under BOTH gradient scenarios, side by side
and never pooled, and across the declared `n0_eff` amplitude sweep.

The `k_io` restriction that a previous revision pre-registered for the minimax
criterion is WITHDRAWN (pre-registration amendment
`2026-09-05-withdraw-kio-restriction`).  The criterion therefore runs over the
full interior `k_io` grid and the expected degeneracy is measured and reported
rather than suppressed: per protocol, which parameter is the argmax, and how
tightly the scores cluster across the sweep.

Phase 2 is the CONDITIONAL layer.  The gradient ceiling, the TE/T2 noise model,
the Rician-validity threshold at the arm's own averaging and the `S/S0` trust
floor all belong here, where each is named in the result and each scenario is
reported separately.  They are applied to, never baked into, the Phase-1
substrate: the run asserts that the substrate covers every column each declared
scenario needs and refuses to proceed if it does not, rather than silently
scoring a smaller acquisition under the requested name.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from madi.fisher_crlb import (PARAMETER_ORDER, artifact_manifest, assert_safe_output,
                              canonical_grid, column_arrays, gradient_feasible_columns,
                              gradient_strength_t_per_m, incomplete_banner, load_preregistration,
                              packed_amplitude_marginal, packed_inverse_diagonal,
                              read_column_domain, require_columns, te_noise_sigma)

AXES = ("rho", "V", "k_io")
# Packed accumulator layout, per (node, column), already weighted by u_c:
#   tissue    (columns, nodes, 6)  packed J J^T with the Var(J) debias already
#                                    subtracted from its diagonal
#   amplitude (columns, nodes, 4)  J*S (three) and S^2 (one)
# Node aggregation.  A node where F is not positive definite has an infinite
# relative CRLB, and EVERY candidate acquisition has such nodes, so the
# pre-registered uniform MEAN over nodes is +inf for every protocol and cannot
# rank anything.  The uniformly weighted order statistic that survives an
# infinite tail is the median: it is finite exactly when a protocol identifies
# more than half the evaluation nodes, and it cannot be gamed by identifying
# fewer of them, which ranking on "mean over the identifiable nodes" could be.
# See the pre-registration amendment 2026-09-05-phase2-node-aggregation.


# ---------------------------------------------------------------------------
# Node table
# ---------------------------------------------------------------------------

def build_node_table(phase1: Path, data) -> dict:
    """Nodes carrying all three k=1 stencils, with entry indices and real steps."""
    samples = {axis: np.load(phase1 / f"samples_{axis}_k1.npy") for axis in AXES}
    keyed = {axis: {tuple(int(v) for v in row): row_index
                    for row_index, row in enumerate(values)}
             for axis, values in samples.items()}
    shared = sorted(set(keyed["rho"]) & set(keyed["V"]) & set(keyed["k_io"]))

    rhos_c, Vs_c, kios_c, _ = canonical_grid()
    nominal = {"rho": np.asarray(data["nominal_rhos"], dtype=float),
               "V": np.asarray(data["nominal_Vs"], dtype=float),
               "kio": np.asarray(data["nominal_kios"], dtype=float)}
    free = np.asarray(data["is_free_water"], dtype=bool)
    realised = {"rho": np.asarray(data["rhos"], dtype=float),
                "V": np.asarray(data["Vs"], dtype=float),
                "kio": np.asarray(data["kios"], dtype=float)}

    # (rho_index, V_index, kio_index) -> entry row, built once by rounding the
    # nominal labels onto the canonical grid rather than by float comparison.
    entry_of: dict[tuple[int, int, int], int] = {}
    cellular = np.flatnonzero(~free)
    for entry in cellular:
        ir = int(np.argmin(np.abs(rhos_c - nominal["rho"][entry])))
        iv = int(np.argmin(np.abs(Vs_c - nominal["V"][entry])))
        ik = int(np.argmin(np.abs(kios_c - nominal["kio"][entry])))
        entry_of[(ir, iv, ik)] = int(entry)

    centre, ends, steps = [], {axis: ([], []) for axis in AXES}, {axis: [] for axis in AXES}
    kept = []
    for node in shared:
        ir, iv, ik = node
        neighbours = {"rho": ((ir - 1, iv, ik), (ir + 1, iv, ik)),
                      "V": ((ir, iv - 1, ik), (ir, iv + 1, ik)),
                      "k_io": ((ir, iv, ik - 1), (ir, iv, ik + 1))}
        if node not in entry_of or any(k not in entry_of for pair in neighbours.values() for k in pair):
            continue
        kept.append(node)
        centre.append(entry_of[node])
        for axis, (minus, plus) in neighbours.items():
            a, b = entry_of[minus], entry_of[plus]
            ends[axis][0].append(a)
            ends[axis][1].append(b)
            if axis == "rho":
                steps[axis].append(np.log(realised["rho"][b]) - np.log(realised["rho"][a]))
            elif axis == "V":
                steps[axis].append(np.log(realised["V"][b]) - np.log(realised["V"][a]))
            else:
                steps[axis].append(realised["kio"][b] - realised["kio"][a])
    nodes = np.asarray(kept, dtype=int)
    table = {"nodes": nodes, "centre": np.asarray(centre, dtype=int),
             "kio": kios_c[nodes[:, 2]],
             "rho": rhos_c[nodes[:, 0]], "V": Vs_c[nodes[:, 1]],
             "phase1_rows": {axis: np.asarray([keyed[axis][tuple(int(v) for v in n)] for n in kept], dtype=int)
                             for axis in AXES}}
    for axis in AXES:
        table[f"minus_{axis}"] = np.asarray(ends[axis][0], dtype=int)
        table[f"plus_{axis}"] = np.asarray(ends[axis][1], dtype=int)
        step = np.asarray(steps[axis], dtype=float)
        if not np.all(step > 0):
            raise RuntimeError(f"non-positive realised {axis} step in the node table")
        table[f"step_{axis}"] = step
    return table


# ---------------------------------------------------------------------------
# Per-column contributions at one timing pair
# ---------------------------------------------------------------------------

def pair_contributions(table: dict, vectors: np.ndarray, variance: np.ndarray,
                       columns: np.ndarray, u_weight: float, n_ensembles: int,
                       trust_floor: float, rician_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column contributions at one timing pair, in `(columns, nodes, ...)` layout.

    Returns `(tissue, amplitude, debias)`.  `tissue` is `(columns, nodes, 6)`,
    the packed `J J^T` with the Monte-Carlo debias ALREADY subtracted from its
    diagonal; `amplitude` is `(columns, nodes, 4)` holding `J S` and `S^2`; and
    `debias` is `(columns, nodes, 3)`, the subtracted diagonal itself, so the
    undebiased Fisher matrix -- the "bias trap" of plan section 2.5 -- can be
    formed from the same call rather than recomputed.
    Both are already multiplied by `u_c = 1/sigma_c^2` and by the per-(node,
    column) hard masks, so a subset's Fisher matrix is a plain sum along the
    column axis.

    The caches are stored column-major precisely so that the 25 columns of one
    timing pair are contiguous here.

    Only the diagonal carries the debias.  Central differences on the three axes
    use four disjoint library entries, so their Monte-Carlo noises are
    independent and the off-diagonals -- the elements that encode the
    degeneracy -- are already unbiased.
    """
    block_v = np.asarray(vectors[columns], dtype=np.float64)     # (columns, entries)
    block_s = np.asarray(variance[columns], dtype=np.float64)
    centre = table["centre"]
    signal = block_v[:, centre]                                   # (columns, nodes)
    jacobian, debias = [], []
    for axis in AXES:
        step = table[f"step_{axis}"][None, :]
        minus, plus = table[f"minus_{axis}"], table[f"plus_{axis}"]
        jacobian.append((block_v[:, plus] - block_v[:, minus]) / step)
        debias.append((block_s[:, plus] + block_s[:, minus]) / (n_ensembles * step ** 2))

    # Rician validity is evaluated at the averaging the ACQUISITION assigns to
    # each column, sigma_1/sqrt(n_c), not at one average.  At one average the
    # median node keeps 2-3 usable columns out of 24 and no three-parameter
    # Fisher matrix exists anywhere on the grid, so the literal
    # `averages_per_column = 1` reading of the pre-registered noise model is not
    # conservative, it is unusable.  Pre-registration amendment
    # 2026-09-05-phase2-averaging-aware-mask.
    keep = (signal >= trust_floor) & (signal >= rician_threshold)
    weight = np.where(keep, u_weight, 0.0)

    tissue = np.empty(signal.shape + (6,), dtype=np.float64)
    tissue[..., 0] = jacobian[0] * jacobian[0] - debias[0]
    tissue[..., 1] = jacobian[0] * jacobian[1]
    tissue[..., 2] = jacobian[0] * jacobian[2]
    tissue[..., 3] = jacobian[1] * jacobian[1] - debias[1]
    tissue[..., 4] = jacobian[1] * jacobian[2]
    tissue[..., 5] = jacobian[2] * jacobian[2] - debias[2]
    tissue *= weight[..., None]

    amplitude = np.empty(signal.shape + (4,), dtype=np.float64)
    for index in range(3):
        amplitude[..., index] = jacobian[index] * signal
    amplitude[..., 3] = signal * signal
    amplitude *= weight[..., None]

    subtracted = np.stack(debias, axis=-1) * weight[..., None]
    return tissue, amplitude, subtracted


def budget_scale(tissue_sum: np.ndarray, amplitude_sum: np.ndarray,
                 scale) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the matched-budget factor `N / n_selected` to a summed acquisition."""
    scale = np.asarray(scale, dtype=float)
    return (tissue_sum * scale[..., None], amplitude_sum[..., 0:3] * scale[..., None],
            amplitude_sum[..., 3] * scale)


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def evaluate(tissue_sum: np.ndarray, amplitude_sum: np.ndarray, kio_ref: np.ndarray,
             scale, s0_prior: float) -> dict:
    """Full criteria for one acquisition (or a batch), in float64.

    `s0_prior` is `lambda`; `inf` selects the known-amplitude (fixed-`S0`) limit.
    """
    packed_tt, F_ta, F_aa = budget_scale(tissue_sum, amplitude_sum, scale)
    packed = packed_tt if np.isinf(s0_prior) else packed_amplitude_marginal(packed_tt, F_ta, F_aa, s0_prior)
    inverse, det, positive = packed_inverse_diagonal(packed)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.stack([np.sqrt(inverse[..., 0]), np.sqrt(inverse[..., 1]),
                             np.sqrt(inverse[..., 2]) / kio_ref], axis=-1)
        diagonal = np.stack([packed[..., 0], packed[..., 3], packed[..., 5]], axis=-1)
        kappa = np.sqrt(np.clip(inverse * diagonal, 0.0, np.inf))
    count = positive.sum(axis=-1)
    pd_fraction = count / positive.shape[-1]
    identifiable = np.where(positive[..., None], relative, np.inf)

    def _mean(values):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(positive[..., None], values, 0.0).sum(axis=-2) / count[..., None]

    per_parameter = np.median(identifiable, axis=-2)
    score = per_parameter.max(axis=-1)
    mean_identifiable = _mean(relative)
    with np.errstate(divide="ignore", invalid="ignore"):
        node_max = np.nanmax(np.where(positive[..., None], relative, -np.inf), axis=-1)
        a_optimality = (np.where(positive, (relative ** 2).sum(axis=-1), 0.0).sum(axis=-1) / count)
        d_term = np.log10(np.abs(det)) - 2.0 * np.log10(kio_ref)
        d_optimality = np.where(positive, d_term, 0.0).sum(axis=-1) / count
    return {
        "score": score,
        "argmax_parameter": np.argmax(per_parameter, axis=-1),
        "per_parameter": per_parameter, "mean_identifiable": mean_identifiable,
        "worst_node": np.nanmax(np.where(positive, node_max, -np.inf), axis=-1),
        "rho_V_only": per_parameter[..., :2].max(axis=-1),
        "a_optimality": a_optimality, "d_optimality": d_optimality,
        "kappa_mean": _mean(kappa),
        "kappa_max": np.nanmax(np.where(positive[..., None], kappa, -np.inf), axis=-2),
        "pd_fraction": pd_fraction,
        "_relative": relative, "_kappa": kappa, "_positive": positive,
    }


def _elementary_symmetric(packed: np.ndarray, order: int) -> np.ndarray:
    """`e_k` of the eigenvalues of a packed symmetric 3x3: trace, minor sum, det.

    D-optimality on a rank-deficient stage.  With fewer than three selected
    columns the determinant is identically zero, so the seed steps maximize the
    largest non-vanishing elementary symmetric polynomial instead.  This is the
    natural rank-k form of the D criterion and needs no arbitrary tie-break.
    """
    a, b, c, d, e, f = (packed[..., i] for i in range(6))
    if order == 1:
        return a + d + f
    if order == 2:
        return (a * d - b * b) + (a * f - c * c) + (d * f - e * e)
    return a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)


def greedy_b_subset(tissue: np.ndarray, kio_ref: np.ndarray,
                    sizes: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Greedy forward b-subset selection, as pre-registered.

    `tissue` is `(columns, nodes, 6)`.  Steps 1-3 use D-optimality at the rank
    the selection has reached -- the largest non-vanishing elementary symmetric
    polynomial of the eigenvalues, which is `trace`, then the principal-minor
    sum, then the determinant.  A determinant is identically zero below three
    columns, so this is the natural rank-k form of the D criterion rather than
    an arbitrary tie-break.  From step 4 the primary criterion takes over, which
    is the first step at which it is defined.

    The selection is scored at unit budget: the fixed-S0 relative CRLB scales as
    `1/sqrt(N/n_selected)`, a factor common to every candidate at a given step,
    so it cannot change which column wins.
    """
    n_columns = tissue.shape[0]
    remaining = list(range(n_columns))
    chosen: list[int] = []
    running = np.zeros(tissue.shape[1:], dtype=np.float64)
    out: dict[int, np.ndarray] = {}
    while len(chosen) < min(max(sizes), n_columns):
        candidates = np.asarray(remaining, dtype=int)
        trial = running[None] + tissue[candidates]
        step = len(chosen) + 1
        if step <= 3:
            with np.errstate(invalid="ignore", divide="ignore"):
                value = np.log(np.clip(_elementary_symmetric(trial, step), 1e-300, None)).mean(axis=-1)
            best = int(np.argmax(np.where(np.isfinite(value), value, -np.inf)))
        else:
            score, _, _ = _score_block(trial, kio_ref, np.ones(len(candidates)))
            if np.any(np.isfinite(score)):
                best = int(np.argmin(np.where(np.isfinite(score), score, np.inf)))
            else:
                # Declared fallback: while no candidate yet identifies more than
                # half the nodes the primary criterion is +inf for all of them
                # and cannot choose.  D-optimality continues to drive selection
                # until the primary criterion can distinguish candidates.
                with np.errstate(invalid="ignore", divide="ignore"):
                    value = np.log(np.clip(_elementary_symmetric(trial, 3), 1e-300, None)).mean(axis=-1)
                best = int(np.argmax(np.where(np.isfinite(value), value, -np.inf)))
        pick = int(candidates[best])
        chosen.append(pick)
        remaining.remove(pick)
        running = running + tissue[pick]
        if len(chosen) in sizes:
            out[len(chosen)] = np.asarray(sorted(chosen), dtype=int)
    return out


# ---------------------------------------------------------------------------
# Search-time scoring
# ---------------------------------------------------------------------------
#
# The search runs under the fixed-S0 bound, which is exactly linear in the
# budget: F = (N/n_selected) * sum_c u_c M_c.  A relative CRLB therefore scales
# as 1/sqrt(N/n_selected), so the whole batch is scored once at unit budget and
# rescaled per protocol.  Every reported number is recomputed afterwards in
# float64 under every declared S0 regime; this path only ranks.

def _score_block(packed: np.ndarray, kio_ref: np.ndarray, scale: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f = (packed[..., i] for i in range(6))
    cof_a = d * f - e * e
    cof_d = a * f - c * c
    cof_f = a * d - b * b
    det = a * cof_a - b * (b * f - c * e) + c * (b * e - c * d)
    # Same strengthened test as madi.fisher_crlb.packed_inverse_diagonal: a node
    # counts as identifiable only when the whole inverse diagonal is positive.
    positive = (a > 0) & (cof_a > 0) & (cof_d > 0) & (cof_f > 0) & (det > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.stack([np.sqrt(cof_a / det), np.sqrt(cof_d / det),
                             np.sqrt(cof_f / det) / kio_ref], axis=-1)
    relative = np.where(positive[..., None], relative, np.inf)
    per = np.median(relative, axis=-2) / np.sqrt(np.asarray(scale, dtype=float))[..., None]
    return per.max(axis=-1), per, positive.mean(axis=-1)


def fast_score(packed: np.ndarray, kio_ref: np.ndarray, scale, chunk: int = 256) -> np.ndarray:
    n = packed.shape[0]
    scale = np.broadcast_to(np.asarray(scale, dtype=float), (n,))
    out = np.empty(n, dtype=float)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        out[start:stop] = _score_block(np.asarray(packed[start:stop], dtype=np.float64),
                                       kio_ref, scale[start:stop])[0]
    return out


def _spread(scores: np.ndarray) -> dict:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return {"finite_arms": 0}
    best = float(finite.min())
    return {"finite_arms": int(finite.size), "non_finite_arms": int(scores.size - finite.size),
            "best": best, "median": float(np.median(finite)), "worst": float(finite.max()),
            "fraction_within_5pct_of_best": float(np.mean(finite <= best * 1.05)),
            "fraction_within_1pct_of_best": float(np.mean(finite <= best * 1.01)),
            "ratio_median_over_best": float(np.median(finite) / best)}


def exhaustive_pairs(packed: np.ndarray, n_selected: np.ndarray, kio_ref: np.ndarray,
                     budget: float, block: int) -> tuple[tuple[int, int], float, int]:
    """Exhaustive m = 2 search over every unordered pair of timing pairs.

    Returns the optimum plus the full score, argmax-parameter and
    identifiable-fraction vectors over the whole sweep, because section 2.1's
    degeneracy diagnosis is a statement about the distribution of scores across
    the sweep, not only about its minimum.
    """
    left, right = np.triu_indices(packed.shape[0], k=1)
    all_scores = np.empty(len(left), dtype=float)
    all_argmax = np.empty(len(left), dtype=np.int8)
    all_fraction = np.empty(len(left), dtype=np.float32)
    best_value, best = np.inf, (0, 1)
    for start in range(0, len(left), block):
        i = left[start:start + block]
        j = right[start:start + block]
        trial = np.asarray(packed[i], dtype=np.float64) + np.asarray(packed[j], dtype=np.float64)
        score, per, fraction = _score_block(trial, kio_ref, budget / (n_selected[i] + n_selected[j]))
        all_scores[start:start + len(i)] = score
        all_argmax[start:start + len(i)] = np.argmax(per, axis=-1)
        all_fraction[start:start + len(i)] = fraction
        position = int(np.argmin(np.where(np.isfinite(score), score, np.inf)))
        if score[position] < best_value:
            best_value = float(score[position])
            best = (int(i[position]), int(j[position]))
    return best, best_value, all_scores, all_argmax, all_fraction


def greedy_add(packed: np.ndarray, n_selected: np.ndarray, kio_ref: np.ndarray,
               budget: float, current: list[int], chunk: int = 256) -> tuple[list[int], float]:
    """Add the one timing pair that most improves the primary criterion."""
    base = np.asarray(packed[current], dtype=np.float64).sum(axis=0)
    base_columns = float(n_selected[current].sum())
    remaining = np.asarray([i for i in range(packed.shape[0]) if i not in set(current)], dtype=int)
    best_value, best = np.inf, int(remaining[0])
    for start in range(0, len(remaining), chunk):
        candidates = remaining[start:start + chunk]
        trial = base[None] + np.asarray(packed[candidates], dtype=np.float64)
        score, per, fraction = _score_block(trial, kio_ref, budget / (base_columns + n_selected[candidates]))
        # While no candidate identifies more than half the nodes the primary
        # criterion is +inf for all of them; rank by identifiable fraction then.
        if not np.any(np.isfinite(score)):
            score = -fraction.astype(float)
        position = int(np.argmin(np.where(np.isfinite(score), score, np.inf)))
        if score[position] < best_value:
            best_value = float(score[position])
            best = int(candidates[position])
    return sorted(current + [best]), best_value


# ---------------------------------------------------------------------------
# Column-major cache view
# ---------------------------------------------------------------------------

def transposed_view(cache_dir: Path, member: str) -> np.ndarray:
    """Return a `(columns, entries)` view of a cached member, building it once.

    A timing pair owns 25 consecutive columns, so the column-major layout makes
    "give me every entry's signal at this pair" a contiguous read.  In the
    row-major layout the same request strides the whole 1.8 GiB file and the
    sweep becomes I/O bound.
    """
    source = cache_dir / f"{member}_selected.npy"
    destination = cache_dir / f"{member}_selected_T.npy"
    if not destination.exists():
        rows = np.load(source, mmap_mode="r")
        out = np.lib.format.open_memmap(destination, mode="w+", dtype=rows.dtype,
                                        shape=(rows.shape[1], rows.shape[0]))
        block = 2048
        for start in range(0, rows.shape[0], block):
            out[:, start:start + block] = rows[start:start + block].T
        out.flush()
        del out, rows
    return np.load(destination, mmap_mode="r")


# ---------------------------------------------------------------------------
# Full float64 report for one selected acquisition
# ---------------------------------------------------------------------------

def protocol_report(pairs, columns, context, averages, maps_path=None) -> dict:
    """Recompute one acquisition in float64 under every declared S0 regime.

    The search ranks in float32 for memory; nothing reported comes from that
    path.  Both the fixed-S0 and the S0-marginalized bound are reported, with
    the gap between them as its own quantity, as section 2.7 of the plan
    requires.
    """
    table = context["table"]
    tissue_total = np.zeros((len(table["nodes"]), 6), dtype=np.float64)
    amplitude_total = np.zeros((len(table["nodes"]), 4), dtype=np.float64)
    debias_total = np.zeros((len(table["nodes"]), 3), dtype=np.float64)
    n_selected = 0
    for pair in pairs:
        local = np.asarray(columns[pair], dtype=int)
        tissue, amplitude, debias = pair_contributions(table, context["vectors"], context["variance"], local,
                                               1.0 / context["sigma_pair"][pair] ** 2,
                                               context["n_ensembles"], context["trust_floor"],
                                               context["rician_min"] * context["sigma_pair"][pair]
                                               / np.sqrt(averages))
        tissue_total += tissue.sum(axis=0)
        amplitude_total += amplitude.sum(axis=0)
        debias_total += debias.sum(axis=0)
        n_selected += len(local)
    scale = context["budget"] / n_selected

    # The b ~ 0 amplitude reference is acquired at the arm's shortest-TE timing
    # pair, which is what an operator would do, and S(0) = 1 exactly in the
    # stored grid, so no extrapolation enters here.  (The BIASED-reference case,
    # a lowest shell at b = 50 rather than 0, is Phase 4 hypothesis H4 and is
    # explicitly not covered by this variance statement.)
    reference_pair = min(pairs, key=lambda pair: context["pair_TE"][pair])
    sigma_reference = float(context["sigma_pair"][reference_pair])

    kio_ref = context["kio_ref"]
    baseline = evaluate(tissue_total, amplitude_total, kio_ref, scale, np.inf)
    # The bias trap of plan section 2.5, made explicit: the same acquisition
    # scored WITHOUT subtracting Var(J_hat) from the Fisher diagonal.  Monte-Carlo
    # noise adds to E[J_hat^2], so the undebiased matrix reports more information
    # than the library actually contains.
    undebiased_packed = tissue_total.copy()
    for index, position in enumerate((0, 3, 5)):
        undebiased_packed[:, position] += debias_total[:, index]
    undebiased = evaluate(undebiased_packed, amplitude_total, kio_ref, scale, np.inf)
    out = {"timing_pairs_ms": [[float(context["pair_deltas"][pair]), float(context["pair_Deltas"][pair])]
                               for pair in pairs],
           "m": len(pairs), "columns_used": int(n_selected),
           "b_values_s_mm2": {str(int(pair)): sorted(float(context["b_of_position"][c])
                                                     for c in columns[pair]) for pair in pairs},
           "averages_per_column": float(averages),
           "budget_scale_N_over_columns": float(scale),
           "amplitude_reference_pair_ms": [float(context["pair_deltas"][reference_pair]),
                                           float(context["pair_Deltas"][reference_pair])],
           "monte_carlo_debias_effect": {
               "identifiable_fraction_debiased": float(baseline["pd_fraction"]),
               "identifiable_fraction_undebiased": float(undebiased["pd_fraction"]),
               "minimax_debiased": float(baseline["score"]),
               "minimax_undebiased": float(undebiased["score"]),
               "mean_debias_over_fisher_diagonal": [
                   float(np.mean(debias_total[:, i] / np.maximum(undebiased_packed[:, position], 1e-300)))
                   for i, position in enumerate((0, 3, 5))],
               "note": ("the debias uses the endpoint-only Var(J_hat); dropping the positive CRN "
                        "covariance overstates it, so the debiased identifiable fraction is a lower "
                        "bound. See debias_calibration for the measured overstatement."),
           },
           "regimes": {}}
    for label, n0 in [("known_amplitude", None)] + [(f"n0_eff_{int(v)}", v) for v in context["n0_values"]]:
        prior = np.inf if n0 is None else float(n0) * 1.0 / sigma_reference ** 2
        result = evaluate(tissue_total, amplitude_total, kio_ref, scale, prior)
        # The gap is a per-node ratio aggregated afterwards.  A ratio of the two
        # medians is not the median of the ratios, and both are +inf at a node
        # neither regime identifies, which would return nan.
        both = baseline["_positive"] & result["_positive"]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_node = result["_relative"][both] / baseline["_relative"][both]
        gap = np.median(per_node, axis=0) if both.any() else np.full(3, np.nan)
        out["regimes"][label] = {
            "s0_prior_precision": None if n0 is None else prior,
            "minimax_score": float(result["score"]),
            "argmax_parameter": PARAMETER_ORDER[int(result["argmax_parameter"])],
            "relative_crlb_median_over_all_nodes_per_parameter": [float(v) for v in result["per_parameter"]],
            "relative_crlb_mean_over_identifiable_nodes_per_parameter": [float(v) for v in result["mean_identifiable"]],
            "worst_node_minimax": float(result["worst_node"]),
            "minimax_rho_V_only_DIAGNOSTIC_NOT_A_CRITERION": float(result["rho_V_only"]),
            "a_optimality_mean_sum_squared_relative_crlb": float(result["a_optimality"]),
            "d_optimality_mean_log10_det": float(result["d_optimality"]),
            "kappa_mean_per_parameter": [float(v) for v in result["kappa_mean"]],
            "kappa_max_per_parameter": [float(v) for v in result["kappa_max"]],
            "positive_definite_fraction": float(result["pd_fraction"]),
            "crlb_ratio_marginal_over_fixed_median_per_node": [float(v) for v in gap],
            "crlb_ratio_nodes_compared": int(both.sum()),
        }
        if maps_path is not None and label in ("known_amplitude", "n0_eff_4"):
            np.save(maps_path.with_name(maps_path.name + f".{label}.relative_crlb.npy"),
                    result["_relative"].astype(np.float32))
            np.save(maps_path.with_name(maps_path.name + f".{label}.kappa.npy"),
                    result["_kappa"].astype(np.float32))
    if maps_path is not None:
        # Where kappa is worst in the (rho, V) plane, per parameter.
        kappa = baseline["_kappa"]
        positive = baseline["_positive"]
        worst = {}
        for index, name in enumerate(PARAMETER_ORDER):
            values = np.where(positive, kappa[..., index], np.nan)
            node = int(np.nanargmax(values))
            worst[name] = {"kappa": float(values[node]),
                           "rho_index": int(table["nodes"][node][0]), "V_index": int(table["nodes"][node][1]),
                           "rho_cells_per_uL": float(table["rho"][node]), "V_pL": float(table["V"][node]),
                           "k_io_s_inv": float(table["kio"][node]),
                           "v_i": float(table["rho"][node] * table["V"][node] * 1e-6),
                           "distribution": _summary(values)}
        out["kappa_worst_node"] = worst
        out["kappa_distribution_per_parameter"] = {
            name: _summary(np.where(positive, kappa[..., index], np.nan))
            for index, name in enumerate(PARAMETER_ORDER)}
        out["relative_crlb_distribution_per_parameter"] = {
            name: _summary(np.where(positive, baseline["_relative"][..., index], np.nan))
            for index, name in enumerate(PARAMETER_ORDER)}
        out["kappa_above_threshold_fraction"] = {
            name: float(np.nanmean(np.where(positive, kappa[..., index], np.nan) > context["kappa_threshold"]))
            for index, name in enumerate(PARAMETER_ORDER)}
        # The k_io region split that the withdrawn restriction was about.
        coarse = table["kio"] > 30.0
        out["k_io_region_split"] = {}
        for region, mask in (("k_io_le_30", ~coarse), ("k_io_gt_30", coarse)):
            keep = mask & positive
            out["k_io_region_split"][region] = {
                "nodes": int(mask.sum()), "positive_definite": int(keep.sum()),
                "relative_crlb_median_per_parameter": [
                    float(np.nanmedian(baseline["_relative"][keep, index])) if keep.any() else None
                    for index in range(3)],
            }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _summary(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {"count": int(finite.size), "min": float(finite.min()),
            "q05": float(np.quantile(finite, 0.05)), "median": float(np.median(finite)),
            "q95": float(np.quantile(finite, 0.95)), "max": float(finite.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snr", type=float, default=50.0)
    parser.add_argument("--budget-images", type=float, default=128.0,
                        help="matched total diffusion-weighted image count N")
    parser.add_argument("--node-stride", type=int, default=1, help="smoke only; subsamples nodes")
    parser.add_argument("--m2-block", type=int, default=96)
    parser.add_argument("--max-m", type=int, default=4)
    parser.add_argument("--skip-m2", action="store_true", help="smoke only")
    args = parser.parse_args()

    started = time.time()
    prereg = load_preregistration()
    T2 = float(prereg["noise_model"]["T2_ms"])
    t_epi = float(prereg["noise_model"]["t_epi_ms"])
    trust_floor = float(prereg["trust_floor"])
    rician_min = float(prereg["rician_magnitude_snr_min"])
    kio_floor = float(prereg["protocol_sweep"]["criteria"]["relative_scale_for_k_io"]["k_io_floor_s^-1"])
    sizes = tuple(int(v) for v in prereg["protocol_sweep"]["b_subset_rule"]["sizes"])
    n0_values = [float(v) for v in prereg["amplitude_model"]["n0_eff_sweep"]["values"]]
    scenarios = {name: float(limit) for name, limit in prereg["gradient_limits_T_per_m"].items()}

    manifest_json = json.loads((args.phase1 / "phase1_manifest.json").read_text(encoding="utf-8"))
    domain = read_column_domain(manifest_json)
    selected = np.asarray(domain.column_indices, dtype=int)
    diagnostic_full = np.asarray(manifest_json["column_selection"]["diagnostic_full_column_indices"], dtype=int)
    print(domain.banner(), flush=True)

    with np.load(args.artifact, allow_pickle=False) as data:
        manifest = artifact_manifest(data)
        assert_safe_output(args.output_dir, manifest)
        pair_deltas = np.asarray(data["pair_deltas"], dtype=float)
        pair_Deltas = np.asarray(data["pair_Deltas"], dtype=float)
        b_values = np.asarray(data["b_values"], dtype=float)
        n_b = int(data["n_b"])
        build = json.loads(str(data["build_metadata_json"]))
        table = build_node_table(args.phase1, data)
    n_ensembles = int(np.asarray(np.load(args.cache_dir / "ensemble_means_subset.npy", mmap_mode="r").shape)[1])

    if args.node_stride > 1:
        keep = np.arange(0, len(table["nodes"]), args.node_stride)
        for key, value in list(table.items()):
            if key == "phase1_rows":
                table[key] = {axis: rows[keep] for axis, rows in value.items()}
            else:
                table[key] = value[keep]
    n_nodes = len(table["nodes"])
    kio_ref = np.maximum(table["kio"], kio_floor)

    delta_col, Delta_col, b_col = column_arrays(pair_deltas, pair_Deltas, b_values)
    gradient = gradient_strength_t_per_m(delta_col, Delta_col, b_col)
    sigma_pair = te_noise_sigma(pair_deltas, pair_Deltas, sigma0=1.0 / args.snr, T2_ms=T2, t_epi_ms=t_epi)
    # Position of every stored column inside the Phase-1 selection, -1 if absent.
    position_of = np.full(len(b_col), -1, dtype=int)
    position_of[selected] = np.arange(len(selected))

    vectors = transposed_view(args.cache_dir, "vectors")
    variance = transposed_view(args.cache_dir, "signal_variance")

    report = {"schema": "madi-fisher-phase2-v1", "artifact": str(args.artifact),
              "phase1": str(args.phase1), "banner": incomplete_banner(manifest),
              **manifest.as_dict(), "parameter_order": list(PARAMETER_ORDER),
              "snr_at_b0": args.snr, "T2_ms": T2, "t_epi_ms": t_epi,
              "budget_images_N": args.budget_images, "n_ensembles": n_ensembles,
              "node_stride": args.node_stride,
              "column_domain": domain.as_dict(), "column_domain_banner": domain.banner(),
              "walkers_per_ensemble": build.get("walkers_per_ensemble"),
              "evaluation_nodes": {}, "derivative_cross_check": {},
              "debias_calibration": {}, "scenarios": {}}

    # --- 3.1 node accounting -------------------------------------------------
    _, _, kios_c, retained = canonical_grid()
    rv = sorted(set((int(a), int(b)) for a, b, _ in table["nodes"]))
    report["evaluation_nodes"] = {
        "definition": "every node carrying all three k=1 central stencils; no k_io ceiling (restriction withdrawn)",
        "count": int(n_nodes),
        "rho_V_pairs_retained": len(rv), "rho_V_pairs_in_mask": len(retained),
        "rho_V_pairs_lost_to_incomplete_stencils": len(retained) - len(rv),
        "k_io_indices": sorted(set(int(k) for _, _, k in table["nodes"])),
        "k_io_range_s_inv": [float(table["kio"].min()), float(table["kio"].max())],
        "k_io_above_30_node_fraction": float(np.mean(table["kio"] > 30.0)),
        "lost_rho_V_pairs": [[int(a), int(b)] for a, b in sorted(retained - set(rv))],
        "why_lost": ("a node needs rho+-1, V+-1 and k_io+-1 all present. The lost pairs are the "
                     "mask-band edge, where a rho or V neighbour falls outside 0.40 <= rho*V*1e-6 <= 0.99, "
                     "plus the two k_io end nodes of every retained pair."),
    }

    # --- derivative cross-check against the stored Phase-1 fields ------------
    # Phase 2 recomputes J from the cached signal rather than reading the 7 GiB
    # Phase-1 fields.  That is the pattern the section-1.2 audit is about, so
    # the two are compared here rather than assumed equal.
    check_columns = np.linspace(0, len(selected) - 1, 64).astype(int)
    check_rows = np.linspace(0, n_nodes - 1, 32).astype(int)
    for axis in AXES:
        stored = np.load(args.phase1 / f"J_{axis}_k1.npy", mmap_mode="r")
        rows = table["phase1_rows"][axis][check_rows]
        stored_block = np.asarray(stored[:, check_columns], dtype=np.float64)[rows]
        step = table[f"step_{axis}"][check_rows][:, None]
        recomputed = (vectors[np.ix_(check_columns, table[f"plus_{axis}"][check_rows])].T.astype(np.float64)
                      - vectors[np.ix_(check_columns, table[f"minus_{axis}"][check_rows])].T.astype(np.float64)) / step
        scale = np.maximum(np.abs(stored_block), 1e-12)
        report["derivative_cross_check"][axis] = {
            "cells": int(stored_block.size),
            "max_abs_difference": float(np.max(np.abs(stored_block - recomputed))),
            "max_relative_difference": float(np.max(np.abs(stored_block - recomputed) / scale)),
            "note": "Phase-1 accumulated in float32; agreement is expected at float32 rounding, not exactly",
        }
        del stored

    # --- debias calibration on the 8 diagnostic timing pairs -----------------
    ensembles = np.load(args.cache_dir / "ensemble_means_subset.npy", mmap_mode="r")
    diagnostic_positions = position_of[diagnostic_full]
    if np.any(diagnostic_positions < 0):
        raise RuntimeError("a diagnostic column is missing from the Phase-1 selection")
    calibration = {}
    for axis in AXES:
        step = table[f"step_{axis}"]
        minus, plus = table[f"minus_{axis}"], table[f"plus_{axis}"]
        endpoint = ((variance[np.ix_(diagnostic_positions, plus)].T.astype(np.float64)
                     + variance[np.ix_(diagnostic_positions, minus)].T.astype(np.float64))
                    / (n_ensembles * step[:, None] ** 2))
        exact = np.empty_like(endpoint)
        block = 1024
        for start in range(0, n_nodes, block):
            stop = min(start + block, n_nodes)
            left = ensembles[minus[start:stop]].astype(np.float64)
            right = ensembles[plus[start:stop]].astype(np.float64)
            covariance = np.sum((left - left.mean(axis=1, keepdims=True)) *
                                (right - right.mean(axis=1, keepdims=True)), axis=1) / (n_ensembles - 1)
            exact[start:stop] = np.maximum(
                endpoint[start:stop] * (n_ensembles * step[start:stop, None] ** 2)
                - 2.0 * covariance, 0.0) / (n_ensembles * step[start:stop, None] ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(exact > 0, endpoint / exact, np.nan)
        calibration[axis] = {
            "endpoint_only_over_exact": _summary(ratio),
            "mean_exact": float(np.mean(exact)), "mean_endpoint_only": float(np.mean(endpoint)),
            "note": ("endpoint-only drops the positive CRN covariance, so it OVERSTATES Var(J_hat); "
                     "the debiased Fisher is therefore a lower bound and the CRLB conservative"),
        }
    report["debias_calibration"] = {
        "columns": "the 200 diagnostic columns (8 timing pairs), the only ones carrying ensemble_means_subset",
        "axes": calibration,
    }
    del ensembles
    print(f"[{time.time()-started:7.1f}s] nodes={n_nodes} cross-check and debias calibration done", flush=True)

    b_of_position = b_col[selected]
    context = {"table": table, "vectors": vectors, "variance": variance, "sigma_pair": sigma_pair,
               "n_ensembles": n_ensembles, "trust_floor": trust_floor, "rician_min": rician_min,
               "kio_ref": kio_ref, "budget": args.budget_images, "n0_values": n0_values,
               "pair_deltas": pair_deltas, "pair_Deltas": pair_Deltas,
               "pair_TE": pair_deltas + pair_Deltas + t_epi, "b_of_position": b_of_position,
               "kappa_threshold": float(prereg["kappa_unidentified_threshold"])}

    # -----------------------------------------------------------------------
    # The sweep, under each gradient scenario separately and never pooled
    # -----------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario, g_max in sorted(scenarios.items()):
        # The gradient ceiling is a CONDITIONAL hardware statement and is applied
        # here, at evaluation, where the scenario is named in the result.  The
        # substrate must already carry every column it admits: silently dropping
        # the ones a restricted cache happens to lack would score a smaller
        # acquisition under this scenario's name.  See docs/fisher_domain_audit.md.
        feasible_full = gradient_feasible_columns(delta_col, Delta_col, b_col, g_max)
        require_columns(domain, feasible_full,
                        f"Phase-2 {scenario} gradient scenario (G <= {g_max} T/m)",
                        delta_col, Delta_col, b_col)
        feasible_column = np.zeros(len(b_col), dtype=bool)
        feasible_column[feasible_full] = True
        # b = 0 is excluded from every diffusion-weighted subset: J is identically
        # zero there (S(0) = 1 for every entry), so it carries no tissue
        # information and enters only as the amplitude reference, which the
        # n0_eff sweep already models.  Including it would double-count it.
        candidates: dict[int, np.ndarray] = {}
        for pair in range(len(pair_deltas)):
            full = pair * n_b + np.arange(1, n_b)
            usable = full[feasible_column[full]]
            if len(usable) >= min(sizes):
                candidates[pair] = position_of[usable]
        scenario_report = {
            "gradient_limit_T_per_m": g_max,
            "feasible_columns": int(np.count_nonzero(feasible_column)),
            "gradient_mask_role": ("conditional hardware scenario applied at evaluation. "
                                   "Every column it admits was verified present in the "
                                   + ("unrestricted substrate."
                                      if domain.is_complete else
                                      f"substrate, whose declared domain is RESTRICTED "
                                      f"(basis={domain.basis!r}, {len(domain.column_indices)} of "
                                      f"{domain.full_stored_columns} stored columns).")),
            "candidate_pairs_by_subset_size": {str(size): int(sum(len(v) >= size for v in candidates.values()))
                                               for size in sizes},
            "declared_sweep_size": {},
        }
        # Declared sweep size, stated before it is run (section 3.3).
        for size in sizes:
            n_ok = sum(len(v) >= size for v in candidates.values())
            scenario_report["declared_sweep_size"][str(size)] = {
                "m1_arms": n_ok, "m2_arms": n_ok * (n_ok - 1) // 2,
                "m3_arms_if_exhaustive": n_ok * (n_ok - 1) * (n_ok - 2) // 6,
                "m4_arms_if_exhaustive": n_ok * (n_ok - 1) * (n_ok - 2) * (n_ok - 3) // 24,
                "b_subsets_if_exhaustive_per_pair": int(np.round(np.exp(
                    sum(np.log(24 - i) - np.log(i + 1) for i in range(size))))),
            }

        pair_index = sorted(candidates)
        scenario_report["arms"] = {}
        for size in sizes:
            # The Rician mask, and therefore the b-subset and the pair sums,
            # depend on the per-column averaging N/(m*size), which is known
            # before the selection because m and size are declared.  Everything
            # is therefore rebuilt at each arm cardinality rather than selected
            # once and reused under a mask that does not apply to it.
            by_m: dict[int, tuple] = {}
            for m in range(1, args.max_m + 1):
                averages = args.budget_images / (m * size)
                packed = np.zeros((len(pair_index), n_nodes, 6), dtype=np.float32)
                available = np.zeros(len(pair_index), dtype=bool)
                chosen: dict[int, list[int]] = {}
                for slot, pair in enumerate(pair_index):
                    columns = candidates[pair]
                    if len(columns) < size:
                        continue
                    tissue, _, _ = pair_contributions(table, vectors, variance, columns,
                                                   1.0 / sigma_pair[pair] ** 2, n_ensembles, trust_floor,
                                                   rician_min * sigma_pair[pair] / np.sqrt(averages))
                    local = greedy_b_subset(tissue, kio_ref, (size,))[size]
                    packed[slot] = tissue[local].sum(axis=0).astype(np.float32)
                    available[slot] = True
                    chosen[pair] = [int(columns[i]) for i in local]
                by_m[m] = (packed, available, chosen, averages)
                print(f"[{time.time()-started:7.1f}s] {scenario} size={size} m={m}: "
                      f"{int(available.sum())} candidate pairs", flush=True)

            arms: dict[str, dict] = {}
            packed1, available1, chosen1, averages1 = by_m[1]
            slots1 = np.flatnonzero(available1)
            score1, per1, fraction1 = _score_block(
                np.asarray(packed1[slots1], dtype=np.float64), kio_ref,
                np.full(len(slots1), args.budget_images / size))
            if np.any(np.isfinite(score1)):
                ranking, ranked_by = np.where(np.isfinite(score1), score1, np.inf), "minimax relative CRLB"
            else:
                # Declared fallback: when NO single-Delta arm identifies more than
                # half the evaluation nodes the primary criterion is +inf for every
                # one of them and cannot rank.  Reporting an arbitrary argmin would
                # be worse than saying so, so the arms are ranked by identifiable
                # node fraction and the failure is stated in the record.
                ranking, ranked_by = -fraction1.astype(float), "identifiable node fraction (no finite minimax)"
            order1 = np.argsort(ranking)
            best1_slot = int(slots1[order1[0]])
            arms["m1"] = {"n_arms": int(len(slots1)), "search": "exhaustive",
                          "best_pairs": [int(pair_index[best1_slot])],
                          "ranked_by": ranked_by,
                          "arms_with_a_finite_minimax": int(np.count_nonzero(np.isfinite(score1))),
                          "score_spread": _spread(score1),
                          "identifiable_fraction_spread": _summary(fraction1),
                          "argmax_parameter_counts": {PARAMETER_ORDER[i]: int(np.count_nonzero(
                              np.argmax(per1, axis=-1) == i)) for i in range(3)},
                          "top_10_pairs_ms": [[float(pair_deltas[pair_index[slots1[i]]]),
                                               float(pair_Deltas[pair_index[slots1[i]]]),
                                               float(score1[i]), float(fraction1[i])] for i in order1[:10]]}

            best2_pairs = None
            if not args.skip_m2 and args.max_m >= 2:
                packed2, available2, chosen2, averages2 = by_m[2]
                slots2 = np.flatnonzero(available2)
                if len(slots2) >= 2:
                    counts = np.full(len(slots2), float(size))
                    best2, best2_score, scores2, argmax2, fraction2 = exhaustive_pairs(
                        packed2[slots2], counts, kio_ref, args.budget_images, args.m2_block)
                    evaluated = len(scores2)
                    best2_pairs = [int(pair_index[slots2[i]]) for i in best2]
                    seed = int(np.flatnonzero(slots2 == best1_slot)[0]) if best1_slot in slots2 else 0
                    greedy2, greedy2_score = greedy_add(packed2[slots2], counts, kio_ref,
                                                        args.budget_images, [seed])
                    arms["m2"] = {"n_arms": int(evaluated), "search": "exhaustive",
                                  "best_pairs": best2_pairs, "best_score": float(best2_score),
                                  "arms_with_a_finite_minimax": int(np.count_nonzero(np.isfinite(scores2))),
                                  "score_spread": _spread(scores2),
                                  "identifiable_fraction_spread": _summary(fraction2),
                                  "argmax_parameter_counts": {PARAMETER_ORDER[i]: int(np.count_nonzero(
                                      argmax2[np.isfinite(scores2)] == i)) for i in range(3)},
                                  "greedy_from_best_m1": {
                                      "pairs_ms": [[float(pair_deltas[pair_index[slots2[i]]]),
                                                    float(pair_Deltas[pair_index[slots2[i]]])] for i in greedy2],
                                      "score": float(greedy2_score)},
                                  "greedy_optimality_gap": float(greedy2_score / best2_score - 1.0)}
                    print(f"[{time.time()-started:7.1f}s] {scenario} size={size}: m2 exhaustive "
                          f"{evaluated} arms, best {best2_score:.5g}", flush=True)

            current_pairs = list(best2_pairs) if best2_pairs is not None else [int(pair_index[best1_slot])]
            for m in range(len(current_pairs) + 1, args.max_m + 1):
                packed_m, available_m, chosen_m, averages_m = by_m[m]
                slots_m = np.flatnonzero(available_m)
                lookup = {int(pair_index[slot]): position for position, slot in enumerate(slots_m)}
                seeds = [lookup[pair] for pair in current_pairs if pair in lookup]
                if len(seeds) != len(current_pairs) or len(slots_m) <= len(seeds):
                    break
                counts = np.full(len(slots_m), float(size))
                picked, value = greedy_add(packed_m[slots_m], counts, kio_ref, args.budget_images, seeds)
                current_pairs = [int(pair_index[slots_m[i]]) for i in picked]
                arms[f"m{m}"] = {"n_arms": int(len(slots_m) - m + 1), "search": "greedy forward",
                                 "best_pairs": list(current_pairs), "best_score": float(value)}

            # Everything reported from here is recomputed in float64 under every
            # declared S0 regime, from the same declared column choices.
            for label in list(arms):
                m = int(label[1:])
                _, _, chosen_m, averages_m = by_m[m]
                maps = (args.output_dir / f"maps_{scenario}_size{size}_{label}") if size == sizes[0] else None
                arms[label]["report"] = protocol_report(arms[label]["best_pairs"], chosen_m,
                                                        context, averages_m, maps_path=maps)
                arms[label]["timing_pairs_ms"] = arms[label]["report"]["timing_pairs_ms"]

            references = {}
            for name, timing in (("madi_ii_20_50", (20.0, 50.0)), ("madi_iii_7_25", (7.0, 25.0))):
                match = np.flatnonzero((pair_deltas == timing[0]) & (pair_Deltas == timing[1]))
                _, _, chosen_1, averages_1 = by_m[1]
                if len(match) and int(match[0]) in chosen_1:
                    references[name] = protocol_report([int(match[0])], chosen_1, context, averages_1)
                else:
                    references[name] = {"unavailable": "fewer than the declared b-subset size survives the "
                                                       f"{scenario} gradient limit at this timing"}
            arms["reference_protocols_context_only"] = references
            scenario_report["arms"][str(size)] = arms
            del by_m
        report["scenarios"][scenario] = scenario_report

    np.save(args.output_dir / "evaluation_nodes.npy", table["nodes"].astype(np.int16))
    np.save(args.output_dir / "evaluation_node_labels.npy",
            np.stack([table["rho"], table["V"], table["kio"]], axis=1).astype(np.float64))
    report["runtime_seconds"] = round(time.time() - started, 1)
    (args.output_dir / "phase2_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(report["banner"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
