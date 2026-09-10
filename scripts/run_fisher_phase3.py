#!/usr/bin/env python3
"""Phase 3: the degeneracy map, and the constant-`v_i` hyperbola hypothesis.

Plan section 6.  Objective: characterize the degeneracy and test the hypothesis
that it runs along constant-`v_i` hyperbolae.  Question answered: **the geometry
of the model's own degeneracy**.

  3.1  Eigendecompose the non-dimensionalized Fisher matrix `D F D` at every
       interior node; report spectra and condition numbers.
  3.2  Report the angle between the sloppy eigenvector and the constant-`v_i`
       direction `(1, -1)/sqrt(2)`.
  3.3  Emit the per-node maps the structural figure is drawn from
       (`scripts/plot_fisher_phase3.py`).

Plan section 6 is explicit that a Fisher matrix is `J^T Sigma^-1 J`, so a
spectrum is meaningful only with its column set and its column weighting named,
and that the sloppy-direction ANGLE is the weighting-robust quantity.  This
script therefore evaluates a DECLARED LIST OF DOMAINS in one streaming pass and
reports each with its declaration attached:

  model layer, unconditional (the reusable substrate of plan section 2.8)
    full_stored_domain                    every stored diffusion-weighted column
    full_stored_domain_uniform_weighting  the same columns, Sigma = I
    full_stored_domain_trust_floor        the same columns, S/S0 >= 0.015 applied

  conditional layer, scenario-wide (a declared gradient ceiling, nothing else)
    scenario_wide_research                G <= 300 mT/m
    scenario_wide_clinical                G <=  80 mT/m

  conditional layer, a declared acquisition (the executed Phase-2 optima)
    phase2_optimum_{scenario}_size8_{m1,m2}

The first, fourth and fifth differ ONLY by the gradient ceiling, so their
contrast isolates the hardware condition; the second and third isolate the
weighting and the trust-floor assumption on the same columns.  Which of these is
the reported result was a scientific choice for the project owner, recorded in
`docs/fisher_phase3.md`; this tool computes all of them and nominates none.

Nothing here regenerates a derivative field: the substrate already spans all
31,125 stored columns, so every domain above is a re-evaluation of it.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from madi.fisher_crlb import (CONSTANT_VI_DIRECTION, PARAMETER_ORDER, VI_CHANGING_DIRECTION,
                              artifact_manifest, assert_safe_output, column_arrays,
                              degeneracy_geometry, gradient_feasible_columns, incomplete_banner,
                              amplitude_prior_precision, load_preregistration,
                              packed_amplitude_marginal,
                              packed_inverse_diagonal, read_column_domain, require_columns,
                              te_noise_sigma)
# The node table, the per-column contribution arithmetic and the distribution
# summary are Phase 2's, imported rather than reimplemented.  A script that
# re-derives arithmetic already committed elsewhere is the exact pattern the
# `Var(J_hat)` defect came from; see fisher_phase2.md section 1.2.
from scripts.run_fisher_phase2 import (_summary, build_node_table, pair_contributions,
                                       transposed_view)

MODEL_LAYER = "model_substrate_unconditional"
SCENARIO_LAYER = "conditional_declared_gradient_scenario"
ACQUISITION_LAYER = "conditional_declared_acquisition"

ANGLE_THRESHOLDS_DEG = (5.0, 10.0, 20.0, 30.0)
# An isotropically random direction in R^3 has median acute angle 60 deg to any
# fixed direction; in R^2 the median is 45 deg.  Those are the null references
# the concentration of the measured angles is read against.
NULL_MEDIAN_ANGLE_3D_DEG = 60.0
NULL_MEDIAN_ANGLE_2D_DEG = 45.0


# ---------------------------------------------------------------------------
# Declared domains
# ---------------------------------------------------------------------------

@dataclass
class Domain:
    """One declared column set plus the weighting and masks evaluated on it."""

    name: str
    layer: str
    declaration: dict
    columns: dict[int, np.ndarray]          # timing pair -> cache positions
    uniform_weighting: bool = False
    trust_floor: float = -np.inf
    rician_averages: float | None = None    # None: not applied, annotated instead
    budget_scale: float = 1.0
    tissue: np.ndarray = field(default=None, repr=False)
    amplitude: np.ndarray = field(default=None, repr=False)
    debias: np.ndarray = field(default=None, repr=False)
    cells: np.ndarray = field(default=None, repr=False)
    trace: np.ndarray = field(default=None, repr=False)   # (nodes, 4) annotation traces

    def allocate(self, n_nodes: int) -> None:
        self.tissue = np.zeros((n_nodes, 6), dtype=np.float64)
        self.amplitude = np.zeros((n_nodes, 4), dtype=np.float64)
        self.debias = np.zeros((n_nodes, 3), dtype=np.float64)
        self.cells = np.zeros(n_nodes, dtype=np.float64)
        self.trace = np.zeros((n_nodes, 4), dtype=np.float64)


def _pair_columns(pairs, n_b: int, position_of: np.ndarray,
                  admissible: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """`pair -> cache positions` of that pair's diffusion-weighted columns.

    `b = 0` is excluded from every domain.  `S(0) = 1` exactly for every stored
    entry, so `J` is identically zero there: the column carries no tissue
    information and enters only as the amplitude reference, which the `n0_eff`
    regimes already model.  Plan section 2.8 audit item R12; the exclusion is
    mathematically forced, not an assumption.
    """
    out: dict[int, np.ndarray] = {}
    for pair in pairs:
        full = pair * n_b + np.arange(1, n_b)
        if admissible is not None:
            full = full[admissible[full]]
        if full.size:
            out[int(pair)] = position_of[full]
    return out


# ---------------------------------------------------------------------------
# Streaming accumulation
# ---------------------------------------------------------------------------

def accumulate(domains: list[Domain], table: dict, vectors, variance, *, n_b: int,
               n_ensembles: int, sigma_pair: np.ndarray, kio_ref: np.ndarray,
               trust_floor: float, rician_min: float, position_of: np.ndarray,
               log_every: int = 200, started: float = 0.0) -> None:
    """One pass over the stored timing pairs, accumulating every declared domain.

    The per-column contributions are computed ONCE per timing pair with unit
    weight and no mask, and each domain then applies its own weighting and its
    own `(node, column)` masks to the same arrays.  A domain is therefore an
    evaluation-time re-weighting of a shared substrate, which is exactly the
    relationship plan section 2.8 requires between the two layers.
    """
    centre = table["centre"]
    n_nodes = len(table["nodes"])
    for domain in domains:
        domain.allocate(n_nodes)
    active_pairs = sorted({pair for domain in domains for pair in domain.columns})
    kio_sq = kio_ref ** 2
    for index, pair in enumerate(active_pairs):
        local_full = pair * n_b + np.arange(1, n_b)
        local = position_of[local_full]
        tissue_raw, amplitude_raw, debias_raw = pair_contributions(
            table, vectors, variance, local, 1.0, n_ensembles, -np.inf, -np.inf)
        signal = np.asarray(vectors[local], dtype=np.float64)[:, centre]   # (columns, nodes)
        # Undebiased diagonal, non-dimensionalized: strictly non-negative, so the
        # annotation shares below are genuine fractions of the information.
        undebiased_trace = (tissue_raw[..., 0] + debias_raw[..., 0]
                            + tissue_raw[..., 3] + debias_raw[..., 1]
                            + (tissue_raw[..., 5] + debias_raw[..., 2]) * kio_sq[None, :])
        sigma_c = float(sigma_pair[pair])
        position_in_local = {int(p): i for i, p in enumerate(local)}
        for domain in domains:
            if pair not in domain.columns:
                continue
            rows = np.asarray([position_in_local[int(p)] for p in domain.columns[pair]], dtype=int)
            whole = rows.size == signal.shape[0]
            block_tissue = tissue_raw if whole else tissue_raw[rows]
            block_amplitude = amplitude_raw if whole else amplitude_raw[rows]
            block_debias = debias_raw if whole else debias_raw[rows]
            block_signal = signal if whole else signal[rows]
            keep = block_signal >= domain.trust_floor
            if domain.rician_averages is not None:
                keep &= block_signal >= rician_min * sigma_c / math.sqrt(domain.rician_averages)
            weight = np.where(keep, 1.0 if domain.uniform_weighting else 1.0 / sigma_c ** 2, 0.0)
            domain.tissue += np.einsum("cn,cnk->nk", weight, block_tissue)
            domain.amplitude += np.einsum("cn,cnk->nk", weight, block_amplitude)
            domain.debias += np.einsum("cn,cnk->nk", weight, block_debias)
            domain.cells += keep.sum(axis=0)
            block_trace = (undebiased_trace if whole else undebiased_trace[rows]) * weight
            domain.trace[:, 0] += block_trace.sum(axis=0)
            domain.trace[:, 1] += np.where(block_signal >= trust_floor, block_trace, 0.0).sum(axis=0)
            domain.trace[:, 2] += np.where(block_signal >= rician_min * sigma_c,
                                           block_trace, 0.0).sum(axis=0)
            domain.trace[:, 3] += np.where(block_signal >= rician_min * sigma_c / 4.0,
                                           block_trace, 0.0).sum(axis=0)
        if log_every and (index + 1) % log_every == 0:
            print(f"[{time.time() - started:7.1f}s] pair {index + 1}/{len(active_pairs)}", flush=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _angle_summary(values: np.ndarray, mask: np.ndarray, null_median: float) -> dict:
    """Distribution of an angle plus the concentration statistics that test it."""
    selected = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    out = _summary(selected)
    out["null_median_for_a_random_direction_deg"] = null_median
    if selected.size:
        for threshold in ANGLE_THRESHOLDS_DEG:
            out[f"fraction_below_{int(threshold)}_deg"] = float(np.mean(selected < threshold))
    return out


def geometry_report(geometry: dict, keep: np.ndarray, table: dict) -> dict:
    """Plan 3.1 and 3.2 aggregates for one Fisher matrix batch."""
    positive = geometry["positive_definite"] & keep
    profiled_positive = geometry["profiled_positive_definite"] & keep
    out = {
        "nodes": int(np.count_nonzero(keep)),
        "positive_definite_nodes": int(np.count_nonzero(positive)),
        "positive_definite_fraction": float(np.mean(geometry["positive_definite"][keep]))
        if np.any(keep) else float("nan"),
        "eigenvalues_of_D_F_D": {
            "lambda_1_stiff": _summary(geometry["eigenvalues"][keep, 0]),
            "lambda_2": _summary(geometry["eigenvalues"][keep, 1]),
            "lambda_3_sloppy": _summary(geometry["eigenvalues"][keep, 2]),
            "note": ("eigenvalues carry the declared noise level and one average per column "
                     "unless a budget scale is recorded; they scale linearly with total "
                     "averaging, while every direction and ratio below does not"),
        },
        "condition_number_lambda1_over_lambda3": _summary(geometry["condition_number"][positive]),
        "sloppy_eigenvalue_separation_lambda2_over_lambda3": _summary(
            geometry["eigenvalue_ratio_2_over_3"][positive]),
        "sloppy_direction_not_uniquely_defined_fraction": float(
            np.mean(geometry["eigenvalue_ratio_2_over_3"][positive] < 2.0))
        if np.any(positive) else float("nan"),
        "sloppy_direction_not_uniquely_defined_note": (
            "share of positive-definite nodes with lambda_2 / lambda_3 < 2, where the smallest "
            "two eigenvalues are close enough that the sloppy eigenvector is a near-arbitrary "
            "choice inside a plane and its angle must be read as such"),
        "sloppy_span_share_lambda2_minus_lambda3_over_span": _summary(
            geometry["sloppy_span_share"][keep]),
        "sloppy_angle_to_constant_vi_deg": _angle_summary(
            geometry["sloppy_angle_deg"], keep, NULL_MEDIAN_ANGLE_3D_DEG),
        "stiff_angle_to_vi_changing_deg": _angle_summary(
            geometry["stiff_angle_deg"], keep, NULL_MEDIAN_ANGLE_3D_DEG),
        "sloppy_in_plane_fraction": _summary(geometry["sloppy_in_plane_fraction"][keep]),
        "sloppy_k_io_fraction": _summary(geometry["sloppy_k_io_fraction"][keep]),
        "sloppy_in_plane_angle_to_constant_vi_deg": _angle_summary(
            geometry["sloppy_in_plane_angle_deg"], keep, NULL_MEDIAN_ANGLE_2D_DEG),
        "k_io_profiled_rho_V_block": {
            "note": ("the (log rho, log V) precision with k_io eliminated by Schur complement. "
                     "Both axes are log parameters, so this companion is independent of the "
                     "D = diag(1, 1, k_io_ref) convention entirely."),
            "valid_nodes": int(np.count_nonzero(profiled_positive)),
            "sloppy_angle_to_constant_vi_deg": _angle_summary(
                geometry["profiled_sloppy_angle_deg"], keep, NULL_MEDIAN_ANGLE_2D_DEG),
            "stiff_angle_to_vi_changing_deg": _angle_summary(
                geometry["profiled_stiff_angle_deg"], keep, NULL_MEDIAN_ANGLE_2D_DEG),
            "condition_number": _summary(geometry["profiled_condition_number"][profiled_positive]),
            "stiff_angle_is_not_independent": (
                "in two dimensions the block's eigenvectors are orthogonal and so are the two "
                "reference directions, so the stiff angle above equals the sloppy angle "
                "identically. It is an arithmetic consistency check, not a second test."),
            "eigenvalues": {
                "lambda_1_stiff": _summary(geometry["profiled_eigenvalues"][keep, 0]),
                "lambda_2_sloppy": _summary(geometry["profiled_eigenvalues"][keep, 1]),
            },
        },
    }
    # Stratified exactly as Phase 2 stratifies, so the two records are readable
    # against each other: k_io sensitivity collapses above roughly 30 s^-1.
    coarse = table["kio"] > 30.0
    out["k_io_region_split"] = {}
    for region, mask in (("k_io_le_30", ~coarse), ("k_io_gt_30", coarse)):
        region_keep = keep & mask
        out["k_io_region_split"][region] = {
            "nodes": int(np.count_nonzero(region_keep)),
            "positive_definite_fraction": float(np.mean(geometry["positive_definite"][region_keep]))
            if np.any(region_keep) else None,
            "sloppy_angle_to_constant_vi_deg": _angle_summary(
                geometry["sloppy_angle_deg"], region_keep, NULL_MEDIAN_ANGLE_3D_DEG),
            "sloppy_k_io_fraction": _summary(geometry["sloppy_k_io_fraction"][region_keep]),
            "k_io_profiled_sloppy_angle_deg": _angle_summary(
                geometry["profiled_sloppy_angle_deg"], region_keep, NULL_MEDIAN_ANGLE_2D_DEG),
        }
    return out


def crlb_report(packed: np.ndarray, kio_ref: np.ndarray) -> dict:
    """Relative CRLB and `kappa`, on Phase 2's definitions, for comparability."""
    inverse, _, positive = packed_inverse_diagonal(packed)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.stack([np.sqrt(inverse[..., 0]), np.sqrt(inverse[..., 1]),
                             np.sqrt(inverse[..., 2]) / kio_ref], axis=-1)
        diagonal = np.stack([packed[..., 0], packed[..., 3], packed[..., 5]], axis=-1)
        kappa = np.sqrt(np.clip(inverse * diagonal, 0.0, np.inf))
    identifiable = np.where(positive[..., None], relative, np.inf)
    per_parameter = np.median(identifiable, axis=-2)
    return {
        "positive_definite_fraction": float(np.mean(positive)),
        "minimax_relative_crlb": float(np.max(per_parameter)),
        "argmax_parameter": PARAMETER_ORDER[int(np.argmax(per_parameter))],
        "relative_crlb_median_over_all_nodes_per_parameter": [float(v) for v in per_parameter],
        "relative_crlb_distribution_per_parameter": {
            name: _summary(np.where(positive, relative[..., i], np.nan))
            for i, name in enumerate(PARAMETER_ORDER)},
        "kappa_distribution_per_parameter": {
            name: _summary(np.where(positive, kappa[..., i], np.nan))
            for i, name in enumerate(PARAMETER_ORDER)},
        "_relative": relative, "_kappa": kappa, "_positive": positive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase2-report", type=Path, default=None,
                        help="executed phase2_report.json; supplies the declared-acquisition domains")
    parser.add_argument("--phase2-subset-size", type=int, default=8,
                        help="which pre-registered b-subset size's Phase-2 optima to reproduce")
    parser.add_argument("--snr", type=float, default=50.0)
    parser.add_argument("--node-stride", type=int, default=1, help="smoke only; subsamples nodes")
    parser.add_argument("--pair-limit", type=int, default=0, help="smoke only; truncates the pair list")
    args = parser.parse_args()

    started = time.time()
    prereg = load_preregistration()
    T2 = float(prereg["noise_model"]["T2_ms"])
    t_epi = float(prereg["noise_model"]["t_epi_ms"])
    trust_floor = float(prereg["trust_floor"])
    rician_min = float(prereg["rician_magnitude_snr_min"])
    kio_floor = float(prereg["protocol_sweep"]["criteria"]["relative_scale_for_k_io"]["k_io_floor_s^-1"])
    n0_values = [float(v) for v in prereg["amplitude_model"]["n0_eff_sweep"]["values"]]
    scenarios = {name: float(limit) for name, limit in prereg["gradient_limits_T_per_m"].items()}

    manifest_json = json.loads((args.phase1 / "phase1_manifest.json").read_text(encoding="utf-8"))
    domain_declaration = read_column_domain(manifest_json)
    print(domain_declaration.banner(), flush=True)
    if not domain_declaration.is_complete:
        raise RuntimeError(
            "Phase 3 answers a question about the model's own degeneracy over the stored "
            "acquisition domain and refuses to run on a restricted substrate. "
            f"{domain_declaration.banner()}")

    with np.load(args.artifact, allow_pickle=False) as data:
        manifest = artifact_manifest(data)
        assert_safe_output(args.output_dir, manifest)
        pair_deltas = np.asarray(data["pair_deltas"], dtype=float)
        pair_Deltas = np.asarray(data["pair_Deltas"], dtype=float)
        b_values = np.asarray(data["b_values"], dtype=float)
        n_b = int(data["n_b"])
        build = json.loads(str(data["build_metadata_json"]))
        table = build_node_table(args.phase1, data)
    n_ensembles = int(np.load(args.cache_dir / "ensemble_means_subset.npy", mmap_mode="r").shape[1])

    if args.node_stride > 1:
        keep_rows = np.arange(0, len(table["nodes"]), args.node_stride)
        for key, value in list(table.items()):
            table[key] = ({axis: rows[keep_rows] for axis, rows in value.items()}
                          if key == "phase1_rows" else value[keep_rows])
    n_nodes = len(table["nodes"])
    kio_ref = np.maximum(table["kio"], kio_floor)

    delta_col, Delta_col, b_col = column_arrays(pair_deltas, pair_Deltas, b_values)
    sigma_pair = te_noise_sigma(pair_deltas, pair_Deltas, sigma0=1.0 / args.snr, T2_ms=T2, t_epi_ms=t_epi)
    pair_TE = pair_deltas + pair_Deltas + t_epi
    position_of = domain_declaration.position_of
    all_pairs = range(len(pair_deltas) if not args.pair_limit else min(args.pair_limit, len(pair_deltas)))

    vectors = transposed_view(args.cache_dir, "vectors")
    variance = transposed_view(args.cache_dir, "signal_variance")

    # ---- declared domains -------------------------------------------------
    domains: list[Domain] = []
    full_columns = _pair_columns(all_pairs, n_b, position_of)
    domains.append(Domain(
        name="full_stored_domain", layer=MODEL_LAYER, columns=full_columns,
        declaration={
            "role": "PRIMARY model-layer result",
            "columns": "every stored diffusion-weighted (delta, Delta, b) column",
            "gradient_ceiling_T_per_m": None,
            "trust_floor_applied": False, "rician_mask_applied": False,
            "column_weighting": ("Sigma from the pre-registered TE/T2 noise model: "
                                 "sigma_c = (1/SNR) exp((delta + Delta + t_epi)/T2)"),
            "why": ("plan section 6: the geometry of the model's own degeneracy. No scanner, "
                    "averaging or trust condition selects these columns."),
        }))
    domains.append(Domain(
        name="full_stored_domain_uniform_weighting", layer=MODEL_LAYER, columns=full_columns,
        uniform_weighting=True,
        declaration={
            "role": "labelled weighting contrast, NOT a criterion",
            "columns": "identical to full_stored_domain",
            "column_weighting": "Sigma = I; every stored column weighted equally, no TE penalty",
            "why": ("plan section 6 names the sloppy-direction angle the weighting-robust "
                    "quantity. This arm measures that robustness instead of asserting it."),
        }))
    domains.append(Domain(
        name="full_stored_domain_trust_floor", layer=MODEL_LAYER, columns=full_columns,
        trust_floor=trust_floor,
        declaration={
            "role": "labelled measurement-trust contrast, NOT a criterion",
            "columns": "identical to full_stored_domain",
            "trust_floor_applied": True, "trust_floor": trust_floor,
            "why": ("the S/S0 trust floor is a conditional measurement-trust assumption "
                    "(domain audit R7). This arm measures whether it rotates the degeneracy."),
        }))
    for scenario, g_max in sorted(scenarios.items(), reverse=True):
        feasible_full = gradient_feasible_columns(delta_col, Delta_col, b_col, g_max)
        require_columns(domain_declaration, feasible_full,
                        f"Phase-3 {scenario} gradient scenario (G <= {g_max} T/m)",
                        delta_col, Delta_col, b_col)
        admissible = np.zeros(len(b_col), dtype=bool)
        admissible[feasible_full] = True
        domains.append(Domain(
            name=f"scenario_wide_{scenario}", layer=SCENARIO_LAYER,
            columns=_pair_columns(all_pairs, n_b, position_of, admissible),
            declaration={
                "role": "conditional scenario-wide result, reported separately and never pooled",
                "columns": f"every stored diffusion-weighted column with G <= {g_max} T/m",
                "gradient_ceiling_T_per_m": g_max,
                "trust_floor_applied": False, "rician_mask_applied": False,
                "column_weighting": "identical to full_stored_domain",
                "why": ("differs from full_stored_domain ONLY by the gradient ceiling, so the "
                        "contrast isolates the hardware condition. The trust-floor and Rician "
                        "conditions are annotated below, and their effect is measured "
                        "separately by full_stored_domain_trust_floor."),
            }))

    phase2 = None
    if args.phase2_report is not None:
        phase2 = json.loads(args.phase2_report.read_text(encoding="utf-8"))
        size = str(args.phase2_subset_size)
        b_index = {float(v): i for i, v in enumerate(b_values)}
        for scenario in sorted(phase2["scenarios"], reverse=True):
            for arm in ("m1", "m2"):
                report = phase2["scenarios"][scenario]["arms"][size][arm]["report"]
                columns = {}
                for pair_key, blist in report["b_values_s_mm2"].items():
                    pair = int(pair_key)
                    full = np.asarray([pair * n_b + b_index[float(b)] for b in blist], dtype=int)
                    columns[pair] = position_of[full]
                domains.append(Domain(
                    name=f"phase2_optimum_{scenario}_size{size}_{arm}", layer=ACQUISITION_LAYER,
                    columns=columns, trust_floor=trust_floor,
                    rician_averages=float(report["averages_per_column"]),
                    budget_scale=float(report["budget_scale_N_over_columns"]),
                    declaration={
                        "role": "conditional declared-acquisition result; reproduces an executed Phase-2 arm",
                        "columns": f"{report['columns_used']} columns at {report['timing_pairs_ms']} ms",
                        "gradient_ceiling_T_per_m": scenarios[scenario],
                        "trust_floor_applied": True, "trust_floor": trust_floor,
                        "rician_mask_applied": True,
                        "averages_per_column": float(report["averages_per_column"]),
                        "budget_scale_N_over_columns": float(report["budget_scale_N_over_columns"]),
                        "budget_images_N": phase2["budget_images_N"],
                        "why": ("attaches the Phase-3 spectra to the executed Phase-2 kappa maps "
                                "and to Phase 4.4, which overlays ill-fitting voxels on them."),
                        "source": str(args.phase2_report),
                    }))

    print(f"[{time.time() - started:7.1f}s] {len(domains)} declared domains, {n_nodes} nodes", flush=True)
    accumulate(domains, table, vectors, variance, n_b=n_b, n_ensembles=n_ensembles,
               sigma_pair=sigma_pair, kio_ref=kio_ref, trust_floor=trust_floor,
               rician_min=rician_min, position_of=position_of, started=started)
    print(f"[{time.time() - started:7.1f}s] accumulation complete", flush=True)

    # ---- report -----------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "evaluation_nodes.npy", table["nodes"].astype(np.int32))
    np.save(args.output_dir / "evaluation_node_labels.npy",
            np.stack([table["rho"], table["V"], table["kio"]], axis=1).astype(np.float64))

    report = {
        "schema": "madi-fisher-phase3-v1",
        "artifact": str(args.artifact), "phase1": str(args.phase1),
        "cache_dir": str(args.cache_dir),
        "phase2_report": None if args.phase2_report is None else str(args.phase2_report),
        "banner": incomplete_banner(manifest), **manifest.as_dict(),
        "column_domain": domain_declaration.as_dict(),
        "column_domain_banner": domain_declaration.banner(),
        "parameter_order": list(PARAMETER_ORDER),
        "snr_at_b0": args.snr, "T2_ms": T2, "t_epi_ms": t_epi,
        "n_ensembles": n_ensembles, "walkers_per_ensemble": build.get("walkers_per_ensemble"),
        "node_stride": args.node_stride, "pair_limit": args.pair_limit,
        "non_dimensionalization": {
            "D": "diag(1, 1, k_io_ref)", "k_io_ref": "max(k_io_node, k_io_floor)",
            "k_io_floor_s^-1": kio_floor,
            "why_this_k_io_ref": ("the pre-registered relative-CRLB floor, reused so Phase-3 "
                                  "spectra and Phase-2 relative CRLBs are non-dimensionalized "
                                  "on one convention"),
            "invariance": ("D leaves the (log rho, log V) axes untouched, so the constant-v_i "
                           "direction and every angle measured in that plane are independent "
                           "of k_io_ref; only the k_io component of a 3-vector depends on it"),
        },
        "reference_directions": {
            "constant_v_i": [float(v) for v in CONSTANT_VI_DIRECTION],
            "v_i_changing": [float(v) for v in VI_CHANGING_DIRECTION],
            "note": ("log v_i = log rho + log V + const, so (1, -1, 0)/sqrt(2) holds v_i and the "
                     "geometry factor g(v_i) of plan section 2.4 exactly constant"),
        },
        "monte_carlo_debias": {
            "form": "endpoint-only Var(J_hat) = (Var(S-) + Var(S+))/(n_ensembles h^2)",
            "why": ("the CRN covariance exists for only the 200 diagnostic columns; dropping a "
                    "positive covariance OVERSTATES Var(J_hat), so every debiased matrix here is "
                    "a lower bound and every CRLB conservative. Calibrated in fisher_phase2.md."),
            "diagonal_only": ("central differences on the three axes use four disjoint library "
                              "entries, so the off-diagonals -- the elements that carry the "
                              "degeneracy this phase measures -- are already unbiased"),
        },
        "evaluation_nodes": {
            "count": int(n_nodes),
            "definition": "every node carrying all three k=1 central stencils; no k_io ceiling",
            "k_io_range_s_inv": [float(table["kio"].min()), float(table["kio"].max())],
            "k_io_above_30_node_fraction": float(np.mean(table["kio"] > 30.0)),
            "rho_V_pairs_retained": len(sorted(set((int(a), int(b)) for a, b, _ in table["nodes"]))),
        },
        "domains": {},
    }

    for domain in domains:
        pairs_used = sorted(domain.columns)
        packed_tt = domain.tissue * domain.budget_scale
        F_ta = domain.amplitude[:, :3] * domain.budget_scale
        F_aa = domain.amplitude[:, 3] * domain.budget_scale
        undebiased = packed_tt.copy()
        for i, position in enumerate((0, 3, 5)):
            undebiased[:, position] += domain.debias[:, i] * domain.budget_scale
        keep_nodes = domain.cells > 0
        if not pairs_used or not np.any(keep_nodes):
            # A declared domain that admits no column is reported as empty rather
            # than silently omitted: an absent scenario is a result about that
            # scenario, not a missing row.
            report["domains"][domain.name] = {
                "layer": domain.layer, "declaration": domain.declaration,
                "timing_pairs": len(pairs_used), "columns": 0,
                "nodes_with_at_least_one_column": int(np.count_nonzero(keep_nodes)),
                "empty": "this declaration admits no (node, column) cell on this node set",
            }
            print(f"[{time.time() - started:7.1f}s] reported {domain.name} (empty)", flush=True)
            continue
        reference_pair = min(pairs_used, key=lambda pair: pair_TE[pair])
        sigma_reference = float(sigma_pair[reference_pair])

        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(domain.trace[:, 0:1] > 0,
                             domain.trace[:, 1:] / domain.trace[:, 0:1], np.nan)
        entry = {
            "layer": domain.layer,
            "declaration": domain.declaration,
            "timing_pairs": len(pairs_used),
            "columns": int(sum(len(v) for v in domain.columns.values())),
            "columns_per_node": _summary(domain.cells[keep_nodes]),
            "nodes_with_at_least_one_column": int(np.count_nonzero(keep_nodes)),
            "amplitude_reference": {
                "pair_ms": [float(pair_deltas[reference_pair]), float(pair_Deltas[reference_pair])],
                "sigma": sigma_reference,
                "convention": ("the shortest-TE timing pair present in this domain, which is what "
                               "an operator acquiring a b ~ 0 reference would use; S(0) = 1 "
                               "exactly in the stored grid, so no extrapolation enters"),
            },
            "conditional_annotations": {
                "note": ("shares of this domain's UNDEBIASED non-dimensionalized Fisher trace, per "
                         "node, contributed by cells satisfying a condition that is NOT applied "
                         "here. Plan section 2.8: a conditional quantity may annotate a substrate, "
                         "never select it."),
                "share_above_trust_floor": _summary(share[:, 0]),
                "share_rician_valid_at_1_average": _summary(share[:, 1]),
                "share_rician_valid_at_16_averages": _summary(share[:, 2]),
            },
            "regimes": {},
        }

        maps = {}
        for label, n0 in [("known_amplitude", None)] + [(f"n0_eff_{int(v)}", v) for v in n0_values]:
            if domain.uniform_weighting and n0 not in (None, 0.0):
                # Sigma = I here is a diagnostic re-weighting, not a physical noise
                # level, so "n0_eff averages of a b ~ 0 reference at noise sigma_ref"
                # has no meaning on this arm: the prior and F_aa would be expressed
                # on two different scales and their ratio would be an artifact of
                # that mismatch rather than a statement about amplitude knowledge.
                # The two scale-free bounds -- known amplitude and unknown amplitude
                # -- are reported, and the finite regimes are declared unavailable
                # rather than computed on a fabricated convention.
                entry["regimes"][label] = {
                    "unavailable": ("this arm's Sigma = I carries no physical noise scale, so a "
                                    "finite amplitude prior cannot be placed on the same scale as "
                                    "its F_s0_s0. Read known_amplitude and n0_eff_0, which are "
                                    "scale-free bounds, and take the finite regimes from the "
                                    "TE-weighted arms."),
                }
                continue
            # `amplitude_prior_precision` is the audited helper; `S(0) = 1` exactly
            # in the stored grid, so its default reference signal applies and no
            # extrapolation enters (unlike the Jackson-thesis b = 50 case).
            prior = np.inf if n0 is None else amplitude_prior_precision(float(n0), sigma_reference)
            packed = (packed_tt if np.isinf(prior)
                      else packed_amplitude_marginal(packed_tt, F_ta, F_aa, prior))
            block = {"s0_prior_precision": None if n0 is None else prior}
            if n0 is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    strength = np.where(F_aa > 0, prior / F_aa, np.nan)
                block["prior_precision_over_F_s0_s0"] = _summary(strength[keep_nodes])
                block["prior_strength_note"] = (
                    "lambda / F_s0_s0 per node. F_s0_s0 grows with the number of tissue columns "
                    "while lambda does not, so a fixed n0_eff constrains the amplitude of a wide "
                    "acquisition proportionally less. Read the finite regimes of a many-column "
                    "domain accordingly: they sit near the unknown-amplitude bound because four "
                    "b ~ 0 averages are a small reference for tens of thousands of images, which "
                    "is a statement about proportion, not about the marginalization.")
            crlb = crlb_report(packed, kio_ref)
            block["crlb"] = {k: v for k, v in crlb.items() if not k.startswith("_")}
            # Plan sections 3.1/3.2 under the bracketing amplitude regimes: the
            # known-amplitude bound of section 2.3, the unknown-amplitude bound that
            # section 2.7 requires be reported beside it, and the pre-registered
            # worked reference n0_eff = 4 between them.  Marginalizing S0 subtracts
            # a rank-one outer product, so it can rotate the sloppy direction;
            # whether it does, and how much of the rotation a real b ~ 0 reference
            # buys back, is measured here rather than assumed.
            if label in ("known_amplitude", "n0_eff_0", "n0_eff_4"):
                geometry = degeneracy_geometry(packed, kio_ref)
                block["geometry"] = geometry_report(geometry, keep_nodes, table)
                if label == "known_amplitude":
                    maps.update({
                        "eigenvalues": geometry["eigenvalues"],
                        "eigenvectors": geometry["eigenvectors"],
                        "condition_number": geometry["condition_number"],
                        "eigenvalue_ratio_2_over_3": geometry["eigenvalue_ratio_2_over_3"],
                        "sloppy_span_share": geometry["sloppy_span_share"],
                        "sloppy_angle_deg": geometry["sloppy_angle_deg"],
                        "stiff_angle_deg": geometry["stiff_angle_deg"],
                        "sloppy_in_plane_fraction": geometry["sloppy_in_plane_fraction"],
                        "sloppy_k_io_fraction": geometry["sloppy_k_io_fraction"],
                        "sloppy_in_plane_angle_deg": geometry["sloppy_in_plane_angle_deg"],
                        "profiled_sloppy_vector": geometry["profiled_sloppy_vector"],
                        "profiled_sloppy_angle_deg": geometry["profiled_sloppy_angle_deg"],
                        "profiled_stiff_angle_deg": geometry["profiled_stiff_angle_deg"],
                        "profiled_condition_number": geometry["profiled_condition_number"],
                        "profiled_eigenvalues": geometry["profiled_eigenvalues"],
                        "positive_definite": geometry["positive_definite"].astype(np.uint8),
                        "relative_crlb": crlb["_relative"], "kappa": crlb["_kappa"],
                    })
            entry["regimes"][label] = block

        # The bias trap of plan section 2.5, carried into the degeneracy geometry:
        # Monte-Carlo noise adds to E[J_hat^2], so an undebiased matrix reports more
        # information than the library holds.  Phase 2 measured what that does to
        # identifiability; this measures what it does to the DIRECTION.
        undebiased_geometry = degeneracy_geometry(undebiased, kio_ref)
        debiased_geometry = degeneracy_geometry(packed_tt, kio_ref)
        both = keep_nodes & debiased_geometry["positive_definite"] & undebiased_geometry["positive_definite"]
        rotation = np.degrees(np.arccos(np.clip(np.abs(np.sum(
            debiased_geometry["sloppy_vector"] * undebiased_geometry["sloppy_vector"], axis=-1)), 0, 1)))
        entry["monte_carlo_debias_effect"] = {
            "geometry_undebiased": geometry_report(undebiased_geometry, keep_nodes, table),
            "sloppy_direction_rotation_deg": _summary(rotation[both]),
            "nodes_compared": int(np.count_nonzero(both)),
            "mean_debias_over_fisher_diagonal": [
                float(np.mean(domain.debias[keep_nodes, i]
                              / np.maximum(undebiased[keep_nodes, position] / domain.budget_scale, 1e-300)))
                for i, position in enumerate((0, 3, 5))],
        }

        prefix = args.output_dir / f"maps_{domain.name}"
        for name, values in maps.items():
            np.save(prefix.with_name(prefix.name + f".{name}.npy"),
                    np.asarray(values, dtype=np.float32 if values.dtype != np.uint8 else np.uint8))
        report["domains"][domain.name] = entry
        print(f"[{time.time() - started:7.1f}s] reported {domain.name}", flush=True)

    # ---- Phase-2 reproduction cross-check ---------------------------------
    if phase2 is not None and (args.node_stride > 1 or args.pair_limit):
        report["phase2_reproduction_cross_check"] = {
            "skipped": ("a subsampled node set or a truncated pair list changes the median over "
                        "nodes, so the Phase-2 quantities are not comparable. The cross-check runs "
                        "only at --node-stride 1 with no --pair-limit."),
            "node_stride": args.node_stride, "pair_limit": args.pair_limit,
        }
    elif phase2 is not None:
        checks = {}
        size = str(args.phase2_subset_size)
        for scenario in sorted(phase2["scenarios"], reverse=True):
            for arm in ("m1", "m2"):
                name = f"phase2_optimum_{scenario}_size{size}_{arm}"
                if name not in report["domains"] or "empty" in report["domains"][name]:
                    continue
                stored = phase2["scenarios"][scenario]["arms"][size][arm]["report"]["regimes"]
                mine = report["domains"][name]["regimes"]
                worst = 0.0
                fields = []
                for label in stored:
                    if label not in mine:
                        continue
                    pairs = [("positive_definite_fraction",
                              stored[label]["positive_definite_fraction"],
                              mine[label]["crlb"]["positive_definite_fraction"]),
                             ("minimax_score", stored[label]["minimax_score"],
                              mine[label]["crlb"]["minimax_relative_crlb"])]
                    for i, parameter in enumerate(PARAMETER_ORDER):
                        pairs.append((
                            f"relative_crlb_median[{parameter}]",
                            stored[label]["relative_crlb_median_over_all_nodes_per_parameter"][i],
                            mine[label]["crlb"]["relative_crlb_median_over_all_nodes_per_parameter"][i]))
                    for field_name, a, b in pairs:
                        if a is None or not np.isfinite(a) or not np.isfinite(b):
                            continue
                        difference = abs(float(a) - float(b)) / max(abs(float(a)), 1e-300)
                        worst = max(worst, difference)
                        fields.append(f"{label}.{field_name}")
                checks[name] = {
                    "quantities_compared": len(fields),
                    "max_relative_difference": worst,
                    "note": ("Phase 3 re-forms the Phase-2 arms from the same cache by an "
                             "independent accumulation path (one pass over all pairs, per-domain "
                             "re-weighting) rather than Phase 2's per-arm path. Agreement at "
                             "float64 rounding confirms both."),
                }
        report["phase2_reproduction_cross_check"] = checks

    report["runtime_seconds"] = time.time() - started
    (args.output_dir / "phase3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "summary.txt").write_text(summarize(report), encoding="utf-8")
    print(summarize(report))
    return 0


def summarize(report: dict) -> str:
    lines = [f"Fisher/CRLB Phase 3 — {report['schema']}",
             report["column_domain_banner"], report["banner"],
             f"nodes={report['evaluation_nodes']['count']}  SNR={report['snr_at_b0']}  "
             f"T2={report['T2_ms']} ms  t_epi={report['t_epi_ms']} ms", ""]
    for name, entry in report["domains"].items():
        if "empty" in entry:
            lines.append(f"[{entry['layer']}] {name}: EMPTY — {entry['empty']}")
            lines.append("")
            continue
        geometry = entry["regimes"]["known_amplitude"]["geometry"]
        angle = geometry["sloppy_angle_to_constant_vi_deg"]
        profiled = geometry["k_io_profiled_rho_V_block"]["sloppy_angle_to_constant_vi_deg"]
        lines.append(f"[{entry['layer']}] {name}")
        lines.append(f"  columns={entry['columns']}  pairs={entry['timing_pairs']}  "
                     f"PD fraction={geometry['positive_definite_fraction']:.4f}")
        lines.append(f"  condition number lambda1/lambda3: median "
                     f"{geometry['condition_number_lambda1_over_lambda3'].get('median', float('nan')):.4g}"
                     f"  q95 {geometry['condition_number_lambda1_over_lambda3'].get('q95', float('nan')):.4g}")
        lines.append(f"  sloppy angle to constant-v_i (3D): median {angle.get('median', float('nan')):.2f} deg"
                     f"  <10 deg {angle.get('fraction_below_10_deg', float('nan')):.4f}"
                     f"  (random-direction median {angle['null_median_for_a_random_direction_deg']:.0f})")
        lines.append(f"  sloppy k_io fraction: median "
                     f"{geometry['sloppy_k_io_fraction'].get('median', float('nan')):.4f}")
        lines.append(f"  k_io-profiled (rho,V) sloppy angle: median "
                     f"{profiled.get('median', float('nan')):.2f} deg"
                     f"  <10 deg {profiled.get('fraction_below_10_deg', float('nan')):.4f}"
                     f"  (random median {profiled['null_median_for_a_random_direction_deg']:.0f})")
        lines.append("")
    cross_check = report.get("phase2_reproduction_cross_check", {})
    if "skipped" in cross_check:
        lines.append(f"Phase-2 reproduction cross-check SKIPPED: {cross_check['skipped']}")
    for name, check in cross_check.items():
        if not isinstance(check, dict):
            continue
        lines.append(f"Phase-2 reproduction {name}: {check['quantities_compared']} quantities, "
                     f"max relative difference {check['max_relative_difference']:.3e}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
