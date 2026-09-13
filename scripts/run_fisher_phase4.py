#!/usr/bin/env python3
"""Phase 4: the unrealistic-cell-volume pathology.

Plan section 7.  Objective: explain the artifact MADI III named and could not
account for, and that the Jackson thesis characterized and handled with a 20 pL
threshold.  Question answered: **explanation of a fitting pathology**.

Four hypotheses, tested against one acquisition with one condition varied per
fit arm (`scripts/run_fisher_phase4_fits.sh` produces the arms):

  H4  S0 mismatch               4.1  refit with --fit-s0; blow-up shrinks?
  H3  trust-floor violation     4.2  refit with the S/S0 floor masked; shrinks?
  H1  degeneracy ridge + mask   4.3  low residuals; 4.4 blow-ups on high kappa, at
                                     the band edge, sliding along the ridge
  H2  out-of-model signal       4.3  high residuals; high ADC

and 4.5, the principled replacement for the 20 pL cutoff: `log v_i` with a
CRLB-derived error bar, `rho` and `V` separately only below the pre-registered
`kappa` threshold, and the Bayes posterior spread validated against the CRLB.

Conditions enter as experimental conditions switched on and off (section 7),
never as selections of the substrate: the Fisher geometry at the acquisition is a
re-evaluation of the unrestricted Phase-1 substrate, and the trust-floor and
amplitude arms are separate fits of the same data.

This script computes no Fisher or fitting arithmetic of its own.  Fisher
quantities come from `madi.fisher_crlb` through the Phase-2 per-column helper;
pathology statistics from `madi.volume_pathology`; the measured signal from the
fitter's own `load_dwi_and_average`, so the ADC and the noise scale are built by
exactly the code path the fits used.

It also records one decision it does not make.  Whether `log v_i` is *reported*
at a voxel whose Fisher matrix is singular depends on how much of the `v_i`
contrast may lie in an uninformative direction; that tolerance is not
pre-registered and it changes the voxel coverage of the 4.5 report materially.
Every tolerance in `DEFECT_TOLERANCES` is therefore reported, and none is
nominated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from madi.fisher_crlb import (LOG_VI_CONTRAST, PARAMETER_ORDER, amplitude_prior_precision,
                              canonical_grid, degeneracy_geometry, directional_crlb,
                              estimable_rho_V_contrast_bound, load_preregistration,
                              nearest_canonical_node, packed_amplitude_marginal,
                              packed_inverse_diagonal, read_column_domain)
from madi.volume_pathology import (FREE_WATER_ADC_UM2_PER_MS, THESIS_VOLUME_CUTOFF_PL, apparent_diffusion_coefficient,
                                   band_position, cutoff_sweep, expected_noise_residual,
                                   goodness_of_fit_ratio, kappa_gated_report, log_displacement,
                                   permuted_displacement_null,
                                   probability_of_superiority, reachable_volume_ceiling,
                                   stratified_comparison, validate_quality_flag)
from scripts.fit_data import B0_THRESHOLD, load_dwi_and_average, save_map
from scripts.run_fisher_phase2 import _summary, build_node_table, pair_contributions, transposed_view

ARMS = ("baseline_map", "baseline_bayes", "fit_s0_map", "fit_s0_bayes",
        "trust_floor_column_map", "trust_floor_column_bayes",
        "trust_floor_candidate_map", "trust_floor_candidate_bayes")
MAP_NAMES = {"kio": "kio_map", "rho": "rho_map", "V": "V_map", "residual": "residual_map"}
BAYES_NAMES = {"kio": "kio_mean", "rho": "rho_mean", "V": "V_mean", "kio_std": "kio_std",
               "rho_std": "rho_std", "V_std": "V_std", "residual": "residual", "n_eff": "n_eff"}
VOLUME_CUTOFFS_PL = (5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 90.0)
DEFECT_TOLERANCES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
REGIMES = ("known_amplitude", "this_scan_reference", "unknown_amplitude")
# Fixed so the geometric null of every ridge comparison is reproducible from the record.
RIDGE_NULL_SEED = 20260910


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arm(root: Path, name: str, mask: np.ndarray, mask_sha256: str) -> dict | None:
    """One completed fit arm as voxel vectors in the mask's C order, or None.

    Refuses an arm whose recorded mask is not the declared mask.  An existing
    output (`madi_output_glioma_v4.0/map`) was fitted with a different subject's
    mask while filed beside a correctly masked run, and its zero voxels read as
    exactly the "not fitted" pattern a pathology count must not absorb; a hash
    check is the only guard that cannot be fooled by a plausible directory name.
    """
    directory = root / name
    sidecar_path = directory / f"{name}.json"
    if not sidecar_path.exists():
        return None
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("status") != "completed":
        raise RuntimeError(f"arm {name}: sidecar status is {sidecar.get('status')!r}, not 'completed'")
    recorded = sidecar["input_data_provenance"]["mask"]["sha256"]
    if recorded != mask_sha256:
        raise RuntimeError(f"arm {name}: fitted with mask sha256 {recorded[:12]}..., not the declared "
                           f"mask {mask_sha256[:12]}...; refusing to compare fits of different voxels")
    configuration = sidecar["fit_configuration"]
    method = configuration["method"]
    names = MAP_NAMES if method == "map" else BAYES_NAMES
    arm = {"name": name, "method": method, "sidecar": sidecar,
           "s0_mode": configuration["S0_mode"],
           "trust_floor_mask": configuration.get("trust_floor_mask"),
           "fit_triples": configuration["fit_triples_delta_Delta_b"]}
    for key, stem in names.items():
        arm[key] = np.asarray(nib.load(directory / f"{stem}.nii.gz").dataobj, dtype=float)[mask]
    # A voxel at rho = V = 0 is the free-water atom or a voxel the fit never
    # reached.  Free water is excluded from every arm's candidate set, so here it
    # can only mean "not fitted", and it is carried as that rather than as a
    # volume of zero.
    arm["fitted"] = (np.isfinite(arm["V"]) & np.isfinite(arm["rho"])
                     & ~((arm["V"] == 0) & (arm["rho"] == 0)))
    arm["not_fitted_count"] = int(np.count_nonzero(~arm["fitted"]))
    return arm


def acquisition_fisher(table: dict, vectors, variance, n_ensembles: int, *, pair: int, n_b: int,
                       b_values: np.ndarray, shells: dict[float, int], sigma_norm: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Debiased tissue, amplitude and debias sums at a declared acquisition.

    Each column is weighted by its own averaging, `u_c = n_c / sigma_norm^2`,
    which `run_fisher_phase3.Domain` cannot express because it carries one noise
    level per timing pair.  The per-column arithmetic is Phase 2's
    `pair_contributions`, called once per column; only the sum is formed here.
    At a single timing pair the TE/T2 factor is common to every column, so every
    ratio, condition number, angle and defect computed from these sums is
    independent of it; only absolute bounds carry the measured noise scale.
    """
    n_nodes = len(table["nodes"])
    tissue = np.zeros((n_nodes, 6))
    amplitude = np.zeros((n_nodes, 4))
    debias = np.zeros((n_nodes, 3))
    for b, averages in shells.items():
        index = np.flatnonzero(np.isclose(b_values, b))
        if index.size != 1:
            raise RuntimeError(f"b = {b:g} s/mm2 is not a stored library b-value")
        column = np.asarray([pair * n_b + int(index[0])])
        t, a, d = pair_contributions(table, vectors, variance, column,
                                     float(averages) / sigma_norm ** 2, n_ensembles, -np.inf, -np.inf)
        tissue += t.sum(axis=0)
        amplitude += a.sum(axis=0)
        debias += d.sum(axis=0)
    return tissue, amplitude, debias


def fisher_layer(tissue: np.ndarray, amplitude: np.ndarray, kio_ref: np.ndarray,
                 priors: dict[str, float]) -> dict[str, dict]:
    """Per-node quantities Phase 4 reads, under each declared amplitude regime."""
    out = {}
    for regime, prior in priors.items():
        packed = (tissue if np.isinf(prior)
                  else packed_amplitude_marginal(tissue, amplitude[:, :3], amplitude[:, 3], prior))
        inverse, _, positive = packed_inverse_diagonal(packed)
        diagonal = np.stack([packed[:, 0], packed[:, 3], packed[:, 5]], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            crlb = np.sqrt(inverse)
            kappa = np.where(positive[:, None], np.sqrt(np.clip(inverse * diagonal, 0, np.inf)), np.nan)
        geometry = degeneracy_geometry(packed, kio_ref)
        estimable = estimable_rho_V_contrast_bound(packed)
        out[regime] = {
            "packed": packed, "positive": positive, "crlb": crlb, "kappa": kappa,
            "condition_number": geometry["condition_number"],
            "profiled_sloppy_angle_deg": geometry["profiled_sloppy_angle_deg"],
            "log_vi_crlb_exact": directional_crlb(packed, LOG_VI_CONTRAST),
            "log_vi_bound": estimable["bound"], "log_vi_defect": estimable["defect"],
            "log_vi_non_positive_count": estimable["non_positive_count"],
        }
    return out


def fisher_layer_summary(layer: dict) -> dict:
    """Node-level distributions for the report (not voxel-level; see `voxel_*`)."""
    out = {}
    for regime, q in layer.items():
        positive = q["positive"]
        out[regime] = {
            "positive_definite_fraction": float(np.mean(positive)),
            "kappa_median_over_pd_nodes": [float(v) for v in np.nanmedian(q["kappa"], axis=0)]
            if positive.any() else None,
            "relative_crlb_log_rho_log_V_median_over_pd_nodes":
                [float(np.median(q["crlb"][positive, i])) for i in (0, 1)] if positive.any() else None,
            "condition_number": _summary(q["condition_number"][positive]),
            "profiled_sloppy_angle_deg": _summary(q["profiled_sloppy_angle_deg"]),
            "log_vi_crlb_exact_over_pd_nodes": _summary(q["log_vi_crlb_exact"]),
            "log_vi_defect": _summary(q["log_vi_defect"]),
            "log_vi_non_positive_count": {str(k): int(np.count_nonzero(q["log_vi_non_positive_count"] == k))
                                          for k in (-1, 0, 1, 2)},
            "log_vi_defect_below_tolerance_node_fraction": {
                str(t): float(np.mean(q["log_vi_defect"] <= t)) for t in DEFECT_TOLERANCES},
        }
    return out


# ---------------------------------------------------------------------------
# Voxel <-> node join
# ---------------------------------------------------------------------------

def node_rows(table: dict, rho: np.ndarray, volume: np.ndarray, kio: np.ndarray) -> dict:
    """Map each voxel to its canonical node and, if it has one, its Fisher row.

    Rows are -1 where the node carries no Fisher matrix: the mask-band edges,
    where 136 of 369 `(rho, V)` pairs have no interior stencil, and the two
    `k_io` end nodes of every pair.  Those voxels are reported as having no
    bound, never assigned a neighbour's.
    """
    ir, iv, ik = nearest_canonical_node(rho, volume, kio)
    nodes = np.asarray(table["nodes"], dtype=int)
    shape = tuple(int(v) for v in np.maximum(nodes.max(axis=0), [ir.max(), iv.max(), ik.max()]) + 1)
    lookup = np.full(shape, -1, dtype=np.int64)
    lookup[nodes[:, 0], nodes[:, 1], nodes[:, 2]] = np.arange(len(nodes))
    valid = (ir >= 0) & (iv >= 0) & (ik >= 0)
    rows = np.full(ir.shape, -1, dtype=np.int64)
    rows[valid] = lookup[ir[valid], iv[valid], ik[valid]]
    rhos_c, volumes_c, _, _ = canonical_grid()
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_offset = np.where(valid, np.abs(np.log(rho) - np.log(rhos_c[np.clip(ir, 0, None)])), np.nan)
        volume_offset = np.where(valid, np.abs(np.log(volume) - np.log(volumes_c[np.clip(iv, 0, None)])), np.nan)
    return {"rows": rows, "has_node": valid, "has_fisher": rows >= 0,
            "rho_index": ir, "V_index": iv, "k_io_index": ik,
            "log_rho_offset": rho_offset, "log_V_offset": volume_offset,
            "half_log_step_rho": float(np.log(rhos_c[1] / rhos_c[0]) / 2),
            "half_log_step_V": float(np.log(volumes_c[1] / volumes_c[0]) / 2)}


def at_rows(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """`values[rows]` with NaN wherever the row is -1."""
    values = np.asarray(values, dtype=float)
    out = np.full(rows.shape + values.shape[1:], np.nan)
    ok = rows >= 0
    out[ok] = values[rows[ok]]
    return out


# ---------------------------------------------------------------------------
# Hypothesis blocks
# ---------------------------------------------------------------------------

def blow_up_block(arm: dict) -> dict:
    fitted = arm["fitted"]
    volume = np.where(fitted, arm["V"], np.nan)
    return {
        "fitted_voxels": int(np.count_nonzero(fitted)),
        "not_fitted_voxels": arm["not_fitted_count"],
        "volume_pL": _summary(volume),
        "cutoff_sweep": cutoff_sweep(volume, np.asarray(VOLUME_CUTOFFS_PL)),
        "fraction_above_thesis_cutoff": float(np.nanmean(volume > THESIS_VOLUME_CUTOFF_PL)),
    }


def condition_contrast(baseline: dict, varied: dict, label: str) -> dict:
    """Did the pathology shrink when one condition was switched on?

    Reported at every cutoff, with a voxel-level transition count on the voxels
    both arms fitted, so a shrinkage that exists at one cutoff only, or that is a
    reshuffling of which voxels blow up rather than fewer of them, is visible.
    """
    both = baseline["fitted"] & varied["fitted"]
    rows = []
    for cutoff in VOLUME_CUTOFFS_PL:
        before = baseline["V"][both] > cutoff
        after = varied["V"][both] > cutoff
        rows.append({"cutoff_pL": cutoff,
                     "fraction_before": float(np.mean(before)), "fraction_after": float(np.mean(after)),
                     "relative_change": (float((np.mean(after) - np.mean(before)) / np.mean(before))
                                         if np.mean(before) > 0 else None),
                     "stayed_blown_up": int(np.count_nonzero(before & after)),
                     "resolved": int(np.count_nonzero(before & ~after)),
                     "newly_blown_up": int(np.count_nonzero(~before & after))})
    changes = [r["relative_change"] for r in rows if r["relative_change"] is not None]
    return {
        "condition": label, "voxels_fitted_in_both": int(np.count_nonzero(both)),
        "by_cutoff": rows,
        "shrinks_at_every_cutoff": bool(changes) and all(c < 0 for c in changes),
        "grows_at_every_cutoff": bool(changes) and all(c > 0 for c in changes),
        "median_volume_before_after_pL": [float(np.median(baseline["V"][both])),
                                          float(np.median(varied["V"][both]))],
    }


def ridge_displacement(from_arm: dict, to_arm: dict, label: str) -> dict:
    """How estimates moved between two fits: along the ridge, or across it?"""
    both = from_arm["fitted"] & to_arm["fitted"]
    step = log_displacement(from_arm["rho"][both], from_arm["V"][both],
                            to_arm["rho"][both], to_arm["V"][both])
    null = permuted_displacement_null(from_arm["rho"][both], from_arm["V"][both],
                                      to_arm["rho"][both], to_arm["V"][both], seed=RIDGE_NULL_SEED)
    moved = step["moved"]
    angle = step["angle_to_hyperbola_deg"][moved]
    null_angle = null["angle_to_hyperbola_deg"][null["moved"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        vi_share = np.abs(step["d_log_vi"][moved]) / np.abs(step["d_log_volume"][moved])
        null_vi_share = np.abs(null["d_log_vi"][null["moved"]]) / np.abs(null["d_log_volume"][null["moved"]])
    return {
        "label": label, "voxels_compared": int(np.count_nonzero(both)),
        "voxels_moved": int(np.count_nonzero(moved)),
        "angle_to_hyperbola_deg": _summary(angle),
        "fraction_within_10_deg": float(np.mean(angle < 10.0)) if angle.size else None,
        "uniform_null_median_deg": 45.0,
        "geometric_null": {
            "definition": ("each voxel's starting estimate paired with a randomly chosen other voxel's perturbed "
                           "estimate (seed RIDGE_NULL_SEED): keeps the band geometry and the fitted distribution, "
                           "breaks only the voxel-to-own-refit link"),
            "angle_to_hyperbola_deg": _summary(null_angle),
            "fraction_within_10_deg": float(np.mean(null_angle < 10.0)) if null_angle.size else None,
            "abs_d_log_vi_over_abs_d_log_V": _summary(null_vi_share),
            "probability_null_angle_exceeds_observed": probability_of_superiority(null_angle, angle),
            "probability_null_vi_share_exceeds_observed": probability_of_superiority(null_vi_share, vi_share),
        },
        "abs_d_log_V": _summary(np.abs(step["d_log_volume"][moved])),
        "abs_d_log_vi": _summary(np.abs(step["d_log_vi"][moved])),
        "abs_d_log_vi_over_abs_d_log_V": _summary(vi_share),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--dwi", type=Path, required=True)
    parser.add_argument("--bval", type=Path, required=True)
    parser.add_argument("--bvec", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--small-delta", type=float, required=True)
    parser.add_argument("--Delta", type=float, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--phase3-run-dir", type=Path, default=None,
                        help="executed Phase-3 output; supplies the model-layer kappa map for 4.4")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    prereg = load_preregistration()
    kappa_threshold = float(prereg["kappa_unidentified_threshold"])
    kio_floor = float(prereg["protocol_sweep"]["criteria"]["relative_scale_for_k_io"]["k_io_floor_s^-1"])

    # ---- substrate and node table ------------------------------------------
    manifest = json.loads((args.phase1 / "phase1_manifest.json").read_text(encoding="utf-8"))
    domain = read_column_domain(manifest)
    print(domain.banner(), flush=True)
    if not domain.is_complete:
        raise RuntimeError("Phase 4 evaluates a declared acquisition against the unrestricted substrate "
                           f"and refuses a restricted one. {domain.banner()}")
    with np.load(args.artifact, allow_pickle=False) as data:
        pair_deltas = np.asarray(data["pair_deltas"], dtype=float)
        pair_Deltas = np.asarray(data["pair_Deltas"], dtype=float)
        lib_b_values = np.asarray(data["b_values"], dtype=float)
        n_b = int(data["n_b"])
        table = build_node_table(args.phase1, data)
    kio_ref = np.maximum(table["kio"], kio_floor)
    matches = np.flatnonzero(np.isclose(pair_deltas, args.small_delta) & np.isclose(pair_Deltas, args.Delta))
    if matches.size != 1:
        raise RuntimeError(f"(delta, Delta) = ({args.small_delta:g}, {args.Delta:g}) ms is not a stored pair")
    pair = int(matches[0])

    # ---- arms ----------------------------------------------------------------
    mask_image = nib.load(args.mask)
    mask = np.asarray(mask_image.dataobj).astype(bool)
    mask_sha = _sha256(args.mask)
    arms = {name: load_arm(args.fit_root, name, mask, mask_sha) for name in ARMS}
    present = [n for n, a in arms.items() if a is not None]
    print(f"[{time.time()-started:6.1f}s] arms present: {present}", flush=True)
    for required in ("baseline_map", "baseline_bayes"):
        if arms[required] is None:
            raise RuntimeError(f"required arm {required} is missing under {args.fit_root}")
    baseline_map, baseline_bayes = arms["baseline_map"], arms["baseline_bayes"]
    grid_policy = baseline_map["sidecar"]["fit_configuration"]["grid_edge_policy"]
    vi_min, vi_max = float(grid_policy["vi_min"]), float(grid_policy["vi_max"])
    requested = baseline_map["sidecar"]["requested_cli_arguments"]

    # ---- measured signal, through the fitter's own loader ---------------------
    # Every loader argument that shapes the measured signal is taken from what the
    # baseline fit actually requested, never from the loader's own defaults.  The
    # CLI's background-dilation default differs from the function's, and a
    # different dilation estimates a different Rician sigma -- which would silently
    # change both the Rician-corrected signal and the noise scale this phase reads
    # the fits against.  The recovered sigma is then asserted equal to the one the
    # fit recorded, so "the same code path" is checked rather than assumed.
    measured, fit_triples, affine, mask_idx, shape, extras, sigma_raw = load_dwi_and_average(
        [(args.small_delta, args.Delta, str(args.dwi), str(args.bval), str(args.bvec))],
        str(args.mask), lib_b_values=list(lib_b_values), default_small_delta=args.small_delta,
        rician_correct=bool(requested["rician_correct"]), noise_sigma=requested.get("noise_sigma"),
        noise_bg_dilate_iters=int(requested["noise_bg_dilate_iters"]), avg_s0=bool(requested["avg_s0"]),
        return_raw=True, return_metadata=True, direction_scheme=requested.get("direction_scheme"))
    recorded_sigma = float(baseline_map["sidecar"]["fit_configuration"]["noise_sigma"])
    if not np.isclose(float(sigma_raw), recorded_sigma, rtol=1e-9):
        raise RuntimeError(f"loader sigma {float(sigma_raw):.6g} differs from the baseline fit's recorded "
                           f"{recorded_sigma:.6g}; the measured signal would not be the one the fits used")
    if measured.shape[0] != int(np.count_nonzero(mask)):
        raise RuntimeError("measured rows do not match the mask voxel count")
    b_measured = np.asarray([t[2] for t in fit_triples], dtype=float)
    s0_voxel = np.asarray(extras["s0"], dtype=float)
    s0_voxel = s0_voxel[:, 0] if s0_voxel.ndim == 2 else s0_voxel
    s0_median = float(extras["s0_median"])
    shells_sidecar = baseline_map["sidecar"]["fit_configuration"]["direction_scheme"]["shells"]
    shells = {float(s["b_s_mm2"]): int(s["n_volumes"]) for s in shells_sidecar}
    if sorted(shells) != sorted(b_measured.tolist()):
        raise RuntimeError("sidecar shell list and measured columns disagree")
    bvals_file = np.loadtxt(args.bval).ravel()
    n_b0_volumes = int(np.count_nonzero(bvals_file < B0_THRESHOLD))
    sigma_norm = float(sigma_raw) / s0_median
    print(f"[{time.time()-started:6.1f}s] measured {measured.shape}, sigma={sigma_raw:.2f}, "
          f"S0_median={s0_median:.1f}, b0 volumes={n_b0_volumes}", flush=True)

    # ---- Fisher geometry at this acquisition ------------------------------------
    vectors = transposed_view(args.cache_dir, "vectors")
    variance = transposed_view(args.cache_dir, "signal_variance")
    n_ensembles = int(np.load(args.cache_dir / "ensemble_means_subset.npy", mmap_mode="r").shape[1])
    priors = {"known_amplitude": np.inf,
              "this_scan_reference": amplitude_prior_precision(float(n_b0_volumes), sigma_norm),
              "unknown_amplitude": 0.0}
    tissue, amplitude, _ = acquisition_fisher(table, vectors, variance, n_ensembles, pair=pair, n_b=n_b,
                                              b_values=lib_b_values, shells=shells, sigma_norm=sigma_norm)
    layer = fisher_layer(tissue, amplitude, kio_ref, priors)
    floor_arm = arms["trust_floor_column_map"]
    column_layer = None
    if floor_arm is not None and floor_arm["trust_floor_mask"]:
        dropped = {float(c["b_s_mm2"]) for c in floor_arm["trust_floor_mask"].get("dropped_columns", [])}
        kept = {b: n for b, n in shells.items() if b not in dropped}
        t2, a2, _ = acquisition_fisher(table, vectors, variance, n_ensembles, pair=pair, n_b=n_b,
                                       b_values=lib_b_values, shells=kept, sigma_norm=sigma_norm)
        column_layer = fisher_layer(t2, a2, kio_ref, {"known_amplitude": np.inf})
    known = layer["known_amplitude"]
    print(f"[{time.time()-started:6.1f}s] acquisition Fisher: PD fraction "
          f"{np.mean(known['positive']):.4f}", flush=True)

    phase3_kappa = None
    if args.phase3_run_dir is not None:
        stored_nodes = np.load(args.phase3_run_dir / "evaluation_nodes.npy")
        if not np.array_equal(stored_nodes, np.asarray(table["nodes"], dtype=stored_nodes.dtype)):
            raise RuntimeError("Phase-3 evaluation nodes are not the node table this run builds")
        phase3_kappa = np.load(args.phase3_run_dir / "maps_full_stored_domain.kappa.npy")

    # ---- joins -------------------------------------------------------------------
    joins = {name: node_rows(table, arm["rho"], arm["V"], arm["kio"])
             for name, arm in arms.items() if arm is not None}
    rows_map = joins["baseline_map"]["rows"]
    blow_map = baseline_map["fitted"] & (baseline_map["V"] > THESIS_VOLUME_CUTOFF_PL)
    blow_bayes = baseline_bayes["fitted"] & (baseline_bayes["V"] > THESIS_VOLUME_CUTOFF_PL)
    adc_all = apparent_diffusion_coefficient(measured, b_measured)
    adc_low = apparent_diffusion_coefficient(measured, b_measured, b_max=1000.0)
    expected = expected_noise_residual(float(sigma_raw), s0_voxel, np.asarray([shells[b] for b in b_measured]))

    report = {
        "schema": "madi-fisher-phase4-v1",
        "inputs": {"fit_root": str(args.fit_root), "dwi": str(args.dwi), "mask": str(args.mask),
                   "mask_sha256": mask_sha, "artifact": str(args.artifact), "phase1": str(args.phase1),
                   "phase3_run_dir": None if args.phase3_run_dir is None else str(args.phase3_run_dir)},
        "column_domain": domain.as_dict(),
        "acquisition": {"delta_ms": args.small_delta, "Delta_ms": args.Delta,
                        "shells_b_to_volumes": {str(k): v for k, v in shells.items()},
                        "b0_volumes": n_b0_volumes, "rician_corrected": bool(requested["rician_correct"]),
                        "noise_bg_dilate_iters": int(requested["noise_bg_dilate_iters"]),
                        "sigma_raw": float(sigma_raw), "s0_median": s0_median,
                        "sigma_normalized": sigma_norm, "snr_b0": 1.0 / sigma_norm,
                        "voxels": int(measured.shape[0]), "vi_band": [vi_min, vi_max]},
        "noise_model_note": ("measured noise, not the pre-registered SNR-50 TE model: this phase explains a "
                             "specific measured acquisition. At one timing pair the TE factor is common to every "
                             "column, so kappa, angles, condition numbers and defects are identical under either "
                             "model; only absolute bounds carry the measured scale."),
        "arms": {name: (None if arm is None else {
            "method": arm["method"], "s0_mode": arm["s0_mode"], "features": len(arm["fit_triples"]),
            "trust_floor_mask": arm["trust_floor_mask"], **blow_up_block(arm)})
            for name, arm in arms.items()},
        "join_integrity": {name: {
            "voxels_with_a_canonical_node": int(np.count_nonzero(j["has_node"])),
            "voxels_whose_node_has_a_fisher_matrix": int(np.count_nonzero(j["has_fisher"])),
            "max_log_rho_offset_over_half_step": float(np.nanmax(j["log_rho_offset"]) / j["half_log_step_rho"]),
            "max_log_V_offset_over_half_step": float(np.nanmax(j["log_V_offset"]) / j["half_log_step_V"])}
            for name, j in joins.items()},
        "acquisition_fisher": fisher_layer_summary(layer),
        "acquisition_fisher_trust_floor_columns": None if column_layer is None else fisher_layer_summary(column_layer),
    }

    # ---- characterization: the two branches --------------------------------------
    report["characterization"] = {
        "definition": (f"blow-up = fitted V > {THESIS_VOLUME_CUTOFF_PL:g} pL, the Jackson-thesis cutoff; every "
                       "contrast is also given across the cutoff sweep"),
        "map": {"adc_all_shells": stratified_comparison(adc_all, blow_map, label="ADC um2/ms, all shells"),
                "adc_b_le_1000": stratified_comparison(adc_low, blow_map, label="ADC um2/ms, b <= 1000"),
                "adc_nonphysical": {
                    "note": ("a Gaussian-diffusion ADC is negative only where S/S0 exceeds 1 and exceeds free water "
                             "only where signal falls faster than free diffusion; both mark signal the tissue model "
                             "does not describe. Counted, never excluded: every statistic uses all voxels."),
                    **{group: {"adc_b_le_1000_below_zero": float(np.mean(adc_low[g] < 0.0)),
                               "adc_b_le_1000_above_free_water": float(np.mean(adc_low[g] > FREE_WATER_ADC_UM2_PER_MS)),
                               "adc_b_le_1000_nonfinite": float(np.mean(~np.isfinite(adc_low[g])))}
                       for group, g in (("blow_up", blow_map), ("rest", baseline_map["fitted"] & ~blow_map))}},
                "volume_over_reachable_ceiling": _summary(
                    (baseline_map["V"] / reachable_volume_ceiling(baseline_map["rho"], vi_max))[blow_map]),
                "band_position": stratified_comparison(
                    np.where(baseline_map["fitted"], band_position(baseline_map["rho"], baseline_map["V"],
                                                                   vi_min, vi_max), np.nan),
                    blow_map, label="log v_i position across the mask band, 0 = lower edge, 1 = upper")},
        "bayes": {"adc_all_shells": stratified_comparison(adc_all, blow_bayes, label="ADC um2/ms, all shells"),
                  "adc_b_le_1000": stratified_comparison(adc_low, blow_bayes, label="ADC um2/ms, b <= 1000")},
    }

    # ---- 4.1 H4 and 4.2 H3 ---------------------------------------------------------
    report["H4_amplitude"] = {"empirical": {}, "fisher": {
        "regimes": fisher_layer_summary(layer),
        "prior_precision_over_F_s0s0": _summary(priors["this_scan_reference"] / amplitude[:, 3]),
        "thesis_mechanism_note": (f"this acquisition has {n_b0_volumes} true b = 0 volumes, so the Jackson-thesis "
                                  "mechanism -- a biased reference from normalizing by a b = 50 shell -- is absent "
                                  "here by construction. What is tested is the variance side: whether letting S0 "
                                  "float changes the pathology, and what the amplitude costs in bound and direction."),
    }}
    report["H3_trust_floor"] = {"empirical": {}}
    for method in ("map", "bayes"):
        base = arms[f"baseline_{method}"]
        for name, key, block in ((f"fit_s0_{method}", "--fit-s0", "H4_amplitude"),
                                 (f"trust_floor_column_{method}", "trust floor, column mode", "H3_trust_floor"),
                                 (f"trust_floor_candidate_{method}", "trust floor, candidate mode", "H3_trust_floor")):
            if arms[name] is None:
                report[block]["empirical"][name] = {"unavailable": "arm not present under --fit-root"}
                continue
            report[block]["empirical"][name] = condition_contrast(base, arms[name], key)

    # ---- 4.3 H1 versus H2 ------------------------------------------------------------
    gof_map = goodness_of_fit_ratio(np.where(baseline_map["fitted"], baseline_map["residual"], np.nan), expected)
    gof_bayes = goodness_of_fit_ratio(np.where(baseline_bayes["fitted"], baseline_bayes["residual"], np.nan), expected)
    report["H1_vs_H2_residuals"] = {
        "discriminator": ("goodness-of-fit ratio = residual / expected noise residual with the voxel's own S0. "
                          "H1 predicts pathological voxels fit about as well as noise allows (probability of "
                          "superiority near or below 0.5); H2 predicts they sit far above it."),
        "map": {"goodness_of_fit_ratio": stratified_comparison(gof_map, blow_map, label="residual / noise floor"),
                "raw_residual": stratified_comparison(np.where(baseline_map["fitted"], baseline_map["residual"],
                                                               np.nan), blow_map, label="raw residual")},
        "bayes": {"goodness_of_fit_ratio": stratified_comparison(gof_bayes, blow_bayes, label="residual / noise floor"),
                  "raw_residual": stratified_comparison(np.where(baseline_bayes["fitted"],
                                                                 baseline_bayes["residual"], np.nan),
                                                        blow_bayes, label="raw residual")},
    }

    # ---- 4.4 H1: the overlay and the ridge -------------------------------------------
    has_fisher = joins["baseline_map"]["has_fisher"]
    overlay = {
        "voxels_whose_node_has_no_fisher_matrix": {
            "blow_up_fraction": float(np.mean(~has_fisher[blow_map])) if blow_map.any() else None,
            "rest_fraction": float(np.mean(~has_fisher[baseline_map["fitted"] & ~blow_map]))},
        "acquisition_kappa_V": stratified_comparison(at_rows(known["kappa"][:, 1], rows_map), blow_map,
                                                     label="kappa_V at this acquisition"),
        "acquisition_log10_condition_number": stratified_comparison(
            np.log10(at_rows(known["condition_number"], rows_map)), blow_map, label="log10 condition number"),
        "acquisition_profiled_sloppy_angle_deg": stratified_comparison(
            at_rows(known["profiled_sloppy_angle_deg"], rows_map), blow_map, label="profiled sloppy angle"),
        "acquisition_node_identifiable": {
            "blow_up_fraction": float(np.mean(at_rows(known["positive"].astype(float), rows_map)[blow_map] == 1)),
            "rest_fraction": float(np.mean(at_rows(known["positive"].astype(float), rows_map)[
                baseline_map["fitted"] & ~blow_map] == 1))},
    }
    if phase3_kappa is not None:
        overlay["phase3_model_layer_kappa_V"] = stratified_comparison(
            at_rows(phase3_kappa[:, 1], rows_map), blow_map, label="Phase-3 full_stored_domain kappa_V")
    # Why a voxel's node carries no Fisher matrix, with the two causes separated,
    # because they mean different things: a k_io end node is the fit railing on
    # the exchange grid, while a band-edge (rho, V) pair is the mask boundary the
    # plan names as the place this pathology is expected to live (section 7).
    join_map = joins["baseline_map"]
    nodes = np.asarray(table["nodes"], dtype=int)
    interior_kio = np.zeros(int(max(nodes[:, 2].max(), join_map["k_io_index"].max())) + 1, dtype=bool)
    interior_kio[np.unique(nodes[:, 2])] = True
    interior_pair = np.zeros((int(max(nodes[:, 0].max(), join_map["rho_index"].max())) + 1,
                              int(max(nodes[:, 1].max(), join_map["V_index"].max())) + 1), dtype=bool)
    interior_pair[nodes[:, 0], nodes[:, 1]] = True
    valid_node = join_map["has_node"]
    kio_end = np.zeros(valid_node.shape, dtype=bool)
    band_edge = np.zeros(valid_node.shape, dtype=bool)
    kio_end[valid_node] = ~interior_kio[join_map["k_io_index"][valid_node]]
    band_edge[valid_node] = ~interior_pair[join_map["rho_index"][valid_node], join_map["V_index"][valid_node]]
    rest = baseline_map["fitted"] & ~blow_map

    def _reasons(group: np.ndarray) -> dict:
        if not group.any():
            return {"voxels": 0}
        return {"voxels": int(np.count_nonzero(group)),
                "k_io_end_node_only": float(np.mean((kio_end & ~band_edge)[group])),
                "rho_V_band_edge_only": float(np.mean((band_edge & ~kio_end)[group])),
                "both": float(np.mean((kio_end & band_edge)[group])),
                "interior_node": float(np.mean((~kio_end & ~band_edge & valid_node)[group]))}

    overlay["why_no_fisher_matrix"] = {
        "definition": ("a node has no Fisher matrix when its k_io index is a grid end (no k_io stencil) or its "
                       "(rho, V) pair has no interior rho and V stencil (the mask-band edge)"),
        "blow_up": _reasons(blow_map), "rest": _reasons(rest)}
    kio_grid = canonical_grid()[2]
    railed = np.isclose(baseline_map["kio"], kio_grid.min()) | np.isclose(baseline_map["kio"], kio_grid.max())
    overlay["k_io_at_grid_ends_fraction"] = {"blow_up": float(np.mean(railed[blow_map])),
                                             "rest": float(np.mean(railed[rest]))}
    overlay["fitted_rho_cells_per_uL"] = stratified_comparison(
        np.where(baseline_map["fitted"], baseline_map["rho"], np.nan), blow_map, label="MAP rho")
    with np.errstate(divide="ignore", invalid="ignore"):
        overlay["fitted_v_i"] = stratified_comparison(
            np.where(baseline_map["fitted"], baseline_map["rho"] * baseline_map["V"] * 1e-6, np.nan),
            blow_map, label="MAP v_i")
    report["H1_overlay"] = overlay
    report["H1_ridge_displacement"] = {
        "map_to_bayes_posterior_mean": ridge_displacement(baseline_map, baseline_bayes,
                                                          "MAP mode -> Bayes posterior mean, same data"),
    }
    for name in ("fit_s0_map", "trust_floor_column_map", "trust_floor_candidate_map"):
        if arms[name] is not None:
            report["H1_ridge_displacement"][f"baseline_map_to_{name}"] = ridge_displacement(
                baseline_map, arms[name], f"baseline MAP -> {name}")

    # ---- 4.5 the replacement ---------------------------------------------------------
    scale = np.where(s0_voxel > 0, s0_median / s0_voxel, np.nan)   # F scales as S0^2
    log_vi_bound = at_rows(known["log_vi_bound"], rows_map) * scale
    log_vi_defect = at_rows(known["log_vi_defect"], rows_map)
    kappa_voxel = at_rows(known["kappa"], rows_map)
    gated = kappa_gated_report(baseline_map["rho"], baseline_map["V"], kappa_voxel, log_vi_bound, kappa_threshold)
    fitted = baseline_map["fitted"]
    coverage = []
    for tolerance in DEFECT_TOLERANCES:
        reportable = fitted & np.isfinite(log_vi_bound) & (log_vi_defect <= tolerance)
        coverage.append({"defect_tolerance": tolerance,
                         "v_i_reportable_fraction_all": float(np.mean(reportable[fitted])),
                         "v_i_reportable_fraction_blow_up": float(np.mean(reportable[blow_map])) if blow_map.any() else None,
                         "v_i_bound_median": float(np.nanmedian(log_vi_bound[reportable])) if reportable.any() else None})
    bayes_rows = joins["baseline_bayes"]["rows"]
    bayes_scale = scale
    with np.errstate(divide="ignore", invalid="ignore"):
        bayes_vi = baseline_bayes["rho"] * baseline_bayes["V"] * 1e-6
    report["item_4_5_replacement"] = {
        "decision_pending": ("the v_i defect tolerance is NOT pre-registered and changes coverage materially; "
                             "every tolerance is reported and none is nominated"),
        "kappa_threshold_preregistered": kappa_threshold,
        "coverage_by_defect_tolerance": coverage,
        "rho_reportable_fraction": float(np.mean(gated.report_rho[fitted])),
        "V_reportable_fraction": float(np.mean(gated.report_volume[fitted])),
        "both_reportable_fraction": float(np.mean(gated.report_both[fitted])),
        "blow_up_voxels": {
            "count": int(np.count_nonzero(blow_map)),
            "V_reportable_fraction": float(np.mean(gated.report_volume[blow_map])) if blow_map.any() else None,
            "log_vi_defect": _summary(log_vi_defect[blow_map]),
            "log_vi_bound": _summary(log_vi_bound[blow_map])},
        "posterior_flag_validation": {
            "V": validate_quality_flag(baseline_bayes["V_std"] / baseline_bayes["V"],
                                       at_rows(known["crlb"][:, 1], bayes_rows) * bayes_scale),
            "rho": validate_quality_flag(baseline_bayes["rho_std"] / baseline_bayes["rho"],
                                         at_rows(known["crlb"][:, 0], bayes_rows) * bayes_scale),
            "note": ("fractional posterior SD against the per-voxel CRLB on the matching log parameter at the "
                     "voxel's nearest node. The rank correlation carries the claim; the ratio says how the "
                     "bounded candidate set caps the posterior where the CRLB is enormous."),
        },
        "bayes_v_i_product_of_means": {
            "definition": "the shipped vi_map for a Bayes fit is <rho><V>*1e-6, a product of posterior means",
            "fraction_outside_mask_band": float(np.mean(((bayes_vi < vi_min) | (bayes_vi > vi_max))[baseline_bayes["fitted"]])),
            "v_i": _summary(bayes_vi[baseline_bayes["fitted"]]),
        },
    }

    # ---- outputs -----------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.output_dir / "maps"
    maps_dir.mkdir(exist_ok=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        output_maps = {
            "adc_all_shells": adc_all, "adc_b_le_1000": adc_low,
            "blow_up_baseline_map": blow_map.astype(float), "blow_up_baseline_bayes": blow_bayes.astype(float),
            "gof_ratio_baseline_map": gof_map, "gof_ratio_baseline_bayes": gof_bayes,
            "node_has_fisher_matrix": has_fisher.astype(float),
            "kappa_V_acquisition": at_rows(known["kappa"][:, 1], rows_map),
            "log10_condition_acquisition": np.log10(at_rows(known["condition_number"], rows_map)),
            "log_vi_baseline_map": gated.log_vi, "log_vi_bound": log_vi_bound, "log_vi_defect": log_vi_defect,
            "report_rho": gated.report_rho.astype(float), "report_V": gated.report_volume.astype(float),
            # The 4.5 flag validation, as voxel arrays, so the figure layer reads rather than recomputes.
            "crlb_log_V_at_bayes_node": at_rows(known["crlb"][:, 1], bayes_rows) * bayes_scale,
            "crlb_log_rho_at_bayes_node": at_rows(known["crlb"][:, 0], bayes_rows) * bayes_scale,
            "posterior_fractional_sd_V": baseline_bayes["V_std"] / baseline_bayes["V"],
            "posterior_fractional_sd_rho": baseline_bayes["rho_std"] / baseline_bayes["rho"],
        }
    # Per-voxel ridge-displacement angles, one map per perturbation, NaN where
    # either fit did not reach the voxel or the estimate did not move.
    perturbations = {"map_to_bayes": (baseline_map, baseline_bayes)}
    for name in ("fit_s0_map", "trust_floor_column_map", "trust_floor_candidate_map"):
        if arms[name] is not None:
            perturbations[f"baseline_map_to_{name}"] = (baseline_map, arms[name])
    for label, (from_arm, to_arm) in perturbations.items():
        both = from_arm["fitted"] & to_arm["fitted"]
        step = log_displacement(from_arm["rho"], from_arm["V"], to_arm["rho"], to_arm["V"])
        output_maps[f"displacement_angle_{label}"] = np.where(both, step["angle_to_hyperbola_deg"], np.nan)
        # The geometric null on exactly the subset and seed ridge_displacement uses,
        # so the figure and the report quote the same distribution.
        index = np.flatnonzero(both)
        null = permuted_displacement_null(from_arm["rho"][index], from_arm["V"][index],
                                          to_arm["rho"][index], to_arm["V"][index], seed=RIDGE_NULL_SEED)
        null_map = np.full(both.shape, np.nan)
        null_map[index] = null["angle_to_hyperbola_deg"]
        output_maps[f"displacement_angle_null_{label}"] = null_map
    for stem, values in output_maps.items():
        values = np.asarray(values, dtype=float)
        # Infinities are written as NaN: a log of a zero condition number or a
        # ratio over a zero denominator is "undefined here", not a huge value.
        save_map(np.where(np.isfinite(values), values, np.nan), mask_idx, shape, affine,
                 str(maps_dir / f"{stem}.nii.gz"))
    report["runtime_seconds"] = time.time() - started
    (args.output_dir / "phase4_report.json").write_text(json.dumps(report, indent=2, default=float),
                                                        encoding="utf-8")
    print(f"[{time.time()-started:6.1f}s] wrote {args.output_dir / 'phase4_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
