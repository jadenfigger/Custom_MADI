"""
MADI library: build a lookup table of simulated signals indexed by
(k_io, ρ, V), then match experimental voxel data to estimate parameters.

(δ, Δ, b)-UNIVERSAL LIBRARY
----------------------------
Each `LibraryEntry.vector` is now a flattened S[δ,Δ,b] block (row-major:
pair-major then b, matching `madi.signal.ColumnGrid`) instead of a
fixed-δ, multi-Δ vector. One MC walk per (ρ,V) ensemble (reused across the
kio sweep, per `build_lookup_table`'s ensemble-reuse — see
`walker_gpu.run_simulation_multi_kio_reduced`) fills the WHOLE (δ,Δ,b)
grid, so building the library no longer requires re-simulating per δ/Δ.

Matching is NEAREST-COLUMN, never interpolated: a measured (δ,Δ,b) that
doesn't exactly land on the library's stored grid is matched to its nearest
stored column and the resulting mismatch is accepted as error (by design —
see `_grid_columns`).

Legacy fixed-δ `.npz` libraries (built before this refactor) can still be
read — `load_library_meta` detects the old format and synthesizes an
equivalent `delta_pairs` list so downstream matching code is format-
agnostic.

match_voxels_batch_fits0()
    Same library/candidate filtering as match_voxels_batch, but operates
    on UN-NORMALIZED measured signals and treats S0 as a free linear
    parameter per voxel.  For each candidate library entry r (a vector
    of simulated S/S0 ratios), the L2-optimal S0 is

        S0* = (M . r) / (r . r)

    and the residual is

        ||M - S0* r||^2 = ||M||^2 - (M.r)^2 / (r.r)

    This is fully vectorizable and adds only one cheap inner-product
    per (voxel, entry) compared with the fixed-S0 matcher.

    The fitted S0 is returned alongside the parameter maps.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, List, Tuple

import numpy as np

from collections import defaultdict

from .config       import (
    SimConfig, DELTA_SMALL, DELTAS_BIG, BVALS_UNIQUE,
    ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS,
    ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2,
)
from . import signal as sig
from . import fitters_gpu


MADI_LIBRARY_SCHEMA_V5 = "madi-library-v5"
MADI_LIBRARY_SCHEMA_V4 = "madi-library-v4"
MADI_LIBRARY_ENTRY_SCHEMA_V5 = "madi-library-entry-v5"

# The on-disk diagnostic column order is intentionally independent of a
# caller's main storage-grid order: declared pair-major, then b-major.
ENSEMBLE_MEAN_SUBSET_N_COLUMNS = (
    len(ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS)
    * len(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2)
)


# ---------------------------------------------------------------------------
# Library entry
# ---------------------------------------------------------------------------

@dataclass
class LibraryEntry:
    kio:    float
    rho:    float
    V:      float
    vector: np.ndarray   # flat S[δ,Δ,b].ravel() — pair-major then b
    # v5 per-column Monte-Carlo diagnostics.  They remain optional so v4 and
    # legacy artifacts continue to load, but every new builder-produced
    # artifact supplies all three.
    signal_imag: np.ndarray | None = None
    signal_variance: np.ndarray | None = None
    ensemble_means_subset: np.ndarray | None = None
    # Everything below is optional so legacy files and small test fixtures
    # remain readable.  New builds always populate these fields.
    vi: float | None = None
    weight: float | None = None
    kio_nominal: float | None = None
    rho_nominal: float | None = None
    V_nominal: float | None = None
    pp: float | None = None
    kio_analytic_eq5: float | None = None
    is_free_water: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def realised_vi(self) -> float:
        if self.vi is not None and np.isfinite(self.vi):
            return float(self.vi)
        return float(self.rho) * float(self.V) * 1e-6


DEFAULT_KIOS = [2, 5, 8, 12, 18, 25, 35, 50, 75, 100]
DEFAULT_RHOS = [100_000, 200_000, 400_000, 600_000, 800_000]
DEFAULT_VS   = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]


@dataclass(frozen=True)
class LogGridDefinition:
    """Masked log-coordinate cellular grid plus one discrete free-water atom.

    ``weights`` are midpoint quadrature elements for a prior density uniform
    in physical ``(rho, V, k_io)`` coordinates.  With x=log(rho), y=log(V),
    d rho dV dk = rho*V dx dy dk, hence the analytic rho*V Jacobian below.
    The k=0 slice uses its one-sided piecewise-cell width.  Free water is a
    separately declared point-mass component and receives
    ``free_water_weight`` rather than pretending to occupy a volume in the
    cellular coordinate system.
    """

    kios: np.ndarray
    rhos: np.ndarray
    Vs: np.ndarray
    vi_min: float = 0.40
    vi_max: float = 0.99
    free_water_weight: float = 1.0

    def triplets_and_weights(self) -> tuple[list[tuple[float, float, float]], dict]:
        if np.any(self.rhos <= 0.0) or np.any(self.Vs <= 0.0):
            raise ValueError("log-coordinate rho and V grid values must be positive")
        if np.any(self.kios < 0.0):
            raise ValueError("k_io grid values must be non-negative")
        lx = np.log(self.rhos)
        ly = np.log(self.Vs)
        # The declared extrema are the integration-domain bounds, not
        # midpoint centres outside the requested box.  Endpoint cells are
        # therefore half-width in transformed coordinates.  Extending them
        # by half a step would silently give posterior mass to rho/V/k_io
        # values outside the stated remediation grid.
        dx = _cell_widths(lx, lower_bound=float(lx[0]), upper_bound=float(lx[-1]))
        dy = _cell_widths(ly, lower_bound=float(ly[0]), upper_bound=float(ly[-1]))
        dk = _cell_widths(
            self.kios, lower_bound=0.0, upper_bound=float(self.kios[-1]),
        )
        triplets: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
        weights: dict = {_entry_key(0.0, 0.0, 0.0): float(self.free_water_weight)}
        for ik, k in enumerate(self.kios):
            for ir, rho in enumerate(self.rhos):
                for iv, volume in enumerate(self.Vs):
                    vi = float(rho * volume * 1e-6)
                    if not (self.vi_min <= vi <= self.vi_max):
                        continue
                    item = (float(k), float(rho), float(volume))
                    triplets.append(item)
                    weights[_entry_key(*item)] = float(rho * volume * dx[ir] * dy[iv] * dk[ik])
        return triplets, weights

    def metadata(self) -> dict:
        return {
            "coordinate_transform": {
                "rho": "uniform_log",
                "V": "uniform_log",
                "kio": "piecewise_uniform_linear",
            },
            "vi_min": float(self.vi_min),
            "vi_max": float(self.vi_max),
            "rho_values": self.rhos.tolist(),
            "V_values": self.Vs.tolist(),
            "kio_values": self.kios.tolist(),
            "free_water_weight": float(self.free_water_weight),
            "weight_expression": "rho * V * dlogrho * dlogV * dkio",
        }


def _cell_widths(
    values: np.ndarray,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> np.ndarray:
    """Midpoint-cell widths for a strictly increasing 1-D grid."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) < 2 or np.any(np.diff(arr) <= 0.0):
        raise ValueError("grid axis must be strictly increasing with at least two values")
    edges = np.empty(len(arr) + 1, dtype=float)
    edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
    edges[0] = (float(lower_bound) if lower_bound is not None
                else arr[0] - 0.5 * (arr[1] - arr[0]))
    edges[-1] = (float(upper_bound) if upper_bound is not None
                 else arr[-1] + 0.5 * (arr[-1] - arr[-2]))
    if edges[0] > arr[0] or edges[-1] < arr[-1]:
        raise ValueError("quadrature bounds must enclose every grid node")
    return np.diff(edges)


def make_remediation_log_grid(
    *,
    n_rho: int = 64,
    n_V: int = 64,
    kios: Optional[np.ndarray] = None,
    rho_min: float = 1.0e4,
    rho_max: float = 1.0e7,
    V_min: float = 0.01,
    V_max: float = 200.0,
    vi_min: float = 0.40,
    vi_max: float = 0.99,
) -> LogGridDefinition:
    """Production-grid design mandated by remediation P0-E (not a build)."""
    if not (0.0 < rho_min < rho_max):
        raise ValueError("remediation rho bounds must satisfy 0 < min < max")
    if not (0.0 < V_min < V_max):
        raise ValueError("remediation V bounds must satisfy 0 < min < max")
    if kios is None:
        kios = np.r_[0.0, np.arange(1.0, 31.0, 1.0), np.arange(35.0, 131.0, 5.0)]
    return LogGridDefinition(
        kios=np.asarray(kios, dtype=float),
        rhos=np.geomspace(float(rho_min), float(rho_max), int(n_rho)),
        Vs=np.geomspace(float(V_min), float(V_max), int(n_V)),
        vi_min=float(vi_min),
        vi_max=float(vi_max),
    )


def _entry_key(kio, rho, V):
    if not np.isfinite(kio) or (rho == 0.0 and V == 0.0):
        return ("free_water", 0.0, 0.0)
    return (round(float(kio), 4), round(float(rho), 1), round(float(V), 6))

def _existing_keys(library):
    return {
        _entry_key(
            e.kio_nominal if e.kio_nominal is not None else e.kio,
            e.rho_nominal if e.rho_nominal is not None else e.rho,
            e.V_nominal if e.V_nominal is not None else e.V,
        )
        for e in library
    }

# Hard physical ceiling: create_ensemble() raises for vi > 0.99, so no build
# may request an entry above it regardless of the caller's vi_max.
VI_HARD_MAX = 0.99


def _filter_valid(triplets, vi_min=0.0, vi_max=VI_HARD_MAX):
    hi = min(vi_max, VI_HARD_MAX)
    valid = []
    for k, r, v in triplets:
        # A genuine free-water atom is a discrete model component, not an
        # invalid rho*V product.  k_io is undefined physically; use NaN in
        # labels and preserve the nominal coordinate as zero in metadata.
        if r == 0.0 and v == 0.0:
            valid.append((k, r, v))
            continue
        vi = (r / 1e9) * (v * 1e3)
        if vi_min <= vi <= hi:
            valid.append((k, r, v))
    return valid


def _summarise_realised_geometry(per_ensemble: list[dict], cfg: SimConfig) -> dict:
    """Aggregate finite Ω_sim measurements without replacing Eq.-5 inputs.

    The requested process parameters determine alpha_star and the governing
    single-cell ``<A/V>``.  The values here describe only the finite realised
    geometries carried by library labels/provenance; none is fed back into
    permeability calibration (SI §S.IV).
    """
    if not per_ensemble:
        return {
            "vi": 0.0, "rho_per_uL": 0.0, "mean_volume_pL": 0.0,
            "governing_mean_A_over_V_um_inv": 0.0, "n_seeds_sim": 0,
        }
    n_cells = np.asarray([int(x["n_seeds_sim"]) for x in per_ensemble], dtype=float)
    vis = np.asarray([float(x["realised_vi"]) for x in per_ensemble], dtype=float)
    volumes_um3 = np.asarray([float(x["sim_side_um"]) ** 3 for x in per_ensemble])
    total_cells = float(n_cells.sum())
    total_volume = float(volumes_um3.sum())
    total_intracellular_volume = float(np.sum(vis * volumes_um3))
    # The aggregate labels satisfy rho*V*1e-6 == aggregate direct v_i.
    rho = total_cells / total_volume * 1e9 if total_volume else 0.0
    V_pL = total_intracellular_volume / total_cells / 1e3 if total_cells else 0.0
    volume_weights = volumes_um3 / max(total_volume, 1.0)
    def volume_weighted(name: str) -> float:
        return float(np.sum(volume_weights * np.asarray([float(x[name]) for x in per_ensemble])))
    return {
        "vi": float(total_intracellular_volume / total_volume) if total_volume else 0.0,
        "vi_between_ensemble_sd": float(np.std(vis, ddof=1)) if len(vis) > 1 else 0.0,
        "rho_per_uL": float(rho),
        "mean_volume_pL": float(V_pL),
        "mean_volume_um3": float(V_pL * 1e3),
        "governing_mean_A_over_V_um_inv": volume_weighted("governing_mean_A_over_V_um_inv"),
        "annulus_mean_um": volume_weighted("annulus_mean_um"),
        "annulus_std_um": volume_weighted("annulus_std_um"),
        "annulus_min_um": float(min(float(x["annulus_min_um"]) for x in per_ensemble)),
        "annulus_max_um": float(max(float(x["annulus_max_um"]) for x in per_ensemble)),
        "annulus_q05_um": volume_weighted("annulus_q05_um"),
        "annulus_q50_um": volume_weighted("annulus_q50_um"),
        "annulus_q95_um": volume_weighted("annulus_q95_um"),
        "n_seeds_sim": int(total_cells),
        "sim_side_um": [float(x["sim_side_um"]) for x in per_ensemble],
        "source_side_um": [float(x["source_side_um"]) for x in per_ensemble],
    }


def ensemble_mean_subset_column_indices(columns: "sig.ColumnGrid") -> np.ndarray:
    """Return full-vector indices for the declared v5 covariance subset.

    The returned indices are in the immutable schema order: the eight
    declared ``(delta, Delta)`` pairs in
    ``ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS``, each crossed with the complete
    declared 25-value b grid.  A v5 builder must have every one of these
    columns; accepting a partial diagnostic subset would make covariance
    comparisons across artifacts ambiguous.
    """
    expected_b = np.asarray(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2, dtype=float)
    actual_b = np.asarray(columns.b_values, dtype=float)
    if actual_b.shape != expected_b.shape or not np.array_equal(actual_b, expected_b):
        raise ValueError(
            "madi-library-v5 requires the declared 25-point b grid for the "
            "per-ensemble covariance diagnostic subset"
        )

    pair_to_index = {
        (float(delta), float(Delta)): index
        for index, (delta, Delta) in enumerate(columns.delta_pairs)
    }
    if len(pair_to_index) != len(columns.delta_pairs):
        raise ValueError("main storage grid contains duplicate (delta, Delta) pairs")
    missing = [pair for pair in ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS
               if pair not in pair_to_index]
    if missing:
        raise ValueError(
            "madi-library-v5 main storage grid omits declared ensemble-mean "
            f"diagnostic pairs: {missing!r}"
        )

    return np.asarray([
        pair_to_index[pair] * columns.n_b + b_index
        for pair in ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS
        for b_index in range(columns.n_b)
    ], dtype=np.int64)


def _v5_diagnostics_from_result(
    result: dict,
    columns: "sig.ColumnGrid",
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Derive the three v5 diagnostic payloads from one reduced entry.

    ``signal_variance`` is deliberately a sample variance across independent
    ensemble means, not a pooled walker/axis variance.  The three axes from
    one 3-D walk and all walkers in one geometry are correlated at the level
    relevant to the Fisher correction, so pooling them would understate the
    uncertainty that this field exists to preserve.
    """
    ensemble_real = result.get("ensemble_S")
    ensemble_imag = result.get("ensemble_S_imag")
    if ensemble_real is None or ensemble_imag is None:
        raise RuntimeError("v5 library build requires per-ensemble reduced signals")
    ensemble_real = np.asarray(ensemble_real, dtype=np.float64)
    ensemble_imag = np.asarray(ensemble_imag, dtype=np.float64)
    expected_shape = (int(cfg.n_ensembles), columns.n_pairs, columns.n_b)
    if ensemble_real.shape != expected_shape or ensemble_imag.shape != expected_shape:
        raise RuntimeError(
            "per-ensemble reduced signal shape does not match the configured "
            f"ensemble/grid contract: got real={ensemble_real.shape}, "
            f"imag={ensemble_imag.shape}, expected={expected_shape}"
        )
    if not (np.all(np.isfinite(ensemble_real)) and np.all(np.isfinite(ensemble_imag))):
        raise RuntimeError("per-ensemble reduced signals contain non-finite values")
    if int(cfg.n_ensembles) < 2:
        raise ValueError("madi-library-v5 requires at least two ensembles for sample variance")

    # W1: the collapsed imaginary signal already accumulated by the reducer.
    signal_imag = np.asarray(result["S_imag"], dtype=np.float64).reshape(
        columns.n_pairs, columns.n_b,
    )
    # W2: sample variance (ddof=1) of independently constructed ensemble
    # means.  The analysis consumer converts it to SE via sqrt(s^2 / n).
    signal_variance = np.var(ensemble_real, axis=0, ddof=1)
    if np.any(signal_variance < 0.0):
        raise RuntimeError("between-ensemble sample variance became negative")

    # W3: retain only the declared small subset, preserving e=0..E-1 as axis
    # zero within each entry.  The artifact adds an entry axis around this.
    subset_indices = ensemble_mean_subset_column_indices(columns)
    ensemble_subset = ensemble_real.reshape(int(cfg.n_ensembles), -1)[:, subset_indices]

    # The artifact intentionally does not retain a full imaginary variance
    # array.  Calculate the exact ensemble-level Student statistic while the
    # means are available and retain only its per-entry maximum in metadata;
    # this lets the validator test the stored S_imag symmetry check without
    # adding another full signal matrix.
    imag_mean = np.mean(ensemble_imag, axis=0)
    imag_se = np.std(ensemble_imag, axis=0, ddof=1) / np.sqrt(cfg.n_ensembles)
    abs_z = np.zeros_like(imag_mean)
    positive_se = imag_se > 0.0
    abs_z[positive_se] = np.abs(imag_mean[positive_se] / imag_se[positive_se])
    zero_se_nonzero = (~positive_se) & (np.abs(imag_mean) > 0.0)
    abs_z[zero_se_nonzero] = np.inf
    max_flat = int(np.argmax(abs_z))
    max_pair_index, max_b_index = np.unravel_index(max_flat, abs_z.shape)
    imaginary_check = {
        "method": (
            "absolute Student statistic of the mean of independent "
            "per-ensemble imaginary signals (ddof=1)"
        ),
        "n_ensembles": int(cfg.n_ensembles),
        "degrees_of_freedom": int(cfg.n_ensembles - 1),
        "max_abs_standardized_deviation": float(abs_z.flat[max_flat]),
        "max_column": {
            "delta_ms": float(columns.delta_pairs[max_pair_index][0]),
            "Delta_ms": float(columns.delta_pairs[max_pair_index][1]),
            "b_s_mm2": float(columns.b_values[max_b_index]),
        },
        # Retaining the value and SE at the maximizing column lets the
        # artifact validator tie this compact standardized check back to the
        # stored float32 signal_imag array without storing all ensemble-level
        # imaginary means.
        "mean_imaginary_signal": float(imag_mean.flat[max_flat]),
        "standard_error": float(imag_se.flat[max_flat]),
        "zero_standard_error_nonzero_count": int(np.count_nonzero(zero_se_nonzero)),
    }

    return (
        np.asarray(signal_imag, dtype=np.float32).ravel(),
        np.asarray(signal_variance, dtype=np.float32).ravel(),
        np.asarray(ensemble_subset, dtype=np.float32),
        imaginary_check,
    )


# ---------------------------------------------------------------------------
# Core builder: works on an explicit list of (kio, rho, V) triplets
# ---------------------------------------------------------------------------

def build_library_from_triplets(
    triplets: List[Tuple[float, float, float]],
    cfg: SimConfig | None = None,
    save_path: Optional[str] = None,
    existing_library: Optional[list[LibraryEntry]] = None,
    seed: int = 0,
    vi_min: float = 0.0,
    vi_max: float = VI_HARD_MAX,
    entry_weights: Optional[Mapping[tuple, float]] = None,
    grid_metadata: Optional[dict] = None,
    verbose: bool = True,
) -> list[LibraryEntry]:
    """Build/extend library from an explicit list of (kio, rho, V) triplets.

    Skips any triplet already present in existing_library.

    `seed` is a BUILD-LEVEL constant, deliberately the SAME across every
    (ρ,V) group — geometry/walker RNG seeds are derived from
    (seed, ensemble_index[, kio]) only, never from (ρ,V), so that
    neighbouring (ρ,V) grid points share correlated random-number streams
    (common random numbers). This keeps future Fisher/CRLB finite
    differences w.r.t. ρ,V low-noise without needing a library rebuild.
    """
    if cfg is None:
        cfg = SimConfig()
    if int(cfg.n_ensembles) < 2:
        raise ValueError(
            "madi-library-v5 requires at least two independently constructed "
            "ensembles so that per-column sample variance is defined"
        )

    if existing_library is not None:
        library = list(existing_library)
        done = _existing_keys(library)
        if verbose:
            print(f"  Loaded {len(library)} existing entries")
    else:
        library = []
        done = set()

    if library:
        diagnostics_present = [
            entry.signal_imag is not None
            and entry.signal_variance is not None
            and entry.ensemble_means_subset is not None
            for entry in library
        ]
        if not all(diagnostics_present):
            raise ValueError(
                "cannot append v5 entries to a pre-v5 library: the existing "
                "artifact lacks required Monte-Carlo diagnostic arrays"
            )

    valid = _filter_valid(triplets, vi_min=vi_min, vi_max=vi_max)
    new_triplets = [t for t in valid if _entry_key(*t) not in done]

    if verbose:
        eff_hi = min(vi_max, VI_HARD_MAX)
        print(f"  Requested: {len(triplets)} triplets")
        print(f"  vi filter: keeping {vi_min:.2f} <= vi <= {eff_hi:.2f}")
        print(f"  Skipped: {len(triplets)-len(valid)} (outside vi range) "
              f"+ {len(valid)-len(new_triplets)} (already exist)")
        print(f"  New to compute: {len(new_triplets)}")

    if not new_triplets:
        if verbose:
            print("  Nothing new to compute!")
        if save_path:
            _save_library(
                library, save_path, cfg=cfg, grid_metadata=grid_metadata,
                build_seed=seed,
            )
        return library

    columns = sig.build_columns(cfg)
    if verbose:
        print(f"  (δ,Δ,b) grid: {columns.n_pairs} pairs × {columns.n_b} "
              f"b-values = {columns.n_pairs * columns.n_b} columns/entry")

    t0 = time.time()

    # -----------------------------------------------------------------
    # Group triplets by (rho, V).  Ensemble geometry depends ONLY on
    # (rho, V); kio only affects the membrane-crossing probability in
    # the walk kernel.  So we build each ensemble once and sweep all
    # kios for that (rho, V) on it — ~N_kios× speedup on the CPU
    # scipy Voronoi / HalfspaceIntersection cost, which dominates at
    # high ρ.
    # -----------------------------------------------------------------
    groups: dict = defaultdict(list)
    for kio, rho, V in new_triplets:
        groups[(rho, V)].append(kio)

    if verbose:
        print(f"  Grouped into {len(groups)} unique (ρ,V) pairs "
              f"(ensembles reused across kio values)")

    # Process groups in order of increasing cost proxy so a shard makes
    # progress on cheap entries first — makes checkpointing useful.
    sorted_groups = sorted(groups.items(), key=lambda kv: kv[0][0] * kv[0][1])

    entry_idx = 0
    for (rho, V), kios_for_group in sorted_groups:
        kios_for_group = sorted(kios_for_group)

        if verbose:
            print(f"  [(ρ,V)=({rho/1e3:.0f}k, {V:.2f})] "
                  f"{len(kios_for_group)} kio values "
                  f"→ {[f'{k:g}' for k in kios_for_group]}",
                  flush=True)

        tt = time.time()

        results = sig.compute_signals_multi_kio(
            rho, V, kios_for_group, cfg, columns=columns,
            seed=seed, verbose=False,
        )

        for kio in kios_for_group:
            result = results[kio]
            vec = sig.signals_to_flat(result)
            signal_imag, signal_variance, ensemble_subset, imaginary_check = (
                _v5_diagnostics_from_result(result, columns, cfg)
            )
            summary = _summarise_realised_geometry(result["geometry_stats"], cfg)
            is_free = rho == 0.0 and V == 0.0
            nominal_key = _entry_key(kio, rho, V)
            weight = None if entry_weights is None else entry_weights.get(nominal_key)
            entry_metadata = {
                "schema": MADI_LIBRARY_ENTRY_SCHEMA_V5,
                "nominal": {"kio": float(kio), "rho": float(rho), "V": float(V)},
                "realised_geometry": summary,
                "per_ensemble_geometry": result["geometry_stats"],
                "exchange": {
                    "kio_analytic_eq5_s_inv": float(result["kio_analytic_eq5"]),
                    "p_p_mean": float(result["pp"]),
                    "definition": (
                        "MADI I Eq. 5 using SI §S.IV untrimmed governing-process "
                        "arithmetic <A/V>; no residence-time proxy is used."
                    ),
                },
                "boundary": {
                    "mode": cfg.boundary_mode,
                    "n_escaped": int(result["n_escaped"]),
                    "n_walkers_total": int(result["n_eff"] // 3),
                    "fatal_on_escape": True,
                    "occupancy_fraction": np.asarray(result["occupancy_fraction"]).tolist(),
                },
                "weight": None if weight is None else float(weight),
                "imaginary_signal_check": imaginary_check,
            }
            library.append(LibraryEntry(
                # k_io is undefined for the free-water atom; preserve NaN
                # rather than pretending it is a zero-exchange cell.
                kio=(float("nan") if is_free else float(kio)),
                rho=(0.0 if is_free else summary["rho_per_uL"]),
                V=(0.0 if is_free else summary["mean_volume_pL"]),
                vector=vec,
                signal_imag=signal_imag,
                signal_variance=signal_variance,
                ensemble_means_subset=ensemble_subset,
                vi=(0.0 if is_free else summary["vi"]),
                weight=None if weight is None else float(weight),
                kio_nominal=float(kio),
                rho_nominal=float(rho),
                V_nominal=float(V),
                pp=float(result["pp"]),
                kio_analytic_eq5=float(result["kio_analytic_eq5"]),
                is_free_water=is_free,
                metadata=entry_metadata,
            ))
            entry_idx += 1

        dt = time.time() - tt
        if verbose:
            print(f"    → {len(kios_for_group)} entries in {dt:.1f}s "
                  f"({dt/len(kios_for_group):.1f}s/entry)  "
                  f"[{entry_idx}/{len(new_triplets)} done]",
                  flush=True)

        # Checkpoint after every (rho, V) group — cheap insurance
        # against SLURM preemption / walltime kills.
        if save_path:
            _save_library(library, save_path, cfg=cfg, columns=columns,
                          grid_metadata=grid_metadata, build_seed=seed)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nLibrary: {len(library)} entries total "
              f"({len(new_triplets)} new in {elapsed:.0f}s)")

    if save_path:
        _save_library(library, save_path, cfg=cfg, columns=columns,
                      grid_metadata=grid_metadata, build_seed=seed)
        if verbose:
            print(f"Saved to {save_path}")

    return library


# ---------------------------------------------------------------------------
# Convenience: build from full cross-product grid
# ---------------------------------------------------------------------------

def build_library(
    kios=None, rhos=None, Vs=None,
    cfg=None, save_path=None, existing_library=None, seed=0,
    vi_min=0.0, vi_max=VI_HARD_MAX, entry_weights=None, grid_metadata=None,
    verbose=True,
) -> list[LibraryEntry]:
    """Build/extend library from kio × rho × V grid (full cross-product)."""
    if kios is None: kios = DEFAULT_KIOS
    if rhos is None: rhos = DEFAULT_RHOS
    if Vs is None:   Vs = DEFAULT_VS

    triplets = [(k, r, v) for k in kios for r in rhos for v in Vs]
    if verbose:
        print(f"  Grid: {len(kios)} kio × {len(rhos)} rho × {len(Vs)} V "
              f"= {len(triplets)} combos")

    return build_library_from_triplets(
        triplets, cfg=cfg, save_path=save_path,
        existing_library=existing_library, seed=seed,
        vi_min=vi_min, vi_max=vi_max, entry_weights=entry_weights,
        grid_metadata=grid_metadata, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-serialise {type(value)!r}")


def _cfg_metadata(
    cfg: SimConfig,
    *,
    schema: str = MADI_LIBRARY_SCHEMA_V5,
    build_seed: int | None = None,
) -> dict:
    """Library-embedded physics/build configuration, with no hidden defaults."""
    metadata = {
        "schema": schema,
        "D0_um2_ms": cfg.D0,
        "ts_ms": cfg.ts,
        "T_max_ms": cfg.T_max_ms,
        "domain_sizing": "MADI I SI Eq. S8; per-entry W unless diagnostic L override",
        "diagnostic_domain_L_override_um": cfg.L,
        "source_fraction": cfg.source_fraction,
        "boundary_mode": cfg.boundary_mode,
        "kappa": cfg.kappa,
        "population_certification": {
            "equations": ["S10", "S11", "S12"],
            "initial_margin_cell_spacings": cfg.population_initial_margin_cell_spacings,
            "certification_spacing_um": cfg.population_certification_spacing_um,
            "max_expansions": cfg.population_max_expansions,
        },
        "geometry_reference": {
            "path": cfg.geometry_reference_path,
            "required_single_cells": cfg.geometry_reference_required_cells,
            "required_alpha_values": cfg.geometry_reference_required_alpha_values,
            "allow_uncertified": cfg.allow_uncertified_geometry_reference,
            "mean_estimator": "untrimmed_arithmetic_mean_A_over_V",
        },
        "geometry_validation_points": cfg.geometry_validation_points,
        "geometry_vi_tolerance": cfg.geometry_vi_tolerance,
        "walkers_per_ensemble": cfg.n_walkers,
        "ensembles_per_entry": cfg.n_ensembles,
        "axis_walks_per_entry": 3 * int(cfg.n_walkers) * int(cfg.n_ensembles),
        "build_seed": None if build_seed is None else int(build_seed),
        "phase_model": cfg.phase_model,
        "checkpoint_h_ms": cfg.h_ms,
        "common_random_numbers": "base_seed + ensemble_index; shared across rho,V,kio",
        "classifier": (
            "exact full-facet shifted-Voronoi contraction at every endpoint; "
            "SI Eq. S2, with the provable d1+2*alpha1 adaptive KD radius bound"
            if cfg.classifier_mode == "exact" else
            "exact full-facet shifted-Voronoi contraction with conservative "
            "two-tier proof cache; SI Eq. S2 fallback uses the provable "
            "d1+2*alpha1 adaptive KD radius bound"
        ),
        "classifier_cache": {
            "mode": cfg.classifier_mode,
            "equivalence_contract": (
                "cache hits are used only after a conservative proof; all other "
                "endpoints invoke the unchanged exact SI Eq. S2 classifier"
            ),
            "tier1": (
                "cumulative displacement from a fixed reference is strictly below "
                "a shifted-facet safety radius"
            ),
            "tier2": (
                "finite candidate superset radius d1_ref + 2*max(alpha) + "
                "2*delta_max; overflow disables the cache and falls back exact"
            ),
            "delta_max_um": float(cfg.classifier_cache_delta_max_um),
            "min_safe_radius_um": float(cfg.classifier_cache_min_safe_radius_um),
            "candidate_capacity": int(cfg.classifier_cache_candidate_capacity),
        },
        "kio_label": "MADI I Eq. 5 with governing-process untrimmed arithmetic <A/V>",
        "residence_time_measurement": "intentionally absent: SI §S.IV states k_io != 1/<tau_i>",
    }
    if schema == MADI_LIBRARY_SCHEMA_V5:
        metadata["uncertainty"] = {
            "signal_imag": {
                "array": "signal_imag",
                "dtype": "float32",
                "shape": ["entry", "column"],
                "definition": "collapsed mean(sin(phase)) over all axis-walks",
            },
            "between_ensemble_variance": {
                "array": "signal_variance",
                "dtype": "float32",
                "shape": ["entry", "column"],
                "definition": (
                    "sample variance (ddof=1) of per-ensemble mean real signals; "
                    "consumer SE is sqrt(signal_variance / n_ensembles)"
                ),
                "n_ensembles": int(cfg.n_ensembles),
            },
            "ensemble_means_subset": {
                "array": "ensemble_means_subset",
                "dtype": "float32",
                "shape": ["entry", "ensemble_index", "declared_column"],
                "declared_delta_Delta_pairs_ms": [
                    [float(delta), float(Delta)]
                    for delta, Delta in ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS
                ],
                "declared_b_values_s_mm2": (
                    np.asarray(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2, dtype=float).tolist()
                ),
                "column_order": "pair-major, then b-major within each pair",
            },
            "ensemble_index_ordering_contract": {
                "axis_position_in_ensemble_means_subset": 1,
                "index_values": "0..n_ensembles-1",
                "same_order_across_entries": True,
                "independent_of": ["rho", "V", "k_io"],
                "walk_seed_formula": (
                    "(build_seed + 104729 * ensemble_index) mod 2^31"
                ),
                "geometry_seed_formula": (
                    "(build_seed + 97003 * ensemble_index) mod 2^31"
                ),
                "purpose": (
                    "ensemble index e is the common-random-number partner "
                    "across entries"
                ),
            },
        }
    return metadata


def _v5_entry_arrays(
    lib: list[LibraryEntry],
    n_columns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Validate and stack the v5 diagnostic arrays, if every entry has them."""
    presence = [
        entry.signal_imag is not None
        and entry.signal_variance is not None
        and entry.ensemble_means_subset is not None
        for entry in lib
    ]
    if not any(presence):
        return None
    if not all(presence):
        raise ValueError(
            "library mixes entries with and without v5 Monte-Carlo diagnostics"
        )

    imag_rows: list[np.ndarray] = []
    variance_rows: list[np.ndarray] = []
    subset_rows: list[np.ndarray] = []
    n_ensembles: int | None = None
    for entry_index, entry in enumerate(lib):
        assert entry.signal_imag is not None
        assert entry.signal_variance is not None
        assert entry.ensemble_means_subset is not None
        imag = np.asarray(entry.signal_imag)
        variance = np.asarray(entry.signal_variance)
        subset = np.asarray(entry.ensemble_means_subset)
        if imag.shape != (n_columns,):
            raise ValueError(
                f"entry {entry_index} signal_imag shape {imag.shape} != ({n_columns},)"
            )
        if variance.shape != (n_columns,):
            raise ValueError(
                f"entry {entry_index} signal_variance shape {variance.shape} != ({n_columns},)"
            )
        if subset.ndim != 2 or subset.shape[1] != ENSEMBLE_MEAN_SUBSET_N_COLUMNS:
            raise ValueError(
                f"entry {entry_index} ensemble_means_subset shape {subset.shape} != "
                f"(n_ensembles, {ENSEMBLE_MEAN_SUBSET_N_COLUMNS})"
            )
        if n_ensembles is None:
            n_ensembles = int(subset.shape[0])
        elif subset.shape[0] != n_ensembles:
            raise ValueError("v5 entries disagree on the number of ensembles")
        if n_ensembles < 2:
            raise ValueError("v5 diagnostic arrays require at least two ensembles")
        if not (np.all(np.isfinite(imag)) and np.all(np.isfinite(variance))
                and np.all(np.isfinite(subset))):
            raise ValueError(f"entry {entry_index} has non-finite v5 diagnostic values")
        if np.any(variance < 0.0):
            raise ValueError(f"entry {entry_index} has negative between-ensemble variance")
        imag_rows.append(np.asarray(imag, dtype=np.float32))
        variance_rows.append(np.asarray(variance, dtype=np.float32))
        subset_rows.append(np.asarray(subset, dtype=np.float32))

    signal_imag = np.stack(imag_rows, axis=0)
    signal_variance = np.stack(variance_rows, axis=0)
    ensemble_means_subset = np.stack(subset_rows, axis=0)
    if not (np.all(np.isfinite(signal_imag)) and np.all(np.isfinite(signal_variance))
            and np.all(np.isfinite(ensemble_means_subset))):
        raise ValueError("float32 conversion made v5 diagnostic values non-finite")
    if np.any(signal_variance < 0.0):
        raise ValueError("float32 conversion made between-ensemble variance negative")
    return signal_imag, signal_variance, ensemble_means_subset


def _save_library(lib: list[LibraryEntry], path: str,
                  cfg: SimConfig | None = None,
                  columns: "sig.ColumnGrid | None" = None,
                  grid_metadata: Optional[dict] = None,
                  build_metadata: Optional[dict] = None,
                  build_seed: int | None = None):
    """Persist library + (δ,Δ,b) grid metadata to npz.

    Stores:
        kios, rhos, Vs, vectors  — entry parameters and flat S vectors
        pair_deltas, pair_Deltas — (n_pairs,) δ, Δ for each stored pair
        b_values                 — (n_b,) b-values [s/mm²]
        n_b                      — number of b-values (for reshaping)
        h_ms                     — Y(t) storage stride the pairs were built on
        signal_imag              — v5 collapsed imaginary signal, float32
        signal_variance          — v5 between-ensemble sample variance, float32
        ensemble_means_subset    — v5 [entry, ensemble, 200] diagnostic, float32
    """
    if cfg is None:
        cfg = SimConfig()
    if columns is None:
        columns = sig.build_columns(cfg)

    n_columns = int(columns.n_pairs * columns.n_b)
    v5_arrays = _v5_entry_arrays(lib, n_columns)
    library_schema = MADI_LIBRARY_SCHEMA_V5 if v5_arrays is not None else MADI_LIBRARY_SCHEMA_V4

    kios = np.array([e.kio for e in lib], dtype=float)
    rhos = np.array([e.rho for e in lib], dtype=float)
    Vs   = np.array([e.V for e in lib], dtype=float)
    vis = np.array([e.realised_vi for e in lib], dtype=float)
    weights = np.array([
        np.nan if e.weight is None else float(e.weight) for e in lib
    ], dtype=float)
    vecs = np.array([e.vector for e in lib])
    nominal_kios = np.array([
        e.kio if e.kio_nominal is None else e.kio_nominal for e in lib
    ], dtype=float)
    nominal_rhos = np.array([
        e.rho if e.rho_nominal is None else e.rho_nominal for e in lib
    ], dtype=float)
    nominal_Vs = np.array([
        e.V if e.V_nominal is None else e.V_nominal for e in lib
    ], dtype=float)
    pps = np.array([np.nan if e.pp is None else e.pp for e in lib], dtype=float)
    kio_eq5 = np.array([
        np.nan if e.kio_analytic_eq5 is None else e.kio_analytic_eq5 for e in lib
    ], dtype=float)
    free = np.array([e.is_free_water for e in lib], dtype=np.bool_)
    entry_json = np.array([
        json.dumps(e.metadata, sort_keys=True, default=_json_default) for e in lib
    ])
    pair_deltas = np.array([d for d, D in columns.delta_pairs], dtype=float)
    pair_Deltas = np.array([D for d, D in columns.delta_pairs], dtype=float)
    embedded_build_metadata = (
        dict(build_metadata) if build_metadata is not None
        else {**_cfg_metadata(cfg, schema=library_schema, build_seed=build_seed), "grid": grid_metadata or {}}
    )
    if embedded_build_metadata.get("schema") != library_schema:
        raise ValueError(
            "library-schema mismatch between entries and build metadata: "
            f"entries imply {library_schema!r}, metadata has "
            f"{embedded_build_metadata.get('schema')!r}"
        )
    # A caller preserving existing build metadata (the shard merger) may
    # still explicitly replace just its grid section.
    if grid_metadata is not None:
        embedded_build_metadata["grid"] = grid_metadata
    payload = dict(
        library_schema=np.array(library_schema),
        kios=kios, rhos=rhos, Vs=Vs, vis=vis, vectors=vecs,
        nominal_kios=nominal_kios, nominal_rhos=nominal_rhos, nominal_Vs=nominal_Vs,
        weights=weights, has_weights=np.array(bool(np.all(np.isfinite(weights)))),
        pps=pps, kio_analytic_eq5=kio_eq5,
        is_free_water=free, entry_metadata_json=entry_json,
        build_metadata_json=np.array(json.dumps(
            embedded_build_metadata, sort_keys=True, default=_json_default,
        )),
        pair_deltas=pair_deltas, pair_Deltas=pair_Deltas,
        b_values=np.asarray(columns.b_values, dtype=float),
        n_b=np.array(columns.n_b),
        h_ms=np.array(float(cfg.h_ms)),
    )
    if v5_arrays is not None:
        signal_imag, signal_variance, ensemble_means_subset = v5_arrays
        payload.update(
            signal_imag=signal_imag,
            signal_variance=signal_variance,
            ensemble_means_subset=ensemble_means_subset,
            ensemble_subset_pair_deltas=np.asarray(
                [delta for delta, _ in ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS], dtype=float,
            ),
            ensemble_subset_pair_Deltas=np.asarray(
                [Delta for _, Delta in ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS], dtype=float,
            ),
            ensemble_subset_b_values=np.asarray(
                ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2, dtype=float,
            ),
            ensemble_subset_n_b=np.array(len(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2)),
        )
    np.savez(path, **payload)


def load_library(path: str) -> list[LibraryEntry]:
    # np.load() on a .npz returns a lazy NpzFile: EVERY `data[key]` subscript
    # re-opens the zip member and re-parses the array from scratch (no
    # caching). Pulling each array out ONCE here (rather than indexing
    # `data['vectors'][i]` inside the loop) avoids re-reading the full
    # (n_entries, n_features) vectors array once per entry -- an O(n^2)
    # blowup that made merging many shards balloon in memory/time.
    data = np.load(path)
    kios    = data['kios']
    rhos    = data['rhos']
    Vs      = data['Vs']
    vectors = data['vectors']
    n = len(kios)
    def optional(name, default):
        return data[name] if name in data.files else default
    vis = optional('vis', np.asarray([None] * n, dtype=object))
    weights = optional('weights', np.full(n, np.nan))
    nominal_kios = optional('nominal_kios', kios)
    nominal_rhos = optional('nominal_rhos', rhos)
    nominal_Vs = optional('nominal_Vs', Vs)
    pps = optional('pps', np.full(n, np.nan))
    kio_eq5 = optional('kio_analytic_eq5', np.full(n, np.nan))
    free = optional('is_free_water', np.zeros(n, dtype=bool))
    entry_json = optional('entry_metadata_json', np.asarray(["{}"] * n))
    signal_imag = optional('signal_imag', None)
    signal_variance = optional('signal_variance', None)
    ensemble_means_subset = optional('ensemble_means_subset', None)
    diagnostic_arrays = (signal_imag, signal_variance, ensemble_means_subset)
    if any(value is not None for value in diagnostic_arrays) and not all(
        value is not None for value in diagnostic_arrays
    ):
        raise ValueError(f"{path}: incomplete v5 Monte-Carlo diagnostic arrays")
    if signal_imag is not None and (
        signal_imag.shape[0] != n or signal_variance.shape[0] != n
        or ensemble_means_subset.shape[0] != n
    ):
        raise ValueError(f"{path}: v5 Monte-Carlo diagnostic entry axis has wrong length")
    lib = []
    for i in range(n):
        try:
            entry_meta = json.loads(str(entry_json[i]))
        except (TypeError, ValueError, json.JSONDecodeError):
            entry_meta = {}
        vi = None if vis.dtype == object or not np.isfinite(vis[i]) else float(vis[i])
        weight = None if not np.isfinite(weights[i]) else float(weights[i])
        lib.append(LibraryEntry(
            kio=float(kios[i]),
            rho=float(rhos[i]),
            V=float(Vs[i]),
            vector=vectors[i],
            signal_imag=None if signal_imag is None else signal_imag[i],
            signal_variance=None if signal_variance is None else signal_variance[i],
            ensemble_means_subset=(
                None if ensemble_means_subset is None else ensemble_means_subset[i]
            ),
            vi=vi,
            weight=weight,
            kio_nominal=float(nominal_kios[i]),
            rho_nominal=float(nominal_rhos[i]),
            V_nominal=float(nominal_Vs[i]),
            pp=None if not np.isfinite(pps[i]) else float(pps[i]),
            kio_analytic_eq5=None if not np.isfinite(kio_eq5[i]) else float(kio_eq5[i]),
            is_free_water=bool(free[i]),
            metadata=entry_meta,
        ))
    return lib


def load_library_meta(path: str) -> dict:
    """Load metadata from a library file — new (δ,Δ,b) format or legacy
    fixed-δ format, normalised to a common shape.

    Returns
    -------
    dict with keys:
        delta_pairs : list of (δ,Δ) [ms] — the pairs each column-group of
            n_b entries in the flat vector corresponds to
        b_values    : list of b-values [s/mm²]
        n_b         : number of b-values per pair
        format      : 'v2' (new, (δ,Δ,b)-universal) or 'legacy'
    """
    data = np.load(path)
    meta = {}

    if 'library_schema' in data.files:
        meta['library_schema'] = str(data['library_schema'])
    if 'has_weights' in data.files:
        meta['has_weights'] = bool(data['has_weights'])
    else:
        meta['has_weights'] = False
    if 'build_metadata_json' in data.files:
        try:
            meta['build_metadata'] = json.loads(str(data['build_metadata_json']))
        except (TypeError, ValueError, json.JSONDecodeError):
            meta['build_metadata'] = None

    if 'pair_deltas' in data.files and 'pair_Deltas' in data.files:
        meta['format'] = 'v2'
        pair_deltas = np.asarray(data['pair_deltas'], dtype=float)
        pair_Deltas = np.asarray(data['pair_Deltas'], dtype=float)
        meta['delta_pairs'] = list(zip(pair_deltas.tolist(), pair_Deltas.tolist()))
        meta['n_b'] = int(data['n_b']) if 'n_b' in data.files else None
        meta['b_values'] = (list(np.asarray(data['b_values'], dtype=float))
                             if 'b_values' in data.files else None)
        meta['h_ms'] = float(data['h_ms']) if 'h_ms' in data.files else None
        meta['has_free_water'] = bool(np.any(data['is_free_water'])) if 'is_free_water' in data.files else False
        diagnostic_names = {
            'signal_imag', 'signal_variance', 'ensemble_means_subset',
            'ensemble_subset_pair_deltas', 'ensemble_subset_pair_Deltas',
            'ensemble_subset_b_values', 'ensemble_subset_n_b',
        }
        meta['has_v5_diagnostics'] = diagnostic_names.issubset(set(data.files))
        if meta['has_v5_diagnostics']:
            meta['signal_imag_shape'] = tuple(data['signal_imag'].shape)
            meta['signal_variance_shape'] = tuple(data['signal_variance'].shape)
            meta['ensemble_means_subset_shape'] = tuple(data['ensemble_means_subset'].shape)
            subset_delta = np.asarray(data['ensemble_subset_pair_deltas'], dtype=float)
            subset_Delta = np.asarray(data['ensemble_subset_pair_Deltas'], dtype=float)
            meta['ensemble_subset_delta_pairs'] = list(zip(
                subset_delta.tolist(), subset_Delta.tolist(),
            ))
            meta['ensemble_subset_b_values'] = list(np.asarray(
                data['ensemble_subset_b_values'], dtype=float,
            ))
            meta['ensemble_subset_n_b'] = int(data['ensemble_subset_n_b'])
        return meta

    # ---- Legacy fixed-δ format: synthesize an equivalent delta_pairs ----
    meta['format'] = 'legacy'
    legacy_deltas = list(data['deltas']) if 'deltas' in data.files else list(DELTAS_BIG)
    n_b = int(data['n_b']) if 'n_b' in data.files else len(BVALS_UNIQUE)
    small_delta = (float(data['small_delta']) if 'small_delta' in data.files
                   else DELTA_SMALL)
    if small_delta is None:
        small_delta = DELTA_SMALL
    b_values = (list(np.asarray(data['b_values'], dtype=float))
                if 'b_values' in data.files
                else (list(BVALS_UNIQUE.astype(float)) if n_b == len(BVALS_UNIQUE) else None))

    meta['delta_pairs'] = [(small_delta, D) for D in legacy_deltas]
    meta['n_b'] = n_b
    meta['b_values'] = b_values
    meta['small_delta'] = small_delta   # kept for legacy call sites
    meta['h_ms'] = None
    meta['has_v5_diagnostics'] = False
    return meta


def library_summary(lib: list[LibraryEntry], meta: dict | None = None):
    """Print a summary of a library."""
    if not lib:
        print("  (empty library)")
        return
    cellular = [e for e in lib if not e.is_free_water]
    kios = sorted(set(e.kio for e in cellular))
    rhos = sorted(set(e.rho for e in cellular))
    Vs   = sorted(set(e.V for e in cellular))
    vis  = sorted(set(round(e.realised_vi, 4) for e in cellular))
    print(f"  Entries: {len(lib)}")
    print(f"  kio  ({len(kios)}): {[f'{k:.1f}' for k in kios]}")
    print(f"  rho  ({len(rhos)}): {[f'{r/1e3:.0f}k' for r in rhos]}")
    print(f"  V    ({len(Vs)}):   {[f'{v:.2f}' for v in Vs]}")
    if vis:
        print(f"  vi range: [{min(vis):.3f}, {max(vis):.3f}]")
    print(f"  free-water atom: {any(e.is_free_water for e in lib)}")

    vec_len = lib[0].vector.size
    if meta is not None:
        n_b = meta['n_b']
        pairs = meta['delta_pairs']
        bvs = meta.get('b_values')
        print(f"  Format: {meta.get('format', '?')}")
        print(f"  Vector length: {vec_len}  ({len(pairs)} (δ,Δ) pairs × "
              f"{n_b} b-values)")
        d_range = sorted(set(d for d, D in pairs))
        D_range = sorted(set(D for d, D in pairs))
        print(f"  δ range [ms]: [{d_range[0]:g}, {d_range[-1]:g}]  "
              f"({len(d_range)} unique)")
        print(f"  Δ range [ms]: [{D_range[0]:g}, {D_range[-1]:g}]  "
              f"({len(D_range)} unique)")
        if bvs is not None:
            print(f"  b-values [s/mm²]: {[f'{b:g}' for b in bvs]}")
        else:
            print(f"  b-values [s/mm²]: (not stored)")
    else:
        print(f"  Vector length: {vec_len}")


# ---------------------------------------------------------------------------
# Matching helpers — NEAREST column, never interpolated (see module docstring)
# ---------------------------------------------------------------------------

# These are acquisition-contract tolerances, not interpolation radii.  A
# requested coordinate inside the tolerance is snapped to a stored column and
# recorded by ``resolve_grid_columns``; a request outside it is rejected.
# Keeping the values here makes the direct matcher and the CLI agree.
B_SNAP_TOL_S_MM2 = 30.0
TIMING_SNAP_TOL_MS = 1.5

def _nearest_pair_index(delta: float, Delta: float, lib_pairs: np.ndarray) -> int:
    d2 = (lib_pairs[:, 0] - delta) ** 2 + (lib_pairs[:, 1] - Delta) ** 2
    return int(np.argmin(d2))


def resolve_grid_columns(fit_triples, lib_delta_pairs, lib_b_values, n_b,
                         b_tol=B_SNAP_TOL_S_MM2,
                         timing_tol=TIMING_SNAP_TOL_MS):
    """Resolve requested columns and return explicit non-interpolating snaps.

    The returned events are suitable for a fit run record.  Each request is
    matched to the nearest stored (δ, Δ, b) coordinate only if it falls
    inside the declared snap tolerance; no signal interpolation occurs.

    Parameters
    ----------
    fit_triples : list of (δ_ms, Δ_ms, b_s_mm2)
    lib_delta_pairs : list of (δ,Δ) [ms]
    lib_b_values : list of float [s/mm²]
    n_b : int — b-values per pair in the flat vector

    Returns
    -------
    cols : (n_triples,) int array
    snap_events : list[dict]
        One event per distinct snapped timing pair or b-value.  Exact matches
        are intentionally absent.
    """
    if lib_b_values is None:
        raise ValueError(
            "Library has no stored b-values. Rebuild the library with the "
            "current _save_library, or pass lib_b_values explicitly.")

    lib_pairs_arr = np.asarray(lib_delta_pairs, dtype=float)
    lib_b_arr = np.asarray(lib_b_values, dtype=float)

    cols = np.empty(len(fit_triples), dtype=int)
    snap_events: list[dict] = []
    seen_timing: set[tuple[float, float, float, float]] = set()
    seen_b: set[tuple[float, float]] = set()
    for k, (delta, Delta, b) in enumerate(fit_triples):
        pi = _nearest_pair_index(delta, Delta, lib_pairs_arr)
        nd, nD = lib_pairs_arr[pi]
        delta_offset = abs(float(nd - delta))
        Delta_offset = abs(float(nD - Delta))
        if delta_offset > timing_tol or Delta_offset > timing_tol:
            raise ValueError(
                f"(δ={delta:g}, Δ={Delta:g}) ms is outside the "
                f"{timing_tol:g} ms timing snap tolerance; nearest stored "
                f"pair is (δ={nd:g}, Δ={nD:g}) ms. No interpolation is "
                "implemented.")
        timing_key = (float(delta), float(Delta), float(nd), float(nD))
        if (delta_offset > 1e-12 or Delta_offset > 1e-12) and timing_key not in seen_timing:
            seen_timing.add(timing_key)
            snap_events.append({
                "axis": "timing",
                "requested": {"delta_ms": float(delta), "Delta_ms": float(Delta)},
                "used": {"delta_ms": float(nd), "Delta_ms": float(nD)},
                "abs_offset": {"delta_ms": delta_offset, "Delta_ms": Delta_offset},
                "rel_offset": {
                    "delta": delta_offset / max(abs(float(delta)), 1e-12),
                    "Delta": Delta_offset / max(abs(float(Delta)), 1e-12),
                },
            })

        bi = int(np.argmin(np.abs(lib_b_arr - b)))
        b_used = float(lib_b_arr[bi])
        b_offset = abs(b_used - float(b))
        if b_offset > b_tol:
            raise ValueError(
                f"b = {b} s/mm² not within {b_tol:g} of any library "
                f"b-value {sorted(lib_b_values)}.")
        b_key = (float(b), b_used)
        if b_offset > 1e-12 and b_key not in seen_b:
            seen_b.add(b_key)
            snap_events.append({
                "axis": "b",
                "requested": float(b),
                "used": b_used,
                "abs_offset": b_offset,
                "rel_offset": b_offset / max(abs(float(b)), 1e-12),
            })
        cols[k] = pi * n_b + bi
    return cols, snap_events


def _grid_columns(fit_triples, lib_delta_pairs, lib_b_values, n_b,
                  b_tol=B_SNAP_TOL_S_MM2,
                  pair_warn_tol=TIMING_SNAP_TOL_MS):
    """Column-only compatibility wrapper around :func:`resolve_grid_columns`.

    Direct API callers retain the historic return type.  The command-line
    fitter calls ``resolve_grid_columns`` once up front to make every snap
    visible and persist it in the run record.
    """
    cols, _ = resolve_grid_columns(
        fit_triples, lib_delta_pairs, lib_b_values, n_b,
        b_tol=b_tol, timing_tol=pair_warn_tol,
    )
    return cols


def _build_candidate_lib_matrix(library, lib_delta_pairs, lib_b_values,
                                 n_b, vi_min, vi_max, rho_max,
                                 fit_triples, *, include_free_water=False,
                                 return_weights=False, require_weights=False):
    """Apply candidate filtering and produce the masked, subset library matrix.

    Returns
    -------
    lib_mat : (n_candidates, n_features)
    kios_arr, rhos_arr, Vs_arr : (n_candidates,)
    """
    vis  = np.array([e.realised_vi for e in library])
    rhos = np.array([e.rho for e in library])
    free = np.array([e.is_free_water for e in library], dtype=bool)

    mask = (vis >= vi_min) & (vis <= vi_max)
    if include_free_water:
        mask |= free
    else:
        mask &= ~free
    if rho_max is not None:
        mask &= (rhos <= rho_max)

    n_candidates = int(mask.sum())
    if n_candidates == 0:
        raise ValueError(
            f"No library entries survive vi in [{vi_min}, {vi_max}] "
            f"and rho <= {rho_max}.")
    if n_candidates < 50:
        import warnings
        warnings.warn(f"Only {n_candidates} library entries pass the filter.")

    lib_entries = [e for e, m in zip(library, mask) if m]
    full_mat    = np.array([e.vector for e in lib_entries])

    col_idx = _grid_columns(fit_triples, lib_delta_pairs, lib_b_values, n_b)
    lib_mat = full_mat[:, col_idx]

    kios_arr = np.array([e.kio for e in lib_entries])
    rhos_arr = np.array([e.rho for e in lib_entries])
    Vs_arr   = np.array([e.V   for e in lib_entries])

    if not return_weights:
        return lib_mat, kios_arr, rhos_arr, Vs_arr
    weights = np.array([
        np.nan if e.weight is None else float(e.weight) for e in lib_entries
    ], dtype=float)
    if require_weights and not np.all(np.isfinite(weights) & (weights > 0.0)):
        raise ValueError(
            "Bayesian fitting requires positive per-entry quadrature weights. "
            "This library is legacy/unweighted; rebuild it with the remediation grid."
        )
    return lib_mat, kios_arr, rhos_arr, Vs_arr, weights


def edge_railing_diagnostics(
    kio_values: np.ndarray,
    rho_values: np.ndarray,
    V_values: np.ndarray,
    library: list[LibraryEntry],
    *,
    vi_min: float,
    vi_max: float,
    rho_max: float | None = None,
    include_free_water: bool = False,
) -> dict:
    """Summarise returned fits at the *actual* candidate-library edges.

    The diagnostic is deliberately based on realised entry labels rather
    than the nominal grid request.  That makes it sensitive to a build whose
    Poisson geometry or direct exchange measurement did not land where its
    requested coordinates suggested.  Free-water selections are reported as
    their own category, never folded into a cellular lower-bound fraction.
    """
    vis = np.asarray([e.realised_vi for e in library], dtype=float)
    rhos = np.asarray([e.rho for e in library], dtype=float)
    free = np.asarray([e.is_free_water for e in library], dtype=bool)
    candidate = (vis >= float(vi_min)) & (vis <= float(vi_max)) & ~free
    if include_free_water:
        candidate |= free
    if rho_max is not None:
        candidate &= rhos <= float(rho_max)
    cellular = [entry for entry, keep in zip(library, candidate) if keep and not entry.is_free_water]
    if not cellular:
        raise ValueError("edge diagnostic has no cellular candidate entries")

    kio = np.asarray(kio_values, dtype=float)
    rho = np.asarray(rho_values, dtype=float)
    volume = np.asarray(V_values, dtype=float)
    if not (kio.shape == rho.shape == volume.shape):
        raise ValueError("kio, rho and V diagnostic arrays must have the same shape")
    n_total = int(kio.size)
    free_selected = (~np.isfinite(kio)) & (rho == 0.0) & (volume == 0.0)

    def one(name: str, returned: np.ndarray, grid: np.ndarray) -> dict:
        finite = np.isfinite(returned) & ~free_selected
        lower = float(np.min(grid))
        upper = float(np.max(grid))
        # Directly measured entries are floating point values, so a tight
        # isclose test is more appropriate than relying on printed rounding.
        at_lower = finite & np.isclose(returned, lower, rtol=1e-10, atol=1e-12)
        at_upper = finite & np.isclose(returned, upper, rtol=1e-10, atol=1e-12)
        denom = max(n_total, 1)
        return {
            "lower_value": lower,
            "upper_value": upper,
            "at_lower_count": int(np.count_nonzero(at_lower)),
            "at_upper_count": int(np.count_nonzero(at_upper)),
            "at_lower_fraction": float(np.count_nonzero(at_lower) / denom),
            "at_upper_fraction": float(np.count_nonzero(at_upper) / denom),
        }

    return {
        "n_voxels": n_total,
        "free_water_count": int(np.count_nonzero(free_selected)),
        "free_water_fraction": float(np.count_nonzero(free_selected) / max(n_total, 1)),
        "rho": one("rho", rho, np.asarray([e.rho for e in cellular], dtype=float)),
        "V": one("V", volume, np.asarray([e.V for e in cellular], dtype=float)),
        "kio": one("kio", kio, np.asarray([e.kio for e in cellular], dtype=float)),
    }


# ---------------------------------------------------------------------------
# FIXED-S0 matcher (existing behavior, kept)
# ---------------------------------------------------------------------------

def match_voxels_batch(
    measured_batch,
    library,
    lib_delta_pairs, lib_b_values, n_b,
    fit_triples,
    log_space=False, s_floor=1e-3,
    vi_min=0.5, vi_max=0.95, rho_max=None,
    include_free_water=False,
    use_gpu=None,
):
    """Log-space nearest-neighbour matching, S0 fixed (data already
    divided by measured b=0).

    Inputs are S/S0 ratios. ``fit_triples`` is a list of (δ,Δ,b) tuples in
    the column order of ``measured_batch`` — see `_grid_columns` for the
    nearest-column (no-interpolation) matching semantics.

    ``use_gpu`` : None (default) = use CUDA if available, else CPU.
    ``True``/``False`` force a path (``True`` raises if CUDA is
    unavailable). The GPU path (``fitters_gpu.map_match_gpu``) is an exact
    reordering of this function's math — same output to float64 precision.
    """

    lib_mat, kios_arr, rhos_arr, Vs_arr = _build_candidate_lib_matrix(
        library, lib_delta_pairs, lib_b_values, n_b, vi_min, vi_max, rho_max,
        fit_triples, include_free_water=include_free_water)

    if log_space:
        measured = np.log(np.clip(measured_batch, s_floor, 1.0))
        lib_m    = np.log(np.clip(lib_mat,         s_floor, 1.0))
    else:
        measured = measured_batch
        lib_m    = lib_mat

    if use_gpu is None:
        use_gpu = fitters_gpu.HAS_CUDA
    if use_gpu:
        if not fitters_gpu.HAS_CUDA:
            raise RuntimeError("use_gpu=True but CUDA is not available.")
        return fitters_gpu.map_match_gpu(measured, lib_m, kios_arr, rhos_arr,
                                          Vs_arr)

    m2 = np.sum(measured ** 2, axis=1, keepdims=True)
    l2 = np.sum(lib_m   ** 2, axis=1, keepdims=True).T
    dists = m2 + l2 - 2.0 * measured @ lib_m.T

    best_idx = np.argmin(dists, axis=1)
    return (kios_arr[best_idx], rhos_arr[best_idx], Vs_arr[best_idx],
            dists[np.arange(len(best_idx)), best_idx])


# ---------------------------------------------------------------------------
# FREE-S0 matcher (new)
# ---------------------------------------------------------------------------

def match_voxels_batch_fits0(
    raw_signal,
    library,
    lib_delta_pairs, lib_b_values, n_b,
    fit_triples,
    vi_min=0.5, vi_max=0.95, rho_max=None,
    include_free_water=False,
    use_gpu=None,
):
    """Match un-normalized signals with S0 as a free per-voxel linear param.

    See ``match_voxels_batch`` for the ``fit_triples`` semantics and the
    ``use_gpu`` convention.

    For each voxel m and each candidate library ratio vector r,

        S0*(m, r) = (m . r) / (r . r)
        residual  = ||m||^2  -  (m . r)^2 / (r . r)

    Returns
    -------
    kio_map, rho_map, V_map, residual_map, s0_fit_map  (each shape (n_voxels,))
    """
    lib_mat, kios_arr, rhos_arr, Vs_arr = _build_candidate_lib_matrix(
        library, lib_delta_pairs, lib_b_values, n_b, vi_min, vi_max, rho_max,
        fit_triples, include_free_water=include_free_water)

    M = raw_signal.astype(np.float64)            # (n_vox, n_feat)
    R = lib_mat.astype(np.float64)               # (n_lib, n_feat)

    if use_gpu is None:
        use_gpu = fitters_gpu.HAS_CUDA
    if use_gpu:
        if not fitters_gpu.HAS_CUDA:
            raise RuntimeError("use_gpu=True but CUDA is not available.")
        return fitters_gpu.map_match_fits0_gpu(M, R, kios_arr, rhos_arr,
                                                Vs_arr)

    # Per-library-entry  r.r   shape (n_lib,)
    rr = np.sum(R * R, axis=1)
    rr = np.maximum(rr, 1e-30)

    # Per-voxel  m.m  shape (n_vox,)
    mm = np.sum(M * M, axis=1)

    # Cross  M @ R.T  shape (n_vox, n_lib)
    MR = M @ R.T

    # S0 candidates per (voxel, library entry)   shape (n_vox, n_lib)
    S0_cand = MR / rr[None, :]

    # Residuals  shape (n_vox, n_lib)
    #   ||m||^2 - (m.r)^2 / (r.r)
    resid = mm[:, None] - (MR ** 2) / rr[None, :]

    # Forbid negative S0 (would correspond to flipping the signal)
    resid_masked = np.where(S0_cand > 0, resid, np.inf)

    best_idx = np.argmin(resid_masked, axis=1)
    rows = np.arange(len(best_idx))

    return (kios_arr[best_idx],
            rhos_arr[best_idx],
            Vs_arr[best_idx],
            resid_masked[rows, best_idx],
            S0_cand[rows, best_idx])
