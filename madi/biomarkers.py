"""Voxel-first MADI derived biomarkers.

The fitted quantities are ``k_io`` [s^-1], ``rho`` [cells/uL], and ``V``
[pL/cell].  Every other MADI biomarker is derived *per voxel* from that
triplet.  This module deliberately has no API that accepts ROI summary
statistics: doing so would make the invalid identity
``median(k_io) * median(V) == median(k_io * V)`` too easy to write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


INTRACELLULAR_WATER_NMOL_PER_PL = 0.042
"""42 M intracellular water, expressed as nmol/pL."""


@dataclass(frozen=True)
class VoxelwiseBiomarkers:
    """Derived arrays tied to one voxel-wise fitted parameter triplet.

    Arrays are intentionally retained rather than reducing to scalars.  Use
    :func:`summarise_voxelwise_biomarkers` only after this object has been
    created, which prevents an ROI-median product from masquerading as a
    voxelwise derived biomarker.
    """

    vi: np.ndarray
    kioV: np.ndarray
    kioVrho: np.ndarray
    water_efflux_nmol_s_cell: np.ndarray


def _as_voxel_array(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        raise TypeError(
            f"{name} must be a voxel-wise array, not an ROI scalar. "
            "Derive biomarkers before aggregating an ROI."
        )
    return arr


def derive_voxelwise_biomarkers(
    kio_s_inv: np.ndarray,
    rho_cells_per_uL: np.ndarray,
    volume_pL_per_cell: np.ndarray,
) -> VoxelwiseBiomarkers:
    """Compute the MADI derived biomarkers independently in every voxel.

    ``NaN`` k_io values (the explicit free-water entry has undefined k_io)
    propagate to exchange-dependent quantities, while its ``v_i`` correctly
    remains zero because rho and V are zero.
    """
    kio = _as_voxel_array("k_io", kio_s_inv)
    rho = _as_voxel_array("rho", rho_cells_per_uL)
    volume = _as_voxel_array("V", volume_pL_per_cell)
    if not (kio.shape == rho.shape == volume.shape):
        raise ValueError("k_io, rho, and V must have identical voxel-array shapes")

    vi = rho * volume * 1.0e-6
    kioV = kio * volume
    kioVrho = kioV * rho
    water_efflux = INTRACELLULAR_WATER_NMOL_PER_PL * kioV
    return VoxelwiseBiomarkers(
        vi=vi,
        kioV=kioV,
        kioVrho=kioVrho,
        water_efflux_nmol_s_cell=water_efflux,
    )


def summarise_voxelwise_biomarkers(
    biomarkers: VoxelwiseBiomarkers,
    *,
    statistic: Literal["median", "mean"] = "median",
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Aggregate already-derived maps, never products of ROI summaries."""
    if not isinstance(biomarkers, VoxelwiseBiomarkers):
        raise TypeError(
            "Pass VoxelwiseBiomarkers produced by derive_voxelwise_biomarkers; "
            "ROI parameter medians cannot be combined into MADI biomarkers."
        )
    if statistic == "median":
        reducer = np.nanmedian
    elif statistic == "mean":
        reducer = np.nanmean
    else:
        raise ValueError("statistic must be 'median' or 'mean'")

    arrays = {
        "vi": biomarkers.vi,
        "kioV_pL_s_cell": biomarkers.kioV,
        "kioVrho_pL_s_uL": biomarkers.kioVrho,
        "water_efflux_nmol_s_cell": biomarkers.water_efflux_nmol_s_cell,
    }
    if mask is None:
        return {name: float(reducer(values)) for name, values in arrays.items()}

    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != biomarkers.vi.shape:
        raise ValueError("ROI mask must have the same shape as the voxelwise maps")
    return {name: float(reducer(values[mask_arr])) for name, values in arrays.items()}
