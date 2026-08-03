"""Tier-B direct k_io calibration study for remediation P0-B.

The production label is the tagged-starting-cell first-exit hazard.  This
test deliberately evaluates a small v_i x V matrix at one common low p_p and
checks the independent survivor-curve estimate plus the recorded Eq.-5
comparator.  It does not make Eq. 5 an acceptance target or a label.
"""

from __future__ import annotations

import numpy as np
import pytest

from madi.config import SimConfig
from madi.ensemble import create_ensemble
from madi.walker_gpu import pp_to_kio_eq5, run_walk_Y


pytestmark = pytest.mark.tier_b


def _event_rate(telemetry: dict) -> tuple[float, float]:
    metrics = np.asarray(telemetry["metrics"], dtype=float)
    at_risk_ms = float(metrics[:, 5].sum())
    events = float(metrics[:, 6].sum())
    return (
        1000.0 * events / at_risk_ms,
        1000.0 * np.sqrt(events) / at_risk_ms,
    )


def _survival_rate(telemetry: dict, cfg: SimConfig) -> float:
    fraction = np.asarray(telemetry["start_survivor_fraction"], dtype=float)
    keep = fraction > 0.10 * fraction[0]
    if np.count_nonzero(keep) < 2:
        return 0.0
    time_ms = np.arange(len(fraction), dtype=float) * cfg.h_ms
    return max(0.0, -1000.0 * np.polyfit(
        time_ms[keep], np.log(fraction[keep] / fraction[0]), 1,
    )[0])


def test_tagged_start_cell_kio_and_eq5_comparator_are_smooth_across_geometry() -> None:
    """P0-B: direct labels remain smooth but are not silently Eq.-5 scaled."""
    cfg = SimConfig(
        L=60.0,
        grid_spacing=0.75,
        classifier_candidates=8,
        geometry_calibration_points=100_000,
        geometry_validation_points=100_000,
        geometry_sample_cells=64,
        geometry_vi_tolerance=0.005,
        n_walkers=512,
        n_ensembles=1,
        T_max_ms=12.0,
        small_deltas=[1.0],
        big_deltas=[2.0, 8.0],
        b_values=[0.0],
    )
    pp = 0.005
    ratios = []
    for i, target_vi in enumerate((0.40, 0.70, 0.90)):
        for j, target_V in enumerate((0.20, 1.0, 5.0)):
            rho = target_vi * 1e6 / target_V
            seed = 20_260_901 + i * 10_007 + j * 1_009
            ens = create_ensemble(rho, target_V, cfg, seed=seed)
            _, escaped, telemetry = run_walk_Y(
                ens, 0.0, cfg, pp=pp, seed=seed + 1,
                verbose=False, return_telemetry=True,
            )
            direct, direct_se = _event_rate(telemetry)
            survivor = _survival_rate(telemetry, cfg)
            analytic = pp_to_kio_eq5(pp, ens.mean_AV, cfg)
            assert escaped == 0
            assert direct > 0.0 and analytic > 0.0
            # Two independent readings of the requested starting-cell rate.
            assert abs(direct - survivor) <= 4.0 * direct_se + 0.08 * direct
            ratios.append(direct / analytic)

    ratios = np.asarray(ratios)
    # The Gaussian-step/geometry discrepancy is retained entry-wise.  This
    # stable study establishes that it is a smooth, finite calibration effect
    # rather than a lookup-table discontinuity or a hidden global label.
    assert np.all((ratios > 0.7) & (ratios < 1.5))
    assert ratios.max() / ratios.min() < 1.5
