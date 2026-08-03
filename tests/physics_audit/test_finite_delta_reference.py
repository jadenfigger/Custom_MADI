"""Reduced direct simulation demonstrating the finite-lobe reference mismatch.

This is intentionally not a replacement for a 12-million-walk production
regression.  It holds trajectories fixed and evaluates two phase definitions,
so a significant difference cannot be attributed to differing random walks.
"""

from __future__ import annotations

import numpy as np
import pytest

from madi.config import GAMMA_RAD
from madi.ensemble import create_ensemble
from madi.signal import G_from_b
from madi.walker_gpu import kio_to_pp

from .test_geometry_diagnostics import _geometry_cfg, _membrane_count, _require_geometry


def _same_paths_finite_and_narrow_signal(
    *,
    delta_ms: float,
    Delta_ms: float,
    b_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mirror the CPU transition rule and evaluate both phase conventions."""
    cfg = _require_geometry()
    # This has target v_i=0.70, comfortably inside the lookup table range.
    ens = create_ensemble(700_000.0, 1.0, cfg, seed=818, verify_vi=False)
    pp = kio_to_pp(20.0, ens.mean_AV, cfg)
    rng = np.random.default_rng(819)
    n = cfg.n_walkers
    position = rng.uniform(cfg.buffer, cfg.L - cfg.buffer, size=(n, 3))
    initial_x = position[:, 0].copy()
    cell, inside = ens.classify_cpu(position)
    frozen = np.zeros(n, dtype=bool)
    m1 = np.zeros(n)
    m2 = np.zeros(n)
    t_d_step = int(round((Delta_ms - delta_ms / 3.0) / cfg.ts))
    x_t_d = None
    end_step = int(round((Delta_ms + delta_ms) / cfg.ts))

    for step in range(end_step):
        active = ~frozen
        old = position.copy()
        proposal = position + rng.normal(0.0, cfg.sigma, size=position.shape)
        escaped = active & ((proposal < 0.0).any(axis=1) | (proposal >= cfg.L).any(axis=1))
        frozen |= escaped
        proposal[escaped] = position[escaped]
        active = ~frozen
        new_cell, new_inside = ens.classify_cpu(proposal)
        crossing = active & (_membrane_count(cell, inside, new_cell, new_inside) > 0)
        if crossing.any() and pp < 1.0:
            m = _membrane_count(cell, inside, new_cell, new_inside)
            rejected = np.flatnonzero(crossing)[rng.uniform(size=int(crossing.sum())) >= pp ** m[crossing]]
            proposal[rejected] = position[rejected]
            new_cell[rejected] = cell[rejected]
            new_inside[rejected] = inside[rejected]
        position = proposal
        cell[active] = new_cell[active]
        inside[active] = new_inside[active]

        t0 = step * cfg.ts
        if t0 < delta_ms:
            m1[active] += 0.5 * (old[active, 0] + position[active, 0]) * cfg.ts
        if Delta_ms <= t0 < Delta_ms + delta_ms:
            m2[active] += 0.5 * (old[active, 0] + position[active, 0]) * cfg.ts
        if step + 1 == t_d_step:
            x_t_d = position[:, 0].copy()

    assert x_t_d is not None
    keep = ~frozen
    finite, narrow = [], []
    for b in b_values:
        gradient = G_from_b(float(b), delta_ms, Delta_ms)
        finite_phase = GAMMA_RAD * gradient * (m1[keep] - m2[keep]) * 1e-9
        q = GAMMA_RAD * gradient * (delta_ms * 1e-3)
        narrow_phase = q * (x_t_d[keep] - initial_x[keep]) * 1e-6
        finite.append(np.cos(finite_phase).mean())
        narrow.append(np.cos(narrow_phase).mean())
    return np.asarray(finite), np.asarray(narrow), int(keep.sum())


@pytest.mark.slow
@pytest.mark.parametrize("delta_ms,Delta_ms", [(20.0, 50.0), (7.0, 25.0)])
def test_finite_lobe_and_narrow_pulse_signals_differ_on_identical_paths(
    delta_ms: float, Delta_ms: float
) -> None:
    _require_geometry()
    finite, narrow, n_kept = _same_paths_finite_and_narrow_signal(
        delta_ms=delta_ms,
        Delta_ms=Delta_ms,
        b_values=np.asarray([1_000.0, 2_000.0, 6_000.0]),
    )
    assert n_kept > 0
    relative = np.abs(finite - narrow) / np.maximum(np.abs(narrow), 1e-12)
    # This is a difference detector, not a target-value assertion.  A future
    # narrow-pulse reference implementation should make this test intentionally
    # fail and should be accompanied by a revised reference comparison.
    assert relative.max() > 0.05, (finite, narrow, relative, n_kept)
