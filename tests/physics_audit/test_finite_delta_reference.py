"""Tier-A finite-gradient reference checks independent of tissue geometry.

Free Gaussian water is the decisive analytic limit for the finite-lobe
moment: for every rectangular PGSE timing, it must give
``S = exp(-b D0)`` when ``b`` includes the ``delta/3`` term.  A second test
uses a deterministic microsecond random stream to ensure Y(t) is integrated
at the walker timestep, not reconstructed from one-millisecond positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from madi.config import GAMMA_RAD, SimConfig
from madi.ensemble import create_dummy_ensemble
from madi.signal import G_from_b
from madi.walker_gpu import WalkRandomStream, run_walk_Y


pytestmark = pytest.mark.tier_a


def _finite_signal(Y: np.ndarray, cfg: SimConfig, delta_ms: float,
                   Delta_ms: float, b_s_mm2: float) -> tuple[float, float]:
    jd = int(round(delta_ms / cfg.h_ms))
    jD = int(round(Delta_ms / cfg.h_ms))
    js = int(round((delta_ms + Delta_ms) / cfg.h_ms))
    dM = Y[:, jd, :] + Y[:, jD, :] - Y[:, js, :]
    phase = GAMMA_RAD * G_from_b(b_s_mm2, delta_ms, Delta_ms) * dM * 1.0e-9
    samples = np.cos(phase).ravel()
    return float(np.mean(samples)), float(np.std(samples, ddof=1) / np.sqrt(samples.size))


def test_free_water_finite_lobe_matches_stejskal_tanner_across_timings() -> None:
    timings = [
        (1.0, 20.0), (2.0, 20.0), (4.0, 20.0), (5.0, 20.0),
        (6.0, 20.0), (8.0, 20.0), (10.0, 20.0), (12.0, 20.0),
        (7.0, 25.0), (20.0, 50.0),
    ]
    cfg = SimConfig(
        # Production free water also uses SI Eq. S8; a hand-sized 200-um
        # diagnostic box is too small for a 70-ms source-domain walk.
        L=None, n_walkers=512, n_ensembles=1, T_max_ms=70.0,
        small_deltas=sorted({delta for delta, _ in timings}),
        big_deltas=sorted({Delta for _, Delta in timings}),
        b_values=[0.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0],
    )
    ensemble = create_dummy_ensemble(cfg)
    Y, escaped = run_walk_Y(ensemble, 0.0, cfg, pp=0.0, seed=20_260_901,
                             verbose=False, use_gpu=False)
    assert escaped == 0
    for delta, Delta in timings:
        for b in cfg.b_values:
            observed, mc_se = _finite_signal(Y, cfg, delta, Delta, b)
            expected = float(np.exp(-float(b) * cfg.D0 * 1.0e-3))
            # This is an absolute Monte-Carlo bound.  At b=6000 the analytic
            # signal is ~1.5e-8, so relative error would be meaningless.
            assert abs(observed - expected) <= 5.0 * mc_se + 0.012, (
                delta, Delta, b, observed, expected, mc_se,
            )


def test_Y_is_microsecond_trapezoid_integral_not_millisecond_reconstruction() -> None:
    cfg = SimConfig(
        L=200.0, n_walkers=1, n_ensembles=1, T_max_ms=2.0, h_ms=1.0,
        small_deltas=[1.0], big_deltas=[1.0, 2.0], b_values=[0.0, 1000.0],
    )
    steps = cfg.n_steps
    initial = np.array([[100.0, 100.0, 100.0]])
    t = np.arange(steps, dtype=float) * cfg.ts
    increments = np.zeros((steps, 1, 3), dtype=float)
    # Nonlinear deterministic motion makes a coarse 1-ms trapezoid visibly
    # wrong; all coordinates remain comfortably inside the finite SI box.
    increments[:, 0, 0] = 4.0e-3 + 2.0e-3 * np.sin(2.0 * np.pi * t / 0.37)
    increments[:, 0, 1] = -1.0e-3 + 1.0e-3 * np.cos(2.0 * np.pi * t / 0.23)
    stream = WalkRandomStream(
        initial_positions=initial,
        increments=increments,
        acceptance_uniforms=np.zeros((steps, 1), dtype=float),
    )
    Y, escaped = run_walk_Y(
        create_dummy_ensemble(cfg), 0.0, cfg, pp=0.0, seed=0,
        verbose=False, random_stream=stream, use_gpu=False,
    )
    assert escaped == 0

    position = initial[0].copy()
    previous = position - cfg.L / 2.0
    integral = np.zeros(3, dtype=float)
    expected = [np.zeros(3, dtype=float)]
    for step, increment in enumerate(increments[:, 0, :], start=1):
        position += increment
        current = position - cfg.L / 2.0
        integral += 0.5 * (previous + current) * cfg.ts
        previous = current
        if step % cfg.steps_per_h == 0:
            expected.append(integral.copy())
    assert np.array_equal(Y[0], np.asarray(expected))
