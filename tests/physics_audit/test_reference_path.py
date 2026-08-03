"""Fast, deterministic checks for the MADI reference path.

These tests intentionally exercise only the fixed-S0, linear, exhaustive MAP
matcher.  Bayesian, NNLS, Rician, and free-S0 modes are not part of this
reference contract.
"""

from __future__ import annotations

import math

import numpy as np

from madi.config import D0_UM2_MS, GAMMA_RAD, SimConfig
from madi.library import LibraryEntry, _grid_columns, match_voxels_batch
from madi.signal import G_from_b
from madi.walker_gpu import kio_to_pp, pp_to_kio


def test_constants_step_convention_and_einstein_increment() -> None:
    cfg = SimConfig()
    assert D0_UM2_MS == 3.0
    assert cfg.ts == 1e-3
    assert np.isclose(cfg.sigma, math.sqrt(2.0 * 3.0 * 1e-3))
    assert np.isclose(cfg.ls_rms, 0.1341640786499874)
    assert np.isclose(cfg.ls_rms**2, 6.0 * cfg.D0 * cfg.ts)
    assert np.isclose(GAMMA_RAD, 2.675222e8)


def test_vi_units_and_permeability_round_trip() -> None:
    cfg = SimConfig()
    # rho [cells/uL] * V [pL/cell] * 1e-6 is dimensionless v_i.
    assert np.isclose((180_000 / 1e9) * (5.0 * 1e3), 0.90)
    assert np.isclose((781_000 / 1e9) * (1.0 * 1e3), 0.781)

    mean_av = 0.5  # um^-1
    pp = kio_to_pp(12.0, mean_av, cfg)
    assert np.isclose(pp_to_kio(pp, mean_av, cfg), 12.0)
    expected = (12.0 / 1000.0) / (math.sqrt(3.0 / 0.006) * mean_av)
    assert np.isclose(pp, expected)


def test_stejskal_tanner_gradient_unit_round_trip() -> None:
    b = 4_000.0  # s/mm^2
    delta = 20.0
    Delta = 50.0
    gradient = G_from_b(b, delta, Delta)
    b_reconstructed = (
        (GAMMA_RAD * gradient * delta * 1e-3) ** 2
        * ((Delta - delta / 3.0) * 1e-3)
        / 1e6
    )
    assert np.isclose(b_reconstructed, b)


def _small_library() -> list[LibraryEntry]:
    # Both entries are within the matcher's default v_i window [0.5, 0.95].
    return [
        LibraryEntry(11.0, 1_000_000.0, 0.80, np.array([0.8, 0.5, 0.3])),
        LibraryEntry(22.0, 1_000_000.0, 0.81, np.array([0.8, 0.5, 0.3])),
        LibraryEntry(33.0, 1_000_000.0, 0.82, np.array([0.7, 0.4, 0.2])),
    ]


def test_reference_matcher_is_linear_exhaustive_and_deterministic() -> None:
    library = _small_library()
    triples = [(20.0, 50.0, b) for b in (500.0, 1_000.0, 1_500.0)]
    measured = np.array([[0.8, 0.5, 0.3], [0.71, 0.39, 0.19]])
    kwargs = dict(
        lib_delta_pairs=[(20.0, 50.0)],
        lib_b_values=[500.0, 1_000.0, 1_500.0],
        n_b=3,
        fit_triples=triples,
        use_gpu=False,
    )

    first = match_voxels_batch(measured, library, **kwargs)
    second = match_voxels_batch(measured, library, **kwargs)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)

    kio, rho, volume, residual = first
    # First row is an exact tie between the first two entries: np.argmin must
    # choose the lower library index, which makes the reference deterministic.
    assert kio.tolist() == [11.0, 33.0]
    assert rho.tolist() == [1_000_000.0, 1_000_000.0]
    assert volume.tolist() == [0.80, 0.82]
    assert np.all(residual >= -1e-14)


def test_pair_major_b_major_column_order() -> None:
    pairs = [(1.0, 1.0), (20.0, 50.0)]
    bvals = [0.0, 500.0, 1_000.0]
    cols = _grid_columns(
        [(20.0, 50.0, 0.0), (20.0, 50.0, 1_000.0)],
        pairs,
        bvals,
        n_b=3,
    )
    assert cols.tolist() == [3, 5]


def test_finite_lobe_moment_is_not_the_narrow_pulse_displacement() -> None:
    """A non-linear trajectory makes the two physics models visibly distinct.

    The current forward path uses ``Y(delta)+Y(Delta)-Y(Delta+delta)``.
    The published-reference model in the audit specification uses an endpoint
    displacement at t_D = Delta - delta/3.  They agree for no general path.
    """
    delta, Delta = 20.0, 50.0
    t_d = Delta - delta / 3.0
    # x(t)=t^2 in arbitrary units; Y(t)=integral x dt=t^3/3.
    y = lambda t: t**3 / 3.0
    x = lambda t: t**2
    finite_lobe_average_difference = (y(delta) + y(Delta) - y(Delta + delta)) / delta
    narrow_pulse_displacement = x(t_d) - x(0.0)
    assert not np.isclose(
        abs(finite_lobe_average_difference), narrow_pulse_displacement
    )
