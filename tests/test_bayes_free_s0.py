"""Free-S0 Bayes weighting: profile likelihood against the measured S0."""

from __future__ import annotations

import numpy as np
import pytest

from madi.fitters import bayes_fit
from madi.library import LibraryEntry


pytestmark = pytest.mark.tier_a

B_VALUES = [0.0, 1000.0, 2000.0, 3000.0]
FIT_KW = dict(
    lib_delta_pairs=[(1.0, 2.0)], lib_b_values=B_VALUES, n_b=len(B_VALUES),
    fit_triples=[(1.0, 2.0, b) for b in B_VALUES[1:]],
    vi_min=0.0, vi_max=1.0, use_gpu=False,
)


def _library():
    """Distinct decay shapes, including flat low-signal ones that only fit
    well with a large amplitude (the candidates the old weighting favoured)."""
    b = np.asarray(B_VALUES) / 1000.0
    params = [(0.2, 0.00), (0.5, 0.05), (0.8, 0.10), (1.1, 0.15), (1.4, 0.20),
              (0.05, 0.00), (0.10, 0.00)]
    entries = []
    for i, (D, K) in enumerate(params):
        vec = np.exp(-b * D + (b * D) ** 2 * K / 6.0)
        entries.append(LibraryEntry(
            kio=10.0 * (i + 1), rho=(i + 1) * 1.0e5, V=1.0 + 0.5 * i,
            vector=vec, vi=0.7, weight=1.0))
    return entries


def _fit_vectors(entries):
    return np.stack([e.vector[1:] for e in entries])


def test_free_s0_matches_fixed_s0_on_scaled_exact_entry():
    lib = _library()
    S = _fit_vectors(lib)
    truth = [0, 2, 4, 1, 3]
    amplitude = np.array([500.0, 3000.0, 80.0, 12000.0, 1.0])
    raw = amplitude[:, None] * S[truth]
    measured = raw / amplitude[:, None]

    fixed = bayes_fit(measured, lib, sigma_m=0.002, **FIT_KW)
    free = bayes_fit(measured, lib, sigma_m=0.002, fit_s0=True,
                     raw_signal=raw, s0_measured=amplitude, **FIT_KW)

    for key in ("kio_mean", "rho_mean", "V_mean"):
        np.testing.assert_allclose(free[key], fixed[key], rtol=1e-6)
    np.testing.assert_allclose(free["kio_mean"],
                               [lib[k].kio for k in truth], rtol=1e-6)
    np.testing.assert_allclose(free["s0_fit"], amplitude, rtol=1e-6)


def test_free_s0_weights_are_profile_likelihood_on_measured_s0():
    lib = _library()
    S = _fit_vectors(lib)
    rng = np.random.default_rng(3)
    s0 = rng.uniform(200.0, 5000.0, 6)
    a_true = rng.uniform(0.8, 1.2, 6)
    raw = (s0 * a_true)[:, None] * S[rng.integers(0, 5, 6)]
    raw += rng.normal(0.0, 0.02, raw.shape) * s0[:, None]
    sigma_m = 0.05

    out = bayes_fit(raw / s0[:, None], lib, sigma_m=sigma_m, fit_s0=True,
                    raw_signal=raw, s0_measured=s0, **FIT_KW)

    # Reference: r_i = min_a ||m - a s_i||^2 on m = M / S0_measured.
    m = raw / s0[:, None]
    a = (m @ S.T) / np.sum(S * S, axis=1)
    r = np.sum(m * m, axis=1)[:, None] - a * (m @ S.T)
    r = np.where(a > 0, r, np.inf)
    w = np.exp(-(r - r.min(axis=1, keepdims=True)) / (2 * sigma_m ** 2))
    w /= w.sum(axis=1, keepdims=True)
    kios = np.array([e.kio for e in lib])

    np.testing.assert_allclose(out["kio_mean"], w @ kios, rtol=1e-10)
    np.testing.assert_allclose(out["n_eff"], 1.0 / np.sum(w ** 2, axis=1),
                               rtol=1e-10)
    np.testing.assert_allclose(out["s0_fit"], np.sum(w * a, axis=1) * s0,
                               rtol=1e-10)


def test_free_s0_posterior_is_unit_invariant_but_not_amplitude_invariant():
    lib = _library()
    S = _fit_vectors(lib)
    raw = 1.12 * 1000.0 * S[[1, 3]] + np.array([[0.0, 4.0, -3.0]])
    s0 = np.array([1000.0, 1000.0])
    kw = dict(sigma_m=0.02, fit_s0=True, **FIT_KW)

    base = bayes_fit(raw / s0[:, None], lib, raw_signal=raw, s0_measured=s0, **kw)
    scaled = bayes_fit(raw / s0[:, None], lib, raw_signal=7.0 * raw,
                       s0_measured=7.0 * s0, **kw)
    for key in ("kio_mean", "n_eff"):
        np.testing.assert_allclose(scaled[key], base[key], rtol=1e-10)

    # Halving the measured S0 doubles the noise-normalized residuals, so the
    # posterior must sharpen; dividing by each candidate's own fitted S0
    # (the old weighting) would leave it unchanged.
    sharper = bayes_fit(raw / s0[:, None], lib, raw_signal=raw,
                        s0_measured=0.5 * s0, **kw)
    assert np.all(sharper["n_eff"] < base["n_eff"])


def test_free_s0_requires_measured_s0():
    lib = _library()
    S = _fit_vectors(lib)
    with pytest.raises(ValueError, match="s0_measured"):
        bayes_fit(S[:1], lib, sigma_m=0.01, fit_s0=True,
                  raw_signal=100.0 * S[:1], **FIT_KW)
