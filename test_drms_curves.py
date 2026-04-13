"""
test_drms_curves.py — Unit Test 3 from the MADI Verification Checklist
======================================================================

Verifies that root-mean-square displacement curves d_rms(t) produced by
the random-walk machinery match the family of curves shown in
Springer et al. NMR in Biomedicine 2023;36:e4781, **Figure 3** (panels
a, b/e, c, d/f).

Reference parameters (from Fig. 3 of MADI I):
    ρ = 781,000 cells/μL
    V = 1.0 pL
    v_i = ρ·V = 0.781
    D_0 = 3.0 μm²/ms (37 °C)
    k_io family: {0, 12, 34, 63, 128} s⁻¹

What this test exercises in YOUR codebase
-----------------------------------------
The walker in `madi.walker_gpu` only outputs encoding moments (`dM`),
not raw positions. So this test imports your geometry / config /
permeability conversion and runs an INSTRUMENTED random walk in this
file that records positions at intermediate timesteps. The instrumented
walker is a line-for-line port of `madi.walker_gpu._walk_cpu`'s step
loop — same proposal distribution, same crossing classification, same
freeze-on-escape handling, same p_p^m rule — so any bug in:

    * Ensemble geometry              (annulus widths, mean_AV)
    * Voxel-grid compartment lookup  (ens.classify_cpu)
    * cfg.sigma derivation           (must be sqrt(2·D0·ts))
    * kio_to_pp conversion           (Eq. 5 of paper)
    * Boundary / escape handling     (must freeze, not reflect)

will show up as a failed test below.

What this test does NOT exercise
--------------------------------
* The CUDA kernel itself (the Numba `_walk_kernel`).
  Cross-validate with a separate test that compares CPU and GPU
  encoding moments for matched parameters.
* The SDE encoding moment accumulation in walker_gpu (M1, M2, dM).
  Use a separate test that compares your library's pure-water S(b)/S0
  curve to exp(−b·D0).

Runtime
-------
First run will trigger a one-time lookup-table build inside
`madi.ensemble._load_or_build_lookup_table` (~10–20 min, cached to
~/.cache/madi/). Subsequent runs reuse the cache.

After the cache is warm, this whole file runs in roughly:
    quick mode  (default):  ~60–90 s on CPU
    --full mode:            ~5–8 min on CPU

Usage
-----
As a pytest module:

    pytest test_drms_curves.py -v

As a script (also produces a Fig. 3-style plot):

    python test_drms_curves.py
    python test_drms_curves.py --full
    python test_drms_curves.py --no-plot
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from typing import Iterable

import numpy as np

# Make sure we can import the user's madi package whether this file lives
# at the repo root (alongside run_simulation.py) or one level above.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from madi.config import SimConfig, D0_UM2_MS
from madi.ensemble import (
    Ensemble,
    create_ensemble,
    create_dummy_ensemble,
)
from madi.walker_gpu import kio_to_pp


# ---------------------------------------------------------------------------
# Test configuration knobs
# ---------------------------------------------------------------------------

# Fig. 3 of Paper I uses these geometric parameters
RHO_FIG3 = 781_000.0   # cells/μL
V_FIG3   = 1.0         # pL
KIO_FAMILY_FULL  = [0.0, 12.0, 34.0, 63.0, 128.0]   # s⁻¹  (paper)
KIO_FAMILY_QUICK = [0.0, 30.0, 100.0]               # s⁻¹  (subset for speed)


def _build_test_cfg(quick: bool = True) -> SimConfig:
    """Construct a SimConfig sized for unit testing (not production library)."""
    if quick:
        n_walkers = 3000     # SE on <d²> ≈ 2.6%  → 5% tolerances are safe
        n_steps   = 25_000   # 25 ms walks (well past intracellular plateau ~5 ms for V=1 pL)
    else:
        n_walkers = 12_000
        n_steps   = 60_000   # 60 ms — comfortably reaches paper's well-mixed regime

    return SimConfig(
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_ensembles=1,    # one ensemble per condition is plenty for d_rms
        # everything else uses package defaults
    )


# ---------------------------------------------------------------------------
# Position-recording walker (mirrors madi.walker_gpu._walk_cpu step-for-step)
# ---------------------------------------------------------------------------

def _instrumented_walk(
    ens: Ensemble,
    pp: float,
    cfg: SimConfig,
    t_record_ms: Iterable[float],
    seed: int,
) -> dict:
    """Run RWs and record per-walker positions at the requested timestamps.

    This is intentionally a near-verbatim port of `_walk_cpu` from
    `madi/walker_gpu.py`. The only differences are:

      * No moment accumulation (M1, M2, dM).
      * Position snapshots are taken at requested timesteps.
      * We track and return per-walker initial compartment so the
        caller can split intracellular vs. extracellular trajectories.

    Returns a dict with:
        times_ms       : (n_record,) recorded times
        drms2_pooled   : (n_record,) <|r-r0|²>
        drms2_intra    : (n_record,) <|r-r0|²> over walkers that started in cells
        drms2_extra    : (n_record,) <|r-r0|²> over walkers that started in interstitium
        slope_pooled   : (n_record,) instantaneous d/dt of <|r-r0|²>/6  (≈ apparent D)
        slope_intra    : (n_record,) same, intracellular
        slope_extra    : (n_record,) same, extracellular
        n_intra, n_extra
        mean_AV, pp
        n_escaped
    """
    rng = np.random.default_rng(seed)
    N = cfg.n_walkers
    sigma = cfg.sigma
    ts = cfg.ts
    L = cfg.L
    lo = cfg.buffer
    hi = L - cfg.buffer

    # Spawn uniformly in Ω_src (matches walker_gpu _walk_cpu)
    positions = rng.uniform(lo, hi, (N, 3))
    initial_positions = positions.copy()
    cur_s1, cur_inside = ens.classify_cpu(positions)
    initial_inside = cur_inside.copy()
    frozen = np.zeros(N, dtype=bool)

    # --- Build record schedule
    record_steps = sorted({int(round(t / ts)) for t in t_record_ms if t > 0})
    record_steps = [s for s in record_steps if 0 < s <= cfg.n_steps]
    if not record_steps:
        raise ValueError("No valid record times within cfg.n_steps")

    n_rec = len(record_steps)
    drms2_pooled = np.zeros(n_rec)
    drms2_intra  = np.zeros(n_rec)
    drms2_extra  = np.zeros(n_rec)

    # For instantaneous slope estimation at each record time we need the
    # mean-square displacement at (record_step ± dwin) too. Easiest path:
    # store <|r-r0|²> at every requested step and compute centered finite
    # differences offline.
    rec_idx = 0
    next_record = record_steps[0]

    for step in range(1, cfg.n_steps + 1):
        active = ~frozen
        if not active.any():
            break

        # 1. Propose Gaussian displacement (per-component σ = sqrt(2·D0·ts))
        dx = rng.normal(0.0, sigma, (N, 3))
        proposed = positions + dx

        # 2. Boundary escape → freeze (Fix #2 in walker_gpu.py)
        oob = (proposed < 0.0).any(axis=1) | (proposed >= L).any(axis=1)
        newly_frozen = oob & active
        if newly_frozen.any():
            frozen |= newly_frozen
            proposed[newly_frozen] = positions[newly_frozen]
            active = ~frozen

        # 3. Compartment classification of proposed positions
        new_s1, new_inside = ens.classify_cpu(proposed)

        # 4. Membrane crossing count m ∈ {0, 1, 2}
        m = np.zeros(N, dtype=np.int32)
        m[cur_inside & ~new_inside] = 1
        m[~cur_inside & new_inside] = 1
        m[cur_inside & new_inside & (cur_s1 != new_s1)] = 2

        crossing = (m > 0) & active
        if crossing.any() and pp < 1.0:
            u = rng.uniform(0.0, 1.0, int(crossing.sum()))
            pp_m = pp ** m[crossing]
            rej_sub = u >= pp_m
            rej_idx = np.where(crossing)[0][rej_sub]
            proposed[rej_idx] = positions[rej_idx]
            new_s1[rej_idx] = cur_s1[rej_idx]
            new_inside[rej_idx] = cur_inside[rej_idx]

        # 5. Commit
        positions = proposed
        cur_s1 = np.where(active, new_s1, cur_s1)
        cur_inside = np.where(active, new_inside, cur_inside)

        # 6. Snapshot
        if step == next_record:
            keep = ~frozen
            d = positions[keep] - initial_positions[keep]
            d2 = (d * d).sum(axis=1)
            ii = initial_inside[keep]

            drms2_pooled[rec_idx] = d2.mean()
            if ii.any():
                drms2_intra[rec_idx] = d2[ii].mean()
            if (~ii).any():
                drms2_extra[rec_idx] = d2[~ii].mean()

            rec_idx += 1
            next_record = record_steps[rec_idx] if rec_idx < n_rec else cfg.n_steps + 1

    times_ms = np.array(record_steps, dtype=float) * ts

    # Centered-difference instantaneous slopes / 6  (≡ apparent D)
    def _slope(arr: np.ndarray) -> np.ndarray:
        s = np.zeros_like(arr)
        if len(arr) >= 3:
            s[1:-1] = (arr[2:] - arr[:-2]) / (times_ms[2:] - times_ms[:-2]) / 6.0
            s[0]  = (arr[1] - arr[0]) / (times_ms[1] - times_ms[0]) / 6.0
            s[-1] = (arr[-1] - arr[-2]) / (times_ms[-1] - times_ms[-2]) / 6.0
        elif len(arr) == 2:
            s[:] = (arr[-1] - arr[0]) / (times_ms[-1] - times_ms[0]) / 6.0
        return s

    return {
        "times_ms": times_ms,
        "drms2_pooled": drms2_pooled,
        "drms2_intra": drms2_intra,
        "drms2_extra": drms2_extra,
        "drms_pooled": np.sqrt(drms2_pooled),
        "drms_intra": np.sqrt(drms2_intra),
        "drms_extra": np.sqrt(drms2_extra),
        "slope_pooled": _slope(drms2_pooled),
        "slope_intra": _slope(drms2_intra),
        "slope_extra": _slope(drms2_extra),
        "n_intra": int(initial_inside.sum()),
        "n_extra": int((~initial_inside).sum()),
        "mean_AV": ens.mean_AV,
        "pp": pp,
        "n_escaped": int(frozen.sum()),
    }


def _run_condition(rho: float, V: float, kio: float,
                   cfg: SimConfig, t_record_ms: Iterable[float],
                   seed: int = 0) -> dict:
    """Build one ensemble for (rho,V), convert kio→pp, run instrumented walk."""
    if rho <= 0 or V <= 0:
        ens = create_dummy_ensemble(cfg)
        pp = 1.0
    else:
        ens = create_ensemble(rho, V, cfg, seed=seed, verbose=False)
        if not np.isfinite(kio):
            pp = 1.0
        else:
            pp = float(np.clip(kio_to_pp(kio, ens.mean_AV, cfg), 0.0, 1.0))
    return _instrumented_walk(ens, pp, cfg, t_record_ms, seed=seed + 1)


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

# A small module-level cache so that tests that share the same ensemble
# don't rebuild it. Pytest discovers tests in declared order; we exploit that.
_RESULT_CACHE: dict = {}


def test_pure_water_drms2_equals_6_D0_t():
    """Sanity test: with no cells (dummy ensemble) and no membranes, the
    walker must produce <|r-r0|²> = 6·D0·t (Gaussian diffusion).

    This is the strictest possible test of:
        - cfg.sigma = sqrt(2·D0·ts)  (per-component variance)
        - the proposal distribution being a true 3D normal
        - the walker not silently scaling positions (units)

    Reference: Paper I Fig. 3a, dashed black "H₂O free" line, slope = 6·D0.
    """
    cfg = _build_test_cfg(quick=True)
    record_ms = np.linspace(2.0, cfg.tRW_max, 12)

    res = _run_condition(rho=0.0, V=0.0, kio=np.inf, cfg=cfg,
                         t_record_ms=record_ms, seed=11)

    expected = 6.0 * D0_UM2_MS * res["times_ms"]   # μm²
    measured = res["drms2_pooled"]

    # Linear regression through origin: measured = α · t
    alpha, _, _, _ = np.linalg.lstsq(
        res["times_ms"][:, None], measured, rcond=None
    )
    slope = float(alpha[0])
    expected_slope = 6.0 * D0_UM2_MS

    rel_err_slope = abs(slope - expected_slope) / expected_slope
    assert rel_err_slope < 0.05, (
        f"Pure-water d_rms² slope = {slope:.3f} μm²/ms, "
        f"expected 6·D0 = {expected_slope:.3f} (off by {rel_err_slope*100:.1f}%). "
        f"Check cfg.sigma derivation and the Gaussian proposal in _walk_cpu / _walk_kernel."
    )

    # And every individual sample point should be close
    rel_err_pointwise = np.abs(measured - expected) / expected
    assert rel_err_pointwise.max() < 0.10, (
        f"Pure-water d_rms² deviates from 6·D0·t by up to "
        f"{rel_err_pointwise.max()*100:.1f}% at some sample point.\n"
        f"  times: {res['times_ms']}\n"
        f"  meas : {measured}\n"
        f"  expt : {expected}"
    )


def test_no_exchange_intracellular_plateaus():
    """With ρ=781k, V=1.0 pL, k_io=0: walkers that started inside cells
    cannot leave. Their <|r-r0|²> must approach a finite plateau.

    Reference: Paper I Fig. 3a, blue H₂O_in curve;
               SI Eq. 6:  t_RW(well-mixed) > V^(2/3)/(17·D0).
               For V=1.0 pL = 1000 μm³, this is ≈ 100/(17·3) ≈ 2 ms.

    The plateau height for a roughly spherical cell of volume V is
    ≲ 2·R² where R = (3V/4π)^(1/3) ≈ 6.2 μm → drms² ≲ 80 μm².
    """
    cfg = _build_test_cfg(quick=True)
    record_ms = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, cfg.tRW_max])

    res = _run_condition(rho=RHO_FIG3, V=V_FIG3, kio=0.0, cfg=cfg,
                         t_record_ms=record_ms, seed=21)
    _RESULT_CACHE["no_exchange"] = res

    assert res["n_intra"] > 100, (
        f"Only {res['n_intra']} walkers started intracellularly; bump n_walkers"
    )

    early_intra = res["drms2_intra"][res["times_ms"] <= 5.0]
    late_intra  = res["drms2_intra"][res["times_ms"] >= 15.0]
    assert early_intra.size > 0 and late_intra.size > 0

    early_max = early_intra.max()
    late_mean = late_intra.mean()
    late_growth_ratio = late_mean / early_max

    # The plateau should be reached: late values shouldn't exceed early max
    # by more than ~50% (a fair amount of slack for finite-cell shape variation).
    assert late_growth_ratio < 1.6, (
        f"Intracellular d_rms² is still growing strongly at late times "
        f"(late/early max ratio = {late_growth_ratio:.2f}). "
        f"Either k_io != 0 leaked through (check kio_to_pp returning 0 for kio=0) "
        f"or membrane permeation is broken (m=1/m=2 detection)."
    )

    # Plateau magnitude sanity: <|r-r0|²> should be ≲ 2·R² with some slack.
    R = (3.0 * V_FIG3 * 1e3 / (4.0 * np.pi)) ** (1.0 / 3.0)   # μm
    plateau_upper = 4.0 * R * R   # generous upper bound for non-spherical cells
    assert late_mean < plateau_upper, (
        f"Intracellular plateau <d²>={late_mean:.1f} μm² exceeds "
        f"4R² = {plateau_upper:.1f} μm² for V={V_FIG3} pL. "
        f"Walkers may be escaping cells through annulus geometry."
    )


def test_no_exchange_extracellular_diffusivity_reduced():
    """At k_io=0, extracellular walkers diffuse but with an apparent
    diffusivity D'_out < D_0 due to tortuosity (membrane reflection).

    Paper I Fig. 3b shows D'_out limiting near 1.8 μm²/ms at ρ=781k,
    V=1.0 pL  (cf. D_0 = 3.0 μm²/ms).
    """
    res = _RESULT_CACHE.get("no_exchange")
    if res is None:
        cfg = _build_test_cfg(quick=True)
        record_ms = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, cfg.tRW_max])
        res = _run_condition(RHO_FIG3, V_FIG3, 0.0, cfg, record_ms, seed=21)
        _RESULT_CACHE["no_exchange"] = res

    assert res["n_extra"] > 50, (
        f"Only {res['n_extra']} walkers started extracellularly; "
        f"increase n_walkers or v_i may be too high."
    )

    # Late-time slope of <|r-r0|²>/6 = apparent D
    # Use a linear fit over t > 5 ms
    late_mask = res["times_ms"] >= 5.0
    t_late = res["times_ms"][late_mask]
    d2_late = res["drms2_extra"][late_mask]

    if t_late.size < 2:
        # Quick mode may be too short — just check the last available point
        D_apparent = res["slope_extra"][-1]
    else:
        # Slope from linear regression of <d²> vs t  (no intercept constraint)
        A = np.vstack([t_late, np.ones_like(t_late)]).T
        slope_d2, _ = np.linalg.lstsq(A, d2_late, rcond=None)[0]
        D_apparent = slope_d2 / 6.0

    # Should be strictly less than D_0 (tortuosity), and not crazy small
    assert D_apparent < D0_UM2_MS, (
        f"Extracellular apparent D = {D_apparent:.2f} ≥ D_0 = {D0_UM2_MS:.2f}. "
        f"Either membranes aren't reflecting (pp leak?) or extracellular "
        f"walkers have nothing in the way (mean_AV too small)."
    )
    assert D_apparent > 0.3 * D0_UM2_MS, (
        f"Extracellular apparent D = {D_apparent:.2f} is far below paper's "
        f"~1.8 μm²/ms — extracellular space may be over-tortuous "
        f"(annulus widths too narrow) or walkers are getting trapped."
    )


def test_kio_family_drms_monotone_in_kio():
    """Pooled <d²> at fixed late time must increase monotonically with k_io.
    Reference: Paper I Fig. 3c — the curve family fans upward as k_io grows.
    """
    cfg = _build_test_cfg(quick=True)
    record_ms = np.array([1.0, 5.0, 10.0, 15.0, 20.0, cfg.tRW_max])

    kio_values = KIO_FAMILY_QUICK
    family = {}
    for k in kio_values:
        family[k] = _run_condition(RHO_FIG3, V_FIG3, k, cfg, record_ms, seed=31)
    _RESULT_CACHE["family_quick"] = family

    # Late-time pooled <d²>
    late_d2 = {k: family[k]["drms2_pooled"][-1] for k in kio_values}

    # Strict monotone increase with some statistical slack
    sorted_kio = sorted(kio_values)
    prev = -np.inf
    for k in sorted_kio:
        d2 = late_d2[k]
        # Allow within-statistical-noise non-monotonicity at small kio steps
        # (~3% standard error on <d²> for N=3000 walkers)
        assert d2 > prev * 0.97, (
            f"Pooled <d²> at t={record_ms[-1]} ms is not monotone in k_io: "
            f"k_io={k} → {d2:.2f}, but a smaller k_io produced {prev:.2f}. "
            f"Either kio_to_pp is wrong-signed, or pp is being clipped to 1 "
            f"and high-kio behaves identically to free water."
        )
        prev = d2

    # Sanity: the highest k_io should noticeably exceed the lowest
    span = late_d2[max(kio_values)] / max(late_d2[min(kio_values)], 1e-9)
    assert span > 1.10, (
        f"Late-time <d²> at k_io={max(kio_values)} is only "
        f"{span:.2f}× that at k_io={min(kio_values)}. Expected fan-out per "
        f"Fig. 3c. kio_to_pp may be miscalibrated (mean_AV factor wrong?)."
    )


def test_kio_family_apparent_D_monotone_and_below_D0():
    """The pooled apparent diffusion coefficient D'_WM (late-time slope of
    <d²>/6) must (a) increase with k_io and (b) lie strictly below D_0.

    Reference: Paper I Fig. 3d/f — D family fans up with k_io but never
    reaches D_0 (since walkers still hit membranes).
    """
    family = _RESULT_CACHE.get("family_quick")
    if family is None:
        cfg = _build_test_cfg(quick=True)
        record_ms = np.array([1.0, 5.0, 10.0, 15.0, 20.0, cfg.tRW_max])
        family = {
            k: _run_condition(RHO_FIG3, V_FIG3, k, cfg, record_ms, seed=31)
            for k in KIO_FAMILY_QUICK
        }
        _RESULT_CACHE["family_quick"] = family

    # Compute late-time slope D from a linear fit of <d²> vs t over t≥5 ms
    def _late_D(res):
        mask = res["times_ms"] >= 5.0
        if mask.sum() < 2:
            return res["slope_pooled"][-1]
        t = res["times_ms"][mask]
        y = res["drms2_pooled"][mask]
        A = np.vstack([t, np.ones_like(t)]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope / 6.0

    D_by_kio = {k: _late_D(family[k]) for k in sorted(family.keys())}

    # (a) Monotone increase  (with the same 3% slack as above)
    prev = -np.inf
    for k in sorted(D_by_kio.keys()):
        D = D_by_kio[k]
        assert D > prev * 0.97, (
            f"Apparent D not monotone in k_io: k_io={k} → D={D:.3f} μm²/ms, "
            f"smaller k_io gave D={prev:.3f}."
        )
        prev = D

    # (b) Strict upper bound: D'_WM ≤ D_0 (tortuosity always slows things down).
    # We allow 5% statistical slack — at very large k_io, D'_WM approaches D_0.
    for k, D in D_by_kio.items():
        assert D < 1.05 * D0_UM2_MS, (
            f"k_io={k} gives apparent D={D:.3f} μm²/ms, exceeding D_0={D0_UM2_MS:.3f}. "
            f"This is unphysical — membranes can never speed diffusion above "
            f"the free-water limit."
        )


def test_no_escape_in_short_walks():
    """Ω_sim should be sized so that ~no walkers escape during a 25 ms walk
    at default cfg. If many escape, cfg.L / cfg.buffer are too small.

    Reference: SI §S.III — paper aborts on any escape; this codebase uses
    cfg.max_escape_frac as a tolerance threshold.
    """
    res = _RESULT_CACHE.get("no_exchange")
    if res is None:
        cfg = _build_test_cfg(quick=True)
        record_ms = np.array([cfg.tRW_max])
        res = _run_condition(RHO_FIG3, V_FIG3, 0.0, cfg, record_ms, seed=21)
        _RESULT_CACHE["no_exchange"] = res

    cfg = _build_test_cfg(quick=True)
    n_total = cfg.n_walkers
    frac = res["n_escaped"] / n_total
    assert frac < 0.02, (
        f"{frac*100:.1f}% of walkers escaped Ω_sim during a {cfg.tRW_max} ms walk. "
        f"This is high enough to bias d_rms estimates. Increase cfg.L or cfg.buffer."
    )


# ---------------------------------------------------------------------------
# __main__ : run all tests verbosely + produce Fig. 3-style plot
# ---------------------------------------------------------------------------

def _make_figure(family: dict, no_exchange: dict, save_path: str = None):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left panel: d_rms vs t for the kio family (Paper Fig. 3c analog)
    ax = axes[0]
    cmap = plt.cm.viridis
    sorted_kio = sorted(family.keys())
    for i, k in enumerate(sorted_kio):
        res = family[k]
        color = cmap(i / max(1, len(sorted_kio) - 1))
        ax.plot(res["times_ms"], res["drms_pooled"], "o-",
                color=color, ms=4, label=f"k_io = {k:g} s⁻¹")

    # Free-water reference
    t_ref = np.linspace(0.1, max(res["times_ms"]), 100)
    ax.plot(t_ref, np.sqrt(6.0 * D0_UM2_MS * t_ref), "k--", lw=1,
            alpha=0.5, label="H₂O free (slope 6·D₀)")

    ax.set_xlabel("t_RW  [ms]")
    ax.set_ylabel("d_rms  [μm]")
    ax.set_title(f"Pooled d_rms vs t  (ρ={RHO_FIG3/1e3:.0f}k, V={V_FIG3:.1f} pL)\n"
                 f"≈ Paper I Fig. 3c")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Right panel: instantaneous slope/6 = D vs t (Paper Fig. 3d/f analog)
    ax = axes[1]
    for i, k in enumerate(sorted_kio):
        res = family[k]
        color = cmap(i / max(1, len(sorted_kio) - 1))
        ax.plot(res["times_ms"], res["slope_pooled"], "o-",
                color=color, ms=4, label=f"k_io = {k:g}")

    # Show D_in and D_out separately for the no-exchange case
    if no_exchange is not None:
        ax.plot(no_exchange["times_ms"], no_exchange["slope_intra"],
                "s--", color="navy", ms=4, alpha=0.6,
                label="D_in (k_io=0, intra only)")
        ax.plot(no_exchange["times_ms"], no_exchange["slope_extra"],
                "^--", color="darkorange", ms=4, alpha=0.6,
                label="D_out (k_io=0, extra only)")

    ax.axhline(D0_UM2_MS, color="k", linestyle=":", alpha=0.5,
               label=f"D₀ = {D0_UM2_MS:.1f}")
    ax.set_xlabel("t_RW  [ms]")
    ax.set_ylabel("Apparent D = d⟨d²⟩/dt / 6  [μm²/ms]")
    ax.set_title("Apparent D vs t   ≈ Paper I Fig. 3d/f")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.1)

    fig.suptitle(
        "Unit Test 3 — d_rms verification against Springer et al. NMRBM 2023 Fig. 3",
        fontsize=11,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved figure → {save_path}")
    else:
        plt.show()


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Use more walkers and longer walks (~5–8 min)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--save-plot", default="test_drms_fig3.png",
                        help="Output path for the diagnostic plot")
    args = parser.parse_args()

    quick = not args.full
    print(f"\n{'='*70}")
    print(f" MADI d_rms verification ({'quick' if quick else 'full'} mode)")
    print(f"{'='*70}\n")

    # ---- Build cfg, warn about lookup-table cost on first run -------------
    cfg = _build_test_cfg(quick=quick)
    cache_dir = os.path.expanduser("~/.cache/madi")
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print("  ⚠ No MADI lookup table cache found in ~/.cache/madi.")
        print("    The first ensemble construction will trigger a ~10–20 min build.")
        print("    Subsequent runs are fast.\n")

    print(f"  cfg.n_walkers   = {cfg.n_walkers}")
    print(f"  cfg.n_steps     = {cfg.n_steps}  (tRW_max = {cfg.tRW_max} ms)")
    print(f"  cfg.sigma       = {cfg.sigma:.4f} μm  (per-axis Gaussian step)")
    print(f"  expected sigma  = {np.sqrt(2*cfg.D0*cfg.ts):.4f}  ({'OK' if abs(cfg.sigma - np.sqrt(2*cfg.D0*cfg.ts)) < 1e-12 else 'MISMATCH!'})")
    print(f"  cfg.L           = {cfg.L} μm,  buffer = {cfg.buffer} μm")
    print()

    # ---- 1. Pure water sanity ---------------------------------------------
    print("[1/6] test_pure_water_drms2_equals_6_D0_t ... ", end="", flush=True)
    t0 = time.time()
    try:
        test_pure_water_drms2_equals_6_D0_t()
        print(f"PASS ({time.time()-t0:.1f}s)")
    except AssertionError as e:
        print(f"FAIL ({time.time()-t0:.1f}s)\n    {e}")

    # ---- 2. Intracellular plateau -----------------------------------------
    print("[2/6] test_no_exchange_intracellular_plateaus ... ", end="", flush=True)
    t0 = time.time()
    try:
        test_no_exchange_intracellular_plateaus()
        print(f"PASS ({time.time()-t0:.1f}s)")
    except AssertionError as e:
        print(f"FAIL ({time.time()-t0:.1f}s)\n    {e}")

    # ---- 3. Extracellular reduced D ---------------------------------------
    print("[3/6] test_no_exchange_extracellular_diffusivity_reduced ... ",
          end="", flush=True)
    t0 = time.time()
    try:
        test_no_exchange_extracellular_diffusivity_reduced()
        print(f"PASS ({time.time()-t0:.1f}s)")
    except AssertionError as e:
        print(f"FAIL ({time.time()-t0:.1f}s)\n    {e}")

    # ---- 4. Monotone d_rms in kio -----------------------------------------
    print("[4/6] test_kio_family_drms_monotone_in_kio ... ", end="", flush=True)
    t0 = time.time()
    try:
        test_kio_family_drms_monotone_in_kio()
        print(f"PASS ({time.time()-t0:.1f}s)")
    except AssertionError as e:
        print(f"FAIL ({time.time()-t0:.1f}s)\n    {e}")

    # ---- 5. Monotone D in kio, bounded by D0 ------------------------------
    print("[5/6] test_kio_family_apparent_D_monotone_and_below_D0 ... ",
          end="", flush=True)
    t0 = time.time()
    try:
        test_kio_family_apparent_D_monotone_and_below_D0()
        print(f"PASS ({time.time()-t0:.1f}s)")
    except AssertionError as e:
        print(f"FAIL ({time.time()-t0:.1f}s)\n    {e}")

    # ---- 6. Escape rate ---------------------------------------------------
    print("[6/6] test_no_escape_in_short_walks ... ", end="", flush=True)
    t0 = time.time()
    try:
        test_no_escape_in_short_walks()
        print(f"PASS ({time.time()-t0:.1f}s)")
    except AssertionError as e:
        print(f"FAIL ({time.time()-t0:.1f}s)\n    {e}")

    # ---- Diagnostic plot --------------------------------------------------
    if not args.no_plot:
        family = _RESULT_CACHE.get("family_quick")
        no_exchange = _RESULT_CACHE.get("no_exchange")
        if family is not None:
            print(f"\nGenerating diagnostic plot...")
            _make_figure(family, no_exchange, save_path=args.save_plot)
        else:
            print("\n(skipping plot — no family results available)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    _main()
