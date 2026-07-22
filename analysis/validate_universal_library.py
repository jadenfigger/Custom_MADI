#!/usr/bin/env python3
"""
validate_universal_library.py — validation suite for the (δ,Δ,b)-universal
MADI library refactor (running position-integral Y(t) replacing fixed PGSE
moment windows).

Run in order; each test is a gate for the next (per the design task):
    1. Free diffusion            -- validates Y, the h-grid, eq.6, eq.5 together
    2. Impermeable sphere         -- validates restricted-diffusion plateau behaviour
    3. Permeation dt constraint   -- analytic/diagnostic check (independent of 1-2)
    4. Imaginary part             -- folded into test 1's output
    5. Grid convergence           -- halve h, halve dt, double N_w independently

Usage:
    conda activate mri
    PYTHONPATH=. python analysis/validate_universal_library.py

NOT covered here (flagged, not implemented — see final summary):
    - Full Kärger two-compartment exchange comparison (needs a dedicated
      two-cell exchange ensemble; the dt/permeation-probability piece of
      that check IS covered, as test 3).
    - Fitting-stage validation (out of scope per current task descope).
"""

from __future__ import annotations

import sys
import time
import numpy as np

from madi.config import SimConfig, grid_time_index
from madi import signal as sig
from madi.walker_gpu import run_walk_Y, kio_to_pp
from madi.ensemble import create_dummy_ensemble


def _hr(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ===========================================================================
# Test 1: free diffusion — Var(a), S(b)=exp(-b*D0), imaginary part
# ===========================================================================

def test_free_diffusion(D0=3.0, n_walkers=200_000, seed=123):
    _hr("TEST 1 — Free diffusion (no geometry)")
    cfg = SimConfig(D0=D0, n_walkers=n_walkers, n_ensembles=1, T_max_ms=100.0)
    ens = create_dummy_ensemble(cfg)

    pairs = [(2.0, 5.0), (5.0, 20.0), (10.0, 50.0), (20.0, 80.0)]
    b_values = [0.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]

    Y, n_esc = run_walk_Y(ens, kio=float("inf"), cfg=cfg, seed=seed, verbose=False)
    n_kept = Y.shape[0]
    print(f"  walkers kept: {n_kept}  escaped: {n_esc}")

    # --- Var(a) vs 2*D0*(Delta - delta/3) ---
    print(f"\n  {'delta':>6} {'Delta':>6} {'Var(a) MC':>12} {'analytic':>12} "
          f"{'rel err':>9}")
    var_ok = True
    for delta, Delta in pairs:
        jd = grid_time_index(delta, cfg.h_ms)
        jD = grid_time_index(Delta, cfg.h_ms)
        js = grid_time_index(Delta + delta, cfg.h_ms)
        dM = Y[:, jd, :] + Y[:, jD, :] - Y[:, js, :]
        a = (dM / (-delta)).ravel()
        var_mc = a.var()
        analytic = 2.0 * D0 * (Delta - delta / 3.0)
        rel = abs(var_mc - analytic) / analytic
        var_ok &= rel < 0.02
        print(f"  {delta:6.1f} {Delta:6.1f} {var_mc:12.4f} {analytic:12.4f} "
              f"{rel:9.4f}")

    # --- S(b) vs exp(-b[s/mm^2] * D0[mm^2/s]) ---
    # D0 is given in um^2/ms; the textbook free-water value ~3e-3 mm^2/s
    # corresponds to D0=3 um^2/ms, and the *matched-units* Stejskal-Tanner
    # form S=exp(-b*D) uses b[s/mm^2] with D[mm^2/s] directly (standard DWI
    # convention) -- NOT b/1e6 (that was an early mistake caught during
    # development; see conversation notes / git history).
    D0_mm2_s = D0 * 1e-3
    cols = sig.build_columns(cfg, delta_pairs=pairs, b_values=b_values)
    res = sig.compute_signals(0, 0, float("inf"), cfg=cfg, columns=cols,
                               seed=seed, verbose=False)
    S = res["S"]
    sigma_mc = 1.0 / np.sqrt(res["n_eff"])
    print(f"\n  sigma_mc = {sigma_mc:.5f}  (n_eff={res['n_eff']})")
    print(f"  {'delta':>6} {'Delta':>6} {'b':>8} {'S_mc':>10} {'analytic':>10} "
          f"{'|diff|/sig':>11}")
    worst = 0.0
    for pi, (delta, Delta) in enumerate(pairs):
        for bi, b in enumerate(b_values):
            analytic = np.exp(-b * D0_mm2_s)
            smc = S[pi, bi]
            nsig = abs(smc - analytic) / sigma_mc
            worst = max(worst, nsig)
            print(f"  {delta:6.1f} {Delta:6.1f} {b:8.0f} {smc:10.5f} "
                  f"{analytic:10.5f} {nsig:11.2f}")

    s_ok = worst < 4.0  # generous: ~28 comparisons, expect a few sigma tail
    imag_max = np.abs(res["S_imag"]).max()
    imag_ok = imag_max < 5 * sigma_mc
    print(f"\n  worst |S_mc - analytic| / sigma_mc = {worst:.2f}  "
          f"({'PASS' if s_ok else 'FAIL'}, threshold 4.0)")
    print(f"  TEST 4 (imaginary part) max|S_imag| = {imag_max:.5f}  "
          f"vs sigma_mc={sigma_mc:.5f}  ({'PASS' if imag_ok else 'FAIL'})")

    passed = var_ok and s_ok and imag_ok
    print(f"\n  TEST 1 (+4): {'PASS' if passed else 'FAIL'}")
    return passed


# ===========================================================================
# Test 2: impermeable sphere — long-time Var(a) plateau
# ===========================================================================

def _sphere_walk_Y(R, D0, T_max_ms, ts, h_ms, n_walkers, seed):
    """Reflecting-sphere MC walk, INDEPENDENT of the Voronoi tissue code —
    used only to check the Y(t)/eq.6/eq.5 pipeline against a case with a
    known exact long-time limit.

    Boundary handling: revert the step if it would exit the sphere (same
    "revert on rejected crossing" pattern the tissue kernel already uses
    for pp=0 impermeable membranes) — an approximate reflecting condition,
    consistent with this codebase's existing MC scheme rather than an
    idealized specular reflection.

    Exact long-time limit: as Delta -> infinity the walker position becomes
    uniformly distributed in the ball, so Var(one Cartesian coordinate) ->
    R^2/5 (standard result for a uniform distribution in a 3-ball), hence
    Var(a) = Var(x2bar - x1bar) -> 2*R^2/5 (independent samples once
    Delta >> R^2/D0).
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * D0 * ts)
    n_steps = int(round(T_max_ms / ts))
    n_grid = int(round(T_max_ms / h_ms)) + 1
    steps_per_h = int(round(h_ms / ts))

    pos = np.empty((n_walkers, 3))
    n_filled = 0
    while n_filled < n_walkers:
        cand = rng.uniform(-R, R, (max(2 * (n_walkers - n_filled), 64), 3))
        inside = (cand ** 2).sum(axis=1) <= R * R
        cand = cand[inside]
        take = min(len(cand), n_walkers - n_filled)
        pos[n_filled:n_filled + take] = cand[:take]
        n_filled += take

    Y = np.zeros((n_walkers, n_grid, 3), dtype=np.float64)
    Y_run = np.zeros((n_walkers, 3), dtype=np.float64)
    xs_prev = pos.copy()

    for step in range(n_steps):
        dx = rng.normal(0.0, sigma, (n_walkers, 3))
        proposed = pos + dx
        outside = (proposed ** 2).sum(axis=1) > R * R
        proposed[outside] = pos[outside]
        pos = proposed

        Y_run += 0.5 * (xs_prev + pos) * ts
        xs_prev = pos

        step_idx = step + 1
        if step_idx % steps_per_h == 0:
            Y[:, step_idx // steps_per_h, :] = Y_run

    return Y


def test_impermeable_sphere(R=8.0, D0=3.0, n_walkers=10_000, seed=7):
    _hr("TEST 2 — Impermeable sphere (long-time Var(a) plateau)")
    tau = R * R / D0
    print(f"  R={R} um, D0={D0} um^2/ms, tau=R^2/D0={tau:.1f} ms")

    delta = 2.0
    Deltas = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 140.0]
    h_ms = 1.0
    ts = 1e-3
    T_max = max(Deltas) + delta

    t0 = time.time()
    Y = _sphere_walk_Y(R, D0, T_max, ts, h_ms, n_walkers, seed)
    print(f"  ({time.time()-t0:.1f}s, {n_walkers} walkers, "
          f"{int(round(T_max/ts))} steps)")

    var_free_limit_ok = True
    plateau_analytic = 2.0 * R * R / 5.0
    print(f"\n  {'Delta':>7} {'Var(a) MC':>12} {'free analytic':>14} "
          f"{'plateau (R^2*2/5)':>18}")
    last_var = None
    for Delta in Deltas:
        jd = grid_time_index(delta, h_ms)
        jD = grid_time_index(Delta, h_ms)
        js = grid_time_index(Delta + delta, h_ms)
        dM = Y[:, jd, :] + Y[:, jD, :] - Y[:, js, :]
        a = (dM / (-delta)).ravel()
        var_mc = a.var()
        free_analytic = 2.0 * D0 * (Delta - delta / 3.0)
        last_var = var_mc
        print(f"  {Delta:7.1f} {var_mc:12.4f} {free_analytic:14.4f} "
              f"{plateau_analytic:18.4f}")
        if Delta <= 5.0:
            # short time: should still track free diffusion (hasn't felt
            # the boundary yet)
            var_free_limit_ok &= abs(var_mc - free_analytic) / free_analytic < 0.05

    rel_plateau = abs(last_var - plateau_analytic) / plateau_analytic
    plateau_ok = rel_plateau < 0.10
    restricted_ok = last_var < 2.0 * D0 * (Deltas[-1] - delta / 3.0) * 0.5
    print(f"\n  short-Delta tracks free diffusion: "
          f"{'PASS' if var_free_limit_ok else 'FAIL'}")
    print(f"  long-Delta Var(a) within 10% of 2R^2/5 plateau: "
          f"{rel_plateau*100:.1f}%  ({'PASS' if plateau_ok else 'FAIL'})")
    print(f"  long-Delta Var(a) well below unrestricted prediction "
          f"(restriction is real): {'PASS' if restricted_ok else 'FAIL'}")

    passed = var_free_limit_ok and plateau_ok and restricted_ok
    print(f"\n  TEST 2: {'PASS' if passed else 'FAIL'}")
    return passed


# ===========================================================================
# Test 3: permeation dt / max-transmission-probability diagnostic
# ===========================================================================

def test_permeation_dt_constraint():
    _hr("TEST 3 — Permeation dt constraint (diagnostic, highest-risk item)")
    print("  This checks whether the EXISTING membrane-crossing model "
          "(pre-dating this refactor, unchanged here) stays in its valid "
          "regime for the ts/kio combinations this library actually uses.")
    print("  NOTE: this is a diagnostic against commonly-cited MC-permeation")
    print("  literature thresholds (crossing probability per attempt should")
    print("  stay well below ~0.3-0.5 for the discrete-time model to")
    print("  reproduce the continuum membrane permeability kio); it is NOT")
    print("  a full Karger-model exchange comparison (not implemented here —")
    print("  would need a dedicated two-compartment exchange ensemble).")

    cfg = SimConfig()
    kios = [2, 8, 25, 50, 75, 100, 150, 250]
    mean_AVs = [0.2, 0.5, 1.0, 2.0]   # um^-1, plausible range for these cell sizes

    print(f"\n  ts={cfg.ts} ms, D0={cfg.D0} um^2/ms")
    print(f"  {'kio [1/s]':>10} {'mean_AV':>8} {'pp':>10}  flag")
    worst_pp = 0.0
    for kio in kios:
        for mav in mean_AVs:
            pp = kio_to_pp(kio, mav, cfg)
            worst_pp = max(worst_pp, pp)
            flag = ""
            if pp > 0.5:
                flag = "*** pp>0.5, discrete-time approximation likely biased ***"
            elif pp > 0.3:
                flag = "pp>0.3, approaching the commonly-cited caution zone"
            print(f"  {kio:10.0f} {mav:8.2f} {pp:10.4f}  {flag}")

    ok = worst_pp < 0.5
    print(f"\n  worst-case pp over this grid = {worst_pp:.4f}  "
          f"({'within caution zone' if ok else 'EXCEEDS 0.5 — revisit ts or kio range'})")
    print("  Recommendation: cross-check this threshold against the source "
          "paper's SI derivation of the permeation rule directly (not done "
          "here) before trusting kio values at the high end of any preset.")
    return ok


# ===========================================================================
# Test 5: grid convergence — halve h, halve dt, double N_w (independently)
# ===========================================================================

def test_grid_convergence(D0=3.0, n_walkers=50_000, seed=99):
    _hr("TEST 5 — Grid convergence (h, dt, N_w independently)")
    pairs = [(4.0, 20.0)]
    b_values = [500.0, 2000.0]

    def run(h_ms, ts, nw, T_max=60.0):
        cfg = SimConfig(D0=D0, n_walkers=nw, n_ensembles=1,
                         T_max_ms=T_max, h_ms=h_ms, ts=ts)
        ens = create_dummy_ensemble(cfg)
        cols = sig.build_columns(cfg, delta_pairs=pairs, b_values=b_values)
        res = sig.compute_signals(0, 0, float("inf"), cfg=cfg, columns=cols,
                                   seed=seed, verbose=False)
        return res["S"], 1.0 / np.sqrt(res["n_eff"])

    S_base, sig_base = run(h_ms=1.0, ts=1e-3, nw=n_walkers)

    print(f"  baseline: h=1.0ms, ts=1e-3ms, N_w={n_walkers}  sigma_mc={sig_base:.5f}")
    print(f"  baseline S: {S_base.ravel()}")

    all_ok = True
    for label, kw in [
        ("h halved (0.5ms)", dict(h_ms=0.5, ts=1e-3, nw=n_walkers)),
        ("dt halved (5e-4ms)", dict(h_ms=1.0, ts=5e-4, nw=n_walkers)),
        ("N_w doubled", dict(h_ms=1.0, ts=1e-3, nw=n_walkers * 2)),
    ]:
        S_alt, sig_alt = run(**kw)
        diff = np.abs(S_alt - S_base)
        combined_sigma = np.sqrt(sig_base**2 + sig_alt**2)
        nsig = (diff / combined_sigma).max()
        ok = nsig < 4.0
        all_ok &= ok
        print(f"\n  {label}: sigma_mc={sig_alt:.5f}  "
              f"max|dS|/combined_sigma={nsig:.2f}  ({'PASS' if ok else 'FAIL'})")
        print(f"    S: {S_alt.ravel()}")

    print(f"\n  TEST 5: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ===========================================================================
# Main
# ===========================================================================

def main():
    results = {}
    results["1+4 free diffusion"] = test_free_diffusion()
    results["2 impermeable sphere"] = test_impermeable_sphere()
    results["3 permeation dt"] = test_permeation_dt_constraint()
    results["5 grid convergence"] = test_grid_convergence()

    _hr("SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL':6s}  {name}")
    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("\n  NOT covered (flagged, not implemented):")
    print("    - Full Karger two-compartment exchange comparison")
    print("    - Fitting-stage validation (descoped)")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
