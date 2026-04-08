"""
GPU-accelerated Monte Carlo random walks with semipermeable membranes.

KEY FIXES vs. the previous version:

  Fix 1 — powder average via 3 independent ensembles (one per axis):
      `run_simulation()` now builds 3 × n_ensembles ensembles per library
      entry.  n_ensembles of them are used to measure the x-axis signal,
      another n_ensembles for y, another n_ensembles for z.  Walker dM
      from each ensemble is projected onto its assigned axis only, so the
      three axis signals are built from statistically independent walker
      pools AND independent ensemble geometries.  `WalkResult` now carries
      `dM_per_axis`, a list of three arrays.

  Fix 2 — freeze-on-escape (not reflect-and-drop):
      Walkers that exit Ω_sim are FROZEN in place (no further position or
      moment updates), and are dropped from the output at the end of the
      run.  If the escape fraction exceeds cfg.max_escape_frac, run_walks()
      raises RuntimeError, matching the paper's "immediately cease"
      semantics (SI §S.III).

  Fix 5 — walker-count defaults bumped in config.py.
"""

from __future__ import annotations

import math
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from .ensemble import Ensemble, create_ensemble, create_dummy_ensemble
from .config   import SimConfig

# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------
try:
    from numba import cuda
    from numba.cuda.random import (create_xoroshiro128p_states,
                                   xoroshiro128p_normal_float64,
                                   xoroshiro128p_uniform_float64)
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False

if not HAS_CUDA:
    print("WARNING: CUDA not available — falling back to CPU (slow)")


# Paper recommendation for walks per library entry (SI / Methods §2)
PAPER_WALKS_PER_ENTRY = 12_000_000


# ---------------------------------------------------------------------------
# Walk result  (new structure: per-axis dM)
# ---------------------------------------------------------------------------

@dataclass
class WalkResult:
    """Output of random walks, now with per-axis independent ensembles.

    Attributes
    ----------
    dM_per_axis : list of 3 ndarrays
        dM_per_axis[ax] has shape (N_ax, n_deltas) and contains the scalar
        encoding-moment differences [μm·ms] for the ax-th gradient axis.
        N_ax = total walkers across all ensembles assigned to that axis
        (after dropping any that escaped).
    n_walkers_per_axis : tuple of 3 ints
    deltas : list
    n_escaped : int
        Total escaped walkers across all 3 axes (for logging).
    """
    dM_per_axis:        List[np.ndarray]
    n_walkers_per_axis: tuple
    deltas:             list
    n_escaped:          int = 0


# ---------------------------------------------------------------------------
# pp ↔ k_io conversion (Eq. 5 of paper)
# ---------------------------------------------------------------------------

def kio_to_pp(kio: float, mean_AV: float, cfg: SimConfig) -> float:
    """k_io [s⁻¹] → permeation probability per attempted membrane crossing.
    Uses the exact <A/V> from the realised ensemble (Fix #4)."""
    factor = math.sqrt(cfg.D0 / (6.0 * cfg.ts)) * mean_AV
    return (kio / 1000.0) / factor

def pp_to_kio(pp: float, mean_AV: float, cfg: SimConfig) -> float:
    factor = math.sqrt(cfg.D0 / (6.0 * cfg.ts)) * mean_AV
    return pp * factor * 1000.0


# ===================================================================
#  CUDA KERNEL  (with freeze-on-escape — Fix #2)
# ===================================================================

if HAS_CUDA:

    @cuda.jit
    def _walk_kernel(
        seeds, annulus,
        grid_s1, grid_s2,
        grid_spacing,
        G_grid,
        sigma, ts, n_steps,
        pp,
        L,
        lo, hi,
        pfg1_start, pfg1_end,
        pfg2_starts, pfg2_ends,
        n_deltas,
        rng_states,
        dM_out,
        escaped_out,           # int8 (n_walkers,) — 1 if walker exited Ω_sim
    ):
        tid = cuda.grid(1)
        if tid >= dM_out.shape[0]:
            return

        # --- Initialise position in Ω_src = [lo, hi]³  (= [buffer, L-buffer]³)
        px = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo
        py = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo
        pz = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo

        # --- Classification helper (returns s1, inside flag) -------------
        def _classify(cx, cy, cz):
            gx = int(cx / grid_spacing)
            gy = int(cy / grid_spacing)
            gz = int(cz / grid_spacing)
            gx = min(max(gx, 0), G_grid - 1)
            gy = min(max(gy, 0), G_grid - 1)
            gz = min(max(gz, 0), G_grid - 1)

            s1 = grid_s1[gx, gy, gz]
            s2 = grid_s2[gx, gy, gz]

            # Re-rank s1, s2 by actual distance to walker position
            d1 = ((cx - seeds[s1, 0])**2 + (cy - seeds[s1, 1])**2
                  + (cz - seeds[s1, 2])**2)
            d2 = ((cx - seeds[s2, 0])**2 + (cy - seeds[s2, 1])**2
                  + (cz - seeds[s2, 2])**2)
            if d2 < d1:
                tmp = s1;  s1 = s2;  s2 = tmp

            mx = 0.5 * (seeds[s1, 0] + seeds[s2, 0])
            my = 0.5 * (seeds[s1, 1] + seeds[s2, 1])
            mz = 0.5 * (seeds[s1, 2] + seeds[s2, 2])
            dx = seeds[s2, 0] - seeds[s1, 0]
            dy = seeds[s2, 1] - seeds[s1, 1]
            dz = seeds[s2, 2] - seeds[s1, 2]
            nrm = math.sqrt(dx*dx + dy*dy + dz*dz)
            if nrm < 1e-30:
                return s1, True
            inv_nrm = 1.0 / nrm
            nx = dx * inv_nrm;  ny = dy * inv_nrm;  nz = dz * inv_nrm
            signed = (cx - mx)*nx + (cy - my)*ny + (cz - mz)*nz
            inside = abs(signed) >= annulus[s1]
            return s1, inside

        cur_s1, cur_inside = _classify(px, py, pz)

        m1x = 0.0; m1y = 0.0; m1z = 0.0
        m2x0 = 0.0; m2y0 = 0.0; m2z0 = 0.0
        m2x1 = 0.0; m2y1 = 0.0; m2z1 = 0.0
        m2x2 = 0.0; m2y2 = 0.0; m2z2 = 0.0
        m2x3 = 0.0; m2y3 = 0.0; m2z3 = 0.0

        escaped = 0  # local frozen flag

        for step in range(n_steps):

            # --- Freeze-on-escape: once escaped, do nothing more ---------
            if escaped == 1:
                continue

            dpx = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            dpy = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            dpz = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            nx_ = px + dpx
            ny_ = py + dpy
            nz_ = pz + dpz

            # --- Boundary-escape check ------------------------------------
            if (nx_ < 0.0 or nx_ >= L or
                ny_ < 0.0 or ny_ >= L or
                nz_ < 0.0 or nz_ >= L):
                escaped = 1
                # Do NOT update position; do NOT accumulate moments.
                # The walker is dead for the remainder of the walk and will
                # be dropped on the host.
                continue

            new_s1, new_inside = _classify(nx_, ny_, nz_)

            # Count membrane crossings for this step
            m = 0
            if cur_inside and (not new_inside):
                m = 1
            elif (not cur_inside) and new_inside:
                m = 1
            elif cur_inside and new_inside and (cur_s1 != new_s1):
                m = 2

            if m > 0 and pp < 1.0:
                u = xoroshiro128p_uniform_float64(rng_states, tid)
                pp_m = pp
                if m == 2:
                    pp_m = pp * pp
                if u >= pp_m:
                    # Failed permeation: revert position
                    nx_ = px;  ny_ = py;  nz_ = pz
                    new_s1 = cur_s1
                    new_inside = cur_inside

            px = nx_;  py = ny_;  pz = nz_
            cur_s1 = new_s1
            cur_inside = new_inside

            # --- Moment accumulation (first moment of position over PFG) -
            if pfg1_start <= step < pfg1_end:
                m1x += px * ts;  m1y += py * ts;  m1z += pz * ts

            if n_deltas > 0 and pfg2_starts[0] <= step < pfg2_ends[0]:
                m2x0 += px * ts;  m2y0 += py * ts;  m2z0 += pz * ts
            if n_deltas > 1 and pfg2_starts[1] <= step < pfg2_ends[1]:
                m2x1 += px * ts;  m2y1 += py * ts;  m2z1 += pz * ts
            if n_deltas > 2 and pfg2_starts[2] <= step < pfg2_ends[2]:
                m2x2 += px * ts;  m2y2 += py * ts;  m2z2 += pz * ts
            if n_deltas > 3 and pfg2_starts[3] <= step < pfg2_ends[3]:
                m2x3 += px * ts;  m2y3 += py * ts;  m2z3 += pz * ts

        escaped_out[tid] = escaped

        if n_deltas > 0:
            dM_out[tid, 0, 0] = m1x - m2x0
            dM_out[tid, 0, 1] = m1y - m2y0
            dM_out[tid, 0, 2] = m1z - m2z0
        if n_deltas > 1:
            dM_out[tid, 1, 0] = m1x - m2x1
            dM_out[tid, 1, 1] = m1y - m2y1
            dM_out[tid, 1, 2] = m1z - m2z1
        if n_deltas > 2:
            dM_out[tid, 2, 0] = m1x - m2x2
            dM_out[tid, 2, 1] = m1y - m2y2
            dM_out[tid, 2, 2] = m1z - m2z2
        if n_deltas > 3:
            dM_out[tid, 3, 0] = m1x - m2x3
            dM_out[tid, 3, 1] = m1y - m2y3
            dM_out[tid, 3, 2] = m1z - m2z3


# ===================================================================
#  CPU FALLBACK  (also freeze-on-escape)
# ===================================================================

def _walk_cpu(ens, pp, cfg, seed=0):
    """Pure-NumPy fallback.  Freeze-on-escape; no reflection."""
    rng = np.random.default_rng(seed)
    N = cfg.n_walkers
    sigma = cfg.sigma
    ts = cfg.ts
    L = cfg.L
    lo = cfg.buffer
    hi = L - cfg.buffer

    positions = rng.uniform(lo, hi, (N, 3))
    cur_s1, cur_inside = ens.classify_cpu(positions)
    frozen = np.zeros(N, dtype=bool)

    n_deltas = len(cfg.Deltas)
    M1 = np.zeros((N, 3))
    M2 = np.zeros((N, n_deltas, 3))

    pfg1_s, pfg1_e = cfg.pfg1_steps()
    pfg2_ranges = [cfg.pfg2_steps(D) for D in cfg.Deltas]

    for step in range(cfg.n_steps):
        active = ~frozen
        if not active.any():
            break

        # Propose displacements
        dx = rng.normal(0, sigma, (N, 3))
        proposed = positions + dx

        # Escape check: newly-escaped walkers become frozen (no position
        # update, no moment update, will be dropped)
        oob = (proposed < 0.0).any(axis=1) | (proposed >= L).any(axis=1)
        newly_frozen = oob & active
        if newly_frozen.any():
            frozen |= newly_frozen
            proposed[newly_frozen] = positions[newly_frozen]
            active = ~frozen

        # Classify proposed positions (only needed for still-active walkers)
        new_s1, new_inside = ens.classify_cpu(proposed)

        # Crossing detection + membrane rejection
        m = np.zeros(N, dtype=np.int32)
        m[cur_inside & ~new_inside] = 1
        m[~cur_inside & new_inside] = 1
        m[cur_inside & new_inside & (cur_s1 != new_s1)] = 2

        crossing = (m > 0) & active
        if crossing.any() and pp < 1.0:
            u = rng.uniform(0, 1, crossing.sum())
            pp_m = pp ** m[crossing]
            rej_sub = u >= pp_m
            rej_idx = np.where(crossing)[0][rej_sub]
            proposed[rej_idx] = positions[rej_idx]
            new_s1[rej_idx] = cur_s1[rej_idx]
            new_inside[rej_idx] = cur_inside[rej_idx]

        # Commit state (frozen walkers already reverted to their pre-step
        # positions above)
        positions = proposed
        cur_s1 = np.where(active, new_s1, cur_s1)
        cur_inside = np.where(active, new_inside, cur_inside)

        # Accumulate moments only for active walkers
        if pfg1_s <= step < pfg1_e:
            M1[active] += positions[active] * ts
        for di, (p2s, p2e) in enumerate(pfg2_ranges):
            if p2s <= step < p2e:
                M2[active, di, :] += positions[active] * ts

    dM = M1[:, np.newaxis, :] - M2  # (N, n_deltas, 3)
    escaped = frozen.astype(np.int8)
    return dM, escaped


# ===================================================================
#  One ensemble's worth of walks  (one axis assignment)
# ===================================================================

def _run_walks_single(
    ens: Ensemble,
    kio: float,
    cfg: SimConfig,
    seed: int,
    verbose: bool,
):
    """Run walks on one ensemble, return (dM[N,n_deltas,3], n_escaped)."""
    pp = kio_to_pp(kio, ens.mean_AV, cfg) if np.isfinite(kio) else 1.0
    pp = float(np.clip(pp, 0.0, 1.0))
    if verbose:
        print(f"    pp = {pp:.6f}  (kio = {kio:.1f} s⁻¹, "
              f"<A/V> = {ens.mean_AV:.3f} μm⁻¹)")

    if HAS_CUDA:
        dM, escaped = _run_gpu(ens, pp, cfg, seed)
    else:
        dM, escaped = _walk_cpu(ens, pp, cfg, seed)

    n_escaped = int(escaped.sum())
    n_total = int(escaped.size)

    # Enforce max escape fraction (Fix #2: paper's abort semantics)
    if n_escaped / max(n_total, 1) > cfg.max_escape_frac:
        raise RuntimeError(
            f"Escape rate {n_escaped/n_total*100:.2f}% exceeds "
            f"max_escape_frac={cfg.max_escape_frac*100:.1f}% "
            f"(rho={ens.rho}, V={ens.V}). "
            f"Increase cfg.L and/or cfg.buffer.")

    # Drop escaped walkers
    if n_escaped > 0:
        keep = escaped == 0
        dM = dM[keep]

    return dM, n_escaped


def _run_gpu(ens, pp, cfg, seed):
    N = cfg.n_walkers
    n_deltas = len(cfg.Deltas)

    d_seeds   = cuda.to_device(ens.seeds)
    d_annulus = cuda.to_device(ens.annulus)
    d_grid_s1 = cuda.to_device(ens.grid_s1)
    d_grid_s2 = cuda.to_device(ens.grid_s2)

    pfg1_s, pfg1_e = cfg.pfg1_steps()
    pfg2_s = np.array([cfg.pfg2_steps(D)[0] for D in cfg.Deltas], dtype=np.int32)
    pfg2_e = np.array([cfg.pfg2_steps(D)[1] for D in cfg.Deltas], dtype=np.int32)
    while len(pfg2_s) < 4:
        pfg2_s = np.append(pfg2_s, 999999)
        pfg2_e = np.append(pfg2_e, 999999)
    d_pfg2_s = cuda.to_device(pfg2_s)
    d_pfg2_e = cuda.to_device(pfg2_e)

    d_dM = cuda.device_array((N, n_deltas, 3), dtype=np.float64)
    d_escaped = cuda.device_array(N, dtype=np.int8)

    rng_states = create_xoroshiro128p_states(N, seed=seed)

    threads_per_block = 128
    blocks = (N + threads_per_block - 1) // threads_per_block

    _walk_kernel[blocks, threads_per_block](
        d_seeds, d_annulus, d_grid_s1, d_grid_s2,
        np.float64(ens.grid_spacing),
        np.int32(cfg.grid_size),
        np.float64(cfg.sigma),
        np.float64(cfg.ts),
        np.int32(cfg.n_steps),
        np.float64(pp),
        np.float64(ens.L),
        np.float64(cfg.buffer),
        np.float64(ens.L - cfg.buffer),
        np.int32(pfg1_s), np.int32(pfg1_e),
        d_pfg2_s, d_pfg2_e,
        np.int32(n_deltas),
        rng_states,
        d_dM,
        d_escaped,
    )

    cuda.synchronize()
    return d_dM.copy_to_host(), d_escaped.copy_to_host()


# ===================================================================
#  PUBLIC API  (3-axis independent ensembles — Fix #1)
# ===================================================================

def run_simulation(
    rho: float, V: float, kio: float,
    cfg: SimConfig | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> WalkResult:
    """Run the full multi-ensemble walk for one library entry.

    Fix #1: builds `3 × cfg.n_ensembles` independent ensembles, with
    cfg.n_ensembles dedicated to each of the three gradient axes (x, y, z).
    Each ensemble's walker dM is sliced to keep only its assigned axis
    component.  The signal computation (in signal.py) averages the three
    axis signals.
    """
    if cfg is None:
        cfg = SimConfig()

    # ---- Walker-count warning (Fix #5) ----
    # Total independent phase measurements per library entry =
    #     3 axes × n_ensembles × n_walkers
    total_walks = 3 * cfg.n_ensembles * cfg.n_walkers
    if total_walks < PAPER_WALKS_PER_ENTRY // 10:
        sys.stderr.write(
            f"    ⚠ Only {total_walks:,} walks per library entry "
            f"(paper target ~{PAPER_WALKS_PER_ENTRY:,}). "
            f"High-b decays will be noisy.\n")

    dM_per_axis: List[list] = [[], [], []]
    n_escaped_total = 0

    # Outer loop: gradient axes.  Inner loop: independent ensembles per axis.
    for ax in range(3):
        for ei in range(cfg.n_ensembles):
            if verbose:
                print(f"  Axis {['x','y','z'][ax]}  "
                      f"Ensemble {ei+1}/{cfg.n_ensembles}  "
                      f"(rho={rho:.0f}, V={V:.2f}, kio={kio:.1f})")

            ens_seed = seed + ax * 10_000_000 + ei * 1000
            walk_seed = ens_seed + 1

            if rho <= 0 or V <= 0:
                ens = create_dummy_ensemble(cfg)
            else:
                ens = create_ensemble(rho, V, cfg,
                                      seed=ens_seed, verbose=verbose)

            dM, n_esc = _run_walks_single(ens, kio, cfg, walk_seed, verbose)
            n_escaped_total += n_esc

            # Keep only the axis component this ensemble is assigned to.
            # dM has shape (N_ax_ensemble, n_deltas, 3); we slice [:, :, ax]
            # → shape (N_ax_ensemble, n_deltas).
            dM_per_axis[ax].append(dM[:, :, ax].copy())

    # Concatenate ensemble contributions per axis
    merged_per_axis = [np.concatenate(chunks, axis=0) for chunks in dM_per_axis]
    n_per_axis = tuple(arr.shape[0] for arr in merged_per_axis)

    return WalkResult(
        dM_per_axis=merged_per_axis,
        n_walkers_per_axis=n_per_axis,
        deltas=list(cfg.Deltas),
        n_escaped=n_escaped_total,
    )


# ---------------------------------------------------------------------------
# Backwards-compat shim: run_walks for a single ensemble (used by tests?)
# ---------------------------------------------------------------------------

def run_walks(
    ens: Ensemble,
    kio: float,
    cfg: SimConfig | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> WalkResult:
    """Run walks on a single prebuilt ensemble.

    Returned `WalkResult.dM_per_axis` contains the same (single) dM sliced
    onto all three axes — i.e. this is the LEGACY correlated-axis powder
    average.  Use `run_simulation()` for the new independent-axis version.
    """
    if cfg is None:
        cfg = SimConfig()

    dM, n_esc = _run_walks_single(ens, kio, cfg, seed, verbose)
    # Same walker pool projected onto all three axes (legacy behaviour)
    per_axis = [dM[:, :, 0].copy(), dM[:, :, 1].copy(), dM[:, :, 2].copy()]
    return WalkResult(
        dM_per_axis=per_axis,
        n_walkers_per_axis=tuple(arr.shape[0] for arr in per_axis),
        deltas=list(cfg.Deltas),
        n_escaped=n_esc,
    )
