"""
GPU-accelerated Monte Carlo random walks with semipermeable membranes.

CORRECTIONS vs. the original implementation:

  Deviation #2 — too few walkers:
      The paper recommends ~12×10⁶ walks per library entry. We now warn
      when n_walkers × n_ensembles is well below that. Production runs
      should aim for at least 10⁶ total walks.

  Deviation #3 — silent boundary escape:
      The original kernel clamped grid indices for walkers that drifted
      outside the simulation cube, silently corrupting their classification
      and accumulated moments. The kernel now flags any walker that exits
      [0, L]³ at any time step; flagged walkers are excluded from the
      output and reported via stderr.
"""

from __future__ import annotations

import math
import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional

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


# Paper recommendation, used for the warning
PAPER_WALKS_PER_ENTRY = 12_000_000


# ---------------------------------------------------------------------------
# Walk result
# ---------------------------------------------------------------------------

@dataclass
class WalkResult:
    """Output of random walks.

    dM : (n_walkers, n_deltas, 3)  encoding moment differences.
         Walkers that escaped Ω_sim have been removed.
    n_escaped : int
    """
    dM:        np.ndarray
    n_walkers: int
    deltas:    list
    n_escaped: int = 0


# ---------------------------------------------------------------------------
# pp ↔ k_io conversion (Eq. 5 of paper)
# ---------------------------------------------------------------------------

def kio_to_pp(kio: float, mean_AV: float, cfg: SimConfig) -> float:
    """k_io [s⁻¹] → permeation probability per attempted membrane crossing.
    Uses the *measured* <A/V> from the realised ensemble (deviation #1)."""
    factor = math.sqrt(cfg.D0 / (6.0 * cfg.ts)) * mean_AV
    return (kio / 1000.0) / factor

def pp_to_kio(pp: float, mean_AV: float, cfg: SimConfig) -> float:
    factor = math.sqrt(cfg.D0 / (6.0 * cfg.ts)) * mean_AV
    return pp * factor * 1000.0


# ===================================================================
#  CUDA KERNEL  (with boundary-escape detection)
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
        escaped_out,           # int8 (n_walkers,) — 1 if walker exited [0,L]³
    ):
        tid = cuda.grid(1)
        if tid >= dM_out.shape[0]:
            return

        # --- Initialise position ---
        px = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo
        py = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo
        pz = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo

        # --- Classification helper (returns s1, inside flag) ---
        def _classify(cx, cy, cz):
            gx = int(cx / grid_spacing)
            gy = int(cy / grid_spacing)
            gz = int(cz / grid_spacing)
            gx = min(max(gx, 0), G_grid - 1)
            gy = min(max(gy, 0), G_grid - 1)
            gz = min(max(gz, 0), G_grid - 1)

            s1 = grid_s1[gx, gy, gz]
            s2 = grid_s2[gx, gy, gz]

            # Re-rank s1, s2 by actual distance to walker position to
            # mitigate voxel-grid quantisation near membranes (deviation #5)
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

        escaped = 0  # local flag

        for step in range(n_steps):
            dpx = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            dpy = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            dpz = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            nx_ = px + dpx
            ny_ = py + dpy
            nz_ = pz + dpz

            # --- Boundary-escape check (deviation #3 fix) ---
            if (nx_ < 0.0 or nx_ >= L or
                ny_ < 0.0 or ny_ >= L or
                nz_ < 0.0 or nz_ >= L):
                escaped = 1
                # Reflect back into the box (still continue to avoid wasted
                # work, but the walker won't be used)
                if nx_ < 0.0:  nx_ = -nx_
                if ny_ < 0.0:  ny_ = -ny_
                if nz_ < 0.0:  nz_ = -nz_
                if nx_ >= L:   nx_ = 2.0 * L - nx_ - 1e-9
                if ny_ >= L:   ny_ = 2.0 * L - ny_ - 1e-9
                if nz_ >= L:   nz_ = 2.0 * L - nz_ - 1e-9

            new_s1, new_inside = _classify(nx_, ny_, nz_)

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
                    nx_ = px;  ny_ = py;  nz_ = pz
                    new_s1 = cur_s1
                    new_inside = cur_inside

            px = nx_;  py = ny_;  pz = nz_
            cur_s1 = new_s1
            cur_inside = new_inside

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
#  CPU FALLBACK
# ===================================================================

def _walk_cpu(ens, pp, cfg, seed=0):
    """Pure-NumPy fallback. Includes boundary-escape detection."""
    rng = np.random.default_rng(seed)
    N = cfg.n_walkers
    sigma = cfg.sigma
    ts = cfg.ts
    L = cfg.L
    lo = cfg.buffer
    hi = L - cfg.buffer

    positions = rng.uniform(lo, hi, (N, 3))
    cur_s1, cur_inside = ens.classify_cpu(positions)
    escaped = np.zeros(N, dtype=np.int8)

    n_deltas = len(cfg.Deltas)
    M1 = np.zeros((N, 3))
    M2 = np.zeros((N, n_deltas, 3))

    pfg1_s, pfg1_e = cfg.pfg1_steps()

    for step in range(cfg.n_steps):
        dx = rng.normal(0, sigma, (N, 3))
        proposed = positions + dx

        # Boundary check
        oob = (proposed < 0).any(axis=1) | (proposed >= L).any(axis=1)
        escaped[oob] = 1
        # Reflect
        proposed = np.where(proposed < 0, -proposed, proposed)
        proposed = np.where(proposed >= L, 2*L - proposed - 1e-9, proposed)

        new_s1, new_inside = ens.classify_cpu(proposed)

        m = np.zeros(N, dtype=np.int32)
        m[cur_inside & ~new_inside] = 1
        m[~cur_inside & new_inside] = 1
        m[cur_inside & new_inside & (cur_s1 != new_s1)] = 2

        crossing = m > 0
        if crossing.any() and pp < 1.0:
            u = rng.uniform(0, 1, crossing.sum())
            pp_m = pp ** m[crossing]
            rej_sub = u >= pp_m
            rej_idx = np.where(crossing)[0][rej_sub]
            proposed[rej_idx] = positions[rej_idx]
            new_s1[rej_idx] = cur_s1[rej_idx]
            new_inside[rej_idx] = cur_inside[rej_idx]

        positions = proposed
        cur_s1 = new_s1
        cur_inside = new_inside

        if pfg1_s <= step < pfg1_e:
            M1 += positions * ts

        for di, Delta in enumerate(cfg.Deltas):
            p2s, p2e = cfg.pfg2_steps(Delta)
            if p2s <= step < p2e:
                M2[:, di, :] += positions * ts

    dM = M1[:, np.newaxis, :] - M2  # (N, n_deltas, 3)
    return dM, escaped


# ===================================================================
#  PUBLIC API
# ===================================================================

def run_walks(
    ens: Ensemble,
    kio: float,
    cfg: SimConfig | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> WalkResult:
    if cfg is None:
        cfg = SimConfig()

    pp = kio_to_pp(kio, ens.mean_AV, cfg) if np.isfinite(kio) else 1.0
    pp = float(np.clip(pp, 0.0, 1.0))
    if verbose:
        print(f"    pp = {pp:.6f}  (kio = {kio:.1f} s⁻¹, "
              f"<A/V>_meas = {ens.mean_AV:.3f} μm⁻¹)")

    if HAS_CUDA:
        dM, escaped = _run_gpu(ens, pp, cfg, seed)
    else:
        dM, escaped = _walk_cpu(ens, pp, cfg, seed)

    n_escaped = int(escaped.sum())
    if n_escaped > 0:
        keep = escaped == 0
        dM = dM[keep]
        if verbose or n_escaped / len(escaped) > 0.01:
            sys.stderr.write(
                f"    ⚠ {n_escaped}/{len(escaped)} walkers "
                f"({n_escaped/len(escaped)*100:.1f}%) escaped Ω_sim "
                f"and were dropped. Consider increasing cfg.L or cfg.buffer.\n")

    return WalkResult(dM=dM, n_walkers=dM.shape[0],
                      deltas=cfg.Deltas, n_escaped=n_escaped)


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


def run_simulation(
    rho: float, V: float, kio: float,
    cfg: SimConfig | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> WalkResult:
    """Run walks across multiple ensembles and merge."""
    if cfg is None:
        cfg = SimConfig()

    # ---- Walker count warning (deviation #2) ----
    total_walks = cfg.n_walkers * cfg.n_ensembles
    if total_walks < PAPER_WALKS_PER_ENTRY // 10:
        sys.stderr.write(
            f"    ⚠ Only {total_walks:,} walks per library entry "
            f"(paper recommends ~{PAPER_WALKS_PER_ENTRY:,}). "
            f"Decays may be noisy at high b.\n")

    all_dM = []
    n_escaped_total = 0
    for ei in range(cfg.n_ensembles):
        if verbose:
            print(f"  Ensemble {ei+1}/{cfg.n_ensembles}  "
                  f"(rho={rho:.0f}, V={V:.2f}, kio={kio:.1f})")
        if rho <= 0 or V <= 0:
            ens = create_dummy_ensemble(cfg)
        else:
            ens = create_ensemble(rho, V, cfg, seed=seed + ei * 1000,
                                  verbose=verbose)
        wr = run_walks(ens, kio, cfg, seed=seed + ei * 1000 + 1,
                       verbose=verbose)
        all_dM.append(wr.dM)
        n_escaped_total += wr.n_escaped

    merged = np.concatenate(all_dM, axis=0)
    return WalkResult(dM=merged, n_walkers=merged.shape[0],
                      deltas=cfg.Deltas, n_escaped=n_escaped_total)
