"""
GPU-accelerated Monte Carlo random walks with semipermeable membranes.

Uses Numba CUDA.  Each CUDA thread runs one walker through all time steps,
accumulating SDE encoding moments for ALL Δ values simultaneously.
Falls back to a CPU implementation if no CUDA device is available.

RTX 2060 Mobile (1920 cores, 6 GB): handles 20 000 walkers comfortably.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .ensemble import Ensemble, create_ensemble, create_dummy_ensemble
from .config   import SimConfig

# ---------------------------------------------------------------------------
# Try to import CUDA — fall back gracefully
# ---------------------------------------------------------------------------
try:
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_uniform_float64
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False

if not HAS_CUDA:
    print("WARNING: CUDA not available — falling back to CPU (slow)")


# ---------------------------------------------------------------------------
# Walk result
# ---------------------------------------------------------------------------

@dataclass
class WalkResult:
    """Output of random walks.

    dM : ndarray (n_walkers, n_deltas, 3)
        Encoding-moment difference  M1 - M2  per walker, per Δ, per axis.
    n_walkers : int
    deltas : list[float]
    """
    dM:        np.ndarray
    n_walkers: int
    deltas:    list


# ---------------------------------------------------------------------------
# pp  ↔  k_io conversion   (Eq. 5)
# ---------------------------------------------------------------------------

def kio_to_pp(kio: float, mean_AV: float, cfg: SimConfig) -> float:
    D0 = cfg.D0;  ts = cfg.ts
    factor = math.sqrt(D0 / (6.0 * ts)) * mean_AV
    return (kio / 1000.0) / factor

def pp_to_kio(pp: float, mean_AV: float, cfg: SimConfig) -> float:
    D0 = cfg.D0;  ts = cfg.ts
    factor = math.sqrt(D0 / (6.0 * ts)) * mean_AV
    return pp * factor * 1000.0


# ===================================================================
#  CUDA KERNEL
# ===================================================================

if HAS_CUDA:

    @cuda.jit
    def _walk_kernel(
        # Ensemble (device arrays)
        seeds,          # (n_seeds, 3) float64
        annulus,        # (n_seeds,)   float64
        grid_s1,        # (G, G, G)   int32
        grid_s2,        # (G, G, G)   int32
        # Scalars
        grid_spacing,   # float64
        G_grid,         # int32  (grid points per axis)
        sigma,          # float64
        ts,             # float64
        n_steps,        # int32
        pp,             # float64  (permeation probability)
        L,              # float64
        lo, hi,         # float64  (init range)
        # PFG timing
        pfg1_start,     # int32
        pfg1_end,       # int32
        pfg2_starts,    # (n_deltas,) int32
        pfg2_ends,      # (n_deltas,) int32
        n_deltas,       # int32
        # RNG
        rng_states,
        # Output
        dM_out,         # (n_walkers, n_deltas, 3) float64
    ):
        tid = cuda.grid(1)
        if tid >= dM_out.shape[0]:
            return

        # --- Initialise walker position ---
        px = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo
        py = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo
        pz = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo

        # --- Classify initial position ---
        def _classify(cx, cy, cz):
            """Return (s1_idx, is_inside)."""
            gx = int(cx / grid_spacing)
            gy = int(cy / grid_spacing)
            gz = int(cz / grid_spacing)
            gx = min(max(gx, 0), G_grid - 1)
            gy = min(max(gy, 0), G_grid - 1)
            gz = min(max(gz, 0), G_grid - 1)

            s1 = grid_s1[gx, gy, gz]
            s2 = grid_s2[gx, gy, gz]

            # Bisecting plane test
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
            nx = dx * inv_nrm
            ny = dy * inv_nrm
            nz = dz * inv_nrm
            signed = (cx - mx)*nx + (cy - my)*ny + (cz - mz)*nz
            dist = abs(signed)
            inside = dist >= annulus[s1]
            return s1, inside

        cur_s1, cur_inside = _classify(px, py, pz)

        # --- Encoding moment accumulators ---
        # M1 is shared across all Δ (PFG-1 timing is identical)
        m1x = 0.0;  m1y = 0.0;  m1z = 0.0
        # M2 is separate for each Δ  (max 4 Δ values)
        m2x0 = 0.0; m2y0 = 0.0; m2z0 = 0.0
        m2x1 = 0.0; m2y1 = 0.0; m2z1 = 0.0
        m2x2 = 0.0; m2y2 = 0.0; m2z2 = 0.0
        m2x3 = 0.0; m2y3 = 0.0; m2z3 = 0.0

        # --- Walk loop ---
        for step in range(n_steps):
            # 1. Propose displacement
            dpx = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            dpy = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            dpz = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            nx_ = px + dpx
            ny_ = py + dpy
            nz_ = pz + dpz

            # 2. Classify proposed
            new_s1, new_inside = _classify(nx_, ny_, nz_)

            # 3. Count membranes
            m = 0
            if cur_inside and (not new_inside):
                m = 1
            elif (not cur_inside) and new_inside:
                m = 1
            elif cur_inside and new_inside and (cur_s1 != new_s1):
                m = 2

            # 4. Permeation test
            if m > 0 and pp < 1.0:
                u = xoroshiro128p_uniform_float64(rng_states, tid)
                pp_m = pp
                if m == 2:
                    pp_m = pp * pp
                if u >= pp_m:
                    # Reject — stay put
                    nx_ = px;  ny_ = py;  nz_ = pz
                    new_s1 = cur_s1
                    new_inside = cur_inside

            # 5. Update position
            px = nx_;  py = ny_;  pz = nz_
            cur_s1 = new_s1
            cur_inside = new_inside

            # 6. Accumulate encoding moments
            if pfg1_start <= step < pfg1_end:
                m1x += px * ts
                m1y += py * ts
                m1z += pz * ts

            if n_deltas > 0 and pfg2_starts[0] <= step < pfg2_ends[0]:
                m2x0 += px * ts;  m2y0 += py * ts;  m2z0 += pz * ts
            if n_deltas > 1 and pfg2_starts[1] <= step < pfg2_ends[1]:
                m2x1 += px * ts;  m2y1 += py * ts;  m2z1 += pz * ts
            if n_deltas > 2 and pfg2_starts[2] <= step < pfg2_ends[2]:
                m2x2 += px * ts;  m2y2 += py * ts;  m2z2 += pz * ts
            if n_deltas > 3 and pfg2_starts[3] <= step < pfg2_ends[3]:
                m2x3 += px * ts;  m2y3 += py * ts;  m2z3 += pz * ts

        # --- Write output ---
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
    """Pure-NumPy fallback for machines without CUDA."""
    rng = np.random.default_rng(seed)
    N = cfg.n_walkers
    sigma = cfg.sigma
    ts = cfg.ts
    lo = cfg.buffer
    hi = cfg.L - cfg.buffer

    positions = rng.uniform(lo, hi, (N, 3))
    cur_s1, cur_inside = ens.classify_cpu(positions)

    n_deltas = len(cfg.Deltas)
    M1 = np.zeros((N, 3))
    M2 = np.zeros((N, n_deltas, 3))

    pfg1_s, pfg1_e = cfg.pfg1_steps()

    for step in range(cfg.n_steps):
        dx = rng.normal(0, sigma, (N, 3))
        proposed = positions + dx
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
    return dM


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
    """Run random walks (GPU if available, else CPU)."""
    if cfg is None:
        cfg = SimConfig()

    pp = kio_to_pp(kio, ens.mean_AV, cfg) if np.isfinite(kio) else 1.0
    pp = float(np.clip(pp, 0.0, 1.0))
    if verbose:
        print(f"    pp = {pp:.6f}  (kio = {kio:.1f} s⁻¹)")

    if HAS_CUDA:
        dM = _run_gpu(ens, pp, cfg, seed)
    else:
        dM = _walk_cpu(ens, pp, cfg, seed)

    return WalkResult(dM=dM, n_walkers=dM.shape[0], deltas=cfg.Deltas)


def _run_gpu(ens, pp, cfg, seed):
    """Launch CUDA kernel."""
    N = cfg.n_walkers
    n_deltas = len(cfg.Deltas)

    # Transfer ensemble to GPU
    d_seeds   = cuda.to_device(ens.seeds)
    d_annulus = cuda.to_device(ens.annulus)
    d_grid_s1 = cuda.to_device(ens.grid_s1)
    d_grid_s2 = cuda.to_device(ens.grid_s2)

    # PFG timing arrays
    pfg1_s, pfg1_e = cfg.pfg1_steps()
    pfg2_s = np.array([cfg.pfg2_steps(D)[0] for D in cfg.Deltas], dtype=np.int32)
    pfg2_e = np.array([cfg.pfg2_steps(D)[1] for D in cfg.Deltas], dtype=np.int32)
    # Pad to length 4
    while len(pfg2_s) < 4:
        pfg2_s = np.append(pfg2_s, 999999)
        pfg2_e = np.append(pfg2_e, 999999)
    d_pfg2_s = cuda.to_device(pfg2_s)
    d_pfg2_e = cuda.to_device(pfg2_e)

    # Output
    d_dM = cuda.device_array((N, n_deltas, 3), dtype=np.float64)

    # RNG
    rng_states = create_xoroshiro128p_states(N, seed=seed)

    # Launch
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
    )

    cuda.synchronize()
    return d_dM.copy_to_host()


def run_simulation(
    rho: float, V: float, kio: float,
    cfg: SimConfig | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> WalkResult:
    """Run walks across multiple ensembles and merge."""
    if cfg is None:
        cfg = SimConfig()

    all_dM = []
    for ei in range(cfg.n_ensembles):
        if verbose:
            print(f"  Ensemble {ei+1}/{cfg.n_ensembles}  "
                  f"(rho={rho:.0f}, V={V:.2f}, kio={kio:.1f})")
        if rho <= 0 or V <= 0:
            ens = create_dummy_ensemble(cfg)
        else:
            ens = create_ensemble(rho, V, cfg, seed=seed + ei * 1000)
        wr = run_walks(ens, kio, cfg, seed=seed + ei * 1000 + 1,
                       verbose=verbose)
        all_dM.append(wr.dM)

    merged = np.concatenate(all_dM, axis=0)
    return WalkResult(dM=merged, n_walkers=merged.shape[0], deltas=cfg.Deltas)
