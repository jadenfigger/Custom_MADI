"""
GPU-accelerated Monte Carlo random walks with semipermeable membranes.

(δ, Δ, b)-UNIVERSAL LIBRARY REFACTOR
-------------------------------------
Instead of accumulating two fixed PGSE moment windows (one walk per fixed
δ/Δ/kio), each walker now accumulates the running position integral

    Y(t) = ∫₀ᵗ x(s) ds        (trapezoid rule; x measured from the box
                                center, i.e. origin-shifted, for fp precision
                                — see module docstring in madi/config.py)

sampled at stride ``cfg.h_ms`` for all three Cartesian components of every
walk. For ANY (δ, Δ) with δ, Δ, Δ+δ on the h-grid:

    dM(δ,Δ) = Y(δ) + Y(Δ) − Y(Δ+δ)          (three lookups, zero error)

which is exactly the old kernel's ``m1 − m2`` (same units, same sign
convention), so ``madi/signal.py`` reuses the existing SI-unit / G_from_b
machinery unchanged. One walk per ensemble therefore fills every (δ,Δ,b)
column instead of one fixed acquisition.

This module is physics-agnostic below the walk itself: the reduction step
(``Y`` → Σcos(phase)) takes arbitrary per-column ``(j_delta, j_Delta, j_sum,
phase_coef)`` arrays — it doesn't know what a b-value or a δ is. Column
construction (turning physical (δ,Δ,b) into those arrays) lives in
``madi/signal.py``.

Single isotropic ensemble (not 3 independent per-axis ensembles): one
Voronoi ensemble per (ρ,V,kio) entry, harvesting all 3 displacement
components of every 3-D walk, flattened to `3·N_w` samples. Isotropy of the
underlying geometry is validated once (see the validation suite), not
re-derived here.

Common-random-numbers (CRN) seeding: ensemble/walk RNG seeds are a function
of a caller-supplied ``seed`` (constant across an entire library build) and
an ensemble index only — NOT of (ρ,V). This means every (ρ,V) grid point at
a fixed ensemble index shares the same underlying Poisson-seed / walker RNG
stream, which is what the (future) Fisher/CRLB phase needs for low-noise
finite-difference derivatives w.r.t. ρ and V.

Preserved from the previous version: freeze-on-escape (not reflect), the
membrane permeation rule (pp / pp²), and ensemble-reuse across a kio sweep.
"""

from __future__ import annotations

import math
import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .ensemble import Ensemble, create_ensemble, create_dummy_ensemble
from .config   import SimConfig

# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------
try:
    from numba import cuda
    import numba
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

# Reduction-kernel block size (compile-time constant; power of 2 for the
# shared-memory tree reduction).
_REDUCE_THREADS = 256


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


# ---------------------------------------------------------------------------
# CRN-friendly deterministic seeding — depends on (base_seed, ensemble index,
# kio) only, NEVER on (rho, V). See module docstring.
# ---------------------------------------------------------------------------

def _ensemble_seed(base_seed: int, ei: int) -> int:
    return (int(base_seed) + ei * 97_003) & 0x7FFFFFFF

def _walk_seed(base_seed: int, kio: float, ei: int) -> int:
    kio_tag = 0 if not np.isfinite(kio) else int(round(kio * 1000))
    return (int(base_seed) + ei * 104_729 + (kio_tag * 7919) % 1_000_003) & 0x7FFFFFFF


# ===================================================================
#  CUDA WALK KERNEL — produces Y(t), not fixed PGSE moments
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
        half_L,
        steps_per_h,
        rng_states,
        Y_out,          # (n_walkers, n_grid, 3) float64
        escaped_out,    # int8 (n_walkers,) — 1 if walker exited Ω_sim
    ):
        tid = cuda.grid(1)
        if tid >= Y_out.shape[0]:
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

        # --- Running position-integral state (origin-shifted by L/2) -----
        xs = px - half_L;  ys = py - half_L;  zs = pz - half_L
        Yx = 0.0;  Yy = 0.0;  Yz = 0.0
        Y_out[tid, 0, 0] = 0.0
        Y_out[tid, 0, 1] = 0.0
        Y_out[tid, 0, 2] = 0.0

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
                # Do NOT update position; do NOT accumulate Y.
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

            # --- Trapezoid Y accumulation (exact for a piecewise-linear
            # path between MC steps) -------------------------------------
            nxs = px - half_L;  nys = py - half_L;  nzs = pz - half_L
            Yx += 0.5 * (xs + nxs) * ts
            Yy += 0.5 * (ys + nys) * ts
            Yz += 0.5 * (zs + nzs) * ts
            xs = nxs;  ys = nys;  zs = nzs

            step_idx = step + 1
            if step_idx % steps_per_h == 0:
                j = step_idx // steps_per_h
                Y_out[tid, j, 0] = Yx
                Y_out[tid, j, 1] = Yy
                Y_out[tid, j, 2] = Yz

        escaped_out[tid] = escaped

    # -----------------------------------------------------------------
    # Reduction kernel: Y[N, n_grid, 3] → Σcos(phase), Σsin(phase) per
    # (δ,Δ,b) column. One block per column; threads stride over the
    # flattened (kept-walker × axis) index space.
    # -----------------------------------------------------------------
    @cuda.jit
    def _reduce_kernel(
        Y,              # (N, n_grid, 3) float64, full (uncompacted) buffer
        keep_idx,       # (n_kept,) int32 — walker indices to include
        j_delta,        # (n_cols,) int32
        j_Delta,        # (n_cols,) int32
        j_sum,          # (n_cols,) int32
        phase_coef,     # (n_cols,) float64
        cos_sum_out,    # (n_cols,) float64
        sin_sum_out,    # (n_cols,) float64
    ):
        col = cuda.blockIdx.x
        if col >= phase_coef.shape[0]:
            return
        tid = cuda.threadIdx.x
        block_size = cuda.blockDim.x

        jd = j_delta[col]
        jD = j_Delta[col]
        js = j_sum[col]
        pc = phase_coef[col]

        n_kept = keep_idx.shape[0]
        n_total = n_kept * 3

        local_cos = 0.0
        local_sin = 0.0
        i = tid
        while i < n_total:
            w = keep_idx[i // 3]
            ax = i % 3
            dM = Y[w, jd, ax] + Y[w, jD, ax] - Y[w, js, ax]
            phase = pc * dM
            local_cos += math.cos(phase)
            local_sin += math.sin(phase)
            i += block_size

        shared_cos = cuda.shared.array(_REDUCE_THREADS, dtype=numba.float64)
        shared_sin = cuda.shared.array(_REDUCE_THREADS, dtype=numba.float64)
        shared_cos[tid] = local_cos
        shared_sin[tid] = local_sin
        cuda.syncthreads()

        s = block_size // 2
        while s > 0:
            if tid < s:
                shared_cos[tid] += shared_cos[tid + s]
                shared_sin[tid] += shared_sin[tid + s]
            cuda.syncthreads()
            s //= 2

        if tid == 0:
            cos_sum_out[col] = shared_cos[0]
            sin_sum_out[col] = shared_sin[0]


# ===================================================================
#  CPU FALLBACK  (also freeze-on-escape; produces Y, not fixed moments)
# ===================================================================

def _walk_cpu(ens, pp, cfg, seed=0):
    """Pure-NumPy fallback. Returns (Y[N, n_grid, 3] float64, escaped[N] int8)."""
    rng = np.random.default_rng(seed)
    N = cfg.n_walkers
    sigma = cfg.sigma
    ts = cfg.ts
    L = cfg.L
    lo = cfg.buffer
    hi = L - cfg.buffer
    half_L = L / 2.0

    positions = rng.uniform(lo, hi, (N, 3))
    cur_s1, cur_inside = ens.classify_cpu(positions)
    frozen = np.zeros(N, dtype=bool)

    n_grid = cfg.n_grid
    steps_per_h = cfg.steps_per_h
    Y = np.zeros((N, n_grid, 3), dtype=np.float64)
    Y_run = np.zeros((N, 3), dtype=np.float64)
    xs_prev = positions - half_L

    for step in range(cfg.n_steps):
        active = ~frozen
        if not active.any():
            break

        dx = rng.normal(0, sigma, (N, 3))
        proposed = positions + dx

        oob = (proposed < 0.0).any(axis=1) | (proposed >= L).any(axis=1)
        newly_frozen = oob & active
        if newly_frozen.any():
            frozen |= newly_frozen
            proposed[newly_frozen] = positions[newly_frozen]
            active = ~frozen

        new_s1, new_inside = ens.classify_cpu(proposed)

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

        positions = proposed
        cur_s1 = np.where(active, new_s1, cur_s1)
        cur_inside = np.where(active, new_inside, cur_inside)

        xs_new = positions - half_L
        Y_run[active] += 0.5 * (xs_prev[active] + xs_new[active]) * ts
        xs_prev[active] = xs_new[active]

        step_idx = step + 1
        if step_idx % steps_per_h == 0:
            j = step_idx // steps_per_h
            Y[:, j, :] = Y_run

    escaped = frozen.astype(np.int8)
    return Y, escaped


def _reduce_cpu(Y, keep_idx, j_delta, j_Delta, j_sum, phase_coef):
    """Vectorised-per-column NumPy reduction (validation/small-scale fallback
    — the GPU path is the one meant for production-scale (δ,Δ,b) grids)."""
    Yk = Y[keep_idx]                       # (n_kept, n_grid, 3)
    n_cols = phase_coef.shape[0]
    cos_sum = np.empty(n_cols, dtype=np.float64)
    sin_sum = np.empty(n_cols, dtype=np.float64)
    for c in range(n_cols):
        dM = (Yk[:, j_delta[c], :] + Yk[:, j_Delta[c], :]
              - Yk[:, j_sum[c], :])          # (n_kept, 3)
        phase = phase_coef[c] * dM
        cos_sum[c] = np.cos(phase).sum()
        sin_sum[c] = np.sin(phase).sum()
    return cos_sum, sin_sum


# ===================================================================
#  One ensemble's walk → Y  (host-side helper shared by GPU/CPU paths)
# ===================================================================

def _run_walk_one_ensemble(ens: Ensemble, kio: float, cfg: SimConfig,
                            seed: int, verbose: bool):
    """Run the walk for one ensemble. Returns (Y_host_or_device, escaped_host,
    is_device: bool). Y keeps ALL walkers (escaped ones included) — the
    caller builds keep_idx from `escaped_host`."""
    pp = kio_to_pp(kio, ens.mean_AV, cfg) if np.isfinite(kio) else 1.0
    pp = float(np.clip(pp, 0.0, 1.0))
    if verbose:
        print(f"    pp = {pp:.6f}  (kio = {kio:.1f} s⁻¹, "
              f"<A/V> = {ens.mean_AV:.3f} μm⁻¹)")

    if HAS_CUDA:
        N = cfg.n_walkers
        d_seeds   = cuda.to_device(ens.seeds)
        d_annulus = cuda.to_device(ens.annulus)
        d_grid_s1 = cuda.to_device(ens.grid_s1)
        d_grid_s2 = cuda.to_device(ens.grid_s2)

        d_Y = cuda.device_array((N, cfg.n_grid, 3), dtype=np.float64)
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
            np.float64(ens.L / 2.0),
            np.int32(cfg.steps_per_h),
            rng_states,
            d_Y,
            d_escaped,
        )
        cuda.synchronize()
        escaped_host = d_escaped.copy_to_host()
        return d_Y, escaped_host, True
    else:
        Y, escaped = _walk_cpu(ens, pp, cfg, seed)
        return Y, escaped, False


def run_walk_Y(ens: Ensemble, kio: float, cfg: Optional[SimConfig] = None,
               seed: int = 0, verbose: bool = True) -> Tuple[np.ndarray, int]:
    """Public, validation-oriented API: run one ensemble's walk and return
    per-walker Y with escaped walkers dropped, as a HOST array.

    Intended for the validation suite (free-diffusion Var(a) checks, the
    imaginary-part assertion, grid-convergence tests) where per-walker
    access is needed directly — NOT for production library building, which
    uses `run_simulation_reduced` to avoid ever materialising Y for all
    (δ,Δ,b) columns in host memory.
    """
    if cfg is None:
        cfg = SimConfig()
    Y, escaped, is_device = _run_walk_one_ensemble(ens, kio, cfg, seed, verbose)

    n_escaped = int(escaped.sum())
    n_total = int(escaped.size)
    if n_escaped / max(n_total, 1) > cfg.max_escape_frac:
        raise RuntimeError(
            f"Escape rate {n_escaped/n_total*100:.2f}% exceeds "
            f"max_escape_frac={cfg.max_escape_frac*100:.1f}%.")

    keep = escaped == 0
    Y_host = Y.copy_to_host() if is_device else Y
    Y_host = Y_host[keep]
    return Y_host, n_escaped


# ===================================================================
#  Reduced (production) API: walk + on-device reduction, chunked over
#  walkers, accumulated across chunks and ensembles.
# ===================================================================

@dataclass
class ReducedResult:
    cos_sum:        np.ndarray   # (n_cols,) float64 — Σ cos(phase)
    sin_sum:        np.ndarray   # (n_cols,) float64 — Σ sin(phase)
    n_walkers_kept: int          # total kept walkers, summed over ensembles
    n_escaped:      int

    @property
    def n_eff(self) -> int:
        """3 axes harvested per walker."""
        return 3 * self.n_walkers_kept


def _walk_and_reduce_one_ensemble(
    ens: Ensemble, kio: float, cfg: SimConfig,
    j_delta: np.ndarray, j_Delta: np.ndarray, j_sum: np.ndarray,
    phase_coef: np.ndarray,
    walk_seed: int, verbose: bool,
):
    """Run one ensemble's walk and reduce it to Σcos/Σsin per column,
    chunking over walkers per cfg.walker_chunk (None = no chunking)."""
    n_cols = phase_coef.shape[0]
    cos_total = np.zeros(n_cols, dtype=np.float64)
    sin_total = np.zeros(n_cols, dtype=np.float64)
    n_kept_total = 0
    n_escaped_total = 0

    chunk = cfg.walker_chunk or cfg.n_walkers
    n_walkers = cfg.n_walkers
    offset = 0
    while offset < n_walkers:
        this_chunk = min(chunk, n_walkers - offset)
        sub_cfg = cfg if this_chunk == cfg.n_walkers else _with_n_walkers(cfg, this_chunk)

        Y, escaped, is_device = _run_walk_one_ensemble(
            ens, kio, sub_cfg, walk_seed + offset, verbose=False)

        n_escaped_chunk = int(escaped.sum())
        n_escaped_total += n_escaped_chunk
        keep_idx = np.where(escaped == 0)[0].astype(np.int32)
        n_kept_total += int(keep_idx.size)

        if is_device:
            d_keep_idx = cuda.to_device(keep_idx)
            d_j_delta = cuda.to_device(j_delta.astype(np.int32))
            d_j_Delta = cuda.to_device(j_Delta.astype(np.int32))
            d_j_sum = cuda.to_device(j_sum.astype(np.int32))
            d_phase_coef = cuda.to_device(phase_coef.astype(np.float64))
            d_cos_sum = cuda.device_array(n_cols, dtype=np.float64)
            d_sin_sum = cuda.device_array(n_cols, dtype=np.float64)

            _reduce_kernel[n_cols, _REDUCE_THREADS](
                Y, d_keep_idx, d_j_delta, d_j_Delta, d_j_sum, d_phase_coef,
                d_cos_sum, d_sin_sum,
            )
            cuda.synchronize()
            cos_total += d_cos_sum.copy_to_host()
            sin_total += d_sin_sum.copy_to_host()
        else:
            cos_chunk, sin_chunk = _reduce_cpu(
                Y, keep_idx, j_delta, j_Delta, j_sum, phase_coef)
            cos_total += cos_chunk
            sin_total += sin_chunk

        offset += this_chunk

    return cos_total, sin_total, n_kept_total, n_escaped_total


def _with_n_walkers(cfg: SimConfig, n: int) -> SimConfig:
    from dataclasses import replace
    return replace(cfg, n_walkers=n)


def run_simulation_reduced(
    rho: float, V: float, kio: float,
    j_delta: np.ndarray, j_Delta: np.ndarray, j_sum: np.ndarray,
    phase_coef: np.ndarray,
    cfg: Optional[SimConfig] = None,
    seed: int = 0,
    verbose: bool = True,
) -> ReducedResult:
    """Build cfg.n_ensembles independent ensembles for one (ρ,V,kio) library
    entry, walk + reduce each, and accumulate Σcos/Σsin across all of them.

    `j_delta, j_Delta, j_sum, phase_coef` are the flattened per-column
    lookup/coefficient arrays from `signal.build_columns`.
    """
    if cfg is None:
        cfg = SimConfig()

    total_walks = 3 * cfg.n_ensembles * cfg.n_walkers
    if total_walks < PAPER_WALKS_PER_ENTRY // 10:
        sys.stderr.write(
            f"    ⚠ Only {total_walks:,} walks per library entry "
            f"(paper target ~{PAPER_WALKS_PER_ENTRY:,}). "
            f"High-b columns will be noisy.\n")

    n_cols = phase_coef.shape[0]
    cos_total = np.zeros(n_cols, dtype=np.float64)
    sin_total = np.zeros(n_cols, dtype=np.float64)
    n_kept_total = 0
    n_escaped_total = 0

    for ei in range(cfg.n_ensembles):
        if verbose:
            print(f"  Ensemble {ei+1}/{cfg.n_ensembles}  "
                  f"(rho={rho:.0f}, V={V:.2f}, kio={kio:.1f})")

        ens_seed = _ensemble_seed(seed, ei)
        walk_seed = _walk_seed(seed, kio, ei)

        if rho <= 0 or V <= 0:
            ens = create_dummy_ensemble(cfg)
        else:
            ens = create_ensemble(rho, V, cfg, seed=ens_seed, verbose=verbose)

        cos_c, sin_c, n_kept, n_esc = _walk_and_reduce_one_ensemble(
            ens, kio, cfg, j_delta, j_Delta, j_sum, phase_coef,
            walk_seed, verbose)

        cos_total += cos_c
        sin_total += sin_c
        n_kept_total += n_kept
        n_escaped_total += n_esc

    return ReducedResult(cos_sum=cos_total, sin_sum=sin_total,
                          n_walkers_kept=n_kept_total,
                          n_escaped=n_escaped_total)


def run_simulation_multi_kio_reduced(
    rho: float, V: float, kios: list,
    j_delta: np.ndarray, j_Delta: np.ndarray, j_sum: np.ndarray,
    phase_coef: np.ndarray,
    cfg: Optional[SimConfig] = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Ensemble-reuse across a kio sweep: build geometry once per ensemble
    index, then walk+reduce for every kio in `kios` before releasing it.

    Returns dict[kio] -> ReducedResult.
    """
    if cfg is None:
        cfg = SimConfig()

    kios = list(kios)

    total_walks = 3 * cfg.n_ensembles * cfg.n_walkers
    if total_walks < PAPER_WALKS_PER_ENTRY // 10:
        sys.stderr.write(
            f"    ⚠ Only {total_walks:,} walks per library entry "
            f"(paper target ~{PAPER_WALKS_PER_ENTRY:,}). "
            f"High-b columns will be noisy.\n")

    n_cols = phase_coef.shape[0]
    cos_per_kio = {k: np.zeros(n_cols, dtype=np.float64) for k in kios}
    sin_per_kio = {k: np.zeros(n_cols, dtype=np.float64) for k in kios}
    n_kept_per_kio = {k: 0 for k in kios}
    n_escaped_per_kio = {k: 0 for k in kios}

    for ei in range(cfg.n_ensembles):
        ens_seed = _ensemble_seed(seed, ei)

        if verbose:
            print(f"  Ensemble {ei+1}/{cfg.n_ensembles}  "
                  f"(rho={rho/1e3:.0f}k, V={V:.2f})  "
                  f"building once, sweeping {len(kios)} kio values")

        if rho <= 0 or V <= 0:
            ens = create_dummy_ensemble(cfg)
        else:
            ens = create_ensemble(rho, V, cfg, seed=ens_seed, verbose=False)

        for i, kio in enumerate(kios):
            walk_seed = _walk_seed(seed, kio, ei)
            cos_c, sin_c, n_kept, n_esc = _walk_and_reduce_one_ensemble(
                ens, kio, cfg, j_delta, j_Delta, j_sum, phase_coef,
                walk_seed, verbose=False)
            cos_per_kio[kio] += cos_c
            sin_per_kio[kio] += sin_c
            n_kept_per_kio[kio] += n_kept
            n_escaped_per_kio[kio] += n_esc

            if verbose:
                print(f"    kio={kio:.1f}  ({i+1}/{len(kios)})  "
                      f"kept={n_kept}  escaped={n_esc}")

        # `ens` goes out of scope at next iteration → freed

    return {
        kio: ReducedResult(cos_sum=cos_per_kio[kio], sin_sum=sin_per_kio[kio],
                            n_walkers_kept=n_kept_per_kio[kio],
                            n_escaped=n_escaped_per_kio[kio])
        for kio in kios
    }
