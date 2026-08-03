"""Monte-Carlo walkers for the periodic MADI forward model.

There is one production membrane algorithm.  CPU and CUDA both:

* use the same periodic K-candidate classifier from :mod:`madi.ensemble`;
* re-rank candidates at the actual walker position;
* count one or two membrane crossings from old/new cell state;
* accept a crossing with ``p_p**m`` and otherwise revert to the old position;
* integrate the unwrapped position at every 1-µs step using the trapezoid
  rule; and
* retain direct residence-time telemetry used to label ``k_io``.

The legacy absorbing/drop-all boundary behaviour is retained only as an
explicit diagnostic mode for measuring its bias.  ``SimConfig`` defaults to
periodic, and library construction rejects the legacy mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import sys
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import SimConfig
from .ensemble import Ensemble, create_dummy_ensemble, create_ensemble

try:
    from numba import cuda
    import numba
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_normal_float64,
        xoroshiro128p_uniform_float64,
    )
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False

if not HAS_CUDA:
    print("WARNING: CUDA not available — falling back to CPU (slow)")


PAPER_WALKS_PER_ENTRY = 12_000_000
_REDUCE_THREADS = 256


# ---------------------------------------------------------------------------
# Analytic Eq. 5 comparator -- retained for metadata, never authoritative.
# ---------------------------------------------------------------------------

def pp_to_kio_eq5(pp: float, mean_AV: float, cfg: SimConfig) -> float:
    """Paper Eq. 5 analytic label, in s⁻¹, for comparison only.

    The primary library label is instead the directly measured tagged-
    starting-cell first-exit hazard.  Keeping this function explicit lets
    every entry record the Gaussian-step/Eq.-5 discrepancy rather than
    hiding it.
    """
    return float(pp) * math.sqrt(cfg.D0 / (6.0 * cfg.ts)) * float(mean_AV) * 1000.0


def kio_to_pp(kio: float, mean_AV: float, cfg: SimConfig) -> float:
    """Inverse of analytic Eq. 5; used only as a calibration starting point."""
    if kio < 0.0:
        raise ValueError(f"k_io must be non-negative, got {kio}")
    if mean_AV <= 0.0:
        raise ValueError("mean_A_over_V must be positive for a cellular ensemble")
    return (float(kio) / 1000.0) / (math.sqrt(cfg.D0 / (6.0 * cfg.ts)) * mean_AV)


def pp_to_kio(pp: float, mean_AV: float, cfg: SimConfig) -> float:
    """Backward-compatible alias for the explicitly named Eq.-5 comparator."""
    return pp_to_kio_eq5(pp, mean_AV, cfg)


def _checked_pp(pp: float) -> float:
    pp = float(pp)
    if not np.isfinite(pp) or pp < 0.0 or pp > 1.0:
        raise ValueError(f"p_p must be finite and in [0, 1], got {pp!r}")
    return pp


def _ensemble_seed(base_seed: int, ensemble_index: int) -> int:
    return (int(base_seed) + ensemble_index * 97_003) & 0x7FFFFFFF


def _walk_seed(base_seed: int, ensemble_index: int) -> int:
    """CRN seed shared across neighbouring rho/V/**and k_io** entries."""
    return (int(base_seed) + ensemble_index * 104_729) & 0x7FFFFFFF


@dataclass(frozen=True)
class ExchangeCalibration:
    """One target-rate selection from a direct, per-geometry response curve.

    ``target_kio`` is only the requested grid coordinate.  ``pp_used`` is
    chosen by inverting ``response`` and the final library label is measured
    again during the signal walk.  This deliberately prevents a target grid
    coordinate, or the paper's Eq. 5, from becoming a false physics label.
    """

    target_kio: float
    pp_initial_eq5: float
    pp_used: float
    response: "ExchangeResponse | None"
    interpolation: str = "monotone_piecewise_linear"


@dataclass(frozen=True)
class WalkRandomStream:
    """Pre-generated inputs for a CPU/CUDA golden walk.

    Production runs use the native CPU/CUDA generators for performance.  A
    golden harness instead supplies *identical* initial positions, Gaussian
    proposals and membrane-acceptance uniforms to both paths.  That makes
    CPU/GPU equivalence a comparison of the shared classifier/crossing
    algorithm, not a comparison of unrelated random streams.
    """

    initial_positions: np.ndarray      # (N, 3), wrapped coordinates [µm]
    increments: np.ndarray             # (n_steps, N, 3), Gaussian [µm]
    acceptance_uniforms: np.ndarray    # (n_steps, N), U[0,1)

    def validate(self, cfg: SimConfig) -> None:
        n = int(cfg.n_walkers)
        expected = (int(cfg.n_steps), n, 3)
        if np.asarray(self.initial_positions).shape != (n, 3):
            raise ValueError("golden initial_positions must have shape (n_walkers, 3)")
        if np.asarray(self.increments).shape != expected:
            raise ValueError(
                "golden increments must have shape "
                f"(n_steps, n_walkers, 3)={expected}"
            )
        if np.asarray(self.acceptance_uniforms).shape != (int(cfg.n_steps), n):
            raise ValueError("golden acceptance_uniforms have an invalid shape")
        if not (np.all(np.isfinite(self.initial_positions))
                and np.all(np.isfinite(self.increments))
                and np.all(np.isfinite(self.acceptance_uniforms))):
            raise ValueError("golden random stream contains a non-finite value")
        if (np.any(self.initial_positions < 0.0)
                or np.any(self.initial_positions >= float(cfg.L))
                or np.any(self.acceptance_uniforms < 0.0)
                or np.any(self.acceptance_uniforms >= 1.0)):
            raise ValueError("golden random stream lies outside its physical range")


def make_walk_random_stream(cfg: SimConfig, seed: int) -> WalkRandomStream:
    """Create a deterministic, serialisable stream for the GPU golden test."""
    rng = np.random.default_rng(int(seed))
    n = int(cfg.n_walkers)
    stream = WalkRandomStream(
        initial_positions=rng.uniform(0.0, float(cfg.L), size=(n, 3)).astype(np.float64),
        increments=rng.normal(
            0.0, float(cfg.sigma), size=(int(cfg.n_steps), n, 3)
        ).astype(np.float64),
        acceptance_uniforms=rng.uniform(
            0.0, 1.0, size=(int(cfg.n_steps), n)
        ).astype(np.float64),
    )
    stream.validate(cfg)
    return stream


@dataclass(frozen=True)
class ExchangeResponse:
    """Direct tagged-starting-cell p_p response for one realised geometry.

    All values are measured before the signal-producing k_io sweep.  The
    raw event rates are retained verbatim; ``monotone_event_rates_s_inv`` is
    a weighted isotonic fit used only to invert the physically monotone
    response.  Recording both makes any non-constant Eq.-5 discrepancy
    auditable rather than hiding it behind a global scale factor.
    """

    pp_values: tuple[float, ...]
    event_rates_s_inv: tuple[float, ...]
    event_rate_ses_s_inv: tuple[float, ...]
    survival_fit_rates_s_inv: tuple[float, ...]
    monotone_event_rates_s_inv: tuple[float, ...]

    def rate_at_pp(self, pp: float) -> float:
        """Piecewise-linear measured response, including the p=0 origin."""
        return float(np.interp(
            float(pp), np.asarray(self.pp_values, dtype=float),
            np.asarray(self.monotone_event_rates_s_inv, dtype=float),
        ))

    @property
    def max_rate_s_inv(self) -> float:
        return float(self.monotone_event_rates_s_inv[-1])

    @property
    def max_pp(self) -> float:
        return float(self.pp_values[-1])

    def pp_for_rate(self, target_kio: float) -> float:
        """Invert the measured monotone curve, loudly rejecting overflow."""
        target = float(target_kio)
        if target < 0.0:
            raise ValueError(f"target k_io must be non-negative, got {target}")
        if target == 0.0:
            return 0.0
        if target > self.max_rate_s_inv:
            raise RuntimeError(
                "Requested k_io exceeds the directly measured response at "
                f"p_p={self.max_pp:.6g}: target={target:.6g} s^-1, "
                f"maximum={self.max_rate_s_inv:.6g} s^-1.  Refusing to "
                "silently clamp p_p."
            )
        return _checked_pp(float(np.interp(
            target,
            np.asarray(self.monotone_event_rates_s_inv, dtype=float),
            np.asarray(self.pp_values, dtype=float),
        )))


@dataclass(frozen=True)
class ExchangeMeasurement:
    """Sufficient statistics for a tagged-starting-cell calibration probe."""

    start_survival_time_ms: float
    first_exit_events: int
    start_survivor_counts: tuple[int, ...]
    start_initial_intra: int
    checkpoint_h_ms: float

    @property
    def event_rate_s_inv(self) -> float:
        if self.start_survival_time_ms <= 0.0:
            return 0.0
        return 1000.0 * self.first_exit_events / self.start_survival_time_ms

    @property
    def event_rate_se_s_inv(self) -> float:
        if self.first_exit_events <= 0 or self.start_survival_time_ms <= 0.0:
            return 0.0
        return 1000.0 * math.sqrt(self.first_exit_events) / self.start_survival_time_ms

    @property
    def survival_fit_s_inv(self) -> float:
        return _survival_fit_rate(
            np.asarray(self.start_survivor_counts, dtype=np.int64),
            self.start_initial_intra,
            self.checkpoint_h_ms,
        )


@dataclass
class ReducedResult:
    cos_sum: np.ndarray
    sin_sum: np.ndarray
    n_walkers_kept: int
    n_escaped: int
    n_walkers_total: int
    intra_time_ms: float
    efflux_events: int
    influx_events: int
    initial_intra: int
    final_intra: int
    occupancy_counts: np.ndarray
    occupancy_denominator: int
    start_survival_time_ms: float
    first_exit_events: int
    start_survivor_counts: np.ndarray
    start_initial_intra: int
    checkpoint_h_ms: float
    pp_values: list[float] = field(default_factory=list)
    analytic_kio_eq5_values: list[float] = field(default_factory=list)
    calibration: list[ExchangeCalibration] = field(default_factory=list)
    geometry_stats: list[dict] = field(default_factory=list)

    @property
    def n_eff(self) -> int:
        """Three Cartesian axes are isotropically harvested per walker."""
        return 3 * self.n_walkers_kept

    @property
    def measured_kio(self) -> float:
        """Authoritative first-exit k_io from tagged starting-cell walkers."""
        if self.start_survival_time_ms <= 0.0:
            return 0.0
        return 1000.0 * float(self.first_exit_events) / float(self.start_survival_time_ms)

    @property
    def measured_kio_se(self) -> float:
        """Poisson counting SE of the tagged first-exit hazard estimator."""
        if self.first_exit_events <= 0 or self.start_survival_time_ms <= 0.0:
            return 0.0
        return 1000.0 * math.sqrt(float(self.first_exit_events)) / float(self.start_survival_time_ms)

    @property
    def stationary_kio(self) -> float:
        """Independent all-cell residence-transition diagnostic rate."""
        if self.intra_time_ms <= 0.0:
            return 0.0
        return 1000.0 * float(self.efflux_events) / float(self.intra_time_ms)

    @property
    def kio_survival_fit(self) -> float:
        """First-exit rate from a log-linear tagged-survivor fit in s⁻¹.

        The fit uses the nonzero portion of S_start(t)/S_start(0), retaining
        at least the early 10% survival range.  Event counting is the label;
        this is an independent representation check stored beside it.
        """
        return _survival_fit_rate(
            self.start_survivor_counts, self.start_initial_intra, self.checkpoint_h_ms
        )

    @property
    def analytic_kio_eq5(self) -> float:
        if not self.analytic_kio_eq5_values:
            return 0.0
        return float(np.mean(self.analytic_kio_eq5_values))

    @property
    def mean_pp(self) -> float:
        return float(np.mean(self.pp_values)) if self.pp_values else 0.0

    @property
    def occupancy_fraction(self) -> np.ndarray:
        if self.occupancy_denominator <= 0:
            return np.zeros_like(self.occupancy_counts, dtype=np.float64)
        return self.occupancy_counts.astype(np.float64) / self.occupancy_denominator


def _survival_fit_rate(counts: np.ndarray, initial: int, h_ms: float) -> float:
    """Log-linear first-exit rate fit shared by probe and full trajectories."""
    if initial <= 0 or len(counts) < 3:
        return 0.0
    t = np.arange(len(counts), dtype=float) * float(h_ms)
    fraction = np.asarray(counts, dtype=float) / float(initial)
    keep = (fraction > 0.10) & (fraction <= 1.0)
    if np.count_nonzero(keep) < 2:
        keep = fraction > 0.0
    if np.count_nonzero(keep) < 2:
        return 0.0
    slope = np.polyfit(t[keep], np.log(fraction[keep]), 1)[0]
    return max(0.0, -1000.0 * float(slope))


def _membrane_transition_counts(
    old_cell: np.ndarray,
    old_inside: np.ndarray,
    new_cell: np.ndarray,
    new_inside: np.ndarray,
) -> np.ndarray:
    """Number of membranes crossed by the proposed endpoint transition."""
    m = np.zeros(len(old_cell), dtype=np.int8)
    m[old_inside & ~new_inside] = 1
    m[~old_inside & new_inside] = 1
    m[old_inside & new_inside & (old_cell != new_cell)] = 2
    return m


def _efflux_and_influx(
    old_cell: np.ndarray,
    old_inside: np.ndarray,
    new_cell: np.ndarray,
    new_inside: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Count cell residence exits/entries after an accepted proposal.

    An inside-cell ``i -> j`` jump crosses two membranes in one MC step; it
    counts both one efflux from ``i`` and one influx to ``j``.
    """
    changed_cell = old_cell != new_cell
    efflux = old_inside & ((~new_inside) | changed_cell)
    influx = new_inside & ((~old_inside) | changed_cell)
    return efflux, influx


# ===========================================================================
# CUDA implementation.  It is a direct transliteration of _walk_cpu.
# ===========================================================================

if HAS_CUDA:

    @cuda.jit
    def _walk_kernel(
        seeds,
        annulus,
        grid_candidates,
        grid_spacing,
        G_grid,
        L,
        sigma,
        ts,
        n_steps,
        pp,
        boundary_periodic,
        has_cells,
        lo,
        hi,
        half_L,
        steps_per_h,
        rng_states,
        use_golden_stream,
        golden_initial_positions,
        golden_increments,
        golden_acceptance_uniforms,
        Y_out,
        inside_trace,
        start_trace,
        metrics_out,
        escaped_out,
    ):
        tid = cuda.grid(1)
        if tid >= Y_out.shape[0]:
            return

        # Periodic production starts uniformly over the full volume.  The
        # legacy diagnostic reproduces the old buffered start condition.
        if use_golden_stream == 1:
            px = golden_initial_positions[tid, 0]
            py = golden_initial_positions[tid, 1]
            pz = golden_initial_positions[tid, 2]
        elif boundary_periodic == 1:
            px = xoroshiro128p_uniform_float64(rng_states, tid) * L
            py = xoroshiro128p_uniform_float64(rng_states, tid) * L
            pz = xoroshiro128p_uniform_float64(rng_states, tid) * L
        else:
            px = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo
            py = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo
            pz = xoroshiro128p_uniform_float64(rng_states, tid) * (hi - lo) + lo
        ux = px
        uy = py
        uz = pz

        def _wrap(x):
            return x - math.floor(x / L) * L

        def _classify(cx, cy, cz):
            # cx/cy/cz must be wrapped in [0,L).  This algorithm is shared
            # with ensemble._classify_candidate_rows: cache -> re-rank K ->
            # closest-two periodic bisector -> contracted-cell test.
            gx = int(cx / grid_spacing)
            gy = int(cy / grid_spacing)
            gz = int(cz / grid_spacing)
            gx = min(max(gx, 0), G_grid - 1)
            gy = min(max(gy, 0), G_grid - 1)
            gz = min(max(gz, 0), G_grid - 1)
            best1 = -1
            best2 = -1
            best1d = math.inf
            best2d = math.inf
            best1x = 0.0; best1y = 0.0; best1z = 0.0
            best2x = 0.0; best2y = 0.0; best2z = 0.0
            for ci in range(grid_candidates.shape[3]):
                sid = grid_candidates[gx, gy, gz, ci]
                if sid < 0:
                    continue
                sx = seeds[sid, 0]
                sy = seeds[sid, 1]
                sz = seeds[sid, 2]
                sx += math.floor((cx - sx) / L + 0.5) * L
                sy += math.floor((cy - sy) / L + 0.5) * L
                sz += math.floor((cz - sz) / L + 0.5) * L
                dx = cx - sx; dy = cy - sy; dz = cz - sz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < best1d:
                    best2 = best1; best2d = best1d
                    best2x = best1x; best2y = best1y; best2z = best1z
                    best1 = sid; best1d = d2
                    best1x = sx; best1y = sy; best1z = sz
                elif d2 < best2d:
                    best2 = sid; best2d = d2
                    best2x = sx; best2y = sy; best2z = sz
            if best1 < 0 or best2 < 0:
                return 0, False
            dx = best2x - best1x
            dy = best2y - best1y
            dz = best2z - best1z
            norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            if norm < 1e-30:
                return best1, True
            mx = 0.5 * (best1x + best2x)
            my = 0.5 * (best1y + best2y)
            mz = 0.5 * (best1z + best2z)
            signed = ((cx - mx) * dx + (cy - my) * dy + (cz - mz) * dz) / norm
            return best1, abs(signed) >= annulus[best1]

        if has_cells == 1:
            cur_cell, cur_inside = _classify(px, py, pz)
        else:
            cur_cell = 0
            cur_inside = False
        start_cell = cur_cell
        start_inside = cur_inside
        ever_left_start = False

        xs = ux - half_L
        ys = uy - half_L
        zs = uz - half_L
        Yx = 0.0; Yy = 0.0; Yz = 0.0
        Y_out[tid, 0, 0] = 0.0
        Y_out[tid, 0, 1] = 0.0
        Y_out[tid, 0, 2] = 0.0
        inside_trace[tid, 0] = 1 if cur_inside else 0
        start_trace[tid, 0] = 1 if start_inside else 0
        intra_time = 0.0
        efflux_count = 0.0
        influx_count = 0.0
        start_survival_time = 0.0
        first_exit_count = 0.0
        escaped = 0

        for step in range(n_steps):
            if escaped == 1:
                continue

            if cur_inside:
                intra_time += ts
            in_start_cell = (start_inside and (not ever_left_start)
                             and cur_inside and cur_cell == start_cell)
            if in_start_cell:
                start_survival_time += ts

            if use_golden_stream == 1:
                dxp = golden_increments[step, tid, 0]
                dyp = golden_increments[step, tid, 1]
                dzp = golden_increments[step, tid, 2]
            else:
                dxp = xoroshiro128p_normal_float64(rng_states, tid) * sigma
                dyp = xoroshiro128p_normal_float64(rng_states, tid) * sigma
                dzp = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            # Consume exactly one acceptance uniform for every walker-step,
            # regardless of whether this proposal touches a membrane.  This
            # is required for *actual* common random numbers across nearby
            # geometries/k_io values: a conditional RNG draw would otherwise
            # desynchronise all later increments after the first differing
            # crossing event.
            if use_golden_stream == 1:
                acceptance_uniform = golden_acceptance_uniforms[step, tid]
            else:
                acceptance_uniform = xoroshiro128p_uniform_float64(rng_states, tid)
            nux = ux + dxp
            nuy = uy + dyp
            nuz = uz + dzp
            nx = px + dxp
            ny = py + dyp
            nz = pz + dzp

            if boundary_periodic == 1:
                nx = _wrap(nx)
                ny = _wrap(ny)
                nz = _wrap(nz)
            else:
                if nx < 0.0 or nx >= L or ny < 0.0 or ny >= L or nz < 0.0 or nz >= L:
                    escaped = 1
                    continue

            if has_cells == 1:
                new_cell, new_inside = _classify(nx, ny, nz)
            else:
                new_cell = 0
                new_inside = False

            m = 0
            if cur_inside and not new_inside:
                m = 1
            elif (not cur_inside) and new_inside:
                m = 1
            elif cur_inside and new_inside and cur_cell != new_cell:
                m = 2
            if m > 0 and pp < 1.0:
                pp_m = pp
                if m == 2:
                    pp_m = pp * pp
                if acceptance_uniform >= pp_m:
                    nx = px; ny = py; nz = pz
                    nux = ux; nuy = uy; nuz = uz
                    new_cell = cur_cell
                    new_inside = cur_inside

            # Stationary residence-time telemetry after acceptance.
            if cur_inside and ((not new_inside) or cur_cell != new_cell):
                efflux_count += 1.0
            if new_inside and ((not cur_inside) or cur_cell != new_cell):
                influx_count += 1.0
            if in_start_cell and ((not new_inside) or new_cell != start_cell):
                first_exit_count += 1.0
                ever_left_start = True

            px = nx; py = ny; pz = nz
            ux = nux; uy = nuy; uz = nuz
            cur_cell = new_cell
            cur_inside = new_inside

            nxs = ux - half_L
            nys = uy - half_L
            nzs = uz - half_L
            Yx += 0.5 * (xs + nxs) * ts
            Yy += 0.5 * (ys + nys) * ts
            Yz += 0.5 * (zs + nzs) * ts
            xs = nxs; ys = nys; zs = nzs

            step_idx = step + 1
            if step_idx % steps_per_h == 0:
                j = step_idx // steps_per_h
                Y_out[tid, j, 0] = Yx
                Y_out[tid, j, 1] = Yy
                Y_out[tid, j, 2] = Yz
                inside_trace[tid, j] = 1 if cur_inside else 0
                start_trace[tid, j] = 1 if (start_inside and (not ever_left_start)) else 0

        metrics_out[tid, 0] = 1.0 if inside_trace[tid, 0] != 0 else 0.0
        metrics_out[tid, 1] = 1.0 if cur_inside else 0.0
        metrics_out[tid, 2] = intra_time
        metrics_out[tid, 3] = efflux_count
        metrics_out[tid, 4] = influx_count
        metrics_out[tid, 5] = start_survival_time
        metrics_out[tid, 6] = first_exit_count
        escaped_out[tid] = escaped

    @cuda.jit
    def _reduce_kernel(
        Y,
        keep_idx,
        j_delta,
        j_Delta,
        j_sum,
        phase_coef,
        cos_sum_out,
        sin_sum_out,
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
        n_total = keep_idx.shape[0] * 3
        local_cos = 0.0
        local_sin = 0.0
        i = tid
        while i < n_total:
            w = keep_idx[i // 3]
            axis = i % 3
            dM = Y[w, jd, axis] + Y[w, jD, axis] - Y[w, js, axis]
            phase = pc * dM
            local_cos += math.cos(phase)
            local_sin += math.sin(phase)
            i += block_size
        shared_cos = cuda.shared.array(_REDUCE_THREADS, dtype=numba.float64)
        shared_sin = cuda.shared.array(_REDUCE_THREADS, dtype=numba.float64)
        shared_cos[tid] = local_cos
        shared_sin[tid] = local_sin
        cuda.syncthreads()
        stride = block_size // 2
        while stride > 0:
            if tid < stride:
                shared_cos[tid] += shared_cos[tid + stride]
                shared_sin[tid] += shared_sin[tid + stride]
            cuda.syncthreads()
            stride //= 2
        if tid == 0:
            cos_sum_out[col] = shared_cos[0]
            sin_sum_out[col] = shared_sin[0]


# ===========================================================================
# CPU reference implementation.
# ===========================================================================

def _walk_cpu(
    ens: Ensemble,
    pp: float,
    cfg: SimConfig,
    seed: int = 0,
    classifier: str = "cache",
    random_stream: Optional[WalkRandomStream] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPU transliteration of the CUDA transition, for validation/fallback."""
    pp = _checked_pp(pp)
    if random_stream is not None:
        random_stream.validate(cfg)
    rng = np.random.default_rng(seed) if random_stream is None else None
    N = int(cfg.n_walkers)
    L = float(cfg.L)
    periodic = cfg.boundary_mode == "periodic"
    if random_stream is not None:
        positions = np.asarray(random_stream.initial_positions, dtype=np.float64).copy()
    elif periodic:
        positions = rng.uniform(0.0, L, size=(N, 3))
    else:
        positions = rng.uniform(cfg.buffer, L - cfg.buffer, size=(N, 3))
    unwrapped = positions.copy()
    if classifier not in {"cache", "exact"}:
        raise ValueError("classifier must be 'cache' or 'exact'")
    classify = ens.classify_cpu if classifier == "cache" else ens.classify_exact_cpu
    if ens.is_free_water:
        cur_cell = np.zeros(N, dtype=np.int32)
        cur_inside = np.zeros(N, dtype=bool)
    else:
        cur_cell, cur_inside = classify(positions)
    initial_inside = cur_inside.copy()
    start_cell = cur_cell.copy()
    ever_left_start = np.zeros(N, dtype=bool)
    escaped = np.zeros(N, dtype=bool)

    Y = np.zeros((N, cfg.n_grid, 3), dtype=np.float64)
    inside_trace = np.zeros((N, cfg.n_grid), dtype=np.int8)
    inside_trace[:, 0] = cur_inside.astype(np.int8)
    start_trace = np.zeros((N, cfg.n_grid), dtype=np.int8)
    start_trace[:, 0] = initial_inside.astype(np.int8)
    Y_run = np.zeros((N, 3), dtype=np.float64)
    previous = unwrapped - L / 2.0
    intra_time = np.zeros(N, dtype=np.float64)
    efflux = np.zeros(N, dtype=np.float64)
    influx = np.zeros(N, dtype=np.float64)
    start_survival_time = np.zeros(N, dtype=np.float64)
    first_exit = np.zeros(N, dtype=np.float64)

    for step in range(cfg.n_steps):
        active = ~escaped
        if not np.any(active):
            break
        intra_time[active] += cur_inside[active].astype(np.float64) * cfg.ts
        in_start_cell = (
            initial_inside & ~ever_left_start & cur_inside & (cur_cell == start_cell) & active
        )
        start_survival_time[in_start_cell] += cfg.ts
        if random_stream is None:
            increments = rng.normal(0.0, cfg.sigma, size=(N, 3))
            # Keep the random stream aligned across every geometry and p_p
            # value. See the matching CUDA code above.
            acceptance_uniform = rng.uniform(0.0, 1.0, size=N)
        else:
            increments = np.asarray(random_stream.increments[step], dtype=np.float64)
            acceptance_uniform = np.asarray(
                random_stream.acceptance_uniforms[step], dtype=np.float64
            )
        proposal_unwrapped = unwrapped + increments
        proposal = positions + increments
        if periodic:
            proposal = np.mod(proposal, L)
        else:
            out = ((proposal < 0.0).any(axis=1) | (proposal >= L).any(axis=1)) & active
            escaped[out] = True
            proposal[out] = positions[out]
            proposal_unwrapped[out] = unwrapped[out]
            active = ~escaped

        if ens.is_free_water:
            new_cell = np.zeros(N, dtype=np.int32)
            new_inside = np.zeros(N, dtype=bool)
        else:
            new_cell, new_inside = classify(proposal)
        m = _membrane_transition_counts(cur_cell, cur_inside, new_cell, new_inside)
        crossing = active & (m > 0)
        if np.any(crossing) and pp < 1.0:
            draw = acceptance_uniform[crossing]
            reject = draw >= pp ** m[crossing]
            rejected = np.flatnonzero(crossing)[reject]
            proposal[rejected] = positions[rejected]
            proposal_unwrapped[rejected] = unwrapped[rejected]
            new_cell[rejected] = cur_cell[rejected]
            new_inside[rejected] = cur_inside[rejected]

        out_events, in_events = _efflux_and_influx(cur_cell, cur_inside, new_cell, new_inside)
        efflux[active] += out_events[active]
        influx[active] += in_events[active]
        first_departure = in_start_cell & ((~new_inside) | (new_cell != start_cell))
        first_exit[first_departure] += 1.0
        ever_left_start[first_departure] = True

        positions[active] = proposal[active]
        unwrapped[active] = proposal_unwrapped[active]
        cur_cell[active] = new_cell[active]
        cur_inside[active] = new_inside[active]

        current = unwrapped - L / 2.0
        Y_run[active] += 0.5 * (previous[active] + current[active]) * cfg.ts
        previous[active] = current[active]

        step_idx = step + 1
        if step_idx % cfg.steps_per_h == 0:
            j = step_idx // cfg.steps_per_h
            Y[:, j, :] = Y_run
            inside_trace[:, j] = cur_inside.astype(np.int8)
            start_trace[:, j] = (initial_inside & ~ever_left_start).astype(np.int8)

    metrics = np.column_stack(
        (
            initial_inside.astype(np.float64),
            cur_inside.astype(np.float64),
            intra_time,
            efflux,
            influx,
            start_survival_time,
            first_exit,
        )
    )
    return Y, escaped.astype(np.int8), metrics, inside_trace, start_trace


def _run_walk_one_ensemble(
    ens: Ensemble,
    pp: float,
    cfg: SimConfig,
    seed: int,
    verbose: bool,
    classifier: str = "cache",
    random_stream: Optional[WalkRandomStream] = None,
    use_gpu: Optional[bool] = None,
) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Run one ensemble and return Y, escape/telemetry arrays, device flag."""
    pp = _checked_pp(pp)
    if verbose:
        print(
            f"    p_p={pp:.7f}; Eq.-5 comparator="
            f"{pp_to_kio_eq5(pp, ens.mean_AV, cfg):.4f} s^-1"
        )
    if classifier not in {"cache", "exact"}:
        raise ValueError("classifier must be 'cache' or 'exact'")
    if random_stream is not None:
        random_stream.validate(cfg)
    if use_gpu is True and not HAS_CUDA:
        raise RuntimeError("GPU golden run requested but CUDA is unavailable")
    if use_gpu is True and classifier == "exact":
        raise ValueError("exact KD classification is CPU validation-only")
    run_on_gpu = HAS_CUDA if use_gpu is None else bool(use_gpu)
    # Exact KD classification is validation-only and deliberately cannot be
    # requested on CUDA.  The production cache is independently bounded by
    # Tier-A CPU tests and checked against CUDA with the golden harness.
    if not run_on_gpu or classifier == "exact":
        Y, escaped, metrics, inside_trace, start_trace = _walk_cpu(
            ens, pp, cfg, seed, classifier=classifier, random_stream=random_stream
        )
        return Y, escaped, metrics, inside_trace, start_trace, False

    N = int(cfg.n_walkers)
    d_seeds = cuda.to_device(ens.seeds)
    d_annulus = cuda.to_device(ens.annulus)
    d_candidates = cuda.to_device(ens.grid_candidates)
    d_Y = cuda.device_array((N, cfg.n_grid, 3), dtype=np.float64)
    d_inside = cuda.device_array((N, cfg.n_grid), dtype=np.int8)
    d_start = cuda.device_array((N, cfg.n_grid), dtype=np.int8)
    d_metrics = cuda.device_array((N, 7), dtype=np.float64)
    d_escaped = cuda.device_array(N, dtype=np.int8)
    states = create_xoroshiro128p_states(N, seed=seed)
    if random_stream is None:
        # The kernel accepts arrays in both modes so it has a single compiled
        # implementation.  These one-element placeholders are never read in
        # native-RNG mode.
        d_golden_initial = cuda.to_device(np.zeros((1, 3), dtype=np.float64))
        d_golden_increments = cuda.to_device(np.zeros((1, 1, 3), dtype=np.float64))
        d_golden_uniforms = cuda.to_device(np.zeros((1, 1), dtype=np.float64))
        use_golden = np.int32(0)
    else:
        d_golden_initial = cuda.to_device(np.ascontiguousarray(
            random_stream.initial_positions, dtype=np.float64))
        d_golden_increments = cuda.to_device(np.ascontiguousarray(
            random_stream.increments, dtype=np.float64))
        d_golden_uniforms = cuda.to_device(np.ascontiguousarray(
            random_stream.acceptance_uniforms, dtype=np.float64))
        use_golden = np.int32(1)
    threads = 128
    blocks = (N + threads - 1) // threads
    _walk_kernel[blocks, threads](
        d_seeds, d_annulus, d_candidates,
        np.float64(ens.grid_spacing), np.int32(ens.grid_candidates.shape[0]),
        np.float64(ens.L), np.float64(cfg.sigma), np.float64(cfg.ts),
        np.int32(cfg.n_steps), np.float64(pp),
        np.int32(1 if cfg.boundary_mode == "periodic" else 0),
        np.int32(0 if ens.is_free_water else 1),
        np.float64(cfg.buffer), np.float64(ens.L - cfg.buffer), np.float64(ens.L / 2.0),
        np.int32(cfg.steps_per_h), states,
        use_golden, d_golden_initial, d_golden_increments, d_golden_uniforms,
        d_Y, d_inside, d_start, d_metrics, d_escaped,
    )
    cuda.synchronize()
    return (
        d_Y,
        d_escaped.copy_to_host(),
        d_metrics.copy_to_host(),
        d_inside.copy_to_host(),
        d_start.copy_to_host(),
        True,
    )


def _reduce_cpu(
    Y: np.ndarray,
    keep_idx: np.ndarray,
    j_delta: np.ndarray,
    j_Delta: np.ndarray,
    j_sum: np.ndarray,
    phase_coef: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    Yk = Y[keep_idx]
    cos_sum = np.empty(len(phase_coef), dtype=np.float64)
    sin_sum = np.empty(len(phase_coef), dtype=np.float64)
    for c in range(len(phase_coef)):
        dM = Yk[:, j_delta[c], :] + Yk[:, j_Delta[c], :] - Yk[:, j_sum[c], :]
        phase = phase_coef[c] * dM
        cos_sum[c] = np.cos(phase).sum()
        sin_sum[c] = np.sin(phase).sum()
    return cos_sum, sin_sum


def _with_n_walkers(cfg: SimConfig, n_walkers: int) -> SimConfig:
    return replace(cfg, n_walkers=int(n_walkers))


def _walk_and_reduce_one_ensemble(
    ens: Ensemble,
    pp: float,
    cfg: SimConfig,
    j_delta: np.ndarray,
    j_Delta: np.ndarray,
    j_sum: np.ndarray,
    phase_coef: np.ndarray,
    walk_seed: int,
    verbose: bool,
) -> ReducedResult:
    n_cols = len(phase_coef)
    cos_total = np.zeros(n_cols, dtype=np.float64)
    sin_total = np.zeros(n_cols, dtype=np.float64)
    kept = escaped_total = total = 0
    intra_time = 0.0
    efflux = influx = initial = final = 0
    occ_counts = np.zeros(cfg.n_grid, dtype=np.int64)
    start_counts = np.zeros(cfg.n_grid, dtype=np.int64)
    start_survival_time = 0.0
    first_exit = 0
    start_initial = 0
    chunk = int(cfg.walker_chunk or cfg.n_walkers)

    for offset in range(0, int(cfg.n_walkers), chunk):
        n_this = min(chunk, int(cfg.n_walkers) - offset)
        sub_cfg = cfg if n_this == cfg.n_walkers else _with_n_walkers(cfg, n_this)
        Y, escaped, metrics, inside_trace, start_trace, device = _run_walk_one_ensemble(
            ens, pp, sub_cfg, walk_seed + offset, verbose=False
        )
        keep_idx = np.flatnonzero(escaped == 0).astype(np.int32)
        if len(keep_idx) == 0:
            raise RuntimeError("All walkers escaped; refusing to build a selected library entry.")
        if device:
            d_keep = cuda.to_device(keep_idx)
            d_jd = cuda.to_device(np.asarray(j_delta, dtype=np.int32))
            d_jD = cuda.to_device(np.asarray(j_Delta, dtype=np.int32))
            d_js = cuda.to_device(np.asarray(j_sum, dtype=np.int32))
            d_pc = cuda.to_device(np.asarray(phase_coef, dtype=np.float64))
            d_cos = cuda.device_array(n_cols, dtype=np.float64)
            d_sin = cuda.device_array(n_cols, dtype=np.float64)
            _reduce_kernel[n_cols, _REDUCE_THREADS](Y, d_keep, d_jd, d_jD, d_js, d_pc, d_cos, d_sin)
            cuda.synchronize()
            cos_total += d_cos.copy_to_host()
            sin_total += d_sin.copy_to_host()
        else:
            c, s = _reduce_cpu(Y, keep_idx, j_delta, j_Delta, j_sum, phase_coef)
            cos_total += c
            sin_total += s

        kept += int(len(keep_idx))
        escaped_total += int(np.sum(escaped))
        total += int(len(escaped))
        initial += int(np.sum(metrics[:, 0]))
        final += int(np.sum(metrics[:, 1]))
        intra_time += float(np.sum(metrics[:, 2]))
        efflux += int(round(float(np.sum(metrics[:, 3]))))
        influx += int(round(float(np.sum(metrics[:, 4]))))
        start_survival_time += float(np.sum(metrics[:, 5]))
        first_exit += int(round(float(np.sum(metrics[:, 6]))))
        start_initial += int(np.sum(metrics[:, 0]))
        occ_counts += np.sum(inside_trace, axis=0, dtype=np.int64)
        start_counts += np.sum(start_trace, axis=0, dtype=np.int64)

    if cfg.boundary_mode == "periodic" and escaped_total:
        raise RuntimeError("Periodic walker escaped, which is an implementation error.")
    if cfg.boundary_mode != "periodic" and escaped_total / max(total, 1) > cfg.max_escape_frac:
        raise RuntimeError(
            f"Legacy escape fraction {escaped_total / total:.3%} exceeds configured "
            f"limit {cfg.max_escape_frac:.3%}."
        )
    return ReducedResult(
        cos_sum=cos_total,
        sin_sum=sin_total,
        n_walkers_kept=kept,
        n_escaped=escaped_total,
        n_walkers_total=total,
        intra_time_ms=intra_time,
        efflux_events=efflux,
        influx_events=influx,
        initial_intra=initial,
        final_intra=final,
        occupancy_counts=occ_counts,
        occupancy_denominator=total,
        start_survival_time_ms=start_survival_time,
        first_exit_events=first_exit,
        start_survivor_counts=start_counts,
        start_initial_intra=start_initial,
        checkpoint_h_ms=cfg.h_ms,
        pp_values=[float(pp)],
        analytic_kio_eq5_values=[pp_to_kio_eq5(pp, ens.mean_AV, cfg)] if not ens.is_free_water else [0.0],
        geometry_stats=[ens.geometry.to_dict()],
    )


def _measure_exchange_with_pp(
    ens: Ensemble,
    pp: float,
    cfg: SimConfig,
    seed: int,
) -> ExchangeMeasurement:
    """Tagged starting-cell first-exit measurement without signal columns."""
    t_ms = min(float(cfg.exchange_calibration_ms), float(cfg.T_max_ms))
    # Calibration is intentionally independent of the signal-walk batch
    # size.  A small smoke-test signal run must not silently downgrade the
    # precision of the geometry's authoritative p_p response curve.
    n_walkers = int(cfg.exchange_calibration_walkers)
    probe_cfg = replace(cfg, T_max_ms=t_ms, n_walkers=n_walkers, walker_chunk=None)
    _, _, metrics, _, start_trace, device = _run_walk_one_ensemble(ens, pp, probe_cfg, seed, verbose=False)
    # The calibration runner returns telemetry on host for both paths.
    _ = device
    survival_time = float(np.sum(metrics[:, 5]))
    first_exit = int(round(float(np.sum(metrics[:, 6]))))
    initial = int(np.sum(metrics[:, 0]))
    return ExchangeMeasurement(
        start_survival_time_ms=survival_time,
        first_exit_events=first_exit,
        start_survivor_counts=tuple(int(x) for x in np.sum(start_trace, axis=0)),
        start_initial_intra=initial,
        checkpoint_h_ms=float(probe_cfg.h_ms),
    )


def _combine_exchange_measurements(
    measurements: Sequence[ExchangeMeasurement],
) -> ExchangeMeasurement:
    """Pool independent calibration batches without averaging rate estimates."""
    if not measurements:
        raise ValueError("cannot combine an empty exchange measurement list")
    h_ms = measurements[0].checkpoint_h_ms
    if any(x.checkpoint_h_ms != h_ms for x in measurements):
        raise ValueError("exchange calibration batches have inconsistent checkpoint grids")
    counts = np.sum(
        np.asarray([x.start_survivor_counts for x in measurements], dtype=np.int64), axis=0
    )
    return ExchangeMeasurement(
        start_survival_time_ms=float(sum(x.start_survival_time_ms for x in measurements)),
        first_exit_events=int(sum(x.first_exit_events for x in measurements)),
        start_survivor_counts=tuple(int(x) for x in counts),
        start_initial_intra=int(sum(x.start_initial_intra for x in measurements)),
        checkpoint_h_ms=float(h_ms),
    )


def _measure_exchange_response_point(
    ens: Ensemble,
    pp: float,
    cfg: SimConfig,
    seed: int,
) -> ExchangeMeasurement:
    """Accumulate enough start-cell exits for one response-curve point."""
    batches: list[ExchangeMeasurement] = []
    for batch in range(int(cfg.exchange_calibration_max_batches)):
        measurement = _measure_exchange_with_pp(
            ens, pp, cfg, seed + 1_299_709 * batch
        )
        batches.append(measurement)
        aggregate = _combine_exchange_measurements(batches)
        if aggregate.first_exit_events >= int(cfg.exchange_calibration_min_events):
            return aggregate
    aggregate = _combine_exchange_measurements(batches)
    if aggregate.first_exit_events == 0:
        raise RuntimeError(
            "Exchange response observed zero tagged first exits at "
            f"p_p={pp:.6g} after {len(batches)} batches. Increase "
            "exchange_calibration_walkers/time; refusing an unlabelled entry."
        )
    # A sparse, nonzero point can still bracket the measured curve but must be
    # conspicuous in its stored Poisson SE.  Production defaults target at
    # least 32 events; this branch is mostly for deliberately small Tier-A
    # smoke configurations.
    return aggregate


def _weighted_isotonic_non_decreasing(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted PAVA without a scikit-learn dependency.

    The first-exit hazard must be non-decreasing in permeability.  Finite
    calibration walks can violate that order by sampling noise, so response
    inversion uses this explicitly recorded isotonic estimate rather than a
    hidden global scale factor.  Infinite/zero-SE points receive the finite
    conservative weight of one; exact p=0 is handled outside this helper.
    """
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.ndim != 1 or w.shape != y.shape:
        raise ValueError("isotonic values and weights must be matching 1-D arrays")
    if len(y) == 0:
        return y.copy()
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)
    # Each item starts as an independent [lo, hi, weighted_sum, weight] block.
    blocks: list[list[float]] = []
    for i, (value, weight) in enumerate(zip(y, w)):
        blocks.append([float(i), float(i + 1), float(value * weight), float(weight)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[2] / left[3] <= right[2] / right[3]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    out = np.empty_like(y)
    for lo, hi, total, weight in blocks:
        out[int(lo):int(hi)] = total / weight
    return out


def _response_pp_values(
    ens: Ensemble,
    target_kios: Sequence[float],
    cfg: SimConfig,
) -> np.ndarray:
    """Choose a small, target-aware p_p calibration curve for one geometry.

    Eq. 5 is used only to locate a useful permeability range.  A fourfold
    bracket around that range, plus p=1, ensures the measured curve rather
    than Eq. 5 controls the actual inverse.  The p=0 origin is represented
    analytically and never wastes a Monte-Carlo probe.
    """
    positive = np.asarray([float(k) for k in target_kios if float(k) > 0.0])
    if len(positive) == 0 or ens.is_free_water:
        return np.empty(0, dtype=float)
    eq5_pp = positive / (
        1000.0 * math.sqrt(cfg.D0 / (6.0 * cfg.ts)) * float(ens.mean_AV)
    )
    # p=0 is an exact lower bracket, so avoid wasting calibration exposure at
    # a quarter of the smallest requested rate.  Starting near 0.8× the Eq.5
    # estimate gives the default 4096×32-ms probe tens of first exits even at
    # k_io=1 s^-1; the measured curve, not this estimate, still sets p_p.
    minimum = max(float(cfg.exchange_calibration_min_pp), float(np.min(eq5_pp)) * 0.8)
    maximum = min(1.0, max(minimum, float(np.max(eq5_pp)) * 4.0))
    n = max(3, int(cfg.exchange_calibration_response_points))
    values = np.geomspace(minimum, maximum, n)
    # p=1 is an explicit physical endpoint.  It lets pp_for_rate reject an
    # unreachable target rather than silently clipping an analytic inverse.
    values = np.unique(np.r_[values, 1.0])
    return values.astype(float)


def measure_exchange_response(
    ens: Ensemble,
    cfg: SimConfig,
    seed: int,
    target_kios: Sequence[float],
) -> ExchangeResponse:
    """Measure and validate the full direct p_p→first-exit response curve.

    This is deliberately performed once per realised geometry, before the
    k_io sweep.  For every positive probe p_p, walkers are tagged with their
    start cell; the event-counting hazard and the decay of the still-in-start-
    cell population are both retained.  No entry label is inferred from Eq. 5.
    """
    pps = _response_pp_values(ens, target_kios, cfg)
    if len(pps) == 0:
        return ExchangeResponse((0.0,), (0.0,), (0.0,), (0.0,), (0.0,))

    event_rates: list[float] = [0.0]
    event_ses: list[float] = [0.0]
    fit_rates: list[float] = [0.0]
    all_pp: list[float] = [0.0]
    for i, pp in enumerate(pps):
        measurement = _measure_exchange_response_point(
            ens, float(pp), cfg, seed + 15_485_863 * (i + 1)
        )
        all_pp.append(float(pp))
        event_rates.append(float(measurement.event_rate_s_inv))
        event_ses.append(float(measurement.event_rate_se_s_inv))
        fit_rates.append(float(measurement.survival_fit_s_inv))

    raw = np.asarray(event_rates, dtype=float)
    # First point is exactly p=0.  Event-counting precision supplies PAVA
    # weights for positive probes; a large origin weight preserves rate(0)=0.
    positive_se = np.asarray(event_ses[1:], dtype=float)
    weights = np.r_[1.0e30, 1.0 / np.maximum(positive_se, 1e-12) ** 2]
    monotone = _weighted_isotonic_non_decreasing(raw, weights)
    monotone[0] = 0.0
    return ExchangeResponse(
        tuple(float(x) for x in all_pp),
        tuple(float(x) for x in event_rates),
        tuple(float(x) for x in event_ses),
        tuple(float(x) for x in fit_rates),
        tuple(float(x) for x in monotone),
    )


def calibrate_pp_for_target_kio(
    ens: Ensemble,
    target_kio: float,
    cfg: SimConfig,
    seed: int,
    response: Optional[ExchangeResponse] = None,
    target_kios: Optional[Sequence[float]] = None,
) -> ExchangeCalibration:
    """Select p_p by inverting the direct per-geometry response curve.

    The Eq.-5 inversion is stored only as an initial comparator.  It is not
    clamped and it does not choose the walk permeability.  A full signal walk
    then remeasures the tagged-starting-cell rate that becomes the entry label.
    """
    if target_kio < 0.0:
        raise ValueError(f"target k_io must be non-negative, got {target_kio}")
    if target_kio == 0.0 or ens.is_free_water:
        return ExchangeCalibration(target_kio, 0.0, 0.0, response)
    # Do not call _checked_pp here: an Eq.-5 value above one is useful
    # evidence in metadata.  Only a measured response may establish whether
    # the requested rate is physically reachable with p_p <= 1.
    pp_initial = kio_to_pp(target_kio, ens.mean_AV, cfg)
    if response is None:
        response = measure_exchange_response(
            ens, cfg, seed,
            target_kios if target_kios is not None else [target_kio],
        )
    pp_final = response.pp_for_rate(float(target_kio))
    return ExchangeCalibration(
        target_kio=float(target_kio),
        pp_initial_eq5=float(pp_initial),
        pp_used=pp_final,
        response=response,
    )


def _merge_results(results: Sequence[ReducedResult]) -> ReducedResult:
    if not results:
        raise ValueError("cannot merge no results")
    shape = results[0].cos_sum.shape
    occ_shape = results[0].occupancy_counts.shape
    merged = ReducedResult(
        cos_sum=np.zeros(shape, dtype=np.float64),
        sin_sum=np.zeros(shape, dtype=np.float64),
        n_walkers_kept=0, n_escaped=0, n_walkers_total=0,
        intra_time_ms=0.0, efflux_events=0, influx_events=0,
        initial_intra=0, final_intra=0,
        occupancy_counts=np.zeros(occ_shape, dtype=np.int64), occupancy_denominator=0,
        start_survival_time_ms=0.0, first_exit_events=0,
        start_survivor_counts=np.zeros(occ_shape, dtype=np.int64),
        start_initial_intra=0, checkpoint_h_ms=results[0].checkpoint_h_ms,
    )
    for r in results:
        merged.cos_sum += r.cos_sum
        merged.sin_sum += r.sin_sum
        merged.n_walkers_kept += r.n_walkers_kept
        merged.n_escaped += r.n_escaped
        merged.n_walkers_total += r.n_walkers_total
        merged.intra_time_ms += r.intra_time_ms
        merged.efflux_events += r.efflux_events
        merged.influx_events += r.influx_events
        merged.initial_intra += r.initial_intra
        merged.final_intra += r.final_intra
        merged.occupancy_counts += r.occupancy_counts
        merged.occupancy_denominator += r.occupancy_denominator
        merged.start_survival_time_ms += r.start_survival_time_ms
        merged.first_exit_events += r.first_exit_events
        merged.start_survivor_counts += r.start_survivor_counts
        merged.start_initial_intra += r.start_initial_intra
        merged.pp_values.extend(r.pp_values)
        merged.analytic_kio_eq5_values.extend(r.analytic_kio_eq5_values)
        merged.calibration.extend(r.calibration)
        merged.geometry_stats.extend(r.geometry_stats)
    return merged


def run_walk_Y(
    ens: Ensemble,
    kio: float,
    cfg: Optional[SimConfig] = None,
    seed: int = 0,
    verbose: bool = True,
    *,
    pp: Optional[float] = None,
    return_telemetry: bool = False,
    classifier: str = "cache",
    random_stream: Optional[WalkRandomStream] = None,
    use_gpu: Optional[bool] = None,
):
    """Validation API: execute one walk and return per-walker Y on host.

    ``pp`` is preferred for physics validation.  Supplying only ``kio`` uses
    the Eq.-5 comparator as a legacy convenience and exposes that choice in
    the returned telemetry when requested.
    """
    if cfg is None:
        cfg = SimConfig()
    actual_pp = _checked_pp(pp if pp is not None else kio_to_pp(kio, ens.mean_AV, cfg))
    Y, escaped, metrics, inside_trace, start_trace, device = _run_walk_one_ensemble(
        ens, actual_pp, cfg, seed, verbose, classifier=classifier,
        random_stream=random_stream, use_gpu=use_gpu,
    )
    Y_host = Y.copy_to_host() if device else Y
    keep = escaped == 0
    if cfg.boundary_mode == "periodic" and not np.all(keep):
        raise RuntimeError("Periodic walk reported an escape")
    if return_telemetry:
        return Y_host[keep], int(np.sum(escaped)), {
            "metrics": metrics,
            "occupancy_counts": np.sum(inside_trace, axis=0),
            "occupancy_fraction": np.mean(inside_trace, axis=0),
            "start_survivor_fraction": np.mean(start_trace, axis=0),
            "pp": actual_pp,
        }
    return Y_host[keep], int(np.sum(escaped))


def _build_ensembles_for_entry(
    rho: float,
    V: float,
    cfg: SimConfig,
    seed: int,
    verbose: bool,
) -> list[Ensemble]:
    if rho <= 0.0 or V <= 0.0:
        return [create_dummy_ensemble(cfg) for _ in range(int(cfg.n_ensembles))]
    return [
        create_ensemble(rho, V, cfg, seed=_ensemble_seed(seed, ei), verbose=verbose)
        for ei in range(int(cfg.n_ensembles))
    ]


def run_simulation_reduced(
    rho: float,
    V: float,
    kio: float,
    j_delta: np.ndarray,
    j_Delta: np.ndarray,
    j_sum: np.ndarray,
    phase_coef: np.ndarray,
    cfg: Optional[SimConfig] = None,
    seed: int = 0,
    verbose: bool = True,
    *,
    calibrate_exchange: bool = True,
) -> ReducedResult:
    """Simulate one nominal (rho,V,kio) point and directly measure k_io."""
    outputs = run_simulation_multi_kio_reduced(
        rho, V, [kio], j_delta, j_Delta, j_sum, phase_coef,
        cfg=cfg, seed=seed, verbose=verbose, calibrate_exchange=calibrate_exchange,
    )
    return outputs[float(kio)]


def run_simulation_multi_kio_reduced(
    rho: float,
    V: float,
    kios: Sequence[float],
    j_delta: np.ndarray,
    j_Delta: np.ndarray,
    j_sum: np.ndarray,
    phase_coef: np.ndarray,
    cfg: Optional[SimConfig] = None,
    seed: int = 0,
    verbose: bool = True,
    *,
    calibrate_exchange: bool = True,
) -> dict[float, ReducedResult]:
    """Reuse actual geometry across a nominal k_io sweep.

    The returned dictionary keys are nominal grid coordinates.  Each result
    contains the independently measured rate that becomes the stored label.
    """
    if cfg is None:
        cfg = SimConfig()
    if cfg.boundary_mode != "periodic":
        raise ValueError("library production requires periodic boundary_mode")
    cfg.assert_grid_alignment()
    kios = [float(k) for k in kios]
    if any(k < 0.0 for k in kios):
        raise ValueError("k_io grid values must be non-negative")
    total_walks = 3 * int(cfg.n_ensembles) * int(cfg.n_walkers)
    if total_walks < PAPER_WALKS_PER_ENTRY // 10:
        sys.stderr.write(
            f"    WARNING: only {total_walks:,} axis-walks per entry; "
            f"paper-scale reference was ~{PAPER_WALKS_PER_ENTRY:,}.\n"
        )
    ensembles = _build_ensembles_for_entry(rho, V, cfg, seed, verbose)
    per_kio: dict[float, list[ReducedResult]] = {k: [] for k in kios}
    for ei, ens in enumerate(ensembles):
        walk_seed = _walk_seed(seed, ei)
        # One direct tagged-starting-cell response curve calibrates this
        # realised geometry for the entire k_io sweep.  It deliberately does
        # not assume a global linear/Eq.-5 conversion factor.
        response = (
            None if ens.is_free_water or not calibrate_exchange
            else measure_exchange_response(ens, cfg, walk_seed + 13, kios)
        )
        for kio in kios:
            if ens.is_free_water:
                calibration = ExchangeCalibration(kio, 0.0, 0.0, response)
            elif calibrate_exchange:
                calibration = calibrate_pp_for_target_kio(
                    ens, kio, cfg, walk_seed + 13,
                    response=response, target_kios=kios,
                )
            else:
                p = _checked_pp(kio_to_pp(kio, ens.mean_AV, cfg)) if kio > 0.0 else 0.0
                calibration = ExchangeCalibration(kio, p, p, None, "analytic_eq5_diagnostic")
            result = _walk_and_reduce_one_ensemble(
                ens, calibration.pp_used, cfg,
                j_delta, j_Delta, j_sum, phase_coef,
                walk_seed, verbose=False,
            )
            result.calibration.append(calibration)
            per_kio[kio].append(result)
            if verbose:
                print(
                    f"    ensemble {ei + 1}/{len(ensembles)}, nominal k_io={kio:g}: "
                    f"p={calibration.pp_used:.6g}, measured={result.measured_kio:.4g} s^-1"
                )
    return {k: _merge_results(v) for k, v in per_kio.items()}
