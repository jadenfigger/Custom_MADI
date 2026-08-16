#!/usr/bin/env python3
"""Measure the SI v5 runtime by geometry, walk, reduction, and walk phase.

This is an analysis-only benchmark.  It builds no library file and does not
change a simulation setting.  The four named cases are intentionally fixed to
the canonical production grid:

``center-c1``
    40 production ensembles at the stencil-probe centre and one k_io value.
    Together with ``center-c3`` this gives C1 = G + W.
``center-c3``
    The same centre and 40 ensembles at k_io = 19, 20, 21 s^-1.  This gives
    C3 = G + 3W on the *same GPU class*, avoiding a MIG/full-A100 comparison.
``rho-low`` / ``rho-high``
    One full-production-walker ensemble at the lowest/highest canonical rho
    node, each at the admissible V node closest to the mask-band centre.
    These are deliberately one-ensemble endpoint probes: their purpose is to
    measure density dependence of geometry and of a representative exact
    walk, not to spend 40 ensembles before the cost model is known.

The optional phase replay is not a replacement classifier.  It repeats the
current ``_walk_kernel`` transition exactly, adding device ``clock64`` reads
around its existing operations.  A direct nearest-seed query is timed before
the unchanged full classifier, so the full-classifier time minus that direct
nearest-query time is a controlled estimate of radius/facet traversal.  The
profiled discrete state is compared bit-for-bit with an uninstrumented replay
using the same seed.  Inserting device-clock reads can change CUDA register
allocation and therefore the floating-point Y accumulation, so its difference
is recorded together with a conservative bound on the resulting signal effect;
it is not a production-output equivalence gate.  Cycle fractions locate work;
synchronized wall times of the unmodified walk and reduction kernels provide
the absolute timings.

Run this only in a Sol GPU batch job; it explicitly refuses a non-CUDA host.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np

from madi.config import SimConfig
from madi.library import make_remediation_log_grid
from madi.signal import build_columns
from madi.walker_gpu import (
    HAS_CUDA,
    _build_ensembles_for_entry,
    _checked_pp,
    _reduce_kernel,
    _walk_and_reduce_one_ensemble,
    _walk_kernel,
    _walk_seed,
    _nearest_one_kdtree,
    _classify_full_facet,
    kio_to_pp,
)

try:
    from numba import cuda, types
    from numba.cuda.extending import intrinsic
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_normal_float64,
        xoroshiro128p_uniform_float64,
    )
    from llvmlite import ir
except ImportError:  # pragma: no cover - exercised only on a non-Sol host
    cuda = None
    types = None
    intrinsic = None
    ir = None


BUILD_SEED = 20260803
PRODUCTION_WALKERS = 100_000
PRODUCTION_ENSEMBLES = 40
PRODUCTION_COLUMNS = 31_125
PROFILE_PHASES = (
    "proposed_position_rng_and_boundary",
    "nearest_kdtree_direct",
    "full_classifier_total",
    "transition_and_accept_reject",
    "Y_accumulation",
    "Y_snapshot_write",
)
# Match the existing committed CPU/GPU golden harness only as a reported
# reference.  A distinct clock-instrumented CUDA kernel is not expected to
# meet it bit-for-bit; no production result uses this profile kernel.
PROFILE_Y_RTOL = 5e-11
PROFILE_Y_ATOL = 5e-12


def _canonical_coordinate(which: str) -> dict[str, float | int]:
    """Resolve one benchmark coordinate from the code-defined production grid."""
    grid = make_remediation_log_grid()
    rho_index = {"low": 0, "center": 21, "high": len(grid.rhos) - 1}[which]
    target_vi = math.sqrt(float(grid.vi_min) * float(grid.vi_max))
    candidates: list[tuple[int, float, float]] = []
    rho = float(grid.rhos[rho_index])
    for V_index, V in enumerate(grid.Vs):
        vi = rho * float(V) * 1.0e-6
        if float(grid.vi_min) <= vi <= float(grid.vi_max):
            candidates.append((V_index, float(V), vi))
    if not candidates:
        raise RuntimeError(f"canonical production rho index {rho_index} has no mask-admissible V")
    V_index, V, vi = min(candidates, key=lambda item: abs(math.log(item[2] / target_vi)))
    return {
        "rho_index": int(rho_index),
        "V_index": int(V_index),
        "rho_per_uL": rho,
        "V_pL": V,
        "vi": vi,
        "target_vi_geometric_center": target_vi,
    }


CASE_SPECS = {
    "center-c1": {"coordinate": "center", "n_ensembles": 40, "kios": (20.0,), "profile": True},
    "center-c3": {"coordinate": "center", "n_ensembles": 40, "kios": (19.0, 20.0, 21.0), "profile": False},
    "rho-low": {"coordinate": "low", "n_ensembles": 1, "kios": (20.0,), "profile": True},
    "rho-high": {"coordinate": "high", "n_ensembles": 1, "kios": (20.0,), "profile": True},
}


if cuda is not None:

    @intrinsic
    def _clock64(typingctx):
        """PTX ``%clock64`` intrinsic for diagnostic cycle accounting only."""
        signature = types.uint64()

        def codegen(context, builder, signature, args):
            function_type = ir.FunctionType(ir.IntType(64), [])
            assembly = ir.InlineAsm(
                function_type,
                "mov.u64 $0, %clock64;",
                "=l",
                side_effect=True,
            )
            return builder.call(assembly, [])

        return signature, codegen


    @cuda.jit
    def _walk_profile_kernel(
        seeds, annulus, node_seed, node_axis, node_left, node_right, node_parent,
        L, source_lo, source_hi, sigma, ts, n_steps, pp, has_cells,
        half_L, steps_per_h, rng_states,
        Y_out, inside_trace, escaped_out, classifier_error_out,
        phase_cycles_out, active_steps_out, classifier_calls_out, snapshot_writes_out,
    ):
        """Instrumented, exact replay of ``_walk_kernel``.

        The direct nearest query is extra diagnostic work only; the actual
        compartment label used for the transition still comes from the
        unchanged ``_classify_full_facet`` device function.
        """
        tid = cuda.grid(1)
        if tid >= Y_out.shape[0]:
            return

        proposal_cycles = np.uint64(0)
        nearest_cycles = np.uint64(0)
        full_classifier_cycles = np.uint64(0)
        transition_cycles = np.uint64(0)
        y_cycles = np.uint64(0)
        snapshot_cycles = np.uint64(0)
        active_steps = 0
        classifier_calls = 0
        snapshot_writes = 0

        t0 = _clock64()
        px = source_lo + xoroshiro128p_uniform_float64(rng_states, tid) * (source_hi - source_lo)
        py = source_lo + xoroshiro128p_uniform_float64(rng_states, tid) * (source_hi - source_lo)
        pz = source_lo + xoroshiro128p_uniform_float64(rng_states, tid) * (source_hi - source_lo)
        proposal_cycles += _clock64() - t0

        if has_cells == 1:
            t0 = _clock64()
            _nearest_one_kdtree(
                px, py, pz, seeds, node_seed, node_axis, node_left, node_right, node_parent
            )
            nearest_cycles += _clock64() - t0
            t0 = _clock64()
            cur_cell, cur_inside, ok = _classify_full_facet(
                px, py, pz, seeds, annulus, node_seed, node_axis, node_left, node_right, node_parent
            )
            full_classifier_cycles += _clock64() - t0
            classifier_calls += 1
            if ok == 0:
                classifier_error_out[tid] = 1
                escaped_out[tid] = 1
                return
        else:
            cur_cell = 0
            cur_inside = False

        xprev = px - half_L
        yprev = py - half_L
        zprev = pz - half_L
        Yx = 0.0
        Yy = 0.0
        Yz = 0.0
        Y_out[tid, 0, 0] = 0.0
        Y_out[tid, 0, 1] = 0.0
        Y_out[tid, 0, 2] = 0.0
        inside_trace[tid, 0] = 1 if cur_inside else 0
        escaped = 0

        for step in range(n_steps):
            if escaped == 1:
                continue
            active_steps += 1
            t0 = _clock64()
            dx = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            dy = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            dz = xoroshiro128p_normal_float64(rng_states, tid) * sigma
            # Match the production stream contract: one uniform every step,
            # even where the transition branch does not use it.
            u = xoroshiro128p_uniform_float64(rng_states, tid)
            nx = px + dx
            ny = py + dy
            nz = pz + dz
            proposal_cycles += _clock64() - t0
            if nx < 0.0 or nx >= L or ny < 0.0 or ny >= L or nz < 0.0 or nz >= L:
                escaped = 1
                continue

            if has_cells == 1:
                t0 = _clock64()
                _nearest_one_kdtree(
                    nx, ny, nz, seeds, node_seed, node_axis, node_left, node_right, node_parent
                )
                nearest_cycles += _clock64() - t0
                t0 = _clock64()
                new_cell, new_inside, ok = _classify_full_facet(
                    nx, ny, nz, seeds, annulus, node_seed, node_axis, node_left, node_right, node_parent
                )
                full_classifier_cycles += _clock64() - t0
                classifier_calls += 1
                if ok == 0:
                    classifier_error_out[tid] = 1
                    escaped = 1
                    continue
            else:
                new_cell = 0
                new_inside = False

            t0 = _clock64()
            m = 0
            if cur_inside and not new_inside:
                m = 1
            elif (not cur_inside) and new_inside:
                m = 1
            elif cur_inside and new_inside and cur_cell != new_cell:
                m = 2
            if m > 0 and pp < 1.0:
                threshold = pp if m == 1 else pp * pp
                if u >= threshold:
                    nx = px
                    ny = py
                    nz = pz
                    new_cell = cur_cell
                    new_inside = cur_inside
            transition_cycles += _clock64() - t0

            t0 = _clock64()
            px = nx
            py = ny
            pz = nz
            cur_cell = new_cell
            cur_inside = new_inside
            xnow = px - half_L
            ynow = py - half_L
            znow = pz - half_L
            Yx += 0.5 * (xprev + xnow) * ts
            Yy += 0.5 * (yprev + ynow) * ts
            Yz += 0.5 * (zprev + znow) * ts
            xprev = xnow
            yprev = ynow
            zprev = znow
            y_cycles += _clock64() - t0

            step_idx = step + 1
            if step_idx % steps_per_h == 0:
                t0 = _clock64()
                j = step_idx // steps_per_h
                Y_out[tid, j, 0] = Yx
                Y_out[tid, j, 1] = Yy
                Y_out[tid, j, 2] = Yz
                inside_trace[tid, j] = 1 if cur_inside else 0
                snapshot_cycles += _clock64() - t0
                snapshot_writes += 1

        escaped_out[tid] = escaped
        phase_cycles_out[tid, 0] = proposal_cycles
        phase_cycles_out[tid, 1] = nearest_cycles
        phase_cycles_out[tid, 2] = full_classifier_cycles
        phase_cycles_out[tid, 3] = transition_cycles
        phase_cycles_out[tid, 4] = y_cycles
        phase_cycles_out[tid, 5] = snapshot_cycles
        active_steps_out[tid] = active_steps
        classifier_calls_out[tid] = classifier_calls
        snapshot_writes_out[tid] = snapshot_writes


def _device_name() -> str:
    device = cuda.get_current_device()
    name = device.name
    return name.decode() if isinstance(name, bytes) else str(name)


def _assert_gpu_ready() -> None:
    if not HAS_CUDA or cuda is None or not cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this benchmark in the Sol GPU batch job")


def _kio_pp(ensemble: Any, kio: float, cfg: SimConfig) -> float:
    if ensemble.is_free_water or kio == 0.0:
        return 0.0
    return _checked_pp(kio_to_pp(float(kio), float(ensemble.mean_AV), cfg))


def _warm_cuda(ensemble: Any, pp: float, cfg: SimConfig, columns: Any, seed: int) -> float:
    """Compile the exact kernels before timing C1/C3; not production work."""
    warm_cfg = replace(cfg, n_walkers=1, n_ensembles=1)
    start = time.perf_counter()
    _walk_and_reduce_one_ensemble(
        ensemble, pp, warm_cfg,
        columns.j_delta, columns.j_Delta, columns.j_sum, columns.phase_coef,
        seed,
    )
    return time.perf_counter() - start


def _run_group(case: str, coordinate: dict[str, float | int], cfg: SimConfig,
               kios: Iterable[float], columns: Any, seed: int) -> tuple[dict[str, Any], list[Any]]:
    """Run the production path explicitly enough to expose geometry time."""
    rho = float(coordinate["rho_per_uL"])
    volume = float(coordinate["V_pL"])
    start_geometry = time.perf_counter()
    ensembles = _build_ensembles_for_entry(rho, volume, cfg, seed=seed, verbose=False)
    geometry_seconds = time.perf_counter() - start_geometry
    if len(ensembles) != int(cfg.n_ensembles):
        raise RuntimeError("ensemble builder returned an unexpected count")

    warm_seconds = _warm_cuda(ensembles[0], _kio_pp(ensembles[0], float(tuple(kios)[0]), cfg), cfg, columns, _walk_seed(seed, 0))

    walk_reduce_seconds = 0.0
    per_ensemble_kio: list[dict[str, Any]] = []
    for ensemble_index, ensemble in enumerate(ensembles):
        walk_seed = _walk_seed(seed, ensemble_index)
        for kio in kios:
            pp = _kio_pp(ensemble, float(kio), cfg)
            start = time.perf_counter()
            result = _walk_and_reduce_one_ensemble(
                ensemble, pp, cfg,
                columns.j_delta, columns.j_Delta, columns.j_sum, columns.phase_coef,
                walk_seed,
            )
            elapsed = time.perf_counter() - start
            if result.n_escaped:
                raise RuntimeError(f"fatal SI escape during benchmark: {result.n_escaped} walkers")
            walk_reduce_seconds += elapsed
            per_ensemble_kio.append({
                "ensemble_index": int(ensemble_index),
                "kio_s_inv": float(kio),
                "pp": float(pp),
                "elapsed_seconds": float(elapsed),
                "n_escaped": int(result.n_escaped),
            })

    group = {
        "case": case,
        "geometry_seconds": float(geometry_seconds),
        "cuda_jit_warmup_seconds_excluded_from_cost": float(warm_seconds),
        "walk_and_reduction_seconds": float(walk_reduce_seconds),
        "C_seconds_excluding_jit": float(geometry_seconds + walk_reduce_seconds),
        "per_ensemble_kio": per_ensemble_kio,
        "ensemble_geometry": [
            {
                "n_seeds_pop": int(ensemble.geometry.n_seeds_pop),
                "n_seeds_sim": int(ensemble.geometry.n_seeds_sim),
                "sim_side_um": float(ensemble.geometry.sim_side_um),
                "target_vi": float(ensemble.geometry.target_vi),
                "realised_vi": float(ensemble.geometry.realised_vi),
                "governing_mean_A_over_V_um_inv": float(ensemble.mean_AV),
            }
            for ensemble in ensembles
        ],
    }
    return group, ensembles


def _upload_walk_inputs(ensemble: Any, cfg: SimConfig, n_walkers: int) -> dict[str, Any]:
    """Upload the immutable exact-geometry inputs once for a diagnostic replay."""
    return {
        "seeds": cuda.to_device(ensemble.seeds),
        "annulus": cuda.to_device(ensemble.annulus),
        "node_seed": cuda.to_device(ensemble.kd_node_seed),
        "node_axis": cuda.to_device(ensemble.kd_node_axis),
        "node_left": cuda.to_device(ensemble.kd_node_left),
        "node_right": cuda.to_device(ensemble.kd_node_right),
        "node_parent": cuda.to_device(ensemble.kd_node_parent),
        # These three are ignored when use_golden=0 but retain the exact
        # kernel signature used by production.
        "initial": cuda.to_device(np.zeros((1, 3), dtype=np.float64)),
        "increments": cuda.to_device(np.zeros((1, 1, 3), dtype=np.float64)),
        "uniforms": cuda.to_device(np.zeros((1, 1), dtype=np.float64)),
        # The reference profiler keeps the cache disabled but must provide
        # valid dummy buffers for the unified production-kernel signature.
        "cache_ref": cuda.device_array((1, 3), dtype=np.float64),
        "cache_cell": cuda.device_array(1, dtype=np.int32),
        "cache_inside": cuda.device_array(1, dtype=np.int8),
        "cache_rsafe": cuda.device_array(1, dtype=np.float64),
        "cache_count": cuda.device_array(1, dtype=np.int32),
        "cache_valid": cuda.device_array(1, dtype=np.int8),
        "cache_ids": cuda.device_array((1, 2), dtype=np.int32),
        "cache_stats": cuda.to_device(np.zeros((1, 9), dtype=np.int64)),
        "n_walkers": int(n_walkers),
        "cfg": cfg,
        "ensemble": ensemble,
    }


def _allocate_walk_outputs(n_walkers: int, cfg: SimConfig) -> tuple[Any, Any, Any, Any]:
    return (
        cuda.device_array((n_walkers, cfg.n_grid, 3), dtype=np.float64),
        cuda.device_array((n_walkers, cfg.n_grid), dtype=np.int8),
        cuda.device_array(n_walkers, dtype=np.int8),
        cuda.to_device(np.zeros(n_walkers, dtype=np.int8)),
    )


def _launch_reference_walk(inputs: dict[str, Any], pp: float, seed: int) -> tuple[Any, Any, Any, Any, float]:
    """Time the unmodified production walk kernel with a CUDA synchronize."""
    ensemble = inputs["ensemble"]
    cfg = inputs["cfg"]
    n_walkers = int(inputs["n_walkers"])
    Y, inside, escaped, classifier_error = _allocate_walk_outputs(n_walkers, cfg)
    states = create_xoroshiro128p_states(n_walkers, seed=seed)
    threads = 128
    blocks = (n_walkers + threads - 1) // threads
    start = time.perf_counter()
    _walk_kernel[blocks, threads](
        inputs["seeds"], inputs["annulus"], inputs["node_seed"], inputs["node_axis"],
        inputs["node_left"], inputs["node_right"], inputs["node_parent"],
        np.float64(ensemble.L), np.float64(ensemble.source_lo), np.float64(ensemble.source_hi),
        np.float64(cfg.sigma), np.float64(cfg.ts), np.int32(cfg.n_steps), np.float64(pp),
        np.int32(0 if ensemble.is_free_water else 1), np.float64(ensemble.L / 2.0),
        np.int32(cfg.steps_per_h), states, np.int32(0),
        inputs["initial"], inputs["increments"], inputs["uniforms"],
        np.int32(0), np.float64(np.max(ensemble.annulus)), np.float64(1.0), np.float64(0.0),
        inputs["cache_ref"], inputs["cache_cell"], inputs["cache_inside"],
        inputs["cache_rsafe"], inputs["cache_count"], inputs["cache_valid"],
        inputs["cache_ids"], inputs["cache_stats"],
        Y, inside, escaped, classifier_error,
    )
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    return Y, inside, escaped, classifier_error, elapsed


def _time_reduction(Y: Any, columns: Any) -> float:
    n_cols = len(columns.phase_coef)
    d_jd = cuda.to_device(np.asarray(columns.j_delta, dtype=np.int32))
    d_jD = cuda.to_device(np.asarray(columns.j_Delta, dtype=np.int32))
    d_js = cuda.to_device(np.asarray(columns.j_sum, dtype=np.int32))
    d_pc = cuda.to_device(np.asarray(columns.phase_coef, dtype=np.float64))
    d_cos = cuda.device_array(n_cols, dtype=np.float64)
    d_sin = cuda.device_array(n_cols, dtype=np.float64)
    start = time.perf_counter()
    _reduce_kernel[n_cols, 256](Y, d_jd, d_jD, d_js, d_pc, d_cos, d_sin)
    cuda.synchronize()
    return time.perf_counter() - start


def _phase_summary(cycles: np.ndarray, active_steps: np.ndarray,
                   classifier_calls: np.ndarray, snapshot_writes: np.ndarray) -> dict[str, Any]:
    active = np.asarray(active_steps, dtype=np.int64)
    classify = np.asarray(classifier_calls, dtype=np.int64)
    snapshots = np.asarray(snapshot_writes, dtype=np.int64)
    if np.any(active <= 0):
        raise RuntimeError("profile replay recorded an inactive walker; fatal escapes invalidate this benchmark")
    if np.any(classify <= 0):
        raise RuntimeError("profile replay recorded no classifier calls")
    c = np.asarray(cycles, dtype=np.float64)
    per_step = {
        PROFILE_PHASES[index]: c[:, index] / active
        for index in range(len(PROFILE_PHASES))
    }
    # The classifier total includes its own nearest query.  The separate
    # direct nearest measurement lets us expose the radius/facet remainder
    # without replacing or approximating the production classifier.
    radius = (c[:, 2] - c[:, 1]) / classify
    direct_nearest = c[:, 1] / classify
    full_classifier = c[:, 2] / classify

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "mean": float(np.mean(values)),
        }

    out = {
        "clock_unit": "SM clock cycles; use phase fractions, not absolute wall time",
        "phase_cycles_per_active_step": {name: stats(values) for name, values in per_step.items()},
        "nearest_cycles_per_classifier_call": stats(direct_nearest),
        "full_classifier_cycles_per_classifier_call": stats(full_classifier),
        "radius_facet_remainder_cycles_per_classifier_call": stats(radius),
        "radius_facet_remainder_negative_fraction": float(np.mean(radius < 0.0)),
        "active_steps_per_walker": stats(active.astype(np.float64)),
        "classifier_calls_per_walker": stats(classify.astype(np.float64)),
        "snapshot_writes_per_walker": stats(snapshots.astype(np.float64)),
    }
    phase_medians = {
        name: out["phase_cycles_per_active_step"][name]["median"]
        for name in PROFILE_PHASES
    }
    # Split the full classifier into its measured direct-nearest component and
    # the residual.  This is intentionally a diagnostic fraction, because
    # the two calls have slightly different cache states.
    phase_medians["radius_facet_remainder"] = out[
        "radius_facet_remainder_cycles_per_classifier_call"
    ]["median"]
    classifier_total_per_step = (
        out["full_classifier_cycles_per_classifier_call"]["median"]
        * out["classifier_calls_per_walker"]["median"]
        / out["active_steps_per_walker"]["median"]
    )
    direct_nearest_per_step = (
        out["nearest_cycles_per_classifier_call"]["median"]
        * out["classifier_calls_per_walker"]["median"]
        / out["active_steps_per_walker"]["median"]
    )
    radius_per_step = classifier_total_per_step - direct_nearest_per_step
    total_estimate = (
        phase_medians["proposed_position_rng_and_boundary"]
        + classifier_total_per_step
        + phase_medians["transition_and_accept_reject"]
        + phase_medians["Y_accumulation"]
        + phase_medians["Y_snapshot_write"]
    )
    out["median_cycle_fraction_estimate"] = {
        "proposed_position_rng_and_boundary": float(
            phase_medians["proposed_position_rng_and_boundary"] / total_estimate
        ),
        "nearest_kdtree_within_full_classifier": float(direct_nearest_per_step / total_estimate),
        "radius_facet_remainder_within_full_classifier": float(radius_per_step / total_estimate),
        "transition_and_accept_reject": float(phase_medians["transition_and_accept_reject"] / total_estimate),
        "Y_accumulation": float(phase_medians["Y_accumulation"] / total_estimate),
        "Y_snapshot_write": float(phase_medians["Y_snapshot_write"] / total_estimate),
    }
    return out


def _floating_replay_comparison(reference: np.ndarray, profiled: np.ndarray) -> dict[str, float | bool]:
    """Quantify clock-instrumentation drift without treating it as physics.

    The profile kernel has the same random-stream consumption and uses the
    unchanged full classifier for every label.  Its clock reads nevertheless
    change the CUDA compiler's register allocation, which can alter
    floating-point accumulation in Y throughout a long walk.  The caller
    separately enforces bit-identical discrete state and reports a
    signal-space bound below; this function deliberately does not pronounce a
    pass/fail verdict based on a tolerance chosen for a different kernel.
    """
    reference = np.asarray(reference, dtype=np.float64)
    profiled = np.asarray(profiled, dtype=np.float64)
    delta = np.abs(profiled - reference)
    denominator = np.maximum(np.abs(reference), PROFILE_Y_ATOL)
    return {
        "bitwise_equal": bool(np.array_equal(reference, profiled)),
        "max_abs": float(np.max(delta)) if delta.size else 0.0,
        "max_rel": float(np.max(delta / denominator)) if delta.size else 0.0,
        "rtol": PROFILE_Y_RTOL,
        "atol": PROFILE_Y_ATOL,
        "allclose": bool(np.allclose(profiled, reference, rtol=PROFILE_Y_RTOL, atol=PROFILE_Y_ATOL)),
    }


def _signal_effect_bound(y_comparison: dict[str, float | bool], columns: Any) -> dict[str, float]:
    """Bound the signal change implied by a maximum stored-Y difference.

    ``dM = Y(delta) + Y(Delta) - Y(Delta + delta)`` has at most three stored
    Y errors.  ``cos`` and ``sin`` are one-Lipschitz, so the same phase-error
    bound applies to every per-walker signal contribution and its mean.  This
    is conservative and needs no assumptions about cancellation.
    """
    max_y = float(y_comparison["max_abs"])
    max_dM = 3.0 * max_y
    max_phase_coefficient = float(np.max(np.abs(np.asarray(columns.phase_coef, dtype=np.float64))))
    max_phase = max_dM * max_phase_coefficient
    return {
        "max_abs_stored_Y_difference": max_y,
        "max_abs_dM_difference_bound": max_dM,
        "max_abs_phase_coefficient": max_phase_coefficient,
        "max_abs_phase_difference_bound": max_phase,
        "max_abs_real_or_imaginary_signal_difference_bound": max_phase,
    }


def _run_phase_profile(ensemble: Any, pp: float, cfg: SimConfig, columns: Any,
                       seed: int, n_walkers: int) -> dict[str, Any]:
    """Compare a full exact replay with its phase-instrumented counterpart."""
    profile_cfg = replace(cfg, n_walkers=int(n_walkers), n_ensembles=1)
    inputs = _upload_walk_inputs(ensemble, profile_cfg, int(n_walkers))
    reference_Y, reference_inside, reference_escaped, reference_error, walk_seconds = _launch_reference_walk(
        inputs, pp, seed
    )
    reference_error_host = reference_error.copy_to_host()
    reference_escaped_host = reference_escaped.copy_to_host()
    if np.any(reference_error_host) or np.any(reference_escaped_host):
        raise RuntimeError("unmodified reference replay encountered a classifier error or fatal escape")
    reduction_seconds = _time_reduction(reference_Y, columns)

    n = int(n_walkers)
    profile_Y, profile_inside, profile_escaped, profile_error = _allocate_walk_outputs(n, profile_cfg)
    phase_cycles = cuda.device_array((n, len(PROFILE_PHASES)), dtype=np.uint64)
    active_steps = cuda.device_array(n, dtype=np.int32)
    classifier_calls = cuda.device_array(n, dtype=np.int32)
    snapshot_writes = cuda.device_array(n, dtype=np.int32)
    states = create_xoroshiro128p_states(n, seed=seed)
    threads = 128
    blocks = (n + threads - 1) // threads
    start = time.perf_counter()
    _walk_profile_kernel[blocks, threads](
        inputs["seeds"], inputs["annulus"], inputs["node_seed"], inputs["node_axis"],
        inputs["node_left"], inputs["node_right"], inputs["node_parent"],
        np.float64(ensemble.L), np.float64(ensemble.source_lo), np.float64(ensemble.source_hi),
        np.float64(profile_cfg.sigma), np.float64(profile_cfg.ts), np.int32(profile_cfg.n_steps), np.float64(pp),
        np.int32(0 if ensemble.is_free_water else 1), np.float64(ensemble.L / 2.0),
        np.int32(profile_cfg.steps_per_h), states,
        profile_Y, profile_inside, profile_escaped, profile_error,
        phase_cycles, active_steps, classifier_calls, snapshot_writes,
    )
    cuda.synchronize()
    profile_seconds = time.perf_counter() - start

    # Clock instructions are diagnostic work and can perturb floating-point
    # register allocation.  Discrete compartment/boundary state must stay
    # bitwise identical.  Y drift is quantified and converted to a rigorous
    # signal-space upper bound, but cannot be a strict equivalence condition
    # for a different CUDA kernel.  The future fast-path validation remains
    # subject to its separate bit-identical requirement.
    y_comparison = _floating_replay_comparison(
        reference_Y.copy_to_host(), profile_Y.copy_to_host()
    )
    equality = {
        "Y_integral_diagnostic": y_comparison,
        "Y_signal_effect_bound": _signal_effect_bound(y_comparison, columns),
        "inside_trace_bitwise_equal": bool(
            np.array_equal(reference_inside.copy_to_host(), profile_inside.copy_to_host())
        ),
        "escaped_bitwise_equal": bool(
            np.array_equal(reference_escaped_host, profile_escaped.copy_to_host())
        ),
        "classifier_error_bitwise_equal": bool(
            np.array_equal(reference_error_host, profile_error.copy_to_host())
        ),
    }
    equality["pass"] = bool(
        equality["inside_trace_bitwise_equal"]
        and equality["escaped_bitwise_equal"]
        and equality["classifier_error_bitwise_equal"]
    )
    if not equality["pass"]:
        raise RuntimeError(f"phase profiler failed its numerical/state replay check: {equality}")

    phases = _phase_summary(
        phase_cycles.copy_to_host(), active_steps.copy_to_host(),
        classifier_calls.copy_to_host(), snapshot_writes.copy_to_host(),
    )
    return {
        "n_walkers": n,
        "kio_s_inv": 20.0,
        "pp": float(pp),
        "unmodified_walk_kernel_seconds": float(walk_seconds),
        "unmodified_reduction_kernel_seconds": float(reduction_seconds),
        "reduction_to_walk_time_ratio": float(reduction_seconds / walk_seconds) if walk_seconds else float("nan"),
        "instrumented_replay_seconds": float(profile_seconds),
        "instrumented_to_unmodified_walk_time_ratio": float(profile_seconds / walk_seconds) if walk_seconds else float("nan"),
        "profile_replay_check": equality,
        "phase_profile": phases,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASE_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--geometry-reference", type=Path,
        default=Path("data/geometry_reference_si_kappa_0p9.npz"),
    )
    parser.add_argument("--seed", type=int, default=BUILD_SEED)
    parser.add_argument("--profile-walkers", type=int, default=PRODUCTION_WALKERS)
    args = parser.parse_args(argv)

    _assert_gpu_ready()
    if args.profile_walkers != PRODUCTION_WALKERS:
        raise ValueError(
            f"profile-walkers must be the production value {PRODUCTION_WALKERS:,}; got {args.profile_walkers:,}"
        )
    if not args.geometry_reference.is_file():
        raise FileNotFoundError(f"certified geometry reference missing: {args.geometry_reference}")

    spec = CASE_SPECS[args.case]
    coordinate = _canonical_coordinate(str(spec["coordinate"]))
    cfg = SimConfig(
        n_walkers=PRODUCTION_WALKERS,
        n_ensembles=int(spec["n_ensembles"]),
        geometry_reference_path=str(args.geometry_reference),
    )
    cfg.assert_grid_alignment()
    columns = build_columns(cfg)
    if columns.n_pairs * columns.n_b != PRODUCTION_COLUMNS:
        raise RuntimeError(
            f"expected {PRODUCTION_COLUMNS} production columns, got {columns.n_pairs * columns.n_b}"
        )

    print(json.dumps({
        "case": args.case,
        "coordinate": coordinate,
        "n_ensembles": int(cfg.n_ensembles),
        "n_walkers": int(cfg.n_walkers),
        "kios_s_inv": list(spec["kios"]),
        "n_columns": int(columns.n_pairs * columns.n_b),
        "gpu": _device_name(),
    }, indent=2, sort_keys=True), flush=True)

    group, ensembles = _run_group(
        args.case, coordinate, cfg, tuple(float(k) for k in spec["kios"]), columns, int(args.seed)
    )
    phase: dict[str, Any] | None = None
    if bool(spec["profile"]):
        phase_pp = _kio_pp(ensembles[0], 20.0, cfg)
        phase = _run_phase_profile(
            ensembles[0], phase_pp, cfg, columns, _walk_seed(int(args.seed), 0), int(args.profile_walkers)
        )

    payload = {
        "schema": "madi-v5-runtime-benchmark-v1",
        "case": args.case,
        "build_seed": int(args.seed),
        "gpu": _device_name(),
        "coordinate": coordinate,
        "production_configuration": {
            "walkers_per_ensemble": int(cfg.n_walkers),
            "ensembles": int(cfg.n_ensembles),
            "kios_s_inv": [float(k) for k in spec["kios"]],
            "walk_duration_ms": float(cfg.T_max_ms),
            "ts_ms": float(cfg.ts),
            "n_steps": int(cfg.n_steps),
            "n_columns": int(columns.n_pairs * columns.n_b),
            "geometry_reference": str(args.geometry_reference.resolve()),
        },
        "group_timing": group,
        "phase_profile": phase,
        "notes": [
            "C1/C3 exclude one one-walker CUDA JIT warm-up from the reported cost.",
            "Endpoint cases use one ensemble only and must be extrapolated to 40 ensembles with that limitation stated.",
            "Phase cycles are from an exact replay of the current kernel; absolute timings are synchronized unmodified-kernel timings.",
            "The direct-nearest and full-classifier calls have different cache states, so their difference is a controlled radius/facet estimate, not a perfectly additive hardware counter.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"v5_runtime_benchmark_{args.case}.json"
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
