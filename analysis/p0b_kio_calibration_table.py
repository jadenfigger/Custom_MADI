#!/usr/bin/env python
"""Measure direct tagged-start-cell ``k_io`` versus MADI-I Eq. 5.

Tier: B (local CPU; a 3 x 3 geometry study is allowed to take minutes).

This is deliberately a *calibration study*, not a library builder.  For each
realised periodic Poisson--Voronoi geometry it uses one explicitly supplied
``p_p``, reports the event-counting first-exit hazard and independent
survival-curve fit, and retains Eq. 5 only as a comparator.  It produces the
table required by remediation P0-B without ever attaching the analytic rate
as a library label.

Example
-------
PYTHONPATH=. python analysis/p0b_kio_calibration_table.py \
  --output /tmp/madi_p0b_kio_table.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from madi.config import SimConfig
from madi.ensemble import create_ensemble
from madi.walker_gpu import pp_to_kio_eq5, run_walk_Y


def _rate(telemetry: dict) -> tuple[float, float]:
    metrics = np.asarray(telemetry["metrics"], dtype=float)
    at_risk_ms = float(np.sum(metrics[:, 5]))
    events = float(np.sum(metrics[:, 6]))
    if at_risk_ms <= 0.0:
        return 0.0, 0.0
    return (
        1000.0 * events / at_risk_ms,
        1000.0 * np.sqrt(events) / at_risk_ms if events else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True,
                    help="JSON table to write.")
    ap.add_argument("--vi", type=float, nargs="+", default=[0.40, 0.70, 0.90])
    ap.add_argument("--V", type=float, nargs="+", default=[0.20, 1.0, 5.0],
                    help="Requested mean volume values [pL].")
    ap.add_argument("--pp", type=float, default=0.005,
                    help="Common membrane-crossing probability for the study.")
    ap.add_argument("--walkers", type=int, default=4096)
    ap.add_argument("--duration-ms", type=float, default=32.0)
    ap.add_argument("--L", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=20_260_901)
    args = ap.parse_args()

    if not (0.0 < args.pp <= 1.0):
        ap.error("--pp must lie in (0, 1]")
    if any(not (0.0 < x < 1.0) for x in args.vi):
        ap.error("every --vi value must lie in (0, 1)")
    if any(x <= 0.0 for x in args.V):
        ap.error("every --V value must be positive")

    cfg = SimConfig(
        L=float(args.L),
        grid_spacing=0.75,
        classifier_candidates=8,
        geometry_calibration_points=100_000,
        geometry_validation_points=100_000,
        geometry_sample_cells=64,
        geometry_vi_tolerance=0.005,
        n_walkers=int(args.walkers),
        n_ensembles=1,
        T_max_ms=float(args.duration_ms),
        small_deltas=[1.0],
        big_deltas=[2.0, min(8.0, float(args.duration_ms) - 1.0)],
        b_values=[0.0],
    )
    rows: list[dict] = []
    for i, target_vi in enumerate(args.vi):
        for j, target_V in enumerate(args.V):
            rho = float(target_vi) * 1e6 / float(target_V)
            ensemble_seed = int(args.seed + i * 10_007 + j * 1_009)
            ens = create_ensemble(rho, float(target_V), cfg, seed=ensemble_seed)
            _, escaped, telemetry = run_walk_Y(
                ens, 0.0, cfg, pp=float(args.pp),
                seed=ensemble_seed + 1, verbose=False, return_telemetry=True,
            )
            event_rate, event_se = _rate(telemetry)
            fractions = np.asarray(telemetry["start_survivor_fraction"], dtype=float)
            counts = fractions * cfg.n_walkers
            start_initial = int(round(counts[0]))
            time_ms = np.arange(len(counts), dtype=float) * cfg.h_ms
            keep = (counts > 0.10 * max(start_initial, 1)) & (counts > 0.0)
            survival_fit = (
                max(0.0, -1000.0 * np.polyfit(
                    time_ms[keep], np.log(counts[keep] / max(counts[0], 1e-30)), 1,
                )[0])
                if np.count_nonzero(keep) >= 2 else 0.0
            )
            analytic = pp_to_kio_eq5(float(args.pp), ens.mean_AV, cfg)
            row = {
                "target_vi": float(target_vi),
                "target_V_pL": float(target_V),
                "target_rho_per_uL": rho,
                "realised_vi": float(ens.vi),
                "realised_V_pL": float(ens.V),
                "realised_rho_per_uL": float(ens.rho),
                "p_p": float(args.pp),
                "mean_A_over_V_um_inv": float(ens.mean_AV),
                "mean_A_over_V_se_um_inv": float(ens.geometry.mean_A_over_V_se_um_inv),
                "kio_event_s_inv": float(event_rate),
                "kio_event_se_s_inv": float(event_se),
                "kio_survival_fit_s_inv": float(survival_fit),
                "kio_eq5_s_inv": float(analytic),
                "event_over_eq5": (float(event_rate / analytic) if analytic > 0 else None),
                "first_exit_events": int(np.sum(telemetry["metrics"][:, 6])),
                "escaped_walkers": int(escaped),
            }
            rows.append(row)
            print(
                "v_i={realised_vi:.4f}, V={realised_V_pL:.4f} pL, "
                "rho={realised_rho_per_uL:.0f}/uL: direct="
                "{kio_event_s_inv:.4g}+/-{kio_event_se_s_inv:.2g}, "
                "survival={kio_survival_fit_s_inv:.4g}, Eq5={kio_eq5_s_inv:.4g}, "
                "ratio={event_over_eq5:.4g}".format(**row),
                flush=True,
            )

    ratios = np.asarray([x["event_over_eq5"] for x in rows], dtype=float)
    payload = {
        "schema": "madi-p0b-kio-calibration-v1",
        "tier": "B",
        "definition": (
            "k_io is tagged-starting-cell first-exit events divided by the "
            "at-risk starting-cell residence time; Eq. 5 is comparator only."
        ),
        "config": {
            "D0_um2_ms": cfg.D0,
            "ts_ms": cfg.ts,
            "L_um": cfg.L,
            "walkers": cfg.n_walkers,
            "duration_ms": cfg.T_max_ms,
            "phase_model": cfg.phase_model,
            "boundary_mode": cfg.boundary_mode,
        },
        "rows": rows,
        "event_over_eq5_range": [float(np.min(ratios)), float(np.max(ratios))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
