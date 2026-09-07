#!/usr/bin/env python3
"""Report fixed-S0 and S0-marginalized CRLBs, and the gap between them.

This is the Phase-2 entry point for the amplitude-model amendment adopted on
2026-09-05 (`madi/fisher_crlb_preregistration.json`, amendment
`2026-09-05-s0-marginalization`).  It is deliberately a reporting tool over
already-computed Phase-1 derivative fields, not a second derivative engine.

Phase 4 hypothesis H4 (the thesis data's lowest shell is b=50, not b=0) and the
Phase-5 estimator-efficiency comparison both consume `amplitude_marginal_fisher`
through this same path rather than reimplementing it.

Column basis.  Each acquisition declares its `(delta, Delta)` and `b` range and
those columns are then required to exist in the Phase-1 substrate; a missing one
raises.  `--gradient-scenario` optionally applies a named hardware ceiling as a
declared condition, which is then recorded in the report.  Before 2026-09-06 the
basis was whatever happened to survive Phase 1's restricted selection, silently:
the MADI III `(7, 25)` row was computed on 24 columns of which 10 exceed the very
300 mT/m limit that had shaped the substrate, because `(7, 25)` is also a
diagnostic timing pair and so kept all of its columns. See
docs/fisher_domain_audit.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from madi.fisher_crlb import (PARAMETER_ORDER, amplitude_marginal_diagnostics,
                               amplitude_marginal_fisher, amplitude_prior_precision,
                               artifact_manifest, assert_safe_output, canonical_grid,
                               column_arrays, gradient_feasible_columns,
                               gradient_strength_t_per_m, incomplete_banner,
                               load_preregistration, read_column_domain, read_npz_rows,
                               require_columns, te_noise_sigma)

AXES = ("rho", "V", "k_io")


def _load_axis(phase1: Path, axis: str) -> tuple[np.ndarray, dict[tuple[int, int, int], int]]:
    samples = np.load(phase1 / f"samples_{axis}_k1.npy")
    index = {(int(a), int(b), int(c)): row for row, (a, b, c) in enumerate(samples)}
    return np.load(phase1 / f"J_{axis}_k1.npy", mmap_mode="r"), index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snr", type=float, default=50.0)
    parser.add_argument("--nodes", type=int, default=24, help="evaluation nodes per acquisition")
    parser.add_argument("--n0-eff", type=float, default=4.0,
                        help="effective averages of the amplitude reference shell")
    parser.add_argument("--gradient-scenario", default=None,
                        help=("optional pre-registered gradient scenario (e.g. clinical, "
                              "research) applied as a DECLARED condition on the columns. "
                              "Omit for the unrestricted stored-column basis."))
    args = parser.parse_args()

    prereg = load_preregistration()
    T2 = float(prereg["noise_model"]["T2_ms"])
    t_epi = float(prereg["noise_model"]["t_epi_ms"])
    manifest_json = json.loads((args.phase1 / "phase1_manifest.json").read_text(encoding="utf-8"))
    domain = read_column_domain(manifest_json)
    selected = np.asarray(domain.column_indices, dtype=int)
    print(domain.banner())
    gradient_limits = prereg["gradient_limits_T_per_m"]
    if args.gradient_scenario is not None and args.gradient_scenario not in gradient_limits:
        raise SystemExit(f"unknown gradient scenario {args.gradient_scenario!r}; "
                         f"pre-registered scenarios are {sorted(gradient_limits)}")
    _, _, kios, _ = canonical_grid()

    fields = {axis: _load_axis(args.phase1, axis) for axis in AXES}
    shared = sorted(set(fields["rho"][1]) & set(fields["V"][1]) & set(fields["k_io"][1]))
    if not shared:
        raise SystemExit("no node carries all three k=1 derivative fields")

    with np.load(args.artifact, allow_pickle=False) as data:
        manifest = artifact_manifest(data)
        assert_safe_output(args.output, manifest)
        pair_deltas = np.asarray(data["pair_deltas"], dtype=float)
        pair_Deltas = np.asarray(data["pair_Deltas"], dtype=float)
        b_values = np.asarray(data["b_values"], dtype=float)
        n_b = int(data["n_b"])
        nominal = {"rho": np.asarray(data["nominal_rhos"], dtype=float),
                   "V": np.asarray(data["nominal_Vs"], dtype=float),
                   "kio": np.asarray(data["nominal_kios"], dtype=float)}
        free = np.asarray(data["is_free_water"], dtype=bool)
    rhos_c, Vs_c, _, _ = canonical_grid()
    delta_col, Delta_col, b_col = column_arrays(pair_deltas, pair_Deltas, b_values)

    def entry_of(ir: int, iv: int, ik: int) -> int:
        mask = (~free) & np.isclose(nominal["rho"], rhos_c[ir]) & \
               np.isclose(nominal["V"], Vs_c[iv]) & np.isclose(nominal["kio"], kios[ik])
        return int(np.flatnonzero(mask)[0])

    # Evenly spread the evaluation nodes over the shared node list so the report
    # is not dominated by one corner of the (rho, V, k_io) grid.
    picks = [shared[i] for i in np.linspace(0, len(shared) - 1, args.nodes).astype(int)]
    entries = [entry_of(*node) for node in picks]
    signals = read_npz_rows(args.artifact, "vectors", entries)

    acquisitions = {
        "madi_ii_single_delta_20_50": {"pairs": [(20.0, 50.0)], "b_min": 500.0, "b_max": 12000.0},
        "madi_iii_single_delta_7_25": {"pairs": [(7.0, 25.0)], "b_min": 500.0, "b_max": 12000.0},
        "jackson_like_b_range_only": {"pairs": [(20.0, 50.0)], "b_min": 500.0, "b_max": 4500.0},
    }

    report = {"schema": "madi-fisher-s0-marginal-crlb-v2", "artifact": str(args.artifact),
              "phase1": str(args.phase1), "banner": incomplete_banner(manifest),
              **manifest.as_dict(), "parameter_order": list(PARAMETER_ORDER),
              "snr_at_b0": args.snr, "T2_ms": T2, "t_epi_ms": t_epi,
              "n0_eff": args.n0_eff, "evaluation_nodes": len(picks),
              "column_domain": domain.as_dict(), "column_domain_banner": domain.banner(),
              "gradient_scenario": args.gradient_scenario,
              "gradient_limit_T_per_m": (None if args.gradient_scenario is None
                                         else gradient_limits[args.gradient_scenario]),
              "column_basis": ("declared (delta, Delta) and b range only; no hardware ceiling"
                               if args.gradient_scenario is None else
                               f"declared (delta, Delta) and b range, restricted to "
                               f"G <= {gradient_limits[args.gradient_scenario]} T/m"),
              "acquisitions": {}}

    for name, spec in acquisitions.items():
        want = np.zeros(len(b_col), dtype=bool)
        for delta, Delta in spec["pairs"]:
            want |= (delta_col == delta) & (Delta_col == Delta)
        want &= (b_col >= spec["b_min"]) & (b_col <= spec["b_max"])
        columns = np.flatnonzero(want)
        gradient_dropped = 0
        if args.gradient_scenario is not None:
            limit = gradient_limits[args.gradient_scenario]
            admissible = np.intersect1d(
                columns, gradient_feasible_columns(delta_col, Delta_col, b_col, limit))
            gradient_dropped = int(len(columns) - len(admissible))
            columns = admissible
        if not len(columns):
            report["acquisitions"][name] = {
                "error": "the declared column set is empty under the requested conditions",
                "gradient_infeasible_columns_excluded": gradient_dropped}
            continue
        # Not a silent intersection: every declared column must exist in the
        # substrate, or the reported acquisition is not the one that was asked for.
        position = require_columns(domain, columns, f"S0-marginal report, acquisition {name!r}",
                                   delta_col, Delta_col, b_col)
        sigma = te_noise_sigma(delta_col[columns], Delta_col[columns], sigma0=1.0 / args.snr,
                               T2_ms=T2, t_epi_ms=t_epi)

        rows = {"unknown_amplitude": [], "finite_b0_precision": [], "known_amplitude": []}
        detail = []
        for node, entry, signal_row in zip(picks, entries, signals):
            J = np.column_stack([np.asarray(fields[axis][0][fields[axis][1][node]], dtype=float)[position]
                                 for axis in AXES])
            s = np.asarray(signal_row, dtype=float)[columns]
            kio_ref = max(float(kios[node[2]]), 5.0)
            # The stored b grid starts at 0 and steps by 500, so the thesis's
            # b=50 reference shell has no stored column.  Its normalized signal
            # is extrapolated from the lowest stored positive shell under a
            # local mono-exponential law.  This only sets a scalar prior
            # precision, and S(50) sits within a few percent of 1, so the
            # extrapolation cannot carry the conclusion.
            s500 = float(np.asarray(signal_row, dtype=float)[np.flatnonzero(b_col == 500.0)[0]])
            adc_low = -np.log(max(s500, 1e-12)) / 500.0
            s_b50 = float(np.exp(-50.0 * adc_low))
            sigma_reference = float(te_noise_sigma(np.array([spec["pairs"][0][0]]),
                                                   np.array([spec["pairs"][0][1]]),
                                                   sigma0=1.0 / args.snr, T2_ms=T2, t_epi_ms=t_epi)[0])
            regimes = {
                "unknown_amplitude": 0.0,
                "finite_b0_precision": amplitude_prior_precision(args.n0_eff, sigma_reference, s_b50),
                "known_amplitude": float("inf"),
            }
            per_node = {"node": {"rho_index": node[0], "V_index": node[1], "k_io_index": node[2],
                                 "k_io": float(kios[node[2]])},
                        "reference_shell_signal_b50_extrapolated": s_b50}
            for regime, prior in regimes.items():
                if np.isinf(prior):
                    result = amplitude_marginal_fisher(J, s, sigma, s0_prior_precision=0.0)
                    diagnostics = amplitude_marginal_diagnostics(result, kio_ref)
                    ratio = np.ones(len(PARAMETER_ORDER))
                else:
                    result = amplitude_marginal_fisher(J, s, sigma, s0_prior_precision=prior)
                    diagnostics = amplitude_marginal_diagnostics(result, kio_ref)
                    ratio = diagnostics["crlb_ratio_marginal_over_fixed"]
                rows[regime].append(ratio)
                per_node[regime] = {"crlb_ratio_marginal_over_fixed": np.asarray(ratio).tolist(),
                                    "s0_prior_precision": None if np.isinf(prior) else prior,
                                    "loewner_ok": bool(result["loewner_ok"])}
                if regime == "unknown_amplitude":
                    per_node["crlb_fixed_s0"] = diagnostics["fixed_s0"]["crlb"].tolist()
                    per_node["crlb_marginal_s0"] = diagnostics["marginal_s0"]["crlb"].tolist()
            detail.append(per_node)

        summary = {}
        for regime, values in rows.items():
            stacked = np.asarray(values, dtype=float)
            finite = np.where(np.isfinite(stacked), stacked, np.nan)
            summary[regime] = {
                "crlb_ratio_median": np.nanmedian(finite, axis=0).tolist(),
                "crlb_ratio_max": np.nanmax(finite, axis=0).tolist(),
                "nonfinite_nodes": int(np.count_nonzero(~np.isfinite(stacked).all(axis=1))),
            }
        report["acquisitions"][name] = {
            "pairs_ms": spec["pairs"], "b_range_s_mm2": [spec["b_min"], spec["b_max"]],
            "columns_used": int(len(columns)),
            "gradient_infeasible_columns_excluded": gradient_dropped,
            "max_gradient_T_per_m_used": float(np.max(
                gradient_strength_t_per_m(delta_col[columns], Delta_col[columns], b_col[columns]))),
            "gap_summary_crlb_marginal_over_fixed": summary,
            "per_node": detail,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["banner"])
    for name, item in report["acquisitions"].items():
        if "error" in item:
            print(f"{name}: {item['error']}")
            continue
        print(f"\n{name}  ({item['columns_used']} columns)")
        for regime, stats in item["gap_summary_crlb_marginal_over_fixed"].items():
            median = ", ".join(f"{p}={v:.4f}" for p, v in zip(PARAMETER_ORDER, stats["crlb_ratio_median"]))
            print(f"   {regime:22s} median CRLB ratio  {median}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
