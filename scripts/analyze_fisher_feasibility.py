#!/usr/bin/env python3
"""Report Phase-0.4 column feasibility masks for a v5 MADI artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from madi.fisher_crlb import (artifact_manifest, assert_safe_output, column_arrays,
                               feasibility_masks, incomplete_banner, iter_npz_array_chunks,
                               load_preregistration)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snr", type=float, default=50.0,
                        help="b=0 magnitude SNR; sigma0=1/SNR")
    parser.add_argument("--T2-ms", type=float, default=80.0)
    parser.add_argument("--t-epi-ms", type=float, default=0.0)
    args = parser.parse_args()
    preregistration = load_preregistration()
    with np.load(args.artifact, allow_pickle=False) as data:
        manifest = artifact_manifest(data)
        assert_safe_output(args.output, manifest)
        free = np.asarray(data["is_free_water"], dtype=bool)
        common = dict(pair_deltas=np.asarray(data["pair_deltas"]),
                      pair_Deltas=np.asarray(data["pair_Deltas"]), b_values=np.asarray(data["b_values"]),
                      sigma0=1.0 / args.snr, T2_ms=args.T2_ms, t_epi_ms=args.t_epi_ms,
                      trust_floor=preregistration["trust_floor"],
                      rician_snr_min=preregistration["rician_magnitude_snr_min"])
        delta, Delta, b = column_arrays(common["pair_deltas"], common["pair_Deltas"], common["b_values"])
        n_columns = len(b)
        aggregate = {
            name: {"trust_any": np.zeros(n_columns, dtype=bool), "trust_all": np.ones(n_columns, dtype=bool),
                   "rician_any": np.zeros(n_columns, dtype=bool), "rician_all": np.ones(n_columns, dtype=bool),
                   "combined_any": np.zeros(n_columns, dtype=bool), "combined_all": np.ones(n_columns, dtype=bool),
                   "counts": []}
            for name in preregistration["gradient_limits_T_per_m"]
        }
        # Stream vectors: a full dense matrix expands beyond a normal local
        # workstation, while every required feasibility statistic is reducible
        # over entry chunks.
        for start, chunk in iter_npz_array_chunks(args.artifact, "vectors"):
            cellular = chunk[~free[start:start + len(chunk)]]
            if not len(cellular):
                continue
            for name, G_max in preregistration["gradient_limits_T_per_m"].items():
                masks = feasibility_masks(signals=cellular, **common, G_max=G_max)
                item = aggregate[name]
                for short, key in (("trust", "trust_floor_per_entry"),
                                   ("rician", "rician_per_entry"),
                                   ("combined", "combined_per_entry")):
                    item[f"{short}_any"] |= np.any(masks[key], axis=0)
                    item[f"{short}_all"] &= np.all(masks[key], axis=0)
                item["gradient"] = masks["gradient"]
                item["counts"].extend(np.count_nonzero(masks["combined_per_entry"], axis=1).tolist())
        results = {}
        derivative_selection: list[int] | None = None
        for name, item in aggregate.items():
            counts = np.asarray(item.pop("counts"), dtype=int)
            results[name] = {
                "gradient_columns": int(np.count_nonzero(item["gradient"])),
                "trust_floor_columns_any_entry": int(np.count_nonzero(item["trust_any"])),
                "trust_floor_columns_all_entries_global_min_diagnostic": int(np.count_nonzero(item["trust_all"])),
                "rician_columns_any_entry": int(np.count_nonzero(item["rician_any"])),
                "rician_columns_all_entries_global_min_diagnostic": int(np.count_nonzero(item["rician_all"])),
                "combined_columns_any_entry": int(np.count_nonzero(item["combined_any"])),
                "combined_columns_all_entries_global_min_diagnostic": int(np.count_nonzero(item["combined_all"])),
                "per_cellular_entry_surviving_columns": {
                    "min": int(np.min(counts)), "q05": float(np.quantile(counts, .05)),
                    "median": float(np.quantile(counts, .5)), "q95": float(np.quantile(counts, .95)),
                    "max": int(np.max(counts)),
                },
            }
            if name == "research":
                derivative_selection = np.flatnonzero(item["combined_any"]).astype(int).tolist()
        report = {"schema": "madi-fisher-column-feasibility-v1", "artifact": str(args.artifact),
                  "banner": incomplete_banner(manifest), **manifest.as_dict(),
                  "columns": int(len(b)), "snr_at_b0": args.snr, "T2_ms": args.T2_ms,
                  "t_epi_ms": args.t_epi_ms, "trust_floor_rule": "per_entry_inside_fisher_sum",
                  "global_minimum_rule": "diagnostic_only_not_used_for_fisher",
                  "counts": results,
                  "derivative_column_selection": {
                      "basis": "research combined mask, any cellular entry; diagnostic columns are added by Phase 1",
                      "column_indices": derivative_selection,
                      "count": len(derivative_selection or []),
                  }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["banner"])
    print(json.dumps(report["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
