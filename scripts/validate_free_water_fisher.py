#!/usr/bin/env python3
"""Run the split free-water Fisher/CRLB validation gates on a v5 artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from madi.fisher_crlb import assert_safe_output, free_water_gate, incomplete_banner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigma-m", type=float, default=0.02)
    parser.add_argument("--tolerance", type=float, default=1e-12,
                        help="Predeclared absolute tolerance (default: %(default)g)")
    parser.add_argument("--observed-to-nominal-se-ratio", type=float, default=0.7,
                        help="Gate-B observed / 1/sqrt(6,000,000) SE ratio")
    args = parser.parse_args()
    report = free_water_gate(args.artifact, sigma_m=args.sigma_m, tolerance=args.tolerance,
                             observed_to_nominal_se_ratio=args.observed_to_nominal_se_ratio)
    # Recreate the small manifest needed only for output safety from report state.
    from madi.fisher_crlb import build_grid_manifest
    import numpy as np
    with np.load(args.artifact, allow_pickle=False) as data:
        manifest = build_grid_manifest(data["nominal_rhos"], data["nominal_Vs"], data["is_free_water"])
    assert_safe_output(args.output, manifest)
    report["banner"] = incomplete_banner(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["banner"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
