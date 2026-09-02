#!/usr/bin/env python3
"""Validate v5 production shards or a merged artifact for Fisher analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from madi.fisher_crlb import (assert_safe_output, build_grid_manifest,
                               incomplete_banner, verify_v5_artifact)


def _paths(value: Path) -> list[Path]:
    if value.is_dir():
        return sorted(value.glob("*.shard*.npz"))
    return [value]


def _combined_manifest(paths: list[Path]):
    rhos, volumes, free = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            rhos.append(np.asarray(data["nominal_rhos"]))
            volumes.append(np.asarray(data["nominal_Vs"]))
            free.append(np.asarray(data["is_free_water"], dtype=bool))
    return build_grid_manifest(np.concatenate(rhos), np.concatenate(volumes), np.concatenate(free))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="merged .npz artifact or directory of shard .npz files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = _paths(args.input)
    if not paths:
        raise SystemExit(f"No shard files found under {args.input}")
    manifest = _combined_manifest(paths)
    assert_safe_output(args.output, manifest)
    reports = [verify_v5_artifact(path) for path in paths]
    report = {"schema": "madi-fisher-shard-verification-collection-v1",
              **manifest.as_dict(), "banner": incomplete_banner(manifest),
              "inputs": [str(p) for p in paths], "reports": reports,
              "pass": all(item["pass"] for item in reports)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["banner"])
    print(json.dumps({key: report[key] for key in ("pass", "grid_complete", "missing_groups")}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
