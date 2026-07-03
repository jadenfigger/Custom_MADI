#!/usr/bin/env python3
"""
run_cohort_manifest.py — Drive scripts/fit_data.py across every row of a
tab-separated manifest produced by build_cohort_manifest.py (or hand-edited).

Each manifest row is one (subject x fitting mode) run.  Columns holding the
sentinel value "default" are NOT passed to fit_data.py as a flag at all, so
fit_data.py falls back to its own built-in default for that parameter.

Usage
-----
    python scripts/run_cohort_manifest.py --manifest data/manifests/edema_cohort_manifest.tsv
    python scripts/run_cohort_manifest.py --manifest ... --dry-run
    python scripts/run_cohort_manifest.py --manifest ... --subject sub-001 sub-002
    python scripts/run_cohort_manifest.py --manifest ... --run-id sub-001_MAP

    python scripts/run_cohort_manifest.py --manifest /mnt/c/miscellaneous/coding_projects/python/mri_processing/data_storage/data/edema/derivatives/edema_cohort_manifest.tsv

"""

import argparse
import csv
import os
import subprocess
import sys
import time

SENTINEL = "default"
TRUE_VALUES = {"true", "1", "yes"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIT_DATA_SCRIPT = os.path.join(SCRIPT_DIR, "fit_data.py")


def is_sentinel(value: str) -> bool:
    return value is None or value.strip() == "" or value.strip().lower() == SENTINEL


def is_true(value: str) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def build_command(row: dict, python_exe: str, fit_data_script: str):
    input_spec = f"{row['delta_ms']}:{row['dwi_path']}:{row['bval_path']}:{row['bvec_path']}"

    cmd = [
        python_exe, fit_data_script, "--fit",
        "--input", input_spec,
        "--mask", row["mask_path"],
        "--library", row["library"],
        "--out", row["out_dir"],
        "--method", row["method"],
        "--device", row["device"],
    ]

    if is_true(row.get("fit_s0")):
        cmd.append("--fit-s0")
    if is_true(row.get("rician_correct")):
        cmd.append("--rician-correct")
    if is_true(row.get("avg_s0")):
        cmd.append("--avg-s0")
    if is_true(row.get("log_space")):
        cmd.append("--log_space")

    for col, flag in [
        ("small_delta", "--small-delta"),
        ("sigma_m", "--sigma-m"),
        ("vi_min", "--vi-min"),
        ("vi_max", "--vi-max"),
        ("rho_max", "--rho-max"),
        ("noise_sigma", "--noise-sigma"),
        ("noise_bg_dilate_iters", "--noise-bg-dilate-iters"),
    ]:
        val = row.get(col)
        if not is_sentinel(val):
            cmd += [flag, val]

    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="Tab-separated manifest file.")
    ap.add_argument("--fit-data-script", default=DEFAULT_FIT_DATA_SCRIPT,
                     help=f"Path to fit_data.py (default {DEFAULT_FIT_DATA_SCRIPT}).")
    ap.add_argument("--python-executable", default=sys.executable,
                     help="Python interpreter to invoke fit_data.py with "
                          "(default: the interpreter running this script).")
    ap.add_argument("--subject", nargs="+", default=None,
                     help="Only run rows whose subject_id is in this list.")
    ap.add_argument("--run-id", nargs="+", default=None,
                     help="Only run rows whose run_id is in this list "
                          "(e.g. sub-001_MAP, sub-001_BAYES-fits0).")
    ap.add_argument("--method", nargs="+", default=None,
                     help="Only run rows whose method column matches "
                          "(map or bayes).")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print the commands that would run, without executing them.")
    ap.add_argument("--stop-on-error", action="store_true",
                     help="Abort the whole cohort on the first failed run "
                          "(default: log the failure and continue).")
    ap.add_argument("--log-dir", default=None,
                     help="Directory to write per-run stdout/stderr logs to "
                          "(default: <out_dir>/run.log next to each run's own output).")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}")
        return 1

    with open(args.manifest, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    if args.subject:
        rows = [r for r in rows if r["subject_id"] in args.subject]
    if args.run_id:
        rows = [r for r in rows if r["run_id"] in args.run_id]
    if args.method:
        rows = [r for r in rows if r["method"] in args.method]

    if not rows:
        print("No manifest rows matched the given filters.")
        return 1

    print(f"{len(rows)} run(s) selected from {args.manifest}")

    failures = []
    for i, row in enumerate(rows, 1):
        cmd = build_command(row, args.python_executable, args.fit_data_script)
        print(f"\n[{i}/{len(rows)}] {row['run_id']}")
        print("  " + " ".join(cmd))

        if args.dry_run:
            continue

        os.makedirs(row["out_dir"], exist_ok=True)
        log_dir = args.log_dir or row["out_dir"]
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "run.log")

        t0 = time.time()
        with open(log_path, "w") as log_fh:
            proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  ✗ FAILED (exit {proc.returncode}, {elapsed:.0f}s) — see {log_path}")
            failures.append(row["run_id"])
            if args.stop_on_error:
                break
        else:
            print(f"  ✓ done ({elapsed:.0f}s)")

    if args.dry_run:
        print(f"\nDry run: {len(rows)} command(s) printed, none executed.")
        return 0

    print(f"\n{len(rows) - len(failures)}/{len(rows)} run(s) succeeded.")
    if failures:
        print(f"Failed run_id(s): {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
