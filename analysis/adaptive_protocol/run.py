"""Windows entry point: python -m analysis.adaptive_protocol.run --config ..."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .core import REPO, SCHEMA, Substrate, Evaluator, canonical, digest, dumps, replace_timing
from .export import append_table, common_metadata, definitions, result_rows, write_json
from .search import Design, ScoreStore, optimize


def sweeps(protocol, design, evaluator, store, meta):
    source = design.source
    pair = max({tuple(source.columns[c, :2]) for c, _ in protocol}, key=sum)
    rows = []

    def record(candidate, kind, **labels):
        result = store.evaluate(candidate, design, "sweep_"+kind)
        primary = result["summaries"].get(evaluator.settings["objective_amplitude"], {})
        pairs = {tuple(source.columns[c, :2]) for c, _ in candidate}
        row = dict(sweep_kind=kind, sweep_id=digest([protocol, kind, labels]), budget=design.budget,
                   base_protocol_id="p_"+digest(protocol), protocol_id="p_"+digest(candidate),
                   evaluation_key=evaluator.key(candidate), score=result["score"], feasible=result["feasible"],
                   coverage=result["coverage"], invalidity_reason=result["failure"],
                   scaled_trace_median=primary.get("scaled_trace_crlb_median"),
                   TE_A_s=result.get("TE_A_s"), sigma_single=result.get("sigma_single"),
                   n_timing_pairs=len(pairs), timing_sum_range_ms=float(np.ptp([sum(p) for p in pairs])),
                   acquisitions_json=dumps(source.specification(candidate)), sweep_group_json=dumps(pair), **labels, **meta)
        rows.append(row)

    for timing in design.pairs:
        candidate = replace_timing(protocol, pair, timing, source)
        record(candidate, "one_timing_group", delta_ms=timing[0], Delta_ms=timing[1])
    dw = [c for c, _ in protocol if source.columns[c, 2] > 0]
    col = max(dw, key=lambda c: source.columns[c, 2])
    for c in design.by_pair[tuple(source.columns[col, :2])]:
        changed = dict(protocol); n = changed.pop(col); changed[int(c)] = changed.get(int(c), 0)+n
        record(canonical(changed.items()), "one_b_column", b_s_mm2=float(source.columns[c, 2]))
    if len(dw) >= 2:
        a, b = dw[:2]
        for n in range(-dict(protocol)[b]+1, dict(protocol)[a]):
            changed = dict(protocol); changed[a] -= n; changed[b] += n
            record(canonical(changed.items()), "repeat_transfer", allocation=n, donor_column=a, recipient_column=b)
    pairs = sorted(design.pairs, key=sum)
    for m in range(1, design.max_pairs+1):
        timings = [pairs[int(i)] for i in np.linspace(0, len(pairs)-1, m)]
        try:
            candidate = design.timing_seed(timings, shells=(1000, 2500, 6000))
        except ValueError:
            continue
        record(candidate, "timing_diversity", requested_timing_pairs=m)
    return rows


def source_hashes():
    directory = Path(__file__).parent
    files = sorted(directory.glob("*.py"))
    files += [REPO/"madi/fisher_crlb.py", REPO/"scripts/run_fisher_phase2.py"]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-validation", action="store_true", help="For development only; result explicitly marked unvalidated")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    if not output.is_absolute():
        output = REPO/output
    # Outputs must stay in this independent workflow. No historical destination accepted.
    base_dir = Path(__file__).resolve().parent
    if not output.resolve().is_relative_to(base_dir) or output.resolve() == base_dir:
        raise ValueError("Output must be a child directory of analysis/adaptive_protocol")
    output.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        from .figures import make_figures
        report = json.loads((output/"run_report.json").read_text())
        # Never relabel saved numerical results with subsequently edited settings.
        make_figures(output, report["config"], report["run_id"])
        return
    lock_path = output/".writer.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("Another writer may be running. Check .writer.lock; remove only after verifying no active run.") from exc
    os.write(lock_fd, str(os.getpid()).encode()); os.close(lock_fd)
    start = time.perf_counter()
    try:
        print(f"Platform: {platform.system()}, Python: {sys.executable}", flush=True)
        source = Substrate(config)
        print(f"Read-only full substrate: {len(source.columns)} columns. Declared evaluation: {len(source.node_ids)} nodes.", flush=True)
        evaluator = Evaluator(source, config)
        hashes = source_hashes()
        run_id = "run_"+digest({"config": config, "source": source.fingerprint, "code": hashes})
        timestamp = datetime.now(timezone.utc).isoformat()
        manifest_path = output/f"{run_id}.json"
        if manifest_path.exists():
            timestamp = json.loads(manifest_path.read_text())["timestamp_utc"]
        code_version = digest(hashes)
        try:
            commit = subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        meta = common_metadata(config, run_id, timestamp, code_version)
        report = dict(run_id=run_id, timestamp_utc=timestamp, code_version=code_version, git_commit=commit,
                      source_hashes=hashes, config=config, platform=platform.platform(), executable=sys.executable,
                      source=source.provenance, budgets={}, validation_status="pending")
        write_json(manifest_path, report)
        all_summary, all_nodes, all_columns, all_history, all_sweeps, tops = [], [], [], [], [], []
        validated_single = None
        previous_best = None
        for budget in config["budgets"]:
            design = Design(source, config["design"], budget)
            # Preload existing J/signals in modest column batches; this is an ephemeral
            # evaluation working set, not a rewritten or restricted reusable cache.
            print(f"N={budget}: {len(design.allowed)} DW columns across {len(design.pairs)} timings", flush=True)
            for i in range(0, len(design.allowed), 64):
                source.get_columns(design.allowed[i:i+64])
                if i % 256 == 0:
                    print(f"  Reading stored columns {i}/{len(design.allowed)}", flush=True)
            store = ScoreStore(output/"candidate_evaluations.sqlite", evaluator, run_id)
            # IDs are global within this run; best-so-far resets at each budget.
            store.calls = len(all_history)
            warm = []
            if previous_best:
                # Preserve a previous budget's allocation proportions, round exactly,
                # and keep the explicitly reserved b0 budget fixed.
                dw = [(c,n) for c,n in previous_best if source.columns[c,2]>0]
                allocation = np.array([n for _,n in dw], float)
                allocation *= (budget-design.b0)/allocation.sum()
                counts = np.floor(allocation).astype(int)
                remaining = budget-design.b0-int(counts.sum())
                counts[np.argsort(-(allocation-counts), kind="stable")[:remaining]] += 1
                b0 = [(c,design.b0) for c,_ in previous_best if source.columns[c,2]==0]
                candidate = canonical([(c,int(n)) for (c,_),n in zip(dw,counts)]+b0)
                if not design.reason(candidate):
                    warm.append(candidate)
            found = optimize(design, store, config["optimizer"], warm)
            if not found["top"]:
                raise RuntimeError(f"No feasible protocol found for N={budget}; inspect saved candidate history. Increase search or revise explicitly declared constraints.")
            original_best = found["top"][0][0]
            all_sweeps.extend(sweeps(original_best, design, evaluator, store, meta))
            for user_protocol in config.get("protocols", []):
                if sum(row[3] for row in user_protocol["columns"]) == budget:
                    store.evaluate(source.protocol(user_protocol["columns"]), design, "user_protocol")
            ranked = sorted(((p, r) for p, r in store.results.items() if r["feasible"] and not design.reason(p)), key=lambda x: (x[1]["score"], x[0]))
            top = ranked[:config["optimizer"]["top_k"]]
            selections = {}
            for rank, (p, r) in enumerate(top, 1):
                selections.setdefault(p, set()).add(f"rank_{rank}")
                tops.append(dict(run_id=run_id, budget=budget, rank=rank, protocol_id="p_"+digest(p),
                                 objective=r["score"], coverage=r["coverage"], TE_A_s=r["TE_A_s"],
                                 sigma_single=r["sigma_single"], acquisitions_json=dumps(source.specification(p)), **config["noise"]))
            selections.setdefault(found["best_single"][0], set()).add("best_single_timing_baseline")
            selections.setdefault(found["best_multi"][0], set()).add("simple_multi_timing_baseline")
            for user_protocol in config.get("protocols", []):
                if sum(row[3] for row in user_protocol["columns"]) == budget:
                    p = source.protocol(user_protocol["columns"])
                    if design.reason(p):
                        raise ValueError(f"User protocol {user_protocol['name']} violates design: {design.reason(p)}")
                    selections.setdefault(p, set()).add(user_protocol["name"])
            for p, roles in selections.items():
                short = "Best found" if "rank_1" in roles else "Single timing" if "best_single_timing_baseline" in roles else "Simple multi" if "simple_multi_timing_baseline" in roles else sorted(roles)[0].replace("_", " ")
                answer = evaluator.evaluate(p, detailed=True)
                summary_rows, node_rows, column_rows = result_rows(answer, source, config, meta, short, roles)
                all_summary.extend(summary_rows); all_nodes.extend(node_rows); all_columns.extend(column_rows)
            all_history.extend(dict(h, **{k:v for k,v in meta.items() if k not in h}) for h in store.history)
            best_p, best_r = top[0]
            previous_best = best_p
            report["budgets"][budget] = dict(best_protocol=source.specification(best_p), best_protocol_id="p_"+digest(best_p),
                objective_value=best_r["score"], coverage=best_r["coverage"], TE_A_s=best_r["TE_A_s"], sigma_single=best_r["sigma_single"],
                summaries=best_r["summaries"], candidate_columns=len(design.allowed), candidate_timing_pairs=len(design.pairs),
                allowed_column_ids=design.allowed.tolist(), seeds=[source.specification(p) for p in found["seeds"]],
                best_single=source.specification(found["best_single"][0]), best_single_score=found["best_single"][1]["score"],
                simple_multi=source.specification(found["best_multi"][0]), simple_multi_score=found["best_multi"][1]["score"],
                candidate_requests=len(store.history), fresh_evaluations=store.fresh, cache_hits=store.cache_hits,
                failed_candidates=sum(not h["feasible"] for h in store.history), feasible_unique_protocols=len(ranked),
                runtime_s=time.perf_counter()-store.start,
                stopping_rule="configured starts x iterations, followed by bounded strict-improvement local passes and declared sweeps")
            validated_single = found["best_single"][0]
            store.close()
            print(f"N={budget}: best={best_r['score']:.6g}, coverage={best_r['coverage']:.3%}, TE={1000*best_r['TE_A_s']:.0f} ms", flush=True)
        if args.skip_validation:
            checks = {"status": "SKIPPED_BY_CLI"}; report["validation_status"] = "not_validated"
        else:
            from .validate import run_checks
            checks = run_checks(source, config, validated_single, output)
            report["validation_status"] = "passed"
        report["validation"] = checks
        report["runtime_s"] = time.perf_counter()-start
        write_json(output/"validation.json", checks)
        write_json(output/"run_report.json", report); write_json(manifest_path, report)
        tables = {
            "protocol_summary": (all_summary, ["evaluation_key", "amplitude_model"]),
            "node_metrics": (all_nodes, ["evaluation_key", "amplitude_model", "node_index"]),
            "acquisition_columns": (all_columns, ["evaluation_key", "full_column"]),
            "optimization_history": (all_history, ["run_id", "evaluation_id"]),
            "protocol_sweeps": (all_sweeps, ["run_id", "budget", "sweep_id"]),
            "top_protocols": (tops, ["run_id", "budget", "rank"]),
            "definitions_config": (definitions(config, source, run_id, {"validation": checks, "run_report": report}), ["run_id", "field"]),
        }
        for name, (rows, keys) in tables.items():
            count = append_table(output/f"{name}.csv", rows, keys)
            print(f"{name}: {count} rows", flush=True)
        # JSON types guide the Windows workbook exporter without reparsing CSV IDs.
        types = {name: {k: next(("number" if isinstance(row.get(k), (float, int, np.number)) and not isinstance(row.get(k), (bool, np.bool_)) else "boolean" if isinstance(row.get(k), (bool, np.bool_)) else "text"
                               for row in rows if row.get(k) is not None), "text")
                        for k in dict.fromkeys(k for row in rows for k in row)} for name,(rows,_) in tables.items()}
        write_json(output/"table_types.json", types)
        if not args.skip_figures:
            from .figures import make_figures
            files = make_figures(output, config, run_id)
            print(f"Saved {len(files)} figure files", flush=True)
        print(f"Complete numerical outputs: {output}", flush=True)
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
