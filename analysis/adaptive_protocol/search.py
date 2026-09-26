"""Integer-budget multi-start annealing, constrained moves and persistent scores."""
from __future__ import annotations

import itertools
import json
import math
import sqlite3
import time

import numpy as np

from .core import canonical, digest, dumps, integer


def restore(obj):
    if isinstance(obj, dict):
        return {k: restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [restore(x) for x in obj]
    return math.inf if obj == "+inf" else -math.inf if obj == "-inf" else obj


class Design:
    def __init__(self, source, config, budget):
        self.source, self.config = source, config
        self.budget = integer(budget, "budget", 1)
        self.b0 = integer(config["reserved_b0_volumes"], "reserved_b0_volumes")
        self.minimum = integer(config["min_repetitions"], "min_repetitions", 1)
        self.maximum = self.budget if config["max_repetitions"] is None else integer(config["max_repetitions"], "max_repetitions", 1)
        self.max_columns = integer(config["max_distinct_columns"], "max_distinct_columns", 1)
        self.max_pairs = integer(config["max_timing_pairs"], "max_timing_pairs", 1)
        if self.b0 >= self.budget or self.maximum < self.minimum:
            raise ValueError("Inconsistent repetition/b0 budget constraints")
        if self.b0 and not self.minimum <= self.b0 <= self.maximum:
            raise ValueError("Reserved b0 count must fit per-column repetition limits")
        keep = source.columns[:, 2] > 0
        for axis, values_key, bounds_key in ((0, "delta_values_ms", "delta_bounds_ms"),
                                            (1, "Delta_values_ms", "Delta_bounds_ms"),
                                            (2, "b_values_s_mm2", "b_bounds_s_mm2")):
            values = config.get(values_key)
            if values is not None:
                if not set(values) <= set(source.columns[:, axis]):
                    raise ValueError(f"{values_key} includes a value absent from stored library")
                keep &= np.isin(source.columns[:, axis], values)
            bounds = config.get(bounds_key)
            if bounds is not None:
                if len(bounds) != 2 or bounds[0] > bounds[1]:
                    raise ValueError(f"Invalid {bounds_key}")
                keep &= (source.columns[:, axis] >= bounds[0]) & (source.columns[:, axis] <= bounds[1])
        limit = config["G_max_T_m"]
        if limit is not None:
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError("G_max_T_m must be positive or null")
            keep &= source.gradient <= limit
        self.allowed = np.flatnonzero(keep)
        self.allowed_set = set(map(int, self.allowed))
        self.pairs = sorted({tuple(source.columns[c, :2]) for c in self.allowed})
        self.by_pair = {pair: np.array([c for c in self.allowed if tuple(source.columns[c, :2]) == pair]) for pair in self.pairs}
        if not self.pairs:
            raise ValueError("No feasible stored diffusion-weighted columns in declared design space")

    def reason(self, protocol):
        try:
            protocol = canonical(protocol)
        except ValueError as exc:
            return str(exc)
        if sum(n for _, n in protocol) != self.budget:
            return "total_volume_budget"
        if len(protocol) > self.max_columns:
            return "max_distinct_columns"
        pairs, b0 = set(), 0
        for c, n in protocol:
            if c >= len(self.source.columns):
                return "absent_column"
            if n < self.minimum or n > self.maximum:
                return "repetition_bounds"
            pair = tuple(self.source.columns[c, :2]); pairs.add(pair)
            if self.source.columns[c, 2] == 0:
                b0 += n
                if pair not in self.by_pair:
                    return "b0_timing_outside_design"
            elif c not in self.allowed_set:
                return "column_outside_declared_design"
        if b0 != self.b0:
            return "reserved_b0_volume_budget"
        if len(pairs) > self.max_pairs:
            return "max_timing_pairs"
        return ""

    def balanced(self, ids, reference_pair=None):
        ids = list(dict.fromkeys(map(int, ids)))
        available = self.budget-self.b0
        if not ids or len(ids)*self.minimum > available or len(ids)*self.maximum < available:
            raise ValueError("Cannot allocate budget to chosen columns with repetition limits")
        q, r = divmod(available, len(ids))
        rows = [(c, q+(i < r)) for i, c in enumerate(ids)]
        if self.b0:
            if reference_pair is None:
                reference_pair = min((tuple(self.source.columns[c, :2]) for c in ids), key=sum)
            rows.append((self.source.exact_column((*reference_pair, 0)), self.b0))
        result = canonical(rows)
        why = self.reason(result)
        if why:
            raise ValueError(why)
        return result

    def timing_seed(self, pairs, shells=(1000, 2500, 4000, 6000)):
        ids = []
        for pair in pairs:
            choices = self.by_pair[pair]
            bs = self.source.columns[choices, 2]
            for b in shells:
                ids.append(int(choices[np.argmin(abs(bs-b))]))
        max_dw = min(self.max_columns-bool(self.b0), (self.budget-self.b0)//self.minimum)
        ids = list(dict.fromkeys(ids))[:max_dw]
        # High budgets and low maximum repeats may require more columns.
        needed = math.ceil((self.budget-self.b0)/self.maximum)
        if len(ids) < needed:
            ids = list(dict.fromkeys(ids+[int(c) for pair in pairs for c in self.by_pair[pair]]))[:max(needed, len(ids))]
        return self.balanced(ids)


class ScoreStore:
    def __init__(self, path, evaluator, run_id):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS evaluations (key TEXT PRIMARY KEY, protocol TEXT NOT NULL, result TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS history (run_id TEXT, evaluation_id INTEGER, body TEXT, PRIMARY KEY(run_id,evaluation_id))")
        self.evaluator, self.run_id = evaluator, run_id
        self.history, self.results = [], {}
        self.calls, self.cache_hits, self.fresh = 0, 0, 0
        self.start = time.perf_counter()
        self.best = math.inf

    def evaluate(self, protocol, design, stage, iteration=0, start=0):
        protocol = canonical(protocol)
        key = self.evaluator.key(protocol)
        self.calls += 1
        reason = design.reason(protocol)
        cached = False
        if reason:
            result = {"score": math.inf, "coverage": 0.0, "feasible": False, "failure": reason, "summaries": {}}
        else:
            row = self.db.execute("SELECT result FROM evaluations WHERE key=?", (key,)).fetchone()
            if row:
                result = restore(json.loads(row[0])); cached = True; self.cache_hits += 1
            else:
                answer = self.evaluator.evaluate(protocol)
                primary = answer["summaries"][self.evaluator.settings["objective_amplitude"]]
                result = dict(score=primary["objective_value"], coverage=primary["identifiable_fraction"],
                              feasible=primary["feasible"], failure="" if primary["feasible"] else "insufficient_coverage_or_infinite_objective",
                              summaries=answer["summaries"], TE_A_s=answer["TE_A_s"], sigma_single=answer["sigma_single"])
                self.db.execute("INSERT INTO evaluations VALUES (?,?,?)", (key, dumps(protocol), dumps(result)))
                self.fresh += 1
            self.results[protocol] = result
        if result["feasible"]:
            self.best = min(self.best, result["score"])
        record = dict(run_id=self.run_id, evaluation_id=self.calls, budget=design.budget,
                      stage=stage, iteration=iteration, start=start, evaluation_key=key,
                      protocol_id="p_"+digest(protocol), canonical_protocol_json=dumps(protocol),
                      acquisitions_json=dumps(self.evaluator.source.specification(protocol)),
                      score=result["score"], feasible=result["feasible"], coverage=result["coverage"],
                      failure=result["failure"], cache_hit=cached, best_so_far=self.best,
                      TE_A_s=result.get("TE_A_s"), sigma_single=result.get("sigma_single"),
                      elapsed_s=time.perf_counter()-self.start)
        self.history.append(record)
        self.db.execute("INSERT OR REPLACE INTO history VALUES (?,?,?)", (self.run_id, self.calls, dumps(record)))
        if self.calls % 100 == 0:
            self.db.commit()
        return result

    def close(self):
        self.db.commit(); self.db.close()


def ordering(result):
    return (not result["feasible"], result["score"] if result["feasible"] else -result["coverage"])


def neighbour(protocol, design, rng, forced_pair=None):
    """Transfer repeats, replace a column, or move a whole timing group."""
    rows = dict(protocol)
    dw = [c for c in rows if design.source.columns[c, 2] > 0]
    c = int(rng.choice(dw))
    mode = int(rng.integers(4))
    pool = design.allowed if forced_pair is None else design.by_pair[forced_pair]
    if mode == 0:
        # Integer repetition transfer may add/remove a selected column.
        target = int(rng.choice(dw if rng.random() < .7 else pool))
        if target == c:
            return canonical(rows.items())
        amount = int(rng.choice([1, max(1, rows[c]//2), rows[c]]))
        rows[c] -= amount; rows[target] = rows.get(target, 0)+amount
    elif mode == 1:
        choices = design.by_pair[tuple(design.source.columns[c, :2])]
        target = int(rng.choice(choices)); n = rows.pop(c)
        rows[target] = rows.get(target, 0)+n
    elif mode == 2 or forced_pair is not None:
        target = int(rng.choice(pool)); n = rows.pop(c)
        rows[target] = rows.get(target, 0)+n
    else:
        old = tuple(design.source.columns[c, :2]); new = design.pairs[int(rng.integers(len(design.pairs)))]
        moved = {}
        for col, n in rows.items():
            if tuple(design.source.columns[col, :2]) == old:
                b = design.source.columns[col, 2]
                col = design.source.exact_column((*new, b))
            moved[col] = moved.get(col, 0)+n
        rows = moved
    return canonical(rows.items())


def local_search(protocol, design, store, rng, passes, stage="local", forced_pair=None):
    current = protocol
    result = store.evaluate(current, design, stage)
    for iteration in range(passes):
        improved = False
        dw = [c for c, _ in current if design.source.columns[c, 2] > 0]
        for a, b in itertools.permutations(dw, 2):
            rows = dict(current)
            if rows[a] <= design.minimum:
                continue
            rows[a] -= 1; rows[b] += 1
            proposal = canonical(rows.items())
            next_result = store.evaluate(proposal, design, stage, iteration)
            if ordering(next_result) < ordering(result):
                current, result, improved = proposal, next_result, True
        for _ in range(80):
            proposal = neighbour(current, design, rng, forced_pair)
            next_result = store.evaluate(proposal, design, stage, iteration)
            if ordering(next_result) < ordering(result):
                current, result, improved = proposal, next_result, True
        if not improved:
            break
    return current, result


def optimize(design, store, config, extra_seeds=()):
    rng = np.random.default_rng(config["random_seed"]+design.budget)
    seeds = []
    for pair in design.pairs:
        try:
            protocol = design.timing_seed([pair])
        except ValueError:
            continue
        result = store.evaluate(protocol, design, "single_timing_seed")
        seeds.append((protocol, result, pair))
    if not seeds:
        raise ValueError("No single-timing seeds satisfy design constraints")
    seeds.sort(key=lambda x: ordering(x[1]))
    singles = []
    for protocol, _, pair in seeds[:min(3, len(seeds))]:
        refined = local_search(protocol, design, store, rng, config["local_passes"], "single_timing_refinement", pair)
        singles.append(refined)
    best_single = min(singles, key=lambda x: ordering(x[1]))
    multi = []
    if design.max_pairs >= 2:
        pairs = [item[2] for item in seeds[:8]]
        for pair in design.pairs[::max(1,len(design.pairs)//8)]:
            if pair not in pairs:
                pairs.append(pair)
        for a, b in itertools.combinations(pairs, 2):
            try:
                protocol = design.timing_seed([a, b])
            except ValueError:
                continue
            result = store.evaluate(protocol, design, "simple_two_timing_baseline")
            multi.append((protocol, result))
    best_multi = min(multi, key=lambda x: ordering(x[1])) if multi else best_single
    warm = [(p, store.evaluate(p, design, "warm_seed")) for p in extra_seeds if not design.reason(p)]
    starts = warm + [best_single, best_multi] + [(p, r) for p, r, _ in seeds[2:]]
    for start in range(config["starts"]):
        current, result = starts[start % len(starts)]
        best_local, best_result = current, result
        for iteration in range(config["iterations_per_start"]):
            proposal = neighbour(current, design, rng)
            next_result = store.evaluate(proposal, design, "annealing", iteration, start)
            fraction = iteration/max(1, config["iterations_per_start"]-1)
            temperature = config["initial_temperature"] * (config["final_temperature"]/config["initial_temperature"])**fraction
            accept = ordering(next_result) < ordering(result)
            if not accept and next_result["feasible"] and result["feasible"]:
                relative_change = (next_result["score"]-result["score"])/max(abs(result["score"]), 1e-12)
                accept = rng.random() < math.exp(-min(700, max(0, relative_change)/temperature))
            elif not next_result["feasible"] and not result["feasible"] and not design.reason(proposal):
                accept = rng.random() < math.exp(-max(0, result["coverage"]-next_result["coverage"])/temperature)
            if accept:
                current, result = proposal, next_result
            if ordering(result) < ordering(best_result):
                best_local, best_result = current, result
        local_search(best_local, design, store, rng, config["local_passes"], f"local_start_{start}")
        print(f"  N={design.budget}, start {start+1}/{config['starts']}, best score={store.best:.6g}, evaluated={store.calls}", flush=True)
    ranked = sorted(((p, r) for p, r in store.results.items()
                     if sum(n for _, n in p) == design.budget and r["feasible"] and not design.reason(p)),
                    key=lambda x: (x[1]["score"], x[0]))
    return {"best_single": best_single, "best_multi": best_multi,
            "top": ranked[:config["top_k"]], "seeds": [p for p, _ in starts[:config["starts"]]],
            "feasible_unique_protocols": len(ranked)}
