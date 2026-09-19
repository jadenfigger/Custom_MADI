"""Shared data access and drawing helpers for `scripts/plot_fisher_visualizations.py`.

This module computes no new Fisher mathematics.  Every matrix it forms goes
through the audited Phase-2 per-column helper `scripts.run_fisher_phase2.
pair_contributions`, and every derived quantity (spectra, profiled blocks,
CRLBs, kappa, positive-definiteness) goes through `madi.fisher_crlb`.  Where a
quantity is already stored by an executed run, it is loaded instead, and any
matrix formed here is cross-checked against the stored Phase-3 maps before it is
drawn (`check_against_phase3`).

Layers, as in docs/fisher_crlb_analysis_plan.md section 2.8:
  * the reusable substrate is the full stored column grid (31,125 columns);
  * a gradient ceiling, the TE/T2 noise model, Rician validity, the trust floor
    and the image budget are CONDITIONAL declarations.  They only ever enter
    here as the declared masks/weights of a named acquisition, or as drawn
    annotations -- never as a filter on what is loaded.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from madi.fisher_crlb import (PARAMETER_ORDER, canonical_grid, column_arrays, fisher_spectrum,
                              gradient_feasible_columns, load_preregistration, nondimensionalized_fisher,
                              packed_inverse_diagonal, read_column_domain, rho_V_profiled_block,
                              te_noise_sigma, unpack_fisher)
from madi.library import make_remediation_log_grid
from scripts.run_fisher_phase2 import (_score_block, build_node_table, greedy_b_subset,
                                       pair_contributions, transposed_view)

_LOG_GRID = make_remediation_log_grid()
REPO = Path(__file__).resolve().parents[1]
RUNS = Path("/home/jaden/madi_fisher_runs/full_domain")

# Authoritative artifact locations, from docs/fisher_phase2.md section 9,
# docs/fisher_phase3.md section 7 and docs/fisher_phase4.md section 6.
DEFAULT_PATHS = {
    "library": REPO / "data/libraries/madi_dense_universal_remediated.npz",
    "phase1": RUNS / "phase1",            # unrestricted substrate manifest + stencil samples
    "cache": RUNS / "cache",              # Phase-2 column cache, all 31,125 columns
    "phase2": RUNS / "phase2_N128",       # executed N = 128 sweep (reproduction run)
    "phase3": RUNS / "phase3",            # nine declared domains, per-node maps
    "figures": REPO / "docs/figures",
    # Derived plotting cache (safe to delete; never read by any analysis).
    "plot_cache": Path.home() / ".cache/madi_fisher_viz",
}

# ---------------------------------------------------------------------------
# Style: the reference palette used unmodified (same tokens as
# scripts/plot_fisher_phase3.py).  Sequential = one hue light->dark; diverging
# = blue <-> orange through a neutral gray; categorical slots in fixed order.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"
NEUTRAL = "#f0efec"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGE_RAMP = ["#fce4d6", "#f9cbb1", "#f5b18e", "#f19a6c", "#ee8050", "#eb6834", "#d75c2b",
               "#c05023", "#a8441c", "#8f3816", "#752c10", "#5c220b", "#431806"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]          # slots 1-3 (validated all-pairs)
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_orange", [BLUE_RAMP[11], BLUE_RAMP[6], BLUE_RAMP[1], NEUTRAL,
                        ORANGE_RAMP[1], ORANGE_RAMP[5], ORANGE_RAMP[9]])
# Undefined-result styling, used identically in every figure.
UNDEFINED_NOT_PD = dict(facecolor="#e7e6e1", edgecolor=INK_MUTED, hatch="////", linewidth=0.0)
UNDEFINED_NO_STENCIL = dict(facecolor="#f3f2ee", edgecolor=AXIS_RULE, hatch="....", linewidth=0.0)

PARAM_LABELS = (r"$\log\rho$", r"$\log V$", r"$k_{io}$")


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.edgecolor": AXIS_RULE, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "grid.color": GRIDLINE, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "legend.frameon": False, "hatch.linewidth": 0.5, "figure.dpi": 110,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, stem: str, out_dir: Path, dpi: int = 300, vector: str | None = "pdf") -> list[Path]:
    """Write `<stem>.png` (and a vector twin) under `out_dir`.

    Each file is written under a temporary name and moved into place.  On the
    Windows-mounted drive an image viewer or Explorer preview can briefly hold
    a figure open, and overwriting it in place then fails with EINVAL; the move
    is retried for a few seconds instead.
    """
    import os
    import time
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / f"{stem}.png"] + ([out_dir / f"{stem}.{vector}"] if vector else [])
    for path in written:
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        fig.savefig(temporary, bbox_inches="tight", **({"dpi": dpi} if path.suffix == ".png" else {}))
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(1.0)
    plt.close(fig)
    return written


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

class FisherData:
    """Lazy, read-only access to the executed Phase 0-3 artifacts.

    Nothing is loaded until a figure asks for it, and every property is cached,
    so `main()` pays each load once.
    """

    def __init__(self, paths: dict | None = None) -> None:
        self.paths = {**DEFAULT_PATHS, **(paths or {})}

    # -- pre-registration and reports ----------------------------------------
    @cached_property
    def prereg(self) -> dict:
        return load_preregistration()

    @cached_property
    def phase2(self) -> dict:
        return json.loads((self.paths["phase2"] / "phase2_report.json").read_text(encoding="utf-8"))

    @cached_property
    def phase3(self) -> dict:
        return json.loads((self.paths["phase3"] / "phase3_report.json").read_text(encoding="utf-8"))

    @cached_property
    def column_domain(self):
        """The Phase-1 substrate declaration; plotting refuses a restricted one."""
        manifest = json.loads((self.paths["phase1"] / "phase1_manifest.json").read_text(encoding="utf-8"))
        domain = read_column_domain(manifest)
        if not domain.is_complete:
            raise RuntimeError("these figures read the model layer and need the unrestricted "
                               f"substrate: {domain.banner()}")
        return domain

    # -- library metadata (small members only; `vectors` is never materialised)
    @cached_property
    def _library_meta(self) -> dict:
        with np.load(self.paths["library"], allow_pickle=False) as data:
            meta = {key: np.asarray(data[key]) for key in
                    ("pair_deltas", "pair_Deltas", "b_values", "n_b", "nominal_rhos", "nominal_Vs",
                     "nominal_kios", "is_free_water")}
            meta["table"] = build_node_table(self.paths["phase1"], data)
        return meta

    @property
    def pair_deltas(self) -> np.ndarray:
        return self._library_meta["pair_deltas"].astype(float)

    @property
    def pair_Deltas(self) -> np.ndarray:
        return self._library_meta["pair_Deltas"].astype(float)

    @property
    def b_values(self) -> np.ndarray:
        return self._library_meta["b_values"].astype(float)

    @property
    def n_b(self) -> int:
        return int(self._library_meta["n_b"])

    @property
    def n_entries(self) -> int:
        return int(len(self._library_meta["is_free_water"]))

    @cached_property
    def table(self) -> dict:
        """Phase-2 node table (nodes carrying all three k = 1 stencils).

        Asserted to be row-aligned with the Phase-3 maps, so a map row and a
        table row always name the same `(rho, V, k_io)` node.
        """
        table = self._library_meta["table"]
        labels = np.load(self.paths["phase3"] / "evaluation_node_labels.npy")
        if not (np.allclose(labels[:, 0], table["rho"]) and np.allclose(labels[:, 1], table["V"])
                and np.allclose(labels[:, 2], table["kio"])):
            raise RuntimeError("node table and Phase-3 map rows are not aligned")
        return table

    @property
    def nodes(self) -> np.ndarray:          # (n, 3) canonical (rho_index, V_index, k_io_index)
        return self.table["nodes"]

    @property
    def rho(self) -> np.ndarray:
        return self.table["rho"]

    @property
    def V(self) -> np.ndarray:
        return self.table["V"]

    @property
    def kio(self) -> np.ndarray:
        return self.table["kio"]

    @property
    def v_i(self) -> np.ndarray:            # intracellular volume fraction, rho[cells/uL] * V[pL] * 1e-6
        return self.table["rho"] * self.table["V"] * 1e-6

    @cached_property
    def kio_ref(self) -> np.ndarray:
        """`max(k_io, k_io_floor)`: the pre-registered D = diag(1, 1, k_io_ref) scale."""
        floor = float(self.prereg["protocol_sweep"]["criteria"]["relative_scale_for_k_io"]["k_io_floor_s^-1"])
        return np.maximum(self.table["kio"], floor)

    @cached_property
    def grid(self) -> dict:
        rhos, Vs, kios, retained = canonical_grid()
        evaluated = {(int(a), int(b)) for a, b, _ in self.table["nodes"]}
        return {"rhos": rhos, "Vs": Vs, "kios": kios, "retained": retained,
                "evaluated_pairs": evaluated,
                # Band-edge groups: in the library, but without a complete k = 1 stencil.
                "edge_pairs": set(retained) - evaluated,
                "h_log10_rho": float(np.log10(rhos[1] / rhos[0])),
                "h_log10_V": float(np.log10(Vs[1] / Vs[0])),
                # The library's own mask band (deviations_from_paper.md, R1 of the domain audit).
                "vi_band": (float(_LOG_GRID.vi_min), float(_LOG_GRID.vi_max))}

    # -- stored per-node maps -------------------------------------------------
    def map3(self, domain: str, name: str) -> np.ndarray:
        """A Phase-3 per-node map, e.g. `map3("full_stored_domain", "kappa")`."""
        return np.load(self.paths["phase3"] / f"maps_{domain}.{name}.npy")

    def map2(self, scenario: str, arm: str, regime: str, quantity: str) -> np.ndarray:
        """A Phase-2 per-node map (subset size 8 only, as the run wrote them)."""
        return np.load(self.paths["phase2"] / f"maps_{scenario}_size8_{arm}.{regime}.{quantity}.npy")

    # -- column cache and noise model -----------------------------------------
    @cached_property
    def n_ensembles(self) -> int:
        return int(np.load(self.paths["cache"] / "ensemble_means_subset.npy", mmap_mode="r").shape[1])

    @cached_property
    def sigma_pair(self) -> np.ndarray:
        """Single-average TE/T2 noise per timing pair at the pre-registered SNR 50."""
        noise = self.prereg["noise_model"]
        return te_noise_sigma(self.pair_deltas, self.pair_Deltas, sigma0=1.0 / float(self.phase3["snr_at_b0"]),
                              T2_ms=float(noise["T2_ms"]), t_epi_ms=float(noise["t_epi_ms"]))

    @cached_property
    def columns(self) -> dict:
        delta, Delta, b = column_arrays(self.pair_deltas, self.pair_Deltas, self.b_values)
        return {"delta": delta, "Delta": Delta, "b": b}

    def feasible_mask(self, scenario: str) -> np.ndarray:
        """CONDITIONAL: which stored columns a declared gradient ceiling admits."""
        g_max = float(self.prereg["gradient_limits_T_per_m"][scenario])
        mask = np.zeros(len(self.columns["b"]), dtype=bool)
        mask[gradient_feasible_columns(self.columns["delta"], self.columns["Delta"], self.columns["b"], g_max)] = True
        return mask

    @cached_property
    def transposed(self) -> tuple[np.ndarray, np.ndarray]:
        """Column-major `(columns, entries)` memmaps of the Phase-2 cache."""
        self.column_domain          # refuse a restricted substrate before reading anything
        return (transposed_view(self.paths["cache"], "vectors"),
                transposed_view(self.paths["cache"], "signal_variance"))

    def pair_index(self, delta: float, Delta: float) -> int:
        match = np.flatnonzero((self.pair_deltas == float(delta)) & (self.pair_Deltas == float(Delta)))
        if match.size != 1:
            raise KeyError(f"(delta, Delta) = ({delta}, {Delta}) ms is not a stored timing pair")
        return int(match[0])


# ---------------------------------------------------------------------------
# Representative nodes
# ---------------------------------------------------------------------------

def select_lattice(data: FisherData, k_io: float, rho_quantiles, vi_quantiles) -> dict:
    """A small lattice of representative evaluation nodes at one k_io slice.

    Selection rule -- purely geometric, no Fisher result is consulted:
      1. take the evaluation nodes at the canonical k_io node nearest `k_io`;
      2. targets are the given quantiles of log10 rho (along the band) and of
         log10 v_i (across it) over exactly those nodes;
      3. each target takes the nearest node in grid-step units
         (log10 rho / h_rho, log10 v_i / h_V); a node already taken passes to
         the next nearest, in stable order.
    Lattice rows run from high v_i (top) to low; columns from low rho to high.
    The centre cell is the reference node used by figures 3, 8 and 9.
    """
    g = data.grid
    k_index = int(np.argmin(np.abs(g["kios"] - float(k_io))))
    rows = np.flatnonzero(data.nodes[:, 2] == k_index)
    log_rho, log_vi = np.log10(data.rho[rows]), np.log10(data.v_i[rows])
    targets_rho = np.quantile(log_rho, rho_quantiles)
    targets_vi = np.quantile(log_vi, vi_quantiles)[::-1]
    lattice, used = {}, set()
    for i, target_vi in enumerate(targets_vi):
        for j, target_rho in enumerate(targets_rho):
            cost = ((log_rho - target_rho) / g["h_log10_rho"]) ** 2 + ((log_vi - target_vi) / g["h_log10_V"]) ** 2
            pick = next(int(rows[k]) for k in np.argsort(cost, kind="stable") if int(rows[k]) not in used)
            used.add(pick)
            lattice[(i, j)] = pick
    centre = (len(vi_quantiles) // 2, len(rho_quantiles) // 2)
    return {"k_io": float(g["kios"][k_index]), "k_io_index": k_index, "slice_rows": rows,
            "lattice": lattice, "reference": lattice[centre]}


def node_text(data: FisherData, row: int, multiline: bool = False) -> str:
    sep = "\n" if multiline else ", "
    return (rf"$\rho$={data.rho[row]:.3g} µL$^{{-1}}${sep}V={data.V[row]:.3g} pL{sep}"
            rf"$v_i$={data.v_i[row]:.2f}")


# ---------------------------------------------------------------------------
# Declared acquisitions and Fisher formation
# ---------------------------------------------------------------------------

@dataclass
class Acquisition:
    """A declared column set with its conditional masks, weights and budget."""

    name: str
    label: str
    layer: str                               # "model layer" or "conditional acquisition"
    b_by_pair: dict                          # timing-pair index -> b-values (s/mm^2)
    averages: float | None = None            # Rician validity at this averaging; None = not applied
    trust_floor: float | None = None         # S/S0 floor; None = not applied
    budget_scale: float = 1.0                # N / n_selected; 1.0 = one average per column
    phase3_domain: str | None = None         # the stored Phase-3 domain this reproduces, if any
    note: str = ""
    n_columns: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_columns = int(sum(len(v) for v in self.b_by_pair.values()))


def timing_text(data: FisherData, pairs) -> str:
    return " + ".join(f"({data.pair_deltas[p]:g}, {data.pair_Deltas[p]:g})" for p in sorted(pairs, key=lambda p: data.pair_Deltas[p]))


def full_stored_domain(data: FisherData) -> Acquisition:
    """MODEL LAYER: every stored diffusion-weighted column, one average each.

    Phase 3's `full_stored_domain` declaration: TE/T2 weights, no gradient,
    trust or Rician mask.  `b = 0` is excluded because S(0) = 1 for every entry,
    so J = 0 there (domain audit R12) -- a mathematical, not a conditional, exclusion.
    """
    b = data.b_values[1:]
    return Acquisition("full_stored_domain", "full stored domain", "model layer",
                       {p: b for p in range(len(data.pair_deltas))},
                       phase3_domain="full_stored_domain",
                       note=f"all {len(data.pair_deltas) * len(b):,} stored DW columns, 1 avg each, no masks")


def phase2_arm(data: FisherData, scenario: str, arm: str, size: int = 8) -> Acquisition:
    """CONDITIONAL: an executed Phase-2 optimum with its recorded b-subsets and budget."""
    report = data.phase2["scenarios"][scenario]["arms"][str(size)][arm]["report"]
    b_by_pair = {int(k): np.asarray(v, dtype=float) for k, v in report["b_values_s_mm2"].items()}
    g_mT = 1e3 * float(data.prereg["gradient_limits_T_per_m"][scenario])
    stored = f"phase2_optimum_{scenario}_size{size}_{arm}"
    return Acquisition(f"phase2_{scenario}_size{size}_{arm}", f"{timing_text(data, b_by_pair)} ms",
                       "conditional acquisition", b_by_pair,
                       averages=float(report["averages_per_column"]),
                       trust_floor=float(data.prereg["trust_floor"]),
                       budget_scale=float(report["budget_scale_N_over_columns"]),
                       phase3_domain=stored if stored in data.phase3["domains"] else None,
                       note=f"{scenario} {g_mT:g} mT/m · N={data.phase2['budget_images_N']:g} · SNR {data.phase2['snr_at_b0']:g}")


def phase2_greedy_second_delta(data: FisherData, scenario: str, size: int = 8) -> Acquisition:
    """CONDITIONAL: Phase 2's greedy m = 2 arm -- the best single-Delta pair plus one timing.

    `greedy_from_best_m1` records the pairs and the score but not the b-subsets,
    so they are re-selected with the audited `greedy_b_subset` under the sweep's
    own declarations (scenario gradient mask, averaging N/(2*size)), and the
    recorded score must reproduce.
    """
    record = data.phase2["scenarios"][scenario]["arms"][str(size)]["m2"]["greedy_from_best_m1"]
    budget = float(data.phase2["budget_images_N"])
    averages = budget / (2 * size)
    feasible = data.feasible_mask(scenario)
    vectors, variance = data.transposed
    trust = float(data.prereg["trust_floor"])
    rician_min = float(data.prereg["rician_magnitude_snr_min"])
    position = data.column_domain.position_of
    b_by_pair, packed = {}, np.zeros((len(data.rho), 6))
    for delta, Delta in record["pairs_ms"]:
        pair = data.pair_index(delta, Delta)
        full = pair * data.n_b + np.arange(1, data.n_b)
        usable = full[feasible[full]]
        sigma = float(data.sigma_pair[pair])
        tissue, _, _ = pair_contributions(data.table, vectors, variance, position[usable], 1.0 / sigma ** 2,
                                          data.n_ensembles, trust, rician_min * sigma / math.sqrt(averages))
        local = greedy_b_subset(tissue, data.kio_ref, (size,))[size]
        b_by_pair[pair] = data.columns["b"][usable[local]]
        packed += tissue[local].sum(axis=0)
    score = float(_score_block(packed[None], data.kio_ref, np.array([budget / (2 * size)]))[0][0])
    if not math.isclose(score, float(record["score"]), rel_tol=1e-4):
        raise RuntimeError(f"greedy m=2 b-subsets do not reproduce Phase 2 ({score} vs {record['score']})")
    g_mT = 1e3 * float(data.prereg["gradient_limits_T_per_m"][scenario])
    return Acquisition(f"phase2_{scenario}_size{size}_greedy_m2", f"{timing_text(data, b_by_pair)} ms",
                       "conditional acquisition", b_by_pair, averages=averages, trust_floor=trust,
                       budget_scale=budget / (2 * size),
                       note=f"{scenario} {g_mT:g} mT/m · N={budget:g} · SNR {data.phase2['snr_at_b0']:g}")


def _compact_inputs(data: FisherData, rows: np.ndarray):
    """The node table and cache restricted to the stencil entries of `rows`.

    Reads a few hundred rows of the row-major cache instead of streaming the
    whole column-major file; `pair_contributions` is unchanged.
    """
    table = data.table
    index_keys = ["centre"] + [f"{side}_{axis}" for side in ("minus", "plus") for axis in ("rho", "V", "k_io")]
    entries = np.unique(np.concatenate([table[key][rows] for key in index_keys]))
    lookup = np.full(data.n_entries, -1, dtype=int)
    lookup[entries] = np.arange(len(entries))
    sub = {key: lookup[table[key][rows]] for key in index_keys}
    sub.update({f"step_{axis}": table[f"step_{axis}"][rows] for axis in ("rho", "V", "k_io")})
    arrays = []
    for member in ("vectors", "signal_variance"):
        rows_major = np.load(data.paths["cache"] / f"{member}_selected.npy", mmap_mode="r")
        arrays.append(np.ascontiguousarray(np.asarray(rows_major[entries]).T))   # (columns, entries)
    return sub, arrays[0], arrays[1]


def form_fisher(data: FisherData, acquisition: Acquisition, rows=None) -> dict:
    """Packed debiased and undebiased Fisher matrices of a declared acquisition.

    Known-amplitude (fixed-S0) bound, matching the stored Phase-2/3 maps.  Each
    timing pair goes through `pair_contributions` with the acquisition's own
    weight 1/sigma_pair^2 and (node, column) masks, then the budget factor.
    `rows=None` evaluates every node; otherwise only those node rows.
    """
    data.column_domain
    if rows is None:
        table, (vectors, variance) = data.table, data.transposed
        rows = np.arange(len(data.rho))
    else:
        rows = np.atleast_1d(np.asarray(rows, dtype=int))
        table, vectors, variance = _compact_inputs(data, rows)
    tissue_sum = np.zeros((len(rows), 6))
    debias_sum = np.zeros((len(rows), 3))
    position = data.column_domain.position_of
    b_index = {float(b): i for i, b in enumerate(data.b_values)}
    trust = -np.inf if acquisition.trust_floor is None else acquisition.trust_floor
    rician_min = float(data.prereg["rician_magnitude_snr_min"])
    for pair, b_list in acquisition.b_by_pair.items():
        full = np.asarray([pair * data.n_b + b_index[float(b)] for b in b_list], dtype=int)
        sigma = float(data.sigma_pair[pair])
        rician = -np.inf if acquisition.averages is None else rician_min * sigma / math.sqrt(acquisition.averages)
        tissue, _, debias = pair_contributions(table, vectors, variance, position[full], 1.0 / sigma ** 2,
                                               data.n_ensembles, trust, rician)
        tissue_sum += tissue.sum(axis=0)
        debias_sum += debias.sum(axis=0)
    debiased = tissue_sum * acquisition.budget_scale
    undebiased = debiased.copy()
    undebiased[:, [0, 3, 5]] += debias_sum * acquisition.budget_scale
    if acquisition.phase3_domain is not None:
        check_against_phase3(data, acquisition.phase3_domain, rows, debiased)
    return {"rows": rows, "debiased": debiased, "undebiased": undebiased, "kio_ref": data.kio_ref[rows]}


def check_against_phase3(data: FisherData, domain: str, rows, packed, rtol: float = 1e-4) -> None:
    """One light check: the formed spectrum equals the stored Phase-3 map (float32)."""
    stored = data.map3(domain, "eigenvalues")[rows].astype(float)
    formed = fisher_spectrum(packed, data.kio_ref[rows])["eigenvalues"]
    if np.max(np.abs(formed - stored) / np.abs(stored[:, :1])) > rtol:
        raise RuntimeError(f"formed Fisher matrices disagree with stored Phase-3 maps for {domain}")


PAIR_PASS_FILE = "pair_pass_v1.npz"


def pair_pass(data: FisherData) -> dict:
    """Per-timing-pair summaries over every evaluation node (figure 2), plus the
    summed model-layer matrices at every node (figure 9).

    For each stored timing pair, with all 24 stored DW b-values and no masks:
      identifiable_fraction -- share of evaluation nodes where that pair alone
          gives a positive-definite debiased Fisher matrix.  All columns of one
          pair share one TE/T2 weight, so this is free of SNR, T2, t_epi and
          averaging: a property of the library's J and Var(J_hat) alone.
      sloppy_share -- median over nodes identifiable under full_stored_domain of
          v3' (D F_pair D) v3 / lambda_3, with (v3, lambda_3) the STORED Phase-3
          sloppy eigenpair; at each node the shares of all pairs sum to 1.
    Summed over pairs this is Phase 3's full_stored_domain accumulation.
    About 90 s; cached in paths["plot_cache"] (delete the file to rebuild).
    """
    cache = Path(data.paths["plot_cache"]) / PAIR_PASS_FILE
    if cache.exists():
        with np.load(cache) as stored:
            return {key: stored[key] for key in stored.files}
    vectors, variance = data.transposed
    n_pairs, n_nodes = len(data.pair_deltas), len(data.rho)
    sloppy = data.map3("full_stored_domain", "eigenvectors")[:, :, 2].astype(float)
    lambda3 = data.map3("full_stored_domain", "eigenvalues")[:, 2].astype(float)
    identifiable = data.map3("full_stored_domain", "positive_definite").astype(bool)
    out = {"identifiable_fraction": np.zeros(n_pairs), "sloppy_share": np.zeros(n_pairs),
           "full_debiased": np.zeros((n_nodes, 6)), "full_debias": np.zeros((n_nodes, 3))}
    position = data.column_domain.position_of
    for pair in range(n_pairs):
        full = pair * data.n_b + np.arange(1, data.n_b)
        tissue, _, debias = pair_contributions(data.table, vectors, variance, position[full], 1.0,
                                               data.n_ensembles, -np.inf, -np.inf)
        weight = 1.0 / float(data.sigma_pair[pair]) ** 2
        packed = tissue.sum(axis=0)
        out["identifiable_fraction"][pair] = packed_inverse_diagonal(packed)[2].mean()
        projected = np.einsum("ni,nij,nj->n", sloppy, nondimensionalized_fisher(packed * weight, data.kio_ref), sloppy)
        out["sloppy_share"][pair] = np.median(projected[identifiable] / lambda3[identifiable])
        out["full_debiased"] += packed * weight
        out["full_debias"] += debias.sum(axis=0) * weight
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def crlb_kappa(packed: np.ndarray, kio_ref) -> dict:
    """Relative CRLB and kappa on the Phase-2/3 definitions (Phase 3's own `crlb_report`).

    Relative CRLB: sqrt([F^-1]_jj) for log rho and log V, sqrt([F^-1]_33)/k_io_ref
    for k_io; kappa_j = sqrt([F^-1]_jj F_jj).  NaN wherever F fails the strengthened
    positive-definiteness test -- no bound exists there.
    """
    from scripts.run_fisher_phase3 import crlb_report
    report = crlb_report(np.atleast_2d(packed), np.atleast_1d(kio_ref))
    return {"relative": report["_relative"], "kappa": report["_kappa"], "positive": report["_positive"]}


# ---------------------------------------------------------------------------
# Drawing helpers: the (log10 rho, log10 V) plane
# ---------------------------------------------------------------------------
# Log10 axes at EQUAL aspect, so a drawn direction has its true angle.  A
# direction in natural-log (log rho, log V) coordinates keeps its angle in
# log10 coordinates because both axes are divided by the same ln(10).

def _plane_edges(data: FisherData) -> tuple[np.ndarray, np.ndarray]:
    g = data.grid
    lr, lv = np.log10(g["rhos"]), np.log10(g["Vs"])
    return (np.concatenate([lr - g["h_log10_rho"] / 2, [lr[-1] + g["h_log10_rho"] / 2]]),
            np.concatenate([lv - g["h_log10_V"] / 2, [lv[-1] + g["h_log10_V"] / 2]]))


def band_pairs(data: FisherData, which: str = "retained") -> tuple[np.ndarray, np.ndarray]:
    """`(rho_index, V_index)` arrays of the library groups: retained, evaluated or edge."""
    key = {"retained": "retained", "evaluated": "evaluated_pairs", "edge": "edge_pairs"}[which]
    pairs = sorted(data.grid[key])
    return (np.asarray([a for a, _ in pairs], dtype=int), np.asarray([b for _, b in pairs], dtype=int))


def setup_plane(ax, data: FisherData, vi_guides=(0.5, 0.6, 0.7, 0.8, 0.9), pad: float = 0.08,
                xlabel: bool = True, ylabel: bool = True) -> None:
    """Equal-aspect (log10 rho, log10 V) axes: band edges and subtle constant-v_i guides.

    Constant v_i is a straight line of slope -1 here: log10 V = log10(v_i 1e6) - log10 rho.
    """
    xe, ye = _plane_edges(data)
    rho_i, V_i = band_pairs(data)
    x0, x1 = xe[rho_i.min()] - pad, xe[rho_i.max() + 1] + pad
    y0, y1 = ye[V_i.min()] - pad, ye[V_i.max() + 1] + pad
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    xs = np.array([x0, x1])
    lo, hi = data.grid["vi_band"]
    for v in vi_guides:
        ax.plot(xs, np.log10(v * 1e6) - xs, color=GRIDLINE, linewidth=0.6, zorder=4)
    for v in (lo, hi):
        ax.plot(xs, np.log10(v * 1e6) - xs, color=AXIS_RULE, linewidth=0.9, zorder=4)
        # label each band edge where it leaves the lower-left / upper-right of the box
        x_text = x0 + 0.75 if v == lo else x1 - 0.95
        ax.text(x_text, np.log10(v * 1e6) - x_text + (0.08 if v == hi else -0.08),
                rf"$v_i$ = {v:.2f}", rotation=-45, rotation_mode="anchor", fontsize=7,
                color=INK_MUTED, ha="left", va="center" if v == hi else "top", zorder=5)
    ax.set_xticks([4, 5, 6, 7])
    ax.set_xticklabels([r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$"])
    yt = [t for t in (-2, -1, 0, 1, 2) if y0 <= t <= y1]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{10.0 ** t:g}" for t in yt])
    if xlabel:
        ax.set_xlabel(r"cell density $\rho$ (cells/µL, log scale)")
    if ylabel:
        ax.set_ylabel(r"cell volume $V$ (pL, log scale)")
    despine(ax)


def draw_cells(ax, data: FisherData, rho_index, V_index, values, cmap, norm, zorder: int = 2):
    """Heatmap cells on the canonical (rho, V) grid; cells without a value stay empty."""
    xe, ye = _plane_edges(data)
    image = np.full((len(ye) - 1, len(xe) - 1), np.nan)
    image[np.asarray(V_index), np.asarray(rho_index)] = values
    return ax.pcolormesh(xe, ye, np.ma.masked_invalid(image), cmap=cmap, norm=norm, shading="flat",
                         linewidth=0, zorder=zorder, rasterized=True)


def draw_marked_cells(ax, data: FisherData, rho_index, V_index, style_kw: dict, zorder: int = 3) -> None:
    """Hatched cells for undefined results (see UNDEFINED_* styles)."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    if len(rho_index) == 0:
        return
    xe, ye = _plane_edges(data)
    g = data.grid
    patches = [Rectangle((xe[i], ye[j]), g["h_log10_rho"], g["h_log10_V"])
               for i, j in zip(rho_index, V_index)]
    ax.add_collection(PatchCollection(patches, zorder=zorder, **style_kw))


def undefined_handles(not_pd: bool = True, no_stencil: bool = True) -> list:
    from matplotlib.patches import Patch
    handles = []
    if not_pd:
        handles.append(Patch(label="no CRLB: debiased F not positive definite", **UNDEFINED_NOT_PD))
    if no_stencil:
        handles.append(Patch(label="no Fisher matrix: band-edge group, incomplete stencil",
                             **UNDEFINED_NO_STENCIL))
    return handles


def per_pair(data: FisherData, rows: np.ndarray, values: np.ndarray, reducer=np.nanmedian):
    """Collapse node rows onto their (rho, V) groups: returns (rho_index, V_index, reduced)."""
    keys = data.nodes[rows, 0] * 1000 + data.nodes[rows, 1]
    out_r, out_v, out_x = [], [], []
    for key in np.unique(keys):
        members = rows[keys == key]
        with np.errstate(all="ignore"):
            out_x.append(float(reducer(values[members])))
        out_r.append(int(key // 1000))
        out_v.append(int(key % 1000))
    return np.asarray(out_r), np.asarray(out_v), np.asarray(out_x)


def inset_colorbar(ax, mappable, label: str, bounds=(0.56, 0.90, 0.38, 0.035), ticks=None, fmt=None):
    """Horizontal colorbar placed in the empty upper-right triangle of a plane panel."""
    cax = ax.inset_axes(bounds)
    bar = plt.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks, format=fmt)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, length=2, colors=INK_MUTED, labelcolor=INK_2)
    bar.set_label(label, fontsize=7.5, color=INK_2, labelpad=2)
    bar.ax.xaxis.set_label_position("top")
    return bar


def title(ax, text: str, sub: str | None = None) -> None:
    ax.set_title(text, loc="left", fontsize=10, color=INK, pad=14 if sub else 6)
    if sub:
        ax.text(0.0, 1.01, sub, transform=ax.transAxes, fontsize=7.5, color=INK_2, ha="left", va="bottom")


# ---------------------------------------------------------------------------
# Drawing helpers: matrices, spectra, ellipses
# ---------------------------------------------------------------------------

def fisher_correlation(packed: np.ndarray) -> np.ndarray:
    """R_jk = F_jk / sqrt(F_jj F_kk): free of any diagonal rescaling, so of D too.

    NaN in a row/column whose diagonal is not positive (possible after the debias).
    """
    F = unpack_fisher(packed)
    d = np.diagonal(F, axis1=-2, axis2=-1)
    root = np.sqrt(np.where(d > 0, d, np.nan))
    return F / (root[..., :, None] * root[..., None, :])


def draw_matrix(ax, M: np.ndarray, cmap, norm, fmt: str = "{:+.2f}", col_labels=PARAM_LABELS,
                row_labels=PARAM_LABELS, blank_diagonal: bool = False, fontsize: float = 8) -> object:
    """A 3x3 heatmap with value labels (ink chosen per cell for contrast) and surface gaps.

    `row_labels` / `col_labels` = None hides them.  `blank_diagonal` greys the
    diagonal (used for correlation matrices, whose diagonal is identically 1).
    """
    from matplotlib.patches import Rectangle
    shown = np.array(M, dtype=float)
    if blank_diagonal:
        np.fill_diagonal(shown, np.nan)
    image = ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, norm=norm)
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            value = shown[i, j]
            if blank_diagonal and i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=NEUTRAL, edgecolor="none"))
                ax.text(j, i, "1", ha="center", va="center", fontsize=fontsize - 0.5, color=INK_MUTED)
                continue
            if not np.isfinite(value):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=fontsize - 1, color=INK_MUTED)
                continue
            r, g, b, _ = cmap(norm(value))
            light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=fontsize,
                    color=INK if light else "white")
    ax.set_xticks(range(3))
    ax.set_yticks(range(len(shown)))
    ax.set_xticklabels(col_labels if col_labels is not None else [], fontsize=8)
    ax.set_yticklabels(row_labels if row_labels is not None else [], fontsize=8)
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, 3), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    despine(ax, keep=())
    return image


def profiled_axes(packed: np.ndarray) -> dict:
    """Eigen-decomposition of the k_io-profiled (log rho, log V) block S (madi.fisher_crlb).

    Eigenvalues descending; `vectors[:, i]` belongs to `values[i]`.  Both axes
    are natural-log parameters, so this is independent of the k_io_ref convention.
    """
    block, valid = rho_V_profiled_block(np.atleast_2d(packed))
    if not bool(valid[0]):
        return {"valid": False}
    values, vectors = np.linalg.eigh(block[0])
    return {"valid": True, "values": values[::-1], "vectors": vectors[:, ::-1], "block": block[0]}


def draw_profiled_ellipse(ax, packed: np.ndarray, color: str, limit: float, linewidth: float = 1.6) -> dict:
    """1-sigma CRLB contour {x : x' S x = 1} of the k_io-profiled block, in (dlog rho, dlog V).

    Where S has a non-positive eigenvalue the bound is infinite along that
    eigenvector: drawn as an open band of half-width 1/sqrt(l_informative),
    never as a closed ellipse.
    """
    from matplotlib.patches import Ellipse, Polygon
    axes_info = profiled_axes(packed)
    if not axes_info["valid"]:
        ax.text(0, 0, "no profiled block\n(F_kk ≤ 0)", ha="center", va="center", fontsize=7.5, color=INK_2)
        return {"kind": "none"}
    values, vectors = axes_info["values"], axes_info["vectors"]
    if values[1] > 0:
        semi = 1.0 / np.sqrt(values)                       # [minor (stiff), major (sloppy)]
        angle = math.degrees(math.atan2(vectors[1, 1], vectors[0, 1]))
        ax.add_patch(Ellipse((0, 0), 2 * semi[1], 2 * semi[0], angle=angle, facecolor=color, alpha=0.12,
                             edgecolor="none", zorder=3))
        ax.add_patch(Ellipse((0, 0), 2 * semi[1], 2 * semi[0], angle=angle, facecolor="none",
                             edgecolor=color, linewidth=linewidth, zorder=4))
        return {"kind": "ellipse", "semi_major": float(semi[1]), "semi_minor": float(semi[0])}
    if values[0] > 0:
        half = 1.0 / math.sqrt(values[0])
        along, across = vectors[:, 1], vectors[:, 0]
        L = 4.0 * limit
        corners = [s * L * along + t * half * across for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        ax.add_patch(Polygon(corners, closed=True, facecolor=color, alpha=0.10, edgecolor="none", zorder=3))
        for t in (-1, 1):
            ax.plot([-L * along[0] + t * half * across[0], L * along[0] + t * half * across[0]],
                    [-L * along[1] + t * half * across[1], L * along[1] + t * half * across[1]],
                    color=color, linewidth=linewidth, zorder=4)
        return {"kind": "band", "half_width": half}
    ax.text(0, 0, "no information\nin the plane", ha="center", va="center", fontsize=7.5, color=INK_2)
    return {"kind": "none"}
