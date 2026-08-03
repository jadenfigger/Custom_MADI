"""
Physical constants and simulation parameters.

Internal units:  μm (length), ms (time).

(δ, Δ, b)-UNIVERSAL LIBRARY
----------------------------
The walk kernel accumulates the running position integral

    Y(t) = ∫₀ᵗ x(s) ds        (trapezoid rule at every 1-µs MC step,
                                origin-shifted for fp precision)

sampled at stride ``h_ms``, instead of two fixed PGSE moment windows. For ANY
(δ, Δ) whose δ, Δ, and Δ+δ are exact multiples of ``h_ms``:

    a(δ,Δ) = [Y(Δ+δ) − Y(Δ) − Y(δ)] / δ         (three lookups, zero error)
    S(b; δ,Δ) = ⟨cos(γ·G(b,δ,Δ)·δ·a)⟩

So δ, Δ, and b are *library storage grid* choices, not simulation inputs —
one Monte Carlo walk per tissue ensemble (to T_max_ms) fills the whole
(δ,Δ,b) table. The grids below are explicit and code-defined: matching at
fit time picks the nearest stored column — there is deliberately NO
interpolation (b-spline, bilinear, or otherwise) anywhere in this pipeline.
Any mismatch between a measured protocol and the stored grid is accepted as
error rather than interpolated away.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
D0_UM2_MS  = 3.0            # Pure water D at 37 °C          [μm²/ms]
D0_MOUSE   = 2.0            # Approximate in-vivo mouse D    [μm²/ms]
GAMMA_RAD  = 2.675222e8     # ¹H gyromagnetic ratio          [rad/(s·T)]


# ---------------------------------------------------------------------------
# Legacy fixed-δ acquisition constants — kept ONLY so old library .npz files
# (built before the (δ,Δ,b)-universal refactor) can still report sensible
# metadata via load_library_meta / library_summary. Do NOT use these in the
# simulation/build path.
# ---------------------------------------------------------------------------
DELTA_SMALL = 20.0          # δ, PFG duration                [ms]
DELTAS_BIG  = [50.0]        # Δ values         [ms]

BVALS_S_MM2 = np.array([
    0,
    *([500]*24), *([1000]*24), *([1500]*24), *([2000]*24), *([2500]*24)
])
BVALS_UNIQUE = np.array([500, 1000, 1500, 2000, 2500])  # s/mm²
BVALS_UNIQUE_INT = BVALS_UNIQUE / 1e6                    # [ms/μm²]


# ---------------------------------------------------------------------------
# (δ, Δ, b) library grid — explicit, code-defined, no interpolation
# ---------------------------------------------------------------------------
H_MS = 1.0                      # Y(t) storage stride [ms]

# δ: every ms, 1..30 ms
SMALL_DELTAS_MS: List[float] = list(range(1, 31))

# Δ: every ms up to 50 ms, then every 5 ms up to 80 ms
BIG_DELTAS_MS: List[float] = list(range(1, 51)) + list(range(55, 81, 5))

# Walk-duration ceiling [ms]. Deliberately larger than the grid's actual max
# Δ+δ (80+30=110 ms) to leave headroom for extending the grid later without
# re-simulating.
T_MAX_MS = 128.0

B_MAX_S_MM2 = 12000.0
B_STEP_S_MM2 = 500.0


def sqrt_spaced_bvalues(b_max: float = B_MAX_S_MM2, n: int = 28) -> np.ndarray:
    """b-values [s/mm²] spaced uniformly in sqrt(b), including b=0.

    Curvature of S(b) is largest near b=0 (Stejskal-Tanner decay is steepest
    there), so uniform spacing in sqrt(b) concentrates resolution where
    cubic-interpolation-style error would otherwise be worst — even though
    this library does not interpolate, it's still the right place to spend
    stored columns.

    NOT the default (see `evenly_spaced_bvalues`): real DWI protocols place
    shells on a coarse, evenly-spaced grid, and this curve's spacing grows
    far larger than a real acquisition's typical ~500 s/mm² shell spacing at
    high b, which silently drops most measured b-values at fit time (nearest-
    column matching has nothing close enough within tolerance). Kept for
    reference / non-default use.
    """
    return np.linspace(0.0, np.sqrt(b_max), n) ** 2


def evenly_spaced_bvalues(step: float = B_STEP_S_MM2,
                          b_max: float = B_MAX_S_MM2) -> np.ndarray:
    """b-values [s/mm²] evenly spaced by `step`, including b=0 and b_max.

    Default storage grid. Real DWI acquisitions place shells on a coarse,
    evenly-spaced grid (commonly multiples of ~500 s/mm²), so an evenly-
    spaced stored grid keeps nearest-column matching error small and
    predictable across the whole range, unlike `sqrt_spaced_bvalues` whose
    spacing varies by >50x from low to high b.
    """
    n = int(round(b_max / step)) + 1
    return np.arange(n, dtype=float) * step


def valid_delta_pairs(small_deltas=None, big_deltas=None) -> List[Tuple[float, float]]:
    """(δ, Δ) pairs with Δ ≥ δ (PGSE non-overlap constraint), triangular grid."""
    small = small_deltas if small_deltas is not None else SMALL_DELTAS_MS
    big   = big_deltas   if big_deltas   is not None else BIG_DELTAS_MS
    return [(float(d), float(D)) for d in small for D in big if D >= d]


def grid_time_index(t_ms: float, h_ms: float = H_MS, tol: float = 1e-6) -> int:
    """Index of t_ms on the Y(t) storage grid.

    Raises if t_ms is not an exact integer multiple of h_ms — this is the
    "assert at build time; fail loudly if violated" alignment check: any
    (δ,Δ) grid point that would require interpolating Y is a bug, not a
    degraded-precision case.
    """
    idx = t_ms / h_ms
    ridx = round(idx)
    if abs(idx - ridx) > tol:
        raise ValueError(
            f"t={t_ms} ms is not an exact integer multiple of h={h_ms} ms "
            f"(grid misalignment — this would require interpolating Y, "
            f"which this library forbids)."
        )
    return int(ridx)


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """All tuneable knobs for a MADI-GPU simulation run."""

    # --- Random walk ---------------------------------------------------------
    D0:          float = D0_UM2_MS      # diffusion coefficient       [μm²/ms]
    ts:          float = 1e-3           # step time                   [ms] (= 1 μs)
    T_max_ms:    float = T_MAX_MS       # walk duration ceiling       [ms]

    # Walkers per ensemble.
    n_walkers:   int   = 50_000

    # Independent ensembles built PER LIBRARY ENTRY (single isotropic
    # ensemble design — all 3 axes of each 3-D walk are harvested, so
    # N_eff = 3 × n_ensembles × n_walkers; NOT multiplied by 3 ensemble
    # builds the way the old per-axis scheme was).
    n_ensembles: int   = 4

    # Optional walker sub-chunk size for the Y-producing walk + reduction
    # (peak-memory tuning knob only — None processes all n_walkers of an
    # ensemble in one launch; set lower on memory-constrained GPUs).
    walker_chunk: Optional[int] = None

    # --- Ensemble geometry -----------------------------------------------------
    L:           float = 250.0          # Ω_sim cube side             [μm]
    # ``buffer`` and ``pop_margin`` are retained for loading old command
    # lines.  The production boundary mode is periodic, so walkers are
    # started uniformly in the whole box and no finite source domain is used.
    # They are used only by the explicit ``absorbing_legacy`` diagnostic mode.
    buffer:      float = 60.0           # legacy walker spawn margin [μm]
    pop_margin:  float = 40.0           # legacy source-seed margin  [μm]
    kappa:       float = 0.95           # per-cell annulus cap: α_i ≤ κ·d_nn/2

    # --- Periodic candidate classifier -------------------------------------
    # The GPU cannot call scipy's KD tree.  Both CPU and GPU therefore use
    # this same finite candidate cache and re-rank every candidate at the
    # actual walker location.  The cache resolution / candidate count are
    # validated against an exact periodic KD-tree in Tier-A tests.
    grid_spacing: float = 0.75           # μm per cache voxel
    classifier_candidates: int = 8       # primary seeds retained per voxel
    boundary_mode: str = "periodic"      # production: periodic; legacy only: absorbing_legacy

    # Geometry calibration is deliberately performed on every realised
    # Poisson ensemble.  ``geometry_vi_tolerance`` is an absolute tolerance:
    # 0.005 is >10x the binomial MC SE for 200k probes at all requested v_i,
    # while remaining far below one practical library grid cell.
    geometry_calibration_points: int = 200_000
    geometry_validation_points: int = 200_000
    geometry_sample_cells: int = 128
    geometry_vi_tolerance: float = 0.005
    geometry_alpha_iterations: int = 24

    # --- Boundary-escape policy --------------------------------------------
    # Only meaningful in the explicit legacy absorbing diagnostic.  A
    # production build rejects that mode rather than silently discarding
    # walkers from all columns.
    max_escape_frac: float = 0.01

    # --- Exchange calibration ------------------------------------------------
    # A direct tagged-starting-cell calibration is run once for each realised
    # geometry before its k_io sweep.  It measures a *response curve*, not a
    # single analytic or linear p_p→k_io slope: Gaussian proposals can make
    # the high-p response sub-linear because rejected walkers remain near a
    # membrane.  The final entry is always labelled by the independently
    # measured first-exit rate from its own signal walk.
    exchange_calibration_walkers: int = 4_096
    exchange_calibration_ms: float = 32.0
    exchange_calibration_response_points: int = 9
    exchange_calibration_min_pp: float = 1e-5
    exchange_calibration_min_events: int = 32
    exchange_calibration_max_batches: int = 4

    # --- Phase model ---------------------------------------------------------
    # ``finite_lobe`` is the only production model: it evaluates
    # gamma*integral(G(t).r(t))dt for rectangular PGSE lobes.  ``narrow_pulse``
    # exists solely for diagnostics and is never accepted by a production
    # library build.
    phase_model: str = "finite_lobe"

    # --- (δ, Δ, b) library grid ---------------------------------------------
    h_ms:         float = H_MS
    small_deltas: list  = field(default_factory=lambda: list(SMALL_DELTAS_MS))
    big_deltas:   list  = field(default_factory=lambda: list(BIG_DELTAS_MS))
    b_values:     list  = field(default_factory=lambda: list(evenly_spaced_bvalues()))

    # --- Derived -------------------------------------------------------------
    @property
    def sigma(self) -> float:
        """Per-component Gaussian displacement std [μm]."""
        return np.sqrt(2.0 * self.D0 * self.ts)

    @property
    def ls_rms(self) -> float:
        """RMS step length [μm]."""
        return np.sqrt(6.0 * self.D0 * self.ts)

    @property
    def n_steps(self) -> int:
        """MC steps per walk, covering the full T_max_ms."""
        return int(round(self.T_max_ms / self.ts))

    @property
    def tRW_max(self) -> float:
        """Total walk duration [ms]."""
        return self.n_steps * self.ts

    @property
    def steps_per_h(self) -> int:
        """MC steps between consecutive Y(t) storage samples."""
        ratio = self.h_ms / self.ts
        rounded = round(ratio)
        if abs(ratio - rounded) > 1e-6:
            raise ValueError(
                f"h_ms={self.h_ms} is not an exact integer multiple of "
                f"ts={self.ts} — Y(t) sampling would drift off the MC step grid."
            )
        return int(rounded)

    @property
    def n_grid(self) -> int:
        """Number of Y(t) storage samples per walker per axis (indices 0..n_grid-1)."""
        return int(round(self.T_max_ms / self.h_ms)) + 1

    @staticmethod
    def tD(delta: float, Delta: float) -> float:
        """Effective diffusion time L = Δ − δ/3 [ms] (Stejskal-Tanner)."""
        return Delta - delta / 3.0

    def delta_pairs(self) -> List[Tuple[float, float]]:
        """(δ, Δ) pairs stored in this config's library grid."""
        return valid_delta_pairs(self.small_deltas, self.big_deltas)

    def assert_grid_alignment(self) -> None:
        """Fail loudly if any δ, Δ, or Δ+δ is not an exact multiple of h_ms,
        or if the grid needs more time than T_max_ms provides."""
        for d in self.small_deltas:
            grid_time_index(d, self.h_ms)
        for D in self.big_deltas:
            grid_time_index(D, self.h_ms)
        pairs = self.delta_pairs()
        max_T = 0.0
        for d, D in pairs:
            grid_time_index(D + d, self.h_ms)
            max_T = max(max_T, D + d)
        if max_T > self.T_max_ms:
            raise ValueError(
                f"(δ,Δ) grid needs Δ+δ up to {max_T} ms but "
                f"T_max_ms={self.T_max_ms} ms."
            )
        _ = self.steps_per_h  # raises if h_ms/ts misaligned
        if self.phase_model not in {"finite_lobe", "narrow_pulse"}:
            raise ValueError(
                "phase_model must be 'finite_lobe' or 'narrow_pulse', got "
                f"{self.phase_model!r}."
            )
        if self.boundary_mode not in {"periodic", "absorbing_legacy"}:
            raise ValueError(
                "boundary_mode must be 'periodic' or 'absorbing_legacy', got "
                f"{self.boundary_mode!r}."
            )
        if self.classifier_candidates < 2:
            raise ValueError("classifier_candidates must be at least two.")
        if self.exchange_calibration_walkers <= 0:
            raise ValueError("exchange_calibration_walkers must be positive.")
        if self.exchange_calibration_ms <= 0.0:
            raise ValueError("exchange_calibration_ms must be positive.")
        if self.exchange_calibration_response_points < 3:
            raise ValueError("exchange_calibration_response_points must be at least three.")
        if not (0.0 < self.exchange_calibration_min_pp <= 1.0):
            raise ValueError("exchange_calibration_min_pp must lie in (0, 1].")
        if self.exchange_calibration_min_events < 1:
            raise ValueError("exchange_calibration_min_events must be positive.")
        if self.exchange_calibration_max_batches < 1:
            raise ValueError("exchange_calibration_max_batches must be positive.")

    @property
    def grid_size(self) -> int:
        """Number of grid points per axis (Ω_sim only)."""
        return int(np.ceil(self.L / self.grid_spacing))
