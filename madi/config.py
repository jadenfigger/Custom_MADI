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

    # --- Ensemble geometry -------------------------------------------------
    # SI §S.III Eq. S8 determines the production Ω_sim edge W separately for
    # every requested density.  ``L`` is intentionally an explicit
    # diagnostic override only; a production build leaves it ``None``.
    L: Optional[float] = None
    source_fraction: float = 0.40        # Ω_src edge / Ω_sim edge (SI §S.III)
    kappa: float = 0.90                  # SI Eq. S5: α_i^lim = κ d_nn / 2

    # Ω_pop is enlarged and certified against SI Eqs. S10--S12.  The
    # certificate evaluates a Lipschitz upper bound for the nearest-seed
    # distance throughout Ω_sim and verifies that every potentially touching
    # cell has an unbiased nearest neighbour for its annulus cap.
    population_initial_margin_cell_spacings: float = 3.0
    population_certification_spacing_um: float = 8.0
    population_max_expansions: int = 8

    # The SI's 5e6 single-cell / 26-alpha reference table supplies the
    # governing-process <A/V>.  Production refuses an uncertified table;
    # unit tests may opt in to a deliberately small fixture.
    geometry_reference_path: Optional[str] = None
    allow_uncertified_geometry_reference: bool = False
    geometry_reference_required_cells: int = 5_000_000
    geometry_reference_required_alpha_values: int = 26

    # Direct volume sampling validates that a finite Ω_sim realization is
    # close to the target generating-process v_i.  It does *not* calibrate
    # α* and it does not supply <A/V> for k_io.
    geometry_validation_points: int = 200_000
    geometry_vi_tolerance: float = 0.005

    # SI §S.III: escaping Ω_sim is a fatal simulation error.  There is no
    # periodic production mode and no survivor selection / escape fraction.
    boundary_mode: str = "si_fatal_escape"

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
        if self.boundary_mode != "si_fatal_escape":
            raise ValueError(
                "boundary_mode must be 'si_fatal_escape'; periodic wrapping "
                "and survivor pruning contradict SI §S.III, got "
                f"{self.boundary_mode!r}."
            )
        if not (0.0 < self.source_fraction <= 1.0):
            raise ValueError("source_fraction must lie in (0, 1].")
        if not np.isclose(self.kappa, 0.90, rtol=0.0, atol=1e-12):
            raise ValueError("kappa must be the SI-specified 0.90.")
        if self.population_initial_margin_cell_spacings <= 0.0:
            raise ValueError("population_initial_margin_cell_spacings must be positive.")
        if self.population_certification_spacing_um <= 0.0:
            raise ValueError("population_certification_spacing_um must be positive.")
        if self.population_max_expansions < 1:
            raise ValueError("population_max_expansions must be positive.")
        if self.geometry_validation_points <= 0:
            raise ValueError("geometry_validation_points must be positive.")
        if self.geometry_reference_required_cells < 1:
            raise ValueError("geometry_reference_required_cells must be positive.")
        if self.geometry_reference_required_alpha_values < 1:
            raise ValueError("geometry_reference_required_alpha_values must be positive.")

    @property
    def grid_size(self) -> int:
        """Removed cache-grid compatibility property.

        The production classifier evaluates the full SI Eq. S2 facet
        conjunction with an exact adaptive KD radius query; therefore a
        voxel-centre classifier grid is not part of the representation.
        """
        raise RuntimeError("the SI-exact classifier has no voxel cache grid")
