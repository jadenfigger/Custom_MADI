"""
Physical constants and simulation parameters.

Tailored to:
    9.4T MRSolutions preclinical scanner
    EPI-DWI, δ = 6 ms, Δ = 15 / 25 / 30 / 40 ms
    b = 0, 1000, 2500, 4000, 6000 s/mm²
    24 directions per shell, 97 volumes per Δ

Internal units:  μm (length), ms (time).

IMPORTANT (new semantics):
    `n_ensembles` is now the number of independent ensembles PER GRADIENT
    AXIS.  Because the powder average now uses three statistically
    independent ensembles (one per Cartesian axis), the total number of
    ensembles built per library entry is `3 × n_ensembles`, and the total
    number of random walks is `3 × n_ensembles × n_walkers`.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
D0_UM2_MS  = 3.0            # Pure water D at 37 °C          [μm²/ms]
D0_MOUSE   = 2.0            # Approximate in-vivo mouse D    [μm²/ms]
GAMMA_RAD  = 2.675222e8     # ¹H gyromagnetic ratio          [rad/(s·T)]


# ---------------------------------------------------------------------------
# Acquisition protocol
# ---------------------------------------------------------------------------
DELTA_SMALL = 6.0           # δ, PFG duration                [ms]
DELTAS_BIG  = [15.0, 25.0, 30.0, 40.0]   # Δ values         [ms]

# b-values in s/mm² (per acquisition, 97 volumes)
BVALS_S_MM2 = np.array([
    0,
    *([1000]*24), *([2500]*24), *([4000]*24), *([6000]*24)
])

# Unique non-zero b-values for library matching
BVALS_UNIQUE = np.array([1000, 2500, 4000, 6000])  # s/mm²

# Convert to internal units: ms/μm²  (divide s/mm² by 1e6)
BVALS_UNIQUE_INT = BVALS_UNIQUE / 1e6               # [ms/μm²]


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """All tuneable knobs for a MADI-GPU simulation run."""

    # --- Random walk -------------------------------------------------------
    D0:          float = D0_UM2_MS      # diffusion coefficient       [μm²/ms]
    ts:          float = 1e-3           # step time                   [ms] (= 1 μs)
    n_steps:     int   = 50_000         # steps per walk  (50 ms covers Δ=40 + δ=6 = 46 ms)

    # Walkers per ensemble (one ensemble = one gradient axis assignment).
    # Bumped from 20k → 50k so that the default library entry has:
    #   total walks = 3 × n_ensembles × n_walkers
    # With n_ensembles=4 below, that is 3 × 4 × 50k = 600 000 walks/entry,
    # 6× the previous default.  Paper target is ~12 M walks/entry — see
    # PAPER_WALKS_PER_ENTRY in walker_gpu.py.  Production users should set
    # n_walkers=400_000 and n_ensembles=10 to match the paper.
    n_walkers:   int   = 50_000

    # Number of independent ensembles PER AXIS.  Total ensembles built per
    # library entry is 3 × n_ensembles (one group of n_ensembles dedicated
    # to each of the x, y, z gradient axes).
    n_ensembles: int   = 4

    # --- Ensemble geometry -------------------------------------------------
    L:           float = 250.0          # Ω_sim cube side             [μm]
    buffer:      float = 60.0           # walker spawn margin (Ω_src) [μm]

    # Populated domain margin: seeds are sampled in [−pop_margin, L+pop_margin]³
    # so that cells near the faces of Ω_sim have their correct Poisson–
    # Voronoi neighbours present.  40 μm is comfortably larger than a few
    # mean cell spacings for ρ ≥ 100 000 cells/μL.  Increase if you push to
    # very low ρ.
    pop_margin:  float = 40.0

    # Per-cell annulus cap: α_i ≤ κ · d_nn/2.
    kappa:       float = 0.95

    # --- Voxelised lookup grid ---------------------------------------------
    grid_spacing: float = 1.0           # μm per grid voxel

    # --- Boundary-escape policy --------------------------------------------
    # Paper behaviour: abort on escape.  We allow a small slack so that the
    # rare statistical escape does not kill an otherwise good run.  If the
    # escaped-walker fraction exceeds this threshold, run_walks() raises
    # RuntimeError — matching the paper's "immediately cease" semantics
    # (SI §S.III).  Escapees below threshold are dropped silently.
    max_escape_frac: float = 0.01

    # --- SDE pulse sequence (your acquisition) -----------------------------
    delta:       float = DELTA_SMALL    # δ, PFG duration             [ms]
    Deltas:      list  = field(default_factory=lambda: list(DELTAS_BIG))

    # --- Library grid ------------------------------------------------------
    n_b_unique:  int   = 4              # unique non-zero b-values

    # --- Derived -----------------------------------------------------------
    @property
    def sigma(self) -> float:
        """Per-component Gaussian displacement std [μm]."""
        return np.sqrt(2.0 * self.D0 * self.ts)

    @property
    def ls_rms(self) -> float:
        """RMS step length [μm]."""
        return np.sqrt(6.0 * self.D0 * self.ts)

    @property
    def tRW_max(self) -> float:
        """Total walk duration [ms]."""
        return self.n_steps * self.ts

    def tD(self, Delta: float) -> float:
        """Effective diffusion time for a given Δ [ms]."""
        return Delta - self.delta / 3.0

    def pfg1_steps(self) -> tuple:
        """Step range for PFG-1 (same for all Δ)."""
        return (0, int(round(self.delta / self.ts)))

    def pfg2_steps(self, Delta: float) -> tuple:
        """Step range for PFG-2 for a given Δ."""
        start = int(round(Delta / self.ts))
        end   = int(round((Delta + self.delta) / self.ts))
        return (start, end)

    @property
    def grid_size(self) -> int:
        """Number of grid points per axis (Ω_sim only)."""
        return int(np.ceil(self.L / self.grid_spacing))
