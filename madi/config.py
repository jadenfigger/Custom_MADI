"""
Physical constants and simulation parameters.

Tailored to:
    9.4T MRSolutions preclinical scanner
    EPI-DWI, δ = 6 ms, Δ = 15 / 25 / 30 / 40 ms
    b = 0, 1000, 2500, 4000, 6000 s/mm²
    24 directions per shell, 97 volumes per Δ

Internal units:  μm (length), ms (time).
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

    # --- Random walk ---
    D0:          float = D0_UM2_MS      # diffusion coefficient       [μm²/ms]
    ts:          float = 1e-3           # step time                   [ms] (= 1 μs)
    n_steps:     int   = 50_000         # steps per walk  (50 ms covers Δ=40 + δ=6 = 46 ms)
    n_walkers:   int   = 20_000         # walkers per ensemble
    n_ensembles: int   = 5              # independent ensembles

    # --- Ensemble geometry ---
    L:           float = 250.0          # simulation cube side        [μm]
    buffer:      float = 60.0           # walker init margin          [μm]
    kappa:       float = 0.40           # annulus limit fraction

    # --- Voxelised lookup grid ---
    grid_spacing: float = 1.0           # μm per grid voxel

    # --- SDE pulse sequence (your acquisition) ---
    delta:       float = DELTA_SMALL    # δ, PFG duration             [ms]
    Deltas:      list  = field(default_factory=lambda: list(DELTAS_BIG))

    # --- Library grid ---
    n_b_unique:  int   = 4              # unique non-zero b-values

    # --- Derived ---
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
        """Number of grid points per axis."""
        return int(np.ceil(self.L / self.grid_spacing))
