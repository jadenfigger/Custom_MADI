"""Non-simulation checks for the approved v5 production allocation."""

from __future__ import annotations

from madi.config import (
    PRODUCTION_AXIS_WALKS_PER_ENTRY,
    PRODUCTION_ENSEMBLES_PER_ENTRY,
    PRODUCTION_WALKERS_PER_ENSEMBLE,
    SimConfig,
)
from madi.library import _cfg_metadata
from scripts.fit_data import PRESETS


def test_dense_production_preset_and_metadata_record_six_million_axis_walks() -> None:
    dense = PRESETS["dense"]["cfg"]
    assert dense["n_walkers"] == PRODUCTION_WALKERS_PER_ENSEMBLE == 50_000
    assert dense["n_ensembles"] == PRODUCTION_ENSEMBLES_PER_ENTRY == 40
    assert PRODUCTION_AXIS_WALKS_PER_ENTRY == 6_000_000

    cfg = SimConfig(**dense)
    metadata = _cfg_metadata(cfg, build_seed=20_260_803)
    assert metadata["walkers_per_ensemble"] == 50_000
    assert metadata["ensembles_per_entry"] == 40
    assert metadata["axis_walks_per_entry"] == 6_000_000
    assert metadata["classifier_cache"]["mode"] == "exact"
