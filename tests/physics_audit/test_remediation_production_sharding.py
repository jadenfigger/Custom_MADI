"""Pure assignment tests for the revised production shard layout."""

from __future__ import annotations

from madi.library import make_remediation_log_grid
from scripts.fit_data import remediation_pairs_for_shard


def _triplets():
    return make_remediation_log_grid().triplets_and_weights()[0]


def test_369_production_shards_are_one_group_each_in_monotonic_rho_order() -> None:
    triplets = _triplets()
    assigned = [
        remediation_pairs_for_shard(
            triplets, shard_id=shard_id, n_shards=369, scheme="rho_monotonic",
        )
        for shard_id in range(369)
    ]
    assert all(len(groups) == 1 for groups in assigned)
    pairs = [groups[0] for groups in assigned]
    assert pairs == sorted(pairs, key=lambda pair: (pair[0], pair[1]))
    assert len(set(pairs)) == 369


def test_smaller_rho_shard_count_uses_complete_nonduplicating_snake_assignment() -> None:
    triplets = _triplets()
    assigned = [
        pair
        for shard_id in range(128)
        for pair in remediation_pairs_for_shard(
            triplets, shard_id=shard_id, n_shards=128, scheme="rho_monotonic",
        )
    ]
    assert len(assigned) == 369
    assert len(set(assigned)) == 369
