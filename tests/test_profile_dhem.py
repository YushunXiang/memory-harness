from __future__ import annotations

import numpy as np
import pytest

from memory_harness.profile_dhem import profile_sequences


def test_profile_dhem_reports_append_discard_and_merge_behavior() -> None:
    sequence = np.asarray(
        [
            [0.0, -1.0],
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    profile = profile_sequences(
        {"episode": sequence},
        capacity=4,
        temporal_decay=3,
    )

    assert profile["num_sources"] == 7
    assert profile["action_counts"] == {
        "append": 4,
        "discard_incoming": 1,
        "merge_history_and_append": 2,
    }
    assert profile["discard_fraction_at_capacity"] == pytest.approx(1 / 3)
    episode = profile["per_episode"][0]
    assert episode["anchor_step_index"] == 0
    assert episode["final_item_count"] == 4
    assert episode["retained_source_mass"] == pytest.approx(6.0)
    assert episode["discarded_source_mass"] == pytest.approx(1.0)
