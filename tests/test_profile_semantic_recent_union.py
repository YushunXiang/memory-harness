from __future__ import annotations

import numpy as np

from memory_harness.profile_semantic_recent_union import profile_sequences


def test_profile_semantic_recent_union_reports_hard_recent_coverage() -> None:
    sequence = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    profile = profile_sequences(
        {"episode": sequence}, semantic_items=1, recent_items=1
    )

    assert profile["full_bank_query_count"] == 4
    assert profile["mean_selected_item_count"] == 2
    assert profile["mean_branch_overlap_count"] > 0
    assert profile["mean_outside_latest_window_fraction"] > 0
    assert profile["selected_lag_quantiles"]["q100"] >= 3
