from __future__ import annotations

import numpy as np

from memory_harness.profile_content_recency import profile_sequences


def test_profile_content_recency_reports_non_fifo_selection() -> None:
    sequence = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    profile = profile_sequences(
        {"episode": sequence},
        max_items=2,
        penalties=(0.0, 1.0),
    )

    semantic, recent = profile["profiles"]
    assert semantic["full_bank_query_count"] == 2
    assert semantic["mean_outside_latest_window_fraction"] > 0
    assert recent["mean_outside_latest_window_fraction"] == 0
