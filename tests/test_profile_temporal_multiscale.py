from __future__ import annotations

import numpy as np

from memory_harness.profile_temporal_multiscale import profile_sequences


def test_profile_uses_actual_multiscale_retriever_and_spans_full_history() -> None:
    report = profile_sequences(
        {"episode": np.arange(100 * 3, dtype=np.float32).reshape(100, 3)},
        max_items=30,
        exponential_items=15,
    )

    assert report["num_episodes"] == 1
    assert report["full_history_query_count"] == 69
    temporal = report["profiles"]["temporal_multiscale"]
    uniform = report["profiles"]["uniform_global"]
    recent_global = report["profiles"]["recent_global"]
    assert temporal["mean_selected_item_count"] == 30
    assert temporal["mean_outside_latest_window_fraction"] > 0
    assert temporal["exact_sliding_query_fraction"] == 0
    assert temporal["oldest_selected_lag_quantiles"]["q100"] == 99
    assert uniform["mean_selected_item_count"] == 30
    assert uniform["oldest_selected_lag_quantiles"]["q100"] == 99
    assert recent_global["mean_selected_item_count"] == 30
    assert recent_global["mean_selected_by_count"] == {
        "global_uniform": 15.0,
        "recent": 15.0,
    }
    assert report["pairwise"]["temporal_multiscale_vs_uniform_global"][
        "exact_selected_set_query_fraction"
    ] < 1
    assert report["pairwise"]["recent_global_vs_uniform_global"][
        "exact_selected_set_query_fraction"
    ] == 0
