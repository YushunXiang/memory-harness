from __future__ import annotations

import numpy as np

from memory_harness.profile_robomme_patch_retrievers import even_sampling_indices
from memory_harness.profile_robomme_patch_retrievers import profile_episode_candidates
from memory_harness.profile_robomme_patch_retrievers import score_token_drop_episode
from memory_harness.profile_robomme_patch_retrievers import select_token_drop_indices


def test_even_sampling_matches_source_endpoints_and_budget() -> None:
    assert even_sampling_indices(3, 8) == [0, 1, 2, 3]
    selected = even_sampling_indices(64, 32)
    assert len(selected) == 32
    assert selected[0] == 0
    assert selected[-1] == 64


def test_token_drop_keeps_first_grid_and_scores_stride_change() -> None:
    images = np.zeros((8, 1, 8, 8, 3), dtype=np.uint8)
    images[7, 0, 0, 0] = 255
    candidates = score_token_drop_episode(images)

    first_frame = [item for item in candidates if item[1] == 0]
    changed = [item for item in candidates if item[1] == 7]
    assert len(first_frame) == 64
    assert {item[0] for item in first_frame} == {1000.0}
    assert len(changed) == 1
    assert changed[0][2:] == (0, 0)
    assert changed[0][0] == 2.0


def test_full_episode_heap_can_let_future_candidates_evict_past_membership() -> None:
    candidates = [(1000.0, 0, 0, patch_index) for patch_index in range(64)]
    candidates.extend(
        (0.1 + patch_index * 1e-7, 7, 0, patch_index)
        for patch_index in range(64)
    )
    candidates.extend(
        (0.9 + item_index * 1e-7, 39 + item_index // 64, 0, item_index % 64)
        for item_index in range(2100)
    )

    causal = set(
        select_token_drop_indices(
            candidates,
            query_step=31,
            token_budget=512,
            kept_size=2048,
            offline_full_episode_heap=False,
        )
    )
    offline = set(
        select_token_drop_indices(
            candidates,
            query_step=31,
            token_budget=512,
            kept_size=2048,
            offline_full_episode_heap=True,
        )
    )

    assert len(causal) == 128
    assert len(offline) == 64
    assert offline < causal
    assert all(step <= 31 for step, _, _ in offline)


def test_profile_reports_selector_distinctness_and_no_returned_future_tokens() -> None:
    candidates = [(1000.0, 0, 0, patch_index) for patch_index in range(64)]
    candidates.extend(
        (float(step + patch_index / 100), step, 0, patch_index)
        for step in range(7, 80, 8)
        for patch_index in range(64)
    )
    report = profile_episode_candidates(candidates, num_steps=80)

    assert report["query_count"] == 48
    assert report["selected_future_token_count"] == 0
    assert report["exact_frame_set_parity_count"] == 0
    assert report["exact_token_parity_count"] == report["query_count"]
