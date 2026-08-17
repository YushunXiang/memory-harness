from __future__ import annotations

import numpy as np

from memory_harness.components import BoundaryChunkRetriever
from memory_harness.components import RingStore
from memory_harness.contracts import MemoryItem
from memory_harness.contracts import MemoryStep
from memory_harness.profile_boundary_chunk import _select_indices
from memory_harness.profile_boundary_chunk import profile_sequences


def test_profile_boundary_chunk_reports_coherent_non_sliding_selection() -> None:
    first = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (20, 1))
    second = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (20, 1))
    third = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (20, 1))
    report = profile_sequences(
        {"episode": np.concatenate([first, second, third])},
        max_items=10,
        boundary_similarity_threshold=0.5,
        min_chunk_items=4,
    )

    assert report["full_history_query_count"] == 49
    assert report["mean_selected_item_count"] <= 10
    assert 0 < report["mean_selected_span_density"] <= 1.0
    assert report["exact_sliding_query_fraction"] < 1.0
    assert report["mean_outside_latest_window_fraction"] > 0
    assert report["minimum_chunk_fallback_fraction"] == 0.0


def test_profile_selection_matches_runtime_retriever() -> None:
    first = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (30, 1))
    second = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (30, 1))
    history = np.concatenate([first, second])
    query = np.asarray([[0.0, 1.0]], dtype=np.float32)
    mask = np.ones((1,), dtype=np.bool_)
    store = RingStore(capacity=100)
    for index, vector in enumerate(history):
        store.write(
            MemoryItem(
                item_id=f"episode:path:{index}",
                path_name="path",
                episode_id="episode",
                step_index=index,
                phase="",
                tokens=vector[None, :],
                mask=mask,
            )
        )

    runtime = BoundaryChunkRetriever(
        max_items=10,
        boundary_similarity_threshold=0.5,
        min_chunk_items=10,
    ).retrieve(
        MemoryStep(
            episode_id="episode",
            step_index=60,
            source_tokens=query,
            source_mask=mask,
        ),
        store,
    )
    normalized = history / np.linalg.norm(history, axis=1, keepdims=True)
    selected, _ = _select_indices(
        query_similarities=normalized @ query[0],
        adjacent_similarities=np.sum(normalized[:-1] * normalized[1:], axis=1),
        max_items=10,
        boundary_similarity_threshold=0.5,
        min_chunk_items=10,
    )

    runtime_indices = [int(item.item_id.rsplit(":", 1)[1]) for item in runtime.items]
    assert runtime_indices == selected.tolist()
