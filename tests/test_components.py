from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from memory_harness.components import AdjacentMergeStore
from memory_harness.components import BoundaryChunkRetriever
from memory_harness.components import CausalKinematicPeakWrite
from memory_harness.components import ContentRecencyRetriever
from memory_harness.components import CompletedPhaseMeanRetriever
from memory_harness.components import DHEMEventStore
from memory_harness.components import Mem0ContextUtilizer
from memory_harness.components import NoveltyWrite
from memory_harness.components import PhaseLifecycle
from memory_harness.components import RingStore
from memory_harness.components import SemanticRecentUnionRetriever
from memory_harness.components import TieredChunkMeanStore
from memory_harness.components import TemporalMultiscaleRetriever
from memory_harness.components import TokenEncoder
from memory_harness.components import TokenUtilizer
from memory_harness.components import UniformGlobalRetriever
from memory_harness.contracts import MemoryStep
from memory_harness.contracts import WriteDecision


def _item(step: int, *, phase: str = "a"):
    return TokenEncoder(max_tokens=2).encode(
        MemoryStep(
            "episode",
            step,
            phase=phase,
            source_tokens=np.full((3, 2), step, dtype=np.float32),
            source_mask=np.ones(3, dtype=np.bool_),
        ),
        path_name="test",
    )


def test_ring_store_evicts_oldest_item() -> None:
    store = RingStore(capacity=2)
    for index in range(3):
        store.write(_item(index))
    assert [item.step_index for item in store.items()] == [1, 2]


def test_content_recency_retriever_balances_similarity_and_frame_gap() -> None:
    store = RingStore(capacity=4)
    encoder = TokenEncoder(max_tokens=1)
    for index, vector in ((0, [1.0, 0.0]), (8, [0.0, 1.0]), (9, [0.8, 0.6])):
        store.write(
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([vector], dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="content",
            )
        )
    query = MemoryStep(
        "episode",
        10,
        source_tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    semantic = ContentRecencyRetriever(max_items=1, recency_penalty=0.01)
    recent = ContentRecencyRetriever(max_items=1, recency_penalty=0.03)

    semantic_result = semantic.retrieve(query, store)
    recent_result = recent.retrieve(query, store)
    assert semantic_result.items[0].step_index == 0
    assert recent_result.items[0].step_index == 9
    assert semantic_result.details["selected"][0]["frame_gap"] == 10


def test_completed_phase_retriever_pools_contiguous_completed_segments() -> None:
    store = RingStore(capacity=8)
    for index, (phase, value) in enumerate(
        (("a", 1.0), ("a", 3.0), ("b", 10.0), ("b", 14.0))
    ):
        store.write(
            TokenEncoder(max_tokens=1).encode(
                MemoryStep(
                    "episode",
                    index,
                    phase=phase,
                    source_tokens=np.full((1, 2), value, dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="handoff",
            )
        )

    result = CompletedPhaseMeanRetriever(max_items=2).retrieve(
        MemoryStep("episode", 4, phase="c"), store
    )

    assert [item.phase for item in result.items] == ["a", "b"]
    np.testing.assert_array_equal(result.items[0].tokens, [[2.0, 2.0]])
    np.testing.assert_array_equal(result.items[1].tokens, [[12.0, 12.0]])
    assert result.details["completed_segment_count"] == 2
    assert result.details["active_segment_excluded"] is False
    assert result.items[1].metadata["handoff_to_phase"] == "c"


def test_completed_phase_retriever_separates_repeated_labels_and_excludes_active() -> (
    None
):
    store = RingStore(capacity=8)
    for index, phase in enumerate(("a", "b", "a")):
        store.write(_item(index, phase=phase))

    result = CompletedPhaseMeanRetriever(max_items=3).retrieve(
        MemoryStep("episode", 3, phase="a"), store
    )

    assert [item.phase for item in result.items] == ["a", "b"]
    assert [item.step_index for item in result.items] == [0, 1]
    assert result.details["segment_count"] == 3
    assert result.details["active_segment_excluded"] is True


def test_completed_phase_retriever_requires_deployment_phase_labels() -> None:
    store = RingStore(capacity=2)
    with pytest.raises(ValueError, match="non-empty deployment phase labels"):
        CompletedPhaseMeanRetriever(max_items=1).retrieve(
            MemoryStep("episode", 0), store
        )


def test_boundary_chunk_retriever_selects_and_uniformly_samples_coherent_chunk() -> (
    None
):
    store = RingStore(capacity=8)
    encoder = TokenEncoder(max_tokens=1)
    vectors = (
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, 0.02],
        [0.0, 1.0],
        [0.01, 0.99],
        [0.02, 0.98],
    )
    for index, vector in enumerate(vectors):
        store.write(
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([vector], dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="chunks",
            )
        )
    query = MemoryStep(
        "episode",
        6,
        source_tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    result = BoundaryChunkRetriever(
        max_items=2,
        boundary_similarity_threshold=0.5,
        min_chunk_items=2,
    ).retrieve(query, store)

    assert [item.step_index for item in result.items] == [0, 2]
    assert result.details["boundary_count"] == 1
    assert result.details["chunk_count"] == 2
    assert result.details["selected_chunk"]["item_count"] == 3
    assert result.details["sampling"] == "uniform_in_chunk"


def test_boundary_chunk_retriever_prefers_recent_chunk_on_score_tie() -> None:
    store = RingStore(capacity=3)
    encoder = TokenEncoder(max_tokens=1)
    for index, vector in enumerate(([1.0, 0.0], [0.0, 1.0], [1.0, 0.0])):
        store.write(
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([vector], dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="chunks",
            )
        )
    query = MemoryStep(
        "episode",
        3,
        source_tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    result = BoundaryChunkRetriever(
        max_items=2,
        boundary_similarity_threshold=1.0,
        min_chunk_items=1,
    ).retrieve(query, store)

    assert [item.step_index for item in result.items] == [2]
    assert result.details["chunk_count"] == 3


def test_boundary_chunk_retriever_reset_clears_embedding_cache() -> None:
    store = RingStore(capacity=2)
    store.write(_item(1))
    retriever = BoundaryChunkRetriever(
        max_items=1,
        boundary_similarity_threshold=0.5,
    )
    query = MemoryStep(
        "episode",
        2,
        source_tokens=np.ones((1, 2), dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )
    retriever.retrieve(query, store)
    assert retriever._cached_item_ids

    retriever.reset()

    assert retriever._cached_item_ids == ()
    assert retriever._key_buffer is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_items": 0, "boundary_similarity_threshold": 0.5},
        {"max_items": 2, "boundary_similarity_threshold": 1.1},
        {
            "max_items": 2,
            "boundary_similarity_threshold": 0.5,
            "min_chunk_items": 0,
        },
    ],
)
def test_boundary_chunk_retriever_rejects_invalid_options(kwargs) -> None:
    with pytest.raises(ValueError):
        BoundaryChunkRetriever(**kwargs)


def test_content_recency_retriever_returns_selected_items_in_time_order() -> None:
    store = RingStore(capacity=4)
    encoder = TokenEncoder(max_tokens=1)
    for index, vector in ((0, [1.0, 0.0]), (1, [0.0, 1.0]), (2, [0.9, 0.1])):
        store.write(
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([vector], dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="content",
            )
        )
    query = MemoryStep(
        "episode",
        3,
        source_tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    result = ContentRecencyRetriever(max_items=2, recency_penalty=0).retrieve(
        query, store
    )

    assert [item.step_index for item in result.items] == [0, 2]
    assert result.details["candidate_count"] == 3


def test_content_recency_retriever_falls_back_to_recency_for_zero_query() -> None:
    store = RingStore(capacity=3)
    for index in range(2):
        store.write(_item(index))
    query = MemoryStep(
        "episode",
        2,
        source_tokens=np.zeros((1, 2), dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    result = ContentRecencyRetriever(max_items=1, recency_penalty=0.1).retrieve(
        query, store
    )

    assert result.items[0].step_index == 1


def test_semantic_recent_union_preserves_old_match_and_recent_tail() -> None:
    store = RingStore(capacity=6)
    encoder = TokenEncoder(max_tokens=1)
    vectors = (
        (0, [1.0, 0.0]),
        (1, [0.0, 1.0]),
        (2, [0.0, 1.0]),
        (3, [0.2, 0.8]),
        (4, [0.0, 1.0]),
    )
    for index, vector in vectors:
        store.write(
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([vector], dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="union",
            )
        )
    query = MemoryStep(
        "episode",
        5,
        source_tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    result = SemanticRecentUnionRetriever(semantic_items=1, recent_items=2).retrieve(
        query, store
    )

    assert [item.step_index for item in result.items] == [0, 3, 4]
    assert result.details["selected"][0]["selected_by_semantic"] is True
    assert result.details["selected"][0]["selected_by_recent"] is False
    assert result.details["selected"][-1]["selected_by_recent"] is True


def test_semantic_recent_union_backfills_overlap_to_match_budget() -> None:
    store = RingStore(capacity=3)
    encoder = TokenEncoder(max_tokens=1)
    for index, vector in ((0, [0.0, 1.0]), (1, [1.0, 0.0])):
        store.write(
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([vector], dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="union",
            )
        )
    query = MemoryStep(
        "episode",
        2,
        source_tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    result = SemanticRecentUnionRetriever(semantic_items=1, recent_items=1).retrieve(
        query, store
    )

    assert [item.step_index for item in result.items] == [0, 1]
    assert result.details["initial_branch_overlap_count"] == 1
    assert result.details["selected"][0]["selected_by_semantic"] is True
    assert result.details["selected"][0]["selected_by_recent"] is False
    assert result.details["selected"][1]["selected_by_semantic"] is False
    assert result.details["selected"][1]["selected_by_recent"] is True


def test_semantic_recent_union_uses_full_budget_and_guarantees_recent_tail() -> None:
    store = RingStore(capacity=40)
    encoder = TokenEncoder(max_tokens=1)
    for index in range(40):
        angle = index / 40 * np.pi
        vector = [float(np.cos(angle)), float(np.sin(angle))]
        store.write(
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([vector], dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="union",
            )
        )
    query = MemoryStep(
        "episode",
        40,
        source_tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    result = SemanticRecentUnionRetriever(semantic_items=20, recent_items=10).retrieve(
        query, store
    )

    assert len(result.items) == 30
    assert result.details["selected_count"] == 30
    assert set(range(30, 40)).issubset({item.step_index for item in result.items})


def test_temporal_multiscale_retriever_is_budget_matched_and_multiscale() -> None:
    store = RingStore(capacity=100)
    for index in range(64):
        store.write(_item(index))
    query = MemoryStep(
        "episode",
        64,
        source_tokens=np.ones((1, 2), dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    result = TemporalMultiscaleRetriever(max_items=30, exponential_items=15).retrieve(
        query, store
    )

    selected_steps = {item.step_index for item in result.items}
    assert len(result.items) == 30
    assert {63, 62, 60, 56, 48, 32}.issubset(selected_steps)
    assert 0 in selected_steps
    assert [item.step_index for item in result.items] == sorted(selected_steps)
    assert result.details["selected_count"] == 30
    assert {row["selected_by"] for row in result.details["selected"]} == {
        "exponential",
        "global_uniform",
    }


def test_uniform_global_retriever_is_budget_matched_and_covers_history() -> None:
    store = RingStore(capacity=100)
    for index in range(64):
        store.write(_item(index))
    query = MemoryStep(
        "episode",
        64,
        source_tokens=np.ones((1, 2), dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )

    result = UniformGlobalRetriever(max_items=30).retrieve(query, store)

    selected_steps = [item.step_index for item in result.items]
    assert len(selected_steps) == 30
    assert selected_steps == sorted(selected_steps)
    assert selected_steps[0] == 0
    assert selected_steps[-1] == 63
    assert result.details["strategy"] == "uniform_global"
    assert all(
        row["selected_by"] == "global_uniform" for row in result.details["selected"]
    )


def test_uniform_global_retriever_returns_all_during_warmup() -> None:
    store = RingStore(capacity=10)
    for index in range(3):
        store.write(_item(index))

    result = UniformGlobalRetriever(max_items=4).retrieve(
        MemoryStep("episode", 3), store
    )

    assert [item.step_index for item in result.items] == [0, 1, 2]
    assert all(row["selected_by"] == "warmup_all" for row in result.details["selected"])


def test_uniform_global_can_reserve_a_disjoint_recent_tail() -> None:
    store = RingStore(capacity=20)
    for index in range(10):
        store.write(_item(index))

    result = UniformGlobalRetriever(max_items=4, exclude_recent_items=4).retrieve(
        MemoryStep("episode", 10), store
    )

    assert [item.step_index for item in result.items] == [0, 1, 3, 5]
    assert result.details["excluded_recent_item_count"] == 4
    assert all(item.step_index < 6 for item in result.items)


def test_uniform_global_retriever_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="uniform global"):
        UniformGlobalRetriever(max_items=0)
    with pytest.raises(ValueError, match="exclude_recent_items"):
        UniformGlobalRetriever(max_items=4, exclude_recent_items=-1)


def test_temporal_multiscale_retriever_returns_all_during_warmup() -> None:
    store = RingStore(capacity=10)
    for index in range(3):
        store.write(_item(index))

    result = TemporalMultiscaleRetriever(max_items=4, exponential_items=2).retrieve(
        MemoryStep("episode", 3), store
    )

    assert [item.step_index for item in result.items] == [0, 1, 2]
    assert all(row["selected_by"] == "warmup_all" for row in result.details["selected"])


@pytest.mark.parametrize(
    ("max_items", "exponential_items"),
    [(0, 1), (4, 0), (4, 4), (4, 5)],
)
def test_temporal_multiscale_retriever_rejects_invalid_budget(
    max_items: int, exponential_items: int
) -> None:
    with pytest.raises(ValueError, match="temporal multiscale"):
        TemporalMultiscaleRetriever(
            max_items=max_items, exponential_items=exponential_items
        )


def test_novelty_writer_skips_redundancy_and_writes_changed_latent() -> None:
    store = RingStore(capacity=3)
    writer = NoveltyWrite(min_cosine_distance=0.1)
    encoder = TokenEncoder(max_tokens=1)

    first_step = MemoryStep(
        "episode",
        0,
        source_tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
        source_mask=np.ones(1, dtype=np.bool_),
    )
    assert writer.decide(first_step, store).write
    store.write(encoder.encode(first_step, path_name="sliding"))

    redundant = dataclasses.replace(first_step, step_index=1)
    redundant_decision = writer.decide(redundant, store)
    assert not redundant_decision.write
    assert redundant_decision.details["reason"] == "redundant"

    changed = dataclasses.replace(
        first_step,
        step_index=2,
        source_tokens=np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    changed_decision = writer.decide(changed, store)
    assert changed_decision.write
    assert changed_decision.details["reason"] == "novel"
    assert changed_decision.details["cosine_distance"] == pytest.approx(1.0)


def test_novelty_writer_forces_bounded_write_interval() -> None:
    store = RingStore(capacity=3)
    writer = NoveltyWrite(min_cosine_distance=2.0, max_steps_without_write=3)
    encoder = TokenEncoder(max_tokens=1)
    tokens = np.asarray([[1.0, 0.0]], dtype=np.float32)
    first = MemoryStep(
        "episode", 0, source_tokens=tokens, source_mask=np.ones(1, dtype=np.bool_)
    )
    store.write(encoder.encode(first, path_name="sliding"))

    decision = writer.decide(dataclasses.replace(first, step_index=3), store)

    assert decision.write
    assert decision.details["reason"] == "max_interval"


def test_causal_kinematic_peak_writer_returns_delayed_candidate_payload() -> None:
    writer = CausalKinematicPeakWrite(
        motion_window=1,
        peak_lookback=1,
        confirmation_delay=1,
        refractory_steps=1,
    )
    store = RingStore(capacity=3)
    decisions = []
    # Displacements 3, 1, 3 yield saliencies .25, .5, .25.  The middle
    # slowdown is confirmed one observable step later.
    states = (0.0, 3.0, 4.0, 7.0)
    for index, state in enumerate(states):
        decisions.append(
            writer.decide(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([[float(index), 0.0]]),
                    source_mask=np.ones(1, dtype=np.bool_),
                    robot_state=np.asarray([state]),
                ),
                store,
            )
        )

    decision = decisions[-1]
    assert decision.write
    assert decision.details["reason"] == "causal_peak"
    assert decision.details["candidate_step_index"] == 2
    assert decision.details["confirmation_step_index"] == 3
    assert decision.write_step is not None
    assert decision.write_step.step_index == 2


def test_causal_kinematic_peak_writer_reset_clears_temporal_state() -> None:
    writer = CausalKinematicPeakWrite(
        motion_window=1,
        peak_lookback=1,
        confirmation_delay=1,
        refractory_steps=1,
    )
    step = MemoryStep(
        "episode",
        0,
        source_tokens=np.ones((1, 2)),
        source_mask=np.ones(1, dtype=np.bool_),
        robot_state=np.asarray([0.0]),
    )
    assert (
        writer.decide(step, RingStore(capacity=2)).details["reason"]
        == "first_robot_state"
    )
    writer.reset()
    assert (
        writer.decide(step, RingStore(capacity=2)).details["reason"]
        == "first_robot_state"
    )


def test_adjacent_merge_store_consolidates_most_similar_neighbors() -> None:
    store = AdjacentMergeStore(capacity=3)
    encoder = TokenEncoder(max_tokens=1)
    vectors = ([1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0])
    details = {}
    for index, vector in enumerate(vectors):
        item = encoder.encode(
            MemoryStep(
                "episode",
                index,
                source_tokens=np.asarray([vector], dtype=np.float32),
                source_mask=np.ones(1, dtype=np.bool_),
            ),
            path_name="consolidating",
        )
        details = store.write(item)

    assert len(store.items()) == 3
    assert details["consolidated"] is True
    assert details["merged_item_ids"] == [
        "episode:consolidating:0",
        "episode:consolidating:1",
    ]
    np.testing.assert_allclose(store.items()[0].tokens, [[0.95, 0.05]])


def test_adjacent_merge_store_reset_clears_items_and_merge_ids() -> None:
    store = AdjacentMergeStore(capacity=2)
    for index in range(3):
        store.write(_item(index))
    assert any(item.item_id.endswith("consolidated:0") for item in store.items())
    store.reset()
    for index in range(3):
        store.write(_item(index))
    assert any(item.item_id.endswith("consolidated:0") for item in store.items())


def test_tiered_chunk_mean_store_preserves_recent_items_and_migrates_oldest() -> None:
    store = TieredChunkMeanStore(
        short_capacity=3,
        migration_chunk_size=2,
        long_capacity=2,
    )
    details = {}
    for index in range(4):
        details = store.write(_item(index))

    items = store.items()
    assert details["maintenance_action"] == "migrate_chunk"
    assert details["migrated_item_ids"] == ["episode:test:0", "episode:test:1"]
    assert details["short_term_count"] == 2
    assert details["long_term_count"] == 1
    assert [item.step_index for item in items] == [1, 2, 3]
    np.testing.assert_allclose(items[0].tokens, np.full((2, 2), 0.5))
    np.testing.assert_array_equal(items[1].tokens, _item(2).tokens)
    assert items[0].metadata["memory_tier"] == "long_term"
    assert items[0].metadata["summary_start_step"] == 0
    assert items[0].metadata["summary_end_step"] == 1
    assert items[-1].metadata["memory_tier"] == "short_term"


def test_tiered_chunk_mean_store_bounds_both_tiers_and_audits_long_merge() -> None:
    store = TieredChunkMeanStore(
        short_capacity=3,
        migration_chunk_size=2,
        long_capacity=2,
    )
    details = {}
    for index in range(8):
        details = store.write(_item(index))

    assert details["maintenance_action"] == "migrate_chunk"
    assert details["long_term_maintenance"]["consolidated"] is True
    assert len(store.items()) == 4
    assert (
        len([item for item in store.items() if "tiered-summary" in item.item_id]) == 1
    )
    assert len([item for item in store.items() if "consolidated" in item.item_id]) == 1
    assert [item.step_index for item in store.items()[-2:]] == [6, 7]
    consolidated = next(
        item for item in store.items() if "consolidated" in item.item_id
    )
    assert consolidated.metadata["source_item_count"] == 4
    assert consolidated.metadata["summary_start_step"] == 0
    assert consolidated.metadata["summary_end_step"] == 3


def test_tiered_chunk_mean_store_reset_restores_summary_ids() -> None:
    store = TieredChunkMeanStore(
        short_capacity=2,
        migration_chunk_size=2,
        long_capacity=2,
    )

    def populate() -> None:
        for index in range(3):
            store.write(_item(index))

    populate()
    assert store.items()[0].item_id.endswith("tiered-summary:0")
    store.reset()
    populate()
    assert store.items()[0].item_id.endswith("tiered-summary:0")


def test_tiered_chunk_mean_store_rejects_incompatible_layout_before_mutation() -> None:
    store = TieredChunkMeanStore(
        short_capacity=2,
        migration_chunk_size=2,
        long_capacity=2,
    )
    store.write(_item(0))
    incompatible = TokenEncoder(max_tokens=1).encode(
        MemoryStep(
            "episode",
            1,
            source_tokens=np.ones((1, 3), dtype=np.float32),
            source_mask=np.ones(1, dtype=np.bool_),
        ),
        path_name="test",
    )

    with pytest.raises(ValueError, match="equal stored layouts"):
        store.write(incompatible)

    assert [item.item_id for item in store.items()] == ["episode:test:0"]


def test_dhem_event_store_discards_redundant_incoming_and_preserves_edges() -> None:
    store = DHEMEventStore(capacity=4, temporal_decay=3)
    encoder = TokenEncoder(max_tokens=1)
    vectors = ([0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0])
    for index, vector in enumerate(vectors):
        store.write(
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([vector], dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="dhem_event",
            )
        )
    before_ids = tuple(item.item_id for item in store.items())
    details = store.write(
        encoder.encode(
            MemoryStep(
                "episode",
                4,
                source_tokens=np.asarray([[0.0, 1.0]], dtype=np.float32),
                source_mask=np.ones(1, dtype=np.bool_),
            ),
            path_name="dhem_event",
        )
    )

    assert details["maintenance_action"] == "discard_incoming"
    assert details["retained"] is False
    assert tuple(item.item_id for item in store.items()) == before_ids


def test_dhem_event_store_uses_accumulated_mass_for_repeated_merges() -> None:
    store = DHEMEventStore(capacity=4, temporal_decay=3)
    encoder = TokenEncoder(max_tokens=1)
    vectors = (
        [0.0, -1.0],
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
    )
    details = {}
    for index, vector in enumerate(vectors):
        details = store.write(
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.asarray([vector], dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name="dhem_event",
            )
        )

    assert details["maintenance_action"] == "merge_history_and_append"
    assert details["accumulated_mass"] == pytest.approx(3.0)
    merged = store.items()[1]
    np.testing.assert_allclose(
        merged.tokens,
        np.asarray([[(1.0 + 0.99) / 3.0, (0.01 + 1.0) / 3.0]]),
    )
    assert merged.metadata["representative_time"] == pytest.approx(2.0)
    assert store.items()[0].step_index == 0
    assert store.items()[-1].step_index == 5


def test_dhem_event_store_reset_clears_items_and_merge_ids() -> None:
    store = DHEMEventStore(capacity=4, temporal_decay=3)
    encoder = TokenEncoder(max_tokens=1)
    vectors = ([0.0, -1.0], [1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [1.0, 0.0])

    def populate() -> None:
        for index, vector in enumerate(vectors):
            store.write(
                encoder.encode(
                    MemoryStep(
                        "episode",
                        index,
                        source_tokens=np.asarray([vector], dtype=np.float32),
                        source_mask=np.ones(1, dtype=np.bool_),
                    ),
                    path_name="dhem_event",
                )
            )

    populate()
    assert any(item.item_id.endswith("dhem-merged:0") for item in store.items())
    store.reset()
    populate()
    assert any(item.item_id.endswith("dhem-merged:0") for item in store.items())


def test_phase_lifecycle_resets_store_on_phase_change() -> None:
    store = RingStore(capacity=2)
    lifecycle = PhaseLifecycle()
    assert not lifecycle.before_step(MemoryStep("episode", 0, phase="a"), store)
    store.write(_item(0))
    assert not lifecycle.before_step(MemoryStep("episode", 1, phase="a"), store)
    assert lifecycle.before_step(MemoryStep("episode", 2, phase="b"), store)
    assert store.items() == ()


def test_token_encoder_honors_mask_and_limit() -> None:
    item = TokenEncoder(max_tokens=2).encode(
        MemoryStep(
            "episode",
            0,
            source_tokens=np.arange(12, dtype=np.float32).reshape(4, 3),
            source_mask=np.array([False, True, True, True]),
        ),
        path_name="test",
    )
    np.testing.assert_array_equal(
        item.tokens, np.arange(12, dtype=np.float32).reshape(4, 3)[1:3]
    )


def test_memory_step_validates_robot_state_without_source_tokens() -> None:
    step = MemoryStep("episode", 0, robot_state=[1.0, 2.0])
    np.testing.assert_array_equal(step.robot_state, [1.0, 2.0])
    with pytest.raises(ValueError, match="rank-1"):
        MemoryStep("episode", 0, robot_state=[[1.0, 2.0]])


def test_skipped_write_cannot_carry_a_delayed_payload() -> None:
    with pytest.raises(ValueError, match="skipped write"):
        WriteDecision(False, write_step=MemoryStep("episode", 0))


def test_token_utilizer_rejects_embedding_width_mismatch() -> None:
    first = _item(0)
    second = TokenEncoder(max_tokens=2).encode(
        MemoryStep(
            "episode",
            1,
            source_tokens=np.ones((2, 3), dtype=np.float32),
            source_mask=np.ones(2, dtype=np.bool_),
        ),
        path_name="other",
    )
    with pytest.raises(ValueError, match="embedding width"):
        TokenUtilizer(token_budget=4).apply({}, [first, second])


def test_mem0_context_uses_fixed_anchor_and_right_aligned_sliding_slots() -> None:
    encoder = TokenEncoder(max_tokens=1)
    anchor = encoder.encode(
        MemoryStep(
            "episode",
            0,
            source_tokens=np.full((1, 3), 10.0),
            source_mask=np.ones(1, dtype=np.bool_),
        ),
        path_name="anchor",
    )
    sliding = [
        encoder.encode(
            MemoryStep(
                "episode",
                index,
                source_tokens=np.full((1, 3), float(index)),
                source_mask=np.ones(1, dtype=np.bool_),
            ),
            path_name="sliding",
        )
        for index in (1, 2)
    ]

    result = Mem0ContextUtilizer(
        embed_dim=3,
        sliding_window_size=4,
        anchor_path="anchor",
        history_path_quotas={"sliding": 4},
    ).apply({}, [anchor, *sliding])

    output = result.observation
    assert result.used_token_count == 3
    np.testing.assert_array_equal(
        output["memory_mask"], [True, False, False, True, True]
    )
    np.testing.assert_array_equal(output["memory_tokens"][0], np.full(3, 10.0))
    np.testing.assert_array_equal(output["memory_tokens"][3], np.full(3, 1.0))
    np.testing.assert_array_equal(output["memory_tokens"][4], np.full(3, 2.0))


def test_mem0_context_emits_static_empty_layout_before_first_write() -> None:
    result = Mem0ContextUtilizer(
        embed_dim=3,
        sliding_window_size=4,
        anchor_path=None,
        history_path_quotas={},
    ).apply({}, [])
    output = result.observation
    assert result.used_token_count == 0
    assert output["memory_tokens"].shape == (5, 3)
    assert not output["memory_mask"].any()


def test_mem0_context_accepts_consolidated_history_path() -> None:
    item = TokenEncoder(max_tokens=1).encode(
        MemoryStep(
            "episode",
            0,
            source_tokens=np.ones((1, 3), dtype=np.float32),
            source_mask=np.ones(1, dtype=np.bool_),
        ),
        path_name="consolidating",
    )
    result = Mem0ContextUtilizer(
        embed_dim=3,
        sliding_window_size=4,
        anchor_path=None,
        history_path_quotas={"consolidating": 4},
    ).apply({}, [item])
    output = result.observation
    assert result.used_token_count == 1
    assert output["memory_mask"][-1]


def test_mem0_context_enforces_each_history_path_quota_and_audits_drops() -> None:
    encoder = TokenEncoder(max_tokens=1)

    def items(path_name: str) -> list:
        return [
            encoder.encode(
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=np.full((1, 3), index, dtype=np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
                path_name=path_name,
            )
            for index in range(3)
        ]

    first = items("first")
    second = items("second")
    result = Mem0ContextUtilizer(
        embed_dim=3,
        sliding_window_size=4,
        anchor_path=None,
        history_path_quotas={"first": 2, "second": 2},
    ).apply({}, [*first, *second])

    assert result.used_token_count == 4
    assert [item.item_id for item in result.used_items] == [
        "episode:first:1",
        "episode:first:2",
        "episode:second:1",
        "episode:second:2",
    ]
    assert result.details["dropped_item_ids"] == [
        "episode:first:0",
        "episode:second:0",
    ]
    assert result.details["path_usage"]["first"] == {
        "quota": 2,
        "retrieved_item_count": 3,
        "used_item_count": 2,
        "dropped_item_count": 1,
    }


def test_mem0_context_rejects_partial_or_invalid_path_allocations() -> None:
    with pytest.raises(ValueError, match="allocate the complete"):
        Mem0ContextUtilizer(
            embed_dim=3,
            sliding_window_size=4,
            anchor_path=None,
            history_path_quotas={"first": 2},
        )
    with pytest.raises(ValueError, match="positive integers"):
        Mem0ContextUtilizer(
            embed_dim=3,
            sliding_window_size=4,
            anchor_path=None,
            history_path_quotas={"first": True, "second": 3},
        )
