from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from memory_harness.audit import JsonlAuditSink
from memory_harness.config import ProgramSpec, load_program_spec
from memory_harness.config import ComponentSpec
from memory_harness.contracts import MemoryStep
from memory_harness.contracts import EpisodeOutcome
from memory_harness.contracts import WriteDecision
from memory_harness.policy import MemoryHarnessPolicy, ObservationFieldTokenSource
from memory_harness.registry import build_program


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _tokens(
    value: float, *, count: int = 4, width: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((count, width), value, dtype=np.float32),
        np.ones((count,), dtype=np.bool_),
    )


def _step(
    program,
    index: int,
    value: float,
    *,
    phase: str = "",
    episode_id: str = "episode",
):
    tokens, mask = _tokens(value)
    return program.step(
        {"state": np.array([index], dtype=np.float32)},
        MemoryStep(
            episode_id, index, phase=phase, source_tokens=tokens, source_mask=mask
        ),
    )


@pytest.mark.parametrize(
    "name",
    [
        "none",
        "anchor",
        "sliding",
        "anchor_sliding",
        "consolidating",
        "novelty_sliding",
        "dhem_event",
        "kinematic_event",
        "content_recency",
        "boundary_chunk",
        "semantic_recent_union",
        "temporal_multiscale",
        "tiered_chunk_mean",
        "uniform_global",
        "recent_global",
        "verified_success_latent",
        "completed_phase_handoff",
    ],
)
def test_all_fixed_programs_validate_and_build(name: str) -> None:
    program = build_program(load_program_spec(CONFIGS / f"fixed_{name}.json"))
    assert program.name == name


def test_boundary_chunk_config_matches_frozen_train_calibration() -> None:
    config = json.loads((CONFIGS / "fixed_boundary_chunk.json").read_text())
    calibration = json.loads(
        (CONFIGS / "calibrations" / "put_back_boundary_chunk.json").read_text()
    )

    retriever_options = config["paths"][0]["retriever"]["options"]
    assert calibration["selection_split"] == "train"
    assert calibration["selection_uses_rollout_outcomes"] is False
    assert (
        retriever_options["boundary_similarity_threshold"]
        == calibration["selected_threshold"]
    )
    assert retriever_options["max_items"] == calibration["max_items"]
    assert retriever_options["min_chunk_items"] == calibration["min_chunk_items"]


def test_mem0_utilizer_rejects_unmapped_or_unknown_path_roles() -> None:
    spec = load_program_spec(CONFIGS / "fixed_sliding.json")
    with pytest.raises(ValueError, match="missing_roles=.*sliding"):
        build_program(
            dataclasses.replace(
                spec,
                utilizer=ComponentSpec(
                    "mem0_context",
                    {
                        "embed_dim": 2048,
                        "sliding_window_size": 30,
                        "anchor_path": None,
                        "history_path_quotas": {},
                    },
                ),
            )
        )
    with pytest.raises(ValueError, match="unknown_roles=.*ghost"):
        build_program(
            dataclasses.replace(
                spec,
                utilizer=ComponentSpec(
                    "mem0_context",
                    {
                        "embed_dim": 2048,
                        "sliding_window_size": 30,
                        "anchor_path": None,
                        "history_path_quotas": {"sliding": 15, "ghost": 15},
                    },
                ),
            )
        )


def test_none_is_exact_identity_and_has_no_memory_events() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_none.json"))
    observation = {"state": np.array([1.0]), "prompt": "task"}
    program.reset(episode_id="episode")
    result = program.step(observation, MemoryStep("episode", 0))

    assert result.observation is observation
    assert result.used_token_count == 0
    assert result.stored_item_count == 0
    assert [event.event for event in result.events] == ["SELECT", "USE"]


def test_anchor_retrieves_first_step_only_and_episode_reset_clears_it() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_anchor.json"))
    program.reset(episode_id="episode")
    first = _step(program, 0, 1.0)
    second = _step(program, 1, 2.0)

    assert first.used_token_count == 0
    assert second.observation["memory_tokens"].shape == (31, 2048)
    np.testing.assert_array_equal(
        second.observation["memory_tokens"][0, :3], np.ones(3)
    )
    assert second.observation["memory_mask"].sum() == 1
    assert second.retrieved_item_ids == ("episode:anchor:0",)
    assert second.stored_item_count == 1

    program.reset(episode_id="next")
    tokens, mask = _tokens(9.0)
    after_reset = program.step(
        {"state": np.array([0.0])},
        MemoryStep("next", 0, source_tokens=tokens, source_mask=mask),
    )
    assert after_reset.used_token_count == 0
    assert after_reset.stored_item_count == 1


def test_sliding_keeps_capacity_and_retrieval_order() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_sliding.json"))
    program.reset(episode_id="episode")
    results = [_step(program, index, float(index)) for index in range(6)]

    assert results[-1].stored_item_count == 6
    assert results[-1].retrieved_item_ids == tuple(
        f"episode:sliding:{index}" for index in range(5)
    )
    expected = np.asarray([float(index) for index in range(5)], dtype=np.float32)
    np.testing.assert_array_equal(
        results[-1].observation["memory_tokens"][-5:, 0], expected
    )
    assert results[-1].used_token_count == 5
    use = next(event for event in results[-1].events if event.event == "USE")
    write = next(event for event in results[-1].events if event.event == "WRITE")
    assert use.details["stored_item_count_before_write"] == 5
    assert write.details["path_stored_item_count"] == 6
    assert write.details["total_stored_item_count"] == 6


def test_verified_success_store_commits_success_and_discards_failure() -> None:
    program = build_program(
        load_program_spec(CONFIGS / "fixed_verified_success_latent.json")
    )
    program.reset(episode_id="successful")
    _step(program, 0, 1.0, episode_id="successful")
    _step(program, 1, 2.0, episode_id="successful")
    assert not program.paths[0].store.items()

    finalized = program.finish_episode(
        EpisodeOutcome("successful", True, final_step_index=1, total_reward=1.0)
    )
    commit = next(event for event in finalized if event.event == "STORE_FINALIZE")
    assert commit.details["action"] == "commit"
    assert commit.details["committed_item_count"] == 2

    program.reset(episode_id="failed")
    first_failed_step = program.step(
        {},
        MemoryStep(
            "failed",
            0,
            source_tokens=_tokens(3.0)[0],
            source_mask=_tokens(3.0)[1],
        ),
    )
    assert first_failed_step.retrieved_item_ids == (
        "successful:success:0",
        "successful:success:1",
    )
    program.finish_episode(EpisodeOutcome("failed", False, final_step_index=0))

    program.reset(episode_id="after-failure")
    result = program.step(
        {},
        MemoryStep(
            "after-failure",
            0,
            source_tokens=_tokens(4.0)[0],
            source_mask=_tokens(4.0)[1],
        ),
    )
    assert result.retrieved_item_ids == (
        "successful:success:0",
        "successful:success:1",
    )
    assert all(
        item.metadata["verified_success"] for item in program.paths[0].store.items()
    )


def test_verified_success_store_requires_explicit_episode_outcome() -> None:
    program = build_program(
        load_program_spec(CONFIGS / "fixed_verified_success_latent.json")
    )
    program.reset(episode_id="episode")
    _step(program, 0, 1.0)

    with pytest.raises(RuntimeError, match="finish_episode"):
        program.reset(episode_id="next")
    with pytest.raises(ValueError, match="does not match"):
        program.finish_episode(EpisodeOutcome("other", True, final_step_index=0))


def test_anchor_sliding_matches_mem0_step_zero_duplication() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_anchor_sliding.json"))
    program.reset(episode_id="episode")
    first = _step(program, 0, 1.0)
    second = _step(program, 1, 2.0)
    third = _step(program, 2, 3.0)

    assert first.stored_item_count == 2
    assert second.retrieved_item_ids == ("episode:anchor:0", "episode:sliding:0")
    assert third.retrieved_item_ids == (
        "episode:anchor:0",
        "episode:sliding:0",
        "episode:sliding:1",
    )
    assert third.used_token_count == 3
    np.testing.assert_array_equal(third.observation["memory_tokens"][0, :3], np.ones(3))
    np.testing.assert_array_equal(
        third.observation["memory_tokens"][-2:, :3], [np.ones(3), np.full(3, 2.0)]
    )


def test_consolidating_program_keeps_capacity_and_audits_merge() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_consolidating.json"))
    program.reset(episode_id="episode")
    results = [_step(program, index, float(index + 1)) for index in range(31)]

    assert results[-1].stored_item_count == 30
    write_event = next(event for event in results[-1].events if event.event == "WRITE")
    assert write_event.details["consolidated"] is True
    assert len(write_event.details["merged_item_ids"]) == 2
    assert str(write_event.details["result_item_id"]).startswith(
        "episode:consolidating:consolidated:"
    )


def test_tiered_chunk_mean_program_migrates_old_history_and_keeps_recent() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_tiered_chunk_mean.json"))
    program.reset(episode_id="episode")
    results = [_step(program, index, float(index)) for index in range(7)]

    assert results[-1].stored_item_count == 5
    write_event = next(event for event in results[-1].events if event.event == "WRITE")
    assert write_event.details["maintenance_action"] == "migrate_chunk"
    assert write_event.details["migrated_item_ids"] == [
        "episode:tiered_chunk_mean:0",
        "episode:tiered_chunk_mean:1",
        "episode:tiered_chunk_mean:2",
    ]
    assert write_event.details["short_term_count"] == 4
    assert write_event.details["long_term_count"] == 1


def test_temporal_multiscale_program_uses_raw_history_at_multiple_scales() -> None:
    program = build_program(
        load_program_spec(CONFIGS / "fixed_temporal_multiscale.json")
    )
    program.reset(episode_id="episode")
    results = [_step(program, index, float(index)) for index in range(33)]

    final = results[-1]
    assert final.used_token_count == 30
    retrieve = next(event for event in final.events if event.event == "RETRIEVE")
    assert retrieve.details["strategy"] == "temporal_multiscale"
    assert retrieve.details["candidate_count"] == 32
    assert retrieve.details["selected_count"] == 30


def test_completed_phase_handoff_uses_only_finished_contiguous_segments() -> None:
    program = build_program(
        load_program_spec(CONFIGS / "fixed_completed_phase_handoff.json")
    )
    program.reset(episode_id="episode")
    phases = ("a", "a", "b", "b", "a", "a")
    results = [
        _step(program, index, float(index + 1), phase=phase)
        for index, phase in enumerate(phases)
    ]

    first_b = results[2]
    assert first_b.used_token_count == 1
    assert first_b.retrieved_item_ids == (
        "episode:completed_phase_handoff:completed-phase:0-1",
    )
    first_repeated_a = results[4]
    assert first_repeated_a.used_token_count == 1
    assert first_repeated_a.retrieved_item_ids == (
        "episode:completed_phase_handoff:completed-phase:2-3",
    )
    next_a = results[5]
    assert next_a.retrieved_item_ids == first_repeated_a.retrieved_item_ids
    retrieve = next(event for event in next_a.events if event.event == "RETRIEVE")
    assert retrieve.details["active_segment_excluded"] is True


def test_uniform_global_program_spans_complete_raw_history() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_uniform_global.json"))
    program.reset(episode_id="episode")
    results = [_step(program, index, float(index)) for index in range(33)]

    final = results[-1]
    assert final.used_token_count == 30
    retrieve = next(event for event in final.events if event.event == "RETRIEVE")
    assert retrieve.details["strategy"] == "uniform_global"
    assert retrieve.details["candidate_count"] == 32
    assert retrieve.details["selected_count"] == 30


def test_boundary_chunk_program_audits_segmentation_and_selection() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_boundary_chunk.json"))
    program.reset(episode_id="episode")
    for index in range(30):
        _step(program, index, 1.0)
    for index in range(30, 60):
        _step(program, index, -1.0)
    final = _step(program, 60, -1.0)

    retrieve = next(event for event in final.events if event.event == "RETRIEVE")
    assert retrieve.details["strategy"] == "boundary_chunk"
    assert retrieve.details["boundary_count"] == 1
    assert retrieve.details["selected_chunk"]["start_step_index"] == 30
    assert final.retrieved_item_ids == tuple(
        f"episode:boundary_chunk:{index}" for index in range(30, 60)
    )


def test_recent_global_composition_uses_both_paths_without_overlap() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_recent_global.json"))
    program.reset(episode_id="episode")
    results = [_step(program, index, float(index)) for index in range(35)]

    final = results[-1]
    assert final.used_token_count == 30
    use = next(event for event in final.events if event.event == "USE")
    assert len(use.item_ids) == 30
    source_steps = [int(item_id.rsplit(":", 1)[1]) for item_id in use.item_ids]
    assert len(source_steps) == len(set(source_steps))
    assert use.details["path_usage"]["recent"]["used_item_count"] == 15
    assert use.details["path_usage"]["global"]["used_item_count"] == 15
    assert use.details["dropped_item_ids"] == []


def test_novelty_sliding_audits_skipped_writes() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_novelty_sliding.json"))
    program.reset(episode_id="episode")
    first = _step(program, 0, 1.0)
    second = _step(program, 1, 1.0)

    assert first.stored_item_count == 1
    assert second.stored_item_count == 1
    decision = next(event for event in second.events if event.event == "WRITE_DECISION")
    assert decision.details["write"] is False
    assert decision.details["reason"] == "redundant"


def test_dhem_event_program_keeps_fixed_anchor_and_bounded_history() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_dhem_event.json"))
    program.reset(episode_id="episode")
    results = [_step(program, index, float(index + 1)) for index in range(31)]

    assert results[-1].stored_item_count == 30
    assert results[-1].retrieved_item_ids[0] == "episode:dhem_event:0"
    write_event = next(event for event in results[-1].events if event.event == "WRITE")
    assert write_event.details["maintenance_action"] in {
        "discard_incoming",
        "merge_history_and_append",
    }


def test_kinematic_event_program_builds_as_typed_anchor_event_composition() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_kinematic_event.json"))

    assert tuple(path.name for path in program.paths) == ("anchor", "event")
    program.reset(episode_id="episode")
    tokens, mask = _tokens(1.0)
    result = program.step(
        {"state": np.asarray([0.0], dtype=np.float32)},
        MemoryStep(
            "episode",
            0,
            source_tokens=tokens,
            source_mask=mask,
            robot_state=np.asarray([0.0], dtype=np.float32),
        ),
    )
    assert result.stored_item_count == 1
    assert result.retrieved_item_ids == ()


def test_content_recency_program_searches_beyond_the_latest_window() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_content_recency.json"))
    program.reset(episode_id="episode")
    width = 2048
    vectors = (
        np.pad(np.asarray([1.0, 0.0]), (0, width - 2)),
        np.pad(np.asarray([0.0, 1.0]), (0, width - 2)),
        np.pad(np.asarray([0.8, 0.6]), (0, width - 2)),
    )
    results = []
    for index, vector in enumerate(vectors):
        results.append(
            program.step(
                {"state": np.asarray([index], dtype=np.float32)},
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=vector[None].astype(np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
            )
        )
    assert results[-1].retrieved_item_ids == (
        "episode:content_recency:0",
        "episode:content_recency:1",
    )
    retrieve = next(event for event in results[-1].events if event.event == "RETRIEVE")
    assert retrieve.details["strategy"] == "content_recency"
    assert retrieve.details["candidate_count"] == 2
    assert len(retrieve.details["selected"]) == 2


def test_semantic_recent_union_program_audits_both_retrieval_branches() -> None:
    program = build_program(
        load_program_spec(CONFIGS / "fixed_semantic_recent_union.json")
    )
    program.reset(episode_id="episode")
    width = 2048
    vectors = (
        np.pad(np.asarray([1.0, 0.0]), (0, width - 2)),
        np.pad(np.asarray([0.0, 1.0]), (0, width - 2)),
        np.pad(np.asarray([0.8, 0.6]), (0, width - 2)),
    )
    results = []
    for index, vector in enumerate(vectors):
        results.append(
            program.step(
                {"state": np.asarray([index], dtype=np.float32)},
                MemoryStep(
                    "episode",
                    index,
                    source_tokens=vector[None].astype(np.float32),
                    source_mask=np.ones(1, dtype=np.bool_),
                ),
            )
        )
    retrieve = next(event for event in results[-1].events if event.event == "RETRIEVE")
    assert retrieve.details["strategy"] == "semantic_recent_union"
    assert retrieve.details["candidate_count"] == 2
    assert all(
        "selected_by_semantic" in row and "selected_by_recent" in row
        for row in retrieve.details["selected"]
    )


def test_inactive_path_lifecycle_observes_skipped_phase_transition() -> None:
    class PhaseSkippingController:
        def select(self, step, path_names):
            return () if step.step_index == 1 else tuple(path_names)

        def reset(self):
            return None

    program = build_program(load_program_spec(CONFIGS / "fixed_sliding.json"))
    program.controller = PhaseSkippingController()
    program.reset(episode_id="episode")
    _step(program, 0, 1.0, phase="a")

    skipped = _step(program, 1, 2.0, phase="b")
    reactivated = _step(program, 2, 3.0, phase="a")

    assert skipped.stored_item_count == 0
    assert any(
        event.event == "RESET" and event.path_name == "sliding"
        for event in skipped.events
    )
    assert reactivated.retrieved_item_ids == ()
    assert reactivated.stored_item_count == 1


def test_delayed_write_commits_the_confirmed_prior_payload() -> None:
    class ConfirmPreviousWrite:
        def __init__(self):
            self.pending = None

        def decide(self, step, store):
            del store
            previous = self.pending
            self.pending = step
            if previous is None:
                return WriteDecision(False, {"reason": "await_confirmation"})
            return WriteDecision(
                True,
                {"reason": "confirmed_prior"},
                write_step=previous,
            )

        def reset(self):
            self.pending = None

    program = build_program(load_program_spec(CONFIGS / "fixed_sliding.json"))
    program.paths[0].writer = ConfirmPreviousWrite()
    program.reset(episode_id="episode")

    first = _step(program, 0, 1.0)
    second = _step(program, 1, 2.0)
    third = _step(program, 2, 3.0)

    assert first.stored_item_count == 0
    write = next(event for event in second.events if event.event == "WRITE")
    assert write.item_ids == ("episode:sliding:0",)
    assert write.details["source_step_index"] == 0
    assert write.details["confirmation_delay_steps"] == 1
    assert third.retrieved_item_ids == ("episode:sliding:0",)


def test_lifecycle_reset_clears_delayed_writer_candidate() -> None:
    class ConfirmPreviousWrite:
        def __init__(self):
            self.pending = None

        def decide(self, step, store):
            del store
            previous = self.pending
            self.pending = step
            if previous is None:
                return WriteDecision(False)
            return WriteDecision(True, write_step=previous)

        def reset(self):
            self.pending = None

    program = build_program(load_program_spec(CONFIGS / "fixed_sliding.json"))
    program.paths[0].writer = ConfirmPreviousWrite()
    program.reset(episode_id="episode")
    _step(program, 0, 1.0, phase="a")
    after_phase_change = _step(program, 1, 2.0, phase="b")

    assert after_phase_change.stored_item_count == 0
    assert not any(event.event == "WRITE" for event in after_phase_change.events)


@pytest.mark.parametrize(
    ("episode_id", "step_index", "metadata", "message"),
    [
        ("other", 0, {}, "active episode"),
        ("episode", 1, {}, "future step"),
        ("episode", 0, {"reward": 1.0}, "undeclared write payload"),
    ],
)
def test_delayed_write_rejects_invalid_payloads(
    episode_id: str,
    step_index: int,
    metadata: dict,
    message: str,
) -> None:
    tokens, mask = _tokens(9.0)
    payload = MemoryStep(
        episode_id,
        step_index,
        source_tokens=tokens,
        source_mask=mask,
        metadata=metadata,
    )

    class PayloadWrite:
        def decide(self, step, store):
            del step, store
            return WriteDecision(True, write_step=payload)

        def reset(self):
            return None

    program = build_program(load_program_spec(CONFIGS / "fixed_sliding.json"))
    program.paths[0].writer = PayloadWrite()
    program.reset(episode_id="episode")
    with pytest.raises(ValueError, match=message):
        _step(program, 0, 1.0)


def test_deployable_program_rejects_undeclared_metadata() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_anchor.json"))
    program.reset(episode_id="episode")
    tokens, mask = _tokens(1.0)
    with pytest.raises(ValueError, match="undeclared metadata.*task_state"):
        program.step(
            {},
            MemoryStep(
                "episode",
                0,
                source_tokens=tokens,
                source_mask=mask,
                metadata={"task_state": 0},
            ),
        )


def test_deployable_program_accepts_only_declared_safe_metadata() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_anchor.json"))
    program.reset(episode_id="episode")
    tokens, mask = _tokens(1.0)
    result = program.step(
        {},
        MemoryStep(
            "episode",
            0,
            source_tokens=tokens,
            source_mask=mask,
            metadata={
                "task_text_present": True,
                "training_representation": "runtime_moment_tokens",
            },
        ),
    )
    assert result.stored_item_count == 1


def test_program_rejects_preinjected_memory_and_nonzero_first_step() -> None:
    program = build_program(load_program_spec(CONFIGS / "fixed_none.json"))
    program.reset(episode_id="episode")
    with pytest.raises(ValueError, match="first step"):
        program.step({}, MemoryStep("episode", 1))

    program.reset(episode_id="episode")
    with pytest.raises(ValueError, match="only owner"):
        program.step({"memory_tokens": np.zeros((1, 2))}, MemoryStep("episode", 0))


def test_config_is_strict() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ProgramSpec.parse(
            {
                "name": "none",
                "deployable": True,
                "paths": [],
                "controller": {"type": "all"},
                "utilizer": {"type": "none"},
                "typo": True,
            }
        )
    spec = ProgramSpec.parse(
        {
            "name": "bad",
            "deployable": True,
            "paths": [],
            "controller": {"type": "all"},
            "utilizer": {"type": "none", "options": {"unexpected": 1}},
        }
    )
    with pytest.raises(ValueError, match="invalid options"):
        build_program(spec)


def test_jsonl_audit_captures_write_retrieve_use_reset(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    program = build_program(
        load_program_spec(CONFIGS / "fixed_anchor.json"),
        audit_sink=JsonlAuditSink(audit_path),
    )
    program.reset(episode_id="episode")
    _step(program, 0, 1.0)
    _step(program, 1, 2.0)

    rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert {row["event"] for row in rows} == {
        "RESET",
        "SELECT",
        "RETRIEVE",
        "USE",
        "WRITE_DECISION",
        "WRITE",
    }
    assert rows[0]["event"] == "RESET"
    assert rows[-1]["event"] == "WRITE_DECISION"


class _RecordingPolicy:
    def __init__(self) -> None:
        self._rng = object()
        self.observations = []
        self.reset_count = 0

    def reset_history(self) -> None:
        self.reset_count += 1

    def infer(self, observation):
        self.observations.append(observation)
        return {"actions": np.array([len(self.observations)])}


def test_policy_wrapper_none_preserves_observation_and_rng() -> None:
    base = _RecordingPolicy()
    wrapper = MemoryHarnessPolicy(
        base,
        build_program(load_program_spec(CONFIGS / "fixed_none.json")),
    )
    observation = {"state": np.array([1.0])}
    original_rng = wrapper._rng
    wrapper.reset_history()
    output = wrapper.infer(observation)

    assert base.reset_count == 1
    assert base.observations == [observation]
    assert base.observations[0] is observation
    assert wrapper._rng is original_rng
    np.testing.assert_array_equal(output["actions"], np.array([1]))


def test_policy_wrapper_none_can_emit_static_masked_model_context() -> None:
    base = _RecordingPolicy()
    wrapper = MemoryHarnessPolicy(
        base,
        build_program(load_program_spec(CONFIGS / "fixed_none.json")),
        empty_context_shape=(5, 3),
    )
    observation = {"state": np.array([1.0])}
    wrapper.reset_history()
    wrapper.infer(observation)

    assert set(observation) == {"state"}
    assert set(base.observations[0]) == {"state", "memory_tokens", "memory_mask"}
    np.testing.assert_array_equal(
        base.observations[0]["memory_tokens"], np.zeros((5, 3))
    )
    assert not base.observations[0]["memory_mask"].any()


def test_policy_wrapper_strips_harness_fields_before_base_policy() -> None:
    base = _RecordingPolicy()
    wrapper = MemoryHarnessPolicy(
        base,
        build_program(load_program_spec(CONFIGS / "fixed_anchor.json")),
        token_source=ObservationFieldTokenSource(),
    )
    wrapper.reset_history()
    tokens, mask = _tokens(1.0)
    wrapper.infer(
        {
            "state": np.array([1.0]),
            "_memory_phase": "visible_phase",
            "_memory_source_tokens": tokens,
            "_memory_source_mask": mask,
        }
    )

    assert set(base.observations[0]) == {"state", "memory_tokens", "memory_mask"}
    assert not base.observations[0]["memory_mask"].any()
