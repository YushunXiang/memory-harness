from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from memory_harness.build_training_data import _validate_prompt_preserving_data_factory
from memory_harness.build_training_data import build_chunks
from memory_harness.build_training_data import build_manifest_and_bank
from memory_harness.build_training_data import simulate_program_contexts


def _data_factory(*, prompt_from_task: bool, preserve_prompt: bool):
    structure = {"prompt": "prompt"} if preserve_prompt else {"state": "state"}
    return SimpleNamespace(
        base_config=SimpleNamespace(prompt_from_task=prompt_from_task),
        repack_transforms=SimpleNamespace(
            inputs=(SimpleNamespace(structure=structure),)
        ),
    )


def test_context_encoder_requires_task_prompt_to_survive_repack():
    _validate_prompt_preserving_data_factory(
        _data_factory(prompt_from_task=True, preserve_prompt=True)
    )

    with pytest.raises(ValueError, match="prompt_from_task=True"):
        _validate_prompt_preserving_data_factory(
            _data_factory(prompt_from_task=False, preserve_prompt=True)
        )
    with pytest.raises(ValueError, match="preserve the prompt field"):
        _validate_prompt_preserving_data_factory(
            _data_factory(prompt_from_task=True, preserve_prompt=False)
        )


def _template():
    return {
        "schema_version": "test/v1",
        "train_lerobot_episode_ids": [0, 1],
        "validation_lerobot_episode_ids": [2, 3],
        "segments": [
            {
                "lerobot_episode_index": episode,
                "start_frame": 0,
                "end_frame": 40,
                "phase_label": "cover_left",
                "executor_prompt": "Cover the left block with the left cover.",
                "split": "train" if episode < 2 else "validation",
            }
            for episode in range(4)
        ],
    }


def test_chunks_are_gap_free_and_use_full_memory_program():
    chunks = build_chunks(_template(), stride=10)

    assert [(chunk.start, chunk.end) for chunk in chunks[0]] == [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
    ]
    assert chunks[0][0].item_id("anchor_sliding").endswith(":anchor_sliding")


def test_real_program_simulation_retrieves_before_write():
    chunks = build_chunks(_template(), stride=10)[0]
    moments = tuple(
        np.full((1, 6), fill_value=index + 1, dtype=np.float32)
        for index in range(len(chunks))
    )

    program_name, contexts = simulate_program_contexts(chunks, moments)

    assert program_name == "anchor_sliding"
    _, first_mask = contexts[0]
    assert not first_mask.any()
    combined_tokens, combined_mask = contexts[3]
    assert combined_mask.sum() == 4
    np.testing.assert_array_equal(combined_tokens[0], moments[0][0])
    np.testing.assert_array_equal(combined_tokens[-3:], np.concatenate(moments[:3]))


def test_manifest_uses_same_episode_history_and_disjoint_negative():
    template = _template()
    chunks = build_chunks(template, stride=10)
    contexts = {}
    for episode, episode_chunks in chunks.items():
        moments = tuple(
            np.full((1, 6), fill_value=100 * episode + index, dtype=np.float32)
            for index in range(len(episode_chunks))
        )
        _, contexts[episode] = simulate_program_contexts(episode_chunks, moments)

    manifest, bank, audit = build_manifest_and_bank(
        chunks,
        contexts,
        template=template,
    )

    assert audit["ready_for_adapter_training"] is True
    assert manifest["condition_cycle"] == ["matched"]
    assert bank["tokens"].shape == (16, 31, 6)
    assert bank["tokens"].dtype == np.float16
    assert len(set(bank["item_ids"].tolist())) == 16
    for row in manifest["segments"]:
        assert row["matched_source_episode_id"] == row["source_episode_id"]
        assert row["mismatched_source_episode_id"] != row["source_episode_id"]
        assert row["matched_uses_only_prior_observations"] is True
        assert row["mismatched_source_disjoint"] is True
        assert row["memory_program"] == "anchor_sliding"
        assert row["executor_prompt"] == "Cover the left block with the left cover."


def test_full_memory_resets_at_subtask_phase_change():
    template = _template()
    template["segments"] = [
        {
            "lerobot_episode_index": episode,
            "start_frame": start,
            "end_frame": end,
            "phase_label": phase,
            "executor_prompt": f"Prompt for {phase}.",
            "split": "train" if episode < 2 else "validation",
        }
        for episode in range(4)
        for start, end, phase in (
            (0, 20, "cover_left"),
            (20, 40, "cover_middle"),
        )
    ]
    chunks = build_chunks(template, stride=10)[0]
    moments = tuple(
        np.full((1, 6), fill_value=index + 1, dtype=np.float32)
        for index in range(len(chunks))
    )
    _, contexts = simulate_program_contexts(chunks, moments)

    assert contexts[1][1].sum() == 2
    assert not contexts[2][1].any()
    assert contexts[3][1].sum() == 2


def test_training_context_generator_accepts_consolidating_program() -> None:
    template = _template()
    template["segments"] = [
        {
            "lerobot_episode_index": episode,
            "start_frame": 0,
            "end_frame": 320,
            "phase_label": "cover_left",
            "executor_prompt": "Cover the left block with the left cover.",
            "split": "train" if episode < 2 else "validation",
        }
        for episode in range(4)
    ]
    chunks = build_chunks(template, stride=10)[0]
    moments = tuple(
        np.asarray([[float(index), 1.0]], dtype=np.float32)
        for index in range(len(chunks))
    )
    config = Path(__file__).resolve().parents[1] / "configs/fixed_consolidating.json"

    program_name, contexts = simulate_program_contexts(
        chunks,
        moments,
        program_config=config,
    )

    assert program_name == "consolidating"
    assert contexts[-1][1].sum() == 30


def test_training_context_generator_accepts_dhem_event_program() -> None:
    template = _template()
    template["segments"] = [
        {
            "lerobot_episode_index": episode,
            "start_frame": 0,
            "end_frame": 320,
            "phase_label": "put_back_block",
            "executor_prompt": "Put the block back in its original position.",
            "split": "train" if episode < 2 else "validation",
        }
        for episode in range(4)
    ]
    chunks = build_chunks(template, stride=10)[0]
    moments = tuple(
        np.asarray([[float(index), 1.0]], dtype=np.float32)
        for index in range(len(chunks))
    )
    config = Path(__file__).resolve().parents[1] / "configs/fixed_dhem_event.json"

    program_name, contexts = simulate_program_contexts(
        chunks,
        moments,
        program_config=config,
    )

    assert program_name == "dhem_event"
    assert contexts[-1][1].sum() == 30


def test_empty_mem0_training_program_emits_fixed_zero_context() -> None:
    chunks = build_chunks(_template(), stride=10)[0]
    moments = tuple(np.zeros((1, 6), dtype=np.float32) for _ in chunks)
    config = Path(__file__).resolve().parents[1] / "configs/training_empty_mem0.json"

    program_name, contexts = simulate_program_contexts(
        chunks,
        moments,
        program_config=config,
    )

    assert program_name == "none"
    assert len(contexts) == 4
    for tokens, mask in contexts:
        assert tokens.shape == (31, 6)
        assert not mask.any()
        assert not tokens.any()
