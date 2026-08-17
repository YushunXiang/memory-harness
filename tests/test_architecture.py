from pathlib import Path

import numpy as np
import pytest

from memory_harness.architecture import ArchitectureSpec, build_architecture
from memory_harness.contracts import MemoryStep


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "architectures"


def _image(value: int) -> np.ndarray:
    return np.full((4, 5, 3), value, dtype=np.uint8)


@pytest.mark.parametrize(
    ("name", "modules", "planner_enabled"),
    [
        ("none", (), False),
        ("anchor", ("anchor",), False),
        ("sliding", ("sliding",), False),
        ("anchor_sliding", ("anchor", "sliding"), False),
        ("consolidating", ("consolidating",), False),
        ("dhem_event", ("dhem_event",), False),
        ("content_recency", ("content_recency",), False),
        ("semantic_recent_union", ("semantic_recent_union",), False),
        ("boundary_chunk", ("boundary_chunk",), False),
        ("tiered_chunk_mean", ("tiered_chunk_mean",), False),
        ("temporal_multiscale", ("temporal_multiscale",), False),
        ("uniform_global", ("uniform_global",), False),
        ("recent_global", ("recent", "global"), False),
        ("completed_phase_handoff", ("completed_phase_handoff",), True),
        ("planner_no_key", (), True),
        ("key", ("key",), True),
        (
            "key_completed_phase_handoff",
            ("completed_phase_handoff", "key"),
            True,
        ),
        ("key_anchor_sliding", ("anchor", "sliding", "key"), True),
    ],
)
def test_fixed_architectures_share_one_facade(
    name: str, modules: tuple[str, ...], planner_enabled: bool
) -> None:
    architecture = build_architecture(
        ArchitectureSpec.load(CONFIGS / f"fixed_{name}.json")
    )
    assert architecture.active_modules == modules
    architecture.reset_episode(
        episode_id="episode",
        global_task="cover then uncover",
        initial_image=_image(1),
    )
    planner_context = architecture.planner_context(current_image=_image(1))
    if planner_enabled:
        assert planner_context is not None
        record = architecture.record_completed_subtask(
            instruction="cover left",
            end_image=_image(2),
        )
        expected_records = 1 if "key" in modules else 0
        assert (record is not None) == bool(expected_records)
        updated_context = architecture.planner_context(current_image=_image(3))
        assert updated_context is not None
        if "key" in modules:
            assert len(updated_context.completed_subtasks) == expected_records
        else:
            np.testing.assert_array_equal(updated_context.current_image, _image(3))
    else:
        assert planner_context is None

    source_tokens = np.ones((1, 2048), dtype=np.float32)
    source_mask = np.ones((1,), dtype=np.bool_)
    result = architecture.executor_step(
        {"state": np.zeros((1,), dtype=np.float32)},
        MemoryStep(
            "episode",
            0,
            phase="phase-a",
            source_tokens=source_tokens if architecture.executor.paths else None,
            source_mask=source_mask if architecture.executor.paths else None,
        ),
    )
    assert result.stored_item_count == len(architecture.executor.paths)


def test_kinematic_event_architecture_exposes_composed_executor_paths() -> None:
    architecture = build_architecture(
        ArchitectureSpec.load(CONFIGS / "fixed_kinematic_event.json")
    )
    assert architecture.active_modules == ("anchor", "event")
