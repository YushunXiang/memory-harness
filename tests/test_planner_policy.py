from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from memory_harness.architecture import ArchitectureSpec, build_architecture
from memory_harness.planner_policy import (
    Mem0PlannerPolicy,
    PlannerResult,
    parse_next_subtask,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeExecutorPolicy:
    def __init__(self) -> None:
        self.observations: list[dict] = []
        self.reset_count = 0
        self._rng = object()

    def reset_history(self) -> None:
        self.reset_count += 1

    def infer(self, observation: dict) -> dict:
        self.observations.append(dict(observation))
        return {"actions": np.zeros((1, 2), dtype=np.float32)}


class FakePlanner:
    def __init__(self) -> None:
        self.contexts = []
        self.seeds = []

    def plan(self, context, *, seed: int) -> PlannerResult:
        self.contexts.append(context)
        self.seeds.append(seed)
        stage = len(getattr(context, "completed_subtasks", ()))
        instruction = f"execute stage {stage}."
        return PlannerResult(instruction, f"next_subtask: {instruction}", 0.01)


def _observation(label: str, value: int) -> dict:
    return {
        "images": {"cam_high": np.full((3, 4, 5), value, dtype=np.uint8)},
        "state": np.zeros(2, dtype=np.float32),
        "prompt": label,
    }


def test_key_planner_writes_on_boundary_and_overrides_executor_prompt() -> None:
    architecture = build_architecture(
        ArchitectureSpec.load(ROOT / "configs/architectures/fixed_key.json")
    )
    executor = FakeExecutorPolicy()
    planner = FakePlanner()
    policy = Mem0PlannerPolicy(
        executor,
        architecture=architecture,
        planner_backend=planner,
        global_task="cover then uncover",
        planner_seed_base=900,
        boundary_mode="oracle_prompt_change",
    )

    policy.reset_history()
    first = policy.infer(_observation("oracle stage zero", 1))
    policy.infer(_observation("oracle stage zero", 2))
    transitioned = policy.infer(_observation("oracle stage one", 3))

    assert executor.reset_count == 1
    assert planner.seeds == [900, 900]
    assert [len(context.completed_subtasks) for context in planner.contexts] == [0, 1]
    record = planner.contexts[-1].completed_subtasks[0]
    assert record.instruction == "execute stage 0."
    assert np.all(record.end_image == 3)
    assert executor.observations[0]["prompt"] == "execute stage 0."
    assert executor.observations[-1]["prompt"] == "execute stage 1."
    assert executor.observations[-1]["memory_task_text"] == "cover then uncover"
    assert first["memory"]["planner_call_count"] == 1
    assert transitioned["memory"]["planner_boundary_changed"] is True
    assert transitioned["memory"]["planner_boundary_deployable"] is False


def test_key_planner_resets_episode_history() -> None:
    architecture = build_architecture(
        ArchitectureSpec.load(ROOT / "configs/architectures/fixed_key.json")
    )
    planner = FakePlanner()
    policy = Mem0PlannerPolicy(
        FakeExecutorPolicy(),
        architecture=architecture,
        planner_backend=planner,
        global_task="task",
        planner_seed_base=10,
        boundary_mode="oracle_prompt_change",
    )
    policy.reset_history()
    policy.infer(_observation("zero", 1))
    policy.infer(_observation("one", 2))
    policy.reset_history()
    policy.infer(_observation("zero", 4))

    assert len(planner.contexts[-1].completed_subtasks) == 0
    assert planner.seeds == [10, 10, 11]


def test_no_key_planner_uses_fresh_current_observation_without_history() -> None:
    architecture = build_architecture(
        ArchitectureSpec.load(ROOT / "configs/architectures/fixed_planner_no_key.json")
    )
    planner = FakePlanner()
    policy = Mem0PlannerPolicy(
        FakeExecutorPolicy(),
        architecture=architecture,
        planner_backend=planner,
        global_task="task",
        planner_seed_base=20,
        boundary_mode="oracle_prompt_change",
    )

    policy.reset_history()
    policy.infer(_observation("zero", 1))
    policy.infer(_observation("one", 8))

    assert len(planner.contexts) == 2
    assert all(
        not hasattr(context, "completed_subtasks") for context in planner.contexts
    )
    assert np.all(planner.contexts[0].current_image == 1)
    assert np.all(planner.contexts[1].current_image == 8)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("next_subtask: Cover the red block.", "Cover the red block."),
        ("reasoning next_subtask: Uncover blue!", "Uncover blue."),
    ],
)
def test_parse_next_subtask(answer: str, expected: str) -> None:
    assert parse_next_subtask(answer) == expected


def test_parse_next_subtask_rejects_malformed_output() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_next_subtask("Cover red")
    with pytest.raises(ValueError, match="one subtask"):
        parse_next_subtask("next_subtask: Cover red\nextra")
