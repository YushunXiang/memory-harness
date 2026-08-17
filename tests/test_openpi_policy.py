import numpy as np
import pytest

from memory_harness.openpi_policy import _empty_mem0_context_shape
from memory_harness.openpi_policy import _validate_program_model_contract
from memory_harness.openpi_policy import _with_prompt_memory_hints
from memory_harness.config import ComponentSpec
from memory_harness.config import ProgramSpec
from memory_harness.registry import build_program


def test_prompt_drives_deployable_subtask_lifecycle_without_mutating_input() -> None:
    observation = {"prompt": np.asarray("Uncover the red block."), "state": np.zeros(2)}

    output = _with_prompt_memory_hints(observation)

    assert output["memory_phase_label"] == "Uncover the red block."
    assert output["memory_task_text"] == "Uncover the red block."
    assert set(observation) == {"prompt", "state"}


def test_explicit_planner_phase_is_preserved() -> None:
    observation = {
        "prompt": "executor text",
        "memory_phase_label": "planner-phase",
        "memory_task_text": "global task",
    }
    assert _with_prompt_memory_hints(observation) is observation


def test_empty_mem0_context_shape_is_derived_from_loaded_model() -> None:
    class Fusion:
        sliding_window_size = 30
        hidden_size = 2048

    class Model:
        _memory_enabled = True
        _memory_utilization_mode = "mem0"
        mem0_fusion = Fusion()

    class Policy:
        _model = Model()

    assert _empty_mem0_context_shape(Policy()) == (31, 2048)


def test_mem0_program_shape_is_checked_before_policy_wrapping() -> None:
    class Fusion:
        sliding_window_size = 30
        hidden_size = 2048

    class Model:
        _memory_enabled = True
        _memory_utilization_mode = "mem0"
        mem0_fusion = Fusion()

    class Policy:
        _model = Model()

    program = build_program(
        ProgramSpec(
            name="empty",
            deployable=True,
            paths=(),
            controller=ComponentSpec("all"),
            utilizer=ComponentSpec(
                "mem0_context",
                {
                    "embed_dim": 1024,
                    "sliding_window_size": 30,
                    "anchor_path": None,
                    "history_path_quotas": {},
                },
            ),
        )
    )

    with pytest.raises(ValueError, match="context shape mismatch"):
        _validate_program_model_contract(Policy(), program)
