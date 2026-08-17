import numpy as np
import pytest

from memory_harness.key_planner import CurrentObservationPlannerContext
from memory_harness.key_planner import KeyPlannerMemory, render_planner_messages


def _image(value: int) -> np.ndarray:
    return np.full((4, 5, 3), value, dtype=np.uint8)


def test_key_planner_memory_preserves_ordered_text_rgb_records() -> None:
    memory = KeyPlannerMemory()
    initial = _image(1)
    memory.reset(global_task="cover then uncover", initial_image=initial)
    memory.write_completed_subtask(instruction="cover left", end_image=_image(2))
    memory.write_completed_subtask(instruction="cover middle", end_image=_image(3))

    context = memory.context()

    assert context.global_task == "cover then uncover"
    assert [record.ordinal for record in context.completed_subtasks] == [0, 1]
    assert [record.instruction for record in context.completed_subtasks] == [
        "cover left",
        "cover middle",
    ]
    np.testing.assert_array_equal(context.initial_image, initial)
    np.testing.assert_array_equal(context.completed_subtasks[-1].end_image, _image(3))


def test_key_planner_context_owns_copies_and_reset_clears_history() -> None:
    memory = KeyPlannerMemory()
    initial = _image(1)
    end = _image(2)
    memory.reset(global_task="task", initial_image=initial)
    memory.write_completed_subtask(instruction="step", end_image=end)
    initial[:] = 9
    end[:] = 9

    first = memory.context()
    assert np.all(first.initial_image == 1)
    assert np.all(first.completed_subtasks[0].end_image == 2)

    memory.reset(global_task="next", initial_image=_image(4))
    assert memory.context().completed_subtasks == ()


def test_key_planner_rejects_use_before_reset() -> None:
    memory = KeyPlannerMemory()
    with pytest.raises(RuntimeError, match="reset"):
        memory.context()
    with pytest.raises(RuntimeError, match="reset"):
        memory.write_completed_subtask(instruction="step", end_image=_image(1))


def test_key_context_renders_official_multimodal_planner_format() -> None:
    memory = KeyPlannerMemory()
    memory.reset(global_task="task", initial_image=_image(1))
    memory.write_completed_subtask(instruction="cover left", end_image=_image(2))

    messages = render_planner_messages(
        memory.context(),
        system_prompt="plan",
        image_to_url=lambda image: f"image:{int(image[0, 0, 0])}",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    content = messages[1]["content"]
    assert content[0]["text"] == "<global_task>: task\n"
    assert content[2]["image_url"]["url"] == "image:1"
    assert content[5]["text"] == "0: cover left. The corresponding image is: "
    assert content[6]["image_url"]["url"] == "image:2"


def test_no_key_context_renders_current_observation_contract() -> None:
    messages = render_planner_messages(
        CurrentObservationPlannerContext(global_task="task", current_image=_image(7)),
        system_prompt="plan current",
        image_to_url=lambda image: f"image:{int(image[0, 0, 0])}",
    )

    content = messages[1]["content"]
    assert content[0]["text"] == "<global_task>: task\n"
    assert content[1]["text"] == "<current_observation>: "
    assert content[2]["image_url"]["url"] == "image:7"
    assert "finished_subtasks" not in str(content)
