from __future__ import annotations

from types import SimpleNamespace

import pytest

from memory_harness.official_mem0_planner_ablation import (
    install_planner_memory_condition,
)


def planner():
    return SimpleNamespace(
        global_task="cover then uncover",
        initial_observation="initial.png",
        key_information=[],
        prepare_qwen_input=lambda: ["key contract"],
        _image_to_data_url=lambda image: f"image:{image}",
    )


def test_key_preserves_released_planner_contract() -> None:
    value = planner()
    original = value.prepare_qwen_input
    assert install_planner_memory_condition(value, "key") is None
    assert value.prepare_qwen_input is original
    assert value._memory_harness_planner_memory_condition == "key"


def test_no_key_uses_initial_then_latest_observation() -> None:
    value = planner()
    original = install_planner_memory_condition(value, "no_key")
    assert original is not None

    first = value.prepare_qwen_input()
    assert first[1]["content"][2]["image_url"]["url"] == "image:initial.png"
    assert "<current_observation>" in first[1]["content"][1]["text"]
    assert "finished_subtasks" not in str(first)

    value.key_information.extend(["stage-1.png", "stage-2.png"])
    latest = value.prepare_qwen_input()
    assert latest[1]["content"][2]["image_url"]["url"] == "image:stage-2.png"


def test_no_key_rejects_missing_observation_and_double_install() -> None:
    value = planner()
    value.initial_observation = None
    install_planner_memory_condition(value, "no_key")
    with pytest.raises(RuntimeError, match="No current observation"):
        value.prepare_qwen_input()
    with pytest.raises(RuntimeError, match="already installed"):
        install_planner_memory_condition(value, "no_key")


def test_rejects_unknown_condition() -> None:
    with pytest.raises(ValueError, match="Unknown planner memory condition"):
        install_planner_memory_condition(planner(), "history")
