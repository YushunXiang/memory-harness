from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from memory_harness.key_planner import MEM0_NO_KEY_PLANNER_SYSTEM_PROMPT


PlannerMemoryCondition = Literal["key", "no_key"]


def install_planner_memory_condition(
    planner: Any,
    condition: PlannerMemoryCondition,
) -> Callable[..., Any] | None:
    """Select the released Mem-0 key or no-key planner input contract.

    The upstream agent continues to record completed subtasks in both conditions.
    ``no_key`` only changes what the planner consumes: the initial observation for
    the first call and the latest subtask-end observation thereafter.
    """
    if condition not in {"key", "no_key"}:
        raise ValueError(
            f"Unknown planner memory condition {condition!r}; expected 'key' or 'no_key'"
        )
    if condition == "key":
        planner._memory_harness_planner_memory_condition = condition
        return None
    if getattr(planner, "_memory_harness_original_prepare_qwen_input", None) is not None:
        raise RuntimeError("A planner memory condition is already installed")

    original_prepare = planner.prepare_qwen_input

    def prepare_without_key() -> list[dict[str, Any]]:
        history = getattr(planner, "key_information", ())
        current_observation = history[-1] if history else planner.initial_observation
        if current_observation is None:
            raise RuntimeError("No current observation is available for the no-key planner")
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": MEM0_NO_KEY_PLANNER_SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<global_task>: {planner.global_task}\n",
                    },
                    {"type": "text", "text": "<current_observation>: "},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": planner._image_to_data_url(current_observation)
                        },
                    },
                    {"type": "text", "text": ".\n"},
                ],
            },
        ]

    planner._memory_harness_original_prepare_qwen_input = original_prepare
    planner._memory_harness_planner_memory_condition = condition
    planner.prepare_qwen_input = prepare_without_key
    return original_prepare
