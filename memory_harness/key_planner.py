from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


MEM0_PLANNER_SYSTEM_PROMPT = (
    "You are a robotic assistant specialized in subtask planning. I will provide you with: "
    "1. global_task: A global task instruction. 2. initial_observation: An image of the initial "
    "observation of the task. 3. finished_subtasks: A list of subtask instructions completed by "
    "the robot. Each instruction is paired with an image showing the visual observation at the end "
    "of that subtask. The indices (0, 1, 2, ...) represent the temporal order of completion, where "
    "0 is the first completed subtask, 1 is the second, and so on.\nFormat: <global_task>: "
    "{global task instruction}. <initial_observation>: {initial observation image}. "
    "<finished_subtasks>: 0: {operation arm: finished subtask instruction}, the corresponding image "
    "is {image}. 1: {operation arm: finished subtask instruction}, the corresponding image is "
    "{image}. ...\nIMPORTANT: The numbers (0, 1, 2, ...) indicate the temporal sequence of "
    "completion. The highest index represents the most recently completed subtask. At the beginning "
    "of the task, the finished_subtasks list is null.\nBased on all the provided information, output "
    "the next subtask to execute in the format: 'next_subtask: {subtask name}.'.\n"
)

MEM0_NO_KEY_PLANNER_SYSTEM_PROMPT = (
    "You are a robotic assistant specialized in subtask planning. I will provide you with: "
    "1. global_task: A global task instruction. 2. current_observation: An image of the "
    "current observation of the task.\nFormat: <global_task>: {global task instruction}. "
    "<current_observation>: {current observation image}.\nBased on all the provided "
    "information, output the next subtask to execute in the format: "
    "'next_subtask: {subtask name}.'.\n"
)


def _rgb_copy(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(
            f"planner memory image must have shape [H, W, 3], got {value.shape}"
        )
    if value.dtype != np.uint8:
        raise ValueError(f"planner memory image must be uint8 RGB, got {value.dtype}")
    return value.copy()


@dataclasses.dataclass(frozen=True)
class CompletedSubtask:
    ordinal: int
    instruction: str
    end_image: np.ndarray
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not self.instruction.strip():
            raise ValueError("completed subtask instruction must be non-empty")
        object.__setattr__(self, "end_image", _rgb_copy(self.end_image))


@dataclasses.dataclass(frozen=True)
class KeyPlannerContext:
    global_task: str
    initial_image: np.ndarray
    completed_subtasks: tuple[CompletedSubtask, ...]

    def __post_init__(self) -> None:
        if not self.global_task.strip():
            raise ValueError("global task must be non-empty")
        object.__setattr__(self, "initial_image", _rgb_copy(self.initial_image))


@dataclasses.dataclass(frozen=True)
class CurrentObservationPlannerContext:
    """Mem-0 ``w/o key`` planner input from the released implementation."""

    global_task: str
    current_image: np.ndarray

    def __post_init__(self) -> None:
        if not self.global_task.strip():
            raise ValueError("global task must be non-empty")
        object.__setattr__(self, "current_image", _rgb_copy(self.current_image))


PlannerContext = KeyPlannerContext | CurrentObservationPlannerContext


class KeyPlannerMemory:
    """Mem-0 planner-side key memory with structured text+RGB records.

    This is deliberately separate from action-side latent memory: the planner
    receives the episode's initial image and every completed subtask paired with
    its end image, in temporal order.
    """

    def __init__(self) -> None:
        self._global_task: str | None = None
        self._initial_image: np.ndarray | None = None
        self._completed: list[CompletedSubtask] = []

    def reset(self, *, global_task: str, initial_image: np.ndarray) -> None:
        if not global_task.strip():
            raise ValueError("global task must be non-empty")
        self._global_task = global_task
        self._initial_image = _rgb_copy(initial_image)
        self._completed.clear()

    def write_completed_subtask(
        self,
        *,
        instruction: str,
        end_image: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> CompletedSubtask:
        if self._initial_image is None:
            raise RuntimeError("reset() must be called before writing planner memory")
        record = CompletedSubtask(
            ordinal=len(self._completed),
            instruction=instruction,
            end_image=end_image,
            metadata={} if metadata is None else dict(metadata),
        )
        self._completed.append(record)
        return record

    def context(self) -> KeyPlannerContext:
        if self._global_task is None or self._initial_image is None:
            raise RuntimeError("reset() must be called before reading planner memory")
        records = tuple(
            dataclasses.replace(record, end_image=record.end_image.copy())
            for record in self._completed
        )
        return KeyPlannerContext(
            global_task=self._global_task,
            initial_image=self._initial_image.copy(),
            completed_subtasks=records,
        )


def render_planner_messages(
    context: PlannerContext,
    *,
    system_prompt: str,
    image_to_url: Callable[[np.ndarray], str],
) -> list[dict[str, Any]]:
    """Render either released Mem-0 planner input contract without mixing them."""
    if isinstance(context, CurrentObservationPlannerContext):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<global_task>: {context.global_task}\n",
                    },
                    {"type": "text", "text": "<current_observation>: "},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_url(context.current_image)},
                    },
                    {"type": "text", "text": ".\n"},
                ],
            },
        ]

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": f"<global_task>: {context.global_task}\n"},
        {"type": "text", "text": "<initial_observation>: "},
        {
            "type": "image_url",
            "image_url": {"url": image_to_url(context.initial_image)},
        },
        {"type": "text", "text": ".\n"},
    ]
    if context.completed_subtasks:
        user_content.append({"type": "text", "text": "<finished_subtasks>: "})
        for index, record in enumerate(context.completed_subtasks):
            instruction = record.instruction.strip()
            if not instruction.endswith((".", "!", "?")):
                instruction += "."
            user_content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"{index}: {instruction} The corresponding image is: "
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_url(record.end_image)},
                    },
                    {"type": "text", "text": ". "},
                ]
            )
        user_content[-1] = {"type": "text", "text": ".\n"}
    else:
        user_content.append({"type": "text", "text": "<finished_subtasks>: null.\n"})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {"role": "user", "content": user_content},
    ]
