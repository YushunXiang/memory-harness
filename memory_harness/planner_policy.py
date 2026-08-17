from __future__ import annotations

import base64
import dataclasses
import io
import time
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import requests
from PIL import Image

from memory_harness.architecture import MemoryArchitecture
from memory_harness.key_planner import MEM0_NO_KEY_PLANNER_SYSTEM_PROMPT
from memory_harness.key_planner import MEM0_PLANNER_SYSTEM_PROMPT
from memory_harness.key_planner import KeyPlannerContext, PlannerContext
from memory_harness.key_planner import render_planner_messages


@dataclasses.dataclass(frozen=True)
class PlannerResult:
    instruction: str
    raw_answer: str
    latency_seconds: float


class PlannerBackend(Protocol):
    def plan(self, context: PlannerContext, *, seed: int) -> PlannerResult: ...


def parse_next_subtask(answer: str) -> str:
    """Parse the exact one-line output format used to train the Mem-0 planner."""
    if not isinstance(answer, str):
        raise TypeError(f"planner answer must be str, got {type(answer).__name__}")
    marker = "next_subtask:"
    if marker not in answer:
        raise ValueError(f"planner answer is missing {marker!r}: {answer!r}")
    subtask = answer.rsplit(marker, 1)[-1].strip()
    if not subtask or "\n" in subtask or "\r" in subtask:
        raise ValueError(
            f"planner answer must contain one subtask sentence: {answer!r}"
        )
    subtask = subtask.rstrip(".!? ")
    if not subtask:
        raise ValueError(f"planner answer contains an empty subtask: {answer!r}")
    return subtask + "."


def _image_to_data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class OpenAICompatiblePlannerBackend:
    """Small client for the reproduced Mem-0 planner served by vLLM."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 3600.0,
        key_system_prompt: str = MEM0_PLANNER_SYSTEM_PROMPT,
        no_key_system_prompt: str = MEM0_NO_KEY_PLANNER_SYSTEM_PROMPT,
        max_tokens: int = 128,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.key_system_prompt = key_system_prompt
        self.no_key_system_prompt = no_key_system_prompt
        self.max_tokens = max_tokens
        self._session = requests.Session()

    def plan(self, context: PlannerContext, *, seed: int) -> PlannerResult:
        system_prompt = (
            self.key_system_prompt
            if isinstance(context, KeyPlannerContext)
            else self.no_key_system_prompt
        )
        messages = render_planner_messages(
            context,
            system_prompt=system_prompt,
            image_to_url=_image_to_data_url,
        )
        start = time.perf_counter()
        response = self._session.post(
            self.endpoint,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "seed": int(seed),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            answer = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"invalid planner response payload: {payload!r}") from exc
        instruction = parse_next_subtask(answer)
        return PlannerResult(
            instruction=instruction,
            raw_answer=answer,
            latency_seconds=time.perf_counter() - start,
        )


def _head_rgb(observation: Mapping[str, Any]) -> np.ndarray:
    images = observation.get("images")
    if not isinstance(images, Mapping) or "cam_high" not in images:
        raise KeyError("key planner requires images.cam_high")
    image = np.asarray(images["cam_high"])
    if image.ndim != 3:
        raise ValueError(f"head image must be CHW or HWC RGB, got {image.shape}")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image[:3], 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"head image must have three RGB channels, got {image.shape}")
    return np.asarray(image, dtype=np.uint8)


def _scalar_prompt(observation: Mapping[str, Any]) -> str:
    value = np.asarray(observation.get("prompt"))
    if value.size != 1:
        raise ValueError("oracle boundary prompt must be a scalar string")
    prompt = str(value.reshape(()).item()).strip()
    if not prompt:
        raise ValueError("oracle boundary prompt must be non-empty")
    return prompt


class Mem0PlannerPolicy:
    """Mem-0 planner around an executor policy, with optional key memory.

    The current implementation intentionally accepts only the diagnostic
    ``oracle_prompt_change`` boundary. It uses simulator-derived subtask text
    solely to signal a transition; the planner output is what reaches π0.5.
    """

    uses_memory_runtime = True

    def __init__(
        self,
        base_policy: Any,
        *,
        architecture: MemoryArchitecture,
        planner_backend: PlannerBackend,
        global_task: str,
        planner_seed_base: int,
        boundary_mode: str,
    ) -> None:
        if not architecture.planner_enabled:
            raise ValueError("Mem-0 planner policy requires planner='mem0'")
        if boundary_mode != "oracle_prompt_change":
            raise ValueError("key planner currently supports only oracle_prompt_change")
        if not global_task.strip():
            raise ValueError("global_task must be non-empty")
        self.base_policy = base_policy
        self.architecture = architecture
        self.planner_backend = planner_backend
        self.global_task = global_task
        self.planner_seed_base = int(planner_seed_base)
        self.boundary_mode = boundary_mode
        self.episode_index = -1
        self._initialized = False
        self._boundary_label: str | None = None
        self._instruction: str | None = None
        self._planner_call_index = 0

    def reset_history(self) -> None:
        reset = getattr(self.base_policy, "reset_history", None)
        if callable(reset):
            reset()
        self.episode_index += 1
        self._initialized = False
        self._boundary_label = None
        self._instruction = None
        self._planner_call_index = 0

    def _plan(self, image: np.ndarray) -> PlannerResult:
        context = self.architecture.planner_context(current_image=image)
        assert context is not None
        result = self.planner_backend.plan(
            context,
            seed=self.planner_seed_base + self.episode_index,
        )
        self._planner_call_index += 1
        self.architecture.record_planner_call(
            instruction=result.instruction,
            raw_answer=result.raw_answer,
            latency_seconds=result.latency_seconds,
            boundary_source=self.boundary_mode,
        )
        self._instruction = result.instruction
        return result

    def infer(self, observation: Mapping[str, Any], *, noise=None) -> dict[str, Any]:
        if self.episode_index < 0:
            raise RuntimeError("reset_history() must be called before infer()")
        image = _head_rgb(observation)
        boundary_label = _scalar_prompt(observation)
        boundary_changed = False
        if not self._initialized:
            self.architecture.reset_planner_episode(
                episode_id=f"episode-{self.episode_index}",
                global_task=self.global_task,
                initial_image=image,
            )
            self._boundary_label = boundary_label
            self._plan(image)
            self._initialized = True
        elif boundary_label != self._boundary_label:
            assert self._instruction is not None
            self.architecture.record_completed_subtask(
                instruction=self._instruction,
                end_image=image,
                metadata={
                    "boundary_mode": self.boundary_mode,
                    "previous_oracle_label": self._boundary_label,
                    "next_oracle_label": boundary_label,
                    "deployable": False,
                },
            )
            self._boundary_label = boundary_label
            self._plan(image)
            boundary_changed = True

        assert self._instruction is not None
        executor_observation = dict(observation)
        executor_observation["prompt"] = self._instruction
        executor_observation["memory_phase_label"] = self._instruction
        executor_observation["memory_task_text"] = self.global_task
        try:
            output = self.base_policy.infer(executor_observation, noise=noise)
        except TypeError as exc:
            if "noise" not in str(exc):
                raise
            output = self.base_policy.infer(executor_observation)
        output["memory"] = {
            **output.get("memory", {}),
            "architecture": self.architecture.name,
            "active_modules": self.architecture.active_modules,
            "planner_instruction": self._instruction,
            "planner_call_count": self._planner_call_index,
            "planner_boundary_changed": boundary_changed,
            "planner_boundary_mode": self.boundary_mode,
            "planner_boundary_deployable": False,
        }
        return output

    def finish_episode(self, *, success: bool, total_reward: float = 0.0) -> None:
        finish = getattr(self.base_policy, "finish_episode", None)
        if callable(finish):
            finish(success=success, total_reward=total_reward)

    @property
    def _rng(self):
        return self.base_policy._rng

    @_rng.setter
    def _rng(self, value):
        self.base_policy._rng = value
