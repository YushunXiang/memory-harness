from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from memory_harness.contracts import MemoryStep
from memory_harness.contracts import EpisodeOutcome
from memory_harness.contracts import StepResult
from memory_harness.runtime import MemoryProgram


@dataclasses.dataclass(frozen=True)
class TokenSourceResult:
    policy_observation: Mapping[str, Any]
    tokens: np.ndarray
    mask: np.ndarray


class TokenSource(Protocol):
    def extract(self, observation: Mapping[str, Any]) -> TokenSourceResult: ...


class ObservationFieldTokenSource:
    """Adapter for policies that already expose deployment-observable moment tokens."""

    def __init__(
        self,
        *,
        tokens_key: str = "_memory_source_tokens",
        mask_key: str = "_memory_source_mask",
    ) -> None:
        self.tokens_key = tokens_key
        self.mask_key = mask_key

    def extract(self, observation: Mapping[str, Any]) -> TokenSourceResult:
        if self.tokens_key not in observation or self.mask_key not in observation:
            raise ValueError(
                f"memory-enabled policy input requires {self.tokens_key!r} and {self.mask_key!r}"
            )
        policy_observation = dict(observation)
        tokens = np.asarray(policy_observation.pop(self.tokens_key), dtype=np.float32)
        mask = np.asarray(policy_observation.pop(self.mask_key), dtype=np.bool_)
        return TokenSourceResult(policy_observation, tokens, mask)


class MemoryHarnessPolicy:
    """Thin policy wrapper; all memory behavior remains inside MemoryProgram."""

    def __init__(
        self,
        base_policy: Any,
        program: MemoryProgram,
        *,
        token_source: TokenSource | None = None,
        empty_context_shape: tuple[int, int] | None = None,
    ) -> None:
        if program.paths and token_source is None:
            raise ValueError("memory-enabled programs require a token_source")
        if not program.paths and token_source is not None:
            raise ValueError("the none program must not have a token_source")
        if program.paths and empty_context_shape is not None:
            raise ValueError("only the none program may emit an empty model context")
        if empty_context_shape is not None and any(
            size <= 0 for size in empty_context_shape
        ):
            raise ValueError("empty_context_shape dimensions must be positive")
        self.base_policy = base_policy
        self.program = program
        self.token_source = token_source
        self.empty_context_shape = empty_context_shape
        self.episode_index = -1
        self.step_index = 0
        self.last_memory_result: StepResult | None = None

    def reset_history(self) -> None:
        reset = getattr(self.base_policy, "reset_history", None)
        if callable(reset):
            reset()
        self.episode_index += 1
        self.step_index = 0
        self.last_memory_result = None
        self.program.reset(episode_id=f"episode-{self.episode_index}")

    def infer(self, observation: Mapping[str, Any]) -> Any:
        if self.episode_index < 0:
            raise RuntimeError("reset_history() must be called before infer()")
        phase = str(observation.get("_memory_phase", ""))
        robot_state = observation.get("state")
        if self.token_source is None:
            policy_observation = observation
            tokens = None
            mask = None
        else:
            source = self.token_source.extract(observation)
            policy_observation = source.policy_observation
            policy_observation.pop("_memory_phase", None)
            tokens = source.tokens
            mask = source.mask
        step = MemoryStep(
            episode_id=f"episode-{self.episode_index}",
            step_index=self.step_index,
            phase=phase,
            source_tokens=tokens,
            source_mask=mask,
            robot_state=(
                None
                if robot_state is None
                else np.asarray(robot_state, dtype=np.float32)
            ),
        )
        result = self.program.step(policy_observation, step)
        if self.empty_context_shape is not None:
            if (
                "memory_tokens" in result.observation
                or "memory_mask" in result.observation
            ):
                raise ValueError(
                    "the harness must be the only owner of model memory inputs"
                )
            model_observation = dict(result.observation)
            model_observation["memory_tokens"] = np.zeros(
                self.empty_context_shape, dtype=np.float32
            )
            model_observation["memory_mask"] = np.zeros(
                (self.empty_context_shape[0],), dtype=np.bool_
            )
        else:
            model_observation = result.observation
        self.last_memory_result = result
        self.step_index += 1
        return self.base_policy.infer(model_observation)

    def finish_episode(self, *, success: bool, total_reward: float = 0.0) -> None:
        if self.episode_index < 0 or self.step_index <= 0:
            raise RuntimeError("cannot finish before an episode has produced a memory step")
        self.program.finish_episode(
            EpisodeOutcome(
                episode_id=f"episode-{self.episode_index}",
                success=bool(success),
                final_step_index=self.step_index - 1,
                total_reward=float(total_reward),
            )
        )

    @property
    def _rng(self):
        return self.base_policy._rng

    @_rng.setter
    def _rng(self, value):
        self.base_policy._rng = value
