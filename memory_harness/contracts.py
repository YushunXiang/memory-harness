from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class MemoryStep:
    """Deployment-observable input available to a memory program at one policy step."""

    episode_id: str
    step_index: int
    phase: str = ""
    source_tokens: np.ndarray | None = None
    source_mask: np.ndarray | None = None
    robot_state: np.ndarray | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if (self.source_tokens is None) != (self.source_mask is None):
            raise ValueError("source_tokens and source_mask must be provided together")
        if self.source_tokens is not None:
            tokens = np.asarray(self.source_tokens, dtype=np.float32)
            mask = np.asarray(self.source_mask, dtype=np.bool_)
            if tokens.ndim != 2:
                raise ValueError(
                    f"source_tokens must have shape [M, D], got {tokens.shape}"
                )
            if mask.shape != (tokens.shape[0],):
                raise ValueError(
                    f"source_mask must have shape {(tokens.shape[0],)}, got {mask.shape}"
                )
            object.__setattr__(self, "source_tokens", tokens)
            object.__setattr__(self, "source_mask", mask)
        if self.robot_state is not None:
            robot_state = np.asarray(self.robot_state, dtype=np.float32)
            if robot_state.ndim != 1 or not robot_state.size:
                raise ValueError(
                    "robot_state must be a non-empty rank-1 vector, "
                    f"got {robot_state.shape}"
                )
            if not np.isfinite(robot_state).all():
                raise ValueError("robot_state must contain only finite values")
            object.__setattr__(self, "robot_state", robot_state)


@dataclasses.dataclass(frozen=True)
class MemoryItem:
    item_id: str
    path_name: str
    episode_id: str
    step_index: int
    phase: str
    tokens: np.ndarray
    mask: np.ndarray
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id or not self.path_name:
            raise ValueError("item_id and path_name must be non-empty")
        tokens = np.asarray(self.tokens, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=np.bool_)
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [M, D], got {tokens.shape}")
        if mask.shape != (tokens.shape[0],):
            raise ValueError(
                f"mask must have shape {(tokens.shape[0],)}, got {mask.shape}"
            )
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "mask", mask)

    @property
    def valid_token_count(self) -> int:
        return int(self.mask.sum())


@dataclasses.dataclass(frozen=True)
class EpisodeOutcome:
    """Deployment outcome supplied after an episode has terminated."""

    episode_id: str
    success: bool
    final_step_index: int
    total_reward: float = 0.0
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.final_step_index < 0:
            raise ValueError("final_step_index must be non-negative")
        if not np.isfinite(self.total_reward):
            raise ValueError("total_reward must be finite")


@dataclasses.dataclass(frozen=True)
class WriteDecision:
    """Auditable output of a write rule before an item is encoded or stored."""

    write: bool
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    write_step: MemoryStep | None = None

    def __post_init__(self) -> None:
        if not self.write and self.write_step is not None:
            raise ValueError("a skipped write cannot carry a write_step")


@dataclasses.dataclass(frozen=True)
class RetrievalResult:
    """Typed retrieval output and the evidence needed to audit the choice."""

    items: tuple[MemoryItem, ...]
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        item_ids = tuple(item.item_id for item in self.items)
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("retrieval result cannot contain duplicate items")


@dataclasses.dataclass(frozen=True)
class UtilizationResult:
    """Policy input plus exact evidence of which retrieved items were consumed."""

    observation: Mapping[str, Any]
    used_items: tuple[MemoryItem, ...]
    used_token_count: int
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.used_token_count < 0:
            raise ValueError("used_token_count must be non-negative")
        item_ids = tuple(item.item_id for item in self.used_items)
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("utilization result cannot contain duplicate items")
        if self.used_token_count == 0 and self.used_items:
            raise ValueError("zero-token utilization cannot report used items")


@dataclasses.dataclass(frozen=True)
class AuditEvent:
    event: str
    episode_id: str
    step_index: int
    program_name: str
    path_name: str | None = None
    item_ids: tuple[str, ...] = ()
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "program_name": self.program_name,
            "path_name": self.path_name,
            "item_ids": list(self.item_ids),
            "details": dict(self.details),
        }


@dataclasses.dataclass(frozen=True)
class StepResult:
    observation: Mapping[str, Any]
    events: tuple[AuditEvent, ...]
    retrieved_item_ids: tuple[str, ...]
    used_token_count: int
    stored_item_count: int
