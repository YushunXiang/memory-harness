from __future__ import annotations

import abc
import collections
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from memory_harness.contracts import MemoryItem
from memory_harness.contracts import EpisodeOutcome
from memory_harness.contracts import MemoryStep
from memory_harness.contracts import RetrievalResult
from memory_harness.contracts import UtilizationResult
from memory_harness.contracts import WriteDecision


MEMORY_INPUT_KEYS = frozenset({"memory_tokens", "memory_mask"})


class Encoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, step: MemoryStep, *, path_name: str) -> MemoryItem:
        raise NotImplementedError


class TokenEncoder(Encoder):
    def __init__(self, *, max_tokens: int):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = int(max_tokens)

    def encode(self, step: MemoryStep, *, path_name: str) -> MemoryItem:
        if step.source_tokens is None or step.source_mask is None:
            raise ValueError(f"memory path {path_name!r} requires source tokens")
        valid = np.asarray(step.source_tokens)[
            np.asarray(step.source_mask, dtype=np.bool_)
        ]
        valid = valid[: self.max_tokens].astype(np.float32, copy=True)
        if not len(valid):
            raise ValueError(
                f"memory path {path_name!r} received no valid source tokens"
            )
        mask = np.ones((valid.shape[0],), dtype=np.bool_)
        return MemoryItem(
            item_id=f"{step.episode_id}:{path_name}:{step.step_index}",
            path_name=path_name,
            episode_id=step.episode_id,
            step_index=step.step_index,
            phase=step.phase,
            tokens=valid,
            mask=mask,
            metadata=dict(step.metadata),
        )


class WriteRule(abc.ABC):
    @abc.abstractmethod
    def decide(self, step: MemoryStep, store: Store) -> WriteDecision:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear writer-local temporal state at a lifecycle boundary."""
        return None


class FirstWrite(WriteRule):
    def decide(self, step: MemoryStep, store: Store) -> WriteDecision:
        del step
        write = len(store.items()) == 0
        return WriteDecision(
            write, {"reason": "empty_store" if write else "already_initialized"}
        )


class AlwaysWrite(WriteRule):
    def decide(self, step: MemoryStep, store: Store) -> WriteDecision:
        del step, store
        return WriteDecision(True, {"reason": "always"})


class AfterFirstStepWrite(WriteRule):
    def decide(self, step: MemoryStep, store: Store) -> WriteDecision:
        del store
        write = step.step_index > 0
        return WriteDecision(
            write, {"reason": "after_first_step" if write else "first_step"}
        )


class PhaseChangeWrite(WriteRule):
    def decide(self, step: MemoryStep, store: Store) -> WriteDecision:
        items = store.items()
        write = not items or items[-1].phase != step.phase
        return WriteDecision(
            write, {"reason": "phase_change" if write else "same_phase"}
        )


class NoveltyWrite(WriteRule):
    """Write when the current latent differs from the last retained latent.

    This is a deterministic, deployment-observable active-write baseline. It is
    intentionally not presented as a faithful reproduction of MemCtrl's learned
    MLLM head; a learned write controller must outperform this cheaper rule.
    """

    def __init__(
        self,
        *,
        min_cosine_distance: float,
        max_steps_without_write: int | None = None,
    ) -> None:
        if not 0 <= min_cosine_distance <= 2:
            raise ValueError("min_cosine_distance must be in [0, 2]")
        if max_steps_without_write is not None and max_steps_without_write <= 0:
            raise ValueError("max_steps_without_write must be positive when provided")
        self.min_cosine_distance = float(min_cosine_distance)
        self.max_steps_without_write = (
            None if max_steps_without_write is None else int(max_steps_without_write)
        )

    @staticmethod
    def _pooled(tokens: np.ndarray, mask: np.ndarray, *, label: str) -> np.ndarray:
        valid = np.asarray(tokens, dtype=np.float32)[np.asarray(mask, dtype=np.bool_)]
        if not len(valid):
            raise ValueError(
                f"novelty writer requires at least one valid {label} token"
            )
        return valid.mean(axis=0)

    @staticmethod
    def _cosine_distance(current: np.ndarray, previous: np.ndarray) -> float:
        if current.shape != previous.shape:
            raise ValueError(
                "novelty writer requires matching current/stored embedding widths, "
                f"got {current.shape} and {previous.shape}"
            )
        current_norm = float(np.linalg.norm(current))
        previous_norm = float(np.linalg.norm(previous))
        if current_norm == 0 or previous_norm == 0:
            return 0.0 if np.array_equal(current, previous) else 1.0
        similarity = float(np.dot(current, previous) / (current_norm * previous_norm))
        return 1.0 - float(np.clip(similarity, -1.0, 1.0))

    def decide(self, step: MemoryStep, store: Store) -> WriteDecision:
        if step.source_tokens is None or step.source_mask is None:
            raise ValueError("novelty writer requires source tokens")
        items = store.items()
        if not items:
            return WriteDecision(
                True,
                {
                    "reason": "empty_store",
                    "min_cosine_distance": self.min_cosine_distance,
                },
            )

        previous_item = items[-1]
        current = self._pooled(step.source_tokens, step.source_mask, label="source")
        previous = self._pooled(
            previous_item.tokens, previous_item.mask, label="stored"
        )
        distance = self._cosine_distance(current, previous)
        forced_by_interval = (
            self.max_steps_without_write is not None
            and step.step_index - previous_item.step_index
            >= self.max_steps_without_write
        )
        write = forced_by_interval or distance >= self.min_cosine_distance
        reason = (
            "max_interval"
            if forced_by_interval
            else ("novel" if write else "redundant")
        )
        return WriteDecision(
            write,
            {
                "reason": reason,
                "cosine_distance": distance,
                "min_cosine_distance": self.min_cosine_distance,
                "max_steps_without_write": self.max_steps_without_write,
                "previous_item_id": previous_item.item_id,
            },
        )


class CausalKinematicPeakWrite(WriteRule):
    """Delay-confirm motion slowdowns and write the historical candidate payload.

    The saliency score follows KEMO, but the paper does not publish its online
    peak-detector implementation.  This class therefore exposes an explicit,
    deterministic causal baseline rather than claiming a faithful reproduction.
    """

    def __init__(
        self,
        *,
        motion_window: int,
        peak_lookback: int,
        confirmation_delay: int,
        refractory_steps: int,
    ) -> None:
        values = {
            "motion_window": motion_window,
            "peak_lookback": peak_lookback,
            "confirmation_delay": confirmation_delay,
            "refractory_steps": refractory_steps,
        }
        if any(not isinstance(value, int) or value <= 0 for value in values.values()):
            raise ValueError(
                f"kinematic peak parameters must be positive ints: {values}"
            )
        self.motion_window = motion_window
        self.peak_lookback = peak_lookback
        self.confirmation_delay = confirmation_delay
        self.refractory_steps = refractory_steps
        self._previous_state: np.ndarray | None = None
        self._displacements: list[float] = []
        self._scored_steps: list[tuple[MemoryStep, float]] = []
        self._last_written_step: int | None = None

    def reset(self) -> None:
        self._previous_state = None
        self._displacements.clear()
        self._scored_steps.clear()
        self._last_written_step = None

    def decide(self, step: MemoryStep, store: Store) -> WriteDecision:
        del store
        if step.robot_state is None:
            raise ValueError("kinematic peak writer requires robot_state")
        if step.source_tokens is None or step.source_mask is None:
            raise ValueError("kinematic peak writer requires source tokens")

        state = np.asarray(step.robot_state, dtype=np.float32)
        if self._previous_state is None:
            self._previous_state = state.copy()
            return WriteDecision(False, {"reason": "first_robot_state"})
        if state.shape != self._previous_state.shape:
            raise ValueError(
                "kinematic peak writer requires a stable robot_state width, "
                f"got {state.shape} after {self._previous_state.shape}"
            )

        displacement = float(np.linalg.norm(state - self._previous_state))
        self._previous_state = state.copy()
        self._displacements.append(displacement)
        if len(self._displacements) < self.motion_window:
            return WriteDecision(
                False,
                {
                    "reason": "motion_window_warmup",
                    "observed_displacements": len(self._displacements),
                    "motion_window": self.motion_window,
                },
            )

        recent_motion = self._displacements[-self.motion_window :]
        mean_displacement = float(np.mean(np.asarray(recent_motion)))
        saliency = 1.0 / (1.0 + mean_displacement)
        self._scored_steps.append((step, saliency))
        candidate_index = len(self._scored_steps) - 1 - self.confirmation_delay
        if candidate_index < self.peak_lookback:
            return WriteDecision(
                False,
                {
                    "reason": "peak_window_warmup",
                    "saliency": saliency,
                    "scored_steps": len(self._scored_steps),
                },
            )

        window_start = candidate_index - self.peak_lookback
        window_stop = candidate_index + self.confirmation_delay + 1
        window = self._scored_steps[window_start:window_stop]
        candidate_offset = self.peak_lookback
        scores = np.asarray([score for _, score in window], dtype=np.float64)
        # The earliest maximum wins ties, preventing stationary plateaus from
        # producing a stream of repeated events.
        is_peak = int(np.argmax(scores)) == candidate_offset
        candidate_step, candidate_saliency = self._scored_steps[candidate_index]
        outside_refractory = (
            self._last_written_step is None
            or candidate_step.step_index - self._last_written_step
            >= self.refractory_steps
        )
        write = is_peak and outside_refractory
        if write:
            self._last_written_step = candidate_step.step_index
        reason = (
            "causal_peak" if write else ("refractory" if is_peak else "not_local_peak")
        )
        return WriteDecision(
            write,
            {
                "reason": reason,
                "candidate_step_index": candidate_step.step_index,
                "confirmation_step_index": step.step_index,
                "confirmation_delay": step.step_index - candidate_step.step_index,
                "candidate_saliency": candidate_saliency,
                "mean_joint_displacement": (1.0 / candidate_saliency) - 1.0,
                "motion_window": self.motion_window,
                "peak_lookback": self.peak_lookback,
                "refractory_steps": self.refractory_steps,
            },
            write_step=candidate_step if write else None,
        )


class Store(abc.ABC):
    requires_episode_outcome = False

    @abc.abstractmethod
    def write(self, item: MemoryItem) -> Mapping[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def items(self) -> tuple[MemoryItem, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def begin_episode(self, episode_id: str) -> Mapping[str, Any]:
        """Start an episode; ordinary stores have episode-local scope."""
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        self.reset()
        return {"persistence": "episode_local", "retained_item_count": 0}

    def finish_episode(self, outcome: EpisodeOutcome) -> Mapping[str, Any]:
        """Finalize episode writes; ordinary stores need no outcome action."""
        del outcome
        return {"persistence": "episode_local", "action": "no_commit"}


class AnchorStore(Store):
    def __init__(self) -> None:
        self._item: MemoryItem | None = None

    def write(self, item: MemoryItem) -> Mapping[str, Any]:
        if self._item is not None:
            raise RuntimeError("anchor store already contains an item")
        self._item = item
        return {}

    def items(self) -> tuple[MemoryItem, ...]:
        return () if self._item is None else (self._item,)

    def reset(self) -> None:
        self._item = None


class RingStore(Store):
    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ring capacity must be positive")
        self._items: collections.deque[MemoryItem] = collections.deque(
            maxlen=int(capacity)
        )

    def write(self, item: MemoryItem) -> Mapping[str, Any]:
        evicted = (
            self._items[0].item_id if len(self._items) == self._items.maxlen else None
        )
        self._items.append(item)
        return {"evicted_item_id": evicted} if evicted is not None else {}

    def items(self) -> tuple[MemoryItem, ...]:
        return tuple(self._items)

    def reset(self) -> None:
        self._items.clear()


class VerifiedSuccessRingStore(Store):
    """Persist only items from completed successful deployment episodes.

    Writes remain invisible in an episode-local transaction buffer until an
    externally observed success outcome commits them. Failed episodes are
    discarded, so retrieval cannot consume unverified or future experience.
    """

    requires_episode_outcome = True

    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("verified-success ring capacity must be positive")
        self.capacity = int(capacity)
        self._committed: collections.deque[MemoryItem] = collections.deque(
            maxlen=self.capacity
        )
        self._pending: collections.deque[MemoryItem] = collections.deque(
            maxlen=self.capacity
        )
        self._active_episode_id: str | None = None

    def begin_episode(self, episode_id: str) -> Mapping[str, Any]:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        if self._active_episode_id is not None:
            raise RuntimeError(
                "verified-success store requires finish_episode() before the next episode"
            )
        self._active_episode_id = episode_id
        self._pending.clear()
        return {
            "persistence": "verified_success",
            "retained_item_count": len(self._committed),
        }

    def write(self, item: MemoryItem) -> Mapping[str, Any]:
        if self._active_episode_id is None:
            raise RuntimeError("begin_episode() must be called before write()")
        if item.episode_id != self._active_episode_id:
            raise ValueError("pending item does not belong to the active episode")
        evicted = (
            self._pending[0].item_id
            if len(self._pending) == self._pending.maxlen
            else None
        )
        self._pending.append(item)
        details: dict[str, Any] = {
            "transaction_state": "pending",
            "pending_item_count": len(self._pending),
            "committed_item_count": len(self._committed),
        }
        if evicted is not None:
            details["evicted_pending_item_id"] = evicted
        return details

    def items(self) -> tuple[MemoryItem, ...]:
        return tuple(self._committed)

    def finish_episode(self, outcome: EpisodeOutcome) -> Mapping[str, Any]:
        if self._active_episode_id is None:
            raise RuntimeError("no active episode to finish")
        if outcome.episode_id != self._active_episode_id:
            raise ValueError("outcome does not belong to the active episode")
        pending = tuple(self._pending)
        evicted_ids: list[str] = []
        if outcome.success:
            for item in pending:
                if len(self._committed) == self._committed.maxlen:
                    evicted_ids.append(self._committed[0].item_id)
                self._committed.append(
                    dataclasses.replace(
                        item,
                        metadata={
                            **dict(item.metadata),
                            "verified_success": True,
                        },
                    )
                )
        self._pending.clear()
        self._active_episode_id = None
        return {
            "persistence": "verified_success",
            "action": "commit" if outcome.success else "discard",
            "pending_item_count": len(pending),
            "committed_item_count": len(self._committed),
            "evicted_item_ids": evicted_ids,
        }

    def reset(self) -> None:
        """Clear the complete deployment-session bank."""
        self._committed.clear()
        self._pending.clear()
        self._active_episode_id = None


class AdjacentMergeStore(Store):
    """Capacity-bounded store using MemoryVLA-style adjacent token consolidation."""

    def __init__(self, *, capacity: int) -> None:
        if capacity <= 1:
            raise ValueError("adjacent-merge capacity must be greater than one")
        self.capacity = int(capacity)
        self._items: list[MemoryItem] = []
        self._merge_count = 0

    @staticmethod
    def _similarity(left: MemoryItem, right: MemoryItem) -> float:
        left_tokens = left.tokens[left.mask]
        right_tokens = right.tokens[right.mask]
        if left_tokens.shape != right_tokens.shape:
            raise ValueError("adjacent consolidation requires equal token shapes")
        left_vector = left_tokens.reshape(-1)
        right_vector = right_tokens.reshape(-1)
        denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denominator == 0:
            return 1.0 if np.array_equal(left_vector, right_vector) else 0.0
        return float(np.dot(left_vector, right_vector) / denominator)

    def write(self, item: MemoryItem) -> Mapping[str, Any]:
        self._items.append(item)
        if len(self._items) <= self.capacity:
            return {}
        similarities = [
            self._similarity(left, right)
            for left, right in zip(self._items[:-1], self._items[1:], strict=True)
        ]
        merge_index = int(np.argmax(np.asarray(similarities)))
        left = self._items[merge_index]
        right = self._items[merge_index + 1]
        if left.tokens.shape != right.tokens.shape or not np.array_equal(
            left.mask, right.mask
        ):
            raise ValueError("adjacent consolidation requires equal stored layouts")
        left_count = int(left.metadata.get("consolidated_count", 1))
        right_count = int(right.metadata.get("consolidated_count", 1))
        left_source_count = int(
            left.metadata.get(
                "source_item_count", left.metadata.get("summary_count", 1)
            )
        )
        right_source_count = int(
            right.metadata.get(
                "source_item_count", right.metadata.get("summary_count", 1)
            )
        )
        summary_start_step = min(
            int(left.metadata.get("summary_start_step", left.step_index)),
            int(right.metadata.get("summary_start_step", right.step_index)),
        )
        summary_end_step = max(
            int(left.metadata.get("summary_end_step", left.step_index)),
            int(right.metadata.get("summary_end_step", right.step_index)),
        )
        merged = MemoryItem(
            item_id=(
                f"{right.episode_id}:{right.path_name}:consolidated:{self._merge_count}"
            ),
            path_name=right.path_name,
            episode_id=right.episode_id,
            step_index=right.step_index,
            phase=right.phase,
            tokens=(left.tokens + right.tokens) / 2.0,
            mask=right.mask.copy(),
            metadata={
                **dict(right.metadata),
                "consolidated_count": left_count + right_count,
                "source_item_count": left_source_count + right_source_count,
                "summary_start_step": summary_start_step,
                "summary_end_step": summary_end_step,
            },
        )
        self._items[merge_index : merge_index + 2] = [merged]
        self._merge_count += 1
        return {
            "consolidated": True,
            "merged_item_ids": [left.item_id, right.item_id],
            "result_item_id": merged.item_id,
            "similarity": similarities[merge_index],
            "consolidated_count": left_count + right_count,
        }

    def items(self) -> tuple[MemoryItem, ...]:
        return tuple(self._items)

    def reset(self) -> None:
        self._items.clear()
        self._merge_count = 0


class TieredChunkMeanStore(Store):
    """Keep recent items losslessly and migrate old chunks into long-term summaries.

    This is a parameter-free lower bound for MemoAct-style two-tier memory.  It
    deliberately uses an arithmetic mean instead of pretending to reproduce the
    paper's learned causal Transformer compressor.  The long-term tier reuses
    adjacent consolidation once its own capacity is reached.
    """

    def __init__(
        self,
        *,
        short_capacity: int,
        migration_chunk_size: int,
        long_capacity: int,
    ) -> None:
        if short_capacity <= 1:
            raise ValueError("tiered short capacity must be greater than one")
        if migration_chunk_size <= 1:
            raise ValueError("tiered migration chunk size must be greater than one")
        if migration_chunk_size > short_capacity:
            raise ValueError("tiered migration chunk size cannot exceed short capacity")
        if long_capacity <= 1:
            raise ValueError("tiered long capacity must be greater than one")
        self.short_capacity = int(short_capacity)
        self.migration_chunk_size = int(migration_chunk_size)
        self.long_capacity = int(long_capacity)
        self._short_items: list[MemoryItem] = []
        self._long_store = AdjacentMergeStore(capacity=self.long_capacity)
        self._summary_count = 0

    def _validate_compatible(self, item: MemoryItem) -> None:
        existing = self.items()
        if not existing:
            return
        reference = existing[0]
        if item.episode_id != reference.episode_id:
            raise ValueError("tiered store cannot mix episodes")
        if item.path_name != reference.path_name:
            raise ValueError("tiered store cannot mix memory paths")
        if item.tokens.shape != reference.tokens.shape or not np.array_equal(
            item.mask, reference.mask
        ):
            raise ValueError("tiered store requires equal stored layouts")

    def _summarize(self, chunk: Sequence[MemoryItem]) -> MemoryItem:
        if not chunk:
            raise ValueError("tiered store cannot summarize an empty chunk")
        first = chunk[0]
        for item in chunk[1:]:
            if (
                item.episode_id != first.episode_id
                or item.path_name != first.path_name
                or item.tokens.shape != first.tokens.shape
                or not np.array_equal(item.mask, first.mask)
            ):
                raise ValueError("tiered store requires a compatible migration chunk")
        last = chunk[-1]
        summary = MemoryItem(
            item_id=(
                f"{last.episode_id}:{last.path_name}:tiered-summary:"
                f"{self._summary_count}"
            ),
            path_name=last.path_name,
            episode_id=last.episode_id,
            step_index=last.step_index,
            phase=last.phase,
            tokens=np.mean(
                np.stack([item.tokens for item in chunk], axis=0),
                axis=0,
                dtype=np.float32,
            ),
            mask=last.mask.copy(),
            metadata={
                **dict(last.metadata),
                "memory_tier": "long_term",
                "summary_count": len(chunk),
                "source_item_count": len(chunk),
                "summary_start_step": first.step_index,
                "summary_end_step": last.step_index,
            },
        )
        self._summary_count += 1
        return summary

    def write(self, item: MemoryItem) -> Mapping[str, Any]:
        self._validate_compatible(item)
        self._short_items.append(
            dataclasses.replace(
                item,
                metadata={**dict(item.metadata), "memory_tier": "short_term"},
            )
        )
        if len(self._short_items) <= self.short_capacity:
            return {
                "maintenance_action": "append_short",
                "short_term_count": len(self._short_items),
                "long_term_count": len(self._long_store.items()),
            }

        chunk = tuple(self._short_items[: self.migration_chunk_size])
        summary = self._summarize(chunk)
        long_term_details = self._long_store.write(summary)
        del self._short_items[: self.migration_chunk_size]
        return {
            "maintenance_action": "migrate_chunk",
            "migrated_item_ids": [chunk_item.item_id for chunk_item in chunk],
            "summary_item_id": summary.item_id,
            "short_term_count": len(self._short_items),
            "long_term_count": len(self._long_store.items()),
            "long_term_maintenance": dict(long_term_details),
        }

    def items(self) -> tuple[MemoryItem, ...]:
        return (*self._long_store.items(), *self._short_items)

    def reset(self) -> None:
        self._short_items.clear()
        self._long_store.reset()
        self._summary_count = 0


class DHEMEventStore(Store):
    """One-bank DiM-WAM historical-event maintenance operator.

    The first event is a fixed anchor and the last event is the latest retained
    event.  At capacity, the store either discards a redundant incoming event or
    mass-weight merges the most redundant adjacent pair in the middle history.
    Bank-conditioned encoding and multi-bank diversity are learned DiM-WAM
    components and deliberately remain outside this parameter-free store.
    """

    def __init__(self, *, capacity: int, temporal_decay: float) -> None:
        if capacity < 4:
            raise ValueError("DHEM event capacity must be at least four")
        if not np.isfinite(temporal_decay) or temporal_decay <= 0:
            raise ValueError("DHEM temporal_decay must be finite and positive")
        self.capacity = int(capacity)
        self.temporal_decay = float(temporal_decay)
        self._items: list[MemoryItem] = []
        self._merge_count = 0

    @staticmethod
    def _mass(item: MemoryItem) -> float:
        mass = float(item.metadata.get("accumulated_mass", 1.0))
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("DHEM accumulated mass must be finite and positive")
        return mass

    @staticmethod
    def _event_time(item: MemoryItem) -> float:
        event_time = float(item.metadata.get("representative_time", item.step_index))
        if not np.isfinite(event_time):
            raise ValueError("DHEM representative time must be finite")
        return event_time

    @classmethod
    def _initialize_event(cls, item: MemoryItem) -> MemoryItem:
        return dataclasses.replace(
            item,
            metadata={
                **dict(item.metadata),
                "accumulated_mass": cls._mass(item),
                "representative_time": cls._event_time(item),
            },
        )

    @staticmethod
    def _cosine_similarity(left: MemoryItem, right: MemoryItem) -> float:
        if left.tokens.shape != right.tokens.shape or not np.array_equal(
            left.mask, right.mask
        ):
            raise ValueError("DHEM event maintenance requires equal stored layouts")
        left_vector = left.tokens[left.mask].reshape(-1)
        right_vector = right.tokens[right.mask].reshape(-1)
        denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denominator == 0:
            return 1.0 if np.array_equal(left_vector, right_vector) else 0.0
        return float(
            np.clip(np.dot(left_vector, right_vector) / denominator, -1.0, 1.0)
        )

    def _redundancy(self, left: MemoryItem, right: MemoryItem) -> float:
        semantic = (1.0 + self._cosine_similarity(left, right)) / 2.0
        time_distance = abs(self._event_time(left) - self._event_time(right))
        return float(semantic * np.exp(-time_distance / self.temporal_decay))

    def _merge(self, left: MemoryItem, right: MemoryItem) -> MemoryItem:
        left_mass = self._mass(left)
        right_mass = self._mass(right)
        merged_mass = left_mass + right_mass
        merged_time = (
            left_mass * self._event_time(left) + right_mass * self._event_time(right)
        ) / merged_mass
        merged = MemoryItem(
            item_id=f"{right.episode_id}:{right.path_name}:dhem-merged:{self._merge_count}",
            path_name=right.path_name,
            episode_id=right.episode_id,
            step_index=right.step_index,
            phase=right.phase,
            tokens=(left_mass * left.tokens + right_mass * right.tokens) / merged_mass,
            mask=right.mask.copy(),
            metadata={
                **dict(right.metadata),
                "accumulated_mass": merged_mass,
                "representative_time": merged_time,
            },
        )
        self._merge_count += 1
        return merged

    def write(self, item: MemoryItem) -> Mapping[str, Any]:
        incoming = self._initialize_event(item)
        if len(self._items) < self.capacity:
            self._items.append(incoming)
            return {"maintenance_action": "append", "retained": True}

        pair_indices = range(1, len(self._items) - 2)
        pair_redundancies = [
            self._redundancy(self._items[index], self._items[index + 1])
            for index in pair_indices
        ]
        merge_offset = int(np.argmax(np.asarray(pair_redundancies)))
        merge_index = merge_offset + 1
        historical_redundancy = pair_redundancies[merge_offset]
        incoming_redundancy = self._redundancy(self._items[-1], incoming)
        if incoming_redundancy >= historical_redundancy:
            return {
                "maintenance_action": "discard_incoming",
                "retained": False,
                "incoming_redundancy": incoming_redundancy,
                "historical_redundancy": historical_redundancy,
                "latest_item_id": self._items[-1].item_id,
            }

        left = self._items[merge_index]
        right = self._items[merge_index + 1]
        merged = self._merge(left, right)
        self._items[merge_index : merge_index + 2] = [merged]
        self._items.append(incoming)
        return {
            "maintenance_action": "merge_history_and_append",
            "retained": True,
            "merged_item_ids": [left.item_id, right.item_id],
            "result_item_id": merged.item_id,
            "incoming_redundancy": incoming_redundancy,
            "historical_redundancy": historical_redundancy,
            "accumulated_mass": self._mass(merged),
            "representative_time": self._event_time(merged),
        }

    def items(self) -> tuple[MemoryItem, ...]:
        return tuple(self._items)

    def reset(self) -> None:
        self._items.clear()
        self._merge_count = 0


class Retriever(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, step: MemoryStep, store: Store) -> RetrievalResult:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear retriever-local state at an episode or lifecycle boundary."""
        return None


class AllRetriever(Retriever):
    def retrieve(self, step: MemoryStep, store: Store) -> RetrievalResult:
        del step
        items = store.items()
        return RetrievalResult(
            items,
            {"strategy": "all", "candidate_count": len(items)},
        )


class LatestRetriever(Retriever):
    def __init__(self, *, max_items: int):
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = int(max_items)

    def retrieve(self, step: MemoryStep, store: Store) -> RetrievalResult:
        del step
        candidates = store.items()
        items = candidates[-self.max_items :]
        return RetrievalResult(
            items,
            {
                "strategy": "latest",
                "candidate_count": len(candidates),
                "max_items": self.max_items,
            },
        )


class CompletedPhaseMeanRetriever(Retriever):
    """Pool each completed contiguous phase into one causal handoff token.

    The current phase label is deployment-observable planner output.  A trailing
    segment with that label is still active and is excluded; at the first step
    after a phase transition, every stored segment is complete.  Repeated phase
    labels remain separate when another phase occurred between them.

    This is a transparent, parameter-free lower bound for learned subtask-event
    pooling such as WeaveLA, not a reproduction of its learned query pooling or
    action-side injection.
    """

    def __init__(self, *, max_items: int):
        if max_items <= 0:
            raise ValueError("completed phase max_items must be positive")
        self.max_items = int(max_items)

    @staticmethod
    def _summary(segment: Sequence[MemoryItem], *, handoff_phase: str) -> MemoryItem:
        first = segment[0]
        last = segment[-1]
        token_blocks: list[np.ndarray] = []
        for item in segment:
            if item.episode_id != first.episode_id:
                raise ValueError("completed phase segment cannot cross episodes")
            if item.path_name != first.path_name:
                raise ValueError("completed phase segment cannot cross memory paths")
            if item.phase != first.phase:
                raise ValueError("completed phase segment must have one phase label")
            valid = np.asarray(item.tokens, dtype=np.float32)[
                np.asarray(item.mask, dtype=np.bool_)
            ]
            if not len(valid):
                raise ValueError("completed phase retrieval requires valid tokens")
            token_blocks.append(valid)
        tokens = np.concatenate(token_blocks, axis=0)
        if not np.isfinite(tokens).all():
            raise ValueError("completed phase retrieval requires finite tokens")
        pooled = tokens.mean(axis=0, dtype=np.float32)[None, :]
        return MemoryItem(
            item_id=(
                f"{first.episode_id}:{first.path_name}:completed-phase:"
                f"{first.step_index}-{last.step_index}"
            ),
            path_name=first.path_name,
            episode_id=first.episode_id,
            step_index=last.step_index,
            phase=first.phase,
            tokens=pooled,
            mask=np.ones((1,), dtype=np.bool_),
            metadata={
                "summary_kind": "completed_contiguous_phase_mean",
                "source_item_count": len(segment),
                "source_token_count": int(tokens.shape[0]),
                "summary_start_step": first.step_index,
                "summary_end_step": last.step_index,
                "completion_phase": first.phase,
                "handoff_to_phase": handoff_phase,
            },
        )

    def retrieve(self, step: MemoryStep, store: Store) -> RetrievalResult:
        if not step.phase.strip():
            raise ValueError(
                "completed phase retrieval requires non-empty deployment phase labels"
            )
        candidates = store.items()
        segments: list[list[MemoryItem]] = []
        previous_step = -1
        for item in candidates:
            if item.episode_id != step.episode_id:
                raise ValueError("completed phase retrieval cannot cross episodes")
            if item.step_index >= step.step_index:
                raise ValueError("completed phase retrieval can use only prior items")
            if item.step_index <= previous_step:
                raise ValueError(
                    "completed phase retrieval requires strictly ordered history"
                )
            if not item.phase.strip():
                raise ValueError(
                    "completed phase retrieval requires non-empty stored phase labels"
                )
            if not segments or segments[-1][-1].phase != item.phase:
                segments.append([item])
            else:
                segments[-1].append(item)
            previous_step = item.step_index

        active_segment_excluded = bool(
            segments and segments[-1][-1].phase == step.phase
        )
        completed = segments[:-1] if active_segment_excluded else segments
        selected_segments = completed[-self.max_items :]
        summaries = tuple(
            self._summary(segment, handoff_phase=step.phase)
            for segment in selected_segments
        )
        return RetrievalResult(
            summaries,
            {
                "strategy": "completed_phase_mean",
                "candidate_count": len(candidates),
                "segment_count": len(segments),
                "completed_segment_count": len(completed),
                "active_segment_excluded": active_segment_excluded,
                "current_phase": step.phase,
                "max_items": self.max_items,
                "selected": [
                    {
                        "item_id": item.item_id,
                        "completion_phase": item.phase,
                        "summary_start_step": item.metadata["summary_start_step"],
                        "summary_end_step": item.metadata["summary_end_step"],
                        "source_item_count": item.metadata["source_item_count"],
                    }
                    for item in summaries
                ],
            },
        )


class ContentRecencyRetriever(Retriever):
    """Select past items by current-latent similarity with a frame-gap penalty.

    This is a representation-level, parameter-free retrieval baseline.  It is
    deliberately separate from TempoFit's layer-wise K/V retrieval: both use
    content plus recency, but this operator acts on the harness's contextual
    memory items and can therefore be tested without modifying the backbone.
    """

    def __init__(self, *, max_items: int, recency_penalty: float):
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if not np.isfinite(recency_penalty) or recency_penalty < 0:
            raise ValueError("recency_penalty must be finite and non-negative")
        self.max_items = int(max_items)
        self.recency_penalty = float(recency_penalty)

    @staticmethod
    def _pooled(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = np.asarray(tokens, dtype=np.float32)[np.asarray(mask, dtype=np.bool_)]
        if not len(valid):
            raise ValueError("content-recency retrieval requires valid tokens")
        pooled = valid.mean(axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(pooled))
        if not np.isfinite(norm):
            raise ValueError("content-recency retrieval requires finite tokens")
        # Define cosine similarity with a zero placeholder as zero.  This makes
        # the operator fall back to its explicit recency term during cold-start
        # and deterministic smoke tests instead of introducing a hidden special
        # case in the runtime.
        if norm == 0:
            return np.zeros_like(pooled)
        return pooled / norm

    def retrieve(self, step: MemoryStep, store: Store) -> RetrievalResult:
        items = store.items()
        if not items:
            return RetrievalResult(
                (),
                {
                    "strategy": "content_recency",
                    "candidate_count": 0,
                    "max_items": self.max_items,
                    "recency_penalty": self.recency_penalty,
                    "selected": [],
                },
            )
        if step.source_tokens is None or step.source_mask is None:
            raise ValueError("content_recency retriever requires current source tokens")
        query = self._pooled(step.source_tokens, step.source_mask)
        scored: list[tuple[float, int, float, int, MemoryItem]] = []
        for item in items:
            if item.episode_id != step.episode_id:
                raise ValueError("content_recency cannot retrieve across episodes")
            frame_gap = step.step_index - item.step_index
            if frame_gap <= 0:
                raise ValueError("content_recency can retrieve only prior items")
            key = self._pooled(item.tokens, item.mask)
            similarity = float(np.dot(query, key))
            score = similarity - self.recency_penalty * frame_gap
            scored.append((score, item.step_index, similarity, frame_gap, item))
        selected_by_score = sorted(scored, key=lambda row: (-row[0], -row[1]))[
            : self.max_items
        ]
        # Mem-0 slot positions encode relative order, so restore chronology
        # after ranking instead of ordering the payload by similarity.
        selected = sorted(selected_by_score, key=lambda row: row[1])
        return RetrievalResult(
            tuple(row[4] for row in selected),
            {
                "strategy": "content_recency",
                "candidate_count": len(items),
                "max_items": self.max_items,
                "recency_penalty": self.recency_penalty,
                "selected": [
                    {
                        "item_id": row[4].item_id,
                        "similarity": row[2],
                        "frame_gap": row[3],
                        "score": row[0],
                    }
                    for row in selected
                ],
            },
        )


class SemanticRecentUnionRetriever(Retriever):
    """Combine a semantic quota with a guaranteed, disjoint recent tail.

    This isolates OnEvoMemory's retrieval rule from its learned value writer:
    semantic matches preserve older relevant evidence, while an explicit recent
    branch prevents content retrieval from dropping all local motion context.
    After measuring substantial branch overlap on RMBench, the implementation
    backfills the semantic quota after deduplication so comparisons use the same
    token budget as a dense sliding window whenever enough history exists.
    """

    def __init__(self, *, semantic_items: int, recent_items: int):
        if semantic_items <= 0 or recent_items <= 0:
            raise ValueError("semantic_items and recent_items must be positive")
        self.semantic_items = int(semantic_items)
        self.recent_items = int(recent_items)

    @staticmethod
    def _pooled(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = np.asarray(tokens, dtype=np.float32)[np.asarray(mask, dtype=np.bool_)]
        if not len(valid):
            raise ValueError("semantic-recent retrieval requires valid tokens")
        pooled = valid.mean(axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(pooled))
        if not np.isfinite(norm):
            raise ValueError("semantic-recent retrieval requires finite tokens")
        return np.zeros_like(pooled) if norm == 0 else pooled / norm

    @classmethod
    def _pooled_items(cls, items: Sequence[MemoryItem]) -> np.ndarray:
        pooled = np.stack(
            [
                np.asarray(item.tokens, dtype=np.float32)[
                    np.asarray(item.mask, dtype=np.bool_)
                ].mean(axis=0, dtype=np.float32)
                for item in items
            ]
        )
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        if not np.isfinite(norms).all():
            raise ValueError("semantic-recent retrieval requires finite tokens")
        return np.divide(
            pooled,
            norms,
            out=np.zeros_like(pooled),
            where=norms > 0,
        )

    def retrieve(self, step: MemoryStep, store: Store) -> RetrievalResult:
        items = store.items()
        if not items:
            return RetrievalResult(
                (),
                {
                    "strategy": "semantic_recent_union",
                    "candidate_count": 0,
                    "semantic_items": self.semantic_items,
                    "recent_items": self.recent_items,
                    "initial_branch_overlap_count": 0,
                    "selected": [],
                },
            )
        if step.source_tokens is None or step.source_mask is None:
            raise ValueError(
                "semantic_recent_union retriever requires current source tokens"
            )
        query = self._pooled(step.source_tokens, step.source_mask)
        keys = self._pooled_items(items)
        similarities = keys @ query
        scored: list[tuple[float, int, int, MemoryItem]] = []
        for item, similarity in zip(items, similarities, strict=True):
            if item.episode_id != step.episode_id:
                raise ValueError(
                    "semantic_recent_union cannot retrieve across episodes"
                )
            frame_gap = step.step_index - item.step_index
            if frame_gap <= 0:
                raise ValueError("semantic_recent_union can retrieve only prior items")
            scored.append((float(similarity), item.step_index, frame_gap, item))

        semantic_ranking = sorted(scored, key=lambda row: (-row[0], -row[1]))
        recent = sorted(scored, key=lambda row: -row[1])[: self.recent_items]
        recent_ids = {row[3].item_id for row in recent}
        initial_semantic_ids = {
            row[3].item_id for row in semantic_ranking[: self.semantic_items]
        }
        semantic = [
            row for row in semantic_ranking if row[3].item_id not in recent_ids
        ][: self.semantic_items]
        semantic_ids = {row[3].item_id for row in semantic}
        selected_by_id = {row[3].item_id: row for row in (*semantic, *recent)}
        selected = sorted(selected_by_id.values(), key=lambda row: row[1])
        return RetrievalResult(
            tuple(row[3] for row in selected),
            {
                "strategy": "semantic_recent_union",
                "candidate_count": len(items),
                "semantic_items": self.semantic_items,
                "recent_items": self.recent_items,
                "initial_branch_overlap_count": len(initial_semantic_ids & recent_ids),
                "selected_count": len(selected),
                "selected": [
                    {
                        "item_id": row[3].item_id,
                        "similarity": row[0],
                        "frame_gap": row[2],
                        "selected_by_semantic": row[3].item_id in semantic_ids,
                        "selected_by_recent": row[3].item_id in recent_ids,
                    }
                    for row in selected
                ],
            },
        )


class BoundaryChunkRetriever(Retriever):
    """Retrieve one coherent history chunk instead of scattered similar items.

    Adjacent retained items are split when their cosine similarity falls below
    ``boundary_similarity_threshold``.  Chunks are scored by the maximum
    current-query similarity of any item in the chunk, then the winning chunk
    is uniformly sampled under ``max_items``.  This is a contextual-token lower
    bound inspired by RoboMME-Interference's visual session retrieval; it does
    not claim to reproduce that method's separately pretrained SigLIP encoder.
    """

    def __init__(
        self,
        *,
        max_items: int,
        boundary_similarity_threshold: float,
        min_chunk_items: int = 1,
    ) -> None:
        if max_items <= 0:
            raise ValueError("boundary chunk max_items must be positive")
        if (
            not np.isfinite(boundary_similarity_threshold)
            or not -1 <= boundary_similarity_threshold <= 1
        ):
            raise ValueError(
                "boundary_similarity_threshold must be finite and in [-1, 1]"
            )
        if min_chunk_items <= 0:
            raise ValueError("min_chunk_items must be positive")
        self.max_items = int(max_items)
        self.boundary_similarity_threshold = float(boundary_similarity_threshold)
        self.min_chunk_items = int(min_chunk_items)
        self._cached_item_ids: tuple[str, ...] = ()
        self._cached_object_ids: tuple[int, ...] = ()
        self._key_buffer: np.ndarray | None = None
        self._key_buffer_length = 0

    def reset(self) -> None:
        self._cached_item_ids = ()
        self._cached_object_ids = ()
        self._key_buffer = None
        self._key_buffer_length = 0

    @staticmethod
    def _pooled(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = np.asarray(tokens, dtype=np.float32)[np.asarray(mask, dtype=np.bool_)]
        if not len(valid):
            raise ValueError("boundary chunk retrieval requires valid tokens")
        pooled = valid.mean(axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(pooled))
        if not np.isfinite(norm):
            raise ValueError("boundary chunk retrieval requires finite tokens")
        return np.zeros_like(pooled) if norm == 0 else pooled / norm

    def _pooled_items(self, items: Sequence[MemoryItem]) -> np.ndarray:
        item_ids = tuple(item.item_id for item in items)
        object_ids = tuple(id(item) for item in items)
        cached_length = len(self._cached_item_ids)
        extends_cache = (
            len(items) >= cached_length
            and item_ids[:cached_length] == self._cached_item_ids
            and object_ids[:cached_length] == self._cached_object_ids
        )
        start = cached_length if extends_cache else 0
        if not extends_cache:
            self._key_buffer_length = 0

        first_key = (
            self._pooled(items[start].tokens, items[start].mask)
            if start < len(items)
            else None
        )
        if first_key is not None:
            required_capacity = len(items)
            current_capacity = 0 if self._key_buffer is None else len(self._key_buffer)
            wrong_width = (
                self._key_buffer is not None
                and self._key_buffer.shape[1] != first_key.shape[0]
            )
            if wrong_width or current_capacity < required_capacity:
                new_capacity = max(required_capacity, max(current_capacity * 2, 32))
                replacement = np.empty(
                    (new_capacity, first_key.shape[0]), dtype=np.float32
                )
                if extends_cache and self._key_buffer is not None and cached_length:
                    replacement[:cached_length] = self._key_buffer[:cached_length]
                self._key_buffer = replacement
            if self._key_buffer is None:
                raise RuntimeError("boundary chunk key buffer was not initialized")
            self._key_buffer[start] = first_key
            for index in range(start + 1, len(items)):
                item = items[index]
                self._key_buffer[index] = self._pooled(item.tokens, item.mask)

        self._cached_item_ids = item_ids
        self._cached_object_ids = object_ids
        self._key_buffer_length = len(items)
        if self._key_buffer is None:
            raise RuntimeError("boundary chunk cannot pool an empty item sequence")
        return self._key_buffer[: self._key_buffer_length]

    def retrieve(self, step: MemoryStep, store: Store) -> RetrievalResult:
        items = store.items()
        empty_details = {
            "strategy": "boundary_chunk",
            "candidate_count": len(items),
            "max_items": self.max_items,
            "boundary_similarity_threshold": self.boundary_similarity_threshold,
            "min_chunk_items": self.min_chunk_items,
        }
        if not items:
            return RetrievalResult(
                (),
                {
                    **empty_details,
                    "boundary_count": 0,
                    "chunk_count": 0,
                    "eligible_chunk_count": 0,
                    "selected_count": 0,
                    "selected": [],
                },
            )
        if step.source_tokens is None or step.source_mask is None:
            raise ValueError("boundary_chunk retriever requires current source tokens")
        for item in items:
            if item.episode_id != step.episode_id:
                raise ValueError("boundary_chunk cannot retrieve across episodes")
            if step.step_index - item.step_index <= 0:
                raise ValueError("boundary_chunk can retrieve only prior items")

        query = self._pooled(step.source_tokens, step.source_mask)
        keys = self._pooled_items(items)
        if keys.shape[1] != query.shape[0]:
            raise ValueError(
                "boundary_chunk requires matching query/key widths, "
                f"got {query.shape[0]} and {keys.shape[1]}"
            )
        adjacent_similarities = np.sum(keys[:-1] * keys[1:], axis=1)
        cut_positions = [
            index + 1
            for index, similarity in enumerate(adjacent_similarities)
            if float(similarity) < self.boundary_similarity_threshold
        ]
        edges = [0, *cut_positions, len(items)]
        chunks = [(edges[index], edges[index + 1]) for index in range(len(edges) - 1)]
        eligible = [
            chunk for chunk in chunks if chunk[1] - chunk[0] >= self.min_chunk_items
        ]
        used_minimum_fallback = False
        if not eligible:
            eligible = [(0, len(items))]
            used_minimum_fallback = True

        query_similarities = keys @ query
        scored_chunks = [
            (
                float(np.max(query_similarities[start:stop])),
                stop,
                start,
            )
            for start, stop in eligible
        ]
        selected_score, selected_stop, selected_start = max(
            scored_chunks,
            key=lambda row: (row[0], row[1]),
        )
        selected_chunk = items[selected_start:selected_stop]
        if len(selected_chunk) <= self.max_items:
            selected_items = selected_chunk
            sampling = "all_in_chunk"
        else:
            positions = np.linspace(
                0,
                len(selected_chunk) - 1,
                num=self.max_items,
                dtype=np.int64,
            )
            selected_items = tuple(
                selected_chunk[int(position)] for position in positions
            )
            sampling = "uniform_in_chunk"
        item_index_by_id = {item.item_id: index for index, item in enumerate(items)}

        return RetrievalResult(
            tuple(selected_items),
            {
                **empty_details,
                "boundary_count": len(cut_positions),
                "chunk_count": len(chunks),
                "eligible_chunk_count": len(eligible),
                "used_minimum_fallback": used_minimum_fallback,
                "selected_chunk": {
                    "start_item_id": selected_chunk[0].item_id,
                    "end_item_id": selected_chunk[-1].item_id,
                    "start_step_index": selected_chunk[0].step_index,
                    "end_step_index": selected_chunk[-1].step_index,
                    "item_count": len(selected_chunk),
                    "max_query_similarity": selected_score,
                },
                "sampling": sampling,
                "selected_count": len(selected_items),
                "boundaries": [
                    {
                        "left_item_id": items[position - 1].item_id,
                        "right_item_id": items[position].item_id,
                        "adjacent_similarity": float(
                            adjacent_similarities[position - 1]
                        ),
                    }
                    for position in cut_positions
                ],
                "selected": [
                    {
                        "item_id": item.item_id,
                        "frame_gap": step.step_index - item.step_index,
                        "query_similarity": float(
                            query_similarities[item_index_by_id[item.item_id]]
                        ),
                    }
                    for item in selected_items
                ],
            },
        )


class UniformGlobalRetriever(Retriever):
    """Evenly cover the complete causal history under a fixed item budget.

    This isolates RoboMME's FrameSamp selection principle while retaining the
    harness's one-contextual-token-per-moment representation and Mem-0
    utilizer.  It is therefore a selection lower bound, not MME-VLA's learned
    visual-patch encoder or memory-as-modulator implementation.
    """

    def __init__(self, *, max_items: int, exclude_recent_items: int = 0) -> None:
        if max_items <= 0:
            raise ValueError("uniform global max_items must be positive")
        if (
            not isinstance(exclude_recent_items, int)
            or isinstance(exclude_recent_items, bool)
            or exclude_recent_items < 0
        ):
            raise ValueError("exclude_recent_items must be a non-negative integer")
        self.max_items = int(max_items)
        self.exclude_recent_items = exclude_recent_items

    def retrieve(self, step: MemoryStep, store: Store) -> RetrievalResult:
        all_candidates = store.items()
        candidates = (
            all_candidates
            if self.exclude_recent_items == 0
            else all_candidates[: -self.exclude_recent_items]
        )
        rows: list[tuple[int, int, MemoryItem]] = []
        for ordinal, item in enumerate(candidates):
            if item.episode_id != step.episode_id:
                raise ValueError("uniform global cannot retrieve across episodes")
            frame_gap = step.step_index - item.step_index
            if frame_gap <= 0:
                raise ValueError("uniform global can retrieve only prior items")
            rows.append((ordinal, frame_gap, item))

        if len(rows) <= self.max_items:
            selected_rows = rows
            selected_by = "warmup_all"
        else:
            positions = np.linspace(
                0,
                len(rows) - 1,
                num=self.max_items,
                dtype=np.int64,
            )
            selected_rows = [rows[int(position)] for position in positions]
            selected_by = "global_uniform"

        denominator = max(len(rows) - 1, 1)
        return RetrievalResult(
            tuple(row[2] for row in selected_rows),
            {
                "strategy": "uniform_global",
                "candidate_count": len(rows),
                "excluded_recent_item_count": len(all_candidates) - len(candidates),
                "max_items": self.max_items,
                "selected_count": len(selected_rows),
                "selected": [
                    {
                        "item_id": row[2].item_id,
                        "frame_gap": row[1],
                        "selected_by": selected_by,
                        "normalized_history_position": row[0] / denominator,
                    }
                    for row in selected_rows
                ],
            },
        )


class TemporalMultiscaleRetriever(Retriever):
    """Select raw moments at exponential and uniformly global time scales.

    The exponential branch guarantees dense temporal resolution near the
    current step.  The disjoint global branch then fills the remaining budget
    across the complete retained history.  This is a deterministic lower bound
    inspired by CycleManip's cost-aware visual sampling, not its full learned
    proprioceptive-history encoder or progress objective.
    """

    def __init__(self, *, max_items: int, exponential_items: int) -> None:
        if max_items <= 0:
            raise ValueError("temporal multiscale max_items must be positive")
        if exponential_items <= 0:
            raise ValueError("temporal multiscale exponential_items must be positive")
        if exponential_items >= max_items:
            raise ValueError(
                "temporal multiscale exponential_items must be less than max_items"
            )
        self.max_items = int(max_items)
        self.exponential_items = int(exponential_items)

    def retrieve(self, step: MemoryStep, store: Store) -> RetrievalResult:
        candidates = store.items()
        rows: list[tuple[int, int, MemoryItem]] = []
        for ordinal, item in enumerate(candidates):
            if item.episode_id != step.episode_id:
                raise ValueError("temporal multiscale cannot retrieve across episodes")
            frame_gap = step.step_index - item.step_index
            if frame_gap <= 0:
                raise ValueError("temporal multiscale can retrieve only prior items")
            rows.append((ordinal, frame_gap, item))

        if len(rows) <= self.max_items:
            return RetrievalResult(
                tuple(row[2] for row in rows),
                {
                    "strategy": "temporal_multiscale",
                    "candidate_count": len(rows),
                    "max_items": self.max_items,
                    "exponential_items": self.exponential_items,
                    "selected_count": len(rows),
                    "selected": [
                        {
                            "item_id": row[2].item_id,
                            "frame_gap": row[1],
                            "selected_by": "warmup_all",
                        }
                        for row in rows
                    ],
                },
            )

        selected_by_id: dict[str, tuple[int, int, MemoryItem, str, int | None]] = {}
        max_gap = max(row[1] for row in rows)
        for exponent in range(self.exponential_items):
            target_gap = 1 << exponent
            if target_gap > max_gap:
                break
            available = [row for row in rows if row[2].item_id not in selected_by_id]
            if not available:
                break
            selected = min(
                available,
                key=lambda row: (
                    abs(row[1] - target_gap),
                    -row[2].step_index,
                ),
            )
            selected_by_id[selected[2].item_id] = (
                *selected,
                "exponential",
                target_gap,
            )

        remaining = [row for row in rows if row[2].item_id not in selected_by_id]
        remaining_budget = min(self.max_items - len(selected_by_id), len(remaining))
        if remaining_budget:
            positions = np.linspace(
                0,
                len(remaining) - 1,
                num=remaining_budget,
                dtype=np.int64,
            )
            for position in positions:
                selected = remaining[int(position)]
                selected_by_id[selected[2].item_id] = (
                    *selected,
                    "global_uniform",
                    None,
                )

        selected_rows = sorted(selected_by_id.values(), key=lambda row: row[0])
        if len(selected_rows) != self.max_items:
            raise RuntimeError(
                "temporal multiscale failed to fill its retrieval budget: "
                f"expected {self.max_items}, got {len(selected_rows)}"
            )
        return RetrievalResult(
            tuple(row[2] for row in selected_rows),
            {
                "strategy": "temporal_multiscale",
                "candidate_count": len(rows),
                "max_items": self.max_items,
                "exponential_items": self.exponential_items,
                "selected_count": len(selected_rows),
                "selected": [
                    {
                        "item_id": row[2].item_id,
                        "frame_gap": row[1],
                        "selected_by": row[3],
                        **({"target_gap": row[4]} if row[4] is not None else {}),
                    }
                    for row in selected_rows
                ],
            },
        )


class Lifecycle(abc.ABC):
    @abc.abstractmethod
    def before_step(self, step: MemoryStep, store: Store) -> bool:
        """Return True when this call reset the store."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class Controller(abc.ABC):
    @abc.abstractmethod
    def select(self, step: MemoryStep, path_names: Sequence[str]) -> tuple[str, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class AllPathsController(Controller):
    """Fixed baseline controller; activates every path declared by the program."""

    def select(self, step: MemoryStep, path_names: Sequence[str]) -> tuple[str, ...]:
        del step
        return tuple(path_names)

    def reset(self) -> None:
        return None


class EpisodeLifecycle(Lifecycle):
    def before_step(self, step: MemoryStep, store: Store) -> bool:
        del step, store
        return False

    def reset(self) -> None:
        return None


class PhaseLifecycle(Lifecycle):
    def __init__(self) -> None:
        self._phase: str | None = None

    def before_step(self, step: MemoryStep, store: Store) -> bool:
        if self._phase is None:
            self._phase = step.phase
            return False
        if step.phase == self._phase:
            return False
        store.reset()
        self._phase = step.phase
        return True

    def reset(self) -> None:
        self._phase = None


class Utilizer(abc.ABC):
    @abc.abstractmethod
    def validate_paths(self, path_names: Sequence[str]) -> None:
        """Reject program paths that this utilizer cannot consume."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply(
        self,
        observation: Mapping[str, Any],
        items: Sequence[MemoryItem],
    ) -> UtilizationResult:
        raise NotImplementedError


class NoMemoryUtilizer(Utilizer):
    def validate_paths(self, path_names: Sequence[str]) -> None:
        if path_names:
            raise ValueError("none utilizer cannot consume memory paths")

    def apply(self, observation, items):
        if items:
            raise ValueError("none utilizer cannot receive memory items")
        return UtilizationResult(observation, (), 0)


class TokenUtilizer(Utilizer):
    def __init__(self, *, token_budget: int):
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.token_budget = int(token_budget)

    def validate_paths(self, path_names: Sequence[str]) -> None:
        if not path_names:
            raise ValueError("memory_tokens utilizer requires at least one path")

    def apply(self, observation, items):
        if not items:
            return UtilizationResult(observation, (), 0)
        token_blocks: list[np.ndarray] = []
        used_items: list[MemoryItem] = []
        path_usage: dict[str, dict[str, int]] = {}
        remaining = self.token_budget
        embed_dim: int | None = None
        for item in items:
            valid = item.tokens[item.mask]
            if embed_dim is None:
                embed_dim = int(valid.shape[1])
            elif valid.shape[1] != embed_dim:
                raise ValueError(
                    "all retrieved memory items must share one embedding width"
                )
            take = min(remaining, len(valid))
            if take:
                token_blocks.append(valid[:take])
                used_items.append(item)
                usage = path_usage.setdefault(
                    item.path_name, {"item_count": 0, "token_count": 0}
                )
                usage["item_count"] += 1
                usage["token_count"] += take
                remaining -= take
            if remaining == 0:
                break
        if not token_blocks:
            return UtilizationResult(observation, (), 0)
        tokens = np.concatenate(token_blocks, axis=0).astype(np.float32, copy=False)
        output = dict(observation)
        output["memory_tokens"] = tokens
        output["memory_mask"] = np.ones((tokens.shape[0],), dtype=np.bool_)
        used_ids = {item.item_id for item in used_items}
        return UtilizationResult(
            output,
            tuple(used_items),
            int(tokens.shape[0]),
            {
                "path_usage": path_usage,
                "dropped_item_ids": [
                    item.item_id for item in items if item.item_id not in used_ids
                ],
            },
        )


class Mem0ContextUtilizer(Utilizer):
    """Pack Mem-0 anchor and sliding latents into a fixed model-side layout.

    Slot 0 is the subtask anchor.  The remaining slots are the sliding window,
    right-aligned from oldest to newest.  Right alignment makes the slot-derived
    relative positions match Mem-0: 1 is the most recent item and K is the
    oldest currently retained item.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        sliding_window_size: int,
        anchor_path: str | None,
        history_path_quotas: Mapping[str, int],
    ):
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive")
        self.embed_dim = int(embed_dim)
        self.sliding_window_size = int(sliding_window_size)
        self.token_budget = 1 + self.sliding_window_size
        if anchor_path is not None and (
            not isinstance(anchor_path, str) or not anchor_path
        ):
            raise ValueError("anchor_path must be null or a non-empty string")
        if not isinstance(history_path_quotas, Mapping):
            raise ValueError("history_path_quotas must be a mapping")
        self.anchor_path = anchor_path
        self.history_path_quotas = dict(history_path_quotas)
        if any(
            not isinstance(name, str) or not name for name in self.history_path_quotas
        ):
            raise ValueError("history_path_quotas keys must be non-empty strings")
        if any(
            not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0
            for quota in self.history_path_quotas.values()
        ):
            raise ValueError("history_path_quotas values must be positive integers")
        allocated = sum(self.history_path_quotas.values())
        if allocated not in (0, self.sliding_window_size):
            raise ValueError(
                "history_path_quotas must be empty or allocate the complete "
                f"sliding window ({self.sliding_window_size}), got {allocated}"
            )
        if self.anchor_path in self.history_path_quotas:
            raise ValueError("anchor_path and history_path_quotas must be disjoint")

    def validate_paths(self, path_names: Sequence[str]) -> None:
        configured = set(self.history_path_quotas)
        if self.anchor_path is not None:
            configured.add(self.anchor_path)
        declared = set(path_names)
        if len(path_names) != len(declared):
            raise ValueError("program path names must be unique")
        if configured != declared:
            raise ValueError(
                "mem0_context path roles must cover every declared path exactly: "
                f"missing_roles={sorted(declared - configured)}, "
                f"unknown_roles={sorted(configured - declared)}"
            )

    def apply(self, observation, items):
        output = dict(observation)
        tokens = np.zeros((self.token_budget, self.embed_dim), dtype=np.float32)
        mask = np.zeros((self.token_budget,), dtype=np.bool_)

        anchor_items = [item for item in items if item.path_name == self.anchor_path]
        history_items = {
            path_name: [item for item in items if item.path_name == path_name]
            for path_name in self.history_path_quotas
        }
        accepted_paths = set(self.history_path_quotas)
        if self.anchor_path is not None:
            accepted_paths.add(self.anchor_path)
        unknown_paths = {item.path_name for item in items} - accepted_paths
        if unknown_paths:
            raise ValueError(
                "Mem-0 context received paths outside its configured roles: "
                f"{sorted(unknown_paths)}"
            )
        if len(anchor_items) > 1:
            raise ValueError("Mem-0 context accepts at most one anchor item")

        def one_latent(item: MemoryItem) -> np.ndarray:
            valid = item.tokens[item.mask]
            if valid.shape != (1, self.embed_dim):
                raise ValueError(
                    "Mem-0 stores exactly one contextual image latent per step; "
                    f"expected (1, {self.embed_dim}), got {valid.shape}"
                )
            return valid[0]

        if anchor_items:
            tokens[0] = one_latent(anchor_items[0])
            mask[0] = True

        sliding_items: list[MemoryItem] = []
        path_usage: dict[str, dict[str, int]] = {}
        for path_name, quota in self.history_path_quotas.items():
            candidates = history_items[path_name]
            selected = candidates[-quota:]
            sliding_items.extend(selected)
            path_usage[path_name] = {
                "quota": quota,
                "retrieved_item_count": len(candidates),
                "used_item_count": len(selected),
                "dropped_item_count": len(candidates) - len(selected),
            }
        sliding_start = self.token_budget - len(sliding_items)
        for slot, item in enumerate(sliding_items, start=sliding_start):
            tokens[slot] = one_latent(item)
            mask[slot] = True

        output["memory_tokens"] = tokens
        output["memory_mask"] = mask
        used_items = tuple([*anchor_items, *sliding_items])
        used_ids = {item.item_id for item in used_items}
        return UtilizationResult(
            output,
            used_items,
            int(mask.sum()),
            {
                "path_usage": path_usage,
                "dropped_item_ids": [
                    item.item_id for item in items if item.item_id not in used_ids
                ],
            },
        )


def reject_preinjected_memory(observation: Mapping[str, Any]) -> None:
    present = MEMORY_INPUT_KEYS.intersection(observation)
    if present:
        raise ValueError(
            "policy observation already contains memory inputs; the harness must be the only owner: "
            + ", ".join(sorted(present))
        )
