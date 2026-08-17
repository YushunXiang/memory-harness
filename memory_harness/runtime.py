from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from memory_harness.components import Encoder
from memory_harness.components import Controller
from memory_harness.components import Lifecycle
from memory_harness.components import Retriever
from memory_harness.components import Store
from memory_harness.components import Utilizer
from memory_harness.components import WriteRule
from memory_harness.components import reject_preinjected_memory
from memory_harness.contracts import AuditEvent
from memory_harness.contracts import EpisodeOutcome
from memory_harness.contracts import MemoryItem
from memory_harness.contracts import MemoryStep
from memory_harness.contracts import StepResult


DEPLOYABLE_STEP_METADATA_KEYS = frozenset(
    {
        "task_text_present",
        "training_representation",
    }
)


class AuditSink(Protocol):
    def write(self, events: Sequence[AuditEvent]) -> None: ...


@dataclasses.dataclass
class MemoryPath:
    name: str
    encoder: Encoder
    writer: WriteRule
    store: Store
    retriever: Retriever
    lifecycle: Lifecycle


class MemoryProgram:
    """Runs a configured memory graph without owning policy or benchmark logic."""

    def __init__(
        self,
        *,
        name: str,
        deployable: bool,
        paths: Sequence[MemoryPath],
        controller: Controller,
        utilizer: Utilizer,
        audit_sink: AuditSink | None = None,
    ) -> None:
        if not name:
            raise ValueError("program name must be non-empty")
        self.name = name
        self.deployable = bool(deployable)
        self.paths = tuple(paths)
        self.controller = controller
        self.utilizer = utilizer
        self.audit_sink = audit_sink
        self._episode_id: str | None = None
        self._last_step = -1
        self._episode_finished = True

    def reset(self, *, episode_id: str) -> tuple[AuditEvent, ...]:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self._episode_finished and any(
            path.store.requires_episode_outcome for path in self.paths
        ):
            raise RuntimeError(
                "finish_episode() is required before resetting a persistent memory program"
            )
        store_details: dict[str, Mapping[str, Any]] = {}
        for path in self.paths:
            store_details[path.name] = path.store.begin_episode(episode_id)
            path.writer.reset()
            path.retriever.reset()
            path.lifecycle.reset()
        self.controller.reset()
        self._episode_id = episode_id
        self._last_step = -1
        self._episode_finished = False
        events = (
            AuditEvent(
                event="RESET",
                episode_id=episode_id,
                step_index=0,
                program_name=self.name,
                details={
                    "scope": "episode",
                    "path_count": len(self.paths),
                    "stores": {
                        name: dict(details)
                        for name, details in store_details.items()
                    },
                },
            ),
        )
        self._write_audit(events)
        return events

    def step(self, observation: Mapping[str, Any], step: MemoryStep) -> StepResult:
        if self._episode_id is None:
            raise RuntimeError("reset() must be called before step()")
        if self._episode_finished:
            raise RuntimeError("reset() must be called after finish_episode()")
        if step.episode_id != self._episode_id:
            raise ValueError(
                f"step episode {step.episode_id!r} does not match active episode {self._episode_id!r}"
            )
        if self._last_step == -1 and step.step_index != 0:
            raise ValueError("the first step after reset must have step_index=0")
        if step.step_index <= self._last_step:
            raise ValueError("step_index must increase strictly within an episode")
        if self.deployable:
            undeclared = set(step.metadata) - DEPLOYABLE_STEP_METADATA_KEYS
            if undeclared:
                raise ValueError(
                    "deployable memory program received undeclared metadata: "
                    + ", ".join(sorted(undeclared))
                )
        reject_preinjected_memory(observation)

        path_names = tuple(path.name for path in self.paths)
        selected_names = self.controller.select(step, path_names)
        if len(selected_names) != len(set(selected_names)):
            raise ValueError("controller selected duplicate memory paths")
        unknown_paths = set(selected_names) - set(path_names)
        if unknown_paths:
            raise ValueError(
                f"controller selected unknown memory paths: {sorted(unknown_paths)}"
            )
        selected = set(selected_names)
        active_paths = tuple(path for path in self.paths if path.name in selected)
        events: list[AuditEvent] = [
            AuditEvent(
                event="SELECT",
                episode_id=step.episode_id,
                step_index=step.step_index,
                program_name=self.name,
                details={"active_paths": list(selected_names)},
            )
        ]
        # Lifecycle is clocked by the environment, not by controller selection.
        # Otherwise a path disabled across an entire phase can be re-enabled
        # with memory that silently survived that phase transition.
        for path in self.paths:
            if path.lifecycle.before_step(step, path.store):
                path.writer.reset()
                path.retriever.reset()
                events.append(
                    AuditEvent(
                        event="RESET",
                        episode_id=step.episode_id,
                        step_index=step.step_index,
                        program_name=self.name,
                        path_name=path.name,
                        details={"scope": "lifecycle", "phase": step.phase},
                    )
                )
        retrieved: list[MemoryItem] = []
        for path in active_paths:
            retrieval = path.retriever.retrieve(step, path.store)
            retrieved_items = retrieval.items
            retrieved.extend(retrieved_items)
            events.append(
                AuditEvent(
                    event="RETRIEVE",
                    episode_id=step.episode_id,
                    step_index=step.step_index,
                    program_name=self.name,
                    path_name=path.name,
                    item_ids=tuple(item.item_id for item in retrieved_items),
                    details={"phase": step.phase, **dict(retrieval.details)},
                )
            )

        stored_before_write = sum(len(path.store.items()) for path in self.paths)
        utilization = self.utilizer.apply(observation, retrieved)
        events.append(
            AuditEvent(
                event="USE",
                episode_id=step.episode_id,
                step_index=step.step_index,
                program_name=self.name,
                item_ids=tuple(item.item_id for item in utilization.used_items),
                details={
                    "applied": utilization.used_token_count > 0,
                    "token_count": utilization.used_token_count,
                    "stored_item_count_before_write": stored_before_write,
                    **dict(utilization.details),
                },
            )
        )

        for path in active_paths:
            decision = path.writer.decide(step, path.store)
            events.append(
                AuditEvent(
                    event="WRITE_DECISION",
                    episode_id=step.episode_id,
                    step_index=step.step_index,
                    program_name=self.name,
                    path_name=path.name,
                    details={
                        "write": decision.write,
                        "source_step_index": (
                            step.step_index
                            if decision.write_step is None
                            else decision.write_step.step_index
                        ),
                        **dict(decision.details),
                    },
                )
            )
            if not decision.write:
                continue
            write_step = step if decision.write_step is None else decision.write_step
            if write_step.episode_id != step.episode_id:
                raise ValueError("write_step must belong to the active episode")
            if write_step.step_index > step.step_index:
                raise ValueError("write_step cannot refer to a future step")
            if self.deployable:
                undeclared = (
                    set(write_step.metadata) - DEPLOYABLE_STEP_METADATA_KEYS
                )
                if undeclared:
                    raise ValueError(
                        "deployable memory program received undeclared write payload: "
                        + ", ".join(sorted(undeclared))
                    )
            item = path.encoder.encode(write_step, path_name=path.name)
            store_details = path.store.write(item)
            path_stored_item_count = len(path.store.items())
            total_stored_item_count = sum(
                len(candidate.store.items()) for candidate in self.paths
            )
            events.append(
                AuditEvent(
                    event="WRITE",
                    episode_id=step.episode_id,
                    step_index=step.step_index,
                    program_name=self.name,
                    path_name=path.name,
                    item_ids=(item.item_id,),
                    details={
                        "phase": write_step.phase,
                        "source_step_index": write_step.step_index,
                        "confirmation_delay_steps": (
                            step.step_index - write_step.step_index
                        ),
                        "token_count": item.valid_token_count,
                        "path_stored_item_count": path_stored_item_count,
                        "total_stored_item_count": total_stored_item_count,
                        **dict(store_details),
                    },
                )
            )

        self._last_step = step.step_index
        result = StepResult(
            observation=utilization.observation,
            events=tuple(events),
            retrieved_item_ids=tuple(item.item_id for item in retrieved),
            used_token_count=utilization.used_token_count,
            stored_item_count=sum(len(path.store.items()) for path in self.paths),
        )
        self._write_audit(result.events)
        return result

    def finish_episode(self, outcome: EpisodeOutcome) -> tuple[AuditEvent, ...]:
        """Finalize transactional stores from a deployment-observable outcome."""
        if self._episode_id is None or self._episode_finished:
            raise RuntimeError("there is no active episode to finish")
        if outcome.episode_id != self._episode_id:
            raise ValueError(
                f"outcome episode {outcome.episode_id!r} does not match "
                f"active episode {self._episode_id!r}"
            )
        if self._last_step < 0:
            raise RuntimeError("cannot finish an episode before its first step")
        if outcome.final_step_index != self._last_step:
            raise ValueError(
                "outcome final_step_index must equal the last memory step: "
                f"{outcome.final_step_index} != {self._last_step}"
            )
        events: list[AuditEvent] = [
            AuditEvent(
                event="EPISODE_OUTCOME",
                episode_id=outcome.episode_id,
                step_index=outcome.final_step_index,
                program_name=self.name,
                details={
                    "success": outcome.success,
                    "total_reward": outcome.total_reward,
                    **dict(outcome.metadata),
                },
            )
        ]
        for path in self.paths:
            details = path.store.finish_episode(outcome)
            events.append(
                AuditEvent(
                    event="STORE_FINALIZE",
                    episode_id=outcome.episode_id,
                    step_index=outcome.final_step_index,
                    program_name=self.name,
                    path_name=path.name,
                    details=dict(details),
                )
            )
        self._episode_finished = True
        result = tuple(events)
        self._write_audit(result)
        return result

    def _write_audit(self, events: Sequence[AuditEvent]) -> None:
        if self.audit_sink is not None:
            self.audit_sink.write(events)
