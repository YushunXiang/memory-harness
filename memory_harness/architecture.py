from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from memory_harness.config import load_program_spec
from memory_harness.contracts import AuditEvent, MemoryStep, StepResult
from memory_harness.key_planner import CompletedSubtask
from memory_harness.key_planner import CurrentObservationPlannerContext
from memory_harness.key_planner import KeyPlannerMemory, PlannerContext
from memory_harness.registry import build_program
from memory_harness.runtime import AuditSink, MemoryProgram


@dataclasses.dataclass(frozen=True)
class ArchitectureSpec:
    """Compose typed planner and executor memory without conflating payloads."""

    name: str
    executor_program: pathlib.Path
    planner: str
    planner_memory: str
    planner_model: str | None

    @classmethod
    def load(cls, path: pathlib.Path | str) -> ArchitectureSpec:
        config_path = pathlib.Path(path)
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "name",
            "executor_program",
            "planner",
            "planner_memory",
            "planner_model",
        }
        if set(raw) != expected:
            raise ValueError(
                "architecture keys mismatch: "
                f"missing={sorted(expected - set(raw))}, unknown={sorted(set(raw) - expected)}"
            )
        if raw["schema_version"] != 3:
            raise ValueError("unsupported architecture schema_version")
        if not isinstance(raw["name"], str) or not raw["name"]:
            raise ValueError("architecture name must be non-empty")
        if raw["planner"] not in {"none", "mem0"}:
            raise ValueError("planner must be 'none' or 'mem0'")
        if raw["planner_memory"] not in {"none", "key"}:
            raise ValueError("planner_memory must be 'none' or 'key'")
        if raw["planner_memory"] == "key" and raw["planner"] != "mem0":
            raise ValueError("key planner memory requires planner='mem0'")
        planner_model = raw["planner_model"]
        if raw["planner"] == "mem0":
            if not isinstance(planner_model, str) or not planner_model.strip():
                raise ValueError("planner='mem0' requires a non-empty planner_model")
        elif planner_model is not None:
            raise ValueError("planner_model must be null when planner='none'")
        executor_path = (config_path.parent / raw["executor_program"]).resolve()
        if not executor_path.is_file():
            raise FileNotFoundError(f"executor program not found: {executor_path}")
        return cls(
            name=raw["name"],
            executor_program=executor_path,
            planner=raw["planner"],
            planner_memory=raw["planner_memory"],
            planner_model=planner_model,
        )


class MemoryArchitecture:
    """One facade for planner-side key and executor-side latent memory modules."""

    def __init__(
        self,
        *,
        name: str,
        executor: MemoryProgram,
        planner_enabled: bool,
        key_memory: KeyPlannerMemory | None,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self.name = name
        self.executor = executor
        self.planner_enabled = planner_enabled
        self.key_memory = key_memory
        self.audit_sink = audit_sink
        self._episode_id: str | None = None
        self._global_task: str | None = None
        self._planner_step_index = 0

    @property
    def active_modules(self) -> tuple[str, ...]:
        modules = [path.name for path in self.executor.paths]
        if self.key_memory is not None:
            modules.append("key")
        return tuple(modules)

    def reset_episode(
        self,
        *,
        episode_id: str,
        global_task: str,
        initial_image: np.ndarray,
    ) -> None:
        self._episode_id = episode_id
        self._planner_step_index = 0
        self.executor.reset(episode_id=episode_id)
        self._global_task = global_task
        if self.key_memory is not None:
            self.key_memory.reset(global_task=global_task, initial_image=initial_image)
        if self.planner_enabled:
            self._audit(
                "RESET",
                details={
                    "target": "planner",
                    "module": self._planner_module,
                    "memory_enabled": self.key_memory is not None,
                },
            )

    def reset_planner_episode(
        self,
        *,
        episode_id: str,
        global_task: str,
        initial_image: np.ndarray,
    ) -> None:
        """Reset only planner state when the executor wrapper owns its reset."""
        if not self.planner_enabled:
            raise RuntimeError("architecture has no planner")
        self._episode_id = episode_id
        self._planner_step_index = 0
        self._global_task = global_task
        if self.key_memory is not None:
            self.key_memory.reset(global_task=global_task, initial_image=initial_image)
        self._audit(
            "RESET",
            details={
                "target": "planner",
                "module": self._planner_module,
                "memory_enabled": self.key_memory is not None,
            },
        )

    def executor_step(
        self,
        observation: Mapping[str, Any],
        step: MemoryStep,
    ) -> StepResult:
        self._require_episode()
        return self.executor.step(observation, step)

    def record_completed_subtask(
        self,
        *,
        instruction: str,
        end_image: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> CompletedSubtask | None:
        self._require_episode()
        if not self.planner_enabled:
            raise RuntimeError("architecture has no planner")
        if self.key_memory is None:
            self._planner_step_index += 1
            return None
        record = self.key_memory.write_completed_subtask(
            instruction=instruction,
            end_image=end_image,
            metadata=metadata,
        )
        self._audit(
            "WRITE",
            item_ids=(f"{self._episode_id}:key:{record.ordinal}",),
            details={"target": "planner", "module": "key", "instruction": instruction},
        )
        self._planner_step_index += 1
        return record

    def planner_context(self, *, current_image: np.ndarray) -> PlannerContext | None:
        self._require_episode()
        if not self.planner_enabled:
            return None
        if self.key_memory is None:
            if self._global_task is None:
                raise RuntimeError("planner architecture has not been reset")
            context: PlannerContext = CurrentObservationPlannerContext(
                global_task=self._global_task,
                current_image=current_image,
            )
            item_ids: tuple[str, ...] = ()
        else:
            context = self.key_memory.context()
            item_ids = tuple(
                f"{self._episode_id}:key:{record.ordinal}"
                for record in context.completed_subtasks
            )
        self._audit(
            "USE",
            item_ids=item_ids,
            details={
                "target": "planner",
                "module": self._planner_module,
                "memory_enabled": self.key_memory is not None,
            },
        )
        return context

    def record_planner_call(
        self,
        *,
        instruction: str,
        raw_answer: str,
        latency_seconds: float,
        boundary_source: str,
    ) -> None:
        self._require_episode()
        if not self.planner_enabled:
            raise RuntimeError("architecture has no planner")
        self._audit(
            "PLAN",
            details={
                "target": "planner",
                "module": self._planner_module,
                "instruction": instruction,
                "raw_answer": raw_answer,
                "latency_seconds": float(latency_seconds),
                "boundary_source": boundary_source,
            },
        )

    def _require_episode(self) -> None:
        if self._episode_id is None:
            raise RuntimeError("reset_episode() must be called first")

    @property
    def _planner_module(self) -> str:
        return "key" if self.key_memory is not None else "none"

    def _audit(
        self,
        event: str,
        *,
        item_ids: tuple[str, ...] = (),
        details: Mapping[str, Any],
    ) -> None:
        if self.audit_sink is None:
            return
        assert self._episode_id is not None
        self.audit_sink.write(
            (
                AuditEvent(
                    event=event,
                    episode_id=self._episode_id,
                    step_index=self._planner_step_index,
                    program_name=self.name,
                    path_name="key" if self.key_memory is not None else None,
                    item_ids=item_ids,
                    details=details,
                ),
            )
        )


def build_architecture(
    spec: ArchitectureSpec,
    *,
    audit_sink: AuditSink | None = None,
) -> MemoryArchitecture:
    executor = build_program(
        load_program_spec(spec.executor_program),
        audit_sink=audit_sink,
    )
    planner_enabled = spec.planner == "mem0"
    key_memory = KeyPlannerMemory() if spec.planner_memory == "key" else None
    return MemoryArchitecture(
        name=spec.name,
        executor=executor,
        planner_enabled=planner_enabled,
        key_memory=key_memory,
        audit_sink=audit_sink,
    )
