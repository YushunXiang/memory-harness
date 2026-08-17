from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

import numpy as np

from memory_harness.architecture import ArchitectureSpec, build_architecture
from memory_harness.audit import JsonlAuditSink
from memory_harness.components import Mem0ContextUtilizer
from memory_harness.contracts import MemoryStep
from memory_harness.contracts import EpisodeOutcome
from memory_harness.planner_policy import Mem0PlannerPolicy
from memory_harness.planner_policy import OpenAICompatiblePlannerBackend
from memory_harness.planner_policy import PlannerBackend
from memory_harness.policy import MemoryHarnessPolicy
from memory_harness.runtime import MemoryProgram


def _openpi_types():
    try:
        from openpi.memory import EpisodicMemoryBank
        from openpi.memory import LatentRetriever
        from openpi.memory import MemoryConfig
        from openpi.memory import RetrieverRuntimeState
        from openpi.models import model as model_lib
        from openpi.policies.memory_policy import MemoryPolicy
        from openpi.policies.memory_policy import build_moment_tokens_from_request
        from openpi.shared import nnx_utils
    except ImportError as exc:
        raise RuntimeError(
            "OpenPI integration requires openpi-libero/src on PYTHONPATH"
        ) from exc
    return {
        "EpisodicMemoryBank": EpisodicMemoryBank,
        "LatentRetriever": LatentRetriever,
        "MemoryConfig": MemoryConfig,
        "RetrieverRuntimeState": RetrieverRuntimeState,
        "Observation": model_lib.Observation,
        "MemoryPolicy": MemoryPolicy,
        "build_moment_tokens_from_request": build_moment_tokens_from_request,
        "nnx_utils": nnx_utils,
    }


def _program_token_budget(program: MemoryProgram) -> int:
    budget = getattr(program.utilizer, "token_budget", None)
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError(
            "memory-enabled programs require a positive token utilizer budget"
        )
    return budget


def _is_mem0_program(program: MemoryProgram) -> bool:
    return isinstance(program.utilizer, Mem0ContextUtilizer)


def _empty_mem0_context_shape(base_policy: Any) -> tuple[int, int] | None:
    """Return the static no-memory tensor shape required by a Mem-0 checkpoint."""
    model = getattr(base_policy, "_model", None)
    if model is None or not bool(getattr(model, "_memory_enabled", False)):
        return None
    if getattr(model, "_memory_utilization_mode", None) != "mem0":
        return None
    fusion = getattr(model, "mem0_fusion", None)
    if fusion is None:
        raise ValueError("Mem-0 checkpoint is missing its fusion module")
    return 1 + int(fusion.sliding_window_size), int(fusion.hidden_size)


def _validate_program_model_contract(
    base_policy: Any, program: MemoryProgram
) -> None:
    if not _is_mem0_program(program):
        return
    expected = _empty_mem0_context_shape(base_policy)
    if expected is None:
        raise ValueError("Mem-0 programs require a Mem-0-enabled policy model")
    actual = (program.utilizer.token_budget, program.utilizer.embed_dim)
    if actual != expected:
        raise ValueError(
            "memory program/model context shape mismatch: "
            f"program={actual}, model={expected}"
        )


def _with_prompt_memory_hints(observation: dict[str, Any]) -> dict[str, Any]:
    """Use the deployable planner/executor prompt as the default subtask phase."""
    if "memory_phase_label" in observation and "memory_task_text" in observation:
        return observation
    prompt = observation.get("prompt")
    if prompt is None:
        return observation
    prompt_value = np.asarray(prompt)
    if prompt_value.size != 1:
        raise ValueError("policy prompt must be a scalar string")
    prompt_text = str(prompt_value.reshape(()).item()).strip()
    if not prompt_text:
        return observation
    output = dict(observation)
    output.setdefault("memory_phase_label", prompt_text)
    output.setdefault("memory_task_text", prompt_text)
    return output


def _fixed_memory_policy_class(memory_policy_base: type):
    class OpenPIFixedMemoryPolicy(memory_policy_base):
        """Reuse OpenPI's query encoder and memory-token action path with a harness program."""

        uses_memory_runtime = True

        @classmethod
        def from_policy(cls, base_policy: Any, program: MemoryProgram):
            types = _openpi_types()
            if bool(getattr(base_policy, "_is_pytorch_model", False)):
                raise ValueError(
                    "fixed latent memory currently requires the JAX π0.5 policy"
                )
            model = getattr(base_policy, "_model", None)
            if model is None or not bool(getattr(model, "_memory_enabled", False)):
                raise ValueError(
                    "memory programs require a π0.5 context-adapter checkpoint with memory.enabled=True"
                )

            token_budget = _program_token_budget(program)
            memory_config = types["MemoryConfig"](
                enabled=True,
                topk=1,
                coarse_topk=1,
                token_budget=token_budget,
                tokens_per_item=token_budget,
            )
            instance = cls.__new__(cls)
            instance.__dict__.update(base_policy.__dict__)
            instance._bank = types["EpisodicMemoryBank"]()
            instance._memory_config = memory_config
            instance._retrieval_mode = "moment_only"
            instance._oracle_observation_key = "memory_oracle_item_ids"
            instance._retrieval_policy = None
            instance._learned_retrieval_deterministic = True
            instance._retriever = types["LatentRetriever"](
                instance._bank, memory_config
            )
            instance._encode_memory_query = types["nnx_utils"].module_jit(
                model.encode_memory_query
            )
            query_features = getattr(model, "encode_memory_query_features", None)
            instance._encode_memory_query_features = (
                None
                if query_features is None
                else types["nnx_utils"].module_jit(query_features)
            )
            if _is_mem0_program(program):
                mem0_features = getattr(model, "encode_mem0_features", None)
                if mem0_features is None:
                    raise ValueError(
                        "Mem-0 programs require a π0.5 model with encode_mem0_features()"
                    )
                instance._encode_memory_query_features = types["nnx_utils"].module_jit(
                    mem0_features
                )
            instance._encode_critical_moment_features = None
            instance._retriever_state = types["RetrieverRuntimeState"]()
            instance._step_index = 0
            instance._cached_memory_tokens = None
            instance._cached_memory_mask = None
            instance._cached_episodic_tokens = None
            instance._cached_episodic_mask = None
            instance._cached_item_ids = ()
            instance._moment_history = collections.deque(maxlen=1)
            instance._harness_program = program
            instance._harness_episode_index = -1
            instance._harness_context = None
            instance._harness_last_result = None
            instance._harness_embed_dim = None
            instance._build_moment_tokens = types["build_moment_tokens_from_request"]
            return instance

        def reset_history(self) -> None:
            super().reset_history()
            self._harness_episode_index += 1
            self._harness_context = None
            self._harness_last_result = None
            self._harness_embed_dim = None
            self._harness_program.reset(
                episode_id=f"episode-{self._harness_episode_index}"
            )

        def _append_moment_context(self, request) -> None:
            if _is_mem0_program(self._harness_program):
                source_tokens = np.asarray(request.prefix_embedding, dtype=np.float32)[
                    None, :
                ]
            else:
                source_tokens = self._build_moment_tokens(request, tokens_per_step=4)
            source_mask = np.ones((source_tokens.shape[0],), dtype=np.bool_)
            self._harness_embed_dim = int(source_tokens.shape[1])
            result = self._harness_program.step(
                {},
                MemoryStep(
                    episode_id=f"episode-{self._harness_episode_index}",
                    step_index=self._step_index,
                    phase=str(request.phase_label or ""),
                    source_tokens=source_tokens,
                    source_mask=source_mask,
                    robot_state=request.current_state,
                    metadata={"task_text_present": bool(request.task_text)},
                ),
            )
            self._harness_last_result = result
            if "memory_tokens" in result.observation:
                self._harness_context = (
                    np.asarray(result.observation["memory_tokens"], dtype=np.float32),
                    np.asarray(result.observation["memory_mask"], dtype=np.bool_),
                )
            else:
                self._harness_context = None

        def _moment_context(self):
            return self._harness_context

        def _memory_embed_dim(self) -> int:
            if self._harness_embed_dim is None:
                raise ValueError(
                    "memory embedding width is unavailable before the first query"
                )
            return self._harness_embed_dim

        def infer(self, obs: dict, *, noise=None) -> dict:
            outputs = super().infer(_with_prompt_memory_hints(obs), noise=noise)
            result = self._harness_last_result
            outputs["memory"] = {
                **outputs.get("memory", {}),
                "program": self._harness_program.name,
                "retrieved_item_ids": (
                    () if result is None else result.retrieved_item_ids
                ),
                "used_token_count": (0 if result is None else result.used_token_count),
                "stored_item_count": (
                    0 if result is None else result.stored_item_count
                ),
                "fixed_memory_program_enabled": True,
            }
            return outputs

        def observe(self, obs: dict) -> dict[str, Any]:
            """Advance moment memory without sampling a new action chunk.

            Mem-0 updates anchor/sliding on every environment observation, while
            the action policy may execute several cached actions between queries.
            This keeps memory time measured in environment steps rather than
            action-query intervals.
            """
            import jax
            import jax.numpy as jnp

            types = _openpi_types()
            hinted = _with_prompt_memory_hints(obs)
            filtered, retrieval_hints = self._extract_retrieval_hints(hinted)
            inputs = jax.tree.map(lambda value: value, filtered)
            inputs = self._pre_history_transform(inputs)
            if self._history_buffer is not None:
                inputs = self._history_buffer.get_temporal_observation(inputs)
                self._history_buffer.push(jax.tree.map(lambda value: value, inputs))
            inputs = self._post_history_transform(inputs)
            batched = jax.tree.map(
                lambda value: jnp.asarray(value)[np.newaxis, ...], inputs
            )
            observation = types["Observation"].from_dict(batched)
            request = self._build_retrieval_request(observation, retrieval_hints)
            self._append_moment_context(request)
            self._refresh_cached_memory_context()
            self._step_index += 1
            result = self._harness_last_result
            return {
                "program": self._harness_program.name,
                "retrieved_item_ids": (
                    () if result is None else result.retrieved_item_ids
                ),
                "used_token_count": (0 if result is None else result.used_token_count),
                "stored_item_count": (
                    0 if result is None else result.stored_item_count
                ),
            }

        def finish_episode(self, *, success: bool, total_reward: float = 0.0) -> None:
            if self._harness_episode_index < 0 or self._step_index <= 0:
                raise RuntimeError(
                    "cannot finish before an episode has produced a memory step"
                )
            self._harness_program.finish_episode(
                EpisodeOutcome(
                    episode_id=f"episode-{self._harness_episode_index}",
                    success=bool(success),
                    final_step_index=self._step_index - 1,
                    total_reward=float(total_reward),
                )
            )

    return OpenPIFixedMemoryPolicy


def _wrap_executor_program(
    base_policy: Any,
    *,
    program: MemoryProgram,
) -> Any:
    _validate_program_model_contract(base_policy, program)
    if not program.paths:
        return MemoryHarnessPolicy(
            base_policy,
            program,
            empty_context_shape=_empty_mem0_context_shape(base_policy),
        )
    types = _openpi_types()
    policy_class = _fixed_memory_policy_class(types["MemoryPolicy"])
    return policy_class.from_policy(base_policy, program)


def wrap_openpi_architecture(
    base_policy: Any,
    *,
    architecture_config: Path | str,
    audit_log: Path | str,
    global_task: str,
    planner_base_url: str = "http://127.0.0.1:8123/v1",
    planner_model: str = "mem0-cover-blocks-key-planner",
    planner_timeout_seconds: float = 3600.0,
    planner_seed_base: int = 130_000,
    planner_boundary_mode: str = "oracle_prompt_change",
    planner_backend: PlannerBackend | None = None,
) -> Any:
    sink = JsonlAuditSink(audit_log)
    spec = ArchitectureSpec.load(architecture_config)
    architecture = build_architecture(
        spec,
        audit_sink=sink,
    )
    policy = _wrap_executor_program(base_policy, program=architecture.executor)
    if not architecture.planner_enabled:
        return policy
    if planner_model != spec.planner_model:
        raise ValueError(
            f"planner model mismatch: architecture requires {spec.planner_model!r}, "
            f"runtime requested {planner_model!r}"
        )
    backend = planner_backend or OpenAICompatiblePlannerBackend(
        base_url=planner_base_url,
        model=planner_model,
        timeout_seconds=planner_timeout_seconds,
    )
    return Mem0PlannerPolicy(
        policy,
        architecture=architecture,
        planner_backend=backend,
        global_task=global_task,
        planner_seed_base=planner_seed_base,
        boundary_mode=planner_boundary_mode,
    )
