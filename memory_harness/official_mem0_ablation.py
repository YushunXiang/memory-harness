from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal


ExecutorMemoryAblation = Literal[
    "full",
    "without_anchor",
    "without_sliding",
    "without_both",
]

VALID_EXECUTOR_MEMORY_ABLATIONS: frozenset[str] = frozenset(
    {"full", "without_anchor", "without_sliding", "without_both"}
)


def install_executor_memory_ablation(
    model: Any,
    condition: ExecutorMemoryAblation,
) -> Callable[..., Any] | None:
    """Mask released Mem-0 memory outputs while preserving its state updates.

    This is an inference intervention on a shared trained checkpoint, not an
    exact replay of the paper's underspecified ablation protocol. Returning the
    current visual token removes the selected memory residual from the action
    model input. The original ``update_on_eval`` is still called first so that
    write, lifecycle, and classifier state remain identical across conditions.

    Returns the original bound method so callers can restore it after a run.
    """
    if condition not in VALID_EXECUTOR_MEMORY_ABLATIONS:
        raise ValueError(
            f"Unknown executor memory ablation {condition!r}; expected one of "
            f"{sorted(VALID_EXECUTOR_MEMORY_ABLATIONS)}"
        )
    if condition == "full":
        return None

    try:
        memory_bank = model.executor.memory_bank
        original_update = memory_bank.update_on_eval
    except AttributeError as exc:
        raise TypeError(
            "Expected a Mem-0 model with executor.memory_bank.update_on_eval"
        ) from exc

    if getattr(memory_bank, "_memory_harness_ablation", None) is not None:
        raise RuntimeError("An executor memory ablation is already installed")

    def update_with_ablation(
        new_vector: Any,
        text_vector: Any,
        classifier: Any,
        episode_id: int,
    ) -> tuple[Any, Any, bool]:
        sliding, anchor, subtask_end = original_update(
            new_vector,
            text_vector,
            classifier,
            episode_id,
        )
        if condition in {"without_sliding", "without_both"}:
            sliding = new_vector
        if condition in {"without_anchor", "without_both"}:
            anchor = new_vector
        return sliding, anchor, subtask_end

    memory_bank.update_on_eval = update_with_ablation
    memory_bank._memory_harness_ablation = condition
    return original_update


def restore_executor_memory_ablation(model: Any, original_update: Callable[..., Any] | None) -> None:
    if original_update is None:
        return
    memory_bank = model.executor.memory_bank
    memory_bank.update_on_eval = original_update
    del memory_bank._memory_harness_ablation
