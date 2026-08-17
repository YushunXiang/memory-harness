from __future__ import annotations

from types import SimpleNamespace

import pytest

from memory_harness.official_mem0_ablation import (
    install_executor_memory_ablation,
    restore_executor_memory_ablation,
)


class FakeMemoryBank:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object, int]] = []

    def update_on_eval(self, new_vector, text_vector, classifier, episode_id):
        self.calls.append((new_vector, text_vector, classifier, episode_id))
        return "sliding-memory", "anchor-memory", True


def make_model() -> SimpleNamespace:
    return SimpleNamespace(executor=SimpleNamespace(memory_bank=FakeMemoryBank()))


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("without_anchor", ("sliding-memory", "current", True)),
        ("without_sliding", ("current", "anchor-memory", True)),
        ("without_both", ("current", "current", True)),
    ],
)
def test_ablation_masks_only_selected_output_and_preserves_update(condition, expected):
    model = make_model()
    bank = model.executor.memory_bank
    original = install_executor_memory_ablation(model, condition)

    result = bank.update_on_eval("current", "text", "classifier", 7)

    assert result == expected
    assert bank.calls == [("current", "text", "classifier", 7)]
    restore_executor_memory_ablation(model, original)
    assert bank.update_on_eval("current-2", "text", "classifier", 8) == (
        "sliding-memory",
        "anchor-memory",
        True,
    )


def test_full_is_a_noop() -> None:
    model = make_model()
    original_method = model.executor.memory_bank.update_on_eval

    original = install_executor_memory_ablation(model, "full")

    assert original is None
    assert model.executor.memory_bank.update_on_eval == original_method


def test_rejects_unknown_or_nested_ablation() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="Unknown executor memory ablation"):
        install_executor_memory_ablation(model, "unknown")

    install_executor_memory_ablation(model, "without_anchor")
    with pytest.raises(RuntimeError, match="already installed"):
        install_executor_memory_ablation(model, "without_sliding")
