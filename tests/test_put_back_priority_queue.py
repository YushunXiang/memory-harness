from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_put_back_priority_queue.sh"
)


def test_priority_queue_materializes_readiness_before_completion_marker() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    comparison_positions = [
        source.index("put_back_block_empty_arch_vs_full_budget_matched_gate3.json"),
        source.index("put_back_block_native_vs_full_budget_matched_gate3.json"),
        source.index("put_back_block_native_vs_empty_arch_budget_matched_gate3.json"),
    ]
    decision_position = source.index("-m memory_harness.decide_executor_readiness")
    output_position = source.index(
        "--output /tmp/put_back_block_executor_readiness_u1200_gate3.json"
    )
    completion_position = source.index("echo NATIVE_BASELINE_COMPLETE")

    assert max(comparison_positions) < decision_position
    assert decision_position < output_position < completion_position
