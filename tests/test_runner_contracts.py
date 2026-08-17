from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_runner_separates_policy_and_validator_runtime_order() -> None:
    source = (ROOT / "scripts" / "run_fixed_pi05_rmbench.sh").read_text(
        encoding="utf-8"
    )

    assert 'export PYTHONPATH="$OUT_DIR/runtime:$HARNESS_ROOT:$PROJECT_ROOT' in source
    assert 'PYTHONPATH="$HARNESS_ROOT:$OUT_DIR/runtime:$PROJECT_ROOT' in source
    assert source.index(
        'export PYTHONPATH="$OUT_DIR/runtime:$HARNESS_ROOT:$PROJECT_ROOT'
    ) < source.index('bash "$PROJECT_ROOT/run_rmbench_baseline_local.sh"')
    assert source.index(
        'PYTHONPATH="$HARNESS_ROOT:$OUT_DIR/runtime:$PROJECT_ROOT'
    ) > source.index('bash "$PROJECT_ROOT/run_rmbench_baseline_local.sh"')


def test_rollout_runners_enable_audited_task_trace() -> None:
    for script_name in (
        "run_clean_pi05_rmbench.sh",
        "run_fixed_pi05_rmbench.sh",
    ):
        source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "export TASK_STATE_TRACE_FREQUENCY=10" in source
        assert "memory_harness.put_back_progress" in source
        assert '"$OUT_DIR/subtask_summary.json"' in source


def test_put_back_replay_queue_runs_before_cover_blocks() -> None:
    source = (ROOT / "scripts" / "queue_put_back_replay_then_cover.sh").read_text(
        encoding="utf-8"
    )

    assert source.index('replay full_memory "$FULL_RUN"') < source.index(
        "memory_harness.summarize_put_back_replays"
    )
    assert source.index("memory_harness.summarize_put_back_replays") < source.index(
        'run_cover_blocks_corrected_executor_gate.sh"'
    )
    assert "while kill -0 \"$WAIT_PID\"" in source
