from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "memory-harness"
    / "scripts"
    / "continue_cover_blocks_corrected_executor_gate.sh"
)


def _run_branch(tmp_path: Path, *, observable_signal: bool) -> str:
    signal = tmp_path / "signal.json"
    signal.write_text(
        json.dumps(
            {
                "schema_version": "memory_harness.executor_run_signal/v2",
                "evidence_scope": "executor_skill_diagnostic_only",
                "observable_executor_signal": observable_signal,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    native_checkpoint = tmp_path / "native"
    native_checkpoint.mkdir()
    base_checkpoint = tmp_path / "base"
    (base_checkpoint / "params").mkdir(parents=True)
    context_root = tmp_path / "contexts"
    context_root.mkdir()
    task_template = context_root / "task_template.json"
    task_template.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPT), str(signal), str(native_checkpoint)],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "PYTHON": sys.executable,
            "BASE_CHECKPOINT": str(base_checkpoint),
            "CONTEXT_ROOT": str(context_root),
            "TASK_TEMPLATE": str(task_template),
            "CHECKPOINT_BASE_DIR": str(tmp_path / "checkpoints"),
            "RUN_ROOT": str(tmp_path / "runs"),
            "DATE_TAG": "test",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_signal_branch_trains_empty_executor_then_runs_key_pair(tmp_path: Path) -> None:
    output = _run_branch(tmp_path, observable_signal=True)

    assert "training_empty_mem0.json" in output
    assert "emac_cover_blocks_subtask_empty_mem0_u1200" in output
    assert "--condition no_key" in output
    assert "--condition key" in output
    assert "--evidence-kind matched_training" in output
    assert "--num-episodes 17" in output
    assert "--seed-start 100003" in output
    assert "--policy-seed-base 120003" in output
    assert "no_key_vs_key_u1200_gate20" in output
    assert "--num-episodes 30" in output
    assert "--seed-start 100020" in output
    assert "--policy-seed-base 120020" in output
    assert "no_key_vs_key_u1200_gate50" in output
    assert "native_none_plus1800" not in output


def test_floor_branch_extends_native_before_budget_matched_key_pair(
    tmp_path: Path,
) -> None:
    output = _run_branch(tmp_path, observable_signal=False)

    native_position = output.index("native_none_plus1800_to_u3000")
    empty_position = output.index("empty_mem0_u3000")
    planner_position = output.index("--condition no_key")
    assert native_position < empty_position < planner_position
    assert "OPTIMIZER_UPDATES=1800" in output
    assert "ORACLE_SUBTASK_DIAGNOSTIC=1" in output
    assert "--condition key" in output
    assert "--num-episodes 17" in output
    assert "no_key_vs_key_u3000_gate20" in output
    assert "--num-episodes 30" in output
    assert "no_key_vs_key_u3000_gate50" in output
