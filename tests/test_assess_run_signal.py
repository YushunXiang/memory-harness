from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_harness.assess_run_signal import assess_run_signal


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _run(
    tmp_path: Path,
    rewards: list[float],
    *,
    manifest: dict | None = None,
    task_progress: list[float] | None = None,
) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "summary.json", {"status": "completed", "num_episodes": len(rewards)})
    _write_json(
        run / "emac_manifest.json",
        manifest or {"architecture": "anchor_sliding"},
    )
    (run / "episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "success": reward >= 1.0,
                    "total_reward": reward * 10,
                    "final_info": {"max_reward": reward},
                    **(
                        {
                            "task_progress": {
                                "task": "put_back_block",
                                "max_progress_score": task_progress[index],
                            }
                        }
                        if task_progress is not None
                        else {}
                    ),
                }
            )
            + "\n"
            for index, reward in enumerate(rewards)
        ),
        encoding="utf-8",
    )
    return run


def test_stage_reward_is_an_executor_signal_but_never_enables_controller(
    tmp_path: Path,
) -> None:
    result = assess_run_signal(_run(tmp_path, [0.0, 0.1, 0.0]))

    assert result["observable_executor_signal"] is True
    assert result["decision"]["next_action"] == "budget_match_controls_before_comparison"
    assert result["decision"]["controller_training_allowed"] is False


def test_all_zero_run_remains_at_floor(tmp_path: Path) -> None:
    result = assess_run_signal(_run(tmp_path, [0.0, 0.0]))

    assert result["observable_executor_signal"] is False
    assert result["decision"]["next_action"] == "full_memory_still_at_floor"


def test_sparse_reward_task_progress_is_an_executor_signal(tmp_path: Path) -> None:
    result = assess_run_signal(
        _run(tmp_path, [0.0, 0.0], task_progress=[0.0, 1.0])
    )

    assert result["max_observed_task_progress"] == 1.0
    assert result["subtask_evaluation"]["subtask_metrics"][0][
        "completion_rate"
    ] == 0.5
    assert [
        row["stopped_at_subtask"]
        for row in result["subtask_evaluation"]["episode_outcomes"]
    ] == ["move_block_to_center", "press_button"]
    assert result["observable_executor_signal"] is True
    assert result["decision"]["next_action"] == "budget_match_controls_before_comparison"


def test_rejects_unvalidated_run(tmp_path: Path) -> None:
    run = _run(tmp_path, [0.0])
    (run / "emac_manifest.json").unlink()

    with pytest.raises(ValueError, match="E-MAC validation"):
        assess_run_signal(run)


def test_oracle_executor_signal_advances_only_to_memory_training(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path,
        [0.0, 0.1, 0.0],
        manifest={
            "condition": "oracle_subtask_pi05_none",
            "deployable": False,
            "evidence_scope": "executor_skill_diagnostic_only",
        },
    )

    result = assess_run_signal(run)

    assert result["deployable"] is False
    assert result["evidence_scope"] == "executor_skill_diagnostic_only"
    assert result["decision"]["next_action"] == "train_budget_matched_memory_executor"
    assert result["decision"]["controller_training_allowed"] is False


def test_oracle_executor_floor_requests_more_native_training(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        [0.0, 0.0],
        manifest={
            "condition": "oracle_subtask_pi05_none",
            "deployable": False,
            "evidence_scope": "executor_skill_diagnostic_only",
        },
    )

    result = assess_run_signal(run)

    assert result["decision"]["next_action"] == "increase_native_executor_training_budget"


def test_rejects_deployable_oracle_executor_diagnostic(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        [0.1],
        manifest={
            "condition": "oracle_subtask_pi05_none",
            "deployable": True,
            "evidence_scope": "executor_skill_diagnostic_only",
        },
    )

    with pytest.raises(ValueError, match="must be non-deployable"):
        assess_run_signal(run)
