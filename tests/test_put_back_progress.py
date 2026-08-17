from __future__ import annotations

import pytest

from memory_harness.put_back_progress import summarize_put_back_subtasks


def _episode(index: int, score: int) -> dict[str, object]:
    return {
        "episode_index": index,
        "seed": 100000 + index,
        "policy_seed": 120000 + index,
        "success": score == 3,
        "task_progress": {
            "task": "put_back_block",
            "max_progress_score": score,
        },
    }


def test_reports_subtask_rates_and_episode_stopping_points() -> None:
    result = summarize_put_back_subtasks(
        [_episode(0, 0), _episode(1, 1), _episode(2, 2), _episode(3, 3)]
    )

    assert result["full_task_success_rate"] == 0.25
    assert result["mean_completed_subtasks"] == 1.5
    assert [row["completion_rate"] for row in result["subtask_metrics"]] == [
        0.75,
        0.5,
        0.25,
    ]
    assert result["stopped_at_counts"] == {
        "move_block_to_center": 1,
        "press_button": 1,
        "return_block_to_origin": 1,
        "task_complete": 1,
    }
    assert result["episode_outcomes"][1] == {
        "episode_index": 1,
        "seed": 100001,
        "policy_seed": 120001,
        "full_task_success": False,
        "completed_subtask_count": 1,
        "completed_subtasks": ["move_block_to_center"],
        "stopped_at_subtask": "press_button",
        "status": "stopped",
    }


def test_rejects_success_that_disagrees_with_subtask_progress() -> None:
    row = _episode(0, 2)
    row["success"] = True

    with pytest.raises(ValueError, match="success disagrees"):
        summarize_put_back_subtasks([row])


def test_rejects_missing_progress() -> None:
    with pytest.raises(ValueError, match="task_progress"):
        summarize_put_back_subtasks([{"episode_index": 0, "success": False}])


def test_rejects_non_contiguous_episodes() -> None:
    with pytest.raises(ValueError, match="ordered and contiguous"):
        summarize_put_back_subtasks([_episode(1, 0)])
