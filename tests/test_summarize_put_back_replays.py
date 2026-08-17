from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_harness.summarize_put_back_replays import summarize_replays


def _replay(path: Path, scores: list[int], *, seed_offset: int = 0) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "memory_harness.put_back_action_replay/v1",
                "episodes": [
                    {
                        "episode_index": index,
                        "seed": 100000 + index + seed_offset,
                        "policy_seed": 120000 + index,
                        "num_actions": 500,
                        "task_progress": {"max_progress_score": score},
                    }
                    for index, score in enumerate(scores)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_summarize_replays_reports_paired_progress_directions(tmp_path: Path) -> None:
    result = summarize_replays(
        {
            "full_memory": _replay(tmp_path / "full.json", [2, 1, 0]),
            "empty_mask": _replay(tmp_path / "empty.json", [1, 1, 0]),
            "native_none": _replay(tmp_path / "native.json", [0, 2, 0]),
        }
    )

    assert result["num_paired_episodes"] == 3
    assert result["conditions"]["full_memory"]["mean_progress_score"] == 1.0
    assert result["conditions"]["full_memory"]["score_counts"] == {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 0,
    }
    assert result["paired_directions"]["empty_mask_to_full_memory"] == {
        "candidate_higher": 1,
        "equal": 2,
        "candidate_lower": 0,
    }
    assert result["paired_directions"]["native_none_to_full_memory"] == {
        "candidate_higher": 1,
        "equal": 1,
        "candidate_lower": 1,
    }
    assert result["formal_utility_claim_allowed"] is False


def test_summarize_replays_rejects_unpaired_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not paired"):
        summarize_replays(
            {
                "full_memory": _replay(tmp_path / "full.json", [1]),
                "empty_mask": _replay(tmp_path / "empty.json", [1], seed_offset=1),
                "native_none": _replay(tmp_path / "native.json", [1]),
            }
        )


def test_summarize_replays_requires_all_three_conditions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="labels must be exactly"):
        summarize_replays(
            {
                "full_memory": _replay(tmp_path / "full.json", [1]),
                "empty_mask": _replay(tmp_path / "empty.json", [1]),
            }
        )
