from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any


SCHEMA_VERSION = "memory_harness.put_back_subtask_summary/v1"
SUBTASKS = (
    ("move_block_to_center", "Move the block to the center"),
    ("press_button", "Press the button"),
    ("return_block_to_origin", "Return the block to its original position"),
)


def _progress_score(row: Mapping[str, Any]) -> int:
    progress = row.get("task_progress")
    if not isinstance(progress, Mapping) or progress.get("task") != "put_back_block":
        raise ValueError("episode has no valid Put Back task_progress")
    raw_score = progress.get("max_progress_score")
    if isinstance(raw_score, bool):
        raise ValueError("Put Back max_progress_score must be an integer in [0, 3]")
    try:
        score = int(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Put Back max_progress_score must be an integer in [0, 3]"
        ) from exc
    if score not in range(len(SUBTASKS) + 1) or score != raw_score:
        raise ValueError("Put Back max_progress_score must be an integer in [0, 3]")
    return score


def summarize_put_back_subtasks(
    episodes: Sequence[Mapping[str, Any]],
    *,
    success_key: str = "success",
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("Put Back subtask summary requires at least one episode")

    outcomes: list[dict[str, Any]] = []
    scores: list[int] = []
    for expected_index, row in enumerate(episodes):
        episode_index = int(row.get("episode_index", expected_index))
        if episode_index != expected_index:
            raise ValueError("Put Back episodes must be ordered and contiguous")
        score = _progress_score(row)
        success = bool(row.get(success_key, score == len(SUBTASKS)))
        if success != (score == len(SUBTASKS)):
            raise ValueError(
                "Put Back full-task success disagrees with completed subtask count: "
                f"episode={episode_index}, success={success}, score={score}"
            )
        completed = [name for name, _ in SUBTASKS[:score]]
        stopped_at = None if score == len(SUBTASKS) else SUBTASKS[score][0]
        outcomes.append(
            {
                "episode_index": episode_index,
                "seed": None if row.get("seed") is None else int(row["seed"]),
                "policy_seed": (
                    None if row.get("policy_seed") is None else int(row["policy_seed"])
                ),
                "full_task_success": success,
                "completed_subtask_count": score,
                "completed_subtasks": completed,
                "stopped_at_subtask": stopped_at,
                "status": "task_complete" if stopped_at is None else "stopped",
            }
        )
        scores.append(score)

    stop_counts = Counter(
        outcome["stopped_at_subtask"] or "task_complete" for outcome in outcomes
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "put_back_block",
        "num_episodes": len(outcomes),
        "full_task_success_count": sum(score == len(SUBTASKS) for score in scores),
        "full_task_success_rate": sum(score == len(SUBTASKS) for score in scores)
        / len(scores),
        "mean_completed_subtasks": mean(scores),
        "subtask_metrics": [
            {
                "index": index,
                "name": name,
                "description": description,
                "completed_count": sum(score >= index for score in scores),
                "completion_rate": sum(score >= index for score in scores)
                / len(scores),
            }
            for index, (name, description) in enumerate(SUBTASKS, start=1)
        ],
        "stopped_at_counts": {
            **{name: stop_counts[name] for name, _ in SUBTASKS},
            "task_complete": stop_counts["task_complete"],
        },
        "episode_outcomes": outcomes,
    }


def load_episodes(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty episode JSONL objects: {path}")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Put Back subtask completion and stopping points."
    )
    parser.add_argument("--episodes", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite subtask summary: {args.output}")
    result = summarize_put_back_subtasks(load_episodes(args.episodes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
