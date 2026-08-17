from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from memory_harness.tasks import load_task_spec


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _subtask_segments(
    *,
    dataset_root: pathlib.Path,
    episode: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    tasks = {
        int(row["task_index"]): str(row["task"])
        for row in _read_jsonl(dataset_root / "meta/tasks.jsonl")
    }
    episode_index = int(episode["episode_index"])
    chunks_size = int(info["chunks_size"])
    data_path = dataset_root / str(info["data_path"]).format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    import pyarrow.parquet as pq

    table = pq.read_table(data_path, columns=["task_index"])
    task_indices = [int(value) for value in table["task_index"].to_pylist()]
    expected_length = int(episode["length"])
    if len(task_indices) != expected_length:
        raise ValueError(
            f"episode {episode_index} task-index length mismatch: "
            f"metadata={expected_length}, parquet={len(task_indices)}"
        )
    if not task_indices:
        raise ValueError(f"episode {episode_index} has no frames")

    segments: list[dict[str, Any]] = []
    start = 0
    for end in range(1, len(task_indices) + 1):
        if end < len(task_indices) and task_indices[end] == task_indices[start]:
            continue
        task_index = task_indices[start]
        if task_index not in tasks:
            raise ValueError(
                f"episode {episode_index} references unknown task index {task_index}"
            )
        segments.append(
            {
                "lerobot_episode_index": episode_index,
                "start_frame": start,
                "end_frame": end,
                "phase_label": f"task_{task_index}",
                "executor_prompt": tasks[task_index],
                "task_index": task_index,
                "split": split,
            }
        )
        start = end
    return segments


def build_template(
    task_config: pathlib.Path,
    *,
    hf_lerobot_home: pathlib.Path,
    validation_fraction: float = 0.2,
    split_seed: int = 20260814,
) -> dict[str, Any]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    spec = load_task_spec(task_config)
    episodes_path = hf_lerobot_home / spec.repo_id / "meta/episodes.jsonl"
    rows = _read_jsonl(episodes_path)
    if len(rows) < spec.expected_episodes:
        raise ValueError(
            f"Expected at least {spec.expected_episodes} LeRobot episodes, found {len(rows)}"
        )
    episode_ids = [int(row["episode_index"]) for row in rows]
    if episode_ids != list(range(len(rows))):
        raise ValueError("LeRobot episode indices must be contiguous and zero-indexed")

    rng = np.random.default_rng(split_seed)
    if len(rows) == spec.expected_episodes:
        selected = episode_ids
    else:
        selected = sorted(
            int(value)
            for value in rng.choice(
                episode_ids, size=spec.expected_episodes, replace=False
            )
        )
    shuffled = rng.permutation(selected).tolist()
    validation_count = max(2, int(round(len(selected) * validation_fraction)))
    validation = sorted(shuffled[:validation_count])
    train = sorted(shuffled[validation_count:])
    if len(train) < 2:
        raise ValueError("training split must contain at least two episodes")
    split_by_episode = {episode: "train" for episode in train}
    split_by_episode.update({episode: "validation" for episode in validation})

    selected_rows = [rows[episode] for episode in selected]
    dataset_root = hf_lerobot_home / spec.repo_id
    if spec.tmc == "M(n)":
        segments = [
            segment
            for row in selected_rows
            for segment in _subtask_segments(
                dataset_root=dataset_root,
                episode=row,
                split=split_by_episode[int(row["episode_index"])],
            )
        ]
        prompt_source = "lerobot_frame_task_index"
    else:
        segments = [
            {
                "lerobot_episode_index": int(row["episode_index"]),
                "start_frame": 0,
                "end_frame": int(row["length"]),
                "phase_label": "episode",
                "executor_prompt": spec.prompt,
                "split": split_by_episode[int(row["episode_index"])],
            }
            for row in selected_rows
        ]
        prompt_source = "task_config_global_prompt"

    return {
        "schema_version": "memory_harness.task_template/v1",
        "task_config": str(task_config.resolve()),
        "task_name": spec.task_name,
        "task_memory_complexity": spec.tmc,
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "available_lerobot_episode_count": len(rows),
        "selected_lerobot_episode_count": len(selected),
        "episode_selection": "all" if len(rows) == len(selected) else "seeded_without_replacement",
        "executor_prompt_source": prompt_source,
        "train_lerobot_episode_ids": train,
        "validation_lerobot_episode_ids": validation,
        "segments": segments,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fixed episode split for one RMBench task"
    )
    parser.add_argument("--task-config", type=pathlib.Path, required=True)
    parser.add_argument("--hf-lerobot-home", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260814)
    args = parser.parse_args(argv)
    template = build_template(
        args.task_config,
        hf_lerobot_home=args.hf_lerobot_home,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(template, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
