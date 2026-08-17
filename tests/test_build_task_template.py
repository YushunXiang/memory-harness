from __future__ import annotations

import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

from memory_harness.build_task_template import build_template


def test_build_template_creates_disjoint_deterministic_split(
    tmp_path: pathlib.Path,
) -> None:
    dataset = tmp_path / "lerobot/local/demo/meta"
    dataset.mkdir(parents=True)
    (dataset / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "length": 100 + index}) + "\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "schema_version": "memory_harness.task/v1",
                "task_name": "demo",
                "task_config": "demo_clean",
                "tmc": "M(1)",
                "source_dir": "source",
                "repo_id": "local/demo",
                "asset_id": "demo",
                "prompt": "Remember the initial location.",
                "max_steps": 500,
                "paired_layout_protocol": False,
                "fps": 30,
                "expected_episodes": 10,
                "camera_map": {"head_camera": "cam_high"},
            }
        ),
        encoding="utf-8",
    )

    first = build_template(task, hf_lerobot_home=tmp_path / "lerobot", split_seed=7)
    second = build_template(task, hf_lerobot_home=tmp_path / "lerobot", split_seed=7)

    assert first == second
    assert first["executor_prompt_source"] == "task_config_global_prompt"
    assert first["available_lerobot_episode_count"] == 10
    assert first["selected_lerobot_episode_count"] == 10
    assert len(first["train_lerobot_episode_ids"]) == 8
    assert len(first["validation_lerobot_episode_ids"]) == 2
    assert not (
        set(first["train_lerobot_episode_ids"])
        & set(first["validation_lerobot_episode_ids"])
    )
    assert first["segments"][3] == {
        "lerobot_episode_index": 3,
        "start_frame": 0,
        "end_frame": 103,
        "phase_label": "episode",
        "executor_prompt": "Remember the initial location.",
        "split": "train" if 3 in first["train_lerobot_episode_ids"] else "validation",
    }


def test_mn_template_uses_frame_task_index_and_selects_requested_episode_count(
    tmp_path: pathlib.Path,
) -> None:
    dataset = tmp_path / "lerobot/local/demo"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
        ),
        encoding="utf-8",
    )
    (dataset / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Cover left."})
        + "\n"
        + json.dumps({"task_index": 1, "task": "Open red."})
        + "\n",
        encoding="utf-8",
    )
    (dataset / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "length": 5}) + "\n"
            for index in range(5)
        ),
        encoding="utf-8",
    )
    for index in range(5):
        pq.write_table(
            pa.table({"task_index": [0, 0, 0, 1, 1]}),
            dataset / f"data/chunk-000/episode_{index:06d}.parquet",
        )
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "schema_version": "memory_harness.task/v1",
                "task_name": "demo",
                "task_config": "demo_clean",
                "tmc": "M(n)",
                "source_dir": "source",
                "repo_id": "local/demo",
                "asset_id": "demo",
                "prompt": "Do both stages.",
                "max_steps": 500,
                "paired_layout_protocol": False,
                "fps": 30,
                "expected_episodes": 4,
                "camera_map": {"head_camera": "cam_high"},
            }
        ),
        encoding="utf-8",
    )

    result = build_template(
        task, hf_lerobot_home=tmp_path / "lerobot", split_seed=7
    )

    assert result["available_lerobot_episode_count"] == 5
    assert result["selected_lerobot_episode_count"] == 4
    assert result["episode_selection"] == "seeded_without_replacement"
    assert result["executor_prompt_source"] == "lerobot_frame_task_index"
    assert len(result["segments"]) == 8
    assert result["segments"][0]["start_frame"] == 0
    assert result["segments"][0]["end_frame"] == 3
    assert result["segments"][0]["executor_prompt"] == "Cover left."
    assert result["segments"][1]["start_frame"] == 3
    assert result["segments"][1]["end_frame"] == 5
    assert result["segments"][1]["executor_prompt"] == "Open red."
