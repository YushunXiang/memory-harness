from __future__ import annotations

import json
import pathlib

import pytest

from memory_harness.tasks import load_task_spec


def _payload() -> dict:
    return {
        "schema_version": "memory_harness.task/v1",
        "task_name": "demo",
        "task_config": "demo_clean",
        "tmc": "M(1)",
        "source_dir": "data",
        "repo_id": "local/demo",
        "asset_id": "demo",
        "prompt": "Do the task.",
        "max_steps": 500,
        "paired_layout_protocol": False,
        "fps": 30,
        "expected_episodes": 2,
        "camera_map": {"head_camera": "cam_high"},
    }


def test_load_task_spec_resolves_paths(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "task.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    spec = load_task_spec(path)

    assert spec.task_name == "demo"
    assert spec.tmc == "M(1)"
    assert spec.paired_layout_protocol is False
    assert spec.source_dir == (tmp_path / "data").resolve()
    assert spec.camera_map == {"head_camera": "cam_high"}


def test_load_task_spec_rejects_unknown_tmc(tmp_path: pathlib.Path) -> None:
    payload = _payload()
    payload["tmc"] = "M(2)"
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported task memory complexity"):
        load_task_spec(path)


def test_load_task_spec_rejects_duplicate_camera_targets(
    tmp_path: pathlib.Path,
) -> None:
    payload = _payload()
    payload["camera_map"] = {"head_camera": "cam_high", "front_camera": "cam_high"}
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="targets must be unique"):
        load_task_spec(path)


@pytest.mark.parametrize(
    "field",
    ["baseline_checkpoint", "memory_checkpoint", "unexpected_option"],
)
def test_load_task_spec_rejects_obsolete_and_unknown_fields(
    tmp_path: pathlib.Path, field: str
) -> None:
    payload = _payload()
    payload[field] = "/obsolete/checkpoint"
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown task config fields"):
        load_task_spec(path)
