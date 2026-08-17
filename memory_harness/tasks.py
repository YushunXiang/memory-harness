from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any


TASK_SCHEMA_VERSION = "memory_harness.task/v1"


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    task_name: str
    task_config: str
    tmc: str
    source_dir: pathlib.Path
    repo_id: str
    asset_id: str
    prompt: str
    max_steps: int
    paired_layout_protocol: bool
    fps: int
    expected_episodes: int
    camera_map: dict[str, str]


TASK_FIELDS = {
    "schema_version",
    "task_name",
    "task_config",
    "tmc",
    "source_dir",
    "repo_id",
    "asset_id",
    "prompt",
    "max_steps",
    "paired_layout_protocol",
    "fps",
    "expected_episodes",
    "camera_map",
}


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Task config field {key!r} must be a non-empty string")
    return value


def _require_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Task config field {key!r} must be a positive integer")
    return value


def _require_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Task config field {key!r} must be a boolean")
    return value


def load_task_spec(path: pathlib.Path) -> TaskSpec:
    config_path = path.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Task config must be a JSON object")
    if payload.get("schema_version") != TASK_SCHEMA_VERSION:
        raise ValueError(f"Task config must use schema_version {TASK_SCHEMA_VERSION!r}")
    unknown_fields = set(payload) - TASK_FIELDS
    if unknown_fields:
        raise ValueError(f"Unknown task config fields: {sorted(unknown_fields)}")

    raw_camera_map = payload.get("camera_map")
    if not isinstance(raw_camera_map, dict) or not raw_camera_map:
        raise ValueError("Task config field 'camera_map' must be a non-empty object")
    camera_map: dict[str, str] = {}
    for source, target in raw_camera_map.items():
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(target, str)
            or not target
        ):
            raise ValueError(
                "Task camera_map keys and values must be non-empty strings"
            )
        camera_map[source] = target
    if len(set(camera_map.values())) != len(camera_map):
        raise ValueError("Task camera_map targets must be unique")

    source_dir = (config_path.parent / _require_str(payload, "source_dir")).resolve()
    tmc = _require_str(payload, "tmc")
    if tmc not in {"M(0)-control", "M(1)", "M(n)"}:
        raise ValueError(f"Unsupported task memory complexity: {tmc!r}")

    return TaskSpec(
        task_name=_require_str(payload, "task_name"),
        task_config=_require_str(payload, "task_config"),
        tmc=tmc,
        source_dir=source_dir,
        repo_id=_require_str(payload, "repo_id"),
        asset_id=_require_str(payload, "asset_id"),
        prompt=_require_str(payload, "prompt"),
        max_steps=_require_positive_int(payload, "max_steps"),
        paired_layout_protocol=_require_bool(payload, "paired_layout_protocol"),
        fps=_require_positive_int(payload, "fps"),
        expected_episodes=_require_positive_int(payload, "expected_episodes"),
        camera_map=camera_map,
    )
