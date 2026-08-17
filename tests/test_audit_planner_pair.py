from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_harness.audit_planner_pair import audit_pair


def _samples(*, no_key: bool) -> list[dict]:
    rows = []
    for index in range(6):
        user = "<global_task>: task\n"
        user += (
            "<current_observation>: <image>.\n"
            if no_key
            else "<initial_observation>: <image>.\n"
        )
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": f"next_subtask: stage {index}"},
                ],
                "images": [
                    f"{position}.png" for position in range(1 if no_key else index + 1)
                ],
            }
        )
    return rows


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_audit_planner_pair_accepts_only_input_representation_difference(
    tmp_path: Path,
) -> None:
    key = tmp_path / "key.json"
    no_key = tmp_path / "no_key.json"
    _write(key, _samples(no_key=False))
    _write(no_key, _samples(no_key=True))

    result = audit_pair(key, no_key)

    assert result["status"] == "paired"
    assert result["sample_count"] == 6
    assert result["labels_identical"] is True


def test_audit_planner_pair_rejects_label_mismatch(tmp_path: Path) -> None:
    key = tmp_path / "key.json"
    no_key = tmp_path / "no_key.json"
    key_samples = _samples(no_key=False)
    no_key_samples = _samples(no_key=True)
    no_key_samples[3]["messages"][-1]["content"] = "next_subtask: wrong"
    _write(key, key_samples)
    _write(no_key, no_key_samples)

    with pytest.raises(ValueError, match="labels differ"):
        audit_pair(key, no_key)
