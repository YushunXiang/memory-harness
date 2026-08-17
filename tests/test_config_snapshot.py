from __future__ import annotations

import json

import pytest

from memory_harness.config_snapshot import create_config_snapshot
from memory_harness.config_snapshot import copy_config_snapshot
from memory_harness.config_snapshot import validate_config_snapshot


def test_config_snapshot_copies_and_verifies_json_tree(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "tasks").mkdir(parents=True)
    (source / "tasks" / "task.json").write_text('{"task": 1}\n', encoding="utf-8")
    (source / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    manifest = create_config_snapshot(source, tmp_path / "snapshot")

    assert [row["path"] for row in manifest["files"]] == ["tasks/task.json"]
    assert validate_config_snapshot(tmp_path / "snapshot") == manifest


def test_config_snapshot_rejects_changed_config(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.json").write_text('{"value": 1}\n', encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    create_config_snapshot(source, snapshot)
    (snapshot / "program.json").write_text('{"value": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="hashes changed"):
        validate_config_snapshot(snapshot)


def test_config_snapshot_rejects_unknown_json(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.json").write_text('{}\n', encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    create_config_snapshot(source, snapshot)
    (snapshot / "unknown.json").write_text(json.dumps({}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file set changed"):
        validate_config_snapshot(snapshot)


def test_config_snapshot_copy_preserves_identity(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.json").write_text('{}\n', encoding="utf-8")
    first = tmp_path / "first"
    original = create_config_snapshot(source, first)

    copied = copy_config_snapshot(first, tmp_path / "copied")

    assert copied["source_sha256"] == original["source_sha256"]
