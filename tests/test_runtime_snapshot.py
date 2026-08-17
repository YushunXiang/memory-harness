from __future__ import annotations

import json

import pytest

from memory_harness.runtime_snapshot import create_runtime_snapshot
from memory_harness.runtime_snapshot import copy_runtime_snapshot
from memory_harness.runtime_snapshot import validate_runtime_snapshot


def test_runtime_snapshot_copies_and_verifies_python_sources(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    (source / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (source / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    manifest = create_runtime_snapshot(source, tmp_path / "runtime")

    assert manifest["package"] == "memory_harness"
    assert [row["path"] for row in manifest["files"]] == [
        "memory_harness/__init__.py",
        "memory_harness/module.py",
    ]
    assert validate_runtime_snapshot(tmp_path / "runtime") == manifest


def test_runtime_snapshot_rejects_source_mutation(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    output = tmp_path / "runtime"
    create_runtime_snapshot(source, output)
    (output / "memory_harness" / "__init__.py").write_text(
        "VERSION = 2\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="hashes changed"):
        validate_runtime_snapshot(output)


def test_runtime_snapshot_rejects_manifest_aggregate_mutation(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    output = tmp_path / "runtime"
    create_runtime_snapshot(source, output)
    manifest_path = output / "runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate hash changed"):
        validate_runtime_snapshot(output)


def test_copy_runtime_snapshot_preserves_verified_identity(tmp_path) -> None:
    source_package = tmp_path / "source_package"
    source_package.mkdir()
    (source_package / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    source_snapshot = tmp_path / "source_snapshot"
    original = create_runtime_snapshot(source_package, source_snapshot)

    copied = copy_runtime_snapshot(source_snapshot, tmp_path / "copied_snapshot")

    assert copied == original
