from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memory_harness.training_provenance import parent_training_evidence


def _write_manifest(path: Path, **updates: object) -> None:
    value = {
        "schema_version": "memory_harness.training/v1",
        "optimizer_updates": 1200,
        "effective_batch": 56,
        **updates,
    }
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_terminal_initial_weights_have_no_parent_training_run(tmp_path: Path) -> None:
    params = tmp_path / "base" / "params"
    params.mkdir(parents=True)

    assert parent_training_evidence(params) == {
        "parent_checkpoint": None,
        "parent_training_manifest_sha256": None,
    }


def test_binds_staged_training_to_parent_manifest(tmp_path: Path) -> None:
    params = tmp_path / "parent" / "params"
    params.mkdir(parents=True)
    manifest = params.parent / "memory_training_manifest.json"
    _write_manifest(manifest)

    assert parent_training_evidence(params) == {
        "parent_checkpoint": str(params.parent.resolve()),
        "parent_training_manifest_sha256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
    }


def test_rejects_invalid_parent_budget(tmp_path: Path) -> None:
    params = tmp_path / "parent" / "params"
    params.mkdir(parents=True)
    _write_manifest(params.parent / "memory_training_manifest.json", optimizer_updates=0)

    with pytest.raises(ValueError, match="invalid parent training budget"):
        parent_training_evidence(params)
