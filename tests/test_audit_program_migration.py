from __future__ import annotations

import hashlib
import json
from pathlib import Path

from memory_harness.audit_program_migration import audit_program_migration


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(*, explicit: bool) -> dict:
    options = {"embed_dim": 4, "sliding_window_size": 3}
    if explicit:
        options.update(
            {"anchor_path": "anchor", "history_path_quotas": {"history": 3}}
        )
    return {
        "name": "test",
        "deployable": True,
        "paths": [
            {
                "name": "anchor",
                "encoder": {"type": "tokens", "options": {"max_tokens": 1}},
                "writer": {"type": "first"},
                "store": {"type": "anchor"},
                "retriever": {"type": "all"},
                "lifecycle": {"type": "episode"},
            },
            {
                "name": "history",
                "encoder": {"type": "tokens", "options": {"max_tokens": 1}},
                "writer": {"type": "always"},
                "store": {"type": "ring", "options": {"capacity": 3}},
                "retriever": {"type": "all"},
                "lifecycle": {"type": "episode"},
            },
        ],
        "controller": {"type": "all"},
        "utilizer": {"type": "mem0_context", "options": options},
    }


def test_explicit_role_migration_replays_legacy_layout(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    current = tmp_path / "current.json"
    manifest = tmp_path / "manifest.json"
    frozen_hash = _write_json(frozen, _config(explicit=False))
    _write_json(current, _config(explicit=True))
    _write_json(
        manifest,
        {"representation": {"program_config_sha256": frozen_hash}},
    )

    result = audit_program_migration(
        frozen_config=frozen,
        current_config=current,
        context_manifest=manifest,
    )

    assert result["ready_for_context_reuse"] is True
    assert all(result["checks"].values())
    assert all(case["exact"] for case in result["replay_cases"])


def test_migration_rejects_unrelated_config_change(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    current = tmp_path / "current.json"
    manifest = tmp_path / "manifest.json"
    frozen_hash = _write_json(frozen, _config(explicit=False))
    changed = _config(explicit=True)
    changed["paths"][1]["store"]["options"]["capacity"] = 4
    _write_json(current, changed)
    _write_json(
        manifest,
        {"representation": {"program_config_sha256": frozen_hash}},
    )

    result = audit_program_migration(
        frozen_config=frozen,
        current_config=current,
        context_manifest=manifest,
    )

    assert result["ready_for_context_reuse"] is False
    assert result["checks"]["only_explicit_roles_added"] is False
