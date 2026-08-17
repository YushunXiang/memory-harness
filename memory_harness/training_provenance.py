from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parent_training_evidence(
    initial_weight_params: pathlib.Path,
) -> dict[str, Any]:
    """Bind a staged run to the exact parent training manifest, when present."""

    initial_weight_params = initial_weight_params.resolve()
    parent_checkpoint = (
        initial_weight_params.parent
        if initial_weight_params.name == "params"
        else initial_weight_params
    )
    manifest_path = parent_checkpoint / "memory_training_manifest.json"
    if not manifest_path.is_file():
        return {
            "parent_checkpoint": None,
            "parent_training_manifest_sha256": None,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "memory_harness.training/v1":
        raise ValueError(f"unsupported parent training manifest: {manifest_path}")
    try:
        optimizer_updates = int(manifest["optimizer_updates"])
        effective_batch = int(manifest["effective_batch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid parent training budget: {manifest_path}") from exc
    if optimizer_updates <= 0 or effective_batch <= 0:
        raise ValueError(f"invalid parent training budget: {manifest_path}")
    return {
        "parent_checkpoint": str(parent_checkpoint),
        "parent_training_manifest_sha256": _sha256(manifest_path),
    }
