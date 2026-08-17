from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from typing import Any


SCHEMA_VERSION = "memory_harness.runtime_snapshot/v1"


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def create_runtime_snapshot(
    source_package: pathlib.Path, output_dir: pathlib.Path
) -> dict[str, Any]:
    source_package = source_package.resolve()
    if not source_package.is_dir():
        raise ValueError(f"runtime source package does not exist: {source_package}")
    if output_dir.exists():
        raise FileExistsError(f"runtime snapshot already exists: {output_dir}")
    package_output = output_dir / "memory_harness"
    package_output.mkdir(parents=True)
    files: list[dict[str, str]] = []
    for source in sorted(source_package.rglob("*.py")):
        relative = source.relative_to(source_package)
        destination = package_output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "path": str(pathlib.Path("memory_harness") / relative),
                "sha256": _sha256(destination),
            }
        )
    if not files:
        raise ValueError(f"runtime source contains no Python files: {source_package}")
    source_sha256 = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package": "memory_harness",
        "source_sha256": source_sha256,
        "files": files,
    }
    (output_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_runtime_snapshot(output_dir: pathlib.Path) -> dict[str, Any]:
    manifest_path = output_dir / "runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("runtime snapshot has an unsupported schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("runtime snapshot manifest has no files")
    declared_paths = {str(row["path"]) for row in files}
    actual_paths = {
        str(path.relative_to(output_dir))
        for path in (output_dir / "memory_harness").rglob("*.py")
    }
    if actual_paths != declared_paths:
        raise ValueError(
            "runtime snapshot file set changed: "
            f"missing={sorted(declared_paths - actual_paths)}, "
            f"unknown={sorted(actual_paths - declared_paths)}"
        )
    failed_hashes = [
        str(row["path"])
        for row in files
        if _sha256(output_dir / str(row["path"])) != row.get("sha256")
    ]
    if failed_hashes:
        raise ValueError(f"runtime snapshot hashes changed: {failed_hashes}")
    expected_source_hash = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("source_sha256") != expected_source_hash:
        raise ValueError("runtime snapshot aggregate hash changed")
    return manifest


def copy_runtime_snapshot(
    source_dir: pathlib.Path, output_dir: pathlib.Path
) -> dict[str, Any]:
    source_manifest = validate_runtime_snapshot(source_dir)
    if output_dir.exists():
        raise FileExistsError(f"runtime snapshot already exists: {output_dir}")
    shutil.copytree(source_dir, output_dir, copy_function=shutil.copy2)
    copied_manifest = validate_runtime_snapshot(output_dir)
    if copied_manifest["source_sha256"] != source_manifest["source_sha256"]:
        raise RuntimeError("copied runtime snapshot identity changed")
    return copied_manifest
