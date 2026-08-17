from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import tempfile
from collections.abc import Mapping
from typing import Any

from memory_harness.architecture import ArchitectureSpec
from memory_harness.config import load_program_spec
from memory_harness.registry import build_program
from memory_harness.runtime_snapshot import create_runtime_snapshot
from memory_harness.runtime_snapshot import validate_runtime_snapshot
from memory_harness.smoke import run_smoke


SCHEMA_VERSION = "memory_harness.search_candidate/v1"
SUBMISSION_SCHEMA_VERSION = "memory_harness.search_candidate_submission/v1"
SEARCH_AXES = frozenset(
    {
        "encoder",
        "writer",
        "store",
        "retriever",
        "utilizer",
        "lifecycle",
        "controller",
        "planner_memory",
    }
)
_SHA256_LENGTH = 64
_SUBMISSION_FILES = frozenset(
    {"submission.json", "architecture.json", "executor_program.json"}
)


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_submission(path: pathlib.Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("candidate submission must be an object")
    expected = {
        "schema_version",
        "hypothesis",
        "parent_content_sha256",
        "search_axes",
    }
    if set(raw) != expected:
        raise ValueError(
            "candidate submission keys mismatch: "
            f"missing={sorted(expected - set(raw))}, "
            f"unknown={sorted(set(raw) - expected)}"
        )
    if raw["schema_version"] != SUBMISSION_SCHEMA_VERSION:
        raise ValueError("candidate submission has an unsupported schema")
    hypothesis = raw["hypothesis"]
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("candidate hypothesis must be non-empty")
    parent = raw["parent_content_sha256"]
    if parent is not None and not _is_sha256(parent):
        raise ValueError("parent_content_sha256 must be null or a lowercase SHA-256")
    axes = raw["search_axes"]
    if (
        not isinstance(axes, list)
        or not axes
        or any(not isinstance(axis, str) for axis in axes)
    ):
        raise ValueError("search_axes must be a non-empty string list")
    if len(axes) != len(set(axes)):
        raise ValueError("search_axes must not contain duplicates")
    unknown_axes = set(axes) - SEARCH_AXES
    if unknown_axes:
        raise ValueError(f"unknown search axes: {sorted(unknown_axes)}")
    return {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "hypothesis": hypothesis.strip(),
        "parent_content_sha256": parent,
        "search_axes": axes,
    }


def _validate_submission_tree(submission_dir: pathlib.Path) -> None:
    if not submission_dir.is_dir():
        raise ValueError(
            f"candidate submission directory does not exist: {submission_dir}"
        )
    entries = {entry.name for entry in submission_dir.iterdir()}
    if entries != _SUBMISSION_FILES:
        raise ValueError(
            "configuration-only candidate file set mismatch: "
            f"missing={sorted(_SUBMISSION_FILES - entries)}, "
            f"unknown={sorted(entries - _SUBMISSION_FILES)}"
        )
    symbolic_links = sorted(
        entry.name for entry in submission_dir.iterdir() if entry.is_symlink()
    )
    if symbolic_links:
        raise ValueError(
            f"candidate submission contains symbolic links: {symbolic_links}"
        )
    non_files = sorted(
        entry.name for entry in submission_dir.iterdir() if not entry.is_file()
    )
    if non_files:
        raise ValueError(f"candidate submission contains non-files: {non_files}")


def _behavior_payload(
    *,
    architecture_raw: Mapping[str, Any],
    program_raw: Mapping[str, Any],
    runtime_source_sha256: str,
) -> dict[str, Any]:
    architecture_behavior = dict(architecture_raw)
    architecture_behavior.pop("name", None)
    architecture_behavior.pop("executor_program", None)
    program_behavior = dict(program_raw)
    program_behavior.pop("name", None)
    return {
        "runtime_source_sha256": runtime_source_sha256,
        "architecture": architecture_behavior,
        "executor_program": program_behavior,
    }


def create_search_candidate(
    *,
    submission_dir: pathlib.Path,
    runtime_source: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    """Freeze and smoke one configuration-only Agent candidate atomically."""

    submission_dir = submission_dir.resolve()
    runtime_source = runtime_source.resolve()
    output_dir = output_dir.resolve()
    _validate_submission_tree(submission_dir)
    submission = _load_submission(submission_dir / "submission.json")
    source_architecture = ArchitectureSpec.load(submission_dir / "architecture.json")
    expected_executor = (submission_dir / "executor_program.json").resolve()
    if source_architecture.executor_program != expected_executor:
        raise ValueError(
            "candidate architecture must reference local executor_program.json"
        )
    source_program = load_program_spec(expected_executor)
    if not source_program.deployable:
        raise ValueError("architecture-search candidates must be deployable")
    build_program(source_program)

    if output_dir.exists():
        raise FileExistsError(f"search candidate artifact already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix=".search-candidate-", dir=output_dir.parent)
    )
    try:
        for name in sorted(_SUBMISSION_FILES):
            shutil.copy2(submission_dir / name, temporary / name)
        architecture = ArchitectureSpec.load(temporary / "architecture.json")
        program_path = temporary / "executor_program.json"
        program = load_program_spec(program_path)
        build_program(program)
        runtime = create_runtime_snapshot(runtime_source, temporary / "runtime")
        smoke = run_smoke(config_path=program_path, output_dir=temporary / "smoke")

        architecture_raw = json.loads(
            (temporary / "architecture.json").read_text(encoding="utf-8")
        )
        program_raw = json.loads(program_path.read_text(encoding="utf-8"))
        content_sha256 = _canonical_sha256(
            _behavior_payload(
                architecture_raw=architecture_raw,
                program_raw=program_raw,
                runtime_source_sha256=runtime["source_sha256"],
            )
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "preflight_passed",
            "activation": "inactive_until_fixed_memory_gate_1",
            "candidate_content_sha256": content_sha256,
            "parent_content_sha256": submission["parent_content_sha256"],
            "hypothesis": submission["hypothesis"],
            "search_axes": submission["search_axes"],
            "candidate_name": architecture.name,
            "deployable": program.deployable,
            "architecture_config": "architecture.json",
            "architecture_config_sha256": _sha256(temporary / "architecture.json"),
            "executor_program": "executor_program.json",
            "executor_program_sha256": _sha256(program_path),
            "runtime": "runtime",
            "runtime_source_sha256": runtime["source_sha256"],
            "runtime_manifest_sha256": _sha256(
                temporary / "runtime" / "runtime_manifest.json"
            ),
            "smoke_summary": "smoke/summary.json",
            "smoke_summary_sha256": _sha256(temporary / "smoke" / "summary.json"),
            "smoke_status": smoke["status"],
            "candidate_kind": "configuration_only_typed_operator_composition",
        }
        (temporary / "candidate_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_search_candidate(temporary)
        temporary.rename(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_search_candidate(candidate_dir: pathlib.Path) -> dict[str, Any]:
    """Validate identity, runtime, typed config and smoke evidence of one candidate."""

    candidate_dir = candidate_dir.resolve()
    expected_top_level = _SUBMISSION_FILES | {
        "candidate_manifest.json",
        "runtime",
        "smoke",
    }
    top_level = {entry.name for entry in candidate_dir.iterdir()}
    if top_level != expected_top_level:
        raise ValueError(
            "search candidate artifact file set mismatch: "
            f"missing={sorted(expected_top_level - top_level)}, "
            f"unknown={sorted(top_level - expected_top_level)}"
        )
    symbolic_links = sorted(
        str(entry.relative_to(candidate_dir))
        for entry in candidate_dir.rglob("*")
        if entry.is_symlink()
    )
    if symbolic_links:
        raise ValueError(f"search candidate artifact contains links: {symbolic_links}")
    manifest_path = candidate_dir / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("search candidate artifact has an unsupported schema")
    if manifest.get("status") != "preflight_passed":
        raise ValueError("search candidate preflight did not pass")
    if manifest.get("activation") != "inactive_until_fixed_memory_gate_1":
        raise ValueError("search candidate activation gate changed")
    submission = _load_submission(candidate_dir / "submission.json")
    if manifest.get("hypothesis") != submission["hypothesis"]:
        raise ValueError("search candidate hypothesis changed")
    if manifest.get("parent_content_sha256") != submission["parent_content_sha256"]:
        raise ValueError("search candidate parent changed")
    if manifest.get("search_axes") != submission["search_axes"]:
        raise ValueError("search candidate axes changed")

    architecture_path = candidate_dir / str(manifest.get("architecture_config", ""))
    program_path = candidate_dir / str(manifest.get("executor_program", ""))
    if architecture_path.resolve() != (candidate_dir / "architecture.json").resolve():
        raise ValueError("search candidate architecture path changed")
    if program_path.resolve() != (candidate_dir / "executor_program.json").resolve():
        raise ValueError("search candidate executor path changed")
    if _sha256(architecture_path) != manifest.get("architecture_config_sha256"):
        raise ValueError("search candidate architecture changed")
    if _sha256(program_path) != manifest.get("executor_program_sha256"):
        raise ValueError("search candidate executor changed")
    architecture = ArchitectureSpec.load(architecture_path)
    if architecture.executor_program != program_path.resolve():
        raise ValueError("search candidate architecture/executor link changed")
    program = load_program_spec(program_path)
    if not program.deployable or manifest.get("deployable") is not True:
        raise ValueError("search candidate must remain deployable")
    if architecture.name != manifest.get("candidate_name"):
        raise ValueError("search candidate name changed")
    build_program(program)

    runtime_path = candidate_dir / str(manifest.get("runtime", ""))
    if runtime_path.resolve() != (candidate_dir / "runtime").resolve():
        raise ValueError("search candidate runtime path changed")
    runtime = validate_runtime_snapshot(runtime_path)
    runtime_manifest = candidate_dir / "runtime" / "runtime_manifest.json"
    if _sha256(runtime_manifest) != manifest.get("runtime_manifest_sha256"):
        raise ValueError("search candidate runtime manifest changed")
    if runtime["source_sha256"] != manifest.get("runtime_source_sha256"):
        raise ValueError("search candidate runtime identity changed")
    declared_runtime_files = {
        "runtime_manifest.json",
        *(str(row["path"]) for row in runtime["files"]),
    }
    actual_runtime_files = {
        str(path.relative_to(runtime_path))
        for path in runtime_path.rglob("*")
        if path.is_file()
    }
    if actual_runtime_files != declared_runtime_files:
        raise ValueError("search candidate runtime contains undeclared files")
    architecture_raw = json.loads(architecture_path.read_text(encoding="utf-8"))
    program_raw = json.loads(program_path.read_text(encoding="utf-8"))
    expected_content = _canonical_sha256(
        _behavior_payload(
            architecture_raw=architecture_raw,
            program_raw=program_raw,
            runtime_source_sha256=runtime["source_sha256"],
        )
    )
    if expected_content != manifest.get("candidate_content_sha256"):
        raise ValueError("search candidate behavior identity changed")
    smoke_path = candidate_dir / str(manifest.get("smoke_summary", ""))
    if smoke_path.resolve() != (candidate_dir / "smoke" / "summary.json").resolve():
        raise ValueError("search candidate smoke path changed")
    smoke_files = {
        str(path.relative_to(candidate_dir / "smoke"))
        for path in (candidate_dir / "smoke").rglob("*")
        if path.is_file()
    }
    if smoke_files != {"memory_audit.jsonl", "program.json", "summary.json"}:
        raise ValueError("search candidate smoke contains undeclared files")
    if _sha256(smoke_path) != manifest.get("smoke_summary_sha256"):
        raise ValueError("search candidate smoke evidence changed")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (
        smoke.get("status") != "completed"
        or manifest.get("smoke_status") != "completed"
    ):
        raise ValueError("search candidate smoke did not complete")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    package_dir = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Freeze or validate one typed architecture-search candidate."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--submission", type=pathlib.Path, required=True)
    create.add_argument("--runtime-source", type=pathlib.Path, default=package_dir)
    create.add_argument("--output-dir", type=pathlib.Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--candidate", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create":
        manifest = create_search_candidate(
            submission_dir=args.submission,
            runtime_source=args.runtime_source,
            output_dir=args.output_dir,
        )
    else:
        manifest = validate_search_candidate(args.candidate)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
