from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import tempfile
from typing import Any

from memory_harness.architecture import ArchitectureSpec
from memory_harness.catalog import component_catalog
from memory_harness.config import load_program_spec
from memory_harness.config_snapshot import create_config_snapshot
from memory_harness.config_snapshot import validate_config_snapshot
from memory_harness.registry import build_program
from memory_harness.runtime_snapshot import create_runtime_snapshot
from memory_harness.runtime_snapshot import validate_runtime_snapshot
from memory_harness.smoke import run_smoke


SCHEMA_VERSION = "memory_harness.candidate_suite/v1"


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _discover_architectures(
    config_source: pathlib.Path,
) -> list[tuple[str, pathlib.Path, ArchitectureSpec]]:
    architecture_dir = config_source / "architectures"
    paths = sorted(architecture_dir.glob("fixed_*.json"))
    if not paths:
        raise ValueError(
            f"candidate config source has no fixed architectures: {architecture_dir}"
        )
    discovered: list[tuple[str, pathlib.Path, ArchitectureSpec]] = []
    declared_names: set[str] = set()
    for path in paths:
        alias = path.stem.removeprefix("fixed_")
        if not alias:
            raise ValueError(f"invalid fixed architecture filename: {path.name}")
        spec = ArchitectureSpec.load(path)
        if spec.name in declared_names:
            raise ValueError(f"duplicate architecture name: {spec.name!r}")
        declared_names.add(spec.name)
        if not spec.executor_program.is_relative_to(config_source):
            raise ValueError(
                f"architecture {alias!r} executor escapes config source: "
                f"{spec.executor_program}"
            )
        build_program(load_program_spec(spec.executor_program))
        discovered.append((alias, path, spec))
    return discovered


def create_candidate_suite(
    *,
    runtime_source: pathlib.Path,
    config_source: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    runtime_source = runtime_source.resolve()
    config_source = config_source.resolve()
    output_dir = output_dir.resolve()
    architectures = _discover_architectures(config_source)
    if output_dir.exists():
        raise FileExistsError(f"candidate suite already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix=".candidate-suite-", dir=output_dir.parent)
    )
    try:
        runtime_manifest = create_runtime_snapshot(
            runtime_source, temporary / "runtime"
        )
        config_manifest = create_config_snapshot(
            config_source, temporary / "experiment_configs"
        )
        frozen_configs = temporary / "experiment_configs"
        rows: list[dict[str, Any]] = []
        for alias, _, _ in architectures:
            architecture_path = (
                frozen_configs / "architectures" / f"fixed_{alias}.json"
            )
            architecture = ArchitectureSpec.load(architecture_path)
            summary = run_smoke(
                config_path=architecture.executor_program,
                output_dir=temporary / "smoke" / alias,
            )
            rows.append(
                {
                    "alias": alias,
                    "name": architecture.name,
                    "architecture_config": str(
                        pathlib.Path("architectures") / f"fixed_{alias}.json"
                    ),
                    "architecture_config_sha256": _sha256(architecture_path),
                    "executor_program": str(
                        architecture.executor_program.relative_to(frozen_configs)
                    ),
                    "executor_program_sha256": _sha256(
                        architecture.executor_program
                    ),
                    "planner": architecture.planner,
                    "planner_memory": architecture.planner_memory,
                    "smoke_status": summary["status"],
                    "smoke_summary": str(
                        pathlib.Path("smoke") / alias / "summary.json"
                    ),
                    "smoke_summary_sha256": _sha256(
                        temporary / "smoke" / alias / "summary.json"
                    ),
                }
            )
        catalog_path = temporary / "component_catalog.json"
        catalog_path.write_text(
            json.dumps(component_catalog(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "runtime_source": str(runtime_source),
            "runtime_source_sha256": runtime_manifest["source_sha256"],
            "config_source": str(config_source),
            "config_source_sha256": config_manifest["source_sha256"],
            "component_catalog": "component_catalog.json",
            "component_catalog_sha256": _sha256(catalog_path),
            "architecture_count": len(rows),
            "architectures": rows,
            "rollout_environment": {
                "MEMORY_RUNTIME_SNAPSHOT": "runtime",
                "MEMORY_CONFIG_SNAPSHOT": "experiment_configs",
            },
            "checkpoint_weights_included": False,
        }
        (temporary / "candidate_suite_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_candidate_suite(temporary)
        temporary.rename(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_candidate_suite(suite_dir: pathlib.Path) -> dict[str, Any]:
    suite_dir = suite_dir.resolve()
    manifest_path = suite_dir / "candidate_suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate suite has an unsupported schema")
    runtime = validate_runtime_snapshot(suite_dir / "runtime")
    configs = validate_config_snapshot(suite_dir / "experiment_configs")
    if manifest.get("runtime_source_sha256") != runtime["source_sha256"]:
        raise ValueError("candidate suite runtime identity changed")
    if manifest.get("config_source_sha256") != configs["source_sha256"]:
        raise ValueError("candidate suite config identity changed")
    catalog_path = suite_dir / str(manifest.get("component_catalog", ""))
    if _sha256(catalog_path) != manifest.get("component_catalog_sha256"):
        raise ValueError("candidate suite component catalog changed")

    rows = manifest.get("architectures")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate suite has no architectures")
    aliases = [row.get("alias") for row in rows if isinstance(row, dict)]
    if len(aliases) != len(rows) or any(
        not isinstance(alias, str) or not alias for alias in aliases
    ):
        raise ValueError("candidate suite architecture aliases are invalid")
    if len(aliases) != len(set(aliases)):
        raise ValueError("candidate suite architecture aliases are not unique")
    if manifest.get("architecture_count") != len(rows):
        raise ValueError("candidate suite architecture count changed")
    frozen_configs = suite_dir / "experiment_configs"
    expected_aliases = {
        path.stem.removeprefix("fixed_")
        for path in (frozen_configs / "architectures").glob("fixed_*.json")
    }
    if set(aliases) != expected_aliases:
        raise ValueError(
            "candidate suite architecture set changed: "
            f"missing={sorted(expected_aliases - set(aliases))}, "
            f"unknown={sorted(set(aliases) - expected_aliases)}"
        )
    for row in rows:
        alias = row["alias"]
        architecture_path = frozen_configs / str(row["architecture_config"])
        if _sha256(architecture_path) != row.get("architecture_config_sha256"):
            raise ValueError(f"candidate architecture changed: {alias}")
        architecture = ArchitectureSpec.load(architecture_path)
        if architecture.name != row.get("name"):
            raise ValueError(f"candidate architecture name changed: {alias}")
        if architecture.planner != row.get("planner"):
            raise ValueError(f"candidate planner changed: {alias}")
        if architecture.planner_memory != row.get("planner_memory"):
            raise ValueError(f"candidate planner memory changed: {alias}")
        executor_relative = architecture.executor_program.relative_to(frozen_configs)
        if str(executor_relative) != row.get("executor_program"):
            raise ValueError(f"candidate executor path changed: {alias}")
        if _sha256(architecture.executor_program) != row.get(
            "executor_program_sha256"
        ):
            raise ValueError(f"candidate executor changed: {alias}")
        build_program(load_program_spec(architecture.executor_program))
        smoke_path = suite_dir / str(row.get("smoke_summary", ""))
        if _sha256(smoke_path) != row.get("smoke_summary_sha256"):
            raise ValueError(f"candidate smoke summary changed: {alias}")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "completed" or row.get("smoke_status") != "completed":
            raise ValueError(f"candidate smoke did not complete: {alias}")
    return manifest


def _mem0_shape(program_config: pathlib.Path) -> tuple[int, int] | None:
    raw = json.loads(program_config.read_text(encoding="utf-8"))
    utilizer = raw.get("utilizer", {})
    if utilizer.get("type") == "none":
        return None
    if utilizer.get("type") != "mem0_context":
        raise ValueError(
            f"candidate uses a non-Mem-0 executor utilizer: {program_config}"
        )
    options = utilizer.get("options", {})
    try:
        return 1 + int(options["sliding_window_size"]), int(options["embed_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Mem-0 context shape: {program_config}") from exc


def validate_candidate_checkpoint(
    suite_dir: pathlib.Path, checkpoint_dir: pathlib.Path
) -> dict[str, Any]:
    suite_dir = suite_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    suite = validate_candidate_suite(suite_dir)
    checkpoint_configs = validate_config_snapshot(
        checkpoint_dir / "experiment_configs"
    )
    training_manifest_path = checkpoint_dir / "memory_training_manifest.json"
    training = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    program = training.get("program")
    if not isinstance(program, str) or not program:
        raise ValueError("checkpoint training manifest has no memory program")
    trained_config_name = (
        "training_empty_mem0.json" if program == "none" else f"fixed_{program}.json"
    )
    trained_config = checkpoint_dir / "experiment_configs" / trained_config_name
    if not trained_config.is_file():
        raise ValueError(
            f"checkpoint snapshot lacks its trained program: {trained_config_name}"
        )
    checkpoint_shape = _mem0_shape(trained_config)
    if checkpoint_shape is None:
        raise ValueError("memory checkpoint training program must use mem0_context")

    candidate_configs = suite_dir / "experiment_configs"
    candidate_contracts: list[dict[str, Any]] = []
    for row in suite["architectures"]:
        executor = candidate_configs / str(row["executor_program"])
        shape = _mem0_shape(executor)
        if shape is not None and shape != checkpoint_shape:
            raise ValueError(
                "candidate/checkpoint Mem-0 shape mismatch: "
                f"architecture={row['alias']}, candidate={shape}, "
                f"checkpoint={checkpoint_shape}"
            )
        candidate_contracts.append(
            {
                "alias": row["alias"],
                "executor_program_sha256": row["executor_program_sha256"],
                "mem0_context_shape": None if shape is None else list(shape),
            }
        )
    metadata = checkpoint_dir / "params" / "_METADATA"
    if not metadata.is_file():
        raise ValueError("checkpoint parameter metadata is missing")
    return {
        "schema_version": "memory_harness.candidate_checkpoint_preflight/v1",
        "candidate_suite_manifest_sha256": _sha256(
            suite_dir / "candidate_suite_manifest.json"
        ),
        "checkpoint": str(checkpoint_dir),
        "checkpoint_parameter_metadata_sha256": _sha256(metadata),
        "checkpoint_training_manifest_sha256": _sha256(training_manifest_path),
        "checkpoint_config_source_sha256": checkpoint_configs["source_sha256"],
        "checkpoint_training_program": program,
        "checkpoint_mem0_context_shape": list(checkpoint_shape),
        "candidate_count": len(candidate_contracts),
        "candidate_contracts": candidate_contracts,
        "compatible": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    package_dir = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create or validate an immutable rollout candidate suite."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--runtime-source", type=pathlib.Path, default=package_dir)
    create.add_argument(
        "--config-source", type=pathlib.Path, default=package_dir.parent / "configs"
    )
    create.add_argument("--output-dir", type=pathlib.Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--suite", type=pathlib.Path, required=True)
    checkpoint = commands.add_parser("validate-checkpoint")
    checkpoint.add_argument("--suite", type=pathlib.Path, required=True)
    checkpoint.add_argument("--checkpoint", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create":
        manifest = create_candidate_suite(
            runtime_source=args.runtime_source,
            config_source=args.config_source,
            output_dir=args.output_dir,
        )
    elif args.command == "validate":
        manifest = validate_candidate_suite(args.suite)
    else:
        manifest = validate_candidate_checkpoint(args.suite, args.checkpoint)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
