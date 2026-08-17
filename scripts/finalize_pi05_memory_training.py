from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
from typing import Any

from memory_harness.config_snapshot import create_config_snapshot
from memory_harness.config_snapshot import copy_config_snapshot
from memory_harness.config_snapshot import validate_config_snapshot
from memory_harness.runtime_snapshot import copy_runtime_snapshot
from memory_harness.runtime_snapshot import create_runtime_snapshot
from memory_harness.training_provenance import parent_training_evidence


REQUIRED_PAIRING_CHECKS = frozenset(
    {
        "matched_uses_only_prior_observations",
        "mismatched_source_disjoint",
        "no_privileged_state",
        "same_runtime_program_implementation",
        "single_program_training",
        "subtask_prompt_and_lifecycle_aligned",
        "train_validation_episode_disjoint",
    }
)
REQUIRED_PROGRAM_MIGRATION_CHECKS = frozenset(
    {
        "anchor_role_matches_legacy",
        "history_roles_match_legacy",
        "layout_replay_exact",
        "manifest_source_hash_matches",
        "only_explicit_roles_added",
    }
)


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _program_config_for_final_snapshot(
    args: argparse.Namespace, declared_path: pathlib.Path
) -> pathlib.Path:
    snapshot_source = getattr(args, "config_snapshot_source", None)
    if snapshot_source is None:
        return declared_path
    snapshot_source = snapshot_source.resolve()
    manifest = validate_config_snapshot(snapshot_source)
    declared_paths = {str(row["path"]) for row in manifest["files"]}
    live_config_source = pathlib.Path(__file__).resolve().parents[1] / "configs"
    try:
        relative = declared_path.resolve().relative_to(live_config_source.resolve())
    except ValueError:
        relative = None
    if relative is not None and str(relative) in declared_paths:
        return snapshot_source / relative
    matches = [
        snapshot_source / str(row["path"])
        for row in manifest["files"]
        if pathlib.Path(str(row["path"])).name == declared_path.name
        and pathlib.Path(str(row["path"])).parent == pathlib.Path(".")
    ]
    if len(matches) != 1:
        raise ValueError(
            "frozen config snapshot must contain exactly one context program "
            f"named {declared_path.name!r}, found {len(matches)}"
        )
    return matches[0]


def _expected_migration_kind(program_config: pathlib.Path) -> tuple[str, set[str]]:
    config = _read_json(program_config)
    options = config.get("utilizer", {}).get("options", {})
    if isinstance(options, dict) and "history_path_quotas" in options:
        return (
            "implicit_path_roles_to_explicit_mem0_path_quotas",
            {"history_budget_matches_legacy"},
        )
    return "implicit_path_roles_to_explicit_mem0_roles", set()


def _training_segments(args: argparse.Namespace) -> list[dict[str, Any]]:
    logs = tuple(args.training_log)
    for path in logs:
        if not path.is_file():
            raise ValueError(f"missing training input: {path}")

    run_root = args.checkpoint.parent.resolve()
    restore_prefix = f"Restoring checkpoint from {run_root}/"
    observed_same_run_restores: list[pathlib.Path] = []
    run_start_count = 0
    restores_by_log: list[list[pathlib.Path]] = []
    log_contents: list[str] = []
    for path in logs:
        log_restores: list[pathlib.Path] = []
        content = path.read_text(encoding="utf-8", errors="replace")
        log_contents.append(content)
        for line in content.splitlines():
            if "Running on:" in line and "train.py:" in line:
                run_start_count += 1
            if restore_prefix not in line:
                continue
            restored = pathlib.Path(
                line.split(restore_prefix, maxsplit=1)[1].split(".", maxsplit=1)[0]
            )
            checkpoint = run_root / restored
            log_restores.append(checkpoint)
            observed_same_run_restores.append(checkpoint)
        restores_by_log.append(log_restores)
    if observed_same_run_restores and run_start_count != (
        len(observed_same_run_restores) + 1
    ):
        raise ValueError(
            "training logs do not contain one initial run plus one run per "
            f"same-run restore: starts={run_start_count}, "
            f"restores={len(observed_same_run_restores)}"
        )

    final_step = int(args.checkpoint.name) if args.checkpoint.name.isdigit() else None
    previous_step = -1
    segments: list[dict[str, Any]] = []
    ordered_log_text = "\n".join(log_contents)
    for log, resume_checkpoints in zip(logs, restores_by_log, strict=True):
        segment: dict[str, Any] = {
            "training_log": str(log.resolve()),
            "training_log_sha256": _sha256(log),
            "same_run_restores": [],
        }
        for checkpoint in resume_checkpoints:
            checkpoint = checkpoint.resolve()
            if checkpoint.parent != run_root:
                raise ValueError(f"resume checkpoint is outside the final run: {checkpoint}")
            if not checkpoint.name.isdigit():
                raise ValueError(f"resume checkpoint step is not numeric: {checkpoint}")
            step = int(checkpoint.name)
            if step <= previous_step or (final_step is not None and step >= final_step):
                raise ValueError("resume checkpoint steps must increase and precede final")
            metadata = checkpoint / "_CHECKPOINT_METADATA"
            restore_marker = f"Restoring checkpoint from {checkpoint}."
            restore_position = ordered_log_text.find(restore_marker)
            restore_completed = any(
                "Finished restoring checkpoint" in line
                and f"from {checkpoint}." in line
                for line in ordered_log_text[restore_position:].splitlines()
            )
            if restore_position < 0 or not restore_completed:
                raise ValueError(f"resume restore did not complete in logs: {checkpoint}")
            restore: dict[str, Any] = {
                "checkpoint": str(checkpoint),
                "checkpoint_retained": metadata.is_file(),
            }
            if metadata.is_file() and (checkpoint / "params").is_dir():
                restore["checkpoint_metadata_sha256"] = _sha256(metadata)
            else:
                save_marker = (
                    "Finished saving checkpoint (finalized tmp dir) to "
                    f"`{checkpoint}`."
                )
                trained_marker = f"TRAINED_CHECKPOINT={checkpoint}"
                save_position = ordered_log_text.rfind(
                    save_marker, 0, restore_position
                )
                trained_position = ordered_log_text.rfind(
                    trained_marker, 0, restore_position
                )
                if save_position < 0 or trained_position < 0:
                    raise ValueError(
                        "deleted resume checkpoint lacks prior committed-save evidence: "
                        f"{checkpoint}"
                    )
                restore["commit_evidence"] = (
                    "prior_log_finalized_save_and_training_finalizer_completion"
                )
            segment["same_run_restores"].append(restore)
            previous_step = step
        segments.append(segment)
    return segments


def _archive_training_segments(
    checkpoint: pathlib.Path, segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    archive = checkpoint / "training_logs"
    if archive.exists():
        raise FileExistsError(f"training log archive already exists: {archive}")
    archive.mkdir()
    archived: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        source = pathlib.Path(str(segment["training_log"]))
        destination = archive / f"{index:03d}_{source.name}"
        shutil.copy2(source, destination)
        expected_hash = str(segment["training_log_sha256"])
        if _sha256(destination) != expected_hash:
            raise RuntimeError(f"archived training log hash mismatch: {destination}")
        archived.append(
            {
                **segment,
                "source_training_log": str(source),
                "training_log": str(destination.resolve()),
            }
        )
    return archived


def _validate_training_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if not (args.checkpoint / "params").is_dir():
        raise ValueError(f"checkpoint has no params: {args.checkpoint}")
    if not args.initial_weight_params.is_dir():
        raise ValueError(f"initial weight params do not exist: {args.initial_weight_params}")
    if args.batch_size <= 0 or args.accumulate_steps <= 0 or args.optimizer_updates <= 0:
        raise ValueError("training budget values must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    inputs = (
        args.task_config,
        args.context_manifest,
        args.pairing_audit,
        args.context_bank,
    )
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"missing training input: {path}")
    training_segments = _training_segments(args)

    task = _read_json(args.task_config)
    context = _read_json(args.context_manifest)
    pairing = _read_json(args.pairing_audit)
    if task.get("schema_version") != "memory_harness.task/v1":
        raise ValueError("task config has an unsupported schema")
    if context.get("schema_version") != "emac_mem0_context/v4":
        raise ValueError("context manifest has an unsupported schema")
    if pairing.get("schema_version") != "emac_mem0_pairing_audit/v4":
        raise ValueError("pairing audit has an unsupported schema")
    if pairing.get("ready_for_adapter_training") is not True:
        raise ValueError("pairing audit is not ready for adapter training")
    checks = pairing.get("checks")
    if not isinstance(checks, dict) or not REQUIRED_PAIRING_CHECKS.issubset(checks):
        raise ValueError("pairing audit is missing required checks")
    if not all(value is True for value in checks.values()):
        raise ValueError("pairing audit contains a failed check")

    task_name = task.get("task_name")
    context_inputs = context.get("inputs", {})
    representation = context.get("representation", {})
    if context_inputs.get("task_name") != task_name:
        raise ValueError("task config and context manifest task names do not match")
    if representation.get("program") != args.program:
        raise ValueError("requested program and context manifest program do not match")
    declared_program_config = pathlib.Path(
        str(representation.get("program_config", ""))
    )
    program_config = _program_config_for_final_snapshot(
        args, declared_program_config
    )
    if not program_config.is_file():
        raise ValueError(f"context program config does not exist: {program_config}")
    declared_program_hash = str(representation.get("program_config_sha256", ""))
    current_program_hash = _sha256(program_config)
    migration: dict[str, Any] | None = None
    if current_program_hash != declared_program_hash:
        if args.program_migration_audit is None:
            raise ValueError(
                "context/current program config hashes differ without a migration audit"
            )
        migration = _read_json(args.program_migration_audit)
        if migration.get("schema_version") != "memory_harness.program_migration_audit/v1":
            raise ValueError("program migration audit has an unsupported schema")
        migration_checks = migration.get("checks")
        migration_kind, extra_checks = _expected_migration_kind(program_config)
        required_migration_checks = REQUIRED_PROGRAM_MIGRATION_CHECKS | extra_checks
        if not isinstance(migration_checks, dict) or not (
            required_migration_checks.issubset(migration_checks)
        ):
            raise ValueError("program migration audit is missing required checks")
        if not all(value is True for value in migration_checks.values()):
            raise ValueError("program migration audit contains a failed check")
        expected_migration = {
            "migration": migration_kind,
            "ready_for_context_reuse": True,
            "declared_program_config_sha256": declared_program_hash,
            "source_config_sha256": declared_program_hash,
            "target_config_sha256": current_program_hash,
            "context_manifest_sha256": _sha256(args.context_manifest),
        }
        mismatches = {
            key: {"expected": expected, "actual": migration.get(key)}
            for key, expected in expected_migration.items()
            if migration.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"program migration audit does not bind this run: {mismatches}")
    program_counts = pairing.get("program_counts")
    if program_counts != {args.program: pairing.get("num_segments")}:
        raise ValueError("pairing audit does not describe exactly the requested program")
    segments = context.get("segments")
    if not isinstance(segments, list) or len(segments) != pairing.get("num_segments"):
        raise ValueError("context manifest and pairing audit segment counts do not match")
    if pairing.get("num_bank_items") != len(segments):
        raise ValueError("context bank item count does not match the context manifest")

    return {
        "task_name": task_name,
        "task_memory_complexity": context_inputs.get("task_memory_complexity"),
        "num_segments": len(segments),
        "num_bank_items": pairing.get("num_bank_items"),
        "pairing_audit_ready": True,
        "program_config": str(program_config.resolve()),
        "program_config_sha256": current_program_hash,
        "program_config_alignment": (
            "exact" if migration is None else "verified_semantic_migration"
        ),
        "program_migration_audit_sha256": (
            None
            if args.program_migration_audit is None
            else _sha256(args.program_migration_audit)
        ),
        "training_segments": training_segments,
        "resume_chain_verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an auditable memory-training manifest.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--task-config", type=pathlib.Path, required=True)
    parser.add_argument("--context-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--pairing-audit", type=pathlib.Path, required=True)
    parser.add_argument("--context-bank", type=pathlib.Path, required=True)
    parser.add_argument("--initial-weight-params", type=pathlib.Path, required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulate-steps", type=int, required=True)
    parser.add_argument("--optimizer-updates", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument(
        "--training-log", type=pathlib.Path, action="append", required=True
    )
    parser.add_argument("--program-migration-audit", type=pathlib.Path)
    parser.add_argument(
        "--runtime-snapshot-source",
        type=pathlib.Path,
        help="Reuse a previously frozen runtime for a paired training branch.",
    )
    parser.add_argument(
        "--config-snapshot-source",
        type=pathlib.Path,
        help="Reuse previously frozen experiment configs for a paired training branch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validated = _validate_training_inputs(args)
    parent_evidence = parent_training_evidence(args.initial_weight_params)
    archived_segments = _archive_training_segments(
        args.checkpoint, validated["training_segments"]
    )
    live_source = pathlib.Path(__file__).resolve().parents[1] / "memory_harness"
    runtime_source = (
        live_source
        if args.runtime_snapshot_source is None
        else args.runtime_snapshot_source.resolve()
    )
    runtime_snapshot = (
        create_runtime_snapshot(runtime_source, args.checkpoint / "runtime")
        if args.runtime_snapshot_source is None
        else copy_runtime_snapshot(runtime_source, args.checkpoint / "runtime")
    )
    live_config_source = pathlib.Path(__file__).resolve().parents[1] / "configs"
    config_source = (
        live_config_source
        if args.config_snapshot_source is None
        else args.config_snapshot_source.resolve()
    )
    config_snapshot = (
        create_config_snapshot(config_source, args.checkpoint / "experiment_configs")
        if args.config_snapshot_source is None
        else copy_config_snapshot(
            config_source, args.checkpoint / "experiment_configs"
        )
    )
    validated = {
        **validated,
        "training_segments": archived_segments,
        "runtime_source_sha256": runtime_snapshot["source_sha256"],
        "config_source_sha256": config_snapshot["source_sha256"],
    }
    payload = {
        "schema_version": "memory_harness.training/v1",
        "task_config": str(args.task_config.resolve()),
        "task_config_sha256": _sha256(args.task_config),
        "context_manifest": str(args.context_manifest.resolve()),
        "context_manifest_sha256": _sha256(args.context_manifest),
        "pairing_audit_sha256": _sha256(args.pairing_audit),
        "context_bank_sha256": _sha256(args.context_bank),
        "initial_weight_params": str(args.initial_weight_params.resolve()),
        "program": args.program,
        "batch_size": args.batch_size,
        "accumulate_steps": args.accumulate_steps,
        "effective_batch": args.batch_size * args.accumulate_steps,
        "optimizer_updates": args.optimizer_updates,
        "learning_rate": args.learning_rate,
        "training_segments": archived_segments,
        "resume_chain_verified": validated["resume_chain_verified"],
        "runtime_snapshot": str((args.checkpoint / "runtime").resolve()),
        "runtime_snapshot_source": str(runtime_source),
        "runtime_source_sha256": runtime_snapshot["source_sha256"],
        "config_snapshot": str(
            (args.checkpoint / "experiment_configs").resolve()
        ),
        "config_snapshot_source": str(config_source),
        "config_source_sha256": config_snapshot["source_sha256"],
        **parent_evidence,
        "program_migration_audit": (
            None
            if args.program_migration_audit is None
            else str(args.program_migration_audit.resolve())
        ),
        "validated_inputs": validated,
    }
    output = args.checkpoint / "memory_training_manifest.json"
    if output.exists():
        raise FileExistsError(f"training manifest already exists: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
