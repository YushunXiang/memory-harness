from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from memory_harness.config_snapshot import create_config_snapshot
from memory_harness.config_snapshot import validate_config_snapshot
from memory_harness.runtime_snapshot import validate_runtime_snapshot


SCRIPT = Path(__file__).parents[1] / "scripts" / "finalize_pi05_memory_training.py"
SPEC = importlib.util.spec_from_file_location("finalize_pi05_memory_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _inputs(tmp_path: Path) -> argparse.Namespace:
    checkpoint = tmp_path / "checkpoint"
    initial_params = tmp_path / "initial" / "params"
    (checkpoint / "params").mkdir(parents=True)
    initial_params.mkdir(parents=True)
    task = tmp_path / "task.json"
    context = tmp_path / "context.json"
    program_config = tmp_path / "program.json"
    audit = tmp_path / "audit.json"
    bank = tmp_path / "bank.npz"
    log = tmp_path / "train.log"
    _write_json(
        task,
        {"schema_version": "memory_harness.task/v1", "task_name": "put_back_block"},
    )
    _write_json(program_config, {"name": "anchor_sliding"})
    program_hash = hashlib.sha256(program_config.read_bytes()).hexdigest()
    _write_json(
        context,
        {
            "schema_version": "emac_mem0_context/v4",
            "inputs": {
                "task_name": "put_back_block",
                "task_memory_complexity": "M(1)",
            },
            "representation": {
                "program": "anchor_sliding",
                "program_config": str(program_config),
                "program_config_sha256": program_hash,
            },
            "segments": [{"source_episode_id": 0}],
        },
    )
    _write_json(
        audit,
        {
            "schema_version": "emac_mem0_pairing_audit/v4",
            "ready_for_adapter_training": True,
            "checks": {
                check: True for check in MODULE.REQUIRED_PAIRING_CHECKS
            },
            "program_counts": {"anchor_sliding": 1},
            "num_segments": 1,
            "num_bank_items": 1,
        },
    )
    bank.write_bytes(b"bank")
    log.write_text("training\n", encoding="utf-8")
    return argparse.Namespace(
        checkpoint=checkpoint,
        task_config=task,
        context_manifest=context,
        pairing_audit=audit,
        context_bank=bank,
        initial_weight_params=initial_params,
        program="anchor_sliding",
        batch_size=2,
        accumulate_steps=28,
        optimizer_updates=1000,
        learning_rate=1e-5,
        training_log=[log],
        program_migration_audit=None,
    )


def test_validates_consistent_training_provenance(tmp_path: Path) -> None:
    validated = MODULE._validate_training_inputs(_inputs(tmp_path))

    assert validated == {
        "task_name": "put_back_block",
        "task_memory_complexity": "M(1)",
        "num_segments": 1,
        "num_bank_items": 1,
        "pairing_audit_ready": True,
        "program_config": str((tmp_path / "program.json").resolve()),
        "program_config_sha256": hashlib.sha256(
            (tmp_path / "program.json").read_bytes()
        ).hexdigest(),
        "program_config_alignment": "exact",
        "program_migration_audit_sha256": None,
        "training_segments": [
            {
                "training_log": str((tmp_path / "train.log").resolve()),
                "training_log_sha256": hashlib.sha256(
                    (tmp_path / "train.log").read_bytes()
                ).hexdigest(),
                "same_run_restores": [],
            }
        ],
        "resume_chain_verified": True,
    }


def test_rejects_program_mismatch(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    args.program = "sliding"

    with pytest.raises(ValueError, match="program and context manifest"):
        MODULE._validate_training_inputs(args)


def test_rejects_failed_pairing_audit(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    audit = json.loads(args.pairing_audit.read_text(encoding="utf-8"))
    audit["ready_for_adapter_training"] = False
    _write_json(args.pairing_audit, audit)

    with pytest.raises(ValueError, match="not ready"):
        MODULE._validate_training_inputs(args)


def test_rejects_incomplete_pairing_audit(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    audit = json.loads(args.pairing_audit.read_text(encoding="utf-8"))
    audit["checks"].pop("no_privileged_state")
    _write_json(args.pairing_audit, audit)

    with pytest.raises(ValueError, match="missing required checks"):
        MODULE._validate_training_inputs(args)


def test_rejects_changed_program_without_bound_migration_audit(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    program_config = Path(
        json.loads(args.context_manifest.read_text(encoding="utf-8"))["representation"][
            "program_config"
        ]
    )
    _write_json(program_config, {"name": "changed"})

    with pytest.raises(ValueError, match="hashes differ without a migration audit"):
        MODULE._validate_training_inputs(args)


def test_accepts_changed_program_with_bound_migration_audit(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    context = json.loads(args.context_manifest.read_text(encoding="utf-8"))
    source_hash = context["representation"]["program_config_sha256"]
    program_config = Path(context["representation"]["program_config"])
    _write_json(program_config, {"name": "changed"})
    target_hash = hashlib.sha256(program_config.read_bytes()).hexdigest()
    migration = tmp_path / "migration.json"
    _write_json(
        migration,
        {
            "schema_version": "memory_harness.program_migration_audit/v1",
            "migration": "implicit_path_roles_to_explicit_mem0_roles",
            "ready_for_context_reuse": True,
            "declared_program_config_sha256": source_hash,
            "source_config_sha256": source_hash,
            "target_config_sha256": target_hash,
            "context_manifest_sha256": hashlib.sha256(
                args.context_manifest.read_bytes()
            ).hexdigest(),
            "checks": {
                check: True for check in MODULE.REQUIRED_PROGRAM_MIGRATION_CHECKS
            },
        },
    )
    args.program_migration_audit = migration

    validated = MODULE._validate_training_inputs(args)

    assert validated["program_config_alignment"] == "verified_semantic_migration"
    assert validated["program_migration_audit_sha256"] == hashlib.sha256(
        migration.read_bytes()
    ).hexdigest()


def test_validates_program_from_the_snapshot_that_will_be_attached(
    tmp_path: Path,
) -> None:
    args = _inputs(tmp_path)
    declared = Path(
        json.loads(args.context_manifest.read_text(encoding="utf-8"))[
            "representation"
        ]["program_config"]
    )
    source = tmp_path / "config_source"
    source.mkdir()
    (source / declared.name).write_bytes(declared.read_bytes())
    snapshot = tmp_path / "config_snapshot"
    create_config_snapshot(source, snapshot)
    _write_json(declared, {"name": "live-config-changed-after-launch"})
    args.config_snapshot_source = snapshot

    validated = MODULE._validate_training_inputs(args)

    assert validated["program_config"] == str(
        (snapshot / declared.name).resolve()
    )
    assert validated["program_config_alignment"] == "exact"


def test_accepts_quota_migration_with_budget_check(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    context = json.loads(args.context_manifest.read_text(encoding="utf-8"))
    source_hash = context["representation"]["program_config_sha256"]
    program_config = Path(context["representation"]["program_config"])
    _write_json(
        program_config,
        {
            "name": "anchor_sliding",
            "utilizer": {
                "options": {"history_path_quotas": {"sliding": 30}}
            },
        },
    )
    target_hash = hashlib.sha256(program_config.read_bytes()).hexdigest()
    migration = tmp_path / "quota_migration.json"
    checks = {check: True for check in MODULE.REQUIRED_PROGRAM_MIGRATION_CHECKS}
    checks["history_budget_matches_legacy"] = True
    _write_json(
        migration,
        {
            "schema_version": "memory_harness.program_migration_audit/v1",
            "migration": "implicit_path_roles_to_explicit_mem0_path_quotas",
            "ready_for_context_reuse": True,
            "declared_program_config_sha256": source_hash,
            "source_config_sha256": source_hash,
            "target_config_sha256": target_hash,
            "context_manifest_sha256": hashlib.sha256(
                args.context_manifest.read_bytes()
            ).hexdigest(),
            "checks": checks,
        },
    )
    args.program_migration_audit = migration

    validated = MODULE._validate_training_inputs(args)

    assert validated["program_config_alignment"] == "verified_semantic_migration"


def test_rejects_migration_audit_with_failed_replay_check(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    context = json.loads(args.context_manifest.read_text(encoding="utf-8"))
    source_hash = context["representation"]["program_config_sha256"]
    program_config = Path(context["representation"]["program_config"])
    _write_json(program_config, {"name": "changed"})
    target_hash = hashlib.sha256(program_config.read_bytes()).hexdigest()
    migration = tmp_path / "migration.json"
    checks = {check: True for check in MODULE.REQUIRED_PROGRAM_MIGRATION_CHECKS}
    checks["layout_replay_exact"] = False
    _write_json(
        migration,
        {
            "schema_version": "memory_harness.program_migration_audit/v1",
            "migration": "implicit_path_roles_to_explicit_mem0_roles",
            "ready_for_context_reuse": False,
            "declared_program_config_sha256": source_hash,
            "source_config_sha256": source_hash,
            "target_config_sha256": target_hash,
            "context_manifest_sha256": hashlib.sha256(
                args.context_manifest.read_bytes()
            ).hexdigest(),
            "checks": checks,
        },
    )
    args.program_migration_audit = migration

    with pytest.raises(ValueError, match="contains a failed check"):
        MODULE._validate_training_inputs(args)


def test_accepts_ordered_training_segments_bound_to_resume_checkpoint(
    tmp_path: Path,
) -> None:
    args = _inputs(tmp_path)
    final_checkpoint = tmp_path / "experiment" / "28000"
    (final_checkpoint / "params").mkdir(parents=True)
    resume_checkpoint = final_checkpoint.parent / "16800"
    (resume_checkpoint / "params").mkdir(parents=True)
    (resume_checkpoint / "_CHECKPOINT_METADATA").write_text(
        "committed\n", encoding="utf-8"
    )
    resume_log = tmp_path / "resume.log"
    resume_log.write_text(
        "Running on: test-host (train.py:209)\n"
        f"Restoring checkpoint from {resume_checkpoint.resolve()}.\n"
        f"Finished restoring checkpoint in 1 second from {resume_checkpoint.resolve()}.\n",
        encoding="utf-8",
    )
    args.training_log[0].write_text(
        "Running on: test-host (train.py:209)\n", encoding="utf-8"
    )
    args.checkpoint = final_checkpoint
    args.training_log = [args.training_log[0], resume_log]

    validated = MODULE._validate_training_inputs(args)

    assert validated["resume_chain_verified"] is True
    assert validated["training_segments"][1]["same_run_restores"] == [
        {
            "checkpoint": str(resume_checkpoint.resolve()),
            "checkpoint_retained": True,
            "checkpoint_metadata_sha256": hashlib.sha256(
                (resume_checkpoint / "_CHECKPOINT_METADATA").read_bytes()
            ).hexdigest(),
        }
    ]


def test_accepts_appended_log_with_initial_and_resumed_runs(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    final_checkpoint = tmp_path / "experiment" / "28000"
    (final_checkpoint / "params").mkdir(parents=True)
    resume_checkpoint = final_checkpoint.parent / "16800"
    (resume_checkpoint / "params").mkdir(parents=True)
    (resume_checkpoint / "_CHECKPOINT_METADATA").write_text(
        "committed\n", encoding="utf-8"
    )
    args.checkpoint = final_checkpoint
    args.training_log[0].write_text(
        "Running on: first-host (train.py:209)\n"
        "stopped\n"
        "Running on: second-host (train.py:209)\n"
        f"Restoring checkpoint from {resume_checkpoint.resolve()}.\n"
        f"Finished restoring checkpoint in 1 second from {resume_checkpoint.resolve()}.\n",
        encoding="utf-8",
    )

    validated = MODULE._validate_training_inputs(args)

    assert validated["resume_chain_verified"] is True
    assert validated["training_segments"][0]["same_run_restores"][0][
        "checkpoint_metadata_sha256"
    ] == hashlib.sha256(
        (resume_checkpoint / "_CHECKPOINT_METADATA").read_bytes()
    ).hexdigest()


def test_rejects_resume_log_without_initial_run_segment(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    resume_checkpoint = args.checkpoint.parent / "16800"
    args.training_log[0].write_text(
        "Running on: resumed-host (train.py:209)\n"
        f"Restoring checkpoint from {resume_checkpoint.resolve()}.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one initial run plus one run per"):
        MODULE._validate_training_inputs(args)


def test_accepts_deleted_resume_checkpoint_with_prior_commit_evidence(
    tmp_path: Path,
) -> None:
    args = _inputs(tmp_path)
    final_checkpoint = tmp_path / "experiment" / "28000"
    (final_checkpoint / "params").mkdir(parents=True)
    deleted_checkpoint = final_checkpoint.parent / "5599"
    initial_log = args.training_log[0]
    initial_log.write_text(
        "Running on: first-host (train.py:209)\n"
        "Finished saving checkpoint (finalized tmp dir) to "
        f"`{deleted_checkpoint.resolve()}`.\n"
        f"TRAINED_CHECKPOINT={deleted_checkpoint.resolve()}\n",
        encoding="utf-8",
    )
    resume_log = tmp_path / "resume.log"
    resume_log.write_text(
        "Running on: second-host (train.py:209)\n"
        f"Restoring checkpoint from {deleted_checkpoint.resolve()}.\n"
        "Finished restoring checkpoint in 1 second from "
        f"{deleted_checkpoint.resolve()}.\n",
        encoding="utf-8",
    )
    args.checkpoint = final_checkpoint
    args.training_log = [initial_log, resume_log]

    validated = MODULE._validate_training_inputs(args)

    assert validated["training_segments"][1]["same_run_restores"] == [
        {
            "checkpoint": str(deleted_checkpoint.resolve()),
            "checkpoint_retained": False,
            "commit_evidence": (
                "prior_log_finalized_save_and_training_finalizer_completion"
            ),
        }
    ]


def test_archives_training_logs_inside_final_checkpoint(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    segments = MODULE._validate_training_inputs(args)["training_segments"]

    archived = MODULE._archive_training_segments(args.checkpoint, segments)

    destination = args.checkpoint / "training_logs" / "000_train.log"
    assert destination.read_text(encoding="utf-8") == "training\n"
    assert archived == [
        {
            **segments[0],
            "source_training_log": str(args.training_log[0]),
            "training_log": str(destination.resolve()),
        }
    ]
    with pytest.raises(FileExistsError, match="archive already exists"):
        MODULE._archive_training_segments(args.checkpoint, segments)


def test_main_freezes_runtime_and_configs(tmp_path: Path, monkeypatch) -> None:
    args = _inputs(tmp_path)
    args.runtime_snapshot_source = None
    args.config_snapshot_source = None
    monkeypatch.setattr(MODULE, "parse_args", lambda: args)

    assert MODULE.main() == 0

    runtime = validate_runtime_snapshot(args.checkpoint / "runtime")
    configs = validate_config_snapshot(args.checkpoint / "experiment_configs")
    manifest = json.loads(
        (args.checkpoint / "memory_training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["runtime_source_sha256"] == runtime["source_sha256"]
    assert manifest["config_source_sha256"] == configs["source_sha256"]


def test_main_reuses_paired_runtime_and_configs(tmp_path: Path, monkeypatch) -> None:
    source_args = _inputs(tmp_path / "source")
    source_args.runtime_snapshot_source = None
    source_args.config_snapshot_source = None
    monkeypatch.setattr(MODULE, "parse_args", lambda: source_args)
    assert MODULE.main() == 0

    paired_args = _inputs(tmp_path / "paired")
    paired_args.runtime_snapshot_source = source_args.checkpoint / "runtime"
    paired_args.config_snapshot_source = (
        source_args.checkpoint / "experiment_configs"
    )
    frozen_program = (
        paired_args.config_snapshot_source / "fixed_anchor_sliding.json"
    )
    paired_context = json.loads(
        paired_args.context_manifest.read_text(encoding="utf-8")
    )
    paired_context["representation"]["program_config"] = str(
        SCRIPT.parents[1] / "configs" / "fixed_anchor_sliding.json"
    )
    paired_context["representation"]["program_config_sha256"] = hashlib.sha256(
        frozen_program.read_bytes()
    ).hexdigest()
    _write_json(paired_args.context_manifest, paired_context)
    monkeypatch.setattr(MODULE, "parse_args", lambda: paired_args)
    assert MODULE.main() == 0

    source_manifest = json.loads(
        (source_args.checkpoint / "memory_training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    paired_manifest = json.loads(
        (paired_args.checkpoint / "memory_training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert paired_manifest["runtime_source_sha256"] == source_manifest[
        "runtime_source_sha256"
    ]
    assert paired_manifest["config_source_sha256"] == source_manifest[
        "config_source_sha256"
    ]
