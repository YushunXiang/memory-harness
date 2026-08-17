from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_harness.candidate_suite import create_candidate_suite
from memory_harness.candidate_suite import validate_candidate_suite
from memory_harness.candidate_suite import validate_candidate_checkpoint
from memory_harness.config_snapshot import validate_config_snapshot
from memory_harness.config_snapshot import create_config_snapshot
from memory_harness.runtime_snapshot import validate_runtime_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_suite_discovers_freezes_and_smokes_all_architectures(
    tmp_path: Path,
) -> None:
    output = tmp_path / "suite"
    manifest = create_candidate_suite(
        runtime_source=ROOT / "memory_harness",
        config_source=ROOT / "configs",
        output_dir=output,
    )

    assert manifest["schema_version"] == "memory_harness.candidate_suite/v1"
    assert manifest["architecture_count"] == len(manifest["architectures"])
    aliases = {row["alias"] for row in manifest["architectures"]}
    assert {
        "none",
        "anchor_sliding",
        "recent_global",
        "key",
        "verified_success_latent",
    }.issubset(aliases)
    assert all(row["smoke_status"] == "completed" for row in manifest["architectures"])
    assert manifest["rollout_environment"] == {
        "MEMORY_RUNTIME_SNAPSHOT": "runtime",
        "MEMORY_CONFIG_SNAPSHOT": "experiment_configs",
    }
    validate_runtime_snapshot(output / "runtime")
    validate_config_snapshot(output / "experiment_configs")
    persisted = json.loads(
        (output / "candidate_suite_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted == manifest
    assert validate_candidate_suite(output) == manifest


def test_candidate_suite_refuses_to_replace_an_existing_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "suite"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        create_candidate_suite(
            runtime_source=ROOT / "memory_harness",
            config_source=ROOT / "configs",
            output_dir=output,
        )


def test_candidate_suite_rejects_smoke_artifact_mutation(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    manifest = create_candidate_suite(
        runtime_source=ROOT / "memory_harness",
        config_source=ROOT / "configs",
        output_dir=output,
    )
    row = next(row for row in manifest["architectures"] if row["alias"] == "recent_global")
    smoke = output / row["smoke_summary"]
    smoke.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="smoke summary changed"):
        validate_candidate_suite(output)


def test_candidate_suite_preflights_checkpoint_context_shape(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    manifest = create_candidate_suite(
        runtime_source=ROOT / "memory_harness",
        config_source=ROOT / "configs",
        output_dir=output,
    )
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "_METADATA").write_text("{}\n", encoding="utf-8")
    create_config_snapshot(ROOT / "configs", checkpoint / "experiment_configs")
    (checkpoint / "memory_training_manifest.json").write_text(
        json.dumps({"program": "anchor_sliding"}) + "\n",
        encoding="utf-8",
    )

    result = validate_candidate_checkpoint(output, checkpoint)

    assert result["compatible"] is True
    assert result["checkpoint_mem0_context_shape"] == [31, 2048]
    assert result["candidate_count"] == manifest["architecture_count"]
