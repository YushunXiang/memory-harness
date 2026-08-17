from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from memory_harness.search_candidate import create_search_candidate
from memory_harness.search_candidate import validate_search_candidate


ROOT = Path(__file__).resolve().parents[1]


def _submission(
    target: Path,
    *,
    hypothesis: str = "Split the fixed token budget across recent and global history.",
    parent: str | None = None,
) -> Path:
    target.mkdir()
    architecture = json.loads(
        (ROOT / "configs/architectures/fixed_recent_global.json").read_text(
            encoding="utf-8"
        )
    )
    architecture["executor_program"] = "executor_program.json"
    (target / "architecture.json").write_text(
        json.dumps(architecture) + "\n", encoding="utf-8"
    )
    shutil.copy2(
        ROOT / "configs/fixed_recent_global.json",
        target / "executor_program.json",
    )
    (target / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": "memory_harness.search_candidate_submission/v1",
                "hypothesis": hypothesis,
                "parent_content_sha256": parent,
                "search_axes": ["retriever", "store", "utilizer"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def test_search_candidate_freezes_typed_config_runtime_and_smoke_atomically(
    tmp_path: Path,
) -> None:
    submission = _submission(tmp_path / "submission")
    output = tmp_path / "candidate"

    manifest = create_search_candidate(
        submission_dir=submission,
        runtime_source=ROOT / "memory_harness",
        output_dir=output,
    )

    assert manifest["schema_version"] == "memory_harness.search_candidate/v1"
    assert manifest["status"] == "preflight_passed"
    assert manifest["activation"] == "inactive_until_fixed_memory_gate_1"
    assert manifest["candidate_kind"] == (
        "configuration_only_typed_operator_composition"
    )
    assert manifest["deployable"] is True
    assert len(manifest["candidate_content_sha256"]) == 64
    assert manifest["search_axes"] == ["retriever", "store", "utilizer"]
    assert (output / "runtime/runtime_manifest.json").is_file()
    assert (output / "smoke/memory_audit.jsonl").is_file()
    assert validate_search_candidate(output) == manifest


def test_candidate_behavior_identity_ignores_research_labels_but_includes_runtime(
    tmp_path: Path,
) -> None:
    first_submission = _submission(tmp_path / "first", hypothesis="first label")
    second_submission = _submission(
        tmp_path / "second",
        hypothesis="second label",
        parent="1" * 64,
    )
    architecture = json.loads(
        (second_submission / "architecture.json").read_text(encoding="utf-8")
    )
    architecture["name"] = "renamed-recent-global"
    (second_submission / "architecture.json").write_text(
        json.dumps(architecture) + "\n", encoding="utf-8"
    )
    program = json.loads(
        (second_submission / "executor_program.json").read_text(encoding="utf-8")
    )
    program["name"] = "renamed-program"
    (second_submission / "executor_program.json").write_text(
        json.dumps(program) + "\n", encoding="utf-8"
    )

    first = create_search_candidate(
        submission_dir=first_submission,
        runtime_source=ROOT / "memory_harness",
        output_dir=tmp_path / "first-artifact",
    )
    second = create_search_candidate(
        submission_dir=second_submission,
        runtime_source=ROOT / "memory_harness",
        output_dir=tmp_path / "second-artifact",
    )
    changed_runtime = tmp_path / "changed-runtime"
    shutil.copytree(ROOT / "memory_harness", changed_runtime)
    init_path = changed_runtime / "__init__.py"
    init_path.write_text(
        init_path.read_text(encoding="utf-8") + "\n# candidate identity test\n",
        encoding="utf-8",
    )
    third = create_search_candidate(
        submission_dir=first_submission,
        runtime_source=changed_runtime,
        output_dir=tmp_path / "third-artifact",
    )

    assert first["candidate_content_sha256"] == second["candidate_content_sha256"]
    assert first["candidate_content_sha256"] != third["candidate_content_sha256"]
    assert first["hypothesis"] != second["hypothesis"]
    assert second["parent_content_sha256"] == "1" * 64


def test_search_candidate_rejects_code_files_and_path_escape(tmp_path: Path) -> None:
    with_code = _submission(tmp_path / "with-code")
    (with_code / "operator.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        create_search_candidate(
            submission_dir=with_code,
            runtime_source=ROOT / "memory_harness",
            output_dir=tmp_path / "unused-code-output",
        )

    escaped = _submission(tmp_path / "escaped")
    architecture = json.loads(
        (escaped / "architecture.json").read_text(encoding="utf-8")
    )
    architecture["executor_program"] = "../outside.json"
    (escaped / "architecture.json").write_text(
        json.dumps(architecture) + "\n", encoding="utf-8"
    )
    (tmp_path / "outside.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local executor_program"):
        create_search_candidate(
            submission_dir=escaped,
            runtime_source=ROOT / "memory_harness",
            output_dir=tmp_path / "unused-escape-output",
        )


def test_search_candidate_rejects_undeployable_or_invalid_submission(
    tmp_path: Path,
) -> None:
    undeployable = _submission(tmp_path / "undeployable")
    program = json.loads(
        (undeployable / "executor_program.json").read_text(encoding="utf-8")
    )
    program["deployable"] = False
    (undeployable / "executor_program.json").write_text(
        json.dumps(program) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be deployable"):
        create_search_candidate(
            submission_dir=undeployable,
            runtime_source=ROOT / "memory_harness",
            output_dir=tmp_path / "unused-undeployable-output",
        )

    invalid = _submission(tmp_path / "invalid")
    submission = json.loads((invalid / "submission.json").read_text(encoding="utf-8"))
    submission["search_axes"] = ["benchmark", "retriever"]
    (invalid / "submission.json").write_text(
        json.dumps(submission) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown search axes"):
        create_search_candidate(
            submission_dir=invalid,
            runtime_source=ROOT / "memory_harness",
            output_dir=tmp_path / "unused-invalid-output",
        )


def test_search_candidate_validation_detects_config_mutation(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    create_search_candidate(
        submission_dir=_submission(tmp_path / "submission"),
        runtime_source=ROOT / "memory_harness",
        output_dir=output,
    )
    program = json.loads((output / "executor_program.json").read_text(encoding="utf-8"))
    program["paths"][0]["store"]["options"]["capacity"] = 14
    (output / "executor_program.json").write_text(
        json.dumps(program) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="executor changed"):
        validate_search_candidate(output)


def test_search_candidate_validation_rejects_injected_artifact_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"
    create_search_candidate(
        submission_dir=_submission(tmp_path / "submission"),
        runtime_source=ROOT / "memory_harness",
        output_dir=output,
    )
    (output / "operator.py").write_text("raise RuntimeError\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact file set mismatch"):
        validate_search_candidate(output)
