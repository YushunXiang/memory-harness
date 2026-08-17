from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memory_harness.compare_fixed_runs import PROTECTED_CONFIG_KEYS
from memory_harness.compare_training_runs import compare_training_runs
from memory_harness.compare_training_runs import training_chain


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _checkpoint(
    root: Path,
    name: str,
    *,
    program: str,
    updates: int,
    initial_params: Path,
) -> Path:
    checkpoint = root / name
    (checkpoint / "params").mkdir(parents=True)
    parent = initial_params.parent if initial_params.name == "params" else initial_params
    parent_manifest = parent / "memory_training_manifest.json"
    parent_evidence = {
        "parent_checkpoint": None,
        "parent_training_manifest_sha256": None,
    }
    if parent_manifest.is_file():
        parent_evidence = {
            "parent_checkpoint": str(parent.resolve()),
            "parent_training_manifest_sha256": hashlib.sha256(
                parent_manifest.read_bytes()
            ).hexdigest(),
        }
    _write_json(
        checkpoint / "memory_training_manifest.json",
        {
            "schema_version": "memory_harness.training/v1",
            "program": program,
            "optimizer_updates": updates,
            "effective_batch": 56,
            "task_config_sha256": "task-hash",
            "initial_weight_params": str(initial_params),
            **parent_evidence,
        },
    )
    return checkpoint


def _run(
    root: Path,
    name: str,
    *,
    checkpoint: Path,
    architecture: str,
    total_reward: float,
    model_config: str = "pi05_aloha_pen_uncap_mem0",
    task_progress: float | None = None,
) -> Path:
    run = root / name
    run.mkdir()
    config = {key: None for key in PROTECTED_CONFIG_KEYS}
    config.update(
        {
            "checkpoint_dir": str(checkpoint),
            "seed": 100000,
            "policy_seed_base": 120000,
            "num_episodes": 1,
            "max_steps": 500,
            "config": model_config,
            "policy_config": model_config,
        }
    )
    _write_json(run / "config.json", config)
    _write_json(
        run / "emac_manifest.json",
        {
            "architecture": architecture,
            "planner_model": None,
            "runtime_source_sha256": "runtime-hash",
            "config_source_sha256": "config-hash",
        },
    )
    episode = {
            "episode_index": 0,
            "seed": 100000,
            "policy_seed": 120000,
            "layout_fingerprint": None,
            "success": total_reward > 0,
            "steps": 500,
            "total_reward": total_reward,
            "final_info": {"max_reward": total_reward / 1000},
    }
    if task_progress is not None:
        episode["task_progress"] = {
            "task": "put_back_block",
            "max_progress_score": task_progress,
        }
    _write_json(run / "episodes.jsonl", episode)
    return run


def test_training_chain_accumulates_staged_optimizer_examples(tmp_path: Path) -> None:
    base_params = tmp_path / "base" / "params"
    base_params.mkdir(parents=True)
    none = _checkpoint(
        tmp_path, "none-u200", program="none", updates=200, initial_params=base_params
    )
    full = _checkpoint(
        tmp_path,
        "full-u1000",
        program="anchor_sliding",
        updates=1000,
        initial_params=none / "params",
    )

    chain = training_chain(full)

    assert chain["num_stages"] == 2
    assert chain["total_optimizer_updates"] == 1200
    assert chain["total_optimizer_examples"] == 1200 * 56
    assert chain["terminal_initial_params"] == str(base_params.resolve())
    assert chain["terminal_program"] == "anchor_sliding"
    assert chain["program_optimizer_updates"] == {
        "anchor_sliding": 1000,
        "none": 200,
    }
    assert chain["chronological_program_schedule"] == [
        {
            "program": "none",
            "optimizer_updates": 200,
            "effective_batch": 56,
            "optimizer_examples": 200 * 56,
        },
        {
            "program": "anchor_sliding",
            "optimizer_updates": 1000,
            "effective_batch": 56,
            "optimizer_examples": 1000 * 56,
        },
    ]
    assert chain["precondition_optimizer_updates"] == 200
    assert not chain["condition_from_terminal_initial_params"]


def test_training_chain_rejects_mutated_parent_manifest(tmp_path: Path) -> None:
    base_params = tmp_path / "base" / "params"
    base_params.mkdir(parents=True)
    parent = _checkpoint(
        tmp_path, "parent", program="none", updates=200, initial_params=base_params
    )
    child = _checkpoint(
        tmp_path,
        "child",
        program="anchor_sliding",
        updates=1000,
        initial_params=parent / "params",
    )
    parent_manifest = parent / "memory_training_manifest.json"
    value = json.loads(parent_manifest.read_text(encoding="utf-8"))
    value["optimizer_updates"] = 201
    _write_json(parent_manifest, value)

    with pytest.raises(ValueError, match="parent manifest hash mismatch"):
        training_chain(child)


def test_compares_budget_matched_training_variants(tmp_path: Path) -> None:
    base_params = tmp_path / "base" / "params"
    base_params.mkdir(parents=True)
    none_u200 = _checkpoint(
        tmp_path, "none-u200", program="none", updates=200, initial_params=base_params
    )
    full = _checkpoint(
        tmp_path,
        "full-u1000",
        program="anchor_sliding",
        updates=1000,
        initial_params=none_u200 / "params",
    )
    none = _checkpoint(
        tmp_path, "none-u1200", program="none", updates=1200, initial_params=base_params
    )
    reference = _run(
        tmp_path,
        "reference",
        checkpoint=none,
        architecture="none",
        total_reward=10.0,
    )
    candidate = _run(
        tmp_path,
        "candidate",
        checkpoint=full,
        architecture="anchor_sliding",
        total_reward=30.0,
        model_config="pi05_aloha_pen_uncap",
    )

    result = compare_training_runs(reference, candidate)

    assert result["status"] == "paired_total_budget_matched_training_variants"
    assert result["reference_training"]["total_optimizer_examples"] == 1200 * 56
    assert result["training_schedule_alignment"] == {
        "reference_terminal_program": "none",
        "candidate_terminal_program": "anchor_sliding",
        "reference_precondition_optimizer_updates": 0,
        "candidate_precondition_optimizer_updates": 200,
        "both_conditions_from_terminal_initial_params": False,
        "evidence_scope": "total_budget_matched_with_condition_warm_start",
    }
    assert result["aggregate"]["total_reward"]["mean_delta"] == 20.0


def test_compares_sparse_reward_task_progress(tmp_path: Path) -> None:
    base_params = tmp_path / "base" / "params"
    base_params.mkdir(parents=True)
    reference_checkpoint = _checkpoint(
        tmp_path, "none", program="none", updates=1200, initial_params=base_params
    )
    candidate_checkpoint = _checkpoint(
        tmp_path,
        "full",
        program="anchor_sliding",
        updates=1200,
        initial_params=base_params,
    )
    reference = _run(
        tmp_path,
        "reference",
        checkpoint=reference_checkpoint,
        architecture="none",
        total_reward=0.0,
        task_progress=0.0,
    )
    candidate = _run(
        tmp_path,
        "candidate",
        checkpoint=candidate_checkpoint,
        architecture="anchor_sliding",
        total_reward=0.0,
        task_progress=1.0,
    )

    result = compare_training_runs(reference, candidate)

    assert result["schema_version"] == "memory_harness.training_run_comparison/v3"
    assert result["training_schedule_alignment"][
        "both_conditions_from_terminal_initial_params"
    ]
    assert result["screening_metrics"] == ["task_progress_score"]
    assert result["aggregate"]["task_progress_score"]["mean_delta"] == 1.0


def test_rejects_unmatched_total_training_exposure(tmp_path: Path) -> None:
    base_params = tmp_path / "base" / "params"
    base_params.mkdir(parents=True)
    reference_checkpoint = _checkpoint(
        tmp_path, "none-u1000", program="none", updates=1000, initial_params=base_params
    )
    candidate_checkpoint = _checkpoint(
        tmp_path,
        "full-u1200",
        program="anchor_sliding",
        updates=1200,
        initial_params=base_params,
    )
    reference = _run(
        tmp_path,
        "reference",
        checkpoint=reference_checkpoint,
        architecture="none",
        total_reward=0.0,
    )
    candidate = _run(
        tmp_path,
        "candidate",
        checkpoint=candidate_checkpoint,
        architecture="anchor_sliding",
        total_reward=0.0,
    )

    with pytest.raises(ValueError, match="not budget matched"):
        compare_training_runs(reference, candidate)


def test_rejects_memory_variants_with_different_runtimes(tmp_path: Path) -> None:
    base_params = tmp_path / "base" / "params"
    base_params.mkdir(parents=True)
    reference_checkpoint = _checkpoint(
        tmp_path, "none", program="none", updates=1200, initial_params=base_params
    )
    candidate_checkpoint = _checkpoint(
        tmp_path,
        "full",
        program="anchor_sliding",
        updates=1200,
        initial_params=base_params,
    )
    reference = _run(
        tmp_path,
        "reference",
        checkpoint=reference_checkpoint,
        architecture="none",
        total_reward=0.0,
    )
    candidate = _run(
        tmp_path,
        "candidate",
        checkpoint=candidate_checkpoint,
        architecture="anchor_sliding",
        total_reward=0.0,
    )
    manifest = json.loads(
        (candidate / "emac_manifest.json").read_text(encoding="utf-8")
    )
    manifest["runtime_source_sha256"] = "different-runtime"
    _write_json(candidate / "emac_manifest.json", manifest)

    with pytest.raises(ValueError, match="different or missing frozen runtimes"):
        compare_training_runs(reference, candidate)


def test_rejects_memory_variants_with_different_config_snapshots(
    tmp_path: Path,
) -> None:
    base_params = tmp_path / "base" / "params"
    base_params.mkdir(parents=True)
    reference_checkpoint = _checkpoint(
        tmp_path, "none", program="none", updates=1200, initial_params=base_params
    )
    candidate_checkpoint = _checkpoint(
        tmp_path,
        "full",
        program="anchor_sliding",
        updates=1200,
        initial_params=base_params,
    )
    reference = _run(
        tmp_path,
        "reference",
        checkpoint=reference_checkpoint,
        architecture="none",
        total_reward=0.0,
    )
    candidate = _run(
        tmp_path,
        "candidate",
        checkpoint=candidate_checkpoint,
        architecture="anchor_sliding",
        total_reward=0.0,
    )
    manifest = json.loads(
        (candidate / "emac_manifest.json").read_text(encoding="utf-8")
    )
    manifest["config_source_sha256"] = "different-config"
    _write_json(candidate / "emac_manifest.json", manifest)

    with pytest.raises(ValueError, match="different or missing frozen config snapshots"):
        compare_training_runs(reference, candidate)
