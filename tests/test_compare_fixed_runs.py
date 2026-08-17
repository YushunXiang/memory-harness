from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_harness.compare_fixed_runs import (
    PROTECTED_CONFIG_KEYS,
    compare_run_sets,
    compare_runs,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _run(
    root: Path,
    name: str,
    *,
    max_reward: float,
    total_reward: float,
    seed: int = 100000,
    task_progress: float | None = None,
) -> Path:
    run = root / name
    run.mkdir()
    config = {key: None for key in PROTECTED_CONFIG_KEYS}
    config.update(
        {
            "checkpoint_dir": "/checkpoint",
            "seed": seed,
            "policy_seed_base": seed + 20000,
            "num_episodes": 1,
            "max_steps": 1500,
        }
    )
    _write_json(run / "config.json", config)
    _write_json(
        run / "emac_manifest.json",
        {
            "architecture": name,
            "planner_model": None,
            "runtime_source_sha256": "runtime-hash",
            "config_source_sha256": "config-hash",
            "task_config_sha256": "task-hash",
            "architecture_config_sha256": f"architecture-{name}",
            "executor_program_config_sha256": f"executor-{name}",
        },
    )
    episode = {
            "episode_index": 0,
            "seed": seed,
            "policy_seed": seed + 20000,
            "layout_fingerprint": {"layout": seed},
            "success": False,
            "steps": 1500,
            "total_reward": total_reward,
            "final_info": {"max_reward": max_reward},
    }
    if task_progress is not None:
        episode["task_progress"] = {
            "task": "put_back_block",
            "max_progress_score": task_progress,
        }
    _write_json(run / "episodes.jsonl", episode)
    return run


def test_compare_runs_reports_strict_paired_deltas(tmp_path: Path) -> None:
    reference = _run(tmp_path, "sliding", max_reward=0.05, total_reward=60.0)
    candidate = _run(tmp_path, "consolidating", max_reward=0.15, total_reward=170.0)

    result = compare_runs(reference, candidate)

    assert result["num_pairs"] == 1
    assert result["aggregate"]["max_reward"]["mean_delta"] == pytest.approx(0.10)
    assert result["aggregate"]["total_reward"]["mean_delta"] == 110.0


def test_compare_runs_uses_put_back_progress_as_screening_metric(
    tmp_path: Path,
) -> None:
    reference = _run(
        tmp_path,
        "none",
        max_reward=0.0,
        total_reward=0.0,
        task_progress=0.0,
    )
    candidate = _run(
        tmp_path,
        "anchor_sliding",
        max_reward=0.0,
        total_reward=0.0,
        task_progress=1.0,
    )

    result = compare_runs(reference, candidate)

    assert result["schema_version"] == "memory_harness.fixed_run_comparison/v2"
    assert result["screening_metrics"] == ["task_progress_score"]
    assert result["aggregate"]["task_progress_score"]["mean_delta"] == 1.0


def test_compare_runs_rejects_unpaired_policy_seed(tmp_path: Path) -> None:
    reference = _run(tmp_path, "reference", max_reward=0.0, total_reward=0.0)
    candidate = _run(tmp_path, "candidate", max_reward=0.0, total_reward=0.0)
    episode = json.loads((candidate / "episodes.jsonl").read_text(encoding="utf-8"))
    episode["policy_seed"] = 120001
    _write_json(candidate / "episodes.jsonl", episode)

    with pytest.raises(ValueError, match="episode pairing mismatch"):
        compare_runs(reference, candidate)


def test_compare_run_sets_aggregates_multiple_paired_runs(tmp_path: Path) -> None:
    reference_a = _run(tmp_path, "reference_a", max_reward=0.10, total_reward=100.0)
    candidate_a = _run(tmp_path, "candidate_a", max_reward=0.15, total_reward=150.0)
    reference_b = _run(
        tmp_path,
        "reference_b",
        max_reward=0.10,
        total_reward=120.0,
        seed=100001,
    )
    candidate_b = _run(
        tmp_path,
        "candidate_b",
        max_reward=0.05,
        total_reward=60.0,
        seed=100001,
    )
    for run in (reference_a, reference_b):
        _write_json(
            run / "emac_manifest.json",
            {
                "architecture": "none",
                "planner_model": "none",
                "runtime_source_sha256": "runtime-hash",
                "config_source_sha256": "config-hash",
                "task_config_sha256": "task-hash",
                "architecture_config_sha256": "architecture-none",
                "executor_program_config_sha256": "executor-none",
            },
        )
    for run in (candidate_a, candidate_b):
        _write_json(
            run / "emac_manifest.json",
            {
                "architecture": "key",
                "planner_model": "key",
                "runtime_source_sha256": "runtime-hash",
                "config_source_sha256": "config-hash",
                "task_config_sha256": "task-hash",
                "architecture_config_sha256": "architecture-key",
                "executor_program_config_sha256": "executor-key",
            },
        )

    result = compare_run_sets([reference_a, reference_b], [candidate_a, candidate_b])

    assert result["status"] == "paired_run_set"
    assert result["num_run_pairs"] == 2
    assert result["num_pairs"] == 2
    assert result["evidence_identity"] == "seed+policy_seed+layout_fingerprint"
    assert result["aggregate"]["max_reward"]["mean_delta"] == pytest.approx(0.0)
    assert result["aggregate"]["total_reward"]["mean_delta"] == -5.0


def test_compare_run_sets_rejects_duplicate_episode_evidence(tmp_path: Path) -> None:
    reference_a = _run(tmp_path, "reference_a", max_reward=0.0, total_reward=0.0)
    candidate_a = _run(tmp_path, "candidate_a", max_reward=0.0, total_reward=0.0)
    reference_b = _run(tmp_path, "reference_b", max_reward=0.0, total_reward=0.0)
    candidate_b = _run(tmp_path, "candidate_b", max_reward=0.0, total_reward=0.0)
    for run in (reference_a, reference_b):
        _write_json(
            run / "emac_manifest.json",
            {
                "architecture": "none",
                "planner_model": None,
                "runtime_source_sha256": "runtime-hash",
                "config_source_sha256": "config-hash",
                "task_config_sha256": "task-hash",
                "architecture_config_sha256": "architecture-none",
                "executor_program_config_sha256": "executor-none",
            },
        )
    for run in (candidate_a, candidate_b):
        _write_json(
            run / "emac_manifest.json",
            {
                "architecture": "sliding",
                "planner_model": None,
                "runtime_source_sha256": "runtime-hash",
                "config_source_sha256": "config-hash",
                "task_config_sha256": "task-hash",
                "architecture_config_sha256": "architecture-sliding",
                "executor_program_config_sha256": "executor-sliding",
            },
        )

    with pytest.raises(ValueError, match="duplicate paired episode evidence"):
        compare_run_sets([reference_a, reference_b], [candidate_a, candidate_b])


def test_compare_runs_rejects_different_frozen_runtimes(tmp_path: Path) -> None:
    reference = _run(tmp_path, "reference", max_reward=0.0, total_reward=0.0)
    candidate = _run(tmp_path, "candidate", max_reward=0.0, total_reward=0.0)
    manifest = json.loads(
        (candidate / "emac_manifest.json").read_text(encoding="utf-8")
    )
    manifest["runtime_source_sha256"] = "different-runtime"
    _write_json(candidate / "emac_manifest.json", manifest)

    with pytest.raises(ValueError, match="different or missing frozen memory runtimes"):
        compare_runs(reference, candidate)


def test_compare_runs_rejects_different_frozen_configs(tmp_path: Path) -> None:
    reference = _run(tmp_path, "reference", max_reward=0.0, total_reward=0.0)
    candidate = _run(tmp_path, "candidate", max_reward=0.0, total_reward=0.0)
    manifest = json.loads(
        (candidate / "emac_manifest.json").read_text(encoding="utf-8")
    )
    manifest["config_source_sha256"] = "different-config"
    _write_json(candidate / "emac_manifest.json", manifest)

    with pytest.raises(ValueError, match="different or missing frozen config snapshots"):
        compare_runs(reference, candidate)


def test_compare_runs_rejects_different_candidate_suites(tmp_path: Path) -> None:
    reference = _run(tmp_path, "reference", max_reward=0.0, total_reward=0.0)
    candidate = _run(tmp_path, "candidate", max_reward=0.0, total_reward=0.0)
    for run, suite_hash in ((reference, "suite-a"), (candidate, "suite-b")):
        manifest = json.loads(
            (run / "emac_manifest.json").read_text(encoding="utf-8")
        )
        manifest["candidate_suite_manifest_sha256"] = suite_hash
        _write_json(run / "emac_manifest.json", manifest)

    with pytest.raises(ValueError, match="candidate-suite provenance"):
        compare_runs(reference, candidate)


def test_compare_run_sets_rejects_architecture_config_drift(tmp_path: Path) -> None:
    reference_a = _run(tmp_path, "reference_a", max_reward=0.0, total_reward=0.0)
    candidate_a = _run(tmp_path, "candidate_a", max_reward=0.0, total_reward=0.0)
    reference_b = _run(
        tmp_path, "reference_b", max_reward=0.0, total_reward=0.0, seed=100001
    )
    candidate_b = _run(
        tmp_path, "candidate_b", max_reward=0.0, total_reward=0.0, seed=100001
    )
    for run in (reference_a, reference_b):
        manifest = json.loads((run / "emac_manifest.json").read_text(encoding="utf-8"))
        manifest["architecture"] = "none"
        manifest["architecture_config_sha256"] = "architecture-none"
        manifest["executor_program_config_sha256"] = "executor-none"
        _write_json(run / "emac_manifest.json", manifest)
    for run in (candidate_a, candidate_b):
        manifest = json.loads((run / "emac_manifest.json").read_text(encoding="utf-8"))
        manifest["architecture"] = "sliding"
        manifest["architecture_config_sha256"] = "architecture-sliding"
        manifest["executor_program_config_sha256"] = "executor-sliding"
        _write_json(run / "emac_manifest.json", manifest)
    drifted = json.loads(
        (candidate_b / "emac_manifest.json").read_text(encoding="utf-8")
    )
    drifted["architecture_config_sha256"] = "changed-sliding"
    _write_json(candidate_b / "emac_manifest.json", drifted)

    with pytest.raises(
        ValueError, match="candidate_architecture_config_sha256"
    ):
        compare_run_sets([reference_a, reference_b], [candidate_a, candidate_b])
