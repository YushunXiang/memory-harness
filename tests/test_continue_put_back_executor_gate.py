from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from memory_harness.config_snapshot import create_config_snapshot
from memory_harness.runtime_snapshot import create_runtime_snapshot


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "memory-harness" / "scripts" / "continue_put_back_executor_gate.sh"


def _run_branch(
    tmp_path: Path,
    action: str,
    *,
    mismatched_runtime: bool = False,
    completed_native: bool = False,
) -> str:
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": "memory_harness.executor_readiness/v2",
                "decision": {"next_action": action},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inputs = []
    for name in ("full", "empty", "native"):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        inputs.append(checkpoint)
    full, empty, _ = inputs
    create_runtime_snapshot(
        ROOT / "memory-harness" / "memory_harness", full / "runtime"
    )
    create_config_snapshot(
        ROOT / "memory-harness" / "configs", full / "experiment_configs"
    )
    shutil.copytree(full / "runtime", empty / "runtime")
    shutil.copytree(full / "experiment_configs", empty / "experiment_configs")
    if mismatched_runtime:
        shutil.rmtree(empty / "runtime")
        create_runtime_snapshot(ROOT / "memory-harness" / "tests", empty / "runtime")
    checkpoint_root = tmp_path / "checkpoints"
    if completed_native:
        checkpoint = (
            checkpoint_root
            / "pi05_aloha_pen_uncap_mem0_control"
            / "emac_put_back_block_native_none_plus1800_to_u3000_b2a28_test"
            / "50399"
        )
        (checkpoint / "params").mkdir(parents=True)
        metadata = checkpoint / "_CHECKPOINT_METADATA"
        metadata.write_text("committed\n", encoding="utf-8")
        manifest = {
            "schema_version": "memory_harness.training/v1",
            "checkpoint_step": 50399,
            "checkpoint_commit_verified": True,
            "checkpoint_metadata_sha256": hashlib.sha256(
                metadata.read_bytes()
            ).hexdigest(),
            "optimizer_updates": 1800,
            "effective_batch": 56,
            "program": "native_none",
            "initial_weight_params": str((inputs[2] / "params").resolve()),
            "parent_checkpoint": str(inputs[2].resolve()),
        }
        (checkpoint / "memory_training_manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )

    result = subprocess.run(
        ["bash", str(SCRIPT), str(decision), *(str(path) for path in inputs)],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "PYTHON": sys.executable,
            "CHECKPOINT_BASE_DIR": str(checkpoint_root),
            "RUN_ROOT": str(tmp_path / "runs"),
            "DATE_TAG": "test",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_rejects_parent_checkpoints_with_different_runtime_snapshots(
    tmp_path: Path,
) -> None:
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run_branch(
            tmp_path,
            "increase_training_budget_before_more_rollouts",
            mismatched_runtime=True,
        )

    assert "different runtime/config snapshots" in exc_info.value.stderr


def test_ready_branch_collects_disjoint_gate17_for_all_fixed_memories(
    tmp_path: Path,
) -> None:
    output = _run_branch(tmp_path, "collect_fixed_ablation_to_20")

    for architecture in ("none", "anchor", "sliding", "anchor_sliding"):
        assert f"emac_put_back_block_{architecture}_u1200_gate17_test" in output
    for candidate in ("anchor", "sliding", "anchor_sliding"):
        assert f"none_vs_{candidate}_u1200_gate20.utility.json" in output
    assert "continue_put_back_candidate_screen.sh" in output
    assert "run_pi05_memory_train.sh" not in output
    assert "run_pi05_baseline_train.sh" not in output


@pytest.mark.parametrize(
    ("action", "expected_order"),
    [
        (
            "increase_training_budget_before_more_rollouts",
            (
                "EXP_NAME=emac_put_back_block_native_none_plus1800",
                "EXP_NAME=emac_put_back_block_anchor_sliding_plus1800",
                "EXP_NAME=emac_put_back_block_none_plus1800",
            ),
        ),
        (
            "retrain_full_memory_at_higher_budget_before_gate20",
            (
                "EXP_NAME=emac_put_back_block_anchor_sliding_plus1800",
                "EXP_NAME=emac_put_back_block_native_none_plus1800",
                "EXP_NAME=emac_put_back_block_none_plus1800",
            ),
        ),
    ],
)
def test_budget_extension_branches_preserve_pre_registered_training_order(
    tmp_path: Path,
    action: str,
    expected_order: tuple[str, ...],
) -> None:
    output = _run_branch(tmp_path, action)

    positions = tuple(output.index(fragment) for fragment in expected_order)
    assert positions == tuple(sorted(positions))
    assert output.count("run_pi05_memory_train.sh") == 2
    assert output.count("run_pi05_baseline_train.sh") == 1
    assert f"RUNTIME_SNAPSHOT_SOURCE={tmp_path / 'full' / 'runtime'}" in output
    assert (
        f"CONFIG_SNAPSHOT_SOURCE={tmp_path / 'full' / 'experiment_configs'}" in output
    )
    assert "put_back_block_executor_readiness_u3000_gate3.json" in output
    assert "put_back_block_native_vs_full_budget_matched_u3000_gate3.json" in output
    assert "put_back_block_empty_arch_vs_full_budget_matched_u3000_gate3.json" in output
    for architecture in ("none", "anchor", "sliding", "anchor_sliding"):
        assert f"emac_put_back_block_{architecture}_u3000_gate17_test" in output
    assert "continue_put_back_candidate_screen.sh" in output


def test_all_floor_branch_reuses_a_verified_native_stage(tmp_path: Path) -> None:
    output = _run_branch(
        tmp_path,
        "increase_training_budget_before_more_rollouts",
        completed_native=True,
    )

    assert "REUSE_COMPLETED_STAGE=" in output
    assert "run_pi05_baseline_train.sh" not in output
    assert output.count("run_pi05_memory_train.sh") == 2
