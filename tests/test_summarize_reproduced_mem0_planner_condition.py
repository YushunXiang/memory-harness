from __future__ import annotations

import json
import pathlib

import pytest

from memory_harness.summarize_reproduced_mem0_planner_condition import (
    planner_output_accuracy,
    summarize,
)


def _write_planner_audit(path: pathlib.Path) -> None:
    rows = [
        {
            "event": "planner_output",
            "episode_index": 0,
            "environment_step": 0,
            "environment_pointer": 0,
            "planner_stage": 0,
            "color_positions_red_green_blue": {
                "red": "middle",
                "green": "right",
                "blue": "left",
            },
            "raw_output": "next_subtask: Cover the left block with the left cover.",
        },
        {
            "event": "planner_output",
            "episode_index": 0,
            "environment_step": 10,
            "environment_pointer": 3,
            "planner_stage": 3,
            "color_positions_red_green_blue": {
                "red": "middle",
                "green": "right",
                "blue": "left",
            },
            "raw_output": (
                "next_subtask: Open the right cover to uncover the blocks in the "
                "order of red, green, and blue."
            ),
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_scores_planner_outputs_against_diagnostic_state(tmp_path) -> None:
    audit = tmp_path / "audit.jsonl"
    _write_planner_audit(audit)

    result = planner_output_accuracy(audit)

    assert result["label_source"] == "privileged_rmbench_diagnostic_only"
    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["rate"] == 0.5
    assert result["by_planner_stage"]["0"]["rate"] == 1.0
    assert result["by_planner_stage"]["3"]["rate"] == 0.0
    assert result["errors"][0]["expected"].startswith(
        "next_subtask: Open the middle cover"
    )


def test_planner_accuracy_isolated_from_executor_pointer_stall(tmp_path) -> None:
    audit = tmp_path / "audit.jsonl"
    audit.write_text(
        json.dumps(
            {
                "event": "planner_output",
                "episode_index": 0,
                "environment_step": 100,
                "environment_pointer": 4,
                "planner_stage": 5,
                "color_positions_red_green_blue": {
                    "red": "middle",
                    "green": "left",
                    "blue": "right",
                },
                "raw_output": (
                    "next_subtask: Open the right cover to uncover the blocks in "
                    "the order of red, green, and blue."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = planner_output_accuracy(audit)

    assert result["correct"] == 1
    assert result["by_planner_stage"]["5"]["rate"] == 1.0


def test_summarizes_paired_no_key_run(tmp_path) -> None:
    log = tmp_path / "eval.log"
    log.write_text(
        "RMBENCH_LIMITED_EVAL_RESULT "
        + json.dumps(
            {
                "num_episodes": 2,
                "successes": 1,
                "success_rate": 0.5,
                "mean_reward": 0.5,
                "simulator_seeds": [100000, 100001],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    official_log = (
        tmp_path
        / "official_results"
        / "cover_blocks"
        / "mem0_episodic_policy"
        / "demo_clean"
        / "run"
        / "timestamp"
        / "eval_log.txt"
    )
    official_log.parent.mkdir(parents=True)
    official_log.write_text(
        "episode_id=0, seed=100000, result=Success\n"
        "episode_id=1, seed=100001, result=Fail\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "executor.pt"
    checkpoint.write_bytes(b"executor")
    planner = tmp_path / "planner"
    planner.mkdir()
    (planner / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    base_index = tmp_path / "base.index.json"
    base_index.write_text("{}", encoding="utf-8")
    pair_manifest = tmp_path / "pair.json"
    pair_manifest.write_text("{}", encoding="utf-8")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "simulator_seeds": [100000, 100001],
                "successes": 2,
                "success_rate": 1.0,
                "successful_seeds": [100000, 100001],
                "policy_seeds": [120000, 120001],
                "planner_online_exact": {
                    "correct": 2,
                    "total": 2,
                    "rate": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.jsonl"
    _write_planner_audit(audit)

    report = summarize(
        eval_log=log,
        audit_log=audit,
        condition="no_key",
        checkpoint=checkpoint,
        planner_model=planner,
        planner_adapter=adapter,
        base_model_index=base_index,
        training_pair_manifest=pair_manifest,
        seed_start=100000,
        policy_seed_base=120000,
        reference_summary=reference,
    )

    assert report["simulator_seeds"] == [100000, 100001]
    assert report["schema_version"].endswith("/v3")
    assert report["policy_seeds"] == [120000, 120001]
    assert report["successful_seeds"] == [100000]
    assert report["planner_output_accuracy"]["rate"] == 0.5
    assert report["comparison"]["candidate_minus_reference_success_rate"] == -0.5
    assert report["comparison"]["candidate_minus_reference_planner_accuracy"] == -0.5
    assert report["comparison"]["paired_success_transitions"] == {
        "both_success": 1,
        "full_only_success": 1,
        "ablation_only_success": 0,
        "both_fail": 0,
    }
    assert "not an exact replay" in report["scope_note"]


def test_rejects_unpaired_reference_seeds(tmp_path) -> None:
    log = tmp_path / "eval.log"
    log.write_text(
        "RMBENCH_LIMITED_EVAL_RESULT "
        '{"num_episodes": 1, "successes": 0, "success_rate": 0, '
        '"mean_reward": 0, "simulator_seeds": [100000]}\n',
        encoding="utf-8",
    )
    official_log = tmp_path / "official_results" / "run" / "eval_log.txt"
    official_log.parent.mkdir(parents=True)
    official_log.write_text(
        "episode_id=0, seed=100000, result=Fail\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "executor.pt"
    checkpoint.write_bytes(b"executor")
    planner = tmp_path / "planner"
    planner.mkdir()
    (planner / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    base_index = tmp_path / "base.index.json"
    base_index.write_text("{}", encoding="utf-8")
    pair_manifest = tmp_path / "pair.json"
    pair_manifest.write_text("{}", encoding="utf-8")
    reference = tmp_path / "reference.json"
    reference.write_text(
        '{"simulator_seeds": [7], "successes": 0, "success_rate": 0, '
        '"planner_online_exact": {"correct": 0, "total": 1, "rate": 0}}',
        encoding="utf-8",
    )
    audit = tmp_path / "audit.jsonl"
    _write_planner_audit(audit)
    with pytest.raises(ValueError, match="Reference seeds do not match"):
        summarize(
            eval_log=log,
            audit_log=audit,
            condition="no_key",
            checkpoint=checkpoint,
            planner_model=planner,
            planner_adapter=adapter,
            base_model_index=base_index,
            training_pair_manifest=pair_manifest,
            seed_start=100000,
            policy_seed_base=120000,
            reference_summary=reference,
        )
