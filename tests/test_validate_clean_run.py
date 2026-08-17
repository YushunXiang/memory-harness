from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_harness.validate_clean_run import validate


ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = ROOT / "configs" / "tasks" / "cover_blocks.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_validate_clean_run_writes_auditable_manifest(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {
            "policy_router_manifest": None,
            "prompt_schedule": None,
            "prompt_protocol": "main",
            "phase_aware_subtask_prompt": False,
            "task_state_trace_frequency": 10,
            "paired_layout_protocol": True,
            "execute_action_chunk_steps": 10,
            "policy_adapt_to_pi": False,
            "memory_enabled": False,
            "seed": 100000,
            "policy_seed_base": 120000,
            "task_name": "cover_blocks",
            "task_config": "demo_clean",
            "max_steps": 1500,
        },
    )
    _write_json(tmp_path / "summary.json", {"num_episodes": 3})
    manifest = validate(tmp_path, ROOT / "configs" / "fixed_none.json", TASK_CONFIG)
    assert manifest["status"] == "validated"
    assert (tmp_path / "emac_manifest.json").is_file()


def test_validate_clean_run_rejects_oracle_prompt(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {
            "policy_router_manifest": None,
            "prompt_schedule": None,
            "prompt_protocol": "main",
            "phase_aware_subtask_prompt": True,
            "task_state_trace_frequency": 10,
            "paired_layout_protocol": True,
            "execute_action_chunk_steps": 10,
            "policy_adapt_to_pi": False,
            "memory_enabled": False,
        },
    )
    _write_json(tmp_path / "summary.json", {"num_episodes": 1})
    with pytest.raises(ValueError, match="not a clean"):
        validate(tmp_path, ROOT / "configs" / "fixed_none.json", TASK_CONFIG)


def test_validate_oracle_subtask_executor_diagnostic(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {
            "policy_router_manifest": None,
            "prompt_schedule": None,
            "prompt_protocol": "diagnostic_spatial",
            "phase_aware_subtask_prompt": True,
            "task_state_trace_frequency": 10,
            "paired_layout_protocol": True,
            "execute_action_chunk_steps": 10,
            "policy_adapt_to_pi": False,
            "memory_enabled": False,
            "seed": 100000,
            "policy_seed_base": 120000,
            "task_name": "cover_blocks",
            "task_config": "demo_clean",
            "max_steps": 1500,
        },
    )
    _write_json(tmp_path / "summary.json", {"num_episodes": 3})

    manifest = validate(
        tmp_path,
        ROOT / "configs" / "fixed_none.json",
        TASK_CONFIG,
        oracle_subtask_diagnostic=True,
    )

    assert manifest["condition"] == "oracle_subtask_pi05_none"
    assert manifest["deployable"] is False
    assert manifest["evidence_scope"] == "executor_skill_diagnostic_only"
    assert "oracle_phase_prompt" not in manifest["disabled"]


def test_rejects_oracle_subtask_diagnostic_for_m1(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid for M\\(n\\)"):
        validate(
            tmp_path,
            ROOT / "configs" / "fixed_none.json",
            ROOT / "configs" / "tasks" / "put_back_block.json",
            oracle_subtask_diagnostic=True,
        )
