from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_harness.validate_fixed_run import validate
from memory_harness.config_snapshot import create_config_snapshot
from memory_harness.runtime_snapshot import create_runtime_snapshot


ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = ROOT / "configs" / "tasks" / "cover_blocks.json"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _snapshot(run_dir: Path) -> None:
    create_runtime_snapshot(ROOT / "memory_harness", run_dir / "runtime")
    create_config_snapshot(ROOT / "configs", run_dir / "experiment_configs")


def _base_config(
    architecture: Path, *, enabled: bool, planner: bool = False, key: bool = False
) -> dict:
    name = json.loads(architecture.read_text(encoding="utf-8"))["name"]
    planner_model = json.loads(architecture.read_text(encoding="utf-8"))[
        "planner_model"
    ]
    return {
        "checkpoint_dir": "/checkpoint",
        "seed": 100000,
        "policy_seed_base": 120000,
        "policy_router_manifest": None,
        "prompt_schedule": None,
        "phase_aware_subtask_prompt": False,
        "task_state_trace_frequency": 10,
        "paired_layout_protocol": True,
        "execute_action_chunk_steps": 10,
        "policy_adapt_to_pi": False,
        "memory_enabled": enabled,
        "memory_architecture_config": str(architecture.resolve()),
        "memory_architecture_name": name,
        "memory_planner_enabled": planner,
        "memory_key_enabled": key,
        "memory_planner_model": planner_model,
        "task_name": "cover_blocks",
        "task_config": "demo_clean",
        "max_steps": 1500,
    }


def test_fixed_run_manifest_records_budget(tmp_path: Path) -> None:
    architecture = ROOT / "configs" / "architectures" / "fixed_anchor.json"
    _write(tmp_path / "config.json", _base_config(architecture, enabled=True))
    _write(tmp_path / "summary.json", {"num_episodes": 1})
    events = [
        {"event": "RESET", "details": {}},
        {"event": "RETRIEVE", "details": {}},
        {
            "event": "USE",
            "details": {"token_count": 4, "stored_item_count_before_write": 2},
        },
        {"event": "WRITE_DECISION", "details": {"write": True}},
        {
            "event": "WRITE",
            "path_name": "anchor",
            "details": {
                "path_stored_item_count": 1,
                "total_stored_item_count": 3,
            },
        },
    ]
    (tmp_path / "memory_architecture_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    _write(
        tmp_path / "action_stats.jsonl",
        {
            "step_index": 0,
            "policy_infer_step_index": 0,
            "policy_timing": {"infer_ms": 12.5},
        },
    )
    _snapshot(tmp_path)

    manifest = validate(tmp_path, architecture, TASK_CONFIG)
    assert manifest["resource_usage"]["total_used_memory_tokens"] == 4
    assert manifest["resource_usage"]["executor_observation_updates"] == 1
    assert manifest["resource_usage"]["write_decision_count"] == 1
    assert manifest["resource_usage"]["write_skip_count"] == 0
    assert manifest["resource_usage"]["max_stored_memory_items"] == 3
    assert manifest["resource_usage"]["max_stored_items_by_path"] == {"anchor": 1}
    assert manifest["resource_usage"]["planner_calls"] == 0


def test_fixed_run_rejects_checkpoint_router(tmp_path: Path) -> None:
    architecture = ROOT / "configs" / "architectures" / "fixed_none.json"
    config = _base_config(architecture, enabled=False)
    config["policy_router_manifest"] = "/router.json"
    _write(tmp_path / "config.json", config)
    _write(tmp_path / "summary.json", {"num_episodes": 1})
    (tmp_path / "memory_architecture_audit.jsonl").write_text(
        json.dumps({"event": "RESET", "details": {}})
        + "\n"
        + json.dumps({"event": "USE", "details": {"token_count": 0}})
        + "\n",
        encoding="utf-8",
    )
    _snapshot(tmp_path)
    with pytest.raises(ValueError, match="protected experimental variable"):
        validate(tmp_path, architecture, TASK_CONFIG)


def test_oracle_phase_run_is_explicitly_non_deployable(tmp_path: Path) -> None:
    architecture = ROOT / "configs" / "architectures" / "fixed_none.json"
    config = _base_config(architecture, enabled=False)
    config["phase_aware_subtask_prompt"] = True
    _write(tmp_path / "config.json", config)
    _write(tmp_path / "summary.json", {"num_episodes": 1})
    (tmp_path / "memory_architecture_audit.jsonl").write_text(
        json.dumps({"event": "RESET", "details": {}})
        + "\n"
        + json.dumps({"event": "USE", "details": {"token_count": 0}})
        + "\n",
        encoding="utf-8",
    )
    _snapshot(tmp_path)

    manifest = validate(
        tmp_path, architecture, TASK_CONFIG, diagnostic_oracle_phase=True
    )

    assert manifest["status"] == "validated_oracle_diagnostic"
    assert manifest["deployable"] is False


def test_key_run_counts_planner_calls_and_is_diagnostic(tmp_path: Path) -> None:
    architecture = ROOT / "configs" / "architectures" / "fixed_key.json"
    config = _base_config(architecture, enabled=True, planner=True, key=True)
    config["phase_aware_subtask_prompt"] = True
    _write(tmp_path / "config.json", config)
    _write(tmp_path / "summary.json", {"num_episodes": 1})
    events = [
        {"event": "RESET", "details": {}},
        {"event": "USE", "details": {"token_count": 0}},
        {"event": "PLAN", "details": {}},
    ]
    (tmp_path / "memory_architecture_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    _snapshot(tmp_path)

    manifest = validate(
        tmp_path, architecture, TASK_CONFIG, diagnostic_oracle_phase=True
    )

    assert manifest["active_modules"] == ["key"]
    assert manifest["resource_usage"]["planner_calls"] == 1
    assert manifest["deployable"] is False


def _persistent_run(tmp_path: Path) -> tuple[Path, list[dict]]:
    architecture = (
        ROOT
        / "configs"
        / "architectures"
        / "fixed_verified_success_latent.json"
    )
    _write(tmp_path / "config.json", _base_config(architecture, enabled=True))
    _write(tmp_path / "summary.json", {"num_episodes": 2})
    episodes = [
        {
            "episode_index": 0,
            "success": True,
            "steps": 1,
            "total_reward": 1.0,
        },
        {
            "episode_index": 1,
            "success": False,
            "steps": 1,
            "total_reward": 0.0,
        },
    ]
    (tmp_path / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    _write(tmp_path / "action_stats.jsonl", {"step_index": 0})
    with (tmp_path / "action_stats.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"step_index": 0}) + "\n")
    events: list[dict] = []
    for index, (success, reward) in enumerate(((True, 1.0), (False, 0.0))):
        episode_id = f"episode-{index}"
        events.extend(
            [
                {"event": "RESET", "episode_id": episode_id, "details": {}},
                {
                    "event": "RETRIEVE",
                    "episode_id": episode_id,
                    "path_name": "success",
                    "item_ids": [] if index == 0 else ["episode-0:success:0"],
                    "details": {},
                },
                {
                    "event": "USE",
                    "episode_id": episode_id,
                    "details": {"token_count": index},
                },
                {
                    "event": "WRITE_DECISION",
                    "episode_id": episode_id,
                    "details": {"write": True},
                },
                {
                    "event": "WRITE",
                    "episode_id": episode_id,
                    "path_name": "success",
                    "details": {
                        "path_stored_item_count": index,
                        "total_stored_item_count": index,
                    },
                },
                {
                    "event": "EPISODE_OUTCOME",
                    "episode_id": episode_id,
                    "step_index": 0,
                    "details": {"success": success, "total_reward": reward},
                },
                {
                    "event": "STORE_FINALIZE",
                    "episode_id": episode_id,
                    "path_name": "success",
                    "details": {"action": "commit" if success else "discard"},
                },
            ]
        )
    (tmp_path / "memory_architecture_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    _snapshot(tmp_path)
    return architecture, events


def test_persistent_run_validates_ordered_success_commits(tmp_path: Path) -> None:
    architecture, _ = _persistent_run(tmp_path)

    manifest = validate(tmp_path, architecture, TASK_CONFIG)

    assert manifest["resource_usage"]["persistent_outcomes"] == {
        "successful_episode_commits": 1,
        "failed_episode_discards": 1,
    }


def test_persistent_run_rejects_missing_outcome(tmp_path: Path) -> None:
    architecture, events = _persistent_run(tmp_path)
    events = [
        event
        for event in events
        if not (
            event["event"] == "EPISODE_OUTCOME"
            and event.get("episode_id") == "episode-1"
        )
    ]
    (tmp_path / "memory_architecture_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exactly one EPISODE_OUTCOME"):
        validate(tmp_path, architecture, TASK_CONFIG)


def test_persistent_run_rejects_unverified_retrieval(tmp_path: Path) -> None:
    architecture, events = _persistent_run(tmp_path)
    retrieval = next(
        event
        for event in events
        if event["event"] == "RETRIEVE" and event["episode_id"] == "episode-1"
    )
    retrieval["item_ids"] = ["episode-1:success:0"]
    (tmp_path / "memory_architecture_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unverified or non-prior"):
        validate(tmp_path, architecture, TASK_CONFIG)
