from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from statistics import mean

from memory_harness import __version__
from memory_harness.architecture import ArchitectureSpec
from memory_harness.config import load_program_spec
from memory_harness.config_snapshot import validate_config_snapshot
from memory_harness.runtime_snapshot import validate_runtime_snapshot
from memory_harness.tasks import load_task_spec
from memory_harness.put_back_progress import load_episodes, summarize_put_back_subtasks


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _jsonl(path: pathlib.Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _validate_persistent_outcomes(
    *,
    events: list[dict],
    episodes: list[dict],
    path_names: tuple[str, ...],
) -> dict[str, int]:
    if not episodes:
        raise ValueError("persistent memory requires non-empty episode results")
    successful_episode_ids: set[str] = set()
    commits = 0
    discards = 0
    for expected_index, episode in enumerate(episodes):
        if int(episode.get("episode_index", -1)) != expected_index:
            raise ValueError("persistent deployment episodes must be ordered and contiguous")
        episode_id = f"episode-{expected_index}"
        success = bool(episode.get("success", False))
        total_reward = float(episode.get("total_reward", 0.0))
        final_step_index = int(episode.get("steps", 0)) - 1
        if final_step_index < 0:
            raise ValueError("persistent deployment episode has no executed steps")

        outcomes = [
            event
            for event in events
            if event.get("event") == "EPISODE_OUTCOME"
            and event.get("episode_id") == episode_id
        ]
        if len(outcomes) != 1:
            raise ValueError(
                f"persistent episode {episode_id} must have exactly one EPISODE_OUTCOME"
            )
        outcome = outcomes[0]
        details = outcome.get("details", {})
        if "success" not in details or "total_reward" not in details:
            raise ValueError(f"persistent outcome is incomplete: {episode_id}")
        if (
            bool(details.get("success")) != success
            or float(details.get("total_reward", float("nan"))) != total_reward
            or int(outcome.get("step_index", -1)) != final_step_index
        ):
            raise ValueError(f"persistent outcome disagrees with episode result: {episode_id}")

        for path_name in path_names:
            finalizes = [
                event
                for event in events
                if event.get("event") == "STORE_FINALIZE"
                and event.get("episode_id") == episode_id
                and event.get("path_name") == path_name
            ]
            if len(finalizes) != 1:
                raise ValueError(
                    f"persistent path {path_name!r} must finalize once in {episode_id}"
                )
            expected_action = "commit" if success else "discard"
            if finalizes[0].get("details", {}).get("action") != expected_action:
                raise ValueError(
                    f"persistent path {path_name!r} used the wrong outcome action in {episode_id}"
                )

            retrievals = [
                event
                for event in events
                if event.get("event") == "RETRIEVE"
                and event.get("episode_id") == episode_id
                and event.get("path_name") == path_name
            ]
            retrieved_source_episodes = {
                str(item_id).split(":", 1)[0]
                for event in retrievals
                for item_id in event.get("item_ids", [])
            }
            invalid_sources = retrieved_source_episodes - successful_episode_ids
            if invalid_sources:
                raise ValueError(
                    "persistent retrieval consumed unverified or non-prior episodes: "
                    f"episode={episode_id}, sources={sorted(invalid_sources)}"
                )

        if success:
            successful_episode_ids.add(episode_id)
            commits += 1
        else:
            discards += 1
    return {
        "successful_episode_commits": commits,
        "failed_episode_discards": discards,
    }


def validate(
    run_dir: pathlib.Path,
    architecture_config: pathlib.Path,
    task_config: pathlib.Path,
    audit_log: pathlib.Path | None = None,
    *,
    diagnostic_oracle_phase: bool = False,
) -> dict[str, object]:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    action_stats_path = run_dir / "action_stats.jsonl"
    audit_path = audit_log or run_dir / "memory_architecture_audit.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    architecture_spec = ArchitectureSpec.load(architecture_config)
    task_spec = load_task_spec(task_config)
    executor_spec = load_program_spec(architecture_spec.executor_program)
    runtime_snapshot = validate_runtime_snapshot(run_dir / "runtime")
    config_snapshot = validate_config_snapshot(run_dir / "experiment_configs")
    planner_enabled = architecture_spec.planner == "mem0"
    key_enabled = architecture_spec.planner_memory == "key"
    persistent_paths = tuple(
        path.name
        for path in executor_spec.paths
        if path.store.type == "verified_success_ring"
    )
    subtask_evaluation = None
    if task_spec.task_name == "put_back_block":
        episodes_path = run_dir / "episodes.jsonl"
        subtask_summary_path = run_dir / "subtask_summary.json"
        if not subtask_summary_path.is_file():
            raise ValueError("Put Back run is missing subtask_summary.json")
        expected_subtasks = summarize_put_back_subtasks(load_episodes(episodes_path))
        recorded_subtasks = json.loads(
            subtask_summary_path.read_text(encoding="utf-8")
        )
        if recorded_subtasks != expected_subtasks:
            raise ValueError("Put Back subtask summary disagrees with episode records")
        if int(recorded_subtasks["num_episodes"]) != int(summary["num_episodes"]):
            raise ValueError("Put Back subtask summary episode count disagrees with run summary")
        subtask_evaluation = {
            "summary": str(subtask_summary_path.resolve()),
            "summary_sha256": _sha256(subtask_summary_path),
            "subtask_metrics": recorded_subtasks["subtask_metrics"],
            "stopped_at_counts": recorded_subtasks["stopped_at_counts"],
        }

    expected = {
        "policy_router_manifest": None,
        "prompt_schedule": None,
        "phase_aware_subtask_prompt": diagnostic_oracle_phase,
        "task_state_trace_frequency": 10,
        "paired_layout_protocol": task_spec.paired_layout_protocol,
        "execute_action_chunk_steps": 10,
        "policy_adapt_to_pi": False,
        "memory_enabled": bool(executor_spec.paths) or planner_enabled,
        "task_name": task_spec.task_name,
        "task_config": task_spec.task_config,
        "max_steps": task_spec.max_steps,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    configured_architecture = config.get("memory_architecture_config")
    if (
        configured_architecture is None
        or pathlib.Path(configured_architecture).resolve()
        != architecture_config.resolve()
    ):
        mismatches["memory_architecture_config"] = {
            "expected": str(architecture_config.resolve()),
            "actual": configured_architecture,
        }
    if config.get("memory_architecture_name") != architecture_spec.name:
        mismatches["memory_architecture_name"] = {
            "expected": architecture_spec.name,
            "actual": config.get("memory_architecture_name"),
        }
    if config.get("memory_planner_enabled") != planner_enabled:
        mismatches["memory_planner_enabled"] = {
            "expected": planner_enabled,
            "actual": config.get("memory_planner_enabled"),
        }
    if config.get("memory_key_enabled") != key_enabled:
        mismatches["memory_key_enabled"] = {
            "expected": key_enabled,
            "actual": config.get("memory_key_enabled"),
        }
    if config.get("memory_planner_model") != architecture_spec.planner_model:
        mismatches["memory_planner_model"] = {
            "expected": architecture_spec.planner_model,
            "actual": config.get("memory_planner_model"),
        }
    if planner_enabled and not diagnostic_oracle_phase:
        mismatches["diagnostic_oracle_phase"] = {
            "expected": True,
            "actual": diagnostic_oracle_phase,
        }
    if mismatches:
        raise ValueError(
            f"fixed-memory run changed a protected experimental variable: {mismatches}"
        )

    events = _jsonl(audit_path)
    event_names = {row.get("event") for row in events}
    required_events = {"RESET", "USE"}
    if executor_spec.paths:
        required_events.update({"RETRIEVE", "WRITE", "WRITE_DECISION"})
    if planner_enabled:
        required_events.add("PLAN")
    if persistent_paths:
        required_events.update({"EPISODE_OUTCOME", "STORE_FINALIZE"})
    if not required_events.issubset(event_names):
        raise ValueError(
            f"memory audit is incomplete: required={sorted(required_events)}, actual={sorted(event_names)}"
        )

    executor_use_events = [
        row
        for row in events
        if row.get("event") == "USE" and "token_count" in row.get("details", {})
    ]
    use_counts = [int(row["details"]["token_count"]) for row in executor_use_events]
    action_rows = _jsonl(action_stats_path) if action_stats_path.is_file() else []
    if executor_spec.paths and len(executor_use_events) != len(action_rows):
        raise ValueError(
            "executor memory must observe every environment step: "
            f"use_events={len(executor_use_events)}, action_steps={len(action_rows)}"
        )
    persistence = None
    if persistent_paths:
        episodes_path = run_dir / "episodes.jsonl"
        if not episodes_path.is_file():
            raise ValueError("persistent memory run is missing episodes.jsonl")
        episode_rows = _jsonl(episodes_path)
        if len(episode_rows) != int(summary["num_episodes"]):
            raise ValueError("persistent episode rows do not match summary count")
        persistence = _validate_persistent_outcomes(
            events=events,
            episodes=episode_rows,
            path_names=persistent_paths,
        )
    inference_ms = [
        float(row["policy_timing"]["infer_ms"])
        for row in action_rows
        if isinstance(row.get("policy_timing"), dict)
        and "infer_ms" in row["policy_timing"]
        and row.get("step_index") == row.get("policy_infer_step_index")
    ]
    observe_ms = [
        float(row["memory_observe_timing"]["observe_ms"])
        for row in action_rows
        if isinstance(row.get("memory_observe_timing"), dict)
        and "observe_ms" in row["memory_observe_timing"]
    ]
    total_stored_counts = [
        int(row["details"][key])
        for row in events
        for key in (
            "stored_item_count_before_write",
            "total_stored_item_count",
        )
        if key in row.get("details", {})
    ]
    stored_by_path: dict[str, list[int]] = {}
    for row in events:
        path_name = row.get("path_name")
        count = row.get("details", {}).get("path_stored_item_count")
        if isinstance(path_name, str) and count is not None:
            stored_by_path.setdefault(path_name, []).append(int(count))
    manifest = {
        "schema_version": 1,
        "harness_version": __version__,
        "runtime_source_sha256": runtime_snapshot["source_sha256"],
        "runtime_manifest_sha256": _sha256(
            run_dir / "runtime" / "runtime_manifest.json"
        ),
        "config_source_sha256": config_snapshot["source_sha256"],
        "config_manifest_sha256": _sha256(
            run_dir / "experiment_configs" / "config_manifest.json"
        ),
        "status": (
            "validated_oracle_diagnostic" if diagnostic_oracle_phase else "validated"
        ),
        "architecture": architecture_spec.name,
        "task_name": task_spec.task_name,
        "task_memory_complexity": task_spec.tmc,
        "task_config": str(task_config.resolve()),
        "task_config_sha256": _sha256(task_config),
        "planner_model": architecture_spec.planner_model,
        "active_modules": [
            *[path.name for path in executor_spec.paths],
            *(["key"] if key_enabled else []),
        ],
        "deployable": executor_spec.deployable and not diagnostic_oracle_phase,
        "diagnostic_oracle_phase": diagnostic_oracle_phase,
        "run_config_sha256": _sha256(config_path),
        "summary_sha256": _sha256(summary_path),
        "architecture_config": str(architecture_config.resolve()),
        "architecture_config_sha256": _sha256(architecture_config),
        "executor_program_config": str(architecture_spec.executor_program),
        "executor_program_config_sha256": _sha256(architecture_spec.executor_program),
        "memory_audit_sha256": _sha256(audit_path),
        "checkpoint_dir": config["checkpoint_dir"],
        "candidate_suite_manifest_sha256": (
            _sha256(run_dir / "candidate_suite_manifest.json")
            if (run_dir / "candidate_suite_manifest.json").is_file()
            else None
        ),
        "simulator_seed_start": config["seed"],
        "policy_seed_base": config["policy_seed_base"],
        "num_episodes": summary["num_episodes"],
        "subtask_evaluation": subtask_evaluation,
        "resource_usage": {
            "write_count": sum(row.get("event") == "WRITE" for row in events),
            "write_decision_count": sum(
                row.get("event") == "WRITE_DECISION" for row in events
            ),
            "write_skip_count": sum(
                row.get("event") == "WRITE_DECISION"
                and not row.get("details", {}).get("write", False)
                for row in events
            ),
            "retrieve_count": sum(row.get("event") == "RETRIEVE" for row in events),
            "total_used_memory_tokens": sum(use_counts),
            "max_used_memory_tokens_per_inference": max(use_counts, default=0),
            "max_stored_memory_items": max(total_stored_counts, default=0),
            "max_stored_items_by_path": {
                path_name: max(counts)
                for path_name, counts in sorted(stored_by_path.items())
            },
            "executor_observation_updates": len(executor_use_events),
            "mean_policy_infer_ms": mean(inference_ms) if inference_ms else None,
            "mean_memory_observe_ms": mean(observe_ms) if observe_ms else None,
            "total_memory_observe_ms": sum(observe_ms),
            "planner_calls": sum(row.get("event") == "PLAN" for row in events),
            "persistent_outcomes": persistence,
        },
    }
    (run_dir / "emac_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and cost a fixed-memory RMBench run."
    )
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--architecture-config", type=pathlib.Path, required=True)
    parser.add_argument("--task-config", type=pathlib.Path, required=True)
    parser.add_argument("--audit-log", type=pathlib.Path)
    parser.add_argument("--diagnostic-oracle-phase", action="store_true")
    args = parser.parse_args()
    manifest = validate(
        args.run_dir.resolve(),
        args.architecture_config.resolve(),
        args.task_config.resolve(),
        None if args.audit_log is None else args.audit_log.resolve(),
        diagnostic_oracle_phase=args.diagnostic_oracle_phase,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
