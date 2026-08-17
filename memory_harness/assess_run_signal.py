from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Sequence
from statistics import mean
from typing import Any

from memory_harness.put_back_progress import summarize_put_back_subtasks


SCHEMA_VERSION = "memory_harness.executor_run_signal/v2"


def _json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError("run must contain non-empty episode records")
    return values


def assess_run_signal(run_dir: pathlib.Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary = _json(run_dir / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError("run summary is not completed")
    manifest_path = run_dir / "emac_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("run has not passed E-MAC validation")
    manifest = _json(manifest_path)
    evidence_scope = str(manifest.get("evidence_scope", "deployment_baseline"))
    if evidence_scope == "executor_skill_diagnostic_only":
        if manifest.get("deployable") is not False:
            raise ValueError("executor skill diagnostic must be non-deployable")
        diagnostic_only = True
    else:
        diagnostic_only = False
    episodes = _jsonl(run_dir / "episodes.jsonl")
    if int(summary.get("num_episodes", -1)) != len(episodes):
        raise ValueError("summary episode count does not match episode records")
    successes = sum(int(bool(row.get("success", False))) for row in episodes)
    max_rewards: list[float] = []
    total_rewards: list[float] = []
    task_progress_scores: list[float] = []
    for row in episodes:
        final_info = row.get("final_info")
        if not isinstance(final_info, dict):
            raise ValueError("episode record has no final_info")
        max_rewards.append(float(final_info.get("max_reward", 0.0)))
        total_rewards.append(float(row.get("total_reward", 0.0)))
        task_progress = row.get("task_progress")
        if task_progress is not None:
            if not isinstance(task_progress, dict) or task_progress.get("task") != "put_back_block":
                raise ValueError("episode has invalid task_progress")
            task_progress_scores.append(float(task_progress["max_progress_score"]))
    max_task_progress = max(task_progress_scores) if task_progress_scores else None
    subtask_evaluation = (
        summarize_put_back_subtasks(episodes) if task_progress_scores else None
    )
    signal = (
        successes > 0
        or max(max_rewards) > 0.0
        or (max_task_progress is not None and max_task_progress > 0.0)
    )
    if diagnostic_only:
        next_action = (
            "train_budget_matched_memory_executor"
            if signal
            else "increase_native_executor_training_budget"
        )
    else:
        next_action = (
            "budget_match_controls_before_comparison"
            if signal
            else "full_memory_still_at_floor"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run": str(run_dir),
        "architecture": manifest.get("architecture", manifest.get("condition")),
        "deployable": manifest.get("deployable", True),
        "evidence_scope": evidence_scope,
        "num_episodes": len(episodes),
        "num_successes": successes,
        "max_observed_reward": max(max_rewards),
        "mean_total_reward": mean(total_rewards),
        "max_observed_task_progress": max_task_progress,
        "subtask_evaluation": subtask_evaluation,
        "observable_executor_signal": signal,
        "decision": {
            "next_action": next_action,
            "controller_training_allowed": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess one validated rollout for success or staged reward."
    )
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = assess_run_signal(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
