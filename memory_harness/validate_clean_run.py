from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from memory_harness import __version__
from memory_harness.put_back_progress import load_episodes, summarize_put_back_subtasks
from memory_harness.tasks import load_task_spec


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def validate(
    run_dir: pathlib.Path,
    program_config: pathlib.Path,
    task_config: pathlib.Path,
    *,
    oracle_subtask_diagnostic: bool = False,
) -> dict[str, object]:
    task_spec = load_task_spec(task_config)
    if oracle_subtask_diagnostic and task_spec.tmc != "M(n)":
        raise ValueError("oracle subtask diagnostic is only valid for M(n) tasks")
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "policy_router_manifest": None,
        "prompt_schedule": None,
        "prompt_protocol": (
            "diagnostic_spatial" if oracle_subtask_diagnostic else "main"
        ),
        "phase_aware_subtask_prompt": oracle_subtask_diagnostic,
        "task_state_trace_frequency": 10,
        "paired_layout_protocol": task_spec.paired_layout_protocol,
        "execute_action_chunk_steps": 10,
        "policy_adapt_to_pi": False,
        "memory_enabled": False,
        "task_name": task_spec.task_name,
        "task_config": task_spec.task_config,
        "max_steps": task_spec.max_steps,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"run is not a clean no-memory baseline: {mismatches}")

    program = json.loads(program_config.read_text(encoding="utf-8"))
    if program.get("name") != "none" or program.get("paths") != []:
        raise ValueError("clean baseline requires the fixed none program")
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
    manifest = {
        "schema_version": 1,
        "harness_version": __version__,
        "status": "validated",
        "condition": (
            "oracle_subtask_pi05_none"
            if oracle_subtask_diagnostic
            else "clean_pi05_none"
        ),
        "deployable": not oracle_subtask_diagnostic,
        "evidence_scope": (
            "executor_skill_diagnostic_only"
            if oracle_subtask_diagnostic
            else "deployment_baseline"
        ),
        "task_name": task_spec.task_name,
        "task_memory_complexity": task_spec.tmc,
        "task_config": str(task_config.resolve()),
        "task_config_sha256": _sha256(task_config),
        "run_config_sha256": _sha256(config_path),
        "summary_sha256": _sha256(summary_path),
        "program_config": str(program_config.resolve()),
        "program_config_sha256": _sha256(program_config),
        "simulator_seed_start": config["seed"],
        "policy_seed_base": config["policy_seed_base"],
        "num_episodes": summary["num_episodes"],
        "subtask_evaluation": subtask_evaluation,
        "disabled": [
            "policy_router",
            "episodic_memory",
            "working_memory",
            "memory_tokens",
            "prompt_schedule",
            "task_state_policy_input",
        ]
        + ([] if oracle_subtask_diagnostic else ["oracle_phase_prompt"]),
    }
    (run_dir / "emac_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject contaminated E-MAC no-memory runs."
    )
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--program-config", type=pathlib.Path, required=True)
    parser.add_argument("--task-config", type=pathlib.Path, required=True)
    parser.add_argument("--oracle-subtask-diagnostic", action="store_true")
    args = parser.parse_args()
    manifest = validate(
        args.run_dir.resolve(),
        args.program_config.resolve(),
        args.task_config.resolve(),
        oracle_subtask_diagnostic=args.oracle_subtask_diagnostic,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
