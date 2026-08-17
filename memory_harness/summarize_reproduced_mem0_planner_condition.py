from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from memory_harness.summarize_official_mem0_ablation import (
    paired_success_transitions,
    read_episode_results,
)


RESULT_PREFIX = "RMBENCH_LIMITED_EVAL_RESULT "


def planner_output_accuracy(audit_log: Path) -> dict[str, Any]:
    """Score planner choices against privileged RMBench diagnostic state.

    Planner stage defines the paired SFT target sequence; the color-position
    mapping is an evaluator label used only to score which cover realizes the
    requested color. Neither diagnostic enters the deployable planner input.
    """
    cover_positions = ("left", "middle", "right")
    uncover_colors = ("red", "green", "blue")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        audit_log.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in planner audit {audit_log}:{line_number}"
            ) from exc
        if row.get("event") == "planner_output":
            rows.append(row)
    if not rows:
        raise ValueError(f"No planner_output events in {audit_log}")

    scored: list[dict[str, Any]] = []
    for row in rows:
        stage = int(row["planner_stage"])
        if stage < 0 or stage >= 6:
            raise ValueError(f"Invalid Cover Blocks planner stage: {stage}")
        pointer = int(row["environment_pointer"])
        if stage < 3:
            position = cover_positions[stage]
            expected = f"next_subtask: Cover the {position} block with the {position} cover."
        else:
            positions = row.get("color_positions_red_green_blue")
            if not isinstance(positions, dict):
                raise ValueError("Planner audit lacks color-position diagnostic labels")
            color = uncover_colors[stage - 3]
            position = positions.get(color)
            if position not in cover_positions:
                raise ValueError(f"Invalid position {position!r} for color {color!r}")
            expected = (
                f"next_subtask: Open the {position} cover to uncover the blocks "
                "in the order of red, green, and blue."
            )
        actual = str(row.get("raw_output", "")).strip()
        scored.append(
            {
                "episode_index": int(row["episode_index"]),
                "environment_step": int(row["environment_step"]),
                "environment_pointer": pointer,
                "planner_stage": stage,
                "expected": expected,
                "actual": actual,
                "correct": actual == expected,
            }
        )

    correct = sum(int(row["correct"]) for row in scored)
    by_stage = {}
    for stage in range(6):
        subset = [row for row in scored if row["planner_stage"] == stage]
        stage_correct = sum(int(row["correct"]) for row in subset)
        by_stage[str(stage)] = {
            "correct": stage_correct,
            "total": len(subset),
            "rate": stage_correct / len(subset) if subset else None,
        }
    return {
        "label_source": "privileged_rmbench_diagnostic_only",
        "correct": correct,
        "total": len(scored),
        "rate": correct / len(scored),
        "by_planner_stage": by_stage,
        "errors": [row for row in scored if not row["correct"]],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_result(log_path: Path) -> dict[str, Any]:
    matches = [
        json.loads(line.removeprefix(RESULT_PREFIX))
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one limited-eval result in {log_path}, found {len(matches)}"
        )
    return matches[0]


def summarize(
    *,
    eval_log: Path,
    audit_log: Path,
    condition: str,
    checkpoint: Path,
    planner_model: Path,
    planner_adapter: Path,
    base_model_index: Path,
    training_pair_manifest: Path,
    seed_start: int,
    policy_seed_base: int,
    reference_summary: Path | None,
) -> dict[str, Any]:
    if condition not in {"key", "no_key"}:
        raise ValueError("condition must be key or no_key")
    result = read_result(eval_log)
    required = {
        "num_episodes",
        "successes",
        "success_rate",
        "mean_reward",
        "simulator_seeds",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Missing result fields in {eval_log}: {sorted(missing)}")
    episode_count = int(result["num_episodes"])
    raw_seeds = result["simulator_seeds"]
    if not isinstance(raw_seeds, list) or any(
        not isinstance(seed, int) for seed in raw_seeds
    ):
        raise ValueError(f"Invalid candidate simulator seeds: {raw_seeds!r}")
    seeds = list(raw_seeds)
    if len(seeds) != episode_count:
        raise ValueError(
            f"Candidate recorded {len(seeds)} simulator seeds for {episode_count} episodes"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Candidate repeats simulator seeds: {seeds}")
    if seeds and seeds[0] < seed_start:
        raise ValueError(
            f"First evaluated seed {seeds[0]} precedes seed start {seed_start}"
        )

    episodes = read_episode_results(eval_log.parent)
    if len(episodes) != episode_count:
        raise ValueError(
            f"Candidate has {len(episodes)} official episode records for "
            f"{episode_count} episodes"
        )
    episode_ids = [record["episode_id"] for record in episodes]
    if episode_ids != list(range(episode_count)):
        raise ValueError(f"Candidate has invalid episode ids: {episode_ids}")
    logged_seeds = [record["simulator_seed"] for record in episodes]
    if logged_seeds != seeds:
        raise ValueError(
            f"Official episode seeds do not match evaluator seeds: {logged_seeds} != {seeds}"
        )
    candidate_successful_seeds = [
        record["simulator_seed"] for record in episodes if record["success"]
    ]
    if len(candidate_successful_seeds) != int(result["successes"]):
        raise ValueError(
            "Per-episode candidate successes do not match aggregate: "
            f"{len(candidate_successful_seeds)} != {result['successes']}"
        )
    policy_seeds = [policy_seed_base + index for index in range(episode_count)]
    candidate_planner_accuracy = planner_output_accuracy(audit_log)
    comparison = None
    if reference_summary is not None:
        reference = json.loads(reference_summary.read_text(encoding="utf-8"))
        reference_seeds = [int(value) for value in reference["simulator_seeds"]]
        if reference_seeds != seeds:
            raise ValueError(
                f"Reference seeds do not match candidate seeds: {reference_seeds} != {seeds}"
            )
        reference_policy_seeds = [int(value) for value in reference["policy_seeds"]]
        if reference_policy_seeds != policy_seeds:
            raise ValueError(
                "Reference policy seeds do not match candidate policy seeds: "
                f"{reference_policy_seeds} != {policy_seeds}"
            )
        reference_successful_seeds = {
            int(value) for value in reference["successful_seeds"]
        }
        if not reference_successful_seeds.issubset(set(reference_seeds)):
            raise ValueError("Reference successful seeds are not a subset of simulator seeds")
        if len(reference_successful_seeds) != int(reference["successes"]):
            raise ValueError(
                "Reference successful seed count does not match aggregate successes"
            )
        reference_episodes = [
            {
                "episode_id": index,
                "simulator_seed": seed,
                "success": seed in reference_successful_seeds,
            }
            for index, seed in enumerate(reference_seeds)
        ]
        reference_rate = float(reference["success_rate"])
        reference_planner_accuracy = reference.get("planner_online_exact")
        if not isinstance(reference_planner_accuracy, dict):
            raise ValueError("Reference summary lacks planner_online_exact")
        reference_planner_correct = int(reference_planner_accuracy["correct"])
        reference_planner_total = int(reference_planner_accuracy["total"])
        reference_planner_rate = float(reference_planner_accuracy["rate"])
        if reference_planner_total <= 0:
            raise ValueError("Reference planner accuracy has no scored outputs")
        if not math.isclose(
            reference_planner_correct / reference_planner_total,
            reference_planner_rate,
        ):
            raise ValueError("Reference planner accuracy aggregate is inconsistent")
        comparison = {
            "reference_condition": "key",
            "reference_summary": str(reference_summary.resolve()),
            "reference_successes": int(reference["successes"]),
            "reference_success_rate": reference_rate,
            "candidate_minus_reference_success_rate": (
                float(result["success_rate"]) - reference_rate
            ),
            "reference_planner_output_accuracy": {
                "correct": reference_planner_correct,
                "total": reference_planner_total,
                "rate": reference_planner_rate,
            },
            "candidate_minus_reference_planner_accuracy": (
                candidate_planner_accuracy["rate"] - reference_planner_rate
            ),
            "paired_success_transitions": paired_success_transitions(
                reference_episodes, episodes
            ),
        }
    model_index = planner_model / "model.safetensors.index.json"
    return {
        "schema_version": "memory_harness.reproduced_mem0_planner_condition/v3",
        "protocol": "released_mem0_executor_reproduced_planner_memory_condition",
        "task": "cover_blocks",
        "planner_memory_condition": condition,
        "simulator_seeds": seeds,
        "policy_seeds": policy_seeds,
        "successful_seeds": candidate_successful_seeds,
        "episodes": episodes,
        "result": result,
        "planner_output_accuracy": candidate_planner_accuracy,
        "execution_checkpoint": {
            "path": str(checkpoint.resolve()),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "planner_model": {
            "path": str(planner_model.resolve()),
            "weight_index_sha256": sha256_file(model_index),
            "adapter_path": str(planner_adapter.resolve()),
            "adapter_sha256": sha256_file(planner_adapter),
            "base_model_index_sha256": sha256_file(base_model_index),
            "training_pair_manifest_sha256": sha256_file(training_pair_manifest),
        },
        "comparison": comparison,
        "scope_note": (
            "The executor is the released Mem-0 Cover Blocks checkpoint. Both planner "
            "variants were locally trained on the same 50 demonstrations; upstream did "
            "not release the paper's planner ablation weights. This is a matched local "
            "reproduction, not an exact replay of Table 2."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-log", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--condition", choices=("key", "no_key"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planner-model", type=Path, required=True)
    parser.add_argument("--planner-adapter", type=Path, required=True)
    parser.add_argument("--base-model-index", type=Path, required=True)
    parser.add_argument("--training-pair-manifest", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--policy-seed-base", type=int, required=True)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        eval_log=args.eval_log,
        audit_log=args.audit_log,
        condition=args.condition,
        checkpoint=args.checkpoint,
        planner_model=args.planner_model,
        planner_adapter=args.planner_adapter,
        base_model_index=args.base_model_index,
        training_pair_manifest=args.training_pair_manifest,
        seed_start=args.seed_start,
        policy_seed_base=args.policy_seed_base,
        reference_summary=args.reference_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
