from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


RESULT_PREFIX = "RMBENCH_LIMITED_EVAL_RESULT "
CONDITIONS = ("full", "without_anchor", "without_sliding")
EPISODE_RESULT_PATTERN = re.compile(
    r"^episode_id=(?P<episode_id>\d+), seed=(?P<seed>\d+), "
    r"result=(?P<result>Success|Fail)$"
)


def read_condition_result(path: Path) -> dict[str, Any]:
    matches = [
        json.loads(line.removeprefix(RESULT_PREFIX))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one limited-eval result in {path}, found {len(matches)}")
    result = matches[0]
    required = {
        "num_episodes",
        "successes",
        "success_rate",
        "mean_reward",
        "simulator_seeds",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Missing result fields in {path}: {sorted(missing)}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_episode_results(condition_dir: Path) -> list[dict[str, Any]]:
    logs = sorted((condition_dir / "official_results").glob("**/eval_log.txt"))
    if len(logs) != 1:
        raise ValueError(
            f"Expected exactly one official eval_log.txt under {condition_dir}, "
            f"found {len(logs)}"
        )
    records: list[dict[str, Any]] = []
    for line in logs[0].read_text(encoding="utf-8").splitlines():
        match = EPISODE_RESULT_PATTERN.fullmatch(line)
        if match is None:
            continue
        records.append(
            {
                "episode_id": int(match.group("episode_id")),
                "simulator_seed": int(match.group("seed")),
                "success": match.group("result") == "Success",
            }
        )
    return records


def paired_success_transitions(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, int]:
    counts = {
        "both_success": 0,
        "full_only_success": 0,
        "ablation_only_success": 0,
        "both_fail": 0,
    }
    for full, ablation in zip(reference, candidate, strict=True):
        pair = (bool(full["success"]), bool(ablation["success"]))
        key = {
            (True, True): "both_success",
            (True, False): "full_only_success",
            (False, True): "ablation_only_success",
            (False, False): "both_fail",
        }[pair]
        counts[key] += 1
    return counts


def summarize(
    run_dir: Path,
    *,
    checkpoint: Path,
    task: str,
    seed_start: int,
) -> dict[str, Any]:
    results = {
        condition: read_condition_result(run_dir / condition / "eval.log")
        for condition in CONDITIONS
    }
    episode_counts = {int(result["num_episodes"]) for result in results.values()}
    if len(episode_counts) != 1:
        raise ValueError(f"Conditions use different episode counts: {sorted(episode_counts)}")
    num_episodes = episode_counts.pop()

    seed_sequences: dict[str, tuple[int, ...]] = {}
    for condition, result in results.items():
        raw_seeds = result["simulator_seeds"]
        if not isinstance(raw_seeds, list) or any(
            not isinstance(seed, int) for seed in raw_seeds
        ):
            raise ValueError(f"Invalid simulator seeds for {condition}: {raw_seeds!r}")
        seeds = tuple(raw_seeds)
        if len(seeds) != num_episodes:
            raise ValueError(
                f"Condition {condition} recorded {len(seeds)} simulator seeds for "
                f"{num_episodes} episodes"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Condition {condition} repeats simulator seeds: {seeds}")
        seed_sequences[condition] = seeds
    reference_seeds = seed_sequences["full"]
    mismatched = {
        condition: seeds
        for condition, seeds in seed_sequences.items()
        if seeds != reference_seeds
    }
    if mismatched:
        raise ValueError(
            "Conditions did not evaluate identical simulator seeds: "
            f"full={reference_seeds}, mismatched={mismatched}"
        )
    if reference_seeds and reference_seeds[0] < seed_start:
        raise ValueError(
            f"First evaluated seed {reference_seeds[0]} precedes seed start {seed_start}"
        )

    episodes = {
        condition: read_episode_results(run_dir / condition)
        for condition in CONDITIONS
    }
    for condition, records in episodes.items():
        if len(records) != num_episodes:
            raise ValueError(
                f"Condition {condition} has {len(records)} per-episode results for "
                f"{num_episodes} episodes"
            )
        episode_ids = [record["episode_id"] for record in records]
        if episode_ids != list(range(num_episodes)):
            raise ValueError(
                f"Condition {condition} has invalid episode ids: {episode_ids}"
            )
        logged_seeds = tuple(record["simulator_seed"] for record in records)
        if logged_seeds != reference_seeds:
            raise ValueError(
                f"Condition {condition} official episode seeds do not match evaluator "
                f"seeds: {logged_seeds} != {reference_seeds}"
            )
        logged_successes = sum(record["success"] for record in records)
        if logged_successes != int(results[condition]["successes"]):
            raise ValueError(
                f"Condition {condition} per-episode successes {logged_successes} do not "
                f"match aggregate {results[condition]['successes']}"
            )

    paired_transitions = {
        condition: paired_success_transitions(episodes["full"], episodes[condition])
        for condition in CONDITIONS[1:]
    }

    full_rate = float(results["full"]["success_rate"])
    return {
        "schema_version": "memory_harness.mem0_released_checkpoint_intervention/v2",
        "protocol": "released_mem0_m1mix_shared_checkpoint_inference_intervention",
        "task": task,
        "simulator_seed_start": seed_start,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "num_episodes_per_condition": num_episodes,
        "simulator_seeds": list(reference_seeds),
        "policy_rng_protocol": (
            "RMBench setup_demo resets NumPy and Torch RNG from each simulator seed"
        ),
        "conditions": results,
        "episodes": episodes,
        "paired_success_transitions_vs_full": paired_transitions,
        "success_rate_delta_vs_full": {
            condition: float(results[condition]["success_rate"]) - full_rate
            for condition in CONDITIONS[1:]
        },
        "scope_note": (
            "without_anchor and without_sliding mask inference outputs on the released full "
            "checkpoint; they are not an exact replay of the paper's ablation protocol"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    args = parser.parse_args()
    report = summarize(
        args.run_dir,
        checkpoint=args.checkpoint,
        task=args.task,
        seed_start=args.seed_start,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
