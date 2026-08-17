from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from memory_harness.compare_fixed_runs import PROTECTED_CONFIG_KEYS
from memory_harness.compare_fixed_runs import _json
from memory_harness.compare_fixed_runs import _jsonl
from memory_harness.compare_fixed_runs import _metrics


ALLOWED_CONFIG_DIFFERENCES = frozenset({"checkpoint_dir", "config", "policy_config"})


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_chain(checkpoint_dir: pathlib.Path) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    seen: set[pathlib.Path] = set()
    current = checkpoint_dir.resolve()
    while (current / "memory_training_manifest.json").is_file():
        if current in seen:
            raise ValueError(f"training provenance cycle at {current}")
        seen.add(current)
        manifest = _json(current / "memory_training_manifest.json")
        if manifest.get("schema_version") != "memory_harness.training/v1":
            raise ValueError(f"unsupported training manifest at {current}")
        updates = int(manifest["optimizer_updates"])
        effective_batch = int(manifest["effective_batch"])
        if updates <= 0 or effective_batch <= 0:
            raise ValueError(f"invalid training budget at {current}")
        stages.append(
            {
                "checkpoint": str(current),
                "program": str(manifest["program"]),
                "optimizer_updates": updates,
                "effective_batch": effective_batch,
                "optimizer_examples": updates * effective_batch,
                "task_config_sha256": str(manifest["task_config_sha256"]),
            }
        )
        initial_params = pathlib.Path(str(manifest["initial_weight_params"])).resolve()
        parent_checkpoint = (
            initial_params.parent if initial_params.name == "params" else initial_params
        )
        parent_manifest = parent_checkpoint / "memory_training_manifest.json"
        if not parent_manifest.is_file():
            terminal_initial_params = initial_params
            break
        if manifest.get("parent_checkpoint") != str(parent_checkpoint):
            raise ValueError(
                f"training stage does not bind its parent checkpoint at {current}"
            )
        if manifest.get("parent_training_manifest_sha256") != _sha256(parent_manifest):
            raise ValueError(
                f"training stage parent manifest hash mismatch at {current}"
            )
        current = parent_checkpoint
    else:
        raise ValueError(
            f"checkpoint has no memory training manifest: {checkpoint_dir}"
        )

    task_hashes = {stage["task_config_sha256"] for stage in stages}
    if len(task_hashes) != 1:
        raise ValueError(f"training chain crosses task configs: {sorted(task_hashes)}")
    chronological_schedule: list[dict[str, Any]] = []
    for stage in reversed(stages):
        if (
            chronological_schedule
            and chronological_schedule[-1]["program"] == stage["program"]
            and chronological_schedule[-1]["effective_batch"]
            == stage["effective_batch"]
        ):
            chronological_schedule[-1]["optimizer_updates"] += stage[
                "optimizer_updates"
            ]
            chronological_schedule[-1]["optimizer_examples"] += stage[
                "optimizer_examples"
            ]
        else:
            chronological_schedule.append(
                {
                    key: stage[key]
                    for key in (
                        "program",
                        "optimizer_updates",
                        "effective_batch",
                        "optimizer_examples",
                    )
                }
            )
    terminal_program = stages[0]["program"]
    program_optimizer_updates: dict[str, int] = {}
    for stage in stages:
        program = str(stage["program"])
        program_optimizer_updates[program] = (
            program_optimizer_updates.get(program, 0)
            + int(stage["optimizer_updates"])
        )
    precondition_optimizer_updates = sum(
        int(stage["optimizer_updates"])
        for stage in stages
        if stage["program"] != terminal_program
    )
    return {
        "stages": stages,
        "num_stages": len(stages),
        "total_optimizer_updates": sum(stage["optimizer_updates"] for stage in stages),
        "total_optimizer_examples": sum(
            stage["optimizer_examples"] for stage in stages
        ),
        "terminal_initial_params": str(terminal_initial_params),
        "task_config_sha256": next(iter(task_hashes)),
        "terminal_program": terminal_program,
        "chronological_program_schedule": chronological_schedule,
        "program_optimizer_updates": program_optimizer_updates,
        "precondition_optimizer_updates": precondition_optimizer_updates,
        "condition_from_terminal_initial_params": (
            precondition_optimizer_updates == 0
        ),
    }


def _paired_episode_metrics(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    tuple[str, ...],
    tuple[str, ...],
]:
    if len(reference_rows) != len(candidate_rows) or not reference_rows:
        raise ValueError("paired runs must contain the same non-zero episode count")
    pairs: list[dict[str, Any]] = []
    metric_names: tuple[str, ...] | None = None
    screening_metrics: tuple[str, ...] | None = None
    for reference, candidate in zip(reference_rows, candidate_rows, strict=True):
        identity_keys = ("episode_index", "seed", "policy_seed", "layout_fingerprint")
        mismatches = {
            key: {"reference": reference.get(key), "candidate": candidate.get(key)}
            for key in identity_keys
            if reference.get(key) != candidate.get(key)
        }
        if mismatches:
            raise ValueError(f"episode pairing mismatch: {mismatches}")
        reference_metrics = _metrics(reference)
        candidate_metrics = _metrics(candidate)
        if reference_metrics.keys() != candidate_metrics.keys():
            raise ValueError("paired episodes disagree on available task metrics")
        pair_metric_names = tuple(reference_metrics)
        pair_screening_metrics = (
            ("task_progress_score",)
            if "task_progress_score" in reference_metrics
            else ("max_reward", "total_reward")
        )
        if metric_names is None:
            metric_names = pair_metric_names
            screening_metrics = pair_screening_metrics
        elif (
            pair_metric_names != metric_names
            or pair_screening_metrics != screening_metrics
        ):
            raise ValueError("paired episodes use inconsistent task metric contracts")
        pairs.append(
            {
                "episode_index": int(reference["episode_index"]),
                "seed": int(reference["seed"]),
                "policy_seed": int(reference["policy_seed"]),
                "reference": reference_metrics,
                "candidate": candidate_metrics,
                "delta": {
                    metric: (
                        int(candidate_metrics[metric]) - int(reference_metrics[metric])
                        if metric == "success"
                        else candidate_metrics[metric] - reference_metrics[metric]
                    )
                    for metric in pair_metric_names
                },
            }
        )
    if metric_names is None or screening_metrics is None:
        raise ValueError("comparison has no paired metric contract")
    aggregate = {
        metric: {
            "reference_mean": mean(float(row["reference"][metric]) for row in pairs),
            "candidate_mean": mean(float(row["candidate"][metric]) for row in pairs),
            "mean_delta": mean(float(row["delta"][metric]) for row in pairs),
        }
        for metric in metric_names
    }
    return pairs, aggregate, metric_names, screening_metrics


def compare_training_runs(
    reference_run: pathlib.Path,
    candidate_run: pathlib.Path,
) -> dict[str, Any]:
    reference_config = _json(reference_run / "config.json")
    candidate_config = _json(candidate_run / "config.json")
    protected_keys = set(PROTECTED_CONFIG_KEYS) - ALLOWED_CONFIG_DIFFERENCES
    config_mismatches = {
        key: {
            "reference": reference_config.get(key),
            "candidate": candidate_config.get(key),
        }
        for key in protected_keys
        if reference_config.get(key) != candidate_config.get(key)
    }
    if config_mismatches:
        raise ValueError(
            f"training runs are not paired on protected variables: {config_mismatches}"
        )

    reference_chain = training_chain(pathlib.Path(reference_config["checkpoint_dir"]))
    candidate_chain = training_chain(pathlib.Path(candidate_config["checkpoint_dir"]))
    budget_mismatches = {}
    for key in (
        "total_optimizer_examples",
        "terminal_initial_params",
        "task_config_sha256",
    ):
        if reference_chain[key] != candidate_chain[key]:
            budget_mismatches[key] = {
                "reference": reference_chain[key],
                "candidate": candidate_chain[key],
            }
    if budget_mismatches:
        raise ValueError(
            f"training provenance is not budget matched: {budget_mismatches}"
        )

    schedule_alignment = {
        "reference_terminal_program": reference_chain["terminal_program"],
        "candidate_terminal_program": candidate_chain["terminal_program"],
        "reference_precondition_optimizer_updates": reference_chain[
            "precondition_optimizer_updates"
        ],
        "candidate_precondition_optimizer_updates": candidate_chain[
            "precondition_optimizer_updates"
        ],
        "both_conditions_from_terminal_initial_params": bool(
            reference_chain["condition_from_terminal_initial_params"]
            and candidate_chain["condition_from_terminal_initial_params"]
        ),
    }
    schedule_alignment["evidence_scope"] = (
        "condition_from_initial_params_comparison"
        if schedule_alignment["both_conditions_from_terminal_initial_params"]
        else "total_budget_matched_with_condition_warm_start"
    )

    pairs, aggregate, metric_names, screening_metrics = _paired_episode_metrics(
        _jsonl(reference_run / "episodes.jsonl"),
        _jsonl(candidate_run / "episodes.jsonl"),
    )
    reference_manifest = _json(reference_run / "emac_manifest.json")
    candidate_manifest = _json(candidate_run / "emac_manifest.json")
    reference_is_memory_run = "architecture" in reference_manifest
    candidate_is_memory_run = "architecture" in candidate_manifest
    runtime_alignment: dict[str, Any]
    if reference_is_memory_run and candidate_is_memory_run:
        reference_runtime = reference_manifest.get("runtime_source_sha256")
        candidate_runtime = candidate_manifest.get("runtime_source_sha256")
        if not reference_runtime or reference_runtime != candidate_runtime:
            raise ValueError(
                "memory training variants use different or missing frozen runtimes: "
                f"reference={reference_runtime}, candidate={candidate_runtime}"
            )
        reference_configs = reference_manifest.get("config_source_sha256")
        candidate_configs = candidate_manifest.get("config_source_sha256")
        if not reference_configs or reference_configs != candidate_configs:
            raise ValueError(
                "memory training variants use different or missing frozen config "
                "snapshots: "
                f"reference={reference_configs}, candidate={candidate_configs}"
            )
        runtime_alignment = {
            "required": True,
            "runtime_source_sha256": reference_runtime,
            "config_source_sha256": reference_configs,
        }
    else:
        runtime_alignment = {
            "required": False,
            "reason": "at_least_one_run_uses_the_native_non_memory_runtime",
        }
    return {
        "schema_version": "memory_harness.training_run_comparison/v3",
        "status": "paired_total_budget_matched_training_variants",
        "reference_run": str(reference_run.resolve()),
        "candidate_run": str(candidate_run.resolve()),
        "reference_architecture": reference_manifest.get(
            "architecture", reference_manifest.get("condition")
        ),
        "candidate_architecture": candidate_manifest.get(
            "architecture", candidate_manifest.get("condition")
        ),
        "reference_training": reference_chain,
        "candidate_training": candidate_chain,
        "training_schedule_alignment": schedule_alignment,
        "runtime_alignment": runtime_alignment,
        "num_pairs": len(pairs),
        "metric_names": list(metric_names),
        "screening_metrics": list(screening_metrics),
        "aggregate": aggregate,
        "pairs": pairs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired rollouts from budget-matched checkpoint training variants."
    )
    parser.add_argument("--reference-run", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-run", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = compare_training_runs(args.reference_run, args.candidate_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
