from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any


PROTECTED_CONFIG_KEYS = (
    "task_name",
    "task_config",
    "config",
    "policy_config",
    "checkpoint_dir",
    "policy_asset_id",
    "policy_assets_dir",
    "policy_norm_stats",
    "seed",
    "policy_seed_base",
    "num_episodes",
    "max_steps",
    "execute_action_chunk_steps",
    "camera_names",
    "action_safety_mode",
    "action_delta_clip_scale",
    "paired_layout_protocol",
    "prompt",
    "prompt_protocol",
    "phase_aware_subtask_prompt",
    "task_state_trace_frequency",
    "memory_planner_global_task",
    "memory_planner_boundary_mode",
)


def _json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _metrics(row: Mapping[str, Any]) -> dict[str, float | bool]:
    final_info = row.get("final_info")
    if not isinstance(final_info, Mapping):
        raise ValueError("episode row is missing final_info")
    metrics: dict[str, float | bool] = {
        "success": bool(row.get("success", False)),
        "max_reward": float(final_info.get("max_reward", 0.0)),
        "total_reward": float(row.get("total_reward", 0.0)),
        "steps": float(row.get("steps", 0.0)),
    }
    task_progress = row.get("task_progress")
    if task_progress is not None:
        if not isinstance(task_progress, Mapping):
            raise ValueError("episode task_progress must be an object")
        if task_progress.get("task") != "put_back_block":
            raise ValueError("unsupported episode task_progress payload")
        try:
            score = float(task_progress["max_progress_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("episode task_progress has no valid max_progress_score") from exc
        if not 0.0 <= score <= 3.0:
            raise ValueError("put_back_block max_progress_score must be in [0, 3]")
        metrics["task_progress_score"] = score
    return metrics


def _metric_contract(
    reference: Mapping[str, float | bool],
    candidate: Mapping[str, float | bool],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if reference.keys() != candidate.keys():
        raise ValueError("paired episodes disagree on available task metrics")
    metric_names = tuple(reference)
    screening_metrics = (
        ("task_progress_score",)
        if "task_progress_score" in reference
        else ("max_reward", "total_reward")
    )
    return metric_names, screening_metrics


def compare_runs(
    reference_run: pathlib.Path, candidate_run: pathlib.Path
) -> dict[str, Any]:
    reference_config = _json(reference_run / "config.json")
    candidate_config = _json(candidate_run / "config.json")
    mismatches = {
        key: {
            "reference": reference_config.get(key),
            "candidate": candidate_config.get(key),
        }
        for key in PROTECTED_CONFIG_KEYS
        if reference_config.get(key) != candidate_config.get(key)
    }
    if mismatches:
        raise ValueError(f"runs are not paired on protected variables: {mismatches}")

    reference_manifest = _json(reference_run / "emac_manifest.json")
    candidate_manifest = _json(candidate_run / "emac_manifest.json")
    reference_runtime = reference_manifest.get("runtime_source_sha256")
    candidate_runtime = candidate_manifest.get("runtime_source_sha256")
    if not reference_runtime or reference_runtime != candidate_runtime:
        raise ValueError(
            "runs use different or missing frozen memory runtimes: "
            f"reference={reference_runtime}, candidate={candidate_runtime}"
        )
    reference_config_source = reference_manifest.get("config_source_sha256")
    candidate_config_source = candidate_manifest.get("config_source_sha256")
    if (
        not reference_config_source
        or reference_config_source != candidate_config_source
    ):
        raise ValueError(
            "runs use different or missing frozen config snapshots: "
            f"reference={reference_config_source}, candidate={candidate_config_source}"
        )
    reference_task = reference_manifest.get("task_config_sha256")
    candidate_task = candidate_manifest.get("task_config_sha256")
    if not reference_task or reference_task != candidate_task:
        raise ValueError(
            "runs use different or missing frozen task configs: "
            f"reference={reference_task}, candidate={candidate_task}"
        )
    reference_suite = reference_manifest.get("candidate_suite_manifest_sha256")
    candidate_suite = candidate_manifest.get("candidate_suite_manifest_sha256")
    if reference_suite != candidate_suite:
        raise ValueError(
            "runs use different candidate-suite provenance: "
            f"reference={reference_suite}, candidate={candidate_suite}"
        )
    required_manifest_hashes = (
        "architecture_config_sha256",
        "executor_program_config_sha256",
    )
    missing_manifest_hashes = {
        side: [key for key in required_manifest_hashes if not manifest.get(key)]
        for side, manifest in (
            ("reference", reference_manifest),
            ("candidate", candidate_manifest),
        )
        if any(not manifest.get(key) for key in required_manifest_hashes)
    }
    if missing_manifest_hashes:
        raise ValueError(
            f"runs are missing frozen architecture identities: {missing_manifest_hashes}"
        )
    reference_episodes = _jsonl(reference_run / "episodes.jsonl")
    candidate_episodes = _jsonl(candidate_run / "episodes.jsonl")
    if len(reference_episodes) != len(candidate_episodes) or not reference_episodes:
        raise ValueError("paired runs must contain the same non-zero episode count")

    pairs: list[dict[str, Any]] = []
    metric_names: tuple[str, ...] | None = None
    screening_metrics: tuple[str, ...] | None = None
    for reference, candidate in zip(
        reference_episodes, candidate_episodes, strict=True
    ):
        identity_keys = ("episode_index", "seed", "policy_seed", "layout_fingerprint")
        identity_mismatches = {
            key: {"reference": reference.get(key), "candidate": candidate.get(key)}
            for key in identity_keys
            if reference.get(key) != candidate.get(key)
        }
        if identity_mismatches:
            raise ValueError(f"episode pairing mismatch: {identity_mismatches}")
        reference_metrics = _metrics(reference)
        candidate_metrics = _metrics(candidate)
        pair_metric_names, pair_screening_metrics = _metric_contract(
            reference_metrics, candidate_metrics
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
                "layout_fingerprint": reference["layout_fingerprint"],
                "reference": reference_metrics,
                "candidate": candidate_metrics,
                "delta": {
                    metric: (
                        int(candidate_metrics[metric]) - int(reference_metrics[metric])
                        if metric == "success"
                        else float(candidate_metrics[metric])
                        - float(reference_metrics[metric])
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
    return {
        "schema_version": "memory_harness.fixed_run_comparison/v2",
        "status": "paired",
        "reference_run": str(reference_run.resolve()),
        "candidate_run": str(candidate_run.resolve()),
        "reference_architecture": reference_manifest["architecture"],
        "candidate_architecture": candidate_manifest["architecture"],
        "reference_planner_model": reference_manifest.get("planner_model"),
        "candidate_planner_model": candidate_manifest.get("planner_model"),
        "runtime_source_sha256": reference_runtime,
        "config_source_sha256": reference_config_source,
        "task_config_sha256": reference_task,
        "candidate_suite_manifest_sha256": reference_suite,
        "reference_architecture_config_sha256": reference_manifest[
            "architecture_config_sha256"
        ],
        "candidate_architecture_config_sha256": candidate_manifest[
            "architecture_config_sha256"
        ],
        "reference_executor_program_config_sha256": reference_manifest[
            "executor_program_config_sha256"
        ],
        "candidate_executor_program_config_sha256": candidate_manifest[
            "executor_program_config_sha256"
        ],
        "num_pairs": len(pairs),
        "metric_names": list(metric_names),
        "screening_metrics": list(screening_metrics),
        "aggregate": aggregate,
        "pairs": pairs,
    }


def compare_run_sets(
    reference_runs: Sequence[pathlib.Path], candidate_runs: Sequence[pathlib.Path]
) -> dict[str, Any]:
    if len(reference_runs) != len(candidate_runs) or not reference_runs:
        raise ValueError("run sets must contain the same non-zero number of runs")
    comparisons = [
        compare_runs(reference.resolve(), candidate.resolve())
        for reference, candidate in zip(reference_runs, candidate_runs, strict=True)
    ]
    identity_keys = (
        "reference_architecture",
        "candidate_architecture",
        "reference_planner_model",
        "candidate_planner_model",
        "runtime_source_sha256",
        "config_source_sha256",
        "task_config_sha256",
        "candidate_suite_manifest_sha256",
        "reference_architecture_config_sha256",
        "candidate_architecture_config_sha256",
        "reference_executor_program_config_sha256",
        "candidate_executor_program_config_sha256",
    )
    for key in identity_keys:
        values = {comparison[key] for comparison in comparisons}
        if len(values) != 1:
            raise ValueError(f"run-set identity mismatch for {key}: {values}")
    metric_contracts = {
        (
            tuple(comparison["metric_names"]),
            tuple(comparison["screening_metrics"]),
        )
        for comparison in comparisons
    }
    if len(metric_contracts) != 1:
        raise ValueError(f"run-set metric contract mismatch: {metric_contracts}")
    metric_names, screening_metrics = next(iter(metric_contracts))

    pairs = [
        {**pair, "run_pair_index": run_pair_index}
        for run_pair_index, comparison in enumerate(comparisons)
        for pair in comparison["pairs"]
    ]
    evidence_identities = [
        (
            pair["seed"],
            pair["policy_seed"],
            json.dumps(pair["layout_fingerprint"], sort_keys=True),
        )
        for pair in pairs
    ]
    if len(evidence_identities) != len(set(evidence_identities)):
        raise ValueError(
            "run sets contain duplicate paired episode evidence; use disjoint "
            "layout and policy seeds for successive evaluation"
        )
    aggregate = {
        metric: {
            "reference_mean": mean(float(row["reference"][metric]) for row in pairs),
            "candidate_mean": mean(float(row["candidate"][metric]) for row in pairs),
            "mean_delta": mean(float(row["delta"][metric]) for row in pairs),
        }
        for metric in metric_names
    }
    first = comparisons[0]
    return {
        "schema_version": "memory_harness.fixed_run_comparison/v2",
        "status": "paired_run_set",
        "reference_runs": [comparison["reference_run"] for comparison in comparisons],
        "candidate_runs": [comparison["candidate_run"] for comparison in comparisons],
        **{key: first[key] for key in identity_keys},
        "num_run_pairs": len(comparisons),
        "num_pairs": len(pairs),
        "metric_names": list(metric_names),
        "screening_metrics": list(screening_metrics),
        "evidence_identity": "seed+policy_seed+layout_fingerprint",
        "aggregate": aggregate,
        "pairs": pairs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly pair and compare two fixed-memory runs."
    )
    parser.add_argument(
        "--reference-run", type=pathlib.Path, action="append", required=True
    )
    parser.add_argument(
        "--candidate-run", type=pathlib.Path, action="append", required=True
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = compare_run_sets(args.reference_run, args.candidate_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
