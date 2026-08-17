from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any


CONDITIONS = ("full_memory", "empty_mask", "native_none")


def _identity(pair: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        pair.get("episode_index"),
        pair.get("seed"),
        pair.get("policy_seed"),
    )


def _metrics(
    pair: Mapping[str, Any], side: str, screening_metrics: Sequence[str]
) -> dict[str, float]:
    try:
        raw = pair[side]
        result = {
            "success": float(raw["success"]),
            "max_reward": float(raw["max_reward"]),
            "total_reward": float(raw["total_reward"]),
        }
        for metric in screening_metrics:
            result[metric] = float(raw[metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"comparison has invalid {side} metrics") from exc
    return result


def _validated_pairs(comparison: Mapping[str, Any], *, label: str) -> list[Mapping[str, Any]]:
    if comparison.get("schema_version") != "memory_harness.training_run_comparison/v3":
        raise ValueError(f"{label} has unsupported comparison schema")
    if comparison.get("status") != "paired_total_budget_matched_training_variants":
        raise ValueError(f"{label} is not a paired total-budget-matched comparison")
    for side in ("reference_training", "candidate_training"):
        training = comparison.get(side)
        if not isinstance(training, dict):
            raise ValueError(f"{label} lacks {side} provenance")
        if not isinstance(training.get("condition_from_terminal_initial_params"), bool):
            raise ValueError(f"{label} lacks {side} condition-schedule provenance")
        if not isinstance(training.get("precondition_optimizer_updates"), int):
            raise ValueError(f"{label} has invalid {side} precondition exposure")
    pairs = comparison.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{label} must contain non-empty pairs")
    if int(comparison.get("num_pairs", -1)) != len(pairs):
        raise ValueError(f"{label} num_pairs does not match its pairs")
    screening_metrics = comparison.get("screening_metrics")
    if (
        not isinstance(screening_metrics, list)
        or not screening_metrics
        or any(
            metric not in {"max_reward", "total_reward", "task_progress_score"}
            for metric in screening_metrics
        )
    ):
        raise ValueError(f"{label} has invalid screening_metrics")
    identities = [_identity(pair) for pair in pairs]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{label} contains duplicate episode identities")
    return pairs


def _assert_same_metrics(
    left: Mapping[str, float], right: Mapping[str, float], *, label: str
) -> None:
    if dict(left) != dict(right):
        raise ValueError(f"inconsistent {label} metrics across comparisons")


def assess_executor_readiness(
    empty_vs_full: Mapping[str, Any],
    native_vs_full: Mapping[str, Any],
    native_vs_empty: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose the next experiment only after three budget-matched controls exist."""

    comparisons = {
        "empty_vs_full": empty_vs_full,
        "native_vs_full": native_vs_full,
        "native_vs_empty": native_vs_empty,
    }
    endpoints = {
        "full": str(empty_vs_full.get("candidate_run")),
        "empty": str(empty_vs_full.get("reference_run")),
        "native": str(native_vs_full.get("reference_run")),
    }
    expected_endpoints = {
        "native_vs_full.candidate": endpoints["full"],
        "native_vs_empty.candidate": endpoints["empty"],
        "native_vs_empty.reference": endpoints["native"],
    }
    actual_endpoints = {
        "native_vs_full.candidate": str(native_vs_full.get("candidate_run")),
        "native_vs_empty.candidate": str(native_vs_empty.get("candidate_run")),
        "native_vs_empty.reference": str(native_vs_empty.get("reference_run")),
    }
    endpoint_mismatches = {
        key: {"expected": expected_endpoints[key], "actual": actual_endpoints[key]}
        for key in expected_endpoints
        if expected_endpoints[key] != actual_endpoints[key]
    }
    if endpoint_mismatches:
        raise ValueError(f"comparison graph does not describe three runs: {endpoint_mismatches}")

    pair_sets = {
        label: _validated_pairs(comparison, label=label)
        for label, comparison in comparisons.items()
    }
    screening_contracts = {
        tuple(comparison["screening_metrics"])
        for comparison in comparisons.values()
    }
    if len(screening_contracts) != 1:
        raise ValueError("comparison graph uses different screening metrics")
    screening_metrics = next(iter(screening_contracts))
    by_identity = {
        label: {_identity(pair): pair for pair in pairs}
        for label, pairs in pair_sets.items()
    }
    identities = set(by_identity["empty_vs_full"])
    for label, pairs in by_identity.items():
        if set(pairs) != identities:
            raise ValueError(f"{label} does not use the same paired episodes")

    observations: dict[str, list[dict[str, float]]] = {
        condition: [] for condition in CONDITIONS
    }
    for identity in sorted(identities):
        empty_full = by_identity["empty_vs_full"][identity]
        native_full = by_identity["native_vs_full"][identity]
        native_empty = by_identity["native_vs_empty"][identity]
        full_a = _metrics(empty_full, "candidate", screening_metrics)
        full_b = _metrics(native_full, "candidate", screening_metrics)
        empty_a = _metrics(empty_full, "reference", screening_metrics)
        empty_b = _metrics(native_empty, "candidate", screening_metrics)
        native_a = _metrics(native_full, "reference", screening_metrics)
        native_b = _metrics(native_empty, "reference", screening_metrics)
        _assert_same_metrics(full_a, full_b, label="full_memory")
        _assert_same_metrics(empty_a, empty_b, label="empty_mask")
        _assert_same_metrics(native_a, native_b, label="native_none")
        observations["full_memory"].append(full_a)
        observations["empty_mask"].append(empty_a)
        observations["native_none"].append(native_a)

    training_conditions = {
        "full_memory": dict(empty_vs_full["candidate_training"]),
        "empty_mask": dict(empty_vs_full["reference_training"]),
        "native_none": dict(native_vs_full["reference_training"]),
    }
    training_graph_checks = (
        (
            "full_memory",
            training_conditions["full_memory"],
            dict(native_vs_full["candidate_training"]),
        ),
        (
            "empty_mask",
            training_conditions["empty_mask"],
            dict(native_vs_empty["candidate_training"]),
        ),
        (
            "native_none",
            training_conditions["native_none"],
            dict(native_vs_empty["reference_training"]),
        ),
    )
    for condition, left, right in training_graph_checks:
        if left != right:
            raise ValueError(
                f"inconsistent {condition} training provenance across comparisons"
            )

    summaries: dict[str, dict[str, Any]] = {}
    for condition, rows in observations.items():
        successes = sum(int(row["success"]) for row in rows)
        max_reward = max(row["max_reward"] for row in rows)
        screening_maxima = {
            metric: max(row[metric] for row in rows) for metric in screening_metrics
        }
        summaries[condition] = {
            "num_episodes": len(rows),
            "num_successes": successes,
            "max_observed_reward": max_reward,
            "mean_total_reward": mean(row["total_reward"] for row in rows),
            "screening_metrics": screening_maxima,
            "observable_executor_signal": successes > 0
            or any(value > 0.0 for value in screening_maxima.values()),
        }

    if summaries["full_memory"]["observable_executor_signal"]:
        status = "full_memory_executor_ready"
        next_action = "collect_fixed_ablation_to_20"
        reason = "full-memory training produced success or non-zero task progress"
    elif any(
        summaries[name]["observable_executor_signal"]
        for name in ("empty_mask", "native_none")
    ):
        status = "controls_ready_full_memory_not_ready"
        next_action = "retrain_full_memory_at_higher_budget_before_gate20"
        reason = "a no-content control shows task progress, but full-memory remains at floor"
    else:
        status = "all_variants_without_observed_executor_signal"
        next_action = "increase_training_budget_before_more_rollouts"
        reason = "all three budget-matched variants have zero success and zero task progress"

    return {
        "schema_version": "memory_harness.executor_readiness/v2",
        "status": status,
        "num_paired_episodes": len(identities),
        "runs": endpoints,
        "conditions": summaries,
        "training_conditions": training_conditions,
        "training_evidence_scope": (
            "condition_from_initial_params_comparison"
            if all(
                training["condition_from_terminal_initial_params"]
                for training in training_conditions.values()
            )
            else "readiness_screen_with_condition_warm_start"
        ),
        "condition_schedule_confirmation": all(
            training["condition_from_terminal_initial_params"]
            for training in training_conditions.values()
        ),
        "screening_metrics": list(screening_metrics),
        "decision": {
            "next_action": next_action,
            "reason": reason,
            "controller_training_allowed": False,
            "fixed_gate20_allowed": next_action == "collect_fixed_ablation_to_20",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decide the next Put Back executor experiment from three paired controls."
    )
    parser.add_argument("--empty-vs-full", type=pathlib.Path, required=True)
    parser.add_argument("--native-vs-full", type=pathlib.Path, required=True)
    parser.add_argument("--native-vs-empty", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def _load(path: pathlib.Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"comparison must be a JSON object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = assess_executor_readiness(
        _load(args.empty_vs_full),
        _load(args.native_vs_full),
        _load(args.native_vs_empty),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
