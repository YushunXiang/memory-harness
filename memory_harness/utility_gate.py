from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


EVIDENCE_KINDS = (
    "fixed_ablation",
    "zero_shot",
    "matched_training",
    "oracle_diagnostic",
    "content_intervention",
    "m0_control",
)
PILOT_PAIRS = 20
CONFIRMATION_PAIRS = 50
CI_LEVEL = 0.95
BOOTSTRAP_SAMPLES = 20_000


def _paired_deltas(comparison: Mapping[str, Any], metric: str) -> np.ndarray:
    pairs = comparison.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("comparison must contain a non-empty pairs list")
    try:
        values = np.asarray(
            [float(pair["delta"][metric]) for pair in pairs], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"comparison has invalid paired {metric} deltas") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"comparison has non-finite paired {metric} deltas")
    return values


def _paired_bootstrap_interval(
    deltas: np.ndarray,
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if deltas.size == 1 or np.all(deltas == deltas[0]):
        value = float(deltas[0])
        return value, value

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    batch_size = min(samples, 2_000)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, deltas.size, size=(stop - start, deltas.size))
        bootstrap_means[start:stop] = deltas[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _metric_summary(
    deltas: np.ndarray,
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    tolerance = 1e-12
    lower, upper = _paired_bootstrap_interval(
        deltas,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "mean_delta": float(deltas.mean()),
        "wins": int(np.count_nonzero(deltas > tolerance)),
        "losses": int(np.count_nonzero(deltas < -tolerance)),
        "ties": int(np.count_nonzero(np.abs(deltas) <= tolerance)),
        "paired_interval": {
            "confidence": confidence,
            "lower": lower,
            "upper": upper,
            "method": "paired_episode_percentile_bootstrap",
            "samples": bootstrap_samples,
            "seed": seed,
        },
    }


def evaluate_candidate_utility(
    comparison: Mapping[str, Any],
    *,
    evidence_kind: str,
    pilot_pairs: int = PILOT_PAIRS,
    confirmation_pairs: int = CONFIRMATION_PAIRS,
    confidence: float = CI_LEVEL,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """Classify one paired comparison without pretending it passes full Gate 1.

    Success is the primary endpoint. Max and total reward are screening signals only;
    they can justify collecting more evidence but can never make a candidate eligible
    for the controller by themselves.
    """
    if evidence_kind not in EVIDENCE_KINDS:
        raise ValueError(
            f"unknown evidence kind {evidence_kind!r}; expected one of {EVIDENCE_KINDS}"
        )
    if pilot_pairs <= 0 or confirmation_pairs < pilot_pairs:
        raise ValueError("require 0 < pilot_pairs <= confirmation_pairs")
    supported_comparison_schemas = {
        "memory_harness.fixed_run_comparison/v2",
        "memory_harness.training_run_comparison/v3",
    }
    if comparison.get("schema_version") not in supported_comparison_schemas:
        raise ValueError("utility gate requires a v2 paired comparison")
    screening_names = comparison.get("screening_metrics")
    if (
        not isinstance(screening_names, list)
        or not screening_names
        or any(
            name not in {"max_reward", "total_reward", "task_progress_score"}
            for name in screening_names
        )
        or len(screening_names) != len(set(screening_names))
    ):
        raise ValueError("comparison declares invalid screening_metrics")

    success = _paired_deltas(comparison, "success")
    screening_deltas = {
        name: _paired_deltas(comparison, name) for name in screening_names
    }
    if any(success.size != values.size for values in screening_deltas.values()):
        raise ValueError("paired metrics must have equal lengths")
    declared_pairs = comparison.get("num_pairs")
    if declared_pairs is not None and int(declared_pairs) != success.size:
        raise ValueError("num_pairs does not match pairs list")

    metrics: dict[str, Any] = {
        "success": _metric_summary(
            success,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }
    for offset, (name, deltas) in enumerate(screening_deltas.items(), start=1):
        metrics[name] = _metric_summary(
            deltas,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=seed + offset,
        )
    num_pairs = int(success.size)
    if num_pairs < pilot_pairs:
        sample_stage = "screen"
    elif num_pairs < confirmation_pairs:
        sample_stage = "pilot"
    else:
        sample_stage = "confirmation"

    success_mean = metrics["success"]["mean_delta"]
    success_interval = metrics["success"]["paired_interval"]
    stage_positive = all(
        metrics[name]["mean_delta"] > 0.0
        and metrics[name]["wins"] > metrics[name]["losses"]
        for name in screening_names
    )
    stage_negative = all(
        metrics[name]["mean_delta"] < 0.0
        and metrics[name]["losses"] > metrics[name]["wins"]
        for name in screening_names
    )
    confirmed_gain = (
        num_pairs >= confirmation_pairs and success_interval["lower"] > 0.0
    )
    confirmed_harm = (
        num_pairs >= confirmation_pairs and success_interval["upper"] < 0.0
    )

    if confirmed_gain:
        signal = "confirmed_success_gain"
    elif confirmed_harm:
        signal = "confirmed_success_harm"
    elif success_mean > 0.0:
        signal = f"{sample_stage}_success_gain"
    elif success_mean < 0.0:
        signal = f"{sample_stage}_success_harm"
    elif stage_positive:
        signal = "positive_stage_only"
    elif stage_negative:
        signal = "negative_stage_only"
    else:
        signal = "no_detectable_direction"

    utility_evidence_kinds = {"fixed_ablation", "matched_training"}
    candidate_requirement_met = confirmed_gain and evidence_kind in utility_evidence_kinds
    positive_screen = success_mean > 0.0 or (
        success_mean == 0.0 and stage_positive
    )
    negative_screen = success_mean < 0.0 or (
        success_mean == 0.0 and stage_negative
    )
    required_evidence_kinds = {
        "fixed_ablation",
        "matched_training",
        "oracle_diagnostic",
        "content_intervention",
        "m0_control",
    }

    if candidate_requirement_met:
        next_action = "assemble_gate1_diagnostic_bundle"
    elif confirmed_harm:
        next_action = "reject_candidate"
    elif num_pairs < pilot_pairs:
        if evidence_kind in required_evidence_kinds or positive_screen:
            next_action = "collect_shared_episodes_to_20"
        elif negative_screen:
            next_action = "reject_or_redesign_candidate"
        else:
            next_action = "retain_as_inconclusive_diagnostic"
    elif evidence_kind == "zero_shot" and positive_screen:
        next_action = "run_budget_matched_training"
    elif num_pairs < confirmation_pairs and positive_screen:
        next_action = "collect_shared_episodes_to_50"
    elif negative_screen:
        next_action = "reject_or_redesign_candidate"
    else:
        next_action = "retain_as_inconclusive_diagnostic"

    missing_to_pilot = max(0, pilot_pairs - num_pairs)
    missing_to_confirmation = max(0, confirmation_pairs - num_pairs)
    return {
        "schema_version": "memory_harness.utility_decision/v2",
        "decision_scope": "single_candidate_utility_requirement",
        "evidence_kind": evidence_kind,
        "num_pairs": num_pairs,
        "sample_stage": sample_stage,
        "thresholds": {
            "pilot_pairs": pilot_pairs,
            "confirmation_pairs": confirmation_pairs,
            "confidence": confidence,
            "primary_endpoint": "success",
            "secondary_endpoints": screening_names,
        },
        "metrics": metrics,
        "signal": signal,
        "candidate_utility_requirement_met": candidate_requirement_met,
        "full_gate1_passed": False,
        "full_gate1_note": (
            "Not evaluated here: Gate 1 additionally requires oracle/content "
            "interventions and an acceptable M(0)-control regression check."
        ),
        "remaining_pairs": {
            "to_pilot": missing_to_pilot,
            "to_confirmation": missing_to_confirmation,
        },
        "next_action": next_action,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a paired run comparison into a pre-registered utility decision."
    )
    parser.add_argument("--comparison", type=pathlib.Path, required=True)
    parser.add_argument("--evidence-kind", choices=EVIDENCE_KINDS, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    if not isinstance(comparison, dict):
        raise ValueError("comparison JSON must contain an object")
    result = evaluate_candidate_utility(
        comparison,
        evidence_kind=args.evidence_kind,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "signal": result["signal"],
                "candidate_utility_requirement_met": result[
                    "candidate_utility_requirement_met"
                ],
                "next_action": result["next_action"],
                "remaining_pairs": result["remaining_pairs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
