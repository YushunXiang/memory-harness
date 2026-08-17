from __future__ import annotations

import pytest

from memory_harness.utility_gate import evaluate_candidate_utility


def _comparison(
    success: list[float],
    max_reward: list[float] | None = None,
    total_reward: list[float] | None = None,
    task_progress: list[float] | None = None,
) -> dict[str, object]:
    max_reward = max_reward if max_reward is not None else success
    total_reward = total_reward if total_reward is not None else success
    assert len(success) == len(max_reward) == len(total_reward)
    screening_metrics = (
        ["task_progress_score"] if task_progress is not None else ["max_reward", "total_reward"]
    )
    return {
        "schema_version": "memory_harness.fixed_run_comparison/v2",
        "screening_metrics": screening_metrics,
        "num_pairs": len(success),
        "pairs": [
            {
                "delta": {
                    "success": success_delta,
                    "max_reward": max_delta,
                    "total_reward": total_delta,
                    **(
                        {"task_progress_score": task_progress[index]}
                        if task_progress is not None
                        else {}
                    ),
                }
            }
            for index, (success_delta, max_delta, total_delta) in enumerate(
                zip(success, max_reward, total_reward, strict=True)
            )
        ],
    }


def test_three_episode_stage_signal_requests_more_shared_episodes() -> None:
    result = evaluate_candidate_utility(
        _comparison([0, 0, 0], [0.1, 0.2, 0.1], [10, 20, 30]),
        evidence_kind="zero_shot",
        bootstrap_samples=100,
    )

    assert result["sample_stage"] == "screen"
    assert result["signal"] == "positive_stage_only"
    assert result["candidate_utility_requirement_met"] is False
    assert result["remaining_pairs"] == {"to_pilot": 17, "to_confirmation": 47}
    assert result["next_action"] == "collect_shared_episodes_to_20"


def test_zero_shot_negative_screen_does_not_consume_pilot_budget() -> None:
    result = evaluate_candidate_utility(
        _comparison([0, 0, 0], [-0.1, -0.2, -0.1], [-10, -20, -30]),
        evidence_kind="zero_shot",
        bootstrap_samples=100,
    )

    assert result["signal"] == "negative_stage_only"
    assert result["next_action"] == "reject_or_redesign_candidate"


def test_sparse_reward_task_uses_declared_progress_metric() -> None:
    result = evaluate_candidate_utility(
        _comparison(
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            task_progress=[1, 1, 2],
        ),
        evidence_kind="zero_shot",
        bootstrap_samples=100,
    )

    assert result["signal"] == "positive_stage_only"
    assert result["thresholds"]["secondary_endpoints"] == ["task_progress_score"]
    assert result["next_action"] == "collect_shared_episodes_to_20"


def test_zero_shot_directionless_screen_is_retained_without_escalation() -> None:
    result = evaluate_candidate_utility(
        _comparison([0, 0, 0], [0, 0, 0], [0, 0, 0]),
        evidence_kind="zero_shot",
        bootstrap_samples=100,
    )

    assert result["signal"] == "no_detectable_direction"
    assert result["next_action"] == "retain_as_inconclusive_diagnostic"


def test_fixed_ablation_reaches_pilot_even_without_screen_direction() -> None:
    result = evaluate_candidate_utility(
        _comparison([0, 0, 0], [0, 0, 0], [0, 0, 0]),
        evidence_kind="fixed_ablation",
        bootstrap_samples=100,
    )

    assert result["next_action"] == "collect_shared_episodes_to_20"


def test_twenty_episode_zero_shot_signal_requests_matched_training() -> None:
    result = evaluate_candidate_utility(
        _comparison([1] * 5 + [0] * 15),
        evidence_kind="zero_shot",
        bootstrap_samples=200,
    )

    assert result["sample_stage"] == "pilot"
    assert result["signal"] == "pilot_success_gain"
    assert result["next_action"] == "run_budget_matched_training"
    assert result["candidate_utility_requirement_met"] is False


def test_confirmed_fixed_ablation_gain_meets_only_candidate_requirement() -> None:
    result = evaluate_candidate_utility(
        _comparison([1] * 15 + [0] * 35),
        evidence_kind="fixed_ablation",
        bootstrap_samples=1_000,
    )

    assert result["sample_stage"] == "confirmation"
    assert result["metrics"]["success"]["paired_interval"]["lower"] > 0.0
    assert result["signal"] == "confirmed_success_gain"
    assert result["candidate_utility_requirement_met"] is True
    assert result["full_gate1_passed"] is False
    assert result["next_action"] == "assemble_gate1_diagnostic_bundle"


def test_stage_only_gain_never_meets_candidate_utility_requirement() -> None:
    result = evaluate_candidate_utility(
        _comparison([0] * 50, [0.1] * 50, [1.0] * 50),
        evidence_kind="matched_training",
        bootstrap_samples=100,
    )

    assert result["signal"] == "positive_stage_only"
    assert result["candidate_utility_requirement_met"] is False
    assert result["next_action"] == "retain_as_inconclusive_diagnostic"


def test_confirmed_harm_rejects_candidate() -> None:
    result = evaluate_candidate_utility(
        _comparison([-1] * 12 + [0] * 38),
        evidence_kind="fixed_ablation",
        bootstrap_samples=1_000,
    )

    assert result["signal"] == "confirmed_success_harm"
    assert result["candidate_utility_requirement_met"] is False
    assert result["next_action"] == "reject_candidate"


def test_rejects_unknown_evidence_kind() -> None:
    with pytest.raises(ValueError, match="unknown evidence kind"):
        evaluate_candidate_utility(
            _comparison([0]), evidence_kind="invented", bootstrap_samples=10
        )
