from __future__ import annotations

import copy

import pytest

from memory_harness.decide_executor_readiness import assess_executor_readiness


def _pair(
    episode: int,
    reference: tuple[int, float, float, float],
    candidate: tuple[int, float, float, float],
):
    def metrics(values):
        return {
            "success": values[0],
            "max_reward": values[1],
            "total_reward": values[2],
            "task_progress_score": values[3],
            "steps": 500,
        }

    return {
        "episode_index": episode,
        "seed": 100000 + episode,
        "policy_seed": 120000 + episode,
        "reference": metrics(reference),
        "candidate": metrics(candidate),
        "delta": {},
    }


def _training(
    program: str,
    *,
    from_initial_params: bool,
    precondition_updates: int,
) -> dict:
    return {
        "terminal_program": program,
        "condition_from_terminal_initial_params": from_initial_params,
        "precondition_optimizer_updates": precondition_updates,
    }


def _comparison(
    reference_run: str,
    candidate_run: str,
    pairs: list[dict],
    *,
    reference_training: dict,
    candidate_training: dict,
):
    return {
        "schema_version": "memory_harness.training_run_comparison/v3",
        "status": "paired_total_budget_matched_training_variants",
        "reference_run": reference_run,
        "candidate_run": candidate_run,
        "reference_training": reference_training,
        "candidate_training": candidate_training,
        "num_pairs": len(pairs),
        "screening_metrics": ["task_progress_score"],
        "pairs": pairs,
    }


def _graph(
    full=(0, 0.0, 0.0, 0.0),
    empty=(0, 0.0, 0.0, 0.0),
    native=(0, 0.0, 0.0, 0.0),
):
    full_training = _training(
        "anchor_sliding", from_initial_params=False, precondition_updates=200
    )
    empty_training = _training(
        "none", from_initial_params=True, precondition_updates=0
    )
    native_training = _training(
        "native_none", from_initial_params=True, precondition_updates=0
    )
    empty_full = _comparison(
        "/empty",
        "/full",
        [_pair(0, empty, full)],
        reference_training=empty_training,
        candidate_training=full_training,
    )
    native_full = _comparison(
        "/native",
        "/full",
        [_pair(0, native, full)],
        reference_training=native_training,
        candidate_training=full_training,
    )
    native_empty = _comparison(
        "/native",
        "/empty",
        [_pair(0, native, empty)],
        reference_training=native_training,
        candidate_training=empty_training,
    )
    return empty_full, native_full, native_empty


def test_all_zero_variants_increase_training_before_more_rollouts() -> None:
    result = assess_executor_readiness(*_graph())
    assert result["status"] == "all_variants_without_observed_executor_signal"
    assert result["decision"]["next_action"] == "increase_training_budget_before_more_rollouts"
    assert not result["decision"]["fixed_gate20_allowed"]
    assert result["training_evidence_scope"] == (
        "readiness_screen_with_condition_warm_start"
    )
    assert not result["condition_schedule_confirmation"]


def test_full_memory_stage_signal_allows_fixed_gate20() -> None:
    result = assess_executor_readiness(*_graph(full=(0, 0.1, 2.0, 1.0)))
    assert result["status"] == "full_memory_executor_ready"
    assert result["decision"]["next_action"] == "collect_fixed_ablation_to_20"
    assert result["decision"]["fixed_gate20_allowed"]


def test_control_only_signal_requires_full_memory_retraining() -> None:
    result = assess_executor_readiness(*_graph(native=(1, 1.0, 10.0, 3.0)))
    assert result["status"] == "controls_ready_full_memory_not_ready"
    assert result["decision"]["next_action"] == "retrain_full_memory_at_higher_budget_before_gate20"


def test_rejects_inconsistent_shared_run_metrics() -> None:
    graph = list(_graph(full=(0, 0.1, 2.0, 1.0)))
    graph[1] = copy.deepcopy(graph[1])
    graph[1]["pairs"][0]["candidate"]["max_reward"] = 0.2
    with pytest.raises(ValueError, match="inconsistent full_memory"):
        assess_executor_readiness(*graph)


def test_rejects_mismatched_comparison_graph() -> None:
    graph = list(_graph())
    graph[2] = copy.deepcopy(graph[2])
    graph[2]["candidate_run"] = "/wrong-empty"
    with pytest.raises(ValueError, match="comparison graph"):
        assess_executor_readiness(*graph)


def test_rejects_inconsistent_shared_training_provenance() -> None:
    graph = list(_graph())
    graph[1] = copy.deepcopy(graph[1])
    graph[1]["candidate_training"]["precondition_optimizer_updates"] = 0

    with pytest.raises(ValueError, match="inconsistent full_memory training provenance"):
        assess_executor_readiness(*graph)


def test_put_back_progress_allows_fixed_gate_without_sparse_reward() -> None:
    result = assess_executor_readiness(
        *_graph(full=(0, 0.0, 0.0, 1.0))
    )

    assert result["status"] == "full_memory_executor_ready"
    assert result["conditions"]["full_memory"]["screening_metrics"] == {
        "task_progress_score": 1.0
    }
    assert result["decision"]["next_action"] == "collect_fixed_ablation_to_20"
