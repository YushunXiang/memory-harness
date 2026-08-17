from __future__ import annotations

import numpy as np
import pytest

from memory_harness.search_archive import ArchiveRecord
from memory_harness.search_archive import ParetoRecord
from memory_harness.search_archive import alma_sampling_probabilities
from memory_harness.search_archive import alma_sampling_scores
from memory_harness.search_archive import pareto_ranks
from memory_harness.search_archive import sample_alma_parents
from memory_harness.search_archive import select_pareto_survivors


def _record(digit: str, success: float, visits: int = 0) -> ArchiveRecord:
    return ArchiveRecord(digit * 64, success, visits)


def test_alma_sampler_balances_success_and_visitation_without_greedy_collapse() -> None:
    records = (
        _record("1", 0.8),
        _record("2", 0.6),
        _record("3", 0.8, visits=4),
    )

    scores = alma_sampling_scores(records, no_memory_success_rate=0.2)
    probabilities = alma_sampling_probabilities(records, no_memory_success_rate=0.2)

    assert scores[0] > scores[1] > scores[2]
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert np.all(probabilities > 0.0)
    assert np.sum(probabilities) == pytest.approx(1.0)


def test_alma_sampler_is_seeded_without_mutating_numpy_global_rng() -> None:
    records = tuple(_record(str(index), 0.1 * index) for index in range(1, 6))
    np.random.seed(91)
    expected_global_draw = np.random.random()
    np.random.seed(91)

    first = sample_alma_parents(records, count=3, seed=42, no_memory_success_rate=0.1)
    second = sample_alma_parents(records, count=3, seed=42, no_memory_success_rate=0.1)

    assert [record.content_sha256 for record in first] == [
        record.content_sha256 for record in second
    ]
    assert len({record.content_sha256 for record in first}) == 3
    assert np.random.random() == expected_global_draw


def test_alma_archive_rejects_non_content_addressed_or_duplicate_candidates() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        ArchiveRecord("random-id", 0.5)
    with pytest.raises(ValueError, match="success_rate"):
        _record("1", 1.1)
    duplicate = _record("1", 0.5)
    with pytest.raises(ValueError, match="duplicate candidate content"):
        alma_sampling_probabilities((duplicate, duplicate), no_memory_success_rate=0.2)


def test_alma_sampler_rejects_invalid_budget_and_hyperparameters() -> None:
    records = (_record("1", 0.5), _record("2", 0.4))
    with pytest.raises(ValueError, match="archive size"):
        sample_alma_parents(records, count=3, seed=0, no_memory_success_rate=0.2)
    with pytest.raises(ValueError, match="temperature"):
        alma_sampling_probabilities(
            records, no_memory_success_rate=0.2, temperature=0.0
        )


def _pareto_record(
    digit: str,
    success: float,
    cost: float,
    latency: float,
    *,
    cost_metric: str = "mean_api_cost_usd",
) -> ParetoRecord:
    return ParetoRecord(digit * 64, success, cost_metric, cost, latency)


def test_pareto_survivors_preserve_real_tradeoffs_and_rank_dominated_candidates() -> (
    None
):
    records = (
        _pareto_record("1", 0.8, 2.0, 2.0),
        _pareto_record("2", 0.7, 1.0, 1.0),
        _pareto_record("3", 0.6, 2.0, 2.0),
        _pareto_record("4", 0.8, 2.0, 3.0),
    )

    assert pareto_ranks(records) == (1, 1, 2, 2)
    assert select_pareto_survivors(records, count=2) == records[:2]


def test_pareto_survivor_ties_are_content_deterministic() -> None:
    records = (
        _pareto_record("2", 0.5, 1.0, 1.0),
        _pareto_record("1", 0.5, 1.0, 1.0),
    )

    selected = select_pareto_survivors(records, count=2)

    assert [record.content_sha256 for record in selected] == ["1" * 64, "2" * 64]


def test_pareto_archive_rejects_invalid_metrics_duplicates_and_budget() -> None:
    with pytest.raises(ValueError, match="mean_cost"):
        _pareto_record("1", 0.5, -1.0, 1.0)
    records = (
        _pareto_record("1", 0.5, 1.0, 1.0),
        _pareto_record("2", 0.4, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="duplicate candidate content"):
        pareto_ranks((records[0], records[0]))
    with pytest.raises(ValueError, match="one declared cost_metric"):
        pareto_ranks(
            (
                records[0],
                _pareto_record("3", 0.4, 1.0, 1.0, cost_metric="mean_memory_tokens"),
            )
        )
    with pytest.raises(ValueError, match="archive size"):
        select_pareto_survivors(records, count=0)
