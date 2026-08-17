from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np
import numpy.typing as npt


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    """Immutable, content-addressed result used by an architecture-search sampler."""

    content_sha256: str
    success_rate: float
    visit_count: int = 0

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.content_sha256) is None:
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if not math.isfinite(self.success_rate) or not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be finite and in [0, 1]")
        if (
            isinstance(self.visit_count, bool)
            or not isinstance(self.visit_count, int)
            or self.visit_count < 0
        ):
            raise ValueError("visit_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ParetoRecord:
    """Budget-aware result for deterministic architecture survivor selection."""

    content_sha256: str
    success_rate: float
    cost_metric: str
    mean_cost: float
    mean_latency_seconds: float

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.content_sha256) is None:
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if not math.isfinite(self.success_rate) or not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be finite and in [0, 1]")
        if not isinstance(self.cost_metric, str) or not self.cost_metric.strip():
            raise ValueError("cost_metric must be a non-empty string")
        if not math.isfinite(self.mean_cost) or self.mean_cost < 0.0:
            raise ValueError("mean_cost must be finite and non-negative")
        if (
            not math.isfinite(self.mean_latency_seconds)
            or self.mean_latency_seconds < 0.0
        ):
            raise ValueError("mean_latency_seconds must be finite and non-negative")


def alma_sampling_scores(
    records: Sequence[ArchiveRecord],
    *,
    no_memory_success_rate: float,
    visit_penalty: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Reproduce ALMA's performance-minus-log-visitation parent score."""

    _validate_inputs(
        records,
        no_memory_success_rate=no_memory_success_rate,
        visit_penalty=visit_penalty,
        temperature=1.0,
    )
    success = np.asarray([record.success_rate for record in records], dtype=np.float64)
    visits = np.asarray([record.visit_count for record in records], dtype=np.float64)
    normalized = 1.0 / (1.0 + np.exp(-(success - no_memory_success_rate)))
    return normalized - visit_penalty * np.log1p(visits)


def alma_sampling_probabilities(
    records: Sequence[ArchiveRecord],
    *,
    no_memory_success_rate: float,
    visit_penalty: float = 0.5,
    temperature: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Return ALMA-compatible non-greedy archive sampling probabilities."""

    _validate_inputs(
        records,
        no_memory_success_rate=no_memory_success_rate,
        visit_penalty=visit_penalty,
        temperature=temperature,
    )
    scores = alma_sampling_scores(
        records,
        no_memory_success_rate=no_memory_success_rate,
        visit_penalty=visit_penalty,
    )
    logits = scores / temperature
    weights = np.exp(logits - np.max(logits))
    probabilities = weights / np.sum(weights)
    if not np.all(probabilities > 0.0):
        raise ValueError("sampling probabilities underflowed to zero")
    return probabilities


def sample_alma_parents(
    records: Sequence[ArchiveRecord],
    *,
    count: int,
    seed: int,
    no_memory_success_rate: float,
    visit_penalty: float = 0.5,
    temperature: float = 0.5,
) -> tuple[ArchiveRecord, ...]:
    """Sample content-distinct archive parents without replacement."""

    probabilities = alma_sampling_probabilities(
        records,
        no_memory_success_rate=no_memory_success_rate,
        visit_penalty=visit_penalty,
        temperature=temperature,
    )
    if isinstance(count, bool) or not 1 <= count <= len(records):
        raise ValueError("count must be between one and the archive size")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=count, replace=False, p=probabilities)
    return tuple(records[int(index)] for index in indices)


def pareto_ranks(records: Sequence[ParetoRecord]) -> tuple[int, ...]:
    """Rank candidates by success (max), cost (min), and latency (min)."""

    _validate_pareto_records(records)
    remaining = set(range(len(records)))
    ranks = [0] * len(records)
    rank = 1
    while remaining:
        front = {
            candidate
            for candidate in remaining
            if not any(
                _dominates(records[other], records[candidate])
                for other in remaining
                if other != candidate
            )
        }
        if not front:
            raise RuntimeError("Pareto ranking failed to produce a non-empty front")
        for candidate in front:
            ranks[candidate] = rank
        remaining -= front
        rank += 1
    return tuple(ranks)


def select_pareto_survivors(
    records: Sequence[ParetoRecord], *, count: int
) -> tuple[ParetoRecord, ...]:
    """Select deterministic MemEvolve-style budget-aware search survivors."""

    ranks = pareto_ranks(records)
    if isinstance(count, bool) or not 1 <= count <= len(records):
        raise ValueError("count must be between one and the archive size")
    order = sorted(
        range(len(records)),
        key=lambda index: (
            ranks[index],
            -records[index].success_rate,
            records[index].mean_cost,
            records[index].mean_latency_seconds,
            records[index].content_sha256,
        ),
    )
    return tuple(records[index] for index in order[:count])


def _validate_inputs(
    records: Sequence[ArchiveRecord],
    *,
    no_memory_success_rate: float,
    visit_penalty: float,
    temperature: float,
) -> None:
    if not records:
        raise ValueError("archive must contain at least one record")
    hashes = [record.content_sha256 for record in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("archive contains duplicate candidate content")
    if not math.isfinite(no_memory_success_rate) or not (
        0.0 <= no_memory_success_rate <= 1.0
    ):
        raise ValueError("no_memory_success_rate must be finite and in [0, 1]")
    if not math.isfinite(visit_penalty) or visit_penalty < 0.0:
        raise ValueError("visit_penalty must be finite and non-negative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")


def _validate_pareto_records(records: Sequence[ParetoRecord]) -> None:
    if not records:
        raise ValueError("archive must contain at least one record")
    hashes = [record.content_sha256 for record in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("archive contains duplicate candidate content")
    cost_metrics = {record.cost_metric for record in records}
    if len(cost_metrics) != 1:
        raise ValueError("Pareto records must use one declared cost_metric")


def _dominates(left: ParetoRecord, right: ParetoRecord) -> bool:
    no_worse = (
        left.success_rate >= right.success_rate
        and left.mean_cost <= right.mean_cost
        and left.mean_latency_seconds <= right.mean_latency_seconds
    )
    strictly_better = (
        left.success_rate > right.success_rate
        or left.mean_cost < right.mean_cost
        or left.mean_latency_seconds < right.mean_latency_seconds
    )
    return no_worse and strictly_better
