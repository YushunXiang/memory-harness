from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class EvolutionTrial:
    """One immutable architecture-search result in a completed search lineage."""

    content_sha256: str
    run_sha256: str
    task_id: str
    generation: int
    success_rate: float
    parent_sha256s: tuple[str, ...] = ()
    mechanism_ids: tuple[str, ...] = ()
    valid: bool = True

    def __post_init__(self) -> None:
        _validate_sha256(self.content_sha256, field="content_sha256")
        _validate_sha256(self.run_sha256, field="run_sha256")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        if not math.isfinite(self.success_rate) or not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be finite and in [0, 1]")
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be boolean")
        if len(self.parent_sha256s) != len(set(self.parent_sha256s)):
            raise ValueError("parent_sha256s cannot contain duplicates")
        for parent_sha256 in self.parent_sha256s:
            _validate_sha256(parent_sha256, field="parent_sha256s item")
        if self.content_sha256 in self.parent_sha256s:
            raise ValueError("a trial cannot be its own parent")
        if len(self.mechanism_ids) != len(set(self.mechanism_ids)):
            raise ValueError("mechanism_ids cannot contain duplicates")
        if any(
            not isinstance(mechanism_id, str) or not mechanism_id.strip()
            for mechanism_id in self.mechanism_ids
        ):
            raise ValueError("mechanism_ids must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class LineagePromotionPolicy:
    """Explicit local thresholds for an EvoMem-inspired evidence gate.

    EvoMem describes a conservative lineage filter but does not publish its
    exact thresholds. Requiring this policy at construction prevents local
    choices from being mistaken for paper-exact defaults.
    """

    minimum_origin_gain: float
    maximum_descendant_downside_rate: float
    minimum_inherited_descendants: int
    minimum_independent_origins: int
    minimum_additional_signals: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_origin_gain) or self.minimum_origin_gain < 0:
            raise ValueError("minimum_origin_gain must be finite and non-negative")
        if not math.isfinite(self.maximum_descendant_downside_rate) or not (
            0.0 <= self.maximum_descendant_downside_rate <= 1.0
        ):
            raise ValueError(
                "maximum_descendant_downside_rate must be finite and in [0, 1]"
            )
        for name, value in (
            ("minimum_inherited_descendants", self.minimum_inherited_descendants),
            ("minimum_independent_origins", self.minimum_independent_origins),
            ("minimum_additional_signals", self.minimum_additional_signals),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_additional_signals > 3:
            raise ValueError("minimum_additional_signals cannot exceed three")


@dataclass(frozen=True, slots=True)
class TrialDelta:
    trial_sha256: str
    strongest_parent_sha256: str
    success_rate_delta: float


@dataclass(frozen=True, slots=True)
class LineagePromotionDecision:
    """Auditable observational evidence for promoting one research lesson."""

    mechanism_id: str
    run_sha256: str
    task_id: str
    origin_sha256: str
    strongest_parent_sha256: str
    origin_gain: float
    sibling_sha256s: tuple[str, ...]
    best_sibling_margin: float | None
    inherited_descendant_sha256s: tuple[str, ...]
    descendant_deltas: tuple[TrialDelta, ...]
    descendant_downside_rate: float
    independent_origin_sha256s: tuple[str, ...]
    additional_signals: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    promote: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "run_sha256": self.run_sha256,
            "task_id": self.task_id,
            "origin_sha256": self.origin_sha256,
            "strongest_parent_sha256": self.strongest_parent_sha256,
            "origin_gain": self.origin_gain,
            "sibling_sha256s": list(self.sibling_sha256s),
            "best_sibling_margin": self.best_sibling_margin,
            "inherited_descendant_sha256s": list(
                self.inherited_descendant_sha256s
            ),
            "descendant_deltas": [
                {
                    "trial_sha256": delta.trial_sha256,
                    "strongest_parent_sha256": delta.strongest_parent_sha256,
                    "success_rate_delta": delta.success_rate_delta,
                }
                for delta in self.descendant_deltas
            ],
            "descendant_downside_rate": self.descendant_downside_rate,
            "independent_origin_sha256s": list(self.independent_origin_sha256s),
            "additional_signals": list(self.additional_signals),
            "rejection_reasons": list(self.rejection_reasons),
            "promote": self.promote,
        }


@dataclass(frozen=True, slots=True)
class LineageEvidencePromoter:
    """Pluggable Research-Agent writer gate over an evaluated lineage."""

    policy: LineagePromotionPolicy

    def evaluate(
        self,
        trials: Sequence[EvolutionTrial],
        *,
        origin_sha256: str,
        mechanism_id: str,
    ) -> LineagePromotionDecision:
        return evaluate_lineage_promotion(
            trials,
            origin_sha256=origin_sha256,
            mechanism_id=mechanism_id,
            policy=self.policy,
        )


def evaluate_lineage_promotion(
    trials: Sequence[EvolutionTrial],
    *,
    origin_sha256: str,
    mechanism_id: str,
    policy: LineagePromotionPolicy,
) -> LineagePromotionDecision:
    """Assess whether one observed mutation lesson merits research-memory storage."""

    if not isinstance(mechanism_id, str) or not mechanism_id.strip():
        raise ValueError("mechanism_id must be a non-empty string")
    _validate_sha256(origin_sha256, field="origin_sha256")
    by_sha = _validate_lineage(trials)
    try:
        origin = by_sha[origin_sha256]
    except KeyError as exc:
        raise ValueError("origin_sha256 is not present in the lineage") from exc
    if not origin.valid:
        raise ValueError("origin trial must be valid")
    if not origin.parent_sha256s:
        raise ValueError("origin trial must be a non-root mutation")
    if mechanism_id not in origin.mechanism_ids:
        raise ValueError("origin trial does not contain mechanism_id")
    if any(
        mechanism_id in by_sha[parent].mechanism_ids
        for parent in origin.parent_sha256s
    ):
        raise ValueError("mechanism_id was already present in an origin parent")

    valid_parents = tuple(
        by_sha[parent] for parent in origin.parent_sha256s if by_sha[parent].valid
    )
    if not valid_parents:
        raise ValueError("origin trial has no valid parent comparator")
    strongest_parent = _strongest(valid_parents)
    origin_gain = origin.success_rate - strongest_parent.success_rate

    siblings = tuple(
        sorted(
            (
                trial
                for trial in trials
                if trial.content_sha256 != origin.content_sha256
                and trial.valid
                and trial.parent_sha256s == origin.parent_sha256s
                and mechanism_id not in trial.mechanism_ids
            ),
            key=lambda trial: trial.content_sha256,
        )
    )
    best_sibling_margin = (
        origin.success_rate - max(sibling.success_rate for sibling in siblings)
        if siblings
        else None
    )

    inherited_descendants = _inherited_descendants(
        trials,
        origin_sha256=origin_sha256,
        mechanism_id=mechanism_id,
    )
    descendant_deltas = tuple(
        _delta_from_strongest_parent(descendant, by_sha)
        for descendant in inherited_descendants
    )
    downside_count = sum(
        delta.success_rate_delta < 0.0 for delta in descendant_deltas
    )
    downside_rate = downside_count / len(descendant_deltas) if descendant_deltas else 1.0

    independent_origins = tuple(
        sorted(
            trial.content_sha256
            for trial in trials
            if trial.valid
            and trial.parent_sha256s
            and mechanism_id in trial.mechanism_ids
            and all(
                mechanism_id not in by_sha[parent].mechanism_ids
                for parent in trial.parent_sha256s
            )
        )
    )

    signals: list[str] = []
    if (
        best_sibling_margin is not None
        and best_sibling_margin > policy.minimum_origin_gain
    ):
        signals.append("better_than_same_parent_siblings")
    if len(inherited_descendants) >= policy.minimum_inherited_descendants:
        signals.append("inherited_descendant_spread")
    if len(independent_origins) >= policy.minimum_independent_origins:
        signals.append("independent_recurrence")

    rejection_reasons: list[str] = []
    if origin_gain <= policy.minimum_origin_gain:
        rejection_reasons.append("origin_gain_below_threshold")
    if len(inherited_descendants) < policy.minimum_inherited_descendants:
        rejection_reasons.append("insufficient_inherited_descendants")
    if downside_rate > policy.maximum_descendant_downside_rate:
        rejection_reasons.append("descendant_downside_rate_above_threshold")
    if len(signals) < policy.minimum_additional_signals:
        rejection_reasons.append("insufficient_additional_signals")

    return LineagePromotionDecision(
        mechanism_id=mechanism_id,
        run_sha256=origin.run_sha256,
        task_id=origin.task_id,
        origin_sha256=origin.content_sha256,
        strongest_parent_sha256=strongest_parent.content_sha256,
        origin_gain=origin_gain,
        sibling_sha256s=tuple(sibling.content_sha256 for sibling in siblings),
        best_sibling_margin=best_sibling_margin,
        inherited_descendant_sha256s=tuple(
            descendant.content_sha256 for descendant in inherited_descendants
        ),
        descendant_deltas=descendant_deltas,
        descendant_downside_rate=downside_rate,
        independent_origin_sha256s=independent_origins,
        additional_signals=tuple(signals),
        rejection_reasons=tuple(rejection_reasons),
        promote=not rejection_reasons,
    )


def _validate_lineage(
    trials: Sequence[EvolutionTrial],
) -> dict[str, EvolutionTrial]:
    if not trials:
        raise ValueError("lineage must contain at least one trial")
    by_sha = {trial.content_sha256: trial for trial in trials}
    if len(by_sha) != len(trials):
        raise ValueError("lineage contains duplicate trial content")
    run_sha256s = {trial.run_sha256 for trial in trials}
    task_ids = {trial.task_id for trial in trials}
    if len(run_sha256s) != 1:
        raise ValueError("lineage must contain exactly one run_sha256")
    if len(task_ids) != 1:
        raise ValueError("lineage must contain exactly one task_id")
    for trial in trials:
        for parent_sha256 in trial.parent_sha256s:
            try:
                parent = by_sha[parent_sha256]
            except KeyError as exc:
                raise ValueError(
                    f"lineage parent is missing: {parent_sha256}"
                ) from exc
            if parent.generation >= trial.generation:
                raise ValueError("parent generation must precede child generation")
    return by_sha


def _inherited_descendants(
    trials: Sequence[EvolutionTrial],
    *,
    origin_sha256: str,
    mechanism_id: str,
) -> tuple[EvolutionTrial, ...]:
    children: dict[str, list[EvolutionTrial]] = defaultdict(list)
    for trial in trials:
        for parent_sha256 in trial.parent_sha256s:
            children[parent_sha256].append(trial)
    queue = deque([origin_sha256])
    seen = {origin_sha256}
    inherited: list[EvolutionTrial] = []
    while queue:
        parent_sha256 = queue.popleft()
        for child in sorted(
            children[parent_sha256],
            key=lambda trial: (trial.generation, trial.content_sha256),
        ):
            if child.content_sha256 in seen:
                continue
            seen.add(child.content_sha256)
            if not child.valid or mechanism_id not in child.mechanism_ids:
                continue
            queue.append(child.content_sha256)
            inherited.append(child)
    return tuple(
        sorted(inherited, key=lambda trial: (trial.generation, trial.content_sha256))
    )


def _delta_from_strongest_parent(
    trial: EvolutionTrial, by_sha: dict[str, EvolutionTrial]
) -> TrialDelta:
    valid_parents = tuple(
        by_sha[parent] for parent in trial.parent_sha256s if by_sha[parent].valid
    )
    if not valid_parents:
        raise ValueError(
            f"inherited descendant has no valid parent: {trial.content_sha256}"
        )
    strongest_parent = _strongest(valid_parents)
    return TrialDelta(
        trial_sha256=trial.content_sha256,
        strongest_parent_sha256=strongest_parent.content_sha256,
        success_rate_delta=trial.success_rate - strongest_parent.success_rate,
    )


def _strongest(trials: Sequence[EvolutionTrial]) -> EvolutionTrial:
    return min(
        trials,
        key=lambda trial: (-trial.success_rate, trial.content_sha256),
    )


def _validate_sha256(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
