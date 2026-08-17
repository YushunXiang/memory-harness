from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_harness.research_memory import EvolutionTrial
from memory_harness.research_memory import LineageEvidencePromoter
from memory_harness.research_memory import LineagePromotionPolicy


RUN = "f" * 64
TASK = "put_back_block"
MECHANISM = "anchor_plus_sliding"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _trial(
    digit: str,
    success: float,
    *,
    generation: int,
    parents: tuple[str, ...] = (),
    mechanisms: tuple[str, ...] = (),
    run: str = RUN,
    task: str = TASK,
    valid: bool = True,
) -> EvolutionTrial:
    return EvolutionTrial(
        content_sha256=digit * 64,
        run_sha256=run,
        task_id=task,
        generation=generation,
        success_rate=success,
        parent_sha256s=parents,
        mechanism_ids=mechanisms,
        valid=valid,
    )


def _policy(**overrides: object) -> LineagePromotionPolicy:
    values: dict[str, object] = {
        "minimum_origin_gain": 0.05,
        "maximum_descendant_downside_rate": 0.25,
        "minimum_inherited_descendants": 2,
        "minimum_independent_origins": 2,
        "minimum_additional_signals": 2,
    }
    values.update(overrides)
    return LineagePromotionPolicy(**values)  # type: ignore[arg-type]


def test_lineage_promoter_uses_origin_siblings_descendants_and_recurrence() -> None:
    trials = (
        _trial("1", 0.20, generation=0),
        _trial("2", 0.50, generation=1, parents=("1" * 64,), mechanisms=(MECHANISM,)),
        _trial("3", 0.25, generation=1, parents=("1" * 64,)),
        _trial("4", 0.60, generation=2, parents=("2" * 64,), mechanisms=(MECHANISM,)),
        _trial("5", 0.65, generation=3, parents=("4" * 64,), mechanisms=(MECHANISM,)),
        _trial("6", 0.55, generation=2, parents=("3" * 64,), mechanisms=(MECHANISM,)),
    )

    decision = LineageEvidencePromoter(_policy()).evaluate(
        trials,
        origin_sha256="2" * 64,
        mechanism_id=MECHANISM,
    )

    assert decision.promote is True
    assert decision.origin_gain == pytest.approx(0.30)
    assert decision.best_sibling_margin == pytest.approx(0.25)
    assert decision.inherited_descendant_sha256s == ("4" * 64, "5" * 64)
    assert decision.descendant_downside_rate == 0.0
    assert decision.independent_origin_sha256s == ("2" * 64, "6" * 64)
    assert decision.additional_signals == (
        "better_than_same_parent_siblings",
        "inherited_descendant_spread",
        "independent_recurrence",
    )
    assert decision.to_json()["promote"] is True


def test_lineage_promoter_rejects_high_descendant_downside() -> None:
    trials = (
        _trial("1", 0.20, generation=0),
        _trial("2", 0.60, generation=1, parents=("1" * 64,), mechanisms=(MECHANISM,)),
        _trial("3", 0.10, generation=2, parents=("2" * 64,), mechanisms=(MECHANISM,)),
        _trial("4", 0.05, generation=2, parents=("2" * 64,), mechanisms=(MECHANISM,)),
    )

    decision = LineageEvidencePromoter(
        _policy(minimum_additional_signals=1, minimum_independent_origins=3)
    ).evaluate(trials, origin_sha256="2" * 64, mechanism_id=MECHANISM)

    assert decision.promote is False
    assert decision.descendant_downside_rate == 1.0
    assert "descendant_downside_rate_above_threshold" in decision.rejection_reasons


def test_lineage_promoter_does_not_cross_a_dropped_mechanism() -> None:
    trials = (
        _trial("1", 0.20, generation=0),
        _trial("2", 0.50, generation=1, parents=("1" * 64,), mechanisms=(MECHANISM,)),
        _trial("3", 0.55, generation=2, parents=("2" * 64,)),
        _trial("4", 0.70, generation=3, parents=("3" * 64,), mechanisms=(MECHANISM,)),
    )

    decision = LineageEvidencePromoter(
        _policy(
            minimum_inherited_descendants=1,
            minimum_additional_signals=1,
            minimum_independent_origins=2,
        )
    ).evaluate(trials, origin_sha256="2" * 64, mechanism_id=MECHANISM)

    assert decision.inherited_descendant_sha256s == ()
    assert decision.independent_origin_sha256s == ("2" * 64, "4" * 64)
    assert decision.promote is False
    assert "insufficient_inherited_descendants" in decision.rejection_reasons


def test_lineage_promoter_rejects_non_origin_and_missing_comparators() -> None:
    trials = (
        _trial("1", 0.20, generation=0, mechanisms=(MECHANISM,)),
        _trial("2", 0.50, generation=1, parents=("1" * 64,), mechanisms=(MECHANISM,)),
    )
    promoter = LineageEvidencePromoter(_policy())

    with pytest.raises(ValueError, match="already present"):
        promoter.evaluate(trials, origin_sha256="2" * 64, mechanism_id=MECHANISM)
    with pytest.raises(ValueError, match="non-root"):
        promoter.evaluate(trials, origin_sha256="1" * 64, mechanism_id=MECHANISM)


def test_lineage_validation_rejects_mixed_or_broken_lineages() -> None:
    promoter = LineageEvidencePromoter(_policy())
    root = _trial("1", 0.20, generation=0)
    origin = _trial(
        "2",
        0.50,
        generation=1,
        parents=("1" * 64,),
        mechanisms=(MECHANISM,),
    )
    mixed_run = _trial(
        "3",
        0.60,
        generation=2,
        parents=("2" * 64,),
        mechanisms=(MECHANISM,),
        run="e" * 64,
    )
    with pytest.raises(ValueError, match="one run_sha256"):
        promoter.evaluate(
            (root, origin, mixed_run),
            origin_sha256="2" * 64,
            mechanism_id=MECHANISM,
        )

    missing_parent = _trial(
        "4",
        0.50,
        generation=1,
        parents=("9" * 64,),
        mechanisms=(MECHANISM,),
    )
    with pytest.raises(ValueError, match="parent is missing"):
        promoter.evaluate(
            (root, missing_parent),
            origin_sha256="4" * 64,
            mechanism_id=MECHANISM,
        )


def test_lineage_policy_and_trial_inputs_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot exceed three"):
        _policy(minimum_additional_signals=4)
    with pytest.raises(ValueError, match="positive integer"):
        _policy(minimum_inherited_descendants=0)
    with pytest.raises(ValueError, match="success_rate"):
        _trial("1", 1.1, generation=0)
    with pytest.raises(ValueError, match="duplicates"):
        _trial(
            "1",
            0.5,
            generation=1,
            parents=("2" * 64, "2" * 64),
        )


def test_evomem_audit_adds_only_the_nonduplicate_research_writer() -> None:
    audit = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "2026-08-16-evomem-paper-release-audit.json"
        ).read_text(encoding="utf-8")
    )

    assert audit["overlap_analysis"]["nonduplicate_mechanism"] == (
        "lineage_evidence_research_memory_promotion_gate"
    )
    assert audit["local_implementation"]["plugin"] == "LineageEvidencePromoter"
    assert audit["public_release_audit"]["arxiv_source_contains_evomem_runtime"] is False
    assert audit["decision"]["add_policy_memory_payload"] is False
    assert audit["decision"]["add_to_source_audited_candidate_catalog"] is False
    assert audit["decision"]["add_to_fixed_executable_suite"] is False


def test_evolvemem_is_a_search_baseline_without_a_duplicate_plugin() -> None:
    audit = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "2026-08-16-evolvemem-source-audit.json"
        ).read_text(encoding="utf-8")
    )

    findings = " ".join(item["finding"] for item in audit["source_findings"])
    decision = audit["decision"]
    assert "maturation_round=5" in findings
    assert "meta_proposals.jsonl" in findings
    assert audit["overlap_analysis"]["nonduplicate_policy_memory_mechanism"] is None
    assert audit["overlap_analysis"]["nonduplicate_search_runtime_mechanism"] is None
    assert decision["worth_retaining"] is True
    assert decision["add_policy_memory_payload"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_fixed_executable_suite"] is False
    assert decision["add_new_search_plugin"] is False
