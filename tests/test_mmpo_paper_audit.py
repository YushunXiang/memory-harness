from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-mmpo-paper-release-audit.json"


def test_mmpo_is_scoped_to_textual_memory_and_not_executable() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    decision = audit["decision"]

    assert decision["worth_retaining"]
    assert decision["implementation_status"] == "paper_contract_only"
    assert decision["applicable_candidate_scope"] == "recursive_textual_summary_memory_only"
    assert not decision["add_payload_family"]
    assert not decision["add_to_source_audited_catalog"]
    assert not decision["add_to_executable_suite"]
    assert not audit["release_status"]["official_code_repository_found"]


def test_mmpo_contract_preserves_outcome_anchor_and_premature_confidence_control() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    evidence = audit["reported_evidence"]
    operators = {item["name"] for item in audit["nonduplicate_typed_contracts"]}

    assert evidence["anchor_probe_ablation_at_56k_accuracy_percent"][
        "direct_answer"
    ] < evidence["anchor_probe_ablation_at_56k_accuracy_percent"]["outcome_only"]
    assert "verified_outcome_anchored_belief_entropy_reward" in operators
    assert "memory_conditioned_anchor_response_entropy_evaluator" in operators
    assert "verified_terminal_outcome_remains_in_the_reward" in audit["decision"][
        "entry_gate"
    ]
