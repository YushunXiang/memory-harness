from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-bpp-paper-release-audit.json"
SOURCE_CATALOG = PROJECT_ROOT / "configs" / "source_audited_candidates.json"


def test_bpp_reuses_keyframe_payload_and_is_not_executable() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    decision = audit["decision"]

    assert decision["worth_retaining"]
    assert decision["implementation_status"] == "paper_contract_only"
    assert decision["reuse_payload_family"] == "task_phase_keyframe_history"
    assert not decision["add_payload_family"]
    assert not decision["add_to_source_audited_catalog"]
    assert not decision["add_to_executable_suite"]
    assert not audit["release_status"]["official_code_repository_found"]

    catalog = json.loads(SOURCE_CATALOG.read_text(encoding="utf-8"))
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }


def test_bpp_retains_latency_parity_not_a_runtime_only_delay() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    operators = {
        item["name"]: item for item in audit["nonduplicate_typed_contracts"]
    }

    assert "binary_semantic_event_rising_edge_writer" in operators
    assert "detector_latency_aligned_keyframe_availability_mask" in operators
    assert "shared_train_deploy_semantic_event_detector_contract" in operators
    assert "training_and_rollout_share_the_detector_contract" in audit["decision"][
        "entry_gate"
    ]
    assert any(
        "runtime-only" in boundary
        for boundary in audit["risks_and_boundaries"]
    )


def test_bpp_reported_results_and_scope_are_preserved() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    evidence = audit["reported_evidence"]
    average = evidence["real_robot_success_or_progress_percent"]["average"]

    assert average["bpp"] == 53.6
    assert average["past_token_prediction"] == 31.8
    assert average["bpp_absolute_gain_over_best_comparison_pp"] == 21.8
    assert evidence["mug_replacement_keyframe_quality_ablation_percent"][
        "oracle_initial_observation_context"
    ] == 70.0
    assert "does not isolate" in evidence["scope_note"]
