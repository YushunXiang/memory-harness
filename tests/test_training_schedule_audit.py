from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "2026-08-16-pi05-put-back-training-schedule-audit.json"
)


def test_training_schedule_audit_separates_budget_and_condition_alignment() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    conditions = audit["completed_conditions"]

    u1200 = [conditions[name] for name in ("full_u1200", "empty_u1200", "native_u1200")]
    assert {condition["total_optimizer_updates"] for condition in u1200} == {1200}
    assert {condition["total_optimizer_examples"] for condition in u1200} == {67200}
    assert conditions["full_u1200"]["precondition_optimizer_updates"] == 200
    assert not conditions["full_u1200"]["condition_from_terminal_initial_params"]
    assert conditions["empty_u1200"]["condition_from_terminal_initial_params"]
    assert conditions["native_u1200"]["condition_from_terminal_initial_params"]
    assert not audit["enforcement"][
        "readiness_condition_schedule_confirmation_expected"
    ]


def test_running_u3000_schedule_is_not_recorded_as_completed_evidence() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    expected = audit["active_full_u3000_expected_schedule"]
    assert expected["status"] == "running_not_yet_evidence"
    assert "full_u3000" not in audit["completed_conditions"]
