from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_harness.screen_plan import load_fixed_screen_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "screens" / "put_back_fixed_v9_screen3.json"


def test_put_back_screen_plan_is_typed_and_nonduplicated() -> None:
    aliases = {
        path.stem.removeprefix("fixed_")
        for path in (ROOT / "configs" / "architectures").glob("fixed_*.json")
    }
    plan = load_fixed_screen_plan(PLAN, available_aliases=aliases)

    assert plan.task == "put_back_block"
    assert plan.num_episodes == 3
    assert len(plan.comparisons) == 10
    assert len({row.candidate for row in plan.comparisons}) == 10
    assert next(
        row.reference
        for row in plan.comparisons
        if row.candidate == "kinematic_event"
    ) == "anchor_sliding"


def test_screen_plan_rejects_alias_absent_from_suite(tmp_path: Path) -> None:
    raw = json.loads(PLAN.read_text(encoding="utf-8"))
    raw["comparisons"][0]["candidate"] = "missing"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unavailable aliases"):
        load_fixed_screen_plan(path, available_aliases={"sliding"})
