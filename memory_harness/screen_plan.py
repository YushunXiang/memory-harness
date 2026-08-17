from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from collections.abc import Collection
from typing import Any

from memory_harness.utility_gate import EVIDENCE_KINDS


SCHEMA_VERSION = "memory_harness.fixed_screen/v1"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclasses.dataclass(frozen=True)
class ScreenComparison:
    reference: str
    candidate: str


@dataclasses.dataclass(frozen=True)
class FixedScreenPlan:
    name: str
    task: str
    num_episodes: int
    seed: int
    policy_seed_base: int
    evidence_kind: str
    comparisons: tuple[ScreenComparison, ...]


def _name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def load_fixed_screen_plan(
    path: pathlib.Path,
    *,
    available_aliases: Collection[str] | None = None,
) -> FixedScreenPlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "name",
        "task",
        "num_episodes",
        "seed",
        "policy_seed_base",
        "evidence_kind",
        "comparisons",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("fixed screen plan keys mismatch")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported fixed screen plan schema")
    if not isinstance(raw["num_episodes"], int) or raw["num_episodes"] <= 0:
        raise ValueError("num_episodes must be a positive integer")
    for key in ("seed", "policy_seed_base"):
        if not isinstance(raw[key], int) or raw[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if raw["evidence_kind"] not in EVIDENCE_KINDS:
        raise ValueError("unknown screen evidence_kind")
    rows = raw["comparisons"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("fixed screen plan must contain comparisons")

    comparisons: list[ScreenComparison] = []
    candidates: set[str] = set()
    aliases = set(available_aliases) if available_aliases is not None else None
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"reference", "candidate"}:
            raise ValueError("screen comparison keys mismatch")
        reference = _name(row["reference"], label="reference")
        candidate = _name(row["candidate"], label="candidate")
        if reference == candidate:
            raise ValueError("screen candidate must differ from its reference")
        if candidate in candidates:
            raise ValueError(f"duplicate screen candidate: {candidate}")
        candidates.add(candidate)
        if aliases is not None:
            missing = {reference, candidate} - aliases
            if missing:
                raise ValueError(f"screen uses unavailable aliases: {sorted(missing)}")
        comparisons.append(ScreenComparison(reference, candidate))

    return FixedScreenPlan(
        name=_name(raw["name"], label="screen name"),
        task=_name(raw["task"], label="task"),
        num_episodes=raw["num_episodes"],
        seed=raw["seed"],
        policy_seed_base=raw["policy_seed_base"],
        evidence_kind=raw["evidence_kind"],
        comparisons=tuple(comparisons),
    )
