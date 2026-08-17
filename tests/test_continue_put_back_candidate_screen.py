from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "memory-harness" / "scripts" / "continue_put_back_candidate_screen.sh"
)


def _run(tmp_path: Path, *, next_action: str) -> str:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    suite = tmp_path / "suite"
    suite.mkdir()
    aliases = [
        path.stem.removeprefix("fixed_")
        for path in (ROOT / "memory-harness" / "configs" / "architectures").glob(
            "fixed_*.json"
        )
    ]
    (suite / "candidate_suite_manifest.json").write_text(
        json.dumps({"architectures": [{"alias": alias} for alias in aliases]})
        + "\n",
        encoding="utf-8",
    )
    utility = tmp_path / "utility.json"
    utility.write_text(
        json.dumps(
            {
                "schema_version": "memory_harness.utility_decision/v2",
                "evidence_kind": "fixed_ablation",
                "candidate_utility_requirement_met": False,
                "next_action": next_action,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), str(checkpoint), str(utility)],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "PYTHON": sys.executable,
            "MEMORY_CANDIDATE_SUITE": str(suite),
            "RUN_ROOT": str(tmp_path / "runs"),
            "RESULT_ROOT": str(tmp_path / "results"),
            "DATE_TAG": "test",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_positive_fixed_pilot_runs_distinct_suite_screen(tmp_path: Path) -> None:
    output = _run(tmp_path, next_action="collect_shared_episodes_to_50")

    for architecture in (
        "sliding",
        "anchor_sliding",
        "novelty_sliding",
        "dhem_event",
        "kinematic_event",
        "content_recency",
        "semantic_recent_union",
        "uniform_global",
        "recent_global",
        "temporal_multiscale",
        "boundary_chunk",
        "tiered_chunk_mean",
    ):
        assert f"{architecture}_suitev9_screen3_test" in output
    assert "sliding_vs_uniform_global_screen3.json" in output
    assert "anchor_sliding_vs_kinematic_event_screen3.json" in output
    assert output.count("--evidence-kind zero_shot") == 10
    assert "MEMORY_CANDIDATE_SUITE=" in output


def test_nonpositive_fixed_pilot_keeps_candidate_screen_gated(tmp_path: Path) -> None:
    output = _run(tmp_path, next_action="retain_as_inconclusive_diagnostic")

    assert "candidate screen and controller remain gated" in output
    assert "run_fixed_pi05_rmbench.sh" not in output
