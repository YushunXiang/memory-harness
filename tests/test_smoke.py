from __future__ import annotations

import json
from pathlib import Path

from memory_harness.smoke import run_smoke


ROOT = Path(__file__).resolve().parents[1]


def test_arbitrary_candidate_smoke_binds_config_and_checks_reset(tmp_path) -> None:
    config = ROOT / "configs" / "fixed_semantic_recent_union.json"

    summary = run_smoke(config_path=config, output_dir=tmp_path / "candidate")

    assert summary["schema_version"] == "memory_harness.candidate_smoke/v1"
    assert summary["program"] == "semantic_recent_union"
    assert summary["steps"] == 4
    assert summary["episode_reset_isolated"] is True
    assert summary["max_stored_item_count"] == 4
    assert summary["event_counts"]["SELECT"] == 4
    assert json.loads(
        (tmp_path / "candidate" / "summary.json").read_text(encoding="utf-8")
    ) == summary


def test_kinematic_candidate_smoke_reaches_delayed_writer_horizon(tmp_path) -> None:
    config = ROOT / "configs" / "fixed_kinematic_event.json"

    summary = run_smoke(config_path=config, output_dir=tmp_path / "candidate")

    assert summary["steps"] == 72
    assert summary["episode_reset_isolated"] is True
    assert summary["event_counts"]["WRITE_DECISION"] == 144
    assert summary["write_counts_by_path"]["event"] >= 1


def test_tiered_candidate_smoke_crosses_both_capacity_boundaries(tmp_path) -> None:
    config = ROOT / "configs" / "fixed_tiered_chunk_mean.json"

    summary = run_smoke(config_path=config, output_dir=tmp_path / "candidate")

    assert summary["steps"] == 31
    assert summary["episode_reset_isolated"] is True
    assert summary["maintenance_counts"]["migrate_chunk"] == 9
    assert summary["maintenance_counts"]["consolidate_long_term_adjacent"] == 1


def test_smoke_verifies_success_commit_and_failure_discard(tmp_path) -> None:
    config = ROOT / "configs" / "fixed_verified_success_latent.json"

    summary = run_smoke(config_path=config, output_dir=tmp_path / "candidate")

    assert summary["persistent_success_memory"] is True
    assert summary["episode_reset_isolated"] is False
    assert summary["committed_item_count"] == 4
    assert summary["committed_retrieved_item_ids"] == [
        f"smoke-episode:success:{index}" for index in range(4)
    ]
    assert summary["failed_episode_excluded"] is True
