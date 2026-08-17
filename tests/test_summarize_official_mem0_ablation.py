from __future__ import annotations

import json

import pytest

from memory_harness.summarize_official_mem0_ablation import summarize


def write_result(root, condition, successes, rate, *, seeds=None) -> None:
    target = root / condition / "eval.log"
    target.parent.mkdir(parents=True)
    payload = {
        "num_episodes": 10,
        "successes": successes,
        "success_rate": rate,
        "mean_reward": 1.5,
        "simulator_seeds": seeds or list(range(100000, 100010)),
    }
    target.write_text(
        "setup\nRMBENCH_LIMITED_EVAL_RESULT " + json.dumps(payload) + "\ndone\n",
        encoding="utf-8",
    )
    official_log = (
        root
        / condition
        / "official_results"
        / "put_back_block"
        / "Mem-0"
        / "demo_clean"
        / f"m1mix_reproduced_{condition}"
        / "timestamp"
        / "eval_log.txt"
    )
    official_log.parent.mkdir(parents=True)
    actual_seeds = seeds or list(range(100000, 100010))
    episode_lines = [
        f"episode_id={index}, seed={seed}, "
        f"result={'Success' if index < successes else 'Fail'}"
        for index, seed in enumerate(actual_seeds)
    ]
    official_log.write_text("header\n" + "\n".join(episode_lines) + "\n", encoding="utf-8")


def test_summarizes_shared_checkpoint_interventions(tmp_path) -> None:
    write_result(tmp_path, "full", 9, 0.9)
    write_result(tmp_path, "without_anchor", 4, 0.4)
    write_result(tmp_path, "without_sliding", 7, 0.7)

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"released checkpoint")
    report = summarize(
        tmp_path,
        checkpoint=checkpoint,
        task="put_back_block",
        seed_start=100000,
    )

    assert report["protocol"] == "released_mem0_m1mix_shared_checkpoint_inference_intervention"
    assert report["schema_version"].endswith("/v2")
    assert report["task"] == "put_back_block"
    assert report["simulator_seed_start"] == 100000
    assert report["checkpoint"]["size_bytes"] == len(b"released checkpoint")
    assert len(report["checkpoint"]["sha256"]) == 64
    assert report["num_episodes_per_condition"] == 10
    assert report["simulator_seeds"] == list(range(100000, 100010))
    assert "resets NumPy and Torch RNG" in report["policy_rng_protocol"]
    assert report["success_rate_delta_vs_full"] == pytest.approx(
        {"without_anchor": -0.5, "without_sliding": -0.2}
    )
    assert report["paired_success_transitions_vs_full"] == {
        "without_anchor": {
            "both_success": 4,
            "full_only_success": 5,
            "ablation_only_success": 0,
            "both_fail": 1,
        },
        "without_sliding": {
            "both_success": 7,
            "full_only_success": 2,
            "ablation_only_success": 0,
            "both_fail": 1,
        },
    }
    assert "not an exact replay" in report["scope_note"]


def test_rejects_missing_result_marker(tmp_path) -> None:
    for condition in ("full", "without_anchor", "without_sliding"):
        target = tmp_path / condition / "eval.log"
        target.parent.mkdir(parents=True)
        target.write_text("no result\n", encoding="utf-8")

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"released checkpoint")
    with pytest.raises(ValueError, match="exactly one"):
        summarize(
            tmp_path,
            checkpoint=checkpoint,
            task="put_back_block",
            seed_start=100000,
        )


def test_rejects_unpaired_simulator_seeds(tmp_path) -> None:
    write_result(tmp_path, "full", 9, 0.9)
    write_result(
        tmp_path,
        "without_anchor",
        4,
        0.4,
        seeds=list(range(100001, 100011)),
    )
    write_result(tmp_path, "without_sliding", 7, 0.7)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"released checkpoint")

    with pytest.raises(ValueError, match="identical simulator seeds"):
        summarize(
            tmp_path,
            checkpoint=checkpoint,
            task="put_back_block",
            seed_start=100000,
        )
