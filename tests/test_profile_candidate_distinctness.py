from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from memory_harness.config import load_program_spec
from memory_harness.profile_candidate_distinctness import _unsupported_reason
from memory_harness.profile_candidate_distinctness import compare_profiles
from memory_harness.profile_candidate_distinctness import profile_program
from memory_harness.profile_candidate_distinctness import reconstruct_phase_sequences


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _sequences() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "validation-a": rng.normal(size=(48, 2048)).astype(np.float32),
        "validation-b": rng.normal(size=(45, 2048)).astype(np.float32),
    }


def test_distinctness_profile_detects_exact_alias_but_not_different_program() -> None:
    sequences = _sequences()
    sliding = load_program_spec(CONFIG_ROOT / "fixed_sliding.json")
    anchor = load_program_spec(CONFIG_ROOT / "fixed_anchor.json")
    kwargs = {
        "phase_sequences": {
            episode_id: ("phase-a",) * len(sequence)
            for episode_id, sequence in sequences.items()
        },
        "episode_ids": tuple(sequences),
        "warmup_steps": 8,
        "query_stride": 5,
    }
    profiles = {
        "sliding_a": profile_program(sliding, sequences, **kwargs),
        "sliding_b": profile_program(sliding, sequences, **kwargs),
        "anchor": profile_program(anchor, sequences, **kwargs),
    }

    comparison = compare_profiles(profiles)

    assert comparison["behaviorally_exact_duplicate_pairs"] == [
        ["sliding_a", "sliding_b"]
    ]
    alias_pair = comparison["pairwise"]["sliding_a_vs_sliding_b"]
    assert alias_pair["mean_valid_token_multiset_jaccard"] == 1.0
    assert alias_pair["exact_valid_token_multiset_fraction"] == 1.0
    anchor_pair = comparison["pairwise"]["anchor_vs_sliding_a"]
    assert anchor_pair["exact_output_fraction"] == 0.0
    assert anchor_pair["exact_used_token_count_fraction"] == 0.0
    assert 0.0 <= anchor_pair["mean_valid_token_multiset_jaccard"] < 1.0


def test_phase_aware_profile_activates_completed_segment_handoff() -> None:
    sequences = {"validation": _sequences()["validation-a"][:8]}
    phase_sequences = {
        "validation": ("a", "a", "b", "b", "a", "a", "c", "c")
    }
    completed = load_program_spec(
        CONFIG_ROOT / "fixed_completed_phase_handoff.json"
    )
    sliding = load_program_spec(CONFIG_ROOT / "fixed_sliding.json")
    kwargs = {
        "phase_sequences": phase_sequences,
        "episode_ids": ("validation",),
        "warmup_steps": 0,
        "query_stride": 1,
    }

    completed_profile = profile_program(completed, sequences, **kwargs)
    sliding_profile = profile_program(sliding, sequences, **kwargs)

    assert completed_profile["mean_used_token_count"] == pytest.approx(0.75)
    assert (
        completed_profile["aggregate_output_sha256"]
        != sliding_profile["aggregate_output_sha256"]
    )


def test_phase_reconstruction_excludes_unrecoverable_final_source(tmp_path) -> None:
    manifest = {
        "segments": [
            {
                "source_episode_id": 7,
                "start_frame": 0,
                "phase_label": "phase-a",
            },
            {
                "source_episode_id": 7,
                "start_frame": 10,
                "phase_label": "phase-b",
            },
            {
                "source_episode_id": 7,
                "start_frame": 20,
                "phase_label": "phase-c",
            },
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert reconstruct_phase_sequences(path) == {"7": ("phase-a", "phase-b")}


def test_distinctness_profile_excludes_missing_payload_protocols() -> None:
    kinematic = load_program_spec(CONFIG_ROOT / "fixed_kinematic_event.json")
    persistent = load_program_spec(
        CONFIG_ROOT / "fixed_verified_success_latent.json"
    )
    sliding = load_program_spec(CONFIG_ROOT / "fixed_sliding.json")

    assert "robot_state" in str(_unsupported_reason(kinematic))
    assert "cross-episode outcomes" in str(_unsupported_reason(persistent))
    assert _unsupported_reason(sliding) is None


def test_near_duplicate_requires_review_without_being_exact_duplicate() -> None:
    shared = [f"shared-{index}" for index in range(49)]
    profiles = {
        "left": {
            "fingerprints": {
                "episode:31": {
                    "output_sha256": "left-output",
                    "mask_sha256": "same-mask",
                    "used_token_count": 50,
                    "valid_token_sha256": [*shared, "left-only"],
                }
            }
        },
        "right": {
            "fingerprints": {
                "episode:31": {
                    "output_sha256": "right-output",
                    "mask_sha256": "same-mask",
                    "used_token_count": 50,
                    "valid_token_sha256": [*shared, "right-only"],
                }
            }
        },
    }

    comparison = compare_profiles(profiles, near_duplicate_threshold=0.95)

    assert comparison["behaviorally_exact_duplicate_pairs"] == []
    assert comparison["near_duplicate_review_pairs"] == [
        {
            "left": "left",
            "right": "right",
            "mean_valid_token_multiset_jaccard": pytest.approx(49 / 51),
            "decision": "review_and_justify_or_remove_before_rollout",
        }
    ]
