from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from memory_harness.build_training_data import Chunk, simulate_program_contexts
from memory_harness.profile_tiered_contexts import profile_tiered_contexts


ROOT = Path(__file__).resolve().parents[1]


def test_profile_recovers_real_moments_and_crosses_sliding_horizon(tmp_path) -> None:
    chunks = tuple(
        Chunk(
            episode=0,
            ordinal=index,
            start=index,
            end=index + 1,
            phase="episode",
            prompt="Put back the block.",
            split="train",
        )
        for index in range(40)
    )
    moments = tuple(
        np.asarray([[float(index + 1), 1.0]], dtype=np.float32)
        for index in range(len(chunks))
    )
    _, contexts = simulate_program_contexts(chunks, moments)
    item_ids = np.asarray([chunk.item_id("anchor_sliding") for chunk in chunks])
    bank_path = tmp_path / "bank.npz"
    np.savez(
        bank_path,
        item_ids=item_ids,
        tokens=np.stack([context[0] for context in contexts]).astype(np.float16),
        masks=np.stack([context[1] for context in contexts]),
    )
    manifest = {
        "schema_version": "emac_mem0_context/v4",
        "representation": {
            "program": "anchor_sliding",
            "execution_order": "RETRIEVE_USE_THEN_WRITE",
        },
        "token_budget": 31,
        "segments": [
            {
                "lerobot_episode_index": 0,
                "start_frame": chunk.start,
                "phase_label": chunk.phase,
                "matched_item_id": chunk.item_id("anchor_sliding"),
            }
            for chunk in chunks
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    profile = profile_tiered_contexts(
        manifest_path=manifest_path,
        context_bank_path=bank_path,
        program_config=ROOT / "configs/fixed_tiered_chunk_mean.json",
    )

    assert profile["causal_recovery"] is True
    assert profile["num_queries"] == 40
    assert profile["exact_same_query_fraction"] == 7 / 40
    assert profile["queries_reaching_beyond_sliding_30_fraction"] > 0
    assert profile["retained_token_count"]["max"] == 14
    assert profile["represented_source_item_count"]["max"] == 39
    assert profile["maintenance_counts"]["migrate_chunk"] == 12
    assert profile["maintenance_counts"]["consolidate_long_term_adjacent"] == 4
