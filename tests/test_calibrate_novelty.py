from __future__ import annotations

import json

import numpy as np
import pytest

from memory_harness.calibrate_novelty import calibrate_sequences
from memory_harness.calibrate_novelty import reconstruct_source_sequences


def _write_tiny_context_data(tmp_path):
    item_ids = np.asarray(["ep1-1", "ep0-2", "ep0-0", "ep1-0", "ep0-1"])
    tokens = np.zeros((5, 3, 2), dtype=np.float32)
    masks = np.zeros((5, 3), dtype=np.bool_)
    index = {item_id: position for position, item_id in enumerate(item_ids)}
    tokens[index["ep0-1"], 2] = [1.0, 0.0]
    tokens[index["ep0-2"], 2] = [1.0, 0.0]
    tokens[index["ep1-1"], 2] = [0.0, 1.0]
    masks[index["ep0-1"], [0, 2]] = True
    masks[index["ep0-2"], [0, 1, 2]] = True
    masks[index["ep1-1"], [0, 2]] = True

    bank = tmp_path / "bank.npz"
    np.savez(bank, item_ids=item_ids, tokens=tokens, masks=masks)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "emac_mem0_context/v4",
                "representation": {"layout": {"history_slots": [1, 3]}},
                "segments": [
                    {
                        "source_episode_id": 0,
                        "start_frame": step,
                        "matched_item_id": f"ep0-{step}",
                    }
                    for step in range(3)
                ]
                + [
                    {
                        "source_episode_id": 1,
                        "start_frame": step,
                        "matched_item_id": f"ep1-{step}",
                    }
                    for step in range(2)
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, bank


def test_reconstructs_sources_from_next_causal_context(tmp_path) -> None:
    manifest, bank = _write_tiny_context_data(tmp_path)

    sequences = reconstruct_source_sequences(manifest, bank)

    np.testing.assert_array_equal(sequences["0"], [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(sequences["1"], [[0.0, 1.0]])


def test_calibration_uses_real_novelty_writer_decisions() -> None:
    sequences = {
        "0": np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        "1": np.asarray([[0.0, 1.0]]),
    }

    result = calibrate_sequences(
        sequences, thresholds=[0.1], max_steps_without_write=2
    )[0]

    assert result["decision_count"] == 4
    assert result["write_count"] == 3
    assert result["write_fraction"] == pytest.approx(0.75)
    assert result["reason_counts"] == {
        "empty_store": 2,
        "max_interval": 1,
        "redundant": 1,
    }
    assert result["max_observed_steps_since_write"] == 2
