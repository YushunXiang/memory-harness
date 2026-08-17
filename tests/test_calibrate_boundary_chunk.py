from __future__ import annotations

import numpy as np
import pytest

from memory_harness.calibrate_boundary_chunk import calibrate


def test_calibration_selects_smallest_budget_safe_segmenting_threshold() -> None:
    sequence = np.concatenate(
        [
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (20, 1)),
            np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (20, 1)),
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (20, 1)),
        ]
    )
    report = calibrate(
        {"train": sequence},
        thresholds=(0.0, 0.5, 0.9),
        max_items=10,
        min_chunk_items=10,
        target_median_chunk_count=2,
    )

    assert report["selected_threshold"] == 0.5
    assert report["selection_split"] == "train"
    assert report["selection_uses_rollout_outcomes"] is False


def test_calibration_rejects_unordered_thresholds() -> None:
    with pytest.raises(ValueError, match="unique and increasing"):
        calibrate(
            {"train": np.ones((12, 2), dtype=np.float32)},
            thresholds=(0.5, 0.4),
            max_items=4,
            min_chunk_items=4,
            target_median_chunk_count=2,
        )
