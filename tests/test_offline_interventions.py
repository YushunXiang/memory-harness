from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_mem0_offline import INTERVENTIONS, build_interventions  # noqa: E402


@dataclasses.dataclass(frozen=True)
class _Observation:
    memory_tokens: np.ndarray
    memory_mask: np.ndarray


def test_build_interventions_respects_mem0_slot_boundaries() -> None:
    matched_tokens = np.arange(2 * 31 * 3, dtype=np.float32).reshape(2, 31, 3)
    mismatched_tokens = matched_tokens + 1_000
    matched_mask = np.zeros((2, 31), dtype=np.bool_)
    matched_mask[:, 0] = True
    matched_mask[0, 28:] = True
    matched_mask[1, 29:] = True
    mismatched_mask = matched_mask.copy()
    matched = _Observation(matched_tokens, matched_mask)
    mismatched = _Observation(mismatched_tokens, mismatched_mask)

    interventions = build_interventions(matched, mismatched)

    assert tuple(interventions) == INTERVENTIONS
    assert not np.asarray(interventions["without_anchor"].memory_mask)[:, 0].any()
    assert not np.asarray(interventions["without_sliding"].memory_mask)[:, 1:].any()
    np.testing.assert_array_equal(
        np.asarray(interventions["anchor_replaced"].memory_tokens)[:, 0],
        mismatched_tokens[:, 0],
    )
    np.testing.assert_array_equal(
        np.asarray(interventions["sliding_replaced"].memory_tokens)[:, 1:],
        mismatched_tokens[:, 1:],
    )
    np.testing.assert_array_equal(
        np.asarray(interventions["sliding_shuffled"].memory_tokens)[0, 28:],
        matched_tokens[0, 28:][::-1],
    )
