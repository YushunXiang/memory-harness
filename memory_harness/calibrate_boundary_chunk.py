from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from memory_harness.calibrate_novelty import reconstruct_source_sequences
from memory_harness.profile_boundary_chunk import profile_sequences
from memory_harness.profile_boundary_chunk import split_sequences


SCHEMA_VERSION = "memory_harness.boundary_chunk_calibration/v1"


def calibrate(
    train_sequences: Mapping[str, np.ndarray],
    *,
    thresholds: Sequence[float],
    max_items: int,
    min_chunk_items: int,
    target_median_chunk_count: int,
) -> dict[str, Any]:
    if not thresholds:
        raise ValueError("at least one threshold is required")
    if list(thresholds) != sorted(set(thresholds)):
        raise ValueError("thresholds must be unique and increasing")
    if target_median_chunk_count < 2:
        raise ValueError("target_median_chunk_count must be at least two")
    rows = []
    for threshold in thresholds:
        profile = profile_sequences(
            train_sequences,
            max_items=max_items,
            boundary_similarity_threshold=float(threshold),
            min_chunk_items=min_chunk_items,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "median_chunk_count": profile["chunk_count_quantiles"]["q50"],
                "median_selected_chunk_length": profile[
                    "selected_chunk_length_quantiles"
                ]["q50"],
                "mean_outside_latest_window_fraction": profile[
                    "mean_outside_latest_window_fraction"
                ],
                "underfilled_query_fraction": profile["underfilled_query_fraction"],
                "exact_sliding_query_fraction": profile["exact_sliding_query_fraction"],
            }
        )
    eligible = [
        row
        for row in rows
        if row["median_chunk_count"] >= target_median_chunk_count
        and row["underfilled_query_fraction"] == 0
    ]
    if not eligible:
        raise ValueError("no threshold meets segmentation and budget criteria")
    selected = eligible[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_split": "train",
        "selection_uses_rollout_outcomes": False,
        "selection_rule": (
            "smallest tested threshold with median chunk count at least "
            f"{target_median_chunk_count} and zero underfilled full-history queries"
        ),
        "max_items": max_items,
        "min_chunk_items": min_chunk_items,
        "target_median_chunk_count": target_median_chunk_count,
        "selected_threshold": selected["threshold"],
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a boundary-chunk threshold using training trajectories only."
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--context-bank", type=pathlib.Path, required=True)
    parser.add_argument("--template-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--threshold", type=float, action="append", required=True)
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--min-chunk-items", type=int, default=30)
    parser.add_argument("--target-median-chunk-count", type=int, default=3)
    args = parser.parse_args()
    sequences = reconstruct_source_sequences(args.manifest, args.context_bank)
    train_sequences = split_sequences(sequences, args.template_manifest)["train"]
    result = calibrate(
        train_sequences,
        thresholds=args.threshold,
        max_items=args.max_items,
        min_chunk_items=args.min_chunk_items,
        target_median_chunk_count=args.target_median_chunk_count,
    )
    result["inputs"] = {
        "context_manifest": str(args.manifest.resolve()),
        "context_bank": str(args.context_bank.resolve()),
        "template_manifest": str(args.template_manifest.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
