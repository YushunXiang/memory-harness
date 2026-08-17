from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from memory_harness.calibrate_novelty import reconstruct_source_sequences


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{int(q * 100):02d}": float(np.quantile(array, q))
        for q in (0.0, 0.5, 0.9, 0.99, 1.0)
    }


def profile_sequences(
    sequences: Mapping[str, np.ndarray],
    *,
    semantic_items: int,
    recent_items: int,
) -> dict[str, Any]:
    if not sequences:
        raise ValueError("at least one source sequence is required")
    if semantic_items <= 0 or recent_items <= 0:
        raise ValueError("semantic_items and recent_items must be positive")
    normalized: dict[str, np.ndarray] = {}
    for episode_id, raw in sorted(sequences.items()):
        sequence = np.asarray(raw, dtype=np.float32)
        if sequence.ndim != 2 or len(sequence) < 2:
            raise ValueError(
                f"episode {episode_id!r} sequence must have shape [T>=2, D], "
                f"got {sequence.shape}"
            )
        norms = np.linalg.norm(sequence, axis=1, keepdims=True)
        if not np.isfinite(norms).all():
            raise ValueError(f"episode {episode_id!r} contains non-finite tokens")
        normalized[episode_id] = np.divide(
            sequence,
            norms,
            out=np.zeros_like(sequence),
            where=norms > 0,
        )

    full_query_count = 0
    selected_counts: list[int] = []
    branch_overlap_counts: list[int] = []
    selected_lags: list[int] = []
    outside_latest_fractions: list[float] = []
    nominal_budget = semantic_items + recent_items
    for sequence in normalized.values():
        for query_index in range(nominal_budget + 1, len(sequence)):
            keys = sequence[:query_index]
            similarities = keys @ sequence[query_index]
            semantic_ranking = [
                int(index)
                for index in np.lexsort(
                    (-np.arange(query_index), -similarities)
                )
            ]
            recent = set(range(query_index - recent_items, query_index))
            initial_semantic = set(semantic_ranking[:semantic_items])
            semantic = set(
                [index for index in semantic_ranking if index not in recent][
                    :semantic_items
                ]
            )
            selected = semantic | recent
            latest_window = set(
                range(max(0, query_index - nominal_budget), query_index)
            )
            selected_counts.append(len(selected))
            branch_overlap_counts.append(len(initial_semantic & recent))
            selected_lags.extend(query_index - index for index in selected)
            outside_latest_fractions.append(
                len(selected - latest_window) / len(selected)
            )
            full_query_count += 1

    return {
        "num_episodes": len(normalized),
        "num_sources": sum(len(sequence) for sequence in normalized.values()),
        "semantic_items": semantic_items,
        "recent_items": recent_items,
        "nominal_token_budget": nominal_budget,
        "full_bank_query_count": full_query_count,
        "mean_selected_item_count": (
            float(np.mean(selected_counts)) if selected_counts else 0.0
        ),
        "mean_branch_overlap_count": (
            float(np.mean(branch_overlap_counts)) if branch_overlap_counts else 0.0
        ),
        "mean_outside_latest_window_fraction": (
            float(np.mean(outside_latest_fractions))
            if outside_latest_fractions
            else 0.0
        ),
        "selected_lag_quantiles": _quantiles(selected_lags),
    }


def build_report(
    manifest_path: pathlib.Path,
    context_bank_path: pathlib.Path,
    *,
    semantic_items: int,
    recent_items: int,
) -> dict[str, Any]:
    sequences = reconstruct_source_sequences(manifest_path, context_bank_path)
    return {
        "schema_version": "memory_harness_semantic_recent_union_profile/v1",
        "inputs": {
            "context_manifest": str(manifest_path.resolve()),
            "context_bank": str(context_bank_path.resolve()),
            "reconstruction": (
                "source t is the newest sliding token in causal context row t+1; "
                "the final source of each episode is excluded"
            ),
        },
        "profile": profile_sequences(
            sequences,
            semantic_items=semantic_items,
            recent_items=recent_items,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile semantic Top-K union recent-K retrieval."
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--context-bank", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--semantic-items", type=int, default=20)
    parser.add_argument("--recent-items", type=int, default=10)
    args = parser.parse_args()
    report = build_report(
        args.manifest,
        args.context_bank,
        semantic_items=args.semantic_items,
        recent_items=args.recent_items,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
