from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
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
    max_items: int,
    penalties: Sequence[float],
) -> dict[str, Any]:
    if not sequences:
        raise ValueError("at least one source sequence is required")
    if max_items <= 0:
        raise ValueError("max_items must be positive")
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

    profiles: list[dict[str, Any]] = []
    for raw_penalty in penalties:
        penalty = float(raw_penalty)
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("penalties must be finite and non-negative")
        full_query_count = 0
        selected_lags: list[int] = []
        latest_overlap: list[float] = []
        anchor_selected = 0
        for sequence in normalized.values():
            for query_index in range(max_items + 1, len(sequence)):
                keys = sequence[:query_index]
                gaps = query_index - np.arange(query_index)
                scores = keys @ sequence[query_index] - penalty * gaps
                order = np.lexsort((-np.arange(query_index), -scores))
                selected = order[:max_items]
                latest = set(range(query_index - max_items, query_index))
                selected_set = set(int(index) for index in selected)
                selected_lags.extend(int(query_index - index) for index in selected)
                latest_overlap.append(len(latest & selected_set) / max_items)
                anchor_selected += int(0 in selected_set)
                full_query_count += 1
        profiles.append(
            {
                "recency_penalty": penalty,
                "full_bank_query_count": full_query_count,
                "selected_lag_quantiles": _quantiles(selected_lags),
                "mean_latest_window_overlap": (
                    float(np.mean(latest_overlap)) if latest_overlap else 0.0
                ),
                "mean_outside_latest_window_fraction": (
                    float(1.0 - np.mean(latest_overlap)) if latest_overlap else 0.0
                ),
                "anchor_selection_fraction": (
                    anchor_selected / full_query_count if full_query_count else 0.0
                ),
            }
        )
    return {
        "num_episodes": len(normalized),
        "num_sources": sum(len(sequence) for sequence in normalized.values()),
        "max_items": max_items,
        "profiles": profiles,
    }


def build_report(
    manifest_path: pathlib.Path,
    context_bank_path: pathlib.Path,
    *,
    max_items: int,
    penalties: Sequence[float],
) -> dict[str, Any]:
    sequences = reconstruct_source_sequences(manifest_path, context_bank_path)
    return {
        "schema_version": "memory_harness_content_recency_profile/v1",
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
            max_items=max_items,
            penalties=penalties,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile contextual-latent content+recency retrieval."
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--context-bank", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=(0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
    )
    args = parser.parse_args()
    report = build_report(
        args.manifest,
        args.context_bank,
        max_items=args.max_items,
        penalties=args.penalties,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
