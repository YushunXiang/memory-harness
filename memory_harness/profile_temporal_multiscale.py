from __future__ import annotations

import argparse
import collections
import json
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from memory_harness.calibrate_novelty import reconstruct_source_sequences
from memory_harness.components import RingStore
from memory_harness.components import LatestRetriever
from memory_harness.components import TemporalMultiscaleRetriever
from memory_harness.components import TokenEncoder
from memory_harness.components import UniformGlobalRetriever
from memory_harness.contracts import MemoryStep


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{int(quantile * 100):02d}": float(np.quantile(array, quantile))
        for quantile in (0.0, 0.5, 0.9, 0.99, 1.0)
    }


def profile_sequences(
    sequences: Mapping[str, np.ndarray],
    *,
    max_items: int,
    exponential_items: int,
) -> dict[str, Any]:
    if not sequences:
        raise ValueError("at least one source sequence is required")
    if max_items < 2:
        raise ValueError("max_items must be at least two for recent/global profiling")
    recent_items = max_items // 2
    global_items = max_items - recent_items
    retrievers = {
        "temporal_multiscale": TemporalMultiscaleRetriever(
            max_items=max_items,
            exponential_items=exponential_items,
        ),
        "uniform_global": UniformGlobalRetriever(max_items=max_items),
    }
    recent_retriever = LatestRetriever(max_items=recent_items)
    reserved_global_retriever = UniformGlobalRetriever(
        max_items=global_items,
        exclude_recent_items=recent_items,
    )
    encoder = TokenEncoder(max_tokens=1)
    query_count = 0
    stats = {
        name: {
            "selected_counts": [],
            "selected_lags": [],
            "oldest_lags": [],
            "outside_latest_fractions": [],
            "branch_counts": collections.Counter(),
            "exact_sliding_count": 0,
        }
        for name in (*retrievers, "recent_global")
    }
    pair_names = (
        ("temporal_multiscale", "uniform_global"),
        ("recent_global", "temporal_multiscale"),
        ("recent_global", "uniform_global"),
    )
    pair_stats = {
        pair: {"exact_count": 0, "jaccards": []} for pair in pair_names
    }
    num_sources = 0

    for episode_id, raw_sequence in sorted(sequences.items()):
        sequence = np.asarray(raw_sequence, dtype=np.float32)
        if sequence.ndim != 2 or len(sequence) < max_items + 2:
            raise ValueError(
                f"episode {episode_id!r} sequence must have shape "
                f"[T>={max_items + 2}, D], got {sequence.shape}"
            )
        if not np.isfinite(sequence).all():
            raise ValueError(f"episode {episode_id!r} contains non-finite tokens")
        store = RingStore(capacity=len(sequence))
        num_sources += len(sequence)
        for query_index, vector in enumerate(sequence):
            step = MemoryStep(
                episode_id=episode_id,
                step_index=query_index,
                source_tokens=vector[None, :],
                source_mask=np.ones((1,), dtype=np.bool_),
            )
            if query_index > max_items:
                sliding_set = set(range(query_index - max_items, query_index))
                selected_sets: dict[str, set[int]] = {}
                selections: dict[str, tuple[list[int], list[str]]] = {}
                for name, retriever in retrievers.items():
                    result = retriever.retrieve(step, store)
                    selected_steps = [item.step_index for item in result.items]
                    selections[name] = (
                        selected_steps,
                        [row["selected_by"] for row in result.details["selected"]],
                    )
                recent_result = recent_retriever.retrieve(step, store)
                global_result = reserved_global_retriever.retrieve(step, store)
                selections["recent_global"] = (
                    [item.step_index for item in recent_result.items]
                    + [item.step_index for item in global_result.items],
                    ["recent"] * len(recent_result.items)
                    + ["global_uniform"] * len(global_result.items),
                )
                for name, (selected_steps, branches) in selections.items():
                    selected_set = set(selected_steps)
                    selected_sets[name] = selected_set
                    current = stats[name]
                    current["selected_counts"].append(len(selected_steps))
                    lags = [query_index - selected for selected in selected_steps]
                    current["selected_lags"].extend(lags)
                    current["oldest_lags"].append(max(lags))
                    current["outside_latest_fractions"].append(
                        len(selected_set - sliding_set) / len(selected_set)
                    )
                    current["branch_counts"].update(branches)
                    current["exact_sliding_count"] += int(
                        selected_set == sliding_set
                    )
                for pair, current_pair in pair_stats.items():
                    first_set, second_set = (selected_sets[name] for name in pair)
                    current_pair["exact_count"] += int(first_set == second_set)
                    current_pair["jaccards"].append(
                        len(first_set & second_set) / len(first_set | second_set)
                    )
                query_count += 1
            store.write(encoder.encode(step, path_name="temporal_multiscale"))

    if not query_count:
        raise ValueError("profile produced no full-history queries")
    profiles: dict[str, Any] = {}
    for name, current in stats.items():
        profiles[name] = {
            "mean_selected_item_count": float(
                np.mean(current["selected_counts"])
            ),
            "mean_outside_latest_window_fraction": float(
                np.mean(current["outside_latest_fractions"])
            ),
            "exact_sliding_query_fraction": (
                current["exact_sliding_count"] / query_count
            ),
            "mean_selected_by_count": {
                branch: count / query_count
                for branch, count in sorted(current["branch_counts"].items())
            },
            "selected_lag_quantiles": _quantiles(current["selected_lags"]),
            "oldest_selected_lag_quantiles": _quantiles(
                current["oldest_lags"]
            ),
        }
    return {
        "num_episodes": len(sequences),
        "num_sources": num_sources,
        "max_items": max_items,
        "recent_global_allocation": {
            "recent_items": recent_items,
            "global_items": global_items,
            "global_excludes_latest_items": recent_items,
        },
        "exponential_items": exponential_items,
        "full_history_query_count": query_count,
        "profiles": profiles,
        "pairwise": {
            f"{first}_vs_{second}": {
                "exact_selected_set_query_fraction": current["exact_count"]
                / query_count,
                "mean_selected_set_jaccard": float(
                    np.mean(current["jaccards"])
                ),
            }
            for (first, second), current in pair_stats.items()
        },
    }


def build_report(
    manifest_path: pathlib.Path,
    context_bank_path: pathlib.Path,
    *,
    max_items: int,
    exponential_items: int,
) -> dict[str, Any]:
    sequences = reconstruct_source_sequences(manifest_path, context_bank_path)
    return {
        "schema_version": "memory_harness.temporal_retrieval_profile/v3",
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
            exponential_items=exponential_items,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile temporal multiscale retrieval on causal source streams."
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--context-bank", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--exponential-items", type=int, default=15)
    args = parser.parse_args()
    report = build_report(
        args.manifest,
        args.context_bank,
        max_items=args.max_items,
        exponential_items=args.exponential_items,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
