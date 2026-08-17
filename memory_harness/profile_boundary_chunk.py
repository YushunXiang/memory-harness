from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from memory_harness.calibrate_novelty import reconstruct_source_sequences


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{int(quantile * 100):02d}": float(np.quantile(array, quantile))
        for quantile in (0.0, 0.5, 0.9, 0.99, 1.0)
    }


def _select_indices(
    *,
    query_similarities: np.ndarray,
    adjacent_similarities: np.ndarray,
    max_items: int,
    boundary_similarity_threshold: float,
    min_chunk_items: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_count = len(query_similarities)
    if adjacent_similarities.shape != (max(candidate_count - 1, 0),):
        raise ValueError("adjacent similarity count does not match candidates")
    cuts = (
        np.flatnonzero(adjacent_similarities < boundary_similarity_threshold) + 1
    ).tolist()
    edges = [0, *cuts, candidate_count]
    chunks = [(edges[index], edges[index + 1]) for index in range(len(edges) - 1)]
    eligible = [chunk for chunk in chunks if chunk[1] - chunk[0] >= min_chunk_items]
    used_fallback = False
    if not eligible:
        eligible = [(0, candidate_count)]
        used_fallback = True
    scored = [
        (float(np.max(query_similarities[start:stop])), stop, start)
        for start, stop in eligible
    ]
    _, stop, start = max(scored, key=lambda row: (row[0], row[1]))
    chunk_indices = np.arange(start, stop, dtype=np.int64)
    if len(chunk_indices) > max_items:
        positions = np.linspace(
            0,
            len(chunk_indices) - 1,
            num=max_items,
            dtype=np.int64,
        )
        selected = chunk_indices[positions]
    else:
        selected = chunk_indices
    return selected, {
        "boundary_count": len(cuts),
        "chunk_count": len(chunks),
        "selected_chunk_length": len(chunk_indices),
        "used_minimum_fallback": used_fallback,
    }


def profile_sequences(
    sequences: Mapping[str, np.ndarray],
    *,
    max_items: int,
    boundary_similarity_threshold: float,
    min_chunk_items: int,
) -> dict[str, Any]:
    if not sequences:
        raise ValueError("at least one source sequence is required")
    selected_counts: list[float] = []
    selected_chunk_lengths: list[float] = []
    chunk_counts: list[float] = []
    boundary_counts: list[float] = []
    outside_latest_fractions: list[float] = []
    selected_span_densities: list[float] = []
    selected_lags: list[float] = []
    exact_sliding_count = 0
    underfilled_count = 0
    fallback_count = 0
    query_count = 0
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
        norms = np.linalg.norm(sequence, axis=1, keepdims=True)
        normalized = np.divide(
            sequence,
            norms,
            out=np.zeros_like(sequence),
            where=norms > 0,
        )
        pairwise_similarities = normalized @ normalized.T
        adjacent_similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)
        num_sources += len(sequence)
        for query_index in range(len(sequence)):
            if query_index > max_items:
                selected, details = _select_indices(
                    query_similarities=pairwise_similarities[:query_index, query_index],
                    adjacent_similarities=adjacent_similarities[: query_index - 1],
                    max_items=max_items,
                    boundary_similarity_threshold=boundary_similarity_threshold,
                    min_chunk_items=min_chunk_items,
                )
                selected_steps = selected.tolist()
                selected_set = set(selected_steps)
                sliding_set = set(range(query_index - max_items, query_index))
                selected_count = len(selected_steps)
                selected_counts.append(float(selected_count))
                underfilled_count += int(selected_count < max_items)
                exact_sliding_count += int(selected_set == sliding_set)
                outside_latest_fractions.append(
                    len(selected_set - sliding_set) / max(selected_count, 1)
                )
                lags = [query_index - selected for selected in selected_steps]
                selected_lags.extend(float(lag) for lag in lags)
                span = max(selected_steps) - min(selected_steps) + 1
                selected_span_densities.append(selected_count / span)
                selected_chunk_lengths.append(float(details["selected_chunk_length"]))
                chunk_counts.append(float(details["chunk_count"]))
                boundary_counts.append(float(details["boundary_count"]))
                fallback_count += int(details["used_minimum_fallback"])
                query_count += 1

    if not query_count:
        raise ValueError("profile produced no full-history queries")
    return {
        "num_episodes": len(sequences),
        "num_sources": num_sources,
        "full_history_query_count": query_count,
        "max_items": max_items,
        "boundary_similarity_threshold": boundary_similarity_threshold,
        "min_chunk_items": min_chunk_items,
        "mean_selected_item_count": float(np.mean(selected_counts)),
        "underfilled_query_fraction": underfilled_count / query_count,
        "exact_sliding_query_fraction": exact_sliding_count / query_count,
        "mean_outside_latest_window_fraction": float(np.mean(outside_latest_fractions)),
        "mean_selected_span_density": float(np.mean(selected_span_densities)),
        "minimum_chunk_fallback_fraction": fallback_count / query_count,
        "selected_chunk_length_quantiles": _quantiles(selected_chunk_lengths),
        "chunk_count_quantiles": _quantiles(chunk_counts),
        "boundary_count_quantiles": _quantiles(boundary_counts),
        "selected_lag_quantiles": _quantiles(selected_lags),
    }


def split_sequences(
    sequences: Mapping[str, np.ndarray], template_manifest_path: pathlib.Path
) -> dict[str, dict[str, np.ndarray]]:
    template = json.loads(template_manifest_path.read_text(encoding="utf-8"))
    split_by_episode: dict[str, str] = {}
    for segment in template.get("segments", []):
        episode_id = str(segment["lerobot_episode_index"])
        split = str(segment["split"])
        previous = split_by_episode.setdefault(episode_id, split)
        if previous != split:
            raise ValueError(f"episode {episode_id} appears in multiple splits")
    unknown = set(sequences) - set(split_by_episode)
    if unknown:
        raise ValueError(
            f"source episodes are absent from template splits: {sorted(unknown)}"
        )
    return {
        split: {
            episode_id: sequence
            for episode_id, sequence in sequences.items()
            if split == "all" or split_by_episode[episode_id] == split
        }
        for split in ("train", "validation", "all")
    }


def build_report(
    manifest_path: pathlib.Path,
    context_bank_path: pathlib.Path,
    template_manifest_path: pathlib.Path,
    calibration_report_path: pathlib.Path,
    *,
    max_items: int,
    min_chunk_items: int,
) -> dict[str, Any]:
    sequences = reconstruct_source_sequences(manifest_path, context_bank_path)
    split_sources = split_sequences(sequences, template_manifest_path)
    calibration = json.loads(calibration_report_path.read_text(encoding="utf-8"))
    boundary_similarity_threshold = float(calibration["selected_threshold"])
    if int(calibration["max_items"]) != max_items:
        raise ValueError("calibration/report max_items mismatch")
    if int(calibration["min_chunk_items"]) != min_chunk_items:
        raise ValueError("calibration/report min_chunk_items mismatch")
    profiles = {
        split: profile_sequences(
            split_sources[split],
            max_items=max_items,
            boundary_similarity_threshold=boundary_similarity_threshold,
            min_chunk_items=min_chunk_items,
        )
        for split in ("train", "validation", "all")
    }
    return {
        "schema_version": "memory_harness.boundary_chunk_profile/v2",
        "method_scope": (
            "contextual-token boundary/chunk lower bound inspired by "
            "RoboMME-Interference; not its SigLIP retrieval reproduction"
        ),
        "inputs": {
            "context_manifest": str(manifest_path.resolve()),
            "context_bank": str(context_bank_path.resolve()),
            "template_manifest": str(template_manifest_path.resolve()),
            "calibration_report": str(calibration_report_path.resolve()),
            "reconstruction": (
                "source t is the newest sliding token in causal context row t+1; "
                "the final source of each episode is excluded"
            ),
        },
        "threshold_selection_scope": "train_split_only_no_rollout_outcomes",
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile coherent boundary-chunk retrieval on causal source streams."
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--context-bank", type=pathlib.Path, required=True)
    parser.add_argument("--template-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--calibration-report", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--min-chunk-items", type=int, default=4)
    args = parser.parse_args()
    report = build_report(
        args.manifest,
        args.context_bank,
        args.template_manifest,
        args.calibration_report,
        max_items=args.max_items,
        min_chunk_items=args.min_chunk_items,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
