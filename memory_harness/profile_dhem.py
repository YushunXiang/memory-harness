from __future__ import annotations

import argparse
import collections
import json
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from memory_harness.calibrate_novelty import reconstruct_source_sequences
from memory_harness.components import DHEMEventStore
from memory_harness.components import TokenEncoder
from memory_harness.contracts import MemoryStep


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {}
    return {
        f"q{int(quantile * 100):02d}": float(np.quantile(array, quantile))
        for quantile in (0.0, 0.5, 0.9, 0.99, 1.0)
    }


def profile_sequences(
    sequences: Mapping[str, np.ndarray],
    *,
    capacity: int,
    temporal_decay: float,
) -> dict[str, Any]:
    if not sequences:
        raise ValueError("at least one source sequence is required")
    encoder = TokenEncoder(max_tokens=1)
    action_counts: collections.Counter[str] = collections.Counter()
    final_ages: list[float] = []
    final_masses: list[float] = []
    per_episode: list[dict[str, Any]] = []
    source_count = 0
    for episode_id, raw_sequence in sorted(sequences.items()):
        sequence = np.asarray(raw_sequence, dtype=np.float32)
        if sequence.ndim != 2 or not len(sequence):
            raise ValueError(
                f"episode {episode_id!r} sequence must have shape [T, D], "
                f"got {sequence.shape}"
            )
        store = DHEMEventStore(
            capacity=capacity,
            temporal_decay=temporal_decay,
        )
        for step_index, vector in enumerate(sequence):
            step = MemoryStep(
                episode_id=episode_id,
                step_index=step_index,
                source_tokens=vector[None, :],
                source_mask=np.ones(1, dtype=np.bool_),
            )
            details = store.write(encoder.encode(step, path_name="dhem_event"))
            action_counts[str(details["maintenance_action"])] += 1
        source_count += len(sequence)
        final_items = store.items()
        representative_times = [
            float(item.metadata["representative_time"]) for item in final_items
        ]
        masses = [float(item.metadata["accumulated_mass"]) for item in final_items]
        ages = [float(len(sequence) - 1 - time) for time in representative_times]
        final_ages.extend(ages)
        final_masses.extend(masses)
        per_episode.append(
            {
                "episode_id": episode_id,
                "source_count": len(sequence),
                "final_item_count": len(final_items),
                "retained_source_mass": float(sum(masses)),
                "discarded_source_mass": float(len(sequence) - sum(masses)),
                "anchor_step_index": final_items[0].step_index,
                "latest_step_index": final_items[-1].step_index,
                "representative_age_quantiles": _quantiles(ages),
                "accumulated_mass_quantiles": _quantiles(masses),
            }
        )

    maintenance_count = (
        action_counts["discard_incoming"]
        + action_counts["merge_history_and_append"]
    )
    return {
        "num_episodes": len(per_episode),
        "num_sources": source_count,
        "capacity": capacity,
        "temporal_decay": temporal_decay,
        "action_counts": dict(sorted(action_counts.items())),
        "full_store_decisions": maintenance_count,
        "discard_fraction_at_capacity": (
            action_counts["discard_incoming"] / maintenance_count
            if maintenance_count
            else 0.0
        ),
        "merge_fraction_at_capacity": (
            action_counts["merge_history_and_append"] / maintenance_count
            if maintenance_count
            else 0.0
        ),
        "final_representative_age_quantiles": _quantiles(final_ages),
        "final_accumulated_mass_quantiles": _quantiles(final_masses),
        "per_episode": per_episode,
    }


def build_report(
    manifest_path: pathlib.Path,
    context_bank_path: pathlib.Path,
    *,
    capacity: int,
    temporal_decay: float,
) -> dict[str, Any]:
    sequences = reconstruct_source_sequences(manifest_path, context_bank_path)
    return {
        "schema_version": "memory_harness_dhem_profile/v1",
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
            capacity=capacity,
            temporal_decay=temporal_decay,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile DHEM event maintenance on causal Mem-0 latents."
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--context-bank", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--capacity", type=int, default=30)
    parser.add_argument("--temporal-decay", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temporal_decay = (
        float(args.capacity - 1)
        if args.temporal_decay is None
        else args.temporal_decay
    )
    report = build_report(
        args.manifest,
        args.context_bank,
        capacity=args.capacity,
        temporal_decay=temporal_decay,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
