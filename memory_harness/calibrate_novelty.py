from __future__ import annotations

import argparse
import collections
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from memory_harness.components import NoveltyWrite
from memory_harness.components import RingStore
from memory_harness.components import TokenEncoder
from memory_harness.contracts import MemoryStep


DEFAULT_THRESHOLDS = (5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4, 7.5e-4, 1e-3)


def _read_manifest(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("context manifest must be a JSON object")
    if payload.get("schema_version") != "emac_mem0_context/v4":
        raise ValueError(
            "novelty calibration requires an emac_mem0_context/v4 manifest"
        )
    return payload


def reconstruct_source_sequences(
    manifest_path: pathlib.Path,
    context_bank_path: pathlib.Path,
) -> dict[str, np.ndarray]:
    """Recover causal source latents from the next step's Mem-0 context.

    Context row t is produced before source t is written. Therefore the newest
    sliding token in row t+1 is source t. The final source in every episode is
    intentionally unavailable and excluded rather than inferred.
    """

    manifest = _read_manifest(manifest_path)
    representation = manifest.get("representation")
    if not isinstance(representation, dict):
        raise ValueError("manifest representation must be an object")
    layout = representation.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("manifest representation.layout must be an object")
    history_slots = layout.get("history_slots")
    if (
        not isinstance(history_slots, list)
        or len(history_slots) != 2
        or not all(isinstance(value, int) for value in history_slots)
    ):
        raise ValueError("manifest history_slots must be [start, stop]")
    history_start, history_stop = history_slots
    if history_start < 0 or history_stop <= history_start:
        raise ValueError("manifest history_slots define an invalid interval")
    newest_slot = history_stop - 1

    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("manifest must contain non-empty segments")
    rows_by_episode: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for raw_row in segments:
        if not isinstance(raw_row, dict):
            raise ValueError("manifest segment must be an object")
        episode_id = str(raw_row["source_episode_id"])
        rows_by_episode[episode_id].append(raw_row)
    for rows in rows_by_episode.values():
        rows.sort(key=lambda row: int(row["start_frame"]))

    with np.load(context_bank_path, allow_pickle=False) as bank:
        required = {"item_ids", "tokens", "masks"}
        if not required.issubset(bank.files):
            raise ValueError(
                f"context bank is missing arrays: {sorted(required - set(bank.files))}"
            )
        item_ids = np.asarray(bank["item_ids"]).astype(str)
        tokens = np.asarray(bank["tokens"])
        masks = np.asarray(bank["masks"], dtype=np.bool_)
        if tokens.ndim != 3 or masks.shape != tokens.shape[:2]:
            raise ValueError("context bank tokens/masks have incompatible shapes")
        if newest_slot >= tokens.shape[1]:
            raise ValueError("manifest history layout exceeds context bank width")
        if len(item_ids) != len(tokens) or len(set(item_ids.tolist())) != len(item_ids):
            raise ValueError("context bank item_ids must be unique and row-aligned")
        index_by_id = {item_id: index for index, item_id in enumerate(item_ids)}

        output: dict[str, np.ndarray] = {}
        for episode_id, rows in sorted(rows_by_episode.items()):
            recovered: list[np.ndarray] = []
            for next_row in rows[1:]:
                item_id = str(next_row["matched_item_id"])
                if item_id not in index_by_id:
                    raise ValueError(
                        f"manifest item is absent from context bank: {item_id}"
                    )
                bank_index = index_by_id[item_id]
                if not masks[bank_index, newest_slot]:
                    raise ValueError(
                        f"next-step context lacks newest history token: {item_id}"
                    )
                recovered.append(
                    np.asarray(tokens[bank_index, newest_slot], dtype=np.float32)
                )
            if recovered:
                output[episode_id] = np.stack(recovered, axis=0)
    if not output:
        raise ValueError("no causal source sequences could be reconstructed")
    return output


def calibrate_sequences(
    sequences: Mapping[str, np.ndarray],
    *,
    thresholds: Sequence[float],
    max_steps_without_write: int | None,
) -> list[dict[str, Any]]:
    if not thresholds:
        raise ValueError("at least one novelty threshold is required")
    results: list[dict[str, Any]] = []
    encoder = TokenEncoder(max_tokens=1)
    for threshold in thresholds:
        writer = NoveltyWrite(
            min_cosine_distance=float(threshold),
            max_steps_without_write=max_steps_without_write,
        )
        reason_counts: collections.Counter[str] = collections.Counter()
        decision_distances: list[float] = []
        write_gaps: list[int] = []
        max_observed_age = 0
        decision_count = 0
        write_count = 0
        for episode_id, raw_sequence in sorted(sequences.items()):
            sequence = np.asarray(raw_sequence, dtype=np.float32)
            if sequence.ndim != 2 or not len(sequence):
                raise ValueError(
                    f"episode {episode_id!r} sequence must have shape [T, D], got {sequence.shape}"
                )
            store = RingStore(capacity=1)
            previous_write_step: int | None = None
            for step_index, vector in enumerate(sequence):
                step = MemoryStep(
                    episode_id=episode_id,
                    step_index=step_index,
                    source_tokens=vector[None, :],
                    source_mask=np.ones(1, dtype=np.bool_),
                )
                decision = writer.decide(step, store)
                decision_count += 1
                reason = str(decision.details["reason"])
                reason_counts[reason] += 1
                if "cosine_distance" in decision.details:
                    decision_distances.append(
                        float(decision.details["cosine_distance"])
                    )
                if previous_write_step is not None:
                    max_observed_age = max(
                        max_observed_age, step_index - previous_write_step
                    )
                if decision.write:
                    write_count += 1
                    if previous_write_step is not None:
                        write_gaps.append(step_index - previous_write_step)
                    store.write(encoder.encode(step, path_name="sliding"))
                    previous_write_step = step_index

        distance_array = np.asarray(decision_distances, dtype=np.float64)
        gap_array = np.asarray(write_gaps, dtype=np.float64)
        results.append(
            {
                "min_cosine_distance": float(threshold),
                "max_steps_without_write": max_steps_without_write,
                "decision_count": decision_count,
                "write_count": write_count,
                "write_fraction": write_count / decision_count,
                "reason_counts": dict(sorted(reason_counts.items())),
                "decision_distance_quantiles": {
                    f"q{int(quantile * 100):02d}": float(
                        np.quantile(distance_array, quantile)
                    )
                    for quantile in (0.1, 0.5, 0.9, 0.99)
                }
                if len(distance_array)
                else {},
                "write_gap_mean": float(gap_array.mean()) if len(gap_array) else None,
                "write_gap_p95": float(np.quantile(gap_array, 0.95))
                if len(gap_array)
                else None,
                "max_observed_steps_since_write": max_observed_age,
            }
        )
    return results


def build_report(
    manifest_path: pathlib.Path,
    context_bank_path: pathlib.Path,
    *,
    thresholds: Sequence[float],
    max_steps_without_write: int | None,
) -> dict[str, Any]:
    sequences = reconstruct_source_sequences(manifest_path, context_bank_path)
    return {
        "schema_version": "memory_harness_novelty_calibration/v1",
        "inputs": {
            "context_manifest": str(manifest_path.resolve()),
            "context_bank": str(context_bank_path.resolve()),
            "reconstruction": (
                "source t is the newest sliding token in causal context row t+1; "
                "the final source of each episode is excluded"
            ),
        },
        "num_episodes": len(sequences),
        "num_reconstructed_sources": sum(
            len(sequence) for sequence in sequences.values()
        ),
        "embedding_width": int(next(iter(sequences.values())).shape[1]),
        "results": calibrate_sequences(
            sequences,
            thresholds=thresholds,
            max_steps_without_write=max_steps_without_write,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the deterministic novelty writer on causal Mem-0 latents."
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--context-bank", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        type=float,
        action="append",
        help="Repeat to scan multiple thresholds; defaults to the standard grid.",
    )
    parser.add_argument("--max-steps-without-write", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.manifest,
        args.context_bank,
        thresholds=args.thresholds or DEFAULT_THRESHOLDS,
        max_steps_without_write=args.max_steps_without_write,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
