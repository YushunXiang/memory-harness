from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from memory_harness.config import load_program_spec
from memory_harness.contracts import MemoryItem, MemoryStep
from memory_harness.registry import build_program


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p100": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p100": float(np.max(array)),
    }


def _source_span(item: MemoryItem) -> tuple[int, int, int]:
    start = int(item.metadata.get("summary_start_step", item.step_index))
    end = int(item.metadata.get("summary_end_step", item.step_index))
    count = int(item.metadata.get("source_item_count", 1))
    if start < 0 or end < start or count <= 0:
        raise ValueError(f"invalid source span metadata for {item.item_id}")
    return start, end, count


def _validate_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "emac_mem0_context/v4":
        raise ValueError("source manifest must use emac_mem0_context/v4")
    representation = manifest.get("representation")
    if not isinstance(representation, dict):
        raise ValueError("source manifest is missing representation")
    if representation.get("program") != "anchor_sliding":
        raise ValueError("source contexts must come from anchor_sliding")
    if representation.get("execution_order") != "RETRIEVE_USE_THEN_WRITE":
        raise ValueError("source contexts must retrieve before write")
    if manifest.get("token_budget") != 31:
        raise ValueError("source context token budget must be 31")
    rows = manifest.get("segments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest has no segments")
    return [dict(row) for row in rows]


def _recover_episode_moments(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokens: np.ndarray,
    masks: np.ndarray,
    index_by_id: Mapping[str, int],
) -> tuple[np.ndarray, ...]:
    moments: list[np.ndarray] = []
    for index in range(len(rows) - 1):
        next_id = str(rows[index + 1]["matched_item_id"])
        next_index = index_by_id[next_id]
        if not bool(masks[next_index, -1]):
            raise ValueError(
                f"cannot recover source moment before {next_id}: latest slot is masked"
            )
        moments.append(np.asarray(tokens[next_index, -1:, :], dtype=np.float32))
    # RETRIEVE->USE->WRITE means the final source token cannot affect any context
    # in this episode.  A finite placeholder lets us execute the final write while
    # keeping every profiled retrieval identical to deployment semantics.
    embed_dim = int(tokens.shape[-1])
    moments.append(np.zeros((1, embed_dim), dtype=np.float32))
    return tuple(moments)


def profile_tiered_contexts(
    *,
    manifest_path: pathlib.Path,
    context_bank_path: pathlib.Path,
    program_config: pathlib.Path,
) -> dict[str, Any]:
    """Profile a two-tier candidate on contextual latents from a real bank.

    The source bank is an anchor+sliding run.  Its next query contains the
    preceding step's moment in the latest history slot, so moments can be
    recovered causally without rerunning the vision backbone.  No derived token
    bank is materialized; only aggregate structural and representation metrics
    are returned.
    """

    manifest = _read_json(manifest_path)
    rows = _validate_manifest(manifest)
    spec = load_program_spec(program_config)
    if len(spec.paths) != 1 or spec.paths[0].store.type != "tiered_chunk_mean":
        raise ValueError("profile program must contain one tiered_chunk_mean store")
    if spec.utilizer.type != "mem0_context":
        raise ValueError("profile program must use the mem0_context utilizer")

    with np.load(context_bank_path, allow_pickle=False) as bank:
        required = {"item_ids", "tokens", "masks"}
        if set(bank.files) != required:
            raise ValueError(
                f"context bank keys mismatch: expected {sorted(required)}, got {sorted(bank.files)}"
            )
        item_ids = np.asarray(bank["item_ids"]).astype(str)
        tokens = np.asarray(bank["tokens"])
        masks = np.asarray(bank["masks"], dtype=np.bool_)
        if tokens.ndim != 3 or tokens.shape[1] != 31:
            raise ValueError(f"invalid context bank token shape: {tokens.shape}")
        if masks.shape != tokens.shape[:2] or len(item_ids) != len(tokens):
            raise ValueError("context bank arrays are not row-aligned")
        if len(set(item_ids.tolist())) != len(item_ids):
            raise ValueError("context bank item_ids must be unique")
        index_by_id = {item_id: index for index, item_id in enumerate(item_ids)}
        missing = {
            str(row["matched_item_id"])
            for row in rows
            if str(row["matched_item_id"]) not in index_by_id
        }
        if missing:
            raise ValueError(f"manifest references missing bank items: {sorted(missing)[:3]}")

        rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_episode[int(row["lerobot_episode_index"])].append(row)

        query_count = 0
        exact_same_count = 0
        long_term_query_count = 0
        beyond_sliding_query_count = 0
        maintenance_counts: dict[str, int] = defaultdict(int)
        relative_l2: list[float] = []
        cosine_similarity: list[float] = []
        retained_tokens: list[float] = []
        source_coverage: list[float] = []
        oldest_source_age: list[float] = []

        for episode, episode_rows in sorted(rows_by_episode.items()):
            episode_rows.sort(key=lambda row: int(row["start_frame"]))
            starts = [int(row["start_frame"]) for row in episode_rows]
            if starts != sorted(starts) or len(starts) != len(set(starts)):
                raise ValueError(f"episode {episode} has unordered or duplicate rows")
            moments = _recover_episode_moments(
                episode_rows,
                tokens=tokens,
                masks=masks,
                index_by_id=index_by_id,
            )
            episode_spec = dataclasses.replace(
                spec,
                utilizer=dataclasses.replace(
                    spec.utilizer,
                    options={
                        **dict(spec.utilizer.options),
                        "embed_dim": int(tokens.shape[-1]),
                        "sliding_window_size": 30,
                    },
                ),
            )
            program = build_program(episode_spec)
            episode_id = f"episode-{episode}"
            program.reset(episode_id=episode_id)

            for ordinal, (row, moment) in enumerate(
                zip(episode_rows, moments, strict=True)
            ):
                store_items = program.paths[0].store.items()
                spans = [_source_span(item) for item in store_items]
                covered = sum(span[2] for span in spans)
                if covered != ordinal:
                    raise ValueError(
                        f"tiered source coverage mismatch in episode {episode} "
                        f"at query {ordinal}: expected {ordinal}, got {covered}"
                    )
                has_long_term = any(
                    item.metadata.get("memory_tier") == "long_term"
                    for item in store_items
                )
                if has_long_term:
                    long_term_query_count += 1
                if spans:
                    age = ordinal - min(span[0] for span in spans)
                    oldest_source_age.append(float(age))
                    beyond_sliding_query_count += int(age > 30)
                source_coverage.append(float(covered))

                result = program.step(
                    {},
                    MemoryStep(
                        episode_id=episode_id,
                        step_index=ordinal,
                        phase=str(row.get("phase_label", "")),
                        source_tokens=moment,
                        source_mask=np.ones((1,), dtype=np.bool_),
                        metadata={"training_representation": "recovered_runtime_moment"},
                    ),
                )
                candidate_tokens = np.asarray(
                    result.observation["memory_tokens"], dtype=np.float32
                )
                candidate_mask = np.asarray(
                    result.observation["memory_mask"], dtype=np.bool_
                )
                source_index = index_by_id[str(row["matched_item_id"])]
                sliding_tokens = np.asarray(tokens[source_index], dtype=np.float32).copy()
                sliding_mask = np.asarray(masks[source_index], dtype=np.bool_).copy()
                # Remove the anchor slot so the comparison isolates tiered history
                # from the source program's separate anchor path.
                sliding_tokens[0] = 0
                sliding_mask[0] = False

                exact_same_count += int(
                    np.array_equal(candidate_mask, sliding_mask)
                    and np.array_equal(candidate_tokens, sliding_tokens)
                )
                difference = (candidate_tokens - sliding_tokens).reshape(-1)
                denominator = max(float(np.linalg.norm(sliding_tokens)), 1e-12)
                relative_l2.append(float(np.linalg.norm(difference)) / denominator)
                candidate_flat = candidate_tokens.reshape(-1)
                sliding_flat = sliding_tokens.reshape(-1)
                cosine_denominator = float(
                    np.linalg.norm(candidate_flat) * np.linalg.norm(sliding_flat)
                )
                cosine_similarity.append(
                    1.0
                    if cosine_denominator == 0
                    and np.array_equal(candidate_flat, sliding_flat)
                    else (
                        0.0
                        if cosine_denominator == 0
                        else float(
                            np.dot(candidate_flat, sliding_flat) / cosine_denominator
                        )
                    )
                )
                retained_tokens.append(float(result.used_token_count))
                query_count += 1
                for event in result.events:
                    if event.event != "WRITE":
                        continue
                    action = event.details.get("maintenance_action")
                    if isinstance(action, str):
                        maintenance_counts[action] += 1
                    nested = event.details.get("long_term_maintenance")
                    if isinstance(nested, dict) and nested.get("consolidated") is True:
                        maintenance_counts["consolidate_long_term_adjacent"] += 1

    if not query_count:
        raise ValueError("profile produced no queries")
    return {
        "schema_version": "memory_harness.tiered_context_profile/v1",
        "candidate_program": spec.name,
        "source_program": "anchor_sliding",
        "comparison": "tiered_history_vs_sliding_history_without_anchor",
        "causal_recovery": True,
        "num_episodes": len(rows_by_episode),
        "num_queries": query_count,
        "exact_same_query_fraction": exact_same_count / query_count,
        "queries_with_long_term_fraction": long_term_query_count / query_count,
        "queries_reaching_beyond_sliding_30_fraction": (
            beyond_sliding_query_count / query_count
        ),
        "retained_token_count": {
            "mean": float(np.mean(retained_tokens)),
            "max": int(max(retained_tokens)),
        },
        "represented_source_item_count": {
            "mean": float(np.mean(source_coverage)),
            "max": int(max(source_coverage)),
        },
        "oldest_source_age": _percentiles(oldest_source_age),
        "representation_difference": {
            "relative_l2_mean": float(np.mean(relative_l2)),
            "cosine_similarity_mean": float(np.mean(cosine_similarity)),
        },
        "maintenance_counts": dict(sorted(maintenance_counts.items())),
        "inputs": {
            "manifest": str(manifest_path.resolve()),
            "context_bank": str(context_bank_path.resolve()),
            "context_bank_bytes": context_bank_path.stat().st_size,
            "program_config": str(program_config.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile tiered memory on an existing anchor+sliding context bank."
    )
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--context-bank", required=True, type=pathlib.Path)
    parser.add_argument("--program-config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    payload = profile_tiered_contexts(
        manifest_path=args.manifest,
        context_bank_path=args.context_bank,
        program_config=args.program_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
