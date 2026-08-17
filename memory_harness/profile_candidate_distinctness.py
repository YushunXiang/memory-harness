from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from memory_harness.calibrate_novelty import reconstruct_source_sequences
from memory_harness.config import ProgramSpec
from memory_harness.config import load_program_spec
from memory_harness.contracts import EpisodeOutcome
from memory_harness.contracts import MemoryStep
from memory_harness.registry import build_program


SCHEMA_VERSION = "memory_harness.candidate_distinctness/v3"


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validation_episode_ids(manifest_path: pathlib.Path) -> tuple[str, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_ids = manifest.get("validation_lerobot_episode_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("context manifest has no validation episode ids")
    episode_ids = tuple(str(value) for value in raw_ids)
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("validation episode ids must be unique")
    return episode_ids


def reconstruct_phase_sequences(
    manifest_path: pathlib.Path,
) -> dict[str, tuple[str, ...]]:
    """Align each recovered source latent with its deployment phase label."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("segments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("context manifest has no segments")
    rows_by_episode: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("context manifest segment must be an object")
        rows_by_episode[str(row["source_episode_id"])].append(row)

    output: dict[str, tuple[str, ...]] = {}
    for episode_id, episode_rows in sorted(rows_by_episode.items()):
        episode_rows.sort(key=lambda row: int(row["start_frame"]))
        # Source t is reconstructed from context row t+1, so the unavailable
        # final source and its phase label are excluded together.
        phases = tuple(str(row.get("phase_label", "")).strip() for row in episode_rows[:-1])
        if not phases:
            continue
        if any(not phase for phase in phases):
            raise ValueError(
                f"episode {episode_id!r} has empty deployment phase labels"
            )
        output[episode_id] = phases
    if not output:
        raise ValueError("no phase sequences could be reconstructed")
    return output


def _unsupported_reason(spec: ProgramSpec) -> str | None:
    if any(path.writer.type == "causal_kinematic_peak" for path in spec.paths):
        return "requires deployment robot_state; latent-only context bank is insufficient"
    if any(path.store.type == "verified_success_ring" for path in spec.paths):
        return "requires ordered cross-episode outcomes; episode-local comparison is invalid"
    return None


def _memory_fingerprint(
    observation: Mapping[str, Any],
) -> tuple[str, str, int, list[int] | None, list[str]]:
    present = {key for key in ("memory_tokens", "memory_mask") if key in observation}
    if not present:
        digest = hashlib.sha256(b"no-memory-input").hexdigest()
        return digest, digest, 0, None, []
    if present != {"memory_tokens", "memory_mask"}:
        raise ValueError("memory_tokens and memory_mask must be emitted together")
    tokens = np.asarray(observation["memory_tokens"], dtype=np.float32)
    mask = np.asarray(observation["memory_mask"], dtype=np.bool_)
    if tokens.ndim != 2 or mask.shape != (tokens.shape[0],):
        raise ValueError(
            f"invalid utilized memory layout: tokens={tokens.shape}, mask={mask.shape}"
        )
    tokens = np.ascontiguousarray(tokens)
    mask = np.ascontiguousarray(mask)
    layout = json.dumps(
        {"tokens_shape": list(tokens.shape), "mask_shape": list(mask.shape)},
        sort_keys=True,
    ).encode("utf-8")
    mask_digest = hashlib.sha256(layout + mask.tobytes()).hexdigest()
    output_digest = hashlib.sha256(
        layout + mask.tobytes() + tokens.tobytes()
    ).hexdigest()
    valid_token_digests = [
        hashlib.sha256(
            np.ascontiguousarray(tokens[index]).tobytes()
        ).hexdigest()
        for index in np.flatnonzero(mask)
    ]
    return (
        output_digest,
        mask_digest,
        int(mask.sum()),
        list(tokens.shape),
        valid_token_digests,
    )


def profile_program(
    spec: ProgramSpec,
    sequences: Mapping[str, np.ndarray],
    *,
    phase_sequences: Mapping[str, Sequence[str]],
    episode_ids: Sequence[str],
    warmup_steps: int,
    query_stride: int,
) -> dict[str, Any]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if query_stride <= 0:
        raise ValueError("query_stride must be positive")
    program = build_program(spec)
    fingerprints: dict[str, dict[str, Any]] = {}
    output_shapes: set[tuple[int, ...] | None] = set()
    for episode_id in episode_ids:
        if episode_id not in sequences:
            raise ValueError(f"validation episode is absent from source bank: {episode_id}")
        sequence = np.asarray(sequences[episode_id], dtype=np.float32)
        if sequence.ndim != 2 or not len(sequence):
            raise ValueError(
                f"episode {episode_id!r} must have shape [T, D], got {sequence.shape}"
            )
        if episode_id not in phase_sequences:
            raise ValueError(
                f"validation episode is absent from phase sequences: {episode_id}"
            )
        phases = tuple(str(value).strip() for value in phase_sequences[episode_id])
        if len(phases) != len(sequence):
            raise ValueError(
                f"episode {episode_id!r} phase/source lengths differ: "
                f"{len(phases)} vs {len(sequence)}"
            )
        if any(not phase for phase in phases):
            raise ValueError(
                f"episode {episode_id!r} has empty deployment phase labels"
            )
        program.reset(episode_id=episode_id)
        for step_index, vector in enumerate(sequence):
            has_paths = bool(program.paths)
            result = program.step(
                {},
                MemoryStep(
                    episode_id=episode_id,
                    step_index=step_index,
                    phase=phases[step_index],
                    source_tokens=vector[None, :] if has_paths else None,
                    source_mask=(
                        np.ones((1,), dtype=np.bool_) if has_paths else None
                    ),
                ),
            )
            if (
                step_index < warmup_steps
                or (step_index - warmup_steps) % query_stride
            ):
                continue
            output_digest, mask_digest, used_tokens, output_shape, token_digests = (
                _memory_fingerprint(result.observation)
            )
            query_id = f"{episode_id}:{step_index}"
            fingerprints[query_id] = {
                "output_sha256": output_digest,
                "mask_sha256": mask_digest,
                "used_token_count": used_tokens,
                "valid_token_sha256": token_digests,
            }
            output_shapes.add(
                None if output_shape is None else tuple(output_shape)
            )
        program.finish_episode(
            EpisodeOutcome(
                episode_id=episode_id,
                success=False,
                final_step_index=len(sequence) - 1,
            )
        )
    if not fingerprints:
        raise ValueError("candidate profile produced no sampled queries")
    ordered = [fingerprints[key] for key in sorted(fingerprints)]
    aggregate_digest = hashlib.sha256(
        json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "query_count": len(fingerprints),
        "output_shapes": [
            None if shape is None else list(shape)
            for shape in sorted(output_shapes, key=lambda value: str(value))
        ],
        "mean_used_token_count": float(
            np.mean([row["used_token_count"] for row in fingerprints.values()])
        ),
        "aggregate_output_sha256": aggregate_digest,
        "fingerprints": fingerprints,
    }


def compare_profiles(
    profiles: Mapping[str, Mapping[str, Any]], *, near_duplicate_threshold: float = 0.95
) -> dict[str, Any]:
    if len(profiles) < 2:
        raise ValueError("at least two supported candidates are required")
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in [0, 1]")
    pairwise: dict[str, dict[str, Any]] = {}
    exact_duplicates: list[list[str]] = []
    near_duplicates: list[dict[str, Any]] = []
    for left_name, right_name in itertools.combinations(sorted(profiles), 2):
        left = profiles[left_name]["fingerprints"]
        right = profiles[right_name]["fingerprints"]
        if set(left) != set(right):
            raise ValueError(
                f"candidate query identities differ: {left_name} vs {right_name}"
            )
        query_ids = sorted(left)
        output_matches = sum(
            left[key]["output_sha256"] == right[key]["output_sha256"]
            for key in query_ids
        )
        mask_matches = sum(
            left[key]["mask_sha256"] == right[key]["mask_sha256"]
            for key in query_ids
        )
        count_matches = sum(
            left[key]["used_token_count"] == right[key]["used_token_count"]
            for key in query_ids
        )
        token_multiset_jaccards: list[float] = []
        exact_token_multisets = 0
        for key in query_ids:
            left_tokens = collections.Counter(left[key]["valid_token_sha256"])
            right_tokens = collections.Counter(right[key]["valid_token_sha256"])
            exact_token_multisets += int(left_tokens == right_tokens)
            intersection = sum((left_tokens & right_tokens).values())
            union = sum((left_tokens | right_tokens).values())
            token_multiset_jaccards.append(1.0 if union == 0 else intersection / union)
        pair_name = f"{left_name}_vs_{right_name}"
        mean_jaccard = float(np.mean(token_multiset_jaccards))
        pairwise[pair_name] = {
            "left": left_name,
            "right": right_name,
            "query_count": len(query_ids),
            "exact_output_fraction": output_matches / len(query_ids),
            "exact_mask_fraction": mask_matches / len(query_ids),
            "exact_used_token_count_fraction": count_matches / len(query_ids),
            "exact_valid_token_multiset_fraction": (
                exact_token_multisets / len(query_ids)
            ),
            "mean_valid_token_multiset_jaccard": mean_jaccard,
            "median_valid_token_multiset_jaccard": float(
                np.median(token_multiset_jaccards)
            ),
        }
        if output_matches == len(query_ids):
            exact_duplicates.append([left_name, right_name])
        elif mean_jaccard >= near_duplicate_threshold:
            near_duplicates.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "mean_valid_token_multiset_jaccard": mean_jaccard,
                    "decision": "review_and_justify_or_remove_before_rollout",
                }
            )
    return {
        "pairwise": pairwise,
        "behaviorally_exact_duplicate_pairs": exact_duplicates,
        "near_duplicate_review_pairs": near_duplicates,
    }


def build_report(
    manifest_path: pathlib.Path,
    context_bank_path: pathlib.Path,
    config_dir: pathlib.Path,
    *,
    warmup_steps: int,
    query_stride: int,
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    sequences = reconstruct_source_sequences(manifest_path, context_bank_path)
    phase_sequences = reconstruct_phase_sequences(manifest_path)
    episode_ids = _validation_episode_ids(manifest_path)
    profiles: dict[str, dict[str, Any]] = {}
    exclusions: dict[str, str] = {}
    config_hashes: dict[str, str] = {}
    config_paths = sorted(config_dir.glob("fixed_*.json"))
    if not config_paths:
        raise ValueError(f"no fixed program configs found in {config_dir}")
    for config_path in config_paths:
        alias = config_path.stem.removeprefix("fixed_")
        spec = load_program_spec(config_path)
        reason = _unsupported_reason(spec)
        if reason is not None:
            exclusions[alias] = reason
            continue
        config_hashes[alias] = _sha256(config_path)
        profiles[alias] = profile_program(
            spec,
            sequences,
            phase_sequences=phase_sequences,
            episode_ids=episode_ids,
            warmup_steps=warmup_steps,
            query_stride=query_stride,
        )
    comparison = compare_profiles(
        profiles, near_duplicate_threshold=near_duplicate_threshold
    )
    compact_profiles = {
        name: {key: value for key, value in profile.items() if key != "fingerprints"}
        for name, profile in profiles.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "context_manifest": str(manifest_path.resolve()),
            "context_manifest_sha256": _sha256(manifest_path),
            "context_bank": str(context_bank_path.resolve()),
            "context_bank_sha256": _sha256(context_bank_path),
            "config_dir": str(config_dir.resolve()),
            "config_sha256": config_hashes,
            "split": "validation_lerobot_episode_ids",
            "phase_source": "context_manifest.segments.phase_label",
        },
        "protocol": {
            "episode_ids": list(episode_ids),
            "warmup_steps": warmup_steps,
            "query_stride": query_stride,
            "near_duplicate_threshold": near_duplicate_threshold,
            "phase_alignment": (
                "source t and phase t are aligned to causal context row t+1; "
                "the unavailable final source is excluded"
            ),
            "fingerprint": "sha256(memory layout + mask bytes + float32 token bytes)",
            "near_duplicate_metric": (
                "mean multiset Jaccard over per-token float32 SHA-256 identities"
            ),
            "phase": "episode",
            "read_write_order": "runtime read-before-write",
        },
        "supported_candidate_count": len(profiles),
        "excluded_candidates": exclusions,
        "profiles": compact_profiles,
        **comparison,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect behaviorally duplicate fixed memory programs on real contexts."
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--context-bank", type=pathlib.Path, required=True)
    parser.add_argument("--config-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=31)
    parser.add_argument("--query-stride", type=int, default=10)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        args.manifest,
        args.context_bank,
        args.config_dir,
        warmup_steps=args.warmup_steps,
        query_stride=args.query_stride,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "supported_candidate_count": report["supported_candidate_count"],
                "excluded_candidates": report["excluded_candidates"],
                "behaviorally_exact_duplicate_pairs": report[
                    "behaviorally_exact_duplicate_pairs"
                ],
                "near_duplicate_review_pairs": report[
                    "near_duplicate_review_pairs"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
