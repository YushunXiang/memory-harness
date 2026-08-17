from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
import math
import pathlib
from typing import Any

import h5py
import numpy as np

from memory_harness.convert_rmbench_hdf5 import _decode_image
from memory_harness.tasks import load_task_spec


ScoredPatch = tuple[float, int, int, int]
PatchIndex = tuple[int, int, int]

OFFICIAL_REPOSITORY = "https://github.com/RoboMME/robomme_policy_learning"
OFFICIAL_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
OFFICIAL_MEM_BUFFER_SHA256 = (
    "ed37406c663cc9be9035cbadb2b961ad2babcc26d7fb8883535aaff1698efef7"
)
OFFICIAL_DATA_UTILS_SHA256 = (
    "dda1583743528403aa97a4bde8c0305deacfb5a618c9c61937703e59ae76d27a"
)


def even_sampling_indices(step_index: int, max_frames: int) -> list[int]:
    """Mirror RoboMME's inclusive uniform frame sampler exactly."""
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if step_index < max_frames:
        return list(range(step_index + 1))
    return np.linspace(0, step_index, max_frames, dtype=np.int32).tolist()


def _patch_change_scores(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_array = np.asarray(previous, dtype=np.uint8)
    current_array = np.asarray(current, dtype=np.uint8)
    if previous_array.shape != current_array.shape:
        raise ValueError("previous and current image batches must have equal shape")
    if previous_array.ndim != 4 or previous_array.shape[-1] != 3:
        raise ValueError("image batches must have shape [views,height,width,3]")
    views, height, width, channels = previous_array.shape
    if height % 8 or width % 8:
        raise ValueError("image height and width must be divisible by 8")
    patch_height = height // 8
    patch_width = width // 8

    def tokenize(images: np.ndarray) -> np.ndarray:
        normalized = images.astype(np.float32) / 255.0 * 2.0 - 1.0
        return (
            normalized.reshape(
                views,
                8,
                patch_height,
                8,
                patch_width,
                channels,
            )
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(views, 64, -1)
        )

    return np.mean(np.abs(tokenize(previous_array) - tokenize(current_array)), axis=-1)


def score_token_drop_episode(
    images: np.ndarray,
    *,
    stride: int = 8,
    difference_threshold: float = 1e-4,
) -> list[ScoredPatch]:
    """Generate RoboMME TokenDrop candidates before its bounded heap."""
    image_array = np.asarray(images, dtype=np.uint8)
    if image_array.ndim != 5 or image_array.shape[-1] != 3:
        raise ValueError("images must have shape [time,views,height,width,3]")
    if not len(image_array):
        raise ValueError("images must contain at least one frame")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if difference_threshold < 0:
        raise ValueError("difference_threshold must be non-negative")

    candidates: list[ScoredPatch] = []
    num_views = image_array.shape[1]
    for patch_index in range(64):
        for view_index in range(num_views):
            candidates.append((1000.0, 0, view_index, patch_index))

    last_scored_frame = -1
    for step_index in range(len(image_array)):
        if step_index != last_scored_frame + stride:
            continue
        previous_index = max(0, last_scored_frame)
        differences = _patch_change_scores(
            image_array[previous_index], image_array[step_index]
        )
        for view_index in range(num_views):
            for patch_index in range(64):
                score = float(differences[view_index, patch_index])
                if score < difference_threshold:
                    continue
                candidates.append((score, step_index, view_index, patch_index))
        last_scored_frame += stride
    return candidates


def _bounded_heap(
    candidates: list[ScoredPatch],
    *,
    kept_size: int,
    through_step: int | None = None,
) -> list[ScoredPatch]:
    if kept_size <= 0:
        raise ValueError("kept_size must be positive")
    heap: list[ScoredPatch] = []
    for item in candidates:
        if through_step is not None and item[1] > through_step:
            continue
        heapq.heappush(heap, item)
        if len(heap) > kept_size:
            heapq.heappop(heap)
    return heap


def select_token_drop_indices(
    candidates: list[ScoredPatch],
    *,
    query_step: int,
    token_budget: int = 512,
    kept_size: int = 2048,
    offline_full_episode_heap: bool,
) -> list[PatchIndex]:
    """Mirror online or released offline-precompute TokenDrop selection."""
    if query_step < 0:
        raise ValueError("query_step must be non-negative")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    heap = _bounded_heap(
        candidates,
        kept_size=kept_size,
        through_step=None if offline_full_episode_heap else query_step,
    )
    ordered = sorted(heap)
    eligible = [item for item in ordered if item[1] <= query_step]
    selected = eligible[-token_budget:]
    return sorted((step, view, patch) for _, step, view, patch in selected)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{int(quantile * 100):02d}": float(np.quantile(array, quantile))
        for quantile in (0.0, 0.5, 0.9, 0.99, 1.0)
    }


def _counter_quantiles(counter: Counter[int]) -> dict[str, float]:
    if not counter:
        return {}
    values = np.fromiter(sorted(counter), dtype=np.int64)
    counts = np.fromiter((counter[int(value)] for value in values), dtype=np.int64)
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])
    result: dict[str, float] = {}
    for quantile in (0.0, 0.5, 0.9, 0.99, 1.0):
        rank = max(1, math.ceil(quantile * total))
        index = int(np.searchsorted(cumulative, rank, side="left"))
        result[f"q{int(quantile * 100):02d}"] = float(values[index])
    return result


def _normalized_frame_entropy(indices: set[PatchIndex]) -> float:
    counts = np.asarray(list(Counter(item[0] for item in indices).values()), dtype=float)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / np.sum(counts)
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(counts)))


def profile_episode_candidates(
    candidates: list[ScoredPatch],
    *,
    num_steps: int,
    token_budget: int = 512,
    frame_tokens: int = 16,
    kept_size: int = 2048,
) -> dict[str, Any]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if token_budget % frame_tokens:
        raise ValueError("token_budget must be divisible by frame_tokens")
    max_frames = token_budget // frame_tokens

    causal_counts: list[float] = []
    offline_counts: list[float] = []
    causal_unique_frames: list[float] = []
    frame_jaccards: list[float] = []
    token_jaccards: list[float] = []
    outside_recent_fractions: list[float] = []
    entropy_values: list[float] = []
    missing_due_to_future: list[float] = []
    lag_counts: Counter[int] = Counter()
    exact_token_parity = 0
    exact_frame_parity = 0
    offline_underfilled = 0
    queries = 0
    selected_future_tokens = 0

    for query_step in range(max_frames, num_steps):
        frame_indices = set(even_sampling_indices(query_step, max_frames))
        causal = set(
            select_token_drop_indices(
                candidates,
                query_step=query_step,
                token_budget=token_budget,
                kept_size=kept_size,
                offline_full_episode_heap=False,
            )
        )
        offline = set(
            select_token_drop_indices(
                candidates,
                query_step=query_step,
                token_budget=token_budget,
                kept_size=kept_size,
                offline_full_episode_heap=True,
            )
        )
        causal_frames = {item[0] for item in causal}
        token_union = causal | offline
        frame_union = frame_indices | causal_frames
        causal_counts.append(float(len(causal)))
        offline_counts.append(float(len(offline)))
        causal_unique_frames.append(float(len(causal_frames)))
        token_jaccards.append(len(causal & offline) / max(len(token_union), 1))
        frame_jaccards.append(len(frame_indices & causal_frames) / max(len(frame_union), 1))
        missing_due_to_future.append(float(len(causal - offline)))
        recent_start = max(0, query_step - max_frames + 1)
        outside_recent_fractions.append(
            sum(item[0] < recent_start for item in causal) / max(len(causal), 1)
        )
        entropy_values.append(_normalized_frame_entropy(causal))
        lag_counts.update(query_step - item[0] for item in causal)
        exact_token_parity += int(causal == offline)
        exact_frame_parity += int(frame_indices == causal_frames)
        offline_underfilled += int(len(offline) < token_budget)
        selected_future_tokens += sum(item[0] > query_step for item in offline)
        queries += 1

    if not queries:
        raise ValueError("episode does not contain a full-history query")
    return {
        "query_count": queries,
        "causal_selected_token_count": causal_counts,
        "offline_selected_token_count": offline_counts,
        "causal_unique_frame_count": causal_unique_frames,
        "framesamp_tokendrop_frame_jaccard": frame_jaccards,
        "causal_offline_token_jaccard": token_jaccards,
        "causal_tokens_missing_offline": missing_due_to_future,
        "causal_outside_latest_frame_window_fraction": outside_recent_fractions,
        "causal_frame_patch_entropy": entropy_values,
        "causal_lag_counts": dict(lag_counts),
        "exact_token_parity_count": exact_token_parity,
        "exact_frame_set_parity_count": exact_frame_parity,
        "offline_underfilled_count": offline_underfilled,
        "selected_future_token_count": selected_future_tokens,
    }


def summarize_episode_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    if not profiles:
        raise ValueError("at least one episode profile is required")
    query_count = sum(int(profile["query_count"]) for profile in profiles)
    lag_counts: Counter[int] = Counter()
    for profile in profiles:
        lag_counts.update(
            {int(key): int(value) for key, value in profile["causal_lag_counts"].items()}
        )

    def concatenate(key: str) -> list[float]:
        return [float(value) for profile in profiles for value in profile[key]]

    def total(key: str) -> int:
        return sum(int(profile[key]) for profile in profiles)

    causal_counts = concatenate("causal_selected_token_count")
    offline_counts = concatenate("offline_selected_token_count")
    missing = concatenate("causal_tokens_missing_offline")
    token_jaccards = concatenate("causal_offline_token_jaccard")
    frame_jaccards = concatenate("framesamp_tokendrop_frame_jaccard")
    outside_recent = concatenate("causal_outside_latest_frame_window_fraction")
    entropy = concatenate("causal_frame_patch_entropy")
    unique_frames = concatenate("causal_unique_frame_count")
    exact_token_parity = total("exact_token_parity_count")
    return {
        "episode_count": len(profiles),
        "full_history_query_count": query_count,
        "causal_selected_token_count_quantiles": _quantiles(causal_counts),
        "offline_selected_token_count_quantiles": _quantiles(offline_counts),
        "offline_underfilled_query_fraction": total("offline_underfilled_count")
        / query_count,
        "causal_offline_exact_token_parity_fraction": exact_token_parity
        / query_count,
        "causal_offline_token_jaccard_mean": float(np.mean(token_jaccards)),
        "causal_offline_token_jaccard_quantiles": _quantiles(token_jaccards),
        "causal_tokens_missing_offline_mean": float(np.mean(missing)),
        "causal_tokens_missing_offline_quantiles": _quantiles(missing),
        "selected_future_token_count": total("selected_future_token_count"),
        "framesamp_tokendrop_exact_frame_set_fraction": total(
            "exact_frame_set_parity_count"
        )
        / query_count,
        "framesamp_tokendrop_frame_jaccard_mean": float(np.mean(frame_jaccards)),
        "framesamp_tokendrop_frame_jaccard_quantiles": _quantiles(frame_jaccards),
        "tokendrop_unique_frame_count_quantiles": _quantiles(unique_frames),
        "tokendrop_outside_latest_32_frames_mean": float(np.mean(outside_recent)),
        "tokendrop_frame_patch_entropy_mean": float(np.mean(entropy)),
        "tokendrop_selected_patch_lag_quantiles": _counter_quantiles(lag_counts),
    }


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(paths: list[pathlib.Path]) -> dict[str, Any]:
    rows = [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    ]
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(
            f"{row['name']}\0{row['size_bytes']}\0{row['sha256']}\n".encode()
        )
    return {
        "episode_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": rows,
    }


def build_report(
    task_config_path: pathlib.Path,
    *,
    camera: str = "head_camera",
    token_budget: int = 512,
    frame_tokens: int = 16,
    kept_size: int = 2048,
    stride: int = 8,
) -> dict[str, Any]:
    task = load_task_spec(task_config_path)
    source_files = sorted(
        task.source_dir.glob("episode*.hdf5"),
        key=lambda path: int(path.stem.removeprefix("episode")),
    )
    if len(source_files) != task.expected_episodes:
        raise ValueError(
            f"expected {task.expected_episodes} source episodes, found {len(source_files)}"
        )
    profiles: list[dict[str, Any]] = []
    episode_lengths: list[float] = []
    image_shape: tuple[int, ...] | None = None
    dataset_path = f"observation/{camera}/rgb"
    for source_file in source_files:
        with h5py.File(source_file, "r") as handle:
            if dataset_path not in handle:
                raise KeyError(f"{source_file} is missing {dataset_path}")
            encoded = handle[dataset_path]
            decoded = [
                _decode_image(value, context=f"{source_file.name}:{camera}:{index}")
                for index, value in enumerate(encoded)
            ]
        images = np.asarray(decoded, dtype=np.uint8)[:, None, ...]
        current_shape = tuple(images.shape[2:])
        if image_shape is None:
            image_shape = current_shape
        elif current_shape != image_shape:
            raise ValueError("all source images must have the same shape")
        candidates = score_token_drop_episode(images, stride=stride)
        profiles.append(
            profile_episode_candidates(
                candidates,
                num_steps=len(images),
                token_budget=token_budget,
                frame_tokens=frame_tokens,
                kept_size=kept_size,
            )
        )
        episode_lengths.append(float(len(images)))

    summary = summarize_episode_profiles(profiles)
    mismatch_confirmed = summary["causal_offline_exact_token_parity_fraction"] < 1.0
    distinct_selectors = (
        summary["framesamp_tokendrop_exact_frame_set_fraction"] < 0.01
        and summary["framesamp_tokendrop_frame_jaccard_mean"] < 0.5
    )
    return {
        "schema_version": "memory_harness.robomme_patch_retriever_profile/v1",
        "method_scope": (
            "Source-equivalent RGB selectors on local RMBench demonstrations; "
            "this does not measure policy utility or reproduce SigLIP patch features"
        ),
        "official_source": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "mem_buffer_sha256": OFFICIAL_MEM_BUFFER_SHA256,
            "data_utils_sha256": OFFICIAL_DATA_UTILS_SHA256,
        },
        "inputs": {
            "task_config": str(task_config_path.resolve()),
            "task_config_sha256": _sha256(task_config_path.resolve()),
            "task_name": task.task_name,
            "task_memory_complexity": task.tmc,
            "camera": camera,
            "dataset_path": dataset_path,
            "source_identity": _source_identity(source_files),
            "episode_length_quantiles": _quantiles(episode_lengths),
            "total_frames": int(sum(episode_lengths)),
            "decoded_image_shape": list(image_shape or ()),
        },
        "selector_contract": {
            "token_budget": token_budget,
            "framesamp_tokens_per_frame": frame_tokens,
            "framesamp_max_frames": token_budget // frame_tokens,
            "tokendrop_grid": "8x8",
            "tokendrop_stride": stride,
            "tokendrop_difference_threshold": 1e-4,
            "tokendrop_heap_kept_size": kept_size,
            "query_scope": "all step indices >= framesamp_max_frames",
        },
        "results": summary,
        "decision": {
            "selectors_behaviorally_distinct_on_rmbench": distinct_selectors,
            "offline_online_membership_mismatch_confirmed": mismatch_confirmed,
            "future_tokens_returned": summary["selected_future_token_count"] > 0,
            "plugin_action": (
                "Keep FrameSamp and TokenDrop as separate retriever operators over one "
                "typed perceptual-patch store. Require prefix-causal heap construction "
                "for both training and deployment; reproduce released full-episode "
                "precompute only as an explicitly named checkpoint-compatibility ablation."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--frame-tokens", type=int, default=16)
    parser.add_argument("--kept-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    report = build_report(
        args.task_config,
        camera=args.camera,
        token_budget=args.token_budget,
        frame_tokens=args.frame_tokens,
        kept_size=args.kept_size,
        stride=args.stride,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
