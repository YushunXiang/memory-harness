from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import os
import pathlib
import shutil
from typing import Any

import cv2
import h5py
import numpy as np

from memory_harness.tasks import TaskSpec
from memory_harness.tasks import load_task_spec


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_files(spec: TaskSpec) -> list[pathlib.Path]:
    files = sorted(
        spec.source_dir.glob("episode*.hdf5"),
        key=lambda path: int(path.stem.removeprefix("episode")),
    )
    if len(files) != spec.expected_episodes:
        raise ValueError(
            f"Expected {spec.expected_episodes} HDF5 episodes in {spec.source_dir}, found {len(files)}"
        )
    expected_names = [f"episode{index}.hdf5" for index in range(spec.expected_episodes)]
    actual_names = [path.name for path in files]
    if actual_names != expected_names:
        raise ValueError("RMBench episode files must be contiguous and zero-indexed")
    return files


def _decode_image(value: Any, *, context: str) -> np.ndarray:
    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {context}")
    # RMBench writes simulator RGB arrays with cv2.imencode and reads them back
    # with cv2.imdecode. Keeping the decoded channel values unchanged recovers
    # the simulator array used by the online policy.
    return np.asarray(image, dtype=np.uint8)


def _load_episode(path: pathlib.Path, spec: TaskSpec) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        states = np.asarray(handle["joint_action/vector"], dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 14:
            raise ValueError(
                f"{path} joint_action/vector must have shape [T,14], got {states.shape}"
            )
        if states.shape[0] < 2:
            raise ValueError(f"{path} must contain at least two frames")
        images: dict[str, list[np.ndarray]] = {}
        for source, target in spec.camera_map.items():
            dataset_path = f"observation/{source}/rgb"
            if dataset_path not in handle:
                raise KeyError(f"{path} is missing {dataset_path}")
            encoded_images = handle[dataset_path]
            if len(encoded_images) != len(states):
                raise ValueError(
                    f"{path} camera {source} length does not match state length"
                )
            images[target] = [
                _decode_image(value, context=f"{path.name}:{source}:{index}")
                for index, value in enumerate(encoded_images)
            ]

    actions = np.concatenate([states[1:], states[-1:]], axis=0)
    return {"states": states, "actions": actions, "images": images}


def _features(first_episode: dict[str, Any]) -> dict[str, Any]:
    joint_names = [f"joint_{index:02d}" for index in range(14)]
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [joint_names],
        },
        "action": {"dtype": "float32", "shape": (14,), "names": [joint_names]},
    }
    for target, images in first_episode["images"].items():
        height, width, channels = images[0].shape
        if channels != 3:
            raise ValueError(f"Camera {target} must contain RGB images")
        features[f"observation.images.{target}"] = {
            "dtype": "image",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def _create_dataset(
    *,
    repo_id: str,
    features: dict[str, Any],
    hf_lerobot_home: pathlib.Path,
    fps: int,
    overwrite: bool,
) -> Any:
    os.environ["HF_LEROBOT_HOME"] = str(hf_lerobot_home.resolve())
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output = (hf_lerobot_home / repo_id).resolve()
    root = hf_lerobot_home.resolve()
    if output == root or root not in output.parents:
        raise ValueError(f"repo_id must resolve below HF_LEROBOT_HOME: {repo_id!r}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"LeRobot dataset already exists: {output}")
        shutil.rmtree(output)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="aloha",
        features=features,
        use_videos=False,
        tolerance_s=0.0001,
        image_writer_processes=4,
        image_writer_threads=4,
        video_backend=None,
    )


def _norm(values: np.ndarray) -> dict[str, list[float]]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    return {
        "mean": np.mean(flat, axis=0).astype(float).tolist(),
        "std": np.std(flat, axis=0).astype(float).tolist(),
        "q01": np.quantile(flat, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(flat, 0.99, axis=0).astype(float).tolist(),
    }


def _delta_actions(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32).copy()
    delta_mask = np.asarray([True] * 6 + [False] + [True] * 6 + [False])
    result[:, delta_mask] -= states[:, delta_mask]
    return result


def convert(
    task_config: pathlib.Path,
    *,
    hf_lerobot_home: pathlib.Path,
    assets_dir: pathlib.Path,
    overwrite: bool = False,
    manifest_path: pathlib.Path | None = None,
    dataset_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    spec = load_task_spec(task_config)
    source_files = _source_files(spec)
    first = _load_episode(source_files[0], spec)
    features = _features(first)
    dataset = (
        dataset_factory(features)
        if dataset_factory is not None
        else _create_dataset(
            repo_id=spec.repo_id,
            features=features,
            hf_lerobot_home=hf_lerobot_home,
            fps=spec.fps,
            overwrite=overwrite,
        )
    )

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episode_rows: list[dict[str, Any]] = []
    for episode_index, source_path in enumerate(source_files):
        episode = first if episode_index == 0 else _load_episode(source_path, spec)
        states = episode["states"]
        actions = episode["actions"]
        for frame_index in range(len(states)):
            frame: dict[str, Any] = {
                "observation.state": states[frame_index],
                "action": actions[frame_index],
                "task": spec.prompt,
            }
            for target, images in episode["images"].items():
                frame[f"observation.images.{target}"] = images[frame_index]
            dataset.add_frame(frame)
        dataset.save_episode()
        all_states.append(states)
        all_actions.append(_delta_actions(states, actions))
        episode_rows.append(
            {
                "episode_index": episode_index,
                "frames": int(len(states)),
                "source": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        )

    state_array = np.concatenate(all_states, axis=0)
    action_array = np.concatenate(all_actions, axis=0)
    norm_path = assets_dir / spec.asset_id / "norm_stats.json"
    _write_json(
        norm_path,
        {"norm_stats": {"state": _norm(state_array), "actions": _norm(action_array)}},
    )
    manifest = {
        "schema_version": "memory_harness.rmbench_conversion/v1",
        "task_config": str(task_config.resolve()),
        "task_name": spec.task_name,
        "tmc": spec.tmc,
        "repo_id": spec.repo_id,
        "asset_id": spec.asset_id,
        "prompt": spec.prompt,
        "episodes": episode_rows,
        "num_episodes": len(episode_rows),
        "num_frames": int(len(state_array)),
        "state_dim": int(state_array.shape[1]),
        "camera_map": spec.camera_map,
        "norm_stats": str(norm_path.resolve()),
    }
    output_manifest = manifest_path or (
        assets_dir / spec.asset_id / "conversion_manifest.json"
    )
    _write_json(output_manifest, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an official RMBench HDF5 task to OpenPI LeRobot data"
    )
    parser.add_argument("--task-config", type=pathlib.Path, required=True)
    parser.add_argument("--hf-lerobot-home", type=pathlib.Path, required=True)
    parser.add_argument("--assets-dir", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = convert(
        args.task_config,
        hf_lerobot_home=args.hf_lerobot_home,
        assets_dir=args.assets_dir,
        overwrite=args.overwrite,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
