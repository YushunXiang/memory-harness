from __future__ import annotations

import json
import pathlib

import cv2
import h5py
import numpy as np

from memory_harness.convert_rmbench_hdf5 import convert


class FakeDataset:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.episode_lengths: list[int] = []
        self._current_length = 0

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)
        self._current_length += 1

    def save_episode(self) -> None:
        self.episode_lengths.append(self._current_length)
        self._current_length = 0


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _write_episode(path: pathlib.Path, offset: float) -> None:
    states = np.arange(42, dtype=np.float32).reshape(3, 14) + offset
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[..., 0] = 200
    encoded = [_jpeg(image) for _ in range(3)]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("joint_action/vector", data=states)
        for camera in ("head_camera", "left_camera", "right_camera"):
            handle.create_dataset(
                f"observation/{camera}/rgb",
                data=np.asarray(encoded, dtype=f"S{max(map(len, encoded))}"),
            )


def test_convert_builds_openpi_dataset_and_norm_stats(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_episode(source / "episode0.hdf5", 0.0)
    _write_episode(source / "episode1.hdf5", 100.0)
    config = tmp_path / "task.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "memory_harness.task/v1",
                "task_name": "demo",
                "task_config": "demo_clean",
                "tmc": "M(1)",
                "source_dir": "source",
                "repo_id": "local/demo",
                "asset_id": "demo",
                "prompt": "Remember the initial location.",
                "max_steps": 500,
                "paired_layout_protocol": False,
                "fps": 30,
                "expected_episodes": 2,
                "camera_map": {
                    "head_camera": "cam_high",
                    "left_camera": "cam_left_wrist",
                    "right_camera": "cam_right_wrist",
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = FakeDataset()

    manifest = convert(
        config,
        hf_lerobot_home=tmp_path / "lerobot",
        assets_dir=tmp_path / "assets",
        dataset_factory=lambda features: dataset,
    )

    assert manifest["num_episodes"] == 2
    assert manifest["num_frames"] == 6
    assert manifest["state_dim"] == 14
    assert dataset.episode_lengths == [3, 3]
    assert dataset.frames[0]["task"] == "Remember the initial location."
    np.testing.assert_allclose(
        dataset.frames[0]["action"], np.arange(14, 28, dtype=np.float32)
    )
    assert dataset.frames[0]["observation.images.cam_high"].shape == (8, 10, 3)
    stats = json.loads(
        (tmp_path / "assets/demo/norm_stats.json").read_text(encoding="utf-8")
    )
    assert set(stats["norm_stats"]) == {"state", "actions"}


def test_convert_rejects_incomplete_episode_set(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_episode(source / "episode0.hdf5", 0.0)
    config = tmp_path / "task.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "memory_harness.task/v1",
                "task_name": "demo",
                "task_config": "demo_clean",
                "tmc": "M(1)",
                "source_dir": "source",
                "repo_id": "local/demo",
                "asset_id": "demo",
                "prompt": "Do it.",
                "max_steps": 500,
                "paired_layout_protocol": False,
                "fps": 30,
                "expected_episodes": 2,
                "camera_map": {"head_camera": "cam_high"},
            }
        ),
        encoding="utf-8",
    )

    try:
        convert(
            config,
            hf_lerobot_home=tmp_path / "lerobot",
            assets_dir=tmp_path / "assets",
            dataset_factory=lambda features: FakeDataset(),
        )
    except ValueError as error:
        assert "Expected 2 HDF5 episodes" in str(error)
    else:
        raise AssertionError("expected incomplete dataset to be rejected")
