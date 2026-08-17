from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "memory-harness"
    / "scripts"
    / "run_pi05_cover_blocks_planner_condition.sh"
)


@pytest.mark.parametrize(
    ("condition", "architecture", "served_model"),
    [
        ("key", "key", "mem0-cover-blocks-key-planner"),
        ("no_key", "planner_no_key", "mem0-cover-blocks-no-key-planner"),
    ],
)
def test_planner_condition_dry_run_binds_model_and_architecture(
    tmp_path: Path,
    condition: str,
    architecture: str,
    served_model: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    base_index = tmp_path / "base.index.json"
    base_index.write_text("{}\n", encoding="utf-8")
    pair_manifest = tmp_path / "pair.json"
    pair_manifest.write_text("{}\n", encoding="utf-8")

    environment = {
        **os.environ,
        "PYTHON": sys.executable,
        "SERVER_PYTHON": sys.executable,
        "PLANNER_BASE_INDEX": str(base_index),
        "TRAINING_PAIR_MANIFEST": str(pair_manifest),
    }
    for name in ("KEY", "NO_KEY"):
        model = tmp_path / f"{name.lower()}_model"
        model.mkdir()
        (model / "model.safetensors.index.json").write_text(
            "{}\n", encoding="utf-8"
        )
        adapter = tmp_path / f"{name.lower()}_adapter.safetensors"
        adapter.write_bytes(b"adapter")
        environment[f"{name}_MODEL"] = str(model)
        environment[f"{name}_ADAPTER"] = str(adapter)

    output_dir = tmp_path / "run"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--condition",
            condition,
            "--checkpoint",
            str(checkpoint),
            "--num-episodes",
            "3",
            "--seed-start",
            "100000",
            "--policy-seed-base",
            "120000",
            "--gpu-id",
            "1",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout
    assert f"--served-model-name {served_model}" in output
    assert f"MEMORY_PLANNER_MODEL={served_model}" in output
    assert f"cover_blocks {architecture}" in output
    assert "NUM_EPISODES=3" in output
    assert "SEED=100000" in output
    assert "POLICY_SEED_BASE=120000" in output
