from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "finalize_pi05_baseline_training.py"
SPEC = importlib.util.spec_from_file_location("finalize_pi05_baseline_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inputs(tmp_path: Path) -> argparse.Namespace:
    checkpoint = tmp_path / "experiment" / "279"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    (checkpoint / "_CHECKPOINT_METADATA").write_text("committed\n", encoding="utf-8")
    log = tmp_path / "training.log"
    log.write_text(
        "Finished saving checkpoint (finalized tmp dir) to "
        f"`{checkpoint.resolve()}`.\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        checkpoint=checkpoint,
        optimizer_updates=10,
        accumulate_steps=28,
        training_log=log,
    )


def test_accepts_committed_checkpoint_at_exact_budget(tmp_path: Path) -> None:
    evidence = MODULE._validate_committed_checkpoint(_inputs(tmp_path))

    assert evidence["checkpoint_step"] == 279
    assert evidence["checkpoint_commit_verified"] is True
    assert len(evidence["checkpoint_metadata_sha256"]) == 64


def test_rejects_checkpoint_from_wrong_training_budget(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    args.optimizer_updates = 11

    with pytest.raises(ValueError, match="requested training budget"):
        MODULE._validate_committed_checkpoint(args)


def test_rejects_uncommitted_checkpoint_directory(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    (args.checkpoint / "_CHECKPOINT_METADATA").unlink()

    with pytest.raises(ValueError, match="committed Orbax metadata"):
        MODULE._validate_committed_checkpoint(args)


def test_rejects_checkpoint_without_completed_save_log(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    args.training_log.write_text("save started but did not finish\n", encoding="utf-8")

    with pytest.raises(ValueError, match="save completed"):
        MODULE._validate_committed_checkpoint(args)
