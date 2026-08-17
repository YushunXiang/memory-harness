from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from memory_harness.training_provenance import parent_training_evidence


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write provenance for a native no-memory π0.5 baseline."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--task-config", type=pathlib.Path, required=True)
    parser.add_argument("--task-template", type=pathlib.Path, required=True)
    parser.add_argument("--initial-weight-params", type=pathlib.Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulate-steps", type=int, required=True)
    parser.add_argument("--optimizer-updates", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--training-log", type=pathlib.Path, required=True)
    return parser.parse_args()


def _validate_committed_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.name.isdigit():
        raise ValueError(f"checkpoint step is not numeric: {checkpoint}")
    checkpoint_step = int(checkpoint.name)
    expected_step = args.optimizer_updates * args.accumulate_steps - 1
    if checkpoint_step != expected_step:
        raise ValueError(
            "checkpoint step does not match the requested training budget: "
            f"expected={expected_step}, actual={checkpoint_step}"
        )
    for item in ("params", "assets"):
        if not (checkpoint / item).is_dir():
            raise ValueError(f"checkpoint has no {item}: {checkpoint}")
    metadata = checkpoint / "_CHECKPOINT_METADATA"
    if not metadata.is_file() or metadata.stat().st_size == 0:
        raise ValueError(f"checkpoint has no committed Orbax metadata: {checkpoint}")
    if not args.training_log.is_file():
        raise ValueError(f"missing training input: {args.training_log}")
    save_marker = (
        "Finished saving checkpoint (finalized tmp dir) to "
        f"`{checkpoint}`."
    )
    if save_marker not in args.training_log.read_text(encoding="utf-8"):
        raise ValueError(
            "training log does not prove the final checkpoint save completed: "
            f"{checkpoint}"
        )
    return {
        "checkpoint_step": checkpoint_step,
        "checkpoint_metadata_sha256": _sha256(metadata),
        "checkpoint_commit_verified": True,
    }


def main() -> int:
    args = parse_args()
    checkpoint_evidence = _validate_committed_checkpoint(args)
    if not args.initial_weight_params.is_dir():
        raise ValueError(
            f"initial weight params do not exist: {args.initial_weight_params}"
        )
    parent_evidence = parent_training_evidence(args.initial_weight_params)
    for path in (args.task_config, args.task_template, args.training_log):
        if not path.is_file():
            raise ValueError(f"missing training input: {path}")
    payload = {
        "schema_version": "memory_harness.training/v1",
        "config": args.config,
        "task_config": str(args.task_config.resolve()),
        "task_config_sha256": _sha256(args.task_config),
        "task_template": str(args.task_template.resolve()),
        "task_template_sha256": _sha256(args.task_template),
        "initial_weight_params": str(args.initial_weight_params.resolve()),
        "program": "native_none",
        "batch_size": args.batch_size,
        "accumulate_steps": args.accumulate_steps,
        "effective_batch": args.batch_size * args.accumulate_steps,
        "optimizer_updates": args.optimizer_updates,
        "learning_rate": args.learning_rate,
        "training_log": str(args.training_log.resolve()),
        "training_log_sha256": _sha256(args.training_log),
        "memory_enabled": False,
        **checkpoint_evidence,
        **parent_evidence,
    }
    output = args.checkpoint / "memory_training_manifest.json"
    if output.exists():
        raise FileExistsError(f"training manifest already exists: {output}")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
