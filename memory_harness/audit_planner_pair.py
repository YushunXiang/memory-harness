from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections.abc import Sequence
from typing import Any


def _json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_pair(
    key_data: pathlib.Path,
    no_key_data: pathlib.Path,
) -> dict[str, Any]:
    key_samples = _json(key_data)
    no_key_samples = _json(no_key_data)
    if not isinstance(key_samples, list) or not isinstance(no_key_samples, list):
        raise ValueError("planner datasets must be JSON lists")
    if len(key_samples) != len(no_key_samples) or not key_samples:
        raise ValueError("planner datasets must have the same non-zero sample count")

    key_labels = [sample["messages"][-1]["content"] for sample in key_samples]
    no_key_labels = [sample["messages"][-1]["content"] for sample in no_key_samples]
    if key_labels != no_key_labels:
        mismatch = next(
            index
            for index, (key_label, no_key_label) in enumerate(
                zip(key_labels, no_key_labels, strict=True)
            )
            if key_label != no_key_label
        )
        raise ValueError(f"planner labels differ at sample {mismatch}")

    key_global_tasks = [
        sample["messages"][1]["content"].splitlines()[0] for sample in key_samples
    ]
    no_key_global_tasks = [
        sample["messages"][1]["content"].splitlines()[0] for sample in no_key_samples
    ]
    if key_global_tasks != no_key_global_tasks:
        raise ValueError("planner global tasks are not paired")
    if any(len(sample["images"]) != 1 for sample in no_key_samples):
        raise ValueError("every no-key sample must contain exactly one current image")
    expected_key_counts = [index % 6 + 1 for index in range(len(key_samples))]
    actual_key_counts = [len(sample["images"]) for sample in key_samples]
    if actual_key_counts != expected_key_counts:
        raise ValueError(
            "key samples must accumulate one ordered image per completed stage"
        )

    return {
        "schema_version": 1,
        "status": "paired",
        "sample_count": len(key_samples),
        "labels_identical": True,
        "global_tasks_identical": True,
        "key_image_counts_per_episode": [1, 2, 3, 4, 5, 6],
        "no_key_images_per_sample": 1,
        "key_data": str(key_data.resolve()),
        "key_data_sha256": _sha256(key_data),
        "no_key_data": str(no_key_data.resolve()),
        "no_key_data_sha256": _sha256(no_key_data),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paired Mem-0 key/no-key SFT data."
    )
    parser.add_argument("--key-data", type=pathlib.Path, required=True)
    parser.add_argument("--no-key-data", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_pair(args.key_data.resolve(), args.no_key_data.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
