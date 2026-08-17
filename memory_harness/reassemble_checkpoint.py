from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tempfile
from collections.abc import Sequence
from typing import Any


SCHEMA_VERSION = "memory_harness.reassembled_checkpoint/v1"
_PART_SUFFIX = re.compile(r"\.part(?P<index>[0-9]+)$")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_checksum(checksum_file: pathlib.Path) -> tuple[str, str]:
    fields = checksum_file.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or len(fields[0]) != 64:
        raise ValueError(f"invalid sha256 file: {checksum_file}")
    try:
        int(fields[0], 16)
    except ValueError as exc:
        raise ValueError(f"invalid sha256 digest: {checksum_file}") from exc
    return fields[0].lower(), fields[1].removeprefix("*")


def _ordered_parts(parts_dir: pathlib.Path, output_name: str) -> list[pathlib.Path]:
    indexed: list[tuple[int, pathlib.Path]] = []
    for path in parts_dir.glob(f"{output_name}.part*"):
        match = _PART_SUFFIX.search(path.name)
        if match is not None and path.is_file():
            indexed.append((int(match.group("index")), path))
    indexed.sort()
    if not indexed:
        raise FileNotFoundError(f"no split parts found for {output_name} in {parts_dir}")
    indices = [index for index, _ in indexed]
    if indices != list(range(len(indices))):
        raise ValueError(f"checkpoint parts are not contiguous from zero: {indices}")
    return [path for _, path in indexed]


def reassemble_checkpoint(
    *,
    parts_dir: pathlib.Path,
    checksum_file: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    parts_dir = parts_dir.resolve()
    checksum_file = checksum_file.resolve()
    output = output.resolve()
    expected_sha256, expected_name = _expected_checksum(checksum_file)
    if output.name != expected_name:
        raise ValueError(
            f"output filename {output.name!r} does not match checksum target "
            f"{expected_name!r}"
        )
    parts = _ordered_parts(parts_dir, expected_name)
    if output.exists():
        actual_sha256 = _sha256(output)
        if actual_sha256 != expected_sha256:
            raise FileExistsError(
                f"existing output has wrong sha256: {output} ({actual_sha256})"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as temporary:
            temporary_path = pathlib.Path(temporary.name)
            try:
                for part in parts:
                    with part.open("rb") as source:
                        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            temporary.write(chunk)
                            digest.update(chunk)
                temporary.flush()
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            temporary_path.unlink(missing_ok=True)
            raise ValueError(
                f"reassembled checkpoint sha256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        temporary_path.replace(output)

    return {
        "schema_version": SCHEMA_VERSION,
        "output": str(output),
        "output_size_bytes": output.stat().st_size,
        "sha256": expected_sha256,
        "parts": [
            {"path": str(path), "size_bytes": path.stat().st_size} for path in parts
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reassemble and verify a byte-split released checkpoint."
    )
    parser.add_argument("--parts-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checksum-file", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = reassemble_checkpoint(
        parts_dir=args.parts_dir,
        checksum_file=args.checksum_file,
        output=args.output,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
