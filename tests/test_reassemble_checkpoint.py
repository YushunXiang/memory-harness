from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from memory_harness.reassemble_checkpoint import reassemble_checkpoint


def _fixture(tmp_path: Path, parts: list[bytes]) -> tuple[Path, Path, Path]:
    name = "model.pt"
    for index, value in enumerate(parts):
        (tmp_path / f"{name}.part{index:02d}").write_bytes(value)
    combined = b"".join(parts)
    checksum = tmp_path / f"{name}.sha256"
    checksum.write_text(
        f"{hashlib.sha256(combined).hexdigest()}  {name}\n", encoding="utf-8"
    )
    return checksum, tmp_path / name, tmp_path


def test_reassemble_checkpoint_orders_parts_and_verifies_digest(tmp_path: Path) -> None:
    checksum, output, parts_dir = _fixture(tmp_path, [b"first", b"second", b"third"])

    result = reassemble_checkpoint(
        parts_dir=parts_dir, checksum_file=checksum, output=output
    )

    assert output.read_bytes() == b"firstsecondthird"
    assert result["schema_version"] == "memory_harness.reassembled_checkpoint/v1"
    assert result["output_size_bytes"] == len(b"firstsecondthird")
    assert [Path(row["path"]).name for row in result["parts"]] == [
        "model.pt.part00",
        "model.pt.part01",
        "model.pt.part02",
    ]


def test_reassemble_checkpoint_reuses_only_verified_output(tmp_path: Path) -> None:
    checksum, output, parts_dir = _fixture(tmp_path, [b"a", b"b"])
    output.write_bytes(b"ab")

    result = reassemble_checkpoint(
        parts_dir=parts_dir, checksum_file=checksum, output=output
    )

    assert result["sha256"] == hashlib.sha256(b"ab").hexdigest()


def test_reassemble_checkpoint_rejects_missing_part(tmp_path: Path) -> None:
    checksum, output, parts_dir = _fixture(tmp_path, [b"a", b"b", b"c"])
    (tmp_path / "model.pt.part01").unlink()

    with pytest.raises(ValueError, match="not contiguous"):
        reassemble_checkpoint(
            parts_dir=parts_dir, checksum_file=checksum, output=output
        )


def test_reassemble_checkpoint_rejects_wrong_existing_output(tmp_path: Path) -> None:
    checksum, output, parts_dir = _fixture(tmp_path, [b"a", b"b"])
    output.write_bytes(b"wrong")

    with pytest.raises(FileExistsError, match="wrong sha256"):
        reassemble_checkpoint(
            parts_dir=parts_dir, checksum_file=checksum, output=output
        )
