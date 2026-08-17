from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


SCHEMA_VERSION = "memory_harness.source_audited_candidates/v2"
_OPERATOR_ROLES = {
    "encoder",
    "writer",
    "store",
    "retriever",
    "utilizer",
    "lifecycle",
    "controller",
    "training",
}


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inside(root: pathlib.Path, relative: str, *, label: str) -> pathlib.Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes project root: {relative}")
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return path


def _executable_aliases(project_root: pathlib.Path) -> set[str]:
    architecture_dir = project_root / "configs" / "architectures"
    return {
        path.stem.removeprefix("fixed_")
        for path in architecture_dir.glob("fixed_*.json")
    }


def validate_source_audited_candidates(
    catalog_path: pathlib.Path,
    *,
    project_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    root = (
        project_root.resolve()
        if project_root is not None
        else pathlib.Path(__file__).resolve().parent.parent
    )
    if not catalog_path.is_relative_to(root):
        raise ValueError("candidate catalog must be inside the project root")
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("source-audited candidate catalog has an unsupported schema")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("source-audited candidate catalog is empty")

    executable_aliases = _executable_aliases(root)
    candidate_ids: set[str] = set()
    payload_families: set[str] = set()
    operator_names: set[tuple[str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate entries must be objects")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate id must be a non-empty string")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate source-audited candidate id: {candidate_id}")
        candidate_ids.add(candidate_id)
        payload_family = candidate.get("payload_family")
        if not isinstance(payload_family, str) or not payload_family:
            raise ValueError(f"candidate payload family is missing: {candidate_id}")
        if payload_family in payload_families:
            raise ValueError(f"duplicate source-audited payload family: {payload_family}")
        payload_families.add(payload_family)
        if candidate.get("implementation_status") != "not_executable":
            raise ValueError(
                f"source-audited candidate must remain not_executable: {candidate_id}"
            )
        alias = candidate.get("executable_architecture_alias")
        if alias is not None:
            raise ValueError(
                f"source-audited candidate cannot claim an executable alias: {candidate_id}"
            )
        if candidate_id in executable_aliases:
            raise ValueError(
                f"source-audited candidate collides with executable alias: {candidate_id}"
            )

        source_audit = candidate.get("source_audit")
        expected_hash = candidate.get("source_audit_sha256")
        if not isinstance(source_audit, str) or not isinstance(expected_hash, str):
            raise ValueError(f"candidate lacks source-audit provenance: {candidate_id}")
        audit_path = _inside(root, source_audit, label="source audit")
        if _sha256(audit_path) != expected_hash:
            raise ValueError(f"source audit hash changed: {candidate_id}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        decision = audit.get("decision", {})
        if not isinstance(decision, dict) or not str(
            decision.get("executable_registry_status", "")
        ).startswith("not_added"):
            raise ValueError(
                f"source audit does not support non-executable status: {candidate_id}"
            )

        operators = candidate.get("operators")
        if not isinstance(operators, list) or not operators:
            raise ValueError(f"candidate has no typed operators: {candidate_id}")
        for operator in operators:
            if not isinstance(operator, dict):
                raise ValueError(f"candidate operator must be an object: {candidate_id}")
            role = operator.get("role")
            name = operator.get("name")
            if role not in _OPERATOR_ROLES or not isinstance(name, str) or not name:
                raise ValueError(f"candidate has an invalid operator: {candidate_id}")
            inputs = operator.get("inputs")
            output = operator.get("output")
            if (
                not isinstance(inputs, list)
                or not inputs
                or any(not isinstance(item, str) or not item for item in inputs)
                or not isinstance(output, str)
                or not output
            ):
                raise ValueError(
                    f"candidate operator lacks a typed edge: {candidate_id}/{name}"
                )
            identity = (str(role), name)
            if identity in operator_names:
                raise ValueError(f"duplicate proposed operator: {role}/{name}")
            operator_names.add(identity)

        requirements = candidate.get("requirements")
        if (
            not isinstance(requirements, list)
            or not requirements
            or any(not isinstance(item, str) or not item for item in requirements)
        ):
            raise ValueError(f"candidate requirements are invalid: {candidate_id}")
        if not isinstance(candidate.get("entry_gate"), str):
            raise ValueError(f"candidate entry gate is missing: {candidate_id}")

    return raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Validate source-audited, non-executable memory candidates."
    )
    parser.add_argument(
        "--catalog",
        type=pathlib.Path,
        default=project_root / "configs" / "source_audited_candidates.json",
    )
    parser.add_argument("--project-root", type=pathlib.Path, default=project_root)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = validate_source_audited_candidates(
        args.catalog, project_root=args.project_root
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
