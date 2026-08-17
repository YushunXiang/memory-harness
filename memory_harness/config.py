from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class ComponentSpec:
    type: str
    options: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def parse(cls, raw: object, *, field: str) -> ComponentSpec:
        if not isinstance(raw, dict):
            raise ValueError(f"{field} must be an object")
        component_type = raw.get("type")
        if not isinstance(component_type, str) or not component_type:
            raise ValueError(f"{field}.type must be a non-empty string")
        unknown = set(raw) - {"type", "options"}
        if unknown:
            raise ValueError(f"{field} contains unknown keys: {sorted(unknown)}")
        options = raw.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"{field}.options must be an object")
        return cls(component_type, dict(options))


@dataclasses.dataclass(frozen=True)
class PathSpec:
    name: str
    encoder: ComponentSpec
    writer: ComponentSpec
    store: ComponentSpec
    retriever: ComponentSpec
    lifecycle: ComponentSpec

    @classmethod
    def parse(cls, raw: object, *, index: int) -> PathSpec:
        if not isinstance(raw, dict):
            raise ValueError(f"paths[{index}] must be an object")
        expected = {"name", "encoder", "writer", "store", "retriever", "lifecycle"}
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown or missing:
            raise ValueError(
                f"paths[{index}] keys mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        name = raw["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"paths[{index}].name must be a non-empty string")
        return cls(
            name=name,
            encoder=ComponentSpec.parse(
                raw["encoder"], field=f"paths[{index}].encoder"
            ),
            writer=ComponentSpec.parse(raw["writer"], field=f"paths[{index}].writer"),
            store=ComponentSpec.parse(raw["store"], field=f"paths[{index}].store"),
            retriever=ComponentSpec.parse(
                raw["retriever"], field=f"paths[{index}].retriever"
            ),
            lifecycle=ComponentSpec.parse(
                raw["lifecycle"], field=f"paths[{index}].lifecycle"
            ),
        )


@dataclasses.dataclass(frozen=True)
class ProgramSpec:
    name: str
    deployable: bool
    paths: tuple[PathSpec, ...]
    controller: ComponentSpec
    utilizer: ComponentSpec

    @classmethod
    def parse(cls, raw: object) -> ProgramSpec:
        if not isinstance(raw, dict):
            raise ValueError("program config must be an object")
        expected = {"name", "deployable", "paths", "controller", "utilizer"}
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown or missing:
            raise ValueError(
                f"program keys mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        name = raw["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("program name must be a non-empty string")
        if not isinstance(raw["deployable"], bool):
            raise ValueError("deployable must be a boolean")
        paths = raw["paths"]
        if not isinstance(paths, list):
            raise ValueError("paths must be a list")
        parsed_paths = tuple(
            PathSpec.parse(path, index=index) for index, path in enumerate(paths)
        )
        names = [path.name for path in parsed_paths]
        if len(names) != len(set(names)):
            raise ValueError("memory path names must be unique")
        return cls(
            name=name,
            deployable=raw["deployable"],
            paths=parsed_paths,
            controller=ComponentSpec.parse(raw["controller"], field="controller"),
            utilizer=ComponentSpec.parse(raw["utilizer"], field="utilizer"),
        )


def load_program_spec(path: pathlib.Path | str) -> ProgramSpec:
    path = pathlib.Path(path)
    return ProgramSpec.parse(json.loads(path.read_text(encoding="utf-8")))
