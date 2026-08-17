from __future__ import annotations

import argparse
import inspect
import json
import pathlib
from collections.abc import Callable, Mapping
from typing import Any

from memory_harness import registry


SCHEMA_VERSION = "memory_harness.component_catalog/v2"

_REGISTRIES: tuple[tuple[str, Mapping[str, Callable[..., Any]]], ...] = (
    ("encoder", registry.ENCODERS),
    ("writer", registry.WRITERS),
    ("store", registry.STORES),
    ("retriever", registry.RETRIEVERS),
    ("lifecycle", registry.LIFECYCLES),
    ("controller", registry.CONTROLLERS),
    ("utilizer", registry.UTILIZERS),
)


def _annotation_name(annotation: Any) -> str | None:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", str(annotation))


def _component_description(factory: Callable[..., Any]) -> dict[str, Any]:
    options: dict[str, dict[str, Any]] = {}
    for name, parameter in inspect.signature(factory).parameters.items():
        if name in {"self", "args", "kwargs"}:
            continue
        option: dict[str, Any] = {
            "required": parameter.default is inspect.Parameter.empty,
        }
        annotation = _annotation_name(parameter.annotation)
        if annotation is not None:
            option["annotation"] = annotation
        if parameter.default is not inspect.Parameter.empty:
            option["default"] = parameter.default
        options[name] = option
    description = inspect.cleandoc(factory.__doc__) if factory.__doc__ else None
    result = {
        "implementation": f"{factory.__module__}.{factory.__qualname__}",
        "options": options,
    }
    if description is not None:
        result["description"] = description
    return result


def component_catalog() -> dict[str, Any]:
    """Return the executable registry as a deterministic, JSON-safe contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "program_contract": {
            "path_components": [
                "encoder",
                "writer",
                "store",
                "retriever",
                "lifecycle",
            ],
            "program_components": ["controller", "utilizer"],
        },
        "components": {
            role: {
                name: _component_description(components[name])
                for name in sorted(components)
            }
            for role, components in _REGISTRIES
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the registered memory components and constructor options."
    )
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    payload = json.dumps(component_catalog(), indent=2, sort_keys=True) + "\n"
    args = parse_args()
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
