from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from memory_harness.config import load_program_spec
from memory_harness.contracts import MemoryItem
from memory_harness.registry import build_program


SCHEMA_VERSION = "memory_harness.program_migration_audit/v1"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _legacy_roles(config: Mapping[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    paths = config.get("paths")
    if not isinstance(paths, list):
        raise ValueError("program paths must be a list")
    names = tuple(str(path["name"]) for path in paths)
    anchor = "anchor" if "anchor" in names else None
    history = tuple(name for name in names if name != anchor)
    return anchor, history


def _without_explicit_roles(config: Mapping[str, Any]) -> dict[str, Any]:
    stripped = copy.deepcopy(dict(config))
    options = stripped.get("utilizer", {}).get("options")
    if not isinstance(options, dict):
        raise ValueError("program utilizer.options must be an object")
    options.pop("anchor_path", None)
    options.pop("history_path_quotas", None)
    return stripped


def _item(path_name: str, step_index: int, width: int) -> MemoryItem:
    tokens = np.full((1, width), float(step_index + 1), dtype=np.float32)
    return MemoryItem(
        item_id=f"episode:{path_name}:{step_index}",
        path_name=path_name,
        episode_id="episode",
        step_index=step_index,
        phase="",
        tokens=tokens,
        mask=np.ones((1,), dtype=np.bool_),
    )


def _legacy_pack(
    items: Sequence[MemoryItem],
    *,
    anchor_path: str | None,
    history_paths: Sequence[str],
    embed_dim: int,
    history_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((1 + history_size, embed_dim), dtype=np.float32)
    mask = np.zeros((1 + history_size,), dtype=np.bool_)
    anchors = [item for item in items if item.path_name == anchor_path]
    if anchors:
        tokens[0] = anchors[-1].tokens[anchors[-1].mask][0]
        mask[0] = True
    history = [item for item in items if item.path_name in set(history_paths)][
        -history_size:
    ]
    start = 1 + history_size - len(history)
    for slot, item in enumerate(history, start=start):
        tokens[slot] = item.tokens[item.mask][0]
        mask[slot] = True
    return tokens, mask


def audit_program_migration(
    *,
    frozen_config: pathlib.Path,
    current_config: pathlib.Path,
    context_manifest: pathlib.Path,
) -> dict[str, Any]:
    frozen = _json(frozen_config)
    current = _json(current_config)
    manifest = _json(context_manifest)
    frozen_hash = _sha256(frozen_config)
    current_hash = _sha256(current_config)
    manifest_hash = _sha256(context_manifest)
    declared_hash = str(manifest.get("representation", {}).get("program_config_sha256"))

    current_options = current.get("utilizer", {}).get("options", {})
    if not isinstance(current_options, dict):
        raise ValueError("current utilizer.options must be an object")
    legacy_anchor, legacy_history = _legacy_roles(frozen)
    checks = {
        "manifest_source_hash_matches": frozen_hash == declared_hash,
        "only_explicit_roles_added": _without_explicit_roles(current) == frozen,
        "anchor_role_matches_legacy": current_options.get("anchor_path")
        == legacy_anchor,
        "history_roles_match_legacy": tuple(
            current_options.get("history_path_quotas", {})
        )
        == legacy_history,
        "history_budget_matches_legacy": sum(
            current_options.get("history_path_quotas", {}).values()
        )
        in (0, int(current_options["sliding_window_size"])),
    }

    program = build_program(load_program_spec(current_config))
    embed_dim = int(current_options["embed_dim"])
    history_size = int(current_options["sliding_window_size"])
    replay_cases = []
    replay_exact = True
    history_counts = (0, 1, 29, 30, 31, 35) if legacy_history else (0,)
    for history_count in history_counts:
        for include_anchor in (False, True):
            items: list[MemoryItem] = []
            if include_anchor and legacy_anchor is not None:
                items.append(_item(legacy_anchor, 0, embed_dim))
            for index in range(history_count):
                path_name = legacy_history[index % len(legacy_history)]
                items.append(_item(path_name, index + 1, embed_dim))
            utilization = program.utilizer.apply({}, items)
            expected_tokens, expected_mask = _legacy_pack(
                items,
                anchor_path=legacy_anchor,
                history_paths=legacy_history,
                embed_dim=embed_dim,
                history_size=history_size,
            )
            exact = np.array_equal(
                utilization.observation["memory_tokens"], expected_tokens
            ) and np.array_equal(
                utilization.observation["memory_mask"], expected_mask
            )
            replay_exact = replay_exact and exact
            replay_cases.append(
                {
                    "include_anchor": include_anchor,
                    "history_count": history_count,
                    "exact": bool(exact),
                }
            )
    checks["layout_replay_exact"] = replay_exact
    ready = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "ready_for_context_reuse": ready,
        "source_config": str(frozen_config.resolve()),
        "source_config_sha256": frozen_hash,
        "target_config": str(current_config.resolve()),
        "target_config_sha256": current_hash,
        "context_manifest": str(context_manifest.resolve()),
        "context_manifest_sha256": manifest_hash,
        "declared_program_config_sha256": declared_hash,
        "migration": "implicit_path_roles_to_explicit_mem0_path_quotas",
        "legacy_roles": {
            "anchor_path": legacy_anchor,
            "history_paths": list(legacy_history),
        },
        "checks": checks,
        "replay_cases": replay_cases,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove a frozen Mem-0 config migration preserves token layout."
    )
    parser.add_argument("--frozen-config", type=pathlib.Path, required=True)
    parser.add_argument("--current-config", type=pathlib.Path, required=True)
    parser.add_argument("--context-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = audit_program_migration(
        frozen_config=args.frozen_config,
        current_config=args.current_config,
        context_manifest=args.context_manifest,
    )
    if not payload["ready_for_context_reuse"]:
        raise ValueError(f"program migration audit failed: {payload['checks']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
