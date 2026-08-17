from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "memory_harness.put_back_action_replay_summary/v1"
REPLAY_SCHEMA = "memory_harness.put_back_action_replay/v1"
REQUIRED_LABELS = ("full_memory", "empty_mask", "native_none")


def _load_replay(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != REPLAY_SCHEMA:
        raise ValueError(f"unsupported Put Back replay artifact: {path}")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Put Back replay has no episodes: {path}")
    return value


def _episode_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    try:
        return (
            int(row["episode_index"]),
            int(row["seed"]),
            int(row["policy_seed"]),
            int(row["num_actions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("replay episode has an invalid pairing key") from exc


def _progress(row: Mapping[str, Any]) -> int:
    try:
        score = int(row["task_progress"]["max_progress_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("replay episode has no valid Put Back progress score") from exc
    if score not in range(4):
        raise ValueError(f"Put Back progress score must be in [0, 3], got {score}")
    return score


def _paired_direction(reference: Sequence[int], candidate: Sequence[int]) -> dict[str, int]:
    return {
        "candidate_higher": sum(c > r for r, c in zip(reference, candidate, strict=True)),
        "equal": sum(c == r for r, c in zip(reference, candidate, strict=True)),
        "candidate_lower": sum(c < r for r, c in zip(reference, candidate, strict=True)),
    }


def summarize_replays(paths: Mapping[str, pathlib.Path]) -> dict[str, Any]:
    if set(paths) != set(REQUIRED_LABELS):
        raise ValueError(f"replay labels must be exactly {list(REQUIRED_LABELS)}")
    replays = {label: _load_replay(paths[label]) for label in REQUIRED_LABELS}
    rows = {label: replay["episodes"] for label, replay in replays.items()}
    pairing = [_episode_key(row) for row in rows["full_memory"]]
    if [key[0] for key in pairing] != list(range(len(pairing))):
        raise ValueError("full-memory replay episodes must be ordered and contiguous")
    for label in REQUIRED_LABELS[1:]:
        if [_episode_key(row) for row in rows[label]] != pairing:
            raise ValueError(f"{label} replay is not paired with full_memory")

    scores = {
        label: [_progress(row) for row in rows[label]] for label in REQUIRED_LABELS
    }
    conditions = {}
    for label in REQUIRED_LABELS:
        counts = Counter(scores[label])
        conditions[label] = {
            "num_episodes": len(scores[label]),
            "mean_progress_score": sum(scores[label]) / len(scores[label]),
            "max_progress_score": max(scores[label]),
            "score_counts": {str(score): counts[score] for score in range(4)},
            "replay_artifact": str(paths[label].resolve()),
            "replay_artifact_sha256": hashlib.sha256(paths[label].read_bytes()).hexdigest(),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "put_back_block",
        "evidence_scope": "retrospective_deterministic_action_replay",
        "formal_utility_claim_allowed": False,
        "num_paired_episodes": len(pairing),
        "pairing_keys": [
            {
                "episode_index": episode,
                "seed": seed,
                "policy_seed": policy_seed,
                "num_actions": num_actions,
            }
            for episode, seed, policy_seed, num_actions in pairing
        ],
        "conditions": conditions,
        "paired_directions": {
            "empty_mask_to_full_memory": _paired_direction(
                scores["empty_mask"], scores["full_memory"]
            ),
            "native_none_to_full_memory": _paired_direction(
                scores["native_none"], scores["full_memory"]
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired Put Back action replays.")
    for label in REQUIRED_LABELS:
        parser.add_argument(f"--{label.replace('_', '-')}", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite replay summary: {args.output}")
    result = summarize_replays(
        {label: getattr(args, label) for label in REQUIRED_LABELS}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
