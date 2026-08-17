from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "memory_harness.put_back_action_replay/v1"


def _json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return values


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_evidence(source_run: pathlib.Path) -> dict[str, Any]:
    source_run = source_run.resolve()
    config_path = source_run / "config.json"
    episodes_path = source_run / "episodes.jsonl"
    actions_path = source_run / "action_stats.jsonl"
    for path in (config_path, episodes_path, actions_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing replay source: {path}")

    config = _json(config_path)
    if config.get("task_name") != "put_back_block":
        raise ValueError("action progress replay only supports put_back_block")
    episodes = _jsonl(episodes_path)
    actions = _jsonl(actions_path)
    actions_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        try:
            episode_index = int(row["episode_index"])
            step_index = int(row["step_index"])
            values = [float(value) for value in row["env_action_values"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("action_stats contains an invalid replay action") from exc
        if not values:
            raise ValueError("action_stats contains an empty replay action")
        actions_by_episode[episode_index].append(
            {**row, "step_index": step_index, "env_action_values": values}
        )

    expected_indices = list(range(len(episodes)))
    actual_indices = [int(row.get("episode_index", -1)) for row in episodes]
    if actual_indices != expected_indices:
        raise ValueError("source episodes must be ordered and contiguous")
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        rows = sorted(actions_by_episode.get(episode_index, []), key=lambda row: row["step_index"])
        expected_steps = int(episode.get("steps", -1))
        if [row["step_index"] for row in rows] != list(range(expected_steps)):
            raise ValueError(
                f"episode {episode_index} action steps are incomplete or non-contiguous"
            )
        actions_by_episode[episode_index] = rows

    unknown_episodes = set(actions_by_episode) - set(expected_indices)
    if unknown_episodes:
        raise ValueError(f"action_stats contains unknown episodes: {sorted(unknown_episodes)}")
    return {
        "source_run": source_run,
        "config": config,
        "episodes": episodes,
        "actions_by_episode": dict(actions_by_episode),
        "source_sha256": {
            "config": _sha256(config_path),
            "episodes": _sha256(episodes_path),
            "action_stats": _sha256(actions_path),
        },
    }


def replay_episode(
    *,
    env: Any,
    episode: Mapping[str, Any],
    action_rows: Sequence[Mapping[str, Any]],
    debug_eval: Any,
) -> dict[str, Any]:
    import numpy as np

    episode_index = int(episode["episode_index"])
    seed = int(episode["seed"])
    debug_eval._reset_env(env, seed)
    progress: dict[str, Any] | None = None
    terminated = False
    truncated = False
    replay_info: dict[str, Any] = {}
    for expected_step, row in enumerate(action_rows):
        if int(row["step_index"]) != expected_step:
            raise ValueError("replay action rows are not contiguous")
        action = np.asarray(row["env_action_values"], dtype=np.float32)
        _, _, terminated, truncated, replay_info = debug_eval._step_env(env, action)
        snapshot = debug_eval._put_back_block_trace(debug_eval._task_state_source(env))
        if snapshot is None:
            raise ValueError("replay environment does not expose Put Back task state")
        progress = debug_eval._update_put_back_progress(progress, snapshot)
        if terminated or truncated:
            if expected_step + 1 != len(action_rows):
                raise ValueError("replay terminated before the recorded action sequence")
            break
    if progress is None:
        raise ValueError("replay produced no Put Back progress evidence")
    source_success = bool(episode.get("success", False))
    replay_success = bool(
        progress["max_progress_score"] >= 3
        or replay_info.get("is_success", replay_info.get("eval_success", False))
    )
    if replay_success != source_success:
        raise ValueError(
            "deterministic replay success disagrees with the source episode: "
            f"episode={episode_index}, source={source_success}, replay={replay_success}"
        )
    return {
        "episode_index": episode_index,
        "seed": seed,
        "policy_seed": int(episode["policy_seed"]),
        "num_actions": len(action_rows),
        "source_success": source_success,
        "replay_success": replay_success,
        "terminated": terminated,
        "truncated": truncated,
        "task_progress": progress,
    }


def replay_source_run(
    *, source_run: pathlib.Path, rmbench_root: pathlib.Path, gpu_id: int
) -> dict[str, Any]:
    evidence = load_source_evidence(source_run)
    config = evidence["config"]
    from scripts import evaluate_rmbench_baseline_debug as debug_eval

    adapter = debug_eval._import_rmbench_adapter()
    runtime = adapter.RMBenchRuntimeConfig(
        root=rmbench_root.resolve(),
        task_name="put_back_block",
        task_config=str(config["task_config"]),
        gpu_id=gpu_id,
        max_steps=int(config["max_steps"]),
        camera_names=tuple(config["camera_names"]),
    )
    adapter.validate_runtime_config(runtime)
    env = adapter.make_rmbench_env(runtime)
    try:
        rows = [
            replay_episode(
                env=env,
                episode=episode,
                action_rows=evidence["actions_by_episode"][int(episode["episode_index"])],
                debug_eval=debug_eval,
            )
            for episode in evidence["episodes"]
        ]
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return {
        "schema_version": SCHEMA_VERSION,
        "source_run": str(evidence["source_run"]),
        "rmbench_root": str(rmbench_root.resolve()),
        "source_sha256": evidence["source_sha256"],
        "determinism_check": "replay success must equal source success",
        "num_episodes": len(rows),
        "max_progress_score": max(
            int(row["task_progress"]["max_progress_score"]) for row in rows
        ),
        "episodes": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded Put Back actions to recover exact task progress."
    )
    parser.add_argument("--source-run", type=pathlib.Path, required=True)
    parser.add_argument("--rmbench-root", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite replay output: {args.output}")
    result = replay_source_run(
        source_run=args.source_run,
        rmbench_root=args.rmbench_root,
        gpu_id=args.gpu_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
