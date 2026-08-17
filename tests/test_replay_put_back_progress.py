from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_harness.replay_put_back_progress import (
    load_source_evidence,
    replay_episode,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _source_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "config.json",
        {
            "task_name": "put_back_block",
            "task_config": "demo_clean",
            "max_steps": 3,
            "camera_names": ["head_camera", "left_camera", "right_camera"],
        },
    )
    _write_jsonl(
        run / "episodes.jsonl",
        [
            {
                "episode_index": 0,
                "seed": 100000,
                "policy_seed": 120000,
                "steps": 2,
                "success": False,
            },
            {
                "episode_index": 1,
                "seed": 100001,
                "policy_seed": 120001,
                "steps": 1,
                "success": False,
            },
        ],
    )
    _write_jsonl(
        run / "action_stats.jsonl",
        [
            {"episode_index": 0, "step_index": 0, "env_action_values": [0.1, 0.2]},
            {"episode_index": 0, "step_index": 1, "env_action_values": [0.3, 0.4]},
            {"episode_index": 1, "step_index": 0, "env_action_values": [0.5, 0.6]},
        ],
    )
    return run


def test_load_source_evidence_validates_and_hashes_exact_inputs(tmp_path: Path) -> None:
    run = _source_run(tmp_path)

    evidence = load_source_evidence(run)

    assert evidence["source_run"] == run.resolve()
    assert [row["step_index"] for row in evidence["actions_by_episode"][0]] == [0, 1]
    assert evidence["actions_by_episode"][1][0]["env_action_values"] == [0.5, 0.6]
    assert evidence["source_sha256"] == {
        name: hashlib.sha256((run / filename).read_bytes()).hexdigest()
        for name, filename in {
            "config": "config.json",
            "episodes": "episodes.jsonl",
            "action_stats": "action_stats.jsonl",
        }.items()
    }


def test_load_source_evidence_rejects_incomplete_action_sequence(tmp_path: Path) -> None:
    run = _source_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (run / "action_stats.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    _write_jsonl(run / "action_stats.jsonl", [row for row in rows if row["step_index"] != 1])

    with pytest.raises(ValueError, match="incomplete or non-contiguous"):
        load_source_evidence(run)


def test_load_source_evidence_rejects_unknown_episode_actions(tmp_path: Path) -> None:
    run = _source_run(tmp_path)
    with (run / "action_stats.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"episode_index": 2, "step_index": 0, "env_action_values": [0.0, 0.0]}
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="unknown episodes"):
        load_source_evidence(run)


class _FakeEnv:
    def __init__(self, progress: list[int]) -> None:
        self.progress = progress
        self.step_index = 0
        self.reset_seeds: list[int] = []

    def reset(self, *, seed: int) -> dict[str, object]:
        self.reset_seeds.append(seed)
        self.step_index = 0
        return {}

    def step(self, action: object) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        del action
        self.step_index += 1
        return {}, 0.0, False, False, {}


def _debug_eval() -> SimpleNamespace:
    def trace(env: _FakeEnv) -> dict[str, int]:
        return {"progress_score_now": env.progress[env.step_index - 1]}

    def update(summary: dict[str, int] | None, snapshot: dict[str, int]) -> dict[str, int]:
        score = snapshot["progress_score_now"]
        return {"max_progress_score": max(score, (summary or {}).get("max_progress_score", 0))}

    return SimpleNamespace(
        _reset_env=lambda env, seed: env.reset(seed=seed),
        _step_env=lambda env, action: env.step(action),
        _task_state_source=lambda env: env,
        _put_back_block_trace=trace,
        _update_put_back_progress=update,
    )


def test_replay_episode_recovers_progress_with_recorded_seed() -> None:
    env = _FakeEnv([0, 2])

    result = replay_episode(
        env=env,
        episode={
            "episode_index": 0,
            "seed": 100000,
            "policy_seed": 120000,
            "success": False,
        },
        action_rows=[
            {"step_index": 0, "env_action_values": [0.1]},
            {"step_index": 1, "env_action_values": [0.2]},
        ],
        debug_eval=_debug_eval(),
    )

    assert env.reset_seeds == [100000]
    assert result["task_progress"]["max_progress_score"] == 2
    assert result["replay_success"] is False


def test_replay_episode_rejects_success_disagreement() -> None:
    with pytest.raises(ValueError, match="success disagrees"):
        replay_episode(
            env=_FakeEnv([3]),
            episode={
                "episode_index": 0,
                "seed": 100000,
                "policy_seed": 120000,
                "success": False,
            },
            action_rows=[{"step_index": 0, "env_action_values": [0.1]}],
            debug_eval=_debug_eval(),
        )
