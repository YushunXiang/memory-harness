from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import shutil

import numpy as np

from memory_harness.audit import JsonlAuditSink
from memory_harness.config import load_program_spec
from memory_harness.contracts import MemoryStep
from memory_harness.contracts import EpisodeOutcome
from memory_harness.registry import build_program


PROGRAM_NAMES = (
    "none",
    "anchor",
    "sliding",
    "anchor_sliding",
    "consolidating",
    "novelty_sliding",
    "dhem_event",
    "kinematic_event",
    "content_recency",
    "boundary_chunk",
    "semantic_recent_union",
    "tiered_chunk_mean",
    "temporal_multiscale",
    "uniform_global",
    "recent_global",
    "verified_success_latent",
    "completed_phase_handoff",
)


def run_smoke(
    *, config_path: pathlib.Path, output_dir: pathlib.Path
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "program.json")
    spec = load_program_spec(config_path)
    program = build_program(
        spec,
        audit_sink=JsonlAuditSink(output_dir / "memory_audit.jsonl"),
    )
    program.reset(episode_id="smoke-episode")
    required_steps = 4
    kinematic_options: dict[str, int] | None = None
    kinematic_paths: set[str] = set()
    required_maintenance_actions: set[str] = set()
    persistent_success_memory = any(
        path.store.type == "verified_success_ring" for path in spec.paths
    )
    for path in spec.paths:
        if path.writer.type == "causal_kinematic_peak":
            kinematic_paths.add(path.name)
            if kinematic_options is None:
                kinematic_options = {
                    key: int(path.writer.options[key])
                    for key in (
                        "motion_window",
                        "peak_lookback",
                        "confirmation_delay",
                    )
                }
            required_steps = max(
                required_steps,
                int(path.writer.options["motion_window"])
                + int(path.writer.options["peak_lookback"])
                + int(path.writer.options["confirmation_delay"])
                + 2,
            )
        if path.store.type == "adjacent_merge":
            phase_reset_padding = 1 if path.lifecycle.type == "phase" else 0
            required_steps = max(
                required_steps,
                int(path.store.options["capacity"]) + 1 + phase_reset_padding,
            )
            required_maintenance_actions.add("consolidate_adjacent")
        elif path.store.type == "dhem_event":
            required_steps = max(
                required_steps, int(path.store.options["capacity"]) + 1
            )
            required_maintenance_actions.add("dhem_capacity_maintenance")
        elif path.store.type == "tiered_chunk_mean":
            short_capacity = int(path.store.options["short_capacity"])
            migration_chunk_size = int(
                path.store.options["migration_chunk_size"]
            )
            long_capacity = int(path.store.options["long_capacity"])
            required_steps = max(
                required_steps,
                short_capacity + 1 + long_capacity * migration_chunk_size,
            )
            required_maintenance_actions.update(
                {"migrate_chunk", "consolidate_long_term_adjacent"}
            )
    used_tokens: list[int] = []
    stored_items: list[int] = []
    event_counts: collections.Counter[str] = collections.Counter()
    write_counts_by_path: collections.Counter[str] = collections.Counter()
    maintenance_counts: collections.Counter[str] = collections.Counter()
    robot_position = 0.0
    for index in range(required_steps):
        phase = "phase-a" if index == 0 else "phase-b"
        source_tokens = None
        source_mask = None
        if program.paths:
            embed_dim = int(spec.utilizer.options.get("embed_dim", 6))
            source_tokens = np.full(
                (1, embed_dim), float(index + 1), dtype=np.float32
            )
            source_mask = np.ones((1,), dtype=np.bool_)
        if index and kinematic_options is not None:
            target_step = (
                kinematic_options["motion_window"]
                + kinematic_options["peak_lookback"]
            )
            center = target_step - (kinematic_options["motion_window"] - 1) / 2
            robot_position += abs(index - center) + 1.0
        result = program.step(
            {"state": np.array([index], dtype=np.float32)},
            MemoryStep(
                episode_id="smoke-episode",
                step_index=index,
                phase=phase,
                source_tokens=source_tokens,
                source_mask=source_mask,
                robot_state=np.asarray([robot_position], dtype=np.float32),
            ),
        )
        used_tokens.append(result.used_token_count)
        stored_items.append(result.stored_item_count)
        event_counts.update(event.event for event in result.events)
        write_counts_by_path.update(
            event.path_name
            for event in result.events
            if event.event == "WRITE" and event.path_name is not None
        )
        for event in result.events:
            if event.event != "WRITE":
                continue
            maintenance_action = event.details.get("maintenance_action")
            if isinstance(maintenance_action, str):
                maintenance_counts[maintenance_action] += 1
                if maintenance_action in {
                    "discard_incoming",
                    "merge_history_and_append",
                }:
                    maintenance_counts["dhem_capacity_maintenance"] += 1
            if event.details.get("consolidated") is True:
                maintenance_counts["consolidate_adjacent"] += 1
            long_term = event.details.get("long_term_maintenance")
            if isinstance(long_term, dict) and long_term.get("consolidated") is True:
                maintenance_counts["consolidate_long_term_adjacent"] += 1

    missing_kinematic_writes = kinematic_paths - set(write_counts_by_path)
    if missing_kinematic_writes:
        raise RuntimeError(
            "candidate smoke did not trigger kinematic writers: "
            f"{sorted(missing_kinematic_writes)}"
        )
    missing_maintenance = required_maintenance_actions - set(maintenance_counts)
    if missing_maintenance:
        raise RuntimeError(
            "candidate smoke did not trigger store maintenance: "
            f"{sorted(missing_maintenance)}"
        )

    finish_events = program.finish_episode(
        EpisodeOutcome(
            episode_id="smoke-episode",
            success=True,
            final_step_index=required_steps - 1,
            total_reward=1.0,
        )
    )
    committed_item_count = max(
        (
            int(event.details.get("committed_item_count", 0))
            for event in finish_events
            if event.event == "STORE_FINALIZE"
        ),
        default=0,
    )
    program.reset(episode_id="smoke-reset")
    reset_tokens = None
    reset_mask = None
    if program.paths:
        embed_dim = int(spec.utilizer.options.get("embed_dim", 6))
        reset_tokens = np.ones((1, embed_dim), dtype=np.float32)
        reset_mask = np.ones((1,), dtype=np.bool_)
    after_reset = program.step(
        {"state": np.zeros((1,), dtype=np.float32)},
        MemoryStep(
            episode_id="smoke-reset",
            step_index=0,
            phase="phase-a",
            source_tokens=reset_tokens,
            source_mask=reset_mask,
            robot_state=np.zeros((1,), dtype=np.float32),
        ),
    )
    reset_isolated = not after_reset.retrieved_item_ids
    committed_retrieved_item_ids: list[str] = []
    failed_episode_excluded: bool | None = None
    if persistent_success_memory:
        if not after_reset.retrieved_item_ids or any(
            item_id.startswith("smoke-reset:")
            for item_id in after_reset.retrieved_item_ids
        ):
            raise RuntimeError(
                "verified-success candidate did not retrieve only committed prior items"
            )
        committed_retrieved_item_ids = list(after_reset.retrieved_item_ids)
        program.finish_episode(
            EpisodeOutcome(
                episode_id="smoke-reset",
                success=False,
                final_step_index=0,
            )
        )
        program.reset(episode_id="smoke-after-failure")
        failure_tokens = np.ones((1, embed_dim), dtype=np.float32)
        failure_result = program.step(
            {"state": np.zeros((1,), dtype=np.float32)},
            MemoryStep(
                episode_id="smoke-after-failure",
                step_index=0,
                phase="phase-a",
                source_tokens=failure_tokens,
                source_mask=np.ones((1,), dtype=np.bool_),
                robot_state=np.zeros((1,), dtype=np.float32),
            ),
        )
        failed_episode_excluded = not any(
            item_id.startswith("smoke-reset:")
            for item_id in failure_result.retrieved_item_ids
        )
        if not failed_episode_excluded:
            raise RuntimeError("failed episode contaminated verified-success memory")
    elif not reset_isolated:
        raise RuntimeError("candidate retained items across episode reset")
    with config_path.open("rb") as stream:
        config_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {
        "schema_version": "memory_harness.candidate_smoke/v1",
        "status": "completed",
        "program": program.name,
        "program_config_sha256": config_sha256,
        "steps": required_steps,
        "used_tokens_by_step": used_tokens,
        "max_used_token_count": max(used_tokens, default=0),
        "max_stored_item_count": max(stored_items, default=0),
        "final_stored_item_count": stored_items[-1],
        "event_counts": dict(sorted(event_counts.items())),
        "write_counts_by_path": dict(sorted(write_counts_by_path.items())),
        "maintenance_counts": dict(sorted(maintenance_counts.items())),
        "episode_reset_isolated": reset_isolated,
        "persistent_success_memory": persistent_success_memory,
        "committed_item_count": committed_item_count,
        "committed_retrieved_item_ids": committed_retrieved_item_ids,
        "failed_episode_excluded": failed_episode_excluded,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic CPU traces for fixed memory programs."
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--program", choices=("all", *PROGRAM_NAMES))
    selection.add_argument(
        "--config",
        type=pathlib.Path,
        help="Validate an arbitrary candidate program config generated by an Agent.",
    )
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.config is not None:
        summary = run_smoke(
            config_path=args.config,
            output_dir=args.output_dir / "candidate",
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps([summary], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    selected_program = args.program or "all"
    names = PROGRAM_NAMES if selected_program == "all" else (selected_program,)
    summaries = []
    for name in names:
        summaries.append(
            run_smoke(
                config_path=root / "configs" / f"fixed_{name}.json",
                output_dir=args.output_dir / name,
            )
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
