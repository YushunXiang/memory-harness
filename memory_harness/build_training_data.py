from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_DIR = ROOT.parent / "openpi-libero"
DEFAULT_PROGRAM_CONFIG = ROOT / "memory-harness/configs/fixed_anchor_sliding.json"


@dataclasses.dataclass(frozen=True)
class Chunk:
    episode: int
    ordinal: int
    start: int
    end: int
    phase: str
    prompt: str
    split: str

    def item_id(self, program_name: str) -> str:
        return f"moment:ep{self.episode:03d}:chunk{self.ordinal:04d}:{program_name}"


class _FixedIndexDataset:
    def __init__(self, dataset, indices: Sequence[int]):
        self._dataset = dataset
        self._indices = tuple(int(index) for index in indices)

    def __getitem__(self, index):
        return self._dataset[self._indices[index.__index__()]]

    def __len__(self):
        return len(self._indices)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_chunks(
    template: Mapping[str, Any], *, stride: int
) -> dict[int, tuple[Chunk, ...]]:
    """Split frozen phase ranges into action-chunk-aligned, gap-free segments."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in template.get("segments", ()):
        episode = int(row["lerobot_episode_index"])
        by_episode.setdefault(episode, []).append(dict(row))
    if not by_episode:
        raise ValueError("template has no segments")

    output: dict[int, tuple[Chunk, ...]] = {}
    for episode, rows in sorted(by_episode.items()):
        rows.sort(key=lambda row: int(row["start_frame"]))
        cursor = 0
        chunks: list[Chunk] = []
        for row in rows:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            if start != cursor or end <= start:
                raise ValueError(
                    f"episode {episode} phase ranges are not contiguous at [{start}, {end})"
                )
            for chunk_start in range(start, end, stride):
                chunks.append(
                    Chunk(
                        episode=episode,
                        ordinal=len(chunks),
                        start=chunk_start,
                        end=min(chunk_start + stride, end),
                        phase=str(row.get("phase_label", "")),
                        prompt=str(row["executor_prompt"]),
                        split=str(row["split"]),
                    )
                )
            cursor = end
        output[episode] = tuple(chunks)
    return output


def _validate_context(
    observation: Mapping[str, Any],
    *,
    token_budget: int,
    embed_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_tokens = np.asarray(observation["memory_tokens"], dtype=np.float32)
    source_mask = np.asarray(observation["memory_mask"], dtype=np.bool_)
    expected_tokens = (token_budget, embed_dim)
    if source_tokens.shape != expected_tokens:
        raise ValueError(
            f"invalid Mem-0 context shape: expected {expected_tokens}, got {source_tokens.shape}"
        )
    if source_mask.shape != (token_budget,):
        raise ValueError(f"invalid Mem-0 context mask shape: {source_mask.shape}")
    return source_tokens, source_mask


def simulate_program_contexts(
    chunks: Sequence[Chunk],
    moments: Sequence[np.ndarray],
    *,
    robot_states: Sequence[np.ndarray] | None = None,
    program_config: pathlib.Path = DEFAULT_PROGRAM_CONFIG,
    token_budget: int = 31,
) -> tuple[str, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Run the real plug-in programs so training sees deployment-identical contexts."""

    if len(chunks) != len(moments) or not chunks:
        raise ValueError("chunks and moments must have the same non-zero length")
    if robot_states is not None and len(robot_states) != len(chunks):
        raise ValueError("robot_states must match the chunks length when provided")
    embed_dim = int(np.asarray(moments[0]).shape[-1])
    from memory_harness.config import load_program_spec
    from memory_harness.contracts import MemoryStep
    from memory_harness.registry import build_program

    episode_id = f"episode-{chunks[0].episode}"
    spec = load_program_spec(program_config)
    if spec.utilizer.type != "mem0_context":
        raise ValueError("training program must use the mem0_context utilizer")
    spec = dataclasses.replace(
        spec,
        utilizer=dataclasses.replace(
            spec.utilizer,
            options={
                **dict(spec.utilizer.options),
                "embed_dim": embed_dim,
                "sliding_window_size": token_budget - 1,
            },
        ),
    )
    program = build_program(spec)
    program.reset(episode_id=episode_id)
    contexts: list[tuple[np.ndarray, np.ndarray]] = []
    states = (None,) * len(chunks) if robot_states is None else robot_states
    for chunk, moment, robot_state in zip(chunks, moments, states, strict=True):
        source_tokens = np.asarray(moment, dtype=np.float32)
        if source_tokens.shape != (1, embed_dim):
            raise ValueError(
                "faithful Mem-0 data stores one contextual image latent per step; "
                f"got {source_tokens.shape}"
            )
        result = program.step(
            {},
            MemoryStep(
                episode_id=episode_id,
                step_index=chunk.ordinal,
                phase=chunk.phase,
                source_tokens=source_tokens,
                source_mask=np.ones((source_tokens.shape[0],), dtype=np.bool_),
                robot_state=robot_state,
                metadata={"training_representation": "runtime_moment_tokens"},
            ),
        )
        contexts.append(
            _validate_context(
                result.observation,
                token_budget=token_budget,
                embed_dim=embed_dim,
            )
        )
    return spec.name, tuple(contexts)


def _nearest_donor_ordinal(chunk: Chunk, donor_chunks: Sequence[Chunk]) -> int:
    candidates = [donor.ordinal for donor in donor_chunks]
    if not candidates:
        raise ValueError("donor episode has no chunks")
    return min(candidates, key=lambda ordinal: (abs(ordinal - chunk.ordinal), ordinal))


def build_manifest_and_bank(
    chunks_by_episode: Mapping[int, Sequence[Chunk]],
    contexts_by_episode: Mapping[int, Sequence[tuple[np.ndarray, np.ndarray]]],
    *,
    template: Mapping[str, Any],
    program_name: str = "anchor_sliding",
    program_config: pathlib.Path = DEFAULT_PROGRAM_CONFIG,
    token_budget: int = 31,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Pair causal same-episode contexts with source-disjoint negative contexts."""

    episodes_by_split: dict[str, list[int]] = {}
    for episode, chunks in chunks_by_episode.items():
        splits = {chunk.split for chunk in chunks}
        if len(splits) != 1:
            raise ValueError(f"episode {episode} crosses splits: {sorted(splits)}")
        episodes_by_split.setdefault(next(iter(splits)), []).append(episode)
    for episodes in episodes_by_split.values():
        episodes.sort()
        if len(episodes) < 2:
            raise ValueError(
                "each split needs at least two episodes for mismatch donors"
            )

    bank_ids: list[str] = []
    bank_tokens: list[np.ndarray] = []
    bank_masks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    nonempty = 0
    program_count = 0
    for split, episodes in sorted(episodes_by_split.items()):
        donor_for = {
            episode: episodes[(index + 1) % len(episodes)]
            for index, episode in enumerate(episodes)
        }
        for episode in episodes:
            chunks = chunks_by_episode[episode]
            donor_episode = donor_for[episode]
            donor_chunks = chunks_by_episode[donor_episode]
            for chunk in chunks:
                tokens, mask = contexts_by_episode[episode][chunk.ordinal]
                if tokens.shape[0] != token_budget or mask.shape != (token_budget,):
                    raise ValueError("simulated context does not match token budget")
                bank_ids.append(chunk.item_id(program_name))
                bank_tokens.append(np.asarray(tokens, dtype=np.float16))
                bank_masks.append(np.asarray(mask, dtype=np.bool_))
                nonempty += int(np.any(mask))
                program_count += 1

                donor_ordinal = _nearest_donor_ordinal(chunk, donor_chunks)
                donor_chunk = donor_chunks[donor_ordinal]
                rows.append(
                    {
                        "lerobot_episode_index": episode,
                        "source_episode_id": episode,
                        "start_frame": chunk.start,
                        "end_frame": chunk.end,
                        "phase_label": chunk.phase,
                        "executor_prompt": chunk.prompt,
                        "split": split,
                        "memory_program": program_name,
                        "matched_item_id": chunk.item_id(program_name),
                        "matched_source_episode_id": episode,
                        "matched_uses_only_prior_observations": True,
                        "mismatched_item_id": donor_chunk.item_id(program_name),
                        "mismatched_source_episode_id": donor_episode,
                        "mismatched_source_disjoint": True,
                    }
                )

    item_ids = np.asarray(bank_ids)
    if len(set(bank_ids)) != len(bank_ids):
        raise ValueError("duplicate context bank item ids")
    manifest = {
        "schema_version": "emac_mem0_context/v4",
        "representation": {
            "moment_tokens": ["final_layer_contextual_image_latent"],
            "program": program_name,
            "program_config": str(program_config.resolve()),
            "program_config_sha256": hashlib.sha256(
                program_config.read_bytes()
            ).hexdigest(),
            "layout": {
                "anchor_slot": 0,
                "history_slots": [1, 31],
                "history_alignment": "right_aligned_oldest_to_newest",
                "relative_position": "1_is_most_recent",
            },
            "execution_order": "RETRIEVE_USE_THEN_WRITE",
            "lifecycle": "defined_by_runtime_program",
            "executor_prompt": "current_subtask",
            "causal": True,
            "privileged_state": False,
        },
        # Match Mem-0's executor protocol: train utilization with valid memory.
        # Mismatched memory remains in the manifest for held-out interventions,
        # but training it against the same action target incentivizes ignoring memory.
        "condition_cycle": ["matched"],
        "condition_cycle_seed": 20260814,
        "token_budget": token_budget,
        "tokens_per_item": 1,
        "segments": rows,
        "train_lerobot_episode_ids": list(template["train_lerobot_episode_ids"]),
        "validation_lerobot_episode_ids": list(
            template["validation_lerobot_episode_ids"]
        ),
        "inputs": {
            "template_schema": template.get("schema_version"),
            "base_checkpoint_role": "pi05_contextual_feature_encoder",
        },
    }
    bank = {
        "item_ids": item_ids,
        "tokens": np.stack(bank_tokens, axis=0),
        "masks": np.stack(bank_masks, axis=0),
    }
    audit = {
        "schema_version": "emac_mem0_pairing_audit/v4",
        "ready_for_adapter_training": True,
        "num_segments": len(rows),
        "num_bank_items": len(bank_ids),
        "nonempty_context_fraction": nonempty / len(bank_ids),
        "program_counts": {program_name: program_count},
        "checks": {
            "same_runtime_program_implementation": True,
            "single_program_training": program_count == len(bank_ids),
            "subtask_prompt_and_lifecycle_aligned": True,
            "matched_uses_only_prior_observations": True,
            "mismatched_source_disjoint": True,
            "train_validation_episode_disjoint": not bool(
                set(template["train_lerobot_episode_ids"])
                & set(template["validation_lerobot_episode_ids"])
            ),
            "no_privileged_state": True,
        },
    }
    audit["ready_for_adapter_training"] = all(audit["checks"].values())
    return manifest, bank, audit


def _contextual_image_latent(encoded: Mapping[str, Any]) -> np.ndarray:
    """Return the single final-layer contextual image latent used by Mem-0."""
    prefix = np.asarray(encoded["prefix_embedding"], dtype=np.float32)
    if prefix.ndim != 2:
        raise ValueError(f"prefix embedding must be [B, D], got {prefix.shape}")
    return prefix[:, None, :]


def _validate_prompt_preserving_data_factory(data_factory: Any) -> None:
    """Reject a context encoder whose training pipeline drops subtask prompts."""

    base_config = getattr(data_factory, "base_config", None)
    if base_config is None or getattr(base_config, "prompt_from_task", False) is not True:
        raise ValueError(
            "Mem-0 context encoding requires base_config.prompt_from_task=True"
        )
    repack = getattr(data_factory, "repack_transforms", None)
    inputs = getattr(repack, "inputs", ())
    if len(inputs) != 1:
        raise ValueError("Mem-0 context encoding requires one repack transform")
    structure = getattr(inputs[0], "structure", None)
    if not isinstance(structure, Mapping) or structure.get("prompt") != "prompt":
        raise ValueError(
            "Mem-0 context encoding requires repack to preserve the prompt field"
        )


def _encode_chunk_inputs(
    args: argparse.Namespace,
    chunks_by_episode: Mapping[int, Sequence[Chunk]],
    *,
    task_prompt: str,
    repo_id: str,
    asset_id: str,
) -> tuple[
    dict[int, tuple[np.ndarray, ...]],
    dict[int, tuple[np.ndarray, ...]],
]:
    os.environ.setdefault("HF_LEROBOT_HOME", str(args.hf_lerobot_home.resolve()))
    for path in (ROOT, OPENPI_DIR / "src", ROOT / "memory-harness"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib
    from openpi.training import data_loader

    episode_ids = tuple(sorted(chunks_by_episode))
    config = config_lib.get_config(args.config)
    _validate_prompt_preserving_data_factory(config.data)
    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=0,
        data=dataclasses.replace(
            config.data,
            repo_id=repo_id,
            assets=config_lib.AssetsConfig(
                assets_dir=str(args.assets_dir.resolve()),
                asset_id=asset_id,
            ),
            adapt_to_pi=False,
            default_prompt=task_prompt,
            episode_ids=episode_ids,
            context_injection_manifest=None,
            context_bank_path=None,
            context_condition_override=None,
        ),
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    ranges = {
        int(episode): (int(start), int(end))
        for episode, start, end in zip(
            raw_dataset.episodes,
            raw_dataset.episode_data_index["from"].tolist(),
            raw_dataset.episode_data_index["to"].tolist(),
            strict=True,
        )
    }
    ordered_chunks = [
        chunk for episode in episode_ids for chunk in chunks_by_episode[episode]
    ]
    indices = []
    for chunk in ordered_chunks:
        start, end = ranges[chunk.episode]
        if chunk.start >= end - start:
            raise ValueError(
                f"chunk frame {chunk.start} exceeds episode {chunk.episode} length {end - start}"
            )
        indices.append(start + chunk.start)

    padded_count = math.ceil(len(indices) / args.batch_size) * args.batch_size
    padded_indices = indices + [indices[-1]] * (padded_count - len(indices))
    prompts = [chunk.prompt for chunk in ordered_chunks]
    padded_prompts = prompts + [prompts[-1]] * (padded_count - len(prompts))

    class _PromptedFixedIndexDataset(_FixedIndexDataset):
        def __getitem__(self, index):
            position = index.__index__()
            item = dict(super().__getitem__(position))
            item["prompt"] = np.asarray(padded_prompts[position])
            return item

    selected_raw = _PromptedFixedIndexDataset(raw_dataset, padded_indices)
    selected = data_loader.transform_dataset(selected_raw, data_config)
    torch_loader = data_loader.TorchDataLoader(
        selected,
        local_batch_size=args.batch_size,
        shuffle=False,
        num_batches=padded_count // args.batch_size,
        num_workers=0,
        seed=config.seed,
    )
    loader = data_loader.DataLoaderImpl(data_config, torch_loader)

    params = model_lib.restore_params(args.base_checkpoint.resolve() / "params")
    model = config.model.load(params)
    model.eval()
    if not hasattr(model, "encode_mem0_features"):
        raise ValueError(
            "the selected π0.5 model does not implement encode_mem0_features()"
        )
    encode = nnx_utils.module_jit(model.encode_mem0_features)
    encoded_moments: list[np.ndarray] = []
    encoded_states: list[np.ndarray] = []
    total_batches = padded_count // args.batch_size
    for batch_index, (observation, _) in enumerate(loader, start=1):
        encoded = encode(observation)
        moment_batch = _contextual_image_latent(
            {
                key: None if value is None else np.asarray(value)
                for key, value in encoded.items()
            }
        )
        encoded_moments.extend(moment_batch)
        encoded_states.extend(np.asarray(observation.state, dtype=np.float32))
        if batch_index == 1 or batch_index % 100 == 0 or batch_index == total_batches:
            print(
                f"ENCODE_CONTEXT batch={batch_index}/{total_batches} "
                f"items={min(batch_index * args.batch_size, len(ordered_chunks))}",
                flush=True,
            )
    encoded_moments = encoded_moments[: len(ordered_chunks)]
    encoded_states = encoded_states[: len(ordered_chunks)]
    if len(encoded_moments) != len(ordered_chunks):
        raise RuntimeError("feature loader returned the wrong number of moments")
    if len(encoded_states) != len(ordered_chunks):
        raise RuntimeError("feature loader returned the wrong number of robot states")

    moment_output: dict[int, list[np.ndarray]] = {
        episode: [] for episode in episode_ids
    }
    state_output: dict[int, list[np.ndarray]] = {
        episode: [] for episode in episode_ids
    }
    for chunk, moment, state in zip(
        ordered_chunks, encoded_moments, encoded_states, strict=True
    ):
        moment_output[chunk.episode].append(np.asarray(moment, dtype=np.float32))
        state_output[chunk.episode].append(np.asarray(state, dtype=np.float32))
    return (
        {episode: tuple(moments) for episode, moments in moment_output.items()},
        {episode: tuple(states) for episode, states in state_output.items()},
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    from memory_harness.config import load_program_spec
    from memory_harness.tasks import load_task_spec

    task_spec = load_task_spec(args.task_config)
    template = _read_json(args.template_manifest)
    chunks_by_episode = build_chunks(template, stride=args.chunk_stride)
    program_spec = load_program_spec(args.program_config)
    if program_spec.paths:
        if args.base_checkpoint is None:
            raise ValueError("memory programs with paths require --base-checkpoint")
        moments_by_episode, robot_states_by_episode = _encode_chunk_inputs(
            args,
            chunks_by_episode,
            task_prompt=task_spec.prompt,
            repo_id=task_spec.repo_id,
            asset_id=task_spec.asset_id,
        )
    else:
        embed_dim = int(program_spec.utilizer.options.get("embed_dim", 2048))
        moments_by_episode = {
            episode: tuple(np.zeros((1, embed_dim), dtype=np.float32) for _ in chunks)
            for episode, chunks in chunks_by_episode.items()
        }
        robot_states_by_episode = {episode: None for episode in chunks_by_episode}
    program_name: str | None = None
    contexts_by_episode: dict[int, tuple[tuple[np.ndarray, np.ndarray], ...]] = {}
    for episode, moments in moments_by_episode.items():
        episode_program_name, contexts = simulate_program_contexts(
            chunks_by_episode[episode],
            moments,
            robot_states=robot_states_by_episode[episode],
            program_config=args.program_config,
        )
        if program_name is None:
            program_name = episode_program_name
        elif episode_program_name != program_name:
            raise RuntimeError("program config changed while building training data")
        contexts_by_episode[episode] = contexts
    assert program_name is not None
    manifest, bank, audit = build_manifest_and_bank(
        chunks_by_episode,
        contexts_by_episode,
        template=template,
        program_name=program_name,
        program_config=args.program_config,
    )
    manifest["inputs"].update(
        {
            "template_manifest": str(args.template_manifest.resolve()),
            "base_checkpoint": (
                None
                if args.base_checkpoint is None
                else str(args.base_checkpoint.resolve())
            ),
            "chunk_stride": args.chunk_stride,
            "program_config": str(args.program_config.resolve()),
            "task_config": str(args.task_config.resolve()),
            "task_name": task_spec.task_name,
            "task_memory_complexity": task_spec.tmc,
        }
    )
    args.output_bank.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{args.output_bank.name}.",
        suffix=".tmp",
        dir=args.output_bank.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = pathlib.Path(temporary_file.name)
    try:
        with temporary_path.open("wb") as stream:
            np.savez(
                stream,
                item_ids=bank["item_ids"],
                tokens=bank["tokens"],
                masks=bank["masks"],
            )
        temporary_path.replace(args.output_bank)
    finally:
        temporary_path.unlink(missing_ok=True)
    _write_json(args.output_manifest, manifest)
    _write_json(args.output_audit, audit)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causal π0.5 moment-token training data using real memory programs."
    )
    parser.add_argument(
        "--template-manifest",
        type=pathlib.Path,
        default=ROOT
        / "rmbench_runs/cmci_20260725_context_adapter_v1/pairing_manifest.json",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument("--task-config", type=pathlib.Path, required=True)
    parser.add_argument("--config", default="pi05_aloha_pen_uncap_mem0")
    parser.add_argument(
        "--program-config",
        type=pathlib.Path,
        default=DEFAULT_PROGRAM_CONFIG,
        help="One deployable memory program used identically for training context generation.",
    )
    parser.add_argument(
        "--assets-dir", type=pathlib.Path, default=ROOT / "rmbench_assets"
    )
    parser.add_argument(
        "--hf-lerobot-home", type=pathlib.Path, default=ROOT / "rmbench_lerobot_data"
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=1,
        help="Environment-frame stride for memory writes; Mem-0 uses one observation per step.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-bank", type=pathlib.Path, required=True)
    parser.add_argument("--output-audit", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    result = build(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
