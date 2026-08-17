from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS_ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI_DIR = ROOT.parent / "openpi-libero"
os.environ.setdefault("HF_LEROBOT_HOME", str((ROOT / "rmbench_lerobot_data").resolve()))
for path in (HARNESS_ROOT, ROOT, OPENPI_DIR / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.models import model as model_lib  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
from openpi.training import config as config_lib  # noqa: E402
from openpi.training import data_loader  # noqa: E402
from memory_harness.tasks import TaskSpec, load_task_spec  # noqa: E402


SOURCE_CONDITIONS = ("matched", "mismatched")
INTERVENTIONS = (
    "matched",
    "empty",
    "mismatched",
    "without_anchor",
    "without_sliding",
    "anchor_replaced",
    "sliding_replaced",
    "sliding_shuffled",
)


class _FixedIndexDataset:
    def __init__(self, dataset, indices: list[int]):
        self._dataset = dataset
        self._indices = indices

    def __getitem__(self, index):
        return self._dataset[self._indices[index.__index__()]]

    def __len__(self) -> int:
        return len(self._indices)


def _read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    unique = np.unique(clusters)
    grouped = {cluster: values[clusters == cluster] for cluster in unique}
    rng = np.random.default_rng(seed)
    means = np.empty((samples,), dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        means[index] = float(
            np.mean(np.concatenate([grouped[cluster] for cluster in selected]))
        )
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _make_config(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    condition: str,
    task_spec: TaskSpec,
):
    config = config_lib.get_config(args.config)
    if config.model.memory.utilization_mode != "mem0":
        raise ValueError("offline Mem-0 evaluation requires utilization_mode='mem0'")
    return dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=0,
        data=dataclasses.replace(
            config.data,
            repo_id=task_spec.repo_id,
            assets=config_lib.AssetsConfig(
                assets_dir=str(args.assets_dir.resolve()),
                asset_id=task_spec.asset_id,
            ),
            adapt_to_pi=False,
            default_prompt=task_spec.prompt,
            episode_ids=tuple(manifest["validation_lerobot_episode_ids"]),
            context_injection_manifest=str(args.manifest.resolve()),
            context_bank_path=str(args.context_bank.resolve()),
            context_condition_override=condition,
        ),
    )


def _interior_indices(start: int, end: int, count: int) -> np.ndarray:
    """Select deterministic interior frames; a single sample is the midpoint."""
    if end <= start or count <= 0:
        raise ValueError(f"invalid frame range/count: start={start}, end={end}, count={count}")
    fractions = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
    indices = start + np.floor(fractions * (end - start)).astype(np.int64)
    return np.minimum(indices, end - 1)


def _make_stratified_loader(config, *, total_samples: int):
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
    )
    ranges = list(
        zip(
            raw_dataset.episode_data_index["from"].tolist(),
            raw_dataset.episode_data_index["to"].tolist(),
            strict=True,
        )
    )
    episode_ids = [int(value) for value in raw_dataset.episodes]
    per_episode = max(1, int(np.ceil(total_samples / len(ranges))))
    by_episode = []
    for episode, (start, end) in zip(episode_ids, ranges, strict=True):
        indices = _interior_indices(int(start), int(end), per_episode)
        by_episode.append([(int(index), episode) for index in indices])
    selected_pairs = [
        by_episode[episode_index][position]
        for position in range(per_episode)
        for episode_index in range(len(by_episode))
    ][:total_samples]
    transformed = data_loader.transform_dataset(raw_dataset, data_config)
    selected = _FixedIndexDataset(
        transformed,
        [index for index, _ in selected_pairs],
    )
    torch_loader = data_loader.TorchDataLoader(
        selected,
        local_batch_size=config.batch_size,
        shuffle=False,
        num_batches=total_samples // config.batch_size,
        num_workers=0,
        seed=config.seed,
    )
    loader = data_loader.DataLoaderImpl(data_config, torch_loader)
    clusters = np.asarray([episode for _, episode in selected_pairs], dtype=np.int64)
    return iter(loader), clusters


def _replace_memory(observation, tokens: np.ndarray, mask: np.ndarray):
    return dataclasses.replace(
        observation,
        memory_tokens=jnp.asarray(tokens),
        memory_mask=jnp.asarray(mask),
    )


def build_interventions(matched, mismatched) -> dict[str, Any]:
    """Construct paired Mem-0 module/content interventions for one batch."""
    matched_tokens = np.asarray(matched.memory_tokens)
    matched_mask = np.asarray(matched.memory_mask, dtype=np.bool_)
    mismatched_tokens = np.asarray(mismatched.memory_tokens)
    mismatched_mask = np.asarray(mismatched.memory_mask, dtype=np.bool_)
    if matched_tokens.shape != mismatched_tokens.shape or matched_mask.shape != mismatched_mask.shape:
        raise ValueError("matched and mismatched memory layouts differ")
    if matched_tokens.shape[1] != 31:
        raise ValueError(f"expected Mem-0's 1+30 layout, got {matched_tokens.shape}")

    empty_tokens = np.zeros_like(matched_tokens)
    empty_mask = np.zeros_like(matched_mask)

    without_anchor_tokens = matched_tokens.copy()
    without_anchor_tokens[:, 0] = 0
    without_anchor_mask = matched_mask.copy()
    without_anchor_mask[:, 0] = False

    without_sliding_tokens = matched_tokens.copy()
    without_sliding_tokens[:, 1:] = 0
    without_sliding_mask = matched_mask.copy()
    without_sliding_mask[:, 1:] = False

    anchor_replaced_tokens = matched_tokens.copy()
    anchor_replaced_tokens[:, 0] = mismatched_tokens[:, 0]
    anchor_replaced_mask = matched_mask.copy()
    anchor_replaced_mask[:, 0] = mismatched_mask[:, 0]

    sliding_replaced_tokens = matched_tokens.copy()
    sliding_replaced_tokens[:, 1:] = mismatched_tokens[:, 1:]
    sliding_replaced_mask = matched_mask.copy()
    sliding_replaced_mask[:, 1:] = mismatched_mask[:, 1:]

    shuffled_tokens = matched_tokens.copy()
    for batch_index in range(matched_tokens.shape[0]):
        valid_slots = np.flatnonzero(matched_mask[batch_index, 1:]) + 1
        shuffled_tokens[batch_index, valid_slots] = matched_tokens[
            batch_index, valid_slots[::-1]
        ]

    return {
        "matched": matched,
        "empty": _replace_memory(matched, empty_tokens, empty_mask),
        "mismatched": mismatched,
        "without_anchor": _replace_memory(
            matched, without_anchor_tokens, without_anchor_mask
        ),
        "without_sliding": _replace_memory(
            matched, without_sliding_tokens, without_sliding_mask
        ),
        "anchor_replaced": _replace_memory(
            matched, anchor_replaced_tokens, anchor_replaced_mask
        ),
        "sliding_replaced": _replace_memory(
            matched, sliding_replaced_tokens, sliding_replaced_mask
        ),
        "sliding_shuffled": _replace_memory(
            matched, shuffled_tokens, matched_mask
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_json(args.manifest)
    if manifest.get("schema_version") != "emac_mem0_context/v4":
        raise ValueError("expected faithful Mem-0 v4 manifest")
    task_spec = load_task_spec(args.task_config)
    if manifest.get("inputs", {}).get("task_name") != task_spec.task_name:
        raise ValueError("context manifest task does not match task config")
    source_conditions = ("matched",) if args.validation_only else SOURCE_CONDITIONS
    if "matched" not in manifest.get("condition_cycle", []):
        raise ValueError("context manifest is not configured for matched training")
    configs = {
        condition: _make_config(args, manifest, condition, task_spec)
        for condition in source_conditions
    }
    total_samples = args.batch_size * args.num_batches
    loaders = {}
    clusters = None
    for condition in source_conditions:
        loaders[condition], condition_clusters = _make_stratified_loader(
            configs[condition], total_samples=total_samples
        )
        if clusters is None:
            clusters = condition_clusters
        elif not np.array_equal(clusters, condition_clusters):
            raise ValueError("condition loaders selected different validation samples")

    params = model_lib.restore_params(args.checkpoint_dir.resolve() / "params")
    model = configs["matched"].model.load(params)
    model.eval()
    loss_fn = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))
    if args.validation_only:
        validation_losses: list[float] = []
        for batch_index in range(args.num_batches):
            observation, actions = next(loaders["matched"])
            rng = jax.random.key(args.rng_seed + batch_index)
            loss = loss_fn(rng, observation, actions, train=False)
            per_sample = np.asarray(loss, dtype=np.float64).reshape(
                args.batch_size, -1
            ).mean(axis=1)
            validation_losses.extend(per_sample.tolist())
        values = np.asarray(validation_losses, dtype=np.float64)
        assert clusters is not None
        return {
            "schema_version": "mem0_validation_loss/v1",
            "task_name": task_spec.task_name,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "context_program": manifest["representation"]["program"],
            "context_condition": "matched",
            "num_samples": len(clusters),
            "num_source_episodes": int(len(np.unique(clusters))),
            "mean_loss": float(np.mean(values)),
            "cluster_bootstrap_95ci": _cluster_bootstrap_ci(
                values,
                clusters,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
            "interpretation": (
                "Held-out action-flow loss is a checkpoint-selection diagnostic, "
                "not an RMBench success-rate result."
            ),
        }
    losses = {condition: [] for condition in INTERVENTIONS}
    for batch_index in range(args.num_batches):
        source_batches = {
            condition: next(loaders[condition]) for condition in SOURCE_CONDITIONS
        }
        matched_observation, actions = source_batches["matched"]
        mismatched_observation, mismatched_actions = source_batches["mismatched"]
        if not np.array_equal(np.asarray(actions), np.asarray(mismatched_actions)):
            raise ValueError("condition loaders produced different action targets")
        observations = build_interventions(matched_observation, mismatched_observation)
        rng = jax.random.key(args.rng_seed + batch_index)
        for condition, observation in observations.items():
            loss = loss_fn(rng, observation, actions, train=False)
            per_sample = np.asarray(loss, dtype=np.float64).reshape(
                args.batch_size, -1
            ).mean(axis=1)
            losses[condition].extend(per_sample.tolist())

    arrays = {
        condition: np.asarray(values, dtype=np.float64)
        for condition, values in losses.items()
    }
    assert clusters is not None
    comparisons = {
        f"matched_minus_{condition}": arrays["matched"] - arrays[condition]
        for condition in INTERVENTIONS
        if condition != "matched"
    }
    summaries = {}
    for index, (name, values) in enumerate(comparisons.items()):
        summaries[name] = {
            "mean": float(np.mean(values)),
            "cluster_bootstrap_95ci": _cluster_bootstrap_ci(
                values,
                clusters,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + index,
            ),
        }
    content_ci = summaries["matched_minus_mismatched"]["cluster_bootstrap_95ci"]
    utility_ci = summaries["matched_minus_empty"]["cluster_bootstrap_95ci"]
    return {
        "schema_version": "mem0_offline_intervention/v1",
        "task_name": task_spec.task_name,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "num_samples": len(clusters),
        "num_source_episodes": int(len(np.unique(clusters))),
        "condition_mean_loss": {
            condition: float(np.mean(values))
            for condition, values in arrays.items()
        },
        "comparisons": summaries,
        "interventions": list(INTERVENTIONS),
        "gates": {
            "matched_better_than_empty": utility_ci[1] < 0.0,
            "matched_better_than_mismatched": content_ci[1] < 0.0,
        },
        "memory_content_used_offline": utility_ci[1] < 0.0 and content_ci[1] < 0.0,
        "interpretation": (
            "This paired held-out loss intervention tests whether the trained Mem-0 "
            "executor uses causal memory content. It is not an RMBench success-rate result."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired held-out Mem-0 memory intervention.")
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--task-config", type=pathlib.Path, required=True)
    parser.add_argument("--config", default="pi05_aloha_pen_uncap_mem0")
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=ROOT / "rmbench_runs/emac_mem0_context_v4/pairing_manifest.json",
    )
    parser.add_argument(
        "--context-bank",
        type=pathlib.Path,
        default=ROOT / "rmbench_runs/emac_mem0_context_v4/context_bank.npz",
    )
    parser.add_argument("--assets-dir", type=pathlib.Path, default=ROOT / "rmbench_assets")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=64)
    parser.add_argument("--rng-seed", type=int, default=20260814)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Evaluate matched held-out loss without requiring a mismatched context bank.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_batches <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("batch-size, num-batches, and bootstrap-samples must be positive")
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
