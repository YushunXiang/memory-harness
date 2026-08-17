#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$PROJECT_ROOT/../openpi-libero}"
PYTHON="${PYTHON:-$OPENPI_DIR/.venv/bin/python}"
TASK_ID="${1:?Usage: run_pi05_baseline_train.sh <task>}"
TASK_SPEC="$HARNESS_ROOT/configs/tasks/${TASK_ID}.json"
[[ -f "$TASK_SPEC" ]] || { echo "Unknown task: $TASK_ID" >&2; exit 2; }

readarray -t task_fields < <(
  PYTHONPATH="$HARNESS_ROOT" "$PYTHON" - "$TASK_SPEC" <<'PY'
import sys
from pathlib import Path
from memory_harness.tasks import load_task_spec

spec = load_task_spec(Path(sys.argv[1]))
for value in (spec.task_name, spec.prompt, spec.repo_id, spec.asset_id):
    print(value)
PY
)
TASK_NAME="${task_fields[0]}"
TASK_PROMPT="${task_fields[1]}"
REPO_ID="${task_fields[2]}"
ASSET_ID="${task_fields[3]}"
CONTEXT_ROOT="${CONTEXT_ROOT:-$PROJECT_ROOT/rmbench_runs/emac_${TASK_NAME}_v1}"
TASK_TEMPLATE="${TASK_TEMPLATE:-$CONTEXT_ROOT/task_template.json}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/rmbench_lerobot_data}"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_ROOT/rmbench_assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$PROJECT_ROOT/rmbench_checkpoints}"
WEIGHT_PARAMS="${WEIGHT_PARAMS:-$PROJECT_ROOT/../rhos_cobot/.worktrees/openpi/models/pi05_base/params}"
GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-28}"
OPTIMIZER_UPDATES="${OPTIMIZER_UPDATES:-1200}"
NUM_TRAIN_STEPS="$((ACCUMULATE_STEPS * OPTIMIZER_UPDATES))"
SAVE_EVERY_UPDATES="${SAVE_EVERY_UPDATES:-200}"
SAVE_INTERVAL="$((ACCUMULATE_STEPS * SAVE_EVERY_UPDATES))"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
CONFIG=pi05_aloha_pen_uncap_mem0_control

[[ -s "$TASK_TEMPLATE" ]] || { echo "Missing task template: $TASK_TEMPLATE" >&2; exit 2; }
[[ -d "$HF_LEROBOT_HOME/$REPO_ID" ]] || { echo "Missing dataset: $HF_LEROBOT_HOME/$REPO_ID" >&2; exit 2; }
[[ -s "$ASSETS_DIR/$ASSET_ID/norm_stats.json" ]] || { echo "Missing norm stats for $ASSET_ID" >&2; exit 2; }
[[ -d "$WEIGHT_PARAMS" ]] || { echo "Missing weight params: $WEIGHT_PARAMS" >&2; exit 2; }

readarray -t train_episode_ids < <(
  "$PYTHON" - "$TASK_TEMPLATE" "$TASK_NAME" <<'PY'
import json
import pathlib
import sys

template = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if template.get("schema_version") != "memory_harness.task_template/v1":
    raise SystemExit("unexpected task template schema")
if template.get("task_name") != sys.argv[2]:
    raise SystemExit("task template does not match requested task")
for episode in template["train_lerobot_episode_ids"]:
    print(int(episode))
PY
)
[[ "${#train_episode_ids[@]}" -gt 1 ]] || { echo "Training episode list is empty" >&2; exit 2; }

EXP_NAME="${EXP_NAME:-emac_${TASK_NAME}_native_none_u${OPTIMIZER_UPDATES}_b${BATCH_SIZE}a${ACCUMULATE_STEPS}_$(date -u +%Y%m%dT%H%M%SZ)}"
TRAINING_LOG="${TRAINING_LOG:-$PROJECT_ROOT/rmbench_runs/training_logs/$EXP_NAME.log}"
RESUME="${RESUME:-0}"
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || { echo "RESUME must be 0 or 1" >&2; exit 2; }
if [[ "${DRY_RUN:-0}" == 0 ]]; then
  if [[ "$RESUME" == 0 ]]; then
    [[ ! -e "$TRAINING_LOG" ]] || { echo "Training log already exists: $TRAINING_LOG" >&2; exit 2; }
  else
    [[ -d "$CHECKPOINT_BASE_DIR/$CONFIG/$EXP_NAME" ]] || {
      echo "Cannot resume missing experiment: $CHECKPOINT_BASE_DIR/$CONFIG/$EXP_NAME" >&2
      exit 2
    }
  fi
  mkdir -p "$(dirname "$TRAINING_LOG")"
  if [[ "$RESUME" == 1 ]]; then
    exec > >(tee -a "$TRAINING_LOG") 2>&1
  else
    exec > >(tee "$TRAINING_LOG") 2>&1
  fi
fi

args=(
  "$PYTHON" "$OPENPI_DIR/scripts/train.py" "$CONFIG"
  --exp-name "$EXP_NAME"
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  --data.repo-id "$REPO_ID"
  --data.assets.assets-dir "$ASSETS_DIR"
  --data.assets.asset-id "$ASSET_ID"
  --data.default-prompt "$TASK_PROMPT"
  --data.no-adapt-to-pi
  --data.episode-ids "${train_episode_ids[@]}"
  --weight-loader.params-path "$WEIGHT_PARAMS"
  --num-train-steps "$NUM_TRAIN_STEPS"
  --save-interval "$SAVE_INTERVAL"
  --keep-period "$SAVE_INTERVAL"
  --log-interval "$ACCUMULATE_STEPS"
  --batch-size "$BATCH_SIZE"
  --num-workers 0
  --fsdp-devices 1
  --lr-schedule.peak-lr "$LEARNING_RATE"
  --lr-schedule.decay-lr "$LEARNING_RATE"
  --optimizer.accumulate-steps "$ACCUMULATE_STEPS"
  --no-wandb-enabled
)
[[ "$RESUME" == 0 ]] || args+=(--resume)

printf 'RUN'
printf ' %q' /usr/bin/env "CUDA_VISIBLE_DEVICES=$GPU_ID" "${args[@]}"
printf '\n'
[[ "${DRY_RUN:-0}" == 0 ]] || exit 0

source "$PROJECT_ROOT/scripts/configure_openpi_cuda_env.sh"
export CUDA_VISIBLE_DEVICES="$GPU_ID" HF_LEROBOT_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_ROOT/.cache/openpi}"
export PYTHONPATH="$HARNESS_ROOT:$PROJECT_ROOT:$OPENPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PROJECT_ROOT/.cache/jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
"${args[@]}"

latest_step="$(
  find "$CHECKPOINT_BASE_DIR/$CONFIG/$EXP_NAME" -mindepth 1 -maxdepth 1 -type d \
    -printf '%f\n' | awk '/^[0-9]+$/' | sort -n | tail -1
)"
[[ -n "$latest_step" ]] || { echo "Training completed without checkpoint" >&2; exit 2; }
checkpoint="$CHECKPOINT_BASE_DIR/$CONFIG/$EXP_NAME/$latest_step"
"$PYTHON" "$HARNESS_ROOT/scripts/finalize_pi05_baseline_training.py" \
  --checkpoint "$checkpoint" \
  --task-config "$TASK_SPEC" \
  --task-template "$TASK_TEMPLATE" \
  --initial-weight-params "$WEIGHT_PARAMS" \
  --config "$CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --accumulate-steps "$ACCUMULATE_STEPS" \
  --optimizer-updates "$OPTIMIZER_UPDATES" \
  --learning-rate "$LEARNING_RATE" \
  --training-log "$TRAINING_LOG"
echo "TRAINED_CHECKPOINT=$checkpoint"
