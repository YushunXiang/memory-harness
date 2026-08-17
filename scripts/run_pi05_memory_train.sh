#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$PROJECT_ROOT/../openpi-libero}"
PYTHON="${PYTHON:-$OPENPI_DIR/.venv/bin/python}"
TASK_ID="${1:?Usage: run_pi05_memory_train.sh <task>}"
CONFIG_ROOT="${CONFIG_SNAPSHOT_SOURCE:-$HARNESS_ROOT/configs}"
if [[ -n "${CONFIG_SNAPSHOT_SOURCE:-}" ]]; then
  [[ -s "$CONFIG_ROOT/config_manifest.json" ]] || {
    echo "Missing frozen config source: $CONFIG_ROOT" >&2
    exit 2
  }
fi
TASK_SPEC="$CONFIG_ROOT/tasks/${TASK_ID}.json"
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
MANIFEST="${MANIFEST:-$CONTEXT_ROOT/none_context_manifest.json}"
PAIRING_AUDIT="${PAIRING_AUDIT:-$CONTEXT_ROOT/none_context_audit.json}"
CONTEXT_BANK="${CONTEXT_BANK:-$CONTEXT_ROOT/none_context_bank.npz}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/rmbench_lerobot_data}"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_ROOT/rmbench_assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$PROJECT_ROOT/rmbench_checkpoints}"
WEIGHT_PARAMS="${WEIGHT_PARAMS:-$PROJECT_ROOT/../rhos_cobot/.worktrees/openpi/models/pi05_base/params}"
GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-28}"
OPTIMIZER_UPDATES="${OPTIMIZER_UPDATES:-200}"
NUM_TRAIN_STEPS="$((ACCUMULATE_STEPS * OPTIMIZER_UPDATES))"
SAVE_EVERY_UPDATES="${SAVE_EVERY_UPDATES:-50}"
SAVE_INTERVAL="$((ACCUMULATE_STEPS * SAVE_EVERY_UPDATES))"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
CONFIG=pi05_aloha_pen_uncap_mem0

for path in "$MANIFEST" "$PAIRING_AUDIT" "$CONTEXT_BANK"; do
  [[ -s "$path" ]] || { echo "Missing training input: $path" >&2; exit 2; }
done
[[ -d "$HF_LEROBOT_HOME/$REPO_ID" ]] || { echo "Missing dataset: $HF_LEROBOT_HOME/$REPO_ID" >&2; exit 2; }
[[ -s "$ASSETS_DIR/$ASSET_ID/norm_stats.json" ]] || { echo "Missing norm stats for $ASSET_ID" >&2; exit 2; }
[[ -d "$WEIGHT_PARAMS" ]] || { echo "Missing weight params: $WEIGHT_PARAMS" >&2; exit 2; }

readarray -t context_fields < <(
  "$PYTHON" - "$MANIFEST" "$PAIRING_AUDIT" "$CONTEXT_BANK" "$TASK_NAME" <<'PY'
import json
import pathlib
import sys
import numpy as np

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
audit = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
bank = np.load(sys.argv[3])
task_name = sys.argv[4]
if manifest.get("schema_version") != "emac_mem0_context/v4":
    raise SystemExit("expected emac_mem0_context/v4 manifest")
if manifest.get("token_budget") != 31 or manifest.get("tokens_per_item") != 1:
    raise SystemExit("invalid fixed Mem-0 context layout")
if manifest.get("condition_cycle") != ["matched"]:
    raise SystemExit("training must use matched contexts only")
if manifest.get("inputs", {}).get("task_name") != task_name:
    raise SystemExit("context manifest task does not match requested task")
if not audit.get("ready_for_adapter_training"):
    raise SystemExit("pairing audit is not ready")
if bank["tokens"].shape[1:] != (31, 2048) or bank["masks"].shape[1:] != (31,):
    raise SystemExit("context bank does not use the fixed 31x2048 layout")
print(manifest["representation"]["program"])
for episode in manifest["train_lerobot_episode_ids"]:
    print(int(episode))
PY
)
PROGRAM_NAME="${context_fields[0]}"
train_episode_ids=("${context_fields[@]:1}")
[[ "${#train_episode_ids[@]}" -gt 1 ]] || { echo "Training episode list is empty" >&2; exit 2; }
EXP_NAME="${EXP_NAME:-emac_${TASK_NAME}_${PROGRAM_NAME}_u${OPTIMIZER_UPDATES}_b${BATCH_SIZE}a${ACCUMULATE_STEPS}_$(date -u +%Y%m%dT%H%M%SZ)}"
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
  --data.context-injection-manifest "$MANIFEST"
  --data.context-bank-path "$CONTEXT_BANK"
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
"$PYTHON" "$PROJECT_ROOT/scripts/audit_openpi_memory_checkpoint.py" \
  --mode mem0 --checkpoint-dir "$checkpoint" \
  --output "$checkpoint/mem0_checkpoint_audit.json"
finalize_args=(
  "$PYTHON" "$HARNESS_ROOT/scripts/finalize_pi05_memory_training.py" \
  --checkpoint "$checkpoint" \
  --task-config "$TASK_SPEC" \
  --context-manifest "$MANIFEST" \
  --pairing-audit "$PAIRING_AUDIT" \
  --context-bank "$CONTEXT_BANK" \
  --initial-weight-params "$WEIGHT_PARAMS" \
  --program "$PROGRAM_NAME" \
  --batch-size "$BATCH_SIZE" \
  --accumulate-steps "$ACCUMULATE_STEPS" \
  --optimizer-updates "$OPTIMIZER_UPDATES" \
  --learning-rate "$LEARNING_RATE" \
  --training-log "$TRAINING_LOG"
)
if [[ -n "${PROGRAM_MIGRATION_AUDIT:-}" ]]; then
  [[ -s "$PROGRAM_MIGRATION_AUDIT" ]] || {
    echo "Missing program migration audit: $PROGRAM_MIGRATION_AUDIT" >&2
    exit 2
  }
  finalize_args+=(--program-migration-audit "$PROGRAM_MIGRATION_AUDIT")
fi
if [[ -n "${RUNTIME_SNAPSHOT_SOURCE:-}" ]]; then
  [[ -s "$RUNTIME_SNAPSHOT_SOURCE/runtime_manifest.json" ]] || {
    echo "Missing frozen runtime source: $RUNTIME_SNAPSHOT_SOURCE" >&2
    exit 2
  }
  finalize_args+=(--runtime-snapshot-source "$RUNTIME_SNAPSHOT_SOURCE")
fi
if [[ -n "${CONFIG_SNAPSHOT_SOURCE:-}" ]]; then
  [[ -s "$CONFIG_SNAPSHOT_SOURCE/config_manifest.json" ]] || {
    echo "Missing frozen config source: $CONFIG_SNAPSHOT_SOURCE" >&2
    exit 2
  }
  finalize_args+=(--config-snapshot-source "$CONFIG_SNAPSHOT_SOURCE")
fi
"${finalize_args[@]}"
echo "TRAINED_CHECKPOINT=$checkpoint"
