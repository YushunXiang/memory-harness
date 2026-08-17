#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/../openpi-libero/.venv/bin/python}"
SERVER_PYTHON="${SERVER_PYTHON:-$PROJECT_ROOT/.cache/mem0_env/bin/python}"
PLANNER_BASE_INDEX="${PLANNER_BASE_INDEX:-$PROJECT_ROOT/../VLMEvalKit/models/Qwen/Qwen3-VL-8B-Instruct/model.safetensors.index.json}"
TRAINING_PAIR_MANIFEST="${TRAINING_PAIR_MANIFEST:-$PROJECT_ROOT/rmbench_runs/emac_mem0_planner_key_no_key_data_pair_20260814.json}"
KEY_MODEL="${KEY_MODEL:-$PROJECT_ROOT/rmbench_checkpoints/mem0_planner/cover_blocks_qwen3vl8b_merged_rgb_balanced_v3_checkpoint75}"
NO_KEY_MODEL="${NO_KEY_MODEL:-$PROJECT_ROOT/rmbench_checkpoints/mem0_planner/cover_blocks_no_key_qwen3vl8b_merged}"
KEY_ADAPTER="${KEY_ADAPTER:-$PROJECT_ROOT/rmbench_checkpoints/mem0_planner/cover_blocks_qwen3vl8b_lora_rgb_balanced_v3/checkpoint-75/adapter_model.safetensors}"
NO_KEY_ADAPTER="${NO_KEY_ADAPTER:-$PROJECT_ROOT/rmbench_checkpoints/mem0_planner/cover_blocks_no_key_qwen3vl8b_lora/checkpoint-75/adapter_model.safetensors}"
PORT="${PORT:-8123}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"

usage() {
  echo "Usage: $0 --condition key|no_key --checkpoint DIR --num-episodes N --seed-start N --policy-seed-base N --gpu-id N --output-dir DIR [--dry-run]" >&2
}

CONDITION=""
CHECKPOINT=""
NUM_EPISODES=""
SEED_START=""
POLICY_SEED_BASE=""
GPU_ID=""
OUTPUT_DIR=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --condition) CONDITION="${2:?Missing value for --condition}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?Missing value for --checkpoint}"; shift 2 ;;
    --num-episodes) NUM_EPISODES="${2:?Missing value for --num-episodes}"; shift 2 ;;
    --seed-start) SEED_START="${2:?Missing value for --seed-start}"; shift 2 ;;
    --policy-seed-base) POLICY_SEED_BASE="${2:?Missing value for --policy-seed-base}"; shift 2 ;;
    --gpu-id) GPU_ID="${2:?Missing value for --gpu-id}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?Missing value for --output-dir}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$CONDITION" in
  key)
    ARCHITECTURE=key
    PLANNER_MODEL="$KEY_MODEL"
    PLANNER_ADAPTER="$KEY_ADAPTER"
    SERVED_MODEL_NAME=mem0-cover-blocks-key-planner
    ;;
  no_key)
    ARCHITECTURE=planner_no_key
    PLANNER_MODEL="$NO_KEY_MODEL"
    PLANNER_ADAPTER="$NO_KEY_ADAPTER"
    SERVED_MODEL_NAME=mem0-cover-blocks-no-key-planner
    ;;
  *) usage; echo "--condition must be key or no_key" >&2; exit 2 ;;
esac
[[ "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
[[ "$SEED_START" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ "$POLICY_SEED_BASE" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ "$PORT" =~ ^[1-9][0-9]*$ ]] || { echo "PORT must be positive" >&2; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { usage; exit 2; }
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

for path in "$PYTHON" "$SERVER_PYTHON" "$CHECKPOINT" "$PLANNER_MODEL/model.safetensors.index.json" "$PLANNER_ADAPTER" "$PLANNER_BASE_INDEX" "$TRAINING_PAIR_MANIFEST"; do
  [[ -e "$path" ]] || { echo "Missing planner-condition input: $path" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists: $OUTPUT_DIR" >&2; exit 2; }

SERVER_COMMAND=(
  "$SERVER_PYTHON" "$PROJECT_ROOT/scripts/serve_mem0_planner.py"
  --model "$PLANNER_MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --gpu-id "$GPU_ID"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len 8192
  --max-images 16
)
RUN_ID="$(basename "$OUTPUT_DIR")"
EVAL_COMMAND=(
  env
  PYTHON="$PYTHON"
  CHECKPOINT_DIR="$CHECKPOINT"
  NUM_EPISODES="$NUM_EPISODES"
  SEED="$SEED_START"
  POLICY_SEED_BASE="$POLICY_SEED_BASE"
  GPU_ID="$GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID"
  RUN_ID="$RUN_ID"
  OUT_DIR="$OUTPUT_DIR"
  MEMORY_PLANNER_BASE_URL="http://127.0.0.1:$PORT/v1"
  MEMORY_PLANNER_MODEL="$SERVED_MODEL_NAME"
  bash "$HARNESS_ROOT/scripts/run_fixed_pi05_rmbench.sh"
  cover_blocks "$ARCHITECTURE"
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'RUN'; printf ' %q' "${SERVER_COMMAND[@]}"; printf '\n'
  printf 'RUN'; printf ' %q' "${EVAL_COMMAND[@]}"; printf '\n'
  exit 0
fi

if "$SERVER_PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=2)" >/dev/null 2>&1; then
  echo "Planner port $PORT is already occupied; refusing to use an unaudited server" >&2
  exit 2
fi
SERVER_LOG="${OUTPUT_DIR}.planner_server.log"
setsid "${SERVER_COMMAND[@]}" >"$SERVER_LOG" 2>&1 &
PLANNER_PID=$!
cleanup() {
  kill -TERM -- "-$PLANNER_PID" 2>/dev/null || true
  wait "$PLANNER_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in {1..120}; do
  if "$SERVER_PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=2)" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$PLANNER_PID" 2>/dev/null; then
    echo "Planner server exited before becoming ready" >&2
    tail -n 80 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
[[ "$ready" == 1 ]] || { echo "Planner server did not become ready" >&2; exit 1; }

"${EVAL_COMMAND[@]}"
"$PYTHON" - "$OUTPUT_DIR" "$CONDITION" "$PLANNER_MODEL" "$PLANNER_ADAPTER" \
  "$PLANNER_BASE_INDEX" "$TRAINING_PAIR_MANIFEST" "$SERVED_MODEL_NAME" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, condition, model, adapter, base_index, pair_manifest, served_name = sys.argv[1:]

def sha256(path: str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

payload = {
    "schema_version": "memory_harness.pi05_planner_condition/v1",
    "condition": condition,
    "served_model_name": served_name,
    "planner_model": str(Path(model).resolve()),
    "planner_model_index_sha256": sha256(str(Path(model) / "model.safetensors.index.json")),
    "planner_adapter": str(Path(adapter).resolve()),
    "planner_adapter_sha256": sha256(adapter),
    "planner_base_index": str(Path(base_index).resolve()),
    "planner_base_index_sha256": sha256(base_index),
    "training_pair_manifest": str(Path(pair_manifest).resolve()),
    "training_pair_manifest_sha256": sha256(pair_manifest),
}
path = Path(output) / "planner_condition_manifest.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
