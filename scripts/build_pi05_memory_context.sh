#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$PROJECT_ROOT/../openpi-libero}"
PYTHON="${PYTHON:-$OPENPI_DIR/.venv/bin/python}"
TASK_ID="${1:?Usage: build_pi05_memory_context.sh <task> <program> <checkpoint>}"
PROGRAM="${2:?Usage: build_pi05_memory_context.sh <task> <program> <checkpoint>}"
BASE_CHECKPOINT="${3:?Usage: build_pi05_memory_context.sh <task> <program> <checkpoint>}"
TASK_SPEC="$HARNESS_ROOT/configs/tasks/${TASK_ID}.json"
PROGRAM_CONFIG="${PROGRAM_CONFIG:-$HARNESS_ROOT/configs/fixed_${PROGRAM}.json}"
[[ -f "$TASK_SPEC" ]] || { echo "Unknown task: $TASK_ID" >&2; exit 2; }
[[ -f "$PROGRAM_CONFIG" ]] || { echo "Unknown program: $PROGRAM" >&2; exit 2; }
[[ -d "$BASE_CHECKPOINT/params" ]] || { echo "Checkpoint has no params: $BASE_CHECKPOINT" >&2; exit 2; }

TASK_NAME="$(
  PYTHONPATH="$HARNESS_ROOT" "$PYTHON" - "$TASK_SPEC" <<'PY'
import sys
from pathlib import Path
from memory_harness.tasks import load_task_spec

print(load_task_spec(Path(sys.argv[1])).task_name)
PY
)"
CONTEXT_ROOT="${CONTEXT_ROOT:-$PROJECT_ROOT/rmbench_runs/emac_${TASK_NAME}_v1}"
TEMPLATE_MANIFEST="${TEMPLATE_MANIFEST:-$CONTEXT_ROOT/task_template.json}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-$CONTEXT_ROOT/${PROGRAM}_context_manifest.json}"
OUTPUT_BANK="${OUTPUT_BANK:-$CONTEXT_ROOT/${PROGRAM}_context_bank.npz}"
OUTPUT_AUDIT="${OUTPUT_AUDIT:-$CONTEXT_ROOT/${PROGRAM}_context_audit.json}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/rmbench_lerobot_data}"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_ROOT/rmbench_assets}"
GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CHUNK_STRIDE="${CHUNK_STRIDE:-1}"
[[ "$CHUNK_STRIDE" == 1 ]] || {
  echo "Faithful Mem-0 context generation requires CHUNK_STRIDE=1" >&2
  exit 2
}
[[ -s "$TEMPLATE_MANIFEST" ]] || { echo "Missing task template: $TEMPLATE_MANIFEST" >&2; exit 2; }
for output in "$OUTPUT_MANIFEST" "$OUTPUT_BANK" "$OUTPUT_AUDIT"; do
  [[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
done

args=(
  "$PYTHON" -m memory_harness.build_training_data
  --task-config "$TASK_SPEC"
  --template-manifest "$TEMPLATE_MANIFEST"
  --base-checkpoint "$BASE_CHECKPOINT"
  --config pi05_aloha_pen_uncap_mem0
  --program-config "$PROGRAM_CONFIG"
  --assets-dir "$ASSETS_DIR"
  --hf-lerobot-home "$HF_LEROBOT_HOME"
  --chunk-stride "$CHUNK_STRIDE"
  --batch-size "$BATCH_SIZE"
  --output-manifest "$OUTPUT_MANIFEST"
  --output-bank "$OUTPUT_BANK"
  --output-audit "$OUTPUT_AUDIT"
)
printf 'RUN'
printf ' %q' /usr/bin/env "CUDA_VISIBLE_DEVICES=$GPU_ID" "${args[@]}"
printf '\n'
[[ "${DRY_RUN:-0}" == 0 ]] || exit 0

source "$PROJECT_ROOT/scripts/configure_openpi_cuda_env.sh"
export CUDA_VISIBLE_DEVICES="$GPU_ID" HF_LEROBOT_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$HARNESS_ROOT:$PROJECT_ROOT:$OPENPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PROJECT_ROOT/.cache/jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
"${args[@]}"

"$PYTHON" - "$OUTPUT_MANIFEST" "$OUTPUT_BANK" "$OUTPUT_AUDIT" "$TASK_NAME" "$PROGRAM" \
  "$TEMPLATE_MANIFEST" <<'PY'
import json
import pathlib
import sys
import numpy as np

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
audit = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
template = json.loads(pathlib.Path(sys.argv[6]).read_text(encoding="utf-8"))
with np.load(sys.argv[2]) as bank:
    shape = tuple(bank["tokens"].shape)
expected_items = sum(
    int(row["end_frame"]) - int(row["start_frame"])
    for row in template["segments"]
)
if manifest.get("inputs", {}).get("task_name") != sys.argv[4]:
    raise SystemExit("generated context task mismatch")
if manifest.get("representation", {}).get("program") != sys.argv[5]:
    raise SystemExit("generated context program mismatch")
if manifest.get("inputs", {}).get("chunk_stride") != 1:
    raise SystemExit("generated context is not environment-step aligned")
if not audit.get("ready_for_adapter_training"):
    raise SystemExit("generated context audit failed")
if shape[1:] != (31, 2048):
    raise SystemExit(f"generated context layout mismatch: {shape}")
if shape[0] != expected_items:
    raise SystemExit(f"generated context is incomplete: items={shape[0]}, expected={expected_items}")
print(f"CONTEXT_READY items={shape[0]} layout={shape[1:]}")
PY
