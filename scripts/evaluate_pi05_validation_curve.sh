#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$PROJECT_ROOT/../openpi-libero}"
PYTHON="${PYTHON:-$OPENPI_DIR/.venv/bin/python}"
TASK_ID="${1:?Usage: evaluate_pi05_validation_curve.sh <task> <checkpoint-run-dir>}"
CHECKPOINT_RUN_DIR="${2:?Usage: evaluate_pi05_validation_curve.sh <task> <checkpoint-run-dir>}"
TASK_SPEC="$HARNESS_ROOT/configs/tasks/${TASK_ID}.json"
[[ -f "$TASK_SPEC" ]] || { echo "Unknown task: $TASK_ID" >&2; exit 2; }
[[ -d "$CHECKPOINT_RUN_DIR" ]] || { echo "Missing checkpoint run: $CHECKPOINT_RUN_DIR" >&2; exit 2; }

TASK_NAME="$(
  PYTHONPATH="$HARNESS_ROOT" "$PYTHON" - "$TASK_SPEC" <<'PY'
import sys
from pathlib import Path
from memory_harness.tasks import load_task_spec

print(load_task_spec(Path(sys.argv[1])).task_name)
PY
)"
CONTEXT_ROOT="${CONTEXT_ROOT:-$PROJECT_ROOT/rmbench_runs/emac_${TASK_NAME}_v1}"
MANIFEST="${MANIFEST:-$CONTEXT_ROOT/none_context_manifest.json}"
CONTEXT_BANK="${CONTEXT_BANK:-$CONTEXT_ROOT/none_context_bank.npz}"
OUTPUT="${OUTPUT:-$CONTEXT_ROOT/none_validation_curve.json}"
GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_BATCHES="${NUM_BATCHES:-20}"

[[ -s "$MANIFEST" ]] || { echo "Missing context manifest: $MANIFEST" >&2; exit 2; }
[[ -s "$CONTEXT_BANK" ]] || { echo "Missing context bank: $CONTEXT_BANK" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
mapfile -t steps < <(
  find "$CHECKPOINT_RUN_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '/^[0-9]+$/' | sort -n
)
[[ "${#steps[@]}" -gt 0 ]] || { echo "No checkpoints found in $CHECKPOINT_RUN_DIR" >&2; exit 2; }

source "$PROJECT_ROOT/scripts/configure_openpi_cuda_env.sh"
export CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/rmbench_lerobot_data}"
export PYTHONPATH="$HARNESS_ROOT:$PROJECT_ROOT:$OPENPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PROJECT_ROOT/.cache/jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1

results=()
for step in "${steps[@]}"; do
  checkpoint="$CHECKPOINT_RUN_DIR/$step"
  [[ -d "$checkpoint/params" ]] || { echo "Checkpoint has no params: $checkpoint" >&2; exit 2; }
  result="$CONTEXT_ROOT/none_validation_step${step}.json"
  [[ ! -e "$result" ]] || { echo "Output already exists: $result" >&2; exit 2; }
  results+=("$result")
  "$PYTHON" "$HARNESS_ROOT/scripts/evaluate_mem0_offline.py" \
    --checkpoint-dir "$checkpoint" \
    --task-config "$TASK_SPEC" \
    --manifest "$MANIFEST" \
    --context-bank "$CONTEXT_BANK" \
    --validation-only \
    --batch-size "$BATCH_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --output "$result"
done

"$PYTHON" - "$OUTPUT" "${results[@]}" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
rows = [json.loads(pathlib.Path(path).read_text(encoding="utf-8")) for path in sys.argv[2:]]
if any(row.get("schema_version") != "mem0_validation_loss/v1" for row in rows):
    raise SystemExit("unexpected validation result schema")
selected = min(rows, key=lambda row: row["mean_loss"])
payload = {
    "schema_version": "memory_harness.validation_curve/v1",
    "selection_metric": "held_out_action_flow_mean_loss",
    "selected_checkpoint": selected["checkpoint_dir"],
    "selected_mean_loss": selected["mean_loss"],
    "points": [
        {
            "checkpoint_dir": row["checkpoint_dir"],
            "mean_loss": row["mean_loss"],
            "cluster_bootstrap_95ci": row["cluster_bootstrap_95ci"],
            "num_samples": row["num_samples"],
        }
        for row in rows
    ],
    "interpretation": (
        "The selected checkpoint minimizes a fixed held-out action-flow loss. "
        "RMBench rollout success remains the final endpoint."
    ),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
