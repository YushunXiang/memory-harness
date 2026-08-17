#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
TASK_ID="${1:?Usage: run_clean_pi05_rmbench.sh <task>}"
TASK_SPEC="$HARNESS_ROOT/configs/tasks/${TASK_ID}.json"
PYTHON="${PYTHON:-$PROJECT_ROOT/../openpi-libero/.venv/bin/python}"
[[ -f "$TASK_SPEC" ]] || { echo "Unknown task: $TASK_ID" >&2; exit 2; }

readarray -t task_fields < <(
  PYTHONPATH="$HARNESS_ROOT" "$PYTHON" - "$TASK_SPEC" <<'PY'
import sys
from pathlib import Path
from memory_harness.tasks import load_task_spec

spec = load_task_spec(Path(sys.argv[1]))
for value in (
    spec.task_name,
    spec.task_config,
    spec.prompt,
    spec.max_steps,
    int(spec.paired_layout_protocol),
    spec.asset_id,
    spec.tmc,
):
    print(value)
PY
)
TASK_NAME="${task_fields[0]}"
TASK_CONFIG="${task_fields[1]}"
TASK_PROMPT="${task_fields[2]}"
MAX_STEPS="${task_fields[3]}"
PAIRED_LAYOUT_PROTOCOL="${task_fields[4]}"
ASSET_ID="${task_fields[5]}"
TMC="${task_fields[6]}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
[[ -n "$CHECKPOINT_DIR" && -d "$CHECKPOINT_DIR" ]] || {
  echo "Set CHECKPOINT_DIR to a trained ${TASK_NAME} π0.5 baseline checkpoint" >&2
  exit 2
}
ORACLE_SUBTASK_DIAGNOSTIC="${ORACLE_SUBTASK_DIAGNOSTIC:-0}"
[[ "$ORACLE_SUBTASK_DIAGNOSTIC" == 0 || "$ORACLE_SUBTASK_DIAGNOSTIC" == 1 ]] || {
  echo "ORACLE_SUBTASK_DIAGNOSTIC must be 0 or 1" >&2
  exit 2
}
if [[ "$ORACLE_SUBTASK_DIAGNOSTIC" == 1 && "$TMC" != "M(n)" ]]; then
  echo "ORACLE_SUBTASK_DIAGNOSTIC is only valid for M(n) tasks" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-emac_${TASK_NAME}_clean_none_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/rmbench_runs/$RUN_ID}"
[[ ! -e "$OUT_DIR" ]] || { echo "Output already exists: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR/configs"
FROZEN_TASK_SPEC="$OUT_DIR/configs/task.json"
FROZEN_PROGRAM_CONFIG="$OUT_DIR/configs/fixed_none.json"
install -m 0644 "$TASK_SPEC" "$FROZEN_TASK_SPEC"
install -m 0644 "$HARNESS_ROOT/configs/fixed_none.json" "$FROZEN_PROGRAM_CONFIG"

export CHECKPOINT_DIR RUN_ID OUT_DIR TASK_NAME TASK_CONFIG MAX_STEPS
export NUM_EPISODES="${NUM_EPISODES:-3}"
export SEED="${SEED:-100000}"
export POLICY_SEED_BASE="${POLICY_SEED_BASE:-120000}"
export CONFIG=pi05_aloha_pen_uncap
export POLICY_CONFIG=pi05_aloha_pen_uncap
export POLICY_ASSET_ID="$ASSET_ID"
export POLICY_ASSETS_DIR="${POLICY_ASSETS_DIR:-$PROJECT_ROOT/rmbench_assets}"
export POLICY_ADAPT_TO_PI=0
export PAIRED_LAYOUT_PROTOCOL
export RMBENCH_PHASE_AWARE_SUBTASK_PROMPT="$ORACLE_SUBTASK_DIAGNOSTIC"
export RMBENCH_PROMPT="$TASK_PROMPT"
export RMBENCH_PROMPT_SCHEDULE=
if [[ "$ORACLE_SUBTASK_DIAGNOSTIC" == 1 ]]; then
  export RMBENCH_PROMPT_PROTOCOL=diagnostic_spatial
else
  export RMBENCH_PROMPT_PROTOCOL=main
fi
export TASK_STATE_TRACE_FREQUENCY=10
export EXECUTE_ACTION_CHUNK_STEPS=10

bash "$PROJECT_ROOT/run_rmbench_baseline_local.sh"

validate_args=(
  -m memory_harness.validate_clean_run
  --run-dir "$OUT_DIR"
  --program-config "$FROZEN_PROGRAM_CONFIG"
  --task-config "$FROZEN_TASK_SPEC"
)
[[ "$ORACLE_SUBTASK_DIAGNOSTIC" == 0 ]] || validate_args+=(--oracle-subtask-diagnostic)
PYTHONPATH="$HARNESS_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}" \
  "$PYTHON" "${validate_args[@]}"
