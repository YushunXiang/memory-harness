#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
TASK_ID="${1:?Usage: run_fixed_pi05_rmbench.sh <task> <architecture>}"
PROGRAM="${2:?Usage: run_fixed_pi05_rmbench.sh <task> <architecture>}"
LIVE_TASK_SPEC="$HARNESS_ROOT/configs/tasks/${TASK_ID}.json"
PYTHON="${PYTHON:-$PROJECT_ROOT/../openpi-libero/.venv/bin/python}"

[[ -f "$LIVE_TASK_SPEC" ]] || { echo "Unknown task: $TASK_ID" >&2; exit 2; }
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
[[ -n "$CHECKPOINT_DIR" && -d "$CHECKPOINT_DIR" ]] || {
  echo "Set CHECKPOINT_DIR to a trained π0.5 Mem-0 checkpoint" >&2
  exit 2
}
CANDIDATE_SUITE="${MEMORY_CANDIDATE_SUITE:-}"
if [[ -n "$CANDIDATE_SUITE" ]]; then
  CANDIDATE_SUITE="$(realpath "$CANDIDATE_SUITE")"
  PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.candidate_suite \
    validate-checkpoint --suite "$CANDIDATE_SUITE" \
    --checkpoint "$CHECKPOINT_DIR" >/dev/null
  RUNTIME_SNAPSHOT="$CANDIDATE_SUITE/runtime"
  CONFIG_SNAPSHOT="$CANDIDATE_SUITE/experiment_configs"
else
  RUNTIME_SNAPSHOT="$CHECKPOINT_DIR/runtime"
  CONFIG_SNAPSHOT="$CHECKPOINT_DIR/experiment_configs"
fi
[[ -s "$RUNTIME_SNAPSHOT/runtime_manifest.json" ]] || {
  echo "Missing frozen memory runtime: $RUNTIME_SNAPSHOT" >&2
  exit 2
}
[[ -s "$CONFIG_SNAPSHOT/config_manifest.json" ]] || {
  echo "Missing frozen experiment configs: $CONFIG_SNAPSHOT" >&2
  exit 2
}
PYTHONPATH="$RUNTIME_SNAPSHOT:$HARNESS_ROOT" "$PYTHON" - "$CONFIG_SNAPSHOT" <<'PY'
import sys
from pathlib import Path
from memory_harness.config_snapshot import validate_config_snapshot

validate_config_snapshot(Path(sys.argv[1]))
PY

TASK_SPEC="$CONFIG_SNAPSHOT/tasks/${TASK_ID}.json"
ARCHITECTURE_CONFIG="$CONFIG_SNAPSHOT/architectures/fixed_${PROGRAM}.json"
[[ -f "$TASK_SPEC" ]] || { echo "Task is absent from frozen configs: $TASK_ID" >&2; exit 2; }
[[ -f "$ARCHITECTURE_CONFIG" ]] || {
  echo "Architecture is absent from frozen configs: $PROGRAM" >&2
  exit 2
}

readarray -t task_fields < <(
  PYTHONPATH="$RUNTIME_SNAPSHOT:$HARNESS_ROOT" "$PYTHON" - "$TASK_SPEC" <<'PY'
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

RUN_ID="${RUN_ID:-emac_${TASK_NAME}_${PROGRAM}_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/rmbench_runs/$RUN_ID}"
[[ ! -e "$OUT_DIR" ]] || { echo "Output already exists: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR/configs/architectures"
cp -R "$RUNTIME_SNAPSHOT" "$OUT_DIR/runtime"
cp -R "$CONFIG_SNAPSHOT" "$OUT_DIR/experiment_configs"
if [[ -n "$CANDIDATE_SUITE" ]]; then
  install -m 0644 "$CANDIDATE_SUITE/candidate_suite_manifest.json" \
    "$OUT_DIR/candidate_suite_manifest.json"
fi
FROZEN_TASK_SPEC="$OUT_DIR/configs/task.json"
FROZEN_ARCHITECTURE_CONFIG="$OUT_DIR/configs/architectures/fixed_${PROGRAM}.json"
install -m 0644 "$TASK_SPEC" "$FROZEN_TASK_SPEC"
install -m 0644 "$ARCHITECTURE_CONFIG" "$FROZEN_ARCHITECTURE_CONFIG"

EXECUTOR_CONFIG_RELATIVE="$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["executor_program"])' "$ARCHITECTURE_CONFIG"
)"
PLANNER_KIND="$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["planner"])' "$ARCHITECTURE_CONFIG"
)"
PLANNER_MODEL="$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["planner_model"] or "")' "$ARCHITECTURE_CONFIG"
)"
EXECUTOR_CONFIG="$(cd "$(dirname "$ARCHITECTURE_CONFIG")" && realpath "$EXECUTOR_CONFIG_RELATIVE")"
install -m 0644 "$EXECUTOR_CONFIG" "$OUT_DIR/configs/$(basename "$EXECUTOR_CONFIG")"

"$PYTHON" "$PROJECT_ROOT/scripts/audit_openpi_memory_checkpoint.py" \
  --mode mem0 --checkpoint-dir "$CHECKPOINT_DIR" \
  --output "$CHECKPOINT_DIR/mem0_checkpoint_audit.json" >/dev/null

export CHECKPOINT_DIR RUN_ID OUT_DIR TASK_NAME TASK_CONFIG MAX_STEPS
export PYTHONPATH="$OUT_DIR/runtime:$HARNESS_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"
export NUM_EPISODES="${NUM_EPISODES:-3}"
export SEED="${SEED:-100000}"
export POLICY_SEED_BASE="${POLICY_SEED_BASE:-120000}"
export CONFIG=pi05_aloha_pen_uncap_mem0
export POLICY_CONFIG=pi05_aloha_pen_uncap_mem0
export POLICY_ASSET_ID="$ASSET_ID"
export POLICY_ASSETS_DIR="${POLICY_ASSETS_DIR:-$PROJECT_ROOT/rmbench_assets}"
export POLICY_ADAPT_TO_PI=0
export PAIRED_LAYOUT_PROTOCOL
export RMBENCH_PHASE_AWARE_SUBTASK_PROMPT=0
export RMBENCH_PROMPT="$TASK_PROMPT"
export RMBENCH_PROMPT_SCHEDULE=
export RMBENCH_PROMPT_PROTOCOL=main
export TASK_STATE_TRACE_FREQUENCY=10
export EXECUTE_ACTION_CHUNK_STEPS=10
export MEMORY_ARCHITECTURE_CONFIG="$FROZEN_ARCHITECTURE_CONFIG"
export MEMORY_ARCHITECTURE_AUDIT_LOG="$OUT_DIR/memory_architecture_audit.jsonl"
export MEMORY_PLANNER_BASE_URL="${MEMORY_PLANNER_BASE_URL:-http://127.0.0.1:8123/v1}"
export MEMORY_PLANNER_MODEL="$PLANNER_MODEL"
export MEMORY_PLANNER_SEED_BASE="${MEMORY_PLANNER_SEED_BASE:-130000}"
export MEMORY_PLANNER_BOUNDARY_MODE=oracle_prompt_change
export MEMORY_PLANNER_GLOBAL_TASK="$TASK_PROMPT"
ORACLE_PHASE_DIAGNOSTIC=0
if [[ "$PLANNER_KIND" == "mem0" ]]; then
  export RMBENCH_PHASE_AWARE_SUBTASK_PROMPT=1
  ORACLE_PHASE_DIAGNOSTIC=1
fi

bash "$PROJECT_ROOT/run_rmbench_baseline_local.sh"

if [[ "$TASK_NAME" == put_back_block ]]; then
  PYTHONPATH="$HARNESS_ROOT:$OUT_DIR/runtime:$PROJECT_ROOT:${PYTHONPATH:-}" \
    "$PYTHON" -m memory_harness.put_back_progress \
    --episodes "$OUT_DIR/episodes.jsonl" \
    --output "$OUT_DIR/subtask_summary.json"
fi

validate_args=(
  -m memory_harness.validate_fixed_run
  --run-dir "$OUT_DIR"
  --architecture-config "$FROZEN_ARCHITECTURE_CONFIG"
  --task-config "$FROZEN_TASK_SPEC"
  --audit-log "$MEMORY_ARCHITECTURE_AUDIT_LOG"
)
[[ "$ORACLE_PHASE_DIAGNOSTIC" == 0 ]] || validate_args+=(--diagnostic-oracle-phase)
PYTHONPATH="$HARNESS_ROOT:$OUT_DIR/runtime:$PROJECT_ROOT:${PYTHONPATH:-}" \
  "$PYTHON" "${validate_args[@]}"
