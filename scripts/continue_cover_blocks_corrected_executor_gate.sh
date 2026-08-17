#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/../openpi-libero/.venv/bin/python}"
SIGNAL="${1:?Usage: continue_cover_blocks_corrected_executor_gate.sh <signal> <native-checkpoint>}"
NATIVE_CHECKPOINT="${2:?Missing native checkpoint}"
DRY_RUN="${DRY_RUN:-0}"
GPU_ID="${GPU_ID:-1}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-28}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CONTEXT_ROOT="${CONTEXT_ROOT:-$PROJECT_ROOT/rmbench_runs/emac_cover_blocks_v2_subtask_prompt}"
TASK_TEMPLATE="${TASK_TEMPLATE:-$CONTEXT_ROOT/task_template.json}"
CHECKPOINT_ROOT="${CHECKPOINT_BASE_DIR:-/tmp/memory-harness-checkpoints}"
RUN_ROOT="${RUN_ROOT:-/tmp/rmbench_runs}"
DATE_TAG="${DATE_TAG:-20260815}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$PROJECT_ROOT/../rhos_cobot/.worktrees/openpi/models/pi05_base}"

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
for path in "$SIGNAL" "$NATIVE_CHECKPOINT" "$TASK_TEMPLATE" "$BASE_CHECKPOINT/params"; do
  [[ -e "$path" ]] || { echo "Missing Cover Blocks continuation input: $path" >&2; exit 2; }
done

read_signal() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("schema_version") != "memory_harness.executor_run_signal/v2":
    raise SystemExit("unsupported executor-run signal")
if value.get("evidence_scope") != "executor_skill_diagnostic_only":
    raise SystemExit("Cover Blocks continuation requires diagnostic-only executor evidence")
print(int(bool(value.get("observable_executor_signal"))))
PY
}

run() {
  printf 'RUN'; printf ' %q' "$@"; printf '\n'
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

has_signal="$(read_signal "$SIGNAL")"
BUDGET=1200
if [[ "$has_signal" == 0 ]]; then
  NATIVE_EXP="emac_cover_blocks_subtask_native_none_plus1800_to_u3000_b${BATCH_SIZE}a${ACCUMULATE_STEPS}_${DATE_TAG}"
  NATIVE_U3000="$CHECKPOINT_ROOT/pi05_aloha_pen_uncap_mem0_control/$NATIVE_EXP/50399"
  run env \
    PYTHON="$PYTHON" CONTEXT_ROOT="$CONTEXT_ROOT" TASK_TEMPLATE="$TASK_TEMPLATE" \
    CHECKPOINT_BASE_DIR="$CHECKPOINT_ROOT" EXP_NAME="$NATIVE_EXP" \
    TRAINING_LOG="/tmp/${NATIVE_EXP}.log" GPU_ID="$GPU_ID" \
    BATCH_SIZE="$BATCH_SIZE" ACCUMULATE_STEPS="$ACCUMULATE_STEPS" \
    OPTIMIZER_UPDATES=1800 SAVE_EVERY_UPDATES=300 \
    WEIGHT_PARAMS="$NATIVE_CHECKPOINT/params" \
    bash "$HARNESS_ROOT/scripts/run_pi05_baseline_train.sh" cover_blocks
  [[ "$DRY_RUN" == 1 || -s "$NATIVE_U3000/memory_training_manifest.json" ]] || {
    echo "Missing extended Cover Blocks native checkpoint: $NATIVE_U3000" >&2
    exit 2
  }
  EXTENDED_RUN="emac_cover_blocks_subtask_native_none_u3000_oracle_gate3_${DATE_TAG}"
  run env \
    PYTHON="$PYTHON" CHECKPOINT_DIR="$NATIVE_U3000" ORACLE_SUBTASK_DIAGNOSTIC=1 \
    NUM_EPISODES=3 SEED=100000 POLICY_SEED_BASE=120000 GPU_ID="$GPU_ID" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" RUN_ID="$EXTENDED_RUN" \
    OUT_DIR="$RUN_ROOT/$EXTENDED_RUN" \
    bash "$HARNESS_ROOT/scripts/run_clean_pi05_rmbench.sh" cover_blocks
  EXTENDED_SIGNAL="/tmp/${EXTENDED_RUN}_signal.json"
  run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.assess_run_signal \
    --run "$RUN_ROOT/$EXTENDED_RUN" --output "$EXTENDED_SIGNAL"
  if [[ "$DRY_RUN" == 1 ]]; then
    has_signal=1
  else
    has_signal="$(read_signal "$EXTENDED_SIGNAL")"
  fi
  if [[ "$has_signal" == 0 ]]; then
    echo "Cover Blocks native u3000 remains at executor floor; planner comparison not interpretable."
    exit 0
  fi
  BUDGET=3000
fi

EMPTY_MANIFEST="$CONTEXT_ROOT/none_context_manifest.json"
EMPTY_BANK="$CONTEXT_ROOT/none_context_bank.npz"
EMPTY_AUDIT="$CONTEXT_ROOT/none_context_audit.json"
if [[ "$DRY_RUN" == 1 ]] || {
  [[ ! -e "$EMPTY_MANIFEST" ]] &&
    [[ ! -e "$EMPTY_BANK" ]] &&
    [[ ! -e "$EMPTY_AUDIT" ]]
}; then
  run env \
    PYTHON="$PYTHON" GPU_ID="$GPU_ID" CONTEXT_ROOT="$CONTEXT_ROOT" \
    TEMPLATE_MANIFEST="$TASK_TEMPLATE" PROGRAM_CONFIG="$HARNESS_ROOT/configs/training_empty_mem0.json" \
    OUTPUT_MANIFEST="$EMPTY_MANIFEST" OUTPUT_BANK="$EMPTY_BANK" OUTPUT_AUDIT="$EMPTY_AUDIT" \
    bash "$HARNESS_ROOT/scripts/build_pi05_memory_context.sh" \
    cover_blocks none "$BASE_CHECKPOINT"
else
  for path in "$EMPTY_MANIFEST" "$EMPTY_BANK" "$EMPTY_AUDIT"; do
    [[ -s "$path" ]] || { echo "Partial empty-context artifact set: $path" >&2; exit 2; }
  done
  echo "REUSE_VALIDATED_BY_TRAINER empty Cover Blocks contexts"
fi

EMPTY_EXP="emac_cover_blocks_subtask_empty_mem0_u${BUDGET}_b${BATCH_SIZE}a${ACCUMULATE_STEPS}_${DATE_TAG}"
EXPECTED_STEP="$((BUDGET * ACCUMULATE_STEPS - 1))"
EMPTY_CHECKPOINT="$CHECKPOINT_ROOT/pi05_aloha_pen_uncap_mem0/$EMPTY_EXP/$EXPECTED_STEP"
run env \
  PYTHON="$PYTHON" GPU_ID="$GPU_ID" OPTIMIZER_UPDATES="$BUDGET" \
  SAVE_EVERY_UPDATES=300 CHECKPOINT_BASE_DIR="$CHECKPOINT_ROOT" \
  EXP_NAME="$EMPTY_EXP" TRAINING_LOG="/tmp/${EMPTY_EXP}.log" \
  WEIGHT_PARAMS="$BASE_CHECKPOINT/params" MANIFEST="$EMPTY_MANIFEST" \
  PAIRING_AUDIT="$EMPTY_AUDIT" CONTEXT_BANK="$EMPTY_BANK" \
  bash "$HARNESS_ROOT/scripts/run_pi05_memory_train.sh" cover_blocks
[[ "$DRY_RUN" == 1 || -s "$EMPTY_CHECKPOINT/memory_training_manifest.json" ]] || {
  echo "Missing budget-matched Cover Blocks empty-memory checkpoint: $EMPTY_CHECKPOINT" >&2
  exit 2
}

run_planner_shard() {
  local condition="$1"
  local num_episodes="$2"
  local seed_start="$3"
  local policy_seed_base="$4"
  local gate_label="$5"
  local run_id="emac_cover_blocks_subtask_${condition}_u${BUDGET}_${gate_label}_${DATE_TAG}"
  command=(
    bash "$HARNESS_ROOT/scripts/run_pi05_cover_blocks_planner_condition.sh"
    --condition "$condition"
    --checkpoint "$EMPTY_CHECKPOINT"
    --num-episodes "$num_episodes"
    --seed-start "$seed_start"
    --policy-seed-base "$policy_seed_base"
    --gpu-id "$GPU_ID"
    --output-dir "$RUN_ROOT/$run_id"
  )
  [[ "$DRY_RUN" == 0 ]] || command+=(--dry-run)
  run "${command[@]}"
}

for condition in no_key key; do
  run_planner_shard "$condition" 3 100000 120000 gate3
done

COMPARISON="/tmp/cover_blocks_subtask_no_key_vs_key_u${BUDGET}_gate3.json"
UTILITY="${COMPARISON%.json}.utility.json"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.compare_fixed_runs \
  --reference-run "$RUN_ROOT/emac_cover_blocks_subtask_no_key_u${BUDGET}_gate3_${DATE_TAG}" \
  --candidate-run "$RUN_ROOT/emac_cover_blocks_subtask_key_u${BUDGET}_gate3_${DATE_TAG}" \
  --output "$COMPARISON"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.utility_gate \
  --comparison "$COMPARISON" --evidence-kind matched_training \
  --output "$UTILITY"

# Gate-3 is an executor-safe screen.  The fixed key/no-key reproduction needs
# a disjoint shared-seed pilot before its direction is reported as π0.5 evidence.
for condition in no_key key; do
  run_planner_shard "$condition" 17 100003 120003 gate17
done

PILOT_COMPARISON="/tmp/cover_blocks_subtask_no_key_vs_key_u${BUDGET}_gate20.json"
PILOT_UTILITY="${PILOT_COMPARISON%.json}.utility.json"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.compare_fixed_runs \
  --reference-run "$RUN_ROOT/emac_cover_blocks_subtask_no_key_u${BUDGET}_gate3_${DATE_TAG}" \
  --candidate-run "$RUN_ROOT/emac_cover_blocks_subtask_key_u${BUDGET}_gate3_${DATE_TAG}" \
  --reference-run "$RUN_ROOT/emac_cover_blocks_subtask_no_key_u${BUDGET}_gate17_${DATE_TAG}" \
  --candidate-run "$RUN_ROOT/emac_cover_blocks_subtask_key_u${BUDGET}_gate17_${DATE_TAG}" \
  --output "$PILOT_COMPARISON"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.utility_gate \
  --comparison "$PILOT_COMPARISON" --evidence-kind matched_training \
  --output "$PILOT_UTILITY"

if [[ "$DRY_RUN" == 1 ]]; then
  pilot_next_action="collect_shared_episodes_to_50"
else
  pilot_next_action="$($PYTHON - "$PILOT_UTILITY" <<'PY'
import json, sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("schema_version") != "memory_harness.utility_decision/v2":
    raise SystemExit("unsupported utility-gate schema")
if value.get("decision_scope") != "single_candidate_utility_requirement":
    raise SystemExit("unexpected utility-gate decision scope")
print(value.get("next_action", ""))
PY
)"
fi

if [[ "$pilot_next_action" != "collect_shared_episodes_to_50" ]]; then
  echo "COVER_BLOCKS_KEY_NO_KEY_GATE20_COMPLETE=$PILOT_COMPARISON"
  exit 0
fi

for condition in no_key key; do
  run_planner_shard "$condition" 30 100020 120020 gate30
done

CONFIRMATION_COMPARISON="/tmp/cover_blocks_subtask_no_key_vs_key_u${BUDGET}_gate50.json"
CONFIRMATION_UTILITY="${CONFIRMATION_COMPARISON%.json}.utility.json"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.compare_fixed_runs \
  --reference-run "$RUN_ROOT/emac_cover_blocks_subtask_no_key_u${BUDGET}_gate3_${DATE_TAG}" \
  --candidate-run "$RUN_ROOT/emac_cover_blocks_subtask_key_u${BUDGET}_gate3_${DATE_TAG}" \
  --reference-run "$RUN_ROOT/emac_cover_blocks_subtask_no_key_u${BUDGET}_gate17_${DATE_TAG}" \
  --candidate-run "$RUN_ROOT/emac_cover_blocks_subtask_key_u${BUDGET}_gate17_${DATE_TAG}" \
  --reference-run "$RUN_ROOT/emac_cover_blocks_subtask_no_key_u${BUDGET}_gate30_${DATE_TAG}" \
  --candidate-run "$RUN_ROOT/emac_cover_blocks_subtask_key_u${BUDGET}_gate30_${DATE_TAG}" \
  --output "$CONFIRMATION_COMPARISON"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.utility_gate \
  --comparison "$CONFIRMATION_COMPARISON" --evidence-kind matched_training \
  --output "$CONFIRMATION_UTILITY"
echo "COVER_BLOCKS_KEY_NO_KEY_GATE50_COMPLETE=$CONFIRMATION_COMPARISON"
