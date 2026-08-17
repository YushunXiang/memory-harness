#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/../openpi-libero/.venv/bin/python}"
GPU_ID="${GPU_ID:-1}"
OPTIMIZER_UPDATES="${OPTIMIZER_UPDATES:-1200}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-28}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CONTEXT_ROOT="${CONTEXT_ROOT:-$PROJECT_ROOT/rmbench_runs/emac_cover_blocks_v2_subtask_prompt}"
TASK_TEMPLATE="${TASK_TEMPLATE:-$CONTEXT_ROOT/task_template.json}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/tmp/memory-harness-checkpoints}"
EXP_NAME="${EXP_NAME:-emac_cover_blocks_subtask_native_none_u${OPTIMIZER_UPDATES}_b${BATCH_SIZE}a${ACCUMULATE_STEPS}_20260815}"
TRAINING_LOG="${TRAINING_LOG:-/tmp/${EXP_NAME}.log}"
RUN_ID="${RUN_ID:-emac_cover_blocks_subtask_native_none_u${OPTIMIZER_UPDATES}_oracle_gate3_20260815}"
OUT_DIR="${OUT_DIR:-/tmp/rmbench_runs/$RUN_ID}"
SIGNAL_OUTPUT="${SIGNAL_OUTPUT:-/tmp/${RUN_ID}_signal.json}"
EXPECTED_STEP="$((OPTIMIZER_UPDATES * ACCUMULATE_STEPS - 1))"
CHECKPOINT="$CHECKPOINT_BASE_DIR/pi05_aloha_pen_uncap_mem0_control/$EXP_NAME/$EXPECTED_STEP"

[[ -s "$TASK_TEMPLATE" ]] || { echo "Missing corrected task template: $TASK_TEMPLATE" >&2; exit 2; }
template_sha="$(sha256sum "$TASK_TEMPLATE" | awk '{print $1}')"
[[ "$template_sha" == "7e4a611764bc3e2323a54858c28b68430a84fb40e42c942e86537b7e2ebfcb74" ]] || {
  echo "Corrected Cover Blocks task template hash changed: $template_sha" >&2
  exit 2
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  CONTEXT_ROOT="$CONTEXT_ROOT" TASK_TEMPLATE="$TASK_TEMPLATE" \
    CHECKPOINT_BASE_DIR="$CHECKPOINT_BASE_DIR" EXP_NAME="$EXP_NAME" \
    TRAINING_LOG="$TRAINING_LOG" GPU_ID="$GPU_ID" BATCH_SIZE="$BATCH_SIZE" \
    ACCUMULATE_STEPS="$ACCUMULATE_STEPS" OPTIMIZER_UPDATES="$OPTIMIZER_UPDATES" \
    DRY_RUN=1 bash "$HARNESS_ROOT/scripts/run_pi05_baseline_train.sh" cover_blocks
  printf 'RUN CHECKPOINT_DIR=%q ORACLE_SUBTASK_DIAGNOSTIC=1 NUM_EPISODES=3 SEED=100000 POLICY_SEED_BASE=120000 CUDA_VISIBLE_DEVICES=%q RUN_ID=%q OUT_DIR=%q bash %q cover_blocks\n' \
    "$CHECKPOINT" "$GPU_ID" "$RUN_ID" "$OUT_DIR" "$HARNESS_ROOT/scripts/run_clean_pi05_rmbench.sh"
  printf 'RUN PYTHONPATH=%q %q -m memory_harness.assess_run_signal --run %q --output %q\n' \
    "$HARNESS_ROOT" "$PYTHON" "$OUT_DIR" "$SIGNAL_OUTPUT"
  printf 'RUN bash %q %q %q\n' \
    "$HARNESS_ROOT/scripts/continue_cover_blocks_corrected_executor_gate.sh" \
    "$SIGNAL_OUTPUT" "$CHECKPOINT"
  exit 0
fi

CONTEXT_ROOT="$CONTEXT_ROOT" TASK_TEMPLATE="$TASK_TEMPLATE" \
  CHECKPOINT_BASE_DIR="$CHECKPOINT_BASE_DIR" EXP_NAME="$EXP_NAME" \
  TRAINING_LOG="$TRAINING_LOG" GPU_ID="$GPU_ID" BATCH_SIZE="$BATCH_SIZE" \
  ACCUMULATE_STEPS="$ACCUMULATE_STEPS" OPTIMIZER_UPDATES="$OPTIMIZER_UPDATES" \
  bash "$HARNESS_ROOT/scripts/run_pi05_baseline_train.sh" cover_blocks

[[ -s "$CHECKPOINT/memory_training_manifest.json" ]] || {
  echo "Corrected native checkpoint did not pass training finalization: $CHECKPOINT" >&2
  exit 2
}

CHECKPOINT_DIR="$CHECKPOINT" ORACLE_SUBTASK_DIAGNOSTIC=1 \
  NUM_EPISODES=3 SEED=100000 POLICY_SEED_BASE=120000 \
  CUDA_VISIBLE_DEVICES="$GPU_ID" RUN_ID="$RUN_ID" OUT_DIR="$OUT_DIR" \
  bash "$HARNESS_ROOT/scripts/run_clean_pi05_rmbench.sh" cover_blocks

PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.assess_run_signal \
  --run "$OUT_DIR" --output "$SIGNAL_OUTPUT"
bash "$HARNESS_ROOT/scripts/continue_cover_blocks_corrected_executor_gate.sh" \
  "$SIGNAL_OUTPUT" "$CHECKPOINT"
echo "COVER_BLOCKS_CORRECTED_EXECUTOR_GATE_COMPLETE=$SIGNAL_OUTPUT"
