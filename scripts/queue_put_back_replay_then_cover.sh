#!/usr/bin/env bash
set -u

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/../openpi-libero/.venv/bin/python}"
WAIT_PID="${1:?Usage: queue_put_back_replay_then_cover.sh <wait-pid> <gpu-id> <full-run> <empty-run> <native-run>}"
GPU_ID="${2:?Missing GPU id}"
FULL_RUN="${3:?Missing full-memory source run}"
EMPTY_RUN="${4:?Missing empty-memory source run}"
NATIVE_RUN="${5:?Missing native source run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HARNESS_ROOT/artifacts}"
DATE_TAG="${DATE_TAG:-2026-08-16}"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
while [[ -n "$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
  sleep 30
done

replay() {
  local label="$1"
  local source_run="$2"
  local output="$OUTPUT_ROOT/${DATE_TAG}-pi05-put-back-u1200-${label}-progress-replay.json"
  local log="/tmp/${DATE_TAG}-pi05-put-back-u1200-${label}-progress-replay.log"
  if [[ -s "$output" ]]; then
    return
  fi
  if ! CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PROJECT_ROOT:$HARNESS_ROOT" \
    "$PYTHON" -m memory_harness.replay_put_back_progress \
      --source-run "$source_run" \
      --rmbench-root "$PROJECT_ROOT/../RMBench" \
      --gpu-id "$GPU_ID" \
      --output "$output" >"$log" 2>&1; then
    printf 'Put Back progress replay failed for %s; see %s\n' "$label" "$log" >&2
  fi
}

replay full_memory "$FULL_RUN"
replay empty_mask "$EMPTY_RUN"
replay native_none "$NATIVE_RUN"

FULL_OUTPUT="$OUTPUT_ROOT/${DATE_TAG}-pi05-put-back-u1200-full_memory-progress-replay.json"
EMPTY_OUTPUT="$OUTPUT_ROOT/${DATE_TAG}-pi05-put-back-u1200-empty_mask-progress-replay.json"
NATIVE_OUTPUT="$OUTPUT_ROOT/${DATE_TAG}-pi05-put-back-u1200-native_none-progress-replay.json"
SUMMARY_OUTPUT="$OUTPUT_ROOT/${DATE_TAG}-pi05-put-back-u1200-progress-replay-summary.json"
if [[ -s "$FULL_OUTPUT" && -s "$EMPTY_OUTPUT" && -s "$NATIVE_OUTPUT" && ! -e "$SUMMARY_OUTPUT" ]]; then
  PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.summarize_put_back_replays \
    --full-memory "$FULL_OUTPUT" \
    --empty-mask "$EMPTY_OUTPUT" \
    --native-none "$NATIVE_OUTPUT" \
    --output "$SUMMARY_OUTPUT" \
    >/tmp/${DATE_TAG}-pi05-put-back-u1200-progress-replay-summary.log 2>&1 || \
    printf 'Put Back replay summary failed; see /tmp/%s-pi05-put-back-u1200-progress-replay-summary.log\n' "$DATE_TAG" >&2
fi

GPU_ID="$GPU_ID" bash "$HARNESS_ROOT/scripts/run_cover_blocks_corrected_executor_gate.sh"
