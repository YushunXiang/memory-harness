#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
RUNNER="$HARNESS_ROOT/scripts/run_official_mem0_m1mix.sh"
PYTHON="$PROJECT_ROOT/.cache/mem0_env/bin/python"

usage() {
  echo "Usage: $0 --asset-root DIR --checkpoint FILE --task TASK --num-episodes N --seed-start N --gpu-id N --output-dir DIR [--dry-run]" >&2
}

ASSET_ROOT=""
CHECKPOINT=""
TASK=""
NUM_EPISODES=""
SEED_START=""
GPU_ID=""
OUTPUT_DIR=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-root) ASSET_ROOT="${2:?Missing value for --asset-root}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?Missing value for --checkpoint}"; shift 2 ;;
    --task) TASK="${2:?Missing value for --task}"; shift 2 ;;
    --num-episodes) NUM_EPISODES="${2:?Missing value for --num-episodes}"; shift 2 ;;
    --seed-start) SEED_START="${2:?Missing value for --seed-start}"; shift 2 ;;
    --gpu-id) GPU_ID="${2:?Missing value for --gpu-id}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?Missing value for --output-dir}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in ASSET_ROOT CHECKPOINT TASK NUM_EPISODES SEED_START GPU_ID OUTPUT_DIR; do
  [[ -n "${!value}" ]] || { usage; echo "Missing required argument for $value" >&2; exit 2; }
done

for condition in full without_anchor without_sliding; do
  command=(
    "$RUNNER"
    --asset-root "$ASSET_ROOT"
    --checkpoint "$CHECKPOINT"
    --task "$TASK"
    --num-episodes "$NUM_EPISODES"
    --seed-start "$SEED_START"
    --gpu-id "$GPU_ID"
    --output-dir "$OUTPUT_DIR/$condition"
    --executor-ablation "$condition"
  )
  if [[ "$DRY_RUN" == 1 ]]; then
    command+=(--dry-run)
  fi
  "${command[@]}"
done

if [[ "$DRY_RUN" == 0 ]]; then
  PYTHONPATH="$HARNESS_ROOT:${PYTHONPATH:-}" "$PYTHON" \
    -m memory_harness.summarize_official_mem0_ablation \
    --run-dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --task "$TASK" \
    --seed-start "$SEED_START" \
    --output "$OUTPUT_DIR/summary.json"
fi
