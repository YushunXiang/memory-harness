#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
RMBENCH_ROOT="$PROJECT_ROOT/../RMBench"
PYTHON="$PROJECT_ROOT/.cache/mem0_env/bin/python"
QWEN_MODEL="$PROJECT_ROOT/../VLMEvalKit/models/Qwen/Qwen3-VL-2B-Instruct"

usage() {
  echo "Usage: $0 --asset-root DIR --checkpoint FILE --task TASK --num-episodes N --seed-start N --gpu-id N --output-dir DIR [--executor-ablation full|without_anchor|without_sliding|without_both] [--dry-run]" >&2
}

ASSET_ROOT=""
CHECKPOINT=""
TASK=""
NUM_EPISODES=""
SEED_START=""
GPU_ID=""
OUTPUT_DIR=""
DRY_RUN=0
EXECUTOR_ABLATION="full"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-root) ASSET_ROOT="${2:?Missing value for --asset-root}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?Missing value for --checkpoint}"; shift 2 ;;
    --task) TASK="${2:?Missing value for --task}"; shift 2 ;;
    --num-episodes) NUM_EPISODES="${2:?Missing value for --num-episodes}"; shift 2 ;;
    --seed-start) SEED_START="${2:?Missing value for --seed-start}"; shift 2 ;;
    --gpu-id) GPU_ID="${2:?Missing value for --gpu-id}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?Missing value for --output-dir}"; shift 2 ;;
    --executor-ablation) EXECUTOR_ABLATION="${2:?Missing value for --executor-ablation}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in ASSET_ROOT CHECKPOINT TASK NUM_EPISODES SEED_START GPU_ID OUTPUT_DIR; do
  [[ -n "${!value}" ]] || { usage; echo "Missing required argument for $value" >&2; exit 2; }
done
[[ "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "--num-episodes must be positive" >&2; exit 2; }
[[ "$SEED_START" =~ ^[0-9]+$ ]] || { echo "--seed-start must be non-negative" >&2; exit 2; }
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "--gpu-id must be non-negative" >&2; exit 2; }
case "$EXECUTOR_ABLATION" in
  full|without_anchor|without_sliding|without_both) ;;
  *) echo "Invalid --executor-ablation: $EXECUTOR_ABLATION" >&2; exit 2 ;;
esac

ASSET_ROOT="$(realpath "$ASSET_ROOT")"
CHECKPOINT="$(realpath "$CHECKPOINT")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
CONFIG="$ASSET_ROOT/configs/deploy_policy.yml"
STATS="$ASSET_ROOT/norm_stats/norm_stats.json"
INSTRUCTIONS="$ASSET_ROOT/task_instructions.json"
for path in "$PYTHON" "$RMBENCH_ROOT" "$QWEN_MODEL" "$CONFIG" "$STATS" "$INSTRUCTIONS" "$CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "Missing official Mem-0 input: $path" >&2; exit 2; }
done

PROMPT="$("$PYTHON" - "$INSTRUCTIONS" "$TASK" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
try:
    print(payload["tasks"][sys.argv[2]]["global_task"])
except KeyError as exc:
    raise SystemExit(
        f"task is absent from official task_instructions.json: {sys.argv[2]}"
    ) from exc
PY
)"

mkdir -p "$OUTPUT_DIR"
COMMAND=(
  "$PYTHON" "$PROJECT_ROOT/scripts/run_mem0_eval.py"
  --rmbench-root "$RMBENCH_ROOT"
  --config "$CONFIG"
  --num-episodes "$NUM_EPISODES"
  --simulator-seed-start "$SEED_START"
  --runtime-root "$OUTPUT_DIR/runtime"
  --mem0-executor-ablation "$EXECUTOR_ABLATION"
)
if [[ "$DRY_RUN" == 1 ]]; then
  COMMAND+=(--dry-run)
fi
COMMAND+=(
  --overrides
  --task_name "$TASK"
  --execution_ckpt "$CHECKPOINT"
  --state_stats_path "$STATS"
  --ckpt_setting "m1mix_reproduced_${EXECUTOR_ABLATION}"
  --global_task "$PROMPT"
  --action_horizon 30
  --execution_module.qwen_vl.model_path "$QWEN_MODEL"
)

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PATH="$(dirname "$PYTHON"):$PATH"
command -v ninja >/dev/null || {
  echo "Official Mem-0 cuRobo evaluation requires ninja on PATH" >&2
  exit 2
}
export PYTHONPATH="$PROJECT_ROOT:$HARNESS_ROOT:$RMBENCH_ROOT:$RMBENCH_ROOT/policy:$RMBENCH_ROOT/envs/curobo/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PROJECT_ROOT/.cache/mem0_env/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export RMBENCH_EVAL_OUTPUT_ROOT="$OUTPUT_DIR/official_results"
"${COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/eval.log"
