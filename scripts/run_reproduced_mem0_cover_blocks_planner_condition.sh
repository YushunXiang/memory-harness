#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
RMBENCH_ROOT="$PROJECT_ROOT/../RMBench"
MEM0_PYTHON="$PROJECT_ROOT/.cache/mem0_env/bin/python"
QWEN_MODEL="$PROJECT_ROOT/../VLMEvalKit/models/Qwen/Qwen3-VL-2B-Instruct"
PLANNER_BASE_INDEX="$PROJECT_ROOT/../VLMEvalKit/models/Qwen/Qwen3-VL-8B-Instruct/model.safetensors.index.json"
EXECUTION_CHECKPOINT="$PROJECT_ROOT/.cache/mem0_assets/checkpoints/cover_blocks/final_step30000_inference.pt"
NORM_STATS="$PROJECT_ROOT/.cache/mem0_assets/norm_stats/cover_blocks/norm_stats.json"
PLANNING_CONFIG="$PROJECT_ROOT/configs/mem0_cover_blocks_planning.yaml"
DEPLOY_CONFIG="$RMBENCH_ROOT/policy/Mem-0/deploy_policy.yml"
GLOBAL_TASK="On the table, red, green, and blue blocks are arranged randomly along with three lids. From the current viewpoint, cover the blocks from left to right using the lids, and then uncover them again in the sequence red, green, and blue."
REFERENCE_SUMMARY="$PROJECT_ROOT/rmbench_runs/weekly_20260804_mem0_rgb_balanced_v3_period_fixed_paper_l8_10seed_v2/full/final_summary.json"
TRAINING_PAIR_MANIFEST="$PROJECT_ROOT/rmbench_runs/emac_mem0_planner_key_no_key_data_pair_20260814.json"

usage() {
  echo "Usage: $0 --condition key|no_key --num-episodes N --seed-start N --gpu-id N --output-dir DIR [--dry-run]" >&2
}

CONDITION=""
NUM_EPISODES=""
SEED_START=""
GPU_ID=""
OUTPUT_DIR=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --condition) CONDITION="${2:?Missing value for --condition}"; shift 2 ;;
    --num-episodes) NUM_EPISODES="${2:?Missing value for --num-episodes}"; shift 2 ;;
    --seed-start) SEED_START="${2:?Missing value for --seed-start}"; shift 2 ;;
    --gpu-id) GPU_ID="${2:?Missing value for --gpu-id}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?Missing value for --output-dir}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$CONDITION" in
  key)
    PLANNER_MODEL="$PROJECT_ROOT/rmbench_checkpoints/mem0_planner/cover_blocks_qwen3vl8b_merged_rgb_balanced_v3_checkpoint75"
    PLANNER_ADAPTER="$PROJECT_ROOT/rmbench_checkpoints/mem0_planner/cover_blocks_qwen3vl8b_lora_rgb_balanced_v3/checkpoint-75/adapter_model.safetensors"
    ;;
  no_key)
    PLANNER_MODEL="$PROJECT_ROOT/rmbench_checkpoints/mem0_planner/cover_blocks_no_key_qwen3vl8b_merged"
    PLANNER_ADAPTER="$PROJECT_ROOT/rmbench_checkpoints/mem0_planner/cover_blocks_no_key_qwen3vl8b_lora/checkpoint-75/adapter_model.safetensors"
    ;;
  *) usage; echo "--condition must be key or no_key" >&2; exit 2 ;;
esac
[[ "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
[[ "$SEED_START" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { usage; exit 2; }
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

for path in "$MEM0_PYTHON" "$QWEN_MODEL" "$PLANNER_BASE_INDEX" "$EXECUTION_CHECKPOINT" "$NORM_STATS" "$PLANNING_CONFIG" "$DEPLOY_CONFIG" "$PLANNER_MODEL/model.safetensors.index.json" "$PLANNER_ADAPTER" "$TRAINING_PAIR_MANIFEST"; do
  [[ -e "$path" ]] || { echo "Missing required asset: $path" >&2; exit 2; }
done

SERVER_COMMAND=(
  "$MEM0_PYTHON" "$PROJECT_ROOT/scripts/serve_mem0_planner.py"
  --model "$PLANNER_MODEL"
  --served-model-name mem0-cover-blocks-planner
  --gpu-id "$GPU_ID"
  --port 8123
  --gpu-memory-utilization 0.48
  --max-model-len 8192
  --max-images 16
)
EVAL_COMMAND=(
  "$MEM0_PYTHON" "$PROJECT_ROOT/scripts/run_mem0_eval.py"
  --rmbench-root "$RMBENCH_ROOT"
  --config "$DEPLOY_CONFIG"
  --num-episodes "$NUM_EPISODES"
  --simulator-seed-start "$SEED_START"
  --runtime-root "$OUTPUT_DIR/runtime"
  --paired-layout-protocol
  --overrides
  --policy_name mem0_episodic_policy
  --task_name cover_blocks
  --task_config demo_clean
  --instruction_type unseen
  --seed 0
  --ckpt_setting "reproduced_planner_${CONDITION}"
  --device cuda:0
  --execution_ckpt "$EXECUTION_CHECKPOINT"
  --state_stats_path "$NORM_STATS"
  --planning_module_config_path "$PLANNING_CONFIG"
  --vllm_url http://127.0.0.1:8123/v1
  --global_task "$GLOBAL_TASK"
  --action_horizon 30
  --threshold 8
  --num_subtasks 6
  --execution_module.qwen_vl.model_path "$QWEN_MODEL"
  --episodic_memory_condition none
  --planner_memory_condition "$CONDITION"
  --planner_temperature 0.0
  --planner_seed 7
  --planner_max_tokens 128
  --policy_seed_base 120000
  --mem0_audit_log "$OUTPUT_DIR/mem0_audit.jsonl"
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf '%q ' "${SERVER_COMMAND[@]}"; printf '\n'
  printf '%q ' "${EVAL_COMMAND[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
export PATH="$(dirname "$MEM0_PYTHON"):$PATH"
export PYTHONPATH="$PROJECT_ROOT:$HARNESS_ROOT:$RMBENCH_ROOT:$RMBENCH_ROOT/policy:$RMBENCH_ROOT/envs/curobo/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
command -v ninja >/dev/null 2>&1 || {
  echo "ninja is required by cuRobo but is not available on PATH" >&2
  exit 2
}
if "$MEM0_PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8123/v1/models', timeout=2)" >/dev/null 2>&1; then
  echo "Planner port 8123 is already occupied; refusing to use an unaudited server" >&2
  exit 2
fi
setsid "${SERVER_COMMAND[@]}" >"$OUTPUT_DIR/planner_server.log" 2>&1 &
PLANNER_PID=$!
cleanup() {
  kill -TERM -- "-$PLANNER_PID" 2>/dev/null || true
  wait "$PLANNER_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in {1..120}; do
  if "$MEM0_PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8123/v1/models', timeout=2)" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$PLANNER_PID" 2>/dev/null; then
    echo "Planner server exited before becoming ready" >&2
    tail -n 80 "$OUTPUT_DIR/planner_server.log" >&2
    exit 1
  fi
  sleep 5
done
[[ "$ready" == 1 ]] || { echo "Planner server did not become ready" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="$PROJECT_ROOT/.cache/mem0_env/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export RMBENCH_EVAL_OUTPUT_ROOT="$OUTPUT_DIR/official_results"
"${EVAL_COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/eval.log"

SUMMARY_COMMAND=(
  "$MEM0_PYTHON" -m memory_harness.summarize_reproduced_mem0_planner_condition
  --eval-log "$OUTPUT_DIR/eval.log"
  --audit-log "$OUTPUT_DIR/mem0_audit.jsonl"
  --condition "$CONDITION"
  --checkpoint "$EXECUTION_CHECKPOINT"
  --planner-model "$PLANNER_MODEL"
  --planner-adapter "$PLANNER_ADAPTER"
  --base-model-index "$PLANNER_BASE_INDEX"
  --training-pair-manifest "$TRAINING_PAIR_MANIFEST"
  --seed-start "$SEED_START"
  --policy-seed-base 120000
  --output "$OUTPUT_DIR/summary.json"
)
if [[ "$CONDITION" == "no_key" && "$NUM_EPISODES" == 10 && "$SEED_START" == 100000 ]]; then
  SUMMARY_COMMAND+=(--reference-summary "$REFERENCE_SUMMARY")
fi
"${SUMMARY_COMMAND[@]}"
