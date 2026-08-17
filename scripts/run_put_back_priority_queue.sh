#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

EMPTY_RUN=/tmp/rmbench_runs/emac_put_back_block_none_checkpoint_u1200_gate3_20260815
while [[ ! -s "$EMPTY_RUN/emac_manifest.json" ]]; do
  sleep 60
done

M1_ROOT=/tmp/mem0-m1mix-put-back-official-gate10-20260815
M1_CHECKPOINT=/tmp/mem0-m1mix-official/checkpoint/m1_mix_final_step50000.inference.pt
if [[ -s "$M1_ROOT/summary.json" ]]; then
  PYTHONPATH=memory-harness ../openpi-libero/.venv/bin/python \
    -m memory_harness.summarize_official_mem0_ablation \
    --run-dir "$M1_ROOT" \
    --checkpoint "$M1_CHECKPOINT" \
    --task put_back_block \
    --seed-start 100000 \
    --output "$M1_ROOT/summary.json" >/dev/null
  echo OFFICIAL_MEM0_M1MIX_INTERVENTION_REUSED_VALIDATED
else
  bash memory-harness/scripts/run_official_mem0_m1mix_ablation.sh \
    --asset-root /tmp/mem0-m1mix-official \
    --checkpoint "$M1_CHECKPOINT" \
    --task put_back_block \
    --num-episodes 10 \
    --seed-start 100000 \
    --gpu-id 1 \
    --output-dir "$M1_ROOT"
  echo OFFICIAL_MEM0_M1MIX_INTERVENTION_COMPLETE
fi

bash memory-harness/scripts/run_reproduced_mem0_cover_blocks_planner_condition.sh \
  --condition no_key \
  --num-episodes 10 \
  --seed-start 100000 \
  --gpu-id 1 \
  --output-dir /tmp/mem0-cover-blocks-no-key-gate10-20260815
echo REPRODUCED_MEM0_COVER_BLOCKS_NO_KEY_COMPLETE

export PYTHON=../openpi-libero/.venv/bin/python
export GPU_ID=1
export OPTIMIZER_UPDATES=1200
export SAVE_EVERY_UPDATES=200
export CHECKPOINT_BASE_DIR=/tmp/memory-harness-checkpoints
export EXP_NAME=emac_put_back_block_native_none_u1200_b2a28_20260815
export TRAINING_LOG=/tmp/emac_put_back_block_native_none_u1200_b2a28_20260815.log
bash memory-harness/scripts/run_pi05_baseline_train.sh put_back_block

NATIVE_CHECKPOINT=/tmp/memory-harness-checkpoints/pi05_aloha_pen_uncap_mem0_control/emac_put_back_block_native_none_u1200_b2a28_20260815/33599
[[ -f "$NATIVE_CHECKPOINT/_CHECKPOINT_METADATA" ]] || {
  echo "missing budget-matched native checkpoint" >&2
  exit 2
}
export CHECKPOINT_DIR="$NATIVE_CHECKPOINT"
export NUM_EPISODES=3
export SEED=100000
export POLICY_SEED_BASE=120000
export CUDA_VISIBLE_DEVICES=1
export RUN_ID=emac_put_back_block_native_none_u1200_gate3_20260815
export OUT_DIR=/tmp/rmbench_runs/$RUN_ID
bash memory-harness/scripts/run_clean_pi05_rmbench.sh put_back_block

FULL_RUN=/tmp/rmbench_runs/emac_put_back_block_anchor_sliding_u1000_gate3_20260815
PYTHONPATH=memory-harness "$PYTHON" -m memory_harness.compare_training_runs \
  --reference-run "$EMPTY_RUN" \
  --candidate-run "$FULL_RUN" \
  --output /tmp/put_back_block_empty_arch_vs_full_budget_matched_gate3.json
PYTHONPATH=memory-harness "$PYTHON" -m memory_harness.compare_training_runs \
  --reference-run "$OUT_DIR" \
  --candidate-run "$FULL_RUN" \
  --output /tmp/put_back_block_native_vs_full_budget_matched_gate3.json
PYTHONPATH=memory-harness "$PYTHON" -m memory_harness.compare_training_runs \
  --reference-run "$OUT_DIR" \
  --candidate-run "$EMPTY_RUN" \
  --output /tmp/put_back_block_native_vs_empty_arch_budget_matched_gate3.json
PYTHONPATH=memory-harness "$PYTHON" -m memory_harness.decide_executor_readiness \
  --empty-vs-full /tmp/put_back_block_empty_arch_vs_full_budget_matched_gate3.json \
  --native-vs-full /tmp/put_back_block_native_vs_full_budget_matched_gate3.json \
  --native-vs-empty /tmp/put_back_block_native_vs_empty_arch_budget_matched_gate3.json \
  --output /tmp/put_back_block_executor_readiness_u1200_gate3.json
echo NATIVE_BASELINE_COMPLETE
