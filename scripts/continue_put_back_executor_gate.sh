#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/../openpi-libero/.venv/bin/python}"
DECISION="${1:?Usage: continue_put_back_executor_gate.sh <decision> <full-checkpoint> <empty-checkpoint> <native-checkpoint>}"
FULL_CHECKPOINT="${2:?Missing full checkpoint}"
EMPTY_CHECKPOINT="${3:?Missing empty checkpoint}"
NATIVE_CHECKPOINT="${4:?Missing native checkpoint}"
DRY_RUN="${DRY_RUN:-0}"
GPU_ID="${GPU_ID:-1}"
CONTEXT_ROOT="$PROJECT_ROOT/rmbench_runs/emac_put_back_block_v1"
CHECKPOINT_ROOT="${CHECKPOINT_BASE_DIR:-/tmp/memory-harness-checkpoints}"
RUN_ROOT="${RUN_ROOT:-/tmp/rmbench_runs}"
DATE_TAG="${DATE_TAG:-20260815}"

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
for path in "$DECISION" "$FULL_CHECKPOINT" "$EMPTY_CHECKPOINT" "$NATIVE_CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "Missing continuation input: $path" >&2; exit 2; }
done

# A budget extension is a continuation of the trained comparison, not a new
# candidate-suite trial. Preserve the exact runtime/config contract frozen in
# the full/empty parent checkpoints and reject silently mixed snapshots before
# spending GPU time.
"$PYTHON" - "$HARNESS_ROOT" "$FULL_CHECKPOINT" "$EMPTY_CHECKPOINT" <<'PY'
import sys
from pathlib import Path

harness_root, full, empty = map(Path, sys.argv[1:])
sys.path.insert(0, str(harness_root))
from memory_harness.config_snapshot import validate_config_snapshot
from memory_harness.runtime_snapshot import validate_runtime_snapshot

identities = []
for label, checkpoint in (("full", full), ("empty", empty)):
    runtime = validate_runtime_snapshot(checkpoint / "runtime")
    config = validate_config_snapshot(checkpoint / "experiment_configs")
    identities.append((runtime["source_sha256"], config["source_sha256"]))
if identities[0] != identities[1]:
    raise SystemExit(
        "full and empty parent checkpoints use different runtime/config snapshots: "
        f"full={identities[0]}, empty={identities[1]}"
    )
PY

PARENT_RUNTIME="$FULL_CHECKPOINT/runtime"
PARENT_CONFIG="$FULL_CHECKPOINT/experiment_configs"

ACTION="$(
  "$PYTHON" - "$DECISION" <<'PY'
import json, sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("schema_version") != "memory_harness.executor_readiness/v2":
    raise SystemExit("unsupported executor-readiness decision")
action = value.get("decision", {}).get("next_action")
allowed = {
    "collect_fixed_ablation_to_20",
    "retrain_full_memory_at_higher_budget_before_gate20",
    "increase_training_budget_before_more_rollouts",
}
if action not in allowed:
    raise SystemExit(f"unknown executor-readiness action: {action!r}")
print(action)
PY
)"

run() {
  printf 'RUN'
  printf ' %q' "$@"
  printf '\n'
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

run_memory_stage() {
  local program="$1"
  local parent_checkpoint="$2"
  local exp_name="$3"
  local manifest pairing bank migration
  if [[ "$program" == anchor_sliding ]]; then
    manifest="$CONTEXT_ROOT/anchor_sliding_context_manifest.json"
    pairing="$CONTEXT_ROOT/anchor_sliding_context_audit.json"
    bank="$CONTEXT_ROOT/anchor_sliding_context_bank.npz"
    migration="$CONTEXT_ROOT/anchor_sliding_quota_program_migration_audit.json"
  else
    manifest="$CONTEXT_ROOT/none_context_manifest.json"
    pairing="$CONTEXT_ROOT/none_context_audit.json"
    bank="$CONTEXT_ROOT/none_context_bank.npz"
    migration="$CONTEXT_ROOT/none_quota_program_migration_audit.json"
  fi
  run env \
    PYTHON="$PYTHON" GPU_ID="$GPU_ID" OPTIMIZER_UPDATES=1800 \
    SAVE_EVERY_UPDATES=300 CHECKPOINT_BASE_DIR="$CHECKPOINT_ROOT" \
    EXP_NAME="$exp_name" TRAINING_LOG="/tmp/${exp_name}.log" \
    WEIGHT_PARAMS="$parent_checkpoint/params" \
    MANIFEST="$manifest" PAIRING_AUDIT="$pairing" CONTEXT_BANK="$bank" \
    PROGRAM_MIGRATION_AUDIT="$migration" \
    RUNTIME_SNAPSHOT_SOURCE="$PARENT_RUNTIME" \
    CONFIG_SNAPSHOT_SOURCE="$PARENT_CONFIG" \
    bash "$HARNESS_ROOT/scripts/run_pi05_memory_train.sh" put_back_block
}

run_native_stage() {
  local parent_checkpoint="$1"
  local exp_name="$2"
  run env \
    PYTHON="$PYTHON" GPU_ID="$GPU_ID" OPTIMIZER_UPDATES=1800 \
    SAVE_EVERY_UPDATES=300 CHECKPOINT_BASE_DIR="$CHECKPOINT_ROOT" \
    EXP_NAME="$exp_name" TRAINING_LOG="/tmp/${exp_name}.log" \
    WEIGHT_PARAMS="$parent_checkpoint/params" \
    bash "$HARNESS_ROOT/scripts/run_pi05_baseline_train.sh" put_back_block
}

reuse_completed_stage() {
  local checkpoint="$1"
  local parent_checkpoint="$2"
  local expected_program="$3"
  local manifest="$checkpoint/memory_training_manifest.json"
  [[ -s "$manifest" ]] || return 1
  "$PYTHON" - "$checkpoint" "$parent_checkpoint" "$expected_program" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
parent = Path(sys.argv[2]).resolve()
expected_program = sys.argv[3]
manifest_path = checkpoint / "memory_training_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
metadata = checkpoint / "_CHECKPOINT_METADATA"
expected = {
    "schema_version": "memory_harness.training/v1",
    "checkpoint_step": 50399,
    "checkpoint_commit_verified": True,
    "optimizer_updates": 1800,
    "effective_batch": 56,
    "program": expected_program,
    "initial_weight_params": str(parent / "params"),
    "parent_checkpoint": str(parent),
}
mismatches = {
    key: {"expected": value, "actual": manifest.get(key)}
    for key, value in expected.items()
    if manifest.get(key) != value
}
if not metadata.is_file() or not (checkpoint / "params").is_dir():
    mismatches["checkpoint_files"] = "missing committed metadata or params"
elif manifest.get("checkpoint_metadata_sha256") != hashlib.sha256(
    metadata.read_bytes()
).hexdigest():
    mismatches["checkpoint_metadata_sha256"] = "does not match committed metadata"
if mismatches:
    raise SystemExit(f"completed-stage validation failed: {mismatches}")
PY
  echo "REUSE_COMPLETED_STAGE=$checkpoint"
}

run_fixed() {
  local checkpoint="$1"
  local architecture="$2"
  local run_id="$3"
  run env \
    PYTHON="$PYTHON" CHECKPOINT_DIR="$checkpoint" NUM_EPISODES=3 \
    SEED=100000 POLICY_SEED_BASE=120000 GPU_ID="$GPU_ID" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" RUN_ID="$run_id" OUT_DIR="$RUN_ROOT/$run_id" \
    bash "$HARNESS_ROOT/scripts/run_fixed_pi05_rmbench.sh" \
    put_back_block "$architecture"
}

run_clean() {
  local checkpoint="$1"
  local run_id="$2"
  run env \
    PYTHON="$PYTHON" CHECKPOINT_DIR="$checkpoint" NUM_EPISODES=3 \
    SEED=100000 POLICY_SEED_BASE=120000 GPU_ID="$GPU_ID" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" RUN_ID="$run_id" OUT_DIR="$RUN_ROOT/$run_id" \
    bash "$HARNESS_ROOT/scripts/run_clean_pi05_rmbench.sh" put_back_block
}

run_gate17() {
  local checkpoint="$1"
  local budget="$2"
  local architecture="$3"
  local run_id="emac_put_back_block_${architecture}_${budget}_gate17_${DATE_TAG}"
  run env \
    PYTHON="$PYTHON" CHECKPOINT_DIR="$checkpoint" NUM_EPISODES=17 \
    SEED=100003 POLICY_SEED_BASE=120003 GPU_ID="$GPU_ID" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" RUN_ID="$run_id" OUT_DIR="$RUN_ROOT/$run_id" \
    bash "$HARNESS_ROOT/scripts/run_fixed_pi05_rmbench.sh" \
    put_back_block "$architecture"
}

collect_fixed_gate20() {
  local checkpoint="$1"
  local gate3_budget="$2"
  local output_budget="$3"
  for architecture in none anchor sliding anchor_sliding; do
    run_gate17 "$checkpoint" "$output_budget" "$architecture"
  done
  local utilities=()
  for candidate in anchor sliding anchor_sliding; do
    comparison="/tmp/put_back_block_none_vs_${candidate}_${output_budget}_gate20.json"
    utility="/tmp/put_back_block_none_vs_${candidate}_${output_budget}_gate20.utility.json"
    run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" \
      -m memory_harness.compare_fixed_runs \
      --reference-run "$RUN_ROOT/emac_put_back_block_none_${gate3_budget}_gate3_${DATE_TAG}" \
      --candidate-run "$RUN_ROOT/emac_put_back_block_${candidate}_${gate3_budget}_gate3_${DATE_TAG}" \
      --reference-run "$RUN_ROOT/emac_put_back_block_none_${output_budget}_gate17_${DATE_TAG}" \
      --candidate-run "$RUN_ROOT/emac_put_back_block_${candidate}_${output_budget}_gate17_${DATE_TAG}" \
      --output "$comparison"
    run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" \
      -m memory_harness.utility_gate --comparison "$comparison" \
      --evidence-kind fixed_ablation --output "$utility"
    utilities+=("$utility")
  done
  run env \
    PYTHON="$PYTHON" GPU_ID="$GPU_ID" RUN_ROOT="$RUN_ROOT" \
    DATE_TAG="$DATE_TAG" \
    bash "$HARNESS_ROOT/scripts/continue_put_back_candidate_screen.sh" \
    "$checkpoint" "${utilities[@]}"
}

if [[ "$ACTION" == collect_fixed_ablation_to_20 ]]; then
  collect_fixed_gate20 "$FULL_CHECKPOINT" u1000 u1200
  exit 0
fi

if [[ "$ACTION" == increase_training_budget_before_more_rollouts ]]; then
  NATIVE_EXP="emac_put_back_block_native_none_plus1800_to_u3000_b2a28_${DATE_TAG}"
  NATIVE_U3000="$CHECKPOINT_ROOT/pi05_aloha_pen_uncap_mem0_control/$NATIVE_EXP/50399"
  if [[ -s "$NATIVE_U3000/memory_training_manifest.json" ]]; then
    reuse_completed_stage "$NATIVE_U3000" "$NATIVE_CHECKPOINT" native_none
  else
    run_native_stage "$NATIVE_CHECKPOINT" "$NATIVE_EXP"
  fi
  [[ "$DRY_RUN" == 1 || -s "$NATIVE_U3000/memory_training_manifest.json" ]] || {
    echo "Missing extended native checkpoint: $NATIVE_U3000" >&2
    exit 2
  }
fi

FULL_EXP="emac_put_back_block_anchor_sliding_plus1800_to_u3000_b2a28_${DATE_TAG}"
FULL_U3000="$CHECKPOINT_ROOT/pi05_aloha_pen_uncap_mem0/$FULL_EXP/50399"
if [[ -s "$FULL_U3000/memory_training_manifest.json" ]]; then
  reuse_completed_stage "$FULL_U3000" "$FULL_CHECKPOINT" anchor_sliding
else
  run_memory_stage anchor_sliding "$FULL_CHECKPOINT" "$FULL_EXP"
fi
[[ "$DRY_RUN" == 1 || -s "$FULL_U3000/memory_training_manifest.json" ]] || {
  echo "Missing extended full-memory checkpoint: $FULL_U3000" >&2
  exit 2
}
FULL_RUN="emac_put_back_block_anchor_sliding_u3000_gate3_${DATE_TAG}"
run_fixed "$FULL_U3000" anchor_sliding "$FULL_RUN"
FULL_SIGNAL="/tmp/put_back_block_full_u3000_executor_signal_gate3.json"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.assess_run_signal \
  --run "$RUN_ROOT/$FULL_RUN" --output "$FULL_SIGNAL"

if [[ "$ACTION" == retrain_full_memory_at_higher_budget_before_gate20 ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    full_has_signal=1
  else
    full_has_signal="$(
      "$PYTHON" - "$FULL_SIGNAL" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["observable_executor_signal"]))
PY
    )"
  fi
  if [[ "$full_has_signal" == 0 ]]; then
    echo "Full-memory u3000 remains at executor floor; controls stay at u1200."
    exit 0
  fi
  NATIVE_EXP="emac_put_back_block_native_none_plus1800_to_u3000_b2a28_${DATE_TAG}"
  NATIVE_U3000="$CHECKPOINT_ROOT/pi05_aloha_pen_uncap_mem0_control/$NATIVE_EXP/50399"
  run_native_stage "$NATIVE_CHECKPOINT" "$NATIVE_EXP"
  [[ "$DRY_RUN" == 1 || -s "$NATIVE_U3000/memory_training_manifest.json" ]] || {
    echo "Missing extended native checkpoint: $NATIVE_U3000" >&2
    exit 2
  }
fi

EMPTY_EXP="emac_put_back_block_none_plus1800_to_u3000_b2a28_${DATE_TAG}"
EMPTY_U3000="$CHECKPOINT_ROOT/pi05_aloha_pen_uncap_mem0/$EMPTY_EXP/50399"

# The all-floor branch is pre-registered as native→full→empty.
if [[ -s "$EMPTY_U3000/memory_training_manifest.json" ]]; then
  reuse_completed_stage "$EMPTY_U3000" "$EMPTY_CHECKPOINT" none
else
  run_memory_stage none "$EMPTY_CHECKPOINT" "$EMPTY_EXP"
fi
[[ "$DRY_RUN" == 1 || -s "$EMPTY_U3000/memory_training_manifest.json" ]] || {
  echo "Missing extended empty checkpoint: $EMPTY_U3000" >&2
  exit 2
}

EMPTY_RUN="emac_put_back_block_none_checkpoint_u3000_gate3_${DATE_TAG}"
NATIVE_RUN="emac_put_back_block_native_none_u3000_gate3_${DATE_TAG}"
run_fixed "$EMPTY_U3000" none "$EMPTY_RUN"
run_clean "$NATIVE_U3000" "$NATIVE_RUN"

EMPTY_FULL="/tmp/put_back_block_empty_arch_vs_full_budget_matched_u3000_gate3.json"
NATIVE_FULL="/tmp/put_back_block_native_vs_full_budget_matched_u3000_gate3.json"
NATIVE_EMPTY="/tmp/put_back_block_native_vs_empty_arch_budget_matched_u3000_gate3.json"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.compare_training_runs \
  --reference-run "$RUN_ROOT/$EMPTY_RUN" --candidate-run "$RUN_ROOT/$FULL_RUN" \
  --output "$EMPTY_FULL"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.compare_training_runs \
  --reference-run "$RUN_ROOT/$NATIVE_RUN" --candidate-run "$RUN_ROOT/$FULL_RUN" \
  --output "$NATIVE_FULL"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.compare_training_runs \
  --reference-run "$RUN_ROOT/$NATIVE_RUN" --candidate-run "$RUN_ROOT/$EMPTY_RUN" \
  --output "$NATIVE_EMPTY"
run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" -m memory_harness.decide_executor_readiness \
  --empty-vs-full "$EMPTY_FULL" --native-vs-full "$NATIVE_FULL" \
  --native-vs-empty "$NATIVE_EMPTY" \
  --output /tmp/put_back_block_executor_readiness_u3000_gate3.json

if [[ "$DRY_RUN" == 1 ]]; then
  U3000_ACTION=collect_fixed_ablation_to_20
else
  U3000_ACTION="$(
    "$PYTHON" - /tmp/put_back_block_executor_readiness_u3000_gate3.json <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["decision"]["next_action"])
PY
  )"
fi

if [[ "$U3000_ACTION" == collect_fixed_ablation_to_20 ]]; then
  for architecture in none anchor sliding; do
    run_fixed \
      "$FULL_U3000" "$architecture" \
      "emac_put_back_block_${architecture}_u3000_gate3_${DATE_TAG}"
  done
  collect_fixed_gate20 "$FULL_U3000" u3000 u3000
fi
