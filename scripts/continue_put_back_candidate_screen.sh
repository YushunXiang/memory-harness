#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/../openpi-libero/.venv/bin/python}"
CHECKPOINT="${1:?Usage: continue_put_back_candidate_screen.sh <checkpoint> <fixed-gate-utility>...}"
shift
[[ "$#" -gt 0 ]] || { echo "At least one fixed-gate utility is required" >&2; exit 2; }

DRY_RUN="${DRY_RUN:-0}"
GPU_ID="${GPU_ID:-1}"
DATE_TAG="${DATE_TAG:-20260816}"
RUN_ROOT="${RUN_ROOT:-/tmp/rmbench_runs}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/put_back_candidate_screen_${DATE_TAG}}"
CANDIDATE_SUITE="${MEMORY_CANDIDATE_SUITE:-$HARNESS_ROOT/artifacts/2026-08-16-candidate-suite-v9}"
SCREEN_PLAN="${SCREEN_PLAN:-$HARNESS_ROOT/configs/screens/put_back_fixed_v9_screen3.json}"

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ -d "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -d "$CANDIDATE_SUITE" ]] || { echo "Missing candidate suite: $CANDIDATE_SUITE" >&2; exit 2; }
[[ -s "$SCREEN_PLAN" ]] || { echo "Missing fixed screen plan: $SCREEN_PLAN" >&2; exit 2; }
for utility in "$@"; do
  [[ -s "$utility" ]] || { echo "Missing fixed-gate utility: $utility" >&2; exit 2; }
done

fixed_signal="$($PYTHON - "$@" <<'PY'
import json
import sys
from pathlib import Path

positive_actions = {
    "collect_shared_episodes_to_50",
    "assemble_gate1_diagnostic_bundle",
}
positive = False
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "memory_harness.utility_decision/v2":
        raise SystemExit(f"unsupported utility schema: {path}")
    if value.get("evidence_kind") != "fixed_ablation":
        raise SystemExit(f"candidate screen requires fixed_ablation evidence: {path}")
    positive = positive or bool(value.get("candidate_utility_requirement_met"))
    positive = positive or value.get("next_action") in positive_actions
print(int(positive))
PY
)"

if [[ "$fixed_signal" == 0 ]]; then
  echo "No positive fixed-memory pilot signal; candidate screen and controller remain gated."
  exit 0
fi

mkdir -p "$RESULT_ROOT"
install -m 0644 "$SCREEN_PLAN" "$RESULT_ROOT/screen_plan.json"

readarray -t plan_rows < <(
  PYTHONPATH="$HARNESS_ROOT" "$PYTHON" - "$SCREEN_PLAN" "$CANDIDATE_SUITE/candidate_suite_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

from memory_harness.screen_plan import load_fixed_screen_plan

manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
aliases = {row["alias"] for row in manifest["architectures"]}
plan = load_fixed_screen_plan(Path(sys.argv[1]), available_aliases=aliases)
print(f"META\t{plan.name}\t{plan.task}\t{plan.num_episodes}\t{plan.seed}\t{plan.policy_seed_base}\t{plan.evidence_kind}")
for reference in sorted({row.reference for row in plan.comparisons}):
    print(f"REFERENCE\t{reference}")
for row in plan.comparisons:
    print(f"COMPARISON\t{row.reference}\t{row.candidate}")
PY
)

IFS=$'\t' read -r row_type SCREEN_NAME TASK_ID NUM_EPISODES SEED POLICY_SEED_BASE EVIDENCE_KIND <<<"${plan_rows[0]}"
[[ "$row_type" == META ]] || { echo "Invalid fixed screen plan output" >&2; exit 2; }

run() {
  printf 'RUN'
  printf ' %q' "$@"
  printf '\n'
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

run_fixed() {
  local architecture="$1"
  local run_id="emac_${TASK_ID}_${architecture}_${SCREEN_NAME}_${DATE_TAG}"
  local out_dir="$RUN_ROOT/$run_id"
  if [[ "$DRY_RUN" == 0 && -s "$out_dir/emac_manifest.json" ]]; then
    "$PYTHON" -m memory_harness.validate_fixed_run \
      --run-dir "$out_dir" \
      --architecture-config "$out_dir/configs/architectures/fixed_${architecture}.json" \
      --task-config "$out_dir/configs/task.json" \
      --audit-log "$out_dir/memory_architecture_audit.jsonl" >/dev/null
    echo "REUSE $out_dir"
    return
  fi
  [[ "$DRY_RUN" == 1 || ! -e "$out_dir" ]] || {
    echo "Incomplete candidate-screen output exists: $out_dir" >&2
    exit 2
  }
  run env \
    PYTHON="$PYTHON" CHECKPOINT_DIR="$CHECKPOINT" \
    MEMORY_CANDIDATE_SUITE="$CANDIDATE_SUITE" \
    NUM_EPISODES="$NUM_EPISODES" SEED="$SEED" POLICY_SEED_BASE="$POLICY_SEED_BASE" \
    GPU_ID="$GPU_ID" CUDA_VISIBLE_DEVICES="$GPU_ID" \
    RUN_ID="$run_id" OUT_DIR="$out_dir" \
    bash "$HARNESS_ROOT/scripts/run_fixed_pi05_rmbench.sh" \
    "$TASK_ID" "$architecture"
}

for row in "${plan_rows[@]:1}"; do
  IFS=$'\t' read -r row_type first second <<<"$row"
  [[ "$row_type" != REFERENCE ]] || run_fixed "$first"
done

for row in "${plan_rows[@]:1}"; do
  IFS=$'\t' read -r row_type reference candidate <<<"$row"
  [[ "$row_type" == COMPARISON ]] || continue
  run_fixed "$candidate"
  reference_run="$RUN_ROOT/emac_${TASK_ID}_${reference}_${SCREEN_NAME}_${DATE_TAG}"
  candidate_run="$RUN_ROOT/emac_${TASK_ID}_${candidate}_${SCREEN_NAME}_${DATE_TAG}"
  comparison="$RESULT_ROOT/${reference}_vs_${candidate}_screen3.json"
  utility="$RESULT_ROOT/${reference}_vs_${candidate}_screen3.utility.json"
  run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" \
    -m memory_harness.compare_fixed_runs \
    --reference-run "$reference_run" --candidate-run "$candidate_run" \
    --output "$comparison"
  run env PYTHONPATH="$HARNESS_ROOT" "$PYTHON" \
    -m memory_harness.utility_gate --comparison "$comparison" \
    --evidence-kind "$EVIDENCE_KIND" --output "$utility"
done

echo "Put Back fixed candidate screen complete: $RESULT_ROOT"
