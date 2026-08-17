from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "memory-harness" / "scripts" / "run_cover_blocks_corrected_executor_gate.sh"


def test_corrected_cover_blocks_gate_dry_run_uses_subtask_lineage() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout
    assert "pi05_aloha_pen_uncap_mem0_control" in output
    assert "local/rmbench_cover_blocks_oracle_success_50plus" in output
    assert "--data.episode-ids" in output
    assert "ORACLE_SUBTASK_DIAGNOSTIC=1" in output
    assert "memory_harness.assess_run_signal" in output
    assert "continue_cover_blocks_corrected_executor_gate.sh" in output
    assert "9999" not in output
