from __future__ import annotations

import json
import subprocess
import sys

from memory_harness.catalog import SCHEMA_VERSION, component_catalog


def test_catalog_exposes_registered_components_and_options() -> None:
    catalog = component_catalog()

    assert catalog["schema_version"] == SCHEMA_VERSION
    assert catalog["program_contract"] == {
        "path_components": [
            "encoder",
            "writer",
            "store",
            "retriever",
            "lifecycle",
        ],
        "program_components": ["controller", "utilizer"],
    }
    assert set(catalog["components"]["writer"]) == {
        "after_first_step",
        "always",
        "causal_kinematic_peak",
        "first",
        "novelty",
        "phase_change",
    }
    novelty = catalog["components"]["writer"]["novelty"]["options"]
    assert novelty == {
        "max_steps_without_write": {
            "annotation": "int | None",
            "default": None,
            "required": False,
        },
        "min_cosine_distance": {"annotation": "float", "required": True},
    }
    tiered = catalog["components"]["store"]["tiered_chunk_mean"]["options"]
    assert set(tiered) == {
        "long_capacity",
        "migration_chunk_size",
        "short_capacity",
    }
    assert all(option["required"] for option in tiered.values())
    uniform = catalog["components"]["retriever"]["uniform_global"]
    assert "complete causal history" in uniform["description"]
    assert uniform["options"] == {
        "exclude_recent_items": {
            "annotation": "int",
            "default": 0,
            "required": False,
        },
        "max_items": {"annotation": "int", "required": True},
    }
    completed_phase = catalog["components"]["retriever"]["completed_phase_mean"]
    assert "completed contiguous phase" in completed_phase["description"]
    assert completed_phase["options"] == {
        "max_items": {"annotation": "int", "required": True}
    }
    assert json.loads(json.dumps(catalog)) == catalog


def test_catalog_cli_writes_machine_readable_inventory(tmp_path) -> None:
    output = tmp_path / "catalog.json"
    subprocess.run(
        [sys.executable, "-m", "memory_harness.catalog", "--output", str(output)],
        check=True,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == component_catalog()
