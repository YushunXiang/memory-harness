from __future__ import annotations

import importlib.util
import pathlib

import numpy as np


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts/evaluate_mem0_offline.py"
SPEC = importlib.util.spec_from_file_location("evaluate_mem0_offline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cluster_bootstrap_resamples_episode_clusters() -> None:
    values = np.asarray([-2.0, -2.0, -1.0, -1.0])
    clusters = np.asarray([0, 0, 1, 1])
    interval = MODULE._cluster_bootstrap_ci(
        values,
        clusters,
        samples=500,
        seed=7,
    )
    assert interval[0] <= -2.0
    assert interval[1] >= -1.0


def test_single_stratified_sample_uses_episode_midpoint() -> None:
    assert MODULE._interior_indices(100, 200, 1).tolist() == [150]
    assert MODULE._interior_indices(100, 200, 3).tolist() == [125, 150, 175]
