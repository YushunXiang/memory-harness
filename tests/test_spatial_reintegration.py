from __future__ import annotations

import numpy as np
import pytest

from memory_harness.spatial_reintegration import KeyframeAwareObservationPruner
from memory_harness.spatial_reintegration import ObservationArchiveConfig
from memory_harness.spatial_reintegration import ObservationArchiveEntry
from memory_harness.spatial_reintegration import PoseGraphReintegrationConfig
from memory_harness.spatial_reintegration import PoseGraphReintegrationPlanner
from memory_harness.spatial_reintegration import PoseStampedObservation
from memory_harness.spatial_reintegration import ReintegrationMode
from memory_harness.spatial_reintegration import SpatialLifecycleProgram


def _pose(*, x: float = 0.0, yaw_degrees: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_degrees)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.asarray(
        (
            (cosine, -sine, 0.0, x),
            (sine, cosine, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _observations(count: int) -> tuple[PoseStampedObservation, ...]:
    return tuple(
        PoseStampedObservation(str(index), _pose()) for index in range(count)
    )


def _config(**overrides: object) -> PoseGraphReintegrationConfig:
    values: dict[str, object] = {
        "translation_threshold_m": 0.1,
        "rotation_threshold_degrees": 5.0,
        "regional_window": 6,
        "realtime_window": 2,
        "global_affected_fraction": 2.0 / 3.0,
        "regional_affected_fraction": 1.0 / 3.0,
        "enable_global": False,
        "enable_regional": True,
        "enable_local": True,
    }
    values.update(overrides)
    return PoseGraphReintegrationConfig(**values)  # type: ignore[arg-type]


def test_pose_graph_planner_selects_local_and_emits_revision_invalidation() -> None:
    observations = _observations(8)
    optimized = {item.observation_id: _pose() for item in observations}
    optimized["7"] = _pose(x=0.11)

    plan = PoseGraphReintegrationPlanner(_config()).plan(observations, optimized)

    assert plan.mode is ReintegrationMode.LOCAL
    assert plan.affected_observation_ids == ("7",)
    assert plan.reintegrate_observation_ids == ("6", "7")
    assert plan.invalidate_revision_dependents
    assert plan.to_json()["mode"] == "local"


def test_pose_graph_planner_selects_regional_before_local() -> None:
    observations = _observations(8)
    optimized = {item.observation_id: _pose() for item in observations}
    for observation_id in ("2", "3", "6", "7"):
        optimized[observation_id] = _pose(yaw_degrees=6.0)

    plan = PoseGraphReintegrationPlanner(_config()).plan(observations, optimized)

    assert plan.mode is ReintegrationMode.REGIONAL
    assert plan.reintegrate_observation_ids == ("2", "3", "4", "5", "6", "7")


def test_pose_graph_planner_global_mode_is_explicitly_configurable() -> None:
    observations = _observations(8)
    optimized = {
        item.observation_id: _pose(x=0.2 if int(item.observation_id) < 6 else 0.0)
        for item in observations
    }

    disabled = PoseGraphReintegrationPlanner(_config()).plan(
        observations, optimized
    )
    enabled = PoseGraphReintegrationPlanner(_config(enable_global=True)).plan(
        observations, optimized
    )

    assert disabled.mode is ReintegrationMode.NONE
    assert disabled.affected_observation_ids == ("0", "1", "2", "3", "4", "5")
    assert not disabled.invalidate_revision_dependents
    assert enabled.mode is ReintegrationMode.GLOBAL
    assert enabled.reintegrate_observation_ids == tuple(str(i) for i in range(8))


def test_pose_graph_planner_ignores_below_threshold_pose_changes() -> None:
    observations = _observations(3)
    optimized = {
        "0": _pose(),
        "1": _pose(x=0.1),
        "2": _pose(yaw_degrees=5.0),
    }

    plan = PoseGraphReintegrationPlanner(_config()).plan(observations, optimized)

    assert plan.mode is ReintegrationMode.NONE
    assert plan.affected_observation_ids == ()
    assert not plan.reintegrate_observation_ids


def test_pose_graph_planner_rejects_invalid_or_duplicate_inputs() -> None:
    planner = PoseGraphReintegrationPlanner(_config())
    duplicated = (
        PoseStampedObservation("same", _pose()),
        PoseStampedObservation("same", _pose()),
    )
    with pytest.raises(ValueError, match="duplicate"):
        planner.plan(duplicated, {"same": _pose()})
    with pytest.raises(ValueError, match="homogeneous"):
        PoseStampedObservation("bad", np.eye(4) * 2)


def test_keyframe_aware_pruner_evicts_old_non_keyframes_first() -> None:
    observations = tuple(
        ObservationArchiveEntry(str(index), index in {0, 2, 4, 6})
        for index in range(8)
    )
    pruner = KeyframeAwareObservationPruner(
        ObservationArchiveConfig(
            capacity=5,
            maximum_keyframes=3,
            recent_feature_window=2,
        )
    )

    plan = pruner.plan(observations)

    assert plan.evict_observation_ids == ("0", "1", "3")
    assert plan.retained_observation_count == 5
    assert plan.retained_keyframe_count == 3
    assert plan.retain_feature_observation_ids == ("2", "4", "6", "7")
    assert plan.drop_feature_observation_ids == ("5",)


def test_keyframe_aware_pruner_validates_capacity_contract() -> None:
    with pytest.raises(ValueError, match="maximum_keyframes"):
        ObservationArchiveConfig(
            capacity=4,
            maximum_keyframes=5,
            recent_feature_window=1,
        )
    with pytest.raises(ValueError, match="recent_feature_window"):
        ObservationArchiveConfig(
            capacity=4,
            maximum_keyframes=2,
            recent_feature_window=5,
        )


def test_spatial_lifecycle_program_composes_revision_and_archive_plans() -> None:
    pose_observations = _observations(8)
    archive_observations = tuple(
        ObservationArchiveEntry(str(index), index % 2 == 0) for index in range(8)
    )
    optimized = {item.observation_id: _pose() for item in pose_observations}
    optimized["7"] = _pose(x=0.2)
    program = SpatialLifecycleProgram(
        reintegration_planner=PoseGraphReintegrationPlanner(_config()),
        observation_pruner=KeyframeAwareObservationPruner(
            ObservationArchiveConfig(
                capacity=6,
                maximum_keyframes=4,
                recent_feature_window=2,
            )
        ),
    )

    decision = program.plan(
        pose_observations=pose_observations,
        optimized_camera_poses=optimized,
        archive_observations=archive_observations,
    )

    assert decision.reintegration.mode is ReintegrationMode.LOCAL
    assert decision.archive.evict_observation_ids == ("1", "3")
    assert decision.to_json()["reintegration"]["mode"] == "local"  # type: ignore[index]


def test_spatial_lifecycle_program_fails_closed_on_capabilities_and_id_mismatch() -> None:
    with pytest.raises(ValueError, match="calibrated_rgbd_observation_archive"):
        SpatialLifecycleProgram.preflight({"optimized_pose_graph"})
    SpatialLifecycleProgram.preflight(
        {
            "calibrated_rgbd_observation_archive",
            "optimized_pose_graph",
            "spatial_evidence_reintegrator",
        }
    )

    program = SpatialLifecycleProgram(
        reintegration_planner=PoseGraphReintegrationPlanner(_config()),
        observation_pruner=KeyframeAwareObservationPruner(
            ObservationArchiveConfig(
                capacity=2,
                maximum_keyframes=1,
                recent_feature_window=1,
            )
        ),
    )
    with pytest.raises(ValueError, match="identical chronological IDs"):
        program.plan(
            pose_observations=_observations(2),
            optimized_camera_poses={"0": _pose(), "1": _pose()},
            archive_observations=(
                ObservationArchiveEntry("0", True),
                ObservationArchiveEntry("different", False),
            ),
        )
