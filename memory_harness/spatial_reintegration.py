from __future__ import annotations

import dataclasses
import enum
from collections.abc import Collection, Mapping, Sequence

import numpy as np


def _validated_pose(value: np.ndarray, *, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"{label} must contain only finite values")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{label} must be a homogeneous transform")
    pose = pose.copy()
    pose.setflags(write=False)
    return pose


@dataclasses.dataclass(frozen=True)
class PoseStampedObservation:
    """Observation pose that was used when its spatial evidence was integrated."""

    observation_id: str
    integrated_camera_pose: np.ndarray

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        object.__setattr__(
            self,
            "integrated_camera_pose",
            _validated_pose(
                self.integrated_camera_pose,
                label=f"integrated_camera_pose[{self.observation_id}]",
            ),
        )


@dataclasses.dataclass(frozen=True)
class PoseDelta:
    observation_id: str
    translation_m: float
    rotation_degrees: float

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        if self.translation_m < 0 or self.rotation_degrees < 0:
            raise ValueError("pose deltas must be non-negative")
        if not np.isfinite((self.translation_m, self.rotation_degrees)).all():
            raise ValueError("pose deltas must be finite")


class ReintegrationMode(str, enum.Enum):
    NONE = "none"
    LOCAL = "local"
    REGIONAL = "regional"
    GLOBAL = "global"


@dataclasses.dataclass(frozen=True)
class PoseGraphReintegrationConfig:
    translation_threshold_m: float
    rotation_threshold_degrees: float
    regional_window: int
    realtime_window: int
    global_affected_fraction: float = 2.0 / 3.0
    regional_affected_fraction: float = 1.0 / 3.0
    enable_global: bool = False
    enable_regional: bool = True
    enable_local: bool = True

    def __post_init__(self) -> None:
        if self.translation_threshold_m < 0:
            raise ValueError("translation_threshold_m must be non-negative")
        if self.rotation_threshold_degrees < 0:
            raise ValueError("rotation_threshold_degrees must be non-negative")
        if self.regional_window <= 0 or self.realtime_window <= 0:
            raise ValueError("reintegration windows must be positive")
        if self.realtime_window > self.regional_window:
            raise ValueError("realtime_window cannot exceed regional_window")
        for label, value in (
            ("global_affected_fraction", self.global_affected_fraction),
            ("regional_affected_fraction", self.regional_affected_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{label} must be in (0, 1]")


DREAM_RELEASED_REINTEGRATION_CONFIG = PoseGraphReintegrationConfig(
    translation_threshold_m=0.1,
    rotation_threshold_degrees=5.0,
    regional_window=50,
    realtime_window=10,
    enable_global=False,
    enable_regional=True,
    enable_local=True,
)


@dataclasses.dataclass(frozen=True)
class ReintegrationPlan:
    mode: ReintegrationMode
    affected_observation_ids: tuple[str, ...]
    reintegrate_observation_ids: tuple[str, ...]
    pose_deltas: tuple[PoseDelta, ...]
    invalidate_revision_dependents: bool

    def __post_init__(self) -> None:
        if len(self.affected_observation_ids) != len(
            set(self.affected_observation_ids)
        ):
            raise ValueError("affected_observation_ids contains duplicates")
        if len(self.reintegrate_observation_ids) != len(
            set(self.reintegrate_observation_ids)
        ):
            raise ValueError("reintegrate_observation_ids contains duplicates")
        if self.mode is ReintegrationMode.NONE:
            if self.reintegrate_observation_ids:
                raise ValueError("none plan cannot reintegrate observations")
            if self.invalidate_revision_dependents:
                raise ValueError("none plan cannot invalidate revision dependents")
        elif not self.reintegrate_observation_ids:
            raise ValueError("active reintegration plan must select observations")

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "affected_observation_ids": list(self.affected_observation_ids),
            "reintegrate_observation_ids": list(self.reintegrate_observation_ids),
            "pose_deltas": [dataclasses.asdict(delta) for delta in self.pose_deltas],
            "invalidate_revision_dependents": self.invalidate_revision_dependents,
        }


class PoseGraphReintegrationPlanner:
    """Plan selective spatial-memory rebuilds after SLAM pose corrections.

    This implements the released DREAM window and threshold decision rule. It is
    deliberately independent of any voxel representation: a spatial writer can
    consume the selected observation IDs and reproject their retained RGB-D
    evidence using the optimized poses.
    """

    def __init__(self, config: PoseGraphReintegrationConfig) -> None:
        self.config = config

    @staticmethod
    def _delta(
        observation_id: str,
        integrated_pose: np.ndarray,
        optimized_pose: np.ndarray,
    ) -> PoseDelta:
        rotation_delta = optimized_pose[:3, :3] @ integrated_pose[:3, :3].T
        cosine = (float(np.trace(rotation_delta)) - 1.0) * 0.5
        skew = np.asarray(
            (
                rotation_delta[2, 1] - rotation_delta[1, 2],
                rotation_delta[0, 2] - rotation_delta[2, 0],
                rotation_delta[1, 0] - rotation_delta[0, 1],
            )
        )
        sine = 0.5 * float(np.linalg.norm(skew))
        rotation_degrees = float(np.degrees(np.arctan2(sine, cosine)))
        translation_m = float(
            np.linalg.norm(optimized_pose[:3, 3] - integrated_pose[:3, 3])
        )
        return PoseDelta(
            observation_id=observation_id,
            translation_m=translation_m,
            rotation_degrees=rotation_degrees,
        )

    def plan(
        self,
        observations: Sequence[PoseStampedObservation],
        optimized_camera_poses: Mapping[str, np.ndarray],
    ) -> ReintegrationPlan:
        observation_ids = tuple(item.observation_id for item in observations)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observations contain duplicate observation_id values")

        optimized = {
            observation_id: _validated_pose(
                pose, label=f"optimized_camera_pose[{observation_id}]"
            )
            for observation_id, pose in optimized_camera_poses.items()
        }
        shared_ids = tuple(
            observation_id
            for observation_id in observation_ids
            if observation_id in optimized
        )
        observation_by_id = {
            observation.observation_id: observation for observation in observations
        }
        deltas = tuple(
            self._delta(
                observation_id,
                observation_by_id[observation_id].integrated_camera_pose,
                optimized[observation_id],
            )
            for observation_id in shared_ids
        )
        affected = tuple(
            delta.observation_id
            for delta in deltas
            if delta.translation_m > self.config.translation_threshold_m
            or delta.rotation_degrees > self.config.rotation_threshold_degrees
        )
        affected_set = set(affected)
        if not affected:
            return ReintegrationPlan(
                mode=ReintegrationMode.NONE,
                affected_observation_ids=(),
                reintegrate_observation_ids=(),
                pose_deltas=deltas,
                invalidate_revision_dependents=False,
            )

        realtime_ids = observation_ids[-self.config.realtime_window :]
        regional_ids = observation_ids[-self.config.regional_window :]
        realtime_affected = affected_set.intersection(realtime_ids)
        regional_affected = affected_set.intersection(regional_ids)

        mode = ReintegrationMode.NONE
        selected: tuple[str, ...] = ()
        if (
            self.config.enable_global
            and len(observations) > self.config.realtime_window
            and shared_ids
            and len(affected) / len(shared_ids)
            > self.config.global_affected_fraction
        ):
            mode = ReintegrationMode.GLOBAL
            selected = shared_ids
        elif (
            self.config.enable_regional
            and len(observations) > self.config.regional_window
            and len(regional_affected) / self.config.regional_window
            > self.config.regional_affected_fraction
            and realtime_affected
        ):
            mode = ReintegrationMode.REGIONAL
            selected = tuple(
                observation_id
                for observation_id in regional_ids
                if observation_id in optimized
            )
        elif self.config.enable_local and realtime_affected:
            mode = ReintegrationMode.LOCAL
            selected = tuple(
                observation_id
                for observation_id in realtime_ids
                if observation_id in optimized
            )

        return ReintegrationPlan(
            mode=mode,
            affected_observation_ids=affected,
            reintegrate_observation_ids=selected,
            pose_deltas=deltas,
            invalidate_revision_dependents=bool(selected),
        )


@dataclasses.dataclass(frozen=True)
class ObservationArchiveEntry:
    observation_id: str
    is_pose_graph_keyframe: bool

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must be non-empty")


@dataclasses.dataclass(frozen=True)
class ObservationArchiveConfig:
    capacity: int
    maximum_keyframes: int
    recent_feature_window: int

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 <= self.maximum_keyframes <= self.capacity:
            raise ValueError("maximum_keyframes must be in [0, capacity]")
        if not 0 < self.recent_feature_window <= self.capacity:
            raise ValueError("recent_feature_window must be in (0, capacity]")


DREAM_RELEASED_ARCHIVE_CONFIG = ObservationArchiveConfig(
    capacity=500,
    maximum_keyframes=100,
    recent_feature_window=10,
)


@dataclasses.dataclass(frozen=True)
class ObservationArchivePlan:
    evict_observation_ids: tuple[str, ...]
    retain_feature_observation_ids: tuple[str, ...]
    drop_feature_observation_ids: tuple[str, ...]
    retained_observation_count: int
    retained_keyframe_count: int

    def to_json(self) -> dict[str, object]:
        return {
            "evict_observation_ids": list(self.evict_observation_ids),
            "retain_feature_observation_ids": list(
                self.retain_feature_observation_ids
            ),
            "drop_feature_observation_ids": list(self.drop_feature_observation_ids),
            "retained_observation_count": self.retained_observation_count,
            "retained_keyframe_count": self.retained_keyframe_count,
        }


class KeyframeAwareObservationPruner:
    """Bound observation history while preserving pose-graph evidence first."""

    def __init__(self, config: ObservationArchiveConfig) -> None:
        self.config = config

    def plan(
        self, observations: Sequence[ObservationArchiveEntry]
    ) -> ObservationArchivePlan:
        observation_ids = tuple(item.observation_id for item in observations)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observations contain duplicate observation_id values")

        overflow = max(0, len(observations) - self.config.capacity)
        keyframe_count = sum(item.is_pose_graph_keyframe for item in observations)
        evicted: list[str] = []
        for item in observations:
            if overflow == 0:
                break
            can_evict = (
                not item.is_pose_graph_keyframe
                or keyframe_count > self.config.maximum_keyframes
            )
            if not can_evict:
                continue
            evicted.append(item.observation_id)
            overflow -= 1
            if item.is_pose_graph_keyframe:
                keyframe_count -= 1

        if overflow:
            raise RuntimeError(
                "keyframe-aware pruning could not satisfy capacity; "
                "check maximum_keyframes"
            )

        evicted_set = set(evicted)
        retained = tuple(
            item for item in observations if item.observation_id not in evicted_set
        )
        recent_ids = {
            item.observation_id
            for item in retained[-self.config.recent_feature_window :]
        }
        retain_features = tuple(
            item.observation_id
            for item in retained
            if item.is_pose_graph_keyframe or item.observation_id in recent_ids
        )
        retain_feature_set = set(retain_features)
        drop_features = tuple(
            item.observation_id
            for item in retained
            if item.observation_id not in retain_feature_set
        )
        return ObservationArchivePlan(
            evict_observation_ids=tuple(evicted),
            retain_feature_observation_ids=retain_features,
            drop_feature_observation_ids=drop_features,
            retained_observation_count=len(retained),
            retained_keyframe_count=sum(
                item.is_pose_graph_keyframe for item in retained
            ),
        )


SPATIAL_LIFECYCLE_REQUIRED_CAPABILITIES = frozenset(
    {
        "calibrated_rgbd_observation_archive",
        "optimized_pose_graph",
        "spatial_evidence_reintegrator",
    }
)


@dataclasses.dataclass(frozen=True)
class SpatialLifecycleDecision:
    reintegration: ReintegrationPlan
    archive: ObservationArchivePlan

    def to_json(self) -> dict[str, object]:
        return {
            "reintegration": self.reintegration.to_json(),
            "archive": self.archive.to_json(),
        }


class SpatialLifecycleProgram:
    """Compose pose revision and retained-evidence capacity as typed operators."""

    def __init__(
        self,
        *,
        reintegration_planner: PoseGraphReintegrationPlanner,
        observation_pruner: KeyframeAwareObservationPruner,
    ) -> None:
        self.reintegration_planner = reintegration_planner
        self.observation_pruner = observation_pruner

    @staticmethod
    def preflight(capabilities: Collection[str]) -> None:
        available = set(capabilities)
        missing = SPATIAL_LIFECYCLE_REQUIRED_CAPABILITIES - available
        if missing:
            raise ValueError(
                "spatial lifecycle is missing deployment capabilities: "
                f"{sorted(missing)}"
            )

    def plan(
        self,
        *,
        pose_observations: Sequence[PoseStampedObservation],
        optimized_camera_poses: Mapping[str, np.ndarray],
        archive_observations: Sequence[ObservationArchiveEntry],
    ) -> SpatialLifecycleDecision:
        pose_ids = tuple(item.observation_id for item in pose_observations)
        archive_ids = tuple(item.observation_id for item in archive_observations)
        if pose_ids != archive_ids:
            raise ValueError(
                "pose and archive observations must have identical chronological IDs"
            )
        return SpatialLifecycleDecision(
            reintegration=self.reintegration_planner.plan(
                pose_observations, optimized_camera_poses
            ),
            archive=self.observation_pruner.plan(archive_observations),
        )
