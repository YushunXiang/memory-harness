from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

from memory_harness.components import AllRetriever
from memory_harness.components import AllPathsController
from memory_harness.components import AfterFirstStepWrite
from memory_harness.components import AdjacentMergeStore
from memory_harness.components import AlwaysWrite
from memory_harness.components import AnchorStore
from memory_harness.components import BoundaryChunkRetriever
from memory_harness.components import CausalKinematicPeakWrite
from memory_harness.components import ContentRecencyRetriever
from memory_harness.components import CompletedPhaseMeanRetriever
from memory_harness.components import DHEMEventStore
from memory_harness.components import EpisodeLifecycle
from memory_harness.components import FirstWrite
from memory_harness.components import LatestRetriever
from memory_harness.components import Mem0ContextUtilizer
from memory_harness.components import NoMemoryUtilizer
from memory_harness.components import NoveltyWrite
from memory_harness.components import PhaseChangeWrite
from memory_harness.components import PhaseLifecycle
from memory_harness.components import RingStore
from memory_harness.components import SemanticRecentUnionRetriever
from memory_harness.components import TemporalMultiscaleRetriever
from memory_harness.components import TieredChunkMeanStore
from memory_harness.components import TokenEncoder
from memory_harness.components import TokenUtilizer
from memory_harness.components import UniformGlobalRetriever
from memory_harness.components import VerifiedSuccessRingStore
from memory_harness.config import ComponentSpec
from memory_harness.config import ProgramSpec
from memory_harness.runtime import AuditSink
from memory_harness.runtime import MemoryPath
from memory_harness.runtime import MemoryProgram


T = TypeVar("T")
Factory = Callable[..., T]


def _construct(
    spec: ComponentSpec, registry: Mapping[str, Factory[T]], *, role: str
) -> T:
    try:
        factory = registry[spec.type]
    except KeyError as exc:
        raise ValueError(
            f"unknown {role} type {spec.type!r}; expected {sorted(registry)}"
        ) from exc
    try:
        return factory(**dict(spec.options))
    except TypeError as exc:
        raise ValueError(f"invalid options for {role} {spec.type!r}: {exc}") from exc


ENCODERS = {"tokens": TokenEncoder}
WRITERS = {
    "first": FirstWrite,
    "always": AlwaysWrite,
    "after_first_step": AfterFirstStepWrite,
    "phase_change": PhaseChangeWrite,
    "novelty": NoveltyWrite,
    "causal_kinematic_peak": CausalKinematicPeakWrite,
}
STORES = {
    "anchor": AnchorStore,
    "ring": RingStore,
    "adjacent_merge": AdjacentMergeStore,
    "dhem_event": DHEMEventStore,
    "tiered_chunk_mean": TieredChunkMeanStore,
    "verified_success_ring": VerifiedSuccessRingStore,
}
RETRIEVERS = {
    "all": AllRetriever,
    "boundary_chunk": BoundaryChunkRetriever,
    "completed_phase_mean": CompletedPhaseMeanRetriever,
    "latest": LatestRetriever,
    "content_recency": ContentRecencyRetriever,
    "semantic_recent_union": SemanticRecentUnionRetriever,
    "temporal_multiscale": TemporalMultiscaleRetriever,
    "uniform_global": UniformGlobalRetriever,
}
LIFECYCLES = {"episode": EpisodeLifecycle, "phase": PhaseLifecycle}
UTILIZERS = {
    "none": NoMemoryUtilizer,
    "memory_tokens": TokenUtilizer,
    "mem0_context": Mem0ContextUtilizer,
}
CONTROLLERS = {"all": AllPathsController}


def build_program(
    spec: ProgramSpec, *, audit_sink: AuditSink | None = None
) -> MemoryProgram:
    paths = [
        MemoryPath(
            name=path.name,
            encoder=_construct(path.encoder, ENCODERS, role="encoder"),
            writer=_construct(path.writer, WRITERS, role="writer"),
            store=_construct(path.store, STORES, role="store"),
            retriever=_construct(path.retriever, RETRIEVERS, role="retriever"),
            lifecycle=_construct(path.lifecycle, LIFECYCLES, role="lifecycle"),
        )
        for path in spec.paths
    ]
    utilizer = _construct(spec.utilizer, UTILIZERS, role="utilizer")
    controller = _construct(spec.controller, CONTROLLERS, role="controller")
    utilizer.validate_paths(tuple(path.name for path in paths))
    if not paths and spec.utilizer.type not in {"none", "mem0_context"}:
        raise ValueError(
            "a program without memory paths must use none or fixed masked mem0_context"
        )
    if paths and spec.utilizer.type == "none":
        raise ValueError("a program with memory paths cannot use the none utilizer")
    return MemoryProgram(
        name=spec.name,
        deployable=spec.deployable,
        paths=paths,
        controller=controller,
        utilizer=utilizer,
        audit_sink=audit_sink,
    )
