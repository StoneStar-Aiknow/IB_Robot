"""Strict lifecycle state machine for local and future distributed pipelines."""

from __future__ import annotations

from enum import Enum

from inference_service.pipeline.errors import PipelineTransitionError


class PipelineState(str, Enum):
    CREATED = "created"
    LOADING = "loading"
    HANDSHAKING = "handshaking"
    READY = "ready"
    RESETTING = "resetting"
    DEGRADED = "degraded"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


_LEGAL_TRANSITIONS: dict[PipelineState, frozenset[PipelineState]] = {
    PipelineState.CREATED: frozenset({PipelineState.LOADING, PipelineState.CLOSING}),
    PipelineState.LOADING: frozenset(
        {
            PipelineState.HANDSHAKING,
            PipelineState.READY,
            PipelineState.DEGRADED,
            PipelineState.FAILED,
            PipelineState.CLOSING,
        }
    ),
    PipelineState.HANDSHAKING: frozenset(
        {PipelineState.READY, PipelineState.DEGRADED, PipelineState.FAILED, PipelineState.CLOSING}
    ),
    PipelineState.READY: frozenset(
        {PipelineState.RESETTING, PipelineState.DEGRADED, PipelineState.FAILED, PipelineState.CLOSING}
    ),
    PipelineState.RESETTING: frozenset(
        {PipelineState.READY, PipelineState.DEGRADED, PipelineState.FAILED, PipelineState.CLOSING}
    ),
    PipelineState.DEGRADED: frozenset({PipelineState.READY, PipelineState.FAILED, PipelineState.CLOSING}),
    PipelineState.FAILED: frozenset({PipelineState.CLOSING}),
    PipelineState.CLOSING: frozenset({PipelineState.CLOSED}),
    PipelineState.CLOSED: frozenset(),
}


class PipelineStateMachine:
    """Small strict state holder; callers provide any required synchronization."""

    def __init__(self) -> None:
        self._state = PipelineState.CREATED

    @property
    def state(self) -> PipelineState:
        return self._state

    def can_transition(self, target: PipelineState) -> bool:
        return target in _LEGAL_TRANSITIONS[self._state]

    def transition(self, target: PipelineState) -> None:
        if not self.can_transition(target):
            raise PipelineTransitionError(
                f"illegal pipeline transition from {self._state.value} to {target.value}",
                details={"source": self._state.value, "target": target.value},
            )
        self._state = target
