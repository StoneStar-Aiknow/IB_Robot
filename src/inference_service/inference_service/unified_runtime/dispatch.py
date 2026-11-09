"""Dependency-neutral execution policy injected at the runtime boundary."""

from __future__ import annotations

from typing import Protocol

from .contracts import Deadline, ExecutionContext, ModelRequest, ModelResult
from .streaming import StreamHandle


class RuntimeExecutionPolicy(Protocol):
    """Own admission and dispatch while callers retain the handle API.

    A policy calls the handle's authorized local entry points after acquiring
    execution ownership. It must retain uncertain work beyond caller timeout.
    Lifecycle methods must finish their ownership barrier before returning;
    close may be retried but must never reopen admission.

    ``cancel`` must cover queued work before local execution starts as well as
    active dispatch. A policy without this boundary is not supported.
    ``input_ready`` is an optional extension for input buffering. Policies for composite
    executors may also expose ``submit_frame`` without admitting the outer
    orchestration as a second device operation. Its signature is
    ``submit_frame(request: ModelRequest, context: ExecutionContext) -> Future``;
    the Future delivers the executor's frame result or execution failure.
    """

    @property
    def ready(self) -> bool: ...

    def execute(self, request: ModelRequest, context: ExecutionContext) -> ModelResult: ...

    def reset(self, context: ExecutionContext) -> None: ...

    def close(self, context: ExecutionContext) -> None: ...

    def cancel(self, request_id: str, *, deadline: Deadline | None = None) -> None: ...


class StreamingExecutionPolicy(RuntimeExecutionPolicy, Protocol):
    """Required extension when the handle owns a streaming runtime."""

    def open_stream(self, context: ExecutionContext) -> StreamHandle: ...

    def step(
        self, stream_handle: StreamHandle | str, request: ModelRequest, context: ExecutionContext
    ) -> ModelResult: ...

    def reset_stream(self, stream_handle: StreamHandle | str, context: ExecutionContext) -> None: ...

    def close_stream(self, stream_handle: StreamHandle | str, context: ExecutionContext) -> None: ...
