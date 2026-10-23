"""Non-gating presentation for the direct-execute typed Program path.

The legacy :class:`~ibrobot_agent.presentation.PresentationGate` blocks the
execution thread until a terminal client acknowledges that the exact plan was
flushed.  The Program path executes directly instead: the robot may move
before (or without) any UI rendering the plan.

Isolation boundaries:

- :class:`AsyncPresentationPort` never waits for a receipt.  It builds the
  presentation payload and hands it to the ``publish`` callback, which only
  appends one local ledger event and enqueues the ROS message.  Any exception
  raised while publishing is logged and swallowed because display is not an
  execution precondition.
- The ROS transport itself is asynchronous (``_RosEventSink`` in ``node.py``),
  so a slow subscriber or a blocked publisher cannot stall the execution
  thread, terminal persistence, or the next request.
- Durable state (task binding, submission intent, terminal results) stays
  synchronous in :class:`~ibrobot_agent.service.AgentService` and is never
  routed through this port.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping

from ibrobot_agent.contracts import ASYNC_PRESENTATION_MODE, Presentation, PresentationPort, RequestKey
from ibrobot_agent.presentation import presentation_digest

_LOGGER = logging.getLogger(__name__)

__all__ = ["AsyncPresentationPort", "ModeRoutingPresentationPort", "build_async_presentation_payload"]


def build_async_presentation_payload(presentation: Presentation) -> dict[str, object]:
    """Return the client-visible payload for a presentation that needs no receipt."""
    value = presentation.to_dict()
    return {
        "presentation": value,
        "presentation_digest": presentation_digest(value),
        "receipt_token": "",
        "async_display": True,
    }


class AsyncPresentationPort(PresentationPort):
    """Publish a presentation without waiting for display or acknowledgement."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Number of presentations that could not be published."""
        with self._lock:
            return self._dropped

    def present(
        self,
        key: RequestKey,
        presentation: Presentation,
        *,
        publish: Callable[[Mapping[str, object]], None],
        cancel_token: threading.Event,
    ) -> None:
        del cancel_token  # Display never gates or stops execution.
        with self._lock:
            if self._closed:
                self._dropped += 1
                return
        try:
            publish(build_async_presentation_payload(presentation))
        except Exception:
            with self._lock:
                self._dropped += 1
            _LOGGER.warning("async presentation for %s was not published", key.request_id, exc_info=True)

    def close(self) -> None:
        with self._lock:
            self._closed = True


class ModeRoutingPresentationPort(PresentationPort):
    """Route by the trusted adapter-provided execution mode.

    Typed Program presentations use the non-gating port.  Every other mode
    (the legacy Agent Plan ``immediate_after_presentation`` flow) keeps the
    receipt gate so old clients cannot silently trigger motion.
    """

    def __init__(self, *, async_port: PresentationPort, legacy_port: PresentationPort) -> None:
        self._async_port = async_port
        self._legacy_port = legacy_port

    def present(
        self,
        key: RequestKey,
        presentation: Presentation,
        *,
        publish: Callable[[Mapping[str, object]], None],
        cancel_token: threading.Event,
    ) -> None:
        port = self._async_port if presentation.execution_mode == ASYNC_PRESENTATION_MODE else self._legacy_port
        port.present(key, presentation, publish=publish, cancel_token=cancel_token)

    def close(self) -> None:
        close = getattr(self._async_port, "close", None)
        if close is not None:
            close()
